//! `overdispersion` — is the cosine chain's ensemble doing work its draws can't?
//!
//! §6.5.3 found that the capped *standard* stack is over-dispersed: its kept
//! draws are individually overconfident (per-draw NLL 4.11–5.72) and mutually
//! disagree, so their probability average is a calibrated predictive
//! (BMA NLL 1.84–1.92) — the ensemble's calibration comes from its bias.
//!
//! The natural question for the architecture this paper actually proposes is
//! whether the *cosine* chain has the same property, and the same two numbers
//! answer it without any new machinery:
//!
//! * **per-draw** — mean over kept draws of each draw's own NLL and accuracy;
//! * **ensemble** — the BMA over the same draws (`experiment::bma_metrics`).
//!
//! If `ensemble NLL ≪ per-draw NLL`, the draws disagree and averaging pays
//! (over-dispersion).  If the two agree, the draws already sit in one region
//! and averaging buys nothing — which is what §6.4 means by "the ensemble
//! found the same mode".  The gap is also the honest size of the "BMA beats a
//! single sample" claim: a posterior whose draws agree cannot be sampled
//! usefully by taking one more draw.
//!
//! **Measured (2026-11-05).** At the published 10 000-step horizon the gap
//! does not grow — it halves (0.027 → 0.015) while per-draw NLL falls by
//! 0.02 nats, so the draws agree *more* with more budget, and the ensemble's
//! advantage over a single draw is 0.015 nats against the uniform
//! predictor's 2.0794.  The same run is how the archived headline
//! (54.7 % / 1.0686, `docs/PCNL_STATUS.md`) was found to be stale: this
//! binary prints 11.7 % / 2.1863 for the documented invocation.
//!
//! A bare number sets the step budget, `--seeds N` / `--chains N` trim the
//! matrix, and `--kappa K` / `--thin N` / `--pretrain sgd` / `--start
//! theta-a|jitter` rebuild the headline report's exact configuration.
//!
//! ```text
//! cargo run --release -p pcnl --example overdispersion
//! cargo run --release -p pcnl --example overdispersion -- 10000 --chains 4 \
//!     --thin 10 --kappa 0.55 --pretrain sgd --seeds 1
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState};
use pcnl::task::{BatchLoader, Dataset, Task, adam, momentum_sgd};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};

const SEEDS: [u64; 3] = [20261020, 20261021, 20261022];

fn cfg_for(seed: u64) -> Config {
    Config {
        seed,
        shapes: Shapes {
            vocab: 9,
            d: 32,
            ff: 64,
            layers: 2,
        },
        n_train: 512,
        n_test: 256,
        chains: 2,
        steps: 1000,
        burn: 500,
        thin: 5,
        batch_size: 64,
        sigma: 0.35,
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.5,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut n_seeds = SEEDS.len();
    let mut steps = 1_000usize;
    let mut chains = 2usize;
    let mut thin = 5usize;
    let mut kappa = 0.5;
    let mut pretrain_adam = true;
    // `experiment.rs` starts chain `c` at `θ_A + σξ_c` with its own streams;
    // reproducing the headline report therefore needs the jitter *and* its
    // seeding, not only the kernel's parameters.
    let mut jitter = false;
    let mut it = args.iter();
    if let Some(n) = args.first().and_then(|a| a.parse::<usize>().ok()) {
        steps = n;
        it.next();
    }
    while let Some(a) = it.next() {
        let raw = it.next();
        match (a.as_str(), raw) {
            ("--seeds", Some(v)) => {
                n_seeds = v.parse::<usize>().expect("--seeds N").clamp(1, SEEDS.len())
            }
            ("--chains", Some(v)) => chains = v.parse::<usize>().expect("--chains N").max(1),
            ("--thin", Some(v)) => thin = v.parse::<usize>().expect("--thin N").max(1),
            ("--kappa", Some(v)) => kappa = v.parse::<f64>().expect("--kappa K"),
            ("--pretrain", Some(v)) if v == "sgd" => pretrain_adam = false,
            ("--pretrain", Some(v)) if v == "adam" => pretrain_adam = true,
            ("--start", Some(v)) if v == "jitter" => jitter = true,
            ("--start", Some(v)) if v == "theta-a" => jitter = false,
            _ => panic!(
                "usage: overdispersion [STEPS] [--seeds N] [--chains N] [--thin N] \
                 [--kappa K] [--pretrain sgd|adam] [--start jitter|theta-a]"
            ),
        }
    }
    println!(
        "cosine (nGPT-lite) chain: per-draw vs ensemble predictive\n\
         reference configuration (P = 16 672), {steps} steps, burn {}, thin {thin},\n\
         {chains} chains, kappa {kappa}, {} pre-train on A,\n\
         start θ_A{}, target B, uniform NLL ln 8 = 2.0794\n",
        steps / 2,
        if pretrain_adam {
            "Adam"
        } else {
            "momentum-SGD"
        },
        if jitter { " + σξ" } else { "" }
    );
    println!(
        "{:>9}  {:>12}  {:>12}  {:>11}  {:>11}  {:>11}  {:>7}",
        "seed", "draw acc", "BMA acc", "draw NLL", "BMA NLL", "d NLL", "K"
    );

    let mut gaps = Vec::new();
    for seed in SEEDS.into_iter().take(n_seeds) {
        let cfg = Config {
            steps,
            burn: steps / 2,
            chains,
            thin,
            kappa,
            ..cfg_for(seed)
        };
        // Exact construction of `experiment::run_experiment`: data A on
        // `seed`, data B on `seed+1`, the initial weights on `seed+2`.
        // Anything else samples a different problem, and the direction's
        // basin is the object under test here.
        let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
        let data_b = Dataset::new(
            cfg.task_b,
            cfg.k,
            cfg.n_train,
            cfg.n_test,
            seed.wrapping_add(1),
        );
        let theta0 = |shapes: &Shapes, s: u64| init_flat(shapes, cfg.init_std, s);
        let theta_a = if pretrain_adam {
            adam(
                &data_a,
                &cfg.shapes,
                Variant::Cosine,
                theta0(&cfg.shapes, seed.wrapping_add(2)),
                cfg.pretrain_steps,
                cfg.adam_lr,
            )
            .0
        } else {
            momentum_sgd(
                &data_a,
                &cfg.shapes,
                Variant::Cosine,
                theta0(&cfg.shapes, seed.wrapping_add(2)),
                cfg.pretrain_steps,
                cfg.pretrain_lr,
                0.9,
            )
            .0
        };

        let sched = SgPcnlScheduler {
            beta_0: cfg.beta_0,
            gamma: cfg.gamma,
            kappa: cfg.kappa,
        };
        let mut chains: Vec<Vec<Tensor>> = Vec::new();
        let (mut draw_nll, mut draw_acc, mut n_draw) = (0.0f64, 0.0f64, 0usize);
        for c in 0..cfg.chains {
            // experiment.rs seeds the noise stream at `seed+200+c`, the batch
            // stream at `seed+400+c`, and the start at `θ_A + σξ`.
            let (start, loader_seed, rng_seed) = if jitter {
                let mut nrng = StdRng::seed_from_u64(seed.wrapping_add(200 + c as u64));
                let p_dim = cfg.shapes.flat_size();
                let z: Vec<f32> = (0..p_dim)
                    .map(|_| {
                        let z: f32 = nrng.sample(rand_distr::StandardNormal);
                        z * cfg.sigma as f32
                    })
                    .collect();
                let jit = Tensor::from_vec(z, vec![p_dim], theta_a.device()).expect("jit");
                (
                    (theta_a.clone() + jit).expect("init"),
                    seed.wrapping_add(400 + c as u64),
                    seed.wrapping_add(200 + c as u64),
                )
            } else {
                (
                    theta_a.clone(),
                    seed ^ ((c as u64) << 8) ^ 0x5eed,
                    seed ^ ((c as u64) << 8),
                )
            };
            let mut state =
                SgPcnlState::new(start, cfg.sigma, sched, cfg.n_train, cfg.batch_size, 0);
            let mut loader = BatchLoader::new(cfg.n_train, cfg.batch_size, loader_seed);
            let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
            let mut kept = Vec::new();
            for t in 0..cfg.steps {
                let (x, y) = loader.next(&data_b);
                let (_l, g) = data_b
                    .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
                    .expect("grad");
                state.step_unadjusted(&g, &mut rng).expect("step");
                if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                    // each draw's *own* score, before any averaging
                    let nll = data_b
                        .nll(&state.theta, &cfg.shapes, Variant::Cosine)
                        .expect("nll");
                    let acc = data_b
                        .accuracy(&state.theta, &cfg.shapes, Variant::Cosine)
                        .expect("acc");
                    if nll.is_finite() {
                        draw_nll += nll;
                        draw_acc += acc;
                        n_draw += 1;
                    }
                    kept.push(state.theta.clone());
                }
            }
            chains.push(kept);
        }

        let (bma_acc, bma_nll, _ece, k) =
            bma_metrics(&chains, &cfg.shapes, Variant::Cosine, &data_b);
        let n = n_draw as f64;
        let gap = draw_nll / n - bma_nll;
        gaps.push(gap);
        println!(
            "{seed:>9}  {:>11.2}%  {bma_acc:>11.2}%  {:>11.4}  {bma_nll:>11.4}  {gap:>11.4}  {k:>7}",
            draw_acc / n,
            draw_nll / n
        );
    }

    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    println!(
        "\nmean (per-draw NLL - BMA NLL) = {mean_gap:+.4} nats\n\
         reading: a large positive gap means the draws disagree and averaging pays —\n\
         over-dispersion, the standard stack's signature (4.11-5.72 -> 1.84-1.92).\n\
         The cosine chain's gap is the size of its own disagreement; because §6.4\n\
         finds the ensemble sitting in one mode, the gap is expected to be small,\n\
         and a small gap means one more draw cannot improve on another."
    );
}
