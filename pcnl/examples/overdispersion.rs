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
//! ```text
//! cargo run --release -p pcnl --example overdispersion
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState};
use pcnl::task::{BatchLoader, Dataset, Task, adam};
use rand::SeedableRng;

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
    println!(
        "cosine (nGPT-lite) chain: per-draw vs ensemble predictive\n\
         reference configuration (P = 16 672), 1 000 steps, burn 500, thin 5,\n\
         2 chains, Adam pre-train on A, target B, uniform NLL ln 8 = 2.0794\n"
    );
    println!(
        "{:>9}  {:>12}  {:>12}  {:>11}  {:>11}  {:>11}  {:>7}",
        "seed", "draw acc", "BMA acc", "draw NLL", "BMA NLL", "d NLL", "K"
    );

    let mut gaps = Vec::new();
    for seed in SEEDS {
        let cfg = cfg_for(seed);
        let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
        let data_b = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, seed);
        let theta_a = adam(
            &data_a,
            &cfg.shapes,
            Variant::Cosine,
            init_flat(&cfg.shapes, 0.1, seed),
            cfg.pretrain_steps,
            cfg.adam_lr,
        )
        .0;

        let sched = SgPcnlScheduler {
            beta_0: cfg.beta_0,
            gamma: cfg.gamma,
            kappa: cfg.kappa,
        };
        let mut chains: Vec<Vec<Tensor>> = Vec::new();
        let (mut draw_nll, mut draw_acc, mut n_draw) = (0.0f64, 0.0f64, 0usize);
        for c in 0..cfg.chains {
            let mut state = SgPcnlState::new(
                theta_a.clone(),
                cfg.sigma,
                sched,
                cfg.n_train,
                cfg.batch_size,
                0,
            );
            let mut loader = BatchLoader::new(
                cfg.n_train,
                cfg.batch_size,
                seed ^ ((c as u64) << 8) ^ 0x5eed,
            );
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ ((c as u64) << 8));
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
