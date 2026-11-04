//! `mh_standard_target` — is the standard stack's overconfident predictive
//! the **target**, or the **unadjusted bias**?
//!
//! §6.5.2 measured that the standard-softmax stack, run under the
//! unadjusted kernel with its drift capped at the normalized geometry's
//! own gradient magnitude, survives but drifts into a confident-and-wrong
//! regime: kept-draw NLL 4.11–5.72 nats against 2.08 for the uniform
//! predictor.  Two explanations fit that number, and they are different
//! claims:
//!
//! 1. **The target is overconfident.**  Under the prior `N(0, σ²I)` and the
//!    cross-entropy potential, the standard stack's posterior genuinely
//!    prefers large weights (bigger logits, more confident predictions), so
//!    even an exact sampler returns an overconfident predictive.
//! 2. **The unadjusted drift is the bias.**  The capped drift is not a
//!    gradient field, so the chain has no Gibbs target; its outward march
//!    is an artefact, and the exact target sits near `θ_A`.
//!
//! The Metropolis filter separates them.  `task::mh_pcnl_chain` proposes
//! with the *same* capped step and accepts with the exact full-batch
//! likelihood ratio, so the cap changes only the proposal and the chain is
//! exact for the target.  If the filtered chain's kept predictive is also
//! overconfident, (1) holds; if it stays near `θ_A`'s calibrated numbers,
//! (2) does.
//!
//! Three arms per seed, reference configuration, two chains:
//! * `unadjusted, capped` — the §6.5.2 row, reproduced here;
//! * `MH, capped` — exact target, same proposal scale;
//! * `MH, uncapped` — the filter at the kernel's own step size, where the
//!   proposal is so large that acceptance should collapse (the reason the
//!   cap is needed for the *proposal*, not for the target).
//!
//! ```text
//! cargo run --release -p pcnl --example mh_standard_target
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState, norm};
use pcnl::task::{BatchLoader, Dataset, MhOptions, Task, adam, mh_pcnl_chain};
use rand::SeedableRng;

const SEEDS: [u64; 3] = [20261020, 20261021, 20261022];

fn cfg_for(seed: u64) -> Config {
    Config {
        seed,
        k: 8,
        task_b: Task::B,
        shapes: Shapes {
            vocab: 9,
            d: 32,
            ff: 64,
            layers: 2,
        },
        n_train: 512,
        n_test: 256,
        pretrain_steps: 1000,
        pretrain_adam: true,
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

/// The normalized geometry's own gradient scale: the largest scaled
/// mini-batch gradient the cosine chain produces over its first 200 steps.
fn cosine_gradient_scale(cfg: &Config, data: &Dataset, theta_a: &Tensor) -> f64 {
    let sched = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    let mut state = SgPcnlState::new(
        theta_a.clone(),
        cfg.sigma,
        sched,
        cfg.n_train,
        cfg.batch_size,
        0,
    );
    let mut loader = BatchLoader::new(cfg.n_train, cfg.batch_size, cfg.seed ^ 0xabc);
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed ^ 0xdef);
    let mut max = 0.0f64;
    for _ in 0..200 {
        let (x, y) = loader.next(data);
        let (_l, g) = data
            .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
            .expect("grad");
        if let Ok(n) = norm(&g) {
            max = max.max(n * cfg.n_train as f64 / cfg.batch_size as f64);
        }
        state.step_unadjusted(&g, &mut rng).expect("step");
    }
    max
}

/// The unadjusted chain with a drift cap, thinned exactly like the MH arm.
fn unadjusted_capped(
    cfg: &Config,
    data: &Dataset,
    theta_a: &Tensor,
    cap: f64,
) -> (Vec<Tensor>, f64, f64) {
    let sched = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    let mut kept = Vec::new();
    let (mut rad_max, mut rad_final) = (0.0f64, 0.0f64);
    for c in 0..cfg.chains {
        let mut state = SgPcnlState::new(
            theta_a.clone(),
            cfg.sigma,
            sched,
            cfg.n_train,
            cfg.batch_size,
            0,
        );
        state.grad_norm_cap = Some(cap);
        let mut loader = BatchLoader::new(
            cfg.n_train,
            cfg.batch_size,
            cfg.seed ^ ((c as u64) << 8) ^ 0x5eed,
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed ^ ((c as u64) << 8));
        for t in 0..cfg.steps {
            let (x, y) = loader.next(data);
            let (_l, g) = data
                .batch_grad(&state.theta, &cfg.shapes, Variant::Standard, &x, &y)
                .expect("grad");
            state.step_unadjusted(&g, &mut rng).expect("step");
            let r = norm(&state.theta).unwrap_or(f64::NAN);
            if r.is_finite() {
                rad_max = rad_max.max(r);
                rad_final = r;
            }
            if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                kept.push(state.theta.clone());
            }
        }
    }
    (kept, rad_max, rad_final)
}

fn main() {
    println!(
        "MH-filtered pCNL on the standard-softmax stack — is the overconfident\n\
         predictive the target or the unadjusted bias?\n\
         reference configuration (P = 16 672), 1 000 steps, burn 500, thin 5, 2 chains,\n\
         Adam pre-train on A, target B, chance 12.5 %, uniform NLL ln 8 = 2.0794\n"
    );
    println!(
        "{:>8}  {:<20}  {:>8}  {:>9}  {:>9}  {:>8}  {:>8}  {:>7}  {:>5}",
        "seed", "arm", "accept %", "radial mx", "radial fn", "BMA acc", "BMA nll", "BMA ece", "K"
    );

    for seed in SEEDS {
        let cfg = cfg_for(seed);
        let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
        let data_b = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, seed);
        let theta_a = adam(
            &data_a,
            &cfg.shapes,
            Variant::Standard,
            init_flat(&cfg.shapes, 0.1, seed),
            cfg.pretrain_steps,
            cfg.adam_lr,
        )
        .0;
        let theta_a_cos = adam(
            &data_a,
            &cfg.shapes,
            Variant::Cosine,
            init_flat(&cfg.shapes, 0.1, seed),
            cfg.pretrain_steps,
            cfg.adam_lr,
        )
        .0;
        let cap = cosine_gradient_scale(&cfg, &data_b, &theta_a_cos);

        let (acc_a, nll_a) = (
            data_b
                .accuracy(&theta_a, &cfg.shapes, Variant::Standard)
                .expect("acc"),
            data_b
                .nll(&theta_a, &cfg.shapes, Variant::Standard)
                .expect("nll"),
        );
        println!(
            "{seed:>8}  {:<20}  {:>8}  {:>9}  {:>9}  {acc_a:>7.1}%  {nll_a:>8.4}  {:>7}  {:>5}",
            "theta_A (reference)", "-", "-", "-", "-", "-"
        );

        // 1. the §6.5.2 row: unadjusted, drift capped at the geometry's scale
        let (kept, rad_max, rad_final) = unadjusted_capped(&cfg, &data_b, &theta_a, cap);
        let (acc, nll, ece, k) = bma_metrics(
            std::slice::from_ref(&kept),
            &cfg.shapes,
            Variant::Standard,
            &data_b,
        );
        println!(
            "{seed:>8}  {:<20}  {:>8}  {rad_max:>9.1}  {rad_final:>9.1}  {acc:>7.1}%  {nll:>8.4}  {ece:>7.3}  {k:>5}",
            "unadjusted, capped", "-"
        );

        // 2-4. the exact target, at the published proposal scale and at
        // smaller ones — the filter is only informative once it accepts.
        // The cap is held fixed (it is the drift budget of §6.5.2); what
        // changes is the *proposal* scale beta_0.
        let mh_arm = |label: &str, beta0: f64| {
            // a fresh config with only the proposal scale changed
            let mut mcfg = cfg_for(cfg.seed);
            mcfg.beta_0 = beta0;
            let mh: Vec<_> = (0..cfg.chains)
                .map(|c| {
                    mh_pcnl_chain(
                        &mcfg,
                        Variant::Standard,
                        &data_b,
                        theta_a.clone(),
                        c,
                        None,
                        MhOptions {
                            grad_norm_cap: Some(cap),
                            potential_scale: 1.0,
                        },
                    )
                })
                .collect();
            let accepted: usize = mh.iter().map(|o| o.accepted).sum();
            let steps: usize = mh.iter().map(|o| o.steps).sum();
            let kept_mh: Vec<Vec<Tensor>> = mh.iter().map(|o| o.kept.clone()).collect();
            let (acc, nll, ece, k) = bma_metrics(&kept_mh, &cfg.shapes, Variant::Standard, &data_b);
            println!(
                "{seed:>8}  {label:<20}  {:>7.1}%  {:>9.1}  {:>9.1}  {acc:>7.1}%  {nll:>8.4}  {ece:>7.3}  {k:>5}",
                100.0 * accepted as f64 / steps.max(1) as f64,
                mh.iter().map(|o| o.radial_max).fold(0.0, f64::max),
                mh.last().map(|o| o.radial_final).unwrap_or(f64::NAN),
            );
        };
        mh_arm("MH, capped, b0", cfg.beta_0);
        mh_arm("MH, capped, b0/10", 0.1 * cfg.beta_0);
        mh_arm("MH, capped, b0/30", cfg.beta_0 / 30.0);

        // 5. the filter at the kernel's own step size (no cap): the
        // proposal is so large that acceptance collapses
        let mh_raw: Vec<_> = (0..cfg.chains)
            .map(|c| {
                mh_pcnl_chain(
                    &cfg,
                    Variant::Standard,
                    &data_b,
                    theta_a.clone(),
                    c,
                    None,
                    MhOptions::default(),
                )
            })
            .collect();
        let accepted: usize = mh_raw.iter().map(|o| o.accepted).sum();
        let steps: usize = mh_raw.iter().map(|o| o.steps).sum();
        let kept_raw: Vec<Vec<Tensor>> = mh_raw.iter().map(|o| o.kept.clone()).collect();
        let (acc, nll, ece, k) = bma_metrics(&kept_raw, &cfg.shapes, Variant::Standard, &data_b);
        println!(
            "{seed:>8}  {:<20}  {:>7.1}%  {:>9.1}  {:>9.1}  {acc:>7.1}%  {nll:>8.4}  {ece:>7.3}  {k:>5}",
            "MH, uncapped",
            100.0 * accepted as f64 / steps.max(1) as f64,
            mh_raw.iter().map(|o| o.radial_max).fold(0.0, f64::max),
            mh_raw.last().map(|o| o.radial_final).unwrap_or(f64::NAN),
        );
        println!("          (cap C = {cap:.3e}; theta_A acc {acc_a:.1} %, NLL {nll_a:.4})");
    }

    println!(
        "\nreading: if the capped MH arm's BMA NLL is also far above ln 8, the target itself\n\
         is overconfident and §6.5.2's row is a property of the posterior; if it sits near\n\
         theta_A's NLL, the outward march was the unadjusted drift's bias.  The uncapped MH\n\
         row is the control for the proposal: near-zero acceptance means the filter is\n\
         rejecting the kernel's own step, not the target."
    );
}
