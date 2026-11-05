//! `adam_start` — can the chain *hold* the fine-tuned solution, or is the
//! failure purely a failure to search?
//!
//! The published-budget report answers "does the sampler find Task B?"
//! with 11.7 % for all three sampling arms, against 32.0 % for momentum-SGD
//! and 49.2 % for Adam **from the same `θ_A`**
//! (`docs/PCNL_STATUS.md`, "The 2026-10-20 report is stale").  That
//! comparison leaves two mechanisms inseparable:
//!
//! * **search** — the chain never visits the informative set, so its
//!   retained draws are the prior's;
//! * **retention** — the kernel cannot keep a draw in the informative set
//!   even when it starts there, so the posterior itself is (nearly) flat
//!   there and any sampler would report the prior.
//!
//! This starts the same unadjusted pCNL kernel **at Adam's Task-B solution**
//! and measures what happens to the predictive.  Two arms separate the
//! mechanisms:
//!
//! * **origin prior** (`N(0, σ²I)`, the published target): the OU term pulls
//!   the state to the shell around the origin, so the fine-tuned direction
//!   is re-drawn by the proposal's fresh prior noise at the measured
//!   information ratio `r ≈ 0.05`.  If the predictive decays to chance
//!   within a few `1/β²` timescales, retention is the problem and no
//!   schedule can fix it.
//! * **anchored prior** (`N(θ_Adam, σ²I)`, `SgPcnlState::centered_at`): the
//!   OU term holds the state on the shell *around the solution*, so the only
//!   thing that can destroy the predictive is the likelihood's own noise.
//!
//! **Measured (2026-11-05): retention is the constraint, and the prior's
//! radius is why.**  `θ_Adam` scores 49.22 % (NLL 1.3136) at `‖θ‖ = 1272.6`,
//! i.e. **28.2× the shell** — Adam's solution does not live where `N(0, σ²I)`
//! puts its mass.  Rescaled onto the shell (prediction-neutral) and handed to
//! the kernel:
//!
//! | β₀ | prior | `t = 10` | `t = 100` | kept BMA | BMA NLL | ECE |
//! |---:|---|---:|---:|---:|---:|---:|
//! | 0.3 | origin | 11.72 % | 11.72 % | 11.72 % | 2.1857 | 0.1802 |
//! | 0.3 | anchored | 11.72 % | 11.72 % | 11.72 % | 2.1819 | 0.1675 |
//! | 0.03 | origin | 12.11 % | 17.58 % | 12.89 % | 2.0979 | 0.1791 |
//! | 0.03 | anchored | 12.11 % | 17.58 % | **17.19 %** | **1.9500** | **0.1487** |
//!
//! At the published proposal one step of prior noise is `βσ√P = 13.6` in a
//! 45-norm shell, and the solution — a sharp fine-tune — is gone by step 10
//! whether or not the prior is centred on it.  At `β₀ = 0.03` (noise 1.4) a
//! `β`-independent neighbourhood survives: 17.58 % at `t = 100` with NLL 2.03,
//! and the anchored ensemble's BMA is **1.9500 nats, the first sampling-arm
//! NLL in this study below the uniform predictor's 2.0794** — and 0.15 nats
//! better than the same arm with the origin prior, so the prior's centre does
//! matter once the step is small enough to preserve structure.
//!
//! The binding constraint is therefore the prior's *radial* geometry, not the
//! kernel and not the schedule: `π ∝ N(0, σ²I)·exp(−λNL)` pays `r²/2σ² ≈
//! 6.6 × 10⁶` nats at the solution's norm, so no sampler of `π` can report
//! 49 %.  §3.3's scale invariance says predictions depend only on direction,
//! so the target that would contain the solution is the **direction** on the
//! sphere with a uniform (Haar) prior — not a Gaussian in `θ`.
//!
//! The start is Adam's `θ` rescaled to `σ√P` — a pure rescale, and the
//! cosine variant's forward pass is scale-invariant, so it changes no
//! prediction (the same argument `posterior_radius` uses for its shell
//! start).  Checkpoints at `t ∈ {0, 10, 100, 1 000, burn, steps−1}` print
//! accuracy, NLL and `‖θ‖`, and each arm also reports its BMA over the kept
//! draws.
//!
//! ```text
//! cargo run --release -p pcnl --example adam_start            # 10 000 steps, 2 chains
//! cargo run --release -p pcnl --example adam_start -- 2000    # shorter
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState, norm};
use pcnl::task::{BatchLoader, Dataset, Task, adam, momentum_sgd};
use rand::SeedableRng;

const SEED: u64 = 20261020;
const CHECKPOINTS: [usize; 6] = [0, 10, 100, 1_000, 5_000, 9_999];

fn cfg() -> Config {
    Config {
        seed: SEED,
        ..Config::default()
    }
}

/// `θ` rescaled to the prior's shell.  No prediction changes (the cosine
/// forward pass is scale-invariant), so this isolates radial contraction.
fn to_shell(theta: &Tensor, target: f64) -> Tensor {
    let n = norm(theta).expect("norm");
    theta.affine(target / n, 0.0).expect("rescale")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = cfg();
    let mut it = args.iter();
    // A bare leading number is the step budget; anything else is a flag and
    // must not be consumed here.
    if let Some(n) = args.first().and_then(|a| a.parse::<usize>().ok()) {
        cfg.steps = n;
        cfg.burn = n / 2;
        it.next();
    }
    while let Some(a) = it.next() {
        match (a.as_str(), it.next()) {
            ("--chains", Some(v)) => cfg.chains = v.parse::<usize>().expect("--chains N").max(1),
            ("--kappa", Some(v)) => cfg.kappa = v.parse::<f64>().expect("--kappa K"),
            ("--thin", Some(v)) => cfg.thin = v.parse::<usize>().expect("--thin N").max(1),
            // The proposal scale is the control that separates "the step is
            // larger than the solution's basin" from "the target excludes it".
            ("--beta0", Some(v)) => cfg.beta_0 = v.parse::<f64>().expect("--beta0 B"),
            other => panic!("unknown argument {other:?}"),
        }
    }

    let p_dim = cfg.shapes.flat_size();
    let shell = cfg.sigma * (p_dim as f64).sqrt();
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
    let data_b = Dataset::new(
        cfg.task_b,
        cfg.k,
        cfg.n_train,
        cfg.n_test,
        cfg.seed.wrapping_add(1),
    );
    let theta0 = init_flat(&cfg.shapes, cfg.init_std, cfg.seed.wrapping_add(2));

    let (theta_a, _) = momentum_sgd(
        &data_a,
        &cfg.shapes,
        Variant::Cosine,
        theta0,
        cfg.pretrain_steps,
        cfg.pretrain_lr,
        0.9,
    );
    let (theta_adam, _) = adam(
        &data_b,
        &cfg.shapes,
        Variant::Cosine,
        theta_a.clone(),
        cfg.adam_steps_on_b,
        cfg.adam_lr,
    );

    println!(
        "adam_start — unadjusted pCNL started at Adam's Task-B solution\n\
         P = {p_dim}, shell σ√P = {shell:.2}, target B, uniform NLL ln {} = {ln_k:.4}\n\
         steps {steps}, burn {burn}, thin {thin}, {chains} chain(s), κ {kappa}, β₀ {beta0}, \
         seed {SEED}\n",
        cfg.k,
        ln_k = (cfg.k as f64).ln(),
        steps = cfg.steps,
        burn = cfg.burn,
        thin = cfg.thin,
        chains = cfg.chains,
        kappa = cfg.kappa,
        beta0 = cfg.beta_0
    );

    let acc = |t: &Tensor| {
        data_b
            .accuracy(t, &cfg.shapes, Variant::Cosine)
            .expect("acc")
    };
    let nll = |t: &Tensor| data_b.nll(t, &cfg.shapes, Variant::Cosine).expect("nll");
    println!(
        "reference points\n  θ_A              acc {:>6.2}%  NLL {:>8.4}  ‖θ‖ {:>9.1} ({:.1}× shell)\n\
         \x20 θ_Adam (B)       acc {:>6.2}%  NLL {:>8.4}  ‖θ‖ {:>9.1} ({:.1}× shell)\n",
        acc(&theta_a),
        nll(&theta_a),
        norm(&theta_a).expect("norm"),
        norm(&theta_a).expect("norm") / shell,
        acc(&theta_adam),
        nll(&theta_adam),
        norm(&theta_adam).expect("norm"),
        norm(&theta_adam).expect("norm") / shell,
    );

    let start = to_shell(&theta_adam, shell);
    let sched = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };

    for anchored in [false, true] {
        println!(
            "\n{} prior ({}):\n{:>7}  {:>8}  {:>9}  {:>9}",
            if anchored { "anchored" } else { "origin" },
            if anchored {
                "N(θ_Adam, σ²I) — the OU term holds the solution"
            } else {
                "N(0, σ²I) — the published target"
            },
            "step",
            "acc",
            "NLL",
            "‖θ‖"
        );
        let mut chains: Vec<Vec<Tensor>> = Vec::new();
        for c in 0..cfg.chains {
            let mut state = SgPcnlState::new(
                start.clone(),
                cfg.sigma,
                sched,
                cfg.n_train,
                cfg.batch_size,
                0,
            );
            if anchored {
                state = state.centered_at(start.clone());
            }
            let mut loader = BatchLoader::new(
                cfg.n_train,
                cfg.batch_size,
                cfg.seed.wrapping_add(400 + c as u64),
            );
            let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(200 + c as u64));
            let mut kept = Vec::new();
            for t in 0..cfg.steps {
                if c == 0 && CHECKPOINTS.contains(&t.min(cfg.steps - 1)) {
                    println!(
                        "{t:>7}  {:>7.2}%  {:>9.4}  {:>9.1}",
                        acc(&state.theta),
                        nll(&state.theta),
                        norm(&state.theta).expect("norm")
                    );
                }
                let (x, y) = loader.next(&data_b);
                let (_l, g) = data_b
                    .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
                    .expect("grad");
                state.step_unadjusted(&g, &mut rng).expect("step");
                if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                    kept.push(state.theta.clone());
                }
            }
            chains.push(kept);
        }
        let (bma_acc, bma_nll, ece, k) =
            bma_metrics(&chains, &cfg.shapes, Variant::Cosine, &data_b);
        println!("  BMA over kept draws: {bma_acc:.2}%  NLL {bma_nll:.4}  ECE {ece:.4}  (K={k})");
    }

    println!(
        "\nreading: both arms lose the solution within ten steps at the published\n\
         proposal, so retention fails before search does; at β₀ = 0.03 a neighbourhood\n\
         of it survives (17.6 % / NLL 2.03) and the anchored BMA beats the uniform\n\
         predictor (1.95 < 2.0794) — which is what a prior centred on the solution\n\
         buys.  The solution itself sits at 28× the shell, where the isotropic prior\n\
         pays 6.6e6 nats, so the target rather than the kernel is the binding\n\
         constraint (see the module docs)."
    );
}
