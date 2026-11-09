//! `sphere_chain` — the target §6.5.7 says the Gaussian prior should have
//! been: a uniform prior on the *direction* of `θ`, sampled by the exact
//! spherical pCN of [`pcnl::sphere`].
//!
//! **The question.**  `pcnl/examples/adam_start.rs` measured that the Task-B
//! fine-tune lives at `‖θ‖ = 1272.6`, 28.2× the `σ√P = 45.2` shell, where
//! `N(0, σ²I)` pays `r²/2σ² ≈ 6.6 × 10⁶` nats.  So `π ∝ N(0,σ²I)·exp(−λNL)`
//! assigns the solution vanishing mass and every ambient sampler reports the
//! shell's Haar-typical direction — chance — no matter what it is handed.
//! Since the cosine forward pass is scale-invariant, what the model actually
//! constrains is the *direction*; this runs the kernel on the sphere, where
//! the solution's direction has full mass.
//!
//! **Cells.**  `β₀ ∈ {0.3, 0.03, 0.01}` × start ∈ {Adam's direction, Haar}:
//!
//! * `β₀ = 0.3` is the published proposal.  In 16 672 dimensions no move that
//!   size survives the filter, so the chain either freezes or diffuses; the
//!   acceptance column says which, and a frozen chain's "retention" is the
//!   start's predictive, not a sample.
//! * `β₀ = 0.03` is 1.4 in ambient terms — the scale that held a neighbourhood
//!   in the ambient target (17.6 %, NLL 1.95) without containing the solution.
//! * the Haar start is the search question: the informative directions are a
//!   sub-1/2000 fraction of the sphere (§6.4(ii)), so this should stay at
//!   chance — the difference from the Adam start is what retention is worth.
//!
//! **Measured (2026-11-05).**  600 steps (burn 300, thin 20, 1 chain, λ = 1,
//! so K = 15 kept draws), both proposals from the same two starts:
//!
//! | β₀ | start | accept | BMA acc | BMA NLL |
//! |---:|---|---:|---:|---:|
//! | 0.3 | Adam | **0.0 %** | 49.22 % | 1.3136 |
//! | 0.3 | Haar | 32.3 % | 11.72 % | 2.1879 |
//! | 0.03 | Adam | **0.0 %** | 49.22 % | 1.3136 |
//! | 0.03 | Haar | 100 % | 11.72 % | 2.2069 |
//! | 0.01 | Adam | 0.2 % | 48.05 % | 1.3288 |
//! | 0.01 | Haar | 99.8 % | 11.72 % | 3.7238 |
//! | **0.003** | Adam | 91.8 % | 37.50 % | 1.4527 |
//! | 0.003 | Haar | 99.8 % | 11.72 % | 4.5922 |
//!
//! The metric arm, re-measured 2026-11-06 with the corrected ratio (400 steps,
//! burn 200, thin 20, K = 10, `--precond --seeds 2` → 2 chains per cell):
//!
//! | β₀ | start | accept | BMA acc | BMA NLL | R̂(loss) |
//! |---:|---|---:|---:|---:|---:|
//! | 0.3 | Adam | 0.0 % | 49.22 % | 1.3136 | 1.000 (frozen) |
//! | 0.3 | Haar | 0.2 % | 11.72 % | 3.1410 | ∞ (frozen) |
//! | 0.03 | Adam | 2.8 % | 48.44 % | 1.3200 | 1.361 |
//! | 0.03 | Haar | 7.2 % | 11.72 % | 4.6420 | 3.220 |
//! | 0.01 | Adam | **53.5 %** | 48.44 % | 1.3174 | 2.058 |
//! | 0.01 | Haar | 55.2 % | 11.72 % | 4.6622 | 2.454 |
//! | **0.003** | Adam | **87.8 %** | 49.02 % ± 3.04 | 1.3140 | 2.998 |
//! | 0.003 | Haar | 83.2 % | 11.72 % | 4.6650 | 2.722 |
//!
//! Five readings.
//!
//! 1. **Scale invariance holds at the target level**: `ρ·u_Adam` scores
//!    49.22 % / 1.3136, *identical* to `θ_Adam` at 28× the radius.  The sphere
//!    target therefore contains the solution, which the Gaussian one did not —
//!    the §6.5.7 diagnosis was right about the prior.
//! 2. **The filter's acceptance is governed by `λN‖g‖σβ`** (the typical pCN
//!    exponent), which is why the *isotropic* `adam` cells are dead above
//!    `β₀ = 0.003`: see `examples/curvature_probe.rs`, which locates the
//!    isotropic window there.  Before that probe the grid had no cell that
//!    could move.  The metric arm moves one rung higher — reading 4.
//! 3. **With a curvature metric the chain holds the solution's neighbourhood —
//!    for a while.**  At `β₀ = 0.003` the metric arm accepts 87.8 % and its BMA
//!    scores **49.02 % ± 3.04** (4 chains, 400 steps) against 49.22 % at the
//!    start, while the isotropic arm accepts 92 % and diffuses to 37.50 % over
//!    600 steps.  **Not converged** (`R̂(loss) = 3.00`), and the 10 000-step run
//!    (`--steps 10000 --seeds 2 --chains 2`, 4 chains, K = 250) shows why the
//!    reading must stay narrow: acceptance 86.1 %, BMA **46.39 %** / NLL 1.3412,
//!    across-chain ± 0.37 pts, `R̂ = 2.894`.  The chains *agree* by 10 000 steps
//!    and agree on something **worse** than the frozen start: the 400-step
//!    49.02 % was retention, and at τ = 2.3×10⁻⁵ rad/step the run explores
//!    √10⁴·τ = 2.3×10⁻³ rad, three times the 7.7×10⁻⁴ rad to the mode
//!    (`examples/sphere_mode.rs`, 51.17 % / NLL 1.2748).  So the neighbourhood
//!    is a transient, not a stationary law.  The earlier metric figures in this
//!    file's history (42 % accept, 51.17 %) came from an inexact ratio and are
//!    withdrawn; see `sphere`'s module docs.
//! 4. **The metric widens the accepting window, by about the predicted
//!    amount, and still cannot reach the published β.**  At `β₀ = 0.01` the
//!    isotropic chain accepts 0.2 %; the metric accepts **53.5 %** — the
//!    frontier moves `0.003 → 0.01`, about 3.3× in β, against the 5.6× ceiling
//!    predicted by the measured curvature heterogeneity (31.6× in τ²).  That
//!    is the `22.2 %` the pre-fix ratio claimed, now re-measured and of the
//!    same order, so the *mechanism* was never the bug — only the numbers.
//!    `β₀ = 0.03` is back to 2.8 % and `β₀ = 0.3` to 0.0 %, so bridging
//!    `0.01 → 0.3` still needs 30× more in β and the `√P` gap stands.
//!    Note what the widened window buys: at `β₀ = 0.01` the chain moves
//!    freely and its BMA is 48.44 %, *below* the frozen start's 49.22 % —
//!    acceptance is reach, not information, which is reading 5's point.
//! 5. **Acceptance is still a property of position, not of the kernel**, and for
//!    the metric it cuts both ways: from the Haar direction the same metric
//!    *lowers* acceptance (32.3 % → 0.2 % at `β₀ = 0.3`) because the flat region's
//!    estimated curvature inflates the drift, while in the informative region it
//!    is what makes the chain stay.  A metric transferred across regions is not
//!    a free improvement.
//!
//! The two mechanism findings are complementary: the isotropic Gaussian prior
//! excludes the solution (§6.5.7), and once the prior is repaired to a Haar law
//! on the sphere a *fixed isotropic* step still cannot both move and accept in
//! 16 672 dimensions.  Preconditioning fixes the second only where the
//! curvature is heterogeneous — enough to hold the solution's neighbourhood,
//! not enough to accept the published step.
//!
//! ```text
//! cargo run --release -p pcnl --example sphere_chain                    # both proposals
//! cargo run --release -p pcnl --example sphere_chain -- 600 --precond   # the metric arm
//! cargo run --release -p pcnl --example sphere_chain -- 400             # shorter
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, norm};
use pcnl::sphere::SpherePcn;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};
use rand::SeedableRng;
use rayon::prelude::*;

const SEED: u64 = 20261020;

/// One chain's contribution: `((accept %, BMA acc, BMA NLL), per-draw test
/// losses, (acc at t = 0, 10, 100))`.
type Cell = (
    (f64, f64, f64),
    Vec<f64>,
    (f64, f64, f64),
    // R0.5: where this chain's metric stopped adapting (None without one)
    Option<pcnl::manifest::MetricRecord>,
);
/// `0.003` is the rung `curvature_probe` located: the filter needs
/// `λN‖g‖σβ ≲ 2` at the solution's direction, and only β ≤ 0.003 satisfies it.
/// Without it the grid never contains a cell that can move.
const BETAS: [f64; 4] = [0.3, 0.03, 0.01, 0.003];

fn unit(v: &Tensor) -> Tensor {
    let n = norm(v).expect("norm");
    v.affine(1.0 / n, 0.0).expect("unit")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = Config {
        seed: SEED,
        ..Config::default()
    };
    if let Some(n) = args.first().and_then(|a| a.parse::<usize>().ok()) {
        cfg.steps = n;
        cfg.burn = n / 2;
    }
    // `--precond` swaps in the metric proposal (`step_preconditioned`), whose
    // metric is estimated over the first `metric_warmup` steps and then frozen.
    let precond = args.iter().any(|a| a == "--precond");
    // `--beta V` restricts the grid to one β; `--seeds K` / `--chains N`
    // replicate each cell, which is what makes a split-R̂ possible.
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let only_beta = flag("--beta").and_then(|v| v.parse::<f64>().ok());
    let only_start = flag("--start");
    let seeds: usize = flag("--seeds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    let chains_per_cell: usize = flag("--chains")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);

    let p_dim = cfg.shapes.flat_size();
    let radius = cfg.sigma * (p_dim as f64).sqrt();
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
        theta_a,
        cfg.adam_steps_on_b,
        cfg.adam_lr,
    );

    let acc = |t: &Tensor| {
        data_b
            .accuracy(t, &cfg.shapes, Variant::Cosine)
            .expect("acc")
    };
    let nll = |t: &Tensor| data_b.nll(t, &cfg.shapes, Variant::Cosine).expect("nll");
    let adam_dir = unit(&theta_adam);
    let on_shell = adam_dir.affine(radius, 0.0).expect("shell");
    let haar_dir = unit(&init_flat(&cfg.shapes, 1.0, cfg.seed.wrapping_add(99)));

    println!(
        "sphere_chain — exact spherical pCN (uniform prior on the direction)\n\
         P = {p_dim}, ρ = σ√P = {radius:.2}, target B, uniform NLL ln {} = {ln_k:.4}\n\
         {steps} steps, burn {burn}, thin 20, λ = 1, seed {SEED}, precond {precond}\n",
        cfg.k,
        ln_k = (cfg.k as f64).ln(),
        steps = cfg.steps,
        burn = cfg.burn
    );
    println!(
        "where the solution sits, and what the sphere can see of it\n  \
         θ_Adam (B)            acc {:>6.2}%  NLL {:>8.4}  ‖θ‖ {:>8.1}\n  \
         ρ·u_Adam              acc {:>6.2}%  NLL {:>8.4}  (same direction, shell radius)\n  \
         ρ·u_Haar              acc {:>6.2}%  NLL {:>8.4}\n",
        acc(&theta_adam),
        nll(&theta_adam),
        norm(&theta_adam).expect("norm"),
        acc(&on_shell),
        nll(&on_shell),
        acc(&haar_dir.affine(radius, 0.0).expect("shell")),
        nll(&haar_dir.affine(radius, 0.0).expect("shell")),
    );

    println!(
        "{:>6}  {:>6}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "β₀", "start", "accept", "acc t=0", "acc t=10", "acc t=100", "BMA acc", "BMA NLL", "K"
    );
    for beta_0 in BETAS {
        if let Some(b) = only_beta
            && (beta_0 - b).abs() > 1e-12
        {
            continue;
        }
        for (label, u0) in [("adam", adam_dir.clone()), ("haar", haar_dir.clone())] {
            if let Some(want) = &only_start
                && want != label
            {
                continue;
            }
            let sched = SgPcnlScheduler {
                beta_0,
                gamma: 0.0,
                kappa: 0.5,
            };
            // One row per cell unless `--seeds`/`--chains` ask for replicates,
            // in which case the row is the mean and the split-R̂ is printed
            // under it: a single chain's BMA with 10–15 kept draws is not a
            // posterior, and every such number here has R̂ ≫ 1 behind it.
            let (mut a0, mut a10, mut a100) = (0.0f64, 0.0f64, 0.0f64);
            // Replicates run in parallel: each chain owns its RNG stream, so the
            // draws do not depend on the schedule.
            let jobs: Vec<(usize, usize)> = (0..seeds)
                .flat_map(|s| (0..chains_per_cell).map(move |c| (s, c)))
                .collect();
            let results: Vec<Cell> = jobs
                .par_iter()
                .map(|&(s, c)| {
                    let mut t0 = (0.0f64, 0.0f64, 0.0f64);
                    let mut chain =
                        SpherePcn::new(u0.clone(), radius, cfg.sigma, sched, cfg.n_train, 1.0);
                    if precond {
                        chain = chain.with_preconditioner();
                    }
                    let mut rng = rand::rngs::StdRng::seed_from_u64(
                        cfg.seed
                            ^ beta_0.to_bits()
                            ^ 0x5ced
                            ^ ((s as u64) << 32)
                            ^ (c as u64).wrapping_mul(0x9e37_79b9),
                    );
                    let mut kept: Vec<Tensor> = Vec::new();
                    for t in 0..cfg.steps {
                        let theta = chain.theta().expect("theta");
                        match t {
                            0 => t0.0 = acc(&theta),
                            10 => t0.1 = acc(&theta),
                            100 => t0.2 = acc(&theta),
                            _ => {}
                        }
                        let pot =
                            |th: &Tensor| data_b.loss_and_grad(th, &cfg.shapes, Variant::Cosine);
                        if precond {
                            chain.step_preconditioned(pot, &mut rng)
                        } else {
                            chain.step(pot, &mut rng)
                        }
                        .expect("sphere step");
                        if t >= cfg.burn && (t - cfg.burn).is_multiple_of(20) {
                            kept.push(chain.theta().expect("theta"));
                        }
                    }
                    let (bma_acc, bma_nll, _ece, _k) =
                        bma_metrics(&[kept.clone()], &cfg.shapes, Variant::Cosine, &data_b);
                    (
                        (
                            100.0 * chain.accepted as f64 / cfg.steps as f64,
                            bma_acc,
                            bma_nll,
                        ),
                        // per-draw test loss, for the cross-chain R̂
                        kept.iter().map(nll).collect::<Vec<f64>>(),
                        t0,
                        precond.then(|| chain.metric_record()),
                    )
                })
                .collect();
            let rows: Vec<(f64, f64, f64)> = results.iter().map(|r| r.0).collect();
            let losses: Vec<Vec<f64>> = results.iter().map(|r| r.1.clone()).collect();
            for r in &results {
                if r.2.0 > 0.0 {
                    (a0, a10, a100) = r.2;
                }
            }
            let m = rows.len() as f64;
            let mean = |f: &dyn Fn(&(f64, f64, f64)) -> f64| rows.iter().map(f).sum::<f64>() / m;
            let sd = |f: &dyn Fn(&(f64, f64, f64)) -> f64| {
                let mu = mean(f);
                (rows.iter().map(|r| (f(r) - mu).powi(2)).sum::<f64>() / (m - 1.0).max(1.0)).sqrt()
            };
            let acc_mean = mean(&|r| r.1);
            let nll_mean = mean(&|r| r.2);
            let rhat = pcnl::diag::gelman_rubin(&losses);
            let spread = if m > 1.0 {
                format!("± {:.2} pts / {:+.4} nats", sd(&|r| r.1), sd(&|r| r.2))
            } else {
                String::from("(1 chain)")
            };
            println!(
                "{beta_0:>6}  {label:>6}  {:>7.1}%  {a0:>8.2}%  {a10:>8.2}%  {a100:>8.2}%  \
                 {acc_mean:>8.2}%  {nll_mean:>9.4}  {:>7}",
                mean(&|r| r.0),
                cfg.steps / 40
            );
            if m > 1.0 {
                println!("          {label:>6}  {m:.0} chains  {spread}  R̂(loss) = {rhat:.4}");
            }
            // R0.5: a preconditioned row is only an exact-π sample if the metric
            // stopped moving before the kept window did.
            if let Some(metric) = results.first().and_then(|r| r.3.as_ref()) {
                let where_ = match metric.freeze_step {
                    Some(k) => format!("frozen at step {k} (dim {})", metric.dim),
                    None => format!(
                        "still adapting at step {} — kept draws are warm-up",
                        cfg.burn
                    ),
                };
                println!("          {label:>6}  metric {} {where_}", metric.kind);
            }
        }
    }
    println!(
        "\nreading: acceptance decides whether a row is a sample or a frozen start.\n\
         Where the filter accepts, the sphere target — unlike the Gaussian one — has\n\
         mass at the solution's direction, so retention there is a property of the\n\
         target; a Haar start on the same kernel is the search question, and §6.4(ii)'s\n\
         volume bound says it stays at chance."
    );
}
