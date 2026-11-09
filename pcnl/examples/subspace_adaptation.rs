//! R1 — restricted affine-subspace adaptation, sampled exactly.
//!
//! The model is `θ(z) = θ_ref + σ·B z`, `z ~ N(0, I_r)`, and the chain samples
//! `π_r(dz) ∝ exp(−λ·L_B(θ(z))) N(0, I_r)(dz)` by MH-pCN in `z`
//! (`pcnl::target::mh_pcn_z`).  This is a *different, smaller* posterior than the
//! ambient one — `r` directions of a `P`-dimensional model — so the numbers
//! below answer "how good is the best rank-`r` affine restriction?" and not
//! "what is the posterior over `θ`?".
//!
//! Two bases at every rank, same target, same anchor, same budget:
//!
//! * `random` — a seeded random orthonormal `B`: the control.  An informed basis
//!   that cannot beat it is not informed, it is just a basis.
//! * `informed` — the top-`r` principal directions of a short Task-B fine-tuning
//!   path (`pcnl::subspace::pca_basis`).  That path is Task-B data, so this arm is
//!   *empirical Bayes with data reuse*: the basis saw the training split the
//!   potential sums over.  Its held-out numbers are therefore optimistic, and the
//!   `random` control is what makes the comparison readable.
//!
//! `θ_ref` is pretrained on Task A (the plan's prior arm), never on B.  `λ` is
//! the target temperature; `β` is the proposal scale.  Because `β` is constant
//! and the target is fixed, these chains are stationary — their ESS/`R̂` are
//! stationary-chain diagnostics, unlike the ambient decaying-step traces (R0.5).
//!
//! ```text
//! cargo run --release -p pcnl --example subspace_adaptation
//! cargo run --release -p pcnl --example subspace_adaptation -- --ranks 1,2,4 --steps 4000
//! ```

use candle_core::Tensor;
use pcnl::diag::{ess_bulk, gelman_rubin};
use pcnl::experiment::bma_metrics;
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::subspace::{AffineSubspace, Basis, pca_basis};
use pcnl::target::{SubspaceTarget, ZPotential, mh_pcn_z};
use pcnl::task::{Dataset, Task, momentum_sgd};
use rand::SeedableRng;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Random,
    Informed,
}

impl Kind {
    fn label(&self) -> &'static str {
        match self {
            Kind::Random => "random",
            Kind::Informed => "informed",
        }
    }
}

struct Cfg {
    ranks: Vec<usize>,
    cells: Vec<Kind>,
    chains: usize,
    steps: usize,
    burn: usize,
    thin: usize,
    beta: f64,
    lambda: f64,
    sigma: f64,
    seed: u64,
    /// Fine-tuning steps used to build the informed basis.
    path_steps: usize,
    pretrain_steps: usize,
    n_train: usize,
    n_test: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            ranks: vec![1, 2, 4, 8],
            cells: vec![Kind::Random, Kind::Informed],
            chains: 4,
            steps: 2_000,
            burn: 500,
            thin: 5,
            // pCN's proposal scale, not a temperature: 0.5 explores a
            // prior-width target in a few hundred steps, and the reported
            // acceptance says whether that was too big for this rank.
            beta: 0.5,
            lambda: 1.0,
            sigma: 0.35,
            seed: 20261020,
            path_steps: 32,
            // The plan's Task-A-prior arm: 1500 momentum steps at lr 0.6 on
            // 512 examples gives the ~20 %-on-A / ~12 %-on-B anchor the
            // ambient reports quote.  A shorter pretrain overfits A and makes
            // every subspace cell look worse for a reason unrelated to the
            // restriction.
            pretrain_steps: 1_500,
            n_train: 512,
            n_test: 256,
        }
    }
}

impl Cfg {
    fn from_args() -> Self {
        let mut cfg = Self::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let flag = |name: &str| -> Option<String> {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        if let Some(v) = flag("--ranks") {
            cfg.ranks = v
                .split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&r| r > 0)
                .collect();
        }
        if let Some(v) = flag("--basis") {
            cfg.cells = v
                .split(',')
                .filter_map(|x| match x.trim() {
                    "random" => Some(Kind::Random),
                    "informed" => Some(Kind::Informed),
                    _ => None,
                })
                .collect();
        }
        if let Some(v) = flag("--chains").and_then(|v| v.parse().ok()) {
            cfg.chains = v;
        }
        if let Some(v) = flag("--steps").and_then(|v| v.parse().ok()) {
            cfg.steps = v;
        }
        if let Some(v) = flag("--burn").and_then(|v| v.parse().ok()) {
            cfg.burn = v;
        }
        if let Some(v) = flag("--thin").and_then(|v| v.parse().ok()) {
            cfg.thin = v;
        }
        if let Some(v) = flag("--seed").and_then(|v| v.parse().ok()) {
            cfg.seed = v;
        }
        if let Some(v) = flag("--beta").and_then(|v| v.parse().ok()) {
            cfg.beta = v;
        }
        if let Some(v) = flag("--lambda").and_then(|v| v.parse().ok()) {
            cfg.lambda = v;
        }
        if let Some(v) = flag("--sigma").and_then(|v| v.parse().ok()) {
            cfg.sigma = v;
        }
        if let Some(v) = flag("--path-steps").and_then(|v| v.parse().ok()) {
            cfg.path_steps = v;
        }
        if let Some(v) = flag("--n-train").and_then(|v| v.parse().ok()) {
            cfg.n_train = v;
        }
        cfg.burn = cfg.burn.min(cfg.steps.saturating_sub(1));
        cfg.thin = cfg.thin.max(1);
        cfg.chains = cfg.chains.max(1);
        cfg
    }
}

fn flat(theta: &Tensor) -> Vec<f32> {
    theta
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to vec")
}

/// A short full-batch momentum-SGD path on Task B, collected as flat `P`-vectors.
/// This is the "training-trajectory PCA" the plan names as a candidate informed
/// basis; `momentum_sgd` itself returns only its endpoint, so the path is
/// collected here rather than reconstructed.
fn fit_path(
    data: &Dataset,
    shapes: &Shapes,
    theta0: &Tensor,
    steps: usize,
    lr: f64,
) -> Vec<Vec<f32>> {
    let mut theta = theta0.clone();
    let mut vel =
        Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("zeros");
    let mut path = Vec::with_capacity(steps);
    for _ in 0..steps {
        let Ok((loss, grad)) = data.loss_and_grad(&theta, shapes, Variant::Cosine) else {
            break;
        };
        if !loss.is_finite() {
            break;
        }
        path.push(flat(&theta));
        vel = (vel.affine(0.9, 0.0).expect("v") - grad.affine(lr, 0.0).expect("lr g")).expect("v'");
        theta = (&theta + &vel).expect("theta'");
    }
    path
}

struct CellLine {
    rank: usize,
    kind: Kind,
    accept: f64,
    /// ESS/Rhat of the *potential* — the state variable, not a functional of it.
    /// A held-out-loss ESS understates mixing whenever the loss is nearly flat
    /// across the posterior (it then has no variance to autocorrelate), so the
    /// mixing column is `Phi` and the predictive MCSE stays on the loss.
    ess_phi: f64,
    rhat_phi: f64,
    ess_loss: f64,
    rhat_loss: f64,
    acc: f64,
    nll: f64,
    ece: f64,
    models: usize,
    nll_mcse: f64,
    evals: u64,
    seconds: f64,
}

fn run_cell(
    cfg: &Cfg,
    shapes: &Shapes,
    data_b: &Dataset,
    ref_flat: &[f32],
    rank: usize,
    kind: Kind,
    basis: Basis,
) -> CellLine {
    let subspace =
        AffineSubspace::isotropic(ref_flat.to_vec(), basis, cfg.sigma).expect("validated subspace");
    let target = SubspaceTarget::new(
        subspace,
        cfg.lambda,
        data_b.clone(),
        *shapes,
        Variant::Cosine,
    );
    let starts = Instant::now();
    let mut per_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut phi_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut kept: Vec<Vec<Tensor>> = Vec::with_capacity(cfg.chains);
    let (mut accepted, mut total) = (0usize, 0usize);
    for c in 0..cfg.chains {
        // Each chain owns its stream: `c` is in the seed, so the chains are
        // genuinely independent and their Rhat means something.
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            cfg.seed
                ^ ((rank as u64) << 24)
                ^ if kind == Kind::Informed { 0x1_000 } else { 0 }
                ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut z = vec![0.0f32; rank];
        let mut phi = target.phi(&z);
        let mut losses = Vec::new();
        let mut phis = Vec::new();
        let mut draws = Vec::new();
        for t in 0..cfg.steps {
            let st = mh_pcn_z(&target, &mut z, phi, cfg.beta, &mut rng);
            phi = st.phi;
            if t >= cfg.burn {
                total += 1;
                accepted += usize::from(st.accepted);
                if (t - cfg.burn).is_multiple_of(cfg.thin) {
                    let theta = target
                        .subspace()
                        .lift_tensor(&z)
                        .expect("lifting a validated z");
                    let loss = data_b
                        .nll(&theta, shapes, Variant::Cosine)
                        .unwrap_or(f64::NAN);
                    losses.push(loss);
                    phis.push(phi);
                    draws.push(theta);
                }
            }
        }
        per_chain.push(losses);
        phi_chain.push(phis);
        kept.push(draws);
    }
    let (acc, nll, ece, models) = bma_metrics(&kept, shapes, Variant::Cosine, data_b);
    let ess = ess_bulk(&per_chain);
    let ess_phi = ess_bulk(&phi_chain);
    let all: Vec<f64> = per_chain.iter().flatten().copied().collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let sd = (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (all.len() - 1) as f64).sqrt();
    // Predictive MCSE of the mean held-out loss: sd / sqrt(ESS), not
    // sd / sqrt(draws) — correlated draws do not buy precision.
    let nll_mcse = if ess.is_finite() && ess > 0.0 {
        sd / ess.sqrt()
    } else {
        f64::NAN
    };
    CellLine {
        rank,
        kind,
        accept: accepted as f64 / total.max(1) as f64,
        ess_phi,
        rhat_phi: gelman_rubin(&phi_chain),
        ess_loss: ess,
        rhat_loss: gelman_rubin(&per_chain),
        acc,
        nll,
        ece,
        models,
        nll_mcse,
        evals: target.evaluations(),
        seconds: starts.elapsed().as_secs_f64(),
    }
}

fn main() {
    let cfg = Cfg::from_args();
    let shapes = Shapes {
        vocab: 9,
        d: 16,
        ff: 32,
        layers: 2,
    };
    let p = shapes.flat_size();
    let data_a = Dataset::new(Task::A, 8, cfg.n_train, cfg.n_test, cfg.seed);
    let data_b = Dataset::new(Task::B, 8, cfg.n_train, cfg.n_test, cfg.seed);
    let theta0 = init_flat(&shapes, 0.1, cfg.seed ^ 2);
    let (theta_ref, trace_a) = momentum_sgd(
        &data_a,
        &shapes,
        Variant::Cosine,
        theta0,
        cfg.pretrain_steps,
        0.6,
        0.9,
    );
    let ref_flat = flat(&theta_ref);
    let (base_acc, base_nll, _, _) = bma_metrics(
        &[vec![theta_ref.clone()]],
        &shapes,
        Variant::Cosine,
        &data_b,
    );

    println!(
        "R1 restricted subspace: r in {:?}, {} chains x {} steps (burn {}, thin {}), \
         beta {:.3} (proposal), lambda {:.2} (target), sigma {:.2}",
        cfg.ranks, cfg.chains, cfg.steps, cfg.burn, cfg.thin, cfg.beta, cfg.lambda, cfg.sigma
    );
    println!(
        "P = {p}, N_train = {}, N_test = {}. anchor = Task-A pretrain \
         (loss {:.4} -> {:.4}); on B: {:.2}% / NLL {:.4} (no adaptation)",
        cfg.n_train,
        cfg.n_test,
        trace_a.first().copied().unwrap_or(f64::NAN),
        trace_a.last().copied().unwrap_or(f64::NAN),
        base_acc,
        base_nll
    );

    let path = fit_path(&data_b, &shapes, &theta_ref, cfg.path_steps, 0.05);
    let path_first = path
        .first()
        .map(|t| t.iter().map(|x| x * x).sum::<f32>().sqrt());
    let path_last = path
        .last()
        .map(|t| t.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!(
        "informed basis: PCA of {} Task-B momentum steps (lr 0.05, |theta| {:.3} -> {:.3}); \
         data reuse — this arm's basis saw the training split",
        path.len(),
        path_first.unwrap_or(f32::NAN),
        path_last.unwrap_or(f32::NAN)
    );
    println!(
        "\n{:>4}  {:>8}  {:>7}  {:>7}  {:>6}  {:>8}  {:>7}  {:>8}  {:>9}  {:>6}  {:>8}  {:>6}  {:>9}  {:>7}",
        "rank",
        "basis",
        "accept",
        "ESS(Phi)",
        "Rhat",
        "ESS(loss)",
        "Rhat",
        "BMA acc",
        "NLL",
        "ECE",
        "MCSE",
        "K",
        "evals",
        "s"
    );
    let mut lines = Vec::new();
    for &rank in &cfg.ranks {
        for &kind in &cfg.cells {
            let built = match kind {
                Kind::Random => Basis::random(p, rank, cfg.seed ^ rank as u64),
                Kind::Informed => pca_basis(&path, rank, cfg.seed ^ rank as u64),
            };
            let basis = match built {
                Ok(b) => b,
                Err(e) => {
                    println!("{rank:>4}  {:>8}  unavailable: {e}", kind.label());
                    continue;
                }
            };
            let line = run_cell(&cfg, &shapes, &data_b, &ref_flat, rank, kind, basis);
            println!(
                "{:>4}  {:>8}  {:>6.1}%  {:>7.0}  {:>6.3}  {:>7.0}  {:>7.3}  {:>7.2}%  {:>9.4}  \
                 {:>6.3}  {:>8.5}  {:>6}  {:>9}  {:>7.1}",
                line.rank,
                line.kind.label(),
                100.0 * line.accept,
                line.ess_phi,
                line.rhat_phi,
                line.ess_loss,
                line.rhat_loss,
                line.acc,
                line.nll,
                line.ece,
                line.nll_mcse,
                line.models,
                line.evals,
                line.seconds
            );
            lines.push(line);
        }
    }

    println!(
        "\nreading: these chains are stationary (fixed target, fixed beta), so ESS/Rhat are \
         stationary-chain diagnostics — unlike the ambient decaying-step traces.  The target is \
         pi_r, the rank-{r}-restricted model, not the ambient posterior: a good BMA here is \
         evidence about that restriction, not about full-dimensional inference.  Compare \
         'informed' against 'random' at the same rank: the control is what decides whether \
         the path's directions carried anything.",
        r = cfg.ranks.iter().copied().max().unwrap_or(0)
    );
    let _ = lines;
}
