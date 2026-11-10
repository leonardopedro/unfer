//! R1/R2 — restricted affine-subspace adaptation, sampled exactly.
//!
//! The model is `θ(z) = θ_ref + σ·B z`, `z ~ N(0, I_r)`, and the chain samples
//! `π_r(dz) ∝ exp(−λ·L_B(θ(z))) N(0, I_r)(dz)` by MH-pCN in `z`
//! (`pcnl::target::mh_pcn_z`).  This is a *different, smaller* posterior than the
//! ambient one — `r` directions of a `P`-dimensional model — so the numbers
//! below answer "how good is the best rank-`r` affine restriction?" and not
//! "what is the posterior over `θ`?".
//!
//! The anchor is the ambient report's own: `AnchorSetup` is the shared code that
//! builds `D_A`, `D_B`, `θ₀` and the Task-A pre-train, and the ambient shapes
//! (`d 32`, `ff 64`, so `P = 16 672`) are the default here.  That is deliberate:
//! the first version of this example re-derived the seeds by hand, used the
//! quick shapes and gave Task B `seed` instead of `seed + 1`, producing a
//! 7.0 %-on-B anchor that could not be read beside the ambient one (4.30 % on B,
//! i.e. below the 12.5 % chance level).  The `anchor` line prints the hash the
//! ambient `run.manifest.json` carries, so "matched" is checkable rather than
//! asserted.
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
//! Two proposals at every basis (R2):
//!
//! * `fixed` — one `β` for all directions, the R1 arm.
//! * `curv` — `β_i = m/√H_ii` from the target's own curvature at `z = 0`, with
//!   `m` adapted during warm-up and frozen at `burn`.  Exact for any `β_i`
//!   (`pcnl::target::mh_pcn_z_scaled`); the point is that one scalar `β` has to
//!   be small enough for the *stiffest* direction, which is why the informed
//!   cells needed `β = 0.05`.  Measured to fail where the mode is displaced from
//!   the anchor: at `z = 0` the potential is locally flat along exactly those
//!   directions, so `1/√H` says "wider than the prior", the `β ≤ 1` cap binds,
//!   and the arm runs pCN's largest step into a steep region.
//! * `pilot` — `β_i = m·sd_i`, the per-coordinate spread of a 300-step pilot run
//!   of the ordinary kernel at `β`, second half only, from the chain's own
//!   stream; those draws are discarded and the chain continues from there.  The
//!   spread cannot exceed the region the pilot visited, so the width is biased
//!   down (safe for acceptance) and the pilot is warmed by the very `β` under
//!   test — both stated because the report's ESS depends on them.
//!   `--pilot-shrink s` damps the log-widths toward their geometric mean
//!   (`s = 1` raw, `s = 0` one isotropic scale); at 300 steps that recovers the
//!   rank-4 loss (ESS 15 → 399) and is the best rank-8 proposal measured
//!   (6 → 154 at `s = 0`), while `s = 0.5` at rank 8 is worse than either end.
//! * `rotcurv` / `rotpilot` — the same two estimates, but as a **full matrix**:
//!   the Hessian at the anchor (`curvature_hessian`) or the pilot's full
//!   covariance, whose eigenbasis is the rotation.  The `β_i` then apply to the
//!   estimate's eigen-directions (`pcnl::target::mh_pcn_z_rotated`), so a stiff
//!   direction that is a combination of the informed coordinates gets its own
//!   step instead of every coordinate paying for the mixture.  Exact for the
//!   same reason as the diagonal arm (`N(0, I)` is invariant under the
//!   rotation), so only mixing can improve — never the target.
//!
//! `λ` is the target temperature (`SubspaceTarget` sums over the training set,
//! so `λ = 1` is `exp(−Σ_i ℓ_i)`); `β` is a proposal scale, not a temperature.
//! Because `β` is constant after burn-in and the target is fixed, these chains
//! are stationary — their ESS/`R̂` are stationary-chain diagnostics, unlike the
//! ambient decaying-step traces (R0.5).
//!
//! ```text
//! cargo run --release -p pcnl --example subspace_adaptation
//! cargo run --release -p pcnl --example subspace_adaptation -- --ranks 1,2 --steps 4000
//! cargo run --release -p pcnl --example subspace_adaptation -- --shapes quick --steps 400
//! ```

use candle_core::Tensor;
use pcnl::diag::{ess_bulk, gelman_rubin};
use pcnl::experiment::{AnchorSetup, Config, anchor_hash, bma_detail, bma_metrics};
use pcnl::model::{Shapes, Variant};
use pcnl::subspace::{AffineSubspace, Basis, pca_basis};
use pcnl::target::{
    DiagScale, FullPotential, FullTarget, RotScale, SubspaceTarget, ZPotential, adapt_beta,
    curvature_diag, curvature_hessian, mh_pcn_w_full, mh_pcn_z, mh_pcn_z_full, mh_pcn_z_rotated,
    mh_pcn_z_scaled,
};
use pcnl::task::Dataset;
use rand::{Rng, SeedableRng};
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

/// Which target the chain samples: the rank-`r` restriction, or the full
/// `P`-dimensional posterior in the `B`/complement split.  These are different
/// targets, so their predictive numbers answer different questions and share a
/// table only to make the comparison visible.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    Restricted,
    Full,
}

impl TargetKind {
    fn label(&self) -> &'static str {
        match self {
            TargetKind::Restricted => "pi_r",
            TargetKind::Full => "pi_P",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Prop {
    /// One `β` for every direction (R1).
    Fixed,
    /// `β_i = m/√H_ii` from the potential's curvature at the anchor (R2).
    Curv,
    /// `β_i = m·sd_i` from a pilot chain's own spread (R2).
    Pilot,
    /// The full Hessian at the anchor, rescaled jointly in its eigenbasis.
    RotCurv,
    /// The pilot's full covariance, rescaled jointly in its eigenbasis.
    RotPilot,
}

impl Prop {
    fn label(&self) -> &'static str {
        match self {
            Prop::Fixed => "fixed",
            Prop::Curv => "curv",
            Prop::Pilot => "pilot",
            Prop::RotCurv => "rotcurv",
            Prop::RotPilot => "rotpilot",
        }
    }

    fn seed_offset(&self) -> u64 {
        match self {
            Prop::Fixed => 0,
            Prop::Curv => 0x2_000,
            Prop::Pilot => 0x4_000,
            Prop::RotCurv => 0x6_000,
            Prop::RotPilot => 0x8_000,
        }
    }
}

/// The per-direction proposal scale a cell uses: the diagonal one, or the same
/// thing in an eigenbasis.  Both adapt and report the same way, so the chain
/// loop does not branch on it.
#[derive(Clone)]
enum Scale {
    Diag(DiagScale),
    Rot(RotScale),
}

impl Scale {
    fn step(
        &self,
        target: &SubspaceTarget,
        z: &mut [f32],
        phi: f64,
        rng: &mut impl Rng,
    ) -> pcnl::target::ZStep {
        match self {
            Scale::Diag(s) => mh_pcn_z_scaled(target, z, phi, &s.betas(), rng),
            Scale::Rot(s) => mh_pcn_z_rotated(target, z, phi, s.rotation(), &s.betas(), rng),
        }
    }

    fn adapt(&mut self, accepted: bool, t: usize) {
        match self {
            Scale::Diag(s) => s.adapt(accepted, t),
            Scale::Rot(s) => s.adapt(accepted, t),
        }
    }

    fn betas(&self) -> Vec<f64> {
        match self {
            Scale::Diag(s) => s.betas(),
            Scale::Rot(s) => s.betas(),
        }
    }

    fn multiplier(&self) -> f64 {
        match self {
            Scale::Diag(s) => s.multiplier(),
            Scale::Rot(s) => s.multiplier(),
        }
    }
}

struct Cfg {
    ranks: Vec<usize>,
    cells: Vec<Kind>,
    proposals: Vec<Prop>,
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
    /// The quick model (`d 16`, `P = 4240`) — for smoke runs only: it is *not*
    /// the ambient anchor, and the report says so.
    quick: bool,
    /// Robbins–Monro target for the per-direction multiplier.
    adapt_target: f64,
    /// Finite-difference step for the curvature diagonal.
    curvature_h: f64,
    /// Length of the width-estimating pilot chain (`pilot` proposal).  Its
    /// second half is used, so the pilot has to be long enough for the first
    /// half to be a warm-up.
    pilot_steps: usize,
    /// Shrink the pilot's log-width spread toward its geometric mean before
    /// using it: `1` is the raw estimate, `0` is one isotropic width.  The
    /// pilot has `150·pilot_steps/300` draws per coordinate, so at higher rank
    /// the per-coordinate differences are mostly noise; this measures how much
    /// of the mixing gain needs them.
    pilot_shrink: f64,
    /// Targets to sample; `full` adds the `pi_P` split chain (two stages per
    /// step, so twice the potential evaluations).
    targets: Vec<TargetKind>,
    /// Starting scale of the complement stage; adapted during burn toward
    /// `adapt_target` and frozen after it.
    beta_q: f64,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            ranks: vec![1, 2, 4, 8],
            cells: vec![Kind::Random, Kind::Informed],
            proposals: vec![Prop::Fixed, Prop::Curv, Prop::Pilot],
            chains: 4,
            steps: 3_000,
            burn: 500,
            thin: 5,
            // The R1 arm's value: small enough for the stiff informed
            // direction, far below what an isotropic target wants.
            beta: 0.05,
            lambda: 1.0,
            sigma: 0.35,
            seed: 20261020,
            path_steps: 32,
            pretrain_steps: 1_500,
            n_train: 512,
            n_test: 256,
            quick: false,
            adapt_target: 0.5,
            curvature_h: 0.01,
            pilot_steps: 300,
            pilot_shrink: 1.0,
            targets: vec![TargetKind::Restricted],
            // The complement is prior-dominated in the intended use, where a
            // step near pCN's maximum is the right proposal; the Robbins–Monro
            // adaptation corrects this start if it is wrong.
            beta_q: 0.5,
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
        if let Some(v) = flag("--proposal") {
            cfg.proposals = v
                .split(',')
                .filter_map(|x| match x.trim() {
                    "fixed" => Some(Prop::Fixed),
                    "curv" => Some(Prop::Curv),
                    "pilot" => Some(Prop::Pilot),
                    "rotcurv" => Some(Prop::RotCurv),
                    "rotpilot" => Some(Prop::RotPilot),
                    _ => None,
                })
                .collect();
        }
        if let Some(v) = flag("--pilot").and_then(|v| v.parse().ok()) {
            cfg.pilot_steps = v;
        }
        if let Some(v) = flag("--pilot-shrink").and_then(|v| v.parse().ok()) {
            cfg.pilot_shrink = v;
        }
        if let Some(v) = flag("--targets") {
            let parsed: Vec<TargetKind> = v
                .split(',')
                .filter_map(|x| match x.trim() {
                    "restricted" => Some(TargetKind::Restricted),
                    "full" => Some(TargetKind::Full),
                    _ => None,
                })
                .collect();
            if !parsed.is_empty() {
                cfg.targets = parsed;
            }
        }
        if let Some(v) = flag("--beta-q").and_then(|v| v.parse().ok()) {
            cfg.beta_q = v;
        }
        if let Some(v) = flag("--shapes") {
            cfg.quick = v.trim() == "quick";
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
        if let Some(v) = flag("--pretrain").and_then(|v| v.parse().ok()) {
            cfg.pretrain_steps = v;
        }
        if let Some(v) = flag("--n-train").and_then(|v| v.parse().ok()) {
            cfg.n_train = v;
        }
        cfg.burn = cfg.burn.min(cfg.steps.saturating_sub(1));
        cfg.thin = cfg.thin.max(1);
        cfg.chains = cfg.chains.max(1);
        cfg
    }

    /// The ambient config, with only the fields this example varies overridden.
    /// Everything else — `k`, `task_b`, `init_std`, the seeds — is the ambient
    /// run's, so `AnchorSetup` reproduces its anchor.
    fn ambient(&self) -> Config {
        Config {
            seed: self.seed,
            n_train: self.n_train,
            n_test: self.n_test,
            pretrain_steps: self.pretrain_steps,
            shapes: if self.quick {
                Shapes {
                    vocab: 9,
                    d: 16,
                    ff: 32,
                    layers: 2,
                }
            } else {
                Config::default().shapes
            },
            ..Config::default()
        }
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

/// Full sample covariance (row-major, `ddof = 1`) of per-coordinate pilot
/// columns that share their step index.  The rotation only needs the matrix to
/// be symmetric; a pilot that never moved gives zero widths in `RotScale`,
/// which the chain then reports as a stuck scale rather than hiding it.
fn column_covariance(columns: &[Vec<f64>]) -> Vec<f64> {
    let r = columns.len();
    let n = columns.first().map(Vec::len).unwrap_or(0);
    let mean: Vec<f64> = columns
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();
    let mut cov = vec![0.0; r * r];
    for i in 0..r {
        for j in i..r {
            let c = if n < 2 {
                f64::NAN
            } else {
                columns[i]
                    .iter()
                    .zip(&columns[j])
                    .map(|(a, b)| (a - mean[i]) * (b - mean[j]))
                    .sum::<f64>()
                    / (n - 1) as f64
            };
            cov[i * r + j] = c;
            cov[j * r + i] = c;
        }
    }
    cov
}

/// Per-column sample standard deviation, `ddof = 1`; `NaN` for a column with
/// fewer than two entries (a rank the pilot never varied).
fn column_sd(kept: &[Vec<f64>]) -> Vec<f64> {
    kept.iter()
        .map(|c| {
            let n = c.len();
            if n < 2 {
                return f64::NAN;
            }
            let mean = c.iter().sum::<f64>() / n as f64;
            (c.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        })
        .collect()
}

/// `(mean, MCSE)` of a linear functional of the chain, `sd/√ESS` over the
/// pooled draws.  `sd/√draws` would be the number for independent draws and is
/// wrong here by whatever factor the autocorrelation gives.
fn mcse(series: &[Vec<f64>]) -> (f64, f64) {
    let all: Vec<f64> = series.iter().flatten().copied().collect();
    if all.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let sd = (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (all.len() - 1) as f64).sqrt();
    // A constant series has no MCSE, and `ess_bulk` is undefined (0/0) there:
    // every draw scored identically, so the estimate's error is 0, not NaN.
    if sd == 0.0 {
        return (mean, 0.0);
    }
    let e = ess_bulk(series);
    let se = if e.is_finite() && e > 0.0 {
        sd / e.sqrt()
    } else {
        f64::NAN
    };
    (mean, se)
}

struct CellLine {
    rank: usize,
    target: TargetKind,
    kind: Kind,
    prop: Prop,
    accept: f64,
    /// Acceptance of the complement stage (`pi_P` only; `NaN` for `pi_r`).
    accept_q: f64,
    /// Mean frozen complement scale (`pi_P` only; `NaN` for `pi_r`).
    beta_q: f64,
    /// ESS/Rhat of the *potential* — the state variable, not a functional of it.
    ess_phi: f64,
    rhat_phi: f64,
    /// BMA predictive accuracy and NLL over the kept draws.
    acc: f64,
    nll: f64,
    ece: f64,
    brier: f64,
    /// MCSE of the *single-draw* accuracy and Brier functionals (not of the BMA
    /// ratio-of-averages; see `BmaDetail`).
    acc_mcse: f64,
    brier_mcse: f64,
    nll_mcse: f64,
    /// Mean of the frozen, per-chain adapted multipliers (`diag` only).
    multiplier: f64,
    /// `(min, max)` of the frozen per-direction proposal scales.
    beta_range: (f64, f64),
    models: usize,
    evals: u64,
    seconds: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    cfg: &Cfg,
    shapes: &Shapes,
    data_b: &Dataset,
    ref_flat: &[f32],
    rank: usize,
    kind: Kind,
    prop: Prop,
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
    // R2 `curv`/`rotcurv`: the shape of the proposal comes from the target's
    // curvature at the anchor, once, outside the chain (`2r` or `2r² + 1`
    // forward passes, not per-step work).  `pilot`/`rotpilot` need the chain's
    // own stream, so they are built inside the chain loop below.
    let seed_scale = match prop {
        Prop::Curv => {
            let h = curvature_diag(&target, &vec![0.0f32; rank], cfg.curvature_h);
            let mut s = DiagScale::from_curvature(&h);
            s.target_accept = cfg.adapt_target;
            Some(Scale::Diag(s))
        }
        Prop::RotCurv => {
            let h = curvature_hessian(&target, &vec![0.0f32; rank], cfg.curvature_h);
            let mut s = RotScale::from_hessian(&h, rank).expect("Hessian is symmetric");
            s.target_accept = cfg.adapt_target;
            Some(Scale::Rot(s))
        }
        Prop::Fixed | Prop::Pilot | Prop::RotPilot => None,
    };
    let starts = Instant::now();
    let mut per_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut phi_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut kept: Vec<Vec<Tensor>> = Vec::with_capacity(cfg.chains);
    let (mut accepted, mut total) = (0usize, 0usize);
    let mut multipliers = Vec::with_capacity(cfg.chains);
    let mut frozen_betas: Vec<f64> = Vec::new();
    for c in 0..cfg.chains {
        // Each chain owns its stream: `c` is in the seed, so the chains are
        // genuinely independent and their Rhat means something.
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            cfg.seed
                ^ ((rank as u64) << 24)
                ^ if kind == Kind::Informed { 0x1_000 } else { 0 }
                ^ prop.seed_offset()
                ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut scale = seed_scale.clone();
        let mut z = vec![0.0f32; rank];
        let mut phi = target.phi(&z);
        if matches!(prop, Prop::Pilot | Prop::RotPilot) {
            // A short chain at the R1 scale, from the same stream, and the
            // spread of its second half: measured widths rather than curvature
            // at a stationary point.  These draws are warm-up (they sit inside
            // `burn`), and the chain continues from here.
            let mut columns = vec![Vec::new(); rank];
            for t in 0..cfg.pilot_steps {
                let st = mh_pcn_z(&target, &mut z, phi, cfg.beta, &mut rng);
                phi = st.phi;
                if t >= cfg.pilot_steps / 2 {
                    for (i, slot) in columns.iter_mut().enumerate() {
                        slot.push(f64::from(z[i]));
                    }
                }
            }
            scale = Some(if prop == Prop::RotPilot {
                let mut s = RotScale::from_covariance(&column_covariance(&columns), rank)
                    .expect("pilot covariance is symmetric and finite");
                s.target_accept = cfg.adapt_target;
                Scale::Rot(s)
            } else {
                let mut s = DiagScale::from_widths_shrunk(&column_sd(&columns), cfg.pilot_shrink);
                s.target_accept = cfg.adapt_target;
                Scale::Diag(s)
            });
        }
        let mut losses = Vec::new();
        let mut phis = Vec::new();
        let mut draws = Vec::new();
        for t in 0..cfg.steps {
            let st = match &mut scale {
                None => mh_pcn_z(&target, &mut z, phi, cfg.beta, &mut rng),
                Some(s) => {
                    let out = s.step(&target, &mut z, phi, &mut rng);
                    // Adaptation is warm-up only; the kept window must see a
                    // fixed kernel or its draws are not a sample of π_r.
                    if t < cfg.burn {
                        s.adapt(out.accepted, t);
                    }
                    out
                }
            };
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
        if let Some(s) = &scale {
            multipliers.push(s.multiplier());
            frozen_betas.extend(s.betas());
        }
        per_chain.push(losses);
        phi_chain.push(phis);
        kept.push(draws);
    }
    let detail = bma_detail(&kept, shapes, Variant::Cosine, data_b);
    let ess_phi = ess_bulk(&phi_chain);
    let (_, nll_mcse) = mcse(&per_chain);
    let (_, acc_mcse) = mcse(&detail.acc_chain);
    let (_, brier_mcse) = mcse(&detail.brier_chain);
    let multiplier = if multipliers.is_empty() {
        f64::NAN
    } else {
        multipliers.iter().sum::<f64>() / multipliers.len() as f64
    };
    let beta_range = if frozen_betas.is_empty() {
        (cfg.beta, cfg.beta)
    } else {
        (
            frozen_betas.iter().copied().fold(f64::INFINITY, f64::min),
            frozen_betas.iter().copied().fold(0.0, f64::max),
        )
    };
    CellLine {
        rank,
        target: TargetKind::Restricted,
        kind,
        prop,
        accept: accepted as f64 / total.max(1) as f64,
        accept_q: f64::NAN,
        beta_q: f64::NAN,
        ess_phi,
        rhat_phi: gelman_rubin(&phi_chain),
        acc: detail.acc,
        nll: detail.nll,
        ece: detail.ece,
        brier: detail.brier,
        acc_mcse,
        brier_mcse,
        nll_mcse,
        multiplier,
        beta_range,
        models: detail.models,
        evals: target.evaluations(),
        seconds: starts.elapsed().as_secs_f64(),
    }
}

/// The same chain on the **full** target `π_P`, sampled in the `B`/complement
/// split: one informed `z` move and one complement `w` move per step, each an
/// exact MH kernel for its conditional (so the sweep is invariant for `π_P`).
///
/// The `pilot` widths are measured here too, but on this target: the pilot runs
/// the same two-stage sweep for `pilot_steps` and takes the spread of the
/// second half of `z`, while `β_q` is adapted alongside.  The `curv*` proposals
/// are not offered for `π_P` — a Hessian of the *split* potential at `w = 0` is
/// a different object from the restricted one and has not been validated, so
/// the example refuses it rather than presenting an unmeasured estimator.
#[allow(clippy::too_many_arguments)]
fn run_full_cell(
    cfg: &Cfg,
    shapes: &Shapes,
    data_b: &Dataset,
    ref_flat: &[f32],
    rank: usize,
    kind: Kind,
    prop: Prop,
    basis: Basis,
) -> CellLine {
    assert!(
        matches!(prop, Prop::Fixed | Prop::Pilot),
        "pi_P supports the fixed and pilot proposals, not {}",
        prop.label()
    );
    let subspace =
        AffineSubspace::isotropic(ref_flat.to_vec(), basis, cfg.sigma).expect("validated subspace");
    let p = subspace.p();
    let target = FullTarget::new(
        subspace,
        cfg.lambda,
        data_b.clone(),
        *shapes,
        Variant::Cosine,
    );
    let fixed_betas = vec![cfg.beta; rank];
    let starts = Instant::now();
    let mut per_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut phi_chain: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut kept: Vec<Vec<Tensor>> = Vec::with_capacity(cfg.chains);
    let (mut accepted_z, mut accepted_w, mut total) = (0usize, 0usize, 0usize);
    let mut multipliers = Vec::with_capacity(cfg.chains);
    let mut frozen_betas: Vec<f64> = Vec::new();
    let mut frozen_q: Vec<f64> = Vec::with_capacity(cfg.chains);
    for c in 0..cfg.chains {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            cfg.seed
                ^ ((rank as u64) << 24)
                ^ if kind == Kind::Informed { 0x1_000 } else { 0 }
                ^ prop.seed_offset()
                // Distinct from the restricted arm's stream: the targets differ,
                // so sharing draws would only invite reading a difference that
                // is not there.
                ^ 0xa_0000
                ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut z = vec![0.0f32; rank];
        // `w = 0` is in the complement, and the projected noise keeps it there.
        let mut w = vec![0.0f32; p];
        let mut phi = target.phi(&z, &w);
        let mut beta_q = cfg.beta_q;
        let mut scale: Option<Scale> = None;
        if prop == Prop::Pilot {
            let mut columns = vec![Vec::new(); rank];
            for t in 0..cfg.pilot_steps {
                let st = mh_pcn_z_full(&target, &mut z, &w, phi, &fixed_betas, &mut rng);
                phi = st.phi;
                let sq = mh_pcn_w_full(&target, &z, &mut w, phi, beta_q, &mut rng);
                phi = sq.phi;
                beta_q = adapt_beta(beta_q, sq.accepted, t, cfg.adapt_target);
                if t >= cfg.pilot_steps / 2 {
                    for (i, slot) in columns.iter_mut().enumerate() {
                        slot.push(f64::from(z[i]));
                    }
                }
            }
            let mut s = DiagScale::from_widths_shrunk(&column_sd(&columns), cfg.pilot_shrink);
            s.target_accept = cfg.adapt_target;
            scale = Some(Scale::Diag(s));
        }
        let mut losses = Vec::new();
        let mut phis = Vec::new();
        let mut draws = Vec::new();
        for t in 0..cfg.steps {
            let zs = match &mut scale {
                None => mh_pcn_z_full(&target, &mut z, &w, phi, &fixed_betas, &mut rng),
                Some(s) => {
                    let out = mh_pcn_z_full(&target, &mut z, &w, phi, &s.betas(), &mut rng);
                    if t < cfg.burn {
                        s.adapt(out.accepted, t);
                    }
                    out
                }
            };
            phi = zs.phi;
            let ws = mh_pcn_w_full(&target, &z, &mut w, phi, beta_q, &mut rng);
            phi = ws.phi;
            if t < cfg.burn {
                beta_q = adapt_beta(beta_q, ws.accepted, t, cfg.adapt_target);
            }
            if t >= cfg.burn {
                total += 1;
                accepted_z += usize::from(zs.accepted);
                accepted_w += usize::from(ws.accepted);
                if (t - cfg.burn).is_multiple_of(cfg.thin) {
                    let theta = target
                        .lift_tensor(&z, &w)
                        .expect("lifting a validated split");
                    let loss = data_b
                        .nll(&theta, shapes, Variant::Cosine)
                        .unwrap_or(f64::NAN);
                    losses.push(loss);
                    phis.push(phi);
                    draws.push(theta);
                }
            }
        }
        if let Some(s) = &scale {
            multipliers.push(s.multiplier());
            frozen_betas.extend(s.betas());
        }
        frozen_q.push(beta_q);
        per_chain.push(losses);
        phi_chain.push(phis);
        kept.push(draws);
    }
    let detail = bma_detail(&kept, shapes, Variant::Cosine, data_b);
    let ess_phi = ess_bulk(&phi_chain);
    let (_, nll_mcse) = mcse(&per_chain);
    let (_, acc_mcse) = mcse(&detail.acc_chain);
    let (_, brier_mcse) = mcse(&detail.brier_chain);
    let multiplier = if multipliers.is_empty() {
        f64::NAN
    } else {
        multipliers.iter().sum::<f64>() / multipliers.len() as f64
    };
    let beta_range = if frozen_betas.is_empty() {
        (cfg.beta, cfg.beta)
    } else {
        (
            frozen_betas.iter().copied().fold(f64::INFINITY, f64::min),
            frozen_betas.iter().copied().fold(0.0, f64::max),
        )
    };
    CellLine {
        rank,
        target: TargetKind::Full,
        kind,
        prop,
        accept: accepted_z as f64 / total.max(1) as f64,
        accept_q: accepted_w as f64 / total.max(1) as f64,
        beta_q: frozen_q.iter().sum::<f64>() / frozen_q.len().max(1) as f64,
        ess_phi,
        rhat_phi: gelman_rubin(&phi_chain),
        acc: detail.acc,
        nll: detail.nll,
        ece: detail.ece,
        brier: detail.brier,
        acc_mcse,
        brier_mcse,
        nll_mcse,
        multiplier,
        beta_range,
        models: detail.models,
        evals: target.evaluations(),
        seconds: starts.elapsed().as_secs_f64(),
    }
}

fn main() {
    let cfg = Cfg::from_args();
    let ec = cfg.ambient();
    let shapes = ec.shapes;
    let p = shapes.flat_size();
    let setup = AnchorSetup::new(&ec);
    let (theta_ref, trace_a) = setup.pretrain(&ec, Variant::Cosine);
    let ref_flat = flat(&theta_ref);
    let data_b = setup.data_b.clone();
    let (base_acc, base_nll, _, _) = bma_metrics(
        &[vec![theta_ref.clone()]],
        &shapes,
        Variant::Cosine,
        &data_b,
    );

    println!(
        "R1/R2 restricted subspace: r in {:?}, {} chains x {} steps (burn {}, thin {}), \
         beta {:.3} (fixed arm, pilot arm), lambda {:.2} (target), sigma {:.2}",
        cfg.ranks, cfg.chains, cfg.steps, cfg.burn, cfg.thin, cfg.beta, cfg.lambda, cfg.sigma
    );
    println!(
        "proposals: fixed = one beta; curv = beta_i from the curvature diagonal at the anchor; \
         pilot = beta_i from {} pilot steps (second half) at beta {:.3}; rotcurv/rotpilot = the \
         same estimates' full eigenbasis (joint rescaling); multiplier m adapted during burn and \
         frozen after it",
        cfg.pilot_steps, cfg.beta
    );
    if cfg.pilot_shrink < 1.0 {
        println!(
            "pilot widths: log-spread shrunk toward the geometric mean by s = {:.2} \
             (1 = raw pilot, 0 = isotropic at the pilot's geometric mean)",
            cfg.pilot_shrink
        );
    }
    println!(
        "targets: {}. pi_r = the rank-r restriction (one potential per step). pi_P = the full \
         P-dimensional posterior in the B/complement split: informed z then projected complement \
         w, both scored with the full likelihood (two potentials per step, full supports fixed \
         and pilot only); q start beta {:.2}",
        cfg.targets
            .iter()
            .map(|t| t.label())
            .collect::<Vec<_>>()
            .join(","),
        cfg.beta_q
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
    println!(
        "anchor hash {:#018x}, data hash {:#018x} — the ambient run's \
         run.manifest.json must carry both; if they differ this table is not comparable",
        anchor_hash(&theta_ref),
        data_b.content_hash()
    );
    if cfg.quick {
        println!(
            "shapes = quick (d 16, P = {p}): NOT the ambient anchor — d 32 is, and the \
             numbers below would be a different model's"
        );
    }

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
        "\n{:>4}  {:>6}  {:>8}  {:>5}  {:>7}  {:>13}  {:>7}  {:>6}  {:>7}  {:>8}  {:>8}  \
         {:>7}  {:>8}  {:>9}  {:>6}  {:>6}  {:>13}  {:>6}  {:>8}  {:>6}",
        "rank",
        "target",
        "basis",
        "prop",
        "accept",
        "q acc/beta",
        "ESS(Phi)",
        "Rhat",
        "BMA acc",
        "MCSE(acc)",
        "Brier",
        "MCSE(B)",
        "NLL",
        "ECE",
        "MCSE(NLL)",
        "m",
        "beta_i",
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
            for &target in &cfg.targets {
                for &prop in &cfg.proposals {
                    if target == TargetKind::Full && !matches!(prop, Prop::Fixed | Prop::Pilot) {
                        continue;
                    }
                    let line = match target {
                        TargetKind::Restricted => run_cell(
                            &cfg,
                            &shapes,
                            &data_b,
                            &ref_flat,
                            rank,
                            kind,
                            prop,
                            basis.clone(),
                        ),
                        TargetKind::Full => run_full_cell(
                            &cfg,
                            &shapes,
                            &data_b,
                            &ref_flat,
                            rank,
                            kind,
                            prop,
                            basis.clone(),
                        ),
                    };
                    let range = if line.prop == Prop::Fixed {
                        format!("{:.2}", cfg.beta)
                    } else {
                        format!("{:.1e}..{:.2}", line.beta_range.0, line.beta_range.1)
                    };
                    let q = if line.accept_q.is_finite() {
                        format!("{:.1}%/{:.2}", 100.0 * line.accept_q, line.beta_q)
                    } else {
                        "-".to_string()
                    };
                    println!(
                        "{:>4}  {:>6}  {:>8}  {:>5}  {:>6.1}%  {:>13}  {:>7.0}  {:>6.3}  \
                         {:>7.2}%  {:>8.3}  {:>8.4}  {:>7.4}  {:>9.4}  {:>6.3}  {:>6.4}  \
                         {:>6.2}  {:>13}  {:>6}  {:>8}  {:>6.1}",
                        line.rank,
                        line.target.label(),
                        line.kind.label(),
                        line.prop.label(),
                        100.0 * line.accept,
                        q,
                        line.ess_phi,
                        line.rhat_phi,
                        line.acc,
                        line.acc_mcse,
                        line.brier,
                        line.brier_mcse,
                        line.nll,
                        line.ece,
                        line.nll_mcse,
                        line.multiplier,
                        range,
                        line.models,
                        line.evals,
                        line.seconds
                    );
                    lines.push(line);
                }
            }
        }
    }

    println!(
        "\nreading: these chains are stationary (fixed target, fixed beta after burn), so \
         ESS/Rhat are stationary-chain diagnostics — unlike the ambient decaying-step traces.  \
         The target is pi_r, the rank-r-restricted model, not the ambient posterior: a good BMA \
         here is evidence about that restriction, not about full-dimensional inference.  The \
         pi_P rows answer the other half: same basis and proposal, but the full likelihood and \
         a complement move, so their predictives are a sample of the full P-dimensional \
         posterior (up to chain error) and are directly comparable with the ambient run's, not \
         with pi_r's.  Compare 'informed' against 'random' at the same rank, target and \
         proposal: the control decides whether the path's directions carried anything.  MCSE(acc)/ \
         MCSE(B) are the sd/sqrt(ESS) of the mean single-draw accuracy/Brier, not of the BMA \
         ratio-of-averages; pi_P spends two evaluations per step, so its 'evals' column is twice \
         its step count."
    );
    let _ = lines;
}
