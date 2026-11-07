//! The toy sequence tasks (plan Phase 0: "character prediction or
//! modular arithmetic") and the baselines the adaptation is measured
//! against (Phase 4's "SGD/Adam" comparator, plus standard SGLD —
//! the classical stochastic-gradient Langevin sampler that is the
//! obvious analogous algorithm to unadjusted SG-pCNL).
//!
//! Token layout: `0 = BOS`, digits `0..K` are tokens `1..=K`.
//! Sequence `[BOS, a, b]` predicts the next token:
//!
//! * [`Task::A`] — `(a + b) mod K`  (the base task)
//! * [`Task::B`] — `(a + 2b) mod K` (the adaptation target)
//! * [`Task::Modular`] — `(pa + qb + r) mod K` (the task×budget
//!   matrix's knob: `r ≠ 0` shifts are label-permutations of A,
//!   `(p,q) ≠ (1,1)` needs genuinely new features)
//!
//! The full training set *is* the pCNL potential Φ — no minibatching
//! inside the MCMC loop, so the Metropolis ratio is exact for the
//! data the plan's `Φ(θ) = L(D; θ)` names.

use crate::model::{Shapes, Variant, loss_and_grad};
use candle_core::{Device, Result, Tensor};
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    A,
    B,
    /// `(p·a + q·b + r) mod K` — the configurable adaptation target
    /// (the task×budget matrix's knob). `A ≡ (1,1,0)`, `B ≡ (1,2,0)`;
    /// the shift `r ≠ 0` family is *label-permutation-close* to A
    /// (same features, rotated head), the `(p,q)` ≠ `(1,1)` family
    /// requires genuinely new features.
    Modular {
        p: usize,
        q: usize,
        r: usize,
    },
}

impl Task {
    pub fn label(&self, a: usize, b: usize, k: usize) -> usize {
        match self {
            Task::A => (a + b) % k,
            Task::B => (a + 2 * b) % k,
            Task::Modular { p, q, r } => (p * a + q * b + r) % k,
        }
    }

    /// One-line formula for the report header, so every run names the
    /// task it measured.
    pub fn formula(&self, k: usize) -> String {
        match self {
            Task::A => format!("(a+b) mod {k}"),
            Task::B => format!("(a+2b) mod {k}"),
            Task::Modular { p, q, r } => format!("({p}a+{q}b+{r}) mod {k}"),
        }
    }
}

pub struct Dataset {
    pub k: usize,
    pub train_x: Tensor,
    pub train_y: Tensor,
    pub test_x: Tensor,
    pub test_y: Tensor,
}

fn materialize(task: Task, k: usize, n: usize, seed: u64) -> (Tensor, Tensor) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut xs = Vec::with_capacity(n * 3);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.random_range(0..k);
        let b = rng.random_range(0..k);
        xs.push(0u32); // BOS
        xs.push(a as u32 + 1);
        xs.push(b as u32 + 1);
        ys.push(task.label(a, b, k) as u32 + 1);
    }
    (
        Tensor::from_vec(xs, vec![n, 3], &Device::Cpu).unwrap(),
        Tensor::from_vec(ys, vec![n], &Device::Cpu).unwrap(),
    )
}

impl Dataset {
    pub fn new(task: Task, k: usize, n_train: usize, n_test: usize, seed: u64) -> Self {
        let (train_x, train_y) = materialize(task, k, n_train, seed);
        let (test_x, test_y) = materialize(task, k, n_test, seed.wrapping_add(1));
        Self {
            k,
            train_x,
            train_y,
            test_x,
            test_y,
        }
    }

    /// The exact data potential: mean cross-entropy over the whole
    /// training set (constant-factor scaling of Φ only affects the
    /// temperature, so it is mean — and stated here, not implied).
    pub fn loss_and_grad(
        &self,
        theta: &Tensor,
        shapes: &Shapes,
        variant: Variant,
    ) -> Result<(f64, Tensor)> {
        loss_and_grad(theta, shapes, variant, &self.train_x, &self.train_y)
    }

    /// The same potential, forward pass only.  Exact samplers ask for
    /// `log π(θ)` at every proposal and never for the gradient, so building the
    /// backward graph only to drop it (`loss_and_grad(..).0`, which examples did)
    /// pays for the whole reverse sweep per step.  Identical value to
    /// `loss_and_grad(..).0`, by construction: same forward, same cross-entropy.
    pub fn train_loss(&self, theta: &Tensor, shapes: &Shapes, variant: Variant) -> Result<f64> {
        let logits = crate::model::forward(theta, shapes, variant, &self.train_x)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &self.train_y)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    /// Classification accuracy on the held-out split, in percent.
    pub fn accuracy(&self, theta: &Tensor, shapes: &Shapes, variant: Variant) -> Result<f64> {
        let logits = crate::model::forward(theta, shapes, variant, &self.test_x)?;
        let preds = logits.argmax(1)?.to_vec1::<u32>()?;
        let targets = self.test_y.to_vec1::<u32>()?;
        let hit = preds.iter().zip(&targets).filter(|(p, t)| p == t).count();
        Ok(100.0 * hit as f64 / targets.len() as f64)
    }

    /// Negative log-likelihood (nats) on the held-out split — the
    /// Phase-4 NLL metric.
    pub fn nll(&self, theta: &Tensor, shapes: &Shapes, variant: Variant) -> Result<f64> {
        let logits = crate::model::forward(theta, shapes, variant, &self.test_x)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &self.test_y)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    /// Sum-form mini-batch gradient for SG-pCNL: the mean gradient ×
    /// B ( = Σ_{i∈B} ∇ℓ_i ), which `SgPcnlState` scales by N/B so the
    /// full-potential estimate is N·mean — the plan §3A `g_batch`.
    /// Returns `(mean_loss, sum_grad)`.
    pub fn batch_grad(
        &self,
        theta: &Tensor,
        shapes: &Shapes,
        variant: Variant,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(f64, Tensor)> {
        let (mean_loss, mean_grad) = loss_and_grad(theta, shapes, variant, x, y)?;
        let b = y.dim(0)? as f64;
        Ok((mean_loss, mean_grad.affine(b, 0.0)?))
    }
}

/// Streaming mini-batch loader (plan Phase 2: "streaming Mini-Batch
/// DataLoader in Candle"): uniform samples *without replacement per
/// batch* over the training split. Each chain owns one loader, so the
/// batch stream is independent of the noise stream (two rngs — a
/// shared one would correlate ξ_t with B_t).
pub struct BatchLoader {
    n: usize,
    batch: usize,
    rng: rand::rngs::StdRng,
}

impl BatchLoader {
    pub fn new(n: usize, batch: usize, seed: u64) -> Self {
        Self {
            n,
            batch: batch.min(n),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// One `(X, Y)` batch `[B, …]` from the training split.
    pub fn next(&mut self, data: &Dataset) -> (Tensor, Tensor) {
        let pick = rand::seq::index::sample(&mut self.rng, self.n, self.batch);
        let ids: Vec<u32> = pick.iter().map(|i| i as u32).collect();
        let n = ids.len();
        let idx = Tensor::from_vec(ids, vec![n], &Device::Cpu).expect("index tensor");
        let x = data.train_x.index_select(&idx, 0).expect("batch x");
        let y = data.train_y.index_select(&idx, 0).expect("batch y");
        (x, y)
    }
}

/// Plain momentum-SGD on the flat vector — the deterministic baseline
/// Squared Euclidean norm of a flat tensor, in f64.
fn squared_norm(x: &Tensor) -> f64 {
    crate::sampler::dot(x, x).unwrap_or(f64::NAN)
}

/// The Gaussian half of the Metropolis–Hastings log ratio for the pCNL
/// proposal family
///
/// ```text
/// q(θ′ | θ) = N(θ′; m(θ), β²σ² I),   m(θ) = √(1−β²)(θ−μ) + μ − h·∇U(θ)
/// ```
///
/// namely `log q(θ|θ′) − log q(θ′|θ) = (‖θ′−m(θ)‖² − ‖θ−m(θ′)‖²) / (2β²σ²)`.
/// The Gaussian's normalizer cancels because both ends use the same
/// covariance — the reason *this* kernel needs no determinant term, unlike
/// the spherical one, where `Σ = P D P` is restricted to the tangent space
/// (`sphere::SpherePcn::step_inner`).
///
/// Both means must come from the **same** proposal map: whatever is done to
/// the forward drift (a cap, a temperature) has to be done to the reverse
/// one, or `q(θ|θ′)` is not the density of the move the chain made and the
/// filter targets neither `π` nor the unadjusted chain. `log_pi_diff` is the
/// target's log ratio `log π(θ′) − log π(θ)`, prior and tempered likelihood
/// already combined.
pub fn pcn_proposal_log_ratio(
    log_pi_diff: f64,
    theta: &Tensor,
    proposal: &Tensor,
    m_now: &Tensor,
    m_prop: &Tensor,
    s2: f64,
) -> f64 {
    if s2 <= 0.0 || !s2.is_finite() {
        return log_pi_diff;
    }
    let d_now = squared_norm(&(proposal - m_now).expect("θ′−m(θ)"));
    let d_prop = squared_norm(&(theta - m_prop).expect("θ−m(θ′)"));
    log_pi_diff + (d_now - d_prop) / (2.0 * s2)
}

/// Phase 4 benchmarks pCNL against, and Phase 3's Task-A pre-trainer.
/// Gradients are clipped to norm `PRETRAIN_CLIP` (both variants alike):
/// the clip is part of the *baseline's* definition, so an unnormalized
/// stack's spike cannot kill the pre-train and leave the adaptation
/// comparison with no starting checkpoint. Returns the final
/// parameters and the loss trace.
pub const PRETRAIN_CLIP: f64 = 5.0;

pub fn momentum_sgd(
    data: &Dataset,
    shapes: &Shapes,
    variant: Variant,
    theta0: Tensor,
    steps: usize,
    lr: f64,
    momentum: f64,
) -> (Tensor, Vec<f64>) {
    let mut theta = theta0;
    let mut vel =
        Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("cpu");
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss, grad) = data.loss_and_grad(&theta, shapes, variant).expect("grad");
        trace.push(loss);
        if !loss.is_finite() {
            break; // the stack diverged; keep the trace as evidence
        }
        // clip to the shared norm cap, then v ← μ·v − lr·g; θ ← θ + v
        let g_norm = grad
            .sqr()
            .and_then(|s| s.sum_all())
            .and_then(|s| s.sqrt())
            .and_then(|s| s.to_scalar::<f32>())
            .map(|v| v as f64)
            .unwrap_or(f64::NAN);
        let grad = if g_norm.is_finite() && g_norm > PRETRAIN_CLIP {
            grad.affine(PRETRAIN_CLIP / g_norm, 0.0).expect("clip")
        } else {
            grad
        };
        vel =
            (vel.affine(momentum, 0.0).expect("v") - grad.affine(lr, 0.0).expect("g")).expect("v'");
        theta = (&theta + &vel).expect("theta'");
    }
    (theta, trace)
}

/// Adam on the flat vector — the second deterministic fine-tuning
/// comparator Phase 4 names ("SGD/Adam"). Full-batch, like
/// [`momentum_sgd`], so the two deterministic baselines differ only in
/// the update rule. Standard hyperparameters (β₁=0.9, β₂=0.999,
/// ε=1e-8), no gradient clipping: Adam's per-coordinate normalization
/// is its own protection, and clipping would silently make it a
/// different algorithm. Non-finite loss stops the run (same
/// contract as `momentum_sgd`: the trace is the evidence).
pub fn adam(
    data: &Dataset,
    shapes: &Shapes,
    variant: Variant,
    theta0: Tensor,
    steps: usize,
    lr: f64,
) -> (Tensor, Vec<f64>) {
    let mut theta = theta0;
    let mut m = Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("cpu");
    let mut v = Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("cpu");
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut trace = Vec::with_capacity(steps);
    for t in 1..=steps {
        let (loss, grad) = data.loss_and_grad(&theta, shapes, variant).expect("grad");
        trace.push(loss);
        if !loss.is_finite() {
            break;
        }
        m = (m.affine(b1, 0.0).expect("m") + grad.affine(1.0 - b1, 0.0).expect("g")).expect("m'");
        let g2 = grad.sqr().expect("g²");
        v = (v.affine(b2, 0.0).expect("v") + g2.affine(1.0 - b2, 0.0).expect("g²")).expect("v'");
        // bias-corrected step: θ ← θ − lr · m̂ / (√v̂ + ε)
        let mh = m.affine(1.0 / (1.0 - b1.powi(t as i32)), 0.0).expect("m̂");
        let vh = v.affine(1.0 / (1.0 - b2.powi(t as i32)), 0.0).expect("v̂");
        let denom = (vh.sqrt().expect("√v̂") + eps).expect("√v̂+ε");
        let step = (mh / denom).expect("m̂/√v̂").affine(lr, 0.0).expect("lr·");
        theta = (&theta - &step).expect("θ'");
    }
    (theta, trace)
}

/// Outcome of one standard SGLD chain (plan Phase 4's "standard
/// SGLD" comparator).
pub struct SgldOutcome {
    pub final_theta: Tensor,
    /// Thinned post-burn samples, same cadence as the SG-pCNL chains.
    pub kept: Vec<Tensor>,
    /// Full-batch loss at the kept steps (the ESS/R̂ series).
    pub kept_loss: Vec<f64>,
    pub radial_max: f64,
    pub radial_final: f64,
    /// Max scaled mini-batch gradient actually applied, for the
    /// Phase-0 spike report (same measure the pCNL chains record).
    pub max_scaled_grad: f64,
    pub grad_nonfinite: bool,
    /// Model evaluations, for Phase 4b's counted throughput: one
    /// mini-batch backward per step, one full-batch forward per kept
    /// step (the ESS/R̂ series).
    pub grad_evals: usize,
    pub loss_evals: usize,
}

/// **Standard SGLD** — stochastic-gradient Langevin dynamics with the
/// prior as a gradient term, the classical analogous algorithm to
/// unadjusted SG-pCNL (plan §4 Phase 4: "benchmark against … standard
/// SGLD"). With `h_t = β_t²·σ²/2` (the *same* Robbins–Monro β
/// schedule, so the comparison isolates the algorithm and not the
/// step-size policy):
///
/// ```text
/// θ ← θ − h_t·(g_batch + θ/σ²) + √(2·h_t)·ξ,   ξ ~ N(0, I)
/// ```
///
/// Matched to the pCNL step by construction: likelihood-channel drift
/// `h·g = (β²σ²/2)·g` and per-step noise variance `2h = β²σ²` are
/// *exactly* the pCNL step's; the difference is the prior channel —
/// SGLD puts `−h·θ/σ²` in the drift, pCNL contracts by `√(1−β²)`
/// (which preserves N(0, σ²I) exactly at constant β). The two agree
/// to O(β⁴) per step, so any measured disagreement is discretization,
/// not target: that is what the cross-algorithm R̂ in the report
/// measures.
pub fn sgld_chain(
    cfg: &crate::experiment::Config,
    variant: Variant,
    data: &Dataset,
    theta0: Tensor,
    chain: usize,
    center: Option<&Tensor>,
) -> SgldOutcome {
    use crate::sampler::SgPcnlScheduler;
    use rand::Rng as _;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    let shapes = &cfg.shapes;
    let sigma = cfg.sigma;
    let steps = cfg.steps;
    let scheduler = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    // same per-chain seed convention as the pCNL chains, offset so the
    // two samplers' streams are independent
    let seed = cfg.seed.wrapping_add(800 + chain as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut loader = BatchLoader::new(cfg.n_train, cfg.batch_size, seed.wrapping_add(1));
    let mut theta = theta0;
    let p_dim = theta.elem_count();
    let dev = theta.device().clone();
    let mut kept = Vec::new();
    let mut kept_loss = Vec::new();
    let mut radial_max = 0.0f64;
    let mut radial_final = 0.0f64;
    let mut max_scaled_grad = 0.0f64;
    let mut grad_nonfinite = false;
    let sigma2 = sigma * sigma;

    for t in 0..steps {
        let (bx, by) = loader.next(data);
        let (_batch_loss, sum_grad) = data
            .batch_grad(&theta, shapes, variant, &bx, &by)
            .expect("mini-batch grad");
        // the scaled g that drives the step: (N/B)·Σ = N·mean_B
        let scale = cfg.n_train as f64 / cfg.batch_size as f64;
        match crate::sampler::norm(&sum_grad) {
            Ok(g) if g.is_finite() => max_scaled_grad = max_scaled_grad.max(g * scale),
            _ => grad_nonfinite = true,
        }

        // h_t = β_t²·σ²/2 from the SAME power-law schedule, with the same
        // likelihood temperature the pCNL chains use, so the two samplers
        // stay coupled (cross-R̂) under an annealed start
        let beta = scheduler.beta_at(t);
        let lambda = cfg.potential_scale
            * if cfg.anneal_steps == 0 {
                1.0
            } else {
                (t as f64 / cfg.anneal_steps as f64).min(1.0)
            };
        let h = lambda * beta * beta * sigma2 / 2.0;
        // prior as gradient: −h·(θ−μ)/σ² ⇒ θ ← μ + (1 − β²/2)(θ−μ)
        // (μ = 0 is the plan's origin-centered prior; μ = θ_A the
        // centered adaptation prior — the SGLD twin must target the
        // same posterior as the pCNL chains it is compared to)
        let contraction = 1.0 - beta * beta / 2.0;
        // likelihood drift: −h·(N/B)·Σ
        let drift = sum_grad.affine(h * scale, 0.0).expect("drift");
        // noise: √(2h)·ξ, ξ ~ N(0, I)  (per-step variance 2h = β²σ²,
        // exactly the pCNL step's noise variance)
        let noise: Vec<f32> = (0..p_dim)
            .map(|_| {
                let x: f32 = rng.sample(StandardNormal);
                x
            })
            .collect();
        let xi = Tensor::from_vec(noise, vec![p_dim], &dev).expect("cpu noise");
        let noise_term = xi.affine((2.0 * h).sqrt(), 0.0).expect("√(2h)·ξ");

        let prior = match center {
            Some(c) => ((&theta - c)
                .expect("θ−μ")
                .affine(contraction, 0.0)
                .expect("(1−β²/2)(θ−μ)")
                + c)
                .expect("μ + (1−β²/2)(θ−μ)"),
            None => theta.affine(contraction, 0.0).expect("prior"),
        };
        theta = ((prior - drift).expect("θ−drift") + noise_term).expect("θ'");

        let r = match center {
            Some(c) => crate::sampler::norm(&(&theta - c).expect("θ−μ for radial")),
            None => crate::sampler::norm(&theta),
        }
        .unwrap_or(f64::NAN);
        if r.is_finite() {
            radial_max = radial_max.max(r);
            radial_final = r;
        }
        if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
            kept.push(theta.clone());
            let logits =
                crate::model::forward(&theta, shapes, variant, &data.train_x).expect("forward");
            let l = candle_nn::loss::cross_entropy(&logits, &data.train_y).expect("full loss");
            kept_loss.push(l.to_scalar::<f32>().expect("scalar") as f64);
        }
    }
    let loss_evals = kept_loss.len();
    SgldOutcome {
        final_theta: theta,
        kept,
        kept_loss,
        radial_max,
        radial_final,
        max_scaled_grad,
        grad_nonfinite,
        grad_evals: cfg.steps,
        loss_evals,
    }
}

/// Outcome of one Metropolis-filtered pCNL chain (Phase 4b's
/// reference comparator — the algorithm the specialist diff deleted,
/// kept to *measure* what the filter bought).
pub struct MhOutcome {
    pub final_theta: Tensor,
    pub kept: Vec<Tensor>,
    /// Full-batch potential at the kept steps (cached across
    /// accept/reject — the filter's second pass is reused, not redone).
    pub kept_loss: Vec<f64>,
    pub radial_max: f64,
    pub radial_final: f64,
    /// Accepted proposals out of `steps` proposals.
    pub accepted: usize,
    pub steps: usize,
    /// Model evaluations (Phase 4b's counted throughput): mini-batch
    /// backward passes and full-batch forward passes.
    pub grad_evals: usize,
    pub loss_evals: usize,
    pub grad_nonfinite: bool,
}

/// **Metropolis-filtered pCNL** — the predecessor algorithm the
/// 2026-10-21 specialist diff deleted, restored as the Phase-4b
/// comparator (plan §4 Phase 4b). Same proposal as the unadjusted
/// step ([`crate::sampler::SgPcnlState::step_unadjusted`]), same
/// schedule (Robbins–Monro or cyclical per `cfg.cyclical_period`),
/// same thinning — the *only* difference between the two chains is
/// the filter, so any disagreement localizes in it.
///
/// The filter is the predecessor's own: the full-batch
/// likelihood ratio `log α = L(θ) − L(θ′)`, evaluated on the toy
/// potential `Φ(θ) = L(D; θ)` (the dataset *is* the potential, so
/// there is no mini-batch bias in the ratio itself). Cost accounting
/// (the report's claim 1, counted by the `*_evals` fields): one
/// mini-batch backward + one full-batch forward per step (the
/// current state's `L` is cached across accept/reject), versus the
/// unadjusted chain's one mini-batch backward + a full-batch forward
/// only every `thin` steps.
///
/// Honesty note on exactness: the drift-preconditioned proposal is
/// not exactly symmetric — the fully exact ratio would carry the
/// Gaussian-measure correction `(‖θ − m(θ′)‖² − ‖θ′ − m(θ)‖²) /
/// (2β²σ²)` with `m(θ) = √(1−β²)θ − (β²σ²/2)g(θ)`, which costs a
/// second backward pass. The predecessor used the likelihood ratio
/// alone and this comparator is faithful to it; at β ≤ 0.3 the
/// asymmetry is O(β²) in the log-ratio, and the cross-R̂ between the
/// filtered and unfiltered loss chains is the empirical check that
/// nothing measurable hides in the omission.
/// **Options for the filtered comparator.**  `Default` is the published
/// configuration — the kernel's own step and the untempered potential — so
/// every reported run is unchanged by this type's existence.
#[derive(Clone, Copy, Debug)]
pub struct MhOptions {
    /// Cap on the `(N/B)`-scaled *proposal* gradient (§6.5.2).  `None`
    /// leaves the proposal at the kernel's own step.  The acceptance ratio
    /// is unaffected, so for a drift-free pCN proposal a cap changes the
    /// proposal and not the target.
    ///
    /// **That invariance is approximate unless `exact` is set.**  The default
    /// ratio is the likelihood ratio only, `exp(λN(L(θ)−L(θ′)))`, which is
    /// exact Metropolis only when the drift is zero (pCN's prior cancellation
    /// needs the drift-free mean `√(1−β²)θ`).  With a drift the chain is pCNL,
    /// whose invariant law is not `π`; read cap-comparisons as "same
    /// approximate law to within the pCNL bias", not as "same target" — and use
    /// `exact: true` when the cap has to be shown not to move the target.
    ///
    /// **The radius is not the diagnostic.**  §6.5.6(c) read `‖θ‖ ≈ 1.1–1.3e3`
    /// as pCNL bias, but that is `‖θ_A‖` — the chain starts at `θ_A + σξ` with
    /// `‖θ_A‖ ≈ 1338`, and at `β = 0.03` the OU contraction's `1/β² ≈ 1111`
    /// steps outlast the 400-step run.  The radius never had a chance to relax.
    /// The cosine likelihood is also exactly flat under a global rescale
    /// (`posterior_radius`, and the `cosine_likelihood_is_flat_along_a_radial_ray`
    /// test), so `π`'s radial marginal is the prior's `σ√P` shell at any `λ` and
    /// a genuine excursion would mean the *proposal*, not the target.
    pub grad_norm_cap: Option<f64>,
    /// Temperature on the potential: the proposal drift and the acceptance
    /// log-ratio both use `λ·L`, so `λ = 1` is the published target and
    /// `λ < 1` flattens the likelihood relative to the prior.  This is the
    /// knob §6.5.3's *usable temperature* is a function of: the standard
    /// stack's likelihood is sharp enough that no proposal scale accepts at
    /// `λ = 1`.  Kept losses are always reported as the raw `L`, untempered,
    /// so a `λ < 1` row is comparable to the `λ = 1` rows.
    pub potential_scale: f64,
    /// Full Metropolis–Hastings instead of the predecessor's likelihood-only
    /// filter.  `true` uses the full-batch gradient (the proposal mean must be
    /// a function of `θ`), computes the reverse proposal's density with a
    /// second backward pass, and includes the prior ratio — so the chain
    /// targets `π_λ ∝ N(0,σ²I)·exp(−λ·N·L)` exactly, at 2 backward passes per
    /// step.  `false` (default) is pCNL, the published comparator.
    pub exact: bool,
}

impl Default for MhOptions {
    fn default() -> Self {
        Self {
            grad_norm_cap: None,
            potential_scale: 1.0,
            exact: false,
        }
    }
}

/// The normalized geometry's own gradient scale: the largest *fully scaled*
/// mini-batch gradient norm `‖∇U‖ = ‖g‖·N/B` that the cosine chain produces
/// over its first `steps` steps.
///
/// §6.5.2 anchors the standard stack's drift cap here, because the cap is a
/// statement about the geometry's displacement budget and so has to be
/// measured rather than chosen.  It used to live twice: a private copy in
/// `examples/mh_standard_target.rs` and a rounded literal `7.0e2` in
/// `examples/temperature_sweep.rs`.  The measured caps are 712 / 757 / 764 for
/// the three seeds, so the literal is 6–9 % low — a cap that only changes the
/// proposal cannot move the target, but the two files' acceptance columns were
/// no longer the same experiment.  One definition, called from both:
///
/// ```text
/// cargo run --release -p pcnl --example mh_standard_target
/// ```
///
/// prints the value it uses (`cap C = …`) next to every row.
pub fn cosine_gradient_scale(
    cfg: &crate::experiment::Config,
    data: &Dataset,
    theta_a: &Tensor,
    steps: usize,
) -> f64 {
    use crate::sampler::{SgPcnlScheduler, SgPcnlState};

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
    for _ in 0..steps {
        let (x, y) = loader.next(data);
        let (_l, g) = data
            .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
            .expect("grad");
        if let Ok(n) = crate::sampler::norm(&g) {
            max = max.max(n * cfg.n_train as f64 / cfg.batch_size as f64);
        }
        state.step_unadjusted(&g, &mut rng).expect("step");
    }
    max
}

pub fn mh_pcnl_chain(
    cfg: &crate::experiment::Config,
    variant: Variant,
    data: &Dataset,
    theta0: Tensor,
    chain: usize,
    center: Option<&Tensor>,
    opts: MhOptions,
) -> MhOutcome {
    use crate::sampler::SgPcnlScheduler;
    use rand::Rng as _;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    let shapes = &cfg.shapes;
    let sigma = cfg.sigma;
    let p_dim = shapes.flat_size();
    let scheduler = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    // three independent streams per chain (noise / accept uniforms /
    // batches), all derived from one chain seed
    let seed = cfg.seed.wrapping_add(1200 + chain as u64);
    let mut noise_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut accept_rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1));
    let mut loader = BatchLoader::new(cfg.n_train, cfg.batch_size, seed.wrapping_add(2));
    let n = StandardNormal;
    let jitter: Vec<f32> = (0..p_dim)
        .map(|_| {
            let z: f32 = noise_rng.sample(n);
            z * cfg.sigma as f32
        })
        .collect();
    let dev = theta0.device().clone();
    let mut theta = (theta0 + Tensor::from_vec(jitter, vec![p_dim], &dev).expect("jit"))
        .expect("mh init = theta_a + noise");

    let full_loss = |th: &Tensor| -> Option<f64> {
        let logits = crate::model::forward(th, shapes, variant, &data.train_x).ok()?;
        let l = candle_nn::loss::cross_entropy(&logits, &data.train_y).ok()?;
        let l = l.to_scalar::<f32>().ok()? as f64;
        l.is_finite().then_some(l)
    };
    let mut grad_evals = 0usize;
    let mut loss_evals = 1usize; // L(θ₀), computed below
    let mut loss_now = full_loss(&theta);

    let mut kept = Vec::new();
    let mut kept_loss = Vec::new();
    let mut radial_max = 0.0f64;
    let mut radial_final = 0.0f64;
    let mut accepted = 0usize;
    let mut grad_nonfinite = false;

    for t in 0..cfg.steps {
        let beta = if cfg.cyclical_period == 0 {
            scheduler.beta_at(t)
        } else {
            scheduler.beta_cyclical(t, cfg.cyclical_period)
        };
        // `exact` needs the *full-batch* gradient: the proposal mean must be
        // a deterministic function of θ for the reverse density to exist.
        let (bx, by) = if opts.exact {
            (data.train_x.clone(), data.train_y.clone())
        } else {
            let (x, y) = loader.next(data);
            (x.clone(), y.clone())
        };
        let (_bl, sum_grad) = data
            .batch_grad(&theta, shapes, variant, &bx, &by)
            .expect("mini-batch grad");
        grad_evals += 1;
        match crate::sampler::norm(&sum_grad) {
            Ok(g) if g.is_finite() => {}
            _ => grad_nonfinite = true,
        }
        // proposal — identical to the unadjusted step, translated to
        // the prior mean μ (a pure translation: the pCN prior-ratio
        // cancellation that leaves the filter a likelihood ratio is
        // shift-invariant, so μ = θ_A needs no filter change)
        let scale = if opts.exact {
            1.0 // the full batch is already the potential's gradient
        } else {
            (cfg.n_train as f64) / (cfg.batch_size as f64)
        };
        // Optional drift cap on the *proposal* (default `None` in every
        // reported run).  This is what makes the filtered kernel a fair
        // test for the standard stack (§6.5.3): the cap changes only the
        // proposal, and the Metropolis ratio below keeps the chain exact
        // for the target, so acceptance and the kept predictive say
        // whether the *target* is overconfident rather than the drift.
        //
        // That claim only holds if the cap is part of the *proposal map* and
        // is therefore applied at both ends of the ratio:
        //
        // ```text
        // m(θ) = √(1−β²)(θ−μ) + μ − h·cap(∇U(θ))
        // ```
        //
        // Before 2026-11-06 the reverse mean used the *raw* gradient, so
        // q(θ|θ′) was the density of a move the chain never proposed.  With a
        // binding cap the reverse `‖θ−m(θ′)‖²` is enormous and `log α` is
        // correspondingly huge: the “exact” filter accepted essentially every
        // move and behaved like the unadjusted capped chain it was supposed to
        // test.  No reported run combined `exact` with a cap, so no published
        // number moves; the combination was simply wrong as an API.
        let apply_cap = |g: &Tensor| -> Tensor {
            match opts.grad_norm_cap {
                Some(cap) => {
                    let n = crate::sampler::norm(g).map(|x| x * scale);
                    match n {
                        Ok(n) if n.is_finite() && n > cap => {
                            g.affine(cap / n, 0.0).expect("capped proposal grad")
                        }
                        _ => g.clone(),
                    }
                }
                None => g.clone(),
            }
        };
        let sum_grad = apply_cap(&sum_grad);
        let noise_vec: Vec<f32> = (0..p_dim)
            .map(|_| {
                let x: f32 = noise_rng.sample(StandardNormal);
                x * sigma as f32
            })
            .collect();
        let xi = Tensor::from_vec(noise_vec, vec![p_dim], &dev).expect("cpu noise");
        let decayed = match center {
            Some(c) => ((&theta - c)
                .expect("θ−μ")
                .affine((1.0 - beta * beta).max(0.0).sqrt(), 0.0)
                .expect("decay")
                + c)
                .expect("centered decay"),
            None => theta
                .affine((1.0 - beta * beta).max(0.0).sqrt(), 0.0)
                .expect("decay"),
        };
        let noised = (&decayed + &xi.affine(beta, 0.0).expect("noise")).expect("proposal mean");
        let drift = sum_grad
            .affine(
                opts.potential_scale * scale * beta * beta * sigma * sigma / 2.0,
                0.0,
            )
            .expect("drift");
        let proposal = (&noised - &drift).expect("proposal");

        // The Gaussian-measure correction: with a drift the proposal is not
        // prior-reversible, so q(θ|θ′)/q(θ′|θ) ≠ 1 and the exact ratio needs
        // both means (a second full-batch backward).  `exact` includes it and
        // targets π_λ ∝ N(0,σ²I)·exp(−λ·N·L) term for term.
        // Exact Metropolis needs three terms, not one: the target ratio, the
        // proposal ratio, and the prior ratio.  pCN cancels the last two only
        // when the drift is zero, which is why the likelihood-only ratio is
        // exact in the drift-free limit and pCNL (approximate) otherwise.
        let (log_prior_ratio, exact_means) = if opts.exact {
            let (_lp, g_prop) = data
                .batch_grad(&proposal, shapes, variant, &data.train_x, &data.train_y)
                .expect("full-batch grad at the proposal");
            grad_evals += 1;
            let mean_at = |th: &Tensor, g: &Tensor| -> Tensor {
                let decayed = match center {
                    Some(c) => ((th - c)
                        .expect("θ−μ")
                        .affine((1.0 - beta * beta).max(0.0).sqrt(), 0.0)
                        .expect("decay")
                        + c)
                        .expect("centered decay"),
                    None => th
                        .affine((1.0 - beta * beta).max(0.0).sqrt(), 0.0)
                        .expect("decay"),
                };
                let d = g
                    .affine(
                        opts.potential_scale * beta * beta * sigma * sigma / 2.0,
                        0.0,
                    )
                    .expect("drift");
                (&decayed - &d).expect("proposal mean")
            };
            // `sum_grad` is already capped and scaled; `g_prop` must be too,
            // or the reverse mean is not the mean of the map that proposed.
            let m_now = mean_at(&theta, &sum_grad);
            let m_prop = mean_at(&proposal, &apply_cap(&g_prop));
            // the prior N(μ, σ²I): its ratio is what the drift destroys
            let prior = |th: &Tensor| -> f64 {
                let d = match center {
                    Some(c) => squared_norm(&(th - c).expect("θ−μ")),
                    None => squared_norm(th),
                };
                -d / (2.0 * sigma * sigma)
            };
            (prior(&proposal) - prior(&theta), Some((m_now, m_prop)))
        } else {
            (0.0, None)
        };
        let s2 = beta * beta * sigma * sigma;

        // the predecessor's full-batch likelihood-ratio filter — exact only
        // in the drift-free limit unless `exact` is on (§6.5.6(c))
        loss_evals += 1;
        let loss_prop = full_loss(&proposal);
        let log_alpha = match (loss_now, loss_prop) {
            // the tempered ratio: λ·N·(L(θ) − L(θ′)).  The `N` matters: the
            // drift is the gradient of the *summed* loss Σℓ = N·L, so a ratio
            // on the mean L underweights the likelihood by N — a metropolis
            // chain then accepts climbs it could not, i.e. it is not the
            // stated target at all.
            (Some(l_now), Some(l_prop)) => {
                let log_pi_diff =
                    opts.potential_scale * cfg.n_train as f64 * (l_now - l_prop) + log_prior_ratio;
                match &exact_means {
                    Some((m_now, m_prop)) => {
                        pcn_proposal_log_ratio(log_pi_diff, &theta, &proposal, m_now, m_prop, s2)
                    }
                    None => log_pi_diff,
                }
            }
            // non-finite potential at the proposal ⇒ reject; at the
            // *current* state ⇒ the chain is dead (kept as evidence)
            (_, None) => f64::NEG_INFINITY,
            (None, _) => break,
        };
        let u: f64 = accept_rng.random();
        if u.is_finite() && u.ln() < log_alpha {
            // `PCNL_MH_DEBUG` prints the acceptance decomposition for moves
            // that push the norm outward — the only ones a prior-confined
            // target can never accept in bulk.
            if std::env::var_os("PCNL_MH_DEBUG").is_some() {
                let (r_before, r_after) = (
                    crate::sampler::norm(&theta).unwrap_or(f64::NAN),
                    crate::sampler::norm(&proposal).unwrap_or(f64::NAN),
                );
                if r_after > r_before * 1.002 {
                    let like = match (loss_now, loss_prop) {
                        (Some(l0), Some(l1)) => {
                            opts.potential_scale * cfg.n_train as f64 * (l0 - l1)
                        }
                        _ => f64::NAN,
                    };
                    let q = match &exact_means {
                        Some((m_now, m_prop)) => {
                            let p =
                                pcn_proposal_log_ratio(0.0, &theta, &proposal, m_now, m_prop, s2);
                            if p.is_finite() { p } else { f64::NAN }
                        }
                        None => f64::NAN,
                    };
                    eprintln!(
                        "t={t:>6} r {r_before:>9.2} -> {r_after:>9.2}  like {like:>10.3}  \
                         prior {log_prior_ratio:>12.3}  q {q:>10.3}  \
                         alpha {log_alpha:>12.3}"
                    );
                }
            }
            theta = proposal;
            accepted += 1;
            loss_now = loss_prop;
        }

        let r = match center {
            Some(c) => crate::sampler::norm(&(&theta - c).expect("θ−μ for radial")),
            None => crate::sampler::norm(&theta),
        }
        .unwrap_or(f64::NAN);
        if r.is_finite() {
            radial_max = radial_max.max(r);
            radial_final = r;
        }
        if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
            match loss_now {
                Some(l) => {
                    kept.push(theta.clone());
                    kept_loss.push(l);
                }
                None => break, // current state has no finite potential
            }
        }
    }
    // `loss_evals` counts L(θ₀) + one full-batch forward per step;
    // the kept series reuses the cached value — no extra pass.
    MhOutcome {
        final_theta: theta,
        kept,
        kept_loss,
        radial_max,
        radial_final,
        accepted,
        steps: cfg.steps,
        grad_evals,
        loss_evals,
        grad_nonfinite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::init_flat;

    fn tiny() -> Shapes {
        Shapes {
            vocab: 9,
            d: 16,
            ff: 32,
            layers: 2,
        }
    }

    /// The forward-only potential used by exact samplers must return the same
    /// number as the gradient form it replaced — same `forward`, same
    /// `cross_entropy`, no backward pass built.  A sampler whose potential
    /// drifts from the trained objective would silently change its target, so
    /// the identity is asserted rather than assumed.
    #[test]
    fn train_loss_equals_the_loss_from_the_gradient_form() {
        let shapes = tiny();
        let data = Dataset::new(Task::A, 4, 64, 16, 7);
        let theta = init_flat(&shapes, 0.1, 11);
        let fwd = data
            .train_loss(&theta, &shapes, Variant::Cosine)
            .expect("train_loss");
        let (grad_form, _) = data
            .loss_and_grad(&theta, &shapes, Variant::Cosine)
            .expect("loss_and_grad");
        assert_eq!(fwd, grad_form, "train_loss must match loss_and_grad.0");
    }

    #[test]
    fn tasks_are_deterministic_and_distinct() {
        let k = 8;
        assert_eq!(Task::A.label(3, 5, k), 0); // (3+5)%8
        assert_eq!(Task::B.label(3, 5, k), 5); // (3+10)%8
        // A = a+b, B = a+2b coincide exactly when b ≡ 0 (mod K); for
        // b ≠ 0 they must differ, which is what makes B a real target.
        for a in 0..k {
            for b in 1..k {
                assert_ne!(
                    Task::A.label(a, b, k),
                    Task::B.label(a, b, k),
                    "labels must differ at ({a},{b})"
                );
            }
        }
        // the Modular knob degenerates to A and B at their parameters
        // (so a matrix cell at (1,1,0)/(1,2,0) *is* the plan's task),
        // and a pure shift r ≠ 0 is a bijection of A's labels.
        assert_eq!(
            Task::Modular { p: 1, q: 1, r: 0 }.label(3, 5, k),
            Task::A.label(3, 5, k)
        );
        assert_eq!(
            Task::Modular { p: 1, q: 2, r: 0 }.label(3, 5, k),
            Task::B.label(3, 5, k)
        );
        let shift = Task::Modular { p: 1, q: 1, r: 3 };
        for a in 0..k {
            for b in 0..k {
                assert_eq!(
                    shift.label(a, b, k),
                    (Task::A.label(a, b, k) + 3) % k,
                    "shift = A + r"
                );
            }
        }
        assert_eq!(shift.formula(k), "(1a+1b+3) mod 8");
        assert_eq!(Task::B.formula(k), "(a+2b) mod 8");
    }

    #[test]
    fn datasets_shape_and_bounds() {
        let d = Dataset::new(Task::A, 8, 64, 32, 5);
        assert_eq!(d.train_x.dims(), &[64, 3]);
        assert_eq!(d.train_y.dims(), &[64]);
        let ys = d.train_y.to_vec1::<u32>().unwrap();
        assert!(
            ys.iter().all(|&y| (1..=8).contains(&y)),
            "labels in token range"
        );
        let xs = d.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(xs.iter().all(|&x| (0..=8).contains(&x)));
    }

    /// The q-ratio is checked against a dense Gaussian evaluated in f64 at
    /// both ends — not against itself. A sign error, a factor of two, or the
    /// two means swapped all move the answer by O(1) here.
    #[test]
    fn exact_q_ratio_matches_a_dense_gaussian_ratio() {
        let dev = Device::Cpu;
        let p = 3usize;
        let t = |v: Vec<f32>| Tensor::from_vec(v, vec![p], &dev).unwrap();
        let theta = t(vec![0.4, -1.1, 2.0]);
        let proposal = t(vec![0.9, -0.7, 1.4]);
        let m_now = t(vec![0.35, -1.0, 1.9]);
        let m_prop = t(vec![0.8, -0.8, 1.5]);
        let s2 = 0.0625f64; // β = 0.25, σ = 1

        // log q(x|y) = −‖x−m(y)‖²/(2s²) + const, const identical both ways
        let sq = |a: &Tensor, b: &Tensor| -> f64 {
            let a = a.to_vec1::<f32>().unwrap();
            let b = b.to_vec1::<f32>().unwrap();
            a.iter()
                .zip(&b)
                .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
                .sum()
        };
        let want = (sq(&proposal, &m_now) - sq(&theta, &m_prop)) / (2.0 * s2);
        let got = pcn_proposal_log_ratio(0.0, &theta, &proposal, &m_now, &m_prop, s2);
        // f32 storage: the dense side is exact in f64, the tensor side
        // accumulates in f32, so the agreement is to float precision.
        assert!((got - want).abs() < 1e-5, "q ratio {got} vs dense {want}");
        // and the target term passes through unchanged
        let with_pi = pcn_proposal_log_ratio(-3.5, &theta, &proposal, &m_now, &m_prop, s2);
        assert!((with_pi - (want - 3.5)).abs() < 1e-5, "{with_pi}");
        // β = 0 has no proposal density to correct
        assert_eq!(
            pcn_proposal_log_ratio(-3.5, &theta, &proposal, &m_now, &m_prop, 0.0),
            -3.5
        );
    }

    #[test]
    fn momentum_sgd_reduces_loss_on_task_a() {
        let s = tiny();
        let data = Dataset::new(Task::A, 8, 128, 64, 9);
        let theta0 = init_flat(&s, 0.1, 4);
        // 600 steps, not the 150 this budget had before 2026-10-21b:
        // normalizing V and O (the fix `tests/formal_claims.rs`
        // forced — plan §2, Lean M4's ‖W_v‖,‖W_o‖ ≤ 1 hypotheses)
        // removes the two radial gauge directions the raw weights
        // offered and shrinks the attention sublayer's per-layer
        // magnitude, so the *same* claims (≥50% loss reduction,
        // acc > 30% vs 12.5% chance) need a larger budget. Measured:
        // 200→15.6%, 400→26.6%, 600→32.8%, 900→39.1%.
        let (theta, trace) = momentum_sgd(&data, &s, Variant::Cosine, theta0, 600, 0.5, 0.9);
        let first = trace[0];
        let last = *trace.last().unwrap();
        assert!(last < 0.5 * first, "loss {first} -> {last}");
        let acc = data.accuracy(&theta, &s, Variant::Cosine).unwrap();
        assert!(acc > 30.0, "accuracy after pre-training {acc}%"); // chance is 12.5%
    }
}
