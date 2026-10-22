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
    Modular { p: usize, q: usize, r: usize },
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

        // h_t = β_t²·σ²/2 from the SAME power-law schedule
        let beta = scheduler.beta_at(t);
        let h = beta * beta * sigma2 / 2.0;
        // prior as gradient: −h·θ/σ² ≡ θ·(1 − β²/2)
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

        let prior = theta.affine(contraction, 0.0).expect("prior");
        theta = ((prior - drift).expect("θ−drift") + noise_term).expect("θ'");

        let r = crate::sampler::norm(&theta).unwrap_or(f64::NAN);
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
/// The filter is the predecessor's own: the exact full-batch
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
pub fn mh_pcnl_chain(
    cfg: &crate::experiment::Config,
    variant: Variant,
    data: &Dataset,
    theta0: Tensor,
    chain: usize,
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
        let (bx, by) = loader.next(data);
        let (_bl, sum_grad) = data
            .batch_grad(&theta, shapes, variant, &bx, &by)
            .expect("mini-batch grad");
        grad_evals += 1;
        match crate::sampler::norm(&sum_grad) {
            Ok(g) if g.is_finite() => {}
            _ => grad_nonfinite = true,
        }
        // proposal — identical to the unadjusted step
        let scale = (cfg.n_train as f64) / (cfg.batch_size as f64);
        let noise_vec: Vec<f32> = (0..p_dim)
            .map(|_| {
                let x: f32 = noise_rng.sample(StandardNormal);
                x * sigma as f32
            })
            .collect();
        let xi = Tensor::from_vec(noise_vec, vec![p_dim], &dev).expect("cpu noise");
        let decayed = theta
            .affine((1.0 - beta * beta).max(0.0).sqrt(), 0.0)
            .expect("decay");
        let noised = (&decayed + &xi.affine(beta, 0.0).expect("noise")).expect("proposal mean");
        let drift = sum_grad
            .affine(scale * beta * beta * sigma * sigma / 2.0, 0.0)
            .expect("drift");
        let proposal = (&noised - &drift).expect("proposal");

        // the predecessor's exact full-batch likelihood-ratio filter
        loss_evals += 1;
        let loss_prop = full_loss(&proposal);
        let log_alpha = match (loss_now, loss_prop) {
            (Some(l_now), Some(l_prop)) => l_now - l_prop,
            // non-finite potential at the proposal ⇒ reject; at the
            // *current* state ⇒ the chain is dead (kept as evidence)
            (_, None) => f64::NEG_INFINITY,
            (None, _) => break,
        };
        let u: f64 = accept_rng.random();
        if u.is_finite() && u.ln() < log_alpha {
            theta = proposal;
            accepted += 1;
            loss_now = loss_prop;
        }

        let r = crate::sampler::norm(&theta).unwrap_or(f64::NAN);
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
