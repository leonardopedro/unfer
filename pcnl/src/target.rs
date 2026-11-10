//! R1 — target identity and the exact MH-pCN kernel *in `z`*.
//!
//! The restricted model is a target on `z ∈ R^r`, not on `θ`:
//!
//! ```text
//! θ(z) = θ_ref + C_half B z,     z ~ N(0, I_r)
//! Φ(z) = λ · Σ_i loss_i(θ(z)) + ½‖z‖²      (the Gibbs potential)
//! π_r(dz) ∝ exp(−Φ(z)) dz
//! ```
//!
//! The prior term is part of `Φ` here even though it cancels in the MH ratio:
//! it is what makes `Φ` a potential rather than a likelihood, and reporting it
//! keeps the target's identity (temperature, prior, restriction) explicit.
//!
//! Why pCN is the proposal: `z' = √(1−β²)·z + β·ξ` is reversible with respect
//! to `N(0, I)`, so the Gaussian prior cancels and the exact ratio is
//! `min(0, Φ(z) − Φ(z'))` with the *common full-data* target at both ends.  A
//! Langevin proposal does not have this property, and using its ratio without
//! the drift-density correction would silently target something else — which is
//! precisely the R0-class error this module exists to avoid.  Note that `β` is
//! a proposal scale here, not a temperature: `λ` changes the target.

use crate::model::{Shapes, Variant};
use crate::normalize::NormGrad;
use crate::subspace::AffineSubspace;
use crate::task::Dataset;
use candle_core::{Device, Tensor};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};

/// A potential on `z` with Gibbs density `∝ exp(−φ(z))`.
///
/// Deliberately one method: an exact MH kernel needs values, and nothing else.
/// Asking for gradients here would invite an unadjusted kernel to use them
/// without the density correction that makes it exact.
pub trait ZPotential {
    fn dim(&self) -> usize;
    fn phi(&self, z: &[f32]) -> f64;
}

/// The restricted target over the full training set, evaluated through the
/// model.  `phi` returns `+∞` where the model cannot produce a finite loss, so
/// a divergent proposal is a rejection rather than a `NaN` ratio.
pub struct SubspaceTarget {
    subspace: AffineSubspace,
    lambda: f64,
    data: Dataset,
    shapes: Shapes,
    variant: Variant,
    /// `N` in the sum-form likelihood.  `Dataset::train_loss` is the *mean*
    /// cross-entropy, so the sum the plan specifies is `N` times it: at `λ = 1`
    /// this target is the likelihood `exp(−Σ_i ℓ_i)`, whereas the ambient
    /// experiment's `Φ` is the mean loss, i.e. this target at `λ = 1/N`.  That
    /// factor is the difference between a prior-dominated posterior and a
    /// data-determined one, so it is named here rather than left to a caller's
    /// convention.
    n_train: usize,
    evals: AtomicU64,
}

impl SubspaceTarget {
    pub fn new(
        subspace: AffineSubspace,
        lambda: f64,
        data: Dataset,
        shapes: Shapes,
        variant: Variant,
    ) -> Self {
        let n_train = data.train_y.dim(0).unwrap_or(0);
        Self {
            subspace,
            lambda,
            data,
            shapes,
            variant,
            n_train,
            evals: AtomicU64::new(0),
        }
    }

    pub fn n_train(&self) -> usize {
        self.n_train
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn subspace(&self) -> &AffineSubspace {
        &self.subspace
    }

    /// Number of full-data forward passes this target has answered.  The
    /// acceptance gate's cost denominator (`per evaluation`, not `per step`):
    /// one step is one evaluation, and a report that omits it hides the price.
    pub fn evaluations(&self) -> u64 {
        self.evals.load(Ordering::Relaxed)
    }

    /// `(λ · N · mean_loss(θ(z)), ½‖z‖²)` — the sum-form likelihood term and the
    /// prior term.
    pub fn phi_parts(&self, z: &[f32]) -> (f64, f64) {
        let theta = match self.subspace.lift_tensor(z) {
            Ok(t) => t,
            Err(_) => return (f64::INFINITY, prior_phi(z)),
        };
        let mean_loss = self
            .data
            .train_loss(&theta, &self.shapes, self.variant)
            .ok()
            .filter(|l| l.is_finite())
            .unwrap_or(f64::INFINITY);
        (self.lambda * self.n_train as f64 * mean_loss, prior_phi(z))
    }

    /// `∇_z Φ(z)` for `Φ = λ·N·mean_loss(θ(z)) + ½‖z‖²`, by the chain rule
    /// `dΦ/dz = λ·Bᵀ C_halfᵀ ∇_θ ℓ_sum`, plus the prior's `z`.
    ///
    /// This exists so a *mode* of the target can be found — the point where the
    /// Laplace widths `1/√H_ii` mean the posterior's spread rather than the
    /// spread of the training loss's local bowl.  It is deliberately not part of
    /// [`ZPotential`]: the kernels must not be able to become unadjusted by
    /// accident (see [`FullTarget`]), so gradients are opt-in here.  `None` when
    /// the model or the backward pass fails, matching `phi`'s
    /// `INFINITY` contract.
    /// The derivative is taken with [`NormGrad::Exact`], not the `Stopped`
    /// convention `forward` defaults to.  `Stopped` keeps the normalising
    /// denominators constant, so it is deliberately *not* the derivative of
    /// `phi`'s value — measured here at ~1 000× on a two-layer model, because
    /// each layer's Frobenius normalisation contributes its own `1/‖W‖` to the
    /// gap.  A mode search driven by the stopped gradient descends a different
    /// function than the one the chain samples, which is exactly the class of
    /// quiet error this file's `subspace_target_gradient_matches_finite_
    /// differences` test exists to catch.
    pub fn grad_phi(&self, z: &[f32]) -> Option<Vec<f32>> {
        let theta = self.subspace.lift_tensor(z).ok()?;
        let (loss, mean_grad) = crate::model::loss_and_grad_norm(
            &theta,
            &self.shapes,
            self.variant,
            &self.data.train_x,
            &self.data.train_y,
            NormGrad::Exact,
        )
        .ok()?;
        if !loss.is_finite() {
            return None;
        }
        let scaled = mean_grad.affine(self.n_train as f64, 0.0).ok()?;
        let g_theta = scaled.flatten_all().ok()?.to_vec1::<f32>().ok()?;
        let mut g = self.subspace.gradient_to_z(&g_theta);
        for (gi, zi) in g.iter_mut().zip(z.iter()) {
            *gi = (self.lambda * f64::from(*gi) + f64::from(*zi)) as f32;
        }
        Some(g)
    }
}

/// `½‖z‖²` — the `N(0, I_r)` potential, in closed form so it costs `r` flops.
pub fn prior_phi(z: &[f32]) -> f64 {
    0.5 * z.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
}

impl ZPotential for SubspaceTarget {
    fn dim(&self) -> usize {
        self.subspace.dim()
    }

    fn phi(&self, z: &[f32]) -> f64 {
        self.evals.fetch_add(1, Ordering::Relaxed);
        let (lik, prior) = self.phi_parts(z);
        lik + prior
    }
}

/// A target on the ambient `P`-dimensional space, split as `v = Bz + w` with
/// `w` in the orthogonal complement of `span(B)`.
///
/// This is the interface the full-target kernels need and no more: values, a
/// dimension for the informed block, and the projector that keeps the
/// complement move in the complement.  In particular there is no gradient —
/// the same reason [`ZPotential`] has none, so an unadjusted kernel cannot be
/// built here by accident.
pub trait FullPotential {
    /// Informed dimension `r` (`B` has `r` columns).
    fn dim(&self) -> usize;
    /// Ambient dimension `P`.
    fn p(&self) -> usize;
    /// `Φ(z, w)`; `z` is in `B`-coordinates, `w` is an ambient vector.
    fn phi(&self, z: &[f32], w: &[f32]) -> f64;
    /// `out = (I − BBᵀ) v` — the component of `v` orthogonal to the basis.
    fn complement(&self, v: &[f32], out: &mut [f32]);
}

/// The **full** target, not the restricted one:
///
/// ```text
/// v = B z + w,   z ∈ R^r,   w ⊥ span(B)
/// θ = θ_ref + C_half v
/// Φ(z, w) = λ · N · mean_loss(θ) + ½‖z‖² + ½‖w‖²
/// ```
///
/// With `w ⊥ Bz` the prior term is `½‖v‖²`, so this is the same
/// `π_P ∝ N(0,I_P)(v) exp(−λN L(θ_ref + C_half v))` that `task::mh_pcnl_chain`
/// filters (its exact/drift-free arm), just represented in the split that lets
/// the informed block and the complement be proposed separately.  The split is
/// a bijection because `w` is constrained to the complement; `Bz` and `w` are
/// orthogonal, so the Gaussian factorises into `N(0, I_r) × N(0, I_{P−r})` and
/// each stage's pCN ratio is unchanged.
///
/// It is a *different target* from [`SubspaceTarget`] at the same `λ` and `σ`:
/// the restricted one integrates the likelihood over `r` directions only, while
/// this one carries all `P`.  Comparing their predictives is the R2 question;
/// nothing here claims the full posterior predicts better.
pub struct FullTarget {
    subspace: AffineSubspace,
    lambda: f64,
    data: Dataset,
    shapes: Shapes,
    variant: Variant,
    n_train: usize,
    evals: AtomicU64,
}

impl FullTarget {
    pub fn new(
        subspace: AffineSubspace,
        lambda: f64,
        data: Dataset,
        shapes: Shapes,
        variant: Variant,
    ) -> Self {
        let n_train = data.train_y.dim(0).unwrap_or(0);
        Self {
            subspace,
            lambda,
            data,
            shapes,
            variant,
            n_train,
            evals: AtomicU64::new(0),
        }
    }

    pub fn n_train(&self) -> usize {
        self.n_train
    }

    pub fn subspace(&self) -> &AffineSubspace {
        &self.subspace
    }

    pub fn evaluations(&self) -> u64 {
        self.evals.load(Ordering::Relaxed)
    }

    /// `θ_ref + C_half (Bz + w)`.
    pub fn lift(&self, z: &[f32], w: &[f32]) -> Vec<f32> {
        let p = self.subspace.p();
        assert_eq!(z.len(), self.subspace.dim(), "z must have r entries");
        assert_eq!(w.len(), p, "w must have P entries");
        let mut v = vec![0.0f32; p];
        self.subspace.basis().apply(z, &mut v);
        for (vi, wi) in v.iter_mut().zip(w) {
            *vi += *wi;
        }
        for ((o, a), s) in v
            .iter_mut()
            .zip(self.subspace.theta_ref())
            .zip(self.subspace.scale())
        {
            *o = a + s * *o;
        }
        v
    }

    pub fn lift_tensor(&self, z: &[f32], w: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::from_vec(self.lift(z, w), vec![self.subspace.p()], &Device::Cpu)
    }

    /// `(λN·mean_loss, ½‖z‖² + ½‖w‖²)`.
    pub fn phi_parts(&self, z: &[f32], w: &[f32]) -> (f64, f64) {
        let theta = match self.lift_tensor(z, w) {
            Ok(t) => t,
            Err(_) => return (f64::INFINITY, prior_phi(z) + prior_phi(w)),
        };
        let mean_loss = self
            .data
            .train_loss(&theta, &self.shapes, self.variant)
            .ok()
            .filter(|l| l.is_finite())
            .unwrap_or(f64::INFINITY);
        (
            self.lambda * self.n_train as f64 * mean_loss,
            prior_phi(z) + prior_phi(w),
        )
    }
}

impl FullPotential for FullTarget {
    fn dim(&self) -> usize {
        self.subspace.dim()
    }

    fn p(&self) -> usize {
        self.subspace.p()
    }

    fn phi(&self, z: &[f32], w: &[f32]) -> f64 {
        self.evals.fetch_add(1, Ordering::Relaxed);
        let (lik, prior) = self.phi_parts(z, w);
        lik + prior
    }

    fn complement(&self, v: &[f32], out: &mut [f32]) {
        self.subspace.complement(v, out);
    }
}

/// What one MH-pCN step in `z` did.
#[derive(Clone, Copy, Debug)]
pub struct ZStep {
    pub accepted: bool,
    /// `Φ(z)` after the step (the current point's potential, accepted or not).
    pub phi: f64,
    pub proposed: f64,
}

/// Exact MH-pCN in `z`: propose `z' = √(1−β²) z + β ξ` and accept with
///
/// ```text
/// log α = min(0, [Φ(z) − Φ(z')] + log q(z|z') − log q(z'|z))
/// log q(z|z') − log q(z'|z) = −(‖z‖² − ‖z'‖²)/2
/// ```
///
/// The proposal-density term is not optional bookkeeping: pCN's proposal is not
/// symmetric in `z`, and its density ratio is what makes a `N(0, I)` prior
/// cancel.  Omitting it while keeping the prior inside `Φ` doubles the
/// quadratic part of the target (a unit Gaussian collapses to variance ½ and a
/// double well narrows) — which is exactly what the first version of this
/// function did, and why `mh_pcn_z_matches_quadrature_on_rank_two` exists.  The
/// term is written out rather than cancelled by hand so that a target *without*
/// a Gaussian prior is also handled correctly, and so the cancellation is
/// visible in the tests rather than assumed.
///
/// A non-finite `log α` is a rejection.  That includes `Φ(z') = +∞` (the
/// proposal is outside the model's support) and the `∞ − ∞` case of a state that
/// is already non-finite: an undefined ratio must not be turned into an accept
/// by `min(0, NaN)`.
pub fn mh_pcn_z(
    phi: &dyn ZPotential,
    z: &mut [f32],
    phi_z: f64,
    beta: f64,
    rng: &mut impl Rng,
) -> ZStep {
    assert_eq!(z.len(), phi.dim(), "z must have the target's dimension");
    assert!(beta > 0.0 && beta <= 1.0, "beta must be in (0, 1]");
    let shrink = (1.0 - beta * beta).sqrt();
    let proposal: Vec<f32> = z
        .iter()
        .map(|&zi| (shrink * f64::from(zi) + beta * normal(rng)) as f32)
        .collect();
    let phi_new = phi.phi(&proposal);
    // −(‖z‖² − ‖z'‖²)/2 = ½‖z'‖² − ½‖z‖²
    let log_q_ratio = prior_phi(&proposal) - prior_phi(z);
    let log_alpha = phi_z - phi_new + log_q_ratio;
    let u: f64 = rng.random();
    let accept = log_alpha.is_finite() && log_alpha >= u.ln();
    if accept {
        z.copy_from_slice(&proposal);
        ZStep {
            accepted: true,
            phi: phi_new,
            proposed: phi_new,
        }
    } else {
        ZStep {
            accepted: false,
            phi: phi_z,
            proposed: phi_new,
        }
    }
}

/// Exact MH-pCN in `z` with a **per-direction** proposal scale.
///
/// ```text
/// z'_i = √(1−β_i²) z_i + β_i ξ_i,        ξ_i ~ N(0, 1)
/// log α = min(0, [Φ(z) − Φ(z')] + ½‖z'‖² − ½‖z‖²)
/// ```
///
/// The point of allowing `β_i` to differ is that the interesting restraint here
/// is anisotropic: in the informed basis a few directions carry the data and
/// are pinned to a narrow posterior, while the rest are still prior-width.  One
/// scalar `β` has to be small enough for the stiffest direction, which is why
/// the informed rank-1/2 cells needed `β = 0.05` (acceptance 78–85 %, ESS of
/// `Φ` ≈ 100) — the flat directions were being moved at the stiff direction's
/// step size.
///
/// Exactness is not sacrificed: each coordinate's kernel is the 1-D pCN kernel,
/// which is reversible with respect to `N(0, 1)` because `a_i² + β_i² = 1`.  The
/// density ratio also keeps its simple form, coordinate by coordinate:
///
/// ```text
/// q(z_i|z'_i) − q(z'_i|z_i) = [(z'_i − a_i z_i)² − (z_i − a_i z'_i)²]/(2β_i²)
///                          = (1 − a_i²)(z_i² − z'_i²)/(2β_i²) = (z_i² − z'_i²)/2
/// ```
///
/// so the sum over coordinates is `½‖z'‖² − ½‖z‖²`, identical to the uniform
/// kernel.  A diagonal preconditioner therefore costs nothing in the ratio and
/// cannot silently change the target — which is exactly why it is worth trying
/// before a proposal whose density is harder to get right.
pub fn mh_pcn_z_scaled(
    phi: &dyn ZPotential,
    z: &mut [f32],
    phi_z: f64,
    betas: &[f64],
    rng: &mut impl Rng,
) -> ZStep {
    assert_eq!(z.len(), phi.dim(), "z must have the target's dimension");
    assert_eq!(betas.len(), z.len(), "one beta per coordinate");
    assert!(
        betas.iter().all(|&b| b > 0.0 && b <= 1.0),
        "every beta must be in (0, 1]"
    );
    let proposal: Vec<f32> = z
        .iter()
        .zip(betas)
        .map(|(&zi, &b)| (((1.0 - b * b).sqrt() * f64::from(zi)) + b * normal(rng)) as f32)
        .collect();
    let phi_new = phi.phi(&proposal);
    let log_q_ratio = prior_phi(&proposal) - prior_phi(z);
    let log_alpha = phi_z - phi_new + log_q_ratio;
    let u: f64 = rng.random();
    let accept = log_alpha.is_finite() && log_alpha >= u.ln();
    if accept {
        z.copy_from_slice(&proposal);
        ZStep {
            accepted: true,
            phi: phi_new,
            proposed: phi_new,
        }
    } else {
        ZStep {
            accepted: false,
            phi: phi_z,
            proposed: phi_new,
        }
    }
}

/// One **informed** MH-pCN move on the full target, holding `w` fixed.
///
/// This is the first stage of the split sweep: `z'` is proposed exactly as in
/// [`mh_pcn_z_scaled`], but the target value is `Φ(z', w)`, the *full* potential
/// at the current complement, and the acceptance ratio is the usual
/// `Φ(z,w) − Φ(z',w) + ½‖z'‖² − ½‖z‖²`.  Holding `w` fixed makes this an exact
/// MH kernel for the conditional `π(z | w)`, so a sweep of this move followed by
/// [`mh_pcn_w_full`] is invariant for the joint target — the two stages are a
/// Metropolis-within-Gibbs sweep, not a single joint proposal.
///
/// The two-stage form is deliberate.  A joint `(z', w')` proposal would need
/// only one potential evaluation per step, but its acceptance would mix the
/// informed and complement scales, so neither `β_i` nor `β_q` could be adapted
/// on a clean signal; here each stage's acceptance means what it says, and a
/// report can show them separately.  The cost is two evaluations per step,
/// which the reports divide by.
pub fn mh_pcn_z_full(
    phi: &dyn FullPotential,
    z: &mut [f32],
    w: &[f32],
    phi_z: f64,
    betas: &[f64],
    rng: &mut impl Rng,
) -> ZStep {
    assert_eq!(z.len(), phi.dim(), "z must have the informed dimension");
    assert_eq!(w.len(), phi.p(), "w must have the ambient dimension");
    assert_eq!(betas.len(), z.len(), "one beta per informed coordinate");
    assert!(
        betas.iter().all(|&b| b > 0.0 && b <= 1.0),
        "every beta must be in (0, 1]"
    );
    let proposal: Vec<f32> = z
        .iter()
        .zip(betas)
        .map(|(&zi, &b)| (((1.0 - b * b).sqrt() * f64::from(zi)) + b * normal(rng)) as f32)
        .collect();
    let phi_new = phi.phi(&proposal, w);
    let log_q_ratio = prior_phi(&proposal) - prior_phi(z);
    let log_alpha = phi_z - phi_new + log_q_ratio;
    let u: f64 = rng.random();
    let accept = log_alpha.is_finite() && log_alpha >= u.ln();
    if accept {
        z.copy_from_slice(&proposal);
        ZStep {
            accepted: true,
            phi: phi_new,
            proposed: phi_new,
        }
    } else {
        ZStep {
            accepted: false,
            phi: phi_z,
            proposed: phi_new,
        }
    }
}

/// What one complement (`w`) MH-pCN step did.
#[derive(Clone, Copy, Debug)]
pub struct WStep {
    pub accepted: bool,
    /// `Φ(z, w)` after the step.
    pub phi: f64,
    pub proposed: f64,
}

/// One **complement** MH-pCN move on the full target, holding `z` fixed.
///
/// ```text
/// w' = √(1−β²) w + β Πξ,        ξ ~ N(0, I_P),  Π = I − BBᵀ
/// log α = Φ(z,w) − Φ(z,w') + ½‖w'‖² − ½‖w‖²
/// ```
///
/// `Πξ` is Gaussian on the complement with covariance `Π`, which restricted to
/// the `P−r` complement coordinates is the identity — so in those coordinates
/// this is the ordinary pCN kernel and `½‖w'‖² − ½‖w‖²` is its exact proposal
/// ratio.  Projecting the noise (rather than the proposal) is what keeps `w` in
/// the complement: `w` starts at zero and every subsequent `w` is a linear
/// combination of projected noise, so it never acquires a `B` component beyond
/// rounding (`~1e-7` after thousands of steps).  Projecting the *proposal*
/// instead would be a nonlinear map applied after the proposal and would break
/// the simple density ratio.
///
/// `β` is a single scalar: the complement is prior-dominated in the intended
/// use, where one isotropic step is the right proposal and a per-direction
/// scale would be fitting noise.  Adapting it on this stage's own acceptance is
/// sound because the stage is an exact kernel by itself.
pub fn mh_pcn_w_full(
    phi: &dyn FullPotential,
    z: &[f32],
    w: &mut [f32],
    phi_w: f64,
    beta: f64,
    rng: &mut impl Rng,
) -> WStep {
    assert_eq!(z.len(), phi.dim(), "z must have the informed dimension");
    assert_eq!(w.len(), phi.p(), "w must have the ambient dimension");
    assert!(beta > 0.0 && beta <= 1.0, "beta must be in (0, 1]");
    let shrink = (1.0 - beta * beta).sqrt();
    let noise: Vec<f32> = (0..w.len()).map(|_| normal(rng) as f32).collect();
    let mut projected = vec![0.0f32; w.len()];
    phi.complement(&noise, &mut projected);
    let proposal: Vec<f32> = w
        .iter()
        .zip(&projected)
        .map(|(&wi, &xi)| (shrink * f64::from(wi) + beta * f64::from(xi)) as f32)
        .collect();
    let phi_new = phi.phi(z, &proposal);
    let log_q_ratio = prior_phi(&proposal) - prior_phi(w);
    let log_alpha = phi_w - phi_new + log_q_ratio;
    let u: f64 = rng.random();
    let accept = log_alpha.is_finite() && log_alpha >= u.ln();
    if accept {
        w.copy_from_slice(&proposal);
        WStep {
            accepted: true,
            phi: phi_new,
            proposed: phi_new,
        }
    } else {
        WStep {
            accepted: false,
            phi: phi_w,
            proposed: phi_new,
        }
    }
}

/// Exact MH-pCN in `z` with a **rotated** per-direction proposal scale.
///
/// ```text
/// w = Q z,   w'_i = √(1−β_i²) w_i + β_i ξ_i,   z' = Qᵀ w'
/// log α = min(0, [Φ(z) − Φ(z')] + ½‖z'‖² − ½‖z‖²)
/// ```
///
/// `rotation` is the orthogonal matrix `V` in row-major order, with column `i`
/// the eigenvector of `betas[i]`; `Q = Vᵀ`.
///
/// This is the diagonal kernel of [`mh_pcn_z_scaled`] applied in a rotated
/// basis, so it is exact for exactly the same reason.  `Q` is a deterministic
/// bijection and an orthogonal matrix leaves the `N(0, I)` prior invariant
/// (`QᵀQ = I`), so the proposal's density ratio is unchanged; a proof of the
/// ratio itself is in that function's documentation.  Equivalently, this is
/// full-covariance preconditioning with `S = Qᵀ diag(β_i²) Q`: the proposal is
/// `z' = (I−S)^{1/2} z + S^{1/2} ξ`, and `(I−S) + S = I` is what preserves the
/// prior.  A *block* preconditioner is the special case where `Q` is the
/// identity outside the block, i.e. only some directions are mixed.
///
/// Why a rotation at all: a diagonal scale can only stretch along the
/// coordinates.  When the stiff direction of a second-moment estimate is a
/// combination (the informed basis is built from a fine-tuning path, so it is
/// not the target's eigenbasis), every diagonal `H_ii` sees a mixture and no
/// per-coordinate `β_i` is the right width for the direction that actually
/// needs the small step.  Rotating into the estimate's eigenbasis lets those
/// directions be rescaled jointly.
///
/// The rotation is checked for orthogonality here as well as in [`RotScale`]:
/// a non-orthogonal `Q` would silently change the target, which is the one
/// failure this module exists to prevent.  The check is `O(r³)` per step,
/// against one full-data forward pass per step, so it is free in practice.
pub fn mh_pcn_z_rotated(
    phi: &dyn ZPotential,
    z: &mut [f32],
    phi_z: f64,
    rotation: &[f64],
    betas: &[f64],
    rng: &mut impl Rng,
) -> ZStep {
    let r = z.len();
    assert_eq!(r, phi.dim(), "z must have the target's dimension");
    assert_eq!(rotation.len(), r * r, "rotation must be r x r");
    assert_eq!(betas.len(), r, "one beta per eigen-direction");
    assert!(
        betas.iter().all(|&b| b > 0.0 && b <= 1.0),
        "every beta must be in (0, 1]"
    );
    assert!(
        is_orthogonal(rotation, r),
        "rotation must be orthogonal to preserve the N(0, I) prior"
    );
    // w = Q z with Q = Vᵀ, i.e. w_i = Σ_j V[j][i] z[j].
    let mut w: Vec<f64> = (0..r)
        .map(|i| (0..r).map(|j| rotation[j * r + i] * f64::from(z[j])).sum())
        .collect();
    for (wi, &b) in w.iter_mut().zip(betas) {
        *wi = (1.0 - b * b).sqrt() * *wi + b * normal(rng);
    }
    // z' = Qᵀ w = V w, i.e. z'_i = Σ_j V[i][j] w[j].
    let proposal: Vec<f32> = (0..r)
        .map(|i| (0..r).map(|j| rotation[i * r + j] * w[j]).sum::<f64>() as f32)
        .collect();
    let phi_new = phi.phi(&proposal);
    // The prior term is the same as the diagonal kernel's: ‖z‖ is invariant
    // under Q, so ½‖z'‖² − ½‖z‖² = ½‖w'‖² − ½‖w‖².
    let log_q_ratio = prior_phi(&proposal) - prior_phi(z);
    let log_alpha = phi_z - phi_new + log_q_ratio;
    let u: f64 = rng.random();
    let accept = log_alpha.is_finite() && log_alpha >= u.ln();
    if accept {
        z.copy_from_slice(&proposal);
        ZStep {
            accepted: true,
            phi: phi_new,
            proposed: phi_new,
        }
    } else {
        ZStep {
            accepted: false,
            phi: phi_z,
            proposed: phi_new,
        }
    }
}

/// Diagonal of the potential's Hessian at `z`, by central second differences
/// with step `h`.
///
/// `phi` is the full potential (prior included), so entry `i` is the target's
/// precision along direction `i`, `1 + λN·∂²ℓ/∂z_i²`, and `1/√H_ii` is that
/// direction's posterior width in units of the prior's.  Two forward passes per
/// coordinate is the price of not assuming a curvature model; on a restricted
/// rank this is `2r` evaluations against thousands of chain steps, and the
/// report quotes the step so a reader can judge the truncation.
///
/// It is a *local* quantity at the point it is called on and a *diagonal* one:
/// it cannot see rotation between directions, and a target whose curvature
/// changes with `z` is only approximated.  That is a limitation of the
/// preconditioner, not of the kernel — the kernel stays exact for any `β_i`.
pub fn curvature_diag(phi: &dyn ZPotential, z: &[f32], h: f64) -> Vec<f64> {
    assert!(h > 0.0, "finite-difference step must be positive");
    let base = phi.phi(z);
    let mut out = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let mut plus = z.to_vec();
        let mut minus = z.to_vec();
        plus[i] += h as f32;
        minus[i] -= h as f32;
        let fp = phi.phi(&plus);
        let fm = phi.phi(&minus);
        out.push((fp - 2.0 * base + fm) / (h * h));
    }
    out
}

/// The full Hessian of the potential at `z`, by central second differences with
/// step `h`, row-major (`out[i * r + j] = ∂²Φ/∂z_i∂z_j`).
///
/// The diagonal costs `2r` evaluations; the off-diagonal entries use the
/// four-point stencil and cost `2r(r−1)` more, `2r² + 1` in total.  At the ranks
/// this project runs (`r ≤ 8`) that is at most 129 forward passes, against
/// `steps ≈ 4000` per chain — the same trade as [`curvature_diag`], made once
/// outside the chain.  It is a *local* quantity at the point it is called on;
/// using it as a posterior precision is only as good as that point.
///
/// The result is symmetrised, and includes the prior's unit diagonal because
/// `phi` is the full potential.
pub fn curvature_hessian(phi: &dyn ZPotential, z: &[f32], h: f64) -> Vec<f64> {
    assert!(h > 0.0, "finite-difference step must be positive");
    let r = z.len();
    let base = phi.phi(z);
    let mut out = vec![0.0f64; r * r];
    for i in 0..r {
        for j in i..r {
            let value = if i == j {
                let mut plus = z.to_vec();
                let mut minus = z.to_vec();
                plus[i] += h as f32;
                minus[i] -= h as f32;
                (phi.phi(&plus) - 2.0 * base + phi.phi(&minus)) / (h * h)
            } else {
                let mut pp = z.to_vec();
                let mut pm = z.to_vec();
                let mut mp = z.to_vec();
                let mut mm = z.to_vec();
                pp[i] += h as f32;
                pp[j] += h as f32;
                pm[i] += h as f32;
                pm[j] -= h as f32;
                mp[i] -= h as f32;
                mp[j] += h as f32;
                mm[i] -= h as f32;
                mm[j] -= h as f32;
                (phi.phi(&pp) - phi.phi(&pm) - phi.phi(&mp) + phi.phi(&mm)) / (4.0 * h * h)
            };
            out[i * r + j] = value;
            out[j * r + i] = value;
        }
    }
    out
}

/// Per-direction pCN scales derived from a curvature diagonal, with one global
/// multiplier adapted during warm-up.
///
/// `width_i = 1/√H_ii` is the direction's posterior width; `β_i = m·width_i`
/// starts at the width itself (`m = 1`) and the multiplier follows a
/// Robbins–Monro rule on the *joint* acceptance rate, so the shape of the
/// proposal comes from the target and only its overall size comes from
/// feedback.  The multiplier must be frozen before the kept window: an adapted
/// kernel is not a fixed kernel, and its draws are only a sample of `π` when the
/// adaptation has stopped.
#[derive(Clone, Debug)]
pub struct DiagScale {
    width: Vec<f64>,
    multiplier: f64,
    /// Acceptance the multiplier is steered to.  0.5 is the usual pCN choice
    /// for a target not far from the prior; it is a tuning target here, not a
    /// claim of optimality.
    pub target_accept: f64,
}

impl DiagScale {
    /// Widths from a curvature diagonal: `1/√H_ii`.
    ///
    /// This is only as good as the point it is evaluated at.  On a *stationary*
    /// anchor (`θ_ref` is a pre-train endpoint, so `∇ℓ ≈ 0` there) the curvature
    /// describes the local bowl, not the posterior: in a direction along which
    /// the likelihood minimum is far from the anchor, small curvature at the
    /// anchor means "the slope is slowly changing here", not "the posterior is
    /// wide".  Measured (`docs/PCNL_STATUS.md` §R2): rank-1 informed with
    /// curvature widths gave 0.1 % acceptance and ESS 7, because `1/√H` came out
    /// at least 50× the prior width, every `β_i` was clamped at the `β ≤ 1` cap,
    /// and each proposal landed where the data has no mass.  Use
    /// [`DiagScale::from_widths`] with a pilot chain's spread instead when the
    /// mode is displaced — or, better, both, and let the report show which is
    /// which.
    pub fn from_curvature(h: &[f64]) -> Self {
        let w: Vec<f64> = h.iter().map(|&x| 1.0 / x.max(1e-8).sqrt()).collect();
        Self::from_widths(&w)
    }

    /// Widths measured, not assumed: per-coordinate spread of a pilot chain.
    /// A pilot cannot be wider than the region it visited, so this is a lower
    /// bound on the posterior's width and `m` is free to grow; the estimate is
    /// biased *down*, which is the safe direction for acceptance.
    pub fn from_widths(widths: &[f64]) -> Self {
        Self {
            width: widths.iter().map(|&w| w.max(1e-3)).collect(),
            multiplier: 1.0,
            target_accept: 0.5,
        }
    }

    /// The same measured widths with their *log*-spread shrunk toward the
    /// geometric mean: `log w_i ← mean(log w) + s·(log w_i − mean(log w))`.
    ///
    /// The pilot splits `150·pilot/300` draws over `r` coordinates, so at rank 8
    /// each width is a 19-draw standard deviation and the spread across `i` is
    /// mostly estimation noise.  Shrinking shrinks the *differences* between
    /// coordinates while keeping the overall scale (the geometric mean is
    /// preserved exactly), so `s = 1` is [`DiagScale::from_widths`] and `s = 0`
    /// is one isotropic width at the pilot's geometric mean.  This is a one-
    /// parameter interpolation for measuring how much of the mixing gain needs
    /// per-coordinate widths at all; the kernel is unchanged, so exactness for
    /// every `s` follows from `β_i`-reversibility, not from this estimator.
    pub fn from_widths_shrunk(widths: &[f64], shrink: f64) -> Self {
        let base = Self::from_widths(widths);
        let s = if shrink.is_finite() {
            shrink.clamp(0.0, 1.0)
        } else {
            1.0
        };
        if s >= 1.0 || base.width.is_empty() {
            return base;
        }
        let n = base.width.len() as f64;
        let log_mean = base.width.iter().map(|w| w.ln()).sum::<f64>() / n;
        Self {
            width: base
                .width
                .iter()
                .map(|w| (log_mean + s * (w.ln() - log_mean)).exp())
                .collect(),
            multiplier: 1.0,
            target_accept: 0.5,
        }
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// The proposal scales, clipped to `[1e-4, 1]`.  `β ≤ 1` is pCN's limit
    /// (beyond it the shrink factor is imaginary); the lower clip only stops a
    /// numerical zero from freezing a coordinate outright.
    pub fn betas(&self) -> Vec<f64> {
        self.width
            .iter()
            .map(|&s| (self.multiplier * s).clamp(1e-4, 1.0))
            .collect()
    }

    /// Robbins–Monro on `log m`, with a decaying step so the multiplier
    /// converges instead of oscillating.  `t` counts adaptation steps from 0.
    pub fn adapt(&mut self, accepted: bool, t: usize) {
        self.multiplier = adapt_log_multiplier(self.multiplier, accepted, t, self.target_accept);
    }
}

/// One Robbins–Monro update of a **scalar** pCN scale, clipped to `(0, 1]`.
/// Same feedback and rate as the per-direction multipliers, for a proposal with
/// a single `β` such as the full-target complement stage.
pub fn adapt_beta(beta: f64, accepted: bool, t: usize, target: f64) -> f64 {
    adapt_log_multiplier(beta, accepted, t, target).clamp(1e-4, 1.0)
}

/// One Robbins–Monro update of `log m`; shared by the diagonal and rotated
/// scales so their warm-up behaves identically.
fn adapt_log_multiplier(m: f64, accepted: bool, t: usize, target: f64) -> f64 {
    let step = 0.3 / (1.0 + t as f64).powf(0.6);
    let a = f64::from(u8::from(accepted));
    (m.ln() + step * (a - target)).clamp(-7.0, 2.0).exp()
}

/// Why a symmetric eigendecomposition failed.
#[derive(Clone, Debug, PartialEq)]
pub enum EigenError {
    BadLength { expected: usize, got: usize },
    NotFinite,
    NotSymmetric { row: usize, col: usize },
    NoConvergence { sweeps: usize },
}

impl std::fmt::Display for EigenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigenError::BadLength { expected, got } => {
                write!(f, "expected {expected} entries, got {got}")
            }
            EigenError::NotFinite => write!(f, "matrix contains a non-finite value"),
            EigenError::NotSymmetric { row, col } => write!(
                f,
                "entry ({row}, {col}) differs from its transpose: not symmetric"
            ),
            EigenError::NoConvergence { sweeps } => {
                write!(f, "Jacobi sweeps did not converge in {sweeps}")
            }
        }
    }
}

impl std::error::Error for EigenError {}

/// Eigen-decomposition of a symmetric `r x r` matrix by cyclic Jacobi rotations.
///
/// Returns `(eigenvalues ascending, eigenvectors)`, the eigenvectors row-major
/// with column `i` belonging to `eigenvalues[i]`.  Cyclic Jacobi is used rather
/// than a general eigensolver because `r` is the subspace rank (at most 8 here):
/// it is a few hundred flops, needs no dependency, and uses only orthogonal
/// similarity transforms, so the returned vectors are orthogonal to working
/// precision instead of merely "eigenvectors as computed".
///
/// A non-symmetric input is rejected rather than silently symmetrised: the
/// callers use this to build a *change of basis*, and a matrix that is not the
/// one they think it is would produce a plausible rotation of the wrong target.
pub fn symmetric_eigen(a: &[f64], r: usize) -> Result<(Vec<f64>, Vec<f64>), EigenError> {
    if r == 0 || a.len() != r * r {
        return Err(EigenError::BadLength {
            expected: r * r,
            got: a.len(),
        });
    }
    if a.iter().any(|x| !x.is_finite()) {
        return Err(EigenError::NotFinite);
    }
    let scale = a.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1.0);
    for i in 0..r {
        for j in 0..i {
            if (a[i * r + j] - a[j * r + i]).abs() > 1e-10 * scale {
                return Err(EigenError::NotSymmetric { row: i, col: j });
            }
        }
    }
    // Symmetric part: removes the last bits of asymmetry rather than trusting
    // that the caller's matrix was exactly symmetric.
    let mut m: Vec<f64> = (0..r * r)
        .map(|k| 0.5 * (a[k] + a[(k % r) * r + k / r]))
        .collect();
    let mut v = vec![0.0f64; r * r];
    for i in 0..r {
        v[i * r + i] = 1.0;
    }
    const SWEEPS: usize = 64;
    let mut converged = false;
    for _ in 0..SWEEPS {
        let mut off = 0.0f64;
        for i in 0..r {
            for j in 0..i {
                off = off.max(m[i * r + j].abs());
            }
        }
        if off <= 1e-13 * scale {
            converged = true;
            break;
        }
        for p in 0..r {
            for q in (p + 1)..r {
                let apq = m[p * r + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q * r + q] - m[p * r + p]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // m <- Jᵀ m J, two one-sided passes; v <- v J.
                for k in 0..r {
                    let (mkp, mkq) = (m[k * r + p], m[k * r + q]);
                    m[k * r + p] = c * mkp - s * mkq;
                    m[k * r + q] = s * mkp + c * mkq;
                }
                for k in 0..r {
                    let (mpk, mqk) = (m[p * r + k], m[q * r + k]);
                    m[p * r + k] = c * mpk - s * mqk;
                    m[q * r + k] = s * mpk + c * mqk;
                }
                m[p * r + q] = 0.0;
                m[q * r + p] = 0.0;
                for k in 0..r {
                    let (vkp, vkq) = (v[k * r + p], v[k * r + q]);
                    v[k * r + p] = c * vkp - s * vkq;
                    v[k * r + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut off = 0.0f64;
    for i in 0..r {
        for j in 0..i {
            off = off.max(m[i * r + j].abs());
        }
    }
    if !converged && off > 1e-9 * scale {
        return Err(EigenError::NoConvergence { sweeps: SWEEPS });
    }
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&i, &j| {
        m[i * r + i]
            .partial_cmp(&m[j * r + j])
            .expect("entries are finite")
    });
    let values = order.iter().map(|&i| m[i * r + i]).collect();
    let mut vectors = vec![0.0f64; r * r];
    for (new, &old) in order.iter().enumerate() {
        for i in 0..r {
            vectors[i * r + new] = v[i * r + old];
        }
    }
    Ok((values, vectors))
}

/// `max |VᵀV − I| ≤ 1e-6`: the preconditioner's `Q` must be orthogonal, and
/// this is the check the kernel runs before using it.
fn is_orthogonal(q: &[f64], r: usize) -> bool {
    for i in 0..r {
        for j in 0..r {
            let dot: f64 = (0..r).map(|k| q[k * r + i] * q[k * r + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            if (dot - want).abs() > 1e-6 {
                return false;
            }
        }
    }
    true
}

/// A **rotated** per-direction pCN scale: the diagonal proposal applied in the
/// eigenbasis of a local second-moment estimate.
///
/// `width_i = 1/√λ_i` when the estimate is a Hessian (a precision) and
/// `√λ_i` when it is a covariance, exactly as [`DiagScale::from_curvature`] and
/// [`DiagScale::from_widths`] do one coordinate at a time.  The difference is
/// that the `β_i` now apply to the estimate's *eigen-directions*, so a stiff
/// direction that is a combination of the informed coordinates gets its own
/// step instead of every coordinate paying for the mixture.  The multiplier `m`
/// and its Robbins–Monro warm-up are identical to `DiagScale`'s, and the same
/// rule applies: freeze it before the kept window, because an adapted kernel is
/// not a fixed kernel.
///
/// The caveat of the diagonal arms carries over unchanged — `1/√H` at a
/// stationary anchor describes the local bowl, not a displaced mode — and the
/// eigensolver adds one of its own: a near-degenerate eigenvalue makes the
/// corresponding eigenvector unstable, so a rotation estimated from a noisy
/// covariance should be read as a conditioning aid, not as a reported quantity.
/// It never changes the target: the kernel is exact for any orthogonal `Q` and
/// any `β_i`.
#[derive(Clone, Debug)]
pub struct RotScale {
    /// `r x r`, row-major; column `i` is the eigenvector of `width[i]`.
    rotation: Vec<f64>,
    width: Vec<f64>,
    multiplier: f64,
    /// Acceptance the multiplier is steered to; see [`DiagScale::target_accept`].
    pub target_accept: f64,
}

impl RotScale {
    /// Widths from a full Hessian: `1/√λ_i` in the Hessian's eigenbasis.
    pub fn from_hessian(h: &[f64], r: usize) -> Result<Self, EigenError> {
        let (values, rotation) = symmetric_eigen(h, r)?;
        let width = values.iter().map(|&l| 1.0 / l.max(1e-8).sqrt()).collect();
        Ok(Self {
            rotation,
            width,
            multiplier: 1.0,
            target_accept: 0.5,
        })
    }

    /// Widths from a full covariance (measured, e.g. by a pilot chain):
    /// `√λ_i` in its eigenbasis.  A measured covariance cannot be wider than the
    /// region it sampled, so the widths are biased down — the safe direction for
    /// acceptance, as with [`DiagScale::from_widths`].
    pub fn from_covariance(c: &[f64], r: usize) -> Result<Self, EigenError> {
        let (values, rotation) = symmetric_eigen(c, r)?;
        let width = values
            .iter()
            .map(|&l| l.max(0.0).sqrt().max(1e-3))
            .collect();
        Ok(Self {
            rotation,
            width,
            multiplier: 1.0,
            target_accept: 0.5,
        })
    }

    /// The orthogonal matrix `V` (row-major, column `i` = eigenvector `i`);
    /// `Q = Vᵀ`.  Pass this to [`mh_pcn_z_rotated`].
    pub fn rotation(&self) -> &[f64] {
        &self.rotation
    }

    /// Eigen-widths before the multiplier: the estimate's marginal scales in
    /// its own basis, for reports that want to see them.
    pub fn widths(&self) -> &[f64] {
        &self.width
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Per-eigen-direction proposal scales, clipped to `[1e-4, 1]`.
    pub fn betas(&self) -> Vec<f64> {
        self.width
            .iter()
            .map(|&s| (self.multiplier * s).clamp(1e-4, 1.0))
            .collect()
    }

    /// Robbins–Monro on `log m`; same rule as [`DiagScale::adapt`].
    pub fn adapt(&mut self, accepted: bool, t: usize) {
        self.multiplier = adapt_log_multiplier(self.multiplier, accepted, t, self.target_accept);
    }
}

fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// R1's controlled validation target: `Φ(z) = a·z₀²/2 + c·(z₁² − m²)²`.
///
/// Separable by construction, so the exact moments come from 1D quadrature on
/// this side of the test instead of from an assumption: the first coordinate is
/// `N(0, 1/a)`, and the second is the double well whose density is
/// `∝ exp(−c(z²−m²)²)`.  It is a rank-2 target (dimension 2) with a
/// non-Gaussian coordinate, which is exactly what a proposal's exactness has to
/// survive: for a Gaussian target even a wrong ratio can look right.
#[derive(Clone, Copy, Debug)]
pub struct QuarticTarget {
    pub a: f64,
    pub c: f64,
    pub m: f64,
}

impl QuarticTarget {
    pub const VALIDATION: QuarticTarget = QuarticTarget {
        a: 1.0,
        c: 2.0,
        m: 1.0,
    };

    /// `(mean, second moment)` of `z₁` under `∝ exp(−c(z²−m²)²)`, by
    /// composite Simpson on `[−L, L]`.
    pub fn quartic_moments(&self, l: f64, n: usize) -> (f64, f64) {
        assert!(
            n.is_multiple_of(2) && n >= 2,
            "Simpson needs an even number of steps"
        );
        let h = 2.0 * l / n as f64;
        let density = |z: f64| (-self.c * (z * z - self.m * self.m).powi(2)).exp();
        let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
        for k in 0..=n {
            let z = -l + k as f64 * h;
            let w = if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let d = w * density(z);
            m0 += d;
            m1 += d * z;
            m2 += d * z * z;
        }
        (m1 / m0, m2 / m0)
    }
}

impl ZPotential for QuarticTarget {
    fn dim(&self) -> usize {
        2
    }

    fn phi(&self, z: &[f32]) -> f64 {
        0.5 * self.a * f64::from(z[0]) * f64::from(z[0])
            + self.c * (f64::from(z[1]) * f64::from(z[1]) - self.m * self.m).powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::ess_bulk;
    use rand::SeedableRng;

    /// The kernel's invariant law on a target whose second moment is known by
    /// quadrature, not by construction of the chain.  Ranks 1 and 2 are the
    /// plan's validation ranks; `QuarticTarget` is rank 2 and its first
    /// coordinate is the rank-1 Gaussian control.
    #[test]
    fn mh_pcn_z_matches_quadrature_on_rank_two() {
        let target = QuarticTarget::VALIDATION;
        let (_, m2_exact) = target.quartic_moments(8.0, 20_000);
        let (mean1_exact, _) = target.quartic_moments(8.0, 20_000);
        assert!(mean1_exact.abs() < 1e-9, "well is symmetric");

        let beta = 0.6;
        let steps = 120_000;
        let burn = 10_000;
        let thin = 10;
        let mut chains: Vec<Vec<f64>> = Vec::new();
        let mut acc = 0usize;
        let mut total = 0usize;
        for seed in 0..4u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(11 + seed);
            let mut z = [0.5f32, -0.5];
            let mut phi = target.phi(&z);
            let mut kept = Vec::new();
            for t in 0..steps {
                let st = mh_pcn_z(&target, &mut z, phi, beta, &mut rng);
                phi = st.phi;
                if t >= burn {
                    total += 1;
                    acc += usize::from(st.accepted);
                    if (t - burn) % thin == 0 {
                        kept.push(f64::from(z[1]));
                    }
                }
            }
            chains.push(kept);
        }
        let all: Vec<f64> = chains.iter().flatten().copied().collect();
        let mean = all.iter().sum::<f64>() / all.len() as f64;
        let var = all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (all.len() - 1) as f64;
        let ess = ess_bulk(&chains);
        assert!(ess > 500.0, "ESS too low to test anything: {ess}");
        let se_mean = (var / ess).sqrt();
        let se_m2 = (var * 2.0 / ess).sqrt(); // δ-method on the second moment
        let m2 = all.iter().map(|x| x * x).sum::<f64>() / all.len() as f64;
        assert!(
            (mean - mean1_exact).abs() < 5.0 * se_mean,
            "z1 mean {mean} vs exact {mean1_exact} (se {se_mean})"
        );
        assert!(
            (m2 - m2_exact).abs() < 5.0 * se_m2,
            "E[z1^2] {m2} vs exact {m2_exact} (se {se_m2})"
        );

        // The bimodality must survive: a chain that collapsed to one well would
        // still match the mean (symmetry) but not the second moment, which is
        // why both are checked.  A hint for readers of failures:
        let rate = acc as f64 / total as f64;
        // Visible with `--nocapture`; the doc quotes these numbers.
        println!(
            "pcn_z rank-2: accept {:.3}, mean(z1) {mean:+.4} (exact {mean1_exact:+.4}), \
             E[z1^2] {m2:.4} (exact {m2_exact:.4}), ESS {ess:.0} of {}",
            rate,
            all.len()
        );
        assert!(rate > 0.2 && rate < 0.99, "accept rate {rate}");
    }

    /// Shrinkage is a monotone interpolation that leaves the *overall* scale
    /// alone: `s = 1` reproduces the measured widths bitwise, `s = 0` collapses
    /// every coordinate onto their geometric mean, and the geometric mean is
    /// invariant in between.  A silently reversed or non-monotone knob would
    /// make the rank-4/8 rows unreadable, so pin the endpoints and one interior
    /// value.
    #[test]
    fn width_shrinkage_keeps_scale_and_moves_monotonically() {
        let widths = [0.02, 0.05, 0.13, 0.4];
        let raw = DiagScale::from_widths(&widths);
        let same = DiagScale::from_widths_shrunk(&widths, 1.0);
        assert_eq!(raw.width, same.width, "s = 1 must be the raw estimate");

        let full = DiagScale::from_widths_shrunk(&widths, 0.0);
        let gmean = widths.iter().map(|w| w.ln()).sum::<f64>() / widths.len() as f64;
        let gmean = gmean.exp();
        for w in &full.width {
            assert!(
                (w - gmean).abs() < 1e-12,
                "s = 0 must be isotropic at {gmean}"
            );
        }

        for s in [0.25, 0.5, 0.75] {
            let shrunk = DiagScale::from_widths_shrunk(&widths, s);
            // Overall scale preserved…
            let g = (shrunk.width.iter().map(|w| w.ln()).sum::<f64>() / shrunk.width.len() as f64)
                .exp();
            assert!(
                (g - gmean).abs() < 1e-12,
                "s = {s} moved the geometric mean"
            );
            // …and the spread decreases with more shrinkage.
            for (i, w) in widths.iter().enumerate() {
                let expected = gmean * (w / gmean).powf(s);
                assert!((shrunk.width[i] - expected).abs() < 1e-12);
            }
        }
    }

    /// The proposal's prior reversibility, isolated: `N(0, 1)` must be invariant
    /// under `z' = √(1−β²)z + βξ` with the ratio of a Gaussian potential.  If
    /// the shrink factor is dropped, the variance inflates and this fails.
    #[test]
    fn pcn_keeps_the_unit_gaussian_variance() {
        struct Gauss;
        impl ZPotential for Gauss {
            fn dim(&self) -> usize {
                1
            }
            fn phi(&self, z: &[f32]) -> f64 {
                0.5 * f64::from(z[0]) * f64::from(z[0])
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut z = [0.0f32];
        let mut phi = 0.0;
        let mut kept = Vec::new();
        for t in 0..60_000 {
            let st = mh_pcn_z(&Gauss, &mut z, phi, 0.7, &mut rng);
            phi = st.phi;
            if t >= 1_000 {
                kept.push(f64::from(z[0]));
            }
        }
        let mean = kept.iter().sum::<f64>() / kept.len() as f64;
        let var = kept.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / kept.len() as f64;
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }

    /// The likelihood term really is the *sum* over the training set: getting
    /// the `N` factor wrong by using the mean silently changes the target's
    /// temperature by a factor of `N` (prior-dominated instead of
    /// data-determined), which is invisible in an accuracy column and obvious
    /// here.
    #[test]
    fn subspace_target_uses_the_sum_form() {
        use crate::model::{Shapes, Variant, init_flat};
        use crate::subspace::{AffineSubspace, Basis};
        use crate::task::{Dataset, Task};

        let shapes = Shapes {
            vocab: 6,
            d: 4,
            ff: 8,
            layers: 1,
        };
        let p = shapes.flat_size();
        let data = Dataset::new(Task::A, 5, 16, 8, 1);
        let anchor = init_flat(&shapes, 0.1, 2)
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        let basis = Basis::random(p, 1, 1).expect("basis");
        let sub = AffineSubspace::isotropic(anchor, basis, 0.35).expect("subspace");
        let lambda = 0.5;
        let target =
            SubspaceTarget::new(sub.clone(), lambda, data.clone(), shapes, Variant::Cosine);
        assert_eq!(target.n_train(), 16);
        let z = [0.3f32];
        let (lik, prior) = target.phi_parts(&z);
        let theta = sub.lift_tensor(&z).expect("theta");
        let mean = data
            .train_loss(&theta, &shapes, Variant::Cosine)
            .expect("loss");
        assert!((lik - lambda * 16.0 * mean).abs() < 1e-6 * lik.abs().max(1.0));
        assert!((prior - 0.045).abs() < 1e-6, "prior {prior}");
    }

    /// `grad_phi` must be the gradient of `phi`, or a mode search on it finds a
    /// point that is not the target's mode and every width derived from that
    /// point is a wrong-but-plausible number.  Central differences at two
    /// distinct `z`, with the prior's contribution present in both.
    #[test]
    fn subspace_target_gradient_matches_finite_differences() {
        use crate::model::{Shapes, Variant, init_flat};
        use crate::subspace::{AffineSubspace, Basis};
        use crate::task::{Dataset, Task};

        let shapes = Shapes {
            vocab: 6,
            d: 4,
            ff: 8,
            layers: 1,
        };
        let p = shapes.flat_size();
        let data = Dataset::new(Task::A, 5, 32, 8, 1);
        let anchor = init_flat(&shapes, 0.1, 2)
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        let basis = Basis::random(p, 2, 3).expect("basis");
        let sub = AffineSubspace::isotropic(anchor, basis, 0.35).expect("subspace");
        let target = SubspaceTarget::new(sub, 0.5, data, shapes, Variant::Cosine);
        for z0 in [[0.3f32, -0.2], [-0.7, 0.9]] {
            let g = target.grad_phi(&z0).expect("gradient");
            let h = 1e-3f32;
            for i in 0..2 {
                let mut plus = z0;
                let mut minus = z0;
                plus[i] += h;
                minus[i] -= h;
                let fd = (target.phi(&plus) - target.phi(&minus)) / (2.0 * f64::from(h));
                let got = f64::from(g[i]);
                assert!(
                    (got - fd).abs() <= 1e-2 * fd.abs().max(1.0),
                    "z = {z0:?}, coord {i}: analytic {got}, finite difference {fd}"
                );
            }
        }
    }

    /// A diagonal preconditioner must be *exactly* the uniform kernel when the
    /// scales coincide: same proposal, same ratio, same uniform draw, so a
    /// divergence here is a bug in the ratio rather than in the statistics.
    /// Bitwise equality is deliberately required — the two functions draw their
    /// normals and the accept uniform in the same order.
    #[test]
    fn scaled_pcn_equals_the_uniform_kernel_at_equal_betas() {
        let target = QuarticTarget::VALIDATION;
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(7);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(7);
        let mut za = [0.5f32, -0.5];
        let mut zb = za;
        let (mut pa, mut pb) = (target.phi(&za), target.phi(&zb));
        let mut accepts = 0;
        for _ in 0..2_000 {
            let a = mh_pcn_z(&target, &mut za, pa, 0.6, &mut rng_a);
            let b = mh_pcn_z_scaled(&target, &mut zb, pb, &[0.6, 0.6], &mut rng_b);
            assert_eq!(a.accepted, b.accepted);
            assert_eq!(za, zb);
            assert_eq!(a.phi.to_bits(), b.phi.to_bits());
            pa = a.phi;
            pb = b.phi;
            accepts += usize::from(a.accepted);
        }
        assert!(accepts > 100, "degenerate run: {accepts} accepts");
    }

    /// The reason for per-direction scales: a target whose two coordinates
    /// differ in width by 10×.  One scalar β has to be small enough for the
    /// narrow coordinate, so the wide one barely moves; a width-derived β per
    /// coordinate recovers both variances.  The kernel is exact for either —
    /// this test is about mixing, and it fails if the widths are ignored.
    #[test]
    fn scaled_pcn_recovers_anisotropic_widths() {
        struct Aniso;
        impl ZPotential for Aniso {
            fn dim(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32]) -> f64 {
                0.5 * 100.0 * f64::from(z[0]) * f64::from(z[0])
                    + 0.5 * f64::from(z[1]) * f64::from(z[1])
            }
        }
        let target = Aniso;
        let scale = DiagScale::from_curvature(&curvature_diag(&target, &[0.0, 0.0], 1e-3));
        let betas = scale.betas();
        assert!(
            (betas[0] - 0.1).abs() < 1e-6 && (betas[1] - 1.0).abs() < 1e-6,
            "widths should be (0.1, 1.0): {betas:?}"
        );
        let mut chains: Vec<Vec<f64>> = Vec::new();
        for seed in 0..2u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(31 + seed);
            let mut z = [0.2f32, -0.3];
            let mut phi = target.phi(&z);
            let mut kept = Vec::new();
            for t in 0..60_000 {
                let st = mh_pcn_z_scaled(&target, &mut z, phi, &betas, &mut rng);
                phi = st.phi;
                if t >= 2_000 && t % 2 == 0 {
                    kept.push(f64::from(z[0]));
                }
            }
            chains.push(kept);
        }
        let all: Vec<f64> = chains.iter().flatten().copied().collect();
        let mean = all.iter().sum::<f64>() / all.len() as f64;
        let var = all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all.len() as f64;
        let ess = ess_bulk(&chains);
        assert!(
            (var - 0.01).abs() < 0.0015,
            "narrow variance {var} (ESS {ess})"
        );
        assert!(ess > 2_000.0, "narrow ESS too low: {ess}");
    }

    /// A rotation of the identity must give the diagonal kernel bit for bit:
    /// same normals in the same order, same ratio, same accept draw.  If the
    /// two differ, the rotation bookkeeping is wrong, not the statistics.
    #[test]
    fn rotated_pcn_with_the_identity_rotation_equals_the_diagonal_kernel() {
        let target = QuarticTarget::VALIDATION;
        let identity = [1.0, 0.0, 0.0, 1.0];
        let betas = [0.6f64, 0.3];
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(17);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(17);
        let mut za = [0.5f32, -0.5];
        let mut zb = za;
        let (mut pa, mut pb) = (target.phi(&za), target.phi(&zb));
        let mut accepts = 0;
        for _ in 0..2_000 {
            let a = mh_pcn_z_scaled(&target, &mut za, pa, &betas, &mut rng_a);
            let b = mh_pcn_z_rotated(&target, &mut zb, pb, &identity, &betas, &mut rng_b);
            assert_eq!(a.accepted, b.accepted);
            assert_eq!(za, zb);
            assert_eq!(a.phi.to_bits(), b.phi.to_bits());
            pa = a.phi;
            pb = b.phi;
            accepts += usize::from(a.accepted);
        }
        assert!(accepts > 100, "degenerate run: {accepts} accepts");
    }

    /// The point of a *rotation*: a Gaussian whose stiff direction is a
    /// combination of the coordinates.  `H = [[50.5, 49.5], [49.5, 50.5]]` has
    /// eigenvalues 1 and 100 along `(1,−1)/√2` and `(1,1)/√2`, so no diagonal
    /// `β_i` can be the right width for both, and getting the width from `H_ii`
    /// makes every coordinate pay for the mixture.  The eigenbasis recovers both
    /// widths (1 and 0.1), and the chain must recover the full posterior
    /// covariance `H⁻¹`, correlation included — a rotation applied to the wrong
    /// basis would still be a valid kernel but would sample a different target.
    #[test]
    fn rotated_pcn_recovers_correlated_widths() {
        struct Corr;
        impl ZPotential for Corr {
            fn dim(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32]) -> f64 {
                let a = (f64::from(z[0]) + f64::from(z[1])) / std::f64::consts::SQRT_2;
                let b = (f64::from(z[0]) - f64::from(z[1])) / std::f64::consts::SQRT_2;
                0.5 * (100.0 * a * a + b * b)
            }
        }
        let target = Corr;
        let h = curvature_hessian(&target, &[0.0, 0.0], 1e-3);
        assert!(
            (h[0] - 50.5).abs() < 1e-2 && (h[1] - 49.5).abs() < 1e-2,
            "Hessian {h:?}"
        );
        let scale = RotScale::from_hessian(&h, 2).expect("symmetric Hessian");
        let betas = scale.betas();
        assert!(
            (betas[0] - 1.0).abs() < 1e-6 && (betas[1] - 0.1).abs() < 1e-6,
            "eigenvalues 1 and 100 should give widths 1 and 0.1: {betas:?}"
        );
        let rotation = scale.rotation().to_vec();
        let mut chains: Vec<Vec<f64>> = Vec::new();
        let mut both: Vec<(f64, f64)> = Vec::new();
        for seed in 0..2u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(41 + seed);
            let mut z = [0.3f32, 0.2];
            let mut phi = target.phi(&z);
            let mut kept = Vec::new();
            for t in 0..40_000 {
                let st = mh_pcn_z_rotated(&target, &mut z, phi, &rotation, &betas, &mut rng);
                phi = st.phi;
                if t >= 2_000 && t % 2 == 0 {
                    kept.push(f64::from(z[0]));
                    both.push((f64::from(z[0]), f64::from(z[1])));
                }
            }
            chains.push(kept);
        }
        // H⁻¹ = [[0.505, −0.495], [−0.495, 0.505]].
        let n = both.len() as f64;
        let (mut m0, mut m1) = (0.0, 0.0);
        for (a, b) in &both {
            m0 += a;
            m1 += b;
        }
        m0 /= n;
        m1 /= n;
        let (mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0);
        for (a, b) in &both {
            v0 += (a - m0).powi(2);
            v1 += (b - m1).powi(2);
            c01 += (a - m0) * (b - m1);
        }
        v0 /= n - 1.0;
        v1 /= n - 1.0;
        c01 /= n - 1.0;
        let ess = ess_bulk(&chains);
        println!(
            "rotated correlated: ESS {ess:.0}, Var(z0) {v0:.4}, Var(z1) {v1:.4}, Cov {c01:+.4}"
        );
        assert!(ess > 2_000.0, "ESS too low: {ess}");
        assert!((v0 - 0.505).abs() < 0.02, "Var(z0) {v0}");
        assert!((v1 - 0.505).abs() < 0.02, "Var(z1) {v1}");
        assert!((c01 + 0.495).abs() < 0.02, "Cov {c01}");
    }

    /// A known decomposition must come back: eigenvalues in ascending order and
    /// eigenvectors that actually diagonalise the input.  Built from Givens
    /// rotations so the expected answer is exact rather than another routine's
    /// output.
    #[test]
    fn symmetric_eigen_matches_a_known_decomposition() {
        let (c1, s1) = (0.6f64, 0.8f64);
        let (c2, s2) = (0.28f64, 0.96f64);
        let q1 = [[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]];
        let q2 = [[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]];
        let mut v = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                v[i][j] = (0..3).map(|k| q1[i][k] * q2[k][j]).sum();
            }
        }
        let lam = [0.25f64, 1.0, 100.0];
        let mut a = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = (0..3).map(|k| v[i][k] * lam[k] * v[j][k]).sum();
            }
        }
        let (values, vectors) = symmetric_eigen(&a, 3).expect("symmetric");
        for (got, want) in values.iter().zip(&lam) {
            assert!((got - want).abs() < 1e-9, "eigenvalues {values:?}");
        }
        assert!(is_orthogonal(&vectors, 3), "eigenvectors not orthogonal");
        for i in 0..3 {
            for j in 0..3 {
                let rebuilt: f64 = (0..3)
                    .map(|k| vectors[i * 3 + k] * values[k] * vectors[j * 3 + k])
                    .sum();
                assert!(
                    (rebuilt - a[i * 3 + j]).abs() < 1e-8,
                    "reconstruction at ({i},{j})"
                );
            }
        }
        // A non-symmetric input is rejected, not symmetrised.
        let mut bad = a;
        bad[1] += 0.5;
        assert!(matches!(
            symmetric_eigen(&bad, 3),
            Err(EigenError::NotSymmetric { .. })
        ));
    }

    /// The rotated multiplier follows the same feedback and clipping rules as
    /// the diagonal one.
    #[test]
    fn rotated_scale_adapts_and_clips() {
        // Covariance diag(4, 0.25); eigenvalues ascend, so widths (0.5, 2)
        // and the betas clip to (0.5, 1).
        let mut s = RotScale::from_covariance(&[4.0, 0.0, 0.0, 0.25], 2).expect("covariance");
        let b0 = s.betas();
        assert!(
            (b0[0] - 0.5).abs() < 1e-9 && (b0[1] - 1.0).abs() < 1e-9,
            "{b0:?}"
        );
        let before = s.multiplier();
        for t in 0..50 {
            s.adapt(true, t);
        }
        assert!(s.multiplier() > before, "accepts should raise m");
        for t in 0..5_000 {
            s.adapt(false, t);
        }
        assert!(s.betas().iter().all(|&b| (1e-4..=1.0).contains(&b)));
        // The scalar rule the full-target complement stage uses.
        assert!(adapt_beta(0.5, true, 0, 0.5) > 0.5);
        assert!(adapt_beta(0.5, false, 0, 0.5) < 0.5);
        assert_eq!(adapt_beta(1.0, true, 0, 0.5), 1.0);
    }

    /// `curvature_hessian` must return the full precision matrix, not just its
    /// diagonal: the off-diagonal entries are what a rotation uses.
    #[test]
    fn curvature_hessian_matches_a_gaussian_precision() {
        struct Corr;
        impl ZPotential for Corr {
            fn dim(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32]) -> f64 {
                let (x, y) = (f64::from(z[0]), f64::from(z[1]));
                0.5 * (50.5 * x * x + 2.0 * 49.5 * x * y + 50.5 * y * y)
            }
        }
        let h = curvature_hessian(&Corr, &[0.2, -0.3], 1e-3);
        assert!((h[0] - 50.5).abs() < 1e-2, "{h:?}");
        assert!((h[1] - 49.5).abs() < 1e-2, "{h:?}");
        assert!((h[2] - 49.5).abs() < 1e-2, "{h:?}");
        assert!((h[3] - 50.5).abs() < 1e-2, "{h:?}");
    }

    /// `curvature_diag` must return the precision it claims to: for a Gaussian
    /// potential `Φ = ½p z²` that is `p`, including the prior's unit
    /// contribution when `p` is the full potential.
    #[test]
    fn curvature_diag_matches_a_gaussian_precision() {
        struct Gauss2;
        impl ZPotential for Gauss2 {
            fn dim(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32]) -> f64 {
                0.5 * 100.0 * f64::from(z[0]) * f64::from(z[0])
                    + 0.5 * f64::from(z[1]) * f64::from(z[1])
            }
        }
        let h = curvature_diag(&Gauss2, &[0.3, -0.2], 1e-3);
        assert!((h[0] - 100.0).abs() < 0.01, "{h:?}");
        assert!((h[1] - 1.0).abs() < 0.01, "{h:?}");
        // A flat potential has zero curvature, so a caller that forgets the
        // prior sees a different number than `SubspaceTarget` (which includes
        // it): the convention is stated, not assumed.
        let flat = curvature_diag(
            &QuarticTarget {
                a: 0.0,
                c: 0.0,
                m: 1.0,
            },
            &[0.0, 0.0],
            1e-3,
        );
        assert!(flat[0].abs() < 1e-6 && flat[1].abs() < 1e-6, "{flat:?}");
    }

    /// The multiplier's feedback direction: accepts raise it, rejects lower it,
    /// and the scales it produces stay inside pCN's `(0, 1]`.
    #[test]
    fn diag_scale_adapts_and_clips() {
        let mut s = DiagScale::from_curvature(&[1e6, 0.0]);
        let b0 = s.betas();
        assert!(b0[0] > 1e-4 && b0[0] <= 1.0 && b0[1] == 1.0, "{b0:?}");
        let before = s.multiplier();
        for t in 0..50 {
            s.adapt(true, t);
        }
        assert!(s.multiplier() > before, "accepts should raise m");
        for t in 0..5_000 {
            s.adapt(false, t);
        }
        assert!(s.betas()[1] <= 1.0 && s.betas()[0] >= 1e-4);
    }

    /// A proposal with a non-finite potential must be rejected, not accepted by
    /// `min(0, NaN)`.  `SubspaceTarget` returns `+∞` off-support; here the
    /// standing state is finite and the proposal is always infinite.
    #[test]
    fn an_infinite_proposal_is_rejected() {
        struct Wall;
        impl ZPotential for Wall {
            fn dim(&self) -> usize {
                1
            }
            fn phi(&self, z: &[f32]) -> f64 {
                if z[0].abs() > 1e-6 {
                    f64::INFINITY
                } else {
                    0.0
                }
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut z = [0.0f32];
        let mut accepted = 0;
        for _ in 0..100 {
            if mh_pcn_z(&Wall, &mut z, 0.0, 0.5, &mut rng).accepted {
                accepted += 1;
            }
        }
        assert_eq!(accepted, 0, "infinite proposal was accepted");
        assert_eq!(z[0], 0.0);
    }

    /// A target that ignores `w` must give the split kernel's `z` move *bit for
    /// bit* the same chain as the restricted diagonal kernel: same normals in
    /// the same order, same ratio, same accept draw.  This is the split
    /// bookkeeping's version of the identity-rotation test.
    #[test]
    fn full_z_move_with_a_w_blind_target_equals_the_diagonal_kernel() {
        struct Blinded;
        impl ZPotential for Blinded {
            fn dim(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32]) -> f64 {
                QuarticTarget::VALIDATION.phi(z)
            }
        }
        impl FullPotential for Blinded {
            fn dim(&self) -> usize {
                2
            }
            fn p(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32], _w: &[f32]) -> f64 {
                QuarticTarget::VALIDATION.phi(z)
            }
            fn complement(&self, _v: &[f32], out: &mut [f32]) {
                out.fill(0.0);
            }
        }
        let target = Blinded;
        let betas = [0.6f64, 0.3];
        let w = [0.0f32, 0.0];
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(23);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(23);
        let mut za = [0.5f32, -0.5];
        let mut zb = za;
        let (mut pa, mut pb) = (
            ZPotential::phi(&target, &za),
            FullPotential::phi(&target, &zb, &w),
        );
        let mut accepts = 0;
        for _ in 0..2_000 {
            let a = mh_pcn_z_scaled(&target, &mut za, pa, &betas, &mut rng_a);
            let b = mh_pcn_z_full(&target, &mut zb, &w, pb, &betas, &mut rng_b);
            assert_eq!(a.accepted, b.accepted);
            assert_eq!(za, zb);
            assert_eq!(a.phi.to_bits(), b.phi.to_bits());
            pa = a.phi;
            pb = b.phi;
            accepts += usize::from(a.accepted);
        }
        assert!(accepts > 100, "degenerate run: {accepts} accepts");
    }

    /// The two-stage sweep must sample the **joint** law, not just any
    /// distribution on the split.  Take a 2-D Gaussian whose precision couples
    /// `z` and the single complement coordinate, `Φ = ½‖v‖² + ½κ(⟨v,u⟩)²` with
    /// `u = (1,1)/√2`, so the exact posterior is
    /// `Σ = (I + κuuᵀ)⁻¹ = [[0.6, −0.4], [−0.4, 0.6]]` at `κ = 4`.  A `w` move
    /// with a wrongly projected noise, or a `z` ratio that ignores the coupling,
    /// changes these entries.
    #[test]
    fn two_stage_full_kernel_samples_the_coupled_gaussian() {
        struct Coupled;
        impl FullPotential for Coupled {
            fn dim(&self) -> usize {
                1
            }
            fn p(&self) -> usize {
                2
            }
            fn phi(&self, z: &[f32], w: &[f32]) -> f64 {
                // The complement of `B = e0` is the *second* coordinate; `w[0]`
                // is identically zero and reading it would silently decouple
                // the two blocks (the first version of this test did).
                let v = [f64::from(z[0]), f64::from(w[1])];
                let u = (v[0] + v[1]) / std::f64::consts::SQRT_2;
                0.5 * (v[0] * v[0] + v[1] * v[1]) + 2.0 * u * u
            }
            fn complement(&self, v: &[f32], out: &mut [f32]) {
                // B = e0: keep the second coordinate.
                out[0] = 0.0;
                out[1] = v[1];
            }
        }
        let target = Coupled;
        let mut zs: Vec<Vec<f64>> = Vec::new();
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for seed in 0..2u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(77 + seed);
            let mut z = [0.4f32];
            let mut w = [0.0f32, -0.3];
            let mut phi = target.phi(&z, &w);
            let mut kept = Vec::new();
            for t in 0..60_000 {
                let st = mh_pcn_z_full(&target, &mut z, &w, phi, &[0.5], &mut rng);
                phi = st.phi;
                let sq = mh_pcn_w_full(&target, &z, &mut w, phi, 0.7, &mut rng);
                phi = sq.phi;
                if t >= 2_000 && t % 2 == 0 {
                    kept.push(f64::from(z[0]));
                    pairs.push((f64::from(z[0]), f64::from(w[1])));
                }
            }
            zs.push(kept);
        }
        let n = pairs.len() as f64;
        let m0 = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let m1 = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let (mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0);
        for (a, b) in &pairs {
            v0 += (a - m0).powi(2);
            v1 += (b - m1).powi(2);
            c01 += (a - m0) * (b - m1);
        }
        v0 /= n - 1.0;
        v1 /= n - 1.0;
        c01 /= n - 1.0;
        let ess = ess_bulk(&zs);
        println!(
            "split sweep: ESS {ess:.0}, mean ({m0:+.3}, {m1:+.3}), \
             cov [[{v0:.3}, {c01:+.3}], [{c01:+.3}, {v1:.3}]]"
        );
        assert!(ess > 2_000.0, "ESS too low: {ess}");
        assert!(m0.abs() < 0.03 && m1.abs() < 0.03, "means ({m0}, {m1})");
        assert!((v0 - 0.6).abs() < 0.03, "Var(z) {v0}");
        assert!((v1 - 0.6).abs() < 0.03, "Var(w) {v1}");
        assert!((c01 + 0.4).abs() < 0.03, "Cov {c01}");
    }
}
