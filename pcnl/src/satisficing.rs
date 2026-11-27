//! R14/P0 — the satisficing ("good-enough region") potential.
//!
//! ```text
//! Φ_{τ,κ,α}(θ) = κ · ψ_δ(L(θ) − τ) + α · L(θ),        L(θ) ≥ 0
//! ```
//!
//! `ψ_δ` is a smooth nondecreasing one-sided hinge.  Inside the declared
//! good-enough set `{L ≤ τ}` the hinge contribution is (identically, for the
//! Huberized hinge) zero, so the potential does not reward memorization any
//! further; only the gentle `α`-tilt remains.  Outside the set the term grows
//! and pushes the configuration back in.
//!
//! The two hinge families are the ones named in `RESEARCH_PLAN.md` §14.1:
//!
//! * [`Hinge::Softplus`] — `ψ_δ(u) = δ·log(1 + e^{u/δ})`, strictly positive
//!   everywhere (so the potential is smooth and its gradient never vanishes
//!   outside the set);
//! * [`Hinge::Huberized`] — `ψ(u) = ½·max(0,u)²`, `C¹` with `ψ′ = max(0,u)`,
//!   exactly zero gradient inside the plateau (also exactly zero value there).
//!
//! `Φ ≥ 0` whenever `L ≥ 0` and `α ≥ 0`, so `0 < Z ≤ 1` under a proper prior
//! (the D0/D1 and Lean S0 proper-target contract).  The target identity hashes
//! the configuration, the loss label/reduction and the prior covariance, so two
//! runs that change any of them cannot be compared as the same target.

use crate::manifest::Fnv;
use crate::target::Potential;
use crate::vector_model::EnergyTarget;
use candle_core::Tensor;

/// The smooth one-sided hinge family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Hinge {
    /// `δ·log(1 + e^{u/δ})`.
    Softplus,
    /// `½·max(0, u)²` (a `C¹` hinge; `ψ′` jumps at `u = 0` only).
    Huberized,
}

impl Hinge {
    pub fn label(self) -> &'static str {
        match self {
            Self::Softplus => "softplus",
            Self::Huberized => "huberized",
        }
    }
}

/// The declared parameters of `Φ_{τ,κ,α}`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SatisficingConfig {
    /// Declared good-enough loss ceiling `τ > 0` — never zero.
    pub tau: f64,
    /// Confinement strength `κ > 0`.
    pub kappa: f64,
    /// Gentle optimization tilt `α ≥ 0`, required to be `α < κ`.
    pub alpha: f64,
    /// Smoothing scale `δ > 0` of the hinge.
    pub delta: f64,
    /// Hinge family.
    pub hinge: Hinge,
}

impl SatisficingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err("satisficing: tau must be finite and > 0".into());
        }
        if !self.kappa.is_finite() || self.kappa <= 0.0 {
            return Err("satisficing: kappa must be finite and > 0".into());
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err("satisficing: alpha must be finite and >= 0".into());
        }
        if self.alpha >= self.kappa {
            return Err("satisficing: alpha must be < kappa (gentle tilt)".into());
        }
        if !self.delta.is_finite() || self.delta <= 0.0 {
            return Err("satisficing: delta must be finite and > 0".into());
        }
        Ok(())
    }

    /// `ψ_δ(u)`.
    pub fn psi(&self, u: f64) -> f64 {
        match self.hinge {
            Hinge::Softplus => {
                let x = u / self.delta;
                if x > 0.0 {
                    u + self.delta * (1.0 + (-x).exp()).ln()
                } else {
                    self.delta * (1.0 + x.exp()).ln()
                }
            }
            Hinge::Huberized => 0.5 * u.max(0.0) * u.max(0.0),
        }
    }

    /// `ψ_δ′(u)`.
    pub fn psi_prime(&self, u: f64) -> f64 {
        match self.hinge {
            Hinge::Softplus => {
                let x = u / self.delta;
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    x.exp() / (1.0 + x.exp())
                }
            }
            Hinge::Huberized => u.max(0.0),
        }
    }

    /// The potential and its weight multiplying `∇L`:
    /// `Φ = κψ(L−τ) + αL` and `∇Φ = (κψ′(L−τ) + α)·∇L`.
    pub fn phi_and_weight(&self, loss: f64) -> (f64, f64) {
        let u = loss - self.tau;
        (
            self.kappa * self.psi(u) + self.alpha * loss,
            self.kappa * self.psi_prime(u) + self.alpha,
        )
    }
}

/// A target identity that changes whenever the potential, the loss contract or
/// the prior covariance changes.
pub fn target_identity(
    config: &SatisficingConfig,
    loss_label: &str,
    reduction: &str,
    prior_c: &[f64],
) -> u64 {
    let mut h = Fnv::new();
    h.write(b"pcnl/r14/satisficing/v1");
    h.write(config.hinge.label().as_bytes());
    for v in [config.tau, config.kappa, config.alpha, config.delta] {
        h.write(&v.to_bits().to_le_bytes());
    }
    h.write(loss_label.as_bytes());
    h.write(b"\0");
    h.write(reduction.as_bytes());
    h.write(b"\0");
    for c in prior_c {
        h.write(&c.to_bits().to_le_bytes());
    }
    h.finish()
}

/// Wraps any differentiable `EnergyTarget` whose energy is a nonnegative loss
/// `L(θ)` and exposes `Φ_{τ,κ,α}` as the energy the exact kernel consumes.
///
/// The prior stays outside this wrapper: `EnergyTarget` is the likelihood part,
/// and the samplers add `½θᵀC⁻¹θ` exactly as they do for every other target.
#[derive(Clone, Debug)]
pub struct Satisficing<T> {
    inner: T,
    config: SatisficingConfig,
    /// Identity of the underlying loss for logs and manifests.
    loss_label: String,
    /// Reduction convention of `L` ("sum" / "mean" / ...).
    reduction: String,
}

impl<T: EnergyTarget> Satisficing<T> {
    pub fn new(
        inner: T,
        config: SatisficingConfig,
        loss_label: impl Into<String>,
        reduction: impl Into<String>,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            inner,
            config,
            loss_label: loss_label.into(),
            reduction: reduction.into(),
        })
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
    pub fn config(&self) -> SatisficingConfig {
        self.config
    }
    pub fn loss_label(&self) -> &str {
        &self.loss_label
    }
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// The underlying loss `L(θ)` (`None` where the inner target is not
    /// finite).
    pub fn loss(&self, theta: &[f32]) -> Option<f64> {
        self.inner
            .energy_and_grad(theta)
            .map(|(l, _)| l)
            .filter(|l| l.is_finite())
    }

    /// `Φ_{τ,κ,α}(θ)`.
    pub fn phi(&self, theta: &[f32]) -> Option<f64> {
        self.loss(theta).map(|l| self.config.phi_and_weight(l).0)
    }

    pub fn identity(&self, prior_c: &[f64]) -> u64 {
        target_identity(&self.config, &self.loss_label, &self.reduction, prior_c)
    }
}

impl<T: EnergyTarget> EnergyTarget for Satisficing<T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (loss, grad) = self.inner.energy_and_grad(theta)?;
        if !loss.is_finite() || grad.len() != self.inner.dim() {
            return None;
        }
        if grad.iter().any(|g| !g.is_finite()) {
            return None;
        }
        let (phi, weight) = self.config.phi_and_weight(loss);
        if !phi.is_finite() || !weight.is_finite() {
            return None;
        }
        Some((phi, grad.iter().map(|g| (f64::from(*g) * weight) as f32).collect()))
    }
}

/// `Potential` adapter for the tensor-based target/optimizer path.
///
/// `Potential` reports a **mean** loss and the target multiplies by
/// `λ · n_terms`.  With `L = λ·n·mean_loss`, the satisficing objective is
/// `Φ(L)`, so the reported mean is `Φ(L)/(λn)` and its gradient is
/// `(κψ′+α)·∇mean_loss`.  Both call paths therefore optimize the same `Φ`.
#[derive(Clone, Debug)]
pub struct SatisficingPotential<P: Potential> {
    inner: P,
    config: SatisficingConfig,
    lambda: f64,
}

impl<P: Potential> SatisficingPotential<P> {
    pub fn new(inner: P, config: SatisficingConfig, lambda: f64) -> Result<Self, String> {
        config.validate()?;
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err("satisficing: lambda must be finite and > 0".into());
        }
        Ok(Self {
            inner,
            config,
            lambda,
        })
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }
    pub fn config(&self) -> SatisficingConfig {
        self.config
    }
    /// The declared loss `L(θ) = λ · n_terms · mean_loss(θ)`.
    pub fn loss(&self, theta: &Tensor) -> f64 {
        let mean = self.inner.mean_loss(theta);
        if !mean.is_finite() {
            return f64::INFINITY;
        }
        self.lambda * self.inner.n_terms() as f64 * mean
    }
}

impl<P: Potential> Potential for SatisficingPotential<P> {
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        let loss = self.loss(theta);
        if !loss.is_finite() {
            return f64::INFINITY;
        }
        let (phi, _) = self.config.phi_and_weight(loss);
        let scale = self.lambda * self.inner.n_terms() as f64;
        if scale <= 0.0 || !phi.is_finite() {
            return f64::INFINITY;
        }
        phi / scale
    }

    fn mean_grad_with(&self, theta: &Tensor, norm: crate::normalize::NormGrad) -> Option<Tensor> {
        let loss = self.loss(theta);
        if !loss.is_finite() {
            return None;
        }
        let (_, weight) = self.config.phi_and_weight(loss);
        if !weight.is_finite() {
            return None;
        }
        let grad = self.inner.mean_grad_with(theta, norm)?;
        // `mean_grad_with` is ∇mean_loss; multiply by the hinge weight.  The
        // `1/(λn)` in `mean_loss` is undone by the target's `λn` multiplier.
        (grad * weight).ok()
    }

    fn label(&self) -> &'static str {
        "satisficing"
    }
}
