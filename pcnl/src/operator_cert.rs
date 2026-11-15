//! D6 — what the operator certificate can and cannot claim.
//!
//! `DEVELOPMENT_PLAN.md` D6 asks for "a full-space positive certificate or a
//! precisely scoped missing estimate; never a completed theorem whose concrete
//! gap assumption is the claim being investigated." This module therefore
//! contains two things, kept apart on purpose:
//!
//! 1. [`GaussianOperatorCertificate`] — a **genuine** full-space spectral
//!    certificate for the D3 Euler–Langevin canary. The target `N(0, v)` with
//!    `v = 2h/(1−a²)` is invariant, detailed balance holds exactly, the kernel
//!    is self-adjoint on `L²(N(0,v))`, and its eigenvalues on the Hermite basis
//!    are `a^k` with `a = 1−hλ`. This certifies the *canary*, not the DT
//!    application target, and it is only valid for `0 < a < 1`.
//! 2. [`missing_dt_estimate`] — the scoped missing estimate for the actual DT
//!    invariant-law operator. No tail, coupling, projection or arithmetic bound
//!    is asserted; each absent ingredient is named.
//!
//! A finite matrix block, a raw polynomial band or a small seed residual is not
//! a full-space certificate, and nothing here upgrades one to that status.

/// A certified full-space spectral statement for one Gaussian canary chain.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianOperatorCertificate {
    /// Target precision `λ`.
    pub lambda: f64,
    /// Step size `h`.
    pub h: f64,
    /// Contraction `a = 1 − hλ`.
    pub contraction: f64,
    /// Invariant variance `v = 2h/(1−a²)`, the measure of `L²(π)`.
    pub variance: f64,
    /// Spectral gap `1 − a` on `L²(π)` (positive only for `a < 1`).
    pub gap: f64,
    /// The certified eigenvalue sequence `a^k`, `k = 0..truncation`.
    pub eigenvalues: Vec<f64>,
    /// Tail bound for the truncated sequence: `a^K/(1−a)`.
    pub tail_mass: f64,
    /// `a²v + 2h − v`, which is exactly zero when invariance holds.
    pub invariance_residual: f64,
}

impl GaussianOperatorCertificate {
    /// Certify the canary at `λ, h`, or return `None` outside its hypotheses.
    ///
    /// The chain `x' = a x + √(2h) ξ` with `a = 1−hλ` has invariant law
    /// `N(0, v)`, `v = 2h/(1−a²)`, and detailed balance
    /// `π(x)p(x'|x) = π(x')p(x|x')`. It is self-adjoint on `L²(π)` and its
    /// Hermite eigenfunctions have eigenvalues `a^k`, so the L² gap is `1−|a|`.
    /// No truncation is needed for the gap; `eigenvalues` only records a finite
    /// prefix and its tail.
    pub fn certify(lambda: f64, h: f64, truncation: usize) -> Option<Self> {
        if !(lambda.is_finite() && lambda > 0.0 && h.is_finite() && h > 0.0) {
            return None;
        }
        let a = 1.0 - h * lambda;
        // Outside 0 < a < 1 the canary is either unstable (a ≤ −1), critical
        // (a = −1 or a = 1) or has a sign-alternating spectrum; none of those
        // is covered by this certificate.
        if !(a > 0.0 && a < 1.0) || truncation == 0 {
            return None;
        }
        let v = 2.0 * h / (1.0 - a * a);
        let eigenvalues: Vec<f64> = (0..truncation).map(|k| a.powi(k as i32)).collect();
        let tail = a.powi(truncation as i32) / (1.0 - a);
        Some(Self {
            lambda,
            h,
            contraction: a,
            variance: v,
            gap: 1.0 - a,
            eigenvalues,
            tail_mass: tail,
            invariance_residual: a * a * v + 2.0 * h - v,
        })
    }

    /// The invariance residual is floating-point zero, not an estimate.
    pub fn invariance_holds(&self, tolerance: f64) -> bool {
        self.invariance_residual.abs() <= tolerance * (1.0 + self.variance)
    }

    /// A finite prefix plus its certified tail: `Σ_{k<K} a^k + tail`.
    pub fn truncation_covers(&self, tolerance: f64) -> bool {
        let partial: f64 = self.eigenvalues.iter().sum();
        partial + self.tail_mass >= 1.0 / (1.0 - self.contraction) - tolerance
    }
}

/// One ingredient D6 requires for the DT operator, and whether it exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingIngredient {
    pub name: &'static str,
    pub present: bool,
}

/// The scoped missing estimate for the DT target's invariant-law operator.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorScope {
    pub target: &'static str,
    pub ingredients: Vec<MissingIngredient>,
}

impl OperatorScope {
    pub fn missing(&self) -> Vec<&'static str> {
        self.ingredients
            .iter()
            .filter(|i| !i.present)
            .map(|i| i.name)
            .collect()
    }

    /// Never true while an ingredient is missing — this is the point of the
    /// record: it prevents a local spectrum from being reported as a theorem.
    pub fn certified(&self) -> bool {
        self.ingredients.iter().all(|i| i.present)
    }
}

/// The actual state of the DT operator certificate: all four bounds are absent.
pub fn missing_dt_estimate() -> OperatorScope {
    OperatorScope {
        target: "DT invariant-law L² operator on the adapted target",
        ingredients: vec![
            MissingIngredient {
                name: "certified retained-block tail bound",
                present: false,
            },
            MissingIngredient {
                name: "lazy/nonreversible coupling bound",
                present: false,
            },
            MissingIngredient {
                name: "ground-projection error bound",
                present: false,
            },
            MissingIngredient {
                name: "arithmetic (f32) spectral perturbation bound",
                present: false,
            },
        ],
    }
}
