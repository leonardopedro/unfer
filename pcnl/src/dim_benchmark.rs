//! D2 — the controlled dimension benchmark `π_r ⊗ N(0, I_n)`.
//!
//! `DEVELOPMENT_PLAN.md` D2 asks for an explicit family with a fixed nonconstant
//! informed likelihood and a frozen proposal, so complement dimension `n` can be
//! varied "without changing rank, task, likelihood scale or observable". The
//! product structure means the simultaneous product-kernel absolute gap is
//!
//! ```text
//! min(γ_r, 1 − ρ),   ρ = √(1 − β²),
//! ```
//!
//! where `γ_r = 1 − c_r` is the informed independence-sampler gap and
//! `1 − ρ` is the complement pCN gap. The point is that the constant contains
//! `n` nowhere: growing the complement preserves the bound exactly, at a
//! per-step cost that does grow. That is the *first* dimension claim in
//! `RESEARCH_PLAN.md` §R8.2, and it is deliberately separated from any coupled
//! target (whose deterministic sweep need not be reversible).
//!
//! The informed block is
//!
//! ```text
//! π_r(z) ∝ N(0, I_r)(z) · exp(−(λ/2)‖z − c‖²),
//! ```
//!
//! i.e. a nonconstant (unless `λ = 0` or `c = 0`) Gaussian likelihood. With
//! `μ0 = N(0,I_r)` as the independence proposal and `L = exp(−(λ/2)‖z−c‖²) ≤ 1`,
//! Q1's minorization is `c_r = Z_r` and the gap is `1 − Z_r`, with `Z_r` the
//! closed-form evidence in [`crate::certificate::gaussian_evidence`]. The
//! benchmark *is* an exact target — no Metropolis approximation is needed to
//! state its rate.

use crate::certificate::gaussian_evidence;
use crate::target::ZPotential;

/// `π_r ⊗ N(0, I_n)` in `z ∈ R^r` and `w ∈ R^n`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductTarget {
    pub r: usize,
    pub n: usize,
    pub lambda: f64,
    pub center: Vec<f64>,
}

impl ProductTarget {
    pub fn new(r: usize, n: usize, lambda: f64, center: Vec<f64>) -> Self {
        assert!(r > 0, "the informed block must be nonempty");
        assert_eq!(
            center.len(),
            r,
            "one center coordinate per informed dimension"
        );
        assert!(lambda >= 0.0, "lambda is a likelihood scale, not a sign");
        Self {
            r,
            n,
            lambda,
            center,
        }
    }

    /// The informed marginal's evidence under the independent prior proposal:
    /// `Z_r = ∫ exp(−λ‖z−c‖²/2) μ0(dz)`.
    pub fn informed_evidence(&self) -> f64 {
        gaussian_evidence(self.r, self.lambda, self.center.iter().map(|x| x * x).sum())
    }

    /// `γ_r = 1 − Z_r` — the informed block's absolute gap from Q1's
    /// independence-kernel minorization (`M = 1`).
    pub fn informed_gap(&self) -> f64 {
        1.0 - self.informed_evidence()
    }

    /// `1 − √(1 − β²)` — the complement pCN gap for the standard Gaussian.
    pub fn complement_gap(beta: f64) -> f64 {
        assert!(beta > 0.0 && beta <= 1.0, "beta must be in (0, 1]");
        1.0 - (1.0 - beta * beta).sqrt()
    }

    /// The simultaneous product-kernel gap `min(γ_r, 1 − ρ)`.
    pub fn product_gap(&self, beta: f64) -> f64 {
        self.informed_gap().min(Self::complement_gap(beta))
    }
}

impl ZPotential for ProductTarget {
    fn dim(&self) -> usize {
        self.r + self.n
    }

    /// `Φ(z,w) = ½‖z‖² + (λ/2)‖z−c‖² + ½‖w‖²` — the total Gibbs potential,
    /// prior included.
    fn phi(&self, x: &[f32]) -> f64 {
        assert_eq!(x.len(), self.dim(), "point has the wrong dimension");
        let mut total = 0.0;
        for (i, &zi) in x[..self.r].iter().enumerate() {
            let z = f64::from(zi);
            total += 0.5 * z * z + 0.5 * self.lambda * (z - self.center[i]).powi(2);
        }
        for &wi in &x[self.r..] {
            let w = f64::from(wi);
            total += 0.5 * w * w;
        }
        total
    }
}
