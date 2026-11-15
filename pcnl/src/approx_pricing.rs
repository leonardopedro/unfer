//! D3 — pricing the approximate (unadjusted) update, with closed-form canaries.
//!
//! `DEVELOPMENT_PLAN.md` D3 asks for the analytic canaries *before* an empirical
//! comparison: a correct Gaussian target with a biased invariant mean/variance,
//! a proper target with an unstable discretization, and independently inflated
//! gradient noise. All three are exact for the Euler–Langevin kernel on a
//! Gaussian target, so they can be stated rather than fitted:
//!
//! ```text
//! target      π(x) ∝ exp(−λ x²/2) ⇒ Var_π = 1/λ
//! update      x' = (1 − hλ) x + √(2h) ξ
//! stationary  Var_h = 2h / (1 − (1−hλ)²) = (1/λ) / (1 − hλ/2)
//! stability   |1 − hλ| < 1  ⇔  0 < h < 2/λ
//! noise       Var_h(σ_g²) = (2h + h²σ_g²) / (1 − (1−hλ)²)
//! ```
//!
//! The point of the module is to make the D3 budget arithmetic checkable, not to
//! certify the crate's SG-pCNL branch: that branch's step, floor, schedule and
//! rounding remain separate theorem IDs (`RESEARCH_PLAN.md` §R8.3). In
//! particular a nonzero invariant-variance bias is a property of the *ideal
//! real-arithmetic* kernel here; an `f32` rounding-to-statistical-error bridge is
//! a different claim and is not made.

use crate::manifest::Fnv;

/// Exact target variance `1/λ`.
pub fn exact_variance(lambda: f64) -> f64 {
    assert!(
        lambda > 0.0 && lambda.is_finite(),
        "lambda must be positive"
    );
    1.0 / lambda
}

/// `Var_h` of the Euler–Langevin chain on `N(0, 1/λ)`, or `∞` outside the
/// stability range.
pub fn stationary_variance(lambda: f64, h: f64) -> f64 {
    assert!(lambda > 0.0 && h > 0.0);
    let a = 1.0 - h * lambda;
    let denom = 1.0 - a * a;
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        2.0 * h / denom
    }
}

/// `Var_h / Var_π = 1/(1 − hλ/2)` — the invariant-variance bias factor.
pub fn variance_bias_factor(lambda: f64, h: f64) -> f64 {
    1.0 / (1.0 - h * lambda / 2.0)
}

/// `h < 2/λ`.
pub fn stability_limit(lambda: f64) -> f64 {
    2.0 / lambda
}

/// Stationary variance with an independent gradient-noise variance `σ_g² ≡ Var`
/// of an unbiased estimate, entering the drift as `h·(λx + noise)`.
pub fn noise_inflated_variance(lambda: f64, h: f64, gradient_noise_variance: f64) -> f64 {
    assert!(gradient_noise_variance >= 0.0);
    let a = 1.0 - h * lambda;
    let denom = 1.0 - a * a;
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        (2.0 * h + h * h * gradient_noise_variance) / denom
    }
}

/// Simulate the scalar Euler–Langevin chain (unit noise scale) and return the
/// draws after `burn`.
pub fn simulate(
    lambda: f64,
    h: f64,
    gradient_noise_variance: f64,
    steps: usize,
    burn: usize,
    seed: u64,
) -> Vec<f64> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x = 0.0f64;
    let mut out = Vec::with_capacity(steps.saturating_sub(burn));
    let a = 1.0 - h * lambda;
    let noise_sd = (2.0 * h + h * h * gradient_noise_variance).sqrt();
    for t in 0..steps {
        let u1 = 1.0 - rng.random::<f64>();
        let u2 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x = a * x + noise_sd * z;
        if t >= burn {
            out.push(x);
        }
    }
    out
}

/// D3's declared error budget record: which quantities are exact, which are
/// bounds, and which remain open. A run that cannot fill every field must say
/// so rather than let a default stand in for a measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorBudget {
    pub precision: &'static str,
    /// `Var_h − Var_π` from the analytic kernel (not a fitted run).
    pub variance_bias: f64,
    /// `h / (2/λ)`, the fraction of the stability range used.
    pub stability_fraction: f64,
    /// Stationary bias incurred by gradient noise.
    pub noise_bias: f64,
    pub open: Vec<&'static str>,
}

impl ErrorBudget {
    pub fn gaussian(lambda: f64, h: f64, gradient_noise_variance: f64) -> Self {
        let exact = exact_variance(lambda);
        let observed = stationary_variance(lambda, h);
        let noisy = noise_inflated_variance(lambda, h, gradient_noise_variance);
        Self {
            precision: "f64 analytic; f32 implementation not bridged",
            variance_bias: observed - exact,
            stability_fraction: h / stability_limit(lambda),
            noise_bias: noisy - observed,
            open: vec![
                "f32 rounding to statistical-error coupling",
                "minibatch log-ratio is not an exact MH ratio",
                "floored schedule is a fixed-step regime",
            ],
        }
    }

    /// The budget is admissible only inside the stability range.
    pub fn is_stable(&self) -> bool {
        self.stability_fraction < 1.0
    }

    /// FNV-1a over the named fields, for a run manifest.
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(self.precision.as_bytes());
        for v in [self.variance_bias, self.stability_fraction, self.noise_bias] {
            h.write(&v.to_bits().to_le_bytes());
        }
        for o in &self.open {
            h.write(o.as_bytes());
            h.write(b"\0");
        }
        h.finish()
    }
}
