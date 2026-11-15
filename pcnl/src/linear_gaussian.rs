//! D4 — an analytic linear-Gaussian coupled benchmark.
//!
//! `DEVELOPMENT_PLAN.md` D4 asks for a coupled target whose slice, informed
//! marginal, complement conditional and predictive are all known in closed
//! form, because "it should expose differences the factorized D2 example
//! cannot". The D2 target is `π_r ⊗ N(0, I_n)`: the likelihood ignores the
//! complement, so a frozen complement is *not* a model error there. Here the
//! likelihood sees the complement through `bᵀw`, so:
//!
//! ```text
//! prior        z ~ N(0,1),        w ~ N(0, I_n)
//! likelihood   y | z,w ~ N(z + bᵀw, σ²)
//! full post.   z | y ~ N(m_f, v_f),  y | z ~ N(z, σ²+‖b‖²)
//! slice        w ≡ 0  ⇒  z | y ~ N(m_s, v_s), y | z ~ N(z, σ²)
//! ```
//!
//! The slice is a **restricted statistical model**, not a full-support
//! approximation to the ambient posterior, and the difference between `v_s` and
//! `v_f` is the model-reduction error D4 asks to report separately from sampler
//! error. With `b = 0` the two coincide — which is exactly why the factorized D2
//! family cannot detect the reduction.
//!
//! The complement conditional is Gaussian too:
//!
//! ```text
//! w | z,y ~ N( b·(y−z)/(σ²+‖b‖²),  I − b bᵀ/(σ²+‖b‖²) )
//! ```
//!
//! and the predictive of a fresh outcome integrates the *joint* posterior
//! rather than treating `z` and `w` as independent.

/// Posterior of the scalar informed coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Posterior1 {
    pub mean: f64,
    pub var: f64,
}

/// `r = 1` informed coordinate, `n` complement coordinates, likelihood
/// `y | z,w ~ N(z + bᵀw, σ²)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearGaussian {
    pub n: usize,
    pub b: Vec<f64>,
    pub sigma2: f64,
}

impl LinearGaussian {
    pub fn new(n: usize, b: Vec<f64>, sigma2: f64) -> Self {
        assert_eq!(
            b.len(),
            n,
            "one coupling coefficient per complement coordinate"
        );
        assert!(
            sigma2 > 0.0,
            "the likelihood noise needs a positive variance"
        );
        assert!(
            b.iter().all(|x| x.is_finite()) && sigma2.is_finite(),
            "benchmark parameters must be finite"
        );
        Self { n, b, sigma2 }
    }

    /// `‖b‖²`.
    pub fn coupling_norm_sq(&self) -> f64 {
        self.b.iter().map(|x| x * x).sum()
    }

    /// Full-support posterior of `z` after marginalising `w` under its prior.
    pub fn full_posterior(&self, y: f64) -> Posterior1 {
        let v = self.sigma2 + self.coupling_norm_sq();
        let var = v / (1.0 + v);
        Posterior1 { mean: var * y, var }
    }

    /// The informed marginal `L̄(z)` induces the same posterior for `z` — the
    /// complement is integrated, not fixed.
    pub fn informed_marginal(&self, y: f64) -> Posterior1 {
        self.full_posterior(y)
    }

    /// The **frozen-complement slice** `w ≡ 0`. A different, restricted target.
    pub fn slice_posterior(&self, y: f64) -> Posterior1 {
        let var = self.sigma2 / (1.0 + self.sigma2);
        Posterior1 { mean: var * y, var }
    }

    /// `|Δmean| + |Δvar|` between the slice and the full posterior at `y`.
    pub fn reduction_error(&self, y: f64) -> f64 {
        let f = self.full_posterior(y);
        let s = self.slice_posterior(y);
        (f.mean - s.mean).abs() + (f.var - s.var).abs()
    }

    /// Posterior mean of `w` given `(z, y)`.
    pub fn complement_mean(&self, z: f64, y: f64) -> Vec<f64> {
        let t = self.sigma2 + self.coupling_norm_sq();
        let r = y - z;
        self.b.iter().map(|bi| bi * r / t).collect()
    }

    /// Variance of `bᵀw` given `(z, y)`: `‖b‖² − ‖b‖⁴/(σ²+‖b‖²)`.
    ///
    /// The full covariance is `I − b bᵀ/(σ²+‖b‖²)`: unit along every direction
    /// orthogonal to `b`, and `1 − ‖b‖²/(σ²+‖b‖²)` along `b`.
    pub fn complement_projection_var(&self) -> f64 {
        let s = self.coupling_norm_sq();
        let t = self.sigma2 + s;
        s - s * s / t
    }

    /// Predictive mean and variance of a fresh outcome
    /// `ỹ = z + bᵀw + ε`, integrating the joint posterior `(z, w) | y`:
    ///
    /// ```text
    /// E[ỹ] = m + ‖b‖²(y − m)/(σ²+‖b‖²)
    /// Var  = v + (s − s²/t) + s²v/t² − 2 s v/t + σ²,   s = ‖b‖², t = σ²+s
    /// ```
    ///
    /// The `−2 s v/t` term is the posterior covariance between `z` and `bᵀw`;
    /// dropping it (a factorized predictive) changes the variance.
    pub fn predictive(&self, y: f64) -> Posterior1 {
        let post = self.full_posterior(y);
        let s = self.coupling_norm_sq();
        let t = self.sigma2 + s;
        let mean = post.mean + s * (y - post.mean) / t;
        let var = post.var + s - s * s / t + s * s * post.var / (t * t) - 2.0 * s * post.var / t
            + self.sigma2;
        Posterior1 { mean, var }
    }

    /// The predictive a factorized treatment would report (`Cov(z,bᵀw)=0` and
    /// no `Var(bᵀw|z)` dependence), for the D4 "separately reported" contrast.
    pub fn predictive_factorized(&self, y: f64) -> Posterior1 {
        let post = self.full_posterior(y);
        let s = self.coupling_norm_sq();
        let mean = post.mean + s * (y - post.mean) / (self.sigma2 + s);
        Posterior1 {
            mean,
            var: post.var + self.complement_projection_var() + self.sigma2,
        }
    }
}
