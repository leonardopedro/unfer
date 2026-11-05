//! pCN on the sphere of directions — the target §6.5.7 says the Gaussian prior
//! should have been.
//!
//! **Why a sphere.**  `docs/PCNL_STATUS.md` (2026-11-05) measured that the
//! Task-B fine-tune lives at `‖θ‖ = 1272.6`, **28.2×** the `σ√P = 45.2` shell,
//! where `N(0, σ²I)` pays `r²/2σ² ≈ 6.6 × 10⁶` nats.  Every sampler of that
//! target therefore reports the shell's Haar-typical direction, which is
//! chance, no matter what it is handed.  But the cosine variant's forward pass
//! is scale-invariant (§3.3), so predictions depend only on the *direction* of
//! `θ`: the geometry the model actually constrains lives on the sphere, and a
//! uniform (Haar) prior there gives the fine-tune's direction full mass.
//!
//! **The kernel.**  Exponential-map pCN: at `u ∈ S^{P−1}` draw a tangent
//! vector `v = e + m(u)` with `e ~ N(0, τ²)` on `T_u` and the Langevin mean
//! `m(u) = −(τ²/2)·λN·∇_T L`, then move along the geodesic,
//! `u′ = cos|v|·u + sin(|v|)·v/|v|`.  `τ = βσ/ρ` is the ambient pCN step
//! `βσ` expressed in angular units, so the same `β` means the same *relative*
//! move as in `SgPcnlState` — the comparison is apples-to-apples.
//!
//! **The filter is exact, and the Jacobian cancels.**  In the tangent
//! coordinates at `u` the proposal has density
//! `q(u′|u) = N_{P−1}(v; m(u), τ²)·J(|v|)` with `J(a) = (sin a / a)^{P−2}`
//! (the exponential map's volume distortion).  The reverse proposal runs back
//! along the same geodesic, so `|w| = |v|` and `J` is the *same factor on both
//! sides*: it drops out of
//!
//! ```text
//! log α = λN(L(u) − L(u′)) + (‖v − m(u)‖² − ‖w − m(u′)‖²) / 2τ²,
//! ```
//!
//! which is therefore exact Metropolis with no numerical integration of a
//! projected Gaussian.  `(‖v − m‖²` is just `‖e‖²` on the forward side.)
//! The reverse tangent vector is obtained by parallel transport along the same
//! geodesic (`w = a·(sin a·u − cos a·v̂)`) rather than by a log map, because
//! `u − cos a·u′` loses ~5 digits to cancellation at `a ≈ τ`.
//!
//! Unlike the ambient kernel, a step here costs **two** full-batch gradient
//! evaluations: the drift of the *reverse* proposal needs `∇L(u′)`.
//!
//! **What it is for, and what it costs.**  At `λ = 1` and the published `β`,
//! no proposal this size is accepted in 16 672 dimensions, so the chain freezes
//! where it starts — which is why [`SpherePcn::step`] returns the acceptance
//! count and every caller reports it.  The informative regime is the small-`β`
//! one where moves are accepted *and* the target still has mass at the
//! fine-tune's direction; `examples/sphere_chain.rs` measures both.
//!
//! **Where the prior went.**  `π(u) ∝ exp(−λNL(ρu))` has *no* prior factor
//! beyond the uniform one, so the drift carries the whole likelihood gradient
//! and the radial marginal is a point mass at `ρ` by construction.  `ρ` is a
//! modelling choice, not a sampled coordinate: the ambient `σ√P` is the
//! comparable value, and scale invariance makes the predictive independent of
//! it.

use crate::sampler::{SgPcnlScheduler, dot, norm};
use candle_core::{Result, Tensor};
use rand::Rng;
use rand_distr::StandardNormal;

/// One spherical pCN chain with an exact Metropolis filter.
pub struct SpherePcn {
    /// Unit direction. The parameter is `radius · u`.
    pub u: Tensor,
    /// `ρ` — the fixed radius the direction is lifted to.
    pub radius: f64,
    /// Ambient `σ`; only the ratio `σ/ρ` matters, and it sets the angular
    /// noise scale `τ = βσ/ρ`.
    pub sigma: f64,
    pub beta_0: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub step_count: usize,
    /// `N`, the dataset size the potential is summed over.
    pub dataset_size: usize,
    /// `λ`, the likelihood temperature of the target.
    pub potential_scale: f64,
    /// How many proposals were accepted — the number that decides whether a
    /// "retained" predictive is a sample or a frozen chain.
    pub accepted: usize,
}

/// `(accepted, log α)` — the second is worth recording: a chain that rejects
/// everything still reports the same predictive it started with.
pub type StepOutcome = (bool, f64);

impl SpherePcn {
    pub fn new(
        u: Tensor,
        radius: f64,
        sigma: f64,
        sched: SgPcnlScheduler,
        dataset_size: usize,
        potential_scale: f64,
    ) -> Self {
        Self {
            u,
            radius,
            sigma,
            beta_0: sched.beta_0,
            gamma: sched.gamma,
            kappa: sched.kappa,
            step_count: 0,
            dataset_size,
            potential_scale,
            accepted: 0,
        }
    }

    pub fn scheduler(&self) -> SgPcnlScheduler {
        SgPcnlScheduler {
            beta_0: self.beta_0,
            gamma: self.gamma,
            kappa: self.kappa,
        }
    }

    pub fn beta(&self) -> f64 {
        self.scheduler().beta_at(self.step_count)
    }

    /// The parameter tensor this direction denotes: `ρ·u`.
    pub fn theta(&self) -> Result<Tensor> {
        self.u.affine(self.radius, 0.0)
    }

    /// One exact MH step.  `potential(θ)` must return `(mean NLL, ∇θ mean NLL)`
    /// over the *whole* training set — the summed potential is `N` times it,
    /// and the `N` is applied here.
    pub fn step<F>(&mut self, potential: F, rng: &mut impl Rng) -> Result<StepOutcome>
    where
        F: Fn(&Tensor) -> Result<(f64, Tensor)>,
    {
        let beta = self.beta();
        let tau = beta * self.sigma / self.radius;
        if tau <= 0.0 {
            self.step_count += 1;
            return Ok((true, 0.0));
        }
        let p = self.u.elem_count();
        let dev = self.u.device();

        let theta = self.theta()?;
        let (l0, g0) = potential(&theta)?;

        // Tangential Gaussian noise: the projection of an isotropic Gaussian
        // onto T_u is isotropic there with the same variance per coordinate,
        // and the scale τ is what puts it on the sphere (τσ missing here is a
        // wrong proposal variance, and the MH ratio stops being exact).
        let xi: Vec<f32> = (0..p)
            .map(|_| {
                let z: f64 = rng.sample(StandardNormal);
                (z * tau) as f32
            })
            .collect();
        let xi = Tensor::from_vec(xi, vec![p], dev)?;
        let xi_u = dot(&xi, &self.u)?;
        let e = (xi - self.u.affine(xi_u, 0.0)?)?;

        // Langevin mean in angular units: the ambient coefficient
        // h = λβ²σ²/2 divided by the radius, times the summed gradient.
        let coeff = self.potential_scale * beta * beta * self.sigma * self.sigma
            / (2.0 * self.radius)
            * self.dataset_size as f64;
        let m0 = tangential_drift(&g0, &self.u, coeff)?;
        let v = (&e + m0)?;
        let a = norm(&v)?;
        let u_next = exp_map(&self.u, &v, a)?;
        let theta_next = u_next.affine(self.radius, 0.0)?;
        let (l1, g1) = potential(&theta_next)?;

        // Reverse proposal: back along the same geodesic (parallel transport).
        let w = reverse_tangent(&self.u, &v, a)?;
        let m1 = tangential_drift(&g1, &u_next, coeff)?;
        let back = (&w - m1)?;

        // ‖v − m₀‖² = ‖e‖² by construction; ‖w − m₁‖² is measured.
        let fwd_sq = dot(&e, &e)?;
        let rev_sq = dot(&back, &back)?;
        let log_alpha = self.potential_scale * self.dataset_size as f64 * (l0 - l1)
            + (fwd_sq - rev_sq) / (2.0 * tau * tau);

        self.step_count += 1;
        let accept =
            log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
        if accept {
            self.u = u_next;
            self.accepted += 1;
        }
        Ok((accept, log_alpha))
    }
}

/// `g − (g·u)u`.
fn tangential_drift(g: &Tensor, u: &Tensor, coeff: f64) -> Result<Tensor> {
    let gu = dot(g, u)?;
    g.affine(-coeff, 0.0)? + u.affine(coeff * gu, 0.0)?
}

fn exp_map(u: &Tensor, v: &Tensor, a: f64) -> Result<Tensor> {
    if a < 1e-12 {
        return Ok(u.clone());
    }
    u.affine(a.cos(), 0.0)? + v.affine(a.sin() / a, 0.0)?
}

/// `a·(sin a·u − cos a·v̂)`: the unit tangent at `u′ = exp_u(v)` that starts the
/// geodesic back to `u`.  It is `(u − cos a·u′)/sin a` (substitute
/// `u′ = cos a·u + sin a·v̂`), written in the transport form because the log
/// map's difference cancels ~5 digits at `a ≈ τ`.
///
/// The sign is easy to get wrong and the moment test catches it: walking from
/// `u′` in `−sin a·u + cos a·v̂` moves *forward* along the geodesic (to angle
/// `2a`), not back to `u`.
fn reverse_tangent(u: &Tensor, v: &Tensor, a: f64) -> Result<Tensor> {
    if a < 1e-12 {
        return Tensor::zeros_like(u);
    }
    let vhat = v.affine(1.0 / a, 0.0)?;
    let dir = (u.affine(a.sin(), 0.0)? - vhat.affine(a.cos(), 0.0)?)?;
    dir.affine(a, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::SeedableRng;

    const VON_MISES_K2: f64 = 0.697_774_657_964_008_6; // I₁(2)/I₀(2)

    /// A circle with a von Mises target: `L(θ) = −κ (θ·e₁)/ρ`, so
    /// `π(u) ∝ exp(κ u₁)` and `E[u₁] = I₁(κ)/I₀(κ)` exactly.  `drift = false`
    /// reports a zero gradient while keeping the loss, which turns the
    /// proposal into a symmetric geodesic random walk — the control that
    /// separates the likelihood ratio from the drift's bookkeeping.
    fn von_mises(
        kappa: f64,
        radius: f64,
        drift: bool,
    ) -> impl Fn(&Tensor) -> Result<(f64, Tensor)> {
        let dev = Device::Cpu;
        move |theta: &Tensor| {
            let cell = theta.flatten_all()?.to_vec1::<f32>()?;
            let u1 = cell[0] as f64 / radius;
            let g = if drift {
                Tensor::from_vec(vec![(-kappa / radius) as f32, 0.0], vec![2], &dev)?
            } else {
                Tensor::zeros(2, candle_core::DType::F32, &dev)?
            };
            Ok((-kappa * u1, g))
        }
    }

    fn circle(kappa: f64, beta_0: f64, steps: usize, drift: bool) -> (f64, f64) {
        let dev = Device::Cpu;
        let u0 = Tensor::from_vec(vec![1.0f32, 0.0], vec![2], &dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0,
            gamma: 0.0,
            kappa: 0.5,
        };
        // one datum, so the summed potential is the mean potential
        let mut chain = SpherePcn::new(u0, 1.0, 1.0, sched, 1, 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let e1 = Tensor::from_vec(vec![1.0f32, 0.0], vec![2], &dev).unwrap();
        let (mut sum, mut n) = (0.0, 0usize);
        for t in 0..steps {
            chain.step(von_mises(kappa, 1.0, drift), &mut rng).unwrap();
            if t % 5 == 0 {
                sum += dot(&chain.u, &e1).unwrap();
                n += 1;
            }
        }
        (sum / n as f64, chain.accepted as f64 / steps as f64)
    }

    /// The identity check the ambient kernel states as "λ = 0 ⇒ pCN": with no
    /// likelihood the sphere target is Haar, so `E[u₁] = 0`.
    #[test]
    fn zero_likelihood_is_haar_on_the_circle() {
        let (mean, rate) = circle(0.0, 0.5, 20_000, true);
        assert!(rate > 0.9, "λ = 0 should accept almost always, got {rate}");
        let se = 1.0 / ((20_000 / 5) as f64).sqrt();
        assert!(
            mean.abs() < 4.0 * se,
            "Haar mean should be ~0, got {mean} (se {se})"
        );
    }

    /// The exactness test: with the drift the mean of `u₁` must still be the
    /// von Mises ratio `I₁(κ)/I₀(κ)`.  Two real bugs were caught here — an
    /// unscaled proposal noise (τ left out of `e`) and a negated reverse
    /// tangent — both of which show up as a bias in this moment and nowhere
    /// else, because a wrong proposal density still yields a plausible-looking
    /// chain.
    #[test]
    fn circle_matches_the_von_mises_moment() {
        let (mean, rate) = circle(2.0, 0.5, 60_000, true);
        assert!(
            rate > 0.25,
            "E[u₁] = {mean}: acceptance {rate} is too low for the moment to mean anything"
        );
        let se = 0.30 / ((60_000 / 5) as f64).sqrt();
        assert!(
            (mean - VON_MISES_K2).abs() < 4.0 * se,
            "E[u₁] = {mean}, von Mises says {VON_MISES_K2} (se {se})"
        );
    }

    /// The same target with the drift off — MH is exact here by symmetry
    /// alone, so a failure isolates the *drift* bookkeeping rather than the
    /// ratio as a whole.
    #[test]
    fn driftless_circle_matches_the_von_mises_moment() {
        let (mean, rate) = circle(2.0, 0.5, 40_000, false);
        assert!(rate > 0.5, "driftless acceptance {rate}");
        let se = 0.30 / ((40_000 / 5) as f64).sqrt();
        assert!(
            (mean - VON_MISES_K2).abs() < 4.0 * se,
            "driftless E[u₁] = {mean}, von Mises says {VON_MISES_K2} (se {se})"
        );
    }
}
