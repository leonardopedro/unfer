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

use crate::sampler::SgPcnlScheduler;
use candle_core::{Result, Tensor};
use rand::Rng;
use rand_distr::{Beta, Distribution, StandardNormal};

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
    /// Told apart from `precond.is_some()`: an *installed* metric (`set_metric`)
    /// is not the same thing as "estimate one while warming up", and conflating
    /// the two is what made `with_preconditioner` a no-op.
    pub precond_enabled: bool,
    /// Diagonal metric `d` (see [`Self::step_preconditioned`]): the frozen
    /// estimate once warm-up is over, empty/unused otherwise.
    pub precond: Option<Vec<f64>>,
    pub precond_rho: f64,
    /// Running `E|Δg|` and `E|v|` per coordinate — the metric's source, as a
    /// *secant* `E|Δg|/E|v| ≈ H_ii`.
    ///
    /// Two estimates this replaced were both wrong, and the second was silently
    /// fatal: `E[g²]` measures the gradient's *level*, not the curvature, and
    /// `E[(Δg)²]` taken between accepted states is **identically zero on a frozen
    /// chain** — from the solution's direction nothing is accepted, the state
    /// never moves, `Δg = 0`, and the metric collapses to the identity on exactly
    /// the cells preconditioning exists to rescue.  The secant is measured along
    /// the *proposal*, which the step evaluates whether or not it is accepted.
    pub abs_dg: Vec<f64>,
    pub abs_disp: Vec<f64>,
    /// Steps during which the metric is estimated; frozen afterwards, which is
    /// what keeps the chain exactly `π`-invariant after warm-up.
    pub metric_warmup: usize,
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
            precond_enabled: false,
            precond: None,
            precond_rho: 0.9,
            abs_dg: Vec::new(),
            abs_disp: Vec::new(),
            metric_warmup: 200,
        }
    }

    /// Turn on diagonal preconditioning and return `self`.
    ///
    /// `examples/curvature_probe.rs` measures why it is needed and how sharp the
    /// point is: from Adam's direction the mean NLL rises by 3.08 nats (1 577
    /// summed) over a 0.3 rad move along `−ĝ`, the proposal's drift is 60 % of
    /// its own length there, and a diagonal secant `|Δg_i/v_i|` runs from a
    /// median of 0.19 to a max of 1 116 — a 6 000× tail.  Heterogeneous
    /// sharpness of that size is what a metric can equalise, and `1/τ²` cannot.
    pub fn with_preconditioner(mut self) -> Self {
        self.precond_enabled = true;
        self
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

    /// One isotropic exact MH step.
    pub fn step<F>(&mut self, potential: F, rng: &mut impl Rng) -> Result<StepOutcome>
    where
        F: Fn(&Tensor) -> Result<(f64, Tensor)>,
    {
        self.step_inner(potential, rng, false)
    }

    /// One exact MH step under a diagonal metric `d` — the preconditioned
    /// proposal.
    ///
    /// The metric enters the proposal as `e_i = τ√d_i·ξ_i` and the drift as
    /// `m_i = −c·d_i·g_i`, and the *same* `d` appears on both sides of the
    /// ratio, `−Σ(v_i−m_i)²/(2τ²d_i)`, so MH stays exact for whatever `d` is
    /// fixed during the step.  `d` is estimated from the diagonal secant
    /// `H_ii ≈ E|Δg_i|/E|v_i|` along the *proposal* **only during
    /// `metric_warmup` steps and then frozen**; that is what makes the
    /// post-warm-up chain exactly `π`-invariant rather than merely an adaptive
    /// approximation of it.  (Adaptive during warm-up is the price of not
    /// knowing `H` in advance; the kept draws all come from the frozen metric.)
    pub fn step_preconditioned<F>(
        &mut self,
        potential: F,
        rng: &mut impl Rng,
    ) -> Result<StepOutcome>
    where
        F: Fn(&Tensor) -> Result<(f64, Tensor)>,
    {
        self.step_inner(potential, rng, true)
    }

    /// Install a metric directly (tests, and experiments that hold `d` fixed).
    pub fn set_metric(&mut self, d: Vec<f64>) {
        assert!(!d.is_empty(), "metric must not be empty");
        assert!(
            d.iter().all(|x| x.is_finite() && *x > 0.0),
            "metric must be positive"
        );
        self.precond = Some(d);
        self.precond_enabled = true;
    }

    fn step_inner<F>(
        &mut self,
        potential: F,
        rng: &mut impl Rng,
        use_metric: bool,
    ) -> Result<StepOutcome>
    where
        F: Fn(&Tensor) -> Result<(f64, Tensor)>,
    {
        let beta = self.beta();
        let tau = beta * self.sigma / self.radius;
        let p = self.u.elem_count();
        if tau <= 0.0 {
            self.step_count += 1;
            return Ok((true, 0.0));
        }

        let u: Vec<f64> = self.u.to_vec1::<f32>()?.iter().map(|x| *x as f64).collect();
        let theta = self.theta()?;
        let (l0, g0) = potential(&theta)?;
        let g0: Vec<f64> = g0.to_vec1::<f32>()?.iter().map(|x| *x as f64).collect();

        // Metric: the frozen estimate after warm-up, the live one during it,
        // all ones when the preconditioner is off.
        let metric_active = use_metric && self.precond_enabled;
        let d: Vec<f64> = if !metric_active {
            vec![1.0; p]
        } else {
            match self.precond.as_ref().filter(|d| d.len() == p) {
                Some(d) => d.clone(),
                None => vec![1.0; p],
            }
        };

        // Tangential Gaussian noise with the metric's variance per coordinate.
        let e: Vec<f64> = (0..p)
            .map(|i| {
                let z: f64 = rng.sample(StandardNormal);
                tau * d[i].sqrt() * z
            })
            .collect();
        let e = project_tangent(&e, &u);

        // Langevin mean in angular units: the ambient coefficient
        // h = λβ²σ²/2 divided by the radius, times the summed gradient,
        // preconditioned coordinate-wise.
        let coeff = self.potential_scale * beta * beta * self.sigma * self.sigma
            / (2.0 * self.radius)
            * self.dataset_size as f64;
        // The drift is `−(τ²/2)·Σ(u)·∇U` with `Σ = P D P`, so the metric is
        // applied *before* the projection: `P D g`, not `D P g`. The two agree
        // only when `d` is isotropic, and on a sphere that difference is what
        // the moment test sees.
        let m0: Vec<f64> = {
            let dg: Vec<f64> = (0..p).map(|i| -coeff * d[i] * g0[i]).collect();
            project_tangent(&dg, &u)
        };

        let v: Vec<f64> = (0..p).map(|i| e[i] + m0[i]).collect();
        let a = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u_next = exp_map_vec(&u, &v, a);
        let u_next_t = Tensor::from_vec(
            u_next.iter().map(|x| *x as f32).collect(),
            vec![p],
            self.u.device(),
        )?;
        let (l1, g1) = potential(&u_next_t.affine(self.radius, 0.0)?)?;
        let g1: Vec<f64> = g1.to_vec1::<f32>()?.iter().map(|x| *x as f64).collect();

        // Reverse proposal: back along the same geodesic.
        let w = reverse_tangent_vec(&u, &v, a);
        let m1: Vec<f64> = {
            let dg: Vec<f64> = (0..p).map(|i| -coeff * d[i] * g1[i]).collect();
            project_tangent(&dg, &u_next)
        };

        // The proposal's covariance is `τ²·(P D P)` *restricted to T_u*, whose
        // inverse is **not** `D⁻¹`: with `q = uᵀD⁻¹u`,
        //
        // ```text
        // xᵀ(P D P|_T)⁻¹x = xᵀD⁻¹x − (xᵀD⁻¹u)²/q,     x ∈ T_u,
        // ```
        //
        // checked against a dense `QᵀDQ` solve in
        // `restricted_metric_inverse_matches_the_dense_one`.  Using only the
        // first term — what this did — leaves an O(1) error in the log ratio at
        // every step in P ≥ 3 dimensions, and the old 2D test could not see it
        // because it starts at `u = (1,0)`, where the missing term is O(u₂).
        let quad = |x: &[f64], u: &[f64]| -> f64 {
            let (mut q, mut cross, mut value) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..p {
                q += u[i] * u[i] / d[i];
                cross += x[i] * u[i] / d[i];
                value += x[i] * x[i] / d[i];
            }
            value - cross * cross / q.max(f64::MIN_POSITIVE)
        };
        // −‖v−m‖²_(Σ⁻¹)/2 on the forward side is −‖e‖²_(Σ⁻¹)/2 by construction.
        let fwd: f64 = quad(&e, &u);
        let back: Vec<f64> = (0..p).map(|i| w[i] - m1[i]).collect();
        let rev: f64 = quad(&back, &u_next);
        // The projected metric is `Σ(u) = τ²·P_u D P_u`, so its determinant is
        // *not* the same at `u` and at `u′` — the isotropic case hides this
        // because the factor is 1. `det(P D P + uuᵀ) = det(D)·(uᵀD⁻¹u)`
        // (block-diagonal on `span(u) ⊕ T_u`), and `det D` cancels, leaving
        // `log(uᵀD⁻¹u) − log(u′ᵀD⁻¹u′)` in the ratio. Dropping it makes the
        // chain sample a different law — the moment test drives `E[u₁]` to 0.
        let log_det = |x: &[f64]| -> f64 {
            let q: f64 = x.iter().zip(&d).map(|(xi, di)| xi * xi / di).sum();
            q.max(f64::MIN_POSITIVE).ln()
        };
        let log_alpha = self.potential_scale * self.dataset_size as f64 * (l0 - l1)
            + (fwd - rev) / (2.0 * tau * tau)
            + 0.5 * (log_det(&u) - log_det(&u_next));

        self.step_count += 1;
        let accept =
            log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
        if accept {
            self.u = Tensor::from_vec(
                u_next.iter().map(|x| *x as f32).collect(),
                vec![p],
                self.u.device(),
            )?;
            self.accepted += 1;
        }
        // Warm-up: collect the diagonal secant along the *proposal* and refresh
        // the metric; after `metric_warmup` steps it is frozen.  Measured on
        // the proposal, not on accepted states — a frozen chain (0 % accept)
        // learns nothing from the latter, which is exactly where the metric is
        // needed.
        if metric_active && self.step_count < self.metric_warmup {
            self.accumulate_curvature(&g0, &g1, &v);
        }
        Ok((accept, log_alpha))
    }
}

impl SpherePcn {
    /// Fold one proposal's secant into the curvature estimate: `H_ii ≈ Δg_i/v_i`
    /// with `v` the tangent displacement actually proposed.
    ///
    /// Two details make this usable rather than a heavy-tailed mess.  The
    /// coordinate's *absolute* change is accumulated, not its square, so one
    /// outlier proposal cannot own the estimate; and coordinates whose `v_i` is a
    /// small fraction of the proposal's typical size are skipped, because
    /// `Δg_i/v_i` there divides by nearly nothing and says more about the
    /// proposal's randomness than about the curvature.
    fn accumulate_curvature(&mut self, g0: &[f64], g1: &[f64], v: &[f64]) {
        let p = g0.len();
        if self.abs_dg.len() != p || self.abs_disp.len() != p {
            self.abs_dg = vec![0.0; p];
            self.abs_disp = vec![0.0; p];
        }
        let rms = (v.iter().map(|x| x * x).sum::<f64>() / p as f64).sqrt();
        let floor = 0.05 * rms;
        let rho = self.precond_rho;
        for i in 0..p {
            let vi = v[i].abs();
            if vi <= floor {
                continue;
            }
            self.abs_dg[i] = rho * self.abs_dg[i] + (1.0 - rho) * (g1[i] - g0[i]).abs();
            self.abs_disp[i] = rho * self.abs_disp[i] + (1.0 - rho) * vi;
        }
        self.precond = Some(metric_from_secant(&self.abs_dg, &self.abs_disp));
    }
}

/// `d_i = 1/H_ii` from `H_ii ≈ E|Δg_i|/E|v_i|`, normalised so `mean(d) = 1`: the
/// metric redistributes the step, it does not change its total size, so `β`
/// keeps meaning "a move of `β` radians".
///
/// This is the optimal scaling for a Gaussian target — with `d = H⁻¹` the
/// acceptance cost is `τ²·Σd_i h_i` whatever the spread of `h`, so *relative*
/// sharpness stops mattering.  What it cannot do is pay for a step that is too
/// long in every coordinate at once, which is the common-mode factor `√P` (see
/// `examples/curvature_probe.rs` for which of the two this point is).
///
/// The clamp at 100× the median keeps one under-estimated secant from turning a
/// coordinate into a free move, and coordinates that were never informed keep
/// `d = 1` before normalisation.
fn metric_from_secant(abs_dg: &[f64], abs_disp: &[f64]) -> Vec<f64> {
    let mut h: Vec<f64> = abs_dg
        .iter()
        .zip(abs_disp)
        .map(|(dg, disp)| if *disp > 0.0 { dg / disp } else { f64::NAN })
        .collect();
    let mut informed: Vec<f64> = h
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if informed.is_empty() {
        return vec![1.0; h.len()];
    }
    informed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = informed[informed.len() / 2];
    let (lo, hi) = (med / 100.0, med * 100.0);
    for x in &mut h {
        *x = if x.is_finite() && *x > 0.0 {
            x.clamp(lo, hi)
        } else {
            med
        };
    }
    let mut d: Vec<f64> = h.iter().map(|x| 1.0 / x).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    if mean.is_finite() && mean > 0.0 {
        for x in &mut d {
            *x /= mean;
        }
    }
    d
}

fn project_tangent(x: &[f64], u: &[f64]) -> Vec<f64> {
    let c: f64 = x.iter().zip(u).map(|(a, b)| a * b).sum();
    x.iter().zip(u).map(|(a, b)| a - c * b).collect()
}

fn exp_map_vec(u: &[f64], v: &[f64], a: f64) -> Vec<f64> {
    if a < 1e-15 {
        return u.to_vec();
    }
    let (c, s) = (a.cos(), a.sin() / a);
    (0..u.len()).map(|i| c * u[i] + s * v[i]).collect()
}

/// `a·(sin a·u − cos a·v̂)`: the unit tangent at `u′ = exp_u v` that points back
/// to `u`.  It is `(u − cos a·u′)/sin a`, written in the transport form because
/// the log map's difference cancels ~5 digits at `a ≈ τ`.  The sign is easy to
/// get wrong and the von Mises moment test catches it: walking from `u′` in
/// `−sin a·u + cos a·v̂` moves *forward* along the geodesic, to angle `2a`.
/// One fixed-radius shell jump: `u′ = exp_u(r·d̂)` with `d̂` uniform in `T_u`.
///
/// **Why it exists.**  [`SpherePcn::step`]'s step is `τ = β_tσ/ρ`, and the
/// schedule shrinks it, so its displacement accumulates as `√t`: a target whose
/// transverse width is a few 10⁻³ rad is diluted by that accumulation whatever
/// the acceptance (`examples/basin_posterior.rs` measures 49.22 % → 46.39 %).
/// This jump moves *exactly* `r` per accepted step, so its dilution is set by
/// `r` alone.
///
/// **It is symmetric, so the ratio is the target's alone.**  The law is the
/// Haar measure on the geodesic sphere of radius `r` around `u`,
/// `q(u′|u) = 1/(Vol(S^{P−2})·(sin r)^{P−2})`, and the reverse jump has the same
/// radius, so the two densities are equal and
///
/// ```text
/// log α = log π(u′) − log π(u)
/// ```
///
/// with no Jacobian, no reverse gradient, and one potential evaluation per step
/// (`log_target_u` is the caller's cached value for `u`).
///
/// **What it cannot fix.**  One step changes `‖log_{u₀} u‖` by `O(r/√P)`, so the
/// distance from the start still diffuses as `r√t`, and the relaxation time of
/// that coordinate is `≈ P/(K r²)` — 10³–10⁵ steps for the `P = 16 672`
/// target, i.e. at or beyond a 10 000-step budget.  The moment tests below pin
/// the invariant law at `P ∈ {2,3,5}`; a long chain that has not converged in
/// high dimension is a mixing statement, not an exactness one.
///
/// Returns `(proposal, accepted, log π(proposal))`.
pub fn shell_jump(
    u: &[f64],
    log_target_u: f64,
    r: f64,
    log_target: &dyn Fn(&[f64]) -> f64,
    rng: &mut impl Rng,
) -> (Vec<f64>, bool, f64) {
    let p = u.len();
    let raw: Vec<f64> = (0..p)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();
    let w = project_tangent(&raw, u);
    let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 0.0 {
        return (u.to_vec(), false, log_target_u);
    }
    let (c, s) = (r.cos(), r.sin() / n);
    let prop: Vec<f64> = (0..p).map(|i| c * u[i] + s * w[i]).collect();
    let log_prop = log_target(&prop);
    let log_alpha = log_prop - log_target_u;
    let take = log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
    if take {
        (prop, true, log_prop)
    } else {
        (u.to_vec(), false, log_target_u)
    }
}

/// **A move at fixed geodesic radius from `anchor`.**  `u′` is `u` rotated
/// about the axis `anchor` by `φ = r / sin d₀`, i.e. a uniform draw on the
/// latitude sphere `{v : ⟨a,v⟩ = ⟨a,u⟩}` at ambient geodesic distance `r`.
///
/// **Why it exists.**  A jump whose direction is uniform can leave the cap a
/// target is restricted to, and in `P` dimensions it leaves *almost always*
/// once the chain is near the boundary: from `d = R` the fraction of steps that
/// stay inside is `Φ̄(r√P/(2R))`, which is `e^{−P/8}` at `R ≈ r`.  Restricted
/// targets therefore reach the boundary and freeze — `examples/basin_posterior
/// --cap` measures 0.1 % acceptance and `R̂ = ∞` over 4 × 10 000 steps at
/// `R = 2.3e-3 = 2.7 r₁` — and no radius fixes it, because in high dimension
/// the cap's own Haar volume sits on its boundary.  A latitude move has
/// `⟨a,u′⟩ = ⟨a,u⟩` to machine precision, so the cap never rejects and the run
/// is spent on the conditional target `π(· | d(u,a) = d₀)`.
///
/// **It is symmetric**, so the ratio is the target's alone: the law is the
/// uniform measure on the geodesic sphere of radius `r` *inside* the latitude,
/// which is the same object at `u′` as at `u`, hence
/// `log α = log π(u′) − log π(u)` with no Jacobian.
///
/// **The step is the ambient distance**, `r`, so callers compare it directly
/// with `shell_jump`'s radius; the latitude sphere has radius `sin d₀` and the
/// rotation angle is `φ = r/sin d₀`.  At the anchor itself (`d₀ = 0`) there is
/// no latitude to move on and the move is rejected.
///
/// Returns `(proposal, accepted, log π(proposal))`.
pub fn latitude_jump(
    u: &[f64],
    log_target_u: f64,
    r: f64,
    anchor: &[f64],
    log_target: &dyn Fn(&[f64]) -> f64,
    rng: &mut impl Rng,
) -> (Vec<f64>, bool, f64) {
    let p = u.len();
    if anchor.len() != p || !r.is_finite() || !log_target_u.is_finite() {
        return (u.to_vec(), false, log_target_u);
    }
    // Both vectors are normalised here, not by the caller.  The anchor is the
    // *model's* direction and arrives from `f32` weights, so `|a| = 1 ± 5e-8`,
    // and a kernel that mixes `⟨u,a⟩` into the state's norm then drifts:
    // `|prop|² = c²|a|² + (1 − c²)` gains `2c²(|a|² − 1)` every step, so `|u|`
    // walks up by ~5e-8 per step until `⟨a,u⟩` reaches 1, `sin d` hits 0 and the
    // chain collapses onto the anchor with every later move rejected.  Measured
    // in `basin_posterior --arm latitude`: at `d₀ = 8.61e-4` the run reached
    // `d = 0` after ~1 000 steps and acceptance decayed as `1/t`, while at
    // `d₀ = 0.079` it survived all 10 000 — the collapse arrives after
    // `(1 − cos d₀)/(2|a|δ)` steps, i.e. ~10³ at 8.6e-4 and ~10⁵ at 0.079.
    let an = (anchor.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let un = (u.iter().map(|x| x * x).sum::<f64>()).sqrt();
    // `ok` rather than the comparison inside the `!`: a NaN fails either test.
    let ok = an > 0.0 && un > 0.0;
    if !ok {
        return (u.to_vec(), false, log_target_u);
    }
    // the state, as a point of the unit sphere, written about the unit anchor
    let c = (sdot(u, anchor) / (un * an)).clamp(-1.0, 1.0);
    let s = (1.0 - c * c).max(0.0).sqrt();
    let v: Vec<f64> = (0..p).map(|i| u[i] / un - c * anchor[i] / an).collect();
    let m = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let ok = m > 1e-12 && s > 1e-12;
    if !ok {
        return (u.to_vec(), false, log_target_u);
    }
    let ub: Vec<f64> = v.iter().map(|x| x / m).collect();
    let ab: Vec<f64> = anchor.iter().map(|x| x / an).collect();
    // A uniform direction in the latitude's tangent space, {v : v ⊥ u, v ⊥ a},
    // which has dimension P − 2.  The span removed is that of `{a, û}`, and
    // those two are *orthonormal* (`û = (u − c·a)/s ⊥ a`), so one pass is an
    // exact projection.  Subtracting `⟨w,u⟩u + ⟨w,a⟩a` in one pass — the
    // obvious way — is Jacobi on a nearly dependent pair and only removes
    // `1 − c²` of the component per pass; at `c = 0.92` two passes still leave a
    // fifth of it, the direction leaves the latitude, and the chain walks off
    // the sphere's `⟨a,u⟩ = const` surface while `|u|` grows (measured `|u|` =
    // 1.12 after one step before this was fixed).
    let raw: Vec<f64> = (0..p)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();
    let (wa, wb) = (sdot(&raw, &ab), sdot(&raw, &ub));
    let w: Vec<f64> = (0..p).map(|i| raw[i] - wa * ab[i] - wb * ub[i]).collect();
    let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 1e-12 {
        return (u.to_vec(), false, log_target_u);
    }
    let phi = (r / s).min(std::f64::consts::PI);
    let (cp, sp) = (phi.cos(), phi.sin() / n);
    let prop: Vec<f64> = (0..p)
        .map(|i| c * ab[i] + s * (cp * ub[i] + sp * w[i]))
        .collect();
    let log_prop = log_target(&prop);
    let log_alpha = log_prop - log_target_u;
    let take = log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
    if take {
        (prop, true, log_prop)
    } else {
        (u.to_vec(), false, log_target_u)
    }
}

/// `log π` restricted to the cap of geodesic radius `radius` about `anchor`.
///
/// **The kernel needs nothing new.**  A restriction is a statement about the
/// *target*, not about the proposal: where `π(u′) = 0` the MH ratio is `0`, so
/// the move is rejected, and no state outside the cap is ever occupied, so the
/// reverse density from it never has to be evaluated with a nonzero weight.
/// The proposal still ranges over the whole geodesic sphere of radius `r` and
/// its density is unchanged — `shell_jump`'s ratio stays the target's alone and
/// remains exact for the restricted target.  The price is wasted proposals that
/// leave the cap, which the acceptance column pays.
///
/// **Why the project needs it.**  The unrestricted target is prior-dominated at
/// `P = 16 672`: Haar gives the equatorial shell `10^{54 000}` times the cone's
/// mass, so the unrestricted chain walks out of the cone and its predictive is
/// the base rate (`docs/PCNL_STATUS.md`, 2026-11-06).  Restricting the target is
/// the only way to ask a question whose answer is a property of the target
/// rather than of a transient: `E[· | d(u, anchor) ≤ radius]`.  That conditional
/// is what the project's 48–51 % numbers have always been measuring.
///
/// Passing `f64::INFINITY` returns the target unchanged (the unrestricted
/// case), so callers can wire one closure and vary only the radius.
///
/// **The radius must be several steps wide.**  Exactness does not depend on the
/// step, but usability does.  From radius `d` a step `r` lands at
/// `d′² = d² + r² − 2dr·t` with `t = ⟨u, ŵ⟩`, so at the boundary `d = R` the
/// move stays inside only when `t ≥ r/(2R)`; on `S^{P−1}`, `t` concentrates
/// with sd `1/√P`, and the acceptance there is `Φ̄(r√P/(2R))`.  At `R = r` and
/// `P = 16 672` that is `e^{−P/8}`: the chain reaches the boundary in one step
/// and freezes on it (`a_cap_must_be_several_steps_wide_or_the_chain_freezes`
/// measures 0.0 % acceptance over 4 × 10 000 steps at `R = r₁`).  With a
/// potential that grows in `d` the boundary is rarely reached, so `R ≳ 5r`
/// keeps the whole run moving; the same rule is what makes the answering
/// conditional tight enough not to just restate the headline.
pub fn restrict_to_cap<'a, F>(
    log_target: F,
    anchor: &'a [f64],
    radius: f64,
) -> impl Fn(&[f64]) -> f64 + 'a
where
    F: Fn(&[f64]) -> f64 + 'a,
{
    move |u: &[f64]| {
        if u.len() != anchor.len() {
            return f64::NEG_INFINITY;
        }
        let c = sdot(u, anchor).clamp(-1.0, 1.0);
        if c.acos() > radius {
            f64::NEG_INFINITY
        } else {
            log_target(u)
        }
    }
}

fn reverse_tangent_vec(u: &[f64], v: &[f64], a: f64) -> Vec<f64> {
    if a < 1e-15 {
        return vec![0.0; u.len()];
    }
    (0..u.len())
        .map(|i| a * (a.sin() * u[i] - a.cos() * v[i] / a))
        .collect()
}

fn sdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A unit tangent at `u`, normalised; `None` when the projection vanishes.
fn unit_tangent(x: &[f64], u: &[f64]) -> Option<Vec<f64>> {
    let v = project_tangent(x, u);
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 1e-12 {
        None
    } else {
        Some(v.iter().map(|x| x / n).collect())
    }
}

/// The direction field a guided jump should use, and the one it should not.
///
/// **Frozen ambient gradient (right).**  Given a fixed vector `g ∈ ℝ^P` — the
/// potential's gradient at the anchor — `μ(x) = g − ⟨g,x⟩x` normalised is a
/// genuine field: near the anchor it is the drift direction, and it is cheap
/// (no gradient evaluation per step) because it is only *projected* at the
/// proposal.  It has the two singularities any nonzero field on a sphere must
/// have (the hairy-ball theorem), at `±ĝ`; `guided_shell_jump` returns a
/// rejection there, which is the correct no-op.
///
/// **Parallel transport from the anchor (wrong).**  The obvious alternative —
/// transport the anchor's gradient along geodesics — is degenerate: transported
/// along the geodesic to every point, the vector *is* the radial direction
/// `∂/∂a` of the distance from the anchor, so the field pushes outward
/// everywhere and the reverse direction always opposes it.  Measured before it
/// was replaced: on a flat target, `κ = 4`, `r = 0.4`, acceptance **4.0 %**
/// (the ratio is `−2κ` at every step, so the chain never moves), where the
/// ambient-gradient field accepts in the tens of percent.  Exactness survives
/// either way — a wrong field costs acceptance, never correctness — which is
/// why the uniformity test, not the moment test, is the one that catches it.
pub fn drift_direction(g: &[f64], x: &[f64]) -> Vec<f64> {
    project_tangent(g, x)
}

/// A unit tangent at `u` drawn from the von Mises–Fisher law
/// `p(ŵ) ∝ exp(κ⟨μ, ŵ⟩)` on the unit tangent sphere `S^{P−2}`.
///
/// The cosine `x = ⟨μ, ŵ⟩` has density `∝ (1−x²)^{(P−4)/2} e^{κx}`; Wood's
/// rejection sampler (1994) draws it exactly for every `P ≥ 3` with a Beta
/// proposal, and the residual direction is uniform in the `(P−3)`-sphere
/// orthogonal to `μ` inside `T_u`.  `P = 2` is the two-point case and is
/// handled directly.  `κ = 0` returns the uniform tangent direction, which is
/// the `shell_jump` law.
fn sample_vmf_tangent(u: &[f64], mu: &[f64], kappa: f64, rng: &mut impl Rng) -> Vec<f64> {
    let p = u.len();
    if !kappa.is_finite() || kappa <= 0.0 {
        return uniform_tangent(u, rng);
    }
    if p == 2 {
        // One-dimensional tangent space: the law has two atoms.
        let n = unit_tangent(&[0.0, 1.0], u).unwrap_or_else(|| vec![-u[1], u[0]]);
        let take = rng.random::<f64>() < 1.0 / (1.0 + (-2.0 * kappa * sdot(mu, &n)).exp());
        return if take {
            n
        } else {
            n.iter().map(|x| -x).collect()
        };
    }
    // `x = ⟨μ, ŵ⟩` has density ∝ (1−x²)^{(p−4)/2} e^{κx}: Wood's rejection
    // sampler with a Beta((p−2)/2, (p−2)/2) proposal.
    let mm = (p - 2) as f64;
    let b = (-2.0 * kappa + (4.0 * kappa * kappa + mm * mm).sqrt()) / mm;
    let x0 = (1.0 - b) / (1.0 + b);
    let cc = kappa * x0 + mm * (1.0 - x0 * x0).ln();
    let beta = Beta::new(0.5 * mm, 0.5 * mm).expect("Beta(1/2) parameters are valid");
    let x = loop {
        let z: f64 = beta.sample(rng);
        let x = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
        let uu = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if kappa * x + mm * (1.0 - x0 * x).ln() - cc >= uu.ln() {
            break x;
        }
    };
    // Uniform unit vector in `T_u ∩ μ⊥`.
    let perp = loop {
        let g = uniform_tangent(u, rng);
        let t = sdot(&g, mu);
        let v: Vec<f64> = g.iter().zip(mu).map(|(x, m)| x - t * m).collect();
        if let Some(v) = unit_tangent(&v, u) {
            break v;
        }
    };
    let s = (1.0 - x * x).max(0.0).sqrt();
    (0..p).map(|i| x * mu[i] + s * perp[i]).collect()
}

/// A uniform unit tangent at `u` — the direction law of [`shell_jump`].
fn uniform_tangent(u: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    loop {
        let raw: Vec<f64> = (0..u.len())
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        if let Some(v) = unit_tangent(&raw, u) {
            return v;
        }
    }
}

/// A fixed-radius shell jump whose *direction* is biased towards
/// `−∇log π`: `u′ = exp_u(r·ŵ)` with `ŵ ~ vMF(κ, μ(u))`.
///
/// **The problem it addresses.**  [`shell_jump`] draws `ŵ` uniformly, so the
/// radial coordinate of the chain performs an unbiased random walk of step `r`.
/// In `P` dimensions that walk needs `O(P)` steps to relax — `basin_posterior`
/// measures a growth exponent of 0.509 over 10 000 steps and a pairwise angle of
/// 89.7° across four chains, i.e. free diffusion — so a target whose mass sits
/// in a sub-milliradian cap is diluted by the budget, not by a biased ratio.
/// Biasing the direction is the one lever left that does not require a smaller
/// step (which only slows the same walk) or a low-rank metric (ruled out by
/// `subspace_bound`: acceptance at the published `β` needs a 39× reduction and
/// the curvature's top 8 directions supply 3.67×).
///
/// **How strong the guidance can be.**  The reverse of a directed step is
/// nearly *antiparallel* to the forward one — `ŵ′ ≈ −ŵ` to `O(r)` — so
/// `⟨μ(u′), ŵ′⟩ ≈ −⟨μ(u), ŵ⟩` at **any** radius and the ratio carries
/// `exp(−2κ⟨μ,ŵ⟩)`.  Measured on a flat target: `κ = 4` accepts 1.5–5 % whatever
/// the step, `κ = 0.5` accepts in the tens of percent.  So the usable window is
/// `2/√P ≪ κ ≪ 1`: the *bias* per step is `≈ κr/2` against a diffusion of
/// `r/√P`, so any `κ ≫ 2/√P` already makes the drift dominate the walk while
/// the filter still accepts.  That window is the whole point — a large `κ` is
/// not "more guidance", it is an optimisation step the filter correctly
/// refuses.
///
/// **It stays exact, and the ratio has one extra term.**  Both directions use
/// the same `κ` and the same dimension, so the vMF normaliser `C_{P−2}(κ)` and
/// the exponential map's volume factor `(sin r)^{P−2}` cancel between the two
/// sides.  What does *not* cancel is the mean direction, because `μ` differs at
/// the two ends:
///
/// ```text
/// log α = log π(u′) − log π(u) + κ(⟨μ(u′), ŵ′⟩ − ⟨μ(u), ŵ⟩),
/// ```
///
/// with `ŵ′` the unit tangent at `u′` pointing back to `u`.  Dropping the last
/// two terms is the same class of error as the restricted-metric bug: the chain
/// would then be a biased random walk in `−∇log π` with no restoring law.
/// `κ = 0` reproduces [`shell_jump`] exactly (same law, same ratio), which is
/// what its test asserts.
pub fn guided_shell_jump(
    u: &[f64],
    log_target_u: f64,
    r: f64,
    kappa: f64,
    direction: &dyn Fn(&[f64]) -> Vec<f64>,
    log_target: &dyn Fn(&[f64]) -> f64,
    rng: &mut impl Rng,
) -> (Vec<f64>, bool, f64) {
    let p = u.len();
    let mu = match unit_tangent(&direction(u), u) {
        Some(m) => m,
        None => return (u.to_vec(), false, log_target_u),
    };
    let w = sample_vmf_tangent(u, &mu, kappa, rng);
    let (c, s) = (r.cos(), r.sin());
    let prop: Vec<f64> = (0..p).map(|i| c * u[i] + s * w[i]).collect();
    let log_prop = log_target(&prop);
    // Unit tangent at `prop` pointing back to `u`.
    let back: Vec<f64> = (0..p).map(|i| s * u[i] - c * w[i]).collect();
    let mu_back = match unit_tangent(&direction(&prop), &prop) {
        Some(m) => m,
        None => return (u.to_vec(), false, log_target_u),
    };
    let log_alpha = log_prop - log_target_u + kappa * (sdot(&mu_back, &back) - sdot(&mu, &w));
    let take = log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
    if take {
        (prop, true, log_prop)
    } else {
        (u.to_vec(), false, log_target_u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::dot;
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

    /// Preconditioning must not change anything when the metric is trivial:
    /// with `d ≡ 1` the two entry points are the same step, bit for bit.
    #[test]
    fn constant_metric_is_the_isotropic_step() {
        let dev = Device::Cpu;
        let sched = SgPcnlScheduler {
            beta_0: 0.4,
            gamma: 0.0,
            kappa: 0.5,
        };
        let mk = || {
            SpherePcn::new(
                Tensor::from_vec(vec![0.6f32, 0.8], vec![2], &dev).unwrap(),
                1.0,
                1.0,
                sched,
                1,
                1.0,
            )
        };
        let mut iso = mk();
        let mut pre = mk().with_preconditioner();
        pre.set_metric(vec![1.0, 1.0]);
        pre.metric_warmup = 0;
        let mut r1 = rand::rngs::StdRng::seed_from_u64(5);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(5);
        for _ in 0..200 {
            let (a1, l1) = iso.step(von_mises(2.0, 1.0, true), &mut r1).unwrap();
            let (a2, l2) = pre
                .step_preconditioned(von_mises(2.0, 1.0, true), &mut r2)
                .unwrap();
            assert_eq!(a1, a2, "acceptance diverged");
            assert!((l1 - l2).abs() < 1e-9, "log α diverged: {l1} vs {l2}");
            let (x1, x2) = (dot(&iso.u, &iso.u).unwrap(), dot(&pre.u, &pre.u).unwrap());
            assert!((x1 - x2).abs() < 1e-9, "states diverged");
        }
    }

    /// Exactness under a *non-trivial* metric: `d = (1, 4)` doubles the
    /// tangent step, and the moment must still be von Mises.  A ratio that
    /// drops the `1/d` or the projected-metric determinant fails here and
    /// passes everywhere else (this test caught both).
    ///
    /// The error bar is by **batch means**, not `sd/√n`: consecutive draws are
    /// correlated, and an iid bar would reject a correct kernel.
    #[test]
    fn anisotropic_metric_still_matches_the_von_mises_moment() {
        let dev = Device::Cpu;
        let u0 = Tensor::from_vec(vec![1.0f32, 0.0], vec![2], &dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0: 0.25,
            gamma: 0.0,
            kappa: 0.5,
        };
        let mut chain = SpherePcn::new(u0, 1.0, 1.0, sched, 1, 1.0).with_preconditioner();
        chain.set_metric(vec![1.0, 4.0]);
        chain.metric_warmup = 0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let e1 = Tensor::from_vec(vec![1.0f32, 0.0], vec![2], &dev).unwrap();
        let steps = 200_000usize;
        let mut draws = Vec::new();
        for t in 0..steps {
            chain
                .step_preconditioned(von_mises(2.0, 1.0, true), &mut rng)
                .unwrap();
            if t % 5 == 0 {
                draws.push(dot(&chain.u, &e1).unwrap());
            }
        }
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let batches = 10usize;
        let per = draws.len() / batches;
        let bmeans: Vec<f64> = (0..batches)
            .map(|b| draws[b * per..(b + 1) * per].iter().sum::<f64>() / per as f64)
            .collect();
        let bm = bmeans.iter().sum::<f64>() / batches as f64;
        let var = bmeans.iter().map(|x| (x - bm).powi(2)).sum::<f64>() / (batches - 1) as f64;
        let se = (var / batches as f64).sqrt();
        let rate = chain.accepted as f64 / steps as f64;
        assert!(
            rate > 0.2,
            "acceptance {rate} too low to judge: mean {mean}"
        );
        assert!(
            (mean - VON_MISES_K2).abs() < 4.0 * se,
            "preconditioned E[u₁] = {mean}, von Mises says {VON_MISES_K2} \
             (batch-means se {se}, accept {rate})"
        );
    }

    /// Regression for the bug that made the preconditioned arm identical to the
    /// isotropic one: the metric used to be estimated from the gradient's change
    /// between *accepted* states, so on a chain that accepts nothing `Δg ≡ 0`,
    /// `d` collapsed to all-ones and the metric was inert on exactly the cells it
    /// exists for.  Measured along the proposal it is available regardless of
    /// acceptance, so a frozen chain still learns the anisotropy.
    #[test]
    fn warmup_learns_the_metric_even_when_nothing_is_accepted() {
        let dev = Device::Cpu;
        // Two directions, curvatures 1e6 and 1: `0.5(a·u₁² + b·u₂²)` on S¹.
        let (a, b) = (1e6f64, 1.0f64);
        let target = move |theta: &Tensor| {
            let cell = theta.flatten_all()?.to_vec1::<f32>()?;
            let (x, y) = (cell[0] as f64, cell[1] as f64);
            let g = Tensor::from_vec(
                vec![(a * x) as f32, (b * y) as f32],
                vec![2],
                &candle_core::Device::Cpu,
            )?;
            Ok((0.5 * (a * x * x + b * y * y), g))
        };
        let sched = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 0.0,
            kappa: 0.5,
        };
        let mut chain = SpherePcn::new(
            Tensor::from_vec(vec![0.8f32, 0.6], vec![2], &dev).unwrap(),
            1.0,
            1.0,
            sched,
            1,
            1.0,
        )
        .with_preconditioner();
        chain.metric_warmup = 40;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..chain.metric_warmup {
            chain
                .step_preconditioned(target, &mut rng)
                .expect("preconditioned step");
        }
        assert_eq!(
            chain.accepted, 0,
            "the point must be frozen for this test to mean anything"
        );
        let d = chain.precond.clone().expect("a metric was estimated");
        assert!(
            d[0] < 0.2 && d[1] > 1.5,
            "frozen chain should still learn that direction 0 is stiff: d = {d:?}"
        );
    }

    /// The exponent identity the preconditioned ratio relies on, against a dense
    /// solve: for `x ⊥ u`, `xᵀ(PDP|_T)⁻¹x = xᵀD⁻¹x − (xᵀD⁻¹u)²/(uᵀD⁻¹u)`.
    ///
    /// The right-hand side is what `step_preconditioned` evaluates.  Dropping
    /// the second term is the bug this file had until 2026-11-05: it is exact in
    /// `d = 1` (the cross term vanishes) and, in 2D, only when `u` is close to an
    /// axis — so the von Mises test passed.  At `P = 4` with `d` spanning 16× the
    /// two sides differ by 50 % here.
    #[test]
    fn restricted_metric_inverse_matches_the_dense_one() {
        let p = 4;
        let d = vec![1.0, 4.0, 0.25, 2.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let mut worst = 0.0f64;
        for _ in 0..20 {
            let raw: Vec<f64> = (0..p).map(|_| rng.random::<f64>() - 0.5).collect();
            let n = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let u: Vec<f64> = raw.iter().map(|x| x / n).collect();
            // orthonormal basis of T_u (Gram–Schmidt on projected random vectors)
            let mut basis: Vec<Vec<f64>> = Vec::new();
            for _ in 0..(p - 1) {
                let r: Vec<f64> = (0..p).map(|_| rng.random::<f64>() - 0.5).collect();
                let c: f64 = r.iter().zip(&u).map(|(a, b)| a * b).sum();
                let mut w: Vec<f64> = r.iter().zip(&u).map(|(a, b)| a - c * b).collect();
                for b in &basis {
                    let cb: f64 = w.iter().zip(b).map(|(a, b)| a * b).sum();
                    for i in 0..p {
                        w[i] -= cb * b[i];
                    }
                }
                let nw = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                basis.push(w.iter().map(|x| x / nw).collect());
            }
            let coef: Vec<f64> = (0..p - 1).map(|_| rng.random::<f64>() - 0.5).collect();
            let x: Vec<f64> = (0..p)
                .map(|j| basis.iter().zip(&coef).map(|(b, c)| c * b[j]).sum())
                .collect();

            // dense: G = QᵀDQ, value = bᵀG⁻¹b with b = Qᵀx
            let m = p - 1;
            let mut g = vec![vec![0.0f64; m]; m];
            for i in 0..m {
                for j in 0..m {
                    g[i][j] = (0..p).map(|k| basis[i][k] * d[k] * basis[j][k]).sum();
                }
            }
            let b: Vec<f64> = basis
                .iter()
                .map(|bi| bi.iter().zip(&x).map(|(p, q)| p * q).sum())
                .collect();
            let a = solve_dense(&g, &b);
            let dense: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

            let q_u: f64 = u.iter().zip(&d).map(|(ui, di)| ui * ui / di).sum();
            let cross: f64 = x
                .iter()
                .zip(&u)
                .zip(&d)
                .map(|((xi, ui), di)| xi * ui / di)
                .sum();
            let plain: f64 = x.iter().zip(&d).map(|(xi, di)| xi * xi / di).sum();
            let formula = plain - cross * cross / q_u;
            worst = worst.max((dense - formula).abs() / dense.abs().max(1e-12));
            assert!(
                (dense - formula).abs() < 1e-9 * dense.abs(),
                "dense {dense} vs formula {formula}"
            );
            // the term the ratio used to drop is not small
            assert!(
                (plain - dense).abs() > 1e-3 * dense.abs(),
                "this configuration does not exercise the correction"
            );
        }
        assert!(worst < 1e-9, "worst relative error {worst}");
    }

    fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut m: Vec<Vec<f64>> = a
            .iter()
            .zip(b)
            .map(|(row, rhs)| {
                let mut r = row.clone();
                r.push(*rhs);
                r
            })
            .collect();
        for c in 0..n {
            let piv = (c..n)
                .max_by(|x, y| m[*x][c].abs().partial_cmp(&m[*y][c].abs()).unwrap())
                .unwrap();
            m.swap(c, piv);
            for r in 0..n {
                if r != c {
                    let f = m[r][c] / m[c][c];
                    let pivot = m[c].clone();
                    for (x, y) in m[r].iter_mut().zip(&pivot).skip(c) {
                        *x -= f * y;
                    }
                }
            }
        }
        (0..n).map(|i| m[i][n] / m[i][i]).collect()
    }

    /// Exactness in **three** dimensions, where the restricted-inverse term does
    /// not vanish: a von Mises–Fisher target `π(u) ∝ exp(κ u₁)` on `S²` has
    /// `E[u₁] = coth κ − 1/κ`, and the chain starts *off* the axis so `u` visits
    /// the whole sphere.
    #[test]
    fn anisotropic_metric_is_exact_on_s2() {
        let dev = Device::Cpu;
        let kappa = 2.0f64;
        let u0 = Tensor::from_vec(vec![0.6f32, 0.8, 0.0], vec![3], &dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 0.0,
            kappa: 0.5,
        };
        let target = move |theta: &Tensor| {
            let cell = theta.flatten_all()?.to_vec1::<f32>()?;
            let g = Tensor::from_vec(vec![-kappa as f32, 0.0, 0.0], vec![3], &Device::Cpu)?;
            Ok((-kappa * cell[0] as f64, g))
        };
        let mut chain = SpherePcn::new(u0, 1.0, 1.0, sched, 1, 1.0).with_preconditioner();
        chain.set_metric(vec![1.0, 4.0, 0.25]);
        chain.metric_warmup = 0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let e1 = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], vec![3], &dev).unwrap();
        let steps = 400_000usize;
        let mut draws = Vec::new();
        for t in 0..steps {
            chain.step_preconditioned(target, &mut rng).unwrap();
            if t % 10 == 0 {
                draws.push(crate::sampler::dot(&chain.u, &e1).unwrap());
            }
        }
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let batches = 20usize;
        let per = draws.len() / batches;
        let bmeans: Vec<f64> = (0..batches)
            .map(|b| draws[b * per..(b + 1) * per].iter().sum::<f64>() / per as f64)
            .collect();
        let bm = bmeans.iter().sum::<f64>() / batches as f64;
        let var = bmeans.iter().map(|x| (x - bm).powi(2)).sum::<f64>() / (batches - 1) as f64;
        let se = (var / batches as f64).sqrt();
        let theory = 1.0 / kappa.tanh() - 1.0 / kappa; // coth κ − 1/κ
        let rate = chain.accepted as f64 / steps as f64;
        assert!(rate > 0.2, "acceptance {rate} too low to judge");
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "E[u₁] = {mean}, von Mises–Fisher says {theory} (batch-means se {se}, accept {rate})"
        );
    }

    // ---- the fixed-radius shell jump -------------------------------------

    /// `steps` proposals of [`shell_jump`], every `thin`-th draw kept after
    /// `burn`, plus the acceptance rate.
    fn shell_chain(
        u0: Vec<f64>,
        r: f64,
        log_target: &dyn Fn(&[f64]) -> f64,
        steps: usize,
        burn: usize,
        thin: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut u = u0;
        let mut lu = log_target(&u);
        let (mut accepted, mut kept) = (0usize, Vec::new());
        for t in 0..steps {
            let (next, take, ln) = shell_jump(&u, lu, r, log_target, &mut rng);
            if take {
                u = next;
                lu = ln;
                accepted += 1;
            }
            if t >= burn && (t - burn).is_multiple_of(thin) {
                kept.push(u.clone());
            }
        }
        (kept, accepted as f64 / steps as f64)
    }

    // ---- the fixed-latitude jump -----------------------------------------

    /// How long to run a test chain: proposals, burn-in, thinning and seed.
    /// Bundled so `latitude_chain` (which also needs the anchor) stays under
    /// clippy's argument limit without touching the knobs themselves.
    #[derive(Clone, Copy)]
    struct Run {
        steps: usize,
        burn: usize,
        thin: usize,
        seed: u64,
    }

    /// [`Run::steps`] proposals of [`latitude_jump`] about `anchor`, every
    /// `thin`-th draw kept after `burn`, plus the acceptance rate.
    fn latitude_chain(
        u0: Vec<f64>,
        r: f64,
        anchor: &[f64],
        log_target: &dyn Fn(&[f64]) -> f64,
        run: Run,
    ) -> (Vec<Vec<f64>>, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(run.seed);
        let mut u = u0;
        let mut lu = log_target(&u);
        let (mut accepted, mut kept) = (0usize, Vec::new());
        for t in 0..run.steps {
            let (next, take, ln) = latitude_jump(&u, lu, r, anchor, log_target, &mut rng);
            if take {
                u = next;
                lu = ln;
                accepted += 1;
            }
            if t >= run.burn && (t - run.burn).is_multiple_of(run.thin) {
                kept.push(u.clone());
            }
        }
        (kept, accepted as f64 / run.steps as f64)
    }

    /// **The move stays on the latitude and is uniform there.**  On a flat
    /// target the conditional law is Haar on the latitude sphere, whose mean is
    /// `⟨a,u⟩·a` exactly — one number, and it fails if the rotation leaks off
    /// the latitude (a `cos φ` applied to the wrong component moves it by `O(r²)`,
    /// which 4 batch-means se resolves).
    ///
    /// The cap arm is the point of the kernel: with `d₀ < cap` and a step that
    /// would leave a *free* walk's cap after two moves, acceptance stays at 100 %
    /// because `⟨a,u′⟩ = ⟨a,u⟩` — the restriction never rejects a latitude move.
    #[test]
    fn latitude_jump_stays_on_the_latitude() {
        let p = 7usize;
        let mut anchor = vec![0.0; p];
        anchor[0] = 1.0;
        let d0 = 0.4f64;
        let mut start = vec![0.0; p];
        start[0] = d0.cos();
        start[1] = d0.sin();
        let flat = |_: &[f64]| 0.0;
        let run = Run {
            steps: 20_000,
            burn: 1_000,
            thin: 5,
            seed: 71,
        };
        let (kept, rate) = latitude_chain(start, 0.25, &anchor, &flat, run);
        assert!(rate > 0.95, "a flat target must accept, got {rate}");
        let c = d0.cos();
        for u in &kept {
            let cm = sdot(u, &anchor);
            assert!((cm - c).abs() < 1e-12, "left the latitude: ⟨a,u⟩ = {cm}");
        }
        // the Haar mean of the latitude is c·a, so every other coordinate is 0
        for i in 1..p {
            let xs: Vec<f64> = kept.iter().map(|u| u[i]).collect();
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let se = batch_means_se(&xs);
            assert!(
                mean.abs() < 4.0 * se.max(1e-4),
                "latitude coordinate {i} has mean {mean} (se {se}), Haar says 0"
            );
        }
        let xs: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!(
            (mean - c).abs() < 1e-3,
            "E[u₁] = {mean}, the latitude pins {c}"
        );
    }

    /// **Exactness against a closed form the ratio cannot fake.**  von
    /// Mises–Fisher on `S³`, `π ∝ exp(κ⟨m,u⟩)`, conditioned on the latitude
    /// `⟨a,u⟩ = c`: with `m = (a + ê₂)/√2` the latitude's mean direction is `ê₂`
    /// and the conditional law is a von Mises–Fisher on the latitude sphere with
    /// concentration `κ·s·|m_⊥| = κ·s/√2`, so
    ///
    /// ```text
    /// E[u₂] = s·A(κ s/√2),   A(z) = coth z − 1/z
    /// ```
    ///
    /// and `E[u₁] = c` (pinned).  A symmetric kernel whose reverse step differed
    /// would move `E[u₂]` and leave `E[u₁]` alone — the pair is what separates
    /// the two failures.
    ///
    /// `P = 4` is the smallest `P` where this is the elementary form: the
    /// latitude there is `S²`, so the conditional law is a 3-dimensional vMF.
    /// At `P = 3` the latitude is a *circle* and the same construction reads
    /// `I₁(z)/I₀(z)` instead — writing `coth z − 1/z` there is a 0.14 error in
    /// `E[u₂]` (0.306 against the measured 0.445) that looks exactly like a
    /// broken kernel and is not one.  The `P` that fixes the closed form is the
    /// dimension of the *conditioned* space, not of the sphere.
    #[test]
    fn latitude_jump_is_exact_conditioned_on_the_latitude() {
        let kappa = 2.0f64;
        let d0 = 1.0f64;
        let (c, s) = (d0.cos(), d0.sin());
        let anchor = [1.0, 0.0, 0.0, 0.0];
        let start = vec![c, s, 0.0, 0.0];
        let log_target = |u: &[f64]| {
            let sq = std::f64::consts::FRAC_1_SQRT_2;
            kappa * (sq * u[0] + sq * u[1])
        };
        let run = Run {
            steps: 60_000,
            burn: 1_000,
            thin: 5,
            seed: 72,
        };
        let (kept, rate) = latitude_chain(start, 0.2, &anchor, &log_target, run);
        assert!(rate > 0.25, "acceptance {rate} too low to judge");
        let z = kappa * s * std::f64::consts::FRAC_1_SQRT_2;
        let theory = s * (1.0 / z.tanh() - 1.0 / z);
        let xs: Vec<f64> = kept.iter().map(|u| u[1]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let se = batch_means_se(&xs);
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "E[u₂] = {mean}, the conditioned von Mises says {theory} (se {se}, accept {rate})"
        );
        let ys: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let m1 = ys.iter().sum::<f64>() / ys.len() as f64;
        assert!(
            (m1 - c).abs() < 1e-12,
            "E[u₁] = {m1}, the latitude pins {c}"
        );
    }

    /// **A small latitude, and an anchor that is only `f32`-unit.**  The
    /// stabilisation in [`latitude_jump`] is invisible at `d₀ = 0.4` and fatal at
    /// `d₀ = 8.6e-4`: the model's direction arrives from `f32` weights, so
    /// `|a| = 1 + 5e-8`, and a kernel that folds `|a|` into `|u|` gains
    /// `2c²(|a|² − 1)` per step — `⟨a,u⟩` reaches 1 after
    /// `(1 − cos d₀)/(2|a|δ) ≈ 10³` steps, `sin d` hits 0, and the chain freezes
    /// on the anchor while the acceptance column decays as `1/t`.
    ///
    /// This is that failure at the radius the basin measurements use, with the
    /// two invariants that must hold at every draw: `|u| = 1` and
    /// `⟨â,u⟩ = cos d₀`.
    #[test]
    fn a_small_latitude_survives_a_non_unit_anchor() {
        let p = 256usize;
        let d0 = 8.61e-4f64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(73);
        let raw: Vec<f64> = (0..p).map(|_| rng.random::<f64>() - 0.5).collect();
        let n0 = (raw.iter().map(|x| x * x).sum::<f64>()).sqrt();
        // an `f32`-precise anchor: unit to 5e-8, not to 1e-16
        let anchor: Vec<f64> = raw.iter().map(|x| (x / n0) * (1.0 + 5e-8)).collect();
        let ab: Vec<f64> = anchor.iter().map(|x| x / (1.0 + 5e-8)).collect();
        let t0: Vec<f64> = (0..p).map(|_| rng.random::<f64>() - 0.5).collect();
        let ta = sdot(&t0, &ab);
        let mut v: Vec<f64> = (0..p).map(|i| t0[i] - ta * ab[i]).collect();
        let vn = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        for x in v.iter_mut() {
            *x /= vn;
        }
        let mut u: Vec<f64> = (0..p).map(|i| d0.cos() * ab[i] + d0.sin() * v[i]).collect();
        let flat = |_: &[f64]| 0.0;
        let mut accepted = 0usize;
        for _ in 0..4_000 {
            let (nx, take, _) = latitude_jump(&u, 0.0, 0.5 * d0, &anchor, &flat, &mut rng);
            if take {
                u = nx;
                accepted += 1;
            }
        }
        assert!(
            accepted > 3_900,
            "a flat target must accept, got {accepted}"
        );
        let nu = (u.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let cu = sdot(&u, &ab);
        assert!(
            (nu - 1.0).abs() < 1e-12,
            "|u| = {nu}, the move left the sphere"
        );
        // 1e-10, not 1e-16: the latitude drifts 5e-16 per step (rounding in
        // `⟨â,u⟩`), which is `2e-12` over this run and `2e-9` rad — and, unlike
        // the failure this test is for, it does not amplify.  The frozen-chain
        // collapse moves `⟨â,u⟩` by `1 − cos d₀ = 3.7e-7`, seven orders above.
        assert!(
            (cu - d0.cos()).abs() < 1e-10,
            "⟨â,u⟩ = {cu}, the latitude has moved from cos {:.12}",
            d0.cos()
        );
        assert!(d0.cos().acos() <= d0 * 1.001, "sanity on the start");
    }

    /// Batch-means standard error: consecutive draws are correlated, and an iid
    /// bar would reject a correct kernel (the same reason the pCN moment tests
    /// use it).
    fn batch_means_se(vals: &[f64]) -> f64 {
        let batches = 20;
        let n = vals.len() / batches;
        let means: Vec<f64> = (0..batches)
            .map(|i| vals[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
            .collect();
        let m = means.iter().sum::<f64>() / batches as f64;
        let var = means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (batches as f64 - 1.0);
        (var / batches as f64).sqrt()
    }

    /// **The circle.**  A von Mises target `π(θ) ∝ exp(κ cos θ)` must return
    /// `E[u₁] = I₁(κ)/I₀(κ)`.  The circle is the one `P` where the geodesic
    /// volume factor is `^0`, so this test certifies the jump's symmetry but
    /// cannot see a Jacobian error — the `P = 3` case below is what does.
    #[test]
    fn shell_jump_matches_the_von_mises_moment() {
        let kappa = 2.0f64;
        let log_target = |u: &[f64]| kappa * u[0];
        let (kept, rate) = shell_chain(vec![1.0, 0.0], 0.3, &log_target, 60_000, 1_000, 5, 41);
        assert!(rate > 0.25, "acceptance {rate} too low to judge");
        let xs: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let se = batch_means_se(&xs);
        assert!(
            (mean - VON_MISES_K2).abs() < 4.0 * se,
            "shell E[u₁] = {mean}, von Mises says {VON_MISES_K2} (batch-means se {se}, accept {rate})"
        );
    }

    /// **The sphere.**  von Mises–Fisher on `S²` has `E[u₁] = coth κ − 1/κ`; here
    /// the volume factor is `(sin r)`, so a jump whose reverse radius differed
    /// from its forward one — or an exp map that landed at the wrong distance —
    /// would bias this moment and nothing else in the module would notice.
    #[test]
    fn shell_jump_is_exact_on_s2() {
        let kappa = 2.0f64;
        let theory = 1.0 / kappa.tanh() - 1.0 / kappa;
        let log_target = |u: &[f64]| kappa * u[0];
        let (kept, rate) = shell_chain(vec![1.0, 0.0, 0.0], 0.3, &log_target, 60_000, 1_000, 5, 42);
        assert!(rate > 0.25, "acceptance {rate} too low to judge");
        let xs: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let se = batch_means_se(&xs);
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "shell E[u₁] = {mean}, von Mises–Fisher says {theory} (batch-means se {se}, accept {rate})"
        );
    }

    /// **An isotropic Gaussian in the tangent.**  Geodesic coordinates carry the
    /// Haar measure `(sin r)^{P−2} dr dΩ = (sin r / r)^{P−2} dv`, so a target
    /// that is `N(0, σ²I)` *in the chart* is
    ///
    /// ```text
    /// log π(u) = −r²/2σ² − (P−2)·log(sin r / r),   r = d(u, μ),
    /// ```
    ///
    /// and then `E[r²] = (P−1)·σ²` in closed form (the tangent is
    /// `(P−1)`-dimensional, so the chart's radial density is `r^{P−2}e^{−r²/2σ²}`
    /// — a `χ` with `P−1` degrees of freedom, not `P`).  This is the test that pins the
    /// *measure*: the jump's own density carries `1/(sin r)^{P−2}`, so a chain
    /// that used the flat measure, or normalised by the wrong power, moves this
    /// second moment by `O(1)` while `E[u₁]` would barely notice.  (It caught
    /// the opposite error first: with the Jacobian's sign flipped the statistic
    /// reads 0.64 against 1.25 — the sampler was right, the test's target was
    /// not `N(0, σ²I)` in its own chart.)
    #[test]
    fn shell_jump_samples_an_isotropic_gaussian_in_the_tangent() {
        let p = 5usize;
        let sigma = 0.5f64;
        let log_target = move |u: &[f64]| {
            let c = u[0].clamp(-1.0, 1.0);
            let r = c.acos();
            let jac = if r < 1e-9 {
                0.0
            } else {
                -((p - 2) as f64) * (r.sin() / r).ln()
            };
            -r * r / (2.0 * sigma * sigma) + jac
        };
        let mut u0 = vec![0.0; p];
        u0[0] = 1.0;
        let (kept, rate) = shell_chain(u0, 0.4, &log_target, 60_000, 1_000, 5, 43);
        assert!(rate > 0.25, "acceptance {rate} too low to judge");
        let r2: Vec<f64> = kept
            .iter()
            .map(|u| u[0].clamp(-1.0, 1.0).acos().powi(2))
            .collect();
        let mean = r2.iter().sum::<f64>() / r2.len() as f64;
        let theory = (p - 1) as f64 * sigma * sigma;
        let se = batch_means_se(&r2);
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "shell E[r²] = {mean}, N(0, σ²I) on R^{} says {theory} (batch-means se {se}, accept {rate})",
            p - 1
        );
    }

    /// **A restriction on the target, not on the proposal.**  On `S²` the Haar
    /// measure is uniform in `cos θ`, where `θ` is the geodesic radius from the
    /// anchor, so a flat target restricted to `θ ≤ d` has `E[cos θ] = (1 +
    /// cos d)/2` in closed form — one number, no fitting.
    ///
    /// It pins both halves of the claim in [`restrict_to_cap`]: the cap is
    /// enforced (every kept draw is inside), and the ratio is still exact (the
    /// moment matches).  The two failure modes it separates are the ones that
    /// look alike in an acceptance column: censoring the *proposal* (which is
    /// also exact, but with a different law) and censoring the target while
    /// keeping a wrong `(sin r)^{P−2}` volume factor.
    #[test]
    fn cap_restriction_is_haar_on_the_cap() {
        let d = 0.8f64;
        let anchor = vec![1.0, 0.0, 0.0];
        let flat = restrict_to_cap(|_: &[f64]| 0.0, &anchor, d);
        let (kept, rate) = shell_chain(anchor.clone(), 0.3, &flat, 60_000, 1_000, 5, 44);
        assert!(rate > 0.25, "acceptance {rate} too low to judge");
        assert!(
            kept.iter()
                .all(|u| u[0].clamp(-1.0, 1.0).acos() <= d + 1e-12),
            "a kept draw left the cap"
        );
        let cosines: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
        let theory = 0.5 * (1.0 + d.cos());
        let se = batch_means_se(&cosines);
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "capped shell E[cos θ] = {mean}, Haar on the cap says {theory} \
             (batch-means se {se}, accept {rate})"
        );
    }

    /// **The cap has to be several steps wide, or the chain stops.**  This is
    /// the geometric condition under which [`restrict_to_cap`] is usable, and
    /// the exactness argument cannot see it: the ratio is right at any radius,
    /// but a chain sitting on the boundary stays inside only when the step's
    /// inward component beats the sphere's curvature.  From radius `R` a step
    /// `r` lands inside iff `⟨u, ŵ⟩ ≥ r/(2R)`, whose probability on `S^{P−1}`
    /// is `Φ̄(r√P/(2R))` — so the ratio `R/r`, not the radius, is what decides
    /// whether the run moves.
    ///
    /// Two flat targets on `S^{63}`, differing only in that ratio: at `R = r`
    /// the chain takes one step, freezes on the boundary, and reports ~0
    /// acceptance; at `R = 10r` it keeps moving.  `examples/basin_posterior
    /// --cap` with the default `R = r₁` hit exactly the first arm (0.0 %
    /// acceptance over 4 × 10 000 steps), which is what motivated this test.
    #[test]
    fn a_cap_must_be_several_steps_wide_or_the_chain_freezes() {
        let p = 64;
        let r = 0.05;
        let mut anchor = vec![0.0; p];
        anchor[0] = 1.0;

        let tight_cap = restrict_to_cap(|_: &[f64]| 0.0, &anchor, r);
        let (tight, tight_rate) = shell_chain(anchor.clone(), r, &tight_cap, 20_000, 1_000, 5, 9);
        let roomy_cap = restrict_to_cap(|_: &[f64]| 0.0, &anchor, 10.0 * r);
        let (roomy, roomy_rate) = shell_chain(anchor.clone(), r, &roomy_cap, 20_000, 1_000, 5, 9);

        for (kept, radius) in [(&tight, r), (&roomy, 10.0 * r)] {
            assert!(
                kept.iter()
                    .all(|u| u[0].clamp(-1.0, 1.0).acos() <= radius + 1e-12),
                "a kept draw left the cap at R = {radius}"
            );
        }
        assert!(
            tight_rate < 0.01,
            "cap = step should freeze the chain, acceptance {tight_rate}"
        );
        assert!(
            roomy_rate > 0.1,
            "cap = 10 steps should let it move, acceptance {roomy_rate}"
        );
    }

    /// The unrestricted case must be bit-identical to the bare target, so a
    /// caller can wire one closure and vary only the radius.
    #[test]
    fn an_infinite_cap_is_the_bare_target() {
        let anchor = vec![0.6, 0.8];
        let bare = |u: &[f64]| 1.25 * u[0] - 0.5 * u[1];
        let capped = restrict_to_cap(bare, &anchor, f64::INFINITY);
        for u in [[1.0, 0.0], [0.0, 1.0], [-0.6, -0.8], [0.6, 0.8]] {
            assert_eq!(capped(&u), bare(&u));
        }
        assert!(restrict_to_cap(bare, &anchor, 0.1)(&[0.0, 1.0]).is_infinite());
    }

    /// The determinant identity the ratio relies on, checked against a dense
    /// computation at small `P`:
    /// `det(P D P + uuᵀ) = det(D)·(uᵀD⁻¹u)`, i.e. the restriction of the
    /// projected metric has determinant `det(D)·(uᵀD⁻¹u)`.  `det D` cancels
    /// between `u` and `u′` and the remaining factor is what
    /// `step_preconditioned` adds to the ratio.
    #[test]
    fn projected_metric_determinant_matches_the_dense_one() {
        let p = 4;
        let d = vec![1.0, 4.0, 0.25, 2.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        for _ in 0..5 {
            let raw: Vec<f64> = (0..p).map(|_| rng.random::<f64>() - 0.5).collect();
            let n = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let u: Vec<f64> = raw.iter().map(|x| x / n).collect();

            let proj = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 } - u[i] * u[j];
            // m = P D P + uuᵀ, all by explicit matrix products
            let mut m = vec![vec![0.0f64; p]; p];
            for i in 0..p {
                for j in 0..p {
                    // (P D P)_{ij} = Σ_k P_{ik} d_k P_{kj}
                    let pdp: f64 = (0..p).map(|k| proj(i, k) * d[k] * proj(k, j)).sum();
                    m[i][j] = pdp + u[i] * u[j];
                }
            }
            let dense_det = det_dense(&m);
            let det_d = d.iter().product::<f64>();
            let qt: f64 = u.iter().zip(&d).map(|(ui, di)| ui * ui / di).sum();
            let formula = det_d * qt;
            assert!(
                (dense_det - formula).abs() < 1e-9 * formula.abs().max(1.0),
                "dense {dense_det} vs formula {formula}"
            );
        }
    }

    fn det_dense(m: &[Vec<f64>]) -> f64 {
        let n = m.len();
        let mut a = m.to_vec();
        let mut det = 1.0;
        for c in 0..n {
            let piv = (c..n)
                .max_by(|x, y| a[*x][c].abs().partial_cmp(&a[*y][c].abs()).unwrap())
                .unwrap();
            if a[piv][c].abs() < 1e-14 {
                return 0.0;
            }
            if piv != c {
                a.swap(piv, c);
                det = -det;
            }
            det *= a[c][c];
            let (pivot_row, rest) = a.split_at_mut(c + 1);
            let prow = pivot_row[c].clone();
            for row in rest {
                let f = row[c] / prow[c];
                for (x, y) in row.iter_mut().zip(&prow).skip(c) {
                    *x -= f * y;
                }
            }
        }
        det
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

    /// In geodesic normal coordinates the sphere's volume element is
    /// `(sin a / a)^{P−2} dv`, so a proposal that draws `v ∈ T_u` and maps it
    /// through `exp_u` carries that factor — at **both** ends of the ratio.
    /// It cancels iff the reverse displacement has the same geodesic radius,
    /// which is why the kernel can legitimately omit it; this pins the fact so
    /// a future reverse map (a different step, a retraction) cannot start
    /// dropping a term that no longer cancels.
    ///
    /// The factor is `^0` at `P = 2`, so every circle test in this module is
    /// blind to it — the same structural blindness that hid the restricted
    /// metric inverse.
    #[test]
    fn the_exp_map_volume_factor_cancels_because_the_radii_match() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        for p in [2usize, 3, 4, 7] {
            for _ in 0..64 {
                let mut u: Vec<f64> = (0..p).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
                let n = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in u.iter_mut() {
                    *x /= n;
                }
                let v = project_tangent(
                    &(0..p)
                        .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                        .collect::<Vec<f64>>(),
                    &u,
                );
                let a = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if a < 1e-6 {
                    continue;
                }
                let w = reverse_tangent_vec(&u, &v, a);
                let b = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    (b - a).abs() < 1e-12 * a.max(1.0),
                    "P={p}: reverse radius {b} != forward {a}, so the volume factor would not cancel"
                );
                // and the reverse tangent really points back to `u`
                let u_next = exp_map_vec(&u, &v, a);
                let back = exp_map_vec(&u_next, &w, b);
                assert!(
                    (0..p).map(|i| (back[i] - u[i]).powi(2)).sum::<f64>() < 1e-20,
                    "P={p}: walking the reverse tangent must land on u"
                );
            }
        }
    }

    /// Run [`guided_shell_jump`] and keep draws, for the moment tests.
    #[allow(clippy::too_many_arguments)]
    fn guided_chain(
        start: Vec<f64>,
        r: f64,
        kappa_guide: f64,
        direction: &dyn Fn(&[f64]) -> Vec<f64>,
        log_target: &dyn Fn(&[f64]) -> f64,
        steps: usize,
        burn: usize,
        thin: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut u = start;
        let mut lu = log_target(&u);
        let mut kept = Vec::new();
        let mut accepted = 0usize;
        for t in 0..steps {
            let (next, take, ln) =
                guided_shell_jump(&u, lu, r, kappa_guide, direction, log_target, &mut rng);
            if take {
                u = next;
                lu = ln;
                accepted += 1;
            }
            if t >= burn && (t - burn).is_multiple_of(thin) {
                kept.push(u.clone());
            }
        }
        (kept, accepted as f64 / steps as f64)
    }

    /// **The guidance is not free, so the ratio must pay for it.**  A von
    /// Mises–Fisher target on `S²` still has `E[u₁] = coth κ − 1/κ` when the
    /// proposal's mean direction points *orthogonally* to the target's
    /// concentration axis: the two proposal densities differ only by the mean
    /// direction, so a missing or mis-signed `κ(⟨μ′,ŵ′⟩ − ⟨μ,ŵ⟩)` term shows up
    /// here and nowhere else.  (Dropping the reverse term is the restricted-
    /// metric bug's twin: the chain becomes a biased walk with no restoring
    /// law.)
    #[test]
    fn guided_jump_is_exact_on_s2() {
        let kappa = 2.0f64;
        let theory = 1.0 / kappa.tanh() - 1.0 / kappa;
        let log_target = |u: &[f64]| kappa * u[0];
        let anchor = vec![1.0, 0.0, 0.0];
        // Frozen ambient gradient, orthogonal to the target's axis.
        let g = vec![0.0, 1.0, 0.0];
        let direction = |x: &[f64]| drift_direction(&g, x);
        let (kept, rate) = guided_chain(
            anchor.clone(),
            0.5,
            1.5,
            &direction,
            &log_target,
            80_000,
            1_000,
            5,
            71,
        );
        assert!(rate > 0.2, "acceptance {rate} too low to judge");
        let xs: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let se = batch_means_se(&xs);
        assert!(
            (mean - theory).abs() < 4.0 * se,
            "guided E[u₁] = {mean}, vMF says {theory} (batch-means se {se}, accept {rate})"
        );
    }

    /// **A strong guide on a flat target must leave the uniform law uniform.**
    /// With `log π ≡ 0` the posterior is Haar, and the guided jump's only
    /// asymmetry is its mean direction; if the reverse term is wrong, the chain
    /// drifts along `μ` to the pole where `μ` vanishes and `E[u₁]` goes to a
    /// positive number instead of 0.  This is the test that separates "biased
    /// proposal, exact filter" from "biased chain", and it is the one that
    /// catches a field which is wrong but still exact.
    ///
    /// The step is chosen in the regime the kernel is for, `κ ≪ 1` (`κ = 0.5`
    /// here, against a field whose own angular scale is `π/2`).  At `κ = 4` the
    /// same test accepts **1.5–5 %** at every radius tried: the reverse of a
    /// directed step is nearly *antiparallel* to the forward one, so
    /// `log α ≈ −2κ⟨μ,ŵ⟩` and the guidance is only as reversible as it is weak.
    #[test]
    fn guided_jump_keeps_a_flat_target_uniform() {
        let log_target = |_: &[f64]| 0.0;
        let anchor = vec![1.0, 0.0, 0.0];
        let g = vec![0.0, 1.0, 0.0];
        let direction = |x: &[f64]| drift_direction(&g, x);
        let (kept, rate) = guided_chain(
            anchor.clone(),
            0.3,
            0.5,
            &direction,
            &log_target,
            60_000,
            1_000,
            5,
            72,
        );
        assert!(rate > 0.2, "acceptance {rate} too low to judge");
        let xs: Vec<f64> = kept.iter().map(|u| u[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let se = batch_means_se(&xs);
        assert!(
            mean.abs() < 4.0 * se + 5e-3,
            "guided flat-target E[u₁] = {mean} (se {se}, accept {rate}) must be 0: the \
             filter, not the proposal, has to remove the drift"
        );
    }

    /// The guidance has to actually direct the step, or the kernel is a slower
    /// `shell_jump` with extra arithmetic.  The direction lives on `S^{P−2}`, so
    /// at `P = 3` it lives on a **circle** and `E[⟨μ, ŵ⟩]` is the von Mises
    /// moment `I₁(κ)/I₀(κ)` — 0.6978 at `κ = 2` — not the `S²` value
    /// `coth κ − 1/κ = 0.5373`.  (Getting that wrong is how this test first
    /// failed, and it is the same dimension trap the tangent-Gaussian moment
    /// set: the ambient `P` and the dimension of the sphere the direction lives
    /// on differ by one.)  `κ = 0` must read ≈ 0.
    #[test]
    fn guidance_directs_the_jump() {
        let u = vec![1.0, 0.0, 0.0];
        let mu = vec![0.0, 1.0, 0.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(73);
        let n = 2_000usize;
        let mut mean = 0.0;
        for _ in 0..n {
            let w = sample_vmf_tangent(&u, &mu, 2.0, &mut rng);
            mean += sdot(&w, &mu);
        }
        mean /= n as f64;
        let theory = VON_MISES_K2;
        assert!(
            (mean - theory).abs() < 0.05,
            "E[⟨μ,ŵ⟩] = {mean} at κ = 2 on the direction circle, von Mises says {theory}"
        );
        let mut flat = 0.0;
        for _ in 0..n {
            flat += sdot(&sample_vmf_tangent(&u, &mu, 0.0, &mut rng), &mu);
        }
        assert!(flat.abs() / (n as f64) < 0.05, "κ = 0 must be isotropic");
    }

    /// **The drift budget of a guided jump in `P` dimensions.**  A jump of
    /// radius `r` guided at concentration `κ` buys `r·E[⟨μ̂, ŵ⟩]` rad of drift
    /// against `r/√(P−1)` of transverse noise per step, and on the
    /// `(P−2)`-sphere of directions the mean cosine is `A_P(κ) ≈ κ/(P−2)` for
    /// `κ ≪ √P`.  Both halves of the ledger therefore scale with `P`: the drift
    /// per step is `rκ/P` while the noise is `r/√P`, so guidance outruns the
    /// random walk only after `t ≈ (P/κ)²` steps.  That number is what the
    /// status doc's no-go bound quotes, and this test is what pins it: the
    /// linear regime is `κ² ≪ P` (not `κ ≪ 1` — this is where a low-
    /// dimensional intuition gives the wrong answer).  At `κ = 400, P = 2000`
    /// the form reads 0.1927 against 0.2002: the first version of this test
    /// asserted that and was wrong, because `κ²/P = 80` is saturation, not the
    /// linear regime.  The statistics set the sample size — `Var[cos] =
    /// 1/(P−2)` against a mean of `κ/P`, so the relative standard error is
    /// `√(P/(κ²n))` and the design is `n·P`-bound, not quality-bound.
    #[test]
    fn vmf_tangent_mean_cosine_scales_as_kappa_over_p() {
        let p = 8000usize;
        let mut u = vec![0.0; p];
        u[0] = 1.0;
        let mut mu = vec![0.0; p];
        mu[1] = 1.0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(74);
        let n = 2_000usize;
        for kappa in [80.0f64, 160.0] {
            let expect = kappa / (p - 2) as f64;
            let mut sum = 0.0;
            for _ in 0..n {
                sum += sdot(&sample_vmf_tangent(&u, &mu, kappa, &mut rng), &mu);
            }
            let mean = sum / n as f64;
            // Var[cos] = (1−A²)/(P−2) for this family.
            let se = ((1.0 - expect * expect) / (p - 2) as f64 / n as f64).sqrt();
            assert!(
                (mean - expect).abs() < 4.0 * se + 1e-3,
                "E[⟨μ,ŵ⟩] at κ = {kappa}, P = {p}: {mean}, κ/(P−2) = {expect} (se {se})"
            );
        }
    }
}
