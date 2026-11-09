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
use crate::subspace::AffineSubspace;
use crate::task::Dataset;
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
        let step = 0.3 / (1.0 + t as f64).powf(0.6);
        let a = f64::from(u8::from(accepted));
        let log_m = (self.multiplier.ln() + step * (a - self.target_accept)).clamp(-7.0, 2.0);
        self.multiplier = log_m.exp();
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
}
