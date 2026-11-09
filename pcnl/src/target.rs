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
