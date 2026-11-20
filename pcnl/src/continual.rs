//! Sequential Bayesian updates for a pCNL posterior.
//!
//! Continual learning here is the recursion
//!
//! ```text
//! π_t(θ) ∝ π_{t-1}(θ) · exp(−λ N_t · mean_loss_t(θ))
//! ```
//!
//! The pCN kernel needs an isotropic standard-normal prior, so the previous
//! posterior is approximated by a Gaussian fitted to its particles.
//! [`DiagonalPosterior`] keeps the mean and per-coordinate width;
//! [`LowRankPosterior`] keeps the top principal directions and an isotropic
//! residual.  Reparameterising the chart turns either prior back into
//! `N(0, I_r) × N(0, I_{P−r})`, and the existing exact full-target pCN kernel
//! samples the new posterior without changing the target.  The approximation is
//! in the prior summary, not in the kernel.
//!
//! The low-rank summary was measured as an alternative, not adopted as the
//! default: at a full replay budget it preserved old classes no better than the
//! diagonal summary and cost final accuracy.  See R11 in `RESEARCH_PLAN.md`.

use crate::subspace::{AffineSubspace, Basis, SubspaceError, pca_basis};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum ContinualError {
    Empty,
    BadLength { expected: usize, got: usize },
    NonFinite,
    BadStd(f32),
    Basis(SubspaceError),
}

impl fmt::Display for ContinualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "posterior summary needs at least one particle"),
            Self::BadLength { expected, got } => {
                write!(f, "particle has {got} values, expected {expected}")
            }
            Self::NonFinite => write!(f, "particle or prior contains a non-finite value"),
            Self::BadStd(s) => write!(f, "standard deviation {s} is not finite and > 0"),
            Self::Basis(e) => write!(f, "posterior basis: {e}"),
        }
    }
}

impl std::error::Error for ContinualError {}

impl From<SubspaceError> for ContinualError {
    fn from(e: SubspaceError) -> Self {
        Self::Basis(e)
    }
}

/// A diagonal Gaussian fitted to posterior particles.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalPosterior {
    mean: Vec<f32>,
    std: Vec<f32>,
    n: usize,
}

impl DiagonalPosterior {
    /// A prior with a fixed mean and isotropic scale.  `sigma` is the prior
    /// standard deviation in parameter units.
    pub fn isotropic(dim: usize, mean: Vec<f32>, sigma: f32) -> Result<Self, ContinualError> {
        if dim == 0 || mean.len() != dim {
            return Err(ContinualError::BadLength {
                expected: dim,
                got: mean.len(),
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(ContinualError::BadStd(sigma));
        }
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(ContinualError::NonFinite);
        }
        Ok(Self {
            mean,
            std: vec![sigma; dim],
            n: 0,
        })
    }

    /// Fit `mean` and the unbiased per-coordinate standard deviation from a
    /// non-empty set of particles.  `min_std` floors coordinates that did not
    /// move in the sample, because `AffineSubspace` requires a positive scale.
    pub fn from_particles(particles: &[Vec<f32>], min_std: f32) -> Result<Self, ContinualError> {
        let first = particles.first().ok_or(ContinualError::Empty)?;
        if first.is_empty() {
            return Err(ContinualError::Empty);
        }
        if !min_std.is_finite() || min_std <= 0.0 {
            return Err(ContinualError::BadStd(min_std));
        }
        let p = first.len();
        if particles.iter().any(|row| row.len() != p) {
            return Err(ContinualError::BadLength {
                expected: p,
                got: particles
                    .iter()
                    .find(|row| row.len() != p)
                    .map_or(0, Vec::len),
            });
        }
        if particles.iter().flatten().any(|v| !v.is_finite()) {
            return Err(ContinualError::NonFinite);
        }
        let n = particles.len();
        let mut mean = vec![0.0f64; p];
        for row in particles {
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += f64::from(v);
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = vec![0.0f64; p];
        if n > 1 {
            for row in particles {
                for (i, &v) in row.iter().enumerate() {
                    let d = f64::from(v) - mean[i];
                    var[i] += d * d;
                }
            }
            for v in &mut var {
                *v /= (n - 1) as f64;
            }
        }
        let mean32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
        let std32: Vec<f32> = var
            .iter()
            .map(|&v| (v.max(0.0).sqrt() as f32).max(min_std))
            .collect();
        Ok(Self {
            mean: mean32,
            std: std32,
            n,
        })
    }

    /// The affine chart whose `z ~ N(0, I)` prior is this diagonal Gaussian:
    /// `θ(z) = mean + std ⊙ Bz`.
    pub fn subspace(&self, basis: Basis) -> Result<AffineSubspace, SubspaceError> {
        AffineSubspace::new(self.mean.clone(), basis, self.std.clone())
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn std(&self) -> &[f32] {
        &self.std
    }

    pub fn len(&self) -> usize {
        self.mean.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Number of particles behind the summary; `0` for a fixed prior.
    pub fn particles(&self) -> usize {
        self.n
    }
}

/// A low-rank-plus-isotropic Gaussian fitted to posterior particles.
///
/// The top `rank` principal directions of the particle cloud carry their own
/// per-direction widths; the remaining variance is one isotropic residual.
/// Under `z, w ~ N(0, I)` this is the prior covariance
/// `B diag(col_scale²) Bᵀ + noise² P_⊥`, which keeps the correlations the
/// diagonal summary throws away while still fitting the full-target chart.
/// It is an opt-in alternative, not the measured default; on this fixture it
/// was tighter than useful and cost accuracy when replay already carried the
/// old classes.
#[derive(Clone, Debug)]
pub struct LowRankPosterior {
    mean: Vec<f32>,
    basis: Basis,
    col_scale: Vec<f32>,
    noise: f32,
    n: usize,
}

impl LowRankPosterior {
    /// Fit the mean, the top-`rank` principal directions, their sample widths,
    /// and the residual isotropic width.  `min_std` floors zero-width
    /// directions and the residual, because the chart requires positive scales.
    pub fn from_particles(
        particles: &[Vec<f32>],
        rank: usize,
        seed: u64,
        min_std: f32,
    ) -> Result<Self, ContinualError> {
        let first = particles.first().ok_or(ContinualError::Empty)?;
        let p = first.len();
        if p == 0 {
            return Err(ContinualError::Empty);
        }
        if !min_std.is_finite() || min_std <= 0.0 {
            return Err(ContinualError::BadStd(min_std));
        }
        if rank == 0 || rank > p {
            return Err(ContinualError::Basis(SubspaceError::BadRank { p, r: rank }));
        }
        if particles.iter().any(|row| row.len() != p) {
            return Err(ContinualError::BadLength {
                expected: p,
                got: particles
                    .iter()
                    .find(|row| row.len() != p)
                    .map_or(0, Vec::len),
            });
        }
        if particles.iter().flatten().any(|v| !v.is_finite()) {
            return Err(ContinualError::NonFinite);
        }
        let n = particles.len();
        let mut mean = vec![0.0f64; p];
        for row in particles {
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += f64::from(v);
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mean32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
        let basis = pca_basis(particles, rank, seed)?;

        // Project every particle on the fitted directions, then measure what is
        // left.  The residual variance is pooled over the complement, whose
        // dimension is `p - rank`.
        let mut proj = vec![0.0f64; rank];
        let mut sum_sq = vec![0.0f64; rank];
        let mut resid_sum = 0.0f64;
        for row in particles {
            let mut row_sq = 0.0f64;
            for (i, &v) in row.iter().enumerate() {
                let d = f64::from(v) - mean[i];
                row_sq += d * d;
            }
            let mut explained = 0.0f64;
            for (j, proj_j) in proj.iter_mut().enumerate() {
                let col = basis.col(j);
                *proj_j = (0..p)
                    .map(|i| (f64::from(row[i]) - mean[i]) * f64::from(col[i]))
                    .sum();
                explained += *proj_j * *proj_j;
                sum_sq[j] += *proj_j * *proj_j;
            }
            resid_sum += (row_sq - explained).max(0.0);
        }
        let col_scale: Vec<f32> = sum_sq
            .iter()
            .map(|&s| {
                let var = if n > 1 { s / (n - 1) as f64 } else { 0.0 };
                (var.max(0.0).sqrt() as f32).max(min_std)
            })
            .collect();
        let df = (p - rank).max(1) as f64;
        let residual_var = if n > 1 {
            resid_sum / (n as f64 * df)
        } else {
            0.0
        };
        let noise = (residual_var.max(0.0).sqrt() as f32).max(min_std);

        Ok(Self {
            mean: mean32,
            basis,
            col_scale,
            noise,
            n,
        })
    }

    /// The chart whose `z, w ~ N(0, I)` prior is this Gaussian.
    pub fn subspace(&self) -> Result<AffineSubspace, SubspaceError> {
        AffineSubspace::low_rank(
            self.mean.clone(),
            self.basis.clone(),
            self.col_scale.clone(),
            self.noise,
        )
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn col_scale(&self) -> &[f32] {
        &self.col_scale
    }

    pub fn noise(&self) -> f32 {
        self.noise
    }

    pub fn basis(&self) -> &Basis {
        &self.basis
    }

    pub fn len(&self) -> usize {
        self.mean.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    pub fn particles(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_mean_and_unbiased_std() {
        let draws = vec![vec![1.0f32, 10.0], vec![2.0, 10.0], vec![3.0, 10.0]];
        let p = DiagonalPosterior::from_particles(&draws, 1e-6).unwrap();
        assert_eq!(p.mean(), &[2.0, 10.0]);
        assert!((p.std()[0] - 1.0).abs() < 1e-6);
        assert_eq!(p.std()[1], 1e-6);
        assert_eq!(p.particles(), 3);
    }

    #[test]
    fn single_particle_is_floored_and_subspace_lifts_to_mean() {
        let p = DiagonalPosterior::from_particles(&[vec![0.3, -0.4]], 1e-4).unwrap();
        assert_eq!(p.std(), &[1e-4, 1e-4]);
        let basis = Basis::random(2, 1, 7).unwrap();
        let sub = p.subspace(basis).unwrap();
        assert_eq!(sub.theta_ref(), p.mean());
        assert_eq!(sub.lift(&[0.0]), p.mean());
    }

    #[test]
    fn low_rank_fits_correlated_particles_and_exposes_the_chart() {
        // Perfectly correlated 2-D cloud: one direction explains it all.
        let draws = vec![
            vec![0.0f32, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let p = LowRankPosterior::from_particles(&draws, 1, 7, 1e-6).unwrap();
        assert_eq!(p.mean(), &[1.5, 1.5]);
        assert_eq!(p.particles(), 4);
        assert!(p.col_scale()[0] > 0.0);
        assert!(p.noise() >= 1e-6);
        let sub = p.subspace().unwrap();
        assert_eq!(sub.theta_ref(), p.mean());
        let z = [0.75f32];
        let lift = sub.lift(&z);
        let col = p.basis().col(0);
        for i in 0..2 {
            let want = p.mean()[i] + col[i] * p.col_scale()[0] * z[0];
            assert!((lift[i] - want).abs() < 1e-6, "lift[{i}] = {}", lift[i]);
        }
        let w = [0.0f32; 2];
        let full = sub.lift_full(&z, &w);
        for i in 0..2 {
            assert!((full[i] - lift[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn low_rank_rejects_bad_inputs() {
        assert!(LowRankPosterior::from_particles(&[], 1, 7, 1e-4).is_err());
        assert!(LowRankPosterior::from_particles(&[vec![0.0]], 2, 7, 1e-4).is_err());
        assert!(LowRankPosterior::from_particles(&[vec![0.0], vec![1.0]], 1, 7, 0.0).is_err());
    }

    #[test]
    fn rejects_bad_particles_and_priors() {
        assert!(DiagonalPosterior::from_particles(&[], 1e-4).is_err());
        assert!(DiagonalPosterior::from_particles(&[vec![0.0], vec![1.0, 2.0]], 1e-4).is_err());
        assert!(DiagonalPosterior::from_particles(&[vec![f32::NAN]], 1e-4).is_err());
        assert!(DiagonalPosterior::isotropic(1, vec![0.0], 0.0).is_err());
    }
}
