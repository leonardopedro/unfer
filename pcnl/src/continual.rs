//! Sequential Bayesian updates for a pCNL posterior.
//!
//! Continual learning here is the recursion
//!
//! ```text
//! π_t(θ) ∝ π_{t-1}(θ) · exp(−λ N_t · mean_loss_t(θ))
//! ```
//!
//! The pCN kernel needs an isotropic standard-normal prior, so the previous
//! posterior is approximated by the diagonal Gaussian `N(μ, diag(σ²))` fitted
//! to its particles.  Reparameterising `θ = μ + σ ⊙ (Bz + w)` turns that prior
//! back into `N(0, I_r) × N(0, I_{P−r})`, and the existing exact full-target
//! pCN kernel samples the new posterior without changing the target.  The
//! approximation is in the prior summary — ignored posterior correlations —
//! not in the kernel.

use crate::subspace::{AffineSubspace, Basis, SubspaceError};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum ContinualError {
    Empty,
    BadLength { expected: usize, got: usize },
    NonFinite,
    BadStd(f32),
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
        }
    }
}

impl std::error::Error for ContinualError {}

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
    fn rejects_bad_particles_and_priors() {
        assert!(DiagonalPosterior::from_particles(&[], 1e-4).is_err());
        assert!(DiagonalPosterior::from_particles(&[vec![0.0], vec![1.0, 2.0]], 1e-4).is_err());
        assert!(DiagonalPosterior::from_particles(&[vec![f32::NAN]], 1e-4).is_err());
        assert!(DiagonalPosterior::isotropic(1, vec![0.0], 0.0).is_err());
    }
}
