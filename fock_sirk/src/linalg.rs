//! Robust linear-algebra helpers for the SIRK solver.
//!
//! The original solver orthonormalized the Krylov basis with a bare Cholesky
//! factorization of the Gram matrix and `.expect()`ed positive-definiteness.
//! In practice the Krylov vectors become numerically linearly dependent (e.g.
//! when a shift reproduces an earlier vector), the Gram matrix loses rank, and
//! the Cholesky panics. [`whiten_gram`] replaces that with a rank-revealing
//! Hermitian eigendecomposition: it drops the numerically-null directions and
//! returns a whitening transform `W` such that `Wᴴ G W = I_r`.

use nalgebra::DMatrix;
use num_complex::Complex64;
use thiserror::Error;

/// Default relative tolerance for dropping near-null Gram eigenvalues.
pub const GRAM_REL_TOL: f64 = 1e-12;

/// Errors produced by the SIRK solver and its numerical kernels.
#[derive(Debug, Error)]
pub enum SirkError {
    /// The Gram matrix has no numerically-significant eigenvalues.
    #[error("Gram matrix is numerically rank-zero (max eigenvalue {max_eig:.3e})")]
    GramDegenerate { max_eig: f64 },
    /// A quantum state grew past the configured component budget.
    #[error("state explosion: {components} components exceed the limit of {limit}")]
    StateExplosion { components: usize, limit: usize },
    /// The matrix-free BRST projection did not converge.
    #[error("BRST projection failed to converge (residual {residual:.3e})")]
    BrstNotConverged { residual: f64 },
    /// Any other numerical failure.
    #[error("numeric error: {0}")]
    Numeric(String),
}

/// A whitening transform for a Hermitian positive-semidefinite Gram matrix.
///
/// `w` is `n x rank`; it satisfies `wᴴ · G · w = I_rank` (the columns span the
/// numerically non-null subspace). `dropped = n - rank` directions were removed.
#[derive(Debug)]
pub struct Whitening {
    pub w: DMatrix<Complex64>,
    pub rank: usize,
    pub dropped: usize,
}

/// Whiten a Hermitian positive-semidefinite Gram matrix.
///
/// Computes the Hermitian eigendecomposition `G = U Λ Uᴴ`, keeps the eigenpairs
/// with `λ_i > rel_tol · λ_max`, and returns `W = U_r Λ_r^{-1/2}`. Returns
/// [`SirkError::GramDegenerate`] only if every eigenvalue is non-positive (rank 0).
pub fn whiten_gram(g: &DMatrix<Complex64>, rel_tol: f64) -> Result<Whitening, SirkError> {
    let n = g.nrows();
    assert_eq!(n, g.ncols(), "Gram matrix must be square");

    // Symmetrize to the Hermitian part to absorb floating-point asymmetry.
    let herm = (g + g.adjoint()) * Complex64::new(0.5, 0.0);
    let eig = herm.symmetric_eigen();
    let eigvals = &eig.eigenvalues; // real (f64) — Hermitian spectrum
    let eigvecs = &eig.eigenvectors; // complex columns

    let max_eig = eigvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Reject a non-positive *or* NaN largest eigenvalue (a NaN spectrum means the
    // Gram matrix is numerically unusable, so treat it as degenerate too).
    if max_eig.is_nan() || max_eig <= 0.0 {
        return Err(SirkError::GramDegenerate { max_eig });
    }

    let threshold = rel_tol * max_eig;
    let cols: Vec<usize> = (0..n).filter(|&i| eigvals[i] > threshold).collect();
    let rank = cols.len();
    if rank == 0 {
        return Err(SirkError::GramDegenerate { max_eig });
    }

    let mut w = DMatrix::zeros(n, rank);
    for (l, &i) in cols.iter().enumerate() {
        let inv_sqrt = Complex64::new(1.0 / eigvals[i].sqrt(), 0.0);
        for r in 0..n {
            w[(r, l)] = eigvecs[(r, i)] * inv_sqrt;
        }
    }

    Ok(Whitening {
        w,
        rank,
        dropped: n - rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    #[test]
    fn whiten_identity_is_full_rank() {
        let g = DMatrix::<Complex64>::identity(3, 3);
        let wh = whiten_gram(&g, GRAM_REL_TOL).unwrap();
        assert_eq!(wh.rank, 3);
        assert_eq!(wh.dropped, 0);
        // Wᴴ G W = I
        let recon = wh.w.adjoint() * &g * &wh.w;
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((recon[(i, j)].re - expect).abs() < 1e-10);
                assert!(recon[(i, j)].im.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn whiten_rank_deficient_drops_null_direction() {
        // [[1,1],[1,1]] has eigenvalues {2, 0}: rank 1, no panic.
        let g = DMatrix::from_row_slice(2, 2, &[c(1.0), c(1.0), c(1.0), c(1.0)]);
        let wh = whiten_gram(&g, GRAM_REL_TOL).unwrap();
        assert_eq!(wh.rank, 1);
        assert_eq!(wh.dropped, 1);
        // Wᴴ G W = I_1
        let recon = wh.w.adjoint() * &g * &wh.w;
        assert_eq!(recon.nrows(), 1);
        assert!((recon[(0, 0)].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn whiten_zero_matrix_is_degenerate() {
        let g = DMatrix::<Complex64>::zeros(2, 2);
        let err = whiten_gram(&g, GRAM_REL_TOL).unwrap_err();
        matches!(err, SirkError::GramDegenerate { .. });
    }
}
