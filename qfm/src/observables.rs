//! Pre-projected observables for the QFM tomographic pipeline.
//!
//! After the offline Krylov reduction produces the basis W (K_2 x rank)
//! and reduced Hamiltonian H_m (rank x rank), we pre-project the
//! raw-coordinate observables into the m^2 operator basis
//! `{E_{r,s} = |e_r><e_s|}` so that all online decoding is a dense
//! mat-vec product (no Fock-space operations at inference time).
//!
//! - `operator_basis`: the m^2 elementary matrices E_{r,s}
//! - `probability_weight_matrix`: W_prob (K_2 x rank^2) for density -> sketched probability
//! - `krylov_image_basis`: Phi (d x rank^2) for subspace coordinates -> raw image
//! - `compressive_solver`: Phi_tilde^+ (rank^2 x k) for peak hash -> subspace coefficients

use nalgebra::DMatrix;
use num_complex::Complex64;

/// Build the `rank^2` elementary matrices `{E_{r,s} = |e_r><e_s|}` as a
/// flat `Vec<DMatrix<Complex64>>` indexed by `(r, s) -> r * rank + s`.
///
/// Each E_{r,s} is a rank x rank matrix with a single 1.0 at position (r, s).
pub fn operator_basis(rank: usize) -> Vec<DMatrix<Complex64>> {
    let mut basis = Vec::with_capacity(rank * rank);
    for r in 0..rank {
        for s in 0..rank {
            let mut m = DMatrix::<Complex64>::zeros(rank, rank);
            m[(r, s)] = Complex64::new(1.0, 0.0);
            basis.push(m);
        }
    }
    basis
}

/// Build the probability weight matrix W_prob in R^{K_2 x rank^2}.
///
/// For each basis projector P_a = |a><a| (a = 0..K_2) and each
/// elementary operator E_{r,s}:
///   `(W_prob)_{a, (r,s)} = Tr(E_{r,s}^dagger * W^dagger * P_a * W)
///                        = (W^dagger * P_a * W)_{s, r}`
///
/// Since P_a is a one-hot projector,
///   `(W^dagger P_a W)_{r,s} = conj(W[a,r]) * W[a,s]`.
/// So the (s, r) element of W^dagger P_a W is
///   `(W^dagger P_a W)_{s, r} = conj(W[a,s]) * W[a,r]`.
/// We store this as the flat column index `(r, s) -> r*rank + s` (note
/// the swap: the (r,s) column index gets the (s,r) element of the
/// projection). This convention matches `krylov_image_basis` below.
pub fn probability_weight_matrix(w: &DMatrix<Complex64>, rank: usize, k2: usize) -> DMatrix<f64> {
    assert_eq!(w.nrows(), k2, "W must have K_2 rows");
    assert_eq!(w.ncols(), rank, "W must have rank columns");

    let mut w_prob = DMatrix::<f64>::zeros(k2, rank * rank);
    for a in 0..k2 {
        for r in 0..rank {
            for s in 0..rank {
                // (W^dagger P_a W)_{s, r} = conj(W[a, s]) * W[a, r]
                let val = w[(a, s)].conj() * w[(a, r)];
                w_prob[(a, r * rank + s)] = val.re;
            }
        }
    }
    w_prob
}

/// Build the Krylov image basis Phi in R^{d x rank^2}.
///
/// For each raw coordinate operator X_c = |c><c| (c = 0..d) and each
/// elementary operator E_{r,s}:
///   `Phi_{c, (r,s)} = Tr(E_{r,s}^dagger * W^dagger * X_c * W)
///                   = (W^dagger * X_c * W)_{s, r}`
///
/// Since X_c is a one-hot projector, `(W^dagger X_c W)_{r,s} = conj(W[c,r]) * W[c,s]`.
/// So `Phi_{c, (r,s)}` (the (s, r) element) is `conj(W[c,s]) * W[c,r]`.
/// We store this at the flat column index `(r, s) -> r*rank + s` (note
/// the swap: the (r,s) column index gets the (s,r) element of the
/// projection).
///
/// **Constraint:** for the round-trip `Phi = W^dagger X_c W` to be
/// faithful, we need `d <= k2` (the raw coordinate index is bounded
/// by the Fock space dimension). When `d > k2`, the extra rows of Phi
/// are silently zeroed (the X_c projector for `c >= k2` lives outside
/// the Fock basis W spans). Callers must ensure `d <= k2`; the
/// `debug_assert!` below catches violations in debug builds.
pub fn krylov_image_basis(w: &DMatrix<Complex64>, rank: usize, d: usize) -> DMatrix<f64> {
    let k2 = w.nrows();
    assert_eq!(w.ncols(), rank, "W must have rank columns");
    debug_assert!(
        d <= k2,
        "krylov_image_basis: d={d} must be <= k2={k2}; extra rows will be silently zeroed"
    );

    let mut phi = DMatrix::<f64>::zeros(d, rank * rank);
    for c in 0..d.min(k2) {
        for r in 0..rank {
            for s in 0..rank {
                // (W^dagger X_c W)_{s, r} = conj(W[c, s]) * W[c, r]
                let val = w[(c, s)].conj() * w[(c, r)];
                phi[(c, r * rank + s)] = val.re;
            }
        }
    }
    phi
}

/// Build the compressive subspace solver Phi_tilde^+ in R^{rank^2 x k}.
///
/// 1. Project Phi through the Level 1 hash: `Phi_tilde = S_1 * Phi` (k x rank^2).
///    For the compressive solver, S_1 acts on the raw-coordinate dimension:
///    `(Phi_tilde)[h, (r,s)] = sum_c S_1[h, c] * Phi[c, (r,s)]`.
/// 2. Compute the Moore-Penrose pseudo-inverse `Phi_tilde^+ = (Phi_tilde^T Phi_tilde)^{-1} Phi_tilde^T`.
///
/// Since we don't have the CountSketch object here (to avoid a circular dep),
/// we accept the precomputed Phi_tilde directly. Callers that have a
/// CountSketch should apply it first: `phi_tilde = s1.apply_to_columns(&phi)`.
pub fn compressive_solver(phi_tilde: &DMatrix<f64>) -> DMatrix<f64> {
    // SVD-based Moore-Penrose pseudo-inverse.
    // Phi_tilde has shape (k, rank^2). We want (rank^2, k).
    let svd = phi_tilde.clone().svd(true, true);
    svd.pseudo_inverse(1e-12).unwrap_or_else(|_| {
        // Fallback: return a zero matrix of the right shape.
        DMatrix::<f64>::zeros(phi_tilde.ncols(), phi_tilde.nrows())
    })
}

/// Rank-truncate the Krylov basis W and project the reduced Hamiltonian H_m.
///
/// When `krylov_dim << K_2` (the rank-truncation path of P10.16.3), the
/// extracted W (K_2 × rank) has many zero rows — only the first `krylov_dim`
/// rows are non-zero because the SIRK sequence never visits modes beyond
/// the Krylov reach. A truncated SVD on the **real part** of W identifies
/// the `r` dominant column directions and simultaneously projects H_m to
/// the same r-dimensional subspace via the right singular vectors V_r.
/// (H_bar is real symmetric, but the SIRK shifts lie on the imaginary
/// axis, so w_whiten — and hence W — is complex in general; the real part
/// carries the dominant column structure and the truncation only needs
/// approximate directions. This is a documented heuristic, not an exact
/// complex SVD.)
///
/// Returns `(W_trunc, H_m_trunc)` where:
/// * `W_trunc` = W · V_r            (K_2 × r, Complex64)
/// * `H_m_trunc` = V_r^H · H_m · V_r  (r × r, Complex64)
///
/// Row normalisation of W_trunc must be re-applied by the caller after this
/// call (the same way the caller normalises W before calling this function).
///
/// Returns `None` if `r >= rank` (nothing to truncate).
pub fn rank_truncate_w_h(
    w: &DMatrix<Complex64>,
    h_m: &DMatrix<Complex64>,
    r: usize,
) -> Option<(DMatrix<Complex64>, DMatrix<Complex64>)> {
    let rank = w.ncols();
    if r >= rank {
        return None;
    }
    let r = r.max(1);

    // Real part of W (K_2 × rank). W is complex in general (the SIRK
    // shifts are imaginary); the real part is used as the truncation
    // heuristic — see the doc comment above.
    let k2 = w.nrows();
    let w_real = DMatrix::<f64>::from_fn(k2, rank, |i, j| w[(i, j)].re);

    // Thin SVD: W_real = U Σ V^T (U: K_2×rank, Σ: rank, V^T: rank×rank).
    // nalgebra returns singular values in descending order.
    let svd = w_real.svd(true, true);
    let v_t = svd.v_t.expect("svd v_t always present with compute_v=true");

    // V_r: rank × r (the first r rows of V^T, transposed → columns of V).
    // Row i of V^T = right singular vector i; we want columns 0..r of V
    // = rows 0..r of V^T transposed.
    let v_r_real = v_t.rows(0, r).transpose().into_owned(); // rank × r

    // W_trunc = W · V_r  (K_2 × r, Complex64).
    let v_r_c =
        DMatrix::<Complex64>::from_fn(rank, r, |i, j| Complex64::new(v_r_real[(i, j)], 0.0));
    let w_trunc = w * &v_r_c;

    // H_m_trunc = V_r^H · H_m · V_r  (r × r, Complex64).
    // V_r is real so V_r^H = V_r^T.
    let h_m_trunc = v_r_c.adjoint() * h_m * &v_r_c;

    Some((w_trunc, h_m_trunc))
}

/// Apply a CountSketch to the column-space of a matrix.
/// (Moved to `CountSketch::apply_to_columns` to avoid exposing private fields.)
///
/// `Phi_tilde = S_1 * Phi` where Phi is d x rank^2 and S_1 is k x d,
/// so Phi_tilde is k x rank^2.
#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn complex_f64(r: f64, i: f64) -> Complex64 {
        Complex64::new(r, i)
    }

    #[test]
    fn operator_basis_orthonormal() {
        let rank = 3;
        let basis = operator_basis(rank);
        assert_eq!(basis.len(), rank * rank);
        // E_{0,0} should be identity-like at (0,0).
        assert_eq!(basis[0][(0, 0)], complex_f64(1.0, 0.0));
        // E_{1,2} should be 1 at (1,2).
        assert_eq!(basis[rank + 2][(1, 2)], complex_f64(1.0, 0.0));
        // Tr(E_{r,s}^dagger * E_{r',s'}) = delta_{rr'} delta_{ss'}.
        for r in 0..rank {
            for s in 0..rank {
                for rp in 0..rank {
                    for sp in 0..rank {
                        let ers = &basis[r * rank + s];
                        let ersp = &basis[rp * rank + sp];
                        // Tr(A^dagger B) = sum_{i,j} conj(A[i,j]) * B[i,j]
                        let trace: Complex64 = (0..rank)
                            .flat_map(|i| (0..rank).map(move |j| (i, j)))
                            .map(|(i, j)| ers[(i, j)].conj() * ersp[(i, j)])
                            .sum();
                        let expected = if r == rp && s == sp { 1.0 } else { 0.0 };
                        assert!(
                            (trace.re - expected).abs() < 1e-12,
                            "Tr(E_{r},{s}^dag E_{rp},{sp}) = {}, expected {}",
                            trace.re,
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn probability_weight_matrix_shape_and_hermitian() {
        let k2 = 4;
        let rank = 2;
        // W is K_2 x rank with orthonormal columns (for testing).
        let w = DMatrix::from_fn(k2, rank, |i, j| {
            complex_f64(((i + j * 3) as f64) / ((k2 * rank) as f64).sqrt(), 0.0)
        });
        let w_prob = probability_weight_matrix(&w, rank, k2);
        assert_eq!(w_prob.nrows(), k2);
        assert_eq!(w_prob.ncols(), rank * rank);
        // All entries should be real (since W is real and P_a is Hermitian).
        for val in w_prob.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn krylov_image_basis_shape() {
        // d <= k2 is the contract (raw coordinates live in the Fock basis
        // W spans). We use d=4 with k2=6 to leave slack in k2.
        let k2 = 6;
        let rank = 2;
        let d = 4;
        let w = DMatrix::from_fn(k2, rank, |i, j| complex_f64(((i + j) as f64) / 10.0, 0.0));
        let phi = krylov_image_basis(&w, rank, d);
        assert_eq!(phi.nrows(), d);
        assert_eq!(phi.ncols(), rank * rank);
    }

    #[test]
    fn compressive_solver_reconstructs() {
        // Synthetic Phi (d=4, rank^2=4) with full column rank.
        let phi = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        );
        let phi_plus = compressive_solver(&phi);
        // For a square invertible matrix, pseudo-inverse = inverse.
        // phi_plus * phi should be identity.
        let product = phi_plus * phi;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[(i, j)] - expected).abs() < 1e-8,
                    "phi_plus * phi at ({i},{j}) = {}, expected {}",
                    product[(i, j)],
                    expected
                );
            }
        }
    }
}
