//! Pure, dependency-free model of the SIRK numeric core (plan T9).
//!
//! This crate is the **Aeneas seam** of `MASS_GAP_CERTIFIED.md` §5.3 /
//! `CONSOLIDATED_PLAN.md` §13.7 T9: the pure numeric core of the probability
//! kernel, written in the Aeneas-supported Rust subset (no interior
//! mutability, no `unsafe`, no raw pointers, no external crates), so that
//! `charon --preset=aeneas` + `aeneas -backend lean` translate it into a
//! Lean 4 model mechanically.
//!
//! Honesty boundary (from the plan): the dense eigendecomposition (LAPACK
//! style) stays a trusted component whose backward error is exactly the T3
//! bound, and the `f64` arithmetic is the rounding layer enclosed by T1–T5.
//! Aeneas therefore verifies the **algorithm** — the fold structure of the
//! forward sequence, the index bookkeeping of the Gram assembly and the
//! projection identity, the shape of the whitening transform and of the
//! residual formula — while the `f64` leaves remain opaque in the generated
//! model. The algebraic identities (projection identity, Gram Hermitian
//! symmetry, residual `e_m = τ_m c_{m−1}`) are the theorems the Lean 4
//! specialist proves *against* this model.
//!
//! The model mirrors the running kernel (`fock_sirk/src/forward_sirk.rs`,
//! `fock_sirk/src/linalg.rs`): every function below is the same computation
//! as the corresponding kernel step, with the same names and the same
//! contracts, but without `nalgebra`, `candle`, GPU or I/O.

/// Complex scalar as an explicit `(re, im)` pair — dependency-free so Charon
/// can extract it, and the conjugation structure is explicit for proofs.
/// Mirrors `num_complex::Complex64` in the kernel.
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    /// Complex conjugation: `conj(a + ib) = a − ib`.  This is the only
    /// operation the Gram assembly needs beyond `+`, `*` — the Hermitian
    /// structure is explicit.
    pub fn conj(&self) -> C64 {
        C64 {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Clone for C64 {
    fn clone(&self) -> C64 {
        C64 {
            re: self.re,
            im: self.im,
        }
    }
}

/// One forward step of the Krylov recurrence:
///
/// ```text
///   w_{k+1} = (H − z_k I) w_k
/// ```
///
/// `h` is the Hamiltonian as a dense `dim × dim` matrix in row-major order
/// (`h[i * dim + j] = ⟨i|H|j⟩`), `w` the current vector, `z` the shift at
/// step `k`, `dim` the Hilbert-space dimension.  This is the **fold body**
/// of the forward sequence: the plan's T9 target "the forward sequence
/// `w_k = (H − z_k)w_{k−1}` as a fold".
///
/// The `f64` arithmetic here is the rounding layer (T1–T5); the *index
/// bookkeeping* (which entries of `h` and `w` combine) is what the generated
/// model carries verbatim.
pub fn forward_step(h: &[C64], w: &[C64], z: &C64, dim: usize) -> Vec<C64> {
    let mut out: Vec<C64> = Vec::with_capacity(dim);
    let mut i = 0;
    while i < dim {
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        let mut j = 0;
        while j < dim {
            // acc += h[i*dim + j] * w[j]
            let hh = &h[i * dim + j];
            let ww = &w[j];
            acc_re += hh.re * ww.re - hh.im * ww.im;
            acc_im += hh.re * ww.im + hh.im * ww.re;
            j += 1;
        }
        // acc -= z * w[i]
        out.push(C64 {
            re: acc_re - (z.re * w[i].re - z.im * w[i].im),
            im: acc_im - (z.re * w[i].im + z.im * w[i].re),
        });
        i += 1;
    }
    out
}

/// The forward sequence as a fold over `k ∈ 0..m`:
///
/// ```text
///   w₀ = v₀,   w_{k+1} = (H − z_k I) w_k   (k = 0 .. m−1)
/// ```
///
/// Returns the `m + 1` Krylov vectors `w₀, …, w_m` — the `m`-dimensional
/// Krylov basis of the SIRK subspace.  This is the plan's "forward sequence
/// as a fold": the loop below *is* the fold, one `forward_step` per shift.
pub fn forward_sequence(h: &[C64], v0: &[C64], z: &[C64], m: usize, dim: usize) -> Vec<Vec<C64>> {
    let mut ws: Vec<Vec<C64>> = Vec::with_capacity(m + 1);
    let mut v: Vec<C64> = Vec::with_capacity(dim);
    let mut i = 0;
    while i < dim {
        v.push(C64 {
            re: v0[i].re,
            im: v0[i].im,
        });
        i += 1;
    }
    ws.push(v);
    let mut k = 0;
    while k < m {
        let prev: &Vec<C64> = &ws[k];
        let next: Vec<C64> = forward_step(h, prev, &z[k], dim);
        ws.push(next);
        k += 1;
    }
    ws
}

/// Gram entry: `G_jk = ⟨w_j | w_k⟩ = Σ_i conj(w_j[i]) · w_k[i]`.
///
/// The Hermitian inner product with the convention that conjugation sits on
/// the *left* argument (bra-vector), matching the kernel's
/// `TensorState::inner_product` and the chapter's `⟨w_j|w_k⟩` notation.
pub fn gram_entry(wj: &[C64], wk: &[C64], dim: usize) -> C64 {
    let mut acc_re: f64 = 0.0;
    let mut acc_im: f64 = 0.0;
    let mut i = 0;
    while i < dim {
        let c = C64 {
            re: wj[i].re,
            im: -wj[i].im,
        };
        // acc += conj(wj[i]) * wk[i]
        acc_re += c.re * wk[i].re - c.im * wk[i].im;
        acc_im += c.re * wk[i].im + c.im * wk[i].re;
        i += 1;
    }
    C64 {
        re: acc_re,
        im: acc_im,
    }
}

/// The Gram matrix `G` of the Krylov basis: `G_jk = ⟨w_j | w_k⟩` for
/// `j, k ∈ 0..m`, stored dense in row-major order (size `(m+1)²`).
///
/// Only the structure is materialized here: the Hermitian symmetry
/// `G_kj = conj(G_jk)` is a theorem the specialist proves against this
/// model (it is not asserted).
pub fn gram_assembly(ws: &[Vec<C64>], m: usize, dim: usize) -> Vec<C64> {
    let n = m + 1;
    let mut g: Vec<C64> = Vec::with_capacity(n * n);
    let mut t = 0;
    while t < n * n {
        let j = t / n;
        let k = t % n;
        let wj: &Vec<C64> = &ws[j];
        let wk: &Vec<C64> = &ws[k];
        let e = gram_entry(wj, wk, dim);
        g.push(C64 {
            re: e.re,
            im: e.im,
        });
        t += 1;
    }
    g
}

/// The projection identity (T9's second target):
///
/// ```text
///   H_jk = ⟨w_j | H | w_k⟩ = σ_{k+1} · G_{j,k+1} + z_k · G_{j,k}
/// ```
///
/// with `G_{j,k+1}` the Gram entry of `w_j` against the *next* Krylov vector
/// and `σ_{k+1}` the step scale (1 in the canonical frame, ‖w_{k+1}‖ in the
/// unit-norm frame).  `g_j_kp1` and `g_j_k` are the two Gram entries; the
/// identity is the pointwise combination the kernel applies in
/// `forward_sirk.rs` (`h_proj_raw[(j,k)] = scales[k+1]·G[j,k+1] +
/// shifts[k]·G[j,k]`).
pub fn projection_identity(g_j_kp1: &C64, g_j_k: &C64, scale_kp1: f64, z_k: &C64) -> C64 {
    C64 {
        re: scale_kp1 * g_j_kp1.re + (z_k.re * g_j_k.re - z_k.im * g_j_k.im),
        im: scale_kp1 * g_j_kp1.im + (z_k.re * g_j_k.im + z_k.im * g_j_k.re),
    }
}

/// The whitening transform `T` with `T* Ĝ T = I` (T9's fourth target).
///
/// The dense eigendecomposition `Ĝ = V diag(λ) V*` is the trusted LAPACK
/// component (backward error exactly the T3 bound); the *assembly* `T = V ·
/// diag(λ^{-1/2})` is what this model materializes.  `eigvals` holds the
/// (real, non-negative, sorted descending) eigenvalues and `eigvecs` the
/// corresponding eigenvectors as columns of a dense `n × n` matrix in
/// row-major order, where `n = m + 1`.  `rank` is the number of eigenvalues
/// above the threshold `rel_tol · λ_max` (the Gram whitening of
/// `linalg::whiten_gram`).
///
/// Returns `T` as a dense `n × rank` matrix (row-major), the *whitened*
/// change of basis; the identity `T* Ĝ T = I` is a theorem of the model.
pub fn whitening_transform(
    eigvals: &[f64],
    eigvecs: &[C64],
    n: usize,
    rank: usize,
) -> Vec<C64> {
    let mut t: Vec<C64> = Vec::with_capacity(n * rank);
    let mut r = 0;
    while r < n {
        let mut l = 0;
        while l < rank {
            // T[r][l] = eigvecs[r][l] / sqrt(eigvals[l])
            let ev = C64 { re: eigvecs[r * n + l].re, im: eigvecs[r * n + l].im };
            let inv_sqrt = 1.0 / eigvals[l].sqrt();
            t.push(C64 {
                re: ev.re * inv_sqrt,
                im: ev.im * inv_sqrt,
            });
            l += 1;
        }
        r += 1;
    }
    t
}

/// The residual formula (T9's fifth target):
///
/// ```text
///   e_j = d_j − θ ĉ_j   (j < m),      e_m = τ_m · ĉ_{m−1},
///   ‖Hψ − θψ‖² = e† G e,
/// ```
///
/// where `d` are the big-space coordinates of `Hψ` in the Krylov basis,
/// `c_hat` the whitened coefficient vector, `theta` the Ritz value, `m` the
/// Krylov rank.  The one component `H` pushes **outside** the projected
/// basis is `e_m = τ_m ĉ_{m−1}` — "the residual formula `‖r‖ = |τ_m
/// c_{m−1}|`" of the plan.  The function returns that boundary component
/// (its norm is the physical content of the residual; the full `e† G e`
/// quadratic form is assembled in the kernel from the Gram matrix).
pub fn residual_boundary_component(c_hat_last: &C64, scale_m: f64) -> C64 {
    C64 {
        re: scale_m * c_hat_last.re,
        im: scale_m * c_hat_last.im,
    }
}

/// The residual norm squared `‖Hψ − θψ‖² = e† G e` over the big Gram,
/// computed from the boundary residual vector `e` (length `m + 1`) and the
/// Gram matrix `g` (dense, row-major, size `(m+1)²`).  Cancellation-free
/// form: the quadratic form is assembled before any square root, exactly as
/// `ForwardSirkResult::ritz_abs_residuals` does.
pub fn residual_norm2(e: &[C64], g: &[C64], m: usize) -> f64 {
    let n = m + 1;
    let mut acc: f64 = 0.0;
    let mut t = 0;
    while t < n * n {
        let i = t / n;
        let j = t % n;
        // acc += conj(e[i]) * G[i][j] * e[j]   (real part)
        let ce_re = e[i].re;
        let ce_im = -e[i].im;
        let ge_re = g[i * n + j].re;
        let ge_im = g[i * n + j].im;
        let ej_re = e[j].re;
        let ej_im = e[j].im;
        // (conj(e[i]) * G[i][j]) * e[j]
        let m_re = ce_re * ge_re - ce_im * ge_im;
        let m_im = ce_re * ge_im + ce_im * ge_re;
        acc += m_re * ej_re - m_im * ej_im;
        t += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    #[test]
    fn forward_sequence_basis_length() {
        // A 2×2 Hamiltonian, 2 shifts → 3 Krylov vectors.
        let h = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)];
        let v0 = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let z = vec![c(0.5, 0.0), c(1.5, 0.0)];
        let ws = forward_sequence(&h, &v0, &z, 2, 2);
        assert_eq!(ws.len(), 3);
    }

    #[test]
    fn gram_hermitian_symmetry_on_small_basis() {
        let h = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)];
        let v0 = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let z = vec![c(0.5, 0.0)];
        let ws = forward_sequence(&h, &v0, &z, 1, 2);
        let g = gram_assembly(&ws, 1, 2);
        // G[0][1] vs conj(G[1][0])
        let g01 = C64 { re: g[0 * 2 + 1].re, im: g[0 * 2 + 1].im };
        let g10 = C64 { re: g[1 * 2 + 0].re, im: g[1 * 2 + 0].im };
        assert!((g01.re - g10.re).abs() < 1e-12);
        assert!((g01.im + g10.im).abs() < 1e-12);
    }

    #[test]
    fn projection_identity_is_linear_in_gram_entries() {
        // H_jk = σ_{k+1} G_{j,k+1} + z_k G_{j,k}
        let gj = c(0.5, 0.25);
        let gjp1 = c(0.75, -0.125);
        let out = projection_identity(&gjp1, &gj, 2.0, &c(0.25, 0.5));
        let expect_re = 2.0 * 0.75 + (0.25 * 0.5 - 0.5 * 0.25);
        let expect_im = 2.0 * (-0.125) + (0.25 * 0.25 + 0.5 * 0.5);
        assert!((out.re - expect_re).abs() < 1e-12);
        assert!((out.im - expect_im).abs() < 1e-12);
    }

    #[test]
    fn whitening_identity_t_star_g_t_is_identity() {
        // G = diag(4, 1): T = diag(1/2, 1), T* G T = I.
        let eigvals = vec![4.0, 1.0];
        let eigvecs = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        let t = whitening_transform(&eigvals, &eigvecs, 2, 2);
        // T* G T = I: T[0][0]² · 4 = 1, T[1][1]² · 1 = 1
        assert!((t[0].re * t[0].re * 4.0 - 1.0).abs() < 1e-12);
        assert!((t[3].re * t[3].re * 1.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn residual_boundary_component_is_tau_times_last_coeff() {
        let out = residual_boundary_component(&c(0.5, -0.25), 2.0);
        assert!((out.re - 1.0).abs() < 1e-12);
        assert!((out.im + 0.5).abs() < 1e-12);
    }
}
