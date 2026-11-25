//! Exact reference tier for one declared window of outer labels (`P13`).
//!
//! [`WindowOperator`] assembles `H` restricted to a window — `P H P` in the
//! orthonormal Fock basis — and evaluates the seed column of
//! `exp(-i t H_window)`, which gives both the exact return probability of a
//! single-label seed and the amplitude gap a certified charge has to dominate.
//!
//! The column is computed by Hermitian Lanczos plus the exponential of the
//! (small) tridiagonal, `O(m n^2)` for Krylov dimension `m`, instead of the
//! dense `n x n` Padé `exp` this tier used before. That dense factorisation
//! was cubic and, past `C = 24`, dominated the sweep cell it was supposed to
//! referee (`docs/VERIFY_PHI4_MOMENTUM.md` §5.7). The action is the same
//! mathematical column; §5.7 records the agreement with the dense values this
//! replaced.

use fock_sirk::joint_sirk::JointSirkResult;
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::{ActionLimits, InnerExpr, JointOuterBasis, NestedVector};
use num_complex::Complex64 as C;
use std::collections::{BTreeMap, BTreeSet};

/// Krylov dimensions at which the seed column is formed and its convergence
/// compared against the previous checkpoint. The sweep windows are Hermitian
/// and evolve for `t = 0.05`, where Lanczos converges superlinearly in the
/// seed's spectral weights; the checkpoints exist so the dimension is chosen
/// by measurement, not by a guess baked into a constant.
const SEED_COLUMN_CHECKPOINTS: [usize; 4] = [64, 128, 256, 512];

/// Absolute tolerance on the seed column (whose norm is 1) used to declare the
/// Krylov expansion converged.
const SEED_COLUMN_TOLERANCE: f64 = 1e-13;

pub struct WindowOperator {
    index: BTreeMap<JointOuterBasis, usize>,
    matrix: DMatrix<C>,
}

impl WindowOperator {
    /// One action application per label: column `j` of `P H P` is the image of
    /// the unit label `j`, kept where it lands inside the window.
    pub fn build(h: &InnerExpr, window: &BTreeSet<JointOuterBasis>) -> Self {
        let labels: Vec<JointOuterBasis> = window.iter().cloned().collect();
        let index: BTreeMap<JointOuterBasis, usize> = labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let n = labels.len();
        let mut matrix = DMatrix::<C>::zeros(n, n);
        for (j, basis) in labels.iter().enumerate() {
            let image = h
                .d_gamma(
                    &NestedVector::from([(basis.clone(), C::new(1.0, 0.0))]),
                    ActionLimits::default(),
                )
                .expect("finite action");
            for (target, amplitude) in image {
                if let Some(&i) = index.get(&target) {
                    matrix[(i, j)] += amplitude;
                }
            }
        }
        Self { index, matrix }
    }

    pub fn labels(&self) -> usize {
        self.matrix.nrows()
    }

    /// Index of the single unit basis label of the seed.
    pub fn seed_index(&self, initial: &NestedVector) -> usize {
        let mut label = None;
        for (basis, amplitude) in initial {
            assert!(
                (amplitude - C::new(1.0, 0.0)).norm() < 1e-15,
                "reference seed must be a unit basis label"
            );
            assert!(label.is_none(), "reference seed must be a single label");
            label = Some(basis.clone());
        }
        self.index[&label.expect("nonempty seed")]
    }

    /// `exp(-i t H_window) e_j` — the evolved seed column.
    ///
    /// Hermitian Lanczos with full reorthogonalisation: the window matrix is
    /// `P H P` in an orthonormal basis, so it is Hermitian up to the rounding
    /// of the two triangles. The Lanczos diagonal is taken real for that
    /// reason, and `imaginary_alpha` is asserted below tolerance so a
    /// non-Hermitian window fails loudly instead of quietly biasing the
    /// reference.
    pub fn exp_action_column(&self, j: usize, t: f64) -> DVector<C> {
        let n = self.matrix.nrows();
        let guard = self.matrix.norm().max(1.0);
        let mut basis: Vec<DVector<C>> = Vec::new();
        let mut alpha: Vec<f64> = Vec::new();
        let mut beta: Vec<f64> = Vec::new();
        let mut v = DVector::<C>::zeros(n);
        v[j] = C::new(1.0, 0.0);
        let mut invariant = false;
        let mut last: Option<DVector<C>> = None;

        for checkpoint in SEED_COLUMN_CHECKPOINTS {
            while !invariant && basis.len() < checkpoint.min(n) {
                let k = basis.len();
                basis.push(v.clone());
                let image = &self.matrix * &basis[k];
                let diagonal = basis[k].dotc(&image);
                assert!(
                    diagonal.im.abs() <= 1e-12 * guard,
                    "window matrix is not Hermitian: imaginary diagonal {}",
                    diagonal.im
                );
                alpha.push(diagonal.re);
                let mut w = image - &basis[k] * C::new(diagonal.re, 0.0);
                if k > 0 {
                    w -= &basis[k - 1] * C::new(beta[k - 1], 0.0);
                }
                // Two passes of classical Gram-Schmidt (DGKS): the short
                // recurrence loses orthogonality on these clustered spectra,
                // and the convergence test below relies on a true basis.
                for _ in 0..2 {
                    for u in &basis {
                        let projection = u.dotc(&w);
                        w -= u * projection;
                    }
                }
                let norm = w.norm();
                if norm <= 1e-14 * guard {
                    // The Krylov subspace is invariant: the column is exact.
                    invariant = true;
                    break;
                }
                beta.push(norm);
                v = w / C::new(norm, 0.0);
            }

            let column = self.column_from_tridiagonal(&basis, &alpha, &beta, t);
            if invariant {
                return column;
            }
            if let Some(previous) = &last
                && (column.clone() - previous).norm() <= SEED_COLUMN_TOLERANCE
            {
                return column;
            }
            last = Some(column);
        }

        // Not converged at the largest checkpoint. Returning the last column
        // would silently downgrade the reference tier, so this is loud.
        let residual = last
            .as_ref()
            .map(|column| column.norm())
            .unwrap_or(f64::NAN);
        panic!(
            "seed column did not converge to {SEED_COLUMN_TOLERANCE} within \
             {} Lanczos steps (n = {n}, |column| = {residual}); raise \
             SEED_COLUMN_CHECKPOINTS",
            SEED_COLUMN_CHECKPOINTS[SEED_COLUMN_CHECKPOINTS.len() - 1]
        );
    }

    /// `V_m f(T_m) e_1` for the real symmetric tridiagonal `T_m`.
    fn column_from_tridiagonal(
        &self,
        basis: &[DVector<C>],
        alpha: &[f64],
        beta: &[f64],
        t: f64,
    ) -> DVector<C> {
        let m = basis.len();
        let mut tridiagonal = DMatrix::<C>::zeros(m, m);
        for k in 0..m {
            tridiagonal[(k, k)] = C::new(alpha[k], 0.0);
        }
        for k in 0..m.saturating_sub(1) {
            tridiagonal[(k, k + 1)] = C::new(beta[k], 0.0);
            tridiagonal[(k + 1, k)] = C::new(beta[k], 0.0);
        }
        let mut seed = DVector::<C>::zeros(m);
        seed[0] = C::new(1.0, 0.0);
        let coefficients = (tridiagonal * C::new(0.0, -t)).exp() * seed;
        let mut column = DVector::<C>::zeros(self.matrix.nrows());
        for (k, coefficient) in coefficients.iter().enumerate() {
            column += &basis[k] * *coefficient;
        }
        column
    }

    /// Exact unitary return probability of a single-label seed in this window:
    /// `|[exp(-i t H_window)]_jj|^2`.
    pub fn reference_return(&self, initial: &NestedVector, t: f64) -> f64 {
        let j = self.seed_index(initial);
        self.exp_action_column(j, t)[j].norm_sqr()
    }

    /// `|⟨ψ|ψ(t)⟩ − ⟨ψ|w(t)⟩|` on the declared window: the amplitude error the
    /// full-action amplitude bound has to dominate. The seed is `e_j`, so the
    /// exact amplitude is entry `j` of the evolved column, and the SIRK
    /// amplitude is the sum of the reconstructed state's weight on label `j`.
    pub fn amplitude_gap(&self, result: &JointSirkResult, initial: &NestedVector, t: f64) -> f64 {
        let j = self.seed_index(initial);
        let exact = self.exp_action_column(j, t);
        let state = result
            .reconstruct(&result.time_evolve(t).expect("finite time"))
            .expect("reconstruction");
        let mut sirk = C::new(0.0, 0.0);
        for (label, amplitude) in &state {
            if self.index.get(label) == Some(&j) {
                sirk += *amplitude;
            }
        }
        (exact[j] - sirk).norm()
    }
}
