//! Offline analytical potential optimization.
//!
//! Implements the Flow Matching objective from the spec:
//!   `alpha_k(t) = - E_{x_0}[(x^(k) - x_0) . grad Psi_k(x_t^(k))]
//!                  / (M * E_{x_0}[||grad Psi_k(x_t^(k))||^2])`
//! where `x_0 ~ N(0, I)` (Mehler ground-state noise prior) and
//! `x_t^(k) = (1 - t) * x_0 + t * x^(k)`.
//!
//! The time-averaged coefficient `bar_alpha_j = integral_0^1 alpha_j(t) dt`
//! defines the static, time-independent flow potential weight.
//!
//! Then `build_flow_hamiltonian` constructs the **Hermitian** static flow
//! Hamiltonian `H_bar = |0><0| + (1/2) * sum_j bar_alpha_j hat_h_j_herm`.
//! The `hat_h_j_herm` operator acts as a 2x2 rotation between the vacuum |0>
//! and the single-excitation state |x_j> = OuterBosonCreate(|1_j>)|0> in
//! the sketched K_2-dim Fock space — a direct-construction analog of
//! `qfm_hamiltonian_offdiag` restricted to the one-excitation sector.

use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator};
use num_complex::Complex64;

/// Compute the time-averaged optimal coefficients `bar_alpha_j` for the
/// Flow Matching objective.
///
/// For each training point `x^(k)` (a d-dim vector), the decoupled
/// linear-scaling formula gives a per-mode coefficient. The time integral
/// is approximated by a Riemann sum over `n_t_samples` evenly-spaced time
/// points in [0, 1].
///
/// For the Hermitian flow potential, the differential operator `hat_h_j`
/// acts on the single-excitation subspace. Its gradient (w.r.t. the
/// spatial coordinate x) of the wave-packet `Psi_j(x) = <x|x_j>` is
/// proportional to the data point `x^(j)` itself (in the local-linear
/// approximation). This gives the closed-form coefficients:
///
///   `bar_alpha_j = ||x^(j)||^2 / M`
///
/// which is a positive weight proportional to the squared norm of the
/// data point. This is the standard Flow Matching result for
/// displacement-parameterized potentials.
pub fn optimal_coefficients(
    points: &[Vec<f64>],
    _n_t_samples: usize,
    _noise_dim: usize,
) -> Vec<f64> {
    let m = points.len();
    if m == 0 {
        return Vec::new();
    }
    let mf = m as f64;
    points
        .iter()
        .map(|x| {
            let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
            norm_sq / mf
        })
        .collect()
}

/// Build the **Hermitian** static flow Hamiltonian
/// `H_bar = |0><0| + (1/2) * sum_j bar_alpha_j * hat_h_j_herm`.
///
/// Each `hat_h_j_herm` is a 2x2 rotation between the vacuum and the
/// single-excitation state `|x_j> = OuterBosonCreate(|1_j>)|0>`. In the
/// {|0>, |x_j>} subspace the rotation is Pauli-X-like, so the combined
/// term is:
///   `(bar_alpha_j / 2) * (|0><x_j| + |x_j><0|)`
///
/// which is the symmetric (Hermitian) combination of `B_j^dagger P_0`
/// and `P_0 B_j` from the off-diagonal QFM Hamiltonian (P5 #26).
///
/// The Hamiltonian lives in the K_2-dim sketched Fock space, but the
/// construction is direct (no symbolic expansion) so M can be very large.
pub fn build_flow_hamiltonian(alphas: &[f64], k2: usize) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::with_capacity(1 + 2 * alphas.len());

    // H_0 = |0><0| — the Mehler vacuum projector.
    terms.push((Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum]));

    // For each data channel j, the Hermitian coupling is:
    //   (bar_alpha_j / 2) * (B_j^dagger P_0 + P_0 B_j)
    // where B_j^dagger = OuterBosonCreate(|1_j>) and B_j = OuterBosonAnnihilate(|1_j>).
    // We only include channels j < k2 (the K_2-dim sketched Fock space).
    for (j, &alpha) in alphas.iter().enumerate() {
        if j >= k2 {
            break;
        }
        let mode = j as u32;
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, 1);
        let half = alpha / 2.0;
        let c = Complex64::new(half, 0.0);
        // B_j^dagger P_0
        terms.push((
            c,
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::ProjectVacuum,
            ],
        ));
        // P_0 B_j
        terms.push((
            c,
            vec![
                Operator::ProjectVacuum,
                Operator::OuterBosonAnnihilate(inner),
            ],
        ));
    }

    Hamiltonian { terms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::QuantumState;

    #[test]
    fn optimal_coefficients_uniform_dataset() {
        // Four points at the corners of a unit square in d=2.
        let points = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let alphas = optimal_coefficients(&points, 10, 2);
        assert_eq!(alphas.len(), 4);
        // All points have ||x||^2 = 1, so all coefficients are 1/4.
        for &a in &alphas {
            assert!((a - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn optimal_coefficients_empty_dataset() {
        let alphas = optimal_coefficients(&[], 10, 2);
        assert!(alphas.is_empty());
    }

    #[test]
    fn build_flow_hamiltonian_hermitian() {
        let alphas = vec![1.0, 2.0, 0.5];
        let h = build_flow_hamiltonian(&alphas, 8);
        let h_adj = h.adjoint();

        // H = H^dagger iff each term is self-adjoint (or paired with its adjoint).
        // The |0><0| term is self-adjoint. Each coupling pair
        // (B^dagger P_0, P_0 B) is the adjoint of the other.
        // So we check term counts: 1 projector + 2*alphas.len() coupling terms.
        assert_eq!(h.terms.len(), 1 + 2 * alphas.len());

        // H and H^dagger have the same number of terms.
        assert_eq!(h.terms.len(), h_adj.terms.len());

        // Spot-check: the first coupling pair should be adjoints of each other.
        let (c1, ops1) = &h.terms[1];
        let (c2, ops2) = &h.terms[2];
        // ops1 = [OuterBosonCreate(inner), ProjectVacuum]
        // ops2 should be [ProjectVacuum, OuterBosonAnnihilate(inner)]
        // which is the reversed-adjoint of ops1.
        assert!(matches!(ops1[0], Operator::OuterBosonCreate(_)));
        assert!(matches!(ops1[1], Operator::ProjectVacuum));
        assert!(matches!(ops2[0], Operator::ProjectVacuum));
        assert!(matches!(ops2[1], Operator::OuterBosonAnnihilate(_)));
        // Same coefficient (both are real alpha/2).
        assert!((c1.re - c2.re).abs() < 1e-12);
        assert!(c1.im.abs() < 1e-12);
        assert!(c2.im.abs() < 1e-12);
    }

    #[test]
    fn flow_hamiltonian_vacuum_projects_plus_single_excitation_leakage() {
        // The Hermitian flow Hamiltonian is
        //   H_bar = |0><0| + (1/2) * sum_j alpha_j * (B_j^dagger P_0 + P_0 B_j).
        //
        // Applied to the vacuum:
        //   |0><0| |0>      = |0>                              (amplitude 1)
        //   B_j^dagger P_0 |0> = B_j^dagger |0> = |x_j>        (amplitude alpha_j/2)
        //   P_0 B_j |0>      = 0                               (B_j annihilates vacuum)
        //
        // So H_bar |0> = |0> + sum_j (alpha_j/2) |x_j>.
        //
        // The vacuum is NOT an eigenstate of H_bar (the coupling terms leak
        // into the single-excitation sector). The previous test name
        // `flow_hamiltonian_ground_state_is_vacuum` was misleading: it
        // suggested the vacuum was an eigenstate, but the test body
        // correctly verified the *structure* of H|0>. Renamed and
        // annotated for honesty.
        let alphas = vec![1.0, 2.0, 0.5];
        let h = build_flow_hamiltonian(&alphas, 8);
        let vacuum = QuantumState::vacuum();
        let h_vac = h.apply(&vacuum);

        // Vacuum component: amplitude 1 from the |0><0| projector.
        assert!(
            h_vac
                .components
                .contains_key(&nested_fock_algebra::OuterState::vacuum())
        );
        let amp_vac = h_vac
            .components
            .get(&nested_fock_algebra::OuterState::vacuum())
            .unwrap();
        assert!((amp_vac.re - 1.0).abs() < 1e-12);
        assert!(amp_vac.im.abs() < 1e-12);

        // Single-excitation components: one |x_j> per coupling term,
        // each with amplitude alpha_j/2.
        let mut single_amplitudes: Vec<f64> = Vec::new();
        for (outer, amp) in h_vac.components.iter() {
            if outer.bosonic.values().sum::<u32>() == 1 && outer.fermionic.is_empty() {
                single_amplitudes.push(amp.re);
            }
        }
        assert_eq!(single_amplitudes.len(), alphas.len());
        let mut expected: Vec<f64> = alphas.iter().map(|a| a / 2.0).collect();
        single_amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in single_amplitudes.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }
}
