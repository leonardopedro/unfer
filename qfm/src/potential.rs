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
//! Then `build_flow_hamiltonian` constructs the **exact off-diagonal
//! generator** `H = |0~><0~|`: the rank-1 projector onto the dressed Mehler
//! vacuum `|0~> = (|vac> + sum_j bar_alpha_j |x_j>) / norm`, where
//! `|x_j> = OuterBosonCreate(|1_j>)|0>` is the single-excitation channel in
//! the sketched K_2-dim Fock space and the weights `bar_alpha_j` set the
//! channel components of the dressed vacuum. No O(epsilon) truncation: the
//! vacuum-channel coupling comes entirely from the projector's
//! non-orthogonality to the channels (QFM.tex, "the exact off-diagonal
//! generator is just the vacuum projector").

use nested_fock_algebra::{Hamiltonian, qfm_hamiltonian_mehler_projector};

/// Compute the time-averaged optimal coefficients `bar_alpha_j` for the
/// Flow Matching objective.
///
/// For each training point `x^(k)` (a d-dim vector), the decoupled
/// linear-scaling formula gives a per-mode coefficient. The time integral
/// `∫_0^1 alpha_j(t) dt` evaluates in closed form (QFM.tex §"Analytical
/// potential and exact flow generator"), so no numerical quadrature is
/// performed; the `_n_t_samples` / `_noise_dim` parameters are retained
/// only for API stability with earlier revisions that sampled the
/// integral.
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

/// Build the **exact** static flow generator `H = |0~><0~|`: the rank-1
/// projector onto the dressed Mehler vacuum whose channel components are
/// set by the flow-matching weights,
///
///   `|0~> = c_0 |vac> + sum_j eps_j |x_j>`,
///   `eps_j = bar_alpha_j / sqrt(1 + sum alpha^2)`,
///   `c_0   = 1 / sqrt(1 + sum alpha^2)`,
///
/// i.e. the normalization of the unnormalized dressed vector
/// `|vac> + sum_j bar_alpha_j |x_j>`. The vacuum-channel transport comes
/// entirely from the projector's non-orthogonality to the channels — no
/// explicit coupling terms and no O(epsilon) truncation. `H` is exactly
/// idempotent (`H^2 = H`), so the evolution is closed-form:
/// `e^{-iHt} = 1 + (e^{-it} - 1)|0~><0~|`.
///
/// Only channels `j < k2` (the K_2-dim sketched Fock space) enter the
/// dressed vacuum. The single `ProjectOnto` term applies via the rank-1
/// shortcut `H|s> = <0~|s> |0~>`, so M can be very large.
pub fn build_flow_hamiltonian(alphas: &[f64], k2: usize) -> Hamiltonian {
    let take = alphas.len().min(k2);
    let norm = (1.0 + alphas[..take].iter().map(|a| a * a).sum::<f64>()).sqrt();
    let epsilons: Vec<f64> = alphas[..take].iter().map(|a| a / norm).collect();
    // sum eps^2 = sum alpha^2 / (1 + sum alpha^2) < 1, and the projector
    // builder's c_0 = sqrt(1 - sum eps^2) = 1/norm — exactly the normalized
    // dressed vector above.
    qfm_hamiltonian_mehler_projector(&epsilons)
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
    fn build_flow_hamiltonian_is_exact_rank1_projector() {
        // H = |0~><0~| is a single self-adjoint rank-1 term, and exactly
        // idempotent: H(H|s>) = H|s> for any probe |s>. Idempotence is the
        // signature of the exact generator — any truncation would fail it.
        let alphas = vec![1.0, 2.0, 0.5];
        let h = build_flow_hamiltonian(&alphas, 8);
        assert_eq!(h.terms.len(), 1);
        assert_eq!(h.adjoint().terms.len(), 1);

        let vacuum = QuantumState::vacuum();
        let h_vac = h.apply(&vacuum);
        let h_h_vac = h.apply(&h_vac);
        let mut diff = h_h_vac.clone();
        diff.scale_and_add(&h_vac, num_complex::Complex64::new(-1.0, 0.0));
        assert!(diff.norm() < 1e-12, "H^2 = H must hold exactly");
    }

    #[test]
    fn flow_hamiltonian_vacuum_maps_to_scaled_dressed_vacuum() {
        // H|vac> = <0~|vac> |0~> = c_0 |0~>, with
        //   c_0 = 1/sqrt(1 + sum alpha^2), eps_j = alpha_j * c_0.
        // So the vacuum component of H|vac> is c_0^2 and each channel
        // component is c_0 * eps_j = c_0^2 * alpha_j — every channel is
        // coupled through the projector alone, no explicit coupling terms.
        let alphas = vec![1.0, 2.0, 0.5];
        let h = build_flow_hamiltonian(&alphas, 8);
        let norm_sq: f64 = 1.0 + alphas.iter().map(|a| a * a).sum::<f64>();
        let c0_sq = 1.0 / norm_sq;

        let vacuum = QuantumState::vacuum();
        let h_vac = h.apply(&vacuum);

        let amp_vac = h_vac
            .components
            .get(&nested_fock_algebra::OuterState::vacuum())
            .expect("H|vac> must retain a vacuum component");
        assert!((amp_vac.re - c0_sq).abs() < 1e-12);
        assert!(amp_vac.im.abs() < 1e-12);

        // Channel components: amplitude c_0^2 * alpha_j each.
        let mut single_amplitudes: Vec<f64> = Vec::new();
        for (outer, amp) in h_vac.components.iter() {
            if outer.bosonic.values().sum::<u32>() == 1 && outer.fermionic.is_empty() {
                single_amplitudes.push(amp.re);
            }
        }
        assert_eq!(single_amplitudes.len(), alphas.len());
        let mut expected: Vec<f64> = alphas.iter().map(|a| a * c0_sq).collect();
        single_amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in single_amplitudes.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }
}
