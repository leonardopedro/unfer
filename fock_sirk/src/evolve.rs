//! Restarted Krylov time evolution for long-running dynamics.
//!
//! A single SIRK solve builds a Krylov subspace of dimension `m` and evolves
//! within it. For long total times `t` the small basis is too restrictive, so
//! [`evolve_restarted`] breaks the evolution into `n_restarts` chunks of
//! `t / n_restarts`, reconstructing the full [`QuantumState`] between chunks
//! and re-building a fresh Krylov subspace from the updated state.

use candle_core::Device;
use nested_fock_algebra::{Hamiltonian, QuantumState};
use num_complex::Complex64;

use crate::forward_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use crate::linalg::SirkError;

/// Evolve `psi0` under `h` for total time `t`, using `n_restarts` Krylov restarts
/// each of dimension `krylov_dim`.
///
/// Each restart: build SIRK → `time_evolve(dt)` → `reconstruct` → `prune` →
/// feed the result back as the next `psi0`. The optional `brst` charge is
/// projected out at every Krylov step inside [`solve_forward_sirk_with_opts`].
// The full SIRK evolution context (Hamiltonian, state, schedule, device, optional
// BRST charge, options) is genuinely distinct data; bundling it would only hide the
// inputs behind an indirection without simplifying the call sites.
#[allow(clippy::too_many_arguments)]
pub fn evolve_restarted(
    h: &Hamiltonian,
    psi0: &QuantumState,
    t: f64,
    n_restarts: usize,
    krylov_dim: usize,
    device: &Device,
    brst: Option<&Hamiltonian>,
    opts: &SirkOpts,
) -> Result<QuantumState, SirkError> {
    if n_restarts == 0 {
        return Ok(psi0.clone());
    }
    let dt = t / n_restarts as f64;

    // Imaginary shifts suited to dissipative / oscillatory dynamics.
    let shifts: Vec<Complex64> = (0..krylov_dim)
        .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
        .collect();

    let mut psi = psi0.clone();
    for _ in 0..n_restarts {
        let result = solve_forward_sirk_with_opts(h, &psi, &shifts, device, brst, opts)?;
        let coeffs = result.time_evolve(dt);
        psi = result.reconstruct(&coeffs);
        psi.prune(opts.prune_eps);
    }
    Ok(psi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_sirk::solve_forward_sirk;
    use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};

    /// Two-state hopping Hamiltonian H = |B><A| + |A><B| (eigenvalues ±1).
    fn hopping_hamiltonian() -> Hamiltonian {
        let a = InnerBosonicState::vacuum();
        let mut b = InnerBosonicState::vacuum();
        b.modes.insert(0, 1);
        Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(b.clone()),
                        Operator::OuterBosonAnnihilate(a.clone()),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(a),
                        Operator::OuterBosonAnnihilate(b),
                    ],
                ),
            ],
        }
    }

    #[test]
    fn norm_conserved_across_restarts() {
        let device = Device::Cpu;
        let h = hopping_hamiltonian();
        let v0 = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts::default();
        let psi_t = evolve_restarted(&h, &v0, 1.0, 3, 4, &device, None, &opts).unwrap();

        let norm = psi_t.norm();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "norm not conserved: |ψ| = {norm:.3e}"
        );
    }

    #[test]
    fn agrees_with_single_shot_for_small_t() {
        let device = Device::Cpu;
        let h = hopping_hamiltonian();
        let v0 = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts::default();
        let shifts: Vec<Complex64> = (0..4)
            .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
            .collect();

        // Single-shot: one SIRK solve, evolve for t, reconstruct.
        let result = solve_forward_sirk(&h, &v0, &shifts, &device, None).unwrap();
        let coeffs_single = result.time_evolve(0.01);
        let psi_single = result.reconstruct(&coeffs_single);

        // Restarted: same total time, 2 restarts.
        let psi_restarted =
            evolve_restarted(&h, &v0, 0.01, 2, 4, &device, None, &opts).unwrap();

        // The two states should agree closely for small t.
        let diff = {
            let mut d = psi_single.clone();
            d.scale_and_add(&psi_restarted, Complex64::new(-1.0, 0.0));
            d.norm()
        };
        assert!(
            diff < 1e-6,
            "restarted and single-shot evolution disagree: |diff| = {diff:.3e}"
        );
    }
}
