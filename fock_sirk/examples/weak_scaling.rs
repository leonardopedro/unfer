//! Weak-scaling benchmark for the SIRK solver (Stage S16).
//!
//! Sweeps the Krylov dimension `m ∈ {3, 6, 9}` over the two-site hopping
//! model, reporting solve/evolve wall-times, the measured Gram rank after
//! whitening, and the auto-`m` recommendation for the next restart.
//!
//! Run: `cargo run -p fock_sirk --example weak_scaling`
//!
//! The bench doubles as a regression gate: it must never panic (Gram whitening
//! stability) and the measured rank must always be `<= m`.

use std::time::Instant;

use candle_core::Device;
use fock_sirk::auto::{budgeted_shift_batches, shifts_for_range};
use fock_sirk::{SirkOpts, evolve_restarted, solve_forward_sirk};
use nested_fock_algebra::{Hamiltonian, InnerBosonicState, Operator, QuantumState};
use num_complex::Complex64;

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

fn main() {
    let device = best_device();
    let h = hopping_hamiltonian();
    let v0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    let opts = SirkOpts::default();

    println!("device: {device:?}");
    println!(
        "{:<4} {:>6} {:>6} {:>12} {:>12} {:>10}",
        "m", "rank", "auto-m", "solve(us)", "evolve(us)", "batches"
    );

    for m in [3usize, 6, 9] {
        let shifts = shifts_for_range((0, m));

        let t_solve = Instant::now();
        let res = solve_forward_sirk(&h, &v0, &shifts, &device, None).unwrap();
        let solve_us = t_solve.elapsed().as_micros();
        let rank = res.rank;
        assert!(rank <= m, "measured rank {rank} exceeds m={m}");

        // Auto-m rule: saturate just past the measured rank with a reserve of 1.
        // (AGENTS.md rank-saturation: the effective Gram rank caps ~6.)
        let auto_m = (rank + 1).clamp(3, 12);

        let t_evolve = Instant::now();
        let psi = evolve_restarted(&h, &v0, 1.0, 3, m, &device, None, &opts).unwrap();
        let evolve_us = t_evolve.elapsed().as_micros();
        let norm = psi.norm();

        let batches = budgeted_shift_batches(m, auto_m.max(2));

        println!(
            "{m:<4} {:>6} {:>6} {:>12} {:>12} {:>10}",
            rank,
            auto_m,
            solve_us,
            evolve_us,
            batches.len()
        );

        assert!(
            (norm - 1.0).abs() < 1e-6,
            "norm not conserved at m={m}: |ψ| = {norm:.3e}"
        );
    }
}

fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}
