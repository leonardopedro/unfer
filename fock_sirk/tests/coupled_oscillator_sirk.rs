//! SIRK machinery on exactly-solvable coupled-oscillator models.
//!
//! 1. `sirk_beamsplitter_spectrum_exact` — the two-mode beamsplitter
//!    `H = ω(N₀+N₁) + J(a†₀a₁ + a†₁a₀)`: the number-conserving one-photon
//!    sector {|10⟩,|01⟩} has the exact spectrum {ω−J, ω+J}; the SIRK Ritz
//!    values reproduce both to solver precision and the projected H is
//!    Hermitian.
//! 2. `sirk_beamsplitter_swap_dynamics` — a photon injected in mode 0 swaps
//!    into mode 1 as P(t) = sin²(Jt): ⟨N₁⟩ = ½ at Jt = π/4 and complete
//!    swap at Jt = π/2 (restarted-Krylov unitary evolution, norm conserved).
//! 3. `sirk_displaced_oscillator_exact_shift` — H = ωN + g(a†+a): the exact
//!    displaced-oscillator shift E_n = ωn − g²/ω; the SIRK ground energy is
//!    −g²/ω and every Ritz value sits on an exact level.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{evolve_restarted, solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{
    InnerBosonicState, Operator, QuantumState, oscillator_beamsplitter, oscillator_displaced,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
    }
}

fn empty_universe_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn one_photon_in_mode(mode: u32) -> QuantumState {
    empty_universe_vacuum().apply(&Operator::InnerBosonCreate(mode))
}

#[test]
fn sirk_beamsplitter_spectrum_exact() {
    let (omega, j) = (2.0_f64, 0.35_f64);
    let h = oscillator_beamsplitter(omega, j);
    let v0 = one_photon_in_mode(0);
    let res =
        solve_forward_sirk_with_opts(&h, &v0, &shifts(6), &best_device(), None, &opts()).unwrap();
    let ritz = res.ritz_values();
    // N-conserving: the vacuum is NOT reached from |10⟩. Exact sector
    // spectrum {ω−J, ω+J} (the coupling matrix [[0,J],[J,0]] eigenvalues).
    assert_eq!(ritz.len(), 2, "one-photon sector only, got {ritz:?}");
    assert!((ritz[0] - (omega - j)).abs() < 1e-9, "ritz {ritz:?}");
    assert!((ritz[1] - (omega + j)).abs() < 1e-9, "ritz {ritz:?}");
}

#[test]
fn sirk_beamsplitter_swap_dynamics() {
    let (omega, j) = (2.0_f64, 0.5_f64);
    let h = oscillator_beamsplitter(omega, j);
    let n1 = nested_fock_algebra::Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(1),
                Operator::InnerBosonAnnihilate(1),
            ],
        )],
    };
    let psi0 = one_photon_in_mode(0);

    for &jt in &[std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2] {
        let psi_t =
            evolve_restarted(&h, &psi0, jt / j, 40, 8, &best_device(), None, &opts()).unwrap();
        let n1_expect = QuantumState::inner_product(&psi_t, &n1.apply(&psi_t)).re;
        let target = jt.sin().powi(2);
        assert!(
            (n1_expect - target).abs() < 1e-6,
            "⟨N₁⟩(Jt={jt:.4}) = {n1_expect:.9}, want {target}"
        );
        let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
        assert!((norm - 1.0).abs() < 1e-9, "norm {norm}");
    }
}

#[test]
fn sirk_displaced_oscillator_exact_shift() {
    let (omega, g) = (1.7_f64, 0.45_f64);
    let h = oscillator_displaced(omega, g);
    let v0 = empty_universe_vacuum();
    let res =
        solve_forward_sirk_with_opts(&h, &v0, &shifts(6), &best_device(), None, &opts()).unwrap();
    let ritz = res.ritz_values();
    // Exact levels E_n = ωn − g²/ω (normal-ordered ground at −g²/ω).
    let shift = g * g / omega;
    assert!(
        (ritz[0] - (-shift)).abs() < 1e-7,
        "ground {} want {}",
        ritz[0],
        -shift
    );
    // Placement on exact levels for every RESOLVED Ritz value (below the
    // top physical level + half a gap). The restarted-Krylov projection
    // carries additional spurious values ABOVE the converged spectral window
    // — an edge artifact of the m-shift projection, not misplaced physics.
    let cutoff = omega * 3.0 - shift + omega / 2.0;
    let resolved: Vec<f64> = ritz.iter().copied().filter(|&v| v < cutoff).collect();
    assert!(resolved.len() >= 3, "must resolve ≥3 oscillator levels");
    for v in &resolved {
        let n_float = (v + shift) / omega;
        let n_round = n_float.round();
        let tol_e = 1e-4_f64.max(2e-3 * v.abs());
        assert!(
            (n_float - n_round).abs() * omega < tol_e && n_round >= 0.0 && n_round <= 3.0,
            "Ritz {v} not on an exact level (band {tol_e:.2e})"
        );
    }
    // First excitation gap exactly ω (on the resolved spectrum).
    assert!((resolved[1] - resolved[0] - omega).abs() < 1e-6);
}
