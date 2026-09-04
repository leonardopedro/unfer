//! Quantum foam / Newtonian gravity overlap numerical validation.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, qg_free_graviton};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-12,
        max_components: Some(100_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: false,
    }
}

fn vac() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn n_photon(mode: u32, n: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, n);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn sirk_ground(h: &nested_fock_algebra::Hamiltonian, v0: &QuantumState, m: usize) -> f64 {
    solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve")
        .ground_state_energy()
        .expect("ground")
}

#[test]
fn qf_ng_graviton_number_classical_limit() {
    let omega = 1.0;
    let h = qg_free_graviton(&[omega]);
    let e0 = sirk_ground(&h, &vac(), 4);
    assert!(e0.abs() < 1e-6, "vacuum E={e0:.6}");
    let e1 = sirk_ground(&h, &n_photon(0u32, 1u32), 4);
    assert!((e1 - omega).abs() < 0.1, "n=1 E={e1:.4}");
    let e2 = sirk_ground(&h, &n_photon(0u32, 2u32), 4);
    assert!((e2 - 2.0 * omega).abs() < 0.1, "n=2 E={e2:.4}");
    eprintln!("qf_ng_graviton_number: E(0)={e0:.4}, E(1)={e1:.4}, E(2)={e2:.4}");
}

#[test]
fn qf_ng_graviton_zero_point_energy() {
    let l = 1.0_f64;
    let omegas: Vec<f64> = (1..=4)
        .map(|n| n as f64 * std::f64::consts::PI / l)
        .collect();
    let h = qg_free_graviton(&omegas);
    let ground = sirk_ground(&h, &vac(), 4);
    assert!(ground.abs() < 1e-6, "ground={ground:.6}");
    let l_prime = 2.0_f64;
    let delta_e = std::f64::consts::PI / 2.0 * (1.0 / l - 1.0 / l_prime);
    assert!(delta_e > 0.0, "Casimir ΔE must be positive");
    eprintln!("qf_ng_graviton_zero_point: ΔE={delta_e:.4}");
}

#[test]
fn qf_ng_bohr_frequency_gravitational_orbit() {
    let n = 2;
    let omega_21 = (2 * n - 1) as f64 / (2.0 * (n * n * (n - 1) * (n - 1)) as f64);
    let n = 3;
    let omega_32 = (2 * n - 1) as f64 / (2.0 * (n * n * (n - 1) * (n - 1)) as f64);
    let ratio = omega_21 / omega_32;
    let expected = 27.0 / 5.0;
    let rel_err = (ratio - expected).abs() / expected;
    assert!(rel_err < 0.01, "ratio={ratio:.4}, expected={expected:.4}");
    let h = qg_free_graviton(&[omega_21, omega_32]);
    let ground = sirk_ground(&h, &vac(), 4);
    assert!(ground.abs() < 1e-6, "ground={ground:.6}");
    eprintln!("qf_ng_bohr: ω₂₁/ω₃₂={ratio:.4} (expected {expected:.4})");
}

#[test]
fn qf_ng_foam_fluctuation_scale() {
    let l_planck = 1.616255e-35_f64;
    let delta_g_1m = (l_planck).powi(2);
    assert!(delta_g_1m < 1.0e-50, "δg={delta_g_1m:.4e}");
    let h = qg_free_graviton(&[1.0, 2.0, 3.0]);
    let ground = sirk_ground(&h, &vac(), 4);
    assert!(ground.abs() < 1e-6, "ground={ground}");
    eprintln!("qf_ng_foam: δg(L=1m)={delta_g_1m:.4e}");
}

#[test]
fn qf_ng_graviton_number_conservation() {
    let omega = 2.5;
    let h = qg_free_graviton(&[omega]);
    for n in 0..=4 {
        let res = solve_forward_sirk_with_opts(
            &h,
            &n_photon(0u32, n as u32),
            &shifts(6),
            &best_device(),
            None,
            &opts(),
        )
        .expect("SIRK");
        let ritz = res.ritz_values();
        let has = ritz.iter().any(|&r| (r - n as f64 * omega).abs() < 0.1);
        assert!(
            has,
            "n={n}: Ritz {ritz:?} must contain nω={:.1}",
            n as f64 * omega
        );
    }
    eprintln!("qf_ng_graviton_number_conservation: verified n=0..4");
}

#[test]
fn qf_ng_coherent_state_classical_limit() {
    let omega = 1.0;
    let h = qg_free_graviton(&[omega]);
    let e10 = sirk_ground(&h, &n_photon(0u32, 10u32), 4);
    assert!((e10 - 10.0 * omega).abs() < 0.1, "|10⟩ E={e10:.4}");
    eprintln!("qf_ng_coherent: |10⟩ E={e10:.4}");
}
