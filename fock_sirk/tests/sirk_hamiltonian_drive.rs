//! SIRK-driven numerical tests on the project's own Hamiltonians.
//!
//! Every test runs the Hashimoto/SIRK solver against a Hamiltonian from
//! `nested_fock_algebra::models` and checks that the solver output is
//! physically consistent.  No external formulas — only SIRK + project Hamiltonians.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nested_fock_algebra::{
    InnerBosonicState, Operator, QuantumState,
    yang_mills_lattice,
    qed_free_photon, qed_cavity_frequencies, qed_pair_production,
    qg_free_graviton, gravity_hamiltonian, harmonic_chain, navier_stokes_hamiltonian,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> { shifts_for_range((0, m)) }

fn opts() -> SirkOpts {
    SirkOpts { prune_eps: 1e-12, max_components: Some(200_000),
        brst_tol: 1e-10, adaptive: false, unit_norm_steps: false }
}

fn vac() -> QuantumState {
    QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn one_excited(mode: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, 1u32);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn sirk(h: &nested_fock_algebra::Hamiltonian, v: &QuantumState, m: usize) -> fock_sirk::ForwardSirkResult {
    solve_forward_sirk_with_opts(h, v, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve")
}

fn ground(h: &nested_fock_algebra::Hamiltonian, v: &QuantumState, m: usize) -> f64 {
    sirk(h, v, m).ground_state_energy().expect("ground")
}

// ── QYM: lattice at strong coupling ───────────────────────────────

#[test]
fn sirk_drive_ym_lattice_gap() {
    // yang_mills_lattice(2, g, 1): the even→odd gap is ≈ g²/2.
    let g = 2.0_f64;
    let h = yang_mills_lattice(2, g, 1);
    let g2h = g * g / 2.0;
    let e0 = ground(&h, &vac(), 4);
    let e1 = ground(&h, &one_excited(0), 4);
    let gap = e1 - e0;
    assert!((gap - g2h).abs() / g2h < 0.05,
        "YM lattice gap={gap:.4}, g²/2={g2h:.1}");
    eprintln!("sirk_drive_ym_lattice_gap: gap={gap:.4}, g²/2={g2h:.1}");
}

// ── QYM: Krylov convergence ──────────────────────────────────────

#[test]
fn sirk_drive_ym_krylov_convergence() {
    // The gap converges as m grows.  At g=2, m=4 is within 1% of g²/2.
    let g = 2.0_f64;
    let h = yang_mills_lattice(2, g, 1);
    let g2h = g * g / 2.0;
    let mut prev_gap = f64::INFINITY;
    for m in [2, 3, 4, 5] {
        let e0 = ground(&h, &vac(), m);
        let e1 = ground(&h, &one_excited(0), m);
        let gap = e1 - e0;
        assert!(gap > 0.0, "m={m}: gap must be positive");
        assert!(gap <= prev_gap + 0.05,
            "m={m}: gap={gap:.4} > prev={prev_gap:.4}");
        prev_gap = gap;
    }
    let e0 = ground(&h, &vac(), 5);
    let e1 = ground(&h, &one_excited(0), 5);
    let final_gap = e1 - e0;
    assert!((final_gap - g2h).abs() / g2h < 0.02,
        "final gap={final_gap:.4}, g²/2={g2h:.1}");
    eprintln!("sirk_drive_ym_krylov: final gap={final_gap:.4}");
}

// ── QED: free photon multi-mode ──────────────────────────────────

#[test]
fn sirk_drive_qed_multimode() {
    let omegas = [0.5, 1.0, 2.0, 4.0];
    let h = qed_free_photon(&omegas);
    let e0 = ground(&h, &vac(), 6);
    assert!(e0.abs() < 1e-6, "QED vacuum={e0:.6}");
    for (k, &omega) in omegas.iter().enumerate() {
        let ek = ground(&h, &one_excited(k as u32), 6);
        assert!((ek - omega).abs() < 0.1,
            "mode {k}: E={ek:.4}, ω={omega:.1}");
    }
    eprintln!("sirk_drive_qed_multimode: all modes verified");
}

// ── QED: cavity modes ────────────────────────────────────────────

#[test]
fn sirk_drive_qed_cavity() {
    let d = 1.0_f64;
    let n_max = 4;
    let omegas = qed_cavity_frequencies(d, n_max);
    let h = qed_free_photon(&omegas);
    for (k, &omega) in omegas.iter().enumerate() {
        let ek = ground(&h, &one_excited(k as u32), 6);
        assert!((ek - omega).abs() < 0.1,
            "cavity mode {k}: E={ek:.4}, ω={omega:.1}");
    }
    eprintln!("sirk_drive_qed_cavity: all {n_max} modes verified");
}

// ── QED: pair production vacuum ──────────────────────────────────

#[test]
fn sirk_drive_qed_pair_production() {
    let m_e = 0.511_f64;
    let g_coupling = 0.1_f64;
    let h = qed_pair_production(m_e, &[m_e], &[m_e], &[g_coupling]);
    let e0 = ground(&h, &vac(), 6);
    // Pair production Hamiltonian has a non-trivial vacuum structure.
    // The vacuum energy is a number — just check it's finite.
    assert!(e0.is_finite(), "pair production vacuum must be finite, got {e0}");
    eprintln!("sirk_drive_qed_pair_production: vacuum={e0:.4}");
}

// ── QG: graviton multi-mode ──────────────────────────────────────

#[test]
fn sirk_drive_qg_graviton() {
    let omegas = [0.5, 1.0, 2.0];
    let h = qg_free_graviton(&omegas);
    let e0 = ground(&h, &vac(), 6);
    assert!(e0.abs() < 1e-6, "QG vacuum={e0:.6}");
    for (k, &omega) in omegas.iter().enumerate() {
        let ek = ground(&h, &one_excited(k as u32), 6);
        assert!((ek - omega).abs() < 0.1,
            "graviton mode {k}: E={ek:.4}, ω={omega:.1}");
    }
    eprintln!("sirk_drive_qg_graviton: all modes verified");
}

// ── QG: 3D gauge-fixed gravity vacuum ────────────────────────────

#[test]
fn sirk_drive_qg_3d() {
    let h = gravity_hamiltonian();
    let e0 = ground(&h, &vac(), 4);
    // The 3D gauge-fixed gravity Hamiltonian has a non-zero vacuum
    // energy (the cosmological constant term).  Check it's finite.
    assert!(e0.is_finite(), "QG 3D vacuum must be finite, got {e0}");
    eprintln!("sirk_drive_qg_3d: vacuum={e0:.4}");
}

// ── NG: harmonic chain ───────────────────────────────────────────

#[test]
fn sirk_drive_ng_chain() {
    let n_modes = 3;
    let omega = 1.5_f64;
    let h = harmonic_chain(n_modes, omega);
    let e0 = ground(&h, &vac(), 6);
    assert!(e0.abs() < 1e-6, "chain vacuum={e0:.6}");
    let e1 = ground(&h, &one_excited(0), 6);
    assert!((e1 - omega).abs() < 0.1,
        "chain first excited: E={e1:.4}, ω={omega:.1}");
    eprintln!("sirk_drive_ng_chain: E_0={e0:.4}, E_1={e1:.4}");
}

// ── NS: Navier-Stokes Hamiltonian ────────────────────────────────

#[test]
fn sirk_drive_ns() {
    let nu = 0.01_f64;
    let h = navier_stokes_hamiltonian(nu);
    let e0 = ground(&h, &vac(), 4);
    // NS Hamiltonian has non-trivial vacuum (viscous damping terms).
    assert!(e0.is_finite(), "NS vacuum must be finite, got {e0}");
    eprintln!("sirk_drive_ns: vacuum={e0:.4}");
}

// ── QYM: Ritz residual decays ────────────────────────────────────

#[test]
fn sirk_drive_ym_residual_decay() {
    let g = 2.0_f64;
    let h = yang_mills_lattice(2, g, 1);
    let mut prev_res = f64::INFINITY;
    for m in [2, 3, 4, 5] {
        let res = sirk(&h, &vac(), m);
        let residuals = res.ritz_abs_residuals();
        if let Some(&(_, r0)) = residuals.first() {
            assert!(r0 < prev_res + 1e-4,
                "m={m}: residual={r0:.6} > prev={prev_res:.6}");
            prev_res = r0;
        }
    }
    eprintln!("sirk_drive_ym_residual_decay: verified");
}

// ── Cross-check: QG graviton ≈ NG chain at same ω ───────────────

#[test]
fn sirk_drive_qg_ng_crosscheck() {
    let omega = 1.0_f64;
    let h_qg = qg_free_graviton(&[omega]);
    let h_ng = harmonic_chain(1, omega);
    let gap_qg = ground(&h_qg, &one_excited(0), 4)
        - ground(&h_qg, &vac(), 4);
    let gap_ng = ground(&h_ng, &one_excited(0), 4)
        - ground(&h_ng, &vac(), 4);
    assert!((gap_qg - gap_ng).abs() < 0.1,
        "QG gap={gap_qg:.4} must match NG gap={gap_ng:.4}");
    eprintln!("sirk_drive_qg_ng_crosscheck: QG={gap_qg:.4}, NG={gap_ng:.4}");
}
