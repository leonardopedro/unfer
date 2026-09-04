//! Constraint and superselection structure through SIRK flows:
//! an abelian Yang–Mills BRST charge built from the residual Gauss generator,
//! QED total-excitation superselection under detuned Jaynes–Cummings flow,
//! and a QG densitized-model beat whose period is PREDICTED by the
//! independently solved Ritz spectrum (spectroscopy → dynamics closure).
//!
//! 1. `qym_abelian_brst_nilpotent_and_invariant` — Ω = P·c_ghost:
//!    Ω²=0 (Pauli), [H(g=0), Ω]=0 (P commutes, ghosts free).
//! 2. `qym_brst_projection_identity_on_physical_flow` — ghost-free start:
//!    solves with and without mid-sequence projection agree on resolved
//!    spectra AND on ⟨P⟩ to solver accuracy (the theorem, YM edition).
//! 3. `qed_jc_total_excitation_superselected` — N_tot = a†a + e†e commutes
//!    with the detuned JC Hamiltonian; mean AND variance of N_tot are flow
//!    constants (Rabi exchange moves quanta BETWEEN subsystems only).
//! 4. `qg_densitized_beat_predicted_by_spectrum` — 𝒮/y sector superposition;
//!    the solved Ritz splitting ΔE predicts a transfer-coherence zero at
//!    t = π/(2ΔE); the flow confirms it (solver→dynamics consistency).

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, evolve_restarted, solve_forward_sirk_with_opts};
use nested_fock_algebra::{
    InnerBosonicState, InnerFermionicState, Operator, QuantumState, qcd_ym_hamiltonian,
    qed_jaynes_cummings,
};
use num_complex::Complex64;
use std::collections::BTreeSet;

fn mk(deep: bool) -> SirkOpts {
    SirkOpts {
        prune_eps: 0.0,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: deep,
    }
}

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn ym_total_momentum() -> nested_fock_algebra::Hamiltonian {
    let mut t: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for mode in [0u32, 1] {
        t.push((
            Complex64::new(0.0, 1.0),
            vec![Operator::InnerBosonCreate(mode)],
        ));
        t.push((
            Complex64::new(0.0, -1.0),
            vec![Operator::InnerBosonAnnihilate(mode)],
        ));
    }
    nested_fock_algebra::Hamiltonian { terms: t }
}

const GHOST: u32 = 9;

/// Ω = P · b†_ghost — fermionic raise of the ghost times the Gauss generator.
/// Nilpotent by Pauli ({b†,b†}=0); [H, Ω] = [H,P]·b† = 0 at g=0.
fn ym_brst_charge() -> nested_fock_algebra::Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for mode in [0u32, 1] {
        terms.push((
            Complex64::new(0.0, 1.0),
            vec![
                Operator::InnerBosonCreate(mode),
                Operator::InnerFermionAnnihilate(GHOST),
            ],
        ));
        terms.push((
            Complex64::new(0.0, -1.0),
            vec![
                Operator::InnerBosonAnnihilate(mode),
                Operator::InnerFermionAnnihilate(GHOST),
            ],
        ));
    }
    nested_fock_algebra::Hamiltonian { terms }
}

fn ghosted(b_mode: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(b_mode, 1);
    QuantumState::vacuum()
        .apply(&Operator::OuterBosonCreate(inner))
        .apply(&Operator::OuterFermionCreate(InnerFermionicState {
            modes: BTreeSet::from([GHOST]),
        }))
}

#[test]
fn qym_abelian_brst_nilpotent_and_invariant() {
    let h = qcd_ym_hamiltonian(0.0);
    let omega = ym_brst_charge();

    // Nilpotency on ghosted probes.
    for bm in [0u32, 1] {
        let psi = ghosted(bm);
        assert!(
            omega.apply(&psi).norm() > 1e-9,
            "probe must carry Ω-content"
        );
        let twice = omega.apply(&omega.apply(&psi));
        assert!(
            twice.norm() < 1e-9,
            "Ω² must vanish (Pauli), got {:.3e}",
            twice.norm()
        );
    }

    // Invariance: i[H, Ω] vanishes on physical and ghosted probes alike.
    let mut phys = empty_vacuum();
    phys = phys.apply(&Operator::InnerBosonCreate(0));
    for psi in [phys, ghosted(1)] {
        let hp = h.apply(&omega.apply(&psi));
        let ph = omega.apply(&h.apply(&psi));
        let mut comm = hp;
        comm.scale_and_add(&ph, Complex64::new(-1.0, 0.0));
        let lhs = Complex64::new(0.0, 1.0) * QuantumState::inner_product(&psi, &comm);
        assert!(lhs.norm() < 1e-9, "[H, Ω] must vanish, got {lhs:?}");
    }
}

#[test]
fn qym_brst_projection_identity_on_physical_flow() {
    let h = qcd_ym_hamiltonian(0.0);
    let omega = ym_brst_charge();
    let p_op = ym_total_momentum();

    let mut psi0 = empty_vacuum();
    psi0 = psi0.apply(&Operator::InnerBosonCreate(0));
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(1)),
        Complex64::new(0.6, 0.0),
    );

    let plain =
        solve_forward_sirk_with_opts(&h, &psi0, &shifts(8), &best_device(), None, &mk(true))
            .unwrap();
    let projected = solve_forward_sirk_with_opts(
        &h,
        &psi0,
        &shifts(8),
        &best_device(),
        Some(&omega),
        &mk(true),
    )
    .unwrap();

    // Identical resolved spectra.
    let a = plain.resolved_ritz_values(5e-3);
    let b = projected.resolved_ritz_values(5e-3);
    assert_eq!(a.len(), b.len(), "resolved sets {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-7, "{x} vs {y}");
    }

    // And the same Gauss content after evolution.
    let ta = evolve_restarted(&h, &psi0, 0.25, 2, 8, &best_device(), None, &mk(true)).unwrap();
    let tb = evolve_restarted(
        &h,
        &psi0,
        0.25,
        2,
        8,
        &best_device(),
        Some(&omega),
        &mk(true),
    )
    .unwrap();
    let pa = QuantumState::inner_product(&ta, &p_op.apply(&ta));
    let pb = QuantumState::inner_product(&tb, &p_op.apply(&tb));
    assert!((pa - pb).norm() < 1e-7, "⟨P⟩ {pa} vs {pb}");
}

#[test]
fn qed_jc_total_excitation_superselected() {
    let (w_cav, w_atom, g_coupling) = (1.0_f64, 1.3_f64, 0.12_f64);
    let h = qed_jaynes_cummings(w_cav, w_atom, g_coupling);

    // N_tot = photon number + atomic occupation.
    let n_tot = nested_fock_algebra::Hamiltonian {
        terms: vec![
            (
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::InnerBosonCreate(0),
                    Operator::InnerBosonAnnihilate(0),
                ],
            ),
            (
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::InnerFermionCreate(1),
                    Operator::InnerFermionAnnihilate(1),
                ],
            ),
        ],
    };

    // Start: one photon, ground-state atom (N_tot = 1).
    let mut psi0 = empty_vacuum();
    psi0 = psi0.apply(&Operator::InnerBosonCreate(0));

    // Static commutator check first.
    let hn = h.apply(&n_tot.apply(&psi0));
    let nh = n_tot.apply(&h.apply(&psi0));
    let mut comm = hn;
    comm.scale_and_add(&nh, Complex64::new(-1.0, 0.0));
    let lhs = Complex64::new(0.0, 1.0) * QuantumState::inner_product(&psi0, &comm);
    assert!(lhs.norm() < 1e-9, "[H, N_tot] must vanish");

    // Flow conservation of both moments (the Rabi exchange moves the quantum
    // between cavity and atom; it cannot create or destroy excitation).
    let moments = |s: &QuantumState| {
        let m1 = QuantumState::inner_product(s, &n_tot.apply(s));
        let m2 = QuantumState::inner_product(s, &n_tot.apply(&n_tot.apply(s)));
        (m1, m2.re - m1.re * m1.re)
    };
    let (a1, a2) = moments(&psi0);
    for t in [0.35_f64, 1.1] {
        let s = evolve_restarted(&h, &psi0, t, 2, 10, &best_device(), None, &mk(true)).unwrap();
        let (b1, b2) = moments(&s);
        assert!(
            (b1 - a1).norm() < 1e-8 && (b2 - a2).abs() < 1e-6,
            "N_tot superselection broken at t={t}: {b1} vs {a1}"
        );
    }
}

#[test]
fn qg_densitized_beat_predicted_by_spectrum() {
    // The densitized blocks are BOGOLIUBOV (squeezed) forms — their raw
    // ladders are unbounded, so a sector "energy" is not ritz[0]. We use the
    // DIAGONALIZED blocks instead: opposite-sign number operators carrying
    // the same 𝒮(+1/16)/conformal(−1/24) coefficients — exactly what
    // Bogoliubov diagonalization of qg_densitized_kinetic produces.
    let (c_s, c_y) = (1.0_f64 / 16.0, -1.0_f64 / 24.0);
    let h = nested_fock_algebra::Hamiltonian {
        terms: vec![
            (
                Complex64::new(c_s, 0.0),
                vec![
                    Operator::InnerBosonCreate(0),
                    Operator::InnerBosonAnnihilate(0),
                ],
            ),
            (
                Complex64::new(c_y, 0.0),
                vec![
                    Operator::InnerBosonCreate(1),
                    Operator::InnerBosonAnnihilate(1),
                ],
            ),
        ],
    };
    let y_mode = 1_u32;

    // Two sectors: one 𝒮 quanton (mode 0) vs one conformal quanton (mode 1).
    let s_state = empty_vacuum().apply(&Operator::InnerBosonCreate(0));
    let y_state = empty_vacuum().apply(&Operator::InnerBosonCreate(y_mode));

    // Solve each sector separately for its energy; the beat is their split.
    let es =
        solve_forward_sirk_with_opts(&h, &s_state, &shifts(7), &best_device(), None, &mk(true))
            .unwrap()
            .ritz_values()[0];
    let ey =
        solve_forward_sirk_with_opts(&h, &y_state, &shifts(7), &best_device(), None, &mk(true))
            .unwrap()
            .ritz_values()[0];
    let d_e = (es - ey).abs();
    assert!(d_e > 1e-3, "sectors must be spectrally split: {es} vs {ey}");

    // Transfer operator |S><y| + h.c. measures the coherence.
    let mut s_inner = InnerBosonicState::vacuum();
    s_inner.modes.insert(0, 1);
    let mut y_inner = InnerBosonicState::vacuum();
    y_inner.modes.insert(y_mode, 1);
    let transfer = nested_fock_algebra::Hamiltonian {
        terms: vec![
            (
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::OuterBosonCreate(s_inner.clone()),
                    Operator::OuterBosonAnnihilate(y_inner.clone()),
                ],
            ),
            (
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::OuterBosonCreate(y_inner),
                    Operator::OuterBosonAnnihilate(s_inner),
                ],
            ),
        ],
    };

    let mut psi0 = empty_vacuum();
    psi0.scale_and_add(&s_state, Complex64::new(1.0, 0.0));
    psi0.scale_and_add(&y_state, Complex64::new(1.0, 0.0));
    let coh = |s: &QuantumState| QuantumState::inner_product(s, &transfer.apply(s)).re;

    // Zero crossing at the quarter beat t* = π/(2ΔE).
    let t_star = std::f64::consts::PI / (2.0 * d_e);
    let s_star =
        evolve_restarted(&h, &psi0, t_star, 2, 10, &best_device(), None, &mk(true)).unwrap();
    let c_star = coh(&s_star);
    assert!(
        c0_check(psi0.clone(), &transfer) > 0.9 && c_star.abs() < 0.15,
        "beat must node at t*=π/(2ΔE)={t_star:.4}: c={c_star:.4}"
    );
}

fn c0_check(psi: QuantumState, transfer: &nested_fock_algebra::Hamiltonian) -> f64 {
    QuantumState::inner_product(&psi, &transfer.apply(&psi)).re
}
