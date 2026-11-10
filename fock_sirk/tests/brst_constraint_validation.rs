//! Constraint and superselection structure through SIRK flows:
//! an abelian Yang–Mills BRST charge built from the residual Gauss generator,
//! QED total-excitation superselection under detuned Jaynes–Cummings flow,
//! and a QG densitized-model beat whose period is PREDICTED by the
//! independently solved Ritz spectrum (spectroscopy → dynamics closure).
//!
//! 1. `qym_abelian_brst_nilpotent_and_invariant` — Ω = P·b_ghost:
//!    Ω²=0 (Pauli, on probes that carry Ω-content), and [H(g=0), Ω]=0 at
//!    operator level because B = A_0 − A_1 is the difference and P = Σ Π the
//!    sum of the two modes' momenta, so [B, P] = i − i = 0.
//! 2. `qym_brst_projection_is_identity_on_ker_q` — the mid-sequence projection
//!    is exactly the identity on the ghost-free flow (which is all ker Q here,
//!    since Ω = P·b annihilates every ghost-free state by construction), and
//!    it does not converge on the ghost-carrying states where it would act.
//! 3. `qed_jc_total_excitation_superselected` — N_tot = a†a + e†e commutes
//!    with the detuned JC Hamiltonian; mean AND variance of N_tot are flow
//!    constants (Rabi exchange moves quanta BETWEEN subsystems only).
//! 4. `qg_densitized_two_level_beat_matches_analytic_splitting` — a
//!    TWO-LEVEL COMPARISON MODEL (hand-entered 1/16, −1/24; the densitized
//!    builder is not solved): each sector's Ritz value reproduces the closed
//!    form, so the splitting is exactly 5/48 and the flow's coherence node sits
//!    at π/(2ΔE). Solver self-consistency, not a physical prediction.

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

const GHOST: u32 = 9;

/// Ω = P · b_ghost — the Gauss generator times the ghost *lowering*, which is
/// the raise only in the filled-sea convention of `ym_brst_charge`. On the
/// empty reference of this file it is the annihilation, so Ω is identically
/// zero on every ghost-free state — by construction, not as a physical
/// statement. Only ghost-carrying probes carry Ω-content. Nilpotent by Pauli
/// (`b² = 0`); `[H, Ω] = [H, P]·b = 0` at g = 0.
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

    // Invariance, as an operator statement. The earlier form of this check was
    // `i<psi|[H, Omega]|psi>`, which is zero for *every* state here whatever
    // the commutator does: Omega = P·b flips the ghost number and H conserves
    // it, so the inner product pairs orthogonal sectors. The vector norm is
    // the check that can fail.
    for psi in [ghosted(0), ghosted(1)] {
        let applied = omega.apply(&psi);
        let mut difference = h.apply(&applied);
        difference.scale_and_add(&omega.apply(&h.apply(&psi)), Complex64::new(-1.0, 0.0));
        assert!(
            difference.norm() < 1e-9,
            "[H, Omega] must vanish as an operator, got {:e}",
            difference.norm()
        );
        eprintln!(
            "brst: |psi|={:e} |Omega psi|={:e} |H Omega psi|={:e} |[H,Omega] psi|={:e}",
            psi.norm(),
            applied.norm(),
            h.apply(&applied).norm(),
            difference.norm()
        );
    }

    // And the ghost-free sector is Omega's kernel by construction: P·b kills a
    // state with no ghost, so `Omega|phys> = 0` says nothing about Gauss.
    let mut phys = empty_vacuum().apply(&Operator::InnerBosonCreate(0));
    phys.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(1)),
        Complex64::new(0.6, 0.0),
    );
    assert!(
        omega.apply(&phys).norm() == 0.0,
        "P·b must annihilate a ghost-free state identically"
    );
}

/// What the mid-sequence projection actually does on this flow.
///
/// `project_physical` returns its input unchanged whenever `‖Q w‖` is below
/// tolerance, and here `Q = P·b_ghost` annihilates every ghost-free state
/// identically. The earlier version of this test compared projected and
/// unprojected solves from a ghost-free start and called the agreement a
/// theorem; it is the projector's own short-circuit, so it cannot fail. What
/// can fail is measured instead:
///
/// * on the ghost-free flow `Q` is exactly zero and the projection is the
///   identity to the last bit (`‖Pw − w‖ = 0`);
/// * on a ghost-carrying state `‖Q w‖ = 2.0` but the projected solve does not
///   converge — `BrstNotConverged { residual: 0.39040823231178456 }` at the
///   configured tolerance and 50 iterations. `Q Q†` is singular on the
///   subspace that carries the charge, so the projector is *untested where it
///   would act*; that is a limitation of the switch, not a statement about
///   gauge invariance.
#[test]
fn qym_brst_projection_is_identity_on_ker_q() {
    let h = qcd_ym_hamiltonian(0.0);
    let omega = ym_brst_charge();

    let mut psi0 = empty_vacuum();
    psi0 = psi0.apply(&Operator::InnerBosonCreate(0));
    psi0.scale_and_add(
        &empty_vacuum().apply(&Operator::InnerBosonCreate(1)),
        Complex64::new(0.6, 0.0),
    );

    // The identity on ker Q, measured rather than assumed.
    assert!(
        omega.apply(&psi0).norm() == 0.0,
        "the ghost-free start must be in ker Q exactly"
    );
    let projected =
        fock_sirk::brst::project_physical(&psi0, &omega, mk(true).brst_tol, 50).unwrap();
    let mut difference = projected.clone();
    difference.scale_and_add(&psi0, Complex64::new(-1.0, 0.0));
    assert!(
        difference.norm() == 0.0,
        "the projection must be the identity on ker Q, got {:e}",
        difference.norm()
    );

    // Which is why the solver-level agreement is plumbing, not physics: with
    // the switch on, nothing in the Krylov sequence is ever touched.
    let plain =
        solve_forward_sirk_with_opts(&h, &psi0, &shifts(8), &best_device(), None, &mk(true))
            .unwrap();
    let with_projection = solve_forward_sirk_with_opts(
        &h,
        &psi0,
        &shifts(8),
        &best_device(),
        Some(&omega),
        &mk(true),
    )
    .unwrap();
    let a = plain.resolved_ritz_values(5e-3);
    let b = with_projection.resolved_ritz_values(5e-3);
    assert_eq!(a.len(), b.len(), "resolved sets {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-9, "{x} vs {y}");
    }

    // Off ker Q the projector is exercised for the first time — and fails.
    let ghost = ghosted(1);
    let charge = omega.apply(&ghost).norm();
    assert!(
        (charge - 2.0).abs() < 1e-12,
        "the ghosted probe must carry Q-content: {charge:e}"
    );
    match fock_sirk::brst::project_physical(&ghost, &omega, mk(true).brst_tol, 50) {
        Ok(projected) => {
            let mut residual = omega.apply(&projected);
            residual.scale_and_add(&omega.apply(&ghost), Complex64::new(-1.0, 0.0));
            eprintln!(
                "brst: projected a ghosted state, |Q proj|={:e}",
                omega.apply(&projected).norm()
            );
            assert!(
                omega.apply(&projected).norm() <= mk(true).brst_tol,
                "if the projection succeeds it must land in ker Q"
            );
        }
        Err(fock_sirk::linalg::SirkError::BrstNotConverged { residual }) => {
            eprintln!("brst: projection off ker Q does not converge, residual={residual:e}");
            assert!(
                (residual - 0.390_408_232_311_784_56).abs() < 1e-9,
                "the projector's residual off ker Q moved: {residual:e}"
            );
        }
        Err(other) => panic!("unexpected projection error: {other:?}"),
    }
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

    // Static commutator check first, as a vector norm rather than the single
    // expectation `i<psi|[H,N]|psi>` — the expectation cannot see a component
    // of [H,N]|psi> that is orthogonal to psi.
    let mut comm = h.apply(&n_tot.apply(&psi0));
    comm.scale_and_add(&n_tot.apply(&h.apply(&psi0)), Complex64::new(-1.0, 0.0));
    assert!(
        comm.norm() < 1e-9,
        "[H, N_tot] must vanish, got {:e}",
        comm.norm()
    );

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

/// A TWO-LEVEL COMPARISON MODEL, not the densitized builder.
///
/// `c_s = 1/16` and `c_y = −1/24` are hand-entered here; `qg_densitized_kinetic`
/// is not solved in this test. On a two-level Hamiltonian the node of the
/// transfer coherence is `π/(2ΔE)` for *any* splitting, so "the spectrum
/// predicts the beat" was never a physical prediction — it is the standard
/// two-level identity. What is worth asserting, and is asserted below, is that
/// each sector's Ritz value reproduces the closed form (`1/16`, `−1/24`), that
/// their difference is therefore exactly `5/48`, and that the flow puts the
/// coherence node where that splitting says it must. A mapping from
/// `qg_densitized_kinetic` to these two number operators is not established
/// here.
#[test]
fn qg_densitized_two_level_beat_matches_analytic_splitting() {
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
    assert!(
        (es - c_s).abs() < 1e-10 && (ey - c_y).abs() < 1e-10,
        "sector Ritz values must reproduce the entered coefficients: {es} vs {c_s}, {ey} vs {c_y}"
    );
    let d_e = (es - ey).abs();
    assert!(
        (d_e - (c_s - c_y).abs()).abs() < 1e-10,
        "the splitting must be |1/16 + 1/24| = 5/48 = {:.12}, got {d_e:.12}",
        (c_s - c_y).abs()
    );

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
        coh(&psi0) > 0.9 && c_star.abs() < 0.15,
        "beat must node at t*=π/(2ΔE)={t_star:.4}: c0={:.6}, c={c_star:.4}",
        coh(&psi0)
    );
}
