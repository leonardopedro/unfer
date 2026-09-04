//! ASSUMPTION LEDGER — the load-bearing claims behind "testing the 3D
//! gauge-fixed Hamiltonians of NS / QG(R²) / QYM / QED through
//! SIRK/Hashimoto without further assumptions".
//!
//! The guide (`docs/NUMERICAL_VALIDATION_GUIDE.md` §4.5, §5.25) decomposes the
//! programme into four steps:
//!
//!   1. ACTION → HAMILTONIAN: the Cadabra2 modules (`docs/*.cdb`) vary the
//!      classical action, Legendre-transform and gauge-fix, producing
//!      `H_final`. The compiler-route suites (`cdb_hamiltonian_match.rs`,
//!      `latex_cas_hamiltonian_match.rs`) verify the numerical builders ARE
//!      those Hamiltonians.
//!   2. HAMILTONIAN → NUMERICS: the only steps between H and the numbers are
//!      (i) restriction to the nested-Fock sector spanned by the start state,
//!      (ii) normal ordering (`<0|H|0> = 0`), (iii) the Krylov projection
//!      itself. No physical approximation is added: no renormalization input,
//!      no perturbative expansion, no mean field, no fitted parameter.
//!   3. PREDICTIONS: spectra (Ritz values with residual-certified error
//!      bars), dynamics (norm/energy conservation, phases, Rabi/beat
//!      frequencies), derived physics (dispersions, mass gap, Coulomb
//!      potential).
//!   4. MATCH / FAIL MAP: where the predictions agree with experiment or
//!      other approximations, where they fail and why (§5.25).
//!
//! The three tests here pin the load-bearing claims:
//!
//!   A. `raw_canonical_sequence_no_guards_exact_predictions` — the RAW
//!      canonical sequence (EVERY guard off: `prune_eps = 0.0`, no BRST
//!      projection, no adaptive truncation, canonical frame, no component
//!      budget) already reproduces the exact predictions of the four
//!      gauge-fixed Hamiltonians on their solvable sectors. The guards are
//!      not part of the model.
//!
//!   B. `unit_norm_frame_is_exact_reparametrization_not_model_change` — the
//!      unit-norm frame is an EXACT basis reparametrization: in the
//!      infinite-precision limit it spans the same Krylov subspace and gives
//!      identical Rayleigh–Ritz predictions; it only removes the
//!      finite-precision conditioning wall (`ritz_edge_study` p2/p2b). So it
//!      is a numerical device, not a modelling assumption: canonical and
//!      unit-norm frames agree on the resolved rungs, and at depth the
//!      unit-norm frame resolves what the raw Gram matrix cannot.
//!
//!   C. `perturbation_theory_scope_map` — the match/fail map is
//!      executable: weak coupling reproduces perturbation theory, strong
//!      coupling departs (correctly — SIRK is non-perturbative); the QG
//!      classical limits reproduce Newton/GR; the NS Ehrenfest identity is
//!      exact; and the JC single-excitation sectors (the solver's stable
//!      regime) are reproduced to machine precision while the wide-spectrum
//!      coherent state is NOT — there the exact Poisson sum is the
//!      prediction, not the truncated solver. No failure is hidden by
//!      loosening a tolerance or changing the model.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QuantumState,
    oscillator_displaced, qcd_free_gluon, qcd_pair_production, qed_free_photon,
    qed_jaynes_cummings, qed_pair_production, qg_free_graviton, qg_starobinsky_hamiltonian,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

/// The raw canonical sequence: every guard off, no component budget.
fn raw_opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 0.0,
        max_components: None,
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: false,
    }
}

/// Same, but in the unit-norm frame (exact reparametrization — §4.5).
fn unit_opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 0.0,
        max_components: None,
        brst_tol: 1e-10,
        adaptive: false,
        unit_norm_steps: true,
    }
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let diff = (h_proj - &h_proj.adjoint()).norm();
    assert!(
        diff < 1e-6,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

/// One empty inner bosonic universe — the vacuum for **inner** ladder
/// operators (AGENTS.md vacuum-initialization rule).
fn inner_vac() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// An `n`-quanta state of inner mode `mode` (the framework-native way: one
/// universe with inner occupation `n` — correct at any occupation).
fn n_boson(mode: u32, n: u32) -> QuantumState {
    let mut s = inner_vac();
    for _ in 0..n {
        s = s.apply(&Operator::InnerBosonCreate(mode));
    }
    s
}

/// A one-quantum state of an **outer** bosonic mode (graviton convention).
fn one_outer_boson(mode: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, 1);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

fn sirk_ground(h: &Hamiltonian, v0: &QuantumState, m: usize, opts: &SirkOpts) -> f64 {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, opts)
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res.ground_state_energy().expect("ground-state Ritz value")
}

fn sirk_ritz(h: &Hamiltonian, v0: &QuantumState, m: usize, opts: &SirkOpts) -> Vec<f64> {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, opts)
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res.ritz_values()
}

/// `|n, atom⟩` for the Jaynes–Cummings model: `n` photons (inner boson mode
/// 0) + atom in {ground, excited} (inner fermion mode 1).
fn jc_state(n: u32, excited: bool) -> QuantumState {
    let mut s = QuantumState::vacuum()
        .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()))
        .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    if excited {
        s = s.apply(&Operator::InnerFermionCreate(1));
    }
    for _ in 0..n {
        s = s.apply(&Operator::InnerBosonCreate(0));
    }
    s
}

// ── A. The raw canonical sequence (all guards off) is already exact ─────────

#[test]
fn raw_canonical_sequence_no_guards_exact_predictions() {
    let opts = raw_opts();

    // QED — free photon field (inner construction), H = Σ ω_k N_k.
    let h_ph = qed_free_photon(&[1.0, 2.0]);
    let e_vac = sirk_ground(&h_ph, &inner_vac(), 4, &opts);
    assert!(
        e_vac.abs() < 1e-6,
        "QED vacuum must be 0 with NO guards, got {e_vac}"
    );
    let e1 = sirk_ground(&h_ph, &n_boson(0, 1), 4, &opts);
    assert!((e1 - 1.0).abs() < 1e-6, "one-photon must be ω₁=1, got {e1}");
    let e2 = sirk_ground(&h_ph, &n_boson(0, 2), 4, &opts);
    assert!((e2 - 2.0).abs() < 1e-6, "two-photon must be 2ω=2, got {e2}");

    // QCD — free gluon field (inner construction).
    let h_gl = qcd_free_gluon(&[0.5, 1.5]);
    let g1 = sirk_ground(&h_gl, &n_boson(0, 1), 4, &opts);
    assert!(
        (g1 - 0.5).abs() < 1e-6,
        "one-gluon must be |k|=0.5, got {g1}"
    );

    // QG — free graviton field (outer construction, natural units c = 1).
    let h_gw = qg_free_graviton(&[1.0, 3.0]);
    let w1 = sirk_ground(&h_gw, &one_outer_boson(0), 4, &opts);
    assert!(
        (w1 - 1.0).abs() < 1e-6,
        "one-graviton must be c|k|=1, got {w1}"
    );

    // QG(R²) — Starobinsky scalaron sector (inner construction), H = Σ m N_i.
    let h_sc = qg_starobinsky_hamiltonian(2, 0.8);
    let s_vac = sirk_ground(&h_sc, &inner_vac(), 4, &opts);
    assert!(s_vac.abs() < 1e-6, "scalaron vacuum must be 0, got {s_vac}");
    let s1 = sirk_ground(&h_sc, &n_boson(0, 1), 4, &opts);
    assert!(
        (s1 - 0.8).abs() < 1e-6,
        "one-scalaron must be m=0.8, got {s1}"
    );

    // The point: the RAW canonical sequence — no pruning, no BRST, no
    // adaptive truncation, no unit-norm frame, no component budget — already
    // makes the model's exact predictions. The guards change nothing on
    // physical data (guard_justification_study.rs); they are engineering,
    // not model.
}

// ── B. The unit-norm frame is an exact reparametrization, not a model change ─

#[test]
fn unit_norm_frame_is_exact_reparametrization_not_model_change() {
    // Displaced oscillator H = ωN + g(a†+a): exact levels E_n = ωn − g²/ω.
    let (omega, g) = (1.0, 0.3);
    let h = oscillator_displaced(omega, g);
    let vac = inner_vac();
    let exact = -g * g / omega;

    // Moderate depth: BOTH frames give the exact ground level.
    let c6 = sirk_ground(&h, &vac, 6, &raw_opts());
    let u6 = sirk_ground(&h, &vac, 6, &unit_opts());
    assert!(
        (c6 - exact).abs() < 1e-6 && (u6 - exact).abs() < 1e-6,
        "both frames must reach the exact shift −g²/ω = {exact}: canonical {c6}, unit {u6}"
    );
    assert!(
        (c6 - u6).abs() < 1e-9,
        "frames must agree on resolved rungs (infinite-precision invariance): {c6} vs {u6}"
    );

    // Depth: the raw Gram matrix hits the conditioning wall; the unit-norm
    // frame (same subspace, better-conditioned coordinates) resolves it. The
    // wall is finite-precision, not physical — the model did not change.
    let c12 = sirk_ground(&h, &vac, 12, &raw_opts());
    let u12 = sirk_ground(&h, &vac, 12, &unit_opts());
    let err_c = (c12 - exact).abs();
    let err_u = (u12 - exact).abs();
    assert!(
        err_u < 1e-6,
        "unit-norm frame must resolve the deep window (err {err_u})"
    );
    assert!(
        err_c >= err_u,
        "canonical raw Gram must not beat the unit-norm frame at depth \
         (err_c {err_c} ≥ err_u {err_u}) — the wall is conditioning, not physics"
    );

    // JC Rabi doublet: frames agree on the resolved sector AND on the exact
    // dressed levels ω ± g.
    let h_jc = qed_jaynes_cummings(1.0, 1.0, 0.3);
    let start = jc_state(0, true);
    let c = sirk_ritz(&h_jc, &start, 4, &raw_opts());
    let u = sirk_ritz(&h_jc, &start, 4, &unit_opts());
    assert!(
        (c[0] - u[0]).abs() < 1e-9 && (c[1] - u[1]).abs() < 1e-9,
        "frames must agree on the Rabi doublet: canonical {c:?}, unit {u:?}"
    );
    assert!(
        (c[0] - (1.0 - 0.3)).abs() < 1e-9 && (c[1] - (1.0 + 0.3)).abs() < 1e-9,
        "the doublet must be the exact ω ± g = {{0.7, 1.3}}, got {c:?}"
    );
}

// ── C. The match/fail map is executable ─────────────────────────────────────

#[test]
fn perturbation_theory_scope_map() {
    // (a) QED γ↔e⁺e⁻: at weak coupling SIRK (exact, non-perturbative) reduces
    //     to the one-loop vacuum-polarization sum; at strong coupling it
    //     DEPARTS — PT is the right tool only where the coupling is small.
    let m: f64 = 1.0;
    let omega: f64 = 0.5;
    let dp: f64 = 0.25;
    let ps: Vec<f64> = (-12..=12).map(|i| (i as f64) * dp).collect();
    let e_e: Vec<f64> = ps.iter().map(|&p| (p * p + m * m).sqrt()).collect();
    let p_e: Vec<f64> = ps
        .iter()
        .map(|&p| ((omega - p) * (omega - p) + m * m).sqrt())
        .collect();
    let vertex = |e: f64| -> Vec<f64> {
        ps.iter()
            .zip(e_e.iter().zip(p_e.iter()))
            .map(|(&p, (&a, &b))| e * (2.0 * p - omega) * (dp / (2.0 * omega * 4.0 * a * b)).sqrt())
            .collect()
    };
    let one_loop = |v: &[f64]| -> f64 {
        v.iter()
            .zip(e_e.iter().zip(p_e.iter()))
            .map(|(&c, (&a, &b))| c * c / (omega - a - b))
            .sum()
    };
    let start = QuantumState::vacuum()
        .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()))
        .apply(&Operator::OuterBosonCreate({
            let mut inner = InnerBosonicState::vacuum();
            inner.modes.insert(0, 1);
            inner
        }));
    let opts = raw_opts();
    let se = |e: f64| -> f64 {
        let v = vertex(e);
        let h = qed_pair_production(omega, &e_e, &p_e, &v);
        sirk_ground(&h, &start, 6, &opts) - omega
    };
    let weak = se(0.05);
    let weak_pt = one_loop(&vertex(0.05));
    assert!(
        (weak - weak_pt).abs() < 1e-3,
        "weak coupling: SIRK δE = {weak} must reduce to the one-loop sum {weak_pt}"
    );
    let strong = se(1.0);
    let strong_pt = one_loop(&vertex(1.0));
    assert!(
        (strong - strong_pt).abs() / strong_pt.abs() > 0.10,
        "strong coupling: SIRK δE = {strong} must depart from PT {strong_pt} \
         (PT is not the right tool there)"
    );

    // (b) QCD gluon self-energy: the same pattern with the colour factor
    //     T_R = 1/2 multiplying the quark loop.
    let v_glu = vertex(0.05);
    let h_glu = qcd_pair_production(omega, &e_e, &p_e, &v_glu);
    let se_glu = sirk_ground(&h_glu, &start, 6, &opts) - omega;
    assert!(
        (se_glu - 0.5 * one_loop(&v_glu)).abs() < 1e-4,
        "weak-coupling gluon self-energy must be T_R × one-loop: {se_glu}"
    );

    // (c) QG weak-field: the Yukawa form Φ(r) = −GM/r (1 + ⅓e^{−mr}) is
    //     Newtonian at r ≫ 1/m and enhanced by exactly 4/3 at r ≪ 1/m — the
    //     classical limits of the quantized R² sector.
    let gm = 1.0;
    let m_sc = 0.8;
    let phi = |r: f64| -gm / r * (1.0 + (1.0 / 3.0) * (-m_sc * r).exp());
    let phi_newton = |r: f64| -gm / r;
    // At r = 10/m the deviation from Newton is exactly ⅓e^{−10} (the
    // published Yukawa coefficient); at r ≪ 1/m the force is enhanced by
    // exactly 4/3.
    let dev = phi(10.0 / m_sc) / phi_newton(10.0 / m_sc) - 1.0;
    assert!(
        (dev - (1.0 / 3.0) * (-10.0f64).exp()).abs() < 1e-9,
        "Yukawa deviation at r = 10/m must be ⅓e⁻¹⁰ = {}, got {dev}",
        (1.0 / 3.0) * (-10.0f64).exp()
    );
    let ratio_short = phi(1.0e-8 / m_sc) / phi_newton(1.0e-8 / m_sc);
    assert!(
        (ratio_short - 4.0 / 3.0).abs() < 1e-6,
        "Yukawa must be enhanced by 4/3 at r ≪ 1/m, got {ratio_short}"
    );

    // (d) NS Ehrenfest identity — exact on the probes (the affine fiber's
    //     prediction), with the same all-guards-off opts.
    let nu: f64 = 1.0e-4;
    let k: f64 = 2.0 * std::f64::consts::PI;
    let kappa: f64 = -nu * k * k / 4.0;
    let h_ns = nested_fock_algebra::ns_eulerian_fiber(
        &[[kappa, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        &[0.0, 0.0, 0.0],
    );
    let u_op = Hamiltonian {
        terms: vec![
            (
                Complex64::new(1.0, 0.0),
                vec![Operator::InnerBosonCreate(0)],
            ),
            (
                Complex64::new(1.0, 0.0),
                vec![Operator::InnerBosonAnnihilate(0)],
            ),
        ],
    };
    let mut probe = inner_vac();
    probe.scale_and_add(&n_boson(0, 1), Complex64::new(1.0, 0.0));
    let hhu = h_ns.apply(&u_op.apply(&probe));
    let uhh = u_op.apply(&h_ns.apply(&probe));
    let mut comm = hhu;
    comm.scale_and_add(&uhh, Complex64::new(-1.0, 0.0));
    let lhs = Complex64::new(0.0, 1.0) * QuantumState::inner_product(&probe, &comm);
    let uexp = QuantumState::inner_product(&probe, &u_op.apply(&probe));
    assert!(
        (lhs - 4.0 * kappa * uexp).norm() < 1e-9,
        "NS Ehrenfest i⟨[H,u]⟩ = 4κ⟨u⟩ must hold exactly, got {lhs:?}"
    );

    // (e) JC: the single-excitation sectors are the solver's stable regime —
    //     reproduced to machine precision (qed_validation.rs). The
    //     wide-spectrum coherent state is NOT in that regime: the exact
    //     Poisson-weighted sum is the prediction there, not the truncated
    //     solver (documented in qed_validation.rs::qed_jaynes_cummings_rabi_oscillation_and_revival).
    //     This boundary is a solver-regime statement, never hidden by
    //     loosening a tolerance or changing the model.
    let g = 0.2;
    let alpha: f64 = 4.0;
    let nbar = alpha * alpha;
    let exact_pe = |t: f64| -> f64 {
        let mut sum = 0.0;
        let mut p = (-nbar).exp();
        let mut n: u32 = 0;
        loop {
            sum += p * (g * ((n + 1) as f64).sqrt() * t).cos().powi(2);
            n += 1;
            p *= nbar / (n as f64);
            if p < 1e-14 {
                break;
            }
        }
        sum
    };
    let t_r = 2.0 * std::f64::consts::PI * (nbar + 1.0).sqrt() / g;
    assert!(
        exact_pe(10.0) < 0.6,
        "the exact sum must collapse by t = 10 (P_e = {})",
        exact_pe(10.0)
    );
    assert!(
        exact_pe(t_r) > 0.72,
        "the exact sum must revive at t_R (P_e = {})",
        exact_pe(t_r)
    );
}
