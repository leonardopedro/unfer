//! Graviton polarization content of the TEGR gauge-fixed Hamiltonian — SIRK.
//!
//! The TEGR/teleparallel kinetic of the 3D gauge-fixed Hamiltonian from
//! `docs/qg_gauge_fixed_hamiltonian.cdb` (book.tex line 8190) is realized in
//! Fock space by [`qg_tegr_hamiltonian`]: each mode carries the normal-ordered
//! 𝒮-sector kinetic `c·(B†B − ½(B†² + B²))` with `c = 1/16`. In quadratures
//! this is `c·(P² − ½)` — a pure momentum-squared operator per mode — so:
//!
//!   • its spectrum is the half-line `[−c/2, ∞) = [−1/32, ∞)` (bounded below,
//!     continuous — the finite-shadow statement of the essential
//!     self-adjointness of the densitized d'Alembertian, Strichartz 1973);
//!     the normal-ordered ground dips to −1/32, not 0;
//!   • the modes are independent, identical copies — one *polarization
//!     direction* of the tetrad momentum each. The two modes of
//!     `qg_tegr_hamiltonian(2)` are the two transverse polarizations of the
//!     graviton sector in the linearized theory (the helicity-±2 pair), so
//!     both share one frequency: **polarization (helicity) degeneracy**.
//!
//! Verified here through SIRK–Hashimoto:
//!
//! 1. `qg_tegr_polarizations_degenerate` — the two polarizations have
//!    IDENTICAL 𝒮-kinetics: the SIRK spectrum from |1₀⟩, from |1₁⟩ and from
//!    the symmetric one-quantum superposition coincide to solver precision —
//!    the two helicity states of the graviton share the dispersion.
//! 2. `qg_tegr_kinetic_continuum_edge_bounded_below` — the normal-ordered
//!    kinetic is bounded below by the exact edge −1/32 per mode: the lowest
//!    SIRK value from the vacuum is never below `−c/2` (1 mode) / `−2c/2`
//!    (2 modes), is genuinely negative (the ground dips below 0), and the
//!    resolved spectrum has positive gaps — the exact sharpening of the
//!    loose "bounded below" band of §5.17.
//! 3. `qg_tegr_flow_conserves_norm_energy` — the restarted SIRK flow from a
//!    one-quantum graviton conserves norm and energy while the 𝒮 pair terms
//!    populate both polarization ladders (the squeezing dynamics).

//!
//! **Ground-state doctrine** — the negative "-1/32 floor" above is the
//! INNER (one-particle) level statement: the normal-ordered one-particle
//! kinetic's truncated continuum edge, exactly what the one-particle
//! constant shift compensates. The nested theory's final Hamiltonian (see
//! `outer_vacuum_ground_validation.rs`) is the one-particle Hamiltonian
//! enclosed in outer creation (left) / annihilation (right) operators — its
//! ground state is ALWAYS the outer-Fock vacuum at energy 0.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{evolve_restarted, solve_forward_sirk_with_opts, SirkOpts};
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, QuantumState, qg_tegr_hamiltonian,
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
        // The unit-norm frame (exact basis reparametrization, see
        // ritz_edge_study p2/p2b) flattens the raw frame's Gram wall for the
        // wide spectral window of the indefinite 𝒮-kinetic.
        unit_norm_steps: true,
    }
}

/// `n` quanta in the outer-ladder polarization mode `mode` (one universe per
/// quantum, each carrying inner occupation `{mode: 1}` — the ladder of
/// `qg_tegr_hamiltonian`).
fn polarization_quantum(mode: u32, n: u32) -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(mode, 1);
    let mut s = QuantumState::vacuum();
    for _ in 0..n {
        s = s.apply(&Operator::OuterBosonCreate(inner.clone()));
    }
    s
}

/// The bare vacuum (no universes).
fn vacuum() -> QuantumState {
    QuantumState::vacuum()
}

/// Normalized superposition (|1₀⟩ + |1₁⟩)/√2 of one quantum in each polarization.
fn one_in_each() -> QuantumState {
    let s0 = polarization_quantum(0, 1);
    let s1 = polarization_quantum(1, 1);
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    let mut s = s0;
    s.scale_and_add(&s1, Complex64::new(1.0, 0.0));
    s.scale_and_add(&s.clone(), Complex64::new(inv_sqrt2 - 1.0, 0.0));
    s
}

fn ritz_of(h: &Hamiltonian, v0: &QuantumState, m: usize) -> Vec<f64> {
    solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .unwrap()
        .ritz_values()
}

#[test]
fn qg_tegr_polarizations_degenerate() {
    // The two tetrad-momentum polarizations carry IDENTICAL 𝒮-kinetics
    // (both coefficient +1/16): the SIRK spectrum of polarization 0 equals
    // that of polarization 1 to solver precision, and the symmetric
    // one-quantum superposition gives the same spectrum — the two helicity
    // states of the graviton share the dispersion (one |k|, one frequency).
    let h = qg_tegr_hamiltonian(2);
    let spec0 = ritz_of(&h, &polarization_quantum(0, 1), 10);
    let spec1 = ritz_of(&h, &polarization_quantum(1, 1), 10);
    assert_eq!(spec0.len(), spec1.len());
    for (a, b) in spec0.iter().zip(spec1.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "polarization spectra must be identical: {a} vs {b}"
        );
    }
    let spec_sym = ritz_of(&h, &one_in_each(), 10);
    assert_eq!(spec_sym.len(), spec0.len());
    for (a, b) in spec0.iter().zip(spec_sym.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "symmetric superposition must give the same spectrum: {a} vs {b}"
        );
    }
    eprintln!(
        "qg_tegr_polarizations_degenerate: spectrum = {:?}, identical across polarizations",
        &spec0[..spec0.len().min(5)]
    );
}

#[test]
fn qg_tegr_kinetic_continuum_edge_bounded_below() {
    // H_mode = (1/16)(P² − ½): spectrum [−1/32, ∞) per mode. The SIRK ground
    // from the vacuum never dips below the exact edge −c/2 per mode (a
    // principal submatrix of the positive P² cannot go negative below it),
    // is genuinely negative (the normal-ordered kinetic's ground is −1/32,
    // not 0), and the resolved window keeps positive gaps.
    let c = 1.0 / 16.0;

    // One mode: edge −c/2 = −1/32.
    let h1 = qg_tegr_hamiltonian(1);
    let ritz1 = ritz_of(&h1, &vacuum(), 10);
    assert!(
        ritz1[0] > -c / 2.0 - 1e-6,
        "1 mode: ground {} must never fall below the continuum edge −1/32",
        ritz1[0]
    );
    assert!(
        ritz1[0] < -0.01,
        "1 mode: ground {} must be genuinely negative (normal-ordered kinetic)",
        ritz1[0]
    );
    for w in ritz1.windows(2) {
        assert!(w[1] - w[0] > -1e-9, "positive gaps required: {ritz1:?}");
    }

    // Two modes: each contributes ≥ −c/2, so the joint ground ≥ −2c/2 = −1/16.
    let h2 = qg_tegr_hamiltonian(2);
    let ritz2 = ritz_of(&h2, &vacuum(), 10);
    assert!(
        ritz2[0] > -2.0 * c / 2.0 - 1e-6,
        "2 modes: ground {} must never fall below −1/16 (two continuum edges)",
        ritz2[0]
    );
    assert!(
        ritz2[0] < 0.0,
        "2 modes: ground {} must be negative",
        ritz2[0]
    );
    for w in ritz2.windows(2) {
        assert!(w[1] - w[0] > -1e-9, "positive gaps required: {ritz2:?}");
    }
    eprintln!(
        "qg_tegr_continuum_edge: 1-mode ground {:.8} ≥ −1/32; 2-mode ground {:.8} ≥ −1/16",
        ritz1[0], ritz2[0]
    );
}

#[test]
fn qg_tegr_flow_conserves_norm_energy() {
    // The 𝒮 pair terms populate both polarization ladders (each mode's
    // :𝒮²: creates/annihilates pairs from any state), so the polarization
    // content genuinely spreads — but the dynamics is unitary: the restarted
    // SIRK flow conserves the norm and the energy exactly.
    let h = qg_tegr_hamiltonian(2);
    let v0 = polarization_quantum(0, 1);
    let e0 = QuantumState::inner_product(&v0, &h.apply(&v0)).re;

    for &t in &[0.0_f64, 1.0, 5.0, 13.7] {
        let psi_t = evolve_restarted(&h, &v0, t, 40, 8, &best_device(), None, &opts()).unwrap();
        let norm = QuantumState::inner_product(&psi_t, &psi_t).re;
        let e_t = QuantumState::inner_product(&psi_t, &h.apply(&psi_t)).re;
        assert!((norm - 1.0).abs() < 1e-8, "t = {t}: norm = {norm}");
        assert!(
            (e_t - e0).abs() < 1e-7,
            "t = {t}: ⟨H⟩ = {e_t}, must stay {e0}"
        );
        eprintln!(
            "qg_tegr_flow: t = {t}, ‖ψ‖ = {norm:.12}, ⟨H⟩ = {e_t:.6} (initial {e0:.6})"
        );
    }
}
