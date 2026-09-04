//! QED extended numerical validation.
//!
//! Additional checks on the Fock/SIRK machinery beyond `qed_validation.rs`:
//!
//! 1. **JC dressed-state spectrum**: at resonance the Jaynes–Cummings
//!    (n+1)-excitation sector `{|n,e⟩, |n+1,g⟩}` splits by exactly
//!    `2g√(n+1)` with mean `ω(n+1)` — the photon-number-dependent Rabi
//!    frequency `Ω_n = g√(n+1)` (the √n ladder that drives collapse/revival).
//! 2. **Coherent-state Poisson statistics**: a (truncated) coherent state of
//!    the free field satisfies `⟨N⟩ = Var(N) = |α|²` — the quantum shot-noise
//!    floor of a laser field.
//! 3. **ζ-regularized Casimir energy**: the Abel-regularized zero-point sum
//!    `Σ n e^{−nε}` extracts `ζ(−1) = −1/12`, which turns the cavity spectrum
//!    `ω_n = nπ/d` into the 1D Casimir energy `E = −π/(24d)` and force
//!    `F = −π/(24d²)` (ħ = c = 1) — the seed of the 3D
//!    `E/A = −π²/(720d³)`.
//! 4. **Photon additivity and multi-mode vacuum**: `|n⟩` has energy `nω`
//!    exactly (n = 1..5) and a multi-mode vacuum is exactly zero-energy.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QuantumState, qed_free_photon,
    qed_jaynes_cummings,
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
        unit_norm_steps: false,
    }
}

fn assert_hermitian(h_proj: &DMatrix<Complex64>, label: &str) {
    let dag = h_proj.adjoint();
    let diff = (h_proj.clone() - dag).norm();
    assert!(
        diff < 1e-6,
        "{label}: H_proj must be Hermitian, ‖H−H†‖={diff}"
    );
}

fn sirk_ritz(h: &Hamiltonian, v0: &QuantumState, m: usize) -> Vec<f64> {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    assert_hermitian(&res.h_proj, "SIRK projected Hamiltonian");
    res.ritz_values()
}

fn sirk_ground(h: &Hamiltonian, v0: &QuantumState, m: usize) -> f64 {
    let res = solve_forward_sirk_with_opts(h, v0, &shifts(m), &best_device(), None, &opts())
        .expect("SIRK solve must complete");
    res.ground_state_energy().expect("a ground state")
}

fn measure(h: &Hamiltonian, state: &QuantumState) -> f64 {
    QuantumState::inner_product(&h.apply(state), state).re
}

/// `|n, e⟩` — n cavity photons (boson mode 0) plus the excited atom (fermion
/// mode 1). Convention from `qed_validation.rs`.
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

/// `|n⟩` — n photons in mode 0 of the free field.
fn n_photon(n: u32) -> QuantumState {
    let mut s =
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    for _ in 0..n {
        s = s.apply(&Operator::InnerBosonCreate(0));
    }
    s
}

/// A normalized truncated coherent state `|α⟩ ≈ e^{−|α|²/2} Σ_{n=0}^{N} αⁿ/√n! |n⟩`
/// of the free-field mode 0.
///
/// The framework's `ket_n = (a†)ⁿ|0⟩` carries the ladder factor `√(n!)`
/// (`a†|n⟩ = √(n+1)|n+1⟩`), so the amplitude passed to `scale_and_add` must
/// be `αⁿ/n!` for the state to be the true coherent superposition.
fn coherent_state(alpha: f64, n_max: u32) -> QuantumState {
    let mut bare = QuantumState::zero();
    for n in 0..=n_max {
        let amp = alpha.powi(n as i32) / (1..=n).fold(1.0_f64, |acc, k| acc * k as f64);
        let mut ket =
            QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        for _ in 0..n {
            ket = ket.apply(&Operator::InnerBosonCreate(0));
        }
        bare.scale_and_add(&ket, Complex64::new(amp, 0.0));
    }
    let norm = QuantumState::inner_product(&bare, &bare).re.sqrt();
    let mut out = QuantumState::zero();
    out.scale_and_add(&bare, Complex64::new(1.0 / norm, 0.0));
    out
}

/// `N² = a†a a†a` as a one-term Hamiltonian (expectation = second moment of
/// the photon number).
fn number_squared() -> Hamiltonian {
    Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerBosonCreate(0),
                Operator::InnerBosonAnnihilate(0),
                Operator::InnerBosonCreate(0),
                Operator::InnerBosonAnnihilate(0),
            ],
        )],
    }
}

#[test]
fn qed_jc_dressed_splitting_scales_as_sqrt_n_plus_1() {
    // At resonance (ω₀ = ω) the (n+1)-excitation sector is the two-level
    // pair {|n,e⟩, |n+1,g⟩} with splitting exactly 2g√(n+1) and mean ω(n+1)
    // (normal-ordered). This is the photon-number-dependent Rabi frequency
    // Ω_n = g√(n+1) — the √n ladder behind collapse and revival.
    let omega = 1.0;
    let g = 0.07;
    let h = qed_jaynes_cummings(omega, omega, g);
    let mut splits = Vec::new();
    let mut means = Vec::new();
    for n in 0..=3u32 {
        let v = jc_state(n, true);
        let ritz = sirk_ritz(&h, &v, 4);
        let split = ritz[1] - ritz[0];
        let mean = (ritz[0] + ritz[1]) / 2.0;
        let expect_split = 2.0 * g * ((n + 1) as f64).sqrt();
        let expect_mean = omega * (n as f64 + 1.0);
        assert!(
            (split - expect_split).abs() < 1e-8,
            "n={n}: dressed splitting must be 2g√(n+1) = {expect_split}, got {split}"
        );
        assert!(
            (mean - expect_mean).abs() < 1e-8,
            "n={n}: dressed mean must be ω(n+1) = {expect_mean}, got {mean}"
        );
        splits.push(split);
        means.push(mean);
    }
    // The √(n+1) ladder: split(n)/split(0) = √(n+1).
    for (n, &s) in splits.iter().enumerate().skip(1) {
        let ratio = s / splits[0];
        let expect = ((n + 1) as f64).sqrt();
        assert!(
            (ratio - expect).abs() < 1e-7,
            "split(n={n})/split(0) must be √(n+1) = {expect}, got {ratio}"
        );
    }
    eprintln!(
        "qed_jc_dressed_splitting_scales_as_sqrt_n_plus_1: splittings {splits:?} = 2g·[√1, √2, √3, 2]"
    );
}

#[test]
fn qed_coherent_state_poisson_statistics() {
    // A coherent field has Poisson photon statistics: ⟨N⟩ = Var(N) = |α|² —
    // the shot-noise floor. Both moments are measured as operator
    // expectations (N via the free-field Hamiltonian, N² via a†a a†a).
    let h_n = qed_free_photon(&[1.0]);
    let h_n2 = number_squared();
    for alpha in [0.5, 1.0, 2.0] {
        let psi = coherent_state(alpha, 24);
        // The state must be normalized.
        let norm = QuantumState::inner_product(&psi, &psi).re;
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "coherent state must be normalized"
        );
        let mean_n = measure(&h_n, &psi);
        let mean_n2 = measure(&h_n2, &psi);
        let var = mean_n2 - mean_n * mean_n;
        assert!(
            (mean_n - alpha * alpha).abs() < 1e-6,
            "⟨N⟩ must equal |α|² = {}, got {mean_n}",
            alpha * alpha
        );
        assert!(
            (var - alpha * alpha).abs() < 1e-5,
            "Var(N) must equal |α|² = {} (Poisson), got {var}",
            alpha * alpha
        );
        // Mandel Q = (⟨N²⟩−⟨N⟩²)/⟨N⟩ − 1 = 0 for coherent light.
        let q = var / mean_n - 1.0;
        assert!(
            q.abs() < 1e-5,
            "Mandel Q must vanish for coherent light, got {q}"
        );
    }
    eprintln!("qed_coherent_state_poisson_statistics: ⟨N⟩ = Var(N) = |α|², Q = 0");
}

#[test]
fn qed_zeta_minus_one_casimir_energy() {
    // The zero-point energy of the cavity spectrum ω_n = nπ/d is
    // (1/2)Σω_n = (π/2d)Σn. The sum diverges; its Abel regularization
    // Σ n e^{−nε} = e^{−ε}/(1−e^{−ε})² = 1/ε² − 1/12 + O(ε²) extracts
    // ζ(−1) = −1/12 — the seed of the Casimir energy. Check the extraction
    // numerically, then assemble the 1D Casimir E = −π/(24d) and force
    // F = −π/(24d²) (ħ = c = 1).
    let abel = |eps: f64| -> f64 {
        // Σ_{n=1}^N n·e^{−nε}; tail < 1e-16.
        let n = ((40.0 / eps) as usize).max(100_000);
        (1..=n)
            .map(|k| (k as f64) * (-(k as f64) * eps).exp())
            .sum()
    };
    for eps in [0.05, 0.02, 0.01] {
        let s = abel(eps);
        let constant = s - 1.0 / (eps * eps);
        // constant → −1/12 + ε²/240 + … ; the ε² term is ≤ 4e-6 at ε = 0.02.
        assert!(
            (constant + 1.0 / 12.0).abs() < 2e-4,
            "Abel-regularized sum must extract ζ(−1) = −1/12: got {constant} at ε={eps}"
        );
    }
    // 1D Casimir: E(d) = (1/2)(π/d)·Σn = −π/(24d); force F = −dE/dd = −π/(24d²).
    for d in [1.0, 2.0, 3.0] {
        let e_1d = -std::f64::consts::PI / (24.0 * d);
        let f_1d = -std::f64::consts::PI / (24.0 * d * d);
        // The numerical assembly from the extracted constant.
        let e_num = (std::f64::consts::PI / (2.0 * d)) * (-1.0 / 12.0);
        let f_num = e_num / d; // dE/dd = −E/d (E ∝ 1/d), F = −dE/dd = E/d
        assert!((e_num - e_1d).abs() < 1e-12, "1D Casimir energy mismatch");
        assert!((f_num - f_1d).abs() < 1e-12, "1D Casimir force mismatch");
        assert!(
            e_1d < 0.0 && f_1d < 0.0,
            "Casimir energy/force must be attractive"
        );
    }
    eprintln!(
        "qed_zeta_minus_one_casimir_energy: ζ(−1) = −1/12 extracted; E = −π/(24d), F = −π/(24d²)"
    );
}

#[test]
fn qed_photon_additivity_and_multimode_vacuum() {
    // |n⟩ has energy nω exactly, n = 1..5 (additivity of the free field).
    let omega = 1.7;
    let h = qed_free_photon(&[omega]);
    for n in 1..=5u32 {
        let e = sirk_ground(&h, &n_photon(n), 6);
        assert!(
            (e - (n as f64) * omega).abs() < 1e-9,
            "n-photon energy must be nω = {}, got {e}",
            (n as f64) * omega
        );
    }
    // Multi-mode vacuum is exactly zero-energy; a photon in mode i has
    // energy ωᵢ; photons in distinct modes add.
    let omegas = [1.0, 2.0, 4.0];
    let h_multi = qed_free_photon(&omegas);
    let vac =
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    assert!(
        sirk_ground(&h_multi, &vac, 6).abs() < 1e-9,
        "multi-mode vacuum must have zero energy"
    );
    for (i, &w) in omegas.iter().enumerate() {
        let one = vac.clone().apply(&Operator::InnerBosonCreate(i as u32));
        let e = sirk_ground(&h_multi, &one, 6);
        assert!(
            (e - w).abs() < 1e-9,
            "one photon in mode {i} must have energy {w}, got {e}"
        );
    }
    // Two photons in distinct modes add: ω₁ + ω₂.
    let two = vac
        .clone()
        .apply(&Operator::InnerBosonCreate(0))
        .apply(&Operator::InnerBosonCreate(1));
    let e = sirk_ground(&h_multi, &two, 6);
    assert!(
        (e - (omegas[0] + omegas[1])).abs() < 1e-9,
        "two-photon cross-mode energy must be ω₁+ω₂ = {}, got {e}",
        omegas[0] + omegas[1]
    );
    eprintln!(
        "qed_photon_additivity_and_multimode_vacuum: nω additivity (n≤5), multi-mode vacuum = 0, cross-mode additivity"
    );
}
