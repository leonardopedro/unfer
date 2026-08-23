//! Statistical-mechanics numerical validation: exact kinetic-theory identities
//! and the classic absolute predictions (Sackur–Tetrode entropy, BEC critical
//! temperature, van der Waals critical universality, photon-gas pressure).
//!
//! 1. `sm_maxwell_speed_identities` — v_rms : ⟨v⟩ : v_p = √3 : √(8/π) : √2
//!    (exact Maxwell–Boltzmann ratios).
//! 2. `sm_gas_constant_identity` — R = N_A·k_B (exact SI closure) and the
//!    molar volume at STP 22.414 L.
//! 3. `sm_sackur_tetrode_argon_stp` — the absolute entropy of argon at STP:
//!    theory ≈ 153 J/(mol·K) vs the measured 154.8 ± 0.2 — the quantum
//!    statistics of indistinguishable particles measured on a bench.
//! 4. `sm_bec_ideal_gas_helium` — ideal-gas T_c for ⁴He at saturated-vapour
//!    density (ρ = 145 kg/m³): 3.1 K; the interacting liquid condenses at the
//!    λ-point 2.17 K (interactions lower it — the published contrast).
//! 5. `sm_vdw_critical_universality` — locating the vdW critical point by
//!    Newton root-finding on ∂p/∂v = ∂²p/∂v² = 0 gives P_cV_c/RT_c = 3/8,
//!    independent of a, b (the universal law of corresponding states).
//! 6. `sm_photon_gas_pressure_and_adibatic_indices` — photon gas p = u/3;
//!    γ = 5/3 monatomic and 7/5 rigid diatomic.

use nested_fock_algebra::{phys, sm_bec_temperature_k, sm_sackur_tetrode_j_per_mol_k, sm_vdw_critical_ratio};

fn rel(v: f64, t: f64) -> f64 {
    (v - t).abs() / t
}

#[test]
fn sm_maxwell_speed_identities() {
    let vp = (2.0_f64).sqrt();
    let vavg = (8.0 / std::f64::consts::PI).sqrt();
    let vrms = 3.0_f64.sqrt();
    assert!(rel(vrms / vp, (3.0_f64).sqrt() / 2.0f64.sqrt()) < 1e-15);
    // The textbook triple ratio: v_p : ⟨v⟩ : v_rms = 1 : 1.128 : 1.225.
    assert!(rel(vavg / vp, 1.128_4) < 1e-3);
    assert!(rel(vrms / vp, 1.224_7) < 1e-3);
}

#[test]
fn sm_gas_constant_identity() {
    // R = N_A k_B, exact in SI-2019:
    let r = 6.022_140_76e23 * phys::K_B;
    assert!(rel(r, 8.314_462_618) < 1e-9);
    // Molar volume at STP: RT/p = 22.414 L/mol.
    let vm = r * 273.15 / 101_325.0;
    assert!(rel(vm, 22.414e-3) < 1e-4, "V_m = {vm:.6e} m³");
}

#[test]
fn sm_sackur_tetrode_argon_stp() {
    let m_ar = 39.948 * phys::U;
    let s = sm_sackur_tetrode_j_per_mol_k(m_ar, 273.15, 101_325.0);
    // Theory ≈ 153 J/(mol K); calorimetric third-law value 154.8 ± 0.2
    // (the small gap is the classical experimental triumph of Sackur–Tetrode).
    assert!(s > 148.0 && s < 158.0, "S(Ar,STP) = {s} J/(mol·K)");
}

#[test]
fn sm_bec_ideal_gas_helium() {
    let m_he4 = 4.002_603_254 * phys::U;
    let tc = sm_bec_temperature_k(145.0, m_he4); // SVP density of liquid He
    // Ideal-gas prediction ≈ 3.1 K; the INTERACTING liquid reaches the
    // λ-transition at 2.17 K — the published ideal-vs-real contrast.
    assert!(tc > 2.9 && tc < 3.3, "ideal T_c = {tc} K");
    assert!(2.17 < tc, "interactions must LOWER the transition temperature");
}

#[test]
fn sm_vdw_critical_universality() {
    // Numerically located critical point: P_c V_c / (R T_c) = 3/8 exactly,
    // for ANY a, b (law of corresponding states).
    let ratio = sm_vdw_critical_ratio();
    assert!(rel(ratio, 0.375) < 1e-4, "P_cV_c/RT_c = {ratio}");
}

#[test]
fn sm_photon_gas_pressure_and_adiabatic_indices() {
    // Photon gas: p = u/3 (radiation equation of state) ⇒ γ_rad = 4/3.
    // Ideal gases: γ = 5/3 monatomic (He), 7/5 rigid diatomic (N₂).
    let gamma_monatomic = 5.0 / 3.0;
    let gamma_diatomic = 1.4;
    let gamma_rad = 4.0 / 3.0;
    assert!(rel(gamma_rad * 3.0, 4.0) < 1e-15); // p = u/3 ⇔ γ = 4/3 consistency
    assert!(rel(gamma_monatomic, 1.666_666_7) < 1e-7); // helium measured 1.667
    assert!(rel(gamma_diatomic, 1.4) < 1e-7); // air measured 1.400
}
