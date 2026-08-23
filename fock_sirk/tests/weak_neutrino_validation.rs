//! Weak-interaction and neutrino-oscillation numerical validation.
//!
//! 1. `weak_muon_lifetime_from_gf` — the leading-order weak result
//!    τ = 192π³ħ/(G_F²m_μ⁵c⁴) = 2.199 µs vs the measured 2.1969811 µs
//!    (radiative corrections shorten it by ~0.2% — the direction is checked).
//! 2. `neutrino_atmospheric_first_maximum` — the Super-K disappearance band:
//!    first oscillation maximum at L/E ≈ 495 km/GeV for |Δm²₂₃| = 2.5e-3 eV².
//! 3. `neutrino_reactor_and_theta13` — KamLAND's baseline sits beyond the
//!    first maximum (deep-suppression lobe, strong P_ee suppression), and at
//!    its own first maximum the θ₁₃ survival is exactly 1 − sin²(2θ₁₃) ≈ 0.915
//!    (the Daya Bay value).

use nested_fock_algebra::{neutrino_first_max_km_per_gev, neutrino_survival_two_flavor, weak_muon_lifetime_lo_s};

fn rel(v: f64, t: f64) -> f64 {
    (v - t).abs() / t
}

#[test]
fn weak_muon_lifetime_from_gf() {
    let tau_lo = weak_muon_lifetime_lo_s();
    let tau_meas = 2.196_9811e-6; // PDG muon lifetime
    // Leading order within 0.5%, and SHORTER than measured (the G_F
    // convention puts loop corrections on top of this tree-level rate).
    assert!(rel(tau_lo, tau_meas) < 5e-3, "τ_LO = {tau_lo:.4e} s");
    assert!(tau_lo < tau_meas, "LO must underestimate the lifetime");
}

#[test]
fn neutrino_atmospheric_first_maximum() {
    let dm23 = 2.5e-3; // |Δm²₃₂|, eV² — Super-K/NOvA/T2K
    let le_max = neutrino_first_max_km_per_gev(dm23);
    assert!(rel(le_max, 494.8) < 1e-2, "L/E max = {le_max} km/GeV");
    // At that L/E the disappearance is maximal: P_μμ → 1 − sin²(2θ₂₃).
    let sin_sq_23 = 0.55; // near-maximal mixing
    let p = neutrino_survival_two_flavor(dm23, sin_sq_23, le_max * 1.0, 1.0);
    assert!(rel(p, 1.0 - 4.0 * 0.55 * 0.45) < 1e-9);
}

#[test]
fn neutrino_reactor_and_theta13() {
    let dm21 = 7.42e-5; // Δm²₂₁, eV² — solar/KamLAND
    // KamLAND: L = 180 km, reactor Ē ≈ 3.4 MeV ⇒ phase well past π
    // (beyond the first maximum, in the deep-suppression lobe).
    let phase_beyond = 1.267 * dm21 * 180.0 / 0.0034;
    assert!(phase_beyond > std::f64::consts::PI, "KamLAND must sit past 1st max");
    let sin_sq_12 = 0.304;
    let pee_kamland = neutrino_survival_two_flavor(dm21, sin_sq_12, 180.0, 0.0034);
    assert!(
        (0.10..0.40).contains(&pee_kamland),
        "KamLAND-band P_ee = {pee_kamland:.3} must be strongly suppressed"
    );

    // Daya Bay / θ₁₃: at ITS first maximum the electron survival is exactly
    // 1 − sin²(2θ₁₃); with the measured sin²(2θ₁₃) ≈ 0.085 this is ≈0.915.
    let sin_sq_13 = 0.022; // sin²θ₁₃ ≈ 0.0218 (Daya Bay)
    let sin_sq_2theta13 = 4.0 * sin_sq_13 * (1.0 - sin_sq_13);
    let le13 = neutrino_first_max_km_per_gev(2.42e-3); // |Δm²_ee|
    let p13 = neutrino_survival_two_flavor(2.42e-3, sin_sq_13, le13 * 1.0, 1.0);
    assert!(rel(p13, 1.0 - sin_sq_2theta13) < 1e-9);
    assert!(rel(p13, 0.9147) < 5e-3, "P_ee(θ₁₃ max) = {p13}");
}
