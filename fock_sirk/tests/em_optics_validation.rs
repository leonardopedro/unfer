//! Electromagnetism / optics / wave-engineering numerical validation: textbook
//! and engineering-exact anchors, from cyclotron frequencies to the Larmor
//! collapse of the classical atom (the paradox that forced quantum mechanics).
//!
//! 1. `em_cyclotron_frequencies` — f = qB/2πm at 1 T: proton 15.245 MHz,
//!    electron 27.992 GHz (the ESR/NMR calibration anchors).
//! 2. `em_waveguide_brewster_critical_skin` — WR-90 TE₁₀ cutoff 6.557 GHz;
//!    Brewster angle 56.31° and critical angle 41.81° for n = 1.5 glass; Cu
//!    skin depth 9.3 mm at 50 Hz.
//! 3. `em_dipole_radiation_resistance` — short-dipole 80π²(l/λ)² and the
//!    thin half-wave dipole's exact 73.129 Ω.
//! 4. `em_rayleigh_blue_sky_ratio` — the λ⁻⁴ law: (650/450)⁴ ≈ 4.35 scatter
//!    dominance of blue over red (why the sky is blue).
//! 5. `em_larmor_collapse_of_classical_hydrogen` — integrating
//!    dr/dt = −e⁴k/(3πε₀c³m²r²) from r = a₀ gives τ ≈ 1.6×10⁻¹¹ s: the
//!    classical Rutherford atom spirals into the nucleus in tens of
//!    picoseconds (the published textbook value).

use nested_fock_algebra::{
    em_brewster_angle_deg, em_critical_angle_deg, em_cyclotron_frequency_hz,
    em_larmor_collapse_time_s, em_short_dipole_r_rad_ohm, em_skin_depth_m,
    em_waveguide_cutoff_hz, phys, EM_HALF_WAVE_DIPOLE_R_RAD_OHM,
};

fn rel(v: f64, t: f64) -> f64 {
    (v - t).abs() / t
}

#[test]
fn em_cyclotron_frequencies() {
    let qm_e = phys::E / phys::M_E;
    let qm_p = phys::E / phys::M_P;
    let fe = em_cyclotron_frequency_hz(qm_e, 1.0);
    let fp = em_cyclotron_frequency_hz(qm_p, 1.0);
    assert!(rel(fe, 27.992_490_110e9) < 1e-5, "f_ce = {fe:.6e} Hz");
    assert!(rel(fp, 15.245_166_014e6) < 1e-5, "f_cp = {fp:.6e} Hz");
}

#[test]
fn em_waveguide_brewster_critical_skin() {
    // WR-90 X-band guide, a = 22.86 mm: TE10 cutoff 6.557 GHz.
    let fc = em_waveguide_cutoff_hz(22.86e-3);
    assert!(rel(fc, 6.557_2e9) < 2e-4, "f_c = {fc:.6e} Hz");
    // Glass n=1.5: Brewster 56.31°, critical angle 41.81°.
    assert!(rel(em_brewster_angle_deg(1.0, 1.5), 56.309_9) < 2e-5);
    assert!(rel(em_critical_angle_deg(1.5, 1.0), 41.809_9) < 2e-5);
    // Copper at mains frequency: δ ≈ 9.3 mm (σ_Cu = 5.96e7 S/m).
    let d = em_skin_depth_m(50.0, 5.96e7);
    assert!(rel(d, 9.35e-3) < 2e-2, "skin depth {d:.4e} m");
}

#[test]
fn em_dipole_radiation_resistance() {
    // Short dipole at l = λ/10: 7.90 Ω; half-wave dipole: 73.129 Ω.
    let r_short = em_short_dipole_r_rad_ohm(0.1);
    assert!(rel(r_short, 7.895_7) < 1e-4, "R_rad(short) = {r_short} Ω");
    assert!(rel(EM_HALF_WAVE_DIPOLE_R_RAD_OHM, 73.129) < 1e-6);
    // Scaling is quadratic in l/λ.
    assert!(rel(
        em_short_dipole_r_rad_ohm(0.2) / em_short_dipole_r_rad_ohm(0.1),
        4.0
    ) < 1e-12);
}

#[test]
fn em_rayleigh_blue_sky_ratio() {
    // σ ∝ λ⁻⁴: red 650 nm scatters (650/450)⁴ ≈ 4.35 weaker than blue 450 nm.
    let ratio = (650.0_f64 / 450.0).powi(4);
    assert!(rel(ratio, 4.352) < 2e-3, "Rayleigh ratio {ratio}");
}

#[test]
fn em_larmor_collapse_of_classical_hydrogen() {
    let a0 = 5.291_772_109_03e-11; // Bohr radius
    let tau = em_larmor_collapse_time_s(a0);
    // Textbook value ~1.6e-11 s (order-of-picoseconds collapse).
    assert!(tau > 1.2e-11 && tau < 2.0e-11, "τ_Larmor = {tau:.4e} s");
}
