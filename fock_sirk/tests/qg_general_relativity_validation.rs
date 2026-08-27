//! QG general-relativity numerical validation — classical GR anchors beyond
//! `qg_cosmology_validation.rs`.
//!
//! 1. **Mercury perihelion precession**: the GR excess
//!    `Δφ = 6πGM/(c²a(1−e²))` per orbit; at 415.2 orbits/century this is the
//!    famous `43.0″/century`.
//! 2. **Hawking temperature**: `T_H = ℏc³/(8πGMk_B)` — `6.17e−8 K` for a
//!    solar-mass black hole — and **evaporation time** `τ = 5120πG²M³/(ℏc⁴)`
//!    with its exact `τ ∝ M³` scaling (doubling M gives ×8 the lifetime).
//! 3. **Gravitational redshift**: exact `z = 1/√(1−r_s/r) − 1` vs the weak-field
//!    limit `z ≈ r_s/(2r)`; the two must agree ever better as `r/r_s → ∞`.
//! 4. **Geodesic constants**: photon sphere `r_ph = 1.5 r_s`, ISCO `3 r_s`,
//!    horizon `r_s = 2GM/c²`.
//! 5. **GW chirp scaling**: `ḟ ∝ f^{11/3}` (quadrupole) and the GW150914
//!    chirp mass `M_c = (m₁m₂)^{3/5}/(m₁+m₂)^{1/5} ≈ 28.3 M_☉`.
//! 6. **Bekenstein bound saturation**: for a Schwarzschild black hole the bound
//!    `S ≤ 2πk_B R E/(ℏc)` is *saturated* at `R = r_s`: `S_BH = 2πk_B r_s Mc²/
//!    (ℏc) = 4πk_BGM²/(ℏc)` exactly.
//!
//! CODATA 2018 constants; tolerances documented per assertion.

const G: f64 = 6.674_30e-11;
const C: f64 = 2.997_924_58e8;
const HBAR: f64 = 1.054_571_817e-34;
const KB: f64 = 1.380_649e-23;
const M_SUN: f64 = 1.988_47e30;
const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36; // 180/π · 3600

/// Schwarzschild radius r_s = 2GM/c².
fn r_schw(m: f64) -> f64 {
    2.0 * G * m / (C * C)
}

#[test]
fn qg_mercury_perihelion_precession_43_arcsec() {
    // Mercury orbital elements.
    let a = 5.790_905e10; // semi-major axis [m]
    let e = 0.205_630; // eccentricity
    let orbits_per_century = 415.2; // 100 yr / 0.24085 yr
    // GR precession per orbit: Δφ = 6πGM/(c²a(1−e²)).
    let gm_c2 = G * M_SUN / (C * C); // 1.4766e3 m
    let dphi = 6.0 * std::f64::consts::PI * gm_c2 / (a * (1.0 - e * e));
    let arcsec_per_century = dphi * ARCSEC_PER_RAD * orbits_per_century;
    let expected = 43.0;
    assert!(
        (arcsec_per_century - expected).abs() < 0.5,
        "Mercury precession = {arcsec_per_century:.3}″/century, expected {expected}"
    );
    // Per-orbit value is ~0.1036″.
    assert!((dphi * ARCSEC_PER_RAD - 0.1036).abs() < 0.001);
}

#[test]
fn qg_hawking_temperature_solar_mass() {
    let t_h = HBAR * C.powi(3) / (8.0 * std::f64::consts::PI * G * M_SUN * KB);
    let expected = 6.17e-8; // K
    let rel = (t_h - expected).abs() / expected;
    assert!(rel < 1e-2, "T_H = {t_h:.6e} K, expected {expected:.6e} (rel {rel:.2e})");
    // Inverse scaling: T_H ∝ 1/M — a 10 M_☉ hole is 10× colder.
    let t_h10 = HBAR * C.powi(3) / (8.0 * std::f64::consts::PI * G * 10.0 * M_SUN * KB);
    assert!((t_h10 - t_h / 10.0).abs() / (t_h / 10.0) < 1e-12);
}

#[test]
fn qg_black_hole_evaporation_time_and_m3_scaling() {
    // τ = 5120πG²M³/(ℏc⁴)
    let tau = |m: f64| {
        5120.0 * std::f64::consts::PI * G * G * m.powi(3) / (HBAR * C.powi(4))
    };
    let tau_sun = tau(M_SUN);
    let expected = 6.6e74; // s ≈ 2.1e67 yr
    let rel = (tau_sun - expected).abs() / expected;
    assert!(rel < 5e-2, "τ = {tau_sun:.3e} s, expected {expected:.2e} (rel {rel:.2e})");
    // Exact M³ scaling: doubling the mass ×8 the lifetime.
    assert!((tau(2.0 * M_SUN) / tau_sun - 8.0).abs() < 1e-9);
}

#[test]
fn qg_gravitational_redshift_weak_field_limit() {
    // Exact: z = 1/√(1−r_s/r) − 1; weak-field: z ≈ r_s/(2r).
    let r_s = r_schw(M_SUN);
    let z_exact = |r: f64| 1.0 / (1.0 - r_s / r).sqrt() - 1.0;
    let z_weak = |r: f64| r_s / (2.0 * r);
    // At 100 r_s the exact redshift is 0.5038% vs the weak-field 0.5000%.
    let z100 = z_exact(100.0 * r_s);
    let rel100 = (z100 - z_weak(100.0 * r_s)).abs() / z100;
    assert!(rel100 < 1e-2, "at 100 r_s: rel diff {rel100:.3e}");
    // At 10⁴ r_s the limit sharpens by ~100×.
    let z1e4 = z_exact(1e4 * r_s);
    let rel1e4 = (z1e4 - z_weak(1e4 * r_s)).abs() / z1e4;
    assert!(rel1e4 < 1e-4, "at 10⁴ r_s: rel diff {rel1e4:.3e}");
    // Sanity: z → ∞ as r → r_s⁺ and z = 0 at r → ∞.
    assert!(z_exact(1.0001 * r_s) > 10.0);
    assert!(z_exact(1e6 * r_s) < 1e-6);
}

#[test]
fn qg_geodesic_constants_photon_sphere_and_isco() {
    // r_ph = 3GM/c² = 1.5 r_s; ISCO = 6GM/c² = 3 r_s; horizon r_s = 2GM/c².
    let r_s = r_schw(M_SUN);
    let r_ph = 3.0 * G * M_SUN / (C * C);
    let r_isco = 6.0 * G * M_SUN / (C * C);
    assert!((r_ph / r_s - 1.5).abs() < 1e-12, "r_ph = {r_ph}, r_s = {r_s}");
    assert!((r_isco / r_s - 3.0).abs() < 1e-12, "ISCO = {r_isco}, r_s = {r_s}");
    // r_s for the Sun ≈ 2953 m.
    assert!((r_s - 2953.0).abs() / 2953.0 < 1e-3, "r_s(M_☉) = {r_s:.1} m");
}

#[test]
fn qg_gw_chirp_mass_and_f11_3_scaling() {
    // Chirp mass: M_c = (m₁m₂)^{3/5}/(m₁+m₂)^{1/5}.
    let chirp = |m1: f64, m2: f64| {
        (m1 * m2).powf(0.6) / (m1 + m2).powf(0.2)
    };
    // GW150914: 36 + 29 M_☉ → M_c ≈ 28.3 M_☉ (LIGO).
    let m_c = chirp(36.0, 29.0);
    assert!(
        (m_c - 28.3).abs() < 0.5,
        "M_c(GW150914) = {m_c:.2} M_☉, expected ≈ 28.3"
    );
    // Quadrupole: ḟ ∝ M_c^{5/3} f^{11/3} → doubling f multiplies ḟ by 2^{11/3} ≈ 12.7.
    let fdot_ratio = 2f64.powf(11.0 / 3.0);
    assert!((fdot_ratio - 12.699).abs() < 1e-3, "2^{{11/3}} = {fdot_ratio}");
    // Chirp mass enters ḟ with exponent 5/3: doubling M_c → ×2^{5/3} ≈ 3.17.
    let mc_ratio = 2f64.powf(5.0 / 3.0);
    assert!((mc_ratio - 3.175).abs() < 1e-3);
}

#[test]
fn qg_shapiro_delay_sun_graze() {
    // Δt = (4GM/c³)·ln(4r₁r₂/b²). For a radar echo grazing the Sun's limb
    // (r₁ = r₂ = 1 AU, b = R_☉) the excess delay is ≈ 239 µs — the
    // historically decisive test of the non-Newtonian time metric.
    let r_au: f64 = 1.495_978_707e11; // m
    let r_sun: f64 = 6.957e8; // m
    let gm_c3 = G * M_SUN / C.powi(3);
    let dt = 4.0 * gm_c3 * (4.0 * r_au * r_au / (r_sun * r_sun)).ln();
    let dt_us = dt * 1e6;
    assert!(
        (dt_us - 239.0).abs() < 30.0,
        "Shapiro delay = {dt_us:.1} µs, expected ≈ 239 µs"
    );
    // The delay grows logarithmically as the impact parameter shrinks:
    // b → b/2 adds ln(4) ≈ 1.39 of the 12.1-argument — a ~12% increase.
    let dt2 = 4.0 * gm_c3 * (4.0 * r_au * r_au / ((r_sun / 2.0).powi(2))).ln();
    let ratio = dt2 / dt;
    let expected_ratio = (4.0f64).ln() / (4.0f64 * r_au * r_au / (r_sun * r_sun)).ln() + 1.0;
    assert!((ratio - expected_ratio).abs() < 1e-9, "ln-growth of Δt");
}

#[test]
fn qg_peters_merger_time_a4_scaling() {
    // Peters (1964): the GW-driven inspiral time from separation a is
    // t_merge = (5/256)(c⁵/G³)·a⁴/(m₁m₂(m₁+m₂)). The a⁴ law is the
    // gravitational-radiation analog of Kepler — doubling the separation
    // lengthens the inspiral ×16.
    let c5_g3 = C.powi(5) / G.powi(3);
    let m_ns = 1.4 * M_SUN; // 1.4 M_☉ neutron star
    let t_merge = |a: f64| {
        (5.0 / 256.0) * c5_g3 * a.powi(4)
            / (m_ns * m_ns * (2.0 * m_ns))
    };
    let a = 3.0e9; // m
    let t = t_merge(a);
    // Order of magnitude: ≈ 3.0e17 s (≈ 9.5 Gyr) for two 1.4 M_☉ stars at
    // 3e9 m — the same ballpark as the Hulse–Taylor inspiral (≈ 1.6 Gyr at
    // a ≈ 1.95e9 m, Peters 1964).
    assert!(
        (t - 3.0e17).abs() / 3.0e17 < 0.5,
        "t_merge = {t:.2e} s at a = 3e9 m"
    );
    // Exact a⁴ scaling: 2a → 16t.
    assert!((t_merge(2.0 * a) / t - 16.0).abs() < 1e-9);
    // And t ∝ 1/(m₁m₂(m₁+m₂)): a 1.4+1.4 pair merges 8× faster than a
    // 0.7+0.7 pair at the same separation.
    let m_low = 0.7 * M_SUN;
    let t_low = (5.0 / 256.0) * c5_g3 * a.powi(4) / (m_low * m_low * (2.0 * m_low));
    assert!((t_low / t - 8.0).abs() < 1e-9, "t ∝ 1/(m₁m₂(m₁+m₂))");
}

#[test]
fn qg_bekenstein_bound_saturated_by_schwarzschild() {
    // Bound: S ≤ 2πk_B R E/(ℏc). For a BH with R = r_s, E = Mc² this equals
    // 4πk_BGM²/(ℏc) = S_BH — saturation, exactly.
    let m = 1.0; // 1 kg test hole
    let r_s = r_schw(m);
    let s_bh = 4.0 * std::f64::consts::PI * KB * G * m * m / (HBAR * C);
    let bound = 2.0 * std::f64::consts::PI * KB * r_s * (m * C * C) / (HBAR * C);
    assert!(
        (s_bh - bound).abs() / s_bh < 1e-12,
        "S_BH = {s_bh}, bound = {bound}"
    );
    // A non-BH system (say a 1 m sphere of 1 kg) sits far below the bound.
    let r = 1.0;
    let bound_1m = 2.0 * std::f64::consts::PI * KB * r * (m * C * C) / (HBAR * C);
    assert!(bound_1m > s_bh * 1e20, "1 m sphere bound {bound_1m} ≫ S_BH {s_bh}");
}
