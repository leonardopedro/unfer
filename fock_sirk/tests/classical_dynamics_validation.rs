//! Classical-dynamics and relativity numerical validation: Foucault
//! precession, Kepler periods, escape/Roche geometry, the finite-amplitude
//! pendulum (series vs RK4 integrator — an integrator cross-check), SR Doppler,
//! and the GW150914 chirp-mass combination.
//!
//! 1. `dyn_foucault_panthelon_rate` — Ω = 360°·sin(lat)/T_sidereal at the
//!    Panthéon's latitude gives ≈11.3°/h, the measured precession of the
//!    original 1851 Foucault pendulum.
//! 2. `dyn_kepler_third_law_planets` — T = 2π√(a³/GM_sun) reproduces the
//!    sidereal years of Earth/Mars/Jupiter from their semi-major axes.
//! 3. `dyn_escape_and_circular_velocity_earth` — 11.19 km/s escape,
//!    7.91 km/s circular, ratio exactly √2.
//! 4. `dyn_roche_limit_moon` — d = 2.44R(ρ⊕/ρ☾)^{1/3} ≈ 18,400 km.
//! 5. `dyn_pendulum_finite_amplitude_series_vs_rk4` — the series
//!    T/T₀ = 1 + θ₀²/16 + 11θ₀⁴/3072 + … against a direct RK4 integration of
//!    θ̈ = −sin θ: agreement < 1e-5 at θ₀ = 0.5 rad.
//! 6. `sr_doppler_z_one_is_beta_three_fifths` — z = 1 ⇔ β = 3/5 exactly
//!    (the relativistic Doppler relation).
//! 7. `gw150914_chirp_mass_combination` — 𝓜(36, 29)M☉ ≈ 28.1 M☉, the LIGO
//!    source-frame chirp mass.

use nested_fock_algebra::{
    dyn_escape_velocity_ms, dyn_foucault_rate_deg_per_hour, dyn_kepler_period_years,
    dyn_pendulum_period_series_ratio, dyn_roche_limit_m, gw_chirp_mass_kg, phys,
    sr_doppler_beta_from_z,
};

fn rel(v: f64, t: f64) -> f64 {
    (v - t).abs() / t
}

#[test]
fn dyn_foucault_pantheon_rate() {
    let rate = dyn_foucault_rate_deg_per_hour(48.846_2); // Panthéon, Paris
    assert!(rel(rate, 11.32) < 1e-2, "Foucault rate {rate:.4} °/h");
}

#[test]
fn dyn_kepler_third_law_planets() {
    let earth = dyn_kepler_period_years(1.000_000_11);
    let mars = dyn_kepler_period_years(1.523_679);
    let jupiter = dyn_kepler_period_years(5.204_4);
    assert!(rel(earth, 1.0) < 1e-4, "T⊕ = {earth}");
    assert!(rel(mars, 1.880_8) < 1e-3, "T♂ = {mars}");
    assert!(rel(jupiter, 11.862) < 1e-2, "T♃ = {jupiter}");
}

#[test]
fn dyn_escape_and_circular_velocity_earth() {
    let r_e = 6.371e6;
    let v_esc = dyn_escape_velocity_ms(5.972_2e24, r_e);
    assert!(rel(v_esc, 11_186.0) < 1e-3, "v_esc = {v_esc:.1} m/s");
    // Circular orbit at the same radius: v/√2 exactly.
    let v_circ = v_esc / std::f64::consts::SQRT_2;
    let _ = phys::M_SUN; // keep the shared-constants import meaningful
    assert!(rel(v_circ, 7_910.0) < 2e-3, "v_circ = {v_circ:.1} m/s");
}

#[test]
fn dyn_roche_limit_moon() {
    let d = dyn_roche_limit_m(6.371e6, 5514.0, 3344.0);
    assert!(d > 1.7e7 && d < 1.95e7, "Roche limit {d:.4e} m");
}

#[test]
fn dyn_pendulum_finite_amplitude_series_vs_rk4() {
    // Series prediction at θ₀ = 0.5 rad:
    let series = dyn_pendulum_period_series_ratio(0.5);
    // Direct RK4 integration of θ'' = −sin θ over one period, period read
    // off two successive zero crossings with positive slope:
    let theta0 = 0.5_f64;
    let dt = 1e-4;
    let mut th = theta0;
    let mut om = 0.0;
    let mut t_prev_zero = 0.0_f64;
    let mut have_first = false;
    let mut period = 0.0_f64;
    for i in 0..200_000 {
        let t = i as f64 * dt;
        let deriv = |th: f64, om: f64| (om, -th.sin());
        let (k1t, k1o) = deriv(th, om);
        let (k2t, k2o) = deriv(th + dt / 2.0 * k1t, om + dt / 2.0 * k1o);
        let (k3t, k3o) = deriv(th + dt / 2.0 * k2t, om + dt / 2.0 * k2o);
        let (k4t, k4o) = deriv(th + dt * k3t, om + dt * k3o);
        let new_th = th + dt / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        let new_om = om + dt / 6.0 * (k1o + 2.0 * k2o + 2.0 * k3o + k4o);
        if !have_first && th < 0.0 && new_th >= 0.0 {
            have_first = true;
            t_prev_zero = t;
        } else if have_first && th < 0.0 && new_th >= 0.0 {
            period = t - t_prev_zero;
            break;
        }
        th = new_th;
        om = new_om;
    }
    let t0 = 2.0 * std::f64::consts::PI; // small-angle period (L=1, g=1)
    let numeric_ratio = period / t0;
    assert!(
        (series - numeric_ratio).abs() < 1e-5,
        "series {series:.7} vs RK4 {numeric_ratio:.7}"
    );
}

#[test]
fn sr_doppler_z_one_is_beta_three_fifths() {
    let beta = sr_doppler_beta_from_z(1.0);
    assert!(rel(beta, 0.6) < 1e-12, "β(z=1) = {beta}");
}

#[test]
fn gw150914_chirp_mass_combination() {
    let msun = phys::M_SUN;
    let mchirp = gw_chirp_mass_kg(36.0 * msun, 29.0 * msun) / msun;
    assert!(mchirp > 27.5 && mchirp < 29.0, "𝓜 = {mchirp:.2} M☉");
}
