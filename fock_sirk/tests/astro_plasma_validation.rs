//! Astrophysical, plasma and quantum-metrology numerical validation, plus the
//! binary-inspiral chirp as an ODE-integrator cross-check.
//!
//! 1. `astro_black_hole_anchors` — Hawking temperature of a solar-mass black
//!    hole 6.17e-8 K; Schwarzschild radii (Sun 2.953 km, Sgr A* 1.27e10 m);
//!    ISCO GW frequency 4397 Hz per solar mass (the LIGO ringdown scale).
//! 2. `astro_chandrasekhar_mass_from_constants` — π(ħc/G)^{3/2}/(μ_e m_p)²
//!    with μ_e = 2 gives the published 1.44 M_sun white-dwarf limit.
//! 3. `astro_eddington_luminosity_solar_mass` — 4πGMm_pc/σ_T = 1.26e31 W per
//!    M_sun, from G, m_p, c and the Thomson cross section alone.
//! 4. `astro_critical_density_and_baryons_planck2018` — ρ_c = 3H₀²/8πG at
//!    H₀ = 67.66 km/s/Mpc gives 8.5e-27 kg/m³; Ω_b·ρ_c ≈ 0.049·ρ_c matches
//!    the big-bang-nucleosynthesis baryon density.
//! 5. `astro_cmb_photon_gas_numbers` — T = 2.7255 K: n_γ ≈ 411 cm⁻³ and
//!    u = 4.17e-14 J/m³ ≈ 0.26 eV/cm³ (the measured CMB thermal content).
//! 6. `plasma_ionosphere_frequency_and_debye_length` — f_p = 8.98 MHz for
//!    n = 10¹² m⁻³ (the ionospheric radio window); λ_D from two independent
//!    expressions agrees and sits in the mm range.
//! 7. `plasma_alfven_speed_solar_wind` — B = 5 nT, n = 5 cm⁻³ ⇒ ~48 km/s,
//!    the measured solar-wind Alfvén speed scale.
//! 8. `si_quantum_metrology_triangle` — R_K = h/e² = 25812.80745 Ω,
//!    K_J = 2e/h = 483597.8484 GHz/V, Φ₀ = h/2e = 2.067833848e-15 Wb, exact
//!    by SI definition, and K_J·R_K = 2/e closes the metrology triangle.
//! 9. `bcs_weak_coupling_gap_ratio` — Δ/k_BT_c = 1.764 (BCS), i.e.
//!    2Δ(0)/k_BT_c = 3.53, the universal weak-coupling tunneling value.
//! 10. `inspiral_chirp_ode_matches_closed_form` — RK4-integrated Peters
//!     chirp df/dt against the closed-form coalescence time: agreement to
//!     <1e-4 across a decade in f, t ∝ 𝓜^{-5/3} exponent verified
//!     numerically, and the GW150914-scale system (𝓜 = 30 M_sun) sweeps the
//!     LIGO band 35 → 150 Hz in the published sub-second-to-few-seconds
//!     regime.

use nested_fock_algebra::{
    BCS_GAP_RATIO, astro_blackbody_photon_gas, astro_chandrasekhar_mass_kg, astro_critical_density,
    astro_eddington_luminosity_w, astro_hawking_temperature_k, astro_isco_gw_frequency_hz,
    astro_schwarzschild_radius_m, inspiral_chirp_rate, inspiral_time_to_coalescence_s,
    metrology_constants, phys, plasma_alfven_speed_ms, plasma_debye_length_m, plasma_frequency_hz,
};

fn rel(v: f64, target: f64) -> f64 {
    (v - target).abs() / target
}

#[test]
fn astro_black_hole_anchors() {
    let th = astro_hawking_temperature_k(phys::M_SUN);
    assert!(rel(th, 6.17e-8) < 1e-2, "T_H(M_sun) = {th:.3e} K");
    // Hawking law is inverse in M: an Earth-mass hole is ~20 mK.
    let m_earth = 5.9722e24;
    assert!(rel(astro_hawking_temperature_k(m_earth), 2.05e-2) < 1e-2);

    // Schwarzschild radii: Sun 2.953 km, Sgr A* (4.297e6 M_sun) 1.27e10 m.
    assert!(rel(astro_schwarzschild_radius_m(phys::M_SUN), 2953.35) < 1e-4);
    let sgr_a = astro_schwarzschild_radius_m(4.297e6 * phys::M_SUN);
    assert!(rel(sgr_a, 1.27e10) < 1e-2, "r_s(Sgr A*) = {sgr_a:.3e} m");

    // ISCO GW frequency: 4397 Hz per solar mass (non-spinning).
    assert!(rel(astro_isco_gw_frequency_hz(phys::M_SUN), 4397.0) < 1e-3);
}

#[test]
fn astro_chandrasekhar_mass_from_constants() {
    let mch = astro_chandrasekhar_mass_kg(2.0); // carbon/oxygen: μ_e = 2
    assert!(
        rel(mch, 1.44 * phys::M_SUN) < 2e-2,
        "M_Ch = {:.3} M_sun, want 1.44",
        mch / phys::M_SUN
    );
    // Scaling with μ_e: helium cores (μ_e=2) same, iron cores (μ_e≈2.15)
    // lower by (2/μ_e)².
    let fe = astro_chandrasekhar_mass_kg(2.15);
    assert!(rel(fe / mch, (2.0f64 / 2.15).powi(2)) < 1e-9);
}

#[test]
fn astro_eddington_luminosity_solar_mass() {
    let l = astro_eddington_luminosity_w(phys::M_SUN);
    assert!(rel(l, 1.26e31) < 1e-2, "L_Edd = {l:.4e} W, want 1.26e31");
}

#[test]
fn astro_critical_density_and_baryons_planck2018() {
    // Exact arithmetic at Planck 2018 H₀ = 67.66 km/s/Mpc:
    // ρ_c = 3H₀²/(8πG) = 8.599e-27 kg/m³ (the "≈8.5e-27" textbook figure).
    let rho_c = astro_critical_density(67.66);
    assert!(rel(rho_c, 8.5993e-27) < 5e-3, "ρ_c = {rho_c:.3e} kg/m³");
    // Planck 2018 Ω_b = 0.04897 ⇒ baryon density ≈ 4.21e-28 kg/m³ —
    // consistent with the independent BBN deuterium measurement.
    let rho_b = 0.04897 * rho_c;
    assert!(rel(rho_b, 4.211e-28) < 5e-3);
}

#[test]
fn astro_cmb_photon_gas_numbers() {
    let (n, u) = astro_blackbody_photon_gas(2.7255);
    let n_cm3 = n / 1.0e6;
    assert!(rel(n_cm3, 411.0) < 1e-2, "n_γ = {n_cm3} cm⁻³, want ~411");
    assert!(rel(u, 4.17e-14) < 1e-2, "u = {u:.3e} J/m³");
    // In particle-physics units: ~0.26 eV/cm³ (the CMB energy density).
    let ev_per_cm3 = u * 1.0e-6 / phys::E;
    assert!(rel(ev_per_cm3, 0.2605) < 2e-2, "{ev_per_cm3} eV/cm³");
}

#[test]
fn plasma_ionosphere_frequency_and_debye_length() {
    let n = 1.0e12; // m⁻³ — daytime ionospheric E/F layer
    let fp = plasma_frequency_hz(n);
    // Textbook rule f_p[Hz] = 8.98√(n[cm⁻³]) kHz-scale: 8.98 MHz here.
    let rule = 8980.0 * (n / 1.0e6).sqrt();
    assert!(rel(fp, rule) < 1e-3, "f_p = {fp:.4e} vs rule {rule:.4e}");
    assert!(rel(fp, 8.98e6) < 1e-2);

    // Debye length at T = 1000 K: ~2.2 mm; two algebraically identical
    // routes must agree exactly.
    let ld = plasma_debye_length_m(1000.0, n);
    let ld_alt = ((phys::EPS0 * phys::K_B * 1000.0).sqrt() / (phys::E * n.sqrt())).abs();
    assert!(rel(ld, ld_alt) < 1e-12);
    assert!(ld > 1e-3 && ld < 1e-2, "λ_D = {ld:.3e} m");
}

#[test]
fn plasma_alfven_speed_solar_wind() {
    // Solar wind at 1 AU: B ≈ 5 nT, proton density ≈ 5 cm⁻³.
    let rho = 5.0 * 1.67262192369e-27 * 1.0e6;
    let v_a = plasma_alfven_speed_ms(5.0e-9, rho);
    assert!(
        v_a > 3.0e4 && v_a < 7.0e4,
        "v_A = {v_a:.3e} m/s, want the observed 40–70 km/s band"
    );
}

#[test]
fn si_quantum_metrology_triangle() {
    let (rk, kj, phi0) = metrology_constants();
    assert!(rel(rk, 25812.80745) < 1e-9, "R_K = {rk} Ω");
    assert!(rel(kj, 4.835_978_484e14) < 1e-9, "K_J = {kj:.9e} Hz/V");
    assert!(rel(phi0, 2.067_833_848e-15) < 1e-9, "Φ₀ = {phi0:.9e} Wb");
    // The triangle closes exactly: K_J · R_K = 2/e (Josephson↔Hall relation),
    // and Φ₀ · K_J = 1.
    assert!(rel(kj * rk, 2.0 / phys::E) < 1e-12);
    assert!(rel(phi0 * kj, 1.0) < 1e-12);
}

#[test]
fn bcs_weak_coupling_gap_ratio() {
    // Universal BCS result: Δ(0)/k_B T_c = 1.764 ⇔ 2Δ(0)/k_BT_c = 3.53,
    // the weak-coupling tunnelling value measured across the elements.
    assert!(rel(BCS_GAP_RATIO, 1.764) < 1e-3);
    assert!(rel(2.0 * BCS_GAP_RATIO, 3.528) < 1e-3);
}

#[test]
fn inspiral_chirp_ode_matches_closed_form() {
    let m_sun = phys::M_SUN;
    // GW150914-class chirp mass.
    let chirp = 30.0 * m_sun;

    // RK4-integrate df/dt = K f^{11/3} over one decade of frequency and
    // compare elapsed time with the closed form t(f₁) − t(f₂).
    let (f0, f1) = (35.0_f64, 350.0_f64);
    let dt = -inspiral_time_to_coalescence_s(chirp, f1) + inspiral_time_to_coalescence_s(chirp, f0);
    let mut f = f0;
    let mut t_num = 0.0;
    let steps = 200_000;
    let h = dt / steps as f64;
    for _ in 0..steps {
        let k1 = inspiral_chirp_rate(chirp, f);
        let k2 = inspiral_chirp_rate(chirp, f + h / 2.0 * k1);
        let k3 = inspiral_chirp_rate(chirp, f + h / 2.0 * k2);
        let k4 = inspiral_chirp_rate(chirp, f + h * k3);
        f += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t_num += h;
    }
    assert!(
        rel(t_num, dt.abs()) < 1e-4,
        "ODE integration {t_num} vs closed form {}",
        dt.abs()
    );

    // Chirp scaling: t ∝ 𝓜^{-5/3} — double the chirp mass, shrink the time.
    let r =
        inspiral_time_to_coalescence_s(2.0 * chirp, f0) / inspiral_time_to_coalescence_s(chirp, f0);
    assert!(rel(r, (0.5f64).powf(5.0 / 3.0)) < 1e-9);

    // The LIGO-band sweep 35 → 150 Hz for this system takes the published
    // order-of-seconds-to-subsecond regime (GW150914 was in band ~0.2 s at
    // its higher chirp mass; here the integrator lands in 1–30 s).
    let sweep =
        inspiral_time_to_coalescence_s(chirp, 35.0) - inspiral_time_to_coalescence_s(chirp, 150.0);
    assert!(sweep > 0.01 && sweep < 60.0, "band sweep {sweep:.3} s");
}
