//! QG cosmology and black-hole thermodynamics numerical validation.
//!
//! Extends `qg_validation.rs` with:
//!
//! 1. **Friedmann integration** — the scale-factor equation
//!    `ȧ = H₀√(Ω_m/a + Ω_r/a² + Ω_Λ a²)` integrated with RK4 and matched
//!    against the closed forms for the pure matter (`a ∝ t^{2/3}`), pure
//!    radiation (`a ∝ t^{1/2}`) and pure Λ (`a ∝ e^{H₀t}`) universes, and
//!    against the flat ΛCDM age formula.
//! 2. **The age of the universe** — the closed-form flat ΛCDM age
//!    `t₀ = (2/(3H₀√Ω_Λ))·arcsinh(√(Ω_Λ/Ω_m))` at the Planck/PDG
//!    `H₀ = 67.66 km/s/Mpc, Ω_m = 0.31, Ω_Λ = 0.69` is ≈ 13.8 Gyr, and the
//!    numerical integration reproduces it.
//! 3. **Starobinsky e-folds** — the slow-roll e-fold number
//!    `N_e(φ) = ∫ V/V' dφ` integrated numerically against the closed form
//!    `(3/4)(e^{kφ} − e^{kφ_end} − k(φ−φ_end))` (k = √(2/3), M = 1) and the
//!    large-φ asymptotic `N_e ≈ (3/4)e^{kφ}` — the inflation count of the
//!    R² scalaron.
//! 4. **Schwarzschild thermodynamics** — the Smarr identity `M = 2TS`,
//!    the Bekenstein–Hawking entropy `S = A/(4ℓ_P²)` (≈ 10⁷⁷ k_B for the
//!    Sun), and the saturation of the Bekenstein bound `S = 2π r_s Mc/ℏ` —
//!    all exact identities evaluated with the CODATA constants.

use nested_fock_algebra::{QG_C, QG_G, QG_HBAR};

/// Hubble constant in s⁻¹ from km/s/Mpc (1 Mpc = 3.0856775814913673e19 km).
fn h0_from_km_s_mpc(h0_km: f64) -> f64 {
    let mpc_km = 3.085_677_581_491_367e19;
    h0_km / mpc_km
}

/// RK4 integration of `dt/da = 1/f(a)` over `a ∈ [a_min, a_max]` with `n`
/// steps. Returns the elapsed time in units of `1/H₀`.
fn friedmann_age(a_min: f64, a_max: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    let h = (a_max - a_min) / n as f64;
    let mut a = a_min;
    let mut t = 0.0;
    for _ in 0..n {
        let k1 = 1.0 / f(a);
        let k2 = 1.0 / f(a + h / 2.0);
        let k3 = 1.0 / f(a + h / 2.0);
        let k4 = 1.0 / f(a + h);
        t += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        a += h;
    }
    t
}

/// `f(a) = ȧ/H₀ = √(Ω_m/a + Ω_r/a² + Ω_Λ a²)`.
fn scale_factor_rate(omega_m: f64, omega_r: f64, omega_l: f64, a: f64) -> f64 {
    (omega_m / a + omega_r / (a * a) + omega_l * a * a).sqrt()
}

#[test]
fn qg_friedmann_matter_and_radiation_closed_forms() {
    // Pure matter: ȧ = H₀/√a ⇒ t(a) = (2/3H₀)a^{3/2}, so the age from a_min
    // to 1 is (2/3H₀)(1 − a_min^{3/2}).
    let a_min = 1e-3;
    let t_num = friedmann_age(a_min, 1.0, 200_000, |a| scale_factor_rate(1.0, 0.0, 0.0, a));
    let t_exact = (2.0 / 3.0) * (1.0 - a_min.powf(1.5));
    assert!(
        (t_num - t_exact).abs() / t_exact < 1e-5,
        "matter era: {t_num} vs closed form {t_exact}"
    );

    // Pure radiation: ȧ = H₀/a ⇒ t(a) = a²/(2H₀), age = (1 − a_min²)/(2H₀).
    let t_num = friedmann_age(a_min, 1.0, 200_000, |a| scale_factor_rate(0.0, 1.0, 0.0, a));
    let t_exact = (1.0 - a_min * a_min) / 2.0;
    assert!(
        (t_num - t_exact).abs() / t_exact < 1e-5,
        "radiation era: {t_num} vs closed form {t_exact}"
    );

    // Pure Λ: ȧ = H₀a ⇒ t(a) = ln(a)/H₀, age = −ln(a_min)/H₀.
    let t_num = friedmann_age(a_min, 1.0, 200_000, |a| scale_factor_rate(0.0, 0.0, 1.0, a));
    let t_exact = -a_min.ln();
    assert!(
        (t_num - t_exact).abs() / t_exact < 1e-5,
        "Λ era: {t_num} vs closed form {t_exact}"
    );

    // Radiation domination early: at small a the radiation term (a^{-2})
    // dominates the matter term (a^{-1}) — the ratio grows as 1/a.
    let f_rad = scale_factor_rate(0.31, 9e-5, 0.69, 1e-6);
    let f_matter_only = scale_factor_rate(0.31, 0.0, 0.69, 1e-6);
    assert!(f_rad / f_matter_only > 5.0, "radiation must dominate early");
    eprintln!(
        "qg_friedmann_matter_and_radiation_closed_forms: RK4 matches a ~ t^(2/3), a ~ t^(1/2), a ~ e^(H0 t)"
    );
}

#[test]
fn qg_lcdm_universe_age() {
    // Flat ΛCDM (Ω_r = 0): the closed-form age
    //   t₀ = (2/(3H₀√Ω_Λ))·arcsinh(√(Ω_Λ/Ω_m))
    // at the PDG values is ≈ 13.8 Gyr; the numerical integration of the full
    // equation reproduces it, and the radiation-era start is negligible.
    let h0 = h0_from_km_s_mpc(67.66);
    let omega_m: f64 = 0.31;
    let omega_l: f64 = 0.69;
    let sec_per_gyr = 3.155_76e16;

    let age_gyr_closed =
        (2.0 / (3.0 * h0 * omega_l.sqrt())) * ((omega_l / omega_m).sqrt()).asinh() / sec_per_gyr;

    let a_min = 1e-6;
    let age_gyr_num = friedmann_age(a_min, 1.0, 500_000, |a| {
        scale_factor_rate(omega_m, 1.0 - omega_m - omega_l, omega_l, a)
    }) / h0
        / sec_per_gyr;

    // Published value: 13.787 ± 0.020 Gyr (Planck 2018).
    assert!(
        (age_gyr_closed - 13.787).abs() < 0.1,
        "closed-form age must be ≈ 13.8 Gyr, got {age_gyr_closed:.3}"
    );
    assert!(
        (age_gyr_num - age_gyr_closed).abs() / age_gyr_closed < 5e-3,
        "numerical age {age_gyr_num:.4} must match the closed form {age_gyr_closed:.4}"
    );
    eprintln!(
        "qg_lcdm_universe_age: t₀ = {age_gyr_num:.3} Gyr (numerical), {age_gyr_closed:.3} Gyr (closed), published 13.787"
    );
}

/// The Starobinsky potential `V = (1 − e^{−kφ})²` (M = 1, k = √(2/3)) and
/// the slow-roll integrand `V/V' = (e^{kφ} − 1)/(2k)`.
fn starobinsky_efold_integrand(k: f64, phi: f64) -> f64 {
    // V/V' = (e^{kφ} − 1) / (2k)
    (((phi * k).exp()) - 1.0) / (2.0 * k)
}

/// Simpson integration of the e-fold integrand from `phi_end` to `phi`.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + (i as f64) * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    sum * h / 3.0
}

#[test]
fn qg_starobinsky_efolds() {
    // N_e(φ) = ∫_{φ_end}^{φ} V/V' dφ' = (e^{kφ} − e^{kφ_end} − k(φ−φ_end))/(2k²)
    // = (3/4)(e^{kφ} − e^{kφ_end} − k(φ−φ_end)) for k = √(2/3), M = 1.
    let k = (2.0f64 / 3.0).sqrt();
    let phi_end = 0.7;
    let phi = 6.0;
    let closed = (3.0 / 4.0) * (((phi * k).exp() - (phi_end * k).exp()) - k * (phi - phi_end));
    let num = simpson(
        |x| starobinsky_efold_integrand(k, x),
        phi_end,
        phi,
        1_000_000,
    );
    assert!(
        (num - closed).abs() / closed < 1e-6,
        "N_e numerical {num:.6} vs closed form {closed:.6}"
    );

    // Large-φ asymptotic: N_e ≈ (3/4)e^{kφ} — the −kφ and −e^{kφ_end} terms
    // are negligible only at genuinely large φ (at φ = 6 the −kφ term is
    // still 4.5%); check at φ = 10 where the asymptotic holds to < 1%.
    let _asym = (3.0 / 4.0) * (phi * k).exp();
    let phi10 = 10.0;
    let num10 = simpson(
        |x| starobinsky_efold_integrand(k, x),
        phi_end,
        phi10,
        2_000_000,
    );
    let asym10 = (3.0 / 4.0) * (phi10 * k).exp();
    assert!(
        (num10 - asym10).abs() / num10 < 1e-2,
        "N_e(10) must approach (3/4)e^(k·10): {num10:.3} vs {asym10:.3}"
    );
    assert!(
        num > 50.0 && num < 200.0,
        "sensible inflation count: {num:.1}"
    );
    eprintln!(
        "qg_starobinsky_efolds: N_e(6) = {num:.3} (closed {closed:.3}); N_e(10) = {num10:.3} (asymptotic {asym10:.3})"
    );
}

#[test]
fn qg_schwarzschild_black_hole_thermodynamics() {
    // Bekenstein–Hawking: S = k_B·A/(4ℓ_P²), A = 4πr_s², r_s = 2GM/c²,
    // ℓ_P² = ℏG/c³. Hawking temperature T = ℏc³/(8πGMk_B). The Smarr
    // identity M c² = 2TS and the saturation of the Bekenstein bound
    // S = 2π r_s M c/ℏ are exact — verified to machine precision with the
    // CODATA constants.
    let m_sun = 1.988_47e30;
    let k_b = 1.380_649e-23;

    for &m in &[m_sun, 6.0 * m_sun, 1.0e10 * m_sun] {
        let r_s = 2.0 * QG_G * m / (QG_C * QG_C);
        let area = 4.0 * std::f64::consts::PI * r_s * r_s;
        let lp2 = QG_HBAR * QG_G / (QG_C * QG_C * QG_C);
        let s_kb = area / (4.0 * lp2); // S / k_B

        // S = A/(4ℓ_P²) is the definition; cross-check the explicit form.
        let s_form = 4.0 * std::f64::consts::PI * QG_G * m * m / (QG_HBAR * QG_C);
        assert!(
            (s_kb - s_form).abs() / s_kb < 1e-12,
            "S must equal 4πGM²/(ℏc)"
        );

        // Hawking temperature (K).
        let t_k = QG_HBAR * QG_C * QG_C * QG_C / (8.0 * std::f64::consts::PI * QG_G * m * k_b);
        // Smarr: M c² = 2 T S.
        let two_ts = 2.0 * t_k * (s_kb * k_b);
        let mc2 = m * QG_C * QG_C;
        assert!(
            (two_ts - mc2).abs() / mc2 < 1e-9,
            "Smarr M c² = 2TS must hold: {two_ts:.6e} vs {mc2:.6e}"
        );

        // Bekenstein bound saturation: S = 2π r_s M c / ℏ.
        let bound = 2.0 * std::f64::consts::PI * r_s * m * QG_C / QG_HBAR;
        assert!(
            (s_kb - bound).abs() / s_kb < 1e-12,
            "BH must saturate the Bekenstein bound: {s_kb:.6e} vs {bound:.6e}"
        );

        // Solar-mass entropy ≈ 10⁷⁷ k_B (published order).
        if m == m_sun {
            assert!(
                (s_kb - 1.050e77).abs() / 1.050e77 < 2e-2,
                "S(M_sun) ≈ 1.05e77 k_B, got {s_kb:.3e}"
            );
        }
    }
    eprintln!(
        "qg_schwarzschild_black_hole_thermodynamics: Smarr M=2TS, S=A/(4ℓ_P²), bound saturation all exact"
    );
}
