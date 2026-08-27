//! QED further numerical validation — perturbative and field-theoretic anchors.
//!
//! Extends `qed_extended_validation.rs` with the perturbative expansion and
//! the Schwinger pair-production sector:
//!
//! 1. **Electron g−2 anomaly**: `a_e = α/(2π) − 0.3284789656(α/π)² + …`
//!    against the CODATA value `a_e = 0.00115965218…` (the `α/(2π)` leading
//!    term is the celebrated Schwinger result; the `(α/π)²` coefficient is the
//!    Petermann–Sommerfield term). We verify both the leading term and the
//!    two-term value.
//! 2. **Schwinger critical field**: `E_c = m_e²c³/(eℏ) = 1.323e18 V/m` — the
//!    field at which the vacuum pair-production rate is unsuppressed.
//! 3. **Schwinger pair rate suppression**: `Γ ∝ exp(−πE_c/E)` — the
//!    exponential barrier is the reason `E ≪ E_c` produces no pairs; we verify
//!    the `−πE_c/E` exponent at two sub-critical fields.
//! 4. **Fine-structure running (leptonic, 1-loop)**: `α(M_Z) = α(0)/
//!    (1 − (α(0)/3π) Σ_Q ln(M_Z/m_Q))` — vacuum polarization *screens* charge,
//!    so `α` increases with energy; the leptonic-only value
//!    `1/α_lept(M_Z) ≈ 134.6` sits above the full `1/α(M_Z) = 128.9` (the
//!    hadronic contribution `Δα_had ≈ 0.028` closes the gap).
//!
//! All constants are CODATA 2018; every assertion is an exact formula
//! evaluation with a documented tolerance.

// --- CODATA 2018 ---
const ALPHA: f64 = 7.297_352_569_3e-3; // fine-structure constant (1/137.035999084)
const M_E_EV: f64 = 0.510_998_950_0e6; // electron mass [eV]
const M_MU_EV: f64 = 105.658_375_5e6; // muon mass [eV]
const M_TAU_EV: f64 = 1_776.86e6; // tau mass [eV]
const M_Z_EV: f64 = 91.1876e9; // Z boson mass [eV]
const M_E_KG: f64 = 9.109_383_701_5e-31;
const C: f64 = 2.997_924_58e8;
const HBAR: f64 = 1.054_571_817e-34;
const H_PLANCK: f64 = 6.626_070_15e-34;
const EPS0: f64 = 8.854_187_812_8e-12;
const E_CHARGE: f64 = 1.602_176_634e-19;
const EV: f64 = 1.602_176_634e-19; // joule per eV

/// Modified Bessel function I_n(x) via the absolutely-convergent series
/// `Σ_k (x/2)^{2k+n} / (k!(k+n)!)`. Converges in < 60 terms for x ≤ 20.
fn bessel_i(n: u32, x: f64) -> f64 {
    let half = x / 2.0;
    let mut sum = 0.0;
    let mut term = (half.powi(n as i32)) / factorial(n as u64);
    let mut k = 0u64;
    while term.abs() > 1e-18 || sum == 0.0 {
        sum += term;
        k += 1;
        term *= half * half / ((k as f64) * ((k + n as u64) as f64));
    }
    sum
}

fn factorial(n: u64) -> f64 {
    (1..=n.max(1)).fold(1.0, |a, i| a * i as f64)
}

#[test]
fn qed_anomalous_moment_leading_schwinger_term() {
    // a_e^(1) = α/(2π) — the one-loop Schwinger value.
    let a1 = ALPHA / (2.0 * std::f64::consts::PI);
    let expected = 0.001_161_409_733; // α/(2π) to 10 digits
    assert!(
        (a1 - expected).abs() < 1e-10,
        "α/(2π) = {a1}, expected {expected}"
    );
}

#[test]
fn qed_anomalous_moment_two_loop_matches_codata() {
    // a_e = α/(2π) − 0.3284789656·(α/π)² + 1.181234017·(α/π)³ + O(α⁴)
    let x = ALPHA / std::f64::consts::PI;
    let a1 = ALPHA / (2.0 * std::f64::consts::PI);
    let a2 = -0.328_478_965_6 * x.powi(2);
    let a3 = 1.181_234_017 * x.powi(3);
    let a_e = a1 + a2 + a3;
    let codata = 0.001_159_652_18; // CODATA 2018 electron anomaly
    assert!(
        (a_e - codata).abs() < 1e-9,
        "a_e(3-loop) = {a_e}, CODATA = {codata}, Δ = {}",
        (a_e - codata).abs()
    );
    // The second-order term must be negative and ~1.5e-6; the third ~+1.5e-8.
    assert!(a2 < 0.0 && a2.abs() > 1e-6, "a_e^(2) = {a2}");
    assert!(a3 > 0.0 && (a3 - 1.481e-8).abs() < 1e-11, "a_e^(3) = {a3}");
}

#[test]
fn qed_schwinger_critical_field_pin() {
    // E_c = m_e² c³ / (e ħ)
    let e_c = M_E_KG * M_E_KG * C.powi(3) / (E_CHARGE * HBAR);
    let expected = 1.323e18; // V/m, textbook value
    let rel = (e_c - expected).abs() / expected;
    assert!(rel < 1e-3, "E_c = {e_c:.6e} V/m, expected {expected:.6e} (rel {rel:.2e})");
}

#[test]
fn qed_schwinger_rate_exponential_barrier() {
    // Γ ∝ (eE)² exp(−π E_c / E): the log of the rate is linear in E_c/E with
    // slope −π. Verify the exponent at two sub-critical fields.
    let e_c = 1.323e18;
    let barrier = |e: f64| std::f64::consts::PI * e_c / e;
    let e1 = 0.1 * e_c;
    let e2 = 0.2 * e_c;
    let ratio = (barrier(e2) - barrier(e1)).exp(); // exp(−πE_c/E2)/exp(−πE_c/E1)
    // exp(−5π) ≈ 1.5e−7 — two orders of magnitude of suppression per decade of field.
    let expected = (-5.0 * std::f64::consts::PI).exp();
    assert!(
        (ratio - expected).abs() / expected < 1e-6,
        "barrier ratio = {ratio}, expected {expected}"
    );
    // Prefactor (eE)² is polynomial and irrelevant next to the exponential at E ≪ E_c.
    let prefactor_ratio = (e2 / e1).powi(2); // 4
    assert!((prefactor_ratio - 4.0).abs() < 1e-12);
}

#[test]
fn qed_fine_structure_runs_upwards_leptonic() {
    // 1-loop leptonic screening: α(μ) = α / (1 − (α/3π) Σ_Q ln(μ/m_Q))
    let sum_ln: f64 = [M_E_EV, M_MU_EV, M_TAU_EV]
        .iter()
        .map(|m| (M_Z_EV / m).ln())
        .sum();
    let d_alpha = ALPHA * sum_ln / (3.0 * std::f64::consts::PI);
    let alpha_z = ALPHA / (1.0 - d_alpha);
    let inv_alpha_z = 1.0 / alpha_z;
    // Leptonic-only running: 1/α(M_Z) ≈ 134.6 (full QED+hadronic is 128.9).
    assert!(
        (inv_alpha_z - 134.6).abs() < 0.3,
        "1/α_lept(M_Z) = {inv_alpha_z}, expected ≈ 134.6"
    );
    // Charge screening: α must increase with energy.
    assert!(alpha_z > ALPHA, "α(M_Z) = {alpha_z} must exceed α(0) = {ALPHA}");
    // The leptonic piece accounts for most of the shift from 137.04 → 128.9.
    assert!((134.6 - inv_alpha_z).abs() < 0.3);
    assert!((inv_alpha_z - 128.9).abs() > 4.0, "hadronic Δα must be missing");
}

#[test]
fn qed_fine_structure_from_si_constants() {
    // The metrology triangle: α = e²/(4πε₀ℏc) from SI constants must return
    // the CODATA fine-structure constant — the single most precisely known
    // physical constant, now defined by the SI redefinition.
    let alpha = E_CHARGE * E_CHARGE / (4.0 * std::f64::consts::PI * EPS0 * HBAR * C);
    let rel = (alpha - ALPHA).abs() / ALPHA;
    assert!(rel < 1e-8, "α from SI = {alpha}, CODATA {ALPHA} (rel {rel:.2e})");
}

#[test]
fn qed_rydberg_and_hydrogen_ionization() {
    // R_∞ = α²m_ec/(2h) = 1.09737e7 m⁻¹ (CODATA), and the hydrogen
    // ionization energy E = ½α²m_ec² = 13.6057 eV — the binding scale of
    // atomic physics built from the same constants.
    let r_inf = ALPHA * ALPHA * M_E_KG * C / (2.0 * H_PLANCK);
    let expected_r = 1.097_373_156_8e7; // m⁻¹
    let rel_r = (r_inf - expected_r).abs() / expected_r;
    assert!(rel_r < 1e-7, "R_∞ = {r_inf:.6e}, expected {expected_r:.6e}");
    let e_ion_j = 0.5 * ALPHA * ALPHA * M_E_KG * C * C;
    let e_ion_ev = e_ion_j / EV;
    assert!(
        (e_ion_ev - 13.6057).abs() < 0.01,
        "E_ion = {e_ion_ev:.5} eV, expected 13.6057"
    );
    // λ_C = h/(m_ec) = 2.42631e-12 m — the electron Compton wavelength.
    let lambda_c = H_PLANCK / (M_E_KG * C);
    let expected_l = 2.426_310_238_67e-12; // m
    let rel_l = (lambda_c - expected_l).abs() / expected_l;
    assert!(rel_l < 1e-8, "λ_C = {lambda_c:.6e}, expected {expected_l:.6e}");
}

#[test]
fn qed_casimir_three_dimensional_coefficient() {
    // The famous 3D Casimir force: E/A = −π²ħc/(720d³). Verify the exact
    // coefficient −π²/720 and the d⁻³ scaling (doubling the gap divides the
    // energy density by 8). The 1D seed E = −π/(24d) is pinned in
    // `qed_extended_validation.rs`; this is the physical 3D lift.
    let coeff = std::f64::consts::PI * std::f64::consts::PI / 720.0;
    let expected = 0.013_707_783_26; // π²/720
    assert!((coeff - expected).abs() < 1e-9, "π²/720 = {coeff}");
    // E/A·d³/(ħc) = −π²/720 is dimensionless: check the scaling law.
    let e_over_a = |d: f64| -coeff * HBAR * C / d.powi(3);
    let d = 1e-6;
    let e1 = e_over_a(d);
    let e2 = e_over_a(2.0 * d);
    assert!((e2 / e1 - 0.125).abs() < 1e-12, "E/A ∝ d⁻³");
    // Numeric pin at d = 1 µm: E/A = −π²ħc/(720·(1e-6)³) = −4.334e-10 J/m².
    let expected_ea = -4.334e-10;
    let rel = (e1 - expected_ea).abs() / expected_ea.abs();
    assert!(rel < 1e-2, "E/A(1 µm) = {e1:.3e} J/m², expected {expected_ea:.3e}");
    // The pressure is the derivative: F/A = −d(E/A)/dd = −π²ħc/(240d⁴),
    // so F·d = 3·(E/A) exactly for the d⁻³ law.
    let f_over_a = -coeff * 3.0 * HBAR * C / d.powi(4); // −π²ħc/(240d⁴)
    assert!((f_over_a * d - 3.0 * e1).abs() / (3.0 * e1).abs() < 1e-12);
}

#[test]
fn qed_bessel_series_matches_known_values() {
    // Sanity-pin the Bessel implementation used by the QYM lattice suite.
    assert!((bessel_i(0, 0.0) - 1.0).abs() < 1e-12);
    assert!((bessel_i(0, 1.0) - 1.266_065_877).abs() < 1e-8);
    assert!((bessel_i(1, 1.0) - 0.565_159_104).abs() < 1e-8);
    assert!((bessel_i(2, 1.0) - 0.135_747_669).abs() < 1e-8);
}
