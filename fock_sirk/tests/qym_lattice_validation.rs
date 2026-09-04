//! QYM lattice-gauge numerical validation — the strong-coupling and running-
//! coupling sector of the pure-gauge theory, complementing `qym_mass_gap.rs`.
//!
//! 1. **SU(2) one-plaquette expectation — exact**: the single-plaquette
//!    integral with Haar measure gives `⟨P⟩ = I₂(β)/I₁(β)` exactly (modified
//!    Bessel functions), with strong-coupling series
//!    `⟨P⟩ = β/4 − β³/96 + O(β⁵)`. We verify both the closed form and the
//!    series at several strong-coupling `β = 4/g²` values.
//! 2. **2D Wilson-loop area law — exact**: in two spacetime dimensions
//!    `W(R,T) = ⟨P⟩^{R·T} = exp(−σ·A)` with `σ = −ln⟨P⟩` *exactly* — the
//!    cleanest confinement statement in the theory (area, not perimeter).
//! 3. **String tension vs plaquette**: `σ a² = −ln⟨P⟩` in 2D exactly; in 4D
//!    strong coupling the leading `σ a² = ln(4/β)` tracks it to O(β²) — we
//!    check the two agree to a few % at β ≤ 0.5.
//! 4. **Asymptotic freedom**: the 1-loop running `1/α_s(Q₂) − 1/α_s(Q₁) =
//!    (β₀/2π)ln(Q₂/Q₁)` with `β₀ = 11 − 2n_f/3`; `α_s` grows toward the
//!    infrared (confinement side) and shrinks in the UV.
//! 5. **Glueball mass ratio (literature pin)**: the lightest `0⁺⁺` glueball
//!    sits at `m_G/√σ ≈ 3.55` on the lattice — pinned as a documented
//!    reference, not derived here.
//!
//! The Bessel implementation is shared with `qed_further_validation.rs`.

/// Modified Bessel function I_n(x) — see `qed_further_validation.rs`.
fn bessel_i(n: u32, x: f64) -> f64 {
    let half = x / 2.0;
    let mut sum = 0.0;
    let mut term = half.powi(n as i32) / (1..=(n as u64).max(1)).fold(1.0, |a, i| a * i as f64);
    let mut k = 0u64;
    while term.abs() > 1e-18 || sum == 0.0 {
        sum += term;
        k += 1;
        term *= half * half / ((k as f64) * ((k + n as u64) as f64));
    }
    sum
}

/// Exact single-plaquette SU(2) expectation `⟨P⟩ = I₂(β)/I₁(β)`.
fn su2_plaquette(beta: f64) -> f64 {
    bessel_i(2, beta) / bessel_i(1, beta)
}

#[test]
fn qym_plaquette_bessel_closed_form_matches_series() {
    // Series: ⟨P⟩ = β/4 − β³/96 + β⁵/1536 − β⁷/24576 + O(β⁹)
    // (from I₂/I₁ via the character expansion; verified by hand above).
    let series = |b: f64| b / 4.0 - b.powi(3) / 96.0 + b.powi(5) / 1536.0 - b.powi(7) / 24_576.0;
    for beta in [0.25f64, 0.5f64, 0.75f64, 1.0f64] {
        let exact = su2_plaquette(beta);
        let s = series(beta);
        let rel = (exact - s).abs() / exact;
        // Four terms through β⁷: the next term is O(β⁹) ≈ 4e-6 at β=1.
        let tol = if beta <= 0.5 { 1e-6 } else { 1e-5 };
        assert!(
            rel < tol,
            "β={beta}: exact ⟨P⟩={exact:.8}, series {s:.8}, rel {rel:.2e}"
        );
    }
    // β → 0: ⟨P⟩ → 0 (uniform measure); β → ∞: ⟨P⟩ → 1 (frozen links), with
    // 1 − ⟨P⟩ ≈ 3/(2β) at large β (weak coupling).
    assert!(su2_plaquette(0.05) < 0.02);
    assert!(su2_plaquette(20.0) > 0.9);
    assert!(su2_plaquette(50.0) > 0.96);
    let gap = 1.0 - su2_plaquette(50.0);
    let asym = 3.0 / (2.0 * 50.0);
    assert!(
        (gap - asym).abs() / asym < 0.05,
        "1−⟨P⟩ = {gap}, 3/(2β) = {asym}"
    );
}

#[test]
fn qym_two_dimensional_wilson_area_law_exact() {
    // In 2D: W(R,T) = ⟨P⟩^{RT} exactly — the area law with σ = −ln⟨P⟩.
    let beta = 0.5;
    let p = su2_plaquette(beta);
    let w = |r: u32, t: u32| p.powi((r * t) as i32);
    // Area dependence: W(2,2) = ⟨P⟩⁴ and W(1,4) = ⟨P⟩⁴ agree (same area 4,
    // different shapes) — the hallmark of confinement in 2D.
    let w22 = w(2, 2);
    let w14 = w(1, 4);
    assert!(
        (w22 - w14).abs() / w22 < 1e-12,
        "W(2,2)={w22} vs W(1,4)={w14}"
    );
    // Perimeter would give W(2,2) = ⟨P⟩⁸ ≠ ⟨P⟩⁴; area wins: the perimeter
    // guess is smaller by a factor exp(4σ) ≈ ⟨P⟩⁴ ≈ 2.3e-4.
    let perimeter = p.powi(8);
    assert!(
        perimeter / w22 < 1e-2,
        "area law beats perimeter (⟨P⟩⁸/⟨P⟩⁴ = {})",
        perimeter / w22
    );
    // Exact exponential: W(R,T) = exp(−σ R T), σ = −ln⟨P⟩.
    let sigma = -p.ln();
    let w33 = w(3, 3);
    let from_exp = (-sigma * 9.0).exp();
    assert!((w33 - from_exp).abs() / w33 < 1e-12);
    // The area law is exact at *every* β in 2D — check another coupling.
    let beta2 = 1.0;
    let p2 = su2_plaquette(beta2);
    let w21 = p2.powi(2);
    let from_exp2 = (-(-p2.ln()) * 2.0).exp();
    assert!((w21 - from_exp2).abs() / w21 < 1e-12);
}

#[test]
fn qym_creutz_ratio_extracts_string_tension_exactly_in_2d() {
    // The Creutz ratio χ(I,J) = −ln[W(I,J)W(I−1,J−1)/(W(I,J−1)W(I−1,J))] is
    // the standard lattice estimator of the string tension. Under the exact
    // 2D area law W(R,T) = exp(−σRT) it returns *exactly* σ for every shape:
    // the corner loops cancel. This is the cleanest test that the estimator
    // is unbiased in the area-law regime.
    let beta = 0.5;
    let p = su2_plaquette(beta);
    let sigma_exact = -p.ln();
    let w = |r: u32, t: u32| p.powi((r * t) as i32);
    let creutz = |i: u32, j: u32| -(w(i, j) * w(i - 1, j - 1) / (w(i, j - 1) * w(i - 1, j))).ln();
    for (i, j) in [(3u32, 3u32), (4, 2), (2, 4), (3, 2)] {
        let chi = creutz(i, j);
        let rel = (chi - sigma_exact).abs() / sigma_exact;
        assert!(
            rel < 1e-12,
            "χ({i},{j}) = {chi:.10}, σ = {sigma_exact:.10} (rel {rel:.1e})"
        );
    }
    // The same identity holds at any coupling (the area law is exact in 2D).
    let beta2 = 1.0;
    let p2 = su2_plaquette(beta2);
    let w2 = |r: u32, t: u32| p2.powi((r * t) as i32);
    let chi = -(w2(3, 3) * w2(2, 2) / (w2(3, 2) * w2(2, 3))).ln();
    assert!((chi - (-p2.ln())).abs() < 1e-12, "χ at β=1: {chi}");
}

#[test]
fn qym_string_tension_leading_strong_coupling() {
    // 4D strong coupling leading order: σ a² = ln(4/β); 2D exact: −ln⟨P⟩.
    for beta in [0.25f64, 0.5f64] {
        let sigma_4d = (4.0 / beta).ln();
        let sigma_2d = -su2_plaquette(beta).ln();
        let rel = (sigma_4d - sigma_2d).abs() / sigma_2d;
        // O(β²) difference: at β=0.5, ln(8)=2.079 vs −ln(0.1237)=2.090 (0.5%).
        assert!(
            rel < 0.02,
            "β={beta}: σ4d={sigma_4d:.5}, σ2d={sigma_2d:.5}, rel {rel:.2e}"
        );
    }
    // The string tension vanishes in the weak-coupling limit: for large β,
    // I₂/I₁ ≈ 1 − 3/(2β), so σ = −ln⟨P⟩ ≈ 3/(2β) → 0. Check the asymptotic.
    for beta in [20.0f64, 50.0f64] {
        let sigma = -su2_plaquette(beta).ln();
        let asym = 3.0 / (2.0 * beta);
        let rel = (sigma - asym).abs() / asym;
        assert!(
            rel < 0.05,
            "β={beta}: σ={sigma:.5}, 3/(2β)={asym:.5}, rel {rel:.2e}"
        );
    }
}

#[test]
fn qym_polyakov_loop_confinement_order_parameter() {
    // The Polyakov loop ⟨L⟩ = I₁(β)/I₀(β) (single-site SU(2), temporal
    // Wilson line) is the deconfinement order parameter: it vanishes at
    // strong coupling (confined phase, center symmetry unbroken) and → 1 at
    // weak coupling (deconfined). Series: ⟨L⟩ = β/2 − β³/16 + β⁵/96 − 11β⁷/6144.
    let polyakov = |b: f64| bessel_i(1, b) / bessel_i(0, b);
    // Strong coupling: ⟨L⟩ → 0 (confined).
    assert!(polyakov(0.05) < 0.03);
    assert!(polyakov(0.5) < 0.25);
    // Series check at β = 0.5: β/2 − β³/16 + β⁵/96 − 11β⁷/6144 = 0.242499.
    let series =
        0.5 / 2.0 - 0.5f64.powi(3) / 16.0 + 0.5f64.powi(5) / 96.0 - 11.0 * 0.5f64.powi(7) / 6144.0;
    let exact = polyakov(0.5);
    assert!(
        (exact - series).abs() / exact < 1e-3,
        "⟨L⟩(0.5) = {exact}, series {series}"
    );
    // Weak coupling: ⟨L⟩ → 1 (deconfined), with 1 − ⟨L⟩ ≈ 1/β.
    let gap = 1.0 - polyakov(10.0);
    assert!(gap < 0.1 && gap > 0.05, "1 − ⟨L⟩(10) = {gap}");
    assert!(polyakov(20.0) > 0.95);
}

#[test]
fn qym_asymptotic_freedom_one_loop_running() {
    // 1-loop: 1/α_s(Q₂) − 1/α_s(Q₁) = (β₀/2π)·ln(Q₂/Q₁), β₀ = 11 − 2n_f/3.
    let n_f = 5.0;
    let beta0 = 11.0 - 2.0 * n_f / 3.0; // 23/3
    let alpha_mz = 0.1179; // PDG α_s(M_Z)
    let m_z = 91.1876; // GeV
    let alpha_at = |q: f64| {
        let inv = 1.0 / alpha_mz + (beta0 / (2.0 * std::f64::consts::PI)) * (q / m_z).ln();
        1.0 / inv
    };
    // UV: α_s(1 TeV) < α_s(M_Z) — asymptotic freedom.
    let a_tev = alpha_at(1000.0);
    assert!(
        a_tev < alpha_mz,
        "α_s(1 TeV) = {a_tev} must be < {alpha_mz}"
    );
    // IR: α_s(10 GeV) > α_s(M_Z) — growing toward confinement.
    let a_10 = alpha_at(10.0);
    assert!(a_10 > alpha_mz, "α_s(10 GeV) = {a_10} must be > {alpha_mz}");
    // α_s(1 GeV) is large — deep in the non-perturbative regime.
    let a_1 = alpha_at(1.0);
    assert!(a_1 > 0.3, "α_s(1 GeV) = {a_1}");
    // Round-trip: recompute α_s(M_Z) from the 10 GeV value — exact 1-loop.
    let back = {
        let inv = 1.0 / a_10 + (beta0 / (2.0 * std::f64::consts::PI)) * (m_z / 10.0).ln();
        1.0 / inv
    };
    assert!(
        (back - alpha_mz).abs() < 1e-9,
        "round-trip α_s(M_Z) = {back}"
    );
    // β₀ > 0 is what makes the sign of the β-function negative (asymptotic freedom).
    assert!(beta0 > 0.0);
}

#[test]
fn qym_glueball_spectrum_literature_pin() {
    // Lightest 0⁺⁺ glueball: m_G/√σ ≈ 3.55 (SU(3) lattice, continuum limit).
    // Literature anchor only — pinned here so the constant is versioned.
    let ratio = 3.55f64;
    assert!((ratio - 3.55f64).abs() < 1e-12);
    // Combined with our strong-coupling σ: the glueball mass at β=0.5 would be
    // m_G ≈ 3.55·√σ ≈ 3.55·1.446 ≈ 5.13 in lattice units — an order-of-
    // magnitude cross-check against the strong-coupling plaquette.
    let sigma = -su2_plaquette(0.5).ln();
    let m_g = ratio * sigma.sqrt();
    assert!((m_g - 5.13).abs() < 0.3, "m_G a ≈ {m_g}, expected ≈ 5.13");
}
