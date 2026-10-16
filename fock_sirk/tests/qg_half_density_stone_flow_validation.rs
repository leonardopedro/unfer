//! Numerical validation of the QG-D-C rows of the TRANSFORMED-ESA work order.
//!
//! Companion to the densitized rows **QG-D-C.2 / QG-D-C.3** of
//! `../timepiece/CONSOLIDATED_PLAN.md` (2026-10-15, "closing the ESA of the
//! TRANSFORMED operator") and, alongside
//! `transformed_esa_charts_validation.rs` (CHECK 8/9 mirrors), the numerical
//! companion of `docs/transformed_esa_charts.cdb`.  The doctrine: a classical
//! singularity does not decide ESA — the variables do.  The densitized change
//! of variables `e = y²` absorbs the singular `1/e` of the gravity kinetic
//! terms, the change of variables is made unitary by the Jacobian
//! half-density factor, and the transformed (flat) operator is the one whose
//! flow is complete.
//!
//! What is checked here, module by module:
//!
//! A. **The half-density unitary Jacobian**
//!    (`timepiece/BookProof/ChapterQuantumGravityHalfDensity.lean` — the
//!    *constructed* unitary that the plan's Part D.4 transfer used to take as
//!    data):
//!
//! 1. `hd_jacobian_and_half_density_factor` — `qgJacobian y = 2y = de/dy` at
//!    `e = y²`, and the weight `2y` is the square of the half-density factor
//!    `qgHalfDensity y = √(2y)`
//!    (`qgSrcMeasure_density_eq_halfDensity_sq`).
//!
//! 2. `hd_unitary_norm_preservation` — `halfDensityUnitary_norm`: the map
//!    `(W g)(y) = g(y²)` from `L²((0,∞), de)` to `L²((0,∞), 2y dy)` preserves
//!    the norm, `∫|g(e)|² de = ∫|g(y²)|² 2y dy`, both sides integrated
//!    independently (the `y`-side through a half-line tan compactification
//!    where the tails are algebraic).
//!
//! 3. `hd_measure_pushforward` — `measurePreserving_qgSquare`: pushing `2y dy`
//!    forward along `y ↦ y²` gives `de`; checked on test functions that are
//!    NOT modulus squares.
//!
//! 4. `hd_intertwines_multiplication` — the `halfDensityUnitary_intertwines`
//!    face: expectations of a multiplication operator transport through the
//!    unitary, `⟨W g, (V∘sq) W g⟩ = ⟨g, V g⟩`, with the two sides on
//!    different measures.
//!
//! B. **The Stone flow of the `R + αR²` mode Hamiltonian**
//!    (`timepiece/BookProof/ChapterStarobinskyPotential.lean` —
//!    `qgR2ModeHamiltonian` = multiplication by the mode symbol
//!    `λ_k = (1/16)a_k² − (1/24)b_k² + V₃(R_c k)` of
//!    `BookProof.QuantumGravityDensitized.qgModeSymbol`, and the deliverable
//!    `qgR2_stone_flow`):
//!
//! 5. `qg_conf_v_completed_square_and_bound` — `confV_completed_square` /
//!    `confV_ge`: `V₃(R) = −(M²/2)R + αR² = α(R − M²/(4α))² − M⁴/(16α)`, so
//!    the `αR²` term regularizes the conformal mode: bounded below by
//!    `−M⁴/(16α)` for `α > 0`, while `α = 0` runs to `−∞`
//!    (`confV_zero_alpha_tendsto_atBot`).
//!
//! 6. `qg_mode_symbol_real_unbounded_potential_ge` — `qgR2Mode_potential_ge`
//!    (the potential part is ≥ `−M⁴/(16α)` uniformly in the mode, the
//!    regularization at the level of the operator) and
//!    `qgModeHamiltonian_not_bounded` (the symbol is real — the multiplication
//!    operator's symmetry face — and unbounded as the mode grows).
//!
//! 7. `qg_stone_flow_group_and_norm` — `IsStoneFlow`: `U(t)c = e^{−itλ_k}c_k`
//!    is a one-parameter group of norm-preserving operators on `ℓ²(ℕ)`.
//!
//! 8. `qg_stone_flow_generator` — the group solves the Schrödinger equation:
//!    `d/dt (U(t)c)_k = −i λ_k (U(t)c)_k` by central differences.
//!
//! 9. `qg_deficiency_resolvent_bound` — the finite-`ℓ²` shadow of
//!    `qgR2Mode_deficiencyTrivialAt`: for `Im z ≠ 0` the resolvent estimate
//!    `‖(H − z)⁻¹‖ ≤ 1/|Im z|` holds for the truncated symbol (equivalently
//!    `|λ_k − z| ≥ |Im z|` for every mode) — the analytic row (deficiency
//!    trivial at every non-real `z`) is proved at the mode level in Lean and
//!    stays open for the continuum.
//!
//! All evaluations are f64 to the exact-arithmetic tolerance class of the
//! guide §6 (1e-9 … 1e-12 for algebra, 1e-6 … 1e-9 for finite differences and
//! quadrature).  No SIRK solve here — pure algebra/analysis, in the style of
//! `transformed_esa_charts_validation.rs`.

use num_complex::Complex64;

// ── shared helpers ─────────────────────────────────────────────────────────

/// Trapezoid rule over the half line `(0, ∞)` through the tan map
/// `y = tan θ`, `θ ∈ (0, π/2)`:
/// `∫₀^∞ h(y) dy = ∫₀^{π/2} h(tan θ) sec²θ dθ`.  Tabulated on a uniform
/// θ-grid including the endpoints, where the transformed integrand is taken
/// by its continuous extension (the callers' kernels extend continuously
/// there).  This is the honest way to integrate algebraic tails — a plain
/// truncation leaves an O(L^{−p}) error.
fn trapz_halfline_compactified(h: impl Fn(f64) -> f64, n: usize) -> f64 {
    assert!(n >= 3);
    let hi = std::f64::consts::FRAC_PI_2;
    let dth = hi / (n - 1) as f64;
    let mut s = 0.0;
    for i in 0..n {
        let th = i as f64 * dth;
        let (snt, cst) = th.sin_cos();
        if cst.abs() < 1e-300 {
            continue; // h(tan θ) sec²θ is taken 0 there by the callers' guards
        }
        let y = snt / cst;
        let w = h(y) / (cst * cst);
        s += if i == 0 || i == n - 1 { 0.5 * w } else { w };
    }
    s * dth
}

// ── A1. the Jacobian and the half-density factor ───────────────────────────

#[test]
fn hd_jacobian_and_half_density_factor() {
    // `qgJacobian y = 2y` is `de/dy` at `e = y²` (central differences), and
    // the weight `2y` of `qgSrcMeasure` is the square of the half-density
    // factor `qgHalfDensity y = √(2y)`
    // (`qgSrcMeasure_density_eq_halfDensity_sq`).
    for y in [0.05f64, 0.4, 1.0, 2.5, 7.3] {
        let e = |t: f64| t * t;
        let h = 1e-6;
        let dedy_fd = (e(y + h) - e(y - h)) / (2.0 * h);
        let jac = 2.0 * y; // qgJacobian
        assert!(
            (dedy_fd - jac).abs() < 1e-6 * (1.0 + jac),
            "de/dy = {dedy_fd} != 2y = {jac} at y={y}"
        );
        // the half-density factor squared is the Jacobian
        let half = (jac).sqrt(); // qgHalfDensity y = sqrt (2y)
        assert!(
            (half * half - jac).abs() < 1e-12 * (1.0 + jac),
            "half-density square mismatch at y={y}"
        );
    }
}

// ── A2. the half-density unitary preserves the norm ────────────────────────

#[test]
fn hd_unitary_norm_preservation() {
    // `halfDensityUnitary_norm`: ∫₀^∞ |g(e)|² de = ∫₀^∞ |g(y²)|² 2y dy.
    // The two sides live on DIFFERENT measures and are integrated by
    // different routes; a missing Jacobian factor 2y, or a mistyped power in
    // the point map, separates them.

    // (a) g(e) = e^{−e}: both sides 1/2 (analytic).
    let g2_a = |e: f64| (-2.0 * e).exp();
    let lhs_a = trapz_halfline_compactified(g2_a, 200_001);
    let rhs_a = trapz_halfline_compactified(|y: f64| g2_a(y * y) * 2.0 * y, 200_001);
    assert!((lhs_a - 0.5).abs() < 1e-9, "LHS (a): {lhs_a}");
    assert!((rhs_a - 0.5).abs() < 1e-9, "RHS (a): {rhs_a}");

    // (b) g(e) = (1+e)^{−3/2}: |g|² = (1+e)^{−3}, both sides 1/2
    // (analytic).  Algebraic tails on both sides — compactified.
    let g2_b = |e: f64| (1.0 + e).powf(-3.0);
    let lhs_b = trapz_halfline_compactified(g2_b, 200_001);
    let rhs_b = trapz_halfline_compactified(|y: f64| g2_b(y * y) * 2.0 * y, 200_001);
    assert!((lhs_b - 0.5).abs() < 1e-6, "LHS (b): {lhs_b}");
    assert!((rhs_b - 0.5).abs() < 1e-6, "RHS (b): {rhs_b}");

    // (c) oscillatory: g(e) = e^{−e/2} cos(3e).  No closed form needed —
    // the two quadratures must agree with each other.
    let g2_c = |e: f64| (-e).exp() * (3.0 * e).cos() * (3.0 * e).cos();
    let lhs_c = trapz_halfline_compactified(g2_c, 400_001);
    let rhs_c = trapz_halfline_compactified(|y: f64| g2_c(y * y) * 2.0 * y, 400_001);
    assert!(
        (lhs_c - rhs_c).abs() < 1e-6 * (1.0 + lhs_c),
        "chart-norm mismatch: {lhs_c} vs {rhs_c}"
    );
}

// ── A3. the measure pushforward (measurePreserving_qgSquare) ───────────────

#[test]
fn hd_measure_pushforward() {
    // ∫₀^∞ f(y²) 2y dy = ∫₀^∞ f(e) de for an independent f (not a modulus
    // square): pushing `2y dy` forward along `y ↦ y²` gives Lebesgue `de`.
    // (a) f(e) = e^{−e} (1 + e): analytic value 1 + 1 = 2.
    let f_a = |e: f64| (-e).exp() * (1.0 + e);
    let lhs_a = trapz_halfline_compactified(|y: f64| f_a(y * y) * 2.0 * y, 200_001);
    let rhs_a = trapz_halfline_compactified(f_a, 200_001);
    assert!((lhs_a - 2.0).abs() < 1e-9, "push (a) LHS: {lhs_a}");
    assert!((rhs_a - 2.0).abs() < 1e-9, "push (a) RHS: {rhs_a}");

    // (b) f(e) = (1+e)^{−3}: analytic value 1/2.  Algebraic tail —
    // compactified both sides.
    let f_b = |e: f64| (1.0 + e).powf(-3.0);
    let lhs_b = trapz_halfline_compactified(|y: f64| f_b(y * y) * 2.0 * y, 200_001);
    let rhs_b = trapz_halfline_compactified(f_b, 200_001);
    assert!((lhs_b - 0.5).abs() < 1e-6, "push (b) LHS: {lhs_b}");
    assert!((rhs_b - 0.5).abs() < 1e-6, "push (b) RHS: {rhs_b}");
}

// ── A4. multiplication operators intertwine (halfDensityUnitary_intertwines) ─

#[test]
fn hd_intertwines_multiplication() {
    // The transfer face of `halfDensityUnitary_intertwines`: a multiplication
    // operator's expectations transport through the unitary.  With
    // (W g)(y) = g(y²) the two sides live on different measures and use
    // different integrands:
    //   LHS = ∫ V(y²) |g(y²)|² 2y dy      (qgSrcMeasure side)
    //   RHS = ∫ V(e) |g(e)|² de           (Lebesgue side).
    // Take V(e) = (1/16) e^{−e} and g(e) = e^{−e/2} (|g|² = e^{−e}), so the
    // RHS is ∫ (1/16) e^{−2e} de = 1/32 analytically.
    let v = |e: f64| (-e).exp() / 16.0;
    let g2 = |e: f64| (-e).exp();
    let lhs = trapz_halfline_compactified(|y: f64| v(y * y) * g2(y * y) * 2.0 * y, 200_001);
    let rhs = trapz_halfline_compactified(|e: f64| v(e) * g2(e), 200_001);
    assert!((rhs - 1.0 / 32.0).abs() < 1e-9, "RHS: {rhs}");
    assert!(
        (lhs - rhs).abs() < 1e-9 * (1.0 + rhs),
        "intertwining mismatch: {lhs} vs {rhs}"
    );

    // A second potential with a slower decay: V(e) = 1/(16 (1+e)²),
    // g(e) = (1+e)^{−1}: RHS = ∫ (1/16)(1+e)^{−4} de = 1/48.
    let v2 = |e: f64| 1.0 / (16.0 * (1.0 + e) * (1.0 + e));
    let g2b = |e: f64| (1.0 + e).powf(-2.0);
    let lhs2 = trapz_halfline_compactified(|y: f64| v2(y * y) * g2b(y * y) * 2.0 * y, 200_001);
    let rhs2 = trapz_halfline_compactified(|e: f64| v2(e) * g2b(e), 200_001);
    assert!((rhs2 - 1.0 / 48.0).abs() < 1e-6, "RHS2: {rhs2}");
    assert!(
        (lhs2 - rhs2).abs() < 1e-6 * (1.0 + rhs2),
        "intertwining-2 mismatch: {lhs2} vs {rhs2}"
    );
}

// ── B5. the conformal-mode potential: completed square and bound ───────────

#[test]
fn qg_conf_v_completed_square_and_bound() {
    // confV M alpha Rc = −(M²/2) Rc + α Rc²  (confV)
    //                 = α (Rc − M²/(4α))² − M⁴/(16α)  (confV_completed_square)
    // and hence ≥ −M⁴/(16α) for α > 0 (confV_ge), with equality at
    // Rc = M²/(4α).  For α = 0 the potential runs to −∞
    // (confV_zero_alpha_tendsto_atBot).
    let conf_v = |m: f64, alpha: f64, rc: f64| -(m * m / 2.0) * rc + alpha * rc * rc;
    let mut seed = 0x4528_21E6_38D0_1377u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 8.0 - 4.0
    };
    for _ in 0..200 {
        let m = rng().abs() + 0.2;
        let alpha = rng().abs() + 0.05;
        let rc = rng() * 3.0;
        let lhs = conf_v(m, alpha, rc);
        let rhs = alpha * (rc - m * m / (4.0 * alpha)).powi(2) - m.powi(4) / (16.0 * alpha);
        assert!(
            (lhs - rhs).abs() < 1e-12 * (1.0 + lhs.abs()),
            "completed square failed: {lhs} vs {rhs}"
        );
        // the lower bound −M⁴/(16α), attained at the vertex
        let floor = -m.powi(4) / (16.0 * alpha);
        assert!(
            lhs >= floor - 1e-12 * (1.0 + floor.abs()),
            "bound violated: {lhs} < {floor}"
        );
        let at_vertex = conf_v(m, alpha, m * m / (4.0 * alpha));
        assert!(
            (at_vertex - floor).abs() < 1e-12 * (1.0 + floor.abs()),
            "vertex value: {at_vertex} vs {floor}"
        );
    }
    // α = 0: the pure-GR conformal potential is linear, running to −∞ —
    // exactly what the αR² term removes.
    let m = 1.7f64;
    let lo = conf_v(m, 0.0, 10.0);
    let hi = conf_v(m, 0.0, 100.0);
    assert!(hi < lo && hi < -100.0, "α = 0 should run to −∞: {lo}, {hi}");
}

// ── B6. the mode symbol: real, potential bounded below, unbounded operator ─

#[test]
fn qg_mode_symbol_real_unbounded_potential_ge() {
    // qgModeSymbol: λ_k = (1/16) a_k² − (1/24) b_k² + V_k with
    // V_k = confV M alpha (Rc k) = qgR2ModePotential.
    // (i) V_k ≥ −M⁴/(16α) uniformly in the mode (qgR2Mode_potential_ge);
    // (ii) λ_k is real — the multiplication-operator symmetry face
    //      (qgR2Mode_symmetric / mulSymbolOp_symmetric);
    // (iii) the operator is unbounded (qgModeHamiltonian_not_bounded):
    //      |λ_k| grows past every bound along a growing mode family.
    let m = 1.3f64;
    let alpha = 0.9f64;
    let floor = -m.powi(4) / (16.0 * alpha);
    let mut seed = 0x082E_FA98_EC4E_6C89u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    let mut lam_max = 0.0f64;
    for k in 0..64usize {
        let a_k = rng();
        let b_k = rng();
        let rc_k = rng();
        let v_k = -(m * m / 2.0) * rc_k + alpha * rc_k * rc_k;
        assert!(
            v_k >= floor - 1e-12 * (1.0 + floor.abs()),
            "V_k below floor at k={k}"
        );
        let lam = (1.0 / 16.0) * a_k * a_k - (1.0 / 24.0) * b_k * b_k + v_k;
        assert!(lam.is_finite(), "mode symbol must be a real number");
        lam_max = lam_max.max(lam.abs());
    }
    // unboundedness: the growing family a_k = k makes |λ_k| → ∞
    let lam_at = |k: f64| (1.0 / 16.0) * k * k;
    let mut prev = 0.0f64;
    for k in [10.0f64, 30.0, 100.0, 300.0] {
        let v = lam_at(k);
        assert!(v > prev, "growth expected");
        prev = v;
    }
    assert!(lam_at(300.0) > 1.0e3, "operator should be unbounded");
}

// ── B7. the Stone flow: group law and norm conservation ────────────────────

#[test]
fn qg_stone_flow_group_and_norm() {
    // `qgR2_stone_flow`'s IsStoneFlow content at the mode level: the symbol
    // λ_k being real, U(t)c = (e^{−itλ_k} c_k) is a one-parameter group of
    // isometries on ℓ²(ℕ) — unitary, U(0) = id, U(t)U(s) = U(t+s).
    let m = 1.1f64;
    let alpha = 0.7f64;
    let n = 64usize;
    let mut seed = 0xA409_3822_299F_31D0u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    // a genuinely ℓ² state (geometric decay), the mode symbol, and the
    // Starobinsky conformal-mode potential per mode.
    let c: Vec<Complex64> = (0..n)
        .map(|k| Complex64::new(rng(), rng()) * 0.9f64.powi(k as i32))
        .collect();
    let lam: Vec<f64> = (0..n)
        .map(|_k| {
            let a = rng();
            let b = rng();
            let rc = rng();
            (1.0 / 16.0) * a * a - (1.0 / 24.0) * b * b + (-(m * m / 2.0) * rc + alpha * rc * rc)
        })
        .collect();
    let u = |t: f64, c: &[Complex64]| -> Vec<Complex64> {
        c.iter()
            .zip(&lam)
            .map(|(ck, l)| {
                let (s, co) = (t * l).sin_cos();
                *ck * Complex64::new(co, -s)
            })
            .collect()
    };
    let norm = |v: &[Complex64]| -> f64 { v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() };
    let n0 = norm(&c);
    for t in [0.0f64, 0.3, 2.0, -7.5, 25.0] {
        let ut = u(t, &c);
        assert!(
            (norm(&ut) - n0).abs() < 1e-12 * (1.0 + n0),
            "norm drift at t={t}: {} vs {n0}",
            norm(&ut)
        );
    }
    // U(0) = id, and the group law U(t)(U(s) c) = U(t+s) c
    let u0 = u(0.0, &c);
    for k in 0..n {
        assert!((u0[k] - c[k]).norm() < 1e-13, "U(0) != id");
    }
    for (t, s) in [(0.7f64, -1.3f64), (2.5, 2.5), (0.1, 4.0)] {
        let lhs = u(t, &u(s, &c));
        let rhs = u(t + s, &c);
        for k in 0..n {
            assert!(
                (lhs[k] - rhs[k]).norm() < 1e-12 * (1.0 + rhs[k].norm()),
                "group law failed at k={k}, t={t}, s={s}"
            );
        }
    }
}

// ── B8. the flow solves the Schrödinger equation of the mode Hamiltonian ───

#[test]
fn qg_stone_flow_generator() {
    // IsStoneFlow's derivative face: d/dt (e^{−itH} c)_k = −i λ_k (e^{−itH}c)_k,
    // central-differenced in time.
    let n = 8usize;
    let mut seed = 0x082E_FA98_EC4E_6C89u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let c: Vec<Complex64> = (0..n).map(|_| Complex64::new(rng(), rng())).collect();
    let lam: Vec<f64> = (0..n).map(|_| rng() * 4.0).collect();
    let u = |t: f64| -> Vec<Complex64> {
        c.iter()
            .zip(&lam)
            .map(|(ck, l)| {
                let (s, co) = (t * l).sin_cos();
                *ck * Complex64::new(co, -s)
            })
            .collect()
    };
    for t in [0.0f64, 0.4, -1.7] {
        let h = 1e-6;
        let up = u(t + h);
        let um = u(t - h);
        let ut = u(t);
        for k in 0..n {
            let fd = (up[k] - um[k]) / (2.0 * h);
            let rhs = Complex64::new(0.0, -1.0) * lam[k] * ut[k];
            assert!(
                (fd - rhs).norm() < 1e-6 * (1.0 + rhs.norm()),
                "generator mismatch at t={t}, k={k}: {fd} vs {rhs}"
            );
        }
    }
}

// ── B9. the deficiency shadow: resolvent bound off the real axis ───────────

#[test]
fn qg_deficiency_resolvent_bound() {
    // `qgR2Mode_deficiencyTrivialAt` at the mode level: ker(H* − z) = 0 for
    // Im z ≠ 0.  The finite-ℓ² shadow is the resolvent estimate
    // ‖(H − z)⁻¹‖ ≤ 1/|Im z| for the truncated symbol — equivalently
    // |λ_k − z| ≥ |Im z| for every mode.  Checked by solving (Λ − z)x = c
    // mode by mode and comparing the norms.  (For the continuum this row is
    // the still-open analytic input; here it is the mode-level face that
    // `strichartz_finiteSpeed_satisfiable` discretizes.)
    let n = 32usize;
    let mut seed = 0xDEAD_BEEF_CAFE_F00Du64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
    };
    let lam: Vec<f64> = (0..n).map(|k| (k as f64) * 0.7 - 3.0 + rng()).collect();
    let c: Vec<Complex64> = (0..n).map(|_| Complex64::new(rng(), rng())).collect();
    let nc = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    for &(re, im) in &[(0.0f64, 0.5f64), (1.3, 1.0), (-4.0, 3.0), (2.0, -0.25)] {
        let z = Complex64::new(re, im);
        // x_k = c_k / (λ_k − z) solves (H − z) x = c
        let x: Vec<Complex64> = c
            .iter()
            .zip(&lam)
            .map(|(ck, l)| *ck / (Complex64::new(*l, 0.0) - z))
            .collect();
        let nx = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            nx <= nc / im.abs() * (1.0 + 1e-9),
            "resolvent bound failed at z={z}: ‖x‖={nx} > ‖c‖/|Im z| = {}",
            nc / im.abs()
        );
        // equivalently: every |λ_k − z| ≥ |Im z|
        for (k, l) in lam.iter().enumerate() {
            let d = (Complex64::new(*l, 0.0) - z).norm();
            assert!(d >= im.abs() - 1e-12, "|λ_k − z| < |Im z| at k={k}, z={z}");
        }
    }
}
