//! Numerical validation of the TRANSFORMED-ESA-CHARTS module.
//!
//! Companion to the symbolic module `docs/transformed_esa_charts.cdb`
//! (CHECK 1–9) and to `docs/VERIFY_TRANSFORMED_ESA_CHARTS.md`.  Every test
//! is a numerical identity of the module's certified content, evaluated in
//! f64 to the exact-arithmetic tolerance class of the guide §6
//! (1e-9 … 1e-12 for algebra, 1e-6 … 1e-9 for finite differences and
//! quadrature).  No SIRK solve here — pure algebra/analysis of the three
//! variable systems in which the program closes essential
//! self-adjointness of the TRANSFORMED operator, in the style of
//! `ns_compressible_lagrangian_validation.rs`.
//!
//! The doctrine the module exists for: a classical singularity does not
//! decide whether an operator is essentially self-adjoint — the variables
//! do.  Each test below is one face of "change variables until the flow is
//! complete, then run the ESA argument on the transformed operator".
//!
//! The module's content, test by test:
//!
//! 1. `xf_classical_solution_solves_ode` — CHECK 1: the classical solution
//!    `x(t) = x0/(1 − t x0)` satisfies `ẋ = x²`, finite-differenced against
//!    `x(t)²`, before and near the blow-up time `t = 1/x0`.
//!
//! 2. `xf_chain_rule_completes_flow` — CHECK 2: in the reciprocal chart
//!    `w = −1/x` the chain rule gives `ẇ = (dw/dx)·ẋ = (1/x²)·x² = 1`:
//!    the chart velocity is exactly 1 along the whole classical solution —
//!    the flow is COMPLETE in `w`.
//!
//! 3. `xf_chart_of_flow_is_translation` — CHECK 3: `w(t) = −1/x(t)` equals
//!    `w0 + t` — the blow-up trajectory is a straight line in the chart
//!    (`odeKoop_chartW`'s translation group).
//!
//! 4. `xf_weyl_commutator_correction` — CHECK 4: `[x², p] = 2ix` for
//!    `p = −i d/dx`, applied to a smooth test function and evaluated by
//!    central finite differences.
//!
//! 5. `xf_weyl_sum_equals_h` — CHECK 5: the Weyl sum
//!    `½(p x² + x² p)` equals `H = x²p − ix` — the book's two spellings
//!    of the ODE Hamiltonian agree as operators (equivalently, cleared
//!    by 2: `p x² + x² p = 2H`).
//!
//! 6. `xf_chartw_norm_preservation` — CHECK 6: the chartW kernel `1/x²`
//!    times the Jacobian `x²` is 1 — `∫|chartW ψ|² dx = ∫|ψ|² dw`
//!    (`chartW_lintegral_normSq`), checked by quadrature on both sides
//!    independently (the `x`-integral through a tan-compactification, its
//!    tails decay only like `1/x²`).
//!
//! 7. `xf_lagrangian_completed_square` — CHECK 7: the Lagrangian one-parcel
//!    operator completes to the shifted oscillator
//!    `½P² + νQ² + fP = ½(P+f)² + νQ² − ½f²` for random states.
//!
//! 8. `xf_densitized_inverse_identity` — CHECK 8: `1/e = 4(dy/de)²` at
//!    `e = y²` — the absorption identity
//!    `inv_eq_four_mul_deriv_densY_sq` (the honest factor 4).
//!
//! 9. `xf_kinetic_absorption` — CHECK 9: `(1/16e)S² = (1/16)(S/y)²` at
//!    `e = y²` — the flat principal part of the densitized operator.
//!
//! 10. `xf_quantized_evolution_conserves_norm` — the ODE chapter's
//!     headline, past the classical blow-up: the operator family
//!     `(e^{−iHt}ψ)(x) = (1+tx)⁻¹ ψ(x/(1+tx))` conserves `∫|ψ|² dx` for
//!     `t` beyond `1/x0`, the measure-level resolution of the singularity
//!     (`odeKoop_lintegral_normSq`).

use num_complex::Complex64;

const PI: f64 = std::f64::consts::PI;

// ── shared helpers ─────────────────────────────────────────────────────────

/// Composite trapezoid rule on a uniform grid.
fn trapz(f: &[f64], dx: f64) -> f64 {
    let n = f.len();
    assert!(n >= 2);
    let mut s = 0.5 * (f[0] + f[n - 1]);
    for v in &f[1..n - 1] {
        s += v;
    }
    s * dx
}

/// Trapezoid rule of `g` over the compactified line: `x = tan θ` maps
/// `θ ∈ (−π/2, π/2)` onto `x ∈ (−∞, ∞)`, so
/// `∫ g(x) dx = ∫ g(tan θ) sec²θ dθ`.  The transformed integrand is
/// tabulated on a uniform θ-grid INCLUDING the endpoints, where it is
/// understood by its continuous extension (the callers' kernels extend
/// continuously there).  This is the honest way to integrate the
/// slowly-decaying tails of the norm integrands below — a plain
/// truncation at |x| ≤ L leaves an O(1/L) tail for a 1/x² kernel.
fn trapz_compactified(g: impl Fn(f64) -> f64, n: usize) -> f64 {
    assert!(n >= 3);
    let lo = -std::f64::consts::FRAC_PI_2;
    let hi = std::f64::consts::FRAC_PI_2;
    let h = (hi - lo) / (n - 1) as f64;
    let mut s = 0.0;
    for i in 0..n {
        let th = lo + i as f64 * h;
        let (snt, cst) = th.sin_cos();
        if cst.abs() < 1e-300 {
            continue; // g(tan θ) sec²θ is taken 0 there by the callers' guards
        }
        let x = snt / cst;
        let w = g(x) / (cst * cst);
        s += if i == 0 || i == n - 1 { 0.5 * w } else { w };
    }
    s * h
}

// ── 1. classical solution solves ẋ = x² ────────────────────────────────────

#[test]
fn xf_classical_solution_solves_ode() {
    // CHECK 1.  x(t) = x0/(1 − t x0); the quotient-rule derivative must
    // equal x(t)².  Sampled at several t before the blow-up t = 1/x0.
    for x0 in [0.5f64, 1.0, 2.0, -1.5] {
        let tb = 1.0 / x0;
        for frac in [0.0f64, 0.25, 0.5, 0.9] {
            let t = frac * tb;
            let x = |s: f64| x0 / (1.0 - s * x0);
            let h = 1e-6;
            let fd = (x(t + h) - x(t - h)) / (2.0 * h);
            let rhs = x(t) * x(t);
            assert!(
                (fd - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "xdot != x^2 at t={t}: {fd} vs {rhs}"
            );
        }
    }
}

// ── 2. chain rule completes the flow: ẇ = 1 ────────────────────────────────

#[test]
fn xf_chain_rule_completes_flow() {
    // CHECK 2.  w = −1/x along the classical solution: dw/dt must be 1 at
    // every time before the blow-up — the chart turns the singular vector
    // field into the constant field (complete flow).
    for x0 in [0.5f64, 1.0, 3.0] {
        let tb = 1.0 / x0;
        let w = |t: f64| -1.0 / (x0 / (1.0 - t * x0));
        for frac in [0.1f64, 0.4, 0.7, 0.99] {
            let t = frac * tb;
            let h = 1e-6;
            let fd = (w(t + h) - w(t - h)) / (2.0 * h);
            assert!((fd - 1.0).abs() < 1e-6, "wdot = {fd}, expected 1");
        }
    }
}

// ── 3. chart of the flow is a translation ──────────────────────────────────

#[test]
fn xf_chart_of_flow_is_translation() {
    // CHECK 3.  w(t) = w0 + t exactly (to round-off) for the whole
    // classical trajectory, including times past the blow-up of x(t).
    for x0 in [0.5f64, 1.0, -2.0] {
        let w0 = -1.0 / x0;
        for t in [0.0f64, 0.3, 1.7, 5.0] {
            if (1.0 - t * x0).abs() < 1e-9 {
                continue;
            }
            let w_t = -1.0 / (x0 / (1.0 - t * x0));
            let target = w0 + t;
            assert!(
                (w_t - target).abs() < 1e-12 * (1.0 + target.abs()),
                "w(t) = {w_t} != w0+t = {target}"
            );
        }
    }
}

// ── 4. Weyl commutator [x², p] = 2 i x ─────────────────────────────────────

#[test]
fn xf_weyl_commutator_correction() {
    // CHECK 4.  p = −i d/dx.  (x²p − p x²)ψ must equal 2 i x ψ.  Both
    // differential operators are applied by central differences to a
    // Gaussian test function.
    let dx = 1e-4;
    let n = 200_001;
    let lo = -5.0;
    let xs: Vec<f64> = (0..n).map(|i| lo + i as f64 * dx).collect();
    let psi = |x: f64| Complex64::new((-0.5 * x * x).exp(), 0.0);
    // −i ψ' by central differences
    let mut ppsi = vec![Complex64::new(0.0, 0.0); n];
    for i in 1..n - 1 {
        let d = (psi(xs[i + 1]) - psi(xs[i - 1])) / (2.0 * dx);
        ppsi[i] = Complex64::new(0.0, -1.0) * d;
    }
    // x² p ψ
    let x2p: Vec<Complex64> = xs.iter().zip(&ppsi).map(|(&x, &v)| v * (x * x)).collect();
    // p (x² ψ) = −i d/dx (x² ψ)
    let x2psi: Vec<Complex64> = xs.iter().map(|&x| psi(x) * (x * x)).collect();
    let mut px2 = vec![Complex64::new(0.0, 0.0); n];
    for i in 1..n - 1 {
        let d = (x2psi[i + 1] - x2psi[i - 1]) / (2.0 * dx);
        px2[i] = Complex64::new(0.0, -1.0) * d;
    }
    // compare (x²p − p x²)ψ against 2 i x ψ on the interior
    for i in (2..n - 2).step_by(1000) {
        let lhs = x2p[i] - px2[i];
        let rhs = Complex64::new(0.0, 2.0 * xs[i]) * psi(xs[i]);
        assert!(
            (lhs - rhs).norm() < 1e-4,
            "commutator mismatch at x={}: {lhs} vs {rhs}",
            xs[i]
        );
    }
}

// ── 5. Weyl sum ½(p x² + x² p) = 2H ───────────────────────────────────────

#[test]
fn xf_weyl_sum_equals_h() {
    // CHECK 5.  H = x²p − ix.  The Weyl sum ½(p x² + x² p) must act as
    // H on a test function: the symmetrized and corrected spellings of
    // the ODE Hamiltonian coincide (from [x², p] = 2ix one gets
    // p x² = x²p − 2ix, hence ½(p x² + x² p) = x²p − ix).
    let dx = 1e-4;
    let n = 200_001;
    let lo = -5.0;
    let xs: Vec<f64> = (0..n).map(|i| lo + i as f64 * dx).collect();
    let psi = |x: f64| Complex64::new((-0.4 * x * x + 0.3 * x).exp(), 0.0);
    let mut ppsi = vec![Complex64::new(0.0, 0.0); n];
    for i in 1..n - 1 {
        let d = (psi(xs[i + 1]) - psi(xs[i - 1])) / (2.0 * dx);
        ppsi[i] = Complex64::new(0.0, -1.0) * d;
    }
    let x2psi: Vec<Complex64> = xs.iter().map(|&x| psi(x) * (x * x)).collect();
    let mut px2 = vec![Complex64::new(0.0, 0.0); n];
    for i in 1..n - 1 {
        let d = (x2psi[i + 1] - x2psi[i - 1]) / (2.0 * dx);
        px2[i] = Complex64::new(0.0, -1.0) * d;
    }
    for i in (2..n - 2).step_by(1000) {
        let x = xs[i];
        // ½(p x² + x² p) ψ
        let weyl = 0.5 * (px2[i] + x * x * ppsi[i]);
        // H ψ = x² p ψ − i x ψ
        let hpsi = x * x * ppsi[i] - Complex64::new(0.0, x) * psi(x);
        assert!(
            (weyl - hpsi).norm() < 1e-4,
            "Weyl sum mismatch at x={x}: {weyl} vs {hpsi}"
        );
    }
}

// ── 6. chartW norm preservation ────────────────────────────────────────────

#[test]
fn xf_chartw_norm_preservation() {
    // CHECK 6.  (chartW ψ)(x) = x⁻¹ ψ(−1/x).  Both norms are integrated
    // independently:
    //   LHS = ∫ x⁻² |ψ(−1/x)|² dx   (the kernel 1/x² of the norm integrand)
    //   RHS = ∫ |ψ(w)|² dw          (the flat measure in the chart)
    // The identity kernel × Jacobian = 1 is what makes them equal; a
    // mistyped power of x in either integrand separates them.  The x-side
    // tails decay only like 1/x², so it is integrated through the tan
    // compactification (its integrand e^{−cot²θ}/sin²θ extends smoothly
    // by 0 at θ = 0 and by 1 at the endpoints).
    let psi2 = |w: f64| (-w * w).exp(); // |ψ|² for a real Gaussian
    // RHS: ∫ e^{−w²} dw over the real line = √π (Gaussian tails vanish).
    let dw = 1e-3;
    let nw = 400_001;
    let wlo = -20.0;
    let ws: Vec<f64> = (0..nw).map(|i| wlo + i as f64 * dw).collect();
    let rhs = trapz(&ws.iter().map(|&w| psi2(w)).collect::<Vec<_>>(), dw);
    // LHS: ∫ x⁻² e^{−1/x²} dx over the real line (continuous by 0 at x = 0).
    let lhs = trapz_compactified(
        |x: f64| {
            if x.abs() < 1e-12 {
                0.0
            } else {
                psi2(-1.0 / x) / (x * x)
            }
        },
        200_001,
    );
    assert!(
        (lhs - rhs).abs() < 1e-6 * (1.0 + rhs),
        "chartW norm mismatch: {lhs} vs {rhs}"
    );
    // sanity: the common value is √π
    assert!((rhs - PI.sqrt()).abs() < 1e-6, "Gaussian norm: {rhs}");
    assert!((lhs - PI.sqrt()).abs() < 1e-6, "chartW side: {lhs}");
}

// ── 7. Lagrangian completed square ─────────────────────────────────────────

#[test]
fn xf_lagrangian_completed_square() {
    // CHECK 7.  ½P² + νQ² + fP = ½(P+f)² + νQ² − ½f² on random states:
    // the transformed one-parcel operator is a shifted oscillator (a
    // complete flow) — the Lagrangian instance of the doctrine.
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
    };
    for _ in 0..64 {
        let p = rng();
        let q = rng();
        let f = rng();
        let nu = rng().abs() + 0.1;
        let lhs = 0.5 * p * p + nu * q * q + f * p;
        let rhs = 0.5 * (p + f) * (p + f) + nu * q * q - 0.5 * f * f;
        assert!(
            (lhs - rhs).abs() < 1e-12 * (1.0 + lhs.abs()),
            "{lhs} vs {rhs}"
        );
    }
}

// ── 8. densitized inverse identity 1/e = 4 (dy/de)² ────────────────────────

#[test]
fn xf_densitized_inverse_identity() {
    // CHECK 8.  y = √e, dy/de = 1/(2√e) = 1/(2y), so 4(dy/de)² = 1/y² = 1/e.
    for y in [0.05f64, 0.5, 1.0, 2.7, 13.0] {
        let e = y * y;
        let lhs = 1.0 / e;
        let dyde = 1.0 / (2.0 * y);
        let rhs = 4.0 * dyde * dyde;
        assert!(
            (lhs - rhs).abs() < 1e-12 * (1.0 + lhs),
            "at y={y}: {lhs} vs {rhs}"
        );
    }
}

// ── 9. kinetic absorption (1/16e) S² = (1/16) (S/y)² ───────────────────────

#[test]
fn xf_kinetic_absorption() {
    // CHECK 9.  The flat principal part: at e = y² the singular coefficient
    // 1/(16e) equals (1/16)(1/y)².
    let mut seed = 0x1319_8A2E_0370_7344u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 8.0 - 4.0
    };
    for _ in 0..64 {
        let y = rng().abs() + 0.05;
        let s = rng();
        let e = y * y;
        let physical = s * s / (16.0 * e);
        let densitized = (1.0 / 16.0) * (s / y) * (s / y);
        assert!(
            (physical - densitized).abs() < 1e-12 * (1.0 + physical.abs()),
            "at y={y}: {physical} vs {densitized}"
        );
    }
}

// ── 10. the quantized evolution conserves the norm past the blow-up ────────

#[test]
fn xf_quantized_evolution_conserves_norm() {
    // The ODE chapter's headline (odeKoop_lintegral_normSq): the operator
    // family (e^{−iHt}ψ)(x) = (1+tx)⁻¹ ψ(x/(1+tx)) conserves ∫|ψ|² dx for
    // EVERY real t — including t past the classical blow-up 1/x0 of every
    // positive initial condition.  The evolved norm integrand
    // ψ₂(x/(1+tx))/(1+tx)² decays in x only like 1/x² (it tends to
    // ψ₂(1/t)/(t x)² at both ends), so the x-integral goes through the tan
    // compactification.  In the θ variable the integrand is
    // ψ₂(sin θ/(cos θ + t sin θ)) / (cos θ + t sin θ)² — bounded on the
    // closed interval (endpoints → ψ₂(1/t)/t²), and flat 0 at the chart
    // pole θ* where cos θ + t sin θ = 0 (the Gaussian beats the pole).
    let psi2 = |x: f64| (-x * x).exp();
    let base = PI.sqrt(); // ∫ e^{−x²} dx
    for t in [0.0f64, 0.5, 1.0, 2.5, -3.0] {
        let norm = trapz_compactified(
            |x: f64| {
                let den = 1.0 + t * x;
                if den.abs() < 1e-12 {
                    0.0
                } else {
                    psi2(x / den) / (den * den)
                }
            },
            400_001,
        );
        assert!(
            (norm - base).abs() < 1e-5 * (1.0 + base),
            "norm drifted at t={t}: {norm} vs {base}"
        );
    }
}
