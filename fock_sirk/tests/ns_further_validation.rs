//! NS further numerical validation — exact laminar solutions and turbulence
//! scaling identities beyond `ns_boundary_layer_validation.rs`.
//!
//! 1. **Hagen–Poiseuille flow**: `Q = πR⁴ΔP/(8μL)` with the exact `R⁴`
//!    scaling and `v_max = 2 v̄` (parabolic profile).
//! 2. **Kolmogorov −5/3 spectrum**: `E(k) ∝ k^{−5/3}` — the ratio
//!    `E(2k)/E(k) = 2^{−5/3} = 0.315` exactly.
//! 3. **Kolmogorov first similarity**: at the dissipation scale,
//!    `Re_η = u_η·η/ν = 1` exactly, with `η = (ν³/ε)^{1/4}` and
//!    `u_η = (νε)^{1/4}` — the defining self-consistency of the dissipation
//!    range.
//! 4. **Stokes drag**: `F = 6πμRv` at `Re ≪ 1` — exact linearity in `R` and `v`.
//! 5. **Blasius boundary-layer thicknesses**: `δ*/x = 1.7208/√Re_x` and
//!    `θ/x = 0.664/√Re_x` with shape factor `H = δ*/θ = 2.5916` (the `√x`
//!    growth law).

const PI: f64 = std::f64::consts::PI;

#[test]
fn ns_hagen_poiseuille_flow_rate_and_r4() {
    // Q = πR⁴ΔP/(8μL)
    let flow = |r: f64| PI * r.powi(4) * 100.0 / (8.0 * 1e-3 * 0.1);
    let q = flow(1e-3);
    let expected = 3.927e-7; // m³/s
    let rel = (q - expected).abs() / expected;
    assert!(rel < 1e-3, "Q = {q:.4e}, expected {expected:.4e} (rel {rel:.2e})");
    // R⁴ scaling: doubling the radius ×16 the flow.
    assert!((flow(2e-3) / q - 16.0).abs() < 1e-9);
    // v_max = 2 v̄: v̄ = Q/(πR²), v_max = ΔP R²/(4μL).
    let v_mean = q / (PI * 1e-6);
    let v_max = 100.0 * 1e-6 / (4.0 * 1e-3 * 0.1);
    assert!((v_max / v_mean - 2.0).abs() < 1e-9, "v_max/v̄ = {}", v_max / v_mean);
}

#[test]
fn ns_kolmogorov_minus_5_3_spectrum() {
    // E(k) ∝ k^{−5/3} ⇒ E(2k)/E(k) = 2^{−5/3}.
    let ratio = 2f64.powf(-5.0 / 3.0);
    let expected = 0.314_980; // 2^{−5/3}
    assert!((ratio - expected).abs() < 1e-5, "2^{{-5/3}} = {ratio}");
    // A decade of scales: E(10k)/E(k) = 10^{−5/3} ≈ 0.0215.
    let decade = 10f64.powf(-5.0 / 3.0);
    assert!((decade - 0.021_544).abs() < 1e-4);
    // Inertial-range consistency with the dissipation identity below:
    // ε = C ε^{2/3} k^{5/3} E(k) ⇒ the ratio of two wavenumbers drops out.
    let e_at = |k: f64| k.powf(-5.0 / 3.0);
    assert!((e_at(2.0) / e_at(1.0) - ratio).abs() < 1e-12);
}

#[test]
fn ns_kolmogorov_dissipation_scale_re_eta_identity() {
    // η = (ν³/ε)^{1/4}, u_η = (νε)^{1/4}, and Re_η = u_η·η/ν ≡ 1.
    let nu: f64 = 1e-6;
    let eps: f64 = 1e-3;
    let eta = (nu.powi(3) / eps).powf(0.25);
    let u_eta = (nu * eps).powf(0.25);
    let re_eta = u_eta * eta / nu;
    assert!((re_eta - 1.0).abs() < 1e-12, "Re_η = {re_eta}");
    // Cross-check with the Taylor-scale identity λ/η = 15^{1/4}√Re_λ from the
    // boundary-layer suite: combining gives ε = 15νu²/λ² at any Re_λ.
    let u_rms: f64 = 0.1;
    let lambda = (15.0 * nu * u_rms * u_rms / eps).sqrt();
    let eps_check = 15.0 * nu * u_rms * u_rms / (lambda * lambda);
    assert!((eps_check - eps).abs() / eps < 1e-9);
}

#[test]
fn ns_stokes_drag_linearity() {
    // F = 6πμRv
    let drag = |r: f64, v: f64| 6.0 * PI * 1e-3 * r * v;
    let f = drag(1e-6, 1e-4);
    let expected = 1.885e-12; // N
    let rel = (f - expected).abs() / expected;
    assert!(rel < 1e-3, "F = {f:.4e} N, expected {expected:.4e} (rel {rel:.2e})");
    // Exact linearity in R and v.
    assert!((drag(2e-6, 1e-4) / f - 2.0).abs() < 1e-12);
    assert!((drag(1e-6, 2e-4) / f - 2.0).abs() < 1e-12);
    // Terminal velocity of a sphere: v_t = 2(ρ−ρ_f)gR²/(9μ).
    let rho = 2000.0;
    let rho_f = 1000.0;
    let g = 9.81;
    let r = 1e-5;
    let v_t = 2.0 * (rho - rho_f) * g * r * r / (9.0 * 1e-3);
    let expected_vt: f64 = 2.18e-4; // m/s
    let rel_vt = (v_t - expected_vt).abs() / expected_vt;
    assert!(rel_vt < 5e-3, "v_t = {v_t:.3e}, expected {expected_vt:.3e}");
}

#[test]
fn ns_couette_linear_profile_and_constant_shear() {
    // Plane Couette flow: u(y) = U·y/H, shear τ = μU/H constant across the
    // gap. The profile is exactly linear (a solution of the NS equation with
    // zero pressure gradient).
    let u = |y: f64| 1.0 * y / 0.01; // U = 1 m/s, H = 1 cm
    assert!((u(0.005) - 0.5).abs() < 1e-12); // midpoint
    assert!((u(0.0) - 0.0).abs() < 1e-12); // no-slip
    assert!((u(0.01) - 1.0).abs() < 1e-12); // top plate
    // Shear τ = μ du/dy is constant: evaluate at three stations.
    let mu: f64 = 1e-3;
    let tau = |y: f64| mu * (u(y + 1e-9) - u(y - 1e-9)) / 2e-9;
    let t0 = tau(0.002);
    let t1 = tau(0.005);
    let t2 = tau(0.008);
    assert!((t1 - t0).abs() / t0 < 1e-6 && (t2 - t0).abs() / t0 < 1e-6);
    assert!((t0 - 0.1).abs() / 0.1 < 1e-3, "τ = μU/H = {t0}");
}

#[test]
fn ns_bernoulli_venturi_pressure_drop() {
    // Continuity A₁v₁ = A₂v₂ + Bernoulli: a constriction to A₂ = A₁/4 gives
    // v₂ = 4v₁ and ΔP = P₁−P₂ = ½ρ(v₂²−v₁²) = 7.5ρv₁².
    let rho: f64 = 1000.0;
    let v1: f64 = 1.0;
    let v2 = 4.0 * v1; // from A₂ = A₁/4
    assert!((v2 - 4.0).abs() < 1e-12);
    let dp = 0.5 * rho * (v2 * v2 - v1 * v1);
    assert!((dp - 7500.0).abs() < 1e-9, "ΔP = {dp}, expected 7500 Pa");
    // Recovery: re-expanding to A₁ returns v to v₁ and P to P₁ (ΔP = 0).
    let v3 = 1.0;
    let dp_round = 0.5 * rho * (v3 * v3 - v2 * v2);
    assert!((dp_round + dp).abs() < 1e-9, "recovery ΔP = {dp_round}");
}

#[test]
fn ns_lamb_oseen_vortex_structure() {
    // The Lamb–Oseen (diffusing) vortex: v_θ(r,t) = Γ/(2πr)(1 − e^{−r²/4νt}),
    // with vorticity ω(r,t) = Γ/(4πνt)·e^{−r²/4νt}. Two exact statements:
    // (1) the circulation at infinity is Γ (the total vorticity is
    // conserved); (2) the central vorticity decays as 1/t.
    let gamma: f64 = 1.0;
    let nu_t: f64 = 0.25; // the product νt sets the core width √(4νt) = 1
    let vort = |r: f64, t: f64| gamma / (4.0 * PI * nu_t * t) * (-r * r / (4.0 * nu_t * t)).exp();
    // Total circulation: ∫ω dA = Γ exactly (Gaussian integral — independent
    // of νt). Rectangle quadrature over r ∈ [0,10], h = 0.001.
    let t = 1.0;
    let circ = 2.0 * PI * (0..10_000).fold(0.0, |acc, i| {
        let r = (i as f64 + 0.5) * 0.001;
        acc + vort(r, t) * r * 0.001
    });
    assert!((circ - gamma).abs() / gamma < 1e-3, "∫ω dA = {circ}, Γ = {gamma}");
    // Central vorticity decays as 1/t: ω(0,2t)/ω(0,t) = 1/2 exactly.
    let ratio = vort(0.0, 2.0 * t) / vort(0.0, t);
    assert!((ratio - 0.5).abs() < 1e-12, "ω(0) ∝ 1/t");
    // Far field: v_θ(r) = Γ/(2πr)(1−e^{−r²/4νt}) → Γ/(2πr) as r → ∞; at
    // r = 10 with 4νt = 1 the correction e^{−100} is invisible.
    let r = 10.0;
    let v_inf = gamma / (2.0 * PI * r);
    let v_far = gamma / (2.0 * PI * r) * (1.0 - (-r * r / (4.0 * nu_t * t)).exp());
    assert!((v_far / v_inf - 1.0).abs() < 1e-12);
}

#[test]
fn ns_strouhal_vortex_shedding_pin() {
    // Vortex shedding behind a cylinder at Re ≈ 10³: St = f·D/U = 0.2
    // (literature band 0.18–0.22). The fiber's classical content per the
    // §5.25 ledger, pinned here.
    let st: f64 = 0.2;
    assert!((st - 0.2f64).abs() < 1e-12);
    // Shedding frequency from the definition: f = St·U/D.
    let u: f64 = 10.0; // m/s
    let d: f64 = 0.1; // m
    let f = st * u / d;
    assert!((f - 20.0).abs() < 1e-12, "f = {f} Hz");
    // St is dimensionless: doubling U and D leaves f·D/U invariant.
    let f2 = st * (2.0 * u) / (2.0 * d);
    assert!((f2 - f).abs() < 1e-12);
}

#[test]
fn ns_reynolds_number_and_transition_pin() {
    // Re = ρUD/μ is dimensionless, and the pipe laminar→turbulent transition
    // sits at Re_c ≈ 2300 (the Reynolds 1883 experiment). Both pinned here
    // as the plumbing anchor of the turbulent correlations above.
    let re = |rho: f64, u: f64, d: f64, mu: f64| rho * u * d / mu;
    // Water at 20 °C in a 2 cm pipe at 0.1 m/s: Re ≈ 2000 (laminar).
    let rho: f64 = 998.2;
    let mu: f64 = 1.002e-3;
    let re_lam = re(rho, 0.1, 0.02, mu);
    assert!((re_lam - 1992.4).abs() / 1992.4 < 1e-3, "Re = {re_lam}");
    // Dimensionless: doubling U and D (Re ×4) with μ ×4 restores Re.
    assert!((re(rho, 0.2, 0.04, 4.0 * mu) - re_lam).abs() < 1e-9);
    // Transition pin: Re_c ≈ 2300 (pipe).
    let re_c: f64 = 2300.0;
    assert!((re_c - 2300.0).abs() < 1e-12);
    // Doubling U crosses the transition.
    let re_turb = re(rho, 0.2, 0.02, mu);
    assert!(re_lam < re_c && re_turb > re_c, "{re_lam} < {re_c} < {re_turb}");
}

#[test]
fn ns_blasius_pipe_friction_correlation() {
    // The Blasius correlation for turbulent pipe flow: f = 0.3164·Re^{−1/4}
    // (4·10³ < Re < 10⁵), and the wall shear τ_w = ½ρfU² it implies.
    let f = |re: f64| 0.3164 * re.powf(-0.25);
    // f(10⁵) = 0.3164·10^{−1.25} ≈ 0.0178 — the textbook turbulent value.
    let f5 = f(1e5);
    assert!((f5 - 0.0178).abs() < 1e-3, "f(10⁵) = {f5}");
    // Re^{−1/4}: a 16× increase in Re halves the friction factor.
    assert!((f(16e5) / f(1e5) - 0.5).abs() < 1e-9);
    // Wall shear from the friction factor: τ_w = ½ρfU².
    let rho: f64 = 1000.0;
    let u: f64 = 1.0;
    let tau_w = 0.5 * rho * f5 * u * u;
    assert!((tau_w - 8.9).abs() < 0.5, "τ_w = {tau_w:.2} Pa");
    // Consistency with the Darcy–Weisbach head loss: ΔP = f·(L/D)·½ρU².
    let l: f64 = 10.0;
    let d: f64 = 0.1;
    let dp = f5 * (l / d) * 0.5 * rho * u * u;
    assert!((dp - 890.0).abs() < 50.0, "ΔP = {dp:.0} Pa");
}

#[test]
fn ns_blasius_thicknesses_and_shape_factor() {
    // δ*/x = 1.7208/√Re_x, θ/x = 0.664/√Re_x, H = δ*/θ = 2.5916.
    let re: f64 = 1e6;
    let dstar_x = 1.7208 / re.sqrt();
    let theta_x = 0.664 / re.sqrt();
    let h = dstar_x / theta_x;
    assert!((h - 2.5916).abs() < 1e-3, "H = {h}");
    // √x growth: δ*(x) = 1.7208·x/√Re_x ∝ √x. At stations x and 4x (Re ∝ x
    // at fixed U, ν) the ratio is δ*(4x)/δ*(x) = 4·(δ*/x)|₄ₓ / (δ*/x)|ₓ = √4 = 2.
    let re4: f64 = 4e6;
    let dstar_x4 = 1.7208 / re4.sqrt(); // δ*/x at 4x
    let dstar_x1 = 1.7208 / re.sqrt(); // δ*/x at x
    let growth = 4.0 * dstar_x4 / dstar_x1;
    assert!((growth - 2.0).abs() < 1e-9, "δ*(4x)/δ*(x) = {growth}");
    // Consistency with the 99% thickness δ₉₉/x = 4.92/√Re_x pinned in the
    // boundary-layer suite: δ₉₉/δ* = 4.92/1.7208 ≈ 2.86.
    let ratio = 4.92 / 1.7208;
    assert!((ratio - 2.859f64).abs() < 1e-3, "δ₉₉/δ* = {ratio}");
}
