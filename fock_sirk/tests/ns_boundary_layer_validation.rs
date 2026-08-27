//! Navier–Stokes boundary-layer and turbulence-scale numerical validation.
//!
//! Extends `ns_numerical_validation.rs` with:
//!
//! 1. **Blasius boundary layer by shooting** — the similarity equation
//!    `f''' + ½ f f'' = 0`, `f(0) = f'(0) = 0`, `f'(∞) = 1` solved
//!    numerically (RK4 + shooting on `f''(0)`): the published constants
//!    `f''(0) = 0.33206`, the shape factor `H = δ*/θ = 2.5916`, the 1%
//!    boundary-layer thickness `δ₉₉ ≈ 4.92`, and the skin-friction
//!    coefficient `C_f = 2f''(0)/√Re_x = 0.664/√Re_x` — the ODE solution the
//!    published numbers are measured from.
//! 2. **Turbulent length-scale identities** — the exact consequences of the
//!    Kolmogorov/Taylor relations: `λ/η = 15^{1/4}√Re_λ` and
//!    `L/λ = Re_λ/15`, verified across the inertial-range Reynolds numbers.

/// RK4 step of the Blasius system `(f, f', f'')' = (f', f'', −½f f'')`.
fn blasius_step(y: [f64; 3], h: f64) -> [f64; 3] {
    let f = |y: [f64; 3]| [y[1], y[2], -0.5 * y[0] * y[2]];
    let k1 = f(y);
    let k2 = f([y[0] + h / 2.0 * k1[0], y[1] + h / 2.0 * k1[1], y[2] + h / 2.0 * k1[2]]);
    let k3 = f([y[0] + h / 2.0 * k2[0], y[1] + h / 2.0 * k2[1], y[2] + h / 2.0 * k2[2]]);
    let k4 = f([y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2]]);
    [
        y[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        y[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        y[2] + h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    ]
}

/// Integrate the Blasius equation from `η = 0` to `η_max` with `f''(0) = s`.
/// Returns the terminal state and the integral thicknesses.
fn blasius_shoot(s: f64, eta_max: f64, h: f64) -> ([f64; 3], f64, f64) {
    let n = (eta_max / h) as usize;
    let mut y = [0.0, 0.0, s];
    let mut delta_star = 0.0; // ∫(1 − f') dη
    let mut theta = 0.0; // ∫ f'(1 − f') dη
    let mut prev = y;
    for _ in 0..n {
        let next = blasius_step(y, h);
        // Trapezoid on the thickness integrals.
        delta_star += h / 2.0 * ((1.0 - prev[1]) + (1.0 - next[1]));
        theta += h / 2.0 * ((prev[1] * (1.0 - prev[1])) + (next[1] * (1.0 - next[1])));
        prev = next;
        y = next;
    }
    (y, delta_star, theta)
}

#[test]
fn ns_blasius_shooting_reproduces_published_profile() {
    let eta_max = 12.0;
    let h = 0.001;

    // Shooting: find f''(0) so that f'(η_max) → 1 (the free-stream match).
    // Bisection on [0.2, 0.5]; the solution is monotone in the initial slope.
    let mut lo = 0.2;
    let mut hi = 0.5;
    let mut s = 0.5 * (lo + hi);
    for _ in 0..60 {
        s = 0.5 * (lo + hi);
        let (end, _, _) = blasius_shoot(s, eta_max, h);
        if end[1] > 1.0 {
            hi = s;
        } else {
            lo = s;
        }
    }
    let (end, delta_star, theta) = blasius_shoot(s, eta_max, h);

    // Free-stream match.
    assert!(
        (end[1] - 1.0).abs() < 1e-6,
        "shooting must match f'(∞) = 1, got {}",
        end[1]
    );
    // Published Blasius constants.
    assert!(
        (s - 0.33206).abs() / 0.33206 < 5e-3,
        "f''(0) must be 0.33206, got {s:.6}"
    );
    let h_shape = delta_star / theta;
    assert!(
        (h_shape - 2.5916).abs() / 2.5916 < 1e-2,
        "shape factor H = δ*/θ must be 2.5916, got {h_shape:.4}"
    );
    // Skin friction: C_f = 2·f''(0)/√Re_x = 0.664/√Re_x.
    let cf_coeff = 2.0 * s;
    assert!(
        (cf_coeff - 0.66406).abs() / 0.66406 < 5e-3,
        "C_f·√Re_x must be 0.664, got {cf_coeff:.5}"
    );

    // The 1% velocity thickness: η where f' = 0.99.
    let mut eta99 = None;
    let mut y = [0.0, 0.0, s];
    let mut prev = y;
    for i in 0..(eta_max / h) as usize {
        let next = blasius_step(y, h);
        if prev[1] < 0.99 && next[1] >= 0.99 {
            eta99 = Some((i as f64 + 1.0) * h);
            break;
        }
        prev = next;
        y = next;
    }
    let eta99 = eta99.expect("f' = 0.99 reached");
    assert!(
        (eta99 - 4.92).abs() / 4.92 < 3e-2,
        "δ₉₉ must be ≈ 4.92, got {eta99:.3}"
    );
    eprintln!(
        "ns_blasius_shooting: f''(0) = {s:.6} (published 0.33206), H = {h_shape:.4} (2.5916), \
         δ₉₉ = {eta99:.3} (4.92), C_f·√Re_x = {cf_coeff:.5} (0.664)"
    );
}

#[test]
fn ns_turbulent_length_scale_identities() {
    // Exact consequences of the Kolmogorov/Taylor relations:
    //   ε = u'³/L  (integral),  ε = 15νu'²/λ²  (Taylor),  η = (ν³/ε)^{1/4}
    // ⇒ λ/η = 15^{1/4}·Re_λ^{1/2}  and  L/λ = Re_λ/15.
    for re_lambda in [30.0, 100.0, 400.0] {
        let re_lambda: f64 = re_lambda;
        let ratio1 = 15.0_f64.powf(0.25) * re_lambda.sqrt();
        let ratio2 = re_lambda / 15.0;
        // Consistency: all three length scales derive from the same ε, ν, u'.
        let nu: f64 = 1.0e-6;
        let u_prime: f64 = 1.0;
        let lambda = re_lambda * nu / u_prime;
        let eps = 15.0 * nu * u_prime * u_prime / (lambda * lambda);
        let eta = (nu * nu * nu / eps).powf(0.25);
        let l = u_prime * u_prime * u_prime / eps;
        assert!(
            (lambda / eta - ratio1).abs() / ratio1 < 1e-9,
            "λ/η must equal 15^{{1/4}}√Re_λ: {} vs {}",
            lambda / eta,
            ratio1
        );
        assert!(
            (l / lambda - ratio2).abs() / ratio2 < 1e-9,
            "L/λ must equal Re_λ/15: {} vs {}",
            l / lambda,
            ratio2
        );
        // The inertial-range ordering L > λ > η at these Re_λ.
        assert!(l > lambda && lambda > eta);
    }
    eprintln!("ns_turbulent_length_scale_identities: λ/η = 15^(1/4)·√Re_λ, L/λ = Re_λ/15 (exact)");
}
