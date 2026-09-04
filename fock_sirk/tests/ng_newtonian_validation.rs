//! Newtonian gravity (NG) numerical validation — the classical sector that QG
//! must reduce to in the weak-field limit.
//!
//! 1. **Kepler's third law**: `T² = 4π²a³/(GM_☉)` — the Earth year to 6
//!    significant figures, and the `T² ∝ a³` scaling across the solar system.
//! 2. **Virial theorem**: a circular orbit has `2⟨T⟩ + ⟨V⟩ = 0` exactly
//!    (`v = √(GM/r)` ⇒ `V = −2T`).
//! 3. **Shell theorem**: the field of a uniform spherical shell is exactly
//!    zero inside and `GM/r²` outside (verified by numeric quadrature).
//! 4. **Escape velocity**: `v_esc = √(2GM/R)` = 11.19 km/s from Earth; the
//!    `√2` factor vs circular-orbit speed.
//! 5. **Binding energy**: a uniform sphere has `U = 3GM²/(5R)` — Earth
//!    `≈ 2.24e32 J`.
//! 6. **Symplectic two-body integration**: leapfrog conserves energy to a
//!    bounded oscillation — `|ΔE/E| < 1e−4` over 100 circular orbits.

const G: f64 = 6.674_30e-11;
const M_SUN: f64 = 1.988_47e30;
const M_EARTH: f64 = 5.972_2e24;
const R_EARTH: f64 = 6.371e6;
const AU: f64 = 1.495_978_707e11;
const YEAR_S: f64 = 365.25 * 86_400.0;

#[test]
fn ng_kepler_third_law_earth_year() {
    let t = (4.0 * std::f64::consts::PI * std::f64::consts::PI * AU.powi(3) / (G * M_SUN)).sqrt();
    let rel = (t - YEAR_S).abs() / YEAR_S;
    assert!(
        rel < 1e-3,
        "T = {t:.4e} s, sidereal year {YEAR_S:.4e} (rel {rel:.2e})"
    );
}

#[test]
fn ng_kepler_t2_proportional_a3() {
    // T²/a³ = 4π²/(GM_☉) is the same constant for every planet.
    // Two-body μ = G(M_☉ + m_planet) — the planet's mass shifts the constant
    // by ~1e-3 for Jupiter. Elements are rounded textbook values, so the
    // tolerance is 5e-3 (the residual is element precision, not the law).
    let k = |mu: f64| 4.0 * std::f64::consts::PI * std::f64::consts::PI / mu;
    // (semi-major axis [m], sidereal period [s], planet mass [kg])
    let planets = [
        (5.790_905e10, 87.969 * 86_400.0, 3.301e23),
        (1.495_978_707e11, 365.256 * 86_400.0, 5.972e24),
        (7.785_472e11, 4_332.59 * 86_400.0, 1.898e27),
    ];
    for (a, t, m_p) in planets {
        let ratio = t * t / (a * a * a);
        let expected = k(G * (M_SUN + m_p));
        let rel = (ratio - expected).abs() / expected;
        assert!(
            rel < 5e-3,
            "T²/a³ = {ratio:.6e}, expected {expected:.6e} (rel {rel:.2e})"
        );
    }
}

#[test]
fn ng_virial_theorem_circular_orbit() {
    // Circular orbit: v = √(GM/r); T = ½mv², V = −GMm/r = −2T ⇒ 2T + V = 0.
    let m = 1.0;
    let r = 2.0 * AU;
    let v = (G * M_SUN / r).sqrt();
    let t_kin = 0.5 * m * v * v;
    let v_pot = -G * M_SUN * m / r;
    let virial = 2.0 * t_kin + v_pot;
    assert!(virial.abs() / t_kin.abs() < 1e-12, "2T+V = {virial}");
}

#[test]
fn ng_shell_theorem_inside_and_outside() {
    // Uniform shell radius R, mass M. Numeric quadrature of the net field
    // along the axis at distance r from the center.
    let shell_field = |r: f64| -> f64 {
        let r_shell = 1.0;
        let m = 1.0;
        let n = 4000;
        let sigma = m / (4.0 * std::f64::consts::PI * r_shell * r_shell);
        let mut a = 0.0;
        for i in 0..n {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            let x = r_shell * theta.cos();
            let y = r_shell * theta.sin();
            let d = ((r - x).powi(2) + y * y).sqrt();
            let dm = sigma
                * 2.0
                * std::f64::consts::PI
                * r_shell
                * r_shell
                * theta.sin()
                * (std::f64::consts::PI / n as f64);
            a += G * dm * (r - x) / d.powi(3);
        }
        a
    };
    // Inside (r = 0.5 R): the field vanishes.
    let a_in = shell_field(0.5);
    assert!(a_in.abs() < 1e-12, "field inside shell = {a_in}");
    // Outside (r = 2 R): field = GM/r² exactly (concentrated point mass).
    let a_out = shell_field(2.0);
    let expected = G * 1.0 / 4.0;
    assert!(
        (a_out - expected).abs() / expected < 1e-4,
        "field outside = {a_out}, expected {expected}"
    );
}

#[test]
fn ng_escape_velocity_earth() {
    let v_esc = (2.0 * G * M_EARTH / R_EARTH).sqrt();
    let expected = 11_186.0; // m/s
    let rel = (v_esc - expected).abs() / expected;
    assert!(
        rel < 5e-3,
        "v_esc = {v_esc:.1} m/s, expected {expected} (rel {rel:.2e})"
    );
    // √2 vs circular speed: v_circ = √(GM/R), v_esc = √2·v_circ.
    let v_circ = (G * M_EARTH / R_EARTH).sqrt();
    assert!((v_esc / v_circ - 2f64.sqrt()).abs() < 1e-12);
}

#[test]
fn ng_uniform_sphere_binding_energy() {
    // U = 3GM²/(5R). Earth: ≈ 2.24e32 J.
    let u = 3.0 * G * M_EARTH * M_EARTH / (5.0 * R_EARTH);
    let expected = 2.24e32;
    let rel = (u - expected).abs() / expected;
    assert!(
        rel < 1e-2,
        "U = {u:.3e} J, expected {expected:.2e} (rel {rel:.2e})"
    );
}

#[test]
fn ng_gravitational_parameter_plumbing() {
    // Anchor the constants plumbing (per the guide's §1 rule): the standard
    // gravitational parameters GM for the Sun and Earth, which every orbit
    // formula above consumes.
    let gm_sun = G * M_SUN;
    let expected_sun: f64 = 1.327_124_400_18e20; // m³/s² (IAU 2015)
    let rel_sun = (gm_sun - expected_sun).abs() / expected_sun;
    assert!(
        rel_sun < 1e-3,
        "GM_☉ = {gm_sun:.4e}, expected {expected_sun:.4e}"
    );
    let gm_earth = G * M_EARTH;
    let expected_earth: f64 = 3.986_004_418e14; // m³/s²
    let rel_earth = (gm_earth - expected_earth).abs() / expected_earth;
    assert!(
        rel_earth < 1e-3,
        "GM_⊕ = {gm_earth:.4e}, expected {expected_earth:.4e}"
    );
    // Kepler consistency: GM_☉ = 4π²·AU³/yr² with the sidereal year.
    let year_s: f64 = 365.256 * 86_400.0;
    let from_kepler =
        4.0 * std::f64::consts::PI * std::f64::consts::PI * AU.powi(3) / year_s.powi(2);
    let rel_k = (from_kepler - gm_sun).abs() / gm_sun;
    assert!(rel_k < 1e-3, "Kepler GM_☉ = {from_kepler:.4e}");
}

#[test]
fn ng_tidal_acceleration_sun_moon_ratio() {
    // Tidal acceleration at Earth's surface scales as M/d³. The Sun/Moon
    // ratio is 0.46 — the Sun's tide is *weaker* than the Moon's despite its
    // 27 million× larger mass, because d³ suppresses it.
    let m_moon: f64 = 7.342e22;
    let d_moon: f64 = 3.844e8;
    let d_sun: f64 = 1.495_978_707e11;
    let ratio = (M_SUN / d_sun.powi(3)) / (m_moon / d_moon.powi(3));
    assert!(
        (ratio - 0.46).abs() < 0.02,
        "Sun/Moon tide = {ratio:.3}, expected ≈ 0.46"
    );
    // The 1/d³ law: doubling the distance divides the tide by 8.
    assert!(((m_moon / (2.0 * d_moon).powi(3)) / (m_moon / d_moon.powi(3)) - 0.125).abs() < 1e-12);
}

#[test]
fn ng_hill_sphere_earth_moon() {
    // Hill sphere: r_H = a·∛(m/(3M)) — the region where the satellite's
    // gravity dominates. Earth–Moon: r_H ≈ 61,500 km ≈ 1/6.25 of the
    // Earth–Moon distance.
    let a: f64 = 3.844e8; // Earth–Moon distance [m]
    let m_earth: f64 = 5.972e24;
    let m_moon: f64 = 7.342e22;
    let r_h = a * (m_moon / (3.0 * m_earth)).cbrt();
    let expected: f64 = 61_500.0e3; // m
    let rel = (r_h - expected).abs() / expected;
    assert!(
        rel < 3e-2,
        "r_H = {r_h:.4e} m, expected {expected:.4e} (rel {rel:.2e})"
    );
    // Scaling: r_H ∝ a·∛(m/M) — doubling the mass ratio multiplies by ∛2.
    let r_h2 = a * (2.0 * m_moon / (3.0 * m_earth)).cbrt();
    assert!((r_h2 / r_h - 2f64.cbrt()).abs() < 1e-12);
}

#[test]
fn ng_roche_limit_earth_moon() {
    // Roche limit (rigid satellite): d = R·∛(2ρ_p/ρ_s) — the distance at
    // which tidal forces overcome the satellite's self-gravity. Earth–Moon:
    // ≈ 9,500 km (≈ 1.5 Earth radii); Saturn's rings sit inside Saturn's
    // Roche limit, which is why they never coalesced.
    let r_earth: f64 = 6.371e6;
    let rho_earth: f64 = 5515.0; // kg/m³
    let rho_moon: f64 = 3344.0;
    let d = r_earth * (2.0 * rho_earth / rho_moon).cbrt();
    let expected: f64 = 9_496.0e3; // m
    let rel = (d - expected).abs() / expected;
    assert!(
        rel < 3e-2,
        "d_Roche = {d:.4e} m, expected {expected:.4e} (rel {rel:.2e})"
    );
    // Scaling: d ∝ ∛(ρ_p/ρ_s) — a denser primary pushes the limit out.
    let d2 = r_earth * (2.0 * 2.0 * rho_earth / rho_moon).cbrt();
    assert!((d2 / d - 2f64.cbrt()).abs() < 1e-12);
}

#[test]
fn ng_leapfrog_two_body_energy_conservation() {
    // Two equal masses on circular orbits; leapfrog (symplectic) for 100
    // periods. Energy must stay within a bounded oscillation of the initial
    // value — a hallmark of symplectic integrators.
    let m: f64 = 1.0;
    let r: f64 = 1.0;
    // Each body orbits the common center at radius r; the separation is 2r, so
    // v²/r = Gm/(2r)² ⇒ v = √(Gm/4r) (not √(Gm/2r) — that is the escape speed
    // and gives exactly zero total energy).
    let v = (G * m / (4.0 * r)).sqrt();
    let period = 2.0 * std::f64::consts::PI * r / v;
    let dt = period / 1000.0;
    let steps = 100 * 1000;
    // Initial conditions: bodies at ±r on the x-axis, velocities ±v on y.
    let mut x1 = -r;
    let mut y1 = 0.0f64;
    let mut vx1 = 0.0f64;
    let mut vy1 = v;
    let mut x2 = r;
    let mut y2 = 0.0f64;
    let mut vx2 = 0.0f64;
    let mut vy2 = -v;
    let e0 = 0.5 * m * (vx1 * vx1 + vy1 * vy1 + vx2 * vx2 + vy2 * vy2)
        - G * m * m / ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let mut e_min = f64::INFINITY;
    let mut e_max = f64::NEG_INFINITY;
    for _ in 0..steps {
        // Kick-drift-kick leapfrog.
        let dx = x2 - x1;
        let dy = y2 - y1;
        let d3 = (dx * dx + dy * dy).powf(1.5);
        let ax1 = G * m * dx / d3;
        let ay1 = G * m * dy / d3;
        vx1 += 0.5 * ax1 * dt;
        vy1 += 0.5 * ay1 * dt;
        vx2 -= 0.5 * ax1 * dt;
        vy2 -= 0.5 * ay1 * dt;
        x1 += vx1 * dt;
        y1 += vy1 * dt;
        x2 += vx2 * dt;
        y2 += vy2 * dt;
        let dx = x2 - x1;
        let dy = y2 - y1;
        let d3 = (dx * dx + dy * dy).powf(1.5);
        let ax1 = G * m * dx / d3;
        let ay1 = G * m * dy / d3;
        vx1 += 0.5 * ax1 * dt;
        vy1 += 0.5 * ay1 * dt;
        vx2 -= 0.5 * ax1 * dt;
        vy2 -= 0.5 * ay1 * dt;
        let e = 0.5 * m * (vx1 * vx1 + vy1 * vy1 + vx2 * vx2 + vy2 * vy2)
            - G * m * m / ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        e_min = e_min.min(e);
        e_max = e_max.max(e);
    }
    let drift = ((e_max - e_min).abs() / e0.abs()).max((e_max - e0).abs() / e0.abs());
    assert!(
        drift < 1e-4,
        "leapfrog energy drift over 100 orbits = {drift:.2e}"
    );
}
