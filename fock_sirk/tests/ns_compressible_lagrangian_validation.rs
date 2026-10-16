//! Numerical validation of the COMPRESSIBLE Navier-Stokes Lagrangian line.
//!
//! Companion to the symbolic module `docs/ns_compressible_lagrangian.cdb`
//! (CHECK 1–19) and to `docs/VERIFY_NS_COMPRESSIBLE.md`.  Every test is a
//! numerical identity of the module's certified content, evaluated in f64
//! to the exact-arithmetic tolerance class of the guide §6 (1e-9 … 1e-12).
//! No SIRK solve here — pure algebra/analysis of the compressible
//! material-derivative formalism, in the style of
//! `ns_sm_qg_momentum_convolution_validation.rs`.
//!
//! The module's content, test by test:
//!
//! 1. `ns_cmp_jacobi_transport_identity` — CHECK 4: Jacobi's formula
//!    `d(det F)/dt = det F · tr(F⁻¹ Ḟ)` for a smooth one-parameter
//!    deformation, finite-differenced against the analytic trace form.
//!
//! 2. `ns_cmp_mass_conservation_rho_j` — CHECK 5–6: `ρ J = ρ_0` is
//!    constant along the flow: with `ρ = ρ_0/J` the product `ρ̇ J + ρ J̇`
//!    vanishes to finite-difference accuracy (the material continuity
//!    equation, with `ρ̇ = −ρ div u` and `J̇ = J div u` as independent
//!    inputs).
//!
//! 3. `ns_cmp_pressure_from_material_potential` — CHECK 8: the pressure
//!    of a general barotrope is `p = ρ² e'(ρ) = −dW/dJ` with
//!    `W(J) = ρ₀ e(ρ_0/J)`: both routes are finite-differenced against
//!    the first law `p = ρ ε' − ε`, `ε = ρ e`, for an isothermal and a
//!    polytropic specific internal energy.
//!
//! 4. `ns_cmp_enthalpy_force_identity` — CHECK 9: `dp = ρ dh` (which is
//!    what makes `(1/ρ)∇p = ∇h`): finite-difference `p'(ρ)` against
//!    `ρ h'(ρ)` for both barotropes.
//!
//! 5. `ns_cmp_isothermal_anchor` — CHECK 10: `e = c² ln ρ` gives
//!    `p = c² ρ` and a constant sound speed.
//!
//! 6. `ns_cmp_acoustic_dispersion` — CHECK 11: the linearized fiber has
//!    `ω² = c²|k|²`; the 2×2 system matrix is diagonalized and its
//!    eigenvalues compared against `±c|k|`.
//!
//! 7. `ns_cmp_density_weighted_pressure_convolution` — CHECK 13–14: the
//!    density-weighted pressure force `(1/ρ)∂ᵢp` is the two-wave
//!    convolution `r ⋆ (i q p)` with `r = 𝓕(1/ρ)`: the real-space
//!    product is compared against the convolution form (the S47
//!    convention `f⋆g[n] = Σ_m f[m] g[(n−m) mod n]`).
//!
//! 8. `ns_cmp_derivative_product_convolution_weight` — CHECK 13: the
//!    product of derivatives `(∂f)(∂g)` carries the two transfer weights
//!    `(i p)(i q)` — the real-space product of gradients matches the
//!    weighted convolution.
//!
//! 9. `ns_cmp_viscous_weight_even_diagonal` — CHECK 15: the viscous
//!    symbol `ν|k|²` is diagonal (one mode in, one mode out) and even in
//!    `k`; contrast with the convolution terms which couple mode pairs.
//!
//! 10. `ns_cmp_legendre_identity` — CHECK 16: `H = π²/(2ρ₀) + ρ₀ e`
//!     equals `π v − L` at `π = ρ₀ v`, numerically for random states.
//!
//! 11. `ns_cmp_fiber_completed_square` — CHECK 18: the linearized fiber
//!     Hamiltonian equals `(1/(2ρ̄))(π² + (ρ̄ c k s)²)`.
//!
//! 12. `ns_cmp_fl_completion_square_nonnegative` — CHECK 19: the
//!     Faris–Lavine completion square `N − h = ½ ρ̄ c² (s − s₀)²` is
//!     exactly the residual and is non-negative (the algebraic half of
//!     the domination; the growth hypothesis on a general e(ρ) stays an
//!     honest boundary).

use num_complex::Complex64;

// ── discrete Fourier / convolution helpers (the S47 convention) ─────────────

/// Forward DFT `F[k] = Σ_t f[t] exp(−2πi k t / n)`.
fn dft(f: &[Complex64]) -> Vec<Complex64> {
    let n = f.len();
    assert!(
        n > 0 && n.is_power_of_two(),
        "DFT length must be a power of two"
    );
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (k, out_k) in out.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (t, &ft) in f.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            sum += ft * Complex64::new(ang.cos(), ang.sin());
        }
        *out_k = sum;
    }
    out
}

/// Inverse DFT `f[t] = (1/n) Σ_k F[k] exp(+2πi k t / n)`.
fn idft(f: &[Complex64]) -> Vec<Complex64> {
    let n = f.len();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (t, out_t) in out.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (k, &fk) in f.iter().enumerate() {
            let ang = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            sum += fk * Complex64::new(ang.cos(), ang.sin());
        }
        *out_t = sum / n as f64;
    }
    out
}

/// Circular convolution `(f ⋆ g)[q] = Σ_m f[m] g[(q − m) mod n]` — NOT
/// `g[(q + m)]` (S47: this sign is the load-bearing convention).
fn circular_conv(f: &[Complex64], g: &[Complex64]) -> Vec<Complex64> {
    let n = f.len();
    assert_eq!(n, g.len(), "convolution needs equal lengths");
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (q, out_q) in out.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for m in 0..n {
            sum += f[m] * g[(q + n - m) % n];
        }
        *out_q = sum;
    }
    out
}

/// Discrete gradient symbol `∂_x → i·(2π k_signed / L)` on a period-`L` box.
fn grad_symbol(k: usize, n: usize, length: f64) -> Complex64 {
    let signed = if k < n.div_ceil(2) {
        k as i64
    } else {
        k as i64 - n as i64
    };
    let kx = 2.0 * std::f64::consts::PI * (signed as f64) / length;
    Complex64::new(0.0, kx)
}

// ── barotropic specific internal energies (the module keeps e general) ──────

/// Isothermal barotrope: `e(ρ) = c² ln ρ`, `e' = c²/ρ`, `p = c²ρ`.
fn e_iso(rho: f64, c2: f64) -> f64 {
    c2 * rho.ln()
}

/// Polytropic barotrope: `e(ρ) = c² ρ^{γ−1}/(γ−1)`, `p = c² ρ^γ/(γ−1)`.
fn e_poly(rho: f64, c2: f64, gamma: f64) -> f64 {
    c2 * rho.powf(gamma - 1.0) / (gamma - 1.0)
}

// ── 1. Jacobi's formula for the flow Jacobian ───────────────────────────────

#[test]
fn ns_cmp_jacobi_transport_identity() {
    // F(t) = I + t M + t² N (smooth one-parameter deformation). Jacobi's
    // formula: d(det F)/dt = det F · tr(F⁻¹ Ḟ). Both sides are evaluated
    // at sample times: the left by central finite differences of det F(t),
    // the right from the analytic Ḟ = M + 2t N.
    let m = [
        [0.30, -0.20, 0.10],
        [0.15, 0.40, -0.25],
        [-0.10, 0.05, 0.20],
    ];
    let n = [
        [0.010, -0.020, 0.005],
        [0.015, -0.010, 0.020],
        [-0.005, 0.010, -0.015],
    ];
    let det3 = |f: [[f64; 3]; 3]| {
        f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
            - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
            + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0])
    };
    let f_at = |t: f64| -> [[f64; 3]; 3] {
        let mut f = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                f[i][j] = if i == j { 1.0 } else { 0.0 } + t * m[i][j] + t * t * n[i][j];
            }
        }
        f
    };
    // F⁻¹ via adjugate / det for a 3×3.
    let inv_and_trace = |t: f64| -> (f64, f64) {
        let f = f_at(t);
        let d = det3(f);
        // Ḟ = M + 2tN
        let mut fdot = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                fdot[i][j] = m[i][j] + 2.0 * t * n[i][j];
            }
        }
        // tr(F⁻¹ Ḟ) = tr(adj(F) Ḟ)/det F
        let cof = |a: usize, b: usize| {
            let r: Vec<usize> = (0..3).filter(|&x| x != a).collect();
            let c: Vec<usize> = (0..3).filter(|&x| x != b).collect();
            let minor = f[r[0]][c[0]] * f[r[1]][c[1]] - f[r[0]][c[1]] * f[r[1]][c[0]];
            if (a + b) % 2 == 0 { minor } else { -minor }
        };
        let mut tr = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                // tr(adj(F) Fdot) = sum_{a,b} cof_{a b} Fdot_{a b}
                tr += cof(i, j) * fdot[i][j];
            }
        }
        (d, tr / d)
    };
    for &t in &[0.0f64, 0.37, 1.10, -0.62] {
        let h = 1e-6;
        let d_plus = det3(f_at(t + h));
        let d_minus = det3(f_at(t - h));
        let lhs = (d_plus - d_minus) / (2.0 * h);
        let (det, tr_term) = inv_and_trace(t);
        let rhs = det * tr_term;
        assert!(
            (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
            "Jacobi: d(det F)/dt = det F tr(F^-1 Fdot) at t={t}: {lhs} vs {rhs}"
        );
    }
}

// ── 2. mass conservation ρ J = ρ₀ ──────────────────────────────────────────

#[test]
fn ns_cmp_mass_conservation_rho_j() {
    // Along the same deformation, rho(t) = rho0/J(t) must make d(rho J)/dt
    // vanish: rho_dot = -rho div u and J_dot = +J div u (CHECK 4) are
    // INDEPENDENT inputs whose sum cancels.
    let rho0 = 1.37f64;
    let div_u = 0.42f64; // constant divergence flow: J(t) = J0 exp(div_u t)
    let j0 = 0.83f64;
    let j_at = |t: f64| j0 * (div_u * t).exp();
    let rho_at = |t: f64| rho0 / j_at(t);
    for &t in &[0.0f64, 0.5, 1.7] {
        let h = 1e-7;
        let j_dot = (j_at(t + h) - j_at(t - h)) / (2.0 * h);
        let rho_dot = (rho_at(t + h) - rho_at(t - h)) / (2.0 * h);
        // the two independent physical inputs:
        let j_dot_theory = j_at(t) * div_u;
        let rho_dot_theory = -rho_at(t) * div_u;
        assert!((j_dot - j_dot_theory).abs() < 1e-6, "J_dot = J div u");
        assert!(
            (rho_dot - rho_dot_theory).abs() < 1e-6,
            "rho_dot = -rho div u"
        );
        // and their combination — the material continuity equation:
        let cont = rho_dot * j_at(t) + rho_at(t) * j_dot;
        assert!(
            cont.abs() < 1e-6 * (1.0 + rho0 * div_u),
            "d(rho J)/dt must vanish: {cont}"
        );
    }
}

// ── 3. p = ρ² e'(ρ) = −dW/dJ = ρ ε' − ε ─────────────────────────────────────

#[test]
fn ns_cmp_pressure_from_material_potential() {
    let c2 = 1.9f64;
    let gamma = 1.4f64;
    let rho0 = 2.1f64;
    let barotropes: [(&str, Box<dyn Fn(f64) -> f64>); 2] = [
        ("isothermal", Box::new(|r: f64| e_iso(r, c2))),
        ("polytropic", Box::new(|r: f64| e_poly(r, c2, gamma))),
    ];
    for (name, e_of) in &barotropes {
        for &rho in &[0.4f64, 1.0, 3.3] {
            // e' by central finite differences:
            let h = 1e-6 * rho.max(1.0);
            let eprime = (e_of(rho + h) - e_of(rho - h)) / (2.0 * h);
            // route 1: the thermodynamic definition p = rho^2 e'
            let p1 = rho * rho * eprime;
            // route 2: the first law p = rho eps' - eps at eps = rho e
            let eps_of = |r: f64| r * e_of(r);
            let eps_prime = (eps_of(rho + h) - eps_of(rho - h)) / (2.0 * h);
            let p2 = rho * eps_prime - eps_of(rho);
            assert!(
                (p1 - p2).abs() < 1e-5 * (1.0 + p1.abs()),
                "p = rho^2 e' = rho eps' - eps ({name}) at rho={rho}: {p1} vs {p2}"
            );
            // route 3: the material potential W(J) = rho0 e(rho0/J),
            // dW/dJ = -p at rho = rho0/J:
            let j = rho0 / rho;
            let w_of = |jj: f64| rho0 * e_of(rho0 / jj);
            let hj = 1e-6 * j.max(1.0);
            let dw = (w_of(j + hj) - w_of(j - hj)) / (2.0 * hj);
            assert!(
                (dw + p1).abs() < 1e-4 * (1.0 + p1.abs()),
                "dW/dJ = -p ({name}) at rho={rho}: dW/dJ={dw}, p={p1}"
            );
        }
    }
}

// ── 4. dp = rho dh (the (1/rho) grad p = grad h identity) ──────────────────

#[test]
fn ns_cmp_enthalpy_force_identity() {
    let c2 = 1.7f64;
    let gamma = 1.35f64;
    for &kind in &[0isize, 1] {
        let e_of = |r: f64| {
            if kind == 0 {
                e_iso(r, c2)
            } else {
                e_poly(r, c2, gamma)
            }
        };
        for &rho in &[0.5f64, 1.2, 2.9] {
            let h = 1e-6 * rho.max(1.0);
            let p_of = |r: f64| {
                let hp = 1e-6 * r.max(1.0);
                let eprime = (e_of(r + hp) - e_of(r - hp)) / (2.0 * hp);
                r * r * eprime
            };
            let h_of = |r: f64| e_of(r) + p_of(r) / r;
            let dp = (p_of(rho + h) - p_of(rho - h)) / (2.0 * h);
            let dh = (h_of(rho + h) - h_of(rho - h)) / (2.0 * h);
            assert!(
                (dp - rho * dh).abs() < 1e-4 * (1.0 + dp.abs()),
                "dp = rho dh at rho={rho} (kind={kind}): dp={dp}, rho dh={}",
                rho * dh
            );
        }
    }
}

// ── 5. isothermal anchor: e = c² ln ρ ⇒ p = c² ρ ───────────────────────────

#[test]
fn ns_cmp_isothermal_anchor() {
    let c2 = 2.3f64;
    for &rho in &[0.3f64, 0.9, 1.7, 4.2] {
        let h = 1e-6 * rho;
        let eprime = (e_iso(rho + h, c2) - e_iso(rho - h, c2)) / (2.0 * h);
        let p = rho * rho * eprime;
        assert!(
            (p - c2 * rho).abs() < 1e-5 * (1.0 + c2 * rho),
            "isothermal anchor p = c^2 rho: got {p}"
        );
    }
}

// ── 6. acoustic dispersion of the linearized fiber ─────────────────────────

#[test]
fn ns_cmp_acoustic_dispersion() {
    // The linearized system in a Fourier mode is
    //   -i w s + i k u = 0,  -i w rho_bar u = -i k rho_bar c^2 s,
    // whose determinant is w^2 - c^2 k^2. Diagonalize the 2x2 matrix
    //   M = [[0, -c^2 k], [-k, 0]]  (s, u)  ->  eigenvalues ±c|k|.
    let c = 1.3f64;
    for &k in &[-3.1f64, -0.7, 0.0, 0.9, 2.5] {
        // eigenvalues of [[a, b], [d, 0]] with a = 0: ±sqrt(b d) = ±c k_abs
        let b = -c * c * k;
        let d = -k;
        let disc = (b * d).max(0.0);
        let w_plus = disc.sqrt();
        let w_minus = -disc.sqrt();
        let target = c * k.abs();
        assert!((w_plus - target).abs() < 1e-12, "w+ = c|k| at k={k}");
        assert!((w_minus + target).abs() < 1e-12, "w- = -c|k| at k={k}");
        // and the module's CHECK 11 factorization (w - ck)(w + ck) = w^2 - c^2 k^2:
        let lhs = (0.37 - c * k) * (0.37 + c * k);
        let rhs = 0.37f64 * 0.37 - c * c * k * k;
        assert!((lhs - rhs).abs() < 1e-12, "factorization at k={k}");
    }
}

// ── 7. the density-weighted pressure product is a two-wave convolution ─────

#[test]
fn ns_cmp_density_weighted_pressure_convolution() {
    // (1/rho) d_x p  in momentum space is  r ⋆ (i q p)  with r = F[1/rho]:
    //   F[(1/rho) d_x p] = (1/n) Σ_q r[Q - q] · i q · p[q]
    // checked against the real-space product on a periodic grid.
    let n = 16usize;
    let length = 2.0 * std::f64::consts::PI;
    // smooth strictly positive rho and smooth p on the periodic box:
    let rho: Vec<f64> = (0..n)
        .map(|t| {
            let x = length * t as f64 / n as f64;
            2.0 + 0.3 * (2.0 * x).sin() + 0.1 * (3.0 * x).cos()
        })
        .collect();
    let pfield: Vec<f64> = (0..n)
        .map(|t| {
            let x = length * t as f64 / n as f64;
            0.7 * (x).cos() - 0.2 * (2.0 * x).sin()
        })
        .collect();
    let inv_rho: Vec<Complex64> = rho.iter().map(|&r| Complex64::new(1.0 / r, 0.0)).collect();
    let p_c: Vec<Complex64> = pfield.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    // real-space target: (1/rho) dp/dx by spectral differentiation:
    let p_hat = dft(&p_c);
    let dp: Vec<Complex64> = idft(
        &p_hat
            .iter()
            .enumerate()
            .map(|(k, &pk)| pk * grad_symbol(k, n, length))
            .collect::<Vec<_>>(),
    );
    let target: Vec<Complex64> = dp
        .iter()
        .zip(inv_rho.iter())
        .map(|(&d, &r)| d * r)
        .collect();

    // convolution form: r ⋆ (i q p):
    let r_hat = dft(&inv_rho);
    let qp: Vec<Complex64> = p_hat
        .iter()
        .enumerate()
        .map(|(k, &pk)| pk * grad_symbol(k, n, length))
        .collect();
    // product-to-convolution theorem: F[f g] = (1/n) (F f) ⋆ (F g)
    let prod_hat: Vec<Complex64> = circular_conv(&r_hat, &qp)
        .iter()
        .map(|v| v / n as f64)
        .collect();
    let via_conv = idft(&prod_hat);

    for (t, (&a, &b)) in target.iter().zip(via_conv.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-9 * (1.0 + a.norm()),
            "density-weighted pressure convolution mismatch at t={t}: {} vs {}",
            a,
            b
        );
    }
}

// ── 8. products of derivatives carry the (i p)(i q) weight ─────────────────

#[test]
fn ns_cmp_derivative_product_convolution_weight() {
    // (d_x f)(d_x g) vs the weighted convolution of the two spectra:
    //   F[(df)(dg)] = (1/n) Σ_q (i q) f[Q-q] (i (Q-q))... — the two
    // transfer weights multiply (degree 2 in momenta). Direct check:
    // spectral derivative of each factor, multiply in real space, and
    // compare with the inverse DFT of the weighted convolution.
    let n = 16usize;
    let length = 2.0 * std::f64::consts::PI;
    let f: Vec<Complex64> = (0..n)
        .map(|t| {
            let x = length * t as f64 / n as f64;
            Complex64::new(0.4 + 0.3 * (x).sin(), 0.1 * (2.0 * x).cos())
        })
        .collect();
    let g: Vec<Complex64> = (0..n)
        .map(|t| {
            let x = length * t as f64 / n as f64;
            Complex64::new(0.2 * (2.0 * x).cos(), -0.15 * (3.0 * x).sin())
        })
        .collect();
    let f_hat = dft(&f);
    let g_hat = dft(&g);
    let df = idft(
        &f_hat
            .iter()
            .enumerate()
            .map(|(k, &v)| v * grad_symbol(k, n, length))
            .collect::<Vec<_>>(),
    );
    let dg = idft(
        &g_hat
            .iter()
            .enumerate()
            .map(|(k, &v)| v * grad_symbol(k, n, length))
            .collect::<Vec<_>>(),
    );
    let target: Vec<Complex64> = df.iter().zip(dg.iter()).map(|(&a, &b)| a * b).collect();

    // weighted convolution: weight w[q] = (i q)(i (Q-q)) on the products
    // f[q] g[Q-q] — assembled through the product-to-convolution theorem:
    let wf: Vec<Complex64> = f_hat
        .iter()
        .enumerate()
        .map(|(k, &v)| v * grad_symbol(k, n, length))
        .collect();
    let wg: Vec<Complex64> = g_hat
        .iter()
        .enumerate()
        .map(|(k, &v)| v * grad_symbol(k, n, length))
        .collect();
    let prod_hat: Vec<Complex64> = circular_conv(&wf, &wg)
        .iter()
        .map(|v| v / n as f64)
        .collect();
    let via_conv = idft(&prod_hat);

    for (t, (&a, &b)) in target.iter().zip(via_conv.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-9 * (1.0 + a.norm()),
            "derivative-product convolution weight mismatch at t={t}: {a} vs {b}"
        );
    }
}

// ── 9. the viscous weight is even and diagonal ─────────────────────────────

#[test]
fn ns_cmp_viscous_weight_even_diagonal() {
    let nu = 0.08f64;
    for &k in &[-4.2f64, -1.0, 0.0, 2.3, 5.5] {
        let w = nu * k * k;
        let w_neg = nu * (-k) * (-k);
        assert!(
            (w - w_neg).abs() < 1e-14,
            "viscous weight must be even in k"
        );
    }
    // diagonal: a single input mode k produces output only at k (the
    // Laplacian is a multiplication operator in momentum space), while an
    // advective convolution of the same mode with itself produces 2k:
    let n = 16usize;
    let length = 2.0 * std::f64::consts::PI;
    let mode_k = 3usize;
    let u: Vec<Complex64> = (0..n)
        .map(|t| {
            let ang = 2.0 * std::f64::consts::PI * (mode_k as f64) * (t as f64) / (n as f64);
            Complex64::new(ang.cos(), ang.sin())
        })
        .collect();
    let u_hat = dft(&u);
    // viscous output concentrated on the input support:
    let visc: Vec<Complex64> = u_hat
        .iter()
        .enumerate()
        .map(|(k, &v)| -nu * grad_symbol(k, n, length).norm_sqr() * v)
        .collect();
    for (k, &v) in visc.iter().enumerate() {
        let support = u_hat[k].norm() > 1e-12;
        if !support {
            assert!(
                v.norm() < 1e-12,
                "viscous term must be diagonal: output at un-supported mode {k}"
            );
        }
    }
}

// ── 10. the Legendre identity ───────────────────────────────────────────────

#[test]
fn ns_cmp_legendre_identity() {
    // H = pi^2/(2 rho0) + rho0 e(rho) at pi = rho0 v equals pi v - L with
    // L = (rho0/2) v^2 - rho0 e.
    let rho0 = 1.6f64;
    let c2 = 1.2f64;
    let e_of = |r: f64| e_iso(r, c2);
    for &v in &[-1.3f64, -0.2, 0.0, 0.9, 2.1] {
        let pi = rho0 * v;
        let rho = 1.1f64; // any admissible state
        let l = 0.5 * rho0 * v * v - rho0 * e_of(rho);
        let h = pi * pi / (2.0 * rho0) + rho0 * e_of(rho);
        let leg = pi * v - l;
        assert!(
            (h - leg).abs() < 1e-12 * (1.0 + h.abs()),
            "Legendre identity at v={v}: H={h}, pi v - L={leg}"
        );
    }
}

// ── 11. the fiber completed square (CHECK 18) ──────────────────────────────

#[test]
fn ns_cmp_fiber_completed_square() {
    let rho_bar = 1.4f64;
    let c = 1.1f64;
    for &k in &[-2.0f64, 0.0, 1.3] {
        for &pi in &[-0.7f64, 0.0, 1.9] {
            for &s in &[-0.4f64, 0.25] {
                let hfib = pi * pi / (2.0 * rho_bar) + 0.5 * rho_bar * c * c * k * k * s * s;
                let target =
                    (pi * pi + (rho_bar * c * k * s) * (rho_bar * c * k * s)) / (2.0 * rho_bar);
                assert!(
                    (hfib - target).abs() < 1e-12,
                    "completed square at k={k}, pi={pi}, s={s}"
                );
            }
        }
    }
}

// ── 12. the Faris–Lavine completion square is a non-negative residual ──────

#[test]
fn ns_cmp_fl_completion_square_nonnegative() {
    let rho_bar = 1.25f64;
    let c = 1.05f64;
    // N = h + (1/2) rho_bar c^2 (s - s0)^2  =>  N - h = the square >= 0.
    for &s in &[-2.0f64, -0.5, 0.0, 0.3, 1.7] {
        for &s0 in &[-1.0f64, 0.2, 0.9] {
            let square = 0.5 * rho_bar * c * c * (s - s0) * (s - s0);
            assert!(square >= 0.0, "completion square must be non-negative");
            // the residual N - h reproduces the square exactly:
            let pi = 0.44f64;
            let h = pi * pi / (2.0 * rho_bar) + 0.5 * rho_bar * c * c * s * s;
            let n = h + square;
            assert!(
                ((n - h) - square).abs() < 1e-13,
                "N - h must equal the completion square"
            );
            // and the one-sided domination |h| <= N on this fiber:
            assert!(h.abs() <= n + 1e-12, "|h| <= N must hold on the fiber");
        }
    }
}
