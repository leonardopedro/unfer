//! Numerical validation of the NS-L-C.3 row: the penalty + Lyapunov faces of
//! the Lagrangian comparison operator, and its lift `dΓ(N_L) + 1`.
//!
//! Companion to row **NS-L-C.3** of the TRANSFORMED-ESA work order
//! (`../timepiece/CONSOLIDATED_PLAN.md`, 2026-10-15): the final penalty +
//! Lyapunov row of `timepiece/BookProof/ChapterNsLagrangianDetFarisLavine.lean`
//! is recorded as conditional today and closes exactly when the **lifted**
//! comparison `dΓ(N_L) + 1` is essentially self-adjoint on the lifted core
//! (toolbox §5 — ESA is reproved on the lifted core, D6b).  What is checked
//! here is everything about that operator that is finite-dimensional and
//! falsifiable: the exact penalty, the energy and Liouville identities, the
//! pointwise Lyapunov bound, the comparison form identity `N = H² + E` with
//! `N ≥ 1`, the exact commutation relation, and the construction of the lift
//! on the Fock sectors (Hermiticity, form bound, the commutator identity
//! `[dΓ(H), dΓ(N)] = dΓ([H,N])`, and the resolvent estimate off the real
//! axis).  ESA of the lift itself remains the analytic row; no finite
//! truncation proves it, and nothing here claims it.
//!
//! **The model.**  The module's fluid is 3D with the exact momentum-
//! convolution determinant (`NsLagrangianDet.det_deformation_eq`), no
//! linearization.  The numerical model is its 2D-interaction face with the
//! same structure and the same conventions:
//!
//!   ξ(a) = (ξ₁ sin a₁, ξ₂ sin a₂) on 𝕋²,   F = I + ∇ξ,
//!   det F − 1 = ξ₁ cos a₁ + ξ₂ cos a₂ + ξ₁ξ₂ cos a₁ cos a₂
//!
//! — the exact nonlinear determinant (degree 2 here; the module's degree-5
//! constraint force comes from the third direction).  Its momentum
//! coefficients are `c_{±e₁} = ξ₁/2`, `c_{±e₂} = ξ₂/2`, `c_{±(1,±1)} = ξ₁ξ₂/4`,
//! and the volume penalty `V_κ = (κ/2) Σ_q |c_q|²` (the Parseval factor
//! absorbed in `κ`, as in the module) is the polynomial
//!
//!   V = (κ/4)(ξ₁² + ξ₂²) + (κ/8) ξ₁²ξ₂² ,
//!
//! checked below against an independent grid-DFT route.  Phase space is
//! `(ξ₁, ξ₂, v₁, v₂)` with `lam_j = |k_j|² = 1`, `Λ = Σ lam_j = 2`:
//!
//!   G = (v₁, v₂, −ν v₁ − ∂₁V, −ν v₂ − ∂₂V),   E = 1 + ½|v|² + V,
//!   flux = −ν Σ lam_j v_j²,   div G = −ν Σ lam_j = −2ν .
//!
//! The operator level mirrors `kvnGen_apply` / `lyapunovComparison`: on the
//! Gauss–polynomial core (vector `f = p e^{−|x|²/2}`, inner product
//! `∫ p q̄ e^{−|x|²} dx` — Gauss–Hermite quadrature, exact for the polynomial
//! integrands involved),
//!
//!   H p = −i [ Σ_i G_i (∂_i p − x_i p) + ½ (div G) p ],   N p = H(H p) + E p ,
//!
//! so `[H, E] = −i (G·∇E)` and `i[H, N] = flux` — the exact commutation
//! relation `lagComparison_commForm`.
//!
//! Tests 0–6 are the polynomial/quadrature level; tests 7–9 build the
//! Galerkin matrices on the tensor Hermite basis (`deg ≤ 1` per axis,
//! 16 states) and lift them to the Fock sectors `m = 0, 1, 2`
//! (dims 1 / 16 / 136) of the symmetric tensor power.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

const PI: f64 = std::f64::consts::PI;

// ── polynomial machinery (4 real variables, complex coefficients) ──────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Exp([u8; 4]);

#[derive(Clone, Debug)]
struct Poly {
    t: Vec<(Exp, Complex64)>,
}

impl Poly {
    fn zero() -> Poly {
        Poly { t: vec![] }
    }
    fn one() -> Poly {
        Poly {
            t: vec![(Exp([0; 4]), Complex64::new(1.0, 0.0))],
        }
    }
    fn term(e: [u8; 4], c: Complex64) -> Poly {
        Poly {
            t: vec![(Exp(e), c)],
        }
    }
    fn var(i: usize) -> Poly {
        let mut e = [0u8; 4];
        e[i] = 1;
        Poly::term(e, Complex64::new(1.0, 0.0))
    }
    fn re(c: f64) -> Poly {
        Poly::term([0; 4], Complex64::new(c, 0.0))
    }
    fn add(&self, other: &Poly) -> Poly {
        let mut t = self.t.clone();
        for (e, c) in &other.t {
            if let Some(slot) = t.iter_mut().find(|(f, _)| f == e) {
                slot.1 += *c;
            } else {
                t.push((*e, *c));
            }
        }
        t.retain(|(_, c)| c.norm_sqr() > 0.0);
        Poly { t }
    }
    fn scale(&self, s: f64) -> Poly {
        Poly {
            t: self.t.iter().map(|(e, c)| (*e, c * s)).collect(),
        }
    }
    fn mul(&self, other: &Poly) -> Poly {
        let mut out = Poly::zero();
        for (e1, c1) in &self.t {
            for (e2, c2) in &other.t {
                let mut e = [0u8; 4];
                for k in 0..4 {
                    e[k] = e1.0[k] + e2.0[k];
                }
                out = out.add(&Poly::term(e, c1 * c2));
            }
        }
        out
    }
    fn deriv(&self, i: usize) -> Poly {
        let mut out = Poly::zero();
        for (e, c) in &self.t {
            if e.0[i] == 0 {
                continue;
            }
            let mut f = e.0;
            let n = f[i];
            f[i] -= 1;
            out = out.add(&Poly::term(f, c * (n as f64)));
        }
        out
    }
    fn eval(&self, x: &[f64; 4]) -> Complex64 {
        let mut s = Complex64::new(0.0, 0.0);
        for (e, c) in &self.t {
            let mut p = *c;
            for k in 0..4 {
                for _ in 0..e.0[k] {
                    p *= x[k];
                }
            }
            s += p;
        }
        s
    }
}

// ── the model: penalty, vector field, energy, flux ─────────────────────────

/// The exact volume penalty of the model: `V = (κ/4)(ξ₁² + ξ₂²) + (κ/8)ξ₁²ξ₂²`
/// (closed form of `V_κ = (κ/2) Σ_q |volCoef_q|²`).
fn pot_v(x0: f64, x1: f64, kappa: f64) -> f64 {
    kappa / 4.0 * (x0 * x0 + x1 * x1) + kappa / 8.0 * x0 * x0 * x1 * x1
}

/// ∂V/∂ξ₁.
fn pot_d0(x0: f64, x1: f64, kappa: f64) -> f64 {
    kappa / 2.0 * x0 + kappa / 4.0 * x0 * x1 * x1
}

/// ∂V/∂ξ₂.
fn pot_d1(x0: f64, x1: f64, kappa: f64) -> f64 {
    kappa / 2.0 * x1 + kappa / 4.0 * x0 * x0 * x1
}

/// The Lagrangian vector field `G` (module `lagDrift`).
fn drift(z: &[f64; 4], nu: f64, kappa: f64) -> [f64; 4] {
    [
        z[2],
        z[3],
        -nu * z[2] - pot_d0(z[0], z[1], kappa),
        -nu * z[3] - pot_d1(z[0], z[1], kappa),
    ]
}

/// The Lagrangian energy `E = 1 + ½|v|² + V` (module `lagEnergy`).
fn energy(z: &[f64; 4], kappa: f64) -> f64 {
    1.0 + 0.5 * (z[2] * z[2] + z[3] * z[3]) + pot_v(z[0], z[1], kappa)
}

/// The viscous dissipation `flux = −ν Σ |k|² v²` (module `lagFlux`).
fn flux(z: &[f64; 4], nu: f64) -> f64 {
    -nu * (z[2] * z[2] + z[3] * z[3])
}

/// The volume penalty `(κ/2) Σ_q |ĉ_q|²` from the grid DFT of `det F − 1`
/// (the momentum-convolution route of `NsLagrangianDet.volCoef`), computed
/// independently of the closed form.
fn dft_penalty(det_f: &dyn Fn(f64, f64) -> f64, kappa: f64, n: usize) -> f64 {
    let mut v = 0.0f64;
    for q1 in 0..n {
        for q2 in 0..n {
            let mut c = Complex64::new(0.0, 0.0);
            for i in 0..n {
                for j in 0..n {
                    let a1 = 2.0 * PI * i as f64 / n as f64;
                    let a2 = 2.0 * PI * j as f64 / n as f64;
                    let ph = q1 as f64 * a1 + q2 as f64 * a2;
                    c += det_f(a1, a2) * Complex64::new(ph.cos(), -ph.sin());
                }
            }
            let c = c / (n * n) as f64;
            v += c.norm_sqr();
        }
    }
    kappa / 2.0 * v
}

// ── polynomial forms of the same data (operator level) ─────────────────────

struct Model {
    g: [Poly; 4],
    e: Poly,
    flux: Poly,
    div_g: f64, // = −ν Σ lam_j = −2ν, constant
}

fn model(nu: f64, kappa: f64) -> Model {
    let x: Vec<Poly> = (0..4).map(Poly::var).collect();
    // V = (κ/4)(x0² + x1²) + (κ/8)x0²x1²
    let v = x[0]
        .mul(&x[0])
        .scale(kappa / 4.0)
        .add(&x[1].mul(&x[1]).scale(kappa / 4.0))
        .add(&x[0].mul(&x[0]).mul(&x[1].mul(&x[1])).scale(kappa / 8.0));
    let d0 = x[0]
        .scale(kappa / 2.0)
        .add(&x[0].mul(&x[1].mul(&x[1])).scale(kappa / 4.0));
    let d1 = x[1]
        .scale(kappa / 2.0)
        .add(&x[0].mul(&x[0]).mul(&x[1]).scale(kappa / 4.0));
    let g = [
        x[2].clone(),
        x[3].clone(),
        x[2].scale(-nu).add(&d0.scale(-1.0)),
        x[3].scale(-nu).add(&d1.scale(-1.0)),
    ];
    let e = Poly::re(1.0)
        .add(&x[2].mul(&x[2]).add(&x[3].mul(&x[3])).scale(0.5))
        .add(&v);
    let flux_p = x[2].mul(&x[2]).add(&x[3].mul(&x[3])).scale(-nu);
    Model {
        g,
        e,
        flux: flux_p,
        div_g: -2.0 * nu,
    }
}

/// `H p = −i [ Σ G_i (∂_i p − x_i p) + ½ (div G) p ]` — `kvnGen_apply` on the
/// Gaussian core vector `p e^{−|x|²/2}` (the Weyl-ordered KvN generator).
fn h_act(m: &Model, p: &Poly) -> Poly {
    let mut acc = Poly::zero();
    for i in 0..4 {
        let xi = Poly::var(i);
        let d = p.deriv(i).add(&xi.mul(p).scale(-1.0));
        acc = acc.add(&m.g[i].mul(&d));
    }
    acc = acc.add(&p.scale(0.5 * m.div_g));
    // multiply by −i
    Poly {
        t: acc
            .t
            .iter()
            .map(|(e, c)| (*e, *c * Complex64::new(0.0, -1.0)))
            .collect(),
    }
}

/// `N p = H(H p) + E p` — the comparison `lyapunovComparison G E`.
fn n_act(m: &Model, p: &Poly) -> Poly {
    h_act(m, &h_act(m, p)).add(&m.e.mul(p))
}

// ── Gauss–Hermite quadrature (weight e^{−x²}) ─────────────────────────────

/// Physicists' Hermite polynomial `H_n(x)` by recurrence.
fn hermite(n: usize, x: f64) -> f64 {
    let (mut h, mut h_prev) = (1.0f64, 0.0f64);
    for j in 1..=n {
        let h_next = h;
        h = 2.0 * x * h - 2.0 * (j - 1) as f64 * h_prev;
        h_prev = h_next;
    }
    h
}

/// Nodes and weights of the n-point Gauss–Hermite rule (weight e^{−x²}).
/// Nodes by Newton on `H_n`, weights from
/// `w_i = 2^{n−1} n! √π / (n² H_{n−1}(x_i)²)`.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1 && n <= 24);
    let mut nodes = vec![0.0f64; n];
    let mut weights = vec![0.0f64; n];
    let mut log_two_pow = 0.0f64; // ln(2^{n-1} n!)
    for j in 1..=n {
        log_two_pow += (j as f64).ln();
    }
    log_two_pow += (n as f64 - 1.0) * 2f64.ln();
    let ln_sqrt_pi = 0.5 * PI.ln();
    // Initial guesses, largest root first (Numerical Recipes `gauher`): the
    // naive asymptotic cos-guesses can collapse several indices onto one
    // root, which silently corrupts the weights.
    let mut z = 0.0f64;
    for i in 0..n {
        z = if i == 0 {
            (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-1.0 / 6.0)
        } else if i == 1 {
            z - 1.14 * (n as f64).powf(0.426) / z
        } else if i == 2 {
            1.86 * z - 0.86 * nodes[0]
        } else if i == 3 {
            1.91 * z - 0.91 * nodes[1]
        } else {
            2.0 * z - nodes[i - 2]
        };
        // Newton on H_n, with H_n' = 2n H_{n−1}
        let mut converged = false;
        for _ in 0..100 {
            let hn = hermite(n, z);
            let dhn = 2.0 * n as f64 * hermite(n - 1, z);
            let z1 = z - hn / dhn;
            if (z1 - z).abs() < 1e-15 * (1.0 + z.abs()) {
                z = z1;
                converged = true;
                break;
            }
            z = z1;
        }
        assert!(converged, "Hermite root {i} of {n} did not converge");
        let hnm1 = hermite(n - 1, z);
        nodes[i] = z;
        weights[i] = (log_two_pow + ln_sqrt_pi).exp() / ((n as f64 * n as f64) * hnm1 * hnm1);
    }
    (nodes, weights)
}

/// Tensor Gauss–Hermite expectation `∫ p q̄ e^{−|x|²} dx` in 4 dimensions.
fn gpair(p: &Poly, q: &Poly, nodes: &[f64], weights: &[f64]) -> Complex64 {
    let n = nodes.len();
    let mut acc = Complex64::new(0.0, 0.0);
    for i0 in 0..n {
        for i1 in 0..n {
            for i2 in 0..n {
                for i3 in 0..n {
                    let x = [nodes[i0], nodes[i1], nodes[i2], nodes[i3]];
                    let w = weights[i0] * weights[i1] * weights[i2] * weights[i3];
                    let pv = p.eval(&x);
                    let qv = q.eval(&x);
                    acc += pv * qv.conj() * w;
                }
            }
        }
    }
    acc
}

fn norm2(p: &Poly, nodes: &[f64], weights: &[f64]) -> f64 {
    gpair(p, p, nodes, weights).re
}

// ── small deterministic RNG (as in the companion suite) ────────────────────

fn rng(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
}

/// A random polynomial state of total degree ≤ 3 in 4 variables.
fn random_state(seed: &mut u64) -> Poly {
    let mut p = Poly::zero();
    for _ in 0..5 {
        let e = loop {
            let mut e = [0u8; 4];
            let mut total = 0u8;
            for k in 0..4 {
                e[k] = (rng(seed).abs() * 1.6) as u8 % 3;
                total += e[k];
            }
            if total <= 3 {
                break e;
            }
        };
        let c = Complex64::new(rng(seed), rng(seed));
        p = p.add(&Poly::term(e, c));
    }
    p
}

// ═════════════════════════════════════════════════════════════════════════
// 0. machinery: the quadrature must be exact before anything is trusted
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_gauss_hermite_quadrature_exact() {
    // ∫ x^{2m} e^{−x²} dx = Γ(m + 1/2); the n-point rule is exact for
    // m ≤ n − 1 (degree ≤ 2n−1).
    let (nodes, weights) = gauss_hermite(10);
    let gamma = |m: f64| -> f64 {
        // Γ(m + 1/2) for half-integers via the recurrence Γ(t+1) = t Γ(t),
        // Γ(1/2) = √π
        let mut val = PI.sqrt();
        let mut t = 0.5;
        while t < m + 0.5 - 1e-12 {
            val *= t;
            t += 1.0;
        }
        val
    };
    for m in 0..10usize {
        let exact = gamma(m as f64);
        let got: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(&x, &w)| w * x.powi(2 * m as i32))
            .sum();
        assert!(
            (got - exact).abs() < 1e-12 * (1.0 + exact),
            "moment {m}: {got} vs {exact}"
        );
    }
    // Σ w = √π
    let s: f64 = weights.iter().sum();
    assert!((s - PI.sqrt()).abs() < 1e-12, "weights sum: {s}");
}

// ═════════════════════════════════════════════════════════════════════════
// 1. the penalty: non-negative, zero set = {det F = 1}
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_penalty_nonneg_and_zero_set() {
    // `volPot` = (κ/2) Σ_q |volCoef_q|² is non-negative at every real
    // configuration (`volPot_eval_nonneg`), and for κ > 0 its zero set is
    // incompressible (`det_eq_one_of_volPot_eq_zero`): V = 0 forces
    // det F ≡ 1.  Two independent routes to V:
    //   (a) the closed form (κ/4)(ξ₁² + ξ₂²) + (κ/8)ξ₁²ξ₂²;
    //   (b) a grid DFT of the exact nonlinear det F − 1 (16×16 grid over
    //       𝕋², all Fourier coefficients, Parseval sum).
    let kappa = 2.3f64;
    let n = 16usize;
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..24 {
        let x0 = rng(&mut seed);
        let x1 = rng(&mut seed);
        // (a) closed form
        let v_closed = pot_v(x0, x1, kappa);
        // (b) grid DFT of the exact nonlinear det F − 1 =
        // ξ₁ cos a₁ + ξ₂ cos a₂ + ξ₁ξ₂ cos a₁ cos a₂
        let v_dft = dft_penalty(
            &|a1, a2| (1.0 + x0 * a1.cos()) * (1.0 + x1 * a2.cos()) - 1.0,
            kappa,
            n,
        );
        assert!(
            (v_closed - v_dft).abs() < 1e-10 * (1.0 + v_closed),
            "penalty mismatch: closed {v_closed} vs DFT {v_dft}"
        );
        assert!(v_closed >= 0.0, "penalty must be non-negative");
    }
    // Zero set (`det_eq_one_of_volPot_eq_zero`), through the DFT route with
    // the determinant assembled from the deformation gradient itself: a
    // shear ξ = (γ sin a₂, 0) has ∇ξ = [[0, γ cos a₂],[0, 0]], so
    // det F = 1·1 − (γ cos a₂)·0 = 1 exactly — det F − 1 vanishes on the
    // grid, every momentum coefficient is 0 and V = 0 although ξ ≠ 0.  A
    // longitudinal mode is compressible and pays.  (The converse direction
    // V = 0 ⇒ det F ≡ 1 is the injectivity of the momentum transform: all
    // coefficients vanish only for det F − 1 ≡ 0.)
    let gamma = 1.7f64;
    let det_shear = |_a1: f64, a2: f64| -> f64 {
        let grad = [[0.0f64, gamma * a2.cos()], [0.0, 0.0]]; // ∇ξ rows
        (1.0 + grad[0][0]) * (1.0 + grad[1][1]) - grad[0][1] * grad[1][0] - 1.0
    };
    let v_shear = dft_penalty(&det_shear, kappa, n);
    assert!(
        v_shear < 1e-20,
        "shear must be exactly incompressible: V = {v_shear}"
    );
    let x0 = 0.8f64;
    let v_long = dft_penalty(&|a1, _a2| x0 * a1.cos(), kappa, n);
    assert!(
        v_long > 1e-6,
        "compressible configuration must pay the penalty"
    );
    // the closed form agrees on the same longitudinal configuration
    assert!((v_long - pot_v(x0, 0.0, kappa)).abs() < 1e-10 * (1.0 + v_long));
    // the leading behaviour V(εξ) = ε² (κ/4)(ξ₁² + ξ₂²) + O(ε⁴)
    let eps = 1e-3f64;
    let lead = eps * eps * kappa / 4.0 * (x0 * x0);
    let got = pot_v(eps * x0, 0.0, kappa);
    assert!(
        (got - lead).abs() < 1e-6 * lead,
        "leading order mismatch: {got} vs {lead}"
    );
    // V is linear in κ (bulk modulus)
    let v1 = pot_v(0.8, -0.3, 1.0);
    let v2 = pot_v(0.8, -0.3, 5.0);
    assert!(
        (5.0 * v1 - v2).abs() < 1e-12 * (1.0 + v2),
        "κ-linearity failed"
    );
}

// ═════════════════════════════════════════════════════════════════════════
// 2. the energy identity: the penalty force does no net work
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_energy_identity_penalty_no_net_work() {
    // `lagFlux_eq`: along the Galerkin flow, F·∇E = −ν Σ |k|² v².  The
    // penalty force is the gradient of the potential part of E and drops
    // out: E(t) = 1 + ½|v|² + V(ξ) loses exactly the viscous dissipation.
    // Checked along RK4 trajectories: central differences of E(t) must
    // match the dissipation evaluated at the midpoint state.  A penalty
    // force with a wrong sign, or one that does net work, separates them.
    let nu = 0.35f64;
    let kappa = 2.0f64;
    let h = 1e-3f64;
    let steps = 400usize;
    let mut seed = 0xFEED_FACE_0011_2233u64;
    for _ in 0..4 {
        let mut z = [0.0f64; 4];
        for v in &mut z {
            *v = rng(&mut seed);
        }
        let mut zs = vec![z];
        for _ in 0..steps {
            // RK4 on ż = G(z)
            let k1 = drift(&z, nu, kappa);
            let mut z2 = z;
            for i in 0..4 {
                z2[i] += 0.5 * h * k1[i];
            }
            let k2 = drift(&z2, nu, kappa);
            let mut z3 = z;
            for i in 0..4 {
                z3[i] += 0.5 * h * k2[i];
            }
            let k3 = drift(&z3, nu, kappa);
            let mut z4 = z;
            for i in 0..4 {
                z4[i] += h * k3[i];
            }
            let k4 = drift(&z4, nu, kappa);
            for i in 0..4 {
                z[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            zs.push(z);
        }
        let e_hist: Vec<f64> = zs.iter().map(|s| energy(s, kappa)).collect();
        // trajectory-level identity: central differences of E(t) along the
        // integrated path match the dissipation at the interior states
        for k in 1..steps {
            let dedt = (e_hist[k + 1] - e_hist[k - 1]) / (2.0 * h);
            let diss = flux(&zs[k], nu);
            assert!(
                (dedt - diss).abs() < 1e-5 * (1.0 + diss.abs()),
                "trajectory energy identity failed at step {k}: dE/dt = {dedt} vs {diss}"
            );
        }
        // point-level identity at random states (independent of the
        // integration): F·∇E by central differences of E along the flow
        // direction must equal the dissipation pointwise.
        for _ in 0..64 {
            let mut p = [0.0f64; 4];
            for v in &mut p {
                *v = rng(&mut seed) * 2.0;
            }
            let g = drift(&p, nu, kappa);
            let eps = 1e-6;
            let mut fde = 0.0f64;
            for i in 0..4 {
                let mut pp = p;
                let mut pm = p;
                pp[i] += eps;
                pm[i] -= eps;
                fde += g[i] * (energy(&pp, kappa) - energy(&pm, kappa)) / (2.0 * eps);
            }
            let diss = flux(&p, nu);
            assert!(
                (fde - diss).abs() < 1e-5 * (1.0 + diss.abs()),
                "energy identity failed: F·∇E = {fde} vs flux = {diss}"
            );
        }
        // and the history-level statement: E is not conserved, it decays by
        // the accumulated dissipation (ν > 0)
        assert!(
            e_hist[e_hist.len() - 1] < e_hist[0] + 1e-12,
            "E must not increase along the viscous flow"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 3. the Liouville identity: div F = −ν Σ |k|², constant
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_liouville_divergence() {
    // `lagDiv_eq`: the divergence of the vector field is the constant
    // −ν Σ_j lam_j = −2ν here — the penalty force contributes nothing (it
    // has no velocity dependence).  Central differences of each component
    // against its own coordinate.
    let nu = 0.6f64;
    let kappa = 1.4f64;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..32 {
        let mut z = [0.0f64; 4];
        for v in &mut z {
            *v = rng(&mut seed) * 2.0;
        }
        let eps = 1e-6;
        let mut div = 0.0f64;
        for i in 0..4 {
            let mut zp = z;
            let mut zm = z;
            zp[i] += eps;
            zm[i] -= eps;
            let fp = drift(&zp, nu, kappa);
            let fm = drift(&zm, nu, kappa);
            div += (fp[i] - fm[i]) / (2.0 * eps);
        }
        let target = -nu * 2.0; // −ν Σ lam_j
        assert!(
            (div - target).abs() < 1e-5,
            "div F = {div}, expected {target}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 4. the pointwise Lyapunov bound |F·∇E| ≤ 2νΛ E
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_pointwise_lyapunov_bound() {
    // `abs_eval_lagFlux_le` with Λ = Σ lam_j = 2: |F·∇E| ≤ 2νΛ E at every
    // real point.  F·∇E is computed independently (central differences of E
    // against the flow direction), then both the identity F·∇E = −νΣlam v²
    // and the bound are checked.  The point ξ = v = 0 (E = 1, flux = 0) is
    // included; near it the bound is tightest relative to E.
    let nu = 0.9f64;
    let kappa = 3.0f64;
    let lam_sum = 2.0f64; // Λ
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut cases: Vec<[f64; 4]> = vec![[0.0; 4], [1.0, -1.0, 0.5, -0.5]];
    for _ in 0..200 {
        let mut z = [0.0f64; 4];
        for v in &mut z {
            *v = rng(&mut seed) * 3.0;
        }
        cases.push(z);
    }
    for z in cases {
        let g = drift(&z, nu, kappa);
        let eps = 1e-6;
        let mut fde = 0.0f64;
        for i in 0..4 {
            let mut zp = z;
            let mut zm = z;
            zp[i] += eps;
            zm[i] -= eps;
            fde += g[i] * (energy(&zp, kappa) - energy(&zm, kappa)) / (2.0 * eps);
        }
        let diss = flux(&z, nu);
        assert!(
            (fde - diss).abs() < 1e-4 * (1.0 + diss.abs()),
            "F·∇E = {fde} vs −νΣlam v² = {diss}"
        );
        let e = energy(&z, kappa);
        assert!(
            fde.abs() <= 2.0 * nu * lam_sum * e * (1.0 + 1e-9),
            "Lyapunov bound violated: |{fde}| > 2νΛ · {e}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 5. the comparison form identity: ⟨ψ, N ψ⟩ = ‖H ψ‖² + ⟨ψ, E ψ⟩ ≥ ‖ψ‖²
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_comparison_form_identity_and_ge_one() {
    // `lyapunovComparison_quadForm` + `lagComparison_quadForm_ge` +
    // `lagComparison_relBound`, at the Gauss–polynomial level with exact
    // quadrature.  Two independent assemblies of ⟨ψ, N ψ⟩:
    //   (a) via N = H² + E applied to the polynomial state;
    //   (b) as ‖Hψ‖² + ⟨ψ, Eψ⟩ (which is what proves N ≥ 1, since E ≥ 1).
    // Their equality is the symmetry face of the Weyl-ordered H
    // (`lagKoopmanOp_symmetricOn`): a wrong ordering or a dropped div/2
    // term separates them.  Also: N ≥ 1, and ‖Hψ‖ ≤ ‖Nψ‖ + ‖ψ‖.
    let nu = 0.4f64;
    let kappa = 1.5f64;
    let m = model(nu, kappa);
    let (nodes, weights) = gauss_hermite(10);
    let mut seed = 0x1319_8A2E_0370_7344u64;
    for _ in 0..8 {
        let p = random_state(&mut seed);
        let hp = h_act(&m, &p);
        let np = n_act(&m, &p);
        let npsi_a = gpair(&p, &np, &nodes, &weights).re;
        let npsi_b = norm2(&hp, &nodes, &weights) + gpair(&p, &m.e.mul(&p), &nodes, &weights).re;
        assert!(
            (npsi_a - npsi_b).abs() < 1e-9 * (1.0 + npsi_a.abs()),
            "form identity failed: ⟨ψ,Nψ⟩ = {npsi_a} vs ‖Hψ‖² + ⟨E⟩ = {npsi_b}"
        );
        // N ≥ 1 (lagComparison_quadForm_ge): E ≥ 1 pointwise
        let pnorm2 = norm2(&p, &nodes, &weights);
        assert!(
            npsi_a >= pnorm2 - 1e-9 * (1.0 + pnorm2),
            "N ≥ 1 violated: ⟨N⟩ = {npsi_a} < ‖ψ‖² = {pnorm2}"
        );
        // the relative bound ‖Hψ‖ ≤ ‖Nψ‖ + ‖ψ‖ (lagComparison_relBound)
        let nhp = norm2(&np, &nodes, &weights).sqrt();
        let hpn = norm2(&hp, &nodes, &weights).sqrt();
        let pn = pnorm2.sqrt();
        assert!(
            hpn <= nhp + pn + 1e-8 * (1.0 + nhp + pn),
            "rel bound violated: ‖Hψ‖ = {hpn} > {nhp} + {pn}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 6. the exact commutation relation and the Faris–Lavine commutator bound
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn pl_commutator_form_identity_and_bound() {
    // `lagComparison_commForm` / `lagComparison_commForm_bound`:
    //   ⟨ψ, i[H,N] ψ⟩ = ⟨ψ, flux ψ⟩   and   |⟨ψ, i[H,N] ψ⟩| ≤ 2νΛ ⟨ψ, N ψ⟩.
    // The commutator is assembled by applying N∘H − H∘N to the state (two
    // independent operator applications), the right side by the dissipation
    // polynomial.  Since N = H² + E and [H, H²] = 0, the commutator must
    // reduce to [H, E] = −i(G·∇E) — the penalty drops out by the energy
    // identity, which is exactly the module's point.
    let nu = 0.5f64;
    let kappa = 2.2f64;
    let lam_sum = 2.0f64;
    let m = model(nu, kappa);
    let (nodes, weights) = gauss_hermite(10);
    let mut seed = 0x6A09_E667_F3BC_C908u64;
    for _ in 0..8 {
        let p = random_state(&mut seed);
        // i[H, N] ψ = i ( H(Nψ) − N(Hψ) )
        let hn_p = h_act(&m, &n_act(&m, &p));
        let nh_p = n_act(&m, &h_act(&m, &p));
        let comm = Poly {
            t: hn_p
                .add(&nh_p.scale(-1.0))
                .t
                .iter()
                .map(|(e, c)| (*e, *c * Complex64::new(0.0, 1.0)))
                .collect(),
        };
        let lhs = gpair(&p, &comm, &nodes, &weights).re;
        let rhs = gpair(&p, &m.flux.mul(&p), &nodes, &weights).re;
        assert!(
            (lhs - rhs).abs() < 1e-9 * (1.0 + rhs.abs()),
            "commutator identity failed: ⟨i[H,N]⟩ = {lhs} vs ⟨flux⟩ = {rhs}"
        );
        // the Faris–Lavine commutator bound against ⟨N⟩
        let n_expect = gpair(&p, &n_act(&m, &p), &nodes, &weights).re;
        assert!(
            lhs.abs() <= 2.0 * nu * lam_sum * n_expect * (1.0 + 1e-9),
            "commutator bound violated: |{lhs}| > 2νΛ ⟨N⟩ = {}",
            2.0 * nu * lam_sum * n_expect
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 7–9. the lifted comparison dΓ(N_L) + 1 on the Fock sectors
// ═════════════════════════════════════════════════════════════════════════

/// Galerkin matrices of H and E on the tensor Hermite basis
/// (`H_0 = 1`, `H_1(x) = 2x` per axis, 16 states), by exact quadrature.
fn galerkin(nu: f64, kappa: f64) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let m = model(nu, kappa);
    let (nodes, weights) = gauss_hermite(6);
    let n1 = 16usize;
    // basis polys: b_α = Π_j H_{α_j}(x_j), α ∈ {0,1}⁴
    let basis: Vec<Poly> = (0..n1)
        .map(|idx| {
            let mut p = Poly::one();
            for j in 0..4 {
                if (idx >> j) & 1 == 1 {
                    // H_1(x_j) = 2 x_j
                    let mut e = [0u8; 4];
                    e[j] = 1;
                    p = p.mul(&Poly::term(e, Complex64::new(2.0, 0.0)));
                }
            }
            p
        })
        .collect();
    let mut h = DMatrix::<Complex64>::zeros(n1, n1);
    let mut em = DMatrix::<Complex64>::zeros(n1, n1);
    for (n, bn) in basis.iter().enumerate() {
        let hbn = h_act(&m, bn);
        let ebn = m.e.mul(bn);
        for (mi, bm) in basis.iter().enumerate() {
            h[(mi, n)] = gpair(bm, &hbn, &nodes, &weights);
            em[(mi, n)] = gpair(bm, &ebn, &nodes, &weights);
        }
    }
    (h, em)
}

/// The one-particle comparison matrix `N₁ = H₁² + E₁` (the module's
/// `lyapunovComparison`), for the Galerkin truncation.
fn comparison_matrix(nu: f64, kappa: f64) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let (h, em) = galerkin(nu, kappa);
    let n1 = &h * &h + &em;
    (h, n1)
}

/// `dΓ(A)` on the m-particle sector of the symmetric tensor power, built
/// from the one-particle matrix `A` (`m = 0` is the vacuum).  Returns the
/// sector matrix of `dΓ(A) + 1` and of `dΓ(A)` for the given `m`.
fn dgamma_sector(a: &DMatrix<Complex64>, m: usize) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let n1 = a.nrows();
    if m == 0 {
        let one = DMatrix::<Complex64>::from_element(1, 1, Complex64::new(1.0, 0.0));
        let zero = DMatrix::<Complex64>::zeros(1, 1);
        return (zero, one); // (dΓ(A), dΓ(A) + 1)
    }
    if m == 1 {
        return (a.clone(), a.clone() + DMatrix::identity(n1, n1));
    }
    assert!(m == 2);
    // full two-particle operator A⊗I + I⊗A on ℂ^{n1} ⊗ ℂ^{n1}
    let id = DMatrix::<Complex64>::identity(n1, n1);
    let full = a.kronecker(&id) + id.kronecker(a);
    // symmetric subspace basis: (e_a ⊗ e_b + e_b ⊗ e_a)/norm, a ≤ b
    let n2 = n1 * (n1 + 1) / 2;
    let mut p = DMatrix::<Complex64>::zeros(n1 * n1, n2);
    let mut col = 0;
    for a_i in 0..n1 {
        for b_i in a_i..n1 {
            let norm = if a_i == b_i { 1.0 } else { 2.0f64.sqrt() };
            p[(a_i * n1 + b_i, col)] = Complex64::new(1.0 / norm, 0.0);
            if a_i != b_i {
                p[(b_i * n1 + a_i, col)] = Complex64::new(1.0 / norm, 0.0);
            }
            col += 1;
        }
    }
    let sector = &p.transpose() * &full * &p;
    let ident = DMatrix::<Complex64>::identity(n2, n2);
    (sector.clone(), sector + ident)
}

/// Max entry modulus of `A − Aᴴ` — the Hermiticity residual.
fn herm_residual(a: &DMatrix<Complex64>) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            worst = worst.max((a[(i, j)] - a[(j, i)].conj()).norm());
        }
    }
    worst
}

/// Smallest eigenvalue of a Hermitian complex matrix (via the real
/// embedding `[[Re, −Im],[Im, Re]]`, whose spectrum is the spectrum of `A`
/// doubled).
fn min_eig_herm(a: &DMatrix<Complex64>) -> f64 {
    let n = a.nrows();
    let mut s = DMatrix::<f64>::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            s[(i, j)] = a[(i, j)].re;
            s[(i, n + j)] = -a[(i, j)].im;
            s[(n + i, j)] = a[(i, j)].im;
            s[(n + i, n + j)] = a[(i, j)].re;
        }
    }
    let eig = s.symmetric_eigenvalues();
    eig.iter().cloned().fold(f64::INFINITY, f64::min)
}

#[test]
fn pl_lifted_comparison_fock_sectors() {
    // The NS-L-C.3 target's finite shadow: the lifted comparison
    // `dΓ(N_L) + 1` on the lifted core (toolbox §5).  Built on the Fock
    // sectors m = 0, 1, 2 over the Galerkin one-particle matrices:
    // every sector matrix is Hermitian (`lagComparison_symmetricOn` lifts),
    // and the form bound N ≥ 1 lifts to `dΓ(N) + 1 ≥ m + 1` (each
    // particle contributes ≥ 1 and the +1 is the vacuum shift).  A botched
    // symmetrization or a missing tensor leg fails both faces.
    let nu = 0.3f64;
    let kappa = 1.8f64;
    let (_h, n1) = comparison_matrix(nu, kappa);
    // the one-particle comparison is Hermitian and ≥ 1
    assert!(herm_residual(&n1) < 1e-10, "N₁ not Hermitian");
    let m1 = min_eig_herm(&n1);
    assert!(m1 >= 1.0 - 1e-8, "N₁ ≥ 1 violated: min eig {m1}");
    for m in 0..3usize {
        let (dg, shifted) = dgamma_sector(&n1, m);
        assert!(
            herm_residual(&shifted) < 1e-9,
            "dΓ(N)+1 not Hermitian on sector {m}"
        );
        let lo = min_eig_herm(&shifted);
        assert!(
            lo >= (m as f64 + 1.0) - 1e-8,
            "sector {m}: min eig {lo} < m+1 = {}",
            m as f64 + 1.0
        );
        // vacuum: dΓ(N) = 0, so dΓ(N)+1 = 1 exactly
        if m == 0 {
            assert!((dg[(0, 0)]).norm() < 1e-14, "vacuum must be zero");
            assert!((shifted[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-14);
        }
    }
}

#[test]
fn pl_lifted_commutator_identity() {
    // The structural identity that makes the lifted Faris–Lavine row
    // meaningful: `[dΓ(H), dΓ(N)] = dΓ([H,N])`, checked as matrices on the
    // two-particle sector.  Both sides are assembled independently: the
    // left from the sector matrices of `dΓ(H)` and `dΓ(N)`, the right from
    // the one-particle commutator `[H₁, N₁] = [H₁, E₁]` (N = H² + E and
    // [H, H²] = 0) lifted through the same projection.  This is a wiring
    // check of the symmetrized tensor assembly — the point of the row
    // "ESA is reproved on the lifted core".
    let nu = 0.3f64;
    let kappa = 1.8f64;
    let (h1, n1) = comparison_matrix(nu, kappa);
    let (dh2, _) = dgamma_sector(&h1, 2);
    let (dn2, _) = dgamma_sector(&n1, 2);
    let comm_lifted = &dh2 * &dn2 - &dn2 * &dh2;
    let comm_one = &h1 * &n1 - &n1 * &h1;
    let (dcomm, _) = dgamma_sector(&comm_one, 2);
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..comm_lifted.nrows() {
        for j in 0..comm_lifted.ncols() {
            worst = worst.max((comm_lifted[(i, j)] - dcomm[(i, j)]).norm());
            scale = scale.max(comm_lifted[(i, j)].norm());
        }
    }
    assert!(
        worst < 1e-9 * (1.0 + scale),
        "lifted commutator identity failed: residual {worst} at scale {scale}"
    );
    // and the one-particle face [H, N] = [H, E] = −i flux·(mult):
    // the commutator matrix must equal −i × (matrix of flux × ...)
    // — checked here through its antisymmetry structure: i[H,N] is
    // Hermitian (flux is a real multiplication operator).
    let ihne = comm_one.map(|v| v * Complex64::new(0.0, 1.0));
    assert!(herm_residual(&ihne) < 1e-9, "i[H₁,N₁] must be Hermitian");
}

#[test]
fn pl_lifted_deficiency_resolvent_bound() {
    // The finite-dimensional shadow of "the lifted comparison is ESA on the
    // lifted core": a self-adjoint operator satisfies the resolvent estimate
    // ‖(A − z)⁻¹‖ ≤ 1/|Im z| for Im z ≠ 0 — precisely the numerical content
    // of trivial deficiency at z.  Checked on `A = dΓ(N_L) + 1` in the
    // two-particle sector by solving `(A − z)x = ψ` with an LU factorization
    // and comparing norms.  (A non-Hermitian assembly would violate the
    // estimate for some z and ψ; the analytic ESA of the lift remains the
    // open row.)
    let nu = 0.3f64;
    let kappa = 1.8f64;
    let (_h, n1) = comparison_matrix(nu, kappa);
    let (_, shifted) = dgamma_sector(&n1, 2);
    let n = shifted.nrows();
    assert!(
        herm_residual(&shifted) < 1e-9,
        "A must be self-adjoint to have the bound"
    );
    let mut seed = 0x5A17_2B3C_4D5E_6F70u64;
    for &(re, im) in &[(0.0f64, 0.5f64), (0.7, 1.0), (-2.0, 2.5), (1.0, -0.25)] {
        let z = Complex64::new(re, im);
        let a_shift = &shifted
            - DMatrix::from_fn(
                n,
                n,
                |i, j| {
                    if i == j { z } else { Complex64::new(0.0, 0.0) }
                },
            );
        let lu = a_shift.clone().lu();
        for _ in 0..8 {
            let psi = DVector::from_fn(n, |_, _| Complex64::new(rng(&mut seed), rng(&mut seed)));
            let np = psi.norm();
            match lu.solve(&psi) {
                Some(x) => {
                    assert!(
                        x.norm() <= np / im.abs() * (1.0 + 1e-7),
                        "resolvent bound violated at z={z}: ‖x‖ = {} > ‖ψ‖/|Im z| = {}",
                        x.norm(),
                        np / im.abs()
                    );
                }
                None => panic!("singular resolvent at z={z} — deficiency at a non-real point!"),
            }
        }
    }
}
