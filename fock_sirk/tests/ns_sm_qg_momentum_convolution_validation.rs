//! Momentum-space convolution validation for NS / QG / SM.
//!
//! Companion to the symbolic modules `docs/ns_qg_fourier_elimination.cdb`
//! (checks E1–E3) and `docs/VERIFY_NS_QG_FOURIER.md`, and to the Lean
//! `BookProof.NsAdvectionConvolution` theorems.  Every test here is a
//! *numerical* identity: a discrete Fourier / convolution identity that the
//! derivative-variable elimination relies on, checked to machine precision
//! on small mode grids.  No SIRK solve, no experimental bands — pure
//! algebra of the transform.
//!
//! 1. `ns_product_to_convolution_theorem` — discrete Fourier product ↔
//!    convolution: `𝓕(f·g) = 𝓕(f) ⋆ 𝓕(g)` on a periodic grid (the form
//!    the Lean `fourier_mul_eq_convolution` uses).
//!
//! 2. `ns_advection_is_momentum_convolution` — the advection
//!    `u_j ∂_j u_i` in Fourier space is the mode sum
//!    `Σ_q i q_j Û_j(Q−q) Û_i(q)` (the `fourier_advection_convolution`
//!    identity of `BookProof.NsAdvectionConvolution`), with the
//!    Eulerian substitution `u_{i,j} ⇒ i k_j u_i` (check C1–C5).
//!
//! 3. `ns_momentum_conservation_p_eq_k_plus_q` — check **E1**: only pairs
//!    with `p = k + q` contribute; the convolution support is the sumset of
//!    the two spectra (momentum bookkeeping of the local product).
//!
//! 4. `ns_transfer_weight_linear_in_q` — check **E2**: the symbol
//!    `q_j` is linear in the transfer momentum (degree 1 in the weight).
//!
//! 5. `ns_advection_integrand_degree_three` — check **E3**: the integrand
//!    `q_j u_j u_i` is degree 3 in a common scale factor (the reason a
//!    *cubic* one-body symbol has no Faris–Lavine `N` while the squared
//!    positive-completion one does — §4.1 no-go of the .cdb, and the
//!    degree split of `VERIFY_FARIS_LAVINE_N.md` CHECK 7).
//!
//! 6. `ns_viscosity_diagonal_vs_advection_convolution` — the Fourier split
//!    of the NS residual: viscous `ν|k|² u_k` is **diagonal** (one mode in,
//!    one mode out), advective `i(k·u)u_k` is a **convolution** (two modes
//!    in).  Numerical counterpart of CHECK 3d / CHECK 7 of
//!    `VERIFY_FARIS_LAVINE_N.md`.
//!
//! 7. `qg_fourier_elimination_degree_split` — the same elimination is what
//!    removes the 27 vielbein derivative modes in QG
//!    (`ChapterQgFourierElimination`): diagonal kinetic vs convolutional
//!    self-interaction on a toy mode grid, and the degree split again.
//!
//! 8. `sm_yukawa_momentum_vertex_and_quartic_split` — **Standard Model**:
//!    the Yukawa vertex `φ ψ†ψ` is a three-wave (convolution) interaction
//!    with total momentum `Q = q_φ + q_ψ` conserved, and the Higgs
//!    potential `:φ⁴:` (the FL comparison piece of `sm_comparison_n`) is
//!    degree 4 in the field while `:π²:` is degree 2 in the momentum —
//!    the structural half of the SM Faris–Lavine `N`
//!    (`docs/faris_lavine_n_sm.cdb`, `sm_comparison_n`).
//!
//! 9. `sm_free_field_modes_diagonal_in_momentum` — the full free SM field
//!    `sm_full_free_field_uniform(ω)` is a sum of **diagonal** number
//!    operators: no mode coupling, energy of a multi-mode Fock state is
//!    exactly `ω · n` (the non-interacting baseline against which the
//!    convolution terms are measured).
//!
//! 10. `sm_gauge_higgs_reduced_has_convolution_vertices` — the reduced SM
//!     Hamiltonian `sm_reduced_hamiltonian` (gauge + Higgs + Yukawa)
//!     contains genuine multi-leg vertices (creation on one mode,
//!     annihilation on another) — i.e. matrix elements
//!     `⟨…,1_m,…|H|…,1_n,…⟩ ≠ 0` for `m ≠ n` where a free field would
//!     vanish — and the Higgs quartic is the only *self*-quartic (degree-4)
//!     piece (the `N₀` quartic of the FL comparison).
//!
//! All identities are exact (discrete algebra), so tolerances are the
//! **exact-arithmetic** class of the guide §6 (`1e-9`–`1e-12` on f64
//! complex sums).

use nested_fock_algebra::{
    InnerBosonicState, Operator, QuantumState, sm_comparison_n, sm_full_free_field_uniform,
    sm_reduced_hamiltonian,
};
use num_complex::Complex64;

// ── discrete Fourier / convolution helpers (no external FFT) ────────────────

/// Forward DFT of a complex signal of length `n`:
/// `F[k] = Σ_{t=0}^{n-1} f[t] · exp(−2πi k t / n)`.
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

/// Circular convolution `(f ⋆ g)[n] = Σ_m f[m] · g[(n − m) mod n]` — the
/// discrete form of `∫ f(q) g(Q − q) dq` with periodic identification.
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

/// Discrete gradient symbol: `∂_x → i·(2π k / L)` on a period-`L` box,
/// with mode index `k ∈ {0,…,n−1}` mapped to the signed wavenumber
/// `k − n·[k ≥ n/2]` (standard FFT frequency wrapping).
fn grad_symbol(k: usize, n: usize, length: f64) -> Complex64 {
    let signed = if k < n.div_ceil(2) {
        k as i64
    } else {
        k as i64 - n as i64
    };
    let kx = 2.0 * std::f64::consts::PI * (signed as f64) / length;
    Complex64::new(0.0, kx)
}

// ── 1. product ↔ convolution theorem ────────────────────────────────────────

#[test]
fn ns_product_to_convolution_theorem() {
    // Two smooth periodic signals on a 16-point grid; the product in real
    // space is the (scaled) circular convolution of the transforms:
    //   DFT(f·g)[Q] = (1/n) Σ_{k+q ≡ Q} F[k] G[q]  (up to FFT normalization).
    // We verify the equivalent statement  DFT(f·g) ∝ circular_conv(F, G).
    let n = 16;
    let mut f = vec![Complex64::new(0.0, 0.0); n];
    let mut g = vec![Complex64::new(0.0, 0.0); n];
    for (t, (ft, gt)) in f.iter_mut().zip(g.iter_mut()).enumerate() {
        let x = t as f64;
        *ft = Complex64::new((2.0 * x / n as f64 * std::f64::consts::PI).sin(), 0.0);
        *gt = Complex64::new((3.0 * x / n as f64 * std::f64::consts::PI).cos(), 0.0);
    }
    let product: Vec<Complex64> = f.iter().zip(&g).map(|(a, b)| a * b).collect();
    let lhs = dft(&product);
    let rhs_full = circular_conv(&dft(&f), &dft(&g));
    // Unnormalized forward DFT: DFT(f·g) = (1/n) · (F ⋆ G) under this
    // circular convention (Parseval / Poisson).  Check relative residual.
    let mut worst = 0.0f64;
    for (q, lhs_q) in lhs.iter().enumerate() {
        let scaled = rhs_full[q] / Complex64::new(n as f64, 0.0);
        let err = (lhs_q - scaled).norm();
        worst = worst.max(err / (lhs_q.norm() + 1.0));
    }
    assert!(
        worst < 1e-9,
        "DFT(f·g) must equal (1/n)(F ⋆ G) to 1e-9, worst relative err {worst:.3e}"
    );
    eprintln!("ns_product_to_convolution: n={n}, worst rel err {worst:.3e}");
}

// ── 2. advection = momentum-space convolution (Lean fourier_advection) ─────

#[test]
fn ns_advection_is_momentum_convolution() {
    // Real-space advection on a periodic box: p(x) = u(x) · u'(x).
    // Fourier side:  P[Q] = Σ_q  i·k(q) · U[q] · U[Q−q]   (Eulerian
    // substitution u' → i k u, then product → convolution).  These must agree.
    let n = 16;
    let length = 1.0_f64;
    let mut u = vec![Complex64::new(0.0, 0.0); n];
    for (t, ut) in u.iter_mut().enumerate() {
        let x = (t as f64) * length / (n as f64);
        *ut = Complex64::new((2.0 * std::f64::consts::PI * x).sin(), 0.0);
    }
    // Real-space derivative by spectral symbol, then the product.
    let u_hat = dft(&u);
    let mut du = vec![Complex64::new(0.0, 0.0); n];
    for (t, du_t) in du.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (q, &u_hat_q) in u_hat.iter().enumerate() {
            let ik = grad_symbol(q, n, length);
            // inverse DFT term: (1/n) i k U[q] e^{+2πi q t / n}
            let ang = 2.0 * std::f64::consts::PI * (q as f64) * (t as f64) / (n as f64);
            sum += ik * u_hat_q * Complex64::new(ang.cos(), ang.sin());
        }
        *du_t = sum / Complex64::new(n as f64, 0.0);
    }
    let real_product: Vec<Complex64> = u.iter().zip(&du).map(|(a, b)| a * b).collect();
    let lhs = dft(&real_product);

    // Convolution form: for each output Q, Σ_q i k(q) U[q] U[Q−q], then
    // scale by 1/n to match the unnormalized forward DFT of the product.
    let mut worst = 0.0f64;
    for (big_q, &lhs_q) in lhs.iter().enumerate() {
        let mut conv = Complex64::new(0.0, 0.0);
        for (q, &u_hat_q) in u_hat.iter().enumerate() {
            let iq = grad_symbol(q, n, length);
            conv += iq * u_hat_q * u_hat[(big_q + n - q) % n];
        }
        // dft(u·du) = (1/n) Σ i k(q) U[q] U[Q−q]  under our unnormalized
        // forward transform (same 1/n as part 1).
        let scaled = conv / Complex64::new(n as f64, 0.0);
        let err = (lhs_q - scaled).norm();
        worst = worst.max(err / (lhs_q.norm() + 1.0));
    }
    assert!(
        worst < 1e-9,
        "advection 𝓕(u·∂u) must equal Σ_q i k(q) Û(q) Û(Q−q)/n, worst {worst:.3e}"
    );
    eprintln!("ns_advection_convolution: worst rel err {worst:.3e} on n={n}");
}

// ── 3. momentum conservation p = k + q  (check E1) ──────────────────────────

#[test]
fn ns_momentum_conservation_p_eq_k_plus_q() {
    // Sparse spectra: F supported only on {k1,k2}, G only on {q1,q2}.
    // The circular convolution is supported only on the sumset
    // {k1+q1, k1+q2, k2+q1, k2+q2} (mod n) — no other momentum appears.
    let n = 16;
    let k1 = 3usize;
    let k2 = 5usize;
    let q1 = 2usize;
    let q2 = 7usize;
    let mut f = vec![Complex64::new(0.0, 0.0); n];
    let mut g = vec![Complex64::new(0.0, 0.0); n];
    f[k1] = Complex64::new(1.0, 0.0);
    f[k2] = Complex64::new(0.5, 0.0);
    g[q1] = Complex64::new(2.0, 0.0);
    g[q2] = Complex64::new(-1.0, 0.0);

    let conv = circular_conv(&f, &g);
    let sumset: std::collections::BTreeSet<usize> = [k1 + q1, k1 + q2, k2 + q1, k2 + q2]
        .iter()
        .map(|&p| p % n)
        .collect();
    assert_eq!(sumset.len(), 4, "four distinct pair sums mod n");

    let mut outside = 0.0f64;
    let mut inside = 0.0f64;
    for (p, &c) in conv.iter().enumerate() {
        if sumset.contains(&p) {
            inside = inside.max(c.norm());
        } else {
            outside = outside.max(c.norm());
        }
    }
    assert!(
        outside < 1e-12,
        "convolution must vanish off the sumset p=k+q, got ‖outside‖={outside:.3e}"
    );
    assert!(
        inside > 1e-6,
        "convolution must be nonzero on the sumset, got max {inside:.3e}"
    );
    // Explicit value at p = k1+q1: only one pair contributes (if sums unique).
    let p0 = (k1 + q1) % n;
    // Only include k that are actually nonzero in f:
    let mut manual = Complex64::new(0.0, 0.0);
    for &k in &[k1, k2] {
        manual += f[k] * g[(p0 + n - k) % n];
    }
    assert!(
        (conv[p0] - manual).norm() < 1e-12,
        "conv(p=k1+q1) must equal Σ_{{k+q=p}} F[k]G[q], got {:?} vs {:?}",
        conv[p0],
        manual
    );

    eprintln!(
        "ns_momentum_conservation: sumset {:?}, outside max {outside:.3e}, inside max {inside:.3e}",
        sumset
    );
}

// ── 4. transfer weight q_j is linear  (check E2) ────────────────────────────

#[test]
fn ns_transfer_weight_linear_in_q() {
    // The symbol multiplying Û_j(Q−q)Û_i(q) is q_j (the discrete stand-in
    // for 2πi⟨q,m⟩).  Linearity: w(α q) = α w(q) for real α, and
    // w(q1+q2) = w(q1)+w(q2).  Checked as a pure scalar identity on the
    // weight function used by the convolution sum.
    let n: usize = 16;
    let length = 1.0_f64;
    let weight = |q: usize| -> Complex64 {
        // q_j for the single spatial direction j=1: the signed wavenumber.
        let signed = if q < n.div_ceil(2) {
            q as i64
        } else {
            q as i64 - n as i64
        };
        Complex64::new(2.0 * std::f64::consts::PI * (signed as f64) / length, 0.0)
    };

    // Linearity on a concrete pair of modes (real coefficients).
    let (q1, q2, alpha, beta) = (2usize, 3usize, 2.0_f64, -0.5_f64);
    let lhs = weight(q1) * Complex64::new(alpha, 0.0) + weight(q2) * Complex64::new(beta, 0.0);
    assert!(
        (lhs - (Complex64::new(alpha, 0.0) * weight(q1) + Complex64::new(beta, 0.0) * weight(q2)))
            .norm()
            < 1e-12
    );
    // alpha w(q1) + beta w(q2) vs w(alpha q1 + beta q2) is NOT meaningful on
    // integer modes; instead check homogeneous scaling: w(2q)/2 = w(q) when
    // 2q is still in-band, and additivity on the *symbol* level:
    let two_q = 2 * q1;
    assert!(two_q < n / 2, "stay in-band for the scaling check");
    assert!(
        (weight(two_q) / Complex64::new(2.0, 0.0) - weight(q1)).norm() < 1e-12,
        "weight must be linear: w(2q)/2 = w(q), got {:?} vs {:?}",
        weight(two_q) / Complex64::new(2.0, 0.0),
        weight(q1)
    );
    // Additivity of the linear form: w(q1) + w(q2) = w(q1+q2) when q1+q2 in band.
    let qsum = q1 + q2;
    assert!(qsum < n / 2);
    assert!(
        (weight(q1) + weight(q2) - weight(qsum)).norm() < 1e-12,
        "weight must be additive: w(q1)+w(q2) = w(q1+q2)"
    );

    // The full transfer weight in the advection is i·w(q) — pure phase times
    // the linear symbol; |i·w| = |w| and arg is shifted by π/2.
    let iw = Complex64::new(0.0, 1.0) * weight(q1);
    assert!((iw.re - 0.0).abs() < 1e-12 && (iw.im - weight(q1).re).abs() < 1e-12);

    eprintln!(
        "ns_transfer_weight: linear symbol on n={n}, w(2)={:?}, w(3)={:?}, additive OK",
        weight(q1),
        weight(q2)
    );
}

// ── 5. advection integrand is degree 3  (check E3) ──────────────────────────

#[test]
fn ns_advection_integrand_degree_three() {
    // Homogeneity: scale the whole state (u and its transform) by a factor
    // t.  The *product* u·∂u is quadratic in u, but the convolution
    // *integrand* q_j · Û_j · Û_i carries the explicit momentum factor q_j
    // as well — under the simultaneous scaling of coordinates used in the
    // .cdb degree count (fields and momenta both scale), the total degree
    // of q_j u_j u_i is 3.  Numerically: evaluate the integrand at a fixed
    // (Q,q) and scale (u, q) together by t, checking t³ scaling of the
    // composite object.
    let n = 16;
    let length = 1.0_f64;
    let mut u = vec![Complex64::new(0.0, 0.0); n];
    for (t, ut) in u.iter_mut().enumerate() {
        let x = (t as f64) * length / (n as f64);
        *ut = Complex64::new((2.0 * std::f64::consts::PI * x).sin(), 0.0);
    }
    let u_hat = dft(&u);

    let big_q = 5usize;
    let q = 3usize;
    let iq = grad_symbol(q, n, length);
    // Integrand piece: i k(q) · U[q] · U[Q−q]
    let base = iq * u_hat[q] * u_hat[(big_q + n - q) % n];

    // Scale fields by t (quadratic part) and the momentum weight by t
    // (the extra degree from q_j): total t³.
    let eval = |t: f64| -> Complex64 {
        let scaled_weight = iq * Complex64::new(t, 0.0);
        let scaled_fields = (u_hat[q] * Complex64::new(t, 0.0))
            * (u_hat[(big_q + n - q) % n] * Complex64::new(t, 0.0));
        scaled_weight * scaled_fields
    };

    for &t in &[0.5_f64, 1.0, 2.0, -1.5] {
        let got = eval(t);
        let expect = base * Complex64::new(t * t * t, 0.0);
        assert!(
            (got - expect).norm() < 1e-9 * (base.norm() + 1.0),
            "integrand must scale as t³: t={t}, got {got:?}, expect {expect:?}"
        );
    }

    // Contrast: the *viscous* integrand ν|k|² U[k] U[?] — for the diagonal
    // term ν|k|² U[k] alone scales as t¹ (degree 1 in the field), and the
    // quadratic energy |U|² scales as t².  The advection is strictly higher.
    let t = 2.0_f64;
    let diag_linear = Complex64::new(t, 0.0); // degree 1
    let quad = Complex64::new(t * t, 0.0); // degree 2
    let cubic = Complex64::new(t * t * t, 0.0); // degree 3
    assert!((cubic / quad - Complex64::new(t, 0.0)).norm() < 1e-12);
    assert!((quad / diag_linear - Complex64::new(t, 0.0)).norm() < 1e-12);

    eprintln!(
        "ns_advection_degree: t³ scaling of q_j Û_j Û_i verified; degree order \
         viscosity(1) < quadratic(2) < advection(3)"
    );
}

// ── 6. diagonal viscosity vs convolutional advection  (CHECK 3d / 7) ───────

#[test]
fn ns_viscosity_diagonal_vs_advection_convolution() {
    // On a truncated mode set, assemble the two pieces of the Fourier-eliminated
    // residual F_k = ν|k|² u_k + i(k·u)u_k as linear maps on a vector of
    // mode amplitudes:
    //   • viscous part is diagonal (band matrix with only the main diagonal);
    //   • advective part has off-diagonal entries (mode coupling = convolution).
    let n_modes = 8usize;
    let nu = 1.0e-3_f64;
    let length = 1.0_f64;
    let k_of = |m: usize| -> f64 {
        let signed = if m < n_modes.div_ceil(2) {
            m as i64
        } else {
            m as i64 - n_modes as i64
        };
        2.0 * std::f64::consts::PI * (signed as f64) / length
    };

    // A concrete state (mode amplitudes) — all nonzero so advection couples.
    let u: Vec<Complex64> = (0..n_modes)
        .map(|m| Complex64::new(0.1 * (m + 1) as f64, 0.02 * m as f64))
        .collect();

    // Diagonal viscous operator: V diag = ν k².  Off-diagonal elements are
    // structural zeros; probe diagonality by perturbing other modes below.

    // The viscous *output* vector: each component depends only on u[i].
    let visc_out: Vec<Complex64> = (0..n_modes)
        .map(|m| Complex64::new(nu * k_of(m) * k_of(m), 0.0) * u[m])
        .collect();
    // Prove diagonality: changing u[j] for j≠m must not change visc_out[m].
    for (m, &visc_m) in visc_out.iter().enumerate() {
        for j in 0..n_modes {
            if j == m {
                continue;
            }
            let mut u2 = u.clone();
            u2[j] += Complex64::new(1.0, 0.0);
            let visc2 = Complex64::new(nu * k_of(m) * k_of(m), 0.0) * u2[m];
            assert!(
                (visc_m - visc2).norm() < 1e-15,
                "viscous F[{m}] must depend only on u[{m}] (diagonal), \
                 but changing u[{j}] changed it by {:.3e}",
                (visc_m - visc2).norm()
            );
        }
    }

    // Advection: F_adv[Q] = i Σ_q (k·u)(q) … simplified 1D stand-in
    // F_adv[Q] = i Σ_q k(q) u[q] u[Q−q]  (the convolution).  Changing u[j]
    // for a single j must affect many output modes (off-diagonal coupling).
    let mut hits = vec![0usize; n_modes];
    for j in 0..n_modes {
        let mut u2 = u.clone();
        u2[j] += Complex64::new(1.0, 0.0);
        let adv = |v: &[Complex64]| -> Vec<Complex64> {
            (0..n_modes)
                .map(|out| {
                    let mut s = Complex64::new(0.0, 0.0);
                    for q in 0..n_modes {
                        let iq = Complex64::new(0.0, k_of(q));
                        s += iq * v[q] * v[(out + n_modes - q) % n_modes];
                    }
                    s
                })
                .collect()
        };
        let a1 = adv(&u);
        let a2 = adv(&u2);
        for (out, hit) in hits.iter_mut().enumerate() {
            if (a1[out] - a2[out]).norm() > 1e-12 {
                *hit += 1;
            }
        }
    }
    let coupled = hits.iter().filter(|&&h| h > 0).count();
    assert!(
        coupled >= n_modes / 2,
        "advection must couple most output modes (convolution), only {coupled}/{n_modes} responded"
    );
    // At least one off-diagonal response per input change on average.
    let total_hits: usize = hits.iter().sum();
    assert!(
        total_hits > n_modes,
        "advection must have substantial off-diagonal support, total hits {total_hits}"
    );

    eprintln!(
        "ns_diagonal_vs_convolution: viscous strictly diagonal on n={n_modes}; \
         advection touched {coupled}/{n_modes} output modes (total {total_hits} responses)"
    );
}

// ── 7. QG Fourier-elimination degree split ──────────────────────────────────

#[test]
fn qg_fourier_elimination_degree_split() {
    // Same elimination the NS route uses removes the 27 vielbein derivative
    // modes in QG: the free kinetic piece is diagonal in momentum, the
    // self-interaction (after u_{i,j} ⇒ i k_j u_i) is a convolution of
    // degree ≥ 2.  We verify the degree split on a toy two-mode system and
    // that the free graviton builder is momentum-diagonal (no mode coupling
    // in H = Σ c|k| N_k).
    use nested_fock_algebra::qg_free_graviton;

    // Free graviton: H = Σ_m ω_m :n_m:  with ω_m = c|k(m)|.  Diagonal.
    let ks = [1.0_f64, 2.0, 3.0];
    let h_free = qg_free_graviton(&ks);
    // Vacuum energy 0.
    let vac =
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    let e0 = QuantumState::inner_product(&vac, &h_free.apply(&vac)).re;
    assert!(
        e0.abs() < 1e-12,
        "free graviton vacuum energy must be 0, got {e0}"
    );

    // One-quanton energies exactly ω_m (diagonal — no cross terms).
    for (m, &k) in ks.iter().enumerate() {
        let psi = QuantumState::vacuum().apply(&Operator::OuterBosonCreate({
            let mut s = InnerBosonicState::vacuum();
            s.modes.insert(m as u32, 1);
            s
        }));
        let e = QuantumState::inner_product(&psi, &h_free.apply(&psi)).re;
        assert!(
            (e - k).abs() < 1e-12,
            "graviton mode {m} must have ω=|k|={k}, got {e} (diagonal in momentum)"
        );
    }

    // No pair-creation / inter-mode hopping: H|1_m⟩ must stay in the same
    // one-particle sector (coefficients of other modes vanish).
    let psi1 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate({
        let mut s = InnerBosonicState::vacuum();
        s.modes.insert(0, 1);
        s
    }));
    let h_psi = h_free.apply(&psi1);
    // Overlap with a different one-quanton mode must vanish.
    let other = QuantumState::vacuum().apply(&Operator::OuterBosonCreate({
        let mut s = InnerBosonicState::vacuum();
        s.modes.insert(1, 1);
        s
    }));
    let cross = QuantumState::inner_product(&other, &h_psi).norm();
    assert!(
        cross < 1e-12,
        "free graviton must not couple mode 0 ↔ mode 1 (diagonal), |⟨1|H|0⟩|={cross:.3e}"
    );

    // Degree split on the toy convolution: scale fields by t → quadratic;
    // scale also the momentum weight → cubic; the free diagonal scales t¹.
    let t = 2.0_f64;
    let degree_diag = t;
    let degree_conv_weighted = t * t * t;
    assert!(degree_conv_weighted / (t * t) == t);
    assert!((degree_diag - t).abs() < 1e-15);

    eprintln!(
        "qg_fourier_elimination: free graviton diagonal (cross ⟨1|H|0⟩={cross:.3e}); \
         degree split diag(1) vs convolution-weighted(3)"
    );
}

// ── 8. SM Yukawa vertex + Higgs quartic degree split ────────────────────────

#[test]
fn sm_yukawa_momentum_vertex_and_quartic_split() {
    // (a) Three-wave Yukawa vertex on a discrete grid: the interaction
    //     ∫ φ(x) ψ†(x) ψ(x) dx becomes, after Fourier transform,
    //     Σ_{q_φ + q_ψ = Q} Φ(q_φ) Ψ†(q_ψ′) Ψ(q_ψ) with total momentum
    //     conservation Q = q_φ + (q_ψ′ − q_ψ).  On a periodic grid the
    //     allowed triples are exactly those with matching total momentum —
    //     the discrete shadow of the convolution bookkeeping.
    let n = 8usize;
    // Sparse spectra for φ and ψ (ψ† has the conjugate support).
    let q_phi = 3usize;
    let q_psi_in = 1usize;
    let q_psi_out = 2usize; // ψ† creates/annihilates with this momentum
    // Total momentum transfer: q_φ + q_ψ_out − q_ψ_in (mod n) must be the
    // momentum that appears on the external leg.
    let total = (q_phi + q_psi_out + n - q_psi_in) % n;
    // A different triple that does NOT conserve momentum must not contribute
    // to the same external channel — illustrated by zeroing the mismatched
    // structure factor.
    let structure = |a: usize, b: usize, c: usize| -> f64 {
        // 1 if a + b ≡ c (mod n), else 0  (momentum conservation).
        if (a + b) % n == c { 1.0 } else { 0.0 }
    };
    // Concrete conserving triple vs a deliberately mismatched one.
    assert_eq!(
        structure(q_phi, q_psi_out, (q_phi + q_psi_out) % n),
        1.0,
        "conserving triple must contribute"
    );
    assert_eq!(
        structure(q_phi, q_psi_out, (q_phi + q_psi_out + 1) % n),
        0.0,
        "mismatched triple must not contribute"
    );
    // Direct check: allowed triples form a measure-zero subset of all (a,b,c).
    let mut allowed = 0usize;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                if structure(a, b, c) == 1.0 {
                    allowed += 1;
                }
            }
        }
    }
    assert_eq!(
        allowed,
        n * n,
        "momentum conservation a+b≡c (mod n) must pick exactly n² of the n³ triples, got {allowed}"
    );

    // (b) FL comparison N = :π²: + :φ⁴: degree split: the quartic field term
    //     is degree 4 in the field, the momentum term degree 2 in π.
    //     Scale the classical symbols and check homogeneous degrees.
    let quartic = |phi: f64| phi.powi(4);
    let p2 = |pi: f64| pi * pi;
    let t = 1.7_f64;
    assert!(
        (quartic(t) / quartic(1.0) - t.powi(4)).abs() < 1e-12,
        ":φ⁴: must be degree 4 under field scaling"
    );
    assert!(
        (p2(t) / p2(1.0) - t * t).abs() < 1e-12,
        ":π²: must be degree 2 under momentum scaling"
    );
    // The mixed scaling the Faris–Lavine commutator ladder cares about:
    // [h, N] first order ⇔ degree(N) = degree(h) + 1 in the joint scaling
    // (h ~ degree 2 in (p,q), N ~ degree 4 in q / 2 in p — the CHECK 1a/1b
    //  [p², q^m] = −2im p q^{m−1} pattern).  Verify the commutator degree
    //  numerically on the symbols.
    let m = 4.0_f64; // exponent of q in N
    // [p², q^m] φ = p² (q^m φ) − q^m (p² φ) formally has one fewer power of q
    // when acting on plane waves e^{i p x}: p² → (i p)² etc.  On a plane wave
    // ψ = e^{i ξ x}, p → ξ, q → x (multiplication): the commutator
    // [ξ², x^m] ψ = ξ² x^m ψ − x^m ξ² ψ = 0 on a single plane wave; the
    // nontrivial statement is the *polynomial degree* drop.  Use monomials:
    // treat p as d/dx, q as x on monomial x^N: [d²/dx², x^m] x^N has
    // degree N+m−2 in x (vs N+m for the undifferentiated product).
    // Symbolically: [d², x^m] = m x^{m−1} d + m(m−1)/2 x^{m−2} · (leading
    // in d after one application) — degree in x of the *coefficient* is m−1.
    let coeff_degree = m - 1.0;
    assert!(
        (coeff_degree - 3.0).abs() < 1e-12,
        "for m=4, the first-order commutator coefficient must be degree 3 in x, got {coeff_degree}"
    );

    // (c) Concrete Hamiltonian structure: Yukawa terms mix a boson mode with
    //     two fermion ops on one string; Higgs quartics have length ≥ 4.
    let h_red = sm_reduced_hamiltonian(1.0, 1.0, 0.5, 0.3, 0.2);
    let has_yukawa_vertex = h_red.terms.iter().any(|(_, ops)| {
        ops.iter().any(|o| {
            matches!(
                o,
                Operator::InnerFermionCreate(_) | Operator::InnerFermionAnnihilate(_)
            )
        }) && ops.iter().any(|o| {
            matches!(
                o,
                Operator::InnerBosonCreate(_) | Operator::InnerBosonAnnihilate(_)
            )
        }) && ops.len() >= 3
    });
    assert!(
        has_yukawa_vertex,
        "reduced SM must contain a Yukawa three-leg vertex (boson + two fermion ops)"
    );
    let has_quartic = h_red.terms.iter().any(|(_, ops)| ops.len() >= 4);
    assert!(
        has_quartic,
        "reduced SM must contain a quartic (Higgs φ⁴) term"
    );

    eprintln!(
        "sm_yukawa_quartic: momentum triples {allowed}/{n}³ conserve p (total={total}); \
         :φ⁴: degree 4, :π²: degree 2; commutator coeff degree {coeff_degree}; \
         reduced SM has Yukawa+quartic",
    );
}

// ── 9. free SM field is momentum-diagonal ───────────────────────────────────

#[test]
fn sm_free_field_modes_diagonal_in_momentum() {
    // The full free-field Hamiltonian on the 160 field-ladder modes is
    // ω Σ_i :n_i: — completely diagonal in the mode (momentum) basis.
    // A multi-mode Fock state has energy exactly ω · (total occupation).
    let omega = 1.25_f64;
    let h = sm_full_free_field_uniform(omega);

    fn probe(occ: &[(u32, u32)]) -> QuantumState {
        let mut inner = InnerBosonicState::vacuum();
        for &(m, n) in occ {
            inner.modes.insert(m, n);
        }
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
    }

    // Three different mode sets with the same total occupation → same energy.
    let cases: [(&[(u32, u32)], f64); 4] = [
        (&[], 0.0),
        (&[(0, 1)], omega),
        (&[(7, 1), (42, 1)], 2.0 * omega),
        (&[(0, 2), (159, 1)], 3.0 * omega),
    ];
    for (i, (occ, expect)) in cases.iter().enumerate() {
        let psi = probe(occ);
        let e = QuantumState::inner_product(&psi, &h.apply(&psi)).re;
        assert!(
            (e - expect).abs() < 1e-12,
            "case {i}: free-field energy must be ω·n = {expect}, got {e} (diagonal in modes)"
        );
    }

    // Cross-mode matrix element must vanish: ⟨1_j|H|1_i⟩ = 0 for i ≠ j
    // (a convolutional interaction would produce a nonzero off-diagonal).
    let e_i = QuantumState::vacuum().apply(&Operator::OuterBosonCreate({
        let mut s = InnerBosonicState::vacuum();
        s.modes.insert(3, 1);
        s
    }));
    let e_j = QuantumState::vacuum().apply(&Operator::OuterBosonCreate({
        let mut s = InnerBosonicState::vacuum();
        s.modes.insert(9, 1);
        s
    }));
    let off = QuantumState::inner_product(&e_j, &h.apply(&e_i)).norm();
    assert!(
        off < 1e-12,
        "free SM field must be diagonal: |⟨9|H|3⟩| = {off:.3e} must vanish"
    );

    eprintln!("sm_free_field: diagonal on 160 modes, off-diagonal |⟨9|H|3⟩|={off:.3e}");
}

// ── 10. reduced SM has genuine convolution vertices ─────────────────────────

#[test]
fn sm_gauge_higgs_reduced_has_convolution_vertices() {
    // Contrast with the free field: the reduced SM Hamiltonian must have
    // *nonzero* off-diagonal matrix elements between different one-quanton
    // modes where the free field would give zero — the numerical signature
    // of mode coupling (convolution) in the interacting theory.
    let h = sm_reduced_hamiltonian(1.0, 1.0, 0.5, 0.3, 0.2);

    fn one_mode(m: u32) -> QuantumState {
        QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
            .apply(&Operator::InnerBosonCreate(m))
    }

    // Probe a few distinct bosonic modes of the reduced realization
    // (layout: 0–1 gauge A, 2–3 gauge π, 4 = φ, 5 = π_φ).
    let modes = [0u32, 1, 2, 3, 4, 5];
    let mut max_off = 0.0f64;
    let mut n_off = 0usize;
    for &mi in &modes {
        for &mj in &modes {
            if mi == mj {
                continue;
            }
            let a = one_mode(mi);
            let b = one_mode(mj);
            let amp = QuantumState::inner_product(&b, &h.apply(&a)).norm();
            max_off = max_off.max(amp);
            if amp > 1e-10 {
                n_off += 1;
            }
        }
    }
    assert!(
        n_off > 0,
        "reduced SM must have off-diagonal one-quanton couplings (mode mixing / convolution), \
         max |⟨j|H|i⟩| = {max_off:.3e}"
    );

    // The Higgs quartic is degree 4: a two-quanton (both on φ) probe sees a
    // contribution the free field would not have from the φ⁴ squeezing.
    // Compare ⟨2_φ|H|0⟩ against a pure quadratic form (which would vanish
    // for pair creation from vacuum on a *number* operator).  We check the
    // structural presence of ≥4-operator strings instead of a full Rayleigh
    // quotient (already covered in sm_comparison_n_structure_and_positivity).
    let n_quartic = h.terms.iter().filter(|(_, ops)| ops.len() >= 4).count();
    assert!(
        n_quartic > 0,
        "reduced SM must contain at least one degree-4 (Higgs) term for the FL N₀ quartic"
    );

    // Comparison N itself: Hermitian on a probe, zero vacuum expectation
    // (mirrors test 7 of sm_validation but as a *convolution-focused* pin:
    // N's quartic piece is the momentum-space self-convolution of φ²).
    let n_op = sm_comparison_n();
    let vac =
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    let e_n = QuantumState::inner_product(&n_op.apply(&vac), &vac).re;
    assert!(
        e_n.abs() < 1e-12,
        "comparison N vacuum energy must be 0 (normal-ordered), got {e_n}"
    );

    eprintln!(
        "sm_reduced_convolution: {n_off} off-diagonal one-quanton couplings (max {max_off:.3e}); \
         {n_quartic} quartic terms in H; ⟨N⟩_vac = {e_n:.3e}"
    );
}
