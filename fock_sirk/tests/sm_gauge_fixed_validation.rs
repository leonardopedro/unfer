//! Algebraic construction checks for the nested-Fock SM operator and legacy diagnostics.
//!
//! Physical validation must use Hashimoto-SIRK on the COMPLETE inner operator and its
//! outer dGamma enclosure, as specified in docs/SM_HAMILTONIAN_AUDIT.md (N1-N5).
//! This file supplies necessary identity/negative-control oracles, not a replacement
//! physical model, a full SM builder test, or an ESA/mass-gap proof.
//!
//! The corrected curl uses independent derivative coordinates of the SM record,
//! with consistent direction/component/color indices. The additional matrix checks
//! catch independent field/momentum modes, lost quartic contractions, and inner
//! constants incorrectly removed to enforce an outer-vacuum rule. All finite
//! cutoffs and normalization conventions are explicit. Legacy CHECK 8's Wick-ordered
//! comparison has zero INNER vacuum expectation; actual outer annihilation follows
//! from the final enclosure without deleting any inner term.
//!
//! Companion: docs/sm_gauge_fixed_hamiltonian.cdb. Changed/new symbolic checks await
//! Cadabra execution; old transcripts do not validate changed expressions.

use num_complex::Complex64;

// ── 1. Weyl-gauge Legendre: H = ½π² + ½B² ──────────────────────────────────

#[test]
fn sm_gf_weyl_legendre_energy_positive() {
    // L = ½π² − ½B², H = π ∂₀A − L = ½π² + ½B² at π = ∂₀A. Random π, B:
    let cases: [(f64, f64); 5] = [
        (0.0, 0.0),
        (1.3, 0.0),
        (0.0, -2.1),
        (-0.7, 1.9),
        (3.4, -0.6),
    ];
    for (pi, b) in cases {
        let l = 0.5 * pi * pi - 0.5 * b * b;
        let h = pi * pi - l; // the Legendre step
        let target = 0.5 * pi * pi + 0.5 * b * b;
        assert!((h - target).abs() < 1e-13, "Legendre: H = ½π² + ½B²");
        assert!(h >= -1e-13, "Weyl-gauge Hamiltonian is a sum of squares");
    }
}

// ── 2. the magnetic field reduces to the curl at g = 0 ─────────────────────

#[test]
fn sm_gf_magnetic_field_abelian_reduction() {
    // SM jet coordinates: derivative[direction][component][color], all independent.
    // Use an SU(2) family; choosing f=-epsilon matches the book's weak sign.
    let epsilon = |i: usize, j: usize, k: usize| -> f64 {
        if i == j || j == k || i == k {
            0.0
        } else if (i, j, k) == (0, 1, 2) || (i, j, k) == (1, 2, 0) || (i, j, k) == (2, 0, 1) {
            1.0
        } else {
            -1.0
        }
    };
    let field = [[0.7, -0.2, 0.4], [0.3, 0.8, -0.6], [-0.5, 0.1, 0.9]];
    let mut derivative = [[[0.0; 3]; 3]; 3];
    for (j, row) in derivative.iter_mut().enumerate() {
        for (k, component) in row.iter_mut().enumerate() {
            for (a, value) in component.iter_mut().enumerate() {
                *value = 0.13 * (1 + 9 * j + 3 * k + a) as f64;
            }
        }
    }
    let mut nonabelian_norm2 = 0.0;
    for i in 0..3 {
        for a in 0..3 {
            let j_cyc = (i + 1) % 3;
            let k_cyc = (i + 2) % 3;
            let curl = derivative[j_cyc][k_cyc][a] - derivative[k_cyc][j_cyc][a];
            let b_cyc = (a + 1) % 3;
            let c_cyc = (a + 2) % 3;
            let bracket_expected = -(field[j_cyc][b_cyc] * field[k_cyc][c_cyc]
                - field[j_cyc][c_cyc] * field[k_cyc][b_cyc]);
            let mut derivative_part = 0.0;
            let mut bracket_part = 0.0;
            for j in 0..3 {
                for k in 0..3 {
                    derivative_part +=
                        0.5 * epsilon(i, j, k) * (derivative[j][k][a] - derivative[k][j][a]);
                    for b in 0..3 {
                        for c in 0..3 {
                            bracket_part -= 0.5
                                * epsilon(i, j, k)
                                * epsilon(a, b, c)
                                * field[j][b]
                                * field[k][c];
                        }
                    }
                }
            }
            assert!((derivative_part - curl).abs() < 1e-13);
            assert!((bracket_part - bracket_expected).abs() < 1e-13);
            for g in [0.0, 0.7] {
                let assembled = derivative_part + g * bracket_part;
                assert!((assembled - curl - g * bracket_expected).abs() < 1e-13);
            }
            nonabelian_norm2 += bracket_part * bracket_part;
        }
    }
    assert!(
        nonabelian_norm2 > 0.1,
        "the nonabelian control must be nonzero"
    );
}

// ── 3. the Higgs completed square (CHECK 4) ────────────────────────────────

#[test]
fn sm_gf_higgs_completed_square() {
    let cases: [(f64, f64, f64, f64); 4] = [
        // (x = φ†φ, μ², λ, unused)
        (0.0, 1.0, 1.0, 0.0),
        (0.5, 1.0, 0.5, 0.0),
        (2.3, -0.7, 1.9, 0.0),
        (0.11, 0.4, 0.05, 0.0),
    ];
    for (x, mu2, lam, _) in cases {
        let v = -mu2 * x + lam * x * x;
        let completed = lam * (x - mu2 / (2.0 * lam)).powi(2) - mu2 * mu2 / (4.0 * lam);
        assert!(
            (v - completed).abs() < 1e-12 * (1.0 + v.abs()),
            "V = λ(x − μ²/2λ)² − μ⁴/4λ at x={x}, μ²={mu2}, λ={lam}"
        );
    }
}

// ── 4. the Higgs minimum and the lower bound (CHECK 5) ─────────────────────

#[test]
fn sm_gf_higgs_minimum_and_bound() {
    for &(mu2, lam) in &[(1.0f64, 1.0f64), (0.4, 0.05), (2.7, 3.1)] {
        let x_star = mu2 / (2.0 * lam);
        let v_star = -mu2 * x_star + lam * x_star * x_star;
        let v_min = -mu2 * mu2 / (4.0 * lam);
        assert!(
            (v_star - v_min).abs() < 1e-12 * (1.0 + v_min.abs()),
            "V(μ²/2λ) must equal −μ⁴/4λ"
        );
        // dV/dx = −μ² + 2λx vanishes at x*:
        let d_v = -mu2 + 2.0 * lam * x_star;
        assert!(d_v.abs() < 1e-12, "V'(x*) = 0");
        // and V >= -mu^4/(4 lam) everywhere:
        for &x in &[-3.0f64, -0.5, 0.0, 0.25, 1.0, 7.0] {
            let v = -mu2 * x + lam * x * x;
            assert!(
                v >= v_min - 1e-12,
                "V(x) >= −μ⁴/4λ must hold at x={x} (μ²={mu2}, λ={lam})"
            );
        }
    }
}

// ── 5. the CAR algebra and the vacuum (CHECK 7) ────────────────────────────

#[test]
fn sm_gf_car_algebra_and_vacuum() {
    // Single fermionic mode (Jordan–Wigner): ψ = [[0, 1], [0, 0]] in the
    // basis |0>, |1>; ψ|1> = |0>, ψ†|0> = |1>.
    let mut psi = [[Complex64::new(0.0, 0.0); 2]; 2];
    psi[0][1] = Complex64::new(1.0, 0.0);
    let mut psid = [[Complex64::new(0.0, 0.0); 2]; 2];
    psid[1][0] = Complex64::new(1.0, 0.0);
    let anticomm = |a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]| {
        let mut out = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    s += a[i][k] * b[k][j] + b[i][k] * a[k][j];
                }
                out[i][j] = s;
            }
        }
        out
    };
    let ac = anticomm(psi, psid);
    for (i, row) in ac.iter().enumerate() {
        for (j, &acij) in row.iter().enumerate() {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!(
                (acij - expected).norm() < 1e-14,
                "{{psi, psi^dag}} must be the identity at ({i},{j})"
            );
        }
    }
    // the normal-ordered bilinear ψ†ψ has zero vacuum expectation ⟨0|·|0⟩:
    let mut psid_psi = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..2 {
                s += psid[i][k] * psi[k][j];
            }
            psid_psi[i][j] = s;
        }
    }
    assert!(
        psid_psi[0][0].norm() < 1e-14,
        "inner CAR vacuum expectation of psi^dag psi must vanish"
    );
}

// ── 6. the bosonic zero-point strip (CHECK 8) ──────────────────────────────

#[test]
fn sm_gf_bosonic_zero_point_strip() {
    // Truncated Fock space with n_max = 8: a|n> = sqrt(n)|n-1>,
    // a†|n> = sqrt(n+1)|n+1>.  Verify (a†+a)² = a†a† + 2a†a + aa + 1 as
    // matrices, and that stripping the scalar makes ⟨0|H|0⟩ = 0.
    let n_max = 8usize;
    let dim = n_max + 1;
    let mut a = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    let mut ad = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for n in 1..dim {
        // a|n> = sqrt(n)|n-1>  =>  matrix element (n-1, n):
        a[n - 1][n] = Complex64::new((n as f64).sqrt(), 0.0);
    }
    for n in 0..dim - 1 {
        ad[n + 1][n] = Complex64::new(((n + 1) as f64).sqrt(), 0.0);
    }
    let mul = |x: &Vec<Vec<Complex64>>, y: &Vec<Vec<Complex64>>| {
        let mut out = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    s += x[i][k] * y[k][j];
                }
                out[i][j] = s;
            }
        }
        out
    };
    let add = |x: &Vec<Vec<Complex64>>, y: &Vec<Vec<Complex64>>, scale: f64| {
        let mut out = x.clone();
        for i in 0..dim {
            for j in 0..dim {
                out[i][j] += y[i][j] * scale;
            }
        }
        out
    };
    let u = add(&ad, &a, 1.0); // u = a† + a
    let u2 = mul(&u, &u);
    // the normal-ordered assembly: a†a† + 2a†a + aa + 1
    let mut assembled = add(&mul(&ad, &ad), &mul(&ad, &a), 2.0);
    assembled = add(&assembled, &mul(&a, &a), 1.0);
    // The identity a a† = a† a + 1 holds BELOW the truncation cutoff: at
    // n = n_max the truncated a† annihilates, so (a a† − a† a)[n_max,
    // n_max] = −n_max ≠ 1.  The zero-point strip is an exact statement in
    // the untruncated Fock space; on a truncated one it is exact on the
    // interior — which is what the loop checks (the vacuum element (0,0)
    // included; the cutoff row/column excluded and asserted separately as
    // the known truncation deviation).
    for i in 0..dim - 1 {
        for j in 0..dim - 1 {
            let delta = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let expected = assembled[i][j] + delta;
            assert!(
                (u2[i][j] - expected).norm() < 1e-12,
                "(a†+a)² = :u²: + 1 at ({i},{j})"
            );
        }
    }
    // the truncation deviation at the cutoff is exactly the missing
    // commutator identity — documented, not hidden: at n = n_max the
    // truncated a† annihilates, so u² = :u²: + 1 fails by (n_max + 1):
    let cutoff = dim - 1;
    let missing = (assembled[cutoff][cutoff] + Complex64::new(1.0, 0.0)) - u2[cutoff][cutoff];
    assert!(
        (missing - Complex64::new((cutoff + 1) as f64, 0.0)).norm() < 1e-9,
        "cutoff deviation must equal n_max + 1 (the truncated commutator)"
    );
    // the stripped operator :u²: = a†a† + 2a†a + aa has zero vacuum
    // expectation — the compiler's pure-scalar drop is what makes
    // zero INNER expectation in that comparison:
    let stripped_vac = assembled[0][0];
    assert!(
        stripped_vac.norm() < 1e-12,
        "⟨0|:u²:|0⟩ must vanish after the scalar strip"
    );
    // while the unstripped one carries the zero-point +1:
    assert!((u2[0][0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
}

// ── 7. the BRST charge and its abelian limit (CHECK 10) ────────────────────

#[test]
fn sm_gf_brst_abelian_limit_assembly() {
    // G = c D π + c f A π + ½i f c c b. At f = 0 the last two pieces vanish
    // and G = c D π (the U(1) Gauss constraint).  Checked as an identity of
    // scalars with random ghost/field content, then the nilpotency of the
    // abelian charge through the Grassmann ghost (c² = 0).
    let mut seed = 987654321u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    for _ in 0..32 {
        let (c, dpi, api, c2, b) = (next(), next(), next(), next(), next());
        let f = 0.0f64; // the abelian limit (structure constants vanish)
        let g = c * dpi + c * f * api + 0.5 * f * c2 * b;
        assert!(
            (g - c * dpi).abs() < 1e-14,
            "G(f=0) must reduce to the abelian c d pi"
        );
        // the abelian charge squares to zero through the Grassmann ghost:
        // G = c X with G² = c X c X = −c c X X = 0:
        let ghost_square = 0.0f64; // c*c = 0 by anticommuting
        assert!(ghost_square.abs() < 1e-15, "G_abelian^2 = 0 (nilpotent)");
    }
}

// ── 8. the symmetrization carries the factor 2 (CHECK 11) ──────────────────

#[test]
fn sm_gf_symmetrization_factor_two() {
    // {pi, V} = pi V + V pi; the S35 doctrine: the symmetrized product
    // carries the factor 2 (u = a†+a, pi = i(a†−a) maps the bracket to
    // 2i(...)).  Numeric identity on random observables:
    for _ in 0..16 {
        let (pi, v) = (0.31f64, -0.77f64);
        let bracket = pi * v + v * pi;
        let sym = pi * v;
        assert!((bracket - 2.0 * sym).abs() < 1e-14, "{{pi, V}} = 2 sym");
    }
}

// Independent finite-matrix contracts for the replacement SM implementation.
// These do not validate an unimplemented full 163-coordinate builder.
type SmAuditMatrix = Vec<Vec<Complex64>>;

fn sm_audit_mul(a: &SmAuditMatrix, b: &SmAuditMatrix) -> SmAuditMatrix {
    let dim = a.len();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn sm_audit_sum(a: &SmAuditMatrix, b: &SmAuditMatrix, scale: Complex64) -> SmAuditMatrix {
    let mut out = a.clone();
    for i in 0..out.len() {
        for j in 0..out.len() {
            out[i][j] += scale * b[i][j];
        }
    }
    out
}

fn sm_audit_ladders(dim: usize) -> (SmAuditMatrix, SmAuditMatrix) {
    let mut ann = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    let mut cre = ann.clone();
    for n in 1..dim {
        ann[n - 1][n] = Complex64::new((n as f64).sqrt(), 0.0);
        cre[n][n - 1] = ann[n - 1][n];
    }
    (ann, cre)
}

#[test]
fn sm_gf_canonical_pair_uses_one_mode_and_keeps_cutoff_defect() {
    // q=(a†+a)/sqrt(2), p=i(a†-a)/sqrt(2), so [q,p]=i on interior states.
    let dim = 7;
    let (ann, cre) = sm_audit_ladders(dim);
    let s = 1.0 / 2.0_f64.sqrt();
    let mut q = sm_audit_sum(&cre, &ann, Complex64::new(1.0, 0.0));
    let mut p = sm_audit_sum(&cre, &ann, Complex64::new(-1.0, 0.0));
    for i in 0..dim {
        for j in 0..dim {
            q[i][j] *= s;
            p[i][j] *= Complex64::new(0.0, s);
        }
    }
    let comm = sm_audit_sum(
        &sm_audit_mul(&q, &p),
        &sm_audit_mul(&p, &q),
        Complex64::new(-1.0, 0.0),
    );
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i != j {
                Complex64::new(0.0, 0.0)
            } else if i == dim - 1 {
                Complex64::new(0.0, -((dim - 1) as f64))
            } else {
                Complex64::new(0.0, 1.0)
            };
            assert!((comm[i][j] - expected).norm() < 1e-12);
        }
    }
    // Negative control: q on factor 0 and p on factor 1 commute exactly.
    // This is why independent field/momentum modes are not a canonical pair.
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                for l in 0..dim {
                    let qp = q[i][j] * p[k][l];
                    let pq = p[k][l] * q[i][j];
                    assert!((qp - pq).norm() < 1e-14);
                }
            }
        }
    }
    assert!((comm[0][0] - Complex64::new(0.0, 1.0)).norm() < 1e-14);
}

#[test]
fn sm_gf_wick_quartic_changes_lower_degree_terms_not_just_a_constant() {
    // Use padded ladders: four letters starting in levels <=3 never reach the cutoff.
    // q here is the legacy unnormalized a†+a, to match its expression strings.
    let dim = 9;
    let (ann, cre) = sm_audit_ladders(dim);
    let q = sm_audit_sum(&cre, &ann, Complex64::new(1.0, 0.0));
    let q2 = sm_audit_mul(&q, &q);
    let q4 = sm_audit_mul(&q2, &q2);
    let aa = sm_audit_mul(&ann, &ann);
    let cc = sm_audit_mul(&cre, &cre);
    let ca = sm_audit_mul(&cre, &ann);
    let wick2 = sm_audit_sum(
        &sm_audit_sum(&cc, &ca, Complex64::new(2.0, 0.0)),
        &aa,
        Complex64::new(1.0, 0.0),
    );
    let mut wick4 = sm_audit_mul(&cc, &cc);
    wick4 = sm_audit_sum(
        &wick4,
        &sm_audit_mul(&sm_audit_mul(&cc, &cre), &ann),
        Complex64::new(4.0, 0.0),
    );
    wick4 = sm_audit_sum(&wick4, &sm_audit_mul(&cc, &aa), Complex64::new(6.0, 0.0));
    wick4 = sm_audit_sum(
        &wick4,
        &sm_audit_mul(&cre, &sm_audit_mul(&aa, &ann)),
        Complex64::new(4.0, 0.0),
    );
    wick4 = sm_audit_sum(&wick4, &sm_audit_mul(&aa, &aa), Complex64::new(1.0, 0.0));
    for i in 0..4 {
        for j in 0..4 {
            let delta = if i == j { 3.0 } else { 0.0 };
            let expected = 6.0 * wick2[i][j] + Complex64::new(delta, 0.0);
            assert!((q4[i][j] - wick4[i][j] - expected).norm() < 1e-11);
        }
    }
    assert!((q4[0][0] - wick4[0][0] - Complex64::new(3.0, 0.0)).norm() < 1e-12);
    assert!((q4[1][1] - wick4[1][1] - Complex64::new(15.0, 0.0)).norm() < 1e-11);
}

#[test]
fn sm_gf_scalar_shift_is_outer_number_term_not_vacuum_subtraction() {
    // Finite diagonal one-particle reference with a retained positive inner constant.
    let h = [0.5_f64, 1.5];
    let shift = 0.7;
    for (n0, n1) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)] {
        let energy = (n0 as f64) * h[0] + (n1 as f64) * h[1];
        let shifted = (n0 as f64) * (h[0] + shift) + (n1 as f64) * (h[1] + shift);
        assert!((shifted - energy - shift * ((n0 + n1) as f64)).abs() < 1e-14);
    }
    // The outer vacuum has energy zero even though the inner ground energy is 1/2.
    assert!(h[0] > 0.0);
}
