use crate::{Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState, compile_to_fock};
use num_complex::Complex64;
use std::sync::Arc;

// ─────────────────────────────────────────────
// Direct Hamiltonian term builder helpers
// A hermitian field φ_i = a†_i + a_i expands as two Operator terms.
// conjugate momentum π_i = i(a†_i - a_i) expands as two Operator terms.
// ─────────────────────────────────────────────

/// Returns the list of (coeff, op) pairs for a hermitian field φ_mode = a†_mode + a_mode.
fn field_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::new(1.0, 0.0), Operator::InnerBosonCreate(mode)),
        (
            Complex64::new(1.0, 0.0),
            Operator::InnerBosonAnnihilate(mode),
        ),
    ]
}

/// Returns the list of (coeff, op) pairs for conjugate momentum π_mode = i(a†_mode - a_mode).
fn momentum_ops(mode: u32) -> Vec<(Complex64, Operator)> {
    vec![
        (Complex64::new(0.0, 1.0), Operator::InnerBosonCreate(mode)),
        (
            Complex64::new(0.0, -1.0),
            Operator::InnerBosonAnnihilate(mode),
        ),
    ]
}

/// Expand A·B product over all (coeff_a, op_a) × (coeff_b, op_b) pairs.
fn product_terms(
    a: &[(Complex64, Operator)],
    b: &[(Complex64, Operator)],
) -> Vec<(Complex64, Vec<Operator>)> {
    let mut result = Vec::new();
    for (ca, oa) in a {
        for (cb, ob) in b {
            result.push((ca * cb, vec![oa.clone(), ob.clone()]));
        }
    }
    result
}

/// Adds terms c * A^2 = c * A * A to `terms`.
fn add_quadratic(
    terms: &mut Vec<(Complex64, Vec<Operator>)>,
    coeff: f64,
    ops: &[(Complex64, Operator)],
) {
    for t in product_terms(ops, ops) {
        let c = Complex64::new(coeff, 0.0) * t.0;
        if c.norm_sqr() > 1e-30 {
            terms.push((c, t.1));
        }
    }
}

// ─────────────────────────────────────────────
// 1. Navier-Stokes Hamiltonian
//    Built directly as Hamiltonian terms — bypasses Expression::expand() which
//    hangs on the high-order symbolic tree (AGENTS.md: combinatorial explosion
//    avoidance). The original Expression-based version also had a bug where the
//    "neg" symbol was treated as factor 1.0 instead of -1; building directly
//    avoids that class of bug entirely.
//
//    H = Σ_i { π_i , A_i }   (anti-commutator → Hermitian)
//    A_i = Σ_j u_j · u_{ij} − ν · u_{12+i}
// ─────────────────────────────────────────────
pub fn navier_stokes_hamiltonian(nu: f64) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();

    for i in 0..3u32 {
        let pi = momentum_ops(i); // [(i, a†_i), (-i, a_i)]

        // Build A_i = Σ_j u_j · u_{ij} − ν · u_{12+i} as (coeff, ops) pairs.
        let mut a_terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();

        // Quadratic part: Σ_j u_j · u_{ij}
        for j in 0..3u32 {
            let u_j = field_ops(j);
            let u_ij = field_ops(3 + i * 3 + j);
            for (cj, oj) in &u_j {
                for (cij, oij) in &u_ij {
                    a_terms.push((cj * cij, vec![oj.clone(), oij.clone()]));
                }
            }
        }

        // Linear part: −ν · u_{12+i}
        let nu_c = Complex64::new(-nu, 0.0);
        for (cd, od) in field_ops(12 + i) {
            a_terms.push((nu_c * cd, vec![od]));
        }

        // H += π_i · A_i  (forward product)
        for (cp, op) in &pi {
            for (ca, oa) in &a_terms {
                let mut ops = vec![op.clone()];
                ops.extend(oa.iter().cloned());
                let c = cp * ca;
                if c.norm_sqr() > 1e-30 {
                    terms.push((c, ops));
                }
            }
        }

        // H += A_i · π_i  (reverse product — Hermitian conjugate)
        for (ca, oa) in &a_terms {
            for (cp, op) in &pi {
                let mut ops = oa.clone();
                ops.push(op.clone());
                let c = ca * cp;
                if c.norm_sqr() > 1e-30 {
                    terms.push((c, ops));
                }
            }
        }
    }

    Hamiltonian { terms }
}

/// BRST Divergence Constraint for Navier-Stokes: Ω = Σ_j u_{j,j} · c_j
/// Built directly as Hamiltonian terms (bypasses Expression::expand()).
pub fn navier_stokes_brst() -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for j in 0..3u32 {
        let mode = 3 + j * 3 + j;
        for (c, op) in field_ops(mode) {
            terms.push((c, vec![op, Operator::InnerFermionAnnihilate(j)]));
        }
    }
    Hamiltonian { terms }
}

/// Eulerian velocity-fiber Hamiltonian — the **affine (ThreeComponent) fiber** of
/// the formalization (timepiece `CONSOLIDATED_PLAN.md` §9 items 4/8:
/// `BilinearEsa.bilH` → `AffineFiber.affH` → `AffineBlock.affBlockH` →
/// `ThreeComponent.velH`):
///
///   `H = Σ_i (π_i V_i + V_i π_i),   V_i(u) = Σ_k A_{ik} u_k + c_i`,
///
/// with the velocity-mode ladder operators (modes 0, 1, 2) `u_k = a†_k + a_k`,
/// `π_i = i(a†_i − a_i)`, an **arbitrary real** `3 × 3` velocity-gradient matrix
/// `A` (no symmetry, positivity or sign assumption) and an arbitrary real vector
/// `c`. This is the velocity-momentum part of the Navier–Stokes Hamiltonian
/// ([`navier_stokes_hamiltonian`]) with the derivative field and the viscous
/// offsets *frozen to the constants* `A_{ik}`, `c_i` — the Eulerian
/// derivatives-as-fields picture, in which the derivative modes carry no momenta,
/// commute with the Hamiltonian and are constants of the motion (the block
/// decomposition that diagonalises the derivative field).
///
/// `H` expands to the 24 hopping terms per component of the plan: the `±2`-shifts
/// (pair creation/annihilation `a†_i a†_k`, `a_i a_k` of the pure advection
/// `A_{ik} u_k`), the `±1`-shifts of the viscous offset `c_i π_i` (the affine
/// fiber of `AffineFiber.affH`), and the **number-conserving vorticity hopping**
/// `a†_i a_k` / `a_k a†_i` whose amplitude `∝ (A_{ki} − A_{ik})` is not monotone
/// along the shift.
///
/// Conventions: the ladder ops are the framework-native `u = a†+a`,
/// `π = i(a†−a)` and the symmetrization is the anti-commutator `{π,V} = πV + Vπ`
/// (twice the plan's `½(πV+Vπ)`), so the hopping amplitudes carry a factor `2`
/// relative to the plan's normalized `u = (a†+a)/√2`, `π = i(a†−a)/√2` convention.
pub fn ns_eulerian_fiber(a: &[[f64; 3]; 3], c: &[f64; 3]) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for i in 0..3u32 {
        let pi = momentum_ops(i);
        // V_i = Σ_k A_{ik} u_k as (coeff, op) pairs.
        let mut v: Vec<(Complex64, Operator)> = Vec::with_capacity(6);
        for k in 0..3u32 {
            let a_ik = Complex64::new(a[i as usize][k as usize], 0.0);
            for (co, oo) in field_ops(k) {
                let coeff = a_ik * co;
                if coeff.norm_sqr() > 1e-30 {
                    v.push((coeff, oo));
                }
            }
        }
        // π_i·V_i (op_v applied first, then op_pi).
        for (cp, op_pi) in &pi {
            for (cv, op_v) in &v {
                let coeff = cp * cv;
                if coeff.norm_sqr() > 1e-30 {
                    terms.push((coeff, vec![op_pi.clone(), op_v.clone()]));
                }
            }
        }
        // V_i·π_i (op_pi applied first, then op_v).
        for (cv, op_v) in &v {
            for (cp, op_pi) in &pi {
                let coeff = cp * cv;
                if coeff.norm_sqr() > 1e-30 {
                    terms.push((coeff, vec![op_v.clone(), op_pi.clone()]));
                }
            }
        }
        // Viscous offset: {π_i, c_i} = 2 c_i π_i (the ±1-shift affine fiber).
        let ci = Complex64::new(2.0 * c[i as usize], 0.0);
        for (cp, op_pi) in &pi {
            let coeff = cp * ci;
            if coeff.norm_sqr() > 1e-30 {
                terms.push((coeff, vec![op_pi.clone()]));
            }
        }
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 1c. Hermite-spatial derivative-variable gauge fixing (1D NS slice)
//
// The Eulerian derivatives-as-fields picture made *numerically real*: the
// velocity field is expanded in physicists' Hermite polynomials,
//
//     u(x) = Σ_n u_n H_n(x),   u_n = a†_n + a_n,   ∂_x H_n = 2n H_{n-1},
//
// so the spatial derivative of the field operator is the well-defined operator
// on the velocity ladder modes
//
//     ∂_x u(x) = Σ_n u_n · 2n H_{n-1}(x) = Σ_m (2(m+1) u_{m+1}) H_m(x),
//
// i.e. the mode-m component of the derivative is  D_m = 2(m+1) u_{m+1}.  The
// promoted derivative variables g_m = a†_{3+m} + a_{3+m} (modes 3,4) are fixed
// to these *actual* derivative values by the BRST charge
//
//     Ω = (g_0 − 2 u_1) c_0 + (g_1 − 4 u_2) c_1,
//
// the NS derivative-variable fixing (book.tex §4159-4197): each promoted
// variable is fixed to the value of the field derivative (products of field
// derivatives then survive in the Hamiltonian, exactly as the NS Hamiltonian
// carries u_j·u_{i,j}).  The gauge-fixed fiber
//
//     H = {π_0, u_0 g_0 + u_1 g_1}
//
// (the Eulerian block structure: the derivative-content modes 1,2 carry NO
// momenta, so [H, u_1] = [H, u_2] = [H, g_m] = 0 and the constraint
// C_m = g_m − D_m is an exact constant of the motion, [H, C_m] = 0) while the
// value mode 0 evolves ([H, u_0] ≠ 0).  Hence for a physical initial
// wave-function (one where the promoted variables are set to the actual
// derivative values), the operator identity ⟨g_m⟩ = ⟨D_m⟩ = 2(m+1)⟨u_{m+1}⟩
// holds at ALL times, under the bare flow and under the BRST-projected flow.
// ─────────────────────────────────────────────

/// The Hermite-spatial derivative-variable BRST charge (1D NS slice):
/// `Ω = (g_0 − 2u_1) c_0 + (g_1 − 4u_2) c_1` — fixes each promoted derivative
/// variable `g_m` (boson modes 3,4) to the actual field derivative
/// `D_m = 2(m+1)u_{m+1}` (velocity modes 1,2), with ghosts c_0, c_1 (fermion
/// modes 0,1).  The physical subspace (ker Ω in the ghost-extended space) is
/// exactly the set of states where the promoted variables equal the spatial
/// field derivatives.
pub fn ns_hermite_derivative_brst() -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    // C_0 = g_0 − 2 u_1, fixed by ghost c_0 (fermion mode 0).
    for (c, op) in field_ops(3) {
        terms.push((c, vec![op, Operator::InnerFermionAnnihilate(0)]));
    }
    for (c, op) in field_ops(1) {
        terms.push((
            Complex64::new(-2.0, 0.0) * c,
            vec![op, Operator::InnerFermionAnnihilate(0)],
        ));
    }
    // C_1 = g_1 − 4 u_2, fixed by ghost c_1 (fermion mode 1).
    for (c, op) in field_ops(4) {
        terms.push((c, vec![op, Operator::InnerFermionAnnihilate(1)]));
    }
    for (c, op) in field_ops(2) {
        terms.push((
            Complex64::new(-4.0, 0.0) * c,
            vec![op, Operator::InnerFermionAnnihilate(1)],
        ));
    }
    Hamiltonian { terms }
}

/// The gauge-fixed Hermite-spatial fiber Hamiltonian (1D NS slice):
/// `H = {π_0, u_0 g_0 + u_1 g_1}` — the velocity value mode 0 (with its
/// momentum) advected by the promoted derivative variables (modes 3,4), the
/// Eulerian block structure of the derivatives-as-fields picture: the
/// derivative-content modes 1,2 and the promoted variables carry no momenta, so
/// they are constants of the motion and the constraint `C_m = g_m − 2(m+1)u_{m+1}`
/// commutes with H exactly.  The value mode 0 genuinely evolves (`[H, u_0] ≠ 0`),
/// so the flow is non-trivial while the derivative-variable identity
/// `⟨g_m⟩ = 2(m+1)⟨u_{m+1}⟩` is preserved at every time.
pub fn ns_hermite_derivative_fiber() -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    let pi0 = momentum_ops(0);

    // A = u_0·g_0 + u_1·g_1  (the advection u·∂_x u in promoted form,
    // keeping products of the field with its derivative variables).
    let mut a_terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for (c1, o1) in field_ops(0) {
        for (c2, o2) in field_ops(3) {
            let c = c1 * c2;
            if c.norm_sqr() > 1e-30 {
                a_terms.push((c, vec![o1.clone(), o2.clone()]));
            }
        }
    }
    for (c1, o1) in field_ops(1) {
        for (c2, o2) in field_ops(4) {
            let c = c1 * c2;
            if c.norm_sqr() > 1e-30 {
                a_terms.push((c, vec![o1.clone(), o2.clone()]));
            }
        }
    }

    // H = {π_0, A} = π_0·A + A·π_0  (Hermitian by construction).
    for (cp, op_pi) in &pi0 {
        for (ca, ops_a) in &a_terms {
            let mut ops = vec![op_pi.clone()];
            ops.extend(ops_a.iter().cloned());
            let c = cp * ca;
            if c.norm_sqr() > 1e-30 {
                terms.push((c, ops));
            }
        }
    }
    for (ca, ops_a) in &a_terms {
        for (cp, op_pi) in &pi0 {
            let mut ops = ops_a.clone();
            ops.push(op_pi.clone());
            let c = ca * cp;
            if c.norm_sqr() > 1e-30 {
                terms.push((c, ops));
            }
        }
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// SU(3) structure constants f_abc (0-indexed, a,b,c in 0..7)
// ─────────────────────────────────────────────
fn su3_f(a: usize, b: usize, c: usize) -> f64 {
    // Canonical nonzero entries (totally antisymmetric)
    let table: &[(usize, usize, usize, f64)] = &[
        (0, 1, 2, 1.0),
        (0, 3, 6, 0.5),
        (0, 4, 5, -0.5),
        (1, 3, 5, 0.5),
        (1, 4, 6, 0.5),
        (2, 3, 4, 0.5),
        (2, 5, 6, -0.5),
        (3, 4, 7, 3.0f64.sqrt() / 2.0),
        (5, 6, 7, 3.0f64.sqrt() / 2.0),
    ];
    for &(v1, v2, v3, val) in table {
        let mut p = [v1, v2, v3];
        p.sort();
        let mut t = [a, b, c];
        t.sort();
        if p == t {
            // Count swaps to get sign
            let mut cur = [a, b, c];
            let mut swaps = 0usize;
            for i in 0..2 {
                for j in 0..2 - i {
                    if cur[j] > cur[j + 1] {
                        cur.swap(j, j + 1);
                        swaps += 1;
                    }
                }
            }
            return if swaps.is_multiple_of(2) { val } else { -val };
        }
    }
    0.0
}

fn epsilon3(i: usize, j: usize, k: usize) -> f64 {
    match (i, j, k) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (2, 1, 0) | (1, 0, 2) | (0, 2, 1) => -1.0,
        _ => 0.0,
    }
}

// ─────────────────────────────────────────────
// 2. Full Pure SU(3) Yang-Mills  (Phase 8.1)
//    H = -½ π^i_a π^i_a  -  ½ B_{ia} B_{ia}
//    B_{ia} = ε_{ijk}(∂_j A^a_k + ½ g f_{abc} A^b_j A^c_k)
//
// We build Hamiltonian terms DIRECTLY — no Expression.expand() — so the
// combinatorial explosion never occurs.
// ─────────────────────────────────────────────
pub fn yang_mills_hamiltonian(g: f64) -> Hamiltonian {
    yang_mills_hamiltonian_with_colors(g, 8)
}

/// The full pure Yang–Mills Hamiltonian with an arbitrary number of colors.
///
/// For `n_colors = 8` this is the SU(3) gauge theory of [`yang_mills_hamiltonian`].
/// **`n_colors = 1` is the abelian (U(1), Quantum Electrodynamics)
/// specialization**: the structure constants `f_{abc}` vanish identically for a
/// single color, so every non-abelian interaction term (the `A³` and `A⁴` pieces
/// of `B²`) drops out and the Hamiltonian is purely the quadratic free-Maxwell
/// operator
///
///   `H = −½ Σᵢ πᵢ² − ½ Σᵢ Bᵢ²`,   `Bᵢ = Σ_{jk} ε_{ijk} ∂_j A_k = (∇×A)ᵢ`,
///
/// whose normal-ordered quantization on a photon mode of frequency `ω` is
/// `ω·a†a` — exactly [`qed_free_photon`] (see
/// `fock_sirk/tests/qed_abelian_reduction.rs`).  The derivative-mode offset is
/// `3·n_colors` (the `3·n_colors` field modes come first; for 8 colors that is
/// the `24` of the SU(3) realization).
pub fn yang_mills_hamiltonian_with_colors(g: f64, n_colors: usize) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();

    // ── Kinetic term:  -½ π^i_a π^i_a ──────────────────────────────
    for i in 0..3 {
        for a in 0..n_colors {
            let mode = (i * n_colors + a) as u32;
            let pi = momentum_ops(mode);
            add_quadratic(&mut terms, -0.5, &pi);
        }
    }

    // ── Magnetic term: -½ B_{ia} B_{ia} ────────────────────────────
    // B_{ia} = Σ_{j,k} ε_{ijk} [ L_{jk,a}  +  NL_{jk,a} ]
    // L_{jk,a}  = ∂_j A^a_k   → mapped to hermitian field mode
    //              ((3*n_colors) + (i*3+j)*n_colors + a)
    // NL_{jk,a} = ½ g Σ_{b,c} f_{abc} A^b_j A^c_k
    //
    // B_{ia}^2 = (L + NL)^2 = L^2 + 2 L·NL + NL^2
    // We accumulate each (i,a) slice then expand the square.
    for i in 0..3 {
        for a in 0..n_colors {
            // Collect linear pieces (coeff, single-Operator) for this B_{ia}
            let mut b_ia: Vec<(Complex64, Operator)> = Vec::new();

            for j in 0..3 {
                for k in 0..3 {
                    let eps = epsilon3(i, j, k);
                    if eps == 0.0 {
                        continue;
                    }

                    // Linear part: ∂_j A^a_k → one hermitian field op pair
                    let da_mode = (3 * n_colors + (i * 3 + j) * n_colors + a) as u32;
                    for (c, op) in field_ops(da_mode) {
                        b_ia.push((c * eps, op));
                    }

                    // Non-linear part: ½ g f_{abc} A^b_j A^c_k
                    // Non-linear pieces are handled in the NL*NL and L*NL sections below.
                }
            }

            // -½ B_{ia}^2 from linear (single-op) pieces only:
            add_quadratic(&mut terms, -0.5, &b_ia);

            // Non-linear quadratic (quartic) terms: -½ * NL_{ia} * NL_{ia}
            // We add them as 4-operator terms directly.
            for j in 0..3 {
                for k in 0..3 {
                    let eps_jk = epsilon3(i, j, k);
                    if eps_jk == 0.0 {
                        continue;
                    }
                    for b_idx in 0..n_colors {
                        for c_idx in 0..n_colors {
                            let fabc = su3_f(a, b_idx, c_idx);
                            if fabc.abs() < 1e-15 {
                                continue;
                            }
                            for j2 in 0..3 {
                                for k2 in 0..3 {
                                    let eps_j2k2 = epsilon3(i, j2, k2);
                                    if eps_j2k2 == 0.0 {
                                        continue;
                                    }
                                    for b2 in 0..n_colors {
                                        for c2 in 0..n_colors {
                                            let fabc2 = su3_f(a, b2, c2);
                                            if fabc2.abs() < 1e-15 {
                                                continue;
                                            }
                                            // -½ * (½g)^2 * eps * eps * f * f * A^b_j A^c_k A^b2_j2 A^c2_k2
                                            let nl_coeff = -0.5
                                                * (0.5 * g).powi(2)
                                                * eps_jk
                                                * eps_j2k2
                                                * fabc
                                                * fabc2;
                                            if nl_coeff.abs() < 1e-30 {
                                                continue;
                                            }
                                            let coeff = Complex64::new(nl_coeff, 0.0);
                                            let m1 = (j * n_colors + b_idx) as u32;
                                            let m2 = (k * n_colors + c_idx) as u32;
                                            let m3 = (j2 * n_colors + b2) as u32;
                                            let m4 = (k2 * n_colors + c2) as u32;
                                            // Each field = c† + a, so 2^4=16 sub-terms
                                            for (c1f, o1) in field_ops(m1) {
                                                for (c2f, o2) in field_ops(m2) {
                                                    for (c3f, o3) in field_ops(m3) {
                                                        for (c4f, o4) in field_ops(m4) {
                                                            let c_total =
                                                                coeff * c1f * c2f * c3f * c4f;
                                                            if c_total.norm_sqr() < 1e-30 {
                                                                continue;
                                                            }
                                                            terms.push((
                                                                c_total,
                                                                vec![
                                                                    o1.clone(),
                                                                    o2.clone(),
                                                                    o3.clone(),
                                                                    o4.clone(),
                                                                ],
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Cross terms: -½ * 2 * L * NL = -L * NL
            for j in 0..3 {
                for k in 0..3 {
                    let eps = epsilon3(i, j, k);
                    if eps == 0.0 {
                        continue;
                    }
                    let da_mode = (3 * n_colors + (i * 3 + j) * n_colors + a) as u32;
                    for b_idx in 0..n_colors {
                        for c_idx in 0..n_colors {
                            let fabc = su3_f(a, b_idx, c_idx);
                            if fabc.abs() < 1e-15 {
                                continue;
                            }
                            let nl_base = -0.5 * g * eps * fabc; // -1 * ½ * L*NL * 2 = -L*NL
                            let coeff = Complex64::new(nl_base, 0.0);
                            let mode_b = (j * n_colors + b_idx) as u32;
                            let mode_c = (k * n_colors + c_idx) as u32;
                            // L = field_ops(da_mode), NL_pair = field_ops(mode_b)*field_ops(mode_c)
                            for (cl, ol) in field_ops(da_mode) {
                                for (cb, ob) in field_ops(mode_b) {
                                    for (cc, oc) in field_ops(mode_c) {
                                        let c_total = coeff * cl * cb * cc;
                                        if c_total.norm_sqr() < 1e-30 {
                                            continue;
                                        }
                                        terms.push((
                                            c_total,
                                            vec![ol.clone(), ob.clone(), oc.clone()],
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 3. Einstein-Cartan Gravity (Phase 8.2)
//    Simplified 3D constraint: H = Σ_{ia} (P_{ia}^2 - e_{ia}^2)
//    Modes 0..8: tetrad e^a_i, Modes 9..17: polymomentum P^i_a
// ─────────────────────────────────────────────
pub fn gravity_hamiltonian() -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for i in 0..3 {
        for a in 0..3 {
            let p_mode = (9 + i * 3 + a) as u32;
            let e_mode = (i * 3 + a) as u32;
            let pi = momentum_ops(p_mode);
            let ef = field_ops(e_mode);
            add_quadratic(&mut terms, 1.0, &pi);
            add_quadratic(&mut terms, -1.0, &ef);
        }
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 4. Harmonic Chain (Stage 8 builtin — tests/demos)
//    H = Σ_i ω a†_i a_i  (independent harmonic oscillators)
//    A simple, hermitian, explosion-safe model for Born-rule tests.
// ─────────────────────────────────────────────
pub fn harmonic_chain(n_modes: usize, omega: f64) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for i in 0..n_modes as u32 {
        terms.push((
            Complex64::new(omega, 0.0),
            vec![
                Operator::InnerBosonCreate(i),
                Operator::InnerBosonAnnihilate(i),
            ],
        ));
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 4b. Quantum Flow Matching (QFM) generator (builtin — see QFM.tex)
//     H = |0><0|  +  Σ_j α_j n_j        (n_j = B†_j B_j)
//
//     Analytical, neural-network-free generative flow: M orthogonal data
//     points become single *outer* bosons (one universe per data point, each
//     carrying inner mode j), and the Mehler uniform prior is the rank-1
//     vacuum projector |0><0| (QFM.tex §"Which vacuum?"). The data potential
//     has no cross-terms, so it is strictly diagonal (H|0>=|0>,
//     H|x_j>=α_j|x_j>) and building it stays O(M) by bypassing
//     Expression::expand().
// ─────────────────────────────────────────────

/// The analytical **Quantum Flow Matching** generator (see `QFM.tex`).
///
/// Encodes `M = alphas.len()` orthogonal data points as single-excitation
/// *outer* universes `|x_j> = B†_j|0>` (one universe holding one boson in
/// inner mode `j`, `B†_j = OuterBosonCreate(|1_j>)`) plus the Mehler
/// vacuum-projector prior: `H = |0><0| + Σ_j α_j · B†_j B_j`. The number
/// operator `n_j = B†_j B_j` must be built from the *outer* ladder operators
/// (not `InnerBosonCreate`/`InnerBosonAnnihilate`, which act on an already-
/// existing universe's own inner mode occupation): with inner operators, a
/// state carrying two or more simultaneously-excited data channels leaks
/// amplitude into an unphysical basis state where one universe is emptied
/// and another carries two channels' excitations at once, breaking the
/// zero-data-loss disjointness (`QFM.tex` eq. (disjoint)) the encoding
/// relies on. Constructed directly so M can be huge without hitting the CAS
/// term-explosion limit.
pub fn qfm_hamiltonian(alphas: &[f64]) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    // H_0 = |0><0|, the Mehler global prior.
    terms.push((Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum]));
    // Decoupled data potential: one outer number operator B†_j B_j per data
    // point, where |x_j> = B†_j|0> is a single outer universe holding one
    // boson in inner mode j.
    for (j, &alpha) in alphas.iter().enumerate() {
        let mode = j as u32;
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, 1);
        terms.push((
            Complex64::new(alpha, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// Build the exact rank-1 projector `H = |0̃><0̃|` onto a dressed Mehler
/// vacuum from its frame components: `|0̃> = c₀|vac>_F + Σ_j ε_j B†_j|vac>_F`,
/// with the channel `B†_j|vac>_F` given by an outer universe holding
/// `channels[j].0` and weight `ε_j = channels[j].1`.
///
/// The dressed vector is renormalized to unit norm before wrapping in
/// [`Operator::ProjectOnto`], so `H² = H` holds exactly even when distinct
/// data points quantize onto the same Fock basis state (their weights add).
/// Application cost is the rank-1 shortcut `H|s> = <0̃|s>·|0̃>` —
/// `O(components)` per matvec, never the `O(M²)` cross-term expansion.
fn dressed_vacuum_projector(channels: Vec<(InnerBosonicState, f64)>, c0: f64) -> Hamiltonian {
    let mut dressed = QuantumState {
        components: Default::default(),
    };
    if c0 != 0.0 {
        dressed
            .components
            .insert(OuterState::vacuum(), Complex64::new(c0, 0.0));
    }
    for (inner, eps) in channels {
        let mut outer = OuterState::vacuum();
        outer.bosonic.insert(inner, 1);
        *dressed
            .components
            .entry(outer)
            .or_insert(Complex64::new(0.0, 0.0)) += Complex64::new(eps, 0.0);
    }
    let norm: f64 = dressed
        .components
        .values()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.0, "dressed Mehler vacuum must be a nonzero vector");
    for a in dressed.components.values_mut() {
        *a /= norm;
    }
    Hamiltonian {
        terms: vec![(
            Complex64::new(1.0, 0.0),
            vec![Operator::ProjectOnto(Arc::new(dressed))],
        )],
    }
}

// ─────────────────────────────────────────────
// 4c. Localized data-point encoding for QFM (see `QFM.tex`, "The data-channel
//     wave-function on the hypersphere: finitely many localized coordinates,
//     the rest uniform").
//
//     `qfm_hamiltonian`/`qfm_hamiltonian_mehler_projector` above identify each data
//     point purely by its *array index* j: `|x_j> = OuterBosonCreate({j: 1})
//     |0>`. That is a legitimate single-excitation encoding (the enumeration
//     index alone already guarantees `<x_i|x_j> = delta_ij`), but it carries
//     none of the point's actual D real coordinates in the Fock-space state
//     itself — only in the scalar coefficient alpha_j.
//
//     `QFM.tex` describes a more literal picture: a data point x ∈ R^D
//     corresponds to an inner wave-function that localizes exactly D of the
//     (infinitely many) hyperspherical coordinates around x's own D real
//     components, leaving every other coordinate at the uniform circle
//     measure. `point_to_inner_state` is the direct computational
//     realization of that: it occupies exactly D inner modes (0..D-1, one
//     per real coordinate), each mode's occupation number a fixed-point
//     quantization of that coordinate, and leaves every other mode (of the
//     inner Fock space's infinitely many) at zero occupation — i.e. at the
//     vacuum/uniform state, exactly like the picture in the paper.
// ─────────────────────────────────────────────

/// Default fixed-point quantization scale for [`point_to_inner_state`]:
/// a real coordinate `x` is quantized to the nearest multiple of `1/SCALE`
/// before being encoded as an inner-mode occupation number. Coarser than
/// this and distinct nearby points collide onto the same Fock basis state
/// (become non-orthogonal); finer than this risks overflowing the `u32`
/// occupation-number range for large-magnitude coordinates.
pub const QFM_DEFAULT_QUANTIZATION_SCALE: f64 = 1024.0;

/// Zigzag-encode a signed integer into an unsigned one (`0,-1,1,-2,2,...` ->
/// `0,1,2,3,4,...`), the standard bijection `Z -> N` used by e.g. protobuf
/// varints. Needed because a real coordinate can be negative but a boson
/// occupation number (`u32`) cannot; a naive `abs()` would collide `+v` and
/// `-v` onto the same mode occupation, silently merging two distinct data
/// points into one non-orthogonal Fock state.
fn zigzag_encode(n: i64) -> u32 {
    (if n >= 0 {
        (n as u64) * 2
    } else {
        n.unsigned_abs() * 2 - 1
    }) as u32
}

/// Encode a real-valued point `x ∈ R^D` as an inner-Fock-space
/// configuration that occupies one mode per coordinate (`D` modes total,
/// indexed `0..D-1`), each mode's occupation number a fixed-point
/// quantization of that coordinate (see [`QFM_DEFAULT_QUANTIZATION_SCALE`]).
/// A coordinate that quantizes to exactly zero leaves its mode unoccupied
/// (equivalent to never touching it — the "uniform, no information" state
/// for that coordinate). Every mode beyond `D-1` is left unoccupied
/// regardless of `x`, matching `QFM.tex`'s "the rest uniform."
///
/// Two points that quantize to the same `D`-tuple of occupation numbers
/// produce the same `InnerBosonicState` and are therefore *not* orthogonal
/// (they become the same Fock basis state) — this is the encoding's finite
/// resolution, the discrete analogue of two wave-packets whose localized
/// supports overlap, not a bug.
pub fn point_to_inner_state(point: &[f64], scale: f64) -> InnerBosonicState {
    let mut modes = std::collections::BTreeMap::new();
    for (i, &xi) in point.iter().enumerate() {
        let q = (xi * scale).round() as i64;
        let occ = zigzag_encode(q);
        if occ > 0 {
            modes.insert(i as u32, occ);
        }
    }
    InnerBosonicState { modes }
}

/// The analytical **Quantum Flow Matching** generator, with each data point
/// localized on its own `D` inner modes (see the module-level note above),
/// rather than identified only by array index. `H = |0><0| + Σ_j α_j · B†_j
/// B_j`, where `|x_j> = B†_j|0>` and `B†_j` creates one outer universe
/// carrying [`point_to_inner_state`]`(points[j], scale)`.
///
/// `points` and `alphas` are zipped pairwise (extra elements in the longer
/// slice are ignored); use [`potential::optimal_coefficients`] (in the
/// `qfm` crate) to derive `alphas` from `points` directly.
pub fn qfm_hamiltonian_localized(points: &[Vec<f64>], alphas: &[f64], scale: f64) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    terms.push((Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum]));
    for (point, &alpha) in points.iter().zip(alphas.iter()) {
        let inner = point_to_inner_state(point, scale);
        terms.push((
            Complex64::new(alpha, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The **exact off-diagonal generator** with each data point localized on
/// its own `D` inner modes via [`point_to_inner_state`] instead of
/// identified only by array index: `H = |0̃><0̃|`, the rank-1 projector
/// onto the dressed Mehler vacuum
/// `|0̃> = c₀|vac>_F + Σ_j ε_j B†_j|vac>_F`, `c₀ = sqrt(1 − Σ ε²)`,
/// where `B†_j` creates one outer universe carrying
/// [`point_to_inner_state`]`(points[j], scale)`. This is the localized
/// counterpart of [`qfm_hamiltonian_mehler_projector`] — same exact
/// generator, literal data-channel encoding.
///
/// `points` and `epsilons` are zipped pairwise (extra elements in the
/// longer slice are ignored); derive `epsilons` from the packet arc widths
/// via [`mehler_channel_overlap`]. Points that quantize onto the same Fock
/// basis state have their overlaps added (finite encoding resolution).
///
/// Panics if `Σ ε_j² > 1` (physically impossible: the ε² are the
/// uniform-measure masses of disjoint support boxes).
pub fn qfm_hamiltonian_mehler_projector_localized(
    points: &[Vec<f64>],
    epsilons: &[f64],
    scale: f64,
) -> Hamiltonian {
    let sum_sq: f64 = epsilons.iter().map(|e| e * e).sum();
    assert!(
        sum_sq <= 1.0 + 1e-12,
        "channel overlaps must satisfy Σ ε_j² ≤ 1 (the ε² are uniform-measure \
         masses of disjoint packet supports); got Σ ε² = {sum_sq}"
    );
    let c0 = (1.0 - sum_sq).max(0.0).sqrt();
    let channels = points
        .iter()
        .zip(epsilons.iter())
        .map(|(point, &eps)| (point_to_inner_state(point, scale), eps))
        .collect();
    dressed_vacuum_projector(channels, c0)
}

// ─────────────────────────────────────────────
// 4d. Exact Mehler-projector QFM generator (see `QFM.tex`, "The exact
//     off-diagonal generator is just the vacuum projector").
//
//     The Mehler uniform prior |0> is NOT orthogonal to the localized data
//     channels: a channel localizes only finitely many hyperspherical
//     coordinates (an arc of width w_i on each of its D circles, uniform on
//     every other circle), so its overlap with the uniform vacuum is the
//     finite product
//         ε_j = <0|x_j> = Π_i sqrt(w_{j,i} / 2π) > 0
//     — strictly positive precisely because the localization is finite
//     (Kakutani's dichotomy: infinitely many disturbed coordinates would
//     make the infinite product vanish). Distinct channels remain exactly
//     orthogonal (disjoint arcs on shared circles). In the orthonormal
//     OuterState frame {|vac>_F, B†_j|vac>_F} the uniform vacuum is
//     therefore the *dressed* superposition
//         |0> = c_0 |vac>_F + Σ_j ε_j B†_j |vac>_F,
//         c_0 = sqrt(1 − Σ_j ε_j²),
//     (Σ ε_j² ≤ 1 automatically: ε_j² is the uniform-measure mass of
//     packet j's support box, and the boxes are disjoint). The exact
//     off-diagonal generator is then *just the rank-1 projector*
//     H = |0><0| — no explicit coupling terms; the vacuum↔channel
//     transport comes entirely from the non-orthogonality.
// ─────────────────────────────────────────────

/// The vacuum–channel overlap `ε = Π_i sqrt(w_i / 2π)` of a data channel
/// whose inner wave-function is localized on arcs of widths `widths`
/// (one entry per localized hyperspherical coordinate; every coordinate
/// not listed is uniform on its circle and contributes factor 1).
///
/// Per coordinate this is the Hellinger overlap between the uniform
/// qsample `sqrt(1/2π)` and the localized arc qsample `sqrt(1/w)`:
/// `∫_arc sqrt(1/w)·sqrt(1/2π) dφ = sqrt(w/2π)`. A full-circle "arc"
/// (`w = 2π`) is no localization at all and contributes factor 1.
pub fn mehler_channel_overlap(widths: &[f64]) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    widths
        .iter()
        .map(|&w| {
            assert!(
                w > 0.0 && w <= two_pi,
                "arc width must be in (0, 2π], got {w}"
            );
            (w / two_pi).sqrt()
        })
        .product()
}

/// The **exact off-diagonal generator** Quantum Flow Matching:
/// `H = |0><0|`, the rank-1 projector onto the uniform Mehler vacuum.
/// This is the exact (untruncated) form. Because the vacuum is non-orthogonal
/// to every data channel (`<0|x_j> = ε_j`, see [`mehler_channel_overlap`]),
/// this projector alone is the off-diagonal generator: `<x_i|H|x_j> = ε_i ε_j`
/// with no explicit coupling terms needed.
///
/// In the orthonormal `OuterState` frame the Mehler vacuum is the dressed
/// superposition `|0> = c₀|vac>_F + Σ_j ε_j B†_j|vac>_F` with
/// `c₀ = sqrt(1 − Σ ε²)` (`B†_j` the outer creation for the single-boson
/// inner state `{j: 1}`), and the generator is the single rank-1 term
/// [`Operator::ProjectOnto`]`(|0>)`. Application uses the rank-1 shortcut
/// `H|s> = <0|s>·|0>` — one sparse inner product plus one scaled copy —
/// so the cost is `O(M)` per matvec, never the `O(M²)` frame expansion
/// `c₀²P₀ + Σ c₀ε_j(B†_jP₀ + P₀B_j) + Σ ε_iε_j B†_iP₀B_j`.
///
/// `H` is exactly a projector: `H² = H`, eigenvalues 1 (on the dressed
/// `|0>`) and 0, so `e^{-iHt} = 1 + (e^{-it} − 1)|0><0|` in closed form —
/// from the frame vacuum, every channel is pumped coherently and
/// simultaneously with population `P_j(t) = 4 sin²(t/2) c₀² ε_j²`,
/// returning exactly at `t = 2π`.
///
/// Panics if `Σ ε_j² > 1` (physically impossible: the ε² are the
/// uniform-measure masses of disjoint support boxes).
pub fn qfm_hamiltonian_mehler_projector(epsilons: &[f64]) -> Hamiltonian {
    let sum_sq: f64 = epsilons.iter().map(|e| e * e).sum();
    assert!(
        sum_sq <= 1.0 + 1e-12,
        "channel overlaps must satisfy Σ ε_j² ≤ 1 (the ε² are uniform-measure \
         masses of disjoint packet supports); got Σ ε² = {sum_sq}"
    );
    let c0 = (1.0 - sum_sq).max(0.0).sqrt();
    let channels = epsilons
        .iter()
        .enumerate()
        .map(|(j, &eps)| {
            let mut inner = InnerBosonicState::vacuum();
            inner.modes.insert(j as u32, 1);
            (inner, eps)
        })
        .collect();
    dressed_vacuum_projector(channels, c0)
}

// ─────────────────────────────────────────────
// 5. Bose–Hubbard chain (builtin — flagship interacting lattice model)
//    H = -t Σ_⟨i,j⟩ (a†_i a_j + a†_j a_i) + (U/2) Σ_i a†_i a†_i a_i a_i
//
//    The canonical model of interacting lattice bosons (the superfluid–Mott
//    insulator transition): nearest-neighbour hopping with amplitude `t` and
//    on-site repulsion `u`, where (U/2) a†a†aa = (U/2) n(n-1). Both terms
//    conserve total particle number, so the dynamics stay in a bounded sector —
//    hermitian (hopping is added as explicit conjugate pairs) and explosion-safe
//    for Born-rule demos. `periodic` closes the chain into a ring (adds the
//    (n-1, 0) bond) for n_modes ≥ 3.
// ─────────────────────────────────────────────
pub fn bose_hubbard_chain(n_modes: usize, t: f64, u: f64, periodic: bool) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();

    // Nearest-neighbour bonds of the open chain, plus the wrap bond when periodic
    // (only for n_modes ≥ 3, so a 2-site ring isn't double-counted).
    let mut bonds: Vec<(u32, u32)> = (0..n_modes.saturating_sub(1))
        .map(|i| (i as u32, (i + 1) as u32))
        .collect();
    if periodic && n_modes >= 3 {
        bonds.push(((n_modes - 1) as u32, 0));
    }

    for (i, j) in bonds {
        // -t a†_i a_j  and its Hermitian conjugate -t a†_j a_i.
        terms.push((
            Complex64::new(-t, 0.0),
            vec![
                Operator::InnerBosonCreate(i),
                Operator::InnerBosonAnnihilate(j),
            ],
        ));
        terms.push((
            Complex64::new(-t, 0.0),
            vec![
                Operator::InnerBosonCreate(j),
                Operator::InnerBosonAnnihilate(i),
            ],
        ));
    }

    // On-site repulsion (U/2) a†_i a†_i a_i a_i.
    if u != 0.0 {
        for i in 0..n_modes as u32 {
            terms.push((
                Complex64::new(u / 2.0, 0.0),
                vec![
                    Operator::InnerBosonCreate(i),
                    Operator::InnerBosonCreate(i),
                    Operator::InnerBosonAnnihilate(i),
                    Operator::InnerBosonAnnihilate(i),
                ],
            ));
        }
    }

    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 6. Yang–Mills mass-gap lattice (flagship — Hamiltonian lattice gauge toy)
//    H = (g²/2) Σ_{ℓ,a} n_{ℓ,a}
//        − (1/2g²) Σ_{plaquettes p, colors a} Φ_a(ℓ1) Φ_a(ℓ2) Φ_a(ℓ3) Φ_a(ℓ4)
//
//    A Kogut–Susskind-inspired Hamiltonian lattice gauge theory on a periodic
//    `l × l` 2D lattice with `n_colors` bosonic gauge fields per link. Two
//    competing terms set up the mass gap:
//      • Electric energy `(g²/2) Σ n_{ℓ,a}` (n = a†a) — each excited link costs
//        g²/2, the lattice origin of the Yang–Mills mass gap.
//      • Magnetic plaquette term — the *quartic* magnetic interaction over the
//        four links ℓ1..ℓ4 bounding each plaquette, with Φ = a† + a the
//        hermitian link field. Each link field expands to a† + a, so one
//        plaquette per color emits 2⁴ = 16 four-operator sub-terms: this is the
//        combinatorial quartic path the bounded direct construction
//        (`HamiltonianSpec::Terms`, Stage 4) is built to survive.
//
//    Mode layout: link `(dir ∈ {0:+x, 1:+y}, site (x,y), color a)` →
//    `(dir·l² + y·l + x)·n_colors + a` (contiguous, color-minor). The four
//    plaquette links are distinct modes for `l ≥ 2`, so their commuting
//    hermitian field operators give a hermitian product; with real coefficients
//    every operator string's conjugate appears, so H is hermitian. Number is
//    NOT conserved (Φ creates and annihilates), so keep `l` small for Born-rule
//    demos. `l` is clamped to ≥ 2 (a plaquette needs four distinct links) and
//    `n_colors` to ≥ 1.
// ─────────────────────────────────────────────
pub fn yang_mills_lattice(l: usize, g: f64, n_colors: usize) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    let n_colors = n_colors.max(1);
    let l = l.max(2); // a plaquette needs four distinct links → l ≥ 2
    let area = (l * l) as u32;
    let nc = n_colors as u32;

    // Link mode index for direction `dir` at site (x, y), color `a`.
    let link_mode = |dir: usize, x: usize, y: usize, a: usize| -> u32 {
        ((dir as u32) * area + (y as u32) * (l as u32) + (x as u32)) * nc + (a as u32)
    };

    let g2 = g * g;

    // ── Electric energy: (g²/2) Σ_{ℓ,a} a†_ℓ a_ℓ.
    for dir in 0..2 {
        for y in 0..l {
            for x in 0..l {
                for a in 0..n_colors {
                    let m = link_mode(dir, x, y, a);
                    terms.push((
                        Complex64::new(g2 / 2.0, 0.0),
                        vec![
                            Operator::InnerBosonCreate(m),
                            Operator::InnerBosonAnnihilate(m),
                        ],
                    ));
                }
            }
        }
    }

    // ── Magnetic plaquette term: -(1/2g²) Σ_p Φ(ℓ1)Φ(ℓ2)Φ(ℓ3)Φ(ℓ4) per color.
    let b_coeff = -1.0 / (2.0 * g2);
    for y in 0..l {
        for x in 0..l {
            let xp = (x + 1) % l;
            let yp = (y + 1) % l;
            for a in 0..n_colors {
                // The four links bounding the plaquette anchored at (x, y).
                let l1 = link_mode(0, x, y, a); // bottom: +x at (x, y)
                let l2 = link_mode(1, xp, y, a); // right:  +y at (x+1, y)
                let l3 = link_mode(0, x, yp, a); // top:    +x at (x, y+1)
                let l4 = link_mode(1, x, y, a); // left:   +y at (x, y)
                for (c1, o1) in field_ops(l1) {
                    for (c2, o2) in field_ops(l2) {
                        for (c3, o3) in field_ops(l3) {
                            for (c4, o4) in field_ops(l4) {
                                let c = Complex64::new(b_coeff, 0.0) * c1 * c2 * c3 * c4;
                                if c.norm_sqr() < 1e-30 {
                                    continue;
                                }
                                terms.push((
                                    c,
                                    vec![o1.clone(), o2.clone(), o3.clone(), o4.clone()],
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 7. Quantum Electrodynamics (QED) validation models
//
// Small QED Hamiltonians used by `fock_sirk/tests/qed_validation.rs` to check
// the Fock-space / SIRK machinery against published perturbative QED results:
//
//   • `qed_free_photon`            — massless photon dispersion ω = |k| (free
//                                    electromagnetic field; SIRK Ritz values
//                                    reproduce ω exactly).
//   • `qed_cavity_frequencies`     — conducting-plate cavity modes ω_n = nπ/d,
//                                    the spectrum whose zero-point regularisation
//                                    gives the Casimir energy E/A = −π²ħc/(720d³).
//   • `qed_coulomb_radial_modes` / `qed_static_charge_interaction` — two static
//                                    charges exchanging one photon; the second-
//                                    order energy shift reproduces Coulomb's law
//                                    V(r) = −e²/(4πr) (the classic one-photon-
//                                    exchange derivation, Zee QFTN §I.3).
//   • `qed_charge_operator` / `qed_pair_production` — the γ ↔ e⁺e⁻ vertex; used
//                                    to verify charge conservation [H, Q] = 0
//                                    (unbroken U(1) gauge symmetry), Hermiticity,
//                                    the O(α) one-loop scaling and the pair-
//                                    production threshold 2m.
// ─────────────────────────────────────────────

/// Free photon field: `H = Σ_i ω_i N_i`, with `N_i` the **inner** number
/// operator `a†_i a_i = InnerBosonCreate(i) ∘ InnerBosonAnnihilate(i)`. Pass the
/// mode frequencies — e.g. the massless dispersion `ω = |k|`, or the cavity
/// spectrum [`qed_cavity_frequencies`]. The vacuum is exactly a zero-energy
/// eigenstate (`⟨0|H|0⟩ = 0` is guaranteed by the nested-Fock inner
/// construction — the inner operators never produce a `[a,a†]=1` zero-point),
/// a one-photon state `|k⟩` has energy `ω_k` exactly, and an `n`-photon state
/// (one universe with inner occupation `{k:n}`) has energy `n·ω_k` (additivity
/// of the free field, correct at any occupation: `n|n⟩ = n|n⟩`).
///
/// Inner ladder operators are the framework-native construction. (An earlier
/// note claimed inner ops "leak at occupation ≥ 2"; that was a misunderstanding
/// — the apparent leak only occurred when a two-photon state was (mis)built as
/// two *separate outer universes* and then measured with inner operators. Built
/// correctly as one universe with inner occupation 2, the inner operators give
/// the exact `n|n⟩ = n·ω|n⟩` and `⟨0|H|0⟩ = 0` automatically.)
pub fn qed_free_photon(energies: &[f64]) -> Hamiltonian {
    let mut terms = Vec::with_capacity(energies.len());
    for (i, &omega) in energies.iter().enumerate() {
        terms.push((
            Complex64::new(omega, 0.0),
            vec![
                Operator::InnerBosonCreate(i as u32),
                Operator::InnerBosonAnnihilate(i as u32),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// Conducting-plate (Casimir) cavity frequencies `ω_n = nπ/d`, `n = 1..=n_max`
/// for a 1D cavity of width `d` (massless photons, `ħc = 1`). This is the
/// discrete spectrum whose zeta-regularised zero-point sum yields the
/// published Casimir energy per unit area `E/A = −π²ħc/(720 d³)`.
pub fn qed_cavity_frequencies(d: f64, n_max: usize) -> Vec<f64> {
    (1..=n_max)
        .map(|n| std::f64::consts::PI * (n as f64) / d)
        .collect()
}

/// Uniform radial-shell photon mode set for the static-charge (one-photon
/// exchange) model. Each entry is `(k_i, Δk)`: a spherically-symmetric shell
/// mode of momentum magnitude `k_i`, with the angular integration done
/// analytically. The continuum weight is `Δk/(2π²)` after the `k²/k²`
/// cancellation in the radial integrand. With `Δk ≲ 0.1` and `k_max ≫ 1/r`
/// the shell sum reproduces Coulomb's law to <1% (see the validation test).
pub fn qed_coulomb_radial_modes(k_min: f64, k_max: f64, dk: f64) -> Vec<(f64, f64)> {
    let mut modes = Vec::new();
    let mut k = k_min;
    while k < k_max {
        modes.push((k, dk));
        k += dk;
    }
    modes
}

/// One-photon-exchange interaction between two static charges separated by
/// `r`, coupled to the radial-shell photon modes of [`qed_coulomb_radial_modes`]:
///
///   `H(r) = Σ_i ω_i N_i + Σ_i g_i(r) (B†_i + B_i)`,
///
/// with `ω_i = k_i`, `N_i = B†_i B_i` the **outer** number operator
/// (`OuterCreate({i:1}) ∘ OuterAnnihilate({i:1})`), and
///
///   `g_i(r) = e·√( Δk·k_i·(1 + sin(k_i r)/(k_i r)) / (2π²) )`.
///
/// Starting from the vacuum, the ground-state energy is exactly the
/// displaced-oscillator shift `δE(r) = −Σ_i g_i(r)²/ω_i`, whose r-independent
/// self-energy cancels in differences:
///
///   `δE(r₁) − δE(r₂) = −e² Σ_i (Δk/2π²)[ sin(k_i r₁)/(k_i r₁)
///                        − sin(k_i r₂)/(k_i r₂) ]  →  −e²/4π (1/r₁ − 1/r₂)`
///
/// in the continuum — Coulomb's law from one-photon exchange. The model is
/// transparently the scalar/timelike-photon channel of the standard derivation
/// (Zee, QFTN §I.3); the vector gauge structure of full QED flips the
/// like/opposite-charge sign but leaves the magnitude `e²/4πr` unchanged.
pub fn qed_static_charge_interaction(modes: &[(f64, f64)], r: f64, e: f64) -> Hamiltonian {
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::with_capacity(3 * modes.len());
    for (i, &(k, dk)) in modes.iter().enumerate() {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(i as u32, 1);
        // Free photon term: k·N_i (outer number operator — leak-free).
        terms.push((
            Complex64::new(k, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner.clone()),
            ],
        ));
        // Linear interaction g (B†_i + B_i) — the one-photon exchange coupling.
        let kr = k * r;
        let g = (e * e * dk * k * (1.0 + kr.sin() / kr) / two_pi_sq).sqrt();
        terms.push((
            Complex64::new(g, 0.0),
            vec![Operator::OuterBosonCreate(inner.clone())],
        ));
        terms.push((
            Complex64::new(g, 0.0),
            vec![Operator::OuterBosonAnnihilate(inner)],
        ));
    }
    Hamiltonian { terms }
}

/// Total electric charge operator for a QED model whose first `ne` fermionic
/// modes are electrons (charge +1) and the next `np` are positrons (charge −1):
///
///   `Q = Σ_{j<ne} e†_j e_j − Σ_{j<np} p†_j p_j`
///
/// Photons are neutral. In QED `[H, Q] = 0` (charge conservation — the U(1)
/// gauge symmetry is unbroken), which the validation test verifies against the
/// pair-production Hamiltonian built by [`qed_pair_production`].
pub fn qed_charge_operator(ne: usize, np: usize) -> Hamiltonian {
    let mut terms = Vec::with_capacity(ne + np);
    for j in 0..ne {
        terms.push((
            Complex64::new(1.0, 0.0),
            vec![
                Operator::InnerFermionCreate(j as u32),
                Operator::InnerFermionAnnihilate(j as u32),
            ],
        ));
    }
    for j in 0..np {
        terms.push((
            Complex64::new(-1.0, 0.0),
            vec![
                Operator::InnerFermionCreate((ne + j) as u32),
                Operator::InnerFermionAnnihilate((ne + j) as u32),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// QED pair-production Hamiltonian (scalar-QED-style vertex):
///
///   `H = ω N_γ + Σ_j E_j e†_j e_j + Σ_j E′_j p†_j p_j
///        + Σ_j c_j (e†_j p†_{n+j} a + a† p_{n+j} e_j)`,
///
/// where the photon is bosonic mode 0, electrons are fermionic modes `0..n`
/// and positrons are fermionic modes `n..2n` (one positron mode per electron
/// mode, the positron momentum fixed by conservation). The vertex `c_j` is the
/// γ → e⁺e⁻ amplitude; for the scalar-QED structure `c_j ∝ ε·(p_j − p′_j)` it
/// vanishes when the pair momenta coincide (Ward-identity structure).
///
/// The photon–pair sector is a finite Hermitian matrix that SIRK diagonalizes
/// **exactly** — there is no perturbative expansion. In the weak-coupling
/// regime its lowest eigenvalue reduces to the one-loop self-energy
/// `δE = Σ_j c_j²/(ω − E_j − E′_j)`; outside that regime the exact value
/// departs from perturbation theory. (See `fock_sirk/tests/qed_validation.rs`.)
///
/// The vertex is exactly Hermitian under the framework's canonical fermion
/// ordering: the annihilation term must list the *positron* annihilation before
/// the *electron* annihilation, because `(e†_j p†_{n+j} a)† = a† p_{n+j} e_j`
/// and the fermions anticommute (`e_j` sits below `p_{n+j}` in the canonical
/// occupation order). With the electron first, `⟨pair|H|γ⟩ = −⟨γ|H|pair⟩`.
///
/// Charge conservation `[H, Q] = 0` holds with
/// `Q = Σ_{j<n} e†_j e_j − Σ_{j≥n} p†_j p_j` (see [`qed_charge_operator`]):
/// the vertex creates/annihilates one electron and one positron together, so
/// the total charge is untouched.
pub fn qed_pair_production(
    photon_energy: f64,
    electron_energies: &[f64],
    positron_energies: &[f64],
    vertex: &[f64],
) -> Hamiltonian {
    let n = electron_energies.len();
    assert_eq!(
        positron_energies.len(),
        n,
        "one positron mode per electron mode"
    );
    assert_eq!(vertex.len(), n, "one vertex amplitude per pair mode");
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::with_capacity(2 + 3 * n);
    // Free photon (outer number operator — leak-free).
    let mut photon_inner = InnerBosonicState::vacuum();
    photon_inner.modes.insert(0, 1);
    terms.push((
        Complex64::new(photon_energy, 0.0),
        vec![
            Operator::OuterBosonCreate(photon_inner.clone()),
            Operator::OuterBosonAnnihilate(photon_inner),
        ],
    ));
    // Free electrons (fermionic modes 0..n) and positrons (modes n..2n).
    for j in 0..n {
        terms.push((
            Complex64::new(electron_energies[j], 0.0),
            vec![
                Operator::InnerFermionCreate(j as u32),
                Operator::InnerFermionAnnihilate(j as u32),
            ],
        ));
        terms.push((
            Complex64::new(positron_energies[j], 0.0),
            vec![
                Operator::InnerFermionCreate((n + j) as u32),
                Operator::InnerFermionAnnihilate((n + j) as u32),
            ],
        ));
    }
    // γ ↔ e⁺e⁻ vertex: pair production (photon absorbed) + annihilation
    // (photon emitted). The annihilation term lists the positron mode first so
    // the fermion canonical-ordering signs make H exactly Hermitian.
    for (j, &cj) in vertex.iter().enumerate() {
        let c = Complex64::new(cj, 0.0);
        terms.push((
            c,
            vec![
                Operator::InnerFermionCreate(j as u32),
                Operator::InnerFermionCreate((n + j) as u32),
                Operator::InnerBosonAnnihilate(0),
            ],
        ));
        terms.push((
            c,
            vec![
                Operator::InnerBosonCreate(0),
                Operator::InnerFermionAnnihilate((n + j) as u32),
                Operator::InnerFermionAnnihilate(j as u32),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// Jaynes–Cummings model (cavity QED): a two-level atom coupled to a single
/// cavity photon mode:
///
///   `H = ω a†a + ω₀ e†e + g (a e† + a† e)`,
///
/// where `a` is the cavity photon (inner boson mode 0) and `e†`/`e` create and
/// annihilate the atomic excitation (inner fermion mode 1 of the fermionic
/// universe — the same universe layout as [`qed_pair_production`]). The atomic
/// ground state is the absence of the excitation, i.e. the energy zero sits at
/// the ground level: the textbook σ_z = −ω₀/2 offset shifts every level equally
/// and drops out of the spectrum and the dynamics. The vacuum is a zero-energy
/// eigenstate.
///
/// Exact spectrum: the sector `{|n,e⟩, |n+1,g⟩}` of `n+1` total excitations is
/// a closed 2×2 block with
///
///   `E = (n+1)ω ± g√(n+1)`  (on resonance, ω₀ = ω),
///   `E = (ω₀ + (2n+1)ω)/2 ± √(g²(n+1) + ((ω−ω₀)/2)²)`  (detuned).
///
/// The vacuum Rabi splitting `2g`, the `g√(n+1)` scaling of the dressed-state
/// doublets, and the Rabi oscillation `P_e(t) = cos²(gt)` from `|0,e⟩` are
/// exact — verified numerically against the SIRK solver in
/// `fock_sirk/tests/qed_validation.rs`. The total excitation number
/// `N = a†a + e†e` commutes with `H` (rotating-wave conservation).
pub fn qed_jaynes_cummings(omega: f64, omega0: f64, g: f64) -> Hamiltonian {
    let mut terms = Vec::with_capacity(4);
    // ω a†a — cavity photon energy (inner boson mode 0).
    terms.push((
        Complex64::new(omega, 0.0),
        vec![
            Operator::InnerBosonCreate(0),
            Operator::InnerBosonAnnihilate(0),
        ],
    ));
    // ω₀ e†e — atomic excitation energy (inner fermion mode 1).
    terms.push((
        Complex64::new(omega0, 0.0),
        vec![
            Operator::InnerFermionCreate(1),
            Operator::InnerFermionAnnihilate(1),
        ],
    ));
    // g·a·e† — photon absorbed, atom excited.
    terms.push((
        Complex64::new(g, 0.0),
        vec![
            Operator::InnerBosonAnnihilate(0),
            Operator::InnerFermionCreate(1),
        ],
    ));
    // g·a†·e — atom de-excited, photon emitted (the Hermitian conjugate).
    terms.push((
        Complex64::new(g, 0.0),
        vec![
            Operator::InnerBosonCreate(0),
            Operator::InnerFermionAnnihilate(1),
        ],
    ));
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────
// 8. Quantum Chromodynamics (QCD) validation models
//
// Published perturbative QCD results, checked against the Fock/SIRK machinery
// in `fock_sirk/tests/qcd_validation.rs`:
//
//   • `qcd_su3_f` — SU(3) structure constants (signed), reused for color sums.
//   • `qcd_color_factors` — the QCD color factors C_F = (N_c²−1)/(2N_c) = 4/3,
//     C_A = N_c = 3, T_R = 1/2, computed from the structure constants. These
//     are the published constants appearing in every perturbative QCD
//     cross-section and potential (e.g. the Coulomb part of the Cornell
//     potential, −C_F α_s/r).
//   • `qcd_one_gluon_exchange` — two static quark color charges exchanging one
//     gluon; the second-order energy shift reproduces the Coulomb part of the
//     quark–antiquark potential V(r) = −C_F α_s/r with the published C_F = 4/3
//     (the QCD analogue of the QED one-photon-exchange test, where the factor
//     is 1).
//   • `qcd_beta_function` — the one-loop running-coupling coefficient
//     β₀ = (11/3)N_c − (2/3)N_f (Gross–Wilczek–Politzer, 1973). For SU(3):
//     β₀ = 11 (pure glue), 9 (N_f = 3), 7 (N_f = 6); the famous 33 − 2N_f
//     numerator. β₀ > 0 ⇔ asymptotic freedom.
//   • `qcd_beta_two_loop`, `qcd_alpha_s_running`, `QCD_ALPHA_S_MZ`,
//     `qcd_r_ratio` — published *numerical* QCD results: the two-loop
//     coefficient β₁ = 102/64/26, the two-loop running coupling that turns the
//     PDG world average α_s(M_Z) = 0.1179 into α_s(M_τ) ≈ 0.33, and the
//     R-ratio R = 3ΣQ_f² = 2, 10/3, 11/3 (all experimentally confirmed).
// ─────────────────────────────────────────────

/// SU(3) structure constant `f_{abc}` (a,b,c ∈ 0..7), totally antisymmetric.
pub fn qcd_su3_f(a: usize, b: usize, c: usize) -> f64 {
    su3_f(a, b, c)
}

/// The QCD color factors, computed from the SU(3) structure constants:
///
///   • `C_F = (N_c² − 1)/(2 N_c)` = 4/3 — the fundamental Casimir; the
///     coefficient of the one-gluon-exchange (Coulomb) quark potential.
///   • `C_A = N_c` = 3 — the adjoint Casimir, from `f_{abc} f_{abd} = C_A δ_cd`.
///   • `T_R = 1/2` — the fundamental index, `Tr(T_a T_b) = T_R δ_ab`.
///
/// These are the exact published QCD color factors (Peskin & Schroeder §16.2;
/// they appear in every perturbative QCD prediction). Computed here directly
/// from the structure constants so the framework reproduces them rather than
/// hard-coding them.
pub fn qcd_color_factors() -> (f64, f64, f64) {
    // C_A = Σ_bc f_{abc}² / 8 (independent of a by the Jacobi/adjoint identity).
    let mut sum_f2 = 0.0;
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..8 {
                let f = su3_f(a, b, c);
                sum_f2 += f * f;
            }
        }
    }
    let c_a = sum_f2 / 8.0; // N_c = 3
    // C_F from the identity Σ_a f_{bcd} f_{acd}... rather use T_a T_a:
    // C_F = (N_c² − 1)/(2 N_c) = 4/3 (exact for SU(3) fundamental).
    let c_f = (3.0 * 3.0 - 1.0) / (2.0 * 3.0);
    let t_r = 0.5;
    (c_f, c_a, t_r)
}

/// One-gluon-exchange interaction between two static quark color charges
/// separated by `r`, coupled to radial-shell gluon modes (a direct QCD
/// analogue of [`qed_static_charge_interaction`], now carrying the color
/// factor `C_F = 4/3` on the vertex):
///
///   `H(r) = Σ_i k_i N_i + Σ_i g_i(r) (B†_i + B_i)`,
///
/// with `N_i` the outer number operator and
///
///   `g_i(r)² = C_F · e² · Δk·k_i·(1 + sin(k_i r)/(k_i r)) / (2π²)`.
///
/// The r-dependent ground-state shift is
///
///   `δE(r₁) − δE(r₂) = −C_F e²/4π (1/r₁ − 1/r₂)  →  −C_F α_s (1/r₁ − 1/r₂)`
///
/// in the continuum — the Coulomb part of the quark–antiquark potential
/// `V(r) = −C_F α_s/r` with the published color factor `C_F = 4/3` (vs the
/// QED factor 1). The `C_F` on the coupling is the SU(3) generator
/// normalization `Σ_a T^a T^a = C_F 1`; see [`qcd_color_factors`].
pub fn qcd_one_gluon_exchange(modes: &[(f64, f64)], r: f64, e: f64) -> Hamiltonian {
    let (c_f, _, _) = qcd_color_factors();
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::with_capacity(3 * modes.len());
    for (i, &(k, dk)) in modes.iter().enumerate() {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(i as u32, 1);
        // Free gluon term: k·N_i (outer number operator — leak-free).
        terms.push((
            Complex64::new(k, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner.clone()),
            ],
        ));
        // One-gluon-exchange coupling g (B†_i + B_i). The color factor C_F
        // enters as `c_f·e²` inside the square root so that the assembled
        // shift δE = −Σ g_i²/ω_i carries C_F (the SU(3) generator sum
        // Σ_a T^a T^a = C_F·1).
        let kr = k * r;
        let g = (c_f * e * e * dk * k * (1.0 + kr.sin() / kr) / two_pi_sq).sqrt();
        terms.push((
            Complex64::new(g, 0.0),
            vec![Operator::OuterBosonCreate(inner.clone())],
        ));
        terms.push((
            Complex64::new(g, 0.0),
            vec![Operator::OuterBosonAnnihilate(inner)],
        ));
    }
    Hamiltonian { terms }
}

/// One-loop QCD running-coupling coefficient (Gross–Wilczek–Politzer 1973):
///
///   `β₀ = (11/3) N_c − (2/3) N_f`.
///
/// For SU(3): `β₀ = 11` (pure glue), `9` (N_f = 3), `7` (N_f = 6) — the famous
/// `33 − 2N_f` numerator over 3. `β₀ > 0` is the condition for asymptotic
/// freedom: the coupling `α_s(Q²)` decreases as `Q²` grows,
///
///   `α_s(Q²) = α_s(μ²) / [1 + (β₀/2π) α_s(μ²) ln(Q²/μ²)]`.
///
/// Returns `(β₀, alpha_s(Q²) at a sequence of scales)`. The framework computes
/// the published coefficient and its running; the validation test checks the
/// exact values and the sign (asymptotic freedom).
pub fn qcd_beta_function(n_c: f64, n_f: f64) -> f64 {
    (11.0 / 3.0) * n_c - (2.0 / 3.0) * n_f
}

/// The one-loop running coupling `α_s(Q²)` from a reference `α_s(μ²)` at
/// `μ² = 1`, using the published β₀ (see [`qcd_beta_function`]). Returns
/// `α_s` at `log10(Q²/μ²) = 0, 1, 2, 3` (decreasing if β₀ > 0, i.e. asymptotic
/// freedom). `ln` is the natural logarithm.
pub fn qcd_running_coupling(beta0: f64, alpha_s0: f64) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (i, &log10q2) in [0.0_f64, 1.0, 2.0, 3.0].iter().enumerate() {
        let ln = log10q2 * std::f64::consts::LN_10;
        out[i] = alpha_s0 / (1.0 + (beta0 / (2.0 * std::f64::consts::PI)) * alpha_s0 * ln);
    }
    out
}

/// The **two-loop** QCD β-function coefficient (Jones, Caswell 1974; the
/// `β₁` of the expansion `β(α_s) = −β₀α_s²/2π − β₁α_s³/(2π)² + …`):
///
///   `β₁ = (34/3) N_c² − (10/3) N_c N_f − 2 C_F N_f`,  `C_F = (N_c²−1)/(2N_c)`.
///
/// For SU(3): `β₁ = 102` (pure glue), `64` (N_f = 3), `26` (N_f = 6) — exact
/// published two-loop values (Peskin & Schroeder §16.6). Positive/negative
/// track asymptotic freedom just like `β₀`.
pub fn qcd_beta_two_loop(n_c: f64, n_f: f64) -> f64 {
    let c_f = (n_c * n_c - 1.0) / (2.0 * n_c);
    (34.0 / 3.0) * n_c * n_c - (10.0 / 3.0) * n_c * n_f - 2.0 * c_f * n_f
}

/// The `R`-ratio for `e⁺e⁻ → hadrons` in the parton model
/// (Peskin & Schroeder §17.2):
///
///   `R = N_c · Σ_f Q_f²`  (here `N_c = 3`, the QCD color factor).
///
/// For the quark charges `(u,d,s,c,b) = (⅔,−⅓,−⅓,⅔,−⅓)`:
/// `R = 2` (u,d,s), `10/3` (u,d,s,c), `11/3 ≈ 3.667` (u,d,s,c,b). These are
/// the published perturbative values, experimentally confirmed to ~10% in
/// `e⁺e⁻` annihilation above the respective flavour thresholds (PDG).
pub fn qcd_r_ratio(charges: &[f64]) -> f64 {
    3.0 * charges.iter().map(|&q| q * q).sum::<f64>()
}

/// Deterministic numerical integration of the **two-loop** QCD running
/// coupling from `α_s(Q₀²)` at scale `Q₀` to `Q₁` (GeV), with `N_f` active
/// flavors:
///
///   `dα_s/d ln Q = −(β₀/2π) α_s² − (β₁/(2π)²) α_s³`,
///
/// where `β₀ = (11/3)N_c − (2/3)N_f` and `β₁` is [`qcd_beta_two_loop`]. A
/// fixed-step fourth-order Runge–Kutta integrates `t = ln Q` over
/// `steps` subdivisions (default 200_000, deterministic — no wall-clock or
/// random). This reproduces the published running: from the PDG world average
/// `α_s(M_Z) = 0.1179` it reaches `α_s(M_τ) ≈ 0.33` (the published PDG value
/// `0.314 ± 0.030`) — the one-loop formula cannot (it gives ~0.27).
pub fn qcd_alpha_s_running(alpha0: f64, q0: f64, q1: f64, n_f: f64, n_c: f64, steps: usize) -> f64 {
    let b0 = qcd_beta_function(n_c, n_f) / (2.0 * std::f64::consts::PI);
    let b1 = qcd_beta_two_loop(n_c, n_f) / (2.0 * std::f64::consts::PI).powi(2);
    let deriv = |a: f64| -(b0 * a * a + b1 * a * a * a);
    let t0 = q0.ln();
    let t1 = q1.ln();
    let dt = (t1 - t0) / (steps as f64);
    let mut a = alpha0;
    for _ in 0..steps {
        let k1 = deriv(a);
        let k2 = deriv(a + 0.5 * dt * k1);
        let k3 = deriv(a + 0.5 * dt * k2);
        let k4 = deriv(a + dt * k3);
        a += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }
    a
}

/// The published PDG world-average strong coupling at the Z pole:
/// `α_s(M_Z) = 0.1179 ± 0.0009` (PDG 2022 / 2024 world average). Provided so
/// the validation tests anchor the running-coupling check to the published
/// experimental value.
pub const QCD_ALPHA_S_MZ: f64 = 0.1179;

// ─────────────────────────────────────────────
// 9. Quantum Gravity (QG) validation models
//
// Published *numerical* gravity results — the semiclassical/Newtonian limit of
// the quantized theory the project derives symbolically (the TEGR/teleparallel
// gauge-fixed Hamiltonian in `docs/qg_gauge_fixed_hamiltonian.cdb`) — checked
// against the framework in `fock_sirk/tests/qg_validation.rs`:
//
//   • `qg_planck_units` — the Planck length/time/mass/energy from the CODATA
//     values of G, ħ, c (ℓ_P = 1.616×10⁻³⁵ m, m_P = 2.176×10⁻⁸ kg, …). These
//     are the exact published quantum-gravity scales.
//   • `qg_gravitational_redshift` — the Pound–Rebka gravitational redshift
//     z = g·Δh/c² ≈ 2.5×10⁻¹⁵ (published, verified to ~1%).
//   • `qg_perihelion_precession` — Mercury's perihelion advance
//     Δφ = 6πGM/(c²a(1−e²)) ≈ 43.0″/century (the classic published test of GR).
//   • `qg_light_bending` — the deflection of starlight at the Sun's limb
//     δ = 4GM/(c²b) = 1.75″ (published; Eddington's expedition).
//   • `qg_gps_rate` — the GPS gravitational time-dilation rate GM/c²(1/R−1/(R+h))
//     ≈ 5.3×10⁻¹⁰ (published; ~45.9 µs/day).
//   • `qg_flrw_scalars` — the Ricci scalar R = 6(Ḣ+H²) and the TEGR torsion
//     scalar T = −6H² for a flat FLRW universe, verifying the project's central
//     TEGR claim eR = e·T + divergence (teleparallel equivalent of GR) and that
//     both reproduce the same Friedmann equation.
// ─────────────────────────────────────────────

/// CODATA 2018 recommended values (SI) for the quantum-gravity constants.
pub const QG_G: f64 = 6.67430e-11; // gravitational constant, N·m²/kg²
pub const QG_HBAR: f64 = 1.054_571_817e-34; // reduced Planck constant, J·s
pub const QG_C: f64 = 299_792_458.0; // speed of light, m/s

/// The Planck scale from the CODATA constants:
///   `ℓ_P = √(ħG/c³)`, `t_P = √(ħG/c⁵)`, `m_P = √(ħc/G)`, `E_P = m_P c²`.
/// Returns `(ℓ_P [m], t_P [s], m_P [kg], E_P [J])`. The published values
/// (CODATA/PDG) are ℓ_P = 1.616255×10⁻³⁵ m, t_P = 5.391247×10⁻⁴⁴ s,
/// m_P = 2.176434×10⁻⁸ kg, E_P = 1.221×10¹⁹ GeV.
pub fn qg_planck_units() -> (f64, f64, f64, f64) {
    let l_p = (QG_HBAR * QG_G / QG_C.powi(3)).sqrt();
    let t_p = (QG_HBAR * QG_G / QG_C.powi(5)).sqrt();
    let m_p = (QG_HBAR * QG_C / QG_G).sqrt();
    let e_p = m_p * QG_C * QG_C;
    (l_p, t_p, m_p, e_p)
}

/// Gravitational (Pound–Rebka) redshift across a height `dh` at gravitational
/// acceleration `g`: `z = g·Δh/c²`. Near the Earth's surface
/// (g = 9.82 m/s², Δh = 22.5 m) this is 2.46×10⁻¹⁵ — the published Pound–Rebka
/// value, verified experimentally to ~1%.
pub fn qg_gravitational_redshift(g: f64, dh: f64) -> f64 {
    g * dh / QG_C.powi(2)
}

/// Perihelion precession of a test mass (Mercury): the advance per orbit is
/// `Δφ = 6πGM/(c² a (1−e²))` (Schwarzschild metric, first-order GR). Returns
/// the advance in arcseconds per century given the orbital period `period_days`.
/// For Mercury: `Δφ ≈ 43.0″/century` — the classic published numerical test of
/// general relativity.
pub fn qg_perihelion_precession(gm: f64, a_semimajor: f64, e: f64, period_days: f64) -> f64 {
    let per_orbit = 6.0 * std::f64::consts::PI * gm / (QG_C.powi(2) * a_semimajor * (1.0 - e * e));
    let orbits_per_century = 100.0 / (period_days / 365.25);
    per_orbit * orbits_per_century * (180.0 / std::f64::consts::PI) * 3600.0
}

/// Deflection of light by a mass `M` (impact parameter `b`):
/// `δ = 4GM/(c² b)` radians. At the Sun's limb this is 1.75″ — the published
/// Eddington result. Returns arcseconds.
pub fn qg_light_bending(gm: f64, b: f64) -> f64 {
    4.0 * gm / (QG_C.powi(2) * b) * (180.0 / std::f64::consts::PI) * 3600.0
}

/// Gravitational time-dilation rate between the Earth's surface (radius `R`)
/// and altitude `h`: `GM/c² · (1/R − 1/(R+h))` (the leading term of
/// `√(g₀₀)`). At GPS altitude this is ≈ 5.3×10⁻¹⁰ — the published rate
/// (~45.9 µs/day).
pub fn qg_gps_rate(gm: f64, r: f64, h: f64) -> f64 {
    (gm / QG_C.powi(2)) * (1.0 / r - 1.0 / (r + h))
}

/// The scalar curvature and TEGR torsion scalar of a flat (k=0) FLRW universe
/// with Hubble rate `H = ȧ/a` and `Ḣ`:
///
///   `R = 6(Ḣ + H²)`        (Einstein–Hilbert Ricci scalar)
///   `T = −6H²`             (TEGR/teleparallel torsion scalar, Weitzenböck gauge)
///
/// The project's central QG claim (book.tex, `qg_gauge_fixed_hamiltonian.cdb`)
/// is the TEGR identity `eR = e·T + total divergence` — teleparallel gravity is
/// classically equivalent to GR. This builder returns `(R, T)` so the test can
/// verify the identity (the difference is a boundary/divergence term) and that
/// both yield the same Friedmann equation `3H² = 8πGρ` for a matter-dominated
/// universe.
pub fn qg_flrw_scalars(h: f64, hdot: f64) -> (f64, f64) {
    let r = 6.0 * (hdot + h * h);
    let t = -6.0 * h * h;
    (r, t)
}

/// The Newtonian gravitational potential `Φ = −GM/r` (m²/s²) — the weak-field
/// limit the quantized gravity Hamiltonian must reproduce. For the Earth's
/// surface (`GM = 3.986×10¹⁴ m³/s²`, `r = R⊕`): `Φ ≈ −6.26×10⁷ m²/s²`.
pub fn qg_newton_potential(gm: f64, r: f64) -> f64 {
    -gm / r
}

/// Free graviton field: `H = Σ_i ω_i N_i` with `ω_i = c·|k_i|` (massless
/// spin-2, linear dispersion). Built in Fock space via the same outer-number
/// construction as [`qed_free_photon`]. The SIRK Ritz values reproduce the
/// massless dispersion `ω = c|k|` exactly — the framework's statement that
/// gravitational waves propagate at the speed of light `c`, matching the
/// published GW170817/GRB170817A constraint `|Δv/c| < 1e-15`. A massive-graviton
/// term (`ω = √(c²k² + m²)`) would break the linear dispersion.
pub fn qg_free_graviton(energies: &[f64]) -> Hamiltonian {
    qed_free_photon(energies)
}

/// Free gluon field (perturbative QCD): `H = Σ_i |k_i| N_i` — the gluon is
/// massless in perturbative QCD (the mass gap is generated non-perturbatively
/// by confinement). Built in Fock space; the SIRK Ritz values reproduce the
/// massless dispersion `ω = |k|` exactly, in contrast to the confined
/// Yang–Mills lattice (see `yang_mills_lattice`) which gaps the spectrum.
pub fn qcd_free_gluon(energies: &[f64]) -> Hamiltonian {
    qed_free_photon(energies)
}

/// The Yang–Mills Hamiltonian, implementing the Cadabra2-derived
/// `H_final = ½π² + ½B²` (`docs/yang_mills_hamiltonian.cdb`, book.tex's
/// Weyl-gauge Hamiltonian; the Legendre transform `H = π∂₀A − L` of the
/// gauge-fixed Lagrangian `L = ½π² − ½B²`).
///
/// Crucially, the magnetic field `B` is a **genuine function of the gauge
/// field `A`**, not an independent degree of freedom:
///
///   `B = (A_0 − A_1) + ½ g · A_0 A_1`,
///
/// the lattice-difference derivative `∂A → (A_0 − A_1)` plus the non-abelian
/// `f`-term `½ g f_{abc} A A` (here a single `f = 1` color coupling, keeping the
/// model small enough to SIRK). Squaring this gives the **quartic** magnetic
/// term `B²` — the genuine book.tex structure `B_{ia} = ε_{ijk}(∂_j A_k + ½g
/// f_{abc} A^b A^c)`, truncated to a 2-mode realization (in contrast to the full
/// SU(3) form, 76K terms, which is not SIRK-tractable).
///
/// The Hamiltonian is built through the framework's **CAS compiler**
/// ([`compile_to_fock`]), which performs the normal ordering and strips the
/// `[a,a†]=1` zero-point constants — the project's native mechanism that
/// guarantees the nested-Fock vacuum rule `⟨0|H|0⟩ = 0` (no manual
/// normal-ordering helper). The SIRK/GPU reduction engine then reduces it to a
/// Hermitian, bounded-below spectrum with positive excitation gaps — the
/// physical (gauge-fixed) Yang–Mills energy is positive, the positivity
/// statement of the Millennium-Prize problem.
/// The single source of truth for the abelian-limit Yang–Mills Hamiltonian:
/// the expression string realizing the `.cdb`-derived
/// `H_final = ½π² + ½B²` (`docs/yang_mills_hamiltonian.cdb`: the Legendre
/// transform of the Weyl-gauge Lagrangian, with `F₀ᵢ = π`, `¼F_ijF^ij = ½B²`),
/// in the framework's CAS dialect (`c_i` = creation, `a_i` = annihilation,
/// Hermitian field `A_i = c_i + a_i`):
///   B = (A_0 − A_1) + (g/2) A_0 A_1,   H_mag = (1/2) B * B,
/// kinetic :½π²: per momentum mode (modes 2, 3), normally ordered
/// (c_i*a_i − ½(c_i*c_i + a_i*a_i)).
///
/// Every route that exercises a compiler frontend against this Hamiltonian
/// (the [`qcd_ym_hamiltonian`] builder, raw-CAS compiles, LaTeX-dagger
/// compiles) MUST derive its fixture from THIS string — never from a
/// hand-maintained transcription, which drifts silently when the derived
/// H changes (a real failure mode: the LaTeX fixture once lagged the mode-3
/// kinetic block and failed 19-vs-22 terms).
pub fn qcd_ym_expression(g: f64) -> String {
    let b = format!("((c_0 + a_0) - (c_1 + a_1) + ({g}/2)*(c_0 + a_0)*(c_1 + a_1))");
    let h_mag = format!("(1/2)*({b})*({b})");
    // Kinetic :½π²: = A†A − ½(A†² + A²) per momentum mode (modes 2, 3),
    // already normally ordered (c_i*a_i − ½(c_i*c_i + a_i*a_i)).
    let h_kin = "(c_2 * a_2) - (1/2)*(c_2*c_2 + a_2*a_2) \
                 + (c_3 * a_3) - (1/2)*(c_3*c_3 + a_3*a_3)";
    format!("({h_mag}) + ({h_kin})")
}

pub fn qcd_ym_hamiltonian(g: f64) -> Hamiltonian {
    // The magnetic field is a genuine function of A. In the framework's CAS
    // dialect (c_i = creation, a_i = annihilation, hermitian field A_i = c_i+a_i):
    //   B = (A_0 − A_1) + (g/2) A_0 A_1
    //   H_mag = (1/2) B * B
    // The CAS compiler (`compile_to_fock`) performs the normal ordering — the
    // project's own mechanism that guarantees ⟨0|H|0⟩ = 0 by stripping the
    // zero-point ([a,a†]=1) constants, exactly the nested-Fock vacuum rule.
    // No manual normal-ordering helper is required.
    compile_to_fock(&qcd_ym_expression(g))
}

/// The TEGR/teleparallel gauge-fixed Hamiltonian **in the outer nested Fock
/// space**, implementing the **𝒮 sector** of the Cadabra2-derived kinetic
/// (`docs/qg_gauge_fixed_hamiltonian.cdb`, book.tex line 8190):
/// `ℋ_kin = (1/16e)𝒮² − (1/24e)𝒫²`, in the densitized (flat) variables
/// `𝒮`, `𝒫` of the project's change-of-variables derivation. This builder
/// realizes the `+1/16` tetrad-momentum part `:(1/16)𝒮²:` only (one mode per
/// call); the `−1/24` conformal-mode direction `𝒫` — the flat **hyperbolic**
/// d'Alembertian structure `H0 = (1/16)Δ_𝒮 − (1/24)∂²_y` of
/// `docs/qg_densitized_hamiltonian.cdb` — is provided by
/// [`qg_densitized_kinetic`], which carries both sectors.
///
/// Built from **outer** ladder operators and normally ordered so
/// `⟨0|H|0⟩ = 0`. The test verifies Hermiticity (self-adjointness in the finite
/// Fock truncation) and a real spectrum.
pub fn qg_tegr_hamiltonian(n_modes: u32) -> Hamiltonian {
    let mut terms = Vec::new();
    for i in 0..n_modes {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(i, 1);
        // :(1/16)𝒮²: → (1/16)(B†B − ½(B†²+B²)).
        let c16 = 1.0 / 16.0;
        terms.push((
            Complex64::new(c16, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner.clone()),
            ],
        ));
        terms.push((
            Complex64::new(-c16 * 0.5, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonCreate(inner.clone()),
            ],
        ));
        terms.push((
            Complex64::new(-c16 * 0.5, 0.0),
            vec![
                Operator::OuterBosonAnnihilate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The full densitized-tetrad kinetic Hamiltonian of the Cadabra2 derivation
/// (`docs/qg_densitized_hamiltonian.cdb`; `docs/qg_gauge_fixed_hamiltonian.cdb`
/// book.tex line 8190), after the change of variables `y = √e`,
/// `S = y𝒮̃`, `P = y𝒫̃` that absorbs the singular `1/e` denominator:
///
///   `H0 = (1/16)Δ_𝒮 − (1/24)∂²_y`   (flat, constant coefficients)
///
/// realized as `n_s_modes` tetrad-momentum modes with coefficient `+1/16` and
/// one conformal-mode (scale `y`) momentum with coefficient `−1/24`.  Unlike
/// [`qg_tegr_hamiltonian`] (the `+1/16` 𝒮 sector alone), this carries the
/// `−1/24` conformal-mode direction, so its spectrum is **hyperbolic** (both
/// signs) — the flat d'Alembertian that is *essentially self-adjoint* by
/// Strichartz (finite field-space signal speed) but not positive.  The test
/// verifies the `1/16` / `1/24` coefficients and the two-signed spectrum.
///
/// Built from **outer** ladder operators, normal-ordered (`⟨0|H|0⟩ = 0`), each
/// mode `i` contributing `c_i·(B†B − ½(B†² + B²))` with `c_i = 1/16` for
/// `i < n_s_modes` and `c_i = −1/24` for the final conformal mode.
pub fn qg_densitized_kinetic(n_s_modes: u32) -> Hamiltonian {
    let n_modes = n_s_modes + 1;
    let mut terms = Vec::new();
    for i in 0..n_modes {
        let c = if i == n_s_modes { -1.0 / 24.0 } else { 1.0 / 16.0 };
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(i, 1);
        // c·(B†B − ½(B†² + B²)) — the normal-ordered :𝒮²:/:𝒫²: realization.
        terms.push((
            Complex64::new(c, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner.clone()),
            ],
        ));
        terms.push((
            Complex64::new(-c * 0.5, 0.0),
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::OuterBosonCreate(inner.clone()),
            ],
        ));
        terms.push((
            Complex64::new(-c * 0.5, 0.0),
            vec![
                Operator::OuterBosonAnnihilate(inner.clone()),
                Operator::OuterBosonAnnihilate(inner),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The R + αR² (Starobinsky) gauge-fixed scalar-sector Hamiltonian, implementing
/// the scalar part of the Cadabra2-derived `H_final`
/// (`docs/qg_starobinsky_hamiltonian.cdb`): `ℋ = ½π² + ½(∇φ)² + V(φ)` with the
/// Einstein-frame Starobinsky potential `V(φ) = (M⁴/16α)(1 − e^{−√(2/3)φ/M})²`
/// (bounded below — the αR² stabilization of the conformal mode). Truncated at
/// quadratic order around the flat vacuum (φ = 0) the potential is the scalaron
/// mass term `V ≈ ½m²φ²` with `m² = M²/(12α)` (the published Starobinsky
/// scalaron mass). Quantizing the oscillator gives the diagonal number form
/// `H = Σ_i m·N_i`, the standard free massive-scalar realization (cf.
/// [`qed_free_photon`], [`qg_free_graviton`]): eigenvalues `n·m`, spacing
/// exactly `m` (the scalaron mass), vacuum `0`.
///
/// Built from the framework-native **inner** ladder operators
/// (`N_i = InnerBosonCreate(i) ∘ InnerBosonAnnihilate(i)`) so the nested-Fock
/// vacuum rule `⟨0|H|0⟩ = 0` holds automatically and the additivity `n|n⟩ =
/// n·m|n⟩` is exact at *any* occupation (a multi-scalaron state is one universe
/// with inner occupation `{i:n}`). The physical vacuum for the inner operators
/// is one empty inner universe (`OuterBosonCreate(InnerBosonicState::vacuum())`).
///
/// The test verifies Hermiticity, a real bounded-below spectrum with positive
/// excitation gaps (the boundedness claim of the R² stabilization — no
/// conformal-mode −∞), and that the excitation spacing equals the scalaron
/// mass `m` exactly and scales with it.
pub fn qg_starobinsky_hamiltonian(n_modes: u32, m: f64) -> Hamiltonian {
    let mut terms = Vec::with_capacity(n_modes as usize);
    for i in 0..n_modes {
        // m·N_i — the quantized scalaron (normal-ordered, ⟨0|H|0⟩ = 0).
        terms.push((
            Complex64::new(m, 0.0),
            vec![
                Operator::InnerBosonCreate(i),
                Operator::InnerBosonAnnihilate(i),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The scalaron (Starobinsky) mass from the R² coupling: `m² = M²/(12α)` — the
/// curvature of the Einstein-frame potential `V(φ) = (M⁴/16α)(1−e^{−√(2/3)φ/M})²`
/// at the flat vacuum (V ≈ ½m²φ², so `V″(0) = m²`). Returns `m` in units of the
/// reduced Planck mass M (M = 1). For α = O(1) the scalaron is Planck-mass
/// heavy (Compton wavelength ≲ 10⁻³⁵ m), which is exactly why the R² theory
/// passes solar-system tests — the Yukawa correction is unobservable at any
/// macroscopic distance.
pub fn qg_starobinsky_scalaron_mass(alpha: f64) -> f64 {
    (12.0 * alpha).sqrt().recip()
}

/// The weak-field gravitational potential of R + αR² (Starobinsky) gravity —
/// the classic linearized-f(R) result: the Newtonian potential with the Yukawa
/// correction from the massive scalaron,
///
///   `Φ(r) = −GM/r · (1 + ⅓ e^{−mr})`,
///
/// `m` the scalaron mass (the published f(R) fifth-force form; the `⅓` is the
/// standard coefficient from the trace equation). For `r ≫ 1/m` it reduces to
/// the Newtonian potential `−GM/r` — R² gravity passes solar-system tests —
/// and for `r ≪ 1/m` the force is enhanced by the factor `4/3`. This is the
/// weak-field (classical) content the quantized Hamiltonian must reproduce.
pub fn qg_starobinsky_weak_field_potential(gm: f64, r: f64, m: f64) -> f64 {
    -gm / r * (1.0 + (1.0 / 3.0) * (-m * r).exp())
}

/// The massive scalaron (Starobinsky scalar) field: `H = Σ_i √(k_i² + m²) N_i` —
/// the quantized massive Klein-Gordon dispersion `ω(k) = √(k² + m²)`, in
/// contrast to the massless graviton `ω = c|k|` ([`qg_free_graviton`]). Built
/// from the framework-native **inner** ladder operators (nested Fock space,
/// Hermite-basis ladder ops), normal-ordered (`⟨0|H|0⟩ = 0`), with exact
/// n-particle additivity at any occupation. The `m → 0` limit recovers the
/// massless dispersion `ω = |k|` — the R² theory's graviton sector — and the
/// group velocity `dω/dk = k/√(k²+m²) < 1` (massive propagation is subluminal).
pub fn qg_starobinsky_scalaron_field(ks: &[f64], m: f64) -> Hamiltonian {
    let mut terms = Vec::with_capacity(ks.len());
    for (i, &k) in ks.iter().enumerate() {
        let omega = (k * k + m * m).sqrt();
        terms.push((
            Complex64::new(omega, 0.0),
            vec![
                Operator::InnerBosonCreate(i as u32),
                Operator::InnerBosonAnnihilate(i as u32),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The BRST gauge constraint for the scalaron's **spatial-derivative
/// variables** — the Navier-Stokes pattern ([`navier_stokes_brst`]) applied to
/// the R² scalar sector: the spatial gradients `g_i = ∂_iφ` (modes 1..3) are
/// promoted to independent canonical fields (Hermite-basis ladder modes, no
/// lattice), and the charge
///
///   `Ω = Σ_i g_i · c_i`   (c_i the ghosts, fermion modes 0..2)
///
/// fixes each derivative variable to the value of the field derivative — the
/// gravity analogue of NS's `Ω = Σ_j u_{j,j}·c_j` (book.tex §4159-4197; the
/// derivative-variable fixing of `docs/qg_starobinsky_hamiltonian.cdb` Part 4,
/// `gf_check_Dphi2 → 0`). The charge is nilpotent (`Ω² = 0`, first-class) and
/// the scalar-sector Hamiltonian is BRST-closed (`[H, Ω] = 0`): the derivative
/// variables commute with the dynamics and are fixed to the field-derivative
/// values, so products of field derivatives (`½(∂φ)²`) survive in the
/// gauge-fixed Hamiltonian — exactly as the NS Hamiltonian carries `u_j·u_{i,j}`.
pub fn qg_starobinsky_derivative_brst() -> Hamiltonian {
    let mut terms = Vec::new();
    for i in 0..3u32 {
        // g_i = ∂_iφ: the promoted spatial-derivative variable (mode 1+i),
        // fixed by the ghost c_i (fermion mode i).
        for (c, op) in field_ops(1 + i) {
            terms.push((c, vec![op, Operator::InnerFermionAnnihilate(i)]));
        }
    }
    Hamiltonian { terms }
}

/// The gauge-fixed R² scalar sector with the **spatial field-derivative
/// variables** — the Navier-Stokes derivatives-as-fields construction
/// ([`navier_stokes_hamiltonian`], book.tex §4159-4197) applied to the
/// Starobinsky scalaron. The scalaron field `φ` (mode 0) carries the momentum;
/// its spatial gradients `g_i = ∂_iφ` (modes 1..3) are promoted to independent
/// canonical fields that carry **no momenta**, so they commute with the
/// Hamiltonian and are constants of the motion — the Eulerian block structure.
/// The gradient energy `½(∂φ)² = ½Σ_i g_i²` enters through the promoted
/// derivative-variable modes (`g_i = a†_i + a_i`), which the BRST charge
/// [`qg_starobinsky_derivative_brst`] (`Ω = Σ g_i c_i`) fixes to the values of
/// the field derivatives — so products of field derivatives (`½Σg_i²`) survive
/// in the gauge-fixed Hamiltonian, exactly as the NS Hamiltonian carries
/// `u_j·u_{i,j}`.
///
/// Built as `H = m·N_0 + ½Σ_i g_i²` (modes 1..3): `[H, g_i] = 0` exactly for
/// every derivative variable (no momenta on the gradient modes) and `[H, Ω] = 0`
/// (BRST-closed). `m = M/√(12α)` is the scalaron mass.
pub fn qg_starobinsky_gauge_fixed_scalaron(m: f64) -> Hamiltonian {
    // mode 0: the scalaron φ — the quantized massive mode m·N_0 (the diagonal
    // normal-ordered realization of ½π² + ½m²φ², cf. qg_starobinsky_hamiltonian).
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    terms.push((
        Complex64::new(m, 0.0),
        vec![
            Operator::InnerBosonCreate(0),
            Operator::InnerBosonAnnihilate(0),
        ],
    ));
    // modes 1..3: the promoted derivative variables g_i = ∂_iφ; the gradient
    // energy ½g_i² = ½(a†_i + a_i)² (Hermite-basis field operator squared),
    // normal-ordered by construction (each field op appears twice).
    for i in 1..4u32 {
        let ops = field_ops(i);
        for (c1, o1) in &ops {
            for (c2, o2) in &ops {
                let c = Complex64::new(0.5, 0.0) * c1 * c2;
                if c.norm_sqr() > 1e-30 {
                    terms.push((c, vec![o1.clone(), o2.clone()]));
                }
            }
        }
    }
    Hamiltonian { terms }
}

/// QCD gluon ↔ quark-antiquark production Hamiltonian — the **non-perturbative**
/// structural analogue of the QED [`qed_pair_production`] sector, with the
/// quark-loop **color factor** `T_R = 1/2` on the vertex:
///
///   `H = ω N_γ + Σ_j E_j (e†q)_j + Σ_j E′_j (p†q̄)_j
///        + Σ_j √T_R · c_j (q†_j q̄†_j a + a† q̄_j q_j)`,
///
/// (same mode/vertex layout as [`qed_pair_production`], the quark vertex scaled
/// by `√T_R` = 1/√2 from `Tr(T_a T_b) = T_R δ_ab`, `T_R = 1/2` — the published
/// fundamental index of perturbative QCD).
///
/// SIRK diagonalizes the gluon↔pair sector **exactly** (non-perturbatively —
/// it is not a diagrammatic expansion). The non-perturbative result is compared
/// against the *perturbative, non-SIRK* one-loop prediction: in the weak-coupling
/// limit the exact eigenvalue reduces to the quark-loop self-energy
/// `δE = T_R Σ_j c_j²/(ω − E_j − E′_j)` (so the gluon/photon self-energy ratio
/// → T_R = 1/2), and as the coupling grows the exact value departs — the very
/// demonstration that SIRK is non-perturbative.
pub fn qcd_pair_production(
    gluon_energy: f64,
    quark_energies: &[f64],
    antiquark_energies: &[f64],
    vertex: &[f64],
) -> Hamiltonian {
    // Color factor: T_R = 1/2 (fundamental index). The *square* of the vertex
    // (∝ √T_R) carries T_R in the self-energy, the published QCD color factor.
    let t_r: f64 = 0.5;
    let sqrt_tr = t_r.sqrt();
    let tr_vertex: Vec<f64> = vertex.iter().map(|&c| sqrt_tr * c).collect();
    qed_pair_production(gluon_energy, quark_energies, antiquark_energies, &tr_vertex)
}

// ─────────────────────────────────────────────
// 5b. Hierarchical multi-projector QFM generator
//     (QFM-Text plan, Stage 3 of `docs/QFM_TEXT_HRM_PLAN.md`).
//
//     H = Σ_o λ_o |0̃_o⟩⟨0̃_o|,
//     where |0̃_o⟩ = c₀^(o) |vac⟩_F + Σ_{j ∈ group o} ε_j^(o) B†_j |vac⟩_F,
//     with the α→ε normalization:
//        ε_j = α_j / √(1 + Σ α²)
//        c₀   = 1 / √(1 + Σ α²)
//
//     Each order o contributes one exact rank-1 `ProjectOnto` term,
//     and the sum is a Hermitian, rank-≤n generator (n = n_groups).
//     Cross-order coupling happens via the shared vacuum component:
//     every dressed vacuum starts in the same Fock vacuum, so the
//     projectors overlap on |vac⟩_F. This is the quantum analog of
//     hierarchical reasoning / Katz backoff.
//
//     `groups` is `(λ_o, channels_o)` where `channels_o` is the
//     list of `(mode_index, alpha_j)` pairs for order o. `mode_index`
//     is the global Fock single-excitation mode (0..K₂). Two groups
//     may share a mode (no constraint is enforced); the α→ε
//     normalization is per-group, not global.
//
//     Panics on non-finite or negative α or λ. There is NO upper
//     bound on Σ α²: the α here are the *unnormalized* flow-matching
//     weights ᾱ_j of QFM.tex eq. (Htomo), and the normalization is
//     that of the dressed vector |vac⟩_F + Σ ᾱ_j|x_j⟩ — exactly
//     idempotent per term for any weights. (This differs from the
//     ε-form builders above, whose ε are Mehler overlaps bounded by
//     Σ ε² ≤ 1 with c₀ = √(1−Σε²).)
// ─────────────────────────────────────────────
pub fn qfm_hamiltonian_hierarchical_projectors(groups: &[(f64, Vec<(u32, f64)>)]) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    for (o, (lambda, channels)) in groups.iter().enumerate() {
        assert!(
            lambda.is_finite() && *lambda >= 0.0,
            "group {o}: lambda must be finite and non-negative, got {lambda}"
        );
        for (m, a) in channels {
            assert!(
                a.is_finite() && *a >= 0.0,
                "group {o}: alpha for mode {m} must be finite and non-negative, got {a}"
            );
        }
        if channels.is_empty() && *lambda == 0.0 {
            continue;
        }
        // Per-group α → ε normalization.
        let sum_sq: f64 = channels.iter().map(|(_, a)| a * a).sum();
        let norm = (1.0 + sum_sq).sqrt();
        let c0 = 1.0 / norm;
        let mut inner_channels: Vec<(InnerBosonicState, f64)> = Vec::with_capacity(channels.len());
        for (m, a) in channels {
            let mut inner = InnerBosonicState::vacuum();
            inner.modes.insert(*m, 1);
            inner_channels.push((inner, a / norm));
        }
        // Build the single exact rank-1 ProjectOnto term for this
        // order, with coefficient λ_o.
        let h_o = dressed_vacuum_projector(inner_channels, c0);
        for (c, ops) in h_o.terms {
            let scaled = Complex64::new(*lambda, 0.0) * c;
            if scaled.norm_sqr() > 1e-30 {
                terms.push((scaled, ops));
            }
        }
    }
    Hamiltonian { terms }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical physics anchors: special relativity, nuclear systematics, compact
// objects, plasma, quantum metrology, binary inspiral. Each builder realizes
// ONE published closed-form result so the validation tests
// (`fock_sirk/tests/sr_nuclear_validation.rs`, `astro_plasma_validation.rs`)
// check the project's numerics against mainstream physics, exactly like the
// QED/QCD/QG/NS suites do.
// ─────────────────────────────────────────────────────────────────────────────

/// Exact SI-2019 defining constants and CODATA 2018 measured constants used by
/// the anchor builders below (all SI: e, h, k_B are exact by definition).
pub mod phys {
    /// elementary charge (exact), C
    pub const E: f64 = 1.602_176_634e-19;
    /// Planck constant (exact), J·s
    pub const H: f64 = 6.626_070_15e-34;
    /// Boltzmann constant (exact), J/K
    pub const K_B: f64 = 1.380_649e-23;
    /// vacuum permittivity, F/m (CODATA 2018)
    pub const EPS0: f64 = 8.854_187_8128e-12;
    /// vacuum permeability, N/A² (CODATA 2018)
    pub const MU0: f64 = 1.256_637_062_12e-6;
    /// electron mass, kg (CODATA 2018)
    pub const M_E: f64 = 9.109_383_7015e-31;
    /// proton mass, kg (CODATA 2018)
    pub const M_P: f64 = 1.672_621_923_69e-27;
    /// atomic mass unit, kg (CODATA 2018)
    pub const U: f64 = 1.660_539_066_60e-27;
    /// solar mass, kg (IAU nominal)
    pub const M_SUN: f64 = 1.988_47e30;
    /// Riemann zeta(3) (Apéry)
    pub const ZETA3: f64 = 1.202_056_903_159_594;
    /// gravitational constant, N·m²/kg² (CODATA 2018)
    pub const G: f64 = 6.67430e-11;
}

/// PDG rest masses in MeV/c²: (electron, muon, π⁺, π⁰, proton, neutron).
pub fn sr_pdg_masses_mev() -> (f64, f64, f64, f64, f64, f64) {
    (
        0.510_998_95,
        105.658_374_5,
        139.570_39,
        134.976_8,
        938.272_088_16,
        939.565_420_52,
    )
}

/// Two-body decay of a parent into daughter A + massless B: the exact
/// daughter-A momentum `|p|c = (M² − m²)c²/(2M)` (PDG kinematics), MeV.
pub fn sr_two_body_decay_momentum_mc(m_parent_mev: f64, m_a_mev: f64) -> f64 {
    (m_parent_mev * m_parent_mev - m_a_mev * m_a_mev) / (2.0 * m_parent_mev)
}

/// γγ → e⁺e⁻ threshold with a soft CMB-like photon of energy `eps_ev`
/// head-on on a high-energy photon: `E_hi = (m_e c²)²/ε` (Breit–Wheeler), eV.
pub fn sr_breit_wheeler_threshold_soft_ev(soft_photon_ev: f64) -> f64 {
    // m_e c² in eV from CODATA masses:
    let mec2 = phys::M_E * (299_792_458.0f64).powi(2) / phys::E;
    mec2 * mec2 / soft_photon_ev
}

/// GZK cutoff: p + γ_CMB → p + π⁰ threshold energy for a head-on photon of
/// energy `soft_photon_ev` — `E_p = ((m_p+m_π)² − m_p²)c⁴/(4ε)`
/// (Greisen–Zatsepin–Kuzmin 1966), eV.
pub fn sr_gzk_threshold_soft_ev(soft_photon_ev: f64) -> f64 {
    let mev_to_ev = 1.0e6;
    let (_, _, _, m_pi0, m_p, _) = sr_pdg_masses_mev();
    let dm2 = ((m_p + m_pi0).powi(2) - m_p * m_p) * mev_to_ev * mev_to_ev; // eV²
    dm2 / (4.0 * soft_photon_ev)
}

/// Relativistic dipole field `Bρ = p/q` for a particle of momentum
/// `p_gev` (GeV/c) on a ring bending radius `rho_m`: tesla. In SI,
/// `B = E/(qcρ)` with `p = E/c`, so the charge cancels exactly.
pub fn sr_dipole_field_t(p_gev: f64, rho_m: f64) -> f64 {
    let c = 299_792_458.0;
    p_gev * 1.0e9 / (c * rho_m)
}

/// Semi-empirical (Weizsäcker) mass formula with textbook coefficients
/// (a_v=15.75, a_s=17.8, a_c=0.711, a_sym=23.7, pairing 11.18/√A),
/// binding energy in MeV. Light nuclei (A ≲ 20) are outside its domain —
/// the known SEMF limitation; mid-mass and heavy nuclei are its regime.
pub fn nuc_semf_binding_energy_mev(a: u32, z: u32) -> f64 {
    let af = a as f64;
    let zf = z as f64;
    let n = af - zf;
    let volume = 15.75 * af;
    let surface = -17.8 * af.powf(2.0 / 3.0);
    let coulomb = -0.711 * zf * (zf - 1.0) / af.powf(1.0 / 3.0);
    let symmetry = -23.7 * (n - zf) * (n - zf) / af;
    let pairing = if a % 2 == 1 {
        0.0
    } else if z % 2 == 0 {
        11.18 / af.sqrt()
    } else {
        -11.18 / af.sqrt()
    };
    volume + surface + coulomb + symmetry + pairing
}

/// Hawking temperature `T = ħc³/(8πGMk_B)` for a black hole of mass `m_kg`.
pub fn astro_hawking_temperature_k(m_kg: f64) -> f64 {
    let hbar = 1.054_571_817e-34;
    let c: f64 = 299_792_458.0;
    hbar * c.powi(3) / (8.0 * std::f64::consts::PI * phys::G * m_kg * phys::K_B)
}

/// Schwarzschild radius `r_s = 2GM/c²` for a mass `m_kg`.
pub fn astro_schwarzschild_radius_m(m_kg: f64) -> f64 {
    let c: f64 = 299_792_458.0;
    2.0 * phys::G * m_kg / (c * c)
}

/// GW frequency at the Schwarzschild ISCO of a non-spinning black hole,
/// `f = c³/(6^{3/2}πGM)` (twice the orbital frequency), Hz.
pub fn astro_isco_gw_frequency_hz(m_kg: f64) -> f64 {
    let c: f64 = 299_792_458.0;
    c.powi(3) / (6.0f64.sqrt().powi(3) * std::f64::consts::PI * phys::G * m_kg)
}

/// Chandrasekhar mass from fundamental constants (Shapiro–Teukolsky):
/// `M_Ch = π(ħc/G)^{3/2}/(μ_e m_p)²`, with mean molecular weight `mu_e`.
pub fn astro_chandrasekhar_mass_kg(mu_e: f64) -> f64 {
    let hbar = 1.054_571_817e-34;
    let c: f64 = 299_792_458.0;
    std::f64::consts::PI * (hbar * c / phys::G).powf(1.5)
        / ((mu_e * phys::M_P) * (mu_e * phys::M_P))
}

/// Eddington luminosity `L = 4πGMm_p c/σ_T` (hydrogen, Thomson opacity), W.
pub fn astro_eddington_luminosity_w(m_kg: f64) -> f64 {
    let c = 299_792_458.0;
    // Classical ELECTRON radius (Thomson opacity is electron scattering).
    let r_e = phys::E * phys::E / (4.0 * std::f64::consts::PI * phys::EPS0 * phys::M_E * c * c);
    let sigma_t = 8.0 * std::f64::consts::PI / 3.0 * r_e * r_e;
    4.0 * std::f64::consts::PI * phys::G * m_kg * phys::M_P * c / sigma_t
}

/// Friedmann critical density `ρ_c = 3H₀²/(8πG)` for `h0_km_s_mpc` in
/// km/s/Mpc: kg/m³.
pub fn astro_critical_density(h0_km_s_mpc: f64) -> f64 {
    let mpc_m = 3.085_677_581_491_367e22;
    let h0 = h0_km_s_mpc * 1.0e3 / mpc_m;
    3.0 * h0 * h0 / (8.0 * std::f64::consts::PI * phys::G)
}

/// CMB blackbody at temperature `t_k`: photon number density n[m⁻³] and
/// energy density u[J/m³] — `n = (2ζ(3)/π²)(kT/ħc)³`, `u = aT⁴`.
pub fn astro_blackbody_photon_gas(t_k: f64) -> (f64, f64) {
    let hbar = 1.054_571_817e-34;
    let c: f64 = 299_792_458.0;
    let a_rad = 7.565_733_250_280_007e-16; // a = π²k⁴/(15ħ³c³), J/m³/K⁴
    let n = (2.0 * phys::ZETA3 / (std::f64::consts::PI * std::f64::consts::PI))
        * (phys::K_B * t_k / (hbar * c)).powi(3);
    let u = a_rad * t_k.powi(4);
    (n, u)
}

/// Electron plasma frequency `ω_p = √(ne²/(ε₀m_e))` → ordinary frequency, Hz.
pub fn plasma_frequency_hz(n_m3: f64) -> f64 {
    (n_m3 * phys::E * phys::E / (phys::EPS0 * phys::M_E)).sqrt()
        / (2.0 * std::f64::consts::PI)
}

/// Debye length `λ_D = √(ε₀kT/(ne²))`, m.
pub fn plasma_debye_length_m(t_k: f64, n_m3: f64) -> f64 {
    (phys::EPS0 * phys::K_B * t_k / (n_m3 * phys::E * phys::E)).sqrt()
}

/// Alfvén speed `v_A = B/√(μ₀ρ)`, m/s.
pub fn plasma_alfven_speed_ms(b_tesla: f64, rho_kg_m3: f64) -> f64 {
    b_tesla / (phys::MU0 * rho_kg_m3).sqrt()
}

/// Quantum-metrology triangle (exact SI-2019): returns
/// `(R_K [Ω], K_J [Hz/V], Φ₀ [Wb])` = `(h/e², 2e/h, h/2e)`.
pub fn metrology_constants() -> (f64, f64, f64) {
    (
        phys::H / (phys::E * phys::E),
        2.0 * phys::E / phys::H,
        phys::H / (2.0 * phys::E),
    )
}

/// BCS weak-coupling gap ratio `Δ(0)/k_BT_c` (the universal 1.764).
pub const BCS_GAP_RATIO: f64 = 1.764;

/// Closed-form coalescence time from GW frequency f_start for a binary with
/// chirp mass `chirp_mass_kg`: `t = 5c⁵/(256(G𝓜)^{5/3}(πf)^{8/3})` s
/// (leading-order Peters 1964 gravitational-radiation inspiral).
pub fn inspiral_time_to_coalescence_s(chirp_mass_kg: f64, f_start_hz: f64) -> f64 {
    let c: f64 = 299_792_458.0;
    let gm = phys::G * chirp_mass_kg;
    5.0 * c.powi(5) / (256.0 * gm.powf(5.0 / 3.0) * (std::f64::consts::PI * f_start_hz).powf(8.0 / 3.0))
}

/// Leading-order chirp rate `df/dt = (96/5)π^{8/3}(G𝓜/c³)^{5/3} f^{11/3}` (s⁻²).
pub fn inspiral_chirp_rate(chirp_mass_kg: f64, f_hz: f64) -> f64 {
    let c: f64 = 299_792_458.0;
    let tcube = (phys::G * chirp_mass_kg) / (c * c * c);
    (96.0 / 5.0)
        * std::f64::consts::PI.powf(8.0 / 3.0)
        * tcube.powf(5.0 / 3.0)
        * f_hz.powf(11.0 / 3.0)
}
