use crate::{Hamiltonian, Operator};
use num_complex::Complex64;

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
    let n_colors: usize = 8;
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
    // L_{jk,a}  = ∂_j A^a_k   → mapped to hermitian field mode (24 + (i*3+j)*n_colors + a)
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
                    let da_mode = (24 + (i * 3 + j) * n_colors + a) as u32;
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
                    let da_mode = (24 + (i * 3 + j) * n_colors + a) as u32;
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
