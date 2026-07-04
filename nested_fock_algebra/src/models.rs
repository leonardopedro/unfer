use crate::{Hamiltonian, InnerBosonicState, Operator};
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
// 4b. Quantum Flow Matching (QFM) generator (builtin — see QFM.tex)
//     H = |0><0|  +  Σ_j α_j n_j        (n_j = a†_j a_j)
//
//     Analytical, neural-network-free generative flow: M orthogonal data
//     points become single bosons in distinct modes, and the Mehler uniform
//     prior is the rank-1 vacuum projector |0><0|. The data potential has no
//     cross-terms, so it is strictly diagonal (H|0>=|0>, H|x_j>=α_j|x_j>) and
//     building it stays O(M) by bypassing Expression::expand().
// ─────────────────────────────────────────────

/// The analytical **Quantum Flow Matching** generator (see `QFM.tex`).
///
/// Encodes `M = alphas.len()` orthogonal data points as single bosons in
/// distinct modes plus the Mehler vacuum-projector prior:
/// `H = |0><0| + Σ_j α_j · a†_j a_j`. Constructed directly so M can be huge
/// without hitting the CAS term-explosion limit.
pub fn qfm_hamiltonian(alphas: &[f64]) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    // H_0 = |0><0|, the Mehler global prior.
    terms.push((Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum]));
    // Decoupled data potential: one number operator per data point.
    for (j, &alpha) in alphas.iter().enumerate() {
        let mode = j as u32;
        terms.push((
            Complex64::new(alpha, 0.0),
            vec![
                Operator::InnerBosonCreate(mode),
                Operator::InnerBosonAnnihilate(mode),
            ],
        ));
    }
    Hamiltonian { terms }
}

/// The **off-diagonal** Quantum Flow Matching generator (P5 #26).
///
/// Where [`qfm_hamiltonian`] uses the diagonal number-operator surrogate
/// `H = |0><0| + Σ α_j n_j` (eigenstates → phase-only evolution), this realizes
/// the Fock-space form of the continuity operator `ĥ_j` of `QFM.tex` §2.3 that
/// *transports amplitude between the vacuum and the data channels*:
///
/// `H = |0><0| + Σ_j α_j (B†_j P₀ + P₀ B_j)`
///
/// where `B†_j = OuterBosonCreate(|x_j⟩)` and `B_j = OuterBosonAnnihilate(|x_j⟩)`
/// are the outer-universe creation/annihilation operators for the single-boson
/// state `|x_j⟩` (one universe holding one boson in inner mode `j`). In the
/// `{|0⟩, |x_j⟩}` subspace, `B†_j P₀ + P₀ B_j` is the Pauli-X that swaps
/// vacuum ↔ data channel `j` (apply is right-to-left: `B†_j` after `P₀` creates
/// the `|x_j⟩` universe from the vacuum; `P₀` after `B_j` projects the
/// vacuum left by annihilating the `|x_j⟩` universe). The generator is
/// **Hermitian** — `B†_j P₀` and `P₀ B_j` are conjugates (`B†† = B`,
/// `P₀† = P₀`, reverse product) — so `e^{-iHt}` is unitary and the Born-rule
/// substrate (norm conservation, `nalgebra` Padé `exp()`) applies unchanged.
///
/// **Honest deviation from the paper:** `QFM.tex` eq. (Hbar) gives the
/// *anti-Hermitian* continuity generator `H̄ = |0><0| - (i/2)Σ α_j ĥ_j` whose
/// evolution is the real Fokker–Planck transport semigroup (irreversible
/// diffusion of amplitude into the data channels). `unfer`'s SIRK solver and
/// Born-rule layer assume a Hermitian Hamiltonian and unitary evolution
/// (AGENTS.md §4: "Always use nalgebra's Padé approximant exp() … to preserve
/// unitarity and Hermiticity"). This builtin therefore implements the
/// **Hermitian** off-diagonal coupling: the result is *coherent Rabi-like
/// oscillation* of amplitude between vacuum and data channels (populations
/// genuinely flow, then flow back), not the paper's irreversible transport. It
/// is the unfer-faithful realization of the vacuum↔data mixing that the
/// diagonal surrogate lacks; setting `α_j → 0` recovers the bare projector.
/// Constructed directly so M can be huge without CAS term-explosion.
pub fn qfm_hamiltonian_offdiag(alphas: &[f64]) -> Hamiltonian {
    let mut terms: Vec<(Complex64, Vec<Operator>)> = Vec::new();
    // H_0 = |0><0|, the Mehler global prior (same as the diagonal builtin).
    terms.push((Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum]));
    // Off-diagonal coupling per data point: α_j (B†_j P₀ + P₀ B_j), where
    // |x_j⟩ is the single-boson inner state {mode j: 1}. Two conjugate terms
    // per j → hermitian; vacuum ↔ |x_j⟩ mixing (Pauli-X in the 2D subspace).
    for (j, &alpha) in alphas.iter().enumerate() {
        let mode = j as u32;
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, 1);
        let c = Complex64::new(alpha, 0.0);
        // B†_j P₀  —  maps |0⟩ → |x_j⟩ (create the |x_j⟩ universe from vacuum)
        terms.push((
            c,
            vec![
                Operator::OuterBosonCreate(inner.clone()),
                Operator::ProjectVacuum,
            ],
        ));
        // P₀ B_j  —  maps |x_j⟩ → |0⟩ (annihilate the |x_j⟩ universe → vacuum)
        terms.push((
            c,
            vec![
                Operator::ProjectVacuum,
                Operator::OuterBosonAnnihilate(inner),
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
