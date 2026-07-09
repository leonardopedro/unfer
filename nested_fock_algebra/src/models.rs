use crate::{Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState};
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
pub fn qfm_hamiltonian_hierarchical_projectors(
    groups: &[(f64, Vec<(u32, f64)>)],
) -> Hamiltonian {
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
        let mut inner_channels: Vec<(InnerBosonicState, f64)> =
            Vec::with_capacity(channels.len());
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
