//! Finite Chevalley--Eilenberg quotient for the twelve-generator gauge algebra.
//!
//! This is an actual finite quotient construction, but it is **not** the
//! field-theoretic BRST complex of the complete E1 operator. The complex is the
//! Chevalley--Eilenberg cochain complex of the gauge Lie algebra in its adjoint
//! representation, truncated at ghost degree three. `Q^2 = 0` follows from the
//! Jacobi identity; the module measures the residual and computes the cohomology
//! dimensions and harmonic representatives by SVD. The N10 bridge keeps its
//! field-level `quotient_map` null and binds this finite result separately.

use crate::sm_bridge::canonical_json_sha256;
use nalgebra::{DMatrix, linalg::SVD};
use nested_fock_algebra::sm_constraints::{GENERATORS, structure};
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::collections::BTreeMap;

const MAX_DEGREE: usize = 3;
const COHOMOLOGY_DEGREES: usize = 3;

fn bits(mask: u16) -> Vec<usize> {
    (0..GENERATORS).filter(|a| mask & (1 << a) != 0).collect()
}

fn mask_of(indices: &[usize]) -> u16 {
    indices.iter().fold(0u16, |m, &a| m | (1 << a))
}

fn position(indices: &[usize], a: usize) -> usize {
    indices.iter().position(|&b| b == a).expect("basis index")
}

fn adjoint_representation() -> Vec<DMatrix<C>> {
    (0..GENERATORS)
        .map(|a| {
            let mut rho = DMatrix::zeros(GENERATORS, GENERATORS);
            for b in 0..GENERATORS {
                for c in 0..GENERATORS {
                    let f = structure(a, b, c);
                    if f != 0.0 {
                        // ad_a(e_b) = [e_a, e_b] = sum_c f_abc e_c.
                        rho[(c, b)] = C::new(f, 0.0);
                    }
                }
            }
            rho
        })
        .collect()
}

/// One homogeneous CE differential: degree `k` to `k + 1`.
///
/// The implementation evaluates the standard cochain formula on ordered
/// `k+1`-tuples directly, so the alternating signs are explicit:
///
/// `dη(t_0..t_k) = Σ_i (-1)^i ρ(t_i) η(t_0..hat i..t_k)`
/// `  + Σ_{i<j} (-1)^{i+j} η([t_i,t_j], t_0..hat i..hat j..t_k)`.
fn ce_apply(
    mask: u16,
    rep: usize,
    degree: usize,
    rho: &[DMatrix<C>],
    out: &mut BTreeMap<(u16, usize), C>,
) {
    let indices = bits(mask);
    debug_assert_eq!(degree, indices.len(), "CE input degree must match the mask");
    let complement: Vec<usize> = (0..GENERATORS).filter(|a| mask & (1 << a) == 0).collect();

    // First term: insert one generator at position i and act with rho(t_i).
    for &a in &complement {
        let mut tuple = indices.clone();
        tuple.push(a);
        tuple.sort_unstable();
        let i = position(&tuple, a);
        let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        let target = mask | (1 << a);
        for output in 0..GENERATORS {
            let value = rho[a][(output, rep)];
            if value != C::new(0.0, 0.0) {
                *out.entry((target, output)).or_default() += value * C::new(sign, 0.0);
            }
        }
    }

    // Second term: remove a0 from S, insert the pair (b0,c0), and evaluate
    // the structure constant [X_b0, X_c0] = sum_a f_b0c0a X_a at a = a0.
    for &a0 in &indices {
        for (left, &b0) in complement.iter().enumerate() {
            for &c0 in &complement[(left + 1)..] {
                let mut tuple: Vec<usize> = indices.iter().copied().filter(|&a| a != a0).collect();
                tuple.push(b0);
                tuple.push(c0);
                tuple.sort_unstable();
                let target = mask_of(&tuple);
                let i = position(&tuple, b0);
                let j = position(&tuple, c0);
                // Sorting `(a0, S \ {a0})` costs one sign per remaining
                // generator below `a0`; the pair positions in `T` contribute
                // `(-1)^{i+j}` from the cochain formula.
                let below = indices.iter().filter(|&&s| s != a0 && s < a0).count();
                let sign = if (i + j + below).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let f = structure(b0, c0, a0);
                if f != 0.0 {
                    *out.entry((target, rep)).or_default() += C::new(sign * f, 0.0);
                }
            }
        }
    }

    // Degree zero has no second term.
}

struct Basis {
    by_degree: Vec<Vec<(u16, usize)>>,
    index: Vec<BTreeMap<(u16, usize), usize>>,
}

impl Basis {
    fn new() -> Self {
        let mut by_degree = vec![Vec::new(); MAX_DEGREE + 1];
        for mask in 0u16..(1 << GENERATORS) {
            let degree = mask.count_ones() as usize;
            if degree > MAX_DEGREE {
                continue;
            }
            for rep in 0..GENERATORS {
                by_degree[degree].push((mask, rep));
            }
        }
        let index = by_degree
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .map(|(position, entry)| (*entry, position))
                    .collect()
            })
            .collect();
        Self { by_degree, index }
    }

    fn dimension(&self, degree: usize) -> usize {
        self.by_degree[degree].len()
    }

    fn matrix(&self, degree: usize, rho: &[DMatrix<C>]) -> DMatrix<C> {
        let rows = self.dimension(degree + 1);
        let columns = self.dimension(degree);
        let mut matrix = DMatrix::zeros(rows, columns);
        for (column, &(mask, rep)) in self.by_degree[degree].iter().enumerate() {
            let mut image = BTreeMap::new();
            ce_apply(mask, rep, degree, rho, &mut image);
            for ((out_mask, out_rep), value) in image {
                let row = self.index[degree + 1][&(out_mask, out_rep)];
                matrix[(row, column)] += value;
            }
        }
        matrix
    }

    fn q_squared_residual(&self, rho: &[DMatrix<C>]) -> f64 {
        let mut worst = 0.0_f64;
        for degree in 0..COHOMOLOGY_DEGREES {
            for &(mask, rep) in &self.by_degree[degree] {
                let mut first = BTreeMap::new();
                ce_apply(mask, rep, degree, rho, &mut first);
                let mut second: BTreeMap<(u16, usize), C> = BTreeMap::new();
                for ((next_mask, next_rep), coefficient) in first {
                    let mut image = BTreeMap::new();
                    ce_apply(next_mask, next_rep, degree + 1, rho, &mut image);
                    for (key, value) in image {
                        *second.entry(key).or_default() += coefficient * value;
                    }
                }
                let norm_square: f64 = second.values().map(|z| z.norm_sqr()).sum();
                worst = worst.max(norm_square.sqrt());
            }
        }
        worst
    }
}

fn spectral_rank(matrix: &DMatrix<C>, tolerance: f64) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    let svd = SVD::new(matrix.clone(), false, false);
    svd.singular_values
        .iter()
        .filter(|value| **value > tolerance)
        .count()
}

/// Deterministic hash of an orthonormal basis for the harmonic representatives:
/// the nullspace of `Q_k` stacked with the adjoint of `Q_{k-1}`. The entries are
/// rounded before hashing, so the record binds the quotient map without
/// depending on an arbitrary SVD phase at the last bit.
fn harmonic_basis(
    degree: usize,
    q_prev: Option<&DMatrix<C>>,
    q: &DMatrix<C>,
    dimension: usize,
    tolerance: f64,
) -> (usize, String) {
    let mut blocks: Vec<DMatrix<C>> = Vec::new();
    if q.nrows() > 0 && q.ncols() > 0 {
        blocks.push(q.clone());
    }
    if let Some(prev) = q_prev
        && prev.nrows() > 0
        && prev.ncols() > 0
    {
        blocks.push(prev.adjoint());
    }
    if blocks.is_empty() {
        return (
            dimension,
            canonical_json_sha256(&json!({
                "degree": degree,
                "dimension": dimension,
                "vectors": [],
            })),
        );
    }
    let rows: usize = blocks.iter().map(DMatrix::nrows).sum();
    let columns = blocks[0].ncols();
    let mut stacked = DMatrix::zeros(rows, columns);
    let mut offset = 0;
    for block in &blocks {
        for row in 0..block.nrows() {
            for column in 0..block.ncols() {
                stacked[(offset + row, column)] = block[(row, column)];
            }
        }
        offset += block.nrows();
    }
    let svd = SVD::new(stacked, true, true);
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| **value > tolerance)
        .count();
    let nullity = columns.saturating_sub(rank);
    let mut vectors = Vec::with_capacity(nullity);
    if let Some(v_t) = svd.v_t {
        for row in rank..columns {
            let vector: Vec<[f64; 2]> = (0..columns)
                .map(|column| {
                    let value = v_t[(row, column)];
                    [round12(value.re), round12(value.im)]
                })
                .collect();
            vectors.push(vector);
        }
    }
    (
        nullity,
        canonical_json_sha256(&json!({
            "degree": degree,
            "dimension": dimension,
            "vectors": vectors,
        })),
    )
}

fn round12(value: f64) -> f64 {
    let scale = 1e12;
    (value * scale).round() / scale
}

fn representation_closure(rho: &[DMatrix<C>]) -> f64 {
    let mut worst = 0.0_f64;
    for a in 0..GENERATORS {
        for b in 0..GENERATORS {
            let commutator = &rho[a] * &rho[b] - &rho[b] * &rho[a];
            let mut expected = DMatrix::zeros(GENERATORS, GENERATORS);
            for (c, rho_c) in rho.iter().enumerate().take(GENERATORS) {
                let f = structure(a, b, c);
                if f != 0.0 {
                    expected += rho_c.map(|value| value * C::new(f, 0.0));
                }
            }
            let residual = &commutator - &expected;
            worst = worst.max(
                residual
                    .iter()
                    .map(|value| value.norm_sqr())
                    .sum::<f64>()
                    .sqrt(),
            );
        }
    }
    worst
}

/// Build the finite adjoint CE quotient record. `status` is `measured`: the
/// quotient is exact algebraically but the numerical rank and representative
/// basis are SVD measurements, not interval enclosures.
pub fn finite_brst_quotient() -> Value {
    let rho = adjoint_representation();
    let basis = Basis::new();
    let q0 = basis.matrix(0, &rho);
    let q1 = basis.matrix(1, &rho);
    let q2 = basis.matrix(2, &rho);
    let tolerance = 1e-9;
    let r0 = spectral_rank(&q0, tolerance);
    let r1 = spectral_rank(&q1, tolerance);
    let r2 = spectral_rank(&q2, tolerance);
    let dims: Vec<usize> = (0..=COHOMOLOGY_DEGREES)
        .map(|degree| basis.dimension(degree))
        .collect();
    let cohomology = vec![dims[0] - r0, dims[1] - r1 - r0, dims[2] - r2 - r1];
    let (h0_count, h0_hash) = harmonic_basis(0, None, &q0, dims[0], tolerance);
    let (h1_count, h1_hash) = harmonic_basis(1, Some(&q0), &q1, dims[1], tolerance);
    let (h2_count, h2_hash) = harmonic_basis(2, Some(&q1), &q2, dims[2], tolerance);
    let closure = representation_closure(&rho);
    let q_squared = basis.q_squared_residual(&rho);
    let representative_hash = canonical_json_sha256(&json!({
        "degree_0": h0_hash,
        "degree_1": h1_hash,
        "degree_2": h2_hash,
    }));
    let operator_identity = canonical_json_sha256(&json!({
        "structure_constants": (0..GENERATORS)
            .map(|a| (0..GENERATORS)
                .map(|b| (0..GENERATORS).map(|c| structure(a, b, c)).collect::<Vec<_>>())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "representation": "adjoint",
        "max_degree": MAX_DEGREE,
    }));

    json!({
        "schema": "sm-finite-brst-quotient/v1",
        "claim_id": "n10_finite_adjoint_ce_quotient",
        "status": "measured",
        "construction": "Chevalley-Eilenberg cochain complex of the twelve-generator gauge algebra in its adjoint representation",
        "operator": {
            "name": "finite adjoint CE differential Q",
            "identity_sha256": operator_identity,
            "generators": GENERATORS,
            "representation_dimension": GENERATORS,
            "max_cochain_degree": MAX_DEGREE,
        },
        "artifact": {
            "cochain_dimensions": dims,
            "differential_ranks": [r0, r1, r2],
            "cohomology_dimensions": cohomology,
            "q_squared_residual": q_squared,
            "representation_closure_residual": closure,
            "tolerance": tolerance,
            "harmonic_representatives": {
                "degree_0": h0_count,
                "degree_1": h1_count,
                "degree_2": h2_count,
                "basis_sha256": representative_hash,
            },
            "quotient_map": {
                "kind": "harmonic representatives (ker Q ∩ ker Q*)",
                "degrees": [0, 1, 2],
                "dimensions": cohomology,
                "representative_basis_sha256": representative_hash,
            },
            "finite_quotient": true,
            "field_level_quotient": false,
        },
        "theorem": {
            "lean_name": "Analytic.SmConstraints.finite_adjoint_ce",
            "statement": "Q^2 = 0 on the finite adjoint CE complex and H^k = ker Q_k / im Q_{k-1}",
            "proof_status": "not-run",
            "remaining": "Q^2 is measured on the finite complex; the field-theoretic BRST complex, its inner product completion and the physical quotient remain open.",
        },
        "hypotheses": [
            "The complex uses the twelve homogeneous gauge generators and the bare structure constants.",
            "The coefficient module is the adjoint representation, not the complete matter/Higgs Fock space.",
            "Cochains are truncated at ghost degree three; degrees 0..2 are reported.",
            "Ranks and harmonic bases are SVD measurements with tolerance 1e-9, not interval enclosures.",
        ],
        "not_claimed": [
            "No field-theoretic BRST cohomology or gauge-invariant Hilbert space.",
            "No representative map for the complete E1 operator; the bridge keeps that quotient_map null.",
            "No anomaly, infinite-mode, cutoff or continuum statement.",
        ],
    })
}
