//! Finite field-level BRST window.
//!
//! The joint charge removes one ghost and preserves the remaining occupation, so
//! the total-occupation filtration `N <= M` is `Q`-invariant. This module builds
//! the smallest nontrivial window (`M = 2`) for the bosonic gauge-Higgs sector,
//! all 96 complex matter fermion modes, and the twelve ghosts. Derivative jets
//! are still excluded. The record reports whether `Q` stayed inside the window,
//! whether it lowered total occupation by exactly one, and the cohomology and
//! harmonic representatives of the finite complex.
//!
//! The finite Hodge representatives are a quotient map for *this* filtered
//! window. They are not a continuum physical quotient and are not promoted to a
//! physical-sector identification.

use crate::sm_bridge::{canonical_json_sha256, reported_residual};
use nalgebra::{DMatrix, linalg::SVD};
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_constraints::*;
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::collections::BTreeMap;

const WINDOW: u32 = 2;
const TOLERANCE: f64 = 1e-9;

fn parameters() -> SmParameters {
    let matrix = |scale: f64| {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                C::new(
                    scale * (1 + i + 2 * j) as f64,
                    scale * 0.17 * (1 + 2 * i + j) as f64,
                )
            })
        })
    };
    SmParameters {
        gs: 0.3,
        gw: 0.2,
        gy: 0.15,
        lambda: 0.1,
        vev: 0.7,
        yu: matrix(0.02),
        yd: matrix(0.013),
        ynu: matrix(0.007),
        ye: matrix(0.019),
    }
}

fn limits() -> ActionLimits {
    ActionLimits {
        max_components: 4_000_000,
        max_work: 40_000_000_000,
    }
}

fn state(basis: &B) -> NestedVector {
    NestedVector::from([(O::graded([(basis.clone(), 1)]).unwrap(), C::new(1.0, 0.0))])
}

fn image(q: &InnerExpr, basis: &B) -> Result<Vec<(B, C)>, String> {
    let out = q
        .d_gamma(&state(basis), limits())
        .map_err(|error| format!("action failed: {error:?}"))?;
    let mut image = Vec::new();
    for (outer, coefficient) in out {
        let occupations = outer.inner_occupations();
        if occupations.len() != 1 || occupations[0].1 != 1 {
            return Err("image is not a single inner universe".to_string());
        }
        image.push((occupations[0].0.clone(), coefficient));
    }
    Ok(image)
}

fn boson_configurations(modes: &[u32], max: u32) -> Vec<Vec<(u32, u32)>> {
    let mut out = vec![Vec::new()];
    if max == 0 {
        return out;
    }
    for &mode in modes {
        out.push(vec![(mode, 1)]);
    }
    if max >= 2 {
        for &mode in modes {
            out.push(vec![(mode, 2)]);
        }
        for (i, &left) in modes.iter().enumerate() {
            for &right in &modes[(i + 1)..] {
                out.push(vec![(left, 1), (right, 1)]);
            }
        }
    }
    out
}

fn fermion_subsets(modes: &[u32], max: u32) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new()];
    if max == 0 {
        return out;
    }
    for &mode in modes {
        out.push(vec![mode]);
    }
    if max >= 2 {
        for (i, &left) in modes.iter().enumerate() {
            for &right in &modes[(i + 1)..] {
                out.push(vec![left, right]);
            }
        }
    }
    out
}

fn occupation_total(bosons: &[(u32, u32)], ghosts: &[u32]) -> u32 {
    bosons.iter().map(|(_, n)| *n).sum::<u32>() + ghosts.len() as u32
}

fn is_ghost(mode: u32) -> bool {
    mode >= GHOST_BASE && mode < GHOST_BASE + GENERATORS as u32
}

fn has_ghost(basis: &B) -> bool {
    basis.fermions().iter().any(|&mode| is_ghost(mode))
}

/// The 120 independent derivative-jet coordinates: `dG` (72), `dW` (27),
/// `dB` (9), and `dphi` (12). They are distinct from the 40 homogeneous
/// coordinates returned by [`coord::active`].
fn jet_modes() -> Vec<u32> {
    let mut modes = Vec::with_capacity(120);
    for a in 0..8 {
        for i in 0..3 {
            for j in 0..3 {
                modes.push(coord::dg(a, i, j));
            }
        }
    }
    for a in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                modes.push(coord::dw(a, i, j));
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            modes.push(coord::db(i, j));
        }
    }
    for a in 0..4 {
        for i in 0..3 {
            modes.push(coord::dphi(a, i));
        }
    }
    modes
}

fn is_jet(mode: u32) -> bool {
    (27..99).contains(&mode)
        || (108..135).contains(&mode)
        || (138..147).contains(&mode)
        || (151..163).contains(&mode)
}

fn jet_number(basis: &B) -> u32 {
    basis
        .bosons()
        .iter()
        .filter(|(mode, _)| is_jet(**mode))
        .map(|(_, occupation)| *occupation)
        .sum()
}

fn sector_basis(include_jets: bool) -> (Vec<B>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut modes = coord::active();
    if include_jets {
        modes.extend(jet_modes());
    }
    let ghosts: Vec<u32> = (0..GENERATORS as u32).map(|a| GHOST_BASE + a).collect();
    let matter: Vec<u32> = (0..MATTER_MODES).collect();
    let matter_by_size: Vec<Vec<Vec<u32>>> = (0..=WINDOW)
        .map(|size| {
            fermion_subsets(&matter, WINDOW)
                .into_iter()
                .filter(|set| set.len() as u32 == size)
                .collect()
        })
        .collect();
    let mut states = Vec::new();
    let mut totals = Vec::new();
    let mut matter_numbers = Vec::new();
    let mut jet_numbers = Vec::new();
    for bosons in boson_configurations(&modes, WINDOW) {
        let jet_count: u32 = bosons
            .iter()
            .filter(|(mode, _)| is_jet(*mode))
            .map(|(_, occupation)| *occupation)
            .sum();
        for ghost in fermion_subsets(&ghosts, WINDOW) {
            let bgh_total = occupation_total(&bosons, &ghost);
            if bgh_total > WINDOW {
                continue;
            }
            for size in 0..=(WINDOW - bgh_total) {
                for matter_set in &matter_by_size[size as usize] {
                    let fermions = ghost.iter().chain(matter_set.iter()).copied();
                    let basis = B::new(bosons.clone(), fermions).unwrap();
                    totals.push(bgh_total + matter_set.len() as u32);
                    matter_numbers.push(matter_set.len() as u32);
                    jet_numbers.push(jet_count);
                    states.push(basis);
                }
            }
        }
    }
    (states, totals, matter_numbers, jet_numbers)
}

fn rank(matrix: &DMatrix<C>) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    SVD::new(matrix.clone(), false, false)
        .singular_values
        .iter()
        .filter(|value| **value > TOLERANCE)
        .count()
}

fn submatrix(matrix: &DMatrix<C>, rows: &[usize], columns: &[usize]) -> DMatrix<C> {
    let mut out = DMatrix::zeros(rows.len(), columns.len());
    for (i, &row) in rows.iter().enumerate() {
        for (j, &column) in columns.iter().enumerate() {
            out[(i, j)] = matrix[(row, column)];
        }
    }
    out
}

fn round12(value: f64) -> f64 {
    let scale = 1e12;
    (value * scale).round() / scale
}

fn entries_json(entries: &[(usize, C)]) -> Vec<Value> {
    entries
        .iter()
        .map(|(index, value)| json!([index, round12(value.re), round12(value.im)]))
        .collect()
}

/// A deterministic, sparse kernel basis from the reduced row-echelon form of
/// `matrix`. For each free column `f` the basis vector is
/// `e_f - sum_i R[i,f] e_{pivot_i}`. The RREF is unique, so the basis depends on
/// the matrix alone — unlike singular vectors, which are defined only up to
/// rotations inside a degenerate kernel. A right kernel of `Q_degree` is closed;
/// at the top degree (`Q_{n+1}=0`) it is the harmonic representative space, and
/// on the stacked `[Q_k ; Q_{k-1}^†]` it is `ker Q_k ∩ ker Q_{k-1}^†`.
///
/// Returns the pivot columns and `(free_column, entries)` for each basis vector.
pub(crate) fn rref_kernel(
    matrix: &DMatrix<C>,
    tolerance: f64,
) -> (Vec<usize>, Vec<(usize, Vec<(usize, C)>)>) {
    let mut reduced = matrix.clone();
    let rows = reduced.nrows();
    let columns = reduced.ncols();
    let mut pivots = Vec::new();
    let mut pivot_row = 0;
    for column in 0..columns {
        if pivot_row >= rows {
            break;
        }
        let mut best = pivot_row;
        let mut best_norm = 0.0_f64;
        for row in pivot_row..rows {
            let norm = reduced[(row, column)].norm();
            if norm > best_norm {
                best_norm = norm;
                best = row;
            }
        }
        if best_norm <= tolerance {
            continue;
        }
        if best != pivot_row {
            reduced.swap_rows(best, pivot_row);
        }
        let pivot = reduced[(pivot_row, column)];
        for c in column..columns {
            reduced[(pivot_row, c)] /= pivot;
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = reduced[(row, column)];
            if factor.norm() <= tolerance {
                reduced[(row, column)] = C::new(0.0, 0.0);
                continue;
            }
            for c in column..columns {
                let pivot_value = reduced[(pivot_row, c)];
                reduced[(row, c)] -= factor * pivot_value;
            }
            reduced[(row, column)] = C::new(0.0, 0.0);
        }
        pivots.push(column);
        pivot_row += 1;
    }

    let mut kernel = Vec::with_capacity(columns - pivots.len());
    for free in 0..columns {
        if pivots.binary_search(&free).is_ok() {
            continue;
        }
        let mut entries = vec![(free, C::new(1.0, 0.0))];
        for (i, &pivot) in pivots.iter().enumerate() {
            let value = -reduced[(i, free)];
            if value.norm() > 1e-12 {
                entries.push((pivot, value));
            }
        }
        entries.sort_by_key(|(index, _)| *index);
        kernel.push((free, entries));
    }
    (pivots, kernel)
}

/// Canonical hash of the right kernel of `Q_degree`.
fn rref_nullspace(matrix: &DMatrix<C>, degree: u32) -> (usize, String, Vec<usize>) {
    let (pivots, kernel) = rref_kernel(matrix, TOLERANCE);
    let dimension = kernel.len();
    let vector_hashes: Vec<String> = kernel
        .iter()
        .map(|(free, entries)| {
            canonical_json_sha256(&json!({
                "free": free,
                "entries": entries_json(entries),
            }))
        })
        .collect();
    let basis_hash = canonical_json_sha256(&json!({
        "degree": degree,
        "dimension": dimension,
        "vectors": vector_hashes,
    }));
    (dimension, basis_hash, pivots)
}

/// A deterministic hash of the left nullspace of `Q_2` (the harmonic `H^1`
/// representatives). `Q_1` vanishes on this window, so `ker Q_2^*` is the
/// harmonic space.
///
/// The basis is read off the augmented reduced row-echelon form `[Q_2 | I]`,
/// not from the trailing left singular vectors: inside a degenerate nullspace
/// the singular vectors are defined only up to a rotation, so hashing them pins
/// an arbitrary basis that does not reproduce across toolchains or platforms.
/// The zero rows of the reduced left block carry the row combinations that
/// annihilate the row space, and the augmented RREF is unique, so the basis (and
/// hence the hash) is a property of the matrix alone. Tracking the transform
/// keeps the work on the small `rows x rows` factor instead of eliminating over
/// the adjoint.
fn left_nullspace(matrix: &DMatrix<C>) -> (usize, String) {
    let mut reduced = matrix.clone();
    let rows = reduced.nrows();
    let columns = reduced.ncols();
    let mut transform = DMatrix::<C>::identity(rows, rows);
    let mut pivot_row = 0;
    for column in 0..columns {
        if pivot_row >= rows {
            break;
        }
        let mut best = pivot_row;
        let mut best_norm = 0.0_f64;
        for row in pivot_row..rows {
            let norm = reduced[(row, column)].norm();
            if norm > best_norm {
                best_norm = norm;
                best = row;
            }
        }
        if best_norm <= TOLERANCE {
            continue;
        }
        if best != pivot_row {
            reduced.swap_rows(best, pivot_row);
            transform.swap_rows(best, pivot_row);
        }
        let pivot = reduced[(pivot_row, column)];
        for c in column..columns {
            reduced[(pivot_row, c)] /= pivot;
        }
        for c in 0..rows {
            transform[(pivot_row, c)] /= pivot;
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = reduced[(row, column)];
            if factor.norm() <= TOLERANCE {
                reduced[(row, column)] = C::new(0.0, 0.0);
                continue;
            }
            for c in column..columns {
                let pivot_value = reduced[(pivot_row, c)];
                reduced[(row, c)] -= factor * pivot_value;
            }
            for c in 0..rows {
                let pivot_value = transform[(pivot_row, c)];
                transform[(row, c)] -= factor * pivot_value;
            }
            reduced[(row, column)] = C::new(0.0, 0.0);
        }
        pivot_row += 1;
    }

    let dimension = rows - pivot_row;
    let mut vectors = Vec::with_capacity(dimension);
    for row in pivot_row..rows {
        let scale = (0..rows)
            .map(|c| transform[(row, c)].norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let vector: Vec<[f64; 2]> = (0..rows)
            .map(|c| {
                let value = transform[(row, c)] / scale;
                [round12(value.re), round12(value.im)]
            })
            .collect();
        vectors.push(vector);
    }
    (
        dimension,
        canonical_json_sha256(&json!({
            "degree": 1,
            "dimension": dimension,
            "vectors": vectors,
        })),
    )
}

#[derive(Default)]
struct WindowMeasurement {
    state_count: usize,
    dims: Vec<usize>,
    matter_dims: Vec<usize>,
    jet_dims: Vec<usize>,
    block_ranks: Vec<usize>,
    cohomology_dimensions: Vec<usize>,
    cohomology_dimension: usize,
    q_squared: f64,
    matter_block_ranks: Vec<usize>,
    jet_block_ranks: Vec<usize>,
    gauge_singlet_kernel_dimension: usize,
    cohomology_by_matter: Vec<Vec<usize>>,
    cohomology_by_jet: Vec<Vec<usize>>,
    basis_hash: String,
    h0_hash: String,
    h1_dimension: usize,
    h1_hash: String,
    h2_dimension: usize,
    h2_hash: String,
    h2_pivots: Vec<usize>,
    representative_hash: String,
    closed: bool,
    graded: bool,
    matter_graded: bool,
    jet_graded: bool,
    failure: String,
}

fn empty_basis_hash(degree: u32, dimension: usize) -> String {
    canonical_json_sha256(&json!({
        "degree": degree,
        "dimension": dimension,
        "vectors": [],
    }))
}

fn measure_window(include_jets: bool) -> WindowMeasurement {
    let q = brst_charge(&parameters());
    let (states, totals, matter_numbers, jet_numbers) = sector_basis(include_jets);
    let index: BTreeMap<B, usize> = states
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, basis)| (basis, position))
        .collect();

    let dims: Vec<usize> = (0..=WINDOW)
        .map(|n| totals.iter().filter(|&&total| total == n).count())
        .collect();
    let rows1: Vec<usize> = (0..states.len()).filter(|&i| totals[i] == 1).collect();
    let columns1: Vec<usize> = rows1.clone();
    let columns2: Vec<usize> = (0..states.len()).filter(|&i| totals[i] == 2).collect();
    let row1_of: BTreeMap<usize, usize> = rows1
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect();
    let column2_of: BTreeMap<usize, usize> = columns2
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect();

    let mut q1 = DMatrix::zeros(1, columns1.len());
    let mut q2 = DMatrix::zeros(rows1.len(), columns2.len());
    let mut closed = true;
    let mut graded = true;
    let mut matter_graded = true;
    let mut jet_graded = true;
    let mut failure = String::new();

    for (local, &column) in columns1.iter().enumerate() {
        for (target, coefficient) in image(&q, &states[column]).unwrap() {
            let Some(&global) = index.get(&target) else {
                closed = false;
                failure = "Q left the window on an N=1 column".to_string();
                continue;
            };
            if totals[global] + 1 != totals[column] {
                graded = false;
                failure = "Q did not lower N by one on an N=1 column".to_string();
                continue;
            }
            if jet_number(&target) != jet_numbers[column] {
                jet_graded = false;
                failure = "Q changed derivative-jet number on an N=1 column".to_string();
                continue;
            }
            if matter_numbers[global] != matter_numbers[column] {
                matter_graded = false;
                failure = "Q changed matter number on an N=1 column".to_string();
                continue;
            }
            if totals[global] == 0 {
                q1[(0, local)] += coefficient;
            }
        }
    }

    for &column in &columns2 {
        if !has_ghost(&states[column]) {
            // With no ghost, both the linear ghost contraction and the cubic
            // ghost term vanish identically.
            continue;
        }
        for (target, coefficient) in image(&q, &states[column]).unwrap() {
            let Some(&global) = index.get(&target) else {
                closed = false;
                failure = "Q left the window on an N=2 column".to_string();
                continue;
            };
            if totals[global] + 1 != totals[column] {
                graded = false;
                failure = "Q did not lower N by one on an N=2 column".to_string();
                continue;
            }
            if jet_number(&target) != jet_numbers[column] {
                jet_graded = false;
                failure = "Q changed derivative-jet number on an N=2 column".to_string();
                continue;
            }
            if matter_numbers[global] != matter_numbers[column] {
                matter_graded = false;
                failure = "Q changed matter number on an N=2 column".to_string();
                continue;
            }
            let row = row1_of[&global];
            let local = column2_of[&column];
            q2[(row, local)] += coefficient;
        }
    }

    if !closed || !graded || !matter_graded || !jet_graded {
        return WindowMeasurement {
            state_count: states.len(),
            closed,
            graded,
            matter_graded,
            jet_graded,
            failure,
            ..WindowMeasurement::default()
        };
    }

    let mut block_ranks = vec![vec![0usize; WINDOW as usize + 1]; WINDOW as usize + 1];
    for jet in 0..=WINDOW {
        for matter in 0..=WINDOW {
            let rows: Vec<usize> = (0..rows1.len())
                .filter(|&row| {
                    jet_numbers[rows1[row]] == jet && matter_numbers[rows1[row]] == matter
                })
                .collect();
            let columns: Vec<usize> = (0..columns2.len())
                .filter(|&column| {
                    jet_numbers[columns2[column]] == jet
                        && matter_numbers[columns2[column]] == matter
                })
                .collect();
            block_ranks[jet as usize][matter as usize] = rank(&submatrix(&q2, &rows, &columns));
        }
    }

    let q1_rank = rank(&q1);
    let q2_rank = if include_jets {
        block_ranks.iter().flatten().sum::<usize>()
    } else {
        rank(&q2)
    };
    let cohomology_dimensions: Vec<usize> = (0..=WINDOW)
        .map(|degree| {
            let rank_here = if degree == 0 {
                0
            } else {
                [q1_rank, q2_rank][(degree - 1) as usize]
            };
            let rank_above = [q1_rank, q2_rank]
                .get(degree as usize)
                .copied()
                .unwrap_or(0);
            dims[degree as usize] - rank_here - rank_above
        })
        .collect();
    let cohomology_dimension: usize = cohomology_dimensions.iter().sum();
    let squared = &q1 * &q2;
    let q_squared = squared
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let matter_dims: Vec<usize> = (0..=WINDOW)
        .map(|matter| {
            matter_numbers
                .iter()
                .filter(|&&number| number == matter)
                .count()
        })
        .collect();
    let jet_dims: Vec<usize> = (0..=WINDOW)
        .map(|jet| jet_numbers.iter().filter(|&&number| number == jet).count())
        .collect();

    let matter_block_ranks: Vec<usize> = (0..=WINDOW)
        .map(|matter| {
            (0..=WINDOW)
                .map(|jet| block_ranks[jet as usize][matter as usize])
                .sum()
        })
        .collect();
    let jet_block_ranks: Vec<usize> = (0..=WINDOW)
        .map(|jet| {
            (0..=WINDOW)
                .map(|matter| block_ranks[jet as usize][matter as usize])
                .sum()
        })
        .collect();
    let matter_graded = matter_graded && matter_block_ranks.iter().sum::<usize>() == q2_rank;
    let jet_graded = jet_block_ranks.iter().sum::<usize>() == q2_rank;
    let gauge_singlet_kernel_dimension = MATTER_MODES as usize - block_ranks[0][1];

    let cohomology_by_matter: Vec<Vec<usize>> = (0..=WINDOW)
        .map(|matter| {
            let c0 = totals
                .iter()
                .zip(&matter_numbers)
                .filter(|(total, number)| **total == 0 && **number == matter)
                .count();
            let c1 = totals
                .iter()
                .zip(&matter_numbers)
                .filter(|(total, number)| **total == 1 && **number == matter)
                .count();
            let c2 = totals
                .iter()
                .zip(&matter_numbers)
                .filter(|(total, number)| **total == 2 && **number == matter)
                .count();
            let rank_here = if matter == 0 { q1_rank } else { 0 };
            let rank_above = matter_block_ranks[matter as usize];
            vec![c0 - rank_here, c1 - rank_here - rank_above, c2 - rank_above]
        })
        .collect();
    let cohomology_by_jet: Vec<Vec<usize>> = (0..=WINDOW)
        .map(|jet| {
            let c0 = totals
                .iter()
                .zip(&jet_numbers)
                .filter(|(total, number)| **total == 0 && **number == jet)
                .count();
            let c1 = totals
                .iter()
                .zip(&jet_numbers)
                .filter(|(total, number)| **total == 1 && **number == jet)
                .count();
            let c2 = totals
                .iter()
                .zip(&jet_numbers)
                .filter(|(total, number)| **total == 2 && **number == jet)
                .count();
            let rank_here = if jet == 0 { q1_rank } else { 0 };
            let rank_above = jet_block_ranks[jet as usize];
            vec![c0 - rank_here, c1 - rank_here - rank_above, c2 - rank_above]
        })
        .collect();

    let basis_hash = canonical_json_sha256(&json!({
        "window": WINDOW,
        "states": states
            .iter()
            .map(|basis| {
                json!({
                    "bosons": basis.bosons(),
                    "fermions": basis.fermions(),
                })
            })
            .collect::<Vec<_>>(),
    }));

    let h0_hash = canonical_json_sha256(&json!({
        "degree": 0,
        "dimension": 1,
        "vectors": [[[1.0, 0.0]]],
    }));
    let (h1_dimension, h1_hash, h2_dimension, h2_hash, h2_pivots) = if include_jets {
        let mut h1_blocks = Vec::new();
        let mut h2_blocks = Vec::new();
        let mut pivots = Vec::new();
        let mut h1_total = 0;
        let mut h2_total = 0;
        for jet in 0..=WINDOW {
            for matter in 0..=WINDOW {
                let rows: Vec<usize> = (0..rows1.len())
                    .filter(|&row| {
                        jet_numbers[rows1[row]] == jet && matter_numbers[rows1[row]] == matter
                    })
                    .collect();
                let columns: Vec<usize> = (0..columns2.len())
                    .filter(|&column| {
                        jet_numbers[columns2[column]] == jet
                            && matter_numbers[columns2[column]] == matter
                    })
                    .collect();
                let block = submatrix(&q2, &rows, &columns);
                let (h1_dimension, h1_hash) = if rows.is_empty() {
                    (0, empty_basis_hash(1, 0))
                } else if columns.is_empty() {
                    let identity = (0..rows.len())
                        .map(|index| vec![(index, C::new(1.0, 0.0))])
                        .collect::<Vec<_>>();
                    let vector_hashes: Vec<String> = identity
                        .iter()
                        .map(|entries| canonical_json_sha256(&json!(entries_json(entries))))
                        .collect();
                    (
                        rows.len(),
                        canonical_json_sha256(&json!({
                            "degree": 1,
                            "dimension": rows.len(),
                            "vectors": vector_hashes,
                        })),
                    )
                } else {
                    left_nullspace(&block)
                };
                h1_total += h1_dimension;
                h1_blocks.push(json!({
                    "jet": jet,
                    "matter": matter,
                    "dimension": h1_dimension,
                    "basis_sha256": h1_hash,
                }));
                let (h2_dimension, h2_hash, local_pivots) = if columns.is_empty() {
                    (0, empty_basis_hash(2, 0), Vec::new())
                } else {
                    rref_nullspace(&block, 2)
                };
                h2_total += h2_dimension;
                pivots.extend(local_pivots.into_iter().map(|pivot| columns[pivot]));
                h2_blocks.push(json!({
                    "jet": jet,
                    "matter": matter,
                    "dimension": h2_dimension,
                    "basis_sha256": h2_hash,
                }));
            }
        }
        pivots.sort_unstable();
        (
            h1_total,
            canonical_json_sha256(&json!({"blocks": h1_blocks})),
            h2_total,
            canonical_json_sha256(&json!({"blocks": h2_blocks})),
            pivots,
        )
    } else {
        let (h1_dimension, h1_hash) = left_nullspace(&q2);
        let (h2_dimension, h2_hash, h2_pivots) = rref_nullspace(&q2, 2);
        (h1_dimension, h1_hash, h2_dimension, h2_hash, h2_pivots)
    };
    let representative_hash = canonical_json_sha256(&json!({
        "degree_0": h0_hash,
        "degree_1": h1_hash,
        "degree_2": h2_hash,
    }));

    WindowMeasurement {
        state_count: states.len(),
        dims,
        matter_dims,
        jet_dims,
        block_ranks: vec![q1_rank, q2_rank],
        cohomology_dimensions,
        cohomology_dimension,
        q_squared,
        matter_block_ranks,
        jet_block_ranks,
        gauge_singlet_kernel_dimension,
        cohomology_by_matter,
        cohomology_by_jet,
        basis_hash,
        h0_hash,
        h1_dimension,
        h1_hash,
        h2_dimension,
        h2_hash,
        h2_pivots,
        representative_hash,
        closed,
        graded,
        matter_graded,
        jet_graded,
        failure,
    }
}

fn window_record(measurement: WindowMeasurement, include_jets: bool) -> Value {
    if !measurement.closed || !measurement.graded || !measurement.jet_graded {
        return json!({
            "schema": if include_jets { "sm-finite-fock-brst-jets/v1" } else { "sm-finite-fock-brst/v2" },
            "status": "measured",
            "closed": measurement.closed,
            "graded": measurement.graded,
            "jet_graded": measurement.jet_graded,
            "reason": measurement.failure,
            "state_count": measurement.state_count,
        });
    }

    let construction = if include_jets {
        "bosonic gauge-Higgs plus all 96 complex matter fermion modes plus twelve ghosts plus the 120 independent derivative-jet coordinates, total occupation <= 2"
    } else {
        "bosonic gauge-Higgs plus all 96 complex matter fermion modes plus twelve ghosts, total occupation <= 2, derivative jets excluded"
    };
    let identity = if include_jets {
        json!({
            "window": WINDOW,
            "ghost_base": GHOST_BASE,
            "generators": GENERATORS,
            "matter_modes": MATTER_MODES,
            "jet_modes": jet_modes().len(),
        })
    } else {
        json!({
            "window": WINDOW,
            "ghost_base": GHOST_BASE,
            "generators": GENERATORS,
            "matter_modes": MATTER_MODES,
        })
    };

    let mut artifact = json!({
        "window": WINDOW,
        "occupation_dimensions": measurement.dims,
        "differential_rank": measurement.block_ranks.iter().sum::<usize>(),
        "block_ranks": measurement.block_ranks,
        "cohomology_dimensions": measurement.cohomology_dimensions,
        "cohomology_dimension": measurement.cohomology_dimension,
        "cohomology_by_matter_number": measurement.cohomology_by_matter,
        "matter_state_dimensions": measurement.matter_dims,
        "matter_block_ranks": measurement.matter_block_ranks,
        "gauge_singlet_kernel_dimension": measurement.gauge_singlet_kernel_dimension,
        "q_squared_residual": reported_residual(measurement.q_squared),
        "tolerance": TOLERANCE,
        "state_count": measurement.state_count,
        "state_basis_sha256": measurement.basis_hash,
        "field_level": true,
        "finite_window": true,
        "physical_continuum_quotient": false,
        "harmonic_representatives": {
            "degree_0": {"dimension": 1, "basis_sha256": measurement.h0_hash},
            "degree_1": {"dimension": measurement.h1_dimension, "basis_sha256": measurement.h1_hash},
            "degree_2": {
                "dimension": measurement.h2_dimension,
                "basis_sha256": measurement.h2_hash,
                "pivot_columns": measurement.h2_pivots,
            },
        },
        "quotient_map": {
            "kind": if include_jets {
                "jet- and matter-block-graded finite harmonic representatives (ker Q_n intersect ker Q_{n+1}*)"
            } else {
                "finite harmonic representatives (ker Q_n intersect ker Q_{n+1}*)"
            },
            "degrees": [0, 1, 2],
            "dimensions": measurement.cohomology_dimensions,
            "representative_basis_sha256": measurement.representative_hash,
            "degree_0": "vacuum representative",
            "degree_1": if include_jets {
                "block-graded left nullspace of Q_2"
            } else {
                "left nullspace of Q_2"
            },
            "degree_2": if include_jets {
                "block-graded reduced-row-echelon free-column nullspace of Q_2"
            } else {
                "reduced-row-echelon free-column nullspace of Q_2"
            },
            "field_level": true,
            "finite_quotient": true,
            "continuum": false,
        },
    });
    if include_jets {
        let artifact = artifact.as_object_mut().unwrap();
        artifact.insert(
            "jet_state_dimensions".to_string(),
            json!(measurement.jet_dims),
        );
        artifact.insert(
            "jet_block_ranks".to_string(),
            json!(measurement.jet_block_ranks),
        );
        artifact.insert(
            "cohomology_by_jet_number".to_string(),
            json!(measurement.cohomology_by_jet),
        );
        artifact.insert("jet_graded".to_string(), json!(measurement.jet_graded));
        artifact.insert(
            "construction_note".to_string(),
            json!("The 120 jet coordinates are independent E1 fields, not constrained to spatial derivatives; the jet-number grading is measured, not assumed."),
        );
    }

    if include_jets {
        json!({
            "schema": "sm-finite-fock-brst-jets/v1",
            "claim_id": "n10_finite_fock_brst_window_jets",
            "status": "measured",
            "construction": construction,
            "closed": measurement.closed,
            "graded": measurement.graded,
            "matter_graded": measurement.matter_graded,
            "jet_graded": measurement.jet_graded,
            "operator": {
                "name": "joint BRST charge on the occupation filtration including derivative jets",
                "identity_sha256": canonical_json_sha256(&identity),
            },
            "artifact": artifact,
            "theorem": {
                "lean_name": "BookProof.SmBrstQuotient.finiteWindow_jet_finrank",
                "statement": "For a three-term complex with q1 o q2 = 0, C0/C1/C2 dimensions 1/268/35938 and block ranks 0/249, the finite cohomology dimensions are 1/19/35689.",
                "proof_status": "kernel-verified conditional",
                "payload": "docs/fixtures/sm_brst_jet_window_proof.ndjson.gz",
                "payload_sha256": "fcd22f8950e4066787a570dd263c3d921439b203e9098b4aa64211e9048c5fe5",
                "raw_sha256": "66ea3e84d009e03c9d34f86d7783def11f46f0c9600d6cc094320f1abf462e20",
                "declarations": 22097,
                "checker": "nanoda_lib via prob_kernel::verify::verify_export",
                "matrix_q_sq": {
                    "lean_name": "BookProof.SmBrstQuotient.jetWindowCharge_sq",
                    "statement": "the block total charge on C0 + C1 + C2 with dimensions 1/268/35938 has Q * Q = 0 when q1 * q2 = 0",
                    "proof_status": "kernel-verified conditional",
                    "scope": "the block-matrix implication is formalized; the measured entries and their q1 * q2 residual remain in the Rust artifact",
                },
                "numeric_binding": "The exported Lean theorems prove the arithmetic consequence of q1 o q2 = 0 and the measured block ranks, and the block-matrix Q^2 = 0 implication. The concrete 36,207-state matrix and its measured q_squared_residual are not formalized. All dimensions, ranks and residuals in this record are measured at tolerance 1e-9 and are not interval enclosures.",
            },
            "hypotheses": [
                "The window is the Q-invariant total-occupation filtration N <= 2.",
                "All 96 complex matter fermion modes, the twelve ghosts, the 40 homogeneous coordinates and the 120 independent derivative-jet coordinates are included.",
                "Q preserves matter number and derivative-jet number and lowers total occupation by exactly one on this window; all three are measured here.",
                "Ranks and harmonic bases are SVD/RREF measurements with tolerance 1e-9, not interval enclosures.",
            ],
            "not_claimed": [
                "No continuum, anomaly or physical-sector identification.",
                "No claim that the independent jet coordinates equal spatial derivatives of the homogeneous fields.",
                "The finite harmonic quotient is not the physical Krein quotient.",
                "No Lean formalization of the concrete 36,207-state matrix or of the measured entries' q1 * q2 = 0 residual; the Lean Q^2 = 0 theorem is conditional on that block identity.",
            ],
        })
    } else {
        json!({
            "schema": "sm-finite-fock-brst/v2",
            "claim_id": "n10_finite_fock_brst_window",
            "status": "measured",
            "construction": construction,
            "closed": measurement.closed,
            "graded": measurement.graded,
            "matter_graded": measurement.matter_graded,
            "operator": {
                "name": "joint BRST charge on the occupation filtration",
                "identity_sha256": canonical_json_sha256(&identity),
            },
            "artifact": artifact,
            "theorem": {
                "lean_name": "BookProof.SmBrstQuotient.finiteWindow_matter_finrank",
                "statement": "For a three-term complex with q1 o q2 = 0, C0/C1/C2 dimensions 1/148/10918 and block ranks 0/138, the finite cohomology dimensions are 1/10/10780.",
                "proof_status": "kernel-verified conditional",
                "payload": "docs/fixtures/sm_brst_finite_window_proof.ndjson.gz",
                "payload_sha256": "e25d796a16b99689a23bb756324ec261ba08f3ccbf8a6026d7de814b810d7ffa",
                "raw_sha256": "bedf7935645a691ed994869474b8e38409f7541ad314fa1c6c86255ca881664a",
                "declarations": 20241,
                "numeric_binding": "The Lean theorem proves the arithmetic consequence of q1 o q2 = 0 and the measured block ranks; the 11,067-dimensional matrix itself is measured in Rust, not formalized.",
            },
            "hypotheses": [
                "The window is the Q-invariant total-occupation filtration N <= 2.",
                "All 96 complex matter fermion modes and the twelve ghosts are included; derivative jets are excluded.",
                "Q preserves matter number and lowers total occupation by exactly one on this window; both are measured here.",
                "Ranks and harmonic bases are SVD/RREF measurements with tolerance 1e-9, not interval enclosures.",
                "The Lean theorem `finiteWindow_matter_finrank` is conditional on the measured block ranks and dimensions; it does not formalize the concrete matrix.",
            ],
            "not_claimed": [
                "No derivative-jet cohomology.",
                "No continuum, anomaly or physical-sector identification.",
                "The finite harmonic quotient is not the physical Krein quotient.",
                "No Lean formalization of the 11,067-dimensional matrix or its Q^2 = 0 identity.",
            ],
        })
    }
}

/// Build the finite field-level BRST window record without derivative jets.
pub fn finite_fock_brst_window() -> Value {
    window_record(measure_window(false), false)
}

/// Build the finite field-level BRST window record including all 120
/// independent derivative-jet coordinates. The jet number is conserved by the
/// joint charge, so `Q_2` is measured blockwise in `(jet, matter)`.
pub fn finite_fock_brst_jet_window() -> Value {
    window_record(measure_window(true), true)
}
