//! Finite field-level BRST window.
//!
//! The joint charge removes one ghost and preserves the remaining occupation, so
//! the total-occupation filtration `N <= M` is `Q`-invariant. This module builds
//! the smallest nontrivial window (`M = 2`) for the bosonic gauge-Higgs sector
//! plus ghosts. Matter fermions and derivative jets are excluded; the record
//! reports whether `Q` stayed inside the chosen sector.

use crate::sm_bridge::canonical_json_sha256;
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
    NestedVector::from([(
        O::graded([(basis.clone(), 1)]).unwrap(),
        C::new(1.0, 0.0),
    )])
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

fn ghost_subsets(ghosts: &[u32], max: u32) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new()];
    if max == 0 {
        return out;
    }
    for &ghost in ghosts {
        out.push(vec![ghost]);
    }
    if max >= 2 {
        for (i, &left) in ghosts.iter().enumerate() {
            for &right in &ghosts[(i + 1)..] {
                out.push(vec![left, right]);
            }
        }
    }
    out
}

fn occupation_total(bosons: &[(u32, u32)], ghosts: &[u32]) -> u32 {
    bosons.iter().map(|(_, n)| *n).sum::<u32>() + ghosts.len() as u32
}

fn sector_basis() -> (Vec<B>, Vec<u32>) {
    let modes = coord::active();
    let ghosts: Vec<u32> = (0..GENERATORS as u32)
        .map(|a| GHOST_BASE + a)
        .collect();
    let mut states = Vec::new();
    let mut totals = Vec::new();
    for bosons in boson_configurations(&modes, WINDOW) {
        for ghost in ghost_subsets(&ghosts, WINDOW) {
            let total = occupation_total(&bosons, &ghost);
            if total > WINDOW {
                continue;
            }
            let basis = B::new(bosons.clone(), ghost).unwrap();
            totals.push(total);
            states.push(basis);
        }
    }
    (states, totals)
}

fn rank(matrix: &DMatrix<C>) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    SVD::new(matrix.clone(), false, false)
        .singular_values
        .iter()
        .filter(|value| **value > 1e-9)
        .count()
}

/// Build the finite field-level BRST window record.
pub fn finite_fock_brst_window() -> Value {
    let q = brst_charge(&parameters());
    let (states, totals) = sector_basis();
    let index: BTreeMap<B, usize> = states
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, basis)| (basis, position))
        .collect();

    let mut columns: Vec<Vec<(usize, C)>> = vec![Vec::new(); states.len()];
    let mut closed = true;
    let mut q_squared = 0.0_f64;
    for (column, basis) in states.iter().enumerate() {
        let first = image(&q, basis).unwrap();
        for (target, coefficient) in &first {
            match index.get(target) {
                Some(&row) => columns[column].push((row, *coefficient)),
                None => {
                    return json!({
                        "schema": "sm-finite-fock-brst/v1",
                        "status": "measured",
                        "closed": false,
                        "reason": "Q left the bosonic gauge-Higgs + ghost sector",
                        "input_occupation": totals[column],
                    });
                }
            }
        }
        // Q^2 on this column.
        let mut second: BTreeMap<usize, C> = BTreeMap::new();
        for (middle, coefficient) in &first {
            let next = image(&q, middle).unwrap();
            for (target, value) in next {
                if let Some(&row) = index.get(&target) {
                    *second.entry(row).or_default() += coefficient * value;
                } else {
                    return json!({
                        "schema": "sm-finite-fock-brst/v1",
                        "status": "measured",
                        "closed": false,
                        "reason": "Q^2 left the bosonic gauge-Higgs + ghost sector",
                        "input_occupation": totals[column],
                    });
                }
            }
        }
        q_squared = q_squared.max(
            second
                .values()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt(),
        );
    }

    let dims: Vec<usize> = (0..=WINDOW)
        .map(|n| totals.iter().filter(|&&total| total == n).count())
        .collect();
    let mut q_matrix = DMatrix::zeros(states.len(), states.len());
    for (column, _basis) in states.iter().enumerate() {
        for (row, value) in &columns[column] {
            q_matrix[(*row, column)] += *value;
        }
    }
    let q_rank = rank(&q_matrix);
    let cohomology_dimension = states.len() - q_rank;
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

    json!({
        "schema": "sm-finite-fock-brst/v1",
        "claim_id": "n10_finite_fock_brst_window",
        "status": "measured",
        "construction": "bosonic gauge-Higgs plus ghosts, total occupation <= 2, matter fermions and derivative jets excluded",
        "closed": closed,
        "operator": {
            "name": "joint BRST charge on the occupation filtration",
            "identity_sha256": canonical_json_sha256(&json!({
                "window": WINDOW,
                "ghost_base": GHOST_BASE,
                "generators": GENERATORS,
            })),
        },
        "artifact": {
            "window": WINDOW,
            "occupation_dimensions": dims,
            "differential_rank": q_rank,
            "cohomology_dimension": cohomology_dimension,
            "q_squared_residual": q_squared,
            "tolerance": 1e-9,
            "state_count": states.len(),
            "state_basis_sha256": basis_hash,
            "field_level": true,
            "finite_window": true,
            "physical_continuum_quotient": false,
        },
        "hypotheses": [
            "The window is the Q-invariant total-occupation filtration N <= 2.",
            "Only bosonic gauge and Higgs modes plus ghosts are included; matter fermions and derivative jets are excluded.",
            "The identity Q(all) = Q(bosonic gauge-Higgs + ghost) is measured on this window, not proved for every mode.",
        ],
        "not_claimed": [
            "No matter-fermion or derivative-jet cohomology.",
            "No continuum, anomaly or physical-sector identification.",
        ],
    })
}
