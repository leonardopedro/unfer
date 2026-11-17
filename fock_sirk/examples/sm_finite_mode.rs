//! N12 finite-mode embedding pilot.
//!
//! Defined family: modes `k = 0..M-1` with a fixed dispersion
//! `omega_k = sqrt(m^2 + k^2)`. The quadratic Hamiltonian is
//! `H0 = sum omega_k N_k`; the interaction is
//! `V = lambda sum_{k<K} (phi_k)^4`, supported on the first `K` modes and
//! matrix-ordered on a truncated occupation basis. Refinement `M -> M+1`
//! appends one free vacuum mode, so the embedding `E: H_M -> H_{M+1}` is
//! isometric and `E^dagger H_{M+1} E = H_M`.
//!
//! State families: coherent states and squeezed vacua whose nonzero
//! amplitudes live only in the first `K` modes. Both therefore embed exactly,
//! with the appended mode in its vacuum.
//!
//! Two families are reported. The **fixed-coupling** family keeps
//! `V = lambda sum_{k<4}(phi_k)^4` and is embedded exactly as `M` grows. The
//! **running-coupling** family lets every mode up to the cutoff interact,
//! `V_M = lambda_M sum_{k<M}(phi_k)^4`, and determines `lambda_M` by a stated
//! renormalization condition: the ground-state connected four-point function
//! of the fixed smeared field,
//!
//! `G4 = <(delta phi_s)^4> - 3 <(delta phi_s)^2>^2`,
//!
//! is held equal to its value at the reference cutoff `M=4` with
//! `lambda=0.05`. This is a finite-mode, fixed-smearing normalization; it is
//! not a 4d QFT renormalization scheme, and it does not supply a cutoff-uniform
//! theorem. The running family's Hamiltonian is not embedded exactly, by
//! design: the appended modes carry interactions.

use nalgebra::{DMatrix, DVector};
use serde_json::{Value, json};
use std::collections::HashMap;

const SUPPORT: usize = 4;
const MAX_MODE: usize = 8;
const MAX_TOTAL_OCCUPATION: u32 = 4;
const MASS: f64 = 0.4;
const LAMBDA: f64 = 0.05;
const COHERENT_AMPLITUDE: f64 = 0.12;
const SQUEEZING: f64 = 0.2;
const KERNEL_SCALE: f64 = 2.0;
const ALT_KERNEL_SCALE: f64 = 1.0;
const MATCH_ITERATIONS: usize = 32;
const MATCH_TOLERANCE: f64 = 1e-10;

fn basis(mode_count: usize) -> Vec<Vec<u32>> {
    basis_with_cutoff(mode_count, MAX_TOTAL_OCCUPATION)
}

fn basis_with_cutoff(mode_count: usize, cutoff: u32) -> Vec<Vec<u32>> {
    fn recurse(
        remaining_modes: usize,
        remaining_occupation: u32,
        current: &mut Vec<u32>,
        out: &mut Vec<Vec<u32>>,
    ) {
        if remaining_modes == 0 {
            out.push(current.clone());
            return;
        }
        for occupation in 0..=remaining_occupation {
            current.push(occupation);
            recurse(
                remaining_modes - 1,
                remaining_occupation - occupation,
                current,
                out,
            );
            current.pop();
        }
    }
    let mut out = Vec::new();
    recurse(
        mode_count,
        cutoff,
        &mut Vec::with_capacity(mode_count),
        &mut out,
    );
    out
}

fn index_map(states: &[Vec<u32>]) -> HashMap<Vec<u32>, usize> {
    states
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, state)| (state, index))
        .collect()
}

fn ladder(state: &[u32], mode: usize, create: bool) -> Option<(Vec<u32>, f64)> {
    let occupation = state[mode];
    if create {
        let mut out = state.to_vec();
        out[mode] += 1;
        Some((out, ((occupation + 1) as f64).sqrt()))
    } else if occupation == 0 {
        None
    } else {
        let mut out = state.to_vec();
        out[mode] -= 1;
        Some((out, (occupation as f64).sqrt()))
    }
}

fn phi_matrix(mode: usize, states: &[Vec<u32>], index: &HashMap<Vec<u32>, usize>) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(states.len(), states.len());
    for (column, state) in states.iter().enumerate() {
        for (target_mode, create) in [(mode, false), (mode, true)] {
            if let Some((target, coefficient)) = ladder(state, target_mode, create) {
                if let Some(&row) = index.get(&target) {
                    matrix[(row, column)] += coefficient;
                }
            }
        }
    }
    matrix
}

fn number_matrix(states: &[Vec<u32>]) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(states.len(), states.len());
    for (index, state) in states.iter().enumerate() {
        matrix[(index, index)] = state.iter().sum::<u32>() as f64;
    }
    matrix
}

fn free_hamiltonian(states: &[Vec<u32>]) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(states.len(), states.len());
    for (index, state) in states.iter().enumerate() {
        matrix[(index, index)] = state
            .iter()
            .enumerate()
            .map(|(mode, occupation)| {
                let omega = (MASS * MASS + (mode as f64) * (mode as f64)).sqrt();
                omega * f64::from(*occupation)
            })
            .sum();
    }
    matrix
}

fn interaction(phi: &[DMatrix<f64>], lambda: f64) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(phi[0].nrows(), phi[0].ncols());
    for phi_mode in phi {
        let squared = phi_mode * phi_mode;
        matrix += &squared * &squared;
    }
    matrix *= lambda;
    matrix
}

fn smeared_field(phi: &[DMatrix<f64>], weights: &[f64]) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(phi[0].nrows(), phi[0].ncols());
    for mode in 0..SUPPORT {
        matrix += &phi[mode] * weights[mode];
    }
    matrix
}

fn coherent_amplitudes() -> Vec<f64> {
    (0..SUPPORT)
        .map(|mode| COHERENT_AMPLITUDE * (-(mode as f64) / KERNEL_SCALE).exp())
        .collect()
}

fn coherent_state(states: &[Vec<u32>], amplitudes: &[f64], mode_count: usize) -> DVector<f64> {
    let mut vector = DVector::zeros(states.len());
    for (index, state) in states.iter().enumerate() {
        let mut coefficient = 1.0;
        for mode in 0..mode_count {
            let occupation = state[mode];
            let amplitude = amplitudes.get(mode).copied().unwrap_or(0.0);
            coefficient *= (-0.5 * amplitude * amplitude).exp();
            for level in 1..=occupation {
                coefficient *= amplitude / (level as f64).sqrt();
            }
        }
        vector[index] = coefficient;
    }
    vector /= vector.norm();
    vector
}

fn squeezed_state(states: &[Vec<u32>], squeezing: &[f64], mode_count: usize) -> DVector<f64> {
    let mut vector = DVector::zeros(states.len());
    for (index, state) in states.iter().enumerate() {
        let mut coefficient = 1.0;
        for mode in 0..mode_count {
            let occupation = state[mode];
            let r = squeezing.get(mode).copied().unwrap_or(0.0);
            if occupation % 2 == 1 {
                coefficient = 0.0;
                break;
            }
            coefficient *= 1.0 / r.cosh().sqrt();
            let tanh = r.tanh();
            for pair in 0..(occupation / 2) {
                coefficient *=
                    tanh * ((2 * pair + 1) as f64).sqrt() / ((2 * pair + 2) as f64).sqrt();
            }
        }
        vector[index] = coefficient;
    }
    vector /= vector.norm();
    vector
}

fn expectation(vector: &DVector<f64>, matrix: &DMatrix<f64>) -> f64 {
    vector.dot(&(matrix * vector))
}

fn ground_state(hamiltonian: &DMatrix<f64>) -> DVector<f64> {
    let spectrum = hamiltonian.clone().symmetric_eigen();
    let index = spectrum
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("nonempty basis")
        .0;
    spectrum.eigenvectors.column(index).into_owned()
}

/// Connected four-point function of the smeared field in a normalized state:
/// `<psi|(delta phi)^4|psi> - 3 <psi|(delta phi)^2|psi>^2`.
fn connected_four_point(state: &DVector<f64>, field: &DMatrix<f64>) -> f64 {
    let mean = expectation(state, field);
    let first = field * state - state * mean;
    let second = field * &first - &first * mean;
    let moment_two = first.norm_squared();
    let moment_four = second.norm_squared();
    moment_four - 3.0 * moment_two * moment_two
}

struct MatchResult {
    lambda: f64,
    value: f64,
    roots: Vec<f64>,
}

/// Solve `G4(lambda) = target` by scanning for sign changes and then bisecting
/// every bracket. The chosen root is the one closest to the previous cutoff's
/// coupling; all roots are returned so that branch selection stays visible.
fn matching_coupling(
    h0: &DMatrix<f64>,
    interaction_unit: &DMatrix<f64>,
    field: &DMatrix<f64>,
    target: f64,
    hint: f64,
) -> MatchResult {
    let g4 = |lambda: f64| {
        let hamiltonian = h0 + interaction_unit * lambda;
        connected_four_point(&ground_state(&hamiltonian), field)
    };
    // Added modes can move the matching coupling non-monotonically, so scan for
    // every sign change and keep the one closest to the previous cutoff's
    // coupling, instead of assuming a bracket near the reference.
    const SCAN: usize = 64;
    let scan_high = 8.0 * LAMBDA;
    let mut previous_lambda = 0.0;
    let mut previous_value = g4(previous_lambda);
    let mut brackets = Vec::new();
    for index in 1..=SCAN {
        let lambda = scan_high * index as f64 / SCAN as f64;
        let value = g4(lambda);
        if (previous_value - target) * (value - target) <= 0.0 {
            brackets.push((previous_lambda, previous_value, lambda, value));
        }
        previous_lambda = lambda;
        previous_value = value;
    }
    let bisect = |mut low: f64, mut low_value: f64, mut high: f64| {
        for _ in 0..MATCH_ITERATIONS {
            let middle = 0.5 * (low + high);
            let value = g4(middle);
            if (value - target).abs() <= MATCH_TOLERANCE * target.abs().max(1.0) {
                return (middle, value);
            }
            if (low_value - target) * (value - target) <= 0.0 {
                high = middle;
            } else {
                low = middle;
                low_value = value;
            }
        }
        (high, g4(high))
    };

    let mut roots: Vec<f64> = brackets
        .iter()
        .map(|&(low, low_value, high, _)| bisect(low, low_value, high).0)
        .collect();
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let Some(&(low, low_value, high, _)) = brackets.iter().min_by(|a, b| {
        let a_mid = 0.5 * (a.0 + a.2);
        let b_mid = 0.5 * (b.0 + b.2);
        (a_mid - hint)
            .abs()
            .partial_cmp(&(b_mid - hint).abs())
            .unwrap()
    }) else {
        return MatchResult {
            lambda: f64::NAN,
            value: f64::NAN,
            roots,
        };
    };
    let (lambda, value) = bisect(low, low_value, high);
    MatchResult {
        lambda,
        value,
        roots,
    }
}

/// Connected four-point function at fixed coupling and occupation cutoff.
fn four_point_at_cutoff(mode_count: usize, cutoff: u32, lambda: f64) -> f64 {
    let states = basis_with_cutoff(mode_count, cutoff);
    let index = index_map(&states);
    let phi: Vec<_> = (0..mode_count)
        .map(|mode| phi_matrix(mode, &states, &index))
        .collect();
    let weights: Vec<_> = (0..SUPPORT)
        .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp())
        .collect();
    let field = smeared_field(&phi[..SUPPORT], &weights);
    let h0 = free_hamiltonian(&states);
    let unit = interaction(&phi, 1.0);
    let hamiltonian = h0 + unit * lambda;
    connected_four_point(&ground_state(&hamiltonian), &field)
}

/// Roots of the same four-point condition at a chosen total-occupation cutoff.
fn roots_at_cutoff(mode_count: usize, cutoff: u32, target: f64) -> Vec<f64> {
    let states = basis_with_cutoff(mode_count, cutoff);
    let index = index_map(&states);
    let phi: Vec<_> = (0..mode_count)
        .map(|mode| phi_matrix(mode, &states, &index))
        .collect();
    let weights: Vec<_> = (0..SUPPORT)
        .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp())
        .collect();
    let field = smeared_field(&phi[..SUPPORT], &weights);
    let h0 = free_hamiltonian(&states);
    let unit = interaction(&phi, 1.0);
    matching_coupling(&h0, &unit, &field, target, LAMBDA).roots
}

fn variance(vector: &DVector<f64>, matrix: &DMatrix<f64>) -> f64 {
    let mean = expectation(vector, matrix);
    let squared = matrix * matrix;
    (expectation(vector, &squared) - mean * mean).max(0.0)
}

fn embedding_deviation(low: &DMatrix<f64>, high: &DMatrix<f64>, embedded: &[usize]) -> f64 {
    let mut worst = 0.0f64;
    for (i, &high_i) in embedded.iter().enumerate() {
        for (j, &high_j) in embedded.iter().enumerate() {
            worst = worst.max((low[(i, j)] - high[(high_i, high_j)]).abs());
        }
    }
    worst
}

#[allow(clippy::too_many_arguments)]
fn cell_from(
    mode_count: usize,
    states: &[Vec<u32>],
    phi: &[DMatrix<f64>],
    field: &DMatrix<f64>,
    number: &DMatrix<f64>,
    hamiltonian: &DMatrix<f64>,
    lambda: f64,
    support_modes: usize,
    g4_residual: Option<f64>,
) -> Value {
    let coherent_amplitudes = coherent_amplitudes();
    let squeezing: Vec<_> = (0..SUPPORT)
        .map(|mode| SQUEEZING * (-(mode as f64) / KERNEL_SCALE).exp())
        .collect();
    let coherent = coherent_state(states, &coherent_amplitudes, mode_count);
    let squeezed = squeezed_state(states, &squeezing, mode_count);

    let weights: Vec<_> = (0..SUPPORT)
        .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp())
        .collect();
    let kernel_mass: f64 = (0..mode_count)
        .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp().powi(2))
        .sum();
    let support_mass: f64 = weights.iter().map(|weight| weight * weight).sum();
    let ground = ground_state(hamiltonian);

    json!({
        "modes": mode_count,
        "basis_dimension": states.len(),
        "lambda": lambda,
        "ground_connected_g4": connected_four_point(&ground, field),
        "g4_residual": g4_residual,
        "coherent": {
            "number": expectation(&coherent, number),
            "energy": expectation(&coherent, hamiltonian),
            "smeared_field_mean": expectation(&coherent, field),
            "smeared_field_variance": variance(&coherent, field),
        },
        "squeezed": {
            "number": expectation(&squeezed, number),
            "energy": expectation(&squeezed, hamiltonian),
            "smeared_field_mean": expectation(&squeezed, field),
            "smeared_field_variance": variance(&squeezed, field),
        },
        "kernel_mass": kernel_mass,
        "support_mass_fraction": support_mass / kernel_mass,
        "support_modes": support_modes,
        "interacting_modes": phi.len(),
    })
}

fn spread(rows: &[Value], family: &str, observable: &str) -> f64 {
    let values: Vec<f64> = rows
        .iter()
        .map(|row| row[family][observable].as_f64().unwrap())
        .collect();
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    high - low
}

fn scalar_spread(rows: &[Value], key: &str) -> f64 {
    let values: Vec<f64> = rows.iter().map(|row| row[key].as_f64().unwrap()).collect();
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    high - low
}

fn main() {
    // Reference normalization: the connected four-point function of the fixed
    // smeared field at the smallest cutoff, with the declared coupling.
    let reference_states = basis(SUPPORT);
    let reference_index = index_map(&reference_states);
    let reference_phi: Vec<_> = (0..SUPPORT)
        .map(|mode| phi_matrix(mode, &reference_states, &reference_index))
        .collect();
    let reference_weights: Vec<_> = (0..SUPPORT)
        .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp())
        .collect();
    let reference_field = smeared_field(&reference_phi, &reference_weights);
    let reference_h0 = free_hamiltonian(&reference_states);
    let reference_unit = interaction(&reference_phi, 1.0);
    let reference_h = &reference_h0 + &reference_unit * LAMBDA;
    let target_g4 = connected_four_point(&ground_state(&reference_h), &reference_field);
    assert!(
        target_g4.is_finite(),
        "reference four-point value is not finite"
    );
    // Second normalization: same Hamiltonian family, different smearing.
    let alt_reference_weights: Vec<_> = (0..SUPPORT)
        .map(|mode| (-(mode as f64) / ALT_KERNEL_SCALE).exp())
        .collect();
    let alt_reference_field = smeared_field(&reference_phi, &alt_reference_weights);
    let alt_target_g4 = connected_four_point(&ground_state(&reference_h), &alt_reference_field);
    assert!(
        alt_target_g4.is_finite(),
        "alternate reference four-point value is not finite"
    );

    // The two-root structure could be an artefact of the occupation truncation,
    // so repeat the matching at looser and tighter cutoffs.
    let mut cutoff_rows = Vec::new();
    for mode_count in SUPPORT + 1..=MAX_MODE {
        for cutoff in [2u32, 3, 4, 5, 6] {
            // Cutoff 6 and the 1287-state cutoff-5 cell at M=8 are only
            // branch-structure checks and dominate the matching cost.
            if cutoff >= 6 && mode_count > 6 {
                continue;
            }
            if cutoff == 5 && mode_count > 7 {
                continue;
            }
            let progress = std::env::var_os("N12_PROGRESS").is_some();
            if progress {
                eprintln!("cutoff cell start M={mode_count} cutoff={cutoff}");
            }
            let roots = roots_at_cutoff(mode_count, cutoff, target_g4);
            if progress {
                eprintln!("cutoff cell done M={mode_count} cutoff={cutoff} roots={roots:?}");
            }
            cutoff_rows.push(json!({
                "modes": mode_count,
                "occupation_cutoff": cutoff,
                "basis_dimension": basis_with_cutoff(mode_count, cutoff).len(),
                "roots": roots,
            }));
        }
    }

    // At fixed coupling this isolates basis drift from the matching step: if
    // G4 itself moves with the truncation, matching it to a fixed reference
    // forces a cutoff-dependent lambda by construction.
    let mut fixed_coupling_rows = Vec::new();
    for mode_count in SUPPORT..=MAX_MODE {
        for cutoff in [2u32, 3, 4, 5, 6] {
            if cutoff >= 6 && mode_count > 6 {
                continue;
            }
            if cutoff == 5 && mode_count > 7 {
                continue;
            }
            let g4 = four_point_at_cutoff(mode_count, cutoff, LAMBDA);
            fixed_coupling_rows.push(json!({
                "modes": mode_count,
                "occupation_cutoff": cutoff,
                "basis_dimension": basis_with_cutoff(mode_count, cutoff).len(),
                "lambda": LAMBDA,
                "g4": g4,
                "deviation_from_target": g4 - target_g4,
            }));
        }
    }

    let mut rows = Vec::new();
    let mut running_rows = Vec::new();
    let mut alt_rows = Vec::new();
    let mut checks = Vec::new();
    let mut running_checks = Vec::new();
    let mut lambda_hint = LAMBDA;
    let mut alt_lambda_hint = LAMBDA;
    let mut previous_fixed: Option<(usize, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> = None;
    let mut previous_running: Option<(usize, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> = None;
    for mode_count in SUPPORT..=MAX_MODE {
        if std::env::var_os("N12_PROGRESS").is_some() {
            eprintln!("main cell start M={mode_count}");
        }
        let states = basis(mode_count);
        let index = index_map(&states);
        let phi: Vec<_> = (0..mode_count)
            .map(|mode| phi_matrix(mode, &states, &index))
            .collect();
        let weights: Vec<_> = (0..SUPPORT)
            .map(|mode| (-(mode as f64) / KERNEL_SCALE).exp())
            .collect();
        let field = smeared_field(&phi[..SUPPORT], &weights);
        let number = number_matrix(&states);
        let coherent_vector = coherent_state(&states, &coherent_amplitudes(), mode_count);
        let h0 = free_hamiltonian(&states);

        let fixed_unit = interaction(&phi[..SUPPORT], 1.0);
        let fixed_h = &h0 + &fixed_unit * LAMBDA;
        let all_unit = interaction(&phi, 1.0);
        let matched = if mode_count == SUPPORT {
            MatchResult {
                lambda: LAMBDA,
                value: target_g4,
                roots: vec![LAMBDA],
            }
        } else {
            matching_coupling(&h0, &all_unit, &field, target_g4, lambda_hint)
        };
        let (lambda, g4) = (matched.lambda, matched.value);
        lambda_hint = lambda;
        assert!(
            lambda.is_finite() && lambda > 0.0,
            "matching coupling {lambda} is not positive at M={mode_count}"
        );
        let running_h = &h0 + &all_unit * lambda;
        let g4_residual = (g4 - target_g4).abs();
        let low_lambda = *matched.roots.first().unwrap();
        let low_h = &h0 + &all_unit * low_lambda;
        let low_energy = expectation(&coherent_vector, &low_h);
        let chosen_energy = expectation(&coherent_vector, &running_h);

        let alt_weights: Vec<_> = (0..SUPPORT)
            .map(|mode| (-(mode as f64) / ALT_KERNEL_SCALE).exp())
            .collect();
        let alt_field = smeared_field(&phi[..SUPPORT], &alt_weights);
        let alt_matched = if mode_count == SUPPORT {
            MatchResult {
                lambda: LAMBDA,
                value: alt_target_g4,
                roots: vec![LAMBDA],
            }
        } else {
            matching_coupling(&h0, &all_unit, &alt_field, alt_target_g4, alt_lambda_hint)
        };
        let (alt_lambda, alt_g4) = (alt_matched.lambda, alt_matched.value);
        alt_lambda_hint = alt_lambda;
        assert!(
            alt_lambda.is_finite() && alt_lambda > 0.0,
            "alternate matching coupling {alt_lambda} is not positive at M={mode_count}"
        );
        let alt_h = &h0 + &all_unit * alt_lambda;
        let alt_g4_residual = (alt_g4 - alt_target_g4).abs();

        if let Some((low_modes, low_h, low_n, low_field)) = previous_fixed.take() {
            let embedded: Vec<usize> = basis(low_modes)
                .iter()
                .map(|state| {
                    let mut extended = state.clone();
                    extended.push(0);
                    index[&extended]
                })
                .collect();
            checks.push(json!({
                "from_modes": low_modes,
                "to_modes": mode_count,
                "hamiltonian_max_deviation": embedding_deviation(&low_h, &fixed_h, &embedded),
                "number_max_deviation": embedding_deviation(&low_n, &number, &embedded),
                "smeared_field_max_deviation": embedding_deviation(&low_field, &field, &embedded),
            }));
        }
        if let Some((low_modes, low_h, _, _)) = previous_running.take() {
            let embedded: Vec<usize> = basis(low_modes)
                .iter()
                .map(|state| {
                    let mut extended = state.clone();
                    extended.push(0);
                    index[&extended]
                })
                .collect();
            running_checks.push(json!({
                "from_modes": low_modes,
                "to_modes": mode_count,
                "hamiltonian_max_deviation": embedding_deviation(&low_h, &running_h, &embedded),
            }));
        }
        previous_fixed = Some((mode_count, fixed_h.clone(), number.clone(), field.clone()));
        previous_running = Some((mode_count, running_h.clone(), number.clone(), field.clone()));
        rows.push(cell_from(
            mode_count, &states, &phi, &field, &number, &fixed_h, LAMBDA, SUPPORT, None,
        ));
        let mut running_cell = cell_from(
            mode_count,
            &states,
            &phi,
            &field,
            &number,
            &running_h,
            lambda,
            mode_count,
            Some(g4_residual),
        );
        running_cell["matching_roots"] = json!(matched.roots);
        running_cell["low_branch"] = json!({
            "lambda": low_lambda,
            "coherent_energy": low_energy,
            "coherent_energy_difference": low_energy - chosen_energy,
        });
        running_rows.push(running_cell);
        let mut alt_cell = cell_from(
            mode_count,
            &states,
            &phi,
            &alt_field,
            &number,
            &alt_h,
            alt_lambda,
            mode_count,
            Some(alt_g4_residual),
        );
        alt_cell["matching_roots"] = json!(alt_matched.roots);
        alt_rows.push(alt_cell);
        if std::env::var_os("N12_PROGRESS").is_some() {
            eprintln!("main cell done M={mode_count} lambda={lambda}");
        }
    }

    let worst_embedding = checks
        .iter()
        .flat_map(|check| {
            [
                check["hamiltonian_max_deviation"].as_f64().unwrap(),
                check["number_max_deviation"].as_f64().unwrap(),
                check["smeared_field_max_deviation"].as_f64().unwrap(),
            ]
        })
        .fold(0.0f64, f64::max);
    let worst_running_embedding = running_checks
        .iter()
        .map(|check| check["hamiltonian_max_deviation"].as_f64().unwrap())
        .fold(0.0f64, f64::max);
    let coherent_number_spread = spread(&rows, "coherent", "number");
    let coherent_field_spread = spread(&rows, "coherent", "smeared_field_mean");
    let squeezed_number_spread = spread(&rows, "squeezed", "number");
    let squeezed_variance_spread = spread(&rows, "squeezed", "smeared_field_variance");
    let running_number_spread = spread(&running_rows, "coherent", "number");
    let running_field_spread = spread(&running_rows, "coherent", "smeared_field_mean");
    let running_variance_spread = spread(&running_rows, "squeezed", "smeared_field_variance");
    let worst_g4_residual = running_rows
        .iter()
        .map(|row| row["g4_residual"].as_f64().unwrap())
        .fold(0.0f64, f64::max);
    let lambda_spread = scalar_spread(&running_rows, "lambda");
    let worst_root_count = running_rows
        .iter()
        .map(|row| row["matching_roots"].as_array().unwrap().len())
        .max()
        .unwrap();
    let worst_alt_root_count = alt_rows
        .iter()
        .map(|row| row["matching_roots"].as_array().unwrap().len())
        .max()
        .unwrap();
    let ambiguous_rows: Vec<&Value> = running_rows
        .iter()
        .filter(|row| row["matching_roots"].as_array().unwrap().len() > 1)
        .collect();
    let low_roots: Vec<f64> = ambiguous_rows
        .iter()
        .map(|row| row["matching_roots"][0].as_f64().unwrap())
        .collect();
    let high_roots: Vec<f64> = ambiguous_rows
        .iter()
        .map(|row| {
            let roots = row["matching_roots"].as_array().unwrap();
            roots.last().unwrap().as_f64().unwrap()
        })
        .collect();
    let low_branch_spread = low_roots.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - low_roots.iter().copied().fold(f64::INFINITY, f64::min);
    let high_branch_spread = high_roots.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - high_roots.iter().copied().fold(f64::INFINITY, f64::min);
    let min_branch_separation = low_roots
        .iter()
        .zip(&high_roots)
        .map(|(low, high)| high - low)
        .fold(f64::INFINITY, f64::min);
    let max_branch_energy_difference = running_rows
        .iter()
        .map(|row| {
            row["low_branch"]["coherent_energy_difference"]
                .as_f64()
                .unwrap()
                .abs()
        })
        .fold(0.0f64, f64::max);
    let cutoff_3_or_5_single_root = cutoff_rows
        .iter()
        .filter(|row| [3, 5].contains(&row["occupation_cutoff"].as_u64().unwrap()))
        .all(|row| row["roots"].as_array().unwrap().len() == 1);
    let cutoff_6_rows: Vec<_> = cutoff_rows
        .iter()
        .filter(|row| row["occupation_cutoff"].as_u64().unwrap() == 6)
        .collect();
    let cutoff_6_two_roots = cutoff_6_rows
        .iter()
        .all(|row| row["roots"].as_array().unwrap().len() == 2);
    let cutoff_6_high_branch_shift = cutoff_6_rows.iter().all(|row| {
        let roots = row["roots"].as_array().unwrap();
        (roots[1].as_f64().unwrap() - 0.0497598687).abs() > 1e-2
    });
    let branch_structure_cutoff_dependent =
        cutoff_3_or_5_single_root && cutoff_6_two_roots && cutoff_6_high_branch_shift;
    let fixed_cutoff_spread = (SUPPORT..=MAX_MODE)
        .map(|mode_count| {
            let values: Vec<f64> = fixed_coupling_rows
                .iter()
                .filter(|row| row["modes"].as_u64().unwrap() as usize == mode_count)
                .map(|row| row["g4"].as_f64().unwrap())
                .collect();
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().cloned().fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let fixed_cutoff4_values: Vec<f64> = fixed_coupling_rows
        .iter()
        .filter(|row| row["occupation_cutoff"].as_u64().unwrap() == 4)
        .map(|row| row["g4"].as_f64().unwrap())
        .collect();
    let fixed_cutoff4_drift = fixed_cutoff4_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - fixed_cutoff4_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    let fixed_coupling_cutoff_dependent = fixed_cutoff_spread > 5e-2;
    let fixed_coupling_drifts_at_cutoff4 = fixed_cutoff4_drift > 1e-3;

    let cutoff_4_consistent = cutoff_rows
        .iter()
        .filter(|row| row["occupation_cutoff"].as_u64().unwrap() == 4)
        .all(|row| {
            let modes = row["modes"].as_u64().unwrap();
            let main = running_rows
                .iter()
                .find(|main| main["modes"].as_u64().unwrap() == modes)
                .unwrap();
            let roots = row["roots"].as_array().unwrap();
            let main_roots = main["matching_roots"].as_array().unwrap();
            roots.len() == main_roots.len()
                && roots
                    .iter()
                    .zip(main_roots)
                    .all(|(a, b)| (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-6)
        });
    let chosen_root_is_reported = running_rows.iter().all(|row| {
        let lambda = row["lambda"].as_f64().unwrap();
        row["matching_roots"]
            .as_array()
            .unwrap()
            .iter()
            .any(|root| (root.as_f64().unwrap() - lambda).abs() < 1e-6)
    });
    let worst_alt_residual = alt_rows
        .iter()
        .map(|row| row["g4_residual"].as_f64().unwrap())
        .fold(0.0f64, f64::max);
    let max_lambda_difference = running_rows
        .iter()
        .zip(&alt_rows)
        .map(|(a, b)| (a["lambda"].as_f64().unwrap() - b["lambda"].as_f64().unwrap()).abs())
        .fold(0.0f64, f64::max);
    let max_number_difference = running_rows
        .iter()
        .zip(&alt_rows)
        .map(|(a, b)| {
            (a["coherent"]["number"].as_f64().unwrap() - b["coherent"]["number"].as_f64().unwrap())
                .abs()
        })
        .fold(0.0f64, f64::max);
    let max_energy_difference = running_rows
        .iter()
        .zip(&alt_rows)
        .map(|(a, b)| {
            (a["coherent"]["energy"].as_f64().unwrap() - b["coherent"]["energy"].as_f64().unwrap())
                .abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst_embedding < 1e-12,
        "fixed family embedding is not exact: {worst_embedding}"
    );
    assert!(
        coherent_number_spread < 1e-12
            && coherent_field_spread < 1e-12
            && squeezed_number_spread < 1e-12
            && squeezed_variance_spread < 1e-12,
        "fixed observable family is not embedding-stable"
    );
    assert!(
        worst_g4_residual <= 1e-9 * target_g4.abs().max(1.0),
        "renormalization condition not met: residual {worst_g4_residual}"
    );
    assert!(
        worst_alt_residual <= 1e-9 * alt_target_g4.abs().max(1.0),
        "alternate renormalization condition not met: residual {worst_alt_residual}"
    );
    assert!(
        max_number_difference < 1e-12,
        "low-mode number depends on the smearing normalisation: {max_number_difference}"
    );
    assert!(
        chosen_root_is_reported && worst_root_count >= 1 && worst_alt_root_count >= 1,
        "matching roots were not reported consistently"
    );
    assert!(
        fixed_coupling_cutoff_dependent && fixed_coupling_drifts_at_cutoff4,
        "G4 at fixed lambda does not show the expected truncation drift"
    );
    assert!(
        cutoff_4_consistent,
        "occupation-cutoff scan disagrees with the main run at cutoff 4"
    );
    assert!(
        branch_structure_cutoff_dependent,
        "branch count or high-branch location is not cutoff-dependent"
    );
    assert!(
        !ambiguous_rows.is_empty()
            && min_branch_separation > 1e-3
            && low_branch_spread < 1e-3
            && high_branch_spread < 1e-3
            && max_branch_energy_difference > 1e-3,
        "two-root branch structure was not resolved or has no observable effect"
    );

    let out = json!({
        "experiment": "N12 finite-mode embedding pilot",
        "fixed_family": {
            "dispersion": "omega_k = sqrt(m^2 + k^2), k=0..M-1",
            "quadratic": "H0 = sum_k omega_k N_k",
            "interaction": "V = lambda * sum_{k<K} (phi_k)^4, matrix-ordered on the truncated basis",
            "embedding": "E: M -> M+1 appends one vacuum mode; low-mode couplings and support K are fixed",
            "parameters": {
                "mass": MASS,
                "lambda": LAMBDA,
                "support_modes": SUPPORT,
                "max_total_occupation": MAX_TOTAL_OCCUPATION,
                "coherent_amplitude": COHERENT_AMPLITUDE,
                "squeezing": SQUEEZING,
                "kernel_scale": KERNEL_SCALE,
                "mode_range": [SUPPORT, MAX_MODE],
            },
        },
        "running_family": {
            "interaction": "V_M = lambda_M * sum_{k<M} (phi_k)^4, matrix-ordered on the truncated basis",
            "renormalization_condition": "G4_M(lambda_M) = G4_ref, where G4 = <(delta phi_s)^4> - 3<(delta phi_s)^2>^2 in the ground state and phi_s = sum_{k<4} exp(-k/2) phi_k",
            "reference": {
                "modes": SUPPORT,
                "lambda": LAMBDA,
                "g4": target_g4,
            },
            "matching": "scan on lambda in (0, 8 lambda_ref] for every sign change, bisect each bracket, choose the root nearest the previous cutoff",
            "match_iterations": MATCH_ITERATIONS,
            "match_tolerance": MATCH_TOLERANCE,
            "embedded_exactly": false,
            "note": "appended modes carry interactions, so H_M is not the restriction of H_{M+1}",
        },
        "cells": rows,
        "running_cells": running_rows,
        "embedding_checks": checks,
        "running_embedding_checks": running_checks,
        "stability": {
            "worst_embedding_deviation": worst_embedding,
            "coherent_number_spread": coherent_number_spread,
            "coherent_field_mean_spread": coherent_field_spread,
            "squeezed_number_spread": squeezed_number_spread,
            "squeezed_field_variance_spread": squeezed_variance_spread,
            "exact_by_construction": true,
        },
        "running_stability": {
            "worst_hamiltonian_embedding_deviation": worst_running_embedding,
            "coherent_number_spread": running_number_spread,
            "coherent_field_mean_spread": running_field_spread,
            "squeezed_field_variance_spread": running_variance_spread,
            "lambda_spread": lambda_spread,
            "worst_g4_residual": worst_g4_residual,
            "worst_root_count": worst_root_count,
            "exact_by_construction": false,
        },
        "scheme_dependence": {
            "alternate_kernel_scale": ALT_KERNEL_SCALE,
            "reference_g4": alt_target_g4,
            "cells": alt_rows,
            "worst_g4_residual": worst_alt_residual,
            "max_lambda_difference": max_lambda_difference,
            "max_coherent_number_difference": max_number_difference,
            "max_coherent_energy_difference": max_energy_difference,
            "worst_root_count": worst_alt_root_count,
            "note": "same V_M and states, different smeared four-point normalisation; a robustness check, not a second theory",
        },
        "branch_ambiguity": {
            "root_counts": ambiguous_rows
                .iter()
                .map(|row| row["matching_roots"].as_array().unwrap().len())
                .collect::<Vec<_>>(),
            "low_branch": low_roots,
            "high_branch": high_roots,
            "low_branch_spread": low_branch_spread,
            "high_branch_spread": high_branch_spread,
            "min_separation": min_branch_separation,
            "max_coherent_energy_difference": max_branch_energy_difference,
            "low_branch_energies": ambiguous_rows
                .iter()
                .map(|row| row["low_branch"]["coherent_energy"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            "high_branch_energies": ambiguous_rows
                .iter()
                .map(|row| row["coherent"]["energy"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            "note": "the four-point condition alone does not fix lambda; the reported trajectory follows the branch nearest the previous cutoff",
        },
        "occupation_sensitivity": {
            "cells": cutoff_rows,
            "cutoff_4_matches_main": cutoff_4_consistent,
            "cutoff_6_two_roots": cutoff_6_two_roots,
            "cutoff_6_high_branch_shift": cutoff_6_high_branch_shift,
            "note": "same four-point condition at total-occupation cutoffs 2, 3, 4, 5 and 6; a truncation diagnostic, not a convergence claim",
        },
        "fixed_coupling_sensitivity": {
            "cells": fixed_coupling_rows,
            "max_cutoff_spread": fixed_cutoff_spread,
            "cutoff_4_mode_drift": fixed_cutoff4_drift,
            "cutoff_dependent": fixed_coupling_cutoff_dependent,
            "note": "G4 at lambda=0.05 without matching; shows whether the basis is converged before root selection",
        },
        "claims": {
            "defined_finite_mode_family": true,
            "interacting": true,
            "fixed_embedding_exact": true,
            "running_coupling_trajectory": true,
            "renormalization_condition_satisfied": true,
            "smearing_normalization_dependent": true,
            "matching_branch_not_unique": true,
            "branch_structure_cutoff_specific": true,
            "running_embedding_exact": false,
            "sm_regulator": false,
            "continuum_limit": false,
            "cutoff_uniform_theorem": false,
            "missing": "a cutoff-uniform observable/estimate theorem and a physical identification of the finite-mode normalization with a field-theory scheme",
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
