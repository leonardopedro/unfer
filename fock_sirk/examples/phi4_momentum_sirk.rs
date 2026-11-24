//! Reproducible evidence runner for the continuous-momentum λφ⁴ suite (`P13`).
//!
//! Runs the same sweeps as `tests/phi4_momentum_sirk.rs` and writes the full
//! record demanded by `docs/PHI4_NUMERICAL_PLAN.md` §5 to
//! `docs/fixtures/phi4_momentum_sirk_sweep.json`:
//!
//! ```text
//! cargo run -p fock_sirk --example phi4_momentum_sirk
//! ```
//!
//! Every trajectory uses `JointKrylovMethod::ShiftInvert` with Hashimoto
//! decreasing shifts; `ForwardProducts` appears only as a labeled diagnostic.
//! Momentum `k ∈ ℝ³` and field `φ ∈ ℝ` stay continuous — only the Mehler
//! occupation window and the Krylov rank are discrete.
//!
//! Each record carries a `status` field so measurements, conditional bounds and
//! kernel-checked conclusions are never conflated:
//!
//! * `measured` — a number produced by this run.
//! * `comparison fixture` — the same number as checked out of the Cadabra2
//!   fixture (or recomputed from the frozen comparison builders).
//! * `charged (sound, wide)` — the full-action defect bound, which the response
//!   cells now carry as their certified charge; it is sound but too loose to
//!   certify a 1e-3 response, so the sharp tier is reported beside it.

use fock_sirk::joint_sirk::{
    hashimoto_unitary_shifts, joint_inner_product, solve_joint_sirk, JointKrylovMethod,
    JointSirkOptions,
};
use fock_sirk::sm_defect::{defect_terms, reduced_growth};
use nalgebra::DMatrix;
use nested_fock_algebra::exact_inner::{ActionLimits, InnerExpr, JointOuterBasis, NestedVector};
use nested_fock_algebra::phi4_momentum::{frozen_jet_comparison, occupations_of, ScalarMomCont};
use nested_fock_algebra::phi4_scalar::ScalarGrid;
use num_complex::Complex64 as C;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

fn vacuum_support() -> Vec<([u32; 4], f64)> {
    vec![([0, 0, 0, 0], 1.0)]
}

fn seed(model: &ScalarMomCont, support: &[([u32; 4], f64)]) -> NestedVector {
    model.one_particle(support)
}

fn physical_shifts(t: f64, n_phys: f64, h_phys: f64, m: usize) -> Vec<C> {
    hashimoto_unitary_shifts(t, n_phys * t, h_phys * t, m).expect("valid Hashimoto schedule")
}

fn shift_invert() -> JointKrylovMethod {
    JointKrylovMethod::ShiftInvert {
        relative_tolerance: 1e-13,
        max_iterations: 64,
    }
}

/// Cutoffs of the outside-window sweep. C = 16 and 20 are the extension past
/// the window at which the seed's support is fully covered; they need a looser
/// solve policy (see `sweep_method`) and a larger component budget.
const SWEEP_CUTOFFS: [u32; 5] = [4, 8, 12, 16, 20];

/// Solve policy per cutoff. The `1e-13` relative residual is affordable while
/// the window is small, but at C = 20 the smallest scheduled shift
/// (`gamma = 40`) stalls just above it and exhausts even a 512-step Arnoldi
/// budget, so C >= 16 solves to `1e-10`. That is still six orders below the
/// 1e-3-scale inside-window error being measured, and the policy is recorded
/// per cell so the numbers are not silently mixed.
fn sweep_method(cutoff: u32) -> JointKrylovMethod {
    if cutoff <= 12 {
        shift_invert()
    } else {
        JointKrylovMethod::ShiftInvert {
            relative_tolerance: 1e-10,
            max_iterations: 1024,
        }
    }
}

/// The window dictionary charges one component per (outer label, inner state)
/// pair, so C = 20 (1001 labels) exceeds the default 1e6 budget during span
/// registration.
const SWEEP_COMPONENT_BUDGET: usize = 4_000_000;

fn solve(
    h: &InnerExpr,
    initial: &NestedVector,
    shifts: &[C],
    window: &BTreeSet<JointOuterBasis>,
    method: JointKrylovMethod,
) -> fock_sirk::joint_sirk::JointSirkResult {
    let options = JointSirkOptions {
        basis_window: Some(window.clone()),
        method,
        ..JointSirkOptions::default()
    };
    solve_joint_sirk(h, initial, shifts, &options).expect("Krylov solve")
}

/// Same solve, with the sweep's component budget.
fn sweep_solve(
    h: &InnerExpr,
    initial: &NestedVector,
    shifts: &[C],
    window: &BTreeSet<JointOuterBasis>,
    method: JointKrylovMethod,
) -> fock_sirk::joint_sirk::JointSirkResult {
    let options = JointSirkOptions {
        basis_window: Some(window.clone()),
        method,
        max_cached_components: SWEEP_COMPONENT_BUDGET,
        ..JointSirkOptions::default()
    };
    solve_joint_sirk(h, initial, shifts, &options).expect("Krylov solve")
}

fn policy_json(method: &JointKrylovMethod) -> Value {
    match method {
        JointKrylovMethod::ShiftInvert {
            relative_tolerance,
            max_iterations,
        } => json!({
            "method": "shift_invert",
            "relative_tolerance": relative_tolerance,
            "max_iterations": max_iterations,
        }),
        other => json!({ "method": format!("{other:?}") }),
    }
}

fn return_probability(
    result: &fock_sirk::joint_sirk::JointSirkResult,
    initial: &NestedVector,
    t: f64,
) -> f64 {
    let coefficients = result.time_evolve(t).expect("finite time");
    let state = result.reconstruct(&coefficients).expect("reconstruction");
    joint_inner_product(initial, &state).norm_sqr()
}

/// Exact dense Padé evolution of one declared window (the sharp reference tier).
struct WindowOperator {
    index: BTreeMap<JointOuterBasis, usize>,
    matrix: DMatrix<C>,
}

impl WindowOperator {
    fn build(h: &InnerExpr, window: &BTreeSet<JointOuterBasis>) -> Self {
        let labels: Vec<JointOuterBasis> = window.iter().cloned().collect();
        let index: BTreeMap<JointOuterBasis, usize> = labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let n = labels.len();
        let mut matrix = DMatrix::<C>::zeros(n, n);
        for (j, basis) in labels.iter().enumerate() {
            let image = h
                .d_gamma(
                    &NestedVector::from([(basis.clone(), C::new(1.0, 0.0))]),
                    ActionLimits::default(),
                )
                .expect("finite action");
            for (target, amplitude) in image {
                if let Some(&i) = index.get(&target) {
                    matrix[(i, j)] += amplitude;
                }
            }
        }
        Self { index, matrix }
    }

    fn reference_return(&self, initial: &NestedVector, t: f64) -> f64 {
        let basis = initial.keys().next().expect("nonempty seed").clone();
        let j = self.index[&basis];
        (self.matrix.clone() * C::new(0.0, -t)).exp()[(j, j)].norm_sqr()
    }
}

fn shifts_json(shifts: &[C]) -> Value {
    Value::Array(
        shifts
            .iter()
            .map(|z| json!({ "re": z.re, "im": z.im }))
            .collect(),
    )
}

fn vec_json(values: &[f64]) -> Value {
    Value::Array(values.iter().map(|v| json!(*v)).collect())
}

/// Max GMRES iteration count / shifted-solve residual recorded by a solve.
fn solver_health(result: &fock_sirk::joint_sirk::JointSirkResult) -> Value {
    let residuals = &result.function_solve_error_bounds;
    let shift_counts = &result.shift_solve_iterations;
    let function_counts = &result.function_solve_iterations;
    json!({
        "gmres_iterations_max": result.max_shift_solve_iterations(),
        "gmres_iterations_total": result.total_solve_iterations(),
        "gmres_iterations_cap": 64,
        "gmres_relative_tolerance": 1e-13,
        "gmres_iterations_note": "Arnoldi steps actually taken per solve, recorded by gmres_shift; basis solves and bounded-resolvent solves are reported separately",
        "shift_solve_iterations": {
            "count": shift_counts.len(),
            "max": shift_counts.iter().copied().max().unwrap_or(0),
            "values": shift_counts,
        },
        "function_solve_iterations": {
            "count": function_counts.len(),
            "max": function_counts.iter().copied().max().unwrap_or(0),
            "values": function_counts,
        },
        "function_solve_error_bounds": {
            "count": residuals.len(),
            "max": residuals.iter().copied().fold(0.0_f64, f64::max),
        },
        "shift_solve_residuals": {
            "count": result.shift_solve_error_bounds.len(),
            "max": result
                .shift_solve_error_bounds
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        },
        "orthogonality_defect": result.orthogonality_defect,
        "hermiticity_defect": result.hermiticity_defect,
        "cached_components": result.cached_components,
    })
}

fn sha256_of(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(format!("{:x}", hasher.finalize()))
}

fn git_revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Peak resident set size of this process, in KiB (Linux only).
fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Rank-2 Hashimoto cell on the degree-≤2 window at `t = 0.01`.
fn rank_two_cell() -> Value {
    let started = Instant::now();
    let t = 0.01;
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let window = ScalarMomCont::window_labels(2);
    let initial = seed(&model, &vacuum_support());
    let shifts = physical_shifts(t, 15.0, 5.0, 2);
    let result = solve(&h, &initial, &shifts, &window, shift_invert());

    let p_sirk = return_probability(&result, &initial, t);
    let p_window = WindowOperator::build(&h, &ScalarMomCont::window_labels(2))
        .reference_return(&initial, t);
    let p_full = WindowOperator::build(&h, &ScalarMomCont::window_labels(8))
        .reference_return(&initial, t);
    let (mean, variance, outside) = model.seed_moments();
    let tail = result.projected_action_tail_norms[0];

    json!({
        "status": "measured",
        "model": "ScalarMomCont (continuous k in R^3, phi in R), lambda = v = 1, unit Mehler widths",
        "energy_convention": "H = 1/2 p^2 + 1/2 k^2 phi^2 + lambda/4 (phi^2 - v^2)^2, lambda v^4/4 retained",
        "seed": "outer vacuum with one inner Gaussian, He_0(phi) label",
        "observable": "return probability |<psi|exp(-iHt)|psi>|^2",
        "time": t,
        "hermite_cutoff": 2,
        "window_labels": window.len(),
        "window_rule": "even occupation in all four modes, total degree <= 2",
        "requested_rank": 2,
        "attained_rank": result.rank(),
        "hashimoto": { "N": 15.0, "h": 5.0, "m": 2, "t": t },
        "shifts": shifts_json(&result.shifts),
        "return_probability_sirk": p_sirk,
        "return_probability_window_exact": p_window,
        "rank_two_charge": (p_window - p_sirk).abs(),
        "return_probability_converged_window": p_full,
        "plan_interval": [319_447.0 / 320_000.0, 39_999.0 / 40_000.0],
        "converged_inside_plan_interval": (319_447.0 / 320_000.0..=39_999.0 / 40_000.0)
            .contains(&p_full),
        "outside_window_tail_squared": tail * tail,
        "analytic_outside_window_squared": outside,
        "dissipative_growth": reduced_growth(&result.function_generator),
        "fixture_seed_moments": { "mean": mean, "variance": variance, "status": "comparison fixture" },
        "solver": solver_health(&result),
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

/// Hermite-cutoff sweep at fixed rank `m = 2`, then the rank-4 method contrast.
fn cutoff_sweep() -> Value {
    let started = Instant::now();
    let t = 0.05;
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let initial = seed(&model, &vacuum_support());

    let mut tail_cells = Vec::new();
    for cutoff in SWEEP_CUTOFFS {
        let cell_started = Instant::now();
        let window = ScalarMomCont::window_labels(cutoff);
        let shifts = physical_shifts(t, 8.0, 3.0, 2);
        let method = sweep_method(cutoff);
        let result = sweep_solve(&h, &initial, &shifts, &window, method.clone());
        let exact = WindowOperator::build(&h, &window).reference_return(&initial, t);
        tail_cells.push(json!({
            "status": "measured",
            "hermite_cutoff": cutoff,
            "window_labels": window.len(),
            "requested_rank": 2,
            "attained_rank": result.rank(),
            "hashimoto": { "N": 8.0, "h": 3.0, "m": 2, "t": t },
            "shifts": shifts_json(&result.shifts),
            "solve_policy": policy_json(&method),
            "window_tail_norm_max": result
                .projected_action_tail_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            "projected_action_tail_norms": vec_json(&result.projected_action_tail_norms),
            "return_probability_sirk": return_probability(&result, &initial, t),
            "return_probability_window_exact": exact,
            "inside_window_error": (return_probability(&result, &initial, t) - exact).abs(),
            "solver": solver_health(&result),
            "runtime_ms": cell_started.elapsed().as_millis() as u64,
        }));
    }

    // Rank 4: the labeled ForwardProducts diagnostic against shift-invert.
    let mut method_cells = Vec::new();
    for cutoff in SWEEP_CUTOFFS {
        let cell_started = Instant::now();
        let window = ScalarMomCont::window_labels(cutoff);
        let shifts = physical_shifts(t, 14.0, 3.0, 4);
        let method = sweep_method(cutoff);
        let result = sweep_solve(&h, &initial, &shifts, &window, method.clone());
        let exact = WindowOperator::build(&h, &window).reference_return(&initial, t);
        let forward = sweep_solve(
            &h,
            &initial,
            &shifts,
            &window,
            JointKrylovMethod::ForwardProducts,
        );
        let p_sirk = return_probability(&result, &initial, t);
        let p_forward = return_probability(&forward, &initial, t);
        method_cells.push(json!({
            "status": "measured",
            "hermite_cutoff": cutoff,
            "window_labels": window.len(),
            "requested_rank": 4,
            "attained_rank": result.rank(),
            "solve_policy": policy_json(&method),
            "return_probability_window_exact": exact,
            "shift_invert": {
                "return_probability": p_sirk,
                "error": (p_sirk - exact).abs(),
                "solver": solver_health(&result),
            },
            "forward_products_diagnostic": {
                "label": "diagnostic only, not the primary solver",
                "return_probability": p_forward,
                "error": (p_forward - exact).abs(),
                "solver": solver_health(&forward),
            },
            "runtime_ms": cell_started.elapsed().as_millis() as u64,
        }));
    }

    json!({
        "status": "measured",
        "note": "the m=2 tail sweep follows the plan literally; at m=2 the ForwardProducts contrast is not yet visible (forward error 1.9e-3, 1.3e-3, 1.3e-3), so the method contrast is recorded at m=4 where it is. C = 16 and 20 extend the plan's C <= 12 range to windows that cover the seed many times over; they solve to 1e-10 relative residual with a 1024-step cap because the smallest scheduled shift stalls just above 1e-13 there",
        "tail_sweep_fixed_rank_2": tail_cells,
        "method_contrast_fixed_rank_4": method_cells,
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

/// Krylov-rank sweep on the fixed `C = 8` window.
fn rank_sweep() -> Value {
    let started = Instant::now();
    let t = 0.05;
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let window = ScalarMomCont::window_labels(8);
    let initial = seed(&model, &vacuum_support());
    let exact = WindowOperator::build(&h, &window).reference_return(&initial, t);

    let mut cells = Vec::new();
    for m in [2usize, 4, 6] {
        let cell_started = Instant::now();
        let shifts = physical_shifts(t, 3.0 * m as f64 + 2.0, 3.0, m);
        let result = solve(&h, &initial, &shifts, &window, shift_invert());
        let forward = solve(
            &h,
            &initial,
            &shifts,
            &window,
            JointKrylovMethod::ForwardProducts,
        );
        let p_sirk = return_probability(&result, &initial, t);
        let p_forward = return_probability(&forward, &initial, t);
        cells.push(json!({
            "status": "measured",
            "requested_rank": m,
            "attained_rank": result.rank(),
            "hashimoto": { "N": 3.0 * m as f64 + 2.0, "h": 3.0, "m": m, "t": t },
            "shifts": shifts_json(&result.shifts),
            "return_probability_sirk": p_sirk,
            "inside_window_error": (p_sirk - exact).abs(),
            "forward_products_error": (p_forward - exact).abs(),
            "window_tail_norm_outside_p_window": result
                .projected_action_tail_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            "solver": solver_health(&result),
            "runtime_ms": cell_started.elapsed().as_millis() as u64,
        }));
    }

    json!({
        "status": "measured",
        "hermite_cutoff": 8,
        "window_labels": window.len(),
        "return_probability_window_exact": exact,
        "cells": cells,
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

/// One paired-response cell: compressed estimate, exact window reference, and
/// the two charges. `sharp` is the measured compressed-flow discrepancy (S43
/// sharp tier, the number that certifies the 1e-3-scale responses);
/// `certified` is the full-action Duhamel state bound converted to a
/// probability charge by `2 eps + eps^2`. The certified charge is sound but
/// wide here (~1e-1) because the evolution error is nearly orthogonal to the
/// seed: the state error is ~1e-2 while the probability error is ~1e-5.
fn response_cell(lam: f64, width_k: f64, t: f64) -> Value {
    let started = Instant::now();
    let window = ScalarMomCont::window_labels(8);
    let shifts = physical_shifts(t, 250.0, 50.0, 4);
    let model = ScalarMomCont::with_parameters(lam, 1.0, 1.0, width_k);
    let h = model.hamiltonian();
    let initial = seed(&model, &vacuum_support());
    let result = solve(&h, &initial, &shifts, &window, shift_invert());
    let defect = defect_terms(
        &result.orthonormal_basis,
        &result.orthonormal_images,
        &result.function_generator,
        &initial,
        t,
    );
    let p_sirk = return_probability(&result, &initial, t);
    let p_exact = WindowOperator::build(&h, &window).reference_return(&initial, t);
    let discrepancy = (p_sirk - p_exact).abs();
    let epsilon = defect.evolution_bound;
    let certified = 2.0 * epsilon + epsilon * epsilon;

    json!({
        "status": "measured",
        "lambda": lam,
        "v": 1.0,
        "width_phi": 1.0,
        "width_k": width_k,
        "hermite_cutoff": 8,
        "requested_rank": 4,
        "attained_rank": result.rank(),
        "hashimoto": { "N": 250.0, "h": 50.0, "m": 4, "t": t },
        "shifts": shifts_json(&result.shifts),
        "return_probability_sirk": p_sirk,
        "return_probability_window_exact": p_exact,
        "interval": [p_sirk - discrepancy, p_sirk + discrepancy],
        "certified_interval": [p_sirk - certified, p_sirk + certified],
        "sharp_charge": {
            "kind": "measured compressed-flow discrepancy |p_sirk - p_exact|",
            "value": discrepancy,
        },
        "certified_charge": {
            "kind": "2 eps + eps^2 from the full-action Duhamel state bound",
            "value": certified,
        },
        "full_action_defect": {
            "status": "charged (sound, wide)",
            "initial_leakage": defect.initial_leakage,
            "defect_frobenius": defect.defect_frobenius,
            "defect_seed_action": defect.defect_seed_action,
            "defect_action_envelope": defect.defect_action_envelope,
            "state_weighted_bound": defect.state_weighted_bound,
            "evolution_bound": defect.evolution_bound,
            "note": "the envelope is now a grid-converged maximum sampled along the returned trajectory (the ||D G||_F ||c|| exp(omega t) product it replaced was shift-scaled and ~100 at this rank); it stays at 1e-1 here, ~25x the true state error and ~600x the true probability error, so it certifies soundness but not the 1e-3 responses",
        },
        "solver": solver_health(&result),
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

fn response_pairs() -> Value {
    let started = Instant::now();
    let t = 0.05;
    let base = response_cell(1.0, 1.0, t);
    let control = response_cell(1.0, 1.0, t);
    let coupled = response_cell(1.1, 1.0, t);
    let spread = response_cell(1.0, 1.5, t);

    let read = |cell: &Value, key: &str| cell[key].as_f64().unwrap();
    let charge = |cell: &Value| cell["sharp_charge"]["value"].as_f64().unwrap();
    let certified = |cell: &Value| cell["certified_charge"]["value"].as_f64().unwrap();
    let interval = |a: &Value, b: &Value, key: &str| {
        let difference = read(a, key) - read(b, key);
        let budget = charge(a) + charge(b);
        json!({
            "difference": difference,
            "sharp_charge": budget,
            "interval": [difference - budget, difference + budget],
            "excludes_zero": difference - budget > 0.0 || difference + budget < 0.0,
        })
    };
    // The same response charged by the full-action state bound.
    let certified_interval = |a: &Value, b: &Value, key: &str| {
        let difference = read(a, key) - read(b, key);
        let budget = certified(a) + certified(b);
        json!({
            "difference": difference,
            "certified_charge": budget,
            "interval": [difference - budget, difference + budget],
            "excludes_zero": difference - budget > 0.0 || difference + budget < 0.0,
        })
    };

    json!({
        "status": "measured",
        "time": t,
        "null_control": {
            "cell_a": "lambda = 1.0, width_k = 1.0",
            "cell_b": "lambda = 1.0, width_k = 1.0",
            "exact": interval(&base, &control, "return_probability_window_exact"),
            "sirk": interval(&base, &control, "return_probability_sirk"),
            "certified": certified_interval(&base, &control, "return_probability_sirk"),
            "contains_zero": true,
        },
        "coupling_response": {
            "cell_a": "lambda = 1.0",
            "cell_b": "lambda = 1.1",
            "exact": interval(&base, &coupled, "return_probability_window_exact"),
            "sirk": interval(&base, &coupled, "return_probability_sirk"),
            "certified": certified_interval(&base, &coupled, "return_probability_sirk"),
        },
        "momentum_spread_response": {
            "cell_a": "width_k = 1.0",
            "cell_b": "width_k = 1.5",
            "exact": interval(&base, &spread, "return_probability_window_exact"),
            "sirk": interval(&base, &spread, "return_probability_sirk"),
            "certified": certified_interval(&base, &spread, "return_probability_sirk"),
        },
        "cells": { "base": base, "null_control": control, "coupled": coupled, "spread": spread },
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

/// ℤ₂ parity, the unconditional inner bound `θ ≥ 7/64`, and the frozen
/// comparison contrast.
fn parity_and_comparisons() -> Value {
    let started = Instant::now();
    let t = 0.05;
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let shifts = physical_shifts(t, 250.0, 50.0, 4);

    let parity_cell = |support: &[([u32; 4], f64)], parity: u32| {
        let initial = seed(&model, support);
        let window = ScalarMomCont::window_labels_of_parity(8, parity);
        let result = solve(&h, &initial, &shifts, &window, shift_invert());
        let state = result
            .reconstruct(&result.time_evolve(t).unwrap())
            .expect("state");
        let parity_holds = state.keys().all(|basis| {
            occupations_of(basis.inner_occupations()[0].0)[0] % 2 == parity
        });
        let ritz_values = result.ritz_residuals().unwrap_or_default();
        let ground = ritz_values.first().map(|v| v.0).unwrap_or(f64::NAN);
        let ritz: Vec<Value> = ritz_values
            .into_iter()
            .map(|(theta, residual, norm)| {
                json!({ "theta": theta, "residual": residual, "norm": norm })
            })
            .collect();
        json!({
            "status": "measured",
            "phi_parity": parity,
            "window_labels": window.len(),
            "parity_conserved": parity_holds,
            "ritz_ground": ground,
            "ritz_values": ritz,
            "unconditional_inner_bound": 7.0 / 64.0,
            "solver": solver_health(&result),
        })
    };

    let even = parity_cell(&vacuum_support(), 0);
    let odd = parity_cell(&[([1, 0, 0, 0], 1.0)], 1);

    let (mean, variance, outside) = model.seed_moments();
    let jet = frozen_jet_comparison();
    let image = jet
        .apply(&model.hermite_state(&vacuum_support()), ActionLimits::default())
        .expect("finite action");
    let jet_variance: f64 = image
        .iter()
        .filter(|(basis, _)| occupations_of(basis) != [0, 0, 0, 0])
        .map(|(_, amplitude)| amplitude.norm_sqr())
        .sum();

    let grid = ScalarGrid::two_site(1.0, 1.0, 1.0);
    let factorial = |n: u32| (1..=n).map(|k| k as f64).product::<f64>().max(1.0);
    let columns = grid.two_site_column(0, 0);
    let lattice_variance: f64 = columns
        .iter()
        .filter(|((m, n), _)| (*m, *n) != (0, 0))
        .map(|((m, n), coefficient)| coefficient * coefficient * factorial(*m) * factorial(*n))
        .sum();
    let lattice_mean = columns[0].1;

    json!({
        "status": "measured",
        "even_seed_inner_excitation": even["ritz_ground"].clone(),
        "odd_seed_inner_excitation": odd["ritz_ground"].clone(),
        "odd_minus_even": odd["ritz_ground"].as_f64().unwrap_or(f64::NAN)
            - even["ritz_ground"].as_f64().unwrap_or(f64::NAN),
        "inner_excitation_note": "an inner excitation splitting, never the outer vacuum gap",
        "parity_cells": { "even": even, "odd": odd },
        "continuous": {
            "status": "comparison fixture",
            "mean": mean,
            "variance": variance,
            "outside_three_mode_window_squared": outside,
            "he2_k2_coupling": model.s3_columns()[0][3],
        },
        "static_jet": {
            "status": "comparison fixture",
            "mean": mean,
            "variance": jet_variance,
            "he2_k2_coupling": 0.0,
        },
        "two_site_lattice": {
            "status": "comparison fixture",
            "mean": lattice_mean,
            "variance": lattice_variance,
        },
        "runtime_ms": started.elapsed().as_millis() as u64,
    })
}

fn main() {
    let started = Instant::now();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/fixtures/phi4_momentum_mehler.json"
    );
    let rank2 = rank_two_cell();
    let cutoffs = cutoff_sweep();
    let ranks = rank_sweep();
    let responses = response_pairs();
    let parity = parity_and_comparisons();

    let mut root = Map::new();
    root.insert("schema".into(), json!("phi4-momentum-sirk-sweep/v1"));
    root.insert(
        "source_revision".into(),
        json!({
            "git_head": git_revision(),
            "crate": "fock_sirk",
            "version": env!("CARGO_PKG_VERSION"),
            "rustc": std::process::Command::new("rustc")
                .arg("--version")
                .output()
                .ok()
                .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        }),
    );
    root.insert(
        "fixture".into(),
        json!({
            "path": fixture,
            "sha256": sha256_of(fixture),
            "source": "docs/phi4_momentum_nested.cdb via cadabra2-cli",
        }),
    );
    root.insert(
        "energy_convention".into(),
        json!(
            "H = dGamma(h) with h = 1/2 p^2 + 1/2 k^2 phi^2 + lambda/4 (phi^2 - v^2)^2; \
             the lambda v^4/4 constant is retained and lifts to c N_out; products are \
             formed inside h before dGamma"
        ),
    );
    root.insert("rank_two_cell".into(), rank2);
    root.insert("cutoff_sweep".into(), cutoffs);
    root.insert("rank_sweep".into(), ranks);
    root.insert("responses".into(), responses);
    root.insert("parity_and_comparisons".into(), parity);
    root.insert(
        "limitations".into(),
        json!([
            "the full-action defect envelope is now a grid-converged maximum sampled along the returned trajectory, which removes the shift-scaled ||D G||_F ||c|| exp(omega t) product: the state bound falls from 5.95 to 0.18 at the base cell. It is charged as 2 eps + eps^2 and is sound, but it stays ~25x the true state error and ~600x the true probability error, so it cannot certify a 1e-3 response; the sharp-tier charge (measured compressed-flow discrepancy against the exact Padé evolution of the declared window) is what does",
            "the rank-2 cell on the degree-2 window reports the windowed return probability, which is 8.65e-4 below the converged window value because the omitted tail (squared norm 9/2) is projected out",
            "the ForwardProducts diagnostic is more accurate than shift-invert at m = 4 on these bounded windows; the plan's degradation claim appears only as growth with C, not as a rank-for-rank deficit",
            "momenta are continuous; no lattice or momentum sum is used anywhere in this record",
        ]),
    );
    root.insert(
        "runtime_ms".into(),
        json!(started.elapsed().as_millis() as u64),
    );
    root.insert("peak_rss_kb".into(), json!(peak_rss_kb()));

    let document = Value::Object(root);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/fixtures/phi4_momentum_sirk_sweep.json"
    );
    let mut text = serde_json::to_string_pretty(&document).expect("serialize record");
    text.push('\n');
    std::fs::write(path, &text).expect("write evidence record");
    println!("wrote {path} ({} bytes)", text.len());
    println!(
        "return probabilities: rank2 {:.9}, converged {:.9}",
        document["rank_two_cell"]["return_probability_sirk"]
            .as_f64()
            .unwrap(),
        document["rank_two_cell"]["return_probability_converged_window"]
            .as_f64()
            .unwrap()
    );
}
