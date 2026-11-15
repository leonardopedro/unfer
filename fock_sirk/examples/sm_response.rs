//! N11 scoped response observable: the final occupation of the seed up-quark
//! mode after one full-operator restart step, for a gauge-coupling quench.
//!
//! Fixed operator family, fixed state, fixed time `t=0.01` and one admissible
//! shift rule. The N8 defect bound is carried separately from the finite window
//! tail and from unenclosed rounding. Controls: a zero quench (null) and an
//! electron-Yukawa quench (spectator at tree level for this seed).

use fock_sirk::joint_sirk::*;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, InnerLadder, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use std::collections::BTreeSet;

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
        max_components: 1_000_000,
        max_work: 5_000_000_000,
    }
}
fn fermions() -> [u32; 2] {
    [
        matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
        matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
    ]
}
fn seed() -> NestedVector {
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}
fn window_ground() -> &'static NestedVector {
    static G: std::sync::OnceLock<NestedVector> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let model = SmHamiltonian::new(parameters()).unwrap();
        let h = model.expression();
        let initial = seed();
        let image = h.d_gamma(&initial, limits()).unwrap();
        let mut window: BTreeSet<_> = project_occupation(&image, 1).0.into_keys().collect();
        window.extend(initial.keys().cloned());
        let basis: Vec<_> = window.into_iter().collect();
        let n = basis.len();
        let mut matrix = nalgebra::DMatrix::<C>::zeros(n, n);
        for (j, b) in basis.iter().enumerate() {
            let vector = NestedVector::from([(b.clone(), C::new(1.0, 0.0))]);
            let image = h.d_gamma(&vector, limits()).unwrap();
            for (i, target) in basis.iter().enumerate() {
                matrix[(i, j)] = image.get(target).copied().unwrap_or_default();
            }
        }
        let spectrum = matrix.symmetric_eigen();
        let index = spectrum
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        basis
            .into_iter()
            .zip(spectrum.eigenvectors.column(index).iter().copied())
            .filter(|(_, a)| *a != C::new(0.0, 0.0))
            .collect()
    })
}

fn prepared(name: &str) -> NestedVector {
    let mut out = seed();
    if name == "ground" {
        return window_ground().clone();
    }
    match name {
        "seed" => {}
        "up" => {
            let up = matter_mode(Species::Up, 0, 0, Chirality::Left, 0);
            let inner = B::new([], [up]).unwrap();
            out = NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))]);
        }
        "up_down" => {
            let inner = B::new(
                [],
                [
                    matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
                    matter_mode(Species::Down, 0, 0, Chirality::Left, 0),
                ],
            )
            .unwrap();
            out = NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))]);
        }
        "seed_gluon" => {
            let inner = B::new([(coord::g(0, 0), 1)], fermions()).unwrap();
            out.insert(O::graded([(inner, 1)]).unwrap(), C::new(0.1, 0.0));
        }
        "seed_phi" => {
            let inner = B::new([(coord::phi(0), 1)], fermions()).unwrap();
            out.insert(O::graded([(inner, 1)]).unwrap(), C::new(0.1, 0.0));
        }
        "seed_squeezed" => {
            let mode = coord::g(0, 0);
            let ell: f64 = std::env::var("SM_RESPONSE_ELL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.25);
            let r = ell.ln();
            let t = r.tanh();
            let mut coeff = 1.0 / r.cosh().sqrt();
            out = NestedVector::new();
            for m in 0..=6u32 {
                let inner = B::new([(mode, 2 * m)], fermions()).unwrap();
                out.insert(O::graded([(inner, 1)]).unwrap(), C::new(coeff, 0.0));
                let mf = (m + 1) as f64;
                coeff *= t * ((2.0 * mf - 1.0) * 2.0 * mf).sqrt() / (2.0 * mf);
            }
        }
        "seed_displaced" => {
            let amp: f64 = std::env::var("SM_RESPONSE_DISP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.5);
            let inner = B::new([(coord::g(0, 0), 1)], fermions()).unwrap();
            out.insert(O::graded([(inner, 1)]).unwrap(), C::new(amp, 0.0));
        }
        other => panic!("unknown SM_RESPONSE_STATE {other}"),
    }
    let n = joint_norm(&out);
    for amplitude in out.values_mut() {
        *amplitude /= n;
    }
    out
}

fn occupation() -> InnerExpr {
    let up = matter_mode(Species::Up, 0, 0, Chirality::Left, 0);
    InnerExpr::Product(vec![
        InnerExpr::Ladder(InnerLadder::FermionCreate(up)),
        InnerExpr::Ladder(InnerLadder::FermionAnnihilate(up)),
    ])
}

/// Directly driven channel of the strong-coupling quench: one gluon mode.
fn gauge_occupation() -> InnerExpr {
    let mode = coord::g(0, 0);
    InnerExpr::Product(vec![
        InnerExpr::Ladder(InnerLadder::BosonCreate(mode)),
        InnerExpr::Ladder(InnerLadder::BosonAnnihilate(mode)),
    ])
}

fn run(delta_gs: f64, delta_ye: f64, time: f64) -> serde_json::Value {
    let mut p = parameters();
    p.gs *= 1.0 + delta_gs;
    p.ye[2][2] *= 1.0 + delta_ye;
    let model = SmHamiltonian::new(p).unwrap();
    let h = model.expression();
    let state_name = std::env::var("SM_RESPONSE_STATE").unwrap_or_else(|_| "seed".into());
    let initial = prepared(&state_name);
    let depth: usize = std::env::var("SM_RESPONSE_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let mut window: BTreeSet<_> = initial.keys().cloned().collect();
    let mut frontier: BTreeSet<_> = window.clone();
    for _ in 0..depth {
        let mut next = BTreeSet::new();
        for basis in &frontier {
            let vector = NestedVector::from([(basis.clone(), C::new(1.0, 0.0))]);
            let image = h.d_gamma(&vector, limits()).unwrap();
            next.extend(project_occupation(&image, 1).0.into_keys());
        }
        window.extend(next.iter().cloned());
        frontier = next;
        if window.len() > 200_000 {
            break;
        }
    }
    let options = JointSirkOptions {
        action_limits: limits(),
        basis_window: Some(window.clone()),
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 2e-11,
            max_iterations: 32,
        },
        ..Default::default()
    };
    let rank: usize = std::env::var("SM_RESPONSE_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .max(1);
    let shifts = hashimoto_unitary_shifts(time, time * 7.0, time, rank).unwrap();
    let result = evolve_restarted_joint(&h, &initial, time, 1, &shifts, &options).unwrap();
    let norm = joint_norm(&result.state);
    let applied = occupation().d_gamma(&result.state, limits()).unwrap();
    let obs = (joint_inner_product(&result.state, &applied) / (norm * norm)).re;
    let applied_g = gauge_occupation().d_gamma(&result.state, limits()).unwrap();
    let obs_g = (joint_inner_product(&result.state, &applied_g) / (norm * norm)).re;
    let defect = result.defect_certificates.last().unwrap();
    serde_json::json!({
        "delta_gs": delta_gs,
        "delta_ye": delta_ye,
        "window_dimension": window.len(),
        "observable_n_up": obs,
        "observable_n_g": obs_g,
        "final_norm": norm,
        "energy_re": result.energies.last().unwrap().re,
        "n8_evolution_bound": defect.evolution_bound,
        "defect_frobenius": defect.defect_frobenius,
        "initial_leakage": defect.initial_leakage,
        "action_tail_norms": result.action_tail_norms.last().unwrap(),
        "shift_solve_error_bounds": result.shift_solve_error_bounds.last().unwrap(),
        "function_solve_error_bounds": result.function_solve_error_bounds.last().unwrap(),
        "orthogonality_defect": result.orthogonality_defects.last().unwrap(),
    })
}

fn main() {
    let time: f64 = std::env::var("SM_RESPONSE_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    let dgs: f64 = std::env::var("SM_RESPONSE_DGS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    let base = run(0.0, 0.0, time);
    if std::env::var("SM_RESPONSE_BASE_ONLY").is_ok() {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "experiment": "N11 base-state defect scan",
                "state": std::env::var("SM_RESPONSE_STATE").unwrap_or_else(|_| "seed".into()),
                "base": base,
            }))
            .unwrap()
        );
        return;
    }
    let plus = run(dgs, 0.0, time);
    let minus = run(-dgs, 0.0, time);
    let null = run(0.0, 0.0, time);
    let spectator = run(0.0, 0.05, time);
    let o = |v: &serde_json::Value| v["observable_n_up"].as_f64().unwrap();
    let og = |v: &serde_json::Value| v["observable_n_g"].as_f64().unwrap();
    let d = |v: &serde_json::Value| v["n8_evolution_bound"].as_f64().unwrap();
    let response = o(&plus) - o(&base);
    let budget = 2.0 * (d(&plus) + d(&base));
    let null_response = o(&null) - o(&base);
    let spectator_response = o(&spectator) - o(&base);
    let gauge_response = og(&plus) - og(&base);
    let gauge_null = og(&null) - og(&base);
    let out = serde_json::json!({
        "experiment": "N11 full-interaction response of the seed up-quark occupation",
        "operator": "complete native E1 SmHamiltonian (163 coordinates, 96 CAR modes)",
        "time": time,
        "steps": 1,
        "delta_gs": dgs,
        "shift_rule": "hashimoto_unitary_shifts(t, budget=7t, spacing=t, rank=2)",
        "cells": {"base": base, "plus_gs": plus, "minus_gs": minus, "null": null, "spectator_ye": spectator},
        "response": {
            "definition": format!("n_up(gs*(1+{dgs})) - n_up(gs)"),
            "value": response,
            "n8_two_sided_budget": budget,
            "within_budget": response.abs() <= budget,
            "antisymmetry": o(&plus) + o(&minus) - 2.0*o(&base),
            "null_control": null_response,
            "spectator_control": spectator_response,
        },
        "gauge_channel": {
            "definition": format!("n_g(gs*(1+{dgs})) - n_g(gs), mode g(0,0)"),
            "value": gauge_response,
            "n8_two_sided_budget": budget,
            "within_budget": gauge_response.abs() <= budget,
            "null_control": gauge_null,
        },
        "error_components": {
            "basis_tail": "action_tail_norms per cell (finite window)",
            "full_action_defect": "N8 evolution_bound per cell (conditional point bound)",
            "arithmetic": "f64/Complex64; not enclosed",
        },
        "claims": {
            "certified": false,
            "kind": "measured finite-window response, not a physical S-matrix or reaction rate",
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
