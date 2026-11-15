//! Scoped equal-tolerance Mehler/SIRK comparison, two-mode cell.
//!
//! Two bosonic modes are unitarily dilated, each with the wide squeezed vacuum
//! `q=ell(a+a^dagger)` and operator map `q -> q/ell`, `p -> ell p`,
//! `ell=1.25`. The product state is cut at total boson occupation 12; the
//! one-mode exact tail there is 1.29e-10 and the product tail is reported with
//! the run. The same two-fermion seed, time, shift rule, rank and action limits
//! are used for the baseline and squeezed runs; only the representation
//! changes. This is a two-mode finite cell, not a 40/163-mode speedup claim.

use fock_sirk::joint_sirk::*;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, InnerLadder as L, JointInnerBasis as B, JointOuterBasis as O,
    NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use std::collections::BTreeSet;
use std::time::Instant;

const ELL: f64 = 1.25;
const CUT: usize = 12;
const TIME: f64 = 0.01;

fn modes() -> Vec<u32> {
    vec![coord::g(0, 0), coord::g(1, 1)]
}

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
        max_components: 2_000_000,
        max_work: 20_000_000_000,
    }
}
fn fermions() -> [u32; 2] {
    [
        matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
        matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
    ]
}
fn seed() -> NestedVector {
    let inner = B::new([], fermions()).unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}

/// Per-mode squeezed-vacuum amplitudes `c_{2m}`, real and positive for ell>1.
fn squeezed_amplitudes() -> Vec<f64> {
    let r = ELL.ln();
    let t = r.tanh();
    let norm = 1.0 / r.cosh().sqrt();
    let mut out = vec![norm];
    for m in 1..=CUT / 2 {
        let mf = m as f64;
        let next = out[m - 1] * t * ((2.0 * mf - 1.0) * 2.0 * mf).sqrt() / (2.0 * mf);
        out.push(next);
    }
    out
}

/// Product squeezed state over `modes`, truncated at total occupation `CUT`.
/// Returns the state and the exact discarded probability.
fn squeezed_state(modes: &[u32]) -> (NestedVector, f64) {
    let amps = squeezed_amplitudes();
    let mut states: Vec<(Vec<(u32, u32)>, f64)> = vec![(Vec::new(), 1.0)];
    for &mode in modes {
        let mut next = Vec::new();
        for (occ, weight) in &states {
            let used: u32 = occ.iter().map(|(_, n)| *n).sum();
            for (m, &amp) in amps.iter().enumerate() {
                let quanta = 2 * m as u32;
                if used + quanta > CUT as u32 {
                    continue;
                }
                let mut occupation = occ.clone();
                if quanta != 0 {
                    occupation.push((mode, quanta));
                }
                next.push((occupation, weight * amp));
            }
        }
        states = next;
    }
    let kept: f64 = states.iter().map(|(_, a)| a * a).sum();
    let mut out = NestedVector::new();
    for (occupation, amp) in states {
        let inner = B::new(occupation, fermions()).unwrap();
        out.insert(O::graded([(inner, 1)]).unwrap(), C::new(amp, 0.0));
    }
    (out, (1.0 - kept).max(0.0))
}

fn is_coordinate(e: &InnerExpr, mode: u32) -> bool {
    matches!(e, InnerExpr::Sum(v)
        if v.len() == 2
        && matches!(&v[0], InnerExpr::Ladder(L::BosonCreate(m)) if *m == mode)
        && matches!(&v[1], InnerExpr::Ladder(L::BosonAnnihilate(m)) if *m == mode))
}
fn is_momentum(e: &InnerExpr, mode: u32) -> bool {
    let ok = |x: &InnerExpr, create: bool, imag: f64| match x {
        InnerExpr::Product(v) if v.len() == 2 => {
            let InnerExpr::Scalar(c) = &v[0] else {
                return false;
            };
            match (&v[1], create) {
                (InnerExpr::Ladder(L::BosonCreate(m)), true)
                | (InnerExpr::Ladder(L::BosonAnnihilate(m)), false) => {
                    *m == mode && (c - C::new(0.0, imag)).norm() < 1e-15
                }
                _ => false,
            }
        }
        _ => false,
    };
    matches!(e, InnerExpr::Sum(v)
        if v.len() == 2 && ok(&v[0], true, 0.5) && ok(&v[1], false, -0.5))
}
/// Unitary dilation of every coordinate/momentum occurrence of every mode.
fn dilate(e: &InnerExpr, modes: &[u32]) -> InnerExpr {
    // The wide squeezed vacuum |0_ell> has q=ell(a+a^dagger), so the
    // corresponding unitary maps q -> q/ell and p -> ell p.
    for &mode in modes {
        if is_coordinate(e, mode) {
            return e.clone().scaled(C::new(1.0 / ELL, 0.0));
        }
        if is_momentum(e, mode) {
            return e.clone().scaled(C::new(ELL, 0.0));
        }
    }
    match e {
        InnerExpr::Scalar(_) | InnerExpr::Ladder(_) => e.clone(),
        InnerExpr::Sum(v) => InnerExpr::Sum(v.iter().map(|x| dilate(x, modes)).collect()),
        InnerExpr::Product(v) => InnerExpr::Product(v.iter().map(|x| dilate(x, modes)).collect()),
    }
}

fn up_occupation() -> InnerExpr {
    let up = matter_mode(Species::Up, 0, 0, Chirality::Left, 0);
    InnerExpr::Product(vec![
        InnerExpr::Ladder(L::FermionCreate(up)),
        InnerExpr::Ladder(L::FermionAnnihilate(up)),
    ])
}

fn boson_occupation(modes: &[u32]) -> InnerExpr {
    InnerExpr::Sum(
        modes
            .iter()
            .map(|&mode| {
                InnerExpr::Product(vec![
                    InnerExpr::Ladder(L::BosonCreate(mode)),
                    InnerExpr::Ladder(L::BosonAnnihilate(mode)),
                ])
            })
            .collect(),
    )
}

struct Run {
    label: &'static str,
    window: usize,
    initial_components: usize,
    defect: f64,
    defect_frobenius: f64,
    leakage: f64,
    tail: Vec<f64>,
    shift_bounds: Vec<f64>,
    function_bounds: Vec<f64>,
    orthogonality: f64,
    energy: f64,
    norm: f64,
    observable: f64,
    observable_up: f64,
    elapsed_s: f64,
}
fn run(label: &'static str, h: &InnerExpr, initial: &NestedVector, modes: &[u32]) -> Run {
    let started = Instant::now();
    let image = h.d_gamma(initial, limits()).unwrap();
    let mut window: BTreeSet<_> = project_occupation(&image, 1).0.into_keys().collect();
    window.extend(initial.keys().cloned());
    let options = JointSirkOptions {
        action_limits: limits(),
        basis_window: Some(window.clone()),
        occupation_cutoff: Some(CUT as u64),
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 2e-11,
            max_iterations: 32,
        },
        ..Default::default()
    };
    let shifts = hashimoto_unitary_shifts(TIME, TIME * 7.0, TIME, 2).unwrap();
    let result = evolve_restarted_joint(h, initial, TIME, 1, &shifts, &options).unwrap();
    let defect = result.defect_certificates.last().unwrap();
    let norm = joint_norm(&result.state);
    let applied = boson_occupation(modes)
        .d_gamma(&result.state, limits())
        .unwrap();
    let applied_up = up_occupation().d_gamma(&result.state, limits()).unwrap();
    let up = (joint_inner_product(&result.state, &applied_up) / (norm * norm)).re;
    Run {
        label,
        window: window.len(),
        initial_components: initial.len(),
        defect: defect.evolution_bound,
        defect_frobenius: defect.defect_frobenius,
        leakage: defect.initial_leakage,
        tail: result.action_tail_norms.last().unwrap().clone(),
        shift_bounds: result.shift_solve_error_bounds.last().unwrap().clone(),
        function_bounds: result.function_solve_error_bounds.last().unwrap().clone(),
        orthogonality: *result.orthogonality_defects.last().unwrap(),
        energy: result.energies.last().unwrap().re,
        norm,
        observable: (joint_inner_product(&result.state, &applied) / (norm * norm)).re,
        observable_up: up,
        elapsed_s: started.elapsed().as_secs_f64(),
    }
}

fn main() {
    let model = SmHamiltonian::new(parameters()).unwrap();
    let h = model.expression();
    let modes = modes();
    let (initial, exact_tail) = squeezed_state(&modes);
    let base = run("baseline", &h, &seed(), &modes);
    let hp = dilate(&h, &modes);
    let squeezed = run("squeezed", &hp, &initial, &modes);
    let rows: Vec<_> = [base, squeezed]
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "label": r.label,
                "window_dimension": r.window,
                "initial_components": r.initial_components,
                "n8_evolution_bound": r.defect,
                "defect_frobenius": r.defect_frobenius,
                "initial_leakage": r.leakage,
                "action_tail_norms": r.tail,
                "shift_solve_error_bounds": r.shift_bounds,
                "function_solve_error_bounds": r.function_bounds,
                "orthogonality_defect": r.orthogonality,
                "energy_re": r.energy,
                "final_norm": r.norm,
                "observable_n_modes": r.observable,
                "observable_n_up": r.observable_up,
                "elapsed_s": r.elapsed_s,
            })
        })
        .collect();
    let out = serde_json::json!({
        "experiment": "N9 equal-tolerance two-mode squeezed-state SIRK cell",
        "modes": modes,
        "ell": ELL,
        "cutoff": CUT,
        "squeezed_tail_at_cutoff": exact_tail,
        "time": TIME,
        "steps": 1,
        "shift_rule": "hashimoto_unitary_shifts(t, 7t, t, rank=2)",
        "dilation": "q -> q/ell, p -> ell p for both modes; wide product squeezed vacuum",
        "errors": {
            "basis_tail": "declared total cutoff 12; exact kept product tail reported",
            "solver": "shift/function solve bounds per run",
            "n8": "full-action uniform column defect per run",
            "arithmetic": "f64/Complex64; not enclosed",
        },
        "claims": {
            "kind": "two-mode finite equal-tolerance comparison, not a 40/163-mode speedup or E2 theorem",
            "certified": false,
        },
        "rows": rows,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
