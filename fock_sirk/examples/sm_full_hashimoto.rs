//! Run the complete Cadabra-derived SM E1 operator through joint rational SIRK.
use anyhow::{Context, Result, bail};
use fock_sirk::joint_sirk::*;
use nested_fock_algebra::exact_inner::{
    ActionLimits, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use std::collections::BTreeSet;

fn main() -> Result<()> {
    let mut time: f64 = 0.01;
    let mut steps = 1usize;
    let mut krylov = 4usize;
    let mut cutoff = 1u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "Full SM E1 joint Hashimoto: --time 0.01 --steps 1 --krylov 4 --occupation-cutoff 1"
            );
            return Ok(());
        }
        let value = args.next().context("option requires a value")?;
        match arg.as_str() {
            "--time" => time = value.parse()?,
            "--steps" => steps = value.parse()?,
            "--krylov" => krylov = value.parse()?,
            "--occupation-cutoff" => cutoff = value.parse()?,
            _ => bail!("unknown option {arg}"),
        }
    }
    if !time.is_finite() || time <= 0.0 || steps == 0 || krylov < 2 {
        bail!("time>0, steps>0 and krylov>=2 required");
    }
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
    let parameters = SmParameters {
        gs: 0.3,
        gw: 0.2,
        gy: 0.15,
        lambda: 0.1,
        vev: 0.7,
        yu: matrix(0.02),
        yd: matrix(0.013),
        ynu: matrix(0.007),
        ye: matrix(0.019),
    };
    let model = cadabra_import::from_cadabra_latex(
        parameters,
        include_str!("../../docs/fixtures/sm_full_cadabra.latex"),
    )
    .map_err(anyhow::Error::msg)?;
    let h = model.expression();
    let limits = ActionLimits {
        max_components: 500_000,
        max_work: 500_000_000,
    };
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )?;
    let initial = NestedVector::from([(O::graded([(inner, 1)])?, C::new(1.0, 0.0))]);
    let complete = h.d_gamma(&initial, limits)?;
    let mut window: BTreeSet<_> = project_occupation(&complete, cutoff)
        .0
        .into_keys()
        .collect();
    window.extend(initial.keys().cloned());
    let dimension = window.len();
    let options = JointSirkOptions {
        action_limits: limits,
        basis_window: Some(window),
        occupation_cutoff: Some(cutoff),
        ..Default::default()
    };
    let dt = time / steps as f64;
    let budget = dt * (krylov as f64 + 5.0);
    let spacing = dt;
    let shifts = hashimoto_unitary_shifts(dt, budget, spacing, krylov)?;
    let result = evolve_restarted_joint(&h, &initial, time, steps, &shifts, &options)?;
    let output = serde_json::json!({
        "operator":"SM E1: complete inner operator enclosed once by outer creation/annihilation",
        "symbolic_source":"Cadabra2 LaTeX fixture -> mathhook -> exact joint action",
        "coordinates":163,"active_field_momenta":40,"squared_forms":49,
        "complex_CAR_coordinates":96,"real_Majorana_coordinates":192,
        "parameter_instance":"nonzero gauge/quartic couplings and four dense complex generation matrices; no mass calibration",
        "time":time,"restart_steps":steps,"rational_dimension_budget":krylov,
        "hashimoto_shifts":{"N":budget,"delta":spacing,"step_time":dt,
            "gamma":shifts.iter().map(|z|z.im).collect::<Vec<_>>()},
        "numerical_projection":{"basis":"support of projected full H applied to the initial state, plus the initial state",
            "occupation_cutoff":cutoff,"dimension":dimension},
        "norms":result.norms,"energies":result.energies.iter().map(|x|[x.re,x.im]).collect::<Vec<_>>(),
        "full_action_tail_norms":result.action_tail_norms,
        "full_resolvent_error_bounds":result.shift_solve_error_bounds,
        "function_resolvent_error_bounds":result.function_solve_error_bounds,
        "orthogonality_defects":result.orthogonality_defects,
        "final_support":result.state.len(),
        "scope":"controlled numerical compression; not an ESA, physical-BRST, cutoff-removal or roundoff certificate"
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
