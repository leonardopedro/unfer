//! Run the complete Cadabra-derived SM E1 operator through joint rational SIRK.
use anyhow::{Context, Result, bail};
use fock_sirk::joint_sirk::*;
use fock_sirk::sm_defect::*;
use fock_sirk::sm_manifest::*;
use nested_fock_algebra::exact_inner::{
    ActionLimits, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::process::Command;
use std::time::Instant;

fn complex_matrix_json(matrix: &[[C; 3]; 3]) -> Value {
    Value::Array(
        matrix
            .iter()
            .map(|row| Value::Array(row.iter().map(|c| json!([c.re, c.im])).collect()))
            .collect(),
    )
}

fn parameters_json(parameters: &SmParameters) -> Value {
    json!({
        "gs": parameters.gs,
        "gw": parameters.gw,
        "gy": parameters.gy,
        "lambda": parameters.lambda,
        "vev": parameters.vev,
        "yu": complex_matrix_json(&parameters.yu),
        "yd": complex_matrix_json(&parameters.yd),
        "ynu": complex_matrix_json(&parameters.ynu),
        "ye": complex_matrix_json(&parameters.ye),
        "units": "input couplings; no measured-mass calibration",
    })
}

fn git_output(repo: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn source_provenance() -> Value {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let revision = git_output(&repo, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let status = git_output(&repo, &["status", "--porcelain"]).unwrap_or_default();
    let diff = git_output(&repo, &["diff", "HEAD"]).unwrap_or_default();
    let dirty = !status.trim().is_empty();
    json!({
        "git_revision": revision,
        "dirty": dirty,
        // Binds the tracked diff and the porcelain listing (which also names
        // untracked files); untracked *contents* are not hashed here.
        "dirty_manifest_sha256": sha256_hex(format!("{status}\n{diff}").as_bytes()),
        "crate": "fock_sirk",
        "binary": "sm_full_hashimoto",
    })
}

fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn state_identity(state: &NestedVector) -> String {
    hash_labels(state.iter().map(|(basis, amplitude)| {
        format!(
            "{basis:?}|{}+{}i",
            amplitude.re.to_bits(),
            amplitude.im.to_bits()
        )
    }))
}

fn main() -> Result<()> {
    let started = Instant::now();
    let mut time: f64 = 0.01;
    let mut steps = 1usize;
    let mut krylov = 4usize;
    let mut cutoff = 1u64;
    // Cached component entries scale with the window squared; larger occupation
    // cutoffs need an explicit budget rather than a silently smaller operator.
    let mut cache_limit = JointSirkOptions::default().max_cached_components;
    // Single-action limits as well: wider occupation cutoffs build larger
    // images, and both bounds must be raised explicitly rather than truncated.
    let mut action_components = 500_000usize;
    let mut action_work = 500_000_000usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "Full SM E1 joint Hashimoto: --time 0.01 --steps 1 --krylov 4 --occupation-cutoff 1 [--cache-limit N] [--action-components N] [--action-work N]"
            );
            return Ok(());
        }
        let value = args.next().context("option requires a value")?;
        match arg.as_str() {
            "--time" => time = value.parse()?,
            "--steps" => steps = value.parse()?,
            "--krylov" => krylov = value.parse()?,
            "--occupation-cutoff" => cutoff = value.parse()?,
            "--cache-limit" => cache_limit = value.parse()?,
            "--action-components" => action_components = value.parse()?,
            "--action-work" => action_work = value.parse()?,
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
    let parameter_record = parameters_json(&parameters);
    let model = cadabra_import::from_cadabra_latex(
        parameters,
        include_str!("../../docs/fixtures/sm_full_cadabra.latex"),
    )
    .map_err(anyhow::Error::msg)?;
    let h = model.expression();
    let limits = ActionLimits {
        max_components: action_components,
        max_work: action_work,
    };
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )?;
    let initial = NestedVector::from([(O::graded([(inner, 1)])?, C::new(1.0, 0.0))]);
    let fixture = include_str!("../../docs/fixtures/sm_full_cadabra.latex");
    let operator_identity = sha256_hex(
        format!(
            "{}\n{}\n{h:?}",
            sha256_hex(fixture.as_bytes()),
            canonical_json(&parameter_record)
        )
        .as_bytes(),
    );
    let state_identity_sha256 = state_identity(&initial);
    let complete = h.d_gamma(&initial, limits)?;
    let mut window: BTreeSet<_> = project_occupation(&complete, cutoff)
        .0
        .into_keys()
        .collect();
    window.extend(initial.keys().cloned());
    let dimension = window.len();
    let label_sha256 = hash_labels(window.iter().map(|basis| format!("{basis:?}")));
    let options = JointSirkOptions {
        action_limits: limits,
        basis_window: Some(window),
        occupation_cutoff: Some(cutoff),
        max_cached_components: cache_limit,
        ..Default::default()
    };
    let (gmres_relative_tolerance, gmres_max_iterations) = match options.method {
        JointKrylovMethod::ShiftInvert {
            relative_tolerance,
            max_iterations,
        } => (relative_tolerance, max_iterations),
        JointKrylovMethod::ForwardProducts => (0.0, 0),
    };
    let dt = time / steps as f64;
    let budget = dt * (krylov as f64 + 5.0);
    let spacing = dt;
    let shifts = hashimoto_unitary_shifts(dt, budget, spacing, krylov)?;
    let result = evolve_restarted_joint(&h, &initial, time, steps, &shifts, &options)?;
    let defect = result
        .defect_certificates
        .first()
        .context("N8 defect certificate missing")?;
    let certificate_hashes = CertificateHashes {
        operator: operator_identity.clone(),
        state: state_identity_sha256.clone(),
        basis: label_sha256.clone(),
    };
    let defect_json = certificate_json(defect, &certificate_hashes);
    verify_certificate(&defect_json, &certificate_hashes)?;
    let deterministic_result_sha256 = canonical_json_sha256(&json!({
        "operator_identity_sha256": &operator_identity,
        "state_identity_sha256": &state_identity_sha256,
        "basis_label_sha256": &label_sha256,
        "parameters": &parameter_record,
        "shifts": shifts.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
        "norms": &result.norms,
        "energies": result.energies.iter().map(|x| [x.re, x.im]).collect::<Vec<_>>(),
        "full_action_tail_norms": &result.action_tail_norms,
        "full_resolvent_error_bounds": &result.shift_solve_error_bounds,
        "function_resolvent_error_bounds": &result.function_solve_error_bounds,
        "orthogonality_defects": &result.orthogonality_defects,
        "final_support": result.state.len(),
        "defect_frobenius": defect.defect_frobenius,
        "defect_evolution_bound": defect.evolution_bound,
    }));
    let manifest = bind_artifact(json!({
        "schema": MANIFEST_SCHEMA,
        "status": "measured",
        "endpoint": "E1-unreduced",
        "source": source_provenance(),
        "operator": {
            "identity_sha256": operator_identity,
            "fixture_sha256": sha256_hex(fixture.as_bytes()),
            "fixture": "docs/fixtures/sm_full_cadabra.latex",
            "fixture_entries": 1241,
            "named_coordinate_blocks": ["x", "G,dG", "W,dW", "B,dB", "phi,dphi"],
            "coordinates": 163,
            "active_field_momenta": 40,
            "squared_forms": 49,
            "complex_CAR_coordinates": 96,
            "real_Majorana_coordinates": 192,
            "realification": "complex chiral CAR; gamma_x=c+c^dagger, gamma_y=i(c^dagger-c)",
            "parameter_instance": parameter_record,
            "native_dictionary": "nested_fock_algebra/src/sm_full.rs; independent Cadabra export",
            "ghost_convention": "none in this operator; N10 owns the joint BRST construction",
            "symbolic_engine": {"cadabra_live": false, "input": "pre-exported LaTeX fixture"},
        },
        "state": {
            "identity_sha256": state_identity_sha256,
            "kind": "outer graded one-particle state over one inner universe",
            "sector": "one up quark and one left neutrino; all three generations remain in h",
            "inner_one_particle": true,
            "outer_graded": true,
            "observable": "full-operator energy expectation and state norm",
        },
        "basis": {
            "kind": "support of the projected complete-image action on the initial state, plus the initial state",
            "occupation_cutoff": cutoff,
            "dimension": dimension,
            "label_sha256": label_sha256,
            "widths": "unit variance q=a+a^dagger, p=i(a^dagger-a)/2; ell=1, no displacement",
            "spatial_cutoff": "none: E1 keeps all 96 CAR coordinates and fixed collective coordinates",
            "boundary_conditions": "none: finite occupation window, not a spatial lattice",
        },
        "defect_certificate": defect_json,
        "solver": {
            "method": "shift-invert GMRES + bounded-resolvent compression",
            "rank_requested": krylov,
            "rank": krylov,
            "shift_count": shifts.len(),
            "shifts": shifts.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
            "shift_generation": {"rule": "gamma_j=(N-j*delta)/t", "N": budget, "delta": spacing, "step_time": dt,
                "time_scale": time},
            "gmres_relative_tolerance": gmres_relative_tolerance,
            "gmres_max_iterations": gmres_max_iterations,
            "gram_relative_tolerance": options.gram_relative_tolerance,
            "hermiticity_tolerance": options.hermiticity_tolerance,
            "unit_norm_steps": options.unit_norm_steps,
            "pade": "nalgebra DMatrix::exp (Pade)",
            "restart": {"steps": steps, "step_time": dt},
            "retained_seed_norm": result.norms.first().copied(),
            "cache_limit": cache_limit,
            "action_component_limit": action_components,
            "action_work_limit": action_work,
            "precision": "f64 / Complex64; no interval enclosure",
            "gram_whitening": "eigendecomposition + two-pass physical-vector reorthogonalization",
        },
        "environment": {
            "backend": if cfg!(feature = "cuda") { "cuda-requested" } else { "cpu" },
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "peak_rss_kb": peak_rss_kb(),
        },
        "claims": {
            "certified": false,
            "kind": "measured finite-window evolution, not a full-operator bound",
            "error_components": {
                "finite_window_tail": "reported in full_action_tail_norms",
                "shifted_solve": "reported in full_resolvent_error_bounds (conditional)",
                "function_solve": "reported in function_resolvent_error_bounds",
                "orthogonality": "reported in orthogonality_defects",
                "rounding": "not enclosed",
                "full_action_defect": format!(
                "measured point bound {:.6e}; conditional on the identified H and the finite exponential; see defect_certificate",
                defect.evolution_bound
            ),
            },
            "deterministic_result_sha256": deterministic_result_sha256,
            "tests_executed": "suite inventory is recorded by scripts/run-sm-baseline",
        },
    }));
    verify_manifest(&manifest)?;
    let output = serde_json::json!({
        "result_manifest": manifest,
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
