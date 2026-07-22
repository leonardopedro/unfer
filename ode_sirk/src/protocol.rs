use crate::change_of_vars::{self, CoV};
use crate::flow;
use crate::hamiltonian::ode_to_hamiltonian;
use crate::ode::ODESystem;
use crate::report::{self, OdeReport};
use crate::singularity;
use crate::result::OdeSirkResult;

/// The public analysis entry point.
///
/// 1. Parses the ODE system from strings.
/// 2. Runs classical flow analysis (Nelson's condition).
/// 3. If incomplete, sweeps for 1D singularities.
/// 4. Optionally applies a change of variables.
/// 5. Returns a full ESA report.
pub fn analyze_ode_system(
    vars: Vec<String>,
    rhs_strs: &[&str],
    cov_str: Option<&str>,
    t_max: f64,
    sample_points: &[Vec<f64>],
) -> OdeSirkResult<(OdeReport, nested_fock_algebra::Hamiltonian)> {
    // 1. Parse
    let sys = ODESystem::parse(vars.clone(), rhs_strs)?;

    // 2. Flow analysis
    let flow = flow::analyze_classical_flow(&sys, sample_points, t_max)?;

    // 3. Singularity sweep (1D only)
    let singularity = if sys.n_vars() == 1 && !flow.is_complete {
        let poly = sys.rhs[0].clone();
        let x0s: Vec<f64> = sample_points.iter().filter_map(|p| p.first().copied()).collect();
        Some(singularity::sweep_singularity_1d(&poly, &x0s, t_max)?)
    } else {
        None
    };

    // 4. Change of variables
    let cov = parse_cov_string(cov_str);
    let transformed_sys = if let Some(ref c) = cov {
        let ts = change_of_vars::apply_cov(&sys, c)?;
        Some(ts.new_ode)
    } else {
        None
    };

    // 5. Build Hamiltonian from (possibly transformed) system
    let hamiltonian_sys = transformed_sys.as_ref().unwrap_or(&sys);
    let hamiltonian = ode_to_hamiltonian(hamiltonian_sys)?;

    // 6. Build report
    let esa = crate::esa::build_esa_report(flow, singularity, cov.clone());
    let report = report::build_report(vars, esa, cov);

    Ok((report, hamiltonian))
}

/// Convenience: just analyze ESA without building the Hamiltonian.
pub fn analyze_esa(
    vars: Vec<String>,
    rhs_strs: &[&str],
    t_max: f64,
    sample_points: &[Vec<f64>],
) -> OdeSirkResult<OdeReport> {
    let (report, _) = analyze_ode_system(vars, rhs_strs, None, t_max, sample_points)?;
    Ok(report)
}

fn parse_cov_string(s: Option<&str>) -> Option<CoV> {
    match s? {
        "none" | "" => None,
        s if s.starts_with("reciprocal:") => {
            let axis: usize = s.trim_start_matches("reciprocal:").parse().ok()?;
            Some(CoV::Reciprocal(axis))
        }
        s if s.starts_with("logarithmic:") => {
            let axis: usize = s.trim_start_matches("logarithmic:").parse().ok()?;
            Some(CoV::Logarithmic(axis))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_stable_linear() {
        let samples: Vec<Vec<f64>> = (1..=3).map(|i| vec![i as f64]).collect();
        let (report, h) = analyze_ode_system(
            vec!["x".into()],
            &["-x"],
            None,
            10.0,
            &samples,
        )
        .unwrap();
        assert!(report.is_esa());
        assert!(!h.terms.is_empty());
    }

    #[test]
    fn analyze_x_squared_no_cov() {
        let samples = vec![vec![0.5], vec![1.0]];
        let (report, _) = analyze_ode_system(
            vec!["x".into()],
            &["x^2"],
            None,
            100.0,
            &samples,
        )
        .unwrap();
        assert!(!report.is_esa());
        assert!(report.diagnostics.contains(&2101));
    }

    #[test]
    fn analyze_x_squared_with_cov() {
        let samples = vec![vec![0.5], vec![1.0]];
        let (report, h) = analyze_ode_system(
            vec!["x".into()],
            &["x^2"],
            Some("reciprocal:0"),
            100.0,
            &samples,
        )
        .unwrap();
        assert_eq!(report.esa.status, crate::esa::EsaStatus::SingularityResolved);
        assert!(!h.terms.is_empty());
    }
}
