use crate::ode::Polynomial;
use crate::result::{OdeError, OdeSirkResult};

/// Type of singularity detected.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum SingularityType {
    /// Finite-time blow-up: ‖x‖ → ∞ in finite time.
    FiniteTimeBlowUp,
    /// Boundary hit: flow reaches a coordinate singularity (e.g. y = 0 for 1/y).
    BoundaryHit,
    /// Gradient blow-up: adaptive step Δt → 0 due to large gradients.
    GradientBlowUp,
}

/// Report from singularity analysis.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SingularityReport {
    pub singular: bool,
    pub singularity_type: Option<SingularityType>,
    /// Estimated blow-up time T(x0) for 1D systems (None for nD).
    pub blowup_time: Option<f64>,
    /// The initial condition at which singularity was detected.
    pub at_point: Option<Vec<f64>>,
}

/// Compute the blow-up time for a 1D ODE dx/dt = f(x) via quadrature:
/// T(x0) = ∫_{x0}^{∞} dx / f(x)
///
/// Uses the trapezoidal rule with an adaptive upper bound.
pub fn compute_blowup_time_1d(poly: &Polynomial, x0: f64) -> OdeSirkResult<f64> {
    if poly.n_vars != 1 {
        return Err(OdeError::Internal(
            "compute_blowup_time_1d requires a 1D polynomial".into(),
        ));
    }

    let upper = find_upper_bound(poly, x0)?;
    let n_steps = 100_000;
    let dx = (upper - x0) / n_steps as f64;

    let mut integral = 0.0;
    let mut x = x0;

    for _ in 0..n_steps {
        let f = poly.eval(&[x]);
        if f.abs() < 1e-15 {
            return Err(OdeError::FlowIntegration(format!(
                "f(x) ≈ 0 at x = {:.6}, integral diverges",
                x
            )));
        }
        let f_next = poly.eval(&[x + dx]);
        if f_next.abs() < 1e-15 {
            // Integrate up to the singularity
            integral += 0.5 * dx / f;
            break;
        }
        integral += 0.5 * dx * (1.0 / f + 1.0 / f_next);
        x += dx;

        // If the integrand is blowing up, we've likely passed the blow-up region
        if !integral.is_finite() {
            break;
        }
    }

    if !integral.is_finite() || integral < 0.0 {
        return Err(OdeError::FlowIntegration(format!(
            "quadrature produced non-physical T = {}",
            integral
        )));
    }

    Ok(integral)
}

/// Find a safe upper bound for the quadrature integral.
///
/// For x' = f(x), the blow-up time is T(x0) = ∫ dx/f(x).
/// If f grows like x^k, T ~ 1/x0^{k-1}, so the integral converges and
/// the upper bound should be chosen large enough that the remaining tail
/// is negligible.  We use: upper = x0 + 100 / |f(x0)| as a heuristic,
/// capped at 1e6 to avoid excessive step counts.
fn find_upper_bound(poly: &Polynomial, x0: f64) -> OdeSirkResult<f64> {
    let f0 = poly.eval(&[x0]);
    if f0.abs() < 1e-15 {
        return Err(OdeError::FlowIntegration(
            "f(x0) ≈ 0, cannot integrate".into(),
        ));
    }
    // Heuristic: the integrand 1/f(x) decays as f grows.
    // Placing the upper bound at x0 + 100/|f(x0)| captures most of the integral.
    let upper = (x0 + 100.0 / f0.abs()).min(1e6);
    Ok(upper)
}

/// Sweep initial conditions for singularity in a 1D system.
pub fn sweep_singularity_1d(
    poly: &Polynomial,
    sample_points: &[f64],
    t_max: f64,
) -> OdeSirkResult<SingularityReport> {
    for &x0 in sample_points {
        let f0 = poly.eval(&[x0]);
        if f0.abs() < 1e-15 {
            continue; // skip singular points
        }
        match compute_blowup_time_1d(poly, x0) {
            Ok(t_blowup) if t_blowup < t_max => {
                return Ok(SingularityReport {
                    singular: true,
                    singularity_type: Some(SingularityType::FiniteTimeBlowUp),
                    blowup_time: Some(t_blowup),
                    at_point: Some(vec![x0]),
                });
            }
            Ok(_) => {} // blow-up time exceeds t_max, no singularity in range
            Err(_) => continue,
        }
    }

    Ok(SingularityReport {
        singular: false,
        singularity_type: None,
        blowup_time: None,
        at_point: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::ODESystem;

    fn parse_poly(expr: &str) -> Polynomial {
        ODESystem::parse(vec!["x".into()], &[expr])
            .unwrap()
            .rhs[0]
            .clone()
    }

    #[test]
    fn x_squared_blowup_time() {
        let p = parse_poly("x^2");
        let t = compute_blowup_time_1d(&p, 1.0).unwrap();
        assert!((t - 1.0).abs() < 0.01, "T(1) = {}", t);
    }

    #[test]
    fn x_squared_blowup_time_half() {
        let p = parse_poly("x^2");
        let t = compute_blowup_time_1d(&p, 0.5).unwrap();
        assert!((t - 2.0).abs() < 0.05, "T(0.5) = {}", t);
    }

    #[test]
    fn stable_linear_no_singularity() {
        let p = parse_poly("-x");
        let report = sweep_singularity_1d(&p, &[1.0, 2.0, 5.0], 10.0).unwrap();
        assert!(!report.singular);
    }

    #[test]
    fn x_squared_sweep_detects() {
        let p = parse_poly("x^2");
        let report = sweep_singularity_1d(&p, &[0.5, 1.0], 10.0).unwrap();
        assert!(report.singular);
        assert_eq!(
            report.singularity_type,
            Some(SingularityType::FiniteTimeBlowUp)
        );
    }
}
