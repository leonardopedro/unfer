use crate::ode::ODESystem;
use crate::result::OdeSirkResult;

/// An escape event recorded during flow integration.
#[derive(Clone, Debug, serde::Serialize)]
pub struct EscapeEvent {
    /// Initial condition that escaped.
    pub initial: Vec<f64>,
    /// Time at which blow-up was detected.
    pub t_blowup: f64,
    /// Indices of variables that diverged.
    pub divergent_axes: Vec<usize>,
}

/// Result of classical flow analysis.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FlowAnalysis {
    /// True if ALL sampled trajectories remain bounded for t ∈ [0, t_max].
    pub is_complete: bool,
    /// List of escape events (empty iff is_complete).
    pub escapes: Vec<EscapeEvent>,
}

const R_MAX: f64 = 1e6;
const DT_MIN: f64 = 1e-14;

/// Integrate the classical flow from multiple initial conditions.
///
/// Uses an adaptive-step Euler method. A trajectory is considered
/// escaped when ‖x‖ > R_MAX or any component is non-finite.
pub fn analyze_classical_flow(
    sys: &ODESystem,
    samples: &[Vec<f64>],
    t_max: f64,
) -> OdeSirkResult<FlowAnalysis> {
    let n = sys.n_vars();
    let mut escapes = Vec::new();

    for x0 in samples {
        if x0.len() != n {
            continue;
        }
        match integrate_trajectory(sys, x0, t_max) {
            Ok(()) => {} // trajectory completed without escape
            Err(ev) => escapes.push(ev),
        }
    }

    Ok(FlowAnalysis {
        is_complete: escapes.is_empty(),
        escapes,
    })
}

fn integrate_trajectory(
    sys: &ODESystem,
    x0: &[f64],
    t_max: f64,
) -> Result<(), EscapeEvent> {
    let n = sys.n_vars();
    let mut x: Vec<f64> = x0.to_vec();
    let mut t = 0.0;
    let mut dt: f64 = 1e-4;

    while t < t_max {
        // Evaluate RHS
        let f: Vec<f64> = sys.eval(&x);

        // Adaptive step: reduce dt if any |f_i| is huge
        let max_f = f.iter().map(|fi| fi.abs()).fold(0.0f64, f64::max);
        if max_f > 0.0 {
            let suggested_dt = (R_MAX / max_f).min(1e-2).max(DT_MIN);
            dt = dt.min(suggested_dt);
        }
        if dt < DT_MIN {
            return Err(EscapeEvent {
                initial: x0.to_vec(),
                t_blowup: t,
                divergent_axes: (0..n).collect(),
            });
        }

        // Euler step
        for i in 0..n {
            x[i] += dt * f[i];
        }
        t += dt;

        // Check blow-up
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        if norm_sq > R_MAX * R_MAX || !x.iter().all(|xi| xi.is_finite()) {
            let divergent_axes = (0..n)
                .filter(|&i| x[i].abs() > R_MAX || !x[i].is_finite())
                .collect();
            return Err(EscapeEvent {
                initial: x0.to_vec(),
                t_blowup: t,
                divergent_axes,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::ODESystem;

    #[test]
    fn stable_linear_flow_complete() {
        let sys = ODESystem::parse(vec!["x".into()], &["-x"]).unwrap();
        let samples: Vec<Vec<f64>> = (1..=5).map(|i| vec![i as f64]).collect();
        let result = analyze_classical_flow(&sys, &samples, 10.0).unwrap();
        assert!(result.is_complete);
        assert!(result.escapes.is_empty());
    }

    #[test]
    fn x_squared_flow_escapes() {
        let sys = ODESystem::parse(vec!["x".into()], &["x^2"]).unwrap();
        let samples = vec![vec![1.0], vec![0.5]];
        let result = analyze_classical_flow(&sys, &samples, 100.0).unwrap();
        assert!(!result.is_complete);
        assert!(!result.escapes.is_empty());
    }

    #[test]
    fn coupled_flow_escapes() {
        let sys = ODESystem::parse(vec!["x".into(), "y".into()], &["y", "2*x*y"]).unwrap();
        let samples = vec![vec![1.0, 1.0], vec![0.5, 0.5]];
        let result = analyze_classical_flow(&sys, &samples, 10.0).unwrap();
        assert!(!result.is_complete);
    }
}
