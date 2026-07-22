use crate::change_of_vars::CoV;
use crate::flow::FlowAnalysis;
use crate::singularity::SingularityReport;

/// Status of the Essential Self-Adjointness analysis.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum EsaStatus {
    /// The classical flow is complete → the operator is ESA.
    EssentiallySelfAdjoint,
    /// The classical flow escapes → not ESA.
    NotEssentiallySelfAdjoint,
    /// Singularity was detected but resolved by a change of variables.
    SingularityResolved,
}

/// Full ESA analysis report.
#[derive(Clone, Debug, serde::Serialize)]
pub struct EsaReport {
    pub status: EsaStatus,
    pub flow: FlowAnalysis,
    pub singularity: Option<SingularityReport>,
    pub cov_applied: Option<CoV>,
    /// UK diagnostic codes raised during analysis.
    pub diagnostics: Vec<u32>,
}

/// Build an EsaReport from flow and singularity analysis results.
pub fn build_esa_report(
    flow: FlowAnalysis,
    singularity: Option<SingularityReport>,
    cov_applied: Option<CoV>,
) -> EsaReport {
    let mut diagnostics = Vec::new();

    if flow.is_complete {
        return EsaReport {
            status: EsaStatus::EssentiallySelfAdjoint,
            flow,
            singularity,
            cov_applied,
            diagnostics,
        };
    }

    // Flow incomplete → not ESA (UK-2101)
    diagnostics.push(2101);

    if let Some(ref s) = singularity {
        if s.singular {
            // UK-2102: singularity detected
            diagnostics.push(2102);
        }
    }

    if cov_applied.is_some() {
        // UK-2103: CoV was applied
        diagnostics.push(2103);
    }

    let status = if cov_applied.is_some() {
        EsaStatus::SingularityResolved
    } else {
        EsaStatus::NotEssentiallySelfAdjoint
    };

    EsaReport {
        status,
        flow,
        singularity,
        cov_applied,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::EscapeEvent;

    #[test]
    fn complete_flow_is_esa() {
        let flow = FlowAnalysis {
            is_complete: true,
            escapes: Vec::new(),
        };
        let report = build_esa_report(flow, None, None);
        assert_eq!(report.status, EsaStatus::EssentiallySelfAdjoint);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn incomplete_flow_not_esa() {
        let flow = FlowAnalysis {
            is_complete: false,
            escapes: vec![EscapeEvent {
                initial: vec![1.0],
                t_blowup: 1.0,
                divergent_axes: vec![0],
            }],
        };
        let report = build_esa_report(flow, None, None);
        assert_eq!(report.status, EsaStatus::NotEssentiallySelfAdjoint);
        assert!(report.diagnostics.contains(&2101));
    }

    #[test]
    fn incomplete_with_cov_resolved() {
        let flow = FlowAnalysis {
            is_complete: false,
            escapes: vec![EscapeEvent {
                initial: vec![1.0],
                t_blowup: 1.0,
                divergent_axes: vec![0],
            }],
        };
        let report = build_esa_report(flow, None, Some(CoV::Reciprocal(0)));
        assert_eq!(report.status, EsaStatus::SingularityResolved);
        assert!(report.diagnostics.contains(&2101));
        assert!(report.diagnostics.contains(&2103));
    }
}
