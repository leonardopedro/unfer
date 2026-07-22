use crate::change_of_vars::CoV;
use crate::esa::{EsaReport, EsaStatus};

/// Unified report from the full ODE analysis pipeline.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OdeReport {
    /// Variable names.
    pub vars: Vec<String>,
    /// ESA analysis result.
    pub esa: EsaReport,
    /// Whether a change of variables was applied.
    pub cov: Option<CoV>,
    /// UK-2xxx diagnostic codes raised.
    pub diagnostics: Vec<u32>,
}

impl OdeReport {
    pub fn is_esa(&self) -> bool {
        self.esa.status == EsaStatus::EssentiallySelfAdjoint
            || self.esa.status == EsaStatus::SingularityResolved
    }

    pub fn summary(&self) -> String {
        let status_str = match self.esa.status {
            EsaStatus::EssentiallySelfAdjoint => "ESA (flow complete)",
            EsaStatus::NotEssentiallySelfAdjoint => "NOT ESA (flow incomplete)",
            EsaStatus::SingularityResolved => "ESA after CoV",
        };
        let sing = match &self.esa.singularity {
            Some(s) if s.singular => {
                let bt = s.blowup_time.map(|t| format!("{:.4}", t)).unwrap_or("?".into());
                format!(", blow-up T={}", bt)
            }
            _ => String::new(),
        };
        format!(
            "[{}] escapes={}{} diagnostics={:?}",
            status_str,
            self.esa.flow.escapes.len(),
            sing,
            self.diagnostics,
        )
    }
}

/// Build a full report from the analysis components.
pub fn build_report(
    vars: Vec<String>,
    esa: EsaReport,
    cov: Option<CoV>,
) -> OdeReport {
    let diagnostics = esa.diagnostics.clone();
    OdeReport {
        vars,
        esa,
        cov,
        diagnostics,
    }
}
