use unfer_protocol::{
    Code, Diagnostic, HintKind, RepairHint, Severity,
};
use nested_fock_algebra::cas::CasError;
use fock_sirk::SirkError;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error(transparent)]
    Sirk(#[from] SirkError),

    #[error(transparent)]
    Cas(#[from] CasError),

    #[error("unknown builtin model: {name}")]
    UnknownBuiltinModel { name: String },

    #[error("bad event predicate: {reason}")]
    BadEventPredicate { reason: String },

    #[error("conditioning on zero-probability event (mass {mass:.3e})")]
    ZeroProbabilityCondition { mass: f64 },

    #[error("bad HamiltonianSpec::Terms: {reason}")]
    BadTerms { reason: String },

    #[error("bad HamiltonianSpec::Builtin params: {reason}")]
    BadBuiltinParams { reason: String },

    #[error("bad prior: {reason}")]
    BadPrior { reason: String },

    #[error("JSON error: {0}")]
    BadJson(#[from] serde_json::Error),

    #[error("internal: {0}")]
    Internal(String),
}

impl KernelError {
    pub fn to_diagnostic(&self) -> Diagnostic {
        match self {
            KernelError::Sirk(SirkError::GramDegenerate { max_eig }) => {
                Diagnostic::new(
                    Code::GRAM_DEGENERATE,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::ReduceScope,
                    "solver.krylov_dim",
                    "reduce the Krylov dimension to avoid linearly dependent vectors",
                ))
                .with_hint(RepairHint::new(
                    HintKind::SetParam,
                    "shifts",
                    "use shifts with larger imaginary separation",
                ))
                .with_data(serde_json::json!({"max_eigenvalue": max_eig}))
            }

            KernelError::Sirk(SirkError::StateExplosion { components, limit }) => {
                Diagnostic::new(
                    Code::STATE_EXPLOSION,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::IncreaseLimit,
                    "solver.max_components",
                    "raise the component ceiling to allow the expansion",
                ))
                .with_hint(RepairHint::new(
                    HintKind::ReduceScope,
                    "solver.krylov_dim",
                    "reduce the Krylov dimension to slow state growth",
                ))
                .with_data(serde_json::json!({"components": components, "limit": limit}))
            }

            KernelError::Sirk(SirkError::BrstNotConverged { residual }) => {
                Diagnostic::new(
                    Code::BRST_NOT_CONVERGED,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::SetParam,
                    "solver.brst_tol",
                    "relax the BRST convergence tolerance",
                ))
                .with_data(serde_json::json!({"residual": residual}))
            }

            KernelError::Sirk(SirkError::Numeric(msg)) => {
                Diagnostic::new(Code::INTERNAL, msg.clone(), Severity::Error)
            }

            KernelError::Cas(CasError::TermExplosion { terms, limit }) => {
                Diagnostic::new(
                    Code::CAS_TERM_EXPLOSION,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::IncreaseLimit,
                    "cas.max_terms",
                    "raise the CAS expansion term limit",
                ))
                .with_hint(RepairHint::new(
                    HintKind::UseAlternativeOp,
                    "hamiltonian",
                    "use HamiltonianSpec::Terms to build operators directly (bypasses CAS expansion)",
                ))
                .with_data(serde_json::json!({"terms": terms, "limit": limit}))
            }

            KernelError::Cas(CasError::Parse(msg)) => {
                Diagnostic::new(
                    Code::BAD_JSON,
                    format!("symbolic parse error: {msg}"),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::ReplaceValue,
                    "hamiltonian.latex",
                    "ensure the LaTeX expression is well-formed",
                ))
            }

            KernelError::UnknownBuiltinModel { name } => {
                Diagnostic::new(
                    Code::UNKNOWN_BUILTIN_MODEL,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::ReplaceValue,
                    "hamiltonian.name",
                    "use one of: yang_mills, navier_stokes, gravity, harmonic_chain",
                ))
                .with_data(serde_json::json!({"requested": name}))
            }

            KernelError::BadEventPredicate { reason } => {
                Diagnostic::new(
                    Code::BAD_EVENT_PREDICATE,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::ReplaceValue,
                    "event",
                    format!("fix predicate: {reason}"),
                ))
            }

            KernelError::ZeroProbabilityCondition { mass } => {
                Diagnostic::new(
                    Code::ZERO_PROBABILITY_CONDITION,
                    self.to_string(),
                    Severity::Warning,
                )
                .with_hint(RepairHint::new(
                    HintKind::UseAlternativeOp,
                    "event",
                    "condition on a less restrictive event or evolve further first",
                ))
                .with_data(serde_json::json!({"prior_mass": mass}))
            }

            KernelError::BadTerms { reason } | KernelError::BadBuiltinParams { reason } |
            KernelError::BadPrior { reason } => {
                Diagnostic::new(
                    Code::BAD_JSON,
                    self.to_string(),
                    Severity::Error,
                )
                .with_hint(RepairHint::new(
                    HintKind::ReplaceValue,
                    "hamiltonian",
                    reason,
                ))
            }

            KernelError::BadJson(e) => {
                Diagnostic::new(
                    Code::BAD_JSON,
                    e.to_string(),
                    Severity::Error,
                )
            }

            KernelError::Internal(msg) => {
                Diagnostic::new(Code::INTERNAL, msg.clone(), Severity::Fatal)
            }
        }
    }
}
