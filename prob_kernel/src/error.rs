use fock_sirk::SirkError;
use nested_fock_algebra::cas::CasError;
use unfer_protocol::{Code, Diagnostic, HintKind, RepairHint, Severity};

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
                Diagnostic::new(Code::GRAM_DEGENERATE, self.to_string(), Severity::Error)
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
                Diagnostic::new(Code::STATE_EXPLOSION, self.to_string(), Severity::Error)
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
                Diagnostic::new(Code::BRST_NOT_CONVERGED, self.to_string(), Severity::Error)
                    .with_hint(RepairHint::new(
                        HintKind::SetParam,
                        "solver.brst_tol",
                        "relax the BRST convergence tolerance",
                    ))
                    .with_data(serde_json::json!({"residual": residual}))
            }

            KernelError::Sirk(SirkError::Numeric(msg)) => {
                Diagnostic::new(Code::INTERNAL, msg.clone(), Severity::Error).with_hint(
                    RepairHint::new(
                        HintKind::ReduceScope,
                        "solver.krylov_dim",
                        "a numerical failure occurred during the solve; reduce the Krylov \
                         dimension or adjust the shifts and retry",
                    ),
                )
            }

            KernelError::Cas(CasError::TermExplosion { terms, limit }) => Diagnostic::new(
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
            .with_data(serde_json::json!({"terms": terms, "limit": limit})),

            KernelError::Cas(CasError::Parse(msg)) => Diagnostic::new(
                Code::BAD_JSON,
                format!("symbolic parse error: {msg}"),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "hamiltonian.latex",
                "ensure the LaTeX expression is well-formed",
            )),

            KernelError::UnknownBuiltinModel { name } => Diagnostic::new(
                Code::UNKNOWN_BUILTIN_MODEL,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "hamiltonian.name",
                "use one of: yang_mills, yang_mills_lattice, navier_stokes, gravity, \
                 harmonic_chain, bose_hubbard, qfm_mehler",
            ))
            .with_data(serde_json::json!({"requested": name})),

            KernelError::BadEventPredicate { reason } => {
                Diagnostic::new(Code::BAD_EVENT_PREDICATE, self.to_string(), Severity::Error)
                    .with_hint(RepairHint::new(
                        HintKind::ReplaceValue,
                        "event",
                        format!("fix predicate: {reason}"),
                    ))
            }

            KernelError::ZeroProbabilityCondition { mass } => Diagnostic::new(
                Code::ZERO_PROBABILITY_CONDITION,
                self.to_string(),
                Severity::Warning,
            )
            .with_hint(RepairHint::new(
                HintKind::UseAlternativeOp,
                "event",
                "condition on a less restrictive event or evolve further first",
            ))
            .with_data(serde_json::json!({"prior_mass": mass})),

            KernelError::BadTerms { reason }
            | KernelError::BadBuiltinParams { reason }
            | KernelError::BadPrior { reason } => {
                Diagnostic::new(Code::BAD_JSON, self.to_string(), Severity::Error).with_hint(
                    RepairHint::new(HintKind::ReplaceValue, "hamiltonian", reason),
                )
            }

            KernelError::BadJson(e) => {
                Diagnostic::new(Code::BAD_JSON, e.to_string(), Severity::Error).with_hint(
                    RepairHint::new(
                        HintKind::ReplaceValue,
                        "request",
                        "ensure the request body is valid JSON matching the documented schema",
                    ),
                )
            }

            KernelError::Internal(msg) => {
                Diagnostic::new(Code::INTERNAL, msg.clone(), Severity::Fatal).with_hint(
                    RepairHint::new(
                        HintKind::UseAlternativeOp,
                        "request",
                        "internal kernel error — retry the operation; if it persists, report it \
                         with the attached diagnostic data",
                    ),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One representative instance of **every** `KernelError` variant — including
    /// every inner `SirkError`/`CasError` variant routed through `Sirk(..)`/`Cas(..)`.
    ///
    /// If a new error variant is added, the exhaustive `match` in `to_diagnostic`
    /// forces a new arm at compile time; adding the variant here forces it through
    /// the coverage contract below. Keep this list complete.
    fn every_variant() -> Vec<KernelError> {
        vec![
            // Sirk(..) — all four SirkError variants.
            KernelError::Sirk(SirkError::GramDegenerate { max_eig: 1e-18 }),
            KernelError::Sirk(SirkError::StateExplosion {
                components: 9000,
                limit: 4096,
            }),
            KernelError::Sirk(SirkError::BrstNotConverged { residual: 1e-2 }),
            KernelError::Sirk(SirkError::Numeric("singular matrix".into())),
            // Cas(..) — both CasError variants.
            KernelError::Cas(CasError::TermExplosion {
                terms: 1_000_000,
                limit: 65_536,
            }),
            KernelError::Cas(CasError::Parse("unexpected token".into())),
            // Native KernelError variants.
            KernelError::UnknownBuiltinModel {
                name: "lattice_qcd".into(),
            },
            KernelError::BadEventPredicate {
                reason: "mode out of range".into(),
            },
            KernelError::ZeroProbabilityCondition { mass: 1e-30 },
            KernelError::BadTerms {
                reason: "empty op string".into(),
            },
            KernelError::BadBuiltinParams {
                reason: "g must be > 0".into(),
            },
            KernelError::BadPrior {
                reason: "negative occupation".into(),
            },
            KernelError::BadJson(serde_json::from_str::<i32>("not json").unwrap_err()),
            KernelError::Internal("unreachable state reached".into()),
        ]
    }

    /// P2.9 — `KernelError` → `Diagnostic` coverage audit.
    ///
    /// The "Zero-language-style" machine surface promises AI agents that every
    /// failure carries (a) a registered `UK-####` code and (b) at least one
    /// actionable `RepairHint`. A single unmapped variant silently degrading to
    /// a hint-less UK-5000 breaks that contract. This test enforces it.
    #[test]
    fn every_variant_maps_to_registered_code_with_hint() {
        let registry = unfer_protocol::codes::all();
        for err in every_variant() {
            let diag = err.to_diagnostic();

            // (a) the code must exist in the canonical registry — never an
            // ad-hoc number an agent can't look up.
            assert!(
                registry.iter().any(|(c, _, _)| *c == diag.code.0),
                "variant {err:?} produced unregistered code UK-{:04}",
                diag.code.0,
            );

            // (b) the repair-hint contract: at least one actionable hint.
            assert!(
                !diag.hints.is_empty(),
                "variant {err:?} (UK-{:04}) produced no RepairHint — breaks the \
                 agent repair contract",
                diag.code.0,
            );

            // every hint must actually name a target an agent can act on.
            for hint in &diag.hints {
                assert!(
                    !hint.target.is_empty() && !hint.suggestion.is_empty(),
                    "variant {err:?} produced an empty RepairHint",
                );
            }
        }
    }

    /// Variants that represent *user-actionable* failures (everything except the
    /// genuinely-internal UK-5000 bucket) must map to a **specific** code, not the
    /// internal catch-all. This is the "silent degradation to UK-5000" guard.
    #[test]
    fn user_actionable_variants_avoid_internal_catchall() {
        for err in every_variant() {
            let diag = err.to_diagnostic();
            let is_internal_variant = matches!(
                err,
                KernelError::Internal(_) | KernelError::Sirk(SirkError::Numeric(_)),
            );
            if !is_internal_variant {
                assert_ne!(
                    diag.code.0,
                    Code::INTERNAL.0,
                    "user-actionable variant {err:?} degraded to the UK-5000 catch-all",
                );
            }
        }
    }
}
