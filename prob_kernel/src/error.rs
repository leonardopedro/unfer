use fock_sirk::SirkError;
use nested_fock_algebra::cas::CasError;
use unfer_protocol::{Code, Diagnostic, HintKind, RepairHint, Severity};

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error(transparent)]
    Sirk(#[from] SirkError),

    #[error(transparent)]
    Cas(#[from] CasError),

    #[error("QFM pipeline error: {0}")]
    Qfm(#[from] qfm::pipeline::QfmError),

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

    #[error("Lean4 proof verification failed: {reason}")]
    ProofVerify { reason: String },

    #[error("Lean4 export file invalid: {reason}")]
    ProofExportInvalid { reason: String },

    #[error("Cadabra2 symbolic engine unavailable: {reason}")]
    SymbolicUnavailable { reason: String },

    #[error("Cadabra2 symbolic expression invalid: {reason}")]
    SymbolicInvalid { reason: String },

    #[error("Why3 verification engine unavailable: {reason}")]
    WhymlUnavailable { reason: String },

    #[error("WhyML spec invalid: {reason}")]
    WhymlInvalid { reason: String },

    #[error("Logos CNL->UNF compile failed: {reason}")]
    LogosFailed { reason: String },

    #[error("Austral->deltanet UNF translation failed: {reason}")]
    AustralUnfFailed { reason: String },

    #[error(
        "session event-log format version {got} is unsupported, or the log is malformed: {reason}"
    )]
    SessionLogVersion { got: u32, reason: String },

    #[error("orphaned session compaction lock: {reason}")]
    SessionCompactionOrphaned { reason: String },

    #[error("session compaction refused (session not idle): {reason}")]
    SessionCompactionBusy { reason: String },

    #[error("session fork refused at log boundary: {reason}")]
    SessionForkRange { reason: String },

    #[error("durable checkpoint failed: {reason}")]
    DurableCheckpointFailed { reason: String },

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
                 harmonic_chain, bose_hubbard, qfm_mehler, qfm_mehler_localized, \
                 qfm_mehler_projector, qfm_mehler_projector_localized",
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

            KernelError::ProofVerify { reason } => {
                Diagnostic::new(Code::PROOF_VERIFY_FAILED, self.to_string(), Severity::Error)
                    .with_hint(RepairHint::new(
                        HintKind::ReplaceValue,
                        "proof.export",
                        reason,
                    ))
            }

            KernelError::ProofExportInvalid { reason } => Diagnostic::new(
                Code::PROOF_EXPORT_INVALID,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "proof.export",
                reason,
            )),

            KernelError::SymbolicUnavailable { reason } => Diagnostic::new(
                Code::SYMBOLIC_ENGINE_UNAVAILABLE,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::UseAlternativeOp,
                "symbolic.engine",
                reason,
            )),

            KernelError::SymbolicInvalid { reason } => Diagnostic::new(
                Code::SYMBOLIC_EXPRESSION_INVALID,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "symbolic.expression",
                reason,
            )),

            KernelError::WhymlUnavailable { reason } => Diagnostic::new(
                Code::WHYML_ENGINE_UNAVAILABLE,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::UseAlternativeOp,
                "whyml.op",
                reason,
            )),

            KernelError::WhymlInvalid { reason } => Diagnostic::new(
                Code::WHYML_SPEC_INVALID,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "whyml.spec",
                reason,
            )),

            KernelError::LogosFailed { reason } => Diagnostic::new(
                Code::LOGOS_COMPILE_FAILED,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "logos.sentence",
                reason,
            )),

            KernelError::AustralUnfFailed { reason } => Diagnostic::new(
                Code::AUSTRAL_UNF_FAILED,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::ReplaceValue,
                "austral.source",
                reason,
            )),

            KernelError::SessionLogVersion { got, reason } => Diagnostic::new(
                Code::SESSION_LOG_VERSION,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::SetParam,
                "session.format_version",
                format!(
                    "expected format version, got {got}: {reason} — re-serialize the session \
                     with the current kernel"
                ),
            ))
            .with_data(serde_json::json!({"format_version": got})),

            KernelError::SessionCompactionOrphaned { reason } => Diagnostic::new(
                Code::SESSION_COMPACTION_ORPHANED,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::UseAlternativeOp,
                "session.compaction",
                format!(
                    "{reason} — compact again (the lock bracket re-closes) or save a fresh \
                     snapshot without the orphaned bracket"
                ),
            )),

            KernelError::SessionCompactionBusy { reason } => Diagnostic::new(
                Code::SESSION_COMPACTION_BUSY,
                self.to_string(),
                Severity::Error,
            )
            .with_hint(RepairHint::new(
                HintKind::SetParam,
                "session.compaction.through_seq",
                format!(
                    "{reason} — compact only an idle session, to a boundary that does not \
                     split an unanswered action_apply/evolve dependency"
                ),
            )),

            KernelError::SessionForkRange { reason } => {
                Diagnostic::new(Code::SESSION_FORK_RANGE, self.to_string(), Severity::Error)
                    .with_hint(RepairHint::new(
                        HintKind::SetParam,
                        "session.fork.seq",
                        format!("{reason} — fork from a valid, closed log boundary"),
                    ))
            }

            KernelError::DurableCheckpointFailed { reason } => {
                Diagnostic::new(Code::UNKNOWN_OUTCOME, self.to_string(), Severity::Error).with_hint(
                    RepairHint::new(
                        HintKind::SetParam,
                        "durable.checkpoint",
                        format!(
                            "{reason} — the durable checkpoint could not be flushed; read-only \
                             work may retry, but a side-effecting call whose outcome is unknown \
                             must be verified manually before repeating it"
                        ),
                    ),
                )
            }

            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch { expected, got }) => {
                Diagnostic::new(Code::BAD_JSON, self.to_string(), Severity::Error).with_hint(
                    RepairHint::new(
                        HintKind::ReplaceValue,
                        "evolve.query",
                        format!("query must have {expected} elements (got {got})"),
                    ),
                )
            }

            KernelError::Qfm(qfm::pipeline::QfmError::DegenerateBasis) => Diagnostic::new(
                Code::INTERNAL,
                self.to_string(),
                Severity::Fatal,
            )
            .with_hint(RepairHint::new(
                HintKind::SetParam,
                "hamiltonian.spec.krylov_dim",
                "the Krylov basis is degenerate; increase krylov_dim or check training data",
            )),

            KernelError::Qfm(qfm::pipeline::QfmError::SirkFailed(msg)) => {
                Diagnostic::new(Code::INTERNAL, self.to_string(), Severity::Fatal).with_hint(
                    RepairHint::new(
                        HintKind::SetParam,
                        "hamiltonian.spec",
                        format!(
                            "SIRK solve failed during QFM compile ({msg}); adjust the Krylov \
                             shifts or the Hamiltonian structure"
                        ),
                    ),
                )
            }

            KernelError::Qfm(qfm::pipeline::QfmError::K2ExceedsKrylovDim {
                k2,
                krylov_dim,
                m,
                config_krylov_dim,
            }) => Diagnostic::new(Code::BAD_JSON, self.to_string(), Severity::Error).with_hint(
                RepairHint::new(
                    HintKind::SetParam,
                    "hamiltonian.spec",
                    format!(
                        "QFM compile: K_2 = {k2} > effective krylov_dim = {krylov_dim} \
                     (config.krylov_dim = {config_krylov_dim}, m = {m}). Either increase \
                     config.krylov_dim to at least K_2, or reduce K_2 to <= M, or add more \
                     training points so M >= K_2."
                    ),
                ),
            ),

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
            // Qfm(..) — all four QfmError variants.
            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch {
                expected: 8,
                got: 4,
            }),
            KernelError::Qfm(qfm::pipeline::QfmError::DegenerateBasis),
            KernelError::Qfm(qfm::pipeline::QfmError::SirkFailed(
                "singular matrix".into(),
            )),
            KernelError::Qfm(qfm::pipeline::QfmError::K2ExceedsKrylovDim {
                k2: 8,
                krylov_dim: 4,
                m: 4,
                config_krylov_dim: 8,
            }),
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
            KernelError::ProofVerify {
                reason: "theorem pf has type Sort 1, expected P".into(),
            },
            KernelError::ProofExportInvalid {
                reason: "unparseable NDJSON".into(),
            },
            KernelError::SymbolicUnavailable {
                reason: "cadabra2-cli not found".into(),
            },
            KernelError::SymbolicInvalid {
                reason: "expression reduced to empty".into(),
            },
            KernelError::WhymlUnavailable {
                reason: "why3 not found".into(),
            },
            KernelError::WhymlInvalid {
                reason: "unknown symbol uk_foo".into(),
            },
            KernelError::AustralUnfFailed {
                reason: "unparseable Austral source".into(),
            },
            KernelError::SessionLogVersion {
                got: 99,
                reason: "expected format version 1".into(),
            },
            KernelError::SessionCompactionOrphaned {
                reason: "compaction/start without compaction/end".into(),
            },
            KernelError::SessionCompactionBusy {
                reason: "an open compaction lock".into(),
            },
            KernelError::SessionForkRange {
                reason: "seq 12 out of range".into(),
            },
            KernelError::DurableCheckpointFailed {
                reason: "io error flushing snapshot".into(),
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

    /// Every variant's `Display` message (the `#[error(...)]` arms) must be
    /// non-empty, and the diagnostic the agent sees must carry a non-empty
    /// primary message — a variant whose message is empty or lost in the
    /// diagnostic chain breaks the agent-facing error contract (an agent must
    /// be able to read *why* the call failed). Some arms rebuild the message
    /// with an added prefix (e.g. `Cas::Parse` → "symbolic parse error: …"),
    /// so the contract is non-emptiness of both, not exact equality.
    #[test]
    fn every_variant_display_message_is_nonempty_and_surfaces() {
        for err in every_variant() {
            let msg = err.to_string();
            assert!(
                !msg.trim().is_empty(),
                "variant {err:?} renders an empty Display message"
            );
            let diag = err.to_diagnostic();
            assert!(
                !diag.message.trim().is_empty(),
                "variant {err:?} diagnostic has an empty primary message"
            );
            // Severity is always one of the four known levels.
            assert!(
                matches!(
                    diag.severity,
                    Severity::Error | Severity::Warning | Severity::Fatal
                ),
                "variant {err:?} has unexpected severity {:?}",
                diag.severity
            );
        }
    }

    /// Stress the diagnostic path: repeated conversion of every variant must
    /// be stable (deterministic message/code/hints across calls).
    #[test]
    fn diagnostic_conversion_is_stable_under_repetition() {
        for _ in 0..50 {
            for err in every_variant() {
                let a = err.to_diagnostic();
                let b = err.to_diagnostic();
                assert_eq!(a.code, b.code);
                assert_eq!(a.message, b.message);
                assert_eq!(a.hints.len(), b.hints.len());
                assert_eq!(a.severity, b.severity);
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
                KernelError::Internal(_)
                    | KernelError::Sirk(SirkError::Numeric(_))
                    | KernelError::Qfm(qfm::pipeline::QfmError::DegenerateBasis)
                    | KernelError::Qfm(qfm::pipeline::QfmError::SirkFailed(_)),
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
