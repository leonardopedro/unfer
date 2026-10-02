//! P4 — the CNL completer: ProofFlow's tactic completer, retargeted.
//!
//! # The substitution
//!
//! ProofFlow's `proof_prover.run_solver_prompt` is a *tactic* completer. It gets
//! a skeleton Lean file ending in `sorry`, asks the model to fill in the proof,
//! and loops until `lean_verify` — the file compiles **with no `sorry`**. The
//! gate is what makes it a completer: `sorry` compiles, so a skeleton always
//! passes `lean_pass`, and only the completer can clear the sorry.
//!
//! L0 has no tactics and no `sorry`, so there is nothing to fill in — the
//! sentence is already complete the moment it parses. Completion is therefore
//! *reformulation*: rewrite the sentence until it reduces to a **unique normal
//! form**, which is the `lean_verify` analogue introduced in P2 and reached by
//! [`crate::formalize::formalizer::formalize_node_verified`].
//!
//! Everything else is ProofFlow's loop, unchanged in shape:
//!
//! - conditions and definitions are **skipped**, not attempted — they are
//!   formalized but never proved, which is §14's retained split;
//! - a node with no formalization is skipped;
//! - the model is told when the previous sentence did not even compile;
//! - the **typed** failure is fed back, so a missing lexicon word is named;
//! - `prove_negation` refutes rather than proves, landing in `solved_negation`.
//!
//! What genuinely differs is the repair *surface*. A tactic completer chooses
//! from a tactic language; this chooses from a 46-word lexicon, so the prompt
//! leads with a table mapping each diagnostic to the class of edit that fixes it
//! — which is the whole content of "completing" a sentence in L0.

use crate::formalize::formalizer::{
    self, CnlFormalization, FormalizeError, FormalizeOptions, VerifyError,
};
use crate::formalize::graph::{GraphNode, NodeKind, Turn, Validated};
use crate::formalize::llm::{LlmClient, Transport};
use crate::lexicon::Lexicon;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The system prompt for CNL completion, ported from
/// `prompts/lemma_prover.md` and retargeted.
pub const CNL_COMPLETER_PROMPT: &str = include_str!("prompts/cnl_completer.md");

/// What to attempt, and how hard.
#[derive(Debug, Clone)]
pub struct CompleteOptions {
    pub max_retries: usize,
    /// Refute rather than prove. The result belongs in `solved_negation`, and
    /// ProofFlow's refutation prompt asks for "the logically equivalent negation"
    /// when the statement cannot be negated syntactically — which L0 always can,
    /// via `not`.
    pub prove_negation: bool,
}

impl Default for CompleteOptions {
    fn default() -> Self {
        CompleteOptions {
            max_retries: 3,
            prove_negation: false,
        }
    }
}

/// Why a node was not completed.
#[derive(Debug, Clone, PartialEq)]
pub enum CompleteError {
    /// The node is a condition or a definition: formalized, never proved.
    NotProvable {
        kind: NodeKind,
    },
    /// Nothing to complete — the formalizer never produced a sentence.
    NothingToComplete,
    /// The completer's own retry loop ran out.
    ///
    /// Distinct from [`CompleteError::NotProvable`] and from a formalizer
    /// failure: this node *had* a sentence and could not be finished.
    Failed {
        last: VerifyError,
        attempts: usize,
    },
    Llm(String),
}

impl fmt::Display for CompleteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompleteError::NotProvable { kind } => {
                write!(f, "{kind} nodes are formalized but never proved")
            }
            CompleteError::NothingToComplete => {
                write!(f, "no formalization to complete")
            }
            CompleteError::Failed { last, attempts } => write!(
                f,
                "did not reach a unique normal form in {attempts} attempts; last: {last}"
            ),
            CompleteError::Llm(m) => write!(f, "LLM call failed: {m}"),
        }
    }
}

impl std::error::Error for CompleteError {}

/// Build the completion turn for one node.
///
/// The shape is ProofFlow's `run_solver_prompt`: the statement, the current
/// sentence, and — when the sentence did not even compile — an explicit warning,
/// which is the analogue of its *"The previous Lean4 code I sent you contains
/// errors"* line.
pub fn build_complete_turn(
    node: &GraphNode,
    current: Option<&CnlFormalization>,
    failure: Option<&VerifyError>,
    opts: &CompleteOptions,
) -> Turn {
    let mut content = if opts.prove_negation {
        format!(
            "Your task is **not to formalize this statement, but to refute it**.\n\n\
             This is the statement I want you to disprove:\n{}\n\n\
             Emit the L0 CNL sentence denoting its negation. Use `not` — it is the \
             only negation in L0 — and do not refute by swapping names or \
             arguments, which asserts something unrelated.\n",
            node.statement
        )
    } else {
        format!(
            "This is the proof step I want you to formalize:\n{}\n\n\
             Repair the CNL sentence below so it reduces to a unique normal \
             form.\n",
            node.statement
        )
    };

    match current {
        Some(f) => {
            content.push_str(&format!(
                "\nCurrent CNL: `{}`\nIt reduces to: {}\n",
                f.cnl, f.readback
            ));
            if !f.verified {
                content.push_str(
                    "\nThat sentence does **not** currently reduce to a unique \
                     normal form, so it is not finished.\n",
                );
            }
        }
        None => content.push_str("\nThere is no current sentence. Start from scratch.\n"),
    }

    if let Some(error) = failure {
        content.push_str(&format!(
            "\nThe last attempt failed with:\n  {error}\n\n\
             Change the sentence according to that diagnosis. Emit exactly one \
             sentence inside a ```cnl fence.\n"
        ));
    }

    Turn::user(content)
}

/// Complete one node's sentence to a unique normal form.
///
/// Returns [`CompleteError::NotProvable`] for conditions and definitions, which
/// is the same early return as ProofFlow's `isinstance` guard, and
/// [`CompleteError::NothingToComplete`] when the formalizer produced nothing.
///
/// The retry loop itself is [`formalize_node_verified`]'s; this function decides
/// *whether* to run it and supplies the completion-specific prompt. Splitting it
/// that way keeps one retry loop in the pipeline rather than two that could
/// drift.
pub fn complete_node<T: Transport>(
    node: &GraphNode,
    formalization: Option<&CnlFormalization>,
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &CompleteOptions,
) -> Result<CnlFormalization, CompleteError> {
    let Some(kind) = node.kind() else {
        // Unreachable for a validated graph; treated as unprovable rather than
        // as a panic, since this function is reachable from the CLI.
        return Err(CompleteError::NotProvable {
            kind: NodeKind::Lemma,
        });
    };

    // Checked before anything else, so it also covers negation mode: refuting a
    // premise is not a goal, and `NotProvable` says so. ProofFlow guards with
    // the same `isinstance` test before branching on `prove_negation`.
    if !formalizer::is_provable(kind) {
        return Err(CompleteError::NotProvable { kind });
    }
    let Some(current) = formalization else {
        return Err(CompleteError::NothingToComplete);
    };

    // The completer repairs *this* sentence, so it seeds the loop with it rather
    // than re-asking from nothing: the whole point is an edit, not a resample.
    let turn = build_complete_turn(node, Some(current), None, opts);

    let repair = FormalizeOptions {
        max_retries: opts.max_retries,
        // Unused by the repair loop, which supplies its own seed turn; set
        // false so that if the loop is ever changed to build its own prompt it
        // will not silently start offering dependency CNL and inviting a copy.
        include_dependency_cnl: false,
    };

    match repair_until_unique(turn, llm, lexicon, &repair) {
        Ok(f) => Ok(f),
        Err(FormalizeError::NotVerified { last, attempts }) => Err(CompleteError::Failed {
            last,
            attempts: attempts.len().max(1),
        }),
        Err(FormalizeError::NoCnlBlock { .. }) => Err(CompleteError::Failed {
            last: VerifyError::NoParse {
                sentence: String::new(),
            },
            attempts: opts.max_retries,
        }),
        Err(FormalizeError::Llm(m)) => Err(CompleteError::Llm(m)),
        // A completion that lands on a dependency's identity is a real failure,
        // reported as one rather than being accepted as a rewrite.
        Err(other @ FormalizeError::DuplicateIdentity { .. }) => Err(CompleteError::Failed {
            last: VerifyError::NotConfluent {
                unf_hash: match &other {
                    FormalizeError::DuplicateIdentity { unf_hash, .. } => unf_hash.clone(),
                    _ => unreachable!("just matched DuplicateIdentity"),
                },
            },
            attempts: 1,
        }),
    }
}

/// Drive [`formalize_node_verified`] from a pre-built turn.
///
/// Drive the retry loop from a pre-built turn, demanding a unique normal form.
///
/// The seed turn is what makes this a *repair* loop rather than a fresh
/// generation loop: the formalizer's own `build_user_turn` would discard the
/// model's first, deliberately incomplete, attempt.
///
/// This is the formalizer's loop rather than a second copy of it. Reaching a
/// unique normal form is its `require_verified` mode, and duplicating the loop
/// would be the one thing worth avoiding here — two copies drift on exactly the
/// error-reporting details that make a retry a correction instead of a resample.
///
/// The identity check is deliberately absent: a repair is expected to change the
/// sentence, and comparing it against the dependency set would fire on the very
/// first attempt whenever the seed *was* correct. §14's identity violation is a
/// property of a *graph*, and that is where `score.rs` checks it, across every
/// node at once.
fn repair_until_unique<T: Transport>(
    seed: Turn,
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &FormalizeOptions,
) -> Result<CnlFormalization, FormalizeError> {
    let mut turns = vec![seed];
    let mut verify_failures: Vec<formalizer::FailedAttempt> = Vec::new();
    let mut replies: Vec<String> = Vec::new();
    let mut missing_block = 0usize;

    for attempt in 0..opts.max_retries {
        let reply = llm
            .chat(CNL_COMPLETER_PROMPT, &turns)
            .map_err(|e| FormalizeError::Llm(e.to_string()))?;
        replies.push(reply.clone());
        turns.push(Turn::assistant(&reply));

        let Some(cnl) = formalizer::extract_cnl_block(&reply) else {
            missing_block += 1;
            turns.push(Turn::user(
                "Error: no ```cnl fenced block found. Emit exactly one CNL \
                 sentence inside a ```cnl fence and nothing else.",
            ));
            continue;
        };

        let reduction = match formalizer::verify_cnl(&cnl, lexicon) {
            Ok(r) => r,
            Err(error) => {
                let failure = formalizer::FailedAttempt { cnl, error };
                turns.push(failure.correction_turn());
                verify_failures.push(failure);
                continue;
            }
        };

        if !reduction.is_verified() {
            let failure = formalizer::FailedAttempt {
                cnl: reduction.cnl.clone(),
                error: VerifyError::NotConfluent {
                    unf_hash: reduction.unf_hash.clone(),
                },
            };
            turns.push(failure.correction_turn());
            verify_failures.push(failure);
            continue;
        }

        return Ok(reduction.into_formalization(attempt + 1));
    }

    if verify_failures.is_empty() && missing_block > 0 {
        return Err(FormalizeError::NoCnlBlock { attempts: replies });
    }
    let last = verify_failures
        .last()
        .map(|f| f.error.clone())
        .ok_or_else(|| {
            FormalizeError::Llm(format!(
                "no attempt was made: max_retries was 0 ({} replies, {} missing blocks)",
                replies.len(),
                missing_block
            ))
        })?;
    Err(FormalizeError::NotVerified {
        last,
        attempts: verify_failures,
    })
}

/// What happened to a whole graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompleteReport {
    pub outcomes: Vec<NodeOutcomeReport>,
    pub completed: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// One node's row in a [`CompleteReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeOutcomeReport {
    pub id: String,
    pub kind: String,
    /// `"completed"`, `"skipped"`, or `"failed"`.
    pub status: String,
    /// The completed sentence's CNL, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cnl: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unf_hash: Option<String>,
    pub reason: String,
}

/// Run the completer over every node of a validated graph, writing results back.
///
/// `formalizations` supplies each node's current sentence — the formalizer's
/// output, which the caller already has in `node.formalization`. Results are
/// written to `formalization` when completing, and to `solved_negation` when
/// refuting, matching ProofFlow's two slots.
///
/// Nodes are completed in graph order, which `validate_proof_graph` has already
/// made topological: a step is only completed once its dependencies are, so a
/// later repair cannot invalidate an earlier node's identity.
pub fn complete_graph<T: Transport>(
    graph: &mut Validated,
    formalizations: &[(String, CnlFormalization)],
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &CompleteOptions,
) -> CompleteReport {
    let mut outcomes = Vec::with_capacity(graph.len());
    let (mut completed, mut skipped, mut failed) = (0, 0, 0);

    let by_id: std::collections::BTreeMap<&str, &CnlFormalization> = formalizations
        .iter()
        .map(|(id, f)| (id.as_str(), f))
        .collect();

    for i in 0..graph.len() {
        let node = graph.nodes[i].clone();
        let current = by_id.get(node.id.as_str()).copied();
        let result = complete_node(&node, current, llm, lexicon, opts);

        let kind = node.kind().unwrap_or(NodeKind::Lemma).to_string();
        let row = match result {
            Ok(f) => {
                completed += 1;
                let slot = if opts.prove_negation {
                    &mut graph.nodes[i].solved_negation
                } else {
                    &mut graph.nodes[i].formalization
                };
                *slot = Some(serde_json::to_value(&f).expect("a formalization serializes"));
                NodeOutcomeReport {
                    id: node.id.clone(),
                    kind,
                    status: "completed".into(),
                    cnl: Some(f.cnl.clone()),
                    unf_hash: Some(f.unf_hash.clone()),
                    reason: String::new(),
                }
            }
            Err(e @ CompleteError::NotProvable { .. })
            | Err(e @ CompleteError::NothingToComplete) => {
                skipped += 1;
                NodeOutcomeReport {
                    id: node.id.clone(),
                    kind,
                    status: "skipped".into(),
                    cnl: None,
                    unf_hash: None,
                    reason: e.to_string(),
                }
            }
            Err(e) => {
                failed += 1;
                NodeOutcomeReport {
                    id: node.id.clone(),
                    kind,
                    status: "failed".into(),
                    cnl: None,
                    unf_hash: None,
                    reason: e.to_string(),
                }
            }
        };
        outcomes.push(row);
    }

    CompleteReport {
        outcomes,
        completed,
        skipped,
        failed,
    }
}

/// True when every node that *can* be proved has been.
///
/// The report's completeness condition, so a caller does not have to re-derive
/// which nodes are legitimately skipped.
pub fn is_complete(report: &CompleteReport) -> bool {
    report.failed == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::formalizer::{base_lexicon, verify_cnl};
    use crate::formalize::llm::{LlmConfig, ScriptedTransport};
    use std::time::Duration;

    fn cfg() -> LlmConfig {
        LlmConfig {
            base_url: "http://local/v1".into(),
            model: "test".into(),
            api_key: None,
            temperature: 0.0,
            max_tokens: 64,
            timeout: Duration::from_secs(1),
        }
    }

    fn say(cnl: &str) -> String {
        format!("```cnl\n{cnl}\n```")
    }

    fn lex() -> Lexicon {
        base_lexicon()
    }

    fn formal(cnl: &str) -> CnlFormalization {
        verify_cnl(cnl, &lex())
            .unwrap_or_else(|e| panic!("{cnl} should verify: {e}"))
            .into_formalization(1)
    }

    fn node(id: &str, statement: &str, deps: &[&str]) -> GraphNode {
        GraphNode::new(id, "quote", statement, deps.iter().map(|d| d.to_string()))
    }

    // ── the skip rules ProofFlow also applies ───────────────────────────────

    #[test]
    fn conditions_and_definitions_are_skipped() {
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("John runs")]), cfg());
        for (id, kind) in [
            ("tc_1", NodeKind::TheoremCondition),
            ("def_1", NodeKind::Definition),
        ] {
            let n = node(id, "an assumption", &[]);
            let err = complete_node(
                &n,
                Some(&formal("John runs")),
                &mut llm,
                &lex(),
                &CompleteOptions::default(),
            )
            .unwrap_err();
            assert_eq!(
                err,
                CompleteError::NotProvable { kind },
                "{id} should be unprovable"
            );
        }
    }

    #[test]
    fn a_node_without_a_formalization_is_skipped() {
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("John runs")]), cfg());
        let n = node("l1", "a step", &[]);
        let err =
            complete_node(&n, None, &mut llm, &lex(), &CompleteOptions::default()).unwrap_err();
        assert_eq!(err, CompleteError::NothingToComplete);
    }

    /// Refuting a premise is not a goal, and that is caught by the *same* guard
    /// ProofFlow applies before it branches on `prove_negation` — so negation
    /// mode skips assumptions rather than attempting them.
    #[test]
    fn negation_mode_skips_assumptions_too() {
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("John runs")]), cfg());
        let n = node("tc_1", "an assumption", &[]);
        let opts = CompleteOptions {
            prove_negation: true,
            ..CompleteOptions::default()
        };
        let err =
            complete_node(&n, Some(&formal("John runs")), &mut llm, &lex(), &opts).unwrap_err();
        assert_eq!(
            err,
            CompleteError::NotProvable {
                kind: NodeKind::TheoremCondition
            }
        );
    }

    // ── the repair loop ─────────────────────────────────────────────────────

    #[test]
    fn an_already_unique_sentence_completes_in_one_attempt() {
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("Bob runs")]), cfg());
        let n = node("l1", "Bob runs", &[]);
        let out = complete_node(
            &n,
            Some(&formal("John runs")),
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        )
        .unwrap();
        assert_eq!(out.cnl, "Bob runs");
        assert!(out.verified);
        assert_eq!(out.tries, 1);
    }

    /// The completion turn must carry the current sentence and its normal form —
    /// that is what makes the loop a repair rather than a resample.
    #[test]
    fn the_completion_turn_carries_the_current_sentence() {
        let f = formal("John runs");
        let turn = build_complete_turn(
            &node("l1", "Bob runs", &[]),
            Some(&f),
            None,
            &CompleteOptions::default(),
        );
        assert!(turn.content.contains("John runs"), "{}", turn.content);
        assert!(turn.content.contains("Run(john)"), "{}", turn.content);
    }

    #[test]
    fn an_unfinished_sentence_is_flagged_as_unfinished() {
        let mut f = formal("John runs");
        f.verified = false;
        let turn = build_complete_turn(
            &node("l1", "s", &[]),
            Some(&f),
            None,
            &CompleteOptions::default(),
        );
        assert!(
            turn.content.contains("does **not** currently reduce"),
            "{}",
            turn.content
        );
    }

    #[test]
    fn the_diagnosis_is_shown_to_the_model() {
        let turn = build_complete_turn(
            &node("l1", "s", &[]),
            Some(&formal("John runs")),
            Some(&VerifyError::OutOfLexicon {
                sentence: "x".into(),
                words: vec!["congruences".into()],
            }),
            &CompleteOptions::default(),
        );
        assert!(turn.content.contains("congruences"), "{}", turn.content);
    }

    #[test]
    fn negation_mode_asks_for_a_refutation() {
        let turn = build_complete_turn(
            &node("ts_1", "Mary sees Bob", &[]),
            Some(&formal("Mary sees Bob")),
            None,
            &CompleteOptions {
                prove_negation: true,
                ..CompleteOptions::default()
            },
        );
        assert!(turn.content.contains("disprove"), "{}", turn.content);
        assert!(turn.content.contains("not"), "{}", turn.content);
    }

    /// A repair that lands on a rejected sentence is reported, not accepted.
    #[test]
    fn exhausting_retries_reports_the_typed_failure() {
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec![say("Euler proves congruences")]),
            cfg(),
        );
        let err = complete_node(
            &node("l1", "s", &[]),
            Some(&formal("John runs")),
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        )
        .unwrap_err();
        match err {
            CompleteError::Failed { last, attempts } => {
                assert_eq!(attempts, 3);
                assert!(matches!(last, VerifyError::OutOfLexicon { .. }));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn a_model_that_stops_using_the_fence_is_reported() {
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec!["Bob runs, I think".into()]),
            cfg(),
        );
        let err = complete_node(
            &node("l1", "s", &[]),
            Some(&formal("John runs")),
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, CompleteError::Failed { .. }), "{err:?}");
    }

    #[test]
    fn a_transport_failure_propagates() {
        struct Broken;
        impl Transport for Broken {
            fn post_chat(
                &mut self,
                _: &LlmConfig,
                _: &str,
                _: &[Turn],
            ) -> Result<String, crate::formalize::llm::LlmError> {
                Err(crate::formalize::llm::LlmError::Transport("down".into()))
            }
        }
        let mut llm = LlmClient::new(Broken, cfg());
        let err = complete_node(
            &node("l1", "s", &[]),
            Some(&formal("John runs")),
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, CompleteError::Llm(_)), "{err:?}");
    }

    // ── the whole graph ─────────────────────────────────────────────────────

    fn euler_graph() -> Validated {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../testdata/formalize/euler_proof_graph.json"
        ))
        .unwrap();
        crate::formalize::graph::validate_proof_graph(&data).unwrap()
    }

    /// Three conditions/definitions to skip, five lemmas to complete, and one
    /// theorem statement — so the skip rule and the topological order are both
    /// exercised end to end.
    #[test]
    fn completing_a_graph_skips_assumptions_and_finishes_the_rest() {
        let mut graph = euler_graph();
        let sentences = [
            "John loves Mary",
            "Mary sees Bob",
            "Alice likes John",
            "Bob sees Alice",
            "Mary sees Alice",
            "Alice likes Mary",
            "Bob likes John",
            "Alice sees Mary",
            "John runs",
        ];
        let existing: Vec<(String, CnlFormalization)> = sentences
            .iter()
            .enumerate()
            .map(|(i, c)| (graph.nodes[i].id.clone(), formal(c)))
            .collect();
        let mut llm = LlmClient::new(
            ScriptedTransport::new(sentences.iter().map(|s| say(s)).collect()),
            cfg(),
        );

        let report = complete_graph(
            &mut graph,
            &existing,
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        );

        assert_eq!(report.completed, 6, "5 lemmas + 1 ts: {report:?}");
        assert_eq!(report.skipped, 3, "tc_1 + def_1 + def_2");
        assert_eq!(report.failed, 0);
        assert!(is_complete(&report));
        assert_eq!(report.outcomes.len(), 9);

        // Skipped rows say why; completed rows carry a hash.
        for row in &report.outcomes {
            match row.status.as_str() {
                "skipped" => assert!(row.reason.contains("never proved"), "{row:?}"),
                "completed" => {
                    assert!(row.cnl.is_some() && row.unf_hash.is_some(), "{row:?}");
                    assert!(row.reason.is_empty());
                }
                other => panic!("unexpected status {other}"),
            }
        }
        // Results are written back onto the graph.
        assert!(
            graph
                .nodes
                .iter()
                .filter(|n| n.formalization.is_some())
                .count()
                >= 6
        );
    }

    #[test]
    fn refuting_a_graph_writes_the_negation_slot() {
        let mut graph = euler_graph();
        // One formalization per node, so the three assumptions are *skipped* and
        // every lemma and the theorem statement is attempted.
        let sentences = [
            "John loves Mary",
            "Mary sees Bob",
            "Alice likes John",
            "Bob sees Alice",
            "Mary sees Alice",
            "Alice likes Mary",
            "Bob likes John",
            "Alice sees Mary",
            "John runs",
        ];
        let existing: Vec<(String, CnlFormalization)> = sentences
            .iter()
            .enumerate()
            .map(|(i, c)| (graph.nodes[i].id.clone(), formal(c)))
            .collect();
        let mut llm = LlmClient::new(
            ScriptedTransport::new(sentences.iter().map(|s| say(s)).collect()),
            cfg(),
        );
        let opts = CompleteOptions {
            prove_negation: true,
            ..CompleteOptions::default()
        };
        let report = complete_graph(&mut graph, &existing, &mut llm, &lex(), &opts);
        assert_eq!(report.completed, 6, "5 lemmas + ts_1: {report:?}");
        assert_eq!(report.skipped, 3, "tc_1 + def_1 + def_2");
        // Results land in `solved_negation`, not in `formalization`.
        assert!(graph.nodes[3].solved_negation.is_some());
        assert!(graph.nodes[3].formalization.is_none());
    }

    #[test]
    fn a_failing_node_makes_the_report_incomplete() {
        let mut graph = euler_graph();
        let sentences = [
            "John loves Mary",
            "Mary sees Bob",
            "Alice likes John",
            "Bob sees Alice",
            "Mary sees Alice",
            "Alice likes Mary",
            "Bob likes John",
            "Alice sees Mary",
            "John runs",
        ];
        let existing: Vec<(String, CnlFormalization)> = sentences
            .iter()
            .enumerate()
            .map(|(i, c)| (graph.nodes[i].id.clone(), formal(c)))
            .collect();
        // Every repair is unparsable, so all six provable nodes fail.
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec![say("Euler proves congruences")]),
            cfg(),
        );
        let report = complete_graph(
            &mut graph,
            &existing,
            &mut llm,
            &lex(),
            &CompleteOptions::default(),
        );
        assert_eq!(report.failed, 6, "{report:?}");
        assert!(!is_complete(&report));
    }
}
