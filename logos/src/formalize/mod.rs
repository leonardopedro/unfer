//! P2–P8 — NL → logos/ControlledNaturalLanguage autoformalization.
//!
//! This is the ProofFlow adaptation of the unfer rewrite plan (`§11`–`§17`):
//! the same dependency-graph pipeline ProofFlow runs against Lean 4, retargeted
//! at logos' L0 CNL, with `deltanet` reduction as the verification substrate
//! instead of a Lean server.
//!
//! | stage | module | ProofFlow original |
//! |---|---|---|
//! | graph build + DAG check | [`graph`] | `proof_graph.py` |
//! | per-node CNL generation | `formalizer` | `proof_formalize.py` |
//! | tactic-completer analogue | `completer` | `proof_prover.py` |
//! | Lean compiler oracle | the `deltanet` pipeline | `lean_check.py` |
//! | ProofScore | `score` | `proof_scorer.py` |
//! | — | `memory` | — (new: engram-backed lemma store) |
//! | — | `llm` | `utils.LLMManager` |
//!
//! The substitution that drives all of it: **a node's identity is its UNF
//! hash**, not its name. ProofFlow identifies a step by the string `l4`; here
//! two CNL sentences that denotate the same normal form *are* the same node,
//! which is what makes the engram lemma store (§16 P3) and the readback
//! round-trip in [`score`] content-addressed rather than string-matched.

pub mod formalizer;
pub mod graph;
pub mod llm;
