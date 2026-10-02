//! P2 — per-node CNL generation, verified by reduction.
//!
//! # What replaces ProofFlow's Lean server
//!
//! ProofFlow's verification substrate is a Lean 4 process
//! (`lean_check.LeanServer`). Two of its results are load-bearing, and each has
//! a direct analogue here:
//!
//! | Lean | meaning | CNL analogue | meaning |
//! |---|---|---|---|
//! | `lean_pass` | the skeleton **compiles** (a `sorry` is allowed) | [`Reduction`] | the sentence gates, parses, and compiles to a net |
//! | `lean_verify` | it compiles **with no `sorry`** | [`Reduction::verified`] | the net reduces to a UNF and a second reduction agrees |
//!
//! That split is why P2 and P4 are separate tasks. ProofFlow's formalizer (P2)
//! only needs `lean_pass`, because the emitted Lean has a `sorry` and is
//! finished off by the tactic completer (P4) which needs `lean_verify`. So
//! [`verify_cnl`] reports both, the formalizer loop stops at `compiles`, and
//! `formalize::completer` (P4) stops at `verified`.
//!
//! The pipeline is the one `prob_kernel::logos::logos_compile` runs, in the same
//! order: gate → CCG parse → CoreIR → linearity → net → reduce → readback →
//! hash. The last four steps are [`crate::translate::translate_coreir`], which
//! is reused rather than re-implemented — the plan's "verify-per-node" is that
//! same call, and a second implementation of reduction would be a second thing
//! that could disagree with the kernel.
//!
//! # The honest limit, stated once
//!
//! L0 is a 46-word toy language ([`BASE_LEXICON_TSV`]). Stock L0 will not
//! formalize a real proof — plan §17.1 calls this the top risk. The machinery
//! below is real; the *coverage* comes from [`with_domain_lexicon`], which
//! extends the lexicon per domain. An un-lexiconable node is reported as a
//! [`VerifyError::NoParse`] with the failing words named, never silently
//! dropped.

use crate::ccg;
use crate::core_ir;
use crate::deltanet;
use crate::formalize::graph::{GraphNode, NodeKind, Role, Turn};
use crate::formalize::llm::{CNL_FORMALIZER_PROMPT, LlmClient, Transport};
use crate::harper_gate::HarperGate;
use crate::lexicon::{LexEntry, Lexicon};
use crate::translate;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The L0 lexicon, embedded so `formalize` works as a library.
///
/// ProofFlow's Python resolves such paths relative to the working directory
/// (`logos/src/cli/mod.rs::find_lexicon` does the same in Rust). Embedding is
/// the better default for a pipeline that will also run inside a kernel process
/// with no CWD of its own — and [`Lexicon::load`] still exists for callers that
/// genuinely want a file.
pub const BASE_LEXICON_TSV: &str = include_str!("../../corpus/lexicon.tsv");

/// The base L0 lexicon.
pub fn base_lexicon() -> Lexicon {
    Lexicon::parse(BASE_LEXICON_TSV).expect("the embedded base lexicon is well-formed")
}

/// The base L0 lexicon extended by a domain TSV in the same format.
///
/// This is plan §17.1's "stage the grammar extension": the L0 *grammar* is
/// fixed, but the lexicon is data, so a new domain costs a TSV and not a
/// grammar change. Words in the extension are appended, so
/// [`Lexicon::lookup`] reports both readings for a homonym and
/// `semantic_template` keeps the base one — the extension adds vocabulary, it
/// does not silently redefine the core language.
///
/// The result carries a flag recording that it is not stock L0, because a score
/// computed against an extended lexicon is not comparable with one computed
/// against base L0.
///
/// # An extension may only add *words*, not *constructors*
///
/// [`with_extension`] rejects a TSV that introduces a `Con("Name", …)` the
/// compiler does not know. That is a real and easily-missed restriction:
/// `core_ir::compiler::tag_id` maps an unrecognized constructor to tag **0**,
/// which is a legal tag, so the constructor's *name is discarded* and the term
/// reads back as `Unknown(...)`. Two different unknown constructors of the same
/// arity would then compile to the same term and therefore to the same UNF hash
/// — a silent identity collapse, which under §14 means two distinct proof steps
/// would compare equal with no diagnostic anywhere.
///
/// Rejecting at construction time turns that into a loud failure. Lifting it
/// properly needs a name-to-tag registry that interns new constructors
/// deterministically; that is a change to `core_ir`, not to this pipeline.
#[derive(Debug, Clone)]
pub struct DomainLexicon {
    lexicon: Lexicon,
    extended: bool,
}

impl DomainLexicon {
    /// Stock L0.
    pub fn base() -> Self {
        DomainLexicon {
            lexicon: base_lexicon(),
            extended: false,
        }
    }

    /// L0 plus a domain TSV (same three columns as the base lexicon).
    ///
    /// # Errors
    ///
    /// [`LexiconError`] for malformed TSV, or [`LexiconError::UnknownConstructor`]
    /// when the extension names a constructor the compiler cannot encode.
    pub fn with_extension(tsv: &str) -> Result<Self, LexiconError> {
        let lexicon = Lexicon::parse(&format!("{BASE_LEXICON_TSV}\n{tsv}"))?;
        let unknown = unknown_constructors(&lexicon);
        if !unknown.is_empty() {
            return Err(LexiconError::UnknownConstructor {
                names: unknown,
                known: core_ir::BUILTIN_CONSTRUCTORS.len(),
            });
        }
        Ok(DomainLexicon {
            lexicon,
            extended: true,
        })
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    /// Whether a domain extension is in effect.
    pub fn is_extended(&self) -> bool {
        self.extended
    }

    pub fn word_count(&self) -> usize {
        self.lexicon.word_count()
    }
}

impl Default for DomainLexicon {
    fn default() -> Self {
        Self::base()
    }
}

/// Only the base lexicon's own failure modes can surface here.
pub type LexiconError = crate::lexicon::LexiconError;

/// Every `Con` name a lexicon can produce that `core_ir` cannot encode.
///
/// Collected with `BTreeSet` so the error names the offenders in a stable order
/// regardless of the entry order in the TSV — the error is a diagnostic, and a
/// diagnostic that reorders itself between runs is hard to diff.
pub fn unknown_constructors(lexicon: &Lexicon) -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    for entry in lexicon.entries() {
        collect_cons(&entry.template, &mut out);
    }
    out.into_iter()
        .filter(|name| !core_ir::known_constructor(name))
        .collect()
}

fn collect_cons(expr: &crate::lexicon::SemExpr, out: &mut std::collections::BTreeSet<String>) {
    use crate::lexicon::SemExpr;
    match expr {
        SemExpr::Con(name, args) => {
            out.insert(name.clone());
            for arg in args {
                collect_cons(arg, out);
            }
        }
        SemExpr::Lam(_, body) => collect_cons(body, out),
        SemExpr::App(f, a) => {
            collect_cons(f, out);
            collect_cons(a, out);
        }
        SemExpr::Var(_) | SemExpr::Lit(_) => {}
    }
}

/// One node's verified CNL sentence.
///
/// This is the typed shape that goes into [`GraphNode::formalization`].
///
/// `unf_hash` is the node's identity — the substitution in plan §14 that
/// replaces ProofFlow's "node identity = name". It is a SHA-256 over the
/// canonical net serialization, so two CNL sentences that denotate the same
/// term hash the same, which is what makes P3's engram store content-addressed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CnlFormalization {
    /// The L0 sentence, as emitted.
    pub cnl: String,
    /// `deltanet::readback` of the reduced net — the human-readable normal form.
    pub readback: String,
    /// Content-addressable UNF digest: the node's canonical identity.
    pub unf_hash: String,
    /// The confluence self-check: a second, independent reduction of the same
    /// sentence reproduced the identical UNF.
    pub verified: bool,
    /// The closed numerical value, when the term has no unknowns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// The canonical polynomial normal form, when the term is in the Int64
    /// arithmetic fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ted: Option<String>,
    /// How many attempts produced this.
    pub tries: usize,
}

impl CnlFormalization {
    /// True iff the sentence both compiled and passed the confluence check.
    ///
    /// This is the `lean_verify` analogue: the completer (P4) is the stage that
    /// drives this to `true`.
    pub fn is_verified(&self) -> bool {
        self.verified
    }
}

/// Why a sentence did not verify.
///
/// The variants are the pipeline's stages, kept separate because P4's completer
/// reacts to them differently: a [`VerifyError::GateRejected`] or
/// [`VerifyError::OutOfLexicon`] is fixed by *changing the words*, while
/// [`VerifyError::NotConfluent`] is a reduction-identity failure that no rewrite
/// fixes and that must never be hashed (§17.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// `harper_gate` rejected the sentence outright.
    GateRejected(Vec<String>),
    /// A word is not in the lexicon. Reported with the offending words, because
    /// "no parse" with no detail is the single least actionable message in the
    /// pipeline — §17.1's whole risk is lexicon coverage, so the coverage gap
    /// has to be visible.
    OutOfLexicon {
        sentence: String,
        words: Vec<String>,
    },
    /// The words are all known but no CCG derivation exists.
    NoParse {
        sentence: String,
    },
    CoreIr(String),
    Net(String),
    Reduce(String),
    Readback(String),
    /// Two reductions of the same term disagreed — no unique normal form.
    ///
    /// Plan §17.4: this blocks identity. The pipeline must surface it as a node
    /// error and must **not** fall back to the `unf_hash` of a non-unique term.
    NotConfluent {
        unf_hash: String,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::GateRejected(errors) => {
                write!(
                    f,
                    "grammar gate rejected the sentence: {}",
                    errors.join("; ")
                )
            }
            VerifyError::OutOfLexicon { words, .. } => {
                write!(f, "words not in the lexicon: {}", words.join(", "))
            }
            VerifyError::NoParse { sentence } => {
                write!(f, "no CCG derivation for CNL sentence: {sentence}")
            }
            VerifyError::CoreIr(m) => write!(f, "core-IR compile failed: {m}"),
            VerifyError::Net(m) => write!(f, "net compile failed: {m}"),
            VerifyError::Reduce(m) => write!(f, "reduction failed: {m}"),
            VerifyError::Readback(m) => write!(f, "readback failed: {m}"),
            VerifyError::NotConfluent { unf_hash } => write!(
                f,
                "no unique normal form: two reductions of the same sentence \
                 disagreed (hash {unf_hash})"
            ),
        }
    }
}

impl std::error::Error for VerifyError {}

/// A node's verified reduction — what `lean_pass` + `lean_verify` become.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    pub cnl: String,
    pub readback: String,
    pub unf_hash: String,
    /// The confluence self-check.
    pub verified: bool,
    pub value: Option<String>,
    pub ted: Option<String>,
}

impl Reduction {
    /// The `lean_pass` analogue: the sentence compiled and reduced at all.
    pub fn compiles(&self) -> bool {
        true
    }

    /// The `lean_verify` analogue: it also reduced *uniquely*.
    pub fn is_verified(&self) -> bool {
        self.verified
    }

    pub fn into_formalization(self, tries: usize) -> CnlFormalization {
        CnlFormalization {
            cnl: self.cnl,
            readback: self.readback,
            unf_hash: self.unf_hash,
            verified: self.verified,
            value: self.value,
            ted: self.ted,
            tries,
        }
    }
}

/// Run the CNL pipeline over one sentence.
///
/// `gate → CCG parse → CoreIR → linearity → net → reduce → readback → hash`,
/// the same chain as `prob_kernel::logos::logos_compile`. `translate_coreir`
/// performs the last five steps and returns `verified` as its double-reduction
/// confluence check, so nothing here re-implements reduction.
///
/// Note the second call: the confluence check inside `translate_coreir` is
/// enough, and re-reducing here as well would be the third reduction of the
/// same sentence for no additional signal.
pub fn verify_cnl(sentence: &str, lexicon: &Lexicon) -> Result<Reduction, VerifyError> {
    let gate = HarperGate::new().lint(sentence);
    if !gate.accepted {
        return Err(VerifyError::GateRejected(gate.errors));
    }

    // The gate's tokenizer is authoritative: it is the tokenization that was
    // accepted, so the parse must see the same tokens rather than a second,
    // possibly different, whitespace split. (This is the same reasoning
    // `engram::segment` uses.)
    let tokens: Vec<String> = gate.tokens.into_iter().map(|t| t.text).collect();

    let oov: Vec<String> = tokens
        .iter()
        .filter(|t| lexicon.lookup(t).is_empty())
        .cloned()
        .collect();
    if !oov.is_empty() {
        return Err(VerifyError::OutOfLexicon {
            sentence: sentence.to_string(),
            words: oov,
        });
    }

    let trees = ccg::parse_sentence(&tokens, lexicon);
    let tree = trees.first().ok_or_else(|| VerifyError::NoParse {
        sentence: sentence.to_string(),
    })?;

    let ir = core_ir::compile_to_core_ir(tree, lexicon)
        .map_err(|e| VerifyError::CoreIr(e.to_string()))?;
    // Linearity insertion is a validation step, not part of identity: reduction
    // is confluent, so inserting explicit closures cannot change the net's
    // canonical form. (`logos::engram::segment` hashes with it inserted and
    // `prob_kernel::logos::compile_with` hashes without, and both agree — the
    // first two sentences of `unf_hash_agrees_with_and_without_linearity` pin
    // that, because a pipeline whose UNF hash depended on this line would make
    // every engram key unreachable from the kernel's report.)
    let ir = core_ir::linearity::insert_linearity(ir);

    // The net is built here rather than taken from `translate_coreir`, for one
    // reason: `deltanet::readback` is the only call that yields the *readable*
    // normal form (`Love(john, mary)`), where `SymExpr::to_prefix_string` gives
    // the compiler's interned constructor tag (`Tag1(john, mary)`). P4's scorer
    // and P7's `kernel_client` path both compare against `LogosReport.result`,
    // so the readable form is the one that has to be in the report. The four
    // steps below are `prob_kernel::logos::compile_with`, in its order.
    let mut net = deltanet::compile_to_net(&ir).map_err(|e| VerifyError::Net(e.to_string()))?;
    deltanet::reduce(&mut net).map_err(|e| VerifyError::Reduce(e.to_string()))?;

    let readback = deltanet::readback(&net).map_err(|e| VerifyError::Readback(e.to_string()))?;
    let unf_hash =
        deltanet::unf_hash_string(&net).map_err(|e| VerifyError::Readback(e.to_string()))?;

    // One extra reduction pair, bought for the confluence self-check plus the
    // TED and closed-value readouts. Per-node cost is irrelevant next to the
    // LLM round trip that produced the sentence.
    let unf = translate::translate_coreir(&ir).map_err(|e| VerifyError::Reduce(e.to_string()))?;

    debug_assert_eq!(
        unf.unf_hash, unf_hash,
        "two reduction paths disagreed on the UNF hash"
    );

    Ok(Reduction {
        cnl: sentence.to_string(),
        readback,
        unf_hash,
        verified: unf.verified,
        value: unf.value.map(|v| v.to_string()),
        ted: unf.ted_string,
    })
}

// ── the per-node formalization loop ─────────────────────────────────────────

/// How many attempts, and what to put in the prompt.
#[derive(Debug, Clone)]
pub struct FormalizeOptions {
    pub max_retries: usize,
    /// `prompts/lemma_formalizer.md`'s `previous_context`: when true, a
    /// dependency's already-reduced CNL is included so the model can vary the
    /// node against its siblings.
    ///
    /// Off by default. §14 makes the UNF hash the node identity, so handing the
    /// model its dependencies' sentences is what keeps sibling nodes from
    /// collapsing onto one hash — but it also invites the model to *copy* a
    /// dependency, which is a silent identity bug. The retry loop detects a
    /// duplicate hash and reports it; this flag controls whether the model is
    /// even given the material.
    pub include_dependency_cnl: bool,
}

impl Default for FormalizeOptions {
    fn default() -> Self {
        FormalizeOptions {
            max_retries: 3,
            include_dependency_cnl: false,
        }
    }
}

/// Everything that can stop a node from being formalized.
#[derive(Debug, Clone, PartialEq)]
pub enum FormalizeError {
    /// No reply contained a ```cnl block in any attempt.
    NoCnlBlock { attempts: Vec<String> },
    /// Every attempt's sentence failed verification; the last error is reported
    /// because it is the one the next model turn is built from.
    NotVerified {
        last: VerifyError,
        attempts: Vec<FailedAttempt>,
    },
    /// The model produced a sentence whose UNF hash equals a dependency's.
    ///
    /// Not a ProofFlow failure mode — it has no content-addressed identity — but
    /// the direct consequence of §14: two nodes that denote the same term are
    /// the same node, so a "formalization" that collapses onto its parent is
    /// not a formalization of anything.
    DuplicateIdentity {
        cnl: String,
        unf_hash: String,
        dependency: String,
    },
    /// The transport failed.
    Llm(String),
}

impl fmt::Display for FormalizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormalizeError::NoCnlBlock { attempts } => write!(
                f,
                "no ```cnl block in any of {} attempts (got {})",
                attempts.len(),
                attempts.len()
            ),
            FormalizeError::NotVerified { last, attempts } => write!(
                f,
                "none of {} attempts verified; last: {last}",
                attempts.len()
            ),
            FormalizeError::DuplicateIdentity {
                cnl,
                unf_hash,
                dependency,
            } => write!(
                f,
                "CNL {cnl:?} reduced to the UNF of dependency '{dependency}' \
                 (hash {unf_hash}); a node must denote its own term"
            ),
            FormalizeError::Llm(m) => write!(f, "LLM call failed: {m}"),
        }
    }
}

impl std::error::Error for FormalizeError {}

/// One rejected attempt, kept so a report can show the model's trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct FailedAttempt {
    pub cnl: String,
    pub error: VerifyError,
}

impl FailedAttempt {
    /// The correction turn fed back to the model.
    ///
    /// ProofFlow's analogue is `"Lean error: …\n\nBased on the error, please
    /// correct the previous response."`, and the shape is kept deliberately:
    /// the pipeline reads as a Lean-transcript diff only in the one string that
    /// is worth diffing.
    pub fn correction_turn(&self) -> Turn {
        Turn::user(format!(
            "CNL error: {}\n\nBased on the error, please correct the previous \
             response. Emit exactly one sentence inside a ```cnl fence.",
            self.error
        ))
    }
}

/// Extract the **last** ```cnl block from a reply.
///
/// "Last", not "first": ProofFlow's `extract_code_validate` takes
/// `matches[-1]`, and a model that thinks out loud will emit a scratch sentence
/// before its answer.
pub fn extract_cnl_block(text: &str) -> Option<String> {
    let mut found = None;
    let mut cursor = 0usize;
    while let Some(offset) = text[cursor..].find("```") {
        let open = cursor + offset;
        let after = &text[open + 3..];
        let nl = after.find('\n');
        let body_start = match nl {
            Some(k) => open + 3 + k + 1,
            None => open + 3,
        };
        let close = match text[body_start..].find("```") {
            Some(k) => body_start + k,
            None => break,
        };
        let body = &text[body_start..close];
        let (info, body) = match nl {
            Some(k) => (after[..k].trim(), body),
            None => {
                // Single-line fence (`` ```cnl John loves Mary``` ``): there is
                // no newline to end the info string, so it ends at the first
                // whitespace and the rest is the body.
                let seg = &after[..close - (open + 3)];
                match seg.find(char::is_whitespace) {
                    Some(k) => (seg[..k].trim(), &seg[k..]),
                    None => (seg.trim(), ""),
                }
            }
        };
        if info.eq_ignore_ascii_case("cnl") {
            found = Some(body.trim().to_string());
        }
        cursor = close + 3;
    }
    found
}

/// Build the user turn for one node.
///
/// ProofFlow's `run_formalizer_prompt` prepends the node's statement and its
/// dependency ids, then appends each dependency's *code* ("Here is the natural
/// language statement of step …" as the fallback when the code is missing). The
/// same shape is used, with the CNL sentence standing in for the Lean code.
pub fn build_user_turn(
    node: &GraphNode,
    dependency_formalizations: &[(String, CnlFormalization)],
    opts: &FormalizeOptions,
) -> Turn {
    let mut content = format!(
        "Please formalize the following proof step as one L0 CNL sentence.\n\
         Use this node id: {}\n\
         The natural-language statement is: {}\n\
         The dependencies are: {:?}\n",
        node.id, node.statement, node.dependencies
    );

    if opts.include_dependency_cnl && !dependency_formalizations.is_empty() {
        content.push_str(
            "\nThis step depends on previous steps. Their CNL sentences and \
             reduced normal forms are given below. Your sentence must denote a \
             *different* term from each of them — node identity is the UNF hash, \
             so a repeated sentence is a repeated node.\n",
        );
        for (id, f) in dependency_formalizations {
            content.push_str(&format!(
                "\nStep {id}: CNL {:?} reduces to {} ({})\n",
                f.cnl, f.readback, f.unf_hash
            ));
        }
    } else if !node.dependencies.is_empty() {
        content.push_str(&format!(
            "\nNote: dependencies {:?} already exist. Do not restate them; \
             your sentence must denote this step's own new fact.\n",
            node.dependencies
        ));
    }

    Turn::user(content)
}

/// Formalize one node: ask, extract, verify, and feed the typed error back.
///
/// The loop is ProofFlow's `run_formalizer_prompt` with the Lean server swapped
/// for [`verify_cnl`]. Two details are kept from the original because they are
/// what make a retry a *correction* rather than a resample:
///
/// - the model reply is appended to the transcript by the client, so the next
///   attempt sees its own failed sentence;
/// - the correction turn carries the **typed** failure, so an out-of-lexicon
///   word is named rather than summarized as "Lean error: <unknown>".
///
/// The loop stops at `compiles`, not `verified`: P4's completer owns the
/// reduction-identity half. Passing `require_verified` moves the stop condition,
/// which is how the completer reuses this function.
pub fn formalize_node<T: Transport>(
    node: &GraphNode,
    dependency_formalizations: &[(String, CnlFormalization)],
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &FormalizeOptions,
) -> Result<CnlFormalization, FormalizeError> {
    formalize_node_inner(
        node,
        dependency_formalizations,
        llm,
        lexicon,
        opts,
        // `lean_pass`: compiles. The completer asks for `verified`.
        false,
    )
}

/// As [`formalize_node`], but demanding the confluence check pass too.
///
/// This is the `lean_verify` stop condition, used by P4's completer.
pub fn formalize_node_verified<T: Transport>(
    node: &GraphNode,
    dependency_formalizations: &[(String, CnlFormalization)],
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &FormalizeOptions,
) -> Result<CnlFormalization, FormalizeError> {
    formalize_node_inner(node, dependency_formalizations, llm, lexicon, opts, true)
}

fn formalize_node_inner<T: Transport>(
    node: &GraphNode,
    dependency_formalizations: &[(String, CnlFormalization)],
    llm: &mut LlmClient<T>,
    lexicon: &Lexicon,
    opts: &FormalizeOptions,
    require_verified: bool,
) -> Result<CnlFormalization, FormalizeError> {
    let mut turns = vec![build_user_turn(node, dependency_formalizations, opts)];
    // `verify_failures` records attempts that produced a sentence which failed
    // verification; `missing_block` counts attempts that produced no sentence at
    // all. The two are tracked apart because they need different fixes, and
    // because "every attempt was unfenced" is a *different* failure from "every
    // sentence was wrong" — ProofFlow conflates them because
    // `extract_code_validate` raises `ValueError` for a missing block and the
    // loop treats that as just another attempt, so a model that never uses the
    // fence reports identically to one whose sentences never compile.
    let mut verify_failures: Vec<FailedAttempt> = Vec::new();
    let mut replies: Vec<String> = Vec::new();
    let mut missing_block = 0usize;

    for attempt in 0..opts.max_retries {
        let reply = llm
            .chat(CNL_FORMALIZER_PROMPT, &turns)
            .map_err(|e| FormalizeError::Llm(e.to_string()))?;
        replies.push(reply.clone());
        turns.push(Turn::assistant(&reply));

        let Some(cnl) = extract_cnl_block(&reply) else {
            missing_block += 1;
            turns.push(Turn::user(
                "Error: no ```cnl fenced block found. Emit exactly one CNL \
                 sentence inside a ```cnl fence and nothing else.",
            ));
            continue;
        };

        let reduction = match verify_cnl(&cnl, lexicon) {
            Ok(r) => r,
            Err(error) => {
                let failure = FailedAttempt { cnl, error };
                turns.push(failure.correction_turn());
                verify_failures.push(failure);
                continue;
            }
        };

        if require_verified && !reduction.is_verified() {
            let failure = FailedAttempt {
                cnl: reduction.cnl.clone(),
                error: VerifyError::NotConfluent {
                    unf_hash: reduction.unf_hash.clone(),
                },
            };
            turns.push(failure.correction_turn());
            verify_failures.push(failure);
            continue;
        }

        if let Some((dep_id, _)) = dependency_formalizations
            .iter()
            .find(|(_, f)| f.unf_hash == reduction.unf_hash)
        {
            return Err(FormalizeError::DuplicateIdentity {
                cnl: reduction.cnl,
                unf_hash: reduction.unf_hash,
                dependency: dep_id.clone(),
            });
        }
        return Ok(reduction.into_formalization(attempt + 1));
    }

    if verify_failures.is_empty() && missing_block > 0 {
        return Err(FormalizeError::NoCnlBlock { attempts: replies });
    }
    // `max_retries == 0` reaches here with neither counter set.
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

/// A node is formalized only if it is a lemma or a theorem statement.
///
/// Conditions and definitions are formalized but never proved — ProofFlow's
/// `proof_prover.run_solver_prompt` returns `{}` for them, and §14's
/// substitution table keeps that split. This predicate is how P4's completer
/// knows which nodes to attempt at all.
pub fn is_provable(kind: NodeKind) -> bool {
    matches!(kind, NodeKind::Lemma | NodeKind::TheoremStatement)
}

/// `true` when the transcript has at least one `user` turn, so a caller can
/// assert a prompt was actually assembled.
pub fn has_user_turn(turns: &[Turn]) -> bool {
    turns.iter().any(|t| t.role == Role::User)
}

/// Build a lexicon report for the coverage diagnostic in P8's harness.
pub fn unknown_words(sentence: &str, lexicon: &Lexicon) -> Vec<String> {
    sentence
        .split_whitespace()
        .filter(|w| lexicon.lookup(w).is_empty())
        .map(String::from)
        .collect()
}

/// Convenience: the entries of a lexicon as `(word, category)` pairs, for
/// prompts and reports.
pub fn vocabulary(lexicon: &Lexicon) -> Vec<(&str, &str)> {
    lexicon
        .entries()
        .iter()
        .map(|LexEntry { word, category, .. }| (word.as_str(), category.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::graph::{self, GraphNode};
    use crate::formalize::llm::{LlmConfig, ScriptedTransport};
    use std::time::Duration;

    fn lexicon() -> Lexicon {
        base_lexicon()
    }

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

    /// A one-sentence reply, wrapped in the fence the prompt demands.
    fn say(cnl: &str) -> String {
        format!("```cnl\n{cnl}\n```")
    }

    fn node(id: &str, deps: &[&str]) -> GraphNode {
        GraphNode::new(
            id,
            "nl",
            "the statement",
            deps.iter().map(|d| d.to_string()),
        )
    }

    // ── the lexicon ─────────────────────────────────────────────────────────

    #[test]
    fn base_lexicon_loads_from_the_embedded_tsv() {
        let lex = base_lexicon();
        // 46 content lines in corpus/lexicon.tsv.
        assert!(lex.word_count() >= 45, "got {}", lex.word_count());
        assert!(!lex.lookup("loves").is_empty());
        assert!(lex.lookup("nonexistentword").is_empty());
    }

    /// `prob_kernel::logos::compile_with` hashes the net *without* linearity
    /// inserted; `logos::engram::segment` hashes *with* it. If those two
    /// disagreed, every engram key derived here would be unreachable from the
    /// kernel's own `LogosReport`, and §14's content-addressed identity would
    /// break across the seam. They agree because reduction is confluent:
    /// inserting explicit closures cannot change the normal form.
    #[test]
    fn unf_hash_agrees_with_and_without_linearity() {
        use crate::ccg;
        let lex = lexicon();
        for sentence in ["John loves Mary", "John adds two three", "the cat sleeps"] {
            let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();
            let tree = &ccg::parse_sentence(&tokens, &lex)[0];
            let ir = core_ir::compile_to_core_ir(tree, &lex).unwrap();

            let mut plain_net = deltanet::compile_to_net(&ir).unwrap();
            deltanet::reduce(&mut plain_net).unwrap();
            let plain = deltanet::unf_hash_string(&plain_net).unwrap();

            let with_lin =
                translate::translate_coreir(&core_ir::linearity::insert_linearity(ir)).unwrap();

            assert_eq!(plain, with_lin.unf_hash, "{sentence}");
        }
    }

    #[test]
    fn a_domain_extension_adds_vocabulary_and_is_flagged() {
        let base = DomainLexicon::base();
        assert!(!base.is_extended());
        // Reuses the built-in `Eq` constructor, so the extension is accepted.
        let tsv =
            "congruent\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Eq\", [Var(\"x\"), Var(\"y\")])))";
        let ext = DomainLexicon::with_extension(tsv).unwrap();
        assert!(ext.is_extended());
        assert!(ext.lexicon().lookup("congruent").len() == 1);
        // The base language is still there.
        assert!(!ext.lexicon().lookup("loves").is_empty());
        assert_eq!(ext.word_count(), base.word_count() + 1);
    }

    /// The extension adds vocabulary; it does not silently redefine the core.
    #[test]
    fn an_extension_cannot_shadow_a_base_word() {
        let tsv = "loves\tN\tCon(\"Cat\", [])";
        let ext = DomainLexicon::with_extension(tsv).unwrap();
        // Both readings are present...
        assert_eq!(ext.lexicon().lookup("loves").len(), 2);
        // ...and the base template still wins, so the extension cannot change
        // the meaning of an existing sentence.
        let t = ext.lexicon().semantic_template("loves").unwrap();
        assert!(format!("{t:?}").contains("Love"));
    }

    // ── verify_cnl: the Lean-oracle replacement ─────────────────────────────

    #[test]
    fn verifies_a_transitive_sentence() {
        let r = verify_cnl("John loves Mary", &lexicon()).unwrap();
        assert_eq!(r.readback, "Love(john, mary)");
        assert!(r.verified);
        assert!(!r.unf_hash.is_empty());
        assert!(r.compiles());
    }

    #[test]
    fn verifies_an_arithmetic_sentence() {
        let r = verify_cnl("John adds two three", &lexicon()).unwrap();
        assert_eq!(r.readback, "Assign(john, Add(3, 2))");
        assert!(r.verified);
    }

    /// The `verified` flag must be stable across calls: it is the confluence
    /// self-check, and an unstable hash would poison the engram store in P3.
    #[test]
    fn unf_hash_is_deterministic_across_runs() {
        let lex = lexicon();
        let a = verify_cnl("John adds two three", &lex).unwrap();
        let b = verify_cnl("John adds two three", &lex).unwrap();
        assert_eq!(a.unf_hash, b.unf_hash);
        assert_eq!(a.readback, b.readback);
    }

    /// Two different sentences denote different terms, so identity is not
    /// accidental string matching. (The *same* sentence twice is covered by
    /// `unf_hash_is_deterministic_across_runs`.)
    #[test]
    fn different_sentences_get_different_unf_hashes() {
        let lex = lexicon();
        let a = verify_cnl("John loves Mary", &lex).unwrap();
        let b = verify_cnl("Bob sees Alice", &lex).unwrap();
        assert_ne!(a.readback, b.readback);
        assert_ne!(a.unf_hash, b.unf_hash);
    }

    #[test]
    fn an_out_of_lexicon_word_is_named_in_the_error() {
        let err = verify_cnl("Euler proves congruences", &lexicon()).unwrap_err();
        match err {
            VerifyError::OutOfLexicon { words, .. } => {
                assert!(words.contains(&"Euler".to_string()), "{words:?}");
                assert!(words.contains(&"congruences".to_string()), "{words:?}");
            }
            other => panic!("expected OutOfLexicon, got {other:?}"),
        }
    }

    /// §17.1: the coverage gap must be visible, not a bare "no parse".
    #[test]
    fn the_coverage_error_is_actionable() {
        let err = verify_cnl("Euler proves congruences", &lexicon()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not in the lexicon"), "{msg}");
        assert!(msg.contains("Euler"), "{msg}");
    }

    #[test]
    fn in_lexicon_but_unparseable_is_a_no_parse() {
        // All four words are in L0, but `John sleeps Mary` has no derivation:
        // `sleeps` is intransitive.
        let err = verify_cnl("John sleeps Mary", &lexicon()).unwrap_err();
        assert!(matches!(err, VerifyError::NoParse { .. }), "{err:?}");
    }

    #[test]
    fn a_too_short_sentence_is_gate_rejected() {
        let err = verify_cnl("John", &lexicon()).unwrap_err();
        assert!(matches!(err, VerifyError::GateRejected(_)), "{err:?}");
    }

    /// A domain extension may add *words* but not *constructors*: `tag_id` maps an
    /// unknown name to the legal tag 0, so the name is discarded, the readback
    /// says `Unknown`, and two distinct constructors of equal arity would
    /// collide on one UNF hash. That is a silent identity collapse, so
    /// construction fails instead.
    #[test]
    fn an_extension_naming_an_unknown_constructor_is_rejected() {
        let tsv = "congruent\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Congruent\", [Var(\"x\"), Var(\"y\")])))";
        let err = DomainLexicon::with_extension(tsv).unwrap_err();
        match &err {
            LexiconError::UnknownConstructor { names, known } => {
                assert_eq!(names, &["Congruent".to_string()]);
                assert_eq!(*known, core_ir::BUILTIN_CONSTRUCTORS.len());
            }
            other => panic!("expected UnknownConstructor, got {other:?}"),
        }
        assert!(err.to_string().contains("Congruent"), "{err}");
    }

    /// An extension that only reuses built-in constructors is fine.
    #[test]
    fn an_extension_using_a_builtin_constructor_is_accepted() {
        let tsv =
            "congruent\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Eq\", [Var(\"x\"), Var(\"y\")])))";
        let ext = DomainLexicon::with_extension(tsv).unwrap();
        assert!(ext.is_extended());
        let r = verify_cnl("one congruent two", ext.lexicon()).unwrap();
        assert_eq!(r.readback, "Eq(1, 2)");
        assert!(r.verified);
    }

    #[test]
    fn the_base_lexicon_uses_only_known_constructors() {
        assert!(unknown_constructors(&lexicon()).is_empty());
    }

    /// Stock L0 cannot express a real proof step, and says so by name.
    #[test]
    fn a_domain_sentence_is_out_of_lexicon_under_stock_l0() {
        let err = verify_cnl("one congruent two", &lexicon()).unwrap_err();
        match err {
            VerifyError::OutOfLexicon { words, .. } => {
                assert!(words.contains(&"congruent".to_string()), "{words:?}");
            }
            other => panic!("expected OutOfLexicon, got {other:?}"),
        }
    }

    // ── block extraction ────────────────────────────────────────────────────

    #[test]
    fn the_last_cnl_block_wins() {
        // A model that thinks out loud emits a scratch sentence first.
        let reply = "Try `John loves Mary`\n```cnl\nMary sees Bob\n```\nactually\n```cnl\nBob likes John\n```";
        assert_eq!(extract_cnl_block(reply).as_deref(), Some("Bob likes John"));
    }

    #[test]
    fn a_cnl_block_with_other_fences_present_is_found() {
        let reply = "```json\n{\"a\": 1}\n```\n```cnl\nJohn loves Mary\n```";
        assert_eq!(extract_cnl_block(reply).as_deref(), Some("John loves Mary"));
    }

    #[test]
    fn a_lean_fence_is_not_a_cnl_block() {
        let reply = "```lean4\ntheorem t : True := by sorry\n```";
        assert_eq!(extract_cnl_block(reply), None);
    }

    #[test]
    fn a_single_line_cnl_fence_is_found() {
        assert_eq!(
            extract_cnl_block("answer: ```cnl John loves Mary```").as_deref(),
            Some("John loves Mary")
        );
    }

    #[test]
    fn no_fence_yields_none() {
        assert_eq!(extract_cnl_block("John loves Mary"), None);
    }

    // ── the retry loop ──────────────────────────────────────────────────────

    #[test]
    fn formalizes_on_the_first_reply() {
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("John loves Mary")]), cfg());
        let f = formalize_node(
            &node("l1", &[]),
            &[],
            &mut llm,
            &lexicon(),
            &FormalizeOptions::default(),
        )
        .unwrap();
        assert_eq!(f.cnl, "John loves Mary");
        assert_eq!(f.readback, "Love(john, mary)");
        assert_eq!(f.tries, 1);
        assert!(f.is_verified());
    }

    /// A model that never uses the fence is reported as such, distinctly from a
    /// model whose sentences are all wrong.
    #[test]
    fn a_missing_fence_is_reported_distinctly_from_a_bad_sentence() {
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec!["I think the answer is John loves Mary".into()]),
            cfg(),
        );
        let err = formalize_node(
            &node("l1", &[]),
            &[],
            &mut llm,
            &lexicon(),
            &FormalizeOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, FormalizeError::NoCnlBlock { .. }), "{err:?}");
        assert!(err.to_string().contains("```cnl"), "{err}");
    }

    /// The out-of-lexicon error is fed back *by name*, which is what lets the
    /// next attempt fix the actual problem.
    #[test]
    fn the_typed_error_is_fed_back_to_the_model() {
        let t = ScriptedTransport::new(vec![
            say("Euler proves congruences"),
            say("John loves Mary"),
        ]);
        let mut llm = LlmClient::new(t.clone(), cfg());
        let f = formalize_node(
            &node("l1", &[]),
            &[],
            &mut llm,
            &lexicon(),
            &FormalizeOptions::default(),
        )
        .unwrap();
        assert_eq!(f.tries, 2);

        // Attempt 2's request body must contain the named offending words.
        let second = &llm.transport().seen()[1];
        let msgs = second["messages"].as_array().unwrap();
        let correction = msgs.last().unwrap()["content"].as_str().unwrap();
        assert!(correction.contains("CNL error"), "{correction}");
        assert!(correction.contains("Euler"), "{correction}");
    }

    #[test]
    fn exhausting_retries_reports_the_last_typed_error() {
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec![say("Euler proves congruences")]),
            cfg(),
        );
        let err = formalize_node(
            &node("l1", &[]),
            &[],
            &mut llm,
            &lexicon(),
            &FormalizeOptions::default(),
        )
        .unwrap_err();
        match err {
            FormalizeError::NotVerified { last, attempts } => {
                assert_eq!(attempts.len(), 3, "one failure per attempt");
                assert!(matches!(last, VerifyError::OutOfLexicon { .. }));
            }
            other => panic!("expected NotVerified, got {other:?}"),
        }
    }

    /// §14: identity is the UNF hash, so a node that collapses onto its
    /// dependency is not a formalization of anything.
    #[test]
    fn a_node_that_repeats_its_dependency_is_rejected() {
        let parent = CnlFormalization {
            cnl: "John loves Mary".into(),
            readback: "Love(john, mary)".into(),
            unf_hash: verify_cnl("John loves Mary", &lexicon()).unwrap().unf_hash,
            verified: true,
            value: None,
            ted: None,
            tries: 1,
        };
        let deps = vec![("l1".to_string(), parent)];
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![say("John loves Mary")]), cfg());
        let err = formalize_node(
            &node("l2", &["l1"]),
            &deps,
            &mut llm,
            &lexicon(),
            &FormalizeOptions::default(),
        )
        .unwrap_err();
        match err {
            FormalizeError::DuplicateIdentity { dependency, .. } => assert_eq!(dependency, "l1"),
            other => panic!("expected DuplicateIdentity, got {other:?}"),
        }
    }

    /// The dependency's CNL is only shown when asked for, and when shown it is
    /// labelled as something *not* to copy.
    #[test]
    fn dependency_context_is_opt_in() {
        let node = node("l2", &["l1"]);
        let off = build_user_turn(&node, &[], &FormalizeOptions::default());
        assert!(
            !off.content.contains("reduced normal form"),
            "{}",
            off.content
        );

        let deps = vec![(
            "l1".to_string(),
            CnlFormalization {
                cnl: "John loves Mary".into(),
                readback: "Love(john, mary)".into(),
                unf_hash: "deadbeef".into(),
                verified: true,
                value: None,
                ted: None,
                tries: 1,
            },
        )];
        let on = build_user_turn(
            &node,
            &deps,
            &FormalizeOptions {
                include_dependency_cnl: true,
                ..FormalizeOptions::default()
            },
        );
        assert!(on.content.contains("Love(john, mary)"), "{}", on.content);
        assert!(on.content.contains("must denote"), "{}", on.content);
    }

    /// The prompt carries the node's statement and its dependencies — the two
    /// things ProofFlow's prompt builder puts in.
    #[test]
    fn the_user_turn_carries_the_statement_and_dependencies() {
        let n = GraphNode::new("l2", "quote", "the exact statement", vec!["l1".into()]);
        let turn = build_user_turn(&n, &[], &FormalizeOptions::default());
        assert!(
            turn.content.contains("the exact statement"),
            "{}",
            turn.content
        );
        assert!(turn.content.contains("l1"), "{}", turn.content);
        assert_eq!(turn.role, Role::User);
        assert!(has_user_turn(&[turn]));
    }

    // ── the verified/compiles split (the Lean pass/verify distinction) ──────

    #[test]
    fn conditions_and_definitions_are_not_provable() {
        assert!(!is_provable(NodeKind::TheoremCondition));
        assert!(!is_provable(NodeKind::Definition));
        assert!(is_provable(NodeKind::Lemma));
        assert!(is_provable(NodeKind::TheoremStatement));
    }

    #[test]
    fn unknown_words_helper_matches_the_verify_error() {
        let lex = lexicon();
        let unknown = unknown_words("Euler proves congruences", &lex);
        assert!(unknown.contains(&"Euler".to_string()));
        match verify_cnl("Euler proves congruences", &lex).unwrap_err() {
            VerifyError::OutOfLexicon { words, .. } => assert_eq!(words, unknown),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn vocabulary_reports_words_and_categories() {
        let lex = lexicon();
        let v = vocabulary(&lex);
        assert!(v.contains(&("loves", "(S\\NP)/NP")), "{v:?}");
        assert_eq!(v.len(), lex.word_count());
    }

    // ── the whole graph, end to end, on the real fixture ───────────────────

    /// The pipeline against ProofFlow's own Euler graph, with every node
    /// formalized from the L0 lexicon. It proves the loop, the prompt, the
    /// extraction and the identity check compose — and it documents exactly how
    /// far stock L0 reaches on a real proof (§17.1).
    #[test]
    fn the_real_euler_graph_formalizes_under_stock_l0() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../testdata/formalize/euler_proof_graph.json"
        ))
        .unwrap();
        let mut graph = graph::validate_proof_graph(&data).unwrap();

        // Nine distinct sentences, one per node. Distinctness is load-bearing: the
        // loop rejects a node whose UNF hash repeats a dependency's, and
        // ScriptedTransport repeats the last reply, so a shorter list would make
        // the final `ts_1` collide with `l5`.
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
        let mut llm = LlmClient::new(
            ScriptedTransport::new(sentences.iter().map(|s| say(s)).collect()),
            cfg(),
        );
        let lex = lexicon();

        let mut failures = Vec::new();
        for i in 0..graph.len() {
            let id = graph.nodes[i].id.clone();
            let deps: Vec<(String, CnlFormalization)> = graph.nodes[..i]
                .iter()
                .filter_map(|n| {
                    n.formalization
                        .as_ref()
                        .map(|f| (n.id.clone(), serde_json::from_value(f.clone()).unwrap()))
                })
                .collect();
            match formalize_node(
                &graph.nodes[i],
                &deps,
                &mut llm,
                &lex,
                &FormalizeOptions::default(),
            ) {
                Ok(f) => {
                    graph.nodes[i].formalization = Some(serde_json::to_value(&f).unwrap());
                }
                Err(e) => failures.push(format!("{id}: {e}")),
            }
        }

        assert!(failures.is_empty(), "nodes failed: {failures:#?}");
        // Every node got a UNF hash, and they are all distinct — which is the
        // content-addressed identity the rest of the pipeline relies on.
        let hashes: Vec<String> = graph
            .nodes
            .iter()
            .map(|n| {
                n.formalization
                    .as_ref()
                    .and_then(|f| f.get("unf_hash"))
                    .and_then(|h| h.as_str())
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(hashes.len(), 9);
        let unique: std::collections::BTreeSet<&String> = hashes.iter().collect();
        assert_eq!(
            unique.len(),
            hashes.len(),
            "duplicate UNF hashes: {hashes:?}"
        );

        // The graph still round-trips through validation with results attached.
        let round: serde_json::Value = serde_json::to_value(&graph.nodes).unwrap();
        assert_eq!(graph::validate_proof_graph(&round).unwrap().len(), 9);
    }
}
