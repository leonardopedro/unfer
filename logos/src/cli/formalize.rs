//! P6 — `logos formalize`: the pipeline as a command.
//!
//! ```text
//! logos formalize <input> [--out report.json] [--vis dag.html]
//! ```
//!
//! `<input>` is either a natural-language proof (markdown, sent to the model) or
//! a proof-graph JSON file (the model is not consulted for the graph stage, so
//! the rest of the pipeline runs with no key at all — which is how the tests
//! drive it).
//!
//! # Why the CLI is a thin shell over a testable function
//!
//! [`run_pipeline`] takes its [`Transport`] as an argument and returns a
//! [`PipelineReport`], so every stage is exercised in `cargo test` without a
//! network, a key, or a temporary directory. This file only parses flags,
//! writes files, and maps failures to exit codes.
//!
//! # Exit codes
//!
//! `0` the pipeline ran to completion · `1` bad usage or a stage failed · `2`
//! the pipeline ran but the result is not trustworthy (a node failed to verify,
//! or a structural fault). `2` is deliberately distinct from `1`: a run that
//! produced a report but found problems is not the same as a run that could not
//! happen, and a script that checks only `!= 0` cannot tell them apart.

use crate::dagviz::{self, escape};
use crate::formalize::completer::{self, CompleteOptions, CompleteReport};
use crate::formalize::embed::{Embedder, HashingEmbedder};
use crate::formalize::formalizer::{self, CnlFormalization, DomainLexicon, FormalizeOptions};
use crate::formalize::graph::{self, Validated};
use crate::formalize::hybrid::{self, LemmaIndex, XerjNode};
use crate::formalize::index::IndexParams;
use crate::formalize::llm::{LlmClient, LlmConfig, Transport};
use crate::formalize::memory::{self, LemmaStore};
use crate::formalize::score::{self, Aggregation, LogosScore};
#[cfg(feature = "llm-http")]
use crate::formalize::embed::HttpEmbedder;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// Which stages to run.
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub max_retries: usize,
    /// ProofFlow's `no_dag` mode: linearize every node's dependencies.
    pub follow_dag: bool,
    /// Run the completer (P4) after formalization.
    pub complete: bool,
    /// Run the LogosScore judge (P4) after completion.
    pub score: bool,
    /// Accept sentences that compile but do not reduce uniquely.
    ///
    /// ProofFlow has no analogue: `lean_pass` and `lean_verify` are separate
    /// stages there too, and this flag is the equivalent of stopping after
    /// `lean_pass`.
    pub no_verify: bool,
    pub aggregation: Aggregation,
    /// A domain lexicon TSV appended to stock L0.
    pub lexicon: Option<PathBuf>,
    /// Refute rather than prove.
    pub prove_negation: bool,
    /// How few-shot examples reach the formalizer's prompt (P3/P9).
    pub retrieval: RetrievalOptions,
}

/// Which few-shot retrieval arms run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    /// Statement/CNL vocabulary overlap (`memory::retrieve_lexical`).
    /// Offline, deterministic, the default.
    #[default]
    Lexical,
    /// RRF fusion of the lexical arm, a semantic arm over the LEANN-shaped
    /// lemma index, and a xerj node when one is configured
    /// (`formalize::hybrid`).
    Hybrid,
}

/// Retrieval configuration: flags, with env fallbacks (`UNFER_*`) applied
/// by the CLI. Nothing here has a network dependency until `embed_url` or
/// `xerj_url` is set — `hybrid` with neither runs the semantic arm on
/// lexical feature hashing, which the report states as such.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalOptions {
    pub mode: RetrievalMode,
    /// OpenAI-compatible `/embeddings` base URL — EmbeddingGemma 2 through
    /// llama.cpp/Ollama/TEI. `None` = lexical hashing (labelled `lexical`,
    /// never `semantic`).
    #[serde(default)]
    pub embed_url: Option<String>,
    /// Model field sent to `embed_url`, and the id reported with every hit.
    #[serde(default = "default_embed_model")]
    pub embed_model: String,
    /// Matryoshka truncation (768/512/256/128); 0 = native.
    #[serde(default)]
    pub embed_dims: usize,
    /// A running xerj node — its ranking joins the fusion as an arm.
    #[serde(default)]
    pub xerj_url: Option<String>,
    #[serde(default = "default_xerj_index")]
    pub xerj_index: String,
}

fn default_embed_model() -> String {
    "embeddinggemma-2".into()
}

fn default_xerj_index() -> String {
    "lemmas".into()
}

impl Default for RetrievalOptions {
    fn default() -> Self {
        RetrievalOptions {
            mode: RetrievalMode::default(),
            embed_url: None,
            embed_model: default_embed_model(),
            embed_dims: 0,
            xerj_url: None,
            xerj_index: default_xerj_index(),
        }
    }
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            max_retries: 3,
            follow_dag: true,
            complete: true,
            score: true,
            no_verify: false,
            aggregation: Aggregation::Katz,
            lexicon: None,
            prove_negation: false,
            retrieval: RetrievalOptions::default(),
        }
    }
}

/// What the run did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    /// A natural-language proof; the graph stage needs the model.
    NaturalLanguage,
    /// A proof-graph JSON file; the graph stage is read from disk.
    ProofGraphJson,
}

/// The whole run's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineReport {
    pub input_kind: InputKind,
    pub warnings: Vec<String>,
    /// One row per node: the formalized CNL, its normal form and its UNF.
    pub nodes: Vec<NodeReport>,
    pub completion: Option<CompleteReport>,
    pub score: Option<LogosScore>,
    pub memory: memory::StoreStats,
    /// What retrieval did: which arms, which model, how many examples the
    /// prompts actually carried. A run that claims to be retrieval-augmented
    /// has to say what retrieved.
    #[serde(default)]
    pub retrieval: RetrievalReport,
    /// Words in the lexicon in force.
    pub lexicon_words: usize,
    /// Whether a domain extension was applied.
    pub extended_lexicon: bool,
}

/// What retrieval did during the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RetrievalReport {
    pub mode: RetrievalMode,
    /// `model@dims` of the semantic arm's embedder; `None` in lexical mode.
    #[serde(default)]
    pub embedder: Option<String>,
    /// `base/index` of a xerj node when its arm was configured.
    #[serde(default)]
    pub xerj: Option<String>,
    /// Few-shot examples actually rendered into prompts (a cache hit asks
    /// for none).
    #[serde(default)]
    pub examples: usize,
}

impl Default for RetrievalReport {
    fn default() -> Self {
        RetrievalReport {
            mode: RetrievalMode::default(),
            embedder: None,
            xerj: None,
            examples: 0,
        }
    }
}

impl PipelineReport {
    /// Nodes that produced no formalization.
    pub fn unformalized(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|n| n.formalization.is_none())
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Formalized nodes whose reduction is not unique.
    pub fn unverified(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|n| n.formalization.as_ref().is_some_and(|f| !f.verified))
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Whether the run is trustworthy: every node formalized, every reduction
    /// unique, and the scorer clean.
    ///
    /// `0` is returned only when this holds, which is what makes exit code `2`
    /// mean "ran, but do not believe it".
    pub fn is_trustworthy(&self) -> bool {
        self.unformalized().is_empty()
            && self.unverified().is_empty()
            && self.score.as_ref().is_some_and(score::LogosScore::is_clean)
    }
}

/// One node's row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeReport {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formalization: Option<CnlFormalization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negation: Option<CnlFormalization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A stage failed.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// The input could not be read or did not parse.
    Input(String),
    /// The graph stage could not produce a valid DAG.
    Graph(graph::GraphError),
    /// A formalization or completion stage failed.
    Formalize(String),
    /// The scorer could not run.
    Score(String),
    /// A file could not be written.
    Io { path: String, reason: String },
    /// The `llm-http` feature is off, so no model is reachable.
    NoTransport,
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Input(m) => write!(f, "bad input: {m}"),
            PipelineError::Graph(e) => write!(f, "proof graph rejected: {e}"),
            PipelineError::Formalize(m) => write!(f, "{m}"),
            PipelineError::Score(m) => write!(f, "scoring failed: {m}"),
            PipelineError::Io { path, reason } => write!(f, "{path}: {reason}"),
            PipelineError::NoTransport => write!(
                f,
                "no LLM transport is compiled in. Rebuild with \
                 `--features llm-http`. The formalization stage needs a model \
                 whatever the input format, so a proof-graph JSON file does not \
                 avoid this."
            ),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Run the pipeline over already-parsed input text.
pub fn run_pipeline<T: Transport>(
    input: &str,
    llm: &mut LlmClient<T>,
    opts: &PipelineOptions,
    store: &mut LemmaStore,
) -> Result<(PipelineReport, Validated), PipelineError> {
    // The lexicon first: every stage needs it, and a bad extension should fail
    // before any model call rather than after.
    let lexicon = match &opts.lexicon {
        Some(path) => {
            let tsv = std::fs::read_to_string(path).map_err(|e| PipelineError::Io {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
            DomainLexicon::with_extension(&tsv).map_err(|e| PipelineError::Input(e.to_string()))?
        }
        None => DomainLexicon::base(),
    };
    // `is_extended` is the flag that records "this run used a domain
    // extension". It was being dropped here and reconstructed downstream as
    // `word_count() > 46` — which happens to agree only because
    // `corpus/lexicon.tsv` currently holds exactly 46 entries. Add one row to a
    // data file the crate expects to grow, and every report starts claiming
    // "(domain extension)" for stock L0.
    let extended_lexicon = lexicon.is_extended();
    let lexicon = lexicon.lexicon();

    // Retrieval setup (P3/P9). Lexical needs nothing; hybrid builds the
    // LEANN-shaped index over the store's lemmas once, before any node runs,
    // and grows it as the run remembers new lemmas.
    let mut retrieval = RetrievalCtx::new(&opts.retrieval, store)?;
    let mut examples_served = 0usize;

    // Graph stage. A JSON input is read from disk; a proof is generated, which
    // is the one stage that cannot be skipped.
    let (mut graph, input_kind) = if looks_like_graph_json(input) {
        let value: serde_json::Value =
            serde_json::from_str(input).map_err(|e| PipelineError::Input(e.to_string()))?;
        (
            graph::validate_proof_graph(&value).map_err(PipelineError::Graph)?,
            InputKind::ProofGraphJson,
        )
    } else {
        let build = graph::BuildOptions {
            follow_dag: opts.follow_dag,
            max_retries: opts.max_retries,
        };
        let (g, _) = graph::build_proof_graph(input, llm, &build).map_err(PipelineError::Graph)?;
        (g, InputKind::NaturalLanguage)
    };

    let formalize_opts = FormalizeOptions {
        max_retries: opts.max_retries,
        // Retrieval-augmented: P3's few-shot examples are the whole reason the
        // store exists, so the dependency CNL is offered by default here.
        include_dependency_cnl: true,
        // Base value only — every node gets its own block, computed just
        // before it formalizes (a cache-hit node asks for none).
        few_shot: String::new(),
    };

    // Formalization, in topological order so a dependency's UNF is known before
    // its dependant's identity check.
    let model = llm.config().model.clone();
    for i in 0..graph.len() {
        let node = graph.nodes[i].clone();
        let deps: Vec<(String, CnlFormalization)> = graph.nodes[..i]
            .iter()
            .filter_map(|n| score::formalization_of(n).map(|f| (n.id.clone(), f)))
            .collect();

        // The provenance cache comes first (§17.3): an identical re-request —
        // same statement, same prompt version, same model — is served from
        // memory instead of being re-asked, which is what makes a pipeline run
        // reproducible even though the model is not. Keyed on the statement and
        // never on the node id, because a node id is a position in *this* graph.
        let cache_key = memory::CacheKey {
            node_text: node.statement.clone(),
            prompt_version: PROMPT_VERSION.to_string(),
            model: model.clone(),
        };
        let cached = store.lookup_provenance(&cache_key).cloned();

        // Few-shot examples, only for a node that will actually ask the
        // model: a cache hit needs no example, and retrieval is a query per
        // node (one batched embed call per hop over the lemma index).
        let examples = if cached.is_none() {
            retrieval.examples_for(store, &node)
        } else {
            Vec::new()
        };
        let mut node_opts = formalize_opts.clone();
        node_opts.few_shot = memory::few_shot_block(&examples);
        examples_served += examples.len();

        let result = match &cached {
            Some(lemma) => Ok(lemma.to_formalization()),
            None if opts.no_verify => {
                formalizer::formalize_node(&node, &deps, llm, lexicon, &node_opts)
            }
            None => {
                formalizer::formalize_node_verified(&node, &deps, llm, lexicon, &node_opts)
            }
        };

        match result {
            Ok(f) => {
                // The identity check applies to a cached sentence too: a cache
                // hit must not smuggle in a step that collides with a dependency.
                if let Some((dep_id, _)) = deps.iter().find(|(_, d)| d.unf_hash == f.unf_hash) {
                    graph.nodes[i].error_report = Some(serde_json::json!({
                        "error_type": "identity",
                        "error_report": format!(
                            "CNL {:?} reduces to the UNF of dependency '{dep_id}' ({})",
                            f.cnl, f.unf_hash
                        ),
                    }));
                    continue;
                }
                // An unverified node is recorded but not trusted into memory —
                // §17.4: no unique normal form, no content address.
                if f.verified && cached.is_none() {
                    let _ = store.remember(&node, &f, &model, PROMPT_VERSION);
                    // Keep the hybrid index current with what was just
                    // remembered — an incremental insert recomputes and
                    // drops vectors, so the store can grow all run without
                    // the index growing at rest.
                    retrieval.index_lemma(store, &f.unf_hash);
                }
                graph.nodes[i].formalization = Some(serde_json::to_value(&f).map_err(|e| {
                    PipelineError::Formalize(format!("a formalization did not serialize: {e}"))
                })?);
            }
            Err(e) => {
                graph.nodes[i].error_report = Some(serde_json::json!({
                    "error_type": "formalize",
                    "error_report": e.to_string(),
                }));
            }
        }
    }

    // Completion.
    let completion = if opts.complete {
        let formalizations: Vec<(String, CnlFormalization)> = graph
            .nodes
            .iter()
            .filter_map(|n| score::formalization_of(n).map(|f| (n.id.clone(), f)))
            .collect();
        let copts = CompleteOptions {
            max_retries: opts.max_retries,
            prove_negation: opts.prove_negation,
        };
        Some(completer::complete_graph(
            &mut graph,
            &formalizations,
            llm,
            lexicon,
            &copts,
        ))
    } else {
        None
    };

    let scored = if opts.score {
        Some(score::judge_graph(&graph, llm, opts.aggregation).map_err(PipelineError::Score)?)
    } else {
        None
    };

    let nodes = graph
        .nodes
        .iter()
        .map(|n| NodeReport {
            id: n.id.clone(),
            kind: format!("{:?}", n.kind().unwrap_or(graph::NodeKind::Lemma)),
            formalization: score::formalization_of(n),
            negation: n
                .solved_negation
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok()),
            error: n
                .error_report
                .as_ref()
                .and_then(|v| v.get("error_report"))
                .and_then(|v| v.as_str())
                .map(String::from),
        })
        .collect();

    Ok((
        PipelineReport {
            input_kind,
            warnings: {
                let mut w: Vec<String> = graph.warnings.iter().map(|w| w.to_string()).collect();
                w.extend(retrieval.warnings.iter().cloned());
                w
            },
            nodes,
            completion,
            score: scored,
            memory: store.stats().clone(),
            retrieval: RetrievalReport {
                examples: examples_served,
                ..retrieval.report.clone()
            },
            lexicon_words: lexicon.word_count(),
            extended_lexicon,
        },
        graph,
    ))
}

/// Bumped whenever a prompt's wording changes, and part of the provenance cache
/// key (§17.3). Kept as a constant so a prompt edit and a cache invalidation are
/// the same commit.
pub const PROMPT_VERSION: &str = "v1";

/// How many examples each arm offers per node. Three: enough for the model
/// to see a pattern, few enough that the prompt stays about *this* node —
/// and RRF over three arms with k=60 still separates rank 1 from rank 3.
const RETRIEVAL_K: usize = 3;

/// The retrieval configuration turned into running state: what was set up,
/// and what the report should say about it.
struct RetrievalCtx {
    report: RetrievalReport,
    embedder: Option<Box<dyn Embedder>>,
    index: Option<LemmaIndex>,
    xerj: Option<XerjNode>,
    /// Non-fatal retrieval problems (an insert that failed mid-run). They
    /// are surfaced as report warnings rather than swallowed: a run whose
    /// index quietly stopped tracking the store must say so.
    warnings: Vec<String>,
}

impl RetrievalCtx {
    fn new(opts: &RetrievalOptions, store: &LemmaStore) -> Result<Self, PipelineError> {
        match opts.mode {
            RetrievalMode::Lexical => Ok(RetrievalCtx {
                report: RetrievalReport::default(),
                embedder: None,
                index: None,
                xerj: None,
                warnings: Vec::new(),
            }),
            RetrievalMode::Hybrid => {
                let embedder = build_embedder(opts)?;
                let index = LemmaIndex::build(store, embedder.as_ref(), IndexParams::default())
                    .map_err(|e| PipelineError::Formalize(format!("retrieval index: {e}")))?;
                let xerj = opts
                    .xerj_url
                    .as_ref()
                    .map(|u| XerjNode::new(u.clone(), opts.xerj_index.clone()));
                Ok(RetrievalCtx {
                    report: RetrievalReport {
                        mode: RetrievalMode::Hybrid,
                        embedder: Some(format!("{}@{}", embedder.id(), embedder.dims())),
                        xerj: opts
                            .xerj_url
                            .as_ref()
                            .map(|u| format!("{u}/{}", opts.xerj_index)),
                        examples: 0,
                    },
                    embedder: Some(embedder),
                    index: Some(index),
                    xerj,
                    warnings: Vec::new(),
                })
            }
        }
    }

    /// One node's few-shot examples, from the configured arms.
    fn examples_for(&self, store: &LemmaStore, node: &graph::GraphNode) -> Vec<memory::Retrieved> {
        match self.report.mode {
            RetrievalMode::Lexical => memory::retrieve_lexical(store, node, RETRIEVAL_K),
            RetrievalMode::Hybrid => match (&self.index, &self.embedder) {
                (Some(index), Some(embedder)) => hybrid::retrieve_hybrid(
                    store,
                    index,
                    embedder.as_ref(),
                    node,
                    RETRIEVAL_K,
                    self.xerj.as_ref(),
                ),
                // Setup guarantees both in hybrid mode; falling back to
                // lexical keeps the prompt honest instead of empty.
                _ => memory::retrieve_lexical(store, node, RETRIEVAL_K),
            },
        }
    }

    /// Index a freshly remembered lemma (hybrid mode only).
    fn index_lemma(&mut self, store: &LemmaStore, unf_hash: &str) {
        if let (Some(index), Some(embedder)) = (&mut self.index, &self.embedder) {
            if let Some(lemma) = store.get(unf_hash) {
                if let Err(e) = index.insert(embedder.as_ref(), lemma) {
                    self.warnings.push(format!(
                        "retrieval: new lemma {unf_hash} not indexed ({e}) — \
                         subsequent hybrid retrieval sees the store without it"
                    ));
                }
            }
        }
    }
}

/// The semantic arm's embedder: EmbeddingGemma 2 when an endpoint is
/// configured, xerj's honest lexical feature-hashing otherwise.
#[cfg(feature = "llm-http")]
fn build_embedder(opts: &RetrievalOptions) -> Result<Box<dyn Embedder>, PipelineError> {
    Ok(match (&opts.embed_url, opts.embed_dims) {
        (Some(url), dims) => {
            Box::new(HttpEmbedder::new(url.clone(), opts.embed_model.clone(), dims))
        }
        (None, 0) => Box::new(HashingEmbedder::default()),
        (None, dims) => Box::new(HashingEmbedder::new(dims)),
    })
}

#[cfg(not(feature = "llm-http"))]
fn build_embedder(opts: &RetrievalOptions) -> Result<Box<dyn Embedder>, PipelineError> {
    if opts.embed_url.is_some() {
        return Err(PipelineError::Formalize(
            "--embed-url needs an HTTP client: rebuild with `--features llm-http` \
             (the embeddings call is a network request, like the LLM one)"
                .into(),
        ));
    }
    Ok(match opts.embed_dims {
        0 => Box::new(HashingEmbedder::default()),
        dims => Box::new(HashingEmbedder::new(dims)),
    })
}

/// Whether the input is a proof-graph JSON document rather than prose.
fn looks_like_graph_json(input: &str) -> bool {
    input.trim_start().starts_with('[')
}

/// The full report as pretty JSON.
pub fn report_json(report: &PipelineReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

// ── the HTML DAG ────────────────────────────────────────────────────────────

/// Render the proof DAG as a self-contained HTML page.
///
/// Self-contained deliberately: ProofFlow's `vis.py` emits an interactive
/// matplotlib figure that needs Python to view. A single file with inline CSS
/// and no scripts can be opened from disk, attached to a CI artifact, or hosted
/// as a mathed `\app` figure without a server — which is what P7's DAG-as-a-
/// figure needs.
///
/// Layout is layered: `layer(n) = 0` for a node with no dependencies, else
/// `1 + max(layer(deps))`. The graph is already topologically ordered, so one
/// pass suffices.
///
/// The figure itself comes from [`crate::dagviz`], the canvas shared with
/// `mm0web`'s submission graph: this function decides what each box *means*,
/// the canvas decides where it is drawn.
pub fn dag_html(report: &PipelineReport, graph: &Validated) -> String {
    let layers = layer_assignment(graph);

    let by_id: std::collections::BTreeMap<&str, &NodeReport> =
        report.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // The page's semantics, not the canvas': what a node is, how it fared,
    // and what it rests on.
    let boxes: Vec<dagviz::BoxView> = graph
        .nodes
        .iter()
        .map(|node| {
            let row = by_id.get(node.id.as_str()).copied();
            let mut lines = Vec::new();
            if let Some(f) = row.and_then(|r| r.formalization.as_ref()) {
                lines.push(f.cnl.clone());
                lines.push(f.readback.clone());
                lines.push(short_hash(&f.unf_hash));
            }
            let accent = row.and_then(|r| r.error.clone()).map(|e| (e, "#a4262c"));
            let badge = row.and_then(|r| {
                report
                    .score
                    .as_ref()
                    .and_then(|sc| sc.nodes.iter().find(|n| n.id == r.id))
                    .map(|s| format!("score {:.2}", s.semantic_score))
            });
            let mut b = dagviz::BoxView::new(
                *layers.get(node.id.as_str()).unwrap_or(&0),
                node.id.clone(),
                row_color(row),
            );
            b.subtitle = row.map(|r| r.kind.clone()).unwrap_or_default();
            b.lines = lines;
            b.accent = accent;
            b.badge = badge;
            b.deps = node.dependencies.clone();
            b
        })
        .collect();
    let svg = dagviz::svg(&boxes, &dagviz::Canvas::default());


    let legend = "<p class=\"legend\">\
<span class=\"sw ok\"></span> formalized &amp; uniquely reduced \
<span class=\"sw bad\"></span> formalized, not uniquely reduced \
<span class=\"sw err\"></span> failed</p>";

    let summary = match &report.score {
        Some(s) => format!(
            "<p><strong>LogosScore</strong>: {} ({}) · dependency report: {}</p>",
            s.total
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "not scored".into()),
            s.aggregation,
            if s.dependency.is_clean() {
                "clean".to_string()
            } else {
                "faults found".to_string()
            }
        ),
        None => "<p><strong>LogosScore</strong>: not run.</p>".to_string(),
    };

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
<title>logos formalize — proof DAG</title>\
<style>\
body{{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:24px;color:#1b1e23;background:#fbfbfc}}\
h1{{font-size:18px;margin:0 0 4px}}\
p{{margin:4px 0}}\
.legend{{color:#6b7076;font-size:12px;margin-top:10px}}\
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;\
vertical-align:-1px;margin:0 4px 0 14px;border:2px solid}}\
.sw.ok{{border-color:#1a7f37}} .sw.bad{{border-color:#bf8700}} .sw.err{{border-color:#a4262c}}\
code{{background:#f0f1f3;padding:1px 5px;border-radius:4px;font-size:12px}}\
</style></head><body>\n\
<h1>logos formalize</h1>\n\
<p>input: <code>{}</code> · lexicon: {} words{} · \
lemmas: {} distinct / {} nodes</p>\n\
{summary}\n{legend}\n{svg}</body></html>\n",
        escape(match report.input_kind {
            InputKind::NaturalLanguage => "natural-language proof",
            InputKind::ProofGraphJson => "proof-graph JSON",
        }),
        report.lexicon_words,
        if report.extended_lexicon {
            " (domain extension)"
        } else {
            " (stock L0)"
        },
        report.memory.lemmas,
        report.memory.cache_entries,
    )
}

fn short_hash(h: &str) -> String {
    h.chars().take(12).collect()
}

fn layer_assignment(graph: &Validated) -> std::collections::BTreeMap<&str, usize> {
    let mut layers = std::collections::BTreeMap::new();
    // `graph.nodes` is topologically ordered, so a dependency is always seen
    // before its dependant and one pass is enough.
    for node in &graph.nodes {
        let layer = node
            .dependencies
            .iter()
            .filter_map(|d| layers.get(d.as_str()).copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        layers.insert(node.id.as_str(), layer);
    }
    layers
}

fn row_color(row: Option<&NodeReport>) -> &'static str {
    match row {
        None => "#8a8f98",
        Some(r) if r.error.is_some() => "#a4262c",
        Some(r) => match &r.formalization {
            Some(f) if f.verified => "#1a7f37",
            Some(_) => "#bf8700",
            None => "#8a8f98",
        },
    }
}

/// Write `report_json` to `path`, creating parent directories.
pub fn write_report(path: &Path, report: &PipelineReport) -> Result<(), PipelineError> {
    let text = report_json(report).map_err(|e| PipelineError::Formalize(e.to_string()))?;
    write_file(path, &text)
}

/// Write the HTML DAG to `path`.
pub fn write_html(
    path: &Path,
    report: &PipelineReport,
    graph: &Validated,
) -> Result<(), PipelineError> {
    write_file(path, &dag_html(report, graph))
}

fn write_file(path: &Path, text: &str) -> Result<(), PipelineError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| PipelineError::Io {
            path: parent.display().to_string(),
            reason: e.to_string(),
        })?;
    }
    std::fs::write(path, text).map_err(|e| PipelineError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })
}

/// Load a lemma store written by `--memory-out`.
pub fn load_store(path: &Path) -> Result<LemmaStore, PipelineError> {
    let text = std::fs::read_to_string(path).map_err(|e| PipelineError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    memory::from_json(&text).map_err(|e| PipelineError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })
}

/// Save a lemma store.
pub fn save_store(path: &Path, store: &LemmaStore) -> Result<(), PipelineError> {
    write_file(
        path,
        &memory::to_json(store).map_err(|e| PipelineError::Formalize(e.to_string()))?,
    )
}

/// The default client configuration, honouring the `LLM_*`/`OPENAI_*` env.
pub fn default_config() -> LlmConfig {
    LlmConfig::default()
}

const USAGE: &str = "\
Usage: logos formalize <input> [options]

  <input>              a natural-language proof (markdown), or a proof-graph JSON
                      file — the latter needs the model only for the nodes.

Options:
  --out <path>         write the JSON report (default: stdout summary only)
  --vis <path>         write a self-contained HTML dependency graph
  --memory-in <path>   load a lemma store before running
  --memory-out <path>  write the lemma store afterwards
  --lexicon <path>     a domain TSV appended to stock L0
  --no-dag             linearize every node's dependencies (ProofFlow's no-DAG mode)
  --max-retries <n>    attempts per stage (default 3)
  --skip-complete      stop after formalization
  --skip-score         stop before judging
  --no-verify          accept sentences that compile without reducing uniquely
  --negate             refute rather than prove
  --aggregation <k>    equal | laplacian | katz (default katz)
  --retrieval <m>      lexical (default) | hybrid — how few-shot examples
                       are retrieved for the formalizer's prompt
  --embed-url <url>    OpenAI-compatible /embeddings endpoint for hybrid's
                       semantic arm (EmbeddingGemma 2 behind llama.cpp,
                       Ollama, TEI, LiteRT). Default: offline lexical
                       feature hashing, reported as lexical.
  --embed-model <id>   model field for --embed-url (default embeddinggemma-2)
  --embed-dims <n>     Matryoshka truncation, 0 = native (default 0)
  --xerj-url <url>     a running xerj node: its ranking joins the fusion
  --xerj-index <name>  that node's index (default lemmas)
  -h, --help           this text

Env fallbacks (flags win): UNFER_EMBED_URL, UNFER_EMBED_MODEL,
UNFER_EMBED_DIMS, UNFER_XERJ_URL, UNFER_XERJ_INDEX.

Exit codes: 0 ran and trustworthy · 1 bad usage or a stage failed ·
2 ran, but the result is not trustworthy.";

/// Parse flags and run. Returns the process exit code.
///
/// The argument parsing is hand-rolled to match the rest of this CLI, which is
/// `std::process::exit`-based and dependency-free by design. An unknown flag is
/// an error rather than a warning: silently ignoring `--out` would write a
/// report the user did not ask for and lose it.
pub fn run(args: &[String]) -> i32 {
    match run_inner(args) {
        Ok(code) => code,
        Err(PipelineError::Input(m)) => {
            eprintln!("error: {m}\n\n{USAGE}");
            1
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn run_inner(args: &[String]) -> Result<i32, PipelineError> {
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut vis: Option<PathBuf> = None;
    let mut memory_in: Option<PathBuf> = None;
    let mut memory_out: Option<PathBuf> = None;
    let mut opts = PipelineOptions::default();

    // Env defaults for retrieval, seeded *before* the flag loop so a flag
    // always wins. Empty strings stay unset: an empty embed URL is a
    // relative request to nowhere, an empty xerj index queries the wrong
    // thing silently.
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    opts.retrieval.embed_url = env("UNFER_EMBED_URL");
    if let Some(m) = env("UNFER_EMBED_MODEL") {
        opts.retrieval.embed_model = m;
    }
    if let Some(d) = env("UNFER_EMBED_DIMS") {
        opts.retrieval.embed_dims = d.parse().map_err(|_| {
            PipelineError::Input(format!("UNFER_EMBED_DIMS {d} is not a number"))
        })?;
    }
    opts.retrieval.xerj_url = env("UNFER_XERJ_URL");
    if let Some(x) = env("UNFER_XERJ_INDEX") {
        opts.retrieval.xerj_index = x;
    }

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        // `--flag value`, with `--flag=value` accepted as well.
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg, None),
        };
        let take_value = |i: &mut usize| -> Result<String, PipelineError> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| PipelineError::Input(format!("{flag} needs a value")))
        };

        match flag {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(0);
            }
            "--out" => out = Some(PathBuf::from(take_value(&mut i)?)),
            "--vis" => vis = Some(PathBuf::from(take_value(&mut i)?)),
            "--memory-in" => memory_in = Some(PathBuf::from(take_value(&mut i)?)),
            "--memory-out" => memory_out = Some(PathBuf::from(take_value(&mut i)?)),
            "--lexicon" => opts.lexicon = Some(PathBuf::from(take_value(&mut i)?)),
            "--max-retries" => {
                let v = take_value(&mut i)?;
                opts.max_retries = v.parse().map_err(|_| {
                    PipelineError::Input(format!("--max-retries {v} is not a number"))
                })?;
            }
            "--no-dag" => opts.follow_dag = false,
            "--skip-complete" => opts.complete = false,
            "--skip-score" => opts.score = false,
            "--no-verify" => opts.no_verify = true,
            "--negate" => opts.prove_negation = true,
            "--retrieval" => {
                let v = take_value(&mut i)?;
                opts.retrieval.mode = match v.as_str() {
                    "lexical" => RetrievalMode::Lexical,
                    "hybrid" => RetrievalMode::Hybrid,
                    other => {
                        return Err(PipelineError::Input(format!(
                            "--retrieval {other} is not one of lexical, hybrid"
                        )));
                    }
                };
            }
            "--embed-url" => opts.retrieval.embed_url = Some(take_value(&mut i)?),
            "--embed-model" => opts.retrieval.embed_model = take_value(&mut i)?,
            "--embed-dims" => {
                let v = take_value(&mut i)?;
                opts.retrieval.embed_dims = v.parse().map_err(|_| {
                    PipelineError::Input(format!("--embed-dims {v} is not a number"))
                })?;
            }
            "--xerj-url" => opts.retrieval.xerj_url = Some(take_value(&mut i)?),
            "--xerj-index" => opts.retrieval.xerj_index = take_value(&mut i)?,
            "--aggregation" => {
                let v = take_value(&mut i)?;
                opts.aggregation = match v.as_str() {
                    "equal" => Aggregation::Equal,
                    "laplacian" => Aggregation::Laplacian,
                    "katz" => Aggregation::Katz,
                    other => {
                        return Err(PipelineError::Input(format!(
                            "--aggregation {other} is not one of equal, laplacian, katz"
                        )));
                    }
                };
            }
            other if other.starts_with('-') => {
                return Err(PipelineError::Input(format!("unknown flag {other}")));
            }
            _ => {
                if input.is_some() {
                    return Err(PipelineError::Input(
                        "exactly one input file is expected".into(),
                    ));
                }
                input = Some(PathBuf::from(arg));
            }
        }
        i += 1;
    }

    let Some(input) = input else {
        return Err(PipelineError::Input("no input file".into()));
    };

    let text = std::fs::read_to_string(&input).map_err(|e| PipelineError::Io {
        path: input.display().to_string(),
        reason: e.to_string(),
    })?;

    let mut store = match &memory_in {
        Some(p) => load_store(p)?,
        None => LemmaStore::new(),
    };

    // The HTTP transport is feature-gated, so this is where "no model reachable"
    // is discovered. Reported before any work is done.
    //
    // The output flags are parsed unconditionally so that `--help` and argument
    // errors behave identically in both builds, but without a transport there is
    // no pipeline to produce a report, a graph or a lemma store — so they are
    // dropped here rather than silently honoured.
    #[cfg(not(feature = "llm-http"))]
    {
        let _ = (&text, &mut store, &opts, &out, &vis, &memory_out);
        Err(PipelineError::NoTransport)
    }
    #[cfg(feature = "llm-http")]
    let (report, graph) = {
        use crate::formalize::llm::HttpTransport;
        let config = default_config();
        let timeout = config.timeout;
        let mut client = LlmClient::new(HttpTransport::with_timeout(timeout), config);
        run_pipeline(&text, &mut client, &opts, &mut store)?
    };

    #[cfg(feature = "llm-http")]
    {
        if let Some(path) = &out {
            write_report(path, &report)?;
        }
        if let Some(path) = &vis {
            write_html(path, &report, &graph)?;
        }
        if let Some(path) = &memory_out {
            save_store(path, &store)?;
        }

        print_summary(&report, out.is_some());
        Ok(if report.is_trustworthy() { 0 } else { 2 })
    }
}

/// The human-readable summary, always printed.
///
/// Written to stdout even when `--out` is given: a JSON file is for machines, and
/// the point of running the command is to see what it found.
///
/// Feature-gated with the tail of `run_inner`, which is its only caller.
#[cfg(feature = "llm-http")]
fn print_summary(report: &PipelineReport, wrote_json: bool) {
    use crate::dagviz::truncate;
    let kind = match report.input_kind {
        InputKind::NaturalLanguage => "natural-language proof",
        InputKind::ProofGraphJson => "proof-graph JSON",
    };
    println!("logos formalize — {kind}");
    println!(
        "  lexicon: {} words{} · lemmas: {} distinct / {} nodes",
        report.lexicon_words,
        if report.extended_lexicon {
            " (domain extension)"
        } else {
            " (stock L0)"
        },
        report.memory.lemmas,
        report.memory.cache_entries,
    );
    for w in &report.warnings {
        println!("  warning: {w}");
    }
    println!("  nodes:");
    for n in &report.nodes {
        match (&n.formalization, &n.error) {
            (Some(f), _) => println!(
                "    {:<6} {:<22} {:<40} {}",
                n.id,
                n.kind,
                truncate(&f.cnl, 40),
                short_hash(&f.unf_hash)
            ),
            (None, Some(e)) => println!("    {:<6} {:<22} FAILED: {}", n.id, n.kind, e),
            (None, None) => println!("    {:<6} {:<22} (not formalized)", n.id, n.kind),
        }
    }
    if let Some(c) = &report.completion {
        println!(
            "  completion: {} completed, {} skipped, {} failed",
            c.completed, c.skipped, c.failed
        );
    }
    match &report.score {
        Some(s) => {
            println!(
                "  LogosScore: {} ({})",
                s.total
                    .map(|t| format!("{t:.2}"))
                    .unwrap_or_else(|| "not scored".into()),
                s.aggregation
            );
            println!(
                "  dependency report: {}",
                if s.dependency.is_clean() {
                    "clean"
                } else {
                    "faults found"
                }
            );
        }
        None => println!("  LogosScore: not run"),
    }
    if wrote_json {
        println!("  (report JSON written)");
    }
    if !report.is_trustworthy() {
        println!(
            "  NOT TRUSTWORTHY — unformalized: {:?}, unverified: {:?}",
            report.unformalized(),
            report.unverified()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::formalizer::base_lexicon;
    use crate::formalize::llm::ScriptedTransport;

    fn say(cnl: &str) -> String {
        format!("```cnl\n{cnl}\n```")
    }

    fn judge(payload: &str) -> String {
        format!("```json\n{payload}\n```")
    }

    fn cfg() -> LlmConfig {
        LlmConfig {
            base_url: "http://local/v1".into(),
            model: "test".into(),
            api_key: None,
            temperature: 0.0,
            max_tokens: 512,
            timeout: std::time::Duration::from_secs(1),
        }
    }

    /// The Euler fixture, so the CLI is exercised on a real proof graph.
    fn euler_json() -> String {
        include_str!("../../testdata/formalize/euler_proof_graph.json").to_string()
    }

    /// A scripted client that answers the formalizer, then the completer, then
    /// the judge, in that order — which is the order the pipeline asks.
    fn staged(answers: Vec<String>) -> LlmClient<ScriptedTransport> {
        // A stage that never asks the model still needs a transport to exist, so
        // an empty script gets one placeholder reply rather than panicking in
        // `ScriptedTransport::new`.
        let mut answers = answers;
        if answers.is_empty() {
            answers.push("unused".into());
        }
        LlmClient::new(ScriptedTransport::new(answers), cfg())
    }

    // ── input handling ──────────────────────────────────────────────────────

    #[test]
    fn a_graph_json_input_is_detected() {
        assert!(looks_like_graph_json("  [{\"id\": \"l1\"}]"));
        assert!(!looks_like_graph_json("Let f be a polynomial."));
        assert!(!looks_like_graph_json(""));
    }

    #[test]
    fn a_graph_json_input_needs_no_model_for_the_graph_stage() {
        // Every reply is a *judgement*, so if the graph stage had asked for
        // anything the parse would be nonsense and the pipeline would fail.
        let mut llm = staged(vec![judge(
            r#"{"evaluation": ["Perfectly match"], "feedback": ["ok"]}"#,
        )]);
        let (report, _) = run_pipeline(
            &euler_json(),
            &mut llm,
            &PipelineOptions::default(),
            &mut LemmaStore::new(),
        )
        .unwrap();
        assert_eq!(report.input_kind, InputKind::ProofGraphJson);
        assert_eq!(report.nodes.len(), 9);
    }

    #[test]
    fn malformed_graph_json_is_reported_as_bad_input() {
        let mut llm = staged(vec![]);
        let err = run_pipeline(
            "[{\"nope\": 1}]",
            &mut llm,
            &PipelineOptions::default(),
            &mut LemmaStore::new(),
        )
        .unwrap_err();
        // A node missing its required fields is a *graph* fault naming the node,
        // not a bare input error.
        match err {
            PipelineError::Graph(graph::GraphError::InvalidNode { id, .. }) => {
                assert_eq!(id, "<missing id>")
            }
            other => panic!("expected InvalidNode, got {other:?}"),
        }
    }

    #[test]
    fn a_cyclic_graph_is_rejected_by_the_pipeline() {
        let cyclic = r#"[{"id":"l1","natural_language":"a","statement":"b","dependencies":["ts_1"]},
                        {"id":"ts_1","natural_language":"a","statement":"b","dependencies":["l1"]}]"#;
        let mut llm = staged(vec![]);
        let err = run_pipeline(
            cyclic,
            &mut llm,
            &PipelineOptions::default(),
            &mut LemmaStore::new(),
        )
        .unwrap_err();
        assert!(
            matches!(err, PipelineError::Graph(graph::GraphError::Cycle)),
            "{err:?}"
        );
    }

    // ── the stages ──────────────────────────────────────────────────────────

    /// Nine distinct sentences, then nine judge replies, in pipeline order.
    fn all_stages() -> (PipelineOptions, Vec<String>) {
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
        let mut answers: Vec<String> = sentences.iter().map(|s| say(s)).collect();
        // The completer asks again for the six provable nodes, then the judge.
        answers.extend(sentences[3..].iter().map(|s| say(s)));
        answers.extend(
            sentences
                .iter()
                .map(|_| judge(r#"{"evaluation": ["Perfectly match"], "feedback": []}"#)),
        );
        (PipelineOptions::default(), answers)
    }

    #[test]
    fn the_whole_pipeline_runs_and_is_trustworthy() {
        let (opts, answers) = all_stages();
        let mut store = LemmaStore::new();
        let mut llm = staged(answers);
        let (report, graph) = run_pipeline(&euler_json(), &mut llm, &opts, &mut store).unwrap();

        assert_eq!(report.nodes.len(), 9);
        assert!(
            report.unformalized().is_empty(),
            "{:?}",
            report.unformalized()
        );
        assert!(report.unverified().is_empty(), "{:?}", report.unverified());

        // Every node that can be proved was completed.
        let completion = report.completion.as_ref().unwrap();
        assert_eq!(completion.completed, 6);
        assert_eq!(completion.skipped, 3);
        assert_eq!(completion.failed, 0);

        // Scored, weighted, and reported with its own weights.
        let scored = report.score.as_ref().unwrap();
        assert!(scored.total.is_some());
        assert_eq!(scored.aggregation, "Katz");
        assert_eq!(scored.weights.len(), 9);

        assert!(report.is_trustworthy(), "{report:#?}");
        assert_eq!(graph.len(), 9);
        assert_eq!(report.memory.lemmas, 9);
    }

    #[test]
    fn skipping_stages_is_reported_as_such() {
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
        let answers: Vec<String> = sentences.iter().map(|s| say(s)).collect();
        let opts = PipelineOptions {
            complete: false,
            score: false,
            ..PipelineOptions::default()
        };
        let mut llm = staged(answers);
        let (report, _) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        assert!(report.completion.is_none());
        assert!(report.score.is_none());
        assert!(report.unformalized().is_empty());
        // Without a scorer, nothing can be *trusted* — and that is the honest
        // answer rather than a default pass.
        assert!(!report.is_trustworthy());
    }

    /// A node that fails to formalize is recorded with its reason and keeps the
    /// rest of the graph.
    #[test]
    fn a_failed_node_is_recorded_and_the_run_is_untrustworthy() {
        let mut answers: Vec<String> = Vec::new();
        // tc_1 fails; the rest get distinct sentences.
        answers.push(say("Euler proves congruences"));
        answers.push("still failing".into());
        answers.push("still failing".into());
        for s in [
            "Mary sees Bob",
            "Alice likes John",
            "Bob sees Alice",
            "Mary sees Alice",
            "Alice likes Mary",
            "Bob likes John",
            "Alice sees Mary",
            "John runs",
        ] {
            answers.push(say(s));
        }
        let opts = PipelineOptions {
            complete: false,
            score: false,
            ..PipelineOptions::default()
        };
        let mut llm = staged(answers);
        let (report, _) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        let tc = report.nodes.iter().find(|n| n.id == "tc_1").unwrap();
        assert!(tc.formalization.is_none());
        assert!(tc.error.as_deref().unwrap().contains("lexicon"), "{tc:?}");
        assert!(
            !report.is_trustworthy(),
            "a failed node must not be trustworthy"
        );
    }

    /// `no_verify` is the `lean_pass` analogue: compile without requiring a unique
    /// normal form.
    #[test]
    fn no_verify_stops_at_the_compiles_bar() {
        let opts = PipelineOptions {
            no_verify: true,
            complete: false,
            score: false,
            ..PipelineOptions::default()
        };
        let answers: Vec<String> = vec![
            say("John loves Mary"),
            say("Mary sees Bob"),
            say("Alice likes John"),
            say("Bob sees Alice"),
            say("Mary sees Alice"),
            say("Alice likes Mary"),
            say("Bob likes John"),
            say("Alice sees Mary"),
            say("John runs"),
        ];
        let mut llm = staged(answers);
        let (report, _) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        assert!(report.unformalized().is_empty());
        assert!(report.nodes.iter().all(|n| n.formalization.is_some()));
    }

    #[test]
    fn the_provenance_cache_is_reused_across_runs() {
        // Two runs over the same graph with the same model and prompt version:
        // the second must not need the formalizer's answers at all, because
        // every node is already in memory. The scripted transport replays its
        // last reply, so a cache miss shows up as a different CNL sentence.
        let (opts, answers) = all_stages();
        let mut store = LemmaStore::new();
        {
            let mut llm = staged(answers.clone());
            let _ = run_pipeline(&euler_json(), &mut llm, &opts, &mut store).unwrap();
        }
        let before = store.len();

        // A client that answers *nothing* usable: if the run needs a
        // formalization, it will fail or produce a different sentence.
        let mut llm = staged(vec![say("Bob likes Alice")]);
        let (report, _) = run_pipeline(&euler_json(), &mut llm, &opts, &mut store).unwrap();
        assert_eq!(store.len(), before, "no new lemmas");
        assert!(
            report.unformalized().is_empty(),
            "{:?}",
            report.unformalized()
        );
    }

    // ── output ──────────────────────────────────────────────────────────────

    #[test]
    fn the_json_report_round_trips() {
        let (opts, answers) = all_stages();
        let mut llm = staged(answers);
        let (report, _) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        let text = report_json(&report).unwrap();
        let back: PipelineReport = serde_json::from_str(&text).unwrap();
        assert_eq!(back.nodes.len(), report.nodes.len());
        assert_eq!(back.input_kind, report.input_kind);
    }

    #[test]
    fn the_html_dag_is_self_contained_and_escapes_content() {
        let (opts, answers) = all_stages();
        let mut llm = staged(answers);
        let (report, graph) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        let html = dag_html(&report, &graph);

        assert!(html.starts_with("<!DOCTYPE html>"));
        // No external fetches: a viewBox SVG, inline CSS, no <script>, no src.
        assert!(!html.contains("<script"), "must not need JavaScript");
        assert!(!html.contains(" src="), "must not fetch anything");
        assert!(html.contains("<svg"));
        assert!(html.contains("Love(john, mary)"), "readbacks are shown");
        assert!(html.contains("score 1.00"), "scores are shown");
        assert!(!html.contains("&\n"), "escaping is applied");
    }

    /// Model output and file content both land in the page, so escaping is a
    /// correctness requirement rather than a nicety.
    #[test]
    fn html_escaping_neutralizes_injected_markup() {
        assert_eq!(
            escape("<script>alert(1)</script>"),
            "&lt;script&gt;alert(1)&lt;/script&gt;"
        );
        assert_eq!(escape("a & b"), "a &amp; b");
        assert_eq!(escape("\"quoted\""), "&quot;quoted&quot;");
    }

    #[test]
    fn truncation_keeps_the_tail_visible_as_an_ellipsis() {
        use crate::dagviz::truncate;
        assert_eq!(truncate("short", 10), "short");
        let long = "x".repeat(50);
        let t = truncate(&long, 10);
        assert_eq!(t.chars().count(), 10);
        assert!(t.ends_with('…'));
    }

    #[test]
    fn the_dag_places_dependencies_below_their_dependants() {
        let (opts, answers) = all_stages();
        let mut llm = staged(answers);
        let (_, graph) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        let layers = layer_assignment(&graph);
        // Every dependency sits in a strictly lower layer, which is what makes
        // the arrows point downward.
        for node in &graph.nodes {
            for dep in &node.dependencies {
                assert!(
                    layers[dep.as_str()] < layers[node.id.as_str()],
                    "{dep} should be below {}",
                    node.id
                );
            }
        }
        assert_eq!(layers["tc_1"], 0);
        assert_eq!(layers["l1"], 1);
        assert_eq!(layers["ts_1"], 4);
    }

    #[test]
    fn a_failing_node_is_drawn_in_the_error_colour() {
        let opts = PipelineOptions {
            complete: false,
            score: false,
            max_retries: 1,
            ..PipelineOptions::default()
        };
        let mut answers: Vec<String> = vec![say("nope not a sentence")];
        answers.extend(
            [
                "Mary sees Bob",
                "Alice likes John",
                "Bob sees Alice",
                "Mary sees Alice",
                "Alice likes Mary",
                "Bob likes John",
                "Alice sees Mary",
                "John runs",
            ]
            .iter()
            .map(|s| say(s)),
        );
        let mut llm = staged(answers);
        let (report, graph) =
            run_pipeline(&euler_json(), &mut llm, &opts, &mut LemmaStore::new()).unwrap();
        let html = dag_html(&report, &graph);
        assert!(html.contains("#a4262c"), "an error node is drawn in red");
    }

    #[test]
    fn files_are_written_with_parent_directories_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/report.json");
        let report = PipelineReport {
            retrieval: RetrievalReport::default(),
            input_kind: InputKind::ProofGraphJson,
            warnings: vec![],
            nodes: vec![],
            completion: None,
            score: None,
            memory: Default::default(),
            lexicon_words: base_lexicon().word_count(),
            extended_lexicon: false,
        };
        write_report(&path, &report).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"nodes\""));
    }

    #[test]
    fn an_unwritable_path_reports_the_path_and_the_reason() {
        let err = write_report(
            Path::new("/proc/definitely/not/writable.json"),
            &PipelineReport {
                input_kind: InputKind::ProofGraphJson,
                warnings: vec![],
                nodes: vec![],
                completion: None,
                score: None,
                memory: Default::default(),
                retrieval: RetrievalReport::default(),
                lexicon_words: 0,
                extended_lexicon: false,
            },
        )
        .unwrap_err();
        match err {
            // The failure surfaces on the *first* unwritable component, which is
            // the parent directory it tried to create — so the path reported is
            // that, not the file.
            PipelineError::Io { path, reason } => {
                assert!(path.starts_with("/proc/definitely"), "{path}");
                assert!(!reason.is_empty());
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    // ── retrieval wiring (P3/P9) ───────────────────────────────────────────

    /// A store with one verified lemma, so retrieval has something to find.
    fn seeded_store() -> LemmaStore {
        let mut store = LemmaStore::new();
        let f = crate::formalize::formalizer::verify_cnl("John loves Mary", &base_lexicon())
            .expect("the stock lexicon sentence verifies")
            .into_formalization(1);
        store
            .insert(
                &f,
                memory::Provenance {
                    node_id: "l1".into(),
                    node_text: "John loves Mary".into(),
                    model: "test-model".into(),
                    prompt_version: "v1".into(),
                },
            )
            .expect("verified formalization inserts");
        store
    }

    fn hybrid_opts() -> PipelineOptions {
        PipelineOptions {
            retrieval: RetrievalOptions {
                mode: RetrievalMode::Hybrid,
                ..RetrievalOptions::default()
            },
            ..PipelineOptions::default()
        }
    }

    #[test]
    fn the_report_states_the_default_retrieval_mode() {
        let mut llm = staged(vec![judge(
            r#"{"evaluation": ["Perfectly match"], "feedback": ["ok"]}"#,
        )]);
        let (report, _) = run_pipeline(
            &euler_json(),
            &mut llm,
            &PipelineOptions::default(),
            &mut LemmaStore::new(),
        )
        .unwrap();
        assert_eq!(report.retrieval.mode, RetrievalMode::Lexical);
        assert_eq!(report.retrieval.embedder, None, "no model claimed when none ran");
        assert_eq!(report.retrieval.xerj, None);
        assert_eq!(report.retrieval.examples, 0, "an empty store serves nothing");
    }

    #[test]
    fn hybrid_mode_runs_offline_and_names_the_lexical_hashing_honestly() {
        // No endpoint configured: the semantic arm is feature hashing, and
        // the report must say `lexical-feature-hashing` — never "semantic".
        let mut llm = staged(vec![judge(
            r#"{"evaluation": ["Perfectly match"], "feedback": ["ok"]}"#,
        )]);
        let (report, _) = run_pipeline(
            &euler_json(),
            &mut llm,
            &hybrid_opts(),
            &mut LemmaStore::new(),
        )
        .unwrap();
        assert_eq!(report.retrieval.mode, RetrievalMode::Hybrid);
        assert_eq!(
            report.retrieval.embedder.as_deref(),
            Some("lexical-feature-hashing@384")
        );
    }

    #[test]
    fn hybrid_mode_counts_the_examples_it_put_into_prompts() {
        let mut llm = staged(vec![judge(
            r#"{"evaluation": ["Perfectly match"], "feedback": ["ok"]}"#,
        )]);
        let mut store = seeded_store();
        let (report, _) = run_pipeline(&euler_json(), &mut llm, &hybrid_opts(), &mut store).unwrap();
        assert!(
            report.retrieval.examples > 0,
            "nodes asked the model with a non-empty store: {:?}",
            report.retrieval
        );
        // The run also indexed the store: the report's embedder is set.
        assert!(report.retrieval.embedder.is_some());
    }
}
