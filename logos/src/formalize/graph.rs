//! P1 — a faithful Rust port of ProofFlow's `proofflow/proof_graph.py`.
//!
//! # What this owns, and what it deliberately does not
//!
//! This module is the **pure** half of the graph builder: the node model, the
//! DAG validation rules, the LLM-transcript retry loop, and the
//! forensically-tuned JSON recovery that ProofFlow relies on. It contains no
//! I/O and no CNL knowledge — the LLM arrives as a [`Respond`] implementation,
//! which `formalize::llm` supplies in P2. That split is what keeps this file
//! exhaustively unit-testable without a network, a key, or a fixture LLM.
//!
//! # Parity is the point
//!
//! ProofFlow is kept runnable as a reference oracle (§17.5), so this is a port
//! and not a redesign. Every rule below is the Python rule, including the ones
//! that look accidental. Where a Python behaviour is a bug that would crash
//! rather than report, the port reports instead and says so at the call site;
//! those are marked `PORT FIX` and are the only intentional divergences.
//!
//! # The quirks that are load-bearing
//!
//! Three of ProofFlow's rules only make sense together, and a "cleaner" port
//! that tidies any of them will accept graphs the oracle rejects:
//!
//! 1. **Node kind comes from an id prefix**, and [`NodeKind::from_id`] tests
//!    `ts` → `l` → `def` → `tc` in *that* order. So `ts_1` is a theorem, and
//!    any id merely *starting* with `l` is a lemma.
//! 2. **The orphan rule exempts `ts_` but the kind rule matched `ts`.** Those
//!    are different prefixes. An id like `ts1` is a theorem statement by (1)
//!    but *not* exempt from the orphan check by (2), so it must still be
//!    referenced. `orphan_exempts` is `starts_with("ts_")`, faithfully.
//! 3. **Unknown dependencies only warn**; *forward* references raise. The
//!    distinction is that a dep id in `all_ids` but not yet available is a
//!    forward reference (an error), while an id that is in neither is simply
//!    unknown (a warning, and the graph proceeds).
//!
//! [`GraphWarning`] exists because Python `print`s those warnings to stdout,
//! where nothing can assert on them. Carrying them is what lets the P6 report
//! and the P8 parity harness compare them.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Which pydantic model ProofFlow would have selected for a node.
///
/// The discriminant is the id prefix, not a `"kind"` field — ProofFlow's own
/// `validate_proof_graph` branches on `item["id"].startswith(...)` and never
/// reads a type tag, so a fixture that omits one parses fine here too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// `tc_*` — a theorem condition (an assumed premise).
    TheoremCondition,
    /// `def_*` — a definition (also an assumed fact, per the prompts).
    Definition,
    /// `l*` — a lemma: one atomic inference.
    Lemma,
    /// `ts*` — the theorem statement that closes the proof.
    TheoremStatement,
}

impl NodeKind {
    /// The prefix ProofFlow matches, and the id prefix it is dispatched on.
    pub fn prefix(self) -> &'static str {
        match self {
            NodeKind::TheoremCondition => "tc",
            NodeKind::Definition => "def",
            NodeKind::Lemma => "l",
            NodeKind::TheoremStatement => "ts",
        }
    }

    /// ProofFlow's dispatch order: `ts`, then `l`, then `def`, then `tc`.
    ///
    /// The order is load-bearing — an id like `l...` is a lemma, never a
    /// condition — and a prefix such as `tc` cannot collide with an earlier
    /// arm, so the arm order is really only pinned by `l` (which would swallow
    /// any future `l*` kind). Unknown ids are [`None`], which
    /// [`validate_proof_graph`] turns into an error naming the four prefixes.
    pub fn from_id(id: &str) -> Option<NodeKind> {
        if id.starts_with("ts") {
            Some(NodeKind::TheoremStatement)
        } else if id.starts_with('l') {
            Some(NodeKind::Lemma)
        } else if id.starts_with("def") {
            Some(NodeKind::Definition)
        } else if id.starts_with("tc") {
            Some(NodeKind::TheoremCondition)
        } else {
            None
        }
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            NodeKind::TheoremCondition => "tc_",
            NodeKind::Definition => "def_",
            NodeKind::Lemma => "l",
            NodeKind::TheoremStatement => "ts_",
        })
    }
}

/// One proof-graph node: the union of ProofFlow's four pydantic models.
///
/// `solved_negation` is declared on `Lemma` and `TheoremStatement` only, but
/// [`NodeKind::from_id`] is applied *after* deserialization here, so the field
/// is accepted on every kind. ProofFlow behaves the same way — pydantic
/// ignores extra fields by default, so a `solved_negation` on a `def_` is
/// dropped from that model's view but does not fail validation.
///
/// The four result slots are `serde_json::Value`, not typed structs, for the
/// same reason they are `Dict[str, Any]` in Python: the pipeline fills them
/// with plain JSON, and ProofFlow never constrains their contents. `formalize`
/// defines the typed shapes and serializes them into these slots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    /// Exact quote from the theorem/proof justifying this inference.
    pub natural_language: String,
    /// Self-contained NL statement of the node.
    pub statement: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formalization: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solved_lemma: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solved_negation: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_report: Option<Value>,
}

impl GraphNode {
    /// The node's kind, from its id prefix.
    pub fn kind(&self) -> Option<NodeKind> {
        NodeKind::from_id(&self.id)
    }

    /// Minimal constructor for the fields a graph node always needs. The
    /// result slots stay `None` until the pipeline fills them.
    pub fn new(
        id: impl Into<String>,
        natural_language: impl Into<String>,
        statement: impl Into<String>,
        dependencies: impl IntoIterator<Item = String>,
    ) -> Self {
        GraphNode {
            id: id.into(),
            natural_language: natural_language.into(),
            statement: statement.into(),
            dependencies: dependencies.into_iter().collect(),
            formalization: None,
            solved_lemma: None,
            solved_negation: None,
            score: None,
            error_report: None,
        }
    }
}

/// A non-fatal finding. Python prints these and moves on.
///
/// Carried, not printed, so that P6's report and P8's parity harness can
/// compare them. Derived for serde so it can round-trip inside
/// [`Validated`]'s derived impls even though it is skipped when serializing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphWarning {
    /// A dependency id that appears nowhere in the graph. Python warns; the
    /// node is still kept.
    UnknownDependency { id: String, dependency: String },
}

impl fmt::Display for GraphWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphWarning::UnknownDependency { id, dependency } => {
                write!(
                    f,
                    "Item '{id}' references unknown dependency '{dependency}'"
                )
            }
        }
    }
}

/// Everything `check_DAG` can reject a graph for, mirroring the Python
/// exceptions one-for-one so P8's parity harness can compare failure modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// `validate_proof_graph`'s `else` arm.
    UnknownNodeType { id: String },
    /// A node nothing depends on, and not the closing `ts_`.
    OrphanNode { id: String },
    /// The dependency graph has a cycle.
    Cycle,
    /// A dependency that exists but is defined later in the list.
    ForwardReference { id: String, dependency: String },
    /// The LLM reply could not be turned into JSON.
    Json(String),
    /// A node was missing a required field.
    InvalidNode { id: String, reason: String },
    /// The LLM transport itself failed.
    Llm(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownNodeType { id } => write!(
                f,
                "Unknown node type: {id} -> Should be one of 'tc_', 'l_', 'def_', or 'ts_'"
            ),
            GraphError::OrphanNode { id } => write!(
                f,
                "'{id}' is not used as a dependency by any subsequent lemma or \
                 theorem statement. '{id}' needs to be used somewhere so that we \
                 have a valid proof.Please reconsider the graph structure."
            ),
            GraphError::Cycle => write!(
                f,
                "Cycle detected in the dependency graph! This violates the DAG \
                 property and will cause issues in proof formalization."
            ),
            GraphError::ForwardReference { id, dependency } => write!(
                f,
                "Item '{id}' has a forward reference to '{dependency}'. \
                 Dependencies should only reference previous items in the proof."
            ),
            GraphError::Json(m) => write!(f, "Failed to parse JSON: {m}"),
            GraphError::InvalidNode { id, reason } => write!(f, "Invalid node '{id}': {reason}"),
            GraphError::Llm(m) => write!(f, "LLM call failed: {m}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A validated graph plus the warnings Python would have printed.
///
/// `warnings` is skipped when serializing: the node list *is* the graph, and a
/// report that round-trips through [`validate_proof_graph`] must not carry
/// Python's `print` output as data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Validated {
    pub nodes: Vec<GraphNode>,
    #[serde(default, skip_serializing)]
    pub warnings: Vec<GraphWarning>,
}

impl Validated {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Node ids in proof order, which is also topological order once
    /// `check_dag` has accepted the graph.
    pub fn ids(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.id.as_str()).collect()
    }

    /// Mutable node lookup, for the pipeline stages that fill result slots.
    ///
    /// Panics on a missing id rather than returning `Option`: every caller is
    /// iterating a graph it just validated, so an unknown id is a bug in this
    /// crate, not bad input.
    pub fn node_mut(&mut self, id: &str) -> &mut GraphNode {
        self.nodes
            .iter_mut()
            .find(|n| n.id == id)
            .unwrap_or_else(|| panic!("no node with id {id}"))
    }
}

// ── DAG validation ─────────────────────────────────────────────────────────

/// ProofFlow's `check_DAG`, in its three passes.
///
/// Consumes and returns the node list so the caller can keep working on the
/// validated values.
pub fn check_dag(nodes: Vec<GraphNode>) -> Result<Validated, GraphError> {
    let all_ids: BTreeSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();

    // Pass 1: every id that appears as somebody's dependency.
    let used: BTreeSet<&str> = nodes
        .iter()
        .flat_map(|n| n.dependencies.iter().map(String::as_str))
        .collect();

    // Pass 2: orphans. Python exempts the `ts_` prefix — see the module docs;
    // this is `starts_with("ts_")`, not `NodeKind::from_id`.
    for node in &nodes {
        if !node.id.starts_with("ts_") && !used.contains(node.id.as_str()) {
            return Err(GraphError::OrphanNode {
                id: node.id.clone(),
            });
        }
    }

    // Pass 3: cycles, before the forward-reference check — matching Python, so
    // a graph that is both cyclic and forward-referencing reports the cycle.
    if has_cycle(&nodes) {
        return Err(GraphError::Cycle);
    }

    // Pass 4: unknown deps warn, forward references raise. A dep id that is
    // neither available nor in `all_ids` is unknown; one that is in `all_ids`
    // but not yet available is a forward reference.
    let mut warnings = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let available: BTreeSet<&str> = nodes[..i].iter().map(|n| n.id.as_str()).collect();
        for dep in &node.dependencies {
            if available.contains(dep.as_str()) {
                continue;
            }
            if all_ids.contains(dep.as_str()) {
                return Err(GraphError::ForwardReference {
                    id: node.id.clone(),
                    dependency: dep.clone(),
                });
            }
            warnings.push(GraphWarning::UnknownDependency {
                id: node.id.clone(),
                dependency: dep.clone(),
            });
        }
    }

    Ok(Validated { nodes, warnings })
}

/// Three-colour DFS over the dependency edges.
///
/// The recursion is the Python recursion, made iterative: a hostile corpus
/// would otherwise be able to overflow the stack, and `logos::engram` is
/// explicitly built for untrusted input. Cycle detection is order-independent,
/// so iterating `all_ids` in sorted order rather than Python's set order gives
/// the same answer deterministically.
///
/// A dependency id that is not a node id is skipped without being visited —
/// Python's `if dep in visit_state and dfs(dep)` guard, and the reason an
/// unknown dependency can never be reported as a cycle.
fn has_cycle(nodes: &[GraphNode]) -> bool {
    const WHITE: u8 = 0;
    const GREY: u8 = 1;
    const BLACK: u8 = 2;

    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        adj.insert(
            node.id.as_str(),
            node.dependencies.iter().map(String::as_str).collect(),
        );
    }

    let mut state: BTreeMap<&str, u8> = adj.keys().map(|id| (*id, WHITE)).collect();
    let start_ids: Vec<&str> = adj.keys().copied().collect();

    for start in start_ids {
        if state[start] != WHITE {
            continue;
        }
        state.insert(start, GREY);
        let mut stack: Vec<(&str, usize)> = vec![(start, 0)];
        while let Some(&(node, idx)) = stack.last() {
            // `adj` is keyed by every node id, so this lookup always succeeds;
            // the `None` arm is the "no dependencies left" case.
            let next = adj.get(node).and_then(|deps| deps.get(idx)).copied();
            match next {
                None => {
                    state.insert(node, BLACK);
                    stack.pop();
                }
                Some(dep) => {
                    stack.last_mut().expect("stack is non-empty").1 += 1;
                    match state.get(dep).copied() {
                        Some(GREY) => return true,
                        Some(WHITE) => {
                            state.insert(dep, GREY);
                            stack.push((dep, 0));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    false
}

// ── validation entry point ──────────────────────────────────────────────────

/// ProofFlow's `validate_proof_graph`: deserialize each item, dispatch on the
/// id prefix, then `check_dag`.
///
/// PORT FIX: ProofFlow iterates its argument directly, so a reply that is a
/// bare object rather than a list of nodes raises `TypeError` on
/// `item["id"]` — an exception its own retry loop does not catch, so it
/// escapes as a crash. The port returns [`GraphError::Json`] instead, which
/// the retry loop *does* catch, so the same bad reply is simply retried.
///
/// Unknown JSON fields are ignored, matching pydantic's default: the real
/// fixtures carry a `lean_hint` that no model declares.
pub fn validate_proof_graph(data: &Value) -> Result<Validated, GraphError> {
    let items = data.as_array().ok_or_else(|| {
        GraphError::Json(format!(
            "expected a JSON array of proof-graph nodes, found {}",
            type_name(data)
        ))
    })?;

    let mut nodes = Vec::with_capacity(items.len());
    for item in items {
        let node: GraphNode = serde_json::from_value(item.clone()).map_err(|e| {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("<missing id>")
                .to_string();
            GraphError::InvalidNode {
                id,
                reason: e.to_string(),
            }
        })?;
        if node.kind().is_none() {
            return Err(GraphError::UnknownNodeType { id: node.id });
        }
        nodes.push(node);
    }

    check_dag(nodes)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// ProofFlow's `condition_on_all_previous_steps`: rewrite every node's
/// dependencies to be *all* preceding nodes, turning a selectively-depending
/// proof into a fully linear one.
///
/// This is the `follow_dag = False` path — `build_proof_graph`'s no-DAG variant
/// and `prompts/proof_graph_no_DAG.md`. The DAG is still checked first, so a
/// chain that would not pass [`check_dag`] never reaches here.
pub fn condition_on_all_previous_steps(nodes: &mut [GraphNode]) {
    let mut passed: Vec<String> = Vec::with_capacity(nodes.len());
    for node in nodes.iter_mut() {
        node.dependencies = passed.clone();
        passed.push(node.id.clone());
    }
}

// ── the LLM transcript ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One conversation turn.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

impl Turn {
    pub fn user(content: impl Into<String>) -> Turn {
        Turn {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Turn {
        Turn {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// The one thing a graph builder needs from an LLM.
///
/// Implemented by `formalize::llm::LlmClient` (P2). The contract mirrors
/// ProofFlow's `LLMManager.call_llm`: the implementation **appends the
/// assistant's reply to `turns`** and returns its text. That matters, because
/// the retry loop below pushes a correction turn onto the same transcript — a
/// client that returned the reply *without* recording it would lose the
/// assistant's own failed attempt from the conversation, and the retry would
/// be a blind repeat rather than a correction.
pub trait Respond {
    fn respond(&mut self, turns: &mut Vec<Turn>) -> Result<String, String>;
}

#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// `false` selects the no-DAG variant: every node depends on all
    /// predecessors.
    pub follow_dag: bool,
    pub max_retries: usize,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            follow_dag: true,
            max_retries: 3,
        }
    }
}

/// ProofFlow's `build_proof_graph`: ask the model for a proof graph, recover
/// JSON from its reply, validate, and feed the error back until it complies.
///
/// Returns the graph and the 1-based number of attempts that produced it, which
/// ProofFlow returns alongside the graph (the benchmark scripts use it).
///
/// Both failure modes append a correction turn and retry — a parse failure with
/// a generic "could not parse" message, and a validation failure with the
/// specific error, exactly as the Python does.
pub fn build_proof_graph<R: Respond>(
    natural_language_proof: &str,
    llm: &mut R,
    opts: &BuildOptions,
) -> Result<(Validated, usize), GraphError> {
    let mut turns = vec![Turn::user(natural_language_proof)];
    let mut last_error: Option<GraphError> = None;

    for attempt in 0..opts.max_retries {
        let content = llm.respond(&mut turns).map_err(GraphError::Llm)?;
        match parse_llm_json(&content) {
            Ok(data) => match validate_proof_graph(&data) {
                Ok(mut validated) => {
                    if !opts.follow_dag {
                        condition_on_all_previous_steps(&mut validated.nodes);
                    }
                    return Ok((validated, attempt + 1));
                }
                Err(e) => {
                    turns.push(Turn::user(format!(
                        "JSON validation failed. Please fix the following errors: {e} \
                         Please provide a valid JSON structure that matches the \
                         expected format."
                    )));
                    last_error = Some(e);
                }
            },
            Err(e) => {
                turns.push(Turn::user(format!(
                    "Could not parse proof graph JSON from the LLM output. Error: {e}"
                )));
                last_error = Some(e);
            }
        }
    }

    // ProofFlow raises a bare RuntimeError here ("Unexpected error in
    // build_proof_graph") and loses the underlying cause. The port reports the
    // real last failure instead, which is the only way a caller can tell a
    // model that keeps emitting a cycle from one that stopped answering.
    Err(last_error
        .unwrap_or_else(|| GraphError::Llm("no attempt was made: max_retries was 0".to_string())))
}

// ── JSON recovery ───────────────────────────────────────────────────────────

/// ProofFlow's `parse_llm_json`: extract a JSON block, then retry harder.
///
/// Three escalating attempts: strict parse of the extracted block, then the
/// same after [`sanitize_backslashes`], then the same after additionally
/// stripping comments and trailing commas.
///
/// PORT FIX: ProofFlow's third attempt uses `json5.loads`, which also accepts
/// unquoted keys and single quotes. The port stands in for it with
/// [`strip_comments`] + [`strip_trailing_commas`], the two constructs that
/// actually appear in model output; unquoted keys are not accepted.
pub fn parse_llm_json(content: &str) -> Result<Value, GraphError> {
    let raw = extract_json_block(content)?;

    if let Ok(v) = serde_json::from_str::<Value>(&raw) {
        return Ok(v);
    }

    let sanitized = sanitize_backslashes(&raw);
    if let Ok(v) = serde_json::from_str::<Value>(&sanitized) {
        return Ok(v);
    }

    let lenient = strip_trailing_commas(&strip_comments(&sanitized));
    match serde_json::from_str::<Value>(&lenient) {
        Ok(v) => Ok(v),
        Err(e) => Err(GraphError::Json(format!(
            "{e} (block began {})",
            preview(&raw)
        ))),
    }
}

fn preview(s: &str) -> String {
    let head: String = s.chars().take(60).collect();
    format!("{head:?}")
}

/// ProofFlow's `extract_json_block`: four regex strategies, then two scans.
///
/// The strategies are tried in order and the first candidate that both starts
/// with `{` or `[` wins:
///
/// 1. a ` ```json ` fenced block,
/// 2. a bare ` ``` ` fenced block,
/// 3. a ` ```json ` fence whose closing fence is on the opening line,
/// 4. a bare ` ``` ` fence in that same single-line shape.
///
/// Strategies 3 and 4 are subsumed by the fenced-block scanner below.
///
/// The strategies are then, if no fence produced a usable candidate:
///
/// 5. the first brace-balanced `{…}`,
/// 6. a brace-balanced span of at most one nesting level.
///
/// Only strategies 1 and 2 are distinguished in code: the scanner classifies a
/// fence by whether its info string is exactly `json`, which already separates
/// the tagged from the untagged case regardless of line breaks.
///
/// Strategy 6 only ever fires when strategy 5 cannot, i.e. when the braces are
/// unbalanced — an unterminated object inside a truncated reply.
pub fn extract_json_block(text: &str) -> Result<String, GraphError> {
    let blocks = fenced_blocks(text);

    // Strategies 1 and 2, in that order: a `json`-tagged fence is preferred
    // over an untagged one even when the untagged one comes first in the text.
    for want_json in [true, false] {
        for (info, body) in &blocks {
            if info.eq_ignore_ascii_case("json") != want_json {
                continue;
            }
            let candidate = body.trim();
            if candidate.starts_with(['{', '[']) {
                return Ok(candidate.to_string());
            }
        }
    }

    // Strategy 5.
    if let Some(span) = balanced_span(text, usize::MAX) {
        return Ok(span);
    }

    // Strategy 6.
    if let Some(span) = shallow_span(text) {
        return Ok(span);
    }

    Err(GraphError::Json(format!(
        "No JSON block found. Text starts with: {}",
        preview(text)
    )))
}

/// The contents of every ` ``` ` fence, with its info string.
///
/// A fence whose info string is `json5` (or anything else that merely starts
/// with `json`) is *not* reported as a `json` fence, so its body is offered
/// only as an untagged block. In ProofFlow the same reply falls out of the
/// tagged strategies and is recovered by the brace scan instead — the outcome
/// is identical, because both paths reject a body that does not begin with `{`
/// or `[`.
fn fenced_blocks(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while let Some(offset) = text[cursor..].find("```") {
        let open = cursor + offset;
        let rest = &text[open + 3..];
        let newline = rest.find('\n');
        // Byte offset of the body's first character.
        let body_start = match newline {
            Some(k) => open + 3 + k + 1,
            None => open + 3,
        };
        let close = match &text[body_start..].find("```") {
            Some(k) => body_start + k,
            None => break, // unterminated fence
        };
        let body = &text[body_start..close];

        // The info string is whatever precedes the body on the fence line, or —
        // when the closing fence is on the opening line, as in
        // ```json{"a": 1}``` — whatever precedes the first `{`/`[`.
        let (info, body) = match newline {
            Some(k) => (rest[..k].trim().to_string(), body.to_string()),
            None => {
                let seg = &rest[..close - (open + 3)];
                match seg.find(['{', '[']) {
                    Some(k) => (seg[..k].trim().to_string(), seg[k..].to_string()),
                    None => (seg.trim().to_string(), String::new()),
                }
            }
        };

        out.push((info, body));
        cursor = close + 3;
    }

    out
}

/// The first brace-balanced `{…}` span, up to `max_depth` levels of nesting.
///
/// `max_depth == usize::MAX` is strategy 5 (fully balanced); a finite bound is
/// used by [`shallow_span`] for strategy 6.
fn balanced_span(text: &str, max_depth: usize) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => {
                if start.is_none() {
                    start = Some(i);
                    depth = 0;
                }
                depth += 1;
            }
            b'}' => {
                if let Some(s) = start {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[s..=i].to_string());
                    }
                }
            }
            _ => {}
        }
        if depth > max_depth {
            return None;
        }
    }
    None
}

/// Strategy 6: `\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}`, hand-rolled.
///
/// Scans for a `{` whose closing `}` comes at nesting depth at most one, which
/// is what the regex's single optional inner group allows. Written as a scan
/// rather than a regex so the crate gains no dependency for one pattern.
fn shallow_span(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'{' {
            continue;
        }
        let mut depth = 0usize;
        for (j, &c) in bytes.iter().enumerate().skip(i) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[i..=j].to_string());
                    }
                    // The regex cannot match a third level, so stop descending
                    // rather than accept a span it would reject.
                    if depth > 1 {
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// ProofFlow's `sanitize_backslashes`, applied in the same order with the same
/// literals.
///
/// The sequence is not idempotent and its intermediate steps interfere — after
/// `\in` → `\\in`, the later `\int` rule matches *inside* that replacement and
/// adds another pair, and the final rule doubles every remaining backslash. So
/// `\int` gains six backslashes rather than two. That is the reference
/// behaviour, reproduced deliberately: `sanitize_backslash_quirks` pins it, so
/// a future "simplification" of this function fails a test instead of silently
/// changing what parses.
///
/// The target is LaTeX-in-JSON, which is where the doubling is needed: an
/// unescaped `\gcd` is not valid JSON, and escaping the backslash is.
pub fn sanitize_backslashes(raw: &str) -> String {
    let s = raw.replace("\\\\", "\\");
    let s = s.replace("\\in", "\\\\in");
    let s = s.replace("\\Q", "\\\\Q");
    let s = s.replace("\\int", "\\\\int");
    let s = s.replace("\\mathbb{Q}", "\\\\mathbb{Q}");
    s.replace('\\', "\\\\")
}

/// Strip `//` and `/* */` comments, leaving string literals alone.
///
/// Stands in for json5 tolerance on the common model output that comments a
/// value out. String-awareness is the whole requirement: a `//` inside a
/// `natural_language` quote is content, not a comment.
pub fn strip_comments(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let bytes = json.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    let mut escaped = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            // Whole chars, not bytes: a byte-at-a-time copy would mangle any
            // multi-byte character inside a `natural_language` value.
            let ch = json[i..].chars().next().expect("in bounds");
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += ch.len_utf8();
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                out.push('"');
                i += 1;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            _ => {
                // Copy one full UTF-8 char.
                let ch = json[i..].chars().next().expect("in bounds");
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    out
}

/// Remove `,` that immediately precedes a `}` or `]`, ignoring strings.
pub fn strip_trailing_commas(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let bytes = json.as_bytes();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            let ch = json[i..].chars().next().expect("in bounds");
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += ch.len_utf8();
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            continue;
        }
        if b == b',' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'}' || bytes[j] == b']') {
                i += 1; // drop the comma, keep the whitespace
                continue;
            }
        }
        let ch = json[i..].chars().next().expect("in bounds");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Euler's theorem, id 5, from ProofFlow's `data/benchmark_0409.json`. A
    /// real graph: 9 nodes, `tc`/`def`/`l`/`ts` all present, `lean_hint`
    /// fields no model declares.
    const EULER: &str = include_str!("../../testdata/formalize/euler_proof_graph.json");

    fn euler() -> Validated {
        let data: Value = serde_json::from_str(EULER).expect("fixture is valid JSON");
        validate_proof_graph(&data).expect("the real fixture must validate")
    }

    fn node(id: &str, deps: &[&str]) -> GraphNode {
        GraphNode::new(
            id,
            format!("nl for {id}"),
            format!("st for {id}"),
            deps.iter().map(|d| d.to_string()),
        )
    }

    // ── NodeKind dispatch ───────────────────────────────────────────────────

    #[test]
    fn node_kind_dispatches_on_id_prefix() {
        assert_eq!(NodeKind::from_id("ts_1"), Some(NodeKind::TheoremStatement));
        assert_eq!(NodeKind::from_id("l1"), Some(NodeKind::Lemma));
        assert_eq!(NodeKind::from_id("l_99"), Some(NodeKind::Lemma));
        assert_eq!(NodeKind::from_id("def_2"), Some(NodeKind::Definition));
        assert_eq!(NodeKind::from_id("tc_1"), Some(NodeKind::TheoremCondition));
        assert_eq!(NodeKind::from_id("x_1"), None);
        assert_eq!(NodeKind::from_id(""), None);
    }

    /// The `ts` / `ts_` mismatch: `ts1` is a theorem by dispatch, but the
    /// orphan rule does not exempt it, so it must still be referenced.
    #[test]
    fn orphan_exemption_is_the_underscore_prefix() {
        let nodes = vec![
            node("l1", &[]),
            node("ts1", &["l1"]),
            node("ts_1", &["ts1"]),
        ];
        // `ts_1` is the only exempt id, and it is referenced anyway, so this
        // passes.
        assert!(check_dag(nodes).is_ok());

        // Without the final reference, `ts1` is not exempt and is orphaned.
        let nodes = vec![node("l1", &[]), node("ts1", &["l1"])];
        assert_eq!(
            check_dag(nodes).unwrap_err(),
            GraphError::OrphanNode {
                id: "ts1".to_string()
            }
        );
    }

    // ── the real fixture ────────────────────────────────────────────────────

    #[test]
    fn real_euler_fixture_validates() {
        let v = euler();
        assert_eq!(v.len(), 9);
        assert!(
            v.warnings.is_empty(),
            "unexpected warnings: {:?}",
            v.warnings
        );
        assert_eq!(
            v.ids(),
            [
                "tc_1", "def_1", "def_2", "l1", "l2", "l3", "l4", "l5", "ts_1"
            ]
        );
        assert_eq!(v.node("l4").unwrap().kind(), Some(NodeKind::Lemma));
        assert_eq!(
            v.node("ts_1").unwrap().kind(),
            Some(NodeKind::TheoremStatement)
        );
        assert_eq!(v.node("l4").unwrap().dependencies, ["def_2", "l2", "l3"]);
    }

    /// `lean_hint` is present in the fixture and declared by no model, so the
    /// port must ignore it exactly as pydantic does.
    #[test]
    fn unknown_fields_are_ignored_like_pydantic() {
        let json = r#"[
            {"id": "l1", "natural_language": "a", "statement": "b",
             "dependencies": [], "lean_hint": "ring", "weight": 3},
            {"id": "ts_1", "natural_language": "a", "statement": "b",
             "dependencies": ["l1"]}
        ]"#;
        let v = validate_proof_graph(&serde_json::from_str(json).unwrap()).unwrap();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn result_slots_round_trip() {
        let mut v = euler();
        v.node_mut("l1").formalization = Some(serde_json::json!({"cnl": "John adds two three"}));
        v.node_mut("l1").score = Some(serde_json::json!({"semantic_score": 1.0}));
        // `Validated` itself is `{nodes, warnings}`; the report format is the
        // bare node list, which is what `validate_proof_graph` consumes.
        let back: Value = serde_json::to_value(&v.nodes).unwrap();
        let again = validate_proof_graph(&back).unwrap();
        assert_eq!(
            again.node("l1").unwrap().formalization,
            Some(serde_json::json!({"cnl": "John adds two three"}))
        );
        // `skip_serializing_if` keeps an untouched node from growing null slots.
        assert!(!back[0].as_object().unwrap().contains_key("score"));
        // Warnings do not travel: a report that round-trips must not carry
        // Python's `print` output as data.
        assert!(
            !serde_json::to_value(&v)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("warnings")
        );
    }

    #[test]
    fn missing_required_field_is_reported_per_node() {
        let json = r#"[{"id": "l1", "natural_language": "a"}]"#;
        let err = validate_proof_graph(&serde_json::from_str(json).unwrap()).unwrap_err();
        match err {
            GraphError::InvalidNode { id, reason } => {
                assert_eq!(id, "l1");
                assert!(reason.contains("statement"), "unhelpful reason: {reason}");
            }
            other => panic!("expected InvalidNode, got {other:?}"),
        }
    }

    #[test]
    fn unknown_node_type_is_rejected_with_the_four_prefixes() {
        let json = r#"[{"id": "zz_1", "natural_language": "a", "statement": "b"}]"#;
        let err = validate_proof_graph(&serde_json::from_str(json).unwrap()).unwrap_err();
        assert_eq!(
            err,
            GraphError::UnknownNodeType {
                id: "zz_1".to_string()
            }
        );
        let msg = err.to_string();
        for p in ["tc_", "l_", "def_", "ts_"] {
            assert!(msg.contains(p), "{msg} should mention {p}");
        }
    }

    /// PORT FIX, made explicit: a bare object must be a clean error, not a
    /// crash. ProofFlow raises an uncaught `TypeError` here.
    #[test]
    fn a_bare_object_is_an_error_not_a_crash() {
        let data = serde_json::json!({"id": "l1"});
        let err = validate_proof_graph(&data).unwrap_err();
        assert!(matches!(err, GraphError::Json(_)), "{err:?}");
        assert!(err.to_string().contains("an object"), "{err}");
    }

    // ── orphan / cycle / forward reference ─────────────────────────────────

    #[test]
    fn orphan_non_theorem_is_rejected() {
        let nodes = vec![node("l1", &[]), node("l2", &["l1"]), node("ts_1", &["l1"])];
        assert_eq!(
            check_dag(nodes).unwrap_err(),
            GraphError::OrphanNode {
                id: "l2".to_string()
            }
        );
    }

    /// A condition nothing depends on is an orphan too — the rule exempts only
    /// `ts_`, not premises.
    #[test]
    fn an_unused_condition_is_an_orphan() {
        let nodes = vec![node("tc_1", &[]), node("l1", &[]), node("ts_1", &["l1"])];
        assert_eq!(
            check_dag(nodes).unwrap_err(),
            GraphError::OrphanNode {
                id: "tc_1".to_string()
            }
        );
    }

    #[test]
    fn unused_final_theorem_is_exempt() {
        let nodes = vec![node("l1", &[]), node("ts_1", &["l1"])];
        assert!(check_dag(nodes).is_ok());
    }

    #[test]
    fn two_cycle_is_detected() {
        let nodes = vec![
            node("l1", &["l2"]),
            node("l2", &["l1"]),
            node("ts_1", &["l1"]),
        ];
        assert_eq!(check_dag(nodes).unwrap_err(), GraphError::Cycle);
    }

    #[test]
    fn three_cycle_is_detected() {
        let nodes = vec![
            node("l1", &["l3"]),
            node("l2", &["l1"]),
            node("l3", &["l2"]),
            node("ts_1", &["l1"]),
        ];
        assert_eq!(check_dag(nodes).unwrap_err(), GraphError::Cycle);
    }

    /// A self-dependency is a cycle in Python's model, but it is *also* a
    /// forward reference (the id is in `all_ids` but not yet available). Cycle
    /// detection runs first, so the cycle is what is reported.
    #[test]
    fn self_dependency_reports_as_a_cycle() {
        let nodes = vec![node("l1", &["l1"]), node("ts_1", &["l1"])];
        assert_eq!(check_dag(nodes).unwrap_err(), GraphError::Cycle);
    }

    /// A diamond is not a cycle.
    #[test]
    fn diamond_is_accepted() {
        let nodes = vec![
            node("l1", &[]),
            node("l2", &["l1"]),
            node("l3", &["l1"]),
            node("ts_1", &["l2", "l3"]),
        ];
        assert!(check_dag(nodes).is_ok());
    }

    #[test]
    fn forward_reference_is_rejected() {
        let nodes = vec![
            node("l1", &["l2"]),
            node("l2", &[]),
            node("ts_1", &["l1", "l2"]),
        ];
        assert_eq!(
            check_dag(nodes).unwrap_err(),
            GraphError::ForwardReference {
                id: "l1".to_string(),
                dependency: "l2".to_string()
            }
        );
    }

    /// Unknown deps only warn — the graph is still returned.
    #[test]
    fn unknown_dependency_warns_but_keeps_the_graph() {
        let nodes = vec![
            node("l1", &[]),
            node("l2", &["l1", "ghost"]),
            node("ts_1", &["l1", "l2"]),
        ];
        let v = check_dag(nodes).expect("an unknown dep is not fatal");
        assert_eq!(
            v.warnings,
            [GraphWarning::UnknownDependency {
                id: "l2".to_string(),
                dependency: "ghost".to_string()
            }]
        );
    }

    /// The cycle check must not be fooled by an unknown dep, which is neither
    /// available nor a node — it can never be a cycle.
    #[test]
    fn unknown_dependency_is_never_a_cycle() {
        let nodes = vec![node("l1", &["ghost"]), node("ts_1", &["l1"])];
        let v = check_dag(nodes).unwrap();
        assert_eq!(v.warnings.len(), 1);
    }

    /// The cycle pass precedes the forward-reference pass, matching Python.
    #[test]
    fn cycle_is_reported_before_forward_reference() {
        // `l3` is referenced by `ts_1` so the orphan pass lets it through, and
        // `l1`/`l2` are mutually dependent *and* out of order, so both the cycle
        // and a forward reference are present. The cycle wins.
        let nodes = vec![
            node("l1", &["l2"]),
            node("l2", &["l1"]),
            node("l3", &[]),
            node("ts_1", &["l1", "l3"]),
        ];
        assert_eq!(check_dag(nodes).unwrap_err(), GraphError::Cycle);
    }

    /// …and the orphan pass precedes the cycle pass, so an unused node is
    /// reported even when a cycle also exists.
    #[test]
    fn orphan_is_reported_before_cycle() {
        let nodes = vec![
            node("l1", &["l2"]),
            node("l2", &["l1"]),
            node("l3", &[]),
            node("ts_1", &["l1"]),
        ];
        assert_eq!(
            check_dag(nodes).unwrap_err(),
            GraphError::OrphanNode {
                id: "l3".to_string()
            }
        );
    }

    /// A long chain exercises the iterative DFS where recursion would be a
    /// stack-depth risk.
    #[test]
    fn deep_chain_is_accepted_and_its_tail_cycle_rejected() {
        let n = 20_000;
        let nodes: Vec<GraphNode> = (0..n)
            .map(|i| {
                let deps = if i == 0 {
                    Vec::new()
                } else {
                    vec![format!("l{}", i - 1)]
                };
                node(
                    &format!("l{i}"),
                    &deps.iter().map(String::as_str).collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut cyclic = nodes.clone();
        cyclic[0].dependencies = vec![format!("l{}", n - 1)];
        assert!(!has_cycle(&nodes));
        assert!(has_cycle(&cyclic));
    }

    // ── condition_on_all_previous_steps ────────────────────────────────────

    #[test]
    fn condition_on_all_previous_steps_linearizes() {
        let mut nodes = vec![
            node("tc_1", &[]),
            node("l1", &["tc_1"]),
            node("l2", &["def_1"]), // unknown dep: the point of the no-DAG mode
            node("ts_1", &["l1", "l2"]),
        ];
        condition_on_all_previous_steps(&mut nodes);
        assert!(nodes[0].dependencies.is_empty());
        assert_eq!(nodes[1].dependencies, ["tc_1"]);
        assert_eq!(nodes[2].dependencies, ["tc_1", "l1"]);
        assert_eq!(nodes[3].dependencies, ["tc_1", "l1", "l2"]);
    }

    // ── JSON recovery ──────────────────────────────────────────────────────

    #[test]
    fn json_fenced_block_is_preferred() {
        let text = "thinking...\n```json\n[{\"id\": \"l1\"}]\n```\ndone";
        let block = extract_json_block(text).unwrap();
        assert_eq!(block, "[{\"id\": \"l1\"}]");
        assert_eq!(
            parse_llm_json(text).unwrap(),
            serde_json::json!([{"id": "l1"}])
        );
    }

    /// A `json`-tagged fence wins even when an untagged one appears first.
    #[test]
    fn json_tag_beats_an_earlier_untagged_fence() {
        let text = "```\nnot json\n```\n```json\n{\"ok\": true}\n```";
        assert_eq!(
            parse_llm_json(text).unwrap(),
            serde_json::json!({"ok": true})
        );
    }

    #[test]
    fn untagged_fence_is_used_when_no_json_fence() {
        let text = "```\n[{\"id\": \"l1\"}]\n```";
        assert!(parse_llm_json(text).is_ok());
    }

    /// Strategies 3/4: the closing fence on the opening line.
    #[test]
    fn single_line_fence_is_recovered() {
        let text = "answer: ```json[{\"id\": \"l1\"}]``` thanks";
        assert_eq!(
            parse_llm_json(text).unwrap(),
            serde_json::json!([{"id": "l1"}])
        );
    }

    /// The unterminated fence has no closing fence, so it yields no block.
    #[test]
    fn unterminated_fence_falls_back_to_the_first_balanced_object() {
        let text = "```json\n[{\"id\": \"l1\"}]";
        assert_eq!(
            parse_llm_json(text).unwrap(),
            serde_json::json!({"id": "l1"})
        );
    }

    /// The same trap without any fence at all — the common un-fenced reply.
    ///
    /// An unterminated fence leaves no block to recover, so the brace scan
    /// takes over, and it returns the first brace-balanced span it finds: for an
    /// *array* payload that is the first **element**. ProofFlow behaves
    /// identically (its strategy 5 starts at the first `{`), so this is parity
    /// rather than a port bug — but it means an un-fenced array payload is
    /// reported as a *shape* error, not a graph error. Pinned because the
    /// distinction is invisible until a build_proof_graph retry loop reports
    /// "expected an array, found an object" for a reply that plainly contained
    /// an array.
    #[test]
    fn unfenced_array_yields_its_first_element() {
        let text = r#"[{"id": "l1"}, {"id": "ts_1"}]"#;
        assert_eq!(extract_json_block(text).unwrap(), r#"{"id": "l1"}"#);
    }

    /// Strategy 5: no fence at all, just a balanced object in prose.
    #[test]
    fn balanced_object_in_prose_is_found() {
        let text = r#"Here you go: {"a": {"b": [1, 2]}} — hope that helps"#;
        assert_eq!(
            parse_llm_json(text).unwrap(),
            serde_json::json!({"a": {"b": [1, 2]}})
        );
    }

    /// Strategy 6, and a genuine limitation: with the outer brace unbalanced, the
    /// regex matches the *innermost* one-level span, not the outer one.
    ///
    /// `\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}` starts matching at the first `{`, but
    /// the optional group can only absorb a complete inner object plus trailing
    /// non-brace text — and here the outer object's closing brace was already
    /// consumed by that group, leaving nothing for the final `\}`. Backtracking
    /// cannot recover it, so `re.search` advances past the opening brace and
    /// matches the inner object instead. Pinned because the port must fail the
    /// same way, and because the more "useful" behaviour (returning the outer
    /// span) would be a silent divergence from the oracle.
    #[test]
    fn shallow_span_matches_the_innermost_one_level_span() {
        let text = r#"result: {"a": {"b": 1}"#;
        let span = extract_json_block(text).unwrap();
        assert_eq!(span, r#"{"b": 1}"#);
    }

    #[test]
    fn no_json_at_all_is_an_error_naming_the_head() {
        let err = extract_json_block("I refuse to answer.").unwrap_err();
        assert!(matches!(err, GraphError::Json(_)));
        assert!(err.to_string().contains("I refuse to answer"), "{err}");
    }

    /// The real reason `sanitize_backslashes` exists: LaTeX in a JSON string,
    /// unescaped.
    #[test]
    fn latex_backslashes_are_recovered() {
        let broken = r#"[{"id": "l1", "statement": "$\gcd(x,n) = 1$"}]"#;
        assert!(serde_json::from_str::<Value>(broken).is_err());
        let value = parse_llm_json(&format!("```json\n{broken}\n```")).unwrap();
        let stmt = value[0]["statement"].as_str().unwrap();
        assert!(stmt.contains("\\gcd"), "{stmt}");
    }

    /// The interference between the rules, pinned so the port cannot drift.
    #[test]
    fn sanitize_backslash_quirks() {
        // Plain doubling.
        assert_eq!(sanitize_backslashes(r"\gcd"), r"\\gcd");
        // Already doubled is normalized back to a single backslash by the first
        // rule, then doubled once — so both spellings agree.
        assert_eq!(
            sanitize_backslashes(r"\\gcd"),
            sanitize_backslashes(r"\gcd")
        );
        // `\in` -> `\\in`, then the later `\int` rule matches *inside* that
        // result, then everything is doubled: six backslashes, not two.
        // Reproduced on purpose — see the function's docs.
        assert_eq!(sanitize_backslashes(r"\int"), r"\\\\\\int");
        // `\mathbb{Q}` is matched by its own rule before the blanket doubling,
        // and its `\m` is not a prefix of any earlier rule, so it only doubles
        // twice.
        assert_eq!(sanitize_backslashes(r"\mathbb{Q}"), r"\\\\mathbb{Q}");
        // A backslash-free string is untouched.
        assert_eq!(sanitize_backslashes(r#"{"a": 1}"#), r#"{"a": 1}"#);
        // The function is *not* idempotent: each pass collapses doubled pairs
        // and then re-doubles, so applying it to its own output strictly
        // increases the escaping (6 backslashes in, 14 out for `\int`).
        // `parse_llm_json` applies it exactly once, and this pins that — a
        // refactor that applied it twice would silently re-escape valid JSON
        // into invalid JSON. Counted rather than spelled out, so the
        // invariant survives the literal being retyped.
        let once = sanitize_backslashes(r"\int");
        let twice = sanitize_backslashes(&once);
        let bs = |s: &str| s.matches('\\').count();
        assert_eq!(bs(&once), 6);
        assert!(bs(&twice) > bs(&once), "{:?} -> {:?}", once, twice);
    }

    #[test]
    fn comments_and_trailing_commas_are_tolerated() {
        let text = r#"```json
[
  // the premise
  {"id": "l1", "statement": "a // b", "dependencies": [],},
]
```"#;
        let value = parse_llm_json(text).unwrap();
        assert_eq!(value[0]["statement"].as_str(), Some("a // b"));
    }

    #[test]
    fn comment_stripping_respects_strings_and_unicode() {
        let json = r#"{"a": "x // y", "b": "π/2 /* z */", "c": 1}"#;
        let out = strip_comments(json);
        assert!(out.contains("x // y"), "{out}");
        assert!(out.contains("π/2 /* z */"), "{out}");
        assert!(serde_json::from_str::<Value>(&out).is_ok());
        assert_eq!(strip_trailing_commas("[1, 2, ]"), "[1, 2 ]");
        // A comma inside a string is not a trailing comma.
        assert_eq!(strip_trailing_commas(r#"["a, "]"#), r#"["a, "]"#);
    }

    // ── build_proof_graph ──────────────────────────────────────────────────

    /// A scripted [`Respond`] so the retry loop can be tested without a model.
    struct Scripted {
        replies: Vec<String>,
        n: usize,
        turn_log: Vec<usize>,
    }

    impl Respond for Scripted {
        fn respond(&mut self, turns: &mut Vec<Turn>) -> Result<String, String> {
            self.turn_log.push(turns.len());
            let reply = self.replies[self.n.min(self.replies.len() - 1)].clone();
            self.n += 1;
            turns.push(Turn::assistant(&reply));
            Ok(reply)
        }
    }

    #[test]
    fn build_succeeds_on_the_first_valid_reply() {
        let mut llm = Scripted {
            replies: vec![format!("```json\n{EULER}\n```")],
            n: 0,
            turn_log: vec![],
        };
        let (v, attempts) = build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap();
        assert_eq!(attempts, 1);
        assert_eq!(v.len(), 9);
        assert!(v.warnings.is_empty());
        // The proof was the only user turn; the reply was recorded by the client.
        assert_eq!(llm.turn_log, [1]);
    }

    #[test]
    fn build_feeds_the_parse_error_back_and_retries() {
        let mut llm = Scripted {
            replies: vec!["no json here".to_string(), format!("```json\n{EULER}\n```")],
            n: 0,
            turn_log: vec![],
        };
        let (v, attempts) = build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn build_feeds_the_validation_error_back_and_retries() {
        let cyclic = r#"[{"id":"l1","natural_language":"a","statement":"b","dependencies":["ts_1"]},
                        {"id":"ts_1","natural_language":"a","statement":"b","dependencies":["l1"]}]"#;
        let mut llm = Scripted {
            replies: vec![cyclic.to_string(), format!("```json\n{EULER}\n```")],
            n: 0,
            turn_log: vec![],
        };
        let (v, attempts) = build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(v.len(), 9);
    }

    /// The transcript must accumulate: attempt 2 sees the failed reply *and*
    /// the correction, which is the whole reason the client records turns.
    #[test]
    fn build_transcript_accumulates_corrections() {
        struct Recorder {
            replies: Vec<String>,
            seen: Vec<usize>,
        }
        impl Respond for Recorder {
            fn respond(&mut self, turns: &mut Vec<Turn>) -> Result<String, String> {
                self.seen.push(turns.len());
                let reply = self.replies[self.seen.len() - 1].clone();
                turns.push(Turn::assistant(&reply));
                Ok(reply)
            }
        }
        let mut llm = Recorder {
            replies: vec!["nope".into(), format!("```json\n{EULER}\n```")],
            seen: vec![],
        };
        build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap();
        assert_eq!(llm.seen, [1, 3]);
    }

    #[test]
    fn build_exhausts_retries_and_reports_the_real_cause() {
        // Fenced, so the extraction returns the array rather than its first
        // element (see `unfenced_array_yields_its_first_element`).
        let cyclic = format!(
            "```json\n{}\n```",
            r#"[{"id":"l1","natural_language":"a","statement":"b","dependencies":["ts_1"]},
                {"id":"ts_1","natural_language":"a","statement":"b","dependencies":["l1"]}]"#
        );
        let mut llm = Scripted {
            replies: vec![cyclic],
            n: 0,
            turn_log: vec![],
        };
        let err = build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap_err();
        // Python raises a bare RuntimeError here and loses this.
        assert_eq!(err, GraphError::Cycle);
        assert_eq!(llm.n, 3, "max_retries is 3 attempts");
    }

    /// The same graph delivered *unfenced* never reaches validation: the brace
    /// scan hands back the first element, so the failure is a shape error. This
    /// is the failure mode a user sees from a model that forgets its fence, and
    /// it must not be confused with a malformed graph.
    #[test]
    fn build_reports_a_shape_error_for_an_unfenced_array() {
        let cyclic = r#"[{"id":"l1","natural_language":"a","statement":"b","dependencies":["ts_1"]},
                        {"id":"ts_1","natural_language":"a","statement":"b","dependencies":["l1"]}]"#;
        let mut llm = Scripted {
            replies: vec![cyclic.to_string()],
            n: 0,
            turn_log: vec![],
        };
        let err = build_proof_graph("proof", &mut llm, &BuildOptions::default()).unwrap_err();
        assert!(matches!(err, GraphError::Json(_)), "{err:?}");
    }

    #[test]
    fn build_propagates_a_transport_failure() {
        struct Broken;
        impl Respond for Broken {
            fn respond(&mut self, _t: &mut Vec<Turn>) -> Result<String, String> {
                Err("connection refused".into())
            }
        }
        let err = build_proof_graph("proof", &mut Broken, &BuildOptions::default()).unwrap_err();
        assert_eq!(err, GraphError::Llm("connection refused".into()));
    }

    /// `follow_dag: false` is the no-DAG mode: validation still runs first, so
    /// a graph that only becomes linear afterwards is still rejected.
    #[test]
    fn no_dag_mode_linearizes_after_validation() {
        let mut llm = Scripted {
            replies: vec![format!("```json\n{EULER}\n```")],
            n: 0,
            turn_log: vec![],
        };
        let opts = BuildOptions {
            follow_dag: false,
            max_retries: 3,
        };
        let (v, _) = build_proof_graph("proof", &mut llm, &opts).unwrap();
        assert_eq!(
            v.node("l4").unwrap().dependencies,
            ["tc_1", "def_1", "def_2", "l1", "l2", "l3"]
        );
        assert_eq!(v.node("ts_1").unwrap().dependencies.len(), 8);
    }

    #[test]
    fn zero_retries_is_an_error_not_a_panic() {
        let mut llm = Scripted {
            replies: vec!["x".into()],
            n: 0,
            turn_log: vec![],
        };
        let opts = BuildOptions {
            follow_dag: true,
            max_retries: 0,
        };
        let err = build_proof_graph("proof", &mut llm, &opts).unwrap_err();
        assert!(matches!(err, GraphError::Llm(_)), "{err:?}");
    }
}
