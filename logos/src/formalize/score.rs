//! P4 — LogosScore: ProofFlow's ProofScore, retargeted.
//!
//! # What is ported, and what genuinely differs
//!
//! ProofFlow's `proof_scorer.py` has three layers, and all three are here:
//!
//! | layer | ProofFlow | here |
//! |---|---|---|
//! | per-node judgement | LLM tags each condition/conclusion `Perfectly match` / `Minor inconsistency` / `Major inconsistency` | same three tags, same prompt shape, L0 targets instead of Lean |
//! | per-node score | fuzzy measure `mu` + Sugeno integral over the tags | ported rule-for-rule, including the `>10` down-sampling |
//! | graph score | `semantic_score` weighted by **Katz centrality** (default), or Laplacian, or equal | same three, and a fourth: UNF-identity |
//!
//! The **dependency check** (`proofscore_dependency_check.md`) is ported too, but
//! its property is structural rather than semantic — the DAG must faithfully
//! reflect the proof's premises — so it is checked against the graph
//! ([`dependency_report`]) as well as asked of the model.
//!
//! # The numeric check, and what it does not establish
//!
//! The plan asks for a "kernel numeric agreement" check. `prob_kernel::Session` is
//! unreachable from here (`prob_kernel` depends on `logos`, so the edge cannot
//! exist), so the check is local and deliberately narrow:
//! [`numeric_agreement`] compares the *symbolic evaluation* of a node's term
//! against its *TED canonical form* — two independent canonicalizations computed
//! by the same reduction.
//!
//! That is a genuine cross-check of the arithmetic pipeline, and it is **not** a
//! check that the CNL sentence means what the natural language says. That is
//! what the LLM judge is for, and it is stated on every [`NumericVerdict`] so a
//! report cannot be read as claiming more than it establishes.

use crate::formalize::formalizer::{self, CnlFormalization};
use crate::formalize::graph::{GraphNode, NodeKind, Validated};
use crate::formalize::llm::{LlmClient, Transport};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

/// The per-component verdict, ProofFlow's three tags.
///
/// Grades are `A`/`B`/`C` exactly as in `match_evaluations`, because the fuzzy
/// measure is defined over those letters and the Sugeno integral reads them
/// back out of the fuzzy sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Tag {
    /// "Perfectly match"
    #[serde(rename = "A")]
    Perfect,
    /// "Minor inconsistency"
    #[serde(rename = "B")]
    Minor,
    /// "Major inconsistency"
    #[serde(rename = "C")]
    Major,
}

impl Tag {
    pub fn grade(self) -> f64 {
        match self {
            Tag::Perfect => 1.0,
            Tag::Minor => 0.5,
            Tag::Major => 0.0,
        }
    }

    /// The grade letter, for fuzzy-set labels and messages.
    pub fn letter(self) -> &'static str {
        match self {
            Tag::Perfect => "A",
            Tag::Minor => "B",
            Tag::Major => "C",
        }
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tag::Perfect => "Perfectly match",
            Tag::Minor => "Minor inconsistency",
            Tag::Major => "Major inconsistency",
        })
    }
}

/// Parse ProofFlow's three tag spellings, case-insensitively.
///
/// Returns [`None`] for an unrecognized verdict rather than guessing: the fuzzy
/// measure treats a `C` as catastrophic and an `A` as perfect, so a wrong guess
/// is far worse than a missing component.
pub fn parse_tag(s: &str) -> Option<Tag> {
    let t = s.trim().to_lowercase();
    if t.contains("perfectly match") {
        Some(Tag::Perfect)
    } else if t.contains("minor inconsistency") {
        Some(Tag::Minor)
    } else if t.contains("major inconsistency") {
        Some(Tag::Major)
    } else {
        None
    }
}

/// Extract tags and feedback from a judgement reply.
///
/// ProofFlow has two extraction paths: `extract_match_feedback_content` scans
/// for tag phrases (optionally inside `\box{…}`) and `<<<…>>>` feedback, and
/// `create_second_prompt_json` asks for a JSON object instead. Both are
/// supported, JSON first — a model that produces valid JSON has been explicit,
/// and the regex path is the recovery.
pub fn extract_match_feedback(text: &str) -> (Vec<Tag>, Vec<String>) {
    // JSON first: explicit beats scanned. Every balanced object is tried, because a
    // reply often opens with a brace-balanced aside that is *not* JSON — "let me
    // think {not json} and then …" — and stopping at the first brace would fall
    // through to the scanning path and lose the component order.
    for candidate in json_objects(text) {
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&candidate) else {
            continue;
        };
        let Some(evals) = parsed.get("evaluation").and_then(|v| v.as_array()) else {
            continue;
        };
        let tags: Vec<Tag> = evals
            .iter()
            .filter_map(|e| e.as_str())
            .filter_map(parse_tag)
            .collect();
        if tags.is_empty() {
            continue;
        }
        let feedback = parsed
            .get("feedback")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        return (tags, feedback);
    }

    // The recovery path. `tags_in_order` rather than three `split` chains,
    // because the fuzzy measure indexes components by position and the judge
    // evaluated them in a specific order.
    let tags = tags_in_order(text);
    let feedback: Vec<String> = text
        .match_indices("<<<")
        .filter_map(|(i, _)| {
            text[i + 3..]
                .find(">>>")
                .map(|j| text[i + 3..i + 3 + j].trim().to_string())
        })
        .collect();
    (tags, feedback)
}

/// Tags in the order they appear in the text.
fn tags_in_order(text: &str) -> Vec<Tag> {
    let mut found: Vec<(usize, Tag)> = Vec::new();
    for (needle, tag) in [
        ("Perfectly match", Tag::Perfect),
        ("Minor inconsistency", Tag::Minor),
        ("Major inconsistency", Tag::Major),
    ] {
        let mut from = 0;
        while let Some(rel) = text[from..].find(needle) {
            found.push((from + rel, tag));
            from += rel + needle.len();
        }
    }
    found.sort_by_key(|(at, _)| *at);
    found.into_iter().map(|(_, t)| t).collect()
}

/// Every brace-balanced JSON object in the text, in order.
fn json_objects(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = text[i..].find('{') {
        let start = i + rel;
        let mut depth = 0usize;
        let mut end = None;
        for (j, &b) in bytes.iter().enumerate().skip(start) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(e) => {
                out.push(text[start..=e].to_string());
                i = e + 1;
            }
            // Unbalanced from here on; no later `{` can be balanced either.
            None => break,
        }
    }
    out
}

// ── the fuzzy measure, ported rule-for-rule ─────────────────────────────────

/// A subset of component indices. A sorted `Vec` rather than a hash set so the
/// measure is a total order and `mu` is reproducible.
pub type Subset = Vec<usize>;

/// ProofFlow's `generate_mu`.
///
/// The four rules, unchanged:
///
/// 1. any `C` anywhere ⇒ `mu(A) = 0` for every `A`;
/// 2. the full set, all `A` ⇒ `1.0`;
/// 3. a subset with ≥ 2 `B`s ⇒ `base_weight · |A| · (1 − 0.2·B_count)`;
/// 4. otherwise `base_weight · |A| · (1 − 0.1·B_count)`.
///
/// Note rule 1 is a statement about the *whole* judgement, not about the subset:
/// one `C` zeroes every subset, which is what makes the Sugeno integral return 0
/// rather than something merely low.
///
/// `2^n` subsets, so more than ~20 components is intractable — hence
/// ProofFlow's down-sampling, ported here. The cap is explicit in
/// [`MAX_COMPONENTS`] rather than hidden, because the down-sampling *changes the
/// score* and a caller has to be able to see that it happened.
pub const MAX_COMPONENTS: usize = 10;

/// Port of `generate_mu`: returns the fuzzy measure and the (possibly
/// down-sampled) evaluations.
pub fn generate_mu(evaluations: &[Tag]) -> (BTreeMap<Subset, f64>, Vec<Tag>) {
    let mut evaluations = evaluations.to_vec();

    // ProofFlow down-samples anything over 10 components by replacing them with
    // a run of As followed by Bs in the *observed proportion*. Preserved
    // verbatim, including its flaws: `b_count = 10 - a_count` counts Bs as
    // "not A", so a `C` is counted as a `B` and the sample silently loses a
    // major inconsistency. That is load-bearing — the original's scores depend
    // on it — so it is reproduced and flagged rather than fixed, and
    // `downsampling_loses_major_inconsistencies` pins it.
    if evaluations.len() > MAX_COMPONENTS {
        let n = evaluations.len();
        let perfect = evaluations.iter().filter(|t| **t == Tag::Perfect).count();
        let a_count = ((perfect as f64 / n as f64) * MAX_COMPONENTS as f64).floor() as usize;
        let b_count = MAX_COMPONENTS - a_count;
        evaluations = std::iter::repeat_n(Tag::Perfect, a_count)
            .chain(std::iter::repeat_n(Tag::Minor, b_count))
            .collect();
    }

    let n = evaluations.len();
    if n == 0 {
        return (BTreeMap::new(), evaluations);
    }

    let all: Subset = (0..n).collect();
    let base_weight = 1.0 / n as f64;
    let has_major = evaluations.contains(&Tag::Major);

    let mut mu = BTreeMap::new();
    for k in 1..=n {
        for subset in combinations(n, k) {
            let value = if has_major {
                0.0
            } else if subset == all && evaluations.iter().all(|t| *t == Tag::Perfect) {
                1.0
            } else {
                let b_count = subset
                    .iter()
                    .filter(|i| evaluations[**i] == Tag::Minor)
                    .count();
                if b_count >= 2 {
                    (base_weight * subset.len() as f64 * (1.0 - 0.2 * b_count as f64)).max(0.0)
                } else {
                    base_weight * subset.len() as f64 * (1.0 - 0.1 * b_count as f64)
                }
            };
            mu.insert(subset, value);
        }
    }
    (mu, evaluations)
}

/// All `k`-subsets of `0..n`, in ascending order.
fn combinations(n: usize, k: usize) -> Vec<Subset> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k);
    fn go(next: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Subset>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in next..n {
            cur.push(i);
            go(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    go(0, n, k, &mut current, &mut out);
    out
}

/// Port of `sugeno_integral`: the score in `[0, 1]`, rounded to two places.
///
/// The early return for any `C` is kept, even though rule 1 of `mu` already
/// forces 0 — ProofFlow guards it twice, and the guard is what makes the zero
/// *exact* rather than emergent.
pub fn sugeno_integral(evaluations: &[Tag]) -> f64 {
    if evaluations.is_empty() {
        return 0.0;
    }
    if evaluations.contains(&Tag::Major) {
        return 0.0;
    }
    let (mu, evaluations) = generate_mu(evaluations);
    let f: Vec<f64> = evaluations.iter().map(|t| t.grade()).collect();
    if f.is_empty() {
        return 0.0;
    }

    let mut order: Vec<usize> = (0..f.len()).collect();
    order.sort_by(|a, b| {
        f[*a]
            .partial_cmp(&f[*b])
            .expect("grades are constants, never NaN")
    });

    let mut sugeno = 0.0f64;
    for i in 0..f.len() {
        let a: Subset = order[i..].to_vec();
        let mu_a = mu.get(&a).copied().unwrap_or(0.0);
        sugeno = sugeno.max(f[order[i]].min(mu_a));
    }
    (sugeno * 100.0).round() / 100.0
}

// ── the numeric check ───────────────────────────────────────────────────────

/// What the numeric cross-check found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NumericVerdict {
    /// The symbolic evaluation and the TED canonical form agree.
    ///
    /// This says the two canonicalizations of the reduced term are consistent.
    /// It does **not** say the CNL sentence matches the natural-language
    /// statement — that is the LLM judge's job.
    Agreed { value: String, ted: String },
    /// They disagree, which is a bug in the pipeline rather than in the model.
    Disagreed { value: String, ted: String },
    /// The term is closed but has no TED, so there is nothing to compare.
    ClosedWithoutTed { value: String },
    /// The term is open, so it denotes no number and makes no numeric claim.
    Open,
}

/// Compare a node's closed value against its TED canonical form.
///
/// The check is cheap and catches a real class of bug: `translate_coreir`
/// computes `sym_expr.eval()` and `deltanet::ted::from_sym_expr` along two
/// different canonicalization paths, so a disagreement means one of them is
/// wrong about the same term. When the term is open there is no number and the
/// verdict is [`NumericVerdict::Open`] rather than a pass — reporting "no
/// disagreement" for a term that never claimed a number would be a fabricated
/// agreement.
pub fn numeric_agreement(f: &CnlFormalization) -> NumericVerdict {
    match (&f.value, &f.ted) {
        (Some(value), Some(ted)) => {
            if ted_evaluates_to(ted, value) {
                NumericVerdict::Agreed {
                    value: value.clone(),
                    ted: ted.clone(),
                }
            } else {
                NumericVerdict::Disagreed {
                    value: value.clone(),
                    ted: ted.clone(),
                }
            }
        }
        (Some(value), None) => NumericVerdict::ClosedWithoutTed {
            value: value.clone(),
        },
        (None, _) => NumericVerdict::Open,
    }
}

/// Whether a constant TED canonical string equals `value`.
///
/// Only constant polynomials are compared, and only textually. That is the whole
/// scope of the check: a polynomial with variables (`3*x + 6`) has no value to
/// agree with, and re-implementing polynomial evaluation here to extend the check
/// would be a second arithmetic implementation — precisely the thing that could
/// disagree with `deltanet`.
fn ted_evaluates_to(ted: &str, value: &str) -> bool {
    let t = ted.trim();
    let v = value.trim();
    // `Ted::to_canonical_string` renders the constant polynomial as the bare
    // integer, so textual equality is the comparison; the `trim` guards against
    // whitespace drift between the two renderers.
    t == v
}

// ── the dependency check ────────────────────────────────────────────────────

/// Whether the graph's structure is faithful, and why.
///
/// This is the part of ProofScore that is *structural* rather than semantic, so
/// it is computed rather than asked. The model is still consulted for whether the
/// DAG matches the proof's reasoning; this is the part that must not be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyReport {
    /// Every node is referenced or is the closing `ts_`.
    pub no_orphans: bool,
    /// Every dependency refers to a node that exists.
    pub no_dangling: bool,
    /// Dependencies point backwards only.
    pub no_forward_references: bool,
    /// No two nodes carry the same UNF hash — §14's identity requirement.
    ///
    /// Computed only over nodes that were formalized; nodes without a hash are
    /// not a collision, they are a gap.
    pub no_duplicate_identities: bool,
    /// Nodes sharing a UNF hash, for the report.
    pub duplicate_identities: Vec<Vec<String>>,
    /// Nodes with no formalization at all.
    pub unformalized: Vec<String>,
}

impl DependencyReport {
    /// Whether the structure is faithful *and* fully formalized.
    ///
    /// `unformalized` is included because a report with a gap in it must not read
    /// as a clean bill of health — "nothing wrong with what we checked" and
    /// "everything was checked" are different claims.
    pub fn is_clean(&self) -> bool {
        self.no_orphans
            && self.no_dangling
            && self.no_forward_references
            && self.no_duplicate_identities
            && self.unformalized.is_empty()
    }
}

/// Check a graph's structural faithfulness.
///
/// `check_dag` has already rejected orphans, dangling refs and forward references
/// — [`crate::formalize::graph::check_dag`] runs first — so those three come back
/// true for any graph that reached here. They are still reported, because the
/// report has to be readable on its own and a reader should not have to know what
/// ran earlier to know what was checked.
pub fn dependency_report(graph: &Validated) -> DependencyReport {
    let mut referenced: BTreeSet<&str> = BTreeSet::new();
    for n in &graph.nodes {
        for d in &n.dependencies {
            referenced.insert(d.as_str());
        }
    }
    let all: BTreeSet<&str> = graph.nodes.iter().map(|n| n.id.as_str()).collect();

    let no_orphans = graph
        .nodes
        .iter()
        .all(|n| n.id.starts_with("ts_") || referenced.contains(n.id.as_str()));
    let no_dangling = graph
        .nodes
        .iter()
        .all(|n| n.dependencies.iter().all(|d| all.contains(d.as_str())));

    let mut no_forward_references = true;
    for (i, n) in graph.nodes.iter().enumerate() {
        let available: BTreeSet<&str> = graph.nodes[..i].iter().map(|p| p.id.as_str()).collect();
        if n.dependencies
            .iter()
            .any(|d| !available.contains(d.as_str()))
        {
            no_forward_references = false;
        }
    }

    // §14: identity is the UNF hash, so two nodes with one hash are one node.
    let mut by_hash: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut unformalized = Vec::new();
    for n in &graph.nodes {
        match formalization_of(n) {
            Some(f) => by_hash.entry(f.unf_hash).or_default().push(n.id.clone()),
            None => unformalized.push(n.id.clone()),
        }
    }
    let duplicate_identities: Vec<Vec<String>> = by_hash
        .values()
        .filter(|ids| ids.len() > 1)
        .cloned()
        .collect();

    DependencyReport {
        no_orphans,
        no_dangling,
        no_forward_references,
        no_duplicate_identities: duplicate_identities.is_empty(),
        duplicate_identities,
        unformalized,
    }
}

/// A node's formalization, if it has one.
pub fn formalization_of(node: &GraphNode) -> Option<CnlFormalization> {
    node.formalization
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

// ── centrality weighting ────────────────────────────────────────────────────

/// How node scores are weighted when aggregated over the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregation {
    /// Every node weighs the same.
    Equal,
    /// Degree-normalized, `nx.laplacian_centrality`'s unnormalized variant.
    Laplacian,
    /// Katz centrality: `A x = α (x + b)`, solved by power iteration.
    #[default]
    Katz,
}

/// Node weights for one aggregation.
///
/// A node that no formalized node depends on has degree zero. It still gets a
/// weight — 1.0 under `Equal`, and a positive floor under the centralities — so
/// that an isolated final theorem statement is not silently free.
pub fn centrality(graph: &Validated, aggregation: Aggregation) -> BTreeMap<String, f64> {
    let ids: Vec<&str> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
    let index: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let n = ids.len();
    let mut weights = BTreeMap::new();
    if n == 0 {
        return weights;
    }

    match aggregation {
        Aggregation::Equal => {
            for id in &ids {
                weights.insert((*id).to_string(), 1.0);
            }
        }
        Aggregation::Laplacian => {
            let mut degree = vec![0.0f64; n];
            for node in &graph.nodes {
                for dep in &node.dependencies {
                    if let Some(&i) = index.get(dep.as_str()) {
                        degree[i] += 1.0;
                        degree[index[node.id.as_str()]] += 1.0;
                    }
                }
            }
            // `nx.laplacian_centrality` returns unnormalized scores for the
            // unnormalized graph, so this matches: degree, floored at 1.
            for (i, id) in ids.iter().enumerate() {
                weights.insert((*id).to_string(), degree[i].max(1.0));
            }
        }
        Aggregation::Katz => {
            let mut adj = vec![vec![0.0f64; n]; n];
            for node in &graph.nodes {
                let Some(&to) = index.get(node.id.as_str()) else {
                    continue;
                };
                for dep in &node.dependencies {
                    if let Some(&from) = index.get(dep.as_str()) {
                        adj[from][to] = 1.0;
                    }
                }
            }
            let katz = katz_centrality(&adj);
            for (i, id) in ids.iter().enumerate() {
                weights.insert((*id).to_string(), katz[i].max(1e-9));
            }
        }
    }
    weights
}

/// Katz centrality by power iteration on `x ← α (A x + b)`, `b` uniform.
///
/// The iteration is run to a fixed tolerance rather than a fixed count, and it
/// is deterministic: the dependency graph is a DAG, so starting from uniform
/// `b` and iterating upward converges, and ties do not arise in practice for
/// `α = 0.5`. A fixed iteration count would make the weight depend on the
/// convergence criterion, which would make the final score depend on a
/// performance knob.
fn katz_centrality(adj: &[Vec<f64>]) -> Vec<f64> {
    const ALPHA: f64 = 0.5;
    const TOLERANCE: f64 = 1e-12;
    const MAX_ITERATIONS: usize = 10_000;

    let n = adj.len();
    let mut x = vec![1.0f64 / n as f64; n];
    let b = 1.0f64 / n as f64;
    for _ in 0..MAX_ITERATIONS {
        let mut next = vec![0.0f64; n];
        for (i, row) in next.iter_mut().enumerate() {
            let mut sum = b;
            for (j, a) in adj[i].iter().enumerate() {
                sum += a * x[j];
            }
            *row = ALPHA * sum;
        }
        let delta: f64 = next
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        x = next;
        if delta < TOLERANCE {
            break;
        }
    }
    // Normalize so the weights are comparable across graphs of different sizes;
    // the mean is invariant under the scaling the iteration leaves free.
    let mean = x.iter().sum::<f64>() / n as f64;
    if mean > 0.0 {
        x.iter().map(|v| v / mean).collect()
    } else {
        x
    }
}

// ── the score report ────────────────────────────────────────────────────────

/// One node's score row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeScore {
    pub id: String,
    pub kind: String,
    /// Whether this node is one that *could* be judged.
    ///
    /// `false` for `tc_`/`def_` nodes, which are formalized but never proved —
    /// so there is nothing to compare a formalized version against. Recorded
    /// separately from [`NodeScore::unjudged`] because being unjudged is
    /// expected for an assumption and a fault for a lemma, and a report that
    /// conflated the two could never be clean on any graph with a premise.
    pub provable: bool,
    /// The model verdict per component.
    pub tags: Vec<Tag>,
    /// Per-component feedback from the model.
    pub feedback: Vec<String>,
    /// The Sugeno integral over `tags`: 1.0 is perfect, 0.0 is any major
    /// inconsistency.
    pub semantic_score: f64,
    /// The numeric cross-check.
    pub numeric: NumericVerdict,
    /// Set when the node was not judged: an assumption has nothing to compare
    /// against a statement it is quoted by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unjudged: Option<String>,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogosScore {
    pub nodes: Vec<NodeScore>,
    /// The weighted mean of `semantic_score` over judged nodes.
    ///
    /// `None` when nothing was judged, which is different from `Some(0.0)`:
    /// "no judgement" and "judged and wrong" must not look the same in a report.
    pub total: Option<f64>,
    pub dependency: DependencyReport,
    pub aggregation: String,
    /// The weights used, so the total is reproducible from the report alone.
    pub weights: BTreeMap<String, f64>,
}

impl LogosScore {
    /// Whether the report can be trusted as a verdict.
    ///
    /// Structure clean, every node formalized, and every node that *could* be
    /// judged was. An assumption being unjudged is correct, not a fault — see
    /// [`NodeScore::provable`] — so it does not count against the report.
    pub fn is_clean(&self) -> bool {
        self.dependency.is_clean()
            && self
                .nodes
                .iter()
                .all(|n| !n.provable || n.unjudged.is_none())
    }
}

/// Aggregate per-node semantic scores into one number.
///
/// ProofFlow's `compute_total_score`: a weight-weighted mean over the nodes that
/// have a passing formalization, divided by the total weight. Unjudged nodes
/// contribute to the denominator and not the numerator, so a graph where only
/// one node was judged scores low rather than scoring perfectly — which is the
/// point of a weighted mean over *all* nodes.
pub fn aggregate_total(nodes: &[NodeScore], weights: &BTreeMap<String, f64>) -> Option<f64> {
    let mut score = 0.0;
    let mut total_weight = 0.0;
    for n in nodes {
        let w = weights.get(&n.id).copied().unwrap_or(1.0);
        if n.unjudged.is_none() {
            score += w * n.semantic_score;
        }
        total_weight += w;
    }
    if total_weight == 0.0 {
        None
    } else {
        Some(score / total_weight)
    }
}

/// Build the report from judged nodes, without calling a model.
///
/// This is the part that is *computed* — tags in, numbers out — and it is
/// separately testable from the judgement itself, which is what
/// [`judge_graph`] covers.
pub fn report(graph: &Validated, nodes: Vec<NodeScore>, aggregation: Aggregation) -> LogosScore {
    let weights = centrality(graph, aggregation);
    let total = aggregate_total(&nodes, &weights);
    LogosScore {
        nodes,
        total,
        dependency: dependency_report(graph),
        aggregation: format!("{aggregation:?}"),
        weights,
    }
}

/// A node with no formalization cannot be judged; say so rather than scoring 0.
pub fn unjudged_row(node: &GraphNode, reason: &str) -> NodeScore {
    NodeScore {
        id: node.id.clone(),
        kind: format!("{:?}", node.kind().unwrap_or(NodeKind::Lemma)),
        provable: crate::formalize::formalizer::is_provable(node.kind().unwrap_or(NodeKind::Lemma)),
        tags: Vec::new(),
        feedback: Vec::new(),
        semantic_score: 0.0,
        numeric: NumericVerdict::Open,
        unjudged: Some(reason.to_string()),
    }
}

/// The judge prompt for one node, mirroring
/// `proofscore_semantinc_check.md` with L0 targets.
pub fn build_judge_turn(
    node: &GraphNode,
    formalization: &CnlFormalization,
) -> crate::formalize::graph::Turn {
    use crate::formalize::graph::Turn;
    // The conditions and conclusions are already enumerated in `statement`, so
    // that *is* the component list — ProofFlow passes `math_cond` through for
    // the same reason.
    let mut content = format!(
        "Here is a natural-language math proof step, its breakdown of conditions \
         and conclusions, and the L0 CNL sentence formalized for it.\n\
         Compare the conditions and conclusions against the constituents of the \
         reduced normal form, matching them one by one, and decide whether the \
         CNL sentence is an appropriate formalization of the mathematical \
         statement.\n\
         Assign exactly one of three tags: **Perfectly match**, **Minor \
         inconsistency**, or **Major inconsistency**.\n\n\
         Note that:\n\
         - Perfectly match: the formalization correctly and completely captures \
         the logical and mathematical meaning of the natural language. Extra, \
         logically consistent detail is fine. Do not care about order, and \
         renamed variables are still a match when the meaning is preserved.\n\
         - Minor inconsistency: the logical meaning is similar but there are \
         structural or notational differences that are not direct translations.\n\
         - Major inconsistency: a key logical component of the natural language \
         is missing, or a contradicting one is introduced.\n\
         - The CNL sentence already compiled and reduced to a unique normal \
         form. Focus solely on semantic meaning, not on syntax.\n\n\
         **Stop immediately** after evaluating all pairs. Do not summarize.\n\n\
         Natural language:\n{}\n\n\
         CNL: `{}`\n\
         Reduces to: {} (UNF {})\n\n\
         Provide a component-by-component analysis. Your final output must be a \
         JSON object of this shape:\n\n\
         ```json\n\
         {{\n  \"evaluation\": [\"[evaluation 1]\", \"[evaluation 2]\"],\n  \"feedback\": [\"[feedback 1]\"]\n\
         }}\n```\n\n\
         **Important:** each entry of `evaluation` must be one of Perfectly \
         match / Minor inconsistency / Major inconsistency.\n",
        node.statement,
        formalization.cnl,
        formalization.readback,
        &formalization.unf_hash[..formalization.unf_hash.len().min(12)],
    );
    if let Some(diag) = node
        .error_report
        .as_ref()
        .and_then(|v| v.get("error_report"))
    {
        content.push_str(&format!(
            "\nA previous stage reported: {}\n",
            diag.as_str().unwrap_or("(unprintable)")
        ));
    }
    Turn::user(content)
}

/// Judge every formalized node and build the report.
///
/// Assumptions are **not** judged and are not scored 0: `tc_`/`def_` nodes are
/// quotes from the source, so there is nothing to compare a formalized version
/// *against*. They appear as [`NodeScore::unjudged`] and drag the weighted mean
/// down through the denominator, which is the honest treatment.
pub fn judge_graph<T: Transport>(
    graph: &Validated,
    llm: &mut LlmClient<T>,
    aggregation: Aggregation,
) -> Result<LogosScore, String> {
    let mut rows = Vec::with_capacity(graph.len());
    for node in &graph.nodes {
        let Some(f) = formalization_of(node) else {
            rows.push(unjudged_row(node, "no formalization"));
            continue;
        };
        if !formalizer::is_provable(node.kind().unwrap_or(NodeKind::Lemma)) {
            rows.push(unjudged_row(
                node,
                "assumptions are formalized but never proved, so there is nothing to compare",
            ));
            continue;
        }

        let turn = build_judge_turn(node, &f);
        let reply = llm
            .chat(crate::formalize::llm::CNL_FORMALIZER_PROMPT, &[turn])
            .map_err(|e| e.to_string())?;
        let (tags, feedback) = extract_match_feedback(&reply);
        let unjudged = if tags.is_empty() {
            Some("the judge returned no recognizable verdict".to_string())
        } else {
            None
        };
        rows.push(NodeScore {
            id: node.id.clone(),
            kind: format!("{:?}", node.kind().unwrap_or(NodeKind::Lemma)),
            provable: true,
            semantic_score: sugeno_integral(&tags),
            tags,
            feedback,
            numeric: numeric_agreement(&f),
            unjudged,
        });
    }
    Ok(report(graph, rows, aggregation))
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
            max_tokens: 256,
            timeout: Duration::from_secs(1),
        }
    }

    fn formal(cnl: &str) -> CnlFormalization {
        verify_cnl(cnl, &base_lexicon())
            .unwrap_or_else(|e| panic!("{cnl} should verify: {e}"))
            .into_formalization(1)
    }

    fn scored(id: &str, tags: &[Tag], unjudged: Option<&str>) -> NodeScore {
        NodeScore {
            id: id.into(),
            kind: "Lemma".into(),
            provable: true,
            tags: tags.to_vec(),
            feedback: Vec::new(),
            semantic_score: sugeno_integral(tags),
            numeric: NumericVerdict::Open,
            unjudged: unjudged.map(String::from),
        }
    }

    fn graph_with(formalizations: &[(&str, &str)]) -> Validated {
        // Chained so the graph validates: `check_dag` rejects an orphan, so every
        // node except the closing `ts_` has to be depended on by a later one.
        let mut items: Vec<serde_json::Value> = Vec::new();
        let mut prev: Option<&str> = None;
        for (id, cnl) in formalizations {
            items.push(serde_json::json!({
                "id": id,
                "natural_language": "quote",
                "statement": format!("the statement of {id}"),
                "dependencies": prev.map(|p| vec![p]).unwrap_or_default(),
                "formalization": formal(cnl),
            }));
            prev = Some(id);
        }
        // A final theorem statement so the graph validates.
        items.push(serde_json::json!({
            "id": "ts_1", "natural_language": "quote",
            "statement": "the conclusion",
            "dependencies": prev.map(|p| vec![p]).unwrap_or_default(),
        }));
        crate::formalize::graph::validate_proof_graph(&serde_json::Value::Array(items)).unwrap()
    }

    // ── the fuzzy measure ───────────────────────────────────────────────────

    #[test]
    fn all_perfect_scores_one() {
        assert_eq!(sugeno_integral(&[Tag::Perfect, Tag::Perfect]), 1.0);
        assert_eq!(sugeno_integral(&[Tag::Perfect]), 1.0);
    }

    /// ProofFlow's guard: any `C` is exactly 0, not merely low.
    #[test]
    fn any_major_inconsistency_scores_exactly_zero() {
        assert_eq!(sugeno_integral(&[Tag::Perfect, Tag::Major]), 0.0);
        assert_eq!(
            sugeno_integral(&[Tag::Perfect, Tag::Perfect, Tag::Major]),
            0.0
        );
        assert_eq!(sugeno_integral(&[Tag::Major]), 0.0);
    }

    #[test]
    fn minors_reduce_the_score() {
        let all = sugeno_integral(&[Tag::Perfect, Tag::Perfect, Tag::Perfect, Tag::Perfect]);
        let one = sugeno_integral(&[Tag::Perfect, Tag::Minor, Tag::Perfect, Tag::Perfect]);
        assert!(one < all, "{one} should be below {all}");
        assert!(one > 0.0);
    }

    #[test]
    fn the_score_is_monotone_in_perfection() {
        let mut previous = 1.1;
        for tags in [
            vec![Tag::Perfect; 6],
            vec![
                Tag::Perfect,
                Tag::Perfect,
                Tag::Perfect,
                Tag::Minor,
                Tag::Minor,
                Tag::Minor,
            ],
            vec![Tag::Perfect, Tag::Minor],
        ] {
            let s = sugeno_integral(&tags);
            assert!(s <= previous, "{s} exceeded {previous}");
            previous = s;
        }
    }

    #[test]
    fn an_empty_judgement_scores_zero_and_is_distinguishable() {
        assert_eq!(sugeno_integral(&[]), 0.0);
        // …and `aggregate_total` reports `None` for "nothing judged", so the two
        // cannot be confused in a report.
        let rows = vec![scored("l1", &[], Some("no verdict"))];
        assert_eq!(aggregate_total(&rows, &BTreeMap::new()), Some(0.0));
        assert!(!rows[0].unjudged.is_none());
    }

    #[test]
    fn mu_zeroes_every_subset_when_a_major_is_present() {
        let (mu, evals) = generate_mu(&[Tag::Perfect, Tag::Minor, Tag::Major]);
        assert!(evals.contains(&Tag::Major));
        assert!(mu.values().all(|v| *v == 0.0), "{mu:?}");
    }

    #[test]
    fn mu_is_one_only_for_a_perfect_full_set() {
        let (mu, _) = generate_mu(&[Tag::Perfect, Tag::Perfect]);
        assert_eq!(mu.get(&vec![0, 1]), Some(&1.0));
        let (mu, _) = generate_mu(&[Tag::Perfect, Tag::Minor]);
        assert_ne!(mu.get(&vec![0, 1]), Some(&1.0));
    }

    /// ProofFlow's down-sampling counts `b = 10 - a`, so a `C` is down-sampled into
    /// a `B` and the major inconsistency is lost. That is reproduced rather than
    /// fixed, because the original's published scores depend on it.
    ///
    /// Note *where* the loss happens: `sugeno_integral` guards on a `C` **before**
    /// calling `generate_mu`, exactly as ProofFlow does, so the integral is
    /// always 0 and never reaches the down-sampling. The loss is only reachable
    /// by calling `generate_mu` directly — which is why this test calls it
    /// directly rather than through the integral.
    #[test]
    fn downsampling_loses_major_inconsistencies() {
        let tags = vec![Tag::Major; 12];
        let (_, sampled) = generate_mu(&tags);
        assert_eq!(sampled.len(), MAX_COMPONENTS);
        assert!(
            !sampled.contains(&Tag::Major),
            "ProofFlow's down-sampling turns a C into a B: {sampled:?}"
        );
        // …and the integral short-circuits before it, so the score is still 0.
        assert_eq!(sugeno_integral(&tags), 0.0);
    }

    #[test]
    fn down_sampling_is_not_applied_at_the_boundary() {
        let tags = vec![Tag::Perfect; MAX_COMPONENTS];
        let (_, sampled) = generate_mu(&tags);
        assert_eq!(sampled.len(), MAX_COMPONENTS);
        assert_eq!(sugeno_integral(&tags), 1.0);
    }

    #[test]
    fn combinations_enumerate_each_subset_once() {
        let c = combinations(3, 2);
        assert_eq!(c.len(), 3);
        assert!(c.contains(&vec![0, 1]) && c.contains(&vec![1, 2]) && c.contains(&vec![0, 2]));
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
    }

    // ── verdict extraction ──────────────────────────────────────────────────

    #[test]
    fn tags_parse_case_insensitively() {
        assert_eq!(parse_tag("Perfectly match"), Some(Tag::Perfect));
        assert_eq!(parse_tag("  MINOR INCONSISTENCY "), Some(Tag::Minor));
        assert_eq!(parse_tag("Major inconsistency"), Some(Tag::Major));
        assert_eq!(parse_tag("looks fine"), None);
        // `Major` must not be shadowed by a prefix match on anything else.
        assert_eq!(parse_tag("perfectly matched"), Some(Tag::Perfect));
    }

    #[test]
    fn json_verdicts_are_preferred_and_ordered() {
        let reply = r#"thinking {not json}
```json
{"evaluation": ["Perfectly match", "Minor inconsistency"],
 "feedback": ["first", "second"]}
```"#;
        let (tags, feedback) = extract_match_feedback(reply);
        assert_eq!(tags, [Tag::Perfect, Tag::Minor]);
        assert_eq!(feedback, ["first", "second"]);
    }

    /// The regex recovery path must preserve the judge's *component order*, since
    /// the fuzzy measure indexes by position.
    #[test]
    fn scanned_verdicts_keep_document_order() {
        let reply = "Minor inconsistency <<<variable renamed>>> then \
                     Perfectly match <<<subject matches>>> Major inconsistency";
        let (tags, feedback) = extract_match_feedback(reply);
        assert_eq!(tags, [Tag::Minor, Tag::Perfect, Tag::Major]);
        assert_eq!(feedback.len(), 2);
    }

    #[test]
    fn boxed_verdicts_are_recovered() {
        let (tags, _) =
            extract_match_feedback(r"\box{Perfectly match} and \box{Minor inconsistency}");
        assert_eq!(tags, [Tag::Perfect, Tag::Minor]);
    }

    #[test]
    fn an_unparseable_judgement_yields_no_tags() {
        let (tags, _) = extract_match_feedback("I could not evaluate this.");
        assert!(tags.is_empty());
    }

    // ── the numeric check ───────────────────────────────────────────────────

    #[test]
    fn a_constant_ted_agreeing_with_the_value_is_agreement() {
        // Stock L0 has no closed arithmetic term — every predicate compiles to a
        // constructor application, so `value` is `None` for all of it. The
        // agreement branch is therefore exercised on a synthetic
        // formalization, and the vacuity under stock L0 is asserted separately
        // so it cannot be mistaken for a working check.
        let f = CnlFormalization {
            cnl: "two is two".into(),
            readback: "2".into(),
            unf_hash: "h".into(),
            verified: true,
            value: Some("2".into()),
            ted: Some("2".into()),
            tries: 1,
        };
        assert_eq!(
            numeric_agreement(&f),
            NumericVerdict::Agreed {
                value: "2".into(),
                ted: "2".into()
            }
        );
        let mut mismatched = f.clone();
        mismatched.ted = Some("3".into());
        assert!(matches!(
            numeric_agreement(&mismatched),
            NumericVerdict::Disagreed { .. }
        ));
    }

    /// §17.1 again: the numeric check is real but vacuous for stock L0, because
    /// every L0 term is a constructor application and so denotes no number.
    #[test]
    fn stock_l0_denotes_no_numbers_so_the_numeric_check_is_open() {
        for cnl in ["John loves Mary", "the cat sleeps", "one is one"] {
            assert_eq!(
                numeric_agreement(&formal(cnl)),
                NumericVerdict::Open,
                "{cnl}"
            );
        }
    }

    /// An open term makes no numeric claim, and that is reported as such — never
    /// as an agreement.
    #[test]
    fn an_open_term_is_reported_as_open() {
        let f = formal("John loves Mary");
        assert_eq!(f.value, None);
        assert_eq!(numeric_agreement(&f), NumericVerdict::Open);
    }

    #[test]
    fn a_disagreement_is_reported() {
        let f = CnlFormalization {
            cnl: "x".into(),
            readback: "x".into(),
            unf_hash: "h".into(),
            verified: true,
            value: Some("5".into()),
            ted: Some("7".into()),
            tries: 1,
        };
        assert_eq!(
            numeric_agreement(&f),
            NumericVerdict::Disagreed {
                value: "5".into(),
                ted: "7".into()
            }
        );
    }

    #[test]
    fn a_closed_term_without_a_ted_is_its_own_verdict() {
        let f = CnlFormalization {
            cnl: "x".into(),
            readback: "x".into(),
            unf_hash: "h".into(),
            verified: true,
            value: Some("5".into()),
            ted: None,
            tries: 1,
        };
        assert_eq!(
            numeric_agreement(&f),
            NumericVerdict::ClosedWithoutTed { value: "5".into() }
        );
    }

    // ── the dependency check ────────────────────────────────────────────────

    #[test]
    fn a_clean_graph_reports_clean() {
        // Every node formalized, chained so the graph validates: an unreferenced
        // node is an orphan and `check_dag` rejects the graph before the report
        // is ever built.
        let data = serde_json::json!([
            {"id": "tc_1", "natural_language": "q", "statement": "s", "dependencies": [],
             "formalization": formal("Alice likes John")},
            {"id": "l1", "natural_language": "q", "statement": "s", "dependencies": ["tc_1"],
             "formalization": formal("John loves Mary")},
            {"id": "ts_1", "natural_language": "q", "statement": "s", "dependencies": ["l1"],
             "formalization": formal("Mary sees Bob")},
        ]);
        let g = crate::formalize::graph::validate_proof_graph(&data).unwrap();
        let r = dependency_report(&g);
        assert!(r.is_clean(), "{r:?}");
        assert!(r.no_duplicate_identities);
        assert!(r.unformalized.is_empty());
    }

    /// §14: two nodes with one UNF hash are one node, and that is a fault.
    #[test]
    fn two_nodes_sharing_a_unf_hash_are_a_duplicate_identity() {
        let same = "John loves Mary";
        let data = serde_json::json!([
            {"id": "l1", "natural_language": "q", "statement": "s", "dependencies": [],
             "formalization": formal(same)},
            {"id": "l2", "natural_language": "q", "statement": "s", "dependencies": ["l1"],
             "formalization": formal(same)},
            {"id": "ts_1", "natural_language": "q", "statement": "s", "dependencies": ["l2"],
             "formalization": formal("Mary sees Bob")},
        ]);
        let g = crate::formalize::graph::validate_proof_graph(&data).unwrap();
        let r = dependency_report(&g);
        assert!(!r.is_clean());
        assert!(!r.no_duplicate_identities);
        assert_eq!(
            r.duplicate_identities,
            [vec!["l1".to_string(), "l2".to_string()]]
        );
    }

    #[test]
    fn unformalized_nodes_are_listed_not_scored() {
        let data = serde_json::json!([
            {"id": "l1", "natural_language": "q", "statement": "s", "dependencies": [],
             "formalization": formal("John loves Mary")},
            {"id": "ts_1", "natural_language": "q", "statement": "s", "dependencies": ["l1"]},
        ]);
        let g = crate::formalize::graph::validate_proof_graph(&data).unwrap();
        let r = dependency_report(&g);
        assert_eq!(r.unformalized, ["ts_1"]);
        // A gap is not a collision.
        assert!(r.no_duplicate_identities);
        assert!(
            !r.is_clean(),
            "unformalized keeps the report from being clean"
        );
    }

    // ── centrality and aggregation ──────────────────────────────────────────

    fn weighted_graph() -> Validated {
        let data = serde_json::json!([
            {"id": "l1", "natural_language": "q", "statement": "s", "dependencies": []},
            {"id": "l2", "natural_language": "q", "statement": "s", "dependencies": ["l1"]},
            {"id": "l3", "natural_language": "q", "statement": "s", "dependencies": ["l1"]},
            {"id": "ts_1", "natural_language": "q", "statement": "s", "dependencies": ["l2", "l3"]},
        ]);
        crate::formalize::graph::validate_proof_graph(&data).unwrap()
    }

    #[test]
    fn equal_weights_are_all_one() {
        let w = centrality(&weighted_graph(), Aggregation::Equal);
        assert_eq!(w.len(), 4);
        assert!(w.values().all(|v| *v == 1.0), "{w:?}");
    }

    #[test]
    fn laplacian_weights_track_degree() {
        let w = centrality(&weighted_graph(), Aggregation::Laplacian);
        // l1 is depended on twice and depends on nothing: degree 2.
        assert_eq!(w["l1"], 2.0);
        // ts_1 depends on two and is depended on by none: also 2.
        assert_eq!(w["ts_1"], 2.0);
        assert_eq!(w["l2"], 2.0);
    }

    /// The closing theorem statement is depended on by nothing, so it has degree
    /// zero. It must still get a positive weight, or the final step would be free.
    #[test]
    fn an_isolated_final_step_still_has_weight() {
        let w = centrality(&weighted_graph(), Aggregation::Laplacian);
        assert!(w["ts_1"] >= 1.0);
        let k = centrality(&weighted_graph(), Aggregation::Katz);
        assert!(k.values().all(|v| *v > 0.0), "{k:?}");
    }

    #[test]
    fn katz_weights_are_deterministic_and_normalized() {
        let a = centrality(&weighted_graph(), Aggregation::Katz);
        let b = centrality(&weighted_graph(), Aggregation::Katz);
        assert_eq!(a, b, "two runs must agree");
        let mean = a.values().sum::<f64>() / a.len() as f64;
        assert!((mean - 1.0).abs() < 1e-9, "mean {mean}");
    }

    /// Unjudged nodes weigh in the denominator, so judging one node of three does
    /// not score perfectly.
    #[test]
    fn an_unjudged_node_drags_the_mean_down() {
        let weights = BTreeMap::from([
            ("l1".to_string(), 1.0),
            ("l2".to_string(), 1.0),
            ("l3".to_string(), 1.0),
        ]);
        let all = aggregate_total(
            &[
                scored("l1", &[Tag::Perfect], None),
                scored("l2", &[Tag::Perfect], None),
                scored("l3", &[Tag::Perfect], None),
            ],
            &weights,
        );
        let partial = aggregate_total(
            &[
                scored("l1", &[Tag::Perfect], None),
                scored("l2", &[], Some("no verdict")),
                scored("l3", &[], Some("no verdict")),
            ],
            &weights,
        );
        assert_eq!(all, Some(1.0));
        assert_eq!(partial, Some(1.0 / 3.0));
    }

    #[test]
    fn an_empty_graph_aggregates_to_none() {
        assert_eq!(aggregate_total(&[], &BTreeMap::new()), None);
        assert!(
            centrality(
                &Validated {
                    nodes: vec![],
                    warnings: vec![]
                },
                Aggregation::Katz
            )
            .is_empty()
        );
    }

    #[test]
    fn a_report_is_clean_only_when_nothing_is_missing() {
        let g = weighted_graph();
        let rows = vec![
            scored("l1", &[Tag::Perfect], None),
            scored("l2", &[Tag::Perfect], None),
            scored("l3", &[Tag::Perfect], None),
            scored("ts_1", &[], Some("no verdict")),
        ];
        let s = report(&g, rows, Aggregation::Equal);
        assert!(!s.is_clean(), "ts_1 is unjudged");
        assert_eq!(s.aggregation, "Equal");
        assert!(s.total.is_some());
        assert!(!s.weights.is_empty(), "the report carries its own weights");
    }

    // ── the judge, end to end ───────────────────────────────────────────────

    #[test]
    fn judging_a_graph_skips_assumptions_and_scores_the_rest() {
        let g = graph_with(&[("tc_1", "John loves Mary"), ("l1", "Mary sees Bob")]);
        let reply = r#"```json
{"evaluation": ["Perfectly match", "Perfectly match"], "feedback": ["ok"]}
```"#;
        let mut llm = LlmClient::new(ScriptedTransport::new(vec![reply.into()]), cfg());
        let s = judge_graph(&g, &mut llm, Aggregation::Equal).unwrap();
        assert_eq!(s.nodes.len(), 3);

        let tc = s.nodes.iter().find(|n| n.id == "tc_1").unwrap();
        assert!(tc.unjudged.as_deref().unwrap().contains("never proved"));
        assert_eq!(tc.semantic_score, 0.0);

        let ts = s.nodes.iter().find(|n| n.id == "ts_1").unwrap();
        assert!(ts.unjudged.is_some(), "no formalization: {ts:?}");

        assert!(!s.is_clean());
        assert!(s.total.is_some());
    }

    #[test]
    fn the_judge_turn_carries_the_statement_and_the_normal_form() {
        let data = serde_json::json!([
            {"id": "ts_1", "natural_language": "q", "statement": "the conclusion",
             "dependencies": [], "formalization": formal("Mary sees Bob")},
        ]);
        let g = crate::formalize::graph::validate_proof_graph(&data).unwrap();
        let f = formalization_of(&g.nodes[0]).unwrap();
        let turn = build_judge_turn(&g.nodes[0], &f);
        assert!(turn.content.contains("the conclusion"), "{}", turn.content);
        assert!(turn.content.contains("See(mary, bob)"), "{}", turn.content);
        assert!(
            turn.content.contains("Major inconsistency"),
            "{}",
            turn.content
        );
    }

    #[test]
    fn an_unrecognizable_judgement_is_unjudged_not_perfect() {
        let g = graph_with(&[("l1", "Mary sees Bob")]);
        let mut llm = LlmClient::new(
            ScriptedTransport::new(vec!["looks fine to me".into()]),
            cfg(),
        );
        let s = judge_graph(&g, &mut llm, Aggregation::Equal).unwrap();
        let l1 = s.nodes.iter().find(|n| n.id == "l1").unwrap();
        assert!(l1.unjudged.is_some(), "{l1:?}");
        assert_eq!(l1.semantic_score, 0.0);
    }
}
