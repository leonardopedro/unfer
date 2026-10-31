//! P9 — hybrid retrieval: RRF fusion of a lexical arm, a semantic arm, and
//! (optionally) a running **xerj** node.
//!
//! # The shape, and whose it is
//!
//! XERJ (sibling checkout `../xerj/`, Apache-2.0) ranks with two legs — BM25
//! and kNN — fused by Reciprocal-Rank Fusion: `score(d) = Σ 1/(k + rank_i(d))`,
//! `k = 60` the Elasticsearch/xerj default. That fusion is the whole of
//! [`rrf_fuse`], ported as a shape rather than a dependency: our arms are a
//! lexical overlap rank, a graph-recomputed semantic rank from
//! [`RecomputeIndex`](crate::formalize::index::RecomputeIndex), and a
//! remote node's own ranking if the operator runs one.
//!
//! Why RRF and not weighted score merging: the arms are *not* comparable —
//! a cosine from EmbeddingGemma 2, a word-overlap ratio, and a BM25 score
//! live on different scales, and a fixed linear blend silently makes one arm
//! dominate once a model changes. Rank fusion needs only orderings, so the
//! arms can disagree about everything except who came first.
//!
//! # The xerj arm
//!
//! [`XerjNode`] queries an operator-run xerj node through its
//! Elasticsearch-compatible URI search (`GET /{index}/_search?q=…`) and
//! reads back `_id`s in rank order. Loading the store into a node is one
//! `curl --data-binary @bulk.ndjson {base}/_bulk`, which
//! [`xerj_bulk_export`] emits. A node that is down makes the arm *absent* —
//! a retrieval never fails because an optional ranker is unreachable.

use crate::formalize::embed::Embedder;
#[cfg(feature = "llm-http")]
use crate::formalize::embed::percent_encode;
use crate::formalize::graph::GraphNode;
use crate::formalize::index::{IndexParams, IndexError, RecomputeIndex};
use crate::formalize::memory::{retrieve_lexical, Lemma, LemmaStore, Retrieved};
use std::collections::HashMap;

/// The fusion constant: ES/xerj's default `rrf.k`. Large enough that ranks
/// 1–5 dominate, small enough that appearing in every arm still beats
/// winning one arm narrowly.
pub const RRF_K: usize = 60;

/// One arm's ranked list fused in: which arm, and 1-based rank within it.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmRank {
    pub arm: String,
    pub rank: usize,
}

/// A fused candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    /// Index into the caller's parallel id table.
    pub idx: usize,
    /// RRF score — only comparable within one fusion.
    pub score: f64,
    /// Every arm that ranked this candidate, in arm order.
    pub arms: Vec<ArmRank>,
}

/// Reciprocal-Rank Fusion over named arms.
///
/// Deterministic: ties break on the candidate id, so two runs over the same
/// arm orders produce byte-identical output — the property a retrieval
/// report is only worth writing down if it holds.
pub fn rrf_fuse(arms: &[(&str, Vec<usize>)], rrf_k: usize) -> Vec<Fused> {
    let mut acc: HashMap<usize, (f64, Vec<ArmRank>)> = HashMap::new();
    for (name, ranked) in arms {
        for (pos, &id) in ranked.iter().enumerate() {
            let entry = acc.entry(id).or_default();
            entry.0 += 1.0 / (rrf_k + pos + 1) as f64;
            entry.1.push(ArmRank { arm: (*name).to_string(), rank: pos + 1 });
        }
    }
    let mut out: Vec<Fused> = acc
        .into_iter()
        .map(|(idx, (score, arms))| Fused { idx, score, arms })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.idx.cmp(&b.idx))
    });
    out
}

/// The stored text a lemma is indexed under: its CNL **and** its original
/// natural-language statement (a provenance's `node_text`).
///
/// Both halves, because queries come in both languages: the pipeline asks
/// with a natural-language statement, the examples are CNL. For a real
/// embedder the two live in one shared space (this is precisely what
/// EmbeddingGemma 2's shared text/code space buys); for the lexical
/// hashing arm the composite keeps the word-overlap signal honest either
/// way. Deterministic: the first provenance after the store's own sort.
pub fn index_text(lemma: &Lemma) -> String {
    let stmt = lemma
        .provenances
        .first()
        .map(|p| p.node_text.as_str())
        .unwrap_or("");
    if stmt.is_empty() {
        lemma.cnl.clone()
    } else {
        format!("{} {}", lemma.cnl, stmt)
    }
}

/// A lemma store seen through a [`RecomputeIndex`]: ids are UNF hashes in a
/// deterministic order (the store's `iter()` is a `HashMap` and *not* one),
/// texts are [`index_text`].
///
/// The index holds no vectors at rest; lemmas added mid-run go in through
/// [`LemmaIndex::insert`], which recomputes and drops, so a pipeline run can
/// grow its store without a growing footprint.
#[derive(Debug, Clone)]
pub struct LemmaIndex {
    index: RecomputeIndex,
    ids: Vec<String>,
}

impl LemmaIndex {
    /// Build from every lemma in the store, sorted by UNF hash.
    pub fn build(
        store: &LemmaStore,
        embedder: &dyn Embedder,
        params: IndexParams,
    ) -> Result<Self, IndexError> {
        let mut lemmas: Vec<&Lemma> = store.iter().collect();
        lemmas.sort_by(|a, b| a.unf_hash.cmp(&b.unf_hash));
        let ids: Vec<String> = lemmas.iter().map(|l| l.unf_hash.clone()).collect();
        let texts: Vec<String> = lemmas.iter().map(|l| index_text(l)).collect();
        Ok(LemmaIndex { index: RecomputeIndex::build(embedder, texts, params)?, ids })
    }

    /// Index one new lemma (idempotent — a shared lemma indexes once).
    pub fn insert(&mut self, embedder: &dyn Embedder, lemma: &Lemma) -> Result<(), IndexError> {
        if self.ids.contains(&lemma.unf_hash) {
            return Ok(());
        }
        self.index.insert(embedder, index_text(lemma))?;
        self.ids.push(lemma.unf_hash.clone());
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// `model@dims` — for reports.
    pub fn model(&self) -> String {
        self.index.model()
    }

    fn position_of(&self, unf_hash: &str) -> Option<usize> {
        self.ids.iter().position(|id| id == unf_hash)
    }
}

/// Retrieve few-shot examples by fusing the arms.
///
/// - **lexical** — [`retrieve_lexical`], unchanged, scored on statement
///   vocabulary shared with the lemma's CNL;
/// - **semantic** — the [`LemmaIndex`] graph, recomputing the visited path
///   through `embedder`;
/// - **xerj** (optional) — the node's own ranking for the same text.
///
/// The result keeps [`Retrieved`]'s shape so `memory::few_shot_block` and
/// the prompt know nothing new; the `because` string names the fusion and
/// the ranks, so a report can say *why* an example was offered.
pub fn retrieve_hybrid(
    store: &LemmaStore,
    index: &LemmaIndex,
    embedder: &dyn Embedder,
    node: &GraphNode,
    k: usize,
    xerj: Option<&XerjNode>,
) -> Vec<Retrieved> {
    if k == 0 {
        return Vec::new();
    }
    let want = k;

    // Lexical arm, in lemma-id space.
    let lexical: Vec<Retrieved> = retrieve_lexical(store, node, want);
    let mut lexical_ranked: Vec<usize> = Vec::with_capacity(lexical.len());
    for r in &lexical {
        if let Some(p) = index.position_of(&r.lemma.unf_hash) {
            lexical_ranked.push(p);
        }
    }

    // Semantic arm: the statement against the index's graph.
    let mut cosine: HashMap<usize, f32> = HashMap::new();
    let semantic_ranked: Vec<usize> = match index.index.search(embedder, &node.statement, want) {
        Ok((hits, _stats)) => hits
            .into_iter()
            .map(|h| {
                cosine.insert(h.idx, h.score);
                h.idx
            })
            .collect(),
        // A semantic arm that cannot run simply is not an arm: the lexical
        // ranking still answers. (Configured-but-down is reported at the
        // call site's report layer, not swallowed into a ranking.)
        Err(_) => Vec::new(),
    };

    // xerj arm: optional, and never fatal.
    let xerj_ranked: Vec<usize> = xerj
        .and_then(|x| x.query(&node.statement, want).ok())
        .map(|ids| ids.iter().filter_map(|id| index.position_of(id)).collect())
        .unwrap_or_default();

    let mut arms: Vec<(&str, Vec<usize>)> = vec![("lexical", lexical_ranked)];
    if !semantic_ranked.is_empty() {
        arms.push(("semantic", semantic_ranked));
    }
    if !xerj_ranked.is_empty() {
        arms.push(("xerj", xerj_ranked));
    }
    // One arm only: fusion over a singleton adds no information — return
    // that arm's order with honest labels rather than dressing it up as
    // a fused score.
    if arms.len() == 1 && arms[0].0 == "lexical" {
        return lexical;
    }

    let fused = rrf_fuse(&arms, RRF_K);
    fused
        .into_iter()
        .take(k)
        .filter_map(|f| {
            let lemma = store.get(&index.ids[f.idx])?;
            let mut because = format!(
                "hybrid rrf(k={RRF_K}): {}",
                f.arms
                    .iter()
                    .map(|a| format!("{} rank {}", a.arm, a.rank))
                    .collect::<Vec<_>>()
                    .join(" · ")
            );
            if let Some(&c) = cosine.get(&f.idx) {
                because.push_str(&format!(", cos {c:.3} via {}", embedder.id()));
            }
            Some(Retrieved { lemma: lemma.clone(), score: f.score, because })
        })
        .collect()
}

// ── the xerj arm ────────────────────────────────────────────────────────────

/// Why a xerj node call failed (it is a *search* service, so these are
/// reported under their own name, not as embedding failures).
#[derive(Debug, thiserror::Error)]
pub enum XerjError {
    #[error("xerj node answered {code}: {body}")]
    Status { code: u16, body: String },
    #[error("xerj node transport: {0}")]
    Transport(String),
    #[error("malformed xerj reply: {0}")]
    Malformed(String),
}

/// A running xerj node, addressed the way any Elasticsearch client would:
/// `{base}/{index}/_search?q=…&size=…` (URI search — xerj's documented
/// compatibility surface).
#[derive(Debug, Clone)]
pub struct XerjNode {
    base: String,
    index: String,
}

impl XerjNode {
    pub fn new(base: impl Into<String>, index: impl Into<String>) -> Self {
        XerjNode { base: base.into(), index: index.into() }
    }

    pub fn index(&self) -> &str {
        &self.index
    }

    /// Ranked `_id`s for `q`. The ids are the UNF hashes the store was
    /// bulk-loaded with, so callers map them straight to lemmas.
    ///
    /// Built without `llm-http` there is no HTTP client in this crate, so
    /// the arm reports itself absent as a transport error — callers treat
    /// that exactly like a node that is down.
    #[cfg(feature = "llm-http")]
    pub fn query(&self, q: &str, size: usize) -> Result<Vec<String>, XerjError> {
        let url = format!(
            "{}/{}/_search?q={}&size={}",
            self.base.trim_end_matches('/'),
            self.index,
            percent_encode(q),
            size
        );
        let response = ureq::Agent::new()
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(code, r) => XerjError::Status {
                    code,
                    body: r.into_string().unwrap_or_default(),
                },
                ureq::Error::Transport(t) => XerjError::Transport(t.to_string()),
            })?;
        let value: serde_json::Value = response
            .into_json()
            .map_err(|e| XerjError::Malformed(e.to_string()))?;
        Self::parse_hits(value)
    }

    #[cfg(not(feature = "llm-http"))]
    pub fn query(&self, _q: &str, _size: usize) -> Result<Vec<String>, XerjError> {
        Err(XerjError::Transport(format!(
            "logos built without `llm-http`: no HTTP client to query {}/{}",
            self.base, self.index
        )))
    }

    /// Parse an Elasticsearch `hits.hits[]._id` reply. Split out so the
    /// reply shape is testable without a socket.
    #[cfg_attr(not(feature = "llm-http"), allow(dead_code))]
    fn parse_hits(value: serde_json::Value) -> Result<Vec<String>, XerjError> {
        let hits = value
            .get("hits")
            .and_then(|h| h.get("hits"))
            .and_then(|h| h.as_array())
            .ok_or_else(|| XerjError::Malformed("no hits.hits".into()))?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            match hit.get("_id").and_then(|i| i.as_str()) {
                Some(id) => out.push(id.to_string()),
                None => return Err(XerjError::Malformed("hit without _id".into())),
            }
        }
        Ok(out)
    }
}

/// The store as an Elasticsearch `_bulk` NDJSON body: one `index` action
/// line per lemma (`_id` = UNF hash) followed by its document. Load with:
///
/// ```text
/// curl -H 'content-type: application/x-ndjson' --data-binary @lemmas.ndjson \
///      http://127.0.0.1:9200/lemmas/_bulk
/// ```
///
/// The document's `text` field is [`index_text`] — the same composite the
/// semantic arm embeds, so the node's BM25 and our vectors rank the same
/// content. Deterministic order (UNF hash), so repeated exports diff clean.
pub fn xerj_bulk_export(store: &LemmaStore) -> String {
    let mut lemmas: Vec<&Lemma> = store.iter().collect();
    lemmas.sort_by(|a, b| a.unf_hash.cmp(&b.unf_hash));
    let mut out = String::new();
    for l in lemmas {
        let doc = serde_json::json!({
            "id": l.unf_hash,
            "cnl": l.cnl,
            "readback": l.readback,
            "text": index_text(l),
        });
        out.push_str(&format!(
            "{{\"index\":{{\"_id\":\"{}\"}}}}\n{}\n",
            l.unf_hash,
            serde_json::to_string(&doc).expect("a flat json object serializes")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::embed::HashingEmbedder;
    use crate::formalize::formalizer::{base_lexicon, verify_cnl};    use crate::formalize::memory::Provenance;

    fn node(id: &str, statement: &str) -> GraphNode {
        GraphNode::new(id, "quote", statement, Vec::<String>::new())
    }

    fn formal(cnl: &str) -> crate::formalize::formalizer::CnlFormalization {
        let r = verify_cnl(cnl, &base_lexicon())
            .unwrap_or_else(|e| panic!("{cnl} should verify: {e}"));
        r.into_formalization(1)
    }

    fn prov(node_id: &str, text: &str) -> Provenance {
        Provenance {
            node_id: node_id.to_string(),
            node_text: text.to_string(),
            model: "test-model".into(),
            prompt_version: "v1".into(),
        }
    }

    /// Four CNL lemmas from the stock lexicon, each with an NL statement.
    fn store_with_examples() -> LemmaStore {
        let mut store = LemmaStore::new();
        for (id, cnl, stmt) in [
            ("l1", "John sees Mary", "someone sees someone"),
            ("l2", "Mary loves John", "someone loves someone"),
            ("l3", "John sees John", "someone sees themselves"),
            ("l4", "Mary sees Mary", "someone sees themself again"),
        ] {
            store
                .insert(&formal(cnl), prov(id, stmt))
                .expect("verified formalization inserts");
        }
        store
    }

    #[test]
    fn rrf_fusion_matches_the_hand_computed_scores() {
        // A at rank 1 in both arms: 1/61 + 1/61. B at rank 2 of one: 1/62.
        let fused = rrf_fuse(&[("lexical", vec![0, 1]), ("semantic", vec![0])], RRF_K);
        assert_eq!(fused[0].idx, 0);
        let expected_a = 1.0 / 61.0 + 1.0 / 61.0;
        assert!((fused[0].score - expected_a).abs() < 1e-12, "{}", fused[0].score);
        assert_eq!(fused[0].arms.len(), 2);
        assert_eq!(fused[1].idx, 1);
        let expected_b = 1.0 / 62.0;
        assert!((fused[1].score - expected_b).abs() < 1e-12);
    }

    #[test]
    fn fusion_is_deterministic_and_total() {
        let a = rrf_fuse(&[("x", vec![3, 1]), ("y", vec![1, 3])], RRF_K);
        let b = rrf_fuse(&[("x", vec![3, 1]), ("y", vec![1, 3])], RRF_K);
        assert_eq!(a, b);
        // Candidates 1 and 3 tie exactly (both rank-1/2 in swapped arms):
        // the id tie-break makes the order total.
        assert_eq!(a[0].idx.min(a[1].idx), 1);
        assert_eq!(a[0].idx.max(a[1].idx), 3);
    }

    #[test]
    fn hybrid_retrieval_returns_fused_examples_with_honest_because() {
        let store = store_with_examples();
        let e = HashingEmbedder::default();
        let index = LemmaIndex::build(&store, &e, IndexParams::default()).unwrap();
        assert_eq!(index.len(), 4);

        let n = node("ts_1", "someone sees someone they love");
        let got = retrieve_hybrid(&store, &index, &e, &n, 3, None);
        assert!(!got.is_empty());
        for r in &got {
            assert!(
                r.because.contains("hybrid rrf") || r.because.contains("lexical"),
                "the reason names the arm: {}",
                r.because
            );
            assert!(store.contains(&r.lemma.unf_hash));
        }
        // Determinism: same call twice, same order.
        let again = retrieve_hybrid(&store, &index, &e, &n, 3, None);
        assert_eq!(got, again);
    }

    #[test]
    fn an_empty_store_retrieves_nothing() {
        let store = LemmaStore::new();
        let e = HashingEmbedder::default();
        let index = LemmaIndex::build(&store, &e, IndexParams::default()).unwrap();
        let got = retrieve_hybrid(&store, &index, &e, &node("l1", "anything at all"), 3, None);
        assert!(got.is_empty());
    }

    #[test]
    fn inserting_a_lemma_mid_run_makes_it_retrievable() {
        let mut store = store_with_examples();
        let e = HashingEmbedder::default();
        let mut index = LemmaIndex::build(&store, &e, IndexParams::default()).unwrap();
        let before = index.len();

        // A lemma the index has never seen, offered the way the CLI does
        // after a node formalizes: store first, then index.
        let fresh = formal("John loves Mary");
        store
            .insert(&fresh, prov("l5", "someone loves someone brand new"))
            .expect("verified formalization inserts");
        let lemma = store.get(&fresh.unf_hash).expect("just inserted");
        index.insert(&e, lemma).unwrap();
        assert_eq!(index.len(), before + 1);

        // Idempotent: a second offer of the same UNF does not double-index.
        index.insert(&e, lemma).unwrap();
        assert_eq!(index.len(), before + 1, "insert is idempotent by UNF");

        let n = node("ts_2", "someone loves someone brand new");
        let got = retrieve_hybrid(&store, &index, &e, &n, 4, None);
        assert!(
            got.iter().any(|r| r.lemma.cnl == "John loves Mary"),
            "the mid-run lemma is retrievable: {:?}",
            got.iter().map(|r| &r.lemma.cnl).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_xerj_reply_shape_is_read_as_ranks() {
        let reply = serde_json::json!({
            "hits": { "hits": [
                { "_id": "unf-aaa" },
                { "_id": "unf-bbb" }
            ] }
        });
        let ids = XerjNode::parse_hits(reply).unwrap();
        assert_eq!(ids, vec!["unf-aaa", "unf-bbb"]);

        let broken = serde_json::json!({ "hits": { "hits": [ { "shard": 1 } ] } });
        assert!(matches!(
            XerjNode::parse_hits(broken),
            Err(XerjError::Malformed(_))
        ));
    }

    #[test]
    fn bulk_export_is_loadable_ndjson_with_stable_ids() {
        let store = store_with_examples();
        let bulk = xerj_bulk_export(&store);
        let lines: Vec<&str> = bulk.lines().collect();
        assert_eq!(lines.len(), 8, "two lines per lemma");
        for pair in lines.chunks(2) {
            let action: serde_json::Value = serde_json::from_str(pair[0]).unwrap();
            let doc: serde_json::Value = serde_json::from_str(pair[1]).unwrap();
            let id = action["index"]["_id"].as_str().unwrap();
            assert_eq!(doc["id"].as_str().unwrap(), id);
            assert!(!doc["text"].as_str().unwrap().is_empty());
        }
        // Deterministic across calls.
        assert_eq!(bulk, xerj_bulk_export(&store));
    }
}
