//! P9 — `RecomputeIndex`: a LEANN-shaped vector index that stores no vectors.
//!
//! # The idea, and where it comes from
//!
//! LEANN (sibling checkout `../LEANN/`, Berkeley Sky Computing Lab) is a
//! vector index whose headline is storage: 97% less than a flat vector
//! store, because at rest it keeps a **pruned graph + the raw text** and
//! *recomputes* the embeddings of the nodes it actually visits during a
//! query ([LEANN paper](https://arxiv.org/abs/2506.08276); its README's
//! "Architecture" section: graph-based selective recomputation,
//! high-degree-preserving pruning, dynamic batching, two-level search).
//!
//! This module ports that shape, not that code — LEANN is Python over
//! faiss/hnswlib, this is a ~400-line Rust type for corpora a formalization
//! pipeline actually has (hundreds to tens of thousands of lemmas and
//! declarations, not 60M documents). What is kept:
//!
//! - **no vectors at rest.** [`RecomputeIndex`] holds texts, a CSR-pruned
//!   adjacency, one entry point, the model id and the dimension. Search
//!   re-embeds the visited path. Storage is O(text + |pruned edges|),
//!   independent of dimension — the whole point.
//! - **pruning that keeps hubs.** Pass 1 keeps each node's nearest
//!   neighbours; pass 2 guarantees every node can reach a hub (top-decile
//!   in-degree) so that dropping an edge never strands a cluster. A
//!   simplified statement of LEANN's *high-degree-preserving pruning*, and
//!   documented as simplified: the parity target is recall against brute
//!   force, not the paper's exact edge set.
//! - **batched recompute.** One hop scores its whole frontier in a single
//!   `embed_batch` call — LEANN's *dynamic batching*, which is also what
//!   makes an HTTP embedder viable (one request per hop, not one per node).
//!
//! # The cost model, stated honestly
//!
//! A stored-vector index pays O(n·d) storage and answers a query with a
//! cheap dot product against precomputed vectors. This index pays
//! **compute** instead: `recompute_budget` embeddings per query (plus the
//! build, which is a full n·d embed and an exact O(n²) kNN — fine at this
//! corpus scale, and a lie at LEANN's). The `SearchStats` a search returns
//! reports exactly what was recomputed, so the trade is measurable rather
//! than asserted.

use crate::formalize::embed::{l2_normalize, Embedder, EmbedError, Embedding};
use serde::{Deserialize, Serialize};

/// Why a query or build could not run against an index.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// The embedder itself failed (endpoint down, malformed reply, …).
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// The query embedder cannot be used with this index. Both halves are
    /// checked: a different dimension breaks the math, and a *different
    /// model* breaks the space — vectors from two models are not comparable
    /// even at the same dimension, and silently ranking across them is the
    /// kind of bug that looks fine on eyeball tests.
    #[error(
        "index was built with `{stored_model}` at {stored_dims}d, \
         queried with `{got_model}` at {got_dims}d — same model and \
         dimension are required (rebuild the index instead)"
    )]
    Mismatch {
        stored_model: String,
        stored_dims: usize,
        got_model: String,
        got_dims: usize,
    },
}

/// Build and search knobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexParams {
    /// Candidate list size per node before pruning (LEANN builds a wider
    /// graph than it stores, then prunes).
    pub build_neighbors: usize,
    /// Stored adjacency cap per node (pass 1).
    pub max_degree: usize,
    /// Vectors embedded per hop (LEANN's dynamic batching). The last hop of
    /// a query may take a smaller batch.
    pub batch: usize,
    /// Hard cap on vectors embedded per query, *including* the query itself.
    pub recompute_budget: usize,
    /// Hub slots per node in pass 2: how many of `max_degree` edges may be
    /// reserved for reachability to a hub rather than pure proximity.
    pub hub_reserve: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        IndexParams {
            build_neighbors: 16,
            max_degree: 8,
            batch: 32,
            recompute_budget: 512,
            hub_reserve: 3,
        }
    }
}

/// A scored result. `idx` indexes into the texts the index was built from —
/// the caller owns the side table (lemma hashes, declaration records).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub idx: usize,
    /// Cosine similarity in [-1, 1].
    pub score: f32,
}

/// What a query actually recomputed. The storage claim's other half:
/// `recomputed <= recompute_budget`, usually far below the corpus size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Vectors embedded for this query (query vector included).
    pub recomputed: usize,
    /// Frontiers scored, i.e. hops that produced vectors.
    pub hops: usize,
    /// `embed_batch` calls made — 1 per hop, so the RPC count too.
    pub batches: usize,
}

/// A LEANN-shaped index: **texts + pruned graph, never vectors**.
///
/// Serde-serializable: what persists is the same thing that rests in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecomputeIndex {
    texts: Vec<String>,
    /// CSR adjacency: `indptr[i]..indptr[i+1]` into `nbrs` are node `i`'s
    /// stored neighbours. CSR (LEANN stores its graph the same way) keeps
    /// the per-node vec overhead out of the picture.
    indptr: Vec<u32>,
    nbrs: Vec<u32>,
    /// Where a search starts: the highest in-degree node at build time.
    entry: u32,
    /// `Embedder::id()` at build time; queries must match (see
    /// [`IndexError::Mismatch`]).
    model_id: String,
    /// `Embedder::dims()` at build time.
    dims: usize,
    params: IndexParams,
}

impl RecomputeIndex {
    /// Build from raw texts: embed (batched), exact kNN, prune, drop the
    /// vectors. Empty input yields a valid empty index that answers no
    /// hits rather than an error — a store starts empty.
    pub fn build(
        embedder: &dyn Embedder,
        texts: Vec<String>,
        params: IndexParams,
    ) -> Result<Self, IndexError> {
        let model_id = embedder.id().to_string();
        let dims = embedder.dims();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = embedder.embed_batch(&refs)?;
        Self::from_vectors(texts, vectors, model_id, dims, params)
    }

    /// As [`build`](Self::build), but embedding in batches of `chunk`
    /// texts instead of one request with the whole corpus.
    ///
    /// What a web corpus needs: one `curl` process and one server forward
    /// pass per batch (LEANN's dynamic batching, at the transport level),
    /// so a first build cannot become a single enormous request an endpoint
    /// refuses — and a failure names the batch it happened in.
    pub fn build_chunked(
        embedder: &dyn Embedder,
        texts: Vec<String>,
        params: IndexParams,
        chunk: usize,
    ) -> Result<Self, IndexError> {
        let model_id = embedder.id().to_string();
        let dims = embedder.dims();
        let mut vectors = Vec::with_capacity(texts.len());
        for window in texts.chunks(chunk.max(1)) {
            let refs: Vec<&str> = window.iter().map(String::as_str).collect();
            vectors.extend(embedder.embed_batch(&refs)?);
        }
        Self::from_vectors(texts, vectors, model_id, dims, params)
    }

    /// The graph half of a build — kNN, pruning, hub pass — from vectors
    /// that are dropped before it returns. Split from the embedding so the
    /// caller chooses the batch shape.
    fn from_vectors(
        texts: Vec<String>,
        mut vectors: Vec<Embedding>,
        model_id: String,
        dims: usize,
        params: IndexParams,
    ) -> Result<Self, IndexError> {
        let n = texts.len();

        let mut index = RecomputeIndex {
            texts,
            indptr: vec![0; n + 1],
            nbrs: Vec::new(),
            entry: 0,
            model_id,
            dims,
            params,
        };
        if n == 0 {
            return Ok(index);
        }

        if vectors.len() != n {
            return Err(EmbedError::Count { want: n, got: vectors.len() }.into());
        }
        for v in &mut vectors {
            l2_normalize(v);
        }

        // Exact kNN, O(n²·d). Honest at pipeline scale; the alternative
        // (an approximate graph build) is LEANN's faiss/hnswlib territory.
        let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut scored: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (dot(&vectors[i], &vectors[j]), j))
                .collect();
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            scored.truncate(index.params.build_neighbors.max(index.params.max_degree));
            candidates.push(scored.into_iter().map(|(_, j)| j).collect());
        }

        // Pass 1: keep each node's nearest neighbours, capped at max_degree.
        let mut adjacency: Vec<Vec<usize>> = candidates
            .iter()
            .map(|c| c.iter().copied().take(index.params.max_degree).collect())
            .collect();

        // Pass 2: hub reachability. Hubs = top-decile in-degree (ties: low
        // index). A node whose neighbourhood contains no hub gets an edge to
        // its closest hub, within its degree budget — so pruning a link
        // never cuts a cluster off from the graph's backbone.
        let hubs = hubs_of(&adjacency, n);
        for i in 0..n {
            if adjacency[i].iter().any(|&j| hubs.contains(&j)) {
                continue;
            }
            if adjacency[i].len() >= index.params.max_degree + index.params.hub_reserve {
                continue;
            }
            let mut best: Option<(f32, usize)> = None;
            for &h in &hubs {
                if h == i {
                    continue;
                }
                let s = dot(&vectors[i], &vectors[h]);
                if best.map(|(bs, _)| s > bs).unwrap_or(true) {
                    best = Some((s, h));
                }
            }
            if let Some((_, h)) = best {
                adjacency[i].push(h);
            }
        }

        index.entry = hubs.first().copied().unwrap_or(0) as u32;
        index.write_adjacency(&adjacency);
        Ok(index)
    }

    /// Append one text (an incremental `insert`): embed the new text, and
    /// recompute the existing vectors — in **one batch** — to find its
    /// neighbours, link, drop. This is the point of storing no vectors:
    /// an update costs compute, not a stored payload that has to be
    /// rewritten everywhere.
    pub fn insert(&mut self, embedder: &dyn Embedder, text: String) -> Result<(), IndexError> {
        self.check_compatible(embedder)?;
        let idx = self.texts.len();
        if idx == 0 {
            self.texts.push(text);
            self.indptr = vec![0, 0];
            self.entry = 0;
            return Ok(());
        }

        let query = embedder.embed_one(&text)?;
        let mut query = query;
        l2_normalize(&mut query);
        let existing = embedder.embed_batch(&self.texts_())?;
        let mut existing: Vec<Vec<f32>> = existing;
        if existing.len() != idx {
            return Err(EmbedError::Count { want: idx, got: existing.len() }.into());
        }
        for v in &mut existing {
            l2_normalize(v);
        }

        let mut scored: Vec<(f32, usize)> = existing
            .iter()
            .enumerate()
            .map(|(j, v)| (dot(&query, v), j))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        scored.truncate(self.params.build_neighbors.max(self.params.max_degree));
        let mut adjacency = self.read_adjacency();
        adjacency.push(Vec::new());
        let nearest: Vec<usize> = scored
            .iter()
            .take(self.params.max_degree)
            .map(|&(_, j)| j)
            .collect();
        adjacency[idx] = nearest;
        for j in adjacency[idx].clone() {
            if !adjacency[j].contains(&idx) {
                adjacency[j].push(idx);
            }
        }
        // Same hub guarantee as at build: link the newcomer to its closest
        // hub when it has none.
        let hubs = hubs_of(&adjacency, idx + 1);
        if !adjacency[idx].iter().any(|j| hubs.contains(j)) {
            if let Some((_, h)) = scored.iter().find(|(_, j)| hubs.contains(j)) {
                adjacency[idx].push(*h);
            }
        }
        self.texts.push(text);
        self.write_adjacency(&adjacency);
        // The entry point is only a heuristic start; recompute it so a
        // store built by inserts is indistinguishable from one built at once.
        self.entry = hubs_of(&adjacency, adjacency.len())
            .first()
            .copied()
            .unwrap_or(0) as u32;
        Ok(())
    }

    /// Query: greedy best-first traversal, one batched recompute per hop,
    /// bounded by `params.recompute_budget`.
    ///
    /// Ordering among *unscored* candidates uses the score of the node that
    /// discovered them (LEANN/HNSW's two-level idea): we have no vector for
    /// an unscored node — that is the storage saving — so the parent's score
    /// is the only admissible estimate.
    pub fn search(
        &self,
        embedder: &dyn Embedder,
        query: &str,
        k: usize,
    ) -> Result<(Vec<Hit>, SearchStats), IndexError> {
        self.check_compatible(embedder)?;
        let mut stats = SearchStats::default();
        if self.texts.is_empty() || k == 0 {
            return Ok((Vec::new(), stats));
        }

        let mut q = embedder.embed_one(query)?;
        l2_normalize(&mut q);
        stats.recomputed = 1;

        let n = self.texts.len();
        let mut discovered = vec![false; n];
        let mut scored: Vec<(usize, f32)> = Vec::new();
        // (parent score, node) for nodes whose vectors are not yet computed.
        let mut frontier: Vec<(f32, usize)> = vec![(f32::INFINITY, self.entry as usize)];
        discovered[self.entry as usize] = true;

        while !frontier.is_empty() && stats.recomputed < self.params.recompute_budget {
            frontier.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            let take = frontier
                .len()
                .min(self.params.batch)
                .min(self.params.recompute_budget - stats.recomputed);
            if take == 0 {
                break;
            }
            let batch: Vec<usize> = frontier.drain(..take).map(|(_, i)| i).collect();
            let texts: Vec<&str> = batch.iter().map(|&i| self.texts[i].as_str()).collect();
            let vectors = embedder.embed_batch(&texts)?;
            if vectors.len() != batch.len() {
                return Err(EmbedError::Count { want: batch.len(), got: vectors.len() }.into());
            }
            stats.recomputed += batch.len();
            stats.hops += 1;
            stats.batches += 1;

            for (i, mut v) in batch.into_iter().zip(vectors) {
                l2_normalize(&mut v);
                let s = dot(&q, &v);
                scored.push((i, s));
                for &j in &self.nbrs_of(i) {
                    if !discovered[j] {
                        discovered[j] = true;
                        frontier.push((s, j));
                    }
                }
            }
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok((
            scored.into_iter().map(|(idx, score)| Hit { idx, score }).collect(),
            stats,
        ))
    }

    /// The texts this index rests on — the raw data LEANN keeps so vectors
    /// can be recomputed rather than stored.
    pub fn texts(&self) -> &[String] {
        &self.texts
    }

    /// Number of stored nodes.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// `model_id@dims`, the identity a caller should report alongside hits.
    pub fn model(&self) -> String {
        format!("{}@{}", self.model_id, self.dims)
    }

    /// Total stored edges — what the CSR holds, for size reports.
    pub fn edge_count(&self) -> usize {
        self.nbrs.len()
    }

    fn check_compatible(&self, embedder: &dyn Embedder) -> Result<(), IndexError> {
        if embedder.id() == self.model_id && embedder.dims() == self.dims {
            Ok(())
        } else {
            Err(IndexError::Mismatch {
                stored_model: self.model_id.clone(),
                stored_dims: self.dims,
                got_model: embedder.id().to_string(),
                got_dims: embedder.dims(),
            })
        }
    }

    fn texts_(&self) -> Vec<&str> {
        self.texts.iter().map(String::as_str).collect()
    }

    fn nbrs_of(&self, i: usize) -> Vec<usize> {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        self.nbrs[start..end].iter().map(|&x| x as usize).collect()
    }

    fn read_adjacency(&self) -> Vec<Vec<usize>> {
        (0..self.texts.len()).map(|i| self.nbrs_of(i)).collect()
    }

    fn write_adjacency(&mut self, adjacency: &[Vec<usize>]) {
        self.indptr = Vec::with_capacity(adjacency.len() + 1);
        self.nbrs = Vec::new();
        self.indptr.push(0);
        for list in adjacency {
            self.nbrs.extend(list.iter().map(|&x| x as u32));
            self.indptr.push(self.nbrs.len() as u32);
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Top-decile in-degree nodes (at least 1, at most 10), ties broken by
/// lowest index — the deterministic choice that keeps builds reproducible.
fn hubs_of(adjacency: &[Vec<usize>], n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut indeg = vec![0u32; n];
    for list in adjacency.iter().take(n) {
        for &j in list {
            if j < n {
                indeg[j] += 1;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| indeg[b].cmp(&indeg[a]).then_with(|| a.cmp(&b)));
    let take = (n.div_ceil(10)).clamp(1, 10);
    order.truncate(take);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::embed::HashingEmbedder;

    /// Six topics, five phrasings each — a corpus small enough to reason
    /// about, large enough for a graph to matter.
    fn corpus() -> Vec<String> {
        let topics = ["zero", "successor", "addition", "multiplication", "parity", "order"];
        let mut out = Vec::new();
        for t in topics {
            for i in 0..5 {
                out.push(format!("theorem about {t} step {i} of the development"));
            }
        }
        out
    }

    fn topic_of(text: &str) -> &str {
        ["zero", "successor", "addition", "multiplication", "parity", "order"]
            .iter()
            .find(|t| text.contains(**t))
            .copied()
            .unwrap()
    }

    #[test]
    fn build_stores_no_vectors_but_still_finds_neighbours() {
        let e = HashingEmbedder::default();
        let idx = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();
        assert_eq!(idx.len(), 30);
        assert!(idx.edge_count() > 0);

        let (hits, stats) = idx.search(&e, "theorem about parity step", 5).unwrap();
        assert!(!hits.is_empty());
        for h in &hits {
            assert_eq!(topic_of(&idx.texts()[h.idx]), "parity", "cluster found");
        }
        assert!(stats.recomputed <= idx.params.recompute_budget);
        assert!(stats.batches >= 1, "one embed_batch per hop");
    }

    #[test]
    fn recall_against_brute_force_is_high() {
        let e = HashingEmbedder::default();
        let texts = corpus();
        let idx = RecomputeIndex::build(&e, texts.clone(), IndexParams::default()).unwrap();

        // Brute force: embed everything, rank by cosine — the oracle.
        let mut hits_ok = 0;
        let mut total = 0;
        for (i, t) in texts.iter().enumerate() {
            let q = e.embed_one(t).unwrap();
            let mut scored: Vec<(f32, usize)> = Vec::new();
            for (j, other) in texts.iter().enumerate() {
                if i == j {
                    continue;
                }
                let v = e.embed_one(other).unwrap();
                let s: f32 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                scored.push((s, j));
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let oracle: Vec<usize> = scored.iter().take(5).map(|&(_, j)| j).collect();

            // k=6 then drop `i`: the query text is itself in the index,
            // so it scores 1.0 and would occupy rank 1 — a self-hit, not a
            // retrieval. The oracle excludes it too.
            let (got6, _) = idx.search(&e, t, 6).unwrap();
            let got: Vec<usize> = got6
                .iter()
                .filter(|h| h.idx != i)
                .map(|h| h.idx)
                .take(5)
                .collect();
            // recall@5: overlap with the oracle's top 5.
            hits_ok += oracle.iter().filter(|j| got.contains(j)).count();
            total += 5;
        }
        let recall = hits_ok as f64 / total as f64;
        assert!(recall >= 0.9, "recall@5 = {recall:.3}, expected >= 0.9");
    }

    #[test]
    fn search_recomputes_a_path_not_the_corpus() {
        let e = HashingEmbedder::default();
        let mut texts = corpus();
        // Grow the corpus: 300 nodes, budget still 512 → but the path is
        // what gets embedded, not the whole store.
        for i in 0..270 {
            texts.push(format!("lemma variant {i} about addition machinery"));
        }
        let params = IndexParams { recompute_budget: 64, ..Default::default() };
        let idx = RecomputeIndex::build(&e, texts, params).unwrap();
        let (_, stats) = idx.search(&e, "theorem about addition step", 10).unwrap();
        assert!(
            stats.recomputed <= 64,
            "budget respected: {} recomputed",
            stats.recomputed
        );
        assert!(
            stats.recomputed < idx.len(),
            "path {} < corpus {}",
            stats.recomputed,
            idx.len()
        );
    }

    #[test]
    fn the_index_is_smaller_than_the_vectors_it_replaces() {
        let e = HashingEmbedder::default();
        let texts = corpus();
        let idx = RecomputeIndex::build(&e, texts.clone(), IndexParams::default()).unwrap();

        let stored = serde_json::to_string(&idx).unwrap().len();
        let vectors: Vec<Vec<f32>> = texts.iter().map(|t| e.embed_one(t).unwrap()).collect();
        let with_vectors = serde_json::to_string(&(&texts, &vectors)).unwrap().len();

        assert!(
            stored * 2 < with_vectors,
            "index {stored}B vs texts+vectors {with_vectors}B — \
             the storage claim must hold for the serialization too"
        );
    }

    #[test]
    fn builds_are_deterministic_and_round_trip() {
        let e = HashingEmbedder::default();
        let a = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();
        let b = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "same inputs, byte-identical index"
        );

        let json = serde_json::to_string(&a).unwrap();
        let back: RecomputeIndex = serde_json::from_str(&json).unwrap();
        let q = "theorem about zero step";
        let (h1, s1) = a.search(&e, q, 5).unwrap();
        let (h2, s2) = back.search(&e, q, 5).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn a_different_model_or_dimension_is_refused_rather_than_ranked() {
        let e = HashingEmbedder::default();
        let idx = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();

        let other_dim = HashingEmbedder::new(128);
        match idx.search(&other_dim, "x", 3) {
            Err(IndexError::Mismatch { stored_dims: 384, got_dims: 128, .. }) => {}
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn insert_keeps_the_index_consistent() {
        let e = HashingEmbedder::default();
        let mut idx = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();
        idx.insert(&e, "theorem about parity from a later insert".to_string()).unwrap();
        assert_eq!(idx.len(), 31);

        // Query with the inserted text itself: finding it at rank 1 proves
        // it was *reached* through the graph (only scored nodes appear),
        // i.e. the incremental edges connect it to the existing component.
        let (hits, _) = idx
            .search(&e, "theorem about parity from a later insert", 5)
            .unwrap();
        assert!(
            hits.first().map(|h| idx.texts()[h.idx].contains("later insert")).unwrap_or(false),
            "the inserted text is reachable and ranks first for its own query"
        );

        // Round-trips still hold after incremental growth.
        let json = serde_json::to_string(&idx).unwrap();
        let back: RecomputeIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 31);
        assert_eq!(back.entry, idx.entry);
    }

    #[test]
    fn an_empty_index_answers_nothing_instead_of_erroring() {
        let e = HashingEmbedder::default();
        let idx = RecomputeIndex::build(&e, Vec::new(), IndexParams::default()).unwrap();
        let (hits, stats) = idx.search(&e, "anything", 5).unwrap();
        assert!(hits.is_empty());
        assert_eq!(stats.recomputed, 0);
    }

    #[test]
    fn two_identical_searches_return_the_same_order() {
        let e = HashingEmbedder::default();
        let idx = RecomputeIndex::build(&e, corpus(), IndexParams::default()).unwrap();
        let (a, _) = idx.search(&e, "theorem about addition step", 7).unwrap();
        let (b, _) = idx.search(&e, "theorem about addition step", 7).unwrap();
        assert_eq!(a, b);
    }
}
