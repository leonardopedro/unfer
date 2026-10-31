# Retrieval plan — LEANN + EmbeddingGemma 2 + XERJ in `logos/src/formalize` and `mm0web`

Status: plan → executed (execution log at the bottom).

## Why

`formalize::memory` already carries the three pieces a retrieval-augmented
pipeline needs — a lemma store, a lexical retriever, and a few-shot block —
and none of them is connected: `retrieve_lexical`'s own doc says it is
lexical *"because there is no embedding model in the pipeline"*, and
`build_user_turn` never receives an example. `mm0web`'s `/search` is
honest substring match over stored declarations and says so.

This plan adds the missing representation engine and the two index/search
shapes studied in this workspace, without breaking either crate's dependency
discipline:

| input | what it contributes | where it lands |
|---|---|---|
| **EmbeddingGemma 2** (Google DeepMind) | the real semantic space: 768-dim shared text/code embedding, Matryoshka truncation (512/256/128), modular sizes, served via any OpenAI-compatible endpoint (llama.cpp/GGUF, Ollama, LiteRT, TEI) | `formalize::embed::HttpEmbedder`; `mm0web`'s `CurlEmbedder` |
| **LEANN** (sibling checkout `../LEANN/`, MIT) | low-storage index: store a pruned graph + raw text, **recompute vectors on demand** at query time with dynamic batching and high-degree-preserving pruning | `formalize::index::RecomputeIndex`, used by both crates |
| **XERJ** (sibling checkout `../xerj/`, Apache-2.0) | hybrid search shape: BM25 + vector fused with **RRF**, an Elasticsearch-compatible `_search` API as the interchange, and an honestly-labelled *lexical feature-hashing* default embedder | `formalize::hybrid` (fusion + optional node arm), `mm0web` search |

The three fit together the way the attached discussion describes: the model
supplies the shared space, the index keeps the storage small by re-embedding
candidates along the traversal instead of storing every vector, and xerj's
hybrid fusion is how a lexical arm and a semantic arm are ranked together —
with xerj's honesty rule kept: **a lexical feature-hashing vector is called
lexical, never semantic**.

## Constraints discovered before planning

1. **`logos` has no network by default.** `ureq` sits behind `--features
   llm-http` because `logos` is a dependency of crates that must not grow a
   TLS stack. New HTTP (embeddings, xerj node) goes behind the same feature.
2. **`mm0web` uses `curl`, not `ureq`** (see `mm0web/src/http.rs` "Why curl"):
   its outbound transport is a `curl` subprocess. The embedding call and the
   xerj call follow that convention; the *trait* comes from `logos`.
3. **No model weights and no xerj node exist in this environment.** So:
   every network path is exercised against a local scripted endpoint (the
   `c7_smoke` pattern — a real `TcpListener`, nothing mocked below the HTTP
   boundary) or skipped when its env var is unset. The offline path must
   still be useful and honest: xerj's own default embedder is lexical
   feature hashing, so `HashingEmbedder` is that — labelled lexical.
4. **Both CRATES stay honest about what a hit means.** No endpoint/page may
   claim "semantic" when the configured embedder is lexical; the JSON says
   which arms answered and with which model.

## Stages

### S1 — `logos::formalize::embed` — the representation engine
- `trait Embedder`: `id()`, `dims()`, `embed_batch(&[&str])`.
- `HashingEmbedder` (always available): 384-dim lexical feature hashing,
  word + char-ngram, L2-normalized — xerj's documented default embedder
  shape, documented as *lexical, not neural*.
- `HttpEmbedder` (feature `llm-http`): `POST {base}/embeddings`
  (OpenAI-compatible — the schema llama.cpp/Ollama/TEI expose for
  EmbeddingGemma 2), optional `dimensions` request field plus client-side
  Matryoshka truncation + renorm (works even when the server ignores
  `dimensions`), API key optional and never defaulted to empty.
- Tests: determinism/norm/truncation for the hasher; a live-socket test for
  the HTTP transport (request shape + parse), keyed to `llm-http`.

### S2 — `logos::formalize::index` — the LEANN-style index
`RecomputeIndex`: at rest it holds **raw texts, a CSR-pruned kNN graph, one
entry point, the model id and the dimension — no vectors**.
- build: embed batch → exact cosine kNN → prune each node's adjacency to
  `max_degree` keeping hub/in-degree edges (LEANN's high-degree-preserving
  idea, simplified and stated as such) → drop vectors;
- search: greedy best-first traversal; each hop re-embeds its unvisited
  frontier **in one batch** (dynamic batching) and truncates per MRL dims;
  bounded by a recompute budget;
- `insert`: embed the new text, batch-recompute every existing vector once
  to find its kNN, link, drop — this is the point of storing no vectors:
  rebuilds cost compute, not storage.
- Tests: (a) recall@k vs a brute-force oracle over a deterministic corpus,
  (b) serialized index is smaller than the vectors it replaces, (c)
  determinism, (d) truncation knob changes dims but not membership much.

### S3 — `logos::formalize::hybrid` — fusion + retrieval wiring
- `rrf_fuse(rank_lists, k=60)` — xerj's hybrid fusion, ported shape.
- `RetrievalMode { Lexical, Hybrid }`; `retrieve_hybrid(store, embedder,
  node, k)` fuses the lexical arm (`retrieve_lexical`) with a semantic arm
  (the `RecomputeIndex` over the store's CNL texts) and returns
  `Retrieved`s whose `because` names the arms and the model id.
- Optional `XerjArm` (feature `llm-http`, env `XERJ_URL`/`XERJ_INDEX`):
  URI-query `GET {base}/{index}/_search?q=…&size=…`, hits fused in too;
  unreachable node = arm absent, never a failed retrieval. A
  `xerj_bulk_export(store)` helper emits the NDJSON `_bulk` body so an
  operator can load the store into a node with one curl.
- **Wiring (the half that was missing):** `FormalizeOptions` gains a
  `few_shot` block; `build_user_turn` appends it; the CLI builds the
  retriever from `--memory-in` + `--embed-url` (env `UNFER_EMBED_URL`),
  mode `--retrieval lexical|hybrid`, and reports the mode in the run
  report. Default stays lexical and offline.
- Tests: fused ranking order, `because` naming, prompt actually containing
  the examples (scripted transport), xerj arm skip/fuse against a local
  scripted node.

### S4 — `mm0web` search: hybrid over stored declarations
- `mm0web::embed::CurlEmbedder` implementing `logos`' `Embedder` over
  `http::curl`; env `MM0WEB_EMBED_URL`, `MM0WEB_EMBED_MODEL` (default
  `embeddinggemma-2`), `MM0WEB_EMBED_DIMS`.
- `search_hits` becomes two arms: substring (unchanged) and semantic (a
  `RecomputeIndex` over every stored declaration's `name + statement`,
  cached per corpus fingerprint and recomputed — never stored — on miss),
  fused with the shared `rrf_fuse`. With no `MM0WEB_EMBED_URL` it is the
  substring search of today, byte-for-byte behaviour preserved.
- `MM0WEB_XERJ_URL`/`MM0WEB_XERJ_INDEX`: a third arm against a running
  node (documented `_bulk` one-liner to load declarations), fused the same
  way; down node = arm reported absent, endpoint never fails.
- JSON keeps `query`/`count`/`results` and adds `arms`, `fusion`,
  `embedder` (name + dims, `null` when lexical-only). The page says which
  arms answered, physlibsearch-style.
- Tests: substring-only regression, hybrid with an injected fake embedder,
  xerj arm merge against a scripted transport, degraded-mode reporting.

### S5 — docs, attribution, gates
- This file; `FORMALIZE.md` entry/preconditions rows; `mm0web/deploy/README`
  env vars; `ATTRIBUTION.md` (root + `unfer/`) rows for LEANN (MIT,
  read firsthand), xerj (Apache-2.0, read firsthand), EmbeddingGemma 2
  (design + wire protocol only, no weights or code vendored); recount
  workspace tests into the README status block; `check-attribution` and
  `check-status` green.

## Non-goals

- No model weights are downloaded or bundled; no LEANN/xerj Python is
  executed by these crates (their *shapes* are ported and cited, their
  *nodes* are optional externals).
- No multimodal indexing (images/audio/video): mm0web's corpus is text and
  the 270M text/code configuration of the discussion is the relevant one.
- No change to verification, publishing, or the graph-as-view rule: search
  ranks, it never judges.

## Execution log

- [x] S1 embedder (trait, lexical hasher, HTTP client, tests)
- [x] S2 `RecomputeIndex` (prune/build/search/insert, recall + storage tests)
- [x] S3 hybrid fusion + few-shot wiring into `build_user_turn` + CLI flags
- [x] S4 mm0web hybrid search (`CurlEmbedder`, cache, xerj arm, tests)
- [x] S5 docs, attribution rows, test recount, gates
