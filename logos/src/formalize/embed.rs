//! P9 — the representation engine: one `Embedder` trait, two implementations.
//!
//! # What this adds, and what was here before
//!
//! `formalize::memory::retrieve_lexical` says in its own doc that it is
//! lexical *"because there is no embedding model in the pipeline"* — the
//! honest state of affairs before this module. There is one now:
//! **EmbeddingGemma 2** (Google DeepMind, 740M modular — text/code at the
//! 270M slice up), whose vectors any OpenAI-compatible endpoint can produce
//! ([`HttpEmbedder`]), and an offline lexical arm ([`HashingEmbedder`]).
//!
//! # The honesty rule, borrowed from xerj
//!
//! XERJ ships a *lexical feature-hashing* embedder as its default and
//! documents it as such — "hybrid lexical+vector, never claimed to be
//! neural" (`xerj/docs/WHY_XERJ.md`). This module keeps the same rule: a
//! [`HashingEmbedder`] vector is **not** a semantic embedding and nothing
//! that uses it may say "semantic". It exists so that the index and the
//! retrieval mechanics are testable and runnable with no weights, no
//! network, and no pretence — the same reason xerj has one.
//!
//! # Matryoshka truncation
//!
//! EmbeddingGemma 2 supports MRL truncation (768 → 512/256/128). Truncation
//! is done **client-side** by [`mrl_truncate`] — cut, then re-normalize —
//! because the servers that expose `/embeddings` (llama.cpp, Ollama, TEI,
//! LiteRT) do not all honour a `dimensions` request field, and a truncated-
//! but-unrenormalized vector silently changes every cosine.
//!
//! # Dependency discipline
//!
//! [`HttpEmbedder`] sits behind the existing `llm-http` feature, like
//! `formalize::llm::HttpTransport`: the crate's own pipeline needs no
//! network and no TLS. The trait, the hasher, and the math are always
//! available, so `mm0web` — which reaches the network through `curl`, not
//! `ureq` — can implement this trait itself (see `mm0web/src/embed.rs`).

/// A vector, as produced by an [`Embedder`] — the workspace's existing
/// `engram::table::Embedding` alias, re-exported so downstream crates
/// (`mm0web`) name it through this module.
pub use crate::engram::table::Embedding;

/// Why an embedding request failed.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The endpoint answered with a non-2xx status. The body is kept: a 401
    /// and a 500 look identical as `Transport` otherwise, and they are
    /// different afternoons.
    #[error("embedding endpoint answered {code}: {body}")]
    Status { code: u16, body: String },
    /// The request never completed — connection refused, DNS, timeout.
    #[error("embedding transport: {0}")]
    Transport(String),
    /// The reply parsed but was not an `/embeddings` answer.
    #[error("malformed embedding reply: {0}")]
    Malformed(String),
    /// A batch reply with the wrong number of vectors.
    #[error("asked for {want} embeddings, got {got}")]
    Count { want: usize, got: usize },
}

/// The representation engine: text in, unit vector out.
///
/// Object-safe on purpose — `mm0web` stores `Arc<dyn Embedder>` in its
/// `Config`, and the CLI stores `Option<Box<dyn Embedder>>`.
pub trait Embedder: Send + Sync {
    /// The model id. Part of every retrieval `because` string and every
    /// search report: a hit must be traceable to the model that made it.
    fn id(&self) -> &str;

    /// Output dimension **after** truncation — what the index stores as its
    /// `dims` and what distance math runs at.
    fn dims(&self) -> usize;

    /// Embed a batch. Order is significant and must be preserved.
    ///
    /// Batches matter beyond convenience: LEANN's search recomputes a
    /// frontier per hop and its cost model assumes one forward pass per
    /// batch, so an embedder that silently looped per text would make the
    /// index's compute budget a lie.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError>;

    /// Embed one text. Default: a batch of one.
    fn embed_one(&self, text: &str) -> Result<Embedding, EmbedError> {
        let mut v = self.embed_batch(std::slice::from_ref(&text))?;
        v.pop().ok_or_else(|| EmbedError::Count { want: 1, got: 0 })
    }
}

/// Percent-encode for a query string: unreserved characters through, UTF-8
/// bytes of everything else. Hand-rolled because neither crate here takes a
/// URL dependency — the alternative is a whole crate for one GET parameter.
pub fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// L2-normalize in place. Zero vectors are left alone (normalizing them
/// would produce NaN, and a NaN cosine poisons a whole ranking).
pub fn l2_normalize(v: &mut Embedding) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Matryoshka truncation: cut to `dims`, then re-normalize.
///
/// `dims == 0` or `dims >= v.len()` means "no truncation" — still
/// re-normalized, because the caller may be holding a vector from a source
/// that never normalized it, and every cosine below assumes unit length.
///
/// The re-normalize is the whole point: MRL guarantees quality after
/// truncation *for the renormalized vector*; cutting without it shrinks the
/// norm by an amount that depends on the text, which is a per-query bias
/// hiding inside a similarity score.
pub fn mrl_truncate(mut v: Embedding, dims: usize) -> Embedding {
    if dims > 0 && dims < v.len() {
        v.truncate(dims);
    }
    l2_normalize(&mut v);
    v
}

// ── the offline arm ─────────────────────────────────────────────────────────

/// FNV-1a, 64-bit — the same non-cryptographic family xerj's feature
/// hasher uses. Deterministic across runs and platforms, which matters:
/// a retrieval test that fails on Tuesdays is worse than no test.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Lexical feature hashing: words and character trigrams into a fixed-size
/// unit vector.
///
/// **Not semantic.** It measures word and sub-word overlap, exactly like
/// xerj's default embedder, and is named and documented that way. Its jobs:
///
/// 1. let `RecomputeIndex` and the hybrid retriever run — and be tested —
///    with no weights and no network;
/// 2. give the lexical arm a vector form, so an arm that answered with
///    hashing can be reported as `lexical`, not as `semantic`.
///
/// Weights: word +1.0, trigram +0.5 (including the boundary-marked
/// `^w`/`w$` markers, so prefix and suffix hits count), then L2 norm, then
/// optional MRL truncation to `dims`.
#[derive(Debug, Clone)]
pub struct HashingEmbedder {
    dims: usize,
}

impl HashingEmbedder {
    /// `dims`-dimensional lexical hashing (xerj's default is 384).
    pub fn new(dims: usize) -> Self {
        HashingEmbedder { dims: dims.max(1) }
    }

    fn bump(&self, acc: &mut Vec<f32>, key: &[u8], weight: f32) {
        let h = fnv1a(key);
        // The sign trick of feature hashing: one bit steers the contribution
        // to ±, which keeps the expected contribution of a collision at 0
        // instead of a systematic +drift.
        let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
        let idx = ((h >> 1) as usize) % self.dims;
        acc[idx] += sign * weight;
    }

    fn embed_text(&self, text: &str) -> Embedding {
        let mut v = vec![0.0f32; self.dims];
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            if word.is_empty() {
                continue;
            }
            let lower = word.to_lowercase();
            self.bump(&mut v, lower.as_bytes(), 1.0);
            // Boundary-marked trigrams: "^ab", "abc", "bc$", so that
            // sub-word matches carry and unrelated words sharing an inner
            // trigram do not get a free full-word's weight.
            let marked = format!("^{lower}$");
            let chars: Vec<char> = marked.chars().collect();
            for w in chars.windows(3) {
                let mut key = String::with_capacity(3);
                key.extend(w);
                self.bump(&mut v, key.as_bytes(), 0.5);
            }
        }
        l2_normalize(&mut v);
        v
    }
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        // 384: xerj's documented default dimension for its lexical embedder.
        Self::new(384)
    }
}

impl Embedder for HashingEmbedder {
    fn id(&self) -> &str {
        // The id goes into `because` strings and search reports. It says
        // plainly what this is, so a report can never read as neural.
        "lexical-feature-hashing"
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }
}

// ── the semantic arm ────────────────────────────────────────────────────────

/// `POST {base_url}/embeddings` over `ureq` — the OpenAI-compatible schema
/// that llama.cpp (GGUF), Ollama, LiteRT and TEI all expose, and the way
/// EmbeddingGemma 2 is served in practice: weights and quantization stay
/// the operator's problem, the wire shape is stable.
///
/// Behind `llm-http`, exactly like `formalize::llm::HttpTransport`.
#[cfg(feature = "llm-http")]
#[derive(Debug, Clone)]
pub struct HttpEmbedder {
    agent: ureq::Agent,
    base_url: String,
    model: String,
    dims: usize,
    api_key: Option<String>,
    /// Ask the server for `dimensions` too (OpenAI extension, honoured by
    /// some servers). Harmless where it is ignored — the client truncates
    /// either way, so behaviour does not depend on server support.
    send_dimensions: bool,
}

#[cfg(feature = "llm-http")]
impl HttpEmbedder {
    /// A client for one endpoint and model. `dims == 0` keeps the model's
    /// native dimension (768 for EmbeddingGemma 2); 512/256/128 select an
    /// MRL truncation.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dims: usize) -> Self {
        HttpEmbedder {
            agent: ureq::Agent::new(),
            base_url: base_url.into(),
            model: model.into(),
            dims,
            api_key: None,
            send_dimensions: true,
        }
    }

    /// With a bearer token. Never defaulted to an empty string: an
    /// `Authorization: Bearer ` header authenticates as nobody and confuses
    /// the server into rejecting anonymously-permitted endpoints.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// A whole-request timeout, same reasoning as `HttpTransport::with_timeout`.
    pub fn with_timeout(self, timeout: std::time::Duration) -> Self {
        HttpEmbedder {
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
            ..self
        }
    }
}

#[cfg(feature = "llm-http")]
impl Embedder for HttpEmbedder {
    fn id(&self) -> &str {
        &self.model
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let mut body = serde_json::json!({ "model": self.model, "input": texts });
        if self.send_dimensions && self.dims > 0 {
            body["dimensions"] = serde_json::json!(self.dims);
        }

        let mut request = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        let response = request.send_json(&body).map_err(ureq_err)?;
        let value: serde_json::Value = response
            .into_json()
            .map_err(|e| EmbedError::Malformed(e.to_string()))?;

        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| EmbedError::Malformed("no `data` array".into()))?;
        if data.len() != texts.len() {
            return Err(EmbedError::Count { want: texts.len(), got: data.len() });
        }

        // OpenAI's schema carries `index`; entries may arrive out of order,
        // and a permuted batch silently assigns the wrong vector to the
        // wrong text — the worst possible failure, because it still ranks.
        let mut indexed: Vec<(usize, Embedding)> = Vec::with_capacity(data.len());
        for (position, entry) in data.iter().enumerate() {
            let idx = entry.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
            let vec = entry
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| EmbedError::Malformed("entry without `embedding`".into()))?;
            let mut v: Embedding = Vec::with_capacity(vec.len());
            for x in vec {
                v.push(x.as_f64().ok_or_else(|| {
                    EmbedError::Malformed("embedding component is not a number".into())
                })? as f32);
            }
            indexed.push((idx.unwrap_or(position), mrl_truncate(v, self.dims)));
        }
        indexed.sort_by_key(|(i, _)| *i);
        Ok(indexed.into_iter().map(|(_, v)| v).collect())
    }
}

/// Map a `ureq` error onto [`EmbedError`], with the same split
/// `llm.rs::classify` insists on: an answered status carries its body.
#[cfg(feature = "llm-http")]
fn ureq_err(e: ureq::Error) -> EmbedError {
    match e {
        ureq::Error::Status(code, response) => EmbedError::Status {
            code,
            body: response.into_string().unwrap_or_default(),
        },
        ureq::Error::Transport(t) => EmbedError::Transport(t.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_deterministic_and_normalized() {
        let e = HashingEmbedder::default();
        let a = e.embed_one("the successor of zero").unwrap();
        let b = e.embed_one("the successor of zero").unwrap();
        assert_eq!(a, b, "same text, same vector, byte for byte");
        assert_eq!(a.len(), 384);
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit vector, got {norm}");
    }

    #[test]
    fn hashing_distinguishes_texts_and_shares_on_overlap() {
        let e = HashingEmbedder::default();
        let a = e.embed_one("successor of zero").unwrap();
        let b = e.embed_one("successor of one").unwrap();
        let c = e.embed_one("completely unrelated vocabulary here").unwrap();
        assert_ne!(a, b);
        let overlap = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
        let unrelated = a.iter().zip(&c).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            overlap > unrelated,
            "shared words must beat unrelated text ({overlap} vs {unrelated})"
        );
    }

    #[test]
    fn ids_are_honest_about_being_lexical() {
        assert_eq!(HashingEmbedder::default().id(), "lexical-feature-hashing");
    }

    #[test]
    fn truncation_renormalizes_and_is_a_noop_at_zero() {
        let mut v: Embedding = (0..16).map(|i| (i + 1) as f32).collect();
        l2_normalize(&mut v);
        let full_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((full_norm - 1.0).abs() < 1e-6);

        let t = mrl_truncate(v.clone(), 4);
        assert_eq!(t.len(), 4);
        let t_norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((t_norm - 1.0).abs() < 1e-5, "truncated vector re-normalized");

        let keep = mrl_truncate(v.clone(), 0);
        assert_eq!(keep, v, "dims=0 keeps the vector");
    }

    #[test]
    fn zero_vectors_survive_normalization() {
        let mut z = vec![0.0f32; 8];
        l2_normalize(&mut z);
        assert!(z.iter().all(|x| x.is_finite()), "no NaN from a zero vector");
    }

    /// The HTTP arm, against a real socket: request shape, `index`
    /// reordering, and truncation — nothing mocked below the boundary
    /// (the `c7_smoke` pattern).
    #[cfg(feature = "llm-http")]
    mod http {
        use super::super::*;
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::TcpListener;

        fn serve_once(
            reply: &'static str,
        ) -> (std::net::SocketAddr, std::thread::JoinHandle<String>) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut content_length = 0usize;
                let mut captured = format!("{request_line}");
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length: ") {
                        content_length = v.trim().parse().unwrap();
                    }
                    captured.push_str(&line);
                }
                let mut body = vec![0u8; content_length];
                reader.read_exact(&mut body).unwrap();
                captured.push_str(&String::from_utf8_lossy(&body));
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                             content-length: {}\r\nconnection: close\r\n\r\n{}",
                            reply.len(),
                            reply
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                let _ = stream.flush();
                captured
            });
            (addr, handle)
        }

        #[test]
        fn posts_the_openai_shape_and_honors_index_order() {
            // Out-of-order `index` values on purpose: entry 1 is the vector
            // for the *second* input text.
            let reply = r#"{"data":[
                {"index":1,"embedding":[0.0,1.0,0.0,0.0]},
                {"index":0,"embedding":[1.0,0.0,0.0,0.0]}
            ]}"#;
            let (addr, captured) = serve_once(reply);
            let e = HttpEmbedder::new(format!("http://{addr}"), "embeddinggemma-2", 0)
                .with_key("test-key");
            let out = e.embed_batch(&["alpha", "beta"]).unwrap();
            let request = captured.join().unwrap();
            assert!(request.starts_with("POST /embeddings HTTP/1.1"), "{request}");
            assert!(request.contains("alpha") && request.contains("beta"));
            assert!(request.contains("embeddinggemma-2"));
            assert!(request.to_ascii_lowercase().contains("authorization: bearer test-key"));
            assert_eq!(out.len(), 2);
            assert_eq!(out[0], vec![1.0, 0.0, 0.0, 0.0], "index 0 first");
            assert_eq!(out[1], vec![0.0, 1.0, 0.0, 0.0], "index 1 second");
        }

        #[test]
        fn truncates_to_the_configured_mrl_dims() {
            let reply = r#"{"data":[{"index":0,"embedding":[1.0,0.0,0.0,0.0]}]}"#;
            let (addr, captured) = serve_once(reply);
            let e = HttpEmbedder::new(format!("http://{addr}"), "embeddinggemma-2", 2);
            let out = e.embed_one("x").unwrap();
            assert_eq!(out.len(), 2, "MRL truncation applied client-side");
            let request = captured.join().unwrap();
            assert!(request.contains("\"dimensions\":2"), "also asked the server");
            let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "renormalized after truncation");
        }

        #[test]
        fn a_wrong_count_is_an_error_not_a_zip() {
            let reply = r#"{"data":[{"index":0,"embedding":[1.0]}]}"#;
            let (addr, _captured) = serve_once(reply);
            let e = HttpEmbedder::new(format!("http://{addr}"), "m", 0);
            match e.embed_batch(&["a", "b"]) {
                Err(EmbedError::Count { want: 2, got: 1 }) => {}
                other => panic!("expected Count {{2,1}}, got {other:?}"),
            }
        }
    }
}
