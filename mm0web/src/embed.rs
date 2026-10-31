//! P9 — the semantic arm's transport: EmbeddingGemma 2 through `curl`.
//!
//! `logos::formalize::embed::Embedder` is the trait (shared with the
//! formalization pipeline); this is this crate's implementation of it, and
//! it follows the crate's standing rule — outbound HTTP is a `curl`
//! subprocess (`http::curl`), because this workspace builds without `ureq`
//! (see `README.md`, "Why curl"). One `/embeddings` call per batch, the
//! OpenAI schema that llama.cpp/GGUF, Ollama, TEI and LiteRT all expose.
//!
//! Failure is an ordinary outcome here, not an exception: `search` treats a
//! failing embedder as an absent semantic arm and says so (`unavailable` in
//! the JSON, a note on the page). The substring arm always answers.

use crate::http;
use logos::formalize::embed::{mrl_truncate, Embedder, EmbedError};
use serde_json::json;

/// Where and how to embed. `MM0WEB_EMBED_URL` etc. — see `main.rs`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedConfig {
    /// Base URL of an OpenAI-compatible server (no trailing `/embeddings`).
    pub url: String,
    /// The model field — and the id reported with every semantic hit, so a
    /// ranking is always traceable to the model that produced it.
    pub model: String,
    /// Matryoshka truncation (EmbeddingGemma 2: 768/512/256/128); 0 keeps
    /// the native dimension. Applied client-side after the reply, so it
    /// behaves identically whether or not the server honours `dimensions`.
    pub dims: usize,
    /// Optional bearer token. Never defaulted to `Some("")`: an empty
    /// `Authorization` header authenticates as nobody.
    pub key: Option<String>,
}

/// The embedder: one struct, one `Embedder` impl, transport = curl.
#[derive(Debug, Clone)]
pub struct CurlEmbedder {
    cfg: EmbedConfig,
}

impl CurlEmbedder {
    pub fn new(cfg: EmbedConfig) -> Self {
        CurlEmbedder { cfg }
    }

    pub fn config(&self) -> &EmbedConfig {
        &self.cfg
    }
}

impl Embedder for CurlEmbedder {
    fn id(&self) -> &str {
        &self.cfg.model
    }

    fn dims(&self) -> usize {
        self.cfg.dims
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<logos::formalize::embed::Embedding>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.cfg.url.trim_end_matches('/'));
        let mut body = json!({ "model": self.cfg.model, "input": texts });
        if self.cfg.dims > 0 {
            body["dimensions"] = json!(self.cfg.dims);
        }
        let body = serde_json::to_vec(&body)
            .map_err(|e| EmbedError::Malformed(format!("request body: {e}")))?;

        let mut headers: Vec<(String, String)> =
            vec![("Content-Type".into(), "application/json".into())];
        if let Some(key) = &self.cfg.key {
            headers.push(("Authorization".into(), format!("Bearer {key}")));
        }

        let (code, reply) =
            http::curl("POST", &url, &headers, Some(&body), None).map_err(EmbedError::Transport)?;
        if code == 0 {
            // curl's `%{http_code}` is 000/0 when the connection never
            // completed: nothing answered, so reporting "answered 0"
            // would name a status that was never sent.
            return Err(EmbedError::Transport(format!(
                "cannot reach the embedding endpoint {url}"
            )));
        }
        if !(200..300).contains(&code) {
            return Err(EmbedError::Status { code, body: reply });
        }

        let value: serde_json::Value = serde_json::from_str(&reply)
            .map_err(|e| EmbedError::Malformed(format!("not JSON: {e}")))?;
        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| EmbedError::Malformed("no `data` array".into()))?;
        if data.len() != texts.len() {
            return Err(EmbedError::Count { want: texts.len(), got: data.len() });
        }

        // Honour `index` when present: entries may arrive out of order and
        // a permuted batch assigns the wrong vector to the wrong text —
        // a failure that still ranks, which is the worst kind.
        let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for (position, entry) in data.iter().enumerate() {
            let idx = entry.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
            let vec = entry
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| EmbedError::Malformed("entry without `embedding`".into()))?;
            let mut v = Vec::with_capacity(vec.len());
            for x in vec {
                v.push(x.as_f64().ok_or_else(|| {
                    EmbedError::Malformed("embedding component is not a number".into())
                })? as f32);
            }
            indexed.push((idx.unwrap_or(position), mrl_truncate(v, self.cfg.dims)));
        }
        indexed.sort_by_key(|(i, _)| *i);
        Ok(indexed.into_iter().map(|(_, v)| v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    /// Serve exactly one `/embeddings` request from a real socket, capture
    /// what curl actually sent, and reply with the given JSON. Nothing is
    /// mocked below the HTTP boundary — the transport *is* the thing under
    /// test here (the `c7_smoke` pattern, on the curl side).
    fn serve_once(
        reply: &'static str,
        status: u16,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut captured = request_line.clone();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                captured.push_str(&line);
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length: ") {
                    content_length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).unwrap();
            captured.push_str(&String::from_utf8_lossy(&body));
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\n\
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
    fn posts_the_openai_shape_over_curl_and_orders_by_index() {
        let reply = r#"{"data":[
            {"index":1,"embedding":[0.0,1.0,0.0,0.0]},
            {"index":0,"embedding":[1.0,0.0,0.0,0.0]}
        ]}"#;
        let (addr, captured) = serve_once(reply, 200);
        let e = CurlEmbedder::new(EmbedConfig {
            url: format!("http://{addr}"),
            model: "embeddinggemma-2".into(),
            dims: 0,
            key: Some("secret".into()),
        });
        let out = e.embed_batch(&["alpha", "beta"]).unwrap();
        let request = captured.join().unwrap();
        assert!(request.starts_with("POST /embeddings HTTP/1.1"), "{request}");
        assert!(request.contains("authorization: Bearer secret") || request.contains("Authorization: Bearer secret"), "{request}");
        assert!(request.contains("embeddinggemma-2") && request.contains("alpha") && request.contains("beta"));
        assert_eq!(out[0], vec![1.0, 0.0, 0.0, 0.0], "index 0 first");
        assert_eq!(out[1], vec![0.0, 1.0, 0.0, 0.0], "index 1 second");
    }

    #[test]
    fn mrl_truncation_and_renormalization_happen_client_side() {
        let reply = r#"{"data":[{"index":0,"embedding":[1.0,0.0,0.0,0.0]}]}"#;
        let (addr, captured) = serve_once(reply, 200);
        let e = CurlEmbedder::new(EmbedConfig {
            url: format!("http://{addr}"),
            model: "embeddinggemma-2".into(),
            dims: 2,
            key: None,
        });
        let v = e.embed_one("x").unwrap();
        assert_eq!(v.len(), 2, "truncated to the configured MRL dims");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "renormalized after truncation");
        let request = captured.join().unwrap();
        assert!(request.contains("\"dimensions\":2"), "also asked the server: {request}");
        assert!(!request.to_ascii_lowercase().contains("authorization"), "no empty Authorization header");
    }

    #[test]
    fn a_non_2xx_is_a_status_error_carrying_the_body() {
        let (addr, _captured) = serve_once(r#"{"error":"no such model"}"#, 404);
        let e = CurlEmbedder::new(EmbedConfig {
            url: format!("http://{addr}"),
            model: "nope".into(),
            dims: 0,
            key: None,
        });
        match e.embed_one("x") {
            Err(EmbedError::Status { code: 404, body }) => assert!(body.contains("no such model"), "{body}"),
            other => panic!("expected Status(404), got {other:?}"),
        }
    }

    #[test]
    fn an_unreachable_endpoint_is_a_transport_error_not_status_zero() {
        // Nothing listens on port 1: curl exits non-zero with http_code
        // 000/0, which must not be reported as a status a server sent.
        let e = CurlEmbedder::new(EmbedConfig {
            url: "http://127.0.0.1:1".to_string(),
            model: "m".into(),
            dims: 0,
            key: None,
        });
        match e.embed_one("x") {
            Err(EmbedError::Transport(msg)) => assert!(msg.contains("cannot reach"), "{msg}"),
            other => panic!("expected Transport, got {other:?}"),
        }
    }

    #[test]
    fn a_wrong_count_is_an_error_not_a_zip() {
        let (addr, _captured) = serve_once(r#"{"data":[{"index":0,"embedding":[1.0]}]}"#, 200);
        let e = CurlEmbedder::new(EmbedConfig {
            url: format!("http://{addr}"),
            model: "m".into(),
            dims: 0,
            key: None,
        });
        match e.embed_batch(&["a", "b"]) {
            Err(EmbedError::Count { want: 2, got: 1 }) => {}
            other => panic!("expected Count{{2,1}}, got {other:?}"),
        }
    }
}
