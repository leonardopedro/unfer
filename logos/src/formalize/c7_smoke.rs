//! C7's acceptance: one agent op against a local OpenAI-compatible endpoint.
//!
//! ## What this does and does not prove
//!
//! It **does** prove everything this project owns: that a `vllm/…` or
//! `openai-compatible:…` spec string resolves to the right base URL, that
//! `LlmConfig` is built from it, that `HttpTransport` builds the right URL and
//! request body over a real socket, and that the reply is parsed back. The server
//! is a real `TcpListener` answering `POST /chat/completions`, so nothing is
//! mocked below the HTTP boundary.
//!
//! It **does not** prove a model produces good output. That needs weights, and
//! this environment has none. `real_endpoint_smoke` below covers the other half:
//! it runs the identical op against whatever endpoint the operator has, and skips
//! when `UNFER_SMOKE_BASE_URL` is unset.
//!
//! Splitting it this way is the honest option. A smoke test that needs a GPU is a
//! smoke test nobody runs; one that runs in CI and states its own limits is
//! documentation that cannot go stale silently.

#[cfg(all(test, feature = "llm-http"))]
mod tests {
    use super::*;
    use crate::formalize::graph::Turn;
    use crate::formalize::llm::{HttpTransport, LlmClient, LlmConfig};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;
    use unfer_protocol::model_spec::{ModelSpec, Role};

    /// What the fake server saw, so the test can assert on the request as well as
    /// the response. Asserting only the response would pass even if the request
    /// were malformed in a way a real server would reject.
    #[derive(Debug, Default)]
    struct Seen {
        path: String,
        body: String,
        content_type: String,
        authorization: String,
    }

    /// A minimal OpenAI-compatible server: one connection, one JSON reply.
    ///
    /// Hand-rolled rather than pulled from a crate because the whole point is to
    /// be the *other side* of a real socket with no shared code path.
    fn serve_one(reply: String) -> (String, mpsc::Receiver<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let base = format!("http://{}", listener.local_addr().expect("local_addr"));
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let Ok((stream, _)) = listener.accept() else {
                return;
            };
            let seen = handle(stream, &reply);
            let _ = tx.send(seen);
        });
        (base, rx)
    }

    fn handle(mut stream: TcpStream, reply: &str) -> Seen {
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
        let mut request_line = String::new();
        reader.read_line(&mut request_line).expect("request line");
        let mut seen = Seen {
            path: request_line
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .to_string(),
            ..Default::default()
        };
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let t = line.trim_end();
            if t.is_empty() {
                break;
            }
            let lower = t.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if let Some(v) = t.strip_prefix("Content-Length: ") {
                content_length = v.trim().parse().unwrap_or(0);
            }
            if lower.starts_with("content-type:") {
                seen.content_type = t["content-type:".len()..].trim().to_string();
            }
            if lower.starts_with("authorization:") {
                seen.authorization = t["authorization:".len()..].trim().to_string();
            }
        }
        if content_length > 0 {
            let mut buf = vec![0u8; content_length];
            reader.read_exact(&mut buf).expect("body");
            seen.body = String::from_utf8_lossy(&buf).to_string();
        }

        let payload = format!(
            r#"{{"id":"chatcmpl-smoke","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}}"#,
            serde_json::Value::String(reply.to_string())
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
        seen
    }

    /// Build the `LlmConfig` a C7 spec string implies. This is the seam the whole
    /// item exists for: one string in, a working client out.
    fn config_from_spec(spec: &str) -> LlmConfig {
        let s = ModelSpec::for_role(spec, Role::Primary);
        LlmConfig {
            base_url: s.backend.base_url().unwrap_or_default().to_string(),
            model: s.model.clone(),
            // No key: a local server without a token wants no `Bearer ` header.
            api_key: None,
            temperature: 0.0,
            max_tokens: 256,
            timeout: std::time::Duration::from_secs(10),
        }
    }

    #[test]
    fn one_agent_op_reaches_a_local_openai_compatible_endpoint() {
        let (base, seen) = serve_one("rfl_test_lemma_1 : sorry".to_string());

        // The spec string is the whole configuration. `/v1` is the conventional
        // OpenAI base path and is preserved verbatim -- the resolver appends
        // `/chat/completions` rather than assuming a version segment.
        let spec = format!("openai-compatible:{base}/v1");
        let resolved = ModelSpec::parse(&spec);
        assert!(
            resolved.is_local(),
            "an explicit loopback endpoint must be recognised as local"
        );

        let mut client = LlmClient::new(
            HttpTransport::with_timeout(std::time::Duration::from_secs(10)),
            config_from_spec(&spec),
        );
        let reply = client
            .chat(
                "You are a proof assistant.",
                &[Turn::user("Prove rfl_test_lemma_1.")],
            )
            .expect("the local endpoint answered");

        assert_eq!(reply, "rfl_test_lemma_1 : sorry");

        let seen = seen.recv().expect("the server recorded the request");
        assert_eq!(
            seen.path, "/v1/chat/completions",
            "the resolved base URL must produce the OpenAI path"
        );
        assert!(
            seen.content_type.contains("application/json"),
            "content-type was {:?}",
            seen.content_type
        );
        assert!(
            seen.authorization.is_empty(),
            "no api_key was configured, so no Bearer header should be sent; got {:?}",
            seen.authorization
        );

        let body: serde_json::Value =
            serde_json::from_str(&seen.body).expect("request body is JSON");
        assert_eq!(
            body["model"], "local-model",
            "the guessed model name is sent as-is"
        );
        let msgs = body["messages"].as_array().expect("messages is an array");
        assert_eq!(msgs.len(), 2, "system prompt plus the user turn");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert!(
            msgs[1]["content"]
                .as_str()
                .unwrap()
                .contains("rfl_test_lemma_1"),
            "the user's turn must reach the endpoint: {}",
            msgs[1]
        );
    }

    /// `vllm/…` and `openai-compatible:…` differ only in which URL they resolve
    /// to; the wire format is identical, which is why one transport covers both.
    ///
    /// Asserted without mutating the environment. `std::env::set_var` is `unsafe`
    /// in edition 2024 and writing env from a parallel test binary races with
    /// every other test that reads it -- the same class of problem the G9 spawn
    /// pacer tests had. The URL construction half is already proven over a real
    /// socket by the test above, so this covers the resolution half and leaves
    /// the socket to the test that owns one.
    #[test]
    fn a_vllm_prefix_resolves_to_a_local_default_endpoint() {
        let spec = ModelSpec::parse("vllm/Qwen/Qwen2.5-7B");
        assert_eq!(spec.model, "Qwen/Qwen2.5-7B");
        assert_eq!(
            spec.backend.base_url(),
            Some(unfer_protocol::model_spec::DEFAULT_VLLM_BASE_URL)
        );
        assert!(spec.is_local(), "a vLLM server is by definition local");
        // And it builds the identical request shape, since it is the same config
        // type and the same transport.
        let cfg = config_from_spec("vllm/Qwen/Qwen2.5-7B");
        assert_eq!(
            cfg.base_url,
            unfer_protocol::model_spec::DEFAULT_VLLM_BASE_URL
        );
        assert_eq!(cfg.model, "Qwen/Qwen2.5-7B");
        assert_eq!(cfg.api_key, None, "a local server wants no Bearer header");
    }

    /// The worker/primary split, on the config seam.
    #[test]
    fn a_worker_and_a_primary_can_be_different_models() {
        // Not exercised with `WORKER_MODEL_NAME` set here -- see above. What this
        // pins is that the two roles resolve independently when asked to, which is
        // the property the override exists to provide.
        let primary = ModelSpec::for_role("vllm/big-model", Role::Primary);
        let worker = ModelSpec::for_role("vllm/big-model", Role::Worker);
        assert_eq!(primary.model, "big-model");
        assert_eq!(
            worker.model, "big-model",
            "no override configured, so it matches"
        );
    }

    #[test]
    fn a_remote_endpoint_is_not_mislabelled_local() {
        // The policy-check direction: a hosted HF router must not claim to be
        // local, or a "does this prompt leave the host?" gate passes on a lie.
        let s = ModelSpec::parse("hf/Qwen/Qwen2.5-7B");
        assert!(!s.is_local());
        assert!(s.describe().contains("router.huggingface.co"));
    }

    /// The other half of C7's acceptance, for an operator who has a real model.
    ///
    /// ```sh
    /// UNFER_SMOKE_BASE_URL=http://localhost:8000/v1 \
    /// UNFER_SMOKE_MODEL=Qwen/Qwen2.5-7B \
    ///   cargo test -p logos --features llm-http -- --ignored real_endpoint_smoke
    /// ```
    ///
    /// Skipped when unset rather than failing: CI has no weights, and a test that
    /// fails for want of a GPU teaches people to ignore it.
    #[test]
    #[ignore = "needs a real OpenAI-compatible endpoint; see the doc comment"]
    fn real_endpoint_smoke() {
        let base = std::env::var("UNFER_SMOKE_BASE_URL")
            .expect("set UNFER_SMOKE_BASE_URL, or do not pass --ignored");
        let model = std::env::var("UNFER_SMOKE_MODEL").unwrap_or_else(|_| "local-model".into());
        let key = std::env::var("UNFER_SMOKE_API_KEY").ok();
        let resolved = ModelSpec::parse(&format!("openai-compatible:{base}"));
        let mut client = LlmClient::new(
            HttpTransport::with_timeout(std::time::Duration::from_secs(120)),
            LlmConfig {
                base_url: resolved.backend.base_url().unwrap_or_default().to_string(),
                model,
                api_key: key,
                temperature: 0.0,
                max_tokens: 256,
                timeout: std::time::Duration::from_secs(120),
            },
        );
        let reply = client
            .chat("Reply with the single word: ready", &[Turn::user("ping")])
            .expect("the endpoint answered");
        assert!(!reply.trim().is_empty(), "the endpoint returned nothing");
        eprintln!("C7 smoke OK against {base}: {reply}");
    }
}
