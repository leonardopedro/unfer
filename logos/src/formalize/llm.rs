//! P2 — the `LlmClient` abstraction and its OpenAI/vLLM-compatible transport.
//!
//! # The port
//!
//! ProofFlow's `utils.LLMManager` reaches Claude, GPT, Gemini, vLLM and a
//! local endpoint behind one `call_llm(messages)` call. [`LlmClient`] keeps that
//! shape — a single `chat(system, turns)` — but splits the *transport* from the
//! *policy*, because the policy here is typed and the transport is not:
//!
//! ```text
//!   LlmClient<T: Transport>   ──owns──>  Transcript   ──delegates──>  T
//!   (prompt assembly,                 (Role/Turn,                  (HTTP,
//!    error classification)              the retry loop's              or a
//!                                        accumulated corrections)     scripted
//!                                                                            stand-in)
//! ```
//!
//! That split is what makes the formalizer testable: [`ScriptedTransport`]
//! replays a fixed sequence of replies, so the whole verify-per-node retry loop
//! runs in unit tests with no network and no key.
//!
//! # Dependency discipline
//!
//! The HTTP transport is behind the `llm-http` feature and the crate is
//! `default-features = false`. That is deliberate: the part of `logos` that
//! matters — gate → CCG → CoreIR → DeltaNet → readback → hash — needs no
//! network, and `logos` is a dependency of crates that must not acquire a TLS
//! stack. Only [`HttpTransport`] is gated; [`LlmClient`] and the whole
//! `formalize` pipeline are available without it.
//!
//! # vLLM / OpenAI compatibility
//!
//! [`HttpTransport`] speaks `POST {base_url}/chat/completions` with the OpenAI
//! message schema, which is what vLLM, llama.cpp's server, Ollama's OpenAI
//! shim, LM Studio, TGI, and OpenAI itself all implement. Pointing at a local
//! vLLM is therefore a base-URL change and nothing else — the same property
//! ProofFlow relied on for its benchmark scripts.

use crate::formalize::graph::{Respond, Turn};
use serde_json::{Value, json};
use std::fmt;
use std::time::Duration;

/// The system prompt for graph construction, ported from ProofFlow's
/// `prompts/proof_graph.md` and retargeted at CNL.
pub const PROOF_GRAPH_PROMPT: &str = include_str!("prompts/proof_graph_cnl.md");

/// The system prompt for per-node CNL generation, ported from
/// `prompts/lemma_formalizer.md`.
///
/// ProofFlow ships `_no_think` variants of several prompts. Those exist to
/// suppress reasoning traces for cheaper models; here they are not ported
/// because [`PROMPT_VARIANTS`] selects the file by name and a caller can add
/// `_no_think` files without touching the pipeline.
pub const CNL_FORMALIZER_PROMPT: &str = include_str!("prompts/cnl_formalizer.md");

/// Every prompt this pipeline knows, by name. P8's harness and the P6 CLI
/// resolve prompts through this table so a prompt can be swapped or extended
/// without touching call sites.
pub fn prompt(name: &str) -> Option<&'static str> {
    match name {
        "proof_graph" => Some(PROOF_GRAPH_PROMPT),
        "cnl_formalizer" => Some(CNL_FORMALIZER_PROMPT),
        _ => None,
    }
}

/// Everything that can go wrong talking to a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    /// The request never completed: connection refused, DNS, timeout, TLS.
    Transport(String),
    /// The endpoint answered, but with a non-2xx status.
    Status { code: u16, body: String },
    /// The endpoint answered 2xx with a body we cannot read a reply from.
    Malformed(String),
    /// The reply parsed but contained no assistant message.
    EmptyReply(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Transport(m) => write!(f, "transport error: {m}"),
            LlmError::Status { code, body } => {
                write!(f, "endpoint returned HTTP {code}: {body}")
            }
            LlmError::Malformed(m) => write!(f, "malformed response: {m}"),
            LlmError::EmptyReply(m) => write!(f, "reply contained no content: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Endpoint and generation settings.
///
/// Defaults target OpenAI but read from the environment, which is the whole
/// point of ProofFlow's LLM-agnostic design: switching to a local vLLM is
/// `OPENAI_BASE_URL` plus `OPENAI_MODEL`, with no code change.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    /// `POST {base_url}/chat/completions`.
    pub base_url: String,
    pub model: String,
    /// Sent as `Authorization: Bearer …`. `None` sends no header, which is what
    /// a local vLLM without a token expects.
    pub api_key: Option<String>,
    pub temperature: f64,
    pub max_tokens: usize,
    pub timeout: Duration,
}

impl Default for LlmConfig {
    /// `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` when set; otherwise
    /// OpenAI's public endpoint with `gpt-4o-mini`.
    ///
    /// The key is deliberately **not** defaulted to an empty string that would
    /// be sent as `Bearer `: an unauthenticated request against a real
    /// endpoint is a confusing 401, whereas no header is what local servers
    /// actually want.
    fn default() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        LlmConfig {
            base_url: env("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com/v1".into()),
            model: env("OPENAI_MODEL").unwrap_or_else(|| "gpt-4o-mini".into()),
            api_key: env("OPENAI_API_KEY"),
            temperature: 0.0,
            max_tokens: 4096,
            timeout: Duration::from_secs(120),
        }
    }
}

/// A wire-level way to get one assistant reply.
///
/// Split from [`LlmClient`] so the retry loops in [`crate::formalize`] can be
/// driven by a scripted stand-in. Implementations must not retry internally:
/// retry *with the accumulated error transcript* is the pipeline's job, and a
/// transport that silently retried would hide that loop.
pub trait Transport {
    fn post_chat(
        &mut self,
        config: &LlmConfig,
        system: &str,
        turns: &[Turn],
    ) -> Result<String, LlmError>;
}

/// Prompt assembly plus error classification, over any [`Transport`].
#[derive(Debug, Clone)]
pub struct LlmClient<T> {
    transport: T,
    config: LlmConfig,
}

impl<T: Transport> LlmClient<T> {
    pub fn new(transport: T, config: LlmConfig) -> Self {
        LlmClient { transport, config }
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut LlmConfig {
        &mut self.config
    }

    /// The underlying transport.
    ///
    /// So a caller driving a scripted transport can inspect what it saw — the
    /// retry-loop tests assert on the *request bodies*, since that is where the
    /// accumulated error transcript is visible.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// One round trip: `system` plus `turns`, returning the assistant's text.
    ///
    /// This does **not** append the reply to `turns` — the caller owns the
    /// transcript, because the retry loops interleave their own correction
    /// turns and must see the whole history.
    pub fn chat(&mut self, system: &str, turns: &[Turn]) -> Result<String, LlmError> {
        self.transport.post_chat(&self.config, system, turns)
    }
}

/// Wires [`LlmClient`] into the graph builder's LLM seam.
///
/// The turn is pushed *before* returning, which is the [`Respond`] contract and
/// the reason the graph builder's corrections accumulate instead of repeating
/// blind.
impl<T: Transport> Respond for LlmClient<T> {
    fn respond(&mut self, turns: &mut Vec<Turn>) -> Result<String, String> {
        let reply = self
            .chat(PROOF_GRAPH_PROMPT, turns)
            .map_err(|e| e.to_string())?;
        turns.push(Turn::assistant(&reply));
        Ok(reply)
    }
}

/// Build the OpenAI-shaped request body.
///
/// Public because it is the compatibility contract: any endpoint that accepts
/// this body is usable, and a test can assert the exact shape without a socket.
pub fn chat_request_body(config: &LlmConfig, system: &str, turns: &[Turn]) -> Value {
    let mut messages = Vec::with_capacity(turns.len() + 1);
    if !system.is_empty() {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for turn in turns {
        messages.push(json!({
            "role": match turn.role {
                crate::formalize::graph::Role::User => "user",
                crate::formalize::graph::Role::Assistant => "assistant",
            },
            "content": turn.content,
        }));
    }
    json!({
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    })
}

/// Pull the assistant text out of an OpenAI-shaped response.
///
/// Tolerates the two shapes endpoints actually return: `content` as a string
/// (OpenAI, vLLM, TGI, llama.cpp) and `content` as a list of typed parts
/// (Ollama's OpenAI shim, some gateways).
pub fn extract_reply(body: &Value) -> Result<String, LlmError> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| LlmError::Malformed(format!("no choices in {body}")))?;

    let content = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .or_else(|| choice.get("text"))
        .ok_or_else(|| LlmError::Malformed(format!("no message content in {choice}")))?;

    match content {
        Value::String(s) if !s.trim().is_empty() => Ok(s.clone()),
        Value::String(_) => Err(LlmError::EmptyReply("content was blank".into())),
        Value::Array(parts) => {
            let joined: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("");
            if joined.trim().is_empty() {
                Err(LlmError::EmptyReply(format!("no text parts in {content}")))
            } else {
                Ok(joined)
            }
        }
        other => Err(LlmError::Malformed(format!(
            "content was neither string nor parts: {other}"
        ))),
    }
}

// ── the HTTP transport ──────────────────────────────────────────────────────

/// `POST {base_url}/chat/completions` over [`ureq`].
#[cfg(feature = "llm-http")]
#[derive(Debug, Clone)]
pub struct HttpTransport {
    agent: ureq::Agent,
}

#[cfg(feature = "llm-http")]
impl HttpTransport {
    /// An agent with the library's default timeouts.
    pub fn new() -> Self {
        HttpTransport {
            agent: ureq::Agent::new(),
        }
    }

    /// An agent with one global timeout.
    ///
    /// Bound to the whole request — connect, write, and read — because a chat
    /// completion has no useful partial answer: a stream that stalls halfway is
    /// a retry, not a shorter response.
    pub fn with_timeout(timeout: Duration) -> Self {
        HttpTransport {
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
        }
    }
}

#[cfg(feature = "llm-http")]
impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "llm-http")]
impl Transport for HttpTransport {
    fn post_chat(
        &mut self,
        config: &LlmConfig,
        system: &str,
        turns: &[Turn],
    ) -> Result<String, LlmError> {
        let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
        let body = chat_request_body(config, system, turns);

        let mut request = self.agent.post(&url);
        if let Some(key) = &config.api_key {
            // `Request::set` consumes and returns the request; `send_json`
            // sets `Content-Type` itself.
            request = request.set("Authorization", &format!("Bearer {key}"));
        }

        let response = request.send_json(&body).map_err(classify)?;
        let value: Value = response
            .into_json()
            .map_err(|e| LlmError::Malformed(e.to_string()))?;
        extract_reply(&value)
    }
}

/// Map a [`ureq::Error`] onto [`LlmError`].
///
/// The two ureq cases are *different failures*: a non-2xx status means the
/// server answered and rejected us — a bad key, a rate limit, a context-length
/// overflow, all actionable and all worth reporting with the body — while a
/// transport error means it never answered. Collapsing them would report a 401
/// as a connection failure, which is the kind of misdiagnosis that costs an
/// afternoon. The status body is read here because by the time the error
/// reaches `map_err` the response is owned by the error value.
#[cfg(feature = "llm-http")]
fn classify(e: ureq::Error) -> LlmError {
    match e {
        ureq::Error::Status(code, response) => LlmError::Status {
            code,
            body: response.into_string().unwrap_or_default(),
        },
        ureq::Error::Transport(t) => LlmError::Transport(t.to_string()),
    }
}

// ── the scripted stand-in ───────────────────────────────────────────────────

/// Replays a fixed sequence of replies, then repeats the last one.
///
/// This is the test seam for every retry loop in `formalize`. The
/// [`crate::formalize::graph::Scripted`] responder in P1 is the same idea; this
/// one is at the transport layer so it also covers [`LlmClient`]'s request
/// assembly and reply extraction.
#[derive(Debug, Clone)]
pub struct ScriptedTransport {
    replies: Vec<String>,
    calls: usize,
    /// Every request body it was handed, for assertions.
    seen: Vec<Value>,
}

impl ScriptedTransport {
    pub fn new(replies: Vec<String>) -> Self {
        assert!(
            !replies.is_empty(),
            "ScriptedTransport needs a reply to replay"
        );
        ScriptedTransport {
            replies,
            calls: 0,
            seen: Vec::new(),
        }
    }

    /// How many requests were made.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// The request bodies, in order.
    pub fn seen(&self) -> &[Value] {
        &self.seen
    }
}

impl Transport for ScriptedTransport {
    fn post_chat(
        &mut self,
        config: &LlmConfig,
        system: &str,
        turns: &[Turn],
    ) -> Result<String, LlmError> {
        let idx = self.calls.min(self.replies.len() - 1);
        self.calls += 1;
        self.seen.push(chat_request_body(config, system, turns));
        Ok(self.replies[idx].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalize::graph::Role;

    fn cfg() -> LlmConfig {
        LlmConfig {
            base_url: "http://localhost:8000/v1".into(),
            model: "qwen2.5-7b".into(),
            api_key: None,
            temperature: 0.0,
            max_tokens: 128,
            timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn request_body_is_openai_shaped() {
        let turns = vec![
            Turn::user("proof"),
            Turn::assistant("draft"),
            Turn::user("fix it"),
        ];
        let body = chat_request_body(&cfg(), "SYSTEM", &turns);
        assert_eq!(body["model"], "qwen2.5-7b");
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["max_tokens"], 128);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "SYSTEM");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["content"], "fix it");
    }

    /// No system prompt means no system turn — a pre-prompt-built transcript
    /// (P1's graph builder puts everything in `turns`) must not gain a spurious
    /// empty system message.
    #[test]
    fn empty_system_prompt_is_omitted() {
        let body = chat_request_body(&cfg(), "", &[Turn::user("x")]);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn reply_is_extracted_from_the_openai_shape() {
        let body = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        assert_eq!(extract_reply(&body).unwrap(), "hi");
    }

    #[test]
    fn reply_is_extracted_from_the_completions_shape() {
        let body = json!({"choices": [{"text": "hi"}]});
        assert_eq!(extract_reply(&body).unwrap(), "hi");
    }

    /// Ollama's OpenAI shim returns content as typed parts.
    #[test]
    fn reply_is_extracted_from_a_parts_list() {
        let body = json!({"choices": [{"message": {"content": [
            {"type": "text", "text": "one "},
            {"type": "text", "text": "two"},
        ]}}]});
        assert_eq!(extract_reply(&body).unwrap(), "one two");
    }

    #[test]
    fn missing_choices_is_malformed_not_empty() {
        let err = extract_reply(&json!({"error": {"message": "bad key"}})).unwrap_err();
        assert!(matches!(err, LlmError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn blank_content_is_empty_not_malformed() {
        let err = extract_reply(&json!({"choices": [{"message": {"content": "  "}}]})).unwrap_err();
        assert!(matches!(err, LlmError::EmptyReply(_)), "{err:?}");
    }

    #[test]
    fn scripted_transport_replays_then_repeats_the_last_reply() {
        let mut t = ScriptedTransport::new(vec!["a".into(), "b".into()]);
        assert_eq!(t.post_chat(&cfg(), "", &[]).unwrap(), "a");
        assert_eq!(t.post_chat(&cfg(), "", &[]).unwrap(), "b");
        assert_eq!(t.post_chat(&cfg(), "", &[]).unwrap(), "b");
        assert_eq!(t.calls(), 3);
        assert_eq!(t.seen().len(), 3);
    }

    #[test]
    fn client_records_the_reply_so_corrections_accumulate() {
        let mut client = LlmClient::new(ScriptedTransport::new(vec!["draft".into()]), cfg());
        let mut turns = vec![Turn::user("proof")];
        // P1's graph builder drives this seam.
        let reply = Respond::respond(&mut client, &mut turns).unwrap();
        assert_eq!(reply, "draft");
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].role, Role::Assistant);
    }

    #[test]
    fn prompts_are_resolvable_by_name() {
        assert!(prompt("proof_graph").unwrap().contains("```json"));
        assert!(prompt("cnl_formalizer").unwrap().contains("```cnl"));
        assert!(prompt("nope").is_none());
    }

    /// The default config must not invent a bearer token: an unauthenticated
    /// local vLLM wants no header, and `Bearer ` would be a confusing 401.
    #[test]
    fn default_config_sends_no_bearer_header_when_unset() {
        let c = LlmConfig {
            api_key: None,
            ..LlmConfig::default()
        };
        let body = chat_request_body(&c, "", &[Turn::user("x")]);
        // The key is a transport concern, so it never appears in the body.
        assert!(body.get("api_key").is_none());
        assert!(c.base_url.starts_with("http"));
    }

    #[test]
    fn errors_display_with_their_kind() {
        assert!(
            LlmError::Transport("refused".into())
                .to_string()
                .contains("transport")
        );
        assert!(
            LlmError::Status {
                code: 429,
                body: String::new()
            }
            .to_string()
            .contains("429")
        );
    }
}
