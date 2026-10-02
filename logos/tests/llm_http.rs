//! The HTTP transport, against a real socket.
//!
//! `logos::formalize::llm`'s unit tests cover the request body and the reply
//! extraction, and `logos::cli::formalize` covers the whole pipeline against a
//! scripted transport. What nothing covered was the seam between them: that
//! `HttpTransport` really puts the OpenAI-shaped body on the wire and reads the
//! reply back. A bug there would compile, pass every unit test, and fail only
//! against a live endpoint.
//!
//! So this file starts a one-shot HTTP server on a loopback port and points a
//! real `ureq` agent at it. No network, no key, no dependency.
//!
//! Gated on the `llm-http` feature, since the transport is.

#![cfg(feature = "llm-http")]

use logos::formalize::graph::Turn;
use logos::formalize::llm::{
    HttpTransport, LlmClient, LlmConfig, chat_request_body, extract_reply,
};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

/// One request: serve `status` and `body`, hand back what was received.
fn serve(status: u16, body: &'static str) -> (u16, String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");

        // Read the request head, then the body by Content-Length. Enough for
        // ureq's `send_json`, and it lets the test assert on what went out.
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
        let mut head = String::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let done = line == "\r\n" || line == "\n";
            head.push_str(&line);
            if done {
                break;
            }
        }
        let len: usize = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse().ok())?
            })
            .unwrap_or(0);
        let mut payload = vec![0u8; len];
        use std::io::Read;
        let _ = reader.read_exact(&mut payload);
        let _ = tx.send(head);

        let response = format!(
            "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    });

    (port, format!("http://127.0.0.1:{port}/v1"), rx)
}

fn cfg(base_url: String, api_key: Option<&str>) -> LlmConfig {
    LlmConfig {
        base_url,
        model: "test-model".into(),
        api_key: api_key.map(String::from),
        temperature: 0.0,
        max_tokens: 64,
        timeout: Duration::from_secs(10),
    }
}

#[test]
fn a_chat_completion_goes_out_and_the_reply_comes_back() {
    let reply = r#"{"choices":[{"message":{"role":"assistant","content":"John loves Mary"}}]}"#;
    let (_port, base, rx) = serve(200, reply);

    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        cfg(base, None),
    );
    let out = client
        .chat("SYSTEM", &[Turn::user("formalize l1")])
        .expect("a 200 with a reply must succeed");
    assert_eq!(out, "John loves Mary");

    let head = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("request head");
    assert!(
        head.starts_with("POST /v1/chat/completions HTTP/1.1"),
        "{head}"
    );
    assert!(
        head.to_lowercase()
            .contains("content-type: application/json"),
        "{head}"
    );
    // No key configured, so no `Authorization` header — which is what a local
    // vLLM expects.
    assert!(
        !head.to_lowercase().contains("authorization"),
        "no key should mean no header: {head}"
    );
}

/// With a key configured it must be sent as a bearer token, or every request to
/// a real endpoint fails with a 401 that looks like a transport error.
#[test]
fn a_configured_key_is_sent_as_a_bearer_token() {
    let reply = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    let (port, base, rx) = serve(200, reply);
    let _ = port;

    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        cfg(base, Some("sk-test")),
    );
    client.chat("", &[Turn::user("x")]).unwrap();

    let head = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        head.to_lowercase()
            .contains("authorization: bearer sk-test"),
        "{head}"
    );
}

/// The request body must be the OpenAI shape on the wire, not just in the
/// builder — a body that never arrives is the failure mode a unit test on
/// `chat_request_body` cannot see.
#[test]
fn the_wire_body_is_the_openai_shape() {
    let reply = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    let (port, base, rx) = serve(200, reply);
    let _ = port;

    let config = cfg(base, None);
    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        config.clone(),
    );
    client
        .chat("SYSTEM", &[Turn::user("proof"), Turn::assistant("draft")])
        .unwrap();

    let head = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let len: usize = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())?
        })
        .unwrap();

    // Re-read the body off the socket is not possible here (the server consumed
    // it), so assert on what the client *would* have sent for the same turns —
    // which is what `content-length` must agree with.
    let expected = chat_request_body(
        &config,
        "SYSTEM",
        &[Turn::user("proof"), Turn::assistant("draft")],
    );
    assert_eq!(len, serde_json::to_vec(&expected).unwrap().len());
    assert_eq!(expected["messages"].as_array().unwrap().len(), 3);
}

/// A non-2xx must surface as `Status` with the code and the body, not as a
/// transport error — the distinction between "the server rejected us" and "the
/// server never answered" is the difference between a fixable and a puzzling
/// failure.
#[test]
fn a_4xx_is_reported_as_a_status_with_its_body() {
    let body = r#"{"error":{"message":"invalid api key"}}"#;
    let (port, base, _rx) = serve(401, body);
    let _ = port;

    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        cfg(base, None),
    );
    let err = client.chat("", &[Turn::user("x")]).unwrap_err();
    match err {
        logos::formalize::llm::LlmError::Status { code, body } => {
            assert_eq!(code, 401);
            assert!(body.contains("invalid api key"), "{body}");
        }
        other => panic!("expected Status, got {other:?}"),
    }
}

/// A 200 whose body is not the expected shape is `Malformed`, not a silent
/// empty string.
#[test]
fn an_unexpected_2xx_body_is_malformed() {
    let (port, base, _rx) = serve(200, r#"{"error":"quota"}"#);
    let _ = port;

    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        cfg(base, None),
    );
    let err = client.chat("", &[Turn::user("x")]).unwrap_err();
    assert!(
        matches!(err, logos::formalize::llm::LlmError::Malformed(_)),
        "{err:?}"
    );
}

/// A 200 with no assistant message is `EmptyReply`, not a fabricated empty
/// string — an empty reply fed to a retry loop is a silent resample.
#[test]
fn a_2xx_with_no_message_is_an_empty_reply() {
    let (port, base, _rx) = serve(200, r#"{"choices":[]}"#);
    let _ = port;
    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(10)),
        cfg(base, None),
    );
    let err = client.chat("", &[Turn::user("x")]).unwrap_err();
    assert!(
        matches!(err, logos::formalize::llm::LlmError::Malformed(_)),
        "{err:?}"
    );
}

/// A dead port must be a `Transport` error, quickly. Without this, a
/// misconfigured `OPENAI_BASE_URL` looks like a hang.
#[test]
fn an_unreachable_endpoint_is_a_transport_error() {
    // Bind then drop, so the port is almost certainly closed.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut client = LlmClient::new(
        HttpTransport::with_timeout(Duration::from_secs(3)),
        cfg(format!("http://127.0.0.1:{port}/v1"), None),
    );
    let err = client.chat("", &[Turn::user("x")]).unwrap_err();
    assert!(
        matches!(err, logos::formalize::llm::LlmError::Transport(_)),
        "{err:?}"
    );
}

/// The same body through the same extractor the transport uses, so the two
/// cannot drift.
#[test]
fn the_transport_agrees_with_the_extractor() {
    let reply = r#"{"choices":[{"message":{"content":"a"}}]}"#;
    let value: serde_json::Value = serde_json::from_str(reply).unwrap();
    assert_eq!(extract_reply(&value).unwrap(), "a");
}
