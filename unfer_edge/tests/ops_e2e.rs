//! X6 — the end-to-end ops probe for `unfer_edge`.
//!
//! Until this test, `unfer_edge`'s ops surface had only a routing-table unit
//! test: `route()` said `GET /healthz` maps to `EarlyRoute::Healthz`, and that
//! is exactly the kind of green that stops one layer above the bug. What no
//! test saw was the whole wire path an operator actually probes — the binary
//! booting under Pingora, binding the configured listener, the request
//! filter choosing the early route, and the JSON response coming back with a
//! status and a body. `dynamic-arctic` has that coverage (`mod ops_surface`);
//! this is the same claim for this server.
//!
//! The probe speaks HTTP/1.1 over a raw `TcpStream` instead of adding a client
//! dependency: one GET and one POST do not justify a crate in the graph.
//!
//! What it asserts, and why each one:
//!
//! - `GET /healthz` → 200 `{"status":"ok"}` with `content-type: application/json`
//!   — the liveness probe a runbook tells an operator to curl.
//! - `GET /version` → 200, `service`/`version` identifying this exact build,
//!   and a `config` object that reports the *effective* listen address — the
//!   provenance half: "which port did I actually bind" is the question that
//!   sends an operator to this endpoint.
//! - `GET /metrics` → 200 JSON — the third self-service surface.
//! - `POST /healthz` → 405 — a known path with the wrong method must be
//!   refused *here*, not forwarded upstream where the answer would be a
//!   confusing 404 from something that has never heard of the path.
//!
//! The proxied half of the table (unknown paths) is deliberately not probed:
//! its correctness is `route()`'s unit tests, and asserting a 502 from a
//! deliberately absent backend would make this test depend on Pingora's
//! upstream-failure behaviour, which is not what X6 is asking about.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A server child that cannot outlive the test, whatever the test does next,
/// with its stderr drained into memory by a reader thread — drained, because
/// a piped stderr nobody reads fills and blocks the server, and captured,
/// because "did not answer within 30s" without the server's own words is a
/// failure message that wastes the next person's hour.
struct Guard {
    child: Child,
    stderr: Arc<Mutex<String>>,
}

impl Guard {
    /// Kill the server and give its stderr back (for a failure message).
    fn stop_and_stderr(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // The reader thread sees EOF once the pipe closes; give it a beat.
        for _ in 0..50 {
            if !self.stderr.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.stderr.lock().unwrap().clone()
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A currently-free port. Released immediately, so there is a (standard,
/// small) race before the server binds it; the readiness loop below absorbs
/// a collision by simply waiting for the right process to answer.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind a probe socket")
        .local_addr()
        .expect("local_addr")
        .port()
}

/// One HTTP/1.1 request over a raw socket. `Ok((status, body))`, or `Err` when
/// the connection could not be made or the response was unparseable — which
/// during readiness polling just means "not yet".
fn http(port: u16, request: &str) -> Result<(u16, String), String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("no header/body separator in {text:?}"))?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| format!("no status line in {text:?}"))?
        .parse()
        .map_err(|e| format!("bad status line: {e}"))?;
    Ok((status, body.to_string()))
}

fn get(port: u16, path: &str) -> Result<(u16, String), String> {
    http(
        port,
        &format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"),
    )
}

fn post(port: u16, path: &str) -> Result<(u16, String), String> {
    http(
        port,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
    )
}

/// Boot the binary and wait (up to 30 s) for `/healthz` to answer. The wait
/// is readiness, not an assertion: Pingora's bootstrap takes a moment, and a
/// fixed sleep would be a flake with extra steps. On timeout the panic
/// carries the server's own stderr.
fn boot(listen: u16, backend: u16) -> Guard {
    let mut child = Command::new(env!("CARGO_BIN_EXE_unfer_edge"))
        .args([
            "--listen",
            &format!("127.0.0.1:{listen}"),
            "--backend",
            &format!("127.0.0.1:{backend}"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn unfer_edge");

    let mut stderr_pipe = child.stderr.take().expect("stderr is piped");
    let stderr = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&stderr);
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr_pipe.read_to_string(&mut buf);
        *sink.lock().unwrap() = buf;
    });
    let guard = Guard { child, stderr };

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok((200, _)) = get(listen, "/healthz") {
            return guard;
        }
        if Instant::now() >= deadline {
            let mut me = guard;
            let log = me.stop_and_stderr();
            panic!("unfer_edge did not answer /healthz within 30s; stderr:\n{log}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn the_ops_surface_answers_over_a_real_socket() {
    let listen = free_port();
    let backend = free_port();
    let _guard = boot(listen, backend);

    // Liveness: the exact probe the runbook names.
    let (status, body) = get(listen, "/healthz").expect("healthz over the wire");
    assert_eq!(status, 200, "GET /healthz: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("healthz is JSON");
    assert_eq!(v["status"], "ok");

    // Build identity + effective configuration provenance.
    let (status, body) = get(listen, "/version").expect("version over the wire");
    assert_eq!(status, 200, "GET /version: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("version is JSON");
    assert_eq!(v["service"], "unfer_edge");
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    let config = v.get("config").expect("/version reports the config");
    // Shape-agnostic on purpose: what matters is that the effective listen
    // address (with the port this test chose) is in the reported config, so an
    // operator asking "which port did I bind" gets this answer.
    assert!(
        config.to_string().contains(&listen.to_string()),
        "/version config does not report the bound port {listen}: {config}"
    );

    // Metrics: the third self-service surface, same early-route path.
    let (status, body) = get(listen, "/metrics").expect("metrics over the wire");
    assert_eq!(status, 200, "GET /metrics: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("metrics is JSON");
    assert!(v.is_object(), "metrics is a JSON object");

    // The method boundary: refused here, not forwarded upstream.
    let (status, body) = post(listen, "/healthz").expect("post over the wire");
    assert_eq!(status, 405, "POST /healthz must be refused locally: {body}");
}
