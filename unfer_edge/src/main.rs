//! unfer_edge — Pingora-based security-first proxy fronting the unfer_agent
//! NDJSON loop (P11.22).
//!
//! ## Architecture
//!
//! ```text
//!   client ──HTTP──► unfer_edge (this binary)
//!                          │ 1. parse body as AgentRequest
//!                          │ 2. validate op against ALLOWED_OPS (UK-4001 on deny)
//!                          │ 3. forward to backend unfer_agent HTTP-wrapper
//!                          ▼
//!                    unfer_agent process (port 3001, NDJSON)
//! ```
//!
//! ## Running
//!
//! ```sh
//! # Start the backend agent first (port 3001).
//! unfer_agent --listen 127.0.0.1:3001 &
//!
//! # Start this proxy (port 3000, forwards to 127.0.0.1:3001).
//! unfer_edge --listen 127.0.0.1:3000 --backend 127.0.0.1:3001
//! ```

#[cfg(feature = "audit")]
mod admin;
#[cfg(feature = "audit")]
mod audit;
#[cfg(feature = "audit")]
mod blueprint;
#[cfg(feature = "audit")]
mod caprpc;
mod cells;
mod filter;
#[cfg(feature = "audit")]
mod gate;
mod mask;
mod metrics;

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use async_trait::async_trait;
use pingora_core::prelude::*;
use pingora_http::ResponseHeader;
use pingora_proxy::{ProxyHttp, Session, http_proxy_service};
use tracing::info;
use unfer_data::CellStore;

/// Process-local content store backing the `/cell/<cid>` read route. Seed points
/// arrive via blueprint publication (S20); absent that, the route is a shape-checked 404.
fn cell_store() -> &'static Mutex<CellStore> {
    static STORE: OnceLock<Mutex<CellStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(CellStore::new()))
}

/// Process-local op counters (S13): `/metrics` serves the snapshot.
fn edge_metrics() -> &'static metrics::Metrics {
    static METRICS: OnceLock<metrics::Metrics> = OnceLock::new();
    METRICS.get_or_init(metrics::Metrics::new)
}

/// Gateway configuration — set by CLI arguments.
#[derive(Clone)]
struct GatewayConf {
    /// Host:port of the backend unfer_agent NDJSON HTTP server.
    backend_addr: String,
}

/// The Pingora `ProxyHttp` implementation for the unfer gateway.
struct UnferGateway {
    conf: Arc<GatewayConf>,
}

/// Per-request state: buffers upstream response body chunks so the
/// data-masking filter can operate on the complete JSON envelope rather
/// than a partial chunk (masking a truncated JSON document would be unsafe).
#[derive(Default)]
struct GatewayCtx {
    upstream_body: Vec<u8>,
}

#[async_trait]
impl ProxyHttp for UnferGateway {
    type CTX = GatewayCtx;

    fn new_ctx(&self) -> Self::CTX {
        GatewayCtx::default()
    }

    /// Route every request to the single backend.
    async fn upstream_peer(
        &self,
        _session: &mut Session,
        _ctx: &mut GatewayCtx,
    ) -> pingora_core::Result<Box<HttpPeer>> {
        let peer = HttpPeer::new(&self.conf.backend_addr, false, String::new());
        Ok(Box::new(peer))
    }

    /// Validate the `AgentRequest` before the request reaches the backend.
    ///
    /// Returns `true` to short-circuit (already sent a rejection response) or
    /// `false` to let Pingora forward the request normally.
    async fn request_filter(
        &self,
        session: &mut Session,
        _ctx: &mut GatewayCtx,
    ) -> pingora_core::Result<bool> {
        // S13 (F12): per-op metrics before any forwarding. GET /metrics → JSON;
        // GET /metrics?format=prometheus → text exposition. Final.
        if session.req_header().uri.path() == "/metrics" {
            if session.req_header().method != "GET" {
                let header = ResponseHeader::build(405u16, None)?;
                session
                    .write_response_header(Box::new(header), false)
                    .await?;
                session.write_response_body(None, true).await?;
                return Ok(true);
            }
            let body = match session.req_header().uri.query() {
                Some(q) if q.contains("format=prometheus") => edge_metrics()
                    .to_prometheus(&filter::allowed_ops_vec(), &[])
                    .into_bytes(),
                _ => serde_json::to_vec(&edge_metrics().to_json(&filter::allowed_ops_vec(), &[]))
                    .unwrap_or_else(|_| b"".to_vec()),
            };
            let body = bytes::Bytes::from(body);
            let mut header = ResponseHeader::build(200u16, None)?;
            header.insert_header("content-type", "application/json")?;
            header.insert_header("content-length", body.len().to_string())?;
            session
                .write_response_header(Box::new(header), false)
                .await?;
            session.write_response_body(Some(body), true).await?;
            return Ok(true);
        }

        // S28 (F27): object-capability RPC — POST /api/cap/invoke executes a
        // capability-bound method (minted only at the loopback chokepoint).
        // A method may return a nested capability stub.
        #[cfg(feature = "audit")]
        if session.req_header().uri.path().starts_with("/api/cap/") {
            let path = session.req_header().uri.path().to_string();
            let raw = match read_body(session).await {
                Ok(b) => b,
                Err(_) => {
                    return write_json(session, 400u16, b"{\"error\":\"body too large\"}")
                        .await
                        .map(|_| true);
                }
            };
            // Mint/promise/revoke routes carry the operation's payload; invoke
            // carries a CapCall. The caller is the request's principal-less
            // identity for now; minted capabilities are owned by the admin
            // principal (S22 seam, `UNFER_ADMIN_PRINCIPAL`). In a full
            // deployment the caller comes from the authenticated session.
            let caller = admin::admin_principal();
            let (status, body): (u16, Vec<u8>) = match path.as_str() {
                "/api/cap/mint" => match serde_json::from_slice::<caprpc::MintReq>(&raw) {
                    Ok(req) => {
                        let grants: Vec<&str> = req.grants.iter().map(|s| s.as_str()).collect();
                        let cap = caprpc::mint(&caller, &req.endpoint, &grants);
                        (
                            200u16,
                            serde_json::to_vec(&caprpc::cap_stub(&cap)).unwrap_or_default(),
                        )
                    }
                    Err(e) => (
                        400u16,
                        format!("{{\"error\":\"bad mint request: {e}\"}}").into_bytes(),
                    ),
                },
                "/api/cap/promise" => match serde_json::from_slice::<caprpc::PromiseReq>(&raw) {
                    Ok(req) => {
                        let p = caprpc::new_promise(&req.endpoint);
                        (
                            200u16,
                            serde_json::json!({
                                "cap_id": p.id,
                                "endpoint": p.endpoint,
                            })
                            .to_string()
                            .into_bytes(),
                        )
                    }
                    Err(e) => (
                        400u16,
                        format!("{{\"error\":\"bad promise request: {e}\"}}").into_bytes(),
                    ),
                },
                "/api/cap/resolve" => match serde_json::from_slice::<caprpc::ResolveReq>(&raw) {
                    Ok(req) => {
                        let grants: Vec<&str> = req.grants.iter().map(|s| s.as_str()).collect();
                        let ok =
                            caprpc::resolve_promise(&caller, req.cap_id, &req.endpoint, &grants);
                        (
                            200u16,
                            serde_json::json!({ "ok": ok }).to_string().into_bytes(),
                        )
                    }
                    Err(e) => (
                        400u16,
                        format!("{{\"error\":\"bad resolve request: {e}\"}}").into_bytes(),
                    ),
                },
                "/api/cap/revoke" => match serde_json::from_slice::<caprpc::RevokeReq>(&raw) {
                    Ok(req) => {
                        let ok = caprpc::revoke(req.cap_id);
                        (
                            200u16,
                            serde_json::json!({ "ok": ok }).to_string().into_bytes(),
                        )
                    }
                    Err(e) => (
                        400u16,
                        format!("{{\"error\":\"bad revoke request: {e}\"}}").into_bytes(),
                    ),
                },
                "/api/cap/invoke" => {
                    let call: caprpc::CapCall = match serde_json::from_slice(&raw) {
                        Ok(c) => c,
                        Err(e) => {
                            return write_json(
                                session,
                                400u16,
                                &format!("{{\"error\":\"bad CapCall: {e}\"}}").into_bytes(),
                            )
                            .await
                            .map(|_| true);
                        }
                    };
                    let result = caprpc::invoke(&caller, &call);
                    let body = match serde_json::to_vec(&result) {
                        Ok(b) => b,
                        Err(_) => b"{\"error\":\"serialize\"}".to_vec(),
                    };
                    let status = if result.ok { 200u16 } else { 403u16 };
                    return write_json(session, status, &body).await.map(|_| true);
                }
                _ => (404u16, b"{\"error\":\"unknown cap route\"}".to_vec()),
            };
            return write_json(session, status, &body).await.map(|_| true);
        }

        // S6 (F6): the audit console short-circuits before proxying — GET /audit lists the
        // kernel audit trail, DELETE /audit clears it (an operator action).
        #[cfg(feature = "audit")]
        {
            let path = session.req_header().uri.path().to_string();
            let method = session.req_header().method.to_string();
            if audit::is_audit_path(&path) {
                let (status, body) = match method.as_str() {
                    "GET" => match audit::audit_list_body() {
                        Ok(b) => (200u16, b),
                        Err(e) => (500u16, e.into_bytes()),
                    },
                    "DELETE" => match audit::audit_clear_count() {
                        Ok(b) => (200u16, b),
                        Err(e) => (500u16, e.into_bytes()),
                    },
                    _ => (405u16, b"{\"error\":\"method not allowed\"}".to_vec()),
                };
                let mut header = ResponseHeader::build(status, None)?;
                header.insert_header("content-type", "application/json")?;
                header.insert_header("content-length", body.len().to_string())?;
                session
                    .write_response_header(Box::new(header), false)
                    .await?;
                session
                    .write_response_body(Some(bytes::Bytes::from(body)), true)
                    .await?;
                return Ok(true);
            }
        }

        // S6 (F6): the gatekeeper console short-circuits before proxying. The operator
        // reviews pending mediated side effects and resolves them (approve applies the
        // simulated outcome; reject discards it). These routes are operator-only: they
        // reach the embedded kernel directly, never the (untrusted) module backend.
        #[cfg(feature = "audit")]
        {
            let path = session.req_header().uri.path().to_string();
            let method = session.req_header().method.to_string();
            if gate::is_gate_path(&path) {
                #[derive(serde::Deserialize)]
                struct HandleRequest {
                    handle: i64,
                }
                let (status, body_bytes): (u16, Vec<u8>) = match method.as_str() {
                    "GET" if path == "/api/gate/pending" => match gate::pending_list_body() {
                        Ok(b) => (200u16, b),
                        Err(e) => (500u16, e.into_bytes()),
                    },
                    "POST" if path == "/api/gate/approve" || path == "/api/gate/reject" => {
                        let raw = match read_body(session).await {
                            Ok(b) => b,
                            Err(_) => {
                                return write_json(
                                    session,
                                    400u16,
                                    b"{\"error\":\"body too large\"}",
                                )
                                .await
                                .map(|_| true);
                            }
                        };
                        let req = match serde_json::from_slice::<HandleRequest>(&raw) {
                            Ok(r) => r,
                            Err(_) => {
                                return write_json(
                                    session,
                                    400u16,
                                    b"{\"error\":\"expects {\\\"handle\\\": N}\"}",
                                )
                                .await
                                .map(|_| true);
                            }
                        };
                        let dispatched = if path == "/api/gate/approve" {
                            gate::approve_body(req.handle)
                        } else {
                            gate::reject_body(req.handle)
                        };
                        match dispatched {
                            Ok(b) => (200u16, b),
                            Err(e) => (500u16, e.into_bytes()),
                        }
                    }
                    _ => (405u16, b"{\"error\":\"method not allowed\"}".to_vec()),
                };
                write_json(session, status, &body_bytes).await?;
                return Ok(true);
            }
        }

        // S22 (F21): the admin console — soft/hard config separation. Admin capability
        // is minted once at session start (env `UNFER_ADMIN_PRINCIPAL`); PATCH of the
        // soft config is refused for non-admin principals (403) and for hard keys
        // (grants/auth/storage/backend, 400). Never proxies to the module backend.
        #[cfg(feature = "audit")]
        {
            let path = session.req_header().uri.path().to_string();
            let method = session.req_header().method.to_string();
            if admin::is_admin_path(&path) {
                let principal = session
                    .req_header()
                    .headers
                    .get("x-principal")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let (status, body): (u16, Vec<u8>) = match method.as_str() {
                    "GET" if path == "/admin/status" => admin::status_body(&principal),
                    "PATCH" if path == "/admin/config" => match read_body(session).await {
                        Ok(b) => admin::patch_body(&principal, &b),
                        Err(_) => (400u16, b"{\"error\":\"body too large\"}".to_vec()),
                    },
                    _ => (405u16, b"{\"error\":\"method not allowed\"}".to_vec()),
                };
                write_json(session, status, &body).await?;
                return Ok(true);
            }
        }

        // S20 (F19): the blueprint content plane — POST /api/blueprint/import seals a
        // verified `.cell` archive into the kernel registry and seeds the /cell/<cid>
        // content store (so a published blueprint is immediately content-resolvable).
        #[cfg(feature = "audit")]
        {
            let path = session.req_header().uri.path().to_string();
            let method = session.req_header().method.to_string();
            if blueprint::is_blueprint_path(&path) {
                #[derive(serde::Deserialize)]
                struct ImportRequest {
                    #[serde(default)]
                    cell_hex: String,
                }
                let (status, body_bytes): (u16, Vec<u8>) = match method.as_str() {
                    "POST" if path == "/api/blueprint/import" => {
                        let raw = match read_body(session).await {
                            Ok(b) => b,
                            Err(_) => {
                                return write_json(
                                    session,
                                    400u16,
                                    b"{\"error\":\"body too large\"}",
                                )
                                .await
                                .map(|_| true);
                            }
                        };
                        let req = match serde_json::from_slice::<ImportRequest>(&raw) {
                            Ok(r) => r,
                            Err(_) => {
                                return write_json(
                                    session,
                                    400u16,
                                    b"{\"error\":\"expects {\\\"cell_hex\\\": \\\"...\\\"}\"}",
                                )
                                .await
                                .map(|_| true);
                            }
                        };
                        match blueprint::from_hex(&req.cell_hex)
                            .and_then(|cell| blueprint::import_record(&cell))
                        {
                            Ok(b) => (200u16, b),
                            Err(e) => (400u16, e.into_bytes()),
                        }
                    }
                    _ => (405u16, b"{\"error\":\"method not allowed\"}".to_vec()),
                };
                write_json(session, status, &body_bytes).await?;
                return Ok(true);
            }
        }

        // S15 (F14): actively shape-checked content reads — GET /cell/<cid> returns the
        // stored cell metadata or a resolved 404; malformed CIDs get 400 (never guess).
        if session.req_header().method == "GET"
            && session.req_header().uri.path().starts_with("/cell/")
        {
            let path = session.req_header().uri.path().to_string();
            let (status, body) = {
                let store = cell_store().lock().unwrap_or_else(|e| e.into_inner());
                cells::resolve_cell(&store, &path)
            };
            let mut header = ResponseHeader::build(status, None)?;
            header.insert_header("content-type", "application/json")?;
            header.insert_header("content-length", body.len().to_string())?;
            session
                .write_response_header(Box::new(header), false)
                .await?;
            session
                .write_response_body(Some(bytes::Bytes::from(body)), true)
                .await?;
            return Ok(true);
        }

        // Read the request body (bounded to 1 MiB).
        let body = match read_body(session).await {
            Ok(b) => b,
            Err(_) => {
                edge_metrics().record("??", false, 0);
                let rejection = filter::Rejection::BadJson("request body too large".to_string());
                send_rejection(session, "unknown", &rejection).await?;
                return Ok(true);
            }
        };

        let start = Instant::now();
        match filter::validate_request(&body) {
            Ok(req) => {
                edge_metrics().record(&req.op, true, start.elapsed().as_micros() as u64);
                Ok(false) // pass through to backend
            }
            Err(rejection) => {
                let op = match &rejection {
                    filter::Rejection::BadJson(_) => "??",
                    filter::Rejection::OpDenied { op } => op.as_str(),
                };
                edge_metrics().record(op, false, start.elapsed().as_micros() as u64);
                send_rejection(session, "unknown", &rejection).await?;
                Ok(true)
            }
        }
    }

    /// Strip `content-length` — data-masking (P11.22) may change the body's
    /// byte length (e.g. `"sk-live-abc123"` → `"***REDACTED***"`), so the
    /// original upstream length no longer applies. Pingora falls back to
    /// chunked/close-delimited framing for the downstream response.
    async fn upstream_response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut GatewayCtx,
    ) -> pingora_core::Result<()>
    where
        GatewayCtx: Send + Sync,
    {
        upstream_response.remove_header("content-length");
        Ok(())
    }

    /// Buffer upstream response body chunks (data-masking needs the whole
    /// JSON envelope; see [`GatewayCtx::upstream_body`]).
    fn upstream_response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<bytes::Bytes>,
        end_of_stream: bool,
        ctx: &mut GatewayCtx,
    ) -> pingora_core::Result<Option<std::time::Duration>> {
        if let Some(chunk) = body.take() {
            ctx.upstream_body.extend_from_slice(&chunk);
        }
        if end_of_stream {
            // Withhold the buffered body from the streaming path; the masked
            // version is emitted in `response_body_filter` below.
            *body = None;
        }
        Ok(None)
    }

    /// Emit the data-masked response body once the full upstream body has
    /// been buffered (P11.22 data-masking/secret-inject protection).
    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<bytes::Bytes>,
        end_of_stream: bool,
        ctx: &mut GatewayCtx,
    ) -> pingora_core::Result<Option<std::time::Duration>>
    where
        GatewayCtx: Send + Sync,
    {
        if end_of_stream && !ctx.upstream_body.is_empty() {
            let masked = mask::mask_body(&ctx.upstream_body);
            *body = Some(bytes::Bytes::from(masked));
        } else {
            *body = None;
        }
        Ok(None)
    }
}

/// Read the full request body up to 1 MiB.
async fn read_body(session: &mut Session) -> Result<Vec<u8>, String> {
    const MAX_BODY: usize = 1 << 20; // 1 MiB
    let mut buf = Vec::new();
    while let Some(chunk) = session
        .read_request_body()
        .await
        .map_err(|e| e.to_string())?
    {
        buf.extend_from_slice(&chunk);
        if buf.len() > MAX_BODY {
            return Err(format!("request body exceeds {MAX_BODY} bytes"));
        }
    }
    Ok(buf)
}

/// Write a JSON maybe-short-circuit response (used by the audit/gate consoles).
#[cfg(feature = "audit")]
async fn write_json(session: &mut Session, status: u16, body: &[u8]) -> pingora_core::Result<()> {
    let mut header = ResponseHeader::build(status, None)?;
    header.insert_header("content-type", "application/json")?;
    header.insert_header("content-length", body.len().to_string())?;
    session
        .write_response_header(Box::new(header), false)
        .await?;
    session
        .write_response_body(Some(bytes::Bytes::from(body.to_vec())), true)
        .await?;
    Ok(())
}

/// Write a JSON rejection response and signal Pingora to stop forwarding.
async fn send_rejection(
    session: &mut Session,
    id: &str,
    rejection: &filter::Rejection,
) -> pingora_core::Result<()> {
    let resp = rejection.to_response(id);
    let body = serde_json::to_vec(&resp).expect("AgentResponse serializes");
    let mut header = ResponseHeader::build(400u16, None)?;
    header.insert_header("content-type", "application/json")?;
    header.insert_header("content-length", body.len().to_string())?;
    session
        .write_response_header(Box::new(header), false)
        .await?;
    session
        .write_response_body(Some(bytes::Bytes::from(body)), true)
        .await?;
    Ok(())
}

fn main() {
    tracing_subscriber::fmt::init();

    // Simple argv parsing (no external clap dep to keep the crate minimal).
    let args: Vec<String> = std::env::args().collect();
    let listen = args
        .windows(2)
        .find(|w| w[0] == "--listen")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "0.0.0.0:3000".to_string());
    let backend = args
        .windows(2)
        .find(|w| w[0] == "--backend")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("UNFER_BACKEND").ok())
        .unwrap_or_else(|| "127.0.0.1:3001".to_string());

    let conf = Arc::new(GatewayConf {
        backend_addr: backend.clone(),
    });

    let mut ops: Vec<&str> = filter::allowed_ops().into_iter().collect();
    ops.sort_unstable();
    info!("unfer_edge: listen={listen} → backend={backend}, allowed ops = {ops:?}");

    let mut server = Server::new(None).expect("Pingora server init");
    server.bootstrap();

    let gateway = UnferGateway { conf };
    let mut proxy = http_proxy_service(&server.configuration, gateway);
    proxy.add_tcp(&listen);
    server.add_service(proxy);
    server.run_forever();
}
