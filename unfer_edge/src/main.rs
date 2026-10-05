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
pub mod config;
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
    /// The effective startup configuration and which layer supplied each key, so
    /// `GET /version` can answer "why is it bound to that port" instead of the
    /// operator having to trust a precedence order. See `config`.
    startup: config::Config,
    provenance: config::Provenance,
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
        // X6: the self-service routes are decided by `route()`, which is a pure
        // function and is unit-tested. The Session I/O below is the untestable
        // remainder; keeping it thin means the part that can be wrong is covered.
        let path = session.req_header().uri.path().to_string();
        let query = session.req_header().uri.query().map(str::to_string);
        let decision = route(
            session.req_header().method.as_str(),
            &path,
            query.as_deref(),
        );

        // S13 (F12): per-op metrics before any forwarding. GET /metrics → JSON;
        // GET /metrics?format=prometheus → text exposition. Final.
        if let Some(EarlyRoute::Metrics { prometheus }) = decision {
            let body = if prometheus {
                edge_metrics()
                    .to_prometheus(&filter::allowed_ops_vec(), &[])
                    .into_bytes()
            } else {
                serde_json::to_vec(&edge_metrics().to_json(&filter::allowed_ops_vec(), &[]))
                    .unwrap_or_else(|_| b"".to_vec())
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

        // X6: liveness and build identity.
        if let Some(EarlyRoute::Healthz) = decision {
            return write_json(
                session,
                200,
                &serde_json::to_vec(&serde_json::json!({"status": "ok"})).unwrap_or_default(),
            )
            .await
            .map(|_| true);
        }
        if let Some(EarlyRoute::Version) = decision {
            return write_json(
                session,
                200,
                &version_json(Some(&self.conf.startup), Some(&self.conf.provenance)),
            )
            .await
            .map(|_| true);
        }

        // A known path reached with the wrong method.
        if decision.is_none() {
            let mut header = ResponseHeader::build(405u16, None)?;
            header.insert_header("allow", "GET")?;
            session
                .write_response_header(Box::new(header), false)
                .await?;
            session.write_response_body(None, true).await?;
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

/// What the gateway answers itself, before anything is forwarded.
///
/// X6: `/healthz` and `/version` for operators, matching the endpoints
/// dynamic-arctic exposes, so a deployment can probe either service the same way.
///
/// Extracted as a pure function on purpose. `request_filter` takes a pingora
/// `Session`, which cannot be constructed without a real connection, so testing
/// the handler means testing nothing. The decision table -- which path, which
/// method, which status -- is where mistakes live, and that is pure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EarlyRoute {
    /// `GET /metrics`, optionally in Prometheus text format.
    Metrics { prometheus: bool },
    /// `GET /healthz`
    Healthz,
    /// `GET /version`
    Version,
    /// `POST /api/cap/...`; the remainder of the path.
    CapInvoke(String),
    /// Anything else: forward upstream.
    Proxy,
}

/// Decide how to handle a request, or `None` if the method is wrong for a path
/// that does exist (the caller then answers 405).
pub fn route(method: &str, path: &str, query: Option<&str>) -> Option<EarlyRoute> {
    match path {
        "/metrics" if method == "GET" => Some(EarlyRoute::Metrics {
            prometheus: query.is_some_and(|q| q.contains("format=prometheus")),
        }),
        "/healthz" if method == "GET" => Some(EarlyRoute::Healthz),
        "/version" if method == "GET" => Some(EarlyRoute::Version),
        p if p.starts_with("/api/cap/") && method == "POST" => {
            Some(EarlyRoute::CapInvoke(p["/api/cap/".len()..].to_string()))
        }
        // A known self-service path reached with the wrong method is a client
        // error, not something to forward. Forwarding `POST /healthz` upstream
        // would return a confusing upstream 404 instead of a 405.
        "/metrics" | "/healthz" | "/version" => None,
        _ => Some(EarlyRoute::Proxy),
    }
}

/// The build identity this gateway reports, plus the effective startup
/// configuration and the layer each key came from.
///
/// The provenance map is the point. `GET /healthz` says the process is alive and
/// `GET /version` says which build it is; neither says which of four configuration
/// layers actually won, which is the question an operator has when a port is not
/// what they expected. It is the same shape dynamic-arctic's `/version` reports.
pub fn version_json(
    startup: Option<&config::Config>,
    provenance: Option<&config::Provenance>,
) -> Vec<u8> {
    let mut v = serde_json::json!({
        "service": "unfer_edge",
        "version": env!("CARGO_PKG_VERSION"),
    });
    if let (Some(cfg), Some(prov)) = (startup, provenance) {
        v["config"] = cfg.to_json(prov);
    }
    serde_json::to_vec(&v).unwrap_or_else(|_| b"{}".to_vec())
}

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

    let argv: Vec<String> = std::env::args().collect();
    let invocation = match config::parse_args(&argv) {
        Ok(i) => i,
        Err(msg) => {
            // `--help` arrives here as Err on purpose: usage plus a non-zero exit.
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let (file, flags) = match invocation {
        config::Invocation::Init { file } => {
            // `init` needs no configuration and does not start a server. Writes a
            // commented starter file where `--config` (in either position) or
            // UNFER_EDGE_CONFIG says, else the default path.
            let path = file.unwrap_or_else(|| match std::env::var("UNFER_EDGE_CONFIG") {
                Ok(v) => std::path::PathBuf::from(v),
                Err(_) => std::path::PathBuf::from("unfer_edge.json"),
            });
            let text = config::starter_config(&[]);
            if let Err(e) = std::fs::write(&path, &text) {
                eprintln!("unfer_edge init: cannot write {}: {e}", path.display());
                std::process::exit(1);
            }
            println!("wrote {}", path.display());
            println!("precedence: defaults < file < env < flags");
            println!("next: unfer_edge --config {}", path.display());
            return;
        }
        config::Invocation::Serve { file, flags } => (file, flags),
    };

    let parsed_file = match config::load_file(file.as_deref()) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("unfer_edge: {msg}");
            std::process::exit(1);
        }
    };

    let env = |k: &str| std::env::var(k).ok();
    let (cfg, provenance) = config::Config::resolve(parsed_file.as_ref(), &env, &flags);
    let listen = cfg.listen.clone();
    let backend = cfg.backend.clone();

    let conf = Arc::new(GatewayConf {
        backend_addr: backend.clone(),
        startup: cfg,
        provenance,
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

#[cfg(test)]
mod early_route_tests {
    use super::*;

    #[test]
    fn healthz_and_version_are_get_only() {
        assert_eq!(route("GET", "/healthz", None), Some(EarlyRoute::Healthz));
        assert_eq!(route("GET", "/version", None), Some(EarlyRoute::Version));
        // Wrong method: None means "405", not "forward upstream".
        for m in ["POST", "PUT", "DELETE", "HEAD"] {
            assert_eq!(route(m, "/healthz", None), None, "{m} /healthz");
            assert_eq!(route(m, "/version", None), None, "{m} /version");
        }
    }

    #[test]
    fn metrics_keeps_its_prometheus_format_switch() {
        assert_eq!(
            route("GET", "/metrics", None),
            Some(EarlyRoute::Metrics { prometheus: false })
        );
        assert_eq!(
            route("GET", "/metrics", Some("format=prometheus")),
            Some(EarlyRoute::Metrics { prometheus: true })
        );
        assert_eq!(route("POST", "/metrics", None), None);
    }

    #[test]
    fn capability_invoke_yields_the_remainder_and_requires_post() {
        assert_eq!(
            route("POST", "/api/cap/mint", None),
            Some(EarlyRoute::CapInvoke("mint".to_string()))
        );
        assert_eq!(
            route("POST", "/api/cap/a/b/c", None),
            Some(EarlyRoute::CapInvoke("a/b/c".to_string()))
        );
        // GET must not execute a capability: that is the whole point of minting
        // one. A GET falls through to the proxy instead of being dispatched.
        assert_eq!(route("GET", "/api/cap/mint", None), Some(EarlyRoute::Proxy));
    }

    #[test]
    fn everything_else_is_forwarded() {
        for p in ["/", "/api/v1/foo", "/healthz/extra", "/api/cap"] {
            assert_eq!(
                route("POST", p, None),
                Some(EarlyRoute::Proxy),
                "{p} should forward"
            );
        }
    }

    #[test]
    fn version_json_names_the_service_and_a_version() {
        let v: serde_json::Value = serde_json::from_slice(&version_json(None, None)).expect("json");
        assert_eq!(v["service"], "unfer_edge");
        assert!(
            !v["version"].as_str().unwrap_or_default().is_empty(),
            "a version endpoint that reports no version is worse than none"
        );
        // With no configuration supplied the endpoint must not invent one. An
        // operator reading `"config": {"listen": null}` learns nothing and may
        // believe the value is unset when it is merely unreported.
        assert!(
            v.get("config").is_none(),
            "config must be omitted rather than reported empty"
        );
    }

    #[test]
    fn version_json_reports_which_configuration_layer_won() {
        // The X6 addition: `/healthz` says alive, `/version` says which build --
        // but neither says which of defaults/file/env/flag actually supplied the
        // listen address, which is the question an operator has when the port is
        // wrong. Resolved through the real resolver so the test cannot drift from
        // the precedence order it claims to report.
        let file = serde_json::json!({"listen": "10.0.0.1:8080"});
        let env = |k: &str| (k == "UNFER_BACKEND").then(|| "10.0.0.9:9000".to_string());
        let flags = config::Flags::default();
        let (cfg, prov) = config::Config::resolve(Some(&file), &env, &flags);

        let v: serde_json::Value =
            serde_json::from_slice(&version_json(Some(&cfg), Some(&prov))).expect("json");
        assert_eq!(v["service"], "unfer_edge");
        assert_eq!(v["config"]["listen"], "10.0.0.1:8080");
        assert_eq!(v["config"]["backend"], "10.0.0.9:9000");
        assert_eq!(v["config"]["layers"]["listen"], "file");
        assert_eq!(v["config"]["layers"]["backend"], "env");
    }
}
