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

mod filter;
mod mask;

use std::sync::Arc;

use async_trait::async_trait;
use pingora_core::prelude::*;
use pingora_http::ResponseHeader;
use pingora_proxy::{http_proxy_service, ProxyHttp, Session};
use tracing::info;

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
        // Read the request body (bounded to 1 MiB).
        let body = match read_body(session).await {
            Ok(b) => b,
            Err(e) => {
                send_rejection(session, "unknown", &filter::Rejection::BadJson(e)).await?;
                return Ok(true);
            }
        };

        match filter::validate_request(&body) {
            Ok(_req) => Ok(false), // pass through to backend
            Err(rejection) => {
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
