//! Routing and rendering. One server-rendered page, no JavaScript: the whole
//! service is a form with a submit button, which is what "smallest version that
//! works" means here. Verification is synchronous (it is two subprocesses);
//! publishing is queued (`crate::publisher`) and observed at `/status`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::audit::Audit;
use crate::checker::{self, CheckerConfig};
use crate::graph::Graph;
use crate::http::{self, Request};
use logos::formalize::embed::{percent_encode, Embedder};
use logos::formalize::hybrid::{rrf_fuse, RRF_K};
use logos::formalize::index::RecomputeIndex;
use crate::publisher::{Publisher, QueueError};
use crate::store::{
    self, Index, STATE_PUBLISH_FAILED, STATE_PUBLISHED, STATE_QUEUED, STATE_VERIFIED, StoreConfig,
    Zenodo, slug,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub store: StoreConfig,
    pub checker: CheckerConfig,
    /// Requests per client per `rate_window`.
    pub rate_max: usize,
    pub rate_window: Duration,
    /// Where per-request verification work directories are created.
    pub work_root: PathBuf,
    /// Per-field cap on submitted proof text.
    pub max_proof_bytes: usize,
    /// Append-only audit trail (`events.jsonl`).
    pub audit: Arc<Audit>,
    /// The publish queue. Empty capability if Zenodo is unconfigured — the
    /// queue exists either way, publishing simply is refused earlier.
    pub publisher: Arc<Publisher>,
    /// Rolling per-client cap on *new* proofs per day.
    pub daily: Arc<DailyCap>,
    /// The address the listener was asked to bind. Part of the service's
    /// svcmux bundle: the supervisor's manifest records the same string, so
    /// a port change rotates this service's digest and nobody else's.
    pub listen: String,
    /// The optional search arms (semantic embedder, xerj node) and the
    /// shared corpus-index cache. `None`/empty = substring search only.
    pub search: SearchConfig,
}

impl Default for Config {
    fn default() -> Self {
        let store = StoreConfig {
            state_dir: PathBuf::from("state"),
            zenodo: Default::default(),
        };
        let audit = Arc::new(Audit::new(&store.state_dir));
        let publisher = Arc::new(Publisher::spawn(store.clone(), audit.clone()));
        Self {
            store,
            checker: CheckerConfig::default(),
            rate_max: 10,
            rate_window: Duration::from_secs(60),
            work_root: std::env::temp_dir().join("mm0web"),
            max_proof_bytes: 4 * 1024 * 1024,
            audit,
            publisher,
            daily: Arc::new(DailyCap::new(DAILY_DEFAULT, Duration::from_secs(24 * 3600))),
            listen: "127.0.0.1:8080".to_string(),
            search: SearchConfig::default(),
        }
    }
}

/// New proofs one client may publish per rolling 24h. `brief-ideas`' rule is
/// "5 ideas a day" per author; with open submission there is no account to
/// attach it to, so it attaches to the client address, and it counts only
/// *first-time* content — re-verifying or revising the same proof is free,
/// because iteration is the workflow and hoarding fresh proofs is not.
pub const DAILY_DEFAULT: usize = 5;

#[derive(Debug)]
pub struct DailyCap {
    max: usize,
    window: Duration,
    hits: Mutex<HashMap<IpAddr, Vec<Instant>>>,
}

impl DailyCap {
    pub fn new(max: usize, window: Duration) -> Self {
        Self {
            max,
            window,
            hits: Mutex::new(HashMap::new()),
        }
    }

    /// Charge one new proof against the client, if there is room left.
    /// `max == 0` means the cap is off.
    pub fn allow(&self, ip: IpAddr) -> bool {
        if self.max == 0 {
            return true;
        }
        let now = Instant::now();
        let mut hits = self.hits.lock().expect("daily cap poisoned");
        let entry = hits.entry(ip).or_default();
        entry.retain(|t| now.duration_since(*t) < self.window);
        if entry.len() >= self.max {
            return false;
        }
        entry.push(now);
        true
    }

    pub fn max(&self) -> usize {
        self.max
    }
}

/// Sliding-window limiter, per client address. Open submission is the point;
/// an unbounded one is not.
pub struct RateLimiter {
    window: Duration,
    max: usize,
    hits: Mutex<HashMap<IpAddr, Vec<Instant>>>,
}

impl RateLimiter {
    pub fn new(max: usize, window: Duration) -> Self {
        Self {
            window,
            max,
            hits: Mutex::new(HashMap::new()),
        }
    }

    pub fn allow(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let mut hits = self.hits.lock().expect("rate limiter poisoned");
        let entry = hits.entry(ip).or_default();
        entry.retain(|t| now.duration_since(*t) < self.window);
        if entry.len() >= self.max {
            return false;
        }
        entry.push(now);
        true
    }
}

pub type Response = (u16, String, Vec<u8>);

fn html(status: u16, body: String) -> Response {
    (
        status,
        "text/html; charset=utf-8".to_string(),
        body.into_bytes(),
    )
}

fn json(status: u16, body: serde_json::Value) -> Response {
    (
        status,
        "application/json".to_string(),
        body.to_string().into_bytes(),
    )
}

const PAGE: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>mm0 proofs</title></head><body>
<h1>mm0 proofs</h1>
<p>Submit an mm0 development (<code>.mm0</code> spec plus <code>.mm1</code> proofs).
Verification runs <code>mm0-rs compile</code> then <code>mm0-c</code> — the same
two-stage root of trust <code>australVM/scripts/lean2mm0 verify</code> uses, where a
warning is a proof hole. Publishing stores the verified artifact in a Zenodo
project deposit (snapshot + incremental deltas).</p>
<p class="status">STATUS</p>
<form method="post" action="/publish">
  <p><label>Project <input name="project" size="24" value="DEFAULT_PROJECT" required></label></p>
  <p><label>Title <input name="title" size="48" required></label></p>
  <p><label>mm0 spec<br><textarea name="mm0" rows="8" cols="80" required></textarea></label></p>
  <p><label>mm1 proofs<br><textarea name="mm1" rows="8" cols="80"></textarea></label></p>
  <p><button name="action" value="verify" type="submit">Verify only</button>
     <button name="action" value="publish" type="submit">Verify and publish</button></p>
</form>
<p>Published proofs: PROJECT_LINKS</p>
<p><a href="/search">search</a> · <a href="/stats">stats</a> · <a href="/events">audit log</a></p>
</body></html>
"#;

/// The whole service, as one function: request in, response out. Everything the
/// tests need is reachable from here without a socket.
pub fn handle(req: &Request, cfg: &Config, limiter: &RateLimiter) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => html(200, render_page(cfg)),
        ("GET", "/healthz") => healthz(cfg),
        ("GET", "/manifest") => get_manifest(req, cfg),
        ("GET", "/proof") => get_proof(req, cfg),
        ("GET", "/proofs") => list_proofs(req, cfg),
        ("GET", "/status") => get_status(req, cfg),
        ("GET", "/attestation") => attestation(req, cfg),
        ("GET", "/events") => get_events(req, cfg),
        ("GET", "/graph") => get_graph(req, cfg, false),
        ("GET", "/api/v1/graph") => get_graph(req, cfg, true),
        ("GET", "/search") => get_search(req, cfg, false),
        ("GET", "/api/v1/search-bulk") => get_search_bulk(cfg),
        ("GET", "/api/v1/search") => get_search(req, cfg, true),
        ("GET", "/stats") => get_stats(cfg, false),
        ("GET", "/api/v1/stats") => get_stats(cfg, true),
        ("POST", "/verify") | ("POST", "/api/v1/verify") => {
            verify_or_publish(req, cfg, limiter, false)
        }
        ("POST", "/publish") | ("POST", "/api/v1/publish") => {
            verify_or_publish(req, cfg, limiter, true)
        }
        (_, "/") => (
            405,
            "text/plain".to_string(),
            b"GET / (or POST to /verify or /publish)".to_vec(),
        ),
        _ => (404, "text/plain".to_string(), b"not found\n".to_vec()),
    }
}

/// This service's own measurement, nonce-bound: the outside-in view
/// (`svcmux`'s manifest) checked against the inside-out one. The canonical
/// `svcmux-bundle-v1` form lives in `unfer_measure`, which pins the same
/// vectors the supervisor does — so the two measurements must agree, and a
/// disagreement means something changed underneath one of them.
fn attestation(req: &Request, cfg: &Config) -> Response {
    const MAX_NONCE: usize = 512;
    let nonce = match req.query.get("nonce") {
        Some(n) if !n.is_empty() && n.len() <= MAX_NONCE => n.clone(),
        _ => {
            return (
                400,
                "text/plain; charset=utf-8".to_string(),
                format!("attestation requires ?nonce=<challenge> (1..{MAX_NONCE} bytes)\n")
                    .into_bytes(),
            )
        }
    };
    let state = cfg.store.state_dir.to_string_lossy().into_owned();
    let identity = unfer_measure::Identity {
        id: "mm0web",
        state_root: &state,
        listen: &cfg.listen,
        env_prefixes: &["MM0WEB_", "ZENODO_"],
    };
    let bundle = match unfer_measure::Bundle::for_service(&identity) {
        Ok(b) => b,
        Err(e) => {
            return (
                500,
                "text/plain; charset=utf-8".to_string(),
                format!("measurement failed: {e}\n").into_bytes(),
            )
        }
    };
    let binding = unfer_measure::attestation_binding(&bundle.bundle, &nonce);
    let mut body = serde_json::to_value(&bundle).unwrap_or(serde_json::Value::Null);
    body["nonce"] = serde_json::json!(nonce);
    body["binding"] = serde_json::json!(binding);
    (200, "application/json".to_string(), body.to_string().into_bytes())
}

fn healthz(cfg: &Config) -> Response {
    let checker_ok = which(&cfg.checker.mm0_rs).is_some() && which(&cfg.checker.mm0_c).is_some();
    let zenodo = Zenodo::new(cfg.store.zenodo.clone(), cfg.store.state_dir.clone());
    let body = format!(
        "checker: {}\nmm0-rs: {}\nmm0-c: {}\nzenodo: {} ({})\n",
        if checker_ok {
            "ready"
        } else {
            "missing toolchain"
        },
        cfg.checker.mm0_rs.display(),
        cfg.checker.mm0_c.display(),
        if zenodo.configured() {
            "configured"
        } else {
            "unconfigured"
        },
        cfg.store.zenodo.base_url,
    );
    // 503 when the service cannot do its one job. `AX`-style health semantics:
    // exit non-zero means unhealthy, and a healthcheck that lies is worse than
    // none.
    if checker_ok {
        (
            200,
            "text/plain; charset=utf-8".to_string(),
            body.into_bytes(),
        )
    } else {
        (
            503,
            "text/plain; charset=utf-8".to_string(),
            body.into_bytes(),
        )
    }
}

fn which(bin: &std::path::Path) -> Option<PathBuf> {
    if bin.is_absolute() || bin.components().count() > 1 {
        return std::fs::metadata(bin).ok().map(|_| bin.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(bin))
        .find(|p| std::fs::metadata(p).is_ok())
}

fn verify_or_publish(
    req: &Request,
    cfg: &Config,
    limiter: &RateLimiter,
    publish: bool,
) -> Response {
    if !limiter.allow(req.peer) {
        return (
            429,
            "text/plain; charset=utf-8".to_string(),
            format!(
                "rate limited: at most {} requests per {}s\n",
                cfg.rate_max,
                cfg.rate_window.as_secs()
            )
            .into_bytes(),
        );
    }
    // A form post is either an explicit action=… or the URL decides.
    let action = req.field("action").unwrap_or_default();
    let want_publish = publish || action == "publish";
    let project = match req.field("project") {
        Some(p) if !p.trim().is_empty() => p.trim().to_string(),
        _ => {
            return bad_request("project is required");
        }
    };
    let title = req
        .field("title")
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| project.clone());
    let mm0 = req.field("mm0").unwrap_or_default();
    let mm1 = req.field("mm1").unwrap_or_default();
    if mm0.trim().is_empty() {
        return bad_request("mm0 spec is required");
    }
    if mm0.len() > cfg.max_proof_bytes || mm1.len() > cfg.max_proof_bytes {
        return bad_request("submission exceeds the per-field size limit");
    }

    let workdir = cfg
        .work_root
        .join(format!("{}-{}", slug(&project), crate::store::now_unix()));
    let outcome = checker::check(&cfg.checker, &mm0, &mm1, &workdir);
    let _ = std::fs::remove_dir_all(&workdir);
    let id = store::proof_id(&title, &mm0, &mm1);
    // The graph is a *view*, built here so a result page can embed it. The
    // verdict below comes from the checkers; nothing here can reject.
    let graph = Graph::build(&mm0, &mm1);
    let graph = if outcome.accepted {
        // An accepted submission's undeclared names are information, not
        // defects — the checkers already judged it.
        graph.accepted()
    } else {
        graph
    };
    let failed = failed_nodes(&graph, &outcome.log);
    let graph_ref: Option<(&Graph, &HashSet<String>)> = Some((&graph, &failed));

    if !outcome.accepted {
        let _ = cfg.audit.record(
            "rejected",
            &project,
            &id,
            json!({"stage": outcome.stage, "title": title}),
        );
        let text = format!(
            "rejected at {}, stage={}\n\n{}",
            cfg.checker.mm0_rs.display(),
            outcome.stage,
            outcome.log
        );
        return if req.wants_json() {
            json(
                400,
                json!({"accepted": false, "stage": outcome.stage, "log": outcome.log}),
            )
        } else {
            html(400, render_result("Rejected", &text, None, graph_ref))
        };
    }
    let _ = cfg.audit.record(
        "verified",
        &project,
        &id,
        json!({"title": title, "publish": want_publish}),
    );

    if !want_publish {
        return if req.wants_json() {
            json(
                200,
                json!({"accepted": true, "stage": outcome.stage, "log": outcome.log}),
            )
        } else {
            html(200, render_result("Accepted", &outcome.log, None, graph_ref))
        };
    }

    // Publishing: verified first, always — now handed to the worker rather
    // than done in this thread.
    let zenodo = Zenodo::new(cfg.store.zenodo.clone(), cfg.store.state_dir.clone());
    let mut index = Index::load(&cfg.store.state_dir, &project);
    let is_new = !index.proofs.contains_key(&id);

    // Idempotence: the id is content-addressed, so an id we already published
    // is this exact proof — saying so beats burning a queue slot (and a daily
    // slot) to re-archive identical bytes.
    if !is_new
        && let Some(p) = index.proofs.get(&id)
        && p.state == STATE_PUBLISHED
        && p.record.is_some()
    {
        let record = p.record.unwrap_or(0);
        let doi = p.doi.clone();
        let url = zenodo.record_url(record);
        let text = format!(
            "{id}\nalready published — record {record}, doi {}\nstatus: {}\n\n{}",
            doi.clone().unwrap_or_else(|| "(none yet)".to_string()),
            status_path(&project, &id),
            outcome.log
        );
        return if req.wants_json() {
            json(
                200,
                json!({"accepted": true, "queued": false, "already": true,
                    "proof": id, "record": record, "doi": doi,
                    "url": url, "status": status_path(&project, &id)}),
            )
        } else {
            html(
                200,
                render_result("Already published", &text, Some(&url), graph_ref),
            )
        };
    }

    if is_new && !cfg.daily.allow(req.peer) {
        let _ = cfg.audit.record(
            "rate-limited",
            &project,
            &id,
            json!({"limit": "daily", "max": cfg.daily.max()}),
        );
        let msg = format!(
            "daily cap: at most {} new proofs per client per 24h \
             (re-verifying or revising one you already published is free)\n",
            cfg.daily.max()
        );
        return if req.wants_json() {
            json(429, json!({"error": "daily-cap", "message": msg}))
        } else {
            html(429, render_result("Daily cap reached", &msg, None, None))
        };
    }

    index.upsert(&title, &mm0, &mm1);
    if !zenodo.configured() {
        // The verification stands and the proof is stored locally; say plainly
        // that the archive step did not happen.
        if let Err(e) = index.save(&cfg.store.state_dir) {
            return server_error(&format!("cannot save the project index: {e}"));
        }
        let text = format!(
            "{id}\nverified and stored locally, but ZENODO_API_KEY is not set, \
             so nothing was published\n\n{}",
            outcome.log
        );        return if req.wants_json() {
            json(
                502,
                json!({"accepted": true, "published": false, "proof": id,
                    // The verification stands; a client showing this record
                    // should still be able to show what the checkers said.
                    "log": outcome.log,
                    "error": "ZENODO_API_KEY is not set"}),
            )
        } else {
            html(
                502,
                render_result("Verified, not published", &text, None, graph_ref),
            )
        };
    }

    {
        let p = index.proofs.get_mut(&id).expect("just upserted");
        p.state = STATE_QUEUED.to_string();
    }
    if let Err(e) = index.save(&cfg.store.state_dir) {
        return server_error(&format!("cannot save the project index: {e}"));
    }

    match cfg.publisher.enqueue(&project, &id) {
        Ok(()) => {
            let _ = cfg
                .audit
                .record("queued", &project, &id, json!({"title": title}));
            let status = status_path(&project, &id);
            let text = format!(
                "{id}\nverified and queued for publishing — the deposit is built by \
                 a background worker, so this page does not wait for Zenodo.\n\
                 status: {status}\n\n{}",
                outcome.log
            );
            if req.wants_json() {
                json(
                    202,
                    json!({"accepted": true, "queued": true, "proof": id, "status": status,
                        "log": outcome.log}),
                )
            } else {
                html(202, render_queued(&project, &id, &text, cfg))
            }
        }
        Err(e) => {
            // Shed load without lying: the proof is verified and stored, the
            // push simply has not been accepted.
            {
                let p = index.proofs.get_mut(&id).expect("just upserted");
                p.state = if e == QueueError::Full {
                    STATE_VERIFIED.to_string()
                } else {
                    STATE_PUBLISH_FAILED.to_string()
                };
            }
            let _ = index.save(&cfg.store.state_dir);
            let _ = cfg
                .audit
                .record("queue-full", &project, &id, json!({"error": e.to_string()}));
            let text =
                format!("{id}\nverified and stored, but publishing could not be queued: {e}\n");
            if req.wants_json() {
                json(
                    503,
                    json!({"accepted": true, "queued": false, "proof": id,
                        "log": outcome.log, "error": e.to_string()}),
                )
            } else {
                html(
                    503,
                    render_result("Verified, queue unavailable", &text, None, graph_ref),
                )
            }
        }
    }
}

fn status_path(project: &str, id: &str) -> String {
    format!("/status?project={}&id={}", project, id)
}

fn bad_request(msg: &str) -> Response {
    (
        400,
        "text/plain; charset=utf-8".to_string(),
        format!("{msg}\n").into_bytes(),
    )
}

fn server_error(msg: &str) -> Response {
    (
        500,
        "text/plain; charset=utf-8".to_string(),
        format!("{msg}\n").into_bytes(),
    )
}

fn get_manifest(req: &Request, cfg: &Config) -> Response {
    let project = req.field("project").unwrap_or_default();
    let index = Index::load(&cfg.store.state_dir, &project);
    json(200, index.to_json())
}

fn get_proof(req: &Request, cfg: &Config) -> Response {
    let project = req.field("project").unwrap_or_default();
    let id = req.field("id").unwrap_or_default();
    let index = Index::load(&cfg.store.state_dir, &project);
    match index.proofs.get(&id) {
        Some(p) => {
            let zenodo = Zenodo::new(cfg.store.zenodo.clone(), cfg.store.state_dir.clone());
            json(
                200,
                json!({
                    "id": p.id, "title": p.title, "fingerprint": p.fingerprint,
                    "seq": p.seq, "verified": p.verified, "record": p.record,
                    "state": p.state, "doi": p.doi,
                    // The Zenodo record page, the same link `/status` renders:
                    // a client showing a proof wants both the DOI and where
                    // the archived bytes live.
                    "url": p.record.map(|r| zenodo.record_url(r)),
                    "status": status_path(&project, &p.id),
                    "mm0": p.mm0, "mm1": p.mm1,
                }),
            )
        }
        None => json(404, json!({"error": "no such proof", "id": id})),
    }
}

/// The browser's view of an asynchronous publish: a plain page that refreshes
/// itself while the job is queued and stops when it is not. This is the whole
/// client side of the queue — no JavaScript beyond a meta refresh.
fn get_status(req: &Request, cfg: &Config) -> Response {
    let project = req.field("project").unwrap_or_default();
    let id = req.field("id").unwrap_or_default();
    let index = Index::load(&cfg.store.state_dir, &project);
    let Some(p) = index.proofs.get(&id) else {
        return (
            404,
            "text/plain; charset=utf-8".to_string(),
            format!("no such proof: {id}\n").into_bytes(),
        );
    };
    let pending = p.state == STATE_QUEUED;
    let refresh = if pending {
        format!(
            "<meta http-equiv=\"refresh\" content=\"1; url={}\">",
            http::escape_html(&status_path(&project, &id))
        )
    } else {
        String::new()
    };
    let zenodo = Zenodo::new(cfg.store.zenodo.clone(), cfg.store.state_dir.clone());
    let record_link = p
        .record
        .map(|r| {
            let url = zenodo.record_url(r);
            format!(
                "<p>record: <a href=\"{}\">{}</a></p>",
                http::escape_html(&url),
                http::escape_html(&url)
            )
        })
        .unwrap_or_default();
    let doi = p
        .doi
        .as_ref()
        .map(|d| {
            format!(
                "<p>doi: <a href=\"https://doi.org/{}\">{}</a></p>",
                http::escape_html(d),
                http::escape_html(d)
            )
        })
        .unwrap_or_default();
    let events: Vec<String> = cfg
        .audit
        .read()
        .iter()
        .filter(|e| e["proof"] == id.as_str())
        .map(|e| {
            format!(
                "<li>{} — {}{}</li>",
                http::escape_html(e["action"].as_str().unwrap_or("?")),
                e["ts"].as_u64().unwrap_or(0),
                e.get("error")
                    .map(|x| format!(" — {}", http::escape_html(x.as_str().unwrap_or(""))))
                    .unwrap_or_default()
            )
        })
        .collect();
    let body = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">{refresh}\
         <title>{} — {}</title></head><body>\
         <h1>{}</h1><p>state: <strong>{}</strong></p>{record_link}{doi}\
         <p>project {} · proof {}</p><ul>{}</ul>\
         <p><a href=\"/\">back</a> · <a href=\"/events\">audit log</a></p></body></html>",
        http::escape_html(&p.title),
        http::escape_html(&p.state),
        http::escape_html(&p.title),
        http::escape_html(&p.state),
        http::escape_html(&project),
        http::escape_html(&id),
        events.join("")
    );
    html(if pending { 202 } else { 200 }, body)
}

/// The audit trail, as JSON — the `brief-ideas` admin/audit view, minimal.
fn get_events(req: &Request, cfg: &Config) -> Response {
    let project = req.field("project").unwrap_or_default();
    let proof = req.field("id").unwrap_or_default();
    let mut ev: Vec<serde_json::Value> = cfg
        .audit
        .read()
        .into_iter()
        .filter(|e| project.is_empty() || e["project"] == project.as_str())
        .filter(|e| proof.is_empty() || e["proof"] == proof.as_str())
        .collect();
    const TAIL: usize = 200;
    if ev.len() > TAIL {
        ev = ev.split_off(ev.len() - TAIL);
    }
    json(200, json!({"events": ev, "count": ev.len()}))
}

/// Every project directory that carries an index, in name order.
fn projects(cfg: &Config) -> Vec<String> {
    let mut projects: Vec<String> = std::fs::read_dir(&cfg.store.state_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().join("index.json").is_file())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    projects.sort();
    projects
}

/// Every stored proof, as `(project, proof)` — what search and stats read.
fn all_proofs(cfg: &Config) -> Vec<(String, store::Proof)> {
    let mut out = Vec::new();
    for p in projects(cfg) {
        let index = Index::load(&cfg.store.state_dir, &p);
        out.extend(index.proofs.into_values().map(|pr| (p.clone(), pr)));
    }
    out
}

/// One submission's dependency graph — the page (`/graph`) or the raw graph
/// (`/api/v1/graph`).
fn get_graph(req: &Request, cfg: &Config, as_json: bool) -> Response {
    let project = req.field("project").unwrap_or_default();
    let id = req.field("id").unwrap_or_default();
    let index = Index::load(&cfg.store.state_dir, &project);
    let Some(p) = index.proofs.get(&id) else {
        let msg = format!(
            "the graph needs ?project=…&id=… of a stored proof \n\
             (got project={project} id={id})\n"
        );
        return if as_json {
            json(
                404,
                json!({"error": "no such proof", "project": project, "id": id}),
            )
        } else {
            (404, "text/plain; charset=utf-8".to_string(), msg.into_bytes())
        };
    };
    let g = Graph::build(&p.mm0, &p.mm1).accepted();
    if as_json {
        return json(200, g.to_json());
    }
    let title = format!("{} — dependency graph", p.title);
    let note = format!(
        "<p>proof <a href=\"/proof?project={pr}&id={pid}\">{t}</a> · seq {} · \
         <a href=\"{st}\">status</a> · \
         <a href=\"/api/v1/graph?project={pr}&id={pid}\">graph as JSON</a></p>",
        p.seq,
        pr = http::escape_html(&project),
        pid = http::escape_html(&id),
        t = http::escape_html(&p.title),
        st = http::escape_html(&status_path(&project, &id)),
    );
    html(200, g.html(&title, &note, &HashSet::new()))
}

/// At most this many declarations come back from a search — a substring
/// query on a whole development matches a lot, and the page is for finding.
const SEARCH_LIMIT: usize = 50;

/// Upper bound on declarations the semantic arm will index. Above it the
/// arm reports itself *unavailable* rather than turning one query into a
/// marathon of `curl` batches; the substring arm always answers.
const SEARCH_INDEX_MAX: usize = 50_000;

/// How many declarations one `/embeddings` request carries: one `curl`
/// process and one server forward pass per batch (LEANN's dynamic batching,
/// at the transport level — and what keeps a first build from becoming one
/// enormous request an endpoint may refuse).
const EMBED_BATCH: usize = 64;

/// Where a running xerj node lives, if the operator has one. Its ranking
/// joins the fusion as an arm; a node that is down makes the arm
/// *unavailable*, never the endpoint an error.
#[derive(Debug, Clone, PartialEq)]
pub struct XerjEndpoint {
    pub base: String,
    pub index: String,
}

/// The corpus index cache: one entry, keyed by a fingerprint of the corpus
/// it was built from, shared across request threads through `Arc` (`Config`
/// is cloned per connection — without the shared cache every request would
/// re-embed the whole store).
#[derive(Debug, Default)]
pub struct IndexCache {
    fingerprint: Option<u64>,
    index: Option<RecomputeIndex>,
    /// Why the last build failed. Cached like a success: an endpoint that
    /// is down then costs one failed round per corpus change, not one per
    /// query, and the reason is reported to the user either way.
    error: Option<String>,
}

/// The optional search arms.
///
/// With neither set (the default, and what `Config::default()` gives the
/// tests) search is exactly what it was before this existed: substring
/// over stored declarations, no vectors, no fusion. See
/// `logos::formalize::hybrid` for the fusion this file applies.
#[derive(Clone, Default)]
pub struct SearchConfig {
    /// The semantic arm — EmbeddingGemma 2 through this crate's
    /// `CurlEmbedder` (`embed.rs`). `None` = no semantic arm, not a broken
    /// one; only a *configured* arm that fails is reported as unavailable.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// A xerj node whose ranking joins the fusion.
    pub xerj: Option<XerjEndpoint>,
    pub cache: Arc<Mutex<IndexCache>>,
}

/// Manual, because `dyn Embedder` is a trait object: the model id is what
/// is interesting, the trait itself has nothing to say.
impl std::fmt::Debug for SearchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchConfig")
            .field(
                "embedder",
                &self.embedder.as_ref().map(|e| format!("{}@{}", e.id(), e.dims())),
            )
            .field("xerj", &self.xerj)
            .field("cache", &self.cache)
            .finish()
    }
}

/// One declaration a search matched — also one row of the corpus, and one
/// node of the semantic arm's index.
#[derive(Debug, Clone)]
struct Hit {
    project: String,
    proof: String,
    title: String,
    decl: String,
    kind: &'static str,
    line: u32,
    statement: String,
}

impl Hit {
    /// The declaration's identity, in the shape a xerj bulk load uses as
    /// `_id` — so the arm's `_id`s map straight back onto corpus rows.
    fn key(&self) -> String {
        format!("{}/{}/{}@{}", self.project, self.proof, self.decl, self.line)
    }

    /// What gets embedded: name and statement together, the same composite
    /// the `_bulk` export offers as `text`, so the node's BM25 and this
    /// index rank the same content.
    fn text(&self) -> String {
        format!("{} {}", self.decl, self.statement)
    }
}

/// Every declaration of every stored proof — the corpus all arms rank over.
fn corpus(cfg: &Config) -> Vec<Hit> {
    let mut out = Vec::new();
    for (project, p) in all_proofs(cfg) {
        let g = Graph::build(&p.mm0, &p.mm1).accepted();
        for d in &g.decls {
            out.push(Hit {
                project: project.clone(),
                proof: p.id.clone(),
                title: p.title.clone(),
                decl: d.name.clone(),
                kind: d.kind.as_str(),
                line: d.line(),
                statement: d.statement.clone(),
            });
        }
    }
    out
}

/// A fingerprint of the corpus text the index rests on. A `Hasher` over the
/// identity fields *and* the statement: a proof's content is
/// content-addressed, so in practice this only changes when a proof is
/// added — which is exactly when the index must be rebuilt.
fn fingerprint(corpus: &[Hit]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for hit in corpus {
        hit.project.hash(&mut h);
        hit.proof.hash(&mut h);
        hit.decl.hash(&mut h);
        hit.line.hash(&mut h);
        hit.statement.hash(&mut h);
    }
    h.finish()
}

/// One scored result with the arms that ranked it.
struct ScoredHit {
    hit: Hit,
    arms: Vec<String>,
    rrf: f64,
}

/// What one search did, across all its arms. The JSON and the page both
/// render from this, so a report cannot claim an arm that did not answer.
#[derive(Default)]
struct SearchOutcome {
    hits: Vec<ScoredHit>,
    /// Declarations considered (the corpus size).
    corpus: usize,
    /// Arms that returned hits, with their counts.
    answering: BTreeMap<&'static str, usize>,
    /// Configured arms that could not run, with why.
    unavailable: BTreeMap<&'static str, String>,
    /// `model@dims` of the configured embedder, if any.
    embedder: Option<String>,
    /// How many *optional* arms were configured — whether or not they
    /// answered. The page reads exactly as it did before fusion existed
    /// when this is 0.
    configured: usize,
}

impl SearchOutcome {
    /// `rrf` when two or more arms ranked, `single` for one, `none` for a
    /// query nothing matched. Named so a client never has to guess whether
    /// a ranking was fused.
    fn fusion(&self) -> &'static str {
        match self.answering.len() {
            0 => "none",
            1 => "single",
            _ => "rrf",
        }
    }

    /// Whether any optional arm was configured at all — with none, the
    /// page must read exactly as it did before fusion existed.
    fn has_optional_arms(&self) -> bool {
        self.configured > 0
    }
}

/// The semantic arm: index (or reuse) the corpus, query the graph.
///
/// Returns `(hits, why-unavailable)`. An unconfigured embedder is not an
/// unavailable arm — it is a service with one arm, which is the default
/// and needs no apology.
fn semantic_arm(
    cfg: &Config,
    rows: &[Hit],
    query: &str,
) -> (Vec<(usize, f32)>, Option<String>) {
    let Some(embedder) = cfg.search.embedder.as_ref() else {
        return (Vec::new(), None);
    };
    if rows.len() > SEARCH_INDEX_MAX {
        return (
            Vec::new(),
            Some(format!(
                "corpus is {} declarations, over the {} cap for embedding",
                rows.len(),
                SEARCH_INDEX_MAX
            )),
        );
    }

    let fp = fingerprint(rows);
    // The lock covers build *and* query: two concurrent first-searches
    // would otherwise both pay for the build (a stampede), and the index
    // is immutable once built, so holding it through the query only
    // serialises queries — which this service already does per request
    // thread anyway.
    let mut cache = cfg.search.cache.lock().unwrap_or_else(|e| e.into_inner());
    if cache.fingerprint != Some(fp) {
        let texts: Vec<String> = rows.iter().map(Hit::text).collect();
        *cache = match build_index(embedder.as_ref(), &texts) {
            Ok(index) => IndexCache { fingerprint: Some(fp), index: Some(index), error: None },
            Err(reason) => IndexCache { fingerprint: Some(fp), index: None, error: Some(reason) },
        };
    }
    let Some(index) = cache.index.as_ref() else {
        return (
            Vec::new(),
            cache.error.clone().or_else(|| Some("the index is not built".to_string())),
        );
    };
    match index.search(embedder.as_ref(), query, SEARCH_LIMIT) {
        Ok((hits, _stats)) => (
            hits.into_iter()
                .map(|h| (h.idx, h.score))
                // A non-positive cosine means the row does not resemble
                // the query at all — a ranked nothing. The graph traversal
                // scores everything it reaches, so the filter is what
                // keeps "visited" from reading as "matched".
                .filter(|(_, score)| *score > 0.0)
                .collect(),
            None,
        ),
        Err(e) => (Vec::new(), Some(e.to_string())),
    }
}

/// Embed the corpus in batches and build the LEANN-shaped index — which
/// stores texts and a pruned graph, **not** these vectors; they are dropped
/// the moment the graph exists.
fn build_index(embedder: &dyn Embedder, texts: &[String]) -> Result<RecomputeIndex, String> {
    RecomputeIndex::build_chunked(
        embedder,
        texts.to_vec(),
        logos::formalize::index::IndexParams::default(),
        EMBED_BATCH,
    )
    .map_err(|e| e.to_string())
}

/// The xerj arm: URI-search a running node, read `_id`s in rank order.
///
/// The reply is Elasticsearch-shaped because that is the interchange xerj
/// documents (`GET /{index}/_search?q=…`); ids are [`Hit::key`]s, so any
/// id that is not in this corpus is dropped rather than guessed at.
fn xerj_arm(cfg: &Config, query: &str, size: usize) -> (Vec<String>, Option<String>) {
    let Some(node) = cfg.search.xerj.as_ref() else {
        return (Vec::new(), None);
    };
    let url = format!(
        "{}/{}/_search?q={}&size={}",
        node.base.trim_end_matches('/'),
        node.index,
        percent_encode(query),
        size
    );
    match http::curl("GET", &url, &[], None, None) {
        Err(e) => (Vec::new(), Some(e)),
        Ok((0, _)) => (
            Vec::new(),
            // `%{http_code}` 000/0: the connection never completed. Not a
            // reply — a reply would have a status.
            Some("node unreachable (connection failed)".to_string()),
        ),
        Ok((code, _body)) if !(200..300).contains(&code) => {
            (Vec::new(), Some(format!("node answered {code}")))
        }
        Ok((_, body)) => {
            let value: serde_json::Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(e) => return (Vec::new(), Some(format!("node reply is not JSON: {e}"))),
            };
            let Some(hits) = value.get("hits").and_then(|h| h.get("hits")).and_then(|h| h.as_array())
            else {
                return (Vec::new(), Some("node reply has no hits.hits".to_string()));
            };
            let ids = hits
                .iter()
                .filter_map(|h| h.get("_id").and_then(|i| i.as_str()).map(String::from))
                .collect();
            (ids, None)
        }
    }
}

/// One search, all arms, fused.
///
/// Arms: **substring** (always — the pre-fusion behaviour, capped and in
/// corpus order), **semantic** (when an embedder is configured), **xerj**
/// (when a node is). Fusion is RRF over the arms' rankings
/// (`logos::formalize::hybrid`), because a cosine, a word-overlap ratio
/// and a BM25 score are not on one scale and must not be added.
fn search(cfg: &Config, query: &str) -> SearchOutcome {
    let rows = corpus(cfg);
    let needle = query.to_lowercase();

    // Substring arm — the same loop and cap as before fusion existed.
    let mut substring: Vec<usize> = Vec::new();
    for (i, hit) in rows.iter().enumerate() {
        if hit.decl.to_lowercase().contains(&needle)
            || hit.statement.to_lowercase().contains(&needle)
        {
            substring.push(i);
            if substring.len() >= SEARCH_LIMIT {
                break;
            }
        }
    }

    // Semantic arm.
    let mut outcome = SearchOutcome {
        corpus: rows.len(),
        embedder: cfg
            .search
            .embedder
            .as_ref()
            .map(|e| format!("{}@{}", e.id(), e.dims())),
        configured: usize::from(cfg.search.embedder.is_some())
            + usize::from(cfg.search.xerj.is_some()),
        ..SearchOutcome::default()
    };
    let (semantic_raw, semantic_error) = semantic_arm(cfg, &rows, query);
    if let Some(reason) = semantic_error {
        outcome.unavailable.insert("semantic", reason);
    }
    let semantic: Vec<usize> = semantic_raw.iter().map(|(i, _)| *i).collect();

    // xerj arm.
    let (xerj_keys, xerj_error) = xerj_arm(cfg, query, SEARCH_LIMIT);
    if let Some(reason) = xerj_error {
        outcome.unavailable.insert("xerj", reason);
    }
    let mut xerj: Vec<usize> = Vec::new();
    if !xerj_keys.is_empty() {
        let by_key: HashMap<String, usize> =
            rows.iter().enumerate().map(|(i, h)| (h.key(), i)).collect();
        for key in xerj_keys {
            if let Some(&i) = by_key.get(&key) {
                if !xerj.contains(&i) {
                    xerj.push(i);
                }
            }
        }
    }

    for (name, list) in [("substring", &substring), ("semantic", &semantic), ("xerj", &xerj)] {
        if !list.is_empty() {
            outcome.answering.insert(name, list.len());
        }
    }

    let fused = rrf_fuse(
        &[
            ("substring", substring),
            ("semantic", semantic),
            ("xerj", xerj),
        ],
        RRF_K,
    );

    outcome.hits = fused
        .into_iter()
        .take(SEARCH_LIMIT)
        .filter_map(|f| {
            let hit = rows.get(f.idx)?.clone();
            // An arm appears here only if it ranked this row, i.e. only if
            // it answered — `answering` and `because` cannot disagree.
            let arms = f.arms.iter().map(|a| a.arm.clone()).collect();
            Some(ScoredHit { hit, arms, rrf: f.score })
        })
        .collect();
    outcome
}

/// `physlibsearch` takes a natural-language sentence and returns the formal
/// statement it matches, with a model behind it. This service answers over
/// *its* corpus — every stored proof's own declarations — and says which
/// arms answered: substring over names and statements always; a semantic
/// arm and a xerj node when they are configured (`MM0WEB_EMBED_URL`,
/// `MM0WEB_XERJ_URL`), fused by RRF. With neither configured it is the
/// substring search it always was, and the reply says only that.
fn get_search(req: &Request, cfg: &Config, as_json: bool) -> Response {
    let q = req.field("q").unwrap_or_default();
    let q = q.trim().to_string();
    if q.is_empty() {
        return if as_json {
            json(200, json!({"query": "", "count": 0, "results": [],
                "corpus": 0, "arms": {}, "unavailable": {},
                "fusion": "none", "embedder": null}))
        } else {
            html(200, search_page("", &SearchOutcome::default()))
        };
    }
    let outcome = search(cfg, &q);
    if as_json {
        let results: Vec<serde_json::Value> = outcome
            .hits
            .iter()
            .map(|s| {
                let h = &s.hit;
                json!({
                    "declaration": h.decl, "kind": h.kind, "line": h.line,
                    "statement": h.statement, "title": h.title,
                    "project": h.project, "proof": h.proof,
                    "graph": format!("/graph?project={}&id={}", h.project, h.proof),
                    "arms": s.arms, "rrf": s.rrf,
                })
            })
            .collect();
        json(
            200,
            json!({
                "query": q,
                "count": outcome.hits.len(),
                "results": results,
                "corpus": outcome.corpus,
                "arms": outcome.answering,
                "unavailable": outcome.unavailable,
                "fusion": outcome.fusion(),
                "embedder": outcome.embedder,
            }),
        )
    } else {
        html(200, search_page(&q, &outcome))
    }
}

/// Every stored declaration as an Elasticsearch `_bulk` NDJSON body —
/// the load side of the xerj arm:
///
/// ```text
/// curl -H 'content-type: application/x-ndjson' --data-binary @decls.ndjson \\
///      http://127.0.0.1:9200/declarations/_bulk
/// ```
///
/// `_id` is [`Hit::key`] (what the arm reads back) and `text` is
/// [`Hit::text`] (what the semantic arm embeds), so the node's BM25 ranks
/// the same content our vectors do.
fn get_search_bulk(cfg: &Config) -> Response {
    let rows = corpus(cfg);
    let mut out = String::new();
    for h in &rows {
        let doc = json!({
            "project": h.project, "proof": h.proof, "title": h.title,
            "name": h.decl, "kind": h.kind, "line": h.line,
            "statement": h.statement, "text": h.text(),
        });
        out.push_str(&format!(
            "{{\"index\":{{\"_id\":\"{}\"}}}}\n{}\n",
            h.key(),
            serde_json::to_string(&doc).expect("a flat json object serializes")
        ));
    }
    (
        200,
        "application/x-ndjson; charset=utf-8".to_string(),
        out.into_bytes(),
    )
}

fn search_page(q: &str, outcome: &SearchOutcome) -> String {
    let rows: Vec<String> = outcome
        .hits
        .iter()
        .map(|s| {
            let h = &s.hit;
            format!(
                "<li><code>{d}</code> <span>{k}</span>, line {l} — \
                 <a href=\"/graph?project={p}&id={i}\">graph</a> · \
                 <a href=\"/proof?project={p}&id={i}\">{t}</a><br>\
                 <span class=\"stmt\">{s}</span>{via}</li>",
                d = http::escape_html(&h.decl),
                k = h.kind,
                l = h.line,
                p = http::escape_html(&h.project),
                i = http::escape_html(&h.proof),
                t = http::escape_html(&h.title),
                s = http::escape_html(&h.statement),
                via = if outcome.has_optional_arms() && s.arms.len() > 1 {
                    format!(" <span>via {}</span>", s.arms.join(" + "))
                } else {
                    String::new()
                },
            )
        })
        .collect();
    let mut summary = if q.is_empty() {
        "type part of a declaration name or of a statement".to_string()
    } else if outcome.hits.is_empty() {
        format!(
            "no declaration in any stored proof matched <code>{}</code>",
            http::escape_html(q)
        )
    } else {
        format!(
            "{} match{} — a match is a declaration whose name or statement \
             contains the query (capped at {SEARCH_LIMIT})",
            outcome.hits.len(),
            if outcome.hits.len() == 1 { "" } else { "es" }
        )
    };
    // The arm note exists only when an optional arm is configured: with
    // none, this page must read exactly as it did before fusion existed.
    if outcome.has_optional_arms() {
        let arms: Vec<String> = outcome
            .answering
            .iter()
            .map(|(a, n)| format!("{a} ({n})"))
            .collect();
        if !arms.is_empty() {
            let how = if outcome.answering.len() > 1 {
                "fused by rrf"
            } else {
                "one arm answered"
            };
            summary.push_str(&format!(" — arms answering: {}, {how}", arms.join(" · ")));
            if let Some(model) = &outcome.embedder {
                summary.push_str(&format!("; semantic via {model}"));
            }
        }
        for (arm, why) in &outcome.unavailable {
            summary.push_str(&format!(
                " · {arm} arm unavailable: {}",
                http::escape_html(why)
            ));
        }
    }
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <title>search</title></head><body><h1>search stored declarations</h1>\
         <form method=\"get\" action=\"/search\"><input name=\"q\" size=\"40\" \
         value=\"{}\"><button>search</button></form><p>{}</p><ul>{}</ul>\
         <p><a href=\"/\">home</a> · <a href=\"/stats\">stats</a> · \
         <a href=\"/api/v1/search?q={}\">json</a></p></body></html>",
        http::escape_html(q),
        summary,
        rows.join(""),
        http::escape_html(q),
    )
}

/// What the corpus looks like in total: how much is stored, and how much
/// graph there is across all of it. The same counters the graph page shows
/// for one submission, summed.
fn get_stats(cfg: &Config, as_json: bool) -> Response {
    let proofs = all_proofs(cfg);
    let mut states: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_kind: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut declarations, mut references, mut faults) = (0usize, 0usize, 0usize);
    for (_, p) in &proofs {
        *states.entry(p.state.clone()).or_insert(0) += 1;
        // Every stored proof passed the checker, so its undeclared names
        // count as information, not faults, in the totals below.
        let g = Graph::build(&p.mm0, &p.mm1).accepted();
        declarations += g.len();
        references += g.reference_count();
        faults += g.fault_count();
        for (k, v) in g.counts_by_kind() {
            *by_kind.entry(k).or_insert(0) += v;
        }
    }
    let body = json!({
        "projects": projects(cfg).len(),
        "proofs": proofs.len(),
        "states": states,
        "graph": {
            "declarations": declarations,
            "references": references,
            "faults": faults,
            "by_kind": by_kind,
        },
    });
    if as_json {
        return json(200, body);
    }
    let pretty = serde_json::to_string_pretty(&body).unwrap_or_default();
    html(
        200,
        format!(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
             <title>stats</title></head><body><h1>stats</h1><p>totals over every \
             stored proof; a fault is a structural one (see \
             <a href=\"/search\">search</a>), never a checker verdict.</p>\
             <pre>{}</pre><p><a href=\"/\">home</a> · \
             <a href=\"/api/v1/stats\">json</a></p></body></html>",
            http::escape_html(&pretty)
        ),
    )
}

fn render_queued(project: &str, id: &str, body: &str, _cfg: &Config) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta http-equiv=\"refresh\" content=\"1; url={}\"><title>queued</title></head>\
         <body><h1>Queued for publishing</h1><pre>{}</pre>\
         <p><a href=\"{}\">watch it publish</a> · <a href=\"/\">back</a></p></body></html>",
        http::escape_html(&status_path(project, id)),
        http::escape_html(body),
        http::escape_html(&status_path(project, id))
    )
}

fn list_proofs(req: &Request, cfg: &Config) -> Response {
    let project = req.field("project").unwrap_or_default();
    let index = Index::load(&cfg.store.state_dir, &project);
    let rows: Vec<String> = index
        .proofs
        .values()
        .map(|p| {
            format!(
                "<li><a href=\"/proof?project={}&id={}\">{}</a> — seq {}, record {}</li>",
                http::escape_html(&project),
                http::escape_html(&p.id),
                http::escape_html(&p.title),
                p.seq,
                p.record
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "unpublished".to_string())
            )
        })
        .collect();
    html(
        200,
        format!(
            "<!doctype html><h1>proofs — {}</h1><ul>{}</ul>",
            http::escape_html(&project),
            rows.join("")
        ),
    )
}

fn render_page(cfg: &Config) -> String {
    let index = Index::load(&cfg.store.state_dir, "");
    let mut projects: Vec<String> = std::fs::read_dir(&cfg.store.state_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().join("index.json").is_file())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    projects.sort();
    let links = if projects.is_empty() {
        "none yet".to_string()
    } else {
        projects
            .iter()
            .map(|p| format!("<a href=\"/proofs?project={p}\">{p}</a>"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let zenodo = Zenodo::new(cfg.store.zenodo.clone(), cfg.store.state_dir.clone());
    let status = format!(
        "checker {} / {} · zenodo {} · {} projects",
        cfg.checker.mm0_rs.display(),
        cfg.checker.mm0_c.display(),
        if zenodo.configured() {
            "configured"
        } else {
            "unconfigured (verify works, publish does not)"
        },
        projects.len()
    );
    let _ = index;
    PAGE.replace("DEFAULT_PROJECT", "unfer")
        .replace("PROJECT_LINKS", &links)
        .replace("STATUS", &http::escape_html(&status))
}

fn render_result(
    heading: &str,
    body: &str,
    url: Option<&str>,
    graph: Option<(&Graph, &HashSet<String>)>,
) -> String {
    let link = url
        .map(|u| {
            format!(
                "<p><a href=\"{}\">{}</a></p>",
                http::escape_html(u),
                http::escape_html(u)
            )
        })
        .unwrap_or_default();
    // The submission's own dependency graph, embedded rather than linked: the
    // verdict above and the shape of what was judged belong on one page
    // (`crate::graph` builds it; nothing here re-judges it).
    let graph_html = graph
        .filter(|(g, _)| !g.is_empty())
        .map(|(g, failed)| {
            format!(
                "<h2>dependency graph</h2><style>{}</style>{}",
                crate::graph::STYLE,
                g.body_fragment(failed)
            )
        })
        .unwrap_or_default();
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{h}</title></head><body>\
         <h1>{h}</h1>{link}<pre>{log}</pre>{graph}<p><a href=\"/\">back</a></p></body></html>",
        h = http::escape_html(heading),
        link = link,
        log = http::escape_html(body),
        graph = graph_html
    )
}

/// The node names a rejection log names. Only lines that read as a diagnostic
/// count: a name that merely appeared in a quoted excerpt elsewhere would
/// paint the whole graph red.
fn failed_nodes(g: &Graph, log: &str) -> HashSet<String> {
    let diag: Vec<String> = log
        .lines()
        .map(str::to_lowercase)
        .filter(|l| {
            l.contains("error") || l.contains("warning") || l.contains("unknown identifier")
        })
        .collect();
    g.decls
        .iter()
        .filter(|d| diag.iter().any(|l| l.contains(d.name.as_str())))
        .map(|d| d.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request as the parser would build it: the query string in the path
    /// is actually parsed, so route tests exercise `field()` for real.
    fn req(method: &str, path: &str, body: &str) -> Request {
        let (p, q) = match path.split_once('?') {
            Some((p, q)) => (p.to_string(), crate::http::parse_query(q)),
            None => (path.to_string(), HashMap::new()),
        };
        Request {
            method: method.into(),
            path: p,
            query: q,
            headers: HashMap::new(),
            body: body.as_bytes().to_vec(),
            peer: "127.0.0.1".parse().unwrap(),
        }
    }

    #[test]
    fn rate_limiter_closes_the_window() {
        let rl = RateLimiter::new(2, Duration::from_millis(80));
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(rl.allow(ip));
        assert!(rl.allow(ip));
        assert!(!rl.allow(ip), "third request inside the window is refused");
        std::thread::sleep(Duration::from_millis(120));
        assert!(rl.allow(ip), "and the window slides");
    }

    #[test]
    fn unknown_routes_and_bad_methods_are_answered_not_panicked() {
        let cfg = Config::default();
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        assert_eq!(handle(&req("GET", "/nope", ""), &cfg, &rl).0, 404);
        assert_eq!(handle(&req("DELETE", "/", ""), &cfg, &rl).0, 405);
    }

    #[test]
    fn attestation_binds_the_nonce_to_this_processes_bundle() {
        let cfg = Config::default();
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let resp = handle(&req("GET", "/attestation?nonce=challenge-1", ""), &cfg, &rl);
        assert_eq!(resp.0, 200, "{}", String::from_utf8_lossy(&resp.2));
        let v: serde_json::Value = serde_json::from_slice(&resp.2).expect("JSON answer");
        assert_eq!(v["spec"], "svcmux-bundle-v1");
        assert_eq!(v["id"], "mm0web");
        for k in ["binary_sha256", "config_sha256", "bundle"] {
            assert_eq!(
                v[k].as_str().unwrap_or("").len(),
                64,
                "{k} must be a hex sha256"
            );
        }
        // Recompute the binding the way a verifier does — raw sha2 over the
        // served digest, not through the crate that produced it.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(v["bundle"].as_str().unwrap().as_bytes());
        h.update([0]);
        h.update(b"challenge-1");
        assert_eq!(v["binding"], hex::encode(h.finalize()), "nonce binds digest");
        // No nonce, no statement: an unbound answer would replay trivially.
        assert_eq!(handle(&req("GET", "/attestation", ""), &cfg, &rl).0, 400);
    }

    #[test]
    fn a_missing_project_is_a_bad_request() {
        let cfg = Config::default();
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let resp = handle(&req("POST", "/verify", "mm0=x"), &cfg, &rl);
        assert_eq!(resp.0, 400);
        assert!(String::from_utf8_lossy(&resp.2).contains("project is required"));
    }

    #[test]
    fn the_index_page_escapes_its_status_line() {
        let cfg = Config::default();
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let resp = handle(&req("GET", "/", ""), &cfg, &rl);
        assert_eq!(resp.0, 200);
        let body = String::from_utf8_lossy(&resp.2);
        assert!(body.contains("mm0 proofs"));
        assert!(!body.contains("STATUS"), "placeholder must be substituted");
    }

    /// A Config whose state lives in a private temp dir, with a working
    /// (stub) checker so submissions take the same path they do in service.
    fn test_cfg(tag: &str) -> Config {
        let dir = std::env::temp_dir().join(format!(
            "mm0web-web-{tag}-{}-{}",
            std::process::id(),
            crate::store::now_unix()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = StoreConfig {
            state_dir: dir.clone(),
            zenodo: Default::default(),
        };
        let audit = Arc::new(Audit::new(&dir));
        Config {
            store,
            checker: CheckerConfig::default(),
            rate_max: 100,
            rate_window: Duration::from_secs(60),
            work_root: dir.join("work"),
            max_proof_bytes: 4 * 1024 * 1024,
            audit,
            publisher: Arc::new(Publisher::spawn(
                StoreConfig {
                    state_dir: dir.clone(),
                    zenodo: Default::default(),
                },
                Arc::new(Audit::new(&dir)),
            )),
            daily: Arc::new(DailyCap::new(DAILY_DEFAULT, Duration::from_secs(24 * 3600))),
            listen: "127.0.0.1:0".to_string(),
            search: SearchConfig::default(),
        }
    }

    fn queued_cfg(tag: &str, state: &str) -> (Config, String) {
        let cfg = test_cfg(tag);
        let mut index = Index::load(&cfg.store.state_dir, "unfer");
        let id = index.upsert("peano", "sort Prop\n", "thm t: $ Prop;\n");
        index.proofs.get_mut(&id).unwrap().state = state.to_string();
        index.proofs.get_mut(&id).unwrap().record = Some(42);
        index.proofs.get_mut(&id).unwrap().doi = Some("10.5281/zenodo.42".to_string());
        index.save(&cfg.store.state_dir).unwrap();
        (cfg, id)
    }

    #[test]
    fn the_status_page_refreshes_while_queued_and_stops_when_it_lands() {
        let (cfg, id) = queued_cfg("queued", STATE_QUEUED);
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let path = format!("/status?project=unfer&id={id}");

        let (status, _, body) = handle(&req("GET", &path, ""), &cfg, &rl);
        let body = String::from_utf8_lossy(&body);
        assert_eq!(status, 202, "a queued publish is still in flight");
        assert!(body.contains("http-equiv=\"refresh\""), "{body}");
        assert!(body.contains("queued"), "{body}");
        assert!(!body.contains("STATUS"));

        // once it publishes, the page is a terminal 200 with no refresh
        let mut index = Index::load(&cfg.store.state_dir, "unfer");
        index.proofs.get_mut(&id).unwrap().state = STATE_PUBLISHED.to_string();
        index.save(&cfg.store.state_dir).unwrap();
        let (status, _, body) = handle(&req("GET", &path, ""), &cfg, &rl);
        let body = String::from_utf8_lossy(&body);
        assert_eq!(status, 200);
        assert!(!body.contains("http-equiv"), "must stop refreshing: {body}");
        assert!(body.contains("10.5281/zenodo.42"), "{body}");
        assert!(
            body.contains("records/42"),
            "links the Zenodo record: {body}"
        );
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn a_missing_proof_on_the_status_page_is_a_404_not_a_crash() {
        let cfg = test_cfg("nostatus");
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let r = handle(&req("GET", "/status?project=unfer&id=nope", ""), &cfg, &rl);
        assert_eq!(r.0, 404);
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn the_events_endpoint_returns_the_audit_trail() {
        let cfg = test_cfg("events");
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        cfg.audit
            .record("queued", "unfer", "p1", json!({"title": "t"}))
            .unwrap();
        cfg.audit
            .record("published", "unfer", "p1", json!({"record": 9}))
            .unwrap();
        cfg.audit
            .record("verified", "other", "p2", json!({}))
            .unwrap();

        let r = handle(&req("GET", "/events?project=unfer", ""), &cfg, &rl);
        assert_eq!(r.0, 200);
        let v: serde_json::Value = serde_json::from_slice(&r.2).unwrap();
        assert_eq!(v["count"], 2, "filtered by project: {v}");
        assert_eq!(v["events"][1]["record"], 9);

        let r = handle(&req("GET", "/events", ""), &cfg, &rl);
        let v: serde_json::Value = serde_json::from_slice(&r.2).unwrap();
        assert_eq!(v["count"], 3);
        assert!(
            v["events"]
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e["ts"].is_number())
        );
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn proof_json_carries_the_publish_state_and_doi() {
        let (cfg, id) = queued_cfg("proofjson", STATE_PUBLISHED);
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let r = handle(
            &req("GET", &format!("/proof?project=unfer&id={id}"), ""),
            &cfg,
            &rl,
        );
        assert_eq!(r.0, 200);
        let v: serde_json::Value = serde_json::from_slice(&r.2).unwrap();
        assert_eq!(v["state"], "published");
        assert_eq!(v["doi"], "10.5281/zenodo.42");
        assert_eq!(v["record"], 42);
        assert!(
            v["status"]
                .as_str()
                .unwrap()
                .starts_with("/status?project=unfer")
        );
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn the_daily_cap_counts_new_proofs_only_and_slides() {
        let cap = DailyCap::new(2, Duration::from_millis(120));
        let ip: IpAddr = "10.0.0.9".parse().unwrap();
        assert!(cap.allow(ip));
        assert!(cap.allow(ip));
        assert!(!cap.allow(ip), "the third new proof is over the cap");
        let other: IpAddr = "10.0.0.8".parse().unwrap();
        assert!(cap.allow(other), "the cap is per client");
        std::thread::sleep(Duration::from_millis(150));
        assert!(cap.allow(ip), "and the window slides");
        assert_eq!(cap.max(), 2);

        let off = DailyCap::new(0, Duration::from_secs(1));
        for _ in 0..10 {
            assert!(off.allow(ip));
        }
    }

    /// The three read-only routes this service gained for looking *at* a
    /// submission: its graph, a search over stored declarations, and the
    /// corpus totals. None of them can accept, reject or publish anything.
    #[test]
    fn the_graph_search_and_stats_routes_answer_their_own_questions() {
        let (cfg, id) = queued_cfg("routes", STATE_PUBLISHED);
        let rl = RateLimiter::new(100, Duration::from_millis(10));

        let page = handle(
            &req("GET", &format!("/graph?project=unfer&id={id}"), ""),
            &cfg,
            &rl,
        );
        assert_eq!(page.0, 200, "{}", String::from_utf8_lossy(&page.2));
        let body = String::from_utf8_lossy(&page.2);
        assert!(body.contains("<svg"), "the figure is on the page");
        assert!(body.contains("dependency graph"));
        assert!(!body.contains("<script"), "no JavaScript, as everywhere here");
        assert!(body.contains("/proof?project=unfer"), "it links back: {body}");

        let raw = handle(
            &req(
                "GET",
                &format!("/api/v1/graph?project=unfer&id={id}"),
                "",
            ),
            &cfg,
            &rl,
        );
        let v: serde_json::Value = serde_json::from_slice(&raw.2).unwrap();
        assert_eq!(v["counts"]["declarations"], 1, "{v}");
        assert_eq!(v["unused"][0], "Prop", "{v}");

        // A graph nobody can be pointed at is a 404, not an empty figure.
        assert_eq!(
            handle(&req("GET", "/graph?project=unfer&id=nope", ""), &cfg, &rl).0,
            404
        );
        assert_eq!(
            handle(
                &req("GET", "/api/v1/graph?project=unfer&id=nope", ""),
                &cfg,
                &rl
            )
            .0,
            404
        );

        // Search: case-insensitive substring over stored declarations, and
        // every hit links at the graph of the proof that declares it.
        let hits = handle(&req("GET", "/api/v1/search?q=prop", ""), &cfg, &rl);
        let v: serde_json::Value = serde_json::from_slice(&hits.2).unwrap();
        assert_eq!(v["count"], 1, "{v}");
        assert_eq!(v["results"][0]["declaration"], "Prop");
        assert_eq!(v["results"][0]["kind"], "sort");
        assert!(
            v["results"][0]["graph"]
                .as_str()
                .unwrap_or_default()
                .contains(&id),
            "{v}"
        );
        let page = handle(&req("GET", "/search?q=prop", ""), &cfg, &rl);
        assert_eq!(page.0, 200);
        let body = String::from_utf8_lossy(&page.2);
        assert!(body.contains("/graph?project=unfer"), "{body}");
        // No query is the form, not a match-all dump.
        let v: serde_json::Value =
            serde_json::from_slice(&handle(&req("GET", "/api/v1/search", ""), &cfg, &rl).2)
                .unwrap();
        assert_eq!(v["count"], 0, "{v}");

        // Stats: the same counters, summed over every stored proof.
        let v: serde_json::Value =
            serde_json::from_slice(&handle(&req("GET", "/api/v1/stats", ""), &cfg, &rl).2)
                .unwrap();
        assert_eq!(v["proofs"], 1, "{v}");
        assert_eq!(v["graph"]["declarations"], 1, "{v}");
        assert_eq!(v["states"]["published"], 1, "{v}");
        assert_eq!(handle(&req("GET", "/stats", ""), &cfg, &rl).0, 200);
        assert_eq!(
            handle(&req("GET", "/api/v1/nope", ""), &cfg, &rl).0,
            404,
            "the api namespace does not swallow typos"
        );

        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    // ── the optional search arms (P9) ──────────────────────────────────────

    use logos::formalize::embed::{EmbedError, Embedding};

    /// A stand-in embedder with a controllable notion of similarity:
    /// `prop`-vocabulary (including the word "special", which this fake
    /// declares a synonym of) maps to one vector, everything else to
    /// another. Deterministic, offline, and — like xerj's default embedder
    /// — honestly not neural: its id says so.
    #[derive(Debug)]
    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn id(&self) -> &str {
            "fake-test-embedder"
        }

        fn dims(&self) -> usize {
            4
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts
                .iter()
                .map(|t| {
                    let lower = t.to_lowercase();
                    if lower.contains("special") || lower.contains("prop") {
                        vec![1.0, 0.0, 0.0, 0.0]
                    } else {
                        vec![0.0, 1.0, 0.0, 0.0]
                    }
                })
                .collect())
        }
    }

    #[test]
    fn with_no_optional_arms_the_search_reply_is_the_shape_it_always_was() {
        let (cfg, _id) = queued_cfg("plainsearch", STATE_PUBLISHED);
        let rl = RateLimiter::new(100, Duration::from_millis(10));
        let v: serde_json::Value =
            serde_json::from_slice(&handle(&req("GET", "/api/v1/search?q=prop", ""), &cfg, &rl).2)
                .unwrap();
        assert_eq!(v["count"], 1, "{v}");
        assert_eq!(v["results"][0]["declaration"], "Prop");
        // …and the new fields tell the same single-arm truth rather than
        // claiming a semantic arm that was never configured.
        assert!(v["embedder"].is_null(), "{v}");
        assert_eq!(v["arms"]["substring"], 1, "{v}");
        assert_eq!(v["fusion"], "single", "{v}");
        assert_eq!(v["unavailable"], serde_json::json!({}), "{v}");
        assert_eq!(v["corpus"], 1, "{v}");
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn the_semantic_arm_finds_what_substring_cannot_and_names_its_model() {
        let (mut cfg, _id) = queued_cfg("semarm", STATE_PUBLISHED);
        cfg.search.embedder = Some(Arc::new(FakeEmbedder));
        let rl = RateLimiter::new(100, Duration::from_millis(10));

        // "special" appears in no declaration name and no statement:
        // substring alone finds nothing, the semantic arm maps it onto the
        // Prop vocabulary the fake declares.
        let v: serde_json::Value = serde_json::from_slice(
            &handle(&req("GET", "/api/v1/search?q=special", ""), &cfg, &rl).2,
        )
        .unwrap();
        assert_eq!(v["count"], 1, "semantic-only hit: {v}");
        assert_eq!(v["results"][0]["declaration"], "Prop", "{v}");
        assert_eq!(v["arms"]["semantic"], 1, "substring did not answer: {v}");
        assert!(v["arms"].get("substring").is_none(), "{v}");
        assert_eq!(v["fusion"], "single", "{v}");
        assert_eq!(v["embedder"], "fake-test-embedder@4", "{v}");
        let via: Vec<String> = v["results"][0]["arms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().to_string())
            .collect();
        assert!(via.contains(&"semantic".to_string()), "{v}");
        // The graph scores everything it reaches, but a row with no
        // similarity is filtered rather than padded in: only the matching
        // declaration counts as a semantic hit, and the corpus counter
        // says how many rows were ranked over.
        assert_eq!(v["corpus"], 1, "{v}");

        // Both arms answer "prop": the reply says rrf, and the hit says
        // which arms ranked it.
        let v: serde_json::Value = serde_json::from_slice(
            &handle(&req("GET", "/api/v1/search?q=prop", ""), &cfg, &rl).2,
        )
        .unwrap();
        assert_eq!(v["fusion"], "rrf", "{v}");
        assert!(
            v["arms"]["substring"].as_u64().unwrap_or(0) >= 1
                && v["arms"]["semantic"].as_u64().unwrap_or(0) >= 1,
            "{v}"
        );
        let via = v["results"][0]["arms"].as_array().unwrap();
        assert_eq!(via.len(), 2, "both arms ranked the one hit: {v}");

        // The page says the same thing in prose.
        let page = handle(&req("GET", "/search?q=special", ""), &cfg, &rl);
        let body = String::from_utf8_lossy(&page.2);
        assert!(body.contains("arms answering: semantic"), "{body}");
        assert!(body.contains("fake-test-embedder@4"), "{body}");
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn a_xerj_arm_merges_hits_and_a_down_node_is_reported_not_fatal() {
        use std::io::{BufRead, BufReader, Write};

        let (mut cfg, _id) = queued_cfg("xerjarm", STATE_PUBLISHED);
        let rl = RateLimiter::new(100, Duration::from_millis(10));

        // Configured but unreachable: search still answers, and the arm is
        // reported unavailable with a reason — never a 500.
        cfg.search.xerj = Some(XerjEndpoint {
            base: "http://127.0.0.1:1".to_string(),
            index: "declarations".to_string(),
        });
        let v: serde_json::Value = serde_json::from_slice(
            &handle(&req("GET", "/api/v1/search?q=prop", ""), &cfg, &rl).2,
        )
        .unwrap();
        assert_eq!(v["count"], 1, "substring still answers: {v}");
        assert!(
            v["unavailable"].get("xerj").is_some(),
            "the down arm is named: {v}"
        );
        assert_eq!(v["fusion"], "single", "{v}");

        // A node that answers: its ranked _id joins the fusion. The id is
        // the one `search-bulk` hands out, so the test reads it back from
        // the live corpus rather than hard-coding a line number.
        let bulk = String::from_utf8_lossy(
            &handle(&req("GET", "/api/v1/search-bulk", ""), &cfg, &rl).2,
        )
        .into_owned();
        let first_doc: serde_json::Value = serde_json::from_str(bulk.lines().nth(1).unwrap()).unwrap();
        let key = format!(
            "{}/{}/{}@{}",
            first_doc["project"].as_str().unwrap(),
            first_doc["proof"].as_str().unwrap(),
            first_doc["name"].as_str().unwrap(),
            first_doc["line"].as_u64().unwrap()
        );

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let reply = format!(
            "{{\"hits\":{{\"hits\":[{{\"_id\":\"{key}\"}}]}}}}"
        );
        let seen = std::sync::Arc::new(Mutex::new(String::new()));
        let seen_in_thread = seen.clone();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h == "\r\n" || h.is_empty() {
                    break;
                }
            }
            *seen_in_thread.lock().unwrap() = line.clone();
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
        });
        cfg.search.xerj = Some(XerjEndpoint {
            base: format!("http://{addr}"),
            index: "declarations".to_string(),
        });

        let v: serde_json::Value = serde_json::from_slice(
            &handle(&req("GET", "/api/v1/search?q=prop", ""), &cfg, &rl).2,
        )
        .unwrap();
        server.join().unwrap();
        let request = seen.lock().unwrap().clone();
        assert!(request.starts_with("GET /declarations/_search?q=prop&size=50"), "{request}");
        assert_eq!(v["count"], 1, "{v}");
        assert_eq!(v["arms"]["xerj"], 1, "{v}");
        assert_eq!(v["arms"]["substring"], 1, "{v}");
        assert_eq!(v["fusion"], "rrf", "two arms fused: {v}");
        assert!(v["unavailable"].get("xerj").is_none(), "{v}");
        let via = v["results"][0]["arms"].as_array().unwrap();
        assert_eq!(via.len(), 2, "the hit carries both arms: {v}");
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    #[test]
    fn the_bulk_export_is_loadable_ndjson_with_the_ids_the_arm_reads_back() {
        let (cfg, _id) = queued_cfg("bulkexport", STATE_PUBLISHED);
        let (status, ctype, body) = handle(&req("GET", "/api/v1/search-bulk", ""), &cfg, &rl_zero());
        assert_eq!(status, 200);
        assert!(ctype.starts_with("application/x-ndjson"), "{ctype}");
        let text = String::from_utf8(body).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len() % 2, 0, "two lines per declaration");
        for pair in lines.chunks(2) {
            let action: serde_json::Value = serde_json::from_str(pair[0]).unwrap();
            let doc: serde_json::Value = serde_json::from_str(pair[1]).unwrap();
            let id = action["index"]["_id"].as_str().unwrap();
            assert_eq!(
                id,
                format!(
                    "{}/{}/{}@{}",
                    doc["project"].as_str().unwrap(),
                    doc["proof"].as_str().unwrap(),
                    doc["name"].as_str().unwrap(),
                    doc["line"].as_u64().unwrap()
                ),
                "the _id is exactly what the xerj arm reads back"
            );
            assert_eq!(
                doc["text"].as_str().unwrap(),
                format!(
                    "{} {}",
                    doc["name"].as_str().unwrap(),
                    doc["statement"].as_str().unwrap()
                ),
                "text is the same composite the semantic arm embeds"
            );
        }
        // Deterministic: two exports diff clean.
        let again = String::from_utf8(
            handle(&req("GET", "/api/v1/search-bulk", ""), &cfg, &rl_zero()).2,
        )
        .unwrap();
        assert_eq!(text, again);
        let _ = std::fs::remove_dir_all(&cfg.store.state_dir);
    }

    /// A limiter that never trips, for tests whose subject is not rate.
    fn rl_zero() -> RateLimiter {
        RateLimiter::new(1000, Duration::from_secs(60))
    }
}
