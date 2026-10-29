//! Routing and rendering. One server-rendered page, no JavaScript: the whole
//! service is a form with a submit button, which is what "smallest version that
//! works" means here. Verification is synchronous (it is two subprocesses);
//! publishing is queued (`crate::publisher`) and observed at `/status`.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::audit::Audit;
use crate::checker::{self, CheckerConfig};
use crate::http::{self, Request};
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
        ("GET", "/events") => get_events(req, cfg),
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
            html(400, render_result("Rejected", &text, None, cfg))
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
            html(200, render_result("Accepted", &outcome.log, None, cfg))
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
                render_result("Already published", &text, Some(&url), cfg),
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
            html(429, render_result("Daily cap reached", &msg, None, cfg))
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
                render_result("Verified, not published", &text, None, cfg),
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
                    render_result("Verified, queue unavailable", &text, None, cfg),
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

fn render_result(heading: &str, body: &str, url: Option<&str>, cfg: &Config) -> String {
    let link = url
        .map(|u| {
            format!(
                "<p><a href=\"{}\">{}</a></p>",
                http::escape_html(u),
                http::escape_html(u)
            )
        })
        .unwrap_or_default();
    let _ = cfg;
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{h}</title></head><body>\
         <h1>{h}</h1>{link}<pre>{log}</pre><p><a href=\"/\">back</a></p></body></html>",
        h = http::escape_html(heading),
        link = link,
        log = http::escape_html(body)
    )
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
}
