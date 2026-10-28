//! `mm0web` — the server binary.
//!
//! ```text
//! cargo run -p mm0web -- --bind 127.0.0.1:8080 --state /var/lib/mm0web
//!
//! ZENODO_API_KEY=…            # required only for publishing
//! MM0WEB_ZENODO_BASE=…        # default https://sandbox.zenodo.org/api
//! MM0WEB_MM0_RS=/path/mm0-rs  # default: from PATH
//! MM0WEB_MM0_C=/path/mm0-c    # default: from PATH
//! MM0WEB_RATE=10              # submissions per client per minute
//! MM0WEB_VERIFY_TIMEOUT=60    # seconds per checker stage
//! ```
//!
//! Binds loopback by default on purpose: this is meant to run on the
//! tailnet-only Oracle host (`../tailscale-ssh-only-oracle-linux-10.md`), and
//! binding a public interface is an explicit `--bind 0.0.0.0:8080` away.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use mm0web::audit::Audit;
use mm0web::checker::CheckerConfig;
use mm0web::publisher::Publisher;
use mm0web::store::{StoreConfig, ZenodoConfig};
use mm0web::web::{self, Config, DailyCap, RateLimiter};

fn main() {
    let mut bind = "127.0.0.1:8080".to_string();
    let mut state = PathBuf::from("state");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args.get(i).cloned().unwrap_or(bind);
            }
            "--state" => {
                i += 1;
                state = args.get(i).cloned().unwrap_or_default().into();
            }
            "--help" | "-h" => {
                println!(
                    "mm0web — verify mm0 proofs and publish them to a Zenodo project deposit\n\
                     \n  --bind ADDR   listen address (default 127.0.0.1:8080)\n\
                     \x20 --state DIR   local index/remote state (default ./state)\n"
                );
                return;
            }
            other => {
                eprintln!("unknown argument {other}; try --help");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let rate_max = env("MM0WEB_RATE")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10usize);
    let timeout = env("MM0WEB_VERIFY_TIMEOUT")
        .and_then(|v| v.parse().ok())
        .unwrap_or(60u64);
    let squash_after = env("MM0WEB_ZENODO_SQUASH_AFTER")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50usize);
    let daily_cap = env("MM0WEB_DAILY_CAP")
        .and_then(|v| v.parse().ok())
        .unwrap_or(web::DAILY_DEFAULT);

    let zenodo = ZenodoConfig {
        api_key: env("ZENODO_API_KEY").unwrap_or_default(),
        base_url: env("MM0WEB_ZENODO_BASE")
            .unwrap_or_else(|| "https://sandbox.zenodo.org/api".to_string()),
        squash_after,
        creator: env("MM0WEB_CREATOR").unwrap_or_else(|| "mm0web".to_string()),
        creator_orcid: env("MM0WEB_CREATOR_ORCID").unwrap_or_default(),
        public_url: env("MM0WEB_PUBLIC_URL").unwrap_or_else(|| format!("http://{bind}")),
    };
    let store = StoreConfig {
        state_dir: state.clone(),
        zenodo: zenodo.clone(),
    };
    // One audit handle shared by the request threads and the publish worker:
    // two handles would append without a common lock.
    let audit = Arc::new(Audit::new(&state));

    let cfg = Config {
        store: store.clone(),
        checker: CheckerConfig {
            mm0_rs: env("MM0WEB_MM0_RS")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("mm0-rs")),
            mm0_c: env("MM0WEB_MM0_C")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("mm0-c")),
            timeout: Duration::from_secs(timeout),
        },
        rate_max,
        rate_window: Duration::from_secs(60),
        work_root: state.join("work"),
        max_proof_bytes: 4 * 1024 * 1024,
        audit: audit.clone(),
        publisher: Arc::new(Publisher::spawn(store.clone(), audit.clone())),
        daily: Arc::new(DailyCap::new(daily_cap, Duration::from_secs(24 * 3600))),
    };

    std::fs::create_dir_all(&cfg.store.state_dir).expect("create state dir");
    let limiter = RateLimiter::new(cfg.rate_max, cfg.rate_window);

    let listener = TcpListener::bind(&bind).unwrap_or_else(|e| {
        eprintln!("cannot bind {bind}: {e}");
        std::process::exit(1);
    });
    println!(
        "mm0web listening on http://{} (state {}, checker {} + {}, zenodo {})",
        listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or(bind.clone()),
        state.display(),
        cfg.checker.mm0_rs.display(),
        cfg.checker.mm0_c.display(),
        if cfg.store.zenodo.api_key.is_empty() {
            "unconfigured".to_string()
        } else {
            cfg.store.zenodo.base_url.clone()
        }
    );

    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let cfg = cfg.clone();
        let limiter = &limiter;
        // One request per connection, one thread per connection: the workload
        // here is a subprocess wait, so threads are the right unit.
        let (status, ctype, body) = {
            match mm0web::http::parse_request(&mut stream) {
                Ok(req) => web::handle(&req, &cfg, limiter),
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => (
                    413,
                    "text/plain; charset=utf-8".to_string(),
                    format!("{e}\n").into_bytes(),
                ),
                Err(_) => (
                    400,
                    "text/plain; charset=utf-8".to_string(),
                    b"malformed request\n".to_vec(),
                ),
            }
        };
        if let Err(e) = mm0web::http::respond(&mut stream, status, &ctype, &body) {
            eprintln!("write failed: {e}");
        }
    }
}
