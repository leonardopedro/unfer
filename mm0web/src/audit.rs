//! The audit trail — an append-only `events.jsonl` with a **closed action
//! vocabulary**, the pattern `brief-ideas`' `AuditLog` model enforces
//! (`validates_inclusion_of :action, :in => [...]`): an audit log whose
//! actions are free text is a log you cannot later query or trust.
//!
//! Events are one JSON object per line:
//!
//! ```json
//! {"ts":1793055420,"action":"published","project":"unfer",
//!  "proof":"peano-66bfb078","record":42,"doi":"10.5281/zenodo.42"}
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};

/// The only actions this service ever records.
pub const ACTIONS: &[&str] = &[
    // a submission passed both checker stages
    "verified",
    // a submission was rejected, with the stage that rejected it
    "rejected",
    // a verified proof was handed to the publish queue
    "queued",
    // the Zenodo push landed, with record id and DOI
    "published",
    // a push attempt failed, with the error and the attempt number
    "publish-failed",
    // the queue refused work (bounded on purpose)
    "queue-full",
    // the daily per-client submission cap refused a new proof
    "rate-limited",
];

#[derive(Debug)]
pub struct Audit {
    path: PathBuf,
    lock: Mutex<()>,
}

impl Audit {
    pub fn new(state_dir: &Path) -> Self {
        Self {
            path: state_dir.join("events.jsonl"),
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one event. `extra` supplies the event's own fields (record id,
    /// error, …); `ts`, `action`, `project` and `proof` are always ours.
    ///
    /// An action outside [`ACTIONS`] is refused — a typo must not become a new
    /// event type. A failed write is not fatal to the caller: the request that
    /// was being audited still happened.
    pub fn record(
        &self,
        action: &str,
        project: &str,
        proof: &str,
        extra: Value,
    ) -> Result<(), String> {
        if !ACTIONS.contains(&action) {
            return Err(format!("action {action:?} is not in the audit vocabulary"));
        }
        let mut event = json!({
            "ts": crate::store::now_unix(),
            "action": action,
            "project": project,
            "proof": proof,
        });
        if let Some(map) = extra.as_object() {
            for (k, v) in map {
                event[k] = v.clone();
            }
        }
        let _guard = self.lock.lock().map_err(|e| e.to_string())?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        writeln!(f, "{}", event).map_err(|e| e.to_string())
    }

    /// Every event recorded so far, oldest first (corrupt lines are skipped:
    /// a half-written tail must not hide the history before it).
    pub fn read(&self) -> Vec<Value> {
        let Ok(text) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit(tag: &str) -> Audit {
        let dir = std::env::temp_dir().join(format!("mm0web-audit-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Audit::new(&dir)
    }

    #[test]
    fn events_are_json_lines_in_order() {
        let a = audit("order");
        a.record("verified", "unfer", "p1", json!({})).unwrap();
        a.record("published", "unfer", "p1", json!({"record": 7, "doi": "10.1/x"}))
            .unwrap();
        let ev = a.read();
        assert_eq!(ev.len(), 2);
        assert_eq!(ev[0]["action"], "verified");
        assert_eq!(ev[1]["record"], 7);
        assert_eq!(ev[1]["doi"], "10.1/x");
        assert!(ev[1]["ts"].as_u64().unwrap() > 0);
        assert_eq!(ev[1]["project"], "unfer");
        assert_eq!(ev[1]["proof"], "p1");
        let _ = std::fs::remove_dir_all(a.path.parent().unwrap());
    }

    #[test]
    fn an_action_outside_the_vocabulary_is_refused() {
        let a = audit("vocab");
        assert!(a.record("tweeted", "p", "x", json!({})).is_err());
        assert!(a.record("PUBLISHED", "p", "x", json!({})).is_err());
        assert!(a.read().is_empty(), "nothing was written");
        let _ = std::fs::remove_dir_all(a.path.parent().unwrap());
    }

    #[test]
    fn a_garbage_line_does_not_hide_the_rest_of_the_log() {
        let a = audit("tail");
        a.record("queued", "p", "x", json!({})).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(a.path())
            .unwrap()
            .write_all(b"{not json\n")
            .unwrap();
        assert_eq!(a.read().len(), 1);
        let _ = std::fs::remove_dir_all(a.path.parent().unwrap());
    }

    #[test]
    fn the_vocabulary_covers_the_publish_lifecycle() {
        for needed in ["verified", "rejected", "queued", "published", "publish-failed"] {
            assert!(ACTIONS.contains(&needed), "{needed} missing");
        }
    }
}
