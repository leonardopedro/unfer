//! Edge call metrics (S13, F12).
//!
//! Per-op counters for the gateway hot path: calls, filter rejections, and
//! accumulated request-filter latency. `/metrics` is served by the edge before
//! any backend forwarding, in JSON or Prometheus text form.

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpRow {
    pub calls: u64,
    pub errors: u64,
    pub total_us: u64,
}

pub struct Metrics {
    ops: Mutex<HashMap<String, OpRow>>,
    spend: Mutex<HashMap<String, SpendRow>>,
}

/// Per-principal spend attribution (S25/F24): how much of each metered op the
/// principal consumed this window. Mirrors Cloudflare's AI-Gateway attribution —
/// administrators can see where spend goes, not just aggregate call counts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpendRow {
    pub calls: u64,
    pub denied: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            ops: Mutex::new(HashMap::new()),
            spend: Mutex::new(HashMap::new()),
        }
    }

    /// Record one op. `ok` distinguishes a forwarded filter pass from a
    /// rejection; `elapsed_us` is the request-filter cost.
    pub fn record(&self, op: &str, ok: bool, elapsed_us: u64) {
        let mut guard = self.ops.lock().unwrap_or_else(|e| e.into_inner());
        let row = guard.entry(op.to_string()).or_default();
        row.calls += 1;
        if !ok {
            row.errors += 1;
        }
        row.total_us += elapsed_us;
    }

    /// Attribute one metered call to `principal` (S25). `denied` marks a call
    /// refused by the budget/rate gate so operators see cost *and* rejections.
    pub fn record_spend(&self, principal: &str, denied: bool) {
        let mut guard = self.spend.lock().unwrap_or_else(|e| e.into_inner());
        let row = guard.entry(principal.to_string()).or_default();
        row.calls += 1;
        if denied {
            row.denied += 1;
        }
    }

    pub fn snapshot(&self) -> HashMap<String, OpRow> {
        self.ops.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn spend_snapshot(&self) -> HashMap<String, SpendRow> {
        self.spend.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// JSON form (`{"version":{"calls":1,"errors":0,"total_us":12},"..."}`)
    /// with `allowed` ops zero-filled so operators see the full surface. Includes
    /// a per-principal `spend` map when `principals` is non-empty.
    pub fn to_json(&self, allowed: &[&str], principals: &[&str]) -> serde_json::Value {
        let snap = self.snapshot();
        let mut map = serde_json::Map::new();
        for op in allowed {
            let row = snap.get(*op).cloned().unwrap_or_default();
            map.insert(
                op.to_string(),
                serde_json::json!({
                    "calls": row.calls,
                    "errors": row.errors,
                    "total_us": row.total_us,
                }),
            );
        }
        let mut out = serde_json::json!({ "ops": map });
        if !principals.is_empty() {
            let spend = self.spend_snapshot();
            let mut sp = serde_json::Map::new();
            for p in principals {
                let row = spend.get(*p).cloned().unwrap_or_default();
                sp.insert(
                    p.to_string(),
                    serde_json::json!({ "calls": row.calls, "denied": row.denied }),
                );
            }
            out.as_object_mut()
                .expect("metrics is an object")
                .insert("spend".to_string(), serde_json::json!(sp));
        }
        out
    }

    /// Prometheus text exposition (`unfer_edge_op_calls{op="version"} 1` …), plus
    /// per-principal spend when `principals` is non-empty.
    pub fn to_prometheus(&self, allowed: &[&str], principals: &[&str]) -> String {
        let snap = self.snapshot();
        let mut out = String::new();
        for op in allowed {
            let row = snap.get(*op).cloned().unwrap_or_default();
            out.push_str(&format!(
                "unfer_edge_op_calls{{op=\"{op}\"}} {}\n",
                row.calls
            ));
            out.push_str(&format!(
                "unfer_edge_op_errors{{op=\"{op}\"}} {}\n",
                row.errors
            ));
        }
        if !principals.is_empty() {
            let spend = self.spend_snapshot();
            for p in principals {
                let row = spend.get(*p).cloned().unwrap_or_default();
                out.push_str(&format!(
                    "unfer_edge_principal_calls{{principal=\"{p}\"}} {}\n",
                    row.calls
                ));
                out.push_str(&format!(
                    "unfer_edge_principal_denied{{principal=\"{p}\"}} {}\n",
                    row.denied
                ));
            }
        }
        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: &[&str] = &["version", "evolve", "condition"];

    #[test]
    fn records_calls_and_errors() {
        let m = Metrics::new();
        m.record("version", true, 42);
        m.record("version", true, 7);
        m.record("unknown", false, 1);

        let j = m.to_json(OPS, &[]);
        assert_eq!(j["ops"]["version"]["calls"], 2);
        assert_eq!(j["ops"]["version"]["errors"], 0);
        assert_eq!(j["ops"]["version"]["total_us"], 49);
        // A rejected (unknown-string) op is counted but not merged into allowed rows.
        assert_eq!(j["ops"]["evolve"]["calls"], 0);
        assert_eq!(m.snapshot()["unknown"].errors, 1);
    }

    #[test]
    fn prometheus_text_lists_all_allowed_ops() {
        let m = Metrics::new();
        m.record("evolve", true, 5);
        let text = m.to_prometheus(OPS, &[]);
        assert!(text.contains("unfer_edge_op_calls{op=\"evolve\"} 1"));
        assert!(text.contains("unfer_edge_op_calls{op=\"version\"} 0"));
        assert!(text.contains("unfer_edge_op_errors{op=\"condition\"} 0"));
    }

    #[test]
    fn metric_rows_never_panic_on_concurrent_record() {
        let m = Metrics::new();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for i in 0..50 {
                        m.record("evolve", i % 3 != 0, 1);
                    }
                });
            }
        });
        let j = m.to_json(OPS, &[]);
        assert_eq!(j["ops"]["evolve"]["calls"], 8 * 50);
    }

    #[test]
    fn spend_attribute_reports_denied_without_consuming_op_rows() {
        // S25 (F24): per-principal spend attribution is separate from op counters.
        let m = Metrics::new();
        m.record_spend("alice", false);
        m.record_spend("alice", true);
        m.record_spend("bob", false);

        let j = m.to_json(OPS, &["alice", "bob"]);
        assert_eq!(j["spend"]["alice"]["calls"], 2);
        assert_eq!(j["spend"]["alice"]["denied"], 1);
        assert_eq!(j["spend"]["bob"]["calls"], 1);
        assert_eq!(j["spend"]["bob"]["denied"], 0);
        // Spend rows don't perturb the op surface.
        assert_eq!(j["ops"]["version"]["calls"], 0);

        let text = m.to_prometheus(OPS, &["alice"]);
        assert!(text.contains("unfer_edge_principal_calls{principal=\"alice\"} 2"));
        assert!(text.contains("unfer_edge_principal_denied{principal=\"alice\"} 1"));

        // Empty principal list omits the spend section entirely.
        let j2 = m.to_json(OPS, &[]);
        assert!(j2.get("spend").is_none());
    }
}
