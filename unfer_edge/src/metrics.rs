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
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            ops: Mutex::new(HashMap::new()),
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

    pub fn snapshot(&self) -> HashMap<String, OpRow> {
        self.ops.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// JSON form (`{"version":{"calls":1,"errors":0,"total_us":12},"..."}`)
    /// with `allowed` ops zero-filled so operators see the full surface.
    pub fn to_json(&self, allowed: &[&str]) -> serde_json::Value {
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
        serde_json::json!({ "ops": map })
    }

    /// Prometheus text exposition (`unfer_edge_op_calls{op="version"} 1` …).
    pub fn to_prometheus(&self, allowed: &[&str]) -> String {
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

        let j = m.to_json(OPS);
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
        let text = m.to_prometheus(OPS);
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
        let j = m.to_json(OPS);
        assert_eq!(j["ops"]["evolve"]["calls"], 8 * 50);
    }
}