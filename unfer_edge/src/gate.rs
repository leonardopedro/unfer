//! Gatekeeper console (S19, F18): exposes the kernel's human-gate mediation
//! tier over HTTP so the operator can review pending side effects and resolve
//! them (approve/apply the provisional forecast, or reject it).
//!
//! The verdict records live in the kernel (`unfer_ffi::uk_gate_list_pending` /
//! `uk_gate_approve` / `uk_gate_reject`). With `--features audit` the edge links
//! the kernel FFI in-process and serves:
//!
//! * `GET  /api/gate/pending`                 — pending mediators, oldest first.
//! * `POST /api/gate/approve` `{"handle": N}` — approve a pending action.
//! * `POST /api/gate/reject`  `{"handle": N}`  — reject a pending action.
//!
//! In a split deployment (edge and kernel in separate processes) the operator
//! proxies these paths to the kernel host instead.

use serde_json::Value;

/// True when `path` belongs to the gatekeeper console. The caller dispatches on
/// the exact route (method + suffix).
pub fn is_gate_path(path: &str) -> bool {
    path == "/api/gate/pending" || path == "/api/gate/approve" || path == "/api/gate/reject"
}

/// The operator principal tag recorded on approvals/rejections so the audit
/// trail attributes the resolution to a human (the console operator) rather
/// than the submitting kernel caller.
const GATE_OPERATOR: &str = r#"{"from":"hook","principal":"operator"}"#;

/// `GET /api/gate/pending` payload: `[[handle, record], ...]` oldest first.
/// Uses the kernel's two-call buffer protocol (`uk_gate_list_pending`).
pub fn pending_list_body() -> Result<Vec<u8>, String> {
    let needed = unfer_ffi::uk_gate_list_pending(std::ptr::null_mut(), 0);
    if needed < 0 {
        return Err(format!("uk_gate_list_pending probe failed: {needed}"));
    }
    if needed == 0 {
        return Ok(b"[]".to_vec());
    }
    let mut buf = vec![0u8; needed as usize];
    let n = unfer_ffi::uk_gate_list_pending(buf.as_mut_ptr(), needed);
    if n < 0 {
        return Err(format!("uk_gate_list_pending receive failed: {n}"));
    }
    // Normalize: the contract is `[[handle, record], ...]`; anything else serves
    // an empty queue rather than leaking a kernel-format drift.
    match serde_json::from_str::<Value>(&String::from_utf8(buf).map_err(|e| e.to_string())?) {
        Ok(v @ Value::Array(_)) => serde_json::to_vec(&v).map_err(|e| e.to_string()),
        _ => Ok(b"[]".to_vec()),
    }
}

fn resolve(handle: i64, f: impl Fn(i64) -> i64) -> Result<Vec<u8>, String> {
    // Attribution: the resolution is a human console action.
    unfer_ffi::uk_set_caller(GATE_OPERATOR).map_err(|e| e.to_string())?;
    let rc = f(handle);
    unfer_ffi::uk_clear_caller();
    match rc {
        // 0 == resolved (approved or rejected).
        0 => serde_json::to_vec(&serde_json::json!({ "ok": true, "handle": handle }))
            .map_err(|e| e.to_string()),
        // Negative values are `-UK-####` codes: 4004 not found, 4005 already resolved.
        other => serde_json::to_vec(&serde_json::json!({
            "ok": false,
            "handle": handle,
            "code": -other,
            "error": format!("gatekeeper refused ({})", -other)
        }))
        .map_err(|e| e.to_string()),
    }
}

/// `POST /api/gate/approve` payload.
pub fn approve_body(handle: i64) -> Result<Vec<u8>, String> {
    resolve(handle, |h| unfer_ffi::uk_gate_approve(h))
}

/// `POST /api/gate/reject` payload.
pub fn reject_body(handle: i64) -> Result<Vec<u8>, String> {
    resolve(handle, |h| unfer_ffi::uk_gate_reject(h))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::CString;
    use std::sync::{Mutex, OnceLock};

    use super::*;

    /// The action/audit stores are process-global (one kernel per process), so
    /// mutation tests must run one at a time.
    pub(crate) fn store_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn submit_action(effect: &str) -> i64 {
        let req = serde_json::json!({
            "principal": "edge_probe",
            "effect": effect,
            "params": { "msg": "gate console test" },
            "provisional": { "forecast": true, "threat": "none" }
        })
        .to_string();
        let c = CString::new(req).expect("NUL-free json");
        unfer_ffi::uk_action_submit(c.as_ptr() as *const u8, c.as_bytes().len() as i64)
    }

    fn pending_list() -> Vec<Value> {
        let body = pending_list_body().expect("pending listing must succeed");
        serde_json::from_slice(&body).expect("pending body must be valid JSON")
    }

    /// The full action record through the two-call buffer protocol.
    fn action_get(handle: i64) -> Value {
        let needed = unfer_ffi::uk_action_get(handle, std::ptr::null_mut(), 0);
        assert!(needed >= 0, "uk_action_get probe failed: {needed}");
        let mut buf = vec![0u8; needed as usize];
        let n = unfer_ffi::uk_action_get(handle, buf.as_mut_ptr(), needed);
        assert!(n >= 0, "uk_action_get receive failed: {n}");
        serde_json::from_slice(&buf).expect("record must be valid JSON")
    }

    #[test]
    fn gate_path_detection() {
        assert!(is_gate_path("/api/gate/pending"));
        assert!(is_gate_path("/api/gate/approve"));
        assert!(is_gate_path("/api/gate/reject"));
        assert!(!is_gate_path("/api/gate/"));
        assert!(!is_gate_path("/audit"));
        assert!(!is_gate_path("/"));
    }

    #[test]
    fn pending_approve_reject_console_roundtrip() {
        let _lock = store_lock().lock().unwrap_or_else(|e| e.into_inner());
        unfer_ffi::uk_audit_clear();
        unfer_ffi::uk_clear_caller();

        let handle = submit_action("notify_admins");
        assert!(handle > 0, "submission must succeed, got {handle}");

        // The pending console lists [handle, record] with the forecast intact.
        let arr = pending_list();
        assert_eq!(
            arr.len(),
            1,
            "queue must carry exactly the submission: {arr:?}"
        );
        assert_eq!(arr[0][0], serde_json::json!(handle));
        assert_eq!(arr[0][1]["effect"], "notify_admins");
        assert_eq!(arr[0][1]["state"], "pending");

        // Operator approves; the console acknowledges and the queue empties.
        let ok = approve_body(handle).expect("approve must serialize");
        assert_eq!(serde_json::from_slice::<Value>(&ok).unwrap()["ok"], true);
        let pending = pending_list();
        assert!(
            pending.is_empty(),
            "queue must be empty after approval: {pending:?}"
        );

        // Approving the same action twice is UK-4005.
        let again = approve_body(handle).expect("double approve must serialize");
        let v: Value = serde_json::from_slice(&again).unwrap();
        assert_eq!(v["code"], 4005);
    }

    #[test]
    fn reject_returns_unknown_handle_error() {
        let _lock = store_lock().lock().unwrap_or_else(|e| e.into_inner());
        unfer_ffi::uk_clear_caller();
        let body = reject_body(999_999).expect("refusal must still serialize");
        let v: Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(v["ok"], false);
        assert_eq!(v["code"], 4004, "unknown action must read UK-4004");
    }

    #[test]
    fn reject_discards_and_frees_queue() {
        let _lock = store_lock().lock().unwrap_or_else(|e| e.into_inner());
        unfer_ffi::uk_audit_clear();
        unfer_ffi::uk_clear_caller();
        let handle = submit_action("rotate_keys");
        assert!(handle > 0);

        let body = reject_body(handle).expect("reject must serialize");
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["ok"], true);

        let record = action_get(handle);
        assert_eq!(record["state"], "rejected");
        assert!(
            pending_list().is_empty(),
            "rejected action leaves the pending console"
        );
    }
}
