//! Audit console (S6, F6): exposes the kernel's audit trail over HTTP so the
//! initiating human can review what agents/gadgets attempted.
//!
//! The trail lives in the kernel (`unfer_ffi::uk_audit_list` / `uk_audit_clear`).
//! With `--features audit` the edge links the kernel FFI in-process and serves:
//!
//! * `GET  /audit` — the audit trail, newest first (`AuditEntry[]`).
//! * `DELETE /audit` — clear the trail (an operator action).
//!
//! In a split deployment (edge and kernel in separate processes) the operator
//! proxies `/audit` to the kernel host instead; this module is the self-contained
//! console for the embedded-kernel case.

use serde_json::Value;

/// True when `path` names the audit console (method-agnostic — the caller
/// dispatches on the HTTP method).
pub fn is_audit_path(path: &str) -> bool {
    path == "/audit"
}

/// `GET /audit` payload: the kernel audit trail as a JSON array (newest first).
/// Returns the raw body bytes on success, an error message otherwise.
pub fn audit_list_body() -> Result<Vec<u8>, String> {
    let needed = unfer_ffi::uk_audit_list(std::ptr::null_mut(), 0);
    if needed < 0 {
        return Err(format!("uk_audit_list failed: {needed}"));
    }
    if needed == 0 {
        return Ok(b"[]".to_vec());
    }
    let mut buf = vec![0u8; needed as usize];
    let n = unfer_ffi::uk_audit_list(buf.as_mut_ptr(), needed);
    if n < 0 {
        return Err(format!("uk_audit_list failed: {n}"));
    }
    // Serve valid JSON regardless of kernel-format drift; trim to the reported length.
    let body = buf;
    let text = String::from_utf8(body).map_err(|e| e.to_string())?;
    // Normalize: if the kernel returned something non-array, wrap it so the HTTP
    // contract stays `AuditEntry[]`.
    match serde_json::from_str::<Value>(&text) {
        Ok(v @ Value::Array(_)) => serde_json::to_vec(&v).map_err(|e| e.to_string()),
        _ => Ok(b"[]".to_vec()),
    }
}

/// `DELETE /audit` payload: the number of entries removed.
pub fn audit_clear_count() -> Result<Vec<u8>, String> {
    let removed = unfer_ffi::uk_audit_clear();
    if removed < 0 {
        return Err(format!("uk_audit_clear failed: {removed}"));
    }
    serde_json::to_vec(&serde_json::json!({ "removed": removed })).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_path_detection() {
        assert!(is_audit_path("/audit"));
        assert!(!is_audit_path("/kernel/uk_version"));
        assert!(!is_audit_path("/audit/"));
        assert!(!is_audit_path(""));
    }

    #[test]
    fn audit_list_body_roundtrips_entries() {
        // Clear, then append tagged entries through the kernel, then read them back
        // through the HTTP payload path.
        unfer_ffi::uk_audit_clear();
        let caller = r#"{"from":"agent","principal":"edge_probe"}"#;
        unfer_ffi::uk_set_caller(caller).expect("caller json must parse");
        let _ =
            unfer_ffi::uk_audit_append(r#"{"symbol":"uk_evolve","args":[{"t":0.1}],"ok":true}"#);
        unfer_ffi::uk_clear_caller();

        let body = audit_list_body().expect("audit listing must succeed");
        let entries: Value = serde_json::from_slice(&body).expect("body must be valid JSON");
        let arr = entries.as_array().expect("audit payload must be an array");
        assert_eq!(
            arr.len(),
            1,
            "expected exactly 1 entry after clear, got {entries}"
        );
        assert_eq!(arr[0]["symbol"], "uk_evolve");
        assert_eq!(arr[0]["caller"]["from"], "agent");
        assert_eq!(arr[0]["caller"]["principal"], "edge_probe");

        let cleared = audit_clear_count().expect("clear must succeed");
        let cleared: Value = serde_json::from_slice(&cleared).expect("valid JSON");
        assert!(cleared["removed"].as_i64().unwrap() >= 1);
        let body = audit_list_body().expect("listing must succeed after clear");
        assert_eq!(
            serde_json::from_slice::<Value>(&body)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            0
        );
    }
}
