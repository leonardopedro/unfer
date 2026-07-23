//! Request-filter: parses the incoming body as `AgentRequest`, validates the
//! `op` field against a static allowlist, and produces UK-1001/UK-4001
//! rejection `AgentResponse` values before any backend forwarding.

use std::collections::HashSet;

use unfer_protocol::codes::{Code, Diagnostic, HintKind, RepairHint, Severity};
use unfer_protocol::types::{AgentRequest, AgentResponse};

/// Ops the gateway accepts. Anything else receives UK-4001 CallDenied.
static ALLOWED_OPS: &[&str] = &[
    "version",
    "create_model",
    "set_prior",
    "evolve",
    "condition",
    "probability",
    "observe",
    "snapshot",
    "list_codes",
    "did_create",
    "did_resolve",
    "did_update",
    "did_revoke",
    "content_publish",
    "content_resolve",
    "consensus_sync",
    "consensus_status",
];

/// Rejection reasons returned to callers.
#[derive(Debug)]
pub enum Rejection {
    /// Body is not valid JSON `AgentRequest`. → UK-1001
    BadJson(String),
    /// The `op` field is not in the allowlist. → UK-4001
    OpDenied { op: String },
}

impl Rejection {
    pub fn to_response(&self, id: &str) -> AgentResponse {
        match self {
            Rejection::BadJson(msg) => AgentResponse::err(
                id,
                Diagnostic::new(Code::BAD_JSON, msg.clone(), Severity::Error).with_hint(
                    RepairHint::new(
                        HintKind::ReplaceValue,
                        "request body",
                        r#"{"id":"…","op":"<op>","params":{}}"#,
                    ),
                ),
            ),
            Rejection::OpDenied { op } => {
                let allowed = ALLOWED_OPS.join(", ");
                AgentResponse::err(
                    id,
                    Diagnostic::new(
                        Code::CALL_DENIED,
                        format!("op '{op}' is not in the gateway allowlist"),
                        Severity::Error,
                    )
                    .with_hint(RepairHint::new(
                        HintKind::ReplaceValue,
                        "op",
                        format!("one of: {allowed}"),
                    )),
                )
            }
        }
    }
}

/// Validate a raw request body. Returns the parsed `AgentRequest` on success
/// or a typed `Rejection` on failure.
pub fn validate_request(body: &[u8]) -> Result<AgentRequest, Rejection> {
    let req: AgentRequest =
        serde_json::from_slice(body).map_err(|e| Rejection::BadJson(e.to_string()))?;

    if !ALLOWED_OPS.contains(&req.op.as_str()) {
        return Err(Rejection::OpDenied { op: req.op.clone() });
    }

    Ok(req)
}

/// Snapshot of the gateway allowlist for health/introspection endpoints.
pub fn allowed_ops() -> HashSet<&'static str> {
    ALLOWED_OPS.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_request_passes() {
        let body = br#"{"id":"1","op":"version","params":{}}"#;
        let req = validate_request(body).unwrap();
        assert_eq!(req.id, "1");
        assert_eq!(req.op, "version");
    }

    #[test]
    fn bad_json_gives_uk1001() {
        let body = b"not json";
        let err = validate_request(body).unwrap_err();
        assert!(matches!(err, Rejection::BadJson(_)));
        let resp = err.to_response("x");
        assert!(!resp.ok);
        assert_eq!(resp.error.as_ref().unwrap().code, Code::BAD_JSON);
    }

    #[test]
    fn denied_op_gives_uk4001() {
        let body = br#"{"id":"2","op":"__internal_reset","params":{}}"#;
        let err = validate_request(body).unwrap_err();
        assert!(matches!(err, Rejection::OpDenied { .. }));
        let resp = err.to_response("2");
        assert!(!resp.ok);
        assert_eq!(resp.error.as_ref().unwrap().code, Code::CALL_DENIED);
        // Hint must list valid ops so the caller can self-repair.
        let hint = &resp.error.as_ref().unwrap().hints[0];
        assert!(hint.suggestion.contains("version"));
    }

    #[test]
    fn all_allowed_ops_pass() {
        for op in ALLOWED_OPS {
            let body = format!(r#"{{"id":"x","op":"{op}","params":{{}}}}"#);
            assert!(
                validate_request(body.as_bytes()).is_ok(),
                "expected op '{op}' to be allowed"
            );
        }
    }

    #[test]
    fn allowlist_has_no_duplicates() {
        let set: HashSet<_> = ALLOWED_OPS.iter().copied().collect();
        assert_eq!(
            set.len(),
            ALLOWED_OPS.len(),
            "ALLOWED_OPS must not have duplicates"
        );
    }
}
