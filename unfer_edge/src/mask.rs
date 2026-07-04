//! Data-masking / secret-injection protection for `AgentRequest`/`AgentResponse`
//! JSON envelopes, re-implementing the spirit of zentinel's `data-masking` and
//! `secret-inject` agent modules as a Pingora response filter (P11.22).
//!
//! Any JSON object key that looks like a credential (`api_key`, `secret`,
//! `token`, `password`, `authorization`, ...) has its string value replaced
//! with a fixed-width redaction marker before the body leaves the gateway.
//! This guards against a misbehaving backend accidentally echoing a secret
//! back in `AgentResponse.result` or `AgentResponse.error.data`.

use serde_json::Value;

const REDACTED: &str = "***REDACTED***";

/// Key fragments (lower-cased, substring match) that mark a field as
/// sensitive. Substring match catches variants like `zenodo_api_key` or
/// `bearer_token`.
const SENSITIVE_KEY_FRAGMENTS: &[&str] = &[
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "credential",
];

fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SENSITIVE_KEY_FRAGMENTS
        .iter()
        .any(|frag| lower.contains(frag))
}

/// Recursively walk a JSON value, redacting the string value of any object
/// key that matches [`is_sensitive_key`]. Arrays and nested objects are
/// walked in place.
pub fn mask_secrets(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if is_sensitive_key(key) && v.is_string() {
                    *v = Value::String(REDACTED.to_string());
                } else {
                    mask_secrets(v);
                }
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                mask_secrets(item);
            }
        }
        _ => {}
    }
}

/// Mask secrets in a raw JSON byte body. Non-JSON or non-object/array bodies
/// pass through unchanged (they carry no keyed fields to redact).
pub fn mask_body(body: &[u8]) -> Vec<u8> {
    match serde_json::from_slice::<Value>(body) {
        Ok(mut v) => {
            mask_secrets(&mut v);
            serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec())
        }
        Err(_) => body.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_top_level_secret_field() {
        let mut v = json!({"id": "1", "api_key": "sk-live-abc123"});
        mask_secrets(&mut v);
        assert_eq!(v["api_key"], REDACTED);
        assert_eq!(v["id"], "1");
    }

    #[test]
    fn redacts_nested_secret_field() {
        let mut v = json!({
            "ok": true,
            "result": {"config": {"zenodo_api_key": "deadbeef"}}
        });
        mask_secrets(&mut v);
        assert_eq!(v["result"]["config"]["zenodo_api_key"], REDACTED);
    }

    #[test]
    fn redacts_secret_inside_array() {
        let mut v = json!({
            "items": [
                {"token": "abc"},
                {"token": "def"}
            ]
        });
        mask_secrets(&mut v);
        assert_eq!(v["items"][0]["token"], REDACTED);
        assert_eq!(v["items"][1]["token"], REDACTED);
    }

    #[test]
    fn leaves_non_sensitive_fields_untouched() {
        let mut v = json!({"id": "1", "op": "version", "params": {"count": 4}});
        let original = v.clone();
        mask_secrets(&mut v);
        assert_eq!(v, original);
    }

    #[test]
    fn mask_body_roundtrips_non_json_unchanged() {
        let raw = b"not json at all";
        assert_eq!(mask_body(raw), raw.to_vec());
    }

    #[test]
    fn mask_body_redacts_agent_response_shape() {
        let raw = br#"{"id":"1","ok":true,"result":{"api_key":"leaked"},"error":null}"#;
        let masked = mask_body(raw);
        let v: Value = serde_json::from_slice(&masked).unwrap();
        assert_eq!(v["result"]["api_key"], REDACTED);
    }
}
