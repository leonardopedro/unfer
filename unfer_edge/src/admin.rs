//! Admin console (S22, F21): soft/hard separation for `unfer_edge` configuration.
//!
//! cloudflare-os ships an `AdminApi` whose *product* settings are console-editable
//! while **auth, grants, and storage are host-global only**. This module adopts that
//! split for the edge:
//!
//! * `GET   /admin/status`           — admin-gated snapshot: soft config + the hard
//!   config shadow (which surfaces are immutable, never their secrets).
//! * `PATCH /admin/config`           — admin-gated patch of `soft_config.json`
//!   (banner, announcements, resource availability modes). The merged value is
//!   mirrored in one KV-style key (`soft_config.json`) in the process store.
//!
//! Hardening:
//! * The admin capability is minted **exactly once** at session start from the
//!   host environment (`UNFER_ADMIN_PRINCIPAL`, default `operator`) — the app never
//!   mints a new admin from a request.
//! * A non-admin principal is answered `403` on both routes.
//! * A `PATCH` naming a hard key (`grants`, `auth`, `storage`, `backend`) is refused
//!   (UK-style 400 with `error`), and never mutates the soft config.

use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

/// Hard-config key namespace — never user-editable. Soft config in `soft_config.json`
/// can advertise or announce anything, but it cannot shadow these host-global keys.
const HARD_KEYS: &[&str] = &["grants", "auth", "storage", "backend"];

/// The one KV-style key the soft config is mirrored under (the process "store").
const SOFT_CONFIG_KEY: &str = "soft_config.json";

/// Default soft config: every field is product-surface (console editable).
fn default_soft() -> Value {
    json!({
        "banner": "",
        "announcements": [],
        "offered_resources": ["cpu", "cell-gateway"],
        "resource_availability": "open",
    })
}

/// Process-global soft-config mirror. `OnceLock` initializes the defaults on first
/// use; patches replace the value wholesale (single KV-style key update).
fn soft_store() -> &'static Mutex<Value> {
    static CONFIG: OnceLock<Mutex<Value>> = OnceLock::new();
    CONFIG.get_or_init(|| Mutex::new(default_soft()))
}

/// The single minted admin principal. Read from the host environment every call so
/// tests can reshape it under the serial lock; in production it is fixed at startup.
pub fn admin_principal() -> String {
    std::env::var("UNFER_ADMIN_PRINCIPAL").unwrap_or_else(|_| "operator".to_string())
}

/// True when `principal` is the minted admin capability.
pub fn is_admin(principal: &str) -> bool {
    principal == admin_principal()
}

/// True when `path` names the admin console. The caller dispatches on method+suffix.
pub fn is_admin_path(path: &str) -> bool {
    path == "/admin/status" || path == "/admin/config"
}

/// The `403` envelope for a non-admin principal. Same body for both routes.
fn admin_required(principal: &str) -> (u16, Vec<u8>) {
    let body = json!({
        "error": "admin required",
        "code": 403,
        "who": principal,
        "hint": "the admin capability is minted once at session start; the app grants none"
    });
    (
        403u16,
        serde_json::to_vec(&body).expect("error body serializes"),
    )
}

/// `GET /admin/status`: admin-gated snapshot of the soft config + the hard-layer
/// shadow. The hard shadow names the immutable keys but never their values.
pub fn status_body(principal: &str) -> (u16, Vec<u8>) {
    if !is_admin(principal) {
        return admin_required(principal);
    }
    let config = soft_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let body = json!({
        "admin": true,
        "admin_principal": principal,
        "config_key": SOFT_CONFIG_KEY,
        "soft_config": config,
        "hard_config": {
            "editable": false,
            "keys": HARD_KEYS,
            "note": "auth, grants, and storage are host-global env / Rust config only"
        },
        "version": unfer_ffi::uk_version(),
    });
    (
        200u16,
        serde_json::to_vec(&body).expect("status body serializes"),
    )
}

/// `PATCH /admin/config`: admin-gated patch of the soft config. Refused (400) if any
/// top-level key would enter the hard namespace — an attempt to change grants/auth or
/// other host-global config leaves `soft_config.json` byte-identical.
pub fn patch_body(principal: &str, patch: &[u8]) -> (u16, Vec<u8>) {
    if !is_admin(principal) {
        return admin_required(principal);
    }
    let patch: Value = match serde_json::from_slice(patch) {
        Ok(v) => v,
        Err(_) => {
            let body = json!({ "error": "expects a JSON object body", "code": 400 });
            return (400u16, serde_json::to_vec(&body).expect("serializes"));
        }
    };
    let patch_obj = match patch.as_object() {
        Some(o) => o,
        None => {
            let body = json!({ "error": "expects a JSON object body", "code": 400 });
            return (400u16, serde_json::to_vec(&body).expect("serializes"));
        }
    };
    if patch_obj.keys().any(|k| HARD_KEYS.contains(&k.as_str())) {
        let soft = soft_store().lock().unwrap_or_else(|e| e.into_inner());
        let body = json!({
            "error": "hard keys are never user-editable",
            "code": 400,
            "refused_keys": patch_obj.keys().filter(|k| HARD_KEYS.contains(&k.as_str())).collect::<Vec<_>>(),
            "soft_config_unchanged": *soft,
        });
        return (400u16, serde_json::to_vec(&body).expect("serializes"));
    }

    let mut config = soft_store().lock().unwrap_or_else(|e| e.into_inner());
    let existing = (*config).as_object().cloned().unwrap_or_default();
    let mut merged = existing;
    for (k, v) in patch_obj {
        merged.insert(k.clone(), v.clone());
    }
    *config = Value::Object(merged);
    let body = json!({ "ok": true, "updated_key": SOFT_CONFIG_KEY, "soft_config": *config });
    (200u16, serde_json::to_vec(&body).expect("serializes"))
}

/// Reset the soft config (host/QA path). Returns the old value. QA-only: the
/// live admin surface never resets the config.
#[cfg(test)]
pub fn reset_soft_config() -> Value {
    let mut config = soft_store().lock().unwrap_or_else(|e| e.into_inner());
    let old = config.clone();
    *config = default_soft();
    old
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_admin<R>(principal: &str, f: impl FnOnce() -> R) -> R {
        let _lock = crate::gate::tests::store_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("UNFER_ADMIN_PRINCIPAL", principal);
        }
        let out = f();
        unsafe {
            std::env::remove_var("UNFER_ADMIN_PRINCIPAL");
        }
        out
    }

    #[test]
    fn admin_path_detection() {
        assert!(is_admin_path("/admin/status"));
        assert!(is_admin_path("/admin/config"));
        assert!(!is_admin_path("/admin"));
        assert!(!is_admin_path("/audit"));
        assert!(!is_admin_path("/api/gate/pending"));
    }

    #[test]
    fn status_is_admin_gated() {
        with_admin("soul", || {
            let (status, body) = status_body("soul");
            assert_eq!(status, 200);
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert!(v["admin"] == json!(true));
            assert_eq!(v["admin_principal"], "soul");
            assert!(v["config_key"] == json!(SOFT_CONFIG_KEY));
            assert!(v["hard_config"]["editable"] == json!(false));
            assert_eq!(v["hard_config"]["keys"], json!(HARD_KEYS));

            // A different principal is refused outright (403 gate).
            let (status, body) = status_body("impostor");
            assert_eq!(status, 403);
            assert!(serde_json::from_slice::<Value>(&body).unwrap()["code"] == json!(403));
        });
    }

    #[test]
    fn patch_updates_soft_config() {
        with_admin("soul", || {
            reset_soft_config();
            let (status, body) = patch_body("soul", b"{\"banner\":\"quantum season\"}");
            assert_eq!(status, 200, "{body:?}");
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["soft_config"]["banner"], "quantum season");
            assert!(v["updated_key"] == json!(SOFT_CONFIG_KEY));

            // Mirrored in the status snapshot: the same KV-style key reflects the patch.
            let (_, body) = status_body("soul");
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["soft_config"]["banner"], "quantum season");

            // Malformed body is a 400, not a mutation.
            let (status, _) = patch_body("soul", b"not json");
            assert_eq!(status, 400);
        });
    }

    #[test]
    fn patch_refuses_hard_keys() {
        with_admin("soul", || {
            reset_soft_config();
            let patch = b"{\"grants\":{\"kernel\":[\"uk_evolve\"]}}";
            let (status, body) = patch_body("soul", patch);
            assert_eq!(status, 400, "{body:?}");
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["code"], json!(400));
            assert_eq!(v["refused_keys"], json!(["grants"]));
            let before: bool = v["soft_config_unchanged"].as_object().is_some();

            // A subsequent status still carries no grants/auth/storage/backend surface.
            let (_, body) = status_body("soul");
            let v: Value = serde_json::from_slice(&body).unwrap();
            for k in HARD_KEYS {
                assert_eq!(
                    v["soft_config"].get(*k),
                    None,
                    "{k} leaked into soft config"
                );
            }
            assert!(before, "unchanged soft config must still be an object");
        });
    }

    #[test]
    fn patch_non_admin_403() {
        with_admin("soul", || {
            patch_body("soul", b"{\"banner\":\"before\"}");
            let (status, body) = patch_body("impostor", b"{\"banner\":\"x\"}");
            assert_eq!(status, 403, "{body:?}");
            assert!(serde_json::from_slice::<Value>(&body).unwrap()["code"] == json!(403));

            // The impostor's patch never lands.
            let (_, body) = status_body("soul");
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["soft_config"]["banner"], "before");
        });
    }
}
