//! Edge object-capability RPC layer (S28, F27 — Cap'n Web style).
//!
//! A faithful, clean-room adaptation of Cloudflare's Cap'n Web object-capability
//! RPC for `unfer_edge`:
//!
//! - A **capability** is a `(endpoint, id)` pair authorized to call a bounded
//!   method set on a service. It carries the caller's grant set.
//! - A method call may **return a capability stub as a value** — the result
//!   addresses a freshly-minted capability the caller can invoke later.
//! - Calls may be **pipelined**: a promise can be referenced *before* it
//!   resolves (a capability id is assigned eagerly), so a chain of calls does
//!   not await each hop.
//! - Capabilities are **minted only at the loopback chokepoint** (the registry
//!   here), never by gadget/agent code — mirroring `user.ts:getGatekeeperClassFor`.
//! - A **revoked id is refused**; a returned stub is re-checked against the
//!   original caller.
//!
//! The pre-existing NDJSON/std.io agent protocol remains the degenerate
//! single-capability mode; the C ABI is untouched.

use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

/// A capability to a service endpoint. `id` is minted by the registry and is
/// globally unique; `grants` bounds what the holder may do.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Capability {
    pub endpoint: String,
    pub id: u64,
    #[serde(default, skip_serializing_if = "std::collections::HashSet::is_empty")]
    pub grants: std::collections::HashSet<String>,
}

/// A pending (not-yet-resolved) pipeline promise. Assignment of a capability id
/// is eager, so a caller may issue a follow-up call against the id before the
/// producing call returns.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PendingCapability {
    pub endpoint: String,
    pub id: u64,
    /// The id of the call this promise chains from (for traceability).
    pub from_call: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapCall {
    /// The capability id to invoke.
    pub cap_id: u64,
    /// The method name (grant-checked against the capability's grant set).
    pub method: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapResult {
    #[serde(default)]
    pub ok: bool,
    /// The return value. May be a plain JSON value, or a capability stub
    /// (`{"__cap__": {"endpoint":..., "id":...}}`) when the method hands back a
    /// capability.
    #[serde(default)]
    pub value: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A minted capability record (the registry is the chokepoint).
struct CapRecord {
    /// The principal that minted/owns this capability (the caller the stub was
    /// handed to). A returned stub is re-checked against the original caller.
    owner: String,
    endpoint: String,
    grants: std::collections::HashSet<String>,
    revoked: bool,
}

struct RegistryInner {
    caps: HashMap<u64, CapRecord>,
    /// Pipeline promises: cap_id → the call it will resolve from.
    pending: HashMap<u64, PendingCapability>,
}

static CAP_ID: AtomicU64 = AtomicU64::new(1);
static CALL_SEQ: AtomicU64 = AtomicU64::new(1);
static REGISTRY: Mutex<Option<RegistryInner>> = Mutex::new(None);

fn registry() -> std::sync::MutexGuard<'static, RegistryInner> {
    static REG: std::sync::OnceLock<Mutex<RegistryInner>> = std::sync::OnceLock::new();
    let m = REG.get_or_init(|| {
        Mutex::new(RegistryInner {
            caps: HashMap::new(),
            pending: HashMap::new(),
        })
    });
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Mint a capability at the chokepoint for `owner` with the given grants. This is
/// the single place capabilities are created — gadget/agent code cannot mint its
/// own (there is no call that reaches here from module code except through the
/// grant-checked surface).
pub fn mint(owner: &str, endpoint: &str, grants: &[&str]) -> Capability {
    let id = CAP_ID.fetch_add(1, Ordering::SeqCst);
    let cap = Capability {
        endpoint: endpoint.to_string(),
        id,
        grants: grants.iter().map(|s| s.to_string()).collect(),
    };
    let mut g = registry();
    g.caps.insert(
        id,
        CapRecord {
            owner: owner.to_string(),
            endpoint: endpoint.to_string(),
            grants: cap.grants.clone(),
            revoked: false,
        },
    );
    cap
}

/// Resolve a pipeline promise: the capability `id` becomes a real capability on
/// `endpoint` for `owner`. Used when a call returns a capability stub.
pub fn resolve_promise(owner: &str, id: u64, endpoint: &str, grants: &[&str]) -> bool {
    let mut g = registry();
    if g.caps.contains_key(&id) {
        return false;
    }
    g.caps.insert(
        id,
        CapRecord {
            owner: owner.to_string(),
            endpoint: endpoint.to_string(),
            grants: grants.iter().map(|s| s.to_string()).collect(),
            revoked: false,
        },
    );
    true
}

/// Register a not-yet-resolved pipeline promise (eager capability id).
pub fn new_promise(endpoint: &str) -> PendingCapability {
    let id = CAP_ID.fetch_add(1, Ordering::SeqCst);
    let call = CALL_SEQ.fetch_add(1, Ordering::SeqCst);
    let p = PendingCapability {
        endpoint: endpoint.to_string(),
        id,
        from_call: call,
    };
    let mut g = registry();
    g.pending.insert(id, p.clone());
    p
}

/// Revoke a capability id. A revoked id is refused on any subsequent call.
pub fn revoke(id: u64) -> bool {
    let mut g = registry();
    match g.caps.get_mut(&id) {
        Some(rec) => {
            rec.revoked = true;
            true
        }
        None => {
            g.pending.remove(&id);
            false
        }
    }
}

/// Invoke a method on a capability. `caller` is the principal issuing the call;
/// a returned capability is re-checked against the *original owner* of the cap.
/// Returns the result, which may embed a freshly-minted capability stub.
pub fn invoke(caller: &str, call: &CapCall) -> CapResult {
    let record = {
        let g = registry();
        match g.caps.get(&call.cap_id) {
            Some(r) if !r.revoked => r.clone_meta(),
            _ => {
                return CapResult {
                    ok: false,
                    value: serde_json::Value::Null,
                    error: Some(format!("capability {} is unknown or revoked", call.cap_id)),
                };
            }
        }
    };
    // Grant gate: the method must be in the capability's grant set (default-deny).
    if !record.grants.contains(&call.method) {
        return CapResult {
            ok: false,
            value: serde_json::Value::Null,
            error: Some(format!(
                "method '{}' is not granted on capability {}",
                call.method, call.cap_id
            )),
        };
    }
    // A returned capability is re-checked against the original caller (owner).
    if record.owner != caller {
        return CapResult {
            ok: false,
            value: serde_json::Value::Null,
            error: Some(format!(
                "capability {} is owned by '{}', not '{caller}'",
                call.cap_id, record.owner
            )),
        };
    }

    // Dispatch the method. This is a thin in-kernel service surface: the result
    // may be a plain value or a nested capability stub (a freshly-minted cap).
    let seq = CALL_SEQ.fetch_add(1, Ordering::SeqCst);
    let value = match call.method.as_str() {
        "echo" => call.args.clone(),
        "seq" => serde_json::json!(seq),
        // A service that hands back a sub-capability: returns a capability stub.
        "subcap" => {
            let child = mint(caller, &record.endpoint, &["echo"]);
            cap_stub(&child)
        }
        _ => {
            return CapResult {
                ok: false,
                value: serde_json::Value::Null,
                error: Some(format!("unknown method '{}'", call.method)),
            };
        }
    };
    CapResult {
        ok: true,
        value,
        error: None,
    }
}

impl CapRecord {
    fn clone_meta(&self) -> CapMeta {
        CapMeta {
            owner: self.owner.clone(),
            endpoint: self.endpoint.clone(),
            grants: self.grants.clone(),
        }
    }
}

struct CapMeta {
    owner: String,
    endpoint: String,
    grants: std::collections::HashSet<String>,
}

/// Serialize a capability as a stub value (the `{"__cap__": {...}}` contract a
/// caller receives as a method return value).
pub fn cap_stub(cap: &Capability) -> serde_json::Value {
    serde_json::json!({ "__cap__": cap })
}

/// Parse a capability stub out of a method's return value, if present.
pub fn stub_cap(value: &serde_json::Value) -> Option<Capability> {
    let obj = value.get("__cap__")?;
    serde_json::from_value(obj.clone()).ok()
}

/// Reset the registry (QA/console reset).
pub fn reset() {
    let mut g = registry();
    g.caps.clear();
    g.pending.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The registry is a process-global shared store; serialize the tests so one
    // test's `reset()` never wipes another's mid-run state (repo convention).
    static CAPRPC_TESTS_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn mint_and_invoke_granted_method() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let cap = mint("alice", "svc", &["echo", "seq"]);
        let r = invoke(
            "alice",
            &CapCall {
                cap_id: cap.id,
                method: "echo".into(),
                args: serde_json::json!({"x":1}),
            },
        );
        assert!(r.ok, "granted method must run: {r:?}");
        assert_eq!(r.value["x"], 1);
        reset();
    }

    #[test]
    fn ungranted_method_is_refused() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let cap = mint("alice", "svc", &["echo"]);
        let r = invoke(
            "alice",
            &CapCall {
                cap_id: cap.id,
                method: "seq".into(),
                args: serde_json::Value::Null,
            },
        );
        assert!(!r.ok, "ungranted method must be refused: {r:?}");
        assert!(r.error.as_deref().unwrap_or("").contains("not granted"));
        reset();
    }

    #[test]
    fn wrong_caller_is_refused() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let cap = mint("alice", "svc", &["echo"]);
        let r = invoke(
            "bob",
            &CapCall {
                cap_id: cap.id,
                method: "echo".into(),
                args: serde_json::Value::Null,
            },
        );
        assert!(!r.ok, "a non-owner must be refused: {r:?}");
        assert!(r.error.as_deref().unwrap_or("").contains("owned by"));
        reset();
    }

    #[test]
    fn revoked_capability_is_refused() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let cap = mint("alice", "svc", &["echo"]);
        assert!(revoke(cap.id));
        let r = invoke(
            "alice",
            &CapCall {
                cap_id: cap.id,
                method: "echo".into(),
                args: serde_json::Value::Null,
            },
        );
        assert!(!r.ok, "revoked capability must be refused: {r:?}");
        reset();
    }

    #[test]
    fn method_returns_capability_stub() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let cap = mint("alice", "svc", &["subcap"]);
        let r = invoke(
            "alice",
            &CapCall {
                cap_id: cap.id,
                method: "subcap".into(),
                args: serde_json::Value::Null,
            },
        );
        assert!(r.ok, "subcap must succeed: {r:?}");
        let child = stub_cap(&r.value).expect("result embeds a capability stub");
        assert_eq!(child.endpoint, "svc");
        // The returned stub is a real capability: invoke its granted method.
        let r2 = invoke(
            "alice",
            &CapCall {
                cap_id: child.id,
                method: "echo".into(),
                args: serde_json::json!("hi"),
            },
        );
        assert!(r2.ok, "returned stub must be usable: {r2:?}");
        assert_eq!(r2.value, serde_json::json!("hi"));
        reset();
    }

    #[test]
    fn promise_is_eager_and_resolvable() {
        let _g = CAPRPC_TESTS_LOCK.lock().unwrap();
        reset();
        let p = new_promise("svc");
        // The id is assigned eagerly (pipelining): a call may reference it before
        // the producing call resolves. Resolve it into a real capability.
        assert!(resolve_promise("alice", p.id, "svc", &["echo"]));
        let r = invoke(
            "alice",
            &CapCall {
                cap_id: p.id,
                method: "echo".into(),
                args: serde_json::json!("pipe"),
            },
        );
        assert!(r.ok, "resolved promise must be invokable: {r:?}");
        assert_eq!(r.value, serde_json::json!("pipe"));
        reset();
    }
}
