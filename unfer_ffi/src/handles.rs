use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
};

use prob_kernel::Session;
use unfer_consensus::auction::AuctionLedger;
use unfer_consensus::certs::{CertificateLedger, MintAuthority};
use unfer_protocol::durable::{DurableStore, streams};
use unfer_protocol::{
    ActionRecord, AgentInfo, AuditEntry, CallerTag, CertId, Diagnostic, EffectKind, EventQuery,
    GrantSet, KernelEvent,
};

/// Maximum events retained per subscription before oldest are dropped.
pub const EVENT_QUEUE_CAPACITY: usize = 64;

struct SessionEntry {
    session: Session,
    last_result: String,
}

struct Subscription {
    model_handle: i64,
    query: EventQuery,
    events: VecDeque<String>,
    /// Events dropped at queue capacity that the subscriber never saw.
    /// Recorded where it happens so the operator consult (`uk_owner_list`)
    /// can surface "this subscription fell behind" — a dropped event must be
    /// a recorded, intentional non-outcome, not invisible silence.
    dropped: u64,
}

static HANDLES: Mutex<Option<HashMap<i64, SessionEntry>>> = Mutex::new(None);
static SUBSCRIPTIONS: Mutex<Option<HashMap<i64, Subscription>>> = Mutex::new(None);
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
static NEXT_SUB: AtomicI64 = AtomicI64::new(1);
static INITIALIZED: AtomicBool = AtomicBool::new(false);

thread_local! {
    // Per-thread last-error slot, mirroring C `errno` semantics: a caller reads
    // back the error raised on its own thread without racing other threads
    // (a shared global slot would let one thread's failure clobber another's
    // between the size-probe and copy calls of the buffer protocol).
    static LAST_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

pub fn ensure_init() {
    INITIALIZED.store(true, Ordering::SeqCst);
    if durable().is_none() {
        // Configure from the environment (H4): `UNFER_DURABLE_DIR` (optional;
        // in-memory when unset) and `UNFER_DURABLE_BACKEND` (default Loro).
        let dir = std::env::var("UNFER_DURABLE_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .map(std::path::PathBuf::from);
        let backend = DurableBackend::parse(std::env::var("UNFER_DURABLE_BACKEND").ok().as_deref());
        if let Err(e) = init_durable(dir.as_deref(), backend) {
            ring_append_owner(format!("(kernel.audit) durable init failed: {e}"));
        } else if let Err(e) = recover_durable() {
            ring_append_owner(format!("(kernel.audit) durable recovery failed: {e}"));
        } else {
            // Fail-visible recovery: if the on-disk snapshot was corrupt and
            // the store started empty, the operator learns it at startup —
            // in the owner log (and durably, so the note survives restarts).
            report_snapshot_load_error();
        }
    }
}

// ── H4 durable store registry ───────────────────────────────────────────
//
// The kernel-global [`DurableStore`] backing every ring in this module. The
// in-memory rings are a *read-through cache* in front of it: writes go to the
// store (write-through) and the ring shadows the same records for fast reads.
// A stream lives in exactly one store, so nothing is mirrored across backends.
//
// `recover_durable()` runs at init and replays the persisted streams into the
// rings, so an operator or agent never reads back RAM-only state. Checkpoints
// (`checkpoint()`) are the flush barrier called before model-facing
// probability/condition reads and before `uk_action_apply` may fire a side
// effect.

/// The kernel's durable store, if configured. `None` until `ensure_init`
/// initialized it (see [`init_durable`]).
static DURABLE: Mutex<Option<Arc<dyn DurableStore>>> = Mutex::new(None);

/// Backend selection (H4). Loro is the default/preferred store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableBackend {
    Loro,
    Jsonl,
    #[cfg(feature = "sqlite")]
    Sqlite,
}

impl DurableBackend {
    pub fn parse(s: Option<&str>) -> Self {
        match s.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("jsonl") => DurableBackend::Jsonl,
            #[cfg(feature = "sqlite")]
            Some("sqlite") => DurableBackend::Sqlite,
            _ => DurableBackend::Loro,
        }
    }
}

/// Configure the global durable store from `UNFER_DURABLE_DIR` /
/// `UNFER_DURABLE_BACKEND` (or explicit values). Defaults: Loro backend, no
/// directory (in-memory) when unset. Fail-closed: a misconfigured dir is a
/// hard error (the kernel must not silently fall back to RAM-only).
pub fn init_durable(dir: Option<&std::path::Path>, backend: DurableBackend) -> Result<(), String> {
    let store: Arc<dyn DurableStore> = crate::durable::open_store(
        dir,
        match backend {
            DurableBackend::Loro => crate::durable::Backend::Loro,
            DurableBackend::Jsonl => crate::durable::Backend::Jsonl,
            #[cfg(feature = "sqlite")]
            DurableBackend::Sqlite => crate::durable::Backend::Sqlite,
        },
    )
    .map_err(|e| format!("durable store open ({backend:?}): {e}"))?
    .into();
    *DURABLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(store);
    Ok(())
}

/// The current durable store (read-through cache backing).
pub fn durable() -> Option<Arc<dyn DurableStore>> {
    DURABLE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The fail-closed checkpoint barrier: make every prior append durable. A
/// `Err` means the caller must NOT serve a probability/condition or fire a
/// side effect (fail-closed). `None` store (no durability configured) counts
/// as Ok — the kernel just runs RAM-only.
pub fn checkpoint() -> Result<(), String> {
    match durable() {
        Some(store) => store.flush().map_err(|e| e.to_string()),
        None => Ok(()),
    }
}

/// The FFI fail-closed checkpoint: flush the global durable store. On failure,
/// map to the `UNKNOWN_OUTCOME` diagnostic (UK-1010) so the call is refused
/// before it dispatches.
pub fn checkpoint_diag() -> Result<(), Diagnostic> {
    checkpoint().map_err(|reason| {
        Diagnostic::new(
            unfer_protocol::Code::UNKNOWN_OUTCOME,
            format!("durable checkpoint failed before dispatch: {reason}"),
            unfer_protocol::Severity::Error,
        )
        .with_hint(unfer_protocol::RepairHint::new(
            unfer_protocol::HintKind::SetParam,
            "durable.checkpoint",
            "check the durable store backend (disk full? bad directory?); read-only work may \
             retry, side-effecting work must be verified manually",
        ))
    })
}

/// Replay the durable streams into the in-memory rings (startup recovery).
/// Idempotent: safe to call multiple times. A store failure is surfaced, never
/// swallowed (the operator must know the kernel could not recover).
pub fn recover_durable() -> Result<(), String> {
    let Some(store) = durable() else {
        return Ok(());
    };
    for rec in store.replay(streams::AUDIT).map_err(|e| e.to_string())? {
        if let Ok(entry) = serde_json::from_slice::<AuditEntry>(&rec) {
            ring_append_audit(entry);
        }
    }
    for rec in store
        .replay(streams::OWNER_LOG)
        .map_err(|e| e.to_string())?
    {
        if let Ok(line) = std::str::from_utf8(&rec) {
            ring_append_owner(line.to_string());
        }
    }
    let mut handle_by_id: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut inflight_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in store.replay(streams::ACTIONS).map_err(|e| e.to_string())? {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&rec) else {
            continue;
        };
        let Some(id) = v.get("id").and_then(|x| x.as_str()) else {
            continue;
        };
        if v.get("inflight").and_then(|x| x.as_bool()).unwrap_or(false) {
            // An interrupted side-effecting call: a resolved record may or may
            // not follow this marker.
            inflight_ids.insert(id.to_string());
            continue;
        }
        if let Some(record) = v
            .get("record")
            .and_then(|r| serde_json::from_value::<ActionRecord>(r.clone()).ok())
        {
            let handle = ring_put_action(record);
            handle_by_id.insert(id.to_string(), handle);
            // A resolved record supersedes any earlier in-flight marker.
            inflight_ids.remove(id);
        }
    }
    // Ids whose LAST action-stream record is an in-flight marker (no resolved
    // record after it): the process died between the durable marker and the
    // resolved marker, so the external outcome is unknown (UK-1010).
    for id in inflight_ids {
        if let Some(handle) = handle_by_id.get(&id) {
            mark_unknown_outcome(*handle);
        } else {
            // No record at all: mint a placeholder so the operator can see it.
            let record = ActionRecord::new(
                id.clone(),
                "kernel",
                "unknown",
                serde_json::json!({"unknown_outcome": true}),
                0,
                None,
            );
            let handle = ring_put_action(record);
            mark_unknown_outcome(handle);
        }
    }
    for rec in store.replay(streams::CONFIG).map_err(|e| e.to_string())? {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&rec)
            && let Some(principal) = v.get("vetted_principal").and_then(|x| x.as_str())
        {
            if v.get("vetted").and_then(|x| x.as_bool()).unwrap_or(true) {
                ring_mark_vetted(principal.to_string());
            } else {
                ring_unmark_vetted(principal);
            }
        }
    }
    Ok(())
}

/// Append a record to a stream, keeping the in-memory ring in sync.
fn durable_append(stream: &str, record: &[u8]) {
    if let Some(store) = durable()
        && let Err(e) = store.append(stream, record)
    {
        ring_append_owner(format!(
            "(kernel.audit) durable append to {stream} failed: {e}"
        ));
    }
}

/// H4: durably append a resolved action record and checkpoint (supersedes the
/// in-flight marker). Used by `uk_action_apply` after the effect fires.
///
/// Fail-closed: a checkpoint failure returns `Err` so the caller can mark the
/// action unknown-outcome (the durable store still holds the in-flight marker
/// with no resolved record after it — the crash window UK-1010 describes).
pub fn durable_append_action_resolved(record: &ActionRecord) -> Result<(), String> {
    if let Ok(json) = serde_json::to_vec(&serde_json::json!({
        "id": record.id,
        "record": record,
    })) {
        durable_append(streams::ACTIONS, &json);
    }
    checkpoint().map_err(|e| {
        format!(
            "durable resolved record for action {} could not be flushed: {e}",
            record.id
        )
    })
}

/// Surface a corrupt-snapshot recovery (if any) in the operator-facing owner
/// log. Write-through: the audit line is durably appended (so the operator
/// note survives restarts) and shadowed in the ring. Returns the error for
/// testability; `None` = clean open (or no store configured).
pub fn report_snapshot_load_error() -> Option<String> {
    let err = durable().as_ref()?.snapshot_load_error()?;
    owner_log(
        "kernel.audit",
        &format!("durable snapshot recovered from corruption: {err}"),
    );
    Some(err)
}

/// Durable live status: backend label, per-stream record counts, and the
/// backend's persist counter — the kernel-side equivalent of Lody's
/// session-live-status, without replaying any history. `backend` is
/// `"none"` when no durable store is configured (the kernel runs RAM-only).
pub fn durable_status_json() -> String {
    let Some(store) = durable() else {
        // Stable schema even RAM-only: every well-known stream reports 0 and
        // the backend is "none", so a host never special-cases the shape.
        let mut streams = serde_json::Map::new();
        for name in STREAM_NAMES {
            streams.insert(name.to_string(), serde_json::json!(0));
        }
        return serde_json::json!({
            "backend": "none",
            "streams": streams,
            "persist_count": 0,
            "snapshot_load_error": null,
        })
        .to_string();
    };
    let mut streams = serde_json::Map::new();
    for name in STREAM_NAMES {
        streams.insert(
            name.to_string(),
            serde_json::json!(store.stream_len(name).unwrap_or(u64::MAX)),
        );
    }
    serde_json::json!({
        "backend": store.backend(),
        "streams": streams,
        "persist_count": store.persist_count(),
        // Null on a clean open; the recovery message when the store opened
        // over a corrupt/torn snapshot (the operator-facing corruption flag).
        "snapshot_load_error": store.snapshot_load_error(),
    })
    .to_string()
}

/// The well-known stream names in status order (single source of truth in
/// `durable::STREAM_NAMES`).
use crate::durable::STREAM_NAMES;

/// Durably record an emitted verification certificate (mass-gap / Ritz bound
/// from the T6 pipeline) as a `certificate-issued` line in the `certificates`
/// stream. Fail-closed: a kernel without a configured durable store refuses
/// (the line would not be replayable), and a checkpoint failure is surfaced.
/// Returns the stream length after the append (a 1-based sequence number).
pub fn record_certificate_issued(cert_json: &str) -> Result<u64, String> {
    let record: serde_json::Value = serde_json::from_str(cert_json)
        .map_err(|e| format!("certificate record not valid JSON: {e}"))?;
    let json = serde_json::to_vec(&serde_json::json!({
        "kind": "certificate-issued",
        "record": record,
    }))
    .map_err(|e| format!("certificate record encode: {e}"))?;
    let Some(store) = durable() else {
        return Err("durable store not configured (kernel runs RAM-only)".to_string());
    };
    store
        .append(streams::CERTIFICATES, &json)
        .map_err(|e| format!("durable append to certificates: {e}"))?;
    checkpoint().map_err(|e| e.to_string())?;
    store
        .stream_len(streams::CERTIFICATES)
        .map_err(|e| format!("certificates stream length: {e}"))
}

pub fn store_session(mut session: Session) -> i64 {
    // H4: every live session is backed by the kernel's durable store, so its
    // committed events land in the `session` stream and probability/condition
    // reads flush first (fail-closed).
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    session.set_durable(durable());
    map.insert(
        handle,
        SessionEntry {
            session,
            last_result: String::new(),
        },
    );
    handle
}

pub fn with_session_mut<R>(handle: i64, f: impl FnOnce(&mut Session) -> R) -> Option<R> {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    let entry = map.get_mut(&handle)?;
    Some(f(&mut entry.session))
}

pub fn with_session<R>(handle: i64, f: impl FnOnce(&Session) -> R) -> Option<R> {
    let guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    let entry = map.get(&handle)?;
    Some(f(&entry.session))
}

pub fn set_last_result(handle: i64, json: String) {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut()
        && let Some(entry) = map.get_mut(&handle)
    {
        entry.last_result = json;
    }
}

/// Clear the session's result slot. Every result-producing `uk_*` op calls
/// this on entry, mirroring the error channel's fresh-per-call discipline
/// (`ffi_entry`): the slot must describe *this* op's outcome, never a
/// previous one. A failed op therefore leaves an EMPTY result — a visible
/// "no result" for `uk_get_result` — instead of silently serving the
/// previous op's JSON as if it were this op's output.
pub fn clear_last_result(handle: i64) {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut()
        && let Some(entry) = map.get_mut(&handle)
    {
        entry.last_result = String::new();
    }
}

pub fn get_last_result(handle: i64) -> Option<String> {
    let guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.get(&handle).map(|e| e.last_result.clone())
}

pub fn push_event(handle: i64, event: KernelEvent) {
    let event_json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    let mut guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        for sub in map.values_mut() {
            if sub.model_handle == handle && matches_query(&sub.query, &event) {
                if sub.events.len() >= EVENT_QUEUE_CAPACITY {
                    sub.events.pop_front();
                    sub.dropped += 1;
                    // Surface the FIRST drop of a burst on the operator ring
                    // (`uk_owner_list`): a sustained overflow would otherwise
                    // flood the owner log with one line per lost event. The
                    // cumulative count is included; further drops in the same
                    // burst stay silent (the operator already knows to look).
                    if sub.dropped == 1 {
                        owner_log(
                            "kernel.subscription",
                            &format!(
                                "model {:?}: subscription overran its {} event queue; \
                                 events are being dropped (lost events will not be delivered)",
                                sub.model_handle, EVENT_QUEUE_CAPACITY,
                            ),
                        );
                    }
                }
                sub.events.push_back(event_json.clone());
            }
        }
    }
}

/// Broadcast a kernel-global action event (S4 approval lane) to subscriptions that
/// **explicitly opted into the approval lane** — a query whose `types` list names
/// `action_pending` / `action_resolved`. An all-types subscription (`{}` / `null`) is
/// the *session* lane and does not receive action events: the two lanes are disjoint,
/// so a module watching its model's event stream never sees foreign approval traffic.
pub fn push_action_event(event: KernelEvent) {
    let event_json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    let mut guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        for sub in map.values_mut() {
            if matches_action_query(&sub.query, &event) {
                if sub.events.len() >= EVENT_QUEUE_CAPACITY {
                    sub.events.pop_front();
                    sub.dropped += 1;
                    // Surface the FIRST drop of a burst (see push_event):
                    // one operator-visible notice, not one per lost event.
                    if sub.dropped == 1 {
                        owner_log(
                            "kernel.subscription",
                            &format!(
                                "model {:?}: approval lane overran its {} event queue; \
                                 action events are being dropped",
                                sub.model_handle, EVENT_QUEUE_CAPACITY,
                            ),
                        );
                    }
                }
                sub.events.push_back(event_json.clone());
            }
        }
    }
}

/// The known event-type vocabulary for `EventQuery.types` — the session
/// lane (model-scoped events) plus the approval lane (`action_*`).
/// `matches_query`/`matches_action_query` accept only these; validating at
/// subscribe time turns a typo (e.g. `evovled`) into an immediate, visible
/// error instead of a subscription that silently never fires.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "evolved",
    "conditioned",
    "observed",
    "verified",
    "simplified",
    "logos_compiled",
    "austral_unf",
    "whyml_compiled",
    "error",
    "prior_set",
    "hamiltonian_set",
    "action_pending",
    "action_resolved",
];

/// Validate a subscription query's `types` list. Every named type must be in
/// [`KNOWN_EVENT_TYPES`]; the first unknown one is reported with the valid
/// vocabulary so the caller can correct it (never a silently-dead
/// subscription that matches nothing).
pub fn validate_event_query(query: &EventQuery) -> Result<(), String> {
    let Some(types) = &query.types else {
        return Ok(());
    };
    for t in types {
        if !KNOWN_EVENT_TYPES.contains(&t.as_str()) {
            return Err(format!(
                "unknown event type '{t}' in subscription query; \
                 known types: [{}]",
                KNOWN_EVENT_TYPES.join(", ")
            ));
        }
    }
    Ok(())
}

fn matches_query(query: &EventQuery, event: &KernelEvent) -> bool {
    let Some(types) = &query.types else {
        return true;
    };
    if types.is_empty() {
        return true;
    };

    let event_type = match event {
        KernelEvent::Evolved { .. } => "evolved",
        KernelEvent::Conditioned { .. } => "conditioned",
        KernelEvent::Observed { .. } => "observed",
        KernelEvent::Verified { .. } => "verified",
        KernelEvent::Simplified { .. } => "simplified",
        KernelEvent::LogosCompiled { .. } => "logos_compiled",
        KernelEvent::AustralUnf { .. } => "austral_unf",
        KernelEvent::WhymlCompiled { .. } => "whyml_compiled",
        KernelEvent::Error { .. } => "error",
        KernelEvent::PriorSet => "prior_set",
        KernelEvent::HamiltonianSet => "hamiltonian_set",
        KernelEvent::ActionPending { .. } | KernelEvent::ActionResolved { .. } => return false,
    };
    types.contains(&event_type.to_string())
}

/// Approval-lane matcher: only explicit action types opt a subscription into the
/// kernel-global action lane. Session events never match here.
fn matches_action_query(query: &EventQuery, event: &KernelEvent) -> bool {
    let Some(types) = &query.types else {
        return false;
    };
    let event_type = match event {
        KernelEvent::ActionPending { .. } => "action_pending",
        KernelEvent::ActionResolved { .. } => "action_resolved",
        _ => return false,
    };
    types.contains(&event_type.to_string())
}

pub fn create_subscription(model_handle: i64, query: EventQuery) -> Result<i64, String> {
    // Reject unknown type names up front: a typo'd type must be a visible
    // error at subscribe time, not a subscription that silently matches
    // nothing forever ("never dead-end the agent").
    validate_event_query(&query)?;
    let guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    if guard
        .as_ref()
        .and_then(|map| map.get(&model_handle))
        .is_none()
    {
        return Err("invalid model handle".to_string());
    }

    let sub_handle = NEXT_SUB.fetch_add(1, Ordering::SeqCst);
    let mut sub_guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = sub_guard.get_or_insert_with(HashMap::new);
    map.insert(
        sub_handle,
        Subscription {
            model_handle,
            query,
            events: VecDeque::new(),
            dropped: 0,
        },
    );
    Ok(sub_handle)
}

pub fn peek_subscription(sub_handle: i64) -> Option<Option<String>> {
    let guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    let sub = map.get(&sub_handle)?;
    Some(sub.events.front().cloned())
}

pub fn poll_subscription(sub_handle: i64) -> Option<Option<String>> {
    let mut guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    let sub = map.get_mut(&sub_handle)?;
    Some(sub.events.pop_front())
}

pub fn free_session(handle: i64) -> bool {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let removed = guard
        .as_mut()
        .map(|map| map.remove(&handle).is_some())
        .unwrap_or(false);

    if removed {
        let mut sub_guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = sub_guard.as_mut() {
            map.retain(|_, sub| sub.model_handle != handle);
        }
    }
    removed
}

pub fn set_last_error(diag: &Diagnostic) {
    let json = serde_json::to_string(diag).unwrap_or_else(|_| "{}".to_string());
    LAST_ERROR.with(|e| *e.borrow_mut() = json);
}

/// Clear the error channel. Every FFI entry point clears before running so
/// `uk_last_error` describes the MOST RECENT call — never a stale failure
/// from an earlier one ("record facts where they happen"). The read symbols
/// (`uk_last_error` itself) must NOT clear before reading (see `ffi_entry`).
pub fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = String::new());
}

pub fn get_last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}

// ── buffer storage for uk_ode_analyze ──────────────────────────────────

/// Wrapper around a raw pointer to make it Send+Sync for static storage.
/// Safety: the pointer is only accessed from the thread that created it
/// via the handles API, which uses a Mutex for synchronization.
struct BufPtr(*mut u8, i64);
unsafe impl Send for BufPtr {}
unsafe impl Sync for BufPtr {}

static BUFFERS: Mutex<Option<HashMap<i64, BufPtr>>> = Mutex::new(None);
static NEXT_BUF: AtomicI64 = AtomicI64::new(100_000);

pub fn store_buffer(ptr: *mut u8, len: i64) -> i64 {
    let handle = NEXT_BUF.fetch_add(1, Ordering::SeqCst);
    let mut guard = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(handle, BufPtr(ptr, len));
    handle
}

pub fn free_buffer(handle: i64) -> bool {
    let mut guard = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut()
        && let Some(BufPtr(ptr, len)) = map.remove(&handle)
    {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len as usize));
        }
        return true;
    }
    false
}

/// Read buffer contents as a string. Returns None if handle is invalid.
#[cfg(test)]
pub fn read_buffer(handle: i64) -> Option<String> {
    let guard = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    let BufPtr(ptr, len) = map.get(&handle)?;
    unsafe {
        let slice = std::slice::from_raw_parts(*ptr, *len as usize);
        Some(std::str::from_utf8(slice).unwrap_or("").to_string())
    }
}

// ── action queue (S4: deferred approval + local simulation) ──────────────
//
// A single kernel-wide store of side-effecting `ActionRecord`s shared by every
// module/session in the process. `uk_action_submit` inserts a Pending record and
// returns its handle; `uk_action_apply/reject/revert` resolve it; `uk_action_get`
// returns the record with the merged (provisional→applied) result; `uk_action_list`
// returns the whole queue for a gatekeeper module to scan.

static ACTIONS: Mutex<Option<HashMap<i64, ActionRecord>>> = Mutex::new(None);
static NEXT_ACTION: AtomicI64 = AtomicI64::new(1);

/// H4: action handles whose outcome is UNKNOWN (a crash left an in-flight
/// marker with no resolved record). `uk_action_apply` on these is refused with
/// UK-1010; `uk_action_get`/`list` surface `unknown_outcome: true`.
static UNKNOWN_OUTCOME_ACTIONS: Mutex<Option<HashSet<i64>>> = Mutex::new(None);

/// Whether `handle`'s outcome is unknown (interrupted side-effecting call).
pub fn is_unknown_outcome(handle: i64) -> bool {
    UNKNOWN_OUTCOME_ACTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|s| s.contains(&handle))
        .unwrap_or(false)
}

/// Mark an action handle as having an unknown outcome (recovery path).
pub fn mark_unknown_outcome(handle: i64) {
    UNKNOWN_OUTCOME_ACTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert(handle);
}

/// Test-only: install a caller-provided durable store (e.g. a wrapper that
/// fails `flush` to exercise the fail-closed checkpoint paths).
#[cfg(test)]
pub fn set_durable_for_tests(store: Option<Arc<dyn DurableStore>>) {
    *DURABLE.lock().unwrap_or_else(|e| e.into_inner()) = store;
}

/// Test-only: wipe the in-memory rings and durable store so a test can
/// simulate a cold process restart against a fresh on-disk store.
#[cfg(test)]
pub fn reset_durable_for_tests() {
    *DURABLE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *ACTIONS.lock().unwrap_or_else(|e| e.into_inner()) = None;
    NEXT_ACTION.store(1, Ordering::SeqCst);
    *UNKNOWN_OUTCOME_ACTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = None;
    *AUDIT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    NEXT_AUDIT_SEQ.store(1, std::sync::atomic::Ordering::SeqCst);
    *OWNER_LOG.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// H4: write the durable in-flight marker for an action and checkpoint. Called
/// by `uk_action_apply` *before* the side effect fires; a failure here means
/// the outcome is unknown (fail-closed — the apply must not dispatch).
pub fn mark_inflight(record: &ActionRecord) -> Result<(), String> {
    let json = serde_json::json!({"id": record.id, "inflight": true})
        .to_string()
        .into_bytes();
    durable_append(streams::ACTIONS, &json);
    checkpoint().map_err(|e| {
        format!(
            "durable in-flight marker for action {} could not be flushed: {e}",
            record.id
        )
    })
}

/// Insert into the action ring (no durable write). Returns the assigned handle.
fn ring_put_action(record: ActionRecord) -> i64 {
    let handle = NEXT_ACTION.fetch_add(1, Ordering::SeqCst);
    let mut guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(handle, record);
    handle
}

pub fn store_action(record: ActionRecord) -> i64 {
    // Write-through: persist the record (stable `id`) before shadowing it in
    // the ring. The `record` is the full `ActionRecord`; recovery remints a
    // fresh handle per id.
    if let Ok(json) = serde_json::to_vec(&serde_json::json!({
        "id": record.id,
        "record": record,
    })) {
        durable_append(streams::ACTIONS, &json);
    }
    ring_put_action(record)
}

pub fn with_action<R>(handle: i64, f: impl FnOnce(&ActionRecord) -> R) -> Option<R> {
    let guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.get(&handle).map(f)
}

pub fn with_action_mut<R>(handle: i64, f: impl FnOnce(&mut ActionRecord) -> R) -> Option<R> {
    let mut guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    map.get_mut(&handle).map(f)
}

/// Snapshot of the full action queue (gatekeeper scan surface). Returns
/// `(handle, record)` pairs ordered by handle (== creation order).
pub fn list_actions() -> Vec<(i64, ActionRecord)> {
    let guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let mut items: Vec<(i64, ActionRecord)> = guard
        .as_ref()
        .map(|map| map.iter().map(|(h, r)| (*h, r.clone())).collect())
        .unwrap_or_default();
    items.sort_by_key(|(h, _)| *h);
    items
}

// ── caller context (S6: GatekeeperCaller tags) ─────────────────────────
//
// A per-thread "who is calling right now" slot. The host loopback sets it at the
// start of each dispatch (a module cannot forge another identity's tag — the
// loopback owns the value), and the kernel reads it when appending audit entries
// or tagging `ActionRecord`s. Direct (trusted) callers default to
// `{from: hook, principal: "kernel"}` with no grant bounds.

thread_local! {
    static CALLER: std::cell::RefCell<CallerContext> =
        std::cell::RefCell::new(CallerContext::default());
}

/// The active caller identity + optional bounded grants, per thread.
#[derive(Debug, Clone, Default)]
pub struct CallerContext {
    /// The audit tag (identity) of the current caller.
    pub tag: CallerTag,
    /// The caller's bounded grant set. `None` = unrestricted (kernel-level
    /// trust — e.g. a direct harness driving the ABI).
    pub grants: Option<GrantSet>,
}

impl CallerContext {
    /// True once `set_caller` has been called on this thread (distinguishes an
    /// explicitly-tagged module/agent call from the default trusted harness).
    pub fn is_explicit(&self) -> bool {
        self.tag != CallerTag::default() || self.grants.is_some()
    }

    /// F8 observer re-check: may this caller read a record/audit entry whose
    /// principal is `principal`? The trusted harness (no bounded grant set) sees
    /// everything; a bounded caller sees its own principal and any principal
    /// declared in its `observers` grant (no read-up).
    pub fn may_observe(&self, principal: &str) -> bool {
        match self.grants.as_ref() {
            None => true,
            Some(grants) => {
                self.tag.principal == principal || grants.observers.iter().any(|o| o == principal)
            }
        }
    }

    /// F20 trust annotation for a granted effect: whether the caller's grant marks
    /// `effect` an observation (applies immediately, never queued) or a mutation
    /// (queued for approval unless the console vetted it). Un-annotated → [`EffectKind::Mutate`]
    /// (conservative — an annotation can only *downgrade* a side-effect, never grant one).
    pub fn effect_kind_of(&self, effect: &str) -> EffectKind {
        self.grants
            .as_ref()
            .and_then(|g| g.effect_kind_of(effect))
            .unwrap_or(EffectKind::Mutate)
    }
}

/// Set the current thread's caller context. Returns the previous context.
pub fn set_caller(tag: CallerTag, grants: Option<GrantSet>) -> CallerContext {
    CALLER.with(|c| std::mem::replace(&mut *c.borrow_mut(), CallerContext { tag, grants }))
}

/// Reset the current thread's caller context to the default (trusted harness).
pub fn clear_caller() {
    CALLER.with(|c| *c.borrow_mut() = CallerContext::default());
}

/// Read the current thread's caller context.
pub fn current_caller() -> CallerContext {
    CALLER.with(|c| c.borrow().clone())
}

// ── vetted markers (S21/F20) ────────────────────────────────────────────
//
// A principal marked *vetted* may auto-apply a `mutate`-kind effect without a
// pending approval. Vetted status is minted ONLY by the operator console
// (`uk_registry_vetted`, hook-only); a module can never self-declare it
// (module.toml carries `effect_kind` annotations, never vetted claims). The
// marker is entirely separate from the approval queue: un-vetting a principal
// leaves every pending action untouched.

static VETTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Ring-only vetted marker insert (recovery path).
fn ring_mark_vetted(principal: String) {
    let mut guard = VETTED.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    set.insert(principal);
}

/// Ring-only vetted marker removal (recovery path).
fn ring_unmark_vetted(principal: &str) {
    let mut guard = VETTED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = guard.as_mut() {
        set.remove(principal);
    }
}

/// Set or clear the console's vetted marker for `principal`. Returns the new value.
/// Write-through: the decision is durably recorded in the `config` stream.
pub fn mark_vetted(principal: &str, vetted: bool) -> bool {
    durable_append(
        streams::CONFIG,
        &serde_json::json!({"vetted_principal": principal, "vetted": vetted})
            .to_string()
            .into_bytes(),
    );
    let mut guard = VETTED.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    if vetted {
        set.insert(principal.to_string());
    } else {
        set.remove(principal);
    }
    set.contains(principal)
}

/// Whether the console has vetted `principal` (auto-apply for mutate effects).
pub fn is_vetted(principal: &str) -> bool {
    let guard = VETTED.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .as_ref()
        .map(|set| set.contains(principal))
        .unwrap_or(false)
}

/// Drop every vetted marker (QA/console reset). Approval queue untouched.
pub fn clear_vetted() {
    let mut guard = VETTED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = guard.as_mut() {
        set.clear();
    }
}

// ── H9: deployment security posture ─────────────────────────────────────
//
// A configuration layer over the existing S21/S22/S23/S25/S26 primitives. The
// posture is set only by the operator console (S22 admin seam, hook + no grant
// bounds — the same gate as `uk_registry_vetted`) and read by every dispatch.

static POSTURE: std::sync::Mutex<unfer_protocol::SecurityPosture> =
    std::sync::Mutex::new(unfer_protocol::SecurityPosture::Auto);

/// Read the current deployment posture.
pub fn posture() -> unfer_protocol::SecurityPosture {
    *POSTURE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Set the deployment posture (operator console only). Returns the previous
/// posture.
pub fn set_posture(p: unfer_protocol::SecurityPosture) -> unfer_protocol::SecurityPosture {
    durable_append(
        streams::CONFIG,
        &serde_json::json!({"posture": format!("{p:?}").to_lowercase()})
            .to_string()
            .into_bytes(),
    );
    let mut guard = POSTURE.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::replace(&mut *guard, p)
}

// ── windowed meter (S25/F24: budgets + rate limits) ────────────────────
//
// A per-principal, per-UTC-day windowed counter that the loopback chokepoint
// consults before dispatching a *metered* `uk_*` symbol. It mirrors Cloudflare's
// `consumeDailyLlmCall`/`DailyQuotaResult`: a DO-local atomic read-modify-write
// with a UTC-day window that resets at the midnight boundary. Unlike the S13
// `Metrics` (which only *counts*), this is a *denial* point — over-budget /
// over-rate calls are refused with UK-46xx and an audit entry, never a
// post-hoc report. Lightweight symbols (reads, version) are unmetered so agents
// stay responsive.

/// A single principal's windowed usage for the current UTC day.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MeterStatus {
    /// The UTC-day window key this status is reported for.
    pub day: String,
    /// Whether the caller is still within its budget for the window.
    pub within_limits: bool,
    /// Budget units remaining this window (saturating at 0).
    pub remaining: u64,
    /// The configured budget ceiling for the window.
    pub limit: u64,
    /// Calls counted against the window so far (post-consume when consuming).
    pub used: u64,
    /// ISO-8601 instant of the next UTC-midnight reset.
    pub reset_at: String,
}

/// UTC-day window key (`%Y-%m-%d`), matching the reset boundary of the meter.
fn utc_day_key() -> String {
    // Use a fixed epoch-day from SystemTime so the value is deterministic and
    // testable without wall-clock parsing; the reset boundary is the same
    // "since epoch" day counter Cloudflare keys on.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("d{}", secs / 86_400)
}

/// ISO instant of the next UTC-midnight reset (approximate, for the status blob).
fn next_utc_midnight() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let next = (secs / 86_400 + 1) * 86_400;
    format!("t{}", next)
}

#[derive(Debug, Clone, Default)]
struct MeterWindow {
    day: String,
    used: u64,
}

static METER: Mutex<Option<HashMap<String, MeterWindow>>> = Mutex::new(None);

/// Read the current window usage for `principal` without consuming, mirroring
/// Cloudflare's `checkDailyLlmCount`. `limit` is only used to compute
/// `remaining`/`within_limits` — it never mutates the store.
pub fn meter_status(principal: &str, limit: u64) -> MeterStatus {
    let day = utc_day_key();
    let guard = METER.lock().unwrap_or_else(|e| e.into_inner());
    let used = guard
        .as_ref()
        .and_then(|m| m.get(principal))
        .filter(|w| w.day == day)
        .map(|w| w.used)
        .unwrap_or(0);
    let remaining = limit.saturating_sub(used);
    MeterStatus {
        day,
        within_limits: used < limit,
        remaining,
        limit,
        used,
        reset_at: next_utc_midnight(),
    }
}

/// Per-window outcome of a metered symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterDecision {
    /// The call is within budget and rate limit; proceed.
    Allowed,
    /// The caller exceeded its windowed call-rate limit (UK-4601).
    RateLimited,
    /// The caller exhausted its windowed budget (UK-4602).
    BudgetExceeded,
}

/// Atomically check the windowed budget and, if within it, count one call. This
/// is the single denial point the loopback calls for metered symbols. A blocked
/// request never counts (no-op once exhausted).
pub fn meter_consume(principal: &str, budget: u64, rate_limit: u64) -> MeterDecision {
    let day = utc_day_key();
    // Rate gate first: a fixed per-window cap independent of the budget.
    if rate_limit > 0 {
        let used = {
            let guard = METER.lock().unwrap_or_else(|e| e.into_inner());
            guard
                .as_ref()
                .and_then(|m| m.get(principal))
                .filter(|w| w.day == day)
                .map(|w| w.used)
                .unwrap_or(0)
        };
        if used >= rate_limit {
            return MeterDecision::RateLimited;
        }
    }
    let mut guard = METER.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    let window = map
        .entry(principal.to_string())
        .or_insert_with(|| MeterWindow {
            day: day.clone(),
            used: 0,
        });
    if window.day != day {
        // New UTC window: reset the counter.
        window.day.clone_from(&day);
        window.used = 0;
    }
    if window.used >= budget.max(1) {
        return MeterDecision::BudgetExceeded;
    }
    window.used += 1;
    MeterDecision::Allowed
}

/// Reset the meter (QA/console reset). Does not touch the approval queue.
pub fn clear_meter() {
    let mut guard = METER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
}

// ── sensitive-forward latch (S26/F25) ──────────────────────────────────
//
// Once a caller observes `<*sensitive*>` data, the *fact of having observed it*
// constrains everything it does next (egress, hand-off, blueprints, writes) until
// an operator clears the latch — mirroring Cloudflare's `prohibitAllSharing`
// workspace latch. The latch is sticky per-principal and sticks to the caller
// set, so a spawned agent inherits its parent's sticky set via `GrantSet`
// ordering. Clearing is console-only (the S22 admin operator).

static SENSITIVE_LATCH: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Whether `principal` is latched (has observed sensitive data and is refused
/// forward-mutating ops until an operator clears it).
pub fn is_sensitive_latched(principal: &str) -> bool {
    let guard = SENSITIVE_LATCH.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .as_ref()
        .map(|s| s.contains(principal))
        .unwrap_or(false)
}

/// Set or clear the sensitive latch for `principal` (operator console only).
pub fn set_sensitive_latch(principal: &str, latched: bool) -> bool {
    let mut guard = SENSITIVE_LATCH.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    if latched {
        set.insert(principal.to_string());
    } else {
        set.remove(principal);
    }
    set.contains(principal)
}

/// Clear every sensitive latch (QA/console reset). Approval queue untouched.
pub fn clear_sensitive_latches() {
    let mut guard = SENSITIVE_LATCH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = guard.as_mut() {
        set.clear();
    }
}

// ── credential vault (S27/F26) ─────────────────────────────────────────
//
// A first-class secret vault so a gatekeeper owns a credential and grants a
// *handle*, never the raw value. Secrets are encrypted at rest with the S15
// KeyRing and are opaque to callers: `uk_secret_get` returns a handle the host
// dereferences at call time (grant-checked), so a secret never reaches gadget
// code. A live secret must never serialize into a `SessionBlob` snapshot or a
// `.cell` blueprint — `uk_snapshot`/`uk_blueprint_export` refuse to package a
// live secret.

/// A stored secret: ciphertext at rest, plus the owner that may dereference it.
#[derive(Clone)]
struct StoredSecret {
    /// Encrypted-at-rest bytes (KeyRing envelope).
    envelope: unfer_data::CellEnvelope,
    /// The principal granted the opaque dereference handle.
    owner: String,
}

static SECRETS: Mutex<Option<std::collections::HashMap<u64, StoredSecret>>> = Mutex::new(None);
static NEXT_SECRET: AtomicU64 = AtomicU64::new(1);

/// Store a secret under the calling `owner`; returns an opaque handle. Encrypted
/// at rest under a process-global KeyRing. The raw value is never returned from
/// the vault — only the handle is.
pub fn vault_put_secret(owner: &str, value: &[u8]) -> Result<u64, String> {
    let handle = NEXT_SECRET.fetch_add(1, Ordering::SeqCst);
    let envelope = {
        let ring = vault_ring();
        ring.encrypt_envelope(value)?
    };
    let mut guard = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(std::collections::HashMap::new);
    map.insert(
        handle,
        StoredSecret {
            envelope,
            owner: owner.to_string(),
        },
    );
    Ok(handle)
}

/// Dereference a secret handle. Only `owner` may read it; the host dereferences
/// at call time (grant-checked) so the raw value never reaches gadget code.
pub fn vault_get_secret(handle: u64, owner: &str) -> Result<Vec<u8>, String> {
    let envelope = {
        let guard = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
        let stored = guard
            .as_ref()
            .and_then(|m| m.get(&handle))
            .ok_or_else(|| "secret handle not found".to_string())?;
        if stored.owner != owner {
            return Err("secret is not owned by this caller".to_string());
        }
        stored.envelope.clone()
    };
    let ring = vault_ring();
    ring.decrypt_envelope(&envelope)
}

/// Revoke a secret handle, invalidating it. The at-rest ciphertext is dropped.
pub fn vault_revoke_secret(handle: u64) -> bool {
    let mut guard = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_mut().and_then(|m| m.remove(&handle)).is_some()
}

/// Whether any live secret exists in the vault. `uk_snapshot`/`uk_blueprint_export`
/// refuse to package a live secret (it must never serialize into a snapshot or a
/// `.cell` blueprint).
pub fn vault_has_live_secrets() -> bool {
    let guard = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map(|m| !m.is_empty()).unwrap_or(false)
}

/// Drop every secret (QA/console reset).
pub fn vault_clear() {
    let mut guard = SECRETS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
}

/// The process-global key ring for at-rest secret encryption (S15 KeyRing).
fn vault_ring() -> std::sync::MutexGuard<'static, unfer_data::KeyRing> {
    static RING: std::sync::OnceLock<Mutex<unfer_data::KeyRing>> = std::sync::OnceLock::new();
    let ring = RING.get_or_init(|| Mutex::new(unfer_data::KeyRing::from_random()));
    ring.lock().unwrap_or_else(|e| e.into_inner())
}

// ── H13: skills registry (discovery/sharing over the existing module path) ──
//
// A skill is a discoverable, shareable reference to an existing module (or a
// packed `module.toml` cell). Registry is process-global; skills are
// scope-owned, shareable by grant, and admin-promotable (S22 seam).

static SKILLS: std::sync::Mutex<Option<unfer_protocol::skills::SkillRegistry>> =
    std::sync::Mutex::new(None);

fn skill_registry() -> std::sync::MutexGuard<'static, Option<unfer_protocol::skills::SkillRegistry>>
{
    SKILLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Register (or replace) a skill. Refused when a promoted skill is replaced by
/// a non-promoted one (promotion is admin-gated).
pub fn skill_register(skill: unfer_protocol::skills::Skill) -> bool {
    let mut guard = skill_registry();
    guard
        .get_or_insert_with(unfer_protocol::skills::SkillRegistry::new)
        .register(skill)
}

/// Fetch a skill by id.
pub fn skill_get(id: &str) -> Option<unfer_protocol::skills::Skill> {
    let guard = skill_registry();
    guard.as_ref().and_then(|r| r.get(id)).cloned()
}

/// List skills visible to `principal` (org-scoped + own + grant-free).
pub fn skill_list_visible(principal: &str) -> Vec<unfer_protocol::skills::Skill> {
    let guard = skill_registry();
    guard
        .as_ref()
        .map(|r| r.list_visible(principal).into_iter().cloned().collect())
        .unwrap_or_default()
}

/// Admin-gated promotion (S22 seam): move a skill to org scope. Test/QA +
/// operator-console surface (no loopback arm yet).
#[allow(dead_code)]
pub fn skill_promote(id: &str) -> bool {
    let mut guard = skill_registry();
    guard.as_mut().map(|r| r.promote(id)).unwrap_or(false)
}

/// Drop every skill (QA/console reset).
pub fn skill_clear() {
    let mut guard = skill_registry();
    if let Some(r) = guard.as_mut() {
        *r = unfer_protocol::skills::SkillRegistry::new();
    }
}

// ── certificate ledger (Plan R: carbon-certificate / UTXO state machine) ──
//
// A single process-global `CertificateLedger` — the same state-transition engine
// every QuePaxa node runs when it applies a `CertificateOp` from the consensus
// log. The FFI exposes it so modules/agents can drive mint/transfer/burn and
// read the committed sparse-Merkle root. The actor DID is passed in the op and
// checked by the ledger (mint authority, ownership, conservation, double-spend).
// At the FFI boundary the caller is trusted (the consensus layer is where a
// signature is verified); the ledger invariants still hold.

static NEXT_CERT_SEQ: AtomicU64 = AtomicU64::new(1);

fn cert_ledger() -> std::sync::MutexGuard<'static, CertificateLedger> {
    static LEDGER: std::sync::OnceLock<Mutex<CertificateLedger>> = std::sync::OnceLock::new();
    let ledger = LEDGER.get_or_init(|| Mutex::new(CertificateLedger::new(MintAuthority::None)));
    ledger.lock().unwrap_or_else(|e| e.into_inner())
}

/// Configure the mint authority. `None` disables minting (the safe default).
pub fn cert_set_authority(did: Option<String>) {
    let mut ledger = cert_ledger();
    *ledger = CertificateLedger::new(match did {
        Some(d) => MintAuthority::Only(d),
        None => MintAuthority::None,
    });
}

/// The current committed sparse-Merkle root.
pub fn cert_root() -> [u8; 32] {
    cert_ledger().root()
}

/// A JSON snapshot of the ledger state.
pub fn cert_status() -> serde_json::Value {
    let ledger = cert_ledger();
    serde_json::json!({
        "root": hex::encode(ledger.root()),
        "unspent_count": ledger.unspent_count(),
        "total_supply": ledger.total_supply(),
    })
}

/// Apply a certificate state transition (mint/transfer/burn) as `actor`.
/// Returns the resulting coin_ids (mint/transfer) on success.
pub fn cert_apply(
    actor: &str,
    kind: &unfer_protocol::CertificateOpKind,
) -> Result<Vec<CertId>, Diagnostic> {
    let seq = NEXT_CERT_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ledger = cert_ledger();
    ledger.apply_op(actor, kind, seq)
}

/// Reset the ledger (QA/console reset). Minting returns to None.
pub fn cert_clear() {
    let mut ledger = cert_ledger();
    *ledger = CertificateLedger::new(MintAuthority::None);
}

// ── unified auction (Prebid-model, carbon credits + publicity inventory) ──
//
// At the FFI boundary the caller is trusted, exactly like the certificate
// ledger above: `auction_apply` runs the deterministic auction engine
// (open/bid/close), and the winning bid is a pure function of the recorded
// bids, so a caller replaying the same ops converges on the same winner.

static NEXT_AUCTION_SEQ: AtomicU64 = AtomicU64::new(1);

fn auction_ledger() -> std::sync::MutexGuard<'static, AuctionLedger> {
    static LEDGER: std::sync::OnceLock<Mutex<AuctionLedger>> = std::sync::OnceLock::new();
    let ledger = LEDGER.get_or_init(|| Mutex::new(AuctionLedger::new()));
    ledger.lock().unwrap_or_else(|e| e.into_inner())
}

/// Apply an auction state transition (open/bid/close) as `actor`. Returns the
/// deterministic winner (Some for a close that selects one, None otherwise).
pub fn auction_apply(
    actor: &str,
    kind: &unfer_protocol::AuctionOpKind,
) -> Result<Option<unfer_protocol::AuctionWinner>, Diagnostic> {
    let seq = NEXT_AUCTION_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ledger = auction_ledger();
    ledger.apply_op(actor, kind, seq)
}

/// Retry-safe close (the round-20 `uk_poll`-class fix): compute the winner,
/// serialize it, and ONLY commit the close when the caller's buffer can hold
/// the full JSON. A probe (`buf` null / `cap <= 0`) or too-small buffer
/// returns the needed size with the lot still OPEN — the caller retries with
/// a bigger buffer, exactly like every sibling buffer op. Committed under one
/// ledger lock hold so the probed size always matches the committed winner.
/// Returns the buffer bytes on success, <0 (-code) on error.
pub fn auction_close_json(
    actor: &str,
    lot_id: &unfer_protocol::AuctionId,
    buf: *mut u8,
    cap: i64,
) -> Result<i64, Diagnostic> {
    let _seq = NEXT_AUCTION_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ledger = auction_ledger();
    // Probe half: read-only winner computation (no mutation).
    let winner = ledger.close_winner(actor, lot_id)?;
    let json = serde_json::to_string(&winner).map_err(|e| {
        Diagnostic::new(
            unfer_protocol::Code::INTERNAL,
            format!("serialize: {e}"),
            unfer_protocol::Severity::Error,
        )
    })?;
    let needed = json.len() as i64;
    if buf.is_null() || cap < needed {
        // Lot stays OPEN: the caller retries with a bigger buffer.
        return Ok(needed);
    }
    // Commit half: the buffer is large enough, so close the lot now.
    ledger.apply_close(actor, lot_id)?;
    unsafe {
        std::ptr::copy_nonoverlapping(json.as_ptr(), buf, json.len());
    }
    Ok(needed)
}

/// A JSON snapshot of one lot (or null).
pub fn auction_report(lot_id: &unfer_protocol::AuctionId) -> Option<unfer_protocol::AuctionReport> {
    auction_ledger().report(lot_id)
}

/// JSON snapshots of every lot currently open for bidding.
pub fn auction_open_lots() -> Vec<unfer_protocol::AuctionReport> {
    auction_ledger().open_lots()
}

/// Reset the auction ledger (QA/console reset).
pub fn auction_clear() {
    let mut ledger = auction_ledger();
    *ledger = AuctionLedger::new();
}

#[cfg(test)]
pub static AUCTION_TESTS_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod auction_ledger_tests {
    use super::*;
    use unfer_protocol::{AuctionAsset, AuctionCurrency, AuctionLot};

    fn lot(seller: &str, floor: u64, id_byte: u8) -> AuctionLot {
        AuctionLot {
            lot_id: unfer_protocol::AuctionId([id_byte; 32]),
            seller_did: seller.to_string(),
            asset: AuctionAsset::CarbonCredits { amount: 1000 },
            currency: AuctionCurrency::Taler,
            floor,
            opens_seq: 1,
            closes_seq: 100,
        }
    }

    #[test]
    fn open_bid_close_roundtrip() {
        let _g = AUCTION_TESTS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        auction_clear();
        let seller = "did:unfer:seller";
        auction_apply(
            seller,
            &unfer_protocol::AuctionOpKind::Open {
                lot: lot(seller, 5, 1),
            },
        )
        .unwrap();
        auction_apply(
            "did:unfer:alice",
            &unfer_protocol::AuctionOpKind::Bid {
                lot_id: unfer_protocol::AuctionId([1; 32]),
                price_per_unit: 6,
                quantity: 500,
            },
        )
        .unwrap();
        let winner = auction_apply(
            seller,
            &unfer_protocol::AuctionOpKind::Close {
                lot_id: unfer_protocol::AuctionId([1; 32]),
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(winner.bidder_did, "did:unfer:alice");
        assert_eq!(winner.total, 3000);
    }
}

/// Single serialization point for every certificate-ledger test in the crate.
/// The ledger is process-global and shared by both `handles::cert_ledger_tests`
/// and `tests::cert_ffi_*`, so all of them must hold this one lock (a
/// per-module lock would let the two modules clobber each other's state).
#[cfg(test)]
pub static CERT_TESTS_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod cert_ledger_tests {
    // Plan R: the FFI ledger is process-global, so these serialize on the shared
    // crate-wide lock and reset it first (repo convention for shared stores).
    use super::*;
    use unfer_protocol::{CertificateOpKind, CoinRef};

    #[test]
    fn mint_transfer_burn_roundtrip() {
        let _g = CERT_TESTS_LOCK.lock().unwrap();
        cert_clear();
        cert_set_authority(Some("did:unfer:authority".to_string()));

        let mint = CertificateOpKind::Mint {
            amount: 1000,
            owner: "did:unfer:alice".to_string(),
            blinding: [1u8; 32],
            source: Some("unfccc:vc:TEST".to_string()),
        };
        let ids = cert_apply("did:unfer:authority", &mint).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(cert_status()["total_supply"], 1000);

        let input = CoinRef {
            coin_id: ids[0],
            amount: 1000,
            owner: "did:unfer:alice".to_string(),
        };
        let transfer = CertificateOpKind::Transfer {
            inputs: vec![input],
            outputs: vec![CoinRef {
                coin_id: CertId([0u8; 32]),
                amount: 1000,
                owner: "did:unfer:bob".to_string(),
            }],
        };
        let ids = cert_apply("did:unfer:alice", &transfer).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(cert_status()["total_supply"], 1000);

        let burn = CertificateOpKind::Burn {
            inputs: vec![CoinRef {
                coin_id: ids[0],
                amount: 1000,
                owner: "did:unfer:bob".to_string(),
            }],
        };
        cert_apply("did:unfer:bob", &burn).unwrap();
        assert_eq!(cert_status()["total_supply"], 0);
        cert_clear();
    }

    #[test]
    fn mint_requires_authority() {
        let _g = CERT_TESTS_LOCK.lock().unwrap();
        cert_clear();
        cert_set_authority(Some("did:unfer:authority".to_string()));
        let mint = CertificateOpKind::Mint {
            amount: 100,
            owner: "did:unfer:alice".to_string(),
            blinding: [2u8; 32],
            source: None,
        };
        let err = cert_apply("did:unfer:nobody", &mint).unwrap_err();
        assert_eq!(err.code, unfer_protocol::Code::CERT_MINT_NOT_AUTHORIZED);
        cert_clear();
    }

    #[test]
    fn double_spend_rejected() {
        let _g = CERT_TESTS_LOCK.lock().unwrap();
        cert_clear();
        cert_set_authority(Some("did:unfer:authority".to_string()));
        let mint = CertificateOpKind::Mint {
            amount: 500,
            owner: "did:unfer:alice".to_string(),
            blinding: [3u8; 32],
            source: None,
        };
        let ids = cert_apply("did:unfer:authority", &mint).unwrap();
        let input = CoinRef {
            coin_id: ids[0],
            amount: 500,
            owner: "did:unfer:alice".to_string(),
        };
        let out = CoinRef {
            coin_id: CertId([0u8; 32]),
            amount: 500,
            owner: "did:unfer:alice".to_string(),
        };
        cert_apply(
            "did:unfer:alice",
            &CertificateOpKind::Transfer {
                inputs: vec![input.clone()],
                outputs: vec![out.clone()],
            },
        )
        .unwrap();
        let err = cert_apply(
            "did:unfer:alice",
            &CertificateOpKind::Transfer {
                inputs: vec![input],
                outputs: vec![out],
            },
        )
        .unwrap_err();
        assert_eq!(err.code, unfer_protocol::Code::CERT_DOUBLE_SPEND);
        cert_clear();
    }
}
//
// An immutable kernel-global audit trail of `uk_*` calls, each tagged with the
// caller that invoked it. The host loopback appends one entry per dispatch;
// `uk_audit_list` exposes it to a gatekeeper/operator; `uk_audit_clear` resets
// it (an operator action, never granted to untrusted modules).

/// Maximum audit entries retained before the oldest are dropped.
pub const AUDIT_CAPACITY: usize = 4096;

static AUDIT: Mutex<Option<VecDeque<AuditEntry>>> = Mutex::new(None);
static NEXT_AUDIT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Append an audit entry to the ring (no durable write).
fn ring_append_audit(entry: AuditEntry) {
    let mut guard = AUDIT.lock().unwrap_or_else(|e| e.into_inner());
    let queue = guard.get_or_insert_with(VecDeque::new);
    if queue.len() >= AUDIT_CAPACITY {
        queue.pop_front();
    }
    queue.push_back(entry);
}

/// Append an audit entry. `seq` is assigned here (monotonic, race-free). Returns
/// the assigned sequence number. Write-through: the entry is durably appended
/// (when a store is configured) *and* shadowed in the ring for fast reads.
pub fn store_audit(mut entry: AuditEntry) -> u64 {
    let seq = NEXT_AUDIT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    entry.seq = seq;
    if let Ok(json) = serde_json::to_vec(&entry) {
        durable_append(streams::AUDIT, &json);
    }
    ring_append_audit(entry);
    seq
}

/// Snapshot of the audit trail, newest first (the operator review order).
pub fn list_audit() -> Vec<AuditEntry> {
    let guard = AUDIT.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .as_ref()
        .map(|q| q.iter().rev().cloned().collect())
        .unwrap_or_default()
}

/// Empty the audit trail. Returns the number of entries removed.
pub fn clear_audit() -> usize {
    let mut guard = AUDIT.lock().unwrap_or_else(|e| e.into_inner());
    let len = guard.as_ref().map(VecDeque::len).unwrap_or(0);
    *guard = None;
    len
}

// ── observability context (S23/F22: AsyncLocal analog) ────────────────
//
// A per-call observability context seeded by the host loopback *before* dispatch
// and cleared *after*: `{trace_id, component, ...}`. The kernel threads those
// fields into every audit entry produced during the call (`context`), so a trace
// id never needs a global frontier — it lives on the same thread as the call.
// Binding: `component` is the dot-separated owner logger component (the F22
// `component = "kernel.audit"` norm); absent an explicit one, entries default to
// `"kernel.audit"`.

thread_local! {
    static OBSERVABILITY: std::cell::RefCell<serde_json::Value> =
        const { std::cell::RefCell::new(serde_json::Value::Null) };
}

/// Seed the current thread's per-call observability context.
pub fn set_observability(value: serde_json::Value) {
    OBSERVABILITY.with(|c| *c.borrow_mut() = value);
}

/// The current per-call observability context, if the host seeded one.
pub fn current_observability() -> Option<serde_json::Value> {
    OBSERVABILITY.with(|c| {
        let v = c.borrow().clone();
        if v.is_null() { None } else { Some(v) }
    })
}

/// The `component` member of the observability context (an owner, e.g.
/// `"kernel.audit"`); defaults to `"kernel.audit"` when unset ("Explicit; owner
/// is kernel.audit" — never foreign modules).
pub fn current_component() -> Option<String> {
    OBSERVABILITY.with(|c| {
        c.borrow()
            .get("component")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    })
}

/// Reset the current thread's per-call observability context.
pub fn clear_observability() {
    OBSERVABILITY.with(|c| *c.borrow_mut() = serde_json::Value::Null);
}

// ── dot-separated owner logger (S23/F2) + secret discipline ───────────
//
// The owner logger writes dot-separated owner component lines (`(component) message`)
// into a bounded ring sink drained by the operator console (`uk_owner_list`/`..._clear`).
// Discipline: the kernel *never logs secrets/prompts/keys*. `sanitize_sensitive`
// rewrites any value under a sensitive key (api_key, token, secret, password,
// authorization, ...) before a payload is stored in the audit trail or owner sink,
// so a leaked credential never lands in either store.

/// Redaction marker for sensitive values.
pub const REDACTED: &str = "***REDACTED***";

/// Sensitive key fragments (lower-cased substring match) — mirrors the edge
/// data-masking filter (`unfer_edge::mask`) so the kernel and the gateway share
/// the discipline.
const SENSITIVE_FRAGMENTS: &[&str] = &[
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "credential",
    "session_id",
];

fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SENSITIVE_FRAGMENTS.iter().any(|f| lower.contains(f))
}

/// Rewrite the string value of every sensitive-keyed field to [`REDACTED`],
/// recursively. Arrays/nested objects are walked in place.
pub fn sanitize_sensitive(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if is_sensitive_key(key) && v.is_string() {
                    *v = serde_json::Value::String(REDACTED.to_string());
                } else {
                    sanitize_sensitive(v);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                sanitize_sensitive(item);
            }
        }
        _ => {}
    }
}

/// Maximum owner-log lines retained before the oldest are dropped.
pub const OWNER_LOG_CAPACITY: usize = 512;

static OWNER_LOG: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Append a dot-separated owner component line to the ring (no durable write).
fn ring_append_owner(line: String) {
    let mut guard = OWNER_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let vec = guard.get_or_insert_with(Vec::new);
    if vec.len() >= OWNER_LOG_CAPACITY {
        vec.remove(0);
    }
    vec.push(line);
}

/// Append a dot-separated owner component line to the owner sink. Write-through:
/// the line is durably appended (when a store is configured) *and* shadowed in
/// the ring.
pub fn owner_log(component: &str, message: &str) {
    let line = format!("({component}) {message}");
    durable_append(streams::OWNER_LOG, line.as_bytes());
    ring_append_owner(line);
}

/// Snapshot of the owner sink, newest first (operator review order).
pub fn list_owner_log() -> Vec<String> {
    let guard = OWNER_LOG.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .as_ref()
        .map(|vec| vec.iter().rev().cloned().collect())
        .unwrap_or_default()
}

/// Empty the owner sink. Returns the number of lines removed.
pub fn clear_owner_log() -> usize {
    let mut guard = OWNER_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let len = guard.as_ref().map(Vec::len).unwrap_or(0);
    *guard = None;
    len
}

// ── agent registry (S6: AgentSpawner) ──────────────────────────────────
//
// A kernel-global registry of spawned sub-agents. `uk_agent_spawn` is the
// capability-minting chokepoint: it records the agent and its **fixed** grant
// set, refusing escalation (a sub-agent can only receive a subset of the
// caller's own grants). The host loopback enforces the bounded set on every
// call attributed to that agent (default-deny).

static AGENTS: Mutex<Option<HashMap<i64, AgentInfo>>> = Mutex::new(None);
static NEXT_AGENT: AtomicI64 = AtomicI64::new(1);

pub fn store_agent(agent: AgentInfo) -> i64 {
    let handle = NEXT_AGENT.fetch_add(1, Ordering::SeqCst);
    let mut guard = AGENTS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(handle, agent);
    handle
}

pub fn with_agent<R>(handle: i64, f: impl FnOnce(&AgentInfo) -> R) -> Option<R> {
    let guard = AGENTS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.get(&handle).map(f)
}

pub fn with_agent_mut<R>(handle: i64, f: impl FnOnce(&mut AgentInfo) -> R) -> Option<R> {
    let mut guard = AGENTS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    map.get_mut(&handle).map(f)
}

/// Snapshot of the whole agent registry, ordered by handle (== spawn order).
pub fn list_agents() -> Vec<(i64, AgentInfo)> {
    let guard = AGENTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut items: Vec<(i64, AgentInfo)> = guard
        .as_ref()
        .map(|map| map.iter().map(|(h, a)| (*h, a.clone())).collect())
        .unwrap_or_default();
    items.sort_by_key(|(h, _)| *h);
    items
}

// ── resource registry (S18/F17: introductions) ─────────────────────────
//
// A kernel-global registry of introduced resources — the single-mint chokepoint.
// Nothing is ambient: an id only becomes usable once `resource_introduce` mints it
// at this chokepoint, and then only for a session whose caller `GrantSet` includes
// the id in `resources`. `uk_request_resource` queues a `PendingResourceRequest`
// here for the human approval queue (resolved by the F18 `uk_gate_*` symbols).

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PendingResourceRequest {
    pub resource_id: String,
    pub requested_by: CallerTag,
}

static RESOURCES: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);
static PENDING_RESOURCE_REQUESTS: Mutex<Option<HashMap<i64, PendingResourceRequest>>> =
    Mutex::new(None);
static NEXT_RESOURCE_REQUEST: AtomicI64 = AtomicI64::new(1);

/// Mint a resource at the kernel chokepoint. `owner` is the introducing principal.
/// Already introduced → `RESOURCE_ALREADY_INTRODUCED`.
pub fn resource_introduce(resource_id: &str, owner: &str) -> Result<(), unfer_protocol::Code> {
    let mut guard = RESOURCES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    if map.contains_key(resource_id) {
        return Err(unfer_protocol::Code::RESOURCE_ALREADY_INTRODUCED);
    }
    map.insert(resource_id.to_string(), owner.to_string());
    Ok(())
}

/// Revoke a minted resource. Unknown id → `RESOURCE_NOT_FOUND`.
pub fn resource_forfeit(resource_id: &str) -> Result<(), unfer_protocol::Code> {
    let mut guard = RESOURCES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    if map.remove(resource_id).is_none() {
        return Err(unfer_protocol::Code::RESOURCE_NOT_FOUND);
    }
    Ok(())
}

/// Has this id ever been minted at the chokepoint?
pub fn resource_is_introduced(resource_id: &str) -> bool {
    let guard = RESOURCES.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().is_some_and(|m| m.contains_key(resource_id))
}

/// The resource-facing gate (F17). The trusted harness (no bounded grant set) may use any
/// minted resource; a bounded caller additionally needs the id in its `resources` grant —
/// the *introduction to this session*. Otherwise UK-4401 `RESOURCE_UNINTRODUCED`.
pub fn resource_authorized(
    resource_id: &str,
    ctx: &CallerContext,
) -> Result<(), unfer_protocol::Code> {
    match ctx.grants.as_ref() {
        None => {
            if resource_is_introduced(resource_id) {
                Ok(())
            } else {
                Err(unfer_protocol::Code::RESOURCE_NOT_FOUND)
            }
        }
        Some(grants)
            if grants.resources.iter().any(|r| r == resource_id)
                && resource_is_introduced(resource_id) =>
        {
            Ok(())
        }
        Some(_) => Err(unfer_protocol::Code::RESOURCE_UNINTRODUCED),
    }
}

/// Queue an approval-pending request for a resource the caller wants introduced to a
/// session. Returns the positive request handle (resolved later by `uk_gate_*`).
pub fn queue_resource_request(resource_id: &str, requested_by: CallerTag) -> i64 {
    let handle = NEXT_RESOURCE_REQUEST.fetch_add(1, Ordering::SeqCst);
    let mut guard = PENDING_RESOURCE_REQUESTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(
        handle,
        PendingResourceRequest {
            resource_id: resource_id.to_string(),
            requested_by,
        },
    );
    handle
}

/// Snapshot of the approval-pending resource requests, ordered by handle (request order).
pub fn list_pending_resource_requests() -> Vec<(i64, PendingResourceRequest)> {
    let guard = PENDING_RESOURCE_REQUESTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut items: Vec<(i64, PendingResourceRequest)> = guard
        .as_ref()
        .map(|map| map.iter().map(|(h, r)| (*h, r.clone())).collect())
        .unwrap_or_default();
    items.sort_by_key(|(h, _)| *h);
    items
}

#[cfg(test)]
mod buffer_proptests {
    // S17 (F16): the probe-then-copy buffer protocol must never panic and must
    // round-trip arbitrary payload lengths through store → read → free.
    use super::{free_buffer, read_buffer, store_buffer};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn buffer_roundtrip_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..4096usize)) {
            let boxed = bytes.clone().into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u8;
            let handle = store_buffer(ptr, bytes.len() as i64);

            // The documented read semantics: valid UTF-8 returns exactly those bytes,
            // invalid bytes degrade to "" — never a probe past the registered length.
            let expected = std::str::from_utf8(&bytes).unwrap_or("").to_string();
            let got = read_buffer(handle);
            prop_assert_eq!(got.as_deref(), Some(expected.as_str()));

            // Free reclaims exactly once; a second free is a clean miss.
            prop_assert!(free_buffer(handle));
            prop_assert!(!free_buffer(handle));
        }
    }

    #[test]
    fn unknown_handles_never_mispelled() {
        // Handles are allocated from a monotone counter starting at 0; a negative
        // handle can never be in use, so reads/frees must miss cleanly.
        assert_eq!(read_buffer(-1_000_000), None);
        assert!(!free_buffer(-1_000_000));
    }
}

#[cfg(test)]
mod meter_tests {
    // S25 (F24): the windowed meter is the single denial point for cost governance.
    use super::{MeterDecision, clear_meter, meter_consume, meter_status};
    use std::sync::Mutex;

    // The meter is a process-global shared store; serialize the tests so one test's
    // `clear_meter()` never wipes another's mid-run window (repo convention).
    static METER_TESTS_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn consumes_within_budget_then_denies() {
        let _g = METER_TESTS_LOCK.lock().unwrap();
        clear_meter();
        // Budget 3, no rate gate: three calls pass, the fourth is denied.
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::Allowed);
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::Allowed);
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::Allowed);
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::BudgetExceeded);
        // Budget is never mutated by a blocked request.
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::BudgetExceeded);
        assert_eq!(meter_consume("alice", 3, 0), MeterDecision::BudgetExceeded);
        clear_meter();
    }

    #[test]
    fn rate_limit_denies_independently_of_budget() {
        let _g = METER_TESTS_LOCK.lock().unwrap();
        clear_meter();
        // Rate limit 2; budget is large so only the rate gate trips.
        assert_eq!(meter_consume("bob", 100, 2), MeterDecision::Allowed);
        assert_eq!(meter_consume("bob", 100, 2), MeterDecision::Allowed);
        assert_eq!(meter_consume("bob", 100, 2), MeterDecision::RateLimited);
        clear_meter();
    }

    #[test]
    fn principals_are_isolated() {
        let _g = METER_TESTS_LOCK.lock().unwrap();
        clear_meter();
        assert_eq!(meter_consume("alice", 1, 0), MeterDecision::Allowed);
        assert_eq!(meter_consume("alice", 1, 0), MeterDecision::BudgetExceeded);
        // Bob starts fresh regardless of Alice's exhaustion.
        assert_eq!(meter_consume("bob", 1, 0), MeterDecision::Allowed);
        clear_meter();
    }

    #[test]
    fn status_is_read_only() {
        let _g = METER_TESTS_LOCK.lock().unwrap();
        clear_meter();
        meter_consume("carol", 5, 0);
        let status = meter_status("carol", 5);
        assert!(status.within_limits);
        assert_eq!(status.used, 1);
        assert_eq!(status.remaining, 4);
        assert_eq!(status.limit, 5);
        // Reading status never consumes.
        let again = meter_status("carol", 5);
        assert_eq!(status.used, again.used);
        clear_meter();
    }

    #[test]
    fn budget_below_rate_limit_is_the_gate() {
        let _g = METER_TESTS_LOCK.lock().unwrap();
        clear_meter();
        // Budget 1 beats a large rate limit: the budget is the binding constraint.
        assert_eq!(meter_consume("dave", 1, 100), MeterDecision::Allowed);
        assert_eq!(meter_consume("dave", 1, 100), MeterDecision::BudgetExceeded);
        clear_meter();
    }
}

#[cfg(test)]
mod sensitive_latch_tests {
    // S26 (F25): the sensitive-forward latch is sticky per-principal and only an
    // operator clears it. It never touches the approval queue.
    use super::{is_sensitive_latched, set_sensitive_latch};

    #[test]
    fn latch_is_sticky_until_cleared() {
        let who = "latch-sticky-alice";
        // The latch store is process-global and tests run in parallel, so each
        // test owns a distinct principal and never issues a global clear.
        assert!(!is_sensitive_latched(who));
        assert!(set_sensitive_latch(who, true));
        assert!(is_sensitive_latched(who));
        // Clearing restores; the return reflects the new state.
        assert!(!set_sensitive_latch(who, false));
        assert!(!is_sensitive_latched(who));
    }

    #[test]
    fn principals_are_independent() {
        let who = "indep-alice";
        assert!(set_sensitive_latch(who, true));
        // Bob is untouched by Alice's latch.
        assert!(!is_sensitive_latched("indep-bob"));
        assert!(!set_sensitive_latch(who, false));
    }
}

#[cfg(test)]
mod secret_vault_tests {
    // S27 (F26): the credential vault stores secrets encrypted at rest and grants
    // an opaque handle; only the owner dereferences; a live secret refuses to
    // serialize (snapshot/blueprint guard). The vault is a process-global store.
    use super::{
        vault_clear, vault_get_secret, vault_has_live_secrets, vault_put_secret,
        vault_revoke_secret,
    };
    use std::sync::Mutex;

    static VAULT_TESTS_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn put_get_roundtrip_under_owner() {
        let _g = VAULT_TESTS_LOCK.lock().unwrap();
        vault_clear();
        let handle = vault_put_secret("alice", b"sup3r-s3cret").expect("put");
        let got = vault_get_secret(handle, "alice").expect("get");
        assert_eq!(got, b"sup3r-s3cret");
        vault_clear();
    }

    #[test]
    fn non_owner_is_denied() {
        let _g = VAULT_TESTS_LOCK.lock().unwrap();
        vault_clear();
        let handle = vault_put_secret("alice", b"x").expect("put");
        // Bob cannot dereference Alice's secret.
        assert!(vault_get_secret(handle, "bob").is_err());
        vault_clear();
    }

    #[test]
    fn revoke_invalidates_handle() {
        let _g = VAULT_TESTS_LOCK.lock().unwrap();
        vault_clear();
        let handle = vault_put_secret("alice", b"y").expect("put");
        assert!(vault_revoke_secret(handle));
        assert!(vault_get_secret(handle, "alice").is_err());
        // Revoking an already-revoked handle is a miss.
        assert!(!vault_revoke_secret(handle));
        vault_clear();
    }

    #[test]
    fn live_secret_blocks_snapshot_export() {
        let _g = VAULT_TESTS_LOCK.lock().unwrap();
        vault_clear();
        assert!(!vault_has_live_secrets());
        vault_put_secret("alice", b"z").expect("put");
        assert!(vault_has_live_secrets());
        vault_clear();
        assert!(!vault_has_live_secrets());
    }
}
