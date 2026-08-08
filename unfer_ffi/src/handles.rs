use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use prob_kernel::Session;
use unfer_protocol::{
    ActionRecord, AgentInfo, AuditEntry, CallerTag, Diagnostic, EffectKind, EventQuery, GrantSet,
    KernelEvent,
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
}

pub fn store_session(session: Session) -> i64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
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
                }
                sub.events.push_back(event_json.clone());
            }
        }
    }
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
        KernelEvent::Error { .. } => "error",
        KernelEvent::PriorSet => "prior_set",
        KernelEvent::HamiltonianSet => "hamiltonian_set",
        KernelEvent::ActionPending { .. } | KernelEvent::ActionResolved { .. } => unreachable!(),
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
    if let Some(map) = guard.as_mut() {
        if let Some(BufPtr(ptr, len)) = map.remove(&handle) {
            unsafe {
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len as usize));
            }
            return true;
        }
    }
    false
}

/// Read buffer contents as a string. Returns None if handle is invalid.
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

pub fn store_action(record: ActionRecord) -> i64 {
    let handle = NEXT_ACTION.fetch_add(1, Ordering::SeqCst);
    let mut guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(handle, record);
    handle
}

pub fn with_action<R>(handle: i64, f: impl FnOnce(&ActionRecord) -> R) -> Option<R> {
    let guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.get(&handle).map(f)
}

pub fn with_action_mut<R>(
    handle: i64,
    f: impl FnOnce(&mut ActionRecord) -> R,
) -> Option<R> {
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
        .map(|map| {
            map.iter()
                .map(|(h, r)| (*h, r.clone()))
                .collect()
        })
        .unwrap_or_default();
    items.sort_by_key(|(h, _)| *h);
    items
}

/// Drop every action record (QA/console reset). The approval lane and vetted
/// markers are separate stores; this never touches either's semantics.
pub fn clear_actions() {
    let mut guard = ACTIONS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
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
#[derive(Debug, Clone)]
pub struct CallerContext {
    /// The audit tag (identity) of the current caller.
    pub tag: CallerTag,
    /// The caller's bounded grant set. `None` = unrestricted (kernel-level
    /// trust — e.g. a direct harness driving the ABI).
    pub grants: Option<GrantSet>,
}

impl Default for CallerContext {
    fn default() -> Self {
        Self {
            tag: CallerTag::default(),
            grants: None,
        }
    }
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
                self.tag.principal == principal
                    || grants.observers.iter().any(|o| o == principal)
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
    let prev = CALLER.with(|c| std::mem::replace(
        &mut *c.borrow_mut(),
        CallerContext { tag, grants },
    ));
    prev
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

/// Set or clear the console's vetted marker for `principal`. Returns the new value.
pub fn mark_vetted(principal: &str, vetted: bool) -> bool {
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

// ── audit trail (S6: human accountability) ─────────────────────────────
//
// An immutable kernel-global audit trail of `uk_*` calls, each tagged with the
// caller that invoked it. The host loopback appends one entry per dispatch;
// `uk_audit_list` exposes it to a gatekeeper/operator; `uk_audit_clear` resets
// it (an operator action, never granted to untrusted modules).

/// Maximum audit entries retained before the oldest are dropped.
pub const AUDIT_CAPACITY: usize = 4096;

static AUDIT: Mutex<Option<VecDeque<AuditEntry>>> = Mutex::new(None);
static NEXT_AUDIT_SEQ: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Append an audit entry. `seq` is assigned here (monotonic, race-free). Returns
/// the assigned sequence number.
pub fn store_audit(mut entry: AuditEntry) -> u64 {
    let seq = NEXT_AUDIT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    entry.seq = seq;
    let mut guard = AUDIT.lock().unwrap_or_else(|e| e.into_inner());
    let queue = guard.get_or_insert_with(VecDeque::new);
    if queue.len() >= AUDIT_CAPACITY {
        queue.pop_front();
    }
    queue.push_back(entry);
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
        std::cell::RefCell::new(serde_json::Value::Null);
}

/// Seed the current thread's per-call observability context.
pub fn set_observability(value: serde_json::Value) {
    OBSERVABILITY.with(|c| *c.borrow_mut() = value);
}

/// The current per-call observability context, if the host seeded one.
pub fn current_observability() -> Option<serde_json::Value> {
    OBSERVABILITY.with(|c| {
        let v = c.borrow().clone();
        if v.is_null() {
            None
        } else {
            Some(v)
        }
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

/// Append a dot-separated owner component line to the owner sink.
pub fn owner_log(component: &str, message: &str) {
    let line = format!("({component}) {message}");
    let mut guard = OWNER_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let vec = guard.get_or_insert_with(Vec::new);
    if vec.len() >= OWNER_LOG_CAPACITY {
        vec.remove(0);
    }
    vec.push(line);
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

pub fn with_agent_mut<R>(
    handle: i64,
    f: impl FnOnce(&mut AgentInfo) -> R,
) -> Option<R> {
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
    guard
        .as_ref()
        .is_some_and(|m| m.contains_key(resource_id))
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
    let mut guard = PENDING_RESOURCE_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
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
    let guard = PENDING_RESOURCE_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
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
