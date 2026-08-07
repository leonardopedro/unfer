use std::collections::{HashMap, VecDeque};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use prob_kernel::Session;
use unfer_protocol::{ActionRecord, Diagnostic, EventQuery, KernelEvent};

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
