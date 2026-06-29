use std::collections::{HashMap, VecDeque};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use prob_kernel::Session;
use unfer_protocol::{Diagnostic, EventQuery, KernelEvent};

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
