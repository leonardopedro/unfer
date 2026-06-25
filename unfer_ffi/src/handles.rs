use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use prob_kernel::Session;
use unfer_protocol::{Diagnostic, EventPredicate};

struct SessionEntry {
    session: Session,
    last_result: String,
}

struct SubscriptionEntry {
    model_handle: i64,
    query: EventPredicate,
}

static HANDLES: Mutex<Option<HashMap<i64, SessionEntry>>> = Mutex::new(None);
static SUBSCRIPTIONS: Mutex<Option<HashMap<i64, SubscriptionEntry>>> = Mutex::new(None);
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

pub fn free_session(handle: i64) -> bool {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .as_mut()
        .is_some_and(|map| map.remove(&handle).is_some())
}

pub fn set_last_error(diag: &Diagnostic) {
    let json = serde_json::to_string(diag).unwrap_or_else(|_| "{}".to_string());
    LAST_ERROR.with(|e| *e.borrow_mut() = json);
}

pub fn get_last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}

pub fn store_subscription(model_handle: i64, query: EventPredicate) -> i64 {
    let sub = NEXT_SUB.fetch_add(1, Ordering::SeqCst);
    let mut guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(
        sub,
        SubscriptionEntry {
            model_handle,
            query,
        },
    );
    sub
}

pub fn get_subscription(sub: i64) -> Option<(i64, EventPredicate)> {
    let guard = SUBSCRIPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    let entry = map.get(&sub)?;
    Some((entry.model_handle, entry.query.clone()))
}
