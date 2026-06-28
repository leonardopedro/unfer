use std::collections::{HashMap, VecDeque};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

use prob_kernel::Session;
use unfer_protocol::Diagnostic;

/// Maximum events retained per model before oldest are dropped.
pub const EVENT_QUEUE_CAPACITY: usize = 64;

struct SessionEntry {
    session: Session,
    last_result: String,
    /// Per-model event queue. Bounded to EVENT_QUEUE_CAPACITY; when full the
    /// oldest event is dropped so slow consumers never block the kernel.
    events: VecDeque<String>,
}

static HANDLES: Mutex<Option<HashMap<i64, SessionEntry>>> = Mutex::new(None);
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
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
            events: VecDeque::new(),
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

/// Push a JSON event string onto the model's event queue.
/// If the queue is full, the oldest event is silently dropped.
pub fn push_event(handle: i64, event_json: String) {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut()
        && let Some(entry) = map.get_mut(&handle)
    {
        if entry.events.len() >= EVENT_QUEUE_CAPACITY {
            entry.events.pop_front();
        }
        entry.events.push_back(event_json);
    }
}

/// Peek at the next event without removing it.
/// Returns `None` if the handle is invalid, `Some(None)` if queue is empty,
/// `Some(Some(json))` for the next pending event (still in queue).
pub fn peek_event(handle: i64) -> Option<Option<String>> {
    let guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    let entry = map.get(&handle)?;
    Some(entry.events.front().cloned())
}

/// Pop one event from the model's queue.
/// Returns `None` if the handle is invalid, `Some(None)` if queue is empty,
/// `Some(Some(json))` for the next pending event.
pub fn poll_event(handle: i64) -> Option<Option<String>> {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    let entry = map.get_mut(&handle)?;
    Some(entry.events.pop_front())
}

/// Drain all pending events from the model's queue.
/// Returns `None` if the handle is invalid.
pub fn drain_events(handle: i64) -> Option<Vec<String>> {
    let mut guard = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_mut()?;
    let entry = map.get_mut(&handle)?;
    Some(entry.events.drain(..).collect())
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
