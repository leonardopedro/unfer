//! C1 — a cursored event log.
//!
//! The existing event path is **subscription-based**: `push_event` fans a
//! `KernelEvent` out to every subscription whose query matches, each with its own
//! bounded queue that silently drops its oldest entry on overflow. That is the
//! right shape for "tell me about this model" and the wrong shape for "run as a
//! worker and be sure you missed nothing" — a dropped event is indistinguishable
//! from one that never happened, and there is no way for a consumer to say "I got
//! up to 41, send me 42 onwards" after a restart.
//!
//! So this is a *separate*, global log with monotonic cursors. It does not
//! replace the subscription path; it answers a question subscriptions cannot.
//!
//! The properties that matter, and which the tests assert:
//!
//!   * cursors are strictly increasing and never reused;
//!   * **two consumers with independent cursors each see every event exactly
//!     once** — that is the multi-worker requirement, and it is why this is a log
//!     with cursors rather than a queue you drain;
//!   * a consumer resuming from its own checkpoint sees only what came after;
//!   * the bounded log **reports** what it dropped instead of pretending the
//!     stream was contiguous. A consumer can tell that it has a gap.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use unfer_protocol::KernelEvent;

/// Bounded log depth. Deep enough that a consumer reconnects after a restart
/// without loss, shallow enough to stay cheap per model.
pub const EVENT_LOG_CAPACITY: usize = 4096;

/// An event with its position in the stream.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CursoredEvent {
    /// Monotonic, starts at 1, never reused.
    pub cursor: u64,
    /// The model handle the event came from. Consumers filter on this.
    pub handle: i64,
    pub event: KernelEvent,
}

/// A bounded, cursored event log.
///
/// A type rather than a bag of globals, because the first version was module-level
/// statics plus a `reset()` for tests. That was a trap: `reset` rewinds the cursor
/// counter to 1 while the queue may still hold entries, so cursors go *backwards*
/// for every other reader. It produced a genuine `6, 1, 2, 3` in a test run — a
/// cursor regression in a structure whose entire contract is that cursors never
/// go backwards. Serialising the tests with a mutex hid it while leaving the
/// hazard in place for anyone who reset without taking the lock.
///
/// So the state is now owned by an instance. `GLOBAL` is the kernel's live log;
/// the tests each build their own and never reset anything.
pub struct Log {
    log: Mutex<VecDeque<CursoredEvent>>,
    next: AtomicU64,
    dropped: AtomicU64,
    capacity: usize,
}

impl Log {
    pub const fn new(capacity: usize) -> Self {
        Self {
            log: Mutex::new(VecDeque::new()),
            next: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
            capacity,
        }
    }

    /// Retained depth. Only the tests ask; the log enforces its own bound.
    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append an event and return its cursor.
    pub fn record(&self, handle: i64, event: KernelEvent) -> u64 {
        let cursor = self.next.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.log.lock().unwrap_or_else(|e| e.into_inner());
        if guard.len() >= self.capacity {
            guard.pop_front();
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
        guard.push_back(CursoredEvent {
            cursor,
            handle,
            event,
        });
        cursor
    }

    /// Events strictly after `since_cursor`, oldest first, at most `max`.
    ///
    /// `since_cursor = 0` replays from the beginning of the retained log. A
    /// cursor older than what is retained does **not** silently return a short
    /// stream: compare `oldest_available()` against your own cursor, or read
    /// `has_gap`, to detect that you fell behind.
    pub fn events_since(&self, since_cursor: u64, max: usize) -> Vec<CursoredEvent> {
        if max == 0 {
            return Vec::new();
        }
        let guard = self.log.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .iter()
            .filter(|e| e.cursor > since_cursor)
            .take(max)
            .cloned()
            .collect()
    }

    /// The oldest cursor still retained, or `None` when the log is empty.
    pub fn oldest_available(&self) -> Option<u64> {
        let guard = self.log.lock().unwrap_or_else(|e| e.into_inner());
        guard.front().map(|e| e.cursor)
    }

    /// The newest cursor issued, whether or not it is still retained.
    pub fn latest_cursor(&self) -> u64 {
        self.next.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Events discarded because the log was full.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::SeqCst)
    }

    /// Has this consumer fallen behind the retained window?
    ///
    /// True means a gap: at least one event it had not seen is gone. Returning a
    /// short stream instead would let a consumer believe it was caught up.
    pub fn has_gap(&self, since_cursor: u64) -> bool {
        match self.oldest_available() {
            Some(oldest) => since_cursor != 0 && since_cursor + 1 < oldest,
            None => false,
        }
    }
}

/// The kernel's live log. `uk_events_poll` reads this.
pub static GLOBAL: Log = Log::new(EVENT_LOG_CAPACITY);

thread_local! {
    /// Test-only: the log this thread should use instead of `GLOBAL`.
    ///
    /// A thread-local, not a global, because `cargo test` runs tests in parallel
    /// threads and a process-wide override would let them clobber each other --
    /// which is exactly what happened with the shared log: the gap test floods
    /// 4160 events, evicts everything another test had recorded, and that test
    /// then sees an empty stream. Per-thread, each test gets its own log and the
    /// flood is harmless.
    ///
    /// `None` in production: this arm is compiled out entirely.
    static TEST_ACTIVE: std::cell::RefCell<Option<&'static Log>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a log for the current thread. Test-only; pass `None` to restore.
#[cfg(test)]
pub fn set_active(log: Option<&'static Log>) {
    TEST_ACTIVE.with(|a| *a.borrow_mut() = log);
}

/// The log this thread reads and writes.
pub fn active() -> &'static Log {
    #[cfg(test)]
    {
        let chosen = TEST_ACTIVE.with(|a| *a.borrow());
        if let Some(log) = chosen {
            return log;
        }
    }
    &GLOBAL
}

/// Install a fresh log for the current thread, returning it.
///
/// Test-only. The log is leaked on purpose: it has to be `&'static` so it can be
/// handed to a thread-local slot, and a test log's lifetime ending with the test
/// is exactly what we want. Tests are short and the memory is a few kB each.
#[cfg(test)]
pub fn isolated(capacity: usize) -> &'static Log {
    let log: &'static Log = Box::leak(Box::new(Log::new(capacity)));
    set_active(Some(log));
    log
}

pub fn record_event(handle: i64, event: KernelEvent) -> u64 {
    active().record(handle, event)
}

pub fn events_since(since_cursor: u64, max: usize) -> Vec<CursoredEvent> {
    active().events_since(since_cursor, max)
}

pub fn oldest_available() -> Option<u64> {
    active().oldest_available()
}

pub fn latest_cursor() -> u64 {
    active().latest_cursor()
}

pub fn dropped_count() -> u64 {
    active().dropped_count()
}

pub fn has_gap(since_cursor: u64) -> bool {
    active().has_gap(since_cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(n: u64) -> KernelEvent {
        KernelEvent::Evolved {
            t: n as f64,
            norm: 1.0,
            solve_ms: 0,
        }
    }

    /// Each test gets its own log. No lock, no reset, no shared cursor space --
    /// which is what let these tests interfere with each other before.
    fn log() -> Log {
        Log::new(64)
    }

    #[test]
    fn cursors_are_strictly_increasing_and_never_reused() {
        let log = log();
        let a = log.record(1, ev(1));
        let b = log.record(1, ev(2));
        let c = log.record(2, ev(3));
        assert!(a < b && b < c, "cursors must increase: {a} {b} {c}");
        assert_eq!(c, log.latest_cursor());
        assert_eq!((1, 2, 3), (a, b, c), "cursors start at 1 and are dense");
    }

    #[test]
    fn events_since_returns_only_what_is_newer() {
        let log = log();
        log.record(1, ev(1));
        let second = log.record(1, ev(2));
        log.record(1, ev(3));

        assert_eq!(
            2,
            log.events_since(1, 10).len(),
            "only the two after cursor 1"
        );
        assert_eq!(1, log.events_since(second, 10).len());
        assert!(
            log.events_since(0, 10).iter().all(|e| e.cursor > 0),
            "cursor 0 replays the retained log"
        );
    }

    #[test]
    fn a_resumed_consumer_never_replays_what_it_already_saw() {
        let log = log();
        log.record(1, ev(1));
        let checkpoint = log.record(1, ev(2));
        for i in 3..=10 {
            log.record(1, ev(i));
        }
        let resumed = log.events_since(checkpoint, 100);
        assert_eq!(8, resumed.len(), "events 3..=10 is eight, not seven");
        assert!(
            resumed.iter().all(|e| e.cursor > checkpoint),
            "nothing at or before the checkpoint may be replayed"
        );
    }

    #[test]
    fn cursor_zero_replays_the_retained_log() {
        // A consumer with no checkpoint asks for everything the log still holds.
        // Distinct from `events_since(nonzero)`, which must not replay.
        let log = log();
        for i in 0..5u64 {
            log.record(1, ev(i));
        }
        let all = log.events_since(0, usize::MAX);
        assert_eq!(5, all.len(), "cursor 0 replays the whole retained log");
        assert!(all.iter().all(|e| e.cursor > 0), "and every cursor is real");
    }

    #[test]
    fn max_caps_the_batch_without_losing_order() {
        let log = log();
        for i in 0..10u64 {
            log.record(1, ev(i));
        }
        let batch = log.events_since(0, 4);
        assert_eq!(4, batch.len());
        let cursors: Vec<u64> = batch.iter().map(|e| e.cursor).collect();
        assert_eq!(vec![1, 2, 3, 4], cursors, "oldest first, no reordering");
        assert!(log.events_since(0, 0).is_empty(), "max=0 yields nothing");
    }

    #[test]
    fn the_log_reports_the_gap_it_created_rather_than_hiding_it() {
        // A short stream must not be mistaken for a caught-up consumer. This is
        // the failure the subscription path has: a dropped event is
        // indistinguishable from one that never happened.
        let log = log();
        let consumer_cursor = log.record(1, ev(0));
        for i in 0..(log.capacity() + 10) as u64 {
            log.record(1, ev(i));
        }

        assert!(
            log.has_gap(consumer_cursor),
            "a consumer left behind must see a gap"
        );
        // Derive the expectation from what is actually retained rather than
        // restating the arithmetic -- the first version asserted
        // `pushes - capacity` and was wrong twice.
        let pushed = log.latest_cursor();
        let retained = log.events_since(0, usize::MAX).len();
        assert_eq!(
            pushed - retained as u64,
            log.dropped_count(),
            "every cursor not retained was counted as dropped"
        );
        assert!(log.dropped_count() > 0, "this log really did overflow");
    }

    #[test]
    fn a_caught_up_consumer_reports_no_gap() {
        let log = log();
        for i in 0..(log.capacity() + 10) as u64 {
            log.record(1, ev(i));
        }
        let caught_up = log.latest_cursor();
        assert!(!log.has_gap(caught_up), "nothing was missed");
        assert!(
            log.events_since(caught_up, 10).is_empty(),
            "and there is nothing new either"
        );
    }

    #[test]
    fn the_empty_log_answers_empty_rather_than_panicking() {
        let log = log();
        assert!(log.events_since(0, 10).is_empty());
        assert_eq!(None, log.oldest_available());
        assert_eq!(0, log.latest_cursor());
        assert!(!log.has_gap(0), "an empty log has no gap to report");
    }

    #[test]
    fn events_carry_the_handle_they_came_from() {
        let log = log();
        log.record(7, ev(1));
        log.record(9, ev(2));
        let handles: Vec<i64> = log.events_since(0, 10).iter().map(|e| e.handle).collect();
        assert_eq!(vec![7, 9], handles, "a consumer can filter by model");
    }

    #[test]
    fn two_consumers_each_see_every_event_exactly_once() {
        // The multi-worker requirement, and the reason this is a cursored log
        // rather than a queue: draining a queue would destroy the other
        // consumer's copy.
        let log = log();
        let mut a_seen = Vec::new();
        let mut b_seen = Vec::new();
        let mut a_cursor = 0;
        let mut b_cursor = 0;

        for i in 0..50u64 {
            log.record(1, ev(i));
            // Interleave different batch sizes so neither consumer can rely on a
            // convenient stride.
            if i % 3 == 0 {
                a_seen.extend(log.events_since(a_cursor, 2).iter().map(|e| e.cursor));
                a_cursor = *a_seen.last().unwrap();
            }
            if i % 5 == 0 {
                b_seen.extend(log.events_since(b_cursor, 7).iter().map(|e| e.cursor));
                b_cursor = *b_seen.last().unwrap();
            }
        }
        a_seen.extend(log.events_since(a_cursor, 100).iter().map(|e| e.cursor));
        b_seen.extend(log.events_since(b_cursor, 100).iter().map(|e| e.cursor));

        let expected: Vec<u64> = (1..=50).collect();
        assert_eq!(expected, a_seen, "consumer A saw each event exactly once");
        assert_eq!(expected, b_seen, "consumer B saw each event exactly once");
    }
}
