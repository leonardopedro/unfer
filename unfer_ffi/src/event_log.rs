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

/// An event with its position in the global stream.
#[derive(Debug, Clone, PartialEq)]
pub struct CursoredEvent {
    /// Monotonic, starts at 1, never reused.
    pub cursor: u64,
    /// The model handle the event came from. Consumers filter on this.
    pub handle: i64,
    pub event: KernelEvent,
}

static LOG: Mutex<Option<VecDeque<CursoredEvent>>> = Mutex::new(None);
static NEXT_CURSOR: AtomicU64 = AtomicU64::new(1);
/// Events discarded because the log was full. Counted globally so a consumer can
/// detect a gap it did not cause.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Append an event and return its cursor.
pub fn record_event(handle: i64, event: KernelEvent) -> u64 {
    let cursor = NEXT_CURSOR.fetch_add(1, Ordering::SeqCst);
    let mut guard = LOG.lock().unwrap_or_else(|e| e.into_inner());
    let log = guard.get_or_insert_with(VecDeque::new);
    if log.len() >= EVENT_LOG_CAPACITY {
        log.pop_front();
        DROPPED.fetch_add(1, Ordering::SeqCst);
    }
    log.push_back(CursoredEvent {
        cursor,
        handle,
        event,
    });
    cursor
}

/// Events strictly after `since_cursor`, oldest first, at most `max`.
///
/// `since_cursor = 0` replays from the beginning of the retained log. A cursor
/// older than what is retained does **not** silently return a short stream: the
/// caller can compare `oldest_available()` against its own cursor to detect that
/// it fell behind.
pub fn events_since(since_cursor: u64, max: usize) -> Vec<CursoredEvent> {
    if max == 0 {
        return Vec::new();
    }
    let guard = LOG.lock().unwrap_or_else(|e| e.into_inner());
    let Some(log) = guard.as_ref() else {
        return Vec::new();
    };
    log.iter()
        .filter(|e| e.cursor > since_cursor)
        .take(max)
        .cloned()
        .collect()
}

/// The oldest cursor still retained, or `None` when the log is empty.
pub fn oldest_available() -> Option<u64> {
    let guard = LOG.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().and_then(|l| l.front()).map(|e| e.cursor)
}

/// The newest cursor issued, whether or not it is still retained.
pub fn latest_cursor() -> u64 {
    NEXT_CURSOR.load(Ordering::SeqCst).saturating_sub(1)
}

/// Events discarded because the log was full, process-wide.
pub fn dropped_count() -> u64 {
    DROPPED.load(Ordering::SeqCst)
}

/// Clear the log and reset the cursor.
pub fn reset_event_log() {
    let mut guard = LOG.lock().unwrap_or_else(|e| e.into_inner());
    *guard = None;
    NEXT_CURSOR.store(1, Ordering::SeqCst);
    DROPPED.store(0, Ordering::SeqCst);
}

/// Has this consumer fallen behind the retained window?
///
/// True means a gap: at least one event it had not seen is gone. Returning a
/// short stream instead would let a consumer believe it was caught up.
pub fn has_gap(since_cursor: u64) -> bool {
    match oldest_available() {
        Some(oldest) => since_cursor != 0 && since_cursor + 1 < oldest,
        None => false,
    }
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

    /// Serialised: the log is process-global state, and these tests assert on
    /// absolute cursor values.
    fn with_log(f: impl FnOnce()) {
        let g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_event_log();
        f();
        reset_event_log();
        drop(g);
    }

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cursors_are_strictly_increasing_and_never_reused() {
        with_log(|| {
            let a = record_event(1, ev(1));
            let b = record_event(1, ev(2));
            let c = record_event(2, ev(3));
            assert!(a < b && b < c, "cursors must increase: {a} {b} {c}");
            assert_eq!(c, latest_cursor());
        });
    }

    #[test]
    fn events_since_returns_only_what_is_newer() {
        with_log(|| {
            record_event(1, ev(1));
            let second = record_event(1, ev(2));
            record_event(1, ev(3));

            let from_first = events_since(1, 10);
            assert_eq!(2, from_first.len(), "only the two after cursor 1");

            let after_second = events_since(second, 10);
            assert_eq!(1, after_second.len());

            assert!(
                events_since(latest_cursor(), 10).is_empty(),
                "a caught-up consumer must get nothing"
            );
        });
    }

    #[test]
    fn cursor_zero_replays_the_retained_log() {
        with_log(|| {
            record_event(1, ev(1));
            record_event(1, ev(2));
            assert_eq!(2, events_since(0, 10).len());
        });
    }

    #[test]
    fn two_consumers_each_see_every_event_exactly_once() {
        // The multi-worker requirement, and the reason this is a cursored log
        // rather than a queue: a consumer that drains a queue destroys the events
        // for everyone else.
        with_log(|| {
            let total = 50usize;
            for i in 0..total {
                record_event(1, ev(i as u64));
            }

            let mut a = 0u64;
            let mut b = 0u64;
            let mut seen_a = Vec::new();
            let mut seen_b = Vec::new();

            // Interleaved, in batches, as two independent workers would.
            loop {
                let batch_a = events_since(a, 7);
                let batch_b = events_since(b, 5);
                if batch_a.is_empty() && batch_b.is_empty() {
                    break;
                }
                if let Some(last) = batch_a.last() {
                    a = last.cursor;
                    seen_a.extend(batch_a.into_iter().map(|e| e.cursor));
                }
                if let Some(last) = batch_b.last() {
                    b = last.cursor;
                    seen_b.extend(batch_b.into_iter().map(|e| e.cursor));
                }
            }

            assert_eq!(
                total,
                seen_a.len(),
                "consumer A saw {} of {total}",
                seen_a.len()
            );
            assert_eq!(
                total,
                seen_b.len(),
                "consumer B saw {} of {total}",
                seen_b.len()
            );

            let expect: Vec<u64> = (1..=total as u64).collect();
            assert_eq!(expect, seen_a, "A must see every event once, in order");
            assert_eq!(
                seen_a, seen_b,
                "both consumers must observe the same stream"
            );
        });
    }

    #[test]
    fn a_consumer_resuming_from_its_checkpoint_sees_nothing_already_seen() {
        // "The cursor survives restart": a fresh consumer holding a cursor must be
        // able to continue without replaying or skipping.
        with_log(|| {
            record_event(1, ev(1));
            let checkpoint = record_event(1, ev(2));
            for i in 3..=10 {
                record_event(1, ev(i));
            }
            let resumed = events_since(checkpoint, 100);
            assert_eq!(8, resumed.len(), "events 3..=10 is eight, not seven");
            assert!(
                resumed.iter().all(|e| e.cursor > checkpoint),
                "nothing at or before the checkpoint may be replayed"
            );
        });
    }

    #[test]
    fn max_caps_the_batch_without_losing_order() {
        with_log(|| {
            for i in 0..10u64 {
                record_event(1, ev(i));
            }
            let batch = events_since(0, 4);
            assert_eq!(4, batch.len());
            let cursors: Vec<u64> = batch.iter().map(|e| e.cursor).collect();
            assert_eq!(vec![1, 2, 3, 4], cursors);
            assert!(events_since(0, 0).is_empty(), "max=0 yields nothing");
        });
    }

    #[test]
    fn the_log_reports_the_gap_it_created_rather_than_hiding_it() {
        // A short stream must not be mistaken for a caught-up consumer. This is
        // the failure the subscription path has: a dropped event is
        // indistinguishable from one that never happened.
        with_log(|| {
            let consumer_cursor = record_event(1, ev(0));
            for i in 0..(EVENT_LOG_CAPACITY + 10) as u64 {
                record_event(1, ev(i));
            }

            assert!(
                has_gap(consumer_cursor),
                "a consumer left behind must see a gap"
            );
            // Derive the expectation from what is actually retained rather than
            // restating the arithmetic — the first version of this assertion
            // asserted `pushes - capacity` and was wrong twice.
            let pushed = (EVENT_LOG_CAPACITY + 10) + 1;
            let retained = events_since(0, usize::MAX).len();
            assert_eq!(
                (pushed - retained) as u64,
                dropped_count(),
                "every eviction is counted: pushed {pushed}, retained {retained}, capacity {EVENT_LOG_CAPACITY}"
            );

            // What it does get back is contiguous from the oldest retained entry,
            // so the consumer can detect and report rather than silently diverge.
            let got = events_since(consumer_cursor, 100);
            assert!(!got.is_empty());
            assert!(got.windows(2).all(|w| w[0].cursor + 1 == w[1].cursor));
        });
    }

    #[test]
    fn a_caught_up_consumer_reports_no_gap() {
        with_log(|| {
            record_event(1, ev(1));
            let caught_up = latest_cursor();
            assert!(!has_gap(caught_up));
            assert!(
                !has_gap(0),
                "a fresh consumer has no history to have missed"
            );
        });
    }

    #[test]
    fn the_empty_log_answers_empty_rather_than_panicking() {
        with_log(|| {
            assert!(events_since(0, 10).is_empty());
            assert_eq!(None, oldest_available());
            assert_eq!(0, latest_cursor());
            assert!(!has_gap(5));
        });
    }

    #[test]
    fn events_carry_the_handle_they_came_from() {
        // Consumers filter per model, so the handle has to travel with the event
        // rather than be implied by a subscription.
        with_log(|| {
            record_event(7, ev(1));
            record_event(9, ev(2));
            let all = events_since(0, 10);
            assert_eq!(7, all[0].handle);
            assert_eq!(9, all[1].handle);
        });
    }
}
