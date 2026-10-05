//! G9 (b): task-scoped budget nudges.
//!
//! A worker with a deadline needs to be told to stop at the right moments, and
//! the moments are policy rather than mechanism: "stop claiming new scope" and
//! "merge what you have or report blocked" are the same two sentences whatever
//! the task is.
//!
//! ## Pure, and told the remaining time rather than reading a clock
//!
//! [`due`] takes *seconds remaining* and returns the checkpoints that have come
//! up. It does not read a clock, and nothing here stores a deadline. Two reasons,
//! and they are why this is a module rather than a `sleep` in the runner:
//!
//! - It is testable by passing a number. Every boundary case (exactly on a
//!   checkpoint, one second either side, a zero-length task) is a table entry.
//! - It stays consistent with the rest of the workspace's refusal to put a
//!   wall-clock in a state machine. The *harness* owns the clock; this owns the
//!   policy. (The G9 spawn pacer does use elapsed time, and that is consistent:
//!   it exists precisely to measure real elapsed time and has no history to
//!   replay.)
//!
//! ## Crossed checkpoints fire once
//!
//! A worker that was not polled for an hour should hear about the deadlines it
//! slept through, once each, not once per poll. [`due`] therefore returns each
//! due checkpoint exactly once per call and the caller tracks what it has already
//! delivered. That is why [`Nudge::kind`] exists: it is the dedup key.

use serde::{Deserialize, Serialize};

/// What a nudge tells the worker to do.
///
/// Ordered by urgency, ascending. The ordering is load-bearing: [`due`] sorts by
/// it, and a schedule is data that may be declared in any order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NudgeKind {
    /// Finish the current step; do not start a new one.
    WrapUp,
    /// Stop claiming new scope. Work in flight is still yours to finish.
    StopClaiming,
    /// Merge what you have, or say explicitly that you are blocked and why.
    MergeOrReportBlocked,
}

impl NudgeKind {
    pub const ALL: &'static [NudgeKind] = &[
        NudgeKind::WrapUp,
        NudgeKind::StopClaiming,
        NudgeKind::MergeOrReportBlocked,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            NudgeKind::WrapUp => "wrap_up",
            NudgeKind::StopClaiming => "stop_claiming",
            NudgeKind::MergeOrReportBlocked => "merge_or_report_blocked",
        }
    }

    /// Parse the wire spelling, case-insensitively.
    pub fn parse(s: &str) -> Option<NudgeKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wrap_up" | "wrapup" => Some(NudgeKind::WrapUp),
            "stop_claiming" | "stopclaiming" => Some(NudgeKind::StopClaiming),
            "merge_or_report_blocked" => Some(NudgeKind::MergeOrReportBlocked),
            _ => None,
        }
    }

    /// The instruction as a worker should read it.
    pub fn instruction(self) -> &'static str {
        match self {
            NudgeKind::WrapUp => "finish the current step; do not start a new one.",
            NudgeKind::StopClaiming => "stop claiming new scope; finish what you already hold.",
            NudgeKind::MergeOrReportBlocked => {
                "merge what you have, or report that you are blocked and why."
            }
        }
    }
}

/// A checkpoint: fire this nudge when the remaining time drops to
/// `at_secs_remaining` or below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub at_secs_remaining: u64,
    pub kind: NudgeKind,
}

/// The default schedule.
///
/// Deliberately short. A nudge schedule is a prompt, and a prompt nobody reads
/// is worse than no prompt: five reminders train a worker to stop reading
/// reminders.
pub const DEFAULT_CHECKPOINTS: &[Checkpoint] = &[
    // Three quarters of an hour left.
    Checkpoint {
        at_secs_remaining: 45 * 60,
        kind: NudgeKind::StopClaiming,
    },
    // Five minutes left.
    Checkpoint {
        at_secs_remaining: 5 * 60,
        kind: NudgeKind::MergeOrReportBlocked,
    },
];

/// One fired nudge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nudge {
    pub kind: NudgeKind,
    /// Seconds remaining when it fired. Recorded so a reader can see how late the
    /// worker was, which is a fact about the harness worth having.
    pub remaining_secs: u64,
    /// Board cursor this nudge was written at, once it has been.
    #[serde(default)]
    pub cursor: u64,
}

/// The checkpoints that have come up at `remaining_secs`, least urgent first.
///
/// Ascending urgency, so the most urgent checkpoint the worker has reached reads
/// **last**. A worker that reads only the final line then reads the one that
/// matters most, which is the common case for a worker skimming a board.
pub fn due(remaining_secs: u64, checkpoints: &[Checkpoint]) -> Vec<Nudge> {
    let mut fired: Vec<Nudge> = checkpoints
        .iter()
        .filter(|c| remaining_secs <= c.at_secs_remaining)
        .map(|c| Nudge {
            kind: c.kind,
            remaining_secs,
            cursor: 0,
        })
        .collect();
    fired.sort_by_key(|n| n.kind);
    fired
}

/// The text written to the board for a nudge.
///
/// Written as a `BoardKind::Observed`, not a `FAIL` or a `CLAIM`: a nudge states
/// a fact about time, commits to nothing, and must apply immediately rather than
/// queueing for approval. A worker blocked on an approval lane cannot act on a
/// "wrap up" reminder.
pub fn board_text(n: &Nudge, worker: &str) -> String {
    format!("nudge to {}: {}", worker, n.kind.instruction())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(m: u64) -> u64 {
        m * 60
    }

    #[test]
    fn nothing_fires_while_there_is_plenty_of_time() {
        assert!(due(secs(60), DEFAULT_CHECKPOINTS).is_empty());
        assert!(due(secs(46), DEFAULT_CHECKPOINTS).is_empty());
        assert!(due(secs(45) + 1, DEFAULT_CHECKPOINTS).is_empty());
    }

    #[test]
    fn the_stop_claiming_checkpoint_fires_at_45_minutes_remaining() {
        let d = due(secs(45), DEFAULT_CHECKPOINTS);
        assert_eq!(d.len(), 1, "exactly on the boundary it fires");
        assert_eq!(d[0].kind, NudgeKind::StopClaiming);
    }

    #[test]
    fn both_fire_once_the_task_is_within_five_minutes() {
        let d = due(secs(4), DEFAULT_CHECKPOINTS);
        assert_eq!(d.len(), 2);
        let kinds: Vec<NudgeKind> = d.iter().map(|n| n.kind).collect();
        assert!(kinds.contains(&NudgeKind::StopClaiming));
        assert!(kinds.contains(&NudgeKind::MergeOrReportBlocked));
    }

    #[test]
    fn a_crossed_checkpoint_fires_exactly_once_per_call() {
        // The property that stops a slow-polling harness from being told the same
        // thing repeatedly.
        for r in [secs(4), 0, secs(100), secs(45)] {
            let d = due(r, DEFAULT_CHECKPOINTS);
            let mut kinds: Vec<NudgeKind> = d.iter().map(|n| n.kind).collect();
            let before = kinds.len();
            kinds.sort();
            kinds.dedup();
            assert_eq!(kinds.len(), before, "a checkpoint fired twice at {r}s");
        }
    }

    #[test]
    fn the_most_urgent_checkpoint_reads_last() {
        // A worker that reads only the final line should read the one that
        // matters most.
        let d = due(0, DEFAULT_CHECKPOINTS);
        assert_eq!(
            d.last().map(|n| n.kind),
            Some(NudgeKind::MergeOrReportBlocked)
        );
    }

    #[test]
    fn urgency_ordering_does_not_depend_on_declaration_order() {
        // Declared "wrong" on purpose: the schedule is data, the ordering is not.
        let sched = &[
            Checkpoint {
                at_secs_remaining: 60,
                kind: NudgeKind::MergeOrReportBlocked,
            },
            Checkpoint {
                at_secs_remaining: 600,
                kind: NudgeKind::StopClaiming,
            },
        ];
        let d = due(0, sched);
        assert_eq!(d[0].kind, NudgeKind::StopClaiming);
        assert_eq!(d[1].kind, NudgeKind::MergeOrReportBlocked);
    }

    #[test]
    fn each_fired_nudge_records_the_remaining_time_it_saw() {
        let d = due(97, DEFAULT_CHECKPOINTS);
        assert_eq!(d[0].remaining_secs, 97);
    }

    #[test]
    fn a_zero_length_task_still_fires_everything() {
        assert_eq!(due(0, DEFAULT_CHECKPOINTS).len(), 2);
    }

    #[test]
    fn an_empty_schedule_fires_nothing() {
        assert!(due(0, &[]).is_empty());
    }

    #[test]
    fn a_custom_schedule_is_honoured() {
        let sched = &[Checkpoint {
            at_secs_remaining: 10,
            kind: NudgeKind::WrapUp,
        }];
        assert!(due(11, sched).is_empty());
        assert_eq!(due(10, sched).len(), 1);
        assert_eq!(due(0, sched)[0].kind, NudgeKind::WrapUp);
    }

    #[test]
    fn the_default_schedule_descends_in_remaining_time() {
        for pair in DEFAULT_CHECKPOINTS.windows(2) {
            assert!(
                pair[0].at_secs_remaining > pair[1].at_secs_remaining,
                "checkpoints should descend in remaining time"
            );
        }
    }

    #[test]
    fn every_kind_says_something_a_worker_can_act_on() {
        for k in NudgeKind::ALL {
            let s = k.instruction();
            assert!(s.len() > 15, "{k:?} is too terse to act on: {s:?}");
            assert!(s.ends_with('.'), "{k:?}: {s:?}");
        }
    }

    #[test]
    fn kinds_round_trip_through_their_wire_spelling() {
        for k in NudgeKind::ALL {
            assert_eq!(NudgeKind::parse(k.as_str()), Some(*k));
            assert_eq!(NudgeKind::parse(&k.as_str().to_uppercase()), Some(*k));
        }
        assert_eq!(NudgeKind::parse("nonsense"), None);
    }

    #[test]
    fn the_board_text_names_the_worker_and_the_instruction() {
        let n = Nudge {
            kind: NudgeKind::StopClaiming,
            remaining_secs: 40 * 60,
            cursor: 0,
        };
        let t = board_text(&n, "w3");
        assert!(t.contains("w3"), "{t}");
        assert!(t.contains(NudgeKind::StopClaiming.instruction()), "{t}");
    }

    #[test]
    fn a_nudge_round_trips_through_json() {
        let n = Nudge {
            kind: NudgeKind::MergeOrReportBlocked,
            remaining_secs: 12,
            cursor: 7,
        };
        let s = serde_json::to_string(&n).unwrap();
        assert_eq!(serde_json::from_str::<Nudge>(&s).unwrap(), n);
    }
}
