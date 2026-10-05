//! G4: verify-before-merge. Evidence for a patch is a **reference to a recorded
//! gate run**, not pasted text.
//!
//! ## The problem with pasted evidence
//!
//! The obvious design for "attach the test output to your patch summary" is a
//! free-text field. It does not work, and the reason is worth stating because it
//! is the whole justification for this module:
//!
//! - A human cannot tell pasted output from a plausible-looking fabrication.
//! - Output pasted *before* the last edit is stale, and nothing marks it as such.
//! - Output from a *different* change is worse than none, because it reads as
//!   evidence.
//!
//! None of those are fixed by asking people to be careful. They are fixed by
//! making evidence a **reference to a run the system recorded**, so that
//! existence, verdict and freshness are all checkable facts rather than claims.
//!
//! ## What is checked
//!
//! | check | why |
//! |---|---|
//! | the run exists | fabricated ids are the trivial forgery; refusing unknown ids catches it |
//! | the run passed | a recorded *failure* is evidence too, just not evidence of correctness |
//! | the run is newer than the worker's last change | otherwise it vouches for a patch that no longer exists |
//! | the files are named and the idea is stated | an empty patch summary is not reviewable |
//!
//! ## Fail-closed on an unknowable
//!
//! Freshness is established from the board: the run's cursor must be newer than
//! the cursor of the newest non-summary entry that worker wrote. If that entry
//! has aged out of the bounded board, freshness **cannot be established**, and
//! this returns [`EvidenceError::UnknownBase`] rather than assuming it was fine.
//! A merge gate that degrades to "probably still valid" is a merge gate that
//! eventually lets an unverified change through.
//!
//! ## What this does not do
//!
//! It does not decide *which* gates matter, and it does not run them. It records
//! runs of gates that already exist — `verify-invariants` (H1), the golden release
//! manifest (S24), a test log — and checks that a patch summary points at one
//! that is recent and green. Running the gates belongs to whoever invokes the
//! record op; grading the merge belongs to the human approval lane.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::board::{Board, BoardEntry, BoardKind};

/// How a gate run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    /// The gate could not decide — missing toolchain, skipped, aborted. Treated
    /// as *not* passing: "I could not check" and "it is fine" are different
    /// sentences, and conflating them is how an unverified change ships.
    Unknown,
}

impl Verdict {
    pub fn is_pass(self) -> bool {
        self == Verdict::Pass
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Unknown => "unknown",
        }
    }
}

/// One recorded run of a gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateRun {
    /// Assigned on record; this is what a summary references.
    pub id: u64,
    /// Which gate: `verify-invariants`, `release-golden`, a test suite name, …
    pub source: String,
    pub verdict: Verdict,
    /// Board cursor at the moment the run was recorded. Freshness is judged
    /// against this, which is why it is the recording system's cursor and not a
    /// timestamp — see [`Verdict`] on replayability.
    pub cursor: u64,
    /// Digest of the gate's own output. Lets a reader confirm which artefact is
    /// being cited without the text being pasted into the board.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    /// One line of machine-extracted summary, for a human scanning the history.
    /// Never the basis of a decision — see the module comment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// The evidence a patch summary carries: a *reference*, plus what changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchSummary {
    /// The board entry's own cursor, filled in when it is written.
    #[serde(default)]
    pub cursor: u64,
    pub worker: String,
    /// Files the patch touches. Must be non-empty: a merge that changes nothing
    /// is not reviewable and is not a merge.
    pub files: Vec<String>,
    /// What the change does, in the author's words. This one genuinely is prose
    /// and is *not* verified — the point of the module is that only the evidence
    /// is checked, so it is clear which half is which.
    pub idea: String,
    /// The gate run this patch is vouched for by.
    pub run_id: u64,
}

/// Registry of recorded gate runs.
#[derive(Debug, Clone, Default)]
pub struct GateRuns {
    runs: BTreeMap<u64, GateRun>,
    next_id: u64,
}

/// Retained runs. Like the board, this is a bounded window: a run nobody cites
/// within a few hundred others is not evidence anyone will reach for.
pub const RUN_HISTORY: usize = 256;

impl GateRuns {
    pub fn new() -> GateRuns {
        GateRuns {
            runs: BTreeMap::new(),
            // Ids start at 1, matching the board cursor convention.
            next_id: 1,
        }
    }

    /// Record a gate run, assigning it an id.
    ///
    /// `cursor` is supplied by the caller because it must come from the same
    /// counter the board uses; taking it here would allow a caller to backdate
    /// evidence and defeat the freshness check.
    pub fn record(
        &mut self,
        source: &str,
        verdict: Verdict,
        cursor: u64,
        digest: Option<String>,
        summary: Option<String>,
    ) -> GateRun {
        let id = self.next_id;
        self.next_id += 1;
        let run = GateRun {
            id,
            source: source.to_string(),
            verdict,
            cursor,
            digest,
            summary,
        };
        self.runs.insert(id, run.clone());
        while self.runs.len() > RUN_HISTORY {
            let oldest = *self.runs.keys().next().expect("non-empty");
            self.runs.remove(&oldest);
        }
        run
    }

    pub fn get(&self, id: u64) -> Option<&GateRun> {
        self.runs.get(&id)
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// Why a patch summary was refused. Every variant is a *fact* about the recorded
/// state, never a judgement about the author.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "snake_case")]
pub enum EvidenceError {
    /// No `run_id` on the summary.
    Missing,
    /// No such run was ever recorded.
    UnknownRun { run_id: u64 },
    /// The run exists and is not green.
    NotPassing {
        run_id: u64,
        source: String,
        verdict: &'static str,
    },
    /// The run predates the worker's last change, so it vouches for a patch that
    /// no longer exists.
    Stale {
        run_id: u64,
        run_cursor: u64,
        last_change: u64,
    },
    /// The worker's last change is not on the board any more, so freshness
    /// cannot be established. Fails closed.
    UnknownBase { worker: String },
    /// The summary names no files.
    NoFiles,
    /// The summary states no idea.
    NoIdea,
}

impl EvidenceError {
    /// A sentence an approving human can act on.
    pub fn explain(&self) -> String {
        match self {
            EvidenceError::Missing => {
                "the patch summary cites no gate run; run a gate and record it.".to_string()
            }
            EvidenceError::UnknownRun { run_id } => format!(
                "gate run {run_id} was never recorded; a summary must cite a run this \
                 system produced, not pasted output."
            ),
            EvidenceError::NotPassing {
                run_id,
                source,
                verdict,
            } => format!("gate run {run_id} ({source}) is `{verdict}`, not `pass`."),
            EvidenceError::Stale {
                run_id,
                run_cursor,
                last_change,
            } => format!(
                "gate run {run_id} ran at cursor {run_cursor}, before the last change at \
                 cursor {last_change}; re-run the gate on the current patch."
            ),
            EvidenceError::UnknownBase { worker } => format!(
                "cannot establish freshness: {worker}'s last change is no longer on the \
                 board, so there is nothing to compare the gate run against."
            ),
            EvidenceError::NoFiles => "the patch summary names no files, so there is nothing to review.".to_string(),
            EvidenceError::NoIdea => "the patch summary states no idea, so the change is unexplained.".to_string(),
        }
    }
}

/// Validate a summary against the recorded runs and the board.
///
/// `summary.cursor` is the cursor of the PATCH_SUMMARY entry itself, which the
/// caller fills in *after* writing the entry. That ordering is what makes the
/// staleness comparison well-founded: everything the worker wrote before the
/// summary has a cursor below it.
pub fn validate(
    summary: &PatchSummary,
    runs: &GateRuns,
    board: &Board,
) -> Result<GateRun, EvidenceError> {
    if summary.files.is_empty() {
        return Err(EvidenceError::NoFiles);
    }
    if summary.idea.trim().is_empty() {
        return Err(EvidenceError::NoIdea);
    }
    let run = runs
        .get(summary.run_id)
        .ok_or(EvidenceError::UnknownRun {
            run_id: summary.run_id,
        })?;
    if !run.verdict.is_pass() {
        return Err(EvidenceError::NotPassing {
            run_id: run.id,
            source: run.source.clone(),
            verdict: run.verdict.as_str(),
        });
    }
    let last_change =
        board
            .last_change_cursor(&summary.worker)
            .ok_or_else(|| EvidenceError::UnknownBase {
                worker: summary.worker.clone(),
            })?;
    if run.cursor <= last_change {
        return Err(EvidenceError::Stale {
            run_id: run.id,
            run_cursor: run.cursor,
            last_change,
        });
    }
    Ok(run.clone())
}

/// Write a `PATCH_SUMMARY` entry for `worker` and validate it.
///
/// The board entry is written **either way**, because a rejected summary is part
/// of the history: a reader later must be able to see that the merge was refused
/// and why, rather than seeing nothing at all.
///
/// `worker` is an explicit parameter rather than inferred from the board's last
/// entry: inferring it would attribute a summary to whoever happened to write
/// most recently, which is wrong the moment two workers are interleaved — and the
/// staleness check below is only meaningful if the worker is the one who actually
/// did the work.
pub fn submit(
    board: &mut Board,
    runs: &GateRuns,
    worker: &str,
    files: &[String],
    idea: &str,
    run_id: u64,
) -> (BoardEntry, Result<GateRun, EvidenceError>) {
    let declared = PatchSummary {
        cursor: 0,
        worker: worker.to_string(),
        files: files.to_vec(),
        idea: idea.to_string(),
        run_id,
    };
    let text = format!(
        "patch by {worker}: {} file(s): {}",
        files.len(),
        files.join(", ")
    );
    let detail = serde_json::to_string(&declared).unwrap_or_default();
    let entry = board.write(BoardKind::PatchSummary, worker, &text, Some(&detail));
    let mut checked = declared;
    checked.cursor = entry.cursor;
    let verdict = validate(&checked, runs, board);
    (entry, verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worker the tests speak as; `submit` takes it explicitly.
    const WORKER: &str = "w1";

    fn board() -> Board {
        Board::new()
    }

    /// A worker who has made a change at cursor 2 (cursor 1 is the board's own
    /// bookkeeping in these tests; real cursors start at the first write).
    fn worker_with_change(b: &mut Board, worker: &str) -> u64 {
        let e = b.write(BoardKind::Fact, worker, "changed something", None);
        e.cursor
    }

    fn pass_run(runs: &mut GateRuns, cursor: u64) -> u64 {
        runs.record("verify-invariants", Verdict::Pass, cursor, Some("sha256:abc".into()), Some("34 passed".into()))
            .id
    }

    // ---- recording ---------------------------------------------------------

    #[test]
    fn a_recorded_run_gets_an_id_and_can_be_fetched_back() {
        let mut runs = GateRuns::new();
        let r = pass_run(&mut runs, 5);
        assert_eq!(r, 1);
        let got = runs.get(1).expect("recorded");
        assert_eq!(got.source, "verify-invariants");
        assert_eq!(got.cursor, 5);
        assert_eq!(got.digest.as_deref(), Some("sha256:abc"));
        assert!(got.verdict.is_pass());
    }

    #[test]
    fn ids_are_unique_and_monotonic() {
        let mut runs = GateRuns::new();
        for c in 1..=5 {
            assert_eq!(pass_run(&mut runs, c), c as u64);
        }
        assert_eq!(runs.len(), 5);
    }

    #[test]
    fn an_unknown_id_is_simply_absent() {
        let runs = GateRuns::new();
        assert!(runs.get(1).is_none());
        assert!(runs.is_empty());
    }

    #[test]
    fn run_history_is_bounded_oldest_first() {
        let mut runs = GateRuns::new();
        for c in 1..=(RUN_HISTORY as u64 + 5) {
            pass_run(&mut runs, c);
        }
        assert_eq!(runs.len(), RUN_HISTORY);
        // The newest are kept; the oldest were dropped.
        assert!(runs.get(1).is_none());
        assert!(runs.get(RUN_HISTORY as u64 + 5).is_some());
    }

    // ---- verdicts ----------------------------------------------------------

    #[test]
    fn only_pass_counts_as_evidence() {
        assert!(Verdict::Pass.is_pass());
        assert!(!Verdict::Fail.is_pass());
        // "I could not check" is not "it is fine".
        assert!(!Verdict::Unknown.is_pass());
    }

    #[test]
    fn a_failing_gate_run_is_still_evidence_just_not_of_correctness() {
        // It must be *recorded*, so the refusal can name it.
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = runs
            .record("release-golden", Verdict::Fail, 10, None, Some("manifest differs".into()))
            .id;
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", id);
        assert_eq!(
            v.unwrap_err(),
            EvidenceError::NotPassing {
                run_id: id,
                source: "release-golden".into(),
                verdict: "fail",
            }
        );
    }

    #[test]
    fn an_unknown_verdict_is_refused_like_a_failure() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = runs
            .record("verify-invariants", Verdict::Unknown, 10, None, None)
            .id;
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", id);
        assert!(matches!(v.unwrap_err(), EvidenceError::NotPassing { .. }));
    }

    // ---- the forgery cases -------------------------------------------------

    #[test]
    fn a_hand_written_summary_citing_nothing_is_refused() {
        let (mut b, runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", 0);
        assert_eq!(v.unwrap_err(), EvidenceError::UnknownRun { run_id: 0 });
    }

    #[test]
    fn a_cited_run_that_was_never_recorded_is_refused() {
        // The trivial forgery: an id that looks fine but was never issued.
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let real = pass_run(&mut runs, 10);
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", real + 99);
        assert_eq!(
            v.unwrap_err(),
            EvidenceError::UnknownRun { run_id: real + 99 }
        );
    }

    #[test]
    fn a_summary_naming_no_files_is_refused() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, 10);
        let (_, v) = submit(&mut b, &runs, WORKER, &[], "fix", id);
        assert_eq!(v.unwrap_err(), EvidenceError::NoFiles);
    }

    #[test]
    fn a_summary_stating_no_idea_is_refused() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, 10);
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "   ", id);
        assert_eq!(v.unwrap_err(), EvidenceError::NoIdea);
    }

    // ---- staleness, the part that matters ---------------------------------

    #[test]
    fn a_gate_run_after_the_change_is_accepted() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, b.latest_cursor() + 1);
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix the thing", id);
        assert!(v.is_ok(), "{:?}", v.unwrap_err());
    }

    #[test]
    fn a_gate_run_from_before_the_change_is_stale_and_refused() {
        // The common real failure: run the tests, then fix one more thing.
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, 1); // ran before w1's change
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", id);
        match v.unwrap_err() {
            EvidenceError::Stale { run_id, last_change, .. } => {
                assert_eq!(run_id, id);
                assert_eq!(last_change, 1, "the change is the board's first entry");
            }
            other => panic!("expected stale, got {other:?}"),
        }
    }

    #[test]
    fn re_running_the_gate_after_the_change_makes_the_same_summary_acceptable() {
        // The remediation path has to actually work, or the check is a dead end.
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let stale = pass_run(&mut runs, 1);
        assert!(submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", stale)
            .1
            .is_err());

        worker_with_change(&mut b, "w1");
        let fresh = pass_run(&mut runs, b.latest_cursor() + 1);
        assert!(submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", fresh)
            .1
            .is_ok());
    }

    #[test]
    fn one_workers_stale_run_does_not_invalidate_anothers_fresh_one() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let w1_stale = pass_run(&mut runs, 1);
        worker_with_change(&mut b, "w2");
        let w2_fresh = pass_run(&mut runs, b.latest_cursor() + 1);
        assert!(submit(&mut b, &runs, "w2", &["a.rs".into()], "w2 work", w2_fresh).1.is_ok());
        assert!(submit(&mut b, &runs, "w1", &["b.rs".into()], "w1 work", w1_stale).1.is_err());
    }

    #[test]
    fn freshness_fails_closed_when_the_base_is_gone() {
        // If the worker's last change has aged out, there is nothing to compare
        // against. Assuming it was fine is how an unverified change ships.
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "ghost");
        let id = pass_run(&mut runs, b.latest_cursor() + 1);
        // Overflow the board past capacity so the ghost's change is dropped.
        for i in 0..(crate::board::CAPACITY + 5) {
            b.write(BoardKind::Observed, "filler", &format!("e{i}"), None);
        }
        let (_, v) = submit(&mut b, &runs, "ghost", &["a.rs".into()], "fix", id);
        assert_eq!(v.unwrap_err(), EvidenceError::UnknownBase { worker: "ghost".into() });
    }

    #[test]
    fn a_refused_summary_is_still_recorded_on_the_board() {
        // A reader later must see that the merge was refused, not see nothing.
        let (mut b, runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let (entry, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", 42);
        assert!(v.is_err());
        assert_eq!(entry.kind, BoardKind::PatchSummary);
        assert_eq!(b.len(), 2);
        assert_eq!(b.all()[1].kind, BoardKind::PatchSummary);
    }

    #[test]
    fn a_accepted_summary_names_its_files_and_idea_in_the_entry() {
        let (mut b, mut runs) = (board(), GateRuns::new());
        worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, b.latest_cursor() + 1);
        let files = vec!["unfer_protocol/src/board.rs".to_string()];
        let (entry, v) = submit(&mut b, &runs, WORKER, &files, "add the board", id);
        assert!(v.is_ok());
        assert!(entry.text.contains("board.rs"), "{}", entry.text);
        let detail: PatchSummary =
            serde_json::from_str(entry.detail.as_deref().expect("detail")).expect("round-trips");
        assert_eq!(detail.idea, "add the board");
        assert_eq!(detail.files, files);
        assert_eq!(detail.run_id, id);
    }

    #[test]
    fn every_refusal_explains_itself_in_a_sentence() {
        for e in [
            EvidenceError::Missing,
            EvidenceError::UnknownRun { run_id: 3 },
            EvidenceError::NotPassing { run_id: 1, source: "x".into(), verdict: "fail" },
            EvidenceError::Stale { run_id: 1, run_cursor: 2, last_change: 9 },
            EvidenceError::UnknownBase { worker: "w".into() },
            EvidenceError::NoFiles,
            EvidenceError::NoIdea,
        ] {
            let s = e.explain();
            assert!(s.len() > 20, "unhelpful: {s:?}");
            assert!(s.ends_with('.') || s.contains(':'), "not a sentence: {s:?}");
        }
    }

    // ---- board support -----------------------------------------------------

    #[test]
    fn last_change_cursor_ignores_a_workers_own_summaries() {
        // Otherwise a summary would be its own "last change" and every run would
        // look stale.
        let mut b = board();
        b.write(BoardKind::Fact, "w1", "change", None);
        let change = b.write(BoardKind::Fact, "w1", "change again", None).cursor;
        b.write(BoardKind::PatchSummary, "w1", "summary", None);
        assert_eq!(b.last_change_cursor("w1"), Some(change));
    }

    #[test]
    fn last_change_cursor_is_per_worker() {
        let mut b = board();
        let w1 = b.write(BoardKind::Fact, "w1", "a", None).cursor;
        b.write(BoardKind::Fact, "w2", "b", None);
        assert_eq!(b.last_change_cursor("w1"), Some(w1));
        assert_eq!(b.last_change_cursor("w2"), Some(w1 + 1));
        assert_eq!(b.last_change_cursor("nobody"), None);
    }

    #[test]
    fn a_gate_run_at_the_exact_change_cursor_is_stale() {
        // Not "newer than" but "strictly after". A run recorded before the
        // change landed cannot have seen it.
        let (mut b, mut runs) = (board(), GateRuns::new());
        let change = worker_with_change(&mut b, "w1");
        let id = pass_run(&mut runs, change);
        let (_, v) = submit(&mut b, &runs, WORKER, &["a.rs".into()], "fix", id);
        assert!(matches!(v.unwrap_err(), EvidenceError::Stale { .. }));
    }
}