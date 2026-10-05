//! G3: claims with overlap resolution, and direct messages between workers.
//!
//! ## The problem this solves
//!
//! Two workers that both start the same task produce duplicated work and — worse
//! — conflicting patches to the same file, which is the failure mode multi-agent
//! work introduces that single-agent work does not have. The mechanism here is
//! deliberately the cheapest one that works: **a claim is a labelled scope on an
//! append-only board, and a claim whose scope overlaps an existing live claim is
//! reported as an overlap to both parties** rather than being silently allowed or
//! silently refused.
//!
//! ## Why overlap is reported and not prevented
//!
//! A board cannot prevent anything: it is a log, not a lock. Anything that *did*
// prevent it would need mutual exclusion over a distributed log, which is a
//! consensus problem, and this project already has `unfer_consensus` for that.
//! What this layer does is make the collision **visible and attributable** at the
//! moment it happens, so the workers negotiate ([`Dm`]) and a human sees it in the
//! audit trail. Reporting is honest; pretending to arbitrate would not be.
//!
//! ## Scope matching
//!
//! Overlap is decided by three rules, tried in order, because "overlapping" means
//! different things for different kinds of work:
//!
//! 1. **Identical scope** — same normalised string. The common case.
//! 2. **Path prefix** — one scope is a directory-prefix of the other, so
//!    `unfer/unfer_ffi/src` overlaps `unfer/unfer_ffi/src/handles.rs`. Uses
//!    segment boundaries, so `src/foo` does *not* overlap `src/foobar`.
//! 3. **Glob** — either side may contain `*`, matched with a simple
//!    segment-aware wildcard. Glob rather than full regex on purpose: every regex
//!    engine is a denial-of-service surface on unauthenticated input, and a claim
//!    scope is exactly that kind of input.
//!
//! Claims expire on an explicit [`Board::expire_claims`] call rather than on a
//! wall clock, for the same reason the rest of this project avoids wall-clock
//! time in state machines: a ledger that depends on `now` is not replayable. The
//! caller passes the current cursor, which is monotonic and already in hand.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::board::{Board, BoardEntry, BoardKind};

/// The default number of direct messages retained per recipient.
pub const DM_CAPACITY: usize = 64;

/// What a claim covers.
///
/// Free-form on purpose — a claim may be a file path, a module name, a Lean
/// chapter, or a task id, and forcing one vocabulary would make half of them
/// unclaimable. Overlap is therefore decided structurally (below) rather than by
/// requiring the writer to use a marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimScope {
    /// The scope string, normalised on construction.
    pub scope: String,
}

impl ClaimScope {
    pub fn new(scope: &str) -> ClaimScope {
        ClaimScope {
            scope: normalize(scope),
        }
    }

    /// Whether this scope and `other` cover any of the same work.
    pub fn overlaps(&self, other: &ClaimScope) -> bool {
        let (a, b) = (self.scope.as_str(), other.scope.as_str());
        if a.is_empty() || b.is_empty() {
            return false;
        }
        if has_glob(a) || has_glob(b) {
            return glob_match(a, b) || glob_match(b, a);
        }
        if a == b {
            return true;
        }
        prefix_overlap(a, b) || prefix_overlap(b, a)
    }
}

/// Trim, and normalise separators so `a\b` and `a/b` are one scope.
fn normalize(s: &str) -> String {
    s.trim().replace('\\', "/").trim_matches('/').to_string()
}

fn has_glob(s: &str) -> bool {
    s.contains('*')
}

/// True when `prefix` covers `path`, on segment boundaries.
///
/// The boundary check is the whole point: `src/foo` must not overlap
/// `src/foobar`, or a claim on one file would block every file whose name starts
/// with the same letters — a bug that would present as "my claim collided with
/// an unrelated file".
fn prefix_overlap(prefix: &str, path: &str) -> bool {
    if prefix.is_empty() || path.is_empty() || prefix == path {
        return false;
    }
    if prefix.ends_with('/') {
        return path.starts_with(prefix);
    }
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// Segment-aware glob: `*` matches within one segment, `**` across segments.
///
/// Hand-rolled rather than pulled from a crate because the semantics matter
/// more than the matching does, and a regex here is an unauthenticated-input DoS
/// surface (see the module comment).
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<&str> = pattern.split('/').collect();
    let t: Vec<&str> = text.split('/').collect();
    seg_match(&p, &t)
}

fn seg_match(p: &[&str], t: &[&str]) -> bool {
    if p.is_empty() {
        return t.is_empty();
    }
    if p[0] == "**" {
        // `**` matches zero or more segments.
        for skip in 0..=t.len() {
            if seg_match(&p[1..], &t[skip..]) {
                return true;
            }
        }
        return false;
    }
    if t.is_empty() {
        return false;
    }
    if !one_seg_match(p[0], t[0]) {
        return false;
    }
    seg_match(&p[1..], &t[1..])
}

/// `*` within a single segment, `?` for one character.
fn one_seg_match(pat: &str, text: &str) -> bool {
    let pc: Vec<char> = pat.chars().collect();
    let tc: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < tc.len() {
        if pi < pc.len() && (pc[pi] == '?' || pc[pi] == tc[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pc.len() && pc[pi] == '*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < pc.len() && pc[pi] == '*' {
        pi += 1;
    }
    pi == pc.len()
}

/// A live claim: who holds which scope, as of which cursor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveClaim {
    pub worker: String,
    pub scope: String,
    /// Cursor of the `CLAIM` board entry this came from. Used for expiry
    /// bookkeeping and for attributing an overlap in a report.
    pub cursor: u64,
}

/// A direct message between workers.
///
/// Priority is a small integer, not an enum: the interesting cases are "normal"
/// and "I am blocked, answer now", and a total order over an open scale is
/// something the reader invents anyway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dm {
    pub from: String,
    pub to: String,
    pub text: String,
    /// Higher sorts first.
    pub priority: i64,
    pub cursor: u64,
}

/// What happened when a claim was attempted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ClaimOutcome {
    /// The scope was free. The claim is live.
    Granted { claim: LiveClaim },
    /// The scope was already claimed by someone else. Both parties are told.
    ///
    /// `existing` is reported rather than refused outright because a duplicate
    /// claim is often benign — the same worker re-claiming after a restart, or two
    /// workers converging on the same obviously-correct owner. The information is
    /// what lets them settle it ([`Dm`]).
    Overlaps { existing: Vec<LiveClaim> },
}

/// Result of attempting a claim.
pub struct ClaimAttempt {
    /// The board entry written for this attempt, when one was written. An
    /// overlapping claim still writes a `CLAIM` entry: the collision is part of
    /// the history and should not be invisible to a third worker reading later.
    pub entry: BoardEntry,
    pub outcome: ClaimOutcome,
}

/// The cooperation state layered over a [`Board`]: live claims and per-recipient
/// message queues.
#[derive(Debug, Clone, Default)]
pub struct Coop {
    claims: Vec<LiveClaim>,
    dms: std::collections::BTreeMap<String, VecDeque<Dm>>,
    /// Messages lost to queue overflow, per recipient.
    ///
    /// A counter rather than `CAPACITY - len()`: the difference between unused
    /// capacity and messages actually dropped is the difference between "nothing
    /// has gone wrong yet" and "five messages went missing", and reporting the
    /// former as the latter would make the number meaningless.
    dm_dropped: std::collections::BTreeMap<String, u64>,
}

impl Coop {
    pub fn new() -> Coop {
        Coop::default()
    }

    /// Every live claim, in the order claimed.
    pub fn claims(&self) -> &[LiveClaim] {
        &self.claims
    }

    /// Attempt to claim `scope` for `worker`.
    pub fn claim(&mut self, board: &mut Board, worker: &str, scope: &str) -> ClaimAttempt {
        let cs = ClaimScope::new(scope);
        let overlapping: Vec<LiveClaim> = self
            .claims
            .iter()
            .filter(|c| c.worker != worker && ClaimScope::new(&c.scope).overlaps(&cs))
            .cloned()
            .collect();

        let text = format!("{} claims {}", worker, cs.scope);
        let entry = board.write(BoardKind::Claim, worker, &text, Some(&cs.scope));

        if overlapping.is_empty() {
            let claim = LiveClaim {
                worker: worker.to_string(),
                scope: cs.scope,
                cursor: entry.cursor,
            };
            self.claims.push(claim.clone());
            ClaimAttempt {
                entry,
                outcome: ClaimOutcome::Granted { claim },
            }
        } else {
            // Still recorded as a claim on the board, but not registered as live:
            // two workers must not both believe they own a scope.
            ClaimAttempt {
                entry,
                outcome: ClaimOutcome::Overlaps {
                    existing: overlapping,
                },
            }
        }
    }

    /// Drop claims made before `cursor`.
    ///
    /// Expiry is explicit and cursor-based rather than time-based: see the module
    /// comment on replayability. Returns the released claims so the caller can
    /// tell the workers involved.
    pub fn expire_claims(&mut self, cursor: u64) -> Vec<LiveClaim> {
        let mut kept = Vec::with_capacity(self.claims.len());
        let mut released = Vec::new();
        for c in self.claims.drain(..) {
            if c.cursor < cursor {
                released.push(c);
            } else {
                kept.push(c);
            }
        }
        self.claims = kept;
        released
    }

    /// Who currently holds a scope overlapping `scope`, excluding `worker`.
    pub fn conflicting(&self, worker: &str, scope: &str) -> Vec<LiveClaim> {
        let cs = ClaimScope::new(scope);
        self.claims
            .iter()
            .filter(|c| c.worker != worker && ClaimScope::new(&c.scope).overlaps(&cs))
            .cloned()
            .collect()
    }

    /// Send a direct message.
    ///
    /// The board entry is a `BoardKind::Observed` rather than a private store so
    /// that a reader auditing the board can see that a negotiation happened. The
    /// message text itself goes only to the recipient's queue.
    pub fn dm(&mut self, board: &mut Board, from: &str, to: &str, text: &str, priority: i64) -> Dm {
        let cursor = board.latest_cursor() + 1;
        let msg = Dm {
            from: from.to_string(),
            to: to.to_string(),
            // Redacted like any board text: a DM is free text from an agent, and
            // the same "somebody pasted a credential" failure applies.
            text: crate::board::redact_secrets(text),
            priority,
            cursor,
        };
        board.write(
            BoardKind::Observed,
            from,
            &format!("dm -> {}: {}", to, truncate_for_board(text)),
            Some(&format!("priority {priority}")),
        );
        let q = self.dms.entry(to.to_string()).or_default();
        q.push_back(msg.clone());
        while q.len() > DM_CAPACITY {
            q.pop_front();
            *self.dm_dropped.entry(to.to_string()).or_default() += 1;
        }
        msg
    }

    /// Messages for `to`, highest priority first, then oldest first within a
    /// priority. Does not consume: a worker polls mid-turn and must not lose a
    /// message to a crashed consumer (the C1 lesson).
    pub fn inbox(&self, to: &str) -> Vec<&Dm> {
        let Some(q) = self.dms.get(to) else {
            return Vec::new();
        };
        let mut v: Vec<&Dm> = q.iter().collect();
        v.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.cursor.cmp(&b.cursor)));
        v
    }

    /// Remove and return the messages for `to`. The consuming counterpart to
    /// [`Coop::inbox`], for a worker that acknowledges what it has read.
    pub fn take_inbox(&mut self, to: &str) -> Vec<Dm> {
        self.dms
            .remove(to)
            .map(|q| q.into_iter().collect())
            .unwrap_or_default()
    }

    /// How many messages were dropped from `to`'s queue for overflow. A non-zero
    /// value means the negotiation missed something and should be re-run from the
    /// board, which records every send.
    pub fn dm_dropped(&self, to: &str) -> u64 {
        self.dm_dropped.get(to).copied().unwrap_or(0)
    }
}

fn truncate_for_board(s: &str) -> String {
    const N: usize = 120;
    if s.chars().count() <= N {
        return s.to_string();
    }
    let mut out: String = s.chars().take(N - 1).collect();
    out.push('…');
    out
}

// ── role hand-off (G7) ────────────────────────────────────────────────────
//
// Kept here rather than in its own module because it is the third kind of
// cooperation record alongside claims and messages, and it shares their
// lifetime: a hand-off names a claim, and a claim can expire underneath it.
//
// A hand-off grants **no privilege**. `role` is a label recorded on the board so
// that "who reviewed this" is observable after the fact; authority still comes
// from the grant set (S21/S28) and nothing in this module can widen it. That is
// the property that makes it safe to add without a security review, and it is
// worth stating loudly because "role messages" is exactly the shape that usually
// smuggles privilege in.

/// A worker's role over a particular claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Writing the change.
    Implementer,
    /// Checking someone else's change.
    Reviewer,
    /// Merging reviewed changes.
    Integrator,
}

/// The two halves of a hand-off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub enum Handoff {
    /// "please take this role on this claim". Records what was *asked for*.
    RoleRequest {
        claim_cursor: u64,
        role: Role,
        by: String,
    },
    /// "I will take it". This is what records the role as **held**.
    ///
    /// It carries the role explicitly rather than assuming `Reviewer`, because a
    /// request may be for integration and assuming otherwise would record the
    /// wrong thing.
    RoleAccept {
        claim_cursor: u64,
        role: Role,
        by: String,
    },
}

impl Handoff {
    pub fn claim_cursor(&self) -> u64 {
        match self {
            Handoff::RoleRequest { claim_cursor, .. } => *claim_cursor,
            Handoff::RoleAccept { claim_cursor, .. } => *claim_cursor,
        }
    }

    /// The label to show in the history.
    pub fn summary(&self) -> String {
        match self {
            Handoff::RoleRequest {
                claim_cursor,
                role,
                by,
            } => format!("{by} requested {role:?} on claim {claim_cursor}"),
            Handoff::RoleAccept {
                claim_cursor,
                role,
                by,
            } => format!("{by} accepted {role:?} on claim {claim_cursor}"),
        }
    }
}

/// Record a hand-off on the board, and return the `OBSERVED` entry.
///
/// `history` is the caller's record; this function only writes the board entry so
/// the event is visible to anyone reading the board rather than only to the two
/// workers involved.
pub fn record_handoff(board: &mut Board, handoff: &Handoff) -> BoardEntry {
    board.write(
        BoardKind::Observed,
        match handoff {
            Handoff::RoleRequest { by, .. } | Handoff::RoleAccept { by, .. } => by,
        },
        &handoff.summary(),
        None,
    )
}

/// The role a worker holds over a claim, if any. Derived from a hand-off log, so
/// it is as trustworthy as that log — which is the point: it is a record, not a
/// capability.
///
/// **Only an accept confers the role.** A request records an intention; treating
/// it as a role would put an unreviewed change into a "reviewed" state because
/// somebody asked, which is the failure this distinction exists to prevent.
pub fn role_over(handoffs: &[Handoff], claim_cursor: u64, worker: &str) -> Option<Role> {
    handoffs
        .iter()
        .filter_map(|h| match h {
            Handoff::RoleAccept {
                claim_cursor: c,
                role,
                by,
            } if *c == claim_cursor && by == worker => Some(*role),
            _ => None,
        })
        .next_back()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, BoardKind, CAPACITY, GrepExpr};

    fn board() -> Board {
        Board::new()
    }

    fn granted(a: ClaimAttempt) -> LiveClaim {
        match a.outcome {
            ClaimOutcome::Granted { claim } => claim,
            other => panic!("expected granted, got {other:?}"),
        }
    }

    fn overlaps(a: ClaimAttempt) -> Vec<LiveClaim> {
        match a.outcome {
            ClaimOutcome::Overlaps { existing } => existing,
            other => panic!("expected overlaps, got {other:?}"),
        }
    }

    // ---- scope normalization ----------------------------------------------

    #[test]
    fn scopes_are_trimmed_and_separators_normalized() {
        assert_eq!(ClaimScope::new("  a/b  ").scope, "a/b");
        assert_eq!(ClaimScope::new("a\\b").scope, "a/b");
        assert_eq!(ClaimScope::new("/a/b/").scope, "a/b");
    }

    #[test]
    fn an_empty_scope_never_overlaps_anything() {
        // An empty claim must not collide with every other claim, which is what a
        // naive "everything starts with ''" rule would do.
        let e = ClaimScope::new("");
        assert!(!e.overlaps(&ClaimScope::new("a")));
        assert!(!e.overlaps(&ClaimScope::new("")));
        assert!(!ClaimScope::new("a").overlaps(&e));
    }

    // ---- the three overlap rules -------------------------------------------

    #[test]
    fn identical_scopes_overlap() {
        assert!(ClaimScope::new("a/b").overlaps(&ClaimScope::new("a/b")));
        assert!(ClaimScope::new("ChapterFoo").overlaps(&ClaimScope::new("ChapterFoo")));
    }

    #[test]
    fn a_directory_scope_overlaps_a_file_inside_it() {
        assert!(
            ClaimScope::new("unfer/unfer_ffi/src")
                .overlaps(&ClaimScope::new("unfer/unfer_ffi/src/handles.rs"))
        );
        assert!(
            ClaimScope::new("unfer/unfer_ffi/src/handles.rs")
                .overlaps(&ClaimScope::new("unfer/unfer_ffi/src"))
        );
    }

    #[test]
    fn prefix_overlap_respects_segment_boundaries() {
        // The regression this guards: `src/foo` must not block `src/foobar`, or a
        // claim on one file collides with every file sharing its first letters.
        assert!(!ClaimScope::new("src/foo").overlaps(&ClaimScope::new("src/foobar")));
        assert!(!ClaimScope::new("a/b").overlaps(&ClaimScope::new("a/bc")));
    }

    #[test]
    fn disjoint_scopes_do_not_overlap() {
        assert!(!ClaimScope::new("a/x").overlaps(&ClaimScope::new("b/x")));
        assert!(!ClaimScope::new("NS/mainstream").overlaps(&ClaimScope::new("QG/density")));
    }

    #[test]
    fn a_glob_scope_overlaps_what_it_matches() {
        assert!(ClaimScope::new("unfer/*.rs").overlaps(&ClaimScope::new("unfer/handles.rs")));
        assert!(ClaimScope::new("unfer/handles.rs").overlaps(&ClaimScope::new("unfer/*.rs")));
        assert!(
            ClaimScope::new("**/*.lean").overlaps(&ClaimScope::new("BookProof/ChapterFoo.lean"))
        );
        assert!(ClaimScope::new("docs/*").overlaps(&ClaimScope::new("docs/RUNBOOK.md")));
    }

    #[test]
    fn a_glob_that_matches_nothing_does_not_overlap() {
        assert!(!ClaimScope::new("docs/*.rs").overlaps(&ClaimScope::new("docs/RUNBOOK.md")));
        assert!(!ClaimScope::new("test/*").overlaps(&ClaimScope::new("unfer/handles.rs")));
    }

    #[test]
    fn a_single_star_does_not_cross_a_segment_boundary() {
        assert!(!ClaimScope::new("a/*").overlaps(&ClaimScope::new("a/b/c")));
        assert!(ClaimScope::new("a/**").overlaps(&ClaimScope::new("a/b/c")));
    }

    #[test]
    fn one_segment_glob_edge_cases() {
        // `one_seg_match` sees `/` as an ordinary character; splitting into
        // segments is `seg_match`'s job. These cases are about the single-segment
        // matcher only.
        assert!(one_seg_match("*", "anything"));
        assert!(one_seg_match("*.rs", "handles.rs"));
        assert!(
            one_seg_match("*.rs", "a/handles.rs"),
            "per-segment matcher, not path-aware"
        );
        assert!(one_seg_match("h?ndles.rs", "handles.rs"));
        assert!(!one_seg_match("h?ndles.rs", "handlesxrs"));
        assert!(one_seg_match("*", ""));
        assert!(one_seg_match("", ""));
        assert!(!one_seg_match("", "x"));
        assert!(one_seg_match("a*b*c", "axxbyyc"));
    }

    #[test]
    fn segment_matching_is_what_keeps_a_star_inside_its_directory() {
        // The path-aware property, at the layer that provides it.
        assert!(!seg_match(&["docs", "*.rs"], &["docs", "a", "b.rs"]));
        assert!(seg_match(&["docs", "*"], &["docs", "a"]));
        assert!(seg_match(
            &["**", "*.lean"],
            &["BookProof", "ChapterFoo.lean"]
        ));
    }

    // ---- claiming ----------------------------------------------------------

    #[test]
    fn a_free_scope_is_granted() {
        let (mut b, mut c) = (board(), Coop::new());
        let cl = granted(c.claim(&mut b, "w1", "unfer/handles.rs"));
        assert_eq!(cl.worker, "w1");
        assert_eq!(cl.scope, "unfer/handles.rs");
        assert_eq!(c.claims().len(), 1);
    }

    #[test]
    fn an_overlapping_claim_reports_the_existing_holder() {
        // The G3 acceptance path: two workers, one scope, collision surfaced.
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "unfer/unfer_ffi/src");
        let second = c.claim(&mut b, "w2", "unfer/unfer_ffi/src/handles.rs");
        let ex = overlaps(second);
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].worker, "w1");
        // And the loser's scope is NOT live — two workers must not both believe
        // they own it.
        assert_eq!(c.claims().len(), 1);
        assert_eq!(c.claims()[0].worker, "w1");
    }

    #[test]
    fn a_worker_may_reclaim_its_own_scope() {
        // Re-claiming after a restart is benign and must not self-collide.
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/b");
        let again = c.claim(&mut b, "w1", "a/b");
        let _ = granted(again);
        assert_eq!(c.claims().len(), 2);
    }

    #[test]
    fn disjoint_claims_do_not_collide() {
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "NS/mainstream");
        let second = c.claim(&mut b, "w2", "QG/density");
        let _ = granted(second);
        assert_eq!(c.claims().len(), 2);
    }

    #[test]
    fn three_way_collision_reports_every_holder() {
        // Three *disjoint* live claims, all covered by a fourth scope.
        //
        // The two obvious setups do not work: claiming the same scope three times
        // means the second and third are never granted (so there is only ever one
        // holder to report), and claiming three scopes that already overlap each
        // other is the same problem. Overlapping claims cannot all be live by
        // construction — which is the point of the feature.
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "src/one.rs");
        c.claim(&mut b, "w2", "src/two.rs");
        c.claim(&mut b, "w3", "src/three.rs");
        assert_eq!(c.claims().len(), 3, "disjoint claims are all granted");

        let ex = overlaps(c.claim(&mut b, "w4", "src/*.rs"));
        assert_eq!(ex.len(), 3);
        let mut ws: Vec<&str> = ex.iter().map(|c| c.worker.as_str()).collect();
        ws.sort_unstable();
        assert_eq!(ws, vec!["w1", "w2", "w3"]);
    }

    #[test]
    fn an_overlapping_claim_is_still_recorded_on_the_board() {
        // The collision is part of the history: a third worker reading later must
        // be able to see that two workers tried.
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/b");
        let a = c.claim(&mut b, "w2", "a/b");
        assert_eq!(a.entry.kind, BoardKind::Claim);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn conflicting_reports_the_current_holder_for_a_scope() {
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/b");
        assert_eq!(c.conflicting("w2", "a/b/c").len(), 1);
        assert_eq!(
            c.conflicting("w1", "a/b/c").len(),
            0,
            "self is not a conflict"
        );
        assert_eq!(c.conflicting("w2", "z/z").len(), 0);
    }

    // ---- expiry ------------------------------------------------------------

    #[test]
    fn expiry_releases_only_claims_older_than_the_cursor() {
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/1");
        c.claim(&mut b, "w2", "a/2");
        let second_cursor = b.latest_cursor();

        // Everything before the last claim goes.
        let released = c.expire_claims(second_cursor);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].scope, "a/1");
        assert_eq!(c.claims().len(), 1);
        assert_eq!(c.claims()[0].scope, "a/2");
    }

    #[test]
    fn an_expired_scope_can_be_claimed_again() {
        // Expiry that does not actually free the scope would strand the work.
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/b");
        c.expire_claims(b.latest_cursor() + 1);
        assert!(c.claims().is_empty());
        let cl = granted(c.claim(&mut b, "w2", "a/b"));
        assert_eq!(cl.worker, "w2");
    }

    #[test]
    fn expiring_nothing_is_harmless() {
        let (_b, mut c) = (board(), Coop::new());
        assert!(c.expire_claims(1000).is_empty());
    }

    // ---- direct messages ---------------------------------------------------

    #[test]
    fn a_message_reaches_only_its_recipient() {
        let (mut b, mut c) = (board(), Coop::new());
        c.dm(&mut b, "w1", "w2", "are you on a/b?", 0);
        assert_eq!(c.inbox("w2").len(), 1);
        assert_eq!(c.inbox("w1").len(), 0);
        assert_eq!(c.inbox("w3").len(), 0);
    }

    #[test]
    fn reading_the_inbox_does_not_consume_it() {
        // A worker polls mid-turn; a crashed consumer must not lose the message.
        let (mut b, mut c) = (board(), Coop::new());
        c.dm(&mut b, "w1", "w2", "hello", 0);
        assert_eq!(c.inbox("w2").len(), 1);
        assert_eq!(c.inbox("w2").len(), 1);
        let taken = c.take_inbox("w2");
        assert_eq!(taken.len(), 1);
        assert_eq!(c.inbox("w2").len(), 0);
    }

    #[test]
    fn taking_an_empty_inbox_is_empty_not_an_error() {
        let (_b, mut c) = (board(), Coop::new());
        assert!(c.take_inbox("nobody").is_empty());
        assert!(c.inbox("nobody").is_empty());
    }

    #[test]
    fn higher_priority_messages_come_first_and_ties_keep_arrival_order() {
        let (mut b, mut c) = (board(), Coop::new());
        c.dm(&mut b, "w1", "w2", "normal a", 0);
        c.dm(&mut b, "w1", "w2", "urgent", 10);
        c.dm(&mut b, "w1", "w2", "normal b", 0);
        let texts: Vec<&str> = c.inbox("w2").iter().map(|d| d.text.as_str()).collect();
        assert_eq!(texts, vec!["urgent", "normal a", "normal b"]);
    }

    #[test]
    fn a_message_queue_is_bounded_and_the_loss_is_visible() {
        let (mut b, mut c) = (board(), Coop::new());
        for i in 0..(DM_CAPACITY + 5) {
            c.dm(&mut b, "w1", "w2", &format!("m{i}"), 0);
        }
        assert_eq!(c.inbox("w2").len(), DM_CAPACITY);
        assert_eq!(c.dm_dropped("w2"), 5, "dropped messages must be countable");
    }

    #[test]
    fn dm_dropped_is_zero_for_a_healthy_queue() {
        let (mut b, mut c) = (board(), Coop::new());
        c.dm(&mut b, "w1", "w2", "hi", 0);
        assert_eq!(c.dm_dropped("w2"), 0);
        assert_eq!(c.dm_dropped("w3"), 0);
    }

    #[test]
    fn a_message_is_recorded_on_the_board_so_a_negotiation_is_auditable() {
        let (mut b, mut c) = (board(), Coop::new());
        c.dm(&mut b, "w1", "w2", "I am taking a/b", 5);
        let all = b.all();
        assert!(!all.is_empty());
        assert!(
            all.iter().any(|e| e.text.contains("dm -> w2")),
            "the board must show that a message was sent: {:?}",
            all.iter().map(|e| &e.text).collect::<Vec<_>>()
        );
        assert!(GrepExpr::parse("dm").matches(all[0]));
    }

    #[test]
    fn a_secret_in_a_message_is_redacted() {
        let (mut b, mut c) = (board(), Coop::new());
        let m = c.dm(&mut b, "w1", "w2", "my token=abc123 failed", 0);
        assert!(!m.text.contains("abc123"), "{}", m.text);
        assert!(c.inbox("w2")[0].text.contains("failed"));
    }

    // ---- role hand-off (G7) -----------------------------------------------

    #[test]
    fn a_hand_off_is_recorded_and_visible() {
        let mut b = board();
        let h = Handoff::RoleRequest {
            claim_cursor: 1,
            role: Role::Reviewer,
            by: "w1".to_string(),
        };
        record_handoff(&mut b, &h);
        assert_eq!(GrepExpr::parse("reviewer").matches_all(&b), 1);
        assert_eq!(b.all()[0].worker, "w1");
    }

    #[test]
    fn the_final_accepted_role_for_a_claim_is_what_counts() {
        let log = vec![
            Handoff::RoleRequest {
                claim_cursor: 1,
                role: Role::Reviewer,
                by: "w2".to_string(),
            },
            Handoff::RoleAccept {
                claim_cursor: 1,
                role: Role::Reviewer,
                by: "w2".to_string(),
            },
        ];
        assert_eq!(role_over(&log, 1, "w2"), Some(Role::Reviewer));
        assert_eq!(role_over(&log, 1, "w1"), None);
        assert_eq!(role_over(&log, 2, "w2"), None);
    }

    #[test]
    fn an_unanswered_request_does_not_make_the_requester_a_reviewer() {
        // The accept is what records the role. Without it, "w2 asked to review" is
        // not "w2 is reviewing", and treating it as such would put an unreviewed
        // change in a reviewed state.
        let log = vec![Handoff::RoleRequest {
            claim_cursor: 1,
            role: Role::Reviewer,
            by: "w2".to_string(),
        }];
        assert_eq!(role_over(&log, 1, "w2"), None);
    }

    #[test]
    fn a_hand_off_carries_no_privilege() {
        // Structural: a Handoff has no field that could name a grant, so there is
        // nothing for it to widen. Stated as a test because that is the property
        // that makes it safe without a security review.
        let json = serde_json::to_string(&Handoff::RoleRequest {
            claim_cursor: 1,
            role: Role::Integrator,
            by: "w1".to_string(),
        })
        .unwrap();
        for forbidden in ["grant", "Grant", "token", "capability", "admin"] {
            assert!(
                !json.contains(forbidden),
                "a hand-off must not be able to name authority: {json}"
            );
        }
    }

    // ---- interaction with the board ----------------------------------------

    #[test]
    fn a_claim_and_a_message_share_one_cursor_sequence() {
        let (mut b, mut c) = (board(), Coop::new());
        c.claim(&mut b, "w1", "a/b");
        c.dm(&mut b, "w1", "w2", "conflict on a/b", 0);
        c.claim(&mut b, "w2", "a/b");
        assert_eq!(b.latest_cursor(), 3);
        assert!(c.has_gap_unused());
    }

    impl Coop {
        /// No-op used only to keep the interaction test readable.
        fn has_gap_unused(&self) -> bool {
            self.claims.iter().all(|c| c.cursor > 0)
        }
    }

    #[test]
    fn a_board_that_dropped_entries_still_reports_its_bounds() {
        let mut b = board();
        let mut c = Coop::new();
        for i in 0..(CAPACITY + 3) {
            c.claim(&mut b, "w1", &format!("scope/{i}"));
        }
        assert_eq!(b.len(), CAPACITY);
        assert_eq!(b.dropped(), 3);
        assert!(b.has_gap(0));
    }
}
