//! H7: job queue with `claimSlot`/`unclaimSlot` semantics for scheduled
//! consensus/auction work.
//!
//! Mirrors qm's `cron/scheduler.ts`: a scheduled job is claimed before it
//! fires, `mark_fired` only advances it after a *successful* fire, a failed
//! fire is unclaimed (re-queued and retried), and an authz-fail disables the
//! job. The queue is pure state — no wall-clock, no random — so a replayed
//! schedule converges the same way the ledgers do.
//!
//! D14: claims are *ordered fenced*. A claim records the agreed tick it was
//! made at, and every settle (fire, unclaim, disable) is refused once the
//! queue's tick has advanced past that claim. This is [`LeaderLease`]'s
//! ordering applied to the queue: the lease decides *who* fires a tick, the
//! fence decides *which* tick a claim belongs to, so a superseded leader's
//! late settle cannot fire against a schedule the cluster has moved past.
//! [`crate::lease::LeaderLease`] couples the two.

use std::collections::HashMap;

use unfer_protocol::{Code, Diagnostic, Severity};

use crate::lease::LeaderLease;

/// Lifecycle of a scheduled job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// Scheduled, awaiting a slot claim.
    Queued,
    /// Claimed by a worker; firing in progress.
    Claimed,
    /// Fired successfully; not scheduled again (unless re-queued).
    Fired,
    /// Disabled on authz-fail; never fires again.
    Disabled,
}

/// A claim token for one job slot.
///
/// `tick` is the ordered fence: the agreed tick the claim was made at. A
/// settle carrying a claim whose tick is behind the queue's current tick is
/// stale and refused (D14).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobClaim {
    job: String,
    generation: u64,
    tick: u64,
}

/// Deterministic scheduler for consensus/auction work.
///
/// `tick` is the queue's current ordered fence. It only moves forward
/// ([`JobQueue::advance_tick`]); a claim settles only while the queue is still
/// at the tick the claim was made at.
#[derive(Debug, Clone, Default)]
pub struct JobQueue {
    jobs: HashMap<String, JobState>,
    generation: HashMap<String, u64>,
    tick: u64,
}

impl JobQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a scheduled job (idempotent — re-registering a known job keeps
    /// its state).
    pub fn enqueue(&mut self, job: &str) {
        self.jobs.entry(job.to_string()).or_insert(JobState::Queued);
        self.generation.entry(job.to_string()).or_insert(0);
    }

    /// `claimSlot`: take the job out of `Queued` into `Claimed`, returning a
    /// token that must accompany the eventual settle. `None` when the job is
    /// not queued (already claimed, fired, disabled, or unknown).
    pub fn claim_slot(&mut self, job: &str) -> Option<JobClaim> {
        if self.jobs.get(job) != Some(&JobState::Queued) {
            return None;
        }
        let generation = *self.generation.get(job).unwrap_or(&0);
        self.jobs.insert(job.to_string(), JobState::Claimed);
        Some(JobClaim {
            job: job.to_string(),
            generation,
            tick: self.tick,
        })
    }

    /// Advance the ordered fence to `tick`. Strictly monotonic: a tick that
    /// does not move forward (a clock regression or a replayed decision) is
    /// refused, so the fence can never be rewound to re-admit a stale claim.
    pub fn advance_tick(&mut self, tick: u64) -> Result<(), Diagnostic> {
        if tick <= self.tick {
            return Err(self.diag(
                Code::DUPLICATE_TRANSACTION,
                format!("tick {tick} does not advance the fence past {}", self.tick),
            ));
        }
        self.tick = tick;
        Ok(())
    }

    /// The queue's current ordered fence (the tick a fresh claim is made at).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Claim `job` at `tick` only if this node is the tick's single leader.
    ///
    /// D14 couples the two orderings at one call site: [`LeaderLease`] decides
    /// *who* fires a tick, and the fence decides *which* tick a claim belongs
    /// to. A non-leader is refused before the fence moves, so a losing node
    /// cannot consume an ordered tick; the leader's claim is then fenced to
    /// `tick` exactly like [`JobQueue::claim_slot`].
    pub fn claim_slot_leased(
        &mut self,
        job: &str,
        tick: u64,
        participants: usize,
        node_index: usize,
    ) -> Option<JobClaim> {
        if self.jobs.get(job) != Some(&JobState::Queued) {
            return None;
        }
        if !LeaderLease::is_leader(tick, participants, node_index) {
            return None;
        }
        self.advance_tick(tick).ok()?;
        self.claim_slot(job)
    }

    /// `unclaimSlot`: a failed fire returns the job to `Queued` so it is
    /// retried by a later claim. A stale/unknown claim is a no-op.
    pub fn unclaim_slot(&mut self, claim: &JobClaim) {
        if self.valid_claim(claim) {
            self.jobs.insert(claim.job.clone(), JobState::Queued);
            *self.generation.entry(claim.job.clone()).or_insert(0) += 1;
        }
    }

    /// `markFired`: only a *successful* fire marks the job fired. A stale
    /// claim (superseded generation) is refused, so a double-fire of the same
    /// slot cannot be recorded twice.
    pub fn mark_fired(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.diag(
                Code::DUPLICATE_TRANSACTION,
                format!("stale slot claim for job '{}'", claim.job),
            ));
        }
        self.jobs.insert(claim.job.clone(), JobState::Fired);
        *self.generation.entry(claim.job.clone()).or_insert(0) += 1;
        Ok(())
    }

    /// Disable the job on authz-fail: it never fires again.
    pub fn disable(&mut self, claim: &JobClaim) {
        if self.valid_claim(claim) {
            self.jobs.insert(claim.job.clone(), JobState::Disabled);
            *self.generation.entry(claim.job.clone()).or_insert(0) += 1;
        }
    }

    fn valid_claim(&self, claim: &JobClaim) -> bool {
        self.jobs.get(&claim.job) == Some(&JobState::Claimed)
            && self.generation.get(&claim.job) == Some(&claim.generation)
            && claim.tick == self.tick
    }

    pub fn state(&self, job: &str) -> Option<JobState> {
        self.jobs.get(job).copied()
    }

    fn diag(&self, code: Code, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::new(code, msg, Severity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_fire_mark_fired_roundtrip() {
        let mut q = JobQueue::new();
        q.enqueue("auction-settle");
        let claim = q.claim_slot("auction-settle").expect("claimable");
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
        q.mark_fired(&claim).unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
        // A fired job cannot be claimed again.
        assert!(q.claim_slot("auction-settle").is_none());
    }

    #[test]
    fn failed_fire_is_unqueued_and_retried() {
        let mut q = JobQueue::new();
        q.enqueue("settle-cert-ops");
        let claim = q.claim_slot("settle-cert-ops").unwrap();
        // Simulate a failed fire: unclaim re-queues; a later claim retries.
        q.unclaim_slot(&claim);
        assert_eq!(q.state("settle-cert-ops"), Some(JobState::Queued));
        let retry = q.claim_slot("settle-cert-ops").expect("retry claimable");
        q.mark_fired(&retry).unwrap();
        assert_eq!(q.state("settle-cert-ops"), Some(JobState::Fired));
    }

    #[test]
    fn stale_claim_cannot_mark_fired_twice() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        let claim = q.claim_slot("job").unwrap();
        q.unclaim_slot(&claim); // failed fire re-queued
        let retry = q.claim_slot("job").unwrap();
        // The old claim is superseded — marking it fired is refused.
        assert!(q.mark_fired(&claim).is_err());
        assert_eq!(q.state("job"), Some(JobState::Claimed));
        q.mark_fired(&retry).unwrap();
        assert_eq!(q.state("job"), Some(JobState::Fired));
    }

    #[test]
    fn authz_fail_disables_job() {
        let mut q = JobQueue::new();
        q.enqueue("sensitive-settle");
        let claim = q.claim_slot("sensitive-settle").unwrap();
        q.disable(&claim);
        assert_eq!(q.state("sensitive-settle"), Some(JobState::Disabled));
        assert!(
            q.claim_slot("sensitive-settle").is_none(),
            "disabled job never fires"
        );
    }

    #[test]
    fn double_claim_is_refused() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        let _first = q.claim_slot("job").unwrap();
        assert!(
            q.claim_slot("job").is_none(),
            "second claim must be refused"
        );
    }

    // ── D14: ordered fencing ────────────────────────────────────────────────

    #[test]
    fn advance_tick_is_strictly_monotonic() {
        let mut q = JobQueue::new();
        assert_eq!(q.tick(), 0);
        q.advance_tick(5).expect("forward tick");
        assert_eq!(q.tick(), 5);
        assert!(q.advance_tick(5).is_err(), "an equal tick does not advance");
        assert!(q.advance_tick(4).is_err(), "the fence never rewinds");
        assert_eq!(q.tick(), 5);
    }

    #[test]
    fn superseded_tick_cannot_fire() {
        // Node 0 claims at tick 0; the cluster advances to tick 1 (another
        // leader for this job). Node 0's late `mark_fired` must be refused
        // even though the claim's generation still matches.
        let mut q = JobQueue::new();
        q.enqueue("auction-settle");
        let claim = q.claim_slot("auction-settle").unwrap();
        q.advance_tick(1).unwrap();
        assert!(
            q.mark_fired(&claim).is_err(),
            "a claim from a superseded tick must not fire"
        );
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
    }

    #[test]
    fn superseded_tick_cannot_unclaim_or_disable() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        let claim = q.claim_slot("job").unwrap();
        q.advance_tick(2).unwrap();
        // Both settles are no-ops against the stale fence, so the claimed job
        // is neither re-queued nor disabled by a superseded leader.
        q.unclaim_slot(&claim);
        q.disable(&claim);
        assert_eq!(q.state("job"), Some(JobState::Claimed));
    }

    #[test]
    fn only_the_tick_leader_claims() {
        // Two participants: tick 3 elects index 1 (3 % 2).
        let mut q = JobQueue::new();
        q.enqueue("job");
        assert!(
            q.claim_slot_leased("job", 3, 2, 0).is_none(),
            "the non-leader must not claim"
        );
        assert_eq!(q.tick(), 0, "a refused claim does not move the fence");
        assert_eq!(q.state("job"), Some(JobState::Queued));
        let claim = q
            .claim_slot_leased("job", 3, 2, 1)
            .expect("the leader claims");
        assert_eq!(q.tick(), 3);
        q.mark_fired(&claim).expect("same-tick settle");
        assert_eq!(q.state("job"), Some(JobState::Fired));
    }

    #[test]
    fn zero_participants_never_claim() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        assert!(q.claim_slot_leased("job", 1, 0, 0).is_none());
        assert_eq!(q.state("job"), Some(JobState::Queued));
        assert_eq!(q.tick(), 0);
    }

    #[test]
    fn leased_claim_refuses_a_tick_that_does_not_advance() {
        let mut q = JobQueue::new();
        q.enqueue("a");
        q.enqueue("b");
        // Single participant: index 0 leads every tick.
        let first = q.claim_slot_leased("a", 4, 1, 0).expect("claim a");
        assert_eq!(q.tick(), 4);
        // Tick 4 cannot be re-used for a second job: the fence is ordered.
        assert!(q.claim_slot_leased("b", 4, 1, 0).is_none());
        assert_eq!(q.state("b"), Some(JobState::Queued));
        // A strictly later tick is admitted.
        let second = q.claim_slot_leased("b", 5, 1, 0).expect("claim b");
        q.mark_fired(&second).expect("b fires at tick 5");
        // `a` was claimed at tick 4 and is now behind the fence.
        assert!(
            q.mark_fired(&first).is_err(),
            "a superseded claim must not fire"
        );
    }

    #[test]
    fn claim_at_the_current_tick_fires() {
        // The fence admits a claim made at the tick the queue is at.
        let mut q = JobQueue::new();
        q.enqueue("job");
        q.advance_tick(7).unwrap();
        let claim = q.claim_slot("job").unwrap();
        assert_eq!(q.tick(), 7, "the claim records the current tick");
        q.mark_fired(&claim).expect("same-tick claim fires");
        assert_eq!(q.state("job"), Some(JobState::Fired));
    }
}
