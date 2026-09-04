//! H7: job queue with `claimSlot`/`unclaimSlot` semantics for scheduled
//! consensus/auction work.
//!
//! Mirrors qm's `cron/scheduler.ts`: a scheduled job is claimed before it
//! fires, `mark_fired` only advances it after a *successful* fire, a failed
//! fire is unclaimed (re-queued and retried), and an authz-fail disables the
//! job. The queue is pure state — no wall-clock, no random — so a replayed
//! schedule converges the same way the ledgers do.

use std::collections::HashMap;

use unfer_protocol::{Code, Diagnostic, Severity};

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobClaim {
    job: String,
    generation: u64,
}

/// Deterministic scheduler for consensus/auction work.
#[derive(Debug, Clone, Default)]
pub struct JobQueue {
    jobs: HashMap<String, JobState>,
    generation: HashMap<String, u64>,
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
        })
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
}
