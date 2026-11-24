//! H7: job queue with `claimSlot`/`unclaimSlot` semantics for scheduled
//! consensus/auction work, and (D14) the ordered fence that decides which
//! worker may act.
//!
//! Mirrors qm's `cron/scheduler.ts`: a scheduled job is claimed before it
//! fires, `mark_fired` only advances it after a *successful* fire, a failed
//! fire is unclaimed (re-queued and retried), and an authz-fail disables the
//! job. The queue is pure state — no wall-clock, no random — so a replayed
//! schedule converges the same way the ledgers do.
//!
//! ## D14: ordering, not consensus
//!
//! A claim used to be a generation counter inside one process, which is enough
//! for one worker and not enough for two: a worker that lost the job while
//! partitioned keeps computing, and the counter it holds no longer says
//! anything about what the *committed* state believes. So the token is now a
//! [`Fence`] — `(deployment, resource, epoch, generation)` — and the current
//! fence for a resource lives in [`FenceTable`], which the node owns and
//! rebuilds from its journal. Only the current fence settles the job or mutates
//! the resource (`admits`); an older one is refused *before* anything is
//! written, so a stale worker can keep computing and cannot commit.
//!
//! Nothing here reads a clock. `epoch` is the reset point (it changes only when
//! `generation` would overflow, or when a caller explicitly advances it), and
//! `generation` advances once per claim transition. Fences are compared
//! lexicographically on `(epoch, generation)`, so a fence is stale if it is not
//! the current one — not if it is "old in seconds". A worker whose clock is
//! wrong is not a case this has to handle, which is the point: leases need
//! synchronised clocks, fences do not.
//!
//! `epoch` is also what a *sink* is told. [`FencedSink`] is the enforcement
//! interface: a sink that implements it rejects a fence older than the newest
//! one it has already applied. A sink with no such interface cannot be
//! protected by a fence, and this module does not pretend otherwise — the plan
//! says so explicitly, and D15's durable outbox is where an unenforceable sink
//! is handled by reconciliation rather than by a promise.
//!
//! The round-robin leader in [`crate::lease`] stays a scheduling *preference*:
//! it decides who is likely to claim next, and correctness rests on the fence.

use std::collections::{HashMap, HashSet};

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

/// The disposition a fence transition leaves its job in.
///
/// This is what recovery restores: a fence record says *which* claim became
/// current **and** what the job became, because "released" and "fired" are not
/// the same thing to a scheduler — one is retried by the next claim, the other
/// is not retried at all. Recovering only the fence would make a restart
/// re-fire a job that had already fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimState {
    /// Held by `holder` under this fence.
    Claimed,
    /// Returned to the queue after a failed fire: the next claim retries it.
    Queued,
    /// Fired successfully under this fence; not retried.
    Fired,
    /// Disabled on an authorization failure; never fires again.
    Disabled,
}

impl ClaimState {
    fn job_state(self) -> JobState {
        match self {
            ClaimState::Claimed => JobState::Claimed,
            ClaimState::Queued => JobState::Queued,
            ClaimState::Fired => JobState::Fired,
            ClaimState::Disabled => JobState::Disabled,
        }
    }

    /// Whether the resource is idle under this disposition.
    fn is_idle(self) -> bool {
        !matches!(self, ClaimState::Claimed)
    }
}

/// D14: the ordered right to act on one protected resource.
///
/// `deployment` and `resource` say *what* is protected; `(epoch, generation)`
/// say *which* claim, and are the only fields that move. The fence is issued by
/// the committed state, never by the worker that receives it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fence {
    pub deployment: String,
    pub resource: String,
    pub epoch: u64,
    pub generation: u64,
}

impl Fence {
    pub fn new(
        deployment: impl Into<String>,
        resource: impl Into<String>,
        epoch: u64,
        generation: u64,
    ) -> Self {
        Self {
            deployment: deployment.into(),
            resource: resource.into(),
            epoch,
            generation,
        }
    }

    /// Whether this fence is strictly older than `other` for the same resource.
    ///
    /// Comparing fences of different resources (or deployments) is meaningless;
    /// callers compare `(epoch, generation)` only within one resource, which is
    /// how the table stores them.
    pub fn is_older_than(&self, other: &Fence) -> bool {
        (self.epoch, self.generation) < (other.epoch, other.generation)
    }

    /// The fence one transition after this one, or `None` when the fence space
    /// is exhausted.
    ///
    /// At `generation == u64::MAX` the epoch advances and the generation restarts
    /// at 1 — an explicit epoch change, not a wrap. A wrap would hand out a
    /// fence that sorts *below* the current one and would therefore look stale to
    /// a sink forever; refusing at the very top instead keeps every issued fence
    /// strictly increasing.
    fn successor(&self) -> Option<Fence> {
        let (epoch, generation) = next_position(self.epoch, self.generation)?;
        Some(Fence {
            deployment: self.deployment.clone(),
            resource: self.resource.clone(),
            epoch,
            generation,
        })
    }
}

/// The first position of a fence space, then each strictly increasing position.
fn next_position(epoch: u64, generation: u64) -> Option<(u64, u64)> {
    if generation < u64::MAX {
        Some((epoch, generation + 1))
    } else if epoch < u64::MAX {
        Some((epoch + 1, 1))
    } else {
        None
    }
}

/// The committed fencing state: the current fence and holder per resource.
///
/// This is deterministic state — the same transitions in the same order produce
/// the same table — so a node rebuilds it by replaying its journal rather than
/// by asking a peer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FenceTable {
    current: HashMap<String, Fence>,
    holders: HashMap<String, String>,
    /// Resources whose current fence was explicitly released. Kept rather than
    /// deleted so a *stale* release (of a fence that is no longer current) is
    /// still distinguishable from a release of an idle resource.
    released: HashSet<String>,
}

impl FenceTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reinstall a fence recovered from the journal.
    pub fn install(&mut self, fence: Fence, holder: Option<String>, released: bool) {
        if released {
            self.released.insert(fence.resource.clone());
        } else {
            self.released.remove(&fence.resource);
        }
        match holder {
            Some(holder) => self.holders.insert(fence.resource.clone(), holder),
            None => self.holders.remove(&fence.resource),
        };
        self.current.insert(fence.resource.clone(), fence);
    }

    pub fn current(&self, resource: &str) -> Option<&Fence> {
        self.current.get(resource)
    }

    pub fn holder(&self, resource: &str) -> Option<&str> {
        self.holders.get(resource).map(String::as_str)
    }

    pub fn is_released(&self, resource: &str) -> bool {
        self.released.contains(resource)
    }

    /// Whether `fence` is the current fence for its resource.
    ///
    /// This is the only question a protected mutation asks. It is deliberately
    /// an identity check and not an ordering comparison: two fences can be
    /// ordered yet only one of them is the committed one.
    pub fn admits(&self, fence: &Fence) -> bool {
        self.current.get(&fence.resource) == Some(fence)
    }

    /// Issue the first (or next) fence for a resource nobody holds.
    ///
    /// Refused while a live claim exists: taking a held resource over is
    /// [`FenceTable::reclaim`]'s decision, and it requires a release first, so
    /// "two workers both believe they lead" cannot be resolved by whoever asks
    /// last.
    pub fn claim(
        &mut self,
        deployment: &str,
        resource: &str,
        holder: &str,
    ) -> Result<Fence, Diagnostic> {
        if self.current.contains_key(resource) && !self.released.contains(resource) {
            return Err(stale(
                "RESOURCE_HELD",
                format!(
                    "resource '{resource}' is held by '{}' under fence {}; a claim needs a release first",
                    self.holders.get(resource).map(String::as_str).unwrap_or("unknown"),
                    self.render(resource)
                ),
            ));
        }
        let next = match self.current.get(resource) {
            None => Fence::new(deployment, resource, 1, 1),
            Some(current) => current.successor().ok_or_else(|| {
                stale(
                    "FENCE_EXHAUSTED",
                    format!(
                        "the fence space for '{resource}' is exhausted at {}",
                        self.render(resource)
                    ),
                )
            })?,
        };
        self.install(next.clone(), Some(holder.to_string()), false);
        Ok(next)
    }

    /// Extend the current claim: a fresh fence, same holder, same epoch.
    ///
    /// A renewal is a *new* fence, so an outstanding order against the old one
    /// is stale by construction — renewal is how a worker re-arms, not how it
    /// widens the window in which an old result may still land.
    pub fn renew(&mut self, fence: &Fence) -> Result<Fence, Diagnostic> {
        if !self.admits(fence) {
            return Err(stale(
                "STALE_FENCE",
                format!(
                    "renew refused: {} is not the current fence ({})",
                    self.describe(fence),
                    self.render(&fence.resource)
                ),
            ));
        }
        if self.released.contains(&fence.resource) {
            return Err(stale(
                "FENCE_RELEASED",
                format!("renew refused: '{}' was released", fence.resource),
            ));
        }
        let next = fence.successor().ok_or_else(|| {
            stale(
                "FENCE_EXHAUSTED",
                format!(
                    "the fence space for '{}' is exhausted at {}",
                    fence.resource,
                    self.describe(fence)
                ),
            )
        })?;
        self.install(next.clone(), self.holders.get(&fence.resource).cloned(), false);
        Ok(next)
    }

    /// Give the resource up. Only the current fence may release it, so a stale
    /// worker cannot knock a newer holder out of its claim.
    pub fn release(&mut self, fence: &Fence) -> Result<(), Diagnostic> {
        if !self.admits(fence) {
            return Err(stale(
                "STALE_FENCE",
                format!(
                    "release refused: {} is not the current fence ({})",
                    self.describe(fence),
                    self.render(&fence.resource)
                ),
            ));
        }
        self.install(fence.clone(), None, true);
        Ok(())
    }

    /// Take over a resource whose claim was released (an idle resource), issuing
    /// a fence that is strictly newer than the released one.
    ///
    /// A claim is refused while the resource is held, so reclaim and claim differ
    /// only in *intent*; the reason both exist is that a takeover of a resource
    /// that was never released must be a deliberate, reviewable act, not a
    /// convenient default.
    pub fn reclaim(
        &mut self,
        deployment: &str,
        resource: &str,
        holder: &str,
    ) -> Result<Fence, Diagnostic> {
        if self.current.contains_key(resource) && !self.released.contains(resource) {
            return Err(stale(
                "RESOURCE_HELD",
                format!(
                    "reclaim refused: '{resource}' is still held under {}",
                    self.render(resource)
                ),
            ));
        }
        self.claim(deployment, resource, holder)
    }

    fn describe(&self, fence: &Fence) -> String {
        format!(
            "({}, {}, epoch {}, generation {})",
            fence.deployment, fence.resource, fence.epoch, fence.generation
        )
    }

    fn render(&self, resource: &str) -> String {
        match self.current.get(resource) {
            Some(fence) => self.describe(fence),
            None => format!("no fence for '{resource}'"),
        }
    }
}

/// A claim token for one job slot: the job plus the fence that authorizes it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobClaim {
    job: String,
    fence: Fence,
}

impl JobClaim {
    pub fn job(&self) -> &str {
        &self.job
    }

    pub fn fence(&self) -> &Fence {
        &self.fence
    }
}

/// The enforcement interface for a protected external effect (D14 step 2).
///
/// A sink that cannot answer "is this fence newer than the last one I applied?"
/// cannot be fenced; it needs a durable outbox and reconciliation instead
/// (D15). Nothing in this module claims protection for such a sink.
pub trait FencedSink {
    /// Apply `payload` under `fence`, refusing anything not newer than the last
    /// fence this sink accepted for the resource.
    fn apply(&mut self, fence: &Fence, payload: &str) -> Result<(), Diagnostic>;
}

/// A sink that enforces ordering itself — the mock an external service stands in
/// for, and the model of one that keeps its applied fence durably.
#[derive(Debug, Clone, Default)]
pub struct OrderedMockSink {
    /// The newest fence this sink has applied, per resource.
    seen: HashMap<String, Fence>,
    /// Every applied effect, in order: `(resource, epoch, generation, payload)`.
    pub applied: Vec<(String, u64, u64, String)>,
}

impl OrderedMockSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn applied_count(&self, resource: &str) -> usize {
        self.applied
            .iter()
            .filter(|(name, _, _, _)| name == resource)
            .count()
    }
}

impl FencedSink for OrderedMockSink {
    fn apply(&mut self, fence: &Fence, payload: &str) -> Result<(), Diagnostic> {
        if let Some(seen) = self.seen.get(&fence.resource) {
            if fence.is_older_than(seen) || fence == seen {
                return Err(stale(
                    "STALE_FENCE",
                    format!(
                        "sink refused ({}, epoch {}, generation {}): it already applied epoch {}, generation {}",
                        fence.resource, fence.epoch, fence.generation, seen.epoch, seen.generation
                    ),
                ));
            }
        }
        self.seen.insert(fence.resource.clone(), fence.clone());
        self.applied.push((
            fence.resource.clone(),
            fence.epoch,
            fence.generation,
            payload.to_string(),
        ));
        Ok(())
    }
}

/// Deterministic scheduler for consensus/auction work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobQueue {
    jobs: HashMap<String, JobState>,
    fences: FenceTable,
    deployment: String,
}

impl JobQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Name the deployment whose fences this queue issues. Fences are scoped by
    /// it, so two deployments can use the same job name without sharing a fence.
    pub fn with_deployment(deployment: impl Into<String>) -> Self {
        Self {
            deployment: deployment.into(),
            ..Self::default()
        }
    }

    pub fn deployment(&self) -> &str {
        &self.deployment
    }

    pub fn fences(&self) -> &FenceTable {
        &self.fences
    }

    /// Reinstall fencing state recovered from the journal: both which claim is
    /// current and what the job itself became.
    pub fn install_fence(&mut self, fence: Fence, holder: Option<String>, state: ClaimState) {
        self.fences.install(fence.clone(), holder, state.is_idle());
        self.jobs.insert(fence.resource.clone(), state.job_state());
    }

    /// Register a scheduled job (idempotent — re-registering a known job keeps
    /// its state).
    pub fn enqueue(&mut self, job: &str) {
        self.jobs.entry(job.to_string()).or_insert(JobState::Queued);
    }

    /// `claimSlot`: take the job out of `Queued` into `Claimed`, returning the
    /// fence that authorizes the fire.
    ///
    /// Refused when the job is not queued (already claimed, fired, disabled, or
    /// unknown), and when the fence space for the job is exhausted — a claim
    /// that cannot be ordered is not issued.
    pub fn claim(&mut self, job: &str, holder: &str) -> Result<JobClaim, Diagnostic> {
        // The fence is asked first: for a job someone already holds, the honest
        // answer is "that resource is held", not "this job is not queued" — the
        // second reads like the caller is late, and the first tells it who owns
        // the resource. A job that is fired, disabled or unknown fails the state
        // check below, and the issued fence is rolled back so nothing is
        // published that no job backs.
        let before = self.fences.clone();
        let fence = self.fences.claim(&self.deployment, job, holder)?;
        if self.jobs.get(job) != Some(&JobState::Queued) {
            self.fences = before;
            return Err(stale(
                "JOB_NOT_QUEUED",
                format!("job '{job}' is not queued and cannot be claimed"),
            ));
        }
        self.jobs.insert(job.to_string(), JobState::Claimed);
        Ok(JobClaim {
            job: job.to_string(),
            fence,
        })
    }

    /// Re-arm the claim: a new fence for the same job and holder.
    pub fn renew(&mut self, claim: &JobClaim) -> Result<JobClaim, Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.stale_claim(claim, "renew"));
        }
        let fence = self.fences.renew(&claim.fence)?;
        Ok(JobClaim {
            job: claim.job.clone(),
            fence,
        })
    }

    /// `unclaimSlot`: a failed fire returns the job to `Queued` so it is retried
    /// by a later claim, and releases the fence. A stale/unknown claim is
    /// refused and changes nothing.
    pub fn unclaim(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.stale_claim(claim, "unclaim"));
        }
        self.fences.release(&claim.fence)?;
        self.jobs.insert(claim.job.clone(), JobState::Queued);
        Ok(())
    }

    /// `markFired`: only a *successful* fire marks the job fired, and only under
    /// the current fence — a double-fire of the same slot cannot be recorded
    /// twice, and a stale worker's completion cannot be recorded at all.
    pub fn mark_fired(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.stale_claim(claim, "mark_fired"));
        }
        self.fences.release(&claim.fence)?;
        self.jobs.insert(claim.job.clone(), JobState::Fired);
        Ok(())
    }

    /// Disable the job on authz-fail: it never fires again.
    pub fn disable(&mut self, claim: &JobClaim) -> Result<(), Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.stale_claim(claim, "disable"));
        }
        self.fences.release(&claim.fence)?;
        self.jobs.insert(claim.job.clone(), JobState::Disabled);
        Ok(())
    }

    /// Settle the job through a protected sink: the sink applies the effect and
    /// the job is marked fired only if the fence is current for *both*.
    ///
    /// The order matters. The fence is checked here first so a stale worker
    /// cannot reach the sink at all; the sink then re-checks, because its
    /// ordering knowledge is its own and a fence that was current when the
    /// worker started may not be current when the effect arrives.
    pub fn settle(&mut self, claim: &JobClaim, sink: &mut dyn FencedSink, payload: &str) -> Result<(), Diagnostic> {
        if !self.valid_claim(claim) {
            return Err(self.stale_claim(claim, "settle"));
        }
        sink.apply(&claim.fence, payload)?;
        self.mark_fired(claim)
    }

    fn valid_claim(&self, claim: &JobClaim) -> bool {
        self.jobs.get(&claim.job) == Some(&JobState::Claimed) && self.fences.admits(&claim.fence)
    }

    fn stale_claim(&self, claim: &JobClaim, operation: &str) -> Diagnostic {
        stale(
            "STALE_FENCE",
            format!(
                "{operation} refused for job '{}': fence (epoch {}, generation {}) is not the current one ({})",
                claim.job,
                claim.fence.epoch,
                claim.fence.generation,
                match self.fences.current(&claim.job) {
                    Some(fence) => format!("epoch {}, generation {}", fence.epoch, fence.generation),
                    None => "no fence".to_string(),
                }
            ),
        )
    }

    pub fn state(&self, job: &str) -> Option<JobState> {
        self.jobs.get(job).copied()
    }

    pub fn fence(&self, job: &str) -> Option<&Fence> {
        self.fences.current(job)
    }
}

/// A fence that is not current is a conflict the caller may recover from (it can
/// simply stop), not a duplicate of something it already sent — but it uses the
/// same code the queue used for a stale slot claim before D14, so a caller that
/// already treats `DUPLICATE_TRANSACTION` as "your token is no longer valid"
/// keeps working.
fn stale(kind: &str, message: String) -> Diagnostic {
    Diagnostic::new(
        Code::DUPLICATE_TRANSACTION,
        format!("{kind}: {message}"),
        Severity::Error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_fire_mark_fired_roundtrip() {
        let mut q = JobQueue::new();
        q.enqueue("auction-settle");
        let claim = q.claim("auction-settle", "worker-0").expect("claimable");
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
        q.mark_fired(&claim).unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
        // A fired job cannot be claimed again.
        assert!(q.claim("auction-settle", "worker-0").is_err());
    }

    #[test]
    fn failed_fire_is_unqueued_and_retried() {
        let mut q = JobQueue::new();
        q.enqueue("settle-cert-ops");
        let claim = q.claim("settle-cert-ops", "worker-0").unwrap();
        // Simulate a failed fire: unclaim re-queues; a later claim retries.
        q.unclaim(&claim).unwrap();
        assert_eq!(q.state("settle-cert-ops"), Some(JobState::Queued));
        let retry = q.claim("settle-cert-ops", "worker-1").expect("retry claimable");
        q.mark_fired(&retry).unwrap();
        assert_eq!(q.state("settle-cert-ops"), Some(JobState::Fired));
        // The fences are strictly ordered across the retry.
        assert!(claim.fence().is_older_than(retry.fence()));
    }

    #[test]
    fn stale_claim_cannot_mark_fired_twice() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        let claim = q.claim("job", "worker-0").unwrap();
        q.unclaim(&claim).unwrap(); // failed fire re-queued
        let retry = q.claim("job", "worker-1").unwrap();
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
        let claim = q.claim("sensitive-settle", "worker-0").unwrap();
        q.disable(&claim).unwrap();
        assert_eq!(q.state("sensitive-settle"), Some(JobState::Disabled));
        assert!(
            q.claim("sensitive-settle", "worker-0").is_err(),
            "disabled job never fires"
        );
    }

    #[test]
    fn double_claim_is_refused() {
        let mut q = JobQueue::new();
        q.enqueue("job");
        let _first = q.claim("job", "worker-0").unwrap();
        assert!(q.claim("job", "worker-1").is_err(), "second claim must be refused");
    }

    #[test]
    fn fences_are_strictly_ordered_per_resource() {
        let mut table = FenceTable::new();
        let first = table.claim("dep", "res", "worker-0").unwrap();
        table.release(&first).unwrap();
        let second = table.reclaim("dep", "res", "worker-1").unwrap();
        assert!(first.is_older_than(&second));
        assert!(table.admits(&second));
        assert!(!table.admits(&first));
    }

    #[test]
    fn generation_overflow_advances_the_epoch_instead_of_wrapping() {
        let mut table = FenceTable::new();
        table.install(Fence::new("dep", "res", 2, u64::MAX - 1), Some("w".into()), false);
        let last = table.current("res").unwrap().clone();
        let next = table.renew(&last).unwrap();
        assert_eq!((next.epoch, next.generation), (2, u64::MAX));
        // At the very top of the space the epoch advances and the generation
        // restarts, so the new fence is strictly newer rather than wrapped.
        let advanced = table.renew(&next).unwrap();
        assert_eq!((advanced.epoch, advanced.generation), (3, 1));
        assert!(next.is_older_than(&advanced));
    }

    #[test]
    fn exhausted_fence_space_is_refused_not_wrapped() {
        let mut table = FenceTable::new();
        table.install(
            Fence::new("dep", "res", u64::MAX, u64::MAX),
            Some("w".into()),
            false,
        );
        let top = table.current("res").unwrap().clone();
        assert!(
            table.renew(&top).is_err(),
            "a fence space with no successor must refuse rather than wrap"
        );
        assert!(table.admits(&top), "the current fence is still the current one");
    }
}
