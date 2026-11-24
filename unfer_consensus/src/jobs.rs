//! H7+D14: scheduled jobs under an ordered claim fence.
//!
//! Mirrors qm's `cron/scheduler.ts`: a scheduled job is claimed before it
//! fires, a successful fire settles it, a failed fire releases it (re-queued
//! and retried), and an authz-fail disables it. The queue is pure state — no
//! wall-clock, no random — so a replayed schedule converges the same way the
//! ledgers do.
//!
//! D14 upgrades the claim from a local counter to an *ordered fence*: the
//! committed state owns one current fence per resource, and a fence is the
//! tuple `(deployment, resource, epoch, generation)`. Only the current fence
//! may settle the job or mutate its protected resource, so a worker that is
//! still computing after its lease ended can finish its work but cannot commit
//! a stale result. `renew`, `release`, `reclaim` and restart are explicit
//! transitions rather than inferences from time. Ordering is by
//! `(epoch, generation)` — no correctness argument depends on synchronized
//! clocks, and an external sink without an enforcement interface is not fenced
//! at all ([`FencedSink`] models the ones that do enforce).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use unfer_protocol::{Code, Diagnostic, Severity};

/// Lifecycle of a scheduled job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    /// Scheduled, awaiting a claim.
    Queued,
    /// Claimed by a worker; firing in progress.
    Claimed,
    /// Settled successfully; not scheduled again (unless re-queued).
    Fired,
    /// Disabled on authz-fail; never fires again.
    Disabled,
}

/// The ordered fence for one protected resource.
///
/// `(epoch, generation)` orders fences for the same `(deployment, resource)`:
/// a later epoch means a later leadership incarnation, and a later generation
/// means a later claim within that incarnation. The committed state stores the
/// current fence; a claim holds a copy and is valid only while it equals it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobFence {
    pub deployment: String,
    pub resource: String,
    pub epoch: u64,
    pub generation: u64,
}

impl JobFence {
    /// Whether this fence is strictly newer than `other` for the same
    /// deployment and resource.
    pub fn supersedes(&self, other: &JobFence) -> bool {
        self.deployment == other.deployment
            && self.resource == other.resource
            && (self.epoch, self.generation) > (other.epoch, other.generation)
    }

    /// Whether this fence names the same deployment and resource as `other`.
    pub fn same_slot_as(&self, other: &JobFence) -> bool {
        self.deployment == other.deployment && self.resource == other.resource
    }
}

/// A held claim: the fence the worker was granted plus the holder it was
/// granted to. The holder is part of validity, so a fence value that leaks to
/// another worker is not enough to settle the job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobClaim {
    fence: JobFence,
    holder: String,
}

#[cfg(test)]
impl JobClaim {
    /// Build a claim from a known fence, for tests in sibling modules that
    /// need to present a superseded claim the queue never granted.
    pub(crate) fn for_test(fence: JobFence, holder: &str) -> Self {
        Self {
            fence,
            holder: holder.to_string(),
        }
    }
}

impl JobClaim {
    pub fn fence(&self) -> &JobFence {
        &self.fence
    }

    pub fn holder(&self) -> &str {
        &self.holder
    }

    pub fn resource(&self) -> &str {
        &self.fence.resource
    }
}

/// D14: the schema an ordered job command is read under. Distinct from the
/// ledger transaction schema, so the apply path can dispatch by schema instead
/// of guessing from the payload.
pub const JOB_COMMAND_SCHEMA: u16 = 2;

/// An ordered job command: the unit the deployment agrees on.
///
/// Every replica applies the same commands in the same order to its own
/// [`JobQueue`], so they converge on the same current fence. A command that is
/// refused (stale fence, wrong holder, not queued) is a deterministic no-op on
/// every replica — the verdict is stable, so it is not an error and changes no
/// state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobCommand {
    Claim { resource: String, holder: String },
    TakeOver {
        resource: String,
        epoch: u64,
        holder: String,
    },
    Renew { claim: JobClaim },
    Release { claim: JobClaim },
    Reclaim { resource: String, holder: String },
    Settle { claim: JobClaim },
    Disable { claim: JobClaim },
}

/// Deterministic scheduler and owner of the current fence per resource.
///
/// Serialized into the ledger snapshot, so the committed state — not a worker
/// — owns which fence is current.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobQueue {
    #[serde(default)]
    deployment: String,
    #[serde(default)]
    jobs: HashMap<String, JobState>,
    #[serde(default)]
    current: HashMap<String, JobFence>,
    #[serde(default)]
    holders: HashMap<String, String>,
    #[serde(default)]
    epoch: u64,
}

impl Default for JobQueue {
    fn default() -> Self {
        Self {
            deployment: String::new(),
            jobs: HashMap::new(),
            current: HashMap::new(),
            holders: HashMap::new(),
            epoch: 0,
        }
    }
}

impl JobQueue {
    pub fn new(deployment: impl Into<String>) -> Self {
        Self {
            deployment: deployment.into(),
            ..Self::default()
        }
    }

    pub fn deployment(&self) -> &str {
        &self.deployment
    }

    /// Set the deployment name when a snapshot predates it. The fence value
    /// from a different deployment is never reused.
    pub fn set_deployment(&mut self, deployment: impl Into<String>) {
        let deployment = deployment.into();
        if self.deployment != deployment {
            // A renamed deployment invalidates every outstanding fence: the
            // fences were ordered under the old name.
            self.current.clear();
            self.holders.clear();
            self.deployment = deployment;
        }
    }

    /// Register a scheduled job (idempotent — re-registering a known job keeps
    /// its state and its current fence).
    pub fn enqueue(&mut self, resource: &str) {
        self.jobs
            .entry(resource.to_string())
            .or_insert(JobState::Queued);
    }

    pub fn state(&self, resource: &str) -> Option<JobState> {
        self.jobs.get(resource).copied()
    }

    /// The current fence the committed state owns for `resource`.
    pub fn current_fence(&self, resource: &str) -> Option<&JobFence> {
        self.current.get(resource)
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Move to a later leadership incarnation. Strictly monotonic: a replay or
    /// a regression is refused, so an old epoch cannot be re-admitted.
    pub fn set_epoch(&mut self, epoch: u64) -> Result<(), Diagnostic> {
        if epoch <= self.epoch {
            return Err(diag(
                Code::DUPLICATE_TRANSACTION,
                format!("epoch {epoch} does not advance past {}", self.epoch),
            ));
        }
        self.epoch = epoch;
        Ok(())
    }

    /// Claim `resource` at the queue's current epoch.
    ///
    /// Replaying the same claim for the same holder is idempotent and returns
    /// the existing claim (no generation bump). A different holder while the
    /// resource is already claimed is refused; taking over needs an explicit
    /// [`JobQueue::reclaim`] or a later epoch via [`JobQueue::take_over`].
    pub fn claim(&mut self, resource: &str, holder: &str) -> Result<JobClaim, Diagnostic> {
        match self.jobs.get(resource) {
            Some(JobState::Queued) => {}
            Some(JobState::Claimed) => {
                let same_holder = self.holders.get(resource).map(String::as_str) == Some(holder);
                let fence = self.current.get(resource).cloned();
                if same_holder {
                    // Claim replay: the ordered command is idempotent.
                    let fence = fence.ok_or_else(|| {
                        diag(Code::INTERNAL, "claimed resource has no current fence")
                    })?;
                    return Ok(JobClaim {
                        fence,
                        holder: holder.to_string(),
                    });
                }
                return Err(diag(
                    Code::DUPLICATE_TRANSACTION,
                    format!("resource '{resource}' is already claimed"),
                ));
            }
            _ => {
                return Err(diag(
                    Code::DUPLICATE_TRANSACTION,
                    format!("resource '{resource}' is not queued"),
                ))
            }
        }
        let generation = self.next_generation(resource)?;
        let fence = self.fence_for(resource, self.epoch, generation);
        self.current.insert(resource.to_string(), fence.clone());
        self.holders.insert(resource.to_string(), holder.to_string());
        self.jobs.insert(resource.to_string(), JobState::Claimed);
        Ok(JobClaim {
            fence,
            holder: holder.to_string(),
        })
    }

    /// Ordered claim command at a later epoch: a worker that believes it leads
    /// a newer incarnation takes the resource over, invalidating the older
    /// fence even while the old holder is still running.
    pub fn take_over(
        &mut self,
        resource: &str,
        epoch: u64,
        holder: &str,
    ) -> Result<JobClaim, Diagnostic> {
        if self.jobs.get(resource).is_none() {
            return Err(diag(
                Code::INTERNAL,
                format!("resource '{resource}' is not registered"),
            ));
        }
        if matches!(
            self.jobs.get(resource),
            Some(JobState::Fired) | Some(JobState::Disabled)
        ) {
            return Err(diag(
                Code::DUPLICATE_TRANSACTION,
                format!("resource '{resource}' is not claimable"),
            ));
        }
        if epoch <= self.epoch {
            return Err(diag(
                Code::DUPLICATE_TRANSACTION,
                format!("epoch {epoch} does not supersede {}", self.epoch),
            ));
        }
        self.epoch = epoch;
        let generation = self.next_generation(resource)?;
        let fence = self.fence_for(resource, epoch, generation);
        self.current.insert(resource.to_string(), fence.clone());
        self.holders.insert(resource.to_string(), holder.to_string());
        self.jobs.insert(resource.to_string(), JobState::Claimed);
        Ok(JobClaim {
            fence,
            holder: holder.to_string(),
        })
    }

    /// Renew a held claim: a new generation for the same epoch, invalidating
    /// the old claim. The holder keeps the resource.
    pub fn renew(&mut self, claim: &JobClaim, holder: &str) -> Result<JobClaim, Diagnostic> {
        self.require_current(claim, holder)?;
        let generation = self.next_generation(claim.resource())?;
        let fence = self.fence_for(claim.resource(), claim.fence.epoch, generation);
        self.current.insert(claim.resource().to_string(), fence.clone());
        self.holders
            .insert(claim.resource().to_string(), holder.to_string());
        Ok(JobClaim {
            fence,
            holder: holder.to_string(),
        })
    }

    /// Release a failed fire: the resource returns to `Queued` and the released
    /// fence is superseded, so a delayed `settle` with it is refused.
    pub fn release(&mut self, claim: &JobClaim, holder: &str) -> Result<(), Diagnostic> {
        self.require_current(claim, holder)?;
        let generation = self.next_generation(claim.resource())?;
        let fence = self.fence_for(claim.resource(), claim.fence.epoch, generation);
        self.current.insert(claim.resource().to_string(), fence);
        self.holders.remove(claim.resource());
        self.jobs.insert(claim.resource().to_string(), JobState::Queued);
        Ok(())
    }

    /// Restart transition: a new process takes over a resource whose holder is
    /// gone. The old claim is invalid even if it is presented later.
    pub fn reclaim(&mut self, resource: &str, holder: &str) -> Result<JobClaim, Diagnostic> {
        match self.jobs.get(resource) {
            Some(JobState::Disabled) | Some(JobState::Fired) | None => {
                return Err(diag(
                    Code::DUPLICATE_TRANSACTION,
                    format!("resource '{resource}' cannot be reclaimed"),
                ))
            }
            Some(JobState::Claimed | JobState::Queued) => {}
        }
        let generation = self.next_generation(resource)?;
        let fence = self.fence_for(resource, self.epoch, generation);
        self.current.insert(resource.to_string(), fence.clone());
        self.holders.insert(resource.to_string(), holder.to_string());
        self.jobs.insert(resource.to_string(), JobState::Claimed);
        Ok(JobClaim {
            fence,
            holder: holder.to_string(),
        })
    }

    /// Only the current fence settles the job.
    pub fn settle(&mut self, claim: &JobClaim, holder: &str) -> Result<(), Diagnostic> {
        self.require_current(claim, holder)?;
        let generation = self.next_generation(claim.resource())?;
        let fence = self.fence_for(claim.resource(), claim.fence.epoch, generation);
        self.current.insert(claim.resource().to_string(), fence);
        self.holders.remove(claim.resource());
        self.jobs.insert(claim.resource().to_string(), JobState::Fired);
        Ok(())
    }

    /// Disable the job on authz-fail: it never fires again.
    pub fn disable(&mut self, claim: &JobClaim, holder: &str) -> Result<(), Diagnostic> {
        self.require_current(claim, holder)?;
        let generation = self.next_generation(claim.resource())?;
        let fence = self.fence_for(claim.resource(), claim.fence.epoch, generation);
        self.current.insert(claim.resource().to_string(), fence);
        self.holders.remove(claim.resource());
        self.jobs
            .insert(claim.resource().to_string(), JobState::Disabled);
        Ok(())
    }

    /// Apply one ordered command. The caller decides what a refusal means; this
    /// module keeps refusals side-effect free (a refused command leaves the
    /// queue exactly as it was), which is what makes the verdict replicable.
    pub fn apply(&mut self, command: &JobCommand) -> Result<(), Diagnostic> {
        match command {
            JobCommand::Claim { resource, holder } => self.claim(resource, holder).map(|_| ()),
            JobCommand::TakeOver {
                resource,
                epoch,
                holder,
            } => self.take_over(resource, *epoch, holder).map(|_| ()),
            JobCommand::Renew { claim } => self.renew(claim, claim.holder()).map(|_| ()),
            JobCommand::Release { claim } => self.release(claim, claim.holder()),
            JobCommand::Reclaim { resource, holder } => self.reclaim(resource, holder).map(|_| ()),
            JobCommand::Settle { claim } => self.settle(claim, claim.holder()),
            JobCommand::Disable { claim } => self.disable(claim, claim.holder()),
        }
    }

    /// Whether `claim` is the committed state's current fence for its resource.
    pub fn is_current(&self, claim: &JobClaim) -> bool {
        self.current.get(claim.resource()) == Some(&claim.fence)
            && self.holders.get(claim.resource()).map(String::as_str) == Some(&claim.holder)
            && self.jobs.get(claim.resource()) == Some(&JobState::Claimed)
    }

    fn require_current(&self, claim: &JobClaim, holder: &str) -> Result<(), Diagnostic> {
        if holder != claim.holder {
            return Err(diag(
                Code::DUPLICATE_TRANSACTION,
                format!("holder '{holder}' does not own the claim for '{}'", claim.resource()),
            ));
        }
        if !self.is_current(claim) {
            return Err(diag(
                Code::DUPLICATE_TRANSACTION,
                format!(
                    "stale fence for '{}' (epoch {}, generation {})",
                    claim.resource(),
                    claim.fence.epoch,
                    claim.fence.generation
                ),
            ));
        }
        Ok(())
    }

    fn fence_for(&self, resource: &str, epoch: u64, generation: u64) -> JobFence {
        JobFence {
            deployment: self.deployment.clone(),
            resource: resource.to_string(),
            epoch,
            generation,
        }
    }

    /// The next generation for a resource. Overflow is refused rather than
    /// wrapped: a wrapped fence would re-admit an old claim.
    fn next_generation(&self, resource: &str) -> Result<u64, Diagnostic> {
        let current = self
            .current
            .get(resource)
            .map(|fence| fence.generation)
            .unwrap_or(0);
        current.checked_add(1).ok_or_else(|| {
            diag(
                Code::INTERNAL,
                format!("fence generation overflow for resource '{resource}'"),
            )
        })
    }
}

/// A protected external sink with an enforcement interface (D14 step 2).
///
/// It remembers the last fence it accepted and refuses anything older, so a
/// stale worker cannot commit. A sink *without* such an interface cannot be
/// fenced; this type exists to show the accepting half, not to promise
/// enforcement for sinks that have none.
#[derive(Debug, Default)]
pub struct FencedSink {
    last: Option<JobFence>,
    effects: Vec<String>,
}

impl FencedSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `effect` under `fence`. A fence older than the last accepted one
    /// is refused; re-presenting the current fence is idempotent.
    pub fn mutate(&mut self, fence: &JobFence, effect: impl Into<String>) -> Result<(), Diagnostic> {
        if let Some(last) = &self.last {
            if !last.same_slot_as(fence) {
                return Err(diag(
                    Code::INTERNAL,
                    format!(
                        "fence for '{}/{}' does not address this sink",
                        fence.deployment, fence.resource
                    ),
                ));
            }
            if last.supersedes(fence) {
                return Err(diag(
                    Code::DUPLICATE_TRANSACTION,
                    format!(
                        "stale fence (epoch {}, generation {}) — current is (epoch {}, generation {})",
                        fence.epoch, fence.generation, last.epoch, last.generation
                    ),
                ));
            }
        }
        self.last = Some(fence.clone());
        self.effects.push(effect.into());
        Ok(())
    }

    pub fn accepted_generation(&self) -> Option<u64> {
        self.last.as_ref().map(|fence| fence.generation)
    }

    pub fn effects(&self) -> &[String] {
        &self.effects
    }
}

fn diag(code: Code, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::new(code, msg, Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEPLOY: &str = "deploy-1";

    fn queue() -> JobQueue {
        let mut q = JobQueue::new(DEPLOY);
        q.enqueue("auction-settle");
        q
    }

    #[test]
    fn claim_settle_roundtrip() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").expect("claimable");
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
        assert!(q.is_current(&claim));
        q.settle(&claim, "worker-a").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
        // A fired job cannot be claimed again.
        assert!(q.claim("auction-settle", "worker-a").is_err());
    }

    #[test]
    fn claim_replay_by_the_same_holder_is_idempotent() {
        let mut q = queue();
        let first = q.claim("auction-settle", "worker-a").unwrap();
        let replay = q.claim("auction-settle", "worker-a").unwrap();
        assert_eq!(first, replay, "a replayed claim command returns the same fence");
        assert_eq!(
            q.current_fence("auction-settle").unwrap().generation,
            first.fence().generation,
            "no generation bump on replay"
        );
    }

    #[test]
    fn a_second_holder_cannot_claim_what_a_live_holder_owns() {
        let mut q = queue();
        let _ = q.claim("auction-settle", "worker-a").unwrap();
        assert!(q.claim("auction-settle", "worker-b").is_err());
    }

    #[test]
    fn failed_fire_is_released_and_retried() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        q.release(&claim, "worker-a").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Queued));
        let retry = q.claim("auction-settle", "worker-a").unwrap();
        q.settle(&retry, "worker-a").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
    }

    #[test]
    fn a_released_fence_cannot_settle_late() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        q.release(&claim, "worker-a").unwrap();
        let retry = q.claim("auction-settle", "worker-a").unwrap();
        // The old fence is superseded even though the same worker holds both.
        assert!(q.settle(&claim, "worker-a").is_err());
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
        q.settle(&retry, "worker-a").unwrap();
    }

    #[test]
    fn renew_invalidates_the_old_claim() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        let renewed = q.renew(&claim, "worker-a").unwrap();
        assert!(renewed.fence().generation > claim.fence().generation);
        assert!(!q.is_current(&claim), "the pre-renew fence is stale");
        assert!(q.settle(&claim, "worker-a").is_err());
        q.settle(&renewed, "worker-a").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
    }

    #[test]
    fn restart_reclaims_the_claimed_job() {
        let mut q = queue();
        let old = q.claim("auction-settle", "worker-a").unwrap();
        // worker-a dies; a restarted process takes the resource over.
        let restarted = q.reclaim("auction-settle", "worker-b").unwrap();
        assert!(!q.is_current(&old));
        assert!(q.settle(&old, "worker-a").is_err(), "the dead holder cannot settle");
        q.settle(&restarted, "worker-b").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Fired));
    }

    #[test]
    fn two_partitioned_workers_both_believing_they_lead() {
        // Commit state: worker-a claims at epoch 1, then the cluster elects a
        // new leader in epoch 2 and worker-b takes over. Worker-a, partitioned,
        // does not know; it finishes computing and tries to settle.
        let mut q = queue();
        let a = q.claim("auction-settle", "worker-a").unwrap();
        let b = q.take_over("auction-settle", 2, "worker-b").unwrap();
        assert!(q.settle(&a, "worker-a").is_err(), "the old leader's result is stale");
        assert!(
            q.settle(&b, "worker-b").is_ok(),
            "only the current fence settles the job"
        );
    }

    #[test]
    fn stale_release_cannot_release_a_newer_claim() {
        let mut q = queue();
        let a = q.claim("auction-settle", "worker-a").unwrap();
        let b = q.take_over("auction-settle", 3, "worker-b").unwrap();
        assert!(q.release(&a, "worker-a").is_err());
        assert_eq!(q.state("auction-settle"), Some(JobState::Claimed));
        assert!(q.is_current(&b), "worker-b's claim is untouched");
    }

    #[test]
    fn generation_overflow_is_refused_not_wrapped() {
        // Drive the committed fence to the last generation. Tests live in this
        // module, so they may set the committed value directly; no public API
        // lets a caller do this.
        let mut q = queue();
        q.current.insert(
            "auction-settle".to_string(),
            JobFence {
                deployment: DEPLOY.to_string(),
                resource: "auction-settle".to_string(),
                epoch: 0,
                generation: u64::MAX,
            },
        );
        let error = q.claim("auction-settle", "worker-a").unwrap_err();
        assert!(
            error.message.contains("generation overflow"),
            "overflow is an error, not a wrap: {error:?}"
        );
        assert_eq!(
            q.current_fence("auction-settle").unwrap().generation,
            u64::MAX,
            "the committed fence is unchanged"
        );
        assert_eq!(q.state("auction-settle"), Some(JobState::Queued));
    }

    #[test]
    fn epoch_only_moves_forward() {
        let mut q = queue();
        q.set_epoch(4).unwrap();
        assert_eq!(q.epoch(), 4);
        assert!(q.set_epoch(4).is_err());
        assert!(q.set_epoch(3).is_err());
        assert_eq!(q.epoch(), 4);
    }

    #[test]
    fn authz_fail_disables_the_job() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        q.disable(&claim, "worker-a").unwrap();
        assert_eq!(q.state("auction-settle"), Some(JobState::Disabled));
        assert!(q.claim("auction-settle", "worker-a").is_err());
        assert!(q.reclaim("auction-settle", "worker-b").is_err());
    }

    #[test]
    fn a_fence_from_another_deployment_is_not_current() {
        let mut q = queue();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        q.set_deployment("deploy-2");
        assert!(!q.is_current(&claim), "fences do not cross deployments");
        assert!(q.settle(&claim, "worker-a").is_err());
    }

    #[test]
    fn sink_rejects_a_stale_fence() {
        let mut q = queue();
        let mut sink = FencedSink::new();
        let first = q.claim("auction-settle", "worker-a").unwrap();
        sink.mutate(first.fence(), "write-1").unwrap();
        let second = q.reclaim("auction-settle", "worker-b").unwrap();
        sink.mutate(second.fence(), "write-2").unwrap();
        let error = sink.mutate(first.fence(), "write-3").unwrap_err();
        assert!(error.message.contains("stale fence"), "{error:?}");
        assert_eq!(sink.effects(), ["write-1", "write-2"]);
        assert_eq!(sink.accepted_generation(), Some(second.fence().generation));
    }

    #[test]
    fn sink_accepts_an_idempotent_retry_of_its_current_fence() {
        let mut q = queue();
        let mut sink = FencedSink::new();
        let claim = q.claim("auction-settle", "worker-a").unwrap();
        sink.mutate(claim.fence(), "write-1").unwrap();
        sink.mutate(claim.fence(), "write-1").unwrap();
        assert_eq!(sink.effects().len(), 2, "the sink applies what it is told");
    }

    #[test]
    fn sink_refuses_a_fence_for_a_different_resource() {
        let mut q = queue();
        q.enqueue("cert-settle");
        let mut sink = FencedSink::new();
        let auction = q.claim("auction-settle", "worker-a").unwrap();
        sink.mutate(auction.fence(), "write").unwrap();
        let cert = q.claim("cert-settle", "worker-a").unwrap();
        assert!(sink.mutate(cert.fence(), "write").is_err());
    }

    #[test]
    fn fence_ordering_is_epoch_then_generation() {
        let base = JobFence {
            deployment: DEPLOY.to_string(),
            resource: "r".to_string(),
            epoch: 1,
            generation: 9,
        };
        let newer_epoch = JobFence {
            epoch: 2,
            generation: 1,
            ..base.clone()
        };
        let newer_generation = JobFence {
            generation: 10,
            ..base.clone()
        };
        assert!(newer_epoch.supersedes(&base));
        assert!(newer_generation.supersedes(&base));
        assert!(!base.supersedes(&newer_epoch));
        assert!(!base.supersedes(&base));
        let other_resource = JobFence {
            resource: "s".to_string(),
            ..newer_epoch.clone()
        };
        assert!(!other_resource.supersedes(&base), "resources are ordered separately");
    }
}
