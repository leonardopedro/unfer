//! D14 acceptance: only the current fence can settle a job or mutate its
//! protected resource, and the old worker can keep computing but cannot commit.
//!
//! The scenarios are the ones the plan names — two partitioned workers both
//! believing they lead, a delayed old completion, claim replay, generation
//! overflow, restart, and a stale release of a newer claim — plus the sink that
//! enforces order on its own. Each one asserts the *state* after the refusal, not
//! just the error: a refusal that still moved the job would pass a check that
//! only looks at the return value.

use unfer_consensus::jobs::{
    ClaimState, Fence, FenceTable, FencedSink, JobQueue, JobState, OrderedMockSink,
};
use unfer_consensus::storage::{FenceRecord, JournalRecord, TransitionLog};
use unfer_consensus::{ConsensusNode, MintAuthority};

struct NoEngine;

impl unfer_consensus::ConsensusEngine for NoEngine {
    fn submit(
        &self,
        _tx: unfer_protocol::ConsensusTransaction,
    ) -> Result<u64, unfer_protocol::Diagnostic> {
        unreachable!("the fencing tests never submit a transaction")
    }

    fn get_log(&self, _from_seq: u64) -> Vec<(u64, unfer_protocol::ConsensusTransaction)> {
        Vec::new()
    }

    fn current_seq(&self) -> u64 {
        0
    }

    fn receipt(
        &self,
        _id: unfer_protocol::submission::SubmissionId,
    ) -> unfer_protocol::submission::ReceiptState {
        unfer_protocol::submission::ReceiptState::Pending
    }

    fn wait(
        &self,
        id: unfer_protocol::submission::SubmissionId,
        _timeout: std::time::Duration,
    ) -> unfer_protocol::submission::Receipt {
        unfer_protocol::submission::Receipt::pending(id)
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "unfer-d14-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("could not create the test directory");
    dir
}

fn durable_node(dir: &std::path::Path) -> ConsensusNode {
    let (mut node, _recovery) =
        ConsensusNode::open(Box::new(NoEngine), dir.join("ledger.log")).expect("open a node");
    node.set_mint_authority(MintAuthority::None);
    node.enqueue_job("settle-auction");
    node
}

/// Two workers partitioned from each other, both believing they lead.
///
/// Worker A holds the claim; worker B asks the committed state for the same
/// resource. B is refused because A's claim is live — the round-robin leader in
/// `lease.rs` says nothing about who *may* settle, so "I think it is my turn"
/// cannot become a second claim.
#[test]
fn two_partitioned_workers_cannot_both_lead() {
    let dir = temp_dir("partitioned");
    let mut node = durable_node(&dir);

    let a = node.claim_job("settle-auction", "worker-a").expect("A claims");
    let refused = node.claim_job("settle-auction", "worker-b");
    assert!(refused.is_err(), "B must not be able to hold the same job");
    let message = refused.unwrap_err().message;
    assert!(
        message.contains("RESOURCE_HELD"),
        "the refusal names the reason: {message}"
    );

    // A's fence is still the current one, and B's attempt changed nothing.
    assert_eq!(node.job_fence("settle-auction"), Some(a.fence()));
    assert_eq!(node.jobs().fences().holder("settle-auction"), Some("worker-a"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// A delayed completion from the previous holder.
///
/// A claims, its fire fails, the job is re-queued and B claims it under a newer
/// fence. A's completion arrives afterwards. A may keep computing; its result
/// cannot be committed.
#[test]
fn delayed_old_completion_is_refused() {
    let mut queue = JobQueue::new();
    queue.enqueue("settle-auction");
    let a = queue.claim("settle-auction", "worker-a").unwrap();
    queue.unclaim(&a).unwrap();
    let b = queue.claim("settle-auction", "worker-b").unwrap();
    assert!(a.fence().is_older_than(b.fence()), "B's fence is newer");

    let mut sink = OrderedMockSink::new();
    // B's effect lands.
    queue.settle(&b, &mut sink, "settle:42").expect("B settles");
    assert_eq!(sink.applied_count("settle-auction"), 1);

    // A's late completion is refused by the state and by the sink.
    assert!(queue.mark_fired(&a).is_err(), "stale completion is refused");
    assert_eq!(queue.state("settle-auction"), Some(JobState::Fired));
    let sink_refusal = sink.apply(a.fence(), "settle:42-late");
    assert!(sink_refusal.is_err(), "the sink refuses a stale fence too");
    assert_eq!(sink.applied_count("settle-auction"), 1, "applied once");
}

/// Claim replay: the same claim command and the same settle, twice.
#[test]
fn claim_and_settle_replay_are_refused() {
    let mut queue = JobQueue::new();
    queue.enqueue("job");
    let claim = queue.claim("job", "worker-0").unwrap();
    // A second claim command for the same job finds it not queued.
    assert!(queue.claim("job", "worker-1").is_err());
    queue.mark_fired(&claim).expect("first settle");
    // Replaying the settle is refused, and the job stays fired.
    assert!(queue.mark_fired(&claim).is_err(), "replayed settle refused");
    assert_eq!(queue.state("job"), Some(JobState::Fired));
}

/// Generation overflow: the epoch advances, the fence stays strictly newer, and
/// a space with no successor refuses rather than wrapping.
#[test]
fn generation_overflow_advances_the_epoch() {
    let mut table = FenceTable::new();
    table.install(Fence::new("dep", "res", 7, u64::MAX), Some("w".into()), false);
    let top = table.current("res").unwrap().clone();
    let advanced = table.renew(&top).expect("epoch advances");
    assert_eq!((advanced.epoch, advanced.generation), (8, 1));
    assert!(top.is_older_than(&advanced), "strictly ordered across the reset");

    // The very top of the space has no successor: refused, not wrapped.
    table.install(Fence::new("dep", "res", u64::MAX, u64::MAX), Some("w".into()), false);
    let ceiling = table.current("res").unwrap().clone();
    let refused = table.renew(&ceiling);
    assert!(refused.is_err(), "no successor means no fence");
    assert!(table.admits(&ceiling), "the current fence is unchanged");
}

/// Restart: the fence and the job's state come back from the journal.
///
/// The reopened node does not re-issue the claim (it would be a second claim of
/// the same job), does not re-fire a job that already fired, and refuses the
/// settle of a claim whose fence is no longer current.
#[test]
fn restart_rebuilds_fences_and_job_state() {
    let dir = temp_dir("restart");
    let mut node = durable_node(&dir);
    let claim = node.claim_job("settle-auction", "worker-a").expect("claim");
    drop(node);

    // Reopen: the claim is still live, held by the same worker under the same fence.
    let (mut reopened, _recovery) =
        ConsensusNode::open(Box::new(NoEngine), dir.join("ledger.log")).expect("reopen");
    reopened.set_mint_authority(MintAuthority::None);
    assert_eq!(reopened.job_fence("settle-auction"), Some(claim.fence()));
    assert_eq!(reopened.job_state("settle-auction"), Some(JobState::Claimed));
    assert!(
        reopened.claim_job("settle-auction", "worker-b").is_err(),
        "a restart does not release the claim for someone else"
    );
    // The recovered claim still settles, and then the job is fired across a restart.
    reopened.mark_job_fired(&claim).expect("the current fence settles");
    drop(reopened);

    let (after, _recovery) =
        ConsensusNode::open(Box::new(NoEngine), dir.join("ledger.log")).expect("reopen again");
    assert_eq!(
        after.job_state("settle-auction"),
        Some(JobState::Fired),
        "a fired job is not re-queued by a restart"
    );
    assert!(after.job_fence("settle-auction").is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

/// A release of a newer claim by an older worker.
///
/// A holds fence 1, B takes over under fence 2. A's release arrives late. It must
/// not release B's claim — a stale worker that gives up must not hand the
/// resource to a third party.
#[test]
fn stale_release_of_a_newer_claim_is_refused() {
    let mut queue = JobQueue::new();
    queue.enqueue("job");
    let a = queue.claim("job", "worker-a").unwrap();
    queue.unclaim(&a).unwrap();
    let b = queue.claim("job", "worker-b").unwrap();

    assert!(queue.unclaim(&a).is_err(), "A's late release is refused");
    assert_eq!(
        queue.state("job"),
        Some(JobState::Claimed),
        "B still holds the job"
    );
    let fence = queue.fence("job").expect("a fence");
    assert_eq!(
        (fence.epoch, fence.generation),
        (b.fence().epoch, b.fence().generation),
        "the current fence is still B's"
    );
    queue.mark_fired(&b).expect("B settles");
}

/// The durable side of the claim: one record per transition, and recovery is the
/// last record per resource.
#[test]
fn fence_transitions_are_durable_and_recover_last_wins() {
    let dir = temp_dir("journal");
    let path = dir.join("ledger.log");
    let mut node = durable_node(&dir);
    let claim = node.claim_job("settle-auction", "worker-a").unwrap();
    node.unclaim_job(&claim).unwrap();
    let second = node.claim_job("settle-auction", "worker-b").unwrap();
    drop(node);

    let (_log, recovery) = TransitionLog::open(&path).expect("reopen the log");
    let fences: Vec<&FenceRecord> = recovery
        .records
        .iter()
        .filter_map(|record| match record {
            JournalRecord::Fence(fence) => Some(fence),
            _ => None,
        })
        .collect();
    assert_eq!(fences.len(), 3, "claim, unclaim, claim");
    assert_eq!(fences[0].state, ClaimState::Claimed);
    assert_eq!(fences[1].state, ClaimState::Queued);
    assert_eq!(fences[2].state, ClaimState::Claimed);
    assert_eq!(fences[2].holder.as_deref(), Some("worker-b"));
    assert_eq!(second.fence().generation, 2, "the second claim is newer");

    // Installed in order, the last record wins.
    let mut table = JobQueue::new();
    for fence in fences {
        table.install_fence(
            Fence::new(
                fence.deployment.clone(),
                fence.resource.clone(),
                fence.epoch,
                fence.generation,
            ),
            fence.holder.clone(),
            fence.state,
        );
    }
    assert_eq!(table.state("settle-auction"), Some(JobState::Claimed));
    assert_eq!(
        table.fence("settle-auction").map(|fence| fence.generation),
        Some(2)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The enforcement interface is the sink's own: a sink that implements it
/// rejects a fence it has already passed, without asking the node.
#[test]
fn a_fenced_sink_enforces_order_without_asking_the_node() {
    let mut sink = OrderedMockSink::new();
    let older = Fence::new("dep", "sink:ledger", 1, 3);
    let newer = Fence::new("dep", "sink:ledger", 1, 4);

    sink.apply(&newer, "second").expect("newest applies");
    assert!(
        sink.apply(&older, "first").is_err(),
        "the sink refuses an older fence"
    );
    assert!(
        sink.apply(&newer, "again").is_err(),
        "the same fence twice is not a new effect"
    );
    assert_eq!(sink.applied_count("sink:ledger"), 1);
    assert_eq!(sink.applied[0].3, "second");
}
