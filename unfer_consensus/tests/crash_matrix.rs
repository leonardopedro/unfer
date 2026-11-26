//! D13: the abrupt-crash and storage-recovery matrix.
//!
//! Two kinds of evidence, kept apart on purpose:
//!
//! * **Deterministic boundaries** (this file's in-process tests) — the D07
//!   journal fault points make "fail before the write", "fail with half a frame
//!   on disk" and "fail after the frame, before the sync" exact, so each case is
//!   a repeatable statement about the crash protocol. These are process-level
//!   crashes of a node in this process: the memory is dropped, the files stay.
//! * **Real-process kills** — three `unfer_node` processes over TLS, killed with
//!   SIGKILL, restarted on the same state directory. These are process crashes,
//!   not power-loss measurements: no signal is sent that a real disk would not
//!   have flushed, and the suite never claims otherwise.
//!
//! Invariants checked after every recovery:
//!
//! 1. an acknowledged durable operation is still there (the log does not lose a
//!    committed slot);
//! 2. no logical request applies twice (the ledger's supply/identity count is
//!    compared, not just the slot count);
//! 3. a failed domain operation never partially applies (a refused slot leaves
//!    the ledger untouched and the watermark advanced);
//! 4. recovery never substitutes empty state (a corrupt committed record is
//!    refused loudly instead of being replaced by a fresh log).
//!
//! Machine-readable lines for `scripts/distributed-recovery` are prefixed
//! `D13\t<case>\t<outcome>` so the script can save and compare them without
//! parsing prose.

use std::path::{Path, PathBuf};

#[cfg(feature = "network")]
mod support;

use unfer_consensus::engine::{ConsensusEngine, LocalConsensus};
use unfer_consensus::node::ConsensusNode;
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_consensus::storage::{FaultPoint, JournalRecord, StorageError, TransitionLog};
use unfer_protocol::{ConsensusTransaction, IdentityOp, IdentityOpKind};

fn temp_root(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("unfer-d13-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn note(case: &str, outcome: &str) {
    println!("D13\t{case}\t{outcome}");
}

/// A signed identity create for a fresh keypair.
fn create(label: u64) -> ConsensusTransaction {
    let did = Keypair::generate();
    let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
        did: did.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: did.public_key(),
        signature: [0u8; 64],
        seq: label,
        service_endpoint: None,
    });
    sign_transaction(&mut tx, &did);
    tx
}

/// An identity create that every node refuses: the DID does not belong to the
/// key it presents.
fn refused(label: u64) -> ConsensusTransaction {
    let signer = Keypair::generate();
    let other = Keypair::generate();
    let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
        did: other.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: signer.public_key(),
        signature: [0u8; 64],
        seq: label,
        service_endpoint: None,
    });
    sign_transaction(&mut tx, &signer);
    tx
}

struct Case {
    node: ConsensusNode,
    journal: PathBuf,
    /// The decided log lives outside the node (it is the cluster's), so a
    /// restart keeps it: only the node's memory and its journal are the process.
    engine: LocalConsensus,
}

impl Case {
    /// Open a node over a journal and arm `fault` for the next append.
    fn open(dir: &Path, fault: Option<FaultPoint>) -> Self {
        let engine = LocalConsensus::new();
        let journal = dir.join("ledger.log");
        let (mut node, _recovery) =
            ConsensusNode::open(Box::new(engine.clone()), &journal).expect("open the journal");
        if let Some(fault) = fault {
            assert!(
                node.inject_storage_fault(fault),
                "a durable node must have a journal to fail"
            );
        }
        Self {
            node,
            journal,
            engine,
        }
    }

    fn submit(&self, tx: ConsensusTransaction) -> u64 {
        self.node.submit(tx).expect("submit")
    }

    fn deliver(&mut self) -> Result<u64, String> {
        self.node
            .sync_report()
            .map(|report| report.entries)
            .map_err(|diagnostic| format!("{} {}", diagnostic.code, diagnostic.message))
    }

    /// Drop the node — the process's memory is gone, the journal stays — and
    /// open it again, which is the recovery path a restart takes.
    fn restart(self) -> Self {
        let journal = self.journal.clone();
        let engine = self.engine.clone();
        drop(self.node);
        let (node, _recovery) =
            ConsensusNode::open(Box::new(engine.clone()), &journal).expect("reopen");
        Self {
            node,
            journal,
            engine,
        }
    }

    fn state(&self) -> (u64, u64) {
        (
            self.node.applied_seq(),
            self.node.certs().total_supply() + self.node.identity().len() as u64,
        )
    }
}

#[test]
fn a_write_failure_before_the_frame_leaves_nothing_and_applies_once() {
    let dir = temp_root("before-write");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));

    // Arm the next append, then deliver: the ledger is evaluated first, so the
    // node halts with its watermark still at 0.
    assert!(case.node.inject_storage_fault(FaultPoint::BeforeWrite));
    let halted = case.deliver().expect_err("a write failure must halt");
    assert!(halted.contains("UK-5011"), "{halted}");
    assert_eq!(case.node.applied_seq(), 0);
    assert!(case.node.journal_error().is_some());
    note(
        "fail-before-write",
        &format!("applied=0 halted={halted} journal_error=set"),
    );

    // Restart: the journal has no record, so slot 1 is applied exactly once.
    case = case.restart();
    assert_eq!(
        case.node.applied_seq(),
        0,
        "recovery must not invent a slot"
    );
    assert_eq!(case.deliver().unwrap(), 1);
    let (applied, identities) = case.state();
    assert_eq!((applied, identities), (1, 1));
    note("fail-before-write/restart", "applied=1 identities=1");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_torn_frame_is_discarded_and_the_slot_applies_once() {
    let dir = temp_root("half-write");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));
    assert!(case.node.inject_storage_fault(FaultPoint::HalfWrite));
    let halted = case.deliver().expect_err("a torn write must halt");
    assert!(halted.contains("UK-5011"), "{halted}");
    let on_disk = std::fs::metadata(&case.journal)
        .map(|m| m.len())
        .unwrap_or(0);
    assert!(
        on_disk > 0,
        "the torn half must be on disk for this to test anything"
    );

    case = case.restart();
    // The torn tail is discarded by recovery, so nothing is applied twice and
    // the node is usable again.
    assert_eq!(case.node.applied_seq(), 0);
    assert_eq!(case.deliver().unwrap(), 1);
    assert_eq!(case.state(), (1, 1));
    note(
        "fail-half-write",
        &format!("torn_bytes={on_disk} recovered_applied=0 then applied=1"),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_uncertain_sync_is_either_durable_or_replayed_but_never_double_applied() {
    let dir = temp_root("after-write");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));
    assert!(case.node.inject_storage_fault(FaultPoint::AfterWrite));
    let halted = case.deliver().expect_err("an uncertain sync must halt");
    assert!(halted.contains("UK-5011"), "{halted}");

    case = case.restart();
    // Two admissible worlds: the frame reached the disk (recovery sees it) or it
    // did not. Either way the ledger ends with one identity, never two.
    let applied_before = case.node.applied_seq();
    case.deliver()
        .expect("the restarted node must make progress");
    let (applied, identities) = case.state();
    assert_eq!(
        identities, 1,
        "one identity, however often the slot was seen"
    );
    assert!(applied == 1, "the watermark covers the slot exactly once");
    note(
        "fail-after-write",
        &format!(
            "recovered_applied={applied_before} final_applied={applied} identities={identities}"
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupt_committed_record_is_refused_not_replaced_by_empty_state() {
    let dir = temp_root("corrupt");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));
    case.deliver().unwrap();
    let before = case.state();
    let journal = case.journal.clone();
    drop(case);

    // Flip a byte inside the recorded frame's payload: a complete frame with a
    // header that no longer matches its contents.
    let mut bytes = std::fs::read(&journal).unwrap();
    assert!(bytes.len() > 24, "the record must be on disk to corrupt it");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&journal, &bytes).unwrap();

    let error = match ConsensusNode::open(Box::new(LocalConsensus::new()), &journal) {
        Ok(_) => panic!("a corrupt committed record must be refused"),
        Err(error) => error,
    };
    let rendered = error.to_string();
    assert!(
        matches!(error, StorageError::Corrupt { .. }),
        "expected corruption, got {rendered}"
    );
    assert!(
        !rendered.contains("could not open"),
        "the refusal must name the corrupt record, not a missing file: {rendered}"
    );
    note("corrupt-record", &format!("refused={rendered}"));
    assert_eq!(before, (1, 1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unreadable_journal_refuses_to_open_instead_of_starting_empty() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_root("permission");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));
    case.deliver().unwrap();
    assert_eq!(case.state(), (1, 1));
    let journal = case.journal.clone();
    let engine = case.engine.clone();
    drop(case);

    // A mode change *while the node holds its writer does not stop it*: access
    // rights are fixed when the file is opened. That is a measured property of
    // this design, so the case is built at the boundary that does exist — the
    // next open.
    let original = std::fs::metadata(&journal).unwrap().permissions();
    std::fs::set_permissions(&journal, std::fs::Permissions::from_mode(0o444)).unwrap();
    let refused = match ConsensusNode::open(Box::new(engine.clone()), &journal) {
        Ok(_) => {
            std::fs::set_permissions(&journal, original).unwrap();
            note(
                "permission",
                "gap: this user can open a read-only journal for writing",
            );
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        Err(error) => error.to_string(),
    };
    assert!(
        refused.contains("Permission denied") || refused.contains("permission"),
        "the refusal must name the permission failure: {refused}"
    );
    // The outcome is the claim, not this run's directory: a saved fixture must
    // stay comparable across runs, so the path is dropped.
    let reason = refused
        .split("ledger.log: ")
        .nth(1)
        .unwrap_or(&refused)
        .to_string();
    note("permission", &format!("open refused: {reason}"));

    // Given the journal back, the node recovers its prefix instead of starting
    // empty, and the next slot applies exactly once.
    std::fs::set_permissions(&journal, original).unwrap();
    let (mut node, recovery) =
        ConsensusNode::open(Box::new(engine), &journal).expect("the restored journal opens");
    assert_eq!(
        recovery.last_slot,
        Some(1),
        "the prefix is what recovery found"
    );
    assert_eq!(node.applied_seq(), 0, "the watermark replays from the log");
    node.submit(create(2)).expect("submit after restore");
    let applied = node.sync().expect("both slots apply");
    assert_eq!(applied, 2, "the recovered slot and the new one");
    assert_eq!(node.applied_seq(), 2);
    assert_eq!(node.identity().len(), 2);
    note("permission/restored", "applied=2 identities=2");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_conflicting_checkpoint_halts_instead_of_merging_two_histories() {
    let dir = temp_root("conflict");
    let mut case = Case::open(&dir, None);
    case.submit(create(1));
    case.deliver().unwrap();
    let journal = case.journal.clone();
    drop(case);

    // The journal holds slot 1 with command A. A node restarted onto a *different*
    // decided log — command B at slot 1 — must refuse to merge the two: the
    // recorded outcome and the replayed one disagree.
    let other = LocalConsensus::new();
    other.submit(create(99)).unwrap();
    let (mut node, recovery) = ConsensusNode::open(Box::new(other), &journal)
        .expect("open succeeds: the conflict is only visible when replaying");
    assert_eq!(recovery.last_slot, Some(1));
    let error = node
        .sync()
        .expect_err("a mismatched checkpoint must halt the node");
    let rendered = error.to_string();
    assert!(
        rendered.contains("corrupt log") && rendered.contains("slot 1"),
        "the halt must name the conflicting slot: {rendered}"
    );
    assert!(node.halted().is_some(), "the node stays halted");
    note("conflicting-checkpoint", &format!("halted={rendered}"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_refused_operation_is_durable_and_not_retried() {
    let dir = temp_root("refusal");
    let mut case = Case::open(&dir, None);
    case.submit(refused(1));
    case.submit(create(2));
    assert_eq!(case.deliver().unwrap(), 2, "the refusal advances the slot");

    let journal = case.journal.clone();
    let slot_of_refusal = case
        .node
        .refusal(unfer_protocol::submission::Slot(1))
        .cloned();
    assert!(slot_of_refusal.is_some(), "slot 1 is recorded as refused");
    case = case.restart();
    // The refusal is replayed, not re-decided: recovery compares the recorded
    // outcome with the new one and halts on a mismatch. Nothing is applied for
    // the refused slot, and the accepted slot behind it is still there.
    case.deliver()
        .expect("the replay must reproduce both recorded outcomes");
    assert_eq!(case.node.applied_seq(), 2);
    assert_eq!(
        case.state(),
        (2, 1),
        "only the accepted entry changed state"
    );
    let records = std::fs::read(&journal).unwrap();
    assert!(!records.is_empty());
    note(
        "refusal",
        &format!(
            "refused_code={:?} applied=2 identities=1",
            slot_of_refusal.map(|r| r.code)
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_journal_format_is_what_recovery_reads() {
    // A snapshot publication (D11's `ledger.json`) is separate from this
    // journal; this case pins the framing assumption the matrix relies on: a
    // recovered log yields the same records a fresh read of the file does.
    let dir = temp_root("format");
    let journal = dir.join("ledger.log");
    let (mut log, _) = TransitionLog::open(&journal).unwrap();
    log.append(&JournalRecord::Transition(
        unfer_consensus::storage::TransitionRecord {
            slot: 1,
            state_version: 1,
            command_digest: [3u8; 32],
            outcome: unfer_consensus::storage::RecordedOutcome::Applied,
            dedup: None,
        },
    ))
    .unwrap();
    let len = log.len();
    drop(log);
    let (_, recovery) = TransitionLog::open(&journal).unwrap();
    assert_eq!(recovery.records.len(), 1);
    assert_eq!(recovery.last_slot, Some(1));
    assert_eq!(recovery.torn_tail_bytes, 0);
    note(
        "journal-format",
        &format!("bytes={len} records=1 last_slot=1 torn=0"),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Real processes: SIGKILL over the TLS interface
// ---------------------------------------------------------------------------
//
// These are process crashes. A SIGKILL gives the process no chance to run any
// cleanup, but it does not pretend to be a power cut: the page cache still has
// whatever the kernel accepted, so nothing here is evidence about a disk that
// loses its write cache. That distinction is the plan's, not this file's
// convenience.

#[cfg(feature = "network")]
mod processes {
    use std::thread;
    use std::time::Duration;

    use super::note;
    use super::support::{
        Deployment, Node, committed, digest, transaction, wait_at_least, wait_digest,
    };

    fn cluster_digest(deployment: &Deployment) -> String {
        // Node 1 is the reference; every assertion that compares replicas reads
        // all three so a disagreement surfaces rather than being averaged.
        digest(&deployment.configs[0])
    }

    #[test]
    fn an_acknowledged_operation_survives_a_kill_and_a_retry_applies_once() {
        let fixture = Deployment::new("d13-ack");
        let mut nodes = fixture.spawn_all();

        let first = fixture.submit(0, 1, &transaction(1));
        assert!(
            first.0,
            "the acknowledged submission must commit: {}",
            first.2
        );
        let acknowledged = wait_digest(&fixture.configs[0], 1);
        assert_eq!(committed(&acknowledged), 1);
        note("process/acknowledged", &format!("digest={acknowledged}"));

        // SIGKILL node 1 with its reply already sent, restart it: the slot it
        // acknowledged must still be there, on all three replicas.
        nodes[0].kill();
        nodes[0] = fixture.spawn(0);
        wait_at_least(&fixture.configs[0], 1);
        let after = cluster_digest(&fixture);
        assert_eq!(
            committed(&after),
            1,
            "an acknowledged slot must not disappear across a crash"
        );
        assert_eq!(wait_at_least(&fixture.configs[1], 1), after);
        assert_eq!(wait_at_least(&fixture.configs[2], 1), after);
        note(
            "process/acknowledged-after-kill",
            &format!("digest={after}"),
        );

        // Kill the serving node *while* a submission is in flight, at several
        // points in the request's lifetime. Each iteration is one logical
        // request; the retry path is the node the request was sent to, which is
        // the case D06's admission book exists for.
        for (iteration, delay_ms) in [0u64, 2, 5, 10, 20].into_iter().enumerate() {
            let id = 10 + iteration as u64;
            let before = committed(&cluster_digest(&fixture));
            let tx = transaction(id);
            // Move the serving process into a killer thread so the kill lands at
            // a chosen offset from the start of the request. The replacement is
            // started only after the old process is gone: the state directory is
            // locked for the lifetime of the process that owns it.
            let mut victim: Node = nodes.remove(1);
            let killer = thread::spawn(move || {
                thread::sleep(Duration::from_millis(delay_ms));
                victim.kill();
            });
            let attempt = fixture.submit(1, id, &tx);
            killer.join().expect("killer thread");
            nodes.insert(1, fixture.spawn(1));

            let retry = fixture.submit(1, id, &tx);
            assert!(
                retry.0,
                "the retry of request {id} must settle: {}",
                retry.2
            );
            let after = wait_digest(&fixture.configs[0], before + 1);
            let count = committed(&after);
            assert!(
                count <= before + 1,
                "request {id} applied more than once: {before} -> {count} slots (attempt: {}, \
                 retry: {})",
                attempt.1.trim(),
                retry.1.trim()
            );
            assert_eq!(wait_at_least(&fixture.configs[1], count), after);
            assert_eq!(wait_at_least(&fixture.configs[2], count), after);
            note(
                "process/kill-mid-submit",
                &format!(
                    "delay_ms={delay_ms} attempt_ok={} slots={before}->{count}",
                    attempt.0
                ),
            );
        }

        fixture.clean(&mut nodes);
    }

    #[test]
    fn no_coordinated_write_without_quorum() {
        let fixture = Deployment::new("d13-quorum");
        let mut nodes = fixture.spawn_all();
        let first = fixture.submit(0, 1, &transaction(1));
        assert!(first.0, "{}", first.2);
        let base = wait_digest(&fixture.configs[0], 1);

        // Two of three processes die: the survivor cannot coordinate a write.
        nodes[1].kill();
        nodes[2].kill();
        let before = committed(&cluster_digest(&fixture));
        let attempt = fixture.submit(0, 2, &transaction(2));
        assert!(
            !attempt.0,
            "a majority-less replica reported success: {}",
            attempt.1
        );
        let after = committed(&cluster_digest(&fixture));
        assert_eq!(after, before, "no slot may be committed without a quorum");
        note(
            "process/no-quorum",
            &format!("submit_ok=false slots={before}->{after} base={base}"),
        );

        // Heal the quorum, retry the same request identity: it settles once.
        // The wait comes after the retry, not before: a restarted replica whose
        // snapshot lagged the commit has nothing new to apply until the cluster
        // decides again.
        nodes[1] = fixture.spawn(1);
        nodes[2] = fixture.spawn(2);
        let retry = fixture.submit(0, 2, &transaction(2));
        assert!(
            retry.0,
            "the healed cluster must accept the retry: {}",
            retry.2
        );
        let healed = wait_digest(&fixture.configs[0], before + 1);
        assert_eq!(committed(&healed), before + 1, "exactly one new slot");
        assert_eq!(wait_digest(&fixture.configs[1], before + 1), healed);
        assert_eq!(wait_digest(&fixture.configs[2], before + 1), healed);
        note("process/quorum-healed", &format!("digest={healed}"));

        fixture.clean(&mut nodes);
    }

    #[test]
    fn a_slow_replica_catches_up_and_its_journal_is_never_empty() {
        let fixture = Deployment::new("d13-slow");
        let mut nodes = fixture.spawn_all();
        let first = fixture.submit(0, 1, &transaction(1));
        assert!(first.0, "{}", first.2);
        wait_digest(&fixture.configs[0], 1);

        // Node 3 is stopped while two more decisions are taken without it.
        nodes[2].kill();
        for (index, id) in [(0usize, 2u64), (1usize, 3)] {
            let outcome = fixture.submit(index, id, &transaction(id));
            assert!(outcome.0, "{}", outcome.2);
        }
        let survivors = wait_digest(&fixture.configs[0], 3);
        assert_eq!(wait_digest(&fixture.configs[1], 3), survivors);
        // A replica's snapshot is written independently of the commit that
        // filled it, so a process killed right after a commit can be behind the
        // slots it had already applied. That is measured here rather than
        // assumed away: the number is what the snapshot says, not what the
        // replica knew.
        let behind = digest(&fixture.configs[2]);
        note(
            "process/slow-replica-snapshot",
            &format!("replica3={behind} survivors={survivors}"),
        );

        // Restart it onto the same directory. The cluster stays idle — nothing
        // is submitted — and the replica must still be pushed the two decisions
        // it missed: that is what separates a catch-up from a wait.
        nodes[2] = fixture.spawn(2);
        let recovered = digest(&fixture.configs[2]);
        note(
            "process/slow-replica-recovered",
            &format!("replica3={recovered}"),
        );
        let pushed = wait_at_least(&fixture.configs[2], 3);
        assert_eq!(
            pushed, survivors,
            "an idle cluster must push the missed decisions to the rejoined replica"
        );
        note(
            "process/slow-replica-idle-catch-up",
            &format!("replica3={pushed}"),
        );
        // It converges when the cluster next decides, not before: the measured
        // limit D11 names, restated here as part of the recovery matrix.
        let fourth = fixture.submit(0, 4, &transaction(4));
        assert!(fourth.0, "{}", fourth.2);
        let caught_up = wait_at_least(&fixture.configs[0], 4);
        assert_eq!(wait_at_least(&fixture.configs[2], 4), caught_up);
        note(
            "process/slow-replica-caught-up",
            &format!("replica3={}", digest(&fixture.configs[2])),
        );

        fixture.clean(&mut nodes);
    }
}
