//! D11: three independent node processes, no shared state.
//!
//! Each process gets its own state directory, its own ledger and its own TLS
//! identity, and is started from a validated JSON config. The test drives them
//! only through their published interface: submit a value over TLS to one
//! member, then compare the digest of the committed log each node applied. A
//! node that read another node's storage could not be distinguished by the
//! digest alone — so the isolation is also asserted directly: each directory
//! holds only its own replica's files, and a second process pointed at a live
//! node's directory is refused.
//!
//! Steps (PLAN D11):
//! 1. startup is per process, from validated config;
//! 2. three processes, disjoint state directories, submissions through
//!    different nodes, identical digests;
//! 3. a node is killed, the survivors commit without it, and it catches up when
//!    restarted.

#![cfg(feature = "network")]

mod support;

use std::path::Path;

use support::{
    Deployment, client_json, committed, digest, job_report, run, transaction, wait_digest,
};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{ConsensusTransaction, JobFence, JobOp, JobOpKind};

fn job_op(signer: &Keypair, deployment: &str, job: &str, kind: JobOpKind) -> ConsensusTransaction {
    let mut tx = ConsensusTransaction::JobOp(JobOp {
        did: signer.did(),
        deployment: deployment.to_owned(),
        job: job.to_owned(),
        kind,
        signature: [0u8; 64],
    });
    sign_transaction(&mut tx, signer);
    tx
}

/// D14 acceptance over the transport: two worker processes submit a claim for
/// the same job at the same time. The committed order serializes them, every
/// replica derives the same fence from that order, and the second claim is a
/// durable refusal rather than a second holder. The fence generation is the
/// measurement: `1` means the refused claim never issued a fence.
#[test]
fn two_processes_contend_for_one_job_and_only_one_claim_commits() {
    let fixture = Deployment::new("d14-job-contention");
    let mut nodes = fixture.spawn_all();

    let registrar = Keypair::generate();
    let worker_a = Keypair::generate();
    let worker_b = Keypair::generate();
    let job = "settle-auction";

    // Registration is itself a consensus transition, so every replica knows the
    // job before either claim is applied.
    let registered = fixture.submit(0, 1, &job_op(&registrar, "1", job, JobOpKind::Register));
    assert!(registered.0, "register failed: {}", registered.2);
    wait_digest(&fixture.configs[0], 1);

    let claim_a = job_op(
        &worker_a,
        "1",
        job,
        JobOpKind::Claim {
            holder: "worker-a".to_owned(),
        },
    );
    let claim_b = job_op(
        &worker_b,
        "1",
        job,
        JobOpKind::Claim {
            holder: "worker-b".to_owned(),
        },
    );

    // Both claims are proposed before either is known to have committed: the
    // race is decided by the consensus order, not by the test's sequencing.
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| fixture.submit(0, 2, &claim_a));
        let second = scope.spawn(|| fixture.submit(1, 3, &claim_b));
        (first.join().unwrap(), second.join().unwrap())
    });
    assert!(first.0, "first claim submission failed: {}", first.2);
    assert!(second.0, "second claim submission failed: {}", second.2);

    let winner = wait_digest(&fixture.configs[0], 3);
    assert_eq!(
        wait_digest(&fixture.configs[1], 3),
        winner,
        "nodes 1 and 2 agree"
    );
    assert_eq!(wait_digest(&fixture.configs[2], 3), winner, "node 3 agrees");

    let reports = fixture
        .configs
        .iter()
        .map(|config| job_report(config, job))
        .collect::<Vec<_>>();
    assert_eq!(reports[1], reports[0], "node 2 reports the same claim");
    assert_eq!(reports[2], reports[0], "node 3 reports the same claim");
    let report = &reports[0];
    assert!(report.contains("\"state\":\"claimed\""), "{report}");
    assert!(
        report.contains("\"holder\":\"worker-a\"") || report.contains("\"holder\":\"worker-b\""),
        "exactly one worker holds the job: {report}"
    );
    assert!(
        report.contains("\"generation\":1"),
        "the losing claim was refused instead of issuing a second fence: {report}"
    );
    assert!(
        !report.contains("worker-a\"") || !report.contains("worker-b\""),
        "the report names one holder, not two: {report}"
    );

    // A fire naming a position that was never issued is refused and changes
    // nothing; the current fence then fires the job. Both outcomes are
    // committed slots, which is the point: the log decides, the state filters.
    let stale = job_op(
        &worker_b,
        "1",
        job,
        JobOpKind::Fire {
            expected: JobFence {
                epoch: 1,
                generation: 2,
            },
        },
    );
    let stale_fire = fixture.submit(2, 4, &stale);
    assert!(
        stale_fire.0,
        "stale fire submission failed: {}",
        stale_fire.2
    );
    wait_digest(&fixture.configs[0], 4);
    let still_claimed = job_report(&fixture.configs[0], job);
    assert!(
        still_claimed.contains("\"state\":\"claimed\"")
            && still_claimed.contains("\"generation\":1"),
        "a fire for a position that was never issued is refused: {still_claimed}"
    );

    let winner_key = if report.contains("worker-a") {
        &worker_a
    } else {
        &worker_b
    };
    let fire = job_op(
        winner_key,
        "1",
        job,
        JobOpKind::Fire {
            expected: JobFence {
                epoch: 1,
                generation: 1,
            },
        },
    );
    let fired = fixture.submit(0, 5, &fire);
    assert!(fired.0, "current fire submission failed: {}", fired.2);
    let final_digest = wait_digest(&fixture.configs[0], 5);
    assert_eq!(wait_digest(&fixture.configs[1], 5), final_digest);
    assert_eq!(wait_digest(&fixture.configs[2], 5), final_digest);
    let fired_report = job_report(&fixture.configs[0], job);
    assert!(
        fired_report.contains("\"state\":\"fired\""),
        "the current fence fires the job: {fired_report}"
    );

    fixture.clean(&mut nodes);
}

#[test]
fn three_processes_commit_identically_and_one_catches_up_after_a_kill() {
    let fixture = Deployment::new("d11-independent");
    let root = &fixture.root;

    // ---- Step 1: three processes, three disjoint state directories.
    let mut nodes = fixture.spawn_all();

    // ---- Step 2: submit through different nodes; everyone applies.
    let mint = fixture.submit(0, 1, &transaction(1));
    assert!(mint.0, "submit through node 1 failed: {}", mint.2);
    assert!(
        mint.1.contains("\"outcome\":\"committed\"") || mint.1.contains("\"outcome\":\"accepted\""),
        "unexpected submit outcome: {}",
        mint.1
    );
    let transfer = fixture.submit(1, 2, &transaction(2));
    assert!(transfer.0, "submit through node 2 failed: {}", transfer.2);

    let first = wait_digest(&fixture.configs[0], 2);
    assert_eq!(
        wait_digest(&fixture.configs[1], 2),
        first,
        "nodes 1 and 2 agree"
    );
    assert_eq!(wait_digest(&fixture.configs[2], 2), first, "node 3 agrees");

    // ---- Isolation: each process wrote only its own directory.
    for (index, dir) in fixture.dirs.iter().enumerate() {
        let own = dir.join(format!("node-{}", index + 1));
        assert!(own.is_dir(), "{} owns {}", dir.display(), own.display());
        assert!(
            dir.join("ledger.json").is_file(),
            "each node has its own ledger"
        );
        for other in 1..=3 {
            if other != index + 1 {
                assert!(
                    !dir.join(format!("node-{other}")).exists(),
                    "{} must not hold replica {other}'s state",
                    dir.display()
                );
            }
        }
    }
    // A second process pointed at a live node's directory is refused, so no two
    // nodes can share one application store even by accident.
    let (ok, _, stderr) = run(&[
        "serve",
        "--config",
        fixture.configs[0].to_str().unwrap(),
        "--ready",
        root.join("second-ready.json").to_str().unwrap(),
    ]);
    assert!(!ok, "a second process on a live state dir must fail");
    assert!(
        stderr.contains("locked by another node process"),
        "unexpected refusal: {stderr}"
    );

    // ---- A command for another deployment is refused by name.
    let other = root.join("other-deployment.json");
    std::fs::write(
        &other,
        client_json("2", &"07".repeat(16), &fixture.identities, &fixture.ports),
    )
    .unwrap();
    let tx_path = root.join("tx-1.json");
    let (ok, _, stderr) = support::submit(&other, &fixture.address(0), 9, Path::new(&tx_path));
    assert!(
        !ok,
        "a command for deployment 2 must not commit into deployment 1"
    );
    assert!(
        !stderr.is_empty(),
        "the refusal must say why it was refused"
    );

    // ---- Step 3a: stop node 3, commit without it, survivors keep the payload.
    nodes[2].kill();
    let third = fixture.submit(1, 3, &transaction(3));
    assert!(third.0, "submit with node 3 down failed: {}", third.2);

    let survivors = wait_digest(&fixture.configs[0], 3);
    assert_eq!(wait_digest(&fixture.configs[1], 3), survivors);
    assert!(
        digest(&fixture.configs[2]).starts_with("2:"),
        "the stopped process must not have applied the third commit"
    );

    // ---- Step 3b: restart node 3 on the same directory, address and identity.
    let restarted = fixture.spawn(2);
    nodes[2] = restarted;

    // The rejoining replica is behind by one commit, and the cluster is idle:
    // nothing new is decided here. Its peers push what it missed, so it catches
    // up on the strength of the log alone (`CATCH_UP_INTERVAL` in `net.rs`).
    let caught_up = wait_digest(&fixture.configs[0], 3);
    let replayed = wait_digest(&fixture.configs[2], 3);
    assert_eq!(
        replayed, caught_up,
        "an idle cluster must push the missed decision, not wait for a new one"
    );
    eprintln!("caught up while idle: {replayed}");

    // A new decision still reaches every replica, including the rejoined one.
    let fourth = fixture.submit(0, 4, &transaction(4));
    assert!(fourth.0, "submit after the restart failed: {}", fourth.2);
    let fourth_digest = wait_digest(&fixture.configs[0], 4);
    assert_eq!(wait_digest(&fixture.configs[1], 4), fourth_digest);
    eprintln!(
        "survivors: {fourth_digest}; rejoined: {}",
        digest(&fixture.configs[2])
    );
    assert_eq!(
        wait_digest(&fixture.configs[2], 4),
        fourth_digest,
        "the restarted node must converge on the same log, not just the new slot"
    );
    assert_eq!(committed(&fourth_digest), 4);

    fixture.clean(&mut nodes);
}
