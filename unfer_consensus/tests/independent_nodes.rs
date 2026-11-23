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
    client_json, committed, digest, run, transaction, wait_digest, Deployment,
};

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
    assert_eq!(wait_digest(&fixture.configs[1], 2), first, "nodes 1 and 2 agree");
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
    let (ok, _, stderr) = support::submit(
        &other,
        &fixture.address(0),
        9,
        Path::new(&tx_path),
    );
    assert!(!ok, "a command for deployment 2 must not commit into deployment 1");
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

    // The rejoining replica is behind by one commit. Decide once more and check
    // that it converges on the survivors' full log — not just on the new slot.
    let fourth = fixture.submit(0, 4, &transaction(4));
    assert!(fourth.0, "submit after the restart failed: {}", fourth.2);
    let caught_up = wait_digest(&fixture.configs[0], 4);
    assert_eq!(wait_digest(&fixture.configs[1], 4), caught_up);
    eprintln!("survivors: {caught_up}; rejoined: {}", digest(&fixture.configs[2]));
    assert_eq!(
        wait_digest(&fixture.configs[2], 4),
        caught_up,
        "the restarted node must converge on the same log, not just the new slot"
    );
    assert_eq!(committed(&caught_up), 4);

    fixture.clean(&mut nodes);
}
