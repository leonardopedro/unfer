//! D12: the deterministic fault scheduler (plan step 3 and its acceptance).
//!
//! Four questions, each answered by a measured number rather than a claim:
//!
//! 1. does the same trace produce the same outputs and state, twice?
//! 2. does a bounded exhaustive search over the fixed domain find a violation?
//! 3. does a seeded random walk past the search bound find one?
//! 4. is each of the four guards (dedup, rejection, restart, revoke) witnessed by
//!    a schedule that minimizes, and does the minimized schedule still show it?
//!
//! The traces are written to `$UNFER_D12_TRACES` (default: a directory beside
//! the test's temporary root) so the evidence log can name the exact schedules.

use std::path::{Path, PathBuf};

use unfer_consensus::sim::{self, Action, Guard, Sim};

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "unfer-d12-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("could not create the schedule root");
    root
}

fn trace_dir() -> PathBuf {
    std::env::var_os("UNFER_D12_TRACES")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("unfer-d12-traces"))
}

fn save(name: &str, body: &str) -> PathBuf {
    let dir = trace_dir();
    std::fs::create_dir_all(&dir).expect("could not create the trace directory");
    let path = dir.join(name);
    std::fs::write(&path, body).expect("could not write the trace");
    path
}

/// Every replica runs, delivers everything, and the domain decides two entries:
/// the authorized mint applies, the unauthorized one is refused in its slot.
fn basic_domain(seed: u64, root: &Path) -> Sim {
    let mut sim = Sim::new(seed, root, 3, 2);
    sim.step(Action::Admit {
        client: 0,
        replica: 0,
        request: sim::Request::AuthorizedMint,
    })
    .unwrap();
    sim.step(Action::Admit {
        client: 1,
        replica: 0,
        request: sim::Request::UnauthorizedMint,
    })
    .unwrap();
    for replica in 0..3 {
        sim.step(Action::Deliver { replica }).unwrap();
    }
    sim
}

#[test]
fn the_same_trace_produces_the_same_state() {
    let root = temp_root("replay");
    let script = [
        Action::Admit {
            client: 0,
            replica: 0,
            request: sim::Request::AuthorizedMint,
        },
        Action::Deliver { replica: 0 },
        Action::Retry {
            client: 0,
            replica: 1,
        },
        Action::Deliver { replica: 1 },
        Action::Crash { replica: 2 },
    ];

    let run = |seed: u64, tag: &str| {
        let mut sim = Sim::new(seed, root.join(tag), 3, 2);
        for action in script {
            sim.step(action).unwrap();
        }
        (sim.fingerprint(), sim.trace().render(), sim.violations.clone())
    };

    let first = run(7, "a");
    let second = run(7, "b");
    assert_eq!(first.0, second.0, "same trace must produce the same state");
    assert_eq!(first.1, second.1, "same trace must produce the same choices");
    assert!(first.2.is_empty(), "no invariant may be violated: {:?}", first.2);

    // A different seed is only allowed to change what the *client* chose; this
    // script fixes every action, so the state must not move at all. That is the
    // point of separating the classes: a seed-dependence here would mean an
    // unregistered nondeterminism site.
    let other_seed = run(11, "c");
    assert_eq!(
        first.0, other_seed.0,
        "a fixed action list must not depend on the seed"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_domain_applies_and_refuses_as_specified() {
    let root = temp_root("domain");
    let sim = basic_domain(1, &root);
    assert!(sim.violations.is_empty(), "{:?}", sim.violations);
    assert_eq!(sim.cluster.len(), 2);
    for replica in 0..3 {
        assert_eq!(sim.applied(replica), 2, "replica {replica} applied the prefix");
        assert!(sim.halted(replica).is_none());
        assert_eq!(
            sim.refusals(replica),
            vec![(2, unfer_protocol::Code::CERT_MINT_NOT_AUTHORIZED)],
            "replica {replica} refuses the unauthorized mint in its own slot"
        );
        assert_eq!(sim.supply(replica), 100);
    }
    // All three replicas hold the same state, digest for digest.
    assert_eq!(sim.digest(0), sim.digest(1));
    assert_eq!(sim.digest(1), sim.digest(2));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_bounded_exhaustive_search_finds_no_violation() {
    let root = temp_root("explore");
    let alphabet = sim::fixed_alphabet();
    let started = std::time::Instant::now();
    let report = sim::explore(19, &root, &alphabet, 4);
    let elapsed = started.elapsed();
    println!(
        "bounded search: {} elapsed_ms={}",
        report.render(),
        elapsed.as_millis()
    );
    assert!(
        report.counterexample.is_none(),
        "a bounded search found a violation: {}",
        report.render()
    );
    assert!(report.leaves > 0, "the search executed no leaf schedule");
    assert!(report.prefixes > report.leaves, "the search never branched");
    save(
        "bounded-search.txt",
        &format!("{}\n", report.render()),
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The same domain one level deeper. It is `#[ignore]`d because it is a long
/// run, not because it is optional evidence: the recorded numbers in
/// `EXECUTION.md` come from exactly this test, and it is run whenever the bounds
/// are reported.
///
///     cargo test -p unfer_consensus --features network --test distributed_schedules \
///         -- --ignored a_deep_search
#[test]
#[ignore = "minutes of wall clock: run it when the D12 bounds are reported"]
fn a_deep_search_to_depth_five_finds_no_violation() {
    let root = temp_root("explore-deep");
    let alphabet = sim::fixed_alphabet();
    // Depth 5 is the recorded bound; `UNFER_D12_DEPTH` exists so a deeper run can
    // be measured instead of guessed at, and it names itself in the trace file.
    let depth: usize = std::env::var("UNFER_D12_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let started = std::time::Instant::now();
    let report = sim::explore(19, &root, &alphabet, depth);
    let elapsed = started.elapsed();
    println!(
        "deep search: {} elapsed_ms={}",
        report.render(),
        elapsed.as_millis()
    );
    assert!(
        report.counterexample.is_none(),
        "the deep search found a violation: {}",
        report.render()
    );
    assert!(report.leaves > 0, "the deep search executed no leaf schedule");
    save(
        &format!("bounded-search-depth{depth}.txt"),
        &format!("{}\n", report.render()),
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn seeded_random_walks_find_no_violation() {
    let root = temp_root("walk");
    let alphabet = sim::fixed_alphabet();
    let mut fingerprints = Vec::new();
    for seed in 0..32u64 {
        let mut sim = Sim::new(seed, root.join(format!("walk-{seed}")), 3, 2);
        sim.walk(&alphabet, 12);
        assert!(
            sim.violations.is_empty(),
            "seed {seed} violated an invariant: {:?}",
            sim.violations
        );
        fingerprints.push(sim.fingerprint());
        let _ = std::fs::remove_dir_all(sim.root());
    }
    // Two different seeds may well reach the same state, but all 32 reaching one
    // state would mean the walk is not choosing anything: the degenerate run the
    // plan forbids as the *only* evidence.
    let distinct: std::collections::HashSet<_> = fingerprints.iter().collect();
    println!(
        "seeded walks: 32 runs of 12 steps, {} distinct final state(s)",
        distinct.len()
    );
    assert!(distinct.len() > 1, "the walk produced no distinct states");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn every_guard_is_witnessed_by_a_minimal_schedule() {
    let root = temp_root("witness");
    let mut report = String::new();
    for (guard, expected) in [
        (Guard::Dedup, 3usize),
        (Guard::Rejection, 2),
        (Guard::Restart, 5),
        (Guard::Revoke, 6),
    ] {
        let witness = sim::witness(guard, 23, &root);
        println!("{}", witness.render());
        report.push_str(&witness.render());
        report.push('\n');
        assert!(
            witness.minimized <= witness.original,
            "{guard:?} minimization grew the schedule"
        );
        assert!(
            witness.minimized > 0,
            "{guard:?} minimized to nothing, which witnesses nothing"
        );
        // The minimized schedule is the smallest one this harness found, but the
        // bound is a search result: compare it to the hand-written script rather
        // than asserting an absolute minimum the search does not compute.
        assert!(
            witness.minimized <= expected,
            "{guard:?} minimized to {} actions, more than the {expected} the search found",
            witness.minimized
        );
        assert!(
            !witness.fingerprint.is_empty(),
            "{guard:?} produced no state fingerprint"
        );
        save(
            &format!("witness-{:?}.trace", guard).to_lowercase(),
            &format!("{}\n", witness.render()),
        );
    }
    save("witnesses.txt", &report);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_duplicated_grant_applies_once() {
    // The naive design (no content key) would show supply 200 here: two entries,
    // two applications. The production ledger applies the second as a settled
    // no-op, which is what the digest comparison across replicas sees.
    let root = temp_root("dedup");
    let mut sim = Sim::new(5, &root, 2, 2);
    sim.step(Action::Admit {
        client: 0,
        replica: 0,
        request: sim::Request::AuthorizedMint,
    })
    .unwrap();
    sim.step(Action::Retry {
        client: 0,
        replica: 1,
    })
    .unwrap();
    assert_eq!(sim.cluster.len(), 2, "both admissions reached the log");
    sim.step(Action::Deliver { replica: 0 }).unwrap();
    sim.step(Action::Deliver { replica: 1 }).unwrap();
    for replica in 0..2 {
        assert_eq!(sim.applied(replica), 2);
        assert_eq!(sim.supply(replica), 100, "one mint, one application");
        assert!(sim.halted(replica).is_none());
    }
    assert_eq!(sim.digest(0), sim.digest(1));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_storage_failure_halts_instead_of_skipping() {
    // The refused-because-broken case: the journal cannot take slot 1, so the
    // node must stop at slot 0 rather than apply and forget. This is the
    // distinction between a domain refusal (advance) and a node failure (halt),
    // and it is what makes the rejection witness above meaningful.
    let root = temp_root("halt");
    let mut sim = Sim::new(3, &root, 2, 2);
    sim.step(Action::Admit {
        client: 0,
        replica: 0,
        request: sim::Request::AuthorizedMint,
    })
    .unwrap();
    sim.step(Action::Fault { replica: 0 }).unwrap();
    let effect = sim.step(Action::Deliver { replica: 0 }).unwrap();
    assert!(
        effect.halted.as_deref().unwrap_or_default().contains("UK-5011"),
        "expected a storage halt, got {:?}",
        effect
    );
    assert_eq!(sim.applied(0), 0, "a halted node advanced no watermark");
    assert!(sim.halted(0).is_some());
    // The ledger was evaluated before the journal refused the record, so the
    // in-memory supply moved while the watermark did not. That is exactly why a
    // halted node is not a state anyone may read: nothing was acknowledged, and
    // `journal_error()` is set. The restore below proves the durable path never
    // double-counts it.
    println!("after the halted apply: applied={} supply={}", sim.applied(0), sim.supply(0));
    assert!(sim.violations.is_empty(), "{:?}", sim.violations);

    // Restoring the process does not magically skip the slot: the journal is
    // intact, so the same slot is applied once the write can succeed.
    sim.step(Action::Crash { replica: 0 }).unwrap();
    sim.step(Action::Restore { replica: 0 }).unwrap();
    sim.step(Action::Deliver { replica: 0 }).unwrap();
    assert_eq!(sim.applied(0), 1);
    assert_eq!(sim.supply(0), 100);
    let _ = std::fs::remove_dir_all(&root);
}

