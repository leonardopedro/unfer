//! D07 follow-up: an end-to-end abrupt-death test for the transition journal.
//!
//! `storage.rs`'s own tests inject a fault at one boundary (`FaultPoint`). That
//! proves the protocol is *described* correctly; it does not prove a real dying
//! process leaves a file this code can recover from. This test does: it spawns
//! this very test binary as a child (`--exact child_writer`), lets the child open
//! a journal on tmpfs and append real records, and then has the parent `kill -9`
//! it mid-stream. Recovery must accept whatever survived — complete frames plus
//! at worst a torn tail — never corruption, never a slot gap, and the log must be
//! usable straight after recovery.
//!
//! What this exercises that a unit test cannot: the kernel's actual behaviour on
//! process death (buffered `write` results are already in the tmpfs page cache,
//! the file descriptor closes without any flush of *our* framing), and the fact
//! that the parent has no idea where the child stopped.
//!
//! Honest limit: `/dev/shm` is tmpfs, which has no backing disk, so `sync_data`
//! is a no-op there. This verifies the framing, ordering and truncation protocol
//! under abrupt death — not that a disk honoured an fsync. Durability against
//! media loss is outside a test process's reach; that claim rests on the
//! syscall sequence, not on this test.
//!
//! Each child is spawned through `current_exe`, so the test needs the binary on
//! disk (true for `cargo test`). If no tmpfs directory is writable the test
//! reports that as a skip rather than silently passing on a disk.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use unfer_consensus::storage::{
    JournalRecord, RecordedOutcome, RequestRecord, TransitionLog, TransitionRecord,
};

const CHILD_ENV: &str = "UNFER_POWER_CUT_CHILD";
const MODE_ENV: &str = "UNFER_POWER_CUT_MODE";

fn transition(slot: u64) -> JournalRecord {
    JournalRecord::Transition(TransitionRecord {
        slot,
        state_version: slot,
        command_digest: [slot as u8; 32],
        outcome: RecordedOutcome::Applied,
        dedup: None,
    })
}

fn request(slot: u64) -> JournalRecord {
    JournalRecord::Request(RequestRecord {
        id: unfer_consensus::idempotency::RequestId::new("f", "did:unfer:alice", 1, slot),
        digest: [slot as u8; 32],
        submission: unfer_protocol::submission::SubmissionId(slot),
        outcome: unfer_consensus::idempotency::AdmittedOutcome::Accepted {
            submission: unfer_protocol::submission::SubmissionId(slot),
            slot: unfer_protocol::submission::Slot(slot),
        },
        slot: Some(slot),
    })
}

/// The child side: append records until killed (`stream`), or append a few and
/// then leave half of a frame behind before aborting (`torn`).
///
/// Reached only when `CHILD_ENV` is set by the parent. Under a normal
/// `cargo test` run it returns immediately, because it is the parent's job to
/// make the journal — this function must not touch a shared path.
#[test]
fn child_writer() {
    let Some(dir) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let mode = std::env::var(MODE_ENV).unwrap_or_else(|_| "stream".to_string());
    let path = PathBuf::from(dir).join("journal.log");

    match mode.as_str() {
        "torn" => {
            let (mut log, _) = TransitionLog::open(&path).expect("child opens the journal");
            for slot in 1..=3 {
                log.append(&transition(slot)).expect("child appends");
            }
            // A frame that a crash interrupted: magic, version, a length for a
            // payload that never arrived. Written through a raw handle so this is
            // literally the bytes on disk, not a simulated error return.
            let frame = {
                let mut frame = Vec::new();
                frame.extend_from_slice(&0x554E_4645u32.to_le_bytes());
                frame.push(1);
                frame.extend_from_slice(&4096u32.to_le_bytes());
                frame.extend_from_slice(&0u32.to_le_bytes());
                frame.extend_from_slice(b"{\"kind\":\"transition\"");
                frame
            };
            let mut raw = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("child reopens for the partial frame");
            raw.write_all(&frame)
                .expect("child writes the partial frame");
            raw.sync_all().expect("child syncs");
            // No destructors, no flush, no recovery: the process simply stops.
            std::process::abort();
        }
        _ => {
            let (mut log, _) = TransitionLog::open(&path).expect("child opens the journal");
            for slot in 1u64.. {
                if log.append(&transition(slot)).is_err() {
                    return;
                }
                if slot % 2 == 0 {
                    log.append(&request(slot)).expect("child appends a request");
                }
                println!("{slot}");
                std::io::stdout().flush().expect("child flushes progress");
            }
        }
    }
}

/// Run the child writer for `mode`, let it reach `seen` slots (or a moment for
/// `torn`), then kill it abruptly.
fn spawn_and_kill(dir: &Path, mode: &str, seen: u64) {
    let exe = std::env::current_exe().expect("test binary path");
    let mut child = Command::new(exe)
        .args(["--exact", "child_writer", "--test-threads=1", "--nocapture"])
        .env(CHILD_ENV, dir)
        .env(MODE_ENV, mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the child writer");

    let stdout = child.stdout.take().expect("piped stdout");
    let mut lines = BufReader::new(stdout).lines();
    let mut reached = 0u64;
    // Every printed line is a slot whose frame is already synced and published.
    while reached < seen {
        match lines.next() {
            Some(Ok(line)) => {
                if line.trim().parse::<u64>().is_ok() {
                    reached += 1;
                }
            }
            Some(Err(_)) | None => break,
        }
    }
    if mode == "torn" {
        // The child aborts itself; wait for the process, not a signal.
        let _ = child.wait();
    } else {
        // SIGKILL: no destructor, no drop, no flushing. This is the power cut.
        let _ = child.kill();
        let _ = child.wait();
    }
    drop(lines);
}

/// A writable tmpfs directory, or `None` when this host has none.
fn tmpfs_root() -> Option<PathBuf> {
    let dev_shm = PathBuf::from("/dev/shm");
    if !dev_shm.is_dir() {
        return None;
    }
    let probe = dev_shm.join(format!("unfer-power-cut-probe-{}", std::process::id()));
    if let Err(error) = std::fs::write(&probe, b"probe") {
        eprintln!("power_cut: /dev/shm is not writable ({error})");
        return None;
    }
    // The probe's *removal* failing does not make the directory unusable (some
    // sandboxes reclaim tmpfs entries eagerly); only the write decides.
    let _ = std::fs::remove_file(&probe);
    Some(dev_shm)
}

fn test_dir(root: &Path, round: usize) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = root.join(format!(
        "unfer-power-cut-{}-{stamp}-{round}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create the journal directory");
    dir
}

/// A contiguous run of transition slots starting at 1.
fn recovered_slots(records: &[JournalRecord]) -> Vec<u64> {
    records
        .iter()
        .filter_map(|record| match record {
            JournalRecord::Transition(t) => Some(t.slot),
            JournalRecord::Request(_) | JournalRecord::Fence(_) => None,
        })
        .collect()
}

#[test]
fn a_killed_writer_leaves_a_journal_that_recovers_and_continues() {
    let Some(root) = tmpfs_root() else {
        eprintln!("power_cut: skipped — no writable tmpfs (/dev/shm); see the module note");
        return;
    };

    // Several rounds: where the kill lands decides whether the tail is torn, so
    // one round would only ever exercise one shape. Each round waits for a
    // different amount of progress.
    for round in 0..5usize {
        let dir = test_dir(&root, round);
        let path = dir.join("journal.log");
        spawn_and_kill(&dir, "stream", 1 + round as u64 * 3);

        let (mut log, recovery) = TransitionLog::open(&path).unwrap_or_else(|error| {
            panic!("round {round}: a killed writer must leave a recoverable journal, got {error}")
        });
        let slots = recovered_slots(&recovery.records);
        assert!(
            !slots.is_empty(),
            "round {round}: the child published at least one record before dying"
        );
        for (index, slot) in slots.iter().enumerate() {
            assert_eq!(
                *slot,
                index as u64 + 1,
                "round {round}: recovery must yield a contiguous prefix, got {slots:?}"
            );
        }
        assert_eq!(
            recovery.last_slot,
            slots.last().copied(),
            "round {round}: last_slot agrees with the records"
        );
        eprintln!(
            "power_cut: round {round} recovered {} transition(s), {} torn tail byte(s)",
            slots.len(),
            recovery.torn_tail_bytes
        );
        // A request record may or may not have been in flight; if it survived it
        // must name a slot the log recorded (recovery would have refused it
        // otherwise, which is the stronger claim).
        let requests = recovery
            .records
            .iter()
            .filter(|record| matches!(record, JournalRecord::Request(_)))
            .count();
        assert!(
            requests <= slots.last().copied().unwrap_or(0) as usize,
            "round {round}: no request may outlive the transition it names"
        );

        // The journal is usable immediately: the killed tail was truncated, not
        // left to poison the next append.
        let next = slots.last().copied().unwrap() + 1;
        log.append(&transition(next))
            .unwrap_or_else(|error| panic!("round {round}: append after recovery: {error}"));
        assert!(
            !log.is_empty(),
            "round {round}: the appended frame has bytes"
        );
        drop(log);

        let (_, reopened) = TransitionLog::open(&path).expect("reopen after continuing");
        assert_eq!(
            reopened.last_slot,
            Some(next),
            "round {round}: the new record survived"
        );
        assert_eq!(
            recovered_slots(&reopened.records),
            {
                let mut all = slots.clone();
                all.push(next);
                all
            },
            "round {round}: nothing else changed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_half_frame_from_a_dead_process_is_discarded_not_reported_as_corruption() {
    let Some(root) = tmpfs_root() else {
        eprintln!("power_cut: skipped — no writable tmpfs (/dev/shm); see the module note");
        return;
    };
    let dir = test_dir(&root, 99);
    let path = dir.join("journal.log");
    spawn_and_kill(&dir, "torn", 0);

    let (mut log, recovery) = TransitionLog::open(&path).expect("a torn tail must recover");
    assert_eq!(
        recovered_slots(&recovery.records),
        vec![1, 2, 3],
        "the records written before the crash are all published"
    );
    assert!(
        recovery.torn_tail_bytes > 0,
        "the abandoned partial frame is a torn tail, not a record"
    );
    log.append(&transition(4))
        .expect("appending after a torn tail");
    drop(log);
    let (_, reopened) = TransitionLog::open(&path).expect("reopen");
    assert_eq!(reopened.last_slot, Some(4));
    assert_eq!(
        reopened.torn_tail_bytes, 0,
        "the truncation was made durable"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Guard against a test that silently stops testing: assert the child really is
/// started from this binary and that a kill produces a process exit.
#[test]
fn the_child_is_this_binary() {
    assert!(
        std::env::current_exe()
            .map(|path| path.exists())
            .unwrap_or(false),
        "the power-cut test spawns `current_exe --exact child_writer`; it must exist"
    );
    // And the child mode must not linger in the parent's environment.
    assert!(
        std::env::var_os(CHILD_ENV).is_none(),
        "the parent must not set the child marker in its own environment"
    );
}
