//! H4 durable storage backends.
//!
//! **Loro is the preferred store.** Every kernel stream (audit, owner log,
//! actions, config, session) is backed by a single [`LoroDurableStore`]
//! unless an operator selects another backend. JSONL and SQLite implement the
//! same [`DurableStore`] contract for the places where a line-oriented file or
//! a relational table is genuinely a better fit — they are *alternatives*, never
//! mirrors: a given stream lives in exactly one store, so no data is duplicated
//! across backends.
//!
//! A store is constructed at a directory (or an in-memory doc when `None`);
//! [`DurableStore::flush`] is the checkpoint barrier that persists buffered
//! records. Concurrent `flush` calls serialize on an internal drain lock so the
//! on-disk state advances by one frontier at a time.

pub mod jsonl;
pub mod loro;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use jsonl::JsonlDurableStore;
pub use loro::LoroDurableStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteDurableStore;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use unfer_protocol::durable::{DurableError, DurableStore};

/// Which backend a store was built with. Selectable per deployment; Loro is
/// the default. Backends are interchangeable for the same stream semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Default/preferred: a single Loro document with one list per stream.
    Loro,
    /// A line-oriented `.jsonl` file per stream.
    Jsonl,
    /// A SQLite table `(stream, seq, payload)` (feature `sqlite`).
    #[cfg(feature = "sqlite")]
    Sqlite,
}

impl Backend {
    /// Parse a backend name (case-insensitive). `""`/`None` → [`Backend::Loro`].
    pub fn parse(s: Option<&str>) -> Self {
        match s.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("jsonl") => Backend::Jsonl,
            #[cfg(feature = "sqlite")]
            Some("sqlite") => Backend::Sqlite,
            _ => Backend::Loro,
        }
    }

    /// The store's `backend()` label.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Loro => "loro",
            Backend::Jsonl => "jsonl",
            #[cfg(feature = "sqlite")]
            Backend::Sqlite => "sqlite",
        }
    }
}

/// Shared drain lock so concurrent `flush()` calls serialize on one writer.
/// The on-disk frontier advances by exactly one checkpoint at a time.
#[derive(Debug, Default)]
pub(crate) struct DrainLock(pub Mutex<()>);

/// Open a store of the given backend at `dir`. `None` = in-memory only
/// (records survive until the process exits; `flush` still commits the
/// in-memory frontier).
pub fn open_store(
    dir: Option<&Path>,
    backend: Backend,
) -> Result<Box<dyn DurableStore>, DurableError> {
    match backend {
        Backend::Loro => Ok(Box::new(LoroDurableStore::open(dir))),
        Backend::Jsonl => Ok(Box::new(JsonlDurableStore::open(dir)?)),
        #[cfg(feature = "sqlite")]
        Backend::Sqlite => Ok(Box::new(SqliteDurableStore::open(dir)?)),
    }
}

/// The well-known stream names in status order. Single source of truth for
/// the operator surfaces (`uk_durable_status` JSON, the Bevy chip, the TUI
/// dashboard) — a new well-known stream must be added here and in
/// `unfer_protocol::durable::streams` together.
pub const STREAM_NAMES: [&str; 6] = [
    unfer_protocol::durable::streams::AUDIT,
    unfer_protocol::durable::streams::OWNER_LOG,
    unfer_protocol::durable::streams::ACTIONS,
    unfer_protocol::durable::streams::CONFIG,
    unfer_protocol::durable::streams::SESSION,
    unfer_protocol::durable::streams::CERTIFICATES,
];

/// A live status consult: backend, persist counter, per-stream lengths, and
/// the fail-visible corrupt-snapshot recovery report. Stable schema; the
/// RAM-only shape (no store) is `backend = "none"`, every stream `0`, no
/// error — the same shape `uk_durable_status` reports.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DurableStatus {
    pub backend: String,
    pub persist_count: u64,
    pub streams: Vec<(String, u64)>,
    pub snapshot_load_error: Option<String>,
}

/// Consult a store into a fresh [`DurableStatus`]. `None` store = RAM-only
/// (the kernel's no-store shape). Pure read-only: never appends or flushes,
/// so a consulter cannot race the kernel's own writes. Shared by the Bevy
/// status chip and the `mathed_mini` TUI dashboard (single implementation).
pub fn consult_status(store: Option<&dyn DurableStore>) -> DurableStatus {
    match store {
        Some(store) => DurableStatus {
            backend: store.backend().to_string(),
            persist_count: store.persist_count(),
            streams: STREAM_NAMES
                .iter()
                .map(|s| (s.to_string(), store.stream_len(s).unwrap_or(u64::MAX)))
                .collect(),
            snapshot_load_error: store.snapshot_load_error(),
        },
        None => DurableStatus {
            backend: "none".to_string(),
            persist_count: 0,
            streams: STREAM_NAMES.iter().map(|s| (s.to_string(), 0)).collect(),
            snapshot_load_error: None,
        },
    }
}

/// Resolve the durable store's backing directory.
pub(crate) fn snapshot_path(dir: &Path) -> PathBuf {
    dir.join("snapshot.bin")
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::durable::streams;

    /// A unique scratch directory for a test, cleaned up on drop.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "unfer-durable-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The shared round-trip suite every backend must pass.
    ///
    /// Asserts the H4 contract: append-only order, replay-after-flush, and
    /// stream independence. This is the "JSONL and SQLite (and Loro) both pass
    /// the round-trip suite" acceptance gate.
    pub(crate) fn round_trip_suite(store: &dyn DurableStore) {
        // Streams are independent and ordered.
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store
            .append(streams::OWNER_LOG, b"(kernel.audit) hello")
            .unwrap();
        store.append(streams::AUDIT, b"{\"n\":3}").unwrap();

        // Nothing readable before flush is required to be durable, but after
        // flush everything must replay in order.
        store.flush().unwrap();

        let audit = store.replay(streams::AUDIT).unwrap();
        assert_eq!(
            audit,
            vec![
                b"{\"n\":1}".to_vec(),
                b"{\"n\":2}".to_vec(),
                b"{\"n\":3}".to_vec()
            ]
        );

        let owner = store.replay(streams::OWNER_LOG).unwrap();
        assert_eq!(owner, vec![b"(kernel.audit) hello".to_vec()]);

        // A stream with no records replays empty.
        assert!(store.replay(streams::ACTIONS).unwrap().is_empty());
        assert!(store.replay(streams::CONFIG).unwrap().is_empty());

        // Live per-stream lengths (no replay on the Loro backend).
        assert_eq!(store.stream_len(streams::AUDIT).unwrap(), 3);
        assert_eq!(store.stream_len(streams::OWNER_LOG).unwrap(), 1);
        assert_eq!(store.stream_len(streams::ACTIONS).unwrap(), 0);

        // Frontiers are opaque but present after writes.
        let frontier = store.frontier().unwrap();
        assert!(!frontier.is_empty());
    }

    #[test]
    fn loro_round_trip_in_memory() {
        let store = LoroDurableStore::open(None);
        round_trip_suite(&store);
        assert_eq!(store.backend(), "loro");
    }

    #[test]
    fn loro_flush_coalesces_when_clean() {
        let scratch = Scratch::new("coalesce");
        let store = LoroDurableStore::open(Some(&scratch.0));
        assert_eq!(store.persist_count(), 0);

        // A flush with nothing appended is a no-op checkpoint.
        store.flush().unwrap();
        assert_eq!(store.persist_count(), 0);

        // Appending dirties the store; the next flush persists once.
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        assert_eq!(store.persist_count(), 1);

        // A second flush without appends does not rewrite the snapshot.
        store.flush().unwrap();
        assert_eq!(store.persist_count(), 1);

        // A burst of appends coalesces into a single persist.
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.append(streams::AUDIT, b"{\"n\":3}").unwrap();
        store.flush().unwrap();
        assert_eq!(store.persist_count(), 2);

        // Kill-and-resume: every appended record survived.
        let store2 = LoroDurableStore::open(Some(&scratch.0));
        let audit = store2.replay(streams::AUDIT).unwrap();
        assert_eq!(
            audit,
            vec![
                b"{\"n\":1}".to_vec(),
                b"{\"n\":2}".to_vec(),
                b"{\"n\":3}".to_vec()
            ]
        );
        assert_eq!(store2.stream_len(streams::AUDIT).unwrap(), 3);
    }

    #[test]
    fn loro_round_trip_persisted() {
        let scratch = Scratch::new("loro");
        let store = LoroDurableStore::open(Some(&scratch.0));
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        // Kill-and-resume: a fresh store on the same dir reproduces the state.
        let store2 = LoroDurableStore::open(Some(&scratch.0));
        let audit = store2.replay(streams::AUDIT).unwrap();
        assert_eq!(audit, vec![b"{\"n\":1}".to_vec(), b"{\"n\":2}".to_vec()]);
    }

    #[test]
    fn loro_fork_at_frontier_inherits_history() {
        let store = LoroDurableStore::open(None);
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        let fork = store.fork_at(&frontier).unwrap();
        let audit = fork.replay(streams::AUDIT).unwrap();
        assert_eq!(audit, vec![b"{\"n\":1}".to_vec()]);
        // The fork diverges independently.
        fork.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        fork.flush().unwrap();
        assert_eq!(
            store.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()]
        );
    }

    #[test]
    fn loro_fork_of_restarted_parent_is_deterministic() {
        // A Loro fork itself is in-memory, so a “restart” of the fork is not
        // applicable — but the fork-determinism property is: a durable
        // frontier must remain forkable after the parent restarts, and the
        // fork taken then must carry exactly the same history as the fork
        // taken before the restart (same frontier → same history).
        let scratch = Scratch::new("loro-fork-restart");
        let store = LoroDurableStore::open(Some(&scratch.0));
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.append(streams::OWNER_LOG, b"(kernel.audit) one").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        // Post-frontier parent writes must never leak into either fork.
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        let fork1 = store.fork_at(&frontier).unwrap();
        let fork1_audit = fork1.replay(streams::AUDIT).unwrap();
        let fork1_owner = fork1.replay(streams::OWNER_LOG).unwrap();
        assert_eq!(fork1_audit, vec![b"{\"n\":1}".to_vec()]);
        assert_eq!(fork1_owner, vec![b"(kernel.audit) one".to_vec()]);
        drop(fork1);
        drop(store);

        // Restart: a fresh store on the same dir, then fork at the SAME
        // (pre-restart) frontier. The fork must be byte-identical to fork1.
        let store2 = LoroDurableStore::open(Some(&scratch.0));
        let fork2 = store2
            .fork_at(&frontier)
            .expect("a durable frontier must remain forkable after restart");
        assert_eq!(
            fork2.replay(streams::AUDIT).unwrap(),
            fork1_audit,
            "restarted-parent fork must match the pre-restart fork"
        );
        assert_eq!(fork2.replay(streams::OWNER_LOG).unwrap(), fork1_owner);
    }

    #[test]
    fn jsonl_fork_at_frontier_honors_byte_cutoff() {
        let scratch = Scratch::new("jsonl-fork");
        let store = JsonlDurableStore::open(Some(&scratch.0)).unwrap();
        // Write and flush two records, capturing the frontier after the first
        // so the fork must cut before the second.
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        let fork = store.fork_at(&frontier).unwrap();
        // History up to and including the frontier: record n=2 must NOT leak.
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()],
            "fork must truncate to the frontier, not copy the current state"
        );
        // The fork diverges independently; the original is untouched.
        fork.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        fork.flush().unwrap();
        assert_eq!(fork.replay(streams::AUDIT).unwrap().len(), 2);
        assert_eq!(store.replay(streams::AUDIT).unwrap().len(), 2);
    }

    #[test]
    fn jsonl_fork_at_unknown_frontier_fails_closed() {
        let scratch = Scratch::new("jsonl-fork-bad");
        let store = JsonlDurableStore::open(Some(&scratch.0)).unwrap();
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        let err = store.fork_at(b"garbage").unwrap_err();
        assert!(
            matches!(err, DurableError::CorruptFrontier(_)),
            "expected CorruptFrontier, got: {err:?}"
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_fork_at_frontier_honors_write_cutoff() {
        let scratch = Scratch::new("sqlite-fork");
        let store = SqliteDurableStore::open(Some(&scratch.0)).unwrap();
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        let fork = store.fork_at(&frontier).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()],
            "fork must keep only the first frontier_count writes"
        );
        // The fork diverges independently; the original is untouched.
        fork.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        fork.flush().unwrap();
        assert_eq!(fork.replay(streams::AUDIT).unwrap().len(), 2);
        assert_eq!(store.replay(streams::AUDIT).unwrap().len(), 2);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_fork_survives_restart() {
        let scratch = Scratch::new("sqlite-fork-restart");
        let store = SqliteDurableStore::open(Some(&scratch.0)).unwrap();
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        // Post-frontier parent writes must never leak into the fork.
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        let fork = store.fork_at(&frontier).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()]
        );

        // “Restart”: drop the fork and reopen it from its own directory, as a
        // new process would. The WAL-resident committed row must come back
        // through the copy, and the truncated cutoff must be durable.
        drop(fork);
        let fork = SqliteDurableStore::open(Some(&scratch.0.join("fork"))).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()],
            "restarted sqlite fork must hold only the frontier history"
        );

        // The fork's own writes survive its own restart too.
        fork.append(streams::AUDIT, b"{\"n\":4}").unwrap();
        fork.flush().unwrap();
        drop(fork);
        let fork = SqliteDurableStore::open(Some(&scratch.0.join("fork"))).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec(), b"{\"n\":4}".to_vec()]
        );
    }

    #[test]
    fn jsonl_fork_survives_restart() {
        let scratch = Scratch::new("jsonl-fork-restart");
        let store = JsonlDurableStore::open(Some(&scratch.0)).unwrap();
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.append(streams::OWNER_LOG, b"(kernel.audit) fork me").unwrap();
        store.flush().unwrap();
        let frontier = store.frontier().unwrap();
        // Post-frontier parent writes must never leak into the fork.
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();

        let fork = store.fork_at(&frontier).unwrap();
        // The parent keeps writing after the fork; the two diverge.
        store.append(streams::AUDIT, b"{\"n\":3}").unwrap();
        store.flush().unwrap();

        // “Restart”: the fork store is dropped and reopened from its own
        // directory, as a new process would. The fork must still hold exactly
        // the frontier history — the parent's later writes stay behind.
        drop(fork);
        let fork = JsonlDurableStore::open(Some(&scratch.0.join("fork"))).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec()],
            "restarted fork must hold only the frontier history"
        );
        assert_eq!(
            fork.replay(streams::OWNER_LOG).unwrap(),
            vec![b"(kernel.audit) fork me".to_vec()]
        );

        // The fork's own writes survive its own restart too.
        fork.append(streams::AUDIT, b"{\"n\":4}").unwrap();
        fork.flush().unwrap();
        drop(fork);
        let fork = JsonlDurableStore::open(Some(&scratch.0.join("fork"))).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":1}".to_vec(), b"{\"n\":4}".to_vec()]
        );
    }

    #[test]
    fn jsonl_inmemory_stores_are_isolated_and_forkable() {
        // Two RAM-only stores in one process must not share a scratch file:
        // each replays exactly its own records, and forking one never hits
        // the other's data (nor a shared fork target).
        let a = JsonlDurableStore::open(None).unwrap();
        let b = JsonlDurableStore::open(None).unwrap();
        a.append(streams::AUDIT, b"{\"owner\":\"a\"}").unwrap();
        a.flush().unwrap();
        b.append(streams::AUDIT, b"{\"owner\":\"b\"}").unwrap();
        b.flush().unwrap();
        assert_eq!(
            a.replay(streams::AUDIT).unwrap(),
            vec![b"{\"owner\":\"a\"}".to_vec()],
            "store a must not see store b's records"
        );
        assert_eq!(
            b.replay(streams::AUDIT).unwrap(),
            vec![b"{\"owner\":\"b\"}".to_vec()]
        );

        // A fork of an in-memory store carries that store's frontier only.
        let frontier = a.frontier().unwrap();
        let fork = a.fork_at(&frontier).unwrap();
        assert_eq!(
            fork.replay(streams::AUDIT).unwrap(),
            vec![b"{\"owner\":\"a\"}".to_vec()]
        );
        // The fork is an independent branch.
        fork.append(streams::AUDIT, b"{\"n\":99}").unwrap();
        fork.flush().unwrap();
        assert_eq!(a.replay(streams::AUDIT).unwrap().len(), 1);
        assert_eq!(fork.replay(streams::AUDIT).unwrap().len(), 2);
    }

    #[test]
    fn loro_corrupt_snapshot_recovers_visibly() {
        let scratch = Scratch::new("corrupt");
        let dir = &scratch.0;

        // A healthy store leaves a valid snapshot behind.
        let store = LoroDurableStore::open(Some(dir));
        store.append(streams::AUDIT, b"{\"n\":1}").unwrap();
        store.flush().unwrap();
        assert!(store.snapshot_load_error().is_none());
        drop(store);

        // Simulate a torn snapshot: garbage replaces the file.
        let garbage: &[u8] = b"\x00garbage: not a loro snapshot";
        std::fs::write(snapshot_path(dir), garbage).unwrap();

        // Reopening must not panic, must start empty, and must say why.
        let store = LoroDurableStore::open(Some(dir));
        assert!(
            store.snapshot_load_error().is_some(),
            "corrupt snapshot must be reported"
        );
        assert!(store.replay(streams::AUDIT).unwrap().is_empty());

        // The corrupt file was preserved for forensics, byte for byte
        // (never overwritten by a later flush).
        let preserved = std::fs::read(dir.join("snapshot.bin.corrupt")).unwrap();
        assert_eq!(preserved, garbage);
        assert!(!snapshot_path(dir).exists());

        // Recovery continues: the recovered store appends and persists a
        // fresh snapshot on its own (new) frontier.
        store.append(streams::AUDIT, b"{\"n\":2}").unwrap();
        store.flush().unwrap();
        assert!(snapshot_path(dir).exists());
        drop(store);

        // A third open is clean again.
        let store = LoroDurableStore::open(Some(dir));
        assert!(store.snapshot_load_error().is_none());
        assert_eq!(
            store.replay(streams::AUDIT).unwrap(),
            vec![b"{\"n\":2}".to_vec()]
        );
    }

    #[test]
    fn jsonl_torn_final_line_reported() {
        let scratch = Scratch::new("jsonl-torn");
        let dir = &scratch.0;

        // An interrupted append: the final line has no trailing newline.
        std::fs::write(dir.join("audit.jsonl"), b"{\"n\":1}\n{\"n\":2").unwrap();
        std::fs::write(dir.join("owner_log.jsonl"), b"(kernel.audit) clean\n").unwrap();

        let store = JsonlDurableStore::open(Some(dir)).unwrap();
        let err = store.snapshot_load_error().expect("torn line must be reported");
        assert!(
            err.contains("audit") && err.contains("torn final line"),
            "report must name the torn stream: {err}"
        );

        // The intact records still replay; the partial final line is served
        // as-is (a final unterminated line is a complete line to the reader),
        // which is exactly why the flag exists: the operator knows the last
        // record may be truncated.
        let audit = store.replay(streams::AUDIT).unwrap();
        assert_eq!(audit.len(), 2, "both lines replay: {audit:?}");

        // A clean reopen (the operator appends a newline-terminated record and
        // flushes) clears the flag: the backend adds the trailing newline
        // itself, so appending a plain record repairs the file.
        store.append(streams::AUDIT, b"{\"n\":3}").unwrap();
        store.flush().unwrap();
        let store2 = JsonlDurableStore::open(Some(dir)).unwrap();
        assert!(store2.snapshot_load_error().is_none());
    }

    #[test]
    fn jsonl_round_trip() {
        let scratch = Scratch::new("jsonl");
        let store = JsonlDurableStore::open(Some(&scratch.0)).unwrap();
        round_trip_suite(&store);
        assert_eq!(store.backend(), "jsonl");

        // Kill-and-resume reproduces state.
        store.append(streams::AUDIT, b"{\"n\":7}").unwrap();
        store.flush().unwrap();
        let store2 = JsonlDurableStore::open(Some(&scratch.0)).unwrap();
        let audit = store2.replay(streams::AUDIT).unwrap();
        assert!(audit.iter().any(|r| r == b"{\"n\":7}"));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_corrupt_db_reported() {
        let scratch = Scratch::new("sqlite-corrupt");
        let dir = &scratch.0;

        // Seed a multi-page database so a data page exists beyond page 1.
        {
            let store = SqliteDurableStore::open(Some(&dir)).unwrap();
            assert!(store.snapshot_load_error().is_none(), "fresh db must be clean");
            for i in 0..40 {
                store
                    .append(streams::AUDIT, format!("record {i}").as_bytes())
                    .unwrap();
            }
            store.flush().unwrap();
        }

        // Corrupt a data page's STRUCTURE (page 2's type byte at offset 4096;
        // page 1 keeps its valid header so the file still opens and the
        // schema init succeeds — only the integrity probe trips. quick_check
        // validates b-tree structure, not payload bytes, so a structural
        // corruption is what it reliably catches).
        let path = dir.join("store.db");
        let mut bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 4096, "expected a multi-page db, got {}", bytes.len());
        bytes[4096] = 0x00; // invalid b-tree page type
        std::fs::write(&path, &bytes).unwrap();

        // Detectable-but-openable corruption: open succeeds (fail-visible,
        // not fail-closed) and the operator learns the file is damaged.
        let store = SqliteDurableStore::open(Some(&dir)).unwrap();
        assert!(
            store
                .snapshot_load_error()
                .as_deref()
                .unwrap_or_default()
                .contains("quick_check"),
            "corruption must be reported: {:?}",
            store.snapshot_load_error()
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_round_trip() {
        let scratch = Scratch::new("sqlite");
        let store = SqliteDurableStore::open(Some(&scratch.0)).unwrap();
        round_trip_suite(&store);
        assert_eq!(store.backend(), "sqlite");

        // Kill-and-resume reproduces state.
        store.append(streams::AUDIT, b"{\"n\":7}").unwrap();
        store.flush().unwrap();
        let store2 = SqliteDurableStore::open(Some(&scratch.0)).unwrap();
        let audit = store2.replay(streams::AUDIT).unwrap();
        assert!(audit.iter().any(|r| r == b"{\"n\":7}"));
    }

    #[test]
    fn concurrent_flushes_share_one_serialized_drain() {
        let scratch = Scratch::new("drain");
        let store = LoroDurableStore::open(Some(&scratch.0));
        let store = std::sync::Arc::new(store);

        let mut handles = Vec::new();
        for i in 0..8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for j in 0..16 {
                    store
                        .append(
                            streams::AUDIT,
                            format!("{{\"t\":{i},\"j\":{j}}}").as_bytes(),
                        )
                        .unwrap();
                }
                store.flush().unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Every record landed exactly once, in order per writer.
        let audit = store.replay(streams::AUDIT).unwrap();
        assert_eq!(audit.len(), 8 * 16);
        let mut seen = std::collections::HashSet::new();
        for rec in &audit {
            assert!(seen.insert(rec.clone()), "duplicate record: {:?}", rec);
        }
    }
}
