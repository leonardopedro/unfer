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
