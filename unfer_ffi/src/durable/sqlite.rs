//! H4 SQLite backend (feature `sqlite`): a `(stream, seq, payload)` table.
//! Chosen where a relational, queryable artifact is the point (aggregation,
//! joins, long-term retention) rather than coordination. A stream lives in
//! exactly one store — SQLite is an *alternative*, never a mirror of the Loro
//! store. `flush` is a single transaction commit under the drain lock, so
//! concurrent flushes serialize into one durable frontier.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::Connection;
use unfer_protocol::durable::{DurableError, DurableStore};

use super::DrainLock;

/// Relational durable store.
///
/// Schema (inside `dir`):
/// - `store.db` — table `records(stream TEXT, seq INTEGER, payload BLOB,
///   PRIMARY KEY (stream, seq))`.
///
/// `append` batches into an open transaction; `flush` commits it (the
/// checkpoint). A crash before `flush` rolls the batch back (WAL). `replay`
/// reads `seq`-ordered payloads.
#[derive(Debug)]
pub struct SqliteDurableStore {
    conn: Mutex<Connection>,
    drain: DrainLock,
    _dir: Option<PathBuf>,
    /// Set at `open` when `PRAGMA quick_check` finds the database damaged
    /// (torn pages after a crash) yet the file still opens. The store serves
    /// whatever survives; the operator learns the file is not trustworthy.
    /// A file that cannot be opened at all stays fail-closed (`open` errors).
    snapshot_error: Option<String>,
}

impl SqliteDurableStore {
    /// Open (or create) the store at `dir`. `None` = in-memory SQLite.
    ///
    /// Detectable-but-openable corruption (a damaged b-tree page) is
    /// reported via [`Self::snapshot_load_error`]; an unopenable file is a
    /// hard error (fail-closed — SQLite cannot start empty over a garbage
    /// file the way a snapshot store can).
    pub fn open(dir: Option<&Path>) -> Result<Self, DurableError> {
        let path = match dir {
            Some(d) => {
                std::fs::create_dir_all(d)
                    .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", d.display())))?;
                d.join("store.db")
            }
            None => PathBuf::from(":memory:"),
        };
        let conn = Connection::open(&path)
            .map_err(|e| DurableError::Io(format!("sqlite open {}: {e}", path.display())))?;
        // Fail-visible integrity probe BEFORE the init writes: a damaged but
        // openable database is reported, not silently served. "ok" is the
        // clean answer; anything else is a corruption report.
        let snapshot_error = match conn.query_row("PRAGMA quick_check", [], |r| {
            r.get::<_, String>(0)
        }) {
            Ok(report) if report.trim() == "ok" => None,
            Ok(report) => Some(format!("sqlite quick_check: {report}")),
            Err(e) => Some(format!("sqlite quick_check failed: {e}")),
        };
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS records (
                 stream TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 payload BLOB NOT NULL,
                 PRIMARY KEY (stream, seq)
             );
             BEGIN;",
        )
        .map_err(|e| DurableError::Io(format!("sqlite init: {e}")))?;
        Ok(Self {
            conn: Mutex::new(conn),
            drain: DrainLock::default(),
            _dir: dir.map(|p| p.to_path_buf()),
            snapshot_error,
        })
    }

    fn next_seq(&self, stream: &str) -> Result<i64, DurableError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let seq: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM records WHERE stream = ?1",
                [stream],
                |r| r.get(0),
            )
            .map_err(|e| DurableError::Io(format!("sqlite max(seq): {e}")))?;
        Ok(seq)
    }
}

impl DurableStore for SqliteDurableStore {
    fn snapshot_load_error(&self) -> Option<String> {
        self.snapshot_error.clone()
    }

    fn append(&self, stream: &str, record: &[u8]) -> Result<(), DurableError> {
        let seq = self.next_seq(stream)?;
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO records (stream, seq, payload) VALUES (?1, ?2, ?3)",
            rusqlite::params![stream, seq, record],
        )
        .map_err(|e| DurableError::Io(format!("sqlite insert: {e}")))?;
        Ok(())
    }

    fn flush(&self) -> Result<(), DurableError> {
        let _guard = self.drain.0.lock().unwrap_or_else(|e| e.into_inner());
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute_batch("COMMIT; BEGIN;")
            .map_err(|e| DurableError::Io(format!("sqlite commit: {e}")))
    }

    fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, DurableError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn
            .prepare("SELECT payload FROM records WHERE stream = ?1 ORDER BY seq ASC")
            .map_err(|e| DurableError::Io(format!("sqlite prepare: {e}")))?;
        let rows = stmt
            .query_map([stream], |r| r.get::<_, Vec<u8>>(0))
            .map_err(|e| DurableError::Io(format!("sqlite query: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| DurableError::Io(format!("sqlite row: {e}")))?);
        }
        Ok(out)
    }

    fn frontier(&self) -> Result<Vec<u8>, DurableError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0))
            .map_err(|e| DurableError::Io(format!("sqlite count: {e}")))?;
        Ok(n.to_le_bytes().to_vec())
    }

    fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
        // A SQLite fork copies the whole database file (bounded store).
        let Some(dir) = &self._dir else {
            return Err(DurableError::Unsupported(
                "fork of an in-memory sqlite store".into(),
            ));
        };
        let fork_dir = dir.join("fork");
        std::fs::create_dir_all(&fork_dir)
            .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", fork_dir.display())))?;
        std::fs::copy(dir.join("store.db"), fork_dir.join("store.db"))
            .map_err(|e| DurableError::Io(format!("sqlite copy: {e}")))?;
        let _ = frontier;
        Ok(Box::new(SqliteDurableStore::open(Some(&fork_dir))?))
    }

    fn backend(&self) -> &'static str {
        "sqlite"
    }
}
