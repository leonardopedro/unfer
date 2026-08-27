//! Durable storage contract (H4): an append-only, stream-addressed log whose
//! reads the operator/agent surface never treats as RAM-only.
//!
//! The kernel's audit trail, owner log, resolved posture/config, queued
//! approval work, and session event log are all *events in a stream*. The
//! in-memory rings ([`crate::unfer_ffi`] `handles`) are a read-through cache
//! in front of this store; [`DurableStore::flush`] is the checkpoint barrier
//! that makes buffered writes durable. A [`DurableStore`] is `Send + Sync` so
//! a single store can back the whole process.
//!
//! ```
//! use unfer_protocol::durable::{DurableStore, DurableError};
//! use std::sync::Arc;
//! # fn demo(s: &dyn DurableStore) -> Result<(), DurableError> {
//! s.append("audit", br#"{"symbol":"uk_version"}"#)?;
//! s.flush()?;
//! let records = s.replay("audit")?;
//! assert_eq!(records.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! Stream names are stable ABI constants: `audit`, `owner_log`, `actions`,
//! `config`, `session`, `certificates`. Backends (Loro, JSONL, SQLite) keep
//! the same stream semantics so a store can be swapped under a running
//! kernel.

use std::fmt;

/// Well-known stream names. New streams must be added here (additive only).
pub mod streams {
    /// Kernel-global audit trail of `uk_*` calls.
    pub const AUDIT: &str = "audit";
    /// Dot-separated owner-log component lines.
    pub const OWNER_LOG: &str = "owner_log";
    /// Side-effecting action queue (`ActionRecord`s, one record per transition).
    pub const ACTIONS: &str = "actions";
    /// Resolved posture/config (vetted markers, soft config, session bindings).
    pub const CONFIG: &str = "config";
    /// Session event logs (H3 `SessionEvent` records).
    pub const SESSION: &str = "session";
    /// Emitted verification certificates (mass-gap / Ritz bound records from
    /// the T6 pipeline, one `certificate-issued` line per certificate).
    pub const CERTIFICATES: &str = "certificates";
}

/// Stream-addressed durable log.
///
/// Implementations MUST be:
/// - **append-only**: `append` never rewrites or removes an existing record;
/// - **fail-closed**: an `Err` means the record was NOT durably committed and
///   the caller must not treat it as such;
/// - **checkpointed**: after `flush()` returns `Ok`, every prior `append` is
///   durable and will be returned by a future `replay` in the same order.
pub trait DurableStore: fmt::Debug + Send + Sync {
    /// Append one record to a stream. Order within a stream is preserved.
    /// An `Err` is fail-closed: the record was not committed.
    fn append(&self, stream: &str, record: &[u8]) -> Result<(), DurableError>;

    /// Number of records currently in a stream (live-status support).
    ///
    /// The default implementation replays the stream and counts; backends
    /// with a cheap length (e.g. a Loro list) override it.
    fn stream_len(&self, stream: &str) -> Result<u64, DurableError> {
        Ok(self.replay(stream)?.len() as u64)
    }

    /// Make every prior `append` durable (the checkpoint barrier). An `Err`
    /// means the checkpoint failed and prior records may or may not be durable;
    /// callers must fail closed (refuse to serve a probability/condition or
    /// fire a side effect).
    fn flush(&self) -> Result<(), DurableError>;

    /// How many checkpoint writes completed since open (live-status support).
    ///
    /// The default reports 0; backends with a coalescing flush (e.g. Loro's
    /// dirty-tracked snapshot) override it so a host can observe that bursts
    /// of appends share one persist.
    fn persist_count(&self) -> u64 {
        0
    }

    /// Replay every record of a stream, in append order.
    fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, DurableError>;

    /// The store's current frontier (Loro version vector, opaque bytes).
    /// Used to detect an interrupted flush and to coordinate forks.
    fn frontier(&self) -> Result<Vec<u8>, DurableError>;

    /// Fork the store at a frontier, producing an independent branch that
    /// contains all history up to (and including) that frontier.
    fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError>;

    /// Backend label for diagnostics (`"loro" | "jsonl" | "sqlite"`).
    fn backend(&self) -> &'static str;

    /// Why the on-disk snapshot could not be imported at `open`, if the store
    /// recovered from a corrupt/torn snapshot. `None` = clean open, fresh
    /// store, or an in-memory store. Default: `None` — backends that cannot
    /// detect corruption report nothing; fail-visible backends (Loro) override
    /// this so operators learn the store started empty and why.
    fn snapshot_load_error(&self) -> Option<String> {
        None
    }
}

/// A durable-store failure. Every variant maps to a UK-#### diagnostic by the
/// caller (`checkpoint()` at the FFI layer maps to the fail-closed return).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DurableError {
    #[error("durable store io: {0}")]
    Io(String),
    #[error("durable store encode: {0}")]
    Encode(String),
    #[error("durable store decode: {0}")]
    Decode(String),
    #[error("durable store unsupported operation: {0}")]
    Unsupported(String),
    #[error("durable store corrupt frontier: {0}")]
    CorruptFrontier(String),
}

impl DurableError {
    /// Whether this error indicates a *crash window*: the process died between
    /// an append and its checkpoint, so the outcome of a side-effecting call is
    /// unknown. Callers surface this as the new UK-1010 `UNKNOWN_OUTCOME`.
    pub fn is_unknown_outcome(&self) -> bool {
        matches!(self, DurableError::CorruptFrontier(_))
    }
}

/// Replay records of a stream into a map keyed by the last record's `id`
/// field (used for action-queue recovery). Records that fail to decode are
/// skipped — a half-written tail after a crash is expected.
pub fn last_record_per_id(
    stream_records: &[Vec<u8>],
    id_field: &str,
) -> Vec<(String, serde_json::Value)> {
    let mut out: Vec<(String, serde_json::Value)> = Vec::new();
    for rec in stream_records {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(rec) else {
            continue;
        };
        let Some(id) = v.get(id_field).and_then(|x| x.as_str()) else {
            continue;
        };
        if let Some(existing) = out.iter_mut().find(|(k, _)| k == id) {
            existing.1 = v;
        } else {
            out.push((id.to_string(), v));
        }
    }
    out
}