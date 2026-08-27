//! The preferred H4 store: a single Loro document with one `LoroList` per
//! stream. Records are appended as JSON strings; `flush()` commits the doc and
//! persists a snapshot to disk atomically (write-tmp-then-rename). Frontier is
//! the Loro version-vector encoding, so a second process can detect an
//! interrupted flush and fork at a known frontier.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use loro::{ExportMode, Frontiers, LoroDoc, LoroList};
use unfer_protocol::durable::{DurableError, DurableStore};

use super::{DrainLock, snapshot_path};

/// H4 default/preferred backend.
///
/// Layout on disk (inside `dir`):
/// - `snapshot.bin` — Loro snapshot (full state + history). Written atomically
///   on `flush` (tmp file + rename), so a crash never leaves a torn snapshot.
///
/// With `None` dir the store is in-memory only (still a real Loro doc; the
/// semantics of `append`/`replay`/`frontier` are identical, just not persisted
/// across a process exit).
#[derive(Debug)]
pub struct LoroDurableStore {
    doc: LoroDoc,
    dir: Option<std::path::PathBuf>,
    drain: DrainLock,
    /// Whether an `append` happened since the last `flush` checkpoint.
    /// A clean flush (nothing new) skips the snapshot write entirely —
    /// the coalescer: bursts of appends share one persist.
    dirty: Mutex<bool>,
    /// Completed snapshot writes (a live-status metric and the
    /// coalescer's observable effect).
    persists: AtomicU64,
    /// Set at `open` when the on-disk snapshot existed but could not be
    /// imported. The corrupt file is moved aside to `snapshot.bin.corrupt`
    /// (so the next `flush` cannot overwrite the evidence) and the store
    /// starts empty; this field records that recovery happened. Fail-visible
    /// rather than fail-silent: a torn snapshot would otherwise be silently
    /// replaced by an empty one on the first flush.
    snapshot_error: Mutex<Option<String>>,
}

impl LoroDurableStore {
    /// Open (or create) the store at `dir`. `None` = in-memory only.
    ///
    /// If `dir` holds an existing `snapshot.bin` that fails to import (a
    /// torn or corrupt snapshot), the file is preserved as
    /// `snapshot.bin.corrupt`, the store starts empty, and
    /// [`Self::snapshot_load_error`] reports why — open never panics and
    /// never silently discards state.
    pub fn open(dir: Option<&Path>) -> Self {
        let doc = LoroDoc::new();
        let mut snapshot_error = None;
        if let Some(d) = dir {
            let path = snapshot_path(d);
            if let Ok(bytes) = std::fs::read(&path)
                && let Err(e) = doc.import(&bytes)
            {
                let corrupt = d.join("snapshot.bin.corrupt");
                let moved = std::fs::rename(&path, &corrupt);
                snapshot_error = Some(format!(
                    "snapshot import failed ({}); {}",
                    e,
                    match moved {
                        Ok(()) => "preserved as snapshot.bin.corrupt".to_string(),
                        Err(mv) => format!("could not preserve it: {mv}"),
                    }
                ));
            }
        }
        Self {
            doc,
            dir: dir.map(Path::to_path_buf),
            drain: DrainLock::default(),
            dirty: Mutex::new(false),
            persists: AtomicU64::new(0),
            snapshot_error: Mutex::new(snapshot_error),
        }
    }

    /// The Loro document backing this store (exposed for tests/coordination).
    pub fn doc(&self) -> &LoroDoc {
        &self.doc
    }

    /// How many snapshot writes have completed since open. A `flush`
    /// with no new appends does not increment this.
    pub fn persist_count(&self) -> u64 {
        self.persists.load(Ordering::Relaxed)
    }

    fn list(&self, stream: &str) -> LoroList {
        self.doc.get_list(stream)
    }

    /// Persist the current state to the snapshot file atomically.
    fn persist(&self) -> Result<(), DurableError> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        let snapshot = self
            .doc
            .export(ExportMode::Snapshot)
            .map_err(|e| DurableError::Encode(format!("loro snapshot: {e}")))?;
        std::fs::create_dir_all(dir)
            .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", dir.display())))?;
        let path = snapshot_path(dir);
        let tmp = dir.join("snapshot.bin.tmp");
        std::fs::write(&tmp, &snapshot)
            .map_err(|e| DurableError::Io(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            DurableError::Io(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            ))
        })?;
        Ok(())
    }
}

impl DurableStore for LoroDurableStore {
    fn append(&self, stream: &str, record: &[u8]) -> Result<(), DurableError> {
        let text = std::str::from_utf8(record)
            .map_err(|e| DurableError::Encode(format!("record not utf-8: {e}")))?;
        self.list(stream)
            .push(text)
            .map_err(|e| DurableError::Io(format!("loro append: {e}")))?;
        *self.dirty.lock().unwrap_or_else(|e| e.into_inner()) = true;
        // Auto-commit keeps the in-memory frontier current; durability is
        // established by `flush`.
        Ok(())
    }

    fn stream_len(&self, stream: &str) -> Result<u64, DurableError> {
        // Cheap: the Loro list knows its own length; no replay needed.
        Ok(self.list(stream).len() as u64)
    }

    fn persist_count(&self) -> u64 {
        self.persists.load(Ordering::Relaxed)
    }

    fn snapshot_load_error(&self) -> Option<String> {
        self.snapshot_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn flush(&self) -> Result<(), DurableError> {
        let _guard = self.drain.0.lock().unwrap_or_else(|e| e.into_inner());
        // Coalesce: a checkpoint with nothing new since the last persist
        // skips the snapshot write entirely (still serialized on the
        // drain, so the on-disk frontier never moves backwards).
        let mut dirty = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
        if !*dirty {
            return Ok(());
        }
        *dirty = false;
        drop(dirty);
        self.doc.commit();
        self.persist()?;
        self.persists.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, DurableError> {
        let list = self.list(stream);
        let mut out = Vec::with_capacity(list.len());
        list.for_each(|v| {
            if let loro::ValueOrContainer::Value(lv) = v
                && let Ok(arc) = <Arc<String>>::try_from(lv)
            {
                out.push(arc.as_bytes().to_vec());
            }
        });
        Ok(out)
    }

    fn frontier(&self) -> Result<Vec<u8>, DurableError> {
        let frontiers = self.doc.state_frontiers();
        Ok(frontiers.encode())
    }

    fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
        let frontiers = Frontiers::decode(frontier)
            .map_err(|e| DurableError::CorruptFrontier(format!("decode: {e}")))?;
        let fork = self
            .doc
            .fork_at(&frontiers)
            .map_err(|e| DurableError::CorruptFrontier(format!("fork_at: {e}")))?;
        // The fork is a new independent branch; keep it in-memory unless a dir
        // is later attached. Streams inherit all history up to the frontier.
        Ok(Box::new(LoroDurableStore {
            doc: fork,
            dir: None,
            drain: DrainLock::default(),
            dirty: Mutex::new(false),
            persists: AtomicU64::new(0),
            snapshot_error: Mutex::new(None),
        }))
    }

    fn backend(&self) -> &'static str {
        "loro"
    }
}
