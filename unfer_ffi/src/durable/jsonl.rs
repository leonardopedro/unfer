//! H4 line-oriented backend: one `.jsonl` file per stream. Chosen where a
//! human-readable, greppable, tail-able artifact is the point (operator review,
//! external tooling) rather than coordination. A stream lives in exactly one
//! store — JSONL is an *alternative*, never a mirror of the Loro store.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use unfer_protocol::durable::{DurableError, DurableStore};

use super::DrainLock;

/// Unique suffix per `JsonlDurableStore` instance, so two RAM-only stores in
/// one process never share a scratch file (see `temp_dir`).
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Line-oriented durable store.
///
/// Layout on disk (inside `dir`):
/// - `<stream>.jsonl` — one JSON record per line, appended; `flush` fsyncs.
///
/// `flush` takes the drain lock and `sync_all`s every open stream file, so a
/// concurrent `flush` never observes a half-synced earlier flush.
#[derive(Debug)]
pub struct JsonlDurableStore {
    dir: Option<PathBuf>,
    /// Open handles per stream (`<stream>` → file).
    files: Mutex<std::collections::HashMap<String, std::fs::File>>,
    drain: DrainLock,
    /// Set at `open` when a stream file ends with a torn final line (a write
    /// interrupted mid-record: the last line has no trailing newline). The
    /// partial record is still served on replay (a final unterminated line is
    /// a complete line to a line reader) — the flag exists so the operator
    /// knows the last record of that stream may be truncated. Fail-visible
    /// rather than fail-silent.
    ///
    /// Only dir-backed stores are scanned: a RAM-only store's scratch files
    /// are private (a fresh unique dir per store) and removed on drop, so no
    /// one ever re-reads them — except via `fork_at`, which opens the fork
    /// with a real dir, so a torn line copied into a fork *is* reported.
    snapshot_error: Option<String>,
    /// Per-store scratch directory for in-memory streams (`dir == None`).
    /// Unique per store so two RAM-only stores in one process cannot collide
    /// on a shared temp file (or a shared fork target). Removed on drop;
    /// forks (siblings under this dir) are the durable artifact and survive.
    temp_dir: Option<PathBuf>,
}

impl JsonlDurableStore {
    /// Open (or create) the store at `dir`. `None` = in-memory only (writes
    /// are buffered to the file map and dropped on exit; `flush` still drains).
    ///
    /// A stream file ending without a trailing newline is reported via
    /// [`Self::snapshot_load_error`] (the store still opens — every line is
    /// served, and the flag tells the operator the final record may be
    /// truncated by an interrupted append).
    pub fn open(dir: Option<&Path>) -> Result<Self, DurableError> {
        if let Some(d) = dir {
            std::fs::create_dir_all(d)
                .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", d.display())))?;
        }
        Ok(Self {
            dir: dir.map(|p| p.to_path_buf()),
            files: Mutex::new(std::collections::HashMap::new()),
            drain: DrainLock::default(),
            snapshot_error: dir.and_then(detect_torn_streams),
            temp_dir: dir.is_none().then(|| {
                std::env::temp_dir().join(format!(
                    "unfer-jsonl-{}-{}",
                    std::process::id(),
                    NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
                ))
            }),
        })
    }

    fn file_for(&self, stream: &str) -> Result<std::fs::File, DurableError> {
        let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = files.get(stream) {
            return f.try_clone().map_err(|e| DurableError::Io(e.to_string()));
        }
        let file = match &self.dir {
            Some(dir) => {
                let path = dir.join(format!("{stream}.jsonl"));
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| DurableError::Io(format!("open {}: {e}", path.display())))?
            }
            None => {
                // In-memory fallback: this store's private scratch file so
                // replay still works and two RAM-only stores cannot collide.
                let td = self
                    .temp_dir
                    .as_ref()
                    .expect("in-memory store has a scratch dir");
                std::fs::create_dir_all(td)
                    .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", td.display())))?;
                let path = td.join(format!("{stream}.jsonl"));
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| DurableError::Io(format!("open {}: {e}", path.display())))?
            }
        };
        files.insert(
            stream.to_string(),
            file.try_clone()
                .map_err(|e| DurableError::Io(e.to_string()))?,
        );
        Ok(file)
    }
}

/// Decode a JSONL frontier into per-stream byte cutoffs.
///
/// Encoding (opaque to callers): `[u32 BE name len][name][u64 BE byte len]`
/// per stream, sorted by stream name for determinism. A fork replays exactly
/// the listed streams, each truncated to its cutoff — history up to and
/// including the frontier, never later records.
fn decode_frontier(frontier: &[u8]) -> Result<Vec<(String, u64)>, DurableError> {
    let mut out = Vec::new();
    let mut rest = frontier;
    while !rest.is_empty() {
        if rest.len() < 4 {
            return Err(DurableError::CorruptFrontier(
                "jsonl frontier: truncated stream-name length".into(),
            ));
        }
        let name_len = u32::from_be_bytes(rest[0..4].try_into().unwrap()) as usize;
        rest = &rest[4..];
        if rest.len() < name_len + 8 {
            return Err(DurableError::CorruptFrontier(
                "jsonl frontier: truncated stream record".into(),
            ));
        }
        let name = String::from_utf8(rest[..name_len].to_vec())
            .map_err(|e| DurableError::CorruptFrontier(format!("jsonl frontier: bad name: {e}")))?;
        let len = u64::from_be_bytes(rest[name_len..name_len + 8].try_into().unwrap());
        out.push((name, len));
        rest = &rest[name_len + 8..];
    }
    Ok(out)
}

/// Scan `dir` for `<stream>.jsonl` files whose last byte is not a newline:
/// a torn final line (an interrupted append). Returns a report of every such
/// stream, or `None` when all files end cleanly (or the dir is empty).
fn detect_torn_streams(dir: &Path) -> Option<String> {
    let mut torn: Vec<String> = Vec::new();
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(stream) = name.strip_suffix(".jsonl") else {
            continue;
        };
        let Ok(meta) = entry.metadata() else { continue };
        if meta.len() == 0 {
            continue;
        }
        let Ok(mut f) = std::fs::File::open(entry.path()) else {
            continue;
        };
        use std::io::{Read, Seek, SeekFrom};
        let mut last = [0u8; 1];
        if f.seek(SeekFrom::End(-1)).is_ok() && f.read_exact(&mut last).is_ok() && last[0] != b'\n'
        {
            torn.push(stream.to_string());
        }
    }
    if torn.is_empty() {
        None
    } else {
        Some(format!(
            "torn final line (interrupted append) in stream(s): {}",
            torn.join(", ")
        ))
    }
}

impl DurableStore for JsonlDurableStore {
    fn snapshot_load_error(&self) -> Option<String> {
        self.snapshot_error.clone()
    }

    fn append(&self, stream: &str, record: &[u8]) -> Result<(), DurableError> {
        let mut file = self.file_for(stream)?;
        file.write_all(record)
            .map_err(|e| DurableError::Io(e.to_string()))?;
        file.write_all(b"\n")
            .map_err(|e| DurableError::Io(e.to_string()))?;
        Ok(())
    }

    fn flush(&self) -> Result<(), DurableError> {
        let _guard = self.drain.0.lock().unwrap_or_else(|e| e.into_inner());
        let files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        for f in files.values() {
            f.sync_all().map_err(|e| DurableError::Io(e.to_string()))?;
        }
        Ok(())
    }

    fn replay(&self, stream: &str) -> Result<Vec<Vec<u8>>, DurableError> {
        let path = match &self.dir {
            Some(dir) => dir.join(format!("{stream}.jsonl")),
            None => self
                .temp_dir
                .as_ref()
                .expect("in-memory store has a scratch dir")
                .join(format!("{stream}.jsonl")),
        };
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = std::fs::File::open(&path)
            .map_err(|e| DurableError::Io(format!("open {}: {e}", path.display())))?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| DurableError::Io(e.to_string()))?;
            if line.is_empty() {
                continue;
            }
            out.push(line.into_bytes());
        }
        Ok(out)
    }

    fn frontier(&self) -> Result<Vec<u8>, DurableError> {
        // JSONL has no version vector; the frontier encodes each stream's
        // current byte length, which a fork truncates to. A single per-stream
        // map (not a sum) so a fork can cut each file at exactly its frontier
        // length. Opaque to callers; see [`decode_frontier`].
        let files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        let mut streams: Vec<(&String, u64)> = files
            .iter()
            .filter_map(|(name, f)| f.metadata().ok().map(|m| (name, m.len())))
            .collect();
        streams.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = Vec::with_capacity(
            streams.len() * (4 + 8) + streams.iter().map(|(n, _)| n.len()).sum::<usize>(),
        );
        for (name, len) in streams {
            out.extend_from_slice(&(name.len() as u32).to_be_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&len.to_be_bytes());
        }
        Ok(out)
    }

    fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
        // A JSONL fork replays exactly the streams recorded in the frontier,
        // each truncated to its frontier byte length — history up to and
        // including the frontier, and nothing later. A stream born after the
        // frontier must not leak into the fork (so: never a blind copy of the
        // current directory).
        let cutoff = decode_frontier(frontier)?;
        let dir = match &self.dir {
            Some(dir) => dir.clone(),
            // A RAM-only store forks into its own scratch dir, so two
            // in-memory stores never aim at the same `fork/` target.
            None => self
                .temp_dir
                .clone()
                .expect("in-memory store has a scratch dir"),
        };
        let fork_dir = dir.join("fork");
        std::fs::create_dir_all(&fork_dir)
            .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", fork_dir.display())))?;
        for (stream, len) in &cutoff {
            let src = dir.join(format!("{stream}.jsonl"));
            let dst = fork_dir.join(format!("{stream}.jsonl"));
            let copied = std::fs::copy(&src, &dst)
                .map_err(|e| DurableError::Io(format!("fork copy {stream}: {e}")))?;
            if copied > *len {
                // Truncate only when shrinking: growing would pad with zeros.
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&dst)
                    .and_then(|f| f.set_len(*len))
                    .map_err(|e| DurableError::Io(format!("fork truncate {stream}: {e}")))?;
            }
        }
        Ok(Box::new(JsonlDurableStore::open(Some(&fork_dir))?))
    }

    fn backend(&self) -> &'static str {
        "jsonl"
    }
}

impl Drop for JsonlDurableStore {
    fn drop(&mut self) {
        // RAM-only scratch files vanish with the store (the open contract
        // says in-memory writes are dropped on exit). Forks live under this
        // dir as `fork/` and are intentionally kept — they are the durable
        // artifact a caller asked for.
        if let Some(td) = &self.temp_dir {
            let files = self.files.lock().unwrap_or_else(|e| e.into_inner());
            for stream in files.keys() {
                let _ = std::fs::remove_file(td.join(format!("{stream}.jsonl")));
            }
        }
    }
}
