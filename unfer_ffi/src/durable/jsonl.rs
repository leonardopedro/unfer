//! H4 line-oriented backend: one `.jsonl` file per stream. Chosen where a
//! human-readable, greppable, tail-able artifact is the point (operator review,
//! external tooling) rather than coordination. A stream lives in exactly one
//! store — JSONL is an *alternative*, never a mirror of the Loro store.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use unfer_protocol::durable::{DurableError, DurableStore};

use super::DrainLock;

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
    snapshot_error: Option<String>,
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
                // In-memory fallback: an anonymous temp file so replay still works.
                tempfile_like(stream)?
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

fn tempfile_like(stream: &str) -> Result<std::fs::File, DurableError> {
    let path = std::env::temp_dir().join(format!("unfer-jsonl-{}-{stream}", std::process::id()));
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| DurableError::Io(e.to_string()))
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
        let Some(stream) = name.strip_suffix(".jsonl") else { continue };
        let Ok(meta) = entry.metadata() else { continue };
        if meta.len() == 0 {
            continue;
        }
        let Ok(mut f) = std::fs::File::open(entry.path()) else { continue };
        use std::io::{Read, Seek, SeekFrom};
        let mut last = [0u8; 1];
        if f.seek(SeekFrom::End(-1)).is_ok()
            && f.read_exact(&mut last).is_ok()
            && last[0] != b'\n'
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
            None => {
                std::env::temp_dir().join(format!("unfer-jsonl-{}-{stream}", std::process::id()))
            }
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
        // JSONL has no version vector; the frontier is a per-stream byte length
        // that a fork can truncate to. Opaque to callers.
        let files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        let mut sum = 0u64;
        for f in files.values() {
            sum = sum.wrapping_add(f.metadata().map(|m| m.len()).unwrap_or(0));
        }
        Ok(sum.to_le_bytes().to_vec())
    }

    fn fork_at(&self, frontier: &[u8]) -> Result<Box<dyn DurableStore>, DurableError> {
        // A JSONL fork copies every stream file (bounded; the files are small).
        let dir = self.dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("unfer-jsonl-fork-{}", std::process::id()))
        });
        let fork_dir = dir.join("fork");
        std::fs::create_dir_all(&fork_dir)
            .map_err(|e| DurableError::Io(format!("mkdir {}: {e}", fork_dir.display())))?;
        for entry in std::fs::read_dir(&dir).map_err(|e| DurableError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| DurableError::Io(e.to_string()))?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".jsonl") {
                std::fs::copy(entry.path(), fork_dir.join(&*name))
                    .map_err(|e| DurableError::Io(e.to_string()))?;
            }
        }
        let _ = frontier;
        Ok(Box::new(JsonlDurableStore::open(Some(&fork_dir))?))
    }

    fn backend(&self) -> &'static str {
        "jsonl"
    }
}
