//! E7: the SSD spill tier (ENGRAM.md §2.5).
//!
//! An append-only file of engram payloads plus a **sparse in-memory index**
//! mapping each address to its file offset. The sparse index is the whole
//! design: a full index of 100M entries would hold 100M payloads in RAM, which
//! defeats the purpose of spilling. An offset index holds no payload, and a
//! lookup is one seek plus one read rather than a scan.
//!
//! Records are fixed-header, variable-payload:
//!
//! ```text
//!   offset 0 : granularity  u8
//!   offset 1 : dim          u32  little-endian
//!   offset 5 : unf_hash    [u8; 32]
//!   offset 37: embedding    f32 * dim, little-endian
//! ```
//!
//! The granularity is part of the *address*, not a hint (ENGRAM.md §2.3), so it
//! is stored in the record and in the index key. A lookup at granularity `g`
//! must not find a `sentence` entry, and that has to survive the trip to disk.
//!
//! Writes are **buffered** and flushed in batches. Measured before buffering,
//! a 200k-entry ingest with a 20k RAM budget spent 584 ms of which essentially
//! all was 360k individual `seek`/`write` pairs — 191% overhead against a 3%
//! target. Batching turns that into a handful of large appends. A read always
//! flushes first, so a record is never indexed before its bytes are on disk.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::engram::table::Embedding;
use crate::engram::Granularity;

/// Bytes of fixed record header: granularity + dim + unf_hash.
pub const SPILL_HEADER: usize = 1 + 4 + 32;

/// Appends are batched until the buffer reaches this size.
const FLUSH_THRESHOLD: usize = 1 << 20;

/// The on-disk address of an entry: granularity byte followed by `unf_hash`.
///
/// Built as a plain byte array rather than a `(Granularity, [u8; 32])` tuple so
/// the index key is exactly the 33 bytes the header already needs, with no
/// dependence on `Granularity` deriving `Hash`.
pub fn spill_addr(granularity: Granularity, unf_hash: &[u8; 32]) -> [u8; SPILL_HEADER - 4] {
    let mut a = [0u8; SPILL_HEADER - 4];
    a[0] = granularity.code();
    a[1..].copy_from_slice(unf_hash);
    a
}

/// An append-only file of spilled entries with an offset index.
#[derive(Debug)]
pub struct SpillTier {
    file: File,
    /// Logical end of the file, including bytes still sitting in `pending`.
    end: u64,
    /// File offset at which `pending` begins.
    pending_at: u64,
    /// Records appended but not yet written.
    pending: Vec<u8>,
    index: HashMap<[u8; SPILL_HEADER - 4], u64>,
}

impl SpillTier {
    /// Open (creating if absent) a spill file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let end = file.metadata()?.len();
        Ok(Self {
            file,
            end,
            pending_at: end,
            pending: Vec::new(),
            index: HashMap::new(),
        })
    }

    /// How many entries the index knows about.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Bytes the file occupies once buffered records are counted.
    pub fn bytes_on_disk(&self) -> u64 {
        self.end
    }

    /// Write every buffered record.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.file.seek(SeekFrom::Start(self.pending_at))?;
        self.file.write_all(&self.pending)?;
        self.pending.clear();
        self.pending_at = self.end;
        Ok(())
    }

    /// Append `emb` for `(granularity, unf_hash)`.
    ///
    /// A later write of the same address wins, and the index is repointed at the
    /// newer record. Earlier records are left in the file: an append-only log is
    /// not rewritten, so the file grows with rewrites. That is the right trade
    /// for a spill tier whose entries are keyed by a content hash and therefore
    /// rewritten only when a key is re-ingested.
    pub fn put(
        &mut self,
        granularity: Granularity,
        unf_hash: &[u8; 32],
        emb: &Embedding,
    ) -> io::Result<()> {
        let addr = spill_addr(granularity, unf_hash);
        // The index must point at where these bytes WILL be, which is known now:
        // appends are strictly sequential and nothing else writes the file.
        // `end` already counts bytes still sitting in `pending`, so it is the
        // whole answer -- adding `pending.len()` here would double-count every
        // unflushed record and point the index into the middle of a payload.
        let offset = self.end;

        let mut header = [0u8; SPILL_HEADER];
        header[0] = granularity.code();
        header[1..5].copy_from_slice(&(emb.len() as u32).to_le_bytes());
        header[5..].copy_from_slice(unf_hash);
        self.pending.extend_from_slice(&header);
        for v in emb {
            self.pending.extend_from_slice(&v.to_le_bytes());
        }

        self.end = offset + SPILL_HEADER as u64 + (emb.len() as u64) * 4;
        self.index.insert(addr, offset);

        if self.pending.len() >= FLUSH_THRESHOLD {
            self.flush()?;
        }
        Ok(())
    }

    /// Read the entry for `(granularity, unf_hash)`, if the index knows it.
    pub fn get(
        &mut self,
        granularity: Granularity,
        unf_hash: &[u8; 32],
    ) -> io::Result<Option<Embedding>> {
        let addr = spill_addr(granularity, unf_hash);
        let Some(&offset) = self.index.get(&addr) else {
            return Ok(None);
        };
        // Buffered records must reach the file before this seek can find them.
        self.flush()?;

        let mut header = [0u8; SPILL_HEADER];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut header)?;
        if header[0] != granularity.code() || &header[5..] != &unf_hash[..] {
            // The index and the file disagree. That is a corrupt spill, and
            // returning a plausible-looking entry would be worse than saying so.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "spill index points at a record for a different address",
            ));
        }
        let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
        let mut emb = Vec::with_capacity(dim);
        let mut raw = [0u8; 4];
        for _ in 0..dim {
            self.file.read_exact(&mut raw)?;
            emb.push(f32::from_le_bytes(raw));
        }
        Ok(Some(emb))
    }

    /// Whether the index knows this address, without touching the file.
    ///
    /// Index-only on purpose: `contains` should not cost a flush plus a seek,
    /// and it is called on the ingest path where a syscall per entry would
    /// dominate.
    pub fn indexed(&self, granularity: Granularity, unf_hash: &[u8; 32]) -> bool {
        self.index.contains_key(&spill_addr(granularity, unf_hash))
    }

    /// Forget an address, used when an entry is promoted back to the hot tier.
    ///
    /// The record stays on disk — an append-only file is not rewritten — but the
    /// index no longer claims to know it, so the hot copy is authoritative.
    /// Dropping the index entry is what makes "offload then prefetch then
    /// offload again" idempotent instead of growing the index without bound.
    pub fn forget(&mut self, granularity: Granularity, unf_hash: &[u8; 32]) {
        self.index.remove(&spill_addr(granularity, unf_hash));
    }
}
