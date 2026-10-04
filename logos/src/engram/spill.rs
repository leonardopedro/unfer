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

use crate::engram::Granularity;
use crate::engram::table::Embedding;

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
    ///
    /// The index is rebuilt by walking the existing records. Without this a
    /// reopened spill reported `len() == 0` and answered `None` for every key
    /// that was physically on disk — the file was opened append-only and
    /// designed for reopen-and-continue, but nothing read it back.
    ///
    /// A trailing partial record (a crash mid-append) truncates the scan there
    /// rather than failing: the bytes are unusable either way, and refusing to
    /// open would turn a recoverable spill into an unusable table.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let file_len = file.metadata()?.len();
        let mut index = HashMap::new();
        let mut scanned = 0u64;

        let mut header = [0u8; SPILL_HEADER];
        while scanned + SPILL_HEADER as u64 <= file_len {
            file.seek(SeekFrom::Start(scanned))?;
            if file.read_exact(&mut header).is_err() {
                break;
            }
            let granularity = match Granularity::from_code(header[0]) {
                Some(g) => g,
                None => break,
            };
            let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as u64;
            let body = dim * 4;
            let next = scanned + SPILL_HEADER as u64 + body;
            if next > file_len {
                // Truncated final record — the crash case.
                break;
            }
            let mut unf_hash = [0u8; 32];
            unf_hash.copy_from_slice(&header[5..]);
            // A later record for the same address wins, matching `put`.
            index.insert(spill_addr(granularity, &unf_hash), scanned);
            scanned = next;
        }

        if scanned < file_len {
            // Drop the unusable tail so the next append starts at a record
            // boundary instead of mid-payload.
            file.set_len(scanned)?;
        }

        Ok(Self {
            file,
            end: scanned,
            pending_at: scanned,
            pending: Vec::new(),
            index,
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
        if header[0] != granularity.code() || header[5..] != unf_hash[..] {
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

impl Drop for SpillTier {
    /// Flush buffered records on the way out.
    ///
    /// Without this, up to `FLUSH_THRESHOLD` (1 MiB) of records were discarded
    /// when the table was dropped: `flush` was `pub` but `TieredTable.spill` is
    /// a private field with no accessor and no flush of its own, so nothing
    /// outside this module could ever reach it. The table reported those bytes
    /// through `bytes_on_disk` while the file on disk was empty.
    ///
    /// A flush failure cannot be reported from `drop`, so it goes to stderr
    /// rather than being swallowed — a lost spill is worth knowing about. This
    /// module has no logger; `eprintln!` matches the reducer's own warning.
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("warning: spill flush on drop failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::table::Embedding;

    /// A unique scratch path per test, removed on drop.
    struct TmpPath(std::path::PathBuf);

    impl TmpPath {
        fn new(stem: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEQ: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "unfer-spill-{}-{stem}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create temp dir");
            TmpPath(dir)
        }
        fn file(&self) -> std::path::PathBuf {
            self.0.join("spill.bin")
        }
    }

    impl Drop for TmpPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn emb(seed: f32) -> Embedding {
        vec![seed, seed + 1.0, seed + 2.0]
    }

    fn key(n: u8) -> [u8; 32] {
        let mut k = [0u8; 32];
        k[0] = n;
        k[31] = n.wrapping_mul(7);
        k
    }

    /// The bug: records were written but `open` started with an empty index, so
    /// a reopened spill reported itself empty and answered `None` for keys that
    /// were physically on disk.
    #[test]
    fn reopening_a_spill_rebuilds_its_index() {
        let tmp = TmpPath::new("reopen");
        let path = tmp.file();

        {
            let mut tier = SpillTier::open(&path).expect("open");
            for i in 0..8u8 {
                tier.put(Granularity::Sentence, &key(i), &emb(i as f32))
                    .expect("put");
            }
            assert_eq!(tier.len(), 8);
        }

        let mut tier = SpillTier::open(&path).expect("reopen");
        assert_eq!(tier.len(), 8, "the index must be rebuilt from the file");
        assert!(!tier.is_empty());
        let got = tier
            .get(Granularity::Sentence, &key(3))
            .expect("get")
            .expect("entry 3 survives a reopen");
        assert_eq!(got, emb(3.0), "and reads back the original embedding");
        assert!(
            tier.get(Granularity::Sentence, &key(99))
                .expect("get")
                .is_none(),
            "a key that was never written is still absent"
        );
    }

    /// The other bug: nothing flushed on drop, so up to 1 MiB of records were
    /// discarded and the file was left empty while the table reported bytes.
    #[test]
    fn buffered_records_survive_the_drop() {
        let tmp = TmpPath::new("drop");
        let path = tmp.file();

        {
            let mut tier = SpillTier::open(&path).expect("open");
            for i in 0..4u8 {
                tier.put(Granularity::Subderiv, &key(i), &emb(i as f32))
                    .expect("put");
            }
            // Well under FLUSH_THRESHOLD, so nothing has flushed yet.
            assert!(path.metadata().expect("stat").len() < 4096);
        }

        let written = std::fs::metadata(&path).expect("stat").len();
        assert!(written > 0, "drop must flush buffered records to disk");

        let mut tier = SpillTier::open(&path).expect("reopen");
        assert_eq!(tier.len(), 4, "all four records reached the file");
        assert_eq!(
            tier.get(Granularity::Subderiv, &key(2))
                .expect("get")
                .expect("entry"),
            emb(2.0)
        );
    }

    /// A later write of the same address wins, and the index points at the
    /// newer record — including across a reopen.
    #[test]
    fn a_rewrite_wins_and_survives_a_reopen() {
        let tmp = TmpPath::new("rewrite");
        let path = tmp.file();
        {
            let mut tier = SpillTier::open(&path).expect("open");
            tier.put(Granularity::Sentence, &key(1), &emb(1.0))
                .expect("put");
            tier.put(Granularity::Sentence, &key(1), &emb(99.0))
                .expect("put again");
        }
        let mut tier = SpillTier::open(&path).expect("reopen");
        assert_eq!(tier.len(), 1, "one address, one index entry");
        assert_eq!(
            tier.get(Granularity::Sentence, &key(1))
                .expect("get")
                .expect("entry"),
            emb(99.0),
            "the newer record must win after a reopen too"
        );
    }

    /// `forget` drops the index entry but keeps the bytes, so the hot copy
    /// becomes authoritative and the record stays append-only.
    #[test]
    fn forget_drops_the_entry_and_keeps_the_bytes() {
        let tmp = TmpPath::new("forget");
        let path = tmp.file();
        let mut tier = SpillTier::open(&path).expect("open");
        tier.put(Granularity::Window, &key(5), &emb(5.0))
            .expect("put");
        tier.flush().expect("flush");
        let on_disk = std::fs::metadata(&path).expect("stat").len();
        assert_eq!(on_disk, (37 + 12) as u64, "one record reached the file");

        tier.forget(Granularity::Window, &key(5));
        assert_eq!(tier.len(), 0, "the index no longer claims it");
        assert!(
            tier.get(Granularity::Window, &key(5))
                .expect("get")
                .is_none(),
            "and answers None, so the hot copy is authoritative"
        );
        assert_eq!(
            std::fs::metadata(&path).expect("stat").len(),
            on_disk,
            "an append-only file is not rewritten: the record stays"
        );
    }

    /// A crash mid-append leaves a partial record. Reopening must truncate it
    /// rather than fail, and the complete records before it must survive.
    #[test]
    fn a_truncated_final_record_is_recovered_from() {
        let tmp = TmpPath::new("truncated");
        let path = tmp.file();
        {
            let mut tier = SpillTier::open(&path).expect("open");
            for i in 0..3u8 {
                tier.put(Granularity::Sentence, &key(i), &emb(i as f32))
                    .expect("put");
            }
        }
        let good_len = std::fs::metadata(&path).expect("stat").len();

        // Simulate a crash partway through a fourth record's header.
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&path).expect("append");
            f.write_all(&[2u8, 9, 0, 0, 0])
                .expect("write partial header");
        }
        assert!(std::fs::metadata(&path).expect("stat").len() > good_len);

        let mut tier = SpillTier::open(&path).expect("reopen over a partial record");
        assert_eq!(tier.len(), 3, "the complete records survive");
        assert_eq!(
            tier.get(Granularity::Sentence, &key(0))
                .expect("get")
                .expect("entry"),
            emb(0.0)
        );
        assert_eq!(
            std::fs::metadata(&path).expect("stat").len(),
            good_len,
            "the unusable tail is truncated so the next append is aligned"
        );
    }

    /// `bytes_on_disk` counts buffered records, so it must agree with the file
    /// once the tier has flushed.
    #[test]
    fn bytes_on_disk_agrees_with_the_file_after_a_flush() {
        let tmp = TmpPath::new("bytes");
        let path = tmp.file();
        let mut tier = SpillTier::open(&path).expect("open");
        for i in 0..5u8 {
            tier.put(Granularity::Sentence, &key(i), &emb(i as f32))
                .expect("put");
        }
        assert_eq!(
            tier.bytes_on_disk(),
            std::fs::metadata(&path).expect("stat").len() + (37 + 12) * 5,
            "buffered records are counted"
        );
        tier.flush().expect("flush");
        assert_eq!(
            tier.bytes_on_disk(),
            std::fs::metadata(&path).expect("stat").len(),
            "and match the file exactly once flushed"
        );
    }
}
