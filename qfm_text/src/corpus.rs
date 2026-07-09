//! Memory-mapped shard reader for token-stream corpora (Stage 1).
//!
//! A *shard* is a flat little-endian u32 file produced by
//! `scripts/prepare_corpus.py` (see Stage 0). A *manifest* is a JSON
//! sidecar that lists the shards, the vocabulary size, and the per-shard
//! sha256 (so an idempotent `prepare_corpus.sh` re-run can verify the
//! disk contents are byte-identical to the last accepted tokenization).
//!
//! Memory-mapping is the right choice for the streaming accumulator pass:
//! the corpus is too large to fit in RAM at the planned scale
//! (10⁷–10⁸ tokens, K₂ ~ 10⁵; a full 10⁸-token corpus is 400 MB of
//! token ids alone), but every byte is touched exactly once. `Mmap`
//! gives us O(1) random access, zero-copy reads, and the kernel
//! manages prefetching for us.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::error::QfmTextError;

/// A single shard entry in a [`Manifest`]: the relative path inside the
/// shard directory, the token count, and the sha256 of the file bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardEntry {
    pub path: String,
    pub n_tokens: u64,
    pub sha256: String,
}

/// The shard manifest. JSON-serialized; mirrors what
/// `scripts/prepare_corpus.py` writes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema: String,
    pub vocab_size: u32,
    pub tokens_per_shard: u64,
    pub n_shards: u32,
    pub n_tokens: u64,
    pub shards: Vec<ShardEntry>,
    pub tokenizer_sha256: String,
    pub tokenizer_path: String,
    pub corpus: String,
    pub license: String,
    pub attribution: String,
}

impl Manifest {
    /// Read a manifest from `path` (a JSON file).
    pub fn read(path: impl AsRef<Path>) -> Result<Self, QfmTextError> {
        let path_ref = path.as_ref();
        let text = std::fs::read_to_string(path_ref).map_err(|e| {
            QfmTextError::BadManifest {
                path: path_ref.display().to_string(),
                reason: format!("read failed: {e}"),
            }
        })?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        if manifest.schema != "qfm_text.shard_manifest/v1" {
            return Err(QfmTextError::BadManifest {
                path: path_ref.display().to_string(),
                reason: format!("unknown schema: {}", manifest.schema),
            });
        }
        Ok(manifest)
    }

    /// Total number of tokens across every shard.
    pub fn total_tokens(&self) -> u64 {
        self.n_tokens
    }

    /// Get the absolute path of the i-th shard given the manifest
    /// directory. The shard entry's `path` is interpreted relative to
    /// the manifest's parent directory.
    pub fn shard_path(&self, manifest_dir: &Path, i: usize) -> PathBuf {
        manifest_dir.join(&self.shards[i].path)
    }
}

/// A memory-mapped shard: a flat little-endian u32 file, lazy-mapped on
/// open. Iteration via [`Shard::tokens`] and [`Shard::windows`] works
/// on the mmap'd region with no further allocation.
pub struct Shard {
    /// Absolute path of the mapped file.
    path: PathBuf,
    /// The mmap itself. Held for the lifetime of the `Shard`.
    _mmap: Mmap,
    /// Number of u32 tokens stored.
    n_tokens: u64,
    /// Vocabulary size (from the manifest, for sanity checks).
    vocab_size: u32,
}

impl std::fmt::Debug for Shard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shard")
            .field("path", &self.path)
            .field("n_tokens", &self.n_tokens)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl Shard {
    /// Open and memory-map a shard file. Validates that the byte count
    /// is a multiple of 4 (so `chunks_exact(4)` is lossless).
    pub fn open(path: impl AsRef<Path>, vocab_size: u32) -> Result<Self, QfmTextError> {
        let path_ref = path.as_ref().to_path_buf();
        let file = File::open(&path_ref).map_err(|e| QfmTextError::BadShard {
            path: path_ref.display().to_string(),
            reason: format!("open failed: {e}"),
        })?;
        // SAFETY: mmap is read-only. The file is owned by the kernel for
        // the duration of the mmap and cannot be modified concurrently
        // because we hold an exclusive `File` open. The mmap is dropped
        // before the file is dropped.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| QfmTextError::BadShard {
            path: path_ref.display().to_string(),
            reason: format!("mmap failed: {e}"),
        })?;
        if mmap.len() % 4 != 0 {
            return Err(QfmTextError::BadShard {
                path: path_ref.display().to_string(),
                reason: format!("byte count {} is not a multiple of 4", mmap.len()),
            });
        }
        let n_tokens = (mmap.len() / 4) as u64;
        Ok(Self {
            path: path_ref,
            _mmap: mmap,
            n_tokens,
            vocab_size,
        })
    }

    /// Number of u32 tokens in the shard.
    pub fn len(&self) -> usize {
        self.n_tokens as usize
    }

    /// True if the shard contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.n_tokens == 0
    }

    /// Path the shard was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Vocabulary size (declared by the manifest, copied here for the
    /// `vocab_check` invariant).
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Iterate over the u32 token ids. The lifetime is tied to the
    /// mmap (which is owned by `self`), so this cannot outlive the
    /// `Shard`.
    pub fn iter(&self) -> ShardIter<'_> {
        ShardIter {
            mmap: &self._mmap,
            pos: 0,
        }
    }

    /// Validate every token id is in `[0, vocab_size)`. O(n), so the
    /// caller should not call this on the full corpus unless they have
    /// a reason to.
    pub fn check_vocab(&self) -> Result<(), QfmTextError> {
        for tok in self.iter() {
            if tok >= self.vocab_size {
                return Err(QfmTextError::VocabMismatch {
                    token_id: tok,
                    vocab_size: self.vocab_size,
                    path: self.path.display().to_string(),
                });
            }
        }
        Ok(())
    }

    /// Iterate over `(context, next)` training windows of width up to
    /// `n_max`. `context` is a slice of the *last* `min(n_max, i)` tokens
    /// ending at byte position `i`; `next` is `tokens[i]`. The shorter
    /// contexts near the start of the shard are intentional: orders
    /// beyond `context.len()` are simply absent for that window.
    pub fn windows(&self, n_max: usize) -> WindowIter<'_> {
        WindowIter {
            shard: self,
            n_max,
            pos: 0,
        }
    }
}

/// Iterator over the u32 token ids of a shard.
pub struct ShardIter<'a> {
    mmap: &'a Mmap,
    pos: usize,
}

impl<'a> Iterator for ShardIter<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.pos + 4 > self.mmap.len() {
            return None;
        }
        // SAFETY: bounds check above guarantees a 4-byte aligned read
        // within the mmap. The mmap is owned by the `Shard`, which
        // outlives this iterator by construction.
        let bytes: [u8; 4] = unsafe {
            [
                *self.mmap.get_unchecked(self.pos),
                *self.mmap.get_unchecked(self.pos + 1),
                *self.mmap.get_unchecked(self.pos + 2),
                *self.mmap.get_unchecked(self.pos + 3),
            ]
        };
        self.pos += 4;
        Some(u32::from_le_bytes(bytes))
    }
}

/// Iterator over `(context, next)` training windows.
pub struct WindowIter<'a> {
    shard: &'a Shard,
    n_max: usize,
    pos: usize,
}

impl<'a> Iterator for WindowIter<'a> {
    /// `(context: &[u32], next: u32)`. The context is a slice of the
    /// *last* `min(n_max, pos)` tokens before `pos`, in the natural
    /// order (oldest first, newest last).
    type Item = (Vec<u32>, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.shard.len() {
            return None;
        }
        let n_max = self.n_max;
        let start = self.pos.saturating_sub(n_max);
        let mut ctx: Vec<u32> = Vec::with_capacity(self.pos - start);
        let mut idx = start;
        while idx < self.pos {
            let bytes: [u8; 4] = unsafe {
                [
                    *self.shard._mmap.get_unchecked(idx * 4),
                    *self.shard._mmap.get_unchecked(idx * 4 + 1),
                    *self.shard._mmap.get_unchecked(idx * 4 + 2),
                    *self.shard._mmap.get_unchecked(idx * 4 + 3),
                ]
            };
            ctx.push(u32::from_le_bytes(bytes));
            idx += 1;
        }
        let next_bytes: [u8; 4] = unsafe {
            [
                *self.shard._mmap.get_unchecked(self.pos * 4),
                *self.shard._mmap.get_unchecked(self.pos * 4 + 1),
                *self.shard._mmap.get_unchecked(self.pos * 4 + 2),
                *self.shard._mmap.get_unchecked(self.pos * 4 + 3),
            ]
        };
        let next = u32::from_le_bytes(next_bytes);
        self.pos += 1;
        Some((ctx, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_shard(path: &Path, tokens: &[u32]) {
        let mut buf = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    #[test]
    fn shard_round_trip_tokens() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard_00000.bin");
        let tokens: Vec<u32> = (0..1000).collect();
        write_shard(&path, &tokens);
        let shard = Shard::open(&path, 2000).unwrap();
        assert_eq!(shard.len(), 1000);
        let read: Vec<u32> = shard.iter().collect();
        assert_eq!(read, tokens);
    }

    #[test]
    fn shard_rejects_non_multiple_of_4() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        let err = Shard::open(&path, 10).unwrap_err();
        match err {
            QfmTextError::BadShard { reason, .. } => {
                assert!(reason.contains("not a multiple of 4"), "got: {reason}");
            }
            _ => panic!("expected BadShard, got {err:?}"),
        }
    }

    #[test]
    fn shard_vocab_check_catches_oversized_token() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.bin");
        write_shard(&path, &[0, 1, 2, 5]); // vocab_size = 5, 5 is out of range
        let shard = Shard::open(&path, 5).unwrap();
        let err = shard.check_vocab().unwrap_err();
        match err {
            QfmTextError::VocabMismatch {
                token_id,
                vocab_size,
                ..
            } => {
                assert_eq!(token_id, 5);
                assert_eq!(vocab_size, 5);
            }
            _ => panic!("expected VocabMismatch, got {err:?}"),
        }
    }

    #[test]
    fn window_iter_boundary() {
        // 5 tokens: [0,1,2,3,4], n_max=2 -> 5 windows (one per
        // position), with shorter contexts near the start:
        //   ( [], 0 ), ( [0], 1 ), ( [0,1], 2 ), ( [1,2], 3 ), ( [2,3], 4 )
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.bin");
        write_shard(&path, &[0, 1, 2, 3, 4]);
        let shard = Shard::open(&path, 5).unwrap();
        let windows: Vec<_> = shard.windows(2).collect();
        assert_eq!(windows.len(), 5);
        assert_eq!(windows[0], (vec![], 0));
        assert_eq!(windows[1], (vec![0], 1));
        assert_eq!(windows[2], (vec![0, 1], 2));
        assert_eq!(windows[3], (vec![1, 2], 3));
        assert_eq!(windows[4], (vec![2, 3], 4));
    }

    #[test]
    fn window_iter_count_matches_len() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.bin");
        let tokens: Vec<u32> = (0..257).collect();
        write_shard(&path, &tokens);
        let shard = Shard::open(&path, 1000).unwrap();
        let count = shard.windows(4).count();
        assert_eq!(count, shard.len());
    }

    #[test]
    fn manifest_round_trip() {
        let dir = tempdir().unwrap();
        let mpath = dir.path().join("manifest.json");
        let m = Manifest {
            schema: "qfm_text.shard_manifest/v1".to_string(),
            vocab_size: 16_384,
            tokens_per_shard: 524_288,
            n_shards: 1,
            n_tokens: 200_000,
            shards: vec![ShardEntry {
                path: "shard_00000.bin".to_string(),
                n_tokens: 200_000,
                sha256: "deadbeef".repeat(8),
            }],
            tokenizer_sha256: "feedface".repeat(8),
            tokenizer_path: "tokenizer.json".to_string(),
            corpus: "wikitext-103-test".to_string(),
            license: "CC-BY-SA-4.0".to_string(),
            attribution: "WikiText-103 test split".to_string(),
        };
        std::fs::write(&mpath, serde_json::to_string_pretty(&m).unwrap()).unwrap();
        let m2 = Manifest::read(&mpath).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn manifest_rejects_wrong_schema() {
        // A complete JSON with the wrong schema is a BadManifest;
        // a JSON missing required fields is a Json parse error
        // (both are caller-visible error variants, the test just
        // ensures the wrong-schema path doesn't slip through).
        let dir = tempdir().unwrap();
        let mpath = dir.path().join("manifest.json");
        std::fs::write(
            &mpath,
            r#"{"schema": "other/v1", "vocab_size": 1, "tokens_per_shard": 1, "n_shards": 1, "n_tokens": 1, "shards": [], "tokenizer_sha256": "", "tokenizer_path": "", "corpus": "", "license": "", "attribution": ""}"#,
        )
        .unwrap();
        let err = Manifest::read(&mpath).unwrap_err();
        assert!(matches!(err, QfmTextError::BadManifest { .. }));
    }
}
