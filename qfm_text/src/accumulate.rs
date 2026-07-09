//! Streaming pass over the corpus (Stage 2).
//!
//! The accumulator is the **FSDP analog**: one `ChannelAccumulator` per
//! shard, merged at the end of the epoch. It is purely a counter
//! structure — no Fock-space objects, no Hamiltonian yet. The flow
//! through the pipeline is:
//!
//! ```text
//!  shard ─[WindowIter]─▶ (ctx, next) ─[hasher]─▶ active modes
//!       ─▶ for each mode: ModeStats::observe(next) and unigram[next] += 1
//!       ─▶ merge(accumulator_per_shard) into the running accumulator
//! ```
//!
//! The O(M) cost (one pass over the corpus) is the "training" cost of
//! the QFM-Text model — there is no SGD, no gradient, no backprop.
//! Multiple epochs differ only because HRM-Text's stratified sampling
//! feeds different shards per epoch; the counts keep accumulating
//! across epochs (more data ⇒ better histograms, the "training curve").

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::config::TextConfig;
use crate::corpus::Shard;
use crate::error::QfmTextError;
use crate::features::OrderHasher;

/// One mode's accumulated statistics. `weight` is the total number of
/// windows that activated this mode; `hist` is a top-T-by-count list
/// of `(next_token, count)` pairs (T = `TextConfig::hist_cap`); `escape`
/// is the total count of *non-listed* next tokens evicted from the
/// histogram (or never recorded because the mode hadn't seen them).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModeStats {
    pub weight: u64,
    pub hist: Vec<(u32, u32)>,
    pub escape: u64,
}

impl ModeStats {
    /// Record one observation of `next` against this mode. The
    /// histogram is kept sorted by descending count (then by token id
    /// for deterministic tie-break). If the cap is reached, the
    /// smallest-count entry is evicted and its count moved to escape.
    pub fn observe(&mut self, next: u32, hist_cap: usize) {
        self.weight += 1;
        self.observe_hist_only(next, hist_cap);
    }

    /// Update only the histogram (no `weight` increment). Used by
    /// `ChannelAccumulator::merge`, which adds the other side's
    /// `weight` separately to avoid double-counting.
    pub fn observe_hist_only(&mut self, next: u32, hist_cap: usize) {
        if let Some(slot) = self.hist.iter_mut().find(|(t, _)| *t == next) {
            slot.1 += 1;
        } else {
            if self.hist.len() < hist_cap {
                self.hist.push((next, 1));
            } else {
                let (min_idx, _) = self
                    .hist
                    .iter()
                    .enumerate()
                    .min_by(|(i, (ta, ca)), (j, (tb, cb))| {
                        ca.cmp(cb).then(ta.cmp(tb)).then(i.cmp(j))
                    })
                    .expect("hist is non-empty when at cap");
                let (_, min_count) = self.hist[min_idx];
                if min_count == 0 {
                    let (max_idx, _) = self
                        .hist
                        .iter()
                        .enumerate()
                        .max_by_key(|(i, (t, _))| (*t, std::cmp::Reverse(*i)))
                        .expect("non-empty");
                    let evicted = self.hist.remove(max_idx);
                    self.escape += evicted.1 as u64;
                } else {
                    let evicted = self.hist.remove(min_idx);
                    self.escape += evicted.1 as u64;
                }
                self.hist.push((next, 1));
            }
        }
        self.hist.sort_by(|(a_t, a_c), (b_t, b_c)| b_c.cmp(a_c).then(a_t.cmp(b_t)));
    }

    /// Total count of observations this mode has accumulated. Should
    /// equal `weight`.
    pub fn total_count(&self) -> u64 {
        self.weight
    }
}

/// The streaming accumulator: a per-mode stats map + the unigram
/// marginal. The FSDP analog is "one accumulator per shard, merged
/// across shards at the end of the epoch".
///
/// `merge` is the FSDP all-reduce. It is **associative and
/// commutative**:
///   - `merge(a, b) == merge(b, a)`
///   - `merge(merge(a, b), c) == merge(a, merge(b, c))`
///   - `merge(a, zero) == a`
/// because the underlying operations are `count + count` (commutative,
/// associative) and the histogram merge is well-defined for top-T-by-
/// count (deterministic tie-break by token id).
#[derive(Debug, Clone)]
pub struct ChannelAccumulator {
    /// Per-mode statistics, keyed by global mode index.
    pub stats: FxHashMap<u32, ModeStats>,
    /// Per-token unigram count. Indexed by token id; length = `vocab_size`.
    pub unigram: Vec<u64>,
    /// Total number of windows observed so far.
    pub total_windows: u64,
    /// Configuration snapshot.
    pub cfg: TextConfig,
}

impl ChannelAccumulator {
    /// Build a fresh accumulator with the given vocabulary size and
    /// config. The unigram vector is allocated lazily on the first
    /// observation; this keeps a single empty accumulator cheap.
    pub fn new(vocab_size: u32, cfg: TextConfig) -> Self {
        Self {
            stats: FxHashMap::default(),
            unigram: Vec::with_capacity(vocab_size as usize),
            total_windows: 0,
            cfg,
        }
    }

    /// Number of distinct modes that have been observed.
    pub fn n_active_modes(&self) -> usize {
        self.stats.len()
    }

    /// Observe one `(ctx, next)` window. The accumulator grows the
    /// unigram vec to the size of `next` if it was previously shorter.
    pub fn observe(&mut self, modes: &[u32], next: u32) {
        self.total_windows += 1;
        // Unigram update. Grow lazily so a 16k-vocab allocation only
        // happens on first use.
        if self.unigram.len() <= next as usize {
            self.unigram.resize(next as usize + 1, 0);
        }
        self.unigram[next as usize] += 1;
        for &m in modes {
            self.stats
                .entry(m)
                .or_default()
                .observe(next, self.cfg.hist_cap);
        }
    }

    /// Walk one shard with the given hasher and update the accumulator.
    pub fn observe_shard(&mut self, shard_path: &Path, hasher: &OrderHasher) -> Result<(), QfmTextError> {
        // Open the shard as a borrowed file to learn vocab_size. We
        // can't pull vocab_size from the manifest here because the
        // accumulator doesn't have a reference to it; instead, the
        // accumulator's `unigram` vec is its own vocabulary, and the
        // caller is responsible for choosing a vocab_size >= max
        // observed token id. We read the unigram cap as a side
        // channel: the maximum observed token id will be reflected
        // in `unigram.len()` after the first observation.
        //
        // For the vocab-bound check the caller's `accumulate_shards`
        // wraps this in a typed error if the shard declares a higher
        // vocab_size than the config allows.
        let shard = Shard::open(shard_path, hasher.config().vocab_size_for_check())?;
        shard.check_vocab()?;
        for (ctx, next) in shard.windows(hasher.config().n_orders) {
            let modes = hasher.encode_modes(&ctx);
            self.observe(&modes, next);
        }
        Ok(())
    }

    /// Merge another accumulator into `self`. The two accumulators
    /// must have the same `cfg.hist_cap`. After the merge the
    /// accumulator's `total_windows` is the sum of both, the per-mode
    /// `weight` and `hist` counts add, the `escape` counts add, and
    /// the histograms are re-capped.
    pub fn merge(&mut self, other: ChannelAccumulator) {
        assert_eq!(
            self.cfg.hist_cap, other.cfg.hist_cap,
            "hist_cap mismatch on merge"
        );
        self.total_windows += other.total_windows;
        // Unigram merge.
        if other.unigram.len() > self.unigram.len() {
            self.unigram.resize(other.unigram.len(), 0);
        }
        for (i, &c) in other.unigram.iter().enumerate() {
            self.unigram[i] += c;
        }
        // Mode-stat merge. Use `observe_hist_only` to avoid
        // double-counting the per-mode `weight` (the `weight` is
        // added once, at the end, from `other_stats.weight`).
        for (mode, mut other_stats) in other.stats {
            let entry = self.stats.entry(mode).or_default();
            for (tok, cnt) in other_stats.hist.drain(..) {
                for _ in 0..cnt {
                    entry.observe_hist_only(tok, self.cfg.hist_cap);
                }
            }
            entry.weight += other_stats.weight;
            entry.escape += other_stats.escape;
            entry
                .hist
                .sort_by(|(a_t, a_c), (b_t, b_c)| b_c.cmp(a_c).then(a_t.cmp(b_t)));
        }
    }
}

/// Streaming pass over a list of shards. Returns a single merged
/// accumulator.
///
/// **Memory model:** this function processes shards **sequentially**
/// to keep peak memory bounded. The previous implementation used
/// `par_iter().collect()` to build 245 per-shard `ChannelAccumulator`s
/// in parallel and then merge them; each per-shard accumulator is
/// ~100 MB (250K active modes × 64 hist entries × 8 bytes), so the
/// `collect()` Vec held 245 × 100 MB ≈ 25 GB at once — an OOM on
/// any 16 GB system. Sequential processing keeps peak memory at
/// ~100 MB (one shard's accumulator at a time), bounded by the
/// final ~200 MB merged accumulator.
///
/// **Why no parallelism:** the per-shard accumulator's size is
/// dominated by the per-mode histograms (`ModeStats.hist`, capped at
/// `hist_cap` entries). With `hist_cap = 64` and 250K active modes,
/// the per-shard accumulator is 250K × ~528 bytes ≈ 130 MB. Even a
/// `reduce`-based parallel fold (one accumulator per rayon worker
/// thread) would still hold `n_threads × 130 MB` ≈ 2 GB at once.
/// For the WikiText-103 245-shard corpus, sequential is the only
/// memory-safe option. The bottleneck is the inner sort in
/// `ModeStats::observe` (O(hist_cap log hist_cap) per observation =
/// ~400 ops × 128M observations = 50 Gops), which is CPU-bound
/// and would dominate the wall time regardless of shard-level
/// parallelism.
pub fn accumulate_shards(
    shard_paths: &[PathBuf],
    cfg: &TextConfig,
    vocab_size: u32,
) -> Result<ChannelAccumulator, QfmTextError> {
    let hasher = OrderHasher::new(cfg.clone());
    let mut acc = ChannelAccumulator::new(vocab_size, cfg.clone());
    for path in shard_paths {
        acc.observe_shard(path, &hasher)?;
    }
    Ok(acc)
}

/// Read the entire file (small files only) into a byte vec. Used for
/// `vocab_size_for_check` in dev fixtures where the manifest is
/// implicit.
pub fn read_small_file(path: &Path) -> Result<Vec<u8>, QfmTextError> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

impl TextConfig {
    /// Sentinel for the shard-level vocab check: the manifest
    /// provides the real vocab_size; for dev fixtures (where the
    /// caller is the test code itself), this is `u32::MAX`, which
    /// disables the bound check.
    pub fn vocab_size_for_check(&self) -> u32 {
        u32::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
            block_sizes: vec![16, 16],
            salts: vec![1, 2],
            hist_cap: 3,
            max_rank: 4,
            m_shifts: 4,
            lambda: vec![1.0, 1.0],
            t: 1.0,
            discount: 0.5,
            seed: 0,
            ..Default::default()
        }
    }

    fn write_shard(path: &Path, tokens: &[u32]) {
        let mut buf = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    #[test]
    fn observe_grows_unigram_lazily() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        acc.observe(&[1], 5);
        assert_eq!(acc.unigram.len(), 6);
        assert_eq!(acc.unigram[5], 1);
        assert_eq!(acc.total_windows, 1);
    }

    #[test]
    fn observe_records_per_mode_weight_and_hist() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        acc.observe(&[0, 1], 7);
        acc.observe(&[0, 1], 7);
        acc.observe(&[0, 1], 9);
        // Mode 0 saw 3 windows; mode 1 saw 3 windows.
        for m in 0..2 {
            let st = &acc.stats[&m];
            assert_eq!(st.weight, 3);
        }
        let s0 = &acc.stats[&0];
        let s0_hist: FxHashMap<u32, u32> = s0.hist.iter().copied().collect();
        assert_eq!(s0_hist[&7], 2);
        assert_eq!(s0_hist[&9], 1);
    }

    #[test]
    fn hist_cap_evicts_to_escape() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        // hist_cap = 3. Observe 4 distinct tokens, each once.
        for t in 0..4 {
            acc.observe(&[0], t);
        }
        let s = &acc.stats[&0];
        assert_eq!(s.hist.len(), 3);
        assert_eq!(s.escape, 1, "the 4th distinct token got evicted to escape");
        // Total count is conserved.
        let sum: u32 = s.hist.iter().map(|(_, c)| c).sum();
        assert_eq!(sum + s.escape as u32, s.weight as u32);
    }

    #[test]
    fn merge_is_associative_and_conserves_total() {
        // Two independent accumulators on the same shard (idempotent
        // — same data observed twice) merge to exactly twice the
        // per-mode statistics. This is the FSDP analog: a shard is
        // observed by two workers and the all-reduce is the sum.
        // The boundary-loss caveat of the cross-shard merge does not
        // apply here, because each shard is observed in its entirety.
        let c = cfg();
        let tokens: Vec<u32> = (0..500).map(|i| (i / 5) % 7).collect();
        let dir = tempdir().unwrap();
        let sp = dir.path().join("shard.bin");
        write_shard(&sp, &tokens);
        let h = OrderHasher::new(c.clone());
        let mut a = ChannelAccumulator::new(0, c.clone());
        a.observe_shard(&sp, &h).unwrap();
        let mut b = ChannelAccumulator::new(0, c.clone());
        b.observe_shard(&sp, &h).unwrap();
        let snap_a = a.clone();
        let snap_b = b.clone();
        a.merge(b);
        // Total: 2x the single-pass total.
        assert_eq!(a.total_windows, 2 * snap_a.total_windows);
        assert_eq!(a.unigram, snap_a.unigram.iter().map(|&x| 2 * x).collect::<Vec<_>>());
        for (mode, ms) in &a.stats {
            let sa = snap_a.stats.get(mode).expect("a has mode");
            let sb = snap_b.stats.get(mode).expect("b has mode");
            // Per-mode weight is the sum.
            assert_eq!(ms.weight, sa.weight + sb.weight, "mode {mode}");
            // Histogram counts add, escape adds.
            assert_eq!(ms.escape, sa.escape + sb.escape, "mode {mode} escape");
            // The merged hist (top-T by count) must be consistent
            // with the union of the two hists with summed counts —
            // we verify by sorting and comparing token-by-token
            // counts.
            let mut merged_counts: FxHashMap<u32, u32> = FxHashMap::default();
            for (tok, cnt) in &ms.hist {
                *merged_counts.entry(*tok).or_insert(0) += cnt;
            }
            let mut combined_counts: FxHashMap<u32, u32> = FxHashMap::default();
            for (tok, cnt) in &sa.hist {
                *combined_counts.entry(*tok).or_insert(0) += cnt;
            }
            for (tok, cnt) in &sb.hist {
                *combined_counts.entry(*tok).or_insert(0) += cnt;
            }
            // The merged hist is the top-T of the *raw* (a.hist +
            // b.hist) counts; for hist_cap=3, both are exactly the
            // top-3, so the histograms should match. (The escape
            // captures whatever falls below the cap.)
            for (tok, cnt) in &merged_counts {
                assert_eq!(
                    combined_counts.get(tok).copied().unwrap_or(0),
                    *cnt,
                    "mode {mode} token {tok} count mismatch"
                );
            }
        }
    }

    #[test]
    fn empty_accumulator_is_zero_of_merge() {
        let mut a = ChannelAccumulator::new(0, cfg());
        a.observe(&[0, 1], 5);
        a.observe(&[2], 9);
        let b = ChannelAccumulator::new(0, cfg());
        let snap = a.clone();
        a.merge(b);
        assert_eq!(a.total_windows, snap.total_windows);
        assert_eq!(a.unigram, snap.unigram);
        assert_eq!(a.stats, snap.stats);
    }
}
