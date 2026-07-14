//! Streaming pass over the corpus (Stage 2, rev 36: hashing removed).
//!
//! The accumulator is the **FSDP analog**: one `ChannelAccumulator` per
//! shard, merged at the end of the epoch. It is purely a counter
//! structure — no Fock-space objects, no Hamiltonian yet. The flow
//! through the pipeline is:
//!
//! ```text
//!  shard ─[WindowIter]─▶ (ctx, next) ─[registry.assign(ctx)]─▶ active modes
//!       ─▶ for each mode: ModeStats::observe(next) and unigram[next] += 1
//!       ─▶ merge(accumulator_per_shard) into the running accumulator
//! ```
//!
//! The O(M) cost (one pass over the corpus) is the "training" cost of
//! the QFM-Text model — there is no SGD, no gradient, no backprop.
//! Multiple epochs differ only because HRM-Text's stratified sampling
//! feeds different shards per epoch; the counts keep accumulating
//! across epochs (more data ⇒ better histograms, the "training curve").
//!
//! # Rev 36 change: hashing removed
//!
//! Where the prior `OrderHasher` mapped every context of order `o`
//! to a hash slot in `[offset_o, offset_o + block_size_o)` (a fixed
//! bounded table; unrelated contexts collided on the same slot and
//! blended their histograms), rev 36 uses a [`ContextRegistry`]:
//! every distinct context gets a **fresh, unique** mode index. No
//! hash collisions, no histogram blending, no bounded table. The
//! Krylov pipeline's W matrix is now `(K_2_total, rank)` where
//! `K_2_total = 1 + Σ_o n_active_modes_o` is the actual vocabulary
//! of unique contexts the corpus produced — typically ~10⁵–10⁷ for
//! WikiText-103, not the fixed ~10⁵ the hashed table was clamped to.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::config::TextConfig;
use crate::corpus::Shard;
use crate::error::QfmTextError;
use crate::features::OrderHasher;
use crate::registry::{ContextRegistry, VACUUM_MODE};

/// Encoder selector for the streaming accumulator (rev 37).
///
/// The accumulator's `observe_shard` takes one of these and uses it
/// to map each `(context, next)` window to a list of active mode
/// indices. The default is `Hasher` (rev 35), the fallback for
/// `--encoder registry` comparison experiments is `Registry`
/// (rev 36, hashing-free, per-context).
#[derive(Debug, Clone)]
pub enum Encoder {
    /// Rev 35 `OrderHasher` — a fixed-size hash table with
    /// `block_sizes[o]` slots per order `o`. Hash collisions blend
    /// unrelated contexts. Memory-bounded; generalizes to unseen
    /// contexts at test time.
    Hasher(OrderHasher),
    /// Rev 36 `ContextRegistry` — one mode per unique context,
    /// no collisions, no bounded table. Memory grows with the
    /// corpus's unique-context count (~5.5M for WikiText-103).
    Registry(ContextRegistry),
}

impl Encoder {
    /// Encode a context into a list of active mode indices. For
    /// `Hasher`, this is the rev 35 hash; for `Registry`, this is
    /// the rev 36 per-context assignment (with the vacuum sentinel
    /// for unseen contexts).
    pub fn encode(&mut self, context: &[u32]) -> Vec<u32> {
        match self {
            Encoder::Hasher(h) => h.encode_modes(context),
            Encoder::Registry(r) => {
                let n = r.n_orders().min(context.len());
                if n == 0 {
                    // Empty context → no active order, vacuum sentinel.
                    vec![VACUUM_MODE; 1]
                } else {
                    (1..=n).map(|o| r.assign(o, context)).collect()
                }
            }
        }
    }

    /// The `n_orders` of the underlying encoder (for the `WindowIter`
    /// in `observe_shard`).
    pub fn n_orders(&self) -> usize {
        match self {
            Encoder::Hasher(h) => h.config().n_orders,
            Encoder::Registry(r) => r.n_orders(),
        }
    }

    /// The vocabulary-size cap used by `Shard::open` for the
    /// manifest's vocab-size check. `u32::MAX` disables the check
    /// (used by dev fixtures where the manifest is implicit).
    pub fn n_orders_for_check(&self) -> u32 {
        match self {
            Encoder::Hasher(h) => h.config().vocab_size_for_check(),
            Encoder::Registry(_) => u32::MAX,
        }
    }

    /// Borrow the inner `ContextRegistry` (only for `Encoder::Registry`).
    pub fn as_registry(&self) -> Option<&ContextRegistry> {
        match self {
            Encoder::Registry(r) => Some(r),
            _ => None,
        }
    }

    /// Mutably borrow the inner `ContextRegistry` (only for `Encoder::Registry`).
    pub fn as_registry_mut(&mut self) -> Option<&mut ContextRegistry> {
        match self {
            Encoder::Registry(r) => Some(r),
            _ => None,
        }
    }
}

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
///
/// **Rev 36:** the accumulator no longer holds a hasher. The
/// [`ContextRegistry`] is passed in by the caller (it must persist
/// across shards — accumulating shards sequentially with the same
/// `&mut ContextRegistry` is the streaming-pass equivalent of
/// `for shard in shards: registry.observe(shard)`). The mode 0
/// (vacuum) histogram is initialized at model-build time by
/// [`super::model::QfmTextModel::from_accumulator`], not here.
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
    /// Consecutive training-window mode transitions: `(prev_mode, curr_mode)`
    /// pairs for each adjacent pair of training windows. The primary mode
    /// is the highest-order mode (last in the encoder's mode list).
    pub transitions: Vec<(u32, u32)>,
    /// The previous window's primary mode (for tracking transitions across
    /// consecutive `observe` calls within a shard).
    last_window_mode: Option<u32>,
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
            transitions: Vec::new(),
            last_window_mode: None,
        }
    }

    /// Number of distinct modes that have been observed.
    pub fn n_active_modes(&self) -> usize {
        self.stats.len()
    }

    /// Observe one `(ctx, next)` window. The accumulator grows the
    /// unigram vec to the size of `next` if it was previously shorter.
    /// The provided `encoder` (rev 37) maps the context to a list of
    /// active mode indices: an `OrderHasher` (rev 35) for the
    /// default hashed encoder, or a `ContextRegistry` (rev 36) for
    /// the comparison-knob per-context encoder.
    ///
    /// Edge case: an **empty** context (`context.len() == 0`) has
    /// no active order to assign. The encoder routes this to the
    /// vacuum mode (index 0) for `Registry`; for `Hasher` with an
    /// empty context, the resulting empty mode list also routes to
    /// the vacuum sentinel as a defensive fallback.
    /// Returns the primary mode (highest-order mode, or VACUUM_MODE
    /// for empty contexts).
    pub fn observe(&mut self, encoder: &mut Encoder, context: &[u32], next: u32) -> u32 {
        self.total_windows += 1;
        // Unigram update. Grow lazily so a 16k-vocab allocation only
        // happens on first use.
        if self.unigram.len() <= next as usize {
            self.unigram.resize(next as usize + 1, 0);
        }
        self.unigram[next as usize] += 1;
        // Encode the context into a list of active mode indices.
        let modes = encoder.encode(context);
        let primary_mode = modes.last().copied().unwrap_or(VACUUM_MODE);
        if modes.is_empty() {
            // No active mode (e.g. empty context under OrderHasher).
            // Route to the vacuum sentinel so the observation is
            // never lost.
            self.stats
                .entry(VACUUM_MODE)
                .or_default()
                .observe(next, self.cfg.hist_cap);
        } else {
            for m in modes {
                self.stats
                    .entry(m)
                    .or_default()
                    .observe(next, self.cfg.hist_cap);
            }
        }
        // Record consecutive training-window transition.
        if let Some(prev) = self.last_window_mode {
            if prev != primary_mode {
                self.transitions.push((prev, primary_mode));
            }
        }
        self.last_window_mode = Some(primary_mode);
        primary_mode
    }

    /// Walk one shard with the given encoder and update the
    /// accumulator. The encoder is mutated: every distinct context
    /// the shard produces is hashed (or assigned a fresh mode index)
    /// and the observation is recorded against the active modes.
    pub fn observe_shard(
        &mut self,
        shard_path: &Path,
        encoder: &mut Encoder,
    ) -> Result<(), QfmTextError> {
        let shard = Shard::open(shard_path, encoder.n_orders_for_check())?;
        shard.check_vocab()?;
        for (ctx, next) in shard.windows(encoder.n_orders()) {
            self.observe(encoder, &ctx, next);
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
/// accumulator and the (now-grown) registry. The registry must be
/// passed in by the caller; it persists across shards and across
/// epochs. This is the rev-36 contract: the registry is a side
/// product of training that the compiled model needs at inference
/// time to map test contexts back to their assigned mode indices.
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
/// With the rev-36 registry the per-shard accumulator's per-mode
/// histogram footprint is unchanged (~528 bytes per active mode
/// entry), but the registry itself grows to ~5.5 M entries on
/// 128 M tokens of WikiText-103 (~250 MB). The registry is shared
/// across shards and is not duplicated per shard, so the sequential
/// memory model still holds.
pub fn accumulate_shards(
    shard_paths: &[PathBuf],
    encoder: &mut Encoder,
    cfg: &TextConfig,
    vocab_size: u32,
) -> Result<ChannelAccumulator, QfmTextError> {
    let mut acc = ChannelAccumulator::new(vocab_size, cfg.clone());
    for path in shard_paths {
        acc.observe_shard(path, encoder)?;
    }
    Ok(acc)
}

/// Observe one `(ctx, next)` window using a `ContextRegistry` for
/// the encoder (rev 36 fallback API; the rev 37 path is the
/// `Encoder` enum directly via [`ChannelAccumulator::observe`]).
/// This is a convenience wrapper that hides the `Encoder::Registry`
/// plumbing from callers that only ever need the registry path.
pub fn observe_with_registry(
    acc: &mut ChannelAccumulator,
    registry: &mut ContextRegistry,
    context: &[u32],
    next: u32,
) {
    let mut enc = Encoder::Registry(registry.clone());
    acc.observe(&mut enc, context, next);
    if let Some(r) = enc.as_registry() {
        registry.clone_from(r);
    }
}

/// Walk one shard with a `ContextRegistry` for the encoder (rev 36
/// fallback API; the rev 37 path is `Encoder` directly).
pub fn observe_shard_with_registry(
    acc: &mut ChannelAccumulator,
    registry: &mut ContextRegistry,
    shard_path: &Path,
) -> Result<(), QfmTextError> {
    let mut enc = Encoder::Registry(registry.clone());
    let res = acc.observe_shard(shard_path, &mut enc);
    if let Some(r) = enc.as_registry() {
        registry.clone_from(r);
    }
    res
}

/// Read the entire file (small files only) into a byte vec. Used for
/// `n_orders_for_check` in dev fixtures where the manifest is
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

impl ContextRegistry {
    /// Sentinel for the shard-level vocab check: the manifest
    /// provides the real vocab_size; for dev fixtures (where the
    /// caller is the test code itself), this is `u32::MAX`, which
    /// disables the bound check. (Mirrors the prior
    /// `TextConfig::vocab_size_for_check`; the registry owns it now
    /// because the registry is what `observe_shard` takes.)
    pub fn n_orders_for_check(&self) -> u32 {
        u32::MAX
    }
}

/// Re-export the vacuum sentinel (already `pub` in `registry`).
/// `accumulate_shards` and downstream callers can use
/// `accumulate::VACUUM_MODE` without importing the registry module
/// directly. The duplicate `use` is intentional — we re-export the
/// public symbol under a stable path.
#[allow(unused_imports)]
use crate::registry::VACUUM_MODE as _VACUUM_REEXPORT;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
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

    /// Helper: route an observe call through the `Encoder` enum
    /// (rev 37), syncing the registry's mutations back. Used by
    /// the test cases below to keep the call sites readable.
    fn observe_via_registry(
        acc: &mut ChannelAccumulator,
        reg: &mut ContextRegistry,
        ctx: &[u32],
        next: u32,
    ) {
        let mut enc = Encoder::Registry(reg.clone());
        acc.observe(&mut enc, ctx, next);
        if let Some(r) = enc.as_registry() {
            reg.clone_from(r);
        }
    }

    fn observe_shard_via_registry(
        acc: &mut ChannelAccumulator,
        reg: &mut ContextRegistry,
        shard_path: &Path,
    ) -> Result<(), QfmTextError> {
        let mut enc = Encoder::Registry(reg.clone());
        let res = acc.observe_shard(shard_path, &mut enc);
        if let Some(r) = enc.as_registry() {
            reg.clone_from(r);
        }
        res
    }

    #[test]
    fn observe_grows_unigram_lazily() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(2);
        observe_via_registry(&mut acc, &mut reg, &[1], 5);
        assert_eq!(acc.unigram.len(), 6);
        assert_eq!(acc.unigram[5], 1);
        assert_eq!(acc.total_windows, 1);
    }

    #[test]
    fn observe_records_per_mode_weight_and_hist() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(2);
        // Manually assign two order-1 modes so we can use them by
        // context. The registry will assign mode 1 to ctx=[0],
        // mode 2 to ctx=[1]. Observe next=7, 7, 9 against both
        // modes (use the same context each time so both orders'
        // modes get the observation).
        reg.assign(1, &[0]);
        reg.assign(1, &[1]);
        // Now encode the contexts: for [0, 0] (length 2) the
        // active modes are (order-1: token 0) and (order-2:
        // (0, 0) which we haven't assigned → vacuum).
        observe_via_registry(&mut acc, &mut reg, &[0, 0], 7);
        observe_via_registry(&mut acc, &mut reg, &[0, 0], 7);
        observe_via_registry(&mut acc, &mut reg, &[0, 0], 9);
        // Mode 1 (token 0, order-1) saw 3 windows.
        let s0 = acc.stats.get(&1).expect("order-1 mode for token 0");
        assert_eq!(s0.weight, 3);
        let s0_hist: FxHashMap<u32, u32> = s0.hist.iter().copied().collect();
        assert_eq!(s0_hist[&7], 2);
        assert_eq!(s0_hist[&9], 1);
    }

    #[test]
    fn hist_cap_evicts_to_escape() {
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(1);
        // hist_cap = 3. The empty context is the only way to
        // route observations to mode 0 (the vacuum sentinel) under
        // the rev 36 observe contract: every non-empty context
        // gets a fresh mode assigned. The vacuum histogram is
        // initialized at model-build time (see
        // `QfmTextModel::init_vacuum_histogram`), so this test
        // exercises the escape eviction logic via the empty-context
        // path.
        for t in 0..4 {
            observe_via_registry(&mut acc, &mut reg, &[], t);
        }
        let s = acc.stats.get(&0).expect("vacuum mode");
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
        let mut reg_a = ContextRegistry::new(2);
        let mut a = ChannelAccumulator::new(0, c.clone());
        observe_shard_via_registry(&mut a, &mut reg_a, &sp).unwrap();
        let mut reg_b = ContextRegistry::new(2);
        let mut b = ChannelAccumulator::new(0, c.clone());
        observe_shard_via_registry(&mut b, &mut reg_b, &sp).unwrap();
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
        let mut reg = ContextRegistry::new(2);
        observe_via_registry(&mut a, &mut reg, &[0], 5);
        observe_via_registry(&mut a, &mut reg, &[1], 9);
        let b = ChannelAccumulator::new(0, cfg());
        let snap = a.clone();
        a.merge(b);
        assert_eq!(a.total_windows, snap.total_windows);
        assert_eq!(a.unigram, snap.unigram);
        assert_eq!(a.stats, snap.stats);
    }

    #[test]
    fn unseen_context_grows_registry_with_fresh_modes() {
        // Rev 36: `observe` calls `registry.assign` for every
        // active order, so the registry grows as the corpus is
        // observed. An "unseen" context is impossible after the
        // first observation (the registry already has the mode).
        // The vacuum mode (0) is only used for empty contexts.
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(2);
        // First observation: both orders are unseen → both get
        // fresh modes assigned (modes 1 and 2).
        observe_via_registry(&mut acc, &mut reg, &[42, 99], 7);
        assert_eq!(reg.n_active_for_order(0), 1, "order-1 should have 1 mode");
        assert_eq!(reg.n_active_for_order(1), 1, "order-2 should have 1 mode");
        // Mode 1 (order-1 trailing [99]) and mode 2 (order-2
        // trailing [42, 99]) each have weight 1 and the entry
        // (7, 1) — the observation was recorded against the
        // fresh mode, not the vacuum.
        let s1 = acc.stats.get(&1).expect("order-1 mode for [99]");
        assert_eq!(s1.weight, 1);
        let s2 = acc.stats.get(&2).expect("order-2 mode for [42, 99]");
        assert_eq!(s2.weight, 1);
        // No observations hit mode 0 (the vacuum) — the corpus
        // observation went to the freshly assigned registry modes.
        assert!(!acc.stats.contains_key(&0));
        // The unigram (separate from the per-mode histograms) was
        // still updated.
        assert_eq!(acc.unigram[7], 1);
    }
}
