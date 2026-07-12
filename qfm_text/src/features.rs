//! Hashed Level-2 feature encoder for token contexts (Stage 2).
//!
//! The QFM-Text analog of `qfm::sketch::FeatureToMode`. The image S₂
//! assigns a fresh mode per unique feature (a `FeatureToMode::register`
//! call), which is unbounded at corpus scale. The text S₂ is a *fixed*
//! hashed table: every context of order `o` maps to a deterministic
//! mode in `[offset_o, offset_o + block_size_o)`, regardless of
//! whether the corpus has seen the context before. This is the
//! standard hashed-LM trade-off: hash collisions blend histograms
//! of unrelated contexts, but memory is bounded and the model can
//! generalize to unseen contexts at test time.
//!
//! Concretely:
//!   `mode_for(order, ctx) = offset_o + splitmix64_seq(last o tokens,
//!                                                     salt_o) % block_size_o`
//! where `splitmix64_seq` is a 64-bit-folded FNV-style mixer of the
//! token sequence. The salt per order decorrelates the hashes so a
//! context of length 2 doesn't get the same mode as a context of
//! length 3 with the same trailing 2 tokens.

use rustc_hash::FxHashMap;

use crate::config::TextConfig;

/// Mix a u64 into a new u64 via the splitmix64 finalizer. Same mixer
/// as `qfm::sketch::splitmix64` — duplicated here to avoid a
/// cross-crate public API.
pub fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Fold a sequence of u32 tokens into a single u64 by repeatedly
/// mixing each token into a running hash. Deterministic; stable
/// across processes for the same input sequence.
pub fn splitmix64_seq(tokens: &[u32], salt: u64) -> u64 {
    let mut h: u64 = salt.wrapping_add(0xcbf29ce484222325);
    for &t in tokens {
        h = splitmix64(h ^ (t as u64).wrapping_mul(0x100000001b3));
    }
    h
}

/// Hashed Level-2 encoder: maps a context of `o` tokens to a mode
/// in `[offset_o, offset_o + block_size_o)`. Stateless after
/// construction; clone-friendly.
#[derive(Debug, Clone)]
pub struct OrderHasher {
    cfg: TextConfig,
}

impl OrderHasher {
    /// Build a hasher from a `TextConfig`. Validates that the per-order
    /// `block_sizes`, `salts`, and `lambda` vectors have length
    /// `n_orders`.
    pub fn new(cfg: TextConfig) -> Self {
        let n = cfg.n_orders;
        assert_eq!(cfg.block_sizes.len(), n, "block_sizes must have n_orders entries");
        assert_eq!(cfg.salts.len(), n, "salts must have n_orders entries");
        assert_eq!(cfg.lambda.len(), n, "lambda must have n_orders entries");
        for (o, &b) in cfg.block_sizes.iter().enumerate() {
            assert!(b > 0, "block_sizes[{o}] must be > 0");
        }
        Self { cfg }
    }

    /// The order-`o` mode for a context. Returns `None` if
    /// `context.len() < order` (the order is *not active* for this
    /// window — the accumulator simply skips it). Otherwise returns
    /// `Some(offset_o + (splitmix64_seq(last o tokens, salt_o) %
    /// block_sizes[o]))`.
    pub fn mode_for(&self, order: usize, context: &[u32]) -> Option<u32> {
        if order == 0 || context.len() < order {
            return None;
        }
        let block = self.cfg.block_sizes[order - 1] as u64;
        let salt = self.cfg.salts[order - 1];
        let last_o = &context[context.len() - order..];
        let h = splitmix64_seq(last_o, salt);
        let mode_in_block = (h % block) as u32;
        Some(self.cfg.offset(order - 1) + mode_in_block)
    }

    /// Encode the full active-mode list for a context. The returned
    /// vector contains the mode for each active order `o in
    /// 1..=min(context.len(), n_orders)`, in order. `context.len() = 0`
    /// gives an empty list (no information to condition on).
    pub fn encode_modes(&self, context: &[u32]) -> Vec<u32> {
        (1..=self.cfg.n_orders)
            .filter_map(|o| self.mode_for(o, context))
            .collect()
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }
}

/// Iterate over `(order, last_o_tokens)` for every active order of a
/// context. Useful for testing the hasher without rebuilding the
/// `Vec<u32>` of modes. Stops at `min(context.len(), n_orders)`.
pub fn context_orders<'a>(
    cfg: &TextConfig,
    context: &'a [u32],
) -> impl Iterator<Item = (usize, &'a [u32])> + 'a {
    let n = cfg.n_orders.min(context.len());
    (1..=n).map(move |o| {
        let start = context.len() - o;
        (o, &context[start..])
    })
}

/// Quick statistics on the hashed encoding for a synthetic corpus:
/// how many collisions a flat hashed encoding would produce. Useful
/// for tuning block sizes.
#[derive(Debug, Default, Clone)]
pub struct CollisionStats {
    /// `mode -> count of contexts that mapped to it`.
    pub counts: FxHashMap<u32, u64>,
}

impl CollisionStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one context.
    pub fn record(&mut self, modes: &[u32]) {
        for &m in modes {
            *self.counts.entry(m).or_insert(0) += 1;
        }
    }

    /// Maximum occupancy of any mode.
    pub fn max_occupancy(&self) -> u64 {
        self.counts.values().copied().max().unwrap_or(0)
    }

    /// Number of distinct modes observed.
    pub fn n_distinct(&self) -> usize {
        self.counts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 4,
            block_sizes: vec![16, 16, 16, 16],
            salts: vec![1, 2, 3, 4],
            ..Default::default()
        }
    }

    #[test]
    fn splitmix64_zero_is_not_zero() {
        // splitmix64(0) is actually 0 (every multiply is by 0).
        // The function only becomes interesting with a non-zero
        // seed. splitmix64(1) is the simplest non-zero seed and
        // mixes to a non-trivial 64-bit value.
        assert_eq!(splitmix64(0), 0);
        assert_ne!(splitmix64(1), 0);
        assert_ne!(splitmix64(1), splitmix64(2));
    }

    #[test]
    fn splitmix64_seq_deterministic() {
        let a = splitmix64_seq(&[1, 2, 3], 7);
        let b = splitmix64_seq(&[1, 2, 3], 7);
        assert_eq!(a, b);
    }

    #[test]
    fn splitmix64_seq_changes_with_salt() {
        let a = splitmix64_seq(&[1, 2, 3], 7);
        let b = splitmix64_seq(&[1, 2, 3], 8);
        assert_ne!(a, b);
    }

    #[test]
    fn splitmix64_seq_changes_with_length() {
        let a = splitmix64_seq(&[1, 2], 7);
        let b = splitmix64_seq(&[1, 2, 3], 7);
        assert_ne!(a, b);
    }

    #[test]
    fn mode_for_short_context_returns_none() {
        let h = OrderHasher::new(cfg());
        assert!(h.mode_for(1, &[]).is_none());
        assert!(h.mode_for(3, &[1, 2]).is_none());
        assert_eq!(h.mode_for(1, &[42]).unwrap(), h.cfg.offset(0) + (splitmix64_seq(&[42], 1) % 16) as u32);
    }

    #[test]
    fn mode_for_inside_correct_block() {
        let h = OrderHasher::new(cfg());
        for o in 1..=4 {
            let block = h.cfg.block_sizes[o - 1];
            let off = h.cfg.offset(o - 1);
            for trial in 0..50 {
                let ctx: Vec<u32> = (0..o as u32 + trial).collect();
                let m = h.mode_for(o, &ctx).unwrap();
                assert!(
                    m >= off && (m as u64) < (off as u64) + (block as u64),
                    "order {o}: mode {m} not in [{off}, {off}+{block})",
                );
            }
        }
    }

    #[test]
    fn encode_modes_returns_one_per_active_order() {
        let h = OrderHasher::new(cfg());
        assert!(h.encode_modes(&[]).is_empty());
        assert_eq!(h.encode_modes(&[10]).len(), 1);
        assert_eq!(h.encode_modes(&[10, 20]).len(), 2);
        assert_eq!(h.encode_modes(&[10, 20, 30]).len(), 3);
        assert_eq!(h.encode_modes(&[10, 20, 30, 40]).len(), 4);
        assert_eq!(h.encode_modes(&[10, 20, 30, 40, 50]).len(), 4);
    }

    #[test]
    fn salts_decorrelate_orders() {
        // A context [1, 2, 3] should hash to different modes in
        // orders 1, 2, 3 — the per-order salts decorrelate them.
        let h = OrderHasher::new(cfg());
        let m1 = h.mode_for(1, &[1, 2, 3]).unwrap();
        let m2 = h.mode_for(2, &[1, 2, 3]).unwrap();
        let m3 = h.mode_for(3, &[1, 2, 3]).unwrap();
        // They might collide by chance, but with 16-mode blocks and
        // a 64-bit hash this is overwhelming unlikely.
        assert_ne!(m1, m2);
        assert_ne!(m2, m3);
        assert_ne!(m1, m3);
    }

    #[test]
    fn context_orders_helper() {
        let c = cfg();
        let orders: Vec<_> = context_orders(&c, &[10, 20, 30]).collect();
        assert_eq!(orders, vec![(1, &[30][..]), (2, &[20, 30][..]), (3, &[10, 20, 30][..])]);
    }
}
