//! Per-context mode registry — the **hashing-free** Level-2 encoder
//! for QFM-Text (Stage 2, rev 36).
//!
//! Where [`OrderHasher`](crate::features::OrderHasher) (rev ≤ 35) mapped
//! every context of order `o` to a *bounded* hash slot in
//! `[offset_o, offset_o + block_size_o)`, `ContextRegistry` assigns
//! a **fresh mode index** to every distinct context the corpus
//! actually produces. There is no collision: a context's mode
//! stores only that context's own next-token histogram.
//!
//! Trade-off: the W matrix is now `(K_2_total, rank)` where
//! `K_2_total = 1 + Σ_o n_active_modes_o` is the actual vocabulary
//! of unique contexts, not a fixed budget. With ~5.5 M active 4-gram
//! contexts in the 128 M-token WikiText-103 train corpus and
//! `rank = 8`, W is 350 MB on disk (vs 16 MB at the prior
//! `block_sizes = [65536; 4]`). Worth it: the prior LLM diagnosed
//! that hash collisions were the dominant cause of the fit ceiling
//! (QFM_TEXT_STATUS.md rev 35) and the only honest fix is to drop
//! the bounded table.
//!
//! # Vacuum mode for unseen contexts
//!
//! A test-time context that was never seen in training has no entry
//! in the registry. The QFM Hamiltonian
//!   `H = Σ_o λ_o |0̃_o⟩⟨0̃_o|`
//! still applies — the dressed-vacuum projector |0̃_o⟩ is the uniform
//! superposition over the order-o modes that the registry did
//! collect, and its Krylov evolution gives the natural backoff
//! distribution for an unseen order-o context. To route this through
//! the standard `encode_modes` / Krylov / marginalize pipeline (so
//! the vacuum projector contributes exactly as the H-matrix says it
//! should, instead of bypassing the Krylov machinery and jumping
//! straight to the unigram), the registry reserves **mode index 0**
//! as the per-order "vacuum" sentinel:
//!
//! - `encode_modes(seen_ctx)` returns the registry-assigned mode per
//!   active order.
//! - `encode_modes(unseen_ctx)` returns `[0]` (one per active order
//!   — the Krylov machinery collapses the equal-weight superposition
//!   to a single mode, mode 0).
//!
//! The accumulator / model initializes `mode_hists[0]` to the
//! unigram counts so that mode 0's histogram *is* the unigram floor
//! the QFM-Text pipeline's `marginalize` step already distributes
//! mass to. The dressed-vacuum projector in H then mixes the
//! unigram-histogram mode 0 with the registry-assigned modes'
//! histograms through the Krylov evolution — the QFM-mandated
//! fallback for unseen contexts, not a degenerate return-unigram.

use rustc_hash::FxHashMap;

/// Sentinel mode index for "context not in the training registry".
/// The Krylov pipeline treats this as the vacuum mode: its
/// histogram is the unigram (set by [`super::model::QfmTextModel::from_accumulator`]),
/// and the dressed-vacuum projector |0̃_o⟩ in H mixes it with the
/// registry-assigned modes' histograms through the Krylov evolution.
pub const VACUUM_MODE: u32 = 0;

/// Per-order mode registry: one `FxHashMap<Vec<u32>, u32>` per order
/// `o in 1..=n_orders`. The map key is the trailing-`o` context
/// (a `Vec<u32>`), the value is the assigned global mode index
/// `≥ 1` (mode 0 is reserved as the vacuum sentinel).
///
/// The registry is **built during the streaming pass**
/// (`accumulate_shards` takes `&mut ContextRegistry`) and then
/// **frozen**: after training, the set of assigned mode indices is
/// fixed, and the `ChannelAccumulator`'s `stats` map only ever uses
/// those indices as keys.
///
/// Cloning a registry is cheap (it's just `Vec<FxHashMap>`), so the
/// compiled `QfmTextModel` can own one and the `NgramBaseline` can
/// own a clone without an extra allocation.
#[derive(Debug, Clone)]
pub struct ContextRegistry {
    /// One map per order, indexed `o - 1` for order `o in 1..=n_orders`.
    /// `maps[o - 1][ctx_tokens] = global_mode_index`.
    pub maps: Vec<FxHashMap<Vec<u32>, u32>>,
    /// Cumulative starting index of each order's block in the
    /// global mode space. `offsets[o] = 1 + Σ_{i<o} maps[i].len()`.
    /// `offsets[0] = 1` always (mode 0 is the vacuum).
    offsets: Vec<u32>,
    /// Total K_2 (mode-space) dimension: `1 + Σ_o maps[o - 1].len()`.
    k2_total: u32,
}

impl ContextRegistry {
    /// Create a fresh registry for the given `n_orders`. All maps
    /// start empty; mode indices are assigned on the first
    /// observation of each unique context.
    pub fn new(n_orders: usize) -> Self {
        Self {
            maps: (0..n_orders).map(|_| FxHashMap::default()).collect(),
            offsets: vec![1; n_orders], // recomputed lazily
            k2_total: 1,                // 1 for the vacuum
        }
    }

    /// Total K_2 (mode-space) dimension. The Krylov pipeline's W
    /// matrix has this many rows. Includes the vacuum (mode 0).
    pub fn k2_total(&self) -> u32 {
        self.k2_total
    }

    /// Number of orders in the registry.
    pub fn n_orders(&self) -> usize {
        self.maps.len()
    }

    /// Number of active (seen) modes for order `o` (0-indexed).
    pub fn n_active_for_order(&self, order: usize) -> usize {
        self.maps.get(order).map(|m| m.len()).unwrap_or(0)
    }

    /// Cumulative starting index of order `o`'s block. Order 0
    /// starts at 1 (mode 0 is vacuum). Order `o` starts at
    /// `1 + Σ_{i<o} n_active_for_order(i)`.
    pub fn offset(&self, order: usize) -> u32 {
        self.offsets[order]
    }

    /// Which order does this mode index belong to? Mode 0 is the
    /// vacuum (returns `n_orders` so it sorts above every real
    /// order; callers that need "no order" check for `mode == 0`).
    /// Out-of-range modes also return `n_orders`.
    pub fn order_of(&self, mode: u32) -> usize {
        if mode == 0 {
            return self.maps.len();
        }
        for (o, &off) in self.offsets.iter().enumerate() {
            let block = self.maps[o].len() as u32;
            if mode < off + block {
                return o;
            }
        }
        self.maps.len()
    }

    /// Look up the mode index for a context of order `o` (1-indexed
    /// — order 0 doesn't exist). Returns `None` if the context has
    /// not been registered. Does **not** assign a new index.
    pub fn lookup(&self, order: usize, context: &[u32]) -> Option<u32> {
        if order == 0 || context.len() < order {
            return None;
        }
        let key = &context[context.len() - order..];
        self.maps[order - 1].get(key).copied()
    }

    /// Assign a mode index to a context of order `o`, registering
    /// it if new. Returns the mode index. The accumulator is the
    /// only intended caller (during the streaming pass).
    pub fn assign(&mut self, order: usize, context: &[u32]) -> u32 {
        debug_assert!(order >= 1, "order must be >= 1");
        debug_assert!(context.len() >= order, "context too short for order");
        let key: Vec<u32> = context[context.len() - order..].to_vec();
        if let Some(&m) = self.maps[order - 1].get(&key) {
            return m;
        }
        let new_mode = self.k2_total;
        self.maps[order - 1].insert(key, new_mode);
        self.k2_total += 1;
        // Recompute offsets for orders >= this one. O(n_orders)
        // which is fine (n_orders <= 8 typically).
        let mut off = 1u32;
        for o in 0..self.maps.len() {
            self.offsets[o] = off;
            off += self.maps[o].len() as u32;
        }
        new_mode
    }

    /// Encode the active-mode list for a context. Returns one mode
    /// per active order `o in 1..=min(context.len(), n_orders)`.
    /// For **unseen** contexts (none of the trailing-token slices
    /// are in the registry), returns a single-element vector
    /// `[VACUUM_MODE]` (the Krylov machinery collapses the
    /// equal-weight superposition across orders to one mode — the
    /// vacuum).
    pub fn encode_modes(&self, context: &[u32]) -> Vec<u32> {
        let n = self.maps.len();
        if context.is_empty() || n == 0 {
            // No context or no orders → only the vacuum is available.
            return vec![VACUUM_MODE];
        }
        let n_active_orders = n.min(context.len());
        let mut out: Vec<u32> = Vec::with_capacity(n_active_orders);
        let mut any_seen = false;
        for o in 1..=n_active_orders {
            let key = &context[context.len() - o..];
            if let Some(&m) = self.maps[o - 1].get(key) {
                out.push(m);
                any_seen = true;
            } else {
                out.push(VACUUM_MODE);
            }
        }
        if !any_seen {
            // All orders are unseen: collapse to a single vacuum
            // mode (the Krylov machinery's equal-weight superposition
            // of multiple identical vacuum modes is the same as one
            // vacuum mode).
            return vec![VACUUM_MODE];
        }
        out
    }

    /// Read-only access to the per-order maps (for serialization).
    pub fn maps(&self) -> &[FxHashMap<Vec<u32>, u32>] {
        &self.maps
    }

    /// Reconstruct a registry from a previously saved per-order map
    /// list. Recomputes `offsets` and `k2_total` from the maps
    /// (so the on-disk format is just the maps; offsets are
    /// derived).
    pub fn from_maps(maps: Vec<FxHashMap<Vec<u32>, u32>>) -> Self {
        let n_orders = maps.len();
        let mut offsets = vec![0u32; n_orders];
        let mut off = 1u32;
        let mut total = 1u32; // vacuum
        for o in 0..n_orders {
            offsets[o] = off;
            off += maps[o].len() as u32;
            total += maps[o].len() as u32;
        }
        Self {
            maps,
            offsets,
            k2_total: total,
        }
    }
}

impl Default for ContextRegistry {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Iterate over `(order, last_o_tokens)` for every active order of a
/// context. Mirrors the pre-rev-36 `context_orders` helper (which
/// lived in `features.rs` alongside the `OrderHasher`) so callers
/// that just need the trailing slices (e.g. tests) don't have to
/// know the registry's internal layout. Stops at
/// `min(context.len(), n_orders)`.
pub fn context_orders<'a>(
    registry: &ContextRegistry,
    context: &'a [u32],
) -> impl Iterator<Item = (usize, &'a [u32])> + 'a {
    let n = registry.n_orders().min(context.len());
    (1..=n).map(move |o| {
        let start = context.len() - o;
        (o, &context[start..])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> ContextRegistry {
        ContextRegistry::new(3)
    }

    #[test]
    fn fresh_registry_has_only_vacuum() {
        let r = reg();
        assert_eq!(r.k2_total(), 1);
        assert_eq!(r.n_orders(), 3);
        for o in 0..3 {
            assert_eq!(r.n_active_for_order(o), 0);
            assert_eq!(r.offset(o), 1);
        }
        assert_eq!(r.order_of(0), 3); // vacuum
    }

    #[test]
    fn assign_gives_distinct_modes_for_distinct_contexts() {
        let mut r = reg();
        let m1 = r.assign(1, &[10]);
        let m2 = r.assign(1, &[20]);
        let m3 = r.assign(2, &[10, 20]);
        let m4 = r.assign(2, &[10, 30]);
        assert_ne!(m1, m2);
        assert_ne!(m3, m4);
        assert_ne!(m1, m3);
        // Offsets advance with the per-order count.
        assert_eq!(r.offset(0), 1);
        assert_eq!(r.offset(1), 3); // 2 order-1 modes + vacuum
        assert_eq!(r.offset(2), 5); // 2 order-2 modes
        assert_eq!(r.k2_total(), 5);
    }

    #[test]
    fn assign_reuses_mode_for_repeated_context() {
        let mut r = reg();
        let m1 = r.assign(2, &[10, 20]);
        let m2 = r.assign(2, &[10, 20]);
        let m3 = r.assign(2, &[10, 20]);
        assert_eq!(m1, m2);
        assert_eq!(m2, m3);
        assert_eq!(r.n_active_for_order(1), 1);
    }

    #[test]
    fn lookup_returns_none_for_unseen() {
        let mut r = reg();
        r.assign(1, &[10]);
        assert!(r.lookup(1, &[10]).is_some());
        assert!(r.lookup(1, &[20]).is_none());
        assert!(r.lookup(2, &[10, 20]).is_none());
    }

    #[test]
    fn encode_modes_returns_one_per_active_order_for_seen() {
        let mut r = reg();
        // For a 3-token context [10, 20, 30] the registry's
        // encode_modes looks up the trailing-o tokens for each
        // active order `o in 1..=3`: order 1 → [30], order 2 →
        // [20, 30], order 3 → [10, 20, 30]. Register all three.
        r.assign(1, &[30]);
        r.assign(2, &[20, 30]);
        r.assign(3, &[10, 20, 30]);
        // All three orders are seen → three distinct modes.
        let m = r.encode_modes(&[10, 20, 30]);
        assert_eq!(m.len(), 3);
        // Each is distinct (different orders → different modes).
        assert_ne!(m[0], m[1]);
        assert_ne!(m[1], m[2]);
        assert_ne!(m[0], m[2]);
    }

    #[test]
    fn encode_modes_returns_vacuum_for_unseen() {
        let mut r = reg();
        r.assign(1, &[10]);
        // [99, 99] is unseen at every active order.
        let m = r.encode_modes(&[99, 99]);
        assert_eq!(m, vec![VACUUM_MODE]);
    }

    #[test]
    fn encode_modes_partial_unseen_keeps_seen_and_inserts_vacuum() {
        let mut r = reg();
        // For a 3-token context [10, 20, 30] the registry looks
        // up the trailing-o tokens for each active order:
        // order 1 → [30], order 2 → [20, 30], order 3 →
        // [10, 20, 30]. Register order-1 (token 30) and order-2
        // (bigram [20, 30]) but leave order-3 ([10, 20, 30])
        // unseen. Then `encode_modes([10, 20, 30])` should
        // return [m_order1, m_order2, VACUUM_MODE] — the seen
        // modes plus the vacuum fallback for the unseen order.
        r.assign(1, &[30]);
        r.assign(2, &[20, 30]);
        let m = r.encode_modes(&[10, 20, 30]);
        assert_eq!(m.len(), 3, "partial-unseen should keep all per-order slots");
        // Order-1 mode and order-2 mode are real (>= 1).
        assert!(m[0] >= 1, "order-1 mode should be a real mode");
        assert!(m[1] >= 1, "order-2 mode should be a real mode");
        // Order-3 (unseen) is the vacuum sentinel.
        assert_eq!(m[2], VACUUM_MODE, "order-3 should fall back to vacuum");
    }

    #[test]
    fn encode_modes_empty_context_returns_vacuum() {
        let r = reg();
        assert_eq!(r.encode_modes(&[]), vec![VACUUM_MODE]);
    }

    #[test]
    fn order_of_round_trips() {
        let mut r = reg();
        let m1 = r.assign(1, &[10]);
        let m2 = r.assign(2, &[10, 20]);
        let m3 = r.assign(3, &[10, 20, 30]);
        assert_eq!(r.order_of(m1), 0);
        assert_eq!(r.order_of(m2), 1);
        assert_eq!(r.order_of(m3), 2);
        assert_eq!(r.order_of(0), 3); // vacuum
    }

    #[test]
    fn from_maps_round_trip() {
        let mut r = reg();
        r.assign(1, &[1]);
        r.assign(1, &[2]);
        r.assign(2, &[1, 2]);
        let maps = r.maps().to_vec();
        let total = r.k2_total();
        let r2 = ContextRegistry::from_maps(maps);
        assert_eq!(r2.k2_total(), total);
        assert_eq!(r2.n_active_for_order(0), 2);
        assert_eq!(r2.n_active_for_order(1), 1);
        assert_eq!(r2.n_active_for_order(2), 0);
        // The same lookups work.
        assert_eq!(r2.lookup(1, &[1]), r.lookup(1, &[1]));
        assert_eq!(r2.lookup(2, &[1, 2]), r.lookup(2, &[1, 2]));
    }
}
