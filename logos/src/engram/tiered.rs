//! E7: the two-tier engram table (ENGRAM.md §2.5).
//!
//! A RAM-hot tier in front of an SSD spill tier, with an explicit prefetch step.
//! The point is not to make the common case faster — the hot tier is the same
//! `HashMap` lookup `EngramTable` already does — but to make the *rare* case
//! bounded: an engram that does not fit in RAM still costs one seek, and a
//! caller that knows which keys it is about to touch can say so and pay that
//! seek once instead of per lookup.
//!
//! Eviction is LRU on a monotonic clock, ordered through a **lazy-deletion
//! min-heap** of `(clock, granularity, unf_hash)`. The heap is what makes this
//! cheap: a linear scan for the coldest entry is O(hot) per eviction, so a
//! corpus larger than the RAM budget turns ingest into O(n²) — measured at
//! 20901% overhead over an untiered baseline before this was a heap, against
//! a 3% target. Popping the coldest is now O(log n).
//!
//! Reads bump an entry's clock without removing its stale heap record; eviction
//! discards a record whose clock no longer matches the live entry (lazy
//! deletion). That is the standard trade: touching an entry can leave one
//! bounded-by-inserts record behind, in exchange for an O(1) read that never
//! touches the heap's interior. Under the ingest workload this table exists
//! for — every key written once, read rarely — the two are indistinguishable.
//!
//! Mass is the invariant that matters and is asserted by the tests: an entry
//! must be in exactly one of the two tiers after any sequence of
//! insert/offload/prefetch, never both and never neither.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::io;
use std::path::Path;

use crate::engram::spill::SpillTier;
use crate::engram::table::Embedding;
use crate::engram::{EngramKey, Granularity};

/// One hot entry, with the clock reading at its last touch.
#[derive(Debug, Clone)]
struct Entry {
    emb: Embedding,
    last: u64,
}

/// What E7's acceptance asks to be reported, plus the counters that make the
/// two numbers explainable rather than merely present.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TierStats {
    /// Entries resident in RAM.
    pub hot: usize,
    /// Entries known to the spill index.
    pub spilled: usize,
    /// Bytes the spill file occupies.
    pub spill_bytes: u64,
    /// How many times a cold entry was evicted to the spill tier.
    pub offload_events: usize,
    /// Entries written to the spill tier.
    pub offloaded_entries: usize,
    /// Prefetch batches requested.
    pub prefetch_batches: usize,
    /// Prefetch requests that found the key in the spill tier.
    pub prefetch_hits: usize,
    /// Prefetch requests that did not.
    pub prefetch_misses: usize,
    /// Lookups served from RAM.
    pub hot_lookups: usize,
    /// Lookups that had to consult the spill tier (a *miss* in the RAM sense,
    /// not an absent key).
    pub cold_lookups: usize,
}

impl TierStats {
    /// The ENGRAM §2.5 number: of the keys a prefetch asked for, how many were
    /// actually recoverable from the spill tier.
    pub fn prefetch_hit_rate(&self) -> f64 {
        let asked = self.prefetch_hits + self.prefetch_misses;
        if asked == 0 {
            return 0.0;
        }
        self.prefetch_hits as f64 / asked as f64
    }

    /// RAM residency as a fraction of all known entries.
    pub fn hot_fraction(&self) -> f64 {
        let total = self.hot + self.spilled;
        if total == 0 {
            return 0.0;
        }
        self.hot as f64 / total as f64
    }
}

/// A RAM-hot tier in front of an SSD spill tier.
pub struct TieredTable {
    hot: BTreeMap<Granularity, HashMap<[u8; 32], Entry>>,
    spill: Option<SpillTier>,
    /// Entries to keep in RAM. Exceeding it triggers one eviction.
    hot_capacity: usize,
    clock: u64,
    /// Min-heap of eviction candidates, oldest first. May contain records
    /// superseded by a later clock bump; `offload_coldest` skips those.
    victims: BinaryHeap<Reverse<(u64, Granularity, [u8; 32])>>,
    stats: TierStats,
}

impl TieredTable {
    /// A table with no spill tier — the degenerate, all-RAM case. Kept so the
    /// offload overhead can be measured against a baseline that differs only in
    /// whether spilling happens.
    pub fn in_memory(hot_capacity: usize) -> Self {
        Self {
            hot: BTreeMap::new(),
            spill: None,
            hot_capacity,
            clock: 0,
            victims: BinaryHeap::new(),
            stats: TierStats::default(),
        }
    }

    /// A table backed by `path`, keeping at most `hot_capacity` in RAM.
    pub fn with_spill(path: &Path, hot_capacity: usize) -> io::Result<Self> {
        Ok(Self {
            hot: BTreeMap::new(),
            spill: Some(SpillTier::open(path)?),
            hot_capacity,
            clock: 0,
            victims: BinaryHeap::new(),
            stats: TierStats::default(),
        })
    }

    /// Entries currently in RAM.
    pub fn hot_len(&self) -> usize {
        self.hot.values().map(|m| m.len()).sum()
    }

    /// Entries currently in RAM for `granularity`.
    pub fn hot_len_at(&self, granularity: Granularity) -> usize {
        self.hot.get(&granularity).map(|m| m.len()).unwrap_or(0)
    }

    pub fn stats(&self) -> TierStats {
        let mut s = self.stats.clone();
        s.hot = self.hot_len();
        s.spilled = self.spill.as_ref().map(|t| t.len()).unwrap_or(0);
        s.spill_bytes = self.spill.as_ref().map(|t| t.bytes_on_disk()).unwrap_or(0);
        s
    }

    /// Insert, evicting to the spill tier if the RAM budget is exceeded.
    ///
    /// A fallback key is inserted like any other. ENGRAM.md §4 stores it, and
    /// it is content-addressed by the surface hash it carries, so there is
    /// nothing special to do here beyond not pretending it is a normal form.
    pub fn insert(&mut self, key: &EngramKey, emb: Embedding) -> io::Result<()> {
        let slot = self.hot.entry(key.granularity).or_default();
        if slot.contains_key(&key.unf_hash) {
            // Same rule as `EngramTable::insert`: first write wins during an
            // ingest pass, so dedup statistics do not depend on ordering.
            return Ok(());
        }
        // The hot check above only asks about RAM. A key that was offloaded and
        // is now re-ingested is in the spill index too, so inserting it here
        // would leave the address in *both* tiers — the mass invariant this
        // module is built around. Drop the spill index entry; the record stays
        // on disk (append-only) and the hot copy is authoritative.
        if let Some(spill) = self.spill.as_mut() {
            spill.forget(key.granularity, &key.unf_hash);
        }
        self.clock += 1;
        let clock = self.clock;
        slot.insert(key.unf_hash, Entry { emb, last: clock });
        self.victims
            .push(Reverse((clock, key.granularity, key.unf_hash)));
        if self.hot_len() > self.hot_capacity {
            self.offload_coldest()?;
        }
        Ok(())
    }

    /// Move a spill-resident entry into the hot tier, respecting the RAM budget.
    ///
    /// Promotion is a write to `hot`, so it has to run the same capacity check
    /// `insert` does. Without it a `prefetch` of the whole key set pulled every
    /// entry into RAM and `hot_capacity` stopped being a bound at all — which
    /// is the whole justification for having two tiers.
    fn promote(&mut self, key: &EngramKey, emb: Embedding) -> io::Result<()> {
        self.clock += 1;
        let last = self.clock;
        self.hot
            .entry(key.granularity)
            .or_default()
            .insert(key.unf_hash, Entry { emb, last });
        self.victims
            .push(Reverse((last, key.granularity, key.unf_hash)));
        if let Some(spill) = self.spill.as_mut() {
            spill.forget(key.granularity, &key.unf_hash);
        }
        if self.hot_len() > self.hot_capacity {
            self.offload_coldest()?;
        }
        Ok(())
    }

    /// Evict the least-recently-touched hot entry to the spill tier.
    fn offload_coldest(&mut self) -> io::Result<()> {
        let Some(spill) = self.spill.as_mut() else {
            // No spill tier: over budget with nowhere to put the victim. Keep
            // it. Dropping it would make an all-RAM table lose data purely
            // because someone set a small capacity.
            return Ok(());
        };
        self.stats.offload_events += 1;

        // Pop the oldest live record. Records superseded by a clock bump are
        // discarded; the live entry they point at is still in the hot tier and
        // will be reached by its own, newer record.
        let mut victim: Option<(Granularity, [u8; 32])> = None;
        while let Some(Reverse((clock, granularity, unf))) = self.victims.pop() {
            let live = self
                .hot
                .get(&granularity)
                .and_then(|m| m.get(&unf))
                .is_some_and(|e| e.last == clock);
            if live {
                victim = Some((granularity, unf));
                break;
            }
        }

        let Some((granularity, unf)) = victim else {
            return Ok(());
        };
        let Some(map) = self.hot.get_mut(&granularity) else {
            return Ok(());
        };
        let Some(entry) = map.remove(&unf) else {
            return Ok(());
        };
        spill.put(granularity, &unf, &entry.emb)?;
        self.stats.offloaded_entries += 1;
        Ok(())
    }

    /// Look up an engram, consulting the spill tier on a RAM miss.
    ///
    /// A spill hit promotes the entry back into RAM: the caller has just paid
    /// for a seek, and the next lookup of the same key should not. Promotion
    /// drops the spill index entry so the address lives in exactly one tier.
    pub fn get(&mut self, key: &EngramKey) -> io::Result<Option<Embedding>> {
        self.clock += 1;
        let clock = self.clock;
        if let Some(entry) = self
            .hot
            .get_mut(&key.granularity)
            .and_then(|m| m.get_mut(&key.unf_hash))
        {
            entry.last = clock;
            self.victims
                .push(Reverse((clock, key.granularity, key.unf_hash)));
            self.stats.hot_lookups += 1;
            return Ok(Some(entry.emb.clone()));
        }
        self.stats.cold_lookups += 1;
        let fetched = match self.spill.as_mut() {
            Some(spill) => spill.get(key.granularity, &key.unf_hash)?,
            None => return Ok(None),
        };
        match fetched {
            Some(emb) => {
                self.promote(key, emb.clone())?;
                Ok(Some(emb))
            }
            None => Ok(None),
        }
    }

    /// Whether the address is known, in either tier, without disturbing either.
    pub fn contains(&self, key: &EngramKey) -> bool {
        if self
            .hot
            .get(&key.granularity)
            .map(|m| m.contains_key(&key.unf_hash))
            .unwrap_or(false)
        {
            return true;
        }
        self.spill
            .as_ref()
            .map(|t| t.indexed(key.granularity, &key.unf_hash))
            .unwrap_or(false)
    }

    /// Pull a batch of keys back into RAM ahead of use — ENGRAM.md §2.5's
    /// prefetch. Returns how many were recovered.
    ///
    /// Batch rather than one-at-a-time because the win is issuing the seeks
    /// while the caller is still doing something else; a caller that knows its
    /// next `k` keys should say so in one call.
    pub fn prefetch(&mut self, keys: &[EngramKey]) -> io::Result<usize> {
        self.stats.prefetch_batches += 1;
        let mut recovered = 0usize;
        for key in keys {
            let already_hot = self
                .hot
                .get(&key.granularity)
                .map(|m| m.contains_key(&key.unf_hash))
                .unwrap_or(false);
            if already_hot {
                // Neither a hit nor a miss: the key was already where the
                // prefetch wanted it, so the spill tier was never consulted.
                // `prefetch_hit_rate` is documented as the fraction of asked
                // keys actually *recoverable from the spill tier*, and counting
                // an already-resident key made the rate read 1.0 for a batch
                // that never touched the heap.
                continue;
            }
            let fetched = match self.spill.as_mut() {
                Some(spill) => spill.get(key.granularity, &key.unf_hash)?,
                None => {
                    self.stats.prefetch_misses += 1;
                    continue;
                }
            };
            match fetched {
                Some(emb) => {
                    self.promote(key, emb)?;
                    self.stats.prefetch_hits += 1;
                    recovered += 1;
                }
                None => self.stats.prefetch_misses += 1,
            }
        }
        Ok(recovered)
    }
}
