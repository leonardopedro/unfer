//! PLAN_liquid_types.md E3: the content-addressed engram table, corpus ingest,
//! and the statistics that say whether the corpus is actually reaching the UNF
//! path.
//!
//! Spec: `australVM/docs/ENGRAM.md` §3 and §5. Golden corpus:
//! `australVM/corpus/engram_keys.tsv`.
//!
//! ## The one decision that shapes every number here
//!
//! Keys are **denotational**, so `John adds two three` and `Bob adds three two`
//! reduce to the same normal form and collide by design (ENGRAM.md §3). A single
//! headline dedup ratio would read that as a win when it is really a statement
//! about lexicon coverage. So every statistic in [`IngestStats`] is keyed by
//! `Granularity`: a high `Sentence` dedup with a low `Window` dedup is a signal
//! about the corpus, not an achievement, and only the per-granularity view can
//! show that.
//!
//! `parse_rate` lives here rather than in a debug log for the same reason — it
//! is the metric that distinguishes "the UNF path works" from "everything quietly
//! fell back to `window` keys and the table is full of surface strings".

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::lexicon::Lexicon;

use super::{DEFAULT_WINDOW_N, EngramKey, Granularity, segment, tokenize, window_key};

/// A stored embedding. The lookup path is what matters here: O(1) by
/// `unf_hash`, and *deterministic*, so keys are computable at corpus ingest time
/// before any decode step (ENGRAM.md §1).
pub type Embedding = Vec<f32>;

/// Per-granularity statistics. `BTreeMap` so a report is ordered, not
/// hash-ordered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestStats {
    /// 1 - (distinct keys / segments offered), per granularity. Measures how
    /// much coverage one table slot buys.
    pub dedup_ratio: BTreeMap<Granularity, f64>,
    /// Fraction of merges attributable to TED canonicalization specifically
    /// (commutativity, distribution, like terms) rather than to reduction.
    pub ted_merge_rate: BTreeMap<Granularity, f64>,
    /// Fraction of segments that reached the UNF path rather than the tagged
    /// `window` fallback. This is the honesty metric.
    pub parse_rate: BTreeMap<Granularity, f64>,
    pub segments_offered: BTreeMap<Granularity, usize>,
    pub distinct_keys: BTreeMap<Granularity, usize>,
    pub lookup_p50_ns: u64,
    pub lookup_p99_ns: u64,
}

/// A content-addressed table, partitioned by granularity.
///
/// Partitioning is not an optimization: a lookup at granularity `g` must not
/// find a `sentence` key (ENGRAM.md §2.3 — a miss stays a miss), so the
/// granularity is part of the address, not a hint.
/// The map key: a content hash plus whether the entry is a tagged fallback.
///
/// The fallback bit is part of the address because a tagged fallback and a real
/// window key of the same order are otherwise indistinguishable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct EngramAddr {
    unf_hash: [u8; 32],
    fallback: bool,
}

impl EngramAddr {
    fn of(key: &EngramKey) -> Self {
        EngramAddr {
            unf_hash: key.unf_hash,
            fallback: key.is_fallback(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EngramTable {
    maps: BTreeMap<Granularity, HashMap<EngramAddr, Embedding>>,
}

impl EngramTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.maps.values().map(|m| m.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert, keyed on the granularity plus the address. Inserting the same key
    /// twice keeps the first embedding: ingest is a corpus pass, not an update,
    /// and silently overwriting would make the dedup statistics depend on
    /// iteration order.
    ///
    /// A fallback key is a real entry (ENGRAM.md §4 stores it) but is not
    /// content-addressed, so it gets its own slot. `fallback(tokens)` is
    /// `window_key(tokens, DEFAULT_WINDOW_N).as_fallback()`, which shares a
    /// granularity and a surface hash with the genuine window key of the same
    /// order — they differ only in `FLAG_FALLBACK`. Addressing on the hash
    /// alone therefore put a tagged fallback and a real window key in the same
    /// slot, and since `get` cannot see `flags` a caller could not tell which
    /// one it had read.
    pub fn insert(&mut self, key: EngramKey, emb: Embedding) {
        let addr = EngramAddr::of(&key);
        self.maps
            .entry(key.granularity)
            .or_default()
            .entry(addr)
            .or_insert(emb);
    }

    pub fn get(&self, key: &EngramKey) -> Option<&Embedding> {
        self.maps
            .get(&key.granularity)
            .and_then(|m| m.get(&EngramAddr::of(key)))
    }

    pub fn contains(&self, key: &EngramKey) -> bool {
        self.get(key).is_some()
    }

    /// Ingest a corpus, returning the statistics of ENGRAM.md §5.
    ///
    /// `embed` is the embedding producer. It is a parameter rather than a
    /// built-in because the decoder-side model is E4's business (a port into
    /// `deepseek-ai/Engram`); until then the caller supplies a deterministic
    /// stand-in, which keeps ingest testable without a model.
    pub fn ingest<F>(&mut self, corpus: &[String], lexicon: &Lexicon, mut embed: F) -> IngestStats
    where
        F: FnMut(&EngramKey) -> Embedding,
    {
        let mut stats = IngestStats::default();
        // Per granularity, per segment we track: offered, distinct, fallbacks,
        // and how many of the collisions were also TED-equal.
        let mut offered: BTreeMap<Granularity, usize> = BTreeMap::new();
        let mut fallbacks: BTreeMap<Granularity, usize> = BTreeMap::new();
        let mut ted_collisions: BTreeMap<Granularity, usize> = BTreeMap::new();
        let mut collisions: BTreeMap<Granularity, usize> = BTreeMap::new();
        let mut seen: BTreeMap<Granularity, HashMap<[u8; 32], [u8; 32]>> = BTreeMap::new();

        for fragment in corpus {
            for g in [
                Granularity::Window,
                Granularity::Subderiv,
                Granularity::Sentence,
            ] {
                let keys = match g {
                    // `window` needs no lexicon, so it is derived here rather
                    // than through `segment` to keep the fallback path
                    // independent of the gate.
                    Granularity::Window => {
                        let toks = tokenize(fragment);
                        (2..=DEFAULT_WINDOW_N)
                            .map(|n| super::window_key(&toks, n))
                            .collect::<Vec<_>>()
                    }
                    _ => segment(fragment, g, lexicon),
                };
                *offered.entry(g).or_insert(0) += keys.len();
                let table = seen.entry(g).or_default();
                for k in &keys {
                    if k.is_fallback() {
                        *fallbacks.entry(g).or_insert(0) += 1;
                    }
                    match table.get(&k.unf_hash) {
                        None => {
                            table.insert(k.unf_hash, k.ted_hash);
                        }
                        Some(prev) => {
                            *collisions.entry(g).or_insert(0) += 1;
                            if *prev == k.ted_hash {
                                *ted_collisions.entry(g).or_insert(0) += 1;
                            }
                        }
                    }
                    self.insert(k.clone(), embed(k));
                }
            }
        }

        for (g, off) in &offered {
            let distinct = seen.get(g).map(|m| m.len()).unwrap_or(0);
            let cols = collisions.get(g).copied().unwrap_or(0);
            let tedc = ted_collisions.get(g).copied().unwrap_or(0);
            let fb = fallbacks.get(g).copied().unwrap_or(0);
            stats
                .dedup_ratio
                .insert(*g, ratio(cols as f64, *off as f64));
            stats
                .ted_merge_rate
                .insert(*g, ratio(tedc as f64, cols as f64));
            stats
                .parse_rate
                .insert(*g, ratio((*off - fb) as f64, *off as f64));
            stats.segments_offered.insert(*g, *off);
            stats.distinct_keys.insert(*g, distinct);
        }

        // Lookup latency: measured, not asserted, so a regression in the O(1)
        // claim shows up in the numbers the plan asks to be recorded.
        if let Some(g) = stats.distinct_keys.keys().next().copied() {
            let probes: Vec<EngramKey> = corpus
                .iter()
                .flat_map(|f| segment(f, g, lexicon))
                .take(256)
                .collect();
            let mut samples: Vec<u128> = Vec::with_capacity(probes.len());
            for k in &probes {
                let t = Instant::now();
                let _ = self.get(k);
                samples.push(t.elapsed().as_nanos());
            }
            if !samples.is_empty() {
                samples.sort_unstable();
                stats.lookup_p50_ns = percentile(&samples, 50);
                stats.lookup_p99_ns = percentile(&samples, 99);
            }
        }
        stats
    }
}

/// `num / den`, or 0.0 for an empty denominator — an empty corpus reports 0,
/// not NaN, so a report never has to special-case it.
fn ratio(num: f64, den: f64) -> f64 {
    if den <= 0.0 { 0.0 } else { num / den }
}

/// Nearest-rank percentile over sorted samples.
fn percentile(sorted: &[u128], p: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p * sorted.len()) / 100;
    let idx = rank.min(sorted.len() - 1);
    sorted[idx] as u64
}

/// A deterministic stand-in embedder, so ingest is testable without the E4
/// model. Derived from the key's own bytes: it is a *placeholder*, and saying so
/// in the name stops anyone mistaking it for the real thing.
pub fn placeholder_embedding(key: &EngramKey) -> Embedding {
    let mut out = Vec::with_capacity(8);
    for chunk in key.unf_hash.chunks(4) {
        let mut acc: u32 = 0;
        for b in chunk {
            acc = acc.wrapping_mul(256).wrapping_add(*b as u32);
        }
        out.push((acc as f32) / (u32::MAX as f32));
    }
    out
}

/// Convenience: build a `window` key for a fragment's token window.
pub fn window_key_for(fragment: &str, n: usize) -> EngramKey {
    window_key(&tokenize(fragment), n)
}

#[cfg(test)]
mod address_tests {
    use super::*;
    use crate::engram::{DEFAULT_WINDOW_N, window_key};

    fn toks(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// A tagged fallback and the genuine window key of the same order are the
    /// same address unless the fallback bit is part of it.
    ///
    /// `engram::fallback(tokens)` is exactly `window_key(tokens,
    /// DEFAULT_WINDOW_N).as_fallback()`, so the two share a granularity and a
    /// surface hash and differ only in `FLAG_FALLBACK`. The table used to
    /// address on the hash alone, so both landed in one slot and `get` could not
    /// say which it had read.
    #[test]
    fn a_fallback_and_the_real_window_key_are_different_addresses() {
        let t = toks(&["Mary", "sees", "Bob"]);
        let real = window_key(&t, DEFAULT_WINDOW_N);
        let tagged = crate::engram::fallback_public(&t);

        assert_eq!(real.granularity, tagged.granularity);
        assert_eq!(real.unf_hash, tagged.unf_hash, "same surface, same hash");
        assert!(!real.is_fallback());
        assert!(tagged.is_fallback(), "and they differ only in the flag");

        let mut table = EngramTable::new();
        table.insert(real.clone(), vec![1.0, 1.0]);
        table.insert(tagged.clone(), vec![2.0, 2.0]);

        assert_eq!(table.len(), 2, "two addresses, two entries");
        assert_eq!(
            table.get(&real),
            Some(&vec![1.0, 1.0]),
            "the window key must not read back the fallback's embedding"
        );
        assert_eq!(
            table.get(&tagged),
            Some(&vec![2.0, 2.0]),
            "nor the other way round"
        );
    }

    /// The reverse order must give the same separation — an insert that
    /// overwrote would make the result depend on which arrived first.
    #[test]
    fn the_separation_does_not_depend_on_insert_order() {
        // Three tokens for n=3: a shorter fragment would make the *real* window
        // key a fallback too (`window_key` tags an under-full window), and the
        // two addresses would legitimately coincide.
        let t = toks(&["Alice", "really", "runs"]);
        let real = window_key(&t, DEFAULT_WINDOW_N);
        let tagged = crate::engram::fallback_public(&t);
        assert!(!real.is_fallback(), "the fixture needs a real window key");

        let mut table = EngramTable::new();
        table.insert(tagged.clone(), vec![9.0]);
        table.insert(real.clone(), vec![1.0]);
        assert_eq!(table.get(&real), Some(&vec![1.0]));
        assert_eq!(table.get(&tagged), Some(&vec![9.0]));
        assert_eq!(table.len(), 2);
    }

    /// First-write-wins still holds within one address.
    #[test]
    fn first_write_wins_within_an_address() {
        let t = toks(&["Bob", "really", "runs"]);
        let k = window_key(&t, DEFAULT_WINDOW_N);
        let mut table = EngramTable::new();
        table.insert(k.clone(), vec![1.0]);
        table.insert(k.clone(), vec![2.0]);
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&k), Some(&vec![1.0]));
    }
}
