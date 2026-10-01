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

use super::{segment, tokenize, window_key, EngramKey, Granularity, DEFAULT_WINDOW_N};

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
#[derive(Debug, Clone, Default)]
pub struct EngramTable {
    maps: BTreeMap<Granularity, HashMap<[u8; 32], Embedding>>,
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

    /// Insert, keyed on the granularity plus `unf_hash`. Inserting the same key
    /// twice keeps the first embedding: ingest is a corpus pass, not an update,
    /// and silently overwriting would make the dedup statistics depend on
    /// iteration order.
    pub fn insert(&mut self, key: EngramKey, emb: Embedding) {
        if key.is_fallback() {
            // A fallback key is a real entry (ENGRAM.md §4 stores it), but it
            // is not content-addressed, so it gets its own slot keyed on the
            // surface hash it already carries rather than pretending to be a
            // normal form.
        }
        self.maps
            .entry(key.granularity)
            .or_default()
            .entry(key.unf_hash)
            .or_insert(emb);
    }

    pub fn get(&self, key: &EngramKey) -> Option<&Embedding> {
        self.maps
            .get(&key.granularity)
            .and_then(|m| m.get(&key.unf_hash))
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
            for g in [Granularity::Window, Granularity::Subderiv, Granularity::Sentence] {
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
            stats.dedup_ratio.insert(*g, ratio(cols as f64, *off as f64));
            stats.ted_merge_rate.insert(*g, ratio(tedc as f64, cols as f64));
            stats.parse_rate.insert(*g, ratio((*off - fb) as f64, *off as f64));
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
    if den <= 0.0 {
        0.0
    } else {
        num / den
    }
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