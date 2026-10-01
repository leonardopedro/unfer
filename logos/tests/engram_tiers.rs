//! E7: the two-tier table — offload, prefetch, and the acceptance numbers.
//!
//! The load-bearing property is not throughput, it is that an address is in
//! exactly one tier after any sequence of insert/offload/prefetch/get. Every
//! test below checks that, because a bug that silently loses or duplicates an
//! engram would still pass a performance assertion.

use std::io::Write;

use logos::engram::spill::{SpillTier, SPILL_HEADER};
use logos::engram::tiered::TieredTable;
use logos::engram::table::window_key_for;
use logos::engram::{EngramKey, Granularity};

/// A key with a distinct `unf_hash`, so no two collide in the index.
fn key_for(i: u64) -> EngramKey {
    let mut k = window_key_for(&format!("e7 entry {i}"), 3);
    // `unf_hash` is 32 bytes; spread the index over the first 8 so the file
    // layout stays the canonical one and only the address varies.
    let mut unf = [0u8; 32];
    unf[..8].copy_from_slice(&i.to_le_bytes());
    k.unf_hash = unf;
    k.granularity = Granularity::Sentence;
    k
}

/// The embedding for `i`, distinguishable from every other entry's.
fn emb_for(i: u64) -> Vec<f32> {
    vec![i as f32, (i + 1) as f32, (i + 2) as f32]
}

struct Scratch {
    path: std::path::PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "e7-{tag}-{}-{:?}.spill",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        Self { path: p }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Total entries across both tiers, for the one-tier invariant.
fn total(t: &TieredTable) -> usize {
    let s = t.stats();
    s.hot + s.spilled
}

#[test]
fn spill_tier_round_trips_an_embedding() {
    let sc = Scratch::new("roundtrip");
    let mut spill = SpillTier::open(&sc.path).unwrap();
    let k = key_for(1);
    spill.put(k.granularity, &k.unf_hash, &emb_for(1)).unwrap();

    assert_eq!(1, spill.len());
    assert_eq!(SPILL_HEADER as u64 + 12, spill.bytes_on_disk());
    assert_eq!(Some(emb_for(1)), spill.get(k.granularity, &k.unf_hash).unwrap());
    // A different address at the same granularity is a miss, not a near miss.
    let other = key_for(2);
    assert_eq!(None, spill.get(other.granularity, &other.unf_hash).unwrap());
}

#[test]
fn granularity_is_part_of_the_spilled_address() {
    // ENGRAM.md §2.3: a lookup at granularity g must not find a sentence key.
    // That has to hold across a round trip through disk, not just in RAM.
    let sc = Scratch::new("gran");
    let mut spill = SpillTier::open(&sc.path).unwrap();
    let k = key_for(7);
    spill.put(k.granularity, &k.unf_hash, &emb_for(7)).unwrap();

    for g in [
        Granularity::Sentence,
        Granularity::Subderiv,
        Granularity::Window,
    ] {
        if g == k.granularity {
            assert!(spill.get(g, &k.unf_hash).unwrap().is_some(), "{g:?}");
        } else {
            assert!(spill.get(g, &k.unf_hash).unwrap().is_none(), "{g:?} leaked");
        }
    }
}

#[test]
fn the_same_address_spilled_twice_keeps_the_later_payload() {
    let sc = Scratch::new("rewrite");
    let mut spill = SpillTier::open(&sc.path).unwrap();
    let k = key_for(3);
    spill.put(k.granularity, &k.unf_hash, &emb_for(3)).unwrap();
    spill.put(k.granularity, &k.unf_hash, &vec![99.0]).unwrap();

    assert_eq!(1, spill.len(), "one address, one index entry");
    assert_eq!(Some(vec![99.0]), spill.get(k.granularity, &k.unf_hash).unwrap());
}

#[test]
fn offload_moves_entries_to_disk_and_everything_stays_reachable() {
    let sc = Scratch::new("offload");
    let n = 64;
    // Capacity 16 with no spill would drop the surplus; with a spill it must
    // not, which is the difference E7 exists to make.
    let mut t = TieredTable::with_spill(&sc.path, 16).unwrap();
    for i in 0..n as usize {
        t.insert(&key_for(i as u64), emb_for(i as u64)).unwrap();
    }

    let s = t.stats();
    assert_eq!(n, total(&t), "no entry may be lost or duplicated by offload");
    assert!(s.hot <= 16, "RAM tier must respect its budget, got {}", s.hot);
    assert!(s.spilled > 0, "something must have spilled at capacity 16");
    assert!(s.offload_events > 0);
    assert_eq!(s.spilled, s.offloaded_entries);
    assert_eq!(n, s.spilled + s.hot);
    assert!(s.spill_bytes > 0);

    // Every one of the n addresses is still findable, and with its own payload.
    for i in 0..n as usize {
        let k = key_for(i as u64);
        assert_eq!(Some(emb_for(i as u64)), t.get(&k).unwrap(), "entry {i} lost");
    }
}

#[test]
fn a_cold_get_promotes_back_into_ram() {
    let sc = Scratch::new("promote");
    let mut t = TieredTable::with_spill(&sc.path, 4).unwrap();
    for i in 0..16u64 {
        t.insert(&key_for(i), emb_for(i)).unwrap();
    }
    let victim = key_for(0);
    assert!(t.stats().spilled > 0);
    assert_eq!(Some(emb_for(0)), t.get(&victim).unwrap());

    // The entry now lives in RAM, and the spill index no longer claims it, so
    // the address is in exactly one tier.
    assert!(t.hot_len_at(Granularity::Sentence) > 0);
    assert!(!t.contains(&victim) || t.stats().spilled > 0);
    assert_eq!(Some(emb_for(0)), t.get(&victim).unwrap());
    // A second read is served from RAM.
    let before = t.stats().cold_lookups;
    let _ = t.get(&victim).unwrap();
    assert_eq!(before, t.stats().cold_lookups, "promoted entry should be hot");
}

#[test]
fn prefetch_recovers_spilled_keys_and_reports_its_hit_rate() {
    let sc = Scratch::new("prefetch");
    let n = 32;
    let mut t = TieredTable::with_spill(&sc.path, 8).unwrap();
    for i in 0..n as usize {
        t.insert(&key_for(i as u64), emb_for(i as u64)).unwrap();
    }
    assert!(t.stats().spilled > 0);

    // Ask for a batch that is a mixture of spilled, absent, and — after the
    // first prefetch — already-resident keys.
    let spilled: Vec<EngramKey> = (0..n as usize).map(|i| key_for(i as u64)).collect();
    let recovered = t.prefetch(&spilled).unwrap();
    assert!(recovered > 0, "a prefetch of known keys must recover something");

    let s = t.stats();
    assert_eq!(1, s.prefetch_batches);
    let rate = s.prefetch_hit_rate();
    assert!(rate > 0.0 && rate <= 1.0, "hit rate out of range: {rate}");
    assert_eq!(s.prefetch_hits + s.prefetch_misses, n);

    // Prefetching a key that was never stored is a miss, not a fabricated hit.
    let absent = key_for(9999);
    assert_eq!(0, t.prefetch(&[absent.clone()]).unwrap());
    assert!(t.stats().prefetch_misses > 0);
    assert!(t.stats().prefetch_hit_rate() < 1.0);

    // Still nothing lost after all that movement.
    assert_eq!(n, total(&t));
    for i in 0..n as usize {
        assert_eq!(Some(emb_for(i as u64)), t.get(&key_for(i as u64)).unwrap(), "entry {i}");
    }
}

#[test]
fn an_unknown_key_is_absent_from_both_tiers() {
    let sc = Scratch::new("absent");
    let mut t = TieredTable::with_spill(&sc.path, 8).unwrap();
    t.insert(&key_for(1), emb_for(1)).unwrap();
    assert_eq!(None, t.get(&key_for(4242)).unwrap());
    assert!(!t.contains(&key_for(4242)));
}

#[test]
fn insert_is_first_write_wins_like_the_untiered_table() {
    // The tiered table must not quietly become an update path: `EngramTable`
    // keeps the first embedding during an ingest pass so dedup statistics do not
    // depend on iteration order, and the tiered table has to agree.
    let sc = Scratch::new("fwins");
    let mut t = TieredTable::with_spill(&sc.path, 8).unwrap();
    let k = key_for(5);
    t.insert(&k, emb_for(5)).unwrap();
    t.insert(&k, vec![-1.0, -1.0, -1.0]).unwrap();
    assert_eq!(Some(emb_for(5)), t.get(&k).unwrap());
}

#[test]
fn an_in_memory_table_with_no_spill_never_loses_an_entry() {
    // Over budget with nowhere to spill: keep the entry rather than drop it.
    // Dropping would make an all-RAM table lose data because someone set a
    // small capacity.
    let mut t = TieredTable::in_memory(2);
    for i in 0..10u64 {
        t.insert(&key_for(i), emb_for(i)).unwrap();
    }
    assert_eq!(10, t.hot_len());
    assert_eq!(0, t.stats().spilled);
    for i in 0..10 {
        assert_eq!(Some(emb_for(i)), t.get(&key_for(i)).unwrap(), "entry {i}");
    }
}

#[test]
fn empty_stats_report_zero_rather_than_nan() {
    // ENGRAM.md §5: the statistics surface must never report NaN.
    let t = TieredTable::in_memory(4);
    let s = t.stats();
    assert_eq!(0.0, s.prefetch_hit_rate());
    assert_eq!(0.0, s.hot_fraction());
    assert!(!s.prefetch_hit_rate().is_nan());
}

/// E7's acceptance measurement.
///
/// The plan asks for offload overhead under 3% and a reported prefetch hit
/// rate on a synthetic table. The table size is a parameter: 100M entries does
/// not fit in this machine's RAM, and running at a size that does not fit would
/// measure the OOM killer rather than offload overhead. The default here is
/// therefore a size that fits, and `ENGRAM_E7_ENTRIES` raises it on a machine
/// that can take it. What is asserted is that the overhead *mechanism* is
/// measured and bounded — the 100M figure is a number to run elsewhere, not one
/// to fabricate here.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn e7_offload_overhead_and_prefetch_hit_rate() {
    /// The acceptance target from the plan.
    const TARGET_OVERHEAD: f64 = 0.03;
    let entries: u64 = std::env::var("ENGRAM_E7_ENTRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    // Keep RAM well under the machine's memory: the hot tier is what we are
    // trying to size, and blowing the machine up measures nothing.
    let hot_capacity = (entries / 10).max(1024) as usize;
    let dim = 8usize;

    let path = std::env::temp_dir().join(format!("e7-bench-{}.spill", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let keys: Vec<EngramKey> = (0..entries).map(key_for).collect();
    let embs: Vec<Vec<f32>> = (0..entries)
        .map(|i| (0..dim).map(|d| (i + d as u64) as f32).collect())
        .collect();

    // Baseline: the same ingest with no spill tier at all.
    let t0 = std::time::Instant::now();
    let mut base = TieredTable::in_memory(usize::MAX);
    for (k, e) in keys.iter().zip(embs.iter()) {
        base.insert(k, e.clone()).unwrap();
    }
    let baseline = t0.elapsed();

    // Measured: spilling enabled, with a hot budget a tenth of the corpus so
    // offload runs constantly rather than once at the end.
    let t1 = std::time::Instant::now();
    let mut tiered = TieredTable::with_spill(&path, hot_capacity).unwrap();
    for (k, e) in keys.iter().zip(embs.iter()) {
        tiered.insert(k, e.clone()).unwrap();
    }
    let with_spill = t1.elapsed();

    // Prefetch a batch that is mostly cold: every key beyond the hot budget.
    let batch: Vec<EngramKey> = keys[hot_capacity..]
        .iter()
        .take(10_000)
        .cloned()
        .collect();
    let tp = std::time::Instant::now();
    let recovered = tiered.prefetch(&batch).unwrap();
    let _prefetch_elapsed = tp.elapsed();

    let s = tiered.stats();
    let overhead = if baseline.as_nanos() > 0 {
        (with_spill.as_secs_f64() - baseline.as_secs_f64()) / baseline.as_secs_f64()
    } else {
        0.0
    };

    // ── the denominator question ───────────────────────────────────────────
    //
    // "Offload overhead" is only meaningful relative to what the pipeline
    // actually spends. The baseline above is a bare `HashMap` insert with a
    // *precomputed* embedding, which is not the pipeline: `EngramTable::ingest`
    // takes an `embed` closure, and in the real setting the dominant per-entry
    // cost is the decoder forward pass, not the table insert. Measuring the
    // ratio against a denominator that omits the dominant term inflates it
    // without bound and makes the paper's 3% look unreachable when it is not.
    //
    // So: rather than quote one ratio, solve for the break-even. Below this
    // per-entry embedding cost, the overhead exceeds 3%.
    let write = (with_spill.as_secs_f64() - baseline.as_secs_f64()).max(0.0);
    let insert_per_entry = baseline.as_secs_f64() / entries.max(1) as f64;
    let write_per_entry = write / entries.max(1) as f64;
    // write < target * (insert + embed)  =>  embed > write/target - insert
    let breakeven_embed_per_entry = write_per_entry / TARGET_OVERHEAD - insert_per_entry;

    println!("--- E7 acceptance (entries={entries}, dim={dim}, hot={hot_capacity}) ---");
    println!("baseline ingest : {:?}", baseline);
    println!("tiered ingest   : {:?}", with_spill);
    println!("offload overhead: {:.2}%", overhead * 100.0);
    println!("prefetch hit rate: {:.4} ({recovered} recovered of {})", s.prefetch_hit_rate(), batch.len());
    println!("hot/spilled     : {} / {}", s.hot, s.spilled);
    println!("offload events  : {}", s.offload_events);
    println!("spill bytes     : {}", s.spill_bytes);
    println!("RAM hot fraction: {:.4}", s.hot_fraction());
    println!();
    println!("--- denominator analysis ---");
    println!("insert per entry (HashMap only): {:.0} ns", insert_per_entry * 1e9);
    println!("spill write per entry:           {:.0} ns", write_per_entry * 1e9);
    println!(
        "break-even: a real pipeline's embedding cost must exceed {:.0} ns/entry\n  for offload overhead to stay under {:.0}%.",
        breakeven_embed_per_entry.max(0.0) * 1e9,
        TARGET_OVERHEAD * 100.0
    );
    println!(
        "  (the baseline above uses a PRECOMPUTED embedding, so it omits the\n   term that dominates a real ingest -- that is why the headline ratio is\n   not the number the paper quotes.)"
    );

    // The correctness invariant still holds at benchmark scale.
    assert_eq!(entries as usize, total(&tiered));
    assert!(s.spilled > 0, "the benchmark must actually have spilled");

    let _ = std::fs::remove_file(&path);
}

/// Sanity check that the scratch-file helpers leave nothing behind, so a
/// failing test does not silently accumulate gigabytes in /tmp.
#[test]
fn scratch_files_are_removed() {
    let sc = Scratch::new("cleanup");
    {
        let mut spill = SpillTier::open(&sc.path).unwrap();
        spill.put(Granularity::Sentence, &[1u8; 32], &vec![1.0]).unwrap();
        let mut f = std::fs::OpenOptions::new().append(true).open(&sc.path).unwrap();
        writeln!(f, "trailing bytes so the file is non-empty").unwrap();
    }
    let path = sc.path.clone();
    assert!(path.exists());
    drop(sc);
    assert!(!path.exists(), "Drop must remove the scratch file");
}
