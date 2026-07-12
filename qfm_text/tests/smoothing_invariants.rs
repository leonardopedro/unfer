//! Test: verify the per-mode histogram + smoothing invariants
//! that BOTH the QFM and the NgramBaseline depend on. If any
//! invariant is violated, the held-out ppl is wrong (and the
//! bug affects both QFM and baseline).
use qfm_text::accumulate::ModeStats;

#[test]
fn hist_cap_eviction_preserves_total_count() {
    // Simulate a mode observed many times with more unique tokens
    // than hist_cap. Verify the invariant
    //   weight = sum(hist counts) + escape
    // holds.
    let mut s = ModeStats::default();
    let cap = 4;
    // 10 unique tokens, each observed 1 time.
    for t in 0..10u32 {
        s.observe(t, cap);
    }
    let hist_sum: u64 = s.hist.iter().map(|(_, c)| *c as u64).sum();
    let total = hist_sum + s.escape;
    assert_eq!(
        s.weight, total,
        "invariant weight = sum(hist) + escape violated: \
         weight={}, sum(hist)={}, escape={}",
        s.weight, hist_sum, s.escape
    );
    // The histogram has at most cap entries.
    assert!(s.hist.len() <= cap);
    // The escape has at least the number of evictions.
    assert!(s.escape > 0);
    eprintln!(
        "hist_cap_eviction_preserves_total_count: K={}, hist.len()={}, hist_sum={}, escape={}",
        s.weight, s.hist.len(), hist_sum, s.escape
    );
}

#[test]
fn hist_reobserve_after_eviction_loses_history() {
    // After a token is evicted, re-observing it adds count 1
    // (not the previous count). The previous count is in escape.
    // This means the per-token probability for an evicted-then-
    // readded token is wrong (the per-mode distribution uses
    // count=1, not count=1+previous_count).
    let mut s = ModeStats::default();
    let cap = 2;
    // First, fill with 3 observations, then re-observe the first.
    s.observe(100, cap);
    s.observe(200, cap);
    s.observe(300, cap);  // evicts 100 (count 1), adds 300 (count 1)
    // Now re-observe 100. It's not in the histogram, so the
    // algorithm evicts some entry and adds 100 with count 1.
    s.observe(100, cap);
    // Find 100 in the histogram.
    let cnt_100 = s.hist.iter().find(|(t, _)| *t == 100).map(|(_, c)| *c).unwrap_or(0);
    // 100 was observed twice (once initially, once after eviction).
    // But the histogram only has count 1 (the second observation).
    // The first count is in escape.
    eprintln!(
        "hist_reobserve_after_eviction_loses_history: weight={}, \
         hist = {:?}, escape = {}",
        s.weight, s.hist, s.escape
    );
    assert_eq!(cnt_100, 1, "100 should have count 1 in histogram (not 2)");
    // The escape has 1 (the first observation of 100).
    assert!(s.escape >= 1, "escape should have at least 1");
}

#[test]
fn smoothing_per_mode_total_is_one() {
    // The smoothing formula
    //   p_seen[tok]   = (cnt - d) / K + escape_mass * unigram[tok]
    //   p_unseen[tok] = escape_mass * unigram[tok]
    //   escape_mass   = (N * d + escape) / K
    // should give per-mode total = 1 (a proper distribution).
    let d = 0.75_f64;
    let unigram: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
    let unigram_sum: f64 = unigram.iter().sum();
    assert!((unigram_sum - 1.0).abs() < 1e-9, "test bug: unigram must sum to 1");
    // A mode observed K=10 times with 4 unique tokens (cnt = 3, 3, 2, 2).
    let hist: Vec<(usize, f64)> = vec![(0, 3.0), (1, 3.0), (2, 2.0), (3, 2.0)];
    let escape = 0.0_f64;
    let k = 10.0_f64;
    let n = hist.len() as f64;
    let escape_mass = (n * d + escape) / k;
    let mut p = vec![0.0_f64; unigram.len()];
    for &(tok, cnt) in &hist {
        p[tok] += (cnt - d) / k;
    }
    for i in 0..unigram.len() {
        p[i] += escape_mass * unigram[i];
    }
    let total: f64 = p.iter().sum();
    eprintln!(
        "smoothing_per_mode_total_is_one: per-mode total = {:.6} (expected 1.0)",
        total
    );
    assert!((total - 1.0).abs() < 1e-9, "per-mode total != 1: got {}", total);
}

#[test]
fn unseen_tokens_get_less_mass_than_unigram() {
    // The current Jelinek-Mercer-style smoothing gives unseen
    // tokens escape_mass * unigram[tok], which is LESS than
    // unigram[tok] alone (when escape_mass < 1). The unigram
    // is the "no context" baseline; the per-mode distribution
    // should not give unseen tokens LESS mass than the unigram.
    let d = 0.75_f64;
    let unigram = 0.001_f64;  // a rare token
    let k = 10.0_f64;
    let n = 1.0_f64;  // 1 unique token seen
    let escape_mass = (n * d + 0.0) / k;  // no escape
    let p_unseen = escape_mass * unigram;
    eprintln!(
        "unseen_tokens_get_less_mass_than_unigram: \
         p_unseen = {:.6e}, unigram = {:.6e}, escape_mass = {:.3}",
        p_unseen, unigram, escape_mass
    );
    // The per-mode distribution gives the unseen token
    // p_unseen = 0.075 * 0.001 = 7.5e-5, vs unigram = 0.001.
    // The per-mode is 13x LESS than the unigram.
    assert!(p_unseen < unigram, "p_unseen should be < unigram");
}
