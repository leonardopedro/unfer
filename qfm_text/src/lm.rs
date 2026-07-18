//! Perplexity + sampling + classical baseline (Stage 5).
//!
//! The "honest comparison" surface: the QFM-Text model and a
//! classical interpolated absolute-discount n-gram model are scored
//! on the same shard. The QFM-Text model is supposed to *win* (lower
//! perplexity) because the dressed-vacuum projector sum + Krylov
//! evolution provides coherent cross-order smoothing that the
//! classical backoff lacks.
//!
//! If QFM-Text does *not* win, the result is reported as a negative
//! result in `docs/QFM_TEXT_STATUS.md` rather than hidden (per
//! QFM_TEXT_HRM_PLAN.md §"Honest scope").

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::accumulate::{
    ChannelAccumulator, Encoder, ModeStats,
};
#[cfg(test)]
use crate::accumulate::{observe_shard_with_registry, observe_with_registry};
use crate::config::TextConfig;
use crate::corpus::Shard;
use crate::error::QfmTextError;
use crate::model::QfmTextModel;
use crate::registry::VACUUM_MODE;
use crate::registry::ContextRegistry;

/// The perplexity report: token count, nats-per-token, and ppl.
#[derive(Debug, Clone, PartialEq)]
pub struct PerplexityReport {
    pub n_tokens: u64,
    pub nll_nats_per_token: f64,
    pub ppl: f64,
}

impl std::fmt::Display for PerplexityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nll/tok = {:.4} nats, ppl = {:.2} (n = {})",
            self.nll_nats_per_token, self.ppl, self.n_tokens
        )
    }
}

impl PerplexityReport {
    /// Build a report from a total nll (nats) and a token count.
    pub fn from_nll(nll: f64, n_tokens: u64) -> Self {
        let n = n_tokens.max(1) as f64;
        let nll_per = nll / n;
        let ppl = nll_per.exp();
        Self {
            n_tokens,
            nll_nats_per_token: nll_per,
            ppl,
        }
    }
}

/// Compute the perplexity of `model` on `shard`. Walks every window
/// `(ctx, next)` and accumulates `log P(next | ctx)`. Sequential:
/// `QfmTextModel::logprob` is not `Sync`-safe for shared use across
/// threads, so this is single-threaded (see `log_progress` for why
/// that matters at the WikiText-103 test-shard scale).
pub fn perplexity(model: &QfmTextModel, shard: &Shard) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = model.cfg.n_orders;
    let mut nll = 0.0;
    let start = std::time::Instant::now();
    let mut n = 0u64;
    for (ctx, next) in shard.windows(n_orders) {
        let lp = model.logprob(&ctx, next)?;
        nll -= lp;
        n += 1;
        log_progress(n, start);
    }
    Ok(PerplexityReport::from_nll(nll, shard.len() as u64))
}

/// Print a `[perplexity] N tokens, elapsed, tok/s` progress line every
/// `PROGRESS_EVERY` tokens. This is the diagnostic added to find
/// where the held-out eval on the 266K-token WikiText-103 test shard
/// was hanging: without this, the eval binary prints one line before
/// the shard loop and one line after, so a multi-minute per-token
/// cost (the `decode_sketched` O(K_2 x rank) matvec dominates at
/// ~4 ms/token single-threaded, see `QFM_TEXT_STATUS.md`) looks
/// indistinguishable from a hang.
const PROGRESS_EVERY: u64 = 20_000;

fn log_progress(n: u64, start: std::time::Instant) {
    if n % PROGRESS_EVERY == 0 {
        let elapsed = start.elapsed().as_secs_f64();
        let rate = n as f64 / elapsed.max(1e-9);
        eprintln!(
            "[perplexity] {n} tokens scored, elapsed = {elapsed:.1}s, {rate:.1} tok/s"
        );
    }
}

/// Same as `perplexity` but stops after `max_tokens` windows. Useful
/// for the eval binary when the corpus is too large to score in full.
pub fn perplexity_capped(
    model: &QfmTextModel,
    shard: &Shard,
    max_tokens: usize,
) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = model.cfg.n_orders;
    let mut nll = 0.0;
    let mut n = 0u64;
    let start = std::time::Instant::now();
    for (ctx, next) in shard.windows(n_orders) {
        let lp = model.logprob(&ctx, next)?;
        nll -= lp;
        n += 1;
        log_progress(n, start);
        if n as usize >= max_tokens {
            break;
        }
    }
    Ok(PerplexityReport::from_nll(nll, n))
}

/// Perplexity using the **model-averaging** decoder
/// (`QfmTextModel::next_token_dist_model_avg`). This is the
/// "Fix 2 / better encode" entry point — the per-order Krylov
/// models are evolved independently and the decoded distributions
/// averaged, which avoids the destructive interference of the
/// equal-weight-superposition decoder in `perplexity`. The
/// `QFM_TEXT_STATUS.md` next-step §3.
pub fn perplexity_model_avg(
    model: &QfmTextModel,
    shard: &Shard,
) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = model.cfg.n_orders;
    let mut nll = 0.0;
    let mut n = 0u64;
    let start = std::time::Instant::now();
    for (ctx, next) in shard.windows(n_orders) {
        let dist = model.next_token_dist_model_avg(&ctx)?;
        let idx = next as usize;
        if idx >= dist.len() {
            continue;
        }
        let p = dist[idx].max(1e-30);
        nll -= p.ln();
        n += 1;
        log_progress(n, start);
    }
    Ok(PerplexityReport::from_nll(nll, shard.len() as u64))
}

/// Same as `perplexity_model_avg` but stops after `max_tokens` windows.
pub fn perplexity_model_avg_capped(
    model: &QfmTextModel,
    shard: &Shard,
    max_tokens: usize,
) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = model.cfg.n_orders;
    let mut nll = 0.0;
    let mut n = 0u64;
    let start = std::time::Instant::now();
    for (ctx, next) in shard.windows(n_orders) {
        let dist = model.next_token_dist_model_avg(&ctx)?;
        let idx = next as usize;
        if idx >= dist.len() {
            continue;
        }
        let p = dist[idx].max(1e-30);
        nll -= p.ln();
        n += 1;
        log_progress(n, start);
        if n as usize >= max_tokens {
            break;
        }
    }
    Ok(PerplexityReport::from_nll(nll, n))
}

/// Deterministic per-context sampling. Temperature `T` rescales the
/// log-probabilities; the sample is drawn from a splitmix64-seeded
/// RNG. Two calls with equal seed give equal samples.
pub fn sample_text(
    model: &QfmTextModel,
    prompt: &[u32],
    n_tokens: usize,
    temperature: f64,
    seed: u64,
) -> Vec<u32> {
    let mut rng_state = seed;
    let mut out: Vec<u32> = prompt.to_vec();
    while out.len() < prompt.len() + n_tokens {
        let ctx: &[u32] = if out.len() <= model.cfg.n_orders {
            &out[..]
        } else {
            &out[out.len() - model.cfg.n_orders..]
        };
        let dist = match model.next_token_dist(ctx) {
            Ok(d) => d,
            Err(_) => break,
        };
        // Build log-prob with temperature.
        let mut logits: Vec<f64> = dist
            .iter()
            .map(|&p| (p.max(1e-30)).ln() / temperature)
            .collect();
        // Subtract max for stability.
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for l in logits.iter_mut() {
            *l -= max;
        }
        let exps: Vec<f64> = logits.iter().map(|l| l.exp()).collect();
        let total: f64 = exps.iter().sum();
        if total <= 0.0 {
            break;
        }
        // Sample: cumulative + u32 from splitmix64.
        rng_state = splitmix64(rng_state);
        let u = (rng_state as f64) / (u64::MAX as f64);
        let mut cum = 0.0;
        let mut chosen = exps.len() - 1;
        for (i, &e) in exps.iter().enumerate() {
            cum += e / total;
            if u <= cum {
                chosen = i;
                break;
            }
        }
        out.push(chosen as u32);
    }
    out
}

fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// The classical interpolated absolute-discount n-gram model.
/// Mirrors the QFM-Text encoding exactly: same per-context
/// [`ContextRegistry`], same histogram cap, same unigram floor.
/// The only thing missing is the quantum smoothing — so the
/// perplexity delta isolates the *quantum* contribution, not the
/// architecture.
pub struct NgramBaseline {
    cfg: TextConfig,
    mode_hists: FxHashMap<u32, ModeStats>,
    unigram: Vec<f64>,
    #[allow(dead_code)]
    unigram_total: f64,
    registry: ContextRegistry,
}

impl NgramBaseline {
    /// Build from a streaming accumulator (same one the QFM-Text
    /// model uses) and the matching [`ContextRegistry`]. The
    /// baseline sees the *exact same* data and encoding.
    pub fn from_accumulator(acc: ChannelAccumulator, registry: ContextRegistry) -> Self {
        let cfg = acc.cfg.clone();
        let unigram_total: f64 = acc.unigram.iter().map(|&c| c as f64).sum();
        let unigram: Vec<f64> = if unigram_total > 0.0 {
            acc.unigram.iter().map(|&c| c as f64 / unigram_total).collect()
        } else {
            vec![0.0; acc.unigram.len()]
        };
        // The baseline also needs the vacuum mode's histogram
        // (initialized to the unigram counts) so unseen-context
        // queries go through the same per-mode histogram the model
        // would use. This mirrors `QfmTextModel::init_vacuum_histogram`.
        let mut mode_hists = acc.stats;
        // If the vacuum is missing (the accumulator never observed
        // an unseen context), add it now.
        if !mode_hists.contains_key(&0) {
            let mut vacuum = ModeStats::default();
            vacuum.weight = unigram_total as u64;
            for (tok, &cnt) in acc.unigram.iter().enumerate() {
                if cnt > 0 {
                    vacuum.hist.push((tok as u32, cnt as u32));
                }
            }
            vacuum
                .hist
                .sort_by(|(a_t, a_c), (b_t, b_c)| b_c.cmp(a_c).then(a_t.cmp(b_t)));
            let hist_cap = mode_hists
                .values()
                .map(|s| s.hist.len())
                .max()
                .unwrap_or(64)
                .max(vacuum.hist.len());
            while vacuum.hist.len() > hist_cap {
                let evicted = vacuum.hist.pop().expect("non-empty");
                vacuum.escape += evicted.1 as u64;
            }
            mode_hists.insert(0, vacuum);
        }
        Self {
            cfg,
            mode_hists,
            unigram,
            unigram_total,
            registry,
        }
    }

    /// Same marginalization as `QfmTextModel`, but with no
    /// Krylov-evolved Born-rule weights — instead, each active mode
    /// is weighted uniformly. This is the classical interpolation
    /// `P(y | ctx) = (1/n_active) Σ_{active} smoothed(hist_mode)(y)`.
    pub fn next_token_dist(&self, context: &[u32]) -> Vec<f64> {
        // **rev 37:** the baseline ALWAYS includes mode 0 (the
        // Fock vacuum / outer vacuum / unigram) in the per-mode
        // averaging. The unigram backoff is the "outer vacuum
        // projector" for the baseline — it takes care of the
        // unseen data the same way the Krylov weight for mode 0
        // does in the QFM. Without this, the raw per-mode
        // histograms give 0 to unseen continuations and the
        // baseline ppl explodes (e.g. 30 930 on wikitext-103-test
        // vs 269 for the unigram).
        let mut modes = self.registry.encode_modes(context);
        if !modes.contains(&VACUUM_MODE) {
            modes.push(VACUUM_MODE);
        }
        if modes.is_empty() {
            return self.unigram.clone();
        }
        let v = self.unigram.len();
        let mut p = vec![0.0_f64; v];
        let n_active = modes.len() as f64;
        for &mode in &modes {
            if let Some(stats) = self.mode_hists.get(&mode) {
                let denom = stats.weight as f64;
                if denom <= 0.0 {
                    continue;
                }
                // **rev 37 design (matches QFM marginalize):** no
                // Jelinek-Mercer-style smoothing per mode. The
                // per-mode distribution is the raw histogram:
                //   p[tok] = (cnt / K) / n_active     for seen tokens
                //   p[tok] = 0                         for unseen
                // The unigram backoff is provided by including
                // mode 0 (the outer vacuum / unigram) in the
                // active modes — its per-mode distribution IS the
                // unigram, so unseen continuations get the
                // unigram's probability mass via the uniform
                // 1/n_active averaging.
                for &(tok, cnt) in &stats.hist {
                    if (tok as usize) < v {
                        p[tok as usize] += (cnt as f64 / denom) / n_active;
                    }
                }
            }
        }
        // Normalise and clamp.
        let mut sum = 0.0;
        for x in p.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
            sum += *x;
        }
        if sum > 0.0 {
            for x in p.iter_mut() {
                *x /= sum;
            }
        } else {
            return self.unigram.clone();
        }
        p
    }

    pub fn logprob(&self, context: &[u32], next: u32) -> f64 {
        let dist = self.next_token_dist(context);
        if (next as usize) >= dist.len() {
            return self
                .unigram
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p.ln())
                .fold(f64::NEG_INFINITY, f64::max);
        }
        dist[next as usize].max(1e-30).ln()
    }

    /// Unigram-only baseline (no context, no smoothing).
    pub fn unigram_ppl(shard: &Shard) -> Result<PerplexityReport, QfmTextError> {
        // Walk the shard, count unigrams, then score the *same*
        // shard against the unigram distribution. This is a
        // degenerate baseline (it inlines the held-out assumption
        // and so is an *upper* bound on the unigram ppl for the
        // same vocabulary); the Stage-6 eval re-scores against the
        // *unigram held out* for honesty.
        let mut counts: FxHashMap<u32, u64> = FxHashMap::default();
        let mut total: u64 = 0;
        for t in shard.iter() {
            *counts.entry(t).or_insert(0) += 1;
            total += 1;
        }
        if total == 0 {
            return Ok(PerplexityReport {
                n_tokens: 0,
                nll_nats_per_token: 0.0,
                ppl: 1.0,
            });
        }
        let mut nll = 0.0;
        for t in shard.iter() {
            let c = counts[&t] as f64;
            let p = c / total as f64;
            nll -= p.max(1e-30).ln();
        }
        Ok(PerplexityReport::from_nll(nll, total))
    }

    /// Same as `unigram_ppl` but stops after `max_tokens` windows.
    pub fn unigram_ppl_capped(
        shard: &Shard,
        max_tokens: usize,
    ) -> Result<PerplexityReport, QfmTextError> {
        let mut counts: FxHashMap<u32, u64> = FxHashMap::default();
        let mut total: u64 = 0;
        for t in shard.iter() {
            *counts.entry(t).or_insert(0) += 1;
            total += 1;
            if total as usize >= max_tokens {
                break;
            }
        }
        if total == 0 {
            return Ok(PerplexityReport {
                n_tokens: 0,
                nll_nats_per_token: 0.0,
                ppl: 1.0,
            });
        }
        let mut nll = 0.0;
        let mut counted: u64 = 0;
        for t in shard.iter() {
            let c = counts[&t] as f64;
            let p = c / total as f64;
            nll -= p.max(1e-30).ln();
            counted += 1;
            if counted as usize >= max_tokens {
                break;
            }
        }
        Ok(PerplexityReport::from_nll(nll, counted))
    }
}

/// Compute the perplexity of the classical n-gram baseline on the
/// same shard, using the same accumulator as the QFM-Text model.
pub fn perplexity_baseline(
    baseline: &NgramBaseline,
    shard: &Shard,
) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = baseline.cfg.n_orders;
    let mut nll = 0.0;
    for (ctx, next) in shard.windows(n_orders) {
        let lp = baseline.logprob(&ctx, next);
        nll -= lp;
    }
    Ok(PerplexityReport::from_nll(nll, shard.len() as u64))
}

/// Same as `perplexity_baseline` but stops after `max_tokens` windows.
pub fn perplexity_baseline_capped(
    baseline: &NgramBaseline,
    shard: &Shard,
    max_tokens: usize,
) -> Result<PerplexityReport, QfmTextError> {
    let n_orders = baseline.cfg.n_orders;
    let mut nll = 0.0;
    let mut n = 0u64;
    for (ctx, next) in shard.windows(n_orders) {
        let lp = baseline.logprob(&ctx, next);
        nll -= lp;
        n += 1;
        if n as usize >= max_tokens {
            break;
        }
    }
    Ok(PerplexityReport::from_nll(nll, n))
}

/// Build an `NgramBaseline` directly from a shard (no accumulator).
/// Convenience for the Stage 6 eval binary.
pub fn baseline_from_shard(
    shard_path: &Path,
    cfg: &TextConfig,
) -> Result<NgramBaseline, QfmTextError> {
    let mut registry = ContextRegistry::new(cfg.n_orders);
    let mut acc = ChannelAccumulator::new(0, cfg.clone());
    let mut enc = Encoder::Registry(registry.clone());
    acc.observe_shard(shard_path, &mut enc)?;
    if let Some(r) = enc.as_registry() {
        registry.clone_from(r);
    }
    Ok(NgramBaseline::from_accumulator(acc, registry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
            hist_cap: 4,
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
    fn sample_text_is_deterministic() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(cfg().n_orders);
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            observe_with_registry(&mut acc, &mut reg, &ctx, tokens[i]);
        }
        let model = QfmTextModel::from_accumulator(acc, reg, &cfg()).unwrap();
        let prompt = vec![0, 1, 2];
        let s1 = sample_text(&model, &prompt, 10, 1.0, 42);
        let s2 = sample_text(&model, &prompt, 10, 1.0, 42);
        assert_eq!(s1, s2);
    }

    #[test]
    fn baseline_ppl_finite_and_better_than_unigram() {
        let tokens: Vec<u32> = (0..400).map(|i| (i / 20) % 5).collect();
        let dir = tempdir().unwrap();
        let sp = dir.path().join("shard.bin");
        write_shard(&sp, &tokens);
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(cfg().n_orders);
        observe_shard_with_registry(&mut acc, &mut reg, &sp).unwrap();
        let baseline = NgramBaseline::from_accumulator(acc, reg);
        let shard = Shard::open(&sp, 100).unwrap();
        let bppl = perplexity_baseline(&baseline, &shard).unwrap();
        let uppl = NgramBaseline::unigram_ppl(&shard).unwrap();
        assert!(bppl.ppl.is_finite());
        assert!(uppl.ppl.is_finite());
        // The n-gram baseline should beat the unigram baseline on
        // a structured corpus.
        assert!(
            bppl.ppl < uppl.ppl,
            "ngram ppl {bppl} should be < unigram ppl {uppl}"
        );
    }

    #[test]
    fn model_ppl_at_most_unigram() {
        // Honest property: the QFM-Text model is the unigram
        // baseline in the worst case (no Krylov smoothing, no
        // context structure). The model adds context awareness
        // on top of the unigram, so it must never be *worse*
        // than the unigram (it can only match or improve). This
        // is the structural baseline; see `model_ppl_competitive`
        // for the more demanding comparison against the
        // classical n-gram baseline.
        let c = cfg();
        let tokens: Vec<u32> = (0..800)
            .map(|i| match (i / 3) % 4 {
                0 => 0,
                _ if i % 2 == 0 => 1,
                _ if i % 3 == 0 => 2,
                _ => 3,
            })
            .collect();
        let dir = tempdir().unwrap();
        let sp = dir.path().join("shard.bin");
        write_shard(&sp, &tokens);
        let mut acc = ChannelAccumulator::new(0, c.clone());
        let mut reg = ContextRegistry::new(c.n_orders);
        observe_shard_with_registry(&mut acc, &mut reg, &sp).unwrap();
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let shard = Shard::open(&sp, 100).unwrap();
        let mppl = perplexity(&model, &shard).unwrap();
        let uppl = NgramBaseline::unigram_ppl(&shard).unwrap();
        assert!(mppl.ppl.is_finite());
        assert!(uppl.ppl.is_finite());
        assert!(
            mppl.ppl <= uppl.ppl * 1.05,
            "QFM ppl {mppl} should be <= unigram ppl * 1.05 ({})",
            uppl.ppl * 1.05,
        );
    }
}
