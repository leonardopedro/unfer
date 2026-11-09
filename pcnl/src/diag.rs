//! Phase 4 — MCMC health checks.
//!
//! Two tiers, deliberately kept apart:
//!
//! * `ess` / `gelman_rubin` — the initial-positive-sequence ESS and the
//!   split Gelman–Rubin statistic on **raw** draws, which is what every table
//!   in `docs/PCNL_STATUS.md` reports.  They are cheap and comparable with the
//!   recorded numbers.
//! * `split_rhat_rank`, `ess_bulk`, `ess_tail` — the rank-normalised
//!   split/folded `R̂` and the bulk/tail ESS of
//!   [Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008), which do not
//!   assume near-normal draws and are the form `RESEARCH_PLAN.md` §R0.3 names
//!   as the gate.
//!
//! **A frozen trace is a failure, not a good score.**  Before 2026-11-08 a
//! constant chain reported `ESS = N` (every draw "agrees") and two identical
//! constant chains reported `R̂ = 1` (no between-chain variance, no
//! within-chain variance, nothing to disagree about) — exactly the two numbers
//! a stuck chain should never receive.  Both now return `NaN`
//! ("unavailable"), [`movement`] reports the evidence, and the distinction
//! between a frozen and a merely sticky chain is pinned by tests.  Chains
//! frozen at *different* values still return `inf` from `gelman_rubin`: the
//! between-chain spread is real, and infinity is the honest report.
//!
//! Any non-finite draw makes every diagnostic here `NaN` rather than
//! propagating a plausible-looking number.

/// Proof that a trace moved at all, and how much.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Movement {
    /// Chains supplied.
    pub chains: usize,
    /// Chains whose finite draws are all identical (0 or 1 distinct value).
    pub frozen: usize,
    /// Largest number of distinct finite values in any one chain.
    pub unique_max: usize,
    /// Non-finite draws seen across all chains.
    pub nonfinite: usize,
}

impl Movement {
    /// No chain moved, and nothing was non-finite: diagnostics are unavailable.
    pub fn is_frozen(&self) -> bool {
        self.nonfinite == 0 && self.frozen == self.chains
    }
    /// At least one diagnostic input is unusable (frozen or non-finite).
    pub fn is_degenerate(&self) -> bool {
        self.nonfinite > 0 || self.frozen > 0
    }
}

/// Count movement in a set of traces.
pub fn movement(chains: &[Vec<f64>]) -> Movement {
    let mut frozen = 0usize;
    let mut unique_max = 0usize;
    let mut nonfinite = 0usize;
    for c in chains {
        let mut finite: Vec<f64> = Vec::with_capacity(c.len());
        for x in c {
            if x.is_finite() {
                finite.push(*x);
            } else {
                nonfinite += 1;
            }
        }
        let mut sorted = finite.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        sorted.dedup();
        let unique = sorted.len();
        unique_max = unique_max.max(unique);
        if !finite.is_empty() && unique <= 1 {
            frozen += 1;
        }
    }
    Movement {
        chains: chains.len(),
        frozen,
        unique_max,
        nonfinite,
    }
}

fn all_finite(chain: &[f64]) -> bool {
    chain.iter().all(|x| x.is_finite())
}

/// Effective sample size via the initial-positive-sequence estimator:
/// integrate the autocorrelation until the pair sums turn negative,
/// `ESS = N / (1 + 2·Σρ)`. White noise → ESS ≈ N; a sticky chain → ESS ≪ N;
/// a frozen or non-finite chain → `NaN`.
pub fn ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return n as f64;
    }
    if !all_finite(chain) {
        return f64::NAN;
    }
    let mean = chain.iter().sum::<f64>() / n as f64;
    let var: f64 = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::NAN; // frozen chain: no autocorrelation to integrate
    }
    let mut sum = 0.0f64;
    let mut t = 1usize;
    while t + 1 < n {
        let mut pair = 0.0;
        for lag in [t, t + 1] {
            let cov: f64 = (0..n - lag)
                .map(|i| (chain[i] - mean) * (chain[i + lag] - mean))
                .sum::<f64>()
                / n as f64;
            pair += cov / var;
        }
        if pair <= 0.0 {
            break;
        }
        sum += pair;
        t += 2;
    }
    (n as f64 / (1.0 + 2.0 * sum)).min(n as f64)
}

/// Split-R̂ on raw draws: each chain is cut in half, the standard
/// between/within statistic runs over the 2m halves. Fresh independent draws →
/// ≈1; chains stuck in different modes → ≫1. Returns `NaN` for non-finite
/// input, fewer than two chains, or chains that are all frozen at the *same*
/// value (there is no variance to form a ratio from).
pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 || chains.iter().any(|c| c.is_empty()) {
        return f64::NAN;
    }
    if chains.iter().any(|c| !all_finite(c)) {
        return f64::NAN;
    }
    // Two chains frozen at the *same* value are the case that used to read 1.0:
    // no between-chain and no within-chain variance, so the ratio is 0/0 dressed
    // up as convergence.  Frozen at different values stays infinite (below).
    let frozen_value = |c: &[f64]| c.iter().all(|x| *x == c[0]);
    if chains.iter().all(|c| frozen_value(c)) && chains.iter().all(|c| c[0] == chains[0][0]) {
        return f64::NAN;
    }
    match split_halves(chains) {
        Some(h) => rhat_statistic(&h),
        None => f64::NAN,
    }
}

/// Cut every chain in half (`2m` halves of equal length), or `None` if the
/// chains cannot be split that way.
fn split_halves(chains: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let mut halves: Vec<Vec<f64>> = Vec::new();
    for c in chains {
        let n = c.len();
        if n < 4 {
            return None;
        }
        let h = n / 2;
        halves.push(c[..h].to_vec());
        halves.push(c[n - h..].to_vec());
    }
    let n = halves[0].len();
    if halves.iter().any(|h| h.len() != n) {
        return None;
    }
    Some(halves)
}

/// The between/within ratio over equally long halves.
fn rhat_statistic(halves: &[Vec<f64>]) -> f64 {
    let m = halves.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = halves[0].len();
    let grand: f64 = halves.iter().flatten().sum::<f64>() / (m * n) as f64;
    let means: Vec<f64> = halves
        .iter()
        .map(|h| h.iter().sum::<f64>() / n as f64)
        .collect();
    let between =
        n as f64 / (m - 1) as f64 * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>();
    let within = halves
        .iter()
        .map(|h| {
            let mu = h.iter().sum::<f64>() / n as f64;
            h.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64
        })
        .sum::<f64>()
        / m as f64;
    if within <= 0.0 {
        return if between > 0.0 { f64::INFINITY } else { 1.0 };
    }
    let var_hat = (n - 1) as f64 / n as f64 * within + between / n as f64;
    (var_hat / within).sqrt()
}

// ---------------------------------------------------------------------
// Rank-normalised split R̂ and bulk/tail ESS (Vehtari et al. 2021)
// ---------------------------------------------------------------------

/// Rank-normalised split `R̂`, folded and bulk combined as the paper
/// prescribes (take the larger). `NaN` on non-finite input, fewer than two
/// chains, or a frozen ensemble.
pub fn split_rhat_rank(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 || chains.iter().any(|c| !all_finite(c)) {
        return f64::NAN;
    }
    let mov = movement(chains);
    if mov.is_frozen() || mov.unique_max < 2 {
        return f64::NAN;
    }
    let pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().cloned()).collect();
    let ranked = normal_scores(&pooled);
    let folded_input: Vec<f64> = {
        let med = median_of(&pooled);
        pooled.iter().map(|x| (x - med).abs()).collect()
    };
    let folded = normal_scores(&folded_input);
    let bulk = halves_of_scores(chains, &ranked).map(|h| rhat_statistic(&h));
    let fold = halves_of_scores(chains, &folded).map(|h| rhat_statistic(&h));
    match (bulk, fold) {
        (Some(a), Some(b)) => a.max(b),
        _ => f64::NAN,
    }
}

/// Bulk ESS: rank-normalise the pooled draws, split each chain in half, and
/// run the multi-chain initial-positive-sequence estimator on the 2m halves.
/// `NaN` for a frozen ensemble.
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    if chains.is_empty() || chains.iter().any(|c| !all_finite(c)) {
        return f64::NAN;
    }
    let mov = movement(chains);
    if mov.is_frozen() || mov.unique_max < 2 {
        return f64::NAN;
    }
    let pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().cloned()).collect();
    let ranked = normal_scores(&pooled);
    match halves_of_scores(chains, &ranked) {
        Some(h) => ess_from_halves(&h),
        None => f64::NAN,
    }
}

/// Tail ESS: the smaller of the lower (5 %) and upper (95 %) tail ESS of the
/// *indicator* series, which is what the paper's `min(ESS_lo, ESS_hi)` asks
/// for. `NaN` for a frozen ensemble.
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    if chains.is_empty() || chains.iter().any(|c| !all_finite(c)) {
        return f64::NAN;
    }
    let mov = movement(chains);
    if mov.is_frozen() || mov.unique_max < 2 {
        return f64::NAN;
    }
    let pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().cloned()).collect();
    let mut sorted = pooled.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let q = |p: f64| -> f64 {
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let (lo, hi) = (q(0.05), q(0.95));
    let tail_ess = |limit: f64, low: bool| -> f64 {
        let mut ind = Vec::with_capacity(pooled.len());
        for c in chains {
            for x in c {
                let inside = if low { *x <= limit } else { *x > limit };
                ind.push(if inside { 1.0 } else { 0.0 });
            }
        }
        match halves_of_scores(chains, &ind) {
            Some(h) => ess_from_halves(&h),
            None => f64::NAN,
        }
    };
    let a = tail_ess(lo, true);
    let b = tail_ess(hi, false);
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// Split `n`-length chains into `2m` halves and cut the pooled score vector
/// the same way, returning the halves in the same order.
fn halves_of_scores(chains: &[Vec<f64>], pooled_scores: &[f64]) -> Option<Vec<Vec<f64>>> {
    let mut halves = Vec::with_capacity(chains.len() * 2);
    let mut at = 0usize;
    for c in chains {
        let t = c.len();
        if t < 4 {
            return None;
        }
        let h = t / 2;
        halves.push(pooled_scores[at..at + h].to_vec());
        halves.push(pooled_scores[at + t - h..at + t].to_vec());
        at += t;
    }
    let n = halves[0].len();
    if halves.iter().any(|x| x.len() != n) {
        return None;
    }
    Some(halves)
}

/// Multi-chain ESS with Geyer's initial positive sequence on the average
/// autocorrelation, in Vehtari et al.'s parameterisation (`var⁺` in the
/// denominator rather than the within-chain variance).
fn ess_from_halves(halves: &[Vec<f64>]) -> f64 {
    let m = halves.len();
    let n = halves[0].len();
    if m < 2 || n < 4 {
        return f64::NAN;
    }
    let total = (m * n) as f64;
    let means: Vec<f64> = halves
        .iter()
        .map(|h| h.iter().sum::<f64>() / n as f64)
        .collect();
    let grand = means.iter().sum::<f64>() / m as f64;
    let within: f64 = halves
        .iter()
        .zip(means.iter())
        .map(|(h, mu)| h.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64)
        .sum::<f64>()
        / m as f64;
    let between =
        n as f64 * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1) as f64;
    let var_plus = (n - 1) as f64 / n as f64 * within + between / n as f64;
    if var_plus <= 0.0 || var_plus.is_nan() {
        return f64::NAN;
    }
    // Cap the lag search: beyond ~2 000 steps a monotone pair sum has long
    // since gone negative in every trace this crate produces, and the O(n²)
    // cost is what made a 20 000-draw report take longer than the chain.
    let max_lag = n.min(2_000);
    let mut rho: Vec<f64> = Vec::with_capacity(max_lag);
    rho.push(1.0);
    for t in 1..max_lag {
        let mut acov = 0.0f64;
        for (h, mu) in halves.iter().zip(means.iter()) {
            for i in 0..n - t {
                acov += (h[i] - mu) * (h[i + t] - mu);
            }
        }
        acov /= total;
        rho.push(1.0 - (within - acov) / var_plus);
    }
    let mut sum = 0.0f64;
    let mut t = 0usize;
    while t + 1 < rho.len() {
        let pair = rho[t] + rho[t + 1];
        if pair < 0.0 {
            break;
        }
        sum += pair;
        t += 2;
    }
    let tau = -1.0 + 2.0 * sum;
    if tau <= 0.0 || tau.is_nan() {
        return total;
    }
    (total / tau).clamp(0.0, total)
}

/// 1-based ranks with ties averaged.
fn average_ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| x[*a].partial_cmp(&x[*b]).expect("finite"));
    let mut r = vec![0.0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// `Φ⁻¹((rank − 3/8)/(n + 1/4))` — the Blom normal scores the paper uses.
fn normal_scores(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    average_ranks(x)
        .into_iter()
        .map(|r| norm_ppf((r - 0.375) / (n + 0.25)))
        .collect()
}

fn median_of(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = s.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Acklam's rational approximation to `Φ⁻¹` (|error| < 1.2e-9 over the open
/// interval). Only used to rank-normalise; the paper's own tolerance is far
/// looser than that.
fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn white_noise_ess_is_close_to_n() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let chain: Vec<f64> = (0..8_000).map(|_| rng.random::<f64>()).collect();
        let e = ess(&chain);
        assert!((0.75 * 8_000.0..1.05 * 8_000.0).contains(&e), "ESS {e}");
    }

    #[test]
    fn ar1_chain_ess_matches_theory() {
        // AR(1) with ρ = 0.9: ESS ≈ N·(1−ρ)/(1+ρ) = N/19
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let rho = 0.9;
        let mut chain = Vec::with_capacity(20_000);
        let mut x = 0.0f64;
        for _ in 0..20_000 {
            let e: f64 = rng.random::<f64>() * 2.0 - 1.0; // bounded noise, var 1/3
            x = rho * x + e;
            chain.push(x);
        }
        let e = ess(&chain);
        let theory = 20_000.0 * (1.0 - rho) / (1.0 + rho);
        assert!(
            (0.4 * theory..2.5 * theory).contains(&e),
            "ESS {e} vs theory {theory}"
        );
    }

    #[test]
    fn independent_chains_have_rhat_near_one() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..2_000).map(|_| rng.random::<f64>()).collect())
            .collect();
        let r = gelman_rubin(&chains);
        assert!((0.99..1.02).contains(&r), "R̂ {r}");
        let rr = split_rhat_rank(&chains);
        assert!((0.99..1.02).contains(&rr), "rank R̂ {rr}");
    }

    #[test]
    fn separated_chains_are_flagged() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let mut a: Vec<f64> = (0..2_000).map(|_| rng.random::<f64>()).collect();
        let mut b: Vec<f64> = (0..2_000).map(|_| rng.random::<f64>() + 5.0).collect();
        // burn a little so the split halves exist
        a.extend(a.clone());
        b.extend(b.clone());
        let r = gelman_rubin(&[a.clone(), b.clone()]);
        assert!(r > 5.0, "R̂ {r} should scream");
        // The rank statistic compresses a pure location shift (both supports
        // become adjacent rank blocks), so it flags this at ~1.8 rather than
        // ~9 — still far outside the 1.01 gate, and that is what matters.
        let rr = split_rhat_rank(&[a, b]);
        assert!(rr > 1.1, "rank R̂ {rr} should flag separation");
    }

    /// A frozen trace must not read as a converged one — the failure mode the
    /// old `degenerate_inputs_are_not_panic` test *blessed*.
    #[test]
    fn frozen_chains_are_unavailable_not_perfect() {
        let frozen = vec![0.5; 100];
        assert!(ess(&frozen).is_nan(), "ESS of a frozen chain must be NaN");
        let a = vec![0.5; 100];
        let b = vec![0.5; 100];
        assert!(
            gelman_rubin(&[a.clone(), b.clone()]).is_nan(),
            "R̂ of two identically frozen chains must be NaN"
        );
        assert!(split_rhat_rank(&[a.clone(), b.clone()]).is_nan());
        assert!(ess_bulk(&[a.clone(), b.clone()]).is_nan());
        assert!(ess_tail(&[a, b]).is_nan());
        let mov = movement(&[frozen]);
        assert_eq!(mov.frozen, 1);
        assert_eq!(mov.unique_max, 1);
        assert!(mov.is_frozen());
        assert!(mov.is_degenerate());
    }

    /// Frozen at *different* values is a real disagreement: the within-chain
    /// variance is zero and the between-chain one is not, so the ratio is
    /// infinite rather than unavailable.
    #[test]
    fn chains_frozen_at_different_values_are_infinite_rhat() {
        let a = vec![0.0; 80];
        let b = vec![4.0; 80];
        let r = gelman_rubin(&[a, b]);
        assert!(r.is_infinite() && r > 0.0, "R̂ {r} should be +inf");
    }

    #[test]
    fn nonfinite_draws_are_unavailable() {
        let mut c = vec![0.0; 50];
        c[7] = f64::NAN;
        assert!(ess(&c).is_nan());
        let ok: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(gelman_rubin(&[c, ok.clone()]).is_nan());
        assert!(split_rhat_rank(&[vec![1.0; 10], ok]).is_nan());
    }

    #[test]
    fn rank_normalisation_handles_ties_and_heavy_tails() {
        // A few huge outliers leave the raw R̂ finite but small; the rank
        // statistic is scale-free, so the same chains still read ≈1.
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let mut chains: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..500).map(|_| rng.random::<f64>()).collect())
            .collect();
        chains[0][0] = 1e9;
        let raw = gelman_rubin(&chains);
        let rank = split_rhat_rank(&chains);
        assert!(raw.is_finite() && rank.is_finite());
        assert!(rank < 1.1, "rank R̂ {rank} should ignore the outlier scale");
        // ties: a discrete chain
        let ties: Vec<Vec<f64>> = (0..2)
            .map(|_| (0..400).map(|i| (i % 3) as f64).collect())
            .collect();
        let r = split_rhat_rank(&ties);
        assert!(r.is_finite(), "tied ranks gave {r}");
    }

    #[test]
    fn bulk_and_tail_ess_are_finite_and_ordered() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..1_000).map(|_| rng.random::<f64>()).collect())
            .collect();
        let total = 4_000.0;
        let bulk = ess_bulk(&chains);
        let tail = ess_tail(&chains);
        assert!(
            (0.5 * total..=total).contains(&bulk),
            "bulk ESS {bulk} of {total} iid draws"
        );
        assert!(tail > 100.0, "tail ESS {tail} of {total} iid draws");
        // sticky chains: both fall far below the draw count
        let mut ar = Vec::with_capacity(1_000);
        let mut x = 0.0;
        for _ in 0..1_000 {
            x = 0.98 * x + (rng.random::<f64>() * 2.0 - 1.0);
            ar.push(x);
        }
        let sticky: Vec<Vec<f64>> = (0..4).map(|_| ar.clone()).collect();
        let sbulk = ess_bulk(&sticky);
        assert!(sbulk < 0.2 * total, "sticky bulk ESS {sbulk}");
    }

    #[test]
    fn median_and_scores_are_sane() {
        assert_eq!(median_of(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_of(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        // normal scores are monotone and centred
        let x: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let z = normal_scores(&x);
        assert!(z.windows(2).all(|w| w[0] < w[1]), "scores not monotone");
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-9, "score mean {mean}");
        assert!((norm_ppf(0.975) - 1.959964).abs() < 1e-5);
        assert!(norm_ppf(0.5).abs() < 1e-12);
    }
}
