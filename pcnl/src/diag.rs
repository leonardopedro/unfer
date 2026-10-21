//! Phase 4 — MCMC health checks: effective sample size and
//! Gelman–Rubin `R̂` (split variant), the two diagnostics the plan
//! requires across independent chains.

/// Effective sample size via the initial-positive-sequence estimator:
/// integrate the autocorrelation until the pair sums turn negative,
/// `ESS = N / (1 + 2·Σρ)`. White noise → ESS ≈ N; a sticky chain →
/// ESS ≪ N.
pub fn ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return n as f64;
    }
    let mean = chain.iter().sum::<f64>() / n as f64;
    let var: f64 = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return n as f64; // constant chain: every draw "agrees"
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

/// Split-R̂: each chain is cut in half, the standard between/within
/// statistic runs over the 2m halves. Fresh independent draws → ≈1;
/// chains stuck in different modes → ≫1. Non-finite input → NaN.
pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return f64::NAN;
    }
    let mut halves: Vec<Vec<f64>> = Vec::new();
    for c in chains {
        let n = c.len();
        if n < 4 {
            return f64::NAN;
        }
        let h = n / 2;
        halves.push(c[..h].to_vec());
        halves.push(c[n - h..].to_vec());
    }
    let m = halves.len();
    let n = halves[0].len();
    if halves.iter().any(|h| h.len() != n) {
        return f64::NAN;
    }
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
    }

    #[test]
    fn separated_chains_are_flagged() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let mut a: Vec<f64> = (0..2_000).map(|_| rng.random::<f64>()).collect();
        let mut b: Vec<f64> = (0..2_000).map(|_| rng.random::<f64>() + 5.0).collect();
        // burn a little so the split halves exist
        a.extend(a.clone());
        b.extend(b.clone());
        let r = gelman_rubin(&[a, b]);
        assert!(r > 5.0, "R̂ {r} should scream");
    }

    #[test]
    fn degenerate_inputs_are_not_panic() {
        assert!(gelman_rubin(&[vec![1.0]]).is_nan());
        assert_eq!(ess(&[1.0, 2.0]), 2.0);
        let const_chain = vec![0.5; 100];
        assert_eq!(ess(&const_chain), 100.0);
        assert_eq!(gelman_rubin(&[const_chain.clone(), const_chain]), 1.0);
    }
}
