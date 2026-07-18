//! P10.16.4: QFM-TSR validation on 2D Ising model critical snapshots.
//!
//! The 2D classical Ising model on an L×L lattice with periodic boundary
//! conditions has an exact critical temperature (Onsager 1944):
//!
//!   β_c = ln(1 + √2) / 2 ≈ 0.44069
//!
//! At β_c the system is at the second-order phase transition; spin
//! configurations exhibit long-range power-law correlations, scale
//! invariance (conformal symmetry), and universal critical exponents —
//! exactly the kind of highly-structured, non-trivial distribution that
//! the QFM-TSR pipeline is designed to learn from.
//!
//! This test generates L=16 (d=256) Ising configurations via Metropolis
//! Monte Carlo at β_c from a fixed seed, then runs the same 4-test
//! validation battery as `cifar10_validation.rs`:
//!
//!   1. Pipeline compiles on d=256 critical Ising data.
//!   2. `generate()` on a training point returns finite pixel values.
//!   3. Cosine similarity to the nearest training point is positive.
//!   4. Bayesian update on a held-out configuration runs end-to-end.
//!
//! The test is fully synthetic (no external data) and deterministic.

use qfm::{
    HmcOpts, Likelihood, Posterior, QfmConfig, QfmPipeline, sample_hmc_single, tsr_evolved_prior,
};

// ── 2D Ising Monte Carlo ──────────────────────────────────────────────────

const L: usize = 16;
const D: usize = L * L; // 256 pixels
/// Exact critical inverse temperature for the 2D Ising model (Onsager 1944).
const BETA_C: f64 = 0.440_686_793_509_771_6; // ln(1+√2)/2

/// splitmix64 — matches the PRNG used throughout qfm for reproducibility.
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e3779b97f4a7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Generate one Metropolis sweep over the L×L lattice.
fn metropolis_sweep(spins: &mut [i8; D], beta: f64, rng: &mut u64) {
    for idx in 0..D {
        let x = idx % L;
        let y = idx / L;
        let s = spins[idx] as i32;
        // Nearest-neighbour sum with periodic boundary.
        let nb: i32 = spins[((y + L - 1) % L) * L + x] as i32
            + spins[((y + 1) % L) * L + x] as i32
            + spins[y * L + (x + L - 1) % L] as i32
            + spins[y * L + (x + 1) % L] as i32;
        let delta_e = 2 * s * nb;
        // Accept flip.
        let accept = if delta_e <= 0 {
            true
        } else {
            *rng = splitmix64(*rng);
            let u = (*rng as f64) / (u64::MAX as f64);
            u < (-beta * delta_e as f64).exp()
        };
        if accept {
            spins[idx] = -spins[idx];
        }
    }
}

/// Generate `n` Ising configurations at β_c from a fixed seed.
/// Returns pixel values in [0, 1]: p_i = (σ_i + 1) / 2.
fn generate_ising_configs(n: usize, beta: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    let mut spins = [1i8; D];
    // Burn-in: 500 sweeps.
    for _ in 0..500 {
        metropolis_sweep(&mut spins, beta, &mut rng);
    }
    // Collect n samples, one every 20 sweeps to reduce autocorrelation.
    let mut configs = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..20 {
            metropolis_sweep(&mut spins, beta, &mut rng);
        }
        configs.push(spins.iter().map(|&s| (s as f64 + 1.0) / 2.0).collect());
    }
    configs
}

// ── Pipeline helpers ──────────────────────────────────────────────────────

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn ising_pipeline(training: &[Vec<f64>], seed: u64) -> QfmPipeline {
    let d = training[0].len(); // 256
    let m = training.len(); // 64 training configs
    // With M=64 training points but d=256 pixels, K_2 = max(M,d) = 256.
    // The lossless path requires krylov_dim >= K_2 = 256, but m=64 clamps
    // krylov_dim to 64 < K_2, triggering K2ExceedsKrylovDim. Use the
    // P10.16.3 rank-truncation path: krylov_dim=32, max_rank=16.
    // This demonstrates both P10.16.3 (rank truncation) and P10.16.4
    // (2D Ising cross-domain dataset) working together.
    let k2 = m.max(d); // 256
    let config = QfmConfig {
        k: 16,
        k2,
        krylov_dim: 32,
        seed,
        n_t_samples: 4,
        noise_dim: d,
        max_rank: Some(16),
        ..Default::default()
    };
    QfmPipeline::compile(training, &config).expect("QFM compile on 2D Ising critical snapshots")
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn ising2d_qfm_pipeline_compiles() {
    // 64 training configs × 256 pixels (L=16 at β_c).
    let training = generate_ising_configs(64, BETA_C, 0xCEA5);
    assert_eq!(training.len(), 64);
    assert_eq!(training[0].len(), D);
    for &p in &training[0] {
        assert!((0.0..=1.0).contains(&p), "pixel {p} out of [0,1]");
    }

    let pipeline = ising_pipeline(&training, 42);
    assert_eq!(pipeline.raw_dim(), D);
    assert!(pipeline.rank() >= 1, "rank must be >= 1");
    println!(
        "ising2d_compile: rank={}, K_2={}, d={}",
        pipeline.rank(),
        pipeline.k2_dim(),
        pipeline.raw_dim()
    );
}

#[test]
fn ising2d_qfm_generate_finite_and_correlated() {
    let training = generate_ising_configs(64, BETA_C, 0xCEA5);
    let pipeline = ising_pipeline(&training, 42);

    // Evaluate on the first 8 training configs.
    let mut sims = Vec::new();
    for (i, query) in training.iter().take(8).enumerate() {
        let x_out = pipeline.generate(query).expect("generate");
        assert_eq!(x_out.len(), D, "output must be d=256");
        for &v in &x_out {
            assert!(v.is_finite(), "generate output must be finite, got {v}");
        }
        let nearest_sim = training
            .iter()
            .map(|t| cosine_similarity(&x_out, t))
            .fold(f64::NEG_INFINITY, f64::max);
        sims.push(nearest_sim);
        println!("ising2d_generate[{i}]: nearest cosine sim = {nearest_sim:.4}");
    }
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_sim > 0.2,
        "expected at least one generate output correlated with training set, max sim = {max_sim:.4}"
    );
    let min_sim = sims.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_sim > 0.0,
        "expected all generate outputs positively correlated with training set, min sim = {min_sim:.4}"
    );
}

#[test]
fn ising2d_qfm_bayes_update_held_out() {
    let training = generate_ising_configs(64, BETA_C, 0xCEA5);
    // 8 held-out configs from a different seed + different burn-in.
    let held_out = generate_ising_configs(8, BETA_C, 0xFEED);

    let pipeline = ising_pipeline(&training, 42);
    let c_prior = tsr_evolved_prior(&pipeline);

    let opts = HmcOpts {
        leapfrog_steps: 20,
        step_size: 0.05,
        n_iterations: 50,
        burn_in: 25,
        seed: 42,
    };

    let mut overlaps = Vec::new();
    for (i, obs) in held_out.iter().enumerate() {
        let c_obs = match pipeline.encode(obs) {
            Ok(c) if c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt() > 1e-6 => c,
            _ => {
                println!("ising2d_bayes[{i}]: c_obs zero (S_2 fallback); skipping");
                continue;
            }
        };
        let likelihood = Likelihood::from_observation(&pipeline, obs).expect("likelihood");
        let posterior = Posterior::new(vec![likelihood], c_prior.clone());
        let sample = sample_hmc_single(&posterior, &opts);
        let overlap: f64 = sample
            .iter()
            .zip(c_obs.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();
        let overlap_sq = overlap * overlap;
        println!("ising2d_bayes[{i}]: |<sample|c_obs>|² = {overlap_sq:.4}");
        overlaps.push(overlap_sq);
    }

    assert!(
        !overlaps.is_empty(),
        "expected at least one held-out config to have a non-zero c_obs"
    );
    let max_overlap = overlaps.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_overlap > 0.01,
        "expected at least one HMC sample to have non-trivial overlap with its observation, max = {max_overlap:.4}"
    );
}

#[test]
fn ising2d_qfm_no_explosion_on_held_out() {
    let training = generate_ising_configs(64, BETA_C, 0xCEA5);
    let held_out = generate_ising_configs(8, BETA_C, 0xFEED);
    let pipeline = ising_pipeline(&training, 42);

    for (i, obs) in held_out.iter().enumerate() {
        let x_out = pipeline.generate(obs).expect("generate held-out");
        for &v in &x_out {
            assert!(
                v.is_finite(),
                "held-out generate output must be finite, got {v} (idx={i})"
            );
        }
    }
}
