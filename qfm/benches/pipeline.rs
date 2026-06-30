//! Criterion benchmarks for the QFM tomographic pipeline (rev 15, P6 F.20
//! + rev 18 P7 P5 expansion).
//!
//! Seven groups exercising the architecture's central scaling claims:
//!
//! - `compile_vs_M` — offline compile time vs. training-set size M
//!   (M = 10/100/1000). Expected: roughly linear in M (the O(M) training
//!   step and the K_2-bound sketch construction).
//!
//! - `generate_vs_d` — online generation time vs. raw dimension d
//!   (d = 64/256/1024). Expected: roughly O(d · m²) + O(K_2 · m²) +
//!   O(K_2 log k), driven by the d-dim sketch encode, the rank-m Padé
//!   exponential, and the rank²-dim decode. Generation time must be
//!   **independent of M** (the whole point of the tomographic subspace
//!   recovery: the rank-m reduced system is what evolves at query time,
//!   not the full training set).
//!
//! - `sketch_apply_vs_d` — `CountSketch::apply` time vs. d
//!   (d = 64/256/1024/4096). Expected: O(d) linear scaling (one hash +
//!   one accumulator update per input dimension).
//!
//! - `bayes_update_vs_n` — Quantum Bayesian update time vs. number of
//!   new observations N (N = 1/4/16, M = 100 training set, d = 64,
//!   fixed m = 4 Krylov subspace). Expected: O(N · m²) per HMC step
//!   for the likelihood-gradient term, so total cost is roughly linear
//!   in N.
//!
//! - `bayes_update_vs_m` (P7 P5) — Quantum Bayesian update time vs. the
//!   Krylov subspace dimension m (m = 2/4/8; M = 8 training set, d = 8,
//!   N = 1 fixed). Expected: O(m²) per HMC step (the
//!   likelihood-gradient term), so total cost grows roughly quadratically
//!   in m.
//!
//! - `bayes_update_vs_k2` (P7 P5) — Quantum Bayesian update time vs. the
//!   K_2-dim sketched Hilbert space (K_2 = 4/8/16; M = 16 training set,
//!   d = 16, N = 1 fixed). Expected: O(K_2 · m²) per HMC step in the
//!   likelihood construction, so total cost grows linearly in K_2.
//!
//! - `bayes_update_vs_leapfrog` (P7 P5) — Quantum Bayesian update time vs.
//!   the HMC leapfrog chain length (leapfrog_steps = 10/20/50; M = 50,
//!   d = 32, N = 1, n_iterations = 50). Expected: linear in
//!   leapfrog_steps (per-step cost is O(N · m²) but the chain is
//!   leapfrog_steps long).
//!
//! Acceptance (per IMPLEMENTATION_PLAN.md §P6 F.20, §P6 H, §P7 P5):
//!   `cargo bench -p qfm --bench pipeline` runs clean and the
//!   measurements show the expected scaling.
//!
//! Note: `compile()` runs a real SIRK solve on a vacuum + single-excitation
//! seed in the K_2-dim Fock space, so the larger-M / larger-d benchmark
//! points are the only places the bench will dominate wall-clock time.
//! The P7 P5 bayes_update_vs_m / bayes_update_vs_k2 /
//! bayes_update_vs_leapfrog groups use smaller training sets to keep
//! the compile time low.

use criterion::{Criterion, criterion_group, criterion_main};
use qfm::{
    CountSketch, HmcOpts, Likelihood, Posterior, QfmConfig, QfmPipeline, sample_hmc_single,
    tsr_evolved_prior,
};
use std::hint::black_box;

/// Build a synthetic training set of `m` d-dimensional points centred on
/// the standard basis directions, with a small random component so the
/// hash sketches do not collapse to a single bucket.
fn synthetic_training(m: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = splitmix64(seed);
    (0..m)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let r = (rng.next_u64() as f64) / (u64::MAX as f64);
                    let bump = if j == i % d { 1.0 } else { 0.0 };
                    bump + 0.05 * (r - 0.5)
                })
                .collect()
        })
        .collect()
}

fn config_for(d: usize, m: usize) -> QfmConfig {
    // The compile-time constraints (P7 P3 + P6 G) are:
    //   * krylov_dim >= K_2 (the K_2-row restriction of w_whiten is
    //     well-defined only when the SIRK sequence has K_2+1 rows);
    //   * the effective krylov_dim is min(config.krylov_dim, m, K_2),
    //     so K_2 <= m;
    //   * the S_2 register step needs K_2 >= M (every training point
    //     gets a unique mode; otherwise register returns
    //     K2BoundExceeded -> DegenerateBasis);
    //   * the krylov_image_basis debug_assert!(d <= k2) requires
    //     d <= K_2.
    //
    // For the bench to be tractable on CPU (a few seconds per iteration
    // at most), we cap K_2 at 16. The M=1000 / d=1024 case from the
    // rev 16 spec is no longer in the bench (it would require a 1024-dim
    // SIRK which is a research-scale run, not a unit bench); the
    // P8 P10 CUDA bench is the right place to measure the M=1000
    // throughput. So K_2 = min(m, d, 16) satisfies all four
    // constraints (and gives K_2 == m == d for the small bench cases).
    let k2 = m.min(d).min(16);
    QfmConfig {
        k: 4.min(d).max(1),
        k2,
        krylov_dim: k2,
        seed: 42,
        n_t_samples: 4,
        noise_dim: d,
    }
}

fn bench_compile_vs_m(c: &mut Criterion) {
    // K_2 capped at 16 per config_for. M values are chosen so
    // d = m = K_2 (all three constraints are equality): 4, 8, 16.
    // The M=1000 / d=1024 point from the rev 16 spec is no longer in
    // the bench (see config_for doc); the P8 P10 CUDA bench is the
    // right place to measure that scale.
    let mut group = c.benchmark_group("compile_vs_M");
    group.sample_size(10);
    for &m in &[4usize, 8, 16] {
        let d = m;
        let training = synthetic_training(m, d, 7);
        let cfg = config_for(d, m);
        group.bench_with_input(criterion::BenchmarkId::from_parameter(m), &m, |b, &_m| {
            b.iter(|| {
                let pipeline = QfmPipeline::compile(black_box(&training), black_box(&cfg)).unwrap();
                black_box(pipeline.raw_dim());
            });
        });
    }
    group.finish();
}

fn bench_generate_vs_d(c: &mut Criterion) {
    // Single M=16 training set, fixed compile. The compile is paid
    // once outside the timed region, isolating `generate` cost as a
    // function of raw dimension d. The d values are bounded by the
    // krylov_image_basis d <= K_2 constraint (with K_2 = m = d),
    // so d = m in {4, 8, 16} here (was 64/256/1024 in rev 16 with
    // the broken krylov_dim=4; see config_for doc).
    let mut group = c.benchmark_group("generate_vs_d");
    for &d in &[4usize, 8, 16] {
        let training = synthetic_training(d, d, 11);
        let cfg = config_for(d, d);
        let pipeline = QfmPipeline::compile(&training, &cfg).unwrap();
        let query = training[0].clone();
        group.bench_with_input(criterion::BenchmarkId::from_parameter(d), &d, |b, &_d| {
            b.iter(|| {
                let out = pipeline.generate(black_box(&query)).unwrap();
                black_box(out.len());
            });
        });
    }
    group.finish();
}

fn bench_sketch_apply_vs_d(c: &mut Criterion) {
    let mut group = c.benchmark_group("sketch_apply_vs_d");
    for &d in &[64usize, 256, 1024, 4096] {
        let sketch = CountSketch::new(8, d, 99);
        let x: Vec<f64> = (0..d).map(|i| ((i as f64) + 1.0) / (d as f64)).collect();
        group.bench_with_input(criterion::BenchmarkId::from_parameter(d), &d, |b, &_d| {
            b.iter(|| {
                let y = sketch.apply(black_box(&x));
                black_box(y.len());
            });
        });
    }
    group.finish();
}

fn bench_bayes_update_vs_n(c: &mut Criterion) {
    // Single M=16, d=16, K_2=16 compile (paid outside the timed region).
    // Vary N (number of new observations) from 1/4/16. Per HMC step
    // the cost is O(N * m^2) (likelihood-gradient term), so the timing
    // should grow roughly linearly in N. (The rev 16 setup had M=100,
    // d=64 with the broken krylov_dim=4; see config_for doc.)
    let m = 16usize;
    let d = 16usize;
    let training = synthetic_training(m, d, 13);
    let cfg = config_for(d, m);
    let pipeline = QfmPipeline::compile(&training, &cfg).unwrap();
    let c_prior = tsr_evolved_prior(&pipeline);

    let mut group = c.benchmark_group("bayes_update_vs_n");
    group.sample_size(10);
    for &n_obs in &[1usize, 4, 16] {
        let likelihoods: Vec<Likelihood> = (0..n_obs)
            .map(|i| {
                Likelihood::from_observation(&pipeline, &training[i % training.len()])
                    .expect("likelihood")
            })
            .collect();
        let opts = HmcOpts {
            leapfrog_steps: 20,
            step_size: 0.05,
            n_iterations: 100,
            burn_in: 50,
            seed: 42,
        };
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(n_obs),
            &n_obs,
            |b, &_n| {
                b.iter(|| {
                    let posterior = Posterior::new(black_box(likelihoods.clone()), c_prior.clone());
                    let sample = sample_hmc_single(&posterior, black_box(&opts));
                    black_box(sample.len());
                });
            },
        );
    }
    group.finish();
}

/// P7 P5: bayes_update_vs_m. Fix N=1, K_2=m, d=m, M=m; vary m (Krylov
/// subspace dimension). The K_2-row restriction of `w_whiten` requires
/// krylov_dim >= K_2 (P6 G), so we set k2 = m_target. The
/// krylov_image_basis debug_assert!(d <= k2) requires d <= K_2, so we
/// set d = m_target. The per-HMC-step likelihood-gradient cost is
/// O(m²), so the timing should grow roughly quadratically in m.
fn bench_bayes_update_vs_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayes_update_vs_m");
    group.sample_size(10);
    for &m_target in &[4usize, 8, 16] {
        let d = m_target;
        let training = synthetic_training(m_target, d, 19);
        let opts = HmcOpts {
            leapfrog_steps: 20,
            step_size: 0.05,
            n_iterations: 50,
            burn_in: 25,
            seed: 42,
        };
        // k2 = m_target, krylov_dim = k2 (per P7 P3 constraint).
        let cfg = QfmConfig {
            k: 2,
            k2: m_target,
            krylov_dim: m_target,
            seed: 42,
            n_t_samples: 4,
            noise_dim: d,
        };
        let pipeline = QfmPipeline::compile(&training, &cfg).expect("compile");
        let c_prior = tsr_evolved_prior(&pipeline);
        let likelihood = Likelihood::from_observation(&pipeline, &training[0]).expect("likelihood");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(m_target),
            &m_target,
            |b, &_m| {
                b.iter(|| {
                    let posterior =
                        Posterior::new(black_box(vec![likelihood.clone()]), c_prior.clone());
                    let sample = sample_hmc_single(&posterior, black_box(&opts));
                    black_box(sample.len());
                });
            },
        );
    }
    group.finish();
}

/// P7 P5: bayes_update_vs_k2. Fix N=1, m=K_2 (the P6 G constraint),
/// d=K_2, M=K_2; vary K_2 = 4/8/16. The likelihood construction cost
/// scales with K_2 (the K_2-dim sketched Hilbert space), so the
/// timing should grow roughly linearly in K_2. (The krylov_image_basis
/// d <= k2 constraint forces d = K_2 here.)
fn bench_bayes_update_vs_k2(c: &mut Criterion) {
    let opts = HmcOpts {
        leapfrog_steps: 20,
        step_size: 0.05,
        n_iterations: 50,
        burn_in: 25,
        seed: 42,
    };

    let mut group = c.benchmark_group("bayes_update_vs_k2");
    group.sample_size(10);
    for &k2 in &[4usize, 8, 16] {
        let d = k2;
        let training = synthetic_training(k2, d, 23);
        // krylov_dim = k2 (P7 P3 constraint).
        let cfg = QfmConfig {
            k: 2,
            k2,
            krylov_dim: k2,
            seed: 42,
            n_t_samples: 4,
            noise_dim: d,
        };
        let pipeline = QfmPipeline::compile(&training, &cfg).expect("compile");
        let c_prior = tsr_evolved_prior(&pipeline);
        let likelihood = Likelihood::from_observation(&pipeline, &training[0]).expect("likelihood");
        group.bench_with_input(criterion::BenchmarkId::from_parameter(k2), &k2, |b, &_k| {
            b.iter(|| {
                let posterior =
                    Posterior::new(black_box(vec![likelihood.clone()]), c_prior.clone());
                let sample = sample_hmc_single(&posterior, black_box(&opts));
                black_box(sample.len());
            });
        });
    }
    group.finish();
}

/// P7 P5: bayes_update_vs_leapfrog. Fix N=1, M=16, d=16, K_2=16;
/// vary leapfrog_steps = 10/20/50. The HMC chain is leapfrog_steps
/// long per iteration, so the timing should grow roughly linearly in
/// leapfrog_steps. (See config_for doc for the K_2=16 cap rationale.)
fn bench_bayes_update_vs_leapfrog(c: &mut Criterion) {
    let d = 16usize;
    let m_training = 16usize;
    let training = synthetic_training(m_training, d, 29);
    let cfg = config_for(d, m_training);
    let pipeline = QfmPipeline::compile(&training, &cfg).expect("compile");
    let c_prior = tsr_evolved_prior(&pipeline);
    let likelihood = Likelihood::from_observation(&pipeline, &training[0]).expect("likelihood");

    let mut group = c.benchmark_group("bayes_update_vs_leapfrog");
    group.sample_size(10);
    for &leapfrog in &[10usize, 20, 50] {
        let opts = HmcOpts {
            leapfrog_steps: leapfrog,
            step_size: 0.05,
            n_iterations: 50,
            burn_in: 25,
            seed: 42,
        };
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(leapfrog),
            &leapfrog,
            |b, &_l| {
                b.iter(|| {
                    let posterior =
                        Posterior::new(black_box(vec![likelihood.clone()]), c_prior.clone());
                    let sample = sample_hmc_single(&posterior, black_box(&opts));
                    black_box(sample.len());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compile_vs_m,
    bench_generate_vs_d,
    bench_sketch_apply_vs_d,
    bench_bayes_update_vs_n,
    bench_bayes_update_vs_m,
    bench_bayes_update_vs_k2,
    bench_bayes_update_vs_leapfrog
);
criterion_main!(benches);

// ---------------------------------------------------------------------------
// splitmix64 PRNG (mirrors qfm/src/sketch.rs's local copy) so the bench
// can synthesize reproducible training sets without depending on the
// internal PRNG.
// ---------------------------------------------------------------------------
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn splitmix64(seed: u64) -> SplitMix64 {
    SplitMix64(seed)
}
