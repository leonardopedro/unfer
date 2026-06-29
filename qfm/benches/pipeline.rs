//! Criterion benchmarks for the QFM tomographic pipeline (rev 15, P6 F.20).
//!
//! Three groups exercising the architecture's central scaling claims:
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
//! Acceptance (per IMPLEMENTATION_PLAN.md §P6 F.20):
//!   `cargo bench -p qfm --bench pipeline` runs clean and the
//!   measurements show the expected scaling.
//!
//! Note: `compile()` runs a real SIRK solve on a vacuum + single-excitation
//! seed in the K_2-dim Fock space, so the larger-M / larger-d benchmark
//! points are the only places the bench will dominate wall-clock time.

use criterion::{Criterion, criterion_group, criterion_main};
use qfm::{CountSketch, QfmConfig, QfmPipeline};
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
    // K_2 must be >= M so every training feature can be registered in S_2
    // (the FeatureToMode map is bounded by K_2). krylov_dim = 4 keeps the
    // SIRK solve cheap; k = min(4, d) is the standard Level 1 sketch size.
    // If d < m (caller is using a small d for a large M), the K_2 = M
    // choice naturally forces the user to grow d.
    QfmConfig {
        k: 4.min(d).max(1),
        k2: m.max(8),
        krylov_dim: 4.min(m).min(m.max(8)),
        seed: 42,
        n_t_samples: 4,
        noise_dim: d,
    }
}

fn bench_compile_vs_m(c: &mut Criterion) {
    // d scales with M so the K_2 = M bound fits: 10 -> d=16, 100 -> d=128,
    // 1000 -> d=1024. This isolates compile time as a function of M, the
    // offline "training" set size. The M=1000 point is the most expensive
    // (full SIRK solve on a K_2=1000 Fock space) and the criterion default
    // sample size is plenty for a clean comparison.
    let mut group = c.benchmark_group("compile_vs_M");
    group.sample_size(10);
    for &(m, d) in &[(10usize, 16usize), (100, 128), (1000, 1024)] {
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
    // Single M=100 training set, fixed compile. The compile is paid once
    // outside the timed region, isolating `generate` cost as a function
    // of raw dimension d.
    let mut group = c.benchmark_group("generate_vs_d");
    for &d in &[64usize, 256, 1024] {
        let training = synthetic_training(100, d, 11);
        let cfg = config_for(d, 100);
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

criterion_group!(
    benches,
    bench_compile_vs_m,
    bench_generate_vs_d,
    bench_sketch_apply_vs_d
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
