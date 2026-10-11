//! The in-repo reproduction of the latency table in
//! `docs/ARTICLE_ANACP_ANNC_VS_QFM.md` §4.1.
//!
//! Run:
//!
//! ```sh
//! cargo bench -p decide_core --bench latency
//! ```
//!
//! # What is measured
//!
//! The three per-query costs an embedding-fed surface actually pays, for
//! both target constructions, at the widths the D-ablation swept:
//!
//! * **observe** — one `observe()` call (sufficient-statistics
//!   accumulation; the streaming half of training),
//! * **fit** — one `fit()` call (the closed-form solve: whitening +
//!   SVD + perturbation search for AnaCP; Procrustes for AnaNC),
//! * **predict** — one `answer_score()` call (RP expansion + cosine
//!   readout; the per-keystroke cost).
//!
//! The corpus is the project's own `properties::native_corpus`
//! (seeded, 24-wide, 24 samples/class), split by its held-out flag so
//! the training stream is the same `n_train = 90` the pinned
//! calibration report and the D-ablation use. That is deliberate: this
//! bench's numbers are comparable to those fixtures' protocol, not to
//! an arbitrary random matrix.
//!
//! # Why the numbers matter (from the article's §5)
//!
//! The fit:predict ratio is the design property both analytic stacks
//! share (~10^5 at D=512 for AnaCP), and the fit gap between the two
//! modes — whitening decomposes in D-space for AnaCP, Procrustes in
//! C-space for AnaNC — is structural. This bench exists so neither
//! claim has to be taken from prose.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use decide_core::properties::{Labelled, native_corpus};
use decide_core::{DecideConfig, DecideCore, Mode};
use nalgebra::DVector;
use std::hint::black_box;

/// The widths the article's table reports, in ablation order.
const DIMS: [usize; 3] = [16, 256, 512];
/// Corpus geometry: same as the pinned fixtures (`per_class`,
/// `dim`), hence the same `n_train = 90` after the held-out split.
const PER_CLASS: usize = 24;
const CORPUS_DIM: usize = 24;

fn training_vectors() -> Vec<DVector<f64>> {
    native_corpus(PER_CLASS, CORPUS_DIM)
        .into_iter()
        .filter(|s: &Labelled| !s.held_out)
        .map(|s| s.embedding)
        .collect()
}

/// A registered core at `d` with the corpus's label vocabulary.
fn core_for(mode: Mode, d: usize, labels: &[String]) -> DecideCore {
    let mut core = DecideCore::new(DecideConfig {
        embedding_version: "decide-bench-latency-v1".to_string(),
        embedding_dim: CORPUS_DIM,
        projected_dim: d,
        heads: 1,
        mode,
        lambda: 1.0,
        alpha: 1.0,
    })
    .expect("config is valid");
    for l in labels {
        core.register(l).expect("fresh vocabulary registers");
    }
    core
}

fn labels_of() -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for s in native_corpus(PER_CLASS, CORPUS_DIM) {
        let name = s.label.name().to_string();
        if !seen.contains(&name) {
            seen.push(name);
        }
    }
    seen
}

/// `observe()` is called with the class mapped one per call on the
/// streaming path; the labels cycle so the class vocabulary stays
/// exercised evenly.
fn bench_observe(c: &mut Criterion) {
    let train = training_vectors();
    let labels = labels_of();
    let mut g = c.benchmark_group("observe_90");
    for mode in [Mode::Anacp, Mode::Ananc] {
        for d in DIMS {
            g.bench_with_input(BenchmarkId::new(format!("{mode:?}"), d), &d, |b, &d| {
                b.iter(|| {
                    let mut core = core_for(mode, d, &labels);
                    for (i, x) in train.iter().enumerate() {
                        let label = labels[i % labels.len()].clone();
                        core.observe(black_box(&label), black_box(x))
                            .expect("observe");
                    }
                    core
                });
            });
        }
    }
    g.finish();
}

/// `fit()` re-solves from the accumulated statistics; the bench
/// re-fits an already-trained core so the timing is the solve alone
/// (re-solving is the real refit path — `decide_report` and the
/// ablation both refit without re-observing).
fn bench_fit(c: &mut Criterion) {
    let train = training_vectors();
    let labels = labels_of();
    let mut g = c.benchmark_group("fit");
    for mode in [Mode::Anacp, Mode::Ananc] {
        for d in DIMS {
            g.bench_with_input(BenchmarkId::new(format!("{mode:?}"), d), &d, |b, &d| {
                let mut core = core_for(mode, d, &labels);
                for (i, x) in train.iter().enumerate() {
                    let label = labels[i % labels.len()].clone();
                    core.observe(&label, x).expect("observe");
                }
                b.iter(|| core.fit().expect("fit re-solves"));
            });
        }
    }
    g.finish();
}

/// `answer_score()` — the per-query cost: RP expansion + readout on a
/// fitted engine. The probe vector is corpus sample 3 (held-out side),
/// so the path matches what an editor ranking query actually does.
fn bench_predict(c: &mut Criterion) {
    let train = training_vectors();
    let labels = labels_of();
    let probe: DVector<f64> = native_corpus(PER_CLASS, CORPUS_DIM)
        .into_iter()
        .find(|s| s.held_out)
        .map(|s| s.embedding)
        .expect("the corpus has held-out samples");
    let mut g = c.benchmark_group("predict");
    for mode in [Mode::Anacp, Mode::Ananc] {
        for d in DIMS {
            g.bench_with_input(BenchmarkId::new(format!("{mode:?}"), d), &d, |b, &d| {
                let mut core = core_for(mode, d, &labels);
                for (i, x) in train.iter().enumerate() {
                    let label = labels[i % labels.len()].clone();
                    core.observe(&label, x).expect("observe");
                }
                core.fit().expect("fit");
                b.iter(|| {
                    core.answer_score(black_box("bench"), black_box(&probe), 5)
                        .expect("answer")
                });
            });
        }
    }
    g.finish();
}

/// Keep the full table runnable in CI-scale time: 10 samples per cell
/// and a 2 s floor give stable microsecond figures without a
/// minutes-long default sweep. The article's numbers were taken from a
/// default-criterion run on one AMD Ryzen AI 5 340; this bench
/// reproduces the order of magnitude on any host.
fn configure() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_observe, bench_fit, bench_predict
}
criterion_main!(benches);
