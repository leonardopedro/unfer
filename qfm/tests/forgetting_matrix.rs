//! D-C6 protocol, run over a QFM-TSR class-arrival sequence.
//!
//! # Why this test exists
//!
//! The comparison article (`docs/ARTICLE_ANACP_ANNC_VS_QFM.md`) observed
//! that `decide_core` *measures* forgetting (the task-incremental accuracy
//! matrix, D-C6) while QFM only *engineers* a continual mechanism (the §8
//! Bayesian update) — and that "QFM forgets less" was therefore a
//! hypothesis, not a result. This file turns the hypothesis into a
//! measurement by running the identical protocol over a QFM-TSR arrival
//! stream.
//!
//! # The protocol (mirrors `decide_core::report::accuracy_matrix`)
//!
//! Classes arrive in tasks. After each task the pipeline is recompiled on
//! the *cumulative* bank (the closed-form refit — the QFM analog of
//! `decide_core`'s refit-from-sufficient-statistics), and every held-out
//! query of every *seen* class is evaluated:
//!
//!   A[t][i] = fraction of class-i held-out queries whose generated
//!             output attributes to class i, where attribution is the
//!             class of the nearest training point (cosine) in the
//!             cumulative bank.
//!
//! Unseen cells are NaN ("undef" in `decide_report`'s rendering).
//! `max_forgetting` is `max_i (best_t A[t][i] − A[T−1][i])` over classes
//! that were seen in at least two tasks.
//!
//! Attribution-via-nearest-training-point-after-generate is the same
//! generation-quality signal the crate's own validation suites use
//! (`max cosine to nearest training point`); using it as a classifier is
//! what makes the generative pipeline answerable on a decision protocol.
//!
//! # The baseline arm
//!
//! The same matrix is computed with attribution taken on the *raw query*
//! (no flow, plain cosine NN over the bank). If the flow's matrix is no
//! better than the baseline's, the measured forgetting belongs to bank
//! growth (interference in the similarity geometry), not to the Krylov
//! dynamics — which is exactly the attribution question the article
//! raises.
//!
//! # Honest scope
//!
//! Synthetic structured fixture (seeded LCG, spot-pattern class centers),
//! 8 classes, d = 64, 8 train + 4 held-out per class — the same fixture
//! philosophy as `cifar10_validation.rs`. This measures the protocol on
//! this stream; it is not a claim about real embedding corpora. The
//! numbers are printed so a reader can see the matrix, not only the
//! scalar that the assertions bound.
//!
//! # The measured result (2026-10-12, this fixture)
//!
//! * QFM-refit arm: `max_forgetting = 1.0000` — at every task the flow's
//!   attribution collapses onto a single class region (final row: one
//!   class at 1.00, all others 0.00). The same number `decide_core`
//!   measures for its AnaNC ETF head.
//! * raw-NN baseline: `max_forgetting = 0.0000` at 1.00 accuracy on every
//!   seen cell — the underlying similarity geometry stays perfectly
//!   separable under bank growth.
//!
//! The attribution of forgetting is therefore *inside the flow* (the
//! documented rank-truncation representative-output collapse), not in
//! the class-arrival interference the baseline would exhibit. The
//! Bayesian-continuation arm (§8 posterior becoming the next prior) is a
//! different mechanism and is *not* exercised here; "QFM forgets less"
//! remains untested for that arm, but is now measured false for the
//! closed-form refit arm.

use qfm::{QfmConfig, QfmPipeline};

const SEED: u64 = 0xD_C6A0_2026;
const D: usize = 64;
const CLASSES: usize = 8;
const PER_CLASS: usize = 8;
const HELD_OUT_PER_CLASS: usize = 4;
/// Arrival tasks: two classes per task, four tasks.
const TASK_CLASSES: usize = 2;

/// Deterministic LCG + Box–Muller (mirrors `cifar10_validation.rs`; no RNG
/// dependency in this crate's dev-deps, and a seeded fixture is the repo's
/// convention for empirical baselines).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A structured 8×8 class center: two bright Gaussian spots at
/// class-specific grid positions (4×4 grid, 16 positions, class `c` takes
/// positions `2c` and `2c+1`), so the classes are separable in raw space
/// and the S1 sketch sees different hashes per class.
fn class_center(class: usize) -> Vec<f64> {
    let grid: [(f64, f64); 16] = [
        (1.0, 1.0),
        (1.0, 3.0),
        (1.0, 5.0),
        (1.0, 7.0),
        (3.0, 1.0),
        (3.0, 3.0),
        (3.0, 5.0),
        (3.0, 7.0),
        (5.0, 1.0),
        (5.0, 3.0),
        (5.0, 5.0),
        (5.0, 7.0),
        (7.0, 1.0),
        (7.0, 3.0),
        (7.0, 5.0),
        (7.0, 7.0),
    ];
    let mut center = vec![0.0_f64; D];
    for &(px, py) in &grid[class * 2..class * 2 + 2] {
        for y in 0..8 {
            for x in 0..8 {
                let r2 = (x as f64 - px).powi(2) + (y as f64 - py).powi(2);
                center[y * 8 + x] += (-r2 / 2.0).exp();
            }
        }
    }
    let max = center.iter().cloned().fold(0.0_f64, f64::max);
    for v in &mut center {
        *v /= max;
    }
    center
}

/// (train vectors, train labels, held-out vectors, held-out labels).
type Corpus = (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>);

fn corpus(seed: u64) -> Corpus {
    let mut rng = Lcg::new(seed);
    let mut train = Vec::new();
    let mut train_y = Vec::new();
    let mut held = Vec::new();
    let mut held_y = Vec::new();
    for c in 0..CLASSES {
        let center = class_center(c);
        for _ in 0..PER_CLASS {
            let mut x = center.clone();
            for v in &mut x {
                *v = (*v + 0.05 * rng.next_normal()).clamp(0.0, 1.0);
            }
            train.push(x);
            train_y.push(c);
        }
        for _ in 0..HELD_OUT_PER_CLASS {
            let mut x = center.clone();
            for v in &mut x {
                *v = (*v + 0.15 * rng.next_normal()).clamp(0.0, 1.0);
            }
            held.push(x);
            held_y.push(c);
        }
    }
    (train, train_y, held, held_y)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn compile(bank: &[Vec<f64>]) -> QfmPipeline {
    // The arrival bank always has M <= 64 <= K_2, so the lossless path
    // (effective krylov_dim = min(config, M, K_2) >= K_2, requiring
    // M >= K_2) is inapplicable at every task — exactly the situation of
    // the rev 29/30 cross-domain fixtures (M=64 < K_2=1024). The rev 26
    // rank-truncation path is therefore the shipped, recommended config
    // here: K_2 = max(M, d), krylov_dim = 32, max_rank <= M.
    let m = bank.len();
    let k2 = m.max(D);
    let config = QfmConfig {
        k: 8,
        k2,
        krylov_dim: 32,
        seed: SEED,
        n_t_samples: 4,
        noise_dim: D,
        max_rank: Some(8.min(m)),
        ..Default::default()
    };
    QfmPipeline::compile(bank, &config).expect("QFM compile on arrival bank")
}

/// One protocol arm: `use_flow` runs the generated output through the
/// pipeline; otherwise attribution is taken on the raw query.
fn run_protocol(use_flow: bool) -> Vec<Vec<f64>> {
    let (train, train_y, held, held_y) = corpus(SEED);
    let tasks: Vec<usize> = (0..CLASSES).step_by(TASK_CLASSES).collect();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for t_start in tasks {
        let seen: Vec<usize> = (0..=t_start + TASK_CLASSES - 1).collect();
        let bank: Vec<Vec<f64>> = train
            .iter()
            .zip(train_y.iter())
            .filter(|(_, y)| seen.contains(y))
            .map(|(x, _)| x.clone())
            .collect();
        let bank_y: Vec<usize> = train_y
            .iter()
            .filter(|y| seen.contains(y))
            .copied()
            .collect();
        let pipeline = if use_flow { Some(compile(&bank)) } else { None };
        let mut row = vec![f64::NAN; CLASSES];
        for &i in &seen {
            let mut hits = 0usize;
            let mut total = 0usize;
            for (q, &qy) in held.iter().zip(held_y.iter()) {
                if qy != i {
                    continue;
                }
                let attributed = match &pipeline {
                    Some(p) => {
                        let out = p.generate(q).expect("generate");
                        nearest(&out, &bank, &bank_y)
                    }
                    None => nearest(q, &bank, &bank_y),
                };
                total += 1;
                if attributed == i {
                    hits += 1;
                }
            }
            row[i] = hits as f64 / total as f64;
        }
        rows.push(row);
    }
    rows
}

fn nearest(query: &[f64], bank: &[Vec<f64>], bank_y: &[usize]) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut best_y = bank_y[0];
    for (x, &y) in bank.iter().zip(bank_y.iter()) {
        let s = cosine(query, x);
        if s > best {
            best = s;
            best_y = y;
        }
    }
    best_y
}

fn max_forgetting(rows: &[Vec<f64>]) -> f64 {
    let last = rows.last().expect("at least one task");
    let mut worst = 0.0_f64;
    for i in 0..CLASSES {
        let best = rows
            .iter()
            .filter(|r| r[i].is_finite())
            .map(|r| r[i])
            .fold(f64::NEG_INFINITY, f64::max);
        if best.is_finite() && last[i].is_finite() {
            worst = worst.max(best - last[i]);
        }
    }
    worst
}

fn render(rows: &[Vec<f64>]) -> String {
    rows.iter()
        .enumerate()
        .map(|(t, row)| {
            let cells: Vec<String> = row
                .iter()
                .map(|v| {
                    if v.is_finite() {
                        format!("{v:.2}")
                    } else {
                        " -- ".to_string()
                    }
                })
                .collect();
            format!("  t={t}: [{}]", cells.join(" "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The measured artifact: the QFM-refit arm's matrix, its max forgetting,
/// and the raw-NN baseline for attribution. Determinism is asserted by
/// running the flow arm twice — a seeded fixture with a fixed S1 seed must
/// not drift.
#[test]
fn d_c6_forgetting_matrix_over_qfm_tsr_arrival() {
    let flow = run_protocol(true);
    let again = run_protocol(true);
    for (a, b) in flow.iter().zip(again.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert_eq!(x, y, "protocol must be deterministic on the seeded fixture");
        }
    }
    let base = run_protocol(false);

    let flow_f = max_forgetting(&flow);
    let base_f = max_forgetting(&base);
    let flow_final: Vec<f64> = flow.last().unwrap().to_vec();
    let base_final: Vec<f64> = base.last().unwrap().to_vec();

    println!("QFM-TSR D-C6 matrix (refit-per-task, cosine-NN attribution on generate()):");
    println!("{}", render(&flow));
    println!("raw-NN baseline matrix (no flow, attribution on the query itself):");
    println!("{}", render(&base));
    println!(
        "max_forgetting: QFM-refit = {flow_f:.4}, raw-NN baseline = {base_f:.4}; \
         final row: QFM-refit = {flow_final:?}, baseline = {base_final:?}"
    );

    // Protocol invariants: the seen-set grows by TASK_CLASSES per row, the
    // matrix cells are accuracies, and the scalars are bounded.
    for (t, row) in flow.iter().enumerate() {
        let seen = (t + 1) * TASK_CLASSES;
        for (i, v) in row.iter().enumerate() {
            if i < seen {
                assert!(v.is_finite(), "seen cell t={t} i={i} must be defined");
                assert!((0.0..=1.0).contains(v), "accuracy out of range: {v}");
            } else {
                assert!(v.is_nan(), "unseen cell t={t} i={i} must be undef");
            }
        }
    }
    assert!((0.0..=1.0).contains(&flow_f));
    assert!((0.0..=1.0).contains(&base_f));
    // The baseline is a *protocol sanity check*, not an outcome claim: the
    // fixture is designed to be perfectly separable in raw space, so any
    // nonzero baseline forgetting would mean the protocol (not the flow)
    // is measuring interference — and the flow arm's number could no longer
    // be attributed to the flow. If a future fixture revision makes the
    // classes genuinely overlapping, this assert fires by design.
    assert_eq!(
        base_f, 0.0,
        "raw-NN baseline must forget nothing on this separable fixture; \
         a nonzero value invalidates the attribution of the flow arm's \
         forgetting to the flow rather than to bank growth"
    );
    // The flow arm's *value* is deliberately not pinned: it is the
    // measurement, and the rev 37 diffusion-Hamiltonian work targets the
    // collapse it currently exhibits. Pinning it here would turn a
    // pipeline improvement into a test failure. It is asserted bounded
    // (above) and printed (below) instead.
}
