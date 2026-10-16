//! The sequential AnaCP→AnaNC model: contract tests plus a report-style
//! measurement against the two single-mode surfaces.
//!
//! # What is asserted vs reported (same discipline as `order_sensitivity.rs`)
//!
//! **Asserted** (must hold for any engine revision): the handoff vector is
//! unit-norm and stage-2-wide; a fitted sequence's stage-2 statistics cover
//! exactly the retained stream (the rebuild invariant); the staging config
//! refuses mode order and width mismatches; the ID score exists for the
//! sequence and *does not* exist for a pure-AnaCP surface (the structural gain);
//! the accuracy matrix protocol invariants (row shapes, values in [0, 1]).
//!
//! **Reported** (the measurement, deliberately unpinned): accuracy and
//! forgetting per configuration. Pinning those would turn any future method
//! improvement into a test failure — the corpus is synthetic, so these numbers
//! compare mechanisms on it and license nothing beyond it.

use decide_core::properties::{Labelled, NativeLabel, native_corpus};
use decide_core::report::{AccuracyMatrix, accuracy_matrix_at};
use decide_core::sequential::{SequentialConfig, SequentialCore};
use decide_core::{DecideConfig, DecideCore, DecideError, Mode};
use nalgebra::DVector;

/// Protocol constants of the pinned fixtures (same as `order_sensitivity`).
const PER_CLASS: usize = 24;
const CORPUS_DIM: usize = 24;
/// The corpus's embedding-model identity, as `report.rs` pins it.
const CORPUS_VERSION: &str = "native-corpus-v1";

fn base_config(embedding_dim: usize, projected_dim: usize, mode: Mode) -> DecideConfig {
    DecideConfig {
        embedding_version: CORPUS_VERSION.to_string(),
        embedding_dim,
        projected_dim,
        heads: 1,
        mode,
        lambda: 100.0,
        alpha: 1.0,
    }
}

fn seq_config(d_cp: usize, d_nc: usize) -> SequentialConfig {
    SequentialConfig {
        cp: base_config(CORPUS_DIM, d_cp, Mode::Anacp),
        nc: base_config(d_cp, d_nc, Mode::Ananc),
    }
}

// ── 1. the staging contract ─────────────────────────────────────────────────

#[test]
fn staging_config_refuses_wrong_modes_and_widths() {
    // Reversed stages: collapse-before-adaptation is not the sequence.
    let wrong_stage1 = SequentialConfig {
        cp: base_config(CORPUS_DIM, 16, Mode::Ananc),
        nc: base_config(16, 8, Mode::Ananc),
    };
    assert!(matches!(
        SequentialCore::new(wrong_stage1),
        Err(DecideError::Staging(_))
    ));
    // Adaptation after collapse: also not the sequence.
    let wrong_stage2 = SequentialConfig {
        cp: base_config(CORPUS_DIM, 16, Mode::Anacp),
        nc: base_config(16, 8, Mode::Anacp),
    };
    assert!(matches!(
        SequentialCore::new(wrong_stage2),
        Err(DecideError::Staging(_))
    ));
    // Width break: stage 2 must consume stage 1's features exactly.
    let broken_width = SequentialConfig {
        cp: base_config(CORPUS_DIM, 16, Mode::Anacp),
        nc: base_config(8, 8, Mode::Ananc), // 8 ≠ 16
    };
    assert!(matches!(
        SequentialCore::new(broken_width),
        Err(DecideError::Staging(_))
    ));
}

// ── 2. the handoff contract ─────────────────────────────────────────────────

#[test]
fn handoff_is_unit_norm_stage2_wide_and_never_stale() {
    let d_cp = 16;
    let d_nc = 8;
    let mut seq = SequentialCore::new(seq_config(d_cp, d_nc)).unwrap();
    for l in ["alpha", "beta", "gamma"] {
        seq.register(l).unwrap();
    }
    let x = DVector::from_vec(vec![0.5; CORPUS_DIM]);

    // Nothing answers before a fit: without a solve there is no theta, and a
    // raw RP expansion must not masquerade as an adapted feature.
    assert!(matches!(seq.project(&x), Err(DecideError::NotFitted)));
    assert!(matches!(seq.similarities(&x), Err(DecideError::NotFitted)));
    assert!(matches!(seq.id_score(&x), Err(DecideError::NotFitted)));

    // One observation per class is enough for both stages to solve.
    for (li, l) in ["alpha", "beta", "gamma"].into_iter().enumerate() {
        let mut v = vec![0.0; CORPUS_DIM];
        v[li] = 1.0;
        seq.observe(l, &DVector::from_vec(v)).unwrap();
    }
    seq.fit().unwrap();

    // The handoff: unit-norm, and exactly stage 2's input width.
    let f = seq.project(&x).unwrap();
    assert_eq!(f.len(), d_cp, "handoff carries stage 1's full width");
    assert!(
        (f.norm() - 1.0).abs() < 1e-9,
        "handoff must be L2-normalized, got norm {}",
        f.norm()
    );
    assert_eq!(seq.stage2().config().embedding_dim, f.len());
    assert_eq!(
        seq.stage2().samples() as usize,
        seq.samples(),
        "the rebuild feeds every retained observation under the current map"
    );

    // The composition inherits AnaNC's open-world mechanism…
    let id = seq.id_score(&x).unwrap();
    assert!(id.is_some(), "the sequence has an ID score");
    let id = id.unwrap();
    assert!(
        (-1.0..=1.0).contains(&id),
        "ID score is a max-cosine, got {id}"
    );

    // …which a pure-AnaCP surface does not have. Structural, not measured.
    let mut cp_only = DecideCore::new(base_config(CORPUS_DIM, d_cp, Mode::Anacp)).unwrap();
    for l in ["alpha", "beta", "gamma"] {
        cp_only.register(l).unwrap();
    }
    for (li, l) in ["alpha", "beta", "gamma"].into_iter().enumerate() {
        let mut v = vec![0.0; CORPUS_DIM];
        v[li] = 1.0;
        cp_only.observe(l, &DVector::from_vec(v)).unwrap();
    }
    cp_only.fit().unwrap();
    assert_eq!(
        cp_only.id_score(&x).unwrap(),
        None,
        "AnaCP honestly has no OOD mechanism; only the sequence carries one"
    );
}

#[test]
fn refit_after_new_observations_reprojects_the_whole_stream() {
    let mut seq = SequentialCore::new(seq_config(16, 8)).unwrap();
    for l in ["a", "b"] {
        seq.register(l).unwrap();
    }
    seq.observe("a", &DVector::from_vec(vec![1.0; CORPUS_DIM]))
        .unwrap();
    seq.observe("b", &DVector::from_vec(vec![-1.0; CORPUS_DIM]))
        .unwrap();
    seq.fit().unwrap();
    let first = seq
        .project(&DVector::from_vec(vec![0.7; CORPUS_DIM]))
        .unwrap();

    // A late arrival moves stage 1's theta, so stage 2 must be rebuilt —
    // answering under the old projection would be defect A4 at sequence level.
    seq.observe("a", &DVector::from_vec(vec![1.2; CORPUS_DIM]))
        .unwrap();
    assert!(!seq.is_fitted(), "observe invalidates the fit");
    assert!(matches!(
        seq.similarities(&DVector::from_vec(vec![0.7; CORPUS_DIM])),
        Err(DecideError::NotFitted)
    ));
    seq.fit().unwrap();
    let second = seq
        .project(&DVector::from_vec(vec![0.7; CORPUS_DIM]))
        .unwrap();
    assert_eq!(seq.stage2().samples() as usize, seq.samples());
    assert!(
        (first - second).norm() > 0.0,
        "stage 1 re-solved, so the handoff moved — and stage 2 followed"
    );
}

// ── 3. the measurement (report-style: asserted invariants, reported numbers) ─

/// The continual-arrival accuracy matrix for the *sequence*, mirroring
/// `report::accuracy_matrix_at`'s protocol exactly (all labels registered
/// up front, one class per task, held-out interleaved split, undefined first
/// row recorded as NaN).
fn sequential_matrix(
    labels: &[NativeLabel],
    per_class: usize,
    dim: usize,
    d_cp: usize,
    d_nc: usize,
) -> Result<AccuracyMatrix, DecideError> {
    let corpus = native_corpus(per_class, dim);
    let mut seq = SequentialCore::new(seq_config(d_cp, d_nc))?;
    for l in labels {
        seq.register(l.name())?;
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (t, arrived) in labels.iter().enumerate() {
        for s in corpus.iter().filter(|s| !s.held_out && s.label == *arrived) {
            seq.observe(arrived.name(), &s.embedding)?;
        }
        // One observed class has no between-class geometry; record undefined
        // rather than 0.0 or 1.0 (same rule as the single-mode protocol).
        if seq.fit().is_err() {
            rows.push(vec![f64::NAN; t + 1]);
            continue;
        }
        let mut row = Vec::with_capacity(t + 1);
        for target in labels.iter().take(t + 1) {
            let samples: Vec<&Labelled> = corpus
                .iter()
                .filter(|s| s.held_out && s.label == *target)
                .collect();
            if samples.is_empty() {
                row.push(f64::NAN);
                continue;
            }
            let mut correct = 0usize;
            for s in &samples {
                let sims = seq.similarities(&s.embedding)?;
                let mut best = 0usize;
                for i in 1..sims.len() {
                    if sims[i] > sims[best] {
                        best = i;
                    }
                }
                if seq.labels().get(best).map(String::as_str) == Some(target.name()) {
                    correct += 1;
                }
            }
            row.push(correct as f64 / samples.len() as f64);
        }
        rows.push(row);
    }
    Ok(AccuracyMatrix {
        tasks: labels.iter().map(|l| l.name().to_string()).collect(),
        rows,
    })
}

fn assert_matrix_invariants(m: &AccuracyMatrix, who: &str) {
    assert_eq!(m.rows.len(), 5, "{who}: one row per arrival step");
    for (t, row) in m.rows.iter().enumerate() {
        assert_eq!(row.len(), t + 1, "{who}: row {t} covers the classes seen");
        for v in row {
            if v.is_finite() {
                assert!(
                    (0.0..=1.0).contains(v),
                    "{who}: accuracies live in [0,1], got {v}"
                );
            }
        }
    }
    let f = m.max_forgetting();
    assert!(
        (0.0..=1.0).contains(&f),
        "{who}: forgetting is a decay in [0,1], got {f}"
    );
}

#[test]
fn sequential_matrix_invariants_and_report() {
    let labels = NativeLabel::all();

    // Each configuration at the width its own surface was measured best at in
    // the pinned D-ablation — which is exactly the point of staging: the two
    // stages no longer share one knob.
    let anacp = accuracy_matrix_at(&labels, PER_CLASS, CORPUS_DIM, 32, Mode::Anacp)
        .expect("the single-mode protocol runs");
    let ananc = accuracy_matrix_at(&labels, PER_CLASS, CORPUS_DIM, 16, Mode::Ananc)
        .expect("the single-mode protocol runs");
    let seq_32_16 = sequential_matrix(&labels, PER_CLASS, CORPUS_DIM, 32, 16)
        .expect("the sequential protocol runs");
    let seq_32_32 = sequential_matrix(&labels, PER_CLASS, CORPUS_DIM, 32, 32)
        .expect("the sequential protocol runs");

    for (who, m) in [
        ("anacp@32", &anacp),
        ("ananc@16", &ananc),
        ("seq@32->16", &seq_32_16),
        ("seq@32->32", &seq_32_32),
    ] {
        assert_matrix_invariants(m, who);
        let (_, avg) = m.last_and_average();
        eprintln!(
            "sequential-rd {who}: max_forgetting={:.3} final_acc_avg={:.3} matrix={:?}",
            m.max_forgetting(),
            avg,
            m.rows
        );
    }
}
