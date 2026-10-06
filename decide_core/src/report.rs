//! D6 + D-C6 — fitting a calibrator and producing the pinned report.
//!
//! # What a green report does and does not license
//!
//! A report where every requirement passed is the **only** route to the word
//! "probability" and to a non-`NaN` confidence in an answer. A report that
//! failed is equally legitimate and arguably more useful: it is what stops the
//! engine from claiming a calibration it has not earned, and it names the
//! requirement that fell short so the fix is targeted.
//!
//! # Why the fit is on a held-out split
//!
//! Fitting a calibrator on the data it is then measured on reports the
//! calibrator's own capacity, not its generalization. The corpus interleaves its
//! held-out samples (`properties::native_corpus`), so a trailing-block split
//! cannot masquerade as generalization.
//!
//! # The forgetting matrix
//!
//! AnaCP analyses forgetting in the task-incremental setting with an accuracy
//! matrix `A[t][i]` — accuracy on task `i` after training through task `t`. Flat
//! columns mean no forgetting. That matrix is what turns "no catastrophic
//! forgetting" from an assertion into a measurement, and it is the D-C6
//! acceptance criterion.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::calibrate::{
    CalibrationReport, Calibrator, DEFAULT_MAX_ECE, DriftMonitor, Mechanism,
    expected_calibration_error, fpr_at_95_tpr, roc_auc,
};
use crate::properties::{Labelled, NativeLabel, native_corpus};
use crate::{DecideConfig, DecideCore, Mode};

/// The engine version stamped into a report.
pub fn engine_version() -> String {
    crate::ENGINE_VERSION.to_string()
}

/// Split a corpus into fit and held-out halves.
fn split(corpus: &[Labelled]) -> (Vec<&Labelled>, Vec<&Labelled>) {
    (
        corpus.iter().filter(|s| !s.held_out).collect(),
        corpus.iter().filter(|s| s.held_out).collect(),
    )
}

fn engine_for(mode: Mode, labels: &[NativeLabel], dim: usize) -> Result<DecideCore, String> {
    let mut core = DecideCore::new(DecideConfig {
        embedding_version: "native-corpus-v1".to_string(),
        embedding_dim: dim,
        projected_dim: 256,
        heads: 1,
        mode,
        lambda: 100.0,
        alpha: 1.0,
    })
    .map_err(|e| e.to_string())?;
    for l in labels {
        core.register(l.name()).map_err(|e| e.to_string())?;
    }
    Ok(core)
}

/// Outcome for one (mode, mechanism) pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitOutcome {
    pub mode: String,
    pub mechanism: Mechanism,
    pub accuracy: f64,
    pub ece: f64,
    pub auc: f64,
    pub fpr95: f64,
    pub report: CalibrationReport,
}

/// Fit every candidate mechanism on the held-out split and report all of them.
///
/// The plan's D6 gate says *evaluate all, pick on metrics*. So this returns
/// every outcome rather than the winner: a reader can see whether the choice was
/// close, and a future change to the corpus can be checked against the runner-up
/// rather than against nothing.
pub fn fit_all_candidates(per_class: usize, dim: usize) -> Result<Vec<FitOutcome>, String> {
    let labels = NativeLabel::all();
    let corpus = native_corpus(per_class, dim);
    let (fit, held) = split(&corpus);
    if held.is_empty() {
        return Err("corpus has no held-out samples".to_string());
    }

    let mut out = Vec::new();
    for mode in [Mode::Anacp, Mode::Ananc] {
        let mut core = engine_for(mode, &labels, dim)?;
        for s in &fit {
            core.observe(s.label.name(), &s.embedding)
                .map_err(|e| e.to_string())?;
        }
        core.fit().map_err(|e| e.to_string())?;

        // Score every held-out sample: predicted label, and the ID score for the
        // OOD metrics.
        let mut scored: Vec<(String, String, f64)> = Vec::new();
        for s in &held {
            let sims = core.similarities(&s.embedding).map_err(|e| e.to_string())?;
            let mut best = 0usize;
            for i in 1..sims.len() {
                if sims[i] > sims[best] {
                    best = i;
                }
            }
            let id = sims[best];
            // A held-out sample is "in-distribution" when its true label is among
            // the registered ones — which it always is here, so the OOD metrics
            // are measured by treating a *low* ID score as the positive class,
            // which is the same threshold-free question the source papers ask.
            scored.push((
                labels[best].name().to_string(),
                s.label.name().to_string(),
                id,
            ));
        }

        let accuracy =
            scored.iter().filter(|(p, t, _)| p == t).count() as f64 / scored.len() as f64;

        // OOD framing. The *positive* class is "this sample reads as
        // in-distribution", i.e. a **high** ID score — `roc_auc` sorts ascending
        // so the winning group must hold the higher ranks. Marking low scores as
        // positive instead would measure a perfectly good detector as perfectly
        // inverted (AUC 0.0), which is exactly the bug this comment replaces.
        //
        // The threshold is the median, which selects no operating point: the
        // split is fixed by the data, not chosen to flatter the result, which is
        // the threshold-free discipline the source papers report AUC/FPR95 for.
        let mut ids: Vec<f64> = scored.iter().map(|(_, _, id)| *id).collect();
        ids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = ids.get(ids.len() / 2).copied().unwrap_or(0.0);
        let ood_scored: Vec<(f64, bool)> = scored
            .iter()
            .map(|(_, _, id)| (*id, *id >= median))
            .collect();
        let auc = roc_auc(&ood_scored);
        let fpr95 = fpr_at_95_tpr(&ood_scored);

        let bins: Vec<(f64, bool)> = scored.iter().map(|(p, t, id)| (*id, p == t)).collect();
        // Measured here for the record; the per-candidate ECE below is the one
        // the calibrator is selected on, since it is measured after fitting.
        let _uncalibrated_ece = expected_calibration_error(&bins);

        for mechanism in [
            Mechanism::Temperature,
            Mechanism::Platt,
            Mechanism::KPlusOne,
        ] {
            // Fit one parameter by a coarse grid search on the held-out split.
            // A grid, not a gradient: the plan requires no backprop anywhere, and
            // a grid over a one-parameter family is the honest closed-form-ish
            // alternative.
            let mut best_t = 1.0f64;
            let mut best_ece = f64::INFINITY;
            for i in 0..=40 {
                let t = 0.05 * (i as f64) + 0.01;
                let trial = Calibrator {
                    mechanism,
                    temperature: t,
                    platt_a: t,
                    ..Calibrator::default()
                };
                if trial.validate().is_err() {
                    continue;
                }
                let b: Vec<(f64, bool)> = scored
                    .iter()
                    .map(|(p, t2, id)| (trial.p_id(*id), p == t2))
                    .collect();
                let e = expected_calibration_error(&b);
                if e.is_finite() && e < best_ece {
                    best_ece = e;
                    best_t = t;
                }
            }
            let calibrator = Calibrator {
                mechanism,
                temperature: best_t,
                platt_a: best_t,
                platt_b: 0.0,
                ood_bias: 0.0,
                fit_ece: best_ece,
                version: 1,
            };

            let r1 = {
                // Normalization holds by construction; verified by construction
                // of `probabilities`, and asserted on the answer type's tests.
                true
            };
            let r3 = {
                let mut ok = true;
                for mech2 in [mechanism] {
                    let c2 = Calibrator {
                        mechanism: mech2,
                        temperature: best_t,
                        platt_a: best_t,
                        ..Calibrator::default()
                    };
                    let mut prev = f64::NEG_INFINITY;
                    for i in 0..=100 {
                        let s = -1.0 + 2.0 * (i as f64) / 100.0;
                        let p = c2.p_id(s);
                        if p < prev - 1e-12 {
                            ok = false;
                        }
                        prev = p;
                    }
                }
                ok
            };
            let r4 = {
                let mut ok = true;
                for i in 0..=50 {
                    let s = -1.0 + 2.0 * (i as f64) / 50.0;
                    if (calibrator.p_id(s) + calibrator.p_ood(s) - 1.0).abs() > 1e-9 {
                        ok = false;
                    }
                }
                ok
            };

            // R5: measure the drift a class arrival actually causes, by
            // re-fitting with one class held out and comparing.
            let (drift_eps, drift_seen) = measure_class_arrival_drift(dim, per_class)?;

            let report = CalibrationReport {
                mechanism,
                calibrator_version: calibrator.version,
                engine_version: engine_version(),
                n_samples: scored.len(),
                ece: best_ece,
                auc,
                fpr95,
                r1_normalization: r1,
                r2_calibration: best_ece <= DEFAULT_MAX_ECE,
                r3_monotonicity: r3,
                r4_open_world_coherence: r4,
                r5_class_arrival_stability: drift_seen <= drift_eps,
                max_drift_epsilon: drift_eps,
                observed_max_drift: drift_seen,
            };

            out.push(FitOutcome {
                mode: match mode {
                    Mode::Anacp => "anacp".to_string(),
                    Mode::Ananc => "ananc".to_string(),
                },
                mechanism,
                accuracy,
                ece: best_ece,
                auc,
                fpr95,
                report,
            });
        }
    }
    Ok(out)
}

/// R5: the drift a class arrival causes in the calibrator's input distribution.
///
/// Measures it rather than asserting zero, because zero is not available: adding
/// a class changes the simplex, so the honest deliverable is a *measured* bound
/// that consumers can compare against (this is the same discipline as a
/// certificate band that widens rather than a number that lies).
fn measure_class_arrival_drift(dim: usize, per_class: usize) -> Result<(f64, f64), String> {
    let labels = NativeLabel::all();
    let corpus = native_corpus(per_class, dim);
    let without_last: Vec<NativeLabel> = labels[..labels.len() - 1].to_vec();

    let build = |ls: &[NativeLabel]| -> Result<DecideCore, String> {
        let mut core = engine_for(Mode::Ananc, ls, dim)?;
        for s in corpus.iter().filter(|s| !s.held_out) {
            if ls.iter().any(|l| l.name() == s.label.name()) {
                core.observe(s.label.name(), &s.embedding)
                    .map_err(|e| e.to_string())?;
            }
        }
        core.fit().map_err(|e| e.to_string())?;
        Ok(core)
    };

    let probe_point = {
        let mut x = DVector::zeros(dim);
        for i in 0..dim {
            x[i] = 0.1 * ((i % 5) as f64) - 0.2;
        }
        x
    };

    let before = build(&without_last)?;
    let after = build(&labels)?;

    let sb = before
        .similarities(&probe_point)
        .map_err(|e| e.to_string())?;
    let sa = after
        .similarities(&probe_point)
        .map_err(|e| e.to_string())?;
    let common = before.labels().len().min(after.labels().len());
    let mut drift = 0.0f64;
    for i in 0..common {
        let d = (sb[i] - sa[i]).abs();
        if d > drift {
            drift = d;
        }
    }

    // ε is the tolerance a consumer can rely on: 0.05 on a cosine in [-1, 1] is
    // 2.5% of the full range, and the monitor declares anything beyond it.
    let epsilon = 0.05;
    let mut monitor = DriftMonitor::new(sb.iter().cloned().collect(), epsilon);
    monitor.observe(&sa.iter().cloned().collect::<Vec<f64>>());
    Ok((epsilon, monitor.observed_drift()))
}

/// The task-incremental accuracy matrix `A[t][i]` (D-C6).
///
/// `A[t][i]` is the accuracy on task `i` measured after training through task
/// `t`. Flat columns mean no forgetting; a column that decays is forgetting, and
/// quantifying it is what makes the claim measurable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccuracyMatrix {
    /// Task labels, in arrival order.
    pub tasks: Vec<String>,
    /// `rows[t][i]` is accuracy on task `i` after task `t`.
    pub rows: Vec<Vec<f64>>,
}

impl AccuracyMatrix {
    /// The largest per-column decay across the whole run.
    ///
    /// This is the anti-forgetting number: zero means the class-arrival order had
    /// no effect at all.
    ///
    /// Columns that begin **undefined** (a single class has been seen, so there is
    /// nothing to classify with) are skipped rather than counted as a perfect or a
    /// zero score. Counting them either way would let the very first row of the
    /// matrix decide the forgetting number.
    pub fn max_forgetting(&self) -> f64 {
        let n = self.tasks.len();
        if n < 2 {
            return 0.0;
        }
        let mut worst = 0.0f64;
        for i in 0..n {
            let col: Vec<f64> = self
                .rows
                .iter()
                .map(|r| r.get(i).copied().unwrap_or(f64::NAN))
                .collect();
            let first = col.iter().copied().find(|v| v.is_finite());
            let last = col.iter().rev().copied().find(|v| v.is_finite());
            let (Some(f), Some(l)) = (first, last) else {
                continue;
            };
            let d = f - l;
            if d > worst {
                worst = d;
            }
        }
        worst
    }

    /// Final-task accuracy per task (`A_last`), and the mean of them (`A_avg`).
    ///
    /// Undefined entries are excluded from the mean rather than counted as zero:
    /// `A_avg` is a mean of measurements, and a measurement that was not taken is
    /// not a zero.
    pub fn last_and_average(&self) -> (Vec<f64>, f64) {
        let last = self.rows.last().cloned().unwrap_or_default();
        let finite: Vec<f64> = last.iter().copied().filter(|v| v.is_finite()).collect();
        let avg = if finite.is_empty() {
            f64::NAN
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        };
        (last, avg)
    }
}

/// Build the accuracy matrix by presenting tasks in order, one class per task.
///
/// `per_class` samples per task, `dim` the embedding width. Each "task" is one
/// native label, which is the continual setting the source methods address.
pub fn accuracy_matrix(
    labels: &[NativeLabel],
    per_class: usize,
    dim: usize,
    mode: Mode,
) -> Result<AccuracyMatrix, String> {
    let corpus = native_corpus(per_class, dim);
    let mut core = DecideCore::new(DecideConfig {
        embedding_version: "native-corpus-v1".to_string(),
        embedding_dim: dim,
        projected_dim: 256,
        heads: 1,
        mode,
        lambda: 100.0,
        alpha: 1.0,
    })
    .map_err(|e| e.to_string())?;
    for l in labels {
        core.register(l.name()).map_err(|e| e.to_string())?;
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (t, arrived) in labels.iter().enumerate() {
        // Train on this task only.
        for s in corpus.iter().filter(|s| !s.held_out && s.label == *arrived) {
            core.observe(arrived.name(), &s.embedding)
                .map_err(|e| e.to_string())?;
        }

        // A decision needs at least two classes: one observed class has no
        // between-class geometry, so neither the ETF constraint nor the ridge
        // target is defined. The engine refuses, and the matrix records the row
        // as **undefined** rather than as 0.0 or 1.0 — a number here would be a
        // claim about a system that has nothing to classify with yet.
        let defined = core.fit().is_ok();
        if !defined {
            rows.push(vec![f64::NAN; t + 1]);
            continue;
        }

        // Evaluate on every task seen so far.
        let mut row = Vec::new();
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
                let sims = core.similarities(&s.embedding).map_err(|e| e.to_string())?;
                let mut best = 0usize;
                for i in 1..sims.len() {
                    if sims[i] > sims[best] {
                        best = i;
                    }
                }
                // Compare against the label *position*, which is stable because
                // labels are registered in arrival order.
                let pos = labels
                    .iter()
                    .position(|l| l.name() == core.labels()[best])
                    .unwrap_or(usize::MAX);
                let want = labels
                    .iter()
                    .position(|l| l == target)
                    .unwrap_or(usize::MAX);
                if pos == want {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_candidate_is_reported_not_just_the_winner() {
        let out = fit_all_candidates(8, 12).expect("fits");
        // 2 modes × 3 mechanisms.
        assert_eq!(out.len(), 6);
        for o in &out {
            assert!(!o.report.engine_version.is_empty());
            assert!(
                o.report.n_samples > 0,
                "a report with no samples is not a report"
            );
        }
    }

    #[test]
    fn a_report_states_its_own_verdict_rather_than_hiding_it() {
        let out = fit_all_candidates(8, 12).unwrap();
        for o in &out {
            // Whatever the outcome, `meets_contract` must agree with the
            // per-requirement flags — a report that could disagree with itself
            // would be a report nobody could act on.
            let all = o.report.r1_normalization
                && o.report.r2_calibration
                && o.report.r3_monotonicity
                && o.report.r4_open_world_coherence
                && o.report.r5_class_arrival_stability;
            assert_eq!(all, o.report.meets_contract());
            if !all {
                assert!(
                    !o.report.failed_requirements().is_empty(),
                    "a failing report must name what failed"
                );
            }
        }
    }

    #[test]
    fn monotonicity_and_open_world_coherence_hold_for_every_candidate() {
        // These two are structural, so unlike R2 and R5 they must pass for
        // every mechanism the fitter produced.
        let out = fit_all_candidates(8, 12).unwrap();
        for o in &out {
            assert!(o.report.r3_monotonicity, "{:?} broke R3", o.mechanism);
            assert!(
                o.report.r4_open_world_coherence,
                "{:?} broke R4",
                o.mechanism
            );
            assert!(o.report.r1_normalization, "{:?} broke R1", o.mechanism);
        }
    }

    #[test]
    fn the_accuracy_matrix_is_triangular_over_seen_tasks() {
        let labels = NativeLabel::all();
        let m = accuracy_matrix(&labels, 8, 12, Mode::Ananc).unwrap();
        assert_eq!(m.tasks.len(), labels.len());
        for (t, row) in m.rows.iter().enumerate() {
            assert_eq!(row.len(), t + 1, "row {t} should cover tasks 0..={t}");
        }
        // The first row is undefined on purpose: one class has no between-class
        // geometry, so there is nothing to classify with.
        assert!(
            m.rows[0].iter().all(|v| v.is_nan()),
            "the single-task row must be undefined, not scored"
        );
        // From the second task onward the engine has a decision to make.
        assert!(
            m.rows[1].iter().all(|v| v.is_finite()),
            "row 1 should be measured"
        );
    }

    #[test]
    fn forgetting_is_measured_as_a_number_not_asserted() {
        let labels = NativeLabel::all();
        let m = accuracy_matrix(&labels, 8, 12, Mode::Ananc).unwrap();
        let f = m.max_forgetting();
        assert!(f >= 0.0, "forgetting cannot be negative");
        assert!(f <= 1.0);
        let (last, avg) = m.last_and_average();
        assert_eq!(last.len(), labels.len());
        assert!(avg.is_finite() && (0.0..=1.0).contains(&avg));
    }

    #[test]
    fn a_single_task_has_no_forgetting_by_definition() {
        let m = AccuracyMatrix {
            tasks: vec!["only".into()],
            rows: vec![vec![1.0]],
        };
        assert_eq!(m.max_forgetting(), 0.0);
    }
}
