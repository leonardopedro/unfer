//! The `D`-ablation: AnaCP Table 2 (rows 13–16) re-run on project data.
//!
//! # Why this module exists
//!
//! `docs/DECIDE_SPEC.md` §4 says the engine's projected dimension is *sensible,
//! not measured*: the plan requires AnaCP's `D`-ablation to be re-run on
//! project data before any default is blessed, and until this module ran, that
//! had not happened. This is that ablation. It sweeps `projected_dim` over the
//! grid in [`DEFAULT_DIMS`] for both modes and measures, per cell:
//!
//! * **held-out accuracy** — through the same scoring protocol as the pinned
//!   calibration report ([`crate::report`]);
//! * **`ece`** — the best of the three D6 calibrator mechanisms, on the same
//!   grid search the report uses, so the numbers are comparable to that report;
//! * **`A_avg` and `max_forgetting`** — the task-incremental accuracy matrix,
//!   AnaCP's own anti-forgetting protocol (D-C6);
//! * **`state_floats`** — the persistent analytic state at that `D`, which is
//!   the cost the ablation is trading accuracy against.
//!
//! # What the sweep refuses, and why that is a result
//!
//! `ananc`'s ETF target needs `D ≥ C` (the Procrustes precondition in
//! [`crate::targets::etf_procrustes`], enforced there by an `assert!`). The
//! sweep therefore contains a sub-class cell (`D = 4` against 5 classes) that
//! the engine would *panic* on. This harness records it as a **refused** cell
//! instead of crashing, and `the_engine_really_does_refuse_a_sub_class_etf`
//! pins the textual precondition to the engine's own assert so the two cannot
//! drift apart. `anacp` has no such precondition and is measured there.
//!
//! One measurement is shared rather than per-row: R5's drift. The drift
//! monitor builds `ananc` engines (it is the only mode the report arms it
//! for), so `AblationRow::drift` is a function of `D` alone and repeats across
//! the two modes at the same `D`. It is not a copied cell, and it is `None`
//! below `D ≥ C` where the monitor cannot be built at all.
//!
//! # Honesty rules, inherited from the report
//!
//! The corpus is the same synthetic `properties::native_corpus`, so this is a
//! mechanism check and a *relative* comparison across `D` — not an accuracy
//! claim on real data. The pinned fixture says so in its `caveats`, and the
//! numbers are rounded to 6 decimals so that last-bit floating-point noise
//! cannot fail a byte-comparison gate while still pinning every digit the
//! corpus can actually support.

use serde::{Deserialize, Serialize};

use crate::properties::{Labelled, NativeLabel, native_corpus};
use crate::report;
use crate::{DecideConfig, DecideCore, Mode, calibrate::Mechanism};

/// The corpus the sweep runs on — identical to the pinned calibration report's,
/// so the two artifacts describe the same data.
pub const DEFAULT_PER_CLASS: usize = 24;

/// Embedding width of that corpus.
pub const DEFAULT_DIM: usize = 24;

/// The sweep grid.
///
/// Two anchors and the doubling sequence between them: `4` sits *below* the
/// class count on purpose (it is the cell the ETF precondition refuses), and
/// `512` is the engine's current default, so every cell smaller than the
/// default is measured against it. `256` is the pinned report's own `D`, which
/// makes the cross-check test possible.
pub const DEFAULT_DIMS: [usize; 8] = [4, 8, 16, 32, 64, 128, 256, 512];

/// The committed fixture. Relative to the crate root, so it moves with the code.
pub const FIXTURE: &str = "tests/fixtures/decide_d_ablation.json";

/// One measured cell of the sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationRow {
    pub mode: String,
    pub projected_dim: usize,
    /// `Some(reason)` when the method's own precondition refuses this `D`, in
    /// which case every measured field is `None` — an unmeasured number must
    /// not look like a measured zero.
    pub refused: Option<String>,
    /// Held-out accuracy, through the report's scoring protocol.
    pub accuracy: Option<f64>,
    /// Best ECE across the three D6 mechanisms on the report's own grid.
    pub ece: Option<f64>,
    /// `A_avg` of the task-incremental accuracy matrix.
    pub a_avg: Option<f64>,
    /// The matrix's largest per-column decay — the anti-forgetting number.
    pub max_forgetting: Option<f64>,
    /// R5's measured class-arrival drift at this `D` (ε is 0.05 everywhere).
    ///
    /// Recorded because R2 and R5 are the two requirements standing between the
    /// engine and `calibrated: true`, and a `D`-ablation that reported only the
    /// ECE half would leave the reader unable to tell whether the *other*
    /// failing bar moves with `D`.
    ///
    /// `None` at `D < C`: the monitor builds `ananc` engines and so is subject
    /// to the same ETF precondition. That is "undefined", distinct from both
    /// "measured at 0.0" and "refused".
    pub drift: Option<f64>,
    /// How many RP-covariance eigenvalues the whitening half-power floored at
    /// `1e-12` during the fit — the directions whitening silently destroyed.
    ///
    /// This is the *explanation* column, read from the instrument the crate
    /// already exposes for it ([`crate::targets::half_power`]: "a degenerate
    /// covariance silently changes the geometry, and the consumer is entitled
    /// to know"). It is why the two modes bend differently:
    /// `n_train = 90` on this corpus caps the covariance rank at 89, so every
    /// `D ≥ 128` is rank-deficient and whitening floors `D − 89` directions.
    /// `ananc`'s Procrustes-onto-ETF is geometry-critical and degrades;
    /// `anacp`'s spread means tolerate it. Measured, not asserted.
    ///
    /// Always `0` for `ananc`: its ETF construction does **no whitening**, so
    /// it never floors anything. (The instrument is cleared before each fit —
    /// otherwise an `ananc` cell would report the previous `anacp` cell's
    /// floors, which is precisely how this column lied on its first run.)
    pub floored_eigenvalues: Option<u64>,
    /// Persistent analytic state at this `D`, in f64s. Counted by the
    /// crate-private `state_floats` helper rather than by running the model.
    pub state_floats: u64,
}

impl AblationRow {
    fn refused(mode: Mode, d: usize, reason: String, state_floats: u64) -> Self {
        AblationRow {
            mode: mode_name(mode).to_string(),
            projected_dim: d,
            refused: Some(reason),
            accuracy: None,
            ece: None,
            a_avg: None,
            max_forgetting: None,
            drift: None,
            floored_eigenvalues: None,
            state_floats,
        }
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Anacp => "anacp",
        Mode::Ananc => "ananc",
    }
}

/// The method's own precondition, checked *before* building the engine.
///
/// `targets::etf_procrustes` asserts this; the ablation cannot let that assert
/// fire (it would abort the whole sweep), so it refuses the cell first and says
/// why. Coupling between the two is a test, not a comment.
fn precondition_refusal(mode: Mode, d: usize, class_count: usize) -> Option<String> {
    match mode {
        Mode::Ananc if d < class_count => Some(format!(
            "ETF precondition: projected_dim {d} < {class_count} classes; \
             targets::etf_procrustes refuses, because Procrustes cannot span \
             the ETF with fewer features than classes (loud precondition)"
        )),
        _ => None,
    }
}

/// Persistent analytic state at `(dim, D)`, in f64s, at `heads = 1`.
///
/// Every term is a real allocation in [`DecideCore`]:
///
/// | term | where |
/// |---|---|
/// | `C·d` | `StreamingStats` per-class input-space sums |
/// | `d·d` | `StreamingStats` shared `Σ x xᵀ` |
/// | `d·D` | `RpLayer`'s frozen projection |
/// | `D·D` | `RidgeHead`'s Gram `ZᵀZ` — the term that dominates as `D` grows |
/// | `C·D` | `RidgeHead`'s per-class RP sums |
/// | `C·D` | the mapped prototypes the NCM readout classifies against |
///
/// Label tables and their `String`s are excluded: they do not depend on `D`,
/// which is the axis this ablation trades along.
fn state_floats(dim: usize, d: usize, class_count: usize) -> u64 {
    let input_stats = (class_count * dim + dim * dim) as u64;
    let per_head = (dim * d + d * d + class_count * d) as u64;
    let readout = (class_count * d) as u64;
    input_stats + per_head + readout
}

/// Round to 6 decimals, the precision this corpus can support.
///
/// The fixture is byte-compared in CI; rounding keeps last-bit FP noise from
/// failing that gate while pinning every digit a corpus of tens of held-out
/// samples can actually distinguish.
fn r6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Run the sweep: one row per (mode, D), in mode-major order.
pub fn d_ablation(
    per_class: usize,
    dim: usize,
    dims: &[usize],
) -> Result<Vec<AblationRow>, String> {
    let labels = NativeLabel::all();
    let corpus = native_corpus(per_class, dim);
    let held: Vec<&Labelled> = corpus.iter().filter(|s| s.held_out).collect();
    if held.is_empty() {
        return Err("corpus has no held-out samples".to_string());
    }

    let mut rows = Vec::new();
    for mode in [Mode::Anacp, Mode::Ananc] {
        for &d in dims {
            rows.push(measure_cell(
                mode, d, per_class, dim, &labels, &corpus, &held,
            )?);
        }
    }
    Ok(rows)
}

fn measure_cell(
    mode: Mode,
    d: usize,
    per_class: usize,
    dim: usize,
    labels: &[NativeLabel],
    corpus: &[Labelled],
    held: &[&Labelled],
) -> Result<AblationRow, String> {
    let class_count = labels.len();
    let cost = state_floats(dim, d, class_count);
    if let Some(reason) = precondition_refusal(mode, d, class_count) {
        return Ok(AblationRow::refused(mode, d, reason, cost));
    }

    let mut core = DecideCore::new(DecideConfig {
        embedding_version: "native-corpus-v1".to_string(),
        embedding_dim: dim,
        projected_dim: d,
        heads: 1,
        mode,
        lambda: 100.0,
        alpha: 1.0,
    })
    .map_err(|e| e.to_string())?;
    for l in labels {
        core.register(l.name()).map_err(|e| e.to_string())?;
    }
    for s in corpus.iter().filter(|s| !s.held_out) {
        core.observe(s.label.name(), &s.embedding)
            .map_err(|e| e.to_string())?;
    }
    // Clear the whitening instrument first: `etf_procrustes` does no
    // whitening, so without this an `ananc` cell would report whatever the
    // previous `anacp` cell last wrote — a stale instrument reading as if it
    // were live. Cleared, "floored nothing" and "did no whitening" are both
    // 0, which is the truth for `ananc`: its geometry is not whitened at all.
    crate::targets::half_power::clear_floored();
    core.fit().map_err(|e| e.to_string())?;

    // Read the whitening instrument *immediately* after the fit: it reports the
    // most recent half-power call, and `accuracy_matrix_at` below fits again
    // and would overwrite it.
    let floored = crate::targets::half_power::floored_eigenvalues()
        .iter()
        .filter(|&&e| e <= 1e-11)
        .count() as u64;

    let scored = report::score_held_out(&core, held, labels)?;
    let accuracy = scored.iter().filter(|(p, t, _)| p == t).count() as f64 / scored.len() as f64;

    // The best of the three mechanisms, on the report's own grid: `ece` here is
    // directly comparable to the pinned calibration report's per-candidate
    // ECE at D = 256, which is exactly what the cross-check test asserts.
    let ece = [
        Mechanism::Temperature,
        Mechanism::Platt,
        Mechanism::KPlusOne,
    ]
    .into_iter()
    .map(|m| report::fit_mechanism(m, &scored).fit_ece)
    .filter(|e| e.is_finite())
    .fold(f64::INFINITY, f64::min);

    let matrix = report::accuracy_matrix_at(labels, per_class, dim, d, mode)?;
    let (_, a_avg) = matrix.last_and_average();
    let max_forgetting = matrix.max_forgetting();

    // R5's drift, measured at this `D` through the report's own monitor.
    //
    // The monitor builds `ananc` engines (it is the only mode the report arms
    // a drift monitor for), so it inherits the ETF precondition: below
    // `D ≥ C` the measurement is *undefined by that precondition*, not merely
    // unmeasured. `anacp` is still fully measured at `D < C` — it simply has
    // no drift column there, and the fixture records `null`.
    let drift = if d < class_count {
        None
    } else {
        let (_, seen) = report::class_arrival_drift_at(dim, per_class, d)?;
        Some(r6(seen))
    };

    Ok(AblationRow {
        mode: mode_name(mode).to_string(),
        projected_dim: d,
        refused: None,
        accuracy: Some(r6(accuracy)),
        ece: Some(r6(ece)),
        a_avg: Some(r6(a_avg)),
        max_forgetting: Some(r6(max_forgetting)),
        drift,
        floored_eigenvalues: Some(floored),
        state_floats: cost,
    })
}

/// The fixture's JSON envelope, shared by the binary and the golden test so the
/// committed bytes are always produced by exactly one code path.
pub fn d_ablation_json(
    per_class: usize,
    dim: usize,
    dims: &[usize],
) -> Result<serde_json::Value, String> {
    let rows = d_ablation(per_class, dim, dims)?;
    Ok(serde_json::json!({
        "engine_version": crate::ENGINE_VERSION,
        "corpus": { "per_class": per_class, "dim": dim, "held_out_every": 4 },
        "dims": dims,
        "rows": rows,
        "caveats": [
            "SYNTHETIC CORPUS. The embeddings come from a seeded PRNG with class anchors (`properties::native_corpus`). The sweep is a RELATIVE comparison across D on this corpus and a mechanism check; it is not an accuracy claim on real project data.",
            "`ece` is the BEST of the three D6 mechanisms on the report's own grid search, measured in-sample on the held-out split exactly as the pinned calibration report measures it. It is comparable to that report, not to a fresh generalization estimate.",
            "`drift` is R5's measured class-arrival drift through the report's own DriftMonitor; the tolerance ε is 0.05. The monitor builds `ananc` engines (the only mode the report arms it for), so `drift` is a function of D ALONE and repeats identically across the two modes at the same D — that repetition is the harness being honest about what it measured, not a copied cell.",
            "`max_forgetting` is the task-incremental accuracy matrix's largest per-column decay (D-C6), measured per row's own mode. It is a different measurement on a different object from `drift`.",
            "`floored_eigenvalues` counts RP-covariance eigenvalues the whitening floored at 1e-12 during the fit — directions whitening destroyed. n_train=90 caps covariance rank at 89, so every D >= 128 is rank-deficient by construction and floors roughly D-89 directions. It is 0 for every ananc row by construction: the ETF/Procrustes target does no whitening at all. That is the mechanism behind the two modes bending differently at large D.",
            "`state_floats` counts persistent analytic state at heads=1: C*d input means + d*d shared outer product + (d*D RP + D*D Gram + C*D per-class RP sums) + C*D readout prototypes. Label strings excluded.",
            "A `refused` row is the method's own precondition, not a crash and not a zero: `targets::etf_procrustes` asserts D >= C, and this harness records the refusal instead of letting the assert abort the sweep.",
            "Rounded to 6 decimals so last-bit floating-point noise cannot fail the CI byte gate; nothing here is measured more finely than tens of held-out samples support.",
        ],
    }))
}

/// The exact text the fixture should contain, trailing newline included.
pub fn fixture_text(per_class: usize, dim: usize, dims: &[usize]) -> Result<String, String> {
    let value = d_ablation_json(per_class, dim, dims)?;
    let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    Ok(format!("{text}\n"))
}

/// Where the fixture lives on disk.
pub fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{PINNED_PROJECTED_DIM, fit_all_candidates};

    /// Golden gate, in-crate so it runs with the suite's ignored tests and not
    /// only in CI's byte-comparison step. `UPDATE_GOLDEN=1` regenerates it —
    /// the same convention as the release manifest gate (S23/S24).
    ///
    /// Ignored because the sweep is minutes in debug (the whitening
    /// eigendecomposition is `O(D³)`, and `D = 512` dominates) and ~30 s in
    /// release. `scripts/run_heavy_tests.sh` runs it in release; so does CI.
    #[test]
    #[ignore = "heavy: full D sweep, minutes in debug / ~30 s release; see run_heavy_tests.sh"]
    fn the_committed_d_ablation_fixture_is_fresh() {
        let fresh =
            fixture_text(DEFAULT_PER_CLASS, DEFAULT_DIM, &DEFAULT_DIMS).expect("the sweep runs");
        let path = fixture_path();
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).expect("fixture dir");
            }
            std::fs::write(&path, &fresh).expect("write fixture");
            return;
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "committed fixture {} is unreadable ({e}); regenerate with \
                 UPDATE_GOLDEN=1 cargo test -p decide_core ablation",
                path.display()
            )
        });
        assert_eq!(
            committed, fresh,
            "the committed D-ablation fixture drifted from a fresh run. If the \
             change is intended, commit the regenerated fixture \
             (UPDATE_GOLDEN=1, or `cargo run -p decide_core --bin decide_ablation -- --write`) \
             and state in the commit message which measured cell changed and why."
        );
    }

    /// The cross-check that makes the ablation comparable to the pinned report:
    /// at the report's own `D`, the two harnesses must agree to the last
    /// rounded digit. If this fails, the ablation is measuring something the
    /// report is not — and every comparative sentence in the spec built on it
    /// would be unsupported.
    ///
    /// Ignored for the same reason as the golden gate above: it runs the sweep
    /// at `D = 256` *and* the full report (≈6 analytic fits at 256²), which is
    /// ~35 s in debug. Both halves are gated separately in CI — the report by
    /// its own freshness step, the sweep by the fixture gate — so this is the
    /// consistency check across those two gates, run in release by
    /// `scripts/run_heavy_tests.sh`.
    #[test]
    #[ignore = "heavy: D=256 sweep + full report, ~35 s debug / seconds in release"]
    fn the_pinned_d_rows_reproduce_the_pinned_report() {
        let rows = d_ablation(DEFAULT_PER_CLASS, DEFAULT_DIM, &[PINNED_PROJECTED_DIM])
            .expect("sweep runs");
        let outcomes = fit_all_candidates(DEFAULT_PER_CLASS, DEFAULT_DIM).expect("report runs");
        for row in &rows {
            let peers: Vec<_> = outcomes.iter().filter(|o| o.mode == row.mode).collect();
            assert!(!peers.is_empty(), "no report rows for mode {}", row.mode);
            let accuracy = row.accuracy.expect("measured");
            let report_accuracy = peers[0].accuracy;
            assert!(
                (accuracy - report_accuracy).abs() < 1e-6,
                "accuracy disagrees at D={PINNED_PROJECTED_DIM} for {}: \
                 {accuracy} vs {report_accuracy}",
                row.mode
            );
            let ece = row.ece.expect("measured");
            let report_best_ece = peers.iter().map(|o| o.ece).fold(f64::INFINITY, f64::min);
            assert!(
                (ece - report_best_ece).abs() < 1e-6,
                "best-ECE disagrees at D={PINNED_PROJECTED_DIM} for {}: \
                 {ece} vs {report_best_ece}",
                row.mode
            );
        }
    }

    /// The sub-class cell: `ananc` refused by its own precondition, `anacp`
    /// measured. Both halves matter — a sweep that skipped the cell entirely
    /// would not demonstrate the precondition, and one that crashed on it could
    /// not report the `anacp` half.
    #[test]
    fn the_sub_class_cell_refuses_ananc_and_still_measures_anacp() {
        let class_count = NativeLabel::all().len();
        let d = class_count - 1;
        let rows = d_ablation(DEFAULT_PER_CLASS, DEFAULT_DIM, &[d]).expect("sweep runs");
        let ananc = rows.iter().find(|r| r.mode == "ananc").unwrap();
        let anacp = rows.iter().find(|r| r.mode == "anacp").unwrap();

        let reason = ananc.refused.clone().expect("ananc must refuse D < C");
        assert!(
            reason.contains(&format!("{d} < {class_count}")),
            "the refusal should name the violated bound: {reason}"
        );
        assert_eq!(
            ananc.accuracy, None,
            "a refused cell must not report a number"
        );
        assert_eq!(ananc.ece, None);
        assert_eq!(ananc.a_avg, None);
        assert_eq!(ananc.max_forgetting, None);
        assert_eq!(ananc.drift, None);

        assert!(anacp.refused.is_none(), "anacp has no ETF precondition");
        assert!(anacp.accuracy.is_some(), "anacp must be measured at D < C");
        // …but its drift column is absent for a different reason: the monitor
        // builds `ananc` engines internally, so below D ≥ C the measurement is
        // undefined by the precondition rather than refused or zero.
        assert_eq!(
            anacp.drift, None,
            "drift is undefined below D ≥ C, since the monitor is ananc-shaped"
        );
    }

    /// Couples the harness's textual precondition to the engine's own assert:
    /// the sweep says `ananc` refuses `D < C` — so the engine had better refuse
    /// it too, loudly. Without this, the harness's check and the engine's
    /// precondition could drift and the sweep would be documenting a
    /// precondition that no longer exists.
    #[test]
    #[should_panic(expected = "at least as many features as classes")]
    fn the_engine_really_does_refuse_a_sub_class_etf() {
        let labels = NativeLabel::all();
        let dim = DEFAULT_DIM;
        let d = labels.len() - 1;
        let corpus = native_corpus(DEFAULT_PER_CLASS, dim);
        let mut core = DecideCore::new(DecideConfig {
            embedding_version: "native-corpus-v1".to_string(),
            embedding_dim: dim,
            projected_dim: d,
            heads: 1,
            mode: Mode::Ananc,
            lambda: 100.0,
            alpha: 1.0,
        })
        .unwrap();
        for l in labels {
            core.register(l.name()).unwrap();
        }
        for s in corpus.iter().filter(|s| !s.held_out) {
            core.observe(s.label.name(), &s.embedding).unwrap();
        }
        // `fit` asserts inside `targets::etf_procrustes` — the loud precondition.
        core.fit().unwrap();
    }

    /// The whitening instrument must be read live, not inherited.
    ///    /// `etf_procrustes` does no whitening, so an `ananc` cell can only report
    /// zero floors if the harness cleared the thread-local before fitting. On
    /// its first run the harness did not, and every `ananc` row reported the
    /// previous `anacp` row's 423 — a stale instrument posing as a
    /// measurement. This pins the fix: `ananc` floors nothing, at every `D`.
    ///
    /// Small `D` on purpose: the property is "ananc never whitens", which any
    /// cell demonstrates, and the large cells are the minutes-long ones.
    #[test]
    fn ananc_reports_zero_whitening_floors_at_every_d() {
        let rows = d_ablation(DEFAULT_PER_CLASS, DEFAULT_DIM, &[16, 64]).expect("sweep runs");
        for row in rows.iter().filter(|r| r.mode == "ananc") {
            assert_eq!(
                row.floored_eigenvalues,
                Some(0),
                "ananc does no whitening, so D={} must floor nothing — a non-zero \
                 value means the instrument was read stale",
                row.projected_dim
            );
        }
    }

    /// The cost axis is real: state grows strictly with `D`, and the Gram term
    /// (`D × D`) dominates at the default — so doubling `D` more than doubles
    /// the state. A cost model that did not grow could not justify a smaller
    /// default, which is the whole point of the sweep.
    ///
    /// Asserted against [`state_floats`] directly rather than by running the
    /// sweep: this is a claim about the cost *model*, and it should not cost
    /// minutes to check. (The sweep rows carry the same numbers, produced by
    /// this function.)
    #[test]
    fn state_cost_grows_strictly_with_d() {
        let class_count = NativeLabel::all().len();
        let dim = DEFAULT_DIM;
        let costs: Vec<u64> = DEFAULT_DIMS
            .iter()
            .map(|&d| state_floats(dim, d, class_count))
            .collect();
        for (pair, d) in costs.windows(2).zip(DEFAULT_DIMS) {
            assert!(
                pair[1] > pair[0],
                "state must grow with D: {} at D={} not > {} at D={}",
                pair[1],
                d,
                pair[0],
                d
            );
        }
        // The Gram quadruples when D doubles; everything else at most doubles.
        // So at the top of the grid the total must more than triple — a number
        // that only holds once D² rules, and would fail if the model quietly
        // stopped counting the Gram.
        let (prev, last) = (costs[costs.len() - 2], *costs.last().unwrap());
        assert!(
            last > 3 * prev,
            "doubling D from {} to {} should more than triple total state \
             (Gram-dominated): {last} vs 3×{prev}",
            DEFAULT_DIMS[DEFAULT_DIMS.len() - 2],
            DEFAULT_DIMS.last().unwrap()
        );
    }
}
