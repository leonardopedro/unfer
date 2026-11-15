//! R9 secondary-endpoint arithmetic: top-label reliability and selective
//! prediction.
//!
//! These are the pure, fixture-testable pieces of `examples/r9_secondary.rs`.
//! They take one `(confidence, correct)` entry per question, so the caller
//! decides which split and arm the numbers cover; nothing here refits, resamples
//! or reads a model.
//!
//! Declared conventions:
//!
//! * top-label confidence is `max p`, and correctness is `argmax p == label`
//!   (ties resolve to the lowest index, matching the model's argmax);
//! * reliability uses **twenty fixed equal-width bins** on `[0, 1]`; empty bins
//!   are reported as empty rather than dropped, and ECE/MCE are computed over
//!   the non-empty bins only;
//! * abstention is `abstain when max p < τ`, so selective accuracy and risk are
//!   undefined (reported as `NaN` / zero coverage) when nothing is answered;
//! * the threshold rule is the grid point in `0.00..=0.90` (step `0.01`) with
//!   the highest selective accuracy, ties resolving to the **lower** τ (the
//!   higher coverage).

/// Fixed calibration bins on top-label confidence.
pub const CALIB_BINS: usize = 20;
/// Grid resolution for the abstention threshold (`0.00 ..= 0.90`, step 0.01).
pub const TAU_GRID: usize = 90;

/// One reliability bin and its accumulated counts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CalibBin {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub conf: f64,
    pub correct: f64,
}

impl CalibBin {
    /// Mean confidence in the bin; `NaN` when the bin is empty.
    pub fn mean_confidence(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.conf / self.n as f64
        }
    }

    /// Empirical accuracy in the bin; `NaN` when the bin is empty.
    pub fn accuracy(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.correct / self.n as f64
        }
    }
}

/// Top-label index and confidence. Ties resolve to the lowest index.
pub fn top_label(probs: &[f64]) -> (usize, f64) {
    probs
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(ti, tv), (j, v)| {
            if *v > tv {
                (j, *v)
            } else {
                (ti, tv)
            }
        })
}

/// Reliability bins, ECE and MCE from one `(confidence, correct)` pair per row.
/// Rows with a non-finite confidence are ignored.
pub fn reliability(conf: &[f64], correct: &[bool]) -> (Vec<CalibBin>, f64, f64) {
    debug_assert_eq!(conf.len(), correct.len());
    let mut bins = vec![CalibBin::default(); CALIB_BINS];
    for (i, b) in bins.iter_mut().enumerate() {
        b.lo = i as f64 / CALIB_BINS as f64;
        b.hi = (i + 1) as f64 / CALIB_BINS as f64;
    }
    let mut used = 0usize;
    for (c, ok) in conf.iter().zip(correct) {
        if !c.is_finite() {
            continue;
        }
        let idx = ((*c * CALIB_BINS as f64) as usize).min(CALIB_BINS - 1);
        let b = &mut bins[idx];
        b.n += 1;
        b.conf += *c;
        b.correct += f64::from(*ok);
        used += 1;
    }
    if used == 0 {
        return (bins, f64::NAN, f64::NAN);
    }
    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    for b in &bins {
        if b.n == 0 {
            continue;
        }
        let gap = (b.accuracy() - b.mean_confidence()).abs();
        ece += b.n as f64 / used as f64 * gap;
        mce = mce.max(gap);
    }
    (bins, ece, mce)
}

/// `(coverage, selective accuracy)` when abstaining below `tau`.
pub fn selective(conf: &[f64], correct: &[bool], tau: f64) -> (f64, f64) {
    debug_assert_eq!(conf.len(), correct.len());
    let finite = conf.iter().filter(|c| c.is_finite()).count();
    if finite == 0 {
        return (0.0, f64::NAN);
    }
    let mut answered = 0usize;
    let mut right = 0usize;
    for (c, ok) in conf.iter().zip(correct) {
        if c.is_finite() && *c >= tau {
            answered += 1;
            if *ok {
                right += 1;
            }
        }
    }
    if answered == 0 {
        return (0.0, f64::NAN);
    }
    (answered as f64 / finite as f64, right as f64 / answered as f64)
}

/// `(coverage, risk)` with `risk = 1 − selective accuracy`; `NaN` risk when
/// nothing is answered.
pub fn risk_coverage(conf: &[f64], correct: &[bool], tau: f64) -> (f64, f64) {
    let (coverage, accuracy) = selective(conf, correct, tau);
    (coverage, if coverage == 0.0 { f64::NAN } else { 1.0 - accuracy })
}

/// Coverage-target rule: the grid point with the **highest coverage** whose
/// calibration risk is at most `target_risk`; ties resolve to the lower `tau`.
/// `None` when no grid point meets the target at positive coverage.
pub fn choose_tau_at_risk(conf: &[f64], correct: &[bool], target_risk: f64) -> Option<f64> {
    let mut best: Option<(f64, f64)> = None; // (coverage, tau)
    for k in 0..=TAU_GRID {
        let tau = k as f64 / 100.0;
        let (coverage, risk) = risk_coverage(conf, correct, tau);
        if coverage > 0.0 && risk.is_finite() && risk <= target_risk + 1e-12 {
            if best.map_or(true, |(c, _)| coverage > c + 1e-12) {
                best = Some((coverage, tau));
            }
        }
    }
    best.map(|(_, tau)| tau)
}

/// The declared threshold rule: highest selective accuracy on the grid, ties
/// to the lower `tau`. `None` when no grid point answers anything.
pub fn choose_tau(conf: &[f64], correct: &[bool]) -> Option<f64> {
    let mut best: Option<(f64, f64)> = None; // (accuracy, tau)
    for k in 0..=TAU_GRID {
        let tau = k as f64 / 100.0;
        let (coverage, accuracy) = selective(conf, correct, tau);
        if coverage == 0.0 || accuracy.is_nan() {
            continue;
        }
        if best.map_or(true, |(a, _)| accuracy > a + 1e-12) {
            best = Some((accuracy, tau));
        }
    }
    best.map(|(_, tau)| tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(rows: &[(f64, bool)]) -> (Vec<f64>, Vec<bool>) {
        (rows.iter().map(|r| r.0).collect(), rows.iter().map(|r| r.1).collect())
    }

    #[test]
    fn top_label_takes_the_largest_and_breaks_ties_low() {
        assert_eq!(top_label(&[0.1, 0.7, 0.2]), (1, 0.7));
        assert_eq!(top_label(&[0.5, 0.5]), (0, 0.5));
        assert_eq!(top_label(&[f64::NAN, 0.2]), (1, 0.2));
    }

    #[test]
    fn single_bin_ece_is_its_confidence_gap() {
        // All in the [0.9, 0.95) bin and all correct: accuracy 1, mean
        // confidence 0.92, so ECE = MCE = 0.08.
        let (conf, correct) = pairs(&[(0.9, true), (0.92, true), (0.94, true)]);
        let (bins, ece, mce) = reliability(&conf, &correct);
        let occupied: Vec<&CalibBin> = bins.iter().filter(|b| b.n > 0).collect();
        assert_eq!(occupied.len(), 1);
        assert_eq!(occupied[0].n, 3);
        assert!((occupied[0].mean_confidence() - 0.92).abs() < 1e-12);
        assert_eq!(occupied[0].accuracy(), 1.0);
        assert!((ece - 0.08).abs() < 1e-12, "ECE {ece}");
        assert!((mce - 0.08).abs() < 1e-12, "MCE {mce}");
        assert!(ece <= mce + 1e-15);
    }

    #[test]
    fn reliability_matches_a_hand_computed_fixture() {
        // Two rows in [0.8,0.85) (one right), two in [0.4,0.45) (both wrong).
        let (conf, correct) = pairs(&[
            (0.81, true),
            (0.83, false),
            (0.41, false),
            (0.43, false),
        ]);
        let (bins, ece, mce) = reliability(&conf, &correct);
        let hi = bins[16].clone(); // 16/20 = 0.80
        let lo = bins[8].clone(); // 8/20 = 0.40
        assert_eq!((hi.n, lo.n), (2, 2));
        assert!((hi.mean_confidence() - 0.82).abs() < 1e-12);
        assert!((hi.accuracy() - 0.5).abs() < 1e-12);
        assert!((lo.mean_confidence() - 0.42).abs() < 1e-12);
        assert_eq!(lo.accuracy(), 0.0);
        // ECE = (2/4)*|0.5 - 0.82| + (2/4)*|0 - 0.42| = 0.16 + 0.21.
        assert!((ece - 0.37).abs() < 1e-12, "ECE {ece}");
        assert!((mce - 0.42).abs() < 1e-12, "MCE {mce}");
    }

    #[test]
    fn empty_bins_are_reported_and_do_not_affect_ece() {
        let (conf, correct) = pairs(&[(0.02, false)]);
        let (bins, ece, _) = reliability(&conf, &correct);
        assert_eq!(bins.iter().filter(|b| b.n > 0).count(), 1);
        assert!(bins[19].n == 0);
        // One bin with accuracy 0 and confidence 0.02 contributes 0.02.
        assert!((ece - 0.02).abs() < 1e-12, "ECE {ece}");
        assert!(bins[19].mean_confidence().is_nan());
    }

    #[test]
    fn selective_and_risk_coverage_are_complements() {
        let (conf, correct) = pairs(&[(0.9, true), (0.8, false), (0.3, true), (0.2, false)]);
        // tau = 0.5 answers the first two: coverage 1/2, accuracy 1/2, risk 1/2.
        assert_eq!(selective(&conf, &correct, 0.5), (0.5, 0.5));
        assert_eq!(risk_coverage(&conf, &correct, 0.5), (0.5, 0.5));
        // tau = 0.85 answers one, correctly.
        assert_eq!(risk_coverage(&conf, &correct, 0.85), (0.25, 0.0));
        // Nothing answered: coverage 0, risk undefined.
        let (coverage, risk) = risk_coverage(&conf, &correct, 0.95);
        assert_eq!(coverage, 0.0);
        assert!(risk.is_nan());
    }

    #[test]
    fn coverage_is_counted_over_all_rows_including_abstentions() {
        let (conf, correct) = pairs(&[(0.9, true), (0.9, true), (0.1, false)]);
        let (coverage, accuracy) = selective(&conf, &correct, 0.5);
        assert!((coverage - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn choose_tau_prefers_accuracy_then_the_lower_threshold() {
        // 0.9 correct, 0.8 wrong, 0.7 correct.
        // Every tau in (0.80, 0.90] answers only the 0.9 row at 100 %, so the
        // tie-break lands on the grid point just above 0.80: 0.81.
        // tau=0.75 -> two answers, 50 %; tau=0.65 -> three answers, 2/3.
        let (conf, correct) = pairs(&[(0.9, true), (0.8, false), (0.7, true)]);
        assert_eq!(choose_tau(&conf, &correct), Some(0.81));
        // Ties resolve downward: both 0.9 and 0.0 answer everything at 50 %,
        // but 0.0 is not a tie at higher accuracy here; construct a real tie.
        let (conf, correct) = pairs(&[(0.6, true), (0.4, true)]);
        // Every tau up to 0.4 gives 100 %, and the lowest grid tau is 0.0.
        assert_eq!(choose_tau(&conf, &correct), Some(0.0));
        // Nothing is ever answered above the largest confidence.
        assert_eq!(choose_tau(&[], &[]), None);
    }

    #[test]
    fn coverage_target_rule_buys_the_most_coverage_at_the_risk_cap() {
        // Full coverage already has risk 1/3, under the 0.5 cap, so the rule
        // should not trade coverage away; ties resolve to the lowest tau.
        let (conf, correct) = pairs(&[(0.9, true), (0.8, false), (0.7, true)]);
        assert_eq!(choose_tau_at_risk(&conf, &correct, 0.5), Some(0.0));
        // A tight cap (0.1) is only met by the single correct 0.9 answer.
        assert_eq!(choose_tau_at_risk(&conf, &correct, 0.1), Some(0.81));
        // An impossible cap has no solution.
        assert_eq!(choose_tau_at_risk(&conf, &correct, -0.1), None);
    }

    #[test]
    fn non_finite_confidences_are_skipped() {
        let (conf, correct) = pairs(&[(0.8, true), (f64::NAN, false)]);
        let (bins, _, _) = reliability(&conf, &correct);
        assert_eq!(bins.iter().map(|b| b.n).sum::<usize>(), 1);
        let (coverage, accuracy) = selective(&conf, &correct, 0.5);
        assert_eq!(coverage, 1.0);
        assert_eq!(accuracy, 1.0);
    }
}
