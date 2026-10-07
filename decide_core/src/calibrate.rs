//! The calibration bridge.
//!
//! # This is the research component, and it is the only part of the engine that
//! # no source provides
//!
//! Neither method outputs `P(Y | X)` or `P(ID | X)`. AnaNC emits a *score* and
//! evaluates it threshold-free. So every number this module produces is **ours**,
//! with its own burden of proof, and until the R1–R5 contract below passes on
//! pinned data the engine must not call its output a probability.
//!
//! Audit defect **A6** is precisely the failure this module exists to prevent:
//! `softmax(cos/τ)` "strictly sums to 1.0" being presented as a calibrated
//! probability. Summing to 1 is normalization. Calibration is a separate property,
//! measured by ECE, and it is the gate on the word "probability".
//!
//! # The R1–R5 contract
//!
//! | req | property | checked by |
//! |---|---|---|
//! | R1 | `probabilities` sums to 1 ± 1e-9 | [`Calibrator::probabilities`] + test |
//! | R2 | ECE ≤ 0.05 on held-out data | [`expected_calibration_error`] + `report.rs`'s `best_ece <= DEFAULT_MAX_ECE` |
//! | R3 | `P(ID)` is non-decreasing in the ID score | [`Calibrator::p_id`] (monotone by construction) |
//! | R4 | `P(ID) + P(OOD) = 1`, one estimator not two | [`Calibrator::p_ood`] |
//! | R5 | class arrival moves probabilities by ≤ ε, else declare `recalibrated` | [`DriftMonitor`] |
//!
//! R3 is the reason [`Calibrator::p_id`] is monotone by construction rather than
//! by fitting: a fitted bridge that inverted the ordering would turn "I have
//! never seen this class" into "I am certain of a class", which is the most
//! damaging possible failure for an open-world system.

use serde::{Deserialize, Serialize};

use crate::heads::NcmHead;

/// Which mechanism produced a calibrator. Chosen on metrics (the plan's D6
/// gate), not by taste — this enum exists so the choice is *recorded* in the
/// pinned report rather than being invisible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mechanism {
    /// Identity: report the raw cosine as a score, no probability claim.
    None,
    /// Temperature scaling on the cosine logits. Preserves argmax by
    /// construction, which is why it is the safe default.
    Temperature,
    /// Platt (1-D logistic) on the ID score. Needed when the cosine
    /// distribution is badly miscalibrated but still correctly *ordered*.
    Platt,
    /// A (K+1)-logit model with a separately calibrated rejection bias.
    KPlusOne,
}

/// A fitted calibrator, versioned so it can be pinned like any other fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibrator {
    pub mechanism: Mechanism,
    /// Temperature for [`Mechanism::Temperature`].
    pub temperature: f64,
    /// Logistic parameters on the ID score for [`Mechanism::Platt`].
    pub platt_a: f64,
    pub platt_b: f64,
    /// The rejection bias for [`Mechanism::KPlusOne`].
    pub ood_bias: f64,
    /// The measured ECE at fit time. Carried with the calibrator so a consumer
    /// can see the evidence without loading the whole report.
    pub fit_ece: f64,
    /// Bumped whenever the fitting procedure changes, so a stale calibrator is
    /// detectable rather than silently reusable.
    pub version: u32,
}

impl Default for Calibrator {
    fn default() -> Self {
        Calibrator {
            mechanism: Mechanism::None,
            // 1.0 is the identity scaling, and is also what `Mechanism::None`
            // means: no probability claim, just a temperature that changes
            // nothing.
            temperature: 1.0,
            platt_a: 1.0,
            platt_b: 0.0,
            ood_bias: 0.0,
            fit_ece: f64::NAN,
            version: 1,
        }
    }
}

/// Why a calibrator is not admissible.
///
/// A named type rather than a `String` so a caller cannot accidentally treat a
/// rejection as a success by matching the wrong variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection(pub String);

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Calibrator {
    /// The identity calibrator: emits scores, claims nothing.
    pub fn uncalibrated() -> Self {
        Calibrator::default()
    }

    /// Check the calibrator against the structural requirements (R3, R4).
    ///
    /// R3 has to be a *validated invariant*, not a happy path: the Platt
    /// parameterization is monotone only when its slope is positive, and a
    /// fitter handed a negative slope would otherwise silently invert the ID
    /// ordering. That is the single most damaging failure available to an
    /// open-world system — "I have never seen this class" would turn into "I am
    /// certain of a class" — so it is refused at construction rather than
    /// discovered downstream.
    ///
    /// This is a property test's finding turned into a check:
    /// `properties::property_tests::p_id_never_inverts_the_id_ordering` found
    /// `platt_a = -2.33` inverting, and this method is the fix.
    pub fn validate(&self) -> Result<(), Rejection> {
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(Rejection(format!(
                "temperature must be finite and positive, got {}",
                self.temperature
            )));
        }
        if !self.platt_a.is_finite() || !self.platt_b.is_finite() {
            return Err(Rejection("Platt parameters must be finite".into()));
        }
        if self.platt_a <= 0.0 {
            return Err(Rejection(format!(
                "R3 violation: Platt slope must be positive, got {}; \
                 a negative slope inverts the ID ordering",
                self.platt_a
            )));
        }
        if !self.ood_bias.is_finite() {
            return Err(Rejection("OOD bias must be finite".into()));
        }
        if self.version == 0 {
            // `version` 0 would make an unversioned calibrator indistinguishable
            // from a current one, so a stale artifact could be silently reused.
            return Err(Rejection("calibrator version must be non-zero".into()));
        }
        Ok(())
    }

    /// Whether this calibrator is allowed to use the word "probability".
    ///
    /// Three conditions, all required: a real mechanism, a measured ECE at or
    /// below the project's bar, **and** structural validity. `fit_ece` starts as
    /// NaN so an unfitted calibrator fails this check rather than passing it —
    /// `NaN <= 0.05` is false, which is the safe direction by construction.
    pub fn is_calibrated(&self, max_ece: f64) -> bool {
        self.mechanism != Mechanism::None
            && self.fit_ece.is_finite()
            && self.fit_ece <= max_ece
            && self.validate().is_ok()
    }

    /// `P(ID | x)` from the ID score.
    ///
    /// Monotone non-decreasing in `id_score` for every mechanism, by
    /// construction (R3). The tempering is `σ((id − θ)/s)` — increasing in `id`
    /// because the slope `1/s` is positive.
    pub fn p_id(&self, id_score: f64) -> f64 {
        match self.mechanism {
            // No probability claim: report the raw ordering, but as a *score*.
            Mechanism::None => id_score.clamp(0.0, 1.0),
            Mechanism::Temperature | Mechanism::KPlusOne => {
                let z = (id_score + self.ood_bias) / self.temperature.max(1e-6);
                1.0 / (1.0 + (-z).exp())
            }
            Mechanism::Platt => {
                let z = self.platt_a * id_score + self.platt_b;
                1.0 / (1.0 + (-z).exp())
            }
        }
    }

    /// `P(OOD | x) = 1 − P(ID | x)`. R4: one estimator, two views.
    pub fn p_ood(&self, id_score: f64) -> f64 {
        1.0 - self.p_id(id_score)
    }

    /// `P(Y | x)` over the `K` known classes, summing to 1 (R1).
    ///
    /// Temperature scaling on the cosine logits, which **preserves the argmax**:
    /// dividing every logit by the same positive constant cannot reorder them.
    /// That is the property that makes this safe for a `Choice` answer — the
    /// ranking is the method's output and the probabilities are only a
    /// re-expression of it.
    ///
    /// In [`Mechanism::None`] the numbers are still normalized (R1 is about
    /// normalization and holds regardless) but they are **not** calibrated, and
    /// the answer carries `calibrated: false` so nothing downstream may call them
    /// probabilities.
    pub fn probabilities(&self, head: &NcmHead, x: &nalgebra::DVector<f64>) -> (Vec<f64>, bool) {
        self.probabilities_from_similarities(head.similarities(x).as_slice())
    }

    /// The same R1 normalization over similarities the caller already has.
    ///
    /// Split out so `DecideCore::answer_*` — which computes per-class
    /// similarities anyway — can attach `probabilities` to a calibrated
    /// answer without rebuilding a head just to re-derive them.
    pub fn probabilities_from_similarities(&self, sims: &[f64]) -> (Vec<f64>, bool) {
        let k = sims.len();
        if k == 0 {
            return (Vec::new(), false);
        }
        let mut logits = Vec::with_capacity(k);
        for sim in sims {
            // A cosine can be negative; a raw negative logit would make exp()
            // underflow to 0 for most classes. Shifting by the max is the
            // standard stabiliser and cannot change the argmax.
            logits.push(sim / self.temperature.max(1e-6));
        }
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exps: Vec<f64> = logits.iter().map(|z| (z - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            // Degenerate: fall back to uniform rather than emit NaN. Uniform is
            // the honest answer when nothing discriminates.
            return (vec![1.0 / k as f64; k], false);
        }
        for e in exps.iter_mut() {
            *e /= sum;
        }
        (exps, self.is_calibrated(DEFAULT_MAX_ECE))
    }
}

/// The project's ECE bar. Pinned as a constant so it cannot drift per call site.
pub const DEFAULT_MAX_ECE: f64 = 0.05;

/// Expected calibration error over a confidence-binned dataset.
///
/// `bins` is `(predicted_probability, was_it_correct)` pairs. The standard
/// top-label ECE: within each equal-width bin, the mean confidence minus the
/// empirical accuracy, weighted by bin population.
pub fn expected_calibration_error(bins: &[(f64, bool)]) -> f64 {
    if bins.is_empty() {
        return f64::NAN;
    }
    let n_bins = 10usize;
    let mut acc = vec![0.0f64; n_bins];
    let mut conf = vec![0.0f64; n_bins];
    let mut count = vec![0usize; n_bins];
    for (p, correct) in bins {
        let idx = ((p.clamp(0.0, 1.0) * n_bins as f64) as usize).min(n_bins - 1);
        acc[idx] += if *correct { 1.0 } else { 0.0 };
        conf[idx] += p;
        count[idx] += 1;
    }
    let n = bins.len() as f64;
    let mut ece = 0.0;
    for b in 0..n_bins {
        if count[b] == 0 {
            continue;
        }
        let m = count[b] as f64;
        ece += (m / n) * (conf[b] / m - acc[b] / m).abs();
    }
    ece
}

/// Area under the ROC curve from `(score, is_positive)` pairs.
///
/// The source methods report threshold-free AUC and FPR95 precisely because
/// threshold selection is application-dependent; this project keeps that
/// discipline and reports AUC/FPR95 rather than picking an operating point.
pub fn roc_auc(scored: &[(f64, bool)]) -> f64 {
    let pos = scored.iter().filter(|(_, p)| *p).count();
    let neg = scored.len() - pos;
    if pos == 0 || neg == 0 {
        return f64::NAN;
    }
    let mut sorted: Vec<(f64, bool)> = scored.to_vec();
    // ASCENDING by score. This direction is load-bearing: the Mann-Whitney
    // identity `U = R - n(n+1)/2` counts the pairs a group *wins* when the
    // winning group holds the HIGHER ranks. Sorting descending would make every
    // positive hold the lowest ranks and return `U = 0` for a perfect
    // separation — the exact opposite of the intended answer.
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    let mut rank = 1.0f64;
    while i < sorted.len() {
        let mut j = i;
        while j < sorted.len() && sorted[j].0 == sorted[i].0 {
            j += 1;
        }
        let avg_rank = (rank + (rank + (j - i) as f64 - 1.0)) / 2.0;
        // The tie block is scored as a whole; there is no per-element work that
        // an iterator would express more clearly.
        for entry in &sorted[i..j] {
            if entry.1 {
                rank_sum += avg_rank;
            }
        }
        rank += (j - i) as f64;
        i = j;
    }
    let u = rank_sum - (pos * (pos + 1)) as f64 / 2.0;
    u / (pos * neg) as f64
}

/// False positive rate at 95% true positive rate.
pub fn fpr_at_95_tpr(scored: &[(f64, bool)]) -> f64 {
    let mut sorted = scored.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let total_pos = sorted.iter().filter(|(_, p)| *p).count();
    let total_neg = sorted.len() - total_pos;
    if total_pos == 0 || total_neg == 0 {
        return f64::NAN;
    }
    let target = (total_pos as f64 * 0.95).ceil() as usize;
    let mut tp = 0usize;
    let mut fp = 0usize;
    for (score, pos) in sorted {
        if tp >= target {
            break;
        }
        if pos {
            tp += 1;
        } else {
            fp += 1;
        }
        let _ = score;
    }
    fp as f64 / total_neg as f64
}

/// D12's drift monitor: watches for the event that invalidates a calibrator.
///
/// The calibrator's input distribution changes when classes arrive (R5) or when
/// the embedding model changes (D7). Both move `P(ID)` without anything being
/// *wrong*, so the honest response is to declare it, not to hide it — the same
/// discipline as a certificate band that widens rather than a number that lies.
#[derive(Debug, Clone)]
pub struct DriftMonitor {
    /// Reference per-class probabilities, captured at calibration time.
    reference: Vec<f64>,
    /// The bound ε within which drift is considered noise.
    epsilon: f64,
    /// Largest drift seen since the last reset.
    observed: f64,
    recalibrated: bool,
}

impl DriftMonitor {
    /// A monitor with the given drift bound.
    pub fn new(reference: Vec<f64>, epsilon: f64) -> Self {
        DriftMonitor {
            observed: 0.0,
            recalibrated: false,
            reference,
            epsilon,
        }
    }

    /// Observe a fresh probability vector; returns the drift seen.
    ///
    /// A class-count change is compared on the shared prefix and recorded as a
    /// *structural* change, because after a new class arrives there is no common
    /// simplex to compare in — that is exactly the R5 caveat ("zero forgetting
    /// for the calibrator is impossible; the deliverable is *declared* drift").
    pub fn observe(&mut self, probs: &[f64]) -> f64 {
        let n = self.reference.len().min(probs.len());
        let mut worst = 0.0f64;
        // Zip over the shared prefix: comparing a reference and an observation of
        // different lengths is the caller's problem, and this reports the
        // structural change separately below.
        for (r, p) in self.reference.iter().zip(probs.iter()).take(n) {
            let d = (r - p).abs();
            if d > worst {
                worst = d;
            }
        }
        if probs.len() != self.reference.len() {
            // Structural change: a new class changed the simplex. Declared, not
            // measured, because there is no like-for-like comparison.
            worst = worst.max(self.epsilon);
        }
        self.observed = worst;
        if worst > self.epsilon {
            self.recalibrated = true;
        }
        worst
    }

    /// Whether the calibrator must be considered stale.
    pub fn needs_recalibration(&self) -> bool {
        self.recalibrated
    }

    /// Largest drift observed since the last reset.
    pub fn observed_drift(&self) -> f64 {
        self.observed
    }

    /// Re-baseline after an actual recalibration.
    pub fn reset(&mut self, reference: Vec<f64>) {
        self.reference = reference;
        self.recalibrated = false;
        self.observed = 0.0;
    }
}

/// The pinned calibration report (D6).
///
/// A *report*, not a claim: `meets_contract` is a single boolean computed from
/// the measured numbers, and `failed_requirements` names what did not pass. A
/// report that failed is a legitimate artifact — it is what stops the engine from
/// claiming probabilities it has not earned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub mechanism: Mechanism,
    pub calibrator_version: u32,
    pub engine_version: String,
    /// Number of held-out samples the report was computed over.
    pub n_samples: usize,
    pub ece: f64,
    pub auc: f64,
    pub fpr95: f64,
    /// Per-requirement verdicts, R1..R5.
    pub r1_normalization: bool,
    pub r2_calibration: bool,
    pub r3_monotonicity: bool,
    pub r4_open_world_coherence: bool,
    pub r5_class_arrival_stability: bool,
    pub max_drift_epsilon: f64,
    pub observed_max_drift: f64,
}

impl CalibrationReport {
    /// Whether every requirement passed.
    pub fn meets_contract(&self) -> bool {
        self.r1_normalization
            && self.r2_calibration
            && self.r3_monotonicity
            && self.r4_open_world_coherence
            && self.r5_class_arrival_stability
    }

    /// Which requirements did not pass, by name.
    pub fn failed_requirements(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.r1_normalization {
            out.push("R1 normalization");
        }
        if !self.r2_calibration {
            out.push("R2 calibration");
        }
        if !self.r3_monotonicity {
            out.push("R3 monotonicity");
        }
        if !self.r4_open_world_coherence {
            out.push("R4 open-world coherence");
        }
        if !self.r5_class_arrival_stability {
            out.push("R5 class-arrival stability");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn head() -> NcmHead {
        NcmHead::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            vec!["a".into(), "b".into()],
        )
    }

    #[test]
    fn the_uncalibrated_calibrator_may_not_claim_probabilities() {
        let c = Calibrator::uncalibrated();
        assert!(!c.is_calibrated(DEFAULT_MAX_ECE));
        let (_, calibrated) =
            c.probabilities(&head(), &nalgebra::DVector::from_row_slice(&[1.0, 0.0]));
        assert!(
            !calibrated,
            "an unfitted calibrator must not set calibrated: true"
        );
    }

    #[test]
    fn an_unfitted_but_named_calibrator_still_fails_the_gate() {
        // The trap this guards: setting a mechanism by hand and forgetting to
        // measure. `fit_ece` is NaN until fitted, and NaN <= 0.05 is false.
        let c = Calibrator {
            mechanism: Mechanism::Temperature,
            temperature: 2.0,
            fit_ece: f64::NAN,
            ..Calibrator::default()
        };
        assert!(
            !c.is_calibrated(DEFAULT_MAX_ECE),
            "unmeasured ECE must not pass R2"
        );
    }

    /// R1: the contract the sketch claimed and normalization alone does not give.
    #[test]
    fn r1_probabilities_always_sum_to_one() {
        for t in [0.1f64, 1.0, 10.0, 1000.0] {
            let c = Calibrator {
                temperature: t,
                ..Calibrator::default()
            };
            for x in [
                nalgebra::DVector::from_row_slice(&[1.0, 0.0]),
                nalgebra::DVector::from_row_slice(&[-3.0, 0.2]),
                nalgebra::DVector::from_row_slice(&[0.0, 0.0]),
            ] {
                let (p, _) = c.probabilities(&head(), &x);
                let s: f64 = p.iter().sum();
                assert!(
                    (s - 1.0).abs() <= 1e-9,
                    "R1 violated at temperature {t}: sum = {s}"
                );
            }
        }
    }

    /// R3: monotonicity. The failure this prevents is the catastrophic one —
    /// inverting the ordering would make an unseen input look confidently known.
    #[test]
    fn r3_p_id_is_monotone_non_decreasing_in_the_id_score() {
        for mech in [
            Mechanism::None,
            Mechanism::Temperature,
            Mechanism::Platt,
            Mechanism::KPlusOne,
        ] {
            let c = Calibrator {
                mechanism: mech,
                temperature: 1.7,
                platt_a: 3.0,
                platt_b: -1.0,
                ..Calibrator::default()
            };
            let mut prev = f64::NEG_INFINITY;
            for i in 0..=200 {
                let score = -1.0 + 2.0 * (i as f64) / 200.0;
                let p = c.p_id(score);
                assert!(
                    p >= prev - 1e-12,
                    "{mech:?} inverted at score {score}: {p} < {prev}"
                );
                prev = p;
            }
        }
    }

    /// R4: one estimator, two views.
    #[test]
    fn r4_p_id_and_p_ood_are_complementary() {
        let c = Calibrator {
            mechanism: Mechanism::Platt,
            platt_a: 2.0,
            platt_b: 0.5,
            ..Calibrator::default()
        };
        for s in [-1.0f64, 0.0, 0.5, 1.0] {
            let sum = c.p_id(s) + c.p_ood(s);
            assert!((sum - 1.0).abs() < 1e-12, "R4 violated at {s}: {sum}");
        }
    }

    #[test]
    fn a_degenerate_zero_vector_yields_uniform_not_nan() {
        let c = Calibrator::uncalibrated();
        let (p, calibrated) = c.probabilities(&head(), &nalgebra::DVector::zeros(2));
        assert_eq!(p, vec![0.5, 0.5]);
        assert!(!calibrated);
    }

    #[test]
    fn temperature_scaling_preserves_the_argmax() {
        // The safety property for a `Choice` answer: the ranking is the method's
        // output, probabilities only re-express it.
        let c = Calibrator {
            mechanism: Mechanism::Temperature,
            temperature: 0.05,
            ..Calibrator::default()
        };
        let h = NcmHead::new(
            DMatrix::from_row_slice(2, 3, &[0.9, 0.1, 0.2, 0.95, 0.6, 0.1]),
            vec!["a".into(), "b".into()],
        );
        let x = nalgebra::DVector::from_row_slice(&[0.9, 0.1, 0.95]);
        let (p, _) = c.probabilities(&h, &x);
        // Compare against the raw cosine ordering rather than an index.
        let sims = h.similarities(&x);
        let best_p = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let best_sim = sims
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best_p, best_sim, "argmax changed under temperature scaling");
    }

    #[test]
    fn ece_is_zero_for_a_perfectly_calibrated_set() {
        // Every prediction 1.0 and always right ⇒ no calibration error.
        let bins: Vec<(f64, bool)> = (0..50).map(|_| (1.0, true)).collect();
        assert!(expected_calibration_error(&bins).abs() < 1e-12);
    }

    #[test]
    fn ece_is_large_for_a_maximally_overconfident_set() {
        let bins: Vec<(f64, bool)> = (0..50).map(|_| (1.0, false)).collect();
        assert!(expected_calibration_error(&bins) > 0.9);
    }

    #[test]
    fn ece_of_nothing_is_nan_not_zero() {
        assert!(expected_calibration_error(&[]).is_nan());
    }

    #[test]
    fn auc_is_one_for_a_perfect_separation_and_half_for_a_coin_flip() {
        let perfect: Vec<(f64, bool)> = vec![(0.9, true), (0.8, true), (0.2, false), (0.1, false)];
        assert!((roc_auc(&perfect) - 1.0).abs() < 1e-12);

        let tied: Vec<(f64, bool)> = vec![(0.5, true), (0.5, false)];
        assert!(
            (roc_auc(&tied) - 0.5).abs() < 1e-12,
            "a tie is worth half a win, got {}",
            roc_auc(&tied)
        );
    }

    #[test]
    fn drift_within_epsilon_does_not_force_recalibration() {
        let mut m = DriftMonitor::new(vec![0.5, 0.5], 0.1);
        assert!(m.observe(&[0.52, 0.48]) <= 0.1);
        assert!(!m.needs_recalibration());
    }

    #[test]
    fn drift_beyond_epsilon_forces_recalibration_and_is_visible() {
        let mut m = DriftMonitor::new(vec![0.9, 0.1], 0.05);
        m.observe(&[0.5, 0.5]);
        assert!(m.needs_recalibration());
        assert!(m.observed_drift() > 0.05);
    }

    #[test]
    fn a_new_class_is_a_declared_structural_change_not_a_silent_one() {
        // R5's honesty: after a new class the simplex genuinely changed, so there
        // is no like-for-like comparison — and the monitor must say so rather
        // than report a comfortable zero.
        let mut m = DriftMonitor::new(vec![0.5, 0.5], 0.05);
        m.observe(&[0.34, 0.33, 0.33]);
        assert!(m.needs_recalibration());
    }

    #[test]
    fn resetting_rebaselines_the_monitor() {
        let mut m = DriftMonitor::new(vec![0.9, 0.1], 0.05);
        m.observe(&[0.1, 0.9]);
        assert!(m.needs_recalibration());
        m.reset(vec![0.1, 0.9]);
        assert!(!m.needs_recalibration());
        assert_eq!(m.observed_drift(), 0.0);
    }

    #[test]
    fn a_failing_report_names_what_failed() {
        let r = CalibrationReport {
            mechanism: Mechanism::Temperature,
            calibrator_version: 1,
            engine_version: "test".into(),
            n_samples: 0,
            ece: 0.4,
            auc: 0.5,
            fpr95: 0.9,
            r1_normalization: true,
            r2_calibration: false,
            r3_monotonicity: true,
            r4_open_world_coherence: true,
            r5_class_arrival_stability: false,
            max_drift_epsilon: 0.05,
            observed_max_drift: 0.4,
        };
        assert!(!r.meets_contract());
        let failed = r.failed_requirements();
        assert_eq!(failed.len(), 2);
        assert!(failed.contains(&"R2 calibration"));
        assert!(failed.contains(&"R5 class-arrival stability"));
    }
}
