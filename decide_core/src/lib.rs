//! `decide_core` — an analytic (closed-form, no-backprop) continual decision
//! layer.
//!
//! # What this crate is
//!
//! Frozen embeddings in, typed answers out, with **no gradient step anywhere**.
//! Every method is reimplemented from published equations and cited in
//! [`../docs/DECIDE_SPEC.md`](https://github.com/leonardopedro/unfer/blob/main/docs/DECIDE_SPEC.md);
//! no third-party source is vendored. See the workspace `ATTRIBUTION.md`.
//!
//! # The load-bearing rule: this engine is not trusted
//!
//! This project certifies computation — bands, golden gates, `verify-invariants`,
//! nanoda re-verification. A statistical decision layer has exactly one
//! legitimate role here: **advice, labelled as advice**. So:
//!
//! * Every interaction is `observe`-kind. There is no `Mutate` path in this
//!   crate, and an engine output can never approve, merge, gate, sign or certify.
//! * Until a pinned calibration report passes R1–R5, the answer type carries
//!   `calibrated: false` and the numbers are **scores**. Sum-to-1 is
//!   normalization, not calibration, and the difference is the whole point.
//! * Non-consumers, permanently: signing and consensus, the golden release gate,
//!   Lean/nanoda checks, certificate reading, and the verify-before-merge
//!   evidence gate. An engine failure can at worst lose advice.
//!
//! # The two modes, and why there are two
//!
//! | mode | targets | readout | open-world |
//! |---|---|---|---|
//! | [`Mode::Anacp`] | whitened, SVD-perturbed means (spread apart) | NCM | no |
//! | [`Mode::Ananc`] | simplex ETF by orthogonal Procrustes | cosine NCM | yes, via the ID score |
//!
//! They share the streaming-statistics and ridge skeleton and differ in what they
//! regress onto. A unification must *select* per surface, not average the two:
//! averaging a spread-mean target with an ETF target yields a third thing that is
//! neither and satisfies neither constraint.
//!
//! # Example
//!
//! ```
//! use decide_core::{DecideCore, DecideConfig, Mode};
//! use nalgebra::DVector;
//!
//! let mut core = DecideCore::new(DecideConfig {
//!     embedding_version: "test-v1".into(),
//!     embedding_dim: 3,
//!     projected_dim: 32,
//!     heads: 1,
//!     mode: Mode::Ananc,
//!     lambda: 1.0,
//!     alpha: 1.0,
//! })
//! .unwrap();
//! for l in ["theory-a", "theory-b", "theory-c"] {
//!     core.register(l).unwrap();
//! }
//! core.observe("theory-a", &DVector::from_row_slice(&[1.0, 0.0, 0.0])).unwrap();
//! core.observe("theory-a", &DVector::from_row_slice(&[0.9, 0.1, 0.0])).unwrap();
//! core.observe("theory-b", &DVector::from_row_slice(&[0.0, 1.0, 0.0])).unwrap();
//! core.observe("theory-b", &DVector::from_row_slice(&[0.1, 0.9, 0.0])).unwrap();
//! core.observe("theory-c", &DVector::from_row_slice(&[0.0, 0.0, 1.0])).unwrap();
//! core.observe("theory-c", &DVector::from_row_slice(&[0.0, 0.1, 0.9])).unwrap();
//! core.fit().unwrap();
//!
//! let a = core.answer_choice(
//!     "which theory?",
//!     &[
//!         DVector::from_row_slice(&[1.0, 0.0, 0.0]),
//!         DVector::from_row_slice(&[0.0, 1.0, 0.0]),
//!     ],
//! );
//! let a = a.unwrap();
//! // Uncalibrated by construction: this is a score, and the answer says so.
//! assert!(!a.calibrated);
//! assert!(a.confidence.is_none(), "no confidence without calibration");
//! ```

pub mod backbone;
pub mod calibrate;
pub mod certified;
pub mod consumers;
pub mod heads;
pub mod properties;
pub mod report;
pub mod rp;
pub mod stats;
pub mod targets;

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

pub use calibrate::{
    CalibrationReport, Calibrator, DEFAULT_MAX_ECE, DriftMonitor, Mechanism,
    expected_calibration_error, fpr_at_95_tpr, roc_auc,
};
pub use certified::{Enclosed, RidgeSolution, enclose_ridge};
pub use heads::{NcmHead, RidgeHead, SolveError};
pub use rp::RpLayer;
pub use stats::StreamingStats;
pub use targets::TargetKind;

/// The engine's version string, recorded in every answer so a consumer can tell
/// which build produced a number.
pub const ENGINE_VERSION: &str = concat!("decide_core/", env!("CARGO_PKG_VERSION"));

/// Which method the engine runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Accuracy-first surfaces: routing, triage. Spread-mean targets.
    Anacp,
    /// Open-world surfaces: editor ranking, board triage. ETF targets plus an
    /// out-of-distribution score.
    Ananc,
}

impl Mode {
    /// The target construction this mode uses.
    pub fn target_kind(self) -> TargetKind {
        match self {
            Mode::Anacp => TargetKind::SpreadMeans,
            Mode::Ananc => TargetKind::Etf,
        }
    }

    /// Whether this mode produces an open-world (OOD) score.
    ///
    /// The honest answer is `false` for `anacp`: it has no OOD mechanism, and
    /// emitting a `p_ood` there would be inventing one.
    pub fn has_open_world(self) -> bool {
        matches!(self, Mode::Ananc)
    }
}

/// Why the engine refused to do something.
#[derive(Debug, thiserror::Error)]
pub enum DecideError {
    #[error("class {0:?} is not registered")]
    UnknownClass(String),
    #[error("class {0:?} is already registered; ids are never reused")]
    DuplicateClass(String),
    #[error("engine is not fitted yet; call fit() first")]
    NotFitted,
    #[error("embedding dimension mismatch: state is {expected}, got {got}")]
    EmbeddingDim { expected: usize, got: usize },
    #[error(
        "embedding model changed from {from} to {to}; use the explicit migration path, because input-space statistics are not comparable across models"
    )]
    EmbeddingVersionChanged { from: String, to: String },
    #[error("embedding seam rejected the batch: {0}")]
    Seam(String),
    #[error("project asked for {requested} heads; the supported range is 1..=3")]
    HeadCount { requested: usize },
    #[error("analytic solve failed: {0}")]
    Solve(#[from] SolveError),
}

/// Construction-time configuration.
///
/// Sized against the papers' own numbers: the default projection there is
/// `D = 5000`, which is 25M floats per head. This project's label sets are tens
/// of classes, not thousands, so the default is much smaller — and the plan
/// requires the `D`-ablation to be re-run on project data before any default is
/// blessed. That has not happened yet, which is why these are `tunable and
/// unpinned` rather than "validated".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideConfig {
    /// The embedding model's identity. Part of engine identity: statistics
    /// accumulated under one model are not comparable under another (audit
    /// defect A4's second half).
    pub embedding_version: String,
    /// Embedding dimension `d`.
    pub embedding_dim: usize,
    /// Projected dimension `D`.
    pub projected_dim: usize,
    /// Independent heads sharing the target construction. The papers use 3; the
    /// ablation says 1 already suffices, so the default is 1.
    pub heads: usize,
    pub mode: Mode,
    /// Ridge penalty `λ`. The papers' tuned range is `[1e-2, 1e6]`.
    pub lambda: f64,
    /// Perturbation strength `α` for the spread-mean construction.
    pub alpha: f64,
}

impl Default for DecideConfig {
    fn default() -> Self {
        DecideConfig {
            embedding_version: "unpinned".to_string(),
            embedding_dim: 384,
            projected_dim: 512,
            heads: 1,
            mode: Mode::Ananc,
            lambda: 100.0,
            alpha: 1.0,
        }
    }
}

impl DecideConfig {
    fn validate(&self) -> Result<(), DecideError> {
        if self.embedding_dim == 0 {
            return Err(DecideError::EmbeddingDim {
                expected: 1,
                got: 0,
            });
        }
        if self.projected_dim == 0 {
            return Err(DecideError::EmbeddingDim {
                expected: 1,
                got: 0,
            });
        }
        if !(1..=3).contains(&self.heads) {
            return Err(DecideError::HeadCount {
                requested: self.heads,
            });
        }
        Ok(())
    }
}

/// One RP + ridge head.
#[derive(Debug, Clone)]
struct Head {
    rp: RpLayer,
    ridge: RidgeHead,
}

/// The engine state.
///
/// Everything here is sufficient statistics plus a solve. Nothing is trained,
/// because there is nothing to train: the answer is a closed-form function of
/// accumulated sums.
#[derive(Debug, Clone)]
pub struct DecideCore {
    config: DecideConfig,
    /// `class_map`: label → id and back. Audit defect A7 was an index↔name
    /// mapping that shifted when names were reordered; the mapping is explicit
    /// here and never derived from position.
    labels: Vec<String>,
    label_ids: rustc_hash::FxHashMap<String, usize>,
    stats: StreamingStats,
    heads: Vec<Head>,
    readouts: Vec<NcmHead>,
    calibrator: Calibrator,
    drift: Option<DriftMonitor>,
    fitted: bool,
}

impl DecideCore {
    /// A fresh engine.
    ///
    /// The projection seeds are derived from the configuration rather than drawn
    /// from entropy, so two engines with the same config are bit-identical —
    /// which is what makes a pinned fixture pinnable.
    pub fn new(config: DecideConfig) -> Result<Self, DecideError> {
        config.validate()?;
        let mut stats = StreamingStats::new(config.embedding_dim);
        stats.ensure_class(0);
        let mut heads = Vec::with_capacity(config.heads);
        for h in 0..config.heads {
            // Distinct, deterministic seeds per head: head 0 must not share a
            // projection with head 1, or "multi-head" is one head counted twice.
            let seed = 0x9E37_79B9_7F4A_7C15u64
                .wrapping_mul(h as u64 + 1)
                .wrapping_add(config.projected_dim as u64);
            heads.push(Head {
                rp: RpLayer::new(config.embedding_dim, config.projected_dim, seed),
                ridge: RidgeHead::new(config.projected_dim, config.lambda),
            });
            heads[h].ridge.ensure_class(0);
        }
        Ok(DecideCore {
            config,
            labels: Vec::new(),
            label_ids: rustc_hash::FxHashMap::default(),
            stats,
            heads,
            readouts: Vec::new(),
            calibrator: Calibrator::uncalibrated(),
            drift: None,
            fitted: false,
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &DecideConfig {
        &self.config
    }

    /// The engine version string.
    pub fn version(&self) -> &'static str {
        ENGINE_VERSION
    }

    /// Whether the current mode can report an out-of-distribution score.
    pub fn has_open_world(&self) -> bool {
        self.config.mode.has_open_world()
    }

    /// Registered labels, in id order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The id of a label.
    pub fn id_of(&self, label: &str) -> Option<usize> {
        self.label_ids.get(label).copied()
    }

    /// The calibrator in force.
    pub fn calibrator(&self) -> &Calibrator {
        &self.calibrator
    }

    /// Whether a fitted readout exists — i.e. whether [`Self::similarities`] and
    /// the `answer_*` methods will answer rather than refuse.
    ///
    /// `observe` invalidates the fit, so the honest answer flips back to `false`
    /// on every update until [`Self::fit`] succeeds again. A host that reports
    /// engine state (`decide_state`) needs this to say "not fitted yet" instead
    /// of implying answers are available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Number of labelled observations recorded, across all classes.
    pub fn samples(&self) -> u64 {
        self.stats.total()
    }

    /// Largest drift observed by the R5 monitor since it was armed.
    ///
    /// `0.0` when no monitor is armed — which is *not* a measurement of zero
    /// drift; it means nothing has been compared. [`Self::needs_recalibration`]
    /// distinguishes "armed and calm" from "never armed" only together with
    /// `calibrator().mechanism`; `decide_state` reports both fields so a reader
    /// never mistakes an unarmed monitor for a clean one.
    pub fn observed_drift(&self) -> f64 {
        self.drift
            .as_ref()
            .map(|d| d.observed_drift())
            .unwrap_or(0.0)
    }

    /// Feed the armed drift monitor a fresh probability vector (D12).
    ///
    /// Returns the drift measured, or `None` when no monitor is armed —
    /// "nothing has been compared", which is not the same as "no drift". The
    /// `decide_state` op calls this on every state read that carries a `probs`
    /// vector, so class-arrival drift is measured where the state is read
    /// rather than whenever someone remembers to check.
    pub fn measure_drift(&mut self, probs: &[f64]) -> Option<f64> {
        self.drift.as_mut().map(|d| d.observe(probs))
    }

    /// Install a calibrator and arm the drift monitor against its reference
    /// distribution (D12).
    pub fn set_calibrator(&mut self, calibrator: Calibrator, reference: Vec<f64>, epsilon: f64) {
        self.calibrator = calibrator;
        self.drift = Some(DriftMonitor::new(reference, epsilon));
    }

    /// Whether the calibrator has been declared stale (R5).
    pub fn needs_recalibration(&self) -> bool {
        self.drift.as_ref().is_some_and(|d| d.needs_recalibration())
    }

    /// Register a class label. Ids are never reused.
    pub fn register(&mut self, label: &str) -> Result<usize, DecideError> {
        if self.label_ids.contains_key(label) {
            return Err(DecideError::DuplicateClass(label.to_string()));
        }
        let id = self.labels.len();
        self.labels.push(label.to_string());
        self.label_ids.insert(label.to_string(), id);
        self.stats.ensure_class(id);
        for h in self.heads.iter_mut() {
            h.ridge.ensure_class(id);
        }
        Ok(id)
    }

    /// Record one labelled embedding.
    ///
    /// `observe` is the name on purpose: this is an `observe`-kind effect in
    /// S21 terms, and there is no `update`/`learn` verb in this crate.
    pub fn observe(&mut self, label: &str, x: &DVector<f64>) -> Result<(), DecideError> {
        let id = *self
            .label_ids
            .get(label)
            .ok_or_else(|| DecideError::UnknownClass(label.to_string()))?;
        if x.len() != self.config.embedding_dim {
            return Err(DecideError::EmbeddingDim {
                expected: self.config.embedding_dim,
                got: x.len(),
            });
        }
        // Input space, under a fixed identity.
        self.stats.push(id, x);
        // RP space, under each head's *fixed* projection.
        for h in self.heads.iter_mut() {
            let z = h.rp.expand(x);
            h.ridge.push(id, &z);
        }
        self.fitted = false;
        Ok(())
    }

    /// Check an embedding against a proposed embedding model version (D7).
    ///
    /// Returns an error rather than silently re-projecting: input-space
    /// statistics accumulated under one model are meaningless under another, and
    /// the only honest response is to refuse and ask for an explicit migration.
    pub fn check_embedding_version(&self, version: &str) -> Result<(), DecideError> {
        if version == self.config.embedding_version {
            Ok(())
        } else {
            Err(DecideError::EmbeddingVersionChanged {
                from: self.config.embedding_version.clone(),
                to: version.to_string(),
            })
        }
    }

    /// Solve every head and build the readouts.
    pub fn fit(&mut self) -> Result<(), DecideError> {
        let mut readouts = Vec::with_capacity(self.heads.len());
        for h in &self.heads {
            // Whitening happens in the space the solve happens in: the head's own
            // RP covariance, not the input-space one.
            let cov = h.ridge.rp_covariance();
            let theta = h.ridge.solve(
                self.config.mode.target_kind(),
                self.config.alpha,
                cov.as_ref(),
            )?;
            // The prototypes the head classifies against are the *images* of the
            // target means under the learned map, exactly as the source methods
            // define them: cosine similarity to a mapped target is the readout.
            let protos = mapped_prototypes(&theta, &h.ridge, cov.as_ref(), &self.config)?;
            readouts.push(NcmHead::new(protos, self.labels.clone()));
        }
        self.readouts = readouts;
        self.fitted = true;
        Ok(())
    }

    /// Cosine similarities from every head, averaged.
    ///
    /// The papers average the heads' outputs; averaging *similarities* rather
    /// than decisions keeps the result a score with the same meaning as a single
    /// head's.
    pub fn similarities(&self, x: &DVector<f64>) -> Result<DVector<f64>, DecideError> {
        if !self.fitted {
            return Err(DecideError::NotFitted);
        }
        let mut acc: Option<DVector<f64>> = None;
        for (h, r) in self.heads.iter().zip(self.readouts.iter()) {
            let z = h.rp.expand(x);
            let s = r.similarities(&z);
            acc = Some(match acc {
                Some(mut a) => {
                    for i in 0..a.len() {
                        a[i] += s[i];
                    }
                    a
                }
                None => s,
            });
        }
        let mut acc = acc.expect("at least one head is configured");
        acc /= self.heads.len() as f64;
        Ok(acc)
    }

    /// The in-distribution score: the maximum mean cosine similarity.
    ///
    /// Only [`Mode::Ananc`] has this mechanism. For `anacp` this returns `None`
    /// rather than a number, because a max-cosine over spread-mean prototypes is
    /// not an OOD score and presenting it as one is audit defect A5.
    pub fn id_score(&self, x: &DVector<f64>) -> Result<Option<f64>, DecideError> {
        if !self.has_open_world() {
            return Ok(None);
        }
        let sims = self.similarities(x)?;
        let mut best = f64::NEG_INFINITY;
        for i in 0..sims.len() {
            if sims[i] > best {
                best = sims[i];
            }
        }
        Ok(Some(best))
    }

    /// Answer a `Choice` question.
    ///
    /// `options` are the candidate *embeddings*, supplied at call time as the
    /// type discipline requires: the engine classifies the candidates it is
    /// given and cannot invent the option set. Fewer than two candidates is
    /// refused — a one-option "choice" is not a decision, and answering it
    /// confidently would hide a caller bug inside a typed answer.
    pub fn answer_choice(
        &self,
        question: &str,
        options: &[DVector<f64>],
    ) -> Result<unfer_protocol::decide::ChoiceAnswer, DecideError> {
        if options.len() < 2 {
            return Err(DecideError::Solve(SolveError::TooFewClasses(options.len())));
        }
        let mut scores = vec![0.0f64; options.len()];
        for (i, o) in options.iter().enumerate() {
            let sims = self.similarities(o)?;
            let mut best = f64::NEG_INFINITY;
            for k in 0..sims.len() {
                if sims[k] > best {
                    best = sims[k];
                }
            }
            scores[i] = best;
        }
        let mut best = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        let calibrated = self
            .calibrator
            .is_calibrated(unfer_protocol::decide::DEFAULT_MAX_ECE);
        // R1 through the answer path: probabilities are `P(label | x)` over
        // the registered classes for the chosen option's embedding, present
        // exactly when the answer claims to be calibrated (see the same note
        // in `answer_score`).
        let probabilities = if calibrated {
            let best_sims = self.similarities(&options[best])?;
            self.calibrator
                .probabilities_from_similarities(best_sims.as_slice())
                .0
        } else {
            Vec::new()
        };
        Ok(unfer_protocol::decide::ChoiceAnswer {
            key: String::new(),
            question: question.to_string(),
            choice: best,
            probabilities,
            confidence: if calibrated {
                Some(self.calibrator.p_id(scores[best]))
            } else {
                None
            },
            calibrated,
            p_ood: self
                .id_score(&options[best])?
                .map(|s| self.calibrator.p_ood(s)),
            engine_version: ENGINE_VERSION.to_string(),
            scores,
        })
    }

    /// Answer a `Score` question: an ordinal rating in `[0, n]`.
    pub fn answer_score(
        &self,
        question: &str,
        x: &DVector<f64>,
        levels: usize,
    ) -> Result<unfer_protocol::decide::ScoreAnswer, DecideError> {
        let sims = self.similarities(x)?;
        let mut best = f64::NEG_INFINITY;
        for i in 0..sims.len() {
            if sims[i] > best {
                best = sims[i];
            }
        }
        // The ordinal is the ID score — the maximum mean cosine similarity,
        // itself in `[-1, 1]` — binned uniformly onto `[0, levels)`. Binned,
        // not ranked: the previous "rank position among the observed
        // similarities" counted `sims[i] > best` where `best` was the maximum
        // of `sims`, so the count was always 0 and every `Score` answer came
        // back as `score: 0` regardless of the input. The cosine range is the
        // method's own scale, so the bin edges stay a function of this
        // question's embedding; the bin count is the question's vocabulary.
        let t = ((best + 1.0) / 2.0).clamp(0.0, 1.0);
        let score = ((t * levels as f64).floor() as usize).min(levels.saturating_sub(1));
        let calibrated = self
            .calibrator
            .is_calibrated(unfer_protocol::decide::DEFAULT_MAX_ECE);
        // R1 through the answer path: a calibrated answer carries the
        // per-class probabilities. An empty list under `calibrated: true` is
        // inadmissible by the protocol's own check, which would refuse every
        // answer a calibrated engine produced.
        let probabilities = if calibrated {
            self.calibrator
                .probabilities_from_similarities(sims.as_slice())
                .0
        } else {
            Vec::new()
        };
        Ok(unfer_protocol::decide::ScoreAnswer {
            key: String::new(),
            question: question.to_string(),
            score,
            levels,
            probabilities,
            confidence: if calibrated {
                Some(self.calibrator.p_id(best))
            } else {
                None
            },
            calibrated,
            p_ood: self.id_score(x)?.map(|s| self.calibrator.p_ood(s)),
            engine_version: ENGINE_VERSION.to_string(),
        })
    }

    /// Answer a `Noul` question: is this true?
    pub fn answer_noul(
        &self,
        question: &str,
        x: &DVector<f64>,
    ) -> Result<unfer_protocol::decide::NoulAnswer, DecideError> {
        let sims = self.similarities(x)?;
        let mut best = f64::NEG_INFINITY;
        for i in 0..sims.len() {
            if sims[i] > best {
                best = sims[i];
            }
        }
        let calibrated = self
            .calibrator
            .is_calibrated(unfer_protocol::decide::DEFAULT_MAX_ECE);
        Ok(unfer_protocol::decide::NoulAnswer {
            key: String::new(),
            question: question.to_string(),
            noul: best,
            confidence: if calibrated {
                Some(self.calibrator.p_id(best))
            } else {
                None
            },
            calibrated,
            p_ood: self.id_score(x)?.map(|s| self.calibrator.p_ood(s)),
            engine_version: ENGINE_VERSION.to_string(),
        })
    }
}

/// Map the target means through the ridge solution to get the prototypes.
///
/// The prototypes are the images of the class means under the learned map — the
/// definition the source methods use. Recomputing the target construction here
/// (rather than storing it in `RidgeHead`) keeps one construction per call site,
/// so the solve and the readout cannot disagree about what was targeted.
fn mapped_prototypes(
    theta: &nalgebra::DMatrix<f64>,
    ridge: &RidgeHead,
    cov: Option<&nalgebra::DMatrix<f64>>,
    config: &DecideConfig,
) -> Result<nalgebra::DMatrix<f64>, DecideError> {
    // RP class means as a `k × C` matrix (classes are rows), matching the
    // orientation the target constructions expect.
    let counts = ridge.counts();
    let cols: Vec<usize> = (0..counts.len()).filter(|&c| counts[c] > 0).collect();
    let mut means = nalgebra::DMatrix::zeros(cols.len(), config.projected_dim);
    for (row, &c) in cols.iter().enumerate() {
        let n = counts[c] as f64;
        let s = ridge.rp_sum(c);
        for k in 0..config.projected_dim {
            means[(row, k)] = s[k] / n;
        }
    }
    let targets = match config.mode.target_kind() {
        // Spread means are written in column-major orientation (features ×
        // classes), so transpose in and back out. This must produce *exactly*
        // what `RidgeHead::solve` produced, or the readout would classify
        // against a different target than the head was fitted to.
        TargetKind::SpreadMeans => {
            targets::spread_means(&means.transpose(), cov, config.alpha).transpose()
        }
        TargetKind::Etf => targets::etf_procrustes(&means, means.nrows()),
    };
    // `theta` is `k × D`; targets are `C × t`. Map each target and transpose to
    // `t × C` so the readout indexes rows by class.
    Ok((theta * targets.transpose()).transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core(mode: Mode) -> DecideCore {
        let mut c = DecideCore::new(DecideConfig {
            embedding_version: "test-v1".into(),
            embedding_dim: 4,
            projected_dim: 32,
            heads: 1,
            mode,
            lambda: 1.0,
            alpha: 1.0,
        })
        .unwrap();
        c.register("a").unwrap();
        c.register("b").unwrap();
        c.register("c").unwrap();
        c
    }

    fn feed(c: &mut DecideCore, label: &str, pts: &[&[f64]]) {
        for p in pts {
            c.observe(label, &DVector::from_row_slice(p)).unwrap();
        }
    }

    #[test]
    fn labels_get_stable_ids_and_are_never_reused() {
        let c = core(Mode::Anacp);
        assert_eq!(c.id_of("a"), Some(0));
        assert_eq!(c.id_of("b"), Some(1));
        assert_eq!(c.id_of("nope"), None);
        assert_eq!(
            c.labels(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn registering_the_same_label_twice_is_refused() {
        let mut c = core(Mode::Anacp);
        assert!(matches!(
            c.register("a"),
            Err(DecideError::DuplicateClass(_))
        ));
    }

    #[test]
    fn observing_an_unregistered_label_is_refused() {
        let mut c = core(Mode::Anacp);
        assert!(matches!(
            c.observe("zzz", &DVector::zeros(4)),
            Err(DecideError::UnknownClass(_))
        ));
    }

    #[test]
    fn a_wrong_embedding_dimension_is_refused_not_truncated() {
        let mut c = core(Mode::Anacp);
        assert!(matches!(
            c.observe("a", &DVector::zeros(3)),
            Err(DecideError::EmbeddingDim {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn answering_before_fitting_is_refused() {
        let mut c = core(Mode::Anacp);
        feed(&mut c, "a", &[&[1.0, 0.0, 0.0, 0.0]]);
        feed(&mut c, "b", &[&[0.0, 1.0, 0.0, 0.0]]);
        feed(&mut c, "c", &[&[0.0, 0.0, 1.0, 0.0]]);
        assert!(matches!(
            c.similarities(&DVector::zeros(4)),
            Err(DecideError::NotFitted)
        ));
    }

    #[test]
    fn too_few_classes_cannot_fit() {
        let mut c = DecideCore::new(DecideConfig {
            embedding_dim: 4,
            projected_dim: 16,
            ..DecideConfig::default()
        })
        .unwrap();
        c.register("only").unwrap();
        c.observe("only", &DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        assert!(
            c.fit().is_err(),
            "a single class cannot define an ETF or a ridge"
        );
    }

    #[test]
    fn separated_clusters_are_classified_correctly() {
        let mut c = core(Mode::Anacp);
        feed(
            &mut c,
            "a",
            &[&[1.0, 0.0, 0.0, 0.0], &[0.95, 0.05, 0.0, 0.0]],
        );
        feed(
            &mut c,
            "b",
            &[&[0.0, 1.0, 0.0, 0.0], &[0.05, 0.95, 0.0, 0.0]],
        );
        feed(
            &mut c,
            "c",
            &[&[0.0, 0.0, 1.0, 0.0], &[0.0, 0.05, 0.95, 0.0]],
        );
        c.fit().unwrap();
        let sims = c
            .similarities(&DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let mut best = 0usize;
        for i in 1..sims.len() {
            if sims[i] > sims[best] {
                best = i;
            }
        }
        assert_eq!(c.labels()[best], "a");
    }

    #[test]
    fn the_two_modes_select_different_target_constructions() {
        assert_eq!(Mode::Anacp.target_kind(), TargetKind::SpreadMeans);
        assert_eq!(Mode::Ananc.target_kind(), TargetKind::Etf);
        assert!(!Mode::Anacp.has_open_world());
        assert!(Mode::Ananc.has_open_world());
    }

    /// Audit defect A5, enforced: `anacp` has no OOD mechanism, so it must not
    /// produce an ID score rather than inventing one.
    #[test]
    fn anacp_reports_no_open_world_score_at_all() {
        let mut c = core(Mode::Anacp);
        feed(&mut c, "a", &[&[1.0, 0.0, 0.0, 0.0]]);
        feed(&mut c, "b", &[&[0.0, 1.0, 0.0, 0.0]]);
        feed(&mut c, "c", &[&[0.0, 0.0, 1.0, 0.0]]);
        c.fit().unwrap();
        assert_eq!(
            c.id_score(&DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]))
                .unwrap(),
            None,
            "anacp must not manufacture an ID score"
        );
    }

    #[test]
    fn an_embedding_model_change_is_refused_with_its_own_error() {
        let c = core(Mode::Anacp);
        assert!(c.check_embedding_version("test-v1").is_ok());
        match c.check_embedding_version("other-v2") {
            Err(DecideError::EmbeddingVersionChanged { from, to }) => {
                assert_eq!(from, "test-v1");
                assert_eq!(to, "other-v2");
            }
            other => panic!("expected a version error, got {other:?}"),
        }
    }

    #[test]
    fn an_unfitted_engine_never_claims_to_be_calibrated() {
        let mut c = core(Mode::Ananc);
        feed(&mut c, "a", &[&[1.0, 0.0, 0.0, 0.0]]);
        feed(&mut c, "b", &[&[0.0, 1.0, 0.0, 0.0]]);
        feed(&mut c, "c", &[&[0.0, 0.0, 1.0, 0.0]]);
        c.fit().unwrap();
        let a = c
            .answer_choice(
                "q",
                &[
                    DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]),
                    DVector::from_row_slice(&[0.0, 1.0, 0.0, 0.0]),
                ],
            )
            .unwrap();
        assert!(!a.calibrated);
        assert!(a.confidence.is_none(), "no confidence without calibration");
        assert!(
            a.probabilities.is_empty(),
            "no probabilities without calibration"
        );
        assert!(a.p_ood.is_some(), "ananc does have an open-world state");
    }

    #[test]
    fn an_engine_with_the_same_config_is_bit_identical() {
        // The property that makes a pinned fixture pinnable.
        let cfg = DecideConfig {
            embedding_version: "v".into(),
            embedding_dim: 6,
            projected_dim: 24,
            heads: 2,
            mode: Mode::Ananc,
            lambda: 1.0,
            alpha: 1.0,
        };
        let mut x = DecideCore::new(cfg.clone()).unwrap();
        let mut y = DecideCore::new(cfg).unwrap();
        for c in [&mut x, &mut y] {
            c.register("a").unwrap();
            c.register("b").unwrap();
            c.observe(
                "a",
                &DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
            c.observe(
                "b",
                &DVector::from_row_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
            c.fit().unwrap();
        }
        let probe = DVector::from_row_slice(&[1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            x.similarities(&probe).unwrap(),
            y.similarities(&probe).unwrap()
        );
    }

    #[test]
    fn a_bad_head_count_is_refused_at_construction() {
        let e = DecideCore::new(DecideConfig {
            heads: 9,
            ..DecideConfig::default()
        })
        .unwrap_err();
        assert!(matches!(e, DecideError::HeadCount { requested: 9 }));
    }

    /// The degenerate `answer_score` this replaces: it counted `sims[i] > best`
    /// where `best` was the maximum of `sims`, so the count was always 0 and
    /// every ordinal came back as 0. The ordinal must move with the input.
    #[test]
    fn score_answers_move_with_the_input_instead_of_always_reporting_zero() {
        let mut c = core(Mode::Anacp);
        feed(&mut c, "a", &[&[1.0, 0.0, 0.0, 0.0]]);
        feed(&mut c, "b", &[&[0.0, 1.0, 0.0, 0.0]]);
        feed(&mut c, "c", &[&[0.0, 0.0, 1.0, 0.0]]);
        c.fit().unwrap();
        let strong = c
            .answer_score(
                "how strong?",
                &DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]),
                4,
            )
            .unwrap();
        let weak = c
            .answer_score(
                "how strong?",
                &DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0]),
                4,
            )
            .unwrap();
        assert!(
            strong.score > weak.score,
            "an on-manifold input must outrank an orthogonal one: {} vs {}",
            strong.score,
            weak.score
        );
        assert!(weak.score < 4, "score must stay in [0, levels)");
        assert!(strong.levels == 4);
    }

    /// R1 through the answer path: a calibrated answer carries probabilities
    /// that sum to 1, so the protocol's admissibility check passes.
    #[test]
    fn a_calibrated_answer_carries_probabilities_that_sum_to_one() {
        let mut c = core(Mode::Ananc);
        feed(&mut c, "a", &[&[1.0, 0.0, 0.0, 0.0]]);
        feed(&mut c, "b", &[&[0.0, 1.0, 0.0, 0.0]]);
        feed(&mut c, "c", &[&[0.0, 0.0, 1.0, 0.0]]);
        c.fit().unwrap();
        c.set_calibrator(
            Calibrator {
                mechanism: Mechanism::Temperature,
                temperature: 2.0,
                fit_ece: 0.01,
                ..Calibrator::default()
            },
            vec![0.5, 0.5],
            0.05,
        );
        let a = c
            .answer_choice(
                "which?",
                &[
                    DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0]),
                    DVector::from_row_slice(&[0.0, 1.0, 0.0, 0.0]),
                ],
            )
            .unwrap();
        assert!(a.calibrated, "measured ECE at the bar means calibrated");
        assert!(
            !a.probabilities.is_empty(),
            "calibrated with no probabilities is inadmissible"
        );
        let sum: f64 = a.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "R1 violated: sum = {sum}");
        assert!(
            unfer_protocol::decide::Answer::Choice(a.clone()).is_admissible_as_advice(),
            "calibrated + non-empty probabilities must be admissible advice"
        );
    }

    /// D12: no monitor armed means *not measured* (None), never "zero drift".
    #[test]
    fn measure_drift_reports_none_until_a_monitor_is_armed_and_then_measures() {
        let mut c = core(Mode::Ananc);
        assert_eq!(c.measure_drift(&[0.5, 0.5]), None);
        assert!(!c.needs_recalibration());
        c.set_calibrator(Calibrator::uncalibrated(), vec![0.9, 0.1], 0.05);
        let d = c.measure_drift(&[0.5, 0.5]).expect("armed monitor");
        assert!(
            d > 0.05,
            "synthetic drift beyond epsilon must be measured: {d}"
        );
        assert!(c.needs_recalibration());
        assert!(c.observed_drift() > 0.0);
    }
}
