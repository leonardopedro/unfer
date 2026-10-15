//! The **sequential AnaCP→AnaNC** decision model: one stage after the other.
//!
//! The governing plan's product shape is strictly staged —
//!
//! > EmbeddingGemma2 → **stage 1: AnaCP** (closed-form contrastive projection
//! > adapts the backbone's features) → **stage 2: AnaNC** (collapse readout:
//! > ETF target geometry, max-cosine decision, ID/OOD score) *on the projected
//! > features* — neither "one or another" nor an average.
//!
//! The shipped engine previously realized that sequence *folded into one
//! closed-form map with a selectable target geometry* (surface selection, see
//! [`crate::Mode`]). This module realizes it **literally, as two engines in
//! sequence**, which is also how the source methods are meant to compose: the
//! official implementations are standalone modules (`SalehMomeni/AnaCP`,
//! `SalehMomeni/AnaNC`) and AnaNC's own README says to integrate it "by
//! passing feature embeddings" — exactly what stage 1 emits.
//!
//! # The handoff
//!
//! Stage 1's adapted feature ([`DecideCore::project`], L2-normalized
//! `f = θᵀ·GELU(xR)`) **is** stage 2's embedding. The reference
//! implementations normalize at this seam (`F.normalize` after the ridge) and
//! stage 2 normalizes again at its own seam — idempotent, so the composition
//! is well-defined without an extra convention.
//!
//! # The staleness problem, and this design's answer
//!
//! Stage 2's class statistics live in *stage-1 feature space*, which moves
//! whenever stage 1 re-solves (`observe` → `fit` changes `θ`). Accumulating
//! stage-2 sums under one projection and answering under another is audit
//! defect **A4** at the sequence level — the same defect the engine already
//! refuses at the embedding seam ([`DecideCore::check_embedding_version`]).
//!
//! This module solves it **by rebuilding**: `fit()` fits stage 1 first, then
//! re-expresses every retained observation through the *current* stage-1 map
//! into a fresh stage-2 engine. Stage-2 statistics are therefore *always*
//! expressed under the projection that will answer — by construction, not by
//! hope. The cost is retaining the raw stream (O(n·d)), where each standalone
//! method streams in O(1) memory. That cost is the honest price of staging
//! without replay; the reference AnaCP pays a related price differently — it
//! regenerates pseudo-replay inputs from class Gaussians to refit its second
//! stage under the current θ. Replaying from a streaming Gaussian class model
//! instead of the raw stream is the obvious upgrade path, and it is *not*
//! implemented here because the replay generator is a statistical claim about
//! the data that would need its own validation.
//!
//! # What the composition inherits
//!
//! * **Open world**: stage 2 is AnaNC, so [`SequentialCore::id_score`] returns
//!   `Some(max-cosine)` — the ID/OOD mechanism a pure-AnaCP surface honestly
//!   does not have (`Mode::Anacp` returns `None`).
//! * **Decoupled widths**: stage 1 and stage 2 each carry their own
//!   `projected_dim`, so the measured conflict in the D-ablation (AnaCP best at
//!   `D = 512`, AnaNC best at `D = 16`, shared default wrong for one of them)
//!   stops being a single shared knob.
//! * **The trust boundary, unchanged**: every answer is advice; nothing here
//!   emits probabilities or claims calibration — see `DECIDE_SPEC` §0.
//!
//! # Example
//!
//! ```rust
//! use decide_core::{DecideConfig, Mode};
//! use decide_core::sequential::{SequentialConfig, SequentialCore};
//! use nalgebra::DVector;
//!
//! let d_backbone = 8;
//! let d_cp = 16; // stage-1 width  = stage-2 input width
//! let d_nc = 8;  // stage-2 width
//! let mut seq = SequentialCore::new(SequentialConfig {
//!     cp: DecideConfig {
//!         embedding_version: "test-v1".into(),
//!         embedding_dim: d_backbone,
//!         projected_dim: d_cp,
//!         heads: 1,
//!         mode: Mode::Anacp,
//!         lambda: 10.0,
//!         alpha: 1.0,
//!     },
//!     nc: DecideConfig {
//!         embedding_version: "test-v1".into(),
//!         embedding_dim: d_cp,
//!         projected_dim: d_nc,
//!         heads: 1,
//!         mode: Mode::Ananc,
//!         lambda: 10.0,
//!         alpha: 1.0,
//!     },
//! })
//! .unwrap();
//! for l in ["a", "b", "c"] {
//!     seq.register(l).unwrap();
//! }
//! seq.observe("a", &DVector::from_vec(vec![1.0; d_backbone])).unwrap();
//! seq.observe("b", &DVector::from_vec(vec![-1.0; d_backbone])).unwrap();
//! seq.observe("c", &DVector::from_vec(vec![0.5; d_backbone])).unwrap();
//! seq.fit().unwrap();
//! let x = DVector::from_vec(vec![1.0; d_backbone]);
//! let sims = seq.similarities(&x).unwrap(); // stage 2's cosines
//! let id = seq.id_score(&x).unwrap().expect("stage 2 is AnaNC: the sequence has an ID score");
//! assert_eq!(sims.len(), 3);
//! assert!((-1.0..=1.0).contains(&id));
//! # Ok::<(), decide_core::DecideError>(())
//! ```

use nalgebra::DVector;

use crate::{DecideConfig, DecideCore, DecideError, Mode};

/// Configuration of the sequence: one config per stage.
///
/// Stage 1 must be [`Mode::Anacp`] (feature adaptation) and stage 2
/// [`Mode::Ananc`] (collapse readout + ID score); the reverse order would be
/// collapse-before-adaptation, which no source specifies. Stage 2's
/// `embedding_dim` must equal stage 1's `projected_dim` — that equality *is*
/// the handoff.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SequentialConfig {
    /// Stage 1 — AnaCP.
    pub cp: DecideConfig,
    /// Stage 2 — AnaNC, consuming stage 1's projected features.
    pub nc: DecideConfig,
}

impl SequentialConfig {
    fn validate(&self) -> Result<(), DecideError> {
        if self.cp.mode != Mode::Anacp {
            return Err(DecideError::Staging(
                "stage 1 must be Mode::Anacp (adaptation first)".into(),
            ));
        }
        if self.nc.mode != Mode::Ananc {
            return Err(DecideError::Staging(
                "stage 2 must be Mode::Ananc (collapse readout second)".into(),
            ));
        }
        if self.nc.embedding_dim != self.cp.projected_dim {
            return Err(DecideError::Staging(format!(
                "stage 2 consumes stage 1's features: nc.embedding_dim = {} must equal \
                 cp.projected_dim = {}",
                self.nc.embedding_dim, self.cp.projected_dim
            )));
        }
        Ok(())
    }
}

/// The staged engine: AnaCP in front, AnaNC behind. See the module docs.
#[derive(Debug, Clone)]
pub struct SequentialCore {
    /// Kept so `fit` can rebuild stage 2 from scratch.
    nc_config: DecideConfig,
    /// Stage 1: the adaptation engine (owns the raw stream's statistics).
    stage1: DecideCore,
    /// Stage 2: rebuilt on every `fit`, always under the current stage-1 map.
    stage2: DecideCore,
    /// Label vocabulary, in registration order — ids are shared by both stages
    /// and by the retained raw stream.
    labels: Vec<String>,
    /// The retained raw stream (the price of staleness-free staging; see the
    /// module docs' memory note).
    raw: Vec<(usize, DVector<f64>)>,
    fitted: bool,
}

impl SequentialCore {
    /// Build the sequence, validating the staging contract.
    pub fn new(config: SequentialConfig) -> Result<Self, DecideError> {
        config.validate()?;
        let stage1 = DecideCore::new(config.cp.clone())?;
        let stage2 = DecideCore::new(config.nc.clone())?;
        Ok(SequentialCore {
            nc_config: config.nc,
            stage1,
            stage2,
            labels: Vec::new(),
            raw: Vec::new(),
            fitted: false,
        })
    }

    /// Registered labels, in id order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Number of retained observations.
    pub fn samples(&self) -> usize {
        self.raw.len()
    }

    /// Stage 1, for introspection (its config, fit state, readouts).
    pub fn stage1(&self) -> &DecideCore {
        &self.stage1
    }

    /// Stage 2, for introspection. Rebuilt on every `fit`.
    pub fn stage2(&self) -> &DecideCore {
        &self.stage2
    }

    /// Whether the sequence can answer (both stages fitted).
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Register a label in both stages. Ids are never reused.
    pub fn register(&mut self, label: &str) -> Result<usize, DecideError> {
        let id = self.stage1.register(label)?;
        // Stage 2 is rebuilt on fit; register there too so partial fits (and
        // the rebuilt engine) agree on vocabulary.
        self.stage2.register(label)?;
        self.labels.push(label.to_string());
        debug_assert_eq!(id + 1, self.labels.len());
        Ok(id)
    }

    /// Record one labelled embedding in stage 1 (and in the retained stream).
    ///
    /// Stage 2 is deliberately *not* fed here: its statistics are created at
    /// `fit` time under the stage-1 map that will actually answer.
    pub fn observe(&mut self, label: &str, x: &DVector<f64>) -> Result<(), DecideError> {
        let id = self
            .stage1
            .id_of(label)
            .ok_or_else(|| DecideError::UnknownClass(label.to_string()))?;
        self.stage1.observe(label, x)?;
        self.raw.push((id, x.clone()));
        self.fitted = false;
        Ok(())
    }

    /// Fit stage 1, then rebuild and fit stage 2 under its current map.
    ///
    /// Order matters and is the whole point: stage 1 solves first, and every
    /// retained observation is then re-projected through the *resulting* `θ`.
    /// A stage-2 failure (e.g. only one class observed — no between-class
    /// geometry) leaves the sequence unfitted rather than half-fitted.
    pub fn fit(&mut self) -> Result<(), DecideError> {
        self.stage1.fit()?;
        let mut fresh = DecideCore::new(self.nc_config.clone())?;
        for label in &self.labels {
            fresh.register(label)?;
        }
        for (id, x) in &self.raw {
            let f = self.stage1.project(x)?;
            fresh.observe(&self.labels[*id], &f)?;
        }
        fresh.fit()?;
        self.stage2 = fresh;
        self.fitted = true;
        Ok(())
    }

    /// The handoff itself: stage 1's adapted feature (what stage 2 sees).
    pub fn project(&self, x: &DVector<f64>) -> Result<DVector<f64>, DecideError> {
        self.stage1.project(x)
    }

    /// Stage 2's cosine similarities for `x`, through the sequence.
    pub fn similarities(&self, x: &DVector<f64>) -> Result<DVector<f64>, DecideError> {
        if !self.fitted {
            return Err(DecideError::NotFitted);
        }
        let f = self.stage1.project(x)?;
        self.stage2.similarities(&f)
    }

    /// The in-distribution score — inherited from stage 2.
    ///
    /// This is one of the composition's concrete gains: a pure AnaCP surface
    /// honestly returns `None` (max-cosine over spread-mean prototypes is not
    /// an OOD score), while the sequence carries AnaNC's max-cosine ID score
    /// *over the adapted features*.
    pub fn id_score(&self, x: &DVector<f64>) -> Result<Option<f64>, DecideError> {
        if !self.fitted {
            return Err(DecideError::NotFitted);
        }
        let f = self.stage1.project(x)?;
        self.stage2.id_score(&f)
    }

    /// The argmax label and its cosine similarity (advice-shaped; scores only).
    pub fn predict(&self, x: &DVector<f64>) -> Result<(String, f64), DecideError> {
        let sims = self.similarities(x)?;
        let mut best = 0usize;
        for i in 1..sims.len() {
            if sims[i] > sims[best] {
                best = i;
            }
        }
        let label = self
            .labels
            .get(best)
            .cloned()
            .unwrap_or_default();
        Ok((label, sims[best]))
    }
}
