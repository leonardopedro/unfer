//! `qfm_text` — train the QFM.tex architecture on token streams.
//!
//! This crate implements the QFM-Text plan (`docs/QFM_TEXT_HRM_PLAN.md`):
//! the *model* is the Tomographic Subspace Recovery pipeline of QFM.tex
//! (hierarchical dressed vacua + SIRK Krylov reduction + Born-rule token
//! head), and the *data + training process* are the ones used by
//! [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) (its
//! `data_io` corpus cleaning/tokenization, its `sample_tokenized.py`
//! stratified epoch sampling, and its perplexity-first evaluation).
//!
//! # Module overview
//!
//! - [`corpus`]: memory-mapped shard reader (Stage 1). `Shard` wraps a
//!   `Mmap` of little-endian u32 token ids; `Manifest` parses the JSON
//!   metadata emitted by `scripts/prepare_corpus.py`; `WindowIter`
//!   yields `(context, next)` training pairs.
//! - [`registry`]: the **per-context** Level-2 encoder (Stage 2, rev
//!   36). `ContextRegistry` assigns a fresh mode index to every
//!   distinct context the corpus produces — no hashing, no
//!   collisions, no bounded table. Unseen test contexts fall back
//!   to the reserved vacuum mode 0, whose histogram is the unigram;
//!   the dressed-vacuum projector in H then mixes it with the
//!   registry modes' histograms through the Krylov evolution.
//! - [`accumulate`]: the streaming pass (Stage 2). `ChannelAccumulator`
//!   is the FSDP-analog — one per shard, `merge` is the all-reduce.
//!   `ModeStats` is a per-mode count + capped histogram + escape count.
//!   The streaming pass takes a `&mut ContextRegistry` and grows it
//!   as new contexts are observed.
//! - [`model`]: the compiled LM (Stage 4). `QfmTextModel` wraps the
//!   `QfmPipeline` + per-mode histograms + unigram floor + the
//!   `ContextRegistry`, with a per-context autoregressive
//!   `next_token_dist` and a serialized save/load.
//! - [`lm`]: scoring + sampling (Stage 5). `perplexity`, `sample_text`,
//!   and the `NgramBaseline` for honest classical comparison.
//! - [`incontext`]: the Quantum Bayesian Update as in-context
//!   adaptation (Stage 5). `adapt_prior` builds a sliding-window
//!   `Posterior` and samples via HMC.
//!
//! # Honest scope
//!
//! This is a quantum-kernel n-gram-family model with coherent Krylov
//! smoothing across backoff orders. The success criterion is *beating
//! classical interpolated/backoff n-gram baselines at equal context
//! order on the same corpus*, and demonstrating the QFM pipeline
//! end-to-end at corpus scale (10⁷–10⁸ tokens, K₂ ~ 10⁵–10⁷). It will
//! not approach a 1B HRM-Text transformer; that comparison is reported
//! for honesty, not as a target.

pub mod accumulate;
pub mod config;
pub mod corpus;
pub mod error;
pub mod features;
pub mod incontext;
pub mod lm;
pub mod model;
pub mod oxieml_decoder;
pub mod registry;

pub use accumulate::{ChannelAccumulator, Encoder, ModeStats, accumulate_shards};
pub use config::{DecodeStrategy, TextConfig};
pub use corpus::{Manifest, Shard, ShardEntry, WindowIter};
pub use error::QfmTextError;
pub use features::{CollisionStats, OrderHasher, context_orders, splitmix64, splitmix64_seq};
pub use incontext::{HmcIncontextOpts, adapt_prior, next_token_dist_adapted};
pub use lm::{
    NgramBaseline, PerplexityReport, perplexity, perplexity_baseline, perplexity_baseline_capped,
    perplexity_capped, perplexity_model_avg, perplexity_model_avg_capped, sample_text,
};
pub use model::{QfmTextModel, TextModelMetadata};
pub use oxieml_decoder::{ColumnFit, OxiemlFitOpts, evaluate_column, fit_column, fit_decoder};
pub use registry::{ContextRegistry, VACUUM_MODE};

/// Schema version stamped on every serialized `QfmTextModel` and shard
/// `Manifest` so future readers can reject incompatible binaries.
///
/// - `1` = hashed Level-2 encoder (`OrderHasher`, `block_sizes`,
///   `salts`). The rev 35 design.
/// - `2` = per-context `ContextRegistry` (rev 36, hashing removed).
/// - `3` = rev 37: `OrderHasher` encoder restored, optional oxieml
///   SymRegEngine decoder (see `qfm_text/src/oxieml_decoder.rs`).
///   The W matrix may be replaced by `Vec<EmlTree>` when
///   `--oxieml-fit` is used during training.
/// - `4` = rev 37 v3: the Krylov initial vector c_0 is the
///   **L^1 outer vacuum** |Ψ_0⟩ = (M^T M)^-1 M^T 1 (M = |W|),
///   precomputed at model-build time and stored in the payload
///   as `outer_vacuum: Vec<Complex64>`. The previous per-context
///   superposition `(1/√(n+1))(W[0, :] + Σ_o W[m_o, :])` is
///   removed (it was a superposition of seen modes, which the
///   user's design constraint forbids).
pub const SCHEMA_VERSION: u32 = 4;
