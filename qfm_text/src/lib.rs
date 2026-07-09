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
//! - [`features`]: the hashed Level-2 encoder (Stage 2). `OrderHasher`
//!   maps a context of `o` tokens to a single mode index in
//!   `[offset_o, offset_o + block_size_o)`, deterministic and bounded.
//! - [`accumulate`]: the streaming pass (Stage 2). `ChannelAccumulator`
//!   is the FSDP-analog — one per shard, `merge` is the all-reduce.
//!   `ModeStats` is a per-mode count + capped histogram + escape count.
//! - [`model`]: the compiled LM (Stage 4). `QfmTextModel` wraps the
//!   `QfmPipeline` + per-mode histograms + unigram floor, with a
//!   per-context autoregressive `next_token_dist` and a serialized
//!   save/load.
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
//! end-to-end at corpus scale (10⁷–10⁸ tokens, K₂ ~ 10⁵). It will not
//! approach a 1B HRM-Text transformer; that comparison is reported for
//! honesty, not as a target.

pub mod accumulate;
pub mod config;
pub mod corpus;
pub mod error;
pub mod features;
pub mod incontext;
pub mod lm;
pub mod model;

pub use accumulate::{ChannelAccumulator, ModeStats, accumulate_shards};
pub use config::{DecodeStrategy, TextConfig};
pub use corpus::{Manifest, Shard, ShardEntry, WindowIter};
pub use error::QfmTextError;
pub use features::{OrderHasher, context_orders, splitmix64_seq};
pub use incontext::{HmcIncontextOpts, adapt_prior, next_token_dist_adapted};
pub use lm::{
    NgramBaseline, PerplexityReport, perplexity, perplexity_baseline, perplexity_baseline_capped,
    perplexity_capped, perplexity_model_avg, perplexity_model_avg_capped, sample_text,
};
pub use model::{QfmTextModel, TextModelMetadata};

/// Schema version stamped on every serialized `QfmTextModel` and shard
/// `Manifest` so future readers can reject incompatible binaries.
pub const SCHEMA_VERSION: u32 = 1;
