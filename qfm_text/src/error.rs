//! Error types for `qfm_text`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QfmTextError {
    /// The manifest JSON failed to parse or is missing required fields.
    #[error("bad manifest at {path}: {reason}")]
    BadManifest { path: String, reason: String },

    /// A shard file failed to open or has the wrong byte count.
    #[error("bad shard {path}: {reason}")]
    BadShard { path: String, reason: String },

    /// A token id in a shard falls outside the declared vocabulary.
    #[error("token id {token_id} in shard {path} is >= vocab_size {vocab_size}")]
    VocabMismatch {
        token_id: u32,
        vocab_size: u32,
        path: String,
    },

    /// The compiled QFM pipeline returned an error during text-model
    /// construction (e.g. degenerate basis, dimension mismatch).
    #[error("qfm error: {0}")]
    Qfm(#[from] qfm::QfmError),

    /// A histogram was capped or a config was asked to do something
    /// infeasible (e.g. `n_orders > MAX_ORDERS`).
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// I/O error from the underlying filesystem.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
