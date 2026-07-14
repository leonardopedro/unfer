//! Text-model configuration (Stage 2+ config object).
//!
//! This is the canonical config struct for `qfm_text`, used by the
//! streaming accumulator (`accumulate.rs`), the model builder
//! (`model.rs`), the LM (Stage 5), and the train/eval binaries
//! (Stage 6). It is `serde::Serialize + Deserialize` so it can be
//! loaded from a TOML config (`bin/qfm_text_train.rs --config foo.toml`).
//!
//! All fields are *sensible defaults* drawn from the QFM-Text plan
//! preamble; the Stage 6 sweep grid searches over `t`, `lambda`, and
//! `discount` only.

use serde::{Deserialize, Serialize};

/// How the Krylov-decoded `p̃` (a K₂-length vector) is interpreted
/// before the per-mode histogram marginalization. See
/// `QfmTextModel::preprocess_p_tilde` and
/// `docs/QFM_TEXT_STATUS.md` §"The real bottleneck" for the
/// motivation.
///
/// The default is [`DecodeStrategy::Renormalize`], which is the
/// structural fix for the unigram-floor collapse. The other
/// variants are research alternatives; [`DecodeStrategy::Dense`]
/// reproduces the original (pre-fix) behavior for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecodeStrategy {
    /// Original behavior: take `p̃` at face value, accumulate
    /// `total_w = Σ p̃[m]` over active modes (those with a
    /// histogram entry), and route the residual `1 - total_w` to
    /// the global unigram. This is the "broken" behavior that
    /// causes the unigram-floor collapse when the Krylov decoder
    /// spreads mass across many inactive modes.
    Dense,
    /// Renormalize `p̃` over active modes, so `total_w → 1` and
    /// the global unigram floor vanishes. The escape mass of each
    /// per-mode histogram is distributed to the unigram (the
    /// standard Katz backoff), and the histograms carry the full
    /// context-conditioned distribution. This is the recommended
    /// default and was the diagnostic in
    /// `QFM_TEXT_STATUS.md` §"What this means for the architecture"
    /// (item 1: "Decode threshold").
    Renormalize,
    /// Sparse top-k selection: keep only the `k` highest-`p̃`
    /// active modes, zero the rest, then renormalize. The
    /// equivalent of `Renormalize` with a hard sparsity cap
    /// (QFM_TEXT_STATUS.md item 2: "Sparse marginalization").
    /// `top_k` is set on [`TextConfig::top_k`].
    TopK,
    /// Multiply each active mode's `p̃[m]` by `λ_o / Σλ_o` (where
    /// `o` is the order of mode `m`) before renormalizing. This
    /// favours higher-order modes (more context) over lower-order
    /// ones, shifting mass away from the unigram floor. QFM_TEXT_STATUS.md
    /// item 3: "Per-mode weight prior".
    OrderPrior,
}

impl Default for DecodeStrategy {
    fn default() -> Self {
        DecodeStrategy::Renormalize
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextConfig {
    /// Number of context orders (1..=n). Order `o` uses the *last* `o`
    /// tokens. Default: 4.
    pub n_orders: usize,
    /// Maximum number of distinct (token, count) entries per mode
    /// histogram. On overflow, the smallest-count entry is evicted
    /// and its count is added to `escape`. Default: 64.
    pub hist_cap: usize,
    /// Krylov rank for the SIRK reduce step. Default: 8.
    pub max_rank: usize,
    /// Number of negative-imaginary-axis shifts for the SIRK solve.
    /// Default: 8.
    pub m_shifts: usize,
    /// Per-order coefficients `λ_o` on the dressed-vacuum projector
    /// terms. `H = Σ_o λ_o |0̃_o⟩⟨0̃_o|`. Default: `vec![1.0; n_orders]`.
    pub lambda: Vec<f64>,
    /// Evolution time `t` in the Born-rule decode
    /// `c_1 = exp(-i H_m t) c_0`. Default: 1.0.
    pub t: f64,
    /// Absolute-discount hyperparameter for the smoothed per-mode
    /// histogram. Default: 0.75 (the classic Katz backoff choice).
    pub discount: f64,
    /// PRNG seed. Default: 0.
    pub seed: u64,
    /// How the Krylov-decoded `p̃` is interpreted before
    /// marginalization. Default: [`DecodeStrategy::Renormalize`]
    /// (the structural fix for the unigram-floor collapse).
    #[serde(default)]
    pub decode_strategy: DecodeStrategy,
    /// Top-k sparsity for [`DecodeStrategy::TopK`]. Default: 4.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Per-order hash block size for the [`crate::features::OrderHasher`]
    /// encoder (rev 35). `block_sizes[o]` is the number of distinct
    /// modes that contexts of order `o+1` are hashed to. Default:
    /// `vec![1 << 20; n_orders]` (≈ 1M per order, 4M total). Each
    /// mode is 32-bit; a 4M-mode table is 32 MB of mode indices in
    /// the accumulator's `stats` map, plus per-mode histogram memory.
    /// For the full WikiText-103 corpus (128M windows, 5M+ unique
    /// tokens), the 4M-mode table is sufficient: hash collisions
    /// blend unrelated contexts but the model generalizes to
    /// unseen contexts at test time.
    #[serde(default = "default_block_sizes")]
    pub block_sizes: Vec<usize>,
    /// Per-order salt for the hash mixer (decorrelates hashes
    /// across orders). Default: `vec![1, 2, ..., n_orders]`.
    #[serde(default = "default_salts")]
    pub salts: Vec<u64>,
    /// Encoder selector (rev 37). When `true`, use the per-context
    /// `ContextRegistry` (rev 36) instead of the `OrderHasher`
    /// (rev 35) for the accumulator's mode assignment. Default:
    /// `true` (rev 36 nohash, the empirically better encoder on
    /// shard 0). Set to `false` to reproduce the rev 35
    /// `OrderHasher` behavior (degenerate Krylov basis, but
    /// bounded memory).
    #[serde(default = "default_true")]
    pub use_registry_encoder: bool,
    /// Resolution of the Fock-space hypersphere partition — the
    /// number of equally-spaced directions (inner wave-functions)
    /// used to define the outer vacuum |Ψ_0⟩ (rev 37 v3).
    ///
    /// The outer vacuum c_0 is uniform in the Fock basis at this
    /// resolution R:
    ///   `c_0[fock] = (1/√R) · (1, 1, ..., 1, 0, ...)`
    /// with non-zero amplitude on the first R basis directions.
    /// The Krylov projection sums the first M ≤ R rows of W
    /// (the training modes) and divides by √R. The remaining
    /// R-M directions have no training data (zero Krylov
    /// projection).
    ///
    /// Must be larger than M (the number of training sequences)
    /// to allow distinguishing the training data. A reasonable
    /// default is `10 * M`.
    ///
    /// When `None`, the model builder computes
    /// `R = 10 * w.nrows()` at build time.
    #[serde(default)]
    pub fock_resolution: Option<u64>,
}

fn default_top_k() -> usize {
    4
}

fn default_true() -> bool {
    true
}

fn default_block_sizes() -> Vec<usize> {
    // 4M modes total by default. The accumulator grows the stats
    // map only on observed modes, so the actual peak memory is
    // dominated by the per-mode histograms (hist_cap × observed).
    vec![1 << 20; 4]
}

fn default_salts() -> Vec<u64> {
    vec![1, 2, 3, 4]
}

impl Default for TextConfig {
    fn default() -> Self {
        const N_ORDERS: usize = 4;
        Self {
            n_orders: N_ORDERS,
            hist_cap: 64,
            max_rank: 8,
            m_shifts: 8,
            lambda: vec![1.0; N_ORDERS],
            t: 1.0,
            discount: 0.75,
            seed: 0,
            decode_strategy: DecodeStrategy::default(),
            top_k: default_top_k(),
            block_sizes: default_block_sizes(),
            salts: default_salts(),
            use_registry_encoder: default_true(),
            fock_resolution: None,
        }
    }
}

impl TextConfig {
    /// Total number of modes across all orders: `Σ_o block_sizes[o]`,
    /// plus 1 for the vacuum. This is the `k2_total` argument to
    /// `qfm::QfmPipeline::compile_channels`.
    pub fn k2_total(&self) -> u32 {
        let sum: u64 = self.block_sizes.iter().map(|&x| x as u64).sum();
        (sum + 1) as u32
    }

    /// Cumulative offset of the order-`o` block in the global mode
    /// space. Order 0 starts at 1 (index 0 is the vacuum).
    pub fn offset(&self, order: usize) -> u32 {
        let mut off = 1u32;
        for o in 0..order {
            off += self.block_sizes[o] as u32;
        }
        off
    }

    /// Map a global mode index to its context order. The mode index
    /// `0` is the vacuum; modes `[offset(o), offset(o) + block_size(o))`
    /// belong to order `o`. Returns `n_orders` for the vacuum
    /// (out-of-range, treated as "no order").
    pub fn order_of(&self, mode: u32) -> usize {
        if mode == 0 {
            return self.n_orders;
        }
        let mut off = 1u32;
        for o in 0..self.n_orders {
            let block = self.block_sizes[o] as u32;
            if mode < off + block {
                return o;
            }
            off += block;
        }
        self.n_orders
    }
}
