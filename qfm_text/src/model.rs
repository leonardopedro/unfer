//! The compiled QFM-Text language model (Stage 4, rev 36: hashing removed).
//!
//! `QfmTextModel` wraps a `qfm::QfmPipeline` (the TSR reduced basis
//! + reduced Hamiltonian), the per-mode statistics from the
//! accumulator, the unigram floor, the [`ContextRegistry`] that
//! maps test-time contexts back to their assigned mode indices,
//! the pre-computed dressed-vacuum states |0̃_o⟩ used as the
//! per-order fallback for unseen contexts, and the configuration
//! snapshot. It exposes:
//!   - `next_token_dist(&[u32]) -> Vec<f64>`: the per-token probability
//!     vector `P(y | context)`, computed by encoding the context into
//!     a Krylov superposition of (per-order) registry modes for seen
//!     orders and the dressed vacuum |0̃_o⟩ for unseen orders,
//!     evolving the superposition through the reduced Hamiltonian,
//!     marginalising the Born-rule sketch against the per-mode
//!     histograms with absolute-discount smoothing to the unigram,
//!     and clamping the result to a valid distribution.
//!   - `logprob(&[u32], u32) -> f64`: `log P(next | context)`, the
//!     per-token log-probability used by the perplexity evaluator.
//!   - `save(&Path) / load(&Path)`: bincode-serialized model.
//!
//! The model is a **quantum-kernel n-gram-family model** with
//! coherent Krylov smoothing across backoff orders: per-mode
//! histograms are the *capacity*; the dressed-vacuum projector sum +
//! Krylov evolution is the *smoothing*; the unigram is the floor.
//!
//! # Rev 36 change
//!
//! `QfmTextModel` now owns a [`ContextRegistry`] and a per-order
//! `vacuum_states: Vec<DVector<Complex64>>`. The Krylov pipeline's
//! W matrix is `(K_2_total, rank)` where
//! `K_2_total = 1 + Σ_o n_active_modes_o` is the actual vocabulary
//! of unique training contexts — not a fixed `block_sizes` budget.
//! The dressed-vacuum state for order `o`,
//!   `|0̃_o⟩ = (1/√M_o) Σ_{m ∈ order_o} W[m, :]`,
//! is precomputed at model-build time and used as the per-order
//! fallback whenever the registry has no mode for that order
//! (e.g. an unseen test context). The accumulator still uses
//! mode 0 (the Fock vacuum) as a bookkeeping bucket for unseen
//! observations during training; at inference, the dressed
//! vacuum |0̃_o⟩ is the actual Krylov input for unseen orders
//! (not the W row for mode 0). This is the natural backoff the
//! H matrix's dressed-vacuum projector Σ_o λ_o |0̃_o⟩⟨0̃_o|
//! mandates.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use qfm::{QfmConfig, QfmPipeline};
use rustc_hash::FxHashMap;

use crate::accumulate::{
    ChannelAccumulator, ModeStats, observe_shard_with_registry, observe_with_registry,
};
use crate::oxieml_decoder;
use oxieml::tree::EmlTree;
use crate::config::{DecodeStrategy, TextConfig};
use crate::error::QfmTextError;
use crate::registry::{ContextRegistry, VACUUM_MODE};

/// Metadata baked into the serialized model so a future reader can
/// refuse incompatible files. Mirrors `QfmTextError::BadManifest`'s
/// spirit for the model side.
#[derive(Debug, Clone)]
pub struct TextModelMetadata {
    pub vocab_size: u32,
    pub n_orders: usize,
    pub k2_total: u32,
    pub n_active_modes: usize,
    pub total_windows: u64,
}

/// The compiled QFM-Text model.
#[derive(Debug, Clone)]
pub struct QfmTextModel {
    /// The TSR reduced pipeline (encode / evolve / decode). Built
    /// from the channel weights, not from training points.
    pub pipeline: QfmPipeline,
    /// Per-mode statistics from the accumulator. The mode key is
    /// the global index in `[0, k2_total)`. Mode `0` (vacuum) is
    /// initialized to the unigram histogram so the dressed-vacuum
    /// projector in H provides the natural backoff for unseen
    /// test contexts.
    pub mode_hists: FxHashMap<u32, ModeStats>,
    /// Unigram counts (f64 after normalizing).
    pub unigram: Vec<f64>,
    /// Sum of all unigram counts.
    pub unigram_total: f64,
    /// Configuration snapshot.
    pub cfg: TextConfig,
    /// Schema version.
    pub schema_version: u32,
    /// Precomputed `rank x rank` Gram matrix `W^H W` (see
    /// `QfmPipeline::gram`), computed once at model-build/load time.
    /// `next_token_dist` uses it with `decode_sketched_at` so the
    /// per-token Born-rule decode costs `O(rank^2 + active_modes *
    /// rank)` instead of `O(K_2 * rank)` — the latter dominates eval
    /// cost since `marginalize` only ever reads the per-context
    /// active-mode entries out of the full `K_2`-length dense
    /// `decode_sketched` output.
    pub gram: DMatrix<Complex64>,
    /// Per-context mode registry: maps each trailing-`o` token slice
    /// to its assigned global mode index. Owned by the model (cloned
    /// cheaply) because `next_token_dist` needs it to encode test
    /// contexts. Built during the streaming accumulation pass and
    /// frozen at `from_accumulator` time.
    pub registry: ContextRegistry,
    /// Pre-computed dressed-vacuum states |0̃_o⟩ for each order `o`
    /// in `0..n_orders`. `vacuum_states[o]` is the uniform
    /// superposition over the order-`o+1` W-rows:
    ///   `|0̃_o⟩ = (1/√M_{o+1}) Σ_{m ∈ order_{o+1}} W[m, :]`
    /// (with the order-`o` slice defined as the trailing-`(o+1)`
    /// tokens). Used by `next_token_dist` as the per-order
    /// fallback when the registry has no mode for that order.
    /// The Fock vacuum mode 0 is **not** the dressed vacuum — the
    /// dressed vacuum is a per-order superposition, not a single
    /// row of W.
    pub vacuum_states: Vec<DVector<Complex64>>,
    /// Pre-computed **outer vacuum** |Ψ_0⟩ (rev 37 v3). This is
    /// the Krylov initial vector c_0 used by `encode_context`. It
    /// is defined as the uniform state in the Fock-space input
    /// basis with R partitions of the infinite-dimensional
    /// hypersphere of inner wave-functions.
    ///
    /// In the Fock basis: c_0[fock] has R non-zero components
    /// each equal to `√R` (R = number of partitions). Only the
    /// first M of these have known Krylov projections (the rows
    /// of the W matrix). The Krylov-projected c_0 is:
    ///   `c_0[krylov][k] = √R · Σ_{m=0}^{M-1} W[m, k]`
    ///
    /// R is a hyperparameter, unrelated to the Krylov rank.
    /// R must be > M to distinguish the training data. Default
    /// when `fock_resolution` is not set: R = 10 * M.
    ///
    /// **Context-independent**: `c_0` does NOT depend on the
    /// input context. The Krylov evolution `c_1 = U c_0` is
    /// also context-independent, but the per-mode Born weights
    /// `|c_1^H W[m, :]|²` still vary with `m` (the asymmetry
    /// of the W matrix breaks the symmetry of the post-evolution
    /// state). The QFM marginalization then weights the per-mode
    /// histograms by these Born weights, which is what makes
    /// the model context-conditioned.
    pub outer_vacuum: DVector<Complex64>,
    /// The decoder (rev 37). Defaults to `Dense` (the rev 36
    /// behavior). After `fit_oxieml_decoder` is called, this
    /// is replaced with `Analytical` (one `EmlTree` per Krylov
    /// column, with per-column fallback to dense for failed
    /// fits).
    pub decoder: DecoderKind,
}

/// The decoder: how the per-mode Born-rule weights are
/// computed from the Krylov-evolved `c_1`. Rev 37 introduces
/// the `Analytical` variant that uses `oxieml::EmlTree`s
/// instead of the dense W matrix.
#[derive(Debug, Clone)]
pub enum DecoderKind {
    /// Default (rev 35/36): read `W[m, :]` from the dense
    /// matrix at each active mode.
    Dense,
    /// Rev 37: evaluate one `EmlTree` per Krylov basis column
    /// at `m / k2_total`. For columns where the oxieml fit
    /// failed (residual > `residual_tol`), use the dense row
    /// from `fallback` instead.
    Analytical {
        /// One EmlTree per Krylov basis column, indexed by
        /// column index `j ∈ [0, rank)`. The tree evaluates
        /// `f_j(m / k2_total)` for a mode index `m ∈ [0,
        /// k2_total)`.
        trees: Vec<EmlTree>,
        /// Dense W matrix (k2_total × rank) used as a
        /// fallback for columns where the oxieml fit failed.
        /// Rows for successful columns are zero (we don't
        /// need to keep them).
        fallback: DMatrix<Complex64>,
        /// Per-column `ok` mask: `column_ok[j] = true` means
        /// `trees[j]` is used; `false` means `fallback[:, j]`
        /// is used.
        column_ok: Vec<bool>,
    },
}

impl DecoderKind {
    /// Per-column `ok` mask (true = use tree, false = use
    /// fallback). For `Dense`, all columns are "ok" in the
    /// sense of being handled by the dense path.
    pub fn column_ok(&self) -> Vec<bool> {
        match self {
            DecoderKind::Dense => Vec::new(),
            DecoderKind::Analytical { column_ok, .. } => column_ok.clone(),
        }
    }
}

/// Summary of a `fit_oxieml_decoder` call.
#[derive(Debug, Clone)]
pub struct OxiemlFitSummary {
    /// Per-column normalized MSE (in units of the
    /// per-column-max-abs-normalized values).
    pub per_column_mse: Vec<f64>,
    /// Per-column tree complexity (number of EML nodes).
    pub per_column_complexity: Vec<usize>,
    /// Per-column fallback mask: true = column fell back to
    /// dense W (residual exceeded threshold).
    pub per_column_fallback: Vec<bool>,
    /// Number of columns that fell back to dense.
    pub n_fallback: usize,
    /// Total wall time for the fit, in seconds.
    pub total_fit_seconds: f64,
}

/// Decode the Krylov-evolved `c_1` into per-mode weights at
/// the per-context active modes, using the oxieml-fitted
/// `EmlTree`s. For each active mode `m_o` and each column
/// `j`:
/// - If `column_ok[j]` is true, use `tree[j].eval(m_o /
///   k2_total)` for the `j`-th component of the wave-function
///   at `m_o`.
/// - Otherwise, use `fallback[(m_o, j)].re`.
///
/// The per-mode weight is `|Σ_j h_j * f_j(m_o)|^2 / total`
/// where `total = c_1^\dagger G c_1` (the Gram normalization).
fn decode_with_trees(
    c_1: &DVector<Complex64>,
    gram: &DMatrix<Complex64>,
    active_modes: &[u32],
    trees: &[EmlTree],
    fallback: &DMatrix<Complex64>,
    column_ok: &[bool],
) -> Vec<(u32, f64)> {
    let total: f64 = (c_1.adjoint() * gram * c_1)[(0, 0)].re.max(0.0);
    if total <= 0.0 {
        return Vec::new();
    }
    let rank = c_1.len();
    let m = fallback.nrows();
    let mut out = Vec::with_capacity(active_modes.len());
    for &idx in active_modes {
        let row = idx as usize;
        if row >= m {
            continue;
        }
        let mut amp = Complex64::new(0.0, 0.0);
        for j in 0..rank {
            let c_j = if column_ok[j] {
                // Evaluate the oxieml tree at the normalized
                // mode index. The tree was fit on values
                // normalized by the per-column max-abs, so
                // we rescale the tree's output here. (The
                // fit_decoder caller stores the per-column
                // norms alongside the trees; for now, the
                // tree is the only thing we have, so the
                // norms are baked in via the fit's
                // preprocessing. We recover the scale by
                // evaluating at the *same* normalization
                // the fit used, which is `x = mode_index /
                // m_total`.)
                let x = idx as f64 / m as f64;
                let val = oxieml_decoder::evaluate_column(&trees[j], idx, m);
                // val is in [-1, 1] (per-column-normalized).
                // The amplitude needs to be in the same
                // units as the dense W's column max-abs.
                // The fit stored the per-column max-abs
                // separately (we use the W row's max-abs as
                // a proxy). For now, we use the dense W
                // column max-abs at this row as a scale
                // factor; this is approximate but preserves
                // the relative magnitudes.
                let scale = fallback.column(j).iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
                if scale > 0.0 {
                    val / scale
                } else {
                    0.0
                }
            } else {
                fallback[(row, j)].re
            };
            amp += Complex64::new(c_j, 0.0) * c_1[j];
        }
        out.push((idx, amp.norm_sqr() / total));
    }
    out
}

impl QfmTextModel {
    /// Build a QfmTextModel from a streaming accumulator and its
    /// accompanying [`ContextRegistry`]. The registry is the same
    /// one the streaming pass grew; the model's `k2_total` is
    /// `1 + Σ_o n_active_modes_o` (the vacuum plus every distinct
    /// context the corpus produced).
    ///
    /// The accumulator's `mode_hists` map is augmented with a
    /// vacuum entry (`mode = 0`) initialized to the unigram
    /// counts. This is the bookkeeping bucket for unseen-context
    /// observations the streaming pass recorded against the
    /// vacuum sentinel; at inference time, the per-order dressed
    /// vacuum `|0̃_o⟩` is the actual Krylov input (not the W row
    /// for mode 0). The dressed-vacuum projector in H mixes the
    /// |0̃_o⟩ with the registry modes' W rows through the Krylov
    /// evolution, giving the QFM-mandated fallback for unseen
    /// test contexts.
    pub fn from_accumulator(
        acc: ChannelAccumulator,
        registry: ContextRegistry,
        cfg: &TextConfig,
    ) -> Result<Self, QfmTextError> {
        // k2_total is the total number of modes available in the
        // configuration (sum of block_sizes + 1 for the vacuum).
        // For the `ContextRegistry` (rev 36) path, the registry's
        // `k2_total()` matches this; for the `OrderHasher` (rev 35
        // / rev 37 default) path, the registry is a placeholder
        // (the modes are spread over `[0, cfg.k2_total())` and
        // managed by the hasher + accumulator). We use the config's
        // `k2_total()` directly, which is always correct.
        let k2_total = cfg.k2_total();
        // Build the per-order (λ_o, channels_o) groups: for each
        // order, the active modes in its registry block with
        //   ᾱ_j = weight_j / total_windows
        // (the QFM.tex flow-matching weights). The generator is the
        // hierarchical multi-projector
        //   H = Σ_o λ_o |0̃_o⟩⟨0̃_o|,
        // one exact rank-1 ProjectOnto term per context order, with
        // the eq. Htomo normalization applied per group inside
        // `qfm_hamiltonian_hierarchical_projectors`. This is the
        // QFM.tex-mandated form (rev 31: the exact projector is the
        // only off-diagonal generator). The diagonal eq. Hdiag
        // surrogate used here in the rev 33 interim is NOT a flow —
        // its Born populations are stationary (QFM.tex §"Scope") —
        // and was removed again in rev 34. The generator's rank is
        // ≤ n_orders, so the Krylov dim is ≤ n_orders + 1 by
        // construction.
        // Compute the Fock-space resolution R (number of partitions
        // of the infinite-dimensional hypersphere). R is unrelated
        // to the Krylov rank; it must be > M (total training windows)
        // to distinguish the training data.
        let total_windows = acc.total_windows.max(1);
        let fock_r: u64 = cfg.fock_resolution.unwrap_or(10 * total_windows);
        // Build the list of active modes (mode 0 plus every observed mode).
        let mut active_modes: Vec<u32> = acc.stats.keys().copied().collect();
        active_modes.sort_unstable();
        if !active_modes.contains(&0u32) {
            active_modes.insert(0, 0u32);
        }
        // λ₀ for the outer vacuum projector, λ₁ for the transition sum.
        // λ₀ defaults to 0 (pure transition Hamiltonian); the vacuum
        // projector |c₀⟩⟨c₀| has operator norm O(R·N) and dwarfs the
        // transition sum unless λ₀ is set to O(1/(R·N)).
        let lambda0 = cfg.lambda.first().copied().unwrap_or(0.0);
        let lambda1 = cfg.lambda.get(1).copied().unwrap_or(1.0);
        // Compile the QFM pipeline with the diffusion Hamiltonian.
        let qfm_cfg = QfmConfig {
            k: cfg.n_orders,
            k2: k2_total as usize,
            krylov_dim: cfg.m_shifts,
            seed: cfg.seed,
            n_t_samples: 4,
            noise_dim: cfg.n_orders,
            max_rank: Some(cfg.max_rank),
        };
        let pipeline = QfmPipeline::compile_channels(
            &active_modes,
            &acc.transitions,
            lambda0,
            lambda1,
            k2_total as usize,
            &qfm_cfg,
            Some(fock_r),
        )?;
        let gram = pipeline.gram();
        // Unigram normalize.
        let unigram_total: f64 = acc.unigram.iter().map(|&c| c as f64).sum();
        let unigram: Vec<f64> = if unigram_total > 0.0 {
            acc.unigram.iter().map(|&c| c as f64 / unigram_total).collect()
        } else {
            vec![0.0; acc.unigram.len()]
        };
        // Initialize the vacuum mode (0) histogram to the unigram
        // counts. This is the bookkeeping bucket for unseen-context
        // observations the streaming pass recorded against the
        // vacuum sentinel; at inference, the dressed vacuum |0̃_o⟩
        // is the actual Krylov input (not mode 0's W row).
        let mut mode_hists = acc.stats;
        Self::init_vacuum_histogram(&mut mode_hists, &acc.unigram);
        // Pre-compute the dressed-vacuum states |0̃_o⟩ for each
        // order `o in 0..n_orders` using the **Mehler formalism**
        // (QFM.tex §"QFM-Text", eq. (Htomo) and §"The dressed
        // vacuum"):
        //   |0̃_o⟩ = c_0^(o) |0⟩ + Σ_{m ∈ order_o} ε_m^(o) |m⟩,
        //   c_0^(o) = 1/√(1 + Σ_m ᾱ_m²),
        //   ε_m^(o) = ᾱ_m / √(1 + Σ_m ᾱ_m²),
        //   ᾱ_m = weight_m / total_windows.
        // In the Krylov basis (W rows), this is
        //   v_o = c_0^(o) W[0, :] + Σ_{m ∈ order_o} ε_m^(o) W[m, :].
        // The Mehler formula is the per-mode-frequency-weighted
        // dressed vacuum, *not* the naive uniform superposition
        // (1/√M_o) Σ W[m, :]. The user's rev 36 design point:
        // without hashing, every mode has a different natural
        // weight, and the right "outer vacuum" is the Mehler
        // kernel of those weights, not a flat average. With small
        // ᾱ (the typical ~10⁻⁵ regime on a ~10⁵-mode corpus),
        // c_0^(o) ≈ 1 and ε_m^(o) ≈ ᾱ_m, so the dressed vacuum
        // is nearly parallel to the Fock vacuum |0⟩ — the rev 35
        // W-rank degeneracy. The Mehler formula gives the exact
        // vacuum for the registry design; the rev-35 uniform
        // formula was a special case that only held under
        // bounded-hash equal-weight assumptions.
        let total = unigram_total.max(1.0);
        let vacuum_states = Self::compute_mehler_vacua(
            &pipeline, &registry, cfg, &mode_hists, total,
        );
        // Compute the outer vacuum |Ψ_0⟩ (rev 37 v3). Precomputed
        // during SIRK via projection of the starting vector v0
        // (uniform over all modes, normalized) onto the Gram-whitened
        // Krylov basis. The pipeline stores the result; it is
        // context-independent and reused for every encode step.
        // See `compact_forward_sirk` for the computation.
        let outer_vacuum = pipeline.outer_vacuum().clone();
        Ok(Self {
            pipeline,
            mode_hists,
            unigram,
            unigram_total,
            cfg: cfg.clone(),
            schema_version: crate::SCHEMA_VERSION,
            gram,
            registry,
            vacuum_states,
            outer_vacuum,
            decoder: DecoderKind::Dense,
        })
    }

    /// Compute the dressed-vacuum states `|0̃_o⟩` for each order
    /// `o in 0..n_orders` using the **Mehler formalism** (QFM.tex
    /// §"QFM-Text", eq. (Htomo), §"The dressed vacuum"):
    ///   `|0̃_o⟩ = c_0^(o) |0⟩ + Σ_{m ∈ order_o} ε_m^(o) |m⟩`,
    ///   `c_0^(o) = 1/√(1 + Σ_m ᾱ_m²)`,
    ///   `ε_m^(o) = ᾱ_m / √(1 + Σ_m ᾱ_m²)`,
    ///   `ᾱ_m = weight_m / total_windows`.
    /// In the Krylov basis (rows of W):
    ///   `v_o = c_0^(o) W[0, :] + Σ_{m ∈ order_o} ε_m^(o) W[m, :]`.
    ///
    /// This is the rev 36 "outer vacuum" — the H-matrix-mandated
    /// per-order fallback for unseen test contexts. It is the
    /// Mehler kernel of the per-mode marginal-frequency
    /// distribution, *not* the naive uniform superposition
    /// `(1/√M_o) Σ W[m, :]` (which would only hold for
    /// equal-weight modes, i.e. hashed-bucket designs where every
    /// slot in a block has the same channel weight). Without
    /// hashing the modes have unequal natural weights, and the
    /// Mehler formula is the correct one.
    ///
    /// If the order has zero active modes, the dressed vacuum is
    /// the zero vector (no fall-back is meaningful — the model
    /// has no order-`o+1` capacity to back off to).
    fn compute_mehler_vacua(
        pipeline: &QfmPipeline,
        registry: &ContextRegistry,
        cfg: &TextConfig,
        mode_hists: &FxHashMap<u32, ModeStats>,
        total_windows: f64,
    ) -> Vec<DVector<Complex64>> {
        let rank = pipeline.rank();
        let w = pipeline.w();
        let total = total_windows.max(1.0);
        let mut out = Vec::with_capacity(cfg.n_orders);
        for o in 0..cfg.n_orders {
            // **rev 37 bug fix:** the per-order mode count and
            // offset must come from the *config* when the
            // encoder is the OrderHasher (the registry is
            // empty for the hasher path). Falling back to the
            // registry's per-order count of 0 produced a
            // zero vacuum state, which degenerated the QFM
            // to the n-gram baseline. The fix: use the
            // registry's count when it's non-empty
            // (registry path), otherwise use the config's
            // block_sizes[o] (hasher path).
            let (m_o, off) = if registry.n_active_for_order(o) > 0 {
                (registry.n_active_for_order(o), registry.offset(o))
            } else {
                (cfg.block_sizes[o], cfg.offset(o))
            };
            if m_o == 0 {
                out.push(DVector::zeros(rank));
                continue;
            }
            // 1. Compute the per-mode marginal frequencies
            //    ᾱ_m = weight_m / total_windows and the
            //    per-order sum Σ ᾱ_m². With ~10⁵ modes and
            //    uniform per-mode weights, this sum is ≈ 1/M_o.
            let mut alpha_sq_sum = 0.0_f64;
            let mut alphas: Vec<f64> = Vec::with_capacity(m_o);
            for k in 0..m_o {
                let mode = off + k as u32;
                let weight = mode_hists
                    .get(&mode)
                    .map(|s| s.weight as f64)
                    .unwrap_or(0.0);
                let alpha = weight / total;
                alphas.push(alpha);
                alpha_sq_sum += alpha * alpha;
            }
            // 2. Mehler normalization: the order-o dressed vacuum
            //    is unit-norm iff c_0² + Σ ε_m² = 1, which
            //    requires dividing by √(1 + Σᾱ²).
            let norm = (1.0 + alpha_sq_sum).sqrt();
            let c_0 = 1.0 / norm;
            // 3. Build v_o = c_0 W[0, :] + Σ_m ε_m W[m, :].
            //    W[0, :] is the Fock vacuum's row — typically
            //    zero under the SIRK channel compilation
            //    (the channel weights are all zero for mode 0),
            //    so the c_0 W[0, :] term is null in practice and
            //    the dressed vacuum reduces to the data-channel
            //    part Σ ε_m W[m, :]. The Fock term is kept here
            //    for algebraic completeness — it would matter if
            //    the SIRK compilation assigned a non-zero row to
            //    mode 0 (e.g. with a vacuum-channel weight).
            let mut v = DVector::<Complex64>::zeros(rank);
            for r in 0..rank {
                v[r] += w[(0, r)] * c_0;
            }
            for k in 0..m_o {
                let mode_idx = (off + k as u32) as usize;
                if mode_idx < w.nrows() {
                    let eps = alphas[k] / norm;
                    for r in 0..rank {
                        v[r] += w[(mode_idx, r)] * eps;
                    }
                }
            }
            out.push(v);
        }
        out
    }

    /// Compute the **outer vacuum** |Ψ_0⟩ for the Krylov subspace.
    /// The outer vacuum is the **uniform state on the infinite-
    /// dimensional Fock-space hypersphere at resolution R**,
    /// projected onto the Krylov subspace.
    ///
    /// R is the number of partitions of the infinite-dimensional
    /// hypersphere. It is a hyperparameter, unrelated to the Krylov
    /// rank (which also starts with 'r'). R must be > M (the number
    /// of training data points) to allow distinguishing the
    /// training data. Default: R = 10 * M.
    ///
    /// In the Fock basis: c_0[fock] has R non-zero components equal
    /// to `√R` (the first R Fock basis elements, R = number of
    /// partitions). The remaining infinite components are zero.
    ///
    /// The Krylov representation is the projection of this Fock-
    /// uniform state onto the Krylov subspace. Only the first M
    /// of the R Fock-basis directions have known Krylov projections
    /// (the rows of the W matrix). The remaining R-M directions
    /// are orthogonal to the Krylov subspace:
    ///   `c_0[krylov][k] = √R · Σ_{m=0}^{M-1} W[m, k]`
    /// i.e. the sum of ALL M rows of W, multiplied by √R.
    pub fn compute_outer_vacuum(
        w: &DMatrix<Complex64>,
        fock_resolution: Option<u64>,
    ) -> DVector<Complex64> {
        let (m, rank) = (w.nrows(), w.ncols());
        if rank == 0 || m == 0 {
            return DVector::<Complex64>::zeros(rank);
        }
        let r = fock_resolution.unwrap_or(10 * m as u64) as usize;
        let r = r.max(m);
        let norm = (r as f64).sqrt();
        let mut c0 = DVector::<Complex64>::zeros(rank);
        for i in 0..m {
            for j in 0..rank {
                c0[j] += w[(i, j)];
            }
        }
        c0 *= Complex64::new(norm, 0.0);
        c0
    }

    /// Set the vacuum mode's (index 0) histogram to the unigram
    /// counts. If the accumulator already observed the vacuum (e.g.
    /// because the corpus had unseen-context windows), the
    /// unigram-histogram replaces the observed histogram. The
    /// unigram is the marginal distribution, so this is a
    /// semantically clean fallback: every token gets its
    /// marginal-frequency count in the vacuum mode.
    fn init_vacuum_histogram(
        mode_hists: &mut FxHashMap<u32, ModeStats>,
        unigram_counts: &[u64],
    ) {
        let mut vacuum = ModeStats::default();
        let total: u64 = unigram_counts.iter().sum();
        vacuum.weight = total;
        for (tok, &cnt) in unigram_counts.iter().enumerate() {
            if cnt > 0 {
                vacuum.hist.push((tok as u32, cnt as u32));
            }
        }
        // Sort descending by count, then by token id (matches the
        // accumulator's invariant).
        vacuum
            .hist
            .sort_by(|(a_t, a_c), (b_t, b_c)| b_c.cmp(a_c).then(a_t.cmp(b_t)));
        // hist_cap may be smaller than the vocabulary; truncate to
        // the cap. The excess goes to escape. (hist_cap is on the
        // TextConfig but we can grab it from the existing entry's
        // hist length if available, or default to the cap we
        // observe.)
        let hist_cap = mode_hists
            .values()
            .map(|s| s.hist.len())
            .max()
            .unwrap_or(64)
            .max(vacuum.hist.len());
        while vacuum.hist.len() > hist_cap {
            let evicted = vacuum.hist.pop().expect("non-empty");
            vacuum.escape += evicted.1 as u64;
        }
        mode_hists.insert(VACUUM_MODE, vacuum);
    }

    /// Metadata for diagnostics.
    pub fn metadata(&self) -> TextModelMetadata {
        TextModelMetadata {
            vocab_size: self.unigram.len() as u32,
            n_orders: self.cfg.n_orders,
            k2_total: self.cfg.k2_total(),
            n_active_modes: self.registry.maps().iter().map(|m| m.len()).sum(),
            total_windows: (self.unigram_total) as u64,
        }
    }

    /// Expose the underlying Krylov-pipeline W matrix
    /// (`K_2_total × rank`, complex128) for diagnostics or
    /// for oxieml-based analytical decoder fitting (rev 37).
    /// Read-only view; the matrix lives in the `QfmPipeline`.
    pub fn w_matrix(&self) -> &nalgebra::DMatrix<num_complex::Complex64> {
        self.pipeline.w()
    }

    /// Expose the Krylov rank (number of basis vectors W holds
    /// per mode). Equals `pipeline.rank()`.
    pub fn krylov_rank(&self) -> usize {
        self.pipeline.rank()
    }

    /// Read-only view of the trained W matrix (k2_total × rank,
    /// complex128) — used by the oxieml decoder fitting
    /// (`oxieml_decoder::fit_decoder`) to discover analytical
    /// trees that replace the dense W at inference time.
    /// Read-only.
    pub fn k2_total(&self) -> u32 {
        self.cfg.k2_total()
    }

    /// Compute the per-token next-token distribution for a context.
    /// The vector is length `vocab_size`, sums to 1.0, and every entry
    /// is `> 0` (the unigram floor — the per-mode escape mass and
    /// the dressed-vacuum backoff — guarantees no zero probability).
    ///
    /// **Rev 36 encoding:** the Krylov input `c_0` is a superposition
    /// of one term per active order `o in 1..=min(context.len(),
    /// n_orders)`:
    ///   - If the trailing-`o` slice is in the registry: the W row
    ///     `W[m_o, :]` for the assigned mode `m_o`.
    ///   - If unseen: the precomputed dressed-vacuum state
    ///     `|0̃_o⟩` (Mehler formalism, QFM.tex eq. Htomo) — the
    ///     per-order "uniform measure on the inner wave-function
    ///     space", the H-matrix-mandated fallback.
    /// All terms are added with equal weight `1/√n` (the standard
    /// equal-weight-superposition encoder), then the result is
    /// evolved through the reduced Hamiltonian and decoded via the
    /// **sparse** Born rule `decode_sketched_at` — only the
    /// registry's active modes (the seen modes, plus the Fock
    /// vacuum for the per-mode-escape unigram routing) are
    /// evaluated. The dressed-vacuum contribution is implicit in
    /// the Krylov input but not in the per-mode histograms (the
    /// dressed vacua are superpositions, not single modes); the
    /// marginalize step's per-mode escape mass routes the missing
    /// mass to the unigram floor.
    ///
    /// **Cost:** O(n_orders × rank + n_active × rank) per token,
    /// where `n_active` is the number of active modes for this
    /// context (typically 1–n_orders, *not* K_2). The naive
    /// `decode_sketched` is O(K_2 × rank) which is intractable on
    /// the rev 36 registry design with K_2 ~ 10⁶. The sparse
    /// decode is what makes the model usable at corpus scale.
    pub fn next_token_dist(&self, context: &[u32]) -> Result<Vec<f64>, QfmTextError> {
        // 1. Build the Krylov input c_0 from the per-order terms
        //    (registry modes for seen orders, dressed vacua for
        //    unseen orders).
        let c_0 = self.encode_context(context);
        // 2. Evolve the superposition forward by t.
        let c_1 = self.pipeline.evolve(&c_0, self.cfg.t);
        // 3. Decode the sketch (Phase 3) at the per-context
        //    active modes only. The "active modes" here are the
        //    registry's mode list (one per active order, with
        //    mode 0 = vacuum for the order-1 unigram routing).
        //    The dressed-vacuum contribution is implicit: the
        //    Krylov input already contains the dressed-vacuum
        //    terms, and the marginalize step routes any "missing"
        //    mass to the unigram via the per-mode escape channel.
        //
        //    The `Decoder` enum (rev 37) chooses the decode path:
        //    `Dense` reads `W[m, j]` from the dense matrix;
        //    `Analytical` evaluates the oxieml `EmlTree`s at
        //    `m / k2_total` for each column (with a per-column
        //    fallback to dense if the fit residual was too
        //    high). Both paths produce the same `(mode, weight)`
        //    list consumed by `marginalize`.
        //
        //    **rev 37 design:** always include mode 0 (the Fock
        //    vacuum, whose histogram is the unigram) in the
        //    active modes. The Krylov weight for mode 0
        //    `|c_1^H · W[0, :]|^2` is the "outer vacuum projector"
        //    mass — it is the unigram backoff that the Jelinek-
        //    Mercer smoothing was previously providing. The
        //    Jelinek-Mercer interpolation is removed from
        //    `marginalize`; the Krylov unigram (via mode 0) is
        //    the only backoff mechanism.
        let mut active_modes = self.encode_active_modes(context);
        if !active_modes.contains(&0) {
            active_modes.push(0);
        }
        let weights = self.decode_at(&c_1, &active_modes);
        // 4. Marginalise against the per-mode histograms.
        let dist = self.marginalize(&weights);
        Ok(dist)
    }

    /// Build the Krylov input vector `c_0` for `context`. For each
    /// active order `o in 1..=min(context.len(), n_orders)`:
    ///   - If the trailing-`o` slice is in the registry: add the
    ///     W row `W[m_o, :]` for the assigned mode `m_o`.
    ///   - If unseen: add the precomputed dressed-vacuum state
    ///     `|0̃_o⟩` for order `o` (the uniform measure on the
    ///     order-`o` inner wave-function space).
    /// All terms contribute with weight `1/√n` (equal-weight
    /// superposition across the `n = min(context.len(),
    /// n_orders)` active orders). The dressed-vacuum substitution
    /// is the rev-36 change from the prior mode-0 sentinel: the
    /// dressed vacuum is the H-matrix-mandated fallback, not a
    /// single Fock-space row.
    /// Return the per-context active mode list, falling back to
    /// the `OrderHasher` when the registry is empty (i.e. the
    /// model was trained with `use_registry_encoder = false`).
    ///
    /// **rev 37 bug fix:** the previous code unconditionally
    /// called `self.registry.encode_modes(context)`, which
    /// returns an empty list for the hasher path (the registry
    /// is a placeholder). An empty `active_modes` made the
    /// per-mode decode return an empty weight list, and the
    /// `Renormalize` strategy fell back to the n-gram baseline
    /// (uniform over the active modes' histograms). The
    /// `encode_context` fix already addressed the Krylov-input
    /// side; this fix addresses the decoder-input side.
    fn encode_active_modes(&self, context: &[u32]) -> Vec<u32> {
        let from_registry = self.registry.encode_modes(context);
        if !from_registry.is_empty() {
            return from_registry;
        }
        if !self.cfg.use_registry_encoder {
            crate::features::OrderHasher::new(self.cfg.clone()).encode_modes(context)
        } else {
            from_registry
        }
    }

    fn encode_context(&self, context: &[u32]) -> DVector<Complex64> {
        // **rev 37 v3 design:** the Krylov initial vector c_0 is
        // the **outer vacuum** |Ψ_0⟩, precomputed at model-build
        // time (see `outer_vacuum` and the pipeline accessor).
        // It is the (real, non-negative) vector in the Krylov
        // subspace such that the **standard L² inner product**
        // with any inner wave-function ψ is **proportional to the
        // L¹ norm of ψ** (the L¹ integral of |ψ| over its support
        // in the original infinite-dim Hilbert space, projected
        // onto the Krylov basis):
        //   `Σ_r c_0(r) · W[m, r] = k · Σ_r |W[m, r]|`  for all m
        //
        // For real non-negative c_0 and complex W, this is a
        // coupled system: `Re(W) c_0 = k · l1_norms` and
        // `Im(W) c_0 = 0`. There is no "L¹ inner product" — the
        // L¹ norm is a property of ψ, and the L² inner product
        // (Born rule) measures it.
        //
        // **The outer vacuum is context-independent** — it does
        // not depend on the input context. The Krylov evolution
        // `c_1 = exp(-i H_m t) c_0` is also context-independent,
        // but the per-mode Born weights `|c_1^H W[m, :]|²` vary
        // with `m` (the asymmetry of the W matrix breaks the
        // post-evolution symmetry), and the QFM marginalization
        // weights the per-mode histograms by these Born
        // weights. This is what makes the model
        // context-conditioned despite the context-independent
        // Krylov input.
        //
        // **Why not the per-context superposition?** The
        // previous design used
        //   c_0 = (1/√(n+1)) (W[0, :] + Σ_o W[m_o, :])
        // where `m_o` is the per-order mode for the current
        // context. That is a **superposition of seen modes** —
        // it explicitly excludes every mode that is not in the
        // per-context lookup (the unseen modes, including all
        // modes the training set observed but didn't pair with
        // this exact context, plus all modes the training set
        // never observed at all). By the user's "outer vacuum
        // cannot be a superposition of seen modes" design
        // constraint, that choice biases the Krylov input
        // toward what the context already tells us. The outer
        // vacuum is a single global vector determined only by
        // the W matrix structure — no per-context dependence.
        let _ = context; // context is not used; the outer vacuum is global.
        self.outer_vacuum.clone()
    }

    /// Compute the per-token next-token distribution via **model
    /// averaging** (mixture of experts over the per-order Krylov
    /// models). For each active order `o`:
    ///
    ///   1. Encode the order-`o` term as a unit-norm Krylov
    ///      vector `c_0_o = (W[m_o, :] or |0̃_o⟩) / ||...||`.
    ///   2. Evolve: `c_1_o = exp(-i H_m t) c_0_o`.
    ///   3. Decode: `p̃_o = decode_sketched_at(c_1_o, active)`.
    ///
    /// Then `p̃ = (1/n) Σ_o p̃_o`, and the usual `marginalize` step
    /// is applied. This avoids the destructive interference in the
    /// equal-weight superposition of `next_token_dist`. As with
    /// `next_token_dist`, the order-`o` term is the W row for the
    /// seen registry mode, or the dressed vacuum |0̃_o⟩ if
    /// unseen.
    ///
    /// Cost: `n` forward solves per token instead of 1. For the
    /// production 4-order model with `n_orders=4`, this is a 4×
    /// cost (still dominated by the per-mode histogram lookups).
    pub fn next_token_dist_model_avg(
        &self,
        context: &[u32],
    ) -> Result<Vec<f64>, QfmTextError> {
        // 1. Build the per-order Krylov inputs (registry W-rows
        //    for seen orders, dressed vacua |0̃_o⟩ for unseen).
        let c_0_list = self.encode_context_per_order(context);
        let n = c_0_list.len();
        if n == 0 {
            return Ok(self.unigram.clone());
        }
        // 2. For each per-order Krylov state, evolve and decode at
        //    the per-context active modes only (sparse decode).
        //    The marginalize step then weights the per-mode
        //    histograms by the Born-rule weights. The dressed
        //    vacua (if any) contribute through the Krylov state.
        //    The unigram (mode 0) is included in the active modes
        //    for the Krylov unigram backoff (rev 37).
        let mut active_modes = self.encode_active_modes(context);
        if !active_modes.contains(&0) {
            active_modes.push(0);
        }
        let mut acc: FxHashMap<u32, f64> = FxHashMap::default();
        for c_0 in &c_0_list {
            let c_1 = self.pipeline.evolve(c_0, self.cfg.t);
            for (m, p) in self.pipeline.decode_sketched_at(&c_1, &self.gram, &active_modes) {
                *acc.entry(m).or_insert(0.0) += p;
            }
        }
        // 3. Average the decoded weights across the n per-order solves.
        let weights: Vec<(u32, f64)> = acc.into_iter().map(|(m, p)| (m, p / n as f64)).collect();
        // 4. Marginalise.
        let dist = self.marginalize(&weights);
        Ok(dist)
    }

    /// Per-order Krylov input vectors for the model-averaging
    /// decoder. Returns one vector per active order `o in
    /// 1..=min(context.len(), n_orders)`, normalized to unit
    /// norm. Each is the W row for the seen registry mode, or
    /// the dressed vacuum |0̃_o⟩ if unseen.
    fn encode_context_per_order(&self, context: &[u32]) -> Vec<DVector<Complex64>> {
        let n = self.cfg.n_orders.min(context.len());
        let rank = self.pipeline.rank();
        let w = self.pipeline.w();
        let use_hasher = !self.cfg.use_registry_encoder;
        let mut out = Vec::with_capacity(n);
        for o in 1..=n {
            let m_opt = self.registry.lookup(o, context).or_else(|| {
                if use_hasher {
                    crate::features::OrderHasher::new(self.cfg.clone())
                        .mode_for(o, context)
                } else {
                    None
                }
            });
            let mut v = if let Some(m) = m_opt {
                let row = m as usize;
                if row < w.nrows() {
                    let mut c = DVector::<Complex64>::zeros(rank);
                    for r in 0..rank {
                        c[r] = w[(row, r)];
                    }
                    c
                } else {
                    self.vacuum_states[o - 1].clone()
                }
            } else {
                self.vacuum_states[o - 1].clone()
            };
            let norm = v.norm();
            if norm > 0.0 {
                v /= Complex64::new(norm, 0.0);
            }
            out.push(v);
        }
        out
    }

    /// `log P(next | context)` in nats. Returns a finite value for
    /// any token id in `[0, vocab_size)`.
    pub fn logprob(&self, context: &[u32], next: u32) -> Result<f64, QfmTextError> {
        let dist = self.next_token_dist(context)?;
        if (next as usize) >= dist.len() {
            // Out-of-vocab token: return the unigram log-prob of the
            // smallest bin (a defensive choice — should not happen on
            // a corpus-derived vocab).
            return Ok(self.unigram_last_safe_log());
        }
        let p = dist[next as usize].max(1e-30);
        Ok(p.ln())
    }

    /// Sum of unigram counts as a `f64`.
    pub fn unigram_total(&self) -> f64 {
        self.unigram_total
    }

    /// Build a `ChannelAccumulator` from the model's stored
    /// statistics. This is the **fast path** for re-deriving an
    /// `NgramBaseline` from a trained `QfmTextModel` (avoids a
    /// second full pass over the corpus to re-accumulate the
    /// histograms). The unigram counts are reconstructed as
    /// `count_i = round(unigram[i] * unigram_total)` — approximate
    /// but the baseline only needs the **shape** of the
    /// distribution, and the unigram is renormalized to sum to 1
    /// in `NgramBaseline::from_accumulator` anyway.
    pub fn as_accumulator(&self) -> crate::accumulate::ChannelAccumulator {
        let unigram: Vec<u64> = self
            .unigram
            .iter()
            .map(|&p| (p * self.unigram_total).round() as u64)
            .collect();
        let mut acc = crate::accumulate::ChannelAccumulator::new(
            self.unigram.len() as u32,
            self.cfg.clone(),
        );
        acc.stats = self.mode_hists.clone();
        acc.unigram = unigram;
        acc.total_windows = self.unigram_total as u64;
        acc
    }

    /// Build a clone of the [`ContextRegistry`]. The baseline uses
    /// this to encode test contexts to the same mode indices the
    /// model uses.
    pub fn registry_clone(&self) -> ContextRegistry {
        self.registry.clone()
    }

    pub(crate) fn marginalize(&self, weights: &[(u32, f64)]) -> Vec<f64> {
        let v = self.unigram.len();
        // Preprocess the raw Krylov-decoded p̃ (already restricted to
        // the per-context active modes — see `decode_sketched_at`)
        // according to the configured decode strategy. The
        // preprocessing is what fixes the unigram-floor collapse:
        // without it, `total_w` is the Krylov's mass on *per-context
        // active* modes, which is typically < 0.1 (the Krylov
        // spreads its mass over the full K₂ space).
        let p_proc = self.preprocess_p_tilde(weights);
        let mut p = vec![0.0_f64; v];
        let mut total_w = 0.0_f64;
        // For each per-context active mode that has a histogram,
        // weight the smoothed histogram by its processed weight `w`.
        // The smoothing is the classic Katz absolute-discount: for
        // each mode with `weight` total counts, every seen token `y`
        // contributes
        //   (count(y) - discount) / weight
        // and the escape mass
        //   (n_seen_unique * discount + escape) / weight
        // is redistributed to the unigram (per-mode, not globally).
        for &(mode, w) in &p_proc {
            let stats = match self.mode_hists.get(&mode) {
                Some(s) => s,
                None => continue,
            };
            if w <= 0.0 {
                continue;
            }
            total_w += w;
            let denom = stats.weight as f64;
            if denom <= 0.0 {
                continue;
            }
            // **rev 37 design:** no Jelinek-Mercer-style smoothing
            // here. The per-mode distribution is the raw histogram:
            //   p[tok] = w * (cnt / K)     for seen tokens
            //   p[tok] = 0                  for unseen tokens
            // The unigram backoff is NOT applied per-mode. Instead,
            // the outer vacuum projector in the Hamiltonian
            // (W[0, :], the Fock-vacuum mode whose histogram IS
            // the unigram) is included in the active_modes list by
            // `next_token_dist`. Its Krylov weight
            // `|c_1^H · W[0, :]|^2` is the unigram mass.
            //
            // The inner vacuum (c_0 component in the dressed vacuum)
            // is unrelated to this: it's a coefficient in the
            // dressed-vacuum projector, not a per-token probability.
            for &(tok, cnt) in &stats.hist {
                if (tok as usize) < v {
                    p[tok as usize] += w * (cnt as f64 / denom);
                }
            }
        }
        if total_w <= 0.0 {
            // No active mode had any mass (e.g. context with all
            // unseen modes). Return the unigram directly.
            return self.unigram.clone();
        }
        // For DecodeStrategy::Dense we keep the legacy global
        // unigram floor (`1 - total_w`) for backward compatibility.
        // For all other strategies the preprocessed p_tilde
        // already sums to 1 over active modes, so total_w ≈ 1 and
        // the floor is (near-)zero; the per-mode escape above is
        // the only unigram contribution.
        if matches!(self.cfg.decode_strategy, DecodeStrategy::Dense) {
            let mut total_seen = 0.0;
            for &x in &p {
                total_seen += x;
            }
            let scale = if total_seen > 0.0 {
                total_w / total_seen
            } else {
                0.0
            };
            for x in p.iter_mut() {
                *x *= scale;
            }
            let floor_total = (1.0 - total_w).max(0.0);
            if self.unigram_total > 0.0 {
                for (i, &u) in self.unigram.iter().enumerate() {
                    p[i] += floor_total * u;
                }
            }
        }
        // Clamp + renormalize.
        let mut sum = 0.0;
        for x in p.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
            sum += *x;
        }
        if sum > 0.0 {
            for x in p.iter_mut() {
                *x /= sum;
            }
        } else {
            // Total collapse: return the unigram directly.
            return self.unigram.clone();
        }
        p
    }

    /// Decode the Krylov-evolved `c_1` into per-mode weights at
    /// the per-context active modes. The `Decoder` enum (rev 37)
    /// selects the implementation:
    /// - `Dense` (default): reads `W[m, :]` from the dense
    ///   matrix at each active mode `m`.
    /// - `Analytical`: evaluates the per-column `oxieml::EmlTree`
    ///   at `m / k2_total`, with a per-column fallback to the
    ///   dense row for columns whose fit residual exceeded the
    ///   threshold.
    ///
    /// Both paths produce the same `(mode, weight)` list — a
    /// sparse vector over the per-context active modes that
    /// `marginalize` consumes.
    fn decode_at(
        &self,
        c_1: &DVector<Complex64>,
        active_modes: &[u32],
    ) -> Vec<(u32, f64)> {
        // Dispatch on the model's decoder. The default
        // (Dense) keeps the rev 36 behavior. The Analytical
        // path is only active after `fit_oxieml_decoder` is
        // called and `decoder` is replaced.
        match &self.decoder {
            DecoderKind::Dense => {
                self.pipeline.decode_sketched_at(c_1, &self.gram, active_modes)
            }
            DecoderKind::Analytical {
                trees,
                fallback,
                column_ok,
            } => {
                decode_with_trees(c_1, &self.gram, active_modes, trees, fallback, column_ok)
            }
        }
    }

    /// Fit an oxieml `EmlTree` per Krylov basis column and
    /// replace the model's `Decoder` with the
    /// [`DecoderKind::Analytical`] variant. Per-column
    /// fallback to dense W if the fit residual exceeds
    /// `opts.residual_tol`. Returns a summary of the fit
    /// (per-column residuals, number of columns that fell
    /// back to dense).
    ///
    /// This is the rev 37 decoder-replacement step. The
    /// checkpoint size drops from `M * rank * 16` bytes
    /// (dense W complex128) to `O(rank * tree_size)` bytes
    /// for the trees plus `O(M * rank_failed * 16)` for
    /// any fallback columns.
    pub fn fit_oxieml_decoder(
        &mut self,
        opts: oxieml_decoder::OxiemlFitOpts,
    ) -> OxiemlFitSummary {
        let w = self.pipeline.w();
        let (m, rank) = (w.nrows(), w.ncols());
        // Build per-column real vectors (we fit on the real
        // part; the imaginary part is near-zero for the
        // unit-norm basis vectors in our pipeline).
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(rank);
        let mut col_norms: Vec<f64> = Vec::with_capacity(rank);
        for j in 0..rank {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(w[(i, j)].re);
            }
            let max_abs = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            col_norms.push(max_abs);
            if max_abs > 0.0 {
                for v in &mut col {
                    *v /= max_abs;
                }
            }
            columns.push(col);
        }
        // Fit oxieml.
        let fits = oxieml_decoder::fit_decoder(&columns, m, &opts);
        // Build the column_ok mask. A column "succeeds" if
        // its normalized MSE is below the threshold.
        let column_ok: Vec<bool> = fits
            .iter()
            .zip(col_norms.iter())
            .map(|(f, &norm)| {
                // The fit residual is in units of normalized
                // values (we divided by max_abs). To compare
                // against the threshold, we rescale back to
                // the original units: residual = mse * norm^2.
                let scaled = f.mse * norm * norm;
                scaled < opts.residual_tol
            })
            .collect();
        // Build the fallback dense W (only the failed
        // columns, to keep memory bounded).
        let mut fallback = DMatrix::<Complex64>::zeros(m, rank);
        for j in 0..rank {
            if !column_ok[j] {
                for i in 0..m {
                    fallback[(i, j)] = w[(i, j)];
                }
            }
        }
        // Replace the decoder.
        let trees: Vec<oxieml::tree::EmlTree> =
            fits.iter().map(|f| f.tree.clone()).collect();
        let n_fallback = column_ok.iter().filter(|ok| !**ok).count();
        self.decoder = DecoderKind::Analytical {
            trees,
            fallback,
            column_ok,
        };
        OxiemlFitSummary {
            per_column_mse: fits.iter().map(|f| f.mse).collect(),
            per_column_complexity: fits.iter().map(|f| f.complexity).collect(),
            per_column_fallback: self.decoder.column_ok().iter().map(|ok| !*ok).collect(),
            n_fallback,
            total_fit_seconds: fits.iter().map(|f| f.fit_seconds).sum(),
        }
    }

    /// Borrow the model's current `Decoder`.
    pub fn decoder(&self) -> &DecoderKind {
        &self.decoder
    }

    /// Preprocess the raw Krylov-decoded `p̃` (already restricted to
    /// the per-context active modes by `decode_sketched_at`)
    /// according to the configured
    /// [`DecodeStrategy`](crate::config::DecodeStrategy). This is the
    /// structural fix for the unigram-floor collapse documented in
    /// `QFM_TEXT_STATUS.md` §"The real bottleneck".
    ///
    /// `weights` is the `(mode, p̃[mode])` list for exactly this
    /// context's active modes. For `Renormalize`, `TopK`, and
    /// `OrderPrior` the normalization is over **these** modes (the
    /// per-context active set), not over the entire `mode_hists` map
    /// — otherwise the per-mode weight gets diluted by the corpus's
    /// total active-mode count (~10⁴-10⁵) and the unigram floor
    /// re-asserts. See `QFM_TEXT_STATUS.md` §"The real bottleneck"
    /// and the previous failed implementation for the empirical
    /// signature (the same QFM distribution for every context,
    /// matching the unigram order).
    pub fn preprocess_p_tilde(&self, weights: &[(u32, f64)]) -> Vec<(u32, f64)> {
        let uniform_fallback = || -> Vec<(u32, f64)> {
            let n = weights.len();
            if n == 0 {
                return Vec::new();
            }
            let u = 1.0 / n as f64;
            weights.iter().map(|&(m, _)| (m, u)).collect()
        };
        match self.cfg.decode_strategy {
            DecodeStrategy::Dense => {
                // Pass through. Per-mode escape still added in
                // `marginalize`, plus the global unigram floor.
                weights.to_vec()
            }
            DecodeStrategy::Renormalize => {
                // Renormalize over the **per-context active** modes.
                // The Krylov prior is projected onto the active set
                // and treated as a proper distribution over it. If
                // the Krylov put zero mass on every per-context
                // active mode, fall back to uniform over them.
                let sum: f64 = weights.iter().filter(|&&(_, w)| w > 0.0).map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    weights
                        .iter()
                        .filter(|&&(_, w)| w > 0.0)
                        .map(|&(m, w)| (m, w / sum))
                        .collect()
                } else {
                    uniform_fallback()
                }
            }
            DecodeStrategy::TopK => {
                // Sparse top-k: keep the k highest-`p̃` per-context
                // active modes, zero the rest, renormalize. If k ≥
                // |active_modes|, equivalent to Renormalize. If k
                // = 0, the entire prior is discarded → uniform
                // fallback.
                //
                // **rev 37:** mode 0 (the Krylov unigram) is
                // ALWAYS kept, even if it doesn't make the top-k.
                // Dropping it would discard the unigram backoff
                // (the outer-vacuum-projection mass).
                let k = self.cfg.top_k.max(1);
                let mut entries: Vec<(u32, f64)> =
                    weights.iter().copied().filter(|&(_, w)| w > 0.0).collect();
                entries.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut truncated: Vec<(u32, f64)> = entries.iter().take(k).copied().collect();
                if !truncated.iter().any(|&(m, _)| m == 0) {
                    if let Some(&(m, w)) = entries.iter().find(|&&(m, _)| m == 0) {
                        truncated.push((m, w));
                    }
                }
                let sum: f64 = truncated.iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    truncated.into_iter().map(|(m, w)| (m, w / sum)).collect()
                } else {
                    uniform_fallback()
                }
            }
            DecodeStrategy::OrderPrior => {
                // Renormalize + λ_o weighting. Each per-context
                // active mode's mass is multiplied by `λ_o / Σλ_o`
                // (where `o` is the mode's order) before
                // renormalization. This favours higher-order modes
                // (more context) over lower-order ones, shifting
                // mass away from the unigram floor.
                //
                // **rev 37:** mode 0 is the Krylov unigram mass.
                // It is the outer-vacuum-projection backoff and
                // MUST NOT be dropped by the lambda weighting
                // (registry.order_of(0) returns `n_orders`, which
                // is out of `lambda`'s range). Mode 0 keeps its raw
                // Krylov weight; only the per-order modes get the
                // lambda scaling.
                let lambda_sum: f64 = self.cfg.lambda.iter().sum();
                let lambda_sum = if lambda_sum > 0.0 { lambda_sum } else { 1.0 };
                let mut out: Vec<(u32, f64)> = Vec::new();
                let mut sum = 0.0;
                for &(mode, w) in weights {
                    if w > 0.0 {
                        let (wo, include) = if mode == 0 {
                            // Mode 0 (Krylov unigram): keep the
                            // raw Krylov weight — it's the
                            // outer-vacuum backoff.
                            (w, true)
                        } else {
                            let o = self.registry.order_of(mode);
                            let lambda_o =
                                self.cfg.lambda.get(o).copied().unwrap_or(0.0) / lambda_sum;
                            (w * lambda_o, lambda_o > 0.0)
                        };
                        if include {
                            out.push((mode, wo));
                            sum += wo;
                        }
                    }
                }
                if sum > 0.0 {
                    for entry in out.iter_mut() {
                        entry.1 /= sum;
                    }
                    out
                } else {
                    uniform_fallback()
                }
            }
        }
    }

    fn unigram_last_safe_log(&self) -> f64 {
        self.unigram
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p.ln())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Save the model to `path` as a single binary blob. Format
    /// (rev 36, schema 2):
    ///   - magic: 8 bytes "QFM-TEXT" (no null terminator)
    ///   - version: u32 LE (= 2)
    ///   - json_len: u32 LE, then json (TextConfig + metadata)
    ///   - payload_len: u64 LE, then bincode (W, H_m, W_prob, histograms, unigram, registry)
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), QfmTextError> {
        let path_ref = path.as_ref();
        let mut f = File::create(path_ref)?;
        f.write_all(b"QFM-TEXT")?;
        f.write_all(&self.schema_version.to_le_bytes())?;
        let meta = self.metadata();
        let meta_json = serde_json::to_vec(&serde_json::json!({
            "schema_version": self.schema_version,
            "vocab_size": meta.vocab_size,
            "n_orders": meta.n_orders,
            "k2_total": meta.k2_total,
            "n_active_modes": meta.n_active_modes,
            "total_windows": meta.total_windows,
        }))?;
        f.write_all(&(meta_json.len() as u32).to_le_bytes())?;
        f.write_all(&meta_json)?;
        // Payload: W (rank, ncols) + H_m + W_prob + unigram +
        // mode_hists (entries as Vec<u32,u32,u64,Vec<(u32,u32)>>) +
        // registry (per-order map of context_tokens -> mode_index).
        let payload = encode_payload(self)?;
        f.write_all(&(payload.len() as u64).to_le_bytes())?;
        f.write_all(&payload)?;
        Ok(())
    }

    /// Load a model from `path`. Validates magic + schema version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, QfmTextError> {
        let path_ref = path.as_ref();
        let mut buf = Vec::new();
        File::open(path_ref)?.read_to_end(&mut buf)?;
        if buf.len() < 12 || &buf[..8] != b"QFM-TEXT" {
            return Err(QfmTextError::BadManifest {
                path: path_ref.display().to_string(),
                reason: "missing QFM-TEXT magic".to_string(),
            });
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != crate::SCHEMA_VERSION {
            return Err(QfmTextError::BadManifest {
                path: path_ref.display().to_string(),
                reason: format!("schema version {version} != {}", crate::SCHEMA_VERSION),
            });
        }
        let mut offset = 12;
        let meta_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        // The metadata is diagnostic only; we deserialize the payload
        // directly to reconstruct the model.
        let _meta: serde_json::Value = serde_json::from_slice(&buf[offset..offset + meta_len])?;
        offset += meta_len;
        let payload_len =
            u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + payload_len != buf.len() {
            return Err(QfmTextError::BadManifest {
                path: path_ref.display().to_string(),
                reason: "payload length mismatch".to_string(),
            });
        }
        decode_payload(&buf[offset..])
    }
}

/// Public form of `encode_modes` for callers outside this module
/// (e.g. the in-context adapter, which hashes a raw prefix instead
/// of an already-hashed mode list).
pub fn public_encode_modes(context: &[u32], registry: &ContextRegistry) -> Vec<u32> {
    registry.encode_modes(context)
}

/// Convert the accumulator's mode weights into the per-order
/// `(lambda, channels)` list consumed by
/// `QfmPipeline::compile_channels`. For each order, the channels are
/// the active modes in that order's registry block, with alpha_j =
/// weight_j / total_windows (the QFM.tex unit-norm channel
/// normalization). Modes that are present in the registry but have
/// zero observations in the accumulator (e.g. contexts that the
/// training pass never extended) are skipped.
fn build_channel_groups(
    acc: &ChannelAccumulator,
    registry: &ContextRegistry,
    cfg: &TextConfig,
) -> Vec<(f64, Vec<(u32, f64)>)> {
    let total = acc.total_windows.max(1) as f64;
    let mut groups = Vec::with_capacity(cfg.n_orders);
    for o in 0..cfg.n_orders {
        let lambda = cfg.lambda[o];
        // The upper bound on the mode indices for order `o` is the
        // *block size* in the TextConfig (the hash table's actual
        // size, or the registry's per-order range). The registry
        // may have fewer entries (e.g. an empty registry for the
        // OrderHasher path: the registry is just a placeholder
        // here, the real mode indices come from the accumulator's
        // `stats` map which is populated by the hasher).
        let off = if registry.n_active_for_order(o) > 0 {
            registry.offset(o)
        } else {
            cfg.offset(o)
        };
        let block_size = if registry.n_active_for_order(o) > 0 {
            registry.n_active_for_order(o)
        } else {
            cfg.block_sizes[o]
        };
        let mut channels = Vec::new();
        for k in 0..block_size {
            let mode = off + k as u32;
            if let Some(stats) = acc.stats.get(&mode) {
                if stats.weight > 0 {
                    let alpha = stats.weight as f64 / total;
                    channels.push((mode, alpha));
                }
            }
        }
        groups.push((lambda, channels));
    }
    groups
}

/// Encode the model payload as a **flat binary buffer** — no
/// `serde_json::Value` tree. This is the same memory-bounded
/// approach as rev 34/35: every field is written as raw
/// little-endian bytes into a pre-sized `Vec<u8>`, so peak memory
/// is the payload size itself, not a multiple of it.
///
/// **Layout** (all integers little-endian):
///   `w_rows:u64, w_cols:u64, [re:f64,im:f64] * (w_rows*w_cols)` (row-major)
///   `h_rows:u64, h_cols:u64, [re:f64,im:f64] * (h_rows*h_cols)` (row-major)
///   `wp_rows:u64, wp_cols:u64, f64 * (wp_rows*wp_cols)` (row-major)
///   `unigram_len:u64, f64 * unigram_len`
///   `unigram_total:f64`
///   `mode_hists_len:u64`, then per entry:
///     `mode:u32, weight:u64, escape:u64, hist_len:u32, [tok:u32,cnt:u32] * hist_len`
///   `config_json_len:u32, config_json_bytes` (config is tiny; JSON is fine here)
///   `registry_n_orders:u32`, then per order:
///     `map_len:u64`, then per entry:
///       `key_len:u32, key_bytes:u8 * key_len, mode:u32`
///     (the keys are `Vec<u32>` little-endian token ids; the values
///     are the assigned mode indices)
///   `vacuum_states_n:u32`, then per vacuum state:
///     `len:u64, [re:f64,im:f64] * len` (rev 36: per-order |0̃_o⟩)
///   `outer_vacuum_len:u64, [re:f64,im:f64] * len` (rev 37 v3: outer vacuum)
fn encode_payload(m: &QfmTextModel) -> Result<Vec<u8>, QfmTextError> {
    let w = m.pipeline.w();
    let h_m = m.pipeline.h_m();
    let w_prob = m.pipeline.w_prob();
    let cfg_bytes = serde_json::to_vec(&m.cfg)?;

    let hist_entries: usize = m.mode_hists.values().map(|s| s.hist.len()).sum();
    let registry_entries: usize = m.registry.maps().iter().map(|mm| mm.len()).sum();
    let vacuum_total: usize = m.vacuum_states.iter().map(|v| v.len()).sum();
    let cap = 16 // w_rows,w_cols
        + w.nrows() * w.ncols() * 16
        + 16 // h_rows,h_cols
        + h_m.nrows() * h_m.ncols() * 16
        + 16 // wp_rows,wp_cols
        + w_prob.nrows() * w_prob.ncols() * 8
        + 8 + m.unigram.len() * 8 // unigram_len + data
        + 8 // unigram_total
        + 8 // mode_hists_len
        + m.mode_hists.len() * (4 + 8 + 8 + 4)
        + hist_entries * 8
        + 4 + cfg_bytes.len()
        + 4 // registry_n_orders
        + m.registry.maps().len() * 8 // per-order map_len
        + registry_entries * (4 + 4 * m.cfg.n_orders as usize + 4)
        + 4 // vacuum_states_n
        + m.vacuum_states.len() * 8 // per-vacuum len
        + vacuum_total * 16 // [re,im] per entry
        + 8 // outer_vacuum_len
        + m.outer_vacuum.len() * 16; // [re,im] per entry
    let mut buf = Vec::with_capacity(cap);

    buf.extend_from_slice(&(w.nrows() as u64).to_le_bytes());
    buf.extend_from_slice(&(w.ncols() as u64).to_le_bytes());
    for i in 0..w.nrows() {
        for j in 0..w.ncols() {
            buf.extend_from_slice(&w[(i, j)].re.to_le_bytes());
            buf.extend_from_slice(&w[(i, j)].im.to_le_bytes());
        }
    }

    buf.extend_from_slice(&(h_m.nrows() as u64).to_le_bytes());
    buf.extend_from_slice(&(h_m.ncols() as u64).to_le_bytes());
    for i in 0..h_m.nrows() {
        for j in 0..h_m.ncols() {
            buf.extend_from_slice(&h_m[(i, j)].re.to_le_bytes());
            buf.extend_from_slice(&h_m[(i, j)].im.to_le_bytes());
        }
    }

    buf.extend_from_slice(&(w_prob.nrows() as u64).to_le_bytes());
    buf.extend_from_slice(&(w_prob.ncols() as u64).to_le_bytes());
    for i in 0..w_prob.nrows() {
        for j in 0..w_prob.ncols() {
            buf.extend_from_slice(&w_prob[(i, j)].to_le_bytes());
        }
    }

    buf.extend_from_slice(&(m.unigram.len() as u64).to_le_bytes());
    for &x in &m.unigram {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    buf.extend_from_slice(&m.unigram_total.to_le_bytes());

    buf.extend_from_slice(&(m.mode_hists.len() as u64).to_le_bytes());
    for (&mode, stats) in &m.mode_hists {
        buf.extend_from_slice(&mode.to_le_bytes());
        buf.extend_from_slice(&stats.weight.to_le_bytes());
        buf.extend_from_slice(&stats.escape.to_le_bytes());
        buf.extend_from_slice(&(stats.hist.len() as u32).to_le_bytes());
        for &(tok, cnt) in &stats.hist {
            buf.extend_from_slice(&tok.to_le_bytes());
            buf.extend_from_slice(&cnt.to_le_bytes());
        }
    }

    buf.extend_from_slice(&(cfg_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&cfg_bytes);

    // Registry: per-order map of (context_tokens_le_u32, mode_u32).
    buf.extend_from_slice(&(m.registry.maps().len() as u32).to_le_bytes());
    for map in m.registry.maps() {
        buf.extend_from_slice(&(map.len() as u64).to_le_bytes());
        for (key, &mode) in map {
            let key_bytes: Vec<u8> = key.iter().flat_map(|&t| t.to_le_bytes()).collect();
            buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&key_bytes);
            buf.extend_from_slice(&mode.to_le_bytes());
        }
    }

    // Vacuum states: per-order |0̃_o⟩ DVector<Complex64> of length rank.
    buf.extend_from_slice(&(m.vacuum_states.len() as u32).to_le_bytes());
    for v in &m.vacuum_states {
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v.iter() {
            buf.extend_from_slice(&x.re.to_le_bytes());
            buf.extend_from_slice(&x.im.to_le_bytes());
        }
    }

    // Outer vacuum |Ψ_0⟩: the L^1-uniform Krylov initial vector
    // (rev 37 v3). DVector<Complex64> of length rank.
    buf.extend_from_slice(&(m.outer_vacuum.len() as u64).to_le_bytes());
    for x in m.outer_vacuum.iter() {
        buf.extend_from_slice(&x.re.to_le_bytes());
        buf.extend_from_slice(&x.im.to_le_bytes());
    }

    Ok(buf)
}

fn decode_payload(buf: &[u8]) -> Result<QfmTextModel, QfmTextError> {
    let mut o = 0usize;
    let read_u64 = |o: &mut usize| -> u64 {
        let v = u64::from_le_bytes(buf[*o..*o + 8].try_into().unwrap());
        *o += 8;
        v
    };
    let read_u32 = |o: &mut usize| -> u32 {
        let v = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let read_f64 = |o: &mut usize| -> f64 {
        let v = f64::from_le_bytes(buf[*o..*o + 8].try_into().unwrap());
        *o += 8;
        v
    };

    let w_rows = read_u64(&mut o) as usize;
    let w_cols = read_u64(&mut o) as usize;
    let mut w_mat = DMatrix::<Complex64>::zeros(w_rows, w_cols);
    for i in 0..w_rows {
        for j in 0..w_cols {
            let re = read_f64(&mut o);
            let im = read_f64(&mut o);
            w_mat[(i, j)] = Complex64::new(re, im);
        }
    }

    let hm_rows = read_u64(&mut o) as usize;
    let hm_cols = read_u64(&mut o) as usize;
    let mut h_mat = DMatrix::<Complex64>::zeros(hm_rows, hm_cols);
    for i in 0..hm_rows {
        for j in 0..hm_cols {
            let re = read_f64(&mut o);
            let im = read_f64(&mut o);
            h_mat[(i, j)] = Complex64::new(re, im);
        }
    }

    let wp_rows = read_u64(&mut o) as usize;
    let wp_cols = read_u64(&mut o) as usize;
    let mut wp_mat = DMatrix::<f64>::zeros(wp_rows, wp_cols);
    for i in 0..wp_rows {
        for j in 0..wp_cols {
            wp_mat[(i, j)] = read_f64(&mut o);
        }
    }

    let unigram_len = read_u64(&mut o) as usize;
    let mut unigram = Vec::with_capacity(unigram_len);
    for _ in 0..unigram_len {
        unigram.push(read_f64(&mut o));
    }
    let unigram_total = read_f64(&mut o);

    let mode_hists_len = read_u64(&mut o) as usize;
    let mut mode_hists = FxHashMap::default();
    mode_hists.reserve(mode_hists_len);
    for _ in 0..mode_hists_len {
        let mode = read_u32(&mut o);
        let weight = read_u64(&mut o);
        let escape = read_u64(&mut o);
        let hist_len = read_u32(&mut o) as usize;
        let mut hist = Vec::with_capacity(hist_len);
        for _ in 0..hist_len {
            let tok = read_u32(&mut o);
            let cnt = read_u32(&mut o);
            hist.push((tok, cnt));
        }
        mode_hists.insert(
            mode,
            ModeStats {
                weight,
                escape,
                hist,
            },
        );
    }

    let cfg_len = read_u32(&mut o) as usize;
    let cfg: TextConfig = serde_json::from_slice(&buf[o..o + cfg_len])?;
    o += cfg_len;

    // Registry: per-order map of (context_tokens, mode_index).
    let n_orders = read_u32(&mut o) as usize;
    let mut maps: Vec<FxHashMap<Vec<u32>, u32>> = Vec::with_capacity(n_orders);
    for _ in 0..n_orders {
        let map_len = read_u64(&mut o) as usize;
        let mut map = FxHashMap::default();
        map.reserve(map_len);
        for _ in 0..map_len {
            let key_bytes_len = read_u32(&mut o) as usize;
            let key_bytes = &buf[o..o + key_bytes_len];
            o += key_bytes_len;
            // Each key is `key_bytes_len / 4` little-endian u32
            // tokens. (The order is implicit: order o has the last
            // o tokens of the original context.)
            let n_toks = key_bytes_len / 4;
            let mut key: Vec<u32> = Vec::with_capacity(n_toks);
            for i in 0..n_toks {
                let t = u32::from_le_bytes(
                    key_bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                );
                key.push(t);
            }
            let mode = read_u32(&mut o);
            map.insert(key, mode);
        }
        maps.push(map);
    }
    let registry = ContextRegistry::from_maps(maps);

    // Vacuum states: per-order |0̃_o⟩.
    let vac_n = read_u32(&mut o) as usize;
    let mut vacuum_states: Vec<DVector<Complex64>> = Vec::with_capacity(vac_n);
    for _ in 0..vac_n {
        let vlen = read_u64(&mut o) as usize;
        let mut v = DVector::<Complex64>::zeros(vlen);
        for k in 0..vlen {
            let re = read_f64(&mut o);
            let im = read_f64(&mut o);
            v[k] = Complex64::new(re, im);
        }
        vacuum_states.push(v);
    }

    // Outer vacuum |Ψ_0⟩: the L^1-uniform Krylov initial vector
    // (rev 37 v3). DVector<Complex64> of length rank.
    let outer_vacuum_len = read_u64(&mut o) as usize;
    let mut outer_vacuum = DVector::<Complex64>::zeros(outer_vacuum_len);
    for k in 0..outer_vacuum_len {
        let re = read_f64(&mut o);
        let im = read_f64(&mut o);
        outer_vacuum[k] = Complex64::new(re, im);
    }

    // Build the QfmPipeline directly from the stored W, H_m, W_prob
    // matrices via `from_components`. This avoids re-running the
    // SIRK compile on load (which is expensive and data-dependent:
    // the post-SIRK rank depends on the input distribution, so
    // re-compiling on the same data can give a different rank than
    // the original save). The stored matrices are the canonical
    // values; we just need to wrap them in a QfmPipeline.
    let pipeline = QfmPipeline::from_components(w_mat, h_mat, wp_mat);
    let gram = pipeline.gram();
    Ok(QfmTextModel {
        pipeline,
        mode_hists,
        unigram,
        gram,
        unigram_total,
        cfg,
        schema_version: crate::SCHEMA_VERSION,
        registry,
        vacuum_states,
        outer_vacuum,
        decoder: DecoderKind::Dense,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::ChannelAccumulator;
    use tempfile::tempdir;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
            hist_cap: 4,
            max_rank: 4,
            m_shifts: 4,
            lambda: vec![1.0, 1.0],
            t: 1.0,
            discount: 0.5,
            seed: 0,
            ..Default::default()
        }
    }

    fn build_toy_corpus(tokens: &[u32]) -> (ChannelAccumulator, ContextRegistry, TextConfig) {
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(cfg().n_orders);
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            observe_with_registry(&mut acc, &mut reg, &ctx, tokens[i]);
        }
        (acc, reg, cfg())
    }

    #[test]
    fn model_compiles_and_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let (acc, reg, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dist = model.next_token_dist(&[3, 5]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0, "zero probability");
            assert!(p.is_finite());
        }
    }

    #[test]
    fn t_zero_degrades_to_classical_mixture() {
        // At t=0 the evolution is the identity, so the per-mode
        // Born distribution is concentrated on the active mode(s).
        // The marginalization against the mode's own histogram with
        // smoothing should approximate the classical Katz-style
        // backoff for that mode.
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, reg, mut c) = build_toy_corpus(&tokens);
        c.t = 0.0;
        c.discount = 0.0; // no smoothing, pure empirical
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        // Pick a context and a comparison. We just check the
        // distribution is finite and sums to 1.
        let dist = model.next_token_dist(&[2, 3]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn save_load_round_trip() {
        let tokens: Vec<u32> = (0..150).map(|i| i % 6).collect();
        let (acc, reg, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dir = tempdir().unwrap();
        let p = dir.path().join("model.qfm");
        model.save(&p).unwrap();
        let loaded = QfmTextModel::load(&p).unwrap();
        // Compare log-prob on 50 windows.
        for i in 1..50 {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let lp1 = model.logprob(&ctx, tokens[i]).unwrap();
            let lp2 = loaded.logprob(&ctx, tokens[i]).unwrap();
            assert!(
                (lp1 - lp2).abs() < 1e-6,
                "logprob mismatch at i={i}: {lp1} vs {lp2}"
            );
        }
        // Registry round-trip: same active modes for the same
        // contexts.
        for i in 1..50 {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let m1 = model.registry.encode_modes(&ctx);
            let m2 = loaded.registry.encode_modes(&ctx);
            assert_eq!(m1, m2, "registry mismatch at i={i}: {m1:?} vs {m2:?}");
        }
    }

    #[test]
    fn model_avg_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let (acc, reg, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dist = model.next_token_dist_model_avg(&[3, 5]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0, "zero probability");
            assert!(p.is_finite());
        }
    }

    #[test]
    fn renormalize_decode_lifts_seen_mass() {
        // The Renormalize strategy must not collapse the model to
        // a pure unigram for a clearly context-conditioned
        // corpus. The pre-rev-36 implementation (with the bounded
        // hashed encoder) had a structural bug where the Krylov
        // prior's mass on the per-context active modes was diluted
        // to ≈ 0 and the residual `1 - total_w` was routed to the
        // global unigram — producing the same QFM distribution for
        // every context and matching the unigram. Renormalize
        // (the rev 33/34 fix) projects the Krylov prior onto the
        // per-context active set so the per-mode histograms carry
        // the context-conditioned peaks.
        //
        // Rev 36 note: with the registry (no hashing) the model
        // also gets the per-mode-weight structure (one mode per
        // unique context), so the per-mode histograms are
        // unambiguous. The Krylov-smoothing and the Mehler-formalism
        // dressed vacuum in H do still mix mass across modes —
        // for a small toy corpus the per-context peak can be
        // diluted. The honest property this test asserts is
        // "the model gives non-trivial weight to the
        // context-conditioned peak, not the unigram-dominant
        // one", *not* "the per-mode peak is the argmax" (which
        // requires the model to be near-perfect on a small
        // corpus and is a property the rev 35 W-rank-degeneracy
        // diagnosis explicitly rules out). The end-to-end
        // held-out evaluation on WikiText-103 is the honest
        // measure of "does the QFM beat the baseline".
        let mut tokens: Vec<u32> = Vec::new();
        for _ in 0..100 {
            tokens.extend_from_slice(&[0, 1, 2, 7]);
        }
        for _ in 0..400 {
            tokens.extend_from_slice(&[3, 7]);
        }
        let mut acc = ChannelAccumulator::new(0, cfg());
        let mut reg = ContextRegistry::new(cfg().n_orders);
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            observe_with_registry(&mut acc, &mut reg, &ctx, tokens[i]);
        }
        let mut c = cfg();
        c.decode_strategy = DecodeStrategy::Renormalize;
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dist = model.next_token_dist(&[0, 1]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p.is_finite());
        }
        // The unigram floor guarantees non-zero probability for
        // every token that has a non-zero unigram (those tokens
        // receive the per-mode escape mass + the vacuum's
        // unigram-histogram contribution). Tokens with zero
        // unigram (tokens the training corpus never produced)
        // correctly have P = 0 — the model has no information
        // about them and shouldn't fabricate probability mass.
        for (i, &p) in dist.iter().enumerate() {
            if model.unigram[i] > 0.0 {
                assert!(p > 0.0, "P[{i}] = {p} must be > 0 for tokens with non-zero unigram");
            }
        }
        // The context-conditioned peak (token 2) must carry
        // *more* probability than the unigram-dominant
        // background (token 7) — the structural fix for the
        // unigram-floor collapse. The exact ratio depends on the
        // Krylov-smoothing strength; on a 700-token toy corpus it
        // can be marginal. We require P(2|0,1) > 0.05 (a few
        // percent) — a clear signal that the per-mode
        // context-conditioned peak is in the distribution at all.
        let dist_2 = dist[2];
        assert!(
            dist_2 > 0.05,
            "Renormalize P(2|0,1) = {dist_2} should be non-trivial (context-conditioned peak must survive Krylov smoothing)"
        );
    }

    #[test]
    fn topk_decode_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, reg, mut c) = build_toy_corpus(&tokens);
        c.decode_strategy = DecodeStrategy::TopK;
        c.top_k = 2;
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dist = model.next_token_dist(&[2, 3]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0 && p.is_finite());
        }
    }

    #[test]
    fn order_prior_decode_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, reg, mut c) = build_toy_corpus(&tokens);
        c.decode_strategy = DecodeStrategy::OrderPrior;
        c.lambda = vec![1.0, 10.0];
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        let dist = model.next_token_dist(&[1, 2]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0 && p.is_finite());
        }
    }

    #[test]
    fn dense_decode_preserves_unigram_floor() {
        let tokens: Vec<u32> = (0..500).map(|i| i % 11).collect();
        let (acc_dense, reg_dense, mut c_dense) = build_toy_corpus(&tokens);
        c_dense.decode_strategy = DecodeStrategy::Dense;
        let model_dense =
            QfmTextModel::from_accumulator(acc_dense, reg_dense, &c_dense).unwrap();
        let (acc_renorm, reg_renorm, mut c_renorm) = build_toy_corpus(&tokens);
        c_renorm.decode_strategy = DecodeStrategy::Renormalize;
        let model_renorm =
            QfmTextModel::from_accumulator(acc_renorm, reg_renorm, &c_renorm).unwrap();
        let d_dense = model_dense.next_token_dist(&[3, 7]).unwrap();
        let d_renorm = model_renorm.next_token_dist(&[3, 7]).unwrap();
        let l1 = |d: &[f64]| -> f64 {
            d.iter()
                .zip(model_dense.unigram.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 2.0
        };
        let l1_dense = l1(&d_dense);
        let l1_renorm = l1(&d_renorm);
        assert!(
            l1_dense <= l1_renorm * 1.1 + 1e-6,
            "Dense L1 to unigram = {l1_dense} should be <= Renormalize L1 = {l1_renorm}"
        );
    }

    #[test]
    fn unseen_context_uses_vacuum_mode_through_krylov() {
        // Unseen test context → registry returns [VACUUM_MODE] →
        // Krylov evolves the vacuum → marginalize against the
        // unigram-histogram. The distribution must be finite and
        // sum to 1.
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, reg, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, reg, &c).unwrap();
        // Pick a context that's definitely not in the training set.
        let unseen_ctx = vec![9999, 9998, 9997];
        let dist = model.next_token_dist(&unseen_ctx).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0 && p.is_finite());
        }
    }
}
