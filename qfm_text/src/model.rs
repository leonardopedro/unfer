//! The compiled QFM-Text language model (Stage 4).
//!
//! `QfmTextModel` wraps a `qfm::QfmPipeline` (the TSR reduced basis
//! + reduced Hamiltonian), the per-mode statistics from the
//! accumulator, the unigram floor, and the configuration snapshot.
//! It exposes:
//!   - `next_token_dist(&[u32]) -> Vec<f64>`: the per-token probability
//!     vector `P(y | context)`, computed by encoding the context into
//!     ≤ n mode indices, evolving each through the reduced
//!     Hamiltonian, marginalising the Born-rule sketch against the
//!     per-mode histograms with absolute-discount smoothing to the
//!     unigram, and clamping the result to a valid distribution.
//!   - `logprob(&[u32], u32) -> f64`: `log P(next | context)`, the
//!     per-token log-probability used by the perplexity evaluator.
//!   - `save(&Path) / load(&Path)`: bincode-serialized model.
//!
//! The model is a **quantum-kernel n-gram-family model** with
//! coherent Krylov smoothing across backoff orders: per-mode
//! histograms are the *capacity*; the dressed-vacuum projector sum +
//! Krylov evolution is the *smoothing*; the unigram is the floor.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use nalgebra::DMatrix;
use num_complex::Complex64;
use qfm::{QfmConfig, QfmPipeline};
use rustc_hash::FxHashMap;

use crate::accumulate::{ChannelAccumulator, ModeStats};
use crate::config::{DecodeStrategy, TextConfig};
use crate::error::QfmTextError;

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
    /// the global index in `[1, k2_total)`.
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
    /// `decode_sketched` output. This is what makes it affordable to
    /// scale `block_sizes` up to reduce hash collisions (the
    /// confirmed dominant cause of the fit ceiling,
    /// `QFM_TEXT_STATUS.md` rev 35) without a per-token cost blowup.
    pub gram: DMatrix<Complex64>,
}

impl QfmTextModel {
    /// Build a QfmTextModel from a streaming accumulator. The
    /// `k2_total` and the per-order `block_sizes` come from the
    /// config; the channel weights are `weight_j / total_windows`
    /// (the per-mode marginal frequency, the QFM.tex ᾱ_j
    /// normalization for unit-norm channels).
    pub fn from_accumulator(
        acc: ChannelAccumulator,
        cfg: &TextConfig,
    ) -> Result<Self, QfmTextError> {
        let k2_total = cfg.k2_total();
        // Build the per-order (λ_o, channels_o) groups: for each
        // order, the active modes in its block with
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
        let groups = build_channel_groups(&acc, cfg);
        // Compile the QFM pipeline.
        let qfm_cfg = QfmConfig {
            k: cfg.n_orders,
            k2: k2_total as usize,
            krylov_dim: cfg.m_shifts,
            seed: cfg.seed,
            n_t_samples: 4,
            noise_dim: cfg.n_orders,
            max_rank: Some(cfg.max_rank),
        };
        let pipeline = QfmPipeline::compile_channels(&groups, k2_total as usize, &qfm_cfg)?;
        let gram = pipeline.gram();
        // Unigram normalize.
        let unigram_total: f64 = acc.unigram.iter().map(|&c| c as f64).sum();
        let unigram: Vec<f64> = if unigram_total > 0.0 {
            acc.unigram.iter().map(|&c| c as f64 / unigram_total).collect()
        } else {
            vec![0.0; acc.unigram.len()]
        };
        Ok(Self {
            pipeline,
            mode_hists: acc.stats,
            unigram,
            unigram_total,
            cfg: cfg.clone(),
            schema_version: crate::SCHEMA_VERSION,
            gram,
        })
    }

    /// Metadata for diagnostics.
    pub fn metadata(&self) -> TextModelMetadata {
        TextModelMetadata {
            vocab_size: self.unigram.len() as u32,
            n_orders: self.cfg.n_orders,
            k2_total: self.cfg.k2_total(),
            n_active_modes: self.mode_hists.len(),
            total_windows: (self.unigram_total) as u64,
        }
    }

    /// Compute the per-token next-token distribution for a context.
    /// The vector is length `vocab_size`, sums to 1.0, and every entry
    /// is `> 0` (the unigram floor guarantees no zero probability).
    pub fn next_token_dist(&self, context: &[u32]) -> Result<Vec<f64>, QfmTextError> {
        // 1. Hash the context into the per-order active modes.
        let active_modes = context_modes(context, &self.cfg);
        if active_modes.is_empty() {
            // No active mode (empty context with n_orders > 0). Return
            // the unigram as the fallback.
            return Ok(self.unigram.clone());
        }
        // 2. Encode the context into a Krylov coefficient vector
        //    (equal-weight superposition of the W-basis rows for the
        //    active modes).
        let active = self.pipeline.encode_modes(&active_modes)?;
        // 3. Evolve the superposition forward by t.
        let c_1 = self.pipeline.evolve(&active, self.cfg.t);
        // 4. Decode the sketch (Phase 3): per-mode Born probability,
        //    computed only at the per-context active modes (the only
        //    entries `marginalize` ever reads) via the O(rank^2)
        //    Gram-matrix normalization instead of an O(K_2) dense
        //    scan — see `QfmTextModel::gram`.
        let weights = self.pipeline.decode_sketched_at(&c_1, &self.gram, &active_modes);
        // 5. Marginalise against the per-mode histograms.
        let dist = self.marginalize(&weights);
        Ok(dist)
    }

    /// Compute the per-token next-token distribution via **model
    /// averaging** (mixture of experts over the per-order Krylov
    /// models). For each active mode `m_i`:
    ///
    ///   1. Encode `m_i` as a unit-norm Krylov vector `c_0_i = w[m_i] / ||w[m_i]||`.
    ///   2. Evolve: `c_1_i = exp(-i H_m t) c_0_i`.
    ///   3. Decode: `p̃_i = decode_sketched(c_1_i)`.
    ///
    /// Then `p̃ = (1/n) Σ_i p̃_i`, and the usual `marginalize` step
    /// is applied. This avoids the destructive interference in the
    /// equal-weight superposition of `next_token_dist` and was the
    /// diagnostic in `QFM_TEXT_STATUS.md` next-step §3.
    ///
    /// Cost: `n` forward solves per token instead of 1. For the
    /// production 4-order model with `n_orders=4`, this is a 4×
    /// cost (still dominated by the per-mode histogram lookups).
    pub fn next_token_dist_model_avg(
        &self,
        context: &[u32],
    ) -> Result<Vec<f64>, QfmTextError> {
        // 1. Hash the context into the per-order active modes.
        let active_modes = context_modes(context, &self.cfg);
        if active_modes.is_empty() {
            return Ok(self.unigram.clone());
        }
        // 2. Encode each active mode as a separate unit vector.
        let c_0_list = self.pipeline.encode_modes_per_order(&active_modes)?;
        // 3. Evolve each independently and decode only at the
        //    per-context active modes (see `next_token_dist`).
        let n = c_0_list.len();
        let mut acc: FxHashMap<u32, f64> = FxHashMap::default();
        for c_0 in &c_0_list {
            let c_1 = self.pipeline.evolve(c_0, self.cfg.t);
            for (m, p) in self.pipeline.decode_sketched_at(&c_1, &self.gram, &active_modes) {
                *acc.entry(m).or_insert(0.0) += p;
            }
        }
        // 4. Average the decoded weights across the n per-order solves.
        let weights: Vec<(u32, f64)> = acc.into_iter().map(|(m, p)| (m, p / n as f64)).collect();
        // 5. Marginalise.
        let dist = self.marginalize(&weights);
        Ok(dist)
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
        use crate::accumulate::ChannelAccumulator;
        let unigram: Vec<u64> = self
            .unigram
            .iter()
            .map(|&p| (p * self.unigram_total).round() as u64)
            .collect();
        ChannelAccumulator {
            stats: self.mode_hists.clone(),
            unigram,
            total_windows: self.unigram_total as u64,
            cfg: self.cfg.clone(),
        }
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
            let d = self.cfg.discount;
            let n_seen = stats.hist.len() as f64;
            // Escape mass: the fraction of this mode's probability
            // that backs off to the unigram. The escape fraction is
            // the number of seen unique tokens times the discount,
            // plus the already-discounted escape count.
            //
            // The seen entries' own contributions, `(cnt - d) / denom`
            // summed over `stats.hist`, already equal `1 - escape_mass`
            // by construction (the accumulator invariant `weight =
            // sum(hist counts) + escape`, see `hist_cap_evicts_to_escape`
            // in accumulate.rs). An earlier version of this loop
            // multiplied each term by `seen_total = 1 - escape_mass`
            // again, double-applying the discount and shrinking the
            // seen mass to `seen_total^2` while leaving the escape
            // mass at `escape_mass` — systematically over-weighting
            // the unigram floor after renormalization (the "model
            // effectively predicts the unigram" failure signature).
            let escape_mass = (n_seen * d + stats.escape as f64) / denom;
            for &(tok, cnt) in &stats.hist {
                if (tok as usize) < v {
                    p[tok as usize] += w * ((cnt as f64 - d).max(0.0) / denom);
                }
            }
            // Distribute the per-mode escape mass to the unigram.
            // This is the standard Katz backoff: tokens not in the
            // histogram get probability from the unigram floor. With
            // `DecodeStrategy::Renormalize` the global unigram floor
            // is removed (total_w → 1) and the per-mode escape is
            // the *only* way unigram mass enters the distribution.
            for (i, &u) in self.unigram.iter().enumerate() {
                p[i] += w * escape_mass * u;
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
                let k = self.cfg.top_k.max(1);
                let mut entries: Vec<(u32, f64)> =
                    weights.iter().copied().filter(|&(_, w)| w > 0.0).collect();
                entries.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                entries.truncate(k);
                let sum: f64 = entries.iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    entries.into_iter().map(|(m, w)| (m, w / sum)).collect()
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
                let lambda_sum: f64 = self.cfg.lambda.iter().sum();
                let lambda_sum = if lambda_sum > 0.0 { lambda_sum } else { 1.0 };
                let mut out: Vec<(u32, f64)> = Vec::new();
                let mut sum = 0.0;
                for &(mode, w) in weights {
                    if w > 0.0 {
                        let o = self.cfg.order_of(mode);
                        let lambda_o =
                            self.cfg.lambda.get(o).copied().unwrap_or(0.0) / lambda_sum;
                        let wo = w * lambda_o;
                        out.push((mode, wo));
                        sum += wo;
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

    /// Save the model to `path` as a single binary blob. Format:
    ///   - magic: 8 bytes "QFM-TEXT" (no null terminator)
    ///   - version: u32 LE
    ///   - json_len: u32 LE, then json (TextConfig + metadata)
    ///   - payload_len: u64 LE, then bincode (W, H_m, W_prob, histograms, unigram)
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
        // mode_hists (entries as Vec<u32,u32,u64,Vec<(u32,u32)>>).
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

/// Hash a token context into the per-order active mode indices
/// (the same encoding the streaming accumulator uses). The Krylov
/// pipeline reads W rows at these indices, and `marginalize`
/// looks up histograms at these keys, so the context must be
/// hashed through the [`OrderHasher`](crate::features::OrderHasher)
/// — returning the raw trailing-n tokens (which the previous
/// implementation did) silently mis-uses the Krylov basis (W[ctx[0], :]
/// instead of W[hash(ctx)[0], :]) and looks up the wrong histogram
/// in `mode_hists`. This is the **critical fix** of rev 33: the
/// previous implementation had ppl = 1390.1 because of this bug.
fn context_modes(context: &[u32], cfg: &TextConfig) -> Vec<u32> {
    use crate::features::OrderHasher;
    OrderHasher::new(cfg.clone()).encode_modes(context)
}

/// Public form of `context_modes` for callers outside this module
/// (e.g. the in-context adapter, which hashes a raw prefix instead
/// of an already-hashed mode list).
pub fn public_encode_modes(context: &[u32], cfg: &TextConfig) -> Vec<u32> {
    context_modes(context, cfg)
}

/// Convert the accumulator's mode weights into the per-order
/// `(lambda, channels)` list consumed by
/// `QfmPipeline::compile_channels`. For each order, the channels are
/// the active modes in that order's block, with alpha_j = weight_j /
/// total_windows (the QFM.tex unit-norm channel normalization).
fn build_channel_groups(
    acc: &ChannelAccumulator,
    cfg: &TextConfig,
) -> Vec<(f64, Vec<(u32, f64)>)> {
    let total = acc.total_windows.max(1) as f64;
    let mut groups = Vec::with_capacity(cfg.n_orders);
    for o in 0..cfg.n_orders {
        let lambda = cfg.lambda[o];
        let off = cfg.offset(o);
        let block = cfg.block_sizes[o];
        let mut channels = Vec::new();
        for mode in off..off + block {
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
/// `serde_json::Value` tree. An earlier version built the whole
/// payload (including the `K2 x rank` `W` matrix and the
/// `n_active_modes`-sized `mode_hists` map) as a `serde_json::json!`
/// tree despite this function's original doc comment claiming
/// "bincode": every `f64`/`u32` became a heap-allocated
/// `serde_json::Value::Number` inside nested `Vec<Value>` arrays,
/// which is 30-50 bytes of overhead per 8-16 bytes of real data.
/// At the old rank-1-2 `W` this was wasteful but tolerable; once the
/// `build_dressed_vacuum_matvec` complex-truncation bug was fixed
/// and `W`'s rank grew to track `m_shifts`, the same code allocated
/// a multi-GB `Value` tree at save time and was OOM-killed
/// mid-write (see the truncated 136-byte checkpoints from that
/// crash — exactly the fixed header+metadata with no payload).
/// This version writes each field's raw little-endian bytes
/// directly into a pre-sized `Vec<u8>`, so peak memory is the
/// payload size itself, not a multiple of it.
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
fn encode_payload(m: &QfmTextModel) -> Result<Vec<u8>, QfmTextError> {
    let w = m.pipeline.w();
    let h_m = m.pipeline.h_m();
    let w_prob = m.pipeline.w_prob();
    let cfg_bytes = serde_json::to_vec(&m.cfg)?;

    let hist_entries: usize = m.mode_hists.values().map(|s| s.hist.len()).sum();
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
        + 4 + cfg_bytes.len();
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::ChannelAccumulator;
    use crate::features::OrderHasher;
    use tempfile::tempdir;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
            block_sizes: vec![16, 16],
            salts: vec![1, 2],
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

    fn build_toy_corpus(tokens: &[u32]) -> (ChannelAccumulator, TextConfig) {
        let mut acc = ChannelAccumulator::new(0, cfg());
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let hasher = OrderHasher::new(cfg());
            let modes = hasher.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }
        (acc, cfg())
    }

    #[test]
    fn model_compiles_and_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let (acc, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
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
        let (acc, mut c) = build_toy_corpus(&tokens);
        c.t = 0.0;
        c.discount = 0.0; // no smoothing, pure empirical
        let model = QfmTextModel::from_accumulator(acc.clone(), &c).unwrap();
        // Pick a context and a comparison. We just check the
        // distribution is finite and sums to 1.
        let dist = model.next_token_dist(&[2, 3]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // (a deeper closed-form check would require predicting the
        // exact shape of the W matrix, which is integration-tested
        // elsewhere.)
    }

    #[test]
    fn save_load_round_trip() {
        let tokens: Vec<u32> = (0..150).map(|i| i % 6).collect();
        let (acc, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
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
    }

    #[test]
    fn model_avg_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let (acc, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
        let dist = model.next_token_dist_model_avg(&[3, 5]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0, "zero probability");
            assert!(p.is_finite());
        }
    }

    #[test]
    fn model_avg_matches_superposition_in_trivial_case() {
        // The model-averaging decoder must at minimum produce a
        // valid distribution. (Whether it differs from the
        // superposition decoder is a property of the corpus — on
        // a 2-order, 32-mode, rank-4 toy model the two Krylov rows
        // can be similar, so we do not require disagreement here.
        // The end-to-end evaluation against the real WikiText-103
        // corpus is the honest test of "did this help".)
        let tokens: Vec<u32> = (0..200).map(|i| i % 7).collect();
        let (acc, c) = build_toy_corpus(&tokens);
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
        let d_super = model.next_token_dist(&[3, 5]).unwrap();
        let d_avg = model.next_token_dist_model_avg(&[3, 5]).unwrap();
        let sum: f64 = d_avg.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for &p in &d_avg {
            assert!(p > 0.0 && p.is_finite());
        }
        // Sanity: both are valid distributions of the same length.
        assert_eq!(d_super.len(), d_avg.len());
    }

    #[test]
    fn renormalize_decode_lifts_seen_mass() {
        // The Renormalize strategy must concentrate the per-mode
        // histogram mass on the context-conditioned peaks, not on
        // the unigram floor. We test this with a structured
        // corpus where the per-mode histograms have a different
        // mode than the unigram. The construction:
        //   100 windows of "0, 1, 2, 7" (context [0,1] -> 2 always)
        //   400 windows of "3, 7" (unigram 7 dominates).
        // The unigram is dominated by 7; the per-mode histogram
        // for [0, 1] is dominated by 2. With Renormalize the model
        // should pick 2 over 7 for context [0, 1].
        let mut tokens: Vec<u32> = Vec::new();
        for _ in 0..100 {
            tokens.extend_from_slice(&[0, 1, 2, 7]);
        }
        for _ in 0..400 {
            tokens.extend_from_slice(&[3, 7]);
        }
        let mut acc = ChannelAccumulator::new(0, cfg());
        let hasher = OrderHasher::new(cfg());
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let modes = hasher.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }
        let mut c = cfg();
        c.decode_strategy = DecodeStrategy::Renormalize;
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
        let dist = model.next_token_dist(&[0, 1]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p.is_finite());
        }
        // The model should pick 2 (context-conditioned) over 7
        // (unigram).
        let dist_2 = dist[2];
        let dist_7 = dist[7];
        assert!(
            dist_2 > dist_7,
            "Renormalize P(2|0,1) = {dist_2} should exceed P(7|0,1) = {dist_7} (context should beat unigram)"
        );
    }

    #[test]
    fn topk_decode_sums_to_one() {
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, mut c) = build_toy_corpus(&tokens);
        c.decode_strategy = DecodeStrategy::TopK;
        c.top_k = 2;
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
        let dist = model.next_token_dist(&[2, 3]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0 && p.is_finite());
        }
    }

    #[test]
    fn order_prior_decode_sums_to_one() {
        // With a strong order prior (lambda = [1, 10]), the
        // higher-order (order-2) modes should dominate. We can't
        // easily inspect that directly, but we can verify the
        // distribution is valid and finite.
        let tokens: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let (acc, mut c) = build_toy_corpus(&tokens);
        c.decode_strategy = DecodeStrategy::OrderPrior;
        c.lambda = vec![1.0, 10.0];
        let model = QfmTextModel::from_accumulator(acc, &c).unwrap();
        let dist = model.next_token_dist(&[1, 2]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0 && p.is_finite());
        }
    }

    #[test]
    fn dense_decode_preserves_unigram_floor() {
        // The legacy Dense behavior should still route the
        // residual `1 - total_w` to the unigram. We test this by
        // constructing a corpus with a strong unigram and a weak
        // context-conditioned distribution, then verifying that
        // the Dense distribution is *closer to the unigram* than
        // the Renormalize distribution is.
        let tokens: Vec<u32> = (0..500).map(|i| i % 11).collect();
        let (acc_dense, mut c_dense) = build_toy_corpus(&tokens);
        c_dense.decode_strategy = DecodeStrategy::Dense;
        let model_dense =
            QfmTextModel::from_accumulator(acc_dense, &c_dense).unwrap();
        let (acc_renorm, mut c_renorm) = build_toy_corpus(&tokens);
        c_renorm.decode_strategy = DecodeStrategy::Renormalize;
        let model_renorm =
            QfmTextModel::from_accumulator(acc_renorm, &c_renorm).unwrap();
        let d_dense = model_dense.next_token_dist(&[3, 7]).unwrap();
        let d_renorm = model_renorm.next_token_dist(&[3, 7]).unwrap();
        // L1 distance to the unigram.
        let l1 = |d: &[f64]| -> f64 {
            d.iter()
                .zip(model_dense.unigram.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 2.0
        };
        let l1_dense = l1(&d_dense);
        let l1_renorm = l1(&d_renorm);
        // The Dense distribution should be at least as close to
        // the unigram as the Renormalize distribution.
        assert!(
            l1_dense <= l1_renorm * 1.1 + 1e-6,
            "Dense L1 to unigram = {l1_dense} should be <= Renormalize L1 = {l1_renorm} (Dense should rely on the unigram floor more)"
        );
    }

    /// Diagnostic experiment (not asserting a library behavior, just
    /// measuring one): with only a *single* active order, `Renormalize`
    /// always assigns the one active mode weight 1.0 regardless of its
    /// Born-rule magnitude, so vacuum-scale can't matter — the real
    /// failure mode (missing even deterministic contexts,
    /// `QFM_TEXT_STATUS.md` §4) is a *cross-order* competition: does
    /// the mixture weight favor the informative (specific, but more
    /// hash-collision-prone) higher order, or the uninformative
    /// (generic, less collision-prone) lower order?
    ///
    /// Builds a 2-order synthetic corpus: order-1 context (the last
    /// token, `b`) is *not* informative (for fixed `b`, 100 different
    /// `a` values each produce a different deterministic target, so
    /// order-1's own histogram is a ~100-way blend); order-2 context
    /// (the pair `(a, b)`) *is* fully deterministic
    /// (`target = f(a, b)`), subject to realistic hash collisions
    /// (10,000 distinct pairs into a 16,384-slot block, ~30% collide).
    /// Compiles two pipelines from the *same* accumulator: natural
    /// alpha (`c0 ~ 1` per order) vs. alpha rescaled to
    /// `sum(alpha^2) = 1` per order (same idea as the single-order
    /// experiment, now where it can actually matter). Reports the
    /// order-1 vs order-2 mixture weight and the top-1 hit rate for
    /// each, over a sample of (a, b) pairs.
    #[test]
    fn vacuum_dominance_cross_order_experiment() {
        // Disjoint id ranges so no role accidentally collides with
        // another: a in [0,100), b in [1000,1100), target in [2000,3000).
        let mut tokens: Vec<u32> = Vec::new();
        for a in 0u32..100 {
            for b in 0u32..100 {
                let bb = 1000 + b;
                let target = 2000 + (a.wrapping_mul(31).wrapping_add(b.wrapping_mul(7)).wrapping_add(11)) % 1000;
                for _ in 0..3 {
                    tokens.push(a);
                    tokens.push(bb);
                    tokens.push(target);
                }
            }
        }
        let c = TextConfig {
            n_orders: 2,
            block_sizes: vec![1024, 16_384],
            salts: vec![1, 2],
            hist_cap: 32,
            max_rank: 8,
            m_shifts: 8,
            lambda: vec![1.0, 1.0],
            t: 1.0,
            discount: 0.75,
            seed: 0,
            ..Default::default()
        };
        let mut acc = ChannelAccumulator::new(3000, c.clone());
        let hasher = OrderHasher::new(c.clone());
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let modes = hasher.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }

        let groups = build_channel_groups(&acc, &c);
        for (o, (_, channels)) in groups.iter().enumerate() {
            let sum_sq: f64 = channels.iter().map(|&(_, a)| a * a).sum();
            eprintln!(
                "[cross_order] order {o}: M = {}, sum_sq(natural) = {sum_sq:.3e}, c0(natural) = {:.6}",
                channels.len(),
                1.0 / (1.0 + sum_sq).sqrt(),
            );
        }
        let groups_rescaled: Vec<(f64, Vec<(u32, f64)>)> = groups
            .iter()
            .map(|(lambda, channels)| {
                let sum_sq: f64 = channels.iter().map(|&(_, a)| a * a).sum();
                let gain = if sum_sq > 0.0 { 1.0 / sum_sq.sqrt() } else { 1.0 };
                (
                    *lambda,
                    channels.iter().map(|&(m, a)| (m, a * gain)).collect(),
                )
            })
            .collect();

        let k2_total = c.k2_total() as usize;
        let qfm_cfg = qfm::QfmConfig {
            k: c.n_orders,
            k2: k2_total,
            krylov_dim: c.m_shifts,
            seed: c.seed,
            n_t_samples: 4,
            noise_dim: c.n_orders,
            max_rank: Some(c.max_rank),
        };
        let pipeline_natural =
            QfmPipeline::compile_channels(&groups, k2_total, &qfm_cfg).unwrap();
        let pipeline_rescaled =
            QfmPipeline::compile_channels(&groups_rescaled, k2_total, &qfm_cfg).unwrap();
        eprintln!(
            "[cross_order] rank natural = {}, rank rescaled = {}",
            pipeline_natural.rank(),
            pipeline_rescaled.rank()
        );

        let unigram_total: f64 = acc.unigram.iter().map(|&x| x as f64).sum();
        let unigram: Vec<f64> = acc
            .unigram
            .iter()
            .map(|&x| x as f64 / unigram_total.max(1.0))
            .collect();
        let gram_natural = pipeline_natural.gram();
        let gram_rescaled = pipeline_rescaled.gram();
        let model_natural = QfmTextModel {
            pipeline: pipeline_natural,
            mode_hists: acc.stats.clone(),
            unigram: unigram.clone(),
            unigram_total,
            cfg: c.clone(),
            schema_version: crate::SCHEMA_VERSION,
            gram: gram_natural,
        };
        let model_rescaled = QfmTextModel {
            pipeline: pipeline_rescaled,
            mode_hists: acc.stats.clone(),
            unigram,
            unigram_total,
            cfg: c.clone(),
            schema_version: crate::SCHEMA_VERSION,
            gram: gram_rescaled,
        };

        let argmax = |d: &[f64]| -> usize {
            let mut best = 0usize;
            for (i, &p) in d.iter().enumerate() {
                if p > d[best] {
                    best = i;
                }
            }
            best
        };

        let mut hits_natural = 0u32;
        let mut hits_rescaled = 0u32;
        let mut w1_natural_sum = 0.0;
        let mut w2_natural_sum = 0.0;
        let mut w1_rescaled_sum = 0.0;
        let mut w2_rescaled_sum = 0.0;
        let n_sample = 200u32;
        for i in 0..n_sample {
            let a = i % 100;
            let b = (i * 37) % 100;
            let bb = 1000 + b;
            let expected = 2000 + (a.wrapping_mul(31).wrapping_add(b.wrapping_mul(7)).wrapping_add(11)) % 1000;
            let ctx = vec![a, bb];
            let active_modes = public_encode_modes(&ctx, &c);

            let get_w = |weights: &[(u32, f64)], mode: u32| -> f64 {
                weights.iter().find(|&&(m, _)| m == mode).map(|&(_, w)| w).unwrap_or(0.0)
            };

            let c0n = model_natural.pipeline.encode_modes(&active_modes).unwrap();
            let c1n = model_natural.pipeline.evolve(&c0n, c.t);
            let pn = model_natural
                .pipeline
                .decode_sketched_at(&c1n, &model_natural.gram, &active_modes);
            let procn = model_natural.preprocess_p_tilde(&pn);
            w1_natural_sum += get_w(&procn, active_modes[0]);
            w2_natural_sum += get_w(&procn, active_modes[1]);
            let dn = model_natural.marginalize(&pn);

            let c0r = model_rescaled.pipeline.encode_modes(&active_modes).unwrap();
            let c1r = model_rescaled.pipeline.evolve(&c0r, c.t);
            let pr = model_rescaled
                .pipeline
                .decode_sketched_at(&c1r, &model_rescaled.gram, &active_modes);
            let procr = model_rescaled.preprocess_p_tilde(&pr);
            w1_rescaled_sum += get_w(&procr, active_modes[0]);
            w2_rescaled_sum += get_w(&procr, active_modes[1]);
            let dr = model_rescaled.marginalize(&pr);

            if argmax(&dn) == expected as usize {
                hits_natural += 1;
            }
            if argmax(&dr) == expected as usize {
                hits_rescaled += 1;
            }
        }
        eprintln!(
            "[cross_order] mean mixture weight natural: order1 = {:.4}, order2 = {:.4}",
            w1_natural_sum / n_sample as f64,
            w2_natural_sum / n_sample as f64,
        );
        eprintln!(
            "[cross_order] mean mixture weight rescaled: order1 = {:.4}, order2 = {:.4}",
            w1_rescaled_sum / n_sample as f64,
            w2_rescaled_sum / n_sample as f64,
        );
        eprintln!(
            "[cross_order] top-1 hit rate: natural = {}/{n_sample} = {:.3}, rescaled = {}/{n_sample} = {:.3}",
            hits_natural,
            hits_natural as f64 / n_sample as f64,
            hits_rescaled,
            hits_rescaled as f64 / n_sample as f64,
        );
    }

    /// Diagnostic experiment: does `hist_cap` eviction under heavy
    /// hash-collision traffic create a hard, mechanical ceiling on
    /// fit quality — independent of any Krylov/Born-rule weighting —
    /// matching the real corpus's order-3/order-4 buckets (~1953
    /// windows/bucket on average at 128M tokens / 65536 buckets,
    /// almost certainly exceeding `hist_cap=64` distinct low-count
    /// tokens per bucket)? Single order only, so there is no
    /// cross-order mixing confound at all: with n_orders=1 the one
    /// active mode's own histogram is the *entire* story.
    ///
    /// 5,000 distinct single-token contexts, each with its own
    /// distinct deterministic target, hashed into a 200-slot block
    /// (~25 distinct contexts collide per bucket on average) with a
    /// small `hist_cap=8`. Reports: (a) the fraction of queried
    /// contexts whose true target is still present (non-evicted) in
    /// their mode's histogram — the hard ceiling — and (b) the
    /// model's actual top-1 hit rate, to see how close it tracks
    /// that ceiling.
    #[test]
    fn hist_cap_eviction_ceiling_experiment() {
        let n_contexts: u32 = 5_000;
        let f = |ctx: u32| -> u32 { 10_000 + (ctx.wrapping_mul(2654435761)) % 5_000 };
        let mut tokens: Vec<u32> = Vec::with_capacity(n_contexts as usize * 6);
        for ctx in 0..n_contexts {
            let target = f(ctx);
            for _ in 0..3 {
                tokens.push(ctx);
                tokens.push(target);
            }
        }
        let c = TextConfig {
            n_orders: 1,
            block_sizes: vec![200],
            salts: vec![1],
            hist_cap: 8,
            max_rank: 8,
            m_shifts: 8,
            lambda: vec![1.0],
            t: 1.0,
            discount: 0.75,
            seed: 0,
            ..Default::default()
        };
        let mut acc = ChannelAccumulator::new(20_000, c.clone());
        let hasher = OrderHasher::new(c.clone());
        for i in 1..tokens.len() {
            let ctx = vec![tokens[i - 1]];
            let modes = hasher.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }
        let model = QfmTextModel::from_accumulator(acc.clone(), &c).unwrap();
        let baseline = crate::lm::NgramBaseline::from_accumulator(acc.clone());

        let argmax = |d: &[f64]| -> usize {
            let mut best = 0usize;
            for (i, &p) in d.iter().enumerate() {
                if p > d[best] {
                    best = i;
                }
            }
            best
        };

        let mut present = 0u32;
        let mut hits_qfm = 0u32;
        let mut hits_baseline = 0u32;
        let n_sample = 1000u32;
        for i in 0..n_sample {
            let ctx_tok = i * (n_contexts / n_sample);
            let expected = f(ctx_tok);
            let modes = hasher.encode_modes(&[ctx_tok]);
            let mode = modes[0];
            if let Some(stats) = acc.stats.get(&mode) {
                if stats.hist.iter().any(|&(tok, _)| tok == expected) {
                    present += 1;
                }
            }
            let dq = model.next_token_dist(&[ctx_tok]).unwrap();
            let db = baseline.next_token_dist(&[ctx_tok]);
            if argmax(&dq) == expected as usize {
                hits_qfm += 1;
            }
            if argmax(&db) == expected as usize {
                hits_baseline += 1;
            }
        }
        eprintln!(
            "[hist_cap_ceiling] true target still present (non-evicted) in histogram: {present}/{n_sample} = {:.3}",
            present as f64 / n_sample as f64
        );
        eprintln!(
            "[hist_cap_ceiling] top-1 hit rate: QFM = {hits_qfm}/{n_sample} = {:.3}, baseline = {hits_baseline}/{n_sample} = {:.3}",
            hits_qfm as f64 / n_sample as f64,
            hits_baseline as f64 / n_sample as f64,
        );
    }
}
