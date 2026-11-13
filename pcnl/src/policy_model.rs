//! R7.3.2 — the decision transformer: a sequence model whose output is a
//! distribution over actions, not a label.
//!
//! The posterior machinery does not change here.  What changes is the
//! likelihood: instead of one categorical cross-entropy per example, a
//! trajectory contributes *one per timestep*, and the model conditions each on
//! the history and on the return-to-go:
//!
//! ```text
//! Φ(θ) = λ · N · mean_traj,t [ −log p_θ(a_t | o_{≤t}, a_{<t}, g_{≤t}) ] + ½‖v‖²
//! ```
//!
//! with `N` counting **transitions** (see [`DtPotential::n_terms`]) while the
//! independent unit of the data is the *trajectory* — the distinction
//! `RESEARCH_PLAN.md` §R7.1 requires every DT cell to state.
//!
//! # Sequence layout
//!
//! One timestep is three slots, interleaved with no separator token:
//!
//! ```text
//! slot   3t     3t+1      3t+2
//!       g_t    o_t        a_t        (t = 0..T-1)
//! ```
//!
//! The prediction for step `t` is the head's output **at the state slot**
//! `3t+1`, i.e. before `a_t` has entered the sequence: causal attention over
//! slots `≤ 3t+1` cannot see slot `3t+2`.  That is why the head may emit logits
//! at every slot and the loss selects only the state slots — a slot-level mask
//! is not needed for correctness, only a timestep mask for the loss and for
//! padding.
//!
//! # Padding
//!
//! Sequences are padded on the right, and [`DtBatch::new`] rejects a mask that
//! is not a prefix (`1…1 0…0` per row).  With that invariant no key mask is
//! needed: a real slot at `t` attends exactly the real slots `≤ t`, because
//! every padded slot comes later.  Padded slots contribute nothing to the loss
//! and their embeddings are never read by a real position, which
//! `padding_content_never_reaches_a_real_step` pins bitwise.  A ragged batch
//! still has to be evaluated; the alternative (one sequence per forward pass)
//! changes the batch shape and with it the numerics, so the mask is the honest
//! choice and its invariance is a test, not a comment.
//!
//! # What this module deliberately does not do
//!
//! The head is softmax over a discrete action vocabulary.  A Gaussian
//! `(μ, log σ)` head, a quantile head and two-hot binning are R7.6's second arm,
//! and the reason is in §R7.1: an unbounded output parametrisation can drive a
//! per-step NLL to `−∞` along directions the prior does not punish, which is a
//! target defect a sampler cannot fix.  Adding that head without the `σ` floor
//! and the proper-density test would put a knowingly invalid potential under the
//! kernel.
//!
//! No biases and no layer-norm affine, matching [`crate::model`]: `P` stays
//! derived from the shapes alone, and the two models' parameter counts stay
//! comparable at the same `d`.  The head is *not* tied to the action embedding
//! (the input space and the output space are different objects here, unlike the
//! classifier's tied head).

use crate::model::{LayerSlices, Variant, layer};
use crate::normalize::{NormGrad, normalize_last};
use crate::target::Potential;
use candle_core::Var;
use candle_core::{D, Device, Result, Tensor};

/// Static shapes of the toy decision transformer. `P` is derived, never stored
/// twice.
///
/// `max_len` is the number of *timesteps*; the sequence has `3 · max_len`
/// slots.  It is a shape because the position table has one row per timestep —
/// the classifier had no sequence axis and so no such parameter, and it is the
/// first place a DT's `P` grows with the horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DtShapes {
    pub rtg_bins: usize,
    pub state_dim: usize,
    pub action_vocab: usize,
    pub d: usize,
    pub ff: usize,
    pub layers: usize,
    pub max_len: usize,
}

impl DtShapes {
    /// RTG table + state encoder + action table + position table + per-layer
    /// transformer blocks + untied head.
    pub fn flat_size(&self) -> usize {
        let per_layer = 4 * self.d * self.d + 2 * self.d * self.ff;
        self.rtg_bins * self.d
            + self.state_dim * self.d
            + self.action_vocab * self.d
            + self.max_len * self.d
            + self.layers * per_layer
            + self.d * self.action_vocab
    }

    /// Parameter layout as `(start, len, shape)` in a fixed order:
    /// `e_rtg [G, D]`, `w_in [S, D]`, `e_act [A, D]`, `e_pos [T, D]`, then per
    /// layer `wq, wk, wv, wo [D, D]`, `w1 [D, F]`, `w2 [F, D]`, then the head
    /// `w_out [D, A]`.
    pub fn layout(&self) -> Vec<(usize, usize, Vec<usize>)> {
        let mut out = Vec::new();
        let mut at = 0usize;
        let mut push = |rows: usize, cols: usize| {
            out.push((at, rows * cols, vec![rows, cols]));
            at += rows * cols;
        };
        push(self.rtg_bins, self.d);
        push(self.state_dim, self.d);
        push(self.action_vocab, self.d);
        push(self.max_len, self.d);
        for _ in 0..self.layers {
            push(self.d, self.d); // wq
            push(self.d, self.d); // wk
            push(self.d, self.d); // wv
            push(self.d, self.d); // wo
            push(self.d, self.ff); // w1
            push(self.ff, self.d); // w2
        }
        push(self.d, self.action_vocab); // w_out
        debug_assert_eq!(at, self.flat_size());
        out
    }
}

/// Named views over the flat vector.  The per-layer blocks are
/// [`LayerSlices`], the *same* struct the classifier's layers use, so the
/// transformer stack itself is one implementation.
pub struct DtSlices {
    pub e_rtg: Tensor,
    pub w_in: Tensor,
    pub e_act: Tensor,
    pub e_pos: Tensor,
    pub layers: Vec<LayerSlices>,
    pub w_out: Tensor,
}

/// Unflatten `θ ∈ [P]` into named tensors (`narrow` + `reshape`, so gradients
/// flow back to the caller's leaf).
pub fn unflatten_dt(theta: &Tensor, shapes: &DtShapes) -> Result<DtSlices> {
    let mut tensors = Vec::new();
    for (start, len, shape) in shapes.layout() {
        tensors.push(theta.narrow(0, start, len)?.reshape(shape)?);
    }
    let mut it = tensors.into_iter();
    let e_rtg = it.next().expect("layout has an RTG table");
    let w_in = it.next().expect("layout has a state encoder");
    let e_act = it.next().expect("layout has an action table");
    let e_pos = it.next().expect("layout has a position table");
    let mut layers = Vec::with_capacity(shapes.layers);
    for _ in 0..shapes.layers {
        layers.push(LayerSlices {
            wq: it.next().unwrap(),
            wk: it.next().unwrap(),
            wv: it.next().unwrap(),
            wo: it.next().unwrap(),
            w1: it.next().unwrap(),
            w2: it.next().unwrap(),
        });
    }
    let w_out = it.next().expect("layout has a head");
    Ok(DtSlices {
        e_rtg,
        w_in,
        e_act,
        e_pos,
        layers,
        w_out,
    })
}

/// One batch of trajectories, already tokenised.
///
/// `rtg` and `act` are `[B, T]` index tensors, `state` is `[B, T, state_dim]`,
/// and `mask` is `[B, T]` with `1` at real timesteps.  Every row's mask must be
/// a prefix (see the module note on padding).
///
/// The *split* is not here: `mdp` owns corpus construction and splits by
/// trajectory seed, because a step-level split leaks across time and no amount
/// of masking in this module can undo that.
#[derive(Clone)]
pub struct DtBatch {
    pub rtg: Tensor,
    pub state: Tensor,
    pub act: Tensor,
    pub mask: Tensor,
}

impl DtBatch {
    pub fn new(rtg: Tensor, state: Tensor, act: Tensor, mask: Tensor) -> Result<Self> {
        let (b, t) = mask.dims2()?;
        if rtg.dims2()? != (b, t) || act.dims2()? != (b, t) {
            return Err(candle_core::Error::Msg(format!(
                "DtBatch: rtg/act must be [{b},{t}], got {:?} and {:?}",
                rtg.dims(),
                act.dims()
            )));
        }
        if state.dims3()?.0 != b || state.dims3()?.1 != t {
            return Err(candle_core::Error::Msg(format!(
                "DtBatch: state must be [{b},{t},S], got {:?}",
                state.dims()
            )));
        }
        let m = mask.flatten_all()?.to_vec1::<f32>()?;
        for row in 0..b {
            let mut seen_pad = false;
            for col in 0..t {
                let real = m[row * t + col] > 0.5;
                if !real {
                    seen_pad = true;
                } else if seen_pad {
                    return Err(candle_core::Error::Msg(format!(
                        "DtBatch: row {row} mask is not a prefix (hole at t={col}); \
                         padded slots may only follow real ones, or causal attention \
                         would let a real step read garbage"
                    )));
                }
            }
        }
        Ok(Self {
            rtg,
            state,
            act,
            mask,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.mask.dims2().map(|d| d.0).unwrap_or(0)
    }

    pub fn timesteps(&self) -> usize {
        self.mask.dims2().map(|d| d.1).unwrap_or(0)
    }

    /// Real transitions in the batch — the count `Φ` sums over.  Deliberately
    /// *not* the parameter count and not the trajectory count; both are read
    /// separately (`batch_size`, `n_terms`) because a likelihood whose unit is
    /// a 16-step trajectory is not 16 independent observations.
    pub fn transitions(&self) -> usize {
        self.mask
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map(|m| m.iter().filter(|&&x| x > 0.5).count())
            .unwrap_or(0)
    }

    /// Flattened indices of `(b, t)` state slots whose step is real, and the
    /// action ids those steps predict.  Shared by the loss and any per-step
    /// readout, so a readout cannot disagree with the loss about which steps
    /// count.  Public for exactly that reason: `policy_readout` selects steps
    /// through this function rather than re-deriving the `3t + 1` rule.
    pub fn scored_steps(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        let (b, t) = self.mask.dims2()?;
        let slots = 3 * t;
        let m = self.mask.flatten_all()?.to_vec1::<f32>()?;
        let act = self.act.flatten_all()?.to_vec1::<u32>()?;
        let mut idx = Vec::new();
        let mut targets = Vec::new();
        for row in 0..b {
            for step in 0..t {
                if m[row * t + step] > 0.5 {
                    idx.push((row * slots + 3 * step + 1) as u32);
                    targets.push(act[row * t + step]);
                }
            }
        }
        Ok((idx, targets))
    }
}

/// Causal DT forward over the interleaved `(g, o, a)` slots, returning
/// `[B, 3T, action_vocab]` logits.
///
/// The step-`t` prediction is the row at slot `3t+1`; the loss selects those
/// rows through [`DtBatch::scored_steps`].  Values do not depend on the
/// derivative convention; [`dt_forward_norm`] exposes it for the exact-Jacobian
/// path, exactly as [`crate::model::forward`] does.
pub fn dt_forward(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
    norm: NormGrad,
) -> Result<Tensor> {
    let s = unflatten_dt(theta, shapes)?;
    let (b, t) = batch.mask.dims2()?;
    if t > shapes.max_len {
        return Err(candle_core::Error::Msg(format!(
            "dt_forward: batch has {t} timesteps, shapes allow {}",
            shapes.max_len
        )));
    }
    if batch.state.dim(2)? != shapes.state_dim {
        return Err(candle_core::Error::Msg(format!(
            "dt_forward: state dim {} != shapes.state_dim {}",
            batch.state.dim(2)?,
            shapes.state_dim
        )));
    }

    let gather = |table: &Tensor, ids: &Tensor| -> Result<Tensor> {
        let flat = ids.flatten_all()?;
        table.index_select(&flat, 0)?.reshape((b, t, table.dim(1)?))
    };
    // Slot embeddings: g_t from the RTG table, o_t through the state encoder,
    // a_t from the action table.  No bias and no per-slot scale: the slot type
    // is already carried by which block produced the vector.
    let eg = gather(&s.e_rtg, &batch.rtg)?;
    let es = batch.state.broadcast_matmul(&s.w_in)?;
    let ea = gather(&s.e_act, &batch.act)?;
    // [B, T, 3, D] → [B, 3T, D]: the timestep-major order of the module docs.
    let mut x = Tensor::stack(&[&eg, &es, &ea], 2)?.reshape((b, 3 * t, shapes.d))?;
    // One position row per timestep, added to all three of its slots.
    let pos = s
        .e_pos
        .narrow(0, 0, t)?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .broadcast_as((b, t, 3, shapes.d))?
        .reshape((b, 3 * t, shapes.d))?;
    x = x.broadcast_add(&pos)?;

    let tau = (shapes.d as f64).sqrt();
    match variant {
        // The cosine variant's invariant is a relation between token vectors
        // on the sphere, so the mixed slots are projected once before the stack
        // and each layer re-normalises as usual.  This arm's formal claims
        // (M2/M4) have not been re-derived for slot-type-specific blocks — it is
        // the plan's second arm, not a default.
        Variant::Cosine => x = normalize_last(&x)?,
        Variant::Standard => {}
    }
    for w in &s.layers {
        x = layer(&x, w, variant, tau, norm)?;
    }
    x.broadcast_matmul(&s.w_out)
}

/// [`dt_forward`] with an explicit `NormGrad` (the value is convention-free;
/// the derivative is not).
pub fn dt_forward_norm(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
    norm: NormGrad,
) -> Result<Tensor> {
    dt_forward(theta, shapes, variant, batch, norm)
}

/// Per-step action distributions: softmax over the head's last dimension,
/// `[B, 3T, action_vocab]`.
///
/// Returned for *every* slot, because the caller may want the distribution at
/// any position; the scored steps are [`DtBatch::scored_steps`]' contract, and a
/// readout that scores a different set of steps than the loss is comparing two
/// different objects.
pub fn dt_probs(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
) -> Result<Tensor> {
    let logits = dt_forward(theta, shapes, variant, batch, NormGrad::Stopped)?;
    candle_nn::ops::softmax(&logits, D::Minus1)
}

/// The DT's action distribution at step `t` of a single rollout.
///
/// `states` has `t + 1` entries (`states[t]` is the state the policy is acting
/// in), `actions` the `t` actions already taken, and `rtgs` the `t + 1`
/// conditioning bins (indices from [`crate::mdp::DtSpec::bin`], not raw
/// returns, because the embedding table's rows are bins).  Per-step bins and
/// not one value: a training sequence's return-to-go falls as the step index
/// rises, so a rollout that conditioned on a constant bin at every step would
/// be asking a row the corpus never visited.
///
/// Only the accumulated prefix is fed in. The forward pass is causal, so row
/// `3t + 1` here is the same row the loss scores at step `t` on a training
/// sequence with the same prefix — which is what lets a rollout be read out by
/// the same functions as the training batch instead of by a second, subtly
/// different inference path.
///
/// The action slot at step `t` is a placeholder: the prediction at `3t + 1` is
/// the distribution *over* `a_t`, so no action at `t` may be visible to it.
///
/// The sequence is always evaluated at the model's full `max_len`, with the
/// real prefix masked in and zeros after it.  That is not padding for
/// convenience: the attention denominator's length changes with the batch
/// shape, so a shorter sequence would give a row that agrees only to a few
/// digits (the same row computed with a different number of `−∞` keys and a
/// different matmul shape).  At fixed length the future slots cannot reach row
/// `3t + 1` at all — causal masking sends them `−∞` — and the row is the one a
/// training batch produces, which `dt_act_is_the_row_the_loss_scores` checks.
pub fn dt_act(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    rtgs: &[u32],
    states: &[[f32; 2]],
    actions: &[u32],
) -> Result<Vec<f64>> {
    let t = actions.len();
    if rtgs.len() != t + 1 {
        return Err(candle_core::Error::Msg(format!(
            "dt_act: {} rtg bins for {t} actions (expected {})",
            rtgs.len(),
            t + 1
        )));
    }
    if states.len() != t + 1 {
        return Err(candle_core::Error::Msg(format!(
            "dt_act: {} states for {t} actions (expected {})",
            states.len(),
            t + 1
        )));
    }
    if t + 1 > shapes.max_len {
        return Err(candle_core::Error::Msg(format!(
            "dt_act: step {t} outside a {}-timestep model",
            shapes.max_len
        )));
    }
    let dev = theta.device();
    let n = shapes.max_len;
    let mut rtg_all = vec![rtgs[0]; n];
    rtg_all[..rtgs.len()].copy_from_slice(rtgs);
    let mut st = vec![0.0f32; n * 2];
    let mut acts = vec![0u32; n];
    let mut mask = vec![0.0f32; n];
    for (i, s) in states.iter().enumerate() {
        st[2 * i] = s[0];
        st[2 * i + 1] = s[1];
        mask[i] = 1.0;
    }
    for (i, &a) in actions.iter().enumerate() {
        acts[i] = a;
    }
    let batch = DtBatch::new(
        Tensor::from_vec(rtg_all, (1, n), dev)?,
        Tensor::from_vec(st, (1, n, 2), dev)?,
        Tensor::from_vec(acts, (1, n), dev)?,
        Tensor::from_vec(mask, (1, n), dev)?,
    )?;
    let probs = dt_probs(theta, shapes, variant, &batch)?;
    let row = probs
        .reshape((3 * n, shapes.action_vocab))?
        .narrow(0, 3 * t + 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(row.into_iter().map(f64::from).collect())
}

/// Mean per-transition NLL, forward only — the value half of the target.
///
/// Uses the historical `Stopped` convention through [`dt_forward`], matching
/// [`crate::task::Dataset::train_loss`]; the value does not depend on the
/// convention (only the derivative does), so this is the same number the exact
/// path integrates.
pub fn dt_loss(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
) -> Result<f64> {
    let logits = dt_forward(theta, shapes, variant, batch, NormGrad::Stopped)?;
    let (idx, targets) = batch.scored_steps()?;
    let n = idx.len();
    if n == 0 {
        return Err(candle_core::Error::Msg(
            "dt_loss: no real transitions in the batch".into(),
        ));
    }
    let (b, t) = batch.mask.dims2()?;
    let head = logits.reshape((b * 3 * t, shapes.action_vocab))?;
    let sel = head.index_select(&Tensor::from_vec(idx, n, logits.device())?, 0)?;
    let y = Tensor::from_vec(targets, n, logits.device())?;
    let loss = candle_nn::loss::cross_entropy(&sel, &y)?;
    Ok(loss.to_scalar::<f32>()? as f64)
}

/// `(mean per-transition NLL, ∇_θ of it)` — the loss-and-gradient seam the
/// kernel and the mode search need.
///
/// `NormGrad::Exact` is the convention whose gradient agrees with a finite
/// difference of [`dt_loss`]; the `Stopped` convention keeps the Frobenius
/// denominators constant and is what every recorded anchor was trained with
/// (`RESEARCH_PLAN.md` §R0.1).  The classifier's `grad_phi` takes the same
/// decision for the same reason, so the two potentials are read the same way.
pub fn dt_loss_and_grad(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
    norm: NormGrad,
) -> Result<(f64, Tensor)> {
    let var = Var::from_tensor(theta)?;
    let logits = dt_forward(var.as_tensor(), shapes, variant, batch, norm)?;
    let (idx, targets) = batch.scored_steps()?;
    let n = idx.len();
    if n == 0 {
        return Err(candle_core::Error::Msg(
            "dt_loss_and_grad: no real transitions in the batch".into(),
        ));
    }
    let (b, t) = batch.mask.dims2()?;
    let head = logits.reshape((b * 3 * t, shapes.action_vocab))?;
    let sel = head.index_select(&Tensor::from_vec(idx, n, logits.device())?, 0)?;
    let y = Tensor::from_vec(targets, n, logits.device())?;
    let loss = candle_nn::loss::cross_entropy(&sel, &y)?;
    let grads = loss.backward()?;
    let grad = grads
        .get(var.as_tensor())
        .ok_or_else(|| candle_core::Error::Msg("theta got no gradient".into()))?;
    Ok((loss.to_scalar::<f32>()? as f64, grad.clone()))
}

/// A flat start vector: i.i.d. `N(0, init_std²)` on CPU, seeded.
pub fn init_flat(shapes: &DtShapes, init_std: f64, seed: u64) -> Tensor {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::StandardNormal;
    let v: Vec<f32> = (0..shapes.flat_size())
        .map(|_| {
            let z: f32 = rng.sample(normal);
            (z as f64 * init_std) as f32
        })
        .collect();
    Tensor::from_vec(v, vec![shapes.flat_size()], &Device::Cpu).expect("cpu tensor")
}

/// The DT likelihood as a [`Potential`], so it can sit under
/// [`crate::target::SubspaceTarget`], the pCN kernel, the curvature machinery
/// and the diagnostics with no classifier-specific branch.
///
/// `n_terms` is the number of **transitions**, matching the sum form
/// `λ · N · mean_loss`.  The independent unit is the trajectory, not the
/// transition: `mean_loss` is computed over correlated steps within a
/// trajectory, so its Monte Carlo error is *not* that of `N` independent draws,
/// and any ESS/MCSE report has to use the within-trajectory correlation.  This
/// type does not pretend otherwise — it exposes `batch_size()` and
/// `transitions()` separately.
pub struct DtPotential {
    batch: DtBatch,
    shapes: DtShapes,
    variant: Variant,
}

impl DtPotential {
    pub fn new(batch: DtBatch, shapes: DtShapes, variant: Variant) -> Self {
        Self {
            batch,
            shapes,
            variant,
        }
    }

    pub fn batch(&self) -> &DtBatch {
        &self.batch
    }

    pub fn shapes(&self) -> &DtShapes {
        &self.shapes
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Trajectories in the corpus (the independent unit).
    pub fn trajectories(&self) -> usize {
        self.batch.batch_size()
    }
}

impl Potential for DtPotential {
    fn n_terms(&self) -> usize {
        self.batch.transitions()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        dt_loss(theta, &self.shapes, self.variant, &self.batch)
            .ok()
            .filter(|l| l.is_finite())
            .unwrap_or(f64::INFINITY)
    }

    fn mean_grad_with(&self, theta: &Tensor, norm: NormGrad) -> Option<Tensor> {
        let (loss, grad) =
            dt_loss_and_grad(theta, &self.shapes, self.variant, &self.batch, norm).ok()?;
        if !loss.is_finite() {
            return None;
        }
        Some(grad)
    }

    fn label(&self) -> &'static str {
        "dt-softmax"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn tiny() -> DtShapes {
        DtShapes {
            rtg_bins: 4,
            state_dim: 2,
            action_vocab: 5,
            d: 6,
            ff: 12,
            layers: 1,
            max_len: 3,
        }
    }

    fn batch(shapes: &DtShapes, lens: &[usize], seed: u64) -> DtBatch {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let b = lens.len();
        let t = *lens.iter().max().expect("non-empty");
        let mut rtg = Vec::with_capacity(b * t);
        let mut state = Vec::with_capacity(b * t * shapes.state_dim);
        let mut act = Vec::with_capacity(b * t);
        let mut mask = Vec::with_capacity(b * t);
        for &len in lens {
            for step in 0..t {
                let real = step < len;
                rtg.push(rng.random_range(0..shapes.rtg_bins) as u32);
                act.push(rng.random_range(0..shapes.action_vocab) as u32);
                for _ in 0..shapes.state_dim {
                    state.push(if real {
                        rng.random_range(-1.0f32..1.0)
                    } else {
                        0.0
                    });
                }
                mask.push(if real { 1.0f32 } else { 0.0 });
            }
        }
        let dev = &Device::Cpu;
        DtBatch::new(
            Tensor::from_vec(rtg, (b, t), dev).unwrap(),
            Tensor::from_vec(state, (b, t, shapes.state_dim), dev).unwrap(),
            Tensor::from_vec(act, (b, t), dev).unwrap(),
            Tensor::from_vec(mask, (b, t), dev).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn layout_covers_all_parameters() {
        let s = tiny();
        let total: usize = s.layout().iter().map(|(_, len, _)| *len).sum();
        assert_eq!(total, s.flat_size());
        // P = D(G + S + A + T) + L(4D² + 2DF) + D·A
        let want = 6 * (4 + 2 + 5 + 3) + (4 * 36 + 2 * 6 * 12) + 6 * 5;
        assert_eq!(s.flat_size(), want);
    }

    #[test]
    fn unflatten_roundtrips_through_reshape() {
        let s = tiny();
        let theta = init_flat(&s, 0.1, 7);
        let u = unflatten_dt(&theta, &s).unwrap();
        assert_eq!(u.e_rtg.dims(), &[4, 6]);
        assert_eq!(u.w_in.dims(), &[2, 6]);
        assert_eq!(u.e_act.dims(), &[5, 6]);
        assert_eq!(u.e_pos.dims(), &[3, 6]);
        assert_eq!(u.w_out.dims(), &[6, 5]);
        assert_eq!(u.layers[0].w1.dims(), &[6, 12]);
        let back = u.w_in.reshape(12).unwrap().to_vec1::<f32>().unwrap();
        let orig = theta.narrow(0, 24, 12).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(back, orig);
    }

    /// Every real state slot must emit a probability distribution, including
    /// the last one — a causal model that only predicts at `t < T` would drop
    /// exactly the step whose action is the last in the trajectory.
    #[test]
    fn real_state_slots_emit_normalised_distributions() {
        let s = tiny();
        let theta = init_flat(&s, 0.2, 3);
        let b = batch(&s, &[3, 2], 11);
        for variant in [Variant::Standard, Variant::Cosine] {
            let probs = dt_probs(&theta, &s, variant, &b).unwrap();
            assert_eq!(probs.dims(), &[2, 9, 5]);
            let sums: Vec<f32> = probs
                .sum(D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-5), "{variant:?}");
            assert!(
                probs
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|p| *p >= 0.0)
            );
        }
    }

    /// The loss must be the mean NLL of exactly the scored steps, i.e. the
    /// per-step probability of the action actually taken — not of the slot
    /// after it, and not a sum over slots.
    #[test]
    fn loss_is_the_mean_step_nll() {
        let s = tiny();
        let theta = init_flat(&s, 0.2, 5);
        let b = batch(&s, &[3, 2], 13);
        let logits = dt_forward(&theta, &s, Variant::Standard, &b, NormGrad::Stopped).unwrap();
        let (idx, targets) = b.scored_steps().unwrap();
        assert_eq!(idx.len(), 5);
        let flat = logits
            .reshape((2 * 9, 5))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let mut nll = 0.0f64;
        for (i, y) in idx.iter().zip(&targets) {
            let row = &flat[*i as usize];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max as f64
                + row
                    .iter()
                    .map(|v| ((*v - max) as f64).exp())
                    .sum::<f64>()
                    .ln();
            nll += lse - row[*y as usize] as f64;
        }
        nll /= targets.len() as f64;
        let got = dt_loss(&theta, &s, Variant::Standard, &b).unwrap();
        assert!((got - nll).abs() < 1e-4, "loss {got} vs manual {nll}");
    }

    /// Padding is only sound if a padded slot is invisible to real steps.  Fill
    /// the padded slots of one batch with other content and the loss must not
    /// move by a single bit: if it does, causal attention is reading the pad
    /// (or the mask is not applied where the loss thinks it is).
    #[test]
    fn padding_content_never_reaches_a_real_step() {
        let s = tiny();
        let theta = init_flat(&s, 0.2, 17);
        let a = batch(&s, &[3, 1], 19);
        let b0 = batch(&s, &[3, 1], 19);
        // Overwrite row 1's padded timesteps 1..3 with different ids/features.
        let act: Vec<u32> = b0.act.flatten_all().unwrap().to_vec1().unwrap();
        let rtg: Vec<u32> = b0.rtg.flatten_all().unwrap().to_vec1().unwrap();
        let mut state: Vec<f32> = b0.state.flatten_all().unwrap().to_vec1().unwrap();
        let (rows, t, row) = (2usize, 3usize, 1usize);
        let (mut act2, mut rtg2) = (act.clone(), rtg.clone());
        for step in 1..t {
            act2[row * t + step] = 4;
            rtg2[row * t + step] = 3;
            for k in 0..s.state_dim {
                state[(row * t + step) * s.state_dim + k] = 7.5;
            }
        }
        let dev = &Device::Cpu;
        let b1 = DtBatch::new(
            Tensor::from_vec(rtg2, (rows, t), dev).unwrap(),
            Tensor::from_vec(state, (rows, t, s.state_dim), dev).unwrap(),
            Tensor::from_vec(act2, (rows, t), dev).unwrap(),
            b0.mask.clone(),
        )
        .unwrap();
        let la = dt_loss(&theta, &s, Variant::Standard, &a).unwrap();
        let lb = dt_loss(&theta, &s, Variant::Standard, &b1).unwrap();
        assert_eq!(
            la.to_bits(),
            lb.to_bits(),
            "pad content reached a real step"
        );
    }

    #[test]
    fn mask_must_be_a_prefix() {
        let dev = &Device::Cpu;
        let rtg = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), dev).unwrap();
        let state = Tensor::from_vec(vec![0f32, 0.0, 0.0, 1.0, 0.0, 0.0], (1, 3, 2), dev).unwrap();
        let act = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), dev).unwrap();
        let hole = Tensor::from_vec(vec![1f32, 0.0, 1.0], (1, 3), dev).unwrap();
        assert!(DtBatch::new(rtg.clone(), state.clone(), act.clone(), hole).is_err());
        let prefix = Tensor::from_vec(vec![1f32, 1.0, 0.0], (1, 3), dev).unwrap();
        assert!(DtBatch::new(rtg, state, act, prefix).is_ok());
    }

    /// `∇_θ` must be the gradient of the *value* `dt_loss` returns.  Central
    /// differences at a few coordinates of the flat vector, with the exact
    /// Jacobian for the Frobenius normalisations in the cosine arm: the
    /// stopped-denominator gradient is deliberately a different function, and
    /// using it here would make a mode search descend something the chain does
    /// not sample.
    #[test]
    fn gradient_matches_finite_differences() {
        let s = tiny();
        let b = batch(&s, &[3, 2], 23);
        let seed_theta = init_flat(&s, 0.2, 29);
        for variant in [Variant::Standard, Variant::Cosine] {
            let base = seed_theta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let theta = Tensor::from_vec(base.clone(), base.len(), &Device::Cpu).unwrap();
            let (loss, g) = dt_loss_and_grad(&theta, &s, variant, &b, NormGrad::Exact).unwrap();
            assert!(loss.is_finite());
            let gv = g.to_vec1::<f32>().unwrap();
            let h = 1e-3f32;
            for &i in &[0usize, 5, 30, 90, base.len() - 1] {
                let mut plus = base.clone();
                let mut minus = base.clone();
                plus[i] += h;
                minus[i] -= h;
                let td = |v: Vec<f32>| Tensor::from_vec(v, base.len(), &Device::Cpu).unwrap();
                let lp = dt_loss(&td(plus), &s, variant, &b).unwrap();
                let lm = dt_loss(&td(minus), &s, variant, &b).unwrap();
                let fd = (lp - lm) / (2.0 * f64::from(h));
                let got = f64::from(gv[i]);
                assert!(
                    (got - fd).abs() <= 2e-2 * fd.abs().max(1.0),
                    "{variant:?} coord {i}: analytic {got}, finite difference {fd}"
                );
            }
        }
    }

    /// The potential wrapper must report the same number the loss computes and
    /// the number of transitions the batch holds — the `N` every DT cell has to
    /// state, since it is not the trajectory count.
    #[test]
    fn dt_potential_counts_transitions_and_matches_the_loss() {
        let s = tiny();
        let b = batch(&s, &[3, 2], 31);
        assert_eq!(b.transitions(), 5);
        assert_eq!(b.batch_size(), 2);
        let theta = init_flat(&s, 0.2, 37);
        let p = DtPotential::new(b.clone(), s, Variant::Standard);
        assert_eq!(p.n_terms(), 5);
        assert_eq!(p.trajectories(), 2);
        assert_eq!(p.label(), "dt-softmax");
        let want = dt_loss(&theta, &s, Variant::Standard, &b).unwrap();
        assert_eq!(p.mean_loss(&theta).to_bits(), want.to_bits());
        let g = p.mean_grad(&theta).expect("gradient");
        assert_eq!(g.dims(), &[s.flat_size()]);
    }
}
