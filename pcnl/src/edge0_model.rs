//! E1 — the small Edge0 latent-MoE decision model (`EDGE0_RLCD_PLAN.md` §2).
//!
//! This is a documented *adaptation* of the grouped latent D/U architecture,
//! not a miniature Edge0 checkpoint: the routing rule (softmax/top-2,
//! renormalized, lowest-index tie break), ordinary multi-head attention and the
//! typed heads are this model's own choices. Nothing here is a claim about the
//! large model's inference semantics.
//!
//! The module's job in the plan is structural: it must make a *real* stochastic
//! decision policy trainable, with typed conditional outputs, at a parameter
//! budget small enough to enumerate. Concretely it provides
//!
//! * a question-independent state encoder (`encode_state`), re-used for every
//!   question;
//! * grouped latent-MoE blocks: per-group compressed down/gate projections,
//!   per-expert expansion, deterministic top-k routing plus one shared expert;
//! * independent question blocks that attend to the state and to their own
//!   tokens — never to another question — in both packed and separate form;
//! * a per-question four-path categorical policy `a_phi(r | s,q)` and
//!   path-conditioned typed heads `u_theta(. | s,q,r)` for Noul, Choice and
//!   ordinal Score;
//! * an exact `∑_r a_phi u_theta` marginal (the deployed output, not a sampled
//!   path);
//! * a full parameter census and a `StateCacheKey` that includes every varying
//!   encoder input, the weights and the inference settings.
//!
//! Every weight is a view into one flat `θ ∈ R^P`, the same convention as
//! [`crate::model`], so candle autodiff and the existing optimizers see one
//! leaf and the E2 gradient gate can differentiate the exact marginal and the
//! sampled-path estimator through the *same* graph.

use crate::edge0_world::{QuestionKind, SchemaError};
use crate::manifest::Fnv;
use candle_core::{D, Device, Result, Tensor};
use candle_nn::ops::{sigmoid, silu, softmax};

/// The R9 first configuration (`RESEARCH_PLAN.md` §R9.0): width 64, 4 state
/// blocks, 8 routed experts in 2 D/U groups, latent width 32, top-2 plus one
/// shared expert, four latent paths. Total parameters stay below one million.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge0Config {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub state_blocks: usize,
    pub question_blocks: usize,
    pub experts: usize,
    pub groups: usize,
    pub latent: usize,
    pub shared_ff: usize,
    pub question_ff: usize,
    pub top_k: usize,
    pub paths: usize,
    pub max_state_tokens: usize,
    pub max_question_tokens: usize,
    pub max_levels: usize,
}

impl Default for Edge0Config {
    fn default() -> Self {
        Self {
            vocab: 512,
            d: 64,
            heads: 4,
            state_blocks: 4,
            question_blocks: 2,
            experts: 8,
            groups: 2,
            latent: 32,
            shared_ff: 128,
            question_ff: 128,
            top_k: 2,
            paths: 4,
            max_state_tokens: 32,
            max_question_tokens: 8,
            max_levels: 5,
        }
    }
}

impl Edge0Config {
    /// A fixture configuration: same *structure* as R9, small enough that an
    /// exhaustive `4^4 = 256`-tuple gradient enumeration is a test, not a
    /// campaign.
    pub fn tiny() -> Self {
        Self {
            vocab: 32,
            d: 8,
            heads: 2,
            state_blocks: 1,
            question_blocks: 1,
            experts: 4,
            groups: 2,
            latent: 4,
            shared_ff: 8,
            question_ff: 8,
            top_k: 2,
            paths: 4,
            max_state_tokens: 8,
            max_question_tokens: 4,
            max_levels: 4,
        }
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.d == 0 || !self.d.is_multiple_of(self.heads) {
            return Err("d must be a positive multiple of heads".into());
        }
        if self.groups == 0 || self.experts == 0 || self.experts % self.groups != 0 {
            return Err("experts must be a positive multiple of groups".into());
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err("top_k must be in 1..=experts".into());
        }
        if self.paths == 0 || self.max_levels < 2 {
            return Err("the model needs >=1 latent path and >=2 score levels".into());
        }
        Ok(())
    }

    /// One parameter slot per trainable tensor, in the order
    /// [`Edge0Slices::from_theta`] consumes them. The census is derived from
    /// this list, so a new tensor cannot be added without appearing in it.
    pub fn layout(&self) -> Vec<NamedSlot> {
        let d = self.d;
        let mut out = Vec::new();
        let mut at = 0usize;
        let mut push = |name: &'static str, shape: Vec<usize>| {
            let len: usize = shape.iter().product();
            out.push(NamedSlot {
                name,
                start: at,
                len,
                shape,
            });
            at += len;
        };
        push("embed", vec![self.vocab, d]);
        push("state_pos", vec![self.max_state_tokens, d]);
        push("question_pos", vec![self.max_question_tokens, d]);
        push("path_embed", vec![self.paths, d]);
        for _ in 0..self.state_blocks {
            push("state_attn_wq", vec![d, d]);
            push("state_attn_wk", vec![d, d]);
            push("state_attn_wv", vec![d, d]);
            push("state_attn_wo", vec![d, d]);
            push("moe_group_down", vec![self.groups, d, self.latent]);
            push("moe_group_gate", vec![self.groups, d, self.latent]);
            push("moe_expert_up", vec![self.experts, self.latent, d]);
            push("moe_router", vec![d, self.experts]);
            push("shared_w1", vec![d, self.shared_ff]);
            push("shared_w3", vec![d, self.shared_ff]);
            push("shared_w2", vec![self.shared_ff, d]);
        }
        for _ in 0..self.question_blocks {
            push("question_attn_wq", vec![d, d]);
            push("question_attn_wk", vec![d, d]);
            push("question_attn_wv", vec![d, d]);
            push("question_attn_wo", vec![d, d]);
            push("question_w1", vec![d, self.question_ff]);
            push("question_w2", vec![self.question_ff, d]);
        }
        push("policy_w", vec![d, self.paths]);
        push("policy_b", vec![self.paths]);
        push("noul_w", vec![self.paths, d]);
        push("noul_b", vec![self.paths]);
        push("score_w", vec![self.paths, d]);
        push("score_thresholds", vec![self.paths, self.max_levels - 1]);
        out
    }

    /// `P` — every trainable (and frozen) tensor, counted from the layout and
    /// never from a hand-written prose number.
    pub fn flat_size(&self) -> usize {
        self.layout().iter().map(|s| s.len).sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedSlot {
    pub name: &'static str,
    pub start: usize,
    pub len: usize,
    pub shape: Vec<usize>,
}

/// Which routing rule a block uses. `TopK` is the deployed rule; `Soft` is the
/// dense reference the plan asks for, and with `top_k == experts` the two are
/// identical because renormalized top-k over every expert *is* the softmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    TopK,
    Soft,
}

/// Named parameter views over one flat `θ`.
pub struct Edge0Slices {
    pub embed: Tensor,
    pub state_pos: Tensor,
    pub question_pos: Tensor,
    pub path_embed: Tensor,
    pub state: Vec<StateSlices>,
    pub question: Vec<QuestionSlices>,
    pub policy_w: Tensor,
    pub policy_b: Tensor,
    pub noul_w: Tensor,
    pub noul_b: Tensor,
    pub score_w: Tensor,
    pub thresholds: Tensor,
}

pub struct StateSlices {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub group_down: Tensor,
    pub group_gate: Tensor,
    pub expert_up: Tensor,
    pub router: Tensor,
    pub shared_w1: Tensor,
    pub shared_w3: Tensor,
    pub shared_w2: Tensor,
}

pub struct QuestionSlices {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub w1: Tensor,
    pub w2: Tensor,
}

impl Edge0Slices {
    pub fn from_theta(theta: &Tensor, config: &Edge0Config) -> Result<Self> {
        let mut views = Vec::new();
        for slot in config.layout() {
            views.push(theta.narrow(0, slot.start, slot.len)?.reshape(slot.shape)?);
        }
        let mut it = views.into_iter();
        let mut next = || it.next().expect("layout and reader agree");
        let embed = next();
        let state_pos = next();
        let question_pos = next();
        let path_embed = next();
        let mut state = Vec::with_capacity(config.state_blocks);
        for _ in 0..config.state_blocks {
            state.push(StateSlices {
                wq: next(),
                wk: next(),
                wv: next(),
                wo: next(),
                group_down: next(),
                group_gate: next(),
                expert_up: next(),
                router: next(),
                shared_w1: next(),
                shared_w3: next(),
                shared_w2: next(),
            });
        }
        let mut question = Vec::with_capacity(config.question_blocks);
        for _ in 0..config.question_blocks {
            question.push(QuestionSlices {
                wq: next(),
                wk: next(),
                wv: next(),
                wo: next(),
                w1: next(),
                w2: next(),
            });
        }
        let policy_w = next();
        let policy_b = next();
        let noul_w = next();
        let noul_b = next();
        let score_w = next();
        let thresholds = next();
        debug_assert!(it.next().is_none());
        Ok(Self {
            embed,
            state_pos,
            question_pos,
            path_embed,
            state,
            question,
            policy_w,
            policy_b,
            noul_w,
            noul_b,
            score_w,
            thresholds,
        })
    }
}

/// One question in a pack. `tokens` is the question text; `options` are the
/// declared Choice option token ids (empty for the other kinds). A Score
/// question's level count is declared in `kind`.
#[derive(Clone, Debug)]
pub struct QuestionInput {
    pub kind: QuestionKind,
    pub tokens: Vec<u32>,
    pub options: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ModelInput<'a> {
    pub state: &'a [u32],
    pub questions: &'a [QuestionInput],
}

impl QuestionInput {
    pub fn validate(&self, config: &Edge0Config) -> std::result::Result<(), String> {
        self.kind
            .validate()
            .map_err(|e: SchemaError| e.to_string())?;
        if self.tokens.is_empty() || self.tokens.len() > config.max_question_tokens {
            return Err(format!(
                "question token count {} outside 1..={}",
                self.tokens.len(),
                config.max_question_tokens
            ));
        }
        if self.options.len() != self.kind.options_count() {
            return Err("declared options do not match the question schema".into());
        }
        if let QuestionKind::Score { levels } = self.kind {
            if levels > config.max_levels {
                return Err(format!(
                    "score levels {levels} exceed the configured maximum {}",
                    config.max_levels
                ));
            }
        }
        for &t in self.tokens.iter().chain(&self.options) {
            if t as usize >= config.vocab {
                return Err(format!("token {t} outside the vocabulary"));
            }
        }
        Ok(())
    }
}

impl QuestionKind {
    fn options_count(self) -> usize {
        match self {
            QuestionKind::Choice { options } => options,
            QuestionKind::Noul | QuestionKind::Score { .. } => 0,
        }
    }
}

/// Per-question policy and path-conditioned heads.
pub struct PathOutput {
    /// `[paths]` — unnormalized `log a_phi(r | s,q)`.
    pub policy_logits: Tensor,
    /// `[paths]` — `a_phi(. | s,q)`.
    pub policy: Tensor,
    /// `paths` tensors of shape `[K_q]` — `u_theta(. | s,q,r)`.
    pub path_probs: Vec<Tensor>,
    /// `[d]` — the question representation the policy and path heads read.
    /// Exposed so the certified head-chart pass can be driven by an interval
    /// dual of this vector instead of re-deriving it.
    pub rep: Tensor,
}

pub struct ModelOutput {
    pub per_question: Vec<PathOutput>,
    /// `[d]` — the question-independent pooled state representation.
    pub state_rep: Tensor,
}

impl PathOutput {
    /// `p = Σ_r a_phi(r) u_theta(.|r)` — the exactly enumerated deployed
    /// marginal, shape `[K_q]`.
    pub fn marginal(&self) -> Result<Tensor> {
        let paths: Vec<&Tensor> = self.path_probs.iter().collect();
        let stacked = Tensor::stack(&paths, 0)?;
        self.policy
            .unsqueeze(0)?
            .broadcast_matmul(&stacked)?
            .squeeze(0)
    }

    /// Read the marginal as plain `f64`s (evaluation, not gradient paths).
    pub fn marginal_vec(&self) -> Result<Vec<f64>> {
        Ok(self
            .marginal()?
            .to_dtype(candle_core::DType::F64)?
            .to_vec1::<f64>()?)
    }

    pub fn policy_vec(&self) -> Result<Vec<f64>> {
        Ok(self
            .policy
            .to_dtype(candle_core::DType::F64)?
            .to_vec1::<f64>()?)
    }

    pub fn path_probs_vec(&self) -> Result<Vec<Vec<f64>>> {
        self.path_probs
            .iter()
            .map(|p| Ok(p.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?))
            .collect()
    }
}

fn ln(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&((var + 1e-5)?.sqrt()?))
}

/// Multi-head attention with an additive mask (`0` allowed, `-inf` forbidden).
fn attention(
    x: &Tensor,
    mask: &Tensor,
    heads: usize,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
) -> Result<Tensor> {
    let (t, d) = x.dims2()?;
    let hd = d / heads;
    let shape = |z: &Tensor| -> Result<Tensor> { z.reshape((t, heads, hd))?.transpose(0, 1) };
    let q = shape(&x.broadcast_matmul(wq)?)?;
    let k = shape(&x.broadcast_matmul(wk)?)?;
    let v = shape(&x.broadcast_matmul(wv)?)?;
    let scores = (q.broadcast_matmul(&k.transpose(1, 2)?)? / (hd as f64).sqrt())?;
    let probs = softmax(&scores.broadcast_add(mask)?, D::Minus1)?;
    let out = probs
        .broadcast_matmul(&v)?
        .transpose(0, 1)?
        .reshape((t, d))?;
    out.broadcast_matmul(wo)
}

/// Deterministic top-k: score descending, then **index ascending**. This tie
/// rule is part of the routing identity, because the selected set changes the
/// computation.
fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// One grouped latent-MoE block.
///
/// `x @ down[g]` is the compressed activation, gated by `x @ gate[g]` through
/// SiLU, expanded by `x @ up[e]`; the selected experts' outputs are renormalized
/// by their softmax gate and added to a dense shared expert.
fn moe(x: &Tensor, s: &StateSlices, config: &Edge0Config, routing: Routing) -> Result<Tensor> {
    let (t, d) = x.dims2()?;
    let logits = x.broadcast_matmul(&s.router)?; // [T, E]
    let probs = softmax(&logits, D::Minus1)?;
    let probs_v = probs.flatten_all()?.to_vec1::<f32>()?;

    let mut outputs = Vec::with_capacity(t);
    for i in 0..t {
        let row_scores = &probs_v[i * config.experts..(i + 1) * config.experts];
        let selected: Vec<(usize, f64)> = match routing {
            Routing::Soft => (0..config.experts)
                .map(|e| (e, f64::from(row_scores[e])))
                .collect(),
            Routing::TopK => {
                let idx = top_k_indices(row_scores, config.top_k);
                let total: f64 = idx.iter().map(|&e| f64::from(row_scores[e])).sum();
                idx.into_iter()
                    .map(|e| {
                        // Renormalize the selected gates; a zero total is
                        // impossible after softmax, but guard the division.
                        let g = if total > 0.0 {
                            f64::from(row_scores[e]) / total
                        } else {
                            1.0 / config.top_k as f64
                        };
                        (e, g)
                    })
                    .collect()
            }
        };
        let mut acc: Option<Tensor> = None;
        for (e, gate) in selected {
            if gate == 0.0 {
                continue;
            }
            let group = e % config.groups;
            let down = s.group_down.narrow(0, group, 1)?.squeeze(0)?; // [d, ℓ]
            let gate_w = s.group_gate.narrow(0, group, 1)?.squeeze(0)?;
            let up = s.expert_up.narrow(0, e, 1)?.squeeze(0)?; // [ℓ, d]
            let row = x.narrow(0, i, 1)?; // [1, d]
            let a = row.broadcast_matmul(&down)?; // [1, ℓ]
            let g = row.broadcast_matmul(&gate_w)?;
            let h = (silu(&a)? * g)?;
            let out = h.broadcast_matmul(&up)?.affine(gate, 0.0)?; // [1, d]
            acc = Some(match acc {
                Some(prev) => (prev + out)?,
                None => out,
            });
        }
        // A row whose gates are all zero would otherwise vanish; the routing
        // always selects at least one expert, so `acc` is Some for T >= 1.
        let acc = acc.unwrap_or(Tensor::zeros((1, d), x.dtype(), x.device())?);
        outputs.push(acc);
    }
    let routed = Tensor::cat(&outputs, 0)?;

    // Shared expert: SwiGLU.
    let h = (silu(&x.broadcast_matmul(&s.shared_w1)?)? * x.broadcast_matmul(&s.shared_w3)?)?;
    let shared = h.broadcast_matmul(&s.shared_w2)?;
    routed.add(&shared)
}

/// Dense (non-MoE) FFN used by the question blocks.
fn dense_ffn(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Result<Tensor> {
    silu(&x.broadcast_matmul(w1)?)?.broadcast_matmul(w2)
}

fn all_allowed(t: usize, device: &Device) -> Result<Tensor> {
    Tensor::zeros((t, t), candle_core::DType::F32, device)
}

fn embed_tokens(embed: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let idx = Tensor::from_vec(ids.to_vec(), vec![ids.len()], embed.device())?;
    embed.index_select(&idx, 0)
}

/// Question-independent state encoding. The same call is made whether questions
/// are packed or separate, which is what makes the cache key (and therefore
/// question ordering) irrelevant to the state representation.
pub fn encode_state(theta: &Tensor, config: &Edge0Config, state: &[u32]) -> Result<Tensor> {
    let s = Edge0Slices::from_theta(theta, config)?;
    encode_state_with(&s, config, state, Routing::TopK)
}

fn encode_state_with(
    s: &Edge0Slices,
    config: &Edge0Config,
    state: &[u32],
    routing: Routing,
) -> Result<Tensor> {
    if state.is_empty() || state.len() > config.max_state_tokens {
        return Err(candle_core::Error::Msg(
            "state token count outside the declared contract".into(),
        ));
    }
    let pos = s.state_pos.narrow(0, 0, state.len())?;
    let mut x = embed_tokens(&s.embed, state)?.add(&pos)?;
    let mask = all_allowed(state.len(), x.device())?;
    for b in &s.state {
        let n = ln(&x)?;
        x = x.add(&attention(
            &n,
            &mask,
            config.heads,
            &b.wq,
            &b.wk,
            &b.wv,
            &b.wo,
        )?)?;
        let n = ln(&x)?;
        x = x.add(&moe(&n, b, config, routing)?)?;
    }
    x.mean_keepdim(0)?.squeeze(0)
}

/// Full forward: typed outputs for every question in the pack.
pub fn forward(
    theta: &Tensor,
    config: &Edge0Config,
    input: &ModelInput<'_>,
) -> Result<ModelOutput> {
    forward_with(theta, config, input, Routing::TopK)
}

/// `forward` with an explicit routing rule (`Routing::Soft` is the dense
/// reference used to check the selected-expert implementation).
pub fn forward_with(
    theta: &Tensor,
    config: &Edge0Config,
    input: &ModelInput<'_>,
    routing: Routing,
) -> Result<ModelOutput> {
    config.validate().map_err(|e| candle_core::Error::Msg(e))?;
    for q in input.questions {
        q.validate(config).map_err(|e| candle_core::Error::Msg(e))?;
    }
    let s = Edge0Slices::from_theta(theta, config)?;
    let state_rep = encode_state_with(&s, config, input.state, routing)?;
    let q_reps = encode_questions(&s, config, input)?;
    let mut per_question = Vec::with_capacity(q_reps.len());
    for (q, rep) in input.questions.iter().zip(&q_reps) {
        per_question.push(heads(&s, config, q, rep)?);
    }
    Ok(ModelOutput {
        per_question,
        state_rep,
    })
}

/// Packed question encoding: state tokens plus all question tokens in one
/// attention call with a segment mask. A question sees the whole state and its
/// own tokens only; the state rows never see a question. This is what makes the
/// packed pass agree with running each question separately.
fn encode_questions(
    s: &Edge0Slices,
    config: &Edge0Config,
    input: &ModelInput<'_>,
) -> Result<Vec<Tensor>> {
    let ns = input.state.len();
    let nq_total: usize = input.questions.iter().map(|q| q.tokens.len()).sum();
    let total = ns + nq_total;
    // Token ids and per-token position rows.
    let mut ids = Vec::with_capacity(total);
    ids.extend_from_slice(input.state);
    let mut pos_rows: Vec<Tensor> = Vec::with_capacity(total);
    for i in 0..ns {
        pos_rows.push(s.state_pos.narrow(0, i, 1)?);
    }
    // Segment boundaries in the concatenated question block.
    let mut segments: Vec<(usize, usize)> = Vec::with_capacity(input.questions.len());
    let mut cursor = ns;
    for q in input.questions {
        ids.extend_from_slice(&q.tokens);
        for j in 0..q.tokens.len() {
            pos_rows.push(s.question_pos.narrow(0, j, 1)?);
        }
        segments.push((cursor, cursor + q.tokens.len()));
        cursor += q.tokens.len();
    }
    let pos = Tensor::cat(&pos_rows, 0)?;
    let mut x = embed_tokens(&s.embed, &ids)?.add(&pos)?;

    // Additive mask: state rows attend to state only; question rows attend to
    // the whole state plus their own segment.
    let mut mask_v = vec![f32::NEG_INFINITY; total * total];
    for r in 0..ns {
        for c in 0..ns {
            mask_v[r * total + c] = 0.0;
        }
    }
    for (qi, &(start, end)) in segments.iter().enumerate() {
        for r in start..end {
            for c in 0..ns {
                mask_v[r * total + c] = 0.0;
            }
            let (s0, e0) = segments[qi];
            for c in s0..e0 {
                mask_v[r * total + c] = 0.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask_v, (total, total), x.device())?;

    for b in &s.question {
        let n = ln(&x)?;
        x = x.add(&attention(
            &n,
            &mask,
            config.heads,
            &b.wq,
            &b.wk,
            &b.wv,
            &b.wo,
        )?)?;
        let n = ln(&x)?;
        x = x.add(&dense_ffn(&n, &b.w1, &b.w2)?)?;
    }
    let mut reps = Vec::with_capacity(segments.len());
    for &(start, end) in &segments {
        reps.push(
            x.narrow(0, start, end - start)?
                .mean_keepdim(0)?
                .squeeze(0)?,
        );
    }
    Ok(reps)
}

/// Per-question policy and typed path-conditioned heads.
fn heads(
    s: &Edge0Slices,
    config: &Edge0Config,
    q: &QuestionInput,
    rep: &Tensor,
) -> Result<PathOutput> {
    let policy_logits = rep
        .unsqueeze(0)?
        .broadcast_matmul(&s.policy_w)?
        .squeeze(0)?
        .add(&s.policy_b)?;
    let policy = softmax(&policy_logits, D::Minus1)?;
    let mut path_probs = Vec::with_capacity(config.paths);
    for r in 0..config.paths {
        let path = s.path_embed.narrow(0, r, 1)?.squeeze(0)?;
        let h = rep.add(&path)?.unsqueeze(0)?; // [1, d]
        let probs = match q.kind {
            QuestionKind::Noul => {
                let w = s.noul_w.narrow(0, r, 1)?.squeeze(0)?;
                let b = s.noul_b.narrow(0, r, 1)?;
                let logit = h.broadcast_matmul(&w.unsqueeze(1)?)?; // [1, 1]
                let logit = logit.squeeze(1)?.add(&b)?; // [1]
                let p_yes = sigmoid(&logit.squeeze(0)?)?;
                let p_no = p_yes.affine(-1.0, 1.0)?;
                Tensor::stack(&[&p_no, &p_yes], 0)?
            }
            QuestionKind::Choice { options } => {
                debug_assert_eq!(options, q.options.len());
                let opt = embed_tokens(&s.embed, &q.options)?; // [K, d]
                let logits = opt.broadcast_matmul(&h.transpose(0, 1)?)?; // [K,1]
                let logits = (logits / (config.d as f64).sqrt())?;
                softmax(&logits.squeeze(1)?, 0)?
            }
            QuestionKind::Score { levels } => {
                let w = s.score_w.narrow(0, r, 1)?.squeeze(0)?;
                let score = h
                    .broadcast_matmul(&w.unsqueeze(1)?)?
                    .squeeze(0)?
                    .squeeze(0)?;
                let raw = s.thresholds.narrow(0, r, 1)?.squeeze(0)?;
                // Enforce ordered thresholds: `t_j = t_0 + Σ_{i≤j} raw_i²`.
                // Without this the learned thresholds are unordered, the
                // cumulative differences can go negative, and the "ordinal"
                // head is not a distribution on ordered levels.
                let mut ordered = Vec::with_capacity(levels - 1);
                let mut t = raw.narrow(0, 0, 1)?.squeeze(0)?;
                ordered.push(t.clone());
                for j in 1..levels - 1 {
                    t = t.add(&raw.narrow(0, j, 1)?.squeeze(0)?.sqr()?)?;
                    ordered.push(t.clone());
                }
                let one = Tensor::ones(1, score.dtype(), score.device())?.squeeze(0)?;
                let zero = Tensor::zeros(1, score.dtype(), score.device())?.squeeze(0)?;
                let mut ge = vec![one];
                for t in &ordered {
                    let diff = score.sub(t)?;
                    ge.push(sigmoid(&diff.affine(1.5, 0.0)?)?);
                }
                ge.push(zero);
                let mut probs = Vec::with_capacity(levels);
                for j in 0..levels {
                    // The cumulative difference is nonnegative in exact
                    // arithmetic; clamp the last bits so a reported marginal
                    // cannot carry a negative probability.
                    probs.push(ge[j].sub(&ge[j + 1])?.clamp(0.0, 1.0)?);
                }
                let parts: Vec<&Tensor> = probs.iter().collect();
                Tensor::stack(&parts, 0)?
            }
        };
        path_probs.push(probs);
    }
    Ok(PathOutput {
        policy_logits,
        policy,
        path_probs,
        rep: rep.clone(),
    })
}

/// The ordinal (cumulative) link used by the Score head, in plain `f64`.
///
/// `P(level ≥ j) = σ(1.5·(score − threshold_j))`, ordered thresholds,
/// `P(level = j) = P(≥j) − P(≥j+1)`. The helper exists so the ordering property
/// can be tested directly, and it is the same formula the tensor path applies.
pub fn ordinal_probs(score: f64, thresholds: &[f64], levels: usize) -> Vec<f64> {
    assert_eq!(thresholds.len(), levels - 1);
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut ge = Vec::with_capacity(levels + 1);
    ge.push(1.0);
    for t in thresholds {
        ge.push(sigmoid(1.5 * (score - t)));
    }
    ge.push(0.0);
    (0..levels).map(|j| (ge[j] - ge[j + 1]).max(0.0)).collect()
}

/// Deterministic initialization from a seed (the model has no other RNG).
pub fn init(config: &Edge0Config, seed: u64) -> Tensor {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let p = config.flat_size();
    let mut v = Vec::with_capacity(p);
    for _ in 0..p {
        // Small Gaussian weights; the typed heads' thresholds start ordered by
        // construction because `heads` reads them through an ordinal link.
        let u1: f64 = 1.0 - rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v.push((0.05 * z) as f32);
    }
    Tensor::from_vec(v, vec![p], &Device::Cpu).expect("flat init")
}

/// The cache identity for a state encoding. It includes every varying encoder
/// input the plan names: the state tokens, the weights and the inference
/// settings. Training, dropout or a route change invalidates it because those
/// all change `weights_hash`/`settings`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCacheKey {
    pub state_hash: u64,
    pub weights_hash: u64,
    pub config_hash: u64,
    pub inference_hash: u64,
}

impl StateCacheKey {
    pub fn new(
        config: &Edge0Config,
        theta: &Tensor,
        state: &[u32],
        inference: &str,
    ) -> Result<Self> {
        let mut sh = Fnv::new();
        for &t in state {
            sh.write(&t.to_le_bytes());
        }
        let values = theta.flatten_all()?.to_vec1::<f32>()?;
        let mut wh = Fnv::new();
        for v in &values {
            wh.write(&v.to_le_bytes());
        }
        let mut ch = Fnv::new();
        for slot in config.layout() {
            ch.write(slot.name.as_bytes());
            for &d in &slot.shape {
                ch.write(&d.to_le_bytes());
            }
        }
        sh.write(&[config.max_state_tokens as u8, config.top_k as u8]);
        let mut ih = Fnv::new();
        ih.write(inference.as_bytes());
        Ok(Self {
            state_hash: sh.finish(),
            weights_hash: wh.finish(),
            config_hash: ch.finish(),
            inference_hash: ih.finish(),
        })
    }
}

/// Parameter and activation census for the manifest (`EDGE0_RLCD_PLAN.md` §2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamCensus {
    pub total: usize,
    pub per_component: Vec<(&'static str, usize)>,
    /// Parameters in the selected experts only (top-k of each MoE block),
    /// excluding shared experts, attention and heads. This is the "active"
    /// routed count, not a latency measurement.
    pub routed_active: usize,
}

pub fn census(config: &Edge0Config) -> ParamCensus {
    let mut per_component: Vec<(&'static str, usize)> = Vec::new();
    for slot in config.layout() {
        if let Some(entry) = per_component.iter_mut().find(|(n, _)| *n == slot.name) {
            entry.1 += slot.len;
        } else {
            per_component.push((slot.name, slot.len));
        }
    }
    let per_block_expert = config.state_blocks * config.experts * config.latent * config.d;
    let routed_active = if config.experts == 0 {
        0
    } else {
        per_block_expert * config.top_k / config.experts
    };
    ParamCensus {
        total: config.flat_size(),
        per_component,
        routed_active,
    }
}
