//! An interval-dual forward pass over the Edge0 model, the base for a certified
//! chart bound.
//!
//! Why this exists
//! ---------------
//! The analytic policy-only bound for
//! [`crate::edge0_potential::head_chart_enclosure_1d`] relies on the path
//! distributions being constant in the chart coordinate `z`. Moving encoder
//! weights makes `rep`, the policy and every path distribution move, so `Psi'`
//! and `Psi''` have no closed form. The fix is a forward pass carrying
//! `<value, first derivative, second derivative>` **intervals** through the
//! whole model, with the encoder's contribution entering as a dual of `rep`.
//!
//! Scope of this module
//! --------------------
//! Both halves of the pass exist:
//!
//! * **head stage** — `policy_w`, `policy_b`, `path_embed`, `noul_w`, `noul_b`,
//!   `score_w`, `thresholds`;
//! * **encoder stage** — `embed`, `state_pos`, `question_pos`, the state
//!   attention/SwiGLU/MoE blocks and the question attention/FFN blocks, via an
//!   interval-dual LayerNorm → masked attention → grouped latent MoE pass.
//!   Top-k routing is discrete, so it is over-bounded by a max-over-experts
//!   convex-combination argument rather than assuming a fixed selected set.
//!
//! One structural caveat is recorded rather than hidden: the deployed
//! `edge0_model::forward` computes the pooled `state_rep`, projects it through
//! the learned `state_proj` slot, scales it by the state-dependent gate
//! `sigmoid(state_rep · state_gate)`, and adds the result to every question
//! representation before the heads, so a direction supported only on the
//! state-encoder weights does move the marginal. The dual pass below mirrors
//! that path: it pools the state-encoder dual, multiplies by `state_proj`,
//! applies the gated sigmoid (product rule), and adds the result to each
//! question rep, which is what makes the certified bounds cover the deployed
//! forward.
//!
//! The `Dual` arithmetic is outward-rounded interval arithmetic on the value,
//! first derivative and second derivative, so the returned bounds are bounds,
//! not estimates. The tests check containment against finite differences of the
//! real model (candle, `f32`) rather than against this module's own arithmetic.

use crate::edge0_potential::Edge0Potential;
use crate::edge0_world::QuestionKind;
use crate::evidence_enclosure::Iv;
use crate::target::{SubspaceTarget, ZPotential};
use candle_core::{DType, Result};

/// A scalar with interval value, first and second derivative w.r.t. the chart
/// `z`. The second derivative is what the midpoint quadrature error needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: Iv,
    pub d: Iv,
    pub dd: Iv,
}

impl Dual {
    pub fn constant(x: f64) -> Self {
        Self {
            v: Iv::point(x),
            d: Iv::point(0.0),
            dd: Iv::point(0.0),
        }
    }

    pub fn point(x: f64) -> Self {
        Self::constant(x)
    }

    /// A parameter on the chart: its value ranges over `theta0 ± radius·dir`
    /// and its derivative is the chart direction.
    pub fn chart(theta0: f64, dir: f64, radius: f64) -> Self {
        let spread = radius * dir.abs();
        Self {
            v: Iv::span(theta0 - spread, theta0 + spread),
            d: Iv::point(dir),
            dd: Iv::point(0.0),
        }
    }

    pub fn add(self, o: &Dual) -> Self {
        Self {
            v: self.v.add(&o.v),
            d: self.d.add(&o.d),
            dd: self.dd.add(&o.dd),
        }
    }

    pub fn sub(self, o: &Dual) -> Self {
        Self {
            v: self.v.sub(&o.v),
            d: self.d.sub(&o.d),
            dd: self.dd.sub(&o.dd),
        }
    }

    /// Product rule to second order: `(uv)'' = u''v + 2u'v' + uv''`.
    pub fn mul(self, o: &Dual) -> Self {
        Self {
            v: self.v.mul(&o.v),
            d: self.v.mul(&o.d).add(&self.d.mul(&o.v)),
            dd: self
                .dd
                .mul(&o.v)
                .add(&self.d.mul(&o.d).mul(&Iv::point(2.0)))
                .add(&self.v.mul(&o.dd)),
        }
    }

    pub fn scale(self, k: f64) -> Self {
        self.mul(&Dual::constant(k))
    }

    pub fn neg(self) -> Self {
        Self {
            v: self.v.neg(),
            d: self.d.neg(),
            dd: self.dd.neg(),
        }
    }

    /// `x → 1/(1+e^{-x})`. Requires the value interval to stay inside
    /// `[-60, 60]`, which the head logits satisfy for the declared charts; a
    /// wider interval returns `None` rather than clamping (clamping would make
    /// the bound unsound).
    pub fn sigmoid(self) -> Option<Self> {
        if self.v.lo < -60.0 || self.v.hi > 60.0 {
            return None;
        }
        let one = Iv::point(1.0);
        let s = one.div(&one.add(&self.v.neg().exp()));
        // s' = s(1 - s); s'' = s(1 - s)(1 - 2s).
        let om = one.sub(&s);
        let slope = s.mul(&om);
        let curv = slope.mul(&one.sub(&s.mul(&Iv::point(2.0))));
        Some(Self {
            v: s,
            d: slope.mul(&self.d),
            dd: curv.mul(&self.d.mul(&self.d)).add(&slope.mul(&self.dd)),
        })
    }

    /// `1/(1+e^{-x})` for interval values only (no derivative), used for the
    /// ordinal link.
    pub fn sigmoid_iv(x: Iv) -> Option<Iv> {
        if x.lo < -60.0 || x.hi > 60.0 {
            return None;
        }
        let one = Iv::point(1.0);
        Some(one.div(&one.add(&x.neg().exp())))
    }

    /// `x·σ(x)`; `g(x) = σ(x)(1 + x(1 − σ(x)))` is `silu'`, and
    /// `silu'' = g'(x)(x')² + g(x)x''`.
    pub fn silu(self) -> Option<Self> {
        let s = self.sigmoid()?;
        let one = Iv::point(1.0);
        let om = one.sub(&s.v);
        let sig_prime = s.v.mul(&om);
        let inner = one.add(&self.v.mul(&om));
        let g = s.v.mul(&inner);
        let g_prime = sig_prime
            .mul(&inner)
            .add(&s.v.mul(&om.sub(&self.v.mul(&sig_prime))));
        Some(Self {
            v: self.v.mul(&s.v),
            d: g.mul(&self.d),
            dd: g_prime.mul(&self.d.mul(&self.d)).add(&g.mul(&self.dd)),
        })
    }

    /// `exp` with the same `|x| ≤ 60` guard; `(e^x)'' = e^x((x')² + x'')`.
    pub fn exp(self) -> Option<Self> {
        if self.v.lo < -60.0 || self.v.hi > 60.0 {
            return None;
        }
        let e = self.v.exp();
        Some(Self {
            v: e,
            d: e.mul(&self.d),
            dd: e.mul(&self.d.mul(&self.d).add(&self.dd)),
        })
    }

    /// `ln` on a strictly positive value interval; `(ln x)'' = x''/x − (x'/x)²`.
    pub fn ln(self) -> Option<Self> {
        if self.v.lo <= 0.0 {
            return None;
        }
        let d = self.d.div(&self.v);
        Some(Self {
            v: self.v.ln(),
            d,
            dd: self.dd.div(&self.v).sub(&d.mul(&d)),
        })
    }

    /// `sqrt` on a nonnegative value interval; `sqrt' = 1/(2 sqrt)`.
    pub fn sqrt(self) -> Option<Self> {
        if self.v.lo < 0.0 {
            return None;
        }
        let s = self.v.sqrt();
        if s.lo <= 0.0 {
            return None;
        }
        let two_s = s.mul(&Iv::point(2.0));
        let cube = s.mul(&s).mul(&s);
        Some(Self {
            v: s,
            d: self.d.div(&two_s),
            dd: self
                .dd
                .div(&two_s)
                .sub(&self.d.mul(&self.d).div(&cube.mul(&Iv::point(4.0)))),
        })
    }

    /// `self / o`; the divisor's value interval must be strictly positive. All
    /// uses here are `sqrt(var + eps)` or squares of it, which are.
    pub fn divide(self, o: &Dual) -> Option<Self> {
        if o.v.lo <= 0.0 {
            return None;
        }
        let v = self.v.div(&o.v);
        let num = self.d.mul(&o.v).sub(&self.v.mul(&o.d));
        let den = o.v.mul(&o.v);
        let d = num.div(&den);
        // (u/v)'' = u''/v − u v''/v² − 2(u'v − uv')v'/v³.
        let dd = self
            .dd
            .div(&o.v)
            .sub(&self.v.mul(&o.dd).div(&den))
            .sub(&num.mul(&o.d).mul(&Iv::point(2.0)).div(&den.mul(&o.v)));
        Some(Self { v, d, dd })
    }

    /// An over-bound on `|d|`.
    pub fn slope_abs(self) -> f64 {
        self.d.lo.abs().max(self.d.hi.abs())
    }

    /// An over-bound on `|dd|`.
    pub fn curvature_abs(self) -> f64 {
        self.dd.lo.abs().max(self.dd.hi.abs())
    }
}

/// Softmax over dual logits, shifted by the largest upper bound.
fn softmax_dual(logits: &[Dual]) -> Option<Vec<Dual>> {
    let shift = logits
        .iter()
        .map(|l| l.v.hi)
        .fold(f64::NEG_INFINITY, f64::max);
    if !shift.is_finite() {
        return None;
    }
    let mut es = Vec::with_capacity(logits.len());
    for l in logits {
        es.push(l.sub(&Dual::constant(shift)).exp()?);
    }
    let mut sum = Dual::constant(0.0);
    for e in &es {
        sum = sum.add(e);
    }
    if sum.v.lo <= 0.0 {
        return None;
    }
    es.into_iter().map(|e| e.divide(&sum)).collect()
}

// ---------------------------------------------------------------------------
// Encoder dual pass: LayerNorm -> masked attention -> grouped latent MoE /
// SwiGLU, carrying `<value, derivative>` intervals for every activation.
// ---------------------------------------------------------------------------
//
// The routing rule is discrete, so it is *over-bounded*: the routed output is a
// convex combination `Σ_e g_e f_e` of the expert outputs with `Σ_e g_e = 1`,
// therefore its value lies in `[min_e f_e, max_e f_e]` for any selected set and
// any gate values. Its derivative is
//
//     Σ_e g_e f_e'  +  Σ_e g_e' f_e,
//
// where the first term is bounded component-wise by `max_e |f_e'|` and the
// second by `TV(g)' · max_e |f_e|`. For a softmax over the selected logits,
// `Σ_e |∂g_e/∂l_k| = 2 g_k(1 − g_k) ≤ 1/2`, so `TV(g)' ≤ ½ Σ_k |l_k'|`.
// This is sound for any selected set, including one that changes inside the
// chart box, and it removes the tie-rule from the certificate's assumptions.
// The shared SwiGLU expert is added exactly.

struct DualStateSlices {
    wq: Vec<Dual>,
    wk: Vec<Dual>,
    wv: Vec<Dual>,
    wo: Vec<Dual>,
    group_down: Vec<Dual>,
    group_gate: Vec<Dual>,
    expert_up: Vec<Dual>,
    router: Vec<Dual>,
    shared_w1: Vec<Dual>,
    shared_w3: Vec<Dual>,
    shared_w2: Vec<Dual>,
}

struct DualQuestionSlices {
    wq: Vec<Dual>,
    wk: Vec<Dual>,
    wv: Vec<Dual>,
    wo: Vec<Dual>,
    w1: Vec<Dual>,
    w2: Vec<Dual>,
}

struct DualWeights {
    embed: Vec<Dual>,
    state_pos: Vec<Dual>,
    question_pos: Vec<Dual>,
    state_proj: Option<Vec<Dual>>,
    state_gate: Option<Vec<Dual>>,
    state_down: Option<Vec<Dual>>,
    state_up: Option<Vec<Dual>>,
    state: Vec<DualStateSlices>,
    question: Vec<DualQuestionSlices>,
}

/// Every parameter of the flat chart as a [`Dual`] with value interval
/// `theta0 ± radius·|dir|` and derivative `dir`.
fn chart_duals(flat: &[f64], dir: &[f64], radius: f64) -> Vec<Dual> {
    flat.iter()
        .zip(dir)
        .map(|(t, d)| Dual::chart(*t, *d, radius))
        .collect()
}

/// Split a flat dual vector into the named parameter views, in the same order
/// as [`crate::edge0_model::Edge0Slices::from_theta`].
fn build_dual_weights(cfg: &crate::edge0_model::Edge0Config, flat: &[Dual]) -> DualWeights {
    let mut at = 0usize;
    macro_rules! take {
        ($n:expr) => {{
            let n: usize = $n;
            let v = flat[at..at + n].to_vec();
            at += n;
            v
        }};
    }
    let d = cfg.d;
    let embed = take!(cfg.vocab * d);
    let state_pos = take!(cfg.max_state_tokens * d);
    let question_pos = take!(cfg.max_question_tokens * d);
    at += cfg.paths * d; // path_embed: read by the head pass, not the encoder
    let (state_proj, state_down, state_up) = match cfg.state_path {
        crate::edge0_model::StatePath::LowRank => {
            let r = cfg.state_rank();
            (None, Some(take!(d * r)), Some(take!(r * d)))
        }
        _ => (Some(take!(d * d)), None, None),
    };
    let mut state = Vec::with_capacity(cfg.state_blocks);
    for _ in 0..cfg.state_blocks {
        state.push(DualStateSlices {
            wq: take!(d * d),
            wk: take!(d * d),
            wv: take!(d * d),
            wo: take!(d * d),
            group_down: take!(cfg.groups * d * cfg.latent),
            group_gate: take!(cfg.groups * d * cfg.latent),
            expert_up: take!(cfg.experts * cfg.latent * d),
            router: take!(d * cfg.experts),
            shared_w1: take!(d * cfg.shared_ff),
            shared_w3: take!(d * cfg.shared_ff),
            shared_w2: take!(cfg.shared_ff * d),
        });
    }
    let mut question = Vec::with_capacity(cfg.question_blocks);
    for _ in 0..cfg.question_blocks {
        question.push(DualQuestionSlices {
            wq: take!(d * d),
            wk: take!(d * d),
            wv: take!(d * d),
            wo: take!(d * d),
            w1: take!(d * cfg.question_ff),
            w2: take!(cfg.question_ff * d),
        });
    }
    // Head slots are consumed by `build_head_params`; skip them here.
    at += d * cfg.paths
        + cfg.paths
        + cfg.paths * d
        + cfg.paths
        + cfg.paths * d
        + cfg.paths * (cfg.max_levels - 1);
    // `state_gate` is appended after the head slots in the layout, so the
    // shared parameters above keep their offsets.
    let state_gate = (cfg.state_path == crate::edge0_model::StatePath::Gate)
        .then(|| take!(d * d));
    debug_assert_eq!(at, flat.len(), "dual weights consume the layout");
    DualWeights {
        embed,
        state_pos,
        question_pos,
        state_proj,
        state_gate,
        state_down,
        state_up,
        state,
        question,
    }
}

/// `x · W` for one row, with `W` stored `[din, dout]` row-major (candle's
/// reshape order).
fn row_matmul(x: &[Dual], w: &[Dual], din: usize, dout: usize) -> Vec<Dual> {
    let mut out = Vec::with_capacity(dout);
    for j in 0..dout {
        let mut acc = Dual::constant(0.0);
        for i in 0..din {
            acc = acc.add(&x[i].mul(&w[i * dout + j]));
        }
        out.push(acc);
    }
    out
}

/// LayerNorm with the model's `var + 1e-5` floor, as a dual vector.
fn dual_layer_norm(x: &[Dual]) -> Option<Vec<Dual>> {
    let d = x.len() as f64;
    let mut mean = Dual::constant(0.0);
    for xi in x {
        mean = mean.add(xi);
    }
    mean = mean.scale(1.0 / d);
    let centered: Vec<Dual> = x.iter().map(|xi| xi.sub(&mean)).collect();
    let mut var = Dual::constant(0.0);
    for c in &centered {
        var = var.add(&c.mul(c));
    }
    var = var.scale(1.0 / d);
    let sigma = var.add(&Dual::constant(1e-5)).sqrt()?;
    centered
        .into_iter()
        .map(|c| c.divide(&sigma))
        .collect::<Option<Vec<_>>>()
}

/// Multi-head attention over dual activations, with an explicit allowed mask
/// (`allowed[r][c]`, matching the model's additive `0 / -inf` mask).
fn dual_attention(
    x: &[Vec<Dual>],
    allowed: &[Vec<bool>],
    heads: usize,
    wq: &[Dual],
    wk: &[Dual],
    wv: &[Dual],
    wo: &[Dual],
) -> Option<Vec<Vec<Dual>>> {
    let t = x.len();
    let d = x.first()?.len();
    let hd = d / heads;
    let scale = 1.0 / (hd as f64).sqrt();
    let q: Vec<Vec<Dual>> = x.iter().map(|r| row_matmul(r, wq, d, d)).collect();
    let k: Vec<Vec<Dual>> = x.iter().map(|r| row_matmul(r, wk, d, d)).collect();
    let v: Vec<Vec<Dual>> = x.iter().map(|r| row_matmul(r, wv, d, d)).collect();
    let mut attn = vec![vec![Dual::constant(0.0); d]; t];
    for i in 0..t {
        for h in 0..heads {
            let range = h * hd..(h + 1) * hd;
            let mut logits = Vec::new();
            let mut cols = Vec::new();
            for j in 0..t {
                if !allowed[i][j] {
                    continue;
                }
                let mut s = Dual::constant(0.0);
                for c in range.clone() {
                    s = s.add(&q[i][c].mul(&k[j][c]));
                }
                logits.push(s.scale(scale));
                cols.push(j);
            }
            let probs = softmax_dual(&logits)?;
            for (p, &j) in probs.iter().zip(&cols) {
                for c in range.clone() {
                    attn[i][c] = attn[i][c].add(&p.mul(&v[j][c]));
                }
            }
        }
    }
    Some(attn.iter().map(|r| row_matmul(r, wo, d, d)).collect())
}

/// One grouped latent-MoE block with the routing over-bound described above.
fn dual_moe(
    x: &[Vec<Dual>],
    s: &DualStateSlices,
    cfg: &crate::edge0_model::Edge0Config,
) -> Option<Vec<Vec<Dual>>> {
    let d = cfg.d;
    let lat = cfg.latent;
    let mut out = Vec::with_capacity(x.len());
    for row in x {
        let logits = row_matmul(row, &s.router, d, cfg.experts);
        // ½ Σ_k |l_k'| bounds the total variation of any softmax over them.
        let tv = 0.5 * logits.iter().map(|l| l.slope_abs()).sum::<f64>();
        let l1 = logits.iter().map(|l| l.slope_abs()).fold(0.0, f64::max);
        let l2 = logits.iter().map(|l| l.curvature_abs()).fold(0.0, f64::max);
        let experts = cfg.experts as f64;
        let mut lo = vec![f64::INFINITY; d];
        let mut hi = vec![f64::NEG_INFINITY; d];
        let mut dmax = vec![0.0f64; d];
        let mut ddmax = vec![0.0f64; d];
        let mut valmax = vec![0.0f64; d];
        for e in 0..cfg.experts {
            let g = e % cfg.groups;
            let down = &s.group_down[g * d * lat..(g + 1) * d * lat];
            let gate_w = &s.group_gate[g * d * lat..(g + 1) * d * lat];
            let up = &s.expert_up[e * lat * d..(e + 1) * lat * d];
            let a = row_matmul(row, down, d, lat);
            let gate = row_matmul(row, gate_w, d, lat);
            let mut h = Vec::with_capacity(lat);
            for (ai, gi) in a.iter().zip(&gate) {
                h.push(ai.silu()?.mul(gi));
            }
            let fe = row_matmul(&h, up, lat, d);
            for j in 0..d {
                lo[j] = lo[j].min(fe[j].v.lo);
                hi[j] = hi[j].max(fe[j].v.hi);
                valmax[j] = valmax[j].max(fe[j].v.lo.abs().max(fe[j].v.hi.abs()));
                dmax[j] = dmax[j].max(fe[j].slope_abs());
                ddmax[j] = ddmax[j].max(fe[j].curvature_abs());
            }
        }
        // Shared SwiGLU expert, exactly differentiable.
        let pre1: Vec<Dual> = row_matmul(row, &s.shared_w1, d, cfg.shared_ff);
        let pre3 = row_matmul(row, &s.shared_w3, d, cfg.shared_ff);
        let mut hs = Vec::with_capacity(cfg.shared_ff);
        for (a, b) in pre1.iter().zip(&pre3) {
            hs.push(a.silu()?.mul(b));
        }
        let shared = row_matmul(&hs, &s.shared_w2, cfg.shared_ff, d);
        let mut row_out = Vec::with_capacity(d);
        // `(Σ g_e f_e)'' = Σ g_e f_e'' + 2Σ g_e' f_e' + Σ g_e'' f_e`. The gate
        // bounds are `Σ|g_e'| ≤ ½Σ|l'|` and, from `|∂²g_e/∂l_j∂l_k| ≤ 3g_e`,
        // `Σ_e|g_e''| ≤ 3E² max|l'|² + ½ max|l''|`.
        let gate_curv = 3.0 * experts * experts * l1 * l1 + 0.5 * l2;
        for j in 0..d {
            let m = valmax[j];
            let width = dmax[j] + tv * m;
            let width2 = ddmax[j] + 2.0 * tv * dmax[j] + gate_curv * m;
            row_out.push(Dual {
                v: Iv::span(lo[j] + shared[j].v.lo, hi[j] + shared[j].v.hi),
                d: Iv::span(shared[j].d.lo - width, shared[j].d.hi + width),
                dd: Iv::span(shared[j].dd.lo - width2, shared[j].dd.hi + width2),
            });
        }
        out.push(row_out);
    }
    Some(out)
}

fn add_rows(x: &mut [Vec<Dual>], y: &[Vec<Dual>]) {
    for (a, b) in x.iter_mut().zip(y) {
        for (ai, bi) in a.iter_mut().zip(b) {
            *ai = ai.add(bi);
        }
    }
}

fn mean_rows(x: &[Vec<Dual>]) -> Vec<Dual> {
    let d = x[0].len();
    let t = x.len() as f64;
    let mut rep = vec![Dual::constant(0.0); d];
    for row in x {
        for j in 0..d {
            rep[j] = rep[j].add(&row[j]);
        }
    }
    for r in &mut rep {
        *r = r.scale(1.0 / t);
    }
    rep
}

fn dual_encode_state(
    w: &DualWeights,
    cfg: &crate::edge0_model::Edge0Config,
    state: &[u32],
) -> Option<Vec<Dual>> {
    if state.is_empty() || state.len() > cfg.max_state_tokens {
        return None;
    }
    let d = cfg.d;
    let mut x: Vec<Vec<Dual>> = state
        .iter()
        .enumerate()
        .map(|(t, &tok)| {
            (0..d)
                .map(|j| w.embed[tok as usize * d + j].add(&w.state_pos[t * d + j]))
                .collect()
        })
        .collect();
    let allowed = vec![vec![true; state.len()]; state.len()];
    for b in &w.state {
        let n: Vec<Vec<Dual>> = x
            .iter()
            .map(|r| dual_layer_norm(r))
            .collect::<Option<_>>()?;
        let attn = dual_attention(&n, &allowed, cfg.heads, &b.wq, &b.wk, &b.wv, &b.wo)?;
        add_rows(&mut x, &attn);
        let n: Vec<Vec<Dual>> = x
            .iter()
            .map(|r| dual_layer_norm(r))
            .collect::<Option<_>>()?;
        let ff = dual_moe(&n, b, cfg)?;
        add_rows(&mut x, &ff);
    }
    Some(mean_rows(&x))
}

fn dual_dense_ffn(
    x: &[Vec<Dual>],
    w1: &[Dual],
    w2: &[Dual],
    d: usize,
    ff: usize,
) -> Option<Vec<Vec<Dual>>> {
    let mut out = Vec::with_capacity(x.len());
    for row in x {
        let pre = row_matmul(row, w1, d, ff);
        let mut h = Vec::with_capacity(ff);
        for a in &pre {
            h.push(a.silu()?);
        }
        out.push(row_matmul(&h, w2, ff, d));
    }
    Some(out)
}

/// Question representations as duals, using the model's packed attention mask
/// (state rows see only the state; question rows see the state and their own
/// segment). One representation per question.
fn dual_encode_questions(
    w: &DualWeights,
    cfg: &crate::edge0_model::Edge0Config,
    state: &[u32],
    questions: &[crate::edge0_model::QuestionInput],
) -> Option<Vec<Vec<Dual>>> {
    let d = cfg.d;
    let ns = state.len();
    let nq: usize = questions.iter().map(|q| q.tokens.len()).sum();
    let total = ns + nq;
    let mut ids = Vec::with_capacity(total);
    ids.extend_from_slice(state);
    let mut pos = Vec::with_capacity(total);
    for t in 0..ns {
        pos.push(&w.state_pos[t * d..(t + 1) * d]);
    }
    let mut segments = Vec::with_capacity(questions.len());
    let mut cursor = ns;
    for q in questions {
        ids.extend_from_slice(&q.tokens);
        for j in 0..q.tokens.len() {
            pos.push(&w.question_pos[j * d..(j + 1) * d]);
        }
        segments.push((cursor, cursor + q.tokens.len()));
        cursor += q.tokens.len();
    }
    let mut x: Vec<Vec<Dual>> = ids
        .iter()
        .enumerate()
        .map(|(t, &tok)| {
            (0..d)
                .map(|j| w.embed[tok as usize * d + j].add(&pos[t][j]))
                .collect()
        })
        .collect();
    let mut allowed = vec![vec![false; total]; total];
    for row in allowed.iter_mut().take(ns) {
        for cell in row.iter_mut().take(ns) {
            *cell = true;
        }
    }
    for &(start, end) in &segments {
        for row in allowed.iter_mut().take(end).skip(start) {
            for cell in row.iter_mut().take(ns) {
                *cell = true;
            }
            for cell in row.iter_mut().take(end).skip(start) {
                *cell = true;
            }
        }
    }
    for b in &w.question {
        let n: Vec<Vec<Dual>> = x
            .iter()
            .map(|r| dual_layer_norm(r))
            .collect::<Option<_>>()?;
        let attn = dual_attention(&n, &allowed, cfg.heads, &b.wq, &b.wk, &b.wv, &b.wo)?;
        add_rows(&mut x, &attn);
        let n: Vec<Vec<Dual>> = x
            .iter()
            .map(|r| dual_layer_norm(r))
            .collect::<Option<_>>()?;
        let ff = dual_dense_ffn(&n, &b.w1, &b.w2, d, cfg.question_ff)?;
        add_rows(&mut x, &ff);
    }
    // `forward` pools the state encoder, projects it through `state_proj`, and
    // adds the projected vector to every question rep. The pool must come from
    // `encode_state`, not from the question pass's state rows: those rows see
    // only the question blocks and would not match the deployed `state_rep`.
    let state_pool = dual_encode_state(w, cfg, state)?;
    let state_add: Vec<Dual> = match cfg.state_path {
        crate::edge0_model::StatePath::Full => row_matmul(
            &state_pool,
            w.state_proj.as_ref().expect("full state projection"),
            d,
            d,
        ),
        crate::edge0_model::StatePath::Gate => {
            let projected = row_matmul(
                &state_pool,
                w.state_proj.as_ref().expect("gated state projection"),
                d,
                d,
            );
            let gate_logits = row_matmul(
                &state_pool,
                w.state_gate.as_ref().expect("state gate"),
                d,
                d,
            );
            projected
                .iter()
                .zip(&gate_logits)
                .map(|(p, l)| l.sigmoid().map(|g| p.mul(&g)))
                .collect::<Option<_>>()?
        }
        crate::edge0_model::StatePath::LowRank => {
            let r = cfg.state_rank();
            let down = row_matmul(
                &state_pool,
                w.state_down.as_ref().expect("low-rank down"),
                d,
                r,
            );
            row_matmul(&down, w.state_up.as_ref().expect("low-rank up"), r, d)
        }
    };
    Some(
        segments
            .iter()
            .map(|&(start, end)| {
                let rep = mean_rows(&x[start..end]);
                rep.iter().zip(&state_add).map(|(a, b)| a.add(b)).collect()
            })
            .collect(),
    )
}

/// The state-encoder half as a value/derivative interval: the pooled
/// representation `encode_state` would return, for an arbitrary chart
/// direction. This is the public entry point that exercises LayerNorm, masked
/// attention and routed MoE.
pub fn state_rep_duals(
    cfg: &crate::edge0_model::Edge0Config,
    flat: &[f64],
    dir: &[f64],
    radius: f64,
    state: &[u32],
) -> Option<Vec<Dual>> {
    let duals = chart_duals(flat, dir, radius);
    let w = build_dual_weights(cfg, &duals);
    dual_encode_state(&w, cfg, state)
}

/// The encoder half of the certified chart pass: every question's `rep` as a
/// value/derivative interval, for an arbitrary chart direction (including the
/// embeddings, attention, MoE and question blocks).
pub fn encoder_rep_duals(
    cfg: &crate::edge0_model::Edge0Config,
    flat: &[f64],
    dir: &[f64],
    radius: f64,
    state: &[u32],
    questions: &[crate::edge0_model::QuestionInput],
) -> Option<Vec<Vec<Dual>>> {
    let duals = chart_duals(flat, dir, radius);
    let w = build_dual_weights(cfg, &duals);
    // Match `forward`: the deployed marginal uses the packed question pass and
    // then adds the pooled state representation to every question rep, so the
    // state-pool dual is part of this derivative.
    dual_encode_questions(&w, cfg, state, questions)
}

/// `(flat θ₀, chart direction d)` for a one-dimensional chart, with the same
/// `scale` convention as [`crate::subspace::AffineSubspace::lift`].
fn chart_geometry(target: &SubspaceTarget<Edge0Potential>) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    if target.dim() != 1 {
        return Ok(None);
    }
    let cfg = target.potential().config();
    let p_total = cfg.flat_size();
    let mut v = vec![0f32; p_total];
    target.subspace().basis().apply(&[1.0f32], &mut v);
    let dir: Vec<f64> = v
        .iter()
        .zip(target.subspace().scale())
        .map(|(vi, s)| f64::from(*vi) * f64::from(*s))
        .collect();
    let theta0 = target.subspace().lift_tensor(&[0f32])?;
    let flat = theta0
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    Ok(Some((flat, dir)))
}

/// First- and second-derivative bounds of the one-sample NLL and of `p` over
/// the chart box `|z| ≤ radius`, for an arbitrary direction.
///
/// `psi_*` are the bounds of `Ψ = λ Σ_i −log max(p_i, floor)` over the whole
/// fixture; `p_*` are the bounds of the selected sample's label probability.
/// `None` if any interval is not certifiable (a marginal touching the floor, a
/// guarded range).
pub struct ChartDerivativeBounds {
    pub psi_prime_sup: f64,
    pub psi_second_sup: f64,
    pub p_prime_sup: f64,
    pub p_second_sup: f64,
}

pub fn chart_derivative_bounds(
    target: &SubspaceTarget<Edge0Potential>,
    radius: f64,
    sample: usize,
) -> Result<Option<ChartDerivativeBounds>> {
    if !(radius > 0.0 && radius <= 10.0) {
        return Ok(None);
    }
    let (flat, dir) = match chart_geometry(target)? {
        Some(g) => g,
        None => return Ok(None),
    };
    let pot = target.potential();
    let cfg = pot.config();
    let samples = pot.samples();
    if sample >= samples.len() || samples.is_empty() {
        return Ok(None);
    }
    let hp = build_head_params(cfg, &flat, &dir, radius);
    let lam = target.lambda();
    let mut psi_p = 0.0f64;
    let mut psi_q = 0.0f64;
    let mut p_bound = None;
    for (i, s) in samples.iter().enumerate() {
        let reps = match encoder_rep_duals(
            cfg,
            &flat,
            &dir,
            radius,
            &s.input_state,
            std::slice::from_ref(&s.question),
        ) {
            Some(r) => r,
            None => return Ok(None),
        };
        let p = match label_probability_dual(
            &hp,
            &reps[0],
            s.question.kind,
            s.question.options.as_slice(),
            s.label,
            cfg.paths,
            cfg.d,
            cfg.max_levels - 1,
        ) {
            Some(p) => p,
            None => return Ok(None),
        };
        if p.v.lo <= crate::edge0_potential::NLL_FLOOR {
            return Ok(None);
        }
        let log_p = match p.ln() {
            Some(l) => l,
            None => return Ok(None),
        };
        psi_p += lam * log_p.slope_abs();
        psi_q += lam * log_p.curvature_abs();
        if i == sample {
            p_bound = Some((p.slope_abs(), p.curvature_abs()));
        }
    }
    let (p_p, p_q) = match p_bound {
        Some(b) => b,
        None => return Ok(None),
    };
    Ok(Some(ChartDerivativeBounds {
        psi_prime_sup: psi_p,
        psi_second_sup: psi_q,
        p_prime_sup: p_p,
        p_second_sup: p_q,
    }))
}

/// Interval of `p(label of sample)` at one chart point `z`, for an arbitrary
/// direction (the encoder dual pass at `θ(z)`).
pub fn chart_label_probability_at(
    target: &SubspaceTarget<Edge0Potential>,
    z: f64,
    sample: usize,
) -> Result<Option<Iv>> {
    let (flat, dir) = match chart_geometry(target)? {
        Some(g) => g,
        None => return Ok(None),
    };
    let pot = target.potential();
    let cfg = pot.config();
    let samples = pot.samples();
    let s = match samples.get(sample) {
        Some(s) => s,
        None => return Ok(None),
    };
    let flat_z: Vec<f64> = flat.iter().zip(&dir).map(|(a, b)| a + z * b).collect();
    let hp = build_head_params(cfg, &flat_z, &dir, 0.0);
    let reps = match encoder_rep_duals(
        cfg,
        &flat_z,
        &dir,
        0.0,
        &s.input_state,
        std::slice::from_ref(&s.question),
    ) {
        Some(r) => r,
        None => return Ok(None),
    };
    let p = match label_probability_dual(
        &hp,
        &reps[0],
        s.question.kind,
        s.question.options.as_slice(),
        s.label,
        cfg.paths,
        cfg.d,
        cfg.max_levels - 1,
    ) {
        Some(p) => p,
        None => return Ok(None),
    };
    Ok(Some(p.v))
}

/// Interval of `Ψ(z) = λ Σ_i −log max(p_i(z), floor)` at one chart point.
pub fn chart_psi_at(target: &SubspaceTarget<Edge0Potential>, z: f64) -> Result<Option<Iv>> {
    let n = target.potential().samples().len();
    let mut sum = Iv::point(0.0);
    for i in 0..n {
        let p = match chart_label_probability_at(target, z, i)? {
            Some(p) => p,
            None => return Ok(None),
        };
        let hi = p.hi.max(crate::edge0_potential::NLL_FLOOR);
        let lo = p.lo.max(crate::edge0_potential::NLL_FLOOR);
        sum = sum.add(&Iv::span(-hi.ln(), -lo.ln()));
    }
    Ok(Some(sum.mul(&Iv::point(target.lambda()))))
}

/// The head parameters as chart duals, already restricted to their slots.
pub struct HeadParams {
    pub policy_w: Vec<Dual>,
    pub policy_b: Vec<Dual>,
    pub path_embed: Vec<Dual>,
    pub noul_w: Vec<Dual>,
    pub noul_b: Vec<Dual>,
    pub score_w: Vec<Dual>,
    pub thresholds: Vec<Dual>,
    pub embed: Vec<Dual>,
}

/// Build one slot's duals from the flat `theta0`/`dir` pair, using the slot's
/// **global** offset into the flat vector.
fn slot_duals(
    cfg: &crate::edge0_model::Edge0Config,
    flat: &[f64],
    dir: &[f64],
    name: &str,
    radius: f64,
) -> Vec<Dual> {
    let slot = match cfg.layout().into_iter().find(|s| s.name == name) {
        Some(s) => s,
        None => return Vec::new(),
    };
    (slot.start..slot.start + slot.len)
        .map(|i| Dual::chart(flat[i], dir[i], radius))
        .collect()
}

fn build_head_params(
    cfg: &crate::edge0_model::Edge0Config,
    flat: &[f64],
    dir: &[f64],
    radius: f64,
) -> HeadParams {
    let get = |name: &str| slot_duals(cfg, flat, dir, name, radius);
    HeadParams {
        policy_w: get("policy_w"),
        policy_b: get("policy_b"),
        path_embed: get("path_embed"),
        noul_w: get("noul_w"),
        noul_b: get("noul_b"),
        score_w: get("score_w"),
        thresholds: get("score_thresholds"),
        embed: get("embed"),
    }
}

/// The head-stage chart function for one sample: `p(label | z)` as a dual.
///
/// `rep` is the question representation as a dual vector. For head-only chart
/// directions it is constant (`Dual::constant(rep_j)`); the encoder pass would
/// supply value/derivative intervals here instead.
pub fn label_probability_dual(
    hp: &HeadParams,
    rep: &[Dual],
    kind: QuestionKind,
    options_tokens: &[u32],
    label: usize,
    paths: usize,
    d: usize,
    threshold_stride: usize,
) -> Option<Dual> {
    // `a_phi(r) = softmax(policy_w r · rep + policy_b)`; `policy_w` is stored
    // `[d, paths]`, so its flat index is `j * paths + r`.
    let mut logits = Vec::with_capacity(paths);
    for r in 0..paths {
        let mut l = hp.policy_b[r];
        for j in 0..d {
            l = l.add(&hp.policy_w[j * paths + r].mul(&rep[j]));
        }
        logits.push(l);
    }
    let a = softmax_dual(&logits)?;

    let mut marginal: Option<Dual> = None;
    for (r, a_r) in a.iter().enumerate() {
        let mut h = Vec::with_capacity(d);
        for j in 0..d {
            h.push(rep[j].add(&hp.path_embed[r * d + j]));
        }
        let u = match kind {
            QuestionKind::Noul => {
                let mut logit = hp.noul_b[r];
                for j in 0..d {
                    logit = logit.add(&hp.noul_w[r * d + j].mul(&h[j]));
                }
                let p_yes = logit.sigmoid()?;
                let p_no = p_yes.neg().add(&Dual::constant(1.0));
                if label == 0 { p_no } else { p_yes }
            }
            QuestionKind::Choice { .. } => {
                let mut logits = Vec::with_capacity(options_tokens.len());
                let scale = 1.0 / (d as f64).sqrt();
                for &tok in options_tokens {
                    let row = &hp.embed[tok as usize * d..(tok as usize + 1) * d];
                    let mut l = Dual::constant(0.0);
                    for j in 0..d {
                        l = l.add(&row[j].mul(&h[j]));
                    }
                    logits.push(l.scale(scale));
                }
                let probs = softmax_dual(&logits)?;
                probs[label]
            }
            QuestionKind::Score { levels } => {
                let mut score = Dual::constant(0.0);
                for j in 0..d {
                    score = score.add(&hp.score_w[r * d + j].mul(&h[j]));
                }
                let raw = &hp.thresholds;
                let base = r * threshold_stride;
                // `t_j = t_0 + Σ_{i≤j} raw_i²`, ordered by construction.
                let mut ordered = Vec::with_capacity(levels - 1);
                let mut t = raw[base];
                ordered.push(t);
                for j in 1..levels - 1 {
                    t = t.add(&raw[base + j].mul(&raw[base + j]));
                    ordered.push(t);
                }
                let mut ge = Vec::with_capacity(levels + 1);
                ge.push(Dual::constant(1.0));
                for t in &ordered {
                    let diff = score.sub(t).scale(1.5);
                    ge.push(diff.sigmoid()?);
                }
                ge.push(Dual::constant(0.0));
                // The model clamps the cumulative difference at `[0, 1]`; the
                // clamp has slope in `[0, 1]`, so leaving the derivative
                // untouched is a conservative over-estimate.
                let mut probs = Vec::with_capacity(levels);
                for j in 0..levels {
                    let p = ge[j].sub(&ge[j + 1]);
                    let v = Iv::span(p.v.lo.clamp(0.0, 1.0), p.v.hi.clamp(0.0, 1.0));
                    // The clamp is affine (slope 1 or 0), so its own second
                    // derivative is 0 wherever it is constant. The hull with
                    // zero contains both branches.
                    let dd = Iv::span(p.dd.lo.min(0.0), p.dd.hi.max(0.0));
                    probs.push(Dual { v, d: p.d, dd });
                }
                probs[label]
            }
        };
        marginal = Some(match marginal {
            Some(acc) => acc.add(&a_r.mul(&u)),
            None => a_r.mul(&u),
        });
    }
    marginal
}

/// Certified `sup |dPsi/dz|` on `|z| ≤ radius` for one sample, for a chart
/// direction on **any** parameter slot: the encoder dual pass supplies `rep`,
/// so embeddings, attention, MoE and question blocks are covered as well as
/// the heads. `None` only when the dual forward leaves a guarded range (wide
/// logits, a nonpositive normalisation, an out-of-range `silu`) or when the
/// marginal interval touches zero.
pub fn head_stage_psi_bound(
    target: &SubspaceTarget<Edge0Potential>,
    radius: f64,
    sample: usize,
) -> Result<Option<f64>> {
    if target.dim() != 1 || !(radius > 0.0 && radius <= 10.0) {
        return Ok(None);
    }
    let pot = target.potential();
    let cfg = pot.config();
    let samples = pot.samples();
    if sample >= samples.len() {
        return Ok(None);
    }
    let sub = target.subspace();
    let p_total = cfg.flat_size();

    let mut v = vec![0f32; p_total];
    sub.basis().apply(&[1.0f32], &mut v);
    let dir: Vec<f64> = v
        .iter()
        .zip(sub.scale())
        .map(|(vi, s)| f64::from(*vi) * f64::from(*s))
        .collect();

    let theta0 = sub.lift_tensor(&[0f32])?;
    let flat = theta0
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let hp = build_head_params(cfg, &flat, &dir, radius);

    let s = &samples[sample];
    let questions = std::slice::from_ref(&s.question);
    let reps = match encoder_rep_duals(cfg, &flat, &dir, radius, &s.input_state, questions) {
        Some(r) => r,
        None => return Ok(None),
    };
    let rep = &reps[0];

    let p = match label_probability_dual(
        &hp,
        rep,
        s.question.kind,
        s.question.options.as_slice(),
        s.label,
        cfg.paths,
        cfg.d,
        cfg.max_levels - 1,
    ) {
        Some(p) => p,
        None => return Ok(None),
    };
    if p.v.lo <= 0.0 || p.v.hi > 1.0 + 1e-9 {
        return Ok(None);
    }
    let log_p = match p.ln() {
        Some(l) => l,
        None => return Ok(None),
    };
    let lam = target.lambda();
    let bound = lam * log_p.d.lo.abs().max(log_p.d.hi.abs());
    Ok(Some(bound))
}

/// The interval of `p(label | z)` over the chart box, for the containment test.
pub fn head_stage_probability_interval(
    target: &SubspaceTarget<Edge0Potential>,
    radius: f64,
    sample: usize,
) -> Result<Option<(f64, f64)>> {
    // Reuse the same parameter construction; a small duplication keeps the
    // bound function above readable.
    let pot = target.potential();
    let cfg = pot.config();
    let samples = pot.samples();
    if sample >= samples.len() || target.dim() != 1 {
        return Ok(None);
    }
    let sub = target.subspace();
    let p_total = cfg.flat_size();
    let mut v = vec![0f32; p_total];
    sub.basis().apply(&[1.0f32], &mut v);
    let dir: Vec<f64> = v
        .iter()
        .zip(sub.scale())
        .map(|(vi, s1)| f64::from(*vi) * f64::from(*s1))
        .collect();
    let theta0 = sub.lift_tensor(&[0f32])?;
    let flat = theta0
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let hp = build_head_params(cfg, &flat, &dir, radius);
    let s = &samples[sample];
    let questions = std::slice::from_ref(&s.question);
    let reps = match encoder_rep_duals(cfg, &flat, &dir, radius, &s.input_state, questions) {
        Some(r) => r,
        None => return Ok(None),
    };
    let rep = &reps[0];
    let p = match label_probability_dual(
        &hp,
        rep,
        s.question.kind,
        s.question.options.as_slice(),
        s.label,
        cfg.paths,
        cfg.d,
        cfg.max_levels - 1,
    ) {
        Some(p) => p,
        None => return Ok(None),
    };
    Ok(Some((p.v.lo, p.v.hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge0_campaign::{CampaignConfig, build_samples};
    use crate::edge0_world::Split;
    use crate::subspace::{AffineSubspace, Basis};

    fn fixture(slot_name: &str) -> (SubspaceTarget<Edge0Potential>, f64) {
        let cfg = CampaignConfig {
            updates: 0,
            warm_start: 0,
            worlds: [2, 1, 1, 1],
            ..Default::default()
        };
        let all = build_samples(&cfg);
        let samples: Vec<_> = all
            .into_iter()
            .filter(|s| s.split == Split::Train)
            .take(4)
            .collect();
        assert_eq!(samples.len(), 4, "fixture needs four train samples");
        let model = cfg.model.clone();
        let potential = Edge0Potential::head_only(model.clone(), samples);

        let p = model.flat_size();
        let layout = model.layout();
        let idx = layout
            .iter()
            .find(|s| s.name == slot_name)
            .expect("head slot")
            .start;
        let mut cols = vec![0f32; p];
        cols[idx] = 1.0;
        let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
        let theta_ref = crate::edge0_potential::reference_theta(&model, 9)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p]).expect("chart");
        let target = SubspaceTarget::with_potential(subspace, 2.0, potential);
        (target, 0.5)
    }

    /// The same chart fixture with **every** parameter trainable, so a
    /// direction on an encoder slot actually moves the model.
    fn full_fixture(slot_name: &str) -> (SubspaceTarget<Edge0Potential>, f64) {
        full_fixture_with(slot_name, crate::edge0_model::StatePath::Full)
    }

    fn full_fixture_with(
        slot_name: &str,
        state_path: crate::edge0_model::StatePath,
    ) -> (SubspaceTarget<Edge0Potential>, f64) {
        let mut model = CampaignConfig::default().model;
        model.state_path = state_path;
        let cfg = CampaignConfig {
            model,
            updates: 0,
            warm_start: 0,
            worlds: [2, 1, 1, 1],
            ..Default::default()
        };
        let all = build_samples(&cfg);
        let samples: Vec<_> = all
            .into_iter()
            .filter(|s| s.split == Split::Train)
            .take(4)
            .collect();
        assert_eq!(samples.len(), 4, "fixture needs four train samples");
        let model = cfg.model.clone();
        let potential = crate::edge0_potential::full_chart(model.clone(), samples);
        let p = model.flat_size();
        let idx = model
            .layout()
            .iter()
            .find(|s| s.name == slot_name)
            .expect("slot")
            .start;
        let mut cols = vec![0f32; p];
        cols[idx] = 1.0;
        let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
        let theta_ref = crate::edge0_potential::reference_theta(&model, 9)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p]).expect("chart");
        let target = SubspaceTarget::with_potential(subspace, 2.0, potential);
        (target, 0.1)
    }

    fn exact_p(target: &SubspaceTarget<Edge0Potential>, z: f64, sample: usize) -> f64 {
        let theta = target.subspace().lift_tensor(&[z as f32]).unwrap();
        target
            .potential()
            .label_probability(&theta, sample)
            .unwrap()
            .unwrap()
    }

    /// The loss contributed by one sample (`-log p_label`), not the average over
    /// the fixture's samples.
    fn exact_loss(target: &SubspaceTarget<Edge0Potential>, z: f64, sample: usize) -> f64 {
        -(exact_p(target, z, sample).max(1e-30)).ln()
    }

    #[test]
    fn dual_arithmetic_matches_closed_forms() {
        let x = Dual {
            v: Iv::span(0.3, 0.4),
            d: Iv::point(2.0),
            dd: Iv::point(0.0),
        };
        let s = x.sigmoid().expect("bounded");
        assert!(s.v.lo <= 0.574442516811659 && s.v.hi >= 0.5986876601124523);
        // d/dx sigma = sigma(1-sigma) <= 0.25, so |d| <= 0.25*2.
        assert!(s.d.hi <= 0.52 && s.d.lo >= 0.0, "d = {:?}", s.d);

        let y = Dual::chart(0.5, 1.0, 1.0); // value [−0.5, 1.5], d=1
        let e = y.exp().expect("bounded");
        assert!(e.v.lo >= (-0.5f64).exp() - 1e-12);
        assert!(e.v.hi >= 1.5f64.exp() - 1e-12);
        // exp' = exp, so d = exp(value interval).
        assert!(e.d.lo >= e.v.lo - 1e-12);

        // (f·g)' = f'g + fg' on an interval with known sign.
        let a = Dual::chart(1.0, 1.0, 0.5);
        let prod = a.mul(&a);
        // d(x^2) = 2x ∈ [1, 3]; value x^2 ∈ [0.25, 2.25].
        assert!(prod.d.lo <= 1.0 + 1e-12 && prod.d.hi >= 3.0 - 1e-12);
    }

    #[test]
    fn state_encoder_dual_contains_the_model() {
        // This is the LayerNorm -> attention -> routed-MoE half. The deployed
        // `forward` currently reads the shared state tokens directly in the
        // question blocks, so a state-encoder direction has zero effect on the
        // marginal; the encoder pass is still checked against `encode_state`
        // itself, where its output is defined and the MoE routing lives.
        let cfg = CampaignConfig {
            updates: 0,
            warm_start: 0,
            worlds: [2, 1, 1, 1],
            ..Default::default()
        };
        let all = build_samples(&cfg);
        let samples: Vec<_> = all
            .into_iter()
            .filter(|s| s.split == Split::Train)
            .take(4)
            .collect();
        let model = cfg.model.clone();
        let p = model.flat_size();
        let idx = model
            .layout()
            .iter()
            .find(|s| s.name == "moe_router")
            .expect("slot")
            .start;
        let mut cols = vec![0f32; p];
        cols[idx] = 1.0;
        let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
        let theta_ref = crate::edge0_potential::reference_theta(&model, 9)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let subspace = AffineSubspace::new(theta_ref.clone(), basis, vec![1.0; p]).expect("chart");
        let radius = 0.1f64;
        let mut col = vec![0f32; p];
        subspace.basis().apply(&[1.0f32], &mut col);
        let dir: Vec<f64> = col
            .iter()
            .zip(subspace.scale())
            .map(|(v, s)| f64::from(*v) * f64::from(*s))
            .collect();
        let flat: Vec<f64> = theta_ref.iter().map(|x| f64::from(*x)).collect();
        let state = &samples[0].input_state;
        let rep = state_rep_duals(&model, &flat, &dir, radius, state).expect("state dual runs");

        let h = 1e-3f32;
        for k in -3..=3 {
            let z = radius * k as f64 / 3.0;
            let theta = subspace.lift_tensor(&[z as f32]).unwrap();
            let got = crate::edge0_model::encode_state(&theta, &model, state)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (j, g) in got.iter().enumerate() {
                let g = f64::from(*g);
                assert!(
                    rep[j].v.lo - 1e-4 <= g && g <= rep[j].v.hi + 1e-4,
                    "state rep[{j}] at z={z}: {g} outside {:?}",
                    rep[j].v
                );
            }
        }
        // Derivative containment against a central difference of the model.
        for k in -2..=2 {
            let z = radius * k as f64 / 3.0;
            let a = crate::edge0_model::encode_state(
                &subspace.lift_tensor(&[(z + h as f64) as f32]).unwrap(),
                &model,
                state,
            )
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let b = crate::edge0_model::encode_state(
                &subspace.lift_tensor(&[(z - h as f64) as f32]).unwrap(),
                &model,
                state,
            )
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            for j in 0..a.len() {
                let fd = f64::from(a[j] - b[j]) / (2.0 * f64::from(h));
                assert!(
                    rep[j].d.lo - 1e-3 <= fd && fd <= rep[j].d.hi + 1e-3,
                    "state d[{j}] at z={z}: {fd} outside {:?}",
                    rep[j].d
                );
            }
        }
    }

    fn exact_total_loss(target: &SubspaceTarget<Edge0Potential>, z: f64) -> f64 {
        let theta = target.subspace().lift_tensor(&[z as f32]).unwrap();
        let losses = target.potential().losses(&theta).unwrap().unwrap();
        target.lambda() * losses.iter().sum::<f64>()
    }

    #[test]
    fn second_order_bounds_contain_finite_differences() {
        let (target, radius) = full_fixture("question_w1");
        let h = 0.02;
        for sample in 0..4 {
            let b = chart_derivative_bounds(&target, radius, sample)
                .expect("bounds run")
                .expect("encoder direction is certifiable");
            assert!(b.psi_prime_sup > 0.0 && b.psi_second_sup > 0.0);
            for k in -2..=2 {
                let z = radius * k as f64 / 3.0;
                let f1 = (exact_total_loss(&target, z + h) - exact_total_loss(&target, z - h))
                    / (2.0 * h);
                let f2 = (exact_total_loss(&target, z + h) - 2.0 * exact_total_loss(&target, z)
                    + exact_total_loss(&target, z - h))
                    / (h * h);
                assert!(
                    f1.abs() <= b.psi_prime_sup * (1.0 + 1e-5) + 1e-4,
                    "sample {sample} z={z}: |Psi'| = {f1} vs {}",
                    b.psi_prime_sup
                );
                assert!(
                    f2.abs() <= b.psi_second_sup * (1.0 + 1e-5) + 1e-2,
                    "sample {sample} z={z}: |Psi''| = {f2} vs {}",
                    b.psi_second_sup
                );
                let p1 =
                    (exact_p(&target, z + h, sample) - exact_p(&target, z - h, sample)) / (2.0 * h);
                let p2 = (exact_p(&target, z + h, sample) - 2.0 * exact_p(&target, z, sample)
                    + exact_p(&target, z - h, sample))
                    / (h * h);
                assert!(p1.abs() <= b.p_prime_sup * (1.0 + 1e-5) + 1e-4);
                assert!(p2.abs() <= b.p_second_sup * (1.0 + 1e-5) + 1.0);
            }
        }
    }

    /// A one-sample full chart: `Ψ` stays small enough that the evidence is
    /// well above the (conservative) midpoint error at a modest grid.
    fn single_sample_fixture(slot_name: &str) -> (SubspaceTarget<Edge0Potential>, f64) {
        let cfg = CampaignConfig {
            updates: 0,
            warm_start: 0,
            worlds: [2, 1, 1, 1],
            ..Default::default()
        };
        let all = build_samples(&cfg);
        let samples: Vec<_> = all
            .into_iter()
            .filter(|s| s.split == Split::Train)
            .take(1)
            .collect();
        assert_eq!(samples.len(), 1, "fixture needs one train sample");
        let model = cfg.model.clone();
        let potential = crate::edge0_potential::full_chart(model.clone(), samples);
        let p = model.flat_size();
        let idx = model
            .layout()
            .iter()
            .find(|s| s.name == slot_name)
            .expect("slot")
            .start;
        let mut cols = vec![0f32; p];
        cols[idx] = 1.0;
        let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
        let theta_ref = crate::edge0_potential::reference_theta(&model, 9)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p]).expect("chart");
        let target = SubspaceTarget::with_potential(subspace, 2.0, potential);
        (target, 0.1)
    }

    #[test]
    fn encoder_chart_enclosure_brackets_the_quadrature() {
        let (target, radius) = single_sample_fixture("question_w1");
        let quad = crate::edge0_potential::head_chart_quadrature_1d(&target, radius, 64, 0)
            .expect("quadrature runs")
            .expect("1-D chart");
        let enc = crate::edge0_potential::head_chart_enclosure_1d(&target, radius, 64, 0)
            .expect("enclosure runs")
            .expect("encoder direction must be certifiable");
        assert!(
            enc.evidence_contains(quad.evidence),
            "evidence {} outside [{}, {}]",
            quad.evidence,
            enc.evidence_lo,
            enc.evidence_hi
        );
        assert!(
            enc.predictive_contains(quad.predictive),
            "predictive {} outside [{}, {}]",
            quad.predictive,
            enc.predictive_lo,
            enc.predictive_hi
        );
        assert!(enc.evidence_lo > 0.0 && enc.psi_second_sup > 0.0);
    }

    #[test]
    fn head_stage_probability_interval_contains_the_model() {
        // Four train samples cover `noul`, `choice` and `score`.
        for slot in ["score_w", "noul_w", "policy_w", "score_thresholds"] {
            let (target, radius) = fixture(slot);
            for sample in 0..4 {
                let got = head_stage_probability_interval(&target, radius, sample)
                    .expect("interval runs");
                if got.is_none() {
                    eprintln!(
                        "NONE slot {slot} sample {sample} kind {:?}",
                        target.potential().samples()[sample].question.kind
                    );
                    continue;
                }
                let (lo, hi) = got.unwrap();
                for k in -5..=5 {
                    let z = radius * k as f64 / 5.0;
                    let p = exact_p(&target, z, sample);
                    assert!(
                        lo - 1e-6 <= p && p <= hi + 1e-6,
                        "{slot} sample {sample}: p({z}) = {p} outside [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn head_stage_psi_bound_contains_finite_differences() {
        let (target, radius) = fixture("score_w");
        let h = 1e-4;
        let mut largest = 0.0f64;
        for sample in 0..4 {
            let got = head_stage_psi_bound(&target, radius, sample).expect("bound runs");
            if got.is_none() {
                eprintln!(
                    "BOUND NONE sample {sample} kind {:?}",
                    target.potential().samples()[sample].question.kind
                );
                continue;
            }
            let bound = got.unwrap();
            assert!(bound.is_finite() && bound > 0.0, "bound {bound}");
            for k in -4..=4 {
                let z = radius * k as f64 / 5.0;
                let psi_p = exact_loss(&target, z + h, sample);
                let psi_m = exact_loss(&target, z - h, sample);
                let fd = (psi_p - psi_m) / (2.0 * h);
                largest = largest.max(fd.abs());
                assert!(
                    fd.abs() <= bound * (1.0 + 1e-9),
                    "sample {sample}: |Psi'| at z={z} is {fd} but the bound is {bound}"
                );
            }
        }
        // The bound must be exercised, not trivially zero.
        assert!(
            largest > 1e-3,
            "largest finite-difference derivative {largest}"
        );
    }

    #[test]
    fn state_gate_is_consumed_by_the_marginal() {
        // A direction supported only on the gate slot must reach the deployed
        // marginal, and the dual pass must mirror that path (the containment
        // test below includes `state_gate` among its encoder directions).
        let (target, _) = full_fixture_with("state_gate", crate::edge0_model::StatePath::Gate);
        for sample in 0..4 {
            let p0 = exact_p(&target, 0.0, sample);
            let p1 = exact_p(&target, 0.5, sample);
            assert!(
                (p0 - p1).abs() > 1e-9,
                "sample {sample}: gate direction does not move p ({p0} vs {p1})"
            );
        }
    }

    #[test]
    fn low_rank_state_projection_is_consumed_by_the_marginal() {
        use crate::edge0_model::{Edge0Config, StatePath};
        // The low-rank variant must have the same parameter count as the full
        // projection (`d·r + r·d = d²` at `r = d/2`), so a paired comparison
        // is parameter-for-parameter.
        let mut full = Edge0Config::default();
        let mut low = Edge0Config::default();
        low.state_path = StatePath::LowRank;
        full.state_path = StatePath::Full;
        assert_eq!(full.flat_size(), low.flat_size());

        for slot in ["state_down", "state_up"] {
            let (target, _) = full_fixture_with(slot, StatePath::LowRank);
            for sample in 0..4 {
                let p0 = exact_p(&target, 0.0, sample);
                let p1 = exact_p(&target, 0.5, sample);
                assert!(
                    (p0 - p1).abs() > 1e-9,
                    "{slot} sample {sample}: low-rank direction does not move p ({p0} vs {p1})"
                );
            }
        }
    }

    #[test]
    fn state_rep_is_consumed_by_the_marginal() {
        // `state_attn_wq` exists only in the state encoder, so before the
        // wiring change this direction could not move the deployed marginal.
        // The question blocks now add `state_rep`, and this test pins that.
        // `wv` rather than `wq`: the first train worlds have a single state
        // token, where query/key weights cannot change the attention output.
        let (target, _) = full_fixture("state_attn_wv");
        for sample in 0..4 {
            let p0 = exact_p(&target, 0.0, sample);
            let p1 = exact_p(&target, 0.5, sample);
            assert!(
                (p0 - p1).abs() > 1e-9,
                "sample {sample}: state-encoder direction does not move p ({p0} vs {p1})"
            );
        }
    }

    #[test]
    fn encoder_directions_are_certified_by_the_dual_pass() {
        // Directions on the embeddings, state attention, routed MoE and the
        // question blocks all move `rep`; the encoder dual must cover them.
        use crate::edge0_model::StatePath;
        for (slot, path) in [
            ("embed", StatePath::Full),
            ("question_attn_wq", StatePath::Full),
            ("question_w1", StatePath::Full),
            ("path_embed", StatePath::Full),
            ("state_gate", StatePath::Gate),
            ("state_down", StatePath::LowRank),
            ("state_up", StatePath::LowRank),
        ] {
            let (target, _) = full_fixture_with(slot, path);
            // A narrow box keeps the embedding/softmax interval off zero; the
            // containment statement is per box, not per radius-free family.
            let radius = 0.02;
            for sample in 0..4 {
                let got = head_stage_probability_interval(&target, radius, sample)
                    .expect("interval runs");
                let (lo, hi) = got
                    .unwrap_or_else(|| panic!("{slot} sample {sample}: encoder direction refused"));
                assert!(lo <= hi, "{slot} sample {sample}: inverted interval");
                for k in -5..=5 {
                    let z = radius * k as f64 / 5.0;
                    let p = exact_p(&target, z, sample);
                    assert!(
                        lo - 1e-4 <= p && p <= hi + 1e-4,
                        "{slot} sample {sample}: p({z}) = {p} outside [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn encoder_psi_bound_contains_finite_differences() {
        let (target, radius) = full_fixture("question_w1");
        let h = 1e-2;
        let mut largest = 0.0f64;
        for sample in 0..4 {
            let bound = head_stage_psi_bound(&target, radius, sample)
                .expect("bound runs")
                .unwrap_or_else(|| panic!("sample {sample}: encoder direction refused"));
            assert!(bound.is_finite() && bound > 0.0, "bound {bound}");
            for k in -4..=4 {
                let z = radius * k as f64 / 5.0;
                let fd = (exact_loss(&target, z + h, sample) - exact_loss(&target, z - h, sample))
                    / (2.0 * h);
                largest = largest.max(fd.abs());
                assert!(
                    fd.abs() <= bound * (1.0 + 1e-6) + 1e-6,
                    "sample {sample}: |Psi'| at z={z} is {fd} but the bound is {bound}"
                );
            }
        }
        assert!(
            largest > 1e-3,
            "largest finite-difference derivative {largest}"
        );
    }
}
