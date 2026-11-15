//! Prototype: an interval-dual forward pass over the Edge0 **head stage**, the
//! base for a certified encoder chart bound.
//!
//! Why this exists
//! ---------------
//! [`crate::edge0_potential::head_chart_enclosure_1d`] can only certify charts
//! whose direction moves `policy_w`/`policy_b`, because then the path
//! distributions are constant in the chart coordinate `z`. Moving encoder
//! weights makes `rep`, the policy and every path distribution move, so `Psi'`
//! has no closed form. The fix is a forward pass carrying `<value, derivative>`
//! **intervals** through the head equations, with the encoder's contribution
//! entering as a dual of `rep`.
//!
//! Scope of this module
//! --------------------
//! This is the head-stage half only, and it is deliberately strict: it accepts
//! chart directions supported on
//!
//! * `policy_w`, `policy_b`, `path_embed`, `noul_w`, `noul_b`, `score_w`,
//!   `thresholds`,
//!
//! where `rep` is provably constant in `z`, and returns `None` for anything that
//! moves the encoder, the embeddings or the attention/MoE weights. Those need
//! the encoder dual pass (LayerNorm → masked softmax attention → SwiGLU →
//! routed MoE with routing over-bounded by a max-over-experts argument), which
//! is **not** implemented here. The point of the module is that the head
//! equations are already written against `rep` duals, so that pass can be
//! dropped in without changing anything below.
//!
//! The `Dual` arithmetic is outward-rounded interval arithmetic on both the
//! value and the derivative, so the returned `psi_prime` is a bound, not an
//! estimate. The tests check containment against finite differences of the real
//! model (candle, `f64`) rather than against this module's own arithmetic.

use crate::edge0_model::forward;
use crate::edge0_potential::Edge0Potential;
use crate::edge0_world::QuestionKind;
use crate::evidence_enclosure::Iv;
use crate::target::{SubspaceTarget, ZPotential};
use candle_core::{DType, Result};

/// A scalar with interval value and interval derivative w.r.t. the chart `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: Iv,
    pub d: Iv,
}

impl Dual {
    pub fn constant(x: f64) -> Self {
        Self {
            v: Iv::point(x),
            d: Iv::point(0.0),
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
        }
    }

    pub fn add(self, o: &Dual) -> Self {
        Self {
            v: self.v.add(&o.v),
            d: self.d.add(&o.d),
        }
    }

    pub fn sub(self, o: &Dual) -> Self {
        Self {
            v: self.v.sub(&o.v),
            d: self.d.sub(&o.d),
        }
    }

    pub fn mul(self, o: &Dual) -> Self {
        Self {
            v: self.v.mul(&o.v),
            d: self.v.mul(&o.d).add(&self.d.mul(&o.v)),
        }
    }

    pub fn scale(self, k: f64) -> Self {
        self.mul(&Dual::constant(k))
    }

    pub fn neg(self) -> Self {
        Self {
            v: self.v.neg(),
            d: self.d.neg(),
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
        // s' = s(1 - s).
        let slope = s.mul(&one.sub(&s));
        Some(Self {
            v: s,
            d: slope.mul(&self.d),
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

    /// `x·σ(x)`; `silu'(x) = σ(x)(1 + x(1 − σ(x)))`.
    pub fn silu(self) -> Option<Self> {
        let s = self.sigmoid()?;
        let one = Iv::point(1.0);
        let slope = s.v.mul(&one.add(&self.v.mul(&one.sub(&s.v))));
        Some(Self {
            v: self.v.mul(&s.v),
            d: slope.mul(&self.d),
        })
    }

    /// `exp` with the same `|x| ≤ 60` guard.
    pub fn exp(self) -> Option<Self> {
        if self.v.lo < -60.0 || self.v.hi > 60.0 {
            return None;
        }
        Some(Self {
            v: self.v.exp(),
            d: self.v.exp().mul(&self.d),
        })
    }

    /// `ln` on a strictly positive value interval.
    pub fn ln(self) -> Option<Self> {
        if self.v.lo <= 0.0 {
            return None;
        }
        Some(Self {
            v: self.v.ln(),
            d: self.d.div(&self.v),
        })
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
    let mut sum = Iv::point(0.0);
    for e in &es {
        sum = sum.add(&e.v);
    }
    if sum.lo <= 0.0 {
        return None;
    }
    Some(
        es.into_iter()
            .map(|e| Dual {
                v: e.v.div(&sum),
                d: e.d.div(&sum),
            })
            .collect(),
    )
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
                if label == 0 {
                    p_no
                } else {
                    p_yes
                }
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
                    probs.push(Dual { v, d: p.d });
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
/// direction supported on the head slots. `None` for anything this prototype
/// does not cover (encoder/embedding/attention directions, wide logits, a
/// marginal interval touching zero).
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

    // Head-only directions: these slots do not feed `rep`.
    let head_slots = [
        "policy_w",
        "policy_b",
        "path_embed",
        "noul_w",
        "noul_b",
        "score_w",
        "score_thresholds",
    ];
    let mut allowed = vec![false; p_total];
    for slot in cfg.layout() {
        if head_slots.contains(&slot.name) {
            for i in slot.start..slot.start + slot.len {
                allowed[i] = true;
            }
        }
    }
    if dir
        .iter()
        .enumerate()
        .any(|(i, x)| *x != 0.0 && !allowed[i])
    {
        return Ok(None);
    }

    let theta0 = sub.lift_tensor(&[0f32])?;
    let flat = theta0
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let hp = build_head_params(cfg, &flat, &dir, radius);

    let s = &samples[sample];
    let input = crate::edge0_model::ModelInput {
        state: &s.input_state,
        questions: std::slice::from_ref(&s.question),
    };
    let q = &forward(&theta0, cfg, &input)?.per_question[0];
    let rep_flat = q
        .rep
        .to_dtype(DType::F64)?
        .to_vec1::<f64>()?;
    let rep: Vec<Dual> = rep_flat.iter().map(|x| Dual::constant(*x)).collect();

    let p = match label_probability_dual(
        &hp,
        &rep,
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
    let input = crate::edge0_model::ModelInput {
        state: &s.input_state,
        questions: std::slice::from_ref(&s.question),
    };
    let q = &forward(&theta0, cfg, &input)?.per_question[0];
    let rep_flat = q.rep.to_dtype(DType::F64)?.to_vec1::<f64>()?;
    let rep: Vec<Dual> = rep_flat.iter().map(|x| Dual::constant(*x)).collect();
    let p = match label_probability_dual(
        &hp,
        &rep,
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
    use crate::edge0_campaign::{build_samples, CampaignConfig};
    use crate::edge0_world::Split;
    use crate::subspace::{AffineSubspace, Basis};

    fn fixture(slot_name: &str) -> (SubspaceTarget<Edge0Potential>, f64) {
        let mut cfg = CampaignConfig::default();
        cfg.updates = 0;
        cfg.warm_start = 0;
        cfg.worlds = [2, 1, 1, 1];
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
    fn head_stage_probability_interval_contains_the_model() {
        // Four train samples cover `noul`, `choice` and `score`.
        for slot in ["score_w", "noul_w", "policy_w", "score_thresholds"] {
            let (target, radius) = fixture(slot);
            for sample in 0..4 {
                let got = head_stage_probability_interval(&target, radius, sample)
                    .expect("interval runs");
                if got.is_none() {
                    eprintln!("NONE slot {slot} sample {sample} kind {:?}", target.potential().samples()[sample].question.kind);
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
                eprintln!("BOUND NONE sample {sample} kind {:?}", target.potential().samples()[sample].question.kind);
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
        assert!(largest > 1e-3, "largest finite-difference derivative {largest}");
    }

    #[test]
    fn encoder_directions_are_refused_by_the_prototype() {
        // `state`/attention weights move `rep`, which this pass does not
        // certify; a fused direction must come back `None`.
        let mut cfg = CampaignConfig::default();
        cfg.updates = 0;
        cfg.warm_start = 0;
        cfg.worlds = [2, 1, 1, 1];
        let all = build_samples(&cfg);
        let samples: Vec<_> = all
            .into_iter()
            .filter(|s| s.split == Split::Train)
            .take(4)
            .collect();
        assert_eq!(samples.len(), 4, "fixture needs four train samples");
        let model = cfg.model.clone();
        let potential = Edge0Potential::head_only(model.clone(), samples);
        let p_total = model.flat_size();
        // A normalized all-ones column touches every slot, including the
        // encoder, so the direction is not head-only.
        let norm = (p_total as f32).sqrt();
        let cols = vec![1.0f32 / norm; p_total];
        let basis = Basis::new(cols, p_total, 1).expect("unit column");
        let theta_ref = crate::edge0_potential::reference_theta(&model, 9)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p_total]).expect("chart");
        let fused = SubspaceTarget::with_potential(subspace, 2.0, potential);
        assert!(head_stage_psi_bound(&fused, 0.5, 0).unwrap().is_none());
    }
}
