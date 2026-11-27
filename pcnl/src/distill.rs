//! R14/P4–P5 — inter-basin posterior-predictive distillation.
//!
//! Two output spaces, two honest compressors:
//!
//! * classification / stochastic policies: average the members in probability
//!   space, `p_ens = M⁻¹ Σ_m p(·|x,θ_m)`, then fit a single student by
//!   `KL(p_ens ‖ p_student)`.  The gradient with respect to the student logits
//!   is `p_student − p_ens`, and the dark-knowledge / Jensen–Shannon gap
//!   `H(p_ens) − mean_m H(p_m) = mean_m KL(p_m ‖ p_ens) ≥ 0` is reported as a
//!   fidelity number (see [`crate::ensemble_diag::js_dark_knowledge_gap`]).
//! * numerical vectors: average the values **and** fit a per-coordinate spread
//!   model, because `+v` and `−v` average to `0` under mean-only distillation.
//!   [`squared_error_ambiguity`](crate::ensemble_diag::squared_error_ambiguity)
//!   is the identity that makes the split explicit.
//!
//! Distillation transfers functions only on the transfer input support; the
//! campaigns must check OOD fidelity separately, and teacher outputs never
//! enter the evidence log.

use crate::ensemble_diag::kl_divergence;
use serde::{Deserialize, Serialize};

/// Numerically stable softmax.
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("distill: softmax needs finite non-empty logits".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        return Err("distill: softmax normalizer is not finite".into());
    }
    Ok(exps.into_iter().map(|e| e / sum).collect())
}

/// Equal-weight ensemble distribution `p_ens = M⁻¹ Σ_m p_m`.
pub fn ensemble_probs(members: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let first = members.first().ok_or("distill: no members")?;
    let k = first.len();
    if k == 0 || members.iter().any(|p| p.len() != k) {
        return Err("distill: member distributions differ in dimension".into());
    }
    let mut mean = vec![0.0; k];
    for p in members {
        for (acc, v) in mean.iter_mut().zip(p) {
            if !v.is_finite() || *v < 0.0 {
                return Err("distill: member distribution has a negative/non-finite entry".into());
            }
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= members.len() as f64;
    }
    Ok(mean)
}

/// `∂/∂z KL(t ‖ softmax(z)) = softmax(z) − t`.
pub fn softmax_kl_grad(student_logits: &[f64], teacher: &[f64]) -> Result<Vec<f64>, String> {
    let p = softmax(student_logits)?;
    if p.len() != teacher.len() {
        return Err("distill: teacher and student dimensions differ".into());
    }
    if teacher.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err("distill: teacher probabilities must be finite and >= 0".into());
    }
    Ok(p.iter().zip(teacher).map(|(p, t)| p - t).collect())
}

/// A linear softmax student `z = W x + b`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxStudent {
    features: usize,
    classes: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl SoftmaxStudent {
    pub fn new(features: usize, classes: usize) -> Result<Self, String> {
        if features == 0 || classes == 0 {
            return Err("distill: student dimensions must be non-zero".into());
        }
        Ok(Self {
            features,
            classes,
            weights: vec![0.0; classes * features],
            bias: vec![0.0; classes],
        })
    }

    pub fn features(&self) -> usize {
        self.features
    }
    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn logits(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.features {
            return Err("distill: input dimension mismatch".into());
        }
        let mut z = self.bias.clone();
        for (c, zc) in z.iter_mut().enumerate() {
            let row = &self.weights[c * self.features..(c + 1) * self.features];
            *zc += row.iter().zip(x).map(|(w, x)| w * x).sum::<f64>();
        }
        Ok(z)
    }

    pub fn probs(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        softmax(&self.logits(x)?)
    }

    /// One KL-SGD step on `(x, teacher_probs)`; returns the pre-step KL.
    pub fn step(&mut self, x: &[f64], teacher: &[f64], lr: f64) -> Result<f64, String> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err("distill: lr must be finite and > 0".into());
        }
        let z = self.logits(x)?;
        let p = softmax(&z)?;
        let kl = kl_divergence(teacher, &p).ok_or("distill: KL is not finite")?;
        for (c, (&pc, &tc)) in p.iter().zip(teacher).enumerate() {
            let g = pc - tc;
            if g == 0.0 {
                continue;
            }
            self.bias[c] -= lr * g;
            for f in 0..self.features {
                self.weights[c * self.features + f] -= lr * g * x[f];
            }
        }
        Ok(kl)
    }
}

/// Fitted student plus the per-epoch mean KL trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistillRecord {
    pub student: SoftmaxStudent,
    pub epochs: usize,
    pub trace: Vec<f64>,
    pub final_kl: f64,
}

/// Fit a softmax student to cached teacher distributions.
pub fn fit_softmax_student(
    features: usize,
    classes: usize,
    data: &[(Vec<f64>, Vec<f64>)],
    epochs: usize,
    lr: f64,
) -> Result<DistillRecord, String> {
    if data.is_empty() || epochs == 0 {
        return Err("distill: empty data or zero epochs".into());
    }
    let mut student = SoftmaxStudent::new(features, classes)?;
    let mut trace = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        let mut sum = 0.0;
        for (x, t) in data {
            sum += student.step(x, t, lr)?;
        }
        trace.push(sum / data.len() as f64);
    }
    let final_kl = *trace.last().unwrap();
    Ok(DistillRecord {
        student,
        epochs,
        trace,
        final_kl,
    })
}

/// Mean and per-coordinate epistemic spread of vector outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorEnsembleStats {
    pub mean: Vec<f32>,
    pub spread: Vec<f32>,
}

pub fn vector_stats(outputs: &[Vec<f32>]) -> Result<VectorEnsembleStats, String> {
    let first = outputs.first().ok_or("distill: no vector outputs")?;
    let q = first.len();
    if q == 0 || outputs.iter().any(|o| o.len() != q) {
        return Err("distill: vector outputs differ in dimension".into());
    }
    if outputs.iter().flatten().any(|v| !v.is_finite()) {
        return Err("distill: non-finite vector output".into());
    }
    let m = outputs.len() as f64;
    let mut mean = vec![0.0f64; q];
    for o in outputs {
        for (acc, v) in mean.iter_mut().zip(o) {
            *acc += f64::from(*v);
        }
    }
    for v in &mut mean {
        *v /= m;
    }
    let mut var = vec![0.0f64; q];
    for o in outputs {
        for (acc, (v, mu)) in var.iter_mut().zip(o.iter().zip(&mean)) {
            let d = f64::from(*v) - mu;
            *acc += d * d;
        }
    }
    for v in &mut var {
        *v /= m;
    }
    Ok(VectorEnsembleStats {
        mean: mean.into_iter().map(|v| v as f32).collect(),
        spread: var.into_iter().map(|v| v.sqrt() as f32).collect(),
    })
}

/// Per-coordinate spread head `s²(x) = exp(a_c + b_c·f_c(x))`, fitted to the
/// teacher's per-coordinate spread.  It gives the distilled vector student an
/// uncertainty that mean-only distillation would collapse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpreadHead {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    /// Floor added to the fitted variance to keep it strictly positive.
    pub floor: f64,
}

impl SpreadHead {
    pub fn fit(
        means: &[Vec<f64>],
        spreads: &[Vec<f64>],
        epochs: usize,
        lr: f64,
    ) -> Result<Self, String> {
        let first = means.first().ok_or("distill: no spread data")?;
        let q = first.len();
        if q == 0
            || means.iter().any(|m| m.len() != q)
            || spreads.len() != means.len()
            || spreads.iter().any(|s| s.len() != q)
        {
            return Err("distill: spread data dimensions disagree".into());
        }
        if epochs == 0 || !lr.is_finite() || lr <= 0.0 {
            return Err("distill: spread fit needs epochs > 0 and lr > 0".into());
        }
        let mut a = vec![0.0; q];
        let mut b = vec![0.0; q];
        let floor = 1e-12;
        for _ in 0..epochs {
            let mut ga = vec![0.0; q];
            let mut gb = vec![0.0; q];
            for (m, s) in means.iter().zip(spreads) {
                for c in 0..q {
                    let target = (s[c] * s[c]).max(floor).ln();
                    let pred = a[c] + b[c] * m[c];
                    let g = pred - target;
                    ga[c] += g;
                    gb[c] += g * m[c];
                }
            }
            let n = means.len() as f64;
            for c in 0..q {
                a[c] -= lr * ga[c] / n;
                b[c] -= lr * gb[c] / n;
            }
        }
        Ok(Self { a, b, floor })
    }

    pub fn variance(&self, mean: &[f64]) -> Result<Vec<f64>, String> {
        if mean.len() != self.a.len() || mean.len() != self.b.len() {
            return Err("distill: spread head dimension mismatch".into());
        }
        Ok(mean
            .iter()
            .enumerate()
            .map(|(c, m)| (self.a[c] + self.b[c] * m).exp() + self.floor)
            .collect())
    }
}

/// Rank-normalized weight distance is not a functional distance; this helper
/// exists so a report can say which one it used.
pub fn permutation_invariant_function_gap(
    member_outputs: &[Vec<f64>],
    reference: &[f64],
) -> Option<f64> {
    if member_outputs.is_empty() {
        return None;
    }
    let mut worst = 0.0f64;
    for o in member_outputs {
        if o.len() != reference.len() {
            return None;
        }
        for (a, b) in o.iter().zip(reference) {
            worst = worst.max((a - b).abs());
        }
    }
    Some(worst)
}
