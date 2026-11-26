//! E2 — genuine RLCD: the four-path policy, the deployed marginal, the
//! Brier U-statistic and its score-function gradient.
//!
//! `EDGE0_RLCD_PLAN.md` §4 defines the object this module implements:
//!
//! ```text
//! r ~ a_phi(. | s,q),              r ∈ {0,1,2,3}
//! u_theta(. | s,q,r) ∈ Δ(K_q)
//! p_phi,theta(y | s,q) = Σ_r a_phi(r|s,q) u_theta(y|s,q,r)
//! B(p,Y) = ‖p − e_Y‖²
//! F(φ,θ) = E_(s,q,Y) B(p_phi,theta(.|s,q), Y).
//! ```
//!
//! Training samples `m = 4` iid paths and uses the unbiased pairwise Brier
//! estimator
//!
//! ```text
//! B_U = 1 − (2/m) Σ_i u_i[Y] + 1/(m(m−1)) Σ_(i≠j) u_i · u_j,
//! E[B_U | s,q,Y] = B(p,Y).
//! ```
//!
//! The stochastic gradient is `B_U · Σ_i ∇ log a_phi(r_i|s,q)` plus the direct
//! derivative of `B_U` through the readout / shared parameters, with `B_U`
//! detached in the score term. A baseline may depend on the input, label and
//! the *other* paths but not on `r_i` given those variables; this module
//! exposes a leave-one-out baseline with that condition and checks it by
//! enumeration rather than assertion.
//!
//! Everything here is `f64` arithmetic on finite distributions. The model-level
//! identity (the same test run through candle autodiff on [`crate::edge0_model`])
//! lives in `tests/edge0_rlcd.rs`; this file provides the exact enumeration the
//! model test is compared against.
//!
//! Sign convention (`EDGE0_RLCD_PLAN.md` §4): a **loss** is lower-is-better,
//! `B_U` is a loss whose expectation is the nonnegative Brier score, and the
//! REINFORCE reward is `−B_U`. `B_U` itself can be negative on a finite draw;
//! it is an unbiased estimator, not a probability and not the deployed score.

use crate::manifest::Fnv;

/// Number of sampled paths in the R9 configuration (`m = 4`).
pub const PATHS: usize = 4;

/// Independent RNG streams. The plan requires data, path policy,
/// initialization and (later) MCMC randomness to be independent; deriving them
/// from one seed by domain-separated hashing keeps a run reproducible without
/// sharing a stream between a policy draw and a later chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Streams {
    pub data: u64,
    pub path: u64,
    pub init: u64,
    pub mcmc: u64,
}

impl Streams {
    pub fn derive(seed: u64) -> Self {
        let mk = |tag: u8| {
            let mut h = Fnv::new();
            h.write(&seed.to_le_bytes());
            h.write(&[tag]);
            h.finish()
        };
        Self {
            data: mk(0),
            path: mk(1),
            init: mk(2),
            mcmc: mk(3),
        }
    }
}

/// `p = Σ_r a_r u_r`.
pub fn marginal(a: &[f64], u: &[Vec<f64>]) -> Vec<f64> {
    assert_eq!(a.len(), u.len(), "one gate per path");
    let k = u.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![0.0; k];
    for (gate, law) in a.iter().zip(u) {
        assert_eq!(law.len(), k, "path laws must share an outcome count");
        for (o, p) in out.iter_mut().zip(law) {
            *o += gate * p;
        }
    }
    out
}

/// `B(p,Y) = ‖p − e_Y‖²`.
pub fn brier(p: &[f64], y: usize) -> f64 {
    p.iter()
        .enumerate()
        .map(|(k, pk)| {
            let d = if k == y { pk - 1.0 } else { *pk };
            d * d
        })
        .sum()
}

/// `‖p − t‖²` — the excess Brier risk at the known law `t`.
pub fn brier_excess(p: &[f64], t: &[f64]) -> f64 {
    p.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// `E_Y∼t B(p,Y) = ‖p−t‖² + (1 − ‖t‖²)`: strictly proper, minimised uniquely
/// at `p = t`. The identity is what makes the Brier excess a proper endpoint.
pub fn expected_brier(p: &[f64], t: &[f64]) -> f64 {
    let norm = 1.0 - t.iter().map(|x| x * x).sum::<f64>();
    brier_excess(p, t) + norm
}

/// The pairwise Brier U-statistic of `m` iid sampled paths.
///
/// `m ≥ 2`; with `m = 1` there is no pairwise term and the estimator is the
/// single-path loss, which is a different objective (the plan forbids
/// transferring mixture claims to it).
pub fn brier_u_statistic(u: &[Vec<f64>], y: usize) -> f64 {
    let m = u.len();
    assert!(m >= 2, "the U-statistic needs m >= 2 paths");
    let k = u[0].len();
    let mut score = 1.0;
    for law in u {
        assert_eq!(law.len(), k);
        score -= 2.0 / m as f64 * law[y];
    }
    let mut pair = 0.0;
    for i in 0..m {
        for j in 0..m {
            if i != j {
                pair += u[i].iter().zip(&u[j]).map(|(a, b)| a * b).sum::<f64>();
            }
        }
    }
    score + pair / (m * (m - 1)) as f64
}

/// All `paths^m` path tuples, in lexicographic order.
pub fn all_tuples(paths: usize, m: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(paths.pow(m as u32));
    let mut tuple = vec![0usize; m];
    loop {
        out.push(tuple.clone());
        let mut i = m;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            tuple[i] += 1;
            if tuple[i] < paths {
                break;
            }
            tuple[i] = 0;
        }
    }
}

/// `Π_i a[tuple[i]]`.
pub fn tuple_probability(a: &[f64], tuple: &[usize]) -> f64 {
    tuple.iter().map(|&r| a[r]).product()
}

/// Exhaustive `E[B_U]`. The E2 gate asserts this equals
/// [`brier`]`(marginal(a,u), y)` for every nonuniform `a` and every readout.
pub fn expected_u_statistic(a: &[f64], u: &[Vec<f64>], y: usize) -> f64 {
    let m = PATHS;
    let mut total = 0.0;
    for tuple in all_tuples(a.len(), m) {
        let p = tuple_probability(a, &tuple);
        if p == 0.0 {
            continue;
        }
        let laws: Vec<Vec<f64>> = tuple.iter().map(|&r| u[r].clone()).collect();
        total += p * brier_u_statistic(&laws, y);
    }
    total
}

/// Exact policy gradient of the deployed marginal objective.
///
/// `∂B/∂logit_j = 2 Σ_r a_r (1[r=j] − a_j) (p − e_y)·u_r`, which is the
/// mixture form of the REINFORCE identity. The `1[r=j] − a_j` factor is the
/// score of the categorical policy, so this is the same quantity the sampled
/// estimator targets; it is computed here in closed form so the estimator can
/// be checked against it rather than against itself.
pub fn exact_policy_gradient(a: &[f64], u: &[Vec<f64>], y: usize) -> Vec<f64> {
    let p = marginal(a, u);
    let resid: Vec<f64> = p
        .iter()
        .enumerate()
        .map(|(k, pk)| pk - if k == y { 1.0 } else { 0.0 })
        .collect();
    let mut grad = vec![0.0; a.len()];
    for j in 0..a.len() {
        let mut acc = 0.0;
        for (r, (a_r, law)) in a.iter().zip(u).enumerate() {
            let score = if r == j { 1.0 } else { 0.0 } - a[j];
            let dot: f64 = resid.iter().zip(law).map(|(e, v)| e * v).sum();
            acc += a_r * score * dot;
        }
        grad[j] = 2.0 * acc;
    }
    grad
}

/// Exact direct derivative of `B(Σ a_r u_r, y)` w.r.t. a scalar shared scale
/// `s` on every readout vector (`u_r = s · base_r`), i.e. `2 (p − e_y)·q` with
/// `q = Σ_r a_r base_r = p/s`. `base` is passed separately so the value is
/// well-defined at `s = 0`.
pub fn exact_shared_gradient(a: &[f64], base: &[Vec<f64>], u: &[Vec<f64>], y: usize) -> f64 {
    let p = marginal(a, u);
    let q = marginal(a, base);
    let resid: Vec<f64> = p
        .iter()
        .enumerate()
        .map(|(k, pk)| pk - if k == y { 1.0 } else { 0.0 })
        .collect();
    2.0 * resid.iter().zip(&q).map(|(e, v)| e * v).sum::<f64>()
}

/// Finite policy/readout fixture whose exhaustive estimator expectation can be
/// differentiated independently of candle.
///
/// The readout is `u_r = s · base[r]`: `s` is the "shared parameter" the plan
/// asks the gate to include. Both `logits` (policy) and `s` (shared readout
/// scale) are differentiable, so the same fixture checks the policy term, the
/// direct readout term and their combination.
#[derive(Clone, Debug)]
pub struct FiniteFixture {
    pub base: Vec<Vec<f64>>,
}

impl FiniteFixture {
    pub fn new(base: Vec<Vec<f64>>) -> Self {
        assert!(base.len() >= 2, "the fixture needs at least two paths");
        let k = base[0].len();
        assert!(base.iter().all(|b| b.len() == k && !b.is_empty()));
        Self { base }
    }

    pub fn path_count(&self) -> usize {
        self.base.len()
    }

    pub fn policy(&self, logits: &[f64]) -> Vec<f64> {
        assert_eq!(logits.len(), self.base.len());
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / total).collect()
    }

    /// `u_r = s · base[r]`.
    pub fn readouts(&self, s: f64) -> Vec<Vec<f64>> {
        self.base
            .iter()
            .map(|b| b.iter().map(|x| x * s).collect())
            .collect()
    }

    pub fn marginal(&self, logits: &[f64], s: f64) -> Vec<f64> {
        marginal(&self.policy(logits), &self.readouts(s))
    }

    /// The exact deployed objective `B(p, y)`.
    pub fn exact_objective(&self, logits: &[f64], s: f64, y: usize) -> f64 {
        brier(&self.marginal(logits, s), y)
    }

    /// Exhaustive expectation of the sampled-path estimator:
    /// `E[B_U · Σ_i ∇log a(r_i)]` for the policy and `E[∂B_U/∂s]` for the
    /// shared scale.
    pub fn estimator_expectation(&self, logits: &[f64], s: f64, y: usize) -> (Vec<f64>, f64) {
        let a = self.policy(logits);
        let u = self.readouts(s);
        let m = PATHS;
        let paths = self.base.len();
        let mut policy = vec![0.0; paths];
        let mut shared = 0.0;
        for tuple in all_tuples(paths, m) {
            let prob = tuple_probability(&a, &tuple);
            if prob == 0.0 {
                continue;
            }
            let laws: Vec<Vec<f64>> = tuple.iter().map(|&r| u[r].clone()).collect();
            let b_u = brier_u_statistic(&laws, y);
            // Score term: detached B_U times the sum of categorical scores.
            for j in 0..paths {
                let mut score_sum = 0.0;
                for &r in &tuple {
                    score_sum += if r == j { 1.0 } else { 0.0 } - a[j];
                }
                policy[j] += prob * b_u * score_sum;
            }
            // Direct term dB_U/ds, with u_r = s·base[r]: the derivative of
            // `−2/m Σ u_i[y]` plus the pairwise product rule.
            let mut direct_terms = 0.0;
            for &ti in &tuple[..m] {
                direct_terms -= 2.0 / m as f64 * self.base[ti][y];
            }
            for i in 0..m {
                for j in 0..m {
                    if i == j {
                        continue;
                    }
                    let dui: &[f64] = &self.base[tuple[i]];
                    let uj: &[f64] = &u[tuple[j]];
                    let ui: &[f64] = &u[tuple[i]];
                    let duj: &[f64] = &self.base[tuple[j]];
                    let t1: f64 = dui.iter().zip(uj).map(|(x, y)| x * y).sum();
                    let t2: f64 = ui.iter().zip(duj).map(|(x, y)| x * y).sum();
                    direct_terms += (t1 + t2) / (m * (m - 1)) as f64;
                }
            }
            shared += prob * direct_terms;
        }
        (policy, shared)
    }

    /// Analytic exact gradients `(∂B/∂logits, ∂B/∂s)`.
    pub fn exact_gradient(&self, logits: &[f64], s: f64, y: usize) -> (Vec<f64>, f64) {
        let a = self.policy(logits);
        let u = self.readouts(s);
        (
            exact_policy_gradient(&a, &u, y),
            exact_shared_gradient(&a, &self.base, &u, y),
        )
    }

    /// Independent central finite differences of [`Self::exact_objective`].
    pub fn finite_difference_gradient(
        &self,
        logits: &[f64],
        s: f64,
        y: usize,
        h: f64,
    ) -> (Vec<f64>, f64) {
        let mut g = vec![0.0; logits.len()];
        for i in 0..logits.len() {
            let mut plus = logits.to_vec();
            let mut minus = logits.to_vec();
            plus[i] += h;
            minus[i] -= h;
            g[i] = (self.exact_objective(&plus, s, y) - self.exact_objective(&minus, s, y))
                / (2.0 * h);
        }
        let gs = (self.exact_objective(logits, s + h, y) - self.exact_objective(logits, s - h, y))
            / (2.0 * h);
        (g, gs)
    }
}

/// A correctness-only reward: `1` if the predicted argmax is `Y`, else `0`.
///
/// The E2 gate uses this to show that correctness is *not* a proper scoring
/// rule: on a noisy binary world (`t = [½, ½]`) both confident corners attain
/// the maximum reward, while the Brier optimum is the unique `p = t`.
pub fn correctness_reward(p: &[f64], y: usize) -> f64 {
    let best = p
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0;
    if best == y { 1.0 } else { 0.0 }
}

/// Leave-one-out baseline for sample `i`: the mean of the *other* paths' Brier
/// terms. It depends on the other independent paths and on `Y`, never on the
/// sample it is subtracted from, which is the plan's condition for a valid
/// baseline. The caller must detach it (it contains no `r_i` term, so its
/// gradient through `a_phi(r_i)` is identically zero).
pub fn leave_one_out_baseline(u: &[Vec<f64>], y: usize, i: usize) -> f64 {
    let others: f64 = u
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, law)| brier(law, y))
        .sum();
    others / (u.len().saturating_sub(1).max(1)) as f64
}
