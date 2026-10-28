//! §9.3 (M1) — the **chain readout**: proper scores of a *posterior
//! predictive* instead of a point readout.
//!
//! The ngpt page's §9.3 asks for one swap: the decision distribution that
//! gets scored should be the chain's posterior predictive
//! `p̄ = E_{θ∼posterior}[u(θ)]`, not the readout of one sampled state
//! `u(θ_i)` (the point readout, which is what a single forward pass or a
//! single rationale gives you). [OpenJev-RLCD's mixture estimator]
//! (`../openjev-rlcd/openjev_rlcd/estimators.py`) already Rao–Blackwellizes
//! over sampled *rationales*; this module does the same over sampled
//! *chain states*, and pins the two facts the swap depends on:
//!
//! 1. **Brier is strictly proper** — its expected loss is minimised
//!    uniquely at the true distribution, so the readout's target is
//!    well-defined (and it is not a "confidence bonus": being sharper
//!    than the truth costs).
//! 2. **The point readout's gap is exactly the posterior covariance**
//!    `tr Cov_posterior(u)`. For a Brier score
//!    `E‖u − e_y‖² = ‖p̄ − e_y‖² + tr Cov`, and the finite-m chain
//!    readout sits `tr Cov / m` above the m → ∞ limit. Both identities
//!    are exact, and both are checked here by enumerating every
//!    m-tuple rather than by trusting the algebra — the same bar
//!    `tests/test_estimators.py` holds its own estimators to.
//!
//! Everything in this module is plain `f64` arithmetic on `[K]`
//! distributions: no candle, no model, no sampling. The measured
//! comparison on the toy transformer's actual chain lives in
//! `examples/readout_compare.rs`, which feeds the states in.
//!
//! Convention: a **loss** is lower-is-better. `Brier` and `Log` are the
//! usual rules; `Spherical` is `1 − p_y/‖p‖`, bounded in `[0, 1]` so a
//! single confidently-wrong state cannot dominate an average the way an
//! unbounded log loss can.

/// The strictly proper scoring rules §9.3 compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// `Σ_k (p_k − 1[k=y])²` — the rule OpenJev-RLCD optimises.
    Brier,
    /// `−log p_y` — the ordinary NLL.
    Log,
    /// `1 − p_y / ‖p‖₂` — bounded, and also strictly proper.
    Spherical,
}

impl Rule {
    pub const ALL: [Rule; 3] = [Rule::Brier, Rule::Log, Rule::Spherical];

    pub fn name(&self) -> &'static str {
        match self {
            Rule::Brier => "Brier",
            Rule::Log => "log",
            Rule::Spherical => "spherical",
        }
    }

    /// The loss of distribution `p` after the outcome `y`.
    pub fn loss(&self, p: &[f64], y: usize) -> f64 {
        debug_assert!(
            y < p.len(),
            "outcome {y} outside a {}-class readout",
            p.len()
        );
        match self {
            Rule::Brier => p
                .iter()
                .enumerate()
                .map(|(k, pk)| {
                    let d = if k == y { pk - 1.0 } else { *pk };
                    d * d
                })
                .sum(),
            Rule::Log => -(p[y] + 1e-12).ln().max(-1e3),
            Rule::Spherical => {
                let n = p.iter().map(|v| v * v).sum::<f64>().sqrt();
                1.0 - p[y] / (n + 1e-12)
            }
        }
    }
}

/// Clamp to a distribution: negatives to 0, non-finite to 0, then
/// renormalise; an all-zero row becomes uniform (the honest "no
/// information" readout rather than a NaN).
pub fn normalize(p: &[f64]) -> Vec<f64> {
    let k = p.len();
    let mut out: Vec<f64> = p
        .iter()
        .map(|v| if v.is_finite() && *v > 0.0 { *v } else { 0.0 })
        .collect();
    let s: f64 = out.iter().sum();
    if s <= 0.0 || !s.is_finite() {
        return vec![1.0 / k as f64; k];
    }
    for v in &mut out {
        *v /= s;
    }
    out
}

/// The chain readout: the arithmetic mean of the per-state distributions
/// (i.e. the Monte-Carlo posterior predictive at the sampled states).
pub fn mean_dist(states: &[Vec<f64>]) -> Vec<f64> {
    assert!(
        !states.is_empty(),
        "the chain readout needs at least one state"
    );
    let k = states[0].len();
    let mut acc = vec![0.0f64; k];
    for s in states {
        assert_eq!(s.len(), k, "every state must read out the same K classes");
        for (a, v) in acc.iter_mut().zip(s.iter()) {
            *a += v;
        }
    }
    let m = states.len() as f64;
    acc.iter().map(|v| v / m).collect()
}

/// The loss of the mean is what §9.3 scores; expose it directly.
pub fn brier(p: &[f64], y: usize) -> f64 {
    Rule::Brier.loss(p, y)
}

/// Leave-one-out gains of each state's contribution to the chain
/// readout (M4's advantage arm): `gain_i = L(p̄_{−i}) − L(p̄)` under
/// Brier, so `gain_i < 0` ⇔ dropping state `i` would have raised the
/// loss, i.e. state `i` helped.
///
/// For a quadratic loss the gain has the closed form
/// `2/(m−1)·(p̄−u_i)·(p̄−e_y) + ‖p̄−u_i‖²/(m−1)²`, which is what makes
/// it cheap enough to use as a reward baseline. `states.len() < 2`
/// yields an empty vector: with one sample there is nothing to leave out.
pub fn loo_brier_gains(states: &[Vec<f64>], y: usize) -> Vec<f64> {
    let m = states.len();
    if m < 2 {
        return Vec::new();
    }
    let p = mean_dist(states);
    let dm = (m - 1) as f64;
    states
        .iter()
        .map(|u| {
            // p̄ − u_i
            let diff: Vec<f64> = p.iter().zip(u.iter()).map(|(a, b)| a - b).collect();
            // p̄ − e_y
            let err: Vec<f64> = p
                .iter()
                .enumerate()
                .map(|(k, v)| if k == y { v - 1.0 } else { *v })
                .collect();
            let cross: f64 = diff.iter().zip(err.iter()).map(|(a, b)| a * b).sum();
            let sq: f64 = diff.iter().map(|v| v * v).sum();
            2.0 / dm * cross + sq / (dm * dm)
        })
        .collect()
}

/// The M4 reward: Brier of the chain readout, negated (higher is better
/// for a policy-gradient arm). At `m = 1` it is the point readout's
/// score, which is the honest way to check the arm is a strict
/// generalisation of what RLCD already optimises.
pub fn chain_reward(states: &[Vec<f64>], y: usize) -> f64 {
    -brier(&mean_dist(states), y)
}

/// A posterior over an *enumerated* latent state space, in the tabular
/// stand-in style of `../openjev-rlcd/tests/test_estimators.py`:
/// `weights[z]` and the per-state readout `dists[z]`. Every quantity
/// below is exact for this object, which is what makes the bias and
/// variance claims falsifiable instead of rhetorical.
#[derive(Clone, Debug)]
pub struct Enumerated {
    pub weights: Vec<f64>,
    pub dists: Vec<Vec<f64>>,
}

impl Enumerated {
    pub fn new(weights: Vec<f64>, dists: Vec<Vec<f64>>) -> Self {
        assert_eq!(weights.len(), dists.len(), "one weight per state");
        assert!(!weights.is_empty(), "empty state space");
        let k = dists[0].len();
        assert!(dists.iter().all(|d| d.len() == k), "ragged readouts");
        let w = normalize(&weights);
        Self {
            weights: w,
            dists: dists.into_iter().map(|d| normalize(&d)).collect(),
        }
    }

    pub fn states(&self) -> usize {
        self.weights.len()
    }

    pub fn classes(&self) -> usize {
        self.dists[0].len()
    }

    /// `p̄ = Σ_z w_z u_z` — the chain readout at `m → ∞`.
    pub fn mean(&self) -> Vec<f64> {
        let k = self.classes();
        let mut acc = vec![0.0f64; k];
        for (w, u) in self.weights.iter().zip(self.dists.iter()) {
            for (a, v) in acc.iter_mut().zip(u.iter()) {
                *a += w * v;
            }
        }
        acc
    }

    /// `tr Cov_posterior(u) = E_z ‖u_z − p̄‖²` — the exact size of the
    /// point readout's Brier gap (`Rule::Brier` only; the other rules
    /// have no such closed form and are enumerated below).
    pub fn covariance_trace(&self) -> f64 {
        let p = self.mean();
        self.weights
            .iter()
            .zip(self.dists.iter())
            .map(|(w, u)| {
                let sq: f64 = u.iter().zip(p.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                w * sq
            })
            .sum()
    }

    /// The **point readout**'s expected loss: `E_z L(u_z, Y)`.
    pub fn expected_point_loss(&self, rule: Rule, y: usize) -> f64 {
        self.weights
            .iter()
            .zip(self.dists.iter())
            .map(|(w, u)| w * rule.loss(u, y))
            .sum()
    }

    /// The **chain readout**'s loss with exact posterior: `L(p̄, Y)`.
    pub fn chain_loss(&self, rule: Rule, y: usize) -> f64 {
        rule.loss(&self.mean(), y)
    }

    /// Exactly `E[L(p̂_m, Y)]` for `m` i.i.d. draws from the posterior,
    /// by enumerating all `states^m` tuples — the finite-m number the
    /// chain readout actually delivers. Explodes combinatorially on
    /// purpose: callers keep `states^m` small (tests use ≤ 4³).
    pub fn exact_expected_chain_loss(&self, rule: Rule, y: usize, m: usize) -> f64 {
        let s = self.states();
        assert!(m >= 1, "m ≥ 1");
        assert!(
            s.pow(m as u32) <= 200_000,
            "exact enumeration would be {s}^{m}"
        );
        let mut total = 0.0f64;
        let mut idx = vec![0usize; m];
        loop {
            let mut prob = 1.0f64;
            for &z in &idx {
                prob *= self.weights[z];
            }
            if prob > 0.0 {
                let mut acc = vec![0.0f64; self.classes()];
                for &z in &idx {
                    for (a, v) in acc.iter_mut().zip(self.dists[z].iter()) {
                        *a += v;
                    }
                }
                for v in &mut acc {
                    *v /= m as f64;
                }
                total += prob * rule.loss(&acc, y);
            }
            // odometer
            let mut pos = 0usize;
            loop {
                if pos == m {
                    return total;
                }
                idx[pos] += 1;
                if idx[pos] < s {
                    break;
                }
                idx[pos] = 0;
                pos += 1;
            }
        }
    }

    /// The Brier closed form of the same quantity:
    /// `‖p̄ − e_y‖² + tr Cov / m`. Exact for independent draws, and the
    /// reason the finite-chain inflation is predictable rather than a
    /// nuisance.
    pub fn predicted_brier_m(&self, y: usize) -> impl Fn(usize) -> f64 + '_ {
        let base = self.chain_loss(Rule::Brier, y);
        let tr = self.covariance_trace();
        move |m| base + tr / m as f64
    }
}

/// Monte-Carlo estimate of `(E[L], Var[L])` for the chain readout at
/// `m` states and for the point readout, from `reps` independent
/// simulations. Seeded and deterministic (the estimate is the evidence,
/// not a tolerance dance): both means are compared against the exact
/// values by the tests.
///
/// Note the two variances are *unconditional*: they include the draw of
/// the outcome `Y`. That term does not shrink with `m` and usually
/// dominates — see [`simulate_fixed_y`] for the reward noise the RL arm
/// actually sees (with the label known, only the state draw fluctuates).
pub fn simulate(
    post: &Enumerated,
    q: &[f64],
    m: usize,
    reps: usize,
    seed: u64,
) -> (f64, f64, f64, f64) {
    use rand::SeedableRng;
    assert_eq!(q.len(), post.classes(), "outcome law must match K");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let cdf_states = cdf(&post.weights);
    let cdf_outcome = cdf(&normalize(q));
    let (mut point_sum, mut point_sq) = (0.0f64, 0.0f64);
    let (mut chain_sum, mut chain_sq) = (0.0f64, 0.0f64);
    for _ in 0..reps {
        let y = draw(&cdf_outcome, &mut rng);
        // the point readout: one state, scored as if it were the answer
        let z0 = draw(&cdf_states, &mut rng);
        let lp = Rule::Brier.loss(&post.dists[z0], y);
        point_sum += lp;
        point_sq += lp * lp;
        // the chain readout: m states averaged
        let mut acc = vec![0.0f64; post.classes()];
        for _ in 0..m {
            let z = draw(&cdf_states, &mut rng);
            for (a, v) in acc.iter_mut().zip(post.dists[z].iter()) {
                *a += v;
            }
        }
        for v in &mut acc {
            *v /= m as f64;
        }
        let lc = Rule::Brier.loss(&acc, y);
        chain_sum += lc;
        chain_sq += lc * lc;
    }
    let n = reps as f64;
    let pm = point_sum / n;
    let cm = chain_sum / n;
    (
        pm,
        (point_sq / n - pm * pm).max(0.0),
        cm,
        (chain_sq / n - cm * cm).max(0.0),
    )
}

/// The conditional counterpart: `y` is fixed, so only the posterior
/// draw is random. `(mean, var)` of the chain readout's Brier at `m`
/// states, plus the point readout's, from common random numbers.
/// This is the variance a policy-gradient arm experiences on a labelled
/// example, and it is the one §9.3 claims the chain readout shrinks.
pub fn simulate_fixed_y(
    post: &Enumerated,
    y: usize,
    m: usize,
    reps: usize,
    seed: u64,
) -> (f64, f64, f64, f64) {
    use rand::SeedableRng;
    assert!(y < post.classes(), "outcome {y} outside K");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let cdf_states = cdf(&post.weights);
    let (mut point_sum, mut point_sq) = (0.0f64, 0.0f64);
    let (mut chain_sum, mut chain_sq) = (0.0f64, 0.0f64);
    for _ in 0..reps {
        let z0 = draw(&cdf_states, &mut rng);
        let lp = Rule::Brier.loss(&post.dists[z0], y);
        point_sum += lp;
        point_sq += lp * lp;
        let mut acc = vec![0.0f64; post.classes()];
        for _ in 0..m {
            let z = draw(&cdf_states, &mut rng);
            for (a, v) in acc.iter_mut().zip(post.dists[z].iter()) {
                *a += v;
            }
        }
        for v in &mut acc {
            *v /= m as f64;
        }
        let lc = Rule::Brier.loss(&acc, y);
        chain_sum += lc;
        chain_sq += lc * lc;
    }
    let n = reps as f64;
    let pm = point_sum / n;
    let cm = chain_sum / n;
    (
        pm,
        (point_sq / n - pm * pm).max(0.0),
        cm,
        (chain_sq / n - cm * cm).max(0.0),
    )
}

fn cdf(p: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    p.iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect()
}

fn draw(cdf: &[f64], rng: &mut impl rand::Rng) -> usize {
    let u: f64 = rng.random::<f64>();
    match cdf.iter().position(|c| u <= *c) {
        Some(i) => i,
        None => cdf.len() - 1, // rounding left u above the last cdf entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tabular stand-in: three latent states, three classes, and a
    /// posterior that is deliberately *spread out* (identical states
    /// would make every identity trivially zero).
    fn post3() -> Enumerated {
        Enumerated::new(
            vec![0.5, 0.3, 0.2],
            vec![
                vec![0.70, 0.20, 0.10],
                vec![0.25, 0.55, 0.20],
                vec![0.10, 0.30, 0.60],
            ],
        )
    }

    /// A coarse probability-simplex grid for the properness checks.
    fn simplices(steps: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        for i in 0..=steps {
            for j in 0..=steps - i {
                let a = i as f64 / steps as f64;
                let b = j as f64 / steps as f64;
                out.push(vec![a, b, 1.0 - a - b]);
            }
        }
        out
    }

    #[test]
    fn brier_is_strictly_proper() {
        let q = normalize(&[0.6, 0.3, 0.1]);
        for y in 0..3 {
            // E_{Y~q}[Brier(p, Y)] is minimised uniquely at p = q.
            let expected =
                |p: &[f64]| -> f64 { (0..3).map(|k| q[k] * Rule::Brier.loss(p, k)).sum() };
            let at_q = expected(&q);
            for p in simplices(20) {
                if (p[0] - q[0]).abs() < 1e-9
                    && (p[1] - q[1]).abs() < 1e-9
                    && (p[2] - q[2]).abs() < 1e-9
                {
                    continue;
                }
                assert!(
                    expected(&p) > at_q + 1e-12,
                    "y={y}: p={p:?} not worse than q ({} vs {at_q})",
                    expected(&p)
                );
            }
        }
    }

    #[test]
    fn log_and_spherical_are_strictly_proper() {
        let q = normalize(&[0.5, 0.3, 0.2]);
        for rule in [Rule::Log, Rule::Spherical] {
            let expected = |p: &[f64]| -> f64 { (0..3).map(|k| q[k] * rule.loss(p, k)).sum() };
            let at_q = expected(&q);
            for p in simplices(20) {
                let same = p.iter().zip(q.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
                if same {
                    continue;
                }
                assert!(
                    expected(&p) > at_q + 1e-12,
                    "{}: p={p:?} not worse than q",
                    rule.name()
                );
            }
        }
    }

    #[test]
    fn point_gap_is_exactly_the_posterior_covariance() {
        // E_z ‖u_z − e_y‖² − ‖p̄ − e_y‖² = tr Cov, for every y — the
        // identity §9.3's gain claim rests on.
        let post = post3();
        let tr = post.covariance_trace();
        assert!(tr > 0.01, "test fixture must have spread (tr={tr})");
        for y in 0..3 {
            let point = post.expected_point_loss(Rule::Brier, y);
            let chain = post.chain_loss(Rule::Brier, y);
            assert!(
                (point - chain - tr).abs() < 1e-12,
                "y={y}: point {point} − chain {chain} ≠ tr {tr}"
            );
        }
    }

    #[test]
    fn finite_m_inflation_matches_enumeration_and_the_closed_form() {
        let post = post3();
        for y in 0..3 {
            let pred = post.predicted_brier_m(y);
            for m in 1..=4 {
                let enumerated = post.exact_expected_chain_loss(Rule::Brier, y, m);
                assert!(
                    (enumerated - pred(m)).abs() < 1e-12,
                    "y={y} m={m}: enumerated {enumerated} ≠ closed form {}",
                    pred(m)
                );
            }
            // m = 1 reproduces the point readout exactly: the chain
            // readout is a strict generalisation of it, not a rival.
            let point = post.expected_point_loss(Rule::Brier, y);
            assert!((pred(1) - point).abs() < 1e-12);
        }
    }

    #[test]
    fn the_chain_readout_is_better_in_expectation() {
        // The Brier gain of the chain readout over the point readout is
        // tr Cov · (1 − 1/m): exact, positive, and growing in m.
        let post = post3();
        let tr = post.covariance_trace();
        for y in 0..3 {
            let point = post.expected_point_loss(Rule::Brier, y);
            let mut previous = f64::INFINITY;
            for m in 1..=6 {
                let chain = post.exact_expected_chain_loss(Rule::Brier, y, m);
                let expected_gain = tr * (1.0 - 1.0 / m as f64);
                assert!(
                    (point - chain - expected_gain).abs() < 1e-12,
                    "y={y} m={m}: gain {} ≠ {expected_gain}",
                    point - chain
                );
                assert!(chain < previous, "m={m} did not improve");
                previous = chain;
            }
        }
    }

    #[test]
    fn loo_gains_match_the_direct_difference() {
        let post = post3();
        let states = post.dists.clone();
        for y in 0..3 {
            let gains = loo_brier_gains(&states, y);
            assert_eq!(gains.len(), 3);
            for (i, gain) in gains.iter().enumerate() {
                let kept: Vec<Vec<f64>> = states
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, u)| u.clone())
                    .collect();
                let direct = brier(&mean_dist(&kept), y) - brier(&mean_dist(&states), y);
                assert!(
                    (gain - direct).abs() < 1e-12,
                    "y={y} i={i}: closed form {gain} ≠ direct {direct}"
                );
            }
        }
    }

    #[test]
    fn loo_gains_sum_to_the_dispersion_identity() {
        // Σ_i gain_i = Σ_i ‖p̄ − u_i‖² / (m−1)²  — the telescoping that
        // makes the reward arm's baseline sum to a measurable quantity.
        let states = post3().dists;
        let m = states.len();
        let p = mean_dist(&states);
        for y in 0..3 {
            let lhs: f64 = loo_brier_gains(&states, y).iter().sum();
            let rhs: f64 = states
                .iter()
                .map(|u| {
                    u.iter()
                        .zip(p.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / ((m - 1) * (m - 1)) as f64;
            assert!((lhs - rhs).abs() < 1e-12, "y={y}: {lhs} ≠ {rhs}");
        }
        assert!(
            loo_brier_gains(&states[..1], 0).is_empty(),
            "m=1 has no LOO"
        );
    }

    #[test]
    fn measured_means_match_the_exact_values() {
        // Unconditional (outcome drawn too): both estimators' means must
        // land on the enumerated expectations, or the MC rig is lying.
        let post = post3();
        let q = [0.60, 0.30, 0.10];
        let reps = 20_000;
        let (point_mean, _, chain1, _) = simulate(&post, &q, 1, reps, 11);
        let exact_point: f64 = (0..3)
            .map(|y| q[y] * post.expected_point_loss(Rule::Brier, y))
            .sum();
        assert!(
            (point_mean - exact_point).abs() < 0.01,
            "point mean {point_mean} vs exact {exact_point}"
        );
        assert!((chain1 - exact_point).abs() < 0.02, "m=1 chain {chain1}");
        for m in [2usize, 4, 8, 16] {
            let (_, _, mean_m, _) = simulate(&post, &q, m, reps, 11);
            let exact_m: f64 = (0..3).map(|y| q[y] * post.predicted_brier_m(y)(m)).sum();
            assert!(
                (mean_m - exact_m).abs() < 0.01,
                "m={m}: simulated {mean_m} vs exact {exact_m}"
            );
        }
    }

    #[test]
    fn conditional_reward_variance_falls_with_m() {
        // With the label known (what an RL arm sees on one example), the
        // only randomness left is the posterior draw, and that is the
        // part the chain readout shrinks. The *unconditional* variance
        // is dominated by the outcome draw and need not fall at all —
        // measured, not assumed: this test exists because the first
        // version of it asserted the wrong invariant and failed at m=8.
        let post = post3();
        let reps = 40_000;
        let tr = post.covariance_trace();
        for y in 0..3 {
            let (point_mean, point_var, _, _) = simulate_fixed_y(&post, y, 1, reps, 13);
            assert!(
                (point_mean - post.expected_point_loss(Rule::Brier, y)).abs() < 0.01,
                "y={y}: point mean {point_mean}"
            );
            assert!(point_var > 0.0, "y={y}: degenerate fixture");
            let mut last = point_var;
            for m in [2usize, 4, 8, 16] {
                let (_, _, mean_m, var_m) = simulate_fixed_y(&post, y, m, reps, 13);
                assert!(
                    (mean_m - post.predicted_brier_m(y)(m)).abs() < 0.01,
                    "y={y} m={m}: mean {mean_m} vs exact {}",
                    post.predicted_brier_m(y)(m)
                );
                assert!(
                    var_m < last,
                    "y={y} m={m}: conditional variance {var_m} ≥ {last}"
                );
                last = var_m;
            }
            // and the m=1 conditional variance is the spread of the
            // per-state Brier about its posterior mean (tr Cov is not
            // that number — Brier spreads quadratically, so the check is
            // the enumerated one, computed here directly).
            let mean_l: f64 = post.expected_point_loss(Rule::Brier, y);
            let var_l: f64 = post
                .weights
                .iter()
                .zip(post.dists.iter())
                .map(|(w, u)| w * (Rule::Brier.loss(u, y) - mean_l).powi(2))
                .sum();
            let (_, point_var_again, _, _) = simulate_fixed_y(&post, y, 1, reps, 13);
            // (the point readout's variance does not depend on m)
            assert!(
                (point_var_again - var_l).abs() < 0.02 + 0.1 * var_l,
                "y={y}: measured {point_var_again} vs enumerated {var_l} (tr Cov {tr})"
            );
        }
    }

    #[test]
    fn chain_reward_reduces_to_the_point_readout_at_m1() {
        let post = post3();
        for y in 0..3 {
            let u = &post.dists[y % 3];
            assert!((chain_reward(std::slice::from_ref(u), y) + brier(u, y)).abs() < 1e-12);
        }
    }

    #[test]
    fn normalize_survives_degenerate_rows() {
        assert_eq!(normalize(&[f64::NAN, 1.0, 1.0]), vec![0.0, 0.5, 0.5]);
        assert_eq!(normalize(&[-1.0, 0.0, 0.0]), vec![1.0 / 3.0; 3]);
        let p = normalize(&[2.0, 2.0]);
        assert!((p[0] - 0.5).abs() < 1e-15);
    }
}
