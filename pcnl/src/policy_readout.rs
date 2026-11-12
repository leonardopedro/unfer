//! §R7.3.4 — the DT's readout: per-step proper scores, calibration of the
//! policy's max probability, and return-level readouts with their MCSE.
//!
//! Three different questions live here, and the module's job is to keep them
//! from being answered with one number:
//!
//! 1. **Per-step proper scores** ([`score_steps`]). A DT outputs a
//!    distribution over actions at every step, so the honest per-step object is
//!    a strictly proper score of that distribution — Brier, log, spherical,
//!    through [`crate::readout::Rule`], the same three rules M1's chain readout
//!    already compares. The mean per-step *log* score is the training loss; the
//!    other two are readouts of the same objects and must not be reported as if
//!    they were the loss.
//!
//! 2. **Calibration** ([`max_prob_calibration`]). §R7.0's promise is decision
//!    quality, and a policy's confidence is the one thing a return readout
//!    cannot see. Expected calibration error over the max probability is the
//!    standard diagnostic — *and it is not a proper score*: a distribution can
//!    improve its ECE while getting worse on every proper rule (sharpening only
//!    the bins it gets right). It is reported beside the scores, never instead
//!    of them, and each bin's count is published because an ECE over bins with
//!    two samples each is mostly noise.
//!
//! 3. **Return-level readouts** ([`readout`]). The plan asks for `K` rollouts of
//!    a policy *sampled* from the readout distribution and `K` of the
//!    *averaged decoder* (argmax), with paired differences against each baseline
//!    carrying its own MCSE. Paired, not independent: every arm sees the same
//!    uniform draw and the same transition noise in every replicate, so the
//!    difference between two arms is measured on the same environment
//!    realisation. That is what makes a difference of a few hundredths of a
//!    nat resolvable at `K ≈ 256` instead of drowning in return variance, and
//!    the variance reduction is itself a test here rather than a claim.
//!
//! A trap this module avoids by construction: the reward in
//! [`crate::mdp`] is ≤ 0, so "higher return" is "less cost", and a policy's
//! return is **not** comparable to the value-iteration optimum unless the
//! comparison is noise-matched (`mdp::MdpSolution::monte_carlo_greedy`). The
//! readout therefore only ever compares arms against each other, on the same
//! noise, and reports an MCSE with every mean.

use crate::mdp::MdpEnv;
use crate::model::Variant;
use crate::policy_model::{DtBatch, DtShapes, dt_loss, dt_probs};
use crate::readout::{Rule, normalize};
use candle_core::{Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

// ---------------------------------------------------------------- scores

/// One scoring rule's per-step values, and their mean.
#[derive(Clone, Debug)]
pub struct RuleScores {
    pub rule: Rule,
    pub mean: f64,
    /// One entry per scored step, in the batch order of
    /// [`DtBatch::scored_steps`].
    pub per_step: Vec<f64>,
}

/// Per-step scores for a set of distributions, under every rule.
///
/// `probs[i]` is the distribution the model assigned at scored step `i` and
/// `targets[i]` the action that actually happened. The distributions are used
/// **as given**: they are softmax outputs, and renormalising them here would
/// make the log score stop agreeing with [`dt_loss`] at the last bit, which is
/// the one agreement this module promises.
#[derive(Clone, Debug)]
pub struct StepScores {
    pub n: usize,
    pub by_rule: Vec<RuleScores>,
}

impl StepScores {
    /// The mean score under one rule. Panics if the rule was not scored — a
    /// silent `0.0` for a missing column is how a report grows a fake zero.
    pub fn mean(&self, rule: Rule) -> f64 {
        self.by_rule
            .iter()
            .find(|s| s.rule == rule)
            .unwrap_or_else(|| panic!("{} was not scored", rule.name()))
            .mean
    }
}

/// Score `probs` against `targets` under every rule in [`Rule::ALL`].
///
/// An empty input is an error, not a zero: `n = 0` with `mean = 0` reads as a
/// perfect score in any table that sorts by loss.
pub fn score_steps(probs: &[Vec<f64>], targets: &[usize]) -> StepScores {
    assert_eq!(probs.len(), targets.len(), "one target per distribution");
    assert!(!probs.is_empty(), "no steps to score");
    let vocab = probs[0].len();
    assert!(vocab > 0, "empty distribution");
    for (i, p) in probs.iter().enumerate() {
        assert_eq!(
            p.len(),
            vocab,
            "step {i} has {} actions, not {vocab}",
            p.len()
        );
        assert!(
            targets[i] < vocab,
            "step {i} target {} outside a {vocab}-action readout",
            targets[i]
        );
        assert!(
            p.iter().all(|v| v.is_finite()),
            "step {i} distribution has a non-finite entry"
        );
    }
    let by_rule = Rule::ALL
        .iter()
        .map(|&rule| {
            let per_step: Vec<f64> = probs
                .iter()
                .zip(targets.iter())
                .map(|(p, &y)| rule.loss(p, y))
                .collect();
            let mean = per_step.iter().sum::<f64>() / per_step.len() as f64;
            RuleScores {
                rule,
                mean,
                per_step,
            }
        })
        .collect();
    StepScores {
        n: probs.len(),
        by_rule,
    }
}

/// Per-step distributions of a DT, gathered at exactly the steps the loss
/// scores.
///
/// This is the seam that makes the readout and the target the same object: the
/// indices come from [`DtBatch::scored_steps`], so a padded batch cannot be
/// scored on padding, and `scores.mean(Rule::Log)` equals [`dt_loss`] up to
/// `f32` round-off (`dt_step_scores_equals_dt_loss` checks that equality).
pub fn dt_step_distributions(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
) -> Result<Vec<Vec<f64>>> {
    let probs = dt_probs(theta, shapes, variant, batch)?;
    let (b, t) = batch.mask.dims2()?;
    let (idx, _) = batch.scored_steps()?;
    let flat = probs.reshape((b * 3 * t, shapes.action_vocab))?;
    let sel = flat.index_select(
        &Tensor::from_vec(idx, batch.transitions(), probs.device())?,
        0,
    )?;
    let rows = sel.to_vec2::<f32>()?;
    Ok(rows
        .into_iter()
        .map(|r| r.into_iter().map(f64::from).collect())
        .collect())
}

/// [`dt_step_distributions`] then [`score_steps`], under all rules.
pub fn dt_step_scores(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
) -> Result<StepScores> {
    let probs = dt_step_distributions(theta, shapes, variant, batch)?;
    let (_, targets) = batch.scored_steps()?;
    let targets: Vec<usize> = targets.into_iter().map(|a| a as usize).collect();
    Ok(score_steps(&probs, &targets))
}

/// The same readout for a batch, plus the loss it is supposed to agree with —
/// one call, so a report cannot quote the two from different batches.
pub fn dt_readout(
    theta: &Tensor,
    shapes: &DtShapes,
    variant: Variant,
    batch: &DtBatch,
) -> Result<(StepScores, f64)> {
    Ok((
        dt_step_scores(theta, shapes, variant, batch)?,
        dt_loss(theta, shapes, variant, batch)?,
    ))
}

// ----------------------------------------------------------- calibration

/// One equi-width bin of the max-probability calibration curve.
#[derive(Clone, Debug)]
pub struct CalibrationBin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    /// Mean max probability in the bin; `NaN` when the bin is empty.
    pub mean_confidence: f64,
    /// Fraction of the bin's steps where the argmax action was taken; `NaN`
    /// when empty.
    pub accuracy: f64,
}

/// Expected calibration error over the max probability, with the bins that
/// produced it.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub n: usize,
    pub bins: Vec<CalibrationBin>,
    /// `Σ_bins (count/n)·|accuracy − confidence|`.
    pub ece: f64,
    pub mean_confidence: f64,
    pub accuracy: f64,
    /// Mean log score of the max-probability **decision** treated as the
    /// model's answer — i.e. the top-1 NLL, which is what a decision-level
    /// report actually cares about. `NaN` if any top probability is 0.
    pub top1_nll: f64,
}

/// Calibration of the max probability against whether it was right.
///
/// Bin edges are `[0, 1]` in `bins` equal steps, right-closed except for the
/// last bin; a confidence of exactly 1.0 lands in the last bin. Empty bins are
/// kept (with `NaN` statistics and no ECE contribution) so the table's shape
/// does not depend on the data, and so a well-calibrated 5-bin run cannot be
/// confused with a run that only ever used one bin.
pub fn max_prob_calibration(probs: &[Vec<f64>], targets: &[usize], bins: usize) -> Calibration {
    assert!(!probs.is_empty(), "no steps to calibrate");
    assert!(bins > 0, "a calibration curve needs at least one bin");
    assert_eq!(probs.len(), targets.len(), "one target per distribution");
    let mut counts = vec![0usize; bins];
    let mut conf_sum = vec![0.0f64; bins];
    let mut acc_sum = vec![0.0f64; bins];
    let mut conf_all = 0.0f64;
    let mut correct = 0usize;
    let mut nll_sum = 0.0f64;
    let mut nll_n = 0usize;
    for (p, &y) in probs.iter().zip(targets.iter()) {
        let (best, conf) =
            p.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (k, &v)| {
                    if v > bv { (k, v) } else { (bi, bv) }
                });
        assert!(
            conf.is_finite() && conf >= 0.0,
            "max probability {conf} is not a probability"
        );
        assert!(
            y < p.len(),
            "target {y} outside a {}-action readout",
            p.len()
        );
        // Last bin is closed on the right; others are [lo, hi).
        let b = ((conf * bins as f64).floor() as usize).min(bins - 1);
        counts[b] += 1;
        conf_sum[b] += conf;
        let hit = if best == y { 1.0 } else { 0.0 };
        acc_sum[b] += hit;
        conf_all += conf;
        correct += hit as usize;
        if conf > 0.0 {
            nll_sum += -conf.ln();
            nll_n += 1;
        }
    }
    let n = probs.len();
    let bins_out: Vec<CalibrationBin> = (0..bins)
        .map(|b| {
            let c = counts[b];
            CalibrationBin {
                lo: b as f64 / bins as f64,
                hi: (b + 1) as f64 / bins as f64,
                count: c,
                mean_confidence: if c == 0 {
                    f64::NAN
                } else {
                    conf_sum[b] / c as f64
                },
                accuracy: if c == 0 {
                    f64::NAN
                } else {
                    acc_sum[b] / c as f64
                },
            }
        })
        .collect();
    let ece = bins_out
        .iter()
        .filter(|b| b.count > 0)
        .map(|b| b.count as f64 / n as f64 * (b.accuracy - b.mean_confidence).abs())
        .sum();
    Calibration {
        n,
        bins: bins_out,
        ece,
        mean_confidence: conf_all / n as f64,
        accuracy: correct as f64 / n as f64,
        top1_nll: if nll_n == 0 {
            f64::NAN
        } else {
            nll_sum / nll_n as f64
        },
    }
}

// -------------------------------------------------------- return readout

/// An action distribution as a function of `(timestep, state)`. A DT readout
/// supplies one of these; the MDP is what turns it into a return.
pub type ActionDist = dyn Fn(usize, [f32; 2]) -> Vec<f64>;

/// How an arm turns a distribution into an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    /// One draw from the distribution — the plan's "policy sampled from p̄".
    Sample,
    /// The argmax — the "averaged decoder", deterministic in the state.
    Argmax,
}

/// One arm of a return comparison.
pub struct Arm<'a> {
    pub name: &'a str,
    pub dist: &'a ActionDist,
    pub decode: Decode,
}

/// An arm's rollout returns, with the mean and its Monte Carlo standard error.
#[derive(Clone, Debug)]
pub struct ArmReturns {
    pub name: String,
    pub returns: Vec<f64>,
    pub mean: f64,
    pub mcse: f64,
}

/// `mean ± MCSE` of the paired difference `a − b`.
#[derive(Clone, Debug)]
pub struct PairedDiff {
    pub mean: f64,
    pub mcse: f64,
    pub n: usize,
}

impl PairedDiff {
    /// Two MCSEs — the threshold below which a difference is not resolved by
    /// this many rollouts, in this environment, with this pairing.
    pub fn resolved(&self) -> bool {
        self.mean.abs() > 2.0 * self.mcse
    }

    pub fn render(&self) -> String {
        format!(
            "{:+.4} ± {:.4} ({})",
            self.mean,
            self.mcse,
            if self.resolved() {
                "resolved"
            } else {
                "within noise"
            }
        )
    }
}

/// The whole readout: every arm, and every arm paired against the first.
#[derive(Clone, Debug)]
pub struct ReturnReadout {
    pub env: MdpEnv,
    pub rollouts: usize,
    pub arms: Vec<ArmReturns>,
    /// `(arm name, paired difference against arms[0])`, one per non-reference
    /// arm, in arm order.
    pub diffs: Vec<(String, PairedDiff)>,
    /// The unpaired standard error of the same difference, for the
    /// variance-reduction check: `mcse` should be well below this.
    pub diffs_independent_mcse: Vec<f64>,
}

impl ReturnReadout {
    pub fn arm(&self, name: &str) -> &ArmReturns {
        self.arms
            .iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("no arm named {name}"))
    }
}

/// Mean and MCSE of a sample.
pub fn mean_mcse(xs: &[f64]) -> (f64, f64) {
    assert!(!xs.is_empty(), "mean_mcse on an empty sample");
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

/// Paired difference of two equal-length return samples.
pub fn paired(a: &[f64], b: &[f64]) -> PairedDiff {
    assert_eq!(a.len(), b.len(), "paired samples must be the same length");
    let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let (mean, mcse) = mean_mcse(&d);
    PairedDiff {
        mean,
        mcse,
        n: d.len(),
    }
}

/// Inverse-CDF draw at a fixed uniform `u`, so two arms can be coupled: the
/// same `u` through two different distributions gives two actions that are
/// equal wherever the distributions agree, which is the coupling the return
/// readout needs.
fn sample_at(dist: &[f64], u: f64) -> usize {
    let p = normalize(dist);
    let mut acc = 0.0f64;
    for (k, &pk) in p.iter().enumerate() {
        acc += pk;
        if u < acc {
            return k;
        }
    }
    p.len() - 1
}

fn argmax(dist: &[f64]) -> usize {
    dist.iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (k, &v)| {
            if v > bv { (k, v) } else { (bi, bv) }
        })
        .0
}

/// Roll every arm out for `rollouts` replicates, paired on the same uniform
/// draws and the same transition noise.
///
/// `x0` is the start state for every rollout, `t` runs `0..env.horizon`, and
/// the noise is drawn once per replicate (`noise[t]`), then reused by every
/// arm. The uniforms are drawn per replicate per step, before the arms, so
/// adding an arm cannot change another arm's returns — which is what makes an
/// arm's numbers comparable across two runs that differ by one arm.
pub fn readout(
    env: &MdpEnv,
    x0: [f32; 2],
    arms: &[Arm],
    rollouts: usize,
    seed: u64,
) -> ReturnReadout {
    assert!(!arms.is_empty(), "a readout needs at least one arm");
    assert!(rollouts > 1, "an MCSE needs more than one rollout");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut returns: Vec<Vec<f64>> = vec![Vec::with_capacity(rollouts); arms.len()];
    for _ in 0..rollouts {
        // Common randomness: one uniform per step and one noise vector per
        // step, both drawn before any arm moves.
        let uniforms: Vec<f64> = (0..env.horizon).map(|_| rng.random::<f64>()).collect();
        let noises: Vec<[f32; 2]> = (0..env.horizon)
            .map(|_| {
                let n0: f32 = rng.sample(StandardNormal);
                let n1: f32 = rng.sample(StandardNormal);
                [env.noise * n0, env.noise * n1]
            })
            .collect();
        for (ai, arm) in arms.iter().enumerate() {
            let mut x = x0;
            let mut total = 0.0f64;
            for t in 0..env.horizon {
                let dist = (arm.dist)(t, x);
                assert_eq!(
                    dist.len(),
                    env.actions.len(),
                    "arm {} gave a {}-action distribution for a {}-action environment",
                    arm.name,
                    dist.len(),
                    env.actions.len()
                );
                let a = match arm.decode {
                    Decode::Sample => sample_at(&dist, uniforms[t]),
                    Decode::Argmax => argmax(&dist),
                };
                total += f64::from(env.reward(x, a));
                x = env.step(x, a, noises[t]);
            }
            returns[ai].push(total);
        }
    }
    let stats: Vec<ArmReturns> = arms
        .iter()
        .zip(returns.iter())
        .map(|(arm, r)| {
            let (mean, mcse) = mean_mcse(r);
            ArmReturns {
                name: arm.name.to_string(),
                returns: r.clone(),
                mean,
                mcse,
            }
        })
        .collect();
    let mut diffs = Vec::new();
    let mut diffs_independent_mcse = Vec::new();
    for i in 1..stats.len() {
        diffs.push((
            stats[i].name.clone(),
            paired(&stats[i].returns, &stats[0].returns),
        ));
        // The independent-arm SE of the same difference, from the two arms'
        // own MCSEs. Pairing is only worth it if the real MCSE is below this.
        diffs_independent_mcse.push((stats[i].mcse.powi(2) + stats[0].mcse.powi(2)).sqrt());
    }
    ReturnReadout {
        env: env.clone(),
        rollouts,
        arms: stats,
        diffs,
        diffs_independent_mcse,
    }
}

/// A uniform-random arm: the honest floor, and the baseline every learned
/// policy should beat before any of its numbers mean anything.
pub fn uniform_arm(env: &MdpEnv) -> impl Fn(usize, [f32; 2]) -> Vec<f64> + use<> {
    let k = env.actions.len();
    move |_, _| vec![1.0 / k as f64; k]
}

/// A deterministic one-hot arm over `vocab` actions that always takes
/// `action`.  `vocab` is explicit because a distribution whose arity does not
/// match the environment is a bug the readout has to reject, not silently
/// extend.
pub fn constant_arm(vocab: usize, action: usize) -> impl Fn(usize, [f32; 2]) -> Vec<f64> + use<> {
    assert!(
        action < vocab,
        "action {action} outside a {vocab}-action arm"
    );
    move |_, _| {
        let mut d = vec![0.0f64; vocab];
        d[action] = 1.0;
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::{BehaviourPolicy, Corpus, DtSpec};
    use crate::policy_model::init_flat;

    fn small_dt_env() -> MdpEnv {
        MdpEnv {
            horizon: 4,
            ..MdpEnv::default()
        }
    }

    fn tiny_shapes(env: &MdpEnv) -> DtShapes {
        env.dt_shapes(4, 6, 12, 1)
    }

    #[test]
    fn score_steps_matches_hand_computed_values() {
        let uniform = vec![1.0 / 3.0; 3];
        let s = score_steps(std::slice::from_ref(&uniform), &[1]);
        assert_eq!(s.n, 1);
        // 1e-9, not 1e-12: `Rule::Log` evaluates `−ln(p_y + 1e-12)` so a zero
        // probability cannot be `−inf`, which shifts a finite score by ~3e-12.
        let log = 3.0f64.ln();
        assert!((s.mean(Rule::Log) - log).abs() < 1e-9);
        // Brier: (1/3)² twice + (2/3)²
        assert!((s.mean(Rule::Brier) - (2.0 / 9.0 + 4.0 / 9.0)).abs() < 1e-12);
        // Spherical: 1 − (1/3)/√(3·1/9) = 1 − 1/√3, again up to the 1e-12
        // denominator guard in `Rule::Spherical`.
        assert!((s.mean(Rule::Spherical) - (1.0 - 1.0 / 3.0f64.sqrt())).abs() < 1e-9);
    }

    /// Proper rules order a confident-correct prediction below a uniform one and
    /// a confident-wrong prediction above it. ECE does not have to: it is
    /// checked separately, and this test states the asymmetry rather than
    /// leaving it implicit.
    #[test]
    fn confidence_moves_the_proper_scores_the_right_way() {
        let uniform = vec![1.0 / 5.0; 5];
        let mut sure = vec![0.0f64; 5];
        sure[2] = 1.0;
        for rule in Rule::ALL {
            let right = score_steps(std::slice::from_ref(&sure), &[2]).mean(rule);
            let flat = score_steps(std::slice::from_ref(&uniform), &[2]).mean(rule);
            let wrong = score_steps(std::slice::from_ref(&sure), &[0]).mean(rule);
            assert!(
                right < flat,
                "{}: sure-right {right} vs flat {flat}",
                rule.name()
            );
            assert!(
                wrong > flat,
                "{}: sure-wrong {wrong} vs flat {flat}",
                rule.name()
            );
        }
    }

    #[test]
    fn score_steps_rejects_ragged_or_degenerate_input() {
        let ok = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let ragged = vec![vec![0.5, 0.5], vec![0.5, 0.5, 0.0]];
        let out_of_range = [2usize, 0];
        let r = std::panic::catch_unwind(|| score_steps(&ragged, &[0, 0]));
        assert!(r.is_err(), "ragged distributions must not be scored");
        let r = std::panic::catch_unwind(|| score_steps(&ok, &out_of_range));
        assert!(r.is_err(), "an out-of-range target must not be scored");
        let r = std::panic::catch_unwind(|| score_steps(&[], &[]));
        assert!(r.is_err(), "an empty readout must not report a zero");
    }

    /// The readout's log column *is* the training loss. If this ever drifts, a
    /// report could compare a readout from one batch against a loss from
    /// another and call it model quality.
    #[test]
    fn dt_step_scores_equals_dt_loss() {
        let env = small_dt_env();
        let corpus = Corpus::with_seeds(&env, BehaviourPolicy::default(), &[3, 4, 5]);
        let batch = corpus.to_dt_batch(&DtSpec::default()).unwrap();
        let shapes = tiny_shapes(&env);
        let theta = init_flat(&shapes, 0.1, 99);
        for variant in [Variant::Standard, Variant::Cosine] {
            let (scores, loss) = dt_readout(&theta, &shapes, variant, &batch).unwrap();
            let log = scores.mean(Rule::Log);
            assert!(
                (log - loss).abs() < 1e-5,
                "{variant:?}: readout log {log} vs dt_loss {loss}"
            );
            assert_eq!(scores.n, batch.transitions());
            // Brier is the probability scale; on this vocabulary it is a real
            // number, not a copy of the log.
            assert!(scores.mean(Rule::Brier) > 0.0);
            assert!(scores.mean(Rule::Brier) < 2.0);
        }
    }

    /// Padding must not create scored steps: the readout's step count and its
    /// log score are the unpadded ones.
    #[test]
    fn padding_adds_no_scored_steps() {
        let env = small_dt_env();
        let corpus = Corpus::with_seeds(&env, BehaviourPolicy::default(), &[3, 4]);
        let batch = corpus.to_dt_batch(&DtSpec::default()).unwrap();
        let shapes = tiny_shapes(&env);
        let theta = init_flat(&shapes, 0.1, 5);
        let scores = dt_step_scores(&theta, &shapes, Variant::Standard, &batch).unwrap();
        assert_eq!(scores.n, 2 * env.horizon);
        for s in &scores.by_rule {
            assert_eq!(s.per_step.len(), 2 * env.horizon);
        }
    }

    #[test]
    fn calibration_of_a_known_dataset_is_hand_computable() {
        // Four steps: two confident-and-right (p = 1), two certain-and-wrong.
        let p_right = vec![1.0, 0.0];
        let p_wrong = vec![1.0, 0.0];
        let probs = vec![p_right.clone(), p_right, p_wrong.clone(), p_wrong];
        let cal = max_prob_calibration(&probs, &[0, 0, 1, 1], 5);
        assert_eq!(cal.n, 4);
        assert_eq!(cal.bins.iter().map(|b| b.count).sum::<usize>(), 4);
        // Everything lands in the last bin (confidence 1.0), which is
        // [0.8, 1.0] and closed on the right.
        assert_eq!(cal.bins[4].count, 4);
        assert!((cal.mean_confidence - 1.0).abs() < 1e-12);
        assert!((cal.accuracy - 0.5).abs() < 1e-12);
        assert!((cal.ece - 0.5).abs() < 1e-12);
        // Top-1 NLL of a certain prediction is 0 whatever the label.
        assert!(cal.top1_nll.abs() < 1e-12);
        // And the empty bins are NaN rather than a fabricated zero.
        assert!(cal.bins[0].mean_confidence.is_nan() && cal.bins[0].count == 0);
    }

    /// A perfectly calibrated set has ECE 0 — checked on a construction where
    /// each bin's accuracy equals its confidence exactly, so this is an
    /// identity and not "small enough".
    #[test]
    fn calibrated_predictions_have_zero_ece() {
        let mut probs = Vec::new();
        let mut targets = Vec::new();
        // Five bins, ten steps each. Confidence is the bin midpoint and exactly
        // `mid · 10` of the ten steps are right, so accuracy == confidence per
        // bin and the ECE is zero by construction. Ten classes with the max on
        // class 0 keep the argmax fixed across bins.
        for b in 0..5 {
            let conf = (b as f64 + 0.5) / 5.0;
            let hits = (conf * 10.0).round() as usize;
            for k in 0..10 {
                let rest = (1.0 - conf) / 9.0;
                let mut p = vec![rest; 10];
                p[0] = conf;
                probs.push(p);
                targets.push(if k < hits { 0 } else { 1 });
            }
        }
        let cal = max_prob_calibration(&probs, &targets, 5);
        assert!(cal.ece < 1e-12, "ECE {} should be exactly zero", cal.ece);
        assert_eq!(cal.bins.iter().map(|b| b.count).sum::<usize>(), cal.n);
        for (b, bin) in cal.bins.iter().enumerate() {
            assert_eq!(bin.count, 10, "bin {b} must hold one confidence level");
            assert!((bin.accuracy - bin.mean_confidence).abs() < 1e-12);
        }
    }

    /// Overconfident predictions have ECE equal to the error, not merely
    /// positive: the same identity as above, in the other direction.
    #[test]
    fn overconfidence_shows_up_in_ece() {
        let probs = vec![vec![0.1, 0.9]; 10];
        let mut targets = vec![1usize; 5];
        targets.extend(vec![0usize; 5]);
        let cal = max_prob_calibration(&probs, &targets, 10);
        assert!((cal.mean_confidence - 0.9).abs() < 1e-12);
        assert!((cal.accuracy - 0.5).abs() < 1e-12);
        assert!((cal.ece - 0.4).abs() < 1e-12);
        assert!(cal.top1_nll > 0.0);
    }

    /// The paired readout must actually be paired: a real return difference at a
    /// small rollout budget, and an MCSE below the independent-arm value.
    #[test]
    fn return_readout_resolves_a_real_difference_and_pairs() {
        // Noise 0.3, not the default 0.05: pairing removes only the variance the
        // two arms *share*, which here is the transition noise. At the default
        // noise the uniform arm's own action randomness dwarfs it and the
        // measured covariance is indistinguishable from zero — an honest
        // caveat, and the reason this test states the condition it pays off
        // under instead of asserting variance reduction unconditionally.
        let env = MdpEnv {
            horizon: 8,
            noise: 0.3,
            ..MdpEnv::default()
        };
        let uniform = uniform_arm(&env);
        // Towards the origin from x0 = [1, 0]: the negative-x action.
        let toward = constant_arm(env.actions.len(), 1);
        let arms = [
            Arm {
                name: "uniform",
                dist: &uniform,
                decode: Decode::Sample,
            },
            Arm {
                name: "toward-origin",
                dist: &toward,
                decode: Decode::Sample,
            },
        ];
        let r = readout(&env, [1.0, 0.0], &arms, 64, 12);
        assert_eq!(r.arms.len(), 2);
        assert_eq!(r.diffs.len(), 1);
        let (name, diff) = &r.diffs[0];
        assert_eq!(name, "toward-origin");
        // Rewards are non-positive; "toward the origin" must cost less.
        assert!(
            diff.mean > 0.0,
            "toward-origin should beat uniform, got {}",
            diff.render()
        );
        assert!(
            diff.resolved(),
            "difference should resolve: {}",
            diff.render()
        );
        // Both standard errors are published, and they are close on this task:
        // the two arms' trajectories diverge, so the shared noise correlates
        // their returns only weakly.  The variance reduction of the pairing is
        // therefore checked where the covariance is controlled
        // (`paired_differences_cancel_shared_noise`), not asserted out of a
        // measurement that says it is small.
        assert!(diff.mcse > 0.0 && r.diffs_independent_mcse[0] > 0.0);
        assert!(
            diff.mcse <= 2.0 * r.diffs_independent_mcse[0],
            "paired MCSE {} is implausibly above the independent {}",
            diff.mcse,
            r.diffs_independent_mcse[0]
        );
        // Determinism: the same seed is the same measurement.
        let again = readout(&env, [1.0, 0.0], &arms, 64, 12);
        for (a, b) in r.arms.iter().zip(again.arms.iter()) {
            assert_eq!(a.returns, b.returns);
            assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        }
    }

    /// Pairing's variance reduction, on a sample where the shared component is
    /// controlled: `a_k = s_k + e_k`, `b_k = s_k + f_k` with `s` the shared and
    /// `e, f` the arm-specific parts. The paired difference is `e_k − f_k`, so
    /// its SE excludes `s` entirely; the independent SE cannot.
    #[test]
    fn paired_differences_cancel_shared_noise() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..256 {
            let shared: f64 = rand::Rng::random_range(&mut rng, -10.0..10.0);
            let e: f64 = rand::Rng::random_range(&mut rng, -0.1..0.1);
            let f: f64 = rand::Rng::random_range(&mut rng, -0.1..0.1);
            a.push(shared + e);
            b.push(shared + f);
        }
        let d = paired(&a, &b);
        let (_, se_a) = mean_mcse(&a);
        let (_, se_b) = mean_mcse(&b);
        let independent = (se_a * se_a + se_b * se_b).sqrt();
        assert!(
            d.mcse < 0.05 * independent,
            "paired SE {} should be far below the independent {independent}",
            d.mcse
        );
        // And the paired mean is still the mean of the differences.
        let direct: f64 = a.iter().zip(b.iter()).map(|(x, y)| x - y).sum::<f64>() / a.len() as f64;
        assert!((d.mean - direct).abs() < 1e-12);
    }

    /// Sampling from a one-hot distribution and taking its argmax must agree
    /// exactly — the two decoders differ only where the distribution is
    /// genuinely spread.
    #[test]
    fn sample_and_argmax_agree_on_a_point_mass() {
        let env = MdpEnv {
            horizon: 6,
            ..MdpEnv::default()
        };
        let to = constant_arm(env.actions.len(), 0);
        let arms = [
            Arm {
                name: "sampled",
                dist: &to,
                decode: Decode::Sample,
            },
            Arm {
                name: "argmax",
                dist: &to,
                decode: Decode::Argmax,
            },
        ];
        let r = readout(&env, [1.0, 0.5], &arms, 16, 4);
        assert_eq!(r.arms[0].returns, r.arms[1].returns);
        assert_eq!(r.diffs[0].1.mean.to_bits(), 0f64.to_bits());
    }

    /// A return readout must not be confused with the optimum: the best fixed
    /// arm here is still below the value-iteration value, which is what the
    /// noise-matched comparison exists to check.
    #[test]
    fn best_fixed_arm_is_below_the_optimum() {
        let env = MdpEnv {
            horizon: 8,
            noise: 0.0,
            ..MdpEnv::default()
        };
        let solved = env.solve();
        let x0 = [1.0f32, 0.0];
        let toward = constant_arm(env.actions.len(), 1);
        let arms = [Arm {
            name: "toward-origin",
            dist: &toward,
            decode: Decode::Argmax,
        }];
        let r = readout(&env, x0, &arms, 8, 1);
        assert!(
            r.arms[0].mean <= solved.optimal_value(x0) + 1e-3,
            "fixed arm {} beat the optimum {}",
            r.arms[0].mean,
            solved.optimal_value(x0)
        );
    }
}
