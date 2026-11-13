//! §R7.4 — the DT arm, assembled so `policy_readout` can report it.
//!
//! This module is the *wiring*, not a new statistical claim.  A decision
//! transformer is only comparable with the classification tables if its anchor,
//! its corpora and its readouts are the ambient ones, so this assembles them
//! from the existing pieces rather than re-deriving any of them:
//!
//! * the anchor is `experiment::PotentialAnchor` (the classifier's seeds, hash,
//!   clip and momentum, with `policy_model::DtPotential` in place of the
//!   likelihood) pre-trained under `NormGrad::Stopped`, the convention every
//!   recorded anchor used;
//! * the corpora are `mdp::Corpus::splits` under two *behaviour policies* — the
//!   anchor's task and the adaptation task differ only in `BehaviourPolicy`,
//!   which is R7.4's "same MDP, different behaviour policy";
//! * chains are the published unadjusted SG-pCNL kernel at the **full** target
//!   (`SgPcnlState::step_unadjusted`, sum-form batch gradient scaled `N/B`),
//!   so the draws are the same kernel as R2r's, restricted to no subspace.
//!
//! # Measured (seed 11, `examples/dt_cell`)
//!
//! Anchor hash `79f9638aa261e3e5`, pre-train NLL 1.6216 → 0.6428; two chains of
//! 600 steps keep 20 draws.  Dev: anchor NLL 1.3620 / Brier 0.7517 / top-1
//! 0.352 / ECE 0.213, `p̄` 1.1045 / 0.5667 / 0.602 / 0.204.  Test: 1.5925 /
//! 0.7858 / 0.344 / 0.215 vs 1.1056 / 0.5434 / 0.680 / 0.177.  Returns at
//! `x0 = [1, 0.5]`, 96 rollouts: behaviour −6.09 ± 0.34, anchor-sample −3.64 ±
//! 0.13, anchor-argmax −2.54 ± 0.03, `p̄`-sample −7.37 ± 0.49, `p̄`-argmax −5.61
//! ± 0.10, optimum −2.42 ± 0.02; paired vs behaviour +2.45 ± 0.34, +3.55 ± 0.34,
//! −1.28 ± 0.52, +0.47 ± 0.33 (within noise), +3.67 ± 0.34.
//!
//! That is the plan's **calibration-only** branch, measured: `p̄` improves every
//! per-step score and top-1 match, and its *sampled* return is worse than the
//! behaviour policy's — better next-action distributions, no decision gain.  The
//! cause is visible in the same run: `P = 17 408` and `σ√P = 46.2`, with the
//! chains at radius 49.4, so the draws span the whole prior shell and `p̄`
//! averages over policies that differ far more than the likelihood does.  R7.4's
//! rank-4 restriction is what makes `σ = 0.35` a meaningful posterior width; at
//! the full target this cell is prior-dominated and must not be quoted as the
//! R7.4 comparison.
//!
//! What this module deliberately lacks: the rank-4 informed basis, the Adam
//! point baseline, the ensemble and the temperature tuning.  A `DtCell`
//! therefore reports *anchor vs posterior* and says so.  Those pieces live in
//! [`crate::dt_subspace`] (`examples/dt_subspace`), which is where the measured
//! R7.4 cell is; its promoted rows are in `docs/PCNL_STATUS.md` §R7.4 and are
//! **not** comparable with the full-target rows above, because the target is a
//! rank-4 restriction.  The readout half here is complete: per-step proper
//! scores under all three rules, max-probability calibration with per-bin
//! counts, and paired return readouts with MCSEs for the behaviour policy, the
//! sampled and averaged policies, and the exact optimum.

use crate::experiment::{PotentialAnchor, anchor_hash};
use crate::mdp::{BehaviourPolicy, Corpus, DtSpec, MdpEnv};
use crate::model::Variant;
use crate::normalize::NormGrad;
use crate::policy_model::{DtBatch, DtPotential, DtShapes, dt_act, dt_loss_and_grad, init_flat};
use crate::policy_readout::{
    Arm, Calibration, Context, Decode, ReturnReadout, StepScores, dt_step_distributions,
    max_prob_calibration, readout, score_steps,
};
use candle_core::{Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The start state the return readout fixes for every arm and every arm's
/// replicate.  The corpora walk starts drawn from `N(0, 1)`, so a return table
/// has to name its own start; `[1.0, 0.5]` is inside the planning grid and
/// off the origin, which is where the cost is.
pub const START: [f32; 2] = [1.0, 0.5];

/// A DT cell: corpora, anchor and (after [`DtCell::run_chains`]) draws.
pub struct DtCell {
    pub env: MdpEnv,
    pub shapes: DtShapes,
    pub variant: Variant,
    pub spec: DtSpec,
    pub anchor_train: Corpus,
    pub anchor_dev: Corpus,
    pub adapt_train: Corpus,
    pub adapt_dev: Corpus,
    pub adapt_test: Corpus,
    pub anchor: Option<Tensor>,
    pub anchor_hash: Option<u64>,
    pub draws: Vec<Tensor>,
}

/// The chain settings R7.4's screening row uses, in one value.
#[derive(Clone, Copy, Debug)]
pub struct ChainCfg {
    pub chains: usize,
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub sigma: f64,
    /// Trajectories per mini-batch, not transitions.
    pub batch: usize,
    pub seed: u64,
}

impl Default for ChainCfg {
    fn default() -> Self {
        Self {
            chains: 2,
            steps: 600,
            burn: 400,
            thin: 20,
            sigma: 0.35,
            batch: 8,
            seed: 7,
        }
    }
}

/// What the chains cost and whether they stayed put — the numbers a reader
/// needs before any score from the draws is quotable.
#[derive(Clone, Debug)]
pub struct ChainReport {
    pub chains: usize,
    pub steps: usize,
    pub kept: usize,
    /// Mean per-transition NLL of each chain's kept draws, chain by chain.
    pub mean_nll_per_chain: Vec<f64>,
    /// Largest radial norm reached by any chain: a chain that drifts to the
    /// boundary is a mixing failure, not a posterior.
    pub rad_max: f64,
}

impl DtCell {
    /// The R7.4 shapes (`d = 32`, `ff = 64`, 2 layers, 5 actions, 4 RTG bins,
    /// `T = 16`) and two behaviour policies over the same MDP.
    pub fn new(seed: u64) -> Self {
        // Anchor: near-greedy feedback, little exploration.  Adaptation: weak
        // feedback, heavy exploration — a different behaviour policy, not a
        // different environment.
        Self::with_policies(
            seed,
            BehaviourPolicy {
                feedback: 1.0,
                epsilon: 0.1,
                seed,
            },
            BehaviourPolicy {
                feedback: 0.3,
                epsilon: 0.3,
                seed: seed ^ 0xa11ce,
            },
        )
    }

    /// [`DtCell::new`] with the two behaviour policies named by the caller.
    ///
    /// This is R7.6's "anchor as behaviour policy" axis: the *anchor* corpus —
    /// hence the prior mean, the informed basis built from it, and the target's
    /// centre — comes from `anchor_policy`, while the adaptation corpora come
    /// from `adapt_policy`.  Only the anchor side is meant to be swept; the
    /// adaptation target is the thing being predicted and must stay fixed.
    pub fn with_policies(
        seed: u64,
        anchor_policy: BehaviourPolicy,
        adapt_policy: BehaviourPolicy,
    ) -> Self {
        let env = MdpEnv::default();
        let spec = DtSpec::default();
        let shapes = env.dt_shapes(spec.rtg_bins, 32, 64, 2);
        let (anchor_train, anchor_dev, _) = Corpus::splits(&env, anchor_policy, (48, 8, 8), seed);
        let (adapt_train, adapt_dev, adapt_test) =
            Corpus::splits(&env, adapt_policy, (48, 8, 8), seed.wrapping_add(7));
        Self {
            env,
            shapes,
            variant: Variant::Cosine,
            spec,
            anchor_train,
            anchor_dev,
            adapt_train,
            adapt_dev,
            adapt_test,
            anchor: None,
            anchor_hash: None,
            draws: Vec::new(),
        }
    }

    pub fn transitions(&self, corpus: &Corpus) -> usize {
        corpus.transitions()
    }

    fn batch(&self, corpus: &Corpus) -> Result<DtBatch> {
        corpus.to_dt_batch(&self.spec)
    }

    /// Pre-train the anchor on the anchor task; returns the loss trace.
    ///
    /// `NormGrad::Stopped` is not a default here — it is the convention the
    /// recorded anchors were trained with, and `PotentialAnchor`'s doc explains
    /// why the caller, not the potential, states it.
    pub fn pretrain(&mut self, steps: usize, lr: f64, momentum: f64) -> Result<Vec<f64>> {
        let theta0 = init_flat(&self.shapes, 0.1, 42);
        let anchor = PotentialAnchor::new(
            DtPotential::new(self.batch(&self.anchor_train)?, self.shapes, self.variant),
            DtPotential::new(self.batch(&self.anchor_dev)?, self.shapes, self.variant),
            theta0,
        );
        let (theta, trace) = anchor.pretrain(steps, lr, momentum, NormGrad::Stopped);
        self.anchor_hash = Some(anchor_hash(&theta));
        self.anchor = Some(theta);
        Ok(trace)
    }

    /// Unadjusted SG-pCNL chains at the full DT target, centred at the anchor.
    ///
    /// A mini-batch is a set of whole trajectories (the DT's terms are
    /// transitions, so `batch_size` is `trajectories × horizon`); the gradient
    /// is passed in sum-form and the kernel rescales by `N/B`, which is the
    /// same convention `Dataset::batch_grad` uses.
    pub fn run_chains(&mut self, cfg: &ChainCfg) -> Result<ChainReport> {
        let ChainCfg {
            chains,
            steps,
            burn,
            thin,
            sigma,
            batch,
            seed,
        } = *cfg;
        let anchor = self
            .anchor
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("run_chains: pretrain first".into()))?;
        let sched = crate::sampler::SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.5,
        };
        let n_total = self.adapt_train.transitions();
        let mut draws = Vec::new();
        let mut per_chain = Vec::new();
        let mut rad_max = 0.0f64;
        for c in 0..chains {
            let mut state = crate::sampler::SgPcnlState::new(
                anchor.clone(),
                sigma,
                sched,
                n_total,
                batch * self.env.horizon,
                0,
            )
            .centered_at(anchor.clone());
            let mut rng = StdRng::seed_from_u64(seed ^ ((c as u64) << 8));
            let mut kept_here = Vec::new();
            let mut order: Vec<usize> = (0..self.adapt_train.trajectories.len()).collect();
            for t in 0..steps {
                // A fresh mini-batch of trajectories, without replacement
                // within the step (the same discipline `BatchLoader` follows).
                for i in (1..order.len()).rev() {
                    let j = rng.random_range(0..=i);
                    order.swap(i, j);
                }
                let idx: Vec<usize> = order[..batch.min(order.len())].to_vec();
                let sub = Corpus {
                    env: self.env.clone(),
                    policy: self.adapt_train.policy,
                    trajectory_ids: idx
                        .iter()
                        .map(|&i| self.adapt_train.trajectory_ids[i])
                        .collect(),
                    trajectories: idx
                        .iter()
                        .map(|&i| self.adapt_train.trajectories[i].clone())
                        .collect(),
                };
                let b = sub.to_dt_batch(&self.spec)?;
                let (loss, mean_grad) = dt_loss_and_grad(
                    &state.theta,
                    &self.shapes,
                    self.variant,
                    &b,
                    NormGrad::Stopped,
                )?;
                if !loss.is_finite() {
                    return Err(candle_core::Error::Msg(format!(
                        "run_chains: non-finite loss {loss} at step {t} of chain {c}"
                    )));
                }
                // Mean per transition over the batch → sum-form over the batch.
                let g_sum = mean_grad.affine(b.transitions() as f64, 0.0)?;
                state.step_unadjusted(&g_sum, &mut rng)?;
                let r = crate::sampler::norm(&state.theta)?;
                if r.is_finite() {
                    rad_max = rad_max.max(r);
                }
                if t >= burn && (t - burn).is_multiple_of(thin) {
                    kept_here.push(state.theta.clone());
                }
            }
            let nll = kept_here
                .iter()
                .map(|th| {
                    let b = self.batch(&self.adapt_dev)?;
                    dt_loss_and_grad(th, &self.shapes, self.variant, &b, NormGrad::Stopped)
                        .map(|(l, _)| l)
                })
                .collect::<Result<Vec<f64>>>()?;
            let mean = nll.iter().sum::<f64>() / nll.len().max(1) as f64;
            per_chain.push(mean);
            draws.extend(kept_here);
        }
        self.draws = draws;
        Ok(ChainReport {
            chains,
            steps,
            kept: self.draws.len(),
            mean_nll_per_chain: per_chain,
            rad_max,
        })
    }

    /// Per-step scores and the loss, for one point, on one corpus.
    pub fn scores_at(&self, theta: &Tensor, corpus: &Corpus) -> Result<(StepScores, f64)> {
        let b = self.batch(corpus)?;
        let scores = crate::policy_readout::dt_step_scores(theta, &self.shapes, self.variant, &b)?;
        let (loss, _) = dt_loss_and_grad(theta, &self.shapes, self.variant, &b, NormGrad::Stopped)?;
        Ok((scores, loss))
    }

    /// Max-probability calibration for one point, on one corpus.
    pub fn calibration_at(
        &self,
        theta: &Tensor,
        corpus: &Corpus,
        bins: usize,
    ) -> Result<Calibration> {
        let b = self.batch(corpus)?;
        let dists = dt_step_distributions(theta, &self.shapes, self.variant, &b)?;
        let mut targets = Vec::new();
        let (_, act) = b.scored_steps()?;
        targets.extend(act.into_iter().map(|a| a as usize));
        Ok(max_prob_calibration(&dists, &targets, bins))
    }

    /// The posterior predictive `p̄`: the mean of the kept draws' per-step
    /// distributions.  This is the object R7.4's score column is about.
    pub fn posterior_distributions(&self, corpus: &Corpus) -> Result<Vec<Vec<f64>>> {
        if self.draws.is_empty() {
            return Err(candle_core::Error::Msg(
                "posterior_distributions: no draws (run_chains first)".into(),
            ));
        }
        let b = self.batch(corpus)?;
        let mut acc: Vec<Vec<f64>> = Vec::new();
        for (i, th) in self.draws.iter().enumerate() {
            let d = dt_step_distributions(th, &self.shapes, self.variant, &b)?;
            if i == 0 {
                acc = d;
            } else {
                for (a, row) in acc.iter_mut().zip(d.iter()) {
                    for (x, y) in a.iter_mut().zip(row.iter()) {
                        *x += y;
                    }
                }
            }
        }
        let n = self.draws.len() as f64;
        for row in acc.iter_mut() {
            for x in row.iter_mut() {
                *x /= n;
            }
        }
        Ok(acc)
    }

    /// The action actually taken at each scored step, in `scored_steps` order
    /// — the targets the score and calibration columns share.
    pub fn targets(&self, corpus: &Corpus) -> Result<Vec<usize>> {
        let b = self.batch(corpus)?;
        let (_, act) = b.scored_steps()?;
        Ok(act.into_iter().map(|a| a as usize).collect())
    }

    /// Calibration of the posterior predictive `p̄` on one corpus.
    pub fn posterior_calibration(&self, corpus: &Corpus, bins: usize) -> Result<Calibration> {
        let dists = self.posterior_distributions(corpus)?;
        let targets = self.targets(corpus)?;
        Ok(max_prob_calibration(&dists, &targets, bins))
    }

    pub fn posterior_scores(&self, corpus: &Corpus) -> Result<StepScores> {
        let b = self.batch(corpus)?;
        let dists = self.posterior_distributions(corpus)?;
        let (_, act) = b.scored_steps()?;
        let targets: Vec<usize> = act.into_iter().map(|a| a as usize).collect();
        Ok(score_steps(&dists, &targets))
    }

    /// The return the DT is conditioned on: the best return in the adaptation
    /// corpus.  A choice, made explicit: conditioning on the value-iteration
    /// optimum is off the returned-return distribution the model was fit on,
    /// and a DT cannot be asked for a return its data never contained.
    pub fn rtg_conditioning(&self) -> f32 {
        self.adapt_train
            .trajectories
            .iter()
            .map(|t| t.total_return())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// The paired return readout: behaviour policy, anchor (sampled and
    /// argmax), posterior predictive (sampled and argmax), and the exact
    /// optimum.  All arms share one uniform stream and one noise sequence per
    /// replicate, so the differences are paired.
    pub fn returns(&self, rollouts: usize, seed: u64) -> Result<ReturnReadout> {
        let anchor = self
            .anchor
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("returns: pretrain first".into()))?;
        let target = self.rtg_conditioning();
        let solved = self.env.solve();
        let behaviour = behaviour_policy(&self.env, self.adapt_train.policy);
        let anchor_sample = dt_policy(
            &self.env,
            self.spec,
            anchor.clone(),
            self.shapes,
            self.variant,
            target,
        );
        let anchor_argmax = dt_policy(
            &self.env,
            self.spec,
            anchor.clone(),
            self.shapes,
            self.variant,
            target,
        );
        let mean_sample = mean_policy(
            &self.env,
            self.spec,
            &self.draws,
            self.shapes,
            self.variant,
            target,
        );
        let mean_argmax = mean_policy(
            &self.env,
            self.spec,
            &self.draws,
            self.shapes,
            self.variant,
            target,
        );
        let k = self.env.actions.len();
        let optimal = move |ctx: &Context| {
            let mut d = vec![0.0f64; k];
            d[solved.greedy_action(ctx.t, ctx.state)] = 1.0;
            d
        };
        let arms = vec![
            Arm {
                name: "behaviour",
                dist: &behaviour,
                decode: Decode::Sample,
            },
            Arm {
                name: "anchor-sample",
                dist: &anchor_sample,
                decode: Decode::Sample,
            },
            Arm {
                name: "anchor-argmax",
                dist: &anchor_argmax,
                decode: Decode::Argmax,
            },
            Arm {
                name: "mean-sample",
                dist: &mean_sample,
                decode: Decode::Sample,
            },
            Arm {
                name: "mean-argmax",
                dist: &mean_argmax,
                decode: Decode::Argmax,
            },
            Arm {
                name: "optimal",
                dist: &optimal,
                decode: Decode::Argmax,
            },
        ];
        Ok(readout(&self.env, START, &arms, rollouts, seed))
    }
}

/// The conditioning bins for a rollout whose target return is `target`: the
/// *remaining* return `G − Σ_{k<t} r_k` at step `t`, binned exactly as
/// [`Corpus::to_dt_batch`] bins `rtgs[t]`.  Feeding a constant `G` instead
/// would query rows the corpus never visited — the same return-to-go appears at
/// every step only if the policy has earned nothing yet.
pub fn conditioning_bins(
    env: &MdpEnv,
    spec: &DtSpec,
    target: f32,
    states: &[[f32; 2]],
    actions: &[usize],
) -> Vec<u32> {
    let mut bins = Vec::with_capacity(actions.len() + 1);
    let mut earned = 0.0f32;
    for t in 0..states.len() {
        bins.push(spec.bin(target - earned) as u32);
        if t < actions.len() {
            earned += env.reward(states[t], actions[t]);
        }
    }
    bins
}

/// A DT as an `ActionDist`: the prefix and the remaining target return are the
/// rollout's own, and the row is the one the loss scored at that step.
pub fn dt_policy(
    env: &MdpEnv,
    spec: DtSpec,
    theta: Tensor,
    shapes: DtShapes,
    variant: Variant,
    target: f32,
) -> impl Fn(&Context) -> Vec<f64> + use<> {
    let env = env.clone();
    move |ctx: &Context| {
        let acts: Vec<u32> = ctx.actions.iter().map(|&a| a as u32).collect();
        let bins = conditioning_bins(&env, &spec, target, ctx.states, ctx.actions);
        dt_act(&theta, &shapes, variant, &bins, ctx.states, &acts).expect("dt_act")
    }
}

/// The posterior predictive as an `ActionDist`: every draw's distribution at
/// the same `(t, history)` averaged, which is the average the predictive `p̄`
/// promises.
pub fn mean_policy(
    env: &MdpEnv,
    spec: DtSpec,
    draws: &[Tensor],
    shapes: DtShapes,
    variant: Variant,
    target: f32,
) -> impl Fn(&Context) -> Vec<f64> + use<> {
    let draws: Vec<Tensor> = draws.to_vec();
    let env = env.clone();
    assert!(!draws.is_empty(), "mean_policy needs at least one draw");
    move |ctx: &Context| {
        let acts: Vec<u32> = ctx.actions.iter().map(|&a| a as u32).collect();
        let bins = conditioning_bins(&env, &spec, target, ctx.states, ctx.actions);
        let k = dt_act(&draws[0], &shapes, variant, &bins, ctx.states, &acts)
            .expect("dt_act")
            .len();
        let mut mean = vec![0.0f64; k];
        for th in &draws {
            let d = dt_act(th, &shapes, variant, &bins, ctx.states, &acts).expect("dt_act");
            for (m, x) in mean.iter_mut().zip(d.iter()) {
                *m += x;
            }
        }
        let n = draws.len() as f64;
        for m in mean.iter_mut() {
            *m /= n;
        }
        mean
    }
}

/// A behaviour policy as a distribution: `ε/k` everywhere plus `1−ε` on the
/// action closest to the feedback target.  The same rule
/// `BehaviourPolicy::action` samples, written as a distribution so the readout
/// can couple it to the other arms.
pub fn behaviour_policy(
    env: &MdpEnv,
    policy: BehaviourPolicy,
) -> impl Fn(&Context) -> Vec<f64> + use<> {
    let k = env.actions.len();
    let feedback = policy.feedback;
    let epsilon = policy.epsilon;
    let actions = env.actions.clone();
    move |ctx: &Context| {
        let want = [-feedback * ctx.state[0], -feedback * ctx.state[1]];
        let best = actions
            .iter()
            .enumerate()
            .fold((0usize, f32::INFINITY), |(bi, bd), (i, a)| {
                let d = (a[0] - want[0]).powi(2) + (a[1] - want[1]).powi(2);
                if d < bd { (i, d) } else { (bi, bd) }
            })
            .0;
        let mut d = vec![f64::from(epsilon) / k as f64; k];
        d[best] += 1.0 - f64::from(epsilon);
        d
    }
}

/// `\bar p`-independent sanity: the shape a caller can rely on.
pub const ARM_NAMES: [&str; 6] = [
    "behaviour",
    "anchor-sample",
    "anchor-argmax",
    "mean-sample",
    "mean-argmax",
    "optimal",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_policies_reproduces_new_and_moves_only_the_anchor_corpus() {
        // R7.6's anchor axis must not quietly change the adaptation target:
        // the default call is bitwise `new`, and a moved anchor policy moves
        // only the anchor corpora.
        let base = DtCell::new(11);
        let same = DtCell::with_policies(
            11,
            BehaviourPolicy {
                feedback: 1.0,
                epsilon: 0.1,
                seed: 11,
            },
            BehaviourPolicy {
                feedback: 0.3,
                epsilon: 0.3,
                seed: 11 ^ 0xa11ce,
            },
        );
        assert_eq!(
            base.anchor_train.content_hash(),
            same.anchor_train.content_hash()
        );
        assert_eq!(
            base.adapt_test.content_hash(),
            same.adapt_test.content_hash()
        );
        let moved = DtCell::with_policies(
            11,
            BehaviourPolicy {
                feedback: 0.3,
                epsilon: 0.1,
                seed: 11,
            },
            BehaviourPolicy {
                feedback: 0.3,
                epsilon: 0.3,
                seed: 11 ^ 0xa11ce,
            },
        );
        assert_ne!(
            base.anchor_train.content_hash(),
            moved.anchor_train.content_hash()
        );
        assert_eq!(
            base.adapt_train.content_hash(),
            moved.adapt_train.content_hash()
        );
    }

    #[test]
    fn dt_act_is_the_row_the_loss_scores() {
        let cell = DtCell::new(3);
        let theta = init_flat(&cell.shapes, 0.1, 7);
        let traj = &cell.anchor_dev.trajectories[0];
        // Walk the true trajectory: dt_act at step t must reproduce the
        // training batch's distribution at that step for the same prefix.
        let b = cell.anchor_dev.to_dt_batch(&cell.spec).unwrap();
        // Gathered at the scored steps, in `scored_steps` order — so for the
        // first trajectory the entry for step `t` is entry `t`, not the flat
        // row `3t + 1`.
        let dists = dt_step_distributions(&theta, &cell.shapes, cell.variant, &b).unwrap();
        for (t, row) in dists.iter().enumerate().take(cell.env.horizon) {
            let states: Vec<[f32; 2]> = traj.states[..=t].to_vec();
            let acts: Vec<u32> = traj.actions[..t].iter().map(|&a| a as u32).collect();
            let bins: Vec<u32> = (0..=t)
                .map(|k| cell.spec.bin(traj.rtgs[k]) as u32)
                .collect();
            let got = dt_act(&theta, &cell.shapes, cell.variant, &bins, &states, &acts).unwrap();
            for (a, b) in got.iter().zip(row.iter()) {
                assert!((a - b).abs() < 1e-6, "step {t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn anchor_scores_and_calibration_are_consistent() {
        let mut cell = DtCell::new(5);
        let trace = cell.pretrain(20, 0.05, 0.9).unwrap();
        assert_eq!(trace.len(), 20);
        let theta = cell.anchor.clone().unwrap();
        let (scores, loss) = cell.scores_at(&theta, &cell.adapt_dev).unwrap();
        assert!((scores.mean(crate::readout::Rule::Log) - loss).abs() < 1e-5);
        let cal = cell.calibration_at(&theta, &cell.adapt_dev, 5).unwrap();
        assert_eq!(cal.bins.iter().map(|b| b.count).sum::<usize>(), scores.n);
    }

    #[test]
    fn the_readout_arms_are_named_and_seeded() {
        let mut cell = DtCell::new(9);
        cell.pretrain(4, 0.02, 0.9).unwrap();
        cell.run_chains(&ChainCfg {
            chains: 1,
            steps: 40,
            burn: 20,
            thin: 5,
            batch: 4,
            seed: 1,
            ..ChainCfg::default()
        })
        .unwrap();
        let r = cell.returns(8, 4).unwrap();
        let names: Vec<&str> = r.arms.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ARM_NAMES);
        assert_eq!(r.diffs.len(), ARM_NAMES.len() - 1);
        // The averaged decoder is deterministic in the state, so it is the
        // comparison against the sampled arm that carries the pairing.
        let again = cell.returns(8, 4).unwrap();
        for (a, b) in r.arms.iter().zip(again.arms.iter()) {
            assert_eq!(a.returns, b.returns);
        }
    }
}
