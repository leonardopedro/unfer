//! E3/E4 — the bounded supervised-vs-RLCD campaign, in the small.
//!
//! This module implements the *mechanics* the plan's E3/E4 packets demand:
//! a common optional supervised warm start, the mandatory CE+Brier comparator,
//! the same-objective exact-marginal control, the required RLCD arm (a genuine
//! score-function update of the four-path policy), a frozen-policy readout arm,
//! the per-path proper-score and correctness-reward controls, world-split
//! evaluation against the known outcome law, route diagnostics and a
//! world-clustered paired contrast.
//!
//! **It is not the declared R9 campaign.** `EDGE0_RLCD_PLAN.md` §5 specifies
//! 5 paired seeds, 2048/256/256/512 worlds and up to 2000 optimizer updates;
//! that run is a compute campaign, not a unit of implementation. The defaults
//! here are a deliberately labelled, reduced-budget mechanics run: the same
//! arms, the same paired split and the same primary endpoint, at a size that can
//! execute in CI. Any number printed by [`run`] is a mechanics result and must
//! not be promoted as an R9 finding.

use crate::edge0_model::{Edge0Config, ModelInput, QuestionInput, census, forward, init};
use crate::edge0_rlcd::{PATHS, Streams};
use crate::edge0_world::{FEATURES, QuestionKind, Split, World, WorldManifest, brier_excess};
use candle_core::{DType, Result, Tensor, Var};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// One labelled question drawn from one world.
#[derive(Clone, Debug)]
pub struct Sample {
    pub world: u32,
    pub split: Split,
    /// Question id within the world (`0..3`), used to rebuild its utility
    /// table for the secondary decision endpoints.
    pub qid: u64,
    pub input_state: Vec<u32>,
    pub question: QuestionInput,
    pub label: usize,
    /// The world's own law — the primary endpoint's `t`.
    pub truth: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CampaignConfig {
    pub model: Edge0Config,
    /// Training seed: parameter initialisation and path streams.
    pub seed: u64,
    /// Seed for the frozen world manifests and label stream. `None` ties it to
    /// `seed`; a paired multi-seed campaign sets it once for all seeds.
    pub data_seed: Option<u64>,
    /// `Questions` reproduces the mechanics runs; `Worlds` is the declared
    /// campaign's batch unit (32 worlds, all of their questions per update).
    pub batch_unit: BatchUnit,
    /// Reduced-budget split; the plan's full split is much larger.
    pub worlds: [usize; 4],
    /// Reduced-budget optimizer updates per arm.
    pub updates: usize,
    /// Shared supervised warm-start updates (zero disables it).
    pub warm_start: usize,
    pub batch_worlds: usize,
    pub lr: f64,
    pub momentum: f64,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            model: Edge0Config::tiny(),
            seed: 2026_11_15,
            data_seed: None,
            batch_unit: BatchUnit::Questions,
            worlds: [24, 8, 8, 16],
            updates: 60,
            warm_start: 30,
            batch_worlds: 4,
            lr: 0.05,
            momentum: 0.9,
        }
    }
}

/// What one optimizer update's batch counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchUnit {
    /// `batch_worlds` labelled questions (the mechanics default, which predates
    /// the declared campaign).
    #[default]
    Questions,
    /// `batch_worlds` distinct worlds with every question of each — the
    /// declared R9 unit, so question count per update is data-driven.
    Worlds,
}

/// The arms the plan's E3 packet requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Mandatory comparator (and shared warm start source): CE + Brier on the
    /// exact marginal.
    Supervised,
    /// Same-objective control: differentiate the exact-marginal Brier through
    /// the policy. This is *not* RLCD; it isolates the objective from the
    /// estimator.
    ExactMarginal,
    /// The required reinforcement arm: sampled paths, U-statistic Brier plus
    /// the score-function term.
    Rlcd,
    /// Frozen policy, direct readout updates only.
    FrozenPolicy,
    /// Per-path proper score: sampled paths, each path's readout scored against
    /// the label. No marginal mixing and no policy-gradient term.
    PerPath,
    /// Correctness-reward policy gradient: sampled paths rewarded by whether
    /// their own argmax is the label. A deliberately non-proper reward control.
    CorrectnessReward,
}

impl Arm {
    pub const ALL: [Arm; 6] = [
        Arm::Supervised,
        Arm::ExactMarginal,
        Arm::Rlcd,
        Arm::FrozenPolicy,
        Arm::PerPath,
        Arm::CorrectnessReward,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Arm::Supervised => "supervised CE+Brier",
            Arm::ExactMarginal => "exact-marginal Brier control",
            Arm::Rlcd => "RLCD (required)",
            Arm::FrozenPolicy => "frozen-policy readout",
            Arm::PerPath => "per-path proper score",
            Arm::CorrectnessReward => "correctness-reward RL",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArmReport {
    pub arm: Arm,
    /// Mean Brier excess over the known laws (primary endpoint).
    pub brier_excess: f64,
    pub nll: f64,
    pub policy_entropy: f64,
    pub mean_path_entropy: f64,
    pub updates: usize,
    /// Forward + backward passes actually run (cost, not a proxy).
    pub passes: usize,
}

#[derive(Clone, Debug)]
pub struct CampaignReport {
    pub config: CampaignConfig,
    pub params_total: usize,
    pub params_routed_active: usize,
    pub world_hash: u64,
    pub arms: Vec<ArmReport>,
    /// The trained parameters, in `Arm::ALL` order. Kept so a campaign run can
    /// export per-question predictions for the secondary endpoints without
    /// retraining.
    pub trained: Vec<(Arm, Tensor)>,
    /// Paired (world-clustered) contrast `RLCD − supervised`, with its
    /// cluster-robust standard error. Negative favours RLCD.
    pub paired_contrast: Option<(f64, f64)>,
}

/// One per-question test/development prediction, the raw material of the
/// secondary endpoints (weighted Brier, temperature scaling, ordinal score,
/// utility/regret). The primary Brier/NLL aggregates are evaluator-side; this
/// is the exported, auditable form.
#[derive(Clone, Debug)]
pub struct QuestionPrediction {
    pub world: u32,
    pub split: Split,
    pub qid: u64,
    /// `noul`, `choice` or `score` — the typed head's kind.
    pub kind: String,
    /// The `QuestionKind` needed to rebuild the utility table.
    pub question_kind: QuestionKind,
    pub label: usize,
    pub truth: Vec<f64>,
    pub probs: Vec<f64>,
}

/// Per-question marginals of `theta` on `samples`.
pub fn predictions(
    theta: &Tensor,
    config: &Edge0Config,
    samples: &[Sample],
) -> Result<Vec<QuestionPrediction>> {
    let mut out = Vec::with_capacity(samples.len());
    for s in samples {
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let out_model = forward(theta, config, &input)?;
        let probs = out_model.per_question[0].marginal_vec()?;
        out.push(QuestionPrediction {
            world: s.world,
            split: s.split,
            qid: s.qid,
            kind: match s.question.kind {
                QuestionKind::Noul => "noul",
                QuestionKind::Choice { .. } => "choice",
                QuestionKind::Score { .. } => "score",
            }
            .into(),
            question_kind: s.question.kind,
            label: s.label,
            truth: s.truth.clone(),
            probs,
        });
    }
    Ok(out)
}

/// Build the campaign's labelled samples. World ids are allocated before
/// question expansion, and every label is drawn from the world's own law.
pub fn build_samples(config: &CampaignConfig) -> Vec<Sample> {
    let [tr, dv, cal, te] = config.worlds;
    let base = config.data_seed.unwrap_or(config.seed);
    let manifest = WorldManifest::new(base, tr, dv, cal, te);
    manifest.assert_disjoint();
    let mut out = Vec::new();
    for spec in manifest.worlds() {
        let world = World::new(*spec, manifest.seed());
        let truth_h = &world.hypotheses[world.truth];
        // A deterministic observation mask: 0..=FEATURES revealed features,
        // so some worlds are ambiguous and some fully observed.
        let revealed = (spec.id as usize + spec.family as usize) % 5;
        let mut state = vec![0u32];
        for i in 0..revealed.min(FEATURES) {
            let bit = if truth_h.features[i] { 1u32 } else { 0 };
            state.push(1 + (i as u32) * 2 + bit);
        }
        let kinds = [
            QuestionKind::Noul,
            QuestionKind::Choice { options: 3 },
            QuestionKind::Score { levels: 3 },
        ];
        for (qi, kind) in kinds.iter().enumerate() {
            let oracle = world.question(*kind, qi as u32).unwrap();
            let truth = world.true_law(&oracle);
            // Sample a label from the world's law with a data stream that is
            // independent of the path-policy stream.
            let streams = Streams::derive(base);
            let mut rng = StdRng::seed_from_u64(
                streams.data ^ (spec.id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ qi as u64,
            );
            let u: f64 = rng.random();
            let mut acc = 0.0;
            let mut label = truth.len() - 1;
            for (i, p) in truth.iter().enumerate() {
                acc += p;
                if u <= acc {
                    label = i;
                    break;
                }
            }
            let question = QuestionInput {
                kind: *kind,
                tokens: vec![12 + qi as u32],
                options: if let QuestionKind::Choice { options } = kind {
                    (0..*options).map(|o| 20 + o as u32).collect()
                } else {
                    vec![]
                },
            };
            out.push(Sample {
                world: spec.id,
                split: spec.split,
                qid: qi as u64,
                input_state: state.clone(),
                question,
                label,
                truth,
            });
        }
    }
    out
}

fn mean_loss(parts: &[Tensor]) -> Result<Tensor> {
    let mut acc = parts[0].clone();
    for p in &parts[1..] {
        acc = acc.add(p)?;
    }
    acc.affine(1.0 / parts.len() as f64, 0.0)
}

fn nll_of(q: &crate::edge0_model::PathOutput, y: usize) -> Result<Tensor> {
    let m = q.marginal()?;
    let p = m.narrow(0, y, 1)?.squeeze(0)?.clamp(1e-6, 1.0)?.log()?;
    p.affine(-1.0, 0.0)
}

fn brier_of(q: &crate::edge0_model::PathOutput, y: usize) -> Result<Tensor> {
    let m = q.marginal()?;
    let k = m.dim(0)?;
    let eye = Tensor::from_vec(
        (0..k)
            .map(|i| if i == y { 1.0f32 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![k],
        m.device(),
    )?;
    m.sub(&eye)?.sqr()?.sum_all()
}

fn brier_of_frozen(q: &crate::edge0_model::PathOutput, y: usize) -> Result<Tensor> {
    let paths: Vec<&Tensor> = q.path_probs.iter().collect();
    let stacked = Tensor::stack(&paths, 0)?;
    let frozen = q.policy.detach();
    let m = frozen
        .unsqueeze(0)?
        .broadcast_matmul(&stacked)?
        .squeeze(0)?;
    let k = m.dim(0)?;
    let eye = Tensor::from_vec(
        (0..k)
            .map(|i| if i == y { 1.0f32 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![k],
        m.device(),
    )?;
    m.sub(&eye)?.sqr()?.sum_all()
}

/// `m = PATHS` conditionally independent path draws from `a_phi(. | s,q)`.
fn sample_paths(q: &crate::edge0_model::PathOutput, rng: &mut StdRng) -> Result<Vec<usize>> {
    let a = q.policy_vec()?;
    let mut tuple = Vec::with_capacity(PATHS);
    for _ in 0..PATHS {
        let mut u: f64 = rng.random();
        let mut chosen = a.len() - 1;
        for (i, p) in a.iter().enumerate() {
            if u <= *p || i == a.len() - 1 {
                chosen = i;
                break;
            }
            u -= *p;
        }
        tuple.push(chosen);
    }
    Ok(tuple)
}

fn vector_brier(p: &Tensor, y: usize) -> Result<Tensor> {
    let k = p.dim(0)?;
    let eye = Tensor::from_vec(
        (0..k)
            .map(|i| if i == y { 1.0f32 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![k],
        p.device(),
    )?;
    p.sub(&eye)?.sqr()?.sum_all()
}

/// Per-path proper score: every sampled path's own readout is scored against
/// the label and averaged. The policy is not differentiated.
fn per_path_loss(q: &crate::edge0_model::PathOutput, y: usize, rng: &mut StdRng) -> Result<Tensor> {
    let tuple = sample_paths(q, rng)?;
    let mut losses = Vec::with_capacity(tuple.len());
    for r in tuple {
        losses.push(vector_brier(&q.path_probs[r], y)?);
    }
    mean_loss(&losses)
}

/// Correctness-reward REINFORCE: `-R Σ_i log a_phi(r_i)` with
/// `R_i = 1[argmax u_theta(.|r_i) = y]`. The reward is read off the readout
/// and is **not** a proper scoring rule; this arm exists to show what the RL
/// term does when the reward is correctness rather than a proper score.
fn correctness_reward_loss(
    q: &crate::edge0_model::PathOutput,
    y: usize,
    rng: &mut StdRng,
) -> Result<Tensor> {
    let tuple = sample_paths(q, rng)?;
    let logp = candle_nn::ops::log_softmax(&q.policy_logits, 0)?;
    let mut total = q.policy_logits.sum_all()?.affine(0.0, 0.0)?;
    for &r in &tuple {
        let vals = q.path_probs[r].to_vec1::<f32>()?;
        let mut best = 0usize;
        for (i, v) in vals.iter().enumerate() {
            if *v > vals[best] {
                best = i;
            }
        }
        if best == y {
            let term = logp
                .narrow(0, r, 1)?
                .squeeze(0)?
                .affine(-1.0 / PATHS as f64, 0.0)?;
            total = total.add(&term)?;
        }
    }
    Ok(total)
}

/// RLCD's sampled-path loss for one example.
fn rlcd_loss(q: &crate::edge0_model::PathOutput, y: usize, rng: &mut StdRng) -> Result<Tensor> {
    let tuple = sample_paths(q, rng)?;
    let d = q.policy.device();
    let mut sum_y = Tensor::zeros(1, DType::F32, d)?.squeeze(0)?;
    for &r in &tuple {
        let v = q.path_probs[r].narrow(0, y, 1)?.squeeze(0)?;
        sum_y = sum_y.add(&v)?;
    }
    let mut b_u = Tensor::ones(1, DType::F32, d)?.squeeze(0)?;
    b_u = b_u.add(&sum_y.affine(-2.0 / PATHS as f64, 0.0)?)?;
    let mut pairwise = Tensor::zeros(1, DType::F32, d)?.squeeze(0)?;
    for i in 0..PATHS {
        for j in 0..PATHS {
            if i == j {
                continue;
            }
            let dot = q.path_probs[tuple[i]]
                .mul(&q.path_probs[tuple[j]])?
                .sum_all()?;
            pairwise = pairwise.add(&dot)?;
        }
    }
    b_u = b_u.add(&pairwise.affine(1.0 / (PATHS * (PATHS - 1)) as f64, 0.0)?)?;
    let logp = candle_nn::ops::log_softmax(&q.policy_logits, 0)?;
    let mut score = Tensor::zeros(1, DType::F32, d)?.squeeze(0)?;
    for &r in &tuple {
        score = score.add(&logp.narrow(0, r, 1)?.squeeze(0)?)?;
    }
    b_u.detach().mul(&score)?.add(&b_u)
}

fn batch_gradient(
    theta: &Tensor,
    config: &Edge0Config,
    samples: &[&Sample],
    arm: Arm,
    rng: &mut StdRng,
) -> Result<Option<Tensor>> {
    let var = Var::from_tensor(theta)?;
    let mut losses = Vec::with_capacity(samples.len());
    for s in samples {
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let out = forward(var.as_tensor(), config, &input)?;
        let q = &out.per_question[0];
        let loss = match arm {
            Arm::Supervised => nll_of(q, s.label)?.add(&brier_of(q, s.label)?)?,
            Arm::ExactMarginal => brier_of(q, s.label)?,
            Arm::Rlcd => rlcd_loss(q, s.label, rng)?,
            Arm::FrozenPolicy => brier_of_frozen(q, s.label)?,
            Arm::PerPath => per_path_loss(q, s.label, rng)?,
            Arm::CorrectnessReward => correctness_reward_loss(q, s.label, rng)?,
        };
        losses.push(loss);
    }
    let loss = mean_loss(&losses)?;
    let grads = loss.backward()?;
    Ok(grads.get(var.as_tensor()).cloned())
}

fn clipped_step(
    theta: &Tensor,
    grad: &Tensor,
    velocity: &mut Tensor,
    lr: f64,
    momentum: f64,
) -> Result<Tensor> {
    let norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
    let g = if norm.is_finite() && norm > 5.0 {
        grad.affine(5.0 / norm, 0.0)?
    } else {
        grad.clone()
    };
    *velocity = velocity.affine(momentum, 0.0)?.sub(&g.affine(lr, 0.0)?)?;
    theta.add(velocity)
}

/// Train one arm. Returns the parameters and the number of forward/backward
/// passes actually executed.
fn train_arm(
    theta0: &Tensor,
    config: &Edge0Config,
    samples: &[&Sample],
    arm: Arm,
    updates: usize,
    cfg: &CampaignConfig,
    rng: &mut StdRng,
) -> Result<(Tensor, usize)> {
    let mut theta = theta0.clone();
    let mut velocity = Tensor::zeros(theta.dims(), DType::F32, theta.device())?;
    let mut passes = 0usize;
    let mut worlds: Vec<u32> = Vec::new();
    if cfg.batch_unit == BatchUnit::Worlds {
        for s in samples {
            if !worlds.contains(&s.world) {
                worlds.push(s.world);
            }
        }
    }
    for step in 0..updates {
        let batch: Vec<&Sample> = match cfg.batch_unit {
            BatchUnit::Questions => {
                let start = (step * cfg.batch_worlds) % samples.len();
                (0..cfg.batch_worlds)
                    .map(|i| samples[(start + i) % samples.len()])
                    .collect()
            }
            BatchUnit::Worlds => {
                let start = (step * cfg.batch_worlds) % worlds.len();
                let picked: Vec<u32> = (0..cfg.batch_worlds.min(worlds.len()))
                    .map(|i| worlds[(start + i) % worlds.len()])
                    .collect();
                samples
                    .iter()
                    .filter(|s| picked.contains(&s.world))
                    .copied()
                    .collect()
            }
        };
        if let Some(grad) = batch_gradient(&theta, config, &batch, arm, rng)? {
            passes += batch.len();
            theta = clipped_step(&theta, &grad, &mut velocity, cfg.lr, cfg.momentum)?;
        }
    }
    Ok((theta, passes))
}

fn evaluate(
    theta: &Tensor,
    config: &Edge0Config,
    samples: &[Sample],
) -> Result<(f64, f64, f64, f64)> {
    let (mut excess, mut nll, mut policy_entropy, mut path_entropy) = (0.0, 0.0, 0.0, 0.0);
    for s in samples {
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let out = forward(theta, config, &input)?;
        let q = &out.per_question[0];
        let p = q.marginal_vec()?;
        excess += brier_excess(&p, &s.truth);
        nll += -(p[s.label].max(1e-12)).ln();
        let pol = q.policy_vec()?;
        policy_entropy += -pol
            .iter()
            .filter(|x| **x > 0.0)
            .map(|x| x * x.ln())
            .sum::<f64>();
        let probs = q.path_probs_vec()?;
        path_entropy += probs
            .iter()
            .map(|law| {
                -law.iter()
                    .filter(|x| **x > 0.0)
                    .map(|x| x * x.ln())
                    .sum::<f64>()
            })
            .sum::<f64>()
            / probs.len() as f64;
    }
    let n = samples.len().max(1) as f64;
    Ok((excess / n, nll / n, policy_entropy / n, path_entropy / n))
}

/// Mean Brier excess per world — the clustering unit for the paired contrast.
fn per_world_excess(
    theta: &Tensor,
    config: &Edge0Config,
    samples: &[Sample],
) -> Result<std::collections::BTreeMap<u32, f64>> {
    let mut sums: std::collections::BTreeMap<u32, (f64, usize)> = std::collections::BTreeMap::new();
    for s in samples {
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let out = forward(theta, config, &input)?;
        let p = out.per_question[0].marginal_vec()?;
        let entry = sums.entry(s.world).or_insert((0.0, 0));
        entry.0 += brier_excess(&p, &s.truth);
        entry.1 += 1;
    }
    Ok(sums
        .into_iter()
        .map(|(w, (sum, n))| (w, sum / n.max(1) as f64))
        .collect())
}

/// Run the reduced-budget campaign. Every printed number is a mechanics
/// result, not an R9 promotion.
pub fn run(cfg: &CampaignConfig) -> Result<CampaignReport> {
    let all = build_samples(cfg);
    let train: Vec<&Sample> = all.iter().filter(|s| s.split == Split::Train).collect();
    let test: Vec<&Sample> = all.iter().filter(|s| s.split == Split::Test).collect();
    if train.is_empty() || test.is_empty() {
        return Err(candle_core::Error::Msg(
            "both train and test splits need at least one sample".into(),
        ));
    }
    let base = cfg.data_seed.unwrap_or(cfg.seed);
    let manifest = WorldManifest::new(
        base,
        cfg.worlds[0],
        cfg.worlds[1],
        cfg.worlds[2],
        cfg.worlds[3],
    );
    let census = census(&cfg.model);
    let init_theta = init(&cfg.model, cfg.seed);
    let streams = Streams::derive(base);

    // Optional shared supervised warm start.
    let warm = if cfg.warm_start > 0 {
        let mut rng = StdRng::seed_from_u64(streams.init ^ 0xA5A5);
        train_arm(
            &init_theta,
            &cfg.model,
            &train,
            Arm::Supervised,
            cfg.warm_start,
            cfg,
            &mut rng,
        )?
        .0
    } else {
        init_theta
    };

    let test_refs: Vec<Sample> = test.iter().map(|s| (*s).clone()).collect();
    let mut arms = Vec::new();
    let mut trained: Vec<(Arm, Tensor)> = Vec::new();
    for arm in Arm::ALL {
        let mut rng = StdRng::seed_from_u64(streams.path ^ arm as u64);
        let (theta, passes) =
            train_arm(&warm, &cfg.model, &train, arm, cfg.updates, cfg, &mut rng)?;
        let (excess, nll, policy_entropy, path_entropy) = evaluate(&theta, &cfg.model, &test_refs)?;
        trained.push((arm, theta.clone()));
        arms.push(ArmReport {
            arm,
            brier_excess: excess,
            nll,
            policy_entropy,
            mean_path_entropy: path_entropy,
            updates: cfg.updates,
            passes,
        });
    }

    // Paired, world-clustered contrast of RLCD against the supervised arm:
    // the mean of per-world paired differences, with a cluster-robust SE.
    let paired_contrast = {
        let find = |arm: Arm| trained.iter().find(|(a, _)| *a == arm).map(|(_, t)| t);
        match (find(Arm::Rlcd), find(Arm::Supervised)) {
            (Some(r), Some(s)) => {
                let rw = per_world_excess(r, &cfg.model, &test_refs)?;
                let sw = per_world_excess(s, &cfg.model, &test_refs)?;
                let diffs: Vec<f64> = rw
                    .iter()
                    .filter_map(|(w, r)| sw.get(w).map(|s| r - s))
                    .collect();
                if diffs.len() > 1 {
                    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
                    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>()
                        / (diffs.len() - 1) as f64;
                    Some((mean, (var / diffs.len() as f64).sqrt()))
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    Ok(CampaignReport {
        config: cfg.clone(),
        params_total: census.total,
        params_routed_active: census.routed_active,
        world_hash: manifest.content_hash(),
        arms,
        trained,
        paired_contrast,
    })
}
