//! R14/P5 — non-RLCD reinforcement-learning training-loss track.
//!
//! The loss is a **differentiable, nonnegative offline RL objective** on a
//! linear-softmax policy `π_θ(a|s) = softmax(W φ(s))_a`:
//!
//! ```text
//! L_RL(θ) = Σ_t w_t ·[−log π_θ(a_t|s_t)] / Σ_t w_t
//!           + β · mean_t KL(π_θ(·|s_t) ‖ π_ref(·|s_t)),
//! ```
//!
//! with nonnegative trajectory weights `w_t` built from positive
//! reward-to-go advantages, and a uniform reference policy `π_ref` (the
//! behaviour corpus's exploration floor).  `β ≥ 0` and `w_t ≥ 0` make the loss
//! strictly nonnegative, matching `Φ_{τ,κ,α}`'s requirement.  This is **not**
//! the retired RLCD track: there is no Edge0 latent mixture and no four-path
//! REINFORCE Brier U-statistic.
//!
//! The RL campaign reports four separate things, never a single blended score:
//! training-loss satisfaction, action-distribution diversity, teacher–student
//! KL/TV (with [`pinsker_return_bound`](crate::ensemble_diag::pinsker_return_bound)),
//! and trajectory-clustered rollout return.  [`rollout_return`] is the plain
//! seeded rollout used by the reduced-budget example; the trajectory-clustered
//! D5 estimator lives in [`crate::policy_readout`].

use crate::ensemble_diag::{kl_divergence, tv_distance};
use crate::mdp::Corpus;
use crate::vector_model::EnergyTarget;

/// One offline transition with its nonnegative training weight.
#[derive(Clone, Debug)]
pub struct RlTransition {
    pub features: Vec<f64>,
    pub action: usize,
    pub weight: f64,
}

/// Advantage-weighted behaviour-regularized offline RL loss.
#[derive(Clone, Debug)]
pub struct RlLoss {
    transitions: Vec<RlTransition>,
    classes: usize,
    features: usize,
    /// Uniform reference-policy KL weight.
    beta: f64,
}

impl RlLoss {
    pub fn new(
        transitions: Vec<RlTransition>,
        classes: usize,
        features: usize,
        beta: f64,
    ) -> Result<Self, String> {
        if transitions.is_empty() || classes == 0 || features == 0 {
            return Err("rl: transitions/classes/features must be non-empty".into());
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err("rl: beta must be finite and >= 0".into());
        }
        for t in &transitions {
            if t.features.len() != features {
                return Err("rl: transition feature dimension mismatch".into());
            }
            if t.action >= classes {
                return Err("rl: transition action out of range".into());
            }
            if !t.weight.is_finite() || t.weight < 0.0 {
                return Err("rl: transition weights must be finite and >= 0".into());
            }
            if t.features.iter().any(|v| !v.is_finite()) {
                return Err("rl: non-finite transition feature".into());
            }
        }
        let sum: f64 = transitions.iter().map(|t| t.weight).sum();
        if !(sum.is_finite() && sum > 0.0) {
            return Err("rl: transition weights sum to zero".into());
        }
        Ok(Self {
            transitions,
            classes,
            features,
            beta,
        })
    }

    /// Build the loss from a seeded behaviour corpus.
    ///
    /// Features are `[x₀, x₁, 1]`; weights are
    /// `exp((rtg_t − mean_rtg)/temperature)`, normalized to mean one (so they
    /// stay nonnegative and the NLL part is a proper weighted mean).  The
    /// reference policy is uniform over actions.
    pub fn from_corpus(corpus: &Corpus, beta: f64, temperature: f64) -> Result<Self, String> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("rl: temperature must be finite and > 0".into());
        }
        let mut raw: Vec<(Vec<f64>, usize, f64)> = Vec::new();
        let mut rtgs: Vec<f64> = Vec::new();
        for traj in &corpus.trajectories {
            for t in 0..traj.len() {
                let s = traj.states[t];
                raw.push((
                    vec![f64::from(s[0]), f64::from(s[1]), 1.0],
                    traj.actions[t],
                    f64::from(traj.rtgs[t]),
                ));
                rtgs.push(f64::from(traj.rtgs[t]));
            }
        }
        if raw.is_empty() {
            return Err("rl: empty corpus".into());
        }
        let mean_rtg = rtgs.iter().sum::<f64>() / rtgs.len() as f64;
        let mut weights: Vec<f64> = rtgs
            .iter()
            .map(|r| ((r - mean_rtg) / temperature).exp())
            .collect();
        // Stability: clamp extreme advantages before normalization.
        let wmax = weights.iter().copied().fold(0.0f64, f64::max);
        if !(wmax.is_finite() && wmax > 0.0) {
            return Err("rl: advantage weights are not finite".into());
        }
        for w in &mut weights {
            *w = (*w / wmax).max(1e-12);
        }
        let wmean = weights.iter().sum::<f64>() / weights.len() as f64;
        let transitions: Vec<RlTransition> = raw
            .into_iter()
            .zip(weights)
            .map(|((features, action, _), weight)| RlTransition {
                features,
                action,
                weight: weight / wmean,
            })
            .collect();
        Self::new(transitions, corpus.env.actions.len(), 3, beta)
    }

    pub fn transitions(&self) -> usize {
        self.transitions.len()
    }
    pub fn classes(&self) -> usize {
        self.classes
    }
    pub fn features(&self) -> usize {
        self.features
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
    pub fn flat_size(&self) -> usize {
        self.classes * self.features
    }

    pub fn logits(&self, theta: &[f32], features: &[f64]) -> Result<Vec<f64>, String> {
        if theta.len() != self.flat_size() {
            return Err("rl: policy dimension mismatch".into());
        }
        let mut z = vec![0.0f64; self.classes];
        for (c, zc) in z.iter_mut().enumerate() {
            let row = &theta[c * self.features..(c + 1) * self.features];
            *zc += row
                .iter()
                .zip(features)
                .map(|(w, f)| f64::from(*w) * f)
                .sum::<f64>();
        }
        Ok(z)
    }

    pub fn probs(&self, theta: &[f32], features: &[f64]) -> Result<Vec<f64>, String> {
        crate::distill::softmax(&self.logits(theta, features)?)
    }

    /// The training loss `L_RL ≥ 0` alone.
    pub fn loss(&self, theta: &[f32]) -> Option<f64> {
        self.energy_and_grad(theta).map(|(l, _)| l)
    }
}

fn softmax_grad(logits: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let p = crate::distill::softmax(logits).ok()?;
    Some((p, logits.to_vec()))
}

impl EnergyTarget for RlLoss {
    fn dim(&self) -> usize {
        self.flat_size()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        if theta.len() != self.flat_size() || theta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut grad = vec![0.0f64; self.flat_size()];
        let total_w: f64 = self.transitions.iter().map(|t| t.weight).sum();
        if !(total_w.is_finite() && total_w > 0.0) {
            return None;
        }
        let mut nll = 0.0;
        let mut kl = 0.0;
        let log_uniform = -(self.classes as f64).ln();
        for t in &self.transitions {
            let (p, _) = softmax_grad(&self.logits(theta, &t.features).ok()?)?;
            let pa = p[t.action].max(1e-300);
            nll += t.weight * (-pa.ln()) / total_w;
            // KL(p||uniform) = Σ p ln p + ln K; ∂/∂z_j = p_j (ln p_j − ln u_j − KL).
            let kl_t = p
                .iter()
                .map(|pc| {
                    if *pc > 0.0 {
                        pc * (pc.ln() - log_uniform)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            kl += kl_t;
            let w = t.weight / total_w;
            for (c, pc) in p.iter().enumerate() {
                let target = if c == t.action { 1.0 } else { 0.0 };
                let g_nll = w * (pc - target);
                let g_kl = if self.beta > 0.0 {
                    self.beta / self.transitions.len() as f64
                        * pc
                        * (pc.max(1e-300).ln() - log_uniform - kl_t)
                } else {
                    0.0
                };
                let g = g_nll + g_kl;
                if g == 0.0 {
                    continue;
                }
                for (f, phi) in t.features.iter().enumerate() {
                    grad[c * self.features + f] += g * phi;
                }
            }
        }
        let value = nll + self.beta * kl / self.transitions.len() as f64;
        if !value.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return None;
        }
        Some((value, grad.into_iter().map(|g| g as f32).collect()))
    }
}

/// Return-aware satisficing configuration.
///
/// The potential is
///
/// ```text
/// Φ(θ) = κ ψ(L(θ) − τ) + α L(θ) + κ_R ψ(R_floor − R̂(θ)),  ψ(u)=½·max(0,u)²
/// ```
///
/// with `R̂` the **differentiable one-step return surrogate**
/// `R̂(θ) = mean_s Σ_a π_θ(a|s) r(s,a)` on cached corpus states.  It is a
/// surrogate, not the sampled rollout return; the campaign reports both.
/// Adding the return floor changes the target from the plain satisficing
/// potential, so results must be labelled as such.
#[derive(Clone, Copy, Debug)]
pub struct ReturnAwareConfig {
    pub tau: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub return_floor: f64,
    pub kappa_return: f64,
}

impl ReturnAwareConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("tau", self.tau),
            ("kappa", self.kappa),
            ("alpha", self.alpha),
            ("return_floor", self.return_floor),
            ("kappa_return", self.kappa_return),
        ] {
            if !v.is_finite() {
                return Err(format!("rl: {name} must be finite"));
            }
        }
        if self.kappa < 0.0 || self.alpha < 0.0 || self.kappa_return < 0.0 {
            return Err("rl: kappa, alpha and kappa_return must be >= 0".into());
        }
        Ok(())
    }
}

/// `RlLoss` plus a differentiable one-step return floor.
pub struct RlReturnTarget {
    loss: RlLoss,
    states: Vec<Vec<f64>>,
    rewards: Vec<Vec<f64>>,
    config: ReturnAwareConfig,
}

impl RlReturnTarget {
    pub fn new(
        loss: RlLoss,
        states: Vec<Vec<f64>>,
        rewards: Vec<Vec<f64>>,
        config: ReturnAwareConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        if states.is_empty() || states.len() != rewards.len() {
            return Err("rl: return-aware states/rewards must be non-empty and aligned".into());
        }
        for (s, r) in states.iter().zip(&rewards) {
            if s.len() != loss.features() {
                return Err("rl: return-aware feature dimension mismatch".into());
            }
            if r.len() != loss.classes() {
                return Err("rl: return-aware reward dimension mismatch".into());
            }
            if s.iter().chain(r).any(|v| !v.is_finite()) {
                return Err("rl: return-aware values must be finite".into());
            }
        }
        Ok(Self {
            loss,
            states,
            rewards,
            config,
        })
    }

    pub fn inner(&self) -> &RlLoss {
        &self.loss
    }

    pub fn config(&self) -> ReturnAwareConfig {
        self.config
    }

    /// `L_RL(θ)` alone.
    pub fn loss_value(&self, theta: &[f32]) -> Option<f64> {
        self.loss.loss(theta)
    }

    /// Differentiable one-step return surrogate `R̂(θ)`.
    pub fn return_surrogate(&self, theta: &[f32]) -> Option<f64> {
        self.return_surrogate_and_grad(theta).map(|(r, _)| r)
    }

    fn return_surrogate_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        if theta.len() != self.loss.flat_size() || theta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let features = self.loss.features();
        let n = self.states.len() as f64;
        let mut grad = vec![0.0f64; theta.len()];
        let mut total = 0.0f64;
        for (s, r) in self.states.iter().zip(&self.rewards) {
            let p = self.loss.probs(theta, s).ok()?;
            let ret: f64 = p.iter().zip(r).map(|(p, r)| p * r).sum();
            total += ret;
            for (a, _pa) in p.iter().enumerate() {
                let g = p[a] * (r[a] - ret) / n;
                if g == 0.0 {
                    continue;
                }
                for (f, phi) in s.iter().enumerate() {
                    grad[a * features + f] += g * phi;
                }
            }
        }
        if !total.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return None;
        }
        Some((total / n, grad.into_iter().map(|g| g as f32).collect()))
    }
}

impl EnergyTarget for RlReturnTarget {
    fn dim(&self) -> usize {
        self.loss.flat_size()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (loss, grad_loss) = self.loss.energy_and_grad(theta)?;
        let (ret, grad_ret) = self.return_surrogate_and_grad(theta)?;
        if !loss.is_finite() || !ret.is_finite() {
            return None;
        }
        let cfg = self.config;
        let l_ex = (loss - cfg.tau).max(0.0);
        let r_ex = (cfg.return_floor - ret).max(0.0);
        let phi =
            0.5 * cfg.kappa * l_ex * l_ex + cfg.alpha * loss + 0.5 * cfg.kappa_return * r_ex * r_ex;
        let dl = cfg.kappa * l_ex + cfg.alpha;
        let dr = -cfg.kappa_return * r_ex;
        let mut grad = vec![0.0f64; theta.len()];
        for i in 0..theta.len() {
            grad[i] = dl * f64::from(grad_loss[i]) + dr * f64::from(grad_ret[i]);
        }
        if !phi.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return None;
        }
        Some((phi, grad.into_iter().map(|g| g as f32).collect()))
    }
}

/// Mean pairwise total-variation distance of a set of policies evaluated on the
/// same states.  Zero for a collapsed ensemble; positive when members disagree.
pub fn action_distribution_diversity(policies: &[Vec<Vec<f64>>]) -> Option<(f64, f64)> {
    if policies.len() < 2 {
        return None;
    }
    let states = policies[0].len();
    if states == 0 || policies.iter().any(|p| p.len() != states) {
        return None;
    }
    let mut tv_sum = 0.0;
    let mut pairs = 0usize;
    for i in 0..policies.len() {
        for j in i + 1..policies.len() {
            for s in 0..states {
                tv_sum += tv_distance(&policies[i][s], &policies[j][s])?;
            }
            pairs += 1;
        }
    }
    if pairs == 0 {
        return None;
    }
    let mean = tv_sum / (pairs * states) as f64;
    // Also report the mean logit-space KL, which is the distillation target.
    let mut kl_sum = 0.0;
    let mut kl_n = 0usize;
    for i in 0..policies.len() {
        for j in 0..policies.len() {
            if i == j {
                continue;
            }
            for s in 0..states {
                kl_sum += kl_divergence(&policies[i][s], &policies[j][s])?;
                kl_n += 1;
            }
        }
    }
    Some((mean, kl_sum / kl_n.max(1) as f64))
}

/// Seeded rollout return of a discrete policy.  `policy` maps a state to action
/// probabilities; actions are sampled, so the number is a Monte Carlo return.
pub fn rollout_return<F>(
    env: &crate::mdp::MdpEnv,
    policy: F,
    episodes: usize,
    seed: u64,
) -> Option<f64>
where
    F: Fn([f32; 2]) -> Vec<f64>,
{
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;
    if episodes == 0 {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut total = 0.0f64;
    for _ in 0..episodes {
        let mut x = [0.0f32; 2];
        for slot in x.iter_mut() {
            let z: f32 = rng.sample(StandardNormal);
            *slot = z.clamp(-0.9 * env.x_max, 0.9 * env.x_max);
        }
        for _ in 0..env.horizon {
            let p = policy(x);
            if p.len() != env.actions.len() || p.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return None;
            }
            let u: f64 = rng.random();
            let mut acc = 0.0;
            let mut action = p.len() - 1;
            for (i, pi) in p.iter().enumerate() {
                acc += pi;
                if u <= acc {
                    action = i;
                    break;
                }
            }
            total += f64::from(env.reward(x, action));
            let noise = [0.0f32, 0.0];
            x = env.step(x, action, noise);
        }
    }
    Some(total / episodes as f64)
}

/// Policy probabilities for a flat linear-softmax parameter vector.
pub fn linear_policy_probs(
    theta: &[f32],
    features: &[f64],
    classes: usize,
) -> Result<Vec<f64>, String> {
    if classes == 0 || theta.len() % classes != 0 {
        return Err("rl: policy parameter dimension mismatch".into());
    }
    let f = theta.len() / classes;
    if features.len() != f {
        return Err("rl: feature dimension mismatch".into());
    }
    let mut z = vec![0.0f64; classes];
    for (c, zc) in z.iter_mut().enumerate() {
        *zc = theta[c * f..(c + 1) * f]
            .iter()
            .zip(features)
            .map(|(w, x)| f64::from(*w) * x)
            .sum();
    }
    crate::distill::softmax(&z)
}
