//! R14/P5 — return-aware satisficing for the hit-and-run tier.
//!
//! Plain satisficing hit-and-run retained the loss region but lost return
//! (`−2.4050 ± 0.9367` nominal vs the knee ensemble).  This campaign adds
//! `κ_R ψ(R_floor − R̂(θ))` to the same satisficing potential, where `R̂` is the
//! differentiable one-step return surrogate on corpus states, and compares:
//!
//! * knee-stopped ensemble (return reference, `M = 3`),
//! * satisficing pCNL on the return-aware target,
//! * satisficing hit-and-run on the **plain** target,
//! * satisficing hit-and-run on the **return-aware** target.
//!
//! `R_floor` is fixed from the knee ensemble's surrogate on the same corpus
//! plus a 0.05 margin.  The objective changes the target, so every arm is
//! labelled; the report includes both the surrogate and sampled rollouts.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_return_aware
//! ```

use pcnl::ensemble::{KneeRule, WarmEntryConfig, warm_entry};
use pcnl::mdp::{BehaviourPolicy, Corpus, MdpEnv};
use pcnl::rl_track::{ReturnAwareConfig, RlLoss, RlReturnTarget, action_distribution_diversity};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::vector_model::{
    HitRunConfig, VectorSamplerConfig, run_chain_with_covariances, run_hit_and_run,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

fn features(x: [f32; 2]) -> Vec<f64> {
    vec![f64::from(x[0]), f64::from(x[1]), 1.0]
}

fn prior_draw(dim: usize, sigma: f64, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim)
        .map(|_| {
            let z: f64 = rng.sample(StandardNormal);
            (sigma * z) as f32
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let m = mean(xs);
    if xs.len() < 2 {
        return (m, f64::NAN);
    }
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    (m, (var / xs.len() as f64).sqrt())
}

fn probs(loss: &RlLoss, theta: &[f32], x: [f32; 2]) -> Vec<f64> {
    loss.probs(theta, &features(x)).expect("probs")
}

fn ensemble_policy(loss: &RlLoss, members: &[Vec<f32>], x: [f32; 2]) -> Vec<f64> {
    let ps: Vec<Vec<f64>> = members.iter().map(|w| probs(loss, w, x)).collect();
    pcnl::distill::ensemble_probs(&ps).expect("ensemble")
}

fn rollout_returns<F>(env: &MdpEnv, policy: F, episodes: usize, seed: u64) -> Vec<f64>
where
    F: Fn([f32; 2]) -> Vec<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(episodes);
    for _ in 0..episodes {
        let mut x = [0.0f32; 2];
        for slot in x.iter_mut() {
            let z: f32 = rng.sample(StandardNormal);
            *slot = z.clamp(-0.9 * env.x_max, 0.9 * env.x_max);
        }
        let mut total = 0.0;
        for _ in 0..env.horizon {
            let p = policy(x);
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
            x = env.step(x, action, [0.0, 0.0]);
        }
        out.push(total);
    }
    out
}

fn train_loss(loss: &RlLoss, theta: &[f32]) -> f64 {
    loss.loss(theta).unwrap_or(f64::INFINITY)
}

fn main() {
    let env = MdpEnv {
        horizon: 8,
        ..MdpEnv::default()
    };
    let behavior = BehaviourPolicy {
        feedback: 0.5,
        epsilon: 0.2,
        seed: 17,
    };
    let (train, _dev, _test) = Corpus::splits(&env, behavior, (6, 4, 4), 5000);
    let loss = RlLoss::from_corpus(&train, 0.05, 20.0).expect("rl loss");
    let tau = 1.2;
    let config = SatisficingConfig {
        tau,
        kappa: 200.0,
        alpha: 0.0,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let strict = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config
    };

    // Cached corpus states and the env's per-action rewards for the surrogate.
    let mut states: Vec<Vec<f64>> = Vec::new();
    let mut rewards: Vec<Vec<f64>> = Vec::new();
    for traj in &train.trajectories {
        for x in &traj.states {
            states.push(features(*x));
            rewards.push(
                (0..loss.classes())
                    .map(|a| f64::from(env.reward(*x, a)))
                    .collect(),
            );
        }
    }

    let families = [401u64, 502, 603];
    let m = 3usize;
    let warm_cfg = WarmEntryConfig {
        max_steps: 400,
        lr: 0.05,
        tau,
        knee: KneeRule::default(),
    };
    let chain_cfg = VectorSamplerConfig {
        steps: 300,
        burn: 60,
        thin: 12,
        rho: 0.9,
        refresh: 0.01,
        prior_sigma: 1.0,
        draws: 20,
    };
    let hit_cfg = HitRunConfig {
        moves: 160,
        burn: 40,
        thin: 4,
        draws: 30,
        width: 1.0,
        max_expand: 32,
        max_shrink: 64,
    };
    let prior_c = vec![1.0f64; loss.flat_size()];
    let arms = [
        "knee-stopped",
        "satisficing-pcnl-return",
        "satisficing-hitrun",
        "satisficing-hitrun-return",
    ];
    let mut in_region: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut surrogate: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut div: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut nom: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut shift: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut paired_ret: Vec<f64> = Vec::new();
    let mut paired_ret_shift: Vec<f64> = Vec::new();
    let mut paired_knee: Vec<f64> = Vec::new();
    let mut floor_rows: Vec<f64> = Vec::new();
    let tau_margin = std::env::var("R14_TAU_MARGIN")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.10);
    let kappa_return = std::env::var("R14_KAPPA_RETURN")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(15_000.0);
    let floor_margin = std::env::var("R14_FLOOR_MARGIN")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.03);
    let episodes = 200usize;
    let mut shifted = env.clone();
    shifted.a_mat[0][1] = 0.25;
    shifted.noise = 0.1;

    println!(
        "r14=p5-return-aware seeds={} members={m} dim={} tau={tau} tau_return_arm={} kappa_return={kappa_return} floor_margin={floor_margin} objective=kappa*hinge(L-tau)+alpha*L+kappa_R*hinge(R_floor-Rhat)",
        families.len(),
        loss.flat_size(),
        tau - tau_margin
    );

    for &family in &families {
        let seeds: Vec<u64> = (0..m as u64).map(|k| family + 11 * k).collect();
        let starts: Vec<Vec<f32>> = seeds
            .iter()
            .map(|&seed| prior_draw(loss.flat_size(), 1.0, seed))
            .collect();
        let mut knee = Vec::new();
        for s in &starts {
            knee.push(warm_entry(&loss, s, warm_cfg).expect("warm").theta);
        }
        // The floor is fixed from the knee ensemble's surrogate + 0.05.
        let probe_cfg = ReturnAwareConfig {
            tau,
            kappa: config.kappa,
            alpha: config.alpha,
            return_floor: 0.0,
            kappa_return: 0.0,
        };
        let probe =
            RlReturnTarget::new(loss.clone(), states.clone(), rewards.clone(), probe_cfg).unwrap();
        let knee_ret = mean(
            &knee
                .iter()
                .map(|w| probe.return_surrogate(w).unwrap())
                .collect::<Vec<_>>(),
        );
        let floor = knee_ret + floor_margin;
        floor_rows.push(floor);
        let return_cfg = ReturnAwareConfig {
            tau: (tau - tau_margin).max(0.0),
            return_floor: floor,
            kappa_return,
            ..probe_cfg
        };
        let target =
            RlReturnTarget::new(loss.clone(), states.clone(), rewards.clone(), return_cfg).unwrap();
        let target_sat = Satisficing::new(&target, config, "rl-return", "mean").expect("sat");
        let target_strict = Satisficing::new(&target, strict, "rl-return", "mean").expect("strict");

        let mut pcnl_ret = Vec::new();
        let mut pcnl_accept = (0usize, 0usize);
        for k in 0..m {
            let chain = run_chain_with_covariances(
                &target_sat,
                &prior_c,
                &prior_c,
                chain_cfg,
                knee[k].clone(),
                seeds[k] + 700,
            )
            .expect("pcnl");
            pcnl_accept.0 += chain.accepted;
            pcnl_accept.1 += chain.proposals;
            pcnl_ret.push(chain.theta);
        }
        let mut hit_plain = Vec::new();
        for k in 0..m {
            let chain = run_hit_and_run(
                &sat_plain(&loss, strict),
                &prior_c,
                hit_cfg.clone(),
                knee[k].clone(),
                seeds[k] + 800,
            )
            .expect("hitrun");
            hit_plain.push(chain.theta);
        }
        let mut hit_ret = Vec::new();
        for k in 0..m {
            let chain = run_hit_and_run(
                &target_strict,
                &prior_c,
                hit_cfg.clone(),
                knee[k].clone(),
                seeds[k] + 900,
            )
            .expect("hitrun return");
            hit_ret.push(chain.theta);
        }

        println!(
            "seed={family} pcnl_accept={}/{}",
            pcnl_accept.0, pcnl_accept.1
        );
        let members: [(&str, &[Vec<f32>]); 4] = [
            (arms[0], &knee),
            (arms[1], &pcnl_ret),
            (arms[2], &hit_plain),
            (arms[3], &hit_ret),
        ];
        let mut nominal_vecs: Vec<Vec<f64>> = Vec::new();
        let mut shifted_vecs: Vec<Vec<f64>> = Vec::new();
        for (index, (name, ms)) in members.iter().enumerate() {
            let losses: Vec<f64> = ms.iter().map(|w| train_loss(&loss, w)).collect();
            let inr = losses.iter().filter(|l| **l <= tau).count() as f64 / ms.len() as f64;
            let surr = mean(
                &ms.iter()
                    .map(|w| target.return_surrogate(w).unwrap())
                    .collect::<Vec<_>>(),
            );
            let policies: Vec<Vec<Vec<f64>>> = ms
                .iter()
                .map(|w| {
                    (0..9)
                        .flat_map(|i| {
                            (0..9).map(move |j| [-1.4 + 0.35 * i as f32, -1.4 + 0.35 * j as f32])
                        })
                        .map(|x| probs(&loss, w, x))
                        .collect()
                })
                .collect();
            let (tv, _) = action_distribution_diversity(&policies).unwrap_or((0.0, 0.0));
            let policy = {
                let ms = ms.to_vec();
                let lref = &loss;
                move |x: [f32; 2]| ensemble_policy(lref, &ms, x)
            };
            let nominal = rollout_returns(&env, &policy, episodes, 909);
            let shifted_returns = rollout_returns(&shifted, &policy, episodes, 909);
            in_region[index].push(inr);
            surrogate[index].push(surr);
            div[index].push(tv);
            nom[index].push(mean(&nominal));
            shift[index].push(mean(&shifted_returns));
            println!(
                "seed={family} arm={name} in_region={inr:.3} surrogate={surr:.5} diversity_tv={tv:.4} nominal_return={:.4} nominal_se={:.6} shifted_return={:.4} shifted_se={:.6}",
                mean(&nominal),
                mean_se(&nominal).1,
                mean(&shifted_returns),
                mean_se(&shifted_returns).1
            );
            nominal_vecs.push(nominal);
            shifted_vecs.push(shifted_returns);
        }
        let diffs = |a: usize, b: usize| -> (f64, f64, f64) {
            let dn: Vec<f64> = nominal_vecs[a]
                .iter()
                .zip(&nominal_vecs[b])
                .map(|(x, y)| x - y)
                .collect();
            let ds: Vec<f64> = shifted_vecs[a]
                .iter()
                .zip(&shifted_vecs[b])
                .map(|(x, y)| x - y)
                .collect();
            (mean(&dn), mean(&ds), mean_se(&dn).1)
        };
        let (ret_n, ret_s, ret_se) = diffs(3, 2);
        paired_ret.push(ret_n);
        paired_ret_shift.push(ret_s);
        let (knee_n, _, knee_se) = diffs(3, 0);
        paired_knee.push(knee_n);
        println!(
            "seed={family} paired hitrun_return_minus_hitrun nominal={ret_n:+.4} se={ret_se:.4} shifted={ret_s:+.4} hitrun_return_minus_knee nominal={knee_n:+.4} se={knee_se:.4}"
        );
    }

    for (index, name) in arms.iter().enumerate() {
        let (im, _) = mean_se(&in_region[index]);
        let (sm, _) = mean_se(&surrogate[index]);
        let (dm, _) = mean_se(&div[index]);
        let (nm, nse) = mean_se(&nom[index]);
        let (hm, hse) = mean_se(&shift[index]);
        println!(
            "aggregate arm={name} in_region_mean={im:.3} surrogate_mean={sm:.5} diversity_tv_mean={dm:.4} nominal_return_mean={nm:.4} nominal_se_across_seeds={nse:.4} shifted_return_mean={hm:.4} shifted_se_across_seeds={hse:.4}"
        );
    }
    let (m1, s1) = mean_se(&paired_ret);
    let (m2, s2) = mean_se(&paired_ret_shift);
    let (m3, s3) = mean_se(&paired_knee);
    let (fm, _) = mean_se(&floor_rows);
    println!(
        "paired hitrun_return_vs_hitrun nominal_delta={m1:+.4} se_across_seeds={s1:.4} shifted_delta={m2:+.4} se={s2:.4}"
    );
    println!("paired hitrun_return_vs_knee nominal_delta={m3:+.4} se_across_seeds={s3:.4}");
    println!(
        "return_floor_mean={fm:.5} objective=return-aware target (changes pi_tau_kappa_alpha)"
    );
}

fn sat_plain<'a>(loss: &'a RlLoss, config: SatisficingConfig) -> Satisficing<&'a RlLoss> {
    Satisficing::new(loss, config, "rl-weight-nll", "mean").expect("plain sat")
}
