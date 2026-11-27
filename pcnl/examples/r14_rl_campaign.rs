//! R14/P5 (reduced-budget pilot) — non-RLCD RL training-loss ensemble and
//! policy distillation.
//!
//! The training loss is [`RlLoss`](pcnl::rl_track::RlLoss): advantage-weighted
//! behaviour-regularized offline policy NLL on the synthetic MDP's seeded
//! behaviour corpus.  There is no Edge0 latent mixture and no four-path
//! REINFORCE Brier U-statistic.
//!
//! Reported separately: (1) RL training-loss satisfaction, (2) action
//! diversity, (3) teacher–student KL/TV with the Pinsker/finite-horizon return
//! bound, and (4) rollout return on nominal and shifted dynamics for the
//! radical / knee / SWA / satisficing / distilled / label-trained / prior-only
//! arms.  Reduced budget, one split, one seed family — a mechanics pilot, not
//! the declared campaign.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_rl_campaign
//! ```

use pcnl::distill::{SoftmaxStudent, ensemble_probs, fit_softmax_student};
use pcnl::ensemble::{KneeRule, WarmEntryConfig, warm_entry};
use pcnl::ensemble_diag::{kl_divergence, pinsker_return_bound, rmse, tv_distance};
use pcnl::mdp::{BehaviourPolicy, Corpus, MdpEnv};
use pcnl::rl_track::{RlLoss, action_distribution_diversity, linear_policy_probs, rollout_return};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::swa::{SwaCheckpoint, SwaKind};
use pcnl::vector_model::{VectorSamplerConfig, run_chain_with_covariances};
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

fn probs(loss: &RlLoss, theta: &[f32], x: [f32; 2]) -> Vec<f64> {
    loss.probs(theta, &features(x)).expect("policy probs")
}

fn ensemble_policy(loss: &RlLoss, members: &[Vec<f32>], x: [f32; 2]) -> Vec<f64> {
    let ps: Vec<Vec<f64>> = members.iter().map(|w| probs(loss, w, x)).collect();
    ensemble_probs(&ps).expect("ensemble probs")
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn se(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return f64::NAN;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    (var / xs.len() as f64).sqrt()
}

/// One rollout where the ensemble member is drawn once per episode.
fn rollout_episode_draw(
    env: &MdpEnv,
    loss: &RlLoss,
    members: &[Vec<f32>],
    episodes: usize,
    seed: u64,
) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut returns = Vec::with_capacity(episodes);
    for _ in 0..episodes {
        let member = rng.random_range(0..members.len());
        let mut x = [0.0f32; 2];
        for slot in x.iter_mut() {
            let z: f32 = rng.sample(StandardNormal);
            *slot = z.clamp(-0.9 * env.x_max, 0.9 * env.x_max);
        }
        let mut total = 0.0;
        for _ in 0..env.horizon {
            let p = probs(loss, &members[member], x);
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
        returns.push(total);
    }
    (mean(&returns), se(&returns))
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
    let (train, dev, test) = Corpus::splits(&env, behavior, (6, 4, 4), 5000);
    let loss = RlLoss::from_corpus(&train, 0.05, 20.0).expect("rl loss");
    let config = SatisficingConfig {
        tau: 1.2,
        kappa: 200.0,
        alpha: 0.0,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let sat = Satisficing::new(&loss, config, "rl-weight-nll", "mean").expect("config");
    let prior_c = vec![1.0f64; loss.flat_size()];
    println!(
        "r14=pilot P5 env=horizon{} actions{} corpus train={} dev={} test={} loss_dim={} beta={} tau={} kappa={} alpha={} hinge={}",
        env.horizon,
        env.actions.len(),
        train.transitions(),
        dev.transitions(),
        test.transitions(),
        loss.flat_size(),
        loss.beta(),
        config.tau,
        config.kappa,
        config.alpha,
        config.hinge.label()
    );
    println!("target_identity={}", sat.identity(&prior_c));

    let seeds = [71u64, 72, 73, 74];
    let starts: Vec<Vec<f32>> = seeds
        .iter()
        .map(|&seed| prior_draw(loss.flat_size(), 1.0, seed))
        .collect();
    let warm_cfg = WarmEntryConfig {
        max_steps: 400,
        lr: 0.05,
        tau: config.tau,
        knee: KneeRule::default(),
    };

    // Radical single point and deep ensemble.
    let mut radical_deep = Vec::new();
    let mut radical_trace = Vec::new();
    for (i, s) in starts.iter().enumerate() {
        let warm = warm_entry(
            &loss,
            s,
            WarmEntryConfig {
                max_steps: 600,
                lr: 0.05,
                tau: 1e-9,
                ..warm_cfg
            },
        )
        .expect("radical warm entry");
        radical_deep.push(warm.theta);
        if i == 0 {
            radical_trace = warm.trace;
        }
    }
    let radical_single = radical_deep[0].clone();

    // Knee-stopped ensemble and a knee + constant-LR SGD approximation.
    let mut knee = Vec::new();
    let mut knee_stops = Vec::new();
    let mut knee_steps = Vec::new();
    for s in starts.iter() {
        let warm = warm_entry(&loss, s, warm_cfg).expect("warm entry");
        knee_stops.push(warm.stop.label());
        knee_steps.push(warm.steps);
        knee.push(warm.theta);
    }

    // Satisficing pCNL inside Φ, plus within-basin SWA.
    let chain_cfg = VectorSamplerConfig {
        steps: 300,
        burn: 60,
        thin: 12,
        rho: 0.9,
        refresh: 0.01,
        prior_sigma: 1.0,
        draws: 20,
    };
    let mut sat_members = Vec::new();
    let mut sat_accept = (0usize, 0usize);
    let mut swa = Vec::new();
    for (i, _) in starts.iter().enumerate() {
        let chain = run_chain_with_covariances(
            &sat,
            &prior_c,
            &prior_c,
            chain_cfg,
            knee[i].clone(),
            seeds[i] + 300,
        )
        .expect("satisficing chain");
        sat_accept.0 += chain.accepted;
        sat_accept.1 += chain.proposals;
        sat_members.push(chain.theta.clone());
        if chain.draws.len() >= 10 {
            swa.push(
                SwaCheckpoint::new(
                    SwaKind::PcnlWithinBasin,
                    i,
                    10,
                    12,
                    &chain.draws[chain.draws.len() - 10..],
                )
                .expect("swa checkpoint"),
            );
        }
    }

    // Distilled student and label/behaviour-trained control.
    let states: Vec<[f32; 2]> = (0..7)
        .flat_map(|i| (0..7).map(move |j| [-1.2 + 0.4 * i as f32, -1.2 + 0.4 * j as f32]))
        .collect();
    let distill_data: Vec<(Vec<f64>, Vec<f64>)> = states
        .iter()
        .map(|&x| (features(x), ensemble_policy(&loss, &sat_members, x)))
        .collect();
    let student_record =
        fit_softmax_student(3, loss.classes(), &distill_data, 600, 0.1).expect("distilled student");
    let mut label_data = Vec::new();
    for traj in &train.trajectories {
        for t in 0..traj.len() {
            let mut onehot = vec![0.0; loss.classes()];
            onehot[traj.actions[t]] = 1.0;
            label_data.push((features(traj.states[t]), onehot));
        }
    }
    let control_record =
        fit_softmax_student(3, loss.classes(), &label_data, 600, 0.1).expect("label student");
    let student = student_record.student.clone();
    let control = control_record.student.clone();

    // Volumes with the same reading in all arms: nominal + shifted dynamics.
    let mut shifted = env.clone();
    shifted.a_mat[0][1] = 0.25;
    shifted.noise = 0.1;

    let nominal_states: Vec<[f32; 2]> = (0..9)
        .flat_map(|i| (0..9).map(move |j| [-1.4 + 0.35 * i as f32, -1.4 + 0.35 * j as f32]))
        .collect();

    let return_of = |policy: &dyn Fn([f32; 2]) -> Vec<f64>, e: &MdpEnv| {
        rollout_return(e, |x| policy(x), 200, 909).expect("rollout")
    };
    let member_policy = |w: Vec<f32>| {
        let loss = loss.clone();
        move |x: [f32; 2]| probs(&loss, &w, x)
    };
    let ens_policy = |members: Vec<Vec<f32>>| {
        let loss = loss.clone();
        move |x: [f32; 2]| ensemble_policy(&loss, &members, x)
    };
    let student_policy = {
        let s = student.clone();
        move |x: [f32; 2]| s.probs(&features(x)).expect("student probs")
    };
    let control_policy = {
        let s = control.clone();
        move |x: [f32; 2]| s.probs(&features(x)).expect("control probs")
    };

    let arms: Vec<(&str, Vec<Vec<f32>>, Box<dyn Fn([f32; 2]) -> Vec<f64>>)> = vec![
        (
            "radical-single",
            vec![radical_single.clone()],
            Box::new(member_policy(radical_single.clone())),
        ),
        (
            "radical-deep",
            radical_deep.clone(),
            Box::new(ens_policy(radical_deep.clone())),
        ),
        (
            "knee-stopped",
            knee.clone(),
            Box::new(ens_policy(knee.clone())),
        ),
        (
            "satisficing-pcnl",
            sat_members.clone(),
            Box::new(ens_policy(sat_members.clone())),
        ),
        (
            "prior-only",
            starts.clone(),
            Box::new(ens_policy(starts.clone())),
        ),
    ];
    let mut table: Vec<(&str, f64, f64, f64, f64)> = Vec::new();
    for (name, members, policy) in &arms {
        let losses: Vec<f64> = members
            .iter()
            .map(|w| loss.loss(w).unwrap_or(f64::NAN))
            .collect();
        let in_region =
            losses.iter().filter(|l| **l <= config.tau).count() as f64 / losses.len() as f64;
        let policies: Vec<Vec<Vec<f64>>> = members
            .iter()
            .map(|w| nominal_states.iter().map(|&x| probs(&loss, w, x)).collect())
            .collect();
        let (div_tv, div_kl) = action_distribution_diversity(&policies).unwrap_or((0.0, 0.0));
        let nominal = return_of(policy.as_ref(), &env);
        let shifted_return = return_of(policy.as_ref(), &shifted);
        println!(
            "arm={name} members={} in_region={in_region:.3} loss_mean={:.4} loss_min={:.4} loss_max={:.4} diversity_tv={div_tv:.4} diversity_kl={div_kl:.4} nominal_return={nominal:.4} shifted_return={shifted_return:.4}",
            members.len(),
            mean(&losses),
            losses.iter().copied().fold(f64::INFINITY, f64::min),
            losses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        table.push((name, in_region, div_tv, nominal, shifted_return));
    }

    // SWA arm (within-basin centers, averaged across basins only for reporting).
    let swa_weights: Vec<Vec<f32>> = swa.iter().map(|s| s.weights.clone()).collect();
    let swa_policy = ens_policy(swa_weights.clone());
    let swa_nominal = return_of(&swa_policy, &env);
    let swa_shifted = return_of(&swa_policy, &shifted);
    let swa_losses: Vec<f64> = swa_weights.iter().map(|w| loss.loss(w).unwrap()).collect();
    println!(
        "arm=sgd-swa members={} in_region={:.3} loss_mean={:.4} nominal_return={swa_nominal:.4} shifted_return={swa_shifted:.4} weight_vs_last_rmse={:.6}",
        swa_weights.len(),
        swa_losses.iter().filter(|l| **l <= config.tau).count() as f64 / swa_losses.len() as f64,
        mean(&swa_losses),
        rmse(
            &swa_weights[0]
                .iter()
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            &knee[0].iter().map(|v| f64::from(*v)).collect::<Vec<_>>()
        )
        .unwrap_or(f64::NAN)
    );
    let student_returns = (
        return_of(&student_policy, &env),
        return_of(&student_policy, &shifted),
    );
    let control_returns = (
        return_of(&control_policy, &env),
        return_of(&control_policy, &shifted),
    );
    println!(
        "arm=distilled-student in_region=0.000 nominal_return={:.4} shifted_return={:.4} final_kl={:.6}",
        student_returns.0, student_returns.1, student_record.final_kl
    );
    println!(
        "arm=label-student-control in_region=0.000 nominal_return={:.4} shifted_return={:.4} final_kl={:.6}",
        control_returns.0, control_returns.1, control_record.final_kl
    );

    // Teacher–student fidelity, nominal and shifted state sets.
    let fidelity = |student: &SoftmaxStudent| {
        let mut kls = Vec::new();
        let mut tvs = Vec::new();
        for &x in &nominal_states {
            let teacher = ensemble_policy(&loss, &sat_members, x);
            let s = student.probs(&features(x)).expect("student probs");
            kls.push(kl_divergence(&teacher, &s).unwrap());
            tvs.push(tv_distance(&teacher, &s).unwrap());
        }
        (
            mean(&kls),
            mean(&tvs),
            kls.iter().copied().fold(0.0, f64::max),
        )
    };
    let (student_kl, student_tv, student_kl_max) = fidelity(&student);
    let (control_kl, control_tv, control_kl_max) = fidelity(&control);
    let reward_span = 10.0;
    let pinsker = |kl_max: f64| {
        let tv = (kl_max / 2.0).sqrt();
        pinsker_return_bound(reward_span, env.horizon, tv).unwrap()
    };
    println!(
        "distillation student_kl={student_kl:.6} student_tv={student_tv:.6} student_kl_max={student_kl_max:.6} pinsker_return_bound={:.6} control_kl={control_kl:.6} control_tv={control_tv:.6} control_kl_max={control_kl_max:.6} control_pinsker={:.6}",
        pinsker(student_kl_max),
        pinsker(control_kl_max)
    );

    // Once-per-episode member draw versus per-step BMA.
    let (episode_mean, episode_se) = rollout_episode_draw(&env, &loss, &sat_members, 200, 4242);
    let bma = return_of(&ens_policy(sat_members.clone()), &env);
    println!(
        "ensemble_draw once_per_episode={episode_mean:.4} se={episode_se:.6} per_step_bma={bma:.4}"
    );

    // Paired return differences across episodes for the headline contrast
    // (satisficing vs knee), using a shared seed and argument order.
    let (a, b) = (
        return_of(&ens_policy(sat_members.clone()), &env),
        return_of(&ens_policy(knee.clone()), &env),
    );
    println!(
        "contrast satisficing_vs_knee_return={:+.4} satisficing_vs_radical_return={:+.4}",
        a - b,
        a - table[0].3
    );
    println!(
        "warm_entry stops={:?} steps={:?} radical_trace_first={:.4} radical_trace_last={:.4}",
        knee_stops,
        knee_steps,
        radical_trace.first().copied().unwrap_or(f64::NAN),
        radical_trace.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "satisficing_accept={}/{} swa_centers={}",
        sat_accept.0,
        sat_accept.1,
        swa.len()
    );
    let _ = linear_policy_probs(&sat_members[0], &features([0.0, 0.0]), loss.classes()).unwrap();
    println!("runtime_ok");
}
