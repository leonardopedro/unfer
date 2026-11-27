//! R14/P5 — multi-seed RL policy campaign with paired intervals.
//!
//! Three seed families, `M = 3`, the non-RLCD advantage-weighted
//! behaviour-regularized policy loss on the synthetic MDP.  Every arm is
//! scored on the same nominal and shifted environments with the same episode
//! seeds, so return contrasts are paired per episode and then aggregated
//! across seeds.
//!
//! Arms: radical single/deep, knee-stopped, constant-step SWA, sharp Gibbs,
//! satisficing pCNL, the geometry-aware hit-and-run tier (`κ = 2e6`, which
//! keeps the `L ≈ τ` boundary layer thin), the distilled student, an
//! equal-budget label-trained control, and the prior-only ensemble.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_rl_multiseed
//! ```

use pcnl::distill::{ensemble_probs, fit_softmax_student};
use pcnl::ensemble::{KneeRule, WarmEntryConfig, warm_entry};
use pcnl::ensemble_diag::{kl_divergence, pinsker_return_bound, tv_distance};
use pcnl::mdp::{BehaviourPolicy, Corpus, MdpEnv};
use pcnl::rl_track::{RlLoss, action_distribution_diversity};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::swa::{SwaKind, batch_average};
use pcnl::vector_model::{
    EnergyTarget, HitRunConfig, VectorSamplerConfig, run_chain_with_covariances, run_hit_and_run,
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

fn mean_se(xs: &[f64]) -> (f64, f64) {
    (mean(xs), se(xs))
}

/// Per-episode returns so two arms can be paired episode by episode.
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

fn paired_delta(a: &[f64], b: &[f64]) -> (f64, f64) {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(a, b)| a - b).collect();
    mean_se(&diffs)
}

struct Sharp<'a> {
    inner: &'a RlLoss,
    lambda: f64,
}

impl EnergyTarget for Sharp<'_> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (u, g) = self.inner.energy_and_grad(theta)?;
        Some((
            self.lambda * u,
            g.into_iter()
                .map(|x| (self.lambda * f64::from(x)) as f32)
                .collect(),
        ))
    }
}

/// Adam on a declared nonnegative loss (used only for the SWA warm continuation).
fn adam_loss<T: EnergyTarget>(target: &T, theta0: &[f32], steps: usize, lr: f64) -> Vec<f32> {
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    let (mut m, mut v) = (vec![0.0f64; n], vec![0.0f64; n]);
    for t in 1..=steps {
        let Some((_, g)) = target.energy_and_grad(&theta) else {
            break;
        };
        let b1 = 0.9f64;
        let b2 = 0.999f64;
        let c1 = 1.0 - b1.powi(t as i32);
        let c2 = 1.0 - b2.powi(t as i32);
        for i in 0..n {
            let gi = f64::from(g[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            let step = lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
            theta[i] = (f64::from(theta[i]) - step) as f32;
        }
    }
    theta
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
    let config = SatisficingConfig {
        tau: 1.2,
        kappa: 200.0,
        alpha: 0.0,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let strict = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config
    };
    let sat = Satisficing::new(&loss, config, "rl-weight-nll", "mean").expect("config");
    let sat_strict =
        Satisficing::new(&loss, strict, "rl-weight-nll", "mean").expect("strict config");
    let prior_c = vec![1.0f64; loss.flat_size()];
    let states: Vec<[f32; 2]> = (0..9)
        .flat_map(|i| (0..9).map(move |j| [-1.4 + 0.35 * i as f32, -1.4 + 0.35 * j as f32]))
        .collect();
    let mut shifted = env.clone();
    shifted.a_mat[0][1] = 0.25;
    shifted.noise = 0.1;

    let m = 3usize;
    let families = [401u64, 502, 603];
    let episodes = 200usize;
    let warm_cfg = WarmEntryConfig {
        max_steps: 400,
        lr: 0.05,
        tau: config.tau,
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
    let hitrun_cfg = HitRunConfig {
        moves: 160,
        burn: 40,
        thin: 4,
        draws: 30,
        width: 1.0,
        max_expand: 32,
        max_shrink: 64,
    };
    let arm_names = [
        "radical-single",
        "radical-deep",
        "knee-stopped",
        "sgd-swa",
        "sharp-gibbs",
        "satisficing-pcnl",
        "satisficing-hitrun",
        "prior-only",
        "distilled-student",
        "label-control",
    ];
    let mut per_in: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_div: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_nom: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_shift: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut paired: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("hitrun_vs_knee", Vec::new(), Vec::new()),
        ("pcnl_vs_knee", Vec::new(), Vec::new()),
        ("distilled_vs_label", Vec::new(), Vec::new()),
        ("hitrun_vs_radical_deep", Vec::new(), Vec::new()),
    ];
    let mut fidelity_rows: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut movement_rows: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut episode_draw_rows: Vec<f64> = Vec::new();

    println!(
        "r14=multiseed P5 seeds={} members={} dim={} tau={} kappa_pcnl={} kappa_strict={} beta={} transitions={}",
        families.len(),
        m,
        loss.flat_size(),
        config.tau,
        config.kappa,
        strict.kappa,
        loss.beta(),
        loss.transitions()
    );
    println!(
        "target_identity={} strict_identity={}",
        sat.identity(&prior_c),
        sat_strict.identity(&prior_c)
    );

    for &family in &families {
        let seeds: Vec<u64> = (0..m as u64).map(|k| family + 11 * k).collect();
        let starts: Vec<Vec<f32>> = seeds
            .iter()
            .map(|&seed| prior_draw(loss.flat_size(), 1.0, seed))
            .collect();

        // Radical single and deep.
        let radical_single = warm_entry(
            &loss,
            &starts[0],
            WarmEntryConfig {
                max_steps: 600,
                lr: 0.05,
                tau: 1e-9,
                ..warm_cfg
            },
        )
        .expect("radical")
        .theta;
        let mut radical_deep = Vec::new();
        for s in &starts {
            radical_deep.push(
                warm_entry(
                    &loss,
                    s,
                    WarmEntryConfig {
                        max_steps: 600,
                        lr: 0.05,
                        tau: 1e-9,
                        ..warm_cfg
                    },
                )
                .expect("radical member")
                .theta,
            );
        }

        // Knee-stopped ensemble.
        let mut knee = Vec::new();
        let mut knee_stops = Vec::new();
        for s in &starts {
            let warm = warm_entry(&loss, s, warm_cfg).expect("warm entry");
            knee_stops.push(warm.stop.label());
            knee.push(warm.theta);
        }

        // Constant-step SWA on the first basin.
        let mut swa_theta = adam_loss(&loss, &knee[0], 100, 0.02);
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            swa_theta = adam_loss(&loss, &swa_theta, 10, 0.02);
            snapshots.push(swa_theta.clone());
        }
        let swa = batch_average(&snapshots).expect("swa average");
        let swa_proto = pcnl::swa::SwaCheckpoint::new(
            SwaKind::SgdConstantLr,
            0,
            snapshots.len(),
            10,
            &snapshots,
        )
        .expect("swa checkpoint");

        // Sharp Gibbs control.
        let sharp = Sharp {
            inner: &loss,
            lambda: 300.0,
        };
        let sharp_chain = run_chain_with_covariances(
            &sharp,
            &prior_c,
            &prior_c,
            VectorSamplerConfig {
                steps: 200,
                burn: 40,
                thin: 1,
                draws: 1,
                ..chain_cfg
            },
            knee[0].clone(),
            family + 900,
        )
        .expect("sharp chain");

        // pCN and geometry-aware hit-and-run ensembles.
        let mut pcnl = Vec::new();
        let mut pcnl_accept = (0usize, 0usize);
        for (i, _) in starts.iter().enumerate() {
            let chain = run_chain_with_covariances(
                &sat,
                &prior_c,
                &prior_c,
                chain_cfg,
                knee[i].clone(),
                seeds[i] + 700,
            )
            .expect("pcnl chain");
            pcnl_accept.0 += chain.accepted;
            pcnl_accept.1 += chain.proposals;
            pcnl.push(chain.theta);
        }
        let mut hitrun = Vec::new();
        let mut hitrun_accept = (0usize, 0usize);
        let mut hitrun_in = 0usize;
        let mut hitrun_draws = 0usize;
        for (i, _) in starts.iter().enumerate() {
            let chain = run_hit_and_run(
                &sat_strict,
                &prior_c,
                hitrun_cfg.clone(),
                knee[i].clone(),
                seeds[i] + 800,
            )
            .expect("hit-and-run");
            hitrun_accept.0 += chain.accepted;
            hitrun_accept.1 += chain.proposals;
            for d in &chain.draws {
                if train_loss(&loss, d) <= strict.tau {
                    hitrun_in += 1;
                }
                hitrun_draws += 1;
            }
            hitrun.push(
                chain
                    .draws
                    .iter()
                    .rev()
                    .find(|d| train_loss(&loss, d) <= strict.tau)
                    .cloned()
                    .unwrap_or(chain.theta),
            );
        }

        // Distillation from the geometry-aware ensemble, and an equal-budget
        // label-trained control.
        let distill_data: Vec<(Vec<f64>, Vec<f64>)> = states
            .iter()
            .map(|&x| (features(x), ensemble_policy(&loss, &hitrun, x)))
            .collect();
        let student_record = fit_softmax_student(3, loss.classes(), &distill_data, 600, 0.1)
            .expect("distilled student");
        let mut label_data = Vec::new();
        for traj in &train.trajectories {
            for t in 0..traj.len() {
                let mut onehot = vec![0.0; loss.classes()];
                onehot[traj.actions[t]] = 1.0;
                label_data.push((features(traj.states[t]), onehot));
            }
        }
        let control_record =
            fit_softmax_student(3, loss.classes(), &label_data, 600, 0.1).expect("label control");
        let student = student_record.student.clone();
        let control = control_record.student.clone();

        // Teacher–student fidelity against the geometry-aware teacher.
        let mut kls = Vec::new();
        let mut tvs = Vec::new();
        let mut control_kls = Vec::new();
        for &x in &states {
            let teacher = ensemble_policy(&loss, &hitrun, x);
            let s = student.probs(&features(x)).expect("student probs");
            kls.push(kl_divergence(&teacher, &s).unwrap());
            tvs.push(tv_distance(&teacher, &s).unwrap());
            let c = control.probs(&features(x)).expect("control probs");
            control_kls.push(kl_divergence(&teacher, &c).unwrap());
        }
        let student_kl = mean(&kls);
        let student_kl_max = kls.iter().copied().fold(0.0, f64::max);
        fidelity_rows.push((student_kl, mean(&tvs), student_kl_max, mean(&control_kls)));

        let arms: [(&str, Vec<Vec<f32>>); 8] = [
            ("radical-single", vec![radical_single.clone()]),
            ("radical-deep", radical_deep.clone()),
            ("knee-stopped", knee.clone()),
            ("sgd-swa", vec![swa.clone()]),
            ("sharp-gibbs", vec![sharp_chain.theta.clone()]),
            ("satisficing-pcnl", pcnl.clone()),
            ("satisficing-hitrun", hitrun.clone()),
            ("prior-only", starts.clone()),
        ];
        let policies_of = |members: &[Vec<f32>]| -> Vec<Vec<Vec<f64>>> {
            members
                .iter()
                .map(|w| states.iter().map(|&x| probs(&loss, w, x)).collect())
                .collect()
        };
        let mut nominal_returns = Vec::new();
        let mut shifted_returns = Vec::new();
        for (index, (name, members)) in arms.iter().enumerate() {
            let losses: Vec<f64> = members.iter().map(|w| train_loss(&loss, w)).collect();
            let in_region =
                losses.iter().filter(|l| **l <= config.tau).count() as f64 / losses.len() as f64;
            let (div, _) =
                action_distribution_diversity(&policies_of(members)).unwrap_or((0.0, 0.0));
            let policy = {
                let members = members.clone();
                let loss_ref = &loss;
                move |x: [f32; 2]| ensemble_policy(loss_ref, &members, x)
            };
            let nominal = rollout_returns(&env, &policy, episodes, 909);
            let shifted = rollout_returns(&shifted, &policy, episodes, 909);
            per_in[index].push(in_region);
            per_div[index].push(div);
            per_nom[index].push(mean(&nominal));
            per_shift[index].push(mean(&shifted));
            println!(
                "seed={family} arm={name} in_region={in_region:.3} loss_mean={:.4} diversity_tv={div:.4} nominal_return={:.4} nominal_se={:.6} shifted_return={:.4} shifted_se={:.6}",
                mean(&losses),
                mean(&nominal),
                se(&nominal),
                mean(&shifted),
                se(&shifted)
            );
            nominal_returns.push(nominal);
            shifted_returns.push(shifted);
        }
        // Students are scored through their own `SoftmaxStudent` forward, not
        // through a flat policy vector (they carry a bias), so they are kept
        // out of the weight arms above and appended at indices 8/9.
        let student_policy = {
            let s = student.clone();
            move |x: [f32; 2]| s.probs(&features(x)).expect("student probs")
        };
        let control_policy = {
            let s = control.clone();
            move |x: [f32; 2]| s.probs(&features(x)).expect("control probs")
        };
        for (index, name, policy) in [
            (
                8usize,
                "distilled-student",
                &student_policy as &dyn Fn([f32; 2]) -> Vec<f64>,
            ),
            (
                9usize,
                "label-control",
                &control_policy as &dyn Fn([f32; 2]) -> Vec<f64>,
            ),
        ] {
            let nominal = rollout_returns(&env, policy, episodes, 909);
            let shifted = rollout_returns(&shifted, policy, episodes, 909);
            per_in[index].push(f64::NAN);
            per_div[index].push(f64::NAN);
            per_nom[index].push(mean(&nominal));
            per_shift[index].push(mean(&shifted));
            println!(
                "seed={family} arm={name} in_region=NA diversity_tv=NA nominal_return={:.4} nominal_se={:.6} shifted_return={:.4} shifted_se={:.6}",
                mean(&nominal),
                se(&nominal),
                mean(&shifted),
                se(&shifted)
            );
            nominal_returns.push(nominal);
            shifted_returns.push(shifted);
        }
        // Paired per-episode contrasts on the same episode seeds.
        let pairs = [(6usize, 2usize), (5, 2), (8, 9), (6, 1)];
        for (slot, (a, b)) in pairs.iter().enumerate() {
            let (dn, _) = paired_delta(&nominal_returns[*a], &nominal_returns[*b]);
            let (ds, _) = paired_delta(&shifted_returns[*a], &shifted_returns[*b]);
            paired[slot].1.push(dn);
            paired[slot].2.push(ds);
        }
        // Once-per-episode member draw for the geometry-aware ensemble.
        let (draw_mean, draw_se) = {
            let mut rng = StdRng::seed_from_u64(4242);
            let mut returns = Vec::new();
            for _ in 0..episodes {
                let pick = rng.random_range(0..hitrun.len());
                let w = &hitrun[pick];
                let mut x = [0.0f32; 2];
                for slot in x.iter_mut() {
                    let z: f32 = rng.sample(StandardNormal);
                    *slot = z.clamp(-0.9 * env.x_max, 0.9 * env.x_max);
                }
                let mut total = 0.0;
                for _ in 0..env.horizon {
                    let p = probs(&loss, w, x);
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
        };
        episode_draw_rows.push(draw_mean);
        movement_rows.push((
            pcnl_accept.0 as f64 / pcnl_accept.1 as f64,
            hitrun_accept.0 as f64 / hitrun_accept.1 as f64,
            hitrun_in as f64 / hitrun_draws as f64,
            sharp_chain.accepted as f64 / sharp_chain.proposals as f64,
        ));
        println!(
            "seed={family} warm_stops={:?} pcnl_accept={}/{} hitrun_accept={}/{} hitrun_in_region={}/{} sharp_accept={}/{} swa_barrier={:.6} once_per_episode={draw_mean:.4} se={draw_se:.6} swa_moves={}",
            knee_stops,
            pcnl_accept.0,
            pcnl_accept.1,
            hitrun_accept.0,
            hitrun_accept.1,
            hitrun_in,
            hitrun_draws,
            sharp_chain.accepted,
            sharp_chain.proposals,
            pcnl::swa::barrier_height(
                &pcnl::swa::straight_line_barrier(
                    snapshots.first().unwrap(),
                    snapshots.last().unwrap(),
                    20,
                    |w| train_loss(&loss, w),
                )
                .expect("barrier")
            )
            .unwrap(),
            swa_proto.snapshots
        );
    }

    for (index, name) in arm_names.iter().enumerate() {
        let (in_mean, in_se) = mean_se(&per_in[index]);
        let (div_mean, _) = mean_se(&per_div[index]);
        let (nom_mean, nom_se) = mean_se(&per_nom[index]);
        let (sh_mean, sh_se) = mean_se(&per_shift[index]);
        println!(
            "aggregate arm={name} in_region_mean={in_mean:.3} in_region_se={in_se:.3} diversity_tv_mean={div_mean:.4} nominal_return_mean={nom_mean:.4} nominal_se={nom_se:.4} shifted_return_mean={sh_mean:.4} shifted_se={sh_se:.4}"
        );
    }
    for (name, nominal, shifted) in &paired {
        let (nm, nse) = mean_se(nominal);
        let (sm, sse) = mean_se(shifted);
        println!(
            "paired {name} nominal_delta={nm:+.6} nominal_se_across_seeds={nse:.6} shifted_delta={sm:+.6} shifted_se_across_seeds={sse:.6}"
        );
    }
    let (kl, tv, kl_max, control_kl) = {
        let mut acc = (0.0, 0.0, 0.0, 0.0);
        for (k, t, km, c) in &fidelity_rows {
            acc.0 += k;
            acc.1 += t;
            acc.2 += km;
            acc.3 += c;
        }
        let n = fidelity_rows.len() as f64;
        (acc.0 / n, acc.1 / n, acc.2 / n, acc.3 / n)
    };
    let tv_bound = (kl_max / 2.0).sqrt();
    println!(
        "fidelity student_kl_mean={kl:.6} student_tv_mean={tv:.6} student_kl_max_mean={kl_max:.6} pinsker_bound={:.6} control_teacher_kl_mean={control_kl:.6}",
        pinsker_return_bound(10.0, env.horizon, tv_bound).unwrap()
    );
    let (pc, hc, hi, sc) = {
        let mut acc = (0.0, 0.0, 0.0, 0.0);
        for row in &movement_rows {
            acc.0 += row.0;
            acc.1 += row.1;
            acc.2 += row.2;
            acc.3 += row.3;
        }
        let n = movement_rows.len() as f64;
        (acc.0 / n, acc.1 / n, acc.2 / n, acc.3 / n)
    };
    println!(
        "movement pcnl_accept_rate={pc:.6} hitrun_accept_rate={hc:.6} hitrun_in_region_rate={hi:.6} sharp_accept_rate={sc:.6}"
    );
    let (dm, dse) = mean_se(&episode_draw_rows);
    let bma = mean(&per_nom[6]);
    println!("ensemble_draw once_per_episode_mean={dm:.4} se={dse:.4} per_step_bma_mean={bma:.4}");
}
