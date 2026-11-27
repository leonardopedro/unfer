//! R14/P5 on RLCD/Edge0 checkpoints (reduced-budget, three seed families).
//!
//! The synthetic 48-transition corpus is replaced by the Edge0 no-ul/choice/
//! score world model.  Each member is a real `edge0_campaign` RLCD checkpoint,
//! written to `target/r14_edge0_checkpoints/*.safetensors` and reloaded before
//! it is used.  The P5 protocol is then applied in function space:
//!
//! * satisficing pCNL and hit-and-run on the Edge0 NLL (via an `EnergyTarget`
//!   adapter over `Edge0Potential`), `τ` fixed on training worlds;
//! * an ensemble of the three RLCD checkpoints, a supervised-checkpoint
//!   control, and an init-only control;
//! * KL distillation of the RLCD ensemble into a linear softmax student over
//!   the model's `state_rep` + question representation, on no-ul questions;
//! * Brier score on held-out test worlds, clustered by world, on both the test
//!   split and the calibration split (the shifted-world readout).
//!
//! This is **not** the declared R9 campaign (2048/256/256/512 worlds) and the
//! student is a linear readout, not a second Edge0.  Those limits are logged.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_rl_edge0_multiseed
//! ```

use candle_core::{Device, Tensor};
use pcnl::distill::{ensemble_probs, fit_softmax_student};
use pcnl::edge0_campaign::{Arm, BatchUnit, CampaignConfig, Sample, build_samples, run};
use pcnl::edge0_model::{Edge0Config, ModelInput, forward};
use pcnl::edge0_potential::Edge0Potential;
use pcnl::edge0_world::{QuestionKind, Split};
use pcnl::ensemble_diag::{effective_rank, tv_distance};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::target::Potential;
use pcnl::vector_model::{
    EnergyTarget, HitRunConfig, VectorSamplerConfig, run_chain_with_covariances, run_hit_and_run,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

struct EdgeTarget {
    potential: Edge0Potential,
    dim: usize,
}

impl EnergyTarget for EdgeTarget {
    fn dim(&self) -> usize {
        self.dim
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        if theta.len() != self.dim {
            return None;
        }
        let t = Tensor::from_vec(theta.to_vec(), self.dim, &Device::Cpu).ok()?;
        let u = self.potential.mean_loss(&t);
        if !u.is_finite() {
            return None;
        }
        let g = self.potential.mean_grad(&t)?;
        Some((u, g.to_vec1::<f32>().ok()?))
    }
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

fn flat(theta: &Tensor) -> Vec<f32> {
    theta
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("values")
}

fn checkpoint_path(arm: &str, family: u64, member: usize) -> PathBuf {
    PathBuf::from("target/r14_edge0_checkpoints")
        .join(format!("{arm}-{family}-{member}.safetensors"))
}

fn save_checkpoint(arm: &str, family: u64, member: usize, theta: &Tensor) -> PathBuf {
    let path = checkpoint_path(arm, family, member);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("checkpoint dir");
    theta
        .save_safetensors("theta", &path)
        .expect("save checkpoint");
    path
}

fn load_checkpoint(path: &PathBuf) -> Tensor {
    let map = candle_core::safetensors::load(path, &Device::Cpu).expect("load checkpoint");
    map.get("theta").expect("theta tensor").clone()
}

fn marginals(theta: &Tensor, config: &Edge0Config, samples: &[Sample]) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|s| {
            let input = ModelInput {
                state: &s.input_state,
                questions: std::slice::from_ref(&s.question),
            };
            forward(theta, config, &input)
                .expect("forward")
                .per_question[0]
                .marginal_vec()
                .expect("marginal")
        })
        .collect()
}

fn brier(probs: &[f64], truth: &[f64]) -> f64 {
    probs
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t) * (p - t))
        .sum()
}

fn world_means(samples: &[Sample], per_question: &[f64]) -> BTreeMap<u32, f64> {
    let mut acc: BTreeMap<u32, (f64, usize)> = BTreeMap::new();
    for (s, v) in samples.iter().zip(per_question) {
        let slot = acc.entry(s.world).or_insert((0.0, 0));
        slot.0 += v;
        slot.1 += 1;
    }
    acc.into_iter()
        .map(|(w, (sum, n))| (w, sum / n as f64))
        .collect()
}

fn cluster_stats(means: &BTreeMap<u32, f64>) -> (f64, f64) {
    let vals: Vec<f64> = means.values().copied().collect();
    mean_se(&vals)
}

fn paired_worlds(a: &BTreeMap<u32, f64>, b: &BTreeMap<u32, f64>) -> (f64, f64) {
    let diffs: Vec<f64> = a
        .iter()
        .filter_map(|(w, av)| b.get(w).map(|bv| av - bv))
        .collect();
    mean_se(&diffs)
}

fn mean_tv(member_probs: &[Vec<Vec<f64>>]) -> f64 {
    let m = member_probs.len();
    if m < 2 {
        return 0.0;
    }
    let q = member_probs[0].len();
    let mut out = 0.0;
    let mut count = 0usize;
    for i in 0..m {
        for j in (i + 1)..m {
            for k in 0..q {
                out += tv_distance(&member_probs[i][k], &member_probs[j][k]).unwrap_or(0.0);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { out / count as f64 }
}

fn mean_probs(member_probs: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    let q = member_probs[0].len();
    (0..q)
        .map(|k| {
            let ps: Vec<Vec<f64>> = member_probs.iter().map(|m| m[k].clone()).collect();
            ensemble_probs(&ps).expect("ensemble")
        })
        .collect()
}

fn features(s: &Sample) -> Vec<f64> {
    // Deliberately teacher-independent: a standalone linear readout must not
    // consume the Edge0 backbone's own representation, or distillation is a
    // copy of the teacher's last layer rather than a compressed policy.
    let mut f = vec![0.0f64; 8];
    for &t in &s.input_state {
        f[(t as usize) % 4] += 1.0;
    }
    let n = s.input_state.len().max(1) as f64;
    for v in f.iter_mut().take(4) {
        *v /= n;
    }
    f[4] = s.input_state.iter().map(|&t| f64::from(t)).sum::<f64>() / n / 32.0;
    let qn = s.question.tokens.len().max(1) as f64;
    f[5] = s.question.tokens.iter().map(|&t| f64::from(t)).sum::<f64>() / qn / 32.0;
    f[6] = s.question.options.len() as f64 / 4.0;
    f[7] = s.question.kind.outcomes() as f64 / 4.0;
    f
}

fn main() {
    let base = CampaignConfig {
        seed: 0,
        data_seed: Some(20_261_127),
        batch_unit: BatchUnit::Questions,
        worlds: [24, 8, 8, 16],
        updates: 60,
        warm_start: 30,
        ..CampaignConfig::default()
    };
    let config = base.model.clone();
    let dim = config.flat_size();
    let samples = build_samples(&base);
    let train: Vec<Sample> = samples
        .iter()
        .filter(|s| s.split == Split::Train)
        .cloned()
        .collect();
    let test: Vec<Sample> = samples
        .iter()
        .filter(|s| s.split == Split::Test)
        .cloned()
        .collect();
    let cal: Vec<Sample> = samples
        .iter()
        .filter(|s| s.split == Split::Calibration)
        .cloned()
        .collect();
    let potential = Edge0Potential::new(config.clone(), train.clone(), vec![true; dim]);
    let target = EdgeTarget { potential, dim };
    let tau = 1.05;
    let config_sat = SatisficingConfig {
        tau,
        kappa: 20.0,
        alpha: 0.0,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let strict = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config_sat
    };
    let sat = Satisficing::new(&target, config_sat, "edge0-nll", "mean").expect("sat");
    let sat_strict = Satisficing::new(&target, strict, "edge0-nll", "mean").expect("strict");
    let prior_c = vec![1.0f64; dim];
    let families = [1u64, 2, 3];
    let members = 3usize;

    let mut kind_counts: BTreeMap<String, usize> = BTreeMap::new();
    for s in &test {
        *kind_counts
            .entry(s.question.kind.label().to_string())
            .or_default() += 1;
    }
    println!("kinds test={kind_counts:?}");
    println!(
        "r14=p5-edge0-multiseed source=rlcd-checkpoints params={dim} families={} members={members} worlds={:?} tau={tau} kappa_pcnl={} kappa_strict={} updates={} warm_start={} scope=reduced-budget",
        families.len(),
        base.worlds,
        config_sat.kappa,
        strict.kappa,
        base.updates,
        base.warm_start
    );

    let mut arm_brier_test: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut arm_brier_cal: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut arm_in: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut arm_rank: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut arm_div: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut student_brier_test = Vec::new();
    let mut teacher_brier_test = Vec::new();
    let mut student_kl = Vec::new();
    let mut student_tv = Vec::new();
    let mut paired_e_s = Vec::new();
    let mut paired_pcnl = Vec::new();
    let mut paired_hit = Vec::new();
    let mut paired_student = Vec::new();

    for &family in &families {
        // Train and persist real RLCD/supervised checkpoints for this family.
        let mut rlcd: Vec<Tensor> = Vec::new();
        let mut supervised: Vec<Tensor> = Vec::new();
        let mut prior: Vec<Tensor> = Vec::new();
        for k in 0..members {
            let mut cfg = base.clone();
            cfg.seed = family * 1000 + k as u64;
            let report = run(&cfg).expect("edge0 campaign");
            let pick = |arm: Arm| {
                report
                    .trained
                    .iter()
                    .find(|(a, _)| *a == arm)
                    .map(|(_, t)| t.clone())
                    .expect("trained arm")
            };
            let r = pick(Arm::Rlcd);
            let s = pick(Arm::Supervised);
            save_checkpoint("rlcd", family, k, &r);
            save_checkpoint("supervised", family, k, &s);
            rlcd.push(load_checkpoint(&checkpoint_path("rlcd", family, k)));
            supervised.push(load_checkpoint(&checkpoint_path("supervised", family, k)));
            prior.push(pcnl::edge0_model::init(&config, cfg.seed));
        }

        // Satisficing pCNL and hit-and-run on the Edge0 NLL.
        let chain_cfg = VectorSamplerConfig {
            steps: 30,
            burn: 8,
            thin: 1,
            rho: 0.9,
            refresh: 0.01,
            prior_sigma: 1.0,
            draws: 1,
        };
        let hit_cfg = HitRunConfig {
            moves: 8,
            burn: 2,
            thin: 1,
            draws: 2,
            width: 1.0,
            max_expand: 16,
            max_shrink: 32,
        };
        let mut pcnl: Vec<Tensor> = Vec::new();
        let mut hitrun: Vec<Tensor> = Vec::new();
        let mut pcnl_accept = (0usize, 0usize);
        let mut hit_accept = (0usize, 0usize);
        let mut hit_in = (0usize, 0usize);
        for k in 0..members {
            let start = flat(&rlcd[k]);
            let chain = run_chain_with_covariances(
                &sat,
                &prior_c,
                &prior_c,
                chain_cfg,
                start,
                family * 7919 + k as u64,
            )
            .expect("pcnl");
            pcnl_accept.0 += chain.accepted;
            pcnl_accept.1 += chain.proposals;
            let theta = Tensor::from_vec(chain.theta.clone(), dim, &Device::Cpu).expect("theta");
            save_checkpoint("pcnl", family, k, &theta);
            pcnl.push(load_checkpoint(&checkpoint_path("pcnl", family, k)));

            let hr = run_hit_and_run(
                &sat_strict,
                &prior_c,
                hit_cfg.clone(),
                flat(&rlcd[k]),
                family * 7919 + 100 + k as u64,
            )
            .expect("hitrun");
            hit_accept.0 += hr.accepted;
            hit_accept.1 += hr.proposals;
            for d in &hr.draws {
                if sat.loss(d).map(|l| l <= strict.tau).unwrap_or(false) {
                    hit_in.0 += 1;
                }
                hit_in.1 += 1;
            }
            let theta = Tensor::from_vec(hr.theta.clone(), dim, &Device::Cpu).expect("theta");
            save_checkpoint("hitrun", family, k, &theta);
            hitrun.push(load_checkpoint(&checkpoint_path("hitrun", family, k)));
        }

        let arms: [(&str, &[Tensor]); 5] = [
            ("rlcd-single", &rlcd[..1]),
            ("rlcd-ensemble", &rlcd),
            ("supervised-ensemble", &supervised),
            ("satisficing-pcnl", &pcnl),
            ("satisficing-hitrun", &hitrun),
        ];
        let mut test_world: BTreeMap<&str, BTreeMap<u32, f64>> = BTreeMap::new();
        let mut mean_marginals: Vec<Vec<Vec<f64>>> = Vec::new();

        for (name, ts) in &arms {
            let losses: Vec<f64> = ts
                .iter()
                .map(|t| sat.loss(&flat(t)).unwrap_or(f64::NAN))
                .collect();
            let in_region =
                losses.iter().filter(|l| **l <= tau).count() as f64 / losses.len() as f64;
            let per_test_all: Vec<Vec<Vec<f64>>> =
                ts.iter().map(|t| marginals(t, &config, &test)).collect();
            let ensemble_test = mean_probs(&per_test_all);
            let brier_test: Vec<f64> = test
                .iter()
                .zip(&ensemble_test)
                .map(|(s, p)| brier(p, &s.truth))
                .collect();
            let wm_test = world_means(&test, &brier_test);
            let (b_test, se_test) = cluster_stats(&wm_test);
            let per_cal_all: Vec<Vec<Vec<f64>>> =
                ts.iter().map(|t| marginals(t, &config, &cal)).collect();
            let ensemble_cal = mean_probs(&per_cal_all);
            let brier_cal: Vec<f64> = cal
                .iter()
                .zip(&ensemble_cal)
                .map(|(s, p)| brier(p, &s.truth))
                .collect();
            let (b_cal, _) = cluster_stats(&world_means(&cal, &brier_cal));
            let div = mean_tv(&per_test_all);
            let per_member_flat: Vec<Vec<f64>> = per_test_all
                .iter()
                .map(|m| m.iter().flatten().copied().collect())
                .collect();
            let rank = effective_rank(&per_member_flat).unwrap_or(f64::NAN);
            println!(
                "family={family} arm={name} in_region={in_region:.3} nll_mean={:.4} brier_test={b_test:.5} brier_test_se={se_test:.5} brier_cal={b_cal:.5} diversity_tv={div:.4} eff_rank={rank:.3}",
                mean(&losses)
            );
            arm_brier_test.entry(name).or_default().push(b_test);
            arm_brier_cal.entry(name).or_default().push(b_cal);
            arm_in.entry(name).or_default().push(in_region);
            arm_rank.entry(name).or_default().push(rank);
            arm_div.entry(name).or_default().push(div);
            test_world.insert(name, wm_test);
            if *name == "rlcd-ensemble" {
                mean_marginals = per_test_all;
            }
        }

        // Distill the RLCD ensemble into a linear softmax student on no-ul
        // questions, using state_rep + question rep as features.
        let noul_train: Vec<&Sample> = train
            .iter()
            .filter(|s| matches!(s.question.kind, QuestionKind::Noul))
            .collect();
        let noul_test: Vec<&Sample> = test
            .iter()
            .filter(|s| matches!(s.question.kind, QuestionKind::Noul))
            .collect();
        let teacher_train: Vec<Vec<f64>> = noul_train
            .iter()
            .map(|s| {
                let ps: Vec<Vec<f64>> = rlcd
                    .iter()
                    .map(|t| {
                        let input = ModelInput {
                            state: &s.input_state,
                            questions: std::slice::from_ref(&s.question),
                        };
                        forward(t, &config, &input).expect("forward").per_question[0]
                            .marginal_vec()
                            .expect("marginal")
                    })
                    .collect();
                ensemble_probs(&ps).expect("teacher")
            })
            .collect();
        let data: Vec<(Vec<f64>, Vec<f64>)> = noul_train
            .iter()
            .zip(&teacher_train)
            .map(|(s, t)| (features(s), t.clone()))
            .collect();
        let record =
            fit_softmax_student(8, noul_train[0].truth.len(), &data, 400, 0.1).expect("student");
        let student = record.student;
        let mut kl = Vec::new();
        let mut tv = Vec::new();
        let mut sb = Vec::new();
        let mut tb = Vec::new();
        let mut tmax = Vec::new();
        for s in &noul_test {
            let ps: Vec<Vec<f64>> = rlcd
                .iter()
                .map(|t| {
                    let input = ModelInput {
                        state: &s.input_state,
                        questions: std::slice::from_ref(&s.question),
                    };
                    forward(t, &config, &input).expect("forward").per_question[0]
                        .marginal_vec()
                        .expect("marginal")
                })
                .collect();
            let teacher = ensemble_probs(&ps).expect("teacher");
            let sp = student.probs(&features(s)).expect("student");
            kl.push(pcnl::ensemble_diag::kl_divergence(&teacher, &sp).unwrap_or(f64::NAN));
            tv.push(tv_distance(&teacher, &sp).unwrap_or(f64::NAN));
            tb.push(brier(&teacher, &s.truth));
            tmax.push(teacher.iter().copied().fold(0.0f64, f64::max));
            sb.push(brier(&sp, &s.truth));
        }
        let (s_brier, _) = mean_se(&sb);
        let (tmax_mean, _) = mean_se(&tmax);
        let (t_brier, _) = mean_se(&tb);
        student_brier_test.push(s_brier);
        teacher_brier_test.push(t_brier);
        student_kl.push(mean(&kl));
        student_tv.push(mean(&tv));
        paired_student.push(s_brier - t_brier);

        let (e_s, e_s_se) = paired_worlds(&test_world["rlcd-ensemble"], &test_world["rlcd-single"]);
        let (p_v, p_se) = paired_worlds(
            &test_world["satisficing-pcnl"],
            &test_world["rlcd-ensemble"],
        );
        let (h_v, h_se) = paired_worlds(
            &test_world["satisficing-hitrun"],
            &test_world["rlcd-ensemble"],
        );
        paired_e_s.push(e_s);
        paired_pcnl.push(p_v);
        paired_hit.push(h_v);
        println!(
            "family={family} distillation noul_train={} noul_test={} teacher_max_mean={tmax_mean:.5} student_kl={:.5} student_tv={:.5} student_brier={s_brier:.5} teacher_brier={t_brier:.5} paired_delta={:+.5} pcnl_accept={}/{} hitrun_accept={}/{} hitrun_in={}/{} ensemble_minus_single={e_s:+.5}±{e_s_se:.5} pcnl_minus_ensemble={p_v:+.5}±{p_se:.5} hitrun_minus_ensemble={h_v:+.5}±{h_se:.5}",
            noul_train.len(),
            noul_test.len(),
            mean(&kl),
            mean(&tv),
            s_brier - t_brier,
            pcnl_accept.0,
            pcnl_accept.1,
            hit_accept.0,
            hit_accept.1,
            hit_in.0,
            hit_in.1
        );
    }

    for (name, vals) in &arm_brier_test {
        let (m, s) = mean_se(vals);
        let (cm, _) = mean_se(arm_brier_cal.get(name).expect("cal"));
        let (im, _) = mean_se(arm_in.get(name).expect("in"));
        println!(
            "aggregate arm={name} brier_test_mean={m:.5} se={s:.5} brier_cal_mean={cm:.5} in_region_mean={im:.3}"
        );
    }
    for name in arm_brier_test.keys() {
        let (dm, dse) = mean_se(arm_div.get(name).expect("div"));
        let (rm, rse) = mean_se(arm_rank.get(name).expect("rank"));
        println!(
            "aggregate arm={name} diversity_tv_mean={dm:.4} se={dse:.4} eff_rank_mean={rm:.3} se={rse:.3}"
        );
    }
    let (sm, sse) = mean_se(&student_brier_test);
    let (tm, _) = mean_se(&teacher_brier_test);
    let (sd, sdse) = mean_se(&paired_student);
    let (klm, _) = mean_se(&student_kl);
    let (tvm, _) = mean_se(&student_tv);
    println!(
        "distillation student_brier_mean={sm:.5} se={sse:.5} teacher_brier_mean={tm:.5} paired_delta_mean={sd:+.5} se={sdse:.5} kl_mean={klm:.5} tv_mean={tvm:.5}"
    );
    for (name, vals) in [
        ("ensemble_minus_single", &paired_e_s),
        ("pcnl_minus_ensemble", &paired_pcnl),
        ("hitrun_minus_ensemble", &paired_hit),
    ] {
        let (m, s) = mean_se(vals);
        println!("paired {name} delta_mean={m:+.5} se={s:.5}");
    }
    println!(
        "scope source=RLCD/Edge0 checkpoints; reduced-budget campaign; linear softmax student on teacher-independent surface features; not the declared R9 split"
    );
}
