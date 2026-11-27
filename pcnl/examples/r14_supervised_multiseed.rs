//! R14/P4 — multi-seed supervised campaign at the declared 57 378-parameter
//! vector-model size.
//!
//! This upgrades the single-seed reduced pilot (`r14_supervised_campaign.rs`)
//! to three independent seed families and the full `VectorShapes::new(768, 2)`
//! head (width 16, `ff` 32, 4 blocks, 4 tokens).  Budgets are still reduced
//! relative to the declared campaign (one fixture split, short chains); every
//! arm is reported per seed and then aggregated, with paired row-level and
//! across-seed intervals.
//!
//! Two satisficing arms are compared:
//!
//! * `satisficing-pcnl` — the plan's CN drift at `κ = 20`, which the pilot
//!   measured stalling at the curved feasible boundary;
//! * `satisficing-hitrun` — exact hit-and-run slice moves (no gradient-scale
//!   knob) with a stiff hinge `κ = 2e6`, which narrows the `L ≈ τ` boundary
//!   layer enough to retain in-region draws.  This is the geometry-aware tier:
//!   it changes the target's confinement, never the exactness of the kernel.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_supervised_multiseed
//! ```

use candle_core::{Device, Tensor};
use pcnl::ensemble::{KneeRule, WarmEntryConfig, member_record, warm_entry};
use pcnl::ensemble_diag::{
    effective_rank, in_region_fraction, normal_coverage, prediction_spread, rmse as rmse_of,
};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::swa::{barrier_height, batch_average, straight_line_barrier};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    EnergyTarget, HitRunConfig, SyntheticTeacher, VectorRegression, VectorSamplerConfig,
    VectorShapes, adam_fit, init_flat_prior, run_chain_with_covariances, run_hit_and_run,
};

#[derive(Clone, Copy)]
struct PerRow<'a> {
    inner: &'a VectorRegression,
    n: f64,
}

impl EnergyTarget for PerRow<'_> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (u, g) = self.inner.energy_and_grad(theta)?;
        Some((
            u / self.n,
            g.into_iter()
                .map(|x| (f64::from(x) / self.n) as f32)
                .collect(),
        ))
    }
}

struct Sharp<'a> {
    inner: &'a Satisficing<PerRow<'a>>,
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

fn tensor(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).expect("cpu tensor")
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_vec1::<f32>().expect("flat weights")
}

fn predict_rows(target: &VectorRegression, theta: &[f32]) -> Vec<Vec<f64>> {
    let t = tensor(theta);
    (0..target.rows())
        .map(|r| {
            target
                .predict_row(&t, r)
                .expect("prediction")
                .into_iter()
                .map(f64::from)
                .collect()
        })
        .collect()
}

fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn ensemble_predictions(target: &VectorRegression, members: &[Vec<f32>]) -> Vec<Vec<f64>> {
    members
        .iter()
        .map(|w| flatten(&predict_rows(target, w)))
        .collect()
}

fn mean_prediction(preds: &[Vec<f64>]) -> Vec<f64> {
    let q = preds[0].len();
    let mut mean = vec![0.0; q];
    for p in preds {
        for (acc, v) in mean.iter_mut().zip(p) {
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= preds.len() as f64;
    }
    mean
}

fn loss_at(target: &VectorRegression, theta: &[f32]) -> f64 {
    target
        .energy_and_grad(theta)
        .map(|(u, _)| u / target.rows() as f64)
        .unwrap_or(f64::INFINITY)
}

fn paired_abs_delta(a: &[f64], b: &[f64], y: &[f32]) -> (f64, f64) {
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .zip(y)
        .map(|((a, b), y)| (a - f64::from(*y)).abs() - (b - f64::from(*y)).abs())
        .collect();
    let n = diffs.len() as f64;
    let mean = diffs.iter().sum::<f64>() / n;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return (mean, f64::NAN);
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

#[derive(Clone, Copy)]
struct Metrics {
    rmse: f64,
    coverage: f64,
    in_region: f64,
    rank: f64,
    disagreement: f64,
}

fn score(
    train: &VectorRegression,
    test: &VectorRegression,
    y: &[f32],
    members: &[Vec<f32>],
    tau: f64,
    noise: f64,
) -> (Metrics, Vec<f64>) {
    let flat_preds = ensemble_predictions(test, members);
    let pred = mean_prediction(&flat_preds);
    let yf: Vec<f64> = y.iter().map(|v| f64::from(*v)).collect();
    let spread = prediction_spread(&flat_preds).expect("spread");
    let sigma: Vec<f64> = spread
        .variance
        .iter()
        .map(|v| (v + noise * noise).sqrt())
        .collect();
    let losses: Vec<f64> = members.iter().map(|w| loss_at(train, w)).collect();
    (
        Metrics {
            rmse: rmse_of(&pred, &yf).unwrap(),
            coverage: normal_coverage(&pred, &sigma, &yf, 0.9).unwrap(),
            in_region: in_region_fraction(&losses, tau).unwrap(),
            rank: effective_rank(&flat_preds).unwrap(),
            disagreement: spread.disagreement,
        },
        pred,
    )
}

fn main() {
    let fixture = std::path::Path::new("docs/fixtures/ultrabert_20ng4_embeddings.safetensors");
    let table = EmbeddingTable::load(fixture).expect("20ng fixture");
    let input = table.dim();
    let q = 2usize;
    let n_train = 120.min(table.x(Split::Train).len() / input);
    let n_test = 60.min(table.x(Split::Test).len() / input);
    let x_train = table.x(Split::Train)[..n_train * input].to_vec();
    let x_test = table.x(Split::Test)[..n_test * input].to_vec();
    let teacher = SyntheticTeacher::new(input, q, 20261127);
    let noise = 0.05;
    let lambda = 1.0;
    let mut y_train = Vec::new();
    let mut y_test = Vec::new();
    for row in 0..n_train {
        y_train.extend(teacher.targets(
            &x_train[row * input..(row + 1) * input],
            noise,
            900 + row as u64,
        ));
    }
    for row in 0..n_test {
        y_test.extend(teacher.targets(
            &x_test[row * input..(row + 1) * input],
            noise,
            1900 + row as u64,
        ));
    }
    let shapes = VectorShapes::new(input, q).expect("declared vector shapes");
    let train =
        VectorRegression::new(shapes, x_train, y_train, noise, lambda).expect("train target");
    let test = VectorRegression::new(shapes, x_test, y_test, noise, lambda).expect("test target");
    let per_row = PerRow {
        inner: &train,
        n: n_train as f64,
    };
    let config = SatisficingConfig {
        tau: 0.75,
        kappa: 20.0,
        alpha: 0.05,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let sat = Satisficing::new(per_row, config, "vector-regression", "mean").expect("config");
    let strict = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config
    };
    let sat_strict =
        Satisficing::new(per_row, strict, "vector-regression", "mean").expect("config");
    let prior_sigma = 0.05;
    let prior_c: Vec<f64> = shapes
        .prior_scales()
        .iter()
        .map(|s| prior_sigma * prior_sigma * s * s)
        .collect();
    let proposal_c: Vec<f64> = prior_c.iter().map(|c| c * 1e-6).collect();
    let m = 3usize;
    let seed_families = [101u64, 202, 303];
    let adam_steps = 400usize;
    let warm_cfg = WarmEntryConfig {
        max_steps: 300,
        lr: 0.02,
        tau: config.tau,
        knee: KneeRule::default(),
    };
    let chain_cfg = VectorSamplerConfig {
        steps: 300,
        burn: 60,
        thin: 12,
        rho: 0.9,
        refresh: 0.01,
        prior_sigma,
        draws: 20,
    };
    let hitrun_cfg = HitRunConfig {
        moves: 80,
        burn: 20,
        thin: 3,
        draws: 20,
        width: 1.0,
        max_expand: 32,
        max_shrink: 64,
    };
    let y_te: Vec<f32> = test.y().to_vec();
    println!(
        "r14=multiseed P4 seeds={} members={} params={} train={n_train} test={n_test} tau={} kappa_pcnl={} kappa_strict={} alpha={} hinge={} noise={noise}",
        seed_families.len(),
        m,
        shapes.flat_size(),
        config.tau,
        config.kappa,
        strict.kappa,
        config.alpha,
        config.hinge.label()
    );
    println!(
        "target_identity={} strict_identity={}",
        sat.identity(&prior_c),
        sat_strict.identity(&prior_c)
    );

    let arm_names = [
        "radical-single",
        "radical-deep",
        "knee-stopped",
        "sgd-swa",
        "sharp-gibbs",
        "satisficing-pcnl",
        "satisficing-hitrun",
        "distilled-student",
        "label-control",
        "prior-only",
    ];
    let mut per_arm: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_arm_cov: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_arm_in: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_arm_rank: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    let mut per_arm_dis: Vec<Vec<f64>> = vec![Vec::new(); arm_names.len()];
    // Paired row-level contrasts, one value per seed.
    let mut paired: Vec<(&str, Vec<f64>)> = vec![
        ("satisficing-pcnl_vs_knee", Vec::new()),
        ("satisficing-hitrun_vs_knee", Vec::new()),
        ("distilled_vs_label", Vec::new()),
    ];
    let mut wall_seed = Vec::new();

    for (family_index, &family) in seed_families.iter().enumerate() {
        let started = std::time::Instant::now();
        let seeds: Vec<u64> = (0..m as u64).map(|k| family + 11 * k).collect();
        let starts: Vec<Vec<f32>> = seeds
            .iter()
            .map(|&seed| flat(&init_flat_prior(&shapes, prior_sigma, seed)))
            .collect();

        // Radical single/deep.
        let (t, radical_trace) = adam_fit(&train, tensor(&starts[0]), adam_steps, 2e-3);
        let radical_single = flat(&t);
        let mut radical_deep = Vec::new();
        for s in &starts {
            let (t, _) = adam_fit(&train, tensor(s), adam_steps, 2e-3);
            radical_deep.push(flat(&t));
        }

        // Knee-stopped ensemble.
        let mut knee = Vec::new();
        for (i, s) in starts.iter().enumerate() {
            let warm = warm_entry(&per_row, s, warm_cfg).expect("warm entry");
            let record = member_record(&sat, &warm.theta, seeds[i], &warm).expect("member");
            knee.push(record.weights);
        }

        // SGD-SWA on the first basin.
        let (t, _) = adam_fit(&train, tensor(&knee[0]), 100, 2e-3);
        let mut swa_theta = flat(&t);
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            let (t, _) = adam_fit(&train, tensor(&swa_theta), 10, 2e-3);
            swa_theta = flat(&t);
            snapshots.push(swa_theta.clone());
        }
        let swa = batch_average(&snapshots).expect("swa average");
        let barrier = straight_line_barrier(
            snapshots.first().unwrap(),
            snapshots.last().unwrap(),
            20,
            |w| loss_at(&train, w),
        )
        .expect("barrier");

        // Sharp Gibbs control (one chain per seed, same starts).
        let sharp = Sharp {
            inner: &sat,
            lambda: 300.0,
        };
        let sharp_chain = run_chain_with_covariances(
            &sharp,
            &prior_c,
            &proposal_c,
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

        // Plan pCNL arm and geometry-aware hit-and-run arm.
        let mut pcnl = Vec::new();
        let mut pcnl_accept = (0usize, 0usize);
        for (i, _) in starts.iter().enumerate() {
            let chain = run_chain_with_covariances(
                &sat,
                &prior_c,
                &proposal_c,
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
            .expect("hit-and-run chain");
            hitrun_accept.0 += chain.accepted;
            hitrun_accept.1 += chain.proposals;
            for d in &chain.draws {
                if loss_at(&train, d) <= strict.tau {
                    hitrun_in += 1;
                }
                hitrun_draws += 1;
            }
            // Retain the last in-region draw when there is one; this is the
            // compressed per-basin configuration, and the selection rule is
            // declared before scoring.
            let retained = chain
                .draws
                .iter()
                .rev()
                .find(|d| loss_at(&train, d) <= strict.tau)
                .cloned()
                .unwrap_or(chain.theta);
            hitrun.push(retained);
        }

        // Distilled student and equal-budget label control.
        let teacher_train = mean_prediction(&ensemble_predictions(&train, &hitrun));
        let student_y: Vec<f32> = teacher_train.iter().map(|v| *v as f32).collect();
        let student_target =
            VectorRegression::new(shapes, train.x().to_vec(), student_y, noise, lambda)
                .expect("distill target");
        let (t, _) = adam_fit(
            &student_target,
            init_flat_prior(&shapes, prior_sigma, 777),
            adam_steps,
            2e-3,
        );
        let student = flat(&t);
        let (t, _) = adam_fit(
            &train,
            init_flat_prior(&shapes, prior_sigma, 777),
            adam_steps,
            2e-3,
        );
        let control = flat(&t);

        let arms: [(&str, Vec<Vec<f32>>); 10] = [
            ("radical-single", vec![radical_single.clone()]),
            ("radical-deep", radical_deep.clone()),
            ("knee-stopped", knee.clone()),
            ("sgd-swa", vec![swa.clone()]),
            ("sharp-gibbs", vec![sharp_chain.theta.clone()]),
            ("satisficing-pcnl", pcnl.clone()),
            ("satisficing-hitrun", hitrun.clone()),
            ("distilled-student", vec![student.clone()]),
            ("label-control", vec![control.clone()]),
            ("prior-only", starts.clone()),
        ];
        let mut preds = Vec::new();
        for (index, (name, members)) in arms.iter().enumerate() {
            let (metrics, pred) = score(&train, &test, &y_te, members, strict.tau, noise);
            per_arm[index].push(metrics.rmse);
            per_arm_cov[index].push(metrics.coverage);
            per_arm_in[index].push(metrics.in_region);
            per_arm_rank[index].push(metrics.rank);
            per_arm_dis[index].push(metrics.disagreement);
            preds.push(pred);
            println!(
                "seed={family} arm={name} rmse={:.6} coverage90={:.4} in_region={:.3} eff_rank={:.4} disagreement={:.6e}",
                metrics.rmse,
                metrics.coverage,
                metrics.in_region,
                metrics.rank,
                metrics.disagreement
            );
        }
        let (d1, _) = paired_abs_delta(&preds[5], &preds[2], &y_te);
        let (d2, _) = paired_abs_delta(&preds[6], &preds[2], &y_te);
        let (d3, _) = paired_abs_delta(&preds[7], &preds[8], &y_te);
        paired[0].1.push(d1);
        paired[1].1.push(d2);
        paired[2].1.push(d3);
        wall_seed.push(started.elapsed().as_secs_f64());
        println!(
            "seed={family} radical_trace_first={:.4} radical_trace_last={:.4} sharp_accept={}/{} pcnl_accept={}/{} hitrun_accept={}/{} hitrun_in_region_draws={}/{} swa_barrier={:.6}",
            radical_trace.first().copied().unwrap_or(f64::NAN),
            radical_trace.last().copied().unwrap_or(f64::NAN),
            sharp_chain.accepted,
            sharp_chain.proposals,
            pcnl_accept.0,
            pcnl_accept.1,
            hitrun_accept.0,
            hitrun_accept.1,
            hitrun_in,
            hitrun_draws,
            barrier_height(&barrier).unwrap()
        );
        let _ = family_index;
    }

    for (index, name) in arm_names.iter().enumerate() {
        let (rmse_mean, rmse_se) = mean_se(&per_arm[index]);
        let (cov_mean, _) = mean_se(&per_arm_cov[index]);
        let (in_mean, _) = mean_se(&per_arm_in[index]);
        let (rank_mean, _) = mean_se(&per_arm_rank[index]);
        let (dis_mean, _) = mean_se(&per_arm_dis[index]);
        println!(
            "aggregate arm={name} rmse_mean={rmse_mean:.6} rmse_se={rmse_se:.6} coverage90_mean={cov_mean:.4} in_region_mean={in_mean:.3} eff_rank_mean={rank_mean:.4} disagreement_mean={dis_mean:.6e}"
        );
    }
    for (name, values) in &paired {
        let (mean, se) = mean_se(values);
        println!("paired {name} delta_mean={mean:+.6} se_across_seeds={se:.6}");
    }
    let (wall_mean, _) = mean_se(&wall_seed);
    println!(
        "wall_clock_s_per_seed={wall_mean:.2} seeds={}",
        seed_families.len()
    );
}
