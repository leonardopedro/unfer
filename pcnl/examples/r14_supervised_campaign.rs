//! R14/P4 (reduced-budget pilot) — supervised SWA, multi-basin resampling and
//! posterior distillation on the frozen 20 Newsgroups embeddings.
//!
//! This is a **reduced-budget mechanics pilot**, not the declared campaign:
//! one split, one seed family, `M = 4`, a small four-block vector head and a
//! 600-step radical control.  It exercises every number the full campaign must
//! report and prints them verbatim, including negative or inconclusive
//! contrasts:
//!
//! * eight matched-compute comparators (radical single/deep, knee-stopped,
//!   SGD-SWA, sharp Gibbs, satisficing pCNL, distilled student with a
//!   label-trained control, prior-only);
//! * in-region fit, parameter and functional diversity, effective rank,
//!   RMSE/NLL/Brier/coverage/`z²`;
//! * SWA weight-mean versus output-mean and straight-line barriers;
//! * teacher–student vector fidelity and a paired row-level interval.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_supervised_campaign
//! ```

use candle_core::{Device, Tensor};
use pcnl::distill::{SpreadHead, VectorEnsembleStats, vector_stats};
use pcnl::ensemble::{Ensemble, KneeRule, WarmEntryConfig, member_record, warm_entry};
use pcnl::ensemble_diag::{
    effective_rank, in_region_fraction, normal_coverage, normalized_residual_z2, prediction_spread,
    rmse as rmse_of,
};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::swa::{SwaCheckpoint, SwaKind, barrier_height, batch_average, straight_line_barrier};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    EnergyTarget, HitRunConfig, SyntheticTeacher, VectorRegression, VectorSamplerConfig,
    VectorShapes, adam_fit, init_flat_prior, run_chain_with_covariances, run_hit_and_run,
};

/// `L = energy / n`: makes the declared loss a per-row mean, matching the
/// plan's units (`L ≈ ½` at the declared observation noise).
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

/// Sharp Gibbs control `λ·L`.
struct Sharp<'a, T: EnergyTarget> {
    inner: &'a T,
    lambda: f64,
}

impl<T: EnergyTarget> EnergyTarget for Sharp<'_, T> {
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

fn weights(t: &Tensor) -> Vec<f32> {
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

fn flatten(preds: &[Vec<f64>]) -> Vec<f64> {
    preds.iter().flatten().copied().collect()
}

fn ensemble_flat(target: &VectorRegression, members: &[Vec<f32>]) -> Vec<Vec<f64>> {
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

/// Paired difference of absolute errors `|a − y| − |b − y|`, with its
/// standard error over coordinates.
fn paired_rmse_delta(a: &[f64], b: &[f64], y: &[f32]) -> (f64, f64) {
    if a.is_empty() || a.len() != b.len() || a.len() != y.len() {
        return (f64::NAN, f64::NAN);
    }
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

fn y_flat(target: &VectorRegression) -> Vec<f32> {
    target.y().to_vec()
}

fn main() {
    let started = std::time::Instant::now();
    let fixture = std::path::Path::new("docs/fixtures/ultrabert_20ng4_embeddings.safetensors");
    let (input, x_train, x_test) = match EmbeddingTable::load(fixture) {
        Ok(table) => {
            let dim = table.dim();
            let n_train = 96.min(table.x(Split::Train).len() / dim);
            let n_test = 48.min(table.x(Split::Test).len() / dim);
            (
                dim,
                table.x(Split::Train)[..n_train * dim].to_vec(),
                table.x(Split::Test)[..n_test * dim].to_vec(),
            )
        }
        Err(e) => {
            println!("fixture=unavailable error={e} fallback=synthetic");
            let dim = 32;
            let mut x_train = Vec::new();
            let mut x_test = Vec::new();
            for i in 0..96 * dim {
                x_train.push(((i as f32 * 0.731).sin() * 0.5) as f32);
            }
            for i in 0..48 * dim {
                x_test.push(((i as f32 * 0.317 + 0.7).cos() * 0.5) as f32);
            }
            (dim, x_train, x_test)
        }
    };
    let q = 2usize;
    let n_train = x_train.len() / input;
    let n_test = x_test.len() / input;
    let noise = 0.05;
    let lambda = 1.0;
    let teacher = SyntheticTeacher::new(input, q, 20261127);
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

    let shapes = VectorShapes::with_width(input, q, 2, 4, 1, 2).expect("shapes");
    let train = VectorRegression::new(shapes, x_train.clone(), y_train.clone(), noise, lambda)
        .expect("train target");
    let test = VectorRegression::new(shapes, x_test.clone(), y_test.clone(), noise, lambda)
        .expect("test target");
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
    let prior_sigma = 0.05;
    let prior_c: Vec<f64> = shapes
        .prior_scales()
        .iter()
        .map(|s| prior_sigma * prior_sigma * s * s)
        .collect();
    println!(
        "r14=pilot P4 fixture_rows train={n_train} test={n_test} dim={input} shapes=w{}_ff{}_l{}_t{} params={} tau={} kappa={} alpha={} delta={} hinge={}",
        shapes.width,
        shapes.ff,
        shapes.layers,
        shapes.tokens,
        shapes.flat_size(),
        config.tau,
        config.kappa,
        config.alpha,
        config.delta,
        config.hinge.label()
    );
    println!("target_identity={}", sat.identity(&prior_c));

    let m = 4usize;
    let seeds = [31u64, 32, 33, 34];
    let adam_steps = 600usize;
    let warm_cfg = WarmEntryConfig {
        max_steps: 300,
        lr: 0.02,
        tau: config.tau,
        knee: KneeRule::default(),
    };

    // Prior-only control and the initial draws every arm shares.
    let mut prior_members = Vec::new();
    let mut starts = Vec::new();
    for &seed in &seeds {
        let t0 = init_flat_prior(&shapes, prior_sigma, seed);
        prior_members.push(weights(&t0));
        starts.push(weights(&t0));
    }

    // (1) Radical single point and (2) radical deep ensemble.
    let radical_start = starts[0].clone();
    let (radical_t, radical_trace) = adam_fit(&train, tensor(&radical_start), adam_steps, 2e-3);
    let radical = weights(&radical_t);
    let mut radical_deep = Vec::new();
    for s in starts.iter().take(m) {
        let (t, _) = adam_fit(&train, tensor(s), adam_steps, 2e-3);
        radical_deep.push(weights(&t));
    }

    // (3) Knee-stopped ensemble and (4) SGD-SWA on member 0.
    let mut knee_members = Vec::new();
    let mut knee_records = Vec::new();
    let mut warm0 = None;
    for (i, s) in starts.iter().enumerate() {
        // Warm entry optimizes the declared loss L (per-row regression),
        // not Φ — the ceiling is a loss ceiling.
        let warm = warm_entry(&per_row, s, warm_cfg).expect("warm entry");
        knee_records.push(member_record(&sat, &warm.theta, seeds[i], &warm).expect("record"));
        knee_members.push(warm.theta.clone());
        if i == 0 {
            warm0 = Some(warm);
        }
    }
    let warm0 = warm0.expect("member 0");
    {
        let norm = |w: &[f32]| {
            w.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt()
        };
        let penalty = |w: &[f32]| {
            0.5 * w
                .iter()
                .zip(&prior_c)
                .map(|(v, c)| f64::from(*v).powi(2) / c)
                .sum::<f64>()
        };
        println!(
            "warm_norm={:.3} warm_prior_penalty={:.3e} start_norm={:.3} start_prior_penalty={:.3e}",
            norm(&warm0.theta),
            penalty(&warm0.theta),
            norm(&starts[0]),
            penalty(&starts[0])
        );
    }
    let mut swa_snapshots = Vec::new();
    let (t, _) = adam_fit(&train, tensor(&warm0.theta), 120, 2e-3);
    let mut swa_theta = weights(&t);
    for _ in 0..10 {
        let (t, _) = adam_fit(&train, tensor(&swa_theta), 10, 2e-3);
        swa_theta = weights(&t);
        swa_snapshots.push(swa_theta.clone());
    }
    let swa_center = batch_average(&swa_snapshots).expect("swa average");
    let swa_checkpoint = SwaCheckpoint::new(
        SwaKind::SgdConstantLr,
        0,
        swa_snapshots.len(),
        10,
        &swa_snapshots,
    )
    .expect("swa checkpoint");

    // (5) Sharp Gibbs pCNL and (6) satisficing pCNL ensembles.
    let chain_cfg = VectorSamplerConfig {
        steps: 300,
        burn: 60,
        thin: 12,
        rho: 0.9,
        refresh: 0.01,
        prior_sigma,
        draws: 20,
    };
    // The prior stays the model's layer-scaled prior; the CN proposal is the
    // R13-style shrunk pilot covariance (the 57k/3k-dim posterior is far
    // narrower than the prior).
    // A 1e-6 prior-scale proposal is what the measured geometry admits: the
    // feasible set is a thin radial shell in weight space (moving every weight
    // by 10 % leaves `L ≤ τ` and lands at Φ ≈ 1.8e3), so any prior-scale move
    // is rejected.  This is R13's curved-boundary stall, reported explicitly.
    let proposal_c: Vec<f64> = prior_c.iter().map(|c| c * 1e-6).collect();
    let sharp = Sharp {
        inner: &per_row,
        lambda: 300.0,
    };
    let mut sharp_members = Vec::new();
    let mut sharp_accept = (0usize, 0usize);
    for (i, _) in starts.iter().enumerate().take(3) {
        let chain = run_chain_with_covariances(
            &sharp,
            &prior_c,
            &proposal_c,
            chain_cfg,
            knee_members[i].clone(),
            seeds[i] + 100,
        )
        .expect("sharp chain");
        sharp_accept.0 += chain.accepted;
        sharp_accept.1 += chain.proposals;
        sharp_members.push(chain.theta);
    }
    let mut sat_members = Vec::new();
    let mut sat_accept = (0usize, 0usize);
    let mut swa_centers = vec![swa_checkpoint];
    for (i, _) in starts.iter().enumerate() {
        let chain = run_chain_with_covariances(
            &sat,
            &prior_c,
            &proposal_c,
            chain_cfg,
            knee_members[i].clone(),
            seeds[i] + 200,
        )
        .expect("satisficing chain");
        sat_accept.0 += chain.accepted;
        sat_accept.1 += chain.proposals;
        sat_members.push(chain.theta.clone());
        if chain.draws.len() >= 10 {
            let window = chain.draws[chain.draws.len() - 10..].to_vec();
            swa_centers.push(
                SwaCheckpoint::new(SwaKind::PcnlWithinBasin, i, 10, 12, &window)
                    .expect("within-basin swa"),
            );
        }
    }
    // Mobility probe: the same kernel at two proposal scales.  This is the
    // measured curved-boundary stall the plan asks to report rather than hide.
    for scale in [1e-6f64, 1e-8] {
        let proposal: Vec<f64> = prior_c.iter().map(|c| c * scale).collect();
        match run_chain_with_covariances(
            &sat,
            &prior_c,
            &proposal,
            chain_cfg,
            knee_members[0].clone(),
            4242,
        ) {
            Ok(chain) => {
                let moved = chain
                    .theta
                    .iter()
                    .zip(&knee_members[0])
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!(
                    "mobility scale={scale:.0e} accept={}/{} cn={}/{} moved_norm={moved:.6e}",
                    chain.accepted, chain.proposals, chain.cn_accepted, chain.cn_attempts
                );
            }
            Err(e) => println!("mobility scale={scale:.0e} error={e}"),
        }
    }
    // Geometry-aware alternative: exact hit-and-run slice moves draw a random
    // direction in prior-whitened coordinates and sample the target's exact
    // one-dimensional conditional, so the proposal has no gradient-scale knob
    // to tune and is π-invariant.  If the shell is reachable along random rays,
    // this is the move that gets there; if it is not, the stall is geometric
    // and not a pCN step-size artefact.
    match run_hit_and_run(
        &sat,
        &prior_c,
        HitRunConfig {
            moves: 300,
            burn: 60,
            thin: 12,
            draws: 20,
            width: 1.0,
            max_expand: 32,
            max_shrink: 64,
        },
        knee_members[0].clone(),
        7777,
    ) {
        Ok(chain) => {
            let moved = chain
                .theta
                .iter()
                .zip(&knee_members[0])
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>()
                .sqrt();
            println!(
                "mobility hitrun accept={}/{} moved_norm={moved:.6e} final_loss={:.6} in_region={}",
                chain.accepted,
                chain.proposals,
                loss_at(&train, &chain.theta),
                loss_at(&train, &chain.theta) <= config.tau
            );
        }
        Err(e) => println!("mobility hitrun error={e}"),
    }
    // The boundary layer is `O(1/√κ)` wide because the prior pulls inward while
    // the quadratic hinge pushes outward.  A ten-times stronger confinement is
    // the geometry-aware arm's setting: the same exact moves, with the target's
    // mass held closer to `L = τ`.
    let strict_cfg = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config
    };
    let sat_strict =
        Satisficing::new(per_row, strict_cfg, "vector-regression", "mean").expect("strict config");
    match run_hit_and_run(
        &sat_strict,
        &prior_c,
        HitRunConfig {
            moves: 300,
            burn: 60,
            thin: 12,
            draws: 20,
            width: 1.0,
            max_expand: 32,
            max_shrink: 64,
        },
        knee_members[0].clone(),
        8888,
    ) {
        Ok(chain) => {
            let moved = chain
                .theta
                .iter()
                .zip(&knee_members[0])
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>()
                .sqrt();
            let in_region = chain
                .draws
                .iter()
                .filter(|d| loss_at(&train, d) <= strict_cfg.tau)
                .count();
            println!(
                "mobility hitrun_strict kappa={} accept={}/{} moved_norm={moved:.6e} final_loss={:.6} in_region_draws={in_region}/{}",
                strict_cfg.kappa,
                chain.accepted,
                chain.proposals,
                loss_at(&train, &chain.theta),
                chain.draws.len()
            );
        }
        Err(e) => println!("mobility hitrun_strict error={e}"),
    }
    let mut ensemble = Ensemble::new("r14-p4-pilot");
    for (i, w) in sat_members.iter().enumerate() {
        ensemble
            .push(pcnl::ensemble::Member {
                weights: w.clone(),
                seed: seeds[i] + 200,
                warm_stop: knee_records[i].warm_stop,
                warm_steps: knee_records[i].warm_steps,
                train_loss: loss_at(&train, w),
                hinge: sat.config().psi(loss_at(&train, w) - config.tau),
            })
            .expect("ensemble member");
    }
    ensemble.swa_centers = swa_centers;
    let checkpoint_path = std::env::temp_dir().join("pcnl-r14-p4-pilot-ensemble.json");
    ensemble.save(&checkpoint_path).expect("save ensemble");
    let reloaded = Ensemble::load(&checkpoint_path).expect("load ensemble");
    let roundtrip = reloaded.to_json().unwrap() == ensemble.to_json().unwrap();
    let _ = std::fs::remove_file(&checkpoint_path);

    // (7) Distilled student and label-trained control.
    let sat_pred_train = ensemble_flat(&train, &sat_members);
    let teacher_mean_train = mean_prediction(&sat_pred_train);
    let teacher_mean_test = mean_prediction(&ensemble_flat(&test, &sat_members));
    let student_y: Vec<f32> = teacher_mean_train.iter().map(|v| *v as f32).collect();
    let student_target = VectorRegression::new(shapes, x_train.clone(), student_y, noise, lambda)
        .expect("distill target");
    let student_start = init_flat_prior(&shapes, prior_sigma, 777);
    let (student_t, _) = adam_fit(&student_target, student_start, adam_steps, 2e-3);
    let student = weights(&student_t);
    let control_start = init_flat_prior(&shapes, prior_sigma, 777);
    let (control_t, _) = adam_fit(&train, control_start, adam_steps, 2e-3);
    let control = weights(&control_t);

    // Spread head fitted to the ensemble's per-row spread.
    let mut means = Vec::new();
    let mut spreads = Vec::new();
    for row in 0..n_train {
        let row_preds: Vec<Vec<f32>> = sat_members
            .iter()
            .map(|w| {
                train
                    .predict_row(&tensor(w), row)
                    .expect("member prediction")
            })
            .collect();
        let stats: VectorEnsembleStats = vector_stats(&row_preds).expect("stats");
        means.push(stats.mean.iter().map(|v| f64::from(*v)).collect());
        spreads.push(stats.spread.iter().map(|v| f64::from(*v)).collect());
    }
    let spread_head = SpreadHead::fit(&means, &spreads, 400, 0.05).expect("spread head");
    let student_test = predict_rows(&test, &student);
    let mut student_test_spread = Vec::new();
    for row in &student_test {
        student_test_spread.extend(spread_head.variance(row).expect("spread at test"));
    }
    let sigma_test: Vec<f64> = student_test_spread
        .iter()
        .map(|v| (v + noise * noise).sqrt())
        .collect();
    let y_te = y_flat(&test);
    let coverage90 = normal_coverage(
        &flatten(&student_test),
        &sigma_test,
        &y_te.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        0.9,
    )
    .unwrap();

    // Reports.
    let report = |name: &str, members: &[Vec<f32>]| {
        let flat = ensemble_flat(&test, members);
        let pred = mean_prediction(&flat);
        let yf: Vec<f64> = y_te.iter().map(|v| f64::from(*v)).collect();
        let rmse = rmse_of(&pred, &yf).unwrap();
        let losses: Vec<f64> = members.iter().map(|w| loss_at(&train, w)).collect();
        let (ln, lmin, lmed, lmax) = pcnl::ensemble_diag::loss_summary(&losses).unwrap();
        let spread = prediction_spread(&flat).unwrap();
        let rank = effective_rank(&flat).unwrap();
        let sigma: Vec<f64> = spread
            .variance
            .iter()
            .map(|v| (v + noise * noise).sqrt())
            .collect();
        let nll = pcnl::ensemble_diag::gaussian_nll(
            &yf,
            &pred,
            &sigma.iter().map(|s| s * s).collect::<Vec<_>>(),
        )
        .unwrap();
        let cov = normal_coverage(&pred, &sigma, &yf, 0.9).unwrap();
        let z2 = normalized_residual_z2(&pred, &sigma, &yf).unwrap();
        let frac = in_region_fraction(&losses, config.tau).unwrap();
        println!(
            "arm={name} members={} rmse={rmse:.6} nll={nll:.6} coverage90={cov:.4} z2={z2:.4} in_region={frac:.3} loss_mean={ln:.4} loss_min={lmin:.4} loss_median={lmed:.4} loss_max={lmax:.4} functional_var={:.6e} disagreement={:.6e} eff_rank={rank:.4}",
            members.len(),
            spread.variance.iter().sum::<f64>() / spread.variance.len() as f64,
            spread.disagreement
        );
        (rmse, frac, rank, spread.disagreement, pred)
    };

    println!(
        "radical_single_trace0={:.6} trace_last={:.6}",
        radical_trace[0],
        radical_trace.last().unwrap()
    );
    let (rad_rmse, _, _, _, _) = report("radical-single", &[radical]);
    report("radical-deep", &radical_deep);
    // Knee members do not necessarily lie in-region after the same loss; report.
    let knee_losses: Vec<f64> = knee_members.iter().map(|w| loss_at(&train, w)).collect();
    println!(
        "warm_entry stops={:?} steps={:?} losses={:?}",
        knee_records
            .iter()
            .map(|r| r.warm_stop.label())
            .collect::<Vec<_>>(),
        knee_records
            .iter()
            .map(|r| r.warm_steps)
            .collect::<Vec<_>>(),
        knee_losses
    );
    let (knee_rmse, _, _, _, knee_pred) = report("knee-stopped", &knee_members);
    let (_, _, _, _, swa_pred) = report("sgd-swa-single", &[swa_center.clone()]);
    let (_, _, _, _, last_pred) = report("last-iterate", &[swa_snapshots.last().unwrap().clone()]);
    let (swa_vs_last, _) = paired_rmse_delta(&swa_pred, &last_pred, &y_te);
    let output_mean = mean_prediction(&ensemble_flat(&test, &swa_snapshots));
    let (swa_vs_output, _) = paired_rmse_delta(&swa_pred, &output_mean, &y_te);
    let barrier = straight_line_barrier(
        swa_snapshots.first().unwrap(),
        swa_snapshots.last().unwrap(),
        20,
        |w| loss_at(&train, w),
    )
    .expect("barrier");
    println!(
        "swa window={} spacing=10 weight_vs_last_rmse_delta={swa_vs_last:+.6} weight_vs_output_rmse_delta={swa_vs_output:+.6} barrier={:.6}",
        swa_snapshots.len(),
        barrier_height(&barrier).unwrap()
    );
    println!("sharp_gibbs accept={}/{}", sharp_accept.0, sharp_accept.1);
    println!(
        "satisficing accept={}/{} roundtrip={roundtrip}",
        sat_accept.0, sat_accept.1
    );
    let (sat_rmse, sat_frac, sat_rank, sat_dis, sat_pred) =
        report("satisficing-pcnl", &sat_members);
    report("prior-only", &prior_members);
    report("distilled-student", &[student.clone()]);
    let (lab_rmse, _, _, _, _) = report("label-student-control", &[control.clone()]);
    let (student_delta, student_se) = paired_rmse_delta(
        &flatten(&student_test),
        &flatten(&predict_rows(&test, &control)),
        &y_te,
    );
    let teacher_student = rmse_of(&flatten(&student_test), &teacher_mean_test).unwrap_or(f64::NAN);
    println!(
        "distillation student_rmse={sat_rmse:.6} label_control_rmse={lab_rmse:.6} paired_delta={student_delta:+.6} se={student_se:.6} teacher_student_rmse={teacher_student:.6} spread_coverage90={coverage90:.4}"
    );
    let (delta, se) = paired_rmse_delta(&sat_pred, &knee_pred, &y_te);
    println!(
        "contrast satisficing_vs_knee_rmse={:.6} satisficing_vs_radical_rmse={:.6} functional_rank={sat_rank:.4} in_region={sat_frac:.3} disagreement={sat_dis:.6e}",
        sat_rmse - knee_rmse,
        sat_rmse - rad_rmse
    );
    println!("paired satisficing_vs_knee delta={delta:+.6} se={se:.6} n={n_test}");
    println!("wall_clock_s={:.2}", started.elapsed().as_secs_f64());
}
