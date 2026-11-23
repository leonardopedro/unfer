//! R13 §4.2 continual-stream pilot: warm-started prefix vs fresh joint fit vs
//! current-batch-only reset on the four-block vector head.
//!
//! The 240 fixture training rows are cut into `--prefixes` contiguous blocks.
//! At each prefix boundary three predictors are trained with the same objective,
//! optimizer settings and prior:
//!
//! * `warm`   — Adam continues from the previous prefix's full parameter vector
//!              on every row seen so far;
//! * `joint`  — Adam restarts from the fixed initial vector on every row seen
//!              so far (the same data as `warm`);
//! * `reset`  — Adam restarts from the fixed initial vector on the **new block
//!              only** (the forgetting control).
//!
//! Each arm's prediction is the mean of its readout-conditional posterior
//! `N(m, A⁻¹)` with the trained backbone frozen, which for a Gaussian target is
//! the exact posterior mean (the sampler's draw mean in expectation) and avoids
//! spending the comparison budget on Monte Carlo noise.
//!
//! Reported per prefix: held-out test RMSE, per-coordinate test RMSE, retention
//! RMSE on every earlier block, and wall time per arm.
//!
//! Scope: the fixture exposes no document ids, so the blocks are contiguous row
//! ranges in fixture order, not verified document-ID splits.  Targets are
//! synthetic (fixed teacher plus known noise).  This is a stream *comparison*,
//! not a calibration or promotion result.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{Device, Tensor};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    ExactPosterior, ReadoutTarget, SyntheticTeacher, VectorRegression, VectorShapes, adam_fit,
    init_flat, last_hidden, predict, readout_range,
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}
struct Score {
    rmse: f64,
    per: Vec<f64>,
    /// 90% predictive-interval coverage, including the stated observation noise.
    coverage90: f64,
    /// Mean squared normalized residual `r²/(var + s²)`.  Under a calibrated
    /// Gaussian predictive this is 1; < 1 is over-coverage, > 1 under-coverage.
    z2_mean: f64,
    /// `(nominal, empirical)` coverage at 50/80/90/95%.
    calibration: Vec<(f64, f64)>,
}

/// Two-sided standard-normal quantiles for the calibration grid.
const CALIBRATION: [(f64, f64); 4] = [
    (0.50, 0.674_489_750_196_081_7),
    (0.80, 1.281_551_565_545_453_5),
    (0.90, 1.644_853_626_951_472_2),
    (0.95, 1.959_963_984_540_054),
];

/// RMSE, per-coordinate RMSE and 90% predictive coverage for a frozen backbone.
///
/// The readout-conditional posterior is Gaussian, so the predictive variance at
/// a row is `‖L⁻¹ f‖² + s²` with `f` the last hidden feature and `A = L Lᵀ` the
/// posterior precision: no sampling is needed to get an honest interval.
fn evaluate(
    theta: &[f32],
    shapes: &VectorShapes,
    x: &[f32],
    y: &[f32],
    post: Option<&ExactPosterior>,
    noise: f64,
) -> Score {
    let q = shapes.output;
    let rows = x.len() / shapes.input;
    let mut sq = vec![0.0f64; q];
    let mut covered = 0usize;
    let mut z2_sum = 0.0f64;
    let mut hits = vec![0usize; CALIBRATION.len()];
    let z90 = CALIBRATION[2].1;
    let cov = post.map(|p| p.dense());
    let theta_t = Tensor::from_vec(theta.to_vec(), shapes.flat_size(), &Device::Cpu).ok();
    for row in 0..rows {
        let e = &x[row * shapes.input..(row + 1) * shapes.input];
        let p = predict(theta, shapes, e).expect("predict");
        // Predictive sd per output coordinate from the feature vector.
        let mut var = vec![0.0f64; q];
        if let (Some(c), Some(t)) = (&cov, &theta_t) {
            let f = last_hidden(
                t,
                shapes,
                &Tensor::from_vec(e.to_vec(), shapes.input, &Device::Cpu).expect("row"),
            )
            .expect("features")
            .to_vec1::<f32>()
            .expect("features flat");
            let f64v: Vec<f64> = f.iter().map(|v| f64::from(*v)).collect();
            // Design vector per output coordinate over the 34 readout
            // parameters: W[i, j] gets f_i, the bias gets one.
            let dim = post.expect("posterior").dim();
            for j in 0..q {
                let mut d = vec![0.0f64; dim];
                for (i, fi) in f64v.iter().enumerate() {
                    d[i * q + j] = *fi;
                }
                d[shapes.width * q + j] = 1.0;
                let scaled = c.apply_cov_factor(&d);
                var[j] = scaled.iter().map(|s| s * s).sum::<f64>();
            }
        }
        for j in 0..q {
            let obs = f64::from(y[row * q + j]);
            let d = f64::from(p[j]) - obs;
            sq[j] += d * d;
            if post.is_some() {
                let sd = (var[j] + noise * noise).sqrt();
                let r = (obs - f64::from(p[j])).abs();
                z2_sum += (r / sd).powi(2);
                for (k, (_, z)) in CALIBRATION.iter().enumerate() {
                    if r <= z * sd {
                        hits[k] += 1;
                    }
                }
                if r <= z90 * sd {
                    covered += 1;
                }
            }
        }
    }
    let per: Vec<f64> = sq.iter().map(|s| (s / rows.max(1) as f64).sqrt()).collect();
    let total = (per.iter().map(|v| v * v).sum::<f64>() / q.max(1) as f64).sqrt();
    let points = (rows.max(1) * q) as f64;
    Score {
        rmse: total,
        per,
        coverage90: if post.is_some() {
            covered as f64 / points
        } else {
            f64::NAN
        },
        z2_mean: if post.is_some() {
            z2_sum / points
        } else {
            f64::NAN
        },
        calibration: if post.is_some() {
            CALIBRATION
                .iter()
                .zip(&hits)
                .map(|((nominal, _), hit)| (*nominal, *hit as f64 / points))
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture = PathBuf::from(
        arg(&args, "--fixture")
            .unwrap_or_else(|| "docs/fixtures/ultrabert_20ng4_embeddings.safetensors".into()),
    );
    let q: usize = arg(&args, "--outputs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let prefixes: usize = arg(&args, "--prefixes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(2);
    let adam_steps: usize = arg(&args, "--adam-steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let lr: f64 = arg(&args, "--lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02);
    let noise = 0.05;
    let prior_sigma: f64 = 0.05;
    let teacher_seed = 20261122u64;

    let table = EmbeddingTable::load(&fixture).expect("embedding fixture");
    let shapes = VectorShapes::new(table.dim(), q).expect("vector shapes");
    let teacher = SyntheticTeacher::new(table.dim(), q, teacher_seed);
    let train_x = table.x(Split::Train).to_vec();
    let train_y = teacher.targets(&train_x, noise, teacher_seed);
    let test_x = table.x(Split::Test).to_vec();
    let test_y = teacher.targets(&test_x, noise, teacher_seed ^ 0x55);

    let rows = train_x.len() / shapes.input;
    let block = rows / prefixes;
    assert!(block > 0, "not enough rows for {prefixes} prefixes");

    let prior_c_all: Vec<f64> = shapes
        .prior_scales()
        .iter()
        .map(|s| prior_sigma.powi(2) * s * s)
        .collect();
    let readout = readout_range(&shapes);
    let prior_c = prior_c_all[readout.clone()].to_vec();

    let scratch = init_flat(&shapes, 0.02, teacher_seed);
    let scratch_vec = scratch.to_vec1::<f32>().expect("flat init");
    let mut warm_theta = scratch_vec.clone();

    println!(
        "## R13 stream comparison: fixture={} rows={rows} block={block} prefixes={prefixes} q={q} adam_steps={adam_steps} lr={lr} noise={noise} prior_sigma={prior_sigma}",
        fixture.display()
    );
    println!("## split=contiguous_row_blocks (no document ids in the fixture)");

    // Per-prefix, per-arm test RMSE and the retention curve (mean RMSE over the
    // blocks seen so far), so the stream's forgetting shows as a curve rather
    // than a single number.
    let mut summary: Vec<(String, Vec<f64>)> = ["warm", "joint", "reset"]
        .iter()
        .map(|name| (name.to_string(), Vec::new()))
        .collect();
    let mut coverage: Vec<(String, Vec<f64>)> = ["warm", "joint", "reset"]
        .iter()
        .map(|name| (name.to_string(), Vec::new()))
        .collect();
    let mut retention_curve: Vec<(String, Vec<f64>)> = ["warm", "joint", "reset"]
        .iter()
        .map(|name| (name.to_string(), Vec::new()))
        .collect();
    let mut z2: Vec<(String, Vec<f64>)> = ["warm", "joint", "reset"]
        .iter()
        .map(|name| (name.to_string(), Vec::new()))
        .collect();
    let mut calib: Vec<(String, Vec<Vec<(f64, f64)>>)> = ["warm", "joint", "reset"]
        .iter()
        .map(|name| (name.to_string(), Vec::new()))
        .collect();

    for p in 1..=prefixes {
        let end = (p * block).min(rows);
        let start = end.saturating_sub(block);
        let seen_x = &train_x[..end * shapes.input];
        let seen_y = &train_y[..end * shapes.output];
        let new_x = &train_x[start * shapes.input..end * shapes.input];
        let new_y = &train_y[start * shapes.output..end * shapes.output];
        let seen =
            VectorRegression::new(shapes.clone(), seen_x.to_vec(), seen_y.to_vec(), noise, 1.0)
                .expect("seen target");
        let new = VectorRegression::new(shapes.clone(), new_x.to_vec(), new_y.to_vec(), noise, 1.0)
            .expect("new target");

        let mut arms: Vec<(&str, Vec<f32>, f64, ExactPosterior)> = Vec::new();
        for (name, target, init) in [
            ("warm", &seen, warm_theta.clone()),
            ("joint", &seen, scratch_vec.clone()),
            ("reset", &new, scratch_vec.clone()),
        ] {
            let t0 = Instant::now();
            let init_t =
                Tensor::from_vec(init, shapes.flat_size(), &scratch.device()).expect("init tensor");
            let (theta, _) = adam_fit(target, init_t, adam_steps, lr);
            let flat = theta.to_vec1::<f32>().expect("flat theta");
            // Readout-conditional exact posterior mean over the frozen backbone.
            let rt = ReadoutTarget::new(target.clone(), flat.clone()).expect("readout target");
            let post = rt.exact_posterior(&prior_c).expect("exact posterior");
            let mut full = flat;
            for (i, v) in post.mean().iter().enumerate() {
                full[readout.start + i] = *v as f32;
            }
            arms.push((name, full, t0.elapsed().as_secs_f64(), post));
        }

        let mut line = format!("prefix={p} rows={start}..{end}");
        for (name, theta, secs, post) in &arms {
            let s = evaluate(theta, &shapes, &test_x, &test_y, Some(post), noise);
            line.push_str(&format!(
                " {name}_test_rmse={s:.6} {name}_test_per_coord={:?} {name}_coverage90={:.4} {name}_z2={z2:.3} {name}_calib={:?} {name}_secs={t:.2}",
                s.per.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
                s.coverage90,
                s.calibration
                    .iter()
                    .map(|(n, c)| (*n, (c * 1e3).round() / 1e3))
                    .collect::<Vec<_>>(),
                t = secs,
                s = s.rmse,
                z2 = s.z2_mean,
            ));
            if let Some(slot) = z2.iter_mut().find(|(n, _)| n == name) {
                slot.1.push(s.z2_mean);
            }
            if let Some(slot) = calib.iter_mut().find(|(n, _)| n == name) {
                slot.1.push(s.calibration.clone());
            }
            if let Some(slot) = summary.iter_mut().find(|(n, _)| n == name) {
                slot.1.push(s.rmse);
            }
            if let Some(slot) = coverage.iter_mut().find(|(n, _)| n == name) {
                slot.1.push(s.coverage90);
            }
        }
        println!("{line}");

        // Retention: RMSE and 90% coverage on each earlier block.
        let mut ret_means: Vec<(&str, f64)> = Vec::new();
        for b in 0..p - 1 {
            let bs = b * block * shapes.input;
            let be = ((b + 1) * block).min(end) * shapes.input;
            let bx = &train_x[bs..be];
            let by =
                &train_y[b * shapes.output * block..((b + 1) * block).min(end) * shapes.output];
            let mut ret = format!("  retention block={b}");
            for (name, theta, _, post) in &arms {
                let s = evaluate(theta, &shapes, bx, by, Some(post), noise);
                ret.push_str(&format!(" {name}={:.6}(cov={:.3})", s.rmse, s.coverage90));
                match ret_means.iter_mut().find(|(n, _)| *n == *name) {
                    Some((_, acc)) => *acc += s.rmse,
                    None => ret_means.push((name, s.rmse)),
                }
            }
            println!("{ret}");
        }
        let denom = (p - 1).max(1) as f64;
        for (name, total) in &ret_means {
            if let Some(slot) = retention_curve.iter_mut().find(|(n, _)| n == name) {
                slot.1.push(total / denom);
            }
        }

        // The warm arm carries its full state into the next prefix.
        if let Some((_, theta, _, _)) = arms.iter().find(|(n, _, _, _)| *n == "warm") {
            warm_theta = theta.clone();
        }
    }

    for (name, values) in &summary {
        if values.is_empty() {
            continue;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let cov = coverage.iter().find(|(n, _)| n == name).map(|(_, v)| v);
        let ret = retention_curve
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v);
        println!(
            "arm={name} test_rmse_mean={mean:.6} per_prefix={:?}",
            values
                .iter()
                .map(|v| (v * 1e6).round() / 1e6)
                .collect::<Vec<_>>()
        );
        if let Some(cov) = cov {
            println!(
                "  arm={name} coverage90_per_prefix={:?}",
                cov.iter()
                    .map(|v| (v * 1e4).round() / 1e4)
                    .collect::<Vec<_>>()
            );
        }
        if let Some(ret) = ret {
            println!(
                "  arm={name} retention_curve_mean_per_prefix={:?}",
                ret.iter()
                    .map(|v| (v * 1e6).round() / 1e6)
                    .collect::<Vec<_>>()
            );
        }
        if let Some((_, values)) = z2.iter().find(|(n, _)| n == name) {
            if !values.is_empty() {
                println!(
                    "  arm={name} z2_per_prefix={:?}",
                    values
                        .iter()
                        .map(|v| (v * 1e3).round() / 1e3)
                        .collect::<Vec<_>>()
                );
            }
        }
        if let Some((_, prefixes)) = calib.iter().find(|(n, _)| n == name) {
            // Average the empirical coverage at each nominal level.
            if let Some(first) = prefixes.first() {
                let mut mean: Vec<(f64, f64)> = first.clone();
                for row in prefixes.iter().skip(1) {
                    for (slot, (_, c)) in mean.iter_mut().zip(row) {
                        slot.1 += c;
                    }
                }
                let n = prefixes.len() as f64;
                println!(
                    "  arm={name} calibration_mean={:?}",
                    mean.iter()
                        .map(|(nom, c)| (*nom, (c / n * 1e3).round() / 1e3))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    println!(
        "limitations: synthetic targets; fixture embeddings; contiguous row blocks (no document ids); readout-conditional posterior mean; not a calibration or promotion result"
    );
}
