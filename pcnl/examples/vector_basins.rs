//! R13 multimodality diagnostic: do distinct low-loss basins exist, and which
//! parameter segment separates them?
//!
//! The four-chain full-parameter HMC gate stalls at bulk ESS ≈ 5 with chains
//! reporting visibly different predictions.  This example tests the explanation
//! directly: it runs the mode finder (`adam_fit_energy`, the same sampled
//! potential the samplers use) from several independent inits and compares the
//! resulting parameter vectors segment by segment using [`VectorShapes::layout`].
//!
//! Two outcomes are informative:
//! * comparable potential and test error but large between-run distances →
//!   several near-equivalent basins, i.e. genuine multimodality;
//! * distances concentrated in one segment → that block is the one the chains
//!   disagree on.
//!
//! Scope: synthetic targets on the fixture embeddings; mode finding, not a
//! sampler; a large distance is evidence of non-identifiability, not a
//! convergence claim.

use std::path::PathBuf;

use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    ReadoutTarget, SyntheticTeacher, VectorRegression, VectorShapes, adam_fit_energy, init_flat,
    init_flat_prior, predict, readout_range,
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn test_rmse(theta: &[f32], shapes: &VectorShapes, x: &[f32], y: &[f32]) -> f64 {
    let q = shapes.output;
    let rows = x.len() / shapes.input;
    let mut sq = 0.0f64;
    for row in 0..rows {
        let e = &x[row * shapes.input..(row + 1) * shapes.input];
        let p = predict(theta, shapes, e).expect("predict");
        for j in 0..q {
            sq += (f64::from(p[j]) - f64::from(y[row * q + j])).powi(2);
        }
    }
    (sq / (rows.max(1) * q) as f64).sqrt()
}

/// Segment labels in `VectorShapes::layout` order: one per transformer block
/// (its six matrices), plus the projection, readout and bias.
fn segments(shapes: &VectorShapes) -> Vec<(String, usize, usize)> {
    let layout = shapes.layout();
    let mut out = vec![("projection".to_string(), layout[0].0, layout[0].1)];
    let per_block = 6usize;
    for b in 0..shapes.layers {
        let first = 1 + b * per_block;
        let start = layout[first].0;
        let len: usize = (first..first + per_block).map(|k| layout[k].1).sum();
        out.push((format!("block{}", b + 1), start, len));
    }
    let tail = 1 + shapes.layers * per_block;
    out.push(("readout".to_string(), layout[tail].0, layout[tail].1));
    out.push(("bias".to_string(), layout[tail + 1].0, layout[tail + 1].1));
    out
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
    let runs: usize = arg(&args, "--runs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(2);
    let steps: usize = arg(&args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000);
    let lr: f64 = arg(&args, "--lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5e-3);
    let init_std: f64 = arg(&args, "--init-std")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02);
    // Backbone init: `prior` (default) draws at the prior's own sd; `std` keeps
    // the legacy flat scale, which pins q/k at the uniform-attention fixed point.
    let init_mode = arg(&args, "--init-mode").unwrap_or_else(|| "prior".into());
    let noise = 0.05;
    let prior_sigma: f64 = 0.05;
    let teacher_seed = 20261122u64;

    let table = EmbeddingTable::load(&fixture).expect("embedding fixture");
    let shapes = VectorShapes::new(table.dim(), q).expect("vector shapes");
    let init = |seed: u64| -> Vec<f32> {
        let t = match init_mode.as_str() {
            "prior" => init_flat_prior(&shapes, prior_sigma, seed),
            "std" => init_flat(&shapes, init_std, seed),
            other => panic!("--init-mode must be prior or std, got {other}"),
        };
        t.to_vec1::<f32>().expect("flat init")
    };
    let teacher = SyntheticTeacher::new(table.dim(), q, teacher_seed);
    let train_x = table.x(Split::Train).to_vec();
    let train_y = teacher.targets(&train_x, noise, teacher_seed);
    let test_x = table.x(Split::Test).to_vec();
    let test_y = teacher.targets(&test_x, noise, teacher_seed ^ 0x55);
    let train = VectorRegression::new(shapes.clone(), train_x.clone(), train_y, noise, 1.0)
        .expect("train target");

    let prior_c: Vec<f64> = shapes
        .prior_scales()
        .iter()
        .map(|s| prior_sigma.powi(2) * s * s)
        .collect();
    let readout = readout_range(&shapes);
    let segs = segments(&shapes);

    println!(
        "## R13 multimodality diagnostic: rows={} dim={} runs={runs} steps={steps} lr={lr} init={init_mode} init_std={init_std}",
        train.rows(),
        shapes.flat_size()
    );

    // Which parameters actually receive gradient?  A matrix whose gradient is
    // identically zero is dead weight: it can never move, and a "distinct
    // block" that contains one is not distinct in behaviour.
    {
        let probe = init(20261122 ^ 0x91b);
        let (_, g) =
            pcnl::vector_model::EnergyTarget::energy_and_grad(&train, &probe).expect("gradient");
        for (i, (start, len, shape)) in shapes.layout().iter().enumerate() {
            let rms = (g[*start..*start + *len]
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                / *len as f64)
                .sqrt();
            println!(
                "init_grad index={i} shape={shape:?} rms={rms:.3e}{}",
                if rms == 0.0 { " ZERO" } else { "" }
            );
        }
    }

    let mut fits: Vec<(Vec<f32>, f64, f64, f64)> = Vec::new();
    for r in 0..runs {
        let theta0 = init(20261122 ^ (0x91b + r as u64 * 7919));
        let (mode, trace) = adam_fit_energy(&train, &prior_c, theta0, steps, lr);
        let (u, rms) = {
            let (u, g) =
                pcnl::vector_model::EnergyTarget::energy_and_grad(&train, &mode).expect("energy");
            let total = u + mode
                .iter()
                .enumerate()
                .map(|(i, t)| 0.5 * f64::from(*t).powi(2) / prior_c[i])
                .sum::<f64>();
            let rms = (g
                .iter()
                .enumerate()
                .map(|(i, g)| (f64::from(*g) + f64::from(mode[i]) / prior_c[i]).powi(2))
                .sum::<f64>()
                / g.len() as f64)
                .sqrt();
            (total, rms)
        };
        // Score the prediction the pipeline actually uses: readout-conditional
        // posterior mean with this backbone frozen.
        let rt = ReadoutTarget::new(train.clone(), mode.clone()).expect("readout target");
        let post = rt
            .exact_posterior(&prior_c[readout.clone()])
            .expect("exact posterior");
        let mut full = mode.clone();
        for (i, v) in post.mean().iter().enumerate() {
            full[readout.start + i] = *v as f32;
        }
        let rmse = test_rmse(&full, &shapes, &test_x, &test_y);
        println!(
            "run={r} steps={} potential={u:.6} grad_rms={rms:.3e} test_rmse={rmse:.6}",
            trace.len()
        );
        fits.push((full, u, rms, rmse));
    }

    // Pairwise segment-relative distances: RMS(Δ) / RMS(θ) per segment.
    let mut far_pairs = 0usize;
    for i in 0..fits.len() {
        for j in i + 1..fits.len() {
            let (a, b) = (&fits[i].0, &fits[j].0);
            let mut overall_sq = 0.0f64;
            let mut overall_ref = 0.0f64;
            let mut seg_rel: Vec<(String, f64)> = Vec::new();
            for (name, start, len) in &segs {
                let mut dsq = 0.0;
                let mut rsq = 0.0;
                for k in *start..*start + *len {
                    let d = f64::from(a[k]) - f64::from(b[k]);
                    dsq += d * d;
                    rsq += 0.25 * (f64::from(a[k]) + f64::from(b[k])).powi(2);
                }
                overall_sq += dsq;
                overall_ref += rsq;
                seg_rel.push((name.clone(), (dsq / rsq.max(1e-30)).sqrt()));
            }
            let rel = (overall_sq / overall_ref.max(1e-30)).sqrt();
            let worst = seg_rel
                .iter()
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .expect("segments");
            if rel > 0.1 {
                far_pairs += 1;
            }
            println!(
                "pair={i}-{j} rel_distance={rel:.4} worst_segment={} ({:.4}) per_segment=[{}]",
                worst.0,
                worst.1,
                seg_rel
                    .iter()
                    .map(|(n, v)| format!("{n}={v:.3}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    let potentials: Vec<f64> = fits.iter().map(|f| f.1).collect();
    let rmses: Vec<f64> = fits.iter().map(|f| f.3).collect();
    println!(
        "summary runs={} potential_range=[{:.4},{:.4}] test_rmse_range=[{:.4},{:.4}] pairs_above_0.1={far_pairs}/{}",
        fits.len(),
        potentials.iter().cloned().fold(f64::INFINITY, f64::min),
        potentials.iter().cloned().fold(0.0f64, f64::max),
        rmses.iter().cloned().fold(f64::INFINITY, f64::min),
        rmses.iter().cloned().fold(0.0f64, f64::max),
        fits.len() * (fits.len() - 1) / 2
    );
    println!(
        "note: mode fits from independent inits; large relative distance at comparable potential means non-identifiability, not sampler failure"
    );
}
