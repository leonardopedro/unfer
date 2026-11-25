//! R13 §1/§4.2 pilot: synthetic signed-vector regression on fixed English
//! embeddings.
//!
//! This is deliberately not a text-performance claim.  The targets come from a
//! fixed synthetic teacher on the frozen UltraBERT fixture, so the run tests
//! the model, objective, checkpoint round-trip and posterior seam on English
//! embeddings without pretending an application supplied the vector schema.
//!
//! ```text
//! cargo run --release -p pcnl --example vector_regression -- \
//!   --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
//!   --checkpoint /tmp/pcnl-vector-checkpoint.json
//! ```
//!
//! `--readout-only` samples the 34 affine-readout coordinates with the backbone
//! frozen at its Adam point estimate.  That is a conditional target, and it is
//! labelled as such; it is not the 57,378-parameter posterior.
//!
//! `--pilot-steps N` runs a short high-`rho` pilot chain and fixes the
//! production proposal covariance from that pilot's per-coordinate variance
//! (`--pilot-inflate`, default 2.0, at `--pilot-rho`, default 0.9999).  The
//! target is unchanged and every production chain is exact MH for its given
//! covariance.  The pilot does not remove the CN kernel's dimension wall; see
//! `RESEARCH_PLAN.md`.
//!
//! The readout conditional is exactly linear-Gaussian, so `--proposal exact`
//! uses the analytic posterior covariance as the CN proposal and `--init exact`
//! starts every chain from an exact posterior draw.  `--kernel hmc` switches to
//! fixed-length HMC with a dual-averaged step size during warmup; with
//! `--hmc-mass exact` the metric is the analytic posterior precision, with
//! `--hmc-mass pilot|lowrank` it is built from the pilot chain (diagonal
//! precision, or `diag + rank-k` low-rank); `--hmc-jitter` randomizes the
//! trajectory length, and `--init mode` starts at a mode of the sampled
//! potential (which includes the prior that `adam_fit` omits).

use candle_core::{Device, Tensor};
use pcnl::diag::{ess_bulk, ess_tail, split_rhat_rank};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    EnergyTarget, ExactPosterior, HitRunConfig, HmcConfig, LowRankCovariance, ProposalCov,
    ReadoutTarget, SyntheticTeacher, TemperConfig, VectorChain, VectorRegression,
    VectorSamplerConfig, VectorShapes, adam_fit, adam_fit_energy, hessian_probe, init_flat,
    init_flat_prior, predict, readout_range, run_chain_with_posterior_reference,
    run_chain_with_prior, run_chain_with_proposal, run_hit_and_run, run_hmc_chain,
    run_hmc_chain_low_rank, run_hmc_chain_whitened,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Either the full flat parameter vector or the affine readout conditional on a
/// frozen backbone.  The kernel is generic over [`EnergyTarget`], so both go
/// through the same transition code.
enum Sampled {
    Full {
        target: VectorRegression,
        theta: Vec<f32>,
    },
    Readout(ReadoutTarget),
}

impl Sampled {
    fn start(&self) -> Vec<f32> {
        match self {
            Self::Full { theta, .. } => theta.clone(),
            Self::Readout(t) => t.start(),
        }
    }

    fn expand(&self, theta: &[f32]) -> Vec<f32> {
        match self {
            Self::Full { .. } => theta.to_vec(),
            Self::Readout(t) => t.expand(theta),
        }
    }

    /// Analytic Gaussian posterior; only the readout conditional has one.
    fn exact_posterior(&self, prior_c: &[f64]) -> Option<ExactPosterior> {
        match self {
            Self::Full { .. } => None,
            Self::Readout(t) => t.exact_posterior(prior_c).ok(),
        }
    }
}

impl EnergyTarget for Sampled {
    fn dim(&self) -> usize {
        match self {
            Self::Full { target, .. } => target.dim(),
            Self::Readout(t) => t.dim(),
        }
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        match self {
            Self::Full { target, .. } => target.energy_and_grad(theta),
            Self::Readout(t) => t.energy_and_grad(theta),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    schema: Vec<String>,
    shapes: VectorShapes,
    theta: Vec<f32>,
    lambda: f64,
    noise: f64,
    teacher_seed: u64,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn rmse(target: &VectorRegression, theta: &Tensor) -> Option<f64> {
    let mut sq = 0.0;
    for row in 0..target.rows() {
        let p = target.predict_row(theta, row).ok()?;
        let y = &target.y()[row * p.len()..(row + 1) * p.len()];
        for (a, b) in p.iter().zip(y) {
            sq += (f64::from(*a) - f64::from(*b)).powi(2);
        }
    }
    Some((sq / (target.rows() * target.shapes().output) as f64).sqrt())
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Posterior predictive summary for one split: draw-mean RMSE, the 90% band
/// (param and predictive) and the mean band half-width.
struct SplitScore {
    rmse: f64,
    halfwidth_mean: f64,
    covered_param: usize,
    covered_predictive: usize,
    total: usize,
}

fn score_split(target: &VectorRegression, draws: &[&Tensor], q: usize) -> SplitScore {
    let mut sq_mean = 0.0;
    let mut covered = 0usize;
    let mut covered_predictive = 0usize;
    let mut halfwidth = 0.0;
    let mut total = 0usize;
    for row in 0..target.rows() {
        let mut coords: Vec<Vec<f64>> = vec![Vec::with_capacity(draws.len()); q];
        for d in draws {
            for (j, v) in target.predict_row(d, row).unwrap().into_iter().enumerate() {
                coords[j].push(f64::from(v));
            }
        }
        let y = &target.y()[row * q..(row + 1) * q];
        for j in 0..q {
            coords[j].sort_by(f64::total_cmp);
            let mean = coords[j].iter().sum::<f64>() / coords[j].len() as f64;
            let lo = quantile(&coords[j], 0.05);
            let hi = quantile(&coords[j], 0.95);
            let var =
                coords[j].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / coords[j].len() as f64;
            let z = 1.6448536269514722;
            let pred_sd = (var + target.noise() * target.noise()).sqrt();
            sq_mean += (mean - f64::from(y[j])).powi(2);
            halfwidth += (hi - lo) / 2.0;
            covered += usize::from(f64::from(y[j]) >= lo && f64::from(y[j]) <= hi);
            covered_predictive += usize::from(
                f64::from(y[j]) >= mean - z * pred_sd && f64::from(y[j]) <= mean + z * pred_sd,
            );
            total += 1;
        }
    }
    SplitScore {
        rmse: (sq_mean / total as f64).sqrt(),
        halfwidth_mean: halfwidth / total as f64,
        covered_param: covered,
        covered_predictive,
        total,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture = PathBuf::from(
        arg(&args, "--fixture")
            .unwrap_or_else(|| "docs/fixtures/ultrabert_20ng4_embeddings.safetensors".into()),
    );
    let checkpoint_path = PathBuf::from(
        arg(&args, "--checkpoint").unwrap_or_else(|| "/tmp/pcnl-vector-checkpoint.json".into()),
    );
    let q: usize = arg(&args, "--outputs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let adam_steps: usize = arg(&args, "--adam-steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let sample_steps: usize = arg(&args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let rho: f64 = arg(&args, "--rho")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.99);
    let prior_sigma: f64 = arg(&args, "--prior-sigma")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let chain_count: usize = arg(&args, "--chains")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let jitter: f64 = arg(&args, "--start-jitter")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let thin: usize = arg(&args, "--thin")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
        .max(1);
    let requested_draws: usize = arg(&args, "--draws")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let proposal_scale: f64 = arg(&args, "--proposal-scale")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let pilot_steps: usize = arg(&args, "--pilot-steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let pilot_inflate: f64 = arg(&args, "--pilot-inflate")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let pilot_rho: f64 = arg(&args, "--pilot-rho")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.9999);
    let mode_steps: usize = arg(&args, "--mode-steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_500);
    let mode_lr: f64 = arg(&args, "--mode-lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2e-3);
    let kernel = arg(&args, "--kernel").unwrap_or_else(|| "cn".into());
    if kernel != "cn" && kernel != "hmc" && kernel != "hitrun" {
        panic!("--kernel must be cn, hmc or hitrun, got {kernel}");
    }
    let proposal_mode = arg(&args, "--proposal").unwrap_or_else(|| "diagonal".into());
    if proposal_mode != "diagonal" && proposal_mode != "exact" {
        panic!("--proposal must be diagonal or exact, got {proposal_mode}");
    }
    let exact_proposal = proposal_mode == "exact";
    let init_mode = arg(&args, "--init").unwrap_or_else(|| "point".into());
    if !matches!(init_mode.as_str(), "point" | "exact" | "mean" | "mode") {
        panic!("--init must be point, exact, mean or mode, got {init_mode}");
    }
    let init_exact = init_mode == "exact" || init_mode == "mean";
    let hmc_eps: f64 = arg(&args, "--hmc-eps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let hmc_leapfrog: usize = arg(&args, "--hmc-leapfrog")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);
    let hmc_mass_mode = arg(&args, "--hmc-mass").unwrap_or_else(|| "exact".into());
    if !matches!(
        hmc_mass_mode.as_str(),
        "exact" | "diagonal" | "prior" | "pilot" | "lowrank" | "hessdiag"
    ) {
        panic!(
            "--hmc-mass must be exact, diagonal, prior, pilot, lowrank or hessdiag, got {hmc_mass_mode}"
        );
    }
    let lowrank_rank: usize = arg(&args, "--lowrank-rank")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);
    let hmc_fixed_step = args.iter().any(|a| a == "--no-hmc-adapt");
    let hmc_jitter = args.iter().any(|a| a == "--hmc-jitter");
    // Parallel tempering: `--temper K` runs K replicas (cold one is the target).
    let temper_k: usize = arg(&args, "--temper")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let temper_min_beta: f64 = arg(&args, "--temper-min-beta")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let temper_swap: usize = arg(&args, "--temper-swap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // Hit-and-run slice move (`--kernel hitrun`).
    let hr_width: f64 = arg(&args, "--hr-width")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let hr_max_expand: usize = arg(&args, "--hr-max-expand")
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let hr_max_shrink: usize = arg(&args, "--hr-max-shrink")
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    let table = EmbeddingTable::load(&fixture).expect("embedding fixture");
    let dim = table.dim();
    let shapes = VectorShapes::new(dim, q).expect("vector shapes");
    let teacher_seed = 20261122u64;
    let noise = 0.05;
    let teacher = SyntheticTeacher::new(dim, q, teacher_seed);
    let train_y = teacher.targets(table.x(Split::Train), noise, teacher_seed);
    let test_y = teacher.targets(table.x(Split::Test), noise, teacher_seed ^ 0x55);
    let train = VectorRegression::new(shapes, table.x(Split::Train).to_vec(), train_y, noise, 1.0)
        .expect("train target");
    let test = VectorRegression::new(shapes, table.x(Split::Test).to_vec(), test_y, noise, 1.0)
        .expect("test target");

    println!(
        "## R13 synthetic vector regression: fixture={} train={} test={} dim={} q={} shapes={:?} teacher_seed={teacher_seed} noise={noise}",
        fixture.display(),
        train.rows(),
        test.rows(),
        dim,
        q,
        shapes
    );

    let theta0 = init_flat(&shapes, 0.02, 20261122);
    let (point, trace) = adam_fit(&train, theta0, adam_steps, 0.02);
    let point_train = rmse(&train, &point).unwrap();
    let point_test = rmse(&test, &point).unwrap();
    println!(
        "point adam_steps={adam_steps} first_loss={:.6} last_loss={:.6} train_rmse={point_train:.6} test_rmse={point_test:.6}",
        trace.first().copied().unwrap_or(f64::NAN),
        trace.last().copied().unwrap_or(f64::NAN)
    );

    let theta_vec = point.to_vec1::<f32>().unwrap();
    // Start scale for the full-parameter target.  `prior` draws θ at the
    // prior's own sd; the legacy flat 0.02 is 10× below it and leaves the
    // attention logits at ≈0, where the q/k gradient is second order and the
    // prior pins those projections at the uniform-attention fixed point.
    let start_scale = arg(&args, "--start-scale").unwrap_or_else(|| "prior".into());
    let init_vec: Vec<f32> = match start_scale.as_str() {
        "prior" => init_flat_prior(&shapes, prior_sigma, 20261122)
            .to_vec1::<f32>()
            .expect("flat init"),
        "legacy" => theta_vec.clone(),
        other => panic!("--start-scale must be prior or legacy, got {other}"),
    };
    let schema: Vec<String> = (0..q).map(|i| format!("coordinate_{i}")).collect();
    let checkpoint = Checkpoint {
        schema: schema.clone(),
        shapes,
        theta: theta_vec.clone(),
        lambda: train.lambda(),
        noise,
        teacher_seed,
    };
    std::fs::write(
        &checkpoint_path,
        serde_json::to_vec_pretty(&checkpoint).unwrap(),
    )
    .expect("write checkpoint");
    let reopened: Checkpoint =
        serde_json::from_slice(&std::fs::read(&checkpoint_path).expect("read checkpoint"))
            .expect("parse checkpoint");
    let first = test.predict_row(&point, 0).unwrap();
    let reopened_theta =
        Tensor::from_vec(reopened.theta.clone(), shapes.flat_size(), &Device::Cpu).unwrap();
    let first_reopened = test.predict_row(&reopened_theta, 0).unwrap();
    println!(
        "checkpoint path={} schema={:?} roundtrip_exact={} prediction={:?}",
        checkpoint_path.display(),
        reopened.schema,
        first == first_reopened,
        first_reopened
    );

    let cfg = VectorSamplerConfig {
        steps: sample_steps,
        burn: sample_steps / 4,
        thin,
        rho,
        refresh: 0.01,
        prior_sigma,
        draws: requested_draws.min(((sample_steps - sample_steps / 4) / thin).max(1)),
    };
    // The posterior is strongly anisotropic: one scalar proposal scale tuned
    // for the stiff directions barely moves the flat ones.  When asked, run a
    // short pilot chain at a high `rho` (where the small step is accepted),
    // estimate the per-coordinate posterior variance, and use that fixed
    // diagonal covariance for the production chains.  The pilot is proposal
    // adaptation only; the target is unchanged and each production chain is
    // an exact MH chain for the covariance it is given.
    let readout_only = args.iter().any(|a| a == "--readout-only");
    let sampled = if readout_only {
        Sampled::Readout(
            ReadoutTarget::new(train.clone(), theta_vec.clone()).expect("readout target"),
        )
    } else {
        Sampled::Full {
            target: train.clone(),
            theta: init_vec.clone(),
        }
    };
    println!(
        "sample_space={} dim={}",
        if readout_only {
            "readout_conditional"
        } else {
            "full_parameter"
        },
        sampled.dim()
    );
    let prior_scales = shapes.prior_scales();
    let all_prior_c: Vec<f64> = prior_scales
        .iter()
        .map(|s| prior_sigma.powi(2) * s * s)
        .collect();
    let prior_c: Vec<f64> = if readout_only {
        all_prior_c[readout_range(&shapes)].to_vec()
    } else {
        all_prior_c
    };
    let exact = sampled.exact_posterior(&prior_c);
    if (exact_proposal || init_exact) && exact.is_none() {
        panic!(
            "--proposal exact and --init exact require --readout-only: the analytic posterior exists only for the readout conditional"
        );
    }
    if let Some(post) = &exact {
        let mean_f32: Vec<f32> = post.mean().iter().map(|v| *v as f32).collect();
        let (_, grad) = sampled
            .energy_and_grad(&mean_f32)
            .expect("gradient at the posterior mean");
        let grad_rms = (grad
            .iter()
            .enumerate()
            .map(|(i, g)| (f64::from(*g) + f64::from(mean_f32[i]) / prior_c[i]).powi(2))
            .sum::<f64>()
            / grad.len() as f64)
            .sqrt();
        let mut ratios: Vec<f64> = post
            .variance()
            .iter()
            .zip(&prior_c)
            .map(|(v, c)| v / c)
            .collect();
        ratios.sort_by(f64::total_cmp);
        println!(
            "exact_posterior dim={} total_grad_rms_at_mean={:.3e} var_over_prior p10={:.3e} p50={:.3e} p90={:.3e}",
            post.dim(),
            grad_rms,
            quantile(&ratios, 0.1),
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.9)
        );
    }
    let mode_base: Option<Vec<f32>> = if init_mode == "mode" {
        let (mode, trace) =
            adam_fit_energy(&sampled, &prior_c, sampled.start(), mode_steps, mode_lr);
        let (_, grad) = sampled.energy_and_grad(&mode).expect("mode energy");
        let mut total = 0.0f64;
        for i in 0..mode.len() {
            let g = f64::from(grad[i]) + f64::from(mode[i]) / prior_c[i];
            total += g * g;
        }
        println!(
            "init_mode=mode adam_steps={} lr={mode_lr} final_potential={:.6} mode_grad_rms={:.6e}",
            trace.len(),
            trace.last().copied().unwrap_or(f64::NAN),
            (total / mode.len() as f64).sqrt()
        );
        Some(mode)
    } else {
        None
    };
    // The pilot covariance cannot see directions its short walk never visits.
    // The potential's Hessian at the mode can: a Hutchinson probe gives a
    // per-coordinate precision estimate in 2·probes gradient evaluations.
    let hessian_diag: Option<Vec<f64>> = if hmc_mass_mode == "hessdiag" {
        let base = mode_base
            .as_ref()
            .expect("--hmc-mass hessdiag requires --init mode");
        let probes: usize = arg(&args, "--hessian-probes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(48);
        let hstep: f64 = arg(&args, "--hessian-step")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-5);
        let t0 = std::time::Instant::now();
        let probe = hessian_probe(&sampled, &prior_c, base, probes, hstep, 20261122 ^ 0x4e55)
            .expect("hessian probe");
        // `U` always contains `½Σθ²/c`, so `1/c_i` is a hard lower bound on
        // every Hessian diagonal entry; estimates below it (or negative, which
        // a nonconvex mode produces) are not usable as a precision.  The
        // upper clamp bounds the metric's dynamic range, which otherwise
        // collapses the adapted step size.
        let clip: f64 = arg(&args, "--hessian-clip")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e4);
        let mut diag = probe.diag;
        let max = diag.iter().cloned().fold(0.0f64, f64::max);
        let mut floored = 0usize;
        let mut capped = 0usize;
        for (i, v) in diag.iter_mut().enumerate() {
            let lo = 1.0 / prior_c[i];
            if !(*v > lo) {
                *v = lo;
                floored += 1;
            } else if *v > clip * lo {
                *v = clip * lo;
                capped += 1;
            }
        }
        let mut sorted = diag.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "hessian_metric probes={} step={hstep:.1e} diag p10={:.3e} p50={:.3e} p90={:.3e} max={:.3e} clipped_to_prior={floored} clipped_to_{clip:.0}x_prior={capped} secs={:.1}",
            probes,
            quantile(&sorted, 0.1),
            quantile(&sorted, 0.5),
            quantile(&sorted, 0.9),
            max,
            t0.elapsed().as_secs_f64()
        );
        Some(diag)
    } else {
        None
    };
    let want_pilot = pilot_steps > 0 && !exact_proposal;
    let mut pilot_draws: Option<Vec<Vec<f32>>> = None;
    let proposal_c: Option<Vec<f64>> = if want_pilot {
        let pilot_cfg = VectorSamplerConfig {
            steps: pilot_steps,
            burn: pilot_steps / 4,
            thin: 1,
            rho: pilot_rho,
            refresh: 0.01,
            prior_sigma,
            draws: (pilot_steps - pilot_steps / 4).max(1),
        };
        let pilot = run_chain_with_prior(
            &sampled,
            &prior_c,
            pilot_cfg,
            mode_base.clone().unwrap_or_else(|| sampled.start()),
            20261122 ^ 0x7c1,
        )
        .expect("pilot chain");
        let n = pilot.draws.len().max(1) as f64;
        let mut mean = vec![0.0f64; prior_c.len()];
        for draw in &pilot.draws {
            for (i, v) in draw.iter().enumerate() {
                mean[i] += f64::from(*v);
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0f64; prior_c.len()];
        for draw in &pilot.draws {
            for (i, v) in draw.iter().enumerate() {
                var[i] += (f64::from(*v) - mean[i]).powi(2);
            }
        }
        for v in &mut var {
            *v = (*v / n).max(0.0);
        }
        let mut ratios: Vec<f64> = var.iter().zip(&prior_c).map(|(v, c)| v / c).collect();
        ratios.sort_by(f64::total_cmp);
        println!(
            "pilot steps={} draws={} accepted={}/{} rho={} var_over_prior p10={:.3e} p50={:.3e} p90={:.3e}",
            pilot_cfg.steps,
            pilot.draws.len(),
            pilot.accepted,
            pilot.proposals,
            pilot_rho,
            quantile(&ratios, 0.1),
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.9)
        );
        pilot_draws = Some(pilot.draws);
        Some(
            var.iter()
                .zip(&prior_c)
                .map(|(v, c)| (v * pilot_inflate).clamp(1e-8 * c, *c))
                .collect(),
        )
    } else {
        None
    };
    let lowrank_cov: Option<LowRankCovariance> = match (&pilot_draws, hmc_mass_mode.as_str()) {
        (Some(draws), "lowrank") => {
            let cov = LowRankCovariance::from_draws(draws, lowrank_rank, 1e-6)
                .expect("low-rank covariance");
            println!(
                "lowrank_metric rank={} dim={} pilot_draws={}",
                cov.rank(),
                cov.dim(),
                draws.len()
            );
            Some(cov)
        }
        _ => None,
    };
    let mut chains = Vec::with_capacity(chain_count);
    for chain_index in 0..chain_count {
        let mut rng = StdRng::seed_from_u64(20261122 ^ 0x9e37 ^ chain_index as u64);
        let start: Vec<f32> = if init_exact {
            let post = exact.as_ref().expect("checked above");
            if init_mode == "mean" {
                post.mean().iter().map(|v| *v as f32).collect()
            } else {
                let mut init_rng =
                    StdRng::seed_from_u64(20261122 ^ 0x1d17 ^ ((chain_index as u64) << 8));
                post.draw(&mut init_rng)
            }
        } else {
            let base: Vec<f32> = mode_base.clone().unwrap_or_else(|| sampled.start());
            base.iter()
                .map(|&v| {
                    let z: f64 = rng.sample(StandardNormal);
                    (f64::from(v) + jitter * z) as f32
                })
                .collect()
        };
        let (start_u, start_grad) = sampled.energy_and_grad(&start).expect("start energy");
        let grad_rms = (start_grad
            .iter()
            .map(|g| f64::from(*g).powi(2))
            .sum::<f64>()
            / start_grad.len() as f64)
            .sqrt();
        println!(
            "chain={chain_index} init={init_mode} start_energy={start_u:.6} start_grad_rms={grad_rms:.6e}"
        );
        let seed = 20261122 ^ 0x5a17 ^ ((chain_index as u64) << 8);
        let mut chain = if kernel == "hmc" {
            let hmc_cfg = HmcConfig {
                steps: sample_steps,
                burn: sample_steps / 4,
                thin,
                draws: requested_draws.min(((sample_steps - sample_steps / 4) / thin).max(1)),
                leapfrog: hmc_leapfrog,
                step_size: hmc_eps,
                adapt_step_size: !hmc_fixed_step,
                target_accept: 0.8,
                jitter_leapfrog: hmc_jitter,
            };
            let hmc = if temper_k > 1 {
                let ladder: Vec<f64> = (0..temper_k)
                    .map(|i| {
                        let frac = if temper_k > 1 {
                            (temper_k - 1 - i) as f64 / (temper_k - 1) as f64
                        } else {
                            0.0
                        };
                        temper_min_beta.powf(frac)
                    })
                    .collect();
                let mass: Vec<f64> = match hmc_mass_mode.as_str() {
                    "hessdiag" => hessian_diag
                        .clone()
                        .expect("--hmc-mass hessdiag requires --init mode"),
                    _ => prior_c.iter().map(|c| 1.0 / c).collect(),
                };
                let tcfg = TemperConfig {
                    steps: sample_steps,
                    burn: sample_steps / 4,
                    thin,
                    draws: requested_draws.min(((sample_steps - sample_steps / 4) / thin).max(1)),
                    leapfrog: hmc_leapfrog,
                    step_size: hmc_eps,
                    adapt_step_size: !hmc_fixed_step,
                    target_accept: 0.8,
                    ladder,
                    swap_interval: temper_swap,
                };
                let out = pcnl::vector_model::run_parallel_tempering(
                    &sampled, &prior_c, &mass, tcfg, start, seed,
                )
                .expect("parallel tempering");
                println!(
                    "chain={chain_index} kernel=tempering replicas={} ladder={:?} swaps={}/{} replica_accept={:?} replica_final_u={:?} eps_cold={:.3e} draws={} accepted={}/{} mean_accept={:.3}",
                    temper_k,
                    (0..temper_k)
                        .map(|i| {
                            let frac = (temper_k - 1 - i) as f64 / (temper_k - 1) as f64;
                            (temper_min_beta.powf(frac) * 1e4).round() / 1e4
                        })
                        .collect::<Vec<_>>(),
                    out.swap_accepted,
                    out.swaps,
                    out.replica_accept
                        .iter()
                        .map(|v| (v * 1e3).round() / 1e3)
                        .collect::<Vec<_>>(),
                    out.replica_final_u
                        .iter()
                        .map(|v| (v * 1e2).round() / 1e2)
                        .collect::<Vec<_>>(),
                    out.cold.step_size,
                    out.cold.draws.len(),
                    out.cold.accepted,
                    out.cold.proposals,
                    out.cold.mean_accept
                );
                out.cold
            } else {
                match (hmc_mass_mode.as_str(), &exact) {
                    ("exact", Some(post)) => run_hmc_chain_whitened(
                        &sampled,
                        &prior_c,
                        &post.dense(),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                    // `run_hmc_chain` takes the metric `M` (mass), which must be
                    // the precision so that `M⁻¹A` has unit scale.
                    ("diagonal", Some(post)) => run_hmc_chain(
                        &sampled,
                        &prior_c,
                        &post.variance().iter().map(|v| 1.0 / v).collect::<Vec<_>>(),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                    ("hessdiag", _) => run_hmc_chain(
                        &sampled,
                        &prior_c,
                        hessian_diag
                            .as_ref()
                            .expect("--hmc-mass hessdiag requires --init mode"),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                    ("lowrank", _) => run_hmc_chain_low_rank(
                        &sampled,
                        &prior_c,
                        lowrank_cov
                            .as_ref()
                            .expect("--hmc-mass lowrank requires --pilot-steps"),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                    ("pilot", _) => run_hmc_chain(
                        &sampled,
                        &prior_c,
                        &proposal_c
                            .as_ref()
                            .expect("--hmc-mass pilot requires --pilot-steps")
                            .iter()
                            .map(|v| 1.0 / v)
                            .collect::<Vec<_>>(),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                    _ => run_hmc_chain(
                        &sampled,
                        &prior_c,
                        &prior_c.iter().map(|v| 1.0 / v).collect::<Vec<_>>(),
                        hmc_cfg,
                        start,
                        seed,
                    ),
                }
                .expect("hmc chain")
            };
            println!(
                "chain={chain_index} kernel=hmc mass={} leapfrog={} jitter={} steps={} burn={} eps_initial={} eps_frozen={:.3e} draws={} accepted={}/{} mean_accept={:.3} grad_evals={}",
                match (hmc_mass_mode.as_str(), exact.is_some()) {
                    ("exact", true) => "exact_dense",
                    ("diagonal", true) => "exact_diagonal",
                    ("pilot", _) => "pilot_diagonal",
                    ("lowrank", _) => "lowrank_plus_diag",
                    ("hessdiag", _) => "hessian_diagonal",
                    _ => "prior_diagonal",
                },
                hmc_cfg.leapfrog,
                hmc_cfg.jitter_leapfrog,
                hmc_cfg.steps,
                hmc_cfg.burn,
                hmc_cfg.step_size,
                hmc.step_size,
                hmc.draws.len(),
                hmc.accepted,
                hmc.proposals,
                hmc.mean_accept,
                hmc.grad_evals
            );
            VectorChain {
                theta: hmc.theta,
                draws: hmc.draws,
                proposals: hmc.proposals,
                accepted: hmc.accepted,
                cn_attempts: 0,
                cn_accepted: 0,
                refresh_attempts: 0,
                refresh_accepted: 0,
            }
        } else if kernel == "hitrun" {
            let hr_cfg = HitRunConfig {
                moves: sample_steps,
                burn: sample_steps / 4,
                thin,
                draws: requested_draws.min(((sample_steps - sample_steps / 4) / thin).max(1)),
                width: hr_width,
                max_expand: hr_max_expand,
                max_shrink: hr_max_shrink,
            };
            let hr = run_hit_and_run(&sampled, &prior_c, hr_cfg.clone(), start, seed)
                .expect("hit-and-run chain");
            println!(
                "chain={chain_index} kernel=hitrun moves={} burn={} width={} draws={} moved={}/{} mean_step={:.3} potential_evals={}",
                hr_cfg.moves,
                hr_cfg.burn,
                hr_cfg.width,
                hr.draws.len(),
                hr.accepted,
                hr.proposals,
                hr.step_size,
                hr.grad_evals
            );
            VectorChain {
                theta: hr.theta,
                draws: hr.draws,
                proposals: hr.proposals,
                accepted: hr.accepted,
                cn_attempts: 0,
                cn_accepted: 0,
                refresh_attempts: 0,
                refresh_accepted: 0,
            }
        } else {
            let cn = if exact_proposal {
                let post = exact.as_ref().expect("checked above");
                run_chain_with_posterior_reference(
                    &sampled,
                    &prior_c,
                    post.mean(),
                    &post.proposal(),
                    cfg,
                    start,
                    seed,
                )
                .expect("sampler chain")
            } else {
                let c = proposal_c.as_ref().unwrap_or(&prior_c);
                let proposal = ProposalCov::Diagonal(
                    c.iter()
                        .map(|v| v * proposal_scale * proposal_scale)
                        .collect(),
                );
                run_chain_with_proposal(&sampled, &prior_c, &proposal, cfg, start, seed)
                    .expect("sampler chain")
            };
            println!(
                "chain={chain_index} kernel=cn proposal={} proposal_scale={} steps={} burn={} rho={} refresh={} draws={} proposals={} accepted={} cn={}/{} refresh={}/{}",
                if exact_proposal {
                    "exact_dense"
                } else {
                    "diagonal"
                },
                proposal_scale,
                cfg.steps,
                cfg.burn,
                cfg.rho,
                cfg.refresh,
                cn.draws.len(),
                cn.proposals,
                cn.accepted,
                cn.cn_accepted,
                cn.cn_attempts,
                cn.refresh_accepted,
                cn.refresh_attempts
            );
            cn
        };
        // Diagnostics and prediction always see full flat parameters.
        chain.draws = chain.draws.iter().map(|d| sampled.expand(d)).collect();
        let mut draw_mean = vec![0.0f64; q];
        for draw in &chain.draws {
            let t = Tensor::from_vec(draw.clone(), shapes.flat_size(), &Device::Cpu).unwrap();
            for (j, v) in test.predict_row(&t, 0).unwrap().into_iter().enumerate() {
                draw_mean[j] += f64::from(v);
            }
        }
        if !chain.draws.is_empty() {
            for v in &mut draw_mean {
                *v /= chain.draws.len() as f64;
            }
        }
        println!("chain={chain_index} first_test_draw_mean={draw_mean:?}");
        chains.push(chain);
    }

    if chains.iter().any(|c| c.draws.is_empty()) {
        println!("posterior predictive unavailable: a chain retained no draws");
        return;
    }
    // Each draw is a full flat parameter vector (~229 KB at the 57,378-
    // coordinate size). Convert the chain buffers to tensors in place and drain
    // the originals: keeping both would double peak memory and OOM the 24k
    // length probe.
    let chain_draws: Vec<usize> = chains.iter().map(|c| c.draws.len()).collect();
    let mut draws: Vec<Tensor> = Vec::with_capacity(chain_draws.iter().sum());
    for chain in chains.iter_mut() {
        for draw in chain.draws.drain(..) {
            draws.push(Tensor::from_vec(draw, shapes.flat_size(), &Device::Cpu).unwrap());
        }
    }
    let draw_refs: Vec<&Tensor> = draws.iter().collect();
    let test_score = score_split(&test, &draw_refs, q);
    println!(
        "posterior draw_mean_rmse={:.6} param_interval90_coverage={:.4} ({}/{}) predictive_interval90_coverage={:.4} ({}/{}) chains={chain_count} warm_started=true jitter={jitter} equilibrium_claim=false",
        test_score.rmse,
        test_score.covered_param as f64 / test_score.total as f64,
        test_score.covered_param,
        test_score.total,
        test_score.covered_predictive as f64 / test_score.total as f64,
        test_score.covered_predictive,
        test_score.total
    );
    // The readout conditional is the only split an exact likelihood conditions
    // on, so score the same 90% bands on it (train) as well as on the held-out
    // split above.  A calibrated sampler must reproduce the analytic Gaussian
    // here; the residual sd is the noise-scale estimate that split can give.
    if readout_only {
        // A band from every draw is the right estimator but O(draws) forward
        // passes per observation; on the 24k length probe that dominates the
        // run.  A deterministic evenly spaced subsample keeps the check
        // honest (it is still the posterior's own draws) at fixed cost.
        const BAND_DRAWS: usize = 2000;
        let band_draws: Vec<&Tensor> = if draw_refs.len() <= BAND_DRAWS {
            draw_refs.clone()
        } else {
            (0..BAND_DRAWS)
                .map(|i| draw_refs[i * draw_refs.len() / BAND_DRAWS])
                .collect()
        };
        let train_score = score_split(&train, &band_draws, q);
        let params = sampled.dim() as f64;
        let dof = train_score.total as f64 - params;
        let sse = train_score.rmse.powi(2) * train_score.total as f64;
        let residual_sd_df = if dof > 0.0 {
            (sse / dof).sqrt()
        } else {
            f64::NAN
        };
        println!(
            "calibration conditioning_split=train rows={} outputs={q} draws={} band_draws={} param_interval90_coverage={:.4} ({}/{}) predictive_interval90_coverage={:.4} ({}/{}) band_halfwidth90_mean={:.6} residual_sd={:.6} residual_sd_df_adjusted={:.6} df={:.0} assumed_likelihood_noise={:.6} sampled_params={}",
            train.rows(),
            draws.len(),
            band_draws.len(),
            train_score.covered_param as f64 / train_score.total as f64,
            train_score.covered_param,
            train_score.total,
            train_score.covered_predictive as f64 / train_score.total as f64,
            train_score.covered_predictive,
            train_score.total,
            train_score.halfwidth_mean,
            train_score.rmse,
            residual_sd_df,
            dof,
            train.noise(),
            sampled.dim()
        );
    }
    if chain_count >= 2 {
        let obs_rows = 4.min(test.rows());
        let mut traces: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); chain_count]; obs_rows * q];
        let mut offset = 0usize;
        for (chain_index, &count) in chain_draws.iter().enumerate() {
            let chain_slice = &draws[offset..offset + count];
            for row in 0..obs_rows {
                for d in chain_slice {
                    for (j, v) in test
                        .predict_row(d, row)
                        .unwrap()
                        .into_iter()
                        .enumerate()
                    {
                        traces[row * q + j][chain_index].push(f64::from(v));
                    }
                }
            }
            offset += count;
        }
        let rhats: Vec<f64> = traces
            .iter()
            .map(|t| split_rhat_rank(t))
            .filter(|v| v.is_finite())
            .collect();
        let bulk: Vec<f64> = traces
            .iter()
            .map(|t| ess_bulk(t))
            .filter(|v| v.is_finite())
            .collect();
        let tail: Vec<f64> = traces
            .iter()
            .map(|t| ess_tail(t))
            .filter(|v| v.is_finite())
            .collect();
        for (index, trace) in traces.iter().enumerate() {
            let mean = trace
                .iter()
                .map(|chain| chain.iter().sum::<f64>() / chain.len() as f64)
                .collect::<Vec<_>>();
            println!(
                "diag_observable={index} rhat_rank={:.5} ess_bulk={:.1} ess_tail={:.1} chain_means={mean:?}",
                split_rhat_rank(trace),
                ess_bulk(trace),
                ess_tail(trace)
            );
        }
        let rhat_max = rhats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bulk_min = bulk.iter().copied().fold(f64::INFINITY, f64::min);
        let tail_min = tail.iter().copied().fold(f64::INFINITY, f64::min);
        let gate =
            chain_count >= 4 && rhats.len() == traces.len() && rhat_max < 1.01 && bulk_min >= 400.0;
        println!(
            "chain_diagnostics observables={} rhat_rank_max={rhat_max:.5} ess_bulk_min={bulk_min:.1} ess_tail_min={tail_min:.1} gate={gate} gate_definition=4_chains_rank_rhat<1.01_bulk_ess>=400",
            traces.len()
        );
    } else {
        println!("chain_diagnostics unavailable: need at least two chains");
    }
    println!(
        "limitations: synthetic targets; fixture embeddings, not live text; sample_space={}; warm-started jittered chains; not a promotion or calibration result",
        if readout_only {
            "readout_conditional_34d"
        } else {
            "full_parameter_57378d"
        }
    );

    // Keep the public flat-vector path exercised even when the caller only
    // inspects the JSON checkpoint.
    let direct =
        predict(&reopened.theta, &shapes, &table.x(Split::Test)[..dim]).expect("flat prediction");
    assert_eq!(direct, first_reopened);
}
