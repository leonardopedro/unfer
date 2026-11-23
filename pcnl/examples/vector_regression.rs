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
//! (`--pilot-inflate`, default 2.0).  The target is unchanged and every
//! production chain is exact MH for its given covariance.  The pilot does not
//! remove the CN kernel's dimension wall; see `RESEARCH_PLAN.md`.
//!
//! The readout conditional is exactly linear-Gaussian, so `--proposal exact`
//! uses the analytic posterior covariance as the CN proposal and `--init exact`
//! starts every chain from an exact posterior draw.  `--kernel hmc` switches to
//! fixed-length HMC with a dual-averaged step size during warmup; with
//! `--hmc-mass exact` the mass matrix is the analytic posterior variance.

use candle_core::{Device, Tensor};
use pcnl::diag::{ess_bulk, ess_tail, split_rhat_rank};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    EnergyTarget, ExactPosterior, HmcConfig, ProposalCov, ReadoutTarget, SyntheticTeacher,
    VectorChain, VectorRegression, VectorSamplerConfig, VectorShapes, adam_fit, init_flat, predict,
    readout_range, run_chain_with_posterior_reference, run_chain_with_prior,
    run_chain_with_proposal, run_hmc_chain, run_hmc_chain_whitened,
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
    let kernel = arg(&args, "--kernel").unwrap_or_else(|| "cn".into());
    if kernel != "cn" && kernel != "hmc" {
        panic!("--kernel must be cn or hmc, got {kernel}");
    }
    let proposal_mode = arg(&args, "--proposal").unwrap_or_else(|| "diagonal".into());
    if proposal_mode != "diagonal" && proposal_mode != "exact" {
        panic!("--proposal must be diagonal or exact, got {proposal_mode}");
    }
    let exact_proposal = proposal_mode == "exact";
    let init_mode = arg(&args, "--init").unwrap_or_else(|| "point".into());
    if init_mode != "point" && init_mode != "exact" && init_mode != "mean" {
        panic!("--init must be point, exact or mean, got {init_mode}");
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
    if !matches!(hmc_mass_mode.as_str(), "exact" | "diagonal" | "prior") {
        panic!("--hmc-mass must be exact, diagonal or prior, got {hmc_mass_mode}");
    }
    let hmc_fixed_step = args.iter().any(|a| a == "--no-hmc-adapt");

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
            theta: theta_vec.clone(),
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
    let want_pilot = pilot_steps > 0 && !exact_proposal;
    let proposal_c: Option<Vec<f64>> = if want_pilot {
        let pilot_cfg = VectorSamplerConfig {
            steps: pilot_steps,
            burn: pilot_steps / 4,
            thin: 1,
            rho: 0.9999,
            refresh: 0.01,
            prior_sigma,
            draws: (pilot_steps - pilot_steps / 4).max(1),
        };
        let pilot = run_chain_with_prior(
            &sampled,
            &prior_c,
            pilot_cfg,
            sampled.start(),
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
            "pilot steps={} draws={} accepted={}/{} rho=0.9999 var_over_prior p10={:.3e} p50={:.3e} p90={:.3e}",
            pilot_cfg.steps,
            pilot.draws.len(),
            pilot.accepted,
            pilot.proposals,
            quantile(&ratios, 0.1),
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.9)
        );
        Some(
            var.iter()
                .zip(&prior_c)
                .map(|(v, c)| (v * pilot_inflate).clamp(1e-8 * c, *c))
                .collect(),
        )
    } else {
        None
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
            sampled
                .start()
                .iter()
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
            };
            let hmc = match (hmc_mass_mode.as_str(), &exact) {
                ("exact", Some(post)) => {
                    run_hmc_chain_whitened(&sampled, &prior_c, &post.dense(), hmc_cfg, start, seed)
                }
                ("diagonal", Some(post)) => {
                    run_hmc_chain(&sampled, &prior_c, &post.variance(), hmc_cfg, start, seed)
                }
                _ => run_hmc_chain(&sampled, &prior_c, &prior_c, hmc_cfg, start, seed),
            }
            .expect("hmc chain");
            println!(
                "chain={chain_index} kernel=hmc mass={} leapfrog={} steps={} burn={} eps_initial={} eps_frozen={:.5} draws={} accepted={}/{} mean_accept={:.3} grad_evals={}",
                match (hmc_mass_mode.as_str(), exact.is_some()) {
                    ("exact", true) => "exact_dense",
                    ("diagonal", true) => "exact_diagonal",
                    _ => "prior_diagonal",
                },
                hmc_cfg.leapfrog,
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
    let draws: Vec<Tensor> = chains
        .iter()
        .flat_map(|c| c.draws.iter())
        .map(|d| Tensor::from_vec(d.clone(), shapes.flat_size(), &Device::Cpu).unwrap())
        .collect();
    let mut sq_mean = 0.0;
    let mut covered = 0usize;
    let mut covered_predictive = 0usize;
    let mut total = 0usize;
    for row in 0..test.rows() {
        let mut coords: Vec<Vec<f64>> = vec![Vec::with_capacity(draws.len()); q];
        for d in &draws {
            for (j, v) in test.predict_row(d, row).unwrap().into_iter().enumerate() {
                coords[j].push(f64::from(v));
            }
        }
        let y = &test.y()[row * q..(row + 1) * q];
        for j in 0..q {
            coords[j].sort_by(f64::total_cmp);
            let mean = coords[j].iter().sum::<f64>() / coords[j].len() as f64;
            let lo = quantile(&coords[j], 0.05);
            let hi = quantile(&coords[j], 0.95);
            let var =
                coords[j].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / coords[j].len() as f64;
            let z = 1.6448536269514722;
            let pred_sd = (var + test.noise() * test.noise()).sqrt();
            sq_mean += (mean - f64::from(y[j])).powi(2);
            covered += usize::from(f64::from(y[j]) >= lo && f64::from(y[j]) <= hi);
            covered_predictive += usize::from(
                f64::from(y[j]) >= mean - z * pred_sd && f64::from(y[j]) <= mean + z * pred_sd,
            );
            total += 1;
        }
    }
    println!(
        "posterior draw_mean_rmse={:.6} param_interval90_coverage={:.4} ({covered}/{total}) predictive_interval90_coverage={:.4} ({covered_predictive}/{total}) chains={chain_count} warm_started=true jitter={jitter} equilibrium_claim=false",
        (sq_mean / total as f64).sqrt(),
        covered as f64 / total as f64,
        covered_predictive as f64 / total as f64
    );
    if chain_count >= 2 {
        let obs_rows = 4.min(test.rows());
        let mut traces: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); chain_count]; obs_rows * q];
        for (chain_index, chain) in chains.iter().enumerate() {
            for row in 0..obs_rows {
                for d in &chain.draws {
                    for (j, v) in test
                        .predict_row(
                            &Tensor::from_vec(d.clone(), shapes.flat_size(), &Device::Cpu).unwrap(),
                            row,
                        )
                        .unwrap()
                        .into_iter()
                        .enumerate()
                    {
                        traces[row * q + j][chain_index].push(f64::from(v));
                    }
                }
            }
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
