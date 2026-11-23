//! Reproducible dimension-scaling sweep for the R13 CN kernel.
//!
//! The target is the standard Gaussian `N(0, I_d)` (the kernel's Gaussian
//! reference contributes a negligible `1e6·I` prior, so `U` alone is the
//! target).  The proposal is the same kernel the R13 runs use,
//!
//! ```text
//! g(θ) = ρθ − (1−ρ) C ∇U(θ),      θ' ~ N(g, (1−ρ²) C),
//! ```
//!
//! with either a diagonal `C = c·I` (the frozen prior-scale proposal) or the
//! exact `C = I` (the analytic covariance, the `--proposal exact` R13 path).
//! Every cell reports acceptance, bulk ESS per retained draw, and the
//! closed-form comparison `acc ≈ exp(−6(1−ρ)d)` / `ESS/step ≈ acc(1−ρ)/2`
//! quoted in `RESEARCH_PLAN.md`, whose maximum over `ρ` is `1/(12 e d)`.
//!
//! ```text
//! cargo run --release -p pcnl --example cn_dimension_sweep -- \
//!   --dims 2,4,8,16,34,64,128,256 --rhos 0.5,0.9,0.99,0.999 --steps 2000
//! ```
//!
//! This measures the kernel on a controlled Gaussian; it is not a rerun of the
//! R13 posterior, whose proposal/target mismatch is reported separately in
//! `pcnl/RESEARCH_PLAN.md`.

use pcnl::vector_model::{EnergyTarget, ProposalCov, VectorSamplerConfig, run_chain_with_proposal};

/// Lag-1 autocorrelation of a trace.  For the CN kernel the retained-draw ACF
/// is approximately `accept · (ρ − (1−ρ)c)`, so `(1−acf)/(1+acf)` is the
/// quantity the `acc(1−ρ)/2` scaling law predicts, and unlike a truncated
/// Geyer sum it stays finite when the chain is nearly frozen.
fn acf1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if var <= 0.0 {
        return f64::NAN;
    }
    let cov: f64 = x
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>();
    cov / var * (n as f64) / (n - 1) as f64
}

/// `U(θ) = ½‖θ‖²`, i.e. the negative log density of `N(0, I_d)`.
struct StandardGaussian {
    dim: usize,
}

impl EnergyTarget for StandardGaussian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let u = 0.5
            * theta
                .iter()
                .map(|t| f64::from(*t) * f64::from(*t))
                .sum::<f64>();
        Some((u, theta.to_vec()))
    }
}

fn list<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Vec<T> {
    let text = args
        .iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string());
    text.split(',')
        .filter_map(|v| v.trim().parse::<T>().ok())
        .collect()
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
    let dims: Vec<usize> = list(&args, "--dims", "2,4,8,16,34,64,128,256");
    let rhos: Vec<f64> = list(&args, "--rhos", "0.5,0.9,0.99,0.999,0.9999");
    let steps: usize = list(&args, "--steps", "2000")
        .first()
        .copied()
        .unwrap_or(2000);
    let mismatch: f64 = list(&args, "--c", "0.1").first().copied().unwrap_or(0.1);
    let exact: bool = args.iter().any(|a| a == "--exact-proposal");
    if steps < 8 || dims.is_empty() || rhos.is_empty() {
        panic!("steps, dims and rhos must be usable");
    }
    if exact {
        println!(
            "sweep kernel=cn proposal=exact_covariance target=N(0,I_d) steps={steps} burn={}",
            steps / 4
        );
    } else {
        println!(
            "sweep kernel=cn proposal=diagonal c={mismatch} target=N(0,I_d) steps={steps} burn={}",
            steps / 4
        );
    }
    println!(
        "law_accept=exp(-6*(1-rho)*d) law_ess_per_step=law_accept*(1-rho)/2 ceiling=1/(12*e*d) ess_acf=(1-acf1)/(1+acf1)"
    );
    let burn = steps / 4;
    let draws = steps - burn;
    let probes = |d: usize| d.min(4).max(1);
    for &d in &dims {
        let target = StandardGaussian { dim: d };
        // A negligible kernel prior leaves the standard Gaussian as the target.
        let prior_c = vec![1e6f64; d];
        let proposal = if exact {
            ProposalCov::Diagonal(vec![1.0f64; d])
        } else {
            ProposalCov::Diagonal(vec![mismatch; d])
        };
        let mut best: Option<(f64, f64, f64)> = None; // (ess/step, rho, accept)
        for &rho in &rhos {
            let cfg = VectorSamplerConfig {
                steps,
                burn,
                thin: 1,
                rho,
                refresh: 0.0,
                prior_sigma: 1.0,
                draws,
            };
            let chain = run_chain_with_proposal(
                &target,
                &prior_c,
                &proposal,
                cfg,
                vec![0.0; d],
                20261122 ^ (d as u64) << 8 ^ (rho * 1e6) as u64,
            )
            .expect("sweep chain");
            let accept = chain.accepted as f64 / chain.proposals as f64;
            let k = probes(d);
            let mut ess_per_step = Vec::with_capacity(k);
            let mut acfs = Vec::with_capacity(k);
            for coord in 0..k {
                let trace: Vec<f64> = chain.draws.iter().map(|x| f64::from(x[coord])).collect();
                let a = acf1(&trace);
                if a.is_finite() && a > -1.0 {
                    acfs.push(a);
                    ess_per_step.push(((1.0 - a) / (1.0 + a)).max(0.0));
                }
            }
            ess_per_step.sort_by(f64::total_cmp);
            acfs.sort_by(f64::total_cmp);
            let ess = quantile(&ess_per_step, 0.5);
            let law_accept = (-6.0 * (1.0 - rho) * d as f64).exp();
            let law_ess = law_accept * (1.0 - rho) / 2.0;
            println!(
                "dim={d} rho={rho} accept={accept:.4} acf1={:.5} ess_acf_per_step={ess:.5} law_accept={law_accept:.4} law_ess_per_step={law_ess:.5} accepted={}/{}",
                quantile(&acfs, 0.5),
                chain.accepted,
                chain.proposals
            );
            if best.is_none_or(|(b, _, _)| ess > b) {
                best = Some((ess, rho, accept));
            }
        }
        if let Some((ess, rho, accept)) = best {
            let ceiling = 1.0 / (12.0 * std::f64::consts::E * d as f64);
            println!(
                "summary dim={d} best_rho={rho} best_ess_bulk_per_step={ess:.5} accept_at_best={accept:.4} ceiling_1_over_12ed={ceiling:.5}"
            );
        }
    }
}
