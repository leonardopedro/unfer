//! In-context adaptation via the Quantum Bayesian Update (Stage 5).
//!
//! The QFM-Text model's static prior is the dressed-vacuum
//! projector sum + Krylov evolution. The **Quantum Bayesian
//! Update** ([QFM.tex] §"Quantum Bayesian Updating", `qfm::bayes`)
//! turns a prefix of observed tokens into a *posterior* sample
//! over the unit sphere of C^m, then decodes that sample for the
//! next-token distribution. The mixing coefficient
//! `bayes_mix = 0.5` (default) interpolates between the static
//! distribution and the in-context-adapted one.
//!
//! The adaptation is a *sliding-window* posterior: we take the last
//! `cap_windows = 32` windows of the prefix, encode each into a
//! Krylov state, and treat them as a multi-observation likelihood.
//! One HMC sample (`sample_hmc_single`) gives a single posterior
//! `c_posterior` that we decode.
//!
//! [QFM.tex]: https://github.com/leonardopedro/unfer/blob/main/QFM.tex

use nalgebra::DVector;
use num_complex::Complex64;
use qfm::{HmcOpts, Likelihood, Posterior, sample_hmc_single, tsr_evolved_prior};

use crate::error::QfmTextError;
use crate::model::QfmTextModel;
#[allow(unused_imports)]
use crate::config::TextConfig; // used in tests

/// Options for the in-context update.
#[derive(Debug, Clone)]
pub struct HmcIncontextOpts {
    /// Mixing coefficient: 0 = pure static prior, 1 = pure
    /// posterior. Default: 0.5.
    pub bayes_mix: f64,
    /// Maximum number of prefix windows to fold into the posterior.
    /// Default: 32.
    pub cap_windows: usize,
    /// HMC step size. Default: 0.1.
    pub hmc_step: f64,
    /// HMC trajectory length. Default: 1.0.
    pub hmc_leapfrog_steps: usize,
}

impl Default for HmcIncontextOpts {
    fn default() -> Self {
        Self {
            bayes_mix: 0.5,
            cap_windows: 32,
            hmc_step: 0.1,
            hmc_leapfrog_steps: 20,
        }
    }
}

/// Adapt the static prior using a prefix of observed tokens. Returns
/// the posterior Krylov state `c_posterior`. The caller can decode it
/// via `QfmPipeline::decode_sketched` and marginalize against the
/// per-mode histograms.
pub fn adapt_prior(
    model: &QfmTextModel,
    prefix: &[u32],
    opts: &HmcIncontextOpts,
) -> Result<DVector<Complex64>, QfmTextError> {
    if prefix.is_empty() {
        // No observation: the posterior is the prior.
        return Ok(tsr_evolved_prior(&model.pipeline));
    }
    // 1. Build the multi-observation likelihood from the last
    //    `cap_windows` windows of the prefix.
    let n = model.cfg.n_orders;
    let windows: Vec<Vec<u32>> = if prefix.len() > n {
        prefix
            .windows(n + 1)
            .rev()
            .take(opts.cap_windows)
            .map(|w| w[..n].to_vec())
            .collect()
    } else {
        vec![prefix.to_vec()]
    };
    let mut likelihoods = Vec::with_capacity(windows.len());
    for ctx in windows {
        let encoded = model.pipeline.encode_modes(&ctx)?;
        likelihoods.push(Likelihood::from_krylov_state(encoded));
    }
    // 2. TSR-evolved prior.
    let c_prior = tsr_evolved_prior(&model.pipeline);
    // 3. HMC sample.
    let posterior = Posterior::new(likelihoods, c_prior.clone());
    let hmc = HmcOpts {
        step_size: opts.hmc_step,
        leapfrog_steps: opts.hmc_leapfrog_steps,
        n_iterations: 1,
        burn_in: 4,
        seed: 0,
    };
    let c_posterior = sample_hmc_single(&posterior, &hmc);
    Ok(c_posterior)
}

/// Next-token distribution with in-context adaptation. Interpolates
/// the static distribution and the posterior distribution via
/// `bayes_mix`.
pub fn next_token_dist_adapted(
    model: &QfmTextModel,
    prefix: &[u32],
) -> Result<Vec<f64>, QfmTextError> {
    let opts = HmcIncontextOpts::default();
    let active_modes = super::model::public_encode_modes(prefix, &model.cfg);
    let c_static = {
        let active = model.pipeline.encode_modes(&active_modes)?;
        model.pipeline.evolve(&active, model.cfg.t)
    };
    let p_static = model
        .pipeline
        .decode_sketched_at(&c_static, &model.gram, &active_modes);
    let c_post = adapt_prior(model, prefix, &opts)?;
    let p_post = model
        .pipeline
        .decode_sketched_at(&c_post, &model.gram, &active_modes);
    // Marginalize both against the per-mode histograms.
    let d_static = model.marginalize(&p_static);
    let d_post = model.marginalize(&p_post);
    let mix = opts.bayes_mix;
    let mut out = vec![0.0_f64; d_static.len()];
    for i in 0..out.len() {
        out[i] = (1.0 - mix) * d_static[i] + mix * d_post[i];
    }
    // Renormalize (defensive — the two marginals already sum to 1).
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for x in out.iter_mut() {
            *x /= sum;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::ChannelAccumulator;
    use crate::features::OrderHasher;

    fn cfg() -> TextConfig {
        TextConfig {
            n_orders: 2,
            block_sizes: vec![32, 32],
            salts: vec![1, 2],
            hist_cap: 4,
            max_rank: 4,
            m_shifts: 4,
            lambda: vec![1.0, 1.0],
            t: 1.0,
            discount: 0.5,
            seed: 0,
            ..Default::default()
        }
    }

    #[test]
    fn adapt_prior_returns_finite_vector() {
        // Build a small toy accumulator.
        let tokens: Vec<u32> = (0..300).map(|i| (i / 30) % 4).collect();
        let mut acc = ChannelAccumulator::new(0, cfg());
        let h = OrderHasher::new(cfg());
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let modes = h.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }
        let model = QfmTextModel::from_accumulator(acc, &cfg()).unwrap();
        let c_post = adapt_prior(&model, &tokens[..30], &HmcIncontextOpts::default()).unwrap();
        for x in c_post.iter() {
            assert!(x.norm().is_finite());
        }
    }

    #[test]
    fn next_token_dist_adapted_sums_to_one() {
        let tokens: Vec<u32> = (0..300).map(|i| (i / 30) % 4).collect();
        let mut acc = ChannelAccumulator::new(0, cfg());
        let h = OrderHasher::new(cfg());
        for i in 1..tokens.len() {
            let ctx: Vec<u32> = if i >= 2 {
                vec![tokens[i - 2], tokens[i - 1]]
            } else {
                vec![tokens[i - 1]]
            };
            let modes = h.encode_modes(&ctx);
            acc.observe(&modes, tokens[i]);
        }
        let model = QfmTextModel::from_accumulator(acc, &cfg()).unwrap();
        let dist = next_token_dist_adapted(&model, &tokens[..30]).unwrap();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
        for &p in &dist {
            assert!(p > 0.0, "zero probability in adapted dist");
        }
    }
}
