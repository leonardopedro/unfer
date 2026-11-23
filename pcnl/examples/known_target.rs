//! R13 §4.1 known-target check: unequal two-mode Gaussian mixture.
//!
//! The target is exactly `0.75 N(-2,1) + 0.25 N(2,1)` on one coordinate.
//! With prior `P0 = N(0,1)`, the sampler's potential is
//! `U = -log(pi / p0)`, so `exp(-U) P0 = pi`.  Four chains start in opposite
//! tails.  The reported basin masses, means, rank-normalized split R-hat and
//! bulk/tail ESS come from retained draws after burn-in.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example known_target
//! ```

use pcnl::diag::{ess_bulk, ess_tail, split_rhat_rank};
use pcnl::vector_model::{EnergyTarget, VectorChain, VectorSamplerConfig, run_chain_with_prior};

#[derive(Clone, Copy)]
struct Mixture;

const WEIGHTS: [f64; 2] = [0.75, 0.25];
const MEANS: [f64; 2] = [-2.0, 2.0];
const SD: f64 = 1.0;

fn log_norm(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn log_mixture(x: f64) -> f64 {
    let terms = [
        WEIGHTS[0].ln() + log_norm(x, MEANS[0], SD),
        WEIGHTS[1].ln() + log_norm(x, MEANS[1], SD),
    ];
    let m = terms[0].max(terms[1]);
    m + ((terms[0] - m).exp() + (terms[1] - m).exp()).ln()
}

impl EnergyTarget for Mixture {
    fn dim(&self) -> usize {
        1
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let x = f64::from(*theta.first()?);
        if !x.is_finite() {
            return None;
        }
        let pi = log_mixture(x);
        // p0 = N(0,1)
        let p0 = log_norm(x, 0.0, 1.0);
        let u = -(pi - p0);
        let terms = [
            WEIGHTS[0].ln() + log_norm(x, MEANS[0], SD),
            WEIGHTS[1].ln() + log_norm(x, MEANS[1], SD),
        ];
        let m = terms[0].max(terms[1]);
        let z = (terms[0] - m).exp() + (terms[1] - m).exp();
        let r0 = (terms[0] - m).exp() / z;
        let r1 = (terms[1] - m).exp() / z;
        let dpi = r0 * (-(x - MEANS[0]) / (SD * SD)) + r1 * (-(x - MEANS[1]) / (SD * SD));
        let dp0 = -x;
        let du = -dpi + dp0;
        Some((u, vec![du as f32]))
    }
}

fn exact_mass_left() -> f64 {
    // Phi((0 - mean)/sd)
    let normal_cdf = |z: f64| 0.5 * libm_erfc(-z / std::f64::consts::SQRT_2);
    WEIGHTS[0] * normal_cdf((0.0 - MEANS[0]) / SD) + WEIGHTS[1] * normal_cdf((0.0 - MEANS[1]) / SD)
}

// A local erfc approximation is unnecessary: the standard normal CDF is
// available through the complementary error function in `f64::erf`? Rust's
// std does not expose it, so use the high-accuracy Abramowitz-Stegun 7.1.26
// polynomial for this independent reference check.
fn libm_erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 { 1.0 - erf } else { 1.0 + erf }
}

fn central_interval(chains: &[VectorChain]) -> (f64, f64) {
    let mut xs: Vec<f64> = chains
        .iter()
        .flat_map(|c| c.draws.iter().map(|d| f64::from(d[0])))
        .collect();
    xs.sort_by(f64::total_cmp);
    let q = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    (q(0.05), q(0.95))
}

fn main() {
    let target = Mixture;
    let cfg = VectorSamplerConfig {
        steps: 50_000,
        burn: 10_000,
        thin: 5,
        rho: 0.9,
        refresh: 0.01,
        prior_sigma: 1.0,
        draws: 8_000,
    };
    let starts = [-6.0f32, -3.0, 3.0, 6.0];
    let seeds = [11u64, 12, 13, 14];
    let mut chains = Vec::new();
    for (i, (&start, &seed)) in starts.iter().zip(&seeds).enumerate() {
        let chain = run_chain_with_prior(&target, &[1.0], cfg, vec![start], seed)
            .expect("known-target chain");
        let left =
            chain.draws.iter().filter(|d| d[0] < 0.0).count() as f64 / chain.draws.len() as f64;
        let mean =
            chain.draws.iter().map(|d| f64::from(d[0])).sum::<f64>() / chain.draws.len() as f64;
        println!(
            "chain={i} start={start:+.1} seed={seed} draws={} accepted={}/{} mass_left={left:.5} mean={mean:+.5}",
            chain.draws.len(),
            chain.accepted,
            chain.proposals
        );
        chains.push(chain);
    }

    let traces: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.draws.iter().map(|d| f64::from(d[0])).collect())
        .collect();
    let indicator: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| {
            c.draws
                .iter()
                .map(|d| if d[0] < 0.0 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let pooled: Vec<f64> = traces.iter().flatten().copied().collect();
    let mass = pooled.iter().filter(|&&x| x < 0.0).count() as f64 / pooled.len() as f64;
    let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
    let (lo, hi) = central_interval(&chains);
    let rhat_x = split_rhat_rank(&traces);
    let rhat_ind = split_rhat_rank(&indicator);
    let ess_x_bulk = ess_bulk(&traces);
    let ess_x_tail = ess_tail(&traces);
    let ess_i_bulk = ess_bulk(&indicator);
    let ess_i_tail = ess_tail(&indicator);
    let exact = exact_mass_left();
    let mcse = (mass * (1.0 - mass) / ess_i_bulk.max(1.0)).sqrt();
    println!(
        "pooled draws={} mass_left={mass:.5} exact_mass_left={exact:.5} mass_error={:+.5} mass_mcse={mcse:.5} mean={mean:+.5} exact_mean=-1.00000 central90=[{lo:+.5},{hi:+.5}]",
        pooled.len(),
        mass - exact
    );
    println!(
        "rhat_rank x={rhat_x:.5} indicator={rhat_ind:.5} ess_bulk x={ess_x_bulk:.1} indicator={ess_i_bulk:.1} ess_tail x={ess_x_tail:.1} indicator={ess_i_tail:.1}"
    );
    let gate = rhat_x < 1.01 && rhat_ind < 1.01 && ess_x_bulk >= 400.0 && ess_i_bulk >= 400.0;
    println!(
        "known_target_gate={gate} gate_definition=rank_rhat<1.01_and_bulk_ess>=400_on_x_and_basin_indicator"
    );
    println!("prior=N(0,1) target=0.75*N(-2,1)+0.25*N(2,1) exact_mean=-1 exact_sd=2");
}
