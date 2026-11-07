//! `temperature_sweep` — at what temperature can the standard stack be sampled?
//!
//! §6.5.3 measured that the exact filter accepts 0.3–20 % of proposals at the
//! kernel's own step size (`β₀ = 0.3`) and **0 %** with no drift cap: the
//! standard stack's likelihood is sharper than the prior's noise scale allows,
//! so the chain freezes near `θ_A` and says nothing about the target.  That is
//! a *temperature* problem, and this example finds the temperature at which it
//! goes away.
//!
//! The target is `π_λ(θ) ∝ N(0, σ²I)·exp(−λ·N·L(θ))`, with `λ` the potential
//! scale (`MhOptions::potential_scale`): `λ = 1` is the published target, and
//! `λ < 1` flattens the likelihood relative to the prior.  The proposal is the
//! capped pCN step of §6.5.2 at `β₀/10`, so the only thing that changes across
//! a row is the temperature.  Kept losses are reported as the raw `L`, so a
//! `λ < 1` row is comparable to the `λ = 1` rows.
//!
//! **The cap is measured, not rounded.**  It was the literal `7.0e2` here while
//! `mh_standard_target` measured it; the measured values are 712 / 757 / 764,
//! so the literal was 6–9 % low and these rows were not quite the §6.5.2 cell
//! they claimed to be.  Both files now call `task::cosine_gradient_scale` — a
//! cap changes only the proposal (the ratio stays exact for `π_λ`), so no
//! conclusion moves, but the acceptance column is now the same experiment.
//!
//! Two questions, and the second is the one that matters:
//! 1. **Where is it usable?**  Acceptance rises as `λ` falls; the first row
//!    with acceptance in the tens of percent is the usable temperature.
//! 2. **What does the target look like there?**  If the predictive at the
//!    usable temperature is still near-uniform, the standard stack's posterior
//!    is uninformative at every temperature a sampler can reach — the
//!    architecture's problem, not the kernel's.  If it becomes informative,
//!    then §6.5.3's near-uniform exact rows were a temperature artefact and
//!    the claim has to be narrowed again.
//!
//! ```text
//! cargo run --release -p pcnl --example temperature_sweep
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::task::{Dataset, MhOptions, Task, adam, cosine_gradient_scale, mh_pcnl_chain};

const SEEDS: [u64; 3] = [20261020, 20261021, 20261022];
/// `(sigma, lambda)` cells: a lambda ladder at the reference width, then two
/// wider priors at the two temperatures that matter.
const CELLS: [(f64, f64); 8] = [
    (0.35, 1.0),
    (0.35, 0.3),
    (0.35, 0.1),
    (0.35, 0.03),
    (0.35, 0.01),
    (0.35, 0.003),
    (2.0, 0.1),
    (2.0, 0.01),
];

fn cfg_for(seed: u64, sigma: f64) -> Config {
    Config {
        seed,
        shapes: Shapes {
            vocab: 9,
            d: 32,
            ff: 64,
            layers: 2,
        },
        n_train: 512,
        n_test: 256,
        chains: 1,
        steps: 500,
        burn: 250,
        thin: 2,
        batch_size: 64,
        sigma,
        // the proposal scale §6.5.3 found usable
        beta_0: 0.03,
        gamma: 1e-3,
        kappa: 0.5,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    println!(
        "standard-softmax stack under the exact filter: usable temperature\n\
         reference configuration (P = 16 672), 500 steps, burn 250, thin 2, 1 chain,\n\
         proposal beta_0 = 0.03 (the scale §6.5.3 found usable), 3 seeds,\n\
         target pi ~ N(0, sigma^2 I) * exp(-lambda * N * L), uniform NLL ln 8 = 2.0794\n"
    );
    println!(
        "{:>9}  {:>6}  {:>6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "seed", "sigma", "lambda", "accept %", "BMA acc", "BMA nll", "radial fn", "K"
    );

    for seed in SEEDS {
        let data_a = Dataset::new(Task::A, 8, 512, 256, seed);
        let data_b = Dataset::new(Task::B, 8, 512, 256, seed);
        let shapes = Shapes {
            vocab: 9,
            d: 32,
            ff: 64,
            layers: 2,
        };
        let theta_a = adam(
            &data_a,
            &shapes,
            Variant::Standard,
            init_flat(&shapes, 0.1, seed),
            1000,
            0.05,
        )
        .0;
        // §6.5.2's cap, measured on the cosine geometry at the reference
        // `β₀ = 0.3` (the same quantity `mh_standard_target` prints), not the
        // rounded literal this file used to carry.
        let theta_a_cos = adam(
            &data_a,
            &shapes,
            Variant::Cosine,
            init_flat(&shapes, 0.1, seed),
            1000,
            0.05,
        )
        .0;
        let cap = {
            let mut cref = cfg_for(seed, 0.35);
            cref.beta_0 = 0.3;
            cosine_gradient_scale(&cref, &data_b, &theta_a_cos, 200)
        };

        for (sigma, lambda) in CELLS {
            let cfg = cfg_for(seed, sigma);
            let opts = MhOptions {
                // the normalized geometry's scale, as in §6.5.2
                grad_norm_cap: Some(cap),
                exact: false,
                potential_scale: lambda,
            };
            let out = mh_pcnl_chain(
                &cfg,
                Variant::Standard,
                &data_b,
                theta_a.clone(),
                0,
                None,
                opts,
            );
            let kept: Vec<Vec<Tensor>> = vec![out.kept.clone()];
            let (acc, nll, _ece, k) = bma_metrics(&kept, &cfg.shapes, Variant::Standard, &data_b);
            println!(
                "{seed:>9}  {sigma:>6.2}  {lambda:>6.3}  {:>8.1}%  {acc:>8.1}%  {nll:>9.4}  {:>9.1}  {k:>7}",
                100.0 * out.accepted as f64 / out.steps.max(1) as f64,
                out.radial_final
            );
        }
        println!("          (cap C = {cap:.3e})");
    }

    println!(
        "\nreading: acceptance rising with 1/lambda is the temperature effect; the BMA\n\
         column is the target's predictive at that temperature.  A usable row whose\n\
         BMA NLL is still near ln 8 means the standard stack's posterior is\n\
         uninformative wherever a sampler can reach it, not merely at lambda = 1."
    );
}
