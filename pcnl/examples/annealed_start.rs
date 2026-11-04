//! `annealed_start` — does switching the likelihood on *after* the chain has
//! spread out help it reach the informative set?
//!
//! §6.4 measured that the directions which beat chance on Task B occupy a
//! sub-1/2000 fraction of the sphere, and that the λ = 1 chain's kept draws
//! all sit in one mode (per-draw NLL ≈ BMA NLL, `overdispersion`).  The
//! hypothesis behind `--anneal N` is that the failure is the *start*: at
//! `λ = 1` the drift pulls the direction into the first basin it sees during
//! burn-in, before the chain has any chance to diffuse.  With
//! `λ_t = min(1, t/N)` the drift is switched off at `t = 0`, so the early
//! chain is the prior's pCN random walk — it decorrelates the direction — and
//! the likelihood is faded in while the OU term keeps the radius on the
//! shell.
//!
//! Two readings were possible:
//!
//! * if the annealed chain finds *better* directions, per-draw and BMA NLL
//!   fall with `N` and the cross-chain direction cosine falls (chains that
//!   search end up in different places);
//! * if the informative set is simply too small to hit by diffusion, the
//!   annealed chain spends the ramp learning nothing and then contracts into
//!   whatever direction it happens to be at, so the numbers are flat or worse
//!   — the same conclusion as §6.4's volume bound, reached dynamically.
//!
//! **What happens is neither, and the third reading is the interesting one.**
//! The kept predictive is identical to every printed digit (`N = 0` and
//! `N = 750` both 12.89 % / NLL 2.2194 / BMA 2.1923), and the chains end in
//! *unrelated* directions (cross-cosine −0.002, i.e. the 1/√P noise floor of
//! two independent sphere draws).  The `drift/noise` column says why: the
//! per-step drift norm is ~5 % of the pCN noise norm, so the proposal — which
//! replaces a `β²` fraction of the norm with *fresh prior* noise every step —
//! re-draws the direction ~20× faster than the likelihood can steer it.  The
//! chain therefore reports the prior's predictive, and a ramp that scales only
//! the numerator cannot change that.
//!
//! `N ≤ burn` in every cell, so the kept draws are at `λ = 1` and the
//! predictive is the published target's (the report line says so too).
//!
//! **Why only two cells, one seed.** A chain-run of 1 500 steps at the
//! published `P = 16 672` costs ~6 min of CPU, so the matrix is `N ∈ {0, 750}`
//! × one seed — the contrast the hypothesis actually predicts, measured once.
//! A smaller model is not a substitute: at `P ≈ 2k` the same probe has the
//! chain at chance (12.9 % on a 1/8 task), so it cannot tell a search that
//! failed from a search that was never happening.
//!
//! ```text
//! cargo run --release -p pcnl --example annealed_start
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState};
use pcnl::task::{BatchLoader, Dataset, Task, adam};
use rand::SeedableRng;

const SEEDS: [u64; 1] = [20261020];
const ANNEALS: [usize; 2] = [0, 750];

fn cfg_for(seed: u64) -> Config {
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
        chains: 2,
        steps: 1_500,
        burn: 750,
        thin: 15,
        batch_size: 64,
        sigma: 0.35,
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.5,
        run_mh: false,
        ..Config::default()
    }
}

/// Mean pairwise cosine between the kept draws of chain `a` and chain `b`.
/// One means the two chains ended in the same direction; near zero means they
/// are unrelated draws of the sphere.  Cosine, not distance, because the
/// target is scale-invariant and the radius is set by the prior.
fn cross_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for x in a {
        let nx = norm(x);
        for y in b {
            let ny = norm(y);
            if nx > 0.0 && ny > 0.0 {
                sum += dot(x, y) / (nx * ny);
                n += 1;
            }
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

fn main() {
    let cfg0 = cfg_for(SEEDS[0]);
    let p = cfg0.shapes.flat_size();
    println!(
        "annealed start: lambda_t = min(1, t/N), kept draws at lambda = 1 (N <= burn {})\n\
         target B, cosine variant, P = {p}, {} steps, burn {}, thin {}, {} chains\n\
         uniform NLL ln 8 = {:.4}; 'gap' = per-draw NLL - BMA NLL; 'cos' = cross-chain \\
         draw cosine\n",
        cfg0.burn,
        cfg0.steps,
        cfg0.burn,
        cfg0.thin,
        cfg0.chains,
        (8.0f64).ln()
    );
    println!(
        "{:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}  {:>9}",
        "N",
        "seed",
        "draw acc",
        "BMA acc",
        "draw NLL",
        "BMA NLL",
        "gap",
        "cos",
        "radial",
        "drift/noise"
    );

    let mut by_anneal: Vec<(usize, f64)> = Vec::new();
    let mut ratio_at_beta0 = f64::NAN;
    for anneal in ANNEALS {
        let (mut gap_sum, mut n_seed) = (0.0f64, 0usize);
        for seed in SEEDS {
            let cfg = cfg_for(seed);
            assert!(
                anneal <= cfg.burn,
                "kept draws must be at lambda = 1 for the measurement to be the target's"
            );
            let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
            let data_b = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, seed);
            let theta_a = adam(
                &data_a,
                &cfg.shapes,
                Variant::Cosine,
                init_flat(&cfg.shapes, 0.1, seed),
                cfg.pretrain_steps,
                cfg.adam_lr,
            )
            .0;

            let sched = SgPcnlScheduler {
                beta_0: cfg.beta_0,
                gamma: cfg.gamma,
                kappa: cfg.kappa,
            };
            let mut chains: Vec<Vec<Tensor>> = Vec::new();
            let (mut draw_nll, mut draw_acc, mut n_draw) = (0.0f64, 0.0f64, 0usize);
            let (mut rad_sum, mut rad_n) = (0.0f64, 0usize);
            let (mut ratio_sum, mut ratio_n) = (0.0f64, 0usize);
            let (p_f, sigma) = (p as f64, cfg.sigma);
            for c in 0..cfg.chains {
                let mut state = SgPcnlState::new(
                    theta_a.clone(),
                    cfg.sigma,
                    sched,
                    cfg.n_train,
                    cfg.batch_size,
                    0,
                );
                state.anneal_steps = anneal;
                let mut loader = BatchLoader::new(
                    cfg.n_train,
                    cfg.batch_size,
                    seed ^ ((c as u64) << 8) ^ 0xa11ea1,
                );
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ ((c as u64) << 8) ^ 0xfeed);
                let mut kept = Vec::new();
                for t in 0..cfg.steps {
                    let (x, y) = loader.next(&data_b);
                    let (_l, g) = data_b
                        .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
                        .expect("grad");
                    // information-to-noise ratio of this step, before it is
                    // taken: |h_t·λ_t·g| / |β_t σ ξ| (the noise norm is
                    // deterministic: β σ √P)
                    let beta = state.scheduled(t);
                    let h = state.lambda_at(t) * beta * beta * sigma * sigma / 2.0;
                    if let Ok(gn) = pcnl::sampler::norm(&g) {
                        let drift = h * gn * (cfg.n_train as f64) / (cfg.batch_size as f64);
                        let noise = beta * sigma * p_f.sqrt();
                        if noise > 0.0 && drift.is_finite() {
                            ratio_sum += drift / noise;
                            ratio_n += 1;
                        }
                    }
                    state.step_unadjusted(&g, &mut rng).expect("step");
                    if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                        let nll = data_b
                            .nll(&state.theta, &cfg.shapes, Variant::Cosine)
                            .expect("nll");
                        let acc = data_b
                            .accuracy(&state.theta, &cfg.shapes, Variant::Cosine)
                            .expect("acc");
                        if nll.is_finite() {
                            draw_nll += nll;
                            draw_acc += acc;
                            n_draw += 1;
                        }
                        if let Ok(r) = pcnl::sampler::norm(&state.theta) {
                            rad_sum += r;
                            rad_n += 1;
                        }
                        kept.push(state.theta.clone());
                    }
                }
                chains.push(kept);
            }

            let (bma_acc, bma_nll, _ece, _k) =
                bma_metrics(&chains, &cfg.shapes, Variant::Cosine, &data_b);
            let n = n_draw as f64;
            let gap = draw_nll / n - bma_nll;
            gap_sum += gap;
            n_seed += 1;

            let flat: Vec<Vec<Vec<f32>>> = chains
                .iter()
                .map(|ch| {
                    ch.iter()
                        .map(|t| t.to_vec1::<f32>().expect("flat draw"))
                        .collect()
                })
                .collect();
            let cos = if flat.len() >= 2 {
                cross_cosine(&flat[0], &flat[1])
            } else {
                f64::NAN
            };
            let mean_ratio = ratio_sum / ratio_n as f64;
            if anneal == 0 {
                ratio_at_beta0 = mean_ratio;
            }
            println!(
                "{anneal:>5}  {seed:>9}  {:>8.2}%  {bma_acc:>8.2}%  {:>9.4}  {bma_nll:>9.4}  \
                 {gap:>9.4}  {cos:>8.4}  {:>7.1}  {:>9.3}",
                draw_acc / n,
                draw_nll / n,
                rad_sum / rad_n as f64,
                mean_ratio
            );
        }
        by_anneal.push((anneal, gap_sum / n_seed as f64));
    }

    println!("\nmean per-draw - BMA NLL by ramp length (the size of the ensemble's disagreement):");
    for (anneal, gap) in &by_anneal {
        println!("  N = {anneal:>4}: {gap:>8.4} nats");
    }
    println!(
        "\nwhy: the per-step ratio is r = beta*sigma*|g| / (2*sqrt(P)), so it grows with\n\
         beta and is invariant to sigma (the trajectory scales with it).  The sharpest\n\
         admissible proposal, beta = 1, would give r = {r1:.3}: even then the likelihood\n\
         pull per step is a fraction of the proposal's prior noise, which is why no\n\
         schedule of lambda moves the predictive.  What annealing would need is a\n\
         larger beta than pCN allows, not a different ramp.",
        r1 = ratio_at_beta0 / cfg0.beta_0
    );
}
