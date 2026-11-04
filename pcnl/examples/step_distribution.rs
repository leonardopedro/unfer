//! `step_distribution` — can any step distribution in this family reach
//! `r ≳ 1` without the standard stack's explosion?
//!
//! §6.5.5's diagnosis is that the pCN step's information-to-noise ratio
//! `r = βσ‖g‖/(2√P)` is capped at ≈ 0.17 by `β ≤ 1`, so the sampled direction
//! is prior-drawn whatever the schedule.  That claim is about *pCN*, and the
//! obvious objection is that `β ≤ 1` is a property of the proposal family, not
//! of the problem.  The natural family that does not pay it is the
//! Euler–Langevin (ULA) step for the same target,
//!
//! ```text
//! θ' = θ − h·(θ/σ² + g) + √(2h)·σ·z,      r_ULA = h‖g‖ / (√(2h)·σ√P)
//! ```
//!
//! where `h` is a free step size: `r_ULA ∝ √h‖g‖`, so a high ratio is reachable
//! *in principle*.  The ladder is therefore expressed in units of the prior's
//! own curvature: `h = σ²·{1/8, 1/4, 1/2, 1}`, which brackets the stability
//! boundary at `h = σ²` (where the prior term `1 − h/σ²` changes sign) instead
//! of hiding it behind a nominal ratio.  `r` is printed as **measured**, from
//! the gradients the run actually sees, because the scale that mattered before
//! the first step is not the scale the chain ends up at.
//!
//! pCN's own `β = 1` row is included only as the degenerate edge: at `β = 1`
//! the OU memory is `√(1−β²) = 0`, so the step forgets `θ` entirely and the
//! chain is not a sampler.  `β = 0.9` is the near-maximal *usable* kernel, and
//! it is the row the ceiling claim should be read against.
//!
//! Two things the columns cannot show, so they are stated here: the `O(h)`
//! discretization bias means a surviving ULA row is not sampling `π`, and a
//! radial column far above `σ√P` means the weights were inflated — the
//! predictive of an inflated network is a different object, not a better
//! posterior.
//!
//! ```text
//! cargo run --release -p pcnl --example step_distribution
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::task::{BatchLoader, Dataset, Task, adam};
use rand::{Rng as _, SeedableRng};
use rand_distr::StandardNormal;

const SEEDS: [u64; 2] = [20261020, 20261021];
/// ULA step sizes as fractions of the prior's curvature `σ²`.
const H_FRACS: [f64; 4] = [0.125, 0.25, 0.5, 1.0];

/// One arm's accumulated summary: `(label, r, BMA NLL, ECE, ever non-finite)`.
type Arm = (String, Vec<f64>, Vec<f64>, Vec<f64>, bool);

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
        chains: 1,
        steps: 400,
        burn: 200,
        thin: 10,
        batch_size: 64,
        sigma: 0.35,
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.5,
        pretrain_steps: 1_000,
        pretrain_adam: true,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    let cfg0 = cfg_for(SEEDS[0]);
    let p = cfg0.shapes.flat_size();
    println!(
        "pCN vs Euler–Langevin on the same target, cosine variant, P = {p}\n\
         π ∝ N(0,σ²I)·exp(−NL), σ = {}, {} steps, burn {}, thin {}, 1 chain, {} seeds\n\
         uniform NLL ln 8 = {:.4}; r = likelihood-drift norm / proposal-noise norm\n",
        cfg0.sigma,
        cfg0.steps,
        cfg0.burn,
        cfg0.thin,
        SEEDS.len(),
        (8.0f64).ln()
    );
    println!(
        "{:>11}  {:>9}  {:>7}  {:>8}  {:>9}  {:>9}  {:>8}  {:>7}  {:>7}",
        "family", "seed", "r meas", "draw acc", "draw NLL", "BMA NLL", "ECE", "radial", "max"
    );

    let mut summary: Vec<Arm> = Vec::new();
    for seed in SEEDS {
        let cfg = Config {
            seed,
            ..cfg0.clone()
        };
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
        let p_f = p as f64;
        let sigma = cfg.sigma;

        // (label, pcn beta or ULA h)
        let mut arms: Vec<(String, Option<f64>, Option<f64>)> = vec![
            ("pCN β=0.3".into(), Some(cfg.beta_0), None),
            ("pCN β=0.9".into(), Some(0.9), None),
            ("pCN β=1.0".into(), Some(1.0), None),
        ];
        for frac in H_FRACS {
            arms.push((
                format!("ULA h=σ²/{:.0}", 1.0 / frac),
                None,
                Some(frac * sigma * sigma),
            ));
        }

        for (label, beta, h) in arms {
            let mut theta = theta_a.clone();
            let mut loader = BatchLoader::new(cfg.n_train, cfg.batch_size, seed ^ 0xbeef);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xfeed);
            let mut kept: Vec<Tensor> = Vec::new();
            let (mut draw_nll, mut draw_acc, mut n_draw) = (0.0f64, 0.0f64, 0usize);
            let (mut ratio_sum, mut ratio_n) = (0.0f64, 0usize);
            let (mut rad_final, mut rad_max) = (f64::NAN, 0.0f64);
            let mut exploded_at: Option<usize> = None;
            let h_val = h.unwrap_or(f64::NAN);

            for t in 0..cfg.steps {
                let (x, y) = loader.next(&data_b);
                let (_l, g) = data_b
                    .batch_grad(&theta, &cfg.shapes, Variant::Cosine, &x, &y)
                    .expect("grad");
                let scale = (cfg.n_train as f64) / (cfg.batch_size as f64);
                let gn = pcnl::sampler::norm(&g).unwrap_or(f64::NAN);
                if !gn.is_finite() {
                    exploded_at = Some(t);
                    break;
                }
                let noise_vec: Vec<f32> = (0..p)
                    .map(|_| {
                        let z: f32 = rng.sample(StandardNormal);
                        z * sigma as f32
                    })
                    .collect();
                let z = Tensor::from_vec(noise_vec, vec![p], theta.device()).expect("noise");

                let next = if let Some(b) = beta {
                    // pCN (the published kernel): θ ← √(1−β²)θ + βξ − h·g
                    let b2 = b * b;
                    let drift = b2 * sigma * sigma / 2.0;
                    let noise_amp = b;
                    ratio_sum += drift * gn * scale / (b * sigma * p_f.sqrt());
                    ratio_n += 1;
                    ((theta
                        .affine((1.0 - b2).max(0.0).sqrt(), 0.0)
                        .expect("decay")
                        + z.affine(noise_amp, 0.0).expect("noise"))
                    .expect("ou")
                        - g.affine(drift * scale, 0.0).expect("drift"))
                    .expect("pcn step")
                } else {
                    // Euler–Langevin: θ ← θ − h(θ/σ² + g) + √(2h)ξ
                    ratio_sum += h_val * gn * scale / ((2.0 * h_val).sqrt() * sigma * p_f.sqrt());
                    ratio_n += 1;
                    let prior_term = theta
                        .affine(1.0 - h_val / (sigma * sigma), 0.0)
                        .expect("prior");
                    let likelihood_term = g.affine(h_val * scale, 0.0).expect("lik");
                    let noise_term = z.affine((2.0 * h_val).sqrt(), 0.0).expect("noise");
                    ((prior_term - likelihood_term).expect("drift sum") + noise_term)
                        .expect("ula step")
                };
                theta = next;
                let r = pcnl::sampler::norm(&theta).unwrap_or(f64::NAN);
                if !r.is_finite() {
                    exploded_at = Some(t);
                    break;
                }
                rad_max = rad_max.max(r);
                rad_final = r;
                if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                    let nll = data_b
                        .nll(&theta, &cfg.shapes, Variant::Cosine)
                        .expect("nll");
                    let acc = data_b
                        .accuracy(&theta, &cfg.shapes, Variant::Cosine)
                        .expect("acc");
                    if nll.is_finite() {
                        draw_nll += nll;
                        draw_acc += acc;
                        n_draw += 1;
                    }
                    kept.push(theta.clone());
                }
            }

            let (_bma_acc, bma_nll, ece, _k) = if kept.is_empty() {
                (f64::NAN, f64::NAN, f64::NAN, 0)
            } else {
                bma_metrics(&[kept.clone()], &cfg.shapes, Variant::Cosine, &data_b)
            };
            let n = n_draw.max(1) as f64;
            let r_mean = if ratio_n == 0 {
                f64::NAN
            } else {
                ratio_sum / ratio_n as f64
            };
            println!(
                "{label:>11}  {seed:>9}  {:>7.3}  {:>7.2}%  {:>9.4}  {bma_nll:>9.4}  {ece:>8.4}  \
                 {rad_final:>7.1}  {rad_max:>7.1}{}",
                r_mean,
                draw_acc / n,
                draw_nll / n,
                match exploded_at {
                    Some(t) => format!("  ← non-finite at step {t}"),
                    None => String::new(),
                }
            );
            let key = label.clone();
            if let Some(entry) = summary.iter_mut().find(|e| e.0 == key) {
                entry.1.push(r_mean);
                entry.2.push(bma_nll);
                entry.3.push(ece);
                entry.4 |= exploded_at.is_some();
            } else {
                summary.push((
                    key,
                    vec![r_mean],
                    vec![bma_nll],
                    vec![ece],
                    exploded_at.is_some(),
                ));
            }
        }
    }

    println!("\nper-family means over seeds:");
    println!(
        "{:>11}  {:>8}  {:>10}  {:>10}  {:>8}",
        "family", "r meas", "BMA NLL", "ECE", "died"
    );
    for (name, r, nll, ece, died) in &summary {
        let mean = |v: &Vec<f64>| {
            let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            if f.is_empty() {
                f64::NAN
            } else {
                f.iter().sum::<f64>() / f.len() as f64
            }
        };
        println!(
            "{name:>11}  {:>8.3}  {:>10.4}  {:>10.4}  {:>8}",
            mean(r),
            mean(nll),
            mean(ece),
            if *died { "yes" } else { "no" }
        );
    }
    println!(
        "\nreading: `died` marks a non-finite state; at h = σ² the prior term\n\
         (1 − h/σ²) vanishes and beyond it the step oscillates, so the top row is\n\
         the boundary, not a bug.  What settles the §6.5.5 question is the pair\n\
         (r meas, radial): a ULA row that reaches a higher r than pCN *and* stays on\n\
         the σ√P shell would mean the β ≤ 1 ceiling was the obstacle; a row that\n\
         reaches it only by inflating the weights is optimising this loss, not\n\
         sampling the posterior."
    );
}
