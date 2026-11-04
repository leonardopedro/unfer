//! `cold_ladder` — is the cold-`λ` calibration gain a three-seed effect at the
//! published budget, or one lucky run?
//!
//! §6.5.5 measured `π_λ ∝ N(0,σ²I)·exp(−λ·NL)` on one seed at 1 000 steps and
//! found BMA NLL 2.1958 → 2.1030 and ECE 0.1897 → 0.0670 as `λ` went 1 → 16,
//! with accuracy flat.  A single seed cannot separate "cold `λ` calibrates"
//! from "this seed was overconfident", so this example reruns the same ladder
//! at the **published budget** (10 000 steps, burn 5 000) on three seeds and
//! reports every seed plus a min/median/max summary.  It also reports the
//! per-draw vs BMA gap, because shrinking ECE without shrinking that gap would
//! mean the draws got individually better calibrated rather than the ensemble
//! moving anywhere.
//!
//! Run only the cells that fit: `smoke` is a timing probe (200 steps, one
//! chain), a bare number sets the steps, and `--seeds N`/`--chains N` trim the
//! matrix.
//!
//! ```text
//! cargo run --release -p pcnl --example cold_ladder            # full matrix
//! cargo run --release -p pcnl --example cold_ladder -- smoke    # timing probe
//! cargo run --release -p pcnl --example cold_ladder -- 2000     # shorter run
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState};
use pcnl::task::{BatchLoader, Dataset, Task, adam};
use rand::SeedableRng;

const SEEDS: [u64; 3] = [20261020, 20261021, 20261022];
const LAMBDAS: [f64; 3] = [1.0, 4.0, 16.0];

/// One seed's `(BMA NLL, ECE, BMA acc, per-draw NLL, gap, r)`.
type Row = (f64, f64, f64, f64, f64, f64);

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
        steps: 10_000,
        burn: 5_000,
        thin: 50,
        batch_size: 64,
        sigma: 0.35,
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.5,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg0 = cfg_for(SEEDS[0]);
    let mut seeds = SEEDS.len();
    let mut lambdas = LAMBDAS.len();
    if args.first().map(|a| a.as_str()) == Some("smoke") {
        cfg0.steps = 200;
        cfg0.burn = 100;
        cfg0.thin = 10;
        cfg0.chains = 1;
        seeds = 1;
        lambdas = 1;
    } else if let Some(n) = args.first().and_then(|a| a.parse::<usize>().ok()) {
        cfg0.steps = n;
        cfg0.burn = n / 2;
    }
    let mut picks: Vec<f64> = LAMBDAS.to_vec();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seeds" => seeds = it.next().and_then(|v| v.parse().ok()).unwrap_or(seeds),
            "--chains" => cfg0.chains = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--lambdas" => {
                picks = it
                    .next()
                    .unwrap_or(&"1,16".to_string())
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
            }
            _ => {}
        }
    }

    let p = cfg0.shapes.flat_size();
    println!(
        "cold ladder: π_λ ∝ N(0,σ²I)·exp(−λ·NL), target B, cosine, P = {p}\n\
         {} steps, burn {}, thin {}, {} chains, {} seed(s); uniform NLL ln 8 = {:.4}\n",
        cfg0.steps,
        cfg0.burn,
        cfg0.thin,
        cfg0.chains,
        seeds,
        (8.0f64).ln()
    );
    println!(
        "{:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}  {:>7}",
        "λ", "seed", "draw acc", "BMA acc", "draw NLL", "BMA NLL", "ECE", "r", "radial"
    );

    // (λ, BMA NLL, ECE, BMA acc, per-draw NLL, gap, r) per seed, for the summary
    let mut cells: Vec<(f64, Vec<Row>)> = Vec::new();
    let _ = lambdas;
    for lam in picks.iter() {
        let mut rows = Vec::new();
        for seed in SEEDS.iter().take(seeds) {
            // clone the base so the smoke/short-run overrides survive
            let mut cfg = cfg0.clone();
            cfg.seed = *seed;
            let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
            let data_b = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
            let theta_a = adam(
                &data_a,
                &cfg.shapes,
                Variant::Cosine,
                init_flat(&cfg.shapes, 0.1, cfg.seed),
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
            let p_f = p as f64;
            for c in 0..cfg.chains {
                let mut state = SgPcnlState::new(
                    theta_a.clone(),
                    cfg.sigma,
                    sched,
                    cfg.n_train,
                    cfg.batch_size,
                    0,
                );
                state.potential_scale = *lam;
                let mut loader = BatchLoader::new(
                    cfg.n_train,
                    cfg.batch_size,
                    cfg.seed ^ ((c as u64) << 8) ^ 0xc01d,
                );
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(cfg.seed ^ ((c as u64) << 8) ^ 0x5eed);
                let mut kept = Vec::with_capacity((cfg.steps - cfg.burn) / cfg.thin + 1);
                for t in 0..cfg.steps {
                    let (x, y) = loader.next(&data_b);
                    let (_l, g) = data_b
                        .batch_grad(&state.theta, &cfg.shapes, Variant::Cosine, &x, &y)
                        .expect("grad");
                    // §6.5.5's ratio, before the step
                    if let Ok(gn) = pcnl::sampler::norm(&g) {
                        let beta = state.scheduled(t);
                        let h = state.lambda_at(t) * beta * beta * cfg.sigma * cfg.sigma / 2.0;
                        let noise = beta * cfg.sigma * p_f.sqrt();
                        if noise > 0.0 && gn.is_finite() {
                            ratio_sum +=
                                h * gn * (cfg.n_train as f64) / (cfg.batch_size as f64) / noise;
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

            let (bma_acc, bma_nll, ece, _k) =
                bma_metrics(&chains, &cfg.shapes, Variant::Cosine, &data_b);
            let n = n_draw as f64;
            let draw_mean = draw_nll / n;
            let r = ratio_sum / ratio_n as f64;
            let radial = rad_sum / rad_n as f64;
            println!(
                "{lam:>5.1}  {:>9}  {:>8.2}%  {bma_acc:>8.2}%  {draw_mean:>9.4}  {bma_nll:>9.4}  \
                 {ece:>8.4}  {r:>7.3}  {radial:>7.1}",
                cfg.seed,
                draw_acc / n
            );
            rows.push((bma_nll, ece, bma_acc, draw_mean, draw_mean - bma_nll, r));
        }
        cells.push((*lam, rows));
    }

    println!("\nsummary over seeds (min / median / max):");
    println!(
        "{:>5}  {:>26}  {:>26}  {:>26}",
        "λ", "BMA NLL", "ECE", "BMA acc"
    );
    for (lam, rows) in &cells {
        let col = |f: &dyn Fn(&Row) -> f64| {
            let mut v: Vec<f64> = rows.iter().map(f).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            (v[0], v[v.len() / 2], v[v.len() - 1])
        };
        let (n0, n1, n2) = col(&|r| r.0);
        let (e0, e1, e2) = col(&|r| r.1);
        let (a0, a1, a2) = col(&|r| r.2);
        println!(
            "{lam:>5.1}  {n0:>8.4} /{n1:>8.4} /{n2:>8.4}  {e0:>8.4} /{e1:>8.4} /{e2:>8.4}  \
             {a0:>7.2}% /{a1:>7.2}% /{a2:>7.2}%"
        );
    }
    println!("\nper-seed gap (per-draw NLL − BMA NLL) and measured r:");
    for (lam, rows) in &cells {
        let gaps: Vec<String> = rows.iter().map(|r| format!("{:.4}", r.4)).collect();
        let rs: Vec<String> = rows.iter().map(|r| format!("{:.3}", r.5)).collect();
        println!(
            "  λ = {lam:>4.1}: gap [{}]   r [{}]",
            gaps.join(", "),
            rs.join(", ")
        );
    }
    println!(
        "\nreading: the claim under test is that cold λ moves BMA NLL and ECE\ndown and leaves accuracy at the chance line; a summary whose min/max\nbands do not overlap between λ = 1 and λ = 16 is the three-seed version of\n§6.5.5's single-seed row.  A constant gap across λ would say the draws\nthemselves got calibrated, not that the ensemble moved."
    );
}
