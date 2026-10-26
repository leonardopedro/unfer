//! §9.3 (M1/M2) measured on the toy: **point readout vs chain readout**.
//!
//! Runs the plan's Phase-3 pipeline at the `Config::quick()` size (Task-A
//! pre-train → unadjusted SG-pCNL chains on Task B, trajectories
//! persisted), then scores two readouts on the same kept states:
//!
//! * **point** — one state's `softmax(logits)` on the test set, i.e. what
//!   a single forward pass (or a single rationale in RLCD's per-rationale
//!   objective) gives you;
//! * **chain** — the mean of `m` states' distributions, the posterior
//!   predictive §9.3 asks for.
//!
//! Three things are reported, and only the first two are claims:
//!
//! 1. Brier / log / ECE / accuracy for both readouts at increasing `m`,
//!    with the *predicted* Brier `‖p̄ − e_y‖² + tr Cov/m` beside the
//!    measured one (`tr Cov` estimated from the same states).
//! 2. The algebraic identity `E_state L(u_s, y) − L(p̄, y) = tr Cov` per
//!    example, evaluated on the real states — its residual is printed,
//!    and it is exact, so the number is a bug detector, not a result.
//! 3. The conditional reward noise: the spread of the Brier across random
//!    `m`-subsets of the chain (what a policy-gradient arm sees on a
//!    labelled example).
//!
//! Everything is seeded and printed; nothing is asserted about a
//! full-scale model, which this example cannot run.
//!
//! ```text
//! cargo run --release -p pcnl --example readout_compare
//! ```

use candle_core::{Device, Tensor};
use pcnl::experiment::{Config, expected_calibration_error, run_experiment, softmax_rows};
use pcnl::model::{Shapes, Variant, forward};
use pcnl::persist::{load_trajectory, slug};
use pcnl::readout::Rule;
use pcnl::task::Dataset;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn variant_of(name: &str) -> Variant {
    if name.starts_with("cosine") {
        Variant::Cosine
    } else {
        Variant::Standard
    }
}

/// `[n_test][K]` distributions for one state, in f64.
fn state_probs(
    theta: &[f32],
    shapes: &Shapes,
    variant: Variant,
    x: &Tensor,
    n_test: usize,
) -> Option<Vec<Vec<f64>>> {
    let dev = Device::Cpu;
    let t = Tensor::from_vec(theta.to_vec(), vec![theta.len()], &dev).ok()?;
    let logits = forward(&t, shapes, variant, x).ok()?;
    let rows = logits.to_vec2::<f32>().ok()?;
    if rows.len() != n_test {
        return None;
    }
    Some(
        rows.iter()
            .map(|row| {
                let v: Vec<f64> = row.iter().map(|z| *z as f64).collect();
                softmax_rows(&v)
            })
            .collect(),
    )
}

/// Mean of the selected states' distributions, per test example.
fn mean_over(
    states: &[Vec<Vec<f64>>],
    picks: &[usize],
    n_test: usize,
    k: usize,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; k]; n_test];
    for &s in picks {
        for (i, row) in states[s].iter().enumerate() {
            for (a, v) in out[i].iter_mut().zip(row.iter()) {
                *a += v;
            }
        }
    }
    let m = picks.len() as f64;
    for row in &mut out {
        for v in row.iter_mut() {
            *v /= m;
        }
    }
    out
}

/// Mean loss, accuracy and ECE of a whole readout on the test set.
fn metrics(probs: &[Vec<f64>], targets: &[u32], rule: Rule) -> (f64, f64, f64) {
    let k = probs[0].len();
    let flat: Vec<f64> = probs.iter().flat_map(|r| r.iter().copied()).collect();
    let mut hits = 0usize;
    let mut loss = 0.0f64;
    for (i, row) in probs.iter().enumerate() {
        let y = targets[i] as usize;
        loss += rule.loss(row, y);
        let (best, _) = row
            .iter()
            .enumerate()
            .fold((0usize, -1.0f64), |acc, (j, p)| if *p > acc.1 { (j, *p) } else { acc });
        if best == y {
            hits += 1;
        }
    }
    let n = probs.len() as f64;
    (
        loss / n,
        100.0 * hits as f64 / n,
        expected_calibration_error(&flat, k, targets, 10),
    )
}

fn main() -> candle_core::Result<()> {
    // `Config::quick()`'s shapes and data sizes, with a longer chain than
    // the smoke test's 400 steps: the point-vs-chain gap is `tr Cov/m`,
    // and a 400-step chain has barely left its starting shell.
    let mut cfg = Config::quick();
    cfg.steps = 2_000;
    cfg.burn = 800;
    cfg.thin = 8;
    cfg.chains = 3;
    let dir = std::env::temp_dir().join(format!("pcnl_readout_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    cfg.save_dir = Some(dir.clone());

    println!(
        "SG-pCNL quick run (seed {}, k {}, shapes {}x{}x{}, chains {}, steps {}, burn {}, thin {})",
        cfg.seed, cfg.k, cfg.shapes.vocab, cfg.shapes.d, cfg.shapes.layers, cfg.chains,
        cfg.steps, cfg.burn, cfg.thin
    );
    let report = run_experiment(&cfg);
    // identical reconstruction to `run_experiment`'s Task-B data
    let data = Dataset::new(
        cfg.task_b,
        cfg.k,
        cfg.n_train,
        cfg.n_test,
        cfg.seed.wrapping_add(1),
    );
    let targets = data.test_y.to_vec1::<u32>()?;
    let n_test = data.test_y.dim(0)?;
    let k = cfg.shapes.vocab;

    for v in &report.variants {
        let variant = variant_of(v.variant);
        println!(
            "\n[{}] BMA acc {:.1}%  NLL {:.4}  ECE {:.4}  ({} models, {} trajectory files)  \
             exploded: {}",
            v.variant,
            v.bma_accuracy,
            v.bma_nll,
            v.bma_ece,
            v.bma_models,
            v.saved_files.len(),
            v.exploded
        );
        // all persisted states, chain by chain (deterministic order)
        let mut states: Vec<Vec<Vec<f64>>> = Vec::new();
        for c in 0..cfg.chains {
            let path = dir.join(format!("{}_chain{c}.safetensors", slug(v.variant)));
            let Ok(rows) = load_trajectory(&path) else {
                continue;
            };
            for theta in rows {
                if let Some(p) = state_probs(&theta, &cfg.shapes, variant, &data.test_x, n_test) {
                    states.push(p);
                }
            }
        }
        if states.len() < 4 {
            println!(
                "  only {} usable states (non-finite forward or no trajectory) — skipping; \
                 the report records why: exploded={}, exploded_at={:?}",
                states.len(),
                v.exploded,
                v.exploded_at_step
            );
            continue;
        }
        let all: Vec<usize> = (0..states.len()).collect();

        // ---- identity check on the real states ----------------------
        // For each test example: mean_s Brier(u_s, y) − Brier(p̄, y) = E_s‖u_s − p̄‖².
        let p_bar = mean_over(&states, &all, n_test, k);
        let mut residual = 0.0f64;
        let mut tr_cov = 0.0f64;
        let mut point_brier = 0.0f64;
        for i in 0..n_test {
            let y = targets[i] as usize;
            let mean_loss: f64 = states.iter().map(|s| Rule::Brier.loss(&s[i], y)).sum::<f64>()
                / states.len() as f64;
            let chain_loss = Rule::Brier.loss(&p_bar[i], y);
            let tr: f64 = states
                .iter()
                .map(|s| {
                    s[i].iter()
                        .zip(p_bar[i].iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / states.len() as f64;
            residual = residual.max((mean_loss - chain_loss - tr).abs());
            tr_cov += tr / n_test as f64;
            point_brier += mean_loss / n_test as f64;
        }
        println!(
            "  {} states, {} test examples: tr Cov(u) = {:.5}, point Brier {:.5} (over all states)",
            states.len(),
            n_test,
            tr_cov,
            point_brier
        );
        println!(
            "  identity check  mean_s Brier(u_s,y) − Brier(p̄,y) − tr Cov : max |residual| = {:.2e}",
            residual
        );

        // ---- readout table -----------------------------------------
        // Each row is the *mean over `reps` random m-subsets* of the
        // chain (seeded), because that is the expectation the identity
        // predicts; a single subset's number is one draw from it.
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(9000));
        let reps = 100usize;
        let (b_inf, acc_inf, ece_inf) = metrics(&p_bar, &targets, Rule::Brier);
        let (l_inf, _, _) = metrics(&p_bar, &targets, Rule::Log);
        println!(
            "  {:>4}  {:>8}  {:>8}  {:>8}  {:>7}  {:>6}  {:>11}  {:>10}",
            "m", "Brier", "sd", "log", "ECE", "acc%", "Brier_pred", "gap"
        );
        let mut m = 1usize;
        while m <= states.len() {
            let mut bs = Vec::with_capacity(reps);
            let mut ls = Vec::with_capacity(reps);
            let (mut ece, mut acc) = (0.0f64, 0.0f64);
            for _ in 0..reps {
                let mut idx = all.clone();
                for i in 0..m {
                    let j = i + (rng.random::<u32>() as usize) % (idx.len() - i);
                    idx.swap(i, j);
                }
                let p = mean_over(&states, &idx[..m], n_test, k);
                let (b, a, e) = metrics(&p, &targets, Rule::Brier);
                let (l, _, _) = metrics(&p, &targets, Rule::Log);
                bs.push(b);
                ls.push(l);
                ece += e / reps as f64;
                acc += a / reps as f64;
            }
            let mean = bs.iter().sum::<f64>() / reps as f64;
            let var = bs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / reps as f64;
            let lmean = ls.iter().sum::<f64>() / reps as f64;
            let predicted = b_inf + tr_cov / m as f64;
            println!(
                "  {:>4}  {:>8.5}  {:>8.1e}  {:>8.5}  {:>7.4}  {:>6.1}  {:>11.5}  {:>10.2e}",
                m,
                mean,
                var.sqrt(),
                lmean,
                ece,
                acc,
                predicted,
                mean - predicted
            );
            m = if m < 4 { m + 1 } else { m * 2 };
        }
        println!(
            "  {:>4}  {:>8.5}  {:>8}  {:>8.5}  {:>7.4}  {:>6.1}  {:>11}  {:>10}",
            "all", b_inf, "—", l_inf, ece_inf, acc_inf, "—", "—"
        );
        println!(
            "  reward noise (Brier across random m-subsets, sd): read off the `sd` column; \
             it is the quantity an RL arm's advantage is built from."
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
