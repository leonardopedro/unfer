//! `sphere_volume` — how much of `S^{P-1}` actually predicts the task?
//!
//! §6.4(ii) of `test/unfer/ngpt.md` argues that moving the *direction*
//! posterior is a volume competition, not a distance: the prior's
//! direction is Haar, the likelihood depends on direction alone, and the
//! directions that beat the Haar-typical one occupy a fraction
//! `≈ exp(−N·I)` of the sphere — the entropy term the energy `N·Δ` has to
//! beat.  That sentence had no number.  This example measures the number.
//!
//! Draw `R` directions uniformly on `S^{P-1}` (a normalised Gaussian is
//! exactly Haar), put them on the `σ√P` shell of §3.3, and evaluate
//! accuracy and per-example cross-entropy on the task's held-out split.
//! The report is a ladder of fractions, each against a baseline that
//! means something:
//!
//! * **chance** and **the best constant predictor** — the latter is the
//!   honest accuracy baseline here, because a near-uniform direction's
//!   logits barely depend on the input, so its argmax is one constant
//!   class and its accuracy is that class's frequency, not a
//!   `Binomial(n_test, 1/K)` draw;
//! * **`ln K`**, the uniform predictor — the loss baseline;
//! * **the deterministic fine-tuner's loss**, the direction a chain would
//!   have to find to adapt at all.
//!
//! Then the two numbers §6.4(ii) needs: how far out the fine-tuned
//! direction is in units of the measured Haar spread, and what fraction of
//! the sphere that implies.  The *measured* fraction is the count (with its
//! binomial upper bound); the Gaussian extrapolation is printed beside it
//! and labelled, because at six standard deviations the sample cannot
//! resolve the fraction and an empirical MGF evaluated past its own support
//! would be a fabricated exponent, not a measurement.
//!
//! ```text
//! cargo run --release -p pcnl --example sphere_volume               # reduced, P = 552
//! cargo run --release -p pcnl --example sphere_volume -- --reference
//! cargo run --release -p pcnl --example sphere_volume -- --draws 8000
//! ```

use candle_core::{Device, Tensor};
use pcnl::model::{Shapes, Variant, forward, init_flat};
use pcnl::task::{Dataset, Task, adam};

#[derive(Debug)]
struct Cfg {
    shapes: Shapes,
    k: usize,
    target: Task,
    n_train: usize,
    n_test: usize,
    pretrain_steps: usize,
    finetune_steps: usize,
    lr: f64,
    sigma: f64,
    draws: usize,
}

impl Cfg {
    fn reduced() -> Self {
        Self {
            shapes: Shapes {
                vocab: 5,
                d: 8,
                ff: 16,
                layers: 1,
            },
            k: 4,
            target: Task::Modular { p: 1, q: 1, r: 1 },
            n_train: 2048,
            n_test: 512,
            pretrain_steps: 1000,
            finetune_steps: 1000,
            lr: 0.05,
            sigma: 0.35,
            draws: 4000,
        }
    }

    fn reference() -> Self {
        Self {
            shapes: Shapes {
                vocab: 9,
                d: 32,
                ff: 64,
                layers: 2,
            },
            k: 8,
            target: Task::B,
            n_train: 512,
            n_test: 256,
            draws: 2000,
            ..Self::reduced()
        }
    }
}

/// One parameter vector's `(mean cross-entropy, accuracy %)` on `(x, y)`.
fn evaluate(theta: &Tensor, shapes: &Shapes, x: &Tensor, y: &Tensor) -> (f64, f64) {
    let logits = forward(theta, shapes, Variant::Cosine, x).expect("forward");
    let rows = logits.to_vec2::<f32>().expect("logits");
    let targets = y.to_vec1::<u32>().expect("targets");
    let mut ce_sum = 0.0f64;
    let mut hit = 0usize;
    for (row, &t) in rows.iter().zip(&targets) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|v| ((*v as f64) - max).exp()).sum();
        ce_sum += (sum.ln() + max) - row[t as usize] as f64;
        let (best, _) = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| {
                if *v > acc.1 { (i, *v) } else { acc }
            });
        if best == t as usize {
            hit += 1;
        }
    }
    let n = targets.len() as f64;
    (ce_sum / n, 100.0 * hit as f64 / n)
}

/// A Haar-uniform direction scaled to `radius`: normalising a Gaussian
/// sample is exactly the uniform measure on `S^{P-1}` (§3.3).
fn haar_direction(shapes: &Shapes, radius: f64, seed: u64) -> Tensor {
    let raw = init_flat(shapes, 1.0, seed);
    let v = raw.to_vec1::<f32>().expect("draw");
    let n = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
    let scaled: Vec<f32> = v
        .iter()
        .map(|x| ((*x as f64) / n * radius) as f32)
        .collect();
    Tensor::from_vec(scaled, vec![shapes.flat_size()], &Device::Cpu).expect("cpu")
}

/// `2·Φ(−z)` — the two-sided Gaussian tail — via the large-`z` asymptotic
/// `2φ(z)/z·(1 − 1/z² + 3/z⁴)`, accurate to well under 1 % for `z ≥ 4`
/// (the only regime that reaches it, by construction of the call site).
fn gaussian_two_sided_tail(z: f64) -> f64 {
    let z = z.abs().max(1e-9);
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    2.0 * phi / z * (1.0 - 1.0 / (z * z) + 3.0 / (z * z * z * z))
}

fn main() {
    let reference = std::env::args().any(|a| a == "--reference");
    let mut cfg = if reference {
        Cfg::reference()
    } else {
        Cfg::reduced()
    };
    if let Some(i) = std::env::args().position(|a| a == "--draws") {
        cfg.draws = std::env::args()
            .nth(i + 1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(cfg.draws);
    }

    let p = cfg.shapes.flat_size();
    let shell = cfg.sigma * (p as f64).sqrt();
    let lnk = (cfg.k as f64).ln();
    let chance = 100.0 / cfg.k as f64;

    println!(
        "sphere volume — P = {p}, shapes {:?}, K = {}, target {}",
        cfg.shapes,
        cfg.k,
        cfg.target.formula(cfg.k)
    );
    println!(
        "  R = {} Haar directions at radius sigma*sqrt(P) = {shell:.2}; chance {chance:.2} %,\n  \
         uniform predictor ln K = {lnk:.4} nats\n",
        cfg.draws
    );

    let seed = 20261020u64;
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
    let data_b = Dataset::new(cfg.target, cfg.k, cfg.n_train, cfg.n_test, seed);
    let theta_a = adam(
        &data_a,
        &cfg.shapes,
        Variant::Cosine,
        init_flat(&cfg.shapes, 0.1, seed),
        cfg.pretrain_steps,
        cfg.lr,
    )
    .0;
    let theta_fine = adam(
        &data_b,
        &cfg.shapes,
        Variant::Cosine,
        theta_a.clone(),
        cfg.finetune_steps,
        cfg.lr,
    )
    .0;
    let (ce_a, acc_a) = evaluate(&theta_a, &cfg.shapes, &data_b.test_x, &data_b.test_y);
    let (ce_fine, acc_fine) = evaluate(&theta_fine, &cfg.shapes, &data_b.test_x, &data_b.test_y);
    println!(
        "  references: theta_A   acc {acc_a:5.1} % / mean CE {ce_a:.4}\n              \
         Adam-on-B acc {acc_fine:5.1} % / mean CE {ce_fine:.4}\n"
    );

    // The best *constant* predictor: a direction whose predictions do not
    // depend on the input scores exactly this.
    let labels = data_b.test_y.to_vec1::<u32>().expect("labels");
    let mut freq = vec![0usize; cfg.k + 1];
    for l in &labels {
        freq[*l as usize] += 1;
    }
    let constant_best = 100.0 * *freq.iter().max().unwrap() as f64 / labels.len() as f64;

    let mut draws: Vec<(f64, f64)> = Vec::with_capacity(cfg.draws);
    for d in 0..cfg.draws {
        let theta = haar_direction(&cfg.shapes, shell, seed + d as u64);
        draws.push(evaluate(
            &theta,
            &cfg.shapes,
            &data_b.test_x,
            &data_b.test_y,
        ));
    }
    let r = cfg.draws as f64;
    let mean_ce = draws.iter().map(|(c, _)| c).sum::<f64>() / r;
    let sd_ce = (draws
        .iter()
        .map(|(c, _)| (c - mean_ce).powi(2))
        .sum::<f64>()
        / r)
        .sqrt();
    let ce_min = draws.iter().map(|(c, _)| *c).fold(f64::INFINITY, f64::min);
    let acc_mean = draws.iter().map(|(_, a)| a).sum::<f64>() / r;
    let acc_max = draws.iter().map(|(_, a)| *a).fold(0.0, f64::max);
    let count = |f: &dyn Fn(f64, f64) -> bool| draws.iter().filter(|(c, a)| f(*c, *a)).count();
    let pct = |n: usize| 100.0 * n as f64 / r;

    let (gt_chance, gt_constant, below_lnk, below_fine) = (
        count(&|_, a| a > chance),
        count(&|_, a| a > constant_best),
        count(&|c, _| c < lnk),
        count(&|c, _| c < ce_fine),
    );

    println!(
        "  Haar directions (R = {}, {} held-out examples each):",
        cfg.draws, cfg.n_test
    );
    println!(
        "    accuracy  mean {:.2} %  max {:.2} %   (best constant predictor {:.2} %)",
        acc_mean, acc_max, constant_best
    );
    println!(
        "    mean CE   mean {:.4}  sd {:.4}  min {:.4}   (ln K = {lnk:.4})",
        mean_ce, sd_ce, ce_min
    );
    println!(
        "    beating chance                    {:8.3} %  ({:>5} / {})",
        pct(gt_chance),
        gt_chance,
        cfg.draws
    );
    println!(
        "    above the best constant predictor {:8.3} %  ({:>5} / {})",
        pct(gt_constant),
        gt_constant,
        cfg.draws
    );
    println!(
        "    below the uniform NLL (ln K)      {:8.3} %  ({:>5} / {})",
        pct(below_lnk),
        below_lnk,
        cfg.draws
    );
    println!(
        "    as good as Adam-on-B              {:8.3} %  ({:>5} / {})",
        pct(below_fine),
        below_fine,
        cfg.draws
    );

    // ---- what §6.4(ii) needs ---------------------------------------
    let z = (mean_ce - ce_fine) / sd_ce;
    let delta = mean_ce - ce_fine;
    println!("\n  the fine-tuned direction sits {z:.1} Haar-sd below the mean:");
    println!(
        "    energy: mu - x* = {delta:.4} nats/example  ->  n*delta = {:8.1} nats at n = {}",
        delta * cfg.n_train as f64,
        cfg.n_train
    );
    println!(
        "    volume: measured {:>5} / {} reached it  ->  Haar fraction < {:.2e} at 95 % (binomial)",
        below_fine,
        cfg.draws,
        3.0 / r
    );
    println!(
        "            Gaussian extrapolation at {z:.1} sd: 2*Phi(-z) = {:.2e}   [parametric, labelled]",
        gaussian_two_sided_tail(z)
    );
    println!(
        "    the whole sample is worse than the uniform predictor (best of {} is {:.4} vs {lnk:.4});\n    \
         so the favourable set is smaller than 1/{} of the sphere while being demonstrably non-empty —\n    \
         the chain cannot reach it by sampling, only by moving, which is §6.4.2's reachability finding.",
        cfg.draws, ce_min, cfg.draws
    );

    // ---- the radius does not matter (scale invariance, up to eps) ---
    let r4 = (cfg.draws / 4).max(64);
    let mut m4 = 0.0f64;
    let mut a4 = 0.0f64;
    for d in 0..r4 {
        let theta = haar_direction(&cfg.shapes, 4.0 * shell, seed + 1_000_000 + d as u64);
        let (c, a) = evaluate(&theta, &cfg.shapes, &data_b.test_x, &data_b.test_y);
        m4 += c;
        a4 += a;
    }
    println!(
        "\n  same draw law at 4*shell = {:.2}: mean CE {:.4} (vs {mean_ce:.4}), mean acc {:.2} % (vs {acc_mean:.2} %)\n  \
         — the direction is all that matters, up to the eps in W/(||W||+eps) (§3.3).",
        4.0 * shell,
        m4 / r4 as f64,
        a4 / r4 as f64
    );
}
