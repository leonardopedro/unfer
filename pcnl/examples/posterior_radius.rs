//! `posterior_radius` — where does `π_λ` actually live radially?
//!
//! Every §6.4–§6.6 measurement compares a chain's `‖θ‖` against the prior's
//! shell `σ√P = 45.2`, and §6.5.6(c) read a radius of `≈ 10³` as proof that the
//! chain had left the target.  That inference only holds if `π_λ` puts its mass
//! on the shell, which is a claim about the target, not the sampler.  This
//! example evaluates the target directly:
//!
//! ```text
//! Φ_λ(θ) = ‖θ‖²/(2σ²) + λ·N·L(θ)      π_λ ∝ exp(−Φ_λ)
//! ```
//!
//! along a radial ray from the prior mean, for a few directions.  The prior
//! term is quadratic in `r` (`r²/2σ²`), the likelihood term is
//! `λ·N·L(r·d)` and `L` *decreases* with `r` as the softmax sharpens, so the
//! minimiser is wherever `r/σ² = λ·N·|dL/dr|` — not automatically `σ√P`.  With
//! `N = 512` and `1/σ² = 8.16`, the likelihood carries 60× the prior's
//! precision, so the balance point can be far outside the shell.
//!
//! Both a `θ_A`-direction ray and Haar directions are scanned, because the
//! likelihood's radial slope depends on the direction: a Haar direction is
//! confidently wrong and its loss may not improve with radius at all.
//!
//! ```text
//! cargo run --release -p pcnl --example posterior_radius
//! ```

use pcnl::experiment::Config;
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::task::{Dataset, Task, adam};

const SEED: u64 = 20261020;
const LAMBDAS: [f64; 2] = [1.0, 16.0];

fn cfg_for() -> Config {
    Config {
        seed: SEED,
        shapes: Shapes {
            vocab: 9,
            d: 32,
            ff: 64,
            layers: 2,
        },
        n_train: 512,
        n_test: 256,
        pretrain_steps: 1_000,
        pretrain_adam: true,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    let cfg = cfg_for();
    let p = cfg.shapes.flat_size();
    let sigma = cfg.sigma;
    let n = cfg.n_train as f64;
    let shell = sigma * (p as f64).sqrt();
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, SEED);
    let data = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, SEED);
    let theta_a = adam(
        &data_a,
        &cfg.shapes,
        Variant::Cosine,
        init_flat(&cfg.shapes, 0.1, SEED),
        cfg.pretrain_steps,
        cfg.adam_lr,
    )
    .0;
    let dir_a = theta_a
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    let na = norm(&dir_a);
    // a Haar draw of the same norm, for the "confidently wrong direction" case
    let mut rng_state = 0x9e3779b97f4a7c15u64;
    let haar: Vec<f32> = (0..p)
        .map(|_| {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u = ((rng_state >> 11) as f64) / ((1u64 << 53) as f64);
            // Box–Muller from one uniform stream
            let v = ((rng_state >> 29) as f64) / ((1u64 << 35) as f64);
            let z = (-2.0 * u.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            z as f32
        })
        .collect();
    let nh = norm(&haar);

    let grid: Vec<f64> = vec![
        10.0, 20.0, shell, 60.0, 90.0, 150.0, 300.0, 600.0, 1_000.0, 2_000.0, 4_000.0,
    ];
    println!(
        "radial profile of the target, cosine variant, P = {p}, σ = {sigma}, N = {n}\n\
         Φ_λ(r) = r²/(2σ²) + λ·N·L(r·d);  shell σ√P = {shell:.1}; prior precision \
         1/σ² = {:.2}, likelihood weight λN = {n}\n",
        1.0 / (sigma * sigma)
    );

    let dirs: [(&str, &Vec<f32>, f64); 2] =
        [("θ_A direction", &dir_a, na), ("Haar direction", &haar, nh)];
    for (name, d, nd) in dirs {
        println!("-- {name} (‖d‖ = {nd:.1})");
        print!("{:>7}  {:>10}", "r", "L(r·d)");
        for lam in LAMBDAS {
            print!("  {:>11}", format!("Φ_λ={lam}"));
        }
        println!();
        let mut best = vec![(f64::NAN, f64::NAN); LAMBDAS.len()];
        for &r in &grid {
            let scale = (r / nd) as f32;
            let row: Vec<f32> = d.iter().map(|v| v * scale).collect();
            let theta =
                candle_core::Tensor::from_vec(row, vec![p], theta_a.device()).expect("ray point");
            let loss = data
                .nll(&theta, &cfg.shapes, Variant::Cosine)
                .expect("loss");
            print!("{r:>7.0}  {loss:>10.5}");
            for (i, lam) in LAMBDAS.iter().enumerate() {
                let phi = r * r / (2.0 * sigma * sigma) + lam * n * loss;
                if phi.is_finite() && (best[i].1.is_nan() || phi < best[i].1) {
                    best[i] = (r, phi);
                }
                print!("  {phi:>11.1}");
            }
            println!();
        }
        for (i, lam) in LAMBDAS.iter().enumerate() {
            println!(
                "   λ = {lam:>4}: argmin Φ over the grid is r = {:.0} ({:.1}× the shell)",
                best[i].0,
                best[i].0 / shell
            );
        }
        println!();
    }
    println!(
        "reading: the prior's quadratic sets a floor at σ√P and the likelihood's\n\
         radial slope pulls outward; whichever wins decides where the target's mass\n\
         is.  A minimiser far outside the shell means \"the chain left the shell\" is\n\
         not evidence of a broken sampler — and \"the shell held\" is not evidence of\n\
         correctness, only that the drift was too weak to follow the target."
    );
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}
