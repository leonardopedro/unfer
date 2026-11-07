//! `curvature_probe` — *why* the exact spherical filter rejects everything from
//! the solution's direction, and what a metric could do about it.
//!
//! `examples/sphere_chain.rs` measures the symptom (0.0 % acceptance at
//! `β₀ = 0.3` from `u_Adam` over a whole run).  This measures the cause at that
//! point, and the acceptance column comes from the **kernel itself** — an
//! earlier version of this file re-implemented the ratio inline, forgot the
//! reverse drift `m₁`, and reported 92 % where the kernel accepts 0/15.  The
//! filter's arithmetic is subtle enough that a second implementation is not a
//! cross-check, it is a second bug.
//!
//! 1. **The moves' lengths, and the quantity that decides acceptance.**  For
//!    `τ = βσ/ρ` the pCN exponent is `λN·ΔL + (‖e‖² − ‖w−m‖²)/2τ²`; with
//!    `ΔL ≈ −g·δ` and the curvature compensating half of it, the exponent is
//!    `≈ ½λN‖g‖·ρτ·√P = ½λN‖g‖σβ`.  That product is the number to look at, and
//!    it is printed per `β`.
//! 2. **The trust radius.**  `L(u − s·ĝ)`: the angle at which the loss stops
//!    falling linearly is the largest step the local quadratic model covers.
//! 3. **Whether a diagonal metric can reach it.**  The spread of `|Δg_i/v_i|`
//!    (a diagonal secant) says how much of the sharpness is *heterogeneous*
//!    (fixable by `d`) versus common to every coordinate (only a smaller `β`
//!    fixes that).
//! 4. **The kernel's verdict** over the β grid, and the run-length `r` at each.
//!
//! ```text
//! cargo run --release -p pcnl --example curvature_probe
//! ```

use candle_core::Tensor;
use pcnl::experiment::Config;
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, norm};
use pcnl::sphere::SpherePcn;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

fn unit(v: &Tensor) -> Tensor {
    v.affine(1.0 / norm(v).expect("norm"), 0.0).expect("unit")
}

fn main() {
    let cfg = Config::default();
    let p = cfg.shapes.flat_size();
    let radius = cfg.sigma * (p as f64).sqrt();
    let data_b = Dataset::new(
        cfg.task_b,
        cfg.k,
        cfg.n_train,
        cfg.n_test,
        cfg.seed.wrapping_add(1),
    );
    // The same pre-train → fine-tune path `sphere_chain` starts from.
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
    let theta0 = init_flat(&cfg.shapes, cfg.init_std, cfg.seed.wrapping_add(2));
    let (theta_a, _) = momentum_sgd(
        &data_a,
        &cfg.shapes,
        Variant::Cosine,
        theta0,
        cfg.pretrain_steps,
        cfg.pretrain_lr,
        0.9,
    );
    let (theta_adam, _) = adam(
        &data_b,
        &cfg.shapes,
        Variant::Cosine,
        theta_a,
        cfg.adam_steps_on_b,
        cfg.adam_lr,
    );
    let u_hat = unit(&theta_adam);
    let theta = u_hat.affine(radius, 0.0).expect("shell");
    let (l0, g) = data_b
        .loss_and_grad(&theta, &cfg.shapes, Variant::Cosine)
        .expect("grad");
    let u: Vec<f64> = u_hat
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| *x as f64)
        .collect();
    let gv: Vec<f64> = g
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| *x as f64)
        .collect();
    let gu: f64 = gv.iter().zip(&u).map(|(a, b)| a * b).sum();
    let g_tan: Vec<f64> = gv.iter().zip(&u).map(|(a, b)| a - gu * b).collect();
    let g_norm = (gv.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let g_tan_norm = (g_tan.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let n = cfg.n_train as f64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);

    println!(
        "curvature_probe — sphere target at Adam's direction\n\
         P = {p}, ρ = σ√P = {radius:.3}, N = {n:.0}, λ = 1, σ = {}\n\
         task-B mean NLL at ρ·u_Adam = {l0:.4},  θ_Adam NLL = {:.4}\n\
         ‖∇L‖ = {g_norm:.4e}, ‖∇_T L‖ = {g_tan_norm:.4e} ({:.2}% tangential)\n",
        cfg.sigma,
        data_b
            .nll(&theta_adam, &cfg.shapes, Variant::Cosine)
            .expect("nll"),
        100.0 * g_tan_norm / g_norm
    );

    // 1. Lengths, and the product that controls the filter.
    println!("1. the moves, and the exponent they imply");
    println!(
        "   {:>7}  {:>9}  {:>9}  {:>6}  {:>12}  {:>8}",
        "β", "‖e‖=τ√P", "‖m‖", "m/e", "λN‖g‖σβ", "r"
    );
    for beta in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001] {
        let tau = beta * cfg.sigma / radius;
        let coeff = beta * beta * cfg.sigma * cfg.sigma / (2.0 * radius) * n;
        let m_norm = coeff * g_tan_norm;
        // ½λN‖g‖σβ is the typical |pCN exponent|; r = βσ‖g_summed‖/(2√P) is
        // §6.5.5's information-to-noise ratio, printed here for the comparison.
        let g_sum = n * g_tan_norm;
        println!(
            "   {beta:>7}  {:>9.4}  {m_norm:>9.4}  {:>6.2}  {:>12.3}  {:>8.4}",
            tau * (p as f64).sqrt(),
            m_norm / (tau * (p as f64).sqrt()),
            0.5 * n * g_tan_norm * cfg.sigma * beta,
            beta * cfg.sigma * g_sum / (2.0 * (p as f64).sqrt())
        );
    }

    // 2. The trust radius along −ĝ.
    println!("\n2. the trust radius: mean NLL along −ĝ");
    println!("   {:>9}  {:>9}  {:>11}", "s (rad)", "mean NLL", "Δ vs s=0");
    for s in [0.3f64, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5] {
        let dir: Vec<f64> = g_tan.iter().map(|x| -x / g_tan_norm).collect();
        let un: Vec<f64> = (0..p).map(|i| s.cos() * u[i] + s.sin() * dir[i]).collect();
        let tn = Tensor::from_vec(
            un.iter().map(|x| (*x * radius) as f32).collect(),
            vec![p],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let l = data_b
            .loss_and_grad(&tn, &cfg.shapes, Variant::Cosine)
            .expect("trust")
            .0;
        println!("   {s:>9.4}  {l:>9.4}  {:>11.4}", l - l0);
    }

    // 3. Heterogeneity of the diagonal secant.
    println!("\n3. diagonal secant |Δg_i / v_i| at τ = 0.3/√P");
    let tau = 0.3 * cfg.sigma / radius;
    let mut secants: Vec<f64> = Vec::new();
    for _ in 0..24 {
        let raw: Vec<f64> = (0..p)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * tau)
            .collect();
        let c: f64 = raw.iter().zip(&u).map(|(a, b)| a * b).sum();
        let e: Vec<f64> = raw.iter().zip(&u).map(|(a, b)| a - c * b).collect();
        let a = e.iter().map(|x| x * x).sum::<f64>().sqrt();
        let un: Vec<f64> = (0..p)
            .map(|i| a.cos() * u[i] + a.sin() * e[i] / a.max(1e-300))
            .collect();
        let tn = Tensor::from_vec(
            un.iter().map(|x| (*x * radius) as f32).collect(),
            vec![p],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let g1 = data_b
            .loss_and_grad(&tn, &cfg.shapes, Variant::Cosine)
            .expect("secant")
            .1;
        let g1v: Vec<f64> = g1
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| *x as f64)
            .collect();
        for i in 0..p {
            if e[i].abs() > 0.25 * tau {
                secants.push((g1v[i] - gv[i]).abs() / e[i].abs());
            }
        }
    }
    secants.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| secants[((secants.len() - 1) as f64 * f) as usize];
    let mean = secants.iter().sum::<f64>() / secants.len() as f64;
    println!(
        "   n = {}, median {:.3e}, mean {:.3e}, p90 {:.3e}, p99 {:.3e}, max {:.3e}\n   \
         mean/median {:>6.1}× (the most a mean-1 diagonal metric can buy in τ²)",
        secants.len(),
        q(0.5),
        mean,
        q(0.9),
        q(0.99),
        q(1.0),
        mean / q(0.5)
    );

    // 4. The kernel's own acceptance, 10 steps from u_Adam, fresh chain each β.
    println!("\n4. the kernel's verdict: 10 steps from u_Adam, λ = 1");
    println!(
        "   {:>7}  {:>11}  {:>10}  {:>10}",
        "β", "accepted", "mean NLL", "‖u‖"
    );
    for beta in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001] {
        let mut c = SpherePcn::new(
            u_hat.clone(),
            radius,
            cfg.sigma,
            SgPcnlScheduler {
                beta_0: beta,
                gamma: 0.0,
                kappa: 0.5,
            },
            cfg.n_train,
            1.0,
        );
        let mut r = rand::rngs::StdRng::seed_from_u64(11);
        for _ in 0..10 {
            c.step(
                |th: &Tensor| data_b.loss_and_grad(th, &cfg.shapes, Variant::Cosine),
                &mut r,
            )
            .expect("step");
        }
        println!(
            "   {beta:>7}  {:>6}/10  {:>10.4}  {:>10.6}",
            c.accepted,
            data_b
                .nll(&c.theta().expect("theta"), &cfg.shapes, Variant::Cosine)
                .expect("nll"),
            norm(&c.u).expect("norm")
        );
    }

    println!(
        "\nreading: acceptance needs λN‖g‖σβ ≲ 2 (or the pCN exponent is negative\n\
         for every draw), while an *informative* drift needs r ≳ 1 — and r is the\n\
         same product divided by 2√P.  In P = 16672 those cannot both hold at the\n\
         solution's direction, which is the difference between §6.5.7's \"the target\n\
         contains the solution\" and \"the target can be sampled there\"."
    );
}
