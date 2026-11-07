//! `subspace_bound` — can a low-rank (subspace) metric pay the `√P` gap that a
//! diagonal one cannot?
//!
//! The diagonal result is a bound, not a failure of implementation:
//! `curvature_probe` measures `λN‖g‖σβ = 77.4 ≫ 2` at the published `β`, and a
//! mean-1 diagonal metric can buy only the curvature's *heterogeneity*
//! (31.6× in `τ²`).  A subspace metric buys a different quantity: shrink the
//! proposal along the `m` directions that carry the curvature and leave the
//! other `P − m` alone, `C = I − (1−c)·P_m`.  Then
//!
//! ```text
//! τ²·Tr(C·H) = τ²·Tr(H)·(1 − (1−c)·f),   f = Tr(P_m H)/Tr(H),
//! ```
//!
//! so the best any subspace metric can do is the factor `1/(1−f)` — the trace
//! *concentration*.  This measures `f(m)` directly, with no kernel involved:
//! power iteration on the finite-difference curvature operator
//! `A·v = ρ·(g(θ+ερv) − g(θ))/ε` (the Hessian in `u` coordinates) for the top
//! directions, and Hutchinson with Rademacher probes for `Tr(H)`.
//!
//! The comparison is then arithmetic.  Acceptance at the published `β` needs the
//! exponent's cost to fall from 77.4 to ≲ 2 — a factor of ~39, so `f ≳ 0.974`.
//! Whatever `f` comes out as answers whether implementing the subspace proposal
//! is worth the code: below that, no such proposal accepts at `β₀ = 0.3`.
//!
//! ```text
//! cargo run --release -p pcnl --example subspace_bound
//! ```

use candle_core::{Device, Tensor};
use pcnl::experiment::Config;
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::norm;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

const M: usize = 8;

fn unit(v: &Tensor) -> Tensor {
    v.affine(1.0 / norm(v).expect("norm"), 0.0).expect("unit")
}

fn as_vec(t: &Tensor) -> Vec<f64> {
    t.flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec")
        .iter()
        .map(|x| *x as f64)
        .collect()
}

fn main() {
    let cfg = Config::default();
    let dev = Device::Cpu;
    let p = cfg.shapes.flat_size();
    let radius = cfg.sigma * (p as f64).sqrt();
    let eps: f64 = 1e-5; // angular radius of the finite difference: 1e-5·45.2 = 4.5e-4 in θ
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
    let data_b = Dataset::new(
        cfg.task_b,
        cfg.k,
        cfg.n_train,
        cfg.n_test,
        cfg.seed.wrapping_add(1),
    );
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
    let u: Vec<f64> = as_vec(&unit(&theta_adam));

    let grad_at = |u: &[f64]| -> Vec<f64> {
        let th = Tensor::from_vec(
            u.iter().map(|x| (*x * radius) as f32).collect(),
            vec![u.len()],
            &dev,
        )
        .expect("theta");
        let (_, g) = data_b
            .loss_and_grad(&th, &cfg.shapes, Variant::Cosine)
            .expect("grad");
        as_vec(&g)
    };
    let g0 = grad_at(&u);
    // A·v: the Hessian in u coordinates, applied to a unit tangent v.
    let apply = |u: &[f64], v: &[f64]| -> Vec<f64> {
        let cand: Vec<f64> = (0..p)
            .map(|i| eps.cos() * u[i] + eps.sin() * v[i])
            .collect();
        let g1 = grad_at(&cand);
        (0..p).map(|i| radius * (g1[i] - g0[i]) / eps).collect()
    };
    let project = |x: &[f64], u: &[f64]| -> Vec<f64> {
        let c: f64 = x.iter().zip(u).map(|(a, b)| a * b).sum();
        x.iter().zip(u).map(|(a, b)| a - c * b).collect()
    };
    let normalize = |x: &[f64]| -> Vec<f64> {
        let n = x.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300);
        x.iter().map(|y| y / n).collect()
    };

    println!(
        "subspace_bound — trace concentration of the sphere target's curvature at u_Adam\n\
         P = {p}, ρ = σ√P = {radius:.3}, finite difference ε = {eps:.0e} rad\n"
    );

    // Power iteration with deflation for the top-M eigenpairs of A (A is
    // symmetric up to the finite-difference error; Rayleigh quotients are what
    // we need, not eigenvectors to machine precision).
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut quotients: Vec<f64> = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    for k in 0..M {
        let mut v: Vec<f64> = (0..p)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        v = project(&v, &u);
        v = normalize(&v);
        for _ in 0..6 {
            let mut w = apply(&u, &v);
            for b in basis.iter() {
                let c: f64 = w.iter().zip(b).map(|(x, y)| x * y).sum();
                for i in 0..p {
                    w[i] -= c * b[i];
                }
            }
            w = project(&w, &u);
            let n: f64 = w.iter().map(|x| x * x).sum::<f64>();
            if !n.is_finite() || n <= 0.0 {
                break;
            }
            v = w.iter().map(|x| x / n.sqrt()).collect();
        }
        let mut av = apply(&u, &v);
        for b in basis.iter() {
            let c: f64 = av.iter().zip(b).map(|(x, y)| x * y).sum();
            for i in 0..p {
                av[i] -= c * b[i];
            }
        }
        let q: f64 = av.iter().zip(&v).map(|(x, y)| x * y).sum();
        quotients.push(q);
        basis.push(v);
        println!("  λ_{} ≈ {:>12.3e}", k + 1, q);
    }

    // Tr(H) by Hutchinson: E[vᵀAv] = Tr(A) for Rademacher v.
    let probes = 16;
    let mut trace = 0.0;
    for _ in 0..probes {
        let v: Vec<f64> = (0..p)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let v = project(&v, &u);
        let av = apply(&u, &v);
        let av = project(&av, &u);
        trace += av.iter().zip(&v).map(|(x, y)| x * y).sum::<f64>();
    }
    let trace = trace / probes as f64;
    println!(
        "\n  Tr(H) by Hutchinson over {probes} Rademacher probes ≈ {trace:.3e}\
         ({:.3e} per coordinate)\n",
        trace / p as f64
    );

    println!("  m   Σλ_k / Tr(H)   best reduction 1/(1−f)   β it would allow");
    let mut acc = 0.0;
    for (k, q) in quotients.iter().enumerate() {
        acc += *q;
        let f = acc / trace;
        let reduction = if f < 1.0 {
            1.0 / (1.0 - f)
        } else {
            f64::INFINITY
        };
        // cost(β) = 77.4·(β/0.3); accepting needs cost ≤ 2 after the reduction.
        let beta_ok = 0.3 * (2.0 / 77.4) * reduction;
        println!(
            "  {:>2}   {:>12.6}   {:>17.2}×   {:.4}",
            k + 1,
            f,
            reduction,
            beta_ok.min(0.3)
        );
    }

    println!(
        "\nreading: acceptance at β₀ = 0.3 needs the cost to fall 39×, i.e. f ≳ 0.974\n\
         at whatever m the metric uses.  f is the fraction of the curvature's *trace*\n\
         the top-m subspace carries — the only thing a subspace metric can trade\n\
         against the √P.  If f is far below that, no subspace proposal of this form\n\
         accepts at the published β, and the honest options are a different target or\n\
         a neighbourhood posterior."
    );
}
