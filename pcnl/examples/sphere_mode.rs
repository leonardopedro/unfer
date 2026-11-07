//! `sphere_mode` — is Adam's direction the mode of the sphere target, or is
//! there somewhere better on `S^{P−1}`?
//!
//! `sphere_chain` measures the *posterior*: at `β₀ = 0.003` the preconditioned
//! chain moves and its BMA scores 51.17 %, against 49.22 % at the direction it
//! started from.  A posterior mean above the start is suspicious on its own —
//! it could be noise in 15 kept draws — so this measures the *mode* directly,
//! with no kernel involved: Riemannian gradient descent (exponential-map steps
//! with Armijo backtracking) on `−mean NLL` over the unit sphere, from Adam's
//! direction and from several Haar draws.
//!
//! Three questions, in one run:
//!
//! 1. **Is `u_Adam` stationary?**  `‖∇_T L‖ = 2.88` says no; the descent then
//!    says how much is available and how far it is.
//! 2. **How good is the sphere's mode?**  Its accuracy/NLL against Adam's
//!    49.22 % / 1.3136 bounds what any sampler of this target could report at
//!    best, and separates "the chain found something better" from "the chain
//!    wandered".
//! 3. **Is it the same basin from a Haar start?**  If independent ascents land
//!    on the same accuracy, the mode is robust; if they land at chance, the
//!    fine-tune's basin is tiny (which is §6.4(ii)'s volume statement).
//!
//! ```text
//! cargo run --release -p pcnl --example sphere_mode
//! ```

use candle_core::{Device, Tensor};
use pcnl::experiment::Config;
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::norm;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};

const MAX_STEPS: usize = 150;

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

fn to_tensor(v: &[f64], dev: &Device) -> Tensor {
    Tensor::from_vec(v.iter().map(|x| *x as f32).collect(), vec![v.len()], dev).expect("tensor")
}

/// One Riemannian descent run.  `s` is the geodesic angle; a successful step
/// grows it and a failed one halves it, so the run finds the local curvature
/// scale by itself instead of needing it as an input.
fn descend(
    u0: &[f64],
    radius: f64,
    data: &Dataset,
    cfg: &Config,
    dev: &Device,
) -> (Vec<f64>, f64, f64, usize, f64) {
    let loss_grad = |u: &[f64]| -> (f64, Vec<f64>) {
        let th = to_tensor(u, dev).affine(radius, 0.0).expect("theta");
        let (l, g) = data
            .loss_and_grad(&th, &cfg.shapes, Variant::Cosine)
            .expect("loss");
        (l, as_vec(&g))
    };
    let mut u = u0.to_vec();
    let (mut l, mut g) = loss_grad(&u);
    let l_start = l;
    let mut s = 1e-3f64;
    let mut used = 0usize;
    for _ in 0..MAX_STEPS {
        // tangential gradient and its unit direction
        let gu: f64 = g.iter().zip(&u).map(|(a, b)| a * b).sum();
        let gt: Vec<f64> = g.iter().zip(&u).map(|(a, b)| a - gu * b).collect();
        let gn = gt.iter().map(|x| x * x).sum::<f64>().sqrt();
        if gn < 1e-6 {
            break;
        }
        let dir: Vec<f64> = gt.iter().map(|x| -x / gn).collect();
        let mut accepted = false;
        for _ in 0..24 {
            let cand: Vec<f64> = (0..u.len())
                .map(|i| s.cos() * u[i] + s.sin() * dir[i])
                .collect();
            let (l1, g1) = loss_grad(&cand);
            if l1 < l - 1e-10 {
                u = cand;
                l = l1;
                g = g1;
                s = (s * 1.5).min(5e-2);
                accepted = true;
                used += 1;
                break;
            }
            s *= 0.5;
            if s < 1e-8 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    let gu: f64 = g.iter().zip(&u).map(|(a, b)| a * b).sum();
    let gt: Vec<f64> = g.iter().zip(&u).map(|(a, b)| a - gu * b).collect();
    (
        u,
        l,
        l_start,
        used,
        gt.iter().map(|x| x * x).sum::<f64>().sqrt(),
    )
}

fn main() {
    let cfg = Config::default();
    let dev = Device::Cpu;
    let p = cfg.shapes.flat_size();
    let radius = cfg.sigma * (p as f64).sqrt();
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
    let u_adam = as_vec(&unit(&theta_adam));

    let score = |u: &[f64]| -> (f64, f64, f64) {
        let th = to_tensor(u, &dev).affine(radius, 0.0).expect("theta");
        (
            data_b
                .accuracy(&th, &cfg.shapes, Variant::Cosine)
                .expect("acc"),
            data_b.nll(&th, &cfg.shapes, Variant::Cosine).expect("nll"),
            norm(&th).expect("norm"),
        )
    };

    let (a0, n0, _) = score(&u_adam);
    println!(
        "sphere_mode — Riemannian descent on −mean NLL over S^{{{p}−1}}\n\
         ρ = σ√P = {radius:.3}, task B, N = {}, Adam's direction: acc {a0:.2}% NLL {n0:.4}\n",
        cfg.n_train
    );

    // 1/2. From Adam's direction.
    let (u_m, l_m, l_s, steps, gn) = descend(&u_adam, radius, &data_b, &cfg, &dev);
    let (am, nm, _) = score(&u_m);
    let ang = {
        let d: f64 = u_m.iter().zip(&u_adam).map(|(a, b)| a * b).sum();
        d.clamp(-1.0, 1.0).acos()
    };
    println!(
        "from u_Adam: {steps} accepted descent steps, mean NLL {l_s:.4} → {l_m:.4}, \
         |∇_T L| at the end {gn:.2e}\n  \
         mode: acc {am:.2}%  NLL {nm:.4}   (Δ acc {:+.2} pts, Δ NLL {:+.4})\n  \
         angle from u_Adam: {ang:.3e} rad ({:.1}× the trust radius 1e-4)\n",
        am - a0,
        nm - n0,
        ang / 1e-4
    );

    // 3. From Haar draws: the same basin, or chance?
    println!("from Haar draws (same descent, 3 starts):");
    let mut haar = init_flat(&cfg.shapes, 1.0, cfg.seed.wrapping_add(99));
    for k in 0..3 {
        if k > 0 {
            haar = init_flat(&cfg.shapes, 1.0, cfg.seed.wrapping_add(99 + k as u64));
        }
        let u0 = as_vec(&unit(&haar));
        let (a_i, n_i, _) = score(&u0);
        let (u_h, l_h, _, steps_h, _) = descend(&u0, radius, &data_b, &cfg, &dev);
        let (a_h, n_h, _) = score(&u_h);
        println!(
            "  start acc {a_i:>5.2}% NLL {n_i:>6.4}  →  {steps_h:>3} steps, \
             mean NLL {l_h:>6.4}, acc {a_h:>5.2}% NLL {n_h:>6.4}"
        );
    }

    println!(
        "\nreading: the sphere's mode is where the descent stops, and it is inside the\n\
         basin Adam's fine-tune reached only if the angle from u_Adam is a handful of\n\
         trust radii and the objective is better by more than the chain's spread.  A\n\
         Haar start that ascends to the same accuracy means the basin is reachable in\n\
         principle and the volume statement (§6.4(ii)) is about random draws, not about\n\
         the basin's size under optimisation."
    );
}
