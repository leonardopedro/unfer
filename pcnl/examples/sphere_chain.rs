//! `sphere_chain` — the target §6.5.7 says the Gaussian prior should have
//! been: a uniform prior on the *direction* of `θ`, sampled by the exact
//! spherical pCN of [`pcnl::sphere`].
//!
//! **The question.**  `pcnl/examples/adam_start.rs` measured that the Task-B
//! fine-tune lives at `‖θ‖ = 1272.6`, 28.2× the `σ√P = 45.2` shell, where
//! `N(0, σ²I)` pays `r²/2σ² ≈ 6.6 × 10⁶` nats.  So `π ∝ N(0,σ²I)·exp(−λNL)`
//! assigns the solution vanishing mass and every ambient sampler reports the
//! shell's Haar-typical direction — chance — no matter what it is handed.
//! Since the cosine forward pass is scale-invariant, what the model actually
//! constrains is the *direction*; this runs the kernel on the sphere, where
//! the solution's direction has full mass.
//!
//! **Cells.**  `β₀ ∈ {0.3, 0.03, 0.01}` × start ∈ {Adam's direction, Haar}:
//!
//! * `β₀ = 0.3` is the published proposal.  In 16 672 dimensions no move that
//!   size survives the filter, so the chain either freezes or diffuses; the
//!   acceptance column says which, and a frozen chain's "retention" is the
//!   start's predictive, not a sample.
//! * `β₀ = 0.03` is 1.4 in ambient terms — the scale that held a neighbourhood
//!   in the ambient target (17.6 %, NLL 1.95) without containing the solution.
//! * the Haar start is the search question: the informative directions are a
//!   sub-1/2000 fraction of the sphere (§6.4(ii)), so this should stay at
//!   chance — the difference from the Adam start is what retention is worth.
//!
//! **Measured (2026-11-05): the target contains the solution, and the filter
//! refuses to move there.**  1 200 steps, one chain, λ = 1, ρ = σ√P:
//!
//! | β₀ | start | accept | BMA acc | BMA NLL |
//! |---:|---|---:|---:|---:|
//! | 0.3 | Adam | **0.0 %** | 49.22 % | 1.3136 |
//! | 0.3 | Haar | 32.2 % | 11.72 % | 2.1818 |
//! | 0.03 | Adam | **0.0 %** | 49.22 % | 1.3136 |
//! | 0.03 | Haar | **100 %** | 11.72 % | 2.1832 |
//! | 0.01 | Adam | 0.6 % | 46.88 % | 1.3281 |
//! | 0.01 | Haar | 99.7 % | 11.72 % | **2.9027** |
//!
//! Three things follow.
//!
//! 1. **Scale invariance holds at the target level**: `ρ·u_Adam` scores
//!    49.22 % / 1.3136, *identical* to `θ_Adam` at 28× the radius.  The sphere
//!    target therefore contains the solution, which the Gaussian one did not —
//!    the §6.5.7 diagnosis was right about the prior.
//! 2. **It cannot be sampled at any proposal scale used in this study.**  From
//!    the solution's direction the exact filter accepts **nothing** at
//!    `β₀ = 0.3` and at `β₀ = 0.03`, and 0.6 % (≈7 moves in 1 200 steps) at
//!    `β₀ = 0.01`.  The 49.22 % rows are a *frozen start*, and the acceptance
//!    column is what says so.
//! 3. **Acceptance is a property of position, not of the kernel.**  The same
//!    chain from a Haar direction accepts 100 % at `β₀ = 0.03` and mixes
//!    freely — at chance, where the likelihood is flat.  So the ambient runs'
//!    90–100 % acceptance (§6.5.6) was a symptom of sampling the flat region,
//!    not evidence that the sampler was working.
//!
//! The two mechanism findings are now complementary: the isotropic Gaussian
//! prior excludes the solution (§6.5.7), and once the prior is repaired to a
//! Haar law on the sphere, a fixed isotropic pCN step still cannot both move
//! and accept in 16 672 dimensions.  The next lever is the proposal's
//! *preconditioning*, not its size — with a step scaled to the local curvature
//! (per-coordinate or Fisher/preconditioned pCN) the body of §6.4–6.5's
//! "prior-dominated" evidence would have to be re-read.
//!
//! ```text
//! cargo run --release -p pcnl --example sphere_chain            # all six cells
//! cargo run --release -p pcnl --example sphere_chain -- 400     # shorter
//! ```

use candle_core::Tensor;
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, norm};
use pcnl::sphere::SpherePcn;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};
use rand::SeedableRng;

const SEED: u64 = 20261020;
const BETAS: [f64; 3] = [0.3, 0.03, 0.01];

fn unit(v: &Tensor) -> Tensor {
    let n = norm(v).expect("norm");
    v.affine(1.0 / n, 0.0).expect("unit")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = Config {
        seed: SEED,
        ..Config::default()
    };
    if let Some(n) = args.first().and_then(|a| a.parse::<usize>().ok()) {
        cfg.steps = n;
        cfg.burn = n / 2;
    }

    let p_dim = cfg.shapes.flat_size();
    let radius = cfg.sigma * (p_dim as f64).sqrt();
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

    let acc = |t: &Tensor| {
        data_b
            .accuracy(t, &cfg.shapes, Variant::Cosine)
            .expect("acc")
    };
    let nll = |t: &Tensor| data_b.nll(t, &cfg.shapes, Variant::Cosine).expect("nll");
    let adam_dir = unit(&theta_adam);
    let on_shell = adam_dir.affine(radius, 0.0).expect("shell");
    let haar_dir = unit(&init_flat(&cfg.shapes, 1.0, cfg.seed.wrapping_add(99)));

    println!(
        "sphere_chain — exact spherical pCN (uniform prior on the direction)\n\
         P = {p_dim}, ρ = σ√P = {radius:.2}, target B, uniform NLL ln {} = {ln_k:.4}\n\
         {steps} steps, burn {burn}, thin 20, 1 chain, λ = 1, seed {SEED}\n",
        cfg.k,
        ln_k = (cfg.k as f64).ln(),
        steps = cfg.steps,
        burn = cfg.burn
    );
    println!(
        "where the solution sits, and what the sphere can see of it\n  \
         θ_Adam (B)            acc {:>6.2}%  NLL {:>8.4}  ‖θ‖ {:>8.1}\n  \
         ρ·u_Adam              acc {:>6.2}%  NLL {:>8.4}  (same direction, shell radius)\n  \
         ρ·u_Haar              acc {:>6.2}%  NLL {:>8.4}\n",
        acc(&theta_adam),
        nll(&theta_adam),
        norm(&theta_adam).expect("norm"),
        acc(&on_shell),
        nll(&on_shell),
        acc(&haar_dir.affine(radius, 0.0).expect("shell")),
        nll(&haar_dir.affine(radius, 0.0).expect("shell")),
    );

    println!(
        "{:>6}  {:>6}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "β₀", "start", "accept", "acc t=0", "acc t=10", "acc t=100", "BMA acc", "BMA NLL", "K"
    );
    for beta_0 in BETAS {
        for (label, u0) in [("adam", adam_dir.clone()), ("haar", haar_dir.clone())] {
            let sched = SgPcnlScheduler {
                beta_0,
                gamma: 0.0,
                kappa: 0.5,
            };
            let mut chain = SpherePcn::new(u0, radius, cfg.sigma, sched, cfg.n_train, 1.0);
            let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed ^ (beta_0.to_bits()) ^ 0x5ced);
            let mut kept: Vec<Tensor> = Vec::new();
            let (mut a0, mut a10, mut a100) = (0.0, 0.0, 0.0);
            for t in 0..cfg.steps {
                let theta = chain.theta().expect("theta");
                match t {
                    0 => a0 = acc(&theta),
                    10 => a10 = acc(&theta),
                    100 => a100 = acc(&theta),
                    _ => {}
                }
                chain
                    .step(
                        |th| data_b.loss_and_grad(th, &cfg.shapes, Variant::Cosine),
                        &mut rng,
                    )
                    .expect("sphere step");
                if t >= cfg.burn && (t - cfg.burn).is_multiple_of(20) {
                    kept.push(chain.theta().expect("theta"));
                }
            }
            let chains = vec![kept];
            let (bma_acc, bma_nll, _ece, k) =
                bma_metrics(&chains, &cfg.shapes, Variant::Cosine, &data_b);
            println!(
                "{beta_0:>6}  {label:>6}  {:>7.1}%  {a0:>8.2}%  {a10:>8.2}%  {a100:>8.2}%  \
                 {bma_acc:>8.2}%  {bma_nll:>9.4}  {k:>7}",
                100.0 * chain.accepted as f64 / cfg.steps as f64
            );
        }
    }
    println!(
        "\nreading: acceptance decides whether a row is a sample or a frozen start.\n\
         Where the filter accepts, the sphere target — unlike the Gaussian one — has\n\
         mass at the solution's direction, so retention there is a property of the\n\
         target; a Haar start on the same kernel is the search question, and §6.4(ii)'s\n\
         volume bound says it stays at chance."
    );
}
