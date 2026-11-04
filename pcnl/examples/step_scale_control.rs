//! `step_scale_control` — is the standard-softmax baseline's explosion a
//! property of its **geometry**, or of the **step size** the pCN
//! coefficient prescribes for it?
//!
//! §6.4 of `test/unfer/ngpt.md` reports that, under the same unadjusted
//! kernel, the cosine (nGPT-lite) chain holds its shell while the
//! standard stack's gradient reaches `10¹⁷–10²²` and the chain dies at
//! step 4–5.  A reviewer's objection is that this compares geometries
//! *and* step scales at once: the drift is `h_t·g_t` with
//! `h_t = β_t²σ²/2`, and the cosine variant's gradient is bounded by its
//! own normalization while the standard stack's is not, so the standard
//! arm is being run at an enormous effective learning rate.  Fair enough
//! — so this example measures which part of the claim is load-bearing.
//!
//! **The control.**  Run the cosine chain as published, record the
//! largest *scaled* full-potential gradient `C = max_t (N/B)‖Σ_B ∇ℓ‖`
//! it ever produced, then hand that same `C` to the standard stack as a
//! drift cap (`SgPcnlState::grad_norm_cap`, a control and not part of the
//! kernel): its drift can then move θ by at most `h_t·C` per step, the
//! same budget the normalized geometry produces on its own.  Both
//! variants start from their own Adam-pre-trained `θ_A`, and the
//! uncapped standard arm is run beside them as the reproduction.
//!
//! **What the answer means either way.**
//! * If the capped standard chain survives, the explosion was a
//!   *step-scale* claim: the standard stack has no intrinsic scale for
//!   `β` to be measured in, so the nGPT result is "the geometry supplies
//!   the step size, so the sampler needs no tuning", not "an
//!   unnormalized transformer cannot be sampled".
//! * If it still explodes, the claim stands in its strong form: the
//!   pathology is in the geometry the drift sees, not in the budget it
//!   is given.
//!
//! The example reports the standard stack's *uncapped* divergence rate
//! too (`max scaled ‖g‖ / C`), which is the factor by which the two
//! arms' step sizes differ — the size of the confound the control
//! removes.
//!
//! ```text
//! cargo run --release -p pcnl --example step_scale_control          # reduced P = 552
//! cargo run --release -p pcnl --example step_scale_control -- --reference
//! ```

use candle_core::Tensor;
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState, norm};
use pcnl::task::{BatchLoader, Dataset, Task, adam};
use rand::SeedableRng;

const SEEDS: [u64; 3] = [20261020, 20261021, 20261022];

struct Cfg {
    shapes: Shapes,
    k: usize,
    target: Task,
    n_train: usize,
    n_test: usize,
    pretrain_steps: usize,
    pretrain_lr: f64,
    steps: usize,
    burn: usize,
    thin: usize,
    chains: usize,
    batch: usize,
    sigma: f64,
    beta_0: f64,
    gamma: f64,
    kappa: f64,
    init_std: f64,
}

impl Cfg {
    /// §6.5.1's reduced configuration: `P = 552`, the shift target.
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
            pretrain_lr: 0.05,
            steps: 2000,
            burn: 1000,
            thin: 8,
            chains: 2,
            batch: 64,
            sigma: 0.35,
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.5,
            init_std: 0.1,
        }
    }

    /// §6.3/§6.4's reference configuration: `P = 16 672`, target B.
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
            steps: 1000,
            burn: 500,
            thin: 5,
            sigma: 0.35,
            ..Self::reduced()
        }
    }

    fn steps_of(&self, variant: Variant, seed: u64) -> Tensor {
        let data_a = Dataset::new(Task::A, self.k, self.n_train, self.n_test, seed);
        let (theta, _) = adam(
            &data_a,
            &self.shapes,
            variant,
            init_flat(&self.shapes, self.init_std, seed),
            self.pretrain_steps,
            self.pretrain_lr,
        );
        theta
    }
}

/// One arm's measurement: survival, the gradient scale it saw, the radius
/// it occupied, and the predictive of the draws it kept.
#[derive(Default, Clone)]
struct Arm {
    exploded_at: Option<usize>,
    first_scaled_grad: f64,
    max_scaled_grad: f64,
    radial_mean: f64,
    radial_max: f64,
    radial_final: f64,
    kept: usize,
    kept_acc: f64,
    kept_nll: f64,
    final_acc: f64,
    final_nll: f64,
}

fn finite(theta: &Tensor) -> bool {
    theta
        .to_vec1::<f32>()
        .map(|v| v.iter().all(|x| x.is_finite()))
        .unwrap_or(false)
}

/// The unadjusted chain, with an optional drift cap.  Nothing here is a
/// second implementation of the kernel: every update goes through
/// `SgPcnlState::step_unadjusted`.
fn chain(
    cfg: &Cfg,
    ds: &Dataset,
    variant: Variant,
    theta_a: &Tensor,
    cap: Option<f64>,
    seed: u64,
) -> Arm {
    let sched = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    let n_train = ds.train_x.dim(0).expect("train rows");
    let mut arms = Vec::with_capacity(cfg.chains);

    for c in 0..cfg.chains {
        let mut state = SgPcnlState::new(theta_a.clone(), cfg.sigma, sched, n_train, cfg.batch, 0);
        state.grad_norm_cap = cap;
        let mut loader = BatchLoader::new(n_train, cfg.batch, seed ^ ((c as u64) << 8) ^ 0x5eed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ ((c as u64) << 8));

        let (mut n_seen, mut rad_sum, mut rad_max, mut rad_final) = (0usize, 0.0f64, 0.0f64, 0.0);
        let (mut first_g, mut max_g) = (f64::NAN, 0.0f64);
        let (mut kept, mut acc_sum, mut nll_sum) = (0usize, 0.0f64, 0.0f64);
        let mut exploded_at = None;

        for t in 0..cfg.steps {
            let (x, y) = loader.next(ds);
            let (_loss, sum_grad) = ds
                .batch_grad(&state.theta, &cfg.shapes, variant, &x, &y)
                .expect("batch grad");
            // the quantity the drift multiplies: (N/B)·Σ_B ∇ℓ
            let scaled = norm(&sum_grad).map(|n| n * (n_train as f64) / (cfg.batch as f64));
            if let Ok(g) = scaled {
                if first_g.is_nan() {
                    first_g = g;
                }
                max_g = max_g.max(g);
            }

            state
                .step_unadjusted(&sum_grad, &mut rng)
                .expect("unadjusted step");

            if !finite(&state.theta) {
                exploded_at = Some(t + 1);
                break;
            }
            let r = norm(&state.theta).unwrap_or(f64::NAN);
            if r.is_finite() {
                rad_sum += r;
                n_seen += 1;
                rad_max = rad_max.max(r);
                rad_final = r;
            }
            if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
                let acc = ds
                    .accuracy(&state.theta, &cfg.shapes, variant)
                    .expect("accuracy");
                let nll = ds.nll(&state.theta, &cfg.shapes, variant).expect("nll");
                if acc.is_finite() && nll.is_finite() {
                    kept += 1;
                    acc_sum += acc;
                    nll_sum += nll;
                }
            }
        }

        let (final_acc, final_nll) = if exploded_at.is_none() {
            let acc = ds
                .accuracy(&state.theta, &cfg.shapes, variant)
                .unwrap_or(f64::NAN);
            let nll = ds
                .nll(&state.theta, &cfg.shapes, variant)
                .unwrap_or(f64::NAN);
            (acc, nll)
        } else {
            (f64::NAN, f64::NAN)
        };

        arms.push(Arm {
            exploded_at,
            first_scaled_grad: first_g,
            max_scaled_grad: max_g,
            radial_mean: if n_seen > 0 {
                rad_sum / n_seen as f64
            } else {
                f64::NAN
            },
            radial_max: rad_max,
            radial_final: rad_final,
            kept,
            kept_acc: if kept > 0 {
                acc_sum / kept as f64
            } else {
                f64::NAN
            },
            kept_nll: if kept > 0 {
                nll_sum / kept as f64
            } else {
                f64::NAN
            },
            final_acc,
            final_nll,
        });
    }

    // one row per arm: the earliest explosion, the summed numbers averaged
    let first = arms[0].clone();
    let mean = |f: fn(&Arm) -> f64| {
        let vals: Vec<f64> = arms.iter().map(f).filter(|v| v.is_finite()).collect();
        if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    Arm {
        exploded_at: arms.iter().filter_map(|a| a.exploded_at).min(),
        first_scaled_grad: first.first_scaled_grad,
        max_scaled_grad: arms.iter().map(|a| a.max_scaled_grad).fold(0.0, f64::max),
        radial_mean: mean(|a| a.radial_mean),
        radial_max: arms.iter().map(|a| a.radial_max).fold(0.0, f64::max),
        radial_final: mean(|a| a.radial_final),
        kept: first.kept,
        kept_acc: mean(|a| a.kept_acc),
        kept_nll: mean(|a| a.kept_nll),
        final_acc: mean(|a| a.final_acc),
        final_nll: mean(|a| a.final_nll),
    }
}

fn main() {
    let reference = std::env::args().any(|a| a == "--reference");
    let cfg = if reference {
        Cfg::reference()
    } else {
        Cfg::reduced()
    };
    let p = cfg.shapes.flat_size();
    let shell = cfg.sigma * (p as f64).sqrt();
    let target = cfg.target.formula(cfg.k);

    println!(
        "step-scale control — P = {p} ({:?}), sigma*sqrt(P) = {shell:.2}, target {}",
        cfg.shapes, target
    );
    println!(
        "  kappa = {:.2}, beta_0 = {}, steps = {} (burn {}), Adam pre-train on A, {} chains, seeds {:?}\n",
        cfg.kappa, cfg.beta_0, cfg.steps, cfg.burn, cfg.chains, SEEDS
    );
    println!(
        "{:>8}  {:<14}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}  {:>7}  {:>7}",
        "seed",
        "arm",
        "g@step0",
        "max g",
        "radial mx",
        "radial fn",
        "kept acc",
        "kept nll",
        "died@"
    );

    let mut cap_sum = 0.0f64;
    let mut mismatch = Vec::new();
    for seed in SEEDS {
        let ds = Dataset::new(cfg.target, cfg.k, cfg.n_train, cfg.n_test, seed);
        let cos_a = cfg.steps_of(Variant::Cosine, seed);
        let std_a = cfg.steps_of(Variant::Standard, seed);

        // 1. the published arm: cosine, uncapped.  Its max scaled gradient
        //    is C, the drift budget the normalized geometry produces.
        let cos = chain(&cfg, &ds, Variant::Cosine, &cos_a, None, seed);
        let c = cos.max_scaled_grad;
        cap_sum += c;

        // 2. the standard stack at the kernel's own step size (no cap):
        //    the reproduction of the published explosion.
        let std_plain = chain(&cfg, &ds, Variant::Standard, &std_a, None, seed);

        // 3. the control: the same standard stack with its drift capped
        //    at C — the cosine arm's own gradient scale.
        let std_capped = chain(&cfg, &ds, Variant::Standard, &std_a, Some(c), seed);

        mismatch.push(std_plain.max_scaled_grad / c);

        for (name, a) in [
            ("cosine plain", &cos),
            ("std plain", &std_plain),
            ("std capped", &std_capped),
        ] {
            println!(
                "{:>8}  {:<14}  {:>10.3e}  {:>10.3e}  {:>10.2}  {:>9.2}  {:>9.2}  {:>7.3}  {:>7}",
                seed,
                name,
                a.first_scaled_grad,
                a.max_scaled_grad,
                a.radial_max,
                a.radial_final,
                a.kept_acc,
                a.kept_nll,
                a.exploded_at
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into())
            );
        }
    }

    let cap = cap_sum / SEEDS.len() as f64;
    let m = mismatch.iter().cloned().fold(f64::INFINITY, f64::min);
    let big = mismatch.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "\nmean drift budget from the normalized geometry: C = {cap:.3e}\n\
         the uncapped standard stack reaches {m:.1e}–{big:.1e} times C:\n\
         that factor is the step-size confound this control removes."
    );
    println!(
        "with the cap, the standard chain's per-step drift is at most\n\
         h_0*C = {:.3} in theta units (shell {shell:.2}); if it survives, the\n\
         published explosion is a statement about the scale the kernel\n\
         prescribes, not about the forward map being unsamplable.\n\
         The kept-draw columns are the *average* accuracy/NLL of kept draws,\n\
         not a BMA; they answer 'is it still adapting', not 'what is the\n\
         posterior'.",
        0.5 * cfg.beta_0 * cfg.beta_0 * cfg.sigma * cfg.sigma * cap
    );
}
