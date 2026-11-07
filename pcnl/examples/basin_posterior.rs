//! `basin_posterior` — what the tempered sphere target's *own* predictive is,
//! whether a proposal can stay in it, and whether the outward climb is a random
//! walk or a drift.
//!
//! **The question.**  `sphere_chain --precond --beta 0.003` accepts 86 % and
//! scores 49.02 % at 400 steps but **46.39 %** at 10 000.  Is 46.39 % the
//! target's predictive, or a chain that has left the region where the target
//! has mass?  The acceptance column cannot separate them.
//!
//! **The measurement without a sampler.**  Isotropic tangent Gaussians of
//! radius `r` around the anchor (`u_Adam`), weighted by `exp(−N·(L−L_min))`:
//! a self-normalized importance-sampling estimate of π at every scale at once.
//! The width with `N·E[ΔL] = 1` is the one-nat radius `r₁`; it is *not* the
//! width with the largest ESS, because a ball so small that π is constant
//! inside it has flat weights by construction (that trap made an earlier
//! version of this example report the mode and call it the posterior).
//!
//! **Two exact proposals.**  `--arm density` is a fixed tangent Gaussian — an
//! independence sampler.  It is exact in principle and fails anyway in 16 672
//! dimensions: two typical draws differ by `−½(χ²_P − χ²_P) = O(√P)` in log
//! density (±1.6 × 10² nats), so it accepts 0.07 %.  `--arm shell` jumps a
//! *fixed* geodesic distance, `u′ = exp_u(r₁·d̂)` with `d̂` uniform in `T_u`;
//! that law is the uniform measure on the geodesic sphere of radius `r₁`,
//! hence symmetric, so `log α = −N·(L′−L)` and nothing else.  The kernel lives
//! in `pcnl::sphere::shell_jump`, where three closed-form moment tests certify
//! it (von Mises on the circle, vMF on `S²`, and an isotropic Gaussian in the
//! tangent).
//!
//! **The climb is a random walk, and that is measured, not argued.**  The shell
//! arm accepts ~78 % and still moves ~`r₁√t` away from the anchor, while the
//! potential is 43 nats up at 0.04 rad.  Two explanations: a **random walk**
//! (the radial coordinate relaxes with time constant `≈ P/(K r₁²)`, which is
//! `10³–10⁵` steps here, so 10⁴ steps is at or inside the relaxation time), or
//! a **drift** (an acceptance that favours outward moves — a bug).  `--chains N`
//! separates them: a random walk grows as `t^{1/2}` with independent directions
//! across chains, a drift grows as `t¹` with a common direction.  Four
//! independent chains of 10 000 steps read a growth exponent of
//! **0.504–0.513** (mean 0.509) and a mean pairwise angle between their net
//! displacement vectors of **89.7°** — independent, one common direction would
//! be 0°.  So the outward creep is diffusion under a very flat potential, not a
//! biased kernel; what is wrong is only the budget, and `split-R̂` says so too
//! (NLL 1.50, train loss 1.85, i.e. no chain is sampling a common law yet).
//!
//! **`--prior-is M` answers the question the sweeps cannot: what is π's own
//! predictive?**  The sweep samples *inside* a ball and normalises within it, so
//! its number is `E[acc | d ≤ r]`, not `E[acc]`.  `--prior-is` draws Haar
//! directions instead — the prior's own law, since `π ∝ exp(−N·L)` against Haar
//! — and prices the only two things that can move mass: cap volume
//! (`(sin(d/2))^{P−1}`, i.e. `10^{52 626}` decades from 1 mrad to 1.6 rad) and
//! `N·ΔL`.  At `P = 16 672` the likelihood loses by `10^{54 000}`: every Haar
//! draw lands at 1.0–1.6 rad scoring the base rate (11.72 %, NLL 5.04), so **π
//! is prior-dominated and 48.8 % is a cone-conditioned quantity.**  The 50 000-
//! step shell run is the confirmation from the other side: `d` tracks the free
//! walk to 90 % with growth exponent 0.521, so the chain is heading to the
//! shell where the mass is, not relaxing inside the cone.
//!
//! **`--cap [R]` makes the question well posed — and freezes the chain.**  If `π`
//! is prior-dominated then "what is the posterior mean?" has the base rate as its
//! answer and every 46–51 % number is a conditional; restricting the *target* to
//! `d(u_Adam, u) ≤ R` (default `r₁`) makes that conditional the law the chain has,
//! and it costs the kernel nothing (`π = 0` outside turns outward moves into
//! rejections, so `shell_jump`'s symmetric ratio stays exact;
//! `pcnl::sphere::restrict_to_cap`, with a Haar-on-the-cap moment test).
//!
//! It does not work in `P = 16 672`, and the reason is worth keeping: *measured*
//! 0.0 % acceptance at `R = r₁` and 0.1 % at `R = 2.3e-3`, both with `R̂ = ∞`.
//! A step `r` from radius `R` stays inside only when `⟨u,ŵ⟩ ≥ r/(2R)`, whose
//! probability is `Φ̄(r√P/(2R))` — `e^{−P/8}` at `R = r` — and `d²` gains `r²` per
//! step, so the walk reaches the boundary in `(R/r)² ≈ 7` steps and never leaves.
//! The pinned run still prices the conditional (48.44 % / 1.3132, against the
//! sweep's 48.95 % / 1.3132) but it is four points, not a sample.
//!
//! **`--arm latitude --d0 R` is the kernel that lives on the boundary.**
//! `sphere::latitude_jump` rotates about the `u_Adam` axis, so `⟨u_Adam,u⟩` is
//! invariant to machine precision: the cap never rejects and the target *is*
//! `π(· | d = R)`.  `--step` sets the shell step (relaxation `(sin R/r)²` steps).
//! Four radii, two chains, 10 000 steps: BMA 47.66 / 46.48 / 48.44 / 39.84 % at
//! `R = 8.61e-4, 2.3e-3, 1e-2, 7.9e-2`, every `R̂(NLL) ≤ 1.04` — the first
//! *converged* posterior numbers in the study, and the first agreement between an
//! exact chain and the ball-IS sweep (48.8–49.0 %) on the same functional.  At
//! `1e-2` rad the conditional's NLL (1.3045) is better than the anchor's (1.3136).
//!
//! **The radial law, and the data it would take.**  With Haar on the sphere,
//! geodesic polars give `p(d) ∝ exp(−N·L(d))·(sin d)^{P−2}`, so the mode solves
//! `N·L′(d)·d = P − 2`.  Fitting this example's own sweep column
//! (`N·L ≈ 2200·d^{1.05}`) puts that mode at `d* = 6.5 rad`: **there is no interior
//! mode**, the density rises to the equatorial shell, and `π`'s predictive is the
//! base rate — the sharp form of prior-dominated, and why the 10 k/50 k walks
//! diffuse as `√t` with no turn-around.  The same equation prices the fix:
//! `N·L′(2.3e-3) ≥ (P−2)/2.3e-3 = 7.25e6` against the measured `1.1e3`, i.e.
//! `N ≈ 3.4e6` examples (6.6e3× the 512 used).
//!
//! **`--arm guided`** replaces the uniform direction with a von Mises–Fisher
//! draw about the ascent direction (`PCNL_KAPPA_GUIDE`, exact for every `κ`).
//! Its drift budget is `r·A`, `A ≈ κ/(P−2)`, so drift and the free walk cross at
//! `t ≈ (P/κ)²` and acceptance falls as `κ` rises — `κ = 0.3` is unmeasurable at
//! 10⁴ steps, `κ = 100` is 44.14 % at 10⁴ (vs 38.28 % isotropic) with `R̂` 1.79,
//! `κ = 240` freezes near the anchor.  `PCNL_LIVE_GRAD=1` recomputes the field
//! at the chain's point: acceptance *up* (67 %), drift *down*, because the part
//! of the proposal the reverse term does not cancel is exactly the drift.
//!
//! ```text
//! cargo run --release -p pcnl --example basin_posterior               # sweep + 1 chain
//! cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 4
//! cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --arm density
//! cargo run --release -p pcnl --example basin_posterior -- --prior-is 2048
//! PCNL_KAPPA_GUIDE=100 cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --arm guided --chains 4
//! cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 4 --cap
//! cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 2 --arm latitude --d0 2.3e-3 --step 1.15e-3
//! ```

use candle_core::{Device, Tensor};
use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Variant, init_flat};
use pcnl::sampler::norm;
use pcnl::task::{Dataset, Task, adam, momentum_sgd};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

const SEED: u64 = 20261020;
/// The per-step angular size of the metric chain at `β = 0.003`: on the unit
/// sphere the noise is `e_i = τ√d_i z_i` with `τ = β/√P` and `mean(d) = 1`, so
/// `‖e‖ ≈ τ√P = β` — **β is the step in radians** (the `2.3 × 10⁻⁵` quoted in
/// `PCNL_STATUS` is `τ`, the *per-coordinate* scale).
const BETA: f64 = 0.003;
const STEPS_10K: f64 = 10_000.0;
/// The radius the sweep finds the mass decaying at; used for "in basin %".
const BASIN: f64 = 2.3e-3;
/// The one-nat radius measured by the 128-probe sweep of 2026-11-06, for
/// `--no-sweep`.  Regenerate it with the sweep; it is not a fit.
const R1_MEASURED: f64 = 8.61e-4;
const WIDTHS: [f64; 6] = [1e-4, 3e-4, 1e-3, 2.3e-3, 5e-3, 1e-2];

fn unit(v: &Tensor) -> Tensor {
    let n = norm(v).expect("norm");
    v.affine(1.0 / n, 0.0).expect("unit")
}

fn to_vec(t: &Tensor) -> Vec<f64> {
    t.flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec")
        .iter()
        .map(|x| *x as f64)
        .collect()
}

/// `log_{u}(x)` for unit vectors: the tangent at `u` whose exponential map
/// reaches `x`, with `‖v‖` the geodesic radius.
fn log_map(u: &[f64], x: &[f64]) -> Vec<f64> {
    let c: f64 = u
        .iter()
        .zip(x)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let ang = c.acos();
    if ang < 1e-12 {
        return vec![0.0; u.len()];
    }
    let s = ang.sin();
    (0..u.len()).map(|i| (x[i] - c * u[i]) * ang / s).collect()
}

fn norm_of(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Everything a chain needs: the data, the potential, and the anchor.
struct Env {
    cfg: Config,
    data_b: Dataset,
    p_dim: usize,
    n: f64,
    radius: f64,
    u0: Vec<f64>,
    l0: f64,
    /// Geodesic radius of the target's cap about `u_Adam`; `inf` = unrestricted.
    cap: f64,
    /// Latitude held fixed by `--arm latitude` (the radius of the conditional).
    d0: f64,
}

impl Env {
    fn new(seed: u64) -> Self {
        let cfg = Config {
            seed,
            ..Config::default()
        };
        let p_dim = cfg.shapes.flat_size();
        let radius = cfg.sigma * (p_dim as f64).sqrt();
        let n = cfg.n_train as f64;
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
        let u0 = to_vec(&unit(&theta_adam));
        let mut env = Env {
            cfg,
            data_b,
            p_dim,
            n,
            radius,
            u0,
            l0: 0.0,
            cap: f64::INFINITY,
            d0: 0.0,
        };
        env.l0 = env.pot(&env.u0.clone());
        env
    }

    fn at(&self, u: &[f64]) -> Tensor {
        Tensor::from_vec(
            u.iter()
                .map(|x| (*x * self.radius) as f32)
                .collect::<Vec<f32>>(),
            vec![self.p_dim],
            &Device::Cpu,
        )
        .expect("lift")
    }

    /// Train mean cross-entropy — the potential `π ∝ exp(−N·L)`.
    ///
    /// Forward only (`Dataset::train_loss`): the shell kernel evaluates
    /// `log π` and never the gradient, and an earlier version called
    /// `loss_and_grad` and kept the loss, which made every step pay for a
    /// reverse sweep it discarded — most of the 10 000-step cost, and the
    /// reason unrelated probes seemed to hang.
    fn pot(&self, u: &[f64]) -> f64 {
        self.data_b
            .train_loss(&self.at(u), &self.cfg.shapes, Variant::Cosine)
            .expect("potential")
    }

    fn acc(&self, u: &[f64]) -> f64 {
        self.data_b
            .accuracy(&self.at(u), &self.cfg.shapes, Variant::Cosine)
            .expect("acc")
    }

    fn nll(&self, u: &[f64]) -> f64 {
        self.data_b
            .nll(&self.at(u), &self.cfg.shapes, Variant::Cosine)
            .expect("nll")
    }
}

/// One chain's kept series, all aligned with `t`.
struct Chain {
    t: Vec<f64>,
    nll: Vec<f64>,
    acc: Vec<f64>,
    loss: Vec<f64>,
    disp: Vec<f64>,
    theta: Vec<Tensor>,
    accepted: usize,
    steps: usize,
    final_u: Vec<f64>,
}

fn run_chain(
    env: &Env,
    arm: &str,
    r1: f64,
    steps: usize,
    burn: usize,
    thin: usize,
    seed: u64,
) -> Chain {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut cur;
    let mut cur_l;
    if arm == "shell" || arm == "guided" {
        cur = env.u0.clone();
        cur_l = env.l0;
    } else if arm == "latitude" {
        // start *on* the latitude, in a seeded direction
        let mut v: Vec<f64> = (0..env.p_dim)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        project(&mut v, &env.u0);
        let nv = norm_of(&v).max(1e-300);
        cur = (0..env.p_dim)
            .map(|i| env.d0.cos() * env.u0[i] + env.d0.sin() * v[i] / nv)
            .collect();
        cur_l = env.pot(&cur);
    } else {
        // start from a draw of the fixed law, not from its mode
        let sd = r1 / (env.p_dim as f64).sqrt();
        let mut v: Vec<f64> = (0..env.p_dim)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * sd)
            .collect();
        project(&mut v, &env.u0);
        let r = norm_of(&v);
        cur = (0..env.p_dim)
            .map(|i| r.cos() * env.u0[i] + if r > 1e-12 { r.sin() * v[i] / r } else { 0.0 })
            .collect();
        cur_l = env.pot(&cur);
    }
    let mut accepted = 0usize;
    let mut chain = Chain {
        t: Vec::new(),
        nll: Vec::new(),
        acc: Vec::new(),
        loss: Vec::new(),
        disp: Vec::new(),
        theta: Vec::new(),
        accepted: 0,
        steps,
        final_u: Vec::new(),
    };
    // A 10 000-step chain is minutes long; without a progress line the only
    // two observable states are "running" and "done", which cannot tell a slow
    // box from a stuck loop.  `PCNL_PROGRESS=1` prints every 10 % of the run.
    let progress = std::env::var("PCNL_PROGRESS").is_ok();
    let mark = (steps / 10).max(1);
    let report = |t: usize, u: &[f64], l: f64, accepted: usize| {
        if progress && t.is_multiple_of(mark) {
            eprintln!(
                "   chain {seed}: t {t:>6}  acc {:>5.1}%  disp {:.3e} rad  train L {l:.4}",
                100.0 * accepted as f64 / (t + 1) as f64,
                norm_of(&log_map(&env.u0, u)),
            );
        }
    };
    if arm == "shell" {
        let log_target = |u: &[f64]| -env.n * env.pot(u);
        let log_target = pcnl::sphere::restrict_to_cap(log_target, &env.u0, env.cap);
        let log_target = &log_target;
        for t in 0..steps {
            let (prop, take, prop_l) =
                pcnl::sphere::shell_jump(&cur, -env.n * cur_l, r1, &log_target, &mut rng);
            if take {
                cur = prop;
                cur_l = -prop_l / env.n;
                accepted += 1;
            }
            record(env, &mut chain, &cur, cur_l, t, burn, thin);
            report(t, &cur, cur_l, accepted);
        }
    } else if arm == "guided" {
        // The descent direction, as an ambient vector projected at each proposal
        // by `drift_direction`.  Two fields are available: the anchor's gradient
        // frozen once (`−N·∇L(u₀)`), or the gradient at the chain's current point
        // (`PCNL_LIVE_GRAD=1`, one extra backward per step).  The frozen field's
        // alignment with the inward geodesic decays as the chain leaves the
        // anchor, which is exactly when the drift is needed, so the live field is
        // the one that can keep `A` at its budgeted value.
        let (_, grad) = env
            .data_b
            .loss_and_grad(&env.at(&env.u0), &env.cfg.shapes, Variant::Cosine)
            .expect("gradient at the anchor");
        let g: Vec<f64> = to_vec(&grad).iter().map(|x| -env.n * x).collect();
        let kappa = std::env::var("PCNL_KAPPA_GUIDE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.3);
        let live = std::env::var("PCNL_LIVE_GRAD").is_ok_and(|v| v == "1");
        let log_target = |u: &[f64]| -env.n * env.pot(u);
        let log_target = pcnl::sphere::restrict_to_cap(log_target, &env.u0, env.cap);
        let log_target = &log_target;
        let direction = |x: &[f64]| {
            if live {
                let (_, g) = env
                    .data_b
                    .loss_and_grad(&env.at(x), &env.cfg.shapes, Variant::Cosine)
                    .expect("gradient at the chain's point");
                let g: Vec<f64> = to_vec(&g).iter().map(|z| -env.n * z).collect();
                pcnl::sphere::drift_direction(&g, x)
            } else {
                pcnl::sphere::drift_direction(&g, x)
            }
        };
        for t in 0..steps {
            let (prop, take, prop_l) = pcnl::sphere::guided_shell_jump(
                &cur,
                -env.n * cur_l,
                r1,
                kappa,
                &direction,
                &log_target,
                &mut rng,
            );
            if take {
                cur = prop;
                cur_l = -prop_l / env.n;
                accepted += 1;
            }
            record(env, &mut chain, &cur, cur_l, t, burn, thin);
            report(t, &cur, cur_l, accepted);
        }
    } else if arm == "latitude" {
        // The move keeps `⟨a,u⟩` fixed, so the target is the conditional
        // `π(· | d(u, u₀) = d₀)` and the cap can never reject a proposal.
        let log_target = |u: &[f64]| -env.n * env.pot(u);
        let log_target = pcnl::sphere::restrict_to_cap(log_target, &env.u0, env.cap);
        let log_target = &log_target;
        for t in 0..steps {
            let (prop, take, prop_l) = pcnl::sphere::latitude_jump(
                &cur,
                -env.n * cur_l,
                r1,
                &env.u0,
                &log_target,
                &mut rng,
            );
            if take {
                cur = prop;
                cur_l = -prop_l / env.n;
                accepted += 1;
            }
            record(env, &mut chain, &cur, cur_l, t, burn, thin);
            report(t, &cur, cur_l, accepted);
        }
    } else {
        let sd = r1 / (env.p_dim as f64).sqrt();
        let s2 = sd * sd;
        let log_phi = |v: &[f64]| -0.5 * v.iter().map(|x| x * x).sum::<f64>() / s2;
        let log_volume = |r: f64| {
            if r < 1e-9 {
                0.0
            } else {
                ((env.p_dim - 2) as f64) * (r.sin() / r).ln()
            }
        };
        let draw = |rng: &mut rand::rngs::StdRng| -> (Vec<f64>, Vec<f64>, f64) {
            let mut v: Vec<f64> = (0..env.p_dim)
                .map(|_| rng.sample::<f64, _>(StandardNormal) * sd)
                .collect();
            project(&mut v, &env.u0);
            let r = norm_of(&v);
            let u = (0..env.p_dim)
                .map(|i| r.cos() * env.u0[i] + if r > 1e-12 { r.sin() * v[i] / r } else { 0.0 })
                .collect();
            (u, v, r)
        };
        let mut cur_v = log_map(&env.u0, &cur);
        let mut cur_r = norm_of(&cur_v);
        // the chain was started from a draw; recover its tangent vector
        cur_v = log_map(&env.u0, &cur);
        let _ = &mut cur_v;
        for t in 0..steps {
            let (prop, v, r) = draw(&mut rng);
            let prop_l = env.pot(&prop);
            let log_alpha = -env.n * (prop_l - cur_l)
                + (log_phi(&v) - log_phi(&cur_v))
                + (log_volume(cur_r) - log_volume(r));
            let take =
                log_alpha.is_finite() && (log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha);
            if take {
                cur = prop;
                cur_l = prop_l;
                cur_v = v;
                cur_r = r;
                accepted += 1;
            }
            record(env, &mut chain, &cur, cur_l, t, burn, thin);
            report(t, &cur, cur_l, accepted);
        }
    }
    chain.accepted = accepted;
    chain.final_u = cur;
    chain
}

fn record(env: &Env, chain: &mut Chain, u: &[f64], l: f64, t: usize, burn: usize, thin: usize) {
    if t >= burn && (t - burn).is_multiple_of(thin) {
        chain.t.push(t as f64 + 1.0);
        chain.nll.push(env.nll(u));
        chain.acc.push(env.acc(u));
        chain.loss.push(l);
        chain.disp.push(norm_of(&log_map(&env.u0, u)));
        if (t - burn).is_multiple_of(2 * thin) {
            chain.theta.push(env.at(u));
        }
    }
}

fn project(v: &mut [f64], u: &[f64]) {
    let c: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
    for i in 0..v.len() {
        v[i] -= c * u[i];
    }
}

/// Split-`R̂`: each chain is split in half, so `2N` sequences of length `n/2`
/// give the usual between/within decomposition.  Splitting is what makes it a
/// convergence diagnostic for a single chain; with `N` chains it also catches
/// chains that sit at different places.
fn split_rhat(series: &[Vec<f64>]) -> f64 {
    let mut seqs: Vec<&[f64]> = Vec::new();
    for c in series {
        let n = c.len() / 2;
        if n < 2 {
            return f64::NAN;
        }
        seqs.push(&c[..n]);
        seqs.push(&c[n..2 * n]);
    }
    let m = seqs.len() as f64;
    let n = seqs[0].len() as f64;
    let means: Vec<f64> = seqs.iter().map(|s| s.iter().sum::<f64>() / n).collect();
    let w = seqs
        .iter()
        .zip(&means)
        .map(|(s, mu)| s.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / m;
    let gm = means.iter().sum::<f64>() / m;
    let b = n * means.iter().map(|mu| (mu - gm).powi(2)).sum::<f64>() / (m - 1.0);
    let var = (n - 1.0) / n * w + b / n;
    (var / w).sqrt()
}

/// Least-squares slope of `ln d` on `ln t`: 0.5 is a random walk, 1.0 a
/// constant-speed drift.
fn growth_slope(t: &[f64], d: &[f64]) -> f64 {
    let n = t.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (ti, di) in t.iter().zip(d) {
        let (x, y) = (ti.ln(), di.ln());
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// **Is the tempered posterior concentrated at all?**  Measured from the
/// prior's own directional law, with no chain and no anchor ball.
///
/// `π(u) ∝ exp(−N·L(u))` against Haar measure, so the prior contributes no
/// directional preference whatever and the likelihood is the only thing that
/// can concentrate `π`.  The competing fact is geometric: a cap of angular
/// radius `d` holds a fraction `≈ (d/2)^{P−1}` of the sphere, so the number of
/// directions available between 1 mrad and 0.3 rad is larger by `10^41000` at
/// `P = 16 672`.  The likelihood can only buy that back with `N·ΔL`, and `ΔL`
/// is bounded by the softmax loss — which is what this table prices.
fn prior_is(env: &Env, m: usize) {
    let p = env.p_dim;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED ^ 0x9e11);
    // The shell edges: 0 → the whole cap, so `d_hi^{P−1}` is the volume proxy.
    let edges = [0.0f64, 1e-3, 1e-2, 0.1, 0.5, 1.0, 1.6, std::f64::consts::PI];
    let mut n = vec![0usize; edges.len() - 1];
    let mut sum_l = vec![0.0f64; edges.len() - 1];
    let mut sum_a = vec![0.0f64; edges.len() - 1];
    let mut sum_n = vec![0.0f64; edges.len() - 1];
    let mut all_l = Vec::with_capacity(m);
    let mut all_a = Vec::with_capacity(m);
    let mut all_nll = Vec::with_capacity(m);
    let mut all_d = Vec::with_capacity(m);
    for _ in 0..m {
        let raw: Vec<f64> = (0..p)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let nr = norm_of(&raw);
        let u: Vec<f64> = raw.iter().map(|x| x / nr).collect();
        let d = norm_of(&log_map(&env.u0, &u));
        let l = env.pot(&u);
        let a = env.acc(&u);
        let nl = env.nll(&u);
        let k = edges
            .windows(2)
            .position(|w| d >= w[0] && d < w[1])
            .unwrap_or(edges.len() - 2);
        n[k] += 1;
        sum_l[k] += l;
        sum_a[k] += a;
        sum_n[k] += nl;
        all_l.push(l);
        all_a.push(a);
        all_nll.push(nl);
        all_d.push(d);
    }
    let l_min = all_l.iter().cloned().fold(f64::INFINITY, f64::min);
    let l_max = all_l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "\n3. the prior's own predictive and where π's mass can be ({m} Haar directions)\n   \
         the prior is Haar, so `exp(−N·L)` is the whole posterior weight\n\n   \
         {:>13}  {:>6}  {:>8}  {:>9}  {:>11}  {:>12}  {:>7}  {:>8}",
        "d range (rad)", "n", "train L", "N·ΔL", "log10 volume", "log10 weight", "acc %", "NLL"
    );
    let pf = p as f64;
    // Cap measure up to a `d`-independent constant: `∫₀^d sin^{P−2} ≈
    // sin^{P−1}(d)/(P−1)` at large `P`, so `(P−1)·log10 sin(d/2)` is the whole
    // volume story.  (Computing `d^{P−1}` directly overflows: `1.6^16671` is `10^3400`.)
    let log_vol_of = |d: f64| (pf - 1.0) * (d.min(3.0) / 2.0).sin().max(1e-300).log10();
    let vol_ref = log_vol_of(edges[1]);
    let mut best = (f64::NEG_INFINITY, 0usize);
    for k in 0..n.len() {
        if n[k] == 0 {
            continue;
        }
        let (lo, hi) = (edges[k], edges[k + 1]);
        let frac = n[k] as f64 / m as f64;
        let mean_l = sum_l[k] / n[k] as f64;
        let d_ndl = env.n * (mean_l - env.l0);
        let log_vol = log_vol_of(hi) - vol_ref;
        let log_w = log_vol - d_ndl / 10f64.ln();
        if log_w > best.0 {
            best = (log_w, k);
        }
        println!(
            "   {:>13}  {:>6}  {:>8.4}  {:>9.1}  {:>13.0}  {:>13.0}  {:>6.2}  {:>8.4}   (sampled {:>5.1} %)",
            format!("{lo:.1e}–{hi:.2}"),
            n[k],
            mean_l,
            d_ndl,
            log_vol,
            log_w,
            sum_a[k] / n[k] as f64,
            sum_n[k] / n[k] as f64,
            100.0 * frac
        );
    }
    // Self-normalized importance sampling, from the prior itself.
    let w: Vec<f64> = all_l.iter().map(|l| (-env.n * (l - l_min)).exp()).collect();
    let sw: f64 = w.iter().sum();
    let sw2: f64 = w.iter().map(|x| x * x).sum();
    let ess = sw * sw / sw2;
    let acc_w: f64 = w.iter().zip(&all_a).map(|(x, a)| x * a).sum::<f64>() / sw;
    let nll_w: f64 = w.iter().zip(&all_nll).map(|(x, a)| x * a).sum::<f64>() / sw;
    let d_w: f64 = w.iter().zip(&all_d).map(|(x, a)| x * a).sum::<f64>() / sw;
    let l_w: f64 = w.iter().zip(&all_l).map(|(x, a)| x * a).sum::<f64>() / sw;
    let spread = env.n * (l_max - l_min) / 10f64.ln();
    // The two sides of the trade, in the same units, for the shell the sampled
    // mass actually sits in.
    let (lo_b, hi_b) = (edges[best.1], edges[best.1 + 1]);
    let vol_shell = log_vol_of(hi_b) - log_vol_of(edges[1]);
    let like_shell = env.n * (sum_l[best.1] / n[best.1] as f64 - env.l0) / 10f64.ln();
    let shell_label = format!("{lo_b:.1e}–{hi_b:.2} rad");
    println!(
        "\n   prior predictive (unweighted): acc {:.2} %, NLL {:.4}   ← what a draw from π ≈ Haar scores\n   \
         self-normalized IS from Haar:   acc {:.2} %, NLL {:.4},  ESS {:.2} of {m}\n   \
         IS-weighted mean d {:.3} rad, train L {:.4} (N·ΔL = {:.0})\n   \
         likelihood spread over the sample: N·(max L − min L) = {:.0} nats = 10^{:.0}\n\n   \
         the two sides of the trade, in decades of mass:\n   \
         volume from the 1 mrad cap to {}: 10^{:.0}\n   \
         likelihood against that shell:    10^{:.0}\n   \
         net mass ratio (shell : cone):    10^{:.0}\n\n   \
         verdict: {}.",
        mean(&all_a),
        mean(&all_nll),
        acc_w,
        nll_w,
        ess,
        d_w,
        l_w,
        env.n * (l_w - env.l0),
        spread,
        spread,
        shell_label,
        vol_shell,
        -like_shell,
        vol_shell - like_shell,
        if vol_shell > like_shell {
            "the shell wins, so π is prior-dominated: its predictive is the base \
             rate, and the cone's 48.8 % is a quantity conditioned on the cone"
        } else {
            "the cone wins, so π is likelihood-dominated and the cone is its bulk"
        }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let steps: usize = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000);
    let probes: usize = flag("--probes").and_then(|v| v.parse().ok()).unwrap_or(128);
    let chains: usize = flag("--chains").and_then(|v| v.parse().ok()).unwrap_or(1);
    let arm = flag("--arm").unwrap_or_else(|| "shell".into());
    let no_sweep = args.iter().any(|a| a == "--no-sweep");
    // `--cap [R]` restricts the *target* to a cap of radius `R` (default: the
    // measured r₁) about u_Adam.  Nothing in the kernel changes: the missing
    // mass outside the cap turns every outward move into a rejection.
    // `--step R` overrides the jump radius (default: the sweep's one-nat r₁).
    // The latitude arm needs it: its relaxation time along the shell is
    // `(sin d₀ / r)²` steps, so a shell at 0.08 rad cannot mix in 10 000 steps
    // at `r = r₁` and has to be walked with a longer step.
    let step_flag = flag("--step").and_then(|v| v.parse::<f64>().ok());
    let cap_flag = args.iter().any(|a| a == "--cap");
    let cap_radius = flag("--cap").and_then(|v| v.parse::<f64>().ok());

    let mut env = Env::new(SEED);
    if let Some(m) = flag("--prior-is").and_then(|v| v.parse::<usize>().ok()) {
        prior_is(&env, m);
        return;
    }
    let p_dim = env.p_dim;
    let p_f = p_dim as f64;
    let n = env.n;
    let l0 = env.l0;
    let walk = BETA * STEPS_10K.sqrt();
    println!(
        "basin_posterior — the tempered sphere target's own predictive\n\
         P = {p_dim}, ρ = σ√P = {:.2}, N = {n:.0}, λ = 1, task B\n\
         u_Adam: train mean loss {l0:.4}, test acc {:.2} %, test NLL {:.4}, \
         uniform NLL {:.4}\n\
         the β = {BETA:.1e} chain's 10 k walk spans β√steps = {walk:.3} rad\n",
        env.radius,
        env.acc(&env.u0.clone()),
        env.nll(&env.u0.clone()),
        (env.cfg.k as f64).ln(),
    );

    // ---- 1-2. the width sweep ----------------------------------------------
    let mut r1 = R1_MEASURED;
    if no_sweep {
        println!(
            "1-2. sweep skipped (--no-sweep): r₁ = {r1:.2e} rad from the \
             128-probe sweep of 2026-11-06\n"
        );
    } else {
        println!(
            "1. width sweep (self-normalized IS, {probes} probes each)\n   \
             {:>9}  {:>8}  {:>11}  {:>9}  {:>9}  {:>10}  {:>10}  {:>9}",
            "width r", "ESS", "angle mean", "acc %", "NLL", "in basin %", "N·medΔL", "N·maxΔL"
        );
        let mut max_ess = (f64::NEG_INFINITY, 0.0f64);
        // (width, N·medΔL) per row: the inputs to the posterior-scale line below.
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let mut one_nat: Option<f64> = None;
        let (mut pred_lo, mut pred_hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED ^ 0x15);
        for width in WIDTHS {
            let sd = width / p_f.sqrt();
            let mut pts: Vec<(Vec<f64>, f64, f64)> = Vec::with_capacity(probes);
            let mut lmin = f64::INFINITY;
            for _ in 0..probes {
                let mut v: Vec<f64> = (0..p_dim)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * sd)
                    .collect();
                project(&mut v, &env.u0);
                let r = norm_of(&v);
                let u: Vec<f64> = (0..p_dim)
                    .map(|i| r.cos() * env.u0[i] + if r > 1e-12 { r.sin() * v[i] / r } else { 0.0 })
                    .collect();
                let l = env.pot(&u);
                lmin = lmin.min(l);
                pts.push((u, l, r));
            }
            let ws: Vec<f64> = pts
                .iter()
                .map(|(_, l, _)| (-n * (l - lmin)).exp())
                .collect();
            let wsum: f64 = ws.iter().sum();
            let ess = wsum * wsum / ws.iter().map(|w| w * w).sum::<f64>();
            let wmean = |vals: Vec<f64>| -> f64 {
                vals.iter().zip(&ws).map(|(x, w)| x * w).sum::<f64>() / wsum
            };
            let w_acc = wmean(pts.iter().map(|(u, _, _)| env.acc(u)).collect());
            let w_nll = wmean(pts.iter().map(|(u, _, _)| env.nll(u)).collect());
            let w_angle = wmean(pts.iter().map(|(_, _, r)| *r).collect());
            let in_cap = wmean(
                pts.iter()
                    .map(|(_, _, r)| if *r <= BASIN { 1.0 } else { 0.0 })
                    .collect(),
            );
            let mut dls: Vec<f64> = pts.iter().map(|(_, l, _)| l - lmin).collect();
            dls.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_dl = dls[dls.len() / 2];
            let dl_max = *dls.last().unwrap();
            println!(
                "   {width:>9.1e}  {ess:>8.1}  {w_angle:>11.3e}  {w_acc:>9.2}  {w_nll:>9.4}  \
                 {:>10.1}  {:>9.2}  {:>9.2}",
                100.0 * in_cap,
                n * med_dl,
                n * dl_max
            );
            rows.push((width, n * med_dl));
            pred_lo = pred_lo.min(w_acc);
            pred_hi = pred_hi.max(w_acc);
            if ess > max_ess.0 {
                max_ess = (ess, width);
            }
            if one_nat.is_none() && n * med_dl >= 0.5 {
                one_nat = Some(width / (n * med_dl).sqrt());
            }
        }
        if let Some(r) = one_nat {
            r1 = r;
        }
        // **The posterior's own scale.**  Haar on the sphere in geodesic polar
        // coordinates is `(sin d)^{P−2} dd dΩ`, so the posterior in the radius is
        // `p(d) ∝ exp(−N·L(d))·(sin d)^{P−2}` and its mode solves
        // `N·L′(d)·d = P−2` (for `d ≪ 1`).  Fitting the sweep's own column as
        // `N·L = c·d^β` gives `d* = ((P−2)/(cβ))^{1/β}`, and the same equation
        // says how much data would put the mode inside the basin.  This is the
        // question every 46–51 % number in the docs is really conditional on.
        let fit: Vec<(f64, f64)> = rows.iter().copied().filter(|(_, y)| *y > 0.0).collect();
        if fit.len() >= 3 {
            let m = fit.len() as f64;
            let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
            for (w, y) in &fit {
                let (x, ly) = (w.ln(), y.ln());
                sx += x;
                sy += ly;
                sxx += x * x;
                sxy += x * ly;
            }
            let beta = (m * sxy - sx * sy) / (m * sxx - sx * sx);
            let c = ((sy - beta * sx) / m).exp();
            let d_star = ((p_f - 2.0) / (c * beta)).powf(1.0 / beta);
            println!(
                "   posterior scale: N·L ≈ {c:.1}·d^{beta:.2} against the Haar volume d^{:.0};\n   \
                 the mode of p(d) ∝ e^{{−N·L}}·d^{{P−2}} is d* = {d_star:.2e} rad",
                p_f - 2.0
            );
            if d_star > 1.5 {
                println!(
                    "   → beyond the equator, i.e. **no interior mode**: the volume wins and\n   \
                     π's predictive is the base rate, whatever the kernel does."
                );
            }
            // the gradient needed at the basin radius, and the data shortfall
            let need = (p_f - 2.0) / BASIN;
            let have = rows
                .iter()
                .find(|(w, _)| (*w - BASIN).abs() < 1e-9)
                .map(|(_, y)| y / BASIN);
            if let Some(have) = have {
                println!(
                    "   \n   to localize at the basin radius {BASIN:.1e}: N·L′ ≥ (P−2)/r = \
                     {need:.2e} nats/rad\n   \
                     against the measured {have:.2e} at N = {n:.0} — a factor \
                     {:.1e}, i.e. N ≈ {:.1e} examples.\n",
                    need / have,
                    n * need / have
                );
            }
        }
        println!(
            "\n2. the target's width and predictive\n   \
             one-nat radius N·E[ΔL] = 1: r₁ = {r1:.2e} rad\n   \
             predictive across the sweep: acc {pred_lo:.2}–{pred_hi:.2} %, NLL ≈ {:.4}\n   \
             (the max-ESS width {:.1e} is *not* the posterior's width — see the note\n   \
             at the top; it is the widest ball inside which π is still constant)\n",
            env.nll(&env.u0.clone()),
            max_ess.1
        );
    }

    // ---- 3. the chains ------------------------------------------------------
    if let Some(st) = step_flag {
        println!(
            "   jump radius overridden: r = {st:.3e} rad ({:.2} r₁)\n",
            st / r1
        );
        r1 = st;
    }
    if arm == "latitude" {
        env.d0 = flag("--d0").and_then(|v| v.parse().ok()).unwrap_or(r1);
        println!(
            "   arm latitude: the target is conditioned on d(u_Adam, u) = {:.3e} rad \
             ({:.2} r₁).\n   \
             The move keeps ⟨u_Adam, u⟩ fixed, so the radius is exact every step, \
             the cap\n   \
             never rejects, and the chain can move *on* the shell a free-angle walk \
             gets stuck on.\n",
            env.d0,
            env.d0 / r1
        );
    }
    if cap_flag {
        env.cap = cap_radius.unwrap_or(r1);
        println!(
            "   target restricted to cap: d(u_Adam, u) ≤ {:.3e} rad (kernel unchanged;\n   \
             proposals outside the cap are rejected by π = 0, so the ratio stays exact)\n",
            env.cap
        );
    }
    let burn = steps / 5;
    let thin = (steps / 400).max(1);
    let runs: Vec<Chain> = if chains <= 1 {
        vec![run_chain(&env, &arm, r1, steps, burn, thin, SEED ^ 0x5a11)]
    } else {
        // `Env::new` is minutes long — 1 500 heavy-ball steps on A and 1 000
        // Adam steps on B — and it depends only on `SEED`, so every chain must
        // share one environment.  Building one per chain (an earlier version
        // did) multiplied the example's dominant cost by `chains` before a
        // single proposal had been drawn, which made a 10 000-step run look
        // hung: minutes of silence with no output at all.
        let env = &env;
        let arm = &arm;
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..chains)
                .map(|c| {
                    let seed = if c == 0 {
                        SEED ^ 0x5a11
                    } else {
                        SEED ^ 0x5a1100 ^ (c as u64)
                    };
                    s.spawn(move || run_chain(env, arm, r1, steps, burn, thin, seed))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("chain thread"))
                .collect()
        })
    };

    let label = if arm == "shell" {
        format!("fixed-radius shell jump (r = {r1:.2e} rad, symmetric)")
    } else if arm == "guided" {
        let kappa = std::env::var("PCNL_KAPPA_GUIDE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.3);
        format!("fixed-radius guided jump (r = {r1:.2e} rad, κ = {kappa})")
    } else if arm == "latitude" {
        // `⟨u₀,u⟩` is fixed by construction, so the target *is* the shell
        // conditional and the cap cannot bite; the radius column below is
        // constant by design rather than frozen by rejection.
        format!("fixed-latitude rotation (r = {r1:.2e} rad, exact ratio)")
    } else {
        format!("fixed-density independence sampler (r = {r1:.2e} rad, exact ratio)")
    };
    let label = if arm == "latitude" {
        format!("{label}, target ∩ shell d = {:.2e} rad", env.d0)
    } else if env.cap.is_finite() {
        format!("{label}, target ∩ cap {:.2e} rad", env.cap)
    } else {
        label
    };
    println!(
        "3. {} chain(s): {label}\n   {steps} steps, burn {burn}, thin {thin}",
        runs.len()
    );
    println!(
        "   {:>5}  {:>8}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
        "chain", "accept", "d final", "acc %", "NLL", "train L", "K"
    );
    for (i, c) in runs.iter().enumerate() {
        let m = c.nll.len() as f64;
        println!(
            "   {i:>5}  {:>7.1}%  {:>10.3e}  {:>9.2}  {:>9.4}  {:>9.4}  {:>9}",
            100.0 * c.accepted as f64 / c.steps as f64,
            c.disp.last().copied().unwrap_or(f64::NAN),
            c.acc.iter().sum::<f64>() / m,
            c.nll.iter().sum::<f64>() / m,
            c.loss.last().copied().unwrap_or(f64::NAN),
            c.theta.len()
        );
    }

    let th: Vec<Vec<Tensor>> = runs.iter().map(|c| c.theta.clone()).collect();
    let (bma_acc, bma_nll, _ece, _k) =
        bma_metrics(&th, &env.cfg.shapes, Variant::Cosine, &env.data_b);
    let nlls: Vec<Vec<f64>> = runs.iter().map(|c| c.nll.clone()).collect();
    let losses: Vec<Vec<f64>> = runs.iter().map(|c| c.loss.clone()).collect();
    println!(
        "\n   BMA over the chains  acc {bma_acc:.2} %  NLL {bma_nll:.4}\n   \
         split-R̂(NLL) {:.4}   split-R̂(train L) {:.4}",
        split_rhat(&nlls),
        split_rhat(&losses)
    );

    // ---- 4. random walk or drift? ------------------------------------------
    // A random walk grows as `t^{1/2}` and its chains pick independent
    // directions; a drift grows as `t¹` with a common direction.
    println!(
        "\n4. random walk or drift?\n   {:>5}  {:>9}  {:>12}  {:>9}  {:>9}",
        "chain", "slope", "d(½)/d(1)", "|d| mid", "|d| end"
    );
    let mut slopes = Vec::new();
    for (i, c) in runs.iter().enumerate() {
        let s = growth_slope(&c.t, &c.disp);
        slopes.push(s);
        let half = c.disp.len() / 2;
        let mid = c.disp[half];
        let end = *c.disp.last().unwrap_or(&f64::NAN);
        println!(
            "   {i:>5}  {s:>9.3}  {:>12.3}  {mid:>9.3e}  {end:>9.3e}",
            end / mid
        );
    }
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;

    // Pairwise angle between the chains' *net* displacements: a drift along a
    // common direction gives ~0, independent random walks ~90°.
    let dirs: Vec<Vec<f64>> = runs.iter().map(|c| log_map(&env.u0, &c.final_u)).collect();
    let mut angles = Vec::new();
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let denom = norm_of(&dirs[i]) * norm_of(&dirs[j]);
            if denom > 0.0 {
                let c: f64 = dirs[i]
                    .iter()
                    .zip(&dirs[j])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / denom;
                angles.push(c.abs().clamp(0.0, 1.0).acos().to_degrees());
            }
        }
    }
    let mean_angle = angles.iter().sum::<f64>() / angles.len().max(1) as f64;
    println!(
        "\n   mean growth exponent {mean_slope:.3} (0.5 = random walk, 1.0 = drift)\n   \
         mean pairwise angle between net displacements {mean_angle:.1}° \
         (0° = one direction, 90° = independent)\n   \
         (a 10 000-step run at r₁ = {r1:.1e} covers √t·r₁ = {:.2e} rad if free)",
        (r1 * (steps as f64).sqrt()),
    );

    // ---- 5. the ridge probe on the last chain ------------------------------
    if args.iter().any(|a| a == "--probe-ridge") {
        let dirv = log_map(&env.u0, &runs[runs.len() - 1].final_u);
        let dn = norm_of(&dirv);
        if dn > 1e-9 {
            println!(
                "\n5. ridge probe (from u_Adam along the geodesic to chain {}'s end)\n   \
                 {:>9}  {:>10}  {:>9}  {:>9}  {:>10}",
                runs.len() - 1,
                "d (rad)",
                "train L",
                "N·ΔL",
                "acc %",
                "test NLL"
            );
            for d in [0.01f64, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3] {
                let u: Vec<f64> = (0..p_dim)
                    .map(|i| d.cos() * env.u0[i] + d.sin() * dirv[i] / dn)
                    .collect();
                let l = env.pot(&u);
                println!(
                    "   {d:>9.3}  {l:>10.4}  {:>9.2}  {:>9.2}  {:>10.4}",
                    n * (l - l0),
                    env.acc(&u),
                    env.nll(&u)
                );
            }
        }
    }
}
