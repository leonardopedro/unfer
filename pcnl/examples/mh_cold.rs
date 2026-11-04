//! `mh_cold` — does the *filtered* chain reproduce the cold target's BMA?
//!
//! §6.5.5 measured `π_λ ∝ N(0,σ²I)·exp(−λ·NL)` with the *unadjusted* kernel
//! and found the BMA calibrating monotonically as `λ` grows (ECE 0.190 →
//! 0.067 at the published scale).  An unadjusted chain does not sample its
//! target, so that number could be either:
//!
//! * the tempered posterior's own predictive, in which case the exact chain at
//!   the same `λ` should reproduce it — and the cold ladder is a real
//!   calibration result;
//! * the unadjusted bias, in which case the exact chain at the same `λ` should
//!   be *worse* (as §6.5.3 found for the standard stack at `λ = 1`: unadjusted
//!   1.84–1.92 against exact 2.40–17.14).
//!
//! The filtered chain needs a proposal it can accept.  The kernel's own
//! `β₀ = 0.3` accepts 0 % with no cap and 0.3–20 % with one (§6.5.3), so the
//! usable scale there was `β₀/10`; a colder target is sharper in the
//! likelihood again, so the proposal scale is part of the cell rather than a
//! fixed choice.
//!
//! **What the filter is.**  Two ratios, not one.  The likelihood-only ratio
//! `exp(λN(L(θ) − L(θ')))` is exact Metropolis for the prior only when the
//! proposal's drift is zero, because pCN's prior cancellation needs the
//! drift-free mean `√(1−β²)θ`; with the drift term the proposal is not
//! prior-reversible, so the chain is pCNL, an *approximate* sampler whose
//! invariant law is not `π_λ`.  `exact` adds the two missing terms — the
//! prior ratio and the reverse-proposal ratio `(‖θ'−m(θ)‖² − ‖θ−m(θ')‖²)/2β²σ²` —
//! and is exact for `π_λ` at any `β`.
//!
//! **The radial column, corrected.**  This example's earlier reading — "a
//! chain at `‖θ‖ ≈ 10³` against the `σ√P = 45.2` shell is off-target" — was
//! wrong twice over.  `posterior_radius` shows the cosine likelihood is
//! *exactly radially flat* (`L = 3.48890` at `r = 10` and `3.48893` at
//! `r = 4000`), so `π_λ`'s radial marginal is the prior's shell at every `λ`,
//! and "off the shell" really would be off-target.  But the chain never had a
//! chance to get there: it starts at `θ_A + σξ` with `‖θ_A‖ ≈ 1338` (30× the
//! shell) and at `β = 0.03` the contraction timescale is `1/β² ≈ 1111` steps,
//! while the table runs 400.  The `max` column was the start radius, not an
//! excursion.  `PCNL_START=shell` rescales `θ_A` onto the shell first — which
//! changes no prediction at all, since the loss is radially flat — and the
//! chains then sit at 60 ± 5 for every cell.
//!
//! **At the shell the cold target needs a smaller proposal.**  With `β = 0.03`
//! the exact filter at `λ = 16` accepts **0 %**: a 1.35-norm move changes the
//! direction enough to shift `N·L` by ~10² nats, so no ratio can cover it.
//! `β = 0.01` accepts 86–89 % at the same `λ` and returns the cold posterior's
//! own predictive: NLL 2.18–2.24, ECE 0.17–0.20, accuracy at chance.  That is
//! better calibrated than `λ = 1` (ECE 0.31) but still worse than the uniform
//! predictor's `ln 8 = 2.0794` — calibration without information, and a third
//! of the size §6.5.5's unadjusted ladder reported.
//!
//! ```text
//! cargo run --release -p pcnl --example mh_cold
//! ```

use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::task::{Dataset, MhOptions, Task, adam, mh_pcnl_chain};

const SEEDS: [u64; 2] = [20261020, 20261021];

fn seeds() -> Vec<u64> {
    match std::env::var("PCNL_SEEDS") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect(),
        Err(_) => SEEDS.to_vec(),
    }
}
/// One seed's `(BMA NLL, ECE, BMA acc, accept %)`.
type Row = (f64, f64, f64, f64);

/// `(λ, proposal β, exact)`: the target's temperature, the pCN proposal scale,
/// and whether the full Metropolis–Hastings ratio (prior + reverse proposal)
/// is used instead of the likelihood-only pCNL one.  `β = 0.03` is §6.5.3's
/// "usable `β₀/10`" — which the mean-loss ratio made usable; the exact rows are
/// what the same cell looks like once the ratio targets `π_λ`.
const CELLS: [(f64, f64, bool); 5] = [
    (1.0, 0.03, false),
    (1.0, 0.03, true),
    (16.0, 0.03, false),
    (16.0, 0.03, true),
    (16.0, 0.01, true),
];

/// Env override for a run length other than the table's 400 steps:
/// `PCNL_STEPS=10000 PCNL_BURN=5000 PCNL_THIN=100 PCNL_SEEDS=20261020`.
/// At `β = 0.03` the OU contraction timescale is `1/β² ≈ 1111` steps, so the
/// 400-step table shows a chain that is still relaxing towards the shell; the
/// long run is what makes "equilibrated" true rather than assumed.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn cfg_for(seed: u64, beta_0: f64) -> Config {
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
        chains: 1,
        steps: env_usize("PCNL_STEPS", 400),
        burn: env_usize("PCNL_BURN", 200),
        thin: env_usize("PCNL_THIN", 10),
        batch_size: 64,
        sigma: 0.35,
        beta_0,
        gamma: 1e-3,
        kappa: 0.5,
        pretrain_steps: 1_000,
        pretrain_adam: true,
        run_mh: false,
        ..Config::default()
    }
}

fn main() {
    let probe = cfg_for(SEEDS[0], 0.03);
    let p = probe.shapes.flat_size();
    println!(
        "exact MH chain on the cold target, cosine variant, P = {p}\n\
         π_λ ∝ N(0,σ²I)·exp(−λ·NL); {} steps, burn {}, thin {}, {} seed(s), \
         start = {}\n\
         §6.5.5's unadjusted cold BMA at the published scale: λ = 1 → 2.1958 (ECE 0.1897), \
         λ = 16 → 2.1030 (ECE 0.0670)\n\
         uniform NLL ln 8 = {:.4}\n",
        probe.steps,
        probe.burn,
        probe.thin,
        seeds().len(),
        if std::env::var("PCNL_START").as_deref() == Ok("shell") {
            "σ√P shell"
        } else {
            "θ_A + σξ"
        },
        (8.0f64).ln()
    );
    println!(
        "{:>5}  {:>6}  {:>6}  {:>9}  {:>8}  {:>8}  {:>9}  {:>9}  {:>7}  {:>7}  {:>6}",
        "λ",
        "prop β",
        "ratio",
        "seed",
        "accept",
        "BMA acc",
        "BMA NLL",
        "ECE",
        "final",
        "max",
        "kept"
    );

    let mut cells: Vec<((f64, f64, bool), Vec<Row>)> = Vec::new();
    // `PCNL_MH_CELLS=1,3` restricts the run to those cells (debugging one row
    // of the table without paying for the others).
    let keep: Option<Vec<usize>> = std::env::var("PCNL_MH_CELLS").ok().map(|v| {
        v.split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    });
    for (ci, (lam, prop, exact)) in CELLS.into_iter().enumerate() {
        if keep.as_ref().is_some_and(|k| !k.contains(&ci)) {
            continue;
        }
        let mut rows = Vec::new();
        for seed in seeds() {
            let cfg = cfg_for(seed, prop);
            let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed);
            let data_b = Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, seed);
            let theta_a = adam(
                &data_a,
                &cfg.shapes,
                Variant::Cosine,
                init_flat(&cfg.shapes, 0.1, seed),
                cfg.pretrain_steps,
                cfg.adam_lr,
            )
            .0;
            // Where the chain starts matters more than any of the control
            // knobs here.  θ_A comes out of Adam with ‖θ_A‖ ≈ 1.3e3, thirty
            // times the prior's shell σ√P = 45.2, and at β = 0.03 the OU
            // contraction has timescale 1/β² = 1.1e3 steps — longer than the
            // whole run.  A chain started there never reaches its stationary
            // radius, so every "final radius" column reads ≈ 1.2e3 whatever
            // the ratio does (`max` is the start itself).  `PCNL_START=shell`
            // rescales θ_A onto the shell first; the loss is exactly invariant
            // under a global rescale (`posterior_radius`), so this changes the
            // prior term and the mixing only, not the model's predictions.
            let shell_start = std::env::var("PCNL_START").as_deref() == Ok("shell");
            let theta0 = if shell_start {
                let flat = theta_a.flatten_all().expect("flat");
                let r = pcnl::sampler::norm(&flat).expect("‖θ_A‖");
                let shell = cfg.sigma * (flat.elem_count() as f64).sqrt();
                theta_a
                    .affine(shell / r, 0.0)
                    .expect("rescale onto the shell")
            } else {
                theta_a.clone()
            };
            let out = mh_pcnl_chain(
                &cfg,
                Variant::Cosine,
                &data_b,
                theta0,
                0,
                None,
                MhOptions {
                    grad_norm_cap: None,
                    exact,
                    potential_scale: lam,
                },
            );
            let kept = vec![out.kept.clone()];
            let (acc, nll, ece, _k) = bma_metrics(&kept, &cfg.shapes, Variant::Cosine, &data_b);
            let accept = 100.0 * out.accepted as f64 / out.steps.max(1) as f64;
            println!(
                "{lam:>5.1}  {prop:>6.3}  {:>6}  {seed:>9}  {accept:>7.1}%  {acc:>7.1}%  \
                 {nll:>9.4}  {ece:>9.4}  {:>7.1}  {:>7.1}  {:>6}",
                if exact { "MH" } else { "pCNL" },
                out.radial_final,
                out.radial_max,
                out.kept.len()
            );
            rows.push((nll, ece, acc, accept));
        }
        cells.push(((lam, prop, exact), rows));
    }

    println!("\nsummary (min / median / max over seeds):");
    println!(
        "{:>5}  {:>6}  {:>6}  {:>26}  {:>26}  {:>22}",
        "λ", "prop β", "ratio", "BMA NLL", "ECE", "accept"
    );
    for ((lam, prop, exact), rows) in &cells {
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
        let (a0, a1, a2) = col(&|r| r.3);
        println!(
            "{lam:>5.1}  {prop:>6.3}  {:>6}  {n0:>8.4} /{n1:>8.4} /{n2:>8.4}  \
             {e0:>8.4} /{e1:>8.4} /{e2:>8.4}  {a0:>6.1}% /{a1:>6.1}% /{a2:>6.1}%",
            if *exact { "MH" } else { "pCNL" }
        );
    }
    println!(
        "\nreading: the acceptance column is not enough — 100 % is what a chain that\n\
         only ever improves the likelihood looks like.  Read the radial column with\n\
         it: a row whose radius is far above σ√P = 45.2 is not sampling the target\n\
         (the prior would forbid it), so its BMA is an optimiser's predictive.  Only\n\
         a row that stays near the shell and accepts in the tens of percent can\n\
         speak to whether the cold posterior's predictive is calibrated."
    );
}
