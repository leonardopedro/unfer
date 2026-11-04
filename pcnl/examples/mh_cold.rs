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
//! **What the filter is.**  Its acceptance ratio is `exp(λ(L(θ) − L(θ')))` —
//! the likelihood ratio only.  That is exact Metropolis for the prior only
//! when the proposal's drift is zero, because pCN's prior cancellation needs
//! the drift-free mean `√(1−β²)θ`; with the drift term the proposal is not
//! prior-reversible, so the chain is pCNL, an *approximate* sampler whose
//! invariant law is not `π_λ`.  The radial column is how that bias shows: a
//! chain that samples `π_λ ∝ N(0,σ²I)·exp(−λ·NL)` must stay on the
//! `σ√P = 45.2` shell, and a chain at `‖θ‖ ≈ 10³` is optimising the likelihood
//! while the prior's restoring term is switched off by the small `β` (whose
//! decay timescale is `1/β² ≈ 1100` steps).
//!
//! ```text
//! cargo run --release -p pcnl --example mh_cold
//! ```

use pcnl::experiment::{Config, bma_metrics};
use pcnl::model::{Shapes, Variant, init_flat};
use pcnl::task::{Dataset, MhOptions, Task, adam, mh_pcnl_chain};

const SEEDS: [u64; 2] = [20261020, 20261021];
/// `(λ, proposal β)`: the target's temperature and the pCN proposal scale the
/// filter is given.  `β = 0.03` is §6.5.3's usable `β₀/10`.
/// One seed's `(BMA NLL, ECE, BMA acc, accept %)`.
type Row = (f64, f64, f64, f64);

const CELLS: [(f64, f64); 5] = [
    (1.0, 0.03),
    (1.0, 0.01),
    (16.0, 0.03),
    (16.0, 0.01),
    (16.0, 0.003),
];

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
        steps: 400,
        burn: 200,
        thin: 10,
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
    let p = cfg_for(SEEDS[0], 0.03).shapes.flat_size();
    println!(
        "exact MH chain on the cold target, cosine variant, P = {p}\n\
         π_λ ∝ N(0,σ²I)·exp(−λ·NL); 400 steps, burn 200, thin 10, 2 seeds\n\
         §6.5.5's unadjusted cold BMA at the published scale: λ = 1 → 2.1958 (ECE 0.1897), \
         λ = 16 → 2.1030 (ECE 0.0670)\n\
         uniform NLL ln 8 = {:.4}\n",
        (8.0f64).ln()
    );
    println!(
        "{:>5}  {:>6}  {:>9}  {:>8}  {:>8}  {:>9}  {:>9}  {:>7}  {:>7}  {:>6}",
        "λ", "prop β", "seed", "accept", "BMA acc", "BMA NLL", "ECE", "keep", "radial", "kept"
    );

    let mut cells: Vec<((f64, f64), Vec<Row>)> = Vec::new();
    for (lam, prop) in CELLS {
        let mut rows = Vec::new();
        for seed in SEEDS {
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
            let out = mh_pcnl_chain(
                &cfg,
                Variant::Cosine,
                &data_b,
                theta_a,
                0,
                None,
                MhOptions {
                    grad_norm_cap: None,
                    potential_scale: lam,
                },
            );
            let kept = vec![out.kept.clone()];
            let (acc, nll, ece, _k) = bma_metrics(&kept, &cfg.shapes, Variant::Cosine, &data_b);
            let accept = 100.0 * out.accepted as f64 / out.steps.max(1) as f64;
            println!(
                "{lam:>5.1}  {prop:>6.3}  {seed:>9}  {accept:>7.1}%  {acc:>7.1}%  {nll:>9.4}  \
                 {ece:>9.4}  {:>7.1}  {:>7.1}  {:>6}",
                out.radial_final,
                out.radial_max,
                out.kept.len()
            );
            rows.push((nll, ece, acc, accept));
        }
        cells.push(((lam, prop), rows));
    }

    println!("\nsummary (min / median / max over seeds):");
    println!(
        "{:>5}  {:>6}  {:>26}  {:>26}  {:>22}",
        "λ", "prop β", "BMA NLL", "ECE", "accept"
    );
    for ((lam, prop), rows) in &cells {
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
            "{lam:>5.1}  {prop:>6.3}  {n0:>8.4} /{n1:>8.4} /{n2:>8.4}  \
             {e0:>8.4} /{e1:>8.4} /{e2:>8.4}  {a0:>6.1}% /{a1:>6.1}% /{a2:>6.1}%"
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
