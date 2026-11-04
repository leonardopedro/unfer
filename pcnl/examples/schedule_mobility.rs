//! `schedule_mobility` — the exponent window of `β_t = β₀(1+γt)^{−κ}`,
//! measured instead of asserted.
//!
//! The decreasing-step conditions belong on the coefficient that
//! multiplies the gradient, `h_t = β_t²σ²/2`, so they read `Σ β_t² = ∞`
//! (the chain never freezes) and `Σ β_t⁴ < ∞` (accumulated gradient
//! noise vanishes) — i.e. `κ ∈ (1/4, 1/2]`. This example prints what that
//! means twice:
//!
//! 1. **The budget.** `Σ β_t²` over a run, the closed-form total, and
//!    what is left. `κ = 0.55` (the plan's original window, and every
//!    run before 2026-11-03) has a *finite* total: 900 at
//!    `β₀ = 0.3, γ = 1e-3`.
//! 2. **The consequence, on an exactly solvable target.** For
//!    `Φ(θ) = ½λ‖θ − c‖²` with prior `N(0, σ²I)` the posterior mean is
//!    `μ* = c·λσ²/(1+λσ²)` and the chain's mean obeys
//!    `m_{t+1} = √(1−β²)m_t − (β²σ²/2)λ(m_t − c)`. Contraction and pull
//!    are both `O(β²)`, so the residual is
//!    `residual(T) = floor · exp(−tail(T))`: the remaining approach is
//!    *exactly* the unspent budget, and the floor is fixed by the whole
//!    of it. A finite budget therefore leaves a floor no horizon can
//!    lower; an unbounded one leaves none.
//!
//! The two `γ` columns are the honest part. At the reference `γ = 1e-3`
//! the budget is 900, far more than any run needs, so `κ = 0.55` and
//! `κ = 1/2` are indistinguishable over 10⁶ steps: the frozen regime is
//! *asymptotic*, and the published runs are valid as fixed-`β` mixing
//! measurements — what they cannot be cited for is convergence to the
//! posterior. At `γ = 1` the budget is 0.9 and the floor is in plain
//! sight.
//!
//! ```text
//! cargo run -p pcnl --example schedule_mobility
//! ```

use pcnl::sampler::SgPcnlScheduler;

/// The chain's mean residual from `μ*` on `Φ(θ) = ½λ‖θ − c‖²`, after
/// `steps` iterations of the exact mean recursion. No sampling, no
/// tensor: this is the kernel's deterministic part.
fn residual(s: SgPcnlScheduler, sigma: f64, lambda: f64, c: f64, steps: usize) -> f64 {
    let mu_star = c * lambda * sigma * sigma / (1.0 + lambda * sigma * sigma);
    let mut m = 0.0f64;
    for t in 0..steps {
        let b = s.beta_at(t);
        let decay = (1.0 - b * b).max(0.0).sqrt();
        let h = b * b * sigma * sigma / 2.0;
        m = decay * m - h * lambda * (m - c);
    }
    (mu_star - m) / mu_star
}

fn main() {
    let (beta_0, sigma, lambda, c) = (0.3f64, 1.0f64, 1.0f64, 100.0f64);
    let horizons = [1_000usize, 10_000, 100_000, 1_000_000];

    println!(
        "mobility of beta_t = {beta_0} (1 + gamma t)^(-kappa), sigma = {sigma}\n\
         kappa = 1/2 is the window's edge (sum b^2 = inf); kappa = 0.55 is outside it"
    );

    for gamma in [1e-3f64, 1.0] {
        println!("\n-- gamma = {gamma} ------------------------------------------------");
        println!(
            "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}",
            "kappa", "sum b^2 @1e4", "sum b^2 @1e6", "budget", "freezes?"
        );
        let mut schedulers = Vec::new();
        for kappa in [0.5f64, 0.55, 0.8] {
            let s = SgPcnlScheduler {
                beta_0,
                gamma,
                kappa,
            };
            schedulers.push((kappa, s));
            let budget = match s.total_mobility() {
                Some(b) => format!("{b:.3}"),
                None => "inf".to_string(),
            };
            println!(
                "{:>6}  {:>12.3}  {:>12.3}  {:>12}  {:>10}",
                kappa,
                s.partial_mobility(10_000),
                s.partial_mobility(1_000_000),
                budget,
                if s.total_mobility().is_some() {
                    "yes"
                } else {
                    "no"
                }
            );
        }

        println!(
            "\n{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "kappa", "res @1e3", "res @1e4", "res @1e5", "res @1e6", "floor", "unspent"
        );
        for (kappa, s) in &schedulers {
            let res: Vec<f64> = horizons
                .iter()
                .map(|t| residual(*s, sigma, lambda, c, *t))
                .collect();
            let (floor, unspent) = match s.tail_mobility(*horizons.last().unwrap()) {
                Some(tail) => (
                    format!("{:.6}", res[3] * (-tail).exp()),
                    format!("{tail:.4}"),
                ),
                None => ("-".to_string(), "inf".to_string()),
            };
            println!(
                "{:>6}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10}  {:>10}",
                kappa, res[0], res[1], res[2], res[3], floor, unspent
            );
        }
        println!(
            "  residual = (mu* - m) / mu*, mu* = {:.1}; at a finite budget the\n  \
             residual's floor is res(T) * exp(-tail(T)) — the same number at every\n  \
             horizon, which is the freeze; with no budget the residual keeps falling.",
            c * lambda * sigma * sigma / (1.0 + lambda * sigma * sigma)
        );
    }
}
