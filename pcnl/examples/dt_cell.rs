//! §R7.4 — the DT cell, read out: per-step scores, calibration, paired returns.
//!
//! Runs `dt_arm::DtCell` end to end at the R7.4 shapes: anchor pre-train on the
//! anchor corpus (ambient seeds, hash printed), two unadjusted SG-pCNL chains
//! at the full DT target on a *different behaviour policy*, then the three
//! readout families `policy_readout` exists for:
//!
//! 1. **per-step proper scores** on dev and test — NLL (the training loss), and
//!    Brier and spherical of the same distributions;
//! 2. **max-probability calibration** with per-bin counts, plus top-1 match
//!    (the decision-level number);
//! 3. **paired return readouts** — behaviour policy, anchor sampled and
//!    argmax, posterior predictive `p̄` sampled and argmax, and the exact
//!    optimum, all on one uniform stream and one noise sequence per replicate.
//!
//! What this prints is a cell *without its baselines*: the Adam point DT, the
//! temperature-tuned point DT and the matched-evaluation ensemble that R7.4
//! lists are not built yet, so the anchor-vs-`p̄` columns are not the R7.4
//! comparison and must not be quoted as one.  It is the readout wiring with
//! measured numbers, which is what the row needs before a promotion run.
//!
//! ```text
//! cargo run --release -p pcnl --example dt_cell
//! ```

use pcnl::dt_arm::{ARM_NAMES, DtCell};
use pcnl::readout::Rule;

fn main() -> candle_core::Result<()> {
    let mut cell = DtCell::new(11);
    let trace = cell.pretrain(400, 0.02, 0.9)?;
    println!(
        "anchor: hash {:016x}, pretrain loss {:.4} -> {:.4}",
        cell.anchor_hash.unwrap_or(0),
        trace.first().copied().unwrap_or(f64::NAN),
        trace.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "corpora: anchor {} / dev {} transitions, adapt {} / dev {} / test {}",
        cell.anchor_train.transitions(),
        cell.anchor_dev.transitions(),
        cell.adapt_train.transitions(),
        cell.adapt_dev.transitions(),
        cell.adapt_test.transitions()
    );

    let p = cell.shapes.flat_size() as f64;
    let report = cell.run_chains(&pcnl::dt_arm::ChainCfg::default())?;
    println!(
        "chains: {} x {} steps, {} draws kept, mean dev NLL per chain {:?}",
        report.chains, report.steps, report.kept, report.mean_nll_per_chain
    );
    println!(
        "  P = {p:.0} parameters, so the prior shell is sigma*sqrt(P) = {:.1}; max radius reached {:.3}",
        0.35 * p.sqrt(),
        report.rad_max
    );

    let anchor = cell.anchor.clone().unwrap();
    println!("\nper-step scores");
    println!("point   split  steps   NLL     Brier   Spherical  top-1    ECE    mean-conf");
    for (split, corpus) in [("dev", &cell.adapt_dev), ("test", &cell.adapt_test)] {
        let (a_scores, a_loss) = cell.scores_at(&anchor, corpus)?;
        let a_cal = cell.calibration_at(&anchor, corpus, 10)?;
        let p_scores = cell.posterior_scores(corpus)?;
        let p_cal = cell.posterior_calibration(corpus, 10)?;
        for (label, scores, cal) in [("anchor", &a_scores, &a_cal), ("p-bar", &p_scores, &p_cal)] {
            println!(
                "{label:<7} {split:<5} {:>5}   {:.4}  {:.4}  {:.4}     {:.3}   {:.4}  {:.4}",
                scores.n,
                scores.mean(Rule::Log),
                scores.mean(Rule::Brier),
                scores.mean(Rule::Spherical),
                cal.accuracy,
                cal.ece,
                cal.mean_confidence
            );
        }
        println!("        anchor loss {a_loss:.6} (equals its NLL column by construction)");
    }

    let returns = cell.returns(96, 5)?;
    println!(
        "\npaired return readout, x0 {:?}, {} rollouts",
        pcnl::dt_arm::START,
        returns.rollouts
    );
    for arm in &returns.arms {
        println!("  {:<14} {:+.4} ± {:.4}", arm.name, arm.mean, arm.mcse);
    }
    println!("paired differences vs behaviour (resolution: |d| > 2 MCSE)");
    for (name, diff) in &returns.diffs {
        println!("  {:<14} {}", name, diff.render());
    }
    assert_eq!(
        returns
            .arms
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        ARM_NAMES
    );
    println!(
        "\nread this as a screening cell, not the R7.4 comparison: no point-DT, \
         ensemble or rank-4 restriction is built yet, and the draws are prior-scale \
         (the chains stay on the sigma*sqrt(P) shell, so p-bar averages over policies \
         that differ far more than the likelihood does)."
    );
    Ok(())
}
