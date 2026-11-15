//! E3/E4 mechanics: the supervised comparator, the exact-marginal control, the
//! required RLCD arm and the frozen-policy arm, on the reduced-budget campaign.
//!
//! `cargo run -p pcnl --example edge0_campaign`
//!
//! This is **not** the declared R9 campaign (`EDGE0_RLCD_PLAN.md` §5: 5 seeds,
//! 2048/256/256/512 worlds, up to 2000 updates). The defaults are a labelled
//! mechanics run; the reported numbers show the arms execute, not that RLCD
//! wins. A negative or inconclusive contrast is a legitimate deliverable and is
//! printed without editing.

use pcnl::edge0_campaign::{Arm, CampaignConfig, run};

fn main() {
    let cfg = CampaignConfig::default();
    let report = run(&cfg).expect("campaign");
    println!("R9 E3/E4 — REDUCED-BUDGET MECHANICS (not the declared campaign)");
    println!(
        "model: P={} routed-active={} | worlds train/dev/cal/test = {:?} | updates/arm={} | batch={} worlds | seed={}",
        report.params_total,
        report.params_routed_active,
        report.config.worlds,
        report.config.updates,
        report.config.batch_worlds,
        report.config.seed
    );
    println!(
        "warm start: {} shared supervised updates | world manifest hash {:#x}",
        report.config.warm_start, report.world_hash
    );
    println!();
    println!(
        "{:<32} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "arm", "Brier excess", "NLL", "pol-ent", "path-ent", "passes"
    );
    for arm in &report.arms {
        println!(
            "{:<32} {:>12.5} {:>10.5} {:>10.5} {:>10.5} {:>10}",
            arm.arm.label(),
            arm.brier_excess,
            arm.nll,
            arm.policy_entropy,
            arm.mean_path_entropy,
            arm.passes
        );
    }
    println!();
    match report.paired_contrast {
        Some((diff, se)) => println!(
            "paired RLCD − supervised (world-clustered): {diff:+.5} ± {se:.5} \
             (negative favours RLCD; reduced-budget, single seed)"
        ),
        None => println!("paired contrast unavailable (too few test worlds)"),
    }
    assert!(
        report.arms.iter().any(|a| a.arm == Arm::Rlcd)
            && report.arms.iter().any(|a| a.arm == Arm::Supervised),
        "both required arms must execute"
    );
    println!();
    println!(
        "scope: mechanics only. The analytic gradient gate (tests/edge0_rlcd.rs) \
         must pass before any training evidence is interpreted; the declared \
         5-seed campaign and the Lean4 statements (E6) are not run here."
    );
}
