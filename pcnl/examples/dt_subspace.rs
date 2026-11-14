//! §R7.4 — the first DT adaptation cell: rank-4 restricted target, point
//! baselines, per-step scores and paired returns.
//!
//! This is the cell `examples/dt_cell` could not be.  It runs the same anchor,
//! corpora and readouts, but the chain samples R1's affine chart
//! `θ(z) = θ_ref + σBz` at rank `r` (default 4) instead of the ambient `P =
//! 17 408` parameters, and it adds the baselines R7.4's table names: a point DT
//! (Adam), a temperature-scaled point DT tuned on development data, and a
//! 4-member ensemble at matched likelihood evaluations.  The basis axis is the
//! first question R7.0 item 2 asks: informed (Task-B path PCA) against a random
//! control at the same rank, both pinned at the ambient anchor.
//!
//! ```text
//! cargo run --release -p pcnl --example dt_subspace
//! cargo run --release -p pcnl --example dt_subspace -- --rank 4 --seeds 7,8,9
//! cargo run --release -p pcnl --example dt_subspace -- --proposals fixed,pilot --steps 600
//! ```
//!
//! The default is a *screening* row: one chain, 1 000 steps, three seeds, so a
//! failure to meet the R7.4 sampling gate is visible before a 4 h run.  Every
//! number is printed per seed because R2s measured that seed/direction spread
//! exceeds most proposal effects in this crate.

use candle_core::Result;
use pcnl::dt_arm::DtCell;
use pcnl::dt_subspace::{
    BasisKind, ChainCfg, DtRestricted, Proposal, RETURN_ARMS, ReturnArms, adam_point_dev,
    informed_line, informed_line_in, mean_scores, point_ensemble, point_scores, random_line,
    random_line_in, return_readout, run_chains, subsample_draws, tune_temperature,
};
use pcnl::mdp::BehaviourPolicy;
use pcnl::policy_readout::paired_delta;
use pcnl::readout::Rule;

struct Args {
    rank: usize,
    seeds: Vec<u64>,
    chains: usize,
    steps: usize,
    burn: usize,
    thin: usize,
    beta: f64,
    pilot_steps: usize,
    pilot_shrink: f64,
    laplace_shrink: f64,
    proposals: Vec<Proposal>,
    bases: Vec<BasisKind>,
    eval_budget: usize,
    rollouts: usize,
    path_steps: usize,
    /// R7.6's anchor-as-behaviour-policy axis: the policy the *anchor* corpus
    /// was drawn from.  The adaptation corpora and the target never move;
    /// defaults are `DtCell::new`'s (feedback 1.0, epsilon 0.1).
    anchor_feedback: f32,
    anchor_epsilon: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            rank: 4,
            seeds: vec![7, 8, 9],
            chains: 1,
            steps: 1_000,
            burn: 500,
            thin: 10,
            beta: 0.05,
            pilot_steps: 300,
            pilot_shrink: 0.0,
            laplace_shrink: 0.5,
            proposals: vec![Proposal::Fixed, Proposal::Pilot, Proposal::Laplace],
            bases: vec![BasisKind::Informed, BasisKind::Random],
            // Full-corpus evaluations: equal to one chain's step count + pilot
            // by default, so the point arms cost what one chain costs.
            eval_budget: 1_300,
            rollouts: 96,
            path_steps: 60,
            anchor_feedback: 1.0,
            anchor_epsilon: 0.1,
        }
    }
}

fn parse_list<T: Copy>(s: &str, parse: impl Fn(&str) -> Option<T>, what: &str) -> Vec<T> {
    s.split(',')
        .map(|x| parse(x.trim()).unwrap_or_else(|| panic!("bad {what} '{}'", x.trim())))
        .collect()
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let value = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i)
                .unwrap_or_else(|| panic!("{flag} needs a value"))
                .clone()
        };
        match flag {
            "--rank" => a.rank = value(&mut i).parse().expect("--rank"),
            "--seeds" => a.seeds = parse_list(&value(&mut i), |x| x.parse().ok(), "seed"),
            "--chains" => a.chains = value(&mut i).parse().expect("--chains"),
            "--steps" => a.steps = value(&mut i).parse().expect("--steps"),
            "--burn" => a.burn = value(&mut i).parse().expect("--burn"),
            "--thin" => a.thin = value(&mut i).parse().expect("--thin"),
            "--beta" => a.beta = value(&mut i).parse().expect("--beta"),
            "--pilot-steps" => a.pilot_steps = value(&mut i).parse().expect("--pilot-steps"),
            "--pilot-shrink" => a.pilot_shrink = value(&mut i).parse().expect("--pilot-shrink"),
            "--laplace-shrink" => {
                a.laplace_shrink = value(&mut i).parse().expect("--laplace-shrink")
            }
            "--proposals" => a.proposals = parse_list(&value(&mut i), Proposal::parse, "proposal"),
            "--bases" => {
                a.bases = parse_list(
                    &value(&mut i),
                    |x| match x {
                        "informed" => Some(BasisKind::Informed),
                        "random" => Some(BasisKind::Random),
                        "head" => Some(BasisKind::Head),
                        "trunk" => Some(BasisKind::Trunk),
                        "head-random" => Some(BasisKind::HeadRandom),
                        "trunk-random" => Some(BasisKind::TrunkRandom),
                        "rtg" => Some(BasisKind::Rtg),
                        "rtg-random" => Some(BasisKind::RtgRandom),
                        _ => None,
                    },
                    "basis",
                )
            }
            "--evals" => a.eval_budget = value(&mut i).parse().expect("--evals"),
            "--rollouts" => a.rollouts = value(&mut i).parse().expect("--rollouts"),
            "--path-steps" => a.path_steps = value(&mut i).parse().expect("--path-steps"),
            "--anchor-feedback" => {
                a.anchor_feedback = value(&mut i).parse().expect("--anchor-feedback")
            }
            "--anchor-epsilon" => {
                a.anchor_epsilon = value(&mut i).parse().expect("--anchor-epsilon")
            }
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    assert!(a.burn < a.steps, "burn must leave a kept window");
    // The return readout is a paired difference against the behaviour policy,
    // so it needs at least two rollouts to have an MCSE at all.
    assert!(a.rollouts >= 2, "--rollouts must be at least 2");
    a
}

fn main() -> Result<()> {
    let args = parse_args();
    println!(
        "R7.4 screening cell: rank {}, {} chain(s) x {} steps, bases {:?}, proposals {:?}, seeds {:?}",
        args.rank,
        args.chains,
        args.steps,
        args.bases.iter().map(|b| b.name()).collect::<Vec<_>>(),
        args.proposals.iter().map(|p| p.name()).collect::<Vec<_>>(),
        args.seeds
    );
    println!(
        "  point baseline budget {} full-corpus evaluations; ensemble is 4 members at 1/4 each",
        args.eval_budget
    );
    println!(
        "\n{:<5} {:<16} {:<8} {:>9} {:>8} {:>9} {:>10} {:>10} {:>9} {:>7}",
        "seed", "model", "split", "NLL", "Brier", "Spherical", "top-1", "ECE", "ESS(Φ)", "Rhat"
    );

    for &seed in &args.seeds {
        let mut cell = if args.anchor_feedback == 1.0 && args.anchor_epsilon == 0.1 {
            DtCell::new(seed)
        } else {
            DtCell::with_policies(
                seed,
                BehaviourPolicy {
                    feedback: args.anchor_feedback,
                    epsilon: args.anchor_epsilon,
                    seed,
                },
                BehaviourPolicy {
                    feedback: 0.3,
                    epsilon: 0.3,
                    seed: seed ^ 0xa11ce,
                },
            )
        };
        let trace = cell.pretrain(400, 0.02, 0.9)?;
        let anchor = cell.anchor.clone().unwrap();
        let p = cell.shapes.flat_size();
        println!(
            "\nseed {seed}: anchor {:016x}, pretrain {:.4} -> {:.4}, P = {p}, corpora {} / {} / {} transitions",
            cell.anchor_hash.unwrap_or(0),
            trace.first().copied().unwrap_or(f64::NAN),
            trace.last().copied().unwrap_or(f64::NAN),
            cell.adapt_train.transitions(),
            cell.adapt_dev.transitions(),
            cell.adapt_test.transitions()
        );

        // ---- point baselines (same budget for every cell of this seed)
        // The point baseline adapts from the *same* anchor the chains do; a
        // random-init point model would differ in its prior mean, not only in
        // how it is summarised.  Learning rate and stopping step are picked on
        // dev (R7.4's "tuned point DT"), never on test.
        // Matched cost: the arm's total adaptation budget is one chain's, split
        // over the learning-rate grid (the dev checks are counted and printed).
        let lrs = [0.001, 0.003, 0.01];
        let per_lr = (args.eval_budget / lrs.len()).max(1);
        let fit = adam_point_dev(&cell, &anchor, per_lr, &lrs, 25)?;
        let point = fit.theta.clone();
        let (temp, rows) = tune_temperature(&cell, &point, &[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])?;
        let ensemble = point_ensemble(&cell, 4, args.eval_budget / 4, 0.003, seed ^ 0x9e37)?;
        println!(
            "  point: lr {} at step {}, dev NLL {:.4} over {} evaluations ({} lrs x {} steps); \
             temperature {temp:.2} on dev, grid {:?}",
            fit.lr,
            fit.step,
            fit.dev_nll,
            fit.evaluations,
            lrs.len(),
            per_lr,
            rows.iter()
                .map(|(t, n)| format!("{t:.2}:{n:.4}"))
                .collect::<Vec<_>>()
        );

        // ---- readout rows for the point arms and the anchor
        for (split, corpus) in [("dev", &cell.adapt_dev), ("test", &cell.adapt_test)] {
            for (label, theta, t) in [
                ("anchor", &anchor, 1.0),
                ("point", &point, 1.0),
                ("point-temp", &point, temp),
            ] {
                let (s, cal, _) = point_scores(&cell, theta, corpus, t, 10)?;
                print_row(seed, label, split, &s, Some(&cal), f64::NAN, f64::NAN);
            }
            let (s, cal, _) = mean_scores(&cell, &ensemble, corpus, 10)?;
            print_row(seed, "ensemble", split, &s, Some(&cal), f64::NAN, f64::NAN);
        }

        // ---- chain cells: basis x proposal
        let cfg = ChainCfg {
            chains: args.chains,
            steps: args.steps,
            burn: args.burn,
            thin: args.thin,
            beta: args.beta,
            pilot_steps: args.pilot_steps,
            pilot_shrink: args.pilot_shrink,
            laplace_shrink: args.laplace_shrink,
            seed,
            ..ChainCfg::default()
        };
        let mut kept: Vec<(BasisKind, Proposal, Vec<candle_core::Tensor>, f64)> = Vec::new();
        for &kind in &args.bases {
            let line = match kind {
                BasisKind::Informed => {
                    informed_line(&cell, &anchor, args.rank, args.path_steps, 0.02, seed)?
                }
                BasisKind::Random => random_line(&anchor, args.rank, seed)?,
                BasisKind::Head | BasisKind::Trunk | BasisKind::Rtg => informed_line_in(
                    &cell,
                    &anchor,
                    args.rank,
                    args.path_steps,
                    0.02,
                    seed,
                    kind.block(),
                )?,
                BasisKind::HeadRandom | BasisKind::TrunkRandom | BasisKind::RtgRandom => {
                    random_line_in(
                        &anchor,
                        &cell.shapes,
                        args.rank,
                        seed,
                        kind.block().expect("a random block kind has a block"),
                    )?
                }
            };
            println!(
                "  basis {:<8}[{}] hash {:016x}, path |θ| {:.3} -> {:.3}",
                line.kind.name(),
                line.rank,
                line.hash,
                line.path_start,
                line.path_end
            );
            let restricted = DtRestricted::new(&cell, line, 0.35, 1.0)?;
            for &proposal in &args.proposals {
                let runs = run_chains(&restricted, proposal, &cfg)?;
                let r = &runs.report;
                println!(
                    "    {:<7} accept {:.3}, ESS {:.1}, Rhat {:.3}, z_max {:.2}, kept {}, {:?}{} ({:.1}s)",
                    proposal.name(),
                    r.accept,
                    r.ess_phi,
                    r.rhat_phi,
                    r.z_max,
                    r.kept,
                    r.betas
                        .iter()
                        .map(|b| format!("{b:.3}"))
                        .collect::<Vec<_>>(),
                    r.mode_note
                        .as_ref()
                        .map(|n| format!(", {n}"))
                        .unwrap_or_default(),
                    r.seconds
                );
                for (split, corpus) in [("dev", &cell.adapt_dev), ("test", &cell.adapt_test)] {
                    let (s, cal, _) = mean_scores(&cell, &runs.draws, corpus, 10)?;
                    print_row(
                        seed,
                        &format!("pbar-{}-{}", kind.name(), proposal.name()),
                        split,
                        &s,
                        Some(&cal),
                        r.ess_phi,
                        r.rhat_phi,
                    );
                    if kind.is_informed() && split == "test" {
                        // The acceptance criterion is a *paired* difference
                        // against the calibrated point baseline, not the gap of
                        // two unpaired means.
                        let (ps, _, _) = point_scores(&cell, &point, corpus, temp, 10)?;
                        let d: Vec<String> = [Rule::Log, Rule::Brier, Rule::Spherical]
                            .iter()
                            .map(|&r| {
                                let (m, se, _) = paired_delta(&s, &ps, r);
                                format!("{} {:+.4} ± {:.4} ({:+.1}σ)", r.name(), m, se, m / se)
                            })
                            .collect();
                        println!(
                            "    pbar - point-temp, paired on test steps (positive = pbar worse): {}",
                            d.join(", ")
                        );
                    }
                }
                kept.push((kind, proposal, runs.draws, r.ess_phi));
            }
        }

        // ---- returns: the best-mixing informed cell, else the first cell.
        let best = kept
            .iter()
            .filter(|(k, _, _, _)| k.is_informed())
            .max_by(|a, b| a.3.total_cmp(&b.3))
            .or_else(|| kept.first());
        if let Some((kind, proposal, draws, _)) = best {
            // The rollout decoder on a declared subset; the score columns above
            // use every draw.  At 2 800 draws the full average would be
            // `rollouts × horizon × 2 800` forwards per cell.
            let rollout_draws = subsample_draws(draws, 64);
            let arms = ReturnArms {
                behaviour: cell.adapt_train.clone(),
                anchor: anchor.clone(),
                point: point.clone(),
                draws: rollout_draws.clone(),
                ensemble: ensemble.clone(),
                temperature: temp,
            };
            let ret = return_readout(&cell.env, &cell, &arms, args.rollouts, seed ^ 0x11);
            println!(
                "  returns x0 {:?}, {} rollouts, basis {}/{}, point T {temp:.2}, \
                 p-bar decoder on {} of {} draws",
                pcnl::dt_arm::START,
                ret.rollouts,
                kind.name(),
                proposal.name(),
                rollout_draws.len(),
                draws.len()
            );
            for arm in &ret.arms {
                println!("    {:<16} {:+.4} ± {:.4}", arm.name, arm.mean, arm.mcse);
            }
            for (name, diff) in &ret.diffs {
                println!("    {:<16} {}", name, diff.render());
            }
            assert_eq!(
                ret.arms.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
                RETURN_ARMS
            );
        }
    }
    println!(
        "\nA cell whose Gibbs Rhat > 1.01 or ESS < 400 has no converged predictive: \
         report the gate, not the score.  The restricted target is a smaller model \
         (R7.0 item 2), so an informed-vs-random tie is a basis result, not a sampler one."
    );
    Ok(())
}

fn print_row(
    seed: u64,
    model: &str,
    split: &str,
    scores: &pcnl::policy_readout::StepScores,
    cal: Option<&pcnl::policy_readout::Calibration>,
    ess: f64,
    rhat: f64,
) {
    let cal = cal.expect("calibration row");
    println!(
        "{:<5} {:<16} {:<8} {:>9.4} {:>8.4} {:>9.4} {:>10.3} {:>10.4} {:>9} {:>7}",
        seed,
        model,
        split,
        scores.mean(Rule::Log),
        scores.mean(Rule::Brier),
        scores.mean(Rule::Spherical),
        cal.accuracy,
        cal.ece,
        if ess.is_finite() {
            format!("{ess:.1}")
        } else {
            "-".to_string()
        },
        if rhat.is_finite() {
            format!("{rhat:.3}")
        } else {
            "-".to_string()
        }
    );
}
