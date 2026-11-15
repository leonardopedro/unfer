//! The declared R9 campaign runner (`EDGE0_RLCD_PLAN.md` §5), with a wall-time
//! cap fixed **before** execution.
//!
//! ```text
//! R9_SEEDS=5 R9_UPDATES=2000 R9_WALL_CAP=2700 \
//!   cargo run --release -p pcnl --example r9_campaign
//! ```
//!
//! Declared split: 2048/256/256/512 worlds (train/dev/calibration/test), 32
//! worlds per update with all their questions, at most 2000 updates per arm,
//! four arms, and five paired training seeds over **one** frozen world manifest
//! (`R9_DATA_SEED`, default `20261115`). The cap is consulted between seeds:
//! a seed either runs to its declared budget or is not started, and the count of
//! started/completed seeds is printed and written. No extra seed or width search
//! happens.
//!
//! Artifacts: one NDJSON row per completed seed at `R9_OUT`
//! (default `target/r9_campaign.ndjson`), each carrying its own world-clustered
//! paired `RLCD − supervised` difference and SE. The summary adds the
//! across-seed mean with a 95 % `t` interval on 4 degrees of freedom (5 seeds).
//! Negative favours RLCD. An interval containing 0 is reported as inconclusive,
//! not promoted.

use pcnl::edge0_campaign::{
    Arm, BatchUnit, CampaignConfig, CampaignReport, build_samples, predictions, run,
};
use pcnl::edge0_model::Edge0Config;
use pcnl::edge0_world::Split;
use std::io::Write;
use std::time::Instant;

/// The declared split (train, dev, calibration, test).
const DECLARED_WORLDS: [usize; 4] = [2048, 256, 256, 512];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rss_peak_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// Two-sided 95 % `t` quantile for the small seed counts this runner allows.
fn t95(df: usize) -> f64 {
    match df {
        0 => f64::INFINITY,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        _ => 1.96,
    }
}

fn arm_row(rep: &CampaignReport, arm: Arm) -> String {
    rep.arms
        .iter()
        .find(|a| a.arm == arm)
        .map(|a| {
            format!(
                "{{\"arm\":\"{}\",\"brier_excess\":{:.8},\"nll\":{:.8},\"passes\":{}}}",
                a.arm.label(),
                a.brier_excess,
                a.nll,
                a.passes
            )
        })
        .unwrap_or_else(|| "null".into())
}

fn main() {
    let seeds = env_usize("R9_SEEDS", 5);
    let updates = env_usize("R9_UPDATES", 2000);
    let cap = env_usize("R9_WALL_CAP", 2700);
    let warm = env_usize("R9_WARM_START", 30);
    let batch_worlds = env_usize("R9_BATCH_WORLDS", 32);
    let data_seed: u64 = std::env::var("R9_DATA_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2026_11_15);
    let out_path = std::env::var("R9_OUT").unwrap_or_else(|_| "target/r9_campaign.ndjson".into());
    let pred_path = std::env::var("R9_PRED_OUT")
        .unwrap_or_else(|_| "target/r9_predictions.ndjson".into());
    // Fresh prediction file per run: the secondary-endpoint evaluator reads one
    // campaign's complete dev/test prediction set, never a mix of runs.
    if let Some(parent) = std::path::Path::new(&pred_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::File::create(&pred_path);

    println!("R9 DECLARED CAMPAIGN");
    println!(
        "split (worlds): {:?} | updates/arm <= {updates} | batch = {batch_worlds} worlds \
         (all questions) | warm start = {warm} shared supervised updates",
        DECLARED_WORLDS
    );
    println!(
        "paired training seeds: {seeds} over fixed data seed {data_seed} | \
         hardware wall-time cap: {cap} s (declared now, checked between seeds)"
    );
    println!("artifact: {out_path}");
    println!("predictions: {pred_path}");
    println!();

    let started = Instant::now();
    let mut rows: Vec<(f64, f64, usize, f64)> = Vec::new(); // diff, se, passes, wall
    let mut completed = 0usize;

    for k in 0..seeds {
        let elapsed = started.elapsed().as_secs() as usize;
        if elapsed >= cap {
            println!("cap reached after {completed} completed seed(s); seed {k} not started");
            break;
        }
        let cfg = CampaignConfig {
            model: Edge0Config::tiny(),
            seed: 9_001 + k as u64,
            data_seed: Some(data_seed),
            batch_unit: BatchUnit::Worlds,
            worlds: DECLARED_WORLDS,
            updates,
            warm_start: warm,
            batch_worlds,
            ..Default::default()
        };
        let t0 = Instant::now();
        let rep = match run(&cfg) {
            Ok(r) => r,
            Err(e) => {
                println!("seed {k} failed before reporting: {e}");
                break;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let (diff, se) = rep.paired_contrast.unwrap_or((f64::NAN, f64::NAN));
        let passes: usize = rep.arms.iter().map(|a| a.passes).sum();
        let rss = rss_peak_kb();
        completed += 1;

        println!(
            "seed {k} (init seed {}): {:.1} s, RSS {:.0} MB, passes {}",
            cfg.seed,
            secs,
            rss as f64 / 1024.0,
            passes
        );
        for arm in Arm::ALL {
            if let Some(a) = rep.arms.iter().find(|a| a.arm == arm) {
                println!(
                    "  {:<30} Brier excess {:>9.5} | NLL {:>9.5} | passes {:>7}",
                    a.arm.label(),
                    a.brier_excess,
                    a.nll,
                    a.passes
                );
            }
        }
        println!(
            "  paired RLCD - supervised: {diff:+.5} +/- {se:.5} (world-clustered SE; negative favours RLCD)"
        );
        println!("  world manifest hash {:#x}", rep.world_hash);
        rows.push((diff, se, passes, secs));

        // Export dev, calibration and test marginals for every arm, so the
        // secondary endpoints (including the calibration-selected abstention
        // threshold) can be recomputed from the artifact without retraining.
        let all = build_samples(&cfg);
        let eval_samples: Vec<_> = all
            .into_iter()
            .filter(|s| matches!(s.split, Split::Dev | Split::Calibration | Split::Test))
            .collect();
        if let Ok(mut pf) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&pred_path)
        {
            for (arm, theta) in &rep.trained {
                let preds = predictions(theta, &cfg.model, &eval_samples).expect("predictions");
                for p in preds {
                    let truth: Vec<String> = p.truth.iter().map(|x| format!("{x:.10}")).collect();
                    let probs: Vec<String> = p.probs.iter().map(|x| format!("{x:.10}")).collect();
                    let _ = writeln!(
                        pf,
                        "{{\"kind\":\"r9-pred\",\"seed\":{k},\"init_seed\":{},\"arm\":\"{}\",\
                         \"split\":\"{:?}\",\"world\":{},\"qid\":{},\"qkind\":\"{}\",\n                         \"label\":{},\"truth\":[{}],\"probs\":[{}]}}",
                        cfg.seed,
                        arm.label(),
                        p.split,
                        p.world,
                        p.qid,
                        p.kind,
                        p.label,
                        truth.join(","),
                        probs.join(",")
                    );
                }
            }
        }

        if let Some(parent) = std::path::Path::new(&out_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path)
        {
            let _ = writeln!(
                f,
                "{{\"kind\":\"r9-seed\",\"seed_index\":{k},\"init_seed\":{},\"data_seed\":{data_seed},\
                 \"worlds\":{:?},\"updates\":{updates},\"warm_start\":{warm},\
                 \"batch_worlds\":{batch_worlds},\"batch_unit\":\"worlds\",\
                 \"wall_s\":{secs:.3},\"maxrss_kb\":{rss},\"passes\":{passes},\
                 \"rlcd\":{},\"supervised\":{},\"brier_diff\":{diff:.8},\"brier_se\":{se:.8},\
                 \"world_hash\":\"0x{:x}\"}}",
                cfg.seed,
                DECLARED_WORLDS,
                arm_row(&rep, Arm::Rlcd),
                arm_row(&rep, Arm::Supervised),
                rep.world_hash
            );
        }
    }

    println!();
    println!(
        "completed seeds: {completed}/{seeds} | wall {:.1} s of cap {cap} s",
        started.elapsed().as_secs_f64()
    );
    let finite: Vec<&(f64, f64, usize, f64)> = rows
        .iter()
        .filter(|r| r.0.is_finite() && r.1.is_finite())
        .collect();
    if finite.len() >= 2 {
        let k = finite.len() as f64;
        let mean = finite.iter().map(|r| r.0).sum::<f64>() / k;
        let var = finite
            .iter()
            .map(|r| (r.0 - mean) * (r.0 - mean))
            .sum::<f64>()
            / (k - 1.0);
        let se = (var / k).sqrt();
        let half = t95(finite.len() - 1) * se;
        let total_passes: usize = rows.iter().map(|r| r.2).sum();
        let total_wall: f64 = rows.iter().map(|r| r.3).sum();
        println!(
            "across-seed paired RLCD - supervised: {mean:+.5} +/- {half:.5} \
             (95% t, df={}; mean +/- {se:.5} SE)",
            finite.len() - 1
        );
        println!(
            "budget: {total_passes} forward/backward passes over {completed} seeds, \
             {total_wall:.1} s summed seed wall time"
        );
        if mean - half <= 0.0 && mean + half >= 0.0 {
            println!(
                "verdict: inconclusive at the declared cost (interval contains no improvement)"
            );
        } else if mean + half < 0.0 {
            println!("verdict: interval excludes no improvement, in RLCD's favour");
        } else {
            println!("verdict: interval excludes improvement; RLCD is worse at this cost");
        }
        let per_seed: Vec<String> = finite
            .iter()
            .map(|r| format!("{:.5}+/-{:.5}", r.0, r.1))
            .collect();
        println!("per-seed contrasts: {}", per_seed.join(", "));
        if let Some(parent) = std::path::Path::new(&out_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path)
        {
            let _ = writeln!(
                f,
                "{{\"kind\":\"r9-summary\",\"seeds\":{},\"completed\":{completed},\
                 \"updates\":{updates},\"warm_start\":{warm},\"batch_worlds\":{batch_worlds},\
                 \"data_seed\":{data_seed},\"mean_diff\":{mean:.8},\"half95\":{half:.8},\
                 \"se\":{se:.8},\"total_passes\":{total_passes},\"wall_s\":{total_wall:.3},\
                 \"cap_s\":{cap}}}",
                seeds
            );
        }
    } else {
        println!("across-seed interval unavailable: fewer than two completed seeds");
    }
}
