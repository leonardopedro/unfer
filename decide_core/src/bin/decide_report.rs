//! D6 / D-C6 — produce the pinned calibration report and the benchmark numbers.
//!
//! Run:
//!
//! ```sh
//! cargo run -p decide_core --bin decide_report --            # print the report
//! cargo run -p decide_core --bin decide_report -- --write    # refresh the fixture
//! ```
//!
//! # What this binary is for
//!
//! The plan calls the calibration report a **pinned fixture** and requires that
//! `calibrated: true` be emitted only for a passing `(engine_version,
//! calibrator_version)` pair. That makes this file load-bearing: it is the only
//! thing that can move the engine from "scores" to "probabilities", so its
//! output is committed, reviewed, and diffed like any other golden artifact.
//!
//! # What a failing report means
//!
//! Not a bug in this binary. A report where R2 (ECE ≤ 0.05) or R5 (class-arrival
//! stability) fails is the engine correctly declining to claim a calibration it
//! has not earned. The binary prints the failures prominently and still writes
//! the fixture, because "we measured and it fell short" is a result worth
//! pinning; omitting it would leave the next run with nothing to compare against.

use std::collections::BTreeMap;

use decide_core::properties::NativeLabel;
use decide_core::report::{accuracy_matrix, engine_version, fit_all_candidates};

/// The committed fixture. Relative to the crate root, so it moves with the code.
const FIXTURE: &str = "tests/fixtures/decide_calibration_report.json";

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write = args.iter().any(|a| a == "--write");

    let per_class = flag(&args, "--per-class").unwrap_or(24);
    let dim = flag(&args, "--dim").unwrap_or(24);

    println!("decide report — engine {}", engine_version());
    println!("corpus: {per_class} samples/class, dim {dim}, held-out every 4th\n");

    let outcomes = match fit_all_candidates(per_class, dim) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    // A stable, diffable rendering: sorted by (mode, mechanism) so adding a
    // candidate does not reshuffle every line.
    let mut rows: BTreeMap<(String, String), _> = BTreeMap::new();
    for o in &outcomes {
        rows.insert((o.mode.clone(), format!("{:?}", o.mechanism)), o.clone());
    }

    println!(
        "{:<8} {:<12} {:>8} {:>8} {:>8} {:>8}",
        "mode", "mechanism", "acc", "ECE", "AUC", "FPR95"
    );
    println!("{}", "-".repeat(56));
    for ((mode, mech), o) in &rows {
        println!(
            "{mode:<8} {mech:<12} {:>8.3} {:>8.4} {:>8.4} {:>8.4}",
            o.accuracy, o.ece, o.auc, o.fpr95
        );
    }

    println!("\nR1–R5 verdicts:");
    let mut any_pass = false;
    let mut all_failures: Vec<String> = Vec::new();
    for ((mode, mech), o) in &rows {
        let r = &o.report;
        let mark = if r.meets_contract() { "PASS" } else { "FAIL" };
        if r.meets_contract() {
            any_pass = true;
        }
        for f in r.failed_requirements() {
            all_failures.push(format!("{mode}/{mech}: {f}"));
        }
        println!(
            "  {mode:<8} {mech:<12} {mark}   (n={}, ece={:.4}, eps={:.3} drift={:.4})",
            r.n_samples, r.ece, r.max_drift_epsilon, r.observed_max_drift
        );
    }

    // D-C6's anti-forgetting matrix, for both modes.
    println!("\nforgetting (task-incremental accuracy matrix):");
    let labels = NativeLabel::all();
    for (name, mode) in [
        ("anacp", decide_core::Mode::Anacp),
        ("ananc", decide_core::Mode::Ananc),
    ] {
        match accuracy_matrix(&labels, per_class, dim, mode) {
            Ok(m) => {
                let (last, avg) = m.last_and_average();
                println!(
                    "  {name}: max_forgetting={:.4}  A_avg={:.4}  A_last={:?}",
                    m.max_forgetting(),
                    avg,
                    last.iter()
                        .map(|v| if v.is_finite() {
                            format!("{v:.2}")
                        } else {
                            "undef".into()
                        })
                        .collect::<Vec<_>>()
                );
                println!("    matrix (A[t][i]):");
                for (t, row) in m.rows.iter().enumerate() {
                    let cells: Vec<String> = row
                        .iter()
                        .map(|v| {
                            if v.is_finite() {
                                format!("{v:.2}")
                            } else {
                                " -- ".to_string()
                            }
                        })
                        .collect();
                    println!("      t={t}: [{}]", cells.join(" "));
                }
            }
            Err(e) => println!("  {name}: unavailable ({e})"),
        }
    }

    let json = serde_json::json!({
        "engine_version": engine_version(),
        "corpus": { "per_class": per_class, "dim": dim, "held_out_every": 4 },
        "candidates": outcomes,
        "any_candidate_meets_contract": any_pass,
        "caveats": [
            "SYNTHETIC CORPUS. `properties::native_corpus` generates its embeddings from a seeded PRNG with class-dependent anchors. It checks that the mechanisms are wired correctly end to end; it does NOT establish accuracy or calibration on real project data.",
            "AUC = 1.0 / FPR95 = 0.0 is a property of that synthetic separability, not a claim about open-world performance. A perfect AUC here should be read as 'the plumbing works', never as 'this detects OOD well'.",
            "The R5 drift bound is measured on a held-out-class comparison, which is a lower bound on the drift a real class-arrival stream would cause.",
            "`n_samples` is small (held-out is one in four of the corpus). A metric on tens of samples has wide error bars; treat ECE differences below ~0.05 as noise.",
            "A candidate that PASSES here licenses `calibrated: true` for the pinned (engine_version, calibrator_version) pair ONLY, and only via `Calibrator::is_calibrated`, which also re-checks the measured ECE.",
        ],
    });
    let text =
        serde_json::to_string_pretty(&json).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));

    if write {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            eprintln!("error: cannot create {}: {e}", dir.display());
            return std::process::ExitCode::FAILURE;
        }
        match std::fs::write(&path, format!("{text}\n")) {
            Ok(()) => println!("\nwrote {}", path.display()),
            Err(e) => {
                eprintln!("error: cannot write {}: {e}", path.display());
                return std::process::ExitCode::FAILURE;
            }
        }
    } else {
        println!("\n(pass --write to refresh {})", FIXTURE);
    }

    if all_failures.is_empty() {
        println!(
            "\nVERDICT: every candidate met the contract; `calibrated: true` is licensed for the pinned pair."
        );
        std::process::ExitCode::SUCCESS
    } else {
        // Not a process failure: a failing calibration report is a legitimate
        // outcome. The exit code stays 0 so CI does not go red on a *measurement*,
        // only on a *crash*. What must not happen is a consumer reading a
        // probability out of this, which `Calibrator::is_calibrated` prevents.
        println!(
            "\nVERDICT: NO candidate met the full contract. These are the unmet requirements:"
        );
        for f in &all_failures {
            println!("  - {f}");
        }
        println!("\nEngine output therefore stays a SCORE. `calibrated: true` is not licensed.");
        println!("(exit code is 0: a failing measurement is a result, not a crash.)");
        println!("\nCAVEATS — read before citing any number above:");
        println!("  * The corpus is SYNTHETIC (seeded PRNG + class anchors). This report");
        println!("    checks that the mechanisms are wired correctly, not that they are");
        println!("    accurate or calibrated on real project data.");
        println!("  * AUC = 1.0 / FPR95 = 0.0 is a property of that separability. It means");
        println!("    'the plumbing works', never 'this detects OOD well'.");
        println!("  * n_samples is small; ECE differences under ~0.05 are noise.");
        std::process::ExitCode::SUCCESS
    }
}

fn flag(args: &[String], name: &str) -> Option<usize> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1)?.parse().ok()
}
