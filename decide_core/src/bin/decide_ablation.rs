//! The `D`-ablation runner: AnaCP Table 2 (rows 13–16) on project data.
//!
//! Run:
//!
//! ```sh
//! cargo run -p decide_core --bin decide_ablation --                # print the sweep
//! cargo run -p decide_core --bin decide_ablation -- --write        # refresh the fixture
//! cargo run -p decide_core --bin decide_ablation -- --dims 64,256  # a sub-sweep
//! ```
//!
//! # What this binary is for
//!
//! `docs/DECIDE_SPEC.md` §4 held the engine's `D = 512` default to be
//! *sensible, not measured*, pending exactly this sweep. This binary produces
//! the measurement and the committed fixture; the spec's verdict is written
//! from its output, not from taste. The verdict it produced is asymmetric —
//! `D = 512` is measured-best for `anacp` and measured-worst for `ananc` — and
//! §4 records that rather than re-tuning a default on synthetic data.
//!
//! Like `decide_report`, it exits 0 whatever the numbers say: a sweep whose
//! result is "the default is not distinguishable from a much smaller D" is a
//! result, not a crash. The gate that keeps it honest is the byte comparison
//! against the committed fixture, in CI and in the crate's own golden test.

use decide_core::ablation::{
    DEFAULT_DIM, DEFAULT_DIMS, DEFAULT_PER_CLASS, d_ablation, fixture_path, fixture_text,
};

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write = args.iter().any(|a| a == "--write");

    let per_class = flag(&args, "--per-class").unwrap_or(DEFAULT_PER_CLASS);
    let dim = flag(&args, "--dim").unwrap_or(DEFAULT_DIM);
    let dims: Vec<usize> = match list(&args, "--dims") {
        Some(v) => v,
        None => DEFAULT_DIMS.to_vec(),
    };
    if dims.is_empty() {
        eprintln!("error: --dims parsed to an empty sweep");
        return std::process::ExitCode::FAILURE;
    }

    println!("decide D-ablation — engine {}", decide_core::ENGINE_VERSION);
    println!(
        "corpus: {per_class} samples/class, dim {dim}, held-out every 4th; \
         sweeping D = {:?}\n",
        dims
    );

    let rows = match d_ablation(per_class, dim, &dims) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    println!(
        "{:<8} {:>6} {:>8} {:>8} {:>8} {:>9} {:>8} {:>7} {:>12}",
        "mode", "D", "acc", "ECE", "A_avg", "forget", "drift", "floors", "state KiB"
    );
    println!("{}", "-".repeat(86));
    for r in &rows {
        let kib = r.state_floats as f64 * 8.0 / 1024.0;
        match (
            &r.refused,
            r.accuracy,
            r.ece,
            r.a_avg,
            r.max_forgetting,
            r.drift,
        ) {
            (Some(reason), _, _, _, _, _) => {
                println!(
                    "{:<8} {:>6}   REFUSED — {}",
                    r.mode, r.projected_dim, reason
                );
            }
            // `drift` may legitimately be absent below D ≥ C: the monitor
            // builds `ananc` engines and inherits the ETF precondition. The
            // four accuracy-family numbers must always be present here.
            (None, Some(acc), Some(ece), Some(a_avg), Some(forget), drift) => {
                let drift = drift
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "--".to_string());
                let floors = r
                    .floored_eigenvalues
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "--".to_string());
                println!(
                    "{:<8} {:>6} {:>8.3} {:>8.4} {:>8.3} {:>9.4} {:>8} {:>7} {:>12.1}",
                    r.mode, r.projected_dim, acc, ece, a_avg, forget, drift, floors, kib
                );
            }
            // A partially-measured row is a bug in the harness, not a state of
            // the world: every unrefused cell measures the four numbers above.
            _ => {
                eprintln!(
                    "error: {}/D={} is neither refused nor fully measured",
                    r.mode, r.projected_dim
                );
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    // The reading, stated rather than left to the table. The bar is one
    // standard error on `n` held-out samples — not a chosen tolerance — so
    // "matches" means "not distinguishable from the best on this corpus",
    // which is the only claim 30 samples can support.
    let n_held = held_out_count(per_class);
    let se_bar = |p: f64| (p * (1.0 - p) / n_held as f64).sqrt();

    println!(
        "\nreading (bar = 1 SE on {n_held} held-out samples; differences within it are noise):",
    );
    for mode in ["anacp", "ananc"] {
        let measured: Vec<_> = rows
            .iter()
            .filter(|r| r.mode == mode && r.accuracy.is_some())
            .collect();
        if measured.is_empty() {
            continue;
        }
        let best = measured
            .iter()
            .max_by(|a, b| {
                a.accuracy
                    .unwrap()
                    .partial_cmp(&b.accuracy.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        let best_acc = best.accuracy.unwrap();
        let tol = se_bar(best_acc);
        let cheapest = measured
            .iter()
            .filter(|r| best_acc - r.accuracy.unwrap() <= tol)
            .min_by_key(|r| r.state_floats)
            .expect("the best row always matches itself");
        let ki = |r: &&decide_core::ablation::AblationRow| r.state_floats as f64 * 8.0 / 1024.0;
        if cheapest.projected_dim == best.projected_dim {
            println!(
                "  {mode}: best acc {best_acc:.3} at D={} ({:.1} KiB); no smaller D \
                 is within {tol:.3}",
                best.projected_dim,
                ki(best),
            );
        } else {
            println!(
                "  {mode}: best acc {best_acc:.3} at D={} ({:.1} KiB); D={} is within \
                 {tol:.3} at {:.1} KiB — {:.0}× less state for a difference this \
                 corpus cannot resolve",
                best.projected_dim,
                ki(best),
                cheapest.projected_dim,
                ki(cheapest),
                ki(best) / ki(cheapest),
            );
        }
        let worst = measured
            .iter()
            .min_by(|a, b| {
                a.accuracy
                    .unwrap()
                    .partial_cmp(&b.accuracy.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        println!(
            "       spread across the sweep: {:.3} (D={}) .. {:.3} (D={})",
            worst.accuracy.unwrap(),
            worst.projected_dim,
            best_acc,
            best.projected_dim,
        );
    }

    // The two requirements that gate `calibrated: true`, stated as they actually
    // came out. A reader must not infer from a tidy accuracy column that the
    // license changed — it did not, at any D.
    let ece_bar = decide_core::DEFAULT_MAX_ECE;
    let mut r2 = Vec::new();
    let mut r5 = Vec::new();
    for r in &rows {
        if let Some(e) = r.ece
            && e <= ece_bar
        {
            r2.push(format!("{}/{}", r.mode, r.projected_dim));
        }
        if let Some(dft) = r.drift
            && dft <= 0.05
        {
            r5.push(format!("{}/{}", r.mode, r.projected_dim));
        }
    }
    println!(
        "\nR2 (ECE ≤ {ece_bar}): {}",
        if r2.is_empty() {
            "fails at every (mode, D) — no D rescues calibration".to_string()
        } else {
            format!("passes only at {}", r2.join(", "))
        }
    );
    println!(
        "R5 (drift ≤ 0.05):  {}",
        if r5.is_empty() {
            "fails at every D — no D rescues class-arrival stability".to_string()
        } else {
            format!("passes only at D={}", r5.join(", "))
        }
    );
    println!(
        "→ `calibrated: true` stays unlicensed: the D-ablation does not change \
         the calibration verdict."
    );

    if write {
        let text = match fixture_text(per_class, dim, &dims) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let path = fixture_path();
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            eprintln!("error: cannot create {}: {e}", dir.display());
            return std::process::ExitCode::FAILURE;
        }
        match std::fs::write(&path, text) {
            Ok(()) => println!("\nwrote {}", path.display()),
            Err(e) => {
                eprintln!("error: cannot write {}: {e}", path.display());
                return std::process::ExitCode::FAILURE;
            }
        }
    } else {
        println!("\n(pass --write to refresh {})", fixture_path().display());
    }

    println!("\nCAVEATS — read before citing any number above:");
    println!("  * The corpus is SYNTHETIC (seeded PRNG + class anchors). The sweep is");
    println!("    a RELATIVE comparison across D on this corpus, not an accuracy claim.");
    println!("  * `ece` is the best of the three D6 mechanisms, in-sample on the");
    println!("    held-out split — comparable to the pinned calibration report.");
    println!("  * `drift` is R5's class-arrival drift (tolerance ε = 0.05); `forget` is");
    println!("    the D-C6 accuracy-matrix decay. Two different measurements.");
    println!("  * `floors` counts whitening eigenvalues floored at 1e-12: directions the");
    println!("    whitening destroyed. n_train=90 caps covariance rank at 89, so D ≥ 128");
    println!("    is rank-deficient and floors D − 89 directions by construction.");
    println!("  * A REFUSED cell is the method's own precondition (ananc needs D >= C),");
    println!("    not a crash and not a zero.");
    std::process::ExitCode::SUCCESS
}

/// How many samples the interleaved split holds out: every 4th of `per_class`
/// per class, across the 5 native labels.
fn held_out_count(per_class: usize) -> f64 {
    let per_class_held = (per_class / 4) as f64;
    (per_class_held * decide_core::properties::NativeLabel::all().len() as f64).max(1.0)
}

fn flag(args: &[String], name: &str) -> Option<usize> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1)?.parse().ok()
}

/// A comma-separated `--dims 64,256,512` list.
fn list(args: &[String], name: &str) -> Option<Vec<usize>> {
    let i = args.iter().position(|a| a == name)?;
    let raw = args.get(i + 1)?;
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(part.parse::<usize>().ok()?);
    }
    Some(out)
}
