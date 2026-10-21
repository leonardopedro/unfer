//! CLI runner for the plan's experiment (docs/plan_bayes_ngpt.md).
//!
//! ```text
//! cargo run --release -p pcnl            # full run (10k steps/chain)
//! cargo run --release -p pcnl -- quick   # debug-sized run
//! cargo run --release -p pcnl -- --report-file OUT.txt
//! cargo run --release -p pcnl -- --save-dir DIR --beta0 0.3 --kappa 0.55
//! ```
//!
//! Deterministic: every number below is a function of `Config::seed`.

use pcnl::experiment::{Config, run_experiment};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = Config::default();
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "quick" => {
                cfg = Config::quick();
                if let Some(f) = out.clone() {
                    out = Some(f); // keep
                }
            }
            "--quick" => cfg = Config::quick(),
            "--steps" => {
                i += 1;
                cfg.steps = args
                    .get(i)
                    .expect("--steps N")
                    .parse()
                    .expect("steps is a number");
                cfg.burn = cfg.steps / 2;
            }
            "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .expect("--seed N")
                    .parse()
                    .expect("seed is a number");
            }
            "--report-file" => {
                i += 1;
                out = Some(args.get(i).expect("--report-file PATH").clone());
            }
            // §1/§3C: sweep configuration + step-size tuning
            "--save-dir" => {
                i += 1;
                cfg.save_dir = Some(args.get(i).expect("--save-dir DIR").clone().into());
            }
            "--sigma" | "--beta0" | "--gamma" | "--kappa" => {
                let key = args[i].trim_start_matches('-').to_string();
                i += 1;
                let v: f64 = args.get(i).expect("FLOAT").parse().expect("a float");
                match key.as_str() {
                    "sigma" => cfg.sigma = v,
                    "beta0" => cfg.beta_0 = v,
                    "gamma" => cfg.gamma = v,
                    "kappa" => cfg.kappa = v,
                    _ => unreachable!(),
                }
            }
            other => {
                eprintln!(
                    "unknown arg: {other} (try: quick | --steps N | --seed N | --report-file PATH | \
                     --save-dir DIR | --sigma F | --beta0 F | --gamma F | --kappa F)"
                );
                std::process::exit(2);
            }
        }
        i += 1;
    }

    eprintln!(
        "running SG-pCNL experiment: {} steps x {} chains x 2 variants, P={} …",
        cfg.steps,
        cfg.chains,
        cfg.shapes.flat_size()
    );
    let report = run_experiment(&cfg);
    let text = report.render();
    print!("{text}");
    if let Some(path) = out {
        std::fs::write(&path, &text).expect("write report");
        eprintln!("report written to {path}");
    }
}
