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
            // §3C Option 2 / Phase 4b: cyclical cosine schedule
            "--cyclical" => {
                i += 1;
                cfg.cyclical_period = args
                    .get(i)
                    .expect("--cyclical PERIOD")
                    .parse()
                    .expect("period is a number");
            }
            // Phase 4b: disable the filtered comparator (default: on)
            "--no-mh" => cfg.run_mh = false,
            // §6.4.1's reachability follow-up: N(θ_A, σ²I) adaptation
            // prior — the chains explore around the pre-trained
            // weights instead of the origin's prior shell.
            "--prior-center-a" => cfg.prior_center_at_a = true,
            // task×budget matrix: adaptation target (p:q:r = "(pa+qb+r) mod k");
            // "b" is the canonical (a+2b) mod k.
            "--task-b" => {
                i += 1;
                let spec = args.get(i).expect("--task-b b | --task-b p:q:r").clone();
                cfg.task_b = if spec == "b" {
                    pcnl::task::Task::B
                } else {
                    let parts: Vec<usize> = spec
                        .split(':')
                        .map(|x| x.parse().expect("p:q:r are numbers"))
                        .collect();
                    assert!(parts.len() == 3, "--task-b p:q:r (or 'b')");
                    pcnl::task::Task::Modular {
                        p: parts[0],
                        q: parts[1],
                        r: parts[2],
                    }
                };
            }
            // task×budget matrix: class count (vocab follows: k+1 incl. BOS)
            "--k" => {
                i += 1;
                cfg.k = args.get(i).expect("--k N").parse().expect("k is a number");
                cfg.shapes.vocab = cfg.k + 1;
            }
            "--n-train" => {
                i += 1;
                cfg.n_train = args
                    .get(i)
                    .expect("--n-train N")
                    .parse()
                    .expect("n-train is a number");
            }
            "--pretrain" => {
                i += 1;
                cfg.pretrain_steps = args
                    .get(i)
                    .expect("--pretrain N")
                    .parse()
                    .expect("pretrain is a number");
            }
            "--chains" => {
                i += 1;
                cfg.chains = args
                    .get(i)
                    .expect("--chains N")
                    .parse()
                    .expect("chains is a number");
            }
            // task×budget matrix: model size (the P axis of the budget).
            "--d" => {
                i += 1;
                cfg.shapes.d = args.get(i).expect("--d N").parse().expect("d is a number");
            }
            "--ff" => {
                i += 1;
                cfg.shapes.ff = args
                    .get(i)
                    .expect("--ff N")
                    .parse()
                    .expect("ff is a number");
            }
            "--layers" => {
                i += 1;
                cfg.shapes.layers = args
                    .get(i)
                    .expect("--layers N")
                    .parse()
                    .expect("layers is a number");
            }
            "--init-std" => {
                i += 1;
                cfg.init_std = args
                    .get(i)
                    .expect("--init-std F")
                    .parse()
                    .expect("init-std is a float");
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
                     --save-dir DIR | --cyclical PERIOD | --no-mh | --task-b b|p:q:r | --k N | \
                     --n-train N | --pretrain N | --chains N | --d N | --ff N | --layers N | \
                     --init-std F | --sigma F | --beta0 F | --gamma F | --kappa F | \
                     --prior-center-a)"
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
