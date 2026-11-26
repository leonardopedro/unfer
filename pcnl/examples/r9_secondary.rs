//! Recompute the declared R9 campaign's **secondary endpoints** from the stored
//! per-question prediction artifact.
//!
//! ```text
//! cargo run --release -p pcnl --example r9_secondary -- \
//!   target/r9_predictions.ndjson target/r9_secondary.ndjson
//! ```
//!
//! The artifact is written by `examples/r9_campaign.rs`; this program never
//! retrains and never reads a live model. It reports, per arm and paired
//! `RLCD − supervised`:
//!
//! * weighted Brier excess with fixed per-type weights (1/3 each),
//! * raw and development-temperature-scaled NLL / Brier (temperature fitted on
//!   the dev split only, by a declared grid search),
//! * the ranked probability score (RPS) on ordinal `Score` questions,
//! * decision regret and the finite-action `2·TV` bound, from the declared
//!   utility tables,
//! * top-label reliability (10 fixed bins) with ECE/MCE, and
//! * coverage vs abstention for the declared rule "abstain when `max p < τ`",
//!   including a risk-coverage curve and a `τ` selected on the **calibration**
//!   split, and
//! * reliability and coverage broken down by question type.
//!
//! Parsing is deliberately minimal (the emitter's field order is fixed); any
//! malformed line is an error rather than a silently skipped row.

use pcnl::edge0_world::{QuestionKind, UtilityTable};
use pcnl::r9_report::{self, CalibBin};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct Pred {
    seed: usize,
    arm: String,
    split: String,
    qid: u64,
    world: u64,
    kind: String,
    label: usize,
    truth: Vec<f64>,
    probs: Vec<f64>,
}

fn after<'a>(line: &'a str, key: &str) -> &'a str {
    let tag = format!("\"{key}\":");
    let at = line
        .find(&tag)
        .unwrap_or_else(|| panic!("missing field {key} in {line}"))
        + tag.len();
    &line[at..]
}

fn quoted(line: &str, key: &str) -> String {
    let s = after(line, key);
    let start = s.find('"').expect("quoted value") + 1;
    let end = s[start..].find('"').expect("closing quote") + start;
    s[start..end].to_string()
}

fn num<T: std::str::FromStr>(line: &str, key: &str) -> T {
    let s = after(line, key);
    let end = s.find([',', '}']).expect("terminated number");
    s[..end].parse().unwrap_or_else(|_| panic!("bad {key}"))
}

fn arr(line: &str, key: &str) -> Vec<f64> {
    let s = after(line, key);
    let start = s.find('[').expect("[") + 1;
    let end = s[start..].find(']').expect("]") + start;
    s[start..end]
        .split(',')
        .map(|x| x.trim().parse::<f64>().expect("array float"))
        .collect()
}

fn load(path: &str) -> Vec<Pred> {
    let text = std::fs::read_to_string(path).expect("read predictions");
    // The emitter wraps long records across lines, so records are delimited by
    // their kind tag rather than by newlines.
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = Vec::new();
    for chunk in flat.split("{\"kind\":\"r9-pred\"").skip(1) {
        let record = format!("{{\"kind\":\"r9-pred\"{chunk}");
        let line: &str = &record;
        out.push(Pred {
            seed: num(line, "seed"),
            arm: quoted(line, "arm"),
            split: quoted(line, "split"),
            qid: num(line, "qid"),
            world: num(line, "world"),
            kind: quoted(line, "qkind"),
            label: num(line, "label"),
            truth: arr(line, "truth"),
            probs: arr(line, "probs"),
        });
    }
    assert!(!out.is_empty(), "no r9-pred records in {path}");
    out
}

/// Rebuild the question kind the utility table was declared for.
fn kind_of(p: &Pred) -> QuestionKind {
    match p.kind.as_str() {
        "noul" => QuestionKind::Noul,
        "choice" => QuestionKind::Choice {
            options: p.truth.len(),
        },
        "score" => QuestionKind::Score {
            levels: p.truth.len(),
        },
        other => panic!("unknown question kind {other}"),
    }
}

fn brier_excess(p: &[f64], t: &[f64]) -> f64 {
    p.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum()
}

fn nll(p: &[f64], y: usize) -> f64 {
    -p[y].max(1e-12).ln()
}

/// Ranked probability score, normalised by `J − 1` so its scale is `[0,1]`.
fn rps(p: &[f64], t: &[f64]) -> f64 {
    let j = p.len();
    if j < 2 {
        return 0.0;
    }
    let mut cp = 0.0;
    let mut ct = 0.0;
    let mut acc = 0.0;
    for i in 0..j - 1 {
        cp += p[i];
        ct += t[i];
        acc += (cp - ct) * (cp - ct);
    }
    acc / (j - 1) as f64
}

/// Temperature on the marginal's log-probabilities: `softmax(log p / T)`.
fn at_temperature(p: &[f64], t: f64) -> Vec<f64> {
    let logs: Vec<f64> = p.iter().map(|x| x.max(1e-300).ln() / t).collect();
    let m = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logs.iter().map(|l| (l - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// Grid for the printed risk-coverage curve (0.00–0.95, step 0.05).
const RISK_GRID: usize = 19;
/// Declared risk cap for the coverage-target threshold rule.
const RISK_TARGET: f64 = 0.55;

/// Confidence/correctness pairs for the library's reliability arithmetic.
fn pairs(rows: &[&Pred]) -> (Vec<f64>, Vec<bool>) {
    rows.iter()
        .map(|p| {
            let (top, conf) = top_label(p);
            (conf, top == p.label)
        })
        .unzip()
}

/// Top-label confidence and whether it is correct.
fn top_label(p: &Pred) -> (usize, f64) {
    r9_report::top_label(&p.probs)
}

fn reliability(rows: &[&Pred]) -> (Vec<CalibBin>, f64, f64) {
    let (conf, correct) = pairs(rows);
    r9_report::reliability(&conf, &correct)
}

fn selective(rows: &[&Pred], tau: f64) -> (f64, f64) {
    let (conf, correct) = pairs(rows);
    r9_report::selective(&conf, &correct, tau)
}

fn risk_coverage(rows: &[&Pred], tau: f64) -> (f64, f64) {
    let (conf, correct) = pairs(rows);
    r9_report::risk_coverage(&conf, &correct, tau)
}

/// The declared rule: highest selective accuracy on the split passed in (the
/// calibration split, or dev when the artifact predates the calibration
/// export), ties to the lower `τ`.
fn choose_tau(rows: &[&Pred]) -> Option<f64> {
    let (conf, correct) = pairs(rows);
    r9_report::choose_tau(&conf, &correct)
}

/// Per-question score used for the world-clustered paired SE.
fn q_brier(p: &Pred) -> f64 {
    brier_excess(&p.probs, &p.truth)
}

fn q_nll(p: &Pred) -> f64 {
    nll(&p.probs, p.label)
}

fn q_regret(p: &Pred) -> f64 {
    let table = UtilityTable::declared(p.qid, kind_of(p));
    table.regret(&p.truth, table.optimal_action(&p.probs))
}

fn q_correct(p: &Pred) -> f64 {
    f64::from(top_label(p).0 == p.label)
}

/// Paired `a − b` mean and its world-clustered SE: questions are paired by
/// `(world, qid)`, each world contributes its mean difference, and the SE is
/// the across-world SD over `sqrt(worlds)`. `None` when nothing pairs.
fn clustered_diff(a: &[&Pred], b: &[&Pred], f: fn(&Pred) -> f64) -> Option<(f64, f64, usize)> {
    let key = |p: &Pred| (p.world, p.qid);
    let rhs: BTreeMap<(u64, u64), f64> = b.iter().map(|p| (key(p), f(p))).collect();
    let mut by_world: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
    for p in a {
        if let Some(other) = rhs.get(&key(p)) {
            let e = by_world.entry(p.world).or_default();
            e.0 += f(p) - other;
            e.1 += 1;
        }
    }
    let per_world: Vec<f64> = by_world.values().map(|(sum, n)| sum / *n as f64).collect();
    let w = per_world.len();
    if w < 2 {
        return None;
    }
    let mean = per_world.iter().sum::<f64>() / w as f64;
    let var = per_world
        .iter()
        .map(|d| (d - mean) * (d - mean))
        .sum::<f64>()
        / (w - 1) as f64;
    Some((mean, (var / w as f64).sqrt(), w))
}

/// Fit `T` on the dev split by the declared grid; ties resolve to the smaller
/// grid value, so the choice is deterministic.
fn fit_temperature(dev: &[&Pred]) -> f64 {
    let grid: Vec<f64> = (0..=60).map(|i| 0.5 + 0.05 * i as f64).collect();
    let mut best = (f64::INFINITY, grid[0]);
    for &t in &grid {
        let mean: f64 = dev
            .iter()
            .map(|p| nll(&at_temperature(&p.probs, t), p.label))
            .sum::<f64>()
            / dev.len().max(1) as f64;
        if mean < best.0 - 1e-12 {
            best = (mean, t);
        }
    }
    best.1
}

/// Fixed per-type weights, declared here rather than fitted.
const TYPE_WEIGHTS: [(&str, f64); 3] = [
    ("noul", 1.0 / 3.0),
    ("choice", 1.0 / 3.0),
    ("score", 1.0 / 3.0),
];

#[derive(Clone, Copy, Debug, Default)]
struct Endpoints {
    weighted_brier: f64,
    raw_nll: f64,
    scaled_brier: f64,
    scaled_nll: f64,
    ordinal_rps: f64,
    regret: f64,
    regret_bound: f64,
    regret_violations: usize,
    questions: usize,
    ece: f64,
    mce: f64,
    tau: f64,
    coverage: f64,
    selective_accuracy: f64,
    risk_tau: f64,
    risk_coverage: f64,
    risk: f64,
}

fn endpoints(rows: &[&Pred], t: f64, tau: Option<f64>, tau_risk: Option<f64>) -> Endpoints {
    let mut e = Endpoints::default();
    let mut by_type: BTreeMap<&str, f64> = BTreeMap::new();
    let mut by_type_n: BTreeMap<&str, usize> = BTreeMap::new();
    for p in rows {
        let be = brier_excess(&p.probs, &p.truth);
        *by_type.entry(p.kind.as_str()).or_default() += be;
        *by_type_n.entry(p.kind.as_str()).or_default() += 1;
        e.raw_nll += nll(&p.probs, p.label);
        let sp = at_temperature(&p.probs, t);
        e.scaled_brier += brier_excess(&sp, &p.truth);
        e.scaled_nll += nll(&sp, p.label);
        if p.kind == "score" {
            e.ordinal_rps += rps(&p.probs, &p.truth);
        }
        let table = UtilityTable::declared(p.qid, kind_of(p));
        let action = table.optimal_action(&p.probs);
        let regret = table.regret(&p.truth, action);
        let bound = UtilityTable::regret_bound(&p.probs, &p.truth);
        e.regret += regret;
        e.regret_bound += bound;
        if regret > bound + 1e-12 {
            e.regret_violations += 1;
        }
        e.questions += 1;
    }
    let n = e.questions.max(1) as f64;
    e.weighted_brier = TYPE_WEIGHTS
        .iter()
        .map(|(kind, w)| {
            w * (*by_type.get(kind).unwrap_or(&0.0) / *by_type_n.get(kind).unwrap_or(&1) as f64)
        })
        .sum();
    e.raw_nll /= n;
    e.scaled_brier /= n;
    e.scaled_nll /= n;
    e.ordinal_rps /= rows.iter().filter(|p| p.kind == "score").count().max(1) as f64;
    e.regret /= n;
    e.regret_bound /= n;
    let (_, ece, mce) = reliability(rows);
    e.ece = ece;
    e.mce = mce;
    if let Some(tau) = tau {
        e.tau = tau;
        let (coverage, accuracy) = selective(rows, tau);
        e.coverage = coverage;
        e.selective_accuracy = accuracy;
    }
    if let Some(tau) = tau_risk {
        e.risk_tau = tau;
        let (coverage, risk) = risk_coverage(rows, tau);
        e.risk_coverage = coverage;
        e.risk = risk;
    }
    e
}

fn t95(df: usize) -> f64 {
    match df {
        0 => f64::INFINITY,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        _ => 1.96,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "target/r9_predictions.ndjson".into());
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "target/r9_secondary.ndjson".into());
    let rows = load(&input);

    let mut seeds: Vec<usize> = rows.iter().map(|p| p.seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    let arms: Vec<String> = {
        let mut a: Vec<String> = rows.iter().map(|p| p.arm.clone()).collect();
        a.sort();
        a.dedup();
        a
    };
    println!("R9 SECONDARY ENDPOINTS (from {input})");
    println!(
        "seeds {:?} | arms {} | questions/seed/arm {}",
        seeds,
        arms.len(),
        rows.len() / (seeds.len() * arms.len()).max(1)
    );
    println!(
        "per-type weights (fixed): noul {:.3}, choice {:.3}, score {:.3}; temperature grid 0.50..3.50 on dev\n",
        TYPE_WEIGHTS[0].1, TYPE_WEIGHTS[1].1, TYPE_WEIGHTS[2].1
    );

    // arm -> seed -> endpoints
    let mut table: BTreeMap<String, BTreeMap<usize, Endpoints>> = BTreeMap::new();
    for &seed in &seeds {
        for arm in &arms {
            let dev: Vec<&Pred> = rows
                .iter()
                .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Dev")
                .collect();
            let test: Vec<&Pred> = rows
                .iter()
                .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Test")
                .collect();
            let cal: Vec<&Pred> = rows
                .iter()
                .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Calibration")
                .collect();
            let picked = if cal.is_empty() { &dev } else { &cal };
            let tau = choose_tau(picked);
            let tau_risk = {
                let (conf, correct) = pairs(picked);
                r9_report::choose_tau_at_risk(&conf, &correct, RISK_TARGET)
            };
            let t = fit_temperature(&dev);
            let e = endpoints(&test, t, tau, tau_risk);
            table.entry(arm.clone()).or_default().insert(seed, e);
        }
    }

    for arm in &arms {
        println!("== {arm} ==");
        println!(
            "seed | wBrier | raw NLL | T | scaled Brier | scaled NLL | RPS | regret | 2TV bound | viol | ECE | MCE | tau | cov | selAcc | tauR | covR | riskR"
        );
        for &seed in &seeds {
            let e = table[arm][&seed];
            let t = {
                let dev: Vec<&Pred> = rows
                    .iter()
                    .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Dev")
                    .collect();
                fit_temperature(&dev)
            };
            println!(
                "{seed:>4} | {:>6.5} | {:>7.5} | {t:.2} | {:>12.5} | {:>10.5} | {:.5} | {:.5} | {:.5} | {} | {:.4} | {:.4} | {:.2} | {:.4} | {:.4} | {:.2} | {:.4} | {:.4}",
                e.weighted_brier,
                e.raw_nll,
                e.scaled_brier,
                e.scaled_nll,
                e.ordinal_rps,
                e.regret,
                e.regret_bound,
                e.regret_violations,
                e.ece,
                e.mce,
                e.tau,
                e.coverage,
                e.selective_accuracy,
                e.risk_tau,
                e.risk_coverage,
                e.risk
            );
        }
        let pooled: Vec<&Pred> = rows
            .iter()
            .filter(|p| &p.arm == arm && p.split == "Test")
            .collect();
        let (bins, ece, mce) = reliability(&pooled);
        println!("reliability (test, all seeds pooled): ECE {ece:.4} | MCE {mce:.4}");
        println!("  bin | mid  | n     | conf  | acc   | gap");
        for (i, b) in bins.iter().enumerate() {
            if b.n == 0 {
                println!("  {i:>3} | empty");
                continue;
            }
            let conf = b.conf / b.n as f64;
            let acc = b.correct / b.n as f64;
            println!(
                "  {i:>3} | {:.3} | {:>5} | {conf:.4} | {acc:.4} | {:+.4}  [{:.2}, {:.2})",
                (b.lo + b.hi) / 2.0,
                b.n,
                acc - conf,
                b.lo,
                b.hi
            );
        }
        println!();
    }

    println!("== threshold rules compared (test, per-seed means) ==");
    println!("arm | accuracy-max tau: cov/selAcc | risk<=0.55 tau: cov/risk");
    for arm in &arms {
        let es: Vec<&Endpoints> = seeds.iter().map(|s| &table[arm][s]).collect();
        let k = es.len().max(1) as f64;
        let cov = es.iter().map(|e| e.coverage).sum::<f64>() / k;
        let acc = es.iter().map(|e| e.selective_accuracy).sum::<f64>() / k;
        let rcov = es.iter().map(|e| e.risk_coverage).sum::<f64>() / k;
        let risk = es.iter().map(|e| e.risk).sum::<f64>() / k;
        println!("{arm} | {cov:.4}/{acc:.4} | {rcov:.4}/{risk:.4}");
    }
    println!();

    println!("== coverage vs abstention (test, all seeds pooled; `cov/selAcc` per arm) ==");
    print!("  tau |");
    for arm in &arms {
        print!(" {:^15} |", arm);
    }
    println!();
    for k in 0..=9 {
        let tau = k as f64 / 10.0;
        print!("  {tau:.1} |");
        for arm in &arms {
            let pooled: Vec<&Pred> = rows
                .iter()
                .filter(|p| &p.arm == arm && p.split == "Test")
                .collect();
            let (coverage, accuracy) = selective(&pooled, tau);
            if coverage == 0.0 {
                print!(" {coverage:.4}/  —   |");
            } else {
                print!(" {coverage:.4}/{accuracy:.4} |");
            }
        }
        println!();
    }
    println!(
        "  tau is selected on the calibration split (falling back to dev only if absent), maximizing selective accuracy on a 0.00–0.90 grid (0.01 steps), ties to lower tau"
    );
    println!();

    let mut summary_rows: Vec<String> = Vec::new();

    println!("== risk-coverage curve (test, pooled over seeds; `coverage/risk` per arm) ==");
    print!("  tau |");
    for arm in &arms {
        print!(" {:^15} |", arm);
    }
    println!();
    for k in 0..=RISK_GRID {
        let tau = 0.05 * k as f64;
        print!("  {tau:.2} |");
        for arm in &arms {
            let pooled: Vec<&Pred> = rows
                .iter()
                .filter(|p| &p.arm == arm && p.split == "Test")
                .collect();
            let (coverage, accuracy) = selective(&pooled, tau);
            if coverage == 0.0 {
                print!(" {coverage:.4}/  —   |");
            } else {
                print!(" {coverage:.4}/{:.4} |", 1.0 - accuracy);
            }
        }
        println!();
    }
    println!();

    println!("== reliability by question type (test, pooled over seeds) ==");
    println!("arm | kind | n | ECE | MCE | mean conf | acc");
    for arm in &arms {
        for kind in ["noul", "choice", "score"] {
            let pooled: Vec<&Pred> = rows
                .iter()
                .filter(|p| &p.arm == arm && p.split == "Test" && p.kind == kind)
                .collect();
            if pooled.is_empty() {
                continue;
            }
            let (bins, ece, mce) = reliability(&pooled);
            let n = pooled.len();
            let mut conf = 0.0;
            let mut acc = 0.0;
            for b in &bins {
                conf += b.conf;
                acc += b.correct;
            }
            println!(
                "{arm} | {kind:<6} | {n} | {ece:.4} | {mce:.4} | {:.4} | {:.4}",
                conf / n as f64,
                acc / n as f64
            );
        }
    }
    println!();

    println!("== coverage by question type at the selected tau (test, pooled over seeds) ==");
    println!("arm | kind | cov | selective acc");
    for arm in &arms {
        for kind in ["noul", "choice", "score"] {
            let mut answered = 0usize;
            let mut correct = 0usize;
            let mut total = 0usize;
            for &seed in &seeds {
                let part: Vec<&Pred> = rows
                    .iter()
                    .filter(|p| {
                        p.seed == seed && &p.arm == arm && p.split == "Test" && p.kind == kind
                    })
                    .collect();
                let cal: Vec<&Pred> = rows
                    .iter()
                    .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Calibration")
                    .collect();
                let dev: Vec<&Pred> = rows
                    .iter()
                    .filter(|p| p.seed == seed && &p.arm == arm && p.split == "Dev")
                    .collect();
                let tau = choose_tau(if cal.is_empty() { &dev } else { &cal }).unwrap_or(0.0);
                for p in &part {
                    let (top, best) = top_label(p);
                    total += 1;
                    if best >= tau {
                        answered += 1;
                        if top == p.label {
                            correct += 1;
                        }
                    }
                }
            }
            if total == 0 {
                continue;
            }
            let cov = answered as f64 / total as f64;
            println!(
                "{arm} | {kind:<6} | {cov:.4} | {:.4}",
                if answered == 0 {
                    f64::NAN
                } else {
                    correct as f64 / answered as f64
                }
            );
        }
    }
    println!();

    for kind in ["noul", "choice", "score"] {
        println!("== risk-coverage, {kind} (test, pooled over seeds; `coverage/risk` per arm) ==");
        print!("  tau |");
        for arm in &arms {
            print!(" {:^15} |", arm);
        }
        println!();
        for k in 0..=RISK_GRID {
            let tau = 0.05 * k as f64;
            print!("  {tau:.2} |");
            for arm in &arms {
                let pooled: Vec<&Pred> = rows
                    .iter()
                    .filter(|p| &p.arm == arm && p.split == "Test" && p.kind == kind)
                    .collect();
                let (coverage, risk) = risk_coverage(&pooled, tau);
                summary_rows.push(format!(
                    "{{\"kind\":\"r9-risk-coverage\",\"arm\":\"{arm}\",\"qkind\":\"{kind}\",\"tau\":{tau:.2},\"coverage\":{coverage:.6},\"risk\":{risk:.6}}}"
                ));
                if coverage == 0.0 {
                    print!(" {coverage:.4}/  —   |");
                } else {
                    print!(" {coverage:.4}/{risk:.4} |");
                }
            }
            println!();
        }
        println!();
    }

    let rlcd = arms
        .iter()
        .find(|a| a.starts_with("RLCD"))
        .expect("RLCD arm");
    let sup = arms
        .iter()
        .find(|a| a.starts_with("supervised"))
        .expect("supervised arm");
    println!("== paired RLCD − supervised (per seed; negative favours RLCD) ==");
    #[allow(clippy::type_complexity)]
    let additive: [(&str, fn(&Pred) -> f64); 4] = [
        ("weighted Brier", q_brier),
        ("raw NLL", q_nll),
        ("decision regret", q_regret),
        ("top-1 accuracy", q_correct),
    ];
    println!("per-question metric | seed | seed-level mean | world-clustered SE | worlds");
    let mut clustered_mean_se: Vec<(&str, f64)> = Vec::new();
    for (name, f) in additive {
        let mut ses = 0.0;
        let mut k = 0usize;
        for &seed in &seeds {
            let a: Vec<&Pred> = rows
                .iter()
                .filter(|p| p.seed == seed && &p.arm == rlcd && p.split == "Test")
                .collect();
            let b: Vec<&Pred> = rows
                .iter()
                .filter(|p| p.seed == seed && &p.arm == sup && p.split == "Test")
                .collect();
            if let Some((mean, se, worlds)) = clustered_diff(&a, &b, f) {
                println!("{name:<18} | {seed:>4} | {mean:+.6} | {se:.6} | {worlds}");
                ses += se * se;
                k += 1;
            }
        }
        if k > 0 {
            clustered_mean_se.push((name, ses.sqrt() / k as f64));
        }
    }
    for (name, se) in &clustered_mean_se {
        println!("clustered SE of the across-seed mean, {name:<18} {se:.6}");
    }
    println!();
    #[allow(clippy::type_complexity)]
    let metrics: [(&str, fn(&Endpoints) -> f64); 11] = [
        ("weighted Brier", |e| e.weighted_brier),
        ("raw NLL", |e| e.raw_nll),
        ("scaled Brier", |e| e.scaled_brier),
        ("scaled NLL", |e| e.scaled_nll),
        ("ordinal RPS", |e| e.ordinal_rps),
        ("decision regret", |e| e.regret),
        ("top-label ECE", |e| e.ece),
        ("coverage at cal tau", |e| e.coverage),
        ("selective accuracy", |e| e.selective_accuracy),
        ("coverage at risk tau", |e| e.risk_coverage),
        ("risk at risk tau", |e| e.risk),
    ];
    for (name, f) in metrics {
        let diffs: Vec<f64> = seeds
            .iter()
            .map(|s| f(&table[rlcd][s]) - f(&table[sup][s]))
            .collect();
        let k = diffs.len() as f64;
        let mean = diffs.iter().sum::<f64>() / k;
        let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (k - 1.0).max(1.0);
        let se = (var / k).sqrt();
        let half = t95(diffs.len().saturating_sub(1)) * se;
        println!(
            "{name:<22} {mean:+.6} +/- {half:.6} (95% t, df={}; SE {se:.6})",
            diffs.len().saturating_sub(1)
        );
        summary_rows.push(format!(
            "{{\"kind\":\"r9-secondary\",\"metric\":\"{name}\",\"rlcd_minus_supervised\":{mean:.8},\"half95\":{half:.8},\"se\":{se:.8},\"seeds\":{}}}",
            diffs.len()
        ));
    }
    let violations: usize = seeds
        .iter()
        .map(|s| table[rlcd][s].regret_violations + table[sup][s].regret_violations)
        .sum();
    println!("regret-bound violations across seeds/arms: {violations}");
    summary_rows.push(format!(
        "{{\"kind\":\"r9-secondary\",\"metric\":\"regret_bound_violations\",\"count\":{violations}}}"
    ));

    if let Some(parent) = std::path::Path::new(&output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&output, summary_rows.join("\n") + "\n").expect("write secondary artifact");
    println!("\nsecondary artifact: {output}");
}
