//! R12 single-step and learned-sequence checks.
//!
//! ```text
//! cargo run --release -p pcnl --example rotation_experiment -- \
//!   --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --mode single
//! cargo run --release -p pcnl --example rotation_experiment -- \
//!   --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --mode sequence
//! ```
//!
//! The fixture is a frozen encoder table; this example never trains the
//! encoder.  The sequence arm generates its labels from a declared oracle
//! rotation head, holds out text rows before building sequences, and compares
//! the learned composed Born law with a classical squared-transition baseline
//! and a single-step-only training ablation.

use candle_core::{Device, Tensor};
use pcnl::rotation_head::{
    BasisId, BornPotential, CounterRng, Event, PureState, RotationHead, RotationSession,
    SessionConfig, TextInput,
};
use pcnl::target::Potential;
use pcnl::text_head::{
    EmbeddingTable, LossKind, MlpHeadPotential, Split, predict_proba_mlp,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug)]
struct Scores {
    acc: f64,
    nll: f64,
    brier: f64,
    ece: f64,
}

fn score(y: &[u32], p: &[Vec<f64>]) -> Scores {
    assert_eq!(y.len(), p.len());
    let mut hits = 0usize;
    let mut nll = 0.0;
    let mut brier = 0.0;
    let mut bins = [[0.0f64; 2]; 10];
    for (row, &truth) in p.iter().zip(y) {
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        if best == truth as usize {
            hits += 1;
        }
        nll -= row[truth as usize].max(1e-300).ln();
        for (i, &v) in row.iter().enumerate() {
            let target = if i == truth as usize { 1.0 } else { 0.0 };
            brier += (v - target) * (v - target);
        }
        let conf = row[best];
        let bin = ((conf * 10.0).floor() as usize).min(9);
        bins[bin][0] += conf;
        bins[bin][1] += if best == truth as usize { 1.0 } else { 0.0 };
    }
    let n = y.len() as f64;
    let ece = bins
        .iter()
        .map(|b| {
            let count = b[1];
            if count == 0.0 {
                0.0
            } else {
                (b[0] - b[1]).abs()
            }
        })
        .sum::<f64>()
        / n;
    Scores {
        acc: 100.0 * hits as f64 / n,
        nll: nll / n,
        brier: brier / n,
        ece,
    }
}

fn temperature(probs: &[Vec<f64>], y: &[u32]) -> f64 {
    let mut best = (f64::INFINITY, 1.0);
    let (lo, hi) = (0.04f64.ln(), 8.0f64.ln());
    for i in 0..240 {
        let t = (lo + i as f64 * (hi - lo) / 239.0).exp();
        let mut nll = 0.0;
        for (row, &truth) in probs.iter().zip(y) {
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = row.iter().map(|&v| ((v.ln() / t) - max / t).exp()).sum();
            let logp = row[truth as usize].max(1e-300).ln() / t - max / t - z.ln();
            nll -= logp;
        }
        if nll < best.0 {
            best = (nll, t);
        }
    }
    best.1
}

fn apply_temperature(probs: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    probs
        .iter()
        .map(|row| {
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = row.iter().map(|&v| ((v.ln() / t) - max / t).exp()).sum();
            row.iter()
                .map(|&v| ((v.ln() / t) - max / t).exp() / z)
                .collect()
        })
        .collect()
}

fn adam_pretrain<P: Potential>(
    potential: &P,
    mut theta: Vec<f32>,
    steps: usize,
    lr: f64,
) -> (Vec<f32>, Vec<f64>) {
    let mut m = vec![0.0f64; theta.len()];
    let mut v = vec![0.0f64; theta.len()];
    let mut trace = Vec::with_capacity(steps);
    for step in 1..=steps {
        let t = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
        let loss = potential.mean_loss(&t);
        trace.push(loss);
        if !loss.is_finite() {
            break;
        }
        let Some(g) = potential.mean_grad(&t) else {
            break;
        };
        let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut norm = 0.0f64;
        for &x in &g {
            norm += f64::from(x) * f64::from(x);
        }
        let scale = if norm.sqrt() > 5.0 {
            5.0 / norm.sqrt()
        } else {
            1.0
        };
        let b1 = 0.9f64;
        let b2 = 0.999f64;
        let c1 = 1.0 - b1.powi(step as i32);
        let c2 = 1.0 - b2.powi(step as i32);
        for i in 0..theta.len() {
            let gi = f64::from(g[i]) * scale;
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            theta[i] -= (lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8)) as f32;
        }
    }
    (theta, trace)
}

fn mixed_table(
    fixture: &EmbeddingTable,
    fit_per_class: usize,
    cal_per_class: usize,
    seed: u64,
) -> (EmbeddingTable, Vec<(Vec<f32>, u32)>) {
    let c = fixture.classes();
    let dim = fixture.dim();
    let x = fixture.x(Split::Train);
    let y = fixture.y(Split::Train);
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); c];
    for (i, &label) in y.iter().enumerate() {
        by_class[label as usize].push(i);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x51e1_c7ed);
    let mut rows = Vec::new();
    let mut cal = Vec::new();
    for class in 0..c {
        by_class[class].shuffle(&mut rng);
        if by_class[class].len() < fit_per_class + cal_per_class {
            panic!(
                "class {class} has {} rows, need {}",
                by_class[class].len(),
                fit_per_class + cal_per_class
            );
        }
        for &i in &by_class[class][..fit_per_class] {
            rows.push((x[i * dim..(i + 1) * dim].to_vec(), y[i], false));
        }
        for &i in &by_class[class][fit_per_class..fit_per_class + cal_per_class] {
            cal.push((x[i * dim..(i + 1) * dim].to_vec(), y[i]));
        }
    }
    let xt = fixture.x(Split::Test);
    let yt = fixture.y(Split::Test);
    for i in 0..yt.len() {
        rows.push((xt[i * dim..(i + 1) * dim].to_vec(), yt[i], true));
    }
    let table = EmbeddingTable::from_rows(dim, c, &rows).expect("mixed table");
    (table, cal)
}

fn single_step(args: &[String], label: &str, fixture: &EmbeddingTable) {
    let per = num(args, "--fit-per-class", 20usize);
    let cal_per = num(args, "--cal-per-class", 10usize);
    let hidden = hidden(args, &[16]);
    let (table, cal) = mixed_table(fixture, per, cal_per, num(args, "--seed", 20261118));
    let dim = table.dim();
    let classes = table.classes();
    let config = SessionConfig {
        sigma: num(args, "--sigma", 0.5),
        lambda: num(args, "--lambda", 1.0),
        beta: num(args, "--beta", 0.08),
        burn: num(args, "--burn", 80usize),
        steps: num(args, "--steps", 320usize),
        thin: num(args, "--thin", 8usize),
        chains: num(args, "--chains", 2usize),
        draws: num(args, "--draws", 20usize),
        seed: num(args, "--seed", 20261118),
    };
    let head = RotationHead::with_name("rotation-head", dim, classes, &hidden).unwrap();
    let mut session = RotationSession::new(head, config).unwrap();
    let mut events = Vec::new();
    for (i, (&y, row)) in table
        .y(Split::Train)
        .iter()
        .zip(table.x(Split::Train).chunks(dim))
        .enumerate()
    {
        let id = format!("fit:{i}");
        session.append_event(id.clone(), vec![row.to_vec()], y).unwrap();
        events.push(Event {
            id,
            steps: vec![row.to_vec()],
            observed: y,
        });
    }
    let potential = BornPotential::new(session.head().clone(), events).unwrap();
    let theta0 = session.prior_theta(num(args, "--seed", 20261118));
    let (start, trace) = adam_pretrain(
        &potential,
        theta0,
        num(args, "--pretrain", 120usize),
        num(args, "--pretrain-lr", 0.01),
    );
    session.warm_start(&start).unwrap();
    let report = session.update().unwrap();
    println!(
        "## {label} single-step: {classes} classes, {dim}-d, fit={} cal={} test={}",
        table.y(Split::Train).len(),
        cal.len(),
        table.y(Split::Test).len()
    );
    println!(
        "pretrain_steps={} pretrain_last={:?} proposals={} accepted={} retained={}",
        trace.len(),
        trace.last(),
        report.proposals,
        report.accepted,
        report.retained_draws
    );

    let test_x = table.x(Split::Test);
    let test_y = table.y(Split::Test).to_vec();
    let mut raw = Vec::new();
    for row in test_x.chunks(dim) {
        let input = TextInput::new("fixture-test", row, dim).unwrap();
        raw.push(session.predict(&input).unwrap().probabilities);
    }
    let cal_inputs: Vec<TextInput> = cal
        .iter()
        .map(|(row, _)| TextInput::new("fixture-cal", row, dim).unwrap())
        .collect();
    let cal_probs: Vec<Vec<f64>> = cal_inputs
        .iter()
        .map(|input| session.predict(input).unwrap().probabilities)
        .collect();
    let cal_y: Vec<u32> = cal.iter().map(|(_, y)| *y).collect();
    let raw_s = score(&test_y, &raw);
    println!("rotation raw      acc={:.2} nll={:.4} brier={:.4} ece={:.4}", raw_s.acc, raw_s.nll, raw_s.brier, raw_s.ece);
    if !cal.is_empty() {
        let t = temperature(&cal_probs, &cal_y);
        let scaled = apply_temperature(&raw, t);
        let scaled_s = score(&test_y, &scaled);
        println!("rotation T={t:.3}  acc={:.2} nll={:.4} brier={:.4} ece={:.4}", scaled_s.acc, scaled_s.nll, scaled_s.brier, scaled_s.ece);
    }

    // Positive-square-root softmax control: its measured one-step Born
    // probabilities are the softmax probabilities by construction.
    let soft = MlpHeadPotential::new_with_loss(&table, &hidden, LossKind::Nll).unwrap();
    let mut rng = CounterRng::new(num(args, "--seed", 20261118) ^ 0x50f7);
    let theta0 = (0..soft.flat_size())
        .map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal) as f32 * 0.02)
        .collect::<Vec<_>>();
    let (theta, soft_trace) = adam_pretrain(
        &soft,
        theta0,
        num(args, "--softmax-pretrain", 200usize),
        0.01,
    );
    println!(
        "softmax_pretrain_steps={} last={:?}",
        soft_trace.len(),
        soft_trace.last()
    );
    let probs = predict_proba_mlp(&table, &theta, &hidden, Split::Test).unwrap();
    let probs: Vec<Vec<f64>> = probs
        .chunks(classes)
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect();
    let s = score(&test_y, &probs);
    println!("softmax control   acc={:.2} nll={:.4} brier={:.4} ece={:.4}", s.acc, s.nll, s.brier, s.ece);
}

fn oracle_state(head: &RotationHead, theta: &[f32], steps: &[Vec<f32>]) -> PureState {
    let mut state = PureState::new(
        {
            let mut a = vec![0.0; head.classes()];
            a[0] = 1.0;
            a
        },
        BasisId::head(head.name(), head.classes()),
        0,
        0,
    )
    .unwrap();
    for step in steps {
        let input = TextInput::new("oracle-step", step, head.dim()).unwrap();
        state = head.apply(theta, &input, &state, 0).unwrap();
    }
    state
}

fn draw_label(p: &[f64], rng: &mut impl Rng) -> u32 {
    let u: f64 = rng.random();
    let mut acc = 0.0;
    for (i, &v) in p.iter().enumerate() {
        acc += v;
        if u <= acc {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}

fn classical(head: &RotationHead, theta: &[f32], steps: &[Vec<f32>]) -> Vec<f64> {
    let k = head.classes();
    let mut p = vec![0.0; k];
    p[0] = 1.0;
    for step in steps {
        let u = head.operator(theta, step).unwrap().to_vec2::<f32>().unwrap();
        let mut next = vec![0.0; k];
        for i in 0..k {
            for j in 0..k {
                next[i] += f64::from(u[i][j]).powi(2) * p[j];
            }
        }
        p = next;
    }
    p
}

fn sequence(args: &[String], fixture: &EmbeddingTable) {
    let dim = fixture.dim();
    let classes = fixture.classes();
    let hidden = hidden(args, &[8]);
    let n_train = num(args, "--seq-train", 120usize);
    let n_test = num(args, "--seq-test", 48usize);
    let len = num(args, "--seq-len", 2usize);
    let seed = num(args, "--seed", 20261118u64);
    let mut rng = CounterRng::new(seed ^ 0x5e9);
    let oracle = RotationHead::new(dim, classes, &hidden).unwrap();
    let oracle_theta: Vec<f32> = (0..oracle.flat_size())
        .map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal) as f32 * 0.8)
        .collect();
    let train_x = fixture.x(Split::Train);
    let test_x = fixture.x(Split::Test);
    let make = |pool: &[f32], n: usize, rng: &mut CounterRng| {
        let rows = pool.len() / dim;
        (0..n)
            .map(|_| {
                let steps: Vec<Vec<f32>> = (0..len)
                    .map(|_| {
                        let r = rng.random_range(0..rows);
                        pool[r * dim..(r + 1) * dim].to_vec()
                    })
                    .collect();
                let state = oracle_state(&oracle, &oracle_theta, &steps);
                (steps, draw_label(&state.measure(), rng))
            })
            .collect::<Vec<_>>()
    };
    let train = make(train_x, n_train, &mut rng);
    let test = make(test_x, n_test, &mut rng);

    let config = SessionConfig {
        sigma: num(args, "--sigma", 0.6),
        lambda: num(args, "--lambda", 1.0),
        beta: num(args, "--beta", 0.08),
        burn: num(args, "--burn", 60usize),
        steps: num(args, "--steps", 220usize),
        thin: num(args, "--thin", 10usize),
        chains: num(args, "--chains", 2usize),
        draws: num(args, "--draws", 12usize),
        seed,
    };
    let fit = |events: &[(Vec<Vec<f32>>, u32)], tag: &str| {
        let head = RotationHead::with_name(tag, dim, classes, &hidden).unwrap();
        let mut session = RotationSession::new(head, config.clone()).unwrap();
        for (i, (steps, y)) in events.iter().enumerate() {
            session
                .append_event(format!("{tag}:{i}"), steps.clone(), *y)
                .unwrap();
        }
        let potential =
            BornPotential::new(session.head().clone(), session.events().to_vec()).unwrap();
        let (start, trace) = adam_pretrain(
            &potential,
            session.prior_theta(seed ^ 0x9e37),
            num(args, "--pretrain", 600usize),
            num(args, "--pretrain-lr", 0.01),
        );
        println!("{tag}: pretrain_steps={} last={:?}", trace.len(), trace.last());
        session.warm_start(&start).unwrap();
        let report = session.update().unwrap();
        println!(
            "{tag}: accepted={} proposals={} retained={}",
            report.accepted, report.proposals, report.retained_draws
        );
        session
    };
    let mut student = fit(&train, "student-sequence");
    let mut ablation_events = Vec::new();
    for (steps, y) in &train {
        ablation_events.push((vec![steps[0].clone()], *y));
    }
    let mut ablation = fit(&ablation_events, "student-single-step");

    fn inputs_for(steps: &[Vec<f32>], dim: usize) -> Vec<TextInput<'_>> {
        steps
            .iter()
            .map(|s| TextInput::new("held-out-step", s, dim).unwrap())
            .collect()
    }
    let test_y: Vec<u32> = test.iter().map(|(_, y)| *y).collect();
    let oracle_probs: Vec<Vec<f64>> = test
        .iter()
        .map(|(steps, _)| oracle_state(&oracle, &oracle_theta, steps).measure())
        .collect();
    let mut student_probs = Vec::new();
    let mut ablate_probs = Vec::new();
    let mut classical_probs = Vec::new();
    for (steps, _) in &test {
        student_probs.push(
            student
                .predict_sequence(&inputs_for(steps, dim))
                .unwrap()
                .probabilities,
        );
        ablate_probs.push(
            ablation
                .predict_sequence(&inputs_for(steps, dim))
                .unwrap()
                .probabilities,
        );
        let mut avg = vec![0.0; classes];
        for theta in ablation.retained_draws() {
            for (i, v) in classical(ablation.head(), theta, steps).iter().enumerate() {
                avg[i] += v;
            }
        }
        let n = ablation.retained_draws().len() as f64;
        for v in &mut avg {
            *v /= n;
        }
        classical_probs.push(avg);
    }
    let l1 = |p: &[Vec<f64>], q: &[Vec<f64>]| {
        p.iter()
            .zip(q)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>())
            .sum::<f64>()
            / p.len() as f64
    };
    println!(
        "## learned sequences: len={len} train={n_train} test={n_test} classes={classes}"
    );
    for (name, p) in [
        ("oracle", &oracle_probs),
        ("learned composed", &student_probs),
        ("single-step ablation", &ablate_probs),
        ("classical squared transition", &classical_probs),
    ] {
        let s = score(&test_y, p);
        println!(
            "{name:28} acc={:.2} nll={:.4} brier={:.4} ece={:.4} l1_to_oracle={:.4}",
            s.acc,
            s.nll,
            s.brier,
            s.ece,
            l1(p, &oracle_probs)
        );
    }
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn hidden(args: &[String], default: &[usize]) -> Vec<usize> {
    args.iter()
        .position(|a| a == "--hidden")
        .and_then(|i| args.get(i + 1))
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture_path = args
        .iter()
        .position(|a| a == "--fixture")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/fixtures/ultrabert_agnews_embeddings.safetensors"));
    let fixture = EmbeddingTable::load(&fixture_path).expect("fixture");
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("all");
    let label = fixture_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fixture");
    if mode == "single" || mode == "all" {
        single_step(&args, label, &fixture);
    }
    if mode == "sequence" || mode == "all" {
        sequence(&args, &fixture);
    }
}
