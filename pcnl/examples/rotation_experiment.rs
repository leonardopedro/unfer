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
//! and a single-step-only training ablation.  Pass `--anchor-prior` to centre
//! the Gaussian prior on the Adam-pretrained head instead of zero.  Pass
//! `--representation standardized` to z-score each embedding coordinate using
//! fit rows only, or `--representation centered` to subtract the fit mean
//! without rescaling; `raw` is the default.  Every single-step run also prints
//! its validation scores on the held-out calibration rows, so a width or
//! representation sweep can select on validation NLL without touching test.
//! Pass `--softmax-tune` to select softmax weight decay and early stopping on
//! those calibration rows; otherwise the fixed-3000-step control is unchanged.
//! Pass `--uniform-prior` to centre the angle prior on a class-balanced signed
//! state instead of the zero-angle identity, which puts all prior mass on the
//! first outcome.  Pass `--rotation-tune` to early-stop the rotation Adam
//! pretrain on calibration Born NLL, mirroring the tuned softmax protocol.
//! Pass `--chain-spread <s>` to start each chain at the anchor plus an
//! independent Gaussian draw of scale `s` in normalized head coordinates;
//! the default `0` starts every chain at the same anchor.

use candle_core::{Device, Tensor};
use pcnl::diag;
use pcnl::rotation_head::{
    BasisId, BornPotential, CounterRng, Event, PureState, RotationHead, RotationSession, Sampler,
    SessionConfig, TextInput,
};
use pcnl::target::Potential;
use pcnl::text_head::{EmbeddingTable, LossKind, MlpHeadPotential, Split, predict_proba_mlp};
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
    theta: Vec<f32>,
    steps: usize,
    lr: f64,
) -> (Vec<f32>, Vec<f64>) {
    adam_pretrain_with_decay(potential, theta, steps, lr, 0.0)
}

/// Adam with an optional L2 penalty added to the gradient.  `weight_decay = 0`
/// is byte-for-byte the same update as [`adam_pretrain`].
fn adam_pretrain_with_decay<P: Potential>(
    potential: &P,
    mut theta: Vec<f32>,
    steps: usize,
    lr: f64,
    weight_decay: f64,
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
            let gi = f64::from(g[i]) * scale + weight_decay * f64::from(theta[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            theta[i] -= (lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8)) as f32;
        }
    }
    (theta, trace)
}

/// Mean NLL of a softmax head on a one-split calibration table.
fn val_nll_mlp(
    table: &EmbeddingTable,
    theta: &[f32],
    hidden: &[usize],
    classes: usize,
) -> Option<f64> {
    let probs = predict_proba_mlp(table, theta, hidden, Split::Test).ok()?;
    let y = table.y(Split::Test);
    if probs.len() != y.len() * classes {
        return None;
    }
    let mut nll = 0.0f64;
    for (row, &truth) in probs.chunks(classes).zip(y) {
        nll -= f64::from(row[truth as usize]).max(1e-300).ln();
    }
    Some(nll / y.len() as f64)
}

/// Train one softmax candidate, keeping the parameter vector with the best
/// calibration NLL.  Early stopping is evaluated only on the calibration rows;
/// the test split is never touched here.
#[allow(clippy::too_many_arguments)]
fn train_softmax_candidate(
    potential: &MlpHeadPotential,
    theta0: Vec<f32>,
    steps: usize,
    lr: f64,
    weight_decay: f64,
    eval_every: usize,
    patience: usize,
    val: Option<&EmbeddingTable>,
    hidden: &[usize],
    classes: usize,
) -> (Vec<f32>, usize, Option<f64>) {
    let mut theta = theta0;
    let mut m = vec![0.0f64; theta.len()];
    let mut v = vec![0.0f64; theta.len()];
    let mut best: Option<(f64, Vec<f32>, usize)> = None;
    let mut stale = 0usize;
    for step in 1..=steps {
        let t = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
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
            let gi = f64::from(g[i]) * scale + weight_decay * f64::from(theta[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            theta[i] -= (lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8)) as f32;
        }
        if let Some(val) = val
            && (step % eval_every == 0 || step == steps)
            && let Some(nll) = val_nll_mlp(val, &theta, hidden, classes)
        {
            if best.as_ref().is_none_or(|b| nll < b.0 - 1e-9) {
                best = Some((nll, theta.clone(), step));
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
    }
    match best {
        Some((nll, theta, step)) => (theta, step, Some(nll)),
        None => (theta, steps, None),
    }
}

/// Adam pretraining with calibration early stopping.  The returned loss is the
/// mean Born NLL (`-log |ψ_y|`), i.e. half of `BornPotential::mean_loss`, so it
/// is comparable to the score table's NLL column.  Test rows are never read.
fn adam_pretrain_early_stop<P: Potential, V: Potential>(
    potential: &P,
    val: &V,
    mut theta: Vec<f32>,
    steps: usize,
    lr: f64,
    eval_every: usize,
    patience: usize,
) -> (Vec<f32>, usize, Option<f64>) {
    let mut m = vec![0.0f64; theta.len()];
    let mut v = vec![0.0f64; theta.len()];
    let mut best: Option<(f64, Vec<f32>, usize)> = None;
    let mut stale = 0usize;
    for step in 1..=steps {
        let t = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
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
        if step % eval_every == 0 || step == steps {
            let t = Tensor::from_vec(theta.clone(), theta.len(), &Device::Cpu).unwrap();
            let loss = val.mean_loss(&t);
            if loss.is_finite() {
                if best.as_ref().is_none_or(|b| loss < b.0 - 1e-9) {
                    best = Some((loss, theta.clone(), step));
                    stale = 0;
                } else {
                    stale += 1;
                    if stale >= patience {
                        break;
                    }
                }
            }
        }
    }
    match best {
        Some((loss, theta, step)) => (theta, step, Some(loss / 2.0)),
        None => (theta, steps, None),
    }
}

fn number_list(args: &[String], name: &str, default: &str) -> Vec<f64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or(default)
        .split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .collect()
}

fn mixed_table(
    fixture: &EmbeddingTable,
    fit_per_class: usize,
    cal_per_class: usize,
    seed: u64,
) -> (EmbeddingTable, Vec<(Vec<f32>, u32)>) {
    // The declared split is fit rows plus the full test split; calibration
    // rows stay out of the table so they cannot enter the likelihood.

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
    for (class, class_rows) in by_class.iter_mut().enumerate() {
        class_rows.shuffle(&mut rng);
        if class_rows.len() < fit_per_class + cal_per_class {
            panic!(
                "class {class} has {} rows, need {}",
                class_rows.len(),
                fit_per_class + cal_per_class
            );
        }
        for &i in &class_rows[..fit_per_class] {
            rows.push((x[i * dim..(i + 1) * dim].to_vec(), y[i], false));
        }
        for &i in &class_rows[fit_per_class..fit_per_class + cal_per_class] {
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

/// Fit-set recentering, optionally followed by variance scaling.  The mean and
/// standard deviation come only from the *training* rows; validation and test
/// rows use those fixed statistics, so the representation adds no label or test
/// information to the likelihood.
fn recenter(
    table: &EmbeddingTable,
    cal: &[(Vec<f32>, u32)],
    scale_variance: bool,
) -> (EmbeddingTable, Vec<(Vec<f32>, u32)>) {
    let dim = table.dim();
    let classes = table.classes();
    let train = table.x(Split::Train);
    let n = table.n_train();
    let mut mean = vec![0.0f64; dim];
    for row in train.chunks(dim) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += f64::from(v);
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut var = vec![0.0f64; dim];
    for row in train.chunks(dim) {
        for (v, (&x, &m)) in var.iter_mut().zip(row.iter().zip(&mean)) {
            let d = f64::from(x) - m;
            *v += d * d;
        }
    }
    for v in &mut var {
        *v /= n as f64;
    }
    let scale: Vec<f64> = if scale_variance {
        var.iter().map(|v| v.sqrt().max(1e-6)).collect()
    } else {
        vec![1.0; dim]
    };
    let transform = |row: &[f32]| -> Vec<f32> {
        row.iter()
            .zip(mean.iter().zip(&scale))
            .map(|(&x, (&m, &s))| ((f64::from(x) - m) / s) as f32)
            .collect()
    };
    let mut rows = Vec::with_capacity(table.n_train() + table.n_test());
    for (row, &y) in table.x(Split::Train).chunks(dim).zip(table.y(Split::Train)) {
        rows.push((transform(row), y, false));
    }
    for (row, &y) in table.x(Split::Test).chunks(dim).zip(table.y(Split::Test)) {
        rows.push((transform(row), y, true));
    }
    let table = EmbeddingTable::from_rows(dim, classes, &rows).expect("recentered table");
    let cal = cal.iter().map(|(row, y)| (transform(row), *y)).collect();
    (table, cal)
}

fn center_only(
    table: &EmbeddingTable,
    cal: &[(Vec<f32>, u32)],
) -> (EmbeddingTable, Vec<(Vec<f32>, u32)>) {
    recenter(table, cal, false)
}

fn standardize(
    table: &EmbeddingTable,
    cal: &[(Vec<f32>, u32)],
) -> (EmbeddingTable, Vec<(Vec<f32>, u32)>) {
    recenter(table, cal, true)
}

/// Flat head parameters whose output is the class-balanced state
/// `(1/√K, …, 1/√K)` for every input.
///
/// The canonical layout applies `(0,1), (0,2), …` first.  Starting from the
/// first basis vector, choosing `sin θ_j = 1/√(K−j+1)` for the first `K−1`
/// angles spreads the amplitude equally; the remaining angles stay zero.  All
/// weights are zero, so the final bias alone sets the state.  This is a null
/// model with no class-0 preference, unlike the zero-angle identity.
fn uniform_prior_theta(head: &RotationHead) -> Vec<f32> {
    let classes = head.classes();
    let angles = head.layout().angles();
    let mut theta = vec![0.0f32; head.flat_size()];
    let bias = theta.len() - angles;
    for j in 0..classes - 1 {
        theta[bias + j] = (1.0 / ((classes - j) as f32).sqrt()).asin();
    }
    let zero = vec![0.0f32; head.dim()];
    let input = TextInput::new("uniform-prior", &zero, head.dim()).unwrap();
    let state = head.prepare(&theta, &input, 0, 0).unwrap();
    let want = 1.0 / classes as f64;
    for p in state.measure() {
        assert!(
            (p - want).abs() < 1e-5,
            "uniform prior state is {p}, expected {want}"
        );
    }
    println!("uniform_prior=class-balanced angles={angles}");
    theta
}

/// A table with one declared split.  `from_rows` requires both splits, so the
/// first row is repeated on the unused side; predictions are requested only on
/// the declared side.
fn one_split_table(
    rows: &[(Vec<f32>, u32)],
    test: bool,
    dim: usize,
    classes: usize,
) -> EmbeddingTable {
    assert!(!rows.is_empty());
    let mut all: Vec<(Vec<f32>, u32, bool)> = rows
        .iter()
        .map(|(row, y)| (row.clone(), *y, test))
        .collect();
    all.push((rows[0].0.clone(), rows[0].1, !test));
    EmbeddingTable::from_rows(dim, classes, &all).expect("one-split table")
}

fn single_step(args: &[String], label: &str, fixture: &EmbeddingTable) {
    let per = num(args, "--fit-per-class", 20usize);
    let cal_per = num(args, "--cal-per-class", 10usize);
    let seed = num(args, "--seed", 20261118);
    // A linear angle map removes the hidden-unit sign and permutation gauges;
    // it is the reduced identifiable parameterization used for the R12 check.
    let hidden = if has_flag(args, "--linear-head") {
        Vec::new()
    } else {
        hidden(args, &[16])
    };
    let representation = arg_str(args, "--representation", "raw");
    let (table, cal) = mixed_table(fixture, per, cal_per, seed);
    let (table, cal) = match representation {
        "raw" => (table, cal),
        "centered" => center_only(&table, &cal),
        "standardized" => standardize(&table, &cal),
        other => panic!("unknown --representation `{other}` (raw|centered|standardized)"),
    };
    let dim = table.dim();
    let classes = table.classes();
    let sampler = match arg_str(args, "--sampler", "pcn") {
        "pcn" => Sampler::Pcn,
        "mala" => Sampler::Mala,
        "pt" | "parallel-tempering" => Sampler::ParallelTempering,
        other => panic!("unknown --sampler `{other}` (pcn|mala|pt)"),
    };
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
        prior_mean: Vec::new(),
        proposal_betas: Vec::new(),
        sampler,
        mala_step: num(args, "--mala-step", 0.01),
        temperatures: number_list(args, "--temperatures", ""),
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
        session
            .append_event(id.clone(), vec![row.to_vec()], y)
            .unwrap();
        events.push(Event {
            id,
            steps: vec![row.to_vec()],
            observed: y,
        });
    }
    let potential = BornPotential::new(session.head().clone(), events).unwrap();
    let uniform = has_flag(args, "--uniform-prior");
    let anchor = has_flag(args, "--anchor-prior");
    if uniform && anchor {
        panic!("--uniform-prior and --anchor-prior are different prior centres");
    }
    if uniform {
        let center = uniform_prior_theta(session.head());
        session.set_prior_mean(&center).unwrap();
    }
    let theta0 = session.prior_theta(num(args, "--seed", 20261118));
    let pretrain_steps = num(args, "--pretrain", 120usize);
    let pretrain_lr = num(args, "--pretrain-lr", 0.01);
    let tune_rotation = has_flag(args, "--rotation-tune");
    let val_potential = if tune_rotation && !cal.is_empty() {
        let val_events = cal
            .iter()
            .enumerate()
            .map(|(i, (row, y))| Event {
                id: format!("cal:{i}"),
                steps: vec![row.clone()],
                observed: *y,
            })
            .collect();
        Some(BornPotential::new(session.head().clone(), val_events).unwrap())
    } else {
        None
    };
    let (start, trace, pretrain_step, rotation_val) = match val_potential.as_ref() {
        Some(val) => {
            let (theta, step, val_nll) = adam_pretrain_early_stop(
                &potential,
                val,
                theta0,
                pretrain_steps,
                pretrain_lr,
                num(args, "--rotation-eval-every", 100usize),
                num(args, "--rotation-patience", 4usize),
            );
            (theta, Vec::new(), step, val_nll)
        }
        None => {
            let (theta, trace) = adam_pretrain(&potential, theta0, pretrain_steps, pretrain_lr);
            let step = trace.len();
            (theta, trace, step, None)
        }
    };
    if anchor {
        let sigma = session.set_anchor_prior(&start).unwrap();
        println!("anchor_prior_sigma={sigma:.6}");
    }
    let chain_spread = num(args, "--chain-spread", 0.0);
    if chain_spread > 0.0 {
        session.warm_start_spread(&start, chain_spread).unwrap();
    } else {
        session.warm_start(&start).unwrap();
    }
    let pilot_steps = num(args, "--pilot-steps", 0usize);
    let pilot_beta = num(args, "--pilot-beta", 0.01);
    let pilot_draws = num(args, "--pilot-draws", 200usize);
    let pilot_fraction = num(args, "--pilot-fraction", 0.5);
    if pilot_steps > 0 {
        let betas = session
            .pilot_betas(pilot_steps, pilot_beta, pilot_draws, pilot_fraction)
            .unwrap();
        let mut sorted = betas.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "pilot_betas steps={pilot_steps} beta={pilot_beta} draws={pilot_draws} fraction={pilot_fraction} min={:.6} median={:.6} max={:.6}",
            sorted.first().copied().unwrap_or(f64::NAN),
            sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN),
            sorted.last().copied().unwrap_or(f64::NAN)
        );
        session.set_proposal_betas(&betas).unwrap();
    }
    let prior = if uniform {
        "uniform"
    } else if anchor {
        "anchor"
    } else {
        "zero"
    };
    let report = session.update().unwrap();
    println!(
        "## {label} single-step: seed={seed} {classes} classes, {dim}-d, hidden={hidden:?}, representation={representation}, prior={prior}, sampler={sampler:?}, mala_step={}, chain_spread={chain_spread}, fit={} cal={} test={}",
        session.config().mala_step,
        table.y(Split::Train).len(),
        cal.len(),
        table.y(Split::Test).len()
    );
    println!(
        "pretrain_trace={} pretrain_step={} pretrain_last={:?} rotation_val_nll={:?} proposals={} accepted={} retained={} swap_attempts={} swap_accepted={}",
        trace.len(),
        pretrain_step,
        trace.last(),
        rotation_val,
        report.proposals,
        report.accepted,
        report.retained_draws,
        report.swap_attempts,
        report.swap_accepted
    );

    let test_x = table.x(Split::Test);
    let test_y = table.y(Split::Test).to_vec();
    let mut raw = Vec::new();
    let mut draw_nll: Vec<f64> = Vec::new();
    let mut draw_probs: Vec<Vec<Vec<f64>>> = Vec::new();
    for (row, &y) in test_x.chunks(dim).zip(&test_y) {
        let input = TextInput::new("fixture-test", row, dim).unwrap();
        let prediction = session.predict(&input).unwrap();
        if draw_nll.is_empty() {
            draw_nll = vec![0.0; prediction.states.len()];
        }
        assert_eq!(draw_nll.len(), prediction.states.len());
        let mut row_probs = Vec::with_capacity(prediction.states.len());
        for (d, state) in prediction.states.iter().enumerate() {
            let p = state.amplitudes()[y as usize].powi(2);
            draw_nll[d] += -p.ln();
            row_probs.push(
                state
                    .amplitudes()
                    .iter()
                    .map(|a| (*a).powi(2))
                    .collect::<Vec<_>>(),
            );
        }
        draw_probs.push(row_probs);
        raw.push(prediction.probabilities);
    }
    let n_test = test_y.len() as f64;
    for value in &mut draw_nll {
        *value /= n_test;
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
    println!(
        "rotation raw      acc={:.2} nll={:.4} brier={:.4} ece={:.4}",
        raw_s.acc, raw_s.nll, raw_s.brier, raw_s.ece
    );
    if !draw_nll.is_empty() {
        let n = draw_nll.len() as f64;
        let mean = draw_nll.iter().sum::<f64>() / n;
        let sd =
            (draw_nll.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
        let prov = session.retained_provenance();
        let chain_count = session.retained_chain_count();
        let mut counts = vec![0usize; chain_count];
        let mut per_chain: Vec<Vec<f64>> = vec![Vec::new(); chain_count];
        for (d, &(chain, _step)) in prov.iter().enumerate() {
            if chain < chain_count {
                counts[chain] += 1;
                per_chain[chain].push(draw_nll[d]);
            }
        }
        let rhat = diag::gelman_rubin(&per_chain);
        println!(
            "draw_nll mean={mean:.4} sd={sd:.4} mcse={:.4} rhat={rhat:.4} chain_counts={counts:?}",
            sd / n.sqrt()
        );

        // The draw-level NLL is a scalar functional of the whole head.  A high
        // R-hat there can come from a few test rows or from a genuinely
        // unconverged parameter posterior, so report the parameter-wise
        // statistic as well.
        let retained = session.retained_draws();
        let flat = session.head().flat_size();
        let mut param_rhats = Vec::with_capacity(flat);
        let mut param_rank_rhats = Vec::with_capacity(flat);
        for i in 0..flat {
            let mut per_chain_param: Vec<Vec<f64>> = vec![Vec::new(); chain_count];
            for (d, theta) in retained.iter().enumerate() {
                let chain = prov[d].0;
                if chain < chain_count {
                    per_chain_param[chain].push(f64::from(theta[i]));
                }
            }
            param_rhats.push(diag::gelman_rubin(&per_chain_param));
            param_rank_rhats.push(diag::split_rhat_rank(&per_chain_param));
        }
        let mut finite: Vec<f64> = param_rhats.into_iter().filter(|v| v.is_finite()).collect();
        finite.sort_by(f64::total_cmp);
        let frac = finite.iter().filter(|v| **v > 1.05).count() as f64 / finite.len().max(1) as f64;
        println!(
            "param_rhat median={:.4} max={:.4} frac_gt_1.05={frac:.3} n={}",
            finite.get(finite.len() / 2).copied().unwrap_or(f64::NAN),
            finite.last().copied().unwrap_or(f64::NAN),
            finite.len()
        );
        let mut rank: Vec<f64> = param_rank_rhats
            .into_iter()
            .filter(|v| v.is_finite())
            .collect();
        rank.sort_by(f64::total_cmp);
        let rank_frac =
            rank.iter().filter(|v| **v > 1.05).count() as f64 / rank.len().max(1) as f64;
        println!(
            "param_rank_rhat median={:.4} max={:.4} frac_gt_1.05={rank_frac:.3} n={}",
            rank.get(rank.len() / 2).copied().unwrap_or(f64::NAN),
            rank.last().copied().unwrap_or(f64::NAN),
            rank.len()
        );
        if let Some(g) = session.posterior_geometry() {
            println!(
                "geometry sd_norm_median={:.4} flat_frac={:.3} identified_frac={:.3} separation_median={:.3} sep_frac_gt_2={:.3} sep_flat={:.3} sep_identified={:.3} n={}",
                g.normalized_sd_median,
                g.flat_fraction,
                g.identified_fraction,
                g.separation_median,
                g.separated_fraction,
                g.separated_flat_fraction,
                g.separated_identified_fraction,
                g.coordinates
            );
        }
        // Chain-mean prediction distance: total variation between each pair
        // of chains' average test distributions.  Unlike draw-NLL R-hat, this
        // says whether the parameter disagreement changes the decisions.
        if !draw_probs.is_empty() {
            let mut chain_pred = vec![vec![0.0f64; classes]; chain_count];
            let mut chain_rows = vec![0usize; chain_count];
            for row_probs in &draw_probs {
                for (d, probs) in row_probs.iter().enumerate() {
                    let chain = prov[d].0;
                    if chain < chain_count {
                        chain_rows[chain] += 1;
                        for (sum, p) in chain_pred[chain].iter_mut().zip(probs) {
                            *sum += p;
                        }
                    }
                }
            }
            for (chain, n) in chain_rows.iter().enumerate() {
                if *n > 0 {
                    for v in &mut chain_pred[chain] {
                        *v /= *n as f64;
                    }
                }
            }
            let mut tvs = Vec::new();
            for a in 0..chain_count {
                for b in a + 1..chain_count {
                    if chain_rows[a] == 0 || chain_rows[b] == 0 {
                        continue;
                    }
                    let l1: f64 = chain_pred[a]
                        .iter()
                        .zip(&chain_pred[b])
                        .map(|(x, y)| (x - y).abs())
                        .sum();
                    tvs.push(0.5 * l1);
                }
            }
            let mean_tv = tvs.iter().sum::<f64>() / tvs.len().max(1) as f64;
            let max_tv = tvs.iter().copied().fold(0.0f64, f64::max);
            println!(
                "prediction_chains tv_mean={mean_tv:.4} tv_max={max_tv:.4} pairs={}",
                tvs.len()
            );

            // The pooled chain mean can hide two different failure shapes: a
            // coherent block of rows (or one label) switching sides, or a
            // diffuse shift spread over every row.  Average each chain's
            // prediction within a row, then compare rows and true classes.
            let mut row_pred = vec![vec![vec![0.0f64; classes]; chain_count]; draw_probs.len()];
            let mut row_counts = vec![vec![0usize; chain_count]; draw_probs.len()];
            for (r, row_probs) in draw_probs.iter().enumerate() {
                for (d, probs) in row_probs.iter().enumerate() {
                    let chain = prov[d].0;
                    if chain < chain_count {
                        row_counts[r][chain] += 1;
                        for (sum, p) in row_pred[r][chain].iter_mut().zip(probs) {
                            *sum += p;
                        }
                    }
                }
                for c in 0..chain_count {
                    let n = row_counts[r][c];
                    if n > 0 {
                        for v in &mut row_pred[r][c] {
                            *v /= n as f64;
                        }
                    }
                }
            }
            let pair_tv = |a: &[f64], b: &[f64]| -> f64 {
                0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
            };
            let mut row_tvs = Vec::new();
            let mut row_tv_by_row = vec![f64::NAN; row_pred.len()];
            let mut row_disagree = 0usize;
            let mut row_scored = 0usize;
            for (r, preds) in row_pred.iter().enumerate() {
                let usable: Vec<usize> =
                    (0..chain_count).filter(|&c| row_counts[r][c] > 0).collect();
                if usable.len() < 2 {
                    continue;
                }
                row_scored += 1;
                let before = row_tvs.len();
                for (i, &a) in usable.iter().enumerate() {
                    for &b in &usable[i + 1..] {
                        row_tvs.push(pair_tv(&preds[a], &preds[b]));
                    }
                }
                row_tv_by_row[r] =
                    row_tvs[before..].iter().sum::<f64>() / (row_tvs.len() - before) as f64;
                let argmax = |p: &[f64]| {
                    p.iter()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| x.total_cmp(y))
                        .map(|(k, _)| k)
                };
                let first = argmax(&preds[usable[0]]);
                if usable.iter().skip(1).any(|&c| argmax(&preds[c]) != first) {
                    row_disagree += 1;
                }
            }
            if !row_tvs.is_empty() {
                let row_mean = row_tvs.iter().sum::<f64>() / row_tvs.len() as f64;
                let row_max = row_tvs.iter().copied().fold(0.0f64, f64::max);
                let row_frac =
                    row_tvs.iter().filter(|v| **v > 0.10).count() as f64 / row_tvs.len() as f64;
                println!(
                    "prediction_rows tv_mean={row_mean:.4} tv_max={row_max:.4} frac_gt_0.10={row_frac:.3} argmax_disagree={row_disagree}/{row_scored}"
                );
            }

            // Does row-level chain disagreement track where the pooled
            // prediction is actually wrong?  This is a descriptive
            // correlation over test rows, not a validated error predictor.
            let mut pairs_xy: Vec<(f64, f64)> = row_tv_by_row
                .iter()
                .zip(&raw)
                .zip(&test_y)
                .filter_map(|((&tv, p), &y)| {
                    let nll = -p.get(y as usize)?.max(1e-12).ln();
                    (tv.is_finite() && nll.is_finite()).then_some((tv, nll))
                })
                .collect();
            if pairs_xy.len() >= 3 {
                let corr = |v: &[(f64, f64)]| -> f64 {
                    let n = v.len() as f64;
                    let mx = v.iter().map(|p| p.0).sum::<f64>() / n;
                    let my = v.iter().map(|p| p.1).sum::<f64>() / n;
                    let cov = v.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>();
                    let sx = v.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>().sqrt();
                    let sy = v.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>().sqrt();
                    if sx * sy > 0.0 {
                        cov / (sx * sy)
                    } else {
                        f64::NAN
                    }
                };
                let pearson = corr(&pairs_xy);
                pairs_xy.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (rank, p) in pairs_xy.iter_mut().enumerate() {
                    p.0 = rank as f64;
                }
                pairs_xy.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (rank, p) in pairs_xy.iter_mut().enumerate() {
                    p.1 = rank as f64;
                }
                let spearman = corr(&pairs_xy);
                println!(
                    "prediction_errors pearson={pearson:.4} spearman={spearman:.4} n={}",
                    pairs_xy.len()
                );
            }

            // Per-chain mean predictions, so a disagreement cannot hide
            // whether one chain's own decision quality is worse.
            let mut chain_acc = vec![0usize; chain_count];
            let mut chain_nll = vec![0.0f64; chain_count];
            let mut chain_n = vec![0usize; chain_count];
            for (r, preds) in row_pred.iter().enumerate() {
                let y = test_y[r] as usize;
                for c in 0..chain_count {
                    if row_counts[r][c] == 0 {
                        continue;
                    }
                    let top = preds[c]
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(k, _)| k)
                        .unwrap_or(0);
                    chain_acc[c] += usize::from(top == y);
                    chain_nll[c] -= preds[c].get(y).copied().unwrap_or(0.0).max(1e-12).ln();
                    chain_n[c] += 1;
                }
            }
            let accs: Vec<f64> = chain_acc
                .iter()
                .zip(&chain_n)
                .map(|(a, n)| *a as f64 / (*n).max(1) as f64)
                .collect();
            let nlls: Vec<f64> = chain_nll
                .iter()
                .zip(&chain_n)
                .map(|(s, n)| *s / (*n).max(1) as f64)
                .collect();
            println!("prediction_chain_scores acc={accs:?} nll={nlls:?}");

            let mut class_pred = vec![vec![vec![0.0f64; classes]; chain_count]; classes];
            let mut class_counts = vec![vec![0usize; chain_count]; classes];
            for (r, &y) in test_y.iter().enumerate() {
                let y = y as usize;
                if y >= classes {
                    continue;
                }
                for c in 0..chain_count {
                    if row_counts[r][c] == 0 {
                        continue;
                    }
                    class_counts[y][c] += 1;
                    for (sum, p) in class_pred[y][c].iter_mut().zip(&row_pred[r][c]) {
                        *sum += p;
                    }
                }
            }
            let mut class_tvs = Vec::with_capacity(classes);
            for c in 0..classes {
                for k in 0..chain_count {
                    if class_counts[c][k] > 0 {
                        let n = class_counts[c][k] as f64;
                        for v in &mut class_pred[c][k] {
                            *v /= n;
                        }
                    }
                }
                let usable: Vec<usize> = (0..chain_count)
                    .filter(|&k| class_counts[c][k] > 0)
                    .collect();
                let mut tvs = Vec::new();
                for (i, &a) in usable.iter().enumerate() {
                    for &b in &usable[i + 1..] {
                        tvs.push(pair_tv(&class_pred[c][a], &class_pred[c][b]));
                    }
                }
                class_tvs.push(tvs.iter().sum::<f64>() / tvs.len().max(1) as f64);
            }
            let class_mean = class_tvs.iter().sum::<f64>() / class_tvs.len().max(1) as f64;
            let class_max = class_tvs.iter().copied().fold(0.0f64, f64::max);
            println!("prediction_classes tv={class_tvs:?} mean={class_mean:.4} max={class_max:.4}");
        }
    }
    if !cal.is_empty() {
        let val_s = score(&cal_y, &cal_probs);
        println!(
            "rotation val raw  acc={:.2} nll={:.4} brier={:.4} ece={:.4}",
            val_s.acc, val_s.nll, val_s.brier, val_s.ece
        );
    }
    if !cal.is_empty() {
        let t = temperature(&cal_probs, &cal_y);
        let scaled = apply_temperature(&raw, t);
        let scaled_s = score(&test_y, &scaled);
        println!(
            "rotation T={t:.3}  acc={:.2} nll={:.4} brier={:.4} ece={:.4}",
            scaled_s.acc, scaled_s.nll, scaled_s.brier, scaled_s.ece
        );
    }

    // Positive-square-root softmax control: its measured one-step Born
    // probabilities are the softmax probabilities by construction.  Sharing
    // `--hidden` with the rotation head keeps the two flat parameter counts
    // comparable; the exact counts are printed for the record.  A linear angle
    // head has no hidden width to share, so the control is omitted rather than
    // given an unmatched architecture.
    if hidden.is_empty() {
        println!("softmax control skipped for linear head");
        return;
    }
    let soft = MlpHeadPotential::new_with_loss(&table, &hidden, LossKind::Nll).unwrap();
    println!(
        "params rotation={} softmax={}",
        session.head().flat_size(),
        soft.flat_size()
    );
    let mut rng = CounterRng::new(num(args, "--seed", 20261118) ^ 0x50f7);
    let theta0 = (0..soft.flat_size())
        .map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal) as f32 * 0.02)
        .collect::<Vec<_>>();
    let theta = if has_flag(args, "--softmax-tune") {
        let val_table = if cal.is_empty() {
            None
        } else {
            Some(one_split_table(&cal, true, dim, classes))
        };
        let wds = number_list(args, "--softmax-wd-grid", "0,0.0001,0.001");
        assert!(!wds.is_empty(), "empty softmax weight-decay grid");
        let steps = num(args, "--softmax-pretrain", 3000usize);
        let eval_every = num(args, "--softmax-eval-every", 100usize);
        let patience = num(args, "--softmax-patience", 4usize);
        let mut best: Option<(f64, Vec<f32>, usize, f64)> = None;
        let mut summary = Vec::new();
        for wd in wds {
            let (theta, step, val_nll) = train_softmax_candidate(
                &soft,
                theta0.clone(),
                steps,
                0.01,
                wd,
                eval_every,
                patience,
                val_table.as_ref(),
                &hidden,
                classes,
            );
            let key = val_nll.unwrap_or(f64::INFINITY);
            summary.push(format!(
                "{wd}:{}@{step}",
                val_nll.map_or_else(|| "nan".to_string(), |v| format!("{v:.4}"))
            ));
            if best.as_ref().is_none_or(|b| key < b.0) {
                best = Some((key, theta, step, wd));
            }
        }
        let (val, theta, step, wd) = best.expect("at least one softmax candidate");
        println!(
            "softmax tune wd={wd} step={step} val_nll={:.4} grid={}",
            val,
            summary.join(",")
        );
        theta
    } else {
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
        theta
    };
    let probs = predict_proba_mlp(&table, &theta, &hidden, Split::Test).unwrap();
    let probs: Vec<Vec<f64>> = probs
        .chunks(classes)
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect();
    let s = score(&test_y, &probs);
    println!(
        "softmax control   acc={:.2} nll={:.4} brier={:.4} ece={:.4}",
        s.acc, s.nll, s.brier, s.ece
    );
    if !cal.is_empty() {
        let val_table = one_split_table(&cal, true, dim, classes);
        let val = predict_proba_mlp(&val_table, &theta, &hidden, Split::Test).unwrap();
        let val: Vec<Vec<f64>> = val
            .chunks(classes)
            .map(|r| r.iter().map(|&v| v as f64).collect())
            .collect();
        let val_s = score(&cal_y, &val);
        println!(
            "softmax val       acc={:.2} nll={:.4} brier={:.4} ece={:.4}",
            val_s.acc, val_s.nll, val_s.brier, val_s.ece
        );
    }
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
        let u = head
            .operator(theta, step)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
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
        prior_mean: Vec::new(),
        proposal_betas: Vec::new(),
        sampler: pcnl::rotation_head::Sampler::Pcn,
        mala_step: num(args, "--mala-step", 0.01),
        temperatures: Vec::new(),
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
        println!(
            "{tag}: pretrain_steps={} last={:?}",
            trace.len(),
            trace.last()
        );
        if has_flag(args, "--anchor-prior") {
            let sigma = session.set_anchor_prior(&start).unwrap();
            println!("{tag}: anchor_prior_sigma={sigma:.6}");
        }
        session.warm_start(&start).unwrap();
        let report = session.update().unwrap();
        println!(
            "{tag}: accepted={} proposals={} retained={} swap_attempts={} swap_accepted={}",
            report.accepted,
            report.proposals,
            report.retained_draws,
            report.swap_attempts,
            report.swap_accepted
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
    println!("## learned sequences: len={len} train={n_train} test={n_test} classes={classes}");
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

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn arg_str<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or(default)
}

fn hidden(args: &[String], default: &[usize]) -> Vec<usize> {
    args.iter()
        .position(|a| a == "--hidden")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
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
