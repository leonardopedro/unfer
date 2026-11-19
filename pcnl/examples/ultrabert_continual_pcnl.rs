//! Continual Bayesian learning on frozen UltraBERT embeddings.
//!
//! The data stream is a sequence of class batches.  At round `t` the target is
//! Bayes' rule with the previous round's posterior as the prior:
//!
//! ```text
//! π_t(θ) ∝ π_{t-1}(θ) · exp(−λ N_t · mean Brier_t(θ))
//! ```
//!
//! `π_{t-1}` is approximated by the diagonal Gaussian fitted to its pCNL
//! particles, so the exact full-target pCN kernel can sample `π_t` in the
//! reparameterised chart.  The `independent` arm resets that prior every round;
//! the `continual` arm carries it.  Both use the same basis, proposal budget and
//! seeds, so the difference is the carried posterior rather than tuning.
//!
//! A diagonal prior alone is not enough on class-disjoint rounds: the shared
//! hidden layers drift.  The delivered target therefore adds replay rows for
//! seen classes, which makes the update an approximate Bayesian-replay target,
//! not exact Bayes over the full history.  `--replay-per-class 0` turns replay
//! off and exposes that failure mode.
//!
//! The encoder is fixed.  Only the head is in the posterior.

use pcnl::continual::DiagonalPosterior;
use pcnl::experiment::expected_calibration_error;
use pcnl::subspace::Basis;
use pcnl::target::{FullPotential, PotentialFullTarget, adapt_beta, mh_pcn_w_full, mh_pcn_z_full};
use pcnl::text_head::{EmbeddingTable, LossKind, MlpHeadPotential, Split, predict_proba_mlp};
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Cfg {
    fixture: PathBuf,
    hidden: Vec<usize>,
    tasks: Vec<Vec<u32>>,
    rank: usize,
    sigma: f32,
    lambda: f64,
    chains: usize,
    burn: usize,
    steps: usize,
    thin: usize,
    beta: f64,
    min_std: f32,
    replay_per_class: usize,
    seed: u64,
    arm: Option<String>,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            fixture: PathBuf::from("docs/fixtures/ultrabert_agnews_embeddings.safetensors"),
            hidden: vec![32, 16],
            // Two classes at a time, revisiting the first pair, so old-class
            // retention is measurable after a disjoint round.
            tasks: vec![vec![0, 1], vec![2, 3], vec![0, 1], vec![2, 3]],
            rank: 8,
            sigma: 0.5,
            lambda: 1.0,
            chains: 4,
            burn: 800,
            steps: 2_400,
            thin: 4,
            beta: 0.1,
            min_std: 1e-3,
            // 60 is the full clean training split per class in this fixture.
            // A smaller value is a fixed memory budget with more forgetting.
            replay_per_class: 60,
            seed: 20261118,
            arm: None,
        }
    }
}

impl Cfg {
    fn from_args() -> Self {
        let mut cfg = Self::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let flag = |name: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        if let Some(v) = flag("--fixture") {
            cfg.fixture = PathBuf::from(v);
        }
        if let Some(v) = flag("--hidden") {
            cfg.hidden = v
                .split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&h| h > 0)
                .collect();
            assert!(!cfg.hidden.is_empty(), "--hidden needs at least one layer");
        }
        if let Some(v) = flag("--tasks") {
            cfg.tasks = v
                .split(';')
                .map(|task| {
                    task.split(',')
                        .filter_map(|x| x.trim().parse::<u32>().ok())
                        .collect::<Vec<_>>()
                })
                .filter(|task| !task.is_empty())
                .collect();
            assert!(!cfg.tasks.is_empty(), "--tasks needs at least one round");
        }
        if let Some(v) = flag("--rank").and_then(|v| v.parse().ok()) {
            cfg.rank = v;
        }
        if let Some(v) = flag("--sigma").and_then(|v| v.parse().ok()) {
            cfg.sigma = v;
        }
        if let Some(v) = flag("--lambda").and_then(|v| v.parse().ok()) {
            cfg.lambda = v;
        }
        if let Some(v) = flag("--chains").and_then(|v| v.parse().ok()) {
            cfg.chains = v;
        }
        if let Some(v) = flag("--burn").and_then(|v| v.parse().ok()) {
            cfg.burn = v;
        }
        if let Some(v) = flag("--steps").and_then(|v| v.parse().ok()) {
            cfg.steps = v;
        }
        if let Some(v) = flag("--thin").and_then(|v| v.parse().ok()) {
            cfg.thin = v;
        }
        if let Some(v) = flag("--beta").and_then(|v| v.parse().ok()) {
            cfg.beta = v;
        }
        if let Some(v) = flag("--min-std").and_then(|v| v.parse().ok()) {
            cfg.min_std = v;
        }
        if let Some(v) = flag("--replay-per-class").and_then(|v| v.parse().ok()) {
            cfg.replay_per_class = v;
        }
        if let Some(v) = flag("--seed").and_then(|v| v.parse().ok()) {
            cfg.seed = v;
        }
        if let Some(v) = flag("--arm") {
            match v.as_str() {
                "both" => cfg.arm = None,
                "continual" | "independent" => cfg.arm = Some(v),
                _ => panic!("--arm must be both, continual or independent"),
            }
        }
        cfg.burn = cfg.burn.min(cfg.steps.saturating_sub(1));
        cfg.thin = cfg.thin.max(1);
        cfg.chains = cfg.chains.max(1);
        cfg
    }
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    acc: f64,
    nll: f64,
    brier: f64,
    ece: f64,
    per_class: [f64; 4],
}

fn eval_probs(table: &EmbeddingTable, p: &[f32]) -> Metrics {
    let c = table.classes();
    assert_eq!(c, 4, "this demonstration assumes four classes");
    let y = table.y(Split::Test);
    let n = y.len();
    let mut hits = 0usize;
    let mut nll = 0.0f64;
    let mut brier = 0.0f64;
    let mut per = [0usize; 4];
    let mut per_n = [0usize; 4];
    for (i, &t) in y.iter().enumerate() {
        let row: Vec<f64> = p[i * c..(i + 1) * c].iter().map(|&v| v as f64).collect();
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(j, _)| j)
            .unwrap_or(0);
        if best == t as usize {
            hits += 1;
            per[t as usize] += 1;
        }
        per_n[t as usize] += 1;
        nll -= row[t as usize].max(1e-12).ln();
        brier += pcnl::readout::brier(&row, t as usize);
    }
    let mut per_class = [0.0f64; 4];
    for i in 0..4 {
        per_class[i] = 100.0 * per[i] as f64 / per_n[i].max(1) as f64;
    }
    Metrics {
        acc: 100.0 * hits as f64 / n as f64,
        nll: nll / n as f64,
        brier: brier / n as f64,
        ece: expected_calibration_error(&p.iter().map(|&v| v as f64).collect::<Vec<_>>(), c, y, 10),
        per_class,
    }
}

fn bma_probs(table: &EmbeddingTable, head: &MlpHeadPotential, draws: &[Vec<f32>]) -> Vec<f32> {
    let c = table.classes();
    let n = table.n_test();
    let mut sum = vec![0.0f64; n * c];
    for theta in draws {
        let p = predict_proba_mlp(table, theta, head.hidden(), Split::Test).expect("probs");
        for (s, &v) in sum.iter_mut().zip(&p) {
            *s += f64::from(v);
        }
    }
    let k = draws.len().max(1) as f64;
    sum.iter().map(|&v| (v / k) as f32).collect()
}

/// One full-target pCNL round.  `z=0, w=0` is the prior mean by construction.
fn sample_round(
    cfg: &Cfg,
    task_table: &EmbeddingTable,
    head: &MlpHeadPotential,
    prior: &DiagonalPosterior,
    round: usize,
    tag: &str,
) -> (Vec<Vec<f32>>, f64) {
    let p = head.flat_size();
    // Keep one basis for the whole stream.  Re-drawing it each round would
    // re-express the carried prior through a different random chart, so the
    // chain could not exploit the directions the previous posterior learned.
    let basis = Basis::random(p, cfg.rank, cfg.seed).expect("random basis");
    let subspace = prior.subspace(basis).expect("prior subspace");
    let target = PotentialFullTarget::new(
        subspace.clone(),
        cfg.lambda,
        MlpHeadPotential::new_with_loss(task_table, &cfg.hidden, LossKind::Brier)
            .expect("task target"),
    );
    let started = Instant::now();
    let mut draws = Vec::new();
    let mut accepts_w = 0usize;
    let mut count_w = 0usize;
    for chain in 0..cfg.chains {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(cfg.seed ^ ((chain as u64 + 1) << 32) ^ round as u64);
        let mut z = vec![0.0f32; cfg.rank];
        let mut w = vec![0.0f32; p];
        let mut phi = target.phi(&z, &w);
        let mut bz = cfg.beta;
        let mut bw = cfg.beta;
        for t in 0..cfg.steps {
            let betas = vec![bz; cfg.rank];
            let sz = mh_pcn_z_full(&target, &mut z, &w, phi, &betas, &mut rng);
            phi = sz.phi;
            let sw = mh_pcn_w_full(&target, &z, &mut w, phi, bw, &mut rng);
            phi = sw.phi;
            if t < cfg.burn {
                bz = adapt_beta(bz, sz.accepted, t, 0.3);
                bw = adapt_beta(bw, sw.accepted, t, 0.3);
            } else {
                count_w += 1;
                accepts_w += usize::from(sw.accepted);
                if (t - cfg.burn) % cfg.thin == 0 {
                    draws.push(target.lift(&z, &w));
                }
            }
        }
    }
    let accept = accepts_w as f64 / count_w.max(1) as f64;
    eprintln!(
        "  {tag} round {round}: {} draws, w acceptance {:.1}%, {:.1}s",
        draws.len(),
        100.0 * accept,
        started.elapsed().as_secs_f64()
    );
    (draws, accept)
}

fn main() {
    let cfg = Cfg::from_args();
    let table = EmbeddingTable::load(&cfg.fixture).expect("load UltraBERT fixture");
    let head = MlpHeadPotential::new(&table, &cfg.hidden).expect("MLP head");
    let p = head.flat_size();
    println!(
        "data: {} train / {} test / {} classes; head=mlp hidden={:?} P={}",
        table.n_train(),
        table.n_test(),
        table.classes(),
        cfg.hidden,
        p
    );
    println!(
        "stream: {:?}; rank={} sigma={} lambda={} chains={} steps={} burn={} min_std={} replay_per_class={}",
        cfg.tasks,
        cfg.rank,
        cfg.sigma,
        cfg.lambda,
        cfg.chains,
        cfg.steps,
        cfg.burn,
        cfg.min_std,
        cfg.replay_per_class
    );
    println!("| round | arm | train classes | acc | NLL | Brier | ECE | c0 | c1 | c2 | c3 |");
    println!("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");

    let arms: Vec<&str> = match cfg.arm.as_deref() {
        Some("independent") => vec!["independent"],
        Some("continual") => vec!["continual"],
        _ => vec!["independent", "continual"],
    };
    for &arm in &arms {
        let mut prior =
            DiagonalPosterior::isotropic(p, vec![0.0; p], cfg.sigma).expect("initial prior");
        let mut replay: Option<EmbeddingTable> = None;
        let mut history: Vec<(Vec<u32>, Metrics)> = Vec::new();
        for (round, classes) in cfg.tasks.iter().enumerate() {
            let task_table = table.training_for_classes(classes).expect("task classes");
            let train_table = match &replay {
                Some(old) => task_table.concat_train(old).expect("replay concat"),
                None => task_table.clone(),
            };
            let (draws, _) = sample_round(&cfg, &train_table, &head, &prior, round, arm);
            let probs = bma_probs(&table, &head, &draws);
            let m = eval_probs(&table, &probs);
            println!(
                "| {} | {} | {:?} | {:.2}% | {:.4} | {:.4} | {:.4} | {:.1}% | {:.1}% | {:.1}% | {:.1}% |",
                round + 1,
                arm,
                classes,
                m.acc,
                m.nll,
                m.brier,
                m.ece,
                m.per_class[0],
                m.per_class[1],
                m.per_class[2],
                m.per_class[3]
            );
            history.push((classes.clone(), m));
            prior = if arm == "continual" {
                DiagonalPosterior::from_particles(&draws, cfg.min_std).expect("posterior summary")
            } else {
                DiagonalPosterior::isotropic(p, vec![0.0; p], cfg.sigma).expect("reset prior")
            };
            if cfg.replay_per_class > 0 {
                let fresh = task_table
                    .train_subset(cfg.replay_per_class, cfg.seed ^ round as u64)
                    .expect("replay sample");
                replay = Some(match replay {
                    Some(old) => old.concat_train(&fresh).expect("replay append"),
                    None => fresh,
                });
            }
        }
        // Forgetting over the classes that a later disjoint round did not train.
        let mut drops = Vec::new();
        for (i, (classes, before)) in history.iter().enumerate() {
            let later_disjoint = history[i + 1..]
                .iter()
                .find(|(later, _)| later.iter().all(|c| !classes.contains(c)));
            if let Some((later, after)) = later_disjoint {
                for &c in classes {
                    drops.push(before.per_class[c as usize] - after.per_class[c as usize]);
                }
                eprintln!("  {arm}: classes {classes:?} before {later:?} drop {drops:?}");
            }
        }
        if !drops.is_empty() {
            let mean = drops.iter().sum::<f64>() / drops.len() as f64;
            println!(
                "| final | {} | forgetting over disjoint later rounds | | | | | | | | mean {:.2} pp |",
                arm, mean
            );
        }
    }
}
