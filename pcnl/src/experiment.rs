//! Phases 0/3/4 of the *updated* plan (unadjusted SG-pCNL) as one
//! deterministic, measured run.
//!
//! * **Phase 0:** both variants are driven by the *same* unadjusted
//!   mini-batch dynamics (§3A): `θ ← √(1−β²)θ + βξ − (β²σ²/2)·g_batch`,
//!   no Metropolis filter anywhere. The plan's hypothesis is measured:
//!   standard dot-product attention should explode or random-walk on
//!   unbounded mini-batch gradient spikes, while nGPT keeps
//!   `‖θ_t‖ ≈ σ√P` over 10,000 iterations. The report says which of
//!   the two actually happened — for the config run — not which was
//!   expected.
//! * **Phase 3:** Task-A pre-train (momentum SGD), adaptation to Task
//!   B by the unadjusted chain, BMA over the thinned trajectory.
//! * **Phase 4:** ESS, split-R̂, radial-norm vs `σ√P`, held-out
//!   accuracy/NLL vs the no-adaptation and SGD baselines. (Acceptance
//!   rate no longer exists — §1's tuning-shift entry: β comes from
//!   the schedule, not from acceptance feedback.)
//!
//! The JAX reference (plan §1, "Phase 0 only") stays skipped: the toy
//! is Rust, and the kernel's sanity anchor is the analytic-Gaussian
//! tracking test in `sampler`.

use crate::diag::{ess, gelman_rubin};
use crate::model::{Shapes, Variant, init_flat};
use crate::persist::save_chain_trajectory;
use crate::sampler::{SgPcnlScheduler, SgPcnlState, norm};
use crate::task::{BatchLoader, Dataset, Task, adam, momentum_sgd, sgld_chain};
use candle_core::Tensor;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub k: usize,
    /// The adaptation target the chains sample (task×budget matrix's
    /// knob). `Task::B` is the plan's canonical target; `Task::Modular`
    /// selects the simple/complex cells. Task A (the pre-train) stays
    /// fixed — the story is "pre-train on A, adapt to B".
    pub task_b: Task,
    pub shapes: Shapes,
    pub n_train: usize,
    pub n_test: usize,
    pub pretrain_steps: usize,
    pub pretrain_lr: f64,
    pub chains: usize,
    /// Total unadjusted iterations (plan Phase 0: 10,000).
    pub steps: usize,
    /// Transient before samples are kept (the trajectory's first half
    /// by default — not an MH burn-in; there is no MH).
    pub burn: usize,
    pub thin: usize,
    pub batch_size: usize,
    pub sigma: f64,
    pub beta_0: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub sgd_steps_on_b: usize,
    /// Adam's fine-tuning steps on Task B (Phase 4's "SGD/Adam" pair).
    pub adam_steps_on_b: usize,
    /// Adam step size — the standard default; SGD's pair uses
    /// `pretrain_lr`, so the two deterministic baselines keep their
    /// own conventional settings rather than sharing one lr.
    pub adam_lr: f64,
    pub init_std: f64,
    /// Week 2 — when set, every chain's thinned trajectory is written
    /// to this directory as `.safetensors` (plan §1, `safetensors`).
    pub save_dir: Option<PathBuf>,
    /// Plan §3C Option 2 (Phase 4b): 0 = Robbins–Monro (Option 1);
    /// any period > 0 = cyclical cosine decay with that period, applied
    /// to every sampler chain (pCNL, SGLD, MH-pCNL) so the schedule is
    /// the only thing that changed between two runs of one seed.
    pub cyclical_period: usize,
    /// Phase 4b: whether to run the Metropolis-filtered predecessor
    /// (`task::mh_pcnl_chain`) as the reference comparator. Tests may
    /// disable it to keep debug-mode runtime down; the full run keeps
    /// it on.
    pub run_mh: bool,
    /// §6.4.1's reachability follow-up: center the adaptation prior
    /// at θ_A — i.e. sample from N(θ_A, σ²I)·L(D_B; θ) instead of
    /// N(0, σ²I)·L(D_B; θ) — so the chains equilibrate on the shell
    /// around the pre-trained weights instead of being contracted to
    /// the origin's shell during burn-in (B = A then fails at 24.6 %
    /// while θ_A itself scores 46.1 %: §6.4.1's open item).
    /// `false` (the plan's original prior) keeps every reported run
    /// reproducible; the SGLD twin and the MH filter receive the same
    /// translation, so the cross-R̂ still couples the same target.
    pub prior_center_at_a: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 20261020,
            k: 8,
            task_b: Task::B,
            shapes: Shapes {
                vocab: 9,
                d: 32,
                ff: 64,
                layers: 2,
            },
            n_train: 512,
            n_test: 256,
            pretrain_steps: 1_500,
            pretrain_lr: 0.6,
            chains: 4,
            steps: 10_000,
            burn: 5_000,
            thin: 10,
            batch_size: 64,
            sigma: 0.35,
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.55,
            sgd_steps_on_b: 1_000,
            adam_steps_on_b: 1_000,
            adam_lr: 0.05,
            init_std: 0.1,
            save_dir: None,
            cyclical_period: 0,
            run_mh: true,
            prior_center_at_a: false,
        }
    }
}

impl Config {
    /// Debug-sized so the experiment has a fast smoke test; the shape
    /// of the output is identical.
    pub fn quick() -> Self {
        Self {
            shapes: Shapes {
                vocab: 9,
                d: 16,
                ff: 32,
                layers: 2,
            },
            n_train: 128,
            n_test: 64,
            pretrain_steps: 120,
            chains: 2,
            steps: 400,
            burn: 200,
            thin: 4,
            batch_size: 32,
            sgd_steps_on_b: 60,
            adam_steps_on_b: 60,
            run_mh: false,
            ..Self::default()
        }
    }

    /// The debug-sized experiment *with* the MH-filtered comparator
    /// (Phase 4b's `mh` quick config).
    pub fn quick_mh() -> Self {
        Self {
            run_mh: true,
            ..Self::quick()
        }
    }
}

/// Everything the run measured, for one model variant.
#[derive(Clone, Debug)]
pub struct VariantReport {
    pub variant: &'static str,
    pub p: usize,
    pub acc_a_pretrained: f64,
    pub loss_a_first: f64,
    pub loss_a_last: f64,
    pub acc_b_before: f64,
    pub nll_b_before: f64,
    pub acc_b_sgd: f64,
    pub nll_b_sgd: f64,
    /// Phase 4's second deterministic comparator: Adam on Task B.
    pub acc_b_adam: f64,
    pub nll_b_adam: f64,
    /// Phase 4's standard-sampler comparator: SGLD chains on Task B —
    /// same β schedule and noise magnitude as the pCNL step, prior
    /// handled as a gradient term (see `task::sgld_chain`).
    pub sgld_radial_final: f64,
    pub sgld_radial_max: f64,
    pub sgld_ess_loss: f64,
    pub sgld_rhat_loss: f64,
    pub sgld_bma_accuracy: f64,
    pub sgld_bma_nll: f64,
    pub sgld_bma_ece: f64,
    pub sgld_bma_models: usize,
    pub sgld_seconds: f64,
    /// Split-R̂ over the *union* of SG-pCNL and SGLD loss chains:
    /// ≈1 is the evidence that both algorithms sample the same
    /// distribution (they agree to O(β⁴) per step — `task::sgld_chain`).
    pub rhat_cross: f64,
    /// Phase 4b — the Metropolis-filtered predecessor (`cfg.run_mh`):
    /// acceptance rate in %, radial shell, diagnostics, BMA, kept
    /// samples per chain, wall-clock, and `rhat_cross_mh` — the
    /// filtered∪unfiltered cross-R̂ that isolates the unadjusted bias.
    pub mh_accept_rate: f64,
    pub mh_radial_final: f64,
    pub mh_ess_loss: f64,
    pub mh_rhat_loss: f64,
    pub mh_bma_accuracy: f64,
    pub mh_bma_nll: f64,
    pub mh_bma_ece: f64,
    pub mh_bma_models: usize,
    pub mh_kept_per_chain: usize,
    pub mh_seconds: f64,
    pub rhat_cross_mh: f64,
    /// Schedule bookkeeping (§3C) — replaces the old acceptance pair.
    pub beta_start: f64,
    pub beta_end: f64,
    pub steps: usize,
    /// True when θ (or the monitored loss) went non-finite: the
    /// Phase-0 "standard transformer explodes" outcome, measured.
    pub exploded: bool,
    pub exploded_at_step: Option<usize>,
    /// Where it happened: `true` = the deterministic pre-train died
    /// before any SG-pCNL step (a baseline fact, not a chain fact).
    pub exploded_in_pretrain: bool,
    /// Any observed non-finite mini-batch gradient — the Phase-0
    /// "gradient spikes" evidence, recorded even when it kills the run.
    pub grad_nonfinite: bool,
    /// The Phase-0 spike measure: max ‖scaled g‖ actually applied
    /// (finite values only; `grad_nonfinite` carries the rest).
    pub max_scaled_grad_norm: f64,
    pub max_batch_loss: f64,
    pub radial_expected: f64, // σ√P
    pub radial_final: f64,
    pub radial_max: f64,
    pub radial_mean: f64,
    pub ess_loss: f64,
    pub ess_median_coords: f64,
    pub rhat_loss: f64,
    pub rhat_coords_median: f64,
    pub rhat_coords_max: f64,
    pub bma_accuracy: f64,
    pub bma_nll: f64,
    /// Phase 4's Expected Calibration Error of the BMA predictive
    /// distribution (10 equal-width confidence bins on [0,1]).
    pub bma_ece: f64,
    pub last_state_accuracy: f64,
    pub bma_models: usize,
    /// Paths of the `.safetensors` trajectory files written (§1).
    pub saved_files: Vec<String>,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub config: Config,
    pub variants: Vec<VariantReport>,
}

impl Report {
    /// Multi-line, fully numeric rendering — the artifact the doc
    /// quotes and the snapshot test pins structurally.
    pub fn render(&self) -> String {
        let mut s = String::new();
        // The centered adaptation prior (§6.4.1's follow-up) is named
        // in the header only when it is on, so every run of the plan's
        // original config renders byte-identically to the reports the
        // doc quotes verbatim.
        let prior = if self.config.prior_center_at_a {
            ", prior=theta_A"
        } else {
            ""
        };
        s.push_str(&format!(
            "SG-pCNL (unadjusted) x nGPT-lite toy run (seed {}, sigma={}, beta0={}, kappa={}, \
             gamma={}, batch={}, chains={}, steps={}, burn={}, thin={}, taskA=(a+b) mod {}, \
             taskB={}, n_train={}, pretrain={}{})\n",
            self.config.seed,
            self.config.sigma,
            self.config.beta_0,
            self.config.kappa,
            self.config.gamma,
            self.config.batch_size,
            self.config.chains,
            self.config.steps,
            self.config.burn,
            self.config.thin,
            self.config.k,
            self.config.task_b.formula(self.config.k),
            self.config.n_train,
            self.config.pretrain_steps,
            prior
        ));
        for v in &self.variants {
            s.push_str(&format!(
                "[{variant}] P={p}\n  pre-train A: loss {l0:.4} -> {l1:.4}, acc_A {aa:.1}%\n  \
                 B baselines: before {ab:.1}% (NLL {nb:.4}), SGD-on-B {as_:.1}% (NLL {ns:.4}), \
                 Adam-on-B {ad:.1}% (NLL {nd:.4})\n",
                variant = v.variant,
                p = v.p,
                l0 = v.loss_a_first,
                l1 = v.loss_a_last,
                aa = v.acc_a_pretrained,
                ab = v.acc_b_before,
                nb = v.nll_b_before,
                as_ = v.acc_b_sgd,
                ns = v.nll_b_sgd,
                ad = v.acc_b_adam,
                nd = v.nll_b_adam,
            ));
            if v.exploded {
                let where_ = if v.exploded_in_pretrain {
                    "during Task-A pre-train (baseline)".to_string()
                } else {
                    format!(
                        "at step {}",
                        v.exploded_at_step
                            .map(|x| x.to_string())
                            .unwrap_or_else(|| "?".into())
                    )
                };
                s.push_str(&format!(
                    "  SG-pCNL: EXPLODED {where_} (max scaled|g| {:.3}{}, max batch loss {:.3}, \
                     radial max {:.3} vs sigma*sqrt(P) {:.3})\n",
                    v.max_scaled_grad_norm,
                    if v.grad_nonfinite {
                        ", gradient non-finite"
                    } else {
                        ""
                    },
                    v.max_batch_loss,
                    v.radial_max,
                    v.radial_expected,
                ));
                continue;
            }
            s.push_str(&format!(
                "  SG-pCNL: beta {b0:.4} -> {b1:.4} over {st} steps | max scaled|g| {g:.3} \
                 max batch loss {bl:.3} | radial final {rf:.1} max {rmx:.1} mean {rmean:.1} \
                 vs sigma*sqrt(P) {re:.1} | ESS(loss) {el:.0} ESS(coord)med {ec:.0} | \
                 Rhat(loss) {rl:.3} Rhat coord med {rmed:.3} max {rmax:.3} | \
                 BMA {ba:.1}% NLL {bn:.4} ECE {be:.4} (K={k}) last-state {la:.1}% | {sec:.1}s\n",
                b0 = v.beta_start,
                b1 = v.beta_end,
                st = v.steps,
                g = v.max_scaled_grad_norm,
                bl = v.max_batch_loss,
                rf = v.radial_final,
                rmx = v.radial_max,
                rmean = v.radial_mean,
                re = v.radial_expected,
                el = v.ess_loss,
                ec = v.ess_median_coords,
                rl = v.rhat_loss,
                rmed = v.rhat_coords_median,
                rmax = v.rhat_coords_max,
                ba = v.bma_accuracy,
                bn = v.bma_nll,
                be = v.bma_ece,
                k = v.bma_models,
                la = v.last_state_accuracy,
                sec = v.seconds,
            ));
            s.push_str(&format!(
                "  SGLD (baseline): radial final {rf:.1} max {rmx:.1} vs sigma*sqrt(P) {re:.1} | \
                 ESS(loss) {el:.0} | Rhat(loss) {rl:.3} | BMA {ba:.1}% NLL {bn:.4} ECE {be:.4} \
                 (K={k}) | cross-Rhat {xc:.3} | {sec:.1}s\n",
                rf = v.sgld_radial_final,
                rmx = v.sgld_radial_max,
                re = v.radial_expected,
                el = v.sgld_ess_loss,
                rl = v.sgld_rhat_loss,
                ba = v.sgld_bma_accuracy,
                bn = v.sgld_bma_nll,
                be = v.sgld_bma_ece,
                k = v.sgld_bma_models,
                xc = v.rhat_cross,
                sec = v.sgld_seconds,
            ));
            if v.mh_kept_per_chain == 0 {
                s.push_str("  MH-pCNL (filtered): off (cfg.run_mh = false)\n");
            } else {
                s.push_str(&format!(
                    "  MH-pCNL (filtered): accept {ac:.1}% | radial final {rf:.1} vs sigma*sqrt(P) {re:.1} | \
                     ESS(loss) {el:.0} | Rhat(loss) {rl:.3} | BMA {ba:.1}% NLL {bn:.4} ECE {be:.4} \
                     (K={k}) | kept {kc} per chain | cross-Rhat {xc:.3} | {sec:.1}s\n",
                    ac = v.mh_accept_rate,
                    rf = v.mh_radial_final,
                    re = v.radial_expected,
                    el = v.mh_ess_loss,
                    rl = v.mh_rhat_loss,
                    ba = v.mh_bma_accuracy,
                    bn = v.mh_bma_nll,
                    be = v.mh_bma_ece,
                    k = v.mh_bma_models,
                    kc = v.mh_kept_per_chain,
                    xc = v.rhat_cross_mh,
                    sec = v.mh_seconds,
                ));
            }
            if !v.saved_files.is_empty() {
                s.push_str(&format!(
                    "  saved: {} trajectory file(s): {}\n",
                    v.saved_files.len(),
                    v.saved_files.join(", ")
                ));
            }
        }
        s
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Phase 4 — Expected Calibration Error over a predictive distribution:
/// 10 equal-width confidence bins on [0,1],
/// `ECE = Σ_b (|b|/n)·|acc(b) − conf(b)|`.
/// `probs` is row-major `[n × vocab]`, already averaged over models.
fn expected_calibration_error(probs: &[f64], vocab: usize, targets: &[u32], n_bins: usize) -> f64 {
    if targets.is_empty() || vocab == 0 {
        return f64::NAN;
    }
    let mut count = vec![0usize; n_bins];
    let mut conf = vec![0f64; n_bins];
    let mut acc = vec![0f64; n_bins];
    for (i, &t) in targets.iter().enumerate() {
        let mut best = 0usize;
        let mut best_p = -1f64;
        for v in 0..vocab {
            let p = probs[i * vocab + v];
            if p > best_p {
                best_p = p;
                best = v;
            }
        }
        let b = ((best_p * n_bins as f64) as usize).min(n_bins - 1);
        count[b] += 1;
        conf[b] += best_p;
        acc[b] += if best as u32 == t { 1.0 } else { 0.0 };
    }
    let n = targets.len() as f64;
    let mut ece = 0.0;
    for b in 0..n_bins {
        if count[b] > 0 {
            let m = count[b] as f64;
            ece += (m / n) * (conf[b] / m - acc[b] / m).abs();
        }
    }
    ece
}

fn softmax_rows(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
    let sum: f64 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// Full-batch loss, forward-only — the low-noise series ESS/R̂ run on.
fn full_loss(theta: &Tensor, shapes: &Shapes, variant: Variant, data: &Dataset) -> Option<f64> {
    let logits = crate::model::forward(theta, shapes, variant, &data.train_x).ok()?;
    let l = candle_nn::loss::cross_entropy(&logits, &data.train_y).ok()?;
    let v = l.to_scalar::<f32>().ok()? as f64;
    v.is_finite().then_some(v)
}

fn radial(theta: &Tensor) -> f64 {
    theta
        .sqr()
        .ok()
        .and_then(|s| s.sum_all().ok())
        .and_then(|s| s.sqrt().ok())
        .and_then(|s| s.to_scalar::<f32>().ok())
        .map(|v| v as f64)
        .unwrap_or(f64::NAN)
}

/// ‖θ − μ‖ — the radius the shell diagnostics are stated in. With
/// the plan's origin prior μ = 0 this is [`radial`] itself; with the
/// centered adaptation prior (§6.4.1's follow-up) the shell the chain
/// equilibrates on is around θ_A, so that is what is compared to
/// σ√P.
fn radial_about(theta: &Tensor, center: Option<&Tensor>) -> f64 {
    match center.and_then(|c| (theta - c).ok()) {
        Some(dev) => radial(&dev),
        None => radial(theta),
    }
}

/// Bayesian model averaging over one or more chains' thinned samples:
/// returns `(accuracy %, NLL, ECE, K models used)`. One implementation
/// shared by the SG-pCNL chains and the SGLD comparator, so the two
/// numbers in the report are computed by the *same* code.
fn bma_metrics(
    chains: &[Vec<Tensor>],
    shapes: &Shapes,
    variant: Variant,
    data: &Dataset,
) -> (f64, f64, f64, usize) {
    let n_test = data.test_y.dim(0).expect("n_test");
    let vocab = shapes.vocab;
    let mut prob_sum = vec![0.0f64; n_test * vocab];
    let mut k_models = 0usize;
    for sample in chains.iter().flatten() {
        if radial(sample).is_finite()
            && let Ok(logits) = crate::model::forward(sample, shapes, variant, &data.test_x)
            && let Ok(rows) = logits.to_vec2::<f32>()
        {
            for (i, row) in rows.iter().enumerate() {
                let p = softmax_rows(&row.iter().map(|x| *x as f64).collect::<Vec<_>>());
                for (v, pv) in p.into_iter().enumerate() {
                    prob_sum[i * vocab + v] += pv;
                }
            }
            k_models += 1;
        }
    }
    if k_models == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, 0);
    }
    let avg: Vec<f64> = prob_sum.iter().map(|x| x / k_models as f64).collect();
    let targets = data.test_y.to_vec1::<u32>().expect("targets");
    let mut hits = 0usize;
    let mut nll = 0.0f64;
    for (i, &t) in targets.iter().enumerate() {
        let mut best = 0usize;
        let mut best_p = -1.0f64;
        for v in 0..vocab {
            let p = avg[i * vocab + v];
            if p > best_p {
                best_p = p;
                best = v;
            }
        }
        if best as u32 == t {
            hits += 1;
        }
        nll -= (avg[i * vocab + t as usize] + 1e-12).ln();
    }
    (
        100.0 * hits as f64 / targets.len() as f64,
        nll / targets.len() as f64,
        expected_calibration_error(&avg, vocab, &targets, 10),
        k_models,
    )
}

/// Full run: per variant — Task-A pre-train, baselines, unadjusted
/// SG-pCNL chains on Task B, diagnostics, BMA.
pub fn run_experiment(cfg: &Config) -> Report {
    let data_a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
    let data_b = Dataset::new(
        cfg.task_b,
        cfg.k,
        cfg.n_train,
        cfg.n_test,
        cfg.seed.wrapping_add(1),
    );
    let theta0 = init_flat(&cfg.shapes, cfg.init_std, cfg.seed.wrapping_add(2));
    let mut variants = Vec::new();
    for variant in [Variant::Cosine, Variant::Standard] {
        variants.push(run_variant(cfg, variant, &data_a, &data_b, &theta0));
    }
    Report {
        config: cfg.clone(),
        variants,
    }
}

/// One chain's outcome — everything the merged report aggregates.
/// Produced independently (per-chain RNG streams), so Phase 2's
/// `rayon` can run chains in parallel; [`run_variant`] merges them in
/// chain order, which keeps every aggregate — and therefore the
/// report — identical to sequential execution.
struct ChainOutcome {
    loss_chain: Vec<f64>,
    kept: Vec<Tensor>,
    final_state: Tensor,
    /// One series per `coords` entry; empty when the chain kept nothing.
    coord_values: Vec<Vec<f64>>,
    max_scaled_grad: f64,
    max_batch_loss: f64,
    grad_nonfinite: bool,
    exploded: bool,
    exploded_at: Option<usize>,
    radial_max: f64,
    radial_sum: f64,
    radial_n: usize,
    radial_final: f64,
}

/// One unadjusted SG-pCNL chain on Task B. Shares nothing mutable
/// with any other chain: the noise stream and the batch-index stream
/// are both seeded from `c`, as in the sequential version.
fn run_chain(
    cfg: &Config,
    variant: Variant,
    data_b: &Dataset,
    theta_a: &Tensor,
    coords: &[usize],
    scheduler: SgPcnlScheduler,
    c: usize,
) -> ChainOutcome {
    let shapes = &cfg.shapes;
    let p_dim = shapes.flat_size();
    // Two independent streams per chain: noise (SgPcnlState) and
    // batch indices (BatchLoader) — no shared rng state.
    let mut noise_rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(200 + c as u64));
    let mut loader = BatchLoader::new(
        cfg.n_train,
        cfg.batch_size,
        cfg.seed.wrapping_add(400 + c as u64),
    );
    let n = rand_distr::StandardNormal;
    let jitter: Vec<f32> = (0..p_dim)
        .map(|_| {
            let z: f32 = noise_rng.sample(n);
            z * cfg.sigma as f32
        })
        .collect();
    let init = (theta_a + Tensor::from_vec(jitter, vec![p_dim], theta_a.device()).expect("jit"))
        .expect("init = theta_a + noise");
    let mut state = SgPcnlState::new(
        init,
        cfg.sigma,
        scheduler,
        cfg.n_train,
        cfg.batch_size,
        cfg.cyclical_period,
    );
    if cfg.prior_center_at_a {
        state = state.centered_at(theta_a.clone());
    }
    let center = cfg.prior_center_at_a.then_some(theta_a);
    let mut loss_chain: Vec<f64> = Vec::new();
    let mut kept: Vec<Tensor> = Vec::new();
    let mut coord_values: Vec<Vec<f64>> = (0..coords.len())
        .map(|_| Vec::with_capacity((cfg.steps - cfg.burn) / cfg.thin + 1))
        .collect();
    let mut max_scaled_grad = 0.0f64;
    let mut max_batch_loss = 0.0f64;
    let mut grad_nonfinite = false;
    let mut exploded = false;
    let mut exploded_at: Option<usize> = None;
    let mut radial_max = 0.0f64;
    let mut radial_sum = 0.0f64;
    let mut radial_n = 0usize;
    let mut radial_final = 0.0f64;

    for t in 0..cfg.steps {
        let (bx, by) = loader.next(data_b);
        let (batch_loss, sum_grad) = data_b
            .batch_grad(&state.theta, shapes, variant, &bx, &by)
            .expect("mini-batch grad");
        // the scaled g that actually drives the step: (N/B)·Σ
        let scale = (cfg.n_train as f64) / (cfg.batch_size as f64);
        if batch_loss.is_finite() {
            max_batch_loss = max_batch_loss.max(batch_loss);
        }
        match norm(&sum_grad) {
            Ok(gn) if gn.is_finite() => {
                max_scaled_grad = max_scaled_grad.max(gn * scale);
            }
            _ => grad_nonfinite = true, // the Phase-0 spike, as evidence
        }

        state
            .step_unadjusted(&sum_grad, &mut noise_rng)
            .expect("unadjusted step");

        let r = radial_about(&state.theta, center);
        if !r.is_finite() {
            exploded = true;
            exploded_at.get_or_insert(t);
            break;
        }
        radial_max = radial_max.max(r);
        radial_sum += r;
        radial_n += 1;
        radial_final = r;

        if t >= cfg.burn && (t - cfg.burn).is_multiple_of(cfg.thin) {
            kept.push(state.theta.clone());
            if let Some(l) = full_loss(&state.theta, shapes, variant, data_b) {
                loss_chain.push(l);
            } else {
                exploded = true;
                break;
            }
            let flat = state.theta.to_vec1::<f32>().expect("flat");
            for (ci, &coord) in coords.iter().enumerate() {
                coord_values[ci].push(flat[coord] as f64);
            }
        }
    }

    ChainOutcome {
        loss_chain,
        kept,
        final_state: state.theta.clone(),
        coord_values,
        max_scaled_grad,
        max_batch_loss,
        grad_nonfinite,
        exploded,
        exploded_at,
        radial_max,
        radial_sum,
        radial_n,
        radial_final,
    }
}

fn run_variant(
    cfg: &Config,
    variant: Variant,
    data_a: &Dataset,
    data_b: &Dataset,
    theta0: &Tensor,
) -> VariantReport {
    let start = std::time::Instant::now();
    let shapes = &cfg.shapes;
    let p_dim = shapes.flat_size();
    let name = match variant {
        Variant::Cosine => "cosine/nGPT-lite",
        Variant::Standard => "standard softmax",
    };

    // ---- Phase 3 pre-train on Task A -------------------------------
    let (theta_a, trace_a) = momentum_sgd(
        data_a,
        shapes,
        variant,
        theta0.clone(),
        cfg.pretrain_steps,
        cfg.pretrain_lr,
        0.9,
    );
    let acc_a = data_a.accuracy(&theta_a, shapes, variant).expect("acc A");
    let loss_a_first = trace_a[0];
    let loss_a_last = *trace_a.last().unwrap();
    // A pre-train that died is a *baseline* fact: record it, skip the
    // chains (there is no finite θ_A to adapt from), keep the report.
    let exploded_in_pretrain = !loss_a_last.is_finite();

    // ---- Phase 4 baselines -----------------------------------------
    let acc_b_before = data_b
        .accuracy(&theta_a, shapes, variant)
        .expect("acc B before");
    let nll_b_before = data_b.nll(&theta_a, shapes, variant).expect("nll B before");
    let (theta_b_sgd, _) = momentum_sgd(
        data_b,
        shapes,
        variant,
        theta_a.clone(),
        cfg.sgd_steps_on_b,
        cfg.pretrain_lr,
        0.9,
    );
    let acc_b_sgd = data_b
        .accuracy(&theta_b_sgd, shapes, variant)
        .expect("acc B sgd");
    let nll_b_sgd = data_b
        .nll(&theta_b_sgd, shapes, variant)
        .expect("nll B sgd");

    // ---- Phase 4's other deterministic comparator: Adam on B -------
    let (theta_b_adam, _) = adam(
        data_b,
        shapes,
        variant,
        theta_a.clone(),
        cfg.adam_steps_on_b,
        cfg.adam_lr,
    );
    let acc_b_adam = data_b
        .accuracy(&theta_b_adam, shapes, variant)
        .expect("acc B adam");
    let nll_b_adam = data_b
        .nll(&theta_b_adam, shapes, variant)
        .expect("nll B adam");

    // ---- Phase 4's standard sampler: SGLD chains on B --------------
    // Same start distribution as the pCNL chains (θ_A + prior jitter),
    // own RNG/loader streams seeded from cfg.seed + 700 + c.
    let sgld_t0 = std::time::Instant::now();
    let jitter_init = |c: usize| -> Tensor {
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(700 + c as u64));
        let n = rand_distr::StandardNormal;
        let j: Vec<f32> = (0..p_dim)
            .map(|_| {
                let z: f32 = rng.sample(n);
                z * cfg.sigma as f32
            })
            .collect();
        (theta_a.clone() + Tensor::from_vec(j, vec![p_dim], theta_a.device()).expect("jit"))
            .expect("sgld init = theta_a + noise")
    };
    let sgld_outcomes: Vec<crate::task::SgldOutcome> = if exploded_in_pretrain {
        Vec::new() // no finite θ_A to adapt from — same guard as the chains
    } else {
        let center = cfg.prior_center_at_a.then(|| theta_a.clone());
        (0..cfg.chains)
            .map(|c| sgld_chain(cfg, variant, data_b, jitter_init(c), c, center.as_ref()))
            .collect()
    };
    let sgld_seconds = sgld_t0.elapsed().as_secs_f64();

    // ---- Phase 4b: the Metropolis-filtered predecessor -----------
    // Same proposal, same schedule, same thinning as the unadjusted
    // chains — the only difference is the exact full-batch filter,
    // so the cross-R̂ over the two families' loss chains isolates
    // what dropping the filter costs (plan §4 Phase 4b).
    let mh_t0 = std::time::Instant::now();
    let mh_outcomes: Vec<crate::task::MhOutcome> = if exploded_in_pretrain || !cfg.run_mh {
        Vec::new()
    } else {
        let center = cfg.prior_center_at_a.then(|| theta_a.clone());
        (0..cfg.chains)
            .map(|c| {
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(900 + c as u64));
                let n = rand_distr::StandardNormal;
                let j: Vec<f32> = (0..p_dim)
                    .map(|_| {
                        let z: f32 = rng.sample(n);
                        z * cfg.sigma as f32
                    })
                    .collect();
                let init = (theta_a.clone()
                    + Tensor::from_vec(j, vec![p_dim], theta_a.device()).expect("jit"))
                .expect("mh init = theta_a + noise");
                crate::task::mh_pcnl_chain(cfg, variant, data_b, init, c, center.as_ref())
            })
            .collect()
    };
    let mh_seconds = mh_t0.elapsed().as_secs_f64();

    // ---- unadjusted SG-pCNL chains on Task B ------------------------
    let scheduler = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    let stride = (p_dim / 256).max(1);
    let coords: Vec<usize> = (0..p_dim).step_by(stride).collect();

    // Chains share no mutable state (each owns its noise and batch
    // streams), so `rayon` runs them in parallel; the merge below walks
    // outcomes in chain order, preserving the sequential aggregates.
    let outcomes: Vec<ChainOutcome> = if exploded_in_pretrain {
        // no finite θ_A to adapt from — the chains never start
        Vec::new()
    } else {
        use rayon::prelude::*;
        (0..cfg.chains)
            .into_par_iter()
            .map(|c| run_chain(cfg, variant, data_b, &theta_a, &coords, scheduler, c))
            .collect()
    };

    let mut max_scaled_grad = 0.0f64;
    let mut max_batch_loss = 0.0f64;
    let mut grad_nonfinite = false;
    let mut exploded = exploded_in_pretrain;
    let mut exploded_at: Option<usize> = None;
    let mut loss_chains: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let mut coord_chains: Vec<Vec<Vec<f64>>> = vec![Vec::new(); coords.len()];
    let mut kept: Vec<Vec<Tensor>> = Vec::with_capacity(cfg.chains);
    let mut final_states: Vec<Tensor> = Vec::with_capacity(cfg.chains);
    let mut radial_max = 0.0f64;
    let mut radial_sum = 0.0f64;
    let mut radial_n = 0usize;
    let mut radial_final = 0.0f64;

    for oc in outcomes {
        max_scaled_grad = max_scaled_grad.max(oc.max_scaled_grad);
        max_batch_loss = max_batch_loss.max(oc.max_batch_loss);
        grad_nonfinite |= oc.grad_nonfinite;
        if oc.exploded {
            // one exploded chain poisons the run's diagnostics; report it
            exploded = true;
            if exploded_at.is_none() {
                exploded_at = Some(oc.exploded_at.unwrap_or(0));
            }
        }
        radial_max = radial_max.max(oc.radial_max);
        radial_sum += oc.radial_sum;
        radial_n += oc.radial_n;
        if oc.radial_n > 0 {
            radial_final = oc.radial_final;
        }
        for (ci, vals) in oc.coord_values.into_iter().enumerate() {
            coord_chains[ci].push(vals);
        }
        loss_chains.push(oc.loss_chain);
        kept.push(oc.kept);
        final_states.push(oc.final_state);
    }

    let radial_expected = cfg.sigma * (p_dim as f64).sqrt();
    let beta_start = scheduler.beta_at(0);
    let beta_end = scheduler.beta_at(cfg.steps.saturating_sub(1));

    // ---- diagnostics (only over finite chains) ----------------------
    let (ess_loss, ess_med, rhat_loss, rhat_med, rhat_max) = if exploded {
        // keep whatever numbers are finite; the EXPLODED line in the
        // render is the headline, not the diagnostics
        let mut e: Vec<f64> = loss_chains
            .iter()
            .filter(|c| !c.is_empty())
            .map(|c| ess(c))
            .collect();
        let el = median(&mut e);
        let rl = if loss_chains.iter().filter(|c| !c.is_empty()).count() >= 2 {
            gelman_rubin(
                &loss_chains
                    .iter()
                    .filter(|c| !c.is_empty())
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        } else {
            f64::NAN
        };
        (el, f64::NAN, rl, f64::NAN, f64::NAN)
    } else {
        let el = median(&mut loss_chains.iter().map(|c| ess(c)).collect::<Vec<_>>());
        let mut ec: Vec<f64> = coord_chains
            .iter()
            .flat_map(|per_chain| per_chain.iter().map(|c| ess(c)))
            .collect();
        let em = median(&mut ec);
        let rl = gelman_rubin(&loss_chains);
        let mut rc: Vec<f64> = coord_chains.iter().map(|cc| gelman_rubin(cc)).collect();
        rc.retain(|r| r.is_finite());
        let rm = median(&mut rc.clone());
        let rx = rc.iter().cloned().fold(0.0f64, f64::max);
        (el, em, rl, rm, rx)
    };

    // ---- Phase 3: BMA over finite thinned samples -------------------
    let (bma_accuracy, bma_nll, bma_ece, k_models) = bma_metrics(&kept, shapes, variant, data_b);

    // ---- Phase 4: the SGLD comparator's diagnostics + cross-R̂ ------
    let mut sgld_kept_all: Vec<Vec<Tensor>> = Vec::new();
    let mut sgld_loss_chains: Vec<Vec<f64>> = Vec::new();
    let (mut sgld_rad_max, mut sgld_rad_final) = (0.0f64, 0.0f64);
    for oc in sgld_outcomes {
        sgld_rad_max = sgld_rad_max.max(oc.radial_max);
        if oc.radial_final.is_finite() {
            sgld_rad_final = oc.radial_final;
        }
        sgld_loss_chains.push(oc.kept_loss);
        sgld_kept_all.push(oc.kept);
    }
    let (sgld_bma_accuracy, sgld_bma_nll, sgld_bma_ece, sgld_bma_models) =
        bma_metrics(&sgld_kept_all, shapes, variant, data_b);
    let sgld_ess_loss = if sgld_loss_chains.is_empty() {
        f64::NAN
    } else {
        median(&mut sgld_loss_chains.iter().map(|c| ess(c)).collect::<Vec<_>>())
    };
    let sgld_rhat_loss = gelman_rubin(&sgld_loss_chains);
    let rhat_cross = {
        let mut all = loss_chains.clone();
        all.extend(sgld_loss_chains.iter().cloned());
        if all.len() >= 2 {
            gelman_rubin(&all)
        } else {
            f64::NAN
        }
    };

    // ---- Phase 4b: the filtered comparator's diagnostics ----------
    let mut mh_kept_all: Vec<Vec<Tensor>> = Vec::new();
    let mut mh_loss_chains: Vec<Vec<f64>> = Vec::new();
    let (mut mh_accept_sum, mut mh_rad_final, mut mh_steps, mut mh_kept_len) =
        (0usize, 0.0f64, 0usize, 0usize);
    for oc in mh_outcomes {
        mh_accept_sum += oc.accepted;
        mh_steps += oc.steps;
        if oc.radial_final.is_finite() {
            mh_rad_final = oc.radial_final;
        }
        mh_kept_len = oc.kept.len();
        mh_loss_chains.push(oc.kept_loss);
        mh_kept_all.push(oc.kept);
    }
    let mh_accept_rate = if mh_steps == 0 {
        f64::NAN
    } else {
        100.0 * mh_accept_sum as f64 / mh_steps as f64
    };
    let (mh_bma_accuracy, mh_bma_nll, mh_bma_ece, mh_bma_models) =
        bma_metrics(&mh_kept_all, shapes, variant, data_b);
    let mh_ess_loss = if mh_loss_chains.is_empty() {
        f64::NAN
    } else {
        median(&mut mh_loss_chains.iter().map(|c| ess(c)).collect::<Vec<_>>())
    };
    let mh_rhat_loss = gelman_rubin(&mh_loss_chains);
    // the filtered ∪ unfiltered cross-R̂ — Phase 4b's bias probe
    let rhat_cross_mh = {
        let mut all = loss_chains.clone();
        all.extend(mh_loss_chains.iter().cloned());
        if all.len() >= 2 {
            gelman_rubin(&all)
        } else {
            f64::NAN
        }
    };

    let last_state_accuracy = if exploded {
        f64::NAN
    } else {
        final_states
            .iter()
            .map(|t| data_b.accuracy(t, shapes, variant).expect("final acc"))
            .sum::<f64>()
            / final_states.len() as f64
    };

    // ---- Week 2: stream the thinned trajectory to disk (§1) --------
    // Written per chain so only one chain's byte buffer is live at a
    // time; an exploded/poisoned run still saves what it collected.
    let mut saved_files = Vec::new();
    if let Some(dir) = &cfg.save_dir {
        for (c, samples) in kept.iter().enumerate() {
            let (path, _) = save_chain_trajectory(dir, name, c, samples).expect("save trajectory");
            if !path.is_empty() {
                saved_files.push(path);
            }
        }
    }

    VariantReport {
        variant: name,
        p: p_dim,
        acc_a_pretrained: acc_a,
        loss_a_first,
        loss_a_last,
        acc_b_before,
        nll_b_before,
        acc_b_sgd,
        nll_b_sgd,
        acc_b_adam,
        nll_b_adam,
        sgld_radial_final: sgld_rad_final,
        sgld_radial_max: sgld_rad_max,
        sgld_ess_loss,
        sgld_rhat_loss,
        sgld_bma_accuracy,
        sgld_bma_nll,
        sgld_bma_ece,
        sgld_bma_models,
        sgld_seconds,
        rhat_cross,
        mh_accept_rate,
        mh_radial_final: mh_rad_final,
        mh_ess_loss,
        mh_rhat_loss,
        mh_bma_accuracy,
        mh_bma_nll,
        mh_bma_ece,
        mh_bma_models,
        mh_kept_per_chain: mh_kept_len,
        mh_seconds,
        rhat_cross_mh,
        beta_start,
        beta_end,
        steps: cfg.steps,
        exploded,
        exploded_at_step: exploded_at,
        exploded_in_pretrain,
        grad_nonfinite,
        max_scaled_grad_norm: max_scaled_grad,
        max_batch_loss,
        radial_expected,
        radial_final: if exploded { f64::NAN } else { radial_final },
        radial_max,
        radial_mean: if radial_n == 0 {
            f64::NAN
        } else {
            radial_sum / radial_n as f64
        },
        ess_loss,
        ess_median_coords: ess_med,
        rhat_loss,
        rhat_coords_median: rhat_med,
        rhat_coords_max: rhat_max,
        bma_accuracy,
        bma_nll,
        bma_ece,
        last_state_accuracy,
        bma_models: k_models,
        saved_files,
        seconds: start.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_experiment_reports_sane_numbers() {
        let mut cfg = Config::quick();
        // exercise the Week-2 safetensors path end to end
        let dir = std::env::temp_dir().join(format!("pcnl_quick_{}", std::process::id()));
        cfg.save_dir = Some(dir.clone());
        let report = run_experiment(&cfg);
        let text = report.render();
        assert_eq!(report.variants.len(), 2);
        for v in &report.variants {
            // baselines and pre-training are valid regardless of what
            // the unadjusted chain did
            for (label, x) in [
                ("acc_a", v.acc_a_pretrained),
                ("acc_b_before", v.acc_b_before),
                ("acc_b_sgd", v.acc_b_sgd),
                ("nll_b_sgd", v.nll_b_sgd),
                ("acc_b_adam", v.acc_b_adam),
                ("nll_b_adam", v.nll_b_adam),
                ("loss_a_first", v.loss_a_first),
            ] {
                assert!(x.is_finite(), "{} {label} not finite: {x}", v.variant);
            }
            assert!(
                (0.0..=100.0).contains(&v.acc_a_pretrained),
                "{}",
                v.acc_a_pretrained
            );
            assert!((v.radial_expected - v.radial_expected).abs() < 1e-12);
            assert!(
                (v.beta_end..=v.beta_start).contains(&v.beta_end),
                "schedule decays"
            );
            assert!(v.steps == cfg.steps);
            if v.exploded {
                // the Phase-0 outcome for a variant, if it happens:
                // the render must say where, and there must be
                // evidence — either a recorded finite spike or a
                // non-finite gradient — never a silent zero.
                assert!(text.contains("EXPLODED"), "render must flag it:\n{text}");
                assert!(
                    v.max_scaled_grad_norm > 0.0 || v.grad_nonfinite,
                    "explosion without gradient evidence: max {} grad_nonfinite {}",
                    v.max_scaled_grad_norm,
                    v.grad_nonfinite
                );
            } else {
                assert!(
                    v.ess_loss > 0.0 && v.ess_loss.is_finite(),
                    "ESS {}",
                    v.ess_loss
                );
                assert!(
                    v.rhat_loss.is_finite() && v.rhat_loss > 0.5,
                    "Rhat {}",
                    v.rhat_loss
                );
                assert!(
                    v.radial_final.is_finite() && v.radial_final > 0.0,
                    "radial {}",
                    v.radial_final
                );
                assert!(v.bma_models > 0, "BMA needs finite samples");
                assert!((0.0..=100.0).contains(&v.bma_accuracy));
                assert!(
                    (0.0..=1.0).contains(&v.bma_ece),
                    "ECE outside [0,1]: {}",
                    v.bma_ece
                );
                // SGLD comparator: finite, in-range diagnostics, and the
                // sampler stays in its stationary ball (‖θ‖ ≈ σ√P).
                assert!(
                    v.sgld_bma_models > 0,
                    "SGLD BMA needs finite samples, got {}",
                    v.sgld_bma_models
                );
                assert!(
                    (0.0..=100.0).contains(&v.sgld_bma_accuracy),
                    "sgld bma acc {}",
                    v.sgld_bma_accuracy
                );
                assert!(v.sgld_bma_nll.is_finite(), "sgld bma nll");
                assert!(
                    v.sgld_radial_final.is_finite() && v.sgld_radial_final > 0.0,
                    "sgld radial {}",
                    v.sgld_radial_final
                );
                assert!(
                    v.sgld_radial_max <= 3.0 * v.radial_expected + 1e-9,
                    "SGLD left the stationary ball: {} vs 3x {}",
                    v.sgld_radial_max,
                    v.radial_expected
                );
                assert!(
                    v.sgld_ess_loss.is_finite() && v.sgld_ess_loss > 0.0,
                    "sgld ESS {}",
                    v.sgld_ess_loss
                );
                assert!(
                    v.sgld_rhat_loss.is_finite() && v.sgld_rhat_loss > 0.5,
                    "sgld Rhat {}",
                    v.sgld_rhat_loss
                );
                // Phase 4's coupling diagnostic: cross-Rhat over the
                // pCNL and SGLD loss chains (defined when both kept).
                assert!(
                    v.rhat_cross.is_finite() && v.rhat_cross > 0.5,
                    "cross-Rhat {}",
                    v.rhat_cross
                );
                // The Phase-4b filtered comparator, when enabled:
                // ran, kept samples, accepted *some* proposals, and
                // agrees with the unadjusted chains across algorithms.
                if v.mh_kept_per_chain > 0 {
                    assert!(
                        (0.0..=100.0).contains(&v.mh_accept_rate),
                        "mh accept {}",
                        v.mh_accept_rate
                    );
                    assert!(v.mh_bma_models > 0, "MH BMA needs finite samples");
                    assert!(
                        (0.0..=100.0).contains(&v.mh_bma_accuracy),
                        "mh bma acc {}",
                        v.mh_bma_accuracy
                    );
                    assert!(v.mh_bma_nll.is_finite(), "mh bma nll");
                    assert!(
                        v.mh_radial_final.is_finite() && v.mh_radial_final > 0.0,
                        "mh radial {}",
                        v.mh_radial_final
                    );
                    assert!(
                        v.rhat_cross_mh.is_finite() && v.rhat_cross_mh > 0.5,
                        "mh cross-Rhat {}",
                        v.rhat_cross_mh
                    );
                }
                // The plan's stability claim is *nGPT's*: the cosine
                // variant must hold ‖θ‖ ≈ σ√P. The standard variant's
                // radial behaviour is exactly what Phase 0 measures —
                // random walks included — so it is reported, not
                // asserted.
                if v.variant.contains("nGPT") {
                    assert!(
                        (v.radial_final - v.radial_expected).abs() / v.radial_expected < 0.5,
                        "{} radial {} vs σ√P {}",
                        v.variant,
                        v.radial_final,
                        v.radial_expected
                    );
                }
            }
            // every reported trajectory file exists and loads back as
            // `[P]`-shaped f32 samples (the §1 persistence contract)
            for p in &v.saved_files {
                let back = crate::persist::load_trajectory(std::path::Path::new(p))
                    .expect("load saved trajectory");
                assert!(!back.is_empty(), "{p} empty");
                assert!(back.iter().all(|row| row.len() == v.p), "{p}: row != P");
            }
        }
        // cleanup the persisted artifacts
        for v in &report.variants {
            for p in &v.saved_files {
                let _ = std::fs::remove_file(p);
            }
        }
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn ece_measures_confidence_gap() {
        // Two samples, both predicted class 0 with confidence 0.9:
        // both correct → |0.9 − 1.0| = 0.1; both wrong → |0.9 − 0| = 0.9.
        let probs = [0.9, 0.1, 0.9, 0.1];
        let right = expected_calibration_error(&probs, 2, &[0, 0], 10);
        assert!((right - 0.1).abs() < 1e-12, "{right}");
        let wrong = expected_calibration_error(&probs, 2, &[1, 1], 10);
        assert!((wrong - 0.9).abs() < 1e-12, "{wrong}");
        assert!(expected_calibration_error(&probs, 2, &[], 10).is_nan());
    }

    #[test]
    fn report_renders_every_headline_field() {
        // quick_mh, so the filtered comparator's line (and the
        // acceptance-rate field) is exercised too
        let cfg = Config::quick_mh();
        let text = run_experiment(&cfg).render();
        for needle in [
            "SG-pCNL (unadjusted)",
            "pre-train A",
            "B baselines",
            "beta",
            "ECE",
            "Adam-on-B",
            "SGLD (baseline)",
            "cross-Rhat",
            "MH-pCNL (filtered)",
            "accept",
        ] {
            assert!(text.contains(needle), "missing {needle}:\n{text}");
        }
    }
}
