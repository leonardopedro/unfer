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
use crate::manifest::{
    CodeId, DataId, MetricRecord, ModelId, Phase, RunManifest, SamplerId, TargetId, WorkId,
};
use crate::model::{Shapes, Variant, init_flat};
use crate::normalize::NormGrad;
use crate::persist::save_chain_trajectory;
use crate::sampler::{SgPcnlScheduler, SgPcnlState, norm};
use crate::target::{ClassifierPotential, Potential};
use crate::task::{BatchLoader, Dataset, PRETRAIN_CLIP, Task, adam, momentum_sgd, sgld_chain};
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
    /// Use Adam (at `adam_lr`) instead of heavy-ball SGD (`pretrain_lr`,
    /// momentum 0.9) for the Task-A pre-train.
    ///
    /// Off by default: every report the doc quotes uses the plan's
    /// momentum-SGD pre-train.  Switched on, it is the *tuned pre-train
    /// control* for the standard-softmax baseline, whose SGD pre-train at
    /// lr 0.6 diverges (loss 851) before the sampler is ever started —
    /// which makes the Phase-0 comparison between the variants test the
    /// pre-train's stability as much as the architecture.
    pub pretrain_adam: bool,
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
    /// Polynomial-decay exponent, `β_t = β₀(1+γt)^{−κ}`.
    ///
    /// The mobility window is `(1/4, 1/2]`, because the decreasing-step
    /// conditions are on the coefficient that multiplies the gradient,
    /// `h_t = β_t²σ²/2` — not on `β_t` (plan §4.3, corrected
    /// 2026-11-03). `κ = 0.55`, the plan's original default, makes
    /// `Σβ_t² → β₀²/(γ(2κ−1)) < ∞`: finite total mobility, a chain that
    /// freezes where it is. The default is `1/2`, the borderline case;
    /// every report states the budget it computed and warns when the
    /// exponent is outside the window.
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
    /// Length of the **likelihood-temperature ramp**: the drift is
    /// `λ_t·β_t²σ²/2·g_t` with `λ_t = min(1, t/anneal_steps)`.
    ///
    /// `0` (the default) is the published kernel at `λ = 1` at every step.
    /// A positive value is §6.4.1's reachability arms race made explicit: the
    /// chain starts at the prior (`λ₀ = 0`) and the likelihood is switched on
    /// over `anneal_steps` steps, so the direction the chain carries into the
    /// adaptation is not one Haar draw's.  The target after the ramp is the
    /// same `N(0, σ²I)·L`, so the kept draws (`t ≥ burn`) are at `λ = 1`
    /// whenever `anneal_steps ≤ burn`; a longer ramp means the kept draws are
    /// still tempered, and the report says so.
    pub anneal_steps: usize,
    /// Likelihood temperature `λ` on the potential: every arm samples
    /// `π_λ ∝ N(0, σ²I)·exp(−λ·NL)`, so `λ = 1` (the default) is the
    /// published target, `λ > 1` is a *cold* posterior and `λ < 1` hot.
    ///
    /// §6.5.5's information-to-noise ratio `r = β_tσ‖g_t‖/(2√P)` grows with
    /// `λ` at fixed `‖g‖` (sublinearly in practice — a sharper target also
    /// shrinks the gradient), which is why cold was the untested direction:
    /// §6.5.4's ladder only went hot (`λ ≤ 1`) and found every rung worse.
    /// The report prints the measured `r` whenever the target or the ramp is
    /// non-default.
    /// All three arms (pCNL, SGLD, MH) receive the same `λ`; the MH filter's
    /// acceptance ratio is tempered in step so it stays exact for `π_λ`, and
    /// the kept losses stay raw.
    pub potential_scale: f64,
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
            pretrain_adam: false,
            chains: 4,
            steps: 10_000,
            burn: 5_000,
            thin: 10,
            batch_size: 64,
            sigma: 0.35,
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.5,
            sgd_steps_on_b: 1_000,
            adam_steps_on_b: 1_000,
            adam_lr: 0.05,
            init_std: 0.1,
            save_dir: None,
            cyclical_period: 0,
            run_mh: true,
            prior_center_at_a: false,
            anneal_steps: 0,
            potential_scale: 1.0,
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

/// R0.5 — the kept window of one trace, and whether a stationary diagnostic
/// may be read off it.
///
/// `stationary` is exact, not a tolerance: the step size must be identical at
/// both ends of the window and the likelihood ramp must have finished before it
/// began. The default `κ = 0.5` schedule decays over any window a run can
/// reach, so the default answer is `false` — and that is the point. ESS and R̂
/// over a decaying-step trace describe *that window*, not a stationary chain;
/// the report now says so at every use.
#[derive(Clone, Debug)]
pub struct Window {
    pub start: usize,
    pub end: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    /// Target temperature at `start` (the ramp's value is fixed after it ends).
    pub lambda: f64,
    pub stationary: bool,
}

impl Window {
    pub fn of(cfg: &Config, sched: &SgPcnlScheduler) -> Self {
        let start = cfg.burn.min(cfg.steps);
        let end = cfg.steps;
        let beta_start = sched.beta_at(start);
        let beta_end = sched.beta_at(end.saturating_sub(1).max(start));
        let lambda = lambda_at(cfg, start);
        let stationary =
            cfg.cyclical_period == 0 && beta_start == beta_end && lambda == cfg.potential_scale;
        Self {
            start,
            end,
            beta_start,
            beta_end,
            lambda,
            stationary,
        }
    }

    /// One line, printed under every variant: which steps the diagnostics
    /// above were computed on and what they are allowed to mean.
    pub fn render(&self, thin: usize, chains: usize) -> String {
        let draws = self
            .end
            .saturating_sub(self.start)
            .checked_div(thin)
            .unwrap_or(0);
        let verdict = if self.stationary {
            "stationary (constant target and step size)".to_string()
        } else {
            "NONSTATIONARY: the ESS/Rhat above describe this decaying-step window, \
             not a stationary chain"
                .to_string()
        };
        format!(
            "  window: kept t={}..{} of {} steps (thin {thin}, {draws} draws/chain, {chains} chains) \
             | beta {:.4}->{:.4} lambda {:.2} | {verdict}\n",
            self.start, self.end, self.end, self.beta_start, self.beta_end, self.lambda
        )
    }
}

/// `λ_t` of the likelihood ramp: `λ·min(1, t/anneal)` (`SgPcnlState::lambda_at`).
fn lambda_at(cfg: &Config, step: usize) -> f64 {
    if cfg.anneal_steps == 0 {
        return cfg.potential_scale;
    }
    cfg.potential_scale * (step as f64 / cfg.anneal_steps as f64).min(1.0)
}

/// R0.5 — warm-up, likelihood ramp and kept window as separate records.
///
/// The three overlap by design (the ramp usually starts with the warm-up), and
/// that is why each carries its own `lambda`/`beta` pair: a phase is a fact
/// about what the kernel was doing over that range, not a partition of the run.
pub fn phases(cfg: &Config, sched: &SgPcnlScheduler) -> Vec<Phase> {
    let mut out = Vec::with_capacity(3);
    if cfg.burn > 0 && cfg.steps > 0 {
        let end = cfg.burn.min(cfg.steps);
        out.push(Phase {
            name: "warmup".into(),
            start: 0,
            end,
            lambda: lambda_at(cfg, end),
            beta_start: sched.beta_at(0),
            beta_end: sched.beta_at(end - 1),
            stationary: false,
        });
    }
    if cfg.anneal_steps > 0 && cfg.steps > 0 {
        let end = cfg.anneal_steps.min(cfg.steps);
        out.push(Phase {
            name: "likelihood-ramp".into(),
            start: 0,
            end,
            lambda: lambda_at(cfg, end),
            beta_start: sched.beta_at(0),
            beta_end: sched.beta_at(end - 1),
            stationary: false,
        });
    }
    let w = Window::of(cfg, sched);
    out.push(Phase {
        name: "kept".into(),
        start: w.start,
        end: w.end,
        lambda: w.lambda,
        beta_start: w.beta_start,
        beta_end: w.beta_end,
        stationary: w.stationary,
    });
    out
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
    /// Mean over steps of the step's information-to-noise ratio,
    /// `h_t·λ_t·‖g_t‖ / (β_t σ √P)` — the drift norm against the pCN
    /// noise norm (§6.5.5).  `r ≈ 0.05` means the direction is re-drawn
    /// from the prior ~20× faster than the likelihood can steer it.
    pub information_ratio: f64,
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
    /// R0.5 — the steps the four diagnostics above were computed on, and
    /// whether that window is a stationary chain (default: it is not).
    pub window: Window,
    /// R0.6 — hash of the frozen Task-A anchor `θ_A` this arm adapted from.
    pub anchor_hash: u64,
    /// R0.6 — hash of the materialised Task-B data the potential sums over.
    pub data_hash: u64,
    pub bma_accuracy: f64,
    pub bma_nll: f64,
    /// Phase 4's Expected Calibration Error of the BMA predictive
    /// distribution (10 equal-width confidence bins on \[0,1\]).
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
    /// R0.6 — identity of the run: revision, data and anchor hashes, the
    /// evaluated schedule, the phase window, cost and non-finite events.
    pub manifest: RunManifest,
}

impl Report {
    /// R0.6 — write `{dir}/{stem}.manifest.json`, the identity half of a saved
    /// run. The `.safetensors` trajectories say *what* was kept; the manifest
    /// says what the numbers are, without which they are not checkable.
    pub fn write_manifest(&self, dir: &std::path::Path, stem: &str) -> std::io::Result<PathBuf> {
        self.manifest.write(dir, stem)
    }

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
        // Same convention for the pre-train optimizer: named only when it
        // is not the plan's momentum SGD, so quoted reports stay identical.
        let opt = if self.config.pretrain_adam {
            ", pretrain_opt=adam"
        } else {
            ""
        };
        s.push_str(&format!(
            "SG-pCNL (unadjusted) x nGPT-lite toy run (seed {}, sigma={}, beta0={}, kappa={}, \
             gamma={}, batch={}, chains={}, steps={}, burn={}, thin={}, taskA=(a+b) mod {}, \
             taskB={}, n_train={}, pretrain={}{}{})\n",
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
            prior,
            opt
        ));
        // A tempered target is part of the run's identity too: the kept
        // losses are raw, so a reader cannot recover λ from them.
        if self.config.potential_scale != 1.0 {
            s.push_str(&format!(
                "  target: likelihood temperature λ = {} — π ∝ N(0,σ²I)·exp(−λ·NL) \
                 ({} posterior); r = βσ‖g‖/(2√P) grows with λ, and the measured r is \
                 printed on each SG-pCNL line\n",
                self.config.potential_scale,
                if self.config.potential_scale > 1.0 {
                    "cold"
                } else {
                    "hot"
                }
            ));
        }
        // The annealed start is part of the run's identity: a reader must
        // be able to tell whether the kept draws are at λ = 1 or mid-ramp.
        if self.config.anneal_steps > 0 {
            let lands_before_burn = self.config.anneal_steps <= self.config.burn;
            s.push_str(&format!(
                "  anneal: likelihood temperature λ_t = min(1, t/{}) (λ₀ = 0); kept draws {} \n",
                self.config.anneal_steps,
                if lands_before_burn {
                    "are at λ = 1"
                } else {
                    "are still tempered — not the λ = 1 posterior"
                }
            ));
        }
        // §4.3 (corrected): the schedule's mobility budget belongs in
        // the artifact, not only in the prose — `Σβ_t²` over the run and
        // whether the schedule has any left afterwards. A finite budget
        // is the freeze: the chain cannot travel further than it allows.
        if self.config.cyclical_period > 0 {
            s.push_str("  schedule: cyclical cosine — Σβ_t² grows without bound (no freeze)\n");
        } else {
            let sched = SgPcnlScheduler {
                beta_0: self.config.beta_0,
                gamma: self.config.gamma,
                kappa: self.config.kappa,
            };
            let used = sched.partial_mobility(self.config.steps);
            match sched.total_mobility() {
                None => s.push_str(&format!(
                    "  schedule: mobility Σβ_t² = {used:.1} over {steps} steps, budget unbounded \
                     (kappa = {kappa} <= 1/2)\n",
                    steps = self.config.steps,
                    kappa = self.config.kappa
                )),
                Some(budget) => s.push_str(&format!(
                    "  schedule: mobility Σβ_t² = {used:.1} of a finite budget {budget:.1} \
                     (kappa = {kappa} > 1/2), {left:.1} unspent — the chain freezes; these numbers \
                     describe the kernel at the beta's visited, not convergence to the posterior\n",
                    kappa = self.config.kappa,
                    left = sched.tail_mobility(self.config.steps).unwrap_or(0.0)
                )),
            }
        }
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
            // R0.5: which steps the diagnostics above were computed on. Printed
            // unconditionally — a decaying-step trace is the default, and its
            // ESS/Rhat are not stationary-chain diagnostics.
            s.push_str(&v.window.render(self.config.thin, self.config.chains));
            // §6.5.5's diagnostic, printed only when the target is not the
            // published one, so every λ = 1 render stays byte-identical.
            if self.config.potential_scale != 1.0 || self.config.anneal_steps > 0 {
                s.push_str(&format!(
                    "  information-to-noise r = {:.3} (mean h_t·λ_t·‖g_t‖ / β_tσ√P; \
                     r ≪ 1 means the direction is prior-drawn)\n",
                    v.information_ratio
                ));
            }
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

/// Median of the finite values; `NaN` when there are none.
///
/// Diagnostics now report `NaN` for frozen or non-finite traces
/// (`diag::Movement`), so this must not be able to panic on them: a single
/// stuck coordinate used to make every median `NaN`-poisoned *and* the sort
/// fail, which turned "this coordinate never moved" into a crash in the quick
/// report.  `total_cmp` orders NaN last, so the finite prefix is the input.
fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.iter().take_while(|x| x.is_finite()).count();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Phase 4 — Expected Calibration Error over a predictive distribution:
/// 10 equal-width confidence bins on \[0,1\],
/// `ECE = Σ_b (|b|/n)·|acc(b) − conf(b)|`.
/// `probs` is row-major `[n × vocab]`, already averaged over models.
pub fn expected_calibration_error(
    probs: &[f64],
    vocab: usize,
    targets: &[u32],
    n_bins: usize,
) -> f64 {
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

/// Row-wise softmax of one `[K]` logit row — shared with
/// `examples/readout_compare.rs` so the readout table and the report's
/// BMA numbers are computed by the *same* code.
pub fn softmax_rows(logits: &[f64]) -> Vec<f64> {
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

/// BMA metrics *and* the per-draw series behind them.
///
/// The four scalars are the ones the reports quote.  The `*_chain` series are
/// per chain, per kept draw, of that draw's *own* readout — the linear
/// functionals whose MCSE is `sd/√ESS`.  They are not the BMA scalars: BMA
/// accuracy is the accuracy of the averaged distribution, a ratio of averages
/// with no closed-form variance, so a `sd/√ESS` attached to it would be a
/// different estimand's error bar wearing its name.  Quoting both, labelled, is
/// the only honest option.
pub struct BmaDetail {
    /// Accuracy %, NLL, ECE, Brier of the averaged predictive distribution.
    pub acc: f64,
    pub nll: f64,
    pub ece: f64,
    pub brier: f64,
    pub models: usize,
    /// Single-draw accuracy % and Brier, `[chain][draw]`.
    pub acc_chain: Vec<Vec<f64>>,
    pub brier_chain: Vec<Vec<f64>>,
}

/// Bayesian model averaging over one or more chains' thinned samples, with the
/// per-draw series a predictive MCSE needs. One implementation shared by the
/// SG-pCNL chains and the SGLD comparator, so the two numbers in the report are
/// computed by the *same* code.
pub fn bma_detail(
    chains: &[Vec<Tensor>],
    shapes: &Shapes,
    variant: Variant,
    data: &Dataset,
) -> BmaDetail {
    let n_test = data.test_y.dim(0).expect("n_test");
    let vocab = shapes.vocab;
    let targets = data.test_y.to_vec1::<u32>().expect("targets");
    let mut prob_sum = vec![0.0f64; n_test * vocab];
    let mut acc_chain: Vec<Vec<f64>> = chains.iter().map(|_| Vec::new()).collect();
    let mut brier_chain: Vec<Vec<f64>> = chains.iter().map(|_| Vec::new()).collect();
    let mut k_models = 0usize;
    for (ci, chain) in chains.iter().enumerate() {
        for sample in chain {
            if radial(sample).is_finite()
                && let Ok(logits) = crate::model::forward(sample, shapes, variant, &data.test_x)
                && let Ok(rows) = logits.to_vec2::<f32>()
            {
                let mut hits = 0usize;
                let mut brier = 0.0f64;
                for (i, row) in rows.iter().enumerate() {
                    let p = softmax_rows(&row.iter().map(|x| *x as f64).collect::<Vec<_>>());
                    let t = targets[i] as usize;
                    // Strictly proper, the same `readout::brier` the readout
                    // table scores: Σ_v (p_v − 1[y])².
                    brier += crate::readout::brier(&p, t);
                    if p.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .is_some_and(|(v, _)| v == t)
                    {
                        hits += 1;
                    }
                    for (v, pv) in p.into_iter().enumerate() {
                        prob_sum[i * vocab + v] += pv;
                    }
                }
                acc_chain[ci].push(100.0 * hits as f64 / targets.len() as f64);
                brier_chain[ci].push(brier / targets.len() as f64);
                k_models += 1;
            }
        }
    }
    if k_models == 0 {
        return BmaDetail {
            acc: f64::NAN,
            nll: f64::NAN,
            ece: f64::NAN,
            brier: f64::NAN,
            models: 0,
            acc_chain,
            brier_chain,
        };
    }
    let avg: Vec<f64> = prob_sum.iter().map(|x| x / k_models as f64).collect();
    let mut hits = 0usize;
    let mut nll = 0.0f64;
    let mut brier = 0.0f64;
    for (i, &t) in targets.iter().enumerate() {
        let row = &avg[i * vocab..(i + 1) * vocab];
        let mut best = 0usize;
        let mut best_p = -1.0f64;
        for (v, &p) in row.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = v;
            }
        }
        if best as u32 == t {
            hits += 1;
        }
        nll -= (row[t as usize] + 1e-12).ln();
        brier += crate::readout::brier(row, t as usize);
    }
    BmaDetail {
        acc: 100.0 * hits as f64 / targets.len() as f64,
        nll: nll / targets.len() as f64,
        ece: expected_calibration_error(&avg, vocab, &targets, 10),
        brier: brier / targets.len() as f64,
        models: k_models,
        acc_chain,
        brier_chain,
    }
}

/// `(accuracy %, NLL, ECE, K models used)` — [`bma_detail`] without the series.
pub fn bma_metrics(
    chains: &[Vec<Tensor>],
    shapes: &Shapes,
    variant: Variant,
    data: &Dataset,
) -> (f64, f64, f64, usize) {
    let d = bma_detail(chains, shapes, variant, data);
    (d.acc, d.nll, d.ece, d.models)
}

/// Full run: per variant — Task-A pre-train, baselines, unadjusted
/// FNV-1a over a parameter vector's little-endian bytes — the identity
/// `VariantReport::anchor_hash` and `DataId::anchor_hash` publish.
pub fn anchor_hash(theta: &Tensor) -> u64 {
    let mut h = crate::manifest::Fnv::new();
    for x in theta
        .flatten_all()
        .expect("flatten anchor")
        .to_vec1::<f32>()
        .expect("anchor")
    {
        h.write(&x.to_le_bytes());
    }
    h.finish()
}

/// The frozen objects the ambient report adapts from: `D_A`, `D_B`, `θ₀` and
/// the Task-A pre-train that produces `θ_A`.
///
/// They exist as one struct because an arm that claims to *match* the ambient
/// anchor has to reproduce all four, and re-deriving the seeds in a second
/// place is exactly how those two runs drift apart.  R1's subspace example did
/// that: it built `D_B` from `seed` instead of `seed + 1` and used the
/// quick-config shapes, so its "anchor" was a different model at a different
/// point (7.0 % on B where the ambient anchor scores ~12.5 %), and no number in
/// its table could be read beside an ambient one.
pub struct AnchorSetup {
    pub data_a: Dataset,
    pub data_b: Dataset,
    pub theta0: Tensor,
}

/// The three seed offsets an anchor is defined by, in one place.
///
/// They are a function and not three call sites because the anchor's identity
/// is the triple: an arm that re-derives `seed + 1` inline is one refactor away
/// from drawing `D_B` from `seed`, which is the mistake `AnchorSetup`'s own
/// doc records.  `PotentialAnchor` uses this too, so the DT anchor cannot drift
/// from the classifier's seeds.
pub fn anchor_seeds(seed: u64) -> (u64, u64, u64) {
    (seed, seed.wrapping_add(1), seed.wrapping_add(2))
}

/// The frozen objects an anchor needs, for **any** potential: the two draws,
/// the start point, and a pre-train that does not care which likelihood it is
/// descending on.
///
/// `AnchorSetup` is the classifier-typed form of this — it carries `Dataset`s
/// because every ambient path (`run_variant`, `bma`, the SGLD twin) takes them
/// concretely.  This type is what the DT arm uses: `AnchorSetup`'s seeds, its
/// `anchor_hash`, and its clip/momentum/trace contract, with
/// `policy_model::DtPotential` in place of the classifier.  Duplicating the
/// seeds or the update rule there instead would produce an anchor that is not
/// comparable with the ambient one while looking like it is.
///
/// The optimizer is *not* the potential's business, and neither is the
/// derivative convention: [`Self::pretrain`] takes a [`NormGrad`] and the
/// caller states it.  Historically that convention is `Stopped` (see
/// `task::momentum_sgd`), which is why the equivalence test below passes
/// `NormGrad::Stopped` — a generic pre-train that silently used `Exact` would
/// move the anchor by a different amount and no table would show it.
pub struct PotentialAnchor<P: Potential> {
    pub data_a: P,
    pub data_b: P,
    pub theta0: Tensor,
}

impl<P: Potential> PotentialAnchor<P> {
    pub fn new(data_a: P, data_b: P, theta0: Tensor) -> Self {
        Self {
            data_a,
            data_b,
            theta0,
        }
    }

    /// Heavy-ball SGD with the ambient contract: `v ← μ·v − lr·g`, `θ ← θ + v`,
    /// gradient clipped to [`PRETRAIN_CLIP`], loss trace kept, a non-finite loss
    /// ends the run and the trace is left as the evidence.
    ///
    /// This is [`momentum_sgd`] with the likelihood replaced, so an arm cannot
    /// “pre-train” by a different rule and still call its anchor ambient.  The
    /// test below pins it bitwise against `momentum_sgd` on the classifier.
    pub fn pretrain(
        &self,
        steps: usize,
        lr: f64,
        momentum: f64,
        norm: NormGrad,
    ) -> (Tensor, Vec<f64>) {
        let mut theta = self.theta0.clone();
        let mut vel =
            Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device()).expect("cpu");
        let mut trace = Vec::with_capacity(steps);
        for _ in 0..steps {
            let loss = self.data_a.mean_loss(&theta);
            trace.push(loss);
            if !loss.is_finite() {
                break;
            }
            let Some(grad) = self.data_a.mean_grad_with(&theta, norm) else {
                break;
            };
            let g_norm = grad
                .sqr()
                .and_then(|s| s.sum_all())
                .and_then(|s| s.sqrt())
                .and_then(|s| s.to_scalar::<f32>())
                .map(|v| v as f64)
                .unwrap_or(f64::NAN);
            let grad = if g_norm.is_finite() && g_norm > PRETRAIN_CLIP {
                grad.affine(PRETRAIN_CLIP / g_norm, 0.0).expect("clip")
            } else {
                grad
            };
            vel = (vel.affine(momentum, 0.0).expect("v") - grad.affine(lr, 0.0).expect("g"))
                .expect("v'");
            theta = (&theta + &vel).expect("theta'");
        }
        (theta, trace)
    }

    /// [`anchor_hash`] of the pre-trained point — the same function the ambient
    /// report publishes, not a second hash for a second model.
    pub fn pretrained_hash(&self, steps: usize, lr: f64, momentum: f64, norm: NormGrad) -> u64 {
        let (theta, _) = self.pretrain(steps, lr, momentum, norm);
        anchor_hash(&theta)
    }
}

impl PotentialAnchor<ClassifierPotential> {
    /// The classifier anchor as a potential (the pre-train half of
    /// [`AnchorSetup`]).
    pub fn classifier(setup: AnchorSetup, shapes: Shapes, variant: Variant) -> Self {
        Self::new(
            ClassifierPotential::new(setup.data_a, shapes, variant),
            ClassifierPotential::new(setup.data_b, shapes, variant),
            setup.theta0,
        )
    }
}

impl AnchorSetup {
    pub fn new(cfg: &Config) -> Self {
        let (seed_a, seed_b, seed_theta) = anchor_seeds(cfg.seed);
        Self {
            data_a: Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, seed_a),
            // Task B gets its own draw: sharing `seed` with Task A would make
            // the two datasets correlated in a way the ambient report never
            // has, and its `DataId::seed` records `cfg.seed` for both.
            data_b: Dataset::new(cfg.task_b, cfg.k, cfg.n_train, cfg.n_test, seed_b),
            theta0: init_flat(&cfg.shapes, cfg.init_std, seed_theta),
        }
    }

    /// Phase 3's pre-train: heavy-ball SGD at `pretrain_lr` on Task A, or Adam
    /// when the config asks for the stable optimizer.  Returns `(θ_A, trace)`.
    pub fn pretrain(&self, cfg: &Config, variant: Variant) -> (Tensor, Vec<f64>) {
        if cfg.pretrain_adam {
            adam(
                &self.data_a,
                &cfg.shapes,
                variant,
                self.theta0.clone(),
                cfg.pretrain_steps,
                cfg.adam_lr,
            )
        } else {
            momentum_sgd(
                &self.data_a,
                &cfg.shapes,
                variant,
                self.theta0.clone(),
                cfg.pretrain_steps,
                cfg.pretrain_lr,
                0.9,
            )
        }
    }
}

/// SG-pCNL chains on Task B, diagnostics, BMA.
pub fn run_experiment(cfg: &Config) -> Report {
    let setup = AnchorSetup::new(cfg);
    let data_b = &setup.data_b;
    let mut variants = Vec::new();
    for variant in [Variant::Cosine, Variant::Standard] {
        variants.push(run_variant(cfg, variant, &setup));
    }

    // ---- R0.6: the run's identity, captured once beside the draws ------
    let sched = SgPcnlScheduler {
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
    };
    let mut manifest = RunManifest::new(
        CodeId::capture(),
        ModelId {
            variant: variants
                .iter()
                .map(|v| v.variant)
                .collect::<Vec<_>>()
                .join(" | "),
            p: cfg.shapes.flat_size(),
            dtype: "f32".into(),
            device: "cpu".into(),
            // The forward/backward this crate evaluates: `fro-stopped` is the
            // labelled surrogate from R0.1, not the exact normalised map.
            normalization: "fro-stopped/v1".into(),
        },
        DataId {
            task: format!("{:?}", cfg.task_b),
            k: cfg.k,
            n_train: cfg.n_train,
            n_test: cfg.n_test,
            seed: cfg.seed,
            hash: data_b.content_hash(),
            anchor_hash: variants.first().map(|v| v.anchor_hash),
        },
    );
    manifest.target = TargetId {
        lambda: cfg.potential_scale,
        prior: if cfg.prior_center_at_a {
            "N(theta_A, sigma^2 I)".into()
        } else {
            "N(0, sigma^2 I)".into()
        },
        anneal_steps: cfg.anneal_steps,
    };
    manifest.sampler = SamplerId {
        kernel: "sg-pcnl-unadjusted + sgld + mh-pcnl".into(),
        beta_0: cfg.beta_0,
        gamma: cfg.gamma,
        kappa: cfg.kappa,
        beta_floor: SgPcnlScheduler::BETA_FLOOR,
        sigma: cfg.sigma,
        steps: cfg.steps,
        burn: cfg.burn,
        thin: cfg.thin,
        chains: cfg.chains,
        batch_size: cfg.batch_size,
        cyclical_period: cfg.cyclical_period,
    };
    manifest.phases = phases(cfg, &sched);
    // The ambient engine's proposal is isotropic: its covariances are the
    // prior's `sigma^2 I`, so there is no adapted metric to freeze here. A
    // sphere chain reports its own record through `SpherePcn::metric_record`.
    manifest.metric = MetricRecord::isotropic();
    // One potential evaluation (forward + backward on one batch) per step per
    // chain per arm: SG-pCNL, SGLD, and MH-pCNL when enabled, for each variant.
    let evals = variants.len() as u64
        * (2 + u64::from(cfg.run_mh))
        * cfg.steps as u64
        * cfg.chains as u64
        * cfg.batch_size as u64;
    manifest.work = WorkId {
        forward_evals: evals,
        backward_evals: evals,
        wall_seconds: variants
            .iter()
            .map(|v| {
                v.seconds
                    + if v.mh_seconds.is_finite() {
                        v.mh_seconds
                    } else {
                        0.0
                    }
            })
            .sum(),
        peak_mem_mb: WorkId::peak_rss_mb(),
        nonfinite_events: variants.iter().filter(|v| v.grad_nonfinite).count() as u64,
    };
    Report {
        config: cfg.clone(),
        variants,
        manifest,
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
    information_ratio: f64,
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
    state.anneal_steps = cfg.anneal_steps;
    state.potential_scale = cfg.potential_scale;
    let center = cfg.prior_center_at_a.then_some(theta_a);
    let mut loss_chain: Vec<f64> = Vec::new();
    let mut kept: Vec<Tensor> = Vec::new();
    let mut coord_values: Vec<Vec<f64>> = (0..coords.len())
        .map(|_| Vec::with_capacity((cfg.steps - cfg.burn) / cfg.thin + 1))
        .collect();
    let mut max_scaled_grad = 0.0f64;
    let mut ratio_sum = 0.0f64;
    let mut ratio_n = 0usize;
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
                // §6.5.5: the drift norm against the pCN noise norm, with
                // this step's λ (the ramp's and the target's, both applied)
                let beta = state.scheduled(t);
                let noise = beta * cfg.sigma * (p_dim as f64).sqrt();
                let h = state.lambda_at(t) * beta * beta * cfg.sigma * cfg.sigma / 2.0;
                if noise > 0.0 {
                    ratio_sum += h * gn * scale / noise;
                    ratio_n += 1;
                }
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
        information_ratio: if ratio_n == 0 {
            f64::NAN
        } else {
            ratio_sum / ratio_n as f64
        },
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

fn run_variant(cfg: &Config, variant: Variant, setup: &AnchorSetup) -> VariantReport {
    let (data_a, data_b) = (&setup.data_a, &setup.data_b);
    let start = std::time::Instant::now();
    let shapes = &cfg.shapes;
    let p_dim = shapes.flat_size();
    let name = match variant {
        Variant::Cosine => "cosine/nGPT-lite",
        Variant::Standard => "standard softmax",
    };

    // ---- Phase 3 pre-train on Task A -------------------------------
    // Default: the plan's heavy-ball SGD.  `pretrain_adam` swaps in the
    // tuned optimizer so that a variant whose SGD pre-train diverges can
    // still be compared on the *sampler* rather than on the pre-train.
    let (theta_a, trace_a) = setup.pretrain(cfg, variant);
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
    // chains — the only difference is the full-batch likelihood filter
    // (pCNL: exact only in the drift-free limit, §6.5.6(c)),
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
                crate::task::mh_pcnl_chain(
                    cfg,
                    variant,
                    data_b,
                    init,
                    c,
                    center.as_ref(),
                    crate::task::MhOptions {
                        potential_scale: cfg.potential_scale,
                        ..Default::default()
                    },
                )
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
    let mut ratio_sum = 0.0f64;
    let mut ratio_n = 0usize;
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
        if oc.information_ratio.is_finite() {
            ratio_sum += oc.information_ratio;
            ratio_n += 1;
        }
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
    let window = Window::of(cfg, &scheduler);

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
        information_ratio: if ratio_n == 0 {
            f64::NAN
        } else {
            ratio_sum / ratio_n as f64
        },
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
        window,
        anchor_hash: anchor_hash(&theta_a),
        data_hash: data_b.content_hash(),
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

    /// The anchor's identity is `(D_A, D_B, θ₀, pre-train)`; a second caller
    /// that re-derives any of them gets a different `θ_A` and a table that
    /// cannot be compared with the ambient one (which is what happened before
    /// this struct existed).  The seed offsets are asserted, not assumed.
    #[test]
    fn anchor_setup_uses_the_ambient_seeds() {
        let cfg = Config::quick();
        let s = AnchorSetup::new(&cfg);
        let a = Dataset::new(Task::A, cfg.k, cfg.n_train, cfg.n_test, cfg.seed);
        let b = Dataset::new(
            cfg.task_b,
            cfg.k,
            cfg.n_train,
            cfg.n_test,
            cfg.seed.wrapping_add(1),
        );
        assert_eq!(s.data_a.content_hash(), a.content_hash());
        assert_eq!(s.data_b.content_hash(), b.content_hash());
        assert_ne!(
            s.data_a.content_hash(),
            s.data_b.content_hash(),
            "Task A and Task B must not share a draw"
        );
        let (t1, trace1) = s.pretrain(&cfg, Variant::Cosine);
        let (t2, trace2) = s.pretrain(&cfg, Variant::Cosine);
        assert_eq!(anchor_hash(&t1), anchor_hash(&t2));
        assert_ne!(anchor_hash(&t1), 0);
        assert_eq!(
            trace1.last().unwrap().to_bits(),
            trace2.last().unwrap().to_bits()
        );
    }

    /// The generic pre-train *is* the ambient one, not a lookalike: same
    /// samples, same clip, same momentum, and — because the caller states it —
    /// the same derivative convention.  The check is bitwise on `θ` and on the
    /// loss trace, so `PotentialAnchor` can be said to reproduce the anchor
    /// rather than to approximate it.
    #[test]
    fn potential_anchor_reproduces_momentum_sgd_bitwise() {
        let cfg = Config::quick();
        let setup = AnchorSetup::new(&cfg);
        let generic =
            PotentialAnchor::classifier(AnchorSetup::new(&cfg), cfg.shapes, Variant::Cosine);
        let (steps, lr, momentum) = (3, cfg.pretrain_lr, 0.9);
        let (sgd, trace_sgd) = momentum_sgd(
            &setup.data_a,
            &cfg.shapes,
            Variant::Cosine,
            setup.theta0.clone(),
            steps,
            lr,
            momentum,
        );
        let (potential, trace_pot) = generic.pretrain(steps, lr, momentum, NormGrad::Stopped);
        assert_eq!(
            anchor_hash(&sgd),
            anchor_hash(&potential),
            "the generic pre-train moved the anchor"
        );
        assert_eq!(trace_sgd.len(), trace_pot.len());
        for (a, b) in trace_sgd.iter().zip(trace_pot.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "loss traces differ: {a} vs {b}");
        }
        // The published hash helper is the same function, not a second one.
        assert_eq!(
            generic.pretrained_hash(steps, lr, momentum, NormGrad::Stopped),
            anchor_hash(&sgd)
        );
    }

    /// And the convention matters: `Exact` is a different anchor, which is why
    /// it is a parameter and not a default baked into the trait.
    #[test]
    fn potential_anchor_convention_changes_the_anchor() {
        let cfg = Config::quick();
        let generic =
            PotentialAnchor::classifier(AnchorSetup::new(&cfg), cfg.shapes, Variant::Cosine);
        let opts = (2, cfg.pretrain_lr, 0.9);
        let exact = generic.pretrain(opts.0, opts.1, opts.2, NormGrad::Exact);
        let stopped = generic.pretrain(opts.0, opts.1, opts.2, NormGrad::Stopped);
        assert_ne!(
            anchor_hash(&exact.0),
            anchor_hash(&stopped.0),
            "Exact and Stopped produced the same anchor"
        );
    }

    /// The Brier column is the strictly proper score of the averaged predictive
    /// distribution, and the per-draw series are that draw's own score — with
    /// identical draws the two coincide, which pins both definitions.
    #[test]
    fn bma_detail_scores_match_on_identical_draws() {
        // `vocab = k + 1`: the BOS token is 0 and the labels are `1..=k`.
        let shapes = Shapes {
            vocab: 6,
            d: 4,
            ff: 8,
            layers: 1,
        };
        let data = Dataset::new(Task::A, 5, 8, 6, 3);
        let theta = init_flat(&shapes, 0.1, 4);
        let d = bma_detail(
            &[vec![theta.clone()], vec![theta]],
            &shapes,
            Variant::Cosine,
            &data,
        );
        assert_eq!(d.models, 2);
        assert_eq!(d.acc_chain.len(), 2);
        assert_eq!(d.acc_chain[0].len(), 1);
        for d2 in [&d.acc_chain[1], &d.brier_chain[1]] {
            assert_eq!(d2.len(), 1);
        }
        assert!((d.brier - d.brier_chain[0][0]).abs() < 1e-12, "{}", d.brier);
        assert!((d.acc - d.acc_chain[0][0]).abs() < 1e-9);
        assert!((0.0..=2.0).contains(&d.brier), "Brier out of range");
        assert!(d.nll.is_finite() && d.ece.is_finite());
    }

    #[test]
    fn quick_experiment_reports_sane_numbers() {
        let mut cfg = Config::quick();
        // exercise the Week-2 safetensors path end to end
        let dir = std::env::temp_dir().join(format!("pcnl_quick_{}", std::process::id()));
        cfg.save_dir = Some(dir.clone());
        let report = run_experiment(&cfg);
        let text = report.render();
        assert_eq!(report.variants.len(), 2);
        // Every arm the report publishes adapted from the *shared* anchor, so a
        // second caller (`examples/subspace_adaptation.rs`) that builds its own
        // through `AnchorSetup` gets the same identity and a comparable table.
        let shared = AnchorSetup::new(&cfg).pretrain(&cfg, Variant::Cosine).0;
        assert_eq!(report.variants[0].anchor_hash, anchor_hash(&shared));
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

    /// R0.5: the schedule decides whether a stationary diagnostic is allowed.
    /// The default `κ = 0.5` schedule decays over any reachable kept window, so
    /// the honest default answer is "this is an annealing trace"; only a fixed
    /// step size with a finished ramp licenses the stationary reading.
    #[test]
    fn window_separates_annealing_from_stationary_sampling() {
        let cfg = Config::default();
        let sched = SgPcnlScheduler {
            beta_0: cfg.beta_0,
            gamma: cfg.gamma,
            kappa: cfg.kappa,
        };
        let w = Window::of(&cfg, &sched);
        assert_eq!(w.start, cfg.burn);
        assert_eq!(w.end, cfg.steps);
        assert!(!w.stationary, "decaying beta must not claim stationarity");
        assert!(w.beta_end < w.beta_start);
        assert_eq!(w.lambda, 1.0, "default target is the likelihood one");
        let line = w.render(cfg.thin, cfg.chains);
        assert!(line.contains("NONSTATIONARY"), "{line}");
        assert!(
            line.contains(&format!("kept t={}..{}", cfg.burn, cfg.steps)),
            "{line}"
        );

        // Fixed step size, finished ramp: the one stationary case.
        let fixed = Config {
            gamma: 0.0,
            ..Config::default()
        };
        let sched = SgPcnlScheduler {
            beta_0: fixed.beta_0,
            gamma: 0.0,
            kappa: fixed.kappa,
        };
        let w = Window::of(&fixed, &sched);
        assert!(w.stationary, "{w:?}");
        assert!(
            w.render(fixed.thin, fixed.chains)
                .contains("stationary (constant target"),
            "{w:?}"
        );
        let p = phases(&fixed, &sched);
        let names: Vec<&str> = p.iter().map(|ph| ph.name.as_str()).collect();
        assert_eq!(names, ["warmup", "kept"]);

        // A ramp that runs into the kept window keeps it nonstationary even at
        // a fixed step size, and the window records the tempered lambda.
        let late = Config {
            gamma: 0.0,
            anneal_steps: fixed.burn + 100,
            ..Config::default()
        };
        let w = Window::of(&late, &sched);
        assert!(!w.stationary);
        assert!(w.lambda < 1.0, "lambda at kept start = {}", w.lambda);
        let p = phases(&late, &sched);
        let names: Vec<&str> = p.iter().map(|ph| ph.name.as_str()).collect();
        assert_eq!(names, ["warmup", "likelihood-ramp", "kept"]);
    }

    /// R0.6: the report carries a manifest that identifies the run — and the
    /// rendered text carries R0.5's window line, not just the numbers.
    #[test]
    fn manifest_identifies_the_run() {
        let cfg = Config::quick();
        let report = run_experiment(&cfg);
        let m = &report.manifest;
        assert_ne!(m.code.revision, "unknown");
        assert_eq!(m.tool, "pcnl");
        assert_eq!(m.sampler.steps, cfg.steps);
        assert_eq!(m.data.seed, cfg.seed);
        assert_eq!(m.data.anchor_hash, Some(report.variants[0].anchor_hash));
        assert_ne!(m.data.hash, 0);
        assert_eq!(report.variants[0].data_hash, m.data.hash);
        assert_eq!(m.metric.kind, "isotropic");
        assert!(
            m.phases.iter().any(|p| p.name == "kept" && !p.stationary),
            "quick config keeps a decaying-step window"
        );
        assert_eq!(m.work.forward_evals, m.work.backward_evals);
        assert!(m.work.forward_evals > 0);
        let json = m.to_json();
        assert!(json.contains(&format!("\"revision\": \"{}\"", m.code.revision)));
        assert!(json.contains("\"phases\": [{\"name\": \"warmup\""));
        let text = report.render();
        assert!(
            text.contains(&format!("window: kept t={}..{}", cfg.burn, cfg.steps)),
            "missing window line:\n{text}"
        );
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
            "mobility",
            "window: kept t=",
            "NONSTATIONARY",
        ] {
            assert!(text.contains(needle), "missing {needle}:\n{text}");
        }
    }
}
