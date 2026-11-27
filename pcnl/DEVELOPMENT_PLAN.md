# pcnl development plan — satisficing pCNL, SWA, and posterior distillation (R14)

Current work order: **2026-11-25 (revised 2026-11-27)**. Read the active specification
**R14** in [RESEARCH_PLAN.md](RESEARCH_PLAN.md), the active Lean 4 packets **S0–S6** (reusing
Q0–Q6 and T0–T5) in [timepiece/pcnl.md](../../timepiece/pcnl.md), and the theoretical
discussions in [ngpt.md](../../test/unfer/ngpt.md). Recorded measurements live in
[PCNL_STATUS.md](../docs/PCNL_STATUS.md). This is a plan; no R14 code, tests, experiments,
or Lean compilation are claimed as executed by creating or revising it. All P0–P7 artifacts
below are proposed until their implementation and checks are recorded.

## Present Project State & Track Disposition (2026-11-27)

| Track / Modules | Status (2026-11-27) | Disposition in active plan |
|---|---|---|
| **R14 (`P0–P7`)** | **Active (proposed)** | Satisficing pCNL over good-enough regions ($\Phi_{\tau,\kappa,\alpha}$), intra-basin SWA, and inter-basin posterior distillation (supervised data loss and non-RLCD RL penalties), aligned with Lean **S0–S6**. |
| **R13** (`src/vector_model.rs`, `examples/vector_{regression,basins,stream}.rs`, `logs/pcnl_r13_gates/`, `scripts/check-pcnl-gates`) | Executed diagnostics & baseline | Readout-conditional 34-dim exact gates pass (`R̂ < 1.01`, `ESS >= 400`, `480/480` train coverage, exposing Adam memorization as the cause of `0.158` test coverage; full-parameter `pilot-calibrated` covers `478/480` train vs `170/240` test at `df = 480 − 57,378 < 0`); full 57,378-dim sharp Gibbs hits geometry/multimodality wall (`ESS ~ 4.5–5.3`). CI-gated via `--quick --structure` canaries (~94 s), scheduled `--all --band 0.2`, and host-local byte-exact default. Retained as baseline infrastructure and regression gates. |
| **R8 (`D0–D6`)** (`certificate.rs`, `dim_benchmark.rs`, `approx_pricing.rs`, `linear_gaussian.rs`, `operator_cert.rs`, `evidence_enclosure.rs`, `tests/{r8_certificates,d_track}.rs`) | Executed mechanics | Retained as reusable verification primitives (claim records, D1/D2 reference checks, D3 SG bias pricing, D5 trajectory-clustered uncertainty). |
| **R9 / Edge0-RLCD (`E0–E5`)** (`src/edge0_*.rs`, `tests/edge0_rlcd.rs`, `EDGE0_RLCD_PLAN.md`) | **Retired (2026-11-25)** | Executed 2026-11-15..16 (RLCD trailed CE+Brier comparator). Frozen historical record; no further RLCD development. |
| **R0–R7, R10–R12** (`model.rs`, `sampler.rs`, `task.rs`, `mdp.rs`, `policy_model.rs`, `policy_readout.rs`, `dt_subspace.rs`, `continual.rs`, `rotation_head.rs`) | Historical record + reusable core | `model.rs`, `sampler.rs`, `task.rs`, `mdp.rs`, `policy_model.rs`, `policy_readout.rs`, `dt_subspace.rs`, and `continual.rs` carry forward directly into R14 (`rotation_head.rs` kept as historical record). |

## P0 — Satisficing potential, SWA/distillation manifest, and Lean S0 parity

Reuse `src/target.rs::Potential`, `src/vector_model.rs::EnergyTarget`, `src/manifest.rs`,
and `src/certificate.rs`. Implement a smooth one-sided satisficing potential wrapper
(proposed `src/satisficing.rs`)
$\Phi_{\tau,\kappa,\alpha}(\theta) = \kappa\,\psi_\delta(L(\theta)-\tau) + \alpha\,L(\theta)$
around any differentiable supervised data loss or non-RLCD RL training penalty $L(\theta)$:
- Support smooth hinge options $\psi_\delta$ (softplus $\delta\log(1+e^{u/\delta})$ and
  $C^1$ Huberized $\frac{1}{2}\max(0,u)^2$), declared loss ceiling $\tau > 0$,
  confinement weight $\kappa > 0$, and gentle optimization tilt $\alpha \ge 0$ ($\alpha \ll \kappa$).
- Expose `EnergyTarget` and `Potential` implementations so both `src/vector_model.rs`
  and `src/target.rs` / `src/sampler.rs` consume the exact same objective and gradient.
- Require $\Phi_{\tau,\kappa,\alpha} \ge 0$ whenever $L \ge 0$ and $\alpha \ge 0$ so
  D0/D1 and Lean **S0** (`SatisficingTarget.lean`) proper-target invariants ($0 < Z \le 1$)
  hold under the proper Gaussian prior $P_0$.
- Extend manifest and checkpoint serialization (`src/manifest.rs`, `src/persist.rs`) to
  record $\tau, \kappa, \alpha, \delta$, hinge kind, loss definition/reduction, $P_0$
  parameters, initialization policy, member provenance, **SWA window/spacing**, and
  **distillation transfer-split/teacher provenance** (keeping synthetic teacher targets out
  of the committed evidence prefix).

**Acceptance:** Finite-difference gradient checks match analytic/autodiff gradients for
both $\psi_\delta$ variants inside ($L < \tau$), near ($L \approx \tau$), and outside
($L > \tau$) the plateau (including zero hinge gradient inside $L < \tau$ for Huberized
$\alpha=0$); two matched configs produce identical target identities while changing
$\tau, \kappa, \alpha, \delta$ or $P_0$ changes the target identity; nonfinite
potentials/gradients fail explicitly.

## P1 — Known-target plateau-vs-spike verification (Lean S0/S1/S2 check)

Add a low-dimensional synthetic target with an exact 1-D/2-D quadrature reference
(proposed `examples/r14_plateau_quadrature.rs`, extending `src/target.rs` /
`examples/known_target.rs`) comparing:
1. A broad, high-prior-volume "good-enough" plateau ($L \le \tau$ over a wide region), and
2. A narrow, deep spike ($L \approx 0$ over tiny prior volume).

Run exact MH-pCNL (with optional prior-refresh mixture $r > 0$) on $\Phi_{\tau,\kappa,\alpha}$
across multiple temperatures $T$, alongside two controls: sharp Gibbs ($\tau = 0$ / high
concentration) and radical full-convergence Adam. Also check intra-basin weight averaging on
a convex/quadratic plateau against an unaligned two-basin midpoint barrier (matching Lean
**S2**).

**Acceptance:** Four dispersed chains pass `R̂ < 1.01` and `bulk ESS >= 400`; estimated
plateau mass, spike mass, and coordinate moments match quadrature within MCSE across
temperatures; $\Phi_{\tau,\kappa,\alpha}$ places dominant mass on the broad good-enough
plateau where radical Adam collapses to the narrow spike.

## P2 — Adapted pCNL ensemble runner and intra-basin SWA

Build a unified ensemble runner (with proposed `src/swa.rs` for online averaging and
straight-line barrier probes) supporting homogeneous weight-configuration ensembles of:
- `src/vector_model.rs` (`VectorShapes` 4-block transformer with `init_flat_prior` so
  attention $W_q, W_k$ stay active), and
- `src/model.rs` (nGPT-lite cosine vs standard sequence transformer).

Protocol:
- Initialize $M$ independent prior-scale members (`init_flat_prior`, distinct RNG streams).
- Support **moderate warm entry**: optional Adam/SGD phase that halts either at first entry
  into $L(\theta) \le \tau$ or by R13 §5's predeclared slowdown knee rule (three consecutive
  checks with fractional drop $< 1\%$ and decreasing trend, logging capped runs verbatim),
  never running to zero-loss memorization.
- Sample/diffuse each member inside $\Phi_{\tau,\kappa,\alpha}$ under two explicit tiers:
  1. *Reference tier:* full-data MH-adjusted pCNL (+ optional prior-refresh mixture),
  2. *Scalable ensemble tier:* unadjusted / minibatch SG-pCNL (`src/sampler.rs`), priced
     against the reference tier using D3 (`src/approx_pricing.rs`) on matched small targets.
- Add **online intra-basin SWA** ($\theta_{\text{SWA}} = K^{-1}\sum_{k=1}^K \theta_k$) over
  predeclared post-knee / within-plateau snapshots (supporting both constant-LR SGD-SWA and
  within-basin pCNL-SWA), plus straight-line interpolation loss evaluation and output-mean
  comparison on the same snapshots.
- Serialize/reload full $M$-member configuration ensembles and SWA centers deterministically.

**Acceptance:** Checkpointed ensemble and SWA weights round-trip bitwise identically; online
SWA matches batch snapshot averaging within floating-point tolerance; SWA vs last-iterate vs
snapshot output-mean and midpoint interpolation barrier are logged on every run;
within-region acceptance and parameter movement strictly exceed R13's sharp-Gibbs
full-parameter ceiling (or any remaining curved-boundary stall is measured and reported
explicitly); D3 comparison reports the exact bias/discretization budget between reference
MH-pCNL and scalable SG-pCNL.

## P3 — Variability, fit, flatness, SWA, and distillation diagnostics

Implement a shared diagnostics module (proposed `src/ensemble_diag.rs`) evaluating every
ensemble $\{\theta_1, \dots, \theta_M\}$
and compressed model ($\theta_{\text{SWA}}$, $\theta_{\text{student}}$) on five axes:
1. **In-region training fit:** fraction of members with $L_{\text{train}}(\theta_j) \le \tau$,
   mean/quantile training loss (verifying neither collapse to $0$ nor escape above $\tau$).
2. **Parameter & functional variability:** pairwise Euclidean/cosine distances in parameter
   space, **per-input prediction variance and disagreement** across train/ID/OOD splits, and
   effective rank (participation ratio) of the centered $M \times (n_{\text{eval}} q)$
   ensemble prediction matrix.
3. **Flatness & PAC-Bayes mixture bound:** finite-radius isotropic/prior-scaled perturbation
   risk around retained configurations and the R13 §6 / Lean **S3** equal-weight
   Gaussian-mixture PAC-Bayes upper risk bound (splitting confidence across data and MC draws).
4. **Generalization & calibration:** ensemble-mean RMSE/NLL/Brier, 50/80/90/95% predictive
   coverage, and normalized residual $z^2$.
5. **Compression diagnostics (Lean S2/S4):** SWA weight-mean vs output-mean discrepancy and
   straight-line loss barrier; teacher–student KL, TV, and dark-knowledge Jensen–Shannon gap
   (classification/RL), plus vector mean+spread fidelity and predictive interval coverage.

**Acceptance:** Synthetic permutation-copy ensembles (identical functions, rotated/permuted
weights) are caught by functional diversity showing zero functional variance/rank 1 despite
nonzero weight distance; finite-sample PAC-Bayes bounds, MC allowances, squared-error
ambiguity identity, Jensen–Shannon nonnegativity, and ReLU-style unaligned midpoint barrier
checks pass unit tests.

## P4 — Supervised campaign: SWA, $M$-basin resampling, and posterior distillation

Execute predeclared, bounded comparisons (proposed `src/distill.rs` and
`examples/r14_supervised_campaign.rs`) with equal total gradient/evaluation budgets (charging
full-loss knee checks, SWA snapshots, teacher forwards, and student training) on:
- **(a) Vector regression & continual stream:** 4-block English-embedding transformer
  (`src/vector_model.rs`, `examples/vector_stream.rs`, 20 Newsgroups / AG News fixtures),
  measuring OOD/shift robustness and continual prefix retention. Distill both the ensemble
  mean $\bar f(x)$ and a per-coordinate epistemic spread head $\hat s^2(x)$ (or Gaussian
  NLL head) on training/unlabeled transfer texts, retaining the committed evidence prefix and
  $M$-member ensemble state for subsequent Bayesian updates (teacher outputs excluded from
  evidence logs).
- **(b) Sequence task:** nGPT-lite cosine transformer (`src/model.rs`, `src/task.rs`),
  distilling $p_{\text{ens}}(y\mid x) = M^{-1}\sum_{m=1}^M p(y\mid x,\theta_m)$ via KL
  minimization into a single-pass student.

Score all **eight matched-compute comparators** (freezing $\tau, \alpha, \kappa$, SWA window,
and distillation tolerance on development data only before touching test):
1. Radical single-point optimizer (full-convergence Adam/SGD, $1\times$),
2. Radical deep ensemble ($M$ full-convergence Adam runs, $M\times$),
3. Early-stopped / knee-stopped deep ensemble (stopped at $\tau$, no pCNL diffusion, $M\times$),
4. Intra-basin SWA single model (post-knee constant-LR SGD-SWA and/or within-basin pCNL-SWA,
   $1\times$),
5. Sharp Gibbs pCNL ($\tau = 0$, $M\times$),
6. Adapted satisficing pCNL ensemble ($\Phi_{\tau,\kappa,\alpha}$, optionally SWA-centered per
   basin, $M\times$),
7. Posterior-distilled single student (trained on #6's ensemble predictive, paired with an
   equal-budget label-trained student control of identical architecture, $1\times$), and
8. Prior-only / vacuous-$\tau$ control.

**Acceptance:** Paired uncertainty intervals across seeds and splits for all primary
endpoints (in-region satisfaction, functional diversity/effective rank, PAC-Bayes flatness,
SWA weight-mean vs output-mean barrier, teacher–student ID/OOD fidelity, held-out ID/OOD
error, calibration $z^2$, stream retention, and $1\times$ vs $M\times$ latency/memory).
Negative or inconclusive contrasts (against the early-stopped deep ensemble or label-trained
student control) are preserved and reported verbatim.

## P5 — Reinforcement learning training-loss ensemble & policy distillation (non-RLCD)

Reuse `src/mdp.rs`, `src/policy_model.rs`, `src/policy_readout.rs`, and `src/dt_subspace.rs`
(proposed `examples/r14_rl_campaign.rs`)
for a **non-RLCD** RL training-loss track (no Edge0 latent-MoE, no four-path REINFORCE
Brier U-statistic). Define a differentiable RL training loss $L_{\text{RL}}(\theta) \ge 0$
on offline/behavior-regularized trajectories or a policy-gradient surrogate, apply the
$\Phi_{\tau,\kappa,\alpha}$ adapted pCNL ensemble protocol, and train a single **KL-distilled
student policy** $\pi_{\text{stu}}(a\mid s)$ to match the per-step ensemble policy
$\pi_{\text{ens}}(a\mid s)$ on training/rollout states.

Evaluate and report separately:
1. RL training-loss satisfaction ($L_{\text{RL}}(\theta_j) \le \tau$),
2. Action-distribution / logit functional diversity across states,
3. Empirical teacher–student KL/TV across nominal and shifted rollout states alongside the
   Pinsker / finite-horizon coupling bound (Lean **S5**), and
4. Trajectory-clustered online rollout return (`policy_readout.rs` D5 estimator) on both
   nominal MDP dynamics and perturbed/shifted evaluation MDPs, comparing $1\times$ policies
   (radical, early-stopped/SWA, loss-trained student control, KL-distilled student) against
   $M\times$ policies (per-step BMA vs once-per-episode draw in `dt_subspace.rs`).

**Acceptance:** Trajectory-clustered SEs pass synthetic correlation checks; both nominal
and shifted-environment rollout returns and teacher–student KL/TV are reported with paired
intervals against matched-cost radical, early-stopped, SWA, and loss-trained controls.

## P6 — Gated extensions (heterogeneous configurations, Git Re-Basin, repulsion)

Strictly gated behind P1–P5 completion and written review:
- **Heterogeneous transformer configurations:** combine distinct widths, depths, head
  counts, or normalization choices inside one ensemble while matching total compute (note
  that distillation handles heterogeneous teachers naturally in function space).
- **Git Re-Basin cross-basin weight merging:** if direct merging of independent runs is
  tested, first verify function-preserving hidden-unit permutations, then report aligned vs
  unaligned straight-line interpolation barriers and compare against P4/P5 posterior
  distillation.
- **Interacting functional-space repulsion (optional):** only if independent P4/P5 chains
  collapse functionally; must carry an explicit label noting that repulsion alters the
  target from $\pi_{\tau,\kappa,\alpha}^{\otimes M}$ and compare against uncoupled chains.

**Acceptance:** Each extension declares its hypothesis, changed objective (if any), and
compute cap before execution.

## P7 — Verification, CI gates, Lean parity, status archival, and RLCD hygiene

- Run targeted Rust unit/integration tests for every new/modified module
  (`cargo test --release -p pcnl --locked`, requiring sibling path-deps such as
  `../australVM/safestos/gpu` in the workspace manifest), `./scripts/check-pcnl-gates --check-manifest`,
  `nix-shell -p python3 --run "python3 scripts/check-links pcnl/RESEARCH_PLAN.md pcnl/DEVELOPMENT_PLAN.md"`,
  and `./scripts/check-attribution --repo-only`.
- Mirror R13's CI-gate pattern for executed R14 gates (leaving `logs/pcnl_r13_gates/`
  undisturbed): commit verified R14 logs under a dedicated `logs/pcnl_r14_gates/` directory
  only after runs exist, with portable `--quick --structure` per-push canaries, scheduled
  `--band 0.2` drift nets, and host-local byte-exact defaults.
- Maintain parity with Lean packets **S0–S6** in [timepiece/pcnl.md](../../timepiece/pcnl.md)
  and record executed P0–P6 results (positive, negative, and inconclusive) in
  [PCNL_STATUS.md](../docs/PCNL_STATUS.md) under a new **R14** section only after runs
  exist; never record planned checks as passed.
- **Legacy RLCD code hygiene (deferred, separate approval):** Existing `src/edge0_*.rs`,
  `tests/edge0_rlcd.rs`, and `examples/{edge0_campaign,r9_campaign,r9_secondary}.rs`
  remain frozen for historical reproducibility (`scripts/pcnl-reproduce` still invokes
  `--test edge0_rlcd`). Removing or archiving them out of the crate is a separate,
  explicitly approved cleanup task.

---

## Appendix — Historical Packet Disposition (`D0–D7` & `E0–E7`)

Existing Rust source comments in `pcnl/src/` and `pcnl/tests/` cite `D0–D6` and `E0–E5`
from the 2026-11-15 revision of this file. Their disposition as of 2026-11-25 is:

- **`D0–D6` (R8 sampler/certificate packets):** Implemented and tested in
  `src/{certificate,dim_benchmark,approx_pricing,linear_gaussian,operator_cert,evidence_enclosure,policy_readout,dt_subspace}.rs`
  and `tests/{r8_certificates,d_track}.rs`. Retained as reusable verification machinery
  supporting R14 (`P0–P3`, `P5`).
- **`E0–E5` (R9 Edge0-RLCD packets):** Mechanics implemented in `src/edge0_*.rs` and
  executed on 2026-11-15..16 (see [PCNL_STATUS.md](../docs/PCNL_STATUS.md)). **Retired
  on 2026-11-25** per project direction; `E6–E7` are closed without further work and
  superseded by `P0–P7`.
