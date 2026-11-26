# pcnl development plan — adapted pCNL transformer configuration ensembles (R14)

Current work order: **2026-11-25**. Read the active specification **R14** in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) and mathematical/sampler packets **Q0–Q7** in
[timepiece/pcnl.md](../../timepiece/pcnl.md). Recorded measurements live in
[PCNL_STATUS.md](../docs/PCNL_STATUS.md). This is a plan; no R14 code, tests,
experiments, or Lean compilation are claimed as executed by creating it. All P0–P7
artifacts below are proposed until their implementation and checks are recorded.

## Present Project State & Track Disposition (2026-11-25)

| Track / Modules | Status (2026-11-25) | Disposition in active plan |
|---|---|---|
| **R14 (`P0–P7`)** | **Active (proposed)** | New work order: modified pCNL for diverse transformer configuration ensembles over large good-enough training-loss regions (supervised and non-RLCD RL). |
| **R13** (`src/vector_model.rs`, `examples/vector_{regression,basins,stream}.rs`, `logs/pcnl_r13_gates/`, `scripts/check-pcnl-gates`) | Executed diagnostics & baseline | Readout-conditional 34-dim exact gates pass (`R̂ < 1.01`, `ESS >= 400`); full 57,378-dim sharp Gibbs hits geometry/multimodality wall (`ESS ~ 4.5–5.3`). Retained as baseline infrastructure and regression gates. |
| **R8 (`D0–D6`)** (`certificate.rs`, `dim_benchmark.rs`, `approx_pricing.rs`, `linear_gaussian.rs`, `operator_cert.rs`, `evidence_enclosure.rs`, `tests/{r8_certificates,d_track}.rs`) | Executed mechanics | Retained as reusable verification primitives (claim records, D1/D2 reference checks, D3 bias pricing, D5 trajectory-clustered uncertainty). |
| **R9 / Edge0-RLCD (`E0–E5`)** (`src/edge0_*.rs`, `tests/edge0_rlcd.rs`, `EDGE0_RLCD_PLAN.md`) | **Retired (2026-11-25)** | Executed 2026-11-15..16 (RLCD trailed CE+Brier comparator). Frozen historical record; no further RLCD development. |
| **R0–R7, R10–R12** (`model.rs`, `sampler.rs`, `task.rs`, `mdp.rs`, `policy_model.rs`, `policy_readout.rs`, `dt_subspace.rs`, `continual.rs`, `rotation_head.rs`) | Historical record + reusable core | `model.rs`, `sampler.rs`, `task.rs`, `mdp.rs`, `policy_model.rs`, `policy_readout.rs`, `dt_subspace.rs`, and `continual.rs` carry forward directly into R14. |

## P0 — Satisficing potential and ensemble manifest contract

Reuse `src/target.rs::Potential`, `src/vector_model.rs::EnergyTarget`, `src/manifest.rs`,
and `src/certificate.rs`. Implement a smooth one-sided satisficing potential wrapper
$\Phi_{\tau,\kappa,\alpha}(\theta) = \kappa\,\psi_\delta(L(\theta)-\tau) + \alpha\,L(\theta)$
around any differentiable supervised or RL training loss $L(\theta)$:
- Support smooth hinge options $\psi_\delta$ (softplus $\delta\log(1+e^{u/\delta})$ and
  $C^1$ Huberized $\frac{1}{2}\max(0,u)^2$), declared loss ceiling $\tau > 0$,
  confinement weight $\kappa > 0$, and gentle optimization tilt $\alpha \ge 0$ ($\alpha \ll \kappa$).
- Expose `EnergyTarget` and `Potential` implementations so both `src/vector_model.rs`
  and `src/target.rs` / `src/sampler.rs` consume the exact same objective and gradient.
- Require $\Phi_{\tau,\kappa,\alpha} \ge 0$ whenever $L \ge 0$ and $\alpha \ge 0$ so
  D0/D1 proper-target invariants ($0 < Z < \infty$) hold under the proper Gaussian prior $P_0$.
- Extend manifest and checkpoint serialization (`src/manifest.rs`, `src/persist.rs`) to
  record $\tau, \kappa, \alpha, \delta$, hinge kind, loss definition/reduction, $P_0$
  parameters, initialization policy, and member provenance.

**Acceptance:** Finite-difference gradient checks match analytic/autodiff gradients for
both $\psi_\delta$ variants inside ($L < \tau$), near ($L \approx \tau$), and outside
($L > \tau$) the plateau; two matched configs produce identical target identities while
changing $\tau, \kappa, \alpha, \delta$ or $P_0$ changes the target identity; nonfinite
potentials/gradients fail explicitly.

## P1 — Known-target plateau-vs-spike verification

Add a low-dimensional synthetic target with an exact 1-D/2-D quadrature reference
(extending `src/target.rs` / `examples/known_target.rs`) comparing:
1. A broad, high-prior-volume "good-enough" plateau ($L \le \tau$ over a wide region), and
2. A narrow, deep spike ($L \approx 0$ over tiny prior volume).

Run exact MH-pCNL (with optional prior-refresh mixture $r > 0$) on $\Phi_{\tau,\kappa,\alpha}$,
alongside two controls: sharp Gibbs ($\tau = 0$ / high concentration) and radical
full-convergence Adam.

**Acceptance:** Four dispersed chains pass `R̂ < 1.01` and `bulk ESS >= 400`; estimated
plateau mass, spike mass, and coordinate moments match quadrature within MCSE;
$\Phi_{\tau,\kappa,\alpha}$ places dominant mass on the broad good-enough plateau where
radical Adam collapses to the narrow spike.

## P2 — Adapted pCNL ensemble runner on transformer architectures

Build a unified ensemble runner supporting homogeneous weight-configuration ensembles of:
- `src/vector_model.rs` (`VectorShapes` 4-block transformer with `init_flat_prior` so
  attention $W_q, W_k$ stay active), and
- `src/model.rs` (nGPT-lite cosine vs standard sequence transformer).

Protocol:
- Initialize $M$ independent prior-scale members (`init_flat_prior`, distinct RNG streams).
- Support **moderate warm entry**: optional Adam/SGD phase that halts at the first entry
  into $L(\theta) \le \tau$ (or a predeclared development slowdown knee), never running to
  zero-loss memorization.
- Sample/diffuse each member inside $\Phi_{\tau,\kappa,\alpha}$ under two explicit tiers:
  1. *Reference tier:* full-data MH-adjusted pCNL (+ optional prior-refresh mixture),
  2. *Scalable ensemble tier:* unadjusted / minibatch SG-pCNL (`src/sampler.rs`), priced
     against the reference tier using D3 (`src/approx_pricing.rs`) on matched small targets.
- Serialize/reload full $M$-member configuration ensembles deterministically.

**Acceptance:** Checkpointed ensemble round-trips bitwise identically; within-region
acceptance and parameter movement strictly exceed R13's sharp-Gibbs full-parameter
ceiling (or any remaining curved-boundary stall is measured and reported explicitly);
D3 comparison reports the exact bias/discretization budget between reference MH-pCNL
and scalable SG-pCNL.

## P3 — Variability, fit, and flatness diagnostics

Implement a shared diagnostics module evaluating every ensemble $\{\theta_1, \dots, \theta_M\}$
on four axes:
1. **In-region training fit:** fraction of members with $L_{\text{train}}(\theta_j) \le \tau$,
   mean/quantile training loss (verifying neither collapse to $0$ nor escape above $\tau$).
2. **Parameter & functional variability:** pairwise Euclidean/cosine distances in parameter
   space, **per-input prediction variance and disagreement** across train/ID/OOD splits, and
   effective rank (participation ratio) of the centered $M \times (n_{\text{eval}} q)$
   ensemble prediction matrix.
3. **Flatness & PAC-Bayes mixture bound:** finite-radius isotropic/prior-scaled perturbation
   risk around retained configurations and the R13 §6 equal-weight Gaussian-mixture
   PAC-Bayes upper risk bound (splitting confidence across data and MC draws).
4. **Generalization & calibration:** ensemble-mean RMSE/NLL/Brier, 50/80/90/95% predictive
   coverage, and normalized residual $z^2$.

**Acceptance:** Synthetic permutation-copy ensembles (identical functions, rotated/permuted
weights) are caught by functional diversity showing zero functional variance/rank 1 despite
nonzero weight distance; finite-sample PAC-Bayes bounds and MC allowances pass unit tests.

## P4 — Supervised transformer ensemble campaign & matched-compute comparators

Execute predeclared, bounded comparisons with equal total gradient/evaluation budgets on:
- **(a) Vector regression & continual stream:** 4-block English-embedding transformer
  (`src/vector_model.rs`, `examples/vector_stream.rs`, 20 Newsgroups / AG News fixtures),
  measuring OOD/shift robustness and continual prefix retention.
- **(b) Sequence task:** nGPT-lite cosine transformer (`src/model.rs`, `src/task.rs`).

Score all six comparators (thresholds $\tau, \alpha, \kappa$ frozen on development data
only before touching test):
1. Radical single-point optimizer (full-convergence Adam/SGD),
2. Radical deep ensemble ($M$ full-convergence Adam runs),
3. Early-stopped / knee-stopped deep ensemble (stopped at $\tau$, no pCNL diffusion),
4. Sharp Gibbs pCNL ($\tau = 0$),
5. Adapted satisficing pCNL ensemble ($\Phi_{\tau,\kappa,\alpha}$),
6. Prior-only / vacuous-$\tau$ control.

**Acceptance:** Paired uncertainty intervals across seeds and splits for all primary
endpoints (in-region satisfaction, functional diversity/effective rank, PAC-Bayes flatness,
held-out ID/OOD error, calibration $z^2$, and stream retention). Negative or inconclusive
contrasts against the early-stopped deep ensemble are preserved and reported verbatim.

## P5 — Reinforcement learning training-loss ensemble track (non-RLCD)

Reuse `src/mdp.rs`, `src/policy_model.rs`, `src/policy_readout.rs`, and `src/dt_subspace.rs`
for a **non-RLCD** RL training-loss track (no Edge0 latent-MoE, no four-path REINFORCE
Brier U-statistic). Define a differentiable RL training loss $L_{\text{RL}}(\theta)$ on
offline/behavior-regularized trajectories or a policy-gradient surrogate, and apply the
same $\Phi_{\tau,\kappa,\alpha}$ adapted pCNL ensemble protocol.

Evaluate and report separately:
1. RL training-loss satisfaction ($L_{\text{RL}}(\theta_j) \le \tau$),
2. Action-distribution / logit functional diversity across states,
3. Trajectory-clustered online rollout return (`policy_readout.rs` D5 estimator) on both
   nominal MDP dynamics and perturbed/shifted evaluation MDPs, auditing per-step BMA vs
   once-per-episode configuration draws (`dt_subspace.rs`).

**Acceptance:** Trajectory-clustered SEs pass synthetic correlation checks; both nominal
and shifted-environment rollout returns are reported with paired intervals against
matched-cost radical and early-stopped policy ensembles.

## P6 — Gated extensions (heterogeneous configurations, repulsion, serving cost)

Strictly gated behind P1–P4 completion and written review:
- **Heterogeneous transformer configurations:** combine distinct widths, depths, head
  counts, or normalization choices inside one ensemble while matching total compute.
- **Interacting functional-space repulsion (optional):** only if independent P4 chains
  collapse functionally; must carry an explicit label noting that repulsion alters the
  target from $\pi_{\tau,\kappa,\alpha}^{\otimes M}$ and compare against uncoupled chains.
- **Serving cost & distillation audit:** compare full $M$-member ensemble inference
  against single-pass or compact $K$-member subsets / distilled students (following R13 §7)
  on latency, memory, and predictive/diversity loss.

**Acceptance:** Each extension declares its hypothesis, changed objective (if any), and
compute cap before execution.

## P7 — Verification, CI gates, status archival, and optional RLCD hygiene

- Run targeted Rust unit/integration tests for every new/modified module (`cargo test --release -p pcnl --locked`),
  `./scripts/check-pcnl-gates --check-manifest`, and `./scripts/check-attribution --repo-only`.
- Record executed P0–P6 results (positive, negative, and inconclusive) in
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
