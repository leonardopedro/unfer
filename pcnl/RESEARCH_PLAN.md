# pcnl research and development plan

Date: 2026-11-08. Implementation inspected at unfer commit
`5b4b79c77f8c35bb4039ab1c755733621b5cd68f`. Revised 2026-11-12: added R7, the
decision-transformer application track, and its milestones; the R0–R2 record is unchanged.

## Objective and reading order

Develop a reproducible Bayesian-adaptation laboratory that can answer two questions:
does a tractable, explicitly specified posterior improve predictions over a strong point
baseline, and can a sampler approximate that posterior within a measured compute budget?
Treat target quality, sampler accuracy and decision/readout quality as separate experiments.

The first research bet is **adaptation in a likelihood-informed parameter subspace**,
followed by a full-target sampler with different proposals inside and outside that subspace.
Do not start another ambient step-size/temperature ladder as the default next experiment.
The accumulated negative results make target specification and verification the next gates.

The application this plan now works toward is **a small decision transformer (DT) whose output
is an action distribution** (R7): offline trajectories, return-to-go conditioning, and a policy
as the posterior predictive. Only the potential and the readout change; the prior, the affine
subspace chart and the pCN kernel carry over. R7 is gated on R0's gradient/diagnostic checks and
R3's proper-score readout, not on R5–R6, and it is the first target here whose predictive *is*
the object that gets scored: a distribution rather than a label.

This file is the forward work order for `pcnl`; it supersedes the unexecuted priorities of
[the original implementation plan](../docs/plan_bayes_ngpt.md) as a reading guide only.
That document remains the original design/formalization record.
[PCNL_STATUS.md](../docs/PCNL_STATUS.md) remains the measured-results ledger.
The motivating discussion is [ngpt.md](../../test/unfer/ngpt.md), especially §6.5.9,
§6.6 and §9.3–§9.9. Cite those headings rather than mutable line numbers.
No experiment, code fix or Lean compilation is claimed by this planning revision.

## Evidence and what it licenses

The discussion records stable normalized runs that nevertheless lose useful predictions;
centered priors that change reachability and posterior concentration; cold/annealed/metric
controls that do not solve the ambient target problem; and informative cap/latitude
conditionals whose predictions are not the unrestricted posterior predictive.
The November 7 discussion also corrects older optimizer anchors and absolute accuracy values.
Regenerate anchors before comparing any new result with a historical table.

The exact Haar cap law is implemented in [budget.rs](src/budget.rs), while the proposed
dimension budget combines that geometry with measured likelihood contrasts. Use it to
choose candidate ranks, not as a theorem that the likelihood constrains exactly 7–29
directions or that every remaining direction is flat. A few random rays, a local Hessian
and the absence of visits to a rare cap cannot exclude unseen informative regions.
In particular, importance sampling with high ESS among observed prior draws can still
miss a small region with large likelihood. Record the overlap limitation.

For a cap of prior mass `p`, a rigorous likelihood-ratio bound `exp(B)` between its
interior and exterior would bound posterior odds by `exp(B) * p / (1-p)`.
Ray-based estimates of `B` do not establish that uniform bound. The measured budget and
this conditional inequality must remain distinct. Use the exact incomplete-beta cap
probability for calculations; reserve its large-dimension approximation for interpretation.

The implementation is finite dimensional with isotropic covariance, not a trace-class
infinite-dimensional Gaussian model. “Dimension robust” needs a controlled scaling study;
bounded gradients and finite gradient-noise variance alone do not prove posterior accuracy,
ergodicity, mixing, or a nonvanishing acceptance rate uniform in dimension.

## Implementation inventory

Paths here are relative to `pcnl/`; existing modules are reuse points, not completed
solutions to the new milestones.

| Component | Existing surface | Next use |
|---|---|---|
| Model and geometry | `src/model.rs`, `src/normalize.rs` | Forward/gradient contract |
| Ambient SG engine | `src/sampler.rs` | Approximate-kernel comparator |
| Direction kernels | `src/sphere.rs` | Audited sphere and conditional controls |
| Geometry accounting | `src/budget.rs`, `src/radial.rs` | Rank hypotheses and cap costs |
| Prediction scores | `src/readout.rs` | BMA, proper-score and reward estimators |
| Experiments | `src/task.rs`, `src/experiment.rs` | Shared tasks and baselines |
| Sequence policy model | `src/policy_model.rs` | DT forward, per-step action distributions, own `layout` |
| Decision task | `src/mdp.rs` | Synthetic MDP, behavior corpus, exact and rollout returns |
| Policy readout | `src/readout.rs` (discrete actions), proposed extension | Per-step proper scores plus return-level readout |
| Potential abstraction | `src/target.rs` | `SubspaceTarget` generalized over a potential trait |
| Diagnostics | `src/diag.rs` | Stationarity and Monte Carlo uncertainty |
| Saved draws | `src/persist.rs` | Trajectories plus run metadata |
| Reproduction | `../scripts/pcnl-reproduce` | Named evidence stages |

Modules marked *proposed* do not yet exist. Do not count the research plan as an
implementation or treat a command naming a proposed example as an available command.

## R0 — establish numerical and experimental trust before new inference

**Development tasks, in order:**

1. Check the end-to-end `loss_and_grad` against directional finite differences on tiny
   models, multiple parameter blocks and several nonzero norm scales. Include zero and
   near-zero behavior. `normalize_fro` currently calls `to_scalar` on the norm and then
   rescales the tensor: its denominator is outside the autodiff graph. Decide explicitly
   whether the intended derivative is a stopped-gradient surrogate or the derivative of
   the normalized forward map. For posterior-gradient claims implement and test the latter;
   if a surrogate is retained, label it and evaluate its drift bias. An exact MH kernel
   can use a surrogate drift only with its actual proposal density in the ratio.
2. Resolve the schedule contract in `sampler.rs`. `beta_at` clamps to `1e-6`, so the
   implemented sequence eventually has a positive floor, even when the mathematical
   power law has sums `sum beta^2 = infinity`, `sum beta^4 < infinity` for
   `1/4 < kappa <= 1/2`. Tail-mobility formulas for the unfloored law do not describe
   the floored implementation. Expose separate annealing and finite-floor modes; report
   the floor crossover and actual mobility. The exponent window is only one necessary
   schedule condition, not a convergence theorem. Cyclical sampling is a separate mode.
3. Make frozen/nonfinite traces diagnostic failures or explicitly unavailable values.
   `diag::ess` currently returns the draw count for zero variance; identical constant
   chains can produce `Rhat = 1`. Neither licenses a mixing claim. Add a movement flag,
   acceptance/unique-state count where relevant, and tests for stuck chains in both
   identical and different states. Use rank-normalized split/folded Rhat and bulk/tail
   ESS, with independent cross-checks on synthetic traces and a documented convention.
   Use [Vehtari et al.](https://arxiv.org/abs/1903.08008) for the diagnostic definitions.
4. Audit every “exact” proposal against its implemented density. For spherical tangent
   Gaussians, handle the exponential map's multiple preimages beyond the injectivity
   radius, or use a support restriction with the correct forward/reverse normalizers.
   Check restricted-metric determinants, tangent covariance, Jacobians and any safeguards.
   A fixed ambient metric still induces a position-dependent tangent law. Treat fixed
   geodesic shell, latitude and cap-conditioned targets separately. Verify detailed balance
   or invariant moments on circle/S2 targets with known answers before high-dimensional runs.
5. Freeze metric adaptation before stationary draws and record the freeze point. Separate
   warm-up, nonstationary annealing and stationary sampling in reports. Do not interpret
   ordinary ESS/Rhat on an arbitrary decaying-step trace as stationary-chain diagnostics.
6. Add a run manifest beside saved draws: code revision and dirty diff identity, model and
   normalization version, task/data/anchor hashes, split seeds, target/prior/temperature,
   optimizer and sampler configuration, actual schedule, basis/metric hash, device/dtype,
   warm-up/thinning, evaluations, wall time, memory and nonfinite events. Saving draws is
   not exact resumption; resumption additionally needs RNG, loader and adaptation state.

*Status (2026-11-09):* items 1–6 are implemented and measured — `docs/PCNL_STATUS.md`,
"R0", (a)–(f). Item 4's residual sub-audit (cap-conditioned target, cyclical schedule) is
still open and is named there; item 1's surrogate is labelled and its drift bias measured
at 0.14 %, so the recorded anchors stand, but every gradient-level claim must name the
normalisation form it used.

**Acceptance:** finite-difference discrepancies are within documented scale/dtype tolerances;
schedule reports agree with evaluated steps; frozen traces fail the reporting gate; each
exact comparator agrees with a known low-dimensional target within Monte Carlo uncertainty.
Re-run affected old cells after a gradient/model fix. Preserve old logs as superseded data.

## R1 — specify and test the informed-subspace target

**Research hypothesis:** a rank-limited adaptation model can retain the pretrained model's
useful predictions while expressing uncertainty in directions informed by task data.
This is a model-selection hypothesis, not a claim that projection preserves the ambient
posterior. Start with an affine Gaussian model because its prior and MH ratio are simple.

For frozen anchor `theta_ref`, prior covariance factor `C_half` and orthonormal columns
`B` of size `P x r`, define

```text
theta(z) = theta_ref + C_half B z
z ~ N(0, I_r)
pi_r(dz) proportional to exp[-lambda * sum_i loss_i(theta(z))] N(0, I_r)(dz)
```

Specify whether `theta_ref` is pretrained on Task A, independently supplied, or fitted
on Task B. A Task-B anchor or basis defining the prior is empirical Bayes/data reuse;
label it, use a held-out selection split or cross-fitting when evaluating that choice,
and keep the ordinary Task-A-prior experiment as a separate arm. Temperature `lambda`
changes the target; it is not a sampler step size. `lambda = 1` is the likelihood target.

Use the exact coordinate gradient `B^T C_half^T grad_theta Phi`. Candidate bases are
training-trajectory PCA, prior-whitened Gauss–Newton/Fisher directions and a random
orthonormal control at matched rank. A raw loss Hessian may be indefinite; state which
curvature operator and prior metric define “informed”. Compare the tangent construction
with this affine one only after specifying the chart prior and volume factor. Do not
silently interpret a global unit-sphere prior as a product of tensor-wise spheres.

**Development:** add `src/subspace.rs` for basis validation, maps and projected gradients;
`src/target.rs` for target identity and shared full-data potential; and a proposed
`examples/subspace_adaptation.rs` using the existing model/task machinery. Implement
exact MH-pCN in `z` first:

```text
z' = sqrt(1-beta^2) z + beta xi,    xi ~ N(0,I_r)
log alpha = min(0, Phi(z) - Phi(z'))
```

Use the common full-data target at both ends. The Gaussian prior cancellation follows
from this proposal's prior reversibility; it does not hold for an arbitrary Langevin
proposal without its drift-density correction. Validate ranks 1–2 against quadrature
on controlled targets before measuring transformer adaptation. Ranks 4/8/16/32 are an
initial experiment grid, not ranks established by the budget estimate.

The motivation for distinguishing informed directions from the Gaussian complement is
the [DILI construction](https://arxiv.org/abs/1411.3688), whose Gaussian-reference
assumptions must be checked for this target. It does not guarantee useful transformer BMA.

*Status (2026-11-09):* the chart (`src/subspace.rs`), the exact MH-pCN kernel in `z` with
its quadrature check (`src/target.rs`) and `examples/subspace_adaptation.rs` are in — see
`docs/PCNL_STATUS.md` "R1". The cells now run on the **ambient anchor** through shared
code: `experiment::AnchorSetup`/`anchor_hash` are used by both the ambient run and the
example, and a test asserts the hash, so the two tables are on the same `θ_ref`
(`0x52d503556818f1e0`, 4.30 % / NLL 4.1523 on B). Measured at ranks 1–2 with 4 chains ×
4 000 steps: the informed (Task-B trajectory PCA) basis gains 1.21–1.24 nat over the anchor
and the random control gains nothing (identical accuracy, NLL within 0.001) — but the
anchor is *below chance* (12.5 %) and worse than the uniform predictor (`ln 8 = 2.0794`), so
most of that gain is undoing the anchor's bias; the best restricted predictive (rank-2
informed, 14.06 % / 2.9083) is at the base rate with a miscalibrated tail. No cell is
converged at `β = 0.05`, which is R2's scale problem. Brier and the predictive MCSE are now
reported. Open: ranks 4–32, the cross-fitted empirical-Bayes arm, ambient-vs-restricted
budget parity, and an anchor that isn't below chance on B.

**Acceptance:** known-target posterior means/covariances and predictive integrals agree;
random and informed rank controls use identical target scale, baseline and budgets; report
accuracy, NLL, Brier, predictive MCSE and mixing for every rank. An informed basis must
beat or explain its tie with a random basis, not just beat a failed ambient run.

## R2 — separate target restriction, full-target proposals and SG bias

Two different extensions follow R1:

* **Restricted adaptation:** freeze the complement and sample `pi_r`. This is the explicit
  smaller model above. Measure its posterior predictive and compare its point optimizer.
* **Full-target informed proposals:** in whitened coordinates split
  `v = B z + B_perp q`; use informed proposals for `z` and prior-preserving pCN for `q`,
  with the full likelihood and correct acceptance ratios. This changes mixing while
  retaining the declared ambient posterior. If that posterior predicts the base rate,
  accurately sampling it is a negative target result, not a failed kernel result.

The prior-preserving proposal principle is from
[Cotter et al.](https://arxiv.org/abs/1202.0709); an infinite-dimensional claim additionally
requires compatible discretizations and trace-class covariance. Diagonal/block covariance
at finite `P` alone supplies neither condition.

After an exact low-rank reference is available, compare full-gradient unadjusted pCNL,
mini-batch SG-pCNL and SGLD at fixed target, with gradient scaling checked as
`(N/batch) * batch_sum = N * batch_mean`. Vary step size and batch size, track distance
to the reference predictive/moments with uncertainty, and distinguish drift-discretization
bias from stochastic-gradient bias. A clipped gradient is a control and changes the drift;
unconditional acceptance is not “100% MH acceptance”.

**Proposed development:** `src/reference.rs` for exact low-dimensional targets and
`examples/subspace_bias.rs` for shared-budget comparisons. Keep covariance block definitions
and whitening maps explicit. Do not adapt the selected basis while collecting fixed-target
samples unless a valid adaptive-MCMC argument is supplied.

**Acceptance:** measured bias decreases or plateaus with a reported reason as step/batch
controls change; gains are reported as ESS or predictive precision per likelihood/backward
evaluation and per second. Require useful accuracy of the reference target before spending
on a more scalable approximate sampler. No new “dimension independent” claim without
holding the underlying statistical target/discretization family meaningfully comparable.

*Status (2026-11-09):* the first half of R2 is in — per-direction pCN scales on the
restricted target (`mh_pcn_z_scaled`, `curvature_diag`, `DiagScale` in `src/target.rs`,
`--proposal fixed,curv,pilot` in the example; see `docs/PCNL_STATUS.md` "R2"). The kernel
is bitwise equal to `mh_pcn_z` at equal `β_i` and reversible for any `β_i`, so the
preconditioner cannot corrupt the target, only the mixing. Measured, ranks 1–2, ambient
anchor: measured-width (`pilot`) proposals take the informed rank-1 cell from ESS 58 /
`R̂` 1.055 to ESS 680 / `R̂` 1.004 (11× per evaluation) with the same predictive (NLL differs
by 0.0046 at a 0.0002 MCSE, inside the fixed arm's 0.0040), and the random arm from ESS 21 /
`R̂` 1.165 to ESS 1 245–1 899 / `R̂` 1.002
without changing its predictive at all — a converged negative control. Curvature-at-anchor
(`curv`) fails on the informed direction (0.1 % acceptance, ESS 7, NLL 3.6072): at a
nearly-uniform anchor the diagonal reads `β_i ≥ 1`, the cap binds, and the arm takes pCN's
largest step into a steep region. The learned `β_i` span 0.017–0.13 within the rank-2
informed arm, which is the part one scalar `β` cannot express. The rotated extension
(`mh_pcn_z_rotated`, `curvature_hessian`, `RotScale`) is in: exact for any orthogonal `Q`
(identity rotation is bitwise the diagonal kernel; a correlated Gaussian's `H⁻¹` is
recovered to ±0.02 at ESS > 2 000), with a block preconditioner as the special case of a
block-diagonal `Q`. Measured at ranks 2 and 4 it is **not** an improvement: informed
`rotpilot` falls from ESS 152 to 29 at rank 2 (`R̂` 1.012 → 1.096) with the predictive
unchanged (NLL 2.9119 vs 2.9123), `rotcurv` fails like `curv` (`β_i` still clamps at 1,
NLL 3.6051/3.5249), and on the random control the two rotated arms move ESS by less than a
factor of 1.6 either way with the predictive identical to four decimals. The reported
`β_i` ranges nearly match the diagonal arm's, which points at eigenvector noise from a
150-draw pilot covariance rather than at the widths. The full-target half is now in too
(`FullTarget`, `mh_pcn_z_full`, `mh_pcn_w_full`, `adapt_beta` in `src/target.rs`,
`--targets restricted,full` in the example): `π_P ∝ N(0,I_P)(v)·exp(−λN·mean_loss)` with
`v = Bz + w`, `w ⊥ span(B)`, an informed `z` move and a projected-complement `w` move,
each scored with the full likelihood and each `π_P`-invariant, with `β_q` adapted during
burn and frozen. Exactness is tested two ways: a `w`-blind potential makes the `z` stage
bitwise the diagonal kernel, and the coupled Gaussian's `Σ = [[0.6,−0.4],[−0.4,0.6]]` is
recovered at ESS > 2 000. Measured at ranks 1, 2, 4 plus a rank-2 random control (4 ×
4 000, ambient): `π_P` gives NLL 1.83–2.03 / Brier 0.84–0.88 against `π_r`'s 2.87–2.95 /
0.99–1.01, and the *random* basis does just as well on `π_P` (1.97–2.04 / 0.865–0.870)
where it collapses to the anchor on `π_r` (NLL 4.15). So the restriction, not the rank or
the mixing, was the limit, and with full support the informed basis is a `z` jump schedule
only. Cost: two evaluations per step; ESS(Φ) 5–10 and `R̂` up to 3.35, so the ~0.9-NLL gap
(12–20× the MCSE) is the measured claim and the absolute NLL is not converged. Open:
per-coordinate or Hessian-informed `β_q` — the adaptation lands at 0.05–0.08, far below
prior scale, which is where the mixing gain would be — longer full-target chains, a pilot
long enough (or shrunk) to pin the rotated eigenbasis, ranks 8+, `λ ≠ 1` / `σ ≠ 0.35`, and
the SG-bias comparison.

The rank scaling of the pilot arm is now measured too (R2d, `π_r`, 4 × 4 000):
ESS(pilot)/ESS(fixed) is 11.7× at rank 1, 4.9× at rank 2, **0.15×** at rank 4 and
unresolvable at rank 8 (ESS 7 vs 6, `R̂` 1.289 vs 1.931, so neither window is a sample and
the 0.31-NLL gap is not quoted). Dimension alone is not the cause: the random rank-8
control still gains **31×** (28 → 872, `R̂` 1.001, predictive unchanged), and a 1 200-step
pilot at informed rank 8 lifts ESS 6 → **24** and `R̂` 1.931 → 1.091. What binds is the
pilot's 150 estimation draws spread over `r` coordinates — 150 at rank 1, 38 at rank 4, 19
at rank 8 — so the 300-step default is a low-rank budget, and `rotpilot` at ranks 4/8 (ESS
9 / 4) remains worse than the diagonal pilot at both. The length rule is not 150·`r`: rank
8 needs 2 400 steps (300·`r`, ESS 6 → 13 → 24 → **137** at 300/600/1 200/2 400) to reach
`R̂` < 1.05, while at rank 4 600 is already past the knee (ESS 15 → 221).

R2e measured the width question directly with `DiagScale::from_widths_shrunk`
(`--pilot-shrink s`: `log w_i ← mean(log w) + s·(log w_i − mean(log w))`, exact for every
`s`, `s = 1` bitwise the raw pilot). At the unchanged 300-step pilot, rank 4 goes
15 → **399** (`s = 0.5`) → **383** (`s = 0`), so its loss is width *noise*, not magnitude,
and 300 shrunk draws beat 1 200 raw ones (245). At rank 8 `s = 0` gives ESS **154** / `R̂`
**1.010** — better than 2 400 raw draws at 67 % of the cost — while `s = 0.5` is the worst
cell measured (7 / **2.164**), so partial shrinkage is not a safe interpolation and one
`--seed` sweep should check whether that non-monotonicity is a mechanism or pilot luck.
The predictive is flat across all twelve cells (rank-4 NLL 2.8658–2.8699, rank-8 pilot
2.7425–2.8219, rank-4 BMA 12.89 % everywhere), i.e. the effect is diffusion, not bias.

R2f ran that seed sweep (rank 8 at seeds 7 and 8, rank 4 at seed 7) and the size of both
headline numbers was seed luck. At rank 4 the *ordering* replicates — `s = 0.5` best at both
seeds (399 / **264**), `s = 0` next (383 / 152), raw last — but the raw cell is ESS 15 on one
stream and 97 (better than the fixed arm's 84) on another, so "the 300-step pilot is 6.7×
worse at rank 4" is one width vector, and what is robust is that shrinkage removes the
lottery (shrunk ≥ 152, raw 15–97). At rank 8 the non-monotonicity does not replicate: seed 7
gives 9 fixed / 12 raw / 20 (`s = 0.5`) / 20 (`s = 0`), and a fifth cell at seed 8 with
`s = 0.5` gives ESS 6 at `R̂` **3.560** — the intermediate point is not systematically bad
but bimodal, i.e. a coin flip between best and worst at 19 draws per coordinate. So
`--pilot-shrink` should be reported with ≥ 3 seeds and their spread, and rank-8 shrinkage
is a variance reduction that must be validated by its variance. Predictives are flat within
a seed and dominated across seeds (rank-4 fixed NLL 2.6168 at seed 7 vs 2.8699 default), so
only same-seed rows are comparable. The open follow-up is a replicate-averaged width (two
independent pilots, or averaging the pilot's own half-samples) to attack the noise without
collapsing the spread.

R2g implemented that follow-up as `--pilot-reps n` (n independent pilots restarted at the
anchor, widths averaged per coordinate as a geometric mean in log space, budget
`pilot_steps × n`; `n = 1` verified bitwise identical to the old estimator) and it is
**negative**. At rank 4 the 2-replicate average moves ESS 15→14 (default seed), 97→128
(seed 7), 70→140 (seed 8): a real gain on two seeds and nothing on the one where the raw
pilot collapses. At rank 8 it is flat or worse (6→5, 12→7, 5→6). The equal-budget control
says why: at rank 4 default seed a single **600-step** pilot gives 221 while two averaged
300-step pilots give 14, i.e. the pilot's error is **bias** (each 300-step pilot explores
the same narrow patch and its narrow width vector is a stable estimate of a biased
quantity), not the variance averaging can remove. So the next estimator attempt must be a
*scale* correction — inflate the pilot widths by a constant or estimate them from a longer
pilot — not more pilots. R2g also extended the shrinkage ladder to rank 16: fixed 5 /
`s = 1` 5 / `s = 0.5` 6 / `s = 0` 7 ESS with `R̂` 1.828 / 2.490 / 1.545 / 1.440, so the
direction holds but the gain over fixed falls from 3.8× (rank 4) and 22× (rank 8) to 1.4×,
and the fixed arm itself is at `R̂` 1.83 — rank 16 is below the experiment's resolution
floor and no claim about shrinkage above rank 8 should be made without more chains or an
exactness check there.

R2h then ran R2g's own predicted follow-up — a uniform scale correction, `--pilot-scale c`,
which multiplies the shrunk widths by `c > 1` before `DiagScale` — and the prediction fails
in a way that retracts R2g's mechanism. Because `DiagScale` adapts a scalar multiplier on
the joint acceptance rate during burn, and only `m · width_i` reaches the kept window, a
uniform inflation is absorbed: `m` halves to within 3–6 % at `c = 2`, and ESS and every
predictive are unchanged at ranks 4, 8 and 16 (7/7, 154/139, 383/350). At `c = 4` the
absorption is imperfect (7–18 % high), the frozen kernel is left slightly off-target, and
no cell improves; the rank-8 default-seed drop (154 → 32) is inside that rank's seed spread
(20 at seed 7). So the pilot's contribution is the relative width **shape**, never its
overall size — which is why `s = 0` (shape discarded, size from feedback) wins, and why the
600-step-vs-two-300-step gap is a shape difference rather than a width bias. The next
experiment is a shape question, not a width question: a curvature shape evaluated where the
curvature means something, or a shape pooled robustly across replicates, against isotropic
`s = 0` at ranks 4 and 8. `--pilot-scale` stays in the example as a verified null.

R2i ran the pooled half of that: `--pilot-ebayes` fits a one-way random-effects model to
three replicated pilots' log widths and keeps the fraction `τ²/(τ² + sd_i²/n)` of each
coordinate's deviation, with an unweighted three-pilot pool as the control. The pooled shape
loses at every rank, seed and strength — ESS `s = 0 / 0.5 / 1` is 383/362/17 at rank 4,
154/64/8 at rank 8 and 20/16/7 at rank 8/seed 7, monotone in the amount of shape retained
with the optimum at none — and the weighting is not what fails: weighted and unweighted
pools land at the single-pilot ESS (17 vs 16 vs 15, 8 vs 6 vs 6), so the replicates agree
about the shape and the shape is *wrong* rather than noisy. That closes the pilot arm: every
pilot-derived shape measured loses to discarding it, and the only untested shape source is
a non-pilot one — curvature at a point where `∇ℓ ≠ 0`, so the Hessian describes local
stiffness instead of a stationary bowl. Until that is run, "shrink the pilot" means `s = 0`.

R2j then closed the rank-16 question that R2g and R2h had both parked: before running
another rank-16 arm, does the fixed (proposal-independent) arm become measurable with more
chains and more steps? It does not. ESS(Φ) per 1 000 kept draws is 0.45 / 0.16 / 0.56 / 0.48
across 11 200–49 600 draws and 4–8 chains — an integrated autocorrelation time of about
2 000 draws at every budget — and the same kernel at two seeds gives `R̂` 1.302 vs 1.906,
BMA accuracy 17.58 % vs 23.44 % and NLL 2.1226 vs 2.0576. That same-kernel seed spread
exceeds the BMA step (3.5 points) that R2g's rank-16 shrinkage ladder was asked to resolve,
so the ladder is withdrawn as a measurement, "no claim above rank 8" stands, and a rank-16
result would need a target-level change (a shorter-autocorrelation functional, an exact
reference, or a less stiff subspace), not more draws or a better estimator.

R2k then ran R2i's remaining shape source: curvature at the target's **mode** rather than at
the anchor, which is a stationary point of the training loss and not of the posterior.
`--proposal laplace` finds the mode by Adam on the exact gradient of `Φ` (mode found in
1 157/1 132 steps at ranks 4/8, gradient 5.4e1 → 9.9e-4, printed with every cell) and takes
`β_i = m/√H_ii` there. Two findings. First, `loss_and_grad`'s gradient is **not** the
derivative of the loss value: `forward` defaults to `NormGrad::Stopped`, which freezes the
Frobenius denominators, and the gap is ~1 021× on the example's model, so
`SubspaceTarget::grad_phi` uses `NormGrad::Exact` and a new test pins it against finite
differences. Second, and unlike every pilot-derived shape, this shape **beats isotropic**:
rank-4 ESS 554 (`s = 1`) and 587 (`s = 0.5`) at the default seed, 287 and 373 at seed 7,
against isotropic 383 and 152 — with identical BMA accuracy and `R̂` 1.009–1.017. Partial
shrinkage wins at both seeds, so the recommendation is `laplace` with `--pilot-shrink 0.5` at
rank 4. At rank 8 the same estimator collapses to ESS 8/7 (`R̂` 1.56–4.45) against isotropic
154 and fixed 7, even though the rank-8 mode converges to an equally good potential
(1604.91 vs 1602.06): the diagonal Laplace width is a *local* quantity, and a rank-8
restriction contains directions whose posterior is much wider than the mode's Hessian. The
next question is therefore about rank and non-Gaussianity, not about more shape sources —
whether the rank-4 win survives a different seed set, a different anchor, or `λ ≠ 1`, and
whether a full-covariance Laplace (the `rot` kernel) behaves the same way at rank 8.

R2l answered the `λ` and full-covariance halves of that. `λ` moves the isotropic baselines,
so each cell carries its own `pilot s = 0` run: the gain over the best isotropic arm is
+53 % at `λ = 1` (383 → 587), +22.5 % at `λ = 0.5` (342 → 419) and **−6.6 % at `λ = 2`**
(376 → 351). The sign change is explained by mode displacement — the search needs 926 steps
at `λ = 0.5`, 1 508 at `λ = 2` — i.e. R2k's rank-8 mechanism at fixed rank: a shape tuned to
a mode the chain reaches late is no use while it is still approaching. For the covariance
question, R2l added `rotlaplace` (the same converged mode, the full Hessian there, β
eigen-range 6.5/7.6 so the rotation is not degenerate): ESS 112 at rank 4 against the
diagonal's 587, and 6 at rank 8 against the diagonal's 8 and isotropic 154. The full
covariance is therefore *worse* than its diagonal, with `m` adapted down to 0.27–0.38 to say
so, which retires "the rank-8 collapse is the discarded off-diagonal" as an explanation. The
recommendation stands as `laplace --pilot-shrink 0.5` at rank 4 for `λ ≤ 1`, isotropic at
`λ = 2`, and isotropic at ranks 8 and above; the outstanding question is the *anchor*, which
only a shifted `θ_ref` can answer, and whether a mode-centred proposal that also carries the
approach (a tempered or annealed reference) can keep the rank-8 gain.

R2m ran all three of those, and retracted one of R2l's claims. Shifting the anchor by 10 % of
its rms (`--anchor-jitter`, hash printed) leaves the rank-4 win at 2.3× (378 vs `fixed` 165)
instead of the ambient 5.9× (587 vs 100), and collapses the isotropic baseline off-anchor
(ESS 7, `R̂` 10.18), so the effect is real but its *size* is one anchor's. The tempered
reference failed in every parametrization: curvature at `τ · z_mode` is a mild compromise at
rank 4 (428 against the mode's 587) and **invalid** at rank 8, where 1 of 8 coordinates has
`H_ii ≤ 0`, the width floor makes a `1e4`-scale coordinate, and the chain reports 0.6 %
acceptance, ESS 5, `R̂` 17.96 — `1/√H_ii` presupposes positive curvature and interior points of
`Φ` are saddles. Log-space *width* blends avoid that trap but not the outcome: (anchor, mode)
is catastrophic at both ranks (5 / 6, the anchor half carrying `H ≈ 0` widths of `1e4`), and
(pilot, mode) is below its own isotropic baseline everywhere (rank 4: 383 → 352 → 271; rank 8:
154 → 85 → 108). Finally the second seed flips the `λ = 2` sign (+31 %, 172 → 226, with a
search that hit the 2 000-step cap), so that half of R2l is **withdrawn**: `λ = 2` is
unresolved, `λ ∈ {0.5, 1}` is positive at both seeds. The largest lever found so far is not a
proposal dial at all — the anchor moves BMA accuracy 12.89 % → 14.45–15.23 % while every
proposal dial in R2c–R2m moves it ≤ 0.5 points — so varying `θ_ref` systematically is R2n.

R2n swept it: `--anchor-jitter f` for f ∈ {0.05, 0.10, 0.20, 0.30} × two seeds, rank 4,
`λ = 1`, three arms per cell (21 runs), and added `--width-report` to settle the rank-8
width question — which it did by refuting it. BMA accuracy does track the jitter at the
default seed and monotonically so, 12.89 % → 14.45 % → 14.45 % → 16.02 % → 17.97 % (+5.1
points from f = 0 to 0.30), but not at seed 7 (16.80 / 18.75 / 16.02 / 16.80 for `fixed`),
so the anchor is confirmed as the largest lever on the *answer* and its direction is not a
law: it displaces the prior mean, hence the model, and Task-A's pretrain anchor is evidently
not Task B's best. ESS meanwhile is a lottery rather than a dose–response — at the default
seed `pilot` gives 383 / 607 / 7 / 93 / 17 across f = 0 / 0.05 / 0.10 / 0.20 / 0.30 (f = 0.30
collapses all three arms to 9 / 17 / 51 with `R̂` 1.44 for `fixed`), so R2m's f = 0.10
collapse is one draw from a long left tail and f = 0.05 is the sweep's best cell for every
arm. The rank-4 `laplace` gain is real on average — 1.53× / 1.07× / 54× / 4.33× / 3.00× at
the default seed, geomean 2.43× over nine cells, ahead in 7 of 9 and behind at the two
seed-7 cells with f ≥ 0.20 — and unrelated to anchor distance; the 54× is a pilot collapse,
not a laplace improvement. Finally the widths: the rank-8 mode vector is
`[0.2351, 0.2033, 0.1046, 0.0925, 0.1686, 0.1190, 0.0983, 0.2435]` (norm 0.4771) against rank
4's `[0.2276, 0.1151, 0.0915, 0.0693]` (0.2797), i.e. *broader* in norm and *less* spread
(2.5× vs 3.3×), with β 0.054–0.11 vs 0.059–0.10. No ~100× tightening exists at the mode; the
only such span is R2m's rank-8 `τ = 0.5` cell (β 0.0065–1.00), produced by the `H_ii` floor.
Rank 8 is instead hypersensitive to *spread*: uniform β 0.05 → ESS 7, uniform 0.065 → 154,
the mode's 2× spread → 7 at 45.5 % acceptance and `R̂` 4.45, which is a mixing failure, not a
narrow proposal. The rank-8 recommendation (isotropic) stands with a corrected reason; the
anchor stays open, and what it needs next is a sweep of `θ_ref` at a *fixed* seed and basis so
that the anchor's effect on the model can be separated from its effect on the sampler.

R2o did that, and it retracts R2n's reading of the trend. A new `--anchor-seed k` re-draws only
the Gaussian shift, leaving `D_A`, `D_B`, `θ₀`, the basis draw seed and the test set on
`--seed 20261020`; two directions × four magnitudes give eight anchors on one data split (all
printing data hash `0xba104db4f677b2ec`). At fixed data the anchor draw alone moves BMA
accuracy over **11.72–19.92 %** (pilot arm 12.50–19.92, mean 14.55, sd 2.56 over 8 draws) —
8.2 points, against R2n's 5.08-point trend. The per-magnitude means (14.46 / 13.67 / 16.60 /
13.47 %) are not monotone and the within-magnitude gaps (3.91 at f = 0.05, **6.64 at f = 0.20**)
are as large as the trend, so jitter magnitude is not the lever: the specific displaced anchor
is. Nor is it a better prior mean — two anchors with identical unadapted Task-B accuracy
(4.30 %) give 16.41 % and 12.50 %, and the best BMA cell comes from a 3.52 % anchor. ESS shows
the same thing: at fixed data it spans 8–406, f = 0.20 alone gives 194 vs 16, and the f = 0.30
magnitude that collapsed all three arms in R2n is unremarkable on both controlled draws. One
controlled cell reproduces the rank-8 signature found in R2n: `R̂` 1.742 at 49.9 % acceptance.
The rank-4 `laplace` advantage also shrinks under the control — median 1.19×, geomean 1.42×,
range 0.08–25× over the eight anchors, with the 25× being the pilot's collapse and the 0.08 the
laplace's own — so R2k's 5.9× and R2n's 2.43× are both draw-optimistic. Limitation: the
informed basis is PCA of a Task-B path from `θ_ref`, so `--anchor-seed` holds the data and the
basis *seed* but not the basis *vectors*; the next cell needs `--basis-anchor` (compute the
basis at the ambient anchor, jitter only the sampling reference) to split the prior mean from
the subspace.

R2p ran that split. `--basis-anchor` builds the informed PCA path at the ambient anchor, so all
eight cells share one restricted model (identical basis line, `94.030 -> 93.784`) and differ only
in the prior mean. Pinning the basis **fixes the mixing**: pilot ESS geomean 2.95×, median
213 → 428, minimum 16 → 229, max `R̂` 1.742 → 1.036, and the proposal-independent arm improves
identically (median 38 → 125) — so R2o's collapses were bad *targets* created by a moving
subspace, not bad samplers. It does **not** fix the accuracy spread: 6.64–17.58 % (range 10.94,
sd 3.16, versus 7.42 / 2.56 in R2o), still non-monotone in `f`, and the new low cell is worst on
Brier and NLL too, i.e. a real model effect. With one basis the two arms agree in 6/8 cells (max
gap 2.35 vs 4.30) and accuracy tracks the proper scores at both extremes. Mean accuracy drops
0.9 points against R2o (13.67 vs 14.55 %), which quantifies the data-reuse optimism of the
anchor-coupled basis. Open after R2p: the `laplace` arm was not re-run under the control (8 more
cells), a second pinned basis would test whether the accuracy attribution is basis-specific, and
two directions per magnitude still give a range rather than a variance.

R2q closed the laplace half: the same eight anchors and the same pinned basis, proposal
`fixed,laplace --pilot-shrink 0.5`. The `fixed` rows reproduce R2p's ESS and accuracies exactly,
so the table is one target with the proposal swapped. Against the pinned pilot arm the laplace
gains are 1.09, 1.24, 0.96, 1.17, 0.80, 1.70, 1.58, 1.21 (median **1.189×**, geomean **1.188×**,
6/8 wins, every ratio in [0.80, 1.70]) — the *median* is identical to R2o's **1.186×**, but R2o's
geomean **1.42×** and 0.075–25.06× range are gone. The two spectacular R2o cells were pilot
collapses created by the moving basis (ESS 16 and 38); under the pin the same anchors give 362 and
233, so laplace is only 1.70× and 1.58×. So the typical laplace advantage is a property of the
rank-4 target's curvature at its mode, and the tail was the basis. Against the pinned isotropic
arm laplace is 3.04–4.93× (geomean 4.08×) in all eight cells, with no accuracy cost (7/8 cells
identical, mean 13.96 vs 13.67 %), all mode searches converged in 982–1382 steps, and
`H_ii ≥ 0` in 4/4 coordinates. Laplace ESS now varies only 2.2× across anchors (276–617, sd 110)
while accuracy still spans 6.64–17.58 %, so the sampler no longer tracks accuracy at all. Remaining
open: one pinned basis only (would the 6.64 % cell move under another?), two anchors per magnitude
still yield a range rather than a variance, and `rotlaplace` / `--laplace-tau` are retired on
R2l/R2m grounds without ever being re-run under the pin.

R2r built that second pin: `--basis-anchor-seed 1` starts the informed PCA path at an
independently drawn anchor one anchor-rms from ambient (`|θ| 132.394`, basis anchor hash
`0xfe9ad411b8756ef7`, path `132.394 -> 132.789` in all eight cells, versus R2p's
`94.030 -> 93.784`), so the eight experimental anchors again share one basis that is *not* the
ambient one. The accuracy attribution survives: mean 13.77 % (R2p 13.67), sd **3.34** (3.16),
span **9.77** points (10.94), and the worst cell is the same anchor (`f = 0.30, k = 2`: 8.98 vs
6.64 %). The mixing attribution does **not**: pilot ESS falls to 24 / 46 / 141 / 7 / 154 / 6 /
101 / 6 (mean **60.6** vs 400.5, **6.6×** lower), with `R̂` 1.715 / 14.777 / 16.048 in three cells
and ≥ 1.04 in four, and the `fixed` arm is frozen at ESS 5–12. So pinning removes basis *drift* but
does not by itself give a good preconditioner — R2p's ESS gain was the **ambient** basis, which sits
in the target's typical set. The accuracies stay put while the geometry degrades because accuracy is
a coarse discrete average; the proper scores do not (NLL mean **3.59** vs 2.96, 0.63 nats, at ESS
6–154), so R3's readouts must be gated on `R̂`/ESS rather than compared across bases. Open after
R2r: one pin seed and one magnitude (1.0 rms) — a *near* pinned basis would separate "pinned" from
"near" — and the already-pinned `--basis random` arm was not re-run as the stronger contrast.

R2s replaced the two-directions-per-magnitude design with **six** (`--anchor-seed 1..6`, the first
two being R2p's cells, verified by re-running `f = 0.05, k = 1` to the same ESS, `R̂`, accuracies,
Brier and NLL), so the accuracy spread can be a variance rather than a range. Within-magnitude sd
is **0.94 / 2.39 / 1.66 / 3.35** points against per-`f` means of 13.41 / 13.80 / 12.44 / 10.22 %:
**69 %** of the total variance is inside a magnitude (within MS 5.14, between component 1.72,
`F(3,20) = 3.00`, `p = 0.055`). Cell `MCSE(acc)` is 0.019–0.099, so the direction spread is 20–50×
the Monte Carlo error and is a property of the prior mean, not the sampler. Variances are
inhomogeneous and grow with `f` (Brown–Forsythe `F(3,20) = 4.10`, `p = 0.020`; 0.89 / 5.71 / 2.76 /
11.19), and `f = 0.30` splits 6.64/6.64/8.59 vs 12.11/13.28/14.06, so the loss is directional
rather than a smooth penalty on distance: the mean shift from `f = 0.05` to `f = 0.30` is only 3.19
points (Welch `p = 0.067`). 23 of 24 cells still mix (`R̂ ≤ 1.031`); one (`f = 0.30, k = 5`, ESS 9,
`R̂` 1.461) is a mixing failure even under the ambient pin, i.e. ~1 in 24 directions. So R2p's
"the spread is the prior mean" holds only in the sense of the mean's *location*, and its `f = 0.30,
k = 2` worst cell is now a tie with `k = 3`. Open after R2s: which geometric property of the
direction predicts the loss (projection into the rank-4 basis, or predictive-logit spread — neither
recorded); whether the same direction reproduces under another basis or data split; and the
`laplace` arm has still only been swept at two directions per magnitude.

## R3 — proper-score readouts and decision quality (ngpt workstream A)

Reuse [readout.rs](src/readout.rs) and `examples/readout_compare.rs`. Keep the empirical
ensemble identity separate from a posterior claim: averaging available predictions can
lower Brier without those predictions being equilibrium posterior draws.

For independent states, finite-mixture Brier inflation is `tr(Cov(p))/m`. For stationary
correlated states the exact inflation is instead

```text
tr Var(p_hat_m) = [m tr Gamma_0 + 2 sum_{k=1}^{m-1}(m-k) tr Gamma_k] / m^2
```

where `Gamma_k = Cov(p_t, p_{t+k})`. Test both against analytic chains or exact
enumeration. Nonstationary states also carry bias. Random subsets of one finite retained
pool answer a finite-population sampling question, with its own covariance correction.
Do not use the iid formula as a certificate for a correlated trajectory.

Compare point Adam, temperature-scaled Adam, matched-cost independent point ensembles,
single draws and BMA at matched evaluation cost. Tune temperature on development data
only; report NLL/Brier as primary proper scores, accuracy as utility, and ECE/AURC/coverage
as diagnostics. Use paired evaluation inputs and bootstrap at the data/seed level;
chain draws are not independent experiment replicates. Include in-domain, noise and
declared distribution-shift tests.

For the RLCD extension, first verify the reward estimator's target gradient. Averaging
parameter-state predictions changes a rationale-mixture estimator; a leave-one-out baseline
is not automatically unbiased when states/rationales share dependence. Provide a finite
enumeration test of the actual expectation. Distinguish variance reduction conditional
on the label from unconditional reward variance, which includes outcome noise.

**Acceptance:** independent-run predictive uncertainty covers reference values, and any
claimed score gain survives a strong calibrated point baseline with a reported interval.
If a usable target's chain readout loses to that baseline at equal cost on three seeds,
record the negative result and stop RL integrations pending a new hypothesis.

R7 consumes this section directly rather than restating it: a DT step score is the same
`Rule` applied with `y` the observed action, `Enumerated`/`mean_dist` give the per-step
predictive and its covariance term, and the correlated-Brier inflation formula above is the
correct correction for steps inside one trajectory. Accuracy stays a utility; for a
continuous action head it does not exist at all, and the score *is* the report.

## R4 — add a learnable structured task before external model scaling

Keep modular arithmetic as a regression fixture, including Task B = A and label-shift
controls. `Dataset::materialize` draws from only `K^2` possible input pairs, so larger
`N` mostly changes repeated-example likelihood weight. Add an exhaustive input census,
pair-overlap report, and separate protocols for repeated observations, disjoint-pair
generalization and task shift. Increasing repeats is not evidence of more diverse data.

Introduce a small CPU-compatible task with known signal/noise and an informative point
solution: first teacher–student classification or a tiny sequence task; then a real
tokenizer/batching/checkpoint task only after the interface exists. Compare the same train,
development and test splits across all inference arms. Add label noise/OOD perturbations
with a specified generative process. Establish that the optimizer can solve each task
before diagnosing a posterior or sampler failure. Do not relabel failure on one task as
a general theorem about standard attention or transformer Bayesian adaptation.

**Proposed development:** a task/evaluation trait beside `src/task.rs`, shared likelihood
and minibatch scaling, and one deterministic benchmark fixture. Measure acquisition,
retention on Task A, calibration and task-B accuracy. GPU/checkpoint work needs a separately
budgeted experiment; it is not needed for R0–R3.

R7 is where the "tiny sequence task" above becomes concrete, and it exposes a limit of the
current data model rather than a missing fixture: `Dataset` is a flat list of `(tokens, label)`
pairs drawn from `K^2` inputs, with no trajectory axis, no per-example length and no
split-by-group. A DT corpus needs all three, so R7's task contract (R7.4) is the input to this
milestone and not a duplicate of it.

## R5 — gated extensions of ngpt §9

These workstreams follow an informative target and audited readout. They are optional
branches with their own falsifiers, not prerequisites for finishing the core crate.

| Branch | Minimum deliverable | Advance only if |
|---|---|---|
| B: latent link | Direction/radius interface, per-mode scores | Toy passes; real link available |
| C: reused depth | Shared-parameter map, dense baseline | Protocol and training code exist |
| D: hierarchy | Level scorer and schedule contract | Task/level interface exists |

**B — HybridLink:** consume the sphere-link prototype as an external reference, not an
already integrated Candle model. Compare raw, normalized and radius-bounded links under
matched data/training; measure task utility and per-mode NLL/Brier/ECE. Include learned
temperature/radius controls. Normalization may limit scale drift without improving
calibration. A real integration also requires rollout replay and model compatibility.
An inspector can show predictive disagreement, but that disagreement is posterior
uncertainty only after the sampling/target gates pass.

**C — reused depth:** compare shared and dense models at both matched unique-parameter
count and matched forward/backward compute, reporting the mismatch when both cannot
be matched. Preserve shared-variable identity in flattening and basis construction.
Measure ICL/IWL and head diagnostics per retained state. A smaller `P` can also change
representation capacity and priors. Growing depth changes dimension: copying a posterior
sample is not a reversible-jump kernel; define an explicit new-model pushforward and
re-equilibration, or postpone posterior-preserving dimension changes.

**D — coarse-to-fine:** import level schedules/metrics as data, not NCM training as a
dependency. Validate the level semantics before computing capacity: `length * log K`
is a per-level code ceiling, and cumulative versus residual/innovation codes have different
joint accounting. It is not automatically comparable to a weight-space Haar cap cost.
Measure actual conditional entropy and held-out conditional NLL per level; pre-register
the proposed link to localization, rather than asserting the first ceiling crossing
predicts collapse. Compare flat and block schedules with the same target and frozen
blocks eventually updated; if fine blocks stay fixed, label a restricted target.
For Sudoku use solved/cell/valid-grid metrics, hold out puzzles and avoid full-solution
leakage into the conditioning interface. Start with a tiny structured synthetic hierarchy.

External training runs and GPU reproduction remain blocked on their actual dependencies
until inspected at execution time. Do not inherit an old “host has no torch/GPU” statement
as a permanent environment fact. Keep external benchmark claims in their source records.

## R6 — formal contracts and claims discipline

Read `../docs/plan_bayes_ngpt.md` §7 and ngpt §6.6 against the actual sibling Lean files.
Name projection-domain, covariance and architecture hypotheses; a numerical test is not
a proof, and a theorem about an ideal projection is not a theorem about detached autodiff.
The original [nGPT paper](https://arxiv.org/abs/2410.01131) motivates hyperspherical
representations; it does not certify this nGPT-lite kernel or its stochastic gradients.

Prioritize bridge statements with direct acceptance tests: projected-gradient chain rule;
Gaussian subspace pushforward; prior reversibility of pCN; correlated Brier decomposition;
and the finite-floor schedule distinction. Only then consider a rigorous cap-odds bound
with its global likelihood assumptions. Match the implemented epsilon map and parameter
domain explicitly. Keep exact density algebra, floating-point tests and empirical mixing
claims in separate evidence categories.

Faris–Lavine, nested-Fock field theory and Hashimoto solver guarantees are not required
to validate this statistical crate. A proposed bridge to those topics needs its own
operator/domain statement; do not infer it from a Lipschitz/noise theorem. MM0 export or
nanoda checking is a separate proof-artifact task, not evidence of sampler stationarity.

## R7 — decision transformers: the predictive is a policy (application track)

**Hypothesis.** A posterior over a small DT's weights yields a predictive action distribution
`p̄(a | o, g)` that is better calibrated and lower-NLL than a tuned point DT at matched
likelihood-evaluation cost, and the induced policy — sampling from `p̄` or decoding with the
averaged distribution — retains the behaviour policy's return without giving up more than its
return MCSE. Both halves are measured; neither is assumed from the classification results.

**Status: none of this is measured.** No DT model, task or cell exists; every code path named
in R7.3 is proposed or an extension of an existing one. Do not cite R0–R2 numbers as DT
evidence, and do not add DT rows to `PCNL_STATUS.md` until a cell has run.

### R7.0 What the accumulated record says about this direction (review)

Three measured results constrain the DT design more than any new proposal idea:

1. **The prior mean, not the proposal, moves the answer.** R2n–R2s: 69 % of the accuracy
   variance sits *within* a jitter magnitude, variance grows with magnitude while means differ
   only at `p = 0.055`, the single failure in 24 directions is a mixing failure under the
   ambient pin, and 1.19× is the median `laplace` gain over the pinned pilot arm (R2q). For a DT
   the prior mean is a *behaviour corpus* and a pretraining split, so the first DT experiment
   must vary that axis (R7.4's adaptation corpus), not the preconditioner.
2. **The restriction, not the mixing, was the limit.** R2's `π_r` cells lose ~0.9 NLL to the
   full target `π_P` and the random basis does equally well on `π_P`. A DT's informative
   directions plausibly live in the action head and the return-to-go embedding, not in the
   trunk, so a rank-4 restriction over the whole flat vector is the most likely way to destroy
   the model. Declare the basis target (head-only, trunk-only, all-blocks) before the sweep.
3. **Proposal gains are target-specific and do not transfer.** `laplace` at rank 4 beat
   isotropic in all eight pinned cells yet collapsed at rank 8 (R2k), off-anchor (R2m) and
   failed for the full covariance (R2l); a non-ambient pinned basis degraded ESS 6.6× with the
   scores nearly unchanged (R2r). Inheriting `laplace` as the DT default would be an untested
   transplant, and the DT adds blocks (head, RTG embedding, layer norms) whose curvature
   differs by orders of magnitude from the trunk's.

### R7.1 The target

For trajectory `j` of length `T_j` with observations `o_{j,t}`, actions `a_{j,t}` and
return-to-go `g_{j,t} = Σ_{k≥t} r_{j,k}`:

```text
theta(z)  = theta_ref + C_half B z
theta_ref  = anchor fitted on corpus A (pretrain); adaptation data = corpus B, split by trajectory
Phi(θ)    = λ Σ_j Σ_{t=1..T_j} −log p_θ(a_{j,t} | o_{j,≤t}, a_{j,<t}, g_{j,≤t})
π_r(dz)   ∝ exp[−Phi(θ(z))] N(0, I_r)(dz)
```

The kernel, prior, chart and dimension budget are R1's, unchanged. What has to be declared
because the classification target never forced it:

* **The likelihood is behaviour cloning, not optimality.** `g` is a covariate copied from the
  data; `p_θ(a | o, g)` is trained to reproduce the actions that *were* taken at that return
  level. A posterior over `θ` is a posterior over a calibrated behavioural model of the corpus.
  No claim about the optimal policy follows, and "return-conditioned" is not a likelihood for
  "this action is optimal at return `g`".
* **What `N` counts.** `Φ` sums per transition, but the independent unit is the trajectory.
  Report trajectories, mean `T`, total transitions, and the ratio of `Φ`'s spread over prior
  draws to the classifier cells' — a 10^4-transition corpus with `P ≈ 2·10^4` is far stiffer
  than `N_train = 512`, so `λ = 1` is a different effective concentration here and must be
  stated rather than inherited.
* **Per-step distribution family.** Discrete actions reuse `model.rs`'s categorical
  cross-entropy exactly, so `readout.rs::Rule` applies per step. A Gaussian head `(μ, log σ)`
  needs a declared floor on `σ` and a check that no prior or posterior draw has a non-finite or
  degenerate per-step NLL; an unbounded optimum in the output parameters is a target defect
  that a sampler cannot fix, and it breaks the coherence between `Phi`, the BMA and the ESS
  diagnostic before any mixing claim is made.
* **One posterior or one per return level.** A single posterior over the mixed corpus and a
  stratified posterior per `g`-bin are different models; BMA over the first is not the
  conditional predictive for a target return. Every cell names which one it is.

### R7.2 Why the predictive is not the answer

`p̄(a | o, g) = ∫ p(a | o, g, θ) Π(dθ)` is a distribution, and a DT decodes a policy from it. Two
non-identities matter:

```text
E[return | π̄]  ≠  E_{θ~Π}[ E[return | π_θ] ]        (return is a functional of whole trajectories)
sample from p̄  ≠  decode with p̄                            (a per-step draw vs a mean action)
```

Averaging over posterior states changes the state distribution each subsequent decision is made
in, so the BMA's Brier/NLL advantage (a per-step statement) does not imply a return advantage.
Report four readouts separately and never merge them: per-step action NLL/Brier/ECE of `p̄`;
return of a policy sampled from `p̄`; return of the averaged-distribution decoder; return of a
single posterior state (the point readout). Their differences are decision effects, not score
effects.

### R7.3 Development, in order

1. **`Potential` trait in `src/target.rs`.** `SubspaceTarget` is currently typed to
   `(Dataset, Shapes, Variant)`, so no other likelihood can sit under the subspace machinery.
   Extract `log_phi(θ)`, `grad_phi(θ)` and `n_terms`, make `SubspaceTarget` generic over it, and
   keep the classification impl as a thin wrapper. Test: the trait path reproduces the existing
   pinned-cell potential **bitwise** at a fixed `θ` and data hash, so R0–R2 evidence is not
   disturbed by the refactor.
2. **`src/policy_model.rs`.** Causal DT forward over tokenized `(g, o, a)` with a padding mask,
   returning per-step logits (or `(μ, log σ)`) and a summed per-step NLL with both `NormGrad`
   conventions. Own `Shapes`/`layout`/`flat_size` with the new blocks (RTG-bin embedding,
   position, action head, optional `σ` head, LN affine), so `unflatten`, `init_flat` and
   `anchor_hash` keep working. Shapes are an initial grid, not budget-derived: `d = 32`,
   `ff = 64`, 2 layers, ≤ 8 RTG bins, ≤ 10 actions, `T ≤ 16` gives `P ≈ 1.7·10^4`, the same
   order as the classification cells so the sampler budgets carry over. `Variant::Standard`
   first (the anchor-comparable baseline); the cosine variant is a second arm with its own
   formal-claims tests, not a default.
3. **`src/mdp.rs`.** One synthetic MDP with an exactly computable optimum: 2-D linear-Gaussian
   dynamics, 5 discrete actions, horizon 16, 4 RTG bins, a behaviour policy with declared
   exploration (`ε = 0.2`), 96 train / 32 dev / 32 test *trajectories*. The exact optimal
   return comes from value iteration (or the LQR solution); the behaviour policy's return is the
   second reference. Splits are by trajectory seed; a step-level split leaks across time.
4. **`src/policy_readout.rs`.** Per-step scores (reusing `readout.rs` for discrete actions),
   ECE of `p̄`'s max probability, and the return-level readout: `K` rollouts of a policy sampled
   from `p̄`, `K` of the averaged decoder, paired differences against each baseline with their
   own MCSE. Return MCSE is reported next to every return number.
5. **`examples/dt_adaptation.rs`.** Same CLI as `subspace_adaptation` (`--anchor-*`,
   `--basis-anchor*`, `--proposal`, `--pilot-shrink`, `--targets`) plus `--traj`, `--horizon`,
   `--rtg-bins`, `--head softmax|gauss`, so every measured arm is a flag rather than a fork.
   `experiment::AnchorSetup` is classifier-typed (`data_a`, `data_b`, `theta0`); parameterize it
   on the potential and pretrain step rather than duplicating it, so the anchor hash keeps its
   one implementation and the DT anchor is comparable across cells.
6. **R0 checks for the new model, before any adaptation cell:** directional finite differences
   on the DT at ≥ 3 norm scales; the `Exact` vs `Stopped` gap measured (R2k found 1 021× on the
   classifier, and a DT's normalised blocks will differ); mask invariance (a padded batch equals
   the unpadded sequence within tolerance); Gaussian-head density integrates to 1 per action
   dimension; the minibatch scaling identity per *sequence*, not per token.

**Status (2026-11-12).** Items 1–3 exist as code with unit tests: `Potential` /
`ClassifierPotential` with a bitwise seam test at the ambient `D_B` hash (`0xba104db4f677b2ec`),
`policy_model` (layout, causal forward, per-step logits, summed NLL, `DtPotential`), and `mdp`
(value iteration, refinement gap, behaviour corpora, trajectory splits, DT tokenisation).
`DtSpec::bin` is `[rtg_lo, 0]`-clipped and `mdp::MdpSolution::optimal_value` is the interpolated
Bellman optimum rather than `interp(V_0)`, because the two differ by ~0.12 out of a −1.99 return
off-grid and only the first matches the policy that is rolled out.

Items 4–5 exist too, and they are the readout rather than a result: `policy_readout` has
per-step proper scores under all three `Rule`s (its log column equals `dt_loss` to 1e-5 on the same
batch, pinned by `dt_step_scores_equals_dt_loss`), max-probability calibration with per-bin counts,
and arm-level return readouts with paired MCSEs. Two negative results are pinned rather than
hidden: max-prob ECE is *not* a proper score (its own doc says so, and the overconfidence test
asserts the ECE identity instead of a direction), and on this MDP the common-random-numbers pairing
buys almost nothing — the arms' trajectories diverge, so the measured covariance is near zero
(paired SE 1.31 vs independent 1.25 at the default arms), which is why variance reduction is
tested on controlled synthetic data and only *published* for the MDP.

Item 6 is now measured (`examples/dt_gradient_gap`, seed 7, 480 parameters, 96 transitions):
under `Variant::Standard` the Exact and Stopped derivatives are the same tensor
(`‖g_e‖/‖g_s‖ = 1.0000`, no Frobenius normalisation in the forward pass), while under
`Variant::Cosine` the exact gradient matches a central difference to 7.4e-6–6e-4 at `h = 1e-2`
across `‖θ‖` of 0.22 / 2.18 / 21.8 and the stopped gradient is off by 2.2e-2–4.0e-2 in the same
direction — with gradient norms agreeing to four digits (ratio 0.9997–0.9999). The gap is a
rotation of the update, not a rescaling: a norm check would have declared the conventions
identical.

**The readout half of R7.4 now exists and has run (`examples/dt_cell`, seed 11).** `dt_arm::DtCell`
assembles the ambient pieces — the `PotentialAnchor` anchor (hash `79f9638aa261e3e5`), two behaviour
policies over the same MDP, unadjusted SG-pCNL chains at the full DT target — and reports the three
columns R7.4 asks for: per-step NLL/Brier/spherical, max-prob calibration with per-bin counts, and
paired returns for behaviour, anchor(sample, argmax), `p̄`(sample, argmax) and the exact optimum.
Measured on test (128 transitions): anchor NLL 1.5925 / Brier 0.7858 / top-1 0.344 / ECE 0.215 vs
`p̄` 1.1056 / 0.5434 / 0.680 / 0.177. Returns at `x0 = [1, 0.5]`, 96 rollouts: behaviour −6.09 ±
0.34, anchor-sample −3.64 ± 0.13, anchor-argmax −2.54 ± 0.03, `p̄`-sample −7.37 ± 0.49, optimum
−2.42 ± 0.02; paired vs behaviour +2.45 ± 0.34, +3.55 ± 0.34, −1.28 ± 0.52 (resolved) and
+0.47 ± 0.33 (within noise) for `p̄`-argmax.

That is the plan's **calibration-only** branch: the predictive improves every per-step score and
top-1 match while its sampled return is *worse* than the behaviour policy's. The same run shows why:
`P = 17 408` and `σ√P = 46.2` against a reached radius of 49.4, so the full-target draws span the
entire prior shell and `p̄` averages over policies that differ far more than the likelihood does.
R7.4's rank-4 restriction is what makes `σ = 0.35` a meaningful posterior width. The Adam point DT,
the temperature-tuned baseline, the matched-evaluation ensemble and the informed basis are still
missing, so this is readout wiring plus a screening observation — not the R7.4 comparison, and every
cell in R7.4–R7.5 remains open with no ledger row owed.

### R7.4 First measured cell (no new dials)

The smallest cell that can falsify the hypothesis. Protocol otherwise R2q's, so the sampler
columns are comparable with the classification tables even though the potential is not:

| Item | Value |
|---|---|
| Model | `d = 32`, `ff = 64`, 2 layers, 5 actions, 4 RTG bins, `T = 16` |
| Anchor | DT pretrained on the Task-A corpus, anchor hash printed |
| Adaptation corpus | same MDP, **different behaviour policy** |
| Rank / λ / prior | 4 / 1.0 / `N(0, σ²)` with the existing σ = 0.35 |
| Basis | informed, pinned at the ambient anchor (`--basis-anchor`) |
| Proposals | `fixed,pilot --pilot-shrink 0` and `laplace --pilot-shrink 0.5` |
| Chains | 4 × 4 000, burn 500, thin 5 (screening: 1 × 1 000 at 2 seeds) |
| Baselines | point DT (Adam, same budget); temperature-scaled point DT tuned on dev; 4-member ensemble at matched likelihood evaluations; behaviour policy; exact optimum |
| Readouts | per-step NLL, Brier, top-1 action match, ECE; return of sampled and averaged policies, with MCSE; ESS/R̂ on mean step NLL |

Budget: the existing CPU-minute caps stay (30 min screening, 4 h promoted), and each cell also
records transition-forwards so cells of different `T` are comparable. A cell is 4 × 4 000 × 2
evaluations × 8 sequences × 16 steps ≈ 8·10^6 transition-forwards at full-target cost; screen at
1 chain × 1 000 steps ≈ 2.5·10^5 and promote only cells whose paired difference is already
larger than the screening MCSE.

**Status (2026-11-13): the cell has run and lands on R7.5's negative branch.**  Implemented as
`pcnl/src/dt_subspace.rs` + `examples/dt_subspace` (the ambient `dt_arm`/`dt_cell` readout is
unchanged).  Deviations to read first: the corpus is `DtCell`'s 768/128/128 transitions, not the
96/32/32 in the table above (changing the split changes the ambient anchor); screening at `1 ×
1 000` reaches ESS 1.4–40 and is directional only, so the promoted cell was run at 4 × **6 000**
(pilot ESS 335.3 / 377.1 / 537.1 at 4 000).  Promoted, seeds 7/8/9: informed `pilot` passes every gate (ESS
590.9 / 792.4 / 858.8, `R̂` 1.004 / 1.003 / 1.002), informed `fixed` fails all three (142.4 / 89.7 /
121.0, `R̂` 1.014–1.045).  Paired test NLL against the dev-temperature-tuned point DT: **+0.0079 ±
0.0182, −0.0171 ± 0.0455, +0.0814 ± 0.0563** — a tie at all three seeds, no seed past 2σ either
way; Brier and spherical favour `p̄` at seed 8 only (−3.6σ, −4.0σ).  The informed basis does beat
the matched random control by 0.35–0.57 nats, and `pilot` beats `fixed` by 0.009–0.022 nats
at screening and by the whole ESS gap at promotion, so the
mixing story is intact while the accuracy claim is not.  `laplace` at rank 4 (the R2k winner on the
classifier) collapses at this target — mode search capped with `φ` still falling, 2–4 of 4
coordinates with `H_ii ≤ 0`, ESS 1.5–2.3 informed — so it is recorded as a proposer defect and the
"transfer of R2k's rank-4 `laplace` win" question is answered *no*.  Returns are seed-inconsistent
(`p̄`-argmax +0.55 ± 0.25 vs behaviour on seed 9, −0.82 ± 0.27 on seed 8) and are not used.  Per
R7.5 the DT adaptation arm stops here pending a *target* hypothesis (R7.6's basis target or
anchor-as-behaviour-policy axes, not a sampler knob).  Tables, gates, defect list and the exact
commands: `docs/PCNL_STATUS.md` §R7.

### R7.5 Acceptance and falsifiers

* **Positive:** `p̄`'s held-out action NLL beats the temperature-tuned point DT by more than 2×
  the paired MCSE on ≥ 3 trajectory seeds, at `R̂ ≤ 1.01` and ESS ≥ 400 on mean step NLL, with the
  sampled and averaged policy returns each within the point baseline's return MCSE. Report the
  per-step and return halves separately even when both pass.
* **Negative:** if the per-step score ties or loses to the calibrated point baseline at matched
  likelihood evaluations, record it and stop the DT adaptation arm pending a *target* hypothesis
  (a different corpus, basis target or restriction), not a new sampler knob.
* **Calibration-only:** if per-step scores improve and both rollout returns do not (beyond their
  MCSE), that is the claim — better next-action distributions, no decision improvement.
* **Sampling gate:** a DT cell failing R̂/ESS is reported as a target/mixing limit with no score
  quoted as a posterior result, exactly as R2r's non-ambient pin was.
* **Target defect:** any degenerate or non-finite per-step NLL among prior or posterior draws
  invalidates that head's target; fix the parametrisation before running chains.
* **Not claimed:** D4RL-style returns (only the synthetic MDP's return is exact here); a posterior
  over optimality; transfer of R2k's rank-4 `laplace` win before the DT's own rank sweep; anything
  about GPU-scale checkpoints, which need the separate checkpoint surface below.

### R7.6 Open questions this section deliberately leaves

* **Basis target.** Head-only, RTG-embedding-only, trunk-only or all-blocks informed directions,
  against a random control at matched rank. R2's `π_r`/`π_P` result says the restriction is what
  costs; the DT's block structure is where that cost is most likely to be visible.  **Measured
  (2026-11-13, `--bases head,trunk,head-random,trunk-random`, 4 × 6 000, seeds 7/8/9):** no block
  restriction beats the full-space informed basis.  NLL cost vs informed is `+0.183 / +0.028 /
  +0.015` for head-only and `+0.009 / +0.042 / +0.070` for trunk-only, so the informed content sits
  in the **trunk** and the head adds nothing at rank 4 (head-only is 4.0σ worse than the point
  baseline at seed 7, never better on any seed).  The informed-over-control gap survives the block
  matching — six of six block informed arms beat an in-block random control by 0.27–0.57 nats — so
  the gain is the directions, not the coordinates.  Two in-block controls fail the sampling gate
  (ESS 236 / 318), which is reported as a gate failure, not a score.  The **RTG table alone**
  (`Block::Rtg`, 128 coordinates) is not a basis target: it costs +0.540 / +0.436 / +0.340 nats
  against the full-space informed basis, is +4.9…+7.0σ worse than the point baseline, and its
  informed-over-control gap is only +0.010…+0.024 nats against +0.342 (head) and +0.566 (trunk) —
  so the content is in the trunk and not in the return embedding, which leaves the state encoder
  and the per-layer products as the only untested part of the trunk.  Detail:
  `docs/PCNL_STATUS.md` `## R7`, Results 89–95.
* **Block preconditioning.** The per-layer blocks of `RotScale` were never the winning arm for the
  classifier (R2l), but a DT's head block and trunk block differ in curvature by far more than
  the classifier's coordinates do, so a block-diagonal `β` is a live arm here and not a retread.
* **Correlated step scores.** Within-trajectory correlation deflates the effective number of
  scored steps; R3's `Γ_k` formula is the correction, and the DT is the first target where it is
  not optional. Whether the deflation behaves like R2j's ~2 000-draw autocorrelation is measurable
  from the same chains.
* **Continuous actions.** Gaussian head with a `σ` floor, a quantile head, or two-hot binning;
  separately budgeted, after the discrete cell.
* **Joint vs stratified return posterior**, and whether a stratified one is any better than
  conditioning at decode time (which is what the DT already does).
* **Anchor as behaviour policy.** R2n–R2s measured an anchor drawn as a Gaussian shift. A DT
  anchor is a corpus, and the analogue of "anchor direction" is which behaviour policy and which
  return distribution it was fitted on — the axis to sweep first, per R7.0 item 1.  **Measured
  (2026-11-13, `--anchor-feedback/--anchor-epsilon`, 4 × 6 000, seeds 7/8/9):** the axis is first
  order for the prior mean (anchor NLL 0.936–2.290, a 0.69–1.35-nat span across policies) and a
  null for the posterior (`p̄` spans 0.010–0.034 nats; every policy within MCSE of the default).
  The best anchor is the one drawn from the *adaptation* behaviour's own feedback (0.3), 0.26–0.59
  nats below the default — a corpus-match effect, not a quality one, since the sharper policy is the
  worst anchor.  No policy reaches R7.5's 2σ bar (best cell: seed 8, −1.7σ, Brier −2.6σ), so this
  axis does not reopen the arm.  Detail: `docs/PCNL_STATUS.md` `## R7`, Results 91–93.
  **Replication (fresh seeds 10/11/12, with the default anchor as control, six seeds per policy):**
  the seed-8 log-NLL win does **not** replicate — pooled over six seeds the weak anchor gives
  log −0.0130 ± 0.0083 (−1.6σ) versus +0.0220 ± 0.0139 (+1.6σ) for the default — but the
  **proper-rule** improvement does: Brier −0.0214 ± 0.0063 (−3.4σ) and spherical −0.0162 ± 0.0044
  (−3.7σ), holding without the one cell that fails `R̂` (seed 12, 1.012).  That is R7.5's
  calibration-only branch: better next-action distributions, no likelihood win, so the arm stays
  closed.  Detail: Results 96–97.
* **Checkpoint surface.** Loading a real (non-toy) DT needs a declared layout/dtype hash and a
  manifest entry; GPU and checkpoint reproduction stay blocked on their own dependencies and are
  not part of the first cell.

## Milestones, budgets and stop rules

Roles denote ownership to assign, not agents spawned by this planning task.

| Milestone | Dependency | Owner role | Timebox | Review artifact |
|---|---|---|---|---|
| M0: trustworthy baseline | none | numerics/experiment | 1–2 weeks | R0 fixes and manifests |
| M1: subspace reference | M0 | inference | 1 week | Known-target tests and rank sweep |
| M2: target comparison | M1 | research | 1–2 weeks | Random/informed/point comparison |
| M3: bias and BMA | M2 | inference/evaluation | 1–2 weeks | Bias curves and predictive MCSE |
| M4: structured task | M0, then M2–M3 | model/data | 1–2 weeks | Data contract and paired report |
| M5: external branch | M3–M4 | integration | scoped later | Falsifiable branch result |
| M6: Lean bridges | M0–M3 | formalization | scoped later | Compiled hypothesis audit |
| M7: DT policy target | M0, M3; R4 data contract | inference/model | 2–3 weeks | Synthetic MDP + per-step scores + return MCSE |
| M8: DT return posterior | M7 | research | scoped later | Joint-vs-stratified comparison |

Timeboxes are planning estimates; hardware and research failures can change them.
For M2 use ranks 4/8/16/32, three task regimes and three paired seeds as a screening grid;
prune failing cells rather than multiply every optional arm. Add at least four independent
stationary chains for promoted cells. Record a screening cap of 30 CPU-minutes per cell
and an initial promoted-cell cap of 4 CPU-hours; exceeding either produces an inconclusive
result unless a revised budget and expected precision are recorded first.

Operational mixing gates are rank/folded Rhat below 1.01, bulk/tail ESS at least 400 for
declared key observables, no frozen/nonfinite traces, and predictive MCSE smaller than
the effect being claimed. These are investigation thresholds, not convergence proofs.
Monitor train potential, radius, informed coordinates, random complement projections and
predictions; loss Rhat alone is insufficient. Report cells that fail the gate without
presenting their predictive as a converged posterior result. Validate any special
finite-horizon Rhat null without using it to waive failed predictive-precision checks.

Stop or redirect when: R0 exposes a changed gradient/model (regenerate affected evidence);
informed and random subspaces tie (revisit basis hypothesis); exact low-rank targets
cannot match calibrated point utility (revisit the model/prior before SG engineering);
or approximate sampling cannot meet precision per compute (retain the exact small-model
result). A well-sampled weak posterior is a target failure; a useful frozen trajectory
is not a sampling success. Publish negative and inconclusive cells alongside positive ones.

For R7 specifically: stop the DT arm if the per-step proper score loses to the calibrated
point DT at matched evaluations (R7.5); do not spend on return-level comparisons until the
per-step score passes a sampling gate, because a return difference from a mixing failure is
uninterpretable; and do not convert a per-step score win into a decision claim without the
rollout readout and its MCSE. M7 runs after M0 and M3 regardless of M4–M6's progress, since
its dependencies are the kernel checks and the readout, not the external branches.

## Reproduction and definition of done

Run existing commands from the **unfer repository root**. The script stage list is the
authority for available named stages; its default set includes a potentially lengthy test
suite, and `--all` includes many hours of historical experiments. Do not run it implicitly.

```bash
./scripts/pcnl-reproduce --list
cargo test --release -p pcnl --locked
./scripts/pcnl-reproduce quick
./scripts/pcnl-reproduce budget
./scripts/pcnl-reproduce rhat-null
cargo run --release -p pcnl --example subspace_bound
cargo run --release -p pcnl --example readout_compare
```

Run the existing gradient and numerical tests appropriate to each implementation change,
then the known-target tests added in R0/R1; do not repeat every expensive historical ladder
without a changed hypothesis. Separate build time from sampling time. Store per-cell logs,
manifests, draws or predictive summaries, diagnostics and uncertainty intervals together.
Every reported table must be regenerable from those artifacts and identify its anchor.

The core plan is complete when R0–R4 deliver an audited target/kernel distinction, exact
small-model references, a tested informed-subspace comparison, a quantified SG bias budget,
and a properly calibrated readout study on a learnable task, whether the primary research
hypothesis succeeds or fails. R7 is the application deliverable and is complete when one
synthetic MDP has a measured per-step score table *and* a rollout-return table from the same
chains, with the negative or inconclusive cells published beside the positive ones; the
checkpoint/GPU arm is a separate follow-on. Record the result in `../docs/PCNL_STATUS.md` and
update the discussion only after results exist. External branches, billion-parameter scaling and
infinite-dimensional convergence are separate follow-on projects.
