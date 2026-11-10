# pcnl research and development plan

Date: 2026-11-08. Implementation inspected at unfer commit
`5b4b79c77f8c35bb4039ab1c755733621b5cd68f`.

## Objective and reading order

Develop a reproducible Bayesian-adaptation laboratory that can answer two questions:
does a tractable, explicitly specified posterior improve predictions over a strong point
baseline, and can a sampler approximate that posterior within a measured compute budget?
Treat target quality, sampler accuracy and decision/readout quality as separate experiments.

The first research bet is **adaptation in a likelihood-informed parameter subspace**,
followed by a full-target sampler with different proposals inside and outside that subspace.
Do not start another ambient step-size/temperature ladder as the default next experiment.
The accumulated negative results make target specification and verification the next gates.

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
| Diagnostics | `src/diag.rs` | Stationarity and Monte Carlo uncertainty |
| Saved draws | `src/persist.rs` | Trajectories plus run metadata |
| Reproduction | `../scripts/pcnl-reproduce` | Named evidence stages |

Proposed new modules below do not yet exist. Do not count the research plan as an
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
hypothesis succeeds or fails. Record the result in `../docs/PCNL_STATUS.md` and update the
discussion only after results exist. External branches, billion-parameter scaling and
infinite-dimensional convergence are separate follow-on projects.
