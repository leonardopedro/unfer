# pCNL × nGPT-lite — measured results (status doc)

> **What this is.** The Week-3 artifact of
> [`docs/plan_bayes_ngpt.md`](plan_bayes_ngpt.md): the full unadjusted
> SG-pCNL experiment run end-to-end (Task-A pre-train → adaptation to
> Task B → diagnostics → BMA), quoted verbatim, with an honest reading
> against the plan's Phase 0 / Phase 4 claims — including the claims
> that did **not** come out clean.

> **Run (2026-10-20).**
> `cargo run --release -p pcnl -- --report-file OUT.txt --save-dir DIR`
> (defaults: seed 20261020, 10 000 steps × 4 chains × 2 variants,
> P = 16 672, σ = 0.35, β₀ = 0.3, κ = 0.55, γ = 1e-3, batch 64).
> Deterministic in the seed. The sequential run's cosine variant
> took 377.7 s on CPU; with Phase 2's `rayon` parallel chains the
> identical run takes 263.1 s (see §"Parallel chains").

## TL;DR

| Variant | Phase-0 outcome | radial ‖θ‖ (σ√P = 45.19) | BMA acc / NLL / ECE | SGD-on-B acc | ESS(loss) | R̂(loss) |
|---|---|---|---|---:|---:|---:|
| **cosine / nGPT-lite** | stable, 10 000 unadjusted steps, no MH | mean 45.5, final 45.7, max 61.1 | **54.7 % / 1.0686 / 0.1589** (K=2000) | 14.1 % | 18 | 1.115 |
| **standard softmax** | **EXPLODED at step 3** (non-finite gradient) | max 6.9e18 | n/a (chain died) | 10.9 % | — | — |

Chance level is 12.5 % (8 classes). The plan's Phase-0 hypothesis is
**confirmed for this config, measured**: the unadjusted update survives
only under the normalized geometry. The plan's §3B radial claim holds
(‖θ‖ stayed at σ√P while the angular components adapted — BMA learned
Task B from a ≈-chance start).

## Report (verbatim)

```text
SG-pCNL (unadjusted) x nGPT-lite toy run (seed 20261020, sigma=0.35, beta0=0.3, kappa=0.55, gamma=0.001, batch=64, chains=4, steps=10000, burn=5000, thin=10)
[cosine/nGPT-lite] P=16672
  pre-train A: loss 4.8819 -> 2.0743, acc_A 12.9%
  B baselines: before 9.4% (NLL 2.0939), SGD-on-B 14.1% (NLL 2.0953)
  SG-pCNL: beta 0.3000 -> 0.0802 over 10000 steps | max scaled|g| 10573.046 max batch loss 4.517 | radial final 45.7 max 61.1 mean 45.5 vs sigma*sqrt(P) 45.2 | ESS(loss) 18 ESS(coord)med 15 | Rhat(loss) 1.115 Rhat coord med 1.088 max 1.380 | BMA 54.7% NLL 1.0686 ECE 0.1589 (K=2000) last-state 41.3% | 377.7s
  saved: 4 trajectory file(s): cosine_nGPT-lite_chain{0..3}.safetensors
[standard softmax] P=16672
  pre-train A: loss 2.1850 -> 851.3527, acc_A 16.0%
  B baselines: before 12.9% (NLL 1151.1021), SGD-on-B 10.9% (NLL 379.6498)
  SG-pCNL: EXPLODED at step 3 (max scaled|g| 1252694024522353344512.000, gradient non-finite, max batch loss 8945304553454643859310321336320.000, radial max 6882751474854526976.000 vs sigma*sqrt(P) 45.192)
```

The saved-path list is shortened above; the report file prints the
absolute paths of all four `.safetensors` files. The quoted block is
the **sequential-chains** run; the `rayon` run below reproduces every
field byte-for-byte except the wall time (263.1 s) and the save paths.

## Parallel chains (`rayon`, Phase 2)

Chains were made RNG-independent (per-chain noise + batch-index
streams) and are now driven by `rayon::into_par_iter`, merged in chain
order. Measured on the full run (same command, same seed, CPU):

| | sequential | with `rayon` (4 chains) |
|---|---:|---:|
| cosine variant wall time | 377.7 s | **263.1 s (1.44×)** |
| full command wall (incl. 37 s build) | ≈ 440 s | 411 s |

The speedup is below 4× because only the chain loop runs in parallel —
the Task-A pre-train, the SGD-on-B baseline, diagnostics, BMA and the
`safetensors` write are sequential in both runs. Verified, not assumed:

- quick-config report before/after: **identical** modulo timing;
- full report before/after: **identical** modulo timing and save paths;
- the four saved trajectory files: **byte-identical** (`cmp`).

## Reading against the plan

**Phase 0 — confirmed, as measured.** The standard dot-product variant
died at unadjusted step 3 with a non-finite gradient (scaled ‖g‖ ~1e21,
radial ~7e18 against σ√P = 45.2); its deterministic Task-A pre-train had
already blown up to loss 851. The cosine variant ran all 10 000
unadjusted mini-batch steps with no Metropolis filter anywhere, radial
mean 45.5 / final 45.7 vs σ√P 45.2 (max 61.1 is a transient), exactly
the §3B claim. This is one config, one seed — not a proof.

**Phase 3 — BMA works, and beats its own baselines *here*.** BMA
54.7 % / NLL 1.0686 from a start at 9.4 % (≈ chance), vs. the
no-adaptation baseline (9.4 % / 2.0939) and this run's SGD-on-B
baseline (14.1 % / 2.0953). Last-state accuracy is 41.3 % < BMA 54.7 % —
the averaging itself is worth ~13 points, which is the plan's argument
for BMA over a single adapted θ.

*Two honest caveats on that comparison:*
- The SGD baseline uses the pre-train learning rate (0.6) for 1 000
  steps — it is **not** a tuned Adam/SGD. "Beats deterministic
  fine-tuning" is not established; "beats this baseline" is.
- The Task-A pre-train converged to the uniform distribution
  (loss 2.0743 ≈ ln 8 = 2.0794, acc 12.9 % ≈ chance), so the chain
  starts from an ≈uninformative θ_A + noise; Task B was learned almost
  entirely by the chain.

**Phase 4 — diagnostics are mixed and are reported as such.**
- ESS(loss) ≈ 18 of 500 kept draws/chain, ESS(coord) median 15 — the
  trajectory is highly autocorrelated.
- R̂(loss) 1.115, coordinate R̂ median 1.088 / **max 1.380** — chains
  have *not* fully mixed in 10 000 steps. Anything downstream of the
  posterior (BMA, ECE) should be read as provisional until R̂ ≤ 1.01.
- ECE 0.1589 (10-bin, BMA predictive): still noticeably overconfident;
  NLL 1.0686 is the better-behaved number.
- The standard variant contributes no diagnostics — it exploded before
  the burn-in; the report says where and with what evidence
  (finite spike recorded *and* `gradient non-finite`), never a silent 0.

**Radial check (§4 Phase 4).** Loading the saved trajectories back
through `pcnl::persist::load_trajectory` gives 500 samples × P=16 672
per chain with per-sample norms in [44.5, 46.0] (chain 0) and
[44.7, 46.0] (chain 3) — concentrated around σ√P = 45.19 while the
predictive distribution adapts, i.e. angular adaptation under a fixed
radius.

## Plan-conformance notes (this run's deviations, stated plainly)

- **`safetensors` (§1, Week 2): done.** `--save-dir DIR` writes one
  file per chain, tensors `s00000…s00499`, each the flat `[P]` f32 θ.
  Files are ~33 MB/chain (500 × 16 672 × 4 B + header); not committed
  — regenerate with the command above. `pcnl/examples/dump_trajectory`
  reads one back.
- **Phase 4 ECE: done** — 10-bin confidence ECE on the BMA predictive,
  rendered next to NLL.
- **Phase 0's JAX reference: skipped deliberately.** The environment
  has no `python3`; the plan called the script a *justified temporary
  alternative*, and the Rust experiment measures the same hypothesis
  directly (both variants, same unadjusted dynamics). The kernel's
  analytic anchor is `sampler`'s quadratic-target tracking test, which
  checks the update formula against a closed-form Gaussian posterior
  mean.
- **`clap` + `indicatif`: not adopted.** The CLI is a small manual
  parser covering the plan's sweep/step-size surface (`--steps --seed
  --sigma --beta0 --gamma --kappa --save-dir --report-file`) plus the
  task×budget matrix knobs (`--task-b p:q:r|b --k --d --ff --layers
  --n-train --pretrain --chains --cyclical --no-mh --init-std`) and
  the centered adaptation prior (`--prior-center-a`, §6.4.2 of
  `test/unfer/ngpt.md`); no progress bar — the run reports wall time
  per variant instead.
- **`rayon` (Phase 2, "parallel chains"): done.** Chains run in
  parallel and are merged in chain order — measured 377.7 s → 263.1 s
  on the full run, with byte-identical output (§"Parallel chains").
- **§3C Option 2 (cyclical β): implemented in
  `SgPcnlScheduler::beta_cyclical`, measured run uses Option 1
  (Robbins–Monro).**

## Corrections (2026-11-04): the schedule's mobility window, and a review's five points

An external review of the write-up (`test/unfer/ngpt.md`) raised five
points. Four are right and one is wrong on a factual detail; the
verdicts below are checked against the engine, not accepted as given.
The one that required a **code** change is point 1.

| # | the review's point | verdict |
|---|---|---|
| 1 | the decreasing-step conditions are applied to `β_t`, but the gradient is multiplied by `h_t = β_t²σ²/2`; with `κ ∈ (0.5,1]`, `Σh_t < ∞`, so the convergence claim fails | **confirmed, now fixed in the crate.** The window is `κ ∈ (1/4, 1/2]`: `Σβ² = ∞` (mobility) and `Σβ⁴ < ∞` (noise). The review's `κ ≤ 1/2` is necessary; the `1/4` end is its missing other half |
| 2 | Theorem 1's gradient bound is vacuous: `1/‖W‖` unbounded, `≈1/ε = 10⁶` with the ε | **confirmed in substance, corrected in wording.** The bound is *finite* but not small where the chain lives (attained only for `‖W‖ ≲ ε`); the drift it permits is ~5.5e3 per step at `β = 0.3`, two orders above the 45-rad shell it is meant to confine. Nothing in the crate ever relied on it for stability — the runs are confined by the pCN contraction, measured |
| 3 | the σ-sweep cannot be evidence: the likelihood is scale-invariant, so `u = θ/σ` evolves σ-free | **confirmed** — and now pinned in code, not just in prose (`sampler::tests::the_kernel_is_sigma_free_in_rescaled_units`). The 4-width agreement is a units check by construction |
| 4 | the `P/2` "entropy wall" conflicts with the Haar prior on direction and fails its own shift cell | **confirmed; already withdrawn** (write-up §6.4, 2026-11-03). The radial marginal is the prior's `σχ_P` because the likelihood is constant on rays; concentration in *direction* is a log-volume question, and finding it is a reachability question |
| 5 | the standard-transformer baseline is a strawman ("no normalization", pre-train blew up), and Kim et al. does not say what it is cited for | **right about the confound, wrong that it rescues the baseline.** `model.rs`'s standard variant *is* an ordinary pre-LN transformer; Kim et al. is about input-Lipschitz, not weight-space Hessians (write-up fixed 2026-11-03). The pre-train confound was real and `--pretrain-adam` closed it (explosion in 3/3 at steps 4–5) — but a **second** confound sat underneath: the drift is `h_t·g_t`, so the standard stack was being run at a step size nobody chose. `examples/step_scale_control` removes it: capping the standard drift at the normalized geometry's own gradient magnitude (`C` measured, 3 seeds, both configurations) makes the chain survive in 3/3 — so the numerical explosion is a step-size claim and "cannot be sampled" is withdrawn. The surviving chain is *not* a sampler of anything (a capped gradient is not a gradient field), overshoots to 3.6× the shell, and its kept NLL is 4.1–5.7 nats against 2.08 for the uniform predictor |

### What changed in the crate

- **`sampler.rs`**: the module documents the corrected window;
  `SgPcnlScheduler::kappa_in_mobility_window()`,
  `partial_mobility(steps)`, `tail_mobility(from)` and
  `total_mobility()` are the budget as API — `tail_mobility` is `None`
  exactly when the schedule never freezes.
- **The default exponent is now `κ = 0.5`** (`Config::default`), the
  window's edge. Every report prints a `schedule: mobility Σβ_t² = …`
  line, and a finite budget is named as such: "the chain freezes; these
  numbers describe the kernel at the beta's visited, not convergence to
  the posterior". `--kappa 0.55` still reproduces the historical runs.
- **Two tests that asserted the wrong condition were flipped**: the
  scheduler unit test (`κ ∈ (0.5,1]` → `(1/4,1/2]`) and
  `numerics.rs` §9, which read the saturation of `Σβ²` at 900 as a
  *pass* — it is the failure mode.- **`examples/schedule_mobility.rs`** measures the consequence on the
  exactly-solvable target `Φ = ½λ‖θ−c‖²`: the mean obeys
  `m_{t+1} = √(1−β²)m_t − (β²σ²/2)λ(m_t − c)`, so contraction and pull are
  both `O(β²)` and the residual is `floor·exp(−tail(T))` — the remaining
  approach *is* the unspent budget, and the floor is fixed by all of it.
- **`SgPcnlState::grad_norm_cap`** (default `None`) caps the scaled
  gradient before the drift; it exists only for §6.5.2's step-scale
  control and a unit test pins that a non-binding cap is bit-identical to
  no cap. **`examples/step_scale_control.rs`** is that control: it measures
  the normalized arm's own gradient magnitude `C` and re-runs the standard
  stack with the drift capped at `C`.

### The step-scale control (2026-11-04), measured

Reference configuration, `P = 16 672`, 1 000 steps, two chains, three seeds:

| arm | max scaled ‖g‖ | radial max | radial final | kept acc | kept NLL | died |
|---|---:|---:|---:|---:|---:|---:|
| cosine, as published | 0.8–1.1e3 | 1206–1299 | 45.0–45.4 | 11.7–13.3 % | 2.219–2.223 | — |
| standard, uncapped | up to 1.2e24 | 1.7e13–6.8e21 | — | — | — | step 5–6 |
| standard, drift capped at `C` | 0.08–1.1e6 | 110–164 | 44.5–45.1 | 14.6–16.1 % | 4.11–5.72 | — |

Three conclusions, and the third is the one that keeps the write-up honest:

1. **The numerical explosion is a step-size claim.** The uncapped standard
   gradient reaches 2.9e12–1.5e21 times `C`; with that budget imposed the
   chain survives 3/3 (reduced configuration too, 2 000 steps). What dries up
   with the blow-up is not "an unnormalized stack cannot be run" but the
   no-tuning property: `β` only means something against a potential with an
   intrinsic scale, which the normalized geometry has and the standard stack
   lacks.
2. **Surviving is not sampling.** A capped gradient is not a gradient field,
   so the capped chain has no Gibbs target and no posterior interpretation.
   The control isolates arithmetic, not validity.
3. **The surviving chain fails differently and worse.** It overshoots the
   shell 3.6× and its predictive is overconfident-and-wrong (kept NLL
   4.11–5.72 against `ln 8 = 2.08`), where the cosine arm's is
   calibrated-uniform. Reduced configuration: radial final 9.1 / 16.1 / 20.7
   from a shell of 8.22, drifting toward the `r ≈ σ²C ≈ 370` balance point —
   consistent with §2.3's "the drift sees a geometry it cannot be stable in",
   now with the step-size confound removed from the evidence.

### Measured (the example's own output)

Reference numbers `β₀ = 0.3`, `σ = λ = 1`, `c = 100`, `μ* = 50`:

| γ | κ | Σβ² @1e4 | Σβ² @1e6 | budget | residual @1e3 → @1e6 | floor |
|---|---:|---:|---:|---:|---|---:|
| 1e-3 | 0.5 | 215.9 | 621.8 | ∞ | 0.00579 → 0.00001 | — |
| 1e-3 | 0.55 | 191.9 | 449.0 | **900.0** | 0.00540 → 0.00001 | 0 |
| 1e-3 | 0.8 | 114.5 | 147.7 | 150.0 | 0.00386 → 0.00000 | 0 |
| 1 | 0.5 | 0.881 | 1.295 | ∞ | 0.506 → 0.272 | — |
| 1 | 0.55 | 0.594 | 0.727 | **0.900** | 0.602 → 0.481 | **0.3833** |
| 1 | 0.8 | 0.205 | 0.206 | 0.150 | 0.812 → 0.810 | **0.8100** |

Two readings, and the distinction matters:

- **At the reference `γ = 1e-3` the freeze is asymptotic and invisible.**
  The budget is 900 — far more than a 10 000-step run needs — so
  `κ = 0.55` and `κ = 1/2` are indistinguishable in the residual column.
  This is why the numbers in the rest of this document stand as *mixing
  and predictive* measurements. What they cannot be cited for is
  convergence to the posterior, which is the review's point and the
  reason the default moved.
- **At `γ = 1` the floor is in plain sight.** `κ = 0.8` is at 0.8100 by
  step 10³ and at 0.8100 at step 10⁶ — frozen; `κ = 0.55` keeps creeping
toward 0.3833; `κ = 1/2` has no floor and keeps closing at about 19 %
  per decade of steps (0.506 → 0.412 → 0.335 → 0.272).
  `residual(T)·exp(−tail(T))` is the *same number at
  every horizon* (0.383333 and 0.810021 to six decimals), which is the
  sharp form of "the unspent budget is the remaining ability".

### The Haar volume of the task, measured (2026-11-04)

The write-up's §6.4(ii) argued that the direction posterior is a volume
competition and that the favourable set is a `≈ exp(−N·I)` fraction of
the sphere.  That sentence now has a measurement
(`pcnl/examples/sphere_volume.rs`): `R` Haar directions at the shell
radius, scored on the held-out split.

| configuration | Haar-typical CE | best of R | uniform `ln K` | Adam-on-B | reached Adam-on-B |
|---|---:|---:|---:|---:|---:|
| reference, `R = 2000` | 4.967 (sd 0.163) | 4.311 | 2.079 | 1.081 | **0 / 2000** |
| reduced, `R = 4000` | 2.510 (sd 0.245) | 1.571 | 1.386 | 0.963 | **0 / 4000** |

So the favourable Haar fraction is below `1.5e-3` at 95 % in the
reference configuration (the fine-tuned direction is 23.8 Haar-sd out; a
Gaussian extrapolation gives `1e-125`, printed as parametric only),
while the energy side is `NΔ = 512 × 3.886 = 1 990` nats.  The
competition is not tight for *sampling*: no Haar draw comes within
2.2 nats of the uniform predictor, so the posterior's mass on the
informative set is reachable only by moving — §6.4.2's finding.  Two
reading notes the example prints rather than hides: a Haar direction's
logits barely depend on the input, so it predicts one constant class and
its accuracy *is* that class's frequency (12.9 % against an 18.4 % best
constant predictor — hence "beats chance" is the wrong threshold and the
loss ladder is the measurement), and the draw law is radius-free
(mean CE 4.965 at `4×` the shell vs 4.967 at the shell, up to `ε`).

### The exact target on the standard stack (2026-11-04)

§6.5.2's capped unadjusted chain was overconfident per draw (kept NLL
4.11–5.72).  `pcnl/examples/mh_standard_target.rs` runs the same stack
under the **Metropolis-filtered** kernel (capped *proposal*, exact
full-batch likelihood ratio) to ask whether that is the target or the
bias.  Reference configuration, 1 000 steps, 2 chains, 3 seeds:

| arm | accept | radial final | BMA acc | BMA NLL |
|---|---:|---:|---:|---:|
| `θ_A` (reference) | — | — | 8.6–12.5 % | 3.17–28.83 |
| unadjusted, capped | — | 44.4–45.1 | 19.9–42.2 % | 1.84–1.92 |
| MH, `β₀ = 0.3` | 0.3–20.2 % | 44.9–148.5 | 8.6–12.5 % | 2.40–11.08 |
| MH, `β₀/10` | **96.0–98.0 %** | 95.5–134.0 | 18.4–24.2 % | **1.96–2.15** |
| MH, `β₀/30` | 99.2–99.4 % | 119.0–170.6 | 8.6–13.3 % | 2.39–2.91 |
| MH, uncapped | 0.0 % | 122.5–176.3 | 10.9–14.5 % | 10.60–17.14 |

Three findings, and the third is a correction to this document's own
wording:

1. **The overconfidence is not the target's.**  At a proposal scale the
   filter can accept (`β₀/10`, 96–98 %), the exact chain's BMA is
   1.96–2.15 nats — at or just above the uniform predictor's 2.079, not
   the 4–5.7 of the unadjusted arm.  The per-draw overconfidence is the
   unadjusted drift's.
2. **At the kernel's own proposal scale the filter freezes.**  At
   `β₀ = 0.3` acceptance is 0.3–20 %, and with no cap it is **0.0 %** in
   3 of 3 seeds — the standard stack's likelihood varies more per pCN
   step than the prior's noise allows.  That is a proposal-scale
   problem, not a temperature one (`β₀/10` accepts 96–98 %), and the
   temperature sweep below confirms it: `λ = 1` is already usable and
   every `λ < 1` is worse.
3. **Bias ≠ worse predictive.**  The unadjusted capped chain's BMA is
   1.84–1.92 NLL at 19.9–42.2 % accuracy, *better* than the exact chain
   at the same budget: individually overconfident, mutually disagreeing
   draws average into a calibrated mixture (an over-dispersed sampler).
   So §6.5.2's "confident-and-wrong" is now qualified as *per draw*; the
   ensemble's calibration comes from its bias, not from sampling the
   posterior.  The target does move out, though: the exact chain's
   radial final is 95–134 against the 45 shell — but see §6.5.6(c): the cosine
   likelihood is exactly flat in the radius, so that excursion is a
   non-equilibrated start (`‖θ_A‖ ≈ 1338`), not the posterior's preference.

### No usable temperature, and the cosine chain is not over-dispersed (2026-11-04)

Two follow-ups the §6.5.3 result raised, both measured.

**(a) Tempering the standard stack makes it worse, not better.**
`pcnl/examples/temperature_sweep.rs`: target
`π_λ ∝ N(0,σ²I)·exp(−λNL)`, proposal at the usable `β₀/10` with the §6.5.2
cap, raw kept losses, 3 seeds, 500 steps.  BMA NLL at `σ = 0.35`:

| `λ` | 1 | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---:|---:|---:|---:|---:|---:|
| BMA NLL | **2.25–2.34** | 2.62–3.57 | 4.56–6.39 | 8.80–12.62 | 11.82–15.70 | 12.92–17.06 |
| acceptance | 93–98 % | 96–99 % | 95–99 % | 98–99 % | 99–100 % | 100 % |

Acceptance was never binding at `β₀/10`, and lowering `λ` moves the target
towards the prior, whose Haar-typical directions are confidently wrong
(§6.4(ii)) — so the predictive degrades monotonically.  At `σ = 2` acceptance
drops (47.6–56.6 % at `λ = 0.1`) and the predictive is no better.  There is no
temperature at which the standard stack's posterior is informative; `λ = 1`
is the best of the ladder, and it is near-uniform.

**(b) The cosine chain's draws agree.**  `pcnl/examples/overdispersion.rs`
scores each kept draw and the BMA over the same draws:

| chain | per-draw NLL | BMA NLL | gap |
|---|---:|---:|---:|
| cosine | 2.220–2.223 | 2.193–2.194 | **+0.028** |
| standard, capped (§6.5.2) | 4.11–5.72 | 1.84–1.92 | ≈ +3.5 |

So the standard chain's calibration comes from averaging disagreeing,
individually overconfident draws, while the cosine chain's ensemble is worth
0.03 nats over a single draw — "an ensemble that has not left its mode",
quantified.  `pcnl/tests/comparators.rs::mh_drift_cap_changes_only_the_proposal`
pins the licence this rests on: a non-binding cap leaves the filtered chain
bit-identical, and two binding caps return the same kept potential (rel ΔL
2–7 %) and radius (rel Δ 0.0–0.2 %) — the cap moves the proposal, not the
target.

### The annealed start: implemented, tested, null (2026-11-04)

§6.5.4(b) said the cosine ensemble "has not left its mode".  The candidate
fix was a schedule rather than a target: `--anneal N` runs
`λ_t = min(1, t/N)` on the drift only (`λ_t β_t²σ²/2·g_t`), so the chain
starts at the prior and the likelihood fades in while the OU term holds
the radius.  It is implemented in `SgPcnlState::lambda_at` (the SGLD twin
gets the same `λ_t`; the MH comparator keeps its target at `λ = 1`, so
probes run `--no-mh`), it is reported (the artifact says whether the kept
draws are at `λ = 1` — a ramp longer than the burn is flagged), and it
changes nothing on the cosine chain at the published scale:

| `N` | draw acc | BMA NLL | gap | cross-cos | drift/noise |
|---:|---:|---:|---:|---:|---:|
| 0 | 12.89 % | 2.1923 | 0.0271 | −0.0018 | 0.050 |
| 750 | 12.89 % | 2.1923 | 0.0271 | −0.0018 | 0.037 |

(`examples/annealed_start.rs`; `P = 16 672`, 2 chains, 1 500 steps,
burn 750, seed 20261020.  The ratio falls because the second run's first
half is at `λ < 1`, i.e. the ramp is applied — the predictive does not
move.)

The mechanism is the per-step information-to-noise ratio
`r = β_tσ‖g_t‖/(2√P)` — the drift norm against the pCN noise norm.  At
`β₀ = 0.3` it is **0.05**: the proposal replaces a `β² = 9 %` fraction of
the norm with fresh prior noise every step, so the direction is re-drawn
~20× faster than the likelihood can steer it, and the chain reports the
prior's predictive.  At the sharpest admissible proposal (`β = 1`) it is
still only 0.17, and a ramp or a temperature scales its numerator and
nothing else.  That single number explains the uniform-ish predictive,
the 0.028-nat ensemble gap, the monotone harm of tempering, this null,
and §6.4.2's centred chains scoring `θ_A`'s accuracy: **the pCN step's
information-to-noise ratio is bounded by ~0.2 here**, so reachability is
a step-distribution problem, not a schedule problem.

Scope note: the flag is not inert.  At the `quick` configuration
(`P = 4 240`) `--anneal 200` takes the *standard* variant from a step-5
explosion to a surviving run (BMA 12.5 % → 50.0 %, NLL → 1.7622, max
scaled `‖g‖` still 2.8e4).  The null is about the cosine chain at the
published scale, not about the code path.

**The same ratio says where to turn instead: cold `λ`.**  At fixed `‖g‖`
`r ∝ λ`, so §6.5.4's hot ladder (`λ ≤ 1`) was the direction that *lowers* the
information ratio.
`--lambda L` now sets `π_λ ∝ N(0,σ²I)·exp(−λ·NL)` on all three arms (the MH
ratio is tempered in step, so it is exact for `π_λ`; kept losses stay raw)
and the report states the target and its `r`.

| `λ` | measured `r` | BMA NLL | ECE | BMA acc |
|---:|---:|---:|---:|---:|
| 1 | 0.048 | 2.1966 | 0.2128 | 15.6 % |
| 2 | 0.073 | 2.1411 | 0.1424 | 15.6 % |
| 4 | 0.112 | 2.1142 | 0.0970 | 15.6 % |
| 8 | 0.169 | 2.1006 | 0.0686 | 15.6 % |
| 16 | 0.248 | 2.0922 | 0.0499 | 15.6 % |

(`quick`, `P = 4 240`, 400 steps, cosine, one seed; `r` is the report's own
measurement, and its growth is sublinear — `∝ λ^0.6` — because a sharper
target also shrinks `‖g‖`.)  Monotone on both scores, accuracy flat, uniform's
2.0794 still unbeaten: cold `λ` removes overconfidence (ECE four-fold) without
buying accuracy.  Published scale
(`P = 16 672`, 1 000 steps, 2 chains, one seed) agrees and shows the cost:

| `λ` | BMA NLL | ECE | BMA acc | radial max |
|---:|---:|---:|---:|---:|
| 1 | 2.1958 | 0.1897 | 11.7 % | 101.1 |
| 4 | 2.1252 | 0.1018 | 11.7 % | 104.7 |
| 16 | **2.1030** | **0.0670** | 11.7 % | 139.2 |

Accuracy never moves (11.7 % vs the 12.5 % chance line) — calibration
improves while the direction stays uninformative — and the excursion grows
(3.1× the shell at `λ = 16` against 2.2× at `λ = 1`).  So §6.5.4(a)'s "no
usable temperature" was a **hot**-ladder statement; cold is the useful
direction, but temperature cannot close the gap: `r = 1` needs `λ ≈ 20` at
fixed `‖g‖` and `λ ≈ 150` at the measured `λ^0.6`, and the standard variant
dies at steps 2–3 in every cold cell.  The ceiling that matters is a step
distribution that does not pay `β ≤ 1`.

### Three follow-ups to the cold result (2026-11-04)

**(a) Three seeds.**  `pcnl/examples/cold_ladder.rs` reruns the ladder at the
same 1 000-step budget with three seeds (`P = 16 672`, 2 chains, thin 50).
The bands are disjoint, so §6.5.5's single-seed row replicates:

| `λ` | BMA NLL | ECE | BMA acc | `r` | per-draw − BMA gap |
|---:|---|---|---|---:|---|
| 1 | 2.1894 / 2.1899 / 2.2036 | 0.1853 / 0.1889 / 0.1909 | 11.72–13.28 % | 0.054 | 0.0267–0.0297 |
| 16 | **2.0932 / 2.0995 / 2.1155** | **0.0576 / 0.0628 / 0.0642** | 11.72–13.28 % | 0.242–0.249 | 0.0189–0.0302 |

The per-draw NLL falls with the BMA (2.2191–2.2316 → 2.1182–2.1457) and the gap
does not move: cold `λ` calibrates each draw, not the ensemble's diversity.  A
10 000-step matrix is ≈ 15 CPU-hours at the measured 0.3 s/step on two cores,
so the longer horizon is untested rather than done and unreported.

**(b) A step distribution that does not pay `β ≤ 1`.**  Euler–Langevin's
`r = h‖g‖/(√(2h)σ√P)` has no `β ≤ 1` ceiling.  `pcnl/examples/step_distribution.rs`
(400 steps, 2 seeds, 1 chain):

| family | `r` | BMA NLL | ECE | radial final |
|---|---:|---:|---:|---:|
| pCN `β = 0.3` | 0.069 | 2.1948 | 0.1937 | 45.1 |
| pCN `β = 0.9` | 0.238 | 2.5609 | 0.4592 | 47.0 |
| pCN `β = 1.0` | 0.370 | 2.9745 | 0.6003 | 47.5 |
| ULA `h = σ²/8` | 0.326 | 2.1443 | 0.1439 | 17.1 |
| ULA `h = σ²/4` | 0.462 | 2.1804 | 0.1854 | 18.8 |
| ULA `h = σ²/2` | 0.728 | 2.2790 | 0.2668 | 26.5 |
| ULA `h = σ²` | **2.082** | 3.0870 | 0.3765 | 40.7 |

It *does* reach `r ≈ 2.1` without a non-finite state, and the first reading was
"and it is worthless".  **That reading was a radius artefact.**  `π`'s radial
marginal is a thin shell (`σ/√2 ≈ 0.25` wide) around `σ√P`, so projecting the
proposal back onto the shell is a small perturbation of the target — and it
removes the `O(h)` OU bias that had collapsed the ULA chain to `‖θ‖ = 17`.  The
`… pin` arms add that projection:

| family | `r` | BMA NLL | ECE | radial |
|---|---:|---:|---:|---:|
| pCN `β = 0.3` | 0.069 | 2.1948 | 0.1937 | 45.1 |
| ULA `h = σ²/8` | 0.326 | 2.1443 | 0.1439 | 17.1 |
| **ULA `h = σ²/8` pin** | 0.187 | **2.1096** | **0.0774** | 45.2 |
| pCN `β = 0.9` pin | 0.249 | 2.5662 | 0.4618 | 45.2 |
| ULA `h = σ²` pin | 3.5e5 | 2.6604 | 0.4351 | 45.2 |

The pinned ULA at `h = σ²/8` is the best-calibrated arm in the table (ECE 0.077
against the kernel's 0.194) and it does **not** have a higher `r` — the
projection removes the radial noise component, so the same `h` buys less drift.
So `r` stays a **diagnosis** of prior-domination rather than a lever (pushing it
via `β` or `h` is worse at any radius), while the useful lever is the *radial*
constraint, which is a statement about the target, not the schedule.  Even the
best arm's NLL (2.110) is above the uniform predictor's 2.0794: calibration
improves, information does not appear.

**(c) The filtered chain, the cold target, and a radial column that misled.**
`pcnl/examples/mh_cold.rs` was written to answer "is the cold BMA the target's
predictive or the unadjusted bias?".  Its first answer — *no row samples
`π_λ`, every cell sits at `‖θ‖ ≈ 1.1–1.3e3` against the 45.2 shell* — was wrong,
for two independent reasons that are now both fixed.

*Ratio.*  The acceptance used `exp(λ(L(θ)−L(θ′)))` on the **mean** loss while
the drift is `∇` of the **summed** loss `N·L`, so the likelihood was
underweighted by `N = 512`; and `λN` alone is still not exact with a drift,
because pCN's prior cancellation needs the drift-free mean `√(1−β²)θ`.
`MhOptions::exact` now adds the prior ratio and the reverse-proposal ratio
`(‖θ′−m(θ)‖² − ‖θ−m(θ′)‖²)/2β²σ²`, at one extra full-batch backward pass per
step; `exact_filter_is_a_pcn_identity_at_zero_temperature` pins the identity.

*Radius.*  `pcnl/examples/posterior_radius.rs` scans
`Φ_λ(r) = r²/2σ² + λNL` along radial rays: the cosine likelihood is **exactly
radially flat** (`L = 3.48890` at `r = 10`, `3.48893` at `r = 4000` in the `θ_A`
direction; `5.16489/5.16490` for a Haar direction), pinned by
`numerics.rs::cosine_likelihood_is_flat_along_a_radial_ray`.  So `π_λ`'s radial
marginal is the prior's shell at every `λ` — and the observed `1.3e3` was
`‖θ_A + σξ‖`: `‖θ_A‖ = 1338`, 30× the shell, with `1/β² ≈ 1111` steps of
contraction against a 400-step run.  The `max` column was the start.  `grep`-able
confirmation: `PCNL_MH_DEBUG=1` prints the decomposition of every accepted
outward move, and in those runs there are none.

Placing the chain **on** the shell (`PCNL_START=shell`; a pure rescale, so no
prediction changes) instead, 400 steps, 2 seeds:

| `λ` | ratio | accept | BMA acc | BMA NLL | ECE | final |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pCNL | 85.8–90.0 % | 11.7 / 12.9 % | 2.3419–2.3525 | 0.2885–0.3023 | 60.1–60.4 |
| 1 | exact MH | 99.5–100 % | 11.7 / 12.9 % | 2.3668–2.3799 | 0.3136–0.3201 | 59.6–60.1 |
| 16 | pCNL | 33.0–55.8 % | 12.9 / 21.5 % | 1.9842–2.1063 | 0.0454–0.0573 | 61.4–63.0 |
| 16 | exact MH `β = 0.03` | **0 %** | 11.7 / 12.9 % | 4.9010–4.9950 | 0.8233–0.8428 | 64.1–64.4 (start) |
| 16 | exact MH `β = 0.01` | 85.5–89.0 % | 11.7 / 12.9 % | **2.1778–2.2386** | **0.1690–0.2048** | 63.6–63.9 |

Uniform NLL is `ln 8 = 2.0794`; the 0 % row's 4.9 is `θ_A`'s own predictive
(the chain never moved).

**The 10 000-step ladder, and what it changes.**  The 400-step rows above are
still relaxing (`1/β² = 1.1e3` steps at `β = 0.03`, `1e4` at `β = 0.01`), so
`scripts/pcnl-cold-10k` reruns the four usable cells with 10 000 steps, burn
5 000, thin 100 (50 kept draws, one seed, shell start):

| `λ` | ratio | accept | BMA acc | BMA NLL | ECE | final |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pCNL | 90.5 % | 12.9 % | 2.1568 | 0.1339 | 49.1 |
| 1 | exact MH | 99.9 % | 12.9 % | 2.1899 | 0.1810 | 48.4 |
| 16 | pCNL | 19.5 % | **34.8 %** | **1.4783** | 0.0891 | 59.0 |
| 16 | exact MH `β = 0.01` | 99.2 % | 12.9 % | **2.0993** | **0.0614** | 61.1 |

1. **The cold result replicates under an exact filter.**  `2.0993 / 0.0614`
   against §6.5.5's unadjusted `2.1030 / 0.0670` — the cold ladder's calibration
   was the target's, not the filter's bias.  My 400-step reading of "a third of
   the size" was under-mixing, and the 10k run is what settles it.
2. **At `λ = 1` the two ratios agree** (2.157/0.134 against 2.190/0.181): where
   the target is flat enough for pCNL's drift to be nearly right, filtering it
   changes little.  The filter matters where the target is sharp.
3. **Long-horizon pCNL at `λ = 16` departs upward**: 19.5 % acceptance,
   34.8 % accuracy, 1.4783 NLL — *above* chance, which no exact draw reaches
   (all at 12.9 %).  That is the unadjusted bias, and it shows up as
   information the posterior does not have, not as miscalibration.
4. **The radius cannot contaminate these columns.**  The exact rows end at
   48–61 against 45.2 and are still contracting (1e4 steps at `β = 0.01`, and
   `1e4` is the whole run); but the cosine likelihood is exactly
   scale-invariant, so every rescaled point gives the *same* predictive.  The
   radial coordinate affects the acceptance (hence mixing) and nothing in the
   accuracy/NLL/ECE columns.
5. **Still nothing beats the uniform predictor**, narrowly: the best row,
   2.0993, is 0.02 nats above `ln 8 = 2.0794`.  Cold `λ` buys calibration, not
   information.

**That corrects §6.5.3's wording twice.**  Its "exact full-batch
likelihood-ratio filter" rows showed the start artefact (radial final 95–134
against 45.2), and their filter was the mean-loss one, so "the overconfidence is
not the target's" narrows to *a pCNL chain at a usable proposal scale is less
overconfident than the unadjusted one*; which of the two is closer to `π` is
answered in the shell-start table, and it is the exact chain.
`MhOptions`'s doc comment, `mh_cold`'s header, the `comparators.rs` cap test and
`examples/mh_standard_target.rs` now say pCNL instead of "exact"; the crate's
numbers are unchanged.

`scripts/pcnl-cold-10k` is the run that produced the 10 000-step table above;
`STEPS=50000` extends it, which is what the still-contracting `β = 0.01` radius
would need.

### What the review did **not** change

Nothing in the Lean development or the readout module is touched: those
are statements at fixed `β` about the kernel (`timepiece/pcnl.md`
already says so in §3's scope note).  The architecture results are
untouched as *measurements* — the numbers in §6.4/§6.5.1(b) stand — but
their scope narrowed on 2026-11-04: the standard-stack comparison is a
step-scale comparison (§6.5.2 above), so what it supports is that the
normalization supplies the step scale, not that an unnormalized stack
cannot be sampled. The load-bearing open item is unchanged and unrelated to the
schedule: **an ensemble that has not left its mode**. The write-up's
§6.5.1(c) measures it — seven seeds, `R̂`(loss) ≈ 1 marks chains that
found the *same* (usually poor) mode, and BMA rises to 48.8–54.7 % only
once 8 chains or 30 000 steps let the ensemble split. §6.5.5 now gives the
mechanism: `r = βσ‖g‖/(2√P) ≈ 0.05` at `β₀ = 0.3`, so the direction is
prior-dominated and neither a ramp nor a temperature can reach the
informative set — what has to change is the step distribution, not the
schedule.

## Reproduce / verify

```bash
cd unfer
cargo test -p pcnl --locked                              # 85 tests (48 unit + 24 numerical
                                                         #  + 8 formal claims + 5 comparators)
# The report quoted above used kappa = 0.55 (the pre-2026-11-04 default):
cargo run --release -p pcnl -- --kappa 0.55 --report-file out.txt --save-dir /tmp/traj
# The corrected default is kappa = 0.5; the schedule's budget is in the report.
cargo run --release -p pcnl -- --report-file out.txt
cargo run -p pcnl --example schedule_mobility            # the mobility/floor table
cargo run --release -p pcnl --example step_scale_control  # §6.5.2, reduced config
cargo run --release -p pcnl --example step_scale_control -- --reference  # the table
cargo run --release -p pcnl --example sphere_volume       # §6.4(ii)'s Haar volume
cargo run --release -p pcnl --example mh_standard_target  # §6.5.3, the exact target
cargo run --release -p pcnl --example temperature_sweep   # §6.5.4(a), λ ladder
cargo run --release -p pcnl --example overdispersion      # §6.5.4(b), per-draw vs BMA
cargo run --release -p pcnl --example annealed_start      # §6.5.5, the ramp, ~12 min
cargo run --release -p pcnl --example cold_ladder -- 1000 --lambdas 1,16  # §6.5.6(a), 3 seeds
cargo run --release -p pcnl --example step_distribution   # §6.5.6(b), pCN vs ULA
cargo run --release -p pcnl --example mh_cold             # §6.5.6(c), the filter's bias
cargo run --release -p pcnl --example posterior_radius    # §6.5.6(c), the target's radial profile
./scripts/pcnl-cold-10k                                   # §6.5.6(c), the 10k-step table
cargo run --release -p pcnl -- quick --no-mh --lambda 16  # §6.5.5's cold ladder
cargo run --release -p pcnl -- --steps 1000 --chains 2 --no-mh --lambda 4  # published
cargo run --release -p pcnl --example dump_trajectory -- /tmp/traj/cosine_nGPT-lite_chain0.safetensors
```

Gates: `cargo fmt -p pcnl -- --check` and
`cargo clippy -p pcnl --all-targets -- -D warnings` are clean.
