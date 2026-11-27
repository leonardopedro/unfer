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
| **cosine / nGPT-lite** (current model) | stable, 10 000 unadjusted steps, no MH | mean 45.3, final 45.7, max 101.1 | **11.7 % / 2.1863 / 0.1798** (K=2000) | 32.0 % (Adam 49.2 %) | 194 | 1.004 |
| cosine / nGPT-lite (2026-10-20 archive, **stale**) | same | mean 45.5, final 45.7, max 61.1 | 54.7 % / 1.0686 / 0.1589 (K=2000) | 14.1 % | 18 | 1.115 |
| **standard softmax** | **EXPLODED at step 3** (non-finite gradient) | max 6.9e18 | n/a (chain died) | 10.9 % | — | — |

Chance level is 12.5 % (8 classes). The plan's Phase-0 hypothesis is
**confirmed for this config, measured**: the unadjusted update survives
only under the normalized geometry, and its radial claim holds (‖θ‖ stayed
at σ√P). The *architecture* result stands; the 2026-10-20 row's 54.7 %
BMA does **not** — see "The 2026-10-20 report is stale" below.

> **Read this before quoting any BMA below (2026-11-06).**  The sphere target
> `π ∝ exp(−N·L)` *against Haar measure* is prior-dominated at `P = 16 672`:
> the cone that carries Task B holds `10^{−55 000}` of the sphere's measure, so
> `π`'s own predictive is the base rate (11.72 %, NLL 5.04), and every
> 46–51 % BMA in this document is `π` **conditioned on the cone**.  Both are
> legitimate; they are not interchangeable, and the NLL column tells them
> apart.  Measured with no chain: `basin_posterior --prior-is 2048`; see
> "π's predictive is the base rate" below.

## The 2026-10-20 report is stale (2026-11-05)

Rerunning the documented invocation under the current model
(`cargo run --release -p pcnl -- --kappa 0.55 --report-file …`, same seed,
same flags) does **not** print the block below:

```text
[cosine/nGPT-lite] P=16672
  pre-train A: loss 5.0673 -> 1.1812, acc_A 43.8%
  B baselines: before 4.3% (NLL 4.1523), SGD-on-B 32.0% (NLL 1.8045), Adam-on-B 49.2% (NLL 1.3136)
  SG-pCNL: beta 0.3000 -> 0.0802 over 10000 steps | max scaled|g| 1002.897 max batch loss 4.918 | radial final 45.7 max 101.1 mean 45.3 vs sigma*sqrt(P) 45.2 | ESS(loss) 194 ESS(coord)med 15 | Rhat(loss) 1.004 Rhat coord med 1.086 max 1.537 | BMA 11.7% NLL 2.1863 ECE 0.1798 (K=2000) last-state 11.7% | 2577.9s
  SGLD (baseline): radial final 45.3 max 101.3 vs sigma*sqrt(P) 45.2 | ESS(loss) 201 | Rhat(loss) 1.004 | BMA 11.7% NLL 2.1839 ECE 0.1775 (K=2000) | cross-Rhat 1.006 | 506.7s
  MH-pCNL (filtered): accept 21.3% | radial final 45.6 vs sigma*sqrt(P) 45.2 | ESS(loss) 66 | Rhat(loss) 1.016 | BMA 11.7% NLL 2.1433 ECE 0.1227 (K=2000) | kept 500 per chain | cross-Rhat 2.572 | 1221.7s
```

`test/unfer/ngpt.md` §6.5.1(b) already says so — "The pre-fix headline —
BMA 54.7 %, last-state 41.3 % — was an artifact of those unnormalized
gradients ... superseded by the 11.7 % measured here on the corrected model" —
and every post-fix number in this document agrees with it. What was stale was
this file's front page, not the study.

**A/B on the commit itself, same seed, same flags.**  Not inferred from dates:
`pcnl` was built standalone at `7efbd97` (the parent of `2fc095f`) and run with
`--kappa 0.55`, then compared with HEAD:

| | `7efbd97` (pre-fix) | HEAD (post-fix) |
|---|---|---|
| pre-train A | 4.8819 → **2.0743**, acc_A 12.9 % | 5.0673 → **1.1812**, acc_A 43.8 % |
| `θ_A` on B | 9.4 % (2.0939) | 4.3 % (4.1523) |
| SGD-on-B (same `θ_A`) | 14.1 % (2.0953) | 32.0 % (1.8045) |
| max scaled ‖g‖ | 8 349.8 | 1 002.9 |
| BMA (K=2000) | **57.0 % / 1.0668** | **11.7 % / 2.1863** |
| last-state / ESS(loss) / R̂(loss) | 41.7 % / 25 / 1.066 | 11.7 % / 194 / 1.004 |

The pre-fix build reproduces the archived *regime* — an informative BMA from a
chance-level `θ_A`, the same 12.9 % pre-train and 14.1 % SGD-on-B — but not its
last digit (57.0 % vs the quoted 54.7 %, 1.0668 vs 1.0686), so the quoted block
was not produced by this exact commit either; it is a pre-fix build of the same
regime.  The change is `2fc095f` (2026-10-22), which joined **V and O to the
Frobenius normalization** in the cosine forward pass — forced by
`tests/formal_claims.rs`, because the pre-fix pass was not scale-invariant and
M2's ⟨θ,∇Φ⟩ = 0 failed for the implementation.  The tell is the pre-train
column: the fix made Task A learnable (43.8 % vs 12.9 %) and cut the gradient's
scale ~8×, so the same `β_t` now moves the chain far less relative to the
likelihood.

What survives and what does not:

* **Phase 0's contrast survives, more sharply.** The standard stack still dies
  at unadjusted step 3 with a non-finite gradient (scaled ‖g‖ 1.2e21, radial
  6.5e18); the cosine variant still runs 10 000 steps at ‖θ‖ = 45.3–45.7
  against σ√P = 45.2. The radial claim is unchanged.
* **The BMA headline does not — at either schedule exponent.** The default
  `κ = 0.5` run gives 11.7 % / 2.1865 with ESS(loss) 226 and R̂(loss) 1.004,
  i.e. the same result with an unbounded mobility budget, so this is not the
  `Σβ² < ∞` freeze of §4.3 either. 11.7 % / 2.1863 is the uniform predictor
  (ln 8 = 2.0794), last-state the same 11.7 % — no averaging gain at all. The
  archived 54.7 % / 1.0686 and its "averaging is worth ~13 points" reading are
  artifacts of the pre-fix model and are marked as such wherever quoted.
* **Task B is learnable, and the chain does not find it.** From the same θ_A,
  the momentum-SGD baseline reaches 32.0 % (1.8045) and Adam 49.2 % (1.3136),
  while every sampling arm — unadjusted, SGLD, and the exact MH filter — sits
  at 11.7 %. That is §6.5's reachability result, at the published budget and
  on the current model, and it is a *stronger* statement than the archived
  report could make.
* **Mixing is now good** (R̂(loss) 1.004, ESS(loss) 194 of 500): the two
  families that used to disagree (R̂ 1.115) agree, and they agree on the
  prior's predictive. The exact filter's cross-R̂ of 2.572 against the
  unadjusted chain is the one non-converged diagnostic — expected, since one
  is `π` and the other is a filtered ascent (§6.5.3).

Everything from §6.5.1 onward in this document and in `test/unfer/ngpt.md`
was measured **after** the fix and is unaffected.

## Report (verbatim, 2026-10-20 — pre-normalization-fix, see above)

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

*Those wall times are pre-fix and pre-`run_mh`-default.* The same command
today takes 2 577.9 s: the V/O normalization did not change the sampling
cost, but the report now also runs the SGLD baseline (506.7 s) and the
exact MH filter (1 221.7 s, one extra full-batch backward per step).
The 1.44× `rayon` speedup is a fact about the chain loop, not about the
total — it is not comparable across those two defaults.

## Reading against the plan

**Phase 0 — confirmed, as measured.** The standard dot-product variant
died at unadjusted step 3 with a non-finite gradient (scaled ‖g‖ 1.2e21,
radial 6.5e18 against σ√P = 45.2); its deterministic Task-A pre-train had
already blown up to loss 3835. The cosine variant ran all 10 000
unadjusted mini-batch steps with no Metropolis filter anywhere, radial
mean 45.3 / final 45.7 vs σ√P = 45.2 (max 101.1 is a transient, and it is
the same 2.2 × shell §6.5.5's `λ = 1` ladder reports), exactly the §3B
claim. This is one config, one seed — not a proof.

**Phase 3 — BMA does not beat its baselines on the current model.** The
archived run's 54.7 % / 1.0686 / last-state 41.3 % is a pre-fix artifact
(see the staleness section above). Today's published-budget run gives
**11.7 % / 2.1863 / last-state 11.7 %** — the uniform predictor, with no
averaging gain — against a momentum-SGD baseline of 32.0 % / 1.8045 and
Adam-on-B at 49.2 % / 1.3136 from the *same* θ_A. So the information is
reachable by supervised training and not by any of the three sampling
arms. That is not a failure of BMA as an estimator; it is §6.5's finding
that the sampler never reaches the informative set, and averaging draws
that all sit in the prior's region cannot manufacture information.

*Two caveats on that comparison, both still true:*
- The SGD baseline uses the pre-train learning rate (0.6) for 1 000
  steps — not a tuned optimiser. The tuned comparison is the Adam arm
  (49.2 %), which is a stronger baseline than the plan's.
- The Task-A pre-train now trains properly (loss 1.1812, acc_A 43.8 %),
  so θ_A is informative about Task A and only 4.3 % on Task B: the chain
  starts from a *good* pre-trained point, which makes its failure to
  adapt harder to attribute to a bad start.

**Phase 4 — diagnostics are reported as measured, current model.**
- ESS(loss) 194 of 500 kept draws/chain, ESS(coord) median 15 — the loss
  chain decorrelates, the coordinates still do not.
- R̂(loss) 1.004, coordinate R̂ median 1.086 / **max 1.537**. The chains
  agree on the *value* of the potential while their coordinates differ —
  the signature of many directions with the same, near-uniform
  likelihood (§6.4(ii)), not of non-convergence.
- ECE 0.1798 (10-bin, BMA predictive) against NLL 2.1863 ≈ ln 8: a
  near-uniform predictor cannot be badly calibrated; the ECE is the
  residual of a slightly excess-confidence at the shell.
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
quantified.

**Does ten times the budget split it?** That reading is a 1 000-step
measurement, and the archived headline (§ above) claimed 54.7 % / 1.0686 at
10 000 steps — so the gap was re-measured at the published horizon.
`overdispersion` now rebuilds the report's construction exactly (data A on
`seed`, data B on `seed+1`, weights on `seed+2`, `θ_A + σξ` start, per-chain
streams at `seed+{200,400}+c`) and takes `--kappa`/`--thin`/`--pretrain`:

| steps | chains | κ | seeds | per-draw NLL | BMA NLL | gap |
|---:|---:|---:|---|---:|---:|---:|
| 1 000 | 4 | 0.55 | 20261020 | 2.2221 | 2.1950 | 0.0271 |
| 1 000 | 4 | 0.55 | 20261021 | 2.2173 | 2.1894 | 0.0278 |
| 10 000 | 4 | 0.55 | 20261020 | 2.2014 | 2.1861 | **0.0154** |
| 10 000 | 2 | 0.5 | 3 seeds | 2.2021–2.2064 | 2.1848–2.1896 | 0.0162–0.0173 |

Ten times the budget moves each draw by −0.02 nats and *shrinks* the gap by
40 %: the draws agree more, not less, and the ensemble's advantage over a
single draw falls to 0.015 nats against the uniform predictor's 2.0794. The
"has not left its mode" reading is therefore not a short-run artifact — it
gets stronger with budget, and the archived 54.7 % cannot be a slow-burn
version of the same chain.  `pcnl/tests/comparators.rs::mh_drift_cap_changes_only_the_proposal`
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

### Handed the solution, the kernel still loses it — the target is the constraint (2026-11-05)

Every sampling arm reports the prior's predictive at the published budget while
Adam-on-B reaches 49.2 % from the same `θ_A`.  That leaves **search** and
**retention** inseparable, so `pcnl/examples/adam_start.rs` starts the same
unadjusted kernel at Adam's Task-B solution (rescaled to `σ√P` — a pure rescale,
and the cosine forward pass is scale-invariant, so no prediction changes) and
follows it, under two priors: the published `N(0, σ²I)`, and `centered_at` the
solution (`N(θ_Adam, σ²I)`, §6.4.2's centred adaptation prior).

The first fact is where the solution lives:

| point | acc (B) | NLL | `‖θ‖` |
|---|---:|---:|---:|
| `θ_A` | 4.30 % | 4.1523 | 94.0 (2.1 × shell) |
| `θ_Adam` (Task B) | **49.22 %** | 1.3136 | **1272.6 (28.2 × shell)** |

> **Both rows are stale — see the 2026-11-07 correction.**  They are the
> pre-fix 2026-10-20 baselines, not a measurement of the current model; the
> current values are `θ_A 13.67 % / 4.2124 / 124.9` and
> `θ_Adam 55.08 % / 1.1367 / 1620.1`.

and the second is what the kernel does with it (2 chains at `β₀ = 0.3`,
10 000 steps; 1 chain at `β₀ = 0.03`, 2 000 steps):

| β₀ | prior | `t = 10` | `t = 100` | kept BMA | BMA NLL | ECE |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | origin | 11.72 % | 11.72 % | 11.72 % | 2.1857 | 0.1802 |
| 0.3 | anchored | 11.72 % | 11.72 % | 11.72 % | 2.1819 | 0.1675 |
| 0.03 | origin | 12.11 % | 17.58 % | 12.89 % | 2.0979 | 0.1791 |
| 0.03 | anchored | 12.11 % | 17.58 % | **17.19 %** | **1.9500** | **0.1487** |

Three readings:

1. **Retention fails before search does.** At the published proposal one step of
   prior noise is `βσ√P = 13.6` in a 45-norm shell, and a sharp 49 % fine-tune
   is gone by step 10 — *with the prior centred on it*. Nothing about finding
   the solution is being tested there; the step itself is larger than the
   solution's basin.
2. **A smaller step keeps a neighbourhood, and the anchored BMA is the first
   sampling result in this study below the uniform predictor**: 1.9500 nats
   against ln 8 = 2.0794, 0.15 nats better than the identical arm with the
   origin prior. So the prior's *centre* is worth something once the step can
   preserve structure — a retention version of §6.4.2's centring finding.
3. **Why nothing can reach 49 % at the published target: the prior's radius.**
   `π ∝ N(0, σ²I)·exp(−λNL)` pays `r²/2σ² ≈ 6.6 × 10⁶` nats at the solution's
   norm, so `π` assigns it vanishing mass and *any* sampler of `π` reports the
   shell's Haar-typical direction, which is chance. This is the same arithmetic
   as §6.5.6(c)'s radial profile, used in the other direction: there it showed
   the likelihood cannot move the chain off the shell, here it shows the shell
   cannot hold the solution.

The consequence is a target change, not a schedule or step-size search.  §3.3's
scale invariance says predictions depend only on the **direction** of `θ`, so
the target that contains the solution is a uniform (Haar) prior on the sphere
at fixed radius — where the solution's direction has full mass — rather than a
Gaussian in `θ` whose shell is `σ√P = 45` while the fine-tune lives at 1 273.
That is the next experiment this study implies and it is not yet run.

### The repaired target: a Haar prior on directions, and a step that still cannot move (2026-11-05)

§6.5.7's diagnosis was that the isotropic Gaussian excludes the solution, and
the repair it implies is a uniform (Haar) prior on the **direction** of `θ` at
fixed radius — legitimate because the cosine forward pass is scale-invariant,
so predictions depend only on the direction.  `pcnl/src/sphere.rs` implements
that target's kernel: exponential-map pCN on `S^{P−1}` with an **exact**
Metropolis filter.  The proposal's density in tangent coordinates is
`N(v; m(u), τ²)·(sin|v|/|v|)^{P−2}`; the reverse move runs back along the same
geodesic, so `|w| = |v|` and the Jacobian is the same factor on both sides and
cancels — the ratio is Gaussian-times-likelihood with no numerical integration.
`sphere.rs`'s tests pin it: `λ = 0` is Haar, a von Mises target on `S¹` gives
`E[u₁] = I₁(κ)/I₀(κ)` at every step size, and a von Mises–**Fisher** target on
`S²` gives `E[u₁] = coth κ − 1/κ` under an anisotropic metric (added
2026-11-05 with the restricted-metric fix below — the 2D moment was blind to
that bug).  Three real bugs were caught by moments and by nothing else: an
unscaled `τ`, a negated reverse tangent, and the restricted-inverse term.

`pcnl/examples/sphere_chain.rs`, **600 steps** (burn 300, thin 20, one chain,
`λ = 1`, `ρ = σ√P = 45.19`, seed 20261020, K = 15 kept draws).  Re-run at this
budget on 2026-11-05 after the preconditioner bugs below were fixed, with the
`β₀ = 0.003` rung added — without it the grid contained no cell that could
move, and the earlier 1 200-step table's `β₀ = 0.01` row (0.6 %) was a
average of the chain's first accepting steps and its later frozen ones:

| where / cell | acc (B) | NLL | `‖θ‖` |
|---|---|---:|---:|
| `θ_Adam` (ambient) | 49.22 % | 1.3136 | 1272.6 |
| `ρ·u_Adam` (same direction, on the shell) | **49.22 %** | **1.3136** | 45.19 |
| `ρ·u_Haar` | 11.72 % | 4.6667 | 45.19 |

| β₀ | start | accept | BMA acc | BMA NLL |
|---:|---|---:|---:|---:|
| 0.3 | Adam | **0.0 %** | 49.22 % | 1.3136 |
| 0.3 | Haar | 32.3 % | 11.72 % | 2.1879 |
| 0.03 | Adam | **0.0 %** | 49.22 % | 1.3136 |
| 0.03 | Haar | 100 % | 11.72 % | 2.2069 |
| 0.01 | Adam | 0.2 % | 48.05 % | 1.3288 |
| 0.01 | Haar | 99.8 % | 11.72 % | 3.7238 |
| **0.003** | Adam | 91.8 % | 37.50 % | 1.4527 |
| 0.003 | Haar | 99.8 % | 11.72 % | 4.5922 |

The metric arm has its own table below (*The metric ladder, re-measured*) — its
original columns came from an inexact ratio (found 2026-11-05, see *The
restricted-metric bug*) and were re-run on 2026-11-06.  The isotropic columns
are unaffected — with `d ≡ 1` the missing term is exactly zero — so everything
above stands.

Four results, in order of how much they change:

1. **The prior diagnosis is confirmed at the target level.** `ρ·u_Adam` scores
   *identically* to `θ_Adam` (49.22 % / 1.3136), so a Haar prior on directions
   contains the solution with full mass.  §6.5.7 was right that the Gaussian's
   radius, not the kernel, was excluding it.
2. **Repairing the prior is not enough by itself: at the published scale the
   repaired target cannot be sampled.**  From the solution's direction the exact
   filter accepts **nothing** at `β₀ = 0.3` and `β₀ = 0.03`, isotropically or
   with the metric.  Those 49.22 % rows are a *frozen start*, not an ensemble,
   and the acceptance column is the only thing that distinguishes them.  The
   grid had no cell that could move until `β₀ = 0.003` was added — the boundary
   is `λN‖g‖σβ ≲ 2`, not a fact about `β`.
3. **With a curvature metric the chain samples near the solution** — corrected
   2026-11-05 after the restricted-metric bug, re-measured on 4 chains
   2026-11-06: at `β₀ = 0.003`, 400 steps, the metric arm accepts **87.8 %** and
   its BMA scores **49.02 % ± 3.04** (`R̂(loss) = 3.00`, so not converged),
   against 49.22 % at the start, while the isotropic arm at `β₀ = 0.003` accepts
   92 % and diffuses to 37.50 % over 600 steps.  The mode it sits on is 51.17 %
   / NLL 1.2748, 7.7 × 10⁻⁴ rad away (`examples/sphere_mode.rs`), so the chain
   reaches the mode's accuracy but does not settle there.  The metric *does*
   widen the accepting range — `0.2 % → 53.5 %` at `β₀ = 0.01`, see the ladder
   below; the pre-fix `22.2 %` was measured with the biased ratio but was not
   an artefact of it.
4. **Acceptance is a property of position, not of the kernel** — and for the
   metric that cuts both ways.  From a Haar direction the same chain accepts
   32 % / 100 % / 99.8 % isotropically, in the region where the likelihood is
   flat, at chance, with NLL 2.19–4.59 (the draws are *confidently* wrong,
   §6.4(ii)'s Haar-typical predictive).  So §6.5.6's 90–100 % acceptance in the
   ambient kernel was a **symptom of sampling the flat region**, not evidence
   that the sampler worked.  And the metric *lowers* acceptance there (32.3 % →
   0.2 % at `β₀ = 0.3`) because the flat region's estimated curvature inflates
   the drift: a metric estimated in one region does not transfer to another.

The two halves of the story are complementary and both are measured: the
isotropic Gaussian prior excludes the solution, and once that is repaired a
fixed *isotropic* pCN step still cannot both move and accept in 16 672
dimensions.  What remains is the gap quantified below — `√P` between accepting
and informative — which no diagonal metric closes.

### Why the preconditioned arm did nothing: two bugs, and the √P gap behind them (2026-11-05)

The next lever the section above named was a curvature-scaled proposal.  The
first two runs of it were **identical to the isotropic ones** — every `adam`
cell at 0.0 % acceptance, to the digit — and that is a bug signature, not a
result.  Two defects in `pcnl/src/sphere.rs`, both real:

1. **`with_preconditioner()` set `precond = None`.**  The method whose name and
docstring say "turn on diagonal preconditioning" disabled the metric.  The
*test* that was supposed to cover it (`constant_metric_is_the_isotropic_step`)
still passed, because with `d ≡ 1` an inert metric and an identity metric are
the same step — a green test guarding nothing.  The flag is now separate from
the installed metric (`precond_enabled` vs `precond`).
2. **The curvature was estimated from `E[(Δg)²]` between *accepted* states.**
   From the solution's direction nothing is accepted, so the state never moves,
   `Δg ≡ 0`, `grad_sq ≡ 0`, and the metric collapsed to the identity — inert on
   exactly the cells it exists to rescue.  It also had the wrong units
   (`Δg ≈ H·Δu` needs dividing by the displacement).  The estimator is now the
   diagonal **secant** `H_ii ≈ E|Δg_i| / E|v_i|` measured along the *proposal*,
   which the step evaluates whether or not it is accepted.  Regression test:
   `warmup_learns_the_metric_even_when_nothing_is_accepted` (asserts
   `accepted == 0` **and** `d₁ > 7·d₀`); it fails on the old code and passes now.

A third discrepancy was mine, not the kernel's: an early probe re-implemented
the acceptance ratio inline, omitted the reverse drift `m₁`, and reported 92 %
acceptance where the kernel accepts 0/15.  The probe now calls the kernel.  The
lesson is in the file's header rather than its numbers: the ratio is subtle
enough that a second implementation is a second bug, not a cross-check.

With the corrected metric the question becomes quantitative, and
`pcnl/examples/curvature_probe.rs` answers it at `u_Adam` (P = 16 672,
N = 512, λ = 1, `‖∇L‖ = 2.881`):

| β₀ | `‖e‖ = τ√P` | `‖m‖` | `m/e` | `λN‖g‖σβ` | `r` | kernel accept (10 steps) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.300 | 0.180 | 0.60 | 77.4 | 0.600 | 0/10 |
| 0.1 | 0.100 | 0.020 | 0.20 | 25.8 | 0.200 | 0/10 |
| 0.03 | 0.030 | 0.0018 | 0.06 | 7.74 | 0.060 | 0/10 |
| 0.01 | 0.010 | 0.0002 | 0.02 | 2.58 | 0.020 | 0/10 |
| **0.003** | 0.003 | 2.0e-5 | 0.01 | **0.77** | 0.006 | **8/10** |
| 0.001 | 0.001 | 6.6e-6 | 0.00 | 0.26 | 0.002 | 10/10 |

Three facts fall out.

1. **The filter's acceptance is governed by `λN‖g‖σβ ≲ 2`** — the typical pCN
exponent, not `β`.  The grid in the six cells (`β₀ ∈ {0.3, 0.03, 0.01}`) never
contained a cell where that holds, which is why every `adam` cell read 0.0 %,
and why the `haar` cells read 99–100 % — there the same product is ~100× smaller
because `‖g‖` is small.  Acceptance is a *position* diagnostic; that was already
§6.5.6's conclusion, and this gives its number.
2. **Informativeness and acceptance do not overlap, and the gap is `√P`.**  The
   same product is `2√P·r`, so `λN‖g‖σβ ≲ 2` (accept) and `r ≳ 1` (an
   informative drift) can only both hold if `√P ≲ 1`.  At P = 16 672 the
   accepting rungs have `r ≈ 0.002–0.006`: a chain that accepts, and is a random
   walk whose moves change the loss by ±0.5 nats, i.e. samples the *neighbourhood*
   of the start, not `π`.
3. **A diagonal metric cannot bridge the gap, though it does move the frontier.**
   Optimal scaling needs `d_i ∝ 1/H_ii`, and the most a mean-1 `d` can buy in
   `τ²` is the heterogeneity of `H` — measured as `E|Δg_i/v_i|`
   `mean/median = 31.6×` (median 0.186, p99 96, max 1 116).  That is ~5.6× in
   `β`, while bridging `β = 0.003 → 0.3` needs `√P = 129×` in `β`, or
   1.7 × 10⁴× in `τ²`.  The prediction is "up to ~10× better, not 100×", and
   `β₀ = 0.3` stays at 0.0 % under the metric too.  The measured widening is
   **`0.2 % → 53.5 %` at `β₀ = 0.01`** (re-run with the exact ratio
   2026-11-06, *The metric ladder* below): the frontier moves to 0.01, ≈ 3.3×
   in β, i.e. inside the 5.6× ceiling and 39× short of 0.3.  Where it does
   accept, the metric is not cosmetic: at `β₀ = 0.003` the corrected chain holds
   the neighbourhood at 49.02 % ± 3.04 where the isotropic one diffuses to
   37.50 % over the same number of steps' worth of acceptance.

The trust radius is the same fact in the loss's own units: along `−ĝ`, the mean
NLL is flat at 1.2454 by `s = 1e-4` rad and *rises* to 4.33 at `s = 0.3` — the
published proposal's move is 3 000× the angle over which the local quadratic
model survives.

What this leaves for the sphere route is not a better *diagonal* metric: it is
a proposal that moves in the low-dimensional subspace the likelihood actually
constrains (the fine-tune's Jacobian is rank ≪ P, so a low-rank or
Fisher/subspace metric is the only kind that can pay the `√P`), a different
target, or accepting that the Bayesian object here is a *neighbourhood*
posterior — which is what `adam_start`'s anchored arm at `β₀ = 0.03`
(BMA NLL 1.9500) and the metric arm's 49.02 % ± 3.04 (4 chains; 49.80 % ± 1.93
on two) both report.

### The restricted-metric bug: the preconditioned ratio was not exact (2026-11-05)

The preconditioned MH ratio used
`−Σ x_i²/(2τ²d_i)` as the proposal's Gaussian term.  That is the exponent of
`N(0, τ²D)`, but the proposal's actual covariance is `τ²·(P D P)` **restricted
to `T_u`**, and the inverse of that restriction is not `D⁻¹`:

```text
xᵀ(P D P|_T)⁻¹x = xᵀD⁻¹x − (xᵀD⁻¹u)² / (uᵀD⁻¹u),    x ∈ T_u.
```

Verified against a dense `QᵀDQ` solve (`restricted_metric_inverse_matches_the_dense_one`):
at P = 4 with `d` spanning 16×, the two sides differ by up to 50 %, and the
missing term is O(1) in the log ratio at every step.  So the preconditioned
chain was sampling a *different law*, at a rate whose error the acceptance
column did not show.

**Why the old tests could not catch it.**  With `d ≡ 1` the term vanishes
exactly (that is why every isotropic number in this document is unaffected).
The anisotropic test ran on `S¹` from `u = (1,0)`, where `xᵀD⁻¹u ∝ u₁u₂ ≈ 0`:
a 2D test at an axis-aligned start is structurally blind to a term that is
first-order in the off-axis component.  The fix is the term above on both
sides, plus `anisotropic_metric_is_exact_on_s2`: a von Mises–Fisher target on
`S²` (`E[u₁] = coth κ − 1/κ`), anisotropic metric, start at `(0.6, 0.8, 0)`.
Reverting the fix makes that test read `E[u₁] = 0.377` against 0.537 — a 30 %
bias — which is the demonstration that the test is not vacuous.

**What it costs the results.**  The metric cell at `β₀ = 0.003` (adam start, 400
steps, 2 chains, K = 10):

| | accept | BMA acc | BMA NLL | `R̂(loss)` |
|---|---:|---:|---:|---:|
| pre-fix ratio (withdrawn) | 42.6 % | 50.20 % | 1.3137 | 3.08 |
| **fixed ratio** | **88.9 %** | **49.80 % ± 1.93** | 1.3147 | **4.85** |

So the biased ratio was rejecting roughly half of the moves the exact filter
accepts (42.6 % vs 88.9 %), and the accuracy level is the same to within its
spread — the *conclusion* ("the metric chain stays near the solution where the
isotropic one diffuses") survives, the *numbers* do not.  Neither run is
converged (`R̂` 3–5), so this is still a statement about where the chain sits.
Every metric-arm figure elsewhere in this document that predates this fix is
marked or withdrawn; the isotropic arm is untouched.

### The metric ladder, re-measured (2026-11-06)

`sphere_chain --precond --seeds 2` at 400 steps (burn 200, thin 20, K = 10,
two chains per cell), exact ratio throughout:

| β₀ | start | accept | BMA acc | BMA NLL | `R̂(loss)` |
|---:|---|---:|---:|---:|---:|
| 0.3 | Adam | 0.0 % | 49.22 % | 1.3136 | 1.000 (frozen) |
| 0.3 | Haar | 0.2 % | 11.72 % | 3.1410 | ∞ (frozen) |
| 0.03 | Adam | 2.8 % | 48.44 % | 1.3200 | 1.361 |
| 0.03 | Haar | 7.2 % | 11.72 % | 4.6420 | 3.220 |
| 0.01 | Adam | **53.5 %** | 48.44 % | 1.3174 | 2.058 |
| 0.01 | Haar | 55.2 % | 11.72 % | 4.6622 | 2.454 |
| **0.003** | Adam | **87.8 %** | 49.02 % ± 3.04 | 1.3140 | 2.998 |
| 0.003 | Haar | 83.2 % | 11.72 % | 4.6650 | 2.722 |

Three things change against the withdrawn pre-fix table, one does not.

1. **The metric does widen the accepting window, and by about the predicted
   amount.**  At `β₀ = 0.01` the isotropic chain accepts 0.2 % and the metric
   accepts **53.5 %**: the frontier moves `0.003 → 0.01`, ≈ 3.3× in β, against
   the 5.6× ceiling implied by the measured 31.6× curvature heterogeneity
   (`curvature_probe`).  The withdrawn `22.2 %` was the right order of
   magnitude, so the *mechanism* was never the bug.  `β₀ = 0.03` falls back to
   2.8 % and `β₀ = 0.3` stays at 0.0 %, so the `√P` gap survives: 30× in β is
   still unreachable with a diagonal metric.
2. **Acceptance is reach, not information.**  At the rungs the metric opens
   (`0.01`, `0.03`) the BMA is 48.44 % — *below* the frozen start's 49.22 %.
   The chain moves freely and its ensemble is not better than where it started;
   only `β₀ = 0.003`, where the proposal is small enough to stay in the
   neighbourhood, holds the start's level (49.02 % ± 3.04).  This is the same
   fact as the ambient target's high acceptance in the flat region (§6.5.6),
   one rung lower on the step-size ladder.
3. **The Haar cells do not improve at any rung**: 11.72 % / NLL 4.64–4.67, at
   83–99 % acceptance, i.e. chance and confidently wrong.  The same metric
   *lowers* acceptance from the Haar direction (0.2 % at `β₀ = 0.3`) because the
   flat region's estimated curvature inflates the drift — a metric estimated in
   one region does not transfer to another.
4. **Unchanged: nothing here is converged.**  `R̂(loss)` runs 1.4–3.2 across the
   moving cells, so these are statements about where the chains sit and how
   often they move, not posterior means.  The 10 000-step attempt at the one
   converged number is in the next section — and it does not converge either.

### The 10 000-step attempt: the neighbourhood is a transient (2026-11-06)

`sphere_chain --precond --beta 0.003 --start adam --seeds 2 --chains 2 --steps
10000` — 4 chains, burn 5 000, thin 20, K = 250 kept draws per chain, 1 h 40 m
on the two-core machine:

| accept | acc t=0 | t=10 | t=100 | BMA acc | BMA NLL | across-chain | `R̂(loss)` |
|---:|---:|---:|---:|---:|---:|---|---:|
| 86.1 % | 49.22 % | 52.73 % | 51.17 % | **46.39 %** | 1.3412 | ± 0.37 pts / ± 0.0116 nats | **2.894** |

Three things follow, and the first is a correction to reading 3 above.

1. **`R̂` did not fall below 1.05 — it barely moved** (2.998 at 400 steps → 2.894
   at 10 000, 25× the budget).  So the split-`R̂` is not a short-run artefact:
   the chains sit at different places on the sphere and 10 000 steps do not
   bring them together.  The per-draw loss is heavily autocorrelated (a random
   walk at τ = 2.3 × 10⁻⁵ rad per step), so the within-chain variance stays
   large; the diagnostic is saying exactly what it should — the draws are not
   exchangeable across chains at this budget.
2. **The across-chain spread collapses while the level degrades.**  ± 3.04 pts at
   400 steps → ± 0.37 pts at 10 000: the chains *agree* by 10 000 steps, and
   they agree on 46.39 % / 1.3412 — **worse than the frozen start's** 49.22 % /
   1.3136.  So the 400-step 49.02 % was retention of the start, not a property of
   the chain's long-run law.  The mechanism is the diffusion itself: RMS
   displacement √(10⁴)·2.3 × 10⁻⁵ = 2.3 × 10⁻³ rad, three times the
   7.7 × 10⁻⁴ rad separating `u_Adam` from the mode, so the chain walks out of
   the informative basin rather than settling in it.
3. **"The metric chain holds the solution's neighbourhood" is therefore a
   transient claim, and is not withdrawn only within that horizon.**  At 400–600
   steps it does stay where the isotropic chain diffuses to 37.50 %; over 10 000
   it does not, and the mode's own accuracy shows up early
   (`acc t=100 = 51.17 %`, exactly `sphere_mode`'s value) and is then lost.

What a converged number needs is not more steps at fixed β.  The step that
accepts at the solution (`β ≤ 0.003`) has angular length `β` and decays from
it, so acceptance and locality are the same constraint — the `√P` arithmetic
of `curvature_probe` seen from the sampler's side rather than the acceptance
column's.  **Correction (2026-11-06):** this paragraph first read the step as
`τ = 2.3 × 10⁻⁵ rad` and concluded that 10⁴ steps "necessarily explore
~2 × 10⁻³ rad".  `τ` is the *per-coordinate* scale; the angular step is
`τ√P = β = 3 × 10⁻³` rad at `t = 0`, decaying to `3 × 10⁻⁵` by `t = 10⁴` at
`κ = 0.5`, so the walk can span `Σβ_t ≈ 2β√T ≈ 0.6` rad.  The conclusion
survives; the number was 300× off, and the next section measures what that
cost.

### π's predictive is the base rate — the cone is a conditional quantity (2026-11-06)

**This section supersedes the "π's mass is inside a few milliradians" reading below**
and, with it, the claim that 48.3–49.0 % is the target's predictive.  The evidence
is a measurement that uses no chain and no anchor ball: `basin_posterior
--prior-is M` draws `M` **Haar** directions and evaluates each one.

Haar is the prior's own directional law.  The target is `π(u) ∝ exp(−N·L(u))`
against Haar measure, so the prior contributes *no* directional preference and
the likelihood is the only thing that can concentrate `π`.  What competes with
it is geometry: a cap of angular radius `d` holds a fraction `≈
(sin(d/2))^{P−1}` of the sphere, so the number of directions between a 1 mrad
cap and the equatorial shell is `10^{55 013}` at `P = 16 672`.  The likelihood
can only buy that back with `N·ΔL` (M = 2048):

| `d` range (rad) | n | train `L` | `N·ΔL` | log₁₀ volume (vs 1 mrad) | log₁₀ mass | acc | test NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0–1.6 | 2047 | 5.0505 | 1944 | 52 626 | 51 782 | **11.72 %** | **5.0413** |
| 1.6–π | 1 | 4.9913 | 1914 | 55 013 | 54 182 | 11.72 % | 5.0401 |

**Every Haar draw lands at `d ∈ [1.0, 1.6]` rad** — concentration of measure, not a
sampling artefact — and every one of them scores the base rate 11.72 % with NLL
5.04 (worse than the uniform predictor's 2.0794, because the model is
*confidently wrong* out there).  The two sides of the trade, in decades:

* volume from the 1 mrad cap to the shell: `10^{55 013}`,
* likelihood against that shell: `10^{−831}`,
* net: **`10^{54 000}` in favour of the shell.**

> Corrected 2026-11-07: these four lines use the cosine-cap proxy
> `(sin(d/2))^{P−1}`, which over-charges a cap by `(P−1)·ln 2 = 11 547` nats at
> this `P`.  The exact Haar cap law gives `10^{50 014}` for the 1 mrad cap — the
> same verdict, five thousand decades smaller.  (The correction does not rescue
> the cone: the gap to the `10^{−831}` likelihood is still `10^{49 000}`.)

The likelihood tilt is real — `N·(max L − min L) = 253` nats, and Haar
self-normalized IS has `ESS = 1.00` of 2048, i.e. the weights are degenerate
because `π` sits on the low-`L` tail of a set Haar cannot resolve.  But 253
nats cannot argue with 55 000 decades.  **The tempered posterior on directions
is prior-dominated: its predictive is the base rate (11.72 % / NLL 5.04), and
every 46–51 % number in this document is `π` *conditioned on the cone*.**

The accuracy column alone cannot see this, because everything far from the cone
sits at the base rate; the **NLL** separates the four regimes, and it is the
column to read from here on:

| quantity | acc | NLL | what it is |
|---|---:|---:|---|
| uniform predictor | 12.5 % | 2.0794 | calibrated, no information |
| ambient chain, 10 000 steps (§6.5.1) | 11.7 % | 2.1863 | collapsed logits — right accuracy, wrong reason |
| **`π` itself (Haar, this section)** | **11.72 %** | **5.04** | the target's mass: confidently wrong at 1.5 rad |
| `π` restricted to the cone (§6.3) | 48.3–49.0 % | 1.3136 | conditioned; the only regime that carries Task B |

The ambient chain's 11.7 % matches the base rate but **not the target** — 2.9
nats apart — so the ambient chain is not sampling `π` either, which is a second
correction to the "the chain retreated from the posterior" reading.  Everything
after this point in the document that reports a BMA should say which target it
is against, and the two are not interchangeable.

The earlier reading was not a sampler failure, it was a conditioning error.
The sweep below samples inside balls of radius `r ≤ 10⁻²` and self-normalizes
*within each ball*, so its "weighted acc" is `E[acc | d ≤ r]`, not `E[acc]`; its
"π-mass inside 2.3e-3 = 100 %" column says every *sampled* weight landed inside
the ball it was sampled from, which is a statement about the ball.  Both are
fine quantities.  Neither is `π`'s predictive, and the 48.8 % figure that the
reproduce section quotes is a conditional mean under a target the project does
not actually have.

**What this explains, and what it changes.**
1. **The 50 000-step run below is not pre-equilibrium dilution — the chain is
   going where `π` lives.**  `d` grew `7.55e-2 → 1.737e-1` (growth exponent
   0.521, 90 % of the free `r√t`), accuracy fell 34.8 % → 33.2 %, NLL rose to
   1.556: the walk is heading for the equatorial shell, where the answer is
   11.7 %.  Combined with the guidance result (drift delivers about half its
   budget, `R̂` 1.79 at `κ = 100`), no proposal and no budget can make the
   chain's BMA *be* 48.8 %, because that is not the target's value.
2. **The reviewer's point 4 is right in mechanism, not only in verdict.**  The
   entropy/volume term is not a correction to the likelihood's accounting, it
   *dominates* it: on `S^{16671}` the likelihood's `N·ΔL ≲ 2 500` nats cannot
   confine a posterior to a cone, because the cone has `10^{−55 000}` of the
   measure.  The "P/2 nats headroom" argument was the small-`P` shadow of this.
3. **The fix is a target, not a kernel.**  If the deliverable is a ~49 %
   predictive, the target must be `π` *restricted* to the cone — the centred/
   anchored prior of §6.4.2, or the conditional `E[acc | d ≤ r₁]` — and then the
   sweep's 48.3–49.0 % is exactly right, by construction, and the sphere chains
   are samplers for that restricted target.  Nothing in the Lean development or
   the readout module is affected: they are statements at fixed `β` about the
   kernel (`timepiece/pcnl.md` §3's scope note already says so).

### Is 46.39 % the target's predictive?  Measured without a sampler (2026-11-06)

The reading above says the chain left the basin; it does not say what the
target's predictive is.  `pcnl/examples/basin_posterior` answers that with no
chain at all: self-normalized importance sampling from the anchor (`u_Adam`,
`λ = 1`, `N = 512`, `P = 16 672`, 128 probes per width), weights
`exp(−N·(L − L_min))`.

| width `r` (rad) | ESS | weighted acc | weighted NLL | `N·medΔL` | `N·maxΔL` | π-mass inside `2.3e-3` |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 127.6 | 48.84 % | 1.3136 | 0.16 | 0.30 | 100 % |
| 3e-4 | 125.2 | 48.31 % | 1.3135 | 0.46 | 0.85 | 100 % |
| 1e-3 | 102.2 | 48.78 % | 1.3135 | 1.35 | 2.87 | 100 % |
| 2.3e-3 | 54.4 | 48.95 % | 1.3132 | 2.53 | 6.28 | 100 % |
| 5e-3 | 6.3 | 48.42 % | 1.3101 | 7.78 | 22.8 | no probe lands inside |
| 1e-2 | 1.1 | 45.82 % | 1.3116 | 24.7 | 94.9 | no probe lands inside |

**The ball's scale is 8.6 × 10⁻⁴ rad.**  The one-nat radius (`N·E[ΔL] = 1`,
read off the sweep) is `r₁ = 8.61e-4`, and the ESS collapses by the next grid
point (102 → 54 at 2.3e-3, → 6.3 at 5e-3).  (ESS is *not* the width criterion —
a ball so small that π is constant inside it has flat weights by construction,
which is why the maximum sits at the narrowest probe; the one-nat radius is the
criterion.)

Two exact proposals were then built to test that reading directly
(`basin_posterior --no-sweep --arm …`, 10 000 steps each):

| arm | proposal | accept | displacement | per-draw acc / NLL | BMA acc | `R̂`(NLL) |
|---|---|---:|---:|---|---:|---:|
| `density` | fixed tangent Gaussian, radius `r₁` | 0.07 % | frozen | 49.31 % / 1.3150 | 46.48 % | 2.66 |
| `shell` | `exp_u(r₁·d̂)`, `d̂` uniform — symmetric | 77.97 % | **7.59e-2 rad** | 36.40 % / 1.4403 | 34.77 % | 2.15 |

`--arm density` is exact in principle and fails in 16 672 dimensions: a draw
from a fixed law sits on a shell of relative width `1/√(2P)`, and the density
ratio between two typical draws is `−½(χ²_P − χ²_P) = O(√P)` — ±1.6 × 10² nats
here — so it accepts 0.07 % and reports its start.  (Two implementation traps
had to be fixed before it got even that far: the geodesic volume factor belongs
in the *proposal's* direction, `log_volume(r) − log_volume(r′)`, not the
reciprocal; and the chain must start from a *draw* of the fixed law, since
starting at its centre pays `φ(v)/φ(0) = e^{−P/2}` every step, freezes at the
anchor and reports `R̂ = 1.0000` because it never moved.)

`--arm shell` has no radial coordinate to get wrong: it proposes
`u′ = exp_u(r₁·d̂)` with `d̂` uniform in `T_u`, which is the uniform measure on
the geodesic sphere of radius `r₁`, hence **symmetric**
(`q(u′|u) = q(u|u′)` up to the constant `J(r₁)`, the same at both ends), so
`log α = −N·(L′−L)` and nothing else.  It accepts **78 %** and diffuses to
`√8000·r₁ = 7.7 × 10⁻²` rad, reporting 36.40 % / 1.4403 (BMA 34.77 %).

**That number cannot be π's predictive, and the contradiction is the useful
result.**  Walking the geodesic from the anchor through the chain's own final
point gives the potential profile (`--probe-ridge`):

| d (rad) | train L | `N·ΔL` | test acc | test NLL |
|---:|---:|---:|---:|---:|
| 0.000 (anchor) | 1.2532 | 0 | 49.22 % | 1.3136 |
| 0.010 | 1.2575 | 2.16 | 51.95 % | 1.3066 |
| 0.020 | 1.2741 | 10.67 | 48.44 % | 1.3198 |
| 0.040 | 1.3369 | 42.84 | 41.02 % | 1.3886 |
| 0.080 | 1.8377 | 299.3 | 20.31 % | 1.9480 |
| 0.120 | 2.7263 | 754.2 | 15.23 % | 2.8935 |
| 0.200 | 3.5811 | 1191.9 | 11.72 % | 3.7761 |
| 0.300 | 4.1030 | 1459.1 | 11.72 % | 4.2463 |

The profile is direction-dependent — a second ray (to a 300-step endpoint)
reads 54 nats at 0.02 rad and 544 at 0.04, ten times steeper — which is the
anisotropy the isotropic ball averages over; both rays put π's mass inside a
few milliradians.  So π is gone by 0.04 rad while the chain sits at 0.076 with
78 % acceptance, and `PCNL_PROGRESS=1` shows it **climbing** (`L` 1.2535 → 1.2630 as `d`
goes 0.0009 → 0.017 rad, 0.19 → 4.9 nats) instead of fluctuating about the
mode.  That first read as an implementation bug.  It is not: the jump now lives
in `sphere::shell_jump` and three closed-form moment tests certify it —

| test | target | statistic | reads |
|---|---|---|---:|
| `shell_jump_matches_the_von_mises_moment` | von Mises, `P = 2`, κ = 2 | `E[u₁]` vs `I₁/I₀ = 0.6978` | passes (4 se) |
| `shell_jump_is_exact_on_s2` | vMF, `P = 3`, κ = 2 | `E[u₁]` vs `coth κ − 1/κ = 0.5373` | passes (4 se) |
| `shell_jump_samples_an_isotropic_gaussian_in_the_tangent` | `N(0, σ²I)` in the chart, `P = 5`, σ = 0.5 | `E[r²]` vs `(P−1)σ² = 1.0` | 0.979 ± 0.014 |

— so the ratio is right and the climb is **mixing**.  The two wrong closed
forms these tests were first written with are worth recording, because both
looked like kernel bugs and neither was: the Gaussian target needs the
Jacobian with a *minus* sign (`log π = −r²/2σ² − (P−2)log(sin r/r)`, since the
Haar measure is `(sin r)^{P−2}dv`), and its moment is `(P−1)σ²`, because the
tangent is `(P−1)`-dimensional — with `Pσ²` the test rejects at 20 se and with
the sign flipped it reads 0.64.

The quantity that governs the climb is the radial relaxation time.  One jump
changes `‖log_{u₀} u‖` by `O(r/√P)`, so that coordinate is an OU process with
time constant `≈ P/(K r₁²)`; with the sweep's `K = 2N·medΔL/r² ≈ 2.7 × 10⁶`
that is `16 672/(2.7e6 · 7.4e-7) ≈ 8 × 10³` steps in a median direction and
`5 × 10⁵` in the softest — the 10 000-step budget sits *at* the relaxation
time.  The numbers say the same: 0.041 rad at 3 000 steps, 0.076 at 8 000
(diffusion still winning), and accuracy falling 44.3 % → 34.8 % as it goes.
**`--arm shell`'s 34.77 % is therefore a pre-equilibrium value, not π's
predictive**; the measurement that would settle it is a 5 × 10⁴-step run, and
the 10 000-step rows must be read as "≈ 8 × 10³ steps of a 10⁴-step
relaxation".

### The climb is a random walk: four chains, 10 000 steps (2026-11-06)

The paragraph above argues that 10 000 steps sits inside the relaxation time.
An argument is not a measurement, and "pre-equilibrium" covers two very
different faults: an unbiased chain that has not had time to contract, or a
kernel that pushes outward (a bug).  They are separable without knowing the
target — a random walk grows as `t^{1/2}` with *independent* directions in each
chain, a drift grows as `t¹` along a *common* direction.  `basin_posterior
--chains 4` asks both questions at once (10 000 steps, burn 2 000, thin 25, four
independent seeds, one shared `Env`; ~18 min on two cores):

| chain | accept | `d` final (rad) | growth exponent | per-draw acc | per-draw NLL | train L |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.0 % | 7.578e-2 | 0.508 | 36.40 % | 1.4403 | 1.4195 |
| 1 | 76.4 % | 7.506e-2 | 0.504 | 37.01 % | 1.4246 | 1.3909 |
| 2 | 77.8 % | 7.555e-2 | 0.511 | 35.77 % | 1.4294 | 1.4438 |
| 3 | 78.1 % | 7.588e-2 | 0.513 | 38.73 % | 1.4322 | 1.4263 |

split-`R̂`(NLL) 1.5003, split-`R̂`(train L) 1.8532, BMA over the four chains
38.28 % / NLL 1.4115.

**Mean growth exponent 0.509** (0.504–0.513) — `√t` over the whole run — and
**mean pairwise angle between the chains' net displacements 89.7°**.  The angle
is the discriminating column, and the magnitudes are not: a walk in `P = 16 672`
dimensions concentrates radially at `1/√(2P) = 0.55 %`, so all four chains land
within 2 % of each other under *either* hypothesis, which is why an earlier
reading of "the chains agree on 0.076 rad" could not tell drift from walk.
Independent directions at `t^{1/2}` say the outward creep is diffusion under a
potential that is nearly flat over the region reached: nothing favours outward
moves, and `shell_jump`'s ratio is doing its job — which is what the three
moment tests say from the other end.  The mix-up was budget, not kernel.

`d(½)/d(1)` = 1.29–1.30 is exactly `√(10⁴/6 × 10³)` (the recorded window starts
at the burn-in, so "mid" is `t ≈ 6 000`), i.e. **no saturation is visible even at
10 000 steps**: the run ends before the OU restoring force has bent the exponent
below ½, and the 5 × 10⁴-step run remains the measurement that would find the
scale where it does.  Both `R̂` are far above 1.05, so these four chains are not
sampling one law yet — pre-equilibrium, now measured on four chains rather than
on two halves of one.

Two by-products.  Chain 0 reproduces the single-chain row above to every digit
(36.40 % / 1.4403 / 7.578e-2), which also certifies `Dataset::train_loss`: `pot`
used to call `loss_and_grad` and keep only the loss, paying a reverse sweep per
proposal for nothing; the forward-only twin returns the same value by
construction, and the identical row is the empirical check.  And the four-chain
BMA (38.28 % / 1.4115) is better than any single chain's NLL while still 0.10
nats above the target's 1.3136 — dilution *inside* the basin, not a biased
kernel, which is the reading the section above already gives for the
single-chain 34.77 %.

**What survives for the original question: 46.39 % is not the target's
predictive — but neither is the sweep's 48.8 %.**  Read this paragraph with the
section above: the sweep is a *conditioned* mean (`E[acc | d ≤ r]`), and `π`'s
unconditional predictive is the base rate, 11.72 % / NLL 5.04.  What the
10 000-step ensemble's extra 1.4–1.9 pts of loss measure is *dilution inside*
the cone on the way out to the shell, not a biased kernel: its per-draw
NLL 1.3412 against 1.3136 is 14 nats of posterior cost, i.e. a typical draw
~3 × 10⁻³ rad out — 3.5× the one-nat radius — and a proposal whose step
accumulates as `√t` dilutes by `√t` whatever its acceptance.  So the number is
a mixing artefact, not a property of the target: the same conclusion as the
400-step row, but with the direction of the artefact corrected — it is *towards*
`π`'s own 11.72 %, not towards the cone's 48.8 %.

**"A proposal that stays in the basin" is therefore not a proposal at all — it
is a decision about the target.**  Sampling `π` exactly gives the base rate no
matter how good the kernel is; getting 48.8 % requires conditioning on the cone
and saying so.  The fixed-radius jump remains the right *shape* for the
conditioned target (exact in principle, no `O(√P)` density term, no
accumulation), and its moment tests are what certify it.

### Guidance: the drift budget is `r·A` per step, and it is measurable (2026-11-06)

A fixed-radius jump has only two knobs left, and the earlier sections ruled out
both: the radius (`≈ r₁`, else `O(√P)` density terms) and a metric (closed by the
subspace bound).  What neither touched is the **direction law**: `shell_jump`
draws it uniformly, so the jump has no mean.  `sphere::guided_shell_jump` keeps
the radius and draws the tangent direction from a von Mises–Fisher law
`p(ŵ) ∝ exp(κ⟨μ̂,ŵ⟩)` about the ascent direction, with the reverse term
`κ(⟨μ̂′,ŵ′⟩ − ⟨μ̂,ŵ⟩)` in the ratio; `μ̂` is the frozen ambient gradient at the
anchor, projected at each proposal (`sphere::drift_direction`).  The filter is
exact for **every** `κ` — a wrong field costs acceptance, never correctness —
and four tests say so: the vMF moment on `S²` with a guide orthogonal to the
target's axis, a flat target staying uniform under a strong guide, the
direction law actually pointing (the `κ = 2` circle moment), and the new
`vmf_tangent_mean_cosine_scales_as_kappa_over_p`, which pins the one constant
the budget below needs.

**The budget.**  On the `(P−2)`-sphere of directions the mean cosine is
`A_P(κ) ≈ κ/(P−2)` (linear while `κ² ≪ P`, which at `P = 16 672` means
`κ ≲ 130`), so one jump of radius `r` buys `r·A` rad of drift against the free
walk `r√t`: the two **cross at `t ≈ (P/κ)²`**, and the guided displacement is
`r√t(1 − A√t)`.  A sweep of `κ` at 3 000 steps tests both halves of that
sentence (free walk measured at 4.10e-2, `A√t` at `t = 3 000`):

| `κ` | `A = κ/(P−2)` | `A√t` | predicted `d` | `d` final | miss | accept | per-draw acc | per-draw NLL | BMA acc / NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 (`shell`) | 0 | 0 | 4.10e-2 | 4.10e-2 | — | ~78 % | 44.3 % | — | 44.3 % / — |
| 0.3 | 1.8e-5 | 0.001 | 4.10e-2 | 4.009e-2 | — | 74.1 % | 43.09 % | 1.3562 | 45.70 % / 1.3495 |
| 30 | 1.8e-3 | 0.099 | 3.69e-2 | 3.967e-2 | −7 % | 72.4 % | 43.09 % | 1.3495 | 40.62 % / 1.3424 |
| 60 | 3.6e-3 | 0.197 | 3.29e-2 | 3.615e-2 | −10 % | 59.4 % | 42.74 % | 1.3659 | 39.84 % / 1.3606 |
| 100 | 6.0e-3 | 0.329 | 2.75e-2 | **3.041e-2** | −11 % | 42.0 % | 42.74 % | 1.3324 | **47.27 % / 1.3276** |
| 160 | 9.6e-3 | 0.526 | 1.94e-2 | **2.134e-2** | −10 % | 20.6 % | 45.57 % | 1.3329 | 42.58 % / 1.3293 |
| 240 | 1.44e-2 | 0.789 | 8.7e-3 | **1.235e-2** | −43 % | 6.8 % | 48.26 % | 1.3289 | 49.22 % / 1.3271 |

Each row is one chain, 3 000 steps, burn 600, thin 7, `--no-sweep --arm guided`,
`PCNL_KAPPA_GUIDE=κ`.  The prediction uses no fitted constant: `A` is the
`κ/P` form the module's `vmf_tangent_mean_cosine_scales_as_kappa_over_p` test
pins, and the free walk is the `κ = 0` row measured in the same harness.

**The contraction is real and priced.**  `d` falls monotonically with `κ` and
the linear form is right to about 10 % wherever its own assumption holds
(`A√t ≤ 0.53`), with a consistent `−10 %` sign: the delivered drift is a little
less than `r·A` per step, as expected from the field's slow misalignment (below).
The `κ = 240` row misses by 43 % and is the model's own failure mode, not a
measurement surprise — `κ²/P = 3.5` there, so `A` has left the linear regime and
`κ/(P−2)` overestimates the true mean cosine.  The price is acceptance:
74 % → 6.8 %, because the reverse of a directed step is nearly antiparallel to
the forward one and the ratio fluctuates like `exp(−2κ⟨μ̂,ŵ⟩)`.  Two consequences
follow, and both are arithmetic rather than fit.

* **The crossover is `(P/κ)²` steps, which is why `κ = 0.3` did nothing.**  At
  `κ = 0.3` the crossover is 3 × 10⁹ steps, so the 10 000-step four-chain run at
  `κ = 0.3` reads like the isotropic one — 40.23 % / 1.3993, `d` 7.4–7.7e-2,
  growth exponent 0.510, pairwise angle 89.6° — a 1.9-pt and 0.012-nat edge from
  guiding being present at all.  Any `κ ≲ 10` is *unmeasurable* at this budget.
* **`κ = 240`'s 49.22 % is the anchor, not the target.**  Its `d` is 1.2e-2 and
  its per-draw accuracy equals `u_Adam`'s own 49.22 %: the chain has been
  walked back to its start, so it reports the start's predictive.  That number
  is *correct sampling of `π`'s inner ball* (the sweep's 1e-3 row is 48.78 %)
  and it costs no bias — the filter is exact — but it buys its accuracy by
  making ~200 moves in 3 000 steps, so it says nothing about ESS or `R̂` until
  those are measured.  The clean statement is that the guided family trades
  **dilution for effective sample size**, with `κ/P` as the exchange rate.

**What this closes.**  The open item was "a proposal that does not diffuse out of
a sub-milliradian basin".  Guidance cannot be that proposal *and* move: the
drift and the acceptance penalty are the same constant, so at `37 %` acceptance
(`2κA = 1`) the drift-to-diffusion ratio is at most `A√t ≤ √(t/(2P))` — 0.55 at
`t = 10⁴`, `P = 16 672`.  What guidance can do is shorten the *time* to the
ball by a known factor, and at `κ ≈ 100` it does: the 3 000-step `d` is 26 %
below the isotropic arm's and the BMA is 47.3 % against 44.3 %.

**The 10 000-step version, four chains** (`--chains 4`, same burn/thin as the
isotropic run above), which is the number the budget predicts rather than the
3 000-step transient:

| arm | accept | `d` final (rad) | growth exponent | BMA acc / NLL | split-`R̂`(NLL) | angle |
|---|---:|---:|---:|---|---:|---:|
| `shell` (isotropic) | 78 % | 7.55e-2 | 0.509 | 38.28 % / 1.4115 | 1.500 | 89.7° |
| guided `κ = 0.3` | 77 % | 7.45e-2 | 0.510 | 40.23 % / 1.3993 | 1.415 | 89.6° |
| guided `κ = 100` | **41 %** | **5.47e-2** | **0.492** | **44.14 % / 1.3661** | 1.794 | 89.8° |

Guided at `κ = 100` is the best of the three on every accuracy column — +5.9 pts
and −0.045 nats over the isotropic arm, still `R̂ = 1.79` and still 2 pts short
of the target's 48.3–49.0 %.  The `d` is 28 % below the free walk (5.47 vs
7.55e-2) where the calibrated budget at `t = 10⁴` (`A√t = 0.60`) predicts 54 %:
the drift delivers **about half of its own budget** over this horizon.  The
missing half is the frozen field's alignment decaying as the chain leaves the
anchor — the anchor's gradient projected at a point 5e-2 rad (64 jump radii)
away is no longer the inward geodesic, so `A_eff < A` exactly where the drift
was needed, and the residual is `√t` again.  That is the mechanism behind the
growth exponent staying at ½: the guidance is spent early and gone late.

The four chains are also *nearly identical to each other* (±0.6 % in `d`,
±0.7 pt in per-draw accuracy) while each is still moving
(`d(½)/d(1) = 1.28`, no saturation).  So `R̂ = 1.79` here is **not** ensemble
disagreement — it is four copies of one trajectory being flagged for being
non-stationary, which is the diagnostic doing its job on a different fault than
the usual one.

**Why the field cannot be fixed: drift *is* the acceptance cost.**  The frozen
field's decay has an obvious repair — evaluate `∇L` at the chain's current point
(`PCNL_LIVE_GRAD=1`, one extra backward per proposal) so `μ̂` is always the
ascent direction where the chain actually is.  It makes acceptance *better* and
the drift *worse*:

| field (`κ = 100`, 3 000 steps) | accept | `d` final | per-draw acc | per-draw NLL | BMA acc / NLL |
|---|---:|---:|---:|---:|---|
| frozen at the anchor | 42.0 % | **3.041e-2** | 42.74 % | 1.3324 | **47.27 % / 1.3276** |
| live at the current point | **66.9 %** | 3.839e-2 | 43.79 % | 1.3804 | 42.97 % / 1.3697 |

The reason is a one-line identity.  With `ŵ′ ≈ −ŵ` (the reverse jump retraces
the geodesic) the ratio's guide term is `exp(κ(⟨μ̂′,ŵ′⟩ − ⟨μ̂,ŵ⟩))`, so it is
`exp(−2κ⟨μ̂,ŵ⟩)` when the field barely varies (frozen: mean ≈ 1, but the
anti-guided half of the draws is killed — that kill *is* the drift, and it is
also the 42 % acceptance), and `exp(κA − κA) ≈ 1` when the field flips with the
point (live: near-symmetric, so near-unbiased — 67 % acceptance, no drift).  A
field aimed straight at the anchor is the same identity at the limit:
`μ̂ = −back` exactly, so the ratio is `exp(−κ(1+⟨μ̂,ŵ⟩)) ≈ e^{−κ}`, i.e. dead at
any useful `κ`.  **The component of the proposal that the reverse term does not
cancel is the drift, and it is also exactly what the filter removes.**  Guidance
cannot be both local and directional, and the measured columns above are that
sentence in numbers (see also the `κ = 4` flat-target note in
`guided_jump_keeps_a_flat_target_uniform`: the same trade-off in two
dimensions).

### The 50 000-step run: the walk does not turn around (2026-11-06)

The relaxation-time argument above predicted that a long enough run would find
the scale where the outward climb bends below `t^{1/2}`.  Five times the budget,
measured (one chain, `--no-sweep --arm shell`, burn 10 000, thin 125, ~46 min):

| steps | accept | `d` final | `d`/free | growth exponent | per-draw acc / NLL | BMA acc / NLL | `R̂`(NLL) |
|---:|---:|---:|---:|---:|---|---:|---:|
| 10 000 | 78 % | 7.55e-2 | 0.88 | 0.509 | 36.4 % / 1.440 | 38.28 % / 1.4115 | 1.50 |
| 50 000 | 83 % | **1.737e-1** | **0.90** | **0.521** | 31.3 % / 1.574 | 33.20 % / 1.5560 | 2.43 |

**There is no turn-around.**  `d` grew `2.3×` for `5×` the steps — exactly
`t^{0.52}` — tracking the free random walk to within 10 %, and the growth
exponent is *above* ½ rather than below it.  Accuracy fell 34.8 → 33.2 % and
the per-draw NLL rose 1.440 → 1.574, which is the chain leaving the cone at
roughly the rate the potential slows it.  So the earlier `τ ≈ 8 × 10³`–`5 × 10⁵`
estimate was optimistic by at least an order of magnitude, and the reason is now
known rather than outstanding: **the potential is not confining, because `π` is
not concentrated.**  Under the prior-IS measurement the chain is doing the
right thing — walking to the equatorial shell where the mass is — and its
accuracy is on the way from the cone's 48.8 % down to `π`'s 11.72 %.

### The same class of error in the ambient kernels (2026-11-06)

Audit of the two ambient kernels after the restricted-metric bug, asking each
of them the question that bug answered wrongly: *is the acceptance ratio the
density of the move the chain actually made?*  The two kernels are
`sampler::SgPcnlState::step_unadjusted` (unadjusted SG-pCNL / SGLD) and
`task::mh_pcnl_chain` (ambient pCN, with `exact` selecting the three-term
Metropolis ratio).

**1. The unadjusted kernel has no ratio, so it has no such bug.**  It is the
SGLD-style schedule; nothing is accepted or rejected, and no exactness is
claimed.  Nothing to change.

**2. The ambient filter needs no determinant term, and that is not an
oversight.**  In ambient space the proposal is `N(m(θ), β²σ²I)` — isotropic,
same covariance at both ends — so the Gaussian normalizer cancels and the
ratio is `(‖θ′−m(θ)‖² − ‖θ−m(θ′)‖²)/2β²σ²` plus the target ratio.  The sphere
differs *only* because `Σ = P D P` is a restricted operator whose determinant
moves with `u`.  So the metric bug has no ambient sibling.  The algebra is now
one pure function (`pcn_proposal_log_ratio`) with a test against a dense f64
Gaussian evaluation (`exact_q_ratio_matches_a_dense_gaussian_ratio`).

**3. The drift cap was applied to only one end of the ratio — a real bug of
exactly that class.**  The exact filter's reverse mean used the *raw* gradient
(`m_prop = mean_at(&proposal, &g_prop)`) while the forward proposal used the
capped one, so `q(θ|θ′)` was the density of a move the chain never proposed.
The cap is part of the proposal map,
`m(θ) = √(1−β²)(θ−μ) + μ − h·cap(∇U(θ))`, and is now applied at both ends
from a single closure.  The error is O(1) in the log ratio whenever the cap
binds, and its sign depends on the regime: in the reduced configuration the
pre-fix code *over-rejected* (23/400 accepted against 58/400), while on a
stack whose gradient is `10⁶×` the cap the spurious term is huge and every
move is accepted instead.  **No reported number moves**: no documented run
combines `exact` with a cap — `mh_standard_target`'s filtered arms all use
`exact: false`, and the only `exact: true` runs (the `λ = 0` pCN identity) pass
no cap.  The combination was simply wrong as an API.

Regression test: `exact_filter_is_cap_consistent` in `tests/comparators.rs`.
A cap small enough that `h·cap` is negligible makes the drift vanish, so the
exact filter must coincide with the likelihood-only one; pre-fix it reads 15
accepted of 300 against 47, post-fix the two agree and an inert cap leaves the
exact chain bit-identical to the uncapped one.

**3b. The examples do not re-implement the kernel.**  All eight examples that
sample (`step_scale_control`, `mh_standard_target`, `overdispersion`,
`temperature_sweep`, `annealed_start`, `cold_ladder`, `mh_cold`, `adam_start`)
routes every update through `SgPcnlState::step_unadjusted` or
`mh_pcnl_chain`; `step_scale_control::chain` says so in a comment and means it.
The two files that do carry inline step arithmetic are `step_distribution`
(unadjusted pCN and ULA arms — no ratio, so nothing to be inconsistent with) and
`curvature_probe` (diagnostics only, no acceptance).  So the cap bug had no
second copy, which is the opposite of the earlier inline-probe defect and worth
recording as such.

**4. The sphere's geodesic proposal carries a volume factor that legitimately
cancels — and no test knew it.**  In geodesic normal coordinates
d`vol = (sin a/a)^{P−2} dv`, so `exp_u` needs that factor at both ends.  It
cancels because the reverse displacement has the same radius `a`
(`‖reverse_tangent_vec‖ = a` identically), which is why the kernel may omit
it.  `the_exp_map_volume_factor_cancels_because_the_radii_match` now pins that
invariant at `P ∈ {2,3,4,7}`: a future reverse map with a different radius (a
different step, a retraction) would silently drop a term that no longer  cancels.  The factor is `^0` at `P = 2`, so every circle test in that module
  is blind to it — the same structural blindness that hid the metric bug.

**5. One cap, two definitions.**  `temperature_sweep` passed
`grad_norm_cap: Some(7.0e2)` with the comment "the normalized geometry's
scale, as in §6.5.2", while `mh_standard_target` *measured* the same constant
per seed: 712 / 757 / 764.  The literal is 6–9 % low, so those rows were not
the §6.5.2 cell they claimed to be.  A cap changes only the proposal (the
ratio stays exact for `π_λ`), so no conclusion moves — but two definitions of a
constant compared across cells is the same failure mode as two definitions of
a proposal map, one level up.  `task::cosine_gradient_scale` is now the single
definition and both examples call it; `temperature_sweep` prints the cap next
to each seed's rows.  Nothing else in the eight sampling examples constructs
an acceptance ratio outside the two audited kernels, so this was the last call
site that could disagree with itself.

### The mode, the subspace bound, and an unconverged cell (2026-11-05)

Three follow-ups the section above named, each with its own example.

**1. Where the sphere's mode is (`examples/sphere_mode.rs`).**  Riemannian
descent on `−mean NLL` with exponential-map steps and Armijo backtracking, from
Adam's direction and from three Haar draws (150 accepted steps each, the
tangential gradient at the end 2.6e-1 — not stationary, just out of iterations):

| start | mean NLL (train), end | acc (test) | NLL (test) | angle from `u_Adam` |
|---|---:|---:|---:|---:|
| `u_Adam` (from 1.2532) | **1.2226** | 49.22 % → **51.17 %** | 1.3136 → **1.2748** | 7.7 × 10⁻⁴ rad |
| Haar #1 (from test 4.6667) | 1.9810 | 11.72 % → 11.72 % | → 2.0225 | — |
| Haar #2 (from test 4.8332) | 2.0054 | 11.72 % → 12.89 % | → 2.0489 | — |
| Haar #3 (from test 4.8847) | 1.6221 | 11.72 % → 25.39 % | → 1.6552 | — |

So Adam's direction is **near the mode but not at it**: the mode is 7.7 trust
radii away and worth +1.95 points of accuracy and −0.039 nats.  Two
consequences.  First, the metric chain's ≈50 % is *the mode's* accuracy, not
evidence that a sampler found something Adam could not — the ascent finds the
same number without any kernel, and the corrected chain lands on 49.02 % ± 3.04
(4 chains).  Second, Haar starts do not reach the
informative basin at all (11.7–25.4 %, NLL 1.66–2.05): scale invariance makes
the likelihood *low* in most directions, but “low” is not “flat → optimisable”,
and §6.4(ii)'s volume statement survives the optimisation test, not only the
random-draw one.

**2. No subspace metric can accept at the published β
(`examples/subspace_bound.rs`).**  A rank-`m` shrink proposal with
`C = I − (1−c)·P_m` reduces the pCN cost by at most `1/(1−f)`, where `f` is the
fraction of `Tr(H)` the top-`m` subspace carries.  Power iteration on the
finite-difference curvature (`A·v = ρ(g(θ+ερv) − g(θ))/ε`, ε = 1e-5 rad, 8
directions) and Hutchinson for the trace (16 Rademacher probes):

| m | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| `Σλ_k / Tr(H)` | 0.193 | 0.365 | 0.543 | **0.728** |
| best reduction | 1.24× | 1.57× | 2.19× | **3.67×** |
| β it would allow | 0.0096 | 0.0122 | 0.0170 | **0.0285** |

`Tr(H) ≈ 1.81 × 10⁷` (1 086 per coordinate), and its top eigenvalues span
3.50e6 … 5.34e5 — the curvature is **delocalised**, not low-rank.  Acceptance
at `β₀ = 0.3` needs a 39× reduction (`f ≳ 0.974`); the ceiling with `m = 8` is
3.67×, and it would cost 56 gradient evaluations to learn.  So the answer to
"try a low-rank metric" is no, with a bound rather than a failed run — and the
trace cross-checks the probe: `τ²Tr(H) = 97.7` at `β₀ = 0.3` against the 77.4
measured for the exponent.

**3. The β₀ = 0.003 metric cell is not converged.**  Two chains, 400 steps
(burn 200, thin 20, K = 10), metric, `β₀ = 0.003`, from `u_Adam`:

| accept | BMA acc | BMA NLL | across-chain spread | `R̂(loss)` |
|---:|---:|---:|---|---:|
| 42.6 % | 50.20 % | 1.3137 | ± 1.38 pts / ± 0.0012 nats | **3.08** |

That cell has since been re-measured with the exact ratio — 88.9 % acceptance,
49.80 % ± 1.93, `R̂ = 4.85` on two chains, then 87.8 %, **49.02 % ± 3.04**,
`R̂ = 3.00` on four (`sphere_chain --precond --seeds 2`) — so treat the numbers
here as superseded; what survives is the verdict — `R̂` 3–5 says the
chains are not sampling one distribution at this budget and are still drifting
through the neighbourhood.

The 10 000-step follow-up (*The 10 000-step attempt* below) settles the question
this paragraph left open, and against the favourable reading: `R̂` stays at
2.894, the chains *do* come to agreement (± 0.37 pts) but on 46.39 % — worse
than the frozen start.  So the 400-step retention was a transient, and a
converged number would need a proposal that does not diffuse out of a
sub-milliradian basin; more steps at `β = 0.003` cannot supply one, because the
step size that accepts at the solution is the step size that leaves it.

### What the flat-region reading changes in §6.4–6.5 (2026-11-05)

The acceptance column above forces a re-read of the earlier sections, because
most of them were measured while the chain sat where the likelihood is flat.
Nothing measured is withdrawn; several *interpretations* narrow.

| earlier claim | status after the sphere/flat-region results |
|---|---|
| §6.4(ii) — beating-chance directions occupy a sub-1/2000 fraction of the sphere | **unchanged.** A statement about the geometry of directions; no sampler involved. |
| §6.5.5 — `r = βσ‖g‖/(2√P) ≈ 0.05` at `β₀ = 0.3`, so the direction is prior-dominated | **unchanged arithmetic, reinterpreted.** `r` is small *because the chain is where `‖g‖` is small*. At the solution the same proposal is rejected outright (0 % at `β ≥ 0.03`), so `r` is a diagnostic of *position*, not a knob on the schedule. |
| §6.5.6 — 90–100 % acceptance means the chain is well tuned | **withdrawn.** Acceptance measures the region, not the sampler: the same kernel accepts 100 % from a Haar direction and 0 % from the solution's. High acceptance and a prior-dominated predictive are one fact. |
| §6.5.3 — "the overconfidence is not the target's" | **narrows to the comparison.** The ambient `π` concentrates its mass on the flat region, and those directions are *confidently wrong* (NLL 4.67 for `ρ·u_Haar`), so overconfidence is a property of where the target lives. What survives: at a usable proposal scale the filtered chain's BMA is less overconfident than the unadjusted one. |
| §6.5.4 / §6.5.5 — temperature and annealing change nothing | **unchanged, and now explained.** No schedule creates curvature where the likelihood is flat; cold `λ` sharpens *within* that region, which is why it bought calibration and never information. |
| §6.4.2 — the centred prior reaches 48.8 % on the shift cell | **unchanged**, and complemented by `adam_start`'s anchored arm at `β₀ = 0.03` (BMA 1.9500, below the uniform predictor). |
| §6.5.1(c) — `R̂(loss) ≈ 1` marks chains that found the same (poor) mode | **unchanged**; the "modes" are flat regions with confidence, and the split at 8 chains / 30 000 steps is chains landing in different ones. |
| §6.5.7 — the Gaussian prior excludes the solution; a Haar prior on directions would contain it | **confirmed at the target level, narrowed at the sampling level.** `ρ·u_Adam` scores identically to `θ_Adam`, so the direction does have full mass — but "contains the solution" is not "can be sampled there": the accepting window (`β ≤ 0.003`) sits `√P` below the informative one, and the chain's BMA (49.8–51.2 % across runs, `R̂` 3–5) sits between the frozen start's 49.22 % and the mode's 51.17 % — the mass is **near** the solution, not on it, and `u_Adam` is not the sphere's mode either. |
| "the sampler cannot find the solution" | **true but trivial for the Gaussian target**, because `π` has 6.6 × 10⁶ nats of prior cost there. The substantive statement is that the published target is mis-specified — not that a sampler failed. |

Two consequences for how to read the rest of this document.  First, every
"prior-dominated" number is a statement about the ambient `N(0, σ²I)` target,
and the repair (§6.5.8's predecessor sections) changes the target rather than
the sampler.  Second, the well-mixed diagnostics that the paper leans on —
ESS(loss) 194–226, `R̂(loss)` 1.004 — certify agreement on a flat region; they
were never evidence that the chain's location was informative, and §6.5.8 is
the measurement that separates the two.

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

### The target's own conditionals, exactly — and why they are a tail (2026-11-07)

The previous section left one thing unnamed: the 48–51 % numbers are estimates
of `E[acc | d(u, u₀) ≤ r]`, and the chain that would sample that conditional
walks *out* of the ball.  One example (`basin_posterior`, four arms) closes the
question in stages, and the last stage closes it *against* the earlier readings:
the exact conditionals are a tail of `π`, not its predictive — the target leaves
99.3 % of the tangent space unconstrained, and the arithmetic says why.

**1. A cap is the exact fix, and the wrong move.**  Restricting the **target** to
the ball is the exact fix in principle (`sphere::restrict_to_cap`; the kernel is
untouched and the ratio stays exact because `π = 0` outside), and it is the wrong
move in `P = 16 672` in practice — measured, both radii:

| run | target | accept | `d` reached | `R̂`(NLL) | per-draw acc / NLL | BMA acc / NLL |
|---|---|---:|---:|---:|---|---|
| `--cap` (= `r₁`) | `π ∩ {d ≤ 8.61e-4}` | **0.0 %** | 8.610e-4 after 1 step | ∞ | — | — |
| `--cap 2.3e-3` | `π ∩ {d ≤ 2.3e-3}` | **0.1 %** | 2.266e-3 by `t ≈ 7` | ∞ | 46.5–48.4 % / 1.310–1.317 | 48.44 % / 1.3132 |

Both runs pin to the cap boundary and stop.  The arithmetic is the geometry of
§3.3 again: a step `r` from radius `d` lands at `d′² = d² + r² − 2dr·t` with
`t = ⟨u, ŵ⟩`, so at the boundary `d = R` the move stays inside only when
`t ≥ r/(2R)`; on `S^{P−1}`, `t` has sd `1/√P`, so the acceptance there is
`Φ̄(r√P/(2R))` — `e^{−P/8}` at `R = r`, and 0.7 % at `R = 2.7 r` (which is the
measured first-1000-step column).  The walk reaches `d = R` in `(R/r)² ≈ 7`
steps because `d²` accumulates `r²` per step, and it never leaves.  No radius
fixes this: inside a cap the Haar measure `(sin d)^{P−2} dd` puts essentially
all of its volume **on the boundary**, so the boundary is exactly where the
chain has to live.

The number the pinned run reports is still worth having, because it is not a
sample but a *point estimate that two independent methods agree on*: four
boundary points give 48.44 % / NLL 1.3132, and the 128-probe self-normalized
importance sample of §"Is 46.39 % the target's predictive?" gives 48.95 % /
1.3132 at the same radius.  The conditional's predictive is 48.4–49.0 %; what
the cap run cannot do is put an error bar on it (`R̂ = ∞` is the diagnostic
saying so, and it is the one column that would have caught this without the
distance column).

**2. `sphere::latitude_jump` is the kernel that can live on the boundary.**  It
rotates `u` about the axis `u₀`, so `⟨u₀, u⟩` is invariant to machine precision:
the proposal is the uniform measure on the geodesic sphere of radius `r`
*inside* the latitude, the reverse law is the same object at `u′`, and
`log α = log π(u′) − log π(u)` with no Jacobian.  The cap never rejects a
proposal, the radius column is constant by construction rather than by
censoring, and the relaxation time along the shell is `(sin d₀ / r)²` steps —
which is the knob `--step` exposes, because at `d₀ = 0.079` the default `r₁`
needs 8 400 steps to cross the shell and a 3e-3 step needs 700.

**3. Writing it found three defects and one test error.**  All are worth
recording, because the first two look like kernel bugs and are, while the third
looks like physics and is arithmetic:

- **Projecting with `w − ⟨w,u⟩u − ⟨w,a⟩a` is not a projection.**  The two
  directions are nearly dependent (`⟨a,u⟩ = c`), so subtracting both at once is
  a Jacobi step whose error shrinks by `c²` per pass: at `c = 0.92` two passes
  still leave a fifth of the component, the direction leaves the latitude, and
  the chain walks off the sphere while `⟨a,u⟩` stays right.  Measured `|u|` =
  1.12 after one step.  The span removed must be an *orthonormal* pair —
  `{a, û}` with `û = (u − c·a)/sin d` — which makes one pass exact.
- **The construction preserves `⟨a,u⟩` but not `|u|`, and the error
  amplifies.**  With `û` taken as `(u − c·a)/sin d`, a rounding error `δ` in
  `|u|` returns as `|û| = 1 + δ/sin²d`, the next proposal is `1 + δ·cot²d`, and
  the growth is geometric: from `1e−16` to `O(1)` in ~50 steps.  Measured
  `|u| = 3.07` at the worst kept draw.  Rebuilding the proposal from `c` and
  the transverse *direction* (rather than its normalised sine) makes every
  accepted state exactly unit and self-heals a bad input.

  What this one cost is worth stating: the moment test did **not** catch it —
  `E[u₂]` read 0.434 off-sphere against 0.445 on-sphere (the correct value is
  0.429, so both are within one se).  The invariant had to be instrumented
  directly (`|u|`, per kept draw).  A tolerance wide enough for mixing noise is
  also wide enough to hide a broken manifold, so the two need separate columns.
- **The anchor is only `f32`-unit, and `|a|` leaks into `|u|`.**  `u₀` is
  `to_vec(&unit(&theta_Adam))`, so `|a| = 1 ± 5e-8`, not 1.  A kernel built as
  `c·a + s·(…)` with `s = √(1−c²)` then gains `2c²(|a|² − 1)` in `|u|²` every
  step: `|u|` walks up by `5e-8` per step until `⟨a,u⟩` reaches 1, `sin d` hits
  0, and the chain freezes on the anchor with every later move rejected.  The
  collapse time is `(1 − cos d₀)/(2|a|δ)` steps — `≈ 10³` at `d₀ = 8.61e-4`,
  `≈ 10⁵` at `d₀ = 0.079` — which is why the tests at `d₀ ∈ {0.4, 1.0}` passed
  and the two smallest radii did not: they read `d = 0`, `R̂ = ∞` and an
  acceptance decaying as `1/t` (0.8 % at `d₀ = r₁`, 6.0 % at `2.3e-3`), which
  from a distance looks like a chain that is simply rejecting.

  The fix is to normalise both vectors *inside* the kernel and rebuild the
  state as `c·â + s·(…)` with `c = ⟨u,â⟩/(|u||a|)`, so every accepted state is
  exactly on the unit sphere and the invariant holds to rounding.  The
  regression test is the pair `(|u|, ⟨â,u⟩)` at `d₀ = 8.61e-4` over 4 000
  steps with an `f32`-unit anchor: the frozen-chain failure moves `⟨â,u⟩` by
  `1 − cos d₀ = 3.7e-7`, the fixed kernel by `5e-16` per step with no
  amplification.
- **The test's closed form was one dimension wrong.**  At `P = 3` the latitude
  is a **circle**, so the conditional law of a vMF target is the 2-D von Mises
  (`I₁(z)/I₀(z)`), not the `S²` formula `coth z − 1/z`.  Writing the `S²` form
  predicts 0.306 where the kernel gives 0.445 (closed form 0.429).  The
  dimension that fixes the closed form is the dimension of the *conditioned*
  space (`P − 2` for the direction), not of the sphere — the test now uses
  `P = 4`, where the latitude is `S²` and the elementary form applies, and
  asserts `E[u₁] = c` (pinned) alongside `E[u₂] = s·A(κ s/√2)`.

**4. What the fixed kernel measures.**  `--arm latitude --d0 R` measures
`E[acc | d = d₀]` exactly, for a radius chosen rather than reached.  All cells
20 000–30 000 steps where noted, otherwise 10 000; burn 20 %, thinning 25–75:

| `d₀` (rad) | step | accept | per-draw acc / NLL | BMA acc / NLL | `R̂`(NLL) | `R̂`(loss) |
|---:|---:|---:|---|---:|---:|---:|
| 8.61e-4 (`r₁`) | 4.30e-4 | 90.9 % | 48.80–48.93 % / 1.3135 | 47.66 % / 1.3133 | 0.997 | 0.998 |
| 2.30e-3 (basin) | 1.15e-3 | 75.3 % | 49.01–49.15 % / 1.3128 | 46.48 % / 1.3124 | 1.002 | 1.003 |
| 1.00e-2 | 3.00e-3 | 35.0 % | 47.30–47.54 % / 1.3093 | 48.44 % / 1.3045 | 1.003 | 1.021 |
| 1.50e-2 | 3.00e-3 | 33.7 % | 46.21–47.38 % / 1.3151 | 48.44 % / 1.3090 | 1.008 | 1.010 |
| 4.00e-2 (4 ch, 40 k, thin 400) | 3.00e-3 | 35.4 % | 43.04–44.12 % / 1.3648 | 44.92 % / 1.3506 | **1.003** | **1.005** |
| 7.90e-2 (4 ch, 40 k, thin 400) | 3.00e-3 | 43.2 % | 36.48–37.40 % / 1.4583 | 42.97 % / 1.4349 | 1.050 | 1.069 |
| anchor (`d₀ = 0`) | — | — | 49.22 % / 1.3136 | — | — | — |

Five readings, in order of how much they change the document:

1. **These are the first *converged* posterior numbers in the project.**  Every
   earlier conditional (46.4 %, 48.8 %, 49.8 %) had `R̂ ≥ 1.5` or was an
   importance-sampling estimate; these are exact-MH chains with `R̂(NLL) ≤ 1.01`
   at every radius through 4e-2 and 1.05 at 7.9e-2, so the quantities are target
   properties with error bars (see the null table in point 4 for what those last
   two digits can and cannot certify).
2. **The conditional predictive is flat at 46.5–48.4 % for `d₀ ≤ 10⁻² rad`, and
   it agrees with the ball-IS sweep** (48.8–49.0 % at 2.3e-3, 48.4 % at 5e-3).
   Two independent methods — self-normalized IS and an exact chain — now agree on
   the same functional, which is the check the whole "conditional, not
   predictive" reading needed.
3. **At 1e-2 rad the conditional mean predicts *better* than the anchor**
   (NLL 1.3045 against 1.3136).  The point estimate the whole study is anchored
   at is not the best predictor even within a shell the target still has mass in.
4. **Where it falls off: between 1.5e-2 and 4e-2 rad.**  Per-draw NLL by radius
   is 1.3128 (2.3e-3), 1.3093 (1e-2), 1.3151 (1.5e-2), **1.3648 (4e-2)**, 1.4572
   (7.9e-2) — flat to 1.5e-2 and then rising 0.05 nats by 4e-2 and 0.14 by 7.9e-2,
   with accuracy 49 → 47 → 43 → 37 %.  The falloff is not a kernel artefact: it
   is the conditional mean leaving the region where the data informs the
   direction, and it coincides with `N·medΔL` crossing ~100–200 nats (78 at
   2e-2, 220 at 4e-2).  Acceptance tracks the potential's transverse variation,
   not the sphere: 91 % at `r₁`, 75 % at 2.7 `r₁`, 35 % at 10⁻², 36 % at 4e-2,
   44 % at 7.9e-2 — the last rising again because the far shell is flat
   (§6.5.6's "acceptance is a property of position").

   **The outer two cells were re-run at a gap wide enough to test the thinning
   story, and the story holds — for one of them.**  The first pass ran them
   thinned at 50–75 steps while the shell's relaxation time along the latitude is
   `(sin d₀/r)² = 178` steps at 4e-2 and `692` at 7.9e-2, i.e. the kept draws were
   75–90 % correlated; `scripts/pcnl-rhat-null` turns that into a prediction for a
   chain that does nothing wrong (stationary AR(1) with `ρ = exp(−thin/τ)`, the
   example's own split formula, 3 000 replicates):

   | cell | `τ` | thin | `ρ` | null `E[R̂]` | null 95 % | null 99 % | observed |
   |---|---:|---:|---:|---:|---|---:|---:|
   | 4e-2, 2 ch / 20 k | 178 | 50 | 0.755 | 1.019 | [0.998, 1.066] | 1.081 | 1.052 — inside |
   | 4e-2, 4 ch / 40 k | 178 | **400** | 0.105 | 1.003 | [0.991, 1.023] | 1.029 | **1.003 — at the null's mean** |
   | 7.9e-2, 2 ch / 30 k | 692 | 75 | 0.897 | 1.056 | [1.001, 1.183] | 1.229 | 1.140 — inside |
   | 7.9e-2, 4 ch / 40 k | 692 | **400** | 0.561 | 1.033 | [0.998, 1.089] | 1.103 | 1.050 — inside |

   So at 4e-2 the wide gap **settles it**: `R̂(NLL)` falls 1.052 → **1.003**,
   exactly the null's expectation for the new thinning, while the BMA is
   unchanged (44.92 % / 1.3506 against 44.92 % / 1.3519) — the old 1.052 was
   thinning, and the cell is converged.  At 7.9e-2 the same treatment only gets
   to 1.050, because `τ = 692` means `thin 400` still leaves `ρ = 0.56`; the null
   for *that* thinning expects 1.033 with a 95 % band up to 1.089, so 1.050 is
   again unindictable and again underdetermined.  Widening the gap further does
   not help: 40 000 steps at `τ = 692` is only ~58 effectively independent draws
   per chain, and at 58 draws the split-`R̂` cannot resolve better than ~1.05
   whatever the thinning (`thin 1 200` leaves 26 draws per half, whose own null
   spans [0.974, 1.093]).  The honest form of that row is "`R̂` 1.05, consistent
   with stationarity but unable to certify it; settling it needs ~200 k steps,
   not more thinning".

   **The obvious cheaper lever — a wider jump radius — was tested and refuted.**
   Since `τ = (sin d₀/r)²`, tripling `r` to `1e-2` should cut the relaxation to
   62 steps.  It does not, because acceptance is the gate: on the same shell and
   the same seed budget (4 chains, 40 k steps, thin 400)

   | `r` (rad) | acceptance | `τ_eff = (sin d₀)²/(a·r²)` | BMA acc / NLL | `R̂`(NLL) / `R̂`(loss) | null for `τ_eff` |
   |---:|---:|---:|---|---|---|
   | 3e-3 | 43.2 % | 692 | 42.97 % / 1.4349 | 1.050 / 1.069 | E 1.033, 99 % 1.103 |
   | 1e-2 | 2.0–2.7 % | **2 661** | 41.41 % / 1.4370 | 1.115 / 1.444 | E 1.161, 95 % [1.029, 1.372] |

   Twenty times fewer moves accepted, so the *effective* diffusion step is
   `r√a ≈ 1.5e-3` — smaller than the `3e-3` that already worked — and
   `τ_eff = (sin d₀)²/(a·r²)` is 4× worse, not 11× better.  What the run reports
   is exactly what that null expects: `R̂(NLL)` 1.115 sits *below* its mean of
   1.161 and `R̂(loss)` 1.444 at its 99th percentile, i.e. no evidence of
   disagreement between chains, only of a walk that barely moves.  Acceptance is
   not even stationary over the run — 8.4 % over the first 2 000 steps, 2.0–
   2.7 % over 40 000 — so a jump this wide is not a diffusive step on this shell
   at all.  The radius has an optimum near where acceptance is ~40 %, and past
   that the only lever left is steps: ~200 k at `r = 3e-3`, whose ~290
   independent draws per chain would put the null's 95 % band inside ±0.02.  What the re-run *did* settle is the BMA, which is what the
   row is quoted for: 4 chains average to **42.97 % / 1.4349** where the 2-chain
   run gave 37.89 % / 1.4410 — the per-chain accuracies are the same (36.5–37.4 %
   both times), so the 5-point gain is the averaging over four chains, and the
   2-chain number was an under-averaged BMA rather than a property of the shell.
   Every radius `d₀ ≤ 1.5e-2` remains cleanly converged.

**5. Why the exact conditionals are a tail, not the predictive.**  Everything in
(4) measures `E[· | d ≤ r]` or `E[· | d = r]`; what the document had never had is
the law that says **which** `r`, or which directions, the posterior itself prefers.
Both come from one inequality — the likelihood must pay for Haar volume, and it
cannot — read once along the radius and once across the tangent space.

*The radial coordinate.*  With Haar on `S^{P−1}`, geodesic polar coordinates give

```text
p(d) ∝ exp(−N·L(d)) · (sin d)^{P−2},      mode:  N·L′(d)·d = P − 2   (d ≪ 1)
```

and a log-log fit of the sweep's `N·medΔL` column gives `N·L ≈ c·d^β`, so
`d* = ((P−2)/(cβ))^{1/β}` — printed by the example as `posterior scale`.

**The fit is not the result: the fit is grid-dependent, and the grid decides.**
Three sweeps of the *same* target over three ranges give

| grid (rad) | `c` | `β` | `d*` |
|---|---:|---:|---:|
| 1e-4 – 1e-2 | 2 200 | 1.05 | 6.5 rad |
| 1e-2 – 8e-2 | 40 840 | 1.63 | 0.43 rad |
| 2e-2 – 4e-1 | 793 | 0.39 | 2.9 × 10⁴ rad |

i.e. seven decades of answer from one function, because the local exponent moves
from 0.4 to 1.6 across the range and the mode equation extrapolates it.  The
quantity to quote is therefore the **measured inequality**, which needs no fit —
the mode exists where `β·N·L(d)` reaches `P − 2 = 16 670`:

| d (rad) | 1e-4 | 1e-3 | 1e-2 | 2e-2 | 4e-2 | 1e-1 | 2e-1 | 4e-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `N·medΔL` | 0.16 | 1.35 | 24.7 | 78 | 220 | **606** | 606 | 302 |
| test acc | 49.2 % | 48.8 % | 46.8 % | 43.4 % | 31.3 % | 14.5 % | 16.8 % | 11.7 % |

The largest value measured anywhere inside 0.4 rad is `606`; even with the
steepest local `β = 1.6` that is `970` against `16 670`, **a factor 17 short**.  So
the density is still rising at 0.4 rad, where the accuracy has already reached
the base rate (11.7 %) — **there is no interior mode, and `π`'s predictive is the
base rate.**

**Extending the profile to the equator turns "inside 0.4 rad" into the whole
sphere.**  One grid (`2e-2, 4e-2, 6e-2, 1e-1, … , 1.4, π/2`, one probe set so
every `ΔL` is measured against the same minimum) measures `N·medΔL` *and* the
left-hand side of the exact mode equation, `N·L′(d)·tan d`, which is what a mode
needs to equal `P − 2` (the `d ≪ 1` form replaces `tan d` by `d`):

| d (rad) | 2e-2 | 4e-2 | 6e-2 | 1e-1 | 2e-1 | 3e-1 | 4e-1 | 6e-1 | 1.0 | 1.4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N·medΔL` | 78 | 220 | **515** | **606** | 296 | 370 | 315 | 217 | 209 | 254 |
| test acc | 43.4 % | 31.3 % | 27.3 % | 14.5 % | 11.7 % | 11.7 % | 11.7 % | 11.7 % | 11.7 % | 11.7 % |
| `N·L′·tan d` | — | 282 | **888** | 227 | −627 | 229 | −235 | −333 | −47 | **1 444** |

Two things happen out there.  `N·medΔL` peaks at `1e-1` rad and then *falls*:
every row past `2e-1` is at the base rate (11.7 %) and the loss is flat, so the
radial likelihood stops growing.  The ceiling is the base-rate gap
`N·ΔL ≤ 512 × 3.73 ≈ 1 910`, which bounds this column — not the grid.  And the
mode residual `N·L′·tan d` never comes near `P − 2 = 16 670`: its largest value
in the informative band (`d ≤ 0.14` rad, where accuracy still beats the base
rate) is `888` at `6e-2` rad, a factor **19** short; its largest value anywhere
is `1 444` at `1.4` rad, and that is a finite-difference artefact — the *median*
wanders `203.8 → 253.6` over `0.2` rad where the loss is flat, multiplied by
`tan 1.4 = 5.8` — still a factor **11.5** short.  Since
`d ln p/dd = (P−2)cot d − N·L′(d) > 0` is exactly `N·L′·tan d < P − 2`, the
density **increases all the way to the equator**.  The mode is not just outside
0.4 rad: it is nowhere on the sphere, and the mass that decides the predictive
sits at `d ≈ π/2`, where the accuracy is the base rate.  That is now a
measurement over the entire sphere rather than an extrapolation from the
informative band.

(The grid has one trap: a row must satisfy `d < π/2`, because `tan` changes sign
just past the pole.  `1.5708` is past it, and its residual is the sign-flipped
`−3.2 × 10⁵` — a row, not a result.  `pcnl::radial` pins that boundary: past the
pole both `tan` and `cot` flip, so the residual goes negative and the volume
weight starts *shrinking*, which is why the density peaks at `π/2` and falls.)
The log-log fit on the same grid returns `β = 0.02`, `d* = 4 × 10¹⁵³` rad: the
third grid in a row to give a different nonsense answer, and the reason the
document quotes the inequality, not the fit.

**The far rows are a weak instrument, so they were re-measured on a denser
grid.**  A 128-probe sweep at `d ≳ 0.1` rad has ESS 1.0, so `N·medΔL` there is a
median over directions the weights do not average.  Eleven rows spanning
`0.1–0.4` rad (same probe budget) read
`563, 614, 556, 538, 362, 606, 376, 380, 274, 204, 350` — a ±30 % scatter about
~400 with no trend — and the residual column swings between `−1 602` and
`+2 469` with no structure.  Two consequences, both going the same way:

* the largest residual anywhere on the denser grid is **2 469** at `0.2` rad,
  still a factor **6.8** short of `P − 2 = 16 670`.  The sparse grid's `1 444`
  was therefore not an upper bound and not a mode: it was one draw from that
  scatter.  What survives every grid is the inequality,
  `N·L′(d)·tan d ≪ P − 2` on `0 ≤ d ≤ π/2`;
* the log-log fit on this grid returns **`NaN`**, because `N·medΔL` is no longer
  monotone in `d`.  Third grid, third different answer — and the first that is
  not even a number, which is the clearest statement yet that the fit is not
  describing the target.

The accuracy column shows the same scatter: two rows (0.1 and 0.2 rad) read
16.8 % where their neighbours read 11.7 %, at ESS 1.0 — probe-set noise, not
target structure.  One by-product is worth flagging because it is easy to
misread: on this grid the one-nat radius is `r₁ = 4.21e-3` rad rather than
`8.61e-4`, both being the max-ESS width scaled by `1/√(N·E[ΔL])`.  `r₁` is a
property of the grid, so no `r₁`-relative claim (`13.33 r₁`, `26.33 r₁`) should be
read without the grid that produced it.  That is the sharp version of "prior-dominated": not a sampler
failing to find a mode, but a target with no mode inside the range where the
model is informative at all.  It also explains the free walk of the 10 k and
50 k runs: `N·L′·d ≪ P − 2` over the whole reachable range, so there is no
restoring force and `d` grows as `√t` without turning around.  (The `d*` figures
the example prints remain useful where the grid brackets the mode — it does for
the two smaller models below — but for this target the inequality is the claim.)

**The same inequality makes a prediction: shrink `P` and the mode comes inside.**
Same data, same `N = 512`, only the model width changed (`--shapes d,ff,layers`),
each row refitting `c, β` on its own grid:

| `P` | shapes | `c` | `β` | `d*` | `r₁` | anchor acc / NLL | best conditional in the grid |
|---:|---|---:|---:|---:|---:|---|---|
| 16 672 | 32,64,2 | 2 200 | 1.05 | 6.5 rad | 8.61e-4 | 49.22 % / 1.3136 | 49.22 % (anchor) |
| 4 824 | 24,48,1 | 3 664 | 0.86 | 1.66 rad | 9.62e-5 | 33.59 % / 1.4626 | **37.75 %** / 1.4647 (1e-3) |
| 2 192 | 16,32,1 | 10 971 | 1.14 | **0.217 rad** | 3.66e-4 | 26.17 % / 1.6651 | 26.17 % (anchor) |
| 584 | 8,16,1 | 5 979 | 0.95 | **0.0905 rad** | 1.37e-4 | 20.70 % / 1.6499 | 28.52 % / 1.7223 (3e-3, ESS 1.0) |

Three things hold and one does not.  **Held:** `d*` falls monotonically with `P`
(6.5 → 1.66 → 0.22 → 0.09 rad), the interior-mode threshold is crossed between
`P = 4 824` (`d* = 1.66 rad`, still outside the sphere's equator) and `P = 2 192`
(`d* = 0.22 rad`, inside), and the two smallest models are the ones whose grid
*brackets* the mode, so their `d*` is a measurement rather than an extrapolation.
**Not held:** the naive form `d* ∝ P` (which assumes `c` is `P`-independent, as a
unit tangent's directional derivative `|∇L|/√P` with `|∇L| ∝ √P` suggests).  Measured
`c` grows 1.7× then 3× as `P` drops 3.5× and 2.2×, i.e. a smaller model is much
more sensitive *per direction*, so `d*` falls faster than linearly: 6.5→0.22 is
30× for a 7.6× drop in `P`.  The scaling law is therefore structural (there is a
mode when `β·N·L(d) = P − 2`, and shrinking `P` finds it) but not parametric in
`P` alone — `c(P)` has to be measured, and `c` grows as the model shrinks.

One more reading from the same table: at `P = 4 824` the best conditional in the
grid (37.75 % at 1e-3 rad) beats the anchor (33.59 %) by 4 points, while at
`P = 16 672` no conditional beats the anchor at all until 4e-2, where it is 5
points *below*.  That is the same fact as the mode: with fewer parameters the
posterior's mass lies where the data has actually informed the direction.

**The mode radius is not the informative radius — measured directly.**  `d*` is
where the *density* peaks; it is not where the *model* still knows anything.  The
two smallest models let that be checked rather than argued: run the latitude
shell at each `d*` (`--no-sweep --chains 4`, 20 000 steps, burn 4 000, thin 50,
`--step 0.05` and `0.02`, four chains so `R̂` is defined):

| `P` | shell `d` | acceptance | per-chain acc | BMA acc / NLL | anchor acc / NLL | split `R̂` |
|---:|---:|---:|---|---|---|---|
| 2 192 | 0.217 rad | 0.6–1.7 % | 11.76 / 11.85 / 12.27 / 10.71 % | **11.72 % / 2.0932** | 26.17 % / 1.6651 | 2.44 / 3.29 |
| 584 | 0.0905 rad | 0.5–0.7 % | 22.91 / 24.48 / 23.53 / 15.14 % | **24.22 % / 1.8075** | 20.70 % / 1.6499 | 7.18 / 7.26 |

The `P = 2 192` answer is unambiguous: at the radius the first grid's fit called
its mode, the shell's predictive is **exactly the base rate** (11.72 %), and every chain agrees — 11.7
% is not a mixing accident, it is where the loss has gone flat.  So the mode sits
*outside* the informative band for that model, and "the density peaks here" says
nothing about accuracy.  The `P = 584` shell is the interesting middle case:
accuracy rises 3.5 points over the anchor (24.22 % vs 20.70 %) while NLL worsens
(1.8075 vs 1.6499) — the averaging is sharpening the vote without improving the
calibration — and the split `R̂` of 7.2 says the four chains have *not* mixed:
one of them sits at 15.14 % while the other three are at 22.9–24.5 %.  An
acceptance of 0.5–0.7 % is the cause: at `β = 0.95` on a shell 0.09 rad out, the
proposal almost never moves anywhere useful, so 20 k steps is a short walk and
`R̂` is reporting it.  This is a **failure**, not a thinning artefact (contrast
the `--thin 400` nulls below, where the same statistic at 1.05–1.14 is
consistent with stationarity), and it means the `P = 584` row of the scaling
table should be read as *a shell that is more accurate than its anchor but not
certified*.

Taken together: the inequality predicts where the mode goes as `P` shrinks, the
small models confirm it, and the shells at `d*` show that landing on the mode
buys nothing — at `P = 2 192` it is the base rate, and at `P = 584` it is a
4-point accuracy gain carried by unmixed chains.  The quantity that decides the
predictive is the mass-weighted loss over the whole sphere, not the location of
the peak.

**Where those models are informative, for comparison** (same sweep, grids ending
at `2.2 × d*`):

| `P` | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 9e-2 | 2.2e-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 192 acc (ESS) | 26.17 (126) | 26.12 (114) | 25.09 (42) | 22.45 (3.5) | 25.00 (1.0) | 9.38 (1.0) | — | 11.72 (1.0) |
| 584 acc (ESS) | 20.83 (123) | 20.76 (86) | 20.81 (5.7) | 28.52 (1.0) | 16.02 (1.0) | 13.28 (1.0) | 11.33 (1.0) | — |

The first three columns are the instrument working (ESS 42–126): `P = 2 192`
sits at its anchor's 26.2 % out to `1e-3` rad and `P = 584` at 20.8 %, i.e. both
models are informative only inside `~10⁻³` rad.  Everything from `3e-3` out has
**ESS 1.0** — one importance weight carries the estimate — so those accuracy
figures (9.38 %, 28.52 %, 11.33 %) are single-probe numbers, not measurements,
and they are exactly why the latitude chains exist: the sweep cannot say what
lives at 0.09 rad, and the chain can.  Read that way the two instruments agree
where both work (`≤ 1e-3` rad) and only the chain speaks beyond it.  The mode
therefore lies outside the informative band under *either* grid: the fit's `d*`
is 0.217 rad from the original grid and 0.52 rad from this one (`P = 2 192`),
0.0905 and 0.0864 rad (`P = 584`) — grid-dependent as always, but every value is
`10–50×` past the radius where accuracy is still above the base rate.

The same equation fixes what would put the mode *inside*.  At `r = 2.3e-3` it
asks for

```text
N·L′(2.3e-3) ≥ (P − 2)/2.3e-3 = 7.25 × 10⁶ nats/rad,
```

against the measured `1.10 × 10³` at `N = 512` — a shortfall of `6.6 × 10³`, i.e.
`N ≈ 3.4 × 10⁶` examples.  **That figure is withdrawn in the next subsection**: it
is a property of how far the anchor had converged, not of the target, and at
*fixed* `N` it moves by a factor 55.  The inequality itself survives, because it
is the outer rows — `d ≥ 10⁻²` rad — that carry the no-mode conclusion, and those
are the rows the anchor control leaves alone.

**Data scaling, measured instead of extrapolated.**  The `N ≈ 3.4e6` claim rests on
one assumption, `N·L′ ∝ N`, taken from a single `N = 512` run.  Two experiments
test it directly: change `N` at fixed anchor schedule, then change the anchor
schedule at fixed `N`.

The paired sweep first (`--n-train 512` vs `2048`, same five widths, same 128-probe
stream, same `u_Adam` task):

| `r` (rad) | `N·medΔL`, `N = 512` | `N·medΔL`, `N = 2048` | ratio | per-example `medΔL` 512 → 2048 |
|---:|---:|---:|---:|---:|
| 1.0e-3 | 1.55 | 19.20 | 12.4 | 0.0030 → 0.0094 |
| 2.3e-3 | 3.40 | 39.95 | 11.8 | 0.0066 → 0.0195 |
| 5.0e-3 | 4.97 | 79.09 | 15.9 | 0.0097 → 0.0386 |
| 1.0e-2 | 17.28 | 160.05 | 9.3 | 0.034 → 0.078 |
| 4.0e-2 | 229.75 | 895.33 | 3.9 | 0.449 → 0.437 |

Four times the data buys `≈12–16×` the loss at small widths and `3.9×` at the
outermost one, i.e. superlinear inside `5e-3` rad and linear outside `1e-2`.  But
the anchor moved in the same step — train loss 1.2532 → 1.1887, test accuracy
49.22 % → 44.92 % — so "more data" and "deeper anchor" are confounded in that
column, and only the per-example column could have separated them.

So hold `N = 512` and train the anchor 4× and 8× longer (`--adam-steps`, 4000 and
8000, same widths, same probes):

| anchor steps | train loss | test acc / NLL | 1e-3 | 2.3e-3 | 5e-3 | 1e-2 | 4e-2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 000 | 1.2532 | 49.22 % / 1.3136 | 1.55 | 3.40 | 4.97 | 17.28 | 229.75 |
| 4 000 | 1.1136 | 50.00 % / 1.2125 | 14.39 | 38.62 | 133.37 | 253.74 | 460.77 |
| 8 000 | 1.0962 | 57.81 % / 1.1804 | 44.62 | 136.87 | 351.87 | 526.72 | 290.86 |
| *for comparison:* `N = 2048`, 1 000 steps | 1.1887 | 44.92 % / 1.2266 | 19.20 | 39.95 | 79.09 | 160.05 | 895.33 |

At the basin radius `2.3e-3` the same 512 examples give 3.40, 38.62 and 136.87 for
1 000, 4 000 and 8 000 anchor steps: **the 8 000-step anchor exceeds what 4× the
data achieved**, and 4 000 steps lands on it (38.62 vs 39.95, 3 %).  The printed data
requirement follows the anchor, not `N`:

```text
N = 512, adam 1 000  →  N ≈ 3.4 × 10⁶
N = 512, adam 4 000  →  N ≈ 2.0 × 10⁵
N = 512, adam 8 000  →  N ≈ 6.2 × 10⁴
```

That is a factor 55 at *identical* data, and a factor 1.4 between 4 000 and 8 000
steps.  Since test accuracy is still rising at 57.81 %, none of these is the
converged value — `6.2e4` is an upper bound that keeps falling with training.  The
honest statement is therefore: **`N·L′` at small `d` is set by the anchor's
convergence depth, so any data requirement computed from it is a statement about
Adam, not about `π`.**  Withdrawing `3.4e6` is not a correction of the arithmetic;
the arithmetic was right and the quantity it measured was the wrong one.

What the control leaves intact is the outer range.  There `4×` data gives `3.9×`
the loss — linear, as assumed — and 8× anchor training does not reproduce that
(290.86, *below* the 4 000-step 460.77, so this row has `±2×` run-to-run scatter of
its own).  The exact mode equation is evaluated on those rows, and across the three
anchors its largest measured value is 283, 461 and 980 (in `N·maxΔL`) against
`P − 2 = 16 670` — short by 17–60× every time.  The **no-interior-mode conclusion at
`P = 16 672` stands on all three anchors**.  The fitted `d*` (1.06, 0.744 and 89.5
rad) does not: the 89.5 is the fit degenerating when the last row drops below the
one before it, and all three are grid-and-anchor artefacts of the same small-`d`
columns, quoted only as "≫ the 2.3e-3 basin radius".

**The one model that does grow an interior mode.**  The same `N` ladder at
`P = 584` (`8,16,1`, seven widths, `N = 512 / 2 048 / 8 192`, 64 probes):

| `r` | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 9e-2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `N = 512` | 0.53 | 1.13 | 12.60 | 111.28 | 223.14 | 184.40 | 172.76 |
| `N = 2 048` | 0.44 | 1.72 | 13.78 | 131.44 | 672.49 | 1 109.99 | 689.42 |
| `N = 8 192` | 0.22 | 1.43 | 16.92 | 140.87 | 1 192.94 | 4 039.83 | 3 811.93 |
| γ (8× `N`) | −0.38 | +0.11 | +0.14 | +0.11 | +0.80 | +1.32 | +1.32 |

Three regimes, and the first one is a warning about the control: at `1e-4` the
loss *falls* with `N` (`γ < 0`), so there the measurement is not the target's own
curvature at all but noise in a ball narrower than the anchor's own basin.
Between `3e-4` and `3e-3` the growth saturates (`γ ≈ 0.11` for 8× data) — this is
the anchor-limited band, and it is why the `P = 16 672` number above could be moved
by training alone.  Past `10⁻²` growth is superlinear (`γ = 0.80–1.32`).

The consequence is the project's first interior mode: `N·maxΔL` reaches 160 → 773
→ 4 272 against `P − 2 = 582`, and the fit's `d*` falls 0.228 → 0.0195 → 0.00857
rad.  Two caveats kept that from being a certificate on its own.  First, **ESS is
1.0 on every row from `1e-2` out**, so the crossing is carried by one importance
weight per probe; the `P = 584` anchor control was reassuring (`1e-2` and `3e-2`
move only ±5 %, 223→193 and 184→177, unlike the small-`d` rows) but it is a
different statistic.  Second, `d* = 0.00857` rad sits *inside* the band where the
sweep is blind, so only the latitude shell can check it.  So the chain was run.

**The mode radius, sampled — and certified.**  Latitude shell at `d₀ = 0.00857`
rad, `N = 8 192`, 4 chains, 20 000 steps, burn 4 000, thin 20, `r = 2e-3`:

```text
acceptance 0.5–0.8 %      per-chain acc 24.10 / 26.31 / 24.98 / 24.89 %
per-chain NLL 1.6009 / 1.6006 / 1.6006 / 1.6006     BMA 24.61 % / 1.6005
split-R̂(NLL) 1.0498   split-R̂(train L) 1.3654
anchor (N = 8 192): 21.88 % / 1.5988      base rate 11.72 %
```

The null at the acceptance-corrected radius `r_eff = 2e-3·√0.0073 = 1.70e-4`
(`τ = (sin d₀/r_eff)² = 2 541` steps, `ρ = e^{−20/2 541} = 0.992`, 3 000
replicates) expects `E[R̂] = 1.328`, 95 % `[1.088, 1.715]`, 99 % 1.812.  The
observed `R̂(NLL)` 1.0498 is **below the null's mean** — better mixed than a
stationary AR(1) with that thinning — and `R̂(loss)` 1.3654 lands on the null's
mean.  This is the first *converged* chain measurement in the project at a mode
radius, and it certifies the sweep's extrapolation: the interior mode at
`P = 584`, `N = 8 192` is real, not a single-probe artefact.

What it is worth is the familiar answer.  At the density's peak the predictive is
24.61 % / 1.6005 against the anchor's 21.88 % / 1.5988: **+2.7 points of accuracy
and a NLL 0.0017 *worse*** — the peak of `p(d)` is not where calibration improves,
which `P = 2 192` reached in its extreme form (base rate exactly) and this model
reaches in a mild one.  Note also that the far 0.0905 rad shell reported a higher
BMA accuracy, 44.92 %, on the same per-chain numbers (23.9–25.5 %): that gap is the
averaging gain from *diversity* across a wide shell, not evidence that accuracy
grows with radius.  The tight shell here holds near-identical draws, so its BMA
is 24.61 % and buys nothing from averaging.  Thinned 20 and `R̂`-certified, the
statement is: the mode buys accuracy over the anchor, buys 1.5 % in NLL at best,
and the wide shell's 46 % belongs to the ensemble rather than to the posterior's
peak.

**The same protocol one rung down, `N = 2 048`.**  `d* = 0.0195` rad, 4 chains,
20 000 steps, burn 4 000, thin 20, `r = 2e-3`, acceptance 0.9–1.2 %:

```text
per-chain acc 16.91 / 21.58 / 17.76 / 18.45 %   NLL 1.7869 / 1.7840 / 1.7869 / 1.7910
BMA 19.53 % / 1.7838      anchor (N = 2 048): 22.66 % / 1.7467
split-R̂(NLL) 1.3702   split-R̂(train L) 1.5759
null at r_eff = 2.08e-4: E[R̂] 1.929, 95 % [1.266, 2.978]
```

Here the mode is **worse than its own anchor** on both metrics — −3.1 accuracy
points and +0.037 NLL.  Its `R̂` is inside the null, but that null is loose
(`τ = 8 788` steps, `ρ = 0.998`, 95 % spans 1.27–2.98), so the honest reading is
"no evidence of non-convergence", not the certification the `N = 8 192` run
earned.

Pairing those two rows suggested a trend — the mode's gap to its anchor closing
as `N` grows — and that reading does not survive the third rung.

**A third rung, `N = 16 384`, and the anchor's learning rate.**  Extrapolating the
ladder's fit (`d* ∝ N^−0.6`) predicted 3.76e-3 rad; the sweep at `N = 16 384`
(64 probes, same protocol) put it at **3.02e-3**, so the inward motion continues
(0.228 → 0.0195 → 0.00857 → 0.00302 rad).  Sampling that shell (4 chains, 20 k
steps, burn 4 000, thin 10, `r = 1e-3`, acceptance 0.4–0.5 %):

```text
per-chain acc 22.98 / 23.91 / 24.39 / 24.26 %   NLL 1.6094 / 1.6099 / 1.6093 / 1.6096
BMA 23.44 % / 1.6095      anchor (N = 16 384): 26.95 % / 1.6086
split-R̂(NLL) 1.2825   split-R̂(train L) 1.3614
null at r_eff = 6.7e-5: E[R̂] 1.272, 95 % [1.071, 1.602]  ->  inside, tight null
```

The NLL gap to the anchor is now 0.0009 — the smallest of the ladder — but the
accuracy gap is **−3.5 points**, the wrong sign for the trend.  So the NLL part of
the trend is real and the accuracy part is not: over three rungs the NLL gap falls
monotonically (0.0371 → 0.0017 → 0.0009) while the accuracy gap goes −3.1 → +2.7 →
−3.5.  With three rungs instead of two, "the mode buys accuracy" is withdrawn.

Before drawing that conclusion, the anchors themselves have to be examined, and
they do not hold up.  `adam` is full-batch with a **fixed** step `lr = 0.05` and no
schedule, so the step is an inverse length, not a dimensionless constant:

| `N` | steps | lr | train L | acc % | NLL |
|---:|---:|---:|---:|---:|---:|
| 2 048 | 1 000 | 0.05 | 1.7446 | 22.66 | 1.7467 |
| 2 048 | 2 000 | 0.05 | 1.7476 | 20.70 | 1.7485 |
| 2 048 | 2 000 | **0.005** | 1.7415 | **28.12** | **1.7318** |
| 8 192 | 1 000 | 0.05 | 1.6058 | 21.88 | 1.5988 |
| 8 192 | 8 000 | 0.05 | 1.6157 | 24.61 | 1.6089 |

Doubling the steps at `N = 2 048` *raises* the train loss (1.7446 → 1.7476): the
trajectory is oscillating around the minimum, not descending into it.  Ten times
fewer `lr` at the same step count cuts it to 1.7415 and buys 7.4 accuracy points and
0.017 NLL.  At `N = 8 192`, 8× the steps leaves train loss above where 1 000 steps
had it.  This is consistent with the `P = 16 672`, `N = 512` control, where the same
`lr` improved monotonically (1.2532 → 1.1136 → 1.0962): the default is calibrated to
`N = 512` and stops being stable by `N = 2 048`.  `--adam-lr` now exists
(`pcnl/examples/basin_posterior.rs`) to move it.

The consequence is that **`d*` is a property of the anchor, not of the model**.  Two
re-anchorings move it without moving the physics: `N = 2 048` gives 0.0195 →
0.0130 rad as the anchor goes 22.66 → 28.12 %, and `N = 8 192` gives 0.00857 →
0.00438 rad at a (itself not better-trained) 8 000-step anchor.  The interior mode's
*existence* survives all of this; its *radius* moves by ×1.5–2.

**The re-anchored `N = 2 048` rung, sampled.**  Sweeping with `--adam-lr 0.005
--adam-steps 2000` puts `d* = 1.30e-2` and reports the predictive there as **11.72 % /
2.1860 — the base rate**.  The chain at 1.30e-2 (same four-chain protocol, `r = 2e-3`,
acceptance 0.6–0.9 %) says otherwise:

```text
per-chain acc 27.30 / 27.56 / 25.54 / 24.94 %   NLL 1.7807 / 1.7575 / 1.7604 / 1.7592
BMA 29.30 % / 1.7611      anchor (2000 @ lr 0.005): 28.12 % / 1.7318
split-R̂(NLL) 1.7862   null E[R̂] 1.659, 95 % [1.192, 2.405]  ->  inside, loose
```

So the chain finds 29.30 % where the sweep found the base rate: a **17.5-point
disagreement**, because the 1e-2 row's ESS is 1.0.  That is the sharpest statement
of the section's recurring theme — an ESS-1 row is not a noisy measurement, it is
one probe, and it can be wrong by more than a base rate.  Every claim in this
document that rests on an ESS-1 row (including "the predictive collapses to the base
rate by 1e-2 rad" in the earlier sweeps) is provisional until a chain confirms it;
here the chain refuted it.

Against its own well-trained anchor the mode is +1.2 accuracy points and 0.029 NLL
*worse* — so on the one rung with a defensible anchor, the mode is a wash on
accuracy and a small loss on calibration.

**The best-measured `N = 2 048` anchor, sampled.**  The 0.005 anchor above is
well-trained but 0.008 nat above the best cell of the `lr` grid, so the rung was run
once more against that one (1 000 steps at `lr = 0.0125`: train 1.7382, 31.25 %,
1.7270).  There `r₁ = 1.02e-4` rad — not the stale 8.61e-4 the `--no-sweep` line
prints — and `d* = 1.45e-2` rad.  The shell at 1.45e-2 (4 chains, 20 k steps, burn
4 000, thin 20, `r = 2e-3`, acceptance 0.8–0.9 %):

```text
per-chain acc 27.76 / 28.01 / 25.89 / 25.76 %   NLL 1.7497 / 1.7490 / 1.7567 / 1.7486
BMA 26.56 % / 1.7493      anchor: 31.25 % / 1.7270
split-R̂(NLL) 1.6119   split-R̂(train L) 1.5784
null at r_eff = 1.82e-4: E[R̂] 1.724, 95 % [1.209, 2.555], 99 % 2.772  ->  inside, loose
```

That is **−4.7 accuracy points and +0.022 NLL**, the largest accuracy deficit of the
five rungs — and it is measured against the best anchor this `N` can reach, so it is
the fairest row in the table, not the least fair.  The same run also settles the
small-`d` claim for `N = 2 048`: its sweep rows 1e-4 → 31.28 % / 1.7271 and 3e-4 →
31.38 % / 1.7278 now sit *on* the anchor's 31.25 % / 1.7270 (ΔNLL +0.0001 and
+0.0008, Δacc +0.03 and +0.13 points, the latter below one test example).  The
0.001–0.004 nat gain the 0.005 anchor reported at those widths was therefore the
anchor's own shortfall, drawn back out of `π` by a slightly wider draw; with an
anchor that has no shortfall the gain disappears.  That is the cost argument for the
small-`d` band evaporating, and it is the reason the `N = 8 192` small-`d` chain
below is the decisive test rather than another sweep row.

**Does `lr` scale with `N`?**  Two laws are worth testing on the cells above — `lr ∝
1/N` (a full-batch step carrying a fixed displacement) and `lr ∝ 1/√N` (a fixed noise
ball, what a stochastic step would give).  Applied to the best cell at `N = 2 048`
(0.0125) they predict 0.003 and 0.006 at `N = 8 192`, and 0.0015 and 0.003 at
`N = 16 384`.  Measured anchors (`u_Adam` train mean loss / test acc / test NLL):

| `N` | `lr` @ steps | train L | acc % | NLL |
|---:|---:|---:|---:|---:|
| 2 048 | 0.05 @ 1000 | 1.7446 | 22.66 | 1.7467 |
| 2 048 | 0.025 @ 1000 | 1.7426 | 22.27 | 1.7403 |
| 2 048 | **0.0125 @ 1000** | **1.7382** | 31.25 | 1.7270 |
| 2 048 | 0.006 @ 1000 | 1.7390 | **32.42** | **1.7144** |
| 8 192 | **0.05 @ 1000** | **1.6058** | 21.88 | 1.5988 |
| 8 192 | 0.05 @ 8000 | 1.6157 | **24.61** | 1.6089 |
| 8 192 | 0.0125 @ 1000 | 1.6188 | 21.48 | 1.6107 |
| 8 192 | 0.0125 @ 4000 | 1.6111 | 19.92 | 1.6042 |
| 8 192 | 0.003 @ 2000 | 1.6915 | 21.09 | 1.6705 |
| 16 384 | **0.05 @ 1000** | **1.6119** | 26.95 | 1.6086 |
| 16 384 | 0.0125 @ 4000 | 1.6231 | 26.95 | 1.6204 |

The `1/N` cell is tested at two `N`s and loses at both: 0.003 @ 2000 at `N = 8 192`
is 0.086 nat worse than the default, and 0.0125 @ 4000 at `N = 16 384` (matched
travel, 4× the steps) is 0.011 nat worse.  The `1/√N` cell (0.006) was not run at
`N = 8 192`, but its neighbour 0.0125 is already 0.013 nat worse there, and at
`N = 2 048` the ordering is the opposite: the default 0.05 is 0.006 nat *worse* than
0.0125 and 0.019 NLL worse.  **The sign of the correction flips between `N = 2 048`
and `N ≥ 8 192`, so no single exponent reproduces the best measured anchor at all
three `N`s.**  Two reasons, both visible in the table: each `N` is a different
training set (not a subsample), so these cells compare anchors under different
landscapes; and `d*` moves ×1.5–2 under such changes, which is wider than the loss
differences that separate the cells.

What the grid does establish is the size of the anchor effect, and it is large: a 4×
lower `lr` at fixed steps is −0.0064 nat and +8.6 points at `N = 2 048`, but +0.013
nat and −0.4 points at `N = 8 192`.  Accuracy moves *oppositely* to train loss
within a `N` (at `N = 2 048` the lowest-loss cell, 0.0125, is not the highest-accuracy
one, 0.006; at `N = 8 192` the lowest-loss cell is the default and the highest-accuracy
one is the 8 000-step default), which is the same train/test dissociation the mode
shells show.  The default stays 0.05 — best of five cells at `N = 8 192`, best of two
at `N = 16 384` — and the `u_Adam` line now prints `(lr, steps)` so every run states
the anchor it was measured against (`pcnl/examples/basin_posterior.rs`).

**The ladder as it now stands.**

| `N` | `d*` | anchor acc / NLL | mode shell BMA | Δacc / ΔNLL | `R̂`(NLL) verdict |
|---:|---:|---|---|---|---|
| 2 048 | 0.0195 | 22.66 / 1.7467 | 19.53 / 1.7838 | −3.1 / +0.037 | inside, loose |
| 2 048 | 0.0130 | 28.12 / 1.7318 | 29.30 / 1.7611 | +1.2 / +0.029 | inside, loose |
| 2 048 | 0.0145 | 31.25 / 1.7270 | 26.56 / 1.7493 | −4.7 / +0.022 | inside, loose |
| 8 192 | 0.00857 | 21.88 / 1.5988 | 24.61 / 1.6005 | +2.7 / +0.0017 | **below null mean** |
| 16 384 | 0.00302 | 26.95 / 1.6086 | 23.44 / 1.6095 | −3.5 / +0.0009 | inside, tight |

Read across, only one quantity is monotone: **the NLL gap closes with `N`**, from
0.037 to under 0.001, i.e. the modal conditional's calibration converges onto the
anchor's.  Accuracy does not follow a trend and changes sign twice, and the anchors
are lr-limited to different degrees — and the best-anchored row (the third, at the
best cell of its own `lr` grid) is the *worst* one on accuracy (−4.7), which removes
"the anchor was undertrained" as the explanation for the negative rows.  The one clearly positive row,
`N = 8 192` +2.7, is also measured against that `N`'s best anchor (1 000 steps at the
default, 1.5988), so it is the row least exposed to the anchor explanation — and the
only rung where the mode's accuracy win survives.

**The small-`d` band, sampled with chains (`N = 8 192`).**  The sweep's 1e-4 and 3e-4
rows at this `N` (24.19 % / 1.6076 and 25.48 % / 1.6082, against the 8 000-step
anchor's 24.61 % / 1.6089) are the only place in any sweep where the predictive beats
its own anchor on NLL.  Tested with chains at two jump radii, started at `u_Adam`
(4 chains, 20 k steps, burn 4 000, thin 10, no cap, no latitude — a plain
random-walk Metropolis from the anchor):

| `r` | acceptance | `|d|` at end | growth exp. | per-chain acc | per-chain NLL | BMA acc / NLL | `R̂`(NLL) / (loss) |
|---:|---|---:|---:|---|---|---|---|
| 1.0e-5 | 86.6–87.8 % | 1.25–1.38e-3 | 0.484 | 23.61–25.13 % | 1.6092–1.6101 | 26.56 % / 1.6097 | 1.4336 / 1.7019 |
| 3.0e-5 | 61.0–65.3 % | 3.07–3.55e-3 | 0.509 | 23.38–24.91 % | 1.6107–1.6117 | 26.56 % / 1.6109 | 1.1063 / 1.4548 |

Two things happen, and neither is the sweep's story.  First, **the walk does not stay
at the width it is given**: a fixed-radius jump of 1e-5 rad random-walks out to
`√20000 × 1e-5 = 1.41e-3` rad in 20 k steps (measured 1.25–1.38e-3, growth exponent
0.48, mean pairwise move angle 88.6° — free diffusion), and the 3e-5 walk reaches
3.1–3.5e-3.  So "sampling the 1e-4 row" is not something a chain at these radii does;
it visits an annulus two decades wider, where the mean train loss ends 0.0013 and
0.0031 nat above the anchor (final `train L` 1.6170 / 1.6188 against 1.6157).  Second,
**at the radii the chains actually reach the NLL is worse than the anchor's** (1.6097
and 1.6109 against 1.6089) — the opposite sign to the sweep rows, and against the
*better* 1 000-step anchor (1.5988) worse by 0.011.  Neither walk is converged
radially (the growth exponent is still ½ and `R̂`(loss) is 1.45–1.70), so these are
not certified numbers; what they establish is the direction, on four chains each.

What does replicate is the **ensemble** gain: both runs report BMA 26.56 % from
per-chain 23.4–25.1 %, i.e. +1.7 to +2.9 points of accuracy from averaging 800 draws
spread over 1.3e-3–3.3e-3 rad — the same averaging gain the 0.0905 rad shell at
`N = 8 192` showed (+23 points over its anchor) at 70× smaller radius.  What that
gain is *worth* is the next block's question; NLL gains nothing from it here.  The
sweep rows
that motivated the test are IS probes at a radius the chain does not occupy: at 1e-4
a 1e-4 jump has acceptance `e^{−17.6}` (that row's `N·medΔL`), so no chain of this
length is ever there.

**Does breadth pay?  Four shells at one `N` and one anchor (`N = 2 048`).**  The
ladder's one positive number is an *accuracy* number, and the reading it invited was
"ensemble spread buys accuracy": the 0.00857 rad shell at `N = 8 192` reported
+2.7 points over its anchor, and the 0.0905 rad shell there reported 44.92 % BMA over
21.88 % (uncertified, and never reproduced).  Radius was never varied at fixed `N`
and fixed anchor, so the reading was never tested.  It has now been, at `N = 2 048`
with the best measured anchor (`lr 0.0125`, 1 000 steps: 31.25 % / 1.7270), four
shells, 4 chains each, 10 k steps, burn 2 000, thin 10:

| shell `d₀` | jump `r` | acceptance | BMA acc / NLL | per-chain acc | anchor | `R̂`(NLL) / (loss) | null |
|---:|---:|---|---|---|---|---|---|
| 3.0e-3 | 1e-3 | 0.5–0.8 % | 31.25 / 1.7352 | 29.72–32.24 | 31.25 / 1.7270 | 1.268 / 1.245 | E 1.356, 95 % [1.096, 1.777] |
| 1.0e-2 | 2e-3 | 0.6–0.8 % | 28.12 / 1.7563 | 25.86–29.95 | — | 1.525 / 1.505 | E 1.753, 95 % [1.213, 2.623] |
| 3.0e-2 | 3e-3 | 1.8–2.5 % | 23.83 / 1.7803 | 20.97–25.21 | — | 3.389 / 3.606 | E 2.022, 95–99 % |
| 9.0e-2 | 1e-2 | 1.6–1.8 % | 21.09 / 1.8780 | 12.32–22.49 | — | 2.115 / 2.548 | E 1.988, 95 % [1.279, 3.112] |

Every column is monotone in the radius, and in the direction *against* the reading:
BMA accuracy falls 31.25 → 28.12 → 23.83 → 21.09 and NLL rises 1.7352 → 1.7563 →
1.7803 → 1.8780.  The best shell is the smallest, and it only **ties** the anchor's
accuracy (31.25) while losing 0.008 nat of NLL; nothing from 1e-2 outward reaches the
anchor at all (−3.1, −7.4, −10.2 points).  All four `R̂` are inside their nulls, but
that is a weak statement here: acceptance of 2 % at `d₀ = 3e-2` means τ ≈ 5 000 steps
and 8 000 post-burn steps is 1.6 τ, so the two far shells are direction, not
certificates — they are also the two that are furthest below the anchor, so the
uncertainty cannot rescue the reading.

What the shells *do* show is the ensemble repairing its own draws, which is a
different claim from beating the anchor: at `d₀ = 9e-2` the worst chain is 12.32 %,
the mean per-chain is 17.7 %, and the BMA is 21.09 % (+7.5 over the worst, +3.4 over
the mean).  Spread makes averaging worth something; it does not make the average
*good*.

The step budget is not innocent in this reading — the same far shell at `N = 8 192`
doubles its BMA accuracy between 20 k and 200 k steps (next block) — so read the four
rows as "at 10 k steps", not as converged shells.  That caveat does not rescue the
reading, because every row here is a *deficit* and longer runs move the two far ones in
the direction that would make their per-chain numbers worse, not better.

This is now the second `N` at which a fixed-radius ladder contradicts "breadth beats
the mode", and the two ladders agree with each other.  The certified `N = 512` table
above has 4e-2 → 44.92 % and 7.9e-2 → 42.97 % against an anchor of **49.22 %**, and
the 200 k-step `N = 8 192` shell reaches 44.92 % against an anchor of **21.88 %**.
So across three `N` the same protocol gives −4.3, −7.4 to −10.2 (four radii) and
+23.0 points: the `N = 8 192` wide shell is the outlier, and it is also the one
whose `R̂` sits above its null and whose per-chain draws (23.9–25.5 %) sit *at* its
anchor rather than below it — a different regime (far draws cost nothing per-draw at
that anchor), not a better one.  Its 44.92 % is therefore not evidence that a shell
buys accuracy; it is an uncertified anomaly that two other `N` reverse.

**How much of the breadth reading is exploration?  (`N = 8 192`, one shell, two step
budgets.)**  The 0.0905 rad shell's 44.92 % came from a 200 k-step run with 800 draws
per chain.  The same cell (default anchor 21.88 % / 1.5988, `r = 3e-3`, `d₀ = 9.05e-2`)
at 20 k steps, burn 4 000, thin 20 (400 draws per chain) gives:

| steps / thin | per-chain acc | per-chain NLL | train L | BMA acc / NLL | `R̂`(NLL) / (loss) |
|---|---|---|---|---|---|
| 20 k / 20 | 23.86–24.79 % | 1.6088–1.6140 | 1.6123–1.6171 | 28.12 % / 1.6100 | 1.689 / 2.338 |
| 200 k / 100 | 23.85–25.54 % | 1.7016–1.7320 | — | 44.92 % / 1.6999 | 1.24 / 1.80 |

Per-chain accuracy barely moves (24.4 → 24.7 %) and per-chain NLL gets 0.09 *worse*,
while the BMA gains 16.8 points.  That is the signature of coverage, not of the radius:
at 20 k steps the four chains sit in the part of the shell nearest the mode direction
(hence the better per-chain NLL), and at 200 k they have spread over the whole shell,
where single draws predict worse but their errors cancel.  So the 44.92 % is an
*exploration* effect that a 20 k budget does not reach, and the honest statement of
where breadth pays is: at a fixed step budget, breadth costs accuracy against a
certified anchor; with 10× the steps on a far shell, the ensemble can outvote its own
per-draw deficit.  The four-shell ladder above is a 10 k-step ladder for exactly this
reason, and its far two rows (2 % acceptance, τ ≈ 5 000 steps) are the least converged
of the four.

**Where a localized posterior is worth computing.**  Five rungs, three of them at
`N = 2 048` under different anchors, put the answer somewhere narrower than the
two-rung reading suggested:

- The earlier guess — a threshold near `N ≈ 10P` above which localizing pays in
  accuracy — is withdrawn.  Three of the five rungs put the mode *below* its anchor
  in accuracy, including both of the best-anchored ones, and `lr` alone is worth 7.4
  points at `N = 2 048`, which is larger than the one positive rung's +2.7.
- What survives is one monotone fact: **the NLL gap closes with `N`** (+0.037 →
  +0.0009) and never changes sign, so localizing at the peak of `p(d)` never buys
  calibration, it only stops costing it.
- And one robust negative: at `N = 2 048` and `16 384` a four-chain shell at the mode
  **ties** the anchor at best (the +1.2 row is within one test example of a tie), and
  the single accuracy win, `N = 8 192` +2.7, does not replicate one rung up.  The
  anchor is what improves with data; the modal shell is a slightly worse copy of it.

So the interior mode is not worth four chains at any `N` measured here.  The
remaining candidate was the small-`d` end of the informative band, and the
best-anchored `N = 2 048` run weakens it: the 1e-4 and 3e-4 widths there report
31.28 % / 1.7271 and 31.38 % / 1.7278 against the anchor's 31.25 % / 1.7270, i.e.
ties, where the same widths beat an undertrained 0.005 anchor by 0.001–0.004 nat.
The apparent gain was the anchor's shortfall, not a property of `π`, and the chains
below show the same collapse at `N = 8 192`: the walk that starts where those rows
are reported ends up 0.0008–0.0020 nat *worse* than its anchor.

Three things follow for where to compute, stated with the evidence each has:

1. **Average over a band, but do not expect the average to beat the anchor.**
   `bma_metrics` (`pcnl/src/experiment.rs`) is the mean softmax probability over
   *every* kept draw, and per-chain acc (line 525 / 1100) is the mean single-draw
   accuracy — so BMA-minus-per-chain is a measurable ensemble effect, not an
   artefact of the reporting.  It is large where the draws disagree: the wide
   0.0905 rad shell at `N = 8 192` reaches 44.92 % on per-chain 23.9–25.5 %, and at
   `N = 2 048`, `d₀ = 9e-2`, 21.09 % on per-chain 12.32–22.49 %.  But the anchor
   wins at both `N` where it is certified: 49.22 % against 44.92 % and 42.97 % at
   `N = 512`, 31.25 % against 31.25 → 21.09 % at `N = 2 048`.  So a band buys
   robustness against a single bad draw, not accuracy.
2. **Do not spend the localization on NLL.**  Every mode-radius measurement here
   lands at or slightly above its anchor's NLL (five rungs, +0.0009 to +0.037).  A
   use case that needs calibrated probabilities gets nothing from the interior mode
   and pays 16× per step to compute it.  The small-`d` band looked like the exception
   and is not: at `N = 2 048` its NLL advantage over the anchor was 0.001–0.004 nat
   against an undertrained anchor but 0.0001–0.0008 against the best one, and at
   `N = 8 192` the chains that start in that band report 1.6097 / 1.6109 against the
   anchor's 1.6089.  The band does not have a calibration gain of its own.
3. **The compute is not the constraint until `N` is large.**  A certified shell at
   `N = 8 192` costs ~35 min of one core — but the same protocol at 0.09 rad took
   200 k steps because its relaxation time is 100× longer.  The expensive dimension
   is the shell's radius relative to the jump, not the model.

One cost note that constrains what can be measured next: the potential is the mean
over the *whole* training set, so a chain step at `N = 8 192` costs 16× one at
`N = 512`.  This shell was affordable only because a 0.0086 rad shell relaxes in
`(sin d₀/r)² ≈ 18` steps, so 20 k steps is ~1 000 relaxation times; the same
protocol at 0.09 rad needed 200 k.  Future `N`-ladders of shells should be priced
in relaxation times, not steps.

**The `P = 584` shell, out to 200 000 steps.**  The uncertified `d₀ = 0.0905` shell
was rerun 10× longer (4 chains, burn 40 000, thin 100, 800 draws each) at two step
sizes, against nulls computed at the acceptance-corrected radius
`r_eff = r√(mean acceptance)`:

| `r` | acceptance | per-chain acc | BMA acc / NLL | `R̂`(NLL) / `R̂`(loss) | null 95 % / 99 % |
|---:|---|---|---|---|---|
| 1.0e-2 | 0.3–1.9 % | 24.86 / 25.29 / 25.31 / 20.20 % | 43.75 % / 1.7215 | 2.80 / 3.91 | [1.021, 1.195] / 1.235 |
| 3.0e-3 | 19.7–34.1 % | 23.85 / 25.54 / 22.38 / 22.97 % | 44.92 % / 1.6999 | 1.24 / 1.80 | [1.010, 1.096] / 1.113 |

A 3× narrower step lifts acceptance from ~1 % to ~30 % and cuts `R̂`(NLL) from
7.18 (at 20 k steps) to 1.24 — the walk-length diagnosis was correct.  It still
sits above the 99 % null, so the shell is **not certified**: what remains is a
level difference between chains (NLL 1.7016, 1.7230, 1.7320, 1.7294 — a spread of
~1.5 standard errors per chain at 800 draws), which only longer chains narrow, not
a different step.  The accuracy it reports (44.9 %) is 24 points above its anchor
in the BMA sense and 3 points per-chain, so the *effect* survives the failure to
certify it; the number does not deserve a confidence interval yet.

Taken as a whole the cycle leaves the project's shape unchanged and its one
quantitative scaling claim retired: accuracy at a fixed radius is real and
anchor-relative, the mode sits outside the informative band at `P = 16 672` under
any anchor, at `P = 584` it moves inside as `N` grows (now certified by a converged
shell at 0.00857 rad, where the mode buys +2.7 accuracy points and no NLL, against
−3.1 / +0.037 and −4.7 / +0.022 two rungs down at `N = 2 048`), and
"how much data to localize" is not yet a measurable quantity — the instrument that
would measure it has to hold the anchor fixed, which means training to convergence
*before* the sweep, not sweeping with more data.

*The tangent coordinate.*  The radial law says `d` never came in; the second
question is how many **directions** the data bought, and it has the same answer
because it is the same inequality.  Every "`π` is prior-dominated" sentence in
§6.5.2–6.5.7 rests on one arithmetic claim — that a cap of angular radius `d` on
`S^{P−1}` holds a fraction `≈ (sin(d/2))^{P−1}` of the sphere.  That is the
measure of a **cosine** cap — right at `P = 3`, wrong everywhere else.
`pcnl/src/budget.rs` carries the exact law instead,

```text
f_m(d) = ½ · I_{sin²d}((m−1)/2, ½),      d ≤ π/2,
```

with `I` the regularized incomplete beta, evaluated in log space (Lentz
continued fraction) because `x^a` underflows outright at
`m = 16 672, d = 10⁻³`.  Nine unit tests pin it: the three closed forms (`d/π`
at `m = 2`, `sin²(d/2)` at `m = 3`, `(d − sin d cos d)/π` at `m = 4`),
Monte-Carlo at `m = 5, 9, 23`, monotonicity in both arguments, the published
configuration's five-figure cost, the inversion the budget is solved with, the
nearest-of-`n` radius, and degenerate radii.

The price is the point.  For large `m`,
`−ln f_m(d) ≈ (m−1)·ln(1/sin d)`, so **directions cost nats linearly**, and a
likelihood can pay at most `N·ΔL` of them:

```text
m* = N·ΔL / ln(1/sin d)        — how many directions the data can localize
```

`examples/dimension_budget` measures both sides at the reference configuration
(`P = 16 672`, `N = 512`, seed 20261020, anchor `u_Adam`):

**The geometry, validated.**  256 Haar draws, each at its distance from
`u_Adam`: min 1.5508, p5 1.5580, median 1.5706, max 1.5934 rad.  The exact law
predicts the nearest of 256 at **1.5502 rad** (0.04 % off), and 0.000 of 256
inside 0.1 rad — measured 0.  The "1.0–1.6 rad" row quoted earlier in this
document is a *bucket label*, not an observation: no Haar draw in this study
has landed nearer than 1.55 rad ≈ 89°.

**The purse.**  `N·ΔL` along eight random rays from the anchor (median; mean
and max tell the same story, the spread is why every `m*` below is quoted with
its best-ray value):

| radius (rad) | `N·ΔL` median | max | held-out acc |
|---:|---:|---:|---:|
| 1.0e-3 | 0.27 | 0.51 | 55.08 % |
| 2.3e-3 | 1.44 | 3.06 | 53.91 % |
| 5.0e-3 | 6.80 | 13.67 | 51.95 % |
| 1.0e-2 | 26.47 | 45.25 | 48.83 % |
| 2.0e-2 | 108.53 | 131.66 | 40.23 % |
| 4.0e-2 | 370.63 | 450.86 | 28.52 % |

`N·ΔL` grows as `d^1.95` over the ladder — quadratic, which is what a smooth
loss must give — and then saturates.  The informative set is a **level set**,
bounded energetically, not a cone: the loss rises in every direction, so there
is no ridge for a sampler to slide along.

**The budget.**  `cost/dir` is `−ln sin d`; `m*` is the exact-law answer,
and the last column is the same quantity on the friendliest ray.

| radius | cost/dir | exact cap cost at `P = 16 672` | nats available | `m*` | best ray |
|---:|---:|---:|---:|---:|---:|
| 1.0e-3 | 6.91 | 115 165 | 0.27 | 2.0 | 2.0 |
| 2.3e-3 | 6.07 | 101 280 | 1.44 | 2.0 | 2.0 |
| 5.0e-3 | 5.30 | 88 334 | 6.80 | 2.5 | 3.5 |
| 1.0e-2 | 4.61 | 76 779 | 26.47 | 6.5 | 10.5 |
| 2.0e-2 | 3.91 | 65 224 | 108.53 | 28.5 | 33.5 |
| 4.0e-2 | 3.22 | 53 672 | 370.63 | 115.5 | 140.5 |

`m* = 2.0` is the module's floor — the budget is below the circle's own price,
so those rows say "cannot pin even a plane", not "two directions".

**Is this a fact about `P = 16 672`?**  Same data, same radius (0.04 rad),
`N·ΔL = 370.63`, varying only the ambient dimension:

| `P` | cap cost | `m*` | `m*/P` | verdict |
|---:|---:|---:|---:|---|
| 42 | 134.8 | 115.5 | 275 % | localizable |
| 100 | 321.9 | 115.5 | 116 % | localizable |
| 512 | 1 649 | 115.5 | 22.6 % | partly localizable |
| 4 240 | 13 651 | 115.5 | 2.7 % | prior-dominated |
| 16 672 | 53 672 | 115.5 | 0.69 % | prior-dominated |

`m*` is identical at every `P` because the budget counts **nats, not
parameters**.  What `P` moves is the *share* of the tangent space the posterior
can reach — 0.69 % here.  Equivalently: the largest model whose 0.04-rad cap
512 examples can still pin is `P ≈ 115`, and `P = 16 672` is 145× short.

**What the proxy cost.**  The old formula's error is not noise but exactly
`(P−1)·ln 2` in the large-`P` limit:

| radius | exact | proxy | difference |
|---:|---:|---:|---:|
| 1.0e-3 | 115 165.0 | 126 714.6 | 11 549.7 |
| 2.0e-2 | 65 224.2 | 76 773.1 | 11 548.8 |
| 4.0e-2 | 53 672.1 | 65 218.4 | 11 546.3 |

At `P = 16 672` that is 11 547 nats — 5 014 decades, against a six-figure
margin, so **the prior-domination verdict never depended on it**.  But the
*crossover* is a different matter: it lives at `m ≈ 10`, where a factor of
`2^m` is the entire answer, which is why the exact law replaces the proxy in
the module rather than beside it.  The old volume estimate (§6.4(ii): the
equatorial shell outweighing a 1 mrad ball by `10^{54 000}`) is the same
inequality seen from the volume side, and the exact law's figure for that ball
is `10^{50 014}` (a fraction `10^{−5.0e4}` of `π`).  So the conditionals of (4)
are a tail, not the predictive, however exactly they are sampled.

**6. The verdict, and what it is not.**  At the radii where the fine-tune's
held-out accuracy survives (0.01–0.02 rad, 48.8 %–40.2 %), the data can localize
**7–29 directions out of 16 671**.  Three unrelated measurements agree on a
single- to double-digit effective dimension:

* this budget: `m* = 6.5` at 0.01 rad, 28.5 at 0.02 rad;
* `subspace_bound`'s Hessian spectrum: the top 8 eigenvectors carry 72.8 % of
  `Tr H` ⇒ rank ≈ 11;
* the fixed-radius chain's free-walk growth exponent (0.509 of a diffusive
  walk) and 89.7° pairwise displacement angles.

The radial mode `d* = 6.5 rad` of (5) and the budget `m* = 6.5` share a digit and
nothing else: the first is a radial position that does not exist, the second is
how many tangent directions a 0.01-rad cap can be paid for.  What follows is that
the sampler results are not a kernel defect.  At `P = 16 672` the **target**
constrains 99.3 % of the tangent space not at all, and a pCN step spends its
noise exactly there; no schedule, metric or guidance term changes that
arithmetic, which is consistent with §6.5.6–6.5.8 having exhausted those levers.
The open question is no longer the kernel but **which `m*`-dimensional
conditioning to sample**: an informed rank-`m*` subspace (e.g. the Adam-on-B
path's own Hessian block) is where the tempered target becomes samplable, and
that is the next experiment.

The whole section is two commands: the four `--arm latitude` runs of the
reproduce block below, and `cargo run --release -p pcnl --example
dimension_budget` (~3 min, wired into `./scripts/pcnl-reproduce budget` as the
anchor-drift canary).

> **Correction, same day: the §6.5.5 reference points are the *pre-fix* block's,
> not a measurement.**  Rerunning `adam_start` (defaults, seed 20261020, 5
> steps) prints
> `θ_A 13.67 % / 4.2124 / ‖θ‖ 124.9 (2.8 × shell)` and
> `θ_Adam 55.08 % / 1.1367 / ‖θ‖ 1620.1 (35.8 × shell)`.  The table above line
> 707 records `θ_A 4.30 % / 4.1523 / 94.0 (2.1 ×)` and
> `θ_Adam 49.22 % / 1.3136 / 1272.6 (28.2 ×)` — and those are *verbatim* the
> `B baselines: before 4.3% (NLL 4.1523) … Adam-on-B 49.2% (NLL 1.3136)` line
> of the 2026-10-20 block that this document's opening section declares
> non-reproducible.  The 2026-11-05 section reused the stale pre-fix baselines
> as its anchor instead of re-measuring them, and every table that quotes
> `49.22 %` as its frozen start row — including the anchor rows of the latitude
> and conditional tables above — inherits that.  Consequences: the anchor
> sits **further** out than recorded (35.8 × shell, not 28.2 ×), and each
> "46 % vs the anchor" comparison in §6.5.6–6.5.7 is ≈6 points low in its gap.
> What does **not** move is any geometric quantity — radii, cap masses, `m*` —
> because they are ratios at fixed data; the budget's verdict is unchanged in
> kind and slightly stronger (a further-out anchor with the same `N·ΔL` buys
> even fewer directions).
>
> One loose end, recorded rather than explained: the two examples score the
> *same* point (direction `u_Adam`, radius 45.19) 0.02 nats apart — 1.1139 in
> `dimension_budget`, 1.1367 in `adam_start` — while agreeing exactly on the
> 55.08 % accuracy.  So the forward pass is exactly rank-invariant across a
> rescale but only approximately value-invariant; `EPS = 1e-6` is too small to
> account for 2 % of 1.14, and nothing in this section depends on which of the
> two is quoted.

## R0 — numerical and diagnostic trust (2026-11-08)

`pcnl/RESEARCH_PLAN.md` gates every new inference result behind R0: the numerics have
to be honest before another ladder is worth running.  Three items are done here, each
with the measurement that licenses it.  The plan's own rule applies to the rest: nothing
below changes a *forward* value, so the recorded anchors and tables stand; what changed
is what the code reports about them.

**(a) The normalisation's derivative (`RESEARCH_PLAN.md` R0.1).**  `normalize_fro`
computes `W/(‖W‖_F + ε)` but reads `‖W‖_F` out with `to_scalar`, which detaches it from
the graph: the backward pass sees a fixed rescaling and drops the `∂‖W‖/∂W` term.  That
is now an explicit choice, `NormGrad::Stopped` (the historical form, still the default,
because every number in this document was produced with it) versus `NormGrad::Exact`
(via `normalize_fro_with`, `forward_norm`, `loss_and_grad_norm`).  Measurements:

* the two agree on the **value** to f32 rounding, at scales 1e-3…40 (unit test);
* the **exact** form matches a directional finite difference at θ-scales 1e-3, 1 and
  50, along a generic direction, the global ray, and one block's ray — the last two are
  what the old coordinate-wise check could not see;
* the **stopped** form's gradient differs from the exact one by
  `|Δg|/|g_e| = 1.4e-3` at `P = 3 232` (worst per-block residual of the identity
  `g_e = g_s − (W·g_s)/(r(r+ε))·W` is 4.8e-6, and the largest radial share of a stopped
  block gradient is 0.069).  The bias is small because the cosine loss is radially flat
  (§6.5, `cosine_likelihood_is_flat_along_a_radial_ray`): the term it drops points where
  the map barely responds.

So: **the surrogate is labelled, not silently kept, and its drift bias is 0.1 %** — small
enough that the recorded anchors are not wrong for a gradient reason, large enough that a
*gradient-level* claim (drift, ⟨θ,∇Φ⟩ floors, SG-bias curves in R2) must say which form it
used.  All reported chain results use MH ratios, which never touch the gradient, so none
of them is affected either way.

**(b) The schedule's floor (R0.2).**  `beta_at` clamps at `1e-6`, so the implemented
sequence has **infinite** total mobility for every `κ`; the Robbins–Monro window
(`Σβ² = ∞`, `Σβ⁴ < ∞`) is a statement about the unfloored power law.  Both laws are now
named (`beta_at_unfloored`), the crossover is computed (`floor_crossover`, exact formula
`((β₀/ε)^{1/κ} − 1)/γ`), and the answer is that it never binds in this project's runs:

| schedule | `β₀` | `γ` | `κ` | floor binds at | longest run |
|---|---:|---:|---:|---:|---:|
| default | 0.3 | 1e-4 | 0.5 | 9.0e14 steps | 2e5 |
| pre-2026-11-04 | 0.3 | 1e-4 | 0.55 | 9.1e13 steps | 2e5 |
| `κ = 0.6` control | 0.3 | 1e-4 | 0.6 | 1.3e13 steps | 2e5 |

`beta_at(step) == beta_at_unfloored(step)` bit-for-bit over 0…200 000 for both reported
exponents (test `floor_crossover_is_where_the_laws_part_company`).  The freeze statements
for `κ > 1/2` therefore describe the power law *and* the code, because the code is the
power law over every horizon anyone has run.

**(c) Frozen traces are failures, not good scores (R0.3).**  Before today `diag::ess`
returned `N` for a constant chain (“every draw agrees”) and `gelman_rubin` returned
`1.0` for two identically frozen chains (no between-chain and no within-chain
variance).  Both are the two numbers a stuck chain must never get.  Now:

* `movement(chains) -> Movement { chains, frozen, unique_max, nonfinite }` is the
  evidence, and `is_frozen`/`is_degenerate` are the flags a report can gate on;
* frozen → `NaN` from `ess`, `gelman_rubin`, `split_rhat_rank`, `ess_bulk`, `ess_tail`;
* frozen at *different* values → `+inf` from `gelman_rubin` (a real disagreement, not a
  missing one) and `NaN` from the rank statistics (their rank spread is finite by
  construction, so a zero-variance within-chain term has no defined ratio);
* any non-finite draw → `NaN` from all five, rather than a plausible-looking number;
* the plan's gate statistics now exist: rank-normalised split `R̂`
  (`split_rhat_rank`, bulk and folded combined, `Φ⁻¹` Blom scores) and `ess_bulk` /
  `ess_tail` as in Vehtari et al. (2021) — validated against iid draws (bulk ESS within
  half of the draw count, tail ESS above 100 of 4 000) and against AR(1) sticks
  (bulk ESS < 20 % of 4 000).

One real crash fell out of this: `experiment::median` sorted with
`partial_cmp().unwrap()`, so a single frozen coordinate made the quick report panic.
It now drops non-finite entries and returns `NaN` when none are left — an unavailable
number instead of a crash.

**(d) The tangent proposal's support (R0.4, in part).**  `SpherePcn::step_inner`
proposes `u′ = exp_u v` with `‖v‖ = a`.  The exponential map is one-to-one only on the
ball `a < π`; past it a proposal has several preimages, the closed-form pushforward
`(sin a/a)^{P−2}` describes one of them, and the reverse transport divides by `sin a`,
which is zero at `a = π`.  The proposal is now restricted to the ball: `a` is a length
along the geodesic, so the reverse move is inside it too, every accepted move keeps the
exact ratio, and an over-long proposal is a plain rejection.  The truncated Gaussian's
normaliser differs from 1 by `exp(−π²/(2τ²))`, i.e. `exp(−3e3)` at the reported
`τ = β/√P ≤ 0.04` — invisible.  Measured consequence, which is also a correction to an
older claim: the warm-up test that used curvature `1e6` was learning a metric from
proposals with `|m| ≈ 3.6e4 ≫ π`, i.e. from proposals that have no ratio at all.  With
the ball enforced that cell proposes nothing, accepts nothing and learns nothing; the
property the test exists for (the metric comes from proposals, not accepted states) is
now checked at curvature 40, inside the ball.  The exact invariant-moment tests
(`zero_likelihood_is_haar_on_the_circle`, `anisotropic_metric_is_exact_on_s2`,
`restricted_metric_inverse_matches_the_dense_one`) are unchanged and still pass, so the
restriction does not touch the regime the reported chains run in.

**(e) The kept window, and where the metric froze (R0.5).**  Every chain in this document
runs a *decaying* step size: `β_t = β₀(1 + γt)^{−κ}` never levels off, so no window of any
run is a stationary chain, and `ESS`/`R̂` computed on the kept draws describe the annealing
window and nothing beyond it.  The reports said that nowhere.  Now:

* `experiment::Window::of(cfg, scheduler)` records `start, end, β_start, β_end, λ` and one
  exact `stationary` bit, and every variant prints it under its diagnostics:
  `window: kept t=200..400 of 400 steps (thin 4, 50 draws/chain, 2 chains) | beta
  0.2739->0.2536 lambda 1.00 | NONSTATIONARY: the ESS/Rhat above describe this
  decaying-step window, not a stationary chain` (`Config::quick()`, measured).
* `stationary` is equality, not a tolerance: `β` identical at both ends of the window and
  `λ` already at the full target.  Under the default `κ = 0.5`, `γ = 1e-3`, 10 000 steps /
  burn 5 000, the kept window runs `β 0.1225 → 0.0905` — 26 % lower at its end — so the
  default answer is `NONSTATIONARY` and 500 draws per chain, not 500 stationary draws.
  Only `γ = 0` reaches `stationary`; a likelihood ramp that runs into the window keeps it
  nonstationary even then, and the record shows the tempered `λ` at the window's start
  (`window_separates_annealing_from_stationary_sampling`).
* `experiment::phases()` emits `warmup`, `likelihood-ramp`, `kept` as separate records
  (each with its own `λ`/`β` pair).  They overlap by design — a phase says what the kernel
  was doing over a range, it does not partition the run.
* Metric freeze: `SpherePcn::metric_freeze()` returns `None`, `Installed{dim,hash}`,
  `Frozen{at,dim,hash}` or `Adapting{step,dim}`, and `metric_record()` is the manifest
  form.  `examples/sphere_chain --precond` prints it under the cell (`metric
  diagonal-secant frozen at step 200 (dim 4240)`), so a preconditioned number now says
  whether its kept draws were drawn with the metric that was reported.
* A real defect fell out of writing the freeze record: `set_metric` installed a metric but
  left the warm-up flag on, so the first `step_preconditioned` replaced the installed `d`
  with a one-proposal secant.  `set_metric` now freezes at step 0
  (`metric_freeze_records_where_adaptation_stops` checks the installed `d` survives a
  step).  The ambient experiment has no metric to freeze — its proposal is isotropic — and
  records `"metric": {"kind": "isotropic"}` explicitly rather than omitting the field.

**(f) The run manifest (R0.6).**  `pcnl/src/manifest.rs` writes `{stem}.manifest.json`
beside the saved draws (`--save-dir`), hand-written JSON with a fixed key order and
non-finite values as `null` (no new dependency).  One measured file, from
`cargo run --release -p pcnl -- quick --save-dir /tmp/pcnl-r05`:

```json
{"tool":"pcnl","version":"0.1.0","created_unix":1794188928,
 "code":{"revision":"8d397fad63ea","dirty_hash":11004078211542389397,"dirty_files":6},
 "model":{"variant":"cosine/nGPT-lite | standard softmax","p":4240,"dtype":"f32",
          "device":"cpu","normalization":"fro-stopped/v1"},
 "data":{"task":"B","k":8,"n_train":128,"n_test":64,"seed":20261020,
         "hash":5721120083230051558,"anchor_hash":3732154325063501809},
 "target":{"lambda":1.0,"prior":"N(0, sigma^2 I)","anneal_steps":0},
 "phases":[{"name":"warmup","start":0,"end":200,...},
           {"name":"kept","start":200,"end":400,...,"stationary":false}],
 "metric":{"kind":"isotropic","freeze_step":null,"adapting":false,"dim":0,"hash":null},
 "work":{"forward_evals":102400,"backward_evals":102400,"wall_seconds":11.36,
         "peak_mem_mb":12.7,"nonfinite_events":1}}
```

The hashes are FNV-1a 64: `data.hash` covers the four materialised index tensors (not the
seed that made them, which two draws can coincide on), `anchor_hash` covers the frozen
Task-A θ_A, `code.dirty_hash` covers the tracked diff.  **Limits, stated because a
manifest that overclaims is worse than none:** it is an *identity*, not a resumption
record — RNG, loader and adaptation states are not saved, so it cannot restart a run; a
dirty hash names a diff that was present, not the binary that ran (that needs
`build.rs`-stamped object hashes); `peak_mem_mb` is process-wide `VmHWM` and includes the
pre-train.  Report text changed by one line per variant (`window:`), so older logs remain
valid as data but no longer byte-match a re-render.

What is *not* done yet from R0: the rest of the exact-proposal audit (the sphere and
latitude conditionals are audited and moment-tested already; the cap-conditioned target
and the cyclical schedule are not).  The tables in this document still report the raw
`ess`/`gelman_rubin` from before the rank statistics existed, which is why their `R̂`
values are comparable across dates; the null tables, which are what certify them, are
unchanged.

## R1 — the restricted affine subspace (2026-11-09)

The plan's R1 target is `θ(z) = θ_ref + C_half B z`, `z ~ N(0, I_r)`,
`π_r(dz) ∝ exp(−λ·Σ_i ℓ_i(θ(z))) N(0, I_r)(dz)`.  This section covers the machinery
(`pcnl/src/subspace.rs`, `pcnl/src/target.rs`, `examples/subspace_adaptation.rs`) and
its first measured cells at ranks 1–2.  It does **not** claim anything about the ambient
`P`-dimensional posterior: `π_r` is a different, smaller model, and the numbers below are
about that model.

**(a) What is validated, and how.**  `Basis` refuses a non-orthonormal `B` (two identical
columns give `max |BᵀB − I| = 1`, rejected; the gate is `1e-5`), `Basis::random` is a
seeded control with the same Gram check, `pca_basis` is power iteration + deflation on the
snapshot matrix (no `P×P` covariance) and recovers a known 2-plane to `4e-5`, refusing a
rank the path does not have (`Degenerate`) instead of emitting a zero column.
`AffineSubspace::gradient_to_z` is checked against a central difference of `f(θ(z))`
itself, so a transposed `B` or a dropped `C_half` fails the test rather than silently
changing the temperature.

**(b) A real bug, caught by quadrature.**  The first `mh_pcn_z` used
`log α = Φ(z) − Φ(z')` with the prior inside `Φ`.  pCN's proposal is **not symmetric** in
`z`, so that omits the proposal-density term and double-counts the quadratic part of the
target: a unit Gaussian came out at variance **0.498**, and the rank-2 double well at
`E[z²] = 0.730` against the quadrature value 0.852.  With the term written out,
`log q(z|z') − log q(z'|z) = ½‖z'‖² − ½‖z‖²` (which visibly cancels the prior), the rank-2
controlled target gives, at `β = 0.6`, 4 chains × 120 000 steps, thin 10:

| quantity | measured | exact (Simpson, 20 000 points) |
|---|---:|---:|
| accept rate | 0.545 | — |
| `E[z₁]` | −0.0075 | 0 |
| `E[z₁²]` | 0.8531 | 0.8521 |
| ESS(bulk, `z₁`) | 18 592 of 44 000 | — |

Both `pcn_keeps_the_unit_gaussian_variance` and the quadrature test fail on the old form,
so the correction is pinned by tests, not by a comment.

**(c) The `λ` factor that would otherwise be silent.**  The plan's target sums the
likelihood over the training set; the ambient experiment's `Φ` is the *mean* loss.  At the
same nominal `λ = 1` these differ by `N = 512`, i.e. the ambient convention is this target
at `λ = 1/N`.  Measured: with the mean form the posterior is prior-dominated (acceptance
96–100 %, rank-1 informed NLL 3.34 against the prior's 3.40); with the sum form acceptance
is 78–95 % and the informed rank-1 NLL is 2.62.  `SubspaceTarget` therefore uses the sum
form and names the factor in its documentation (`subspace_target_uses_the_sum_form`).

**(d) First transformer cells, on the ambient anchor.**  `P = 16 672` (the ambient
shapes `vocab 9, d 32, ff 64, layers 2`), `θ_ref` = the ambient `AnchorSetup`: Task-A
momentum pretrain, seed 20261020, 1 500 steps at lr 0.6, loss 5.0673 → 1.1812.  That
anchor scores **4.30 % / NLL 4.1523** on B (`anchor hash 0x52d503556818f1e0`,
`data hash 0xba104db4f677b2ec`), which is the same anchor the ambient experiment's
`run.manifest.json` records — `AnchorSetup`/`anchor_hash` are now shared code, and
`quick_experiment_reports_sane_numbers` asserts the equality, so "same anchor" is a
checkable claim rather than a description.  `λ = 1`, `σ = 0.35`, `β = 0.05`, 4 chains ×
4 000 steps (burn 500, thin 5; 2 800 draws per chain, 11 200 total), `N_train = 512`,
`N_test = 256`, 16 004 `Φ` evaluations, ~285 s per cell:

| rank | basis | accept | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | MCSE(Brier) | NLL | ECE | MCSE(NLL) | s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random | 98.0 % | 21 | 1.165 | 4.30 % | 0.000 | 1.0960 | 0.0000 | 4.1516 | 0.325 | 0.0004 | 286 |
| 1 | informed | 76.7 % | 58 | 1.055 | 13.28 % | 0.187 | 1.0127 | 0.0006 | 2.9462 | 0.264 | 0.0040 | 284 |
| 2 | random | 98.5 % | 36 | 1.170 | 4.30 % | 0.000 | 1.0958 | 0.0002 | 4.1508 | 0.325 | 0.0009 | 287 |
| 2 | informed | 71.8 % | 31 | 1.098 | 14.06 % | 0.049 | 0.9973 | 0.0006 | 2.9083 | 0.261 | 0.0068 | 280 |

The informed basis is PCA of 32 Task-B momentum steps (lr 0.05) — **data reuse**, so its
held-out numbers are optimistic by construction; the random control is what makes the
comparison readable, and at both ranks the random basis is indistinguishable from the
anchor (NLL 4.1516 / 4.1508 vs 4.1523, Brier 1.0960 / 1.0958, accuracy identical), which
is the control's job.  The informed basis buys 1.21 nat and +9.0 accuracy points at rank 1,
1.24 nat and +9.8 at rank 2 — now *increasing* in rank, and the ordering is within the
0.39 % accuracy granularity of 256 test examples (a 0.8-point difference is 2 examples).

Two things the ambient anchor changes, both negative and both published:

* the anchor is **below chance and worse than the uniform predictor**.  Chance is 12.5 %
  (`1/8`); the uniform predictor has NLL `ln 8 = 2.0794` and Brier `1 − 1/8 = 0.875`.
  The anchor's 4.30 % / 4.1523 / 1.0960 is worse on all three, i.e. 1 500 momentum steps
  on 512 Task-A examples produce a model that is confidently wrong on B.  At the quick
  shapes `P = 4240` the same recipe gives 7.03 % / 3.4220 — the larger model overfits the
  same data harder, which is why the ambient anchor is the worse one to start from.
* consequently the informed arms' gain is still largely *undoing the anchor's bias*: rank-2
  informed at 14.06 % / NLL 2.9083 / Brier 0.9973 is one accuracy point above chance and
  0.83 nat worse than the uniform predictor.  The R1 target's posterior predictive on B is
  at the base rate with a miscalibrated tail — the plan's negative target outcome.  For
  scale, the ambient experiment's BMA is 11.7 % / 2.1863 / ECE 0.1798 (10 000 unadjusted
  steps, K = 2 000, *conditioned on the cone* — see the note at the top of this document):
  the restricted arm is 2.3 accuracy points higher (MCSE 0.049) and 0.72 nat **worse** in
  NLL, i.e. restriction bought a base-rate accuracy bump and no calibration.  That is not
  the matched-budget comparison the plan asks for (the evaluation counts differ by 3×), and
  it is quoted as a pointer, not as a win.

Mixing is the limit, and it is reported rather than hidden: no cell is converged
(`R̂(Φ)` 1.055–1.170; rank-1 informed ESS(Φ) 58 of 2 800 per chain).  The random arms are
the worst — `Φ` is nearly flat in random directions, so a chain there wanders in the prior
and its potential's ESS is small even at 98 % acceptance — which is why R2's
preconditioning is measured next (it takes the rank-1 random arm from ESS 21 / `R̂` 1.165
to ESS 1 899 / `R̂` 1.002 without moving the predictive at all; §R2).  `β = 0.05` is not a
tuned value: at `β = 0.5` the informed direction accepts 2.8–5.2 %, and at `β = 0.05` it
accepts 72–77 %.

**(d′) The superseded quick-shape cells.**  Kept because they were the first published
cells and the difference is informative, not because they are comparable.  `P = 4 240`
(quick shapes), anchor 7.03 % / 3.4220, `Dataset::new(Task::B, …, seed)` instead of the
ambient `seed + 1`, 11 200 draws, s = 101–116:

| rank | basis | accept | ESS(Φ) | `R̂(Φ)` | ESS(loss) | `R̂(loss)` | BMA acc | NLL | ECE | MCSE(NLL) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random | 94.2 % | 13 | 1.338 | 6 | 1.731 | 7.03 % | 3.4019 | 0.283 | 0.0036 |
| 1 | informed | 85.0 % | 102 | 1.034 | 16 | 1.141 | 20.31 % | 2.6233 | 0.166 | 0.0023 |
| 2 | random | 94.6 % | 8 | 1.465 | 11 | 1.328 | 7.03 % | 3.4182 | 0.284 | 0.0005 |
| 2 | informed | 78.3 % | 474 | 1.006 | 100 | 1.031 | 18.36 % | 2.6086 | 0.145 | 0.0006 |

The rank-1 informed arm looks better here (20.31 % / 2.6233) than on the ambient anchor
(13.28 % / 2.9462) only because its anchor starts 1.4× higher on B; the *gain* over the
anchor is smaller at the quick shapes (0.80 nat) than at the ambient ones (1.21 nat).  The
earlier text of this section said "chance is 11.1 %" — that is `1/9`, the vocabulary size,
not the label count; chance is 12.5 %.

**(e) Comparability resolved.**  The earlier version of this section said "the ambient
reports' anchor is ~12.5 % / NLL 2.21 on B at the same `N`" and used it to excuse the
quick-shape anchor.  Both halves of that sentence were wrong: 12.5 % is *chance*, and 2.079
is the *uniform predictor's* NLL.  The ambient anchor scores 4.30 % / 4.1523 — below chance
and worse than uniform, `P = 16 672` overfitting 512 examples harder than `P = 4 240` did.
The cells above are now on that anchor, by shared code and a hash assertion, so the
restricted and ambient numbers are finally on one anchor; the caveat that remains is the
data reuse in the informed basis, not the anchor.

**(f) What is not done.**  Ranks 4–32; a Task-B-fitted anchor/basis arm with cross-fitting
(the empirical-Bayes arm); ambient vs restricted at matched likelihood evaluations;
`λ ≠ 1`; and the informed-basis construction the plan names besides trajectory PCA
(prior-whitened Gauss–Newton/Fisher directions).  Brier and the predictive MCSE are now in
the table: MCSE(acc) and MCSE(Brier) are `sd/√ESS` over the pooled draws' *single-draw*
score (`bma_detail`), not the MCSE of the BMA ratio-of-averages — the two differ, and
quoting the first as if it were the second would understate the BMA's variance.  The
programme's gate — "an informed basis must beat or explain its tie with a random basis,
not just beat a failed ambient run" — is met for the negative control at ranks 1–2 and not
yet for anything larger.

Reproduce:

```bash
cargo run --release -p pcnl --example subspace_adaptation                    # ranks 1,2,4,8
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 1,2 --steps 4000 --beta 0.05
# R2: diagonal and rotated arms at ranks 2 and 4 (the table above)
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 2,4 \
  --steps 4000 --burn 500 --thin 5 --chains 4 --proposal curv,pilot,rotcurv,rotpilot
# R2c: full target vs restricted, informed ranks 1,2,4 plus the rank-2 random control
cargo run --release -p pcnl --example subspace_adaptation -- --targets restricted,full \
  --ranks 1,2,4 --basis informed --proposal fixed,pilot --chains 4 --steps 4000 --burn 500 --thin 5
cargo run --release -p pcnl --example subspace_adaptation -- --targets restricted,full \
  --ranks 2 --basis random --proposal fixed,pilot --chains 4 --steps 4000 --burn 500 --thin 5
# R2d: the pilot arm across ranks, and the controls that separate dimension from
# pilot length (`--pilot` is the number of pilot steps, default 300; 1 200 = 150/rank at r=8)
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 1,2,4 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot,rotpilot
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis random --proposal fixed,pilot
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal rotpilot
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal pilot --pilot 1200
# R2e: pilot length (rank 8 at 2 400 = 300/rank, rank 4 at 600 = 150/rank) and the
# log-width shrinkage ladder (`--pilot-shrink`: 1 = raw pilot, 0 = isotropic at the
# pilot's geometric mean; default 1).  Every cell is one process, 4 chains.
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot 600
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot 2400
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot 600
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot 1200
for r in 4 8; do for s in 0.5 0.0; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot-shrink $s
done; done
# R2f: the same ladder on other pilot streams (`--seed` moves the pilot and the chains;
# default 20261020).  Each extra seed is the same 4 cells for whichever rank is in question.
for s in 1 0.5 0.0; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --seed 7 --pilot-shrink $s
done
for s in 1 0.5 0.0; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --seed 7 --pilot-shrink $s
done
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --seed 8 --pilot-shrink 0.5
# R2g: `--pilot-reps n` = n independent pilots of `--pilot` steps each, averaged in log
# space (total pilot evals = pilot * reps).  `--pilot-reps 1` is the old estimator.
for r in 4 8; do for sd in 20261020 7 8; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot 300 --pilot-reps 2 --seed $sd
done; done
for r in 4 8; do   # equal-budget single 600-step pilot
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot 600 --seed 7
done
for r in 4 8; do   # the seed-8 single-pilot baselines
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --seed 8
done
for s in 1 0.5 0.0; do   # shrinkage above rank 8
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 16 --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot --pilot-shrink $s
done
# R2h: `--pilot-scale c` multiplies the widths by c after shrinkage (`c = 1` = unchanged):
for r in 16 8 4; do for c in 2 4; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot-shrink 0 --pilot-scale $c
done; done
for c in 1 4; do   # the rank-8 seed-7 pair (no s = 0 cell existed at that seed)
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot-shrink 0 --pilot-scale $c --seed 7
done
# R2i: `--pilot-ebayes` = per-coordinate reliability weights on the pooled shape
# (needs `--pilot-reps >= 2`); `--pilot-shrink` still multiplies on top:
for r in 4 8; do for s in 1 0.5; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot 300 --pilot-reps 3 --pilot-ebayes --pilot-shrink $s
done; done
for s in 1 0.5; do   # the rank-8 seed-7 pair
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 8 --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot 300 --pilot-reps 3 --pilot-ebayes --pilot-shrink $s --seed 7
done
for r in 4 8; do   # control: the same 3 pilots pooled with no reliability weights
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r --steps 4000 \
    --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
    --pilot 300 --pilot-reps 3 --pilot-shrink 1
done
# R2j: is rank 16 resolvable?  Fixed arm only — no pilot, no estimator:
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 16 --steps 16000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed      # ~20 min
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 16 --steps 8000 \
  --burn 500 --thin 5 --chains 8 --basis informed --proposal fixed      # ~23 min
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 16 --steps 8000 \
  --burn 500 --thin 5 --chains 8 --basis informed --proposal fixed --seed 7  # ~23 min
# R2k: `--proposal laplace` — curvature at the target's mode (Adam on the exact
# gradient of Φ, printed per cell with its convergence check).  s = 0.5 is the
# rank-4 recommendation; rank 8 is the negative control:
for r in 4 8; do for s in 1 0.5; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --proposal fixed,laplace --pilot-shrink $s          # ~18 min each (4 min is the mode)
done; done
for s in 1 0.5; do   # rank-4 seed-7 replication, with its own isotropic baseline
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --proposal fixed,laplace --pilot-shrink $s --seed 7
done
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 --steps 4000 \
  --burn 500 --thin 5 --chains 4 --basis informed --proposal fixed,pilot \
  --pilot-shrink 0 --seed 7                             # that baseline (ESS 152)
# R2l: the λ boundary (each λ needs its own isotropic baseline), and the full-
# covariance control at the same mode (`rotlaplace` prints its β eigen-range):
for lam in 0.5 2; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed --lambda $lam \
    --proposal fixed,laplace --pilot-shrink 0.5          # 419 vs 351, ESS(Φ)
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed --lambda $lam \
    --proposal fixed,pilot --pilot-shrink 0              # the isotropic baselines 342 / 376
done
for r in 4 8; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --proposal fixed,rotlaplace                          # 112 / 6 vs diagonal 587 / 8
done
# R2m: the anchor shift, the interior-point curvature (`--laplace-tau`) and the two
# log-width blends.  Each `fixed` row is the control; `pilot`/`laplace` are the arms:
for f in 0 0.1; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --anchor-jitter $f --proposal fixed,pilot --pilot-shrink 0   # 383 vs 7 (R̂ 10.18)
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --anchor-jitter $f --proposal fixed,laplace --pilot-shrink 0.5  # 587 vs 378
done
for r in 4 8; do   # τ = 0.5: 428 (0/4 non-positive H_ii) vs ESS 5 / R̂ 17.96 (1/8)
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --laplace-tau 0.5 --proposal fixed,laplace --pilot-shrink 0.5
done
for mix in 0.5 0.85; do   # anchor/mode widths: 5 / 254 at rank 4, 6 / 6 at rank 8
  for r in 4 8; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --laplace-mix $mix --proposal fixed,laplace --pilot-shrink 0.5
  done
done
for mix in 0.25 0.5; do   # pilot/mode widths: 352 / 271 at rank 4, 85 / 108 at rank 8
  for r in 4 8; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --pilot-mix $mix --proposal fixed,pilot --pilot-shrink 0
  done
done
# the seed-7 λ sweep (two seeds per λ; λ = 2's mode search hits the 2 000-step cap):
for lam in 0.5 2; do
  for prop in "fixed,pilot --pilot-shrink 0" "fixed,laplace --pilot-shrink 0.5"; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --lambda $lam --seed 7 --proposal $prop
  done
done
# R2n: the anchor × seed sweep (rank 4, λ = 1, two arms per invocation plus `fixed`):
for f in 0.05 0.10 0.20 0.30; do
  for s in 20261020 7; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --seed $s --proposal fixed,pilot --pilot-shrink 0
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --seed $s --proposal fixed,laplace --pilot-shrink 0.5
  done
done
# the width vectors behind Result 59 (prints `width-report pilot|mode[rank] = [...]`);
# the same cells reproduce 100 / 383 at rank 4 and 7 / 154 at rank 8:
for r in 4 8; do
  cargo run --release -p pcnl --example subspace_adaptation -- --ranks $r \
    --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
    --width-report --proposal fixed,pilot --pilot-shrink 0
done
# R2o: eight anchors on ONE data split.  `--anchor-seed` re-draws only the Gaussian
# shift; D_A, D_B, θ₀, the basis draw seed and the test set stay on `--seed`.
for f in 0.05 0.10 0.20 0.30; do
  for k in 1 2; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --proposal fixed,pilot --pilot-shrink 0
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --proposal fixed,laplace --pilot-shrink 0.5
  done
done
# R2p: the same eight anchors, basis pinned to the ambient anchor (`--basis-anchor`),
# so `--anchor-jitter` moves only the sampling reference before the chain starts:
for f in 0.05 0.10 0.20 0.30; do
  for k in 1 2; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --basis-anchor \
      --proposal fixed,pilot --pilot-shrink 0
  done
done
# R2q: the same pinned basis and anchors, laplace arm (the `fixed` rows reproduce R2p exactly):
for f in 0.05 0.10 0.20 0.30; do
  for k in 1 2; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --basis-anchor \
      --proposal fixed,laplace --pilot-shrink 0.5
  done
done
# R2r: a second pinned basis — the informed path starts at an independently drawn
# anchor one anchor-rms from ambient (`|θ| 132.394`, basis anchor hash
# 0xfe9ad411b8756ef7 in all eight cells), so the basis is shared but is not the ambient one:
for f in 0.05 0.10 0.20 0.30; do
  for k in 1 2; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --basis-anchor-seed 1 \
      --proposal fixed,pilot --pilot-shrink 0
  done
done
# R2s: six directions per magnitude on the ambient pin (`k = 1, 2` are R2p's cells and
# reproduce exactly; only `k = 3..6` were run in R2s, 24 cells in total):
for f in 0.05 0.10 0.20 0.30; do
  for k in 1 2 3 4 5 6; do
    cargo run --release -p pcnl --example subspace_adaptation -- --ranks 4 \
      --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed \
      --anchor-jitter $f --anchor-seed $k --basis-anchor \
      --proposal fixed,pilot --pilot-shrink 0
  done
done
# the seven shape-estimator tests (fast, no sampling) and the gradient contract:
cargo test -p pcnl --example subspace_adaptation
cargo test -p pcnl --lib --release subspace_target_gradient
# the `--pilot-reps 1` / `--pilot-scale 1` bitwise check (old vs new binary, same reduced config):
cargo run --release -p pcnl --example subspace_adaptation -- --shapes quick --ranks 2 \
  --steps 600 --burn 100 --thin 5 --chains 2 --basis informed --proposal fixed,pilot --seed 3
```

## R2 — per-direction scales for the `z` chain (2026-11-09)

R1's limit was that one `β` had to serve every direction: rank-1 informed accepts 77 % at
`β = 0.05` and 2.8–5.2 % at `β = 0.5`, while ESS(Φ) stayed at 21–58. R2 asks whether the
scale can be *derived* per direction instead of hand-set, holding the target fixed.

The kernel side is small and exact (`src/target.rs`):

* `mh_pcn_z_scaled` — pCN with per-coordinate `β_i`, `z'_i = √(1−β_i²)·z_i + β_i·ξ_i` and
  `log α = Φ(z) − Φ(z') + ½‖z'‖² − ½‖z‖²`.  The quadratic term is the one R1's bug fix
  derived, and at equal `β_i` the kernel is bitwise identical to `mh_pcn_z`
  (`scaled_pcn_equals_the_uniform_kernel_at_equal_betas`), so this changes the proposal
  only, never the target.  Reversibility is per coordinate, so the chain is exact for any
  `β_i ∈ (0, 1]` — the preconditioner does not have to be right to stay correct.
* `curvature_diag(φ, z, h)` — central second differences of the *full* potential,
  `H_ii = 1 + λN·∂²ℓ/∂z_i²`, two forward passes per coordinate, no `P×P` matrix.
* `DiagScale` — widths `w_i`, a multiplier `m`, `β_i = clamp(m·w_i, 1e-4, 1)`; Robbins–Monro
  on `log m` toward 50 % acceptance with step `0.3/(1+t)^0.6`, applied **during burn only**
  and frozen afterwards, so the kept window is a fixed kernel.

A diagonal scale can only stretch along the coordinates, and the informed basis is a
fine-tuning path's PCA, not the target's eigenbasis: a stiff direction that is a
*combination* of the informed coordinates cannot get the small step it needs while its
partners get a large one.  The rotated extension is the same pCN applied in an orthogonal
basis:

* `mh_pcn_z_rotated` — `w = Qz`, `w'_i = √(1−β_i²)·w_i + β_i·ξ_i`, `z' = Qᵀw'`, accepted
  with the **unchanged** ratio `Φ(z) − Φ(z') + ½‖z'‖² − ½‖z‖²`.  This is exact for any
  orthogonal `Q` because `N(0, I)` is invariant under it; equivalently it is
  full-covariance preconditioning with `S = Qᵀ diag(β_i²) Q` and `(I−S) + S = I`.  The
  kernel asserts `‖QᵀQ − I‖ ≤ 1e-6` before use (`O(r³)` against one full forward pass), so
  a bad rotation is refused rather than silently sampling a different target.  A *block*
  preconditioner is the special case where `Q` is the identity outside a block; at the
  restricted target all `r` coordinates are in the block, so the code takes the full
  rotation.
* `curvature_hessian` — the full Hessian at `z` by central differences, `2r² + 1` forward
  passes (≤ 129 at `r = 8`), symmetrised; the diagonal-only version costs `2r`.
* `RotScale` — widths `1/√λ_i` from a Hessian and `√λ_i` from a covariance, in the
  estimate's eigenbasis, with the identity eigen-decomposition a cyclic Jacobi
  (`symmetric_eigen`, no new dependency).  The multiplier, its Robbins–Monro rule and the
  freeze-after-burn are `DiagScale`'s, now shared (`adapt_log_multiplier`).  A
  non-symmetric matrix is rejected, not symmetrised, because it is used as a change of
  basis.

The kernel tests pin the two things that could go wrong: with the identity rotation the
kernel is **bitwise** equal to `mh_pcn_z_scaled` (accept flags, `z`, `Φ`), and on a
correlated Gaussian with `H = [[50.5, 49.5], [49.5, 50.5]]` (eigenvalues 1 and 100 along
`(1,±1)/√2`) the rotated chain recovers the full posterior `H⁻¹ = [[0.505, −0.495],
[−0.495, 0.505]]`, correlation included, at ESS > 2 000 — acceptance alone would not prove
the covariance is right.

Two arms, neither of which adds a gradient evaluation:

* `curv` — `w_i = 1/√H_ii` at the anchor `z = 0`, computed once per cell (`2r` forward
  passes).
* `pilot` — 300 steps of the ordinary `mh_pcn_z` at `β = 0.05` from the chain's own
  stream, then `w_i` = per-coordinate sd over the *second half* of that pilot; the pilot's
  draws are discarded and the chain continues from where it ended.  These steps sit inside
  `burn`, so nothing about the kept window is adaptive.  A pilot cannot visit a wider
  region than its kernel allows, so `w_i` is biased *down* — the safe direction for
  acceptance, since `m` may grow — but the estimate is warmed by the very hand-set `β`
  under test, which is the arm's self-reference.

All cells: ambient anchor (4.30 % / 4.1523, `0x52d503556818f1e0`), `P = 16 672`, `λ = 1`,
`σ = 0.35`, `N = 512/256`, 4 chains × 4 000 steps, burn 500, thin 5 (2 800 draws per
chain).  Each arm gets its own RNG stream offset, so arms are independent replicates
rather than re-used draws.  `m` is the mean over chains, `β_i` the range over chains ×
coordinates, `ESS/1k evals` = ESS(Φ) per 1 000 `Φ` evaluations, which removes wall-clock
noise (the cells ran on a 2-core box, partly concurrently):

| rank | basis | arm | accept | ESS(Φ) | `R̂(Φ)` | ESS/1k evals | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random | fixed | 98.0 % | 21 | 1.165 | 1.3 | 4.30 % | 0.000 | 1.0960 | 4.1516 | 0.325 | 0.0004 | — | 0.05 |
| 1 | random | curv | 51.9 % | **1 899** | **1.002** | **118.6** | 4.30 % | 0.000 | 1.0959 | 4.1507 | 0.325 | 0.0000 | 3.23 | 0.92–1.00 |
| 1 | random | pilot | 50.0 % | 1 245 | 1.004 | 72.4 | 4.30 % | 0.000 | 1.0959 | 4.1507 | 0.325 | 0.0000 | 5.74 | 0.90–1.00 |
| 1 | informed | fixed | 76.7 % | 58 | 1.055 | 3.6 | 13.28 % | 0.187 | 1.0127 | 2.9462 | 0.264 | 0.0040 | — | 0.05 |
| 1 | informed | curv | 0.1 % | 7 | 1.637 | 0.4 | 11.33 % | 0.247 | 1.0595 | 3.6072 | 0.267 | 0.0218 | 0.02 | 1.00 |
| 1 | informed | pilot | 47.1 % | **680** | **1.004** | **39.5** | 13.28 % | 0.070 | 1.0124 | 2.9508 | 0.266 | 0.0002 | 0.18 | 0.12–0.13 |
| 2 | random | fixed | 98.5 % | 36 | 1.170 | 2.3 | 4.30 % | 0.000 | 1.0958 | 4.1508 | 0.325 | 0.0009 | — | 0.05 |
| 2 | random | curv | 52.8 % | 1 481 | 1.002 | 92.5 | 4.30 % | 0.000 | 1.0955 | 4.1498 | 0.325 | 0.0001 | 3.69 | 1.00 |
| 2 | random | pilot | 59.3 % | **1 769** | **1.000** | **102.8** | 4.30 % | 0.000 | 1.0955 | 4.1499 | 0.325 | 0.0001 | 5.22 | 0.60–1.00 |
| 2 | informed | fixed | 71.8 % | 31 | 1.098 | 1.9 | 14.06 % | 0.049 | 0.9973 | 2.9083 | 0.261 | 0.0068 | — | 0.05 |
| 2 | informed | curv | 0.1 % | 7 | 1.671 | 0.4 | 11.33 % | 0.501 | 1.0562 | 3.6196 | 0.268 | 0.0145 | 0.03 | 4.3e-3–1.00 |
| 2 | informed | pilot | 44.5 % | 152 | 1.012 | 8.8 | 15.62 % | 0.046 | 0.9971 | 2.9123 | 0.272 | 0.0004 | 0.17 | 0.017–0.13 |

**Result 1 — curvature at the anchor is the wrong preconditioner where the mode is
displaced, and the failure is bias, not just slowness.**  Informed rank 1 with `curv`
accepts 0.1 %, ESS(Φ) 7, `R̂` 1.637, and its predictive *degrades* against the plain fixed
arm: NLL 3.6072 vs 2.9462, accuracy 11.33 % vs 13.28 % (below chance).  With seven
effective draws that number is not an estimate of `π_r`, which is the point — a
preconditioner that is wrong by two orders of magnitude produces a confidently wrong
answer, not merely a slow one.  The mechanism is visible in the printed scales: `β_i =
1.00` for every coordinate at `m = 0.02`, i.e. `m·w_i ≥ 1` everywhere, so the `β ≤ 1` cap
binds and the arm runs pCN's *largest possible* step and rejects 99.9 % of it.  In other
words `1/√H_ii` at the anchor says "the posterior is wider than the prior" in every
informed direction — at 4.30 % accuracy on B the model's output there is nearly uniform,
so the Task-B cross-entropy is locally flat.  Local flatness at a stationary but
unrepresentative point is not posterior width: these directions are flat *here* and steep
one prior-width away.  A second detail: once every `β_i` is clamped, Robbins–Monro's
multiplications no longer change the proposal, so `m` drifted to 0.02 while acceptance
stayed at 0.1 % — the adaptation loop is uninformative after saturation, and the report
shows the clamp rather than hiding it.

**Result 2 — on the random (prior-dominated) directions, better mixing changes nothing.**
Rank 1 random goes from ESS 21 / `R̂` 1.165 to ESS 1 899 / `R̂` 1.002 (`curv`) and 1 245 /
1.004 (`pilot`), a 50–90× gain in ESS per evaluation, and the predictive is identical to
four decimals (NLL 4.1507 vs 4.1516, accuracy 4.30 %, Brier 1.0959 vs 1.0960).  Rank 2
behaves the same (36 → 1 481 / 1 769).  This is the control R1 needed: the random
subspace really is prior-dominated, and a *converged* chain there agrees with the
unconverged one, so R1's random-arm error bars were honest about a flat direction rather
than hiding a bias.  It also means the mixing fix alone cannot improve the answer, which
is why Result 3 is the one that matters.

**Result 3 — measured widths fix the informed arm without moving its answer.**  Informed
rank 1 goes from ESS 58 / `R̂` 1.055 to ESS 680 / `R̂` 1.004 (12× ESS, 11× per evaluation)
with the same predictive: NLL 2.9508 vs 2.9462, accuracy 13.28 % vs 13.28 %, Brier 1.0124
vs 1.0127.  The MCSEs say what that agreement is worth: the fixed arm's MCSE(NLL) is 0.0040
and the pilot arm's is 0.0002, so the two estimates differ by 0.0046 — 1.2 σ of the fixed
arm's own uncertainty — while the pilot's precision is 20× finer, which is what makes
"unchanged" measurable rather than merely plausible.  Rank 2 improves 31 → 152 ESS (`R̂` 1.098 → 1.012, accuracy 14.06 % →
15.62 %, inside MCSE(acc) 0.046) but is still not converged at 152 of 2 800 draws per
chain.

**Result 4 — the scale the arms find is not one number.**  The pilot's `β_i` are
0.12–0.13 at rank-1 informed (≈2.5× the hand-set 0.05), 0.017–0.13 at rank-2 informed (an
8× spread *within* one arm), and 0.60–1.00 at rank-2 random (≈12–20× the hand-set value).
The within-arm spread is exactly what a single `β` cannot represent, and the random-arm
values explain why `β = 0.05` wasted R1's budget there: the useful step in those
directions was near pCN's maximum, so R1 was taking 1/20 of a step and calling it
acceptance (98 %).

**Result 5 — cost.**  The pilot arm spends 1 200 extra `Φ` evaluations (300 steps × 4
chains, +7.5 % of 16 004) and no extra gradients; `curv` spends `2r` evaluations.  Per
evaluation and per second, the informed rank-1 cell improves from 0.27 ESS/s to 2.24
ESS/s (8×; the seconds there are noisy because cells shared two cores) and the random
rank-1 cell from 0.07 to 4.1–6.7 ESS/s.

### R2b — rotating the scale into the estimate's eigenbasis (same day)

Same anchor, same 4 chains × 4 000 (burn 500, thin 5), all four arms in the **same
process** so they share the anchor and the runtime.  The diagonal rows reproduce the
earlier table exactly (rank-2 random `curv` ESS 1 481, `pilot` 59.3 %), which is the
harness's determinism check; the rotated arms use their own stream offsets
(`0x6_000`/`0x8_000`), so they are independent replicates rather than re-used draws.

| rank | basis | arm | accept | ESS(Φ) | `R̂(Φ)` | ESS/1k evals | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | random | curv | 52.8 % | 1 481 | 1.002 | 92.5 | 4.30 % | 0.000 | 1.0955 | 4.1498 | 0.325 | 0.0001 | 3.69 | 1.00 |
| 2 | random | pilot | 59.3 % | 1 769 | 1.000 | 102.8 | 4.30 % | 0.000 | 1.0955 | 4.1499 | 0.325 | 0.0001 | 5.22 | 0.60–1.00 |
| 2 | random | rotcurv | 51.7 % | 1 808 | 1.001 | 112.9 | 4.30 % | 0.000 | 1.0955 | 4.1498 | 0.325 | 0.0001 | 3.64 | 1.00 |
| 2 | random | rotpilot | 58.7 % | 1 249 | 1.000 | 72.6 | 4.30 % | 0.000 | 1.0955 | 4.1498 | 0.325 | 0.0001 | 5.53 | 0.34–1.00 |
| 2 | informed | curv | 0.1 % | 7 | 1.671 | 0.4 | 11.33 % | 0.501 | 1.0562 | 3.6196 | 0.268 | 0.0145 | 0.03 | 4.3e-3–1.00 |
| 2 | informed | pilot | 44.5 % | **152** | **1.012** | **8.8** | 15.62 % | 0.046 | 0.9971 | 2.9123 | 0.272 | 0.0004 | 0.17 | 0.017–0.13 |
| 2 | informed | rotcurv | 0.1 % | 6 | 1.947 | 0.4 | 11.33 % | 0.996 | 1.0577 | 3.6051 | 0.267 | 0.0196 | 0.03 | 4.2e-3–1.00 |
| 2 | informed | rotpilot | 48.7 % | 29 | 1.096 | 1.7 | 15.62 % | 0.048 | 0.9974 | 2.9119 | 0.272 | 0.0010 | 0.15 | 0.016–0.13 |
| 4 | random | curv | 51.5 % | 1 646 | 1.000 | 102.8 | 4.30 % | 0.000 | 1.0953 | 4.1480 | 0.325 | 0.0001 | 2.75 | 0.75–1.00 |
| 4 | random | pilot | 52.4 % | 1 548 | 1.001 | 90.0 | 4.30 % | 0.000 | 1.0952 | 4.1480 | 0.325 | 0.0001 | 3.82 | 0.49–1.00 |
| 4 | random | rotcurv | 55.2 % | 1 916 | 1.000 | 119.5 | 4.30 % | 0.000 | 1.0953 | 4.1482 | 0.325 | 0.0001 | 2.63 | 0.71–1.00 |
| 4 | random | rotpilot | 56.8 % | 1 257 | 1.002 | 73.1 | 4.30 % | 0.000 | 1.0953 | 4.1481 | 0.325 | 0.0001 | 6.22 | 0.31–1.00 |
| 4 | informed | curv | 0.1 % | 4 | 2.327 | 0.2 | 8.98 % | NaN | 1.0602 | 3.6282 | 0.290 | 0.0169 | 0.03 | 2.9e-3–1.00 |
| 4 | informed | pilot | 50.4 % | **15** | **1.353** | **0.9** | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 0.17 | 0.012–0.13 |
| 4 | informed | rotcurv | 0.0 % | 5 | 3.087 | 0.3 | 10.94 % | NaN | 1.0572 | 3.5249 | 0.277 | 0.0204 | 0.03 | 2.3e-3–1.00 |
| 4 | informed | rotpilot | 50.1 % | 9 | 2.893 | 0.5 | 12.89 % | 0.296 | 0.9911 | 2.8739 | 0.251 | 0.0371 | 0.21 | 0.014–0.16 |

**Result 6 — the rotation cannot rescue a point estimate that is flat where it is
measured.**  Informed `rotcurv` behaves like `curv`: 0.1 % / 0.0 % acceptance, ESS 6 and 5,
`R̂` 1.947 and 3.087, NLL 3.6051 and 3.5249 against the diagonal arms' 3.6196 and 3.6282 —
all three numbers far from the converged 2.91.  The printed scales show why: `m = 0.03`
with `β_i` still hitting the `1` cap, exactly the diagonal arm's signature.  The reported
scales (β from 2.3e-3 to 1 at `m = 0.03`) put the informed Hessian's eigenvalues between
roughly 1e-3 and 1.7e2: the rotation does *order* them, by five decades, but the `β ≤ 1`
cap still binds across the whole flat end and one `m` still has to cover the rest.  The
failure is in the magnitude of the estimate at a nearly-uniform anchor, and a change of
basis cannot manufacture information the estimate did not have.

**Result 7 — for the measured widths the rotation is worse, and only in mixing.**  The
rotated pilot does not have an acceptance problem — 48.7 % vs 44.5 % at rank 2 and 50.1 %
vs 50.4 % at rank 4 — and still has lower ESS at both ranks: 29 vs 152 at rank 2, 9 vs 15
at rank 4, with `R̂` 1.096 vs 1.012 and 2.893 vs 1.353.  At rank 4 the rotated arm is not converged, so its
NLL (2.8739) is not usable and its MCSE(NLL) 0.0371 is 34× the diagonal pilot's 0.0011;
at rank 2, where both are usable, the predictive is unchanged (NLL 2.9119 vs 2.9123,
inside the 0.0010 MCSE, accuracy 15.62 % both).  So the target is still right — as the
exactness argument requires — and the cost is entirely diffusion: an accepted rotated
step changes `Φ` less than an accepted diagonal step.  The `β_i` ranges are nearly
identical (1.6e-2–0.13 vs 1.7e-2–0.13 at rank 2), which locates the problem in the
*orientation*, not the widths: 150 correlated pilot draws estimate the marginal scales
well enough, but the eigen*vectors* of a 2–4 dimensional sample covariance whose
eigenvalues are close are much noisier, and the joint step is then spent along the pilot's
least reliable mixture.  This is a hypothesis consistent with the data, not a measured
decomposition; separating it needs a longer pilot or a shrinkage estimator, which this
cycle did not run.

**Result 8 — the controls say the same thing.**  On the random (prior-dominated) basis
`rotcurv` is modestly better (1 808 vs 1 481 ESS at rank 2, 1 916 vs 1 646 at rank 4) and
`rotpilot` modestly worse (1 249 vs 1 769, 1 257 vs 1 548), with the predictive identical
to four decimals in every pair (NLL 4.1498 vs 4.1498, 4.1481 vs 4.1480).  Where the target
is the prior, no rotation of it can matter, and none does; where it matters, the joint
rescaling did not find a better proposal than one `β_i` per coordinate at these ranks.

**Result 9 — what the rotated code establishes even though it did not help.**  The kernel
is exact for any orthogonal `Q` (asserted every step), reduces to the diagonal kernel
bitwise at the identity, and samples a *correlated* Gaussian's full covariance correctly
(`H⁻¹` recovered to ±0.02 with ESS > 2 000).  The block form is the same code with `Q`
block-diagonal; at the restricted target the block is the whole space, so this grid cannot
distinguish block from full rotation.  The honest summary is that joint rescaling is
implemented, verified, and **not** an improvement here: the rank-2 informed pilot is
5× *worse* in ESS, and the curvature arm fails for a reason rotation cannot address.

### R2c — the full target: a `B`/`B_perp` split with the full likelihood (same day)

R2 and R2b sample `π_r`, a rank-`r` model.  The ambient target is not that model: it is
`π_P ∝ N(0, I_P)(v)·exp(−λN·mean_loss(θ_ref + σv))` over all `P = 16 672` whitened
coordinates, the same potential `task::mh_pcnl_chain`'s exact drift-free arm uses.  R2c
keeps the informed basis and drops the restriction: the basis proposes, it no longer
defines the support.

* `FullTarget` (`src/target.rs`) wraps the existing `phi_parts`, so this is the ambient
target rather than a second derivation of it.  A point is `v = Bz + w` with
`w ⊥ span(B)`: `AffineSubspace::complement` returns `Π = I − BBᵀ`, and `lift`/`lift_tensor`
map `(z, w)` to `θ = θ_ref + σv`.  The prior term is `½‖v‖²` over all `P` coordinates.
* `mh_pcn_z_full` — the informed move with the **full** potential,
  `z' = √(1−β²)·z + β·ξ`, `log α = Φ(v) − Φ(v') + ½‖v'‖² − ½‖v‖²`.  Exact for any
  `β ∈ (0, 1]` for the same reason as R2's diagonal kernel: the target enters only
  through `Φ`, and the proposal is prior-preserving by construction.
* `mh_pcn_w_full` — the complement move, `w' = √(1−β_q²)·w + β_q·Πξ`, same ratio.  `w`
  starts at 0 and the *driving noise* is projected, not the proposal; projecting `w'`
  would make the proposal a non-Gaussian with a different density and break the ratio.  A
  small numerical leak remains (`‖Πw − w‖/‖w‖ ≈ 1e-7` over a run) because the projection
  is applied once per step rather than re-orthogonalised.
* One step is `z` then `w`.  Each stage is `π_P`-invariant on its own (it is pCN on a
  block with the other block fixed, scored with the full likelihood), so the composition
  is too, and the two acceptances are separate adaptation signals.  `β_q` is a scalar
  adapted with `adapt_beta` — `DiagScale`'s Robbins–Monro on `log m`, re-used — toward
  `cfg.adapt_target = 0.5` during burn only, then frozen, so the kept window is one fixed
  kernel.

Two tests carry the exactness argument, because acceptance rates cannot: with a potential
that does not depend on `w`, the `z` stage is **bitwise** R2's diagonal kernel
(`full_z_move_with_a_w_blind_target_equals_the_diagonal_kernel`), and on the coupled
Gaussian `Φ(v) = ½vᵀHv` with `B = e₀` (`v = (z, w₁)`), `H = [[1.6, −0.4], [−0.4, 1.6]]`,
the two-stage kernel recovers `Σ = [[0.6, −0.4], [−0.4, 0.6]]` at ESS > 2 000, off-diagonal
included (`two_stage_full_kernel_samples_the_coupled_gaussian`).  The first draft of that
test read `w[0]`, which is identically zero under `B = e₀`, and reported `Var(z) = 0.334`
with a healthy 33 % acceptance: the kernel was right and the test's index was wrong.  This
is recorded because it is the failure mode these tests exist for.

Ambient anchor (`4.30 % / 4.1523`, `0x52d503556818f1e0`), `P = 16 672`, `λ = 1`,
`σ = 0.35`, `N = 512/256`, 4 chains × 4 000 steps, burn 500, thin 5 (2 000 draws per
chain).  All sixteen cells share that budget.  The restricted half was first run at
3 000 steps (2 000 draws per chain, `12 004`/`13 204` evaluations) and its rows in the
first version of this table were labelled 4 000; it was rerun at the full budget and the
rows below are the rerun's.  That rerun is also the determinism check across a third
process: it reproduces R2's rank-1/2 rows and R2b's rank-4 pilot row exactly (ESS 58/680,
31/152, 15; NLL 2.9462/2.9508, 2.9083/2.9123, 2.8658; `R̂` 1.055/1.004, 1.098/1.012,
1.353), so the arm comparison is not a comparison of budgets.  `q acc/β_q`
is the complement stage's acceptance at its frozen scale (`—` for `π_r`, which has no
complement); `s` is wall time on the 2-core box where the two processes shared it, so it
is a cost ratio, not a benchmark:

| rank | basis | target | prop | z acc | q acc/β_q | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | evals | s |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | informed | `π_r` | fixed | 76.7 % | — | 58 | 1.055 | 13.28 % | 0.187 | 1.0127 | 2.9462 | 0.264 | 0.0040 | 16 004 | 564 |
| 1 | informed | `π_r` | pilot | 47.1 % | — | **680** | **1.004** | 13.28 % | 0.070 | 1.0124 | 2.9508 | 0.266 | 0.0002 | 17 204 | 556 |
| 2 | informed | `π_r` | fixed | 71.8 % | — | 31 | 1.098 | 14.06 % | 0.049 | 0.9973 | 2.9083 | 0.261 | 0.0068 | 16 004 | 289 |
| 2 | informed | `π_r` | pilot | 44.5 % | — | 152 | 1.012 | 15.62 % | 0.046 | 0.9971 | 2.9123 | 0.272 | 0.0004 | 17 204 | 304 |
| 4 | informed | `π_r` | fixed | 60.6 % | — | 100 | 1.023 | 12.89 % | 0.089 | 0.9925 | 2.8699 | 0.270 | 0.0035 | 16 004 | 282 |
| 4 | informed | `π_r` | pilot | 50.4 % | — | 15 | 1.353 | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 17 204 | 301 |
| 2 | random | `π_r` | fixed | 98.5 % | — | 36 | 1.170 | 4.30 % | 0.000 | 1.0958 | 4.1508 | 0.325 | 0.0009 | 16 004 | 562 |
| 2 | random | `π_r` | pilot | 59.3 % | — | **1 769** | **1.000** | 4.30 % | 0.000 | 1.0955 | 4.1499 | 0.325 | 0.0001 | 17 204 | 557 |
| 1 | informed | `π_P` | fixed | 87.9 % | 21.9 %/0.08 | 8 | 1.602 | 19.14 % | 0.454 | 0.8674 | 1.9505 | 0.170 | 0.0700 | 24 004 | 524 |
| 1 | informed | `π_P` | pilot | 49.2 % | 45.1 %/0.05 | 10 | 1.319 | 14.45 % | 0.958 | 0.8618 | 1.9163 | 0.173 | 0.0654 | 26 404 | 570 |
| 2 | informed | `π_P` | fixed | 84.3 % | 24.5 %/0.08 | 8 | 1.467 | 20.31 % | 0.661 | 0.8396 | **1.8313** | 0.101 | 0.0704 | 24 004 | 483 |
| 2 | informed | `π_P` | pilot | 51.2 % | 43.9 %/0.05 | 7 | 1.620 | 20.31 % | 0.674 | 0.8834 | 2.0299 | 0.125 | 0.0684 | 26 404 | 509 |
| 4 | informed | `π_P` | fixed | 76.5 % | 27.0 %/0.08 | 6 | 1.847 | 14.84 % | 1.126 | 0.8764 | 1.9565 | 0.172 | 0.0763 | 24 004 | 390 |
| 4 | informed | `π_P` | pilot | 47.8 % | 41.9 %/0.05 | 5 | 3.350 | 14.45 % | 1.284 | 0.8552 | 1.9261 | 0.171 | 0.0646 | 26 404 | 422 |
| 2 | random | `π_P` | fixed | 99.2 % | 28.1 %/0.08 | 6 | 1.748 | 16.80 % | 1.506 | 0.8648 | 1.9740 | 0.145 | 0.0724 | 24 004 | 390 |
| 2 | random | `π_P` | pilot | 79.1 % | 40.7 %/0.05 | 6 | 1.796 | 17.58 % | 1.270 | 0.8699 | 2.0359 | 0.155 | 0.0659 | 26 404 | 423 |

**Result 10 — the full target's predictive is better than the restricted model's at every
matched cell.**  NLL 1.83–2.03 against 2.87–2.95, Brier 0.84–0.88 against 0.99–1.01, ECE
0.10–0.17 against 0.26–0.27 (the anchor's own predictive is 4.1523 / 1.0960 / 0.325).  All
six informed `π_P` cells beat all six informed `π_r` cells, and the gaps (0.84–1.11 NLL,
0.11–0.17 Brier) are 12–14× the largest NLL MCSE in either grid (0.0763) and 6–17× the
largest Brier MCSE (0.0170).  The restricted arms were not
broken by this test — rank 4 pilot has ESS 15 / `R̂` 1.353 and still lands at NLL 2.8658,
within 0.09 of its rank-1 partner — so what bounds them is the restriction, not the rank
or the mixing.  Accuracy moves much less (12.9–15.6 % vs 14.5–20.3 %, overlapping at
MCSE 0.45–1.28); the gain is in the probability score, not in top-1.

**Result 11 — with the full target the informed basis stops mattering.**  `π_P` on a
*random* rank-2 basis gives NLL 1.9740 / 2.0359, Brier 0.8648 / 0.8699, accuracy 16.80 /
17.58 % — inside ~2 MCSE of the informed `π_P` cells at the same rank (NLL 1.8313–2.0299,
Brier 0.8396–0.8834, accuracy 20.31 %).  On `π_r` the same control is the anchor itself
(NLL 4.1499 / 4.1508, accuracy 4.30 %, `R̂` 1.000 at ESS 1 769).  So the informed PCA buys
a `z` step schedule in `π_P`, not the answer: the complement supplies the directions the
basis misses, which is what a full-support kernel must do.  The consequence for R2's
control design is that "informed beats random" is evidence about the *restricted* target
only; for `π_P` a random basis is not a weaker arm and should not be read as one.

**Result 12 — the complement is data-constrained, and the adaptation says so.**  The `w`
stage starts at `β_q = 0.5` and is adapted down during burn in every cell: it freezes at
0.08 in the fixed-`z` cells (acceptance 21.9–28.1 %) and at 0.05 in the pilot cells
(acceptance 40.7–45.1 %, which get 300 extra adapting steps from the pilot).  The
complement's conditional is therefore far narrower than the prior along these directions —
a pCN at prior scale would be proposing into a likelihood wall — and the scalar `β_q` finds
that scale without a Hessian.  The `z` stage, by contrast, accepts 76.5–87.9 % at the
hand-set `β = 0.05`: the informed directions are shallow, which is consistent with R2's
finding that `curv` reads them as wider than the prior at this anchor.

**Result 13 — cost and mixing, stated against the claim.**  `π_P` spends two `Φ`
evaluations per step (24 004 / 26 404 against 16 004 / 17 204) and 0.7–2.0× the wall time
(390–570 s against 282–564 s; the restricted cells' time depends strongly on rank because
their basis changes size, and the two-process timings are not a benchmark).  Its ESS(Φ) is
5–10 with `R̂` 1.32–3.35, against 15–680 for
the restricted cells, so `MCSE(NLL)` is 0.065–0.076 rather than ~0.001.  ESS across
targets is not a like-for-like efficiency number — the potentials differ in dimension and
scale — but within `π_P` it is the binding limitation, and it means Result 10 is a claim
about a 0.9-NLL *gap*, not about the full target's NLL to two decimals.  The `π_P`
accuracy column is correspondingly noisy (MCSE 0.45–1.51): its ranking across ranks and
proposals is not resolvable at this budget, and is not used.

**Result 14 — tripling one cell's budget moves the predictive by about one MCSE, and
`Φ`'s mixing not at all.**  Informed rank-2 fixed at 12 000 steps (burn 1 000, thin 5,
8 800 draws, 96 004 evaluations, 1 752 s) gives NLL 1.7559 (MCSE 0.0315), Brier 0.8232
(0.0101), accuracy 21.48 %, ECE 0.092 — against the 4 000-step cell's 1.8313 (0.0704) /
0.8396 (0.0138) / 20.31 % / 0.101.  The NLL difference is 0.075, ≈1.1× the shorter run's
own MCSE, and the Brier difference 0.016 is ≈1.2× its MCSE, so the two runs are consistent;
if anything the longer chain scores *better*, which means the 4 000-step grid does not
flatter `π_P`.  Mixing inside the potential is what does not improve: ESS(Φ) 10 against 8,
`R̂` 1.293 against 1.467, `β_q` 0.06 against 0.08 (twice the burn).  An ESS that grows by
25 % when the chain triples in length is not an equilibrium sample — the effective count is
set by one slow collective mode of the complement, not by the number of steps — so the
`π_P` cells are best read as transient averages that have settled in their *predictive*
while still relaxing in `Φ`.  Length, not more chains, is what would pin the absolute
numbers, and the complement proposal is the lever that would make length affordable: a
scalar pCN decorrelates `Φ` at ≈0.1 per 1 000 evaluations even after adaptation, which is
the measured reason Result 12's narrow conditional is not the whole story.

**Result 15 — this changes the plan's reading, in the target's favour.**  R2 said that if
the ambient posterior predicts the base rate, accurately sampling it is a negative result
about the target rather than about the kernel.  That is not what happened here: sampled
properly, `π_P` has the best probability score of anything measured in R2 (Brier
0.84–0.88, NLL 1.83–2.03, and 0.8232 / 1.7559 in the 12 000-step cell) against the
restricted model's 0.99–1.01 / 2.87–2.95 and the
anchor's 1.096 / 4.15, while accuracy stays at 15–20 %.  So R1/R2's "is the restriction the
limit?" question is answered *yes* at `σ = 0.35`, `λ = 1`, `N = 512`: the restricted-model
predictive was the artifact, and the ambient posterior is not merely defensible, it scores
better.  Nothing here establishes a base-rate claim, and the low ESS means the absolute
numbers should be re-measured before being quoted as posterior summaries.

**(f) What R2 has not done.**  `π_P` at 4 000 steps has ESS(Φ) 5–10 and `R̂` up to 3.35, so
the full-target *absolute* predictives are not converged, even though the 0.9-NLL gap is
far outside their error bars; Result 14 is one cell, not a systematic convergence study.
The complement proposal is still a scalar pCN: nothing here tests per-coordinate or
Hessian-informed `β_q`, which Result 12 suggests is where the gain would be, and the
curvature/rotated arms are not implemented for `π_P` because the split-potential Hessian is
unvalidated.  Ranks 8+, `λ ≠ 1`, `σ ≠ 0.35`, and the SG-bias comparison remain untouched;
`pilot` still measures width in the R1 metric, so a badly chosen `σ` limits it from below,
and its bias direction is safe for acceptance but is not a posterior-width estimate with an
error bar.  There is no adaptive-MCMC claim: the adaptation freezes after burn and the
pilot draws are discarded.

### R2d — does the pilot's ESS gain survive more dimensions? (same day)

R2's headline is one rank (ESS 58 → 680 at rank 1) and R2b adds the rank-2 gain (31 →
152) — while its rank-4 pilot row (ESS 15) already sits below the rank-4 fixed arm that
this section measures at ESS 100, i.e. a *loss*.  That makes the scaling a measured
question rather than an extrapolation: the same 300-step pilot and the same fixed arm are run at
ranks 1, 2, 4 and 8 on the informed basis at one budget, plus three probes that separate
"more dimensions" from the two other explanations — a noisy width estimate, and
prior-dominated directions.  The probes are a random rank-8 control, a 1 200-step pilot at
rank 8, and `rotpilot` (R2b's eigenbasis form of the pilot widths) at ranks 4 and 8.

4 chains × 4 000 steps, burn 500, thin 5 (2 000 draws per chain), ambient anchor
(`4.30 % / 4.1523`, `0x52d503556818f1e0`), target `π_r`.  Ranks 1, 2 and 4 are one process
`/tmp/pcnl_r2f_inf.txt`; rank 8 is `/tmp/pcnl_r2d_inf.txt`.  The rank-1/2 rows reproduce
R2's table and the rank-4 rows reproduce R2b's pilot row exactly, so the rows are again a
cross-process determinism check:

| rank | prop | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` | evals | s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | fixed | 76.7 % | 58 | 1.055 | 13.28 % | 0.187 | 1.0127 | 2.9462 | 0.264 | 0.0040 | — | 0.05 | 16 004 | 564 |
| 1 | pilot | 47.1 % | **680** | **1.004** | 13.28 % | 0.070 | 1.0124 | 2.9508 | 0.266 | 0.0002 | 0.18 | 0.12–0.13 | 17 204 | 556 |
| 2 | fixed | 71.8 % | 31 | 1.098 | 14.06 % | 0.049 | 0.9973 | 2.9083 | 0.261 | 0.0068 | — | 0.05 | 16 004 | 289 |
| 2 | pilot | 44.5 % | 152 | 1.012 | 15.62 % | 0.046 | 0.9971 | 2.9123 | 0.272 | 0.0004 | 0.17 | 0.017–0.13 | 17 204 | 304 |
| 4 | fixed | 60.6 % | 100 | 1.023 | 12.89 % | 0.089 | 0.9925 | 2.8699 | 0.270 | 0.0035 | — | 0.05 | 16 004 | 282 |
| 4 | pilot | 50.4 % | 15 | 1.353 | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 0.17 | 0.012–0.13 | 17 204 | 301 |
| 8 | fixed | 55.4 % | 7 | 1.289 | 10.94 % | 0.658 | 0.9485 | 2.4693 | 0.237 | 0.0527 | — | 0.05 | 16 004 | 230 |
| 8 | pilot | 48.1 % | 6 | 1.931 | 14.06 % | 0.594 | 0.9768 | 2.7792 | 0.280 | 0.0164 | 0.20 | 0.013–0.12 | 17 204 | 225 |

| rank | basis | prop | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` | evals | s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | random | fixed | 96.2 % | 28 | 1.122 | 4.30 % | 0.000 | 1.0957 | 4.1468 | 0.325 | 0.0010 | — | 0.05 | 16 004 | 379 |
| 8 | random | pilot | 46.3 % | **872** | **1.001** | 4.30 % | 0.000 | 1.0953 | 4.1442 | 0.325 | 0.0002 | 2.95 | 0.22–1.00 | 17 204 | 276 |
| 8 | informed | pilot 1 200 | 47.8 % | 24 | 1.091 | 14.84 % | 0.246 | 0.9758 | 2.7990 | 0.239 | 0.0099 | 0.17 | 0.023–0.12 | 20 804 | 461 |
| 4 | informed | rotpilot | 50.1 % | 9 | 2.893 | 12.89 % | 0.296 | 0.9911 | 2.8739 | 0.251 | 0.0371 | 0.21 | 0.014–0.16 | 17 204 | 578 |
| 8 | informed | rotpilot | 43.3 % | 4 | 4.189 | 14.45 % | 0.468 | 0.9628 | 2.7668 | 0.240 | 0.0511 | 0.24 | 0.0085–0.17 | 17 204 | 379 |

**Result 16 — the ESS gain decays with rank and is gone by rank 4.**  ESS(pilot)/ESS(fixed)
is 680/58 = **11.7×** at rank 1, 152/31 = **4.9×** at rank 2, 100/15 = **0.15×** at rank 4
and 7/6 = **0.86×** at rank 8.  The series is not monotone in `r` — rank 8 gains back most
of what rank 4 lost — so "more dimensions" is not the whole story, and the rank-8 number
is not a measured tie either (Result 17).  Where both cells are usable, at rank 4, the
pilot's 6.7× ESS loss buys no measurable predictive gain: NLL 2.8658 vs 2.8699 (0.0041,
1.1σ of the 0.0035 fixed-arm MCSE), Brier 0.9917 vs 0.9925 (0.0008, 1.0σ of 0.0005),
accuracy 12.89 % both.  So the loss is diffusion, not bias — the same conclusion R2b
reached for the rotated arm — but it is a real cost at these ranks.

**Result 17 — rank 8 is unanswerable at 300 pilot steps, in both directions.**  The fixed
arm has ESS 7 / `R̂` 1.289 and the pilot ESS 6 / `R̂` 1.931, so neither kept window is a
sample of `π_r`.  Their predictives differ a lot — NLL 2.4693 vs 2.7792 (5.6σ on the stated
MCSEs), Brier 0.9485 vs 0.9768 (4.9σ), accuracy 10.94 % vs 14.06 % — and that is exactly
why the gap is not quoted: at `R̂` ≥ 1.29 the MCSEs describe within-chain noise, not the
distance to the target.  What the cell does establish is negative: the rank-8 informed
`π_r` predictive is not measurable at this budget by either proposal.

**Result 18 — dimension alone does not defeat the pilot: the random rank-8 control gains
31×.**  28 → **872** ESS with `R̂` 1.122 → 1.001, at `m = 2.95` and `β_i` 0.22–1.00, and
the predictive is unchanged (NLL 4.1468 vs 4.1442, Brier 1.0957 vs 1.0953, accuracy
4.30 % both; differences inside MCSE 0.0010 / 0.0002).  This is larger than the rank-1 and
rank-2 informed gains, i.e. at 8 dimensions the proposal still does its job when the
directions are prior-dominated and the estimated widths are near `β ≈ 1`.  Whatever breaks
at informed rank 4/8 is therefore not the number of coordinates.

**Result 19 — pilot length is the binding factor.**  At rank 8 informed, keeping everything
else fixed and giving the pilot 1 200 steps instead of 300 (600 estimation draws instead of
150) raises ESS from 6 to **24** — 4× the 300-step pilot and 3.4× the fixed arm — and pulls
`R̂` from 1.931 to **1.091**, for 3 600 extra evaluations (20 804 vs 17 204, +21 %) and
461 s.  The widths themselves barely move (`β_i` 0.013–0.12 → 0.023–0.12, `m` 0.20 →
0.17), so the improvement is not a different scale but a less noisy one: the pilot's 150
second-half draws are 150 per coordinate at rank 1, 38 at rank 4 and 19 at rank 8, and the
informed directions at higher rank are estimated from too few draws to place the per-
coordinate steps well.  The 1 200-step pilot's predictive does not move toward the fixed
arm (NLL 2.7990 vs 2.7792, 1.0σ), so length fixes mixing without settling the rank-8
predictive — consistent with Result 17's reading that the fixed arm at ESS 7 is not the
reference either.

**Result 20 — re-orienting the widths does not rescue them either.**  `rotpilot` at rank 4
is ESS 9 / `R̂` 2.893 and at rank 8 ESS 4 / `R̂` 4.189, both worse than the diagonal pilot's
15 / 1.353 and 6 / 1.931.  Combined with R2b's rank-2 result (29 vs 152), the eigenbasis
form is worse at every rank where it has been run; the diagonal pilot is the better
proposal, and neither the orientation nor the pilot's step count in the *rotation* is the
lever — the length of the estimate is.

**Result 21 — what this changes in R2's claim.**  R2's ESS gain is a rank-1/2 statement: it
is 11.7× and 4.9× there, 0.15× at rank 4 and unresolvable at rank 8.  The mechanism is
intact — the random rank-8 control gains 31×, and a longer pilot restores 4× at informed
rank 8 — so the pilot is not rank-limited in principle, but its default budget (300 steps)
is: the number of estimation draws per dimension falls as 150/`r`, and at `r = 8` the
estimate is too noisy to place eight scales.  The cost rule the data supports is that pilot
steps should grow with rank (≥ 1 200 at rank 8, i.e. 75 draws per dimension after the
halving) rather than stay at 300; R2e tests both halves of that sentence and finds the
constant is not 150 but that shrinkage removes the need for length at all.  Tables, not the prose,
should be updated if this is re-run: the `π_P` cells in R2c are unaffected (they use the
full target and its own complement move).

### R2e — pilot length vs width shrinkage (same day)

R2d left two follow-ups: whether the rule should be 150·`r` pilot steps (1 200 at rank 8),
and whether the rank-4 loss is the *magnitude* of the pilot widths or their *noise*.  This
section answers both at once, because they are the same budget question.  Length is probed
at rank 8 (600, 2 400) and rank 4 (600, 1 200); noise is probed with a shrinkage estimator
that leaves the 300-step pilot alone and only damps its per-coordinate spread:
`log w_i ← mean(log w) + s·(log w_i − mean(log w))`, so `s = 1` is the raw pilot (bitwise),
`s = 0` is isotropic at the pilot's geometric mean, and every `s` leaves the kernel
`β_i`-reversible and the target untouched.  The geometric-mean form is deliberate: it
preserves the overall step size and varies only the spread across coordinates, which is
what separates "wrong scale" from "noisy scale".

4 chains × 4 000 steps, burn 500, thin 5, informed basis, target `π_r`, ambient anchor
(`4.30 %` base, `4.1523`, `0x52d503556818f1e0`), one process per cell (8 cells, 1.6 h on 2
cores).  Shrinkage cells use the default 300-step pilot, so all of their extra cost over
fixed is 1 200 evaluations; the length cells cost `4 · pilot` more.

| rank | pilot steps | `s` | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` | evals | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | fixed | — | 60.6 % | 100 | 1.023 | 12.89 % | 0.089 | 0.9925 | 2.8699 | 0.270 | 0.0035 | — | 0.05 | 16 004 | 280 |
| 4 | 300 (R2d) | 1 | 50.4 % | 15 | 1.353 | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 0.17 | 0.012–0.13 | 17 204 | 301 |
| 4 | 600 | 1 | 48.4 % | **221** | 1.016 | 12.89 % | 0.145 | 0.9925 | 2.8669 | 0.271 | 0.0005 | 0.12 | 0.017–0.13 | 18 404 | 318 |
| 4 | 1 200 | 1 | 50.3 % | **245** | 1.021 | 12.89 % | 0.080 | 0.9923 | 2.8668 | 0.270 | 0.0005 | 0.30 | 0.026–0.12 | 20 804 | 358 |
| 4 | 300 | 0.5 | 47.9 % | **399** | 1.012 | 12.89 % | 0.081 | 0.9924 | 2.8669 | 0.270 | 0.0005 | 0.27 | 0.035–0.11 | 17 204 | 291 |
| 4 | 300 | 0.0 | 46.5 % | **383** | 1.010 | 12.89 % | 0.057 | 0.9922 | 2.8662 | 0.270 | 0.0005 | 0.33 | 0.069–0.08 | 17 204 | 299 |
| 8 | fixed | — | 55.4 % | 7 | 1.289 | 10.94 % | 0.658 | 0.9485 | 2.4693 | 0.237 | 0.0527 | — | 0.05 | 16 004 | 281 |
| 8 | 300 (R2d) | 1 | 48.1 % | 6 | 1.931 | 14.06 % | 0.594 | 0.9768 | 2.7792 | 0.280 | 0.0164 | 0.20 | 0.013–0.12 | 17 204 | 225 |
| 8 | 600 | 1 | 46.7 % | 13 | 1.225 | 14.84 % | 0.187 | 0.9792 | 2.8219 | 0.252 | 0.0077 | 0.15 | 0.018–0.13 | 18 404 | 246 |
| 8 | 1 200 (R2d) | 1 | 47.8 % | 24 | 1.091 | 14.84 % | 0.246 | 0.9758 | 2.7990 | 0.239 | 0.0099 | 0.17 | 0.023–0.12 | 20 804 | 461 |
| 8 | 2 400 | 1 | 47.6 % | **137** | 1.040 | 14.84 % | 0.126 | 0.9751 | 2.8025 | 0.238 | 0.0058 | 0.29 | 0.037–0.13 | 25 604 | 321 |
| 8 | 300 | 0.5 | 47.1 % | 7 | **2.164** | 14.84 % | 0.355 | 0.9727 | 2.7425 | 0.262 | 0.0160 | 0.29 | 0.035–0.11 | 17 204 | 299 |
| 8 | 300 | 0.0 | 48.7 % | **154** | **1.010** | 14.84 % | 0.277 | 0.9780 | 2.8105 | 0.253 | 0.0087 | 0.34 | 0.064–0.07 | 17 204 | 231 |

Every `fixed` row reproduces R2d's fixed row bitwise (rank 4: 60.6 % / 12.89 % / 0.9925 /
2.8699; rank 8: 55.4 % / 10.94 % / 0.9485 / 2.4693), and the `300 / s = 1` rows reproduce
R2d's pilot rows, so the eight cells are eight independent determinism checks.

**Result 22 — the ESS gain is monotone in pilot length, but 150·`r` is not the knee at
rank 8.**  Rank 8 informed goes 6 → 13 → 24 → **137** at 300 / 600 / 1 200 / 2 400 steps
(`R̂` 1.931 → 1.225 → 1.091 → **1.040**), so R2d's length mechanism is confirmed over a 4×
range of budgets.  The increments are not proportional: 300 → 1 200 (4× the draws) buys
4×, but 1 200 → 2 400 (2× more) buys 5.7×.  The curve is convex in log-budget, so there is
no single "enough" point at this budget — 150·`r` = 1 200 sits on the steep part, and the
first cell that clears `R̂` < 1.05 is 2 400 (300·`r`).  Verified at rank 4 that the rule is
not too *small* there: 600 steps (150·`r`) takes ESS 15 → **221** and `R̂` 1.353 → 1.016,
and 1 200 adds only 245.  So rank 4's knee is at or below 150·`r`, rank 8's is at 300·`r`:
the per-rank constant is not 150, and it rises with `r`.

**Result 23 — at rank 4 the pilot's loss is width noise, not width magnitude.**  The
shrinkage ladder at the *same* 300-step pilot is 15 (`s = 1`) → **399** (`s = 0.5`) →
**383** (`s = 0`), with `R̂` 1.353 → 1.012 → 1.010.  Three things follow.  (i) The damage is
essentially all in the spread across the four coordinates: damping it recovers R2d's loss
and then some — the best rank-4 cell in either section is a pilot cell at 399 vs the fixed
arm's 100, a **4.0×** gain.  (ii) *Extra length and shrinkage are substitutes*: 300 shrunk
draws (399, 17 204 evals) beat 600 (221) and 1 200 (245) raw draws at 18 404 and 20 804,
so the noise is worth more than 4× the draws — the pilot's problem at rank 4 is that 75
second-half draws per coordinate give a per-coordinate width noisier than the width is
informative.  (iii) `s = 0.5` and `s = 0` are indistinguishable here (399 vs 383, `R̂`
1.012 vs 1.010, identical NLL to 7e-4), so at rank 4 the per-coordinate information
surviving 75 draws is worth nothing over a single common scale: the honest summary is
"one scale, 300 draws" not "four scales, 75 draws each".

**Result 24 — at rank 8, on this seed, isotropic is the best proposal measured and partial
shrinkage is the worst.**  (R2f repeats this cell on other seeds and does not find either
ordering; read the numbers below as the default seed's, not as the rank's.)  `s = 0` gives
ESS **154** / `R̂` **1.010** — 22× the fixed arm, 6.4× the raw
300-step pilot, and better than 2 400 raw draws (137 / 1.040) at 17 204 evals instead of
25 604, i.e. at 67 % of the cost and a third of the wall time (231 s vs 321 s).  But
`s = 0.5` is the worst cell in the table: ESS 7, `R̂` **2.164** — worse than no shrinkage
at all (1.931 earlier, 1.225 at 600 steps).  The response in `s` is therefore non-monotone
at rank 8, and the intermediate point is not near either end (a linear interpolation of ESS
between 6 and 154 would give ~80; the measured value is 7).  One cell cannot separate a
mechanism from pilot-draw luck — the widths differ run to run because `--seed` moves the
pilot stream, and this is the only cell where a repeat is the obvious next probe — but the
negative is usable now: **partial shrinkage is not a safe interpolation**, and if shrinkage
is turned on at rank 8 it should be taken to the isotropic end.  Note also that `s = 0`
puts `β_i` at 0.064–0.07 while `s = 0.5` spans 0.035–0.11: the intermediate case is the
only one whose largest step is ~3× its smallest, and the scale mismatch is what the
rank-8 chain cannot absorb.

**Result 25 — none of this moves the predictive.**  Across all six rank-4 pilot cells, NLL
is 2.8658–2.8669 (spread 0.0011, well inside the fixed arm's 0.0035 MCSE), Brier
0.9917–0.9925, accuracy 12.89 % in every cell, ECE 0.269–0.271.  Across the six rank-8
pilot cells, NLL 2.7425–2.8219 and BMA accuracy 14.06–14.84 % (`s = 0`: 14.84 %), while ESS
spans 6 to 154 — a **25×** range in mixing with no trend in the predictive.  This is R2d's
diffusion-not-bias reading, now established over the whole ESS range rather than at one
cell: the pilot's ESS loss at rank 4 and its noise at rank 8 cost only within-chain
variance.  The rank-8 pilot NLL (2.74–2.82) is still worse than the fixed arm's 2.4693, and
that gap remains unquoted — the fixed arm has `R̂` 1.289, so Result 17 applies unchanged.

**Result 26 — what to actually run.**  `--pilot-shrink 0` is the proposal to use at both
ranks tested: rank 4 ESS 383 vs 100 (fixed) and rank 8 ESS 154 vs 7, with `R̂` ≤ 1.010 in
both and no predictive cost; it needs no extra pilot steps at all.  (R2f revises the
magnitudes: the rank-4 gain is 2.6–3.1× on two seeds rather than 4×, and the rank-8
advantage is 2.2–22× with `s = 0.5` unstable rather than uniformly bad.)  Length should be treated
as the fallback for callers who must keep raw widths (the flag defaults to `s = 1` precisely
so the published R2/R2b/R2c/R2d tables stay reproducible): with raw widths, rank 8 needs
≥ 2 400 steps (300·`r`), not the 1 200 that 150·`r` gives.  Two things are deliberately not
claimed.  First, the default is not changed, because R2d's headline numbers were measured at
`s = 1` and a default switch would silently move every table in this document.  Second,
`s = 0` beating `s = 1` does not say the widths are useless in general — the random rank-8
control's widths sit near `β ≈ 1`, where shrinkage is a no-op — only that 75 draws per
coordinate at rank 4 and 19 at rank 8 are not enough to place per-coordinate scales, and
that at informal rank 8 the single isotropic scale at the pilot's geometric mean is the one
that fits.  The untested follow-up this leaves is `s` with a *replicate-averaged* width
(averaging the pilot's own half-sample estimates, or two independent pilots), which would
attack the same noise without collapsing the spread.  The `--seed` sweep this leaves open
was run immediately after (R2f below) and answers it: the non-monotonicity does not
replicate.

### R2f — the shrinkage gain across seeds (same day)

R2e ended on one cell with no explanation (rank 8, `s = 0.5`: ESS 7, `R̂` 2.164, worse than
either end) and one number standing on a single pilot stream (rank 4 raw 15 → shrunk 399).
Both are `--seed` questions: `--seed` moves the pilot stream and the chain streams together,
so a repeat at another seed is the only way to separate a mechanism from a lucky width
vector.  Four rank-8 cells at seed 7 (fixed, `s = 1`, `0.5`, `0`) plus a fifth `s = 0.5` cell
at seed 8 chase the anomaly; three rank-4 cells at seed 7 (fixed comes free with `s = 1`)
chase the magnitude.  Same protocol as R2e otherwise (4 chains × 4 000 steps, burn 500,
thin 5, informed basis, `π_r`, ambient anchor); 7 cells, 2.2 h on 2 cores, every process
`EXIT 0`.  Seed 20261020 is the default used by R2/R2b/R2c/R2d/R2e and the first four rows
below are the ones already quoted there.

| rank | seed | pilot steps | `s` | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` | evals | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20261020 | fixed | — | 60.6 % | 100 | 1.023 | 12.89 % | 0.089 | 0.9925 | 2.8699 | 0.270 | 0.0035 | — | 0.05 | 16 004 | 280 |
| 4 | 20261020 | 300 | 1 | 50.4 % | 15 | 1.353 | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 0.17 | 0.012–0.13 | 17 204 | 301 |
| 4 | 20261020 | 300 | 0.5 | 47.9 % | **399** | 1.012 | 12.89 % | 0.081 | 0.9924 | 2.8669 | 0.270 | 0.0005 | 0.27 | 0.035–0.11 | 17 204 | 291 |
| 4 | 20261020 | 300 | 0.0 | 46.5 % | **383** | 1.010 | 12.89 % | 0.057 | 0.9922 | 2.8662 | 0.270 | 0.0005 | 0.33 | 0.069–0.08 | 17 204 | 299 |
| 4 | 7 | fixed | — | 58.1 % | 84 | 1.036 | 19.53 % | 0.142 | 0.9054 | 2.6168 | 0.194 | 0.0042 | — | 0.05 | 16 004 | 281 |
| 4 | 7 | 300 | 1 | 48.6 % | 97 | 1.062 | 17.58 % | 0.085 | 0.9054 | 2.6092 | 0.212 | 0.0022 | 0.22 | 0.014–0.14 | 17 204 | 233 |
| 4 | 7 | 300 | 0.5 | 47.1 % | **264** | 1.014 | 17.58 % | 0.057 | 0.9050 | 2.6092 | 0.212 | 0.0013 | 0.33 | 0.037–0.12 | 17 204 | 300 |
| 4 | 7 | 300 | 0.0 | 47.5 % | **152** | 1.025 | 17.58 % | 0.067 | 0.9048 | 2.6068 | 0.212 | 0.0016 | 0.38 | 0.064–0.07 | 17 204 | 226 |
| 8 | 20261020 | fixed | — | 55.4 % | 7 | 1.289 | 10.94 % | 0.658 | 0.9485 | 2.4693 | 0.237 | 0.0527 | — | 0.05 | 16 004 | 281 |
| 8 | 20261020 | 300 | 1 | 48.1 % | 6 | 1.931 | 14.06 % | 0.594 | 0.9768 | 2.7792 | 0.280 | 0.0164 | 0.20 | 0.013–0.12 | 17 204 | 225 |
| 8 | 20261020 | 300 | 0.5 | 47.1 % | 7 | **2.164** | 14.84 % | 0.355 | 0.9727 | 2.7425 | 0.262 | 0.0160 | 0.29 | 0.035–0.11 | 17 204 | 299 |
| 8 | 20261020 | 300 | 0.0 | 48.7 % | **154** | **1.010** | 14.84 % | 0.277 | 0.9780 | 2.8105 | 0.253 | 0.0087 | 0.34 | 0.064–0.07 | 17 204 | 231 |
| 8 | 7 | fixed | — | 41.0 % | 9 | 1.156 | 20.70 % | 0.601 | 0.8719 | 2.4896 | 0.173 | 0.0241 | — | 0.05 | 16 004 | 275 |
| 8 | 7 | 300 | 1 | 41.8 % | 12 | 1.164 | 22.27 % | 0.275 | 0.8741 | 2.5055 | 0.167 | 0.0052 | 0.19 | 0.018–0.12 | 17 204 | 227 |
| 8 | 7 | 300 | 0.5 | 38.6 % | **20** | 1.112 | 21.88 % | 0.306 | 0.8702 | 2.4895 | 0.167 | 0.0056 | 0.25 | 0.031–0.09 | 17 204 | 295 |
| 8 | 7 | 300 | 0.0 | 35.2 % | **20** | 1.121 | 21.88 % | 0.319 | 0.8699 | 2.4895 | 0.166 | 0.0100 | 0.30 | 0.052–0.06 | 17 204 | 223 |
| 8 | 8 | fixed | — | 50.6 % | 7 | 1.336 | 28.91 % | 0.351 | 0.8096 | 1.9958 | 0.108 | 0.0225 | — | 0.05 | 16 004 | 274 |
| 8 | 8 | 300 | 0.5 | 47.3 % | 6 | **3.560** | 31.25 % | 0.199 | 0.8081 | 1.9813 | 0.114 | 0.0066 | 0.24 | 0.026–0.08 | 17 204 | 227 |

**Result 27 — at rank 4 the gain replicates and its magnitude does not.**  Seed 7 reproduces
the *ordering* of R2e exactly: `s = 0.5` is the best cell (264 vs 399 at the default seed),
`s = 0` second (152 vs 383), raw last (97 vs 15), and shrinkage beats the fixed arm at both
seeds (264 vs 84, 399 vs 100 — 3.1× and 4.0×).  What does not replicate is the size of the
raw pilot's loss: at 300 steps it is ESS 15 on one stream and 97 on another, and 97 is
*better* than the fixed arm's 84.  So R2d's and R2e's "the raw pilot is 6.7× worse at rank
4" is one unlucky width vector, not a property of a 300-step pilot at rank 4.  The robust
statement is the one that survives both seeds: **raw widths at 300 steps are a lottery that
shrinkage removes** — the shrunk cells never fall below 152 while the raw cell spans 15–97.

**Result 28 — the rank-8 non-monotonicity does not replicate, and `s = 0.5` is unstable
rather than systematically bad.**  At seed 7 the ladder is 9 (fixed) / 12 (raw) / **20**
(`s = 0.5`) / **20** (`s = 0`) — `s = 0.5` ties the best cell and beats raw, the opposite of
R2e Result 24.  But the extra `s = 0.5` cell at seed 8 gives ESS 6 with `R̂` **3.560**, the
worst diagnostic anywhere in R2/R2b/R2c/R2d/R2e/R2f (R2e's "worst" 2.164 is mild next to
it).  So the intermediate point is not systematically bad: it is *bimodal* — occasionally
the best proposal, occasionally a chain that has not mixed.  At 19 pilot draws per
coordinate a single bad width vector can stop the rank-8 chain outright, and what the sweep
measures is that variance, not a mean.  This also retracts the mechanistic reading in
Result 24 ("the scale mismatch is what the rank-8 chain cannot absorb"): seed 7 has the same
`β_i` spread 0.031–0.09 for `s = 0.5` and mixes to 20.

**Result 29 — the rank-8 isotropic advantage is real but 10× smaller off the default
seed.**  `s = 0` gives 154 vs fixed 7 at the default seed but 20 vs 9 at seed 7; raw gives
6 vs 12.  Shrinkage helps at both seeds — 20 > 12 > 9, 154 > 7 ≥ 6 — so the direction is
the one robust thing in the rank-8 table.  But "beats 2 400 raw pilot steps (137) at 67 %
of the cost", the headline of Result 24, is a default-seed statement: on seed 7 the same
cell is 20 against a fixed arm of 9, entirely inside the ±1–2 ESS scale of the fixed arm
itself.  Rank-8 conclusions need ≥ 3 seeds and a reported spread before they mean anything.

**Result 30 — the predictives are seed-dominated, so no cross-seed predictive comparison
is valid.**  Rank-4 fixed NLL is 2.8699 at the default seed and **2.6168** at seed 7 — 0.25
apart, more than 3× every within-seed difference in R2e taken together (≤ 0.08), and BMA
accuracy moves 12.89 % → 19.53 % while ECE moves 0.270 → 0.194.  Within each seed R2e
Result 25 holds again: at seed 7 the rank-4 pilot cells span NLL 2.6068–2.6092 across an
ESS range of 97–264, and every rank-8 pilot cell sits at 2.4895–2.5055 against a fixed arm
of 2.4896.  So the predictive is flat in ESS *within* a seed and dominated *by* the seed:
the pilot's mixing problem is a within-chain-variance problem and stays one, but any table
row is only comparable to rows with the same `--seed`.

**Result 31 — revised recommendation.**  `--pilot-shrink` stays on, and at rank 4 the
measured best is `s = 0.5` on both seeds (264, 399) with `s = 0` close behind (152, 383) —
the one claim in R2e whose ordering survived replication.  At rank 8 the flag is defensible
only with several seeds and the spread reported; a single rank-8 cell with shrinkage is not
evidence of anything, and `s = 0.5` there is "not safe" for a different reason than Result
24 gave: not that it is systematically worse, but that it is a coin flip between the best
and the worst cell in the whole series.  The rule this leaves: **shrinkage is a variance
reduction, so it must be validated by its variance** — a one-cell improvement at rank 8 is
what a lucky width vector looks like, and the run that reports it should report at least
three seeds.

### R2g — averaged pilot widths, and shrinkage above rank 8 (same day)

R2e/R2f left two open items.  First, the proposal widths come from **one** 300-step pilot,
so a bad width vector could be estimator *variance* that independent pilots would average
away.  Second, shrinkage had only been measured at ranks 4 and 8, so "isotropic wins
everywhere" was an extrapolation to rank 16, not a measurement.

`--pilot-reps n` now runs `n` independent pilots of `pilot_steps` each — every replicate is
restarted at the anchor `z = 0`, `w = 0` on the same RNG stream — and combines the widths
per coordinate by geometric mean in log space; the total pilot budget is
`pilot_steps × pilot_reps`.  `--pilot-reps 1` is bitwise identical to the old estimator:
the new binary and a pre-change binary were run on the same reduced config
(`--shapes quick --ranks 2 --steps 600 --burn 100 --thin 5 --chains 2 --seed 3`) and all 19
numeric columns agree, eval count and ESS included.  (The first version of the change did
not: it re-evaluated the anchor potential at the top of every replicate, adding `chains`
evals to the run — 18 408 vs 18 404 for a 2-chain config.  The replicate loop now skips
that re-evaluation on replicate 0, which is what "bitwise identical" is checking.)

Thirteen cells, 146 min on 2 cores, every process `EXIT 0`.  Protocol and ambient anchor as
R2e/R2f; the equal-budget baseline for a 300 × 2 pair (600 pilot evals) is a single 600-step
raw pilot, which is a different thing from two 300-step pilots and is why columns (B) exist.

| rank | seed | pilot | `s` | reps | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | `β_i` | evals | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20261020 | 300 | 1 | 2 | 52.2 % | 14 | 1.412 | 12.89 % | 0.455 | 0.9913 | 2.8654 | 0.269 | 0.0018 | 0.17 | 0.012–0.13 | 18 408 | 321 |
| 4 | 7 | 300 | 1 | 2 | 48.6 % | **128** | 1.034 | 17.58 % | 0.085 | 0.9050 | 2.6085 | 0.212 | 0.0011 | 0.20 | 0.016–0.14 | 18 408 | 316 |
| 4 | 8 | 300 | 1 | 2 | 46.9 % | **140** | 1.069 | 24.22 % | 0.059 | 0.8177 | 2.1322 | 0.155 | 0.0007 | 0.20 | 0.021–0.13 | 18 408 | 318 |
| 4 | 7 | 600 | 1 | 1 | 48.2 % | 99 | 1.038 | 17.58 % | 0.079 | 0.9050 | 2.6081 | 0.212 | 0.0021 | 0.13 | 0.011–0.14 | 18 404 | 244 |
| 8 | 20261020 | 300 | 1 | 2 | 45.8 % | 5 | **4.002** | 14.06 % | 0.384 | 0.9741 | 2.7682 | 0.280 | 0.0167 | 0.20 | 0.016–0.13 | 18 408 | 320 |
| 8 | 7 | 300 | 1 | 2 | 42.9 % | 7 | 1.358 | 22.27 % | 0.325 | 0.8743 | 2.5041 | 0.166 | 0.0059 | 0.19 | 0.017–0.13 | 18 408 | 319 |
| 8 | 8 | 300 | 1 | 2 | 44.8 % | 6 | 1.777 | 25.78 % | 0.541 | 0.8146 | 2.0094 | 0.158 | 0.0085 | 0.21 | 0.021–0.10 | 18 408 | 321 |
| 8 | 7 | 600 | 1 | 1 | 42.5 % | 7 | 1.303 | 22.27 % | 0.557 | 0.8756 | 2.5019 | 0.162 | 0.0075 | 0.14 | 0.0095–0.13 | 18 404 | 237 |
| 4 | 8 | 300 | 1 | 1 | 50.7 % | 70 | 1.060 | 24.22 % | 0.068 | 0.8177 | 2.1319 | 0.155 | 0.0003 | 0.16 | 0.018–0.12 | 17 204 | 228 |
| 8 | 8 | 300 | 1 | 1 | 47.0 % | 5 | **2.761** | 25.00 % | 0.300 | 0.8086 | 1.9976 | 0.148 | 0.0100 | 0.18 | 0.012–0.10 | 17 204 | 301 |
| 16 | 20261020 | fixed | — | — | 49.5 % | 5 | 1.828 | 16.02 % | 1.101 | 0.9135 | 2.3154 | 0.215 | 0.0718 | — | 0.05 | 16 004 | 285 |
| 16 | 20261020 | 300 | 1 | 1 | 45.4 % | 5 | **2.490** | 15.23 % | 0.551 | 0.9450 | 2.5566 | 0.237 | 0.0558 | 0.18 | 0.012–0.12 | 17 204 | 302 |
| 16 | 20261020 | 300 | 0.5 | 1 | 40.2 % | 6 | 1.545 | 19.53 % | 0.907 | 0.9172 | 2.4973 | 0.208 | 0.0368 | 0.28 | 0.031–0.11 | 17 204 | 232 |
| 16 | 20261020 | 300 | 0 | 1 | 38.6 % | 7 | 1.440 | 19.53 % | 1.044 | 0.9125 | 2.4632 | 0.194 | 0.0354 | 0.35 | 0.053–0.07 | 17 204 | 234 |
| 4 | 20261020 | 300 | 1 | 1 | 50.4 % | 15 | 1.353 | 12.89 % | 0.429 | 0.9917 | 2.8658 | 0.269 | 0.0011 | 0.17 | 0.012–0.13 | 17 204 | 301 |
| 4 | 7 | 300 | 1 | 1 | 48.6 % | 97 | 1.062 | 17.58 % | 0.085 | 0.9054 | 2.6092 | 0.212 | 0.0022 | 0.22 | 0.014–0.14 | 17 204 | 233 |
| 8 | 20261020 | 300 | 1 | 1 | 48.1 % | 6 | 1.931 | 14.06 % | 0.594 | 0.9768 | 2.7792 | 0.280 | 0.0164 | 0.20 | 0.013–0.12 | 17 204 | 225 |
| 8 | 7 | 300 | 1 | 1 | 41.8 % | 12 | 1.164 | 22.27 % | 0.275 | 0.8741 | 2.5055 | 0.167 | 0.0052 | 0.19 | 0.018–0.12 | 17 204 | 227 |

(The last four rows are the R2f cells the new rows pair with; they are repeated so every
comparison above can be read off one table.  Rows 11–14 are group (D), the rank-16 ladder.)

**Result 32 — averaging independent pilots does not fix the raw pilot at rank 4, and does
nothing at rank 8.**  Rank 4, three seeds, ESS fixed/raw/2-rep: 100/15/**14** at the default
seed, 84/97/**128** at seed 7, 9/70/**140** at seed 8.  So a 2-replicate average is a real
improvement on two of three seeds — 1.3× and 2.0× the single pilot — and *no* improvement
exactly on the seed where the raw pilot already fails catastrophically (14 vs 15, and `R̂`
1.412 vs 1.353, both against fixed `R̂` 1.023).  Rank 8 is flat or worse, 6/6/**5**, 12/**7**,
5/**6**.  Neither four pilots' worth of averaging nor a doubled budget is on the table here:
two pilots is what "the estimator has variance" predicts to help, and it helps least where
it matters most.

**Result 33 — the pilot's error is bias, not variance; that is why R2e's lottery has a
mechanism.**  At equal total cost the three rank-4 cells order themselves
**single 600-step pilot 221 > two averaged 300-step pilots 14** at the default seed (R2e
measured the 221), while at seed 7 the same pair is 99 vs 128 — so neither budget shape
dominates, and the spread *between* the two estimators at the same cost (14 vs 221) is the
same size as the spread between seeds.  The one variable that explains both facts is how far
the pilot chain itself travels: a 600-step pilot explores more of the target and reports a
wider (less wrong) width vector, whereas two 300-step pilots each explore the same narrow
patch and their geometric mean is a stable estimate of a **biased** quantity.  Averaging
reduces variance; the failure mode is not variance, so the result is what R2f's width
lottery already looked like from outside.  The prediction this makes is testable and
negative for the averaging idea: a *scale* correction (multiply the pilot widths by a
constant > 1, or estimate them from a longer pilot) should fix the default-seed collapse;
more pilots at the same length should not, and do not.

*R2h runs that prediction and retracts this mechanism: the pilot's overall scale is absorbed
by the warm-up multiplier, so what survives into the kept window is only the width
**shape** — see Result 36 and Result 38 below.*

**Result 34 — shrinkage still points the right way at rank 16, but the effect has collapsed
into the resolution floor, so "isotropic wins everywhere" is retracted as a claim.**  The
rank-16 ladder is fixed 5 / `s = 1` **5** / `s = 0.5` **6** / `s = 0` **7** ESS, with `R̂`
1.828 / 2.490 / 1.545 / 1.440: the ordering is monotone in `s` except that `s = 1` is the
worst cell in the ladder, which is the same qualitative story as ranks 4 and 8.  But the
*sizes* are not comparable — ESS gain over fixed is 3.8× at rank 4 (383/100), 22× at rank 8
(154/7) and **1.4×** at rank 16 (7/5) — and at rank 16 the fixed arm itself has `R̂` 1.828,
i.e. the chains do not mix even with the *known* proposal, on 4 chains × 2 800 draws in a
16-dimensional subspace.  At that point a 5-vs-7 ESS difference with `R̂` MCSE of this size
is not a measurement of shrinkage; the honest statement is that shrinkage **does not hurt**
at rank 16 and marginally improves `R̂`, and that the label-selection win does survive
(BMA accuracy 16.02 % → 19.53 %, the `s = 0.5` and `s = 0` cells tying) while the NLL
worsens 2.3154 → 2.4632 exactly as at rank 4/8 (Result 30).  Rank 16 needs a different
experiment (more chains, longer runs, or an exactness check at that rank) before it can say
anything about the estimator.

**Result 35 — the housekeeping decision, and the revised recommendation.**  The unrelated
`cargo fmt` churn (20 files across `logos`, `mm0web`, `prob_kernel`, `unfer_measure`,
`unfer_taler` — 397 insertions / 188 deletions that survive `git diff -w`) was **reverted**
rather than committed: it is reproducible with `cargo fmt --all` at any time, `HEAD` is
already not fmt-clean in those crates (`cargo fmt --all --check` reports diffs there before
any change), and mixing it into a pcnl commit would bury the one real change and collide
with the concurrent commits those crates receive.  The recommendation that survives R2g:
`--pilot-shrink` on, `s = 0.5` at rank 4 and `s = 0` at rank 8 (the two measured optima),
`--pilot-reps` left off (it is available and verified, but not a fix), and *no* claim about
ranks above 8 until the comparison there is above the resolution floor.

### R2h — the scale dial: uniform inflation is absorbed by warm-up (same day)

R2g's Result 33 made a testable prediction: if the pilot's failure is that it reports widths
that are too *narrow* (bias), then multiplying the whole width vector by a constant `c > 1`
should rescue the default-seed collapse at rank 4.  `--pilot-scale c` is that dial.  It
multiplies every width by `c` after shrinkage and before `DiagScale::from_widths_shrunk`, so
with `--pilot-shrink 0` it is the only free parameter, and `c = 1` reproduces the earlier
tables bitwise (re-checked on the reduced config against the R2g binary: all 19 numeric
columns and the 1 802-eval count agree).

The prediction fails, and the reason is already in the kernel.  The pilot supplies a
*shape*; the proposal's overall size is not the pilot's to set.  `DiagScale` starts at
multiplier `m = 1` and Robbins–Monro-steers `m` on the joint acceptance rate through the
500-step burn, and only `m · width_i` reaches the kept window — its own doc comment says
"the shape of the proposal comes from the target and only its overall size comes from
feedback".  A uniform inflation of all widths is therefore, up to the finite-length
transient, the same kernel with `m → m / c`.  The `m` column measures exactly that.  Eight
cells, 111 min on 2 cores, every process `EXIT 0`: the six-cell default-seed grid (ranks
16/8/4 × `c` ∈ {2, 4}) plus a seed-7 pair at rank 8 (`c = 1` and `c = 4`) to separate a
warm-up effect from this estimator's seed noise.

| rank | seed | `c` | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | MCSE(acc) | Brier | NLL | ECE | MCSE(NLL) | `m` | evals | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 20261020 | 1 | 38.6 % | 7 | 1.440 | 19.53 % | 1.044 | 0.9125 | 2.4632 | 0.194 | 0.0354 | 0.35 | 17 204 | 234 |
| 16 | 20261020 | 2 | 38.0 % | 7 | 1.486 | 19.53 % | 1.007 | 0.9103 | 2.4620 | 0.177 | 0.0338 | **0.17** | 17 204 | 229 |
| 16 | 20261020 | 4 | 33.2 % | 9 | 1.321 | 19.53 % | 0.628 | 0.9131 | 2.5182 | 0.196 | 0.0316 | **0.10** | 17 204 | 308 |
| 8 | 20261020 | 1 | 48.7 % | 154 | 1.010 | 14.84 % | 0.277 | 0.9780 | 2.8105 | 0.253 | 0.0087 | 0.34 | 17 204 | 231 |
| 8 | 20261020 | 2 | 46.3 % | 139 | 1.005 | 14.84 % | 0.331 | 0.9774 | 2.8091 | 0.251 | 0.0092 | **0.18** | 17 204 | 228 |
| 8 | 20261020 | 4 | 40.6 % | 32 | 1.075 | 14.84 % | 0.144 | 0.9766 | 2.8033 | 0.248 | 0.0102 | **0.10** | 17 204 | 228 |
| 8 | 7 | 1 | 35.2 % | 20 | 1.121 | 21.88 % | 0.319 | 0.8699 | 2.4862 | 0.166 | 0.0100 | 0.30 | 17 204 | 302 |
| 8 | 7 | 4 | 32.2 % | 14 | 1.374 | 21.88 % | 0.491 | 0.8703 | 2.4833 | 0.164 | 0.0159 | **0.08** | 17 204 | 302 |
| 4 | 20261020 | 1 | 46.5 % | 383 | 1.010 | 12.89 % | 0.057 | 0.9922 | 2.8662 | 0.270 | 0.0005 | 0.33 | 17 204 | 299 |
| 4 | 20261020 | 2 | 44.4 % | 350 | 1.019 | 12.89 % | 0.059 | 0.9923 | 2.8667 | 0.270 | 0.0005 | **0.17** | 17 204 | 296 |
| 4 | 20261020 | 4 | 41.7 % | 416 | 1.015 | 12.89 % | 0.044 | 0.9923 | 2.8666 | 0.270 | 0.0005 | **0.09** | 17 204 | 225 |

(`c = 1` rows are the R2e/R2g `--pilot-shrink 0` cells, repeated for one-table comparison;
rank 8/seed 7 `c = 1` is new because R2g had no `s = 0` cell at that seed.  All rows share the
protocol of R2e–R2g and the default 600-step pilot; ESS is at the resolution floor for every
rank-16 cell, so those three rows are not a measurement of `c`.)

**Result 36 — `c = 2` is absorbed almost exactly and changes nothing measurable at any
rank.**  The multiplier halves: 0.35 → **0.17** at rank 16, 0.34 → **0.18** at rank 8,
0.33 → **0.17** at rank 4, i.e. within 3–6 % of the `m / 2` that exact absorption predicts.
With the size re-normalised, ESS is the `c = 1` value within noise at all three ranks
(7/7, 154/139, 383/350) and so are the predictives: NLL 2.4632 → 2.4620, 2.8105 → 2.8091,
2.8662 → 2.8667, BMA accuracy identical in every pair (19.53 / 14.84 / 12.89 %), Brier
moving in the fourth decimal.  Run to run the acceptance feedback does to a uniform scale
exactly what the posterior would have done anyway.

**Result 37 — `c = 4` is absorbed only partly, and the residue is a worse warm-up, not a
better proposal.**  Now `m` lands 7–18 % *above* the absorbed value (0.35/4 = 0.0875 →
0.10; 0.34/4 = 0.085 → 0.10; 0.33/4 = 0.0825 → 0.09; 0.30/4 = 0.075 → 0.08) but four times
the starting width, so the frozen kernel is left 5–20 % larger than the `c = 1` kernel and
its acceptance is correspondingly lower (48.7 → 40.6 %, 35.2 → 32.2 %, 46.5 → 41.7 %,
38.6 → 33.2 %).  ESS at rank 8 falls in both seeds measured (154 → 32, 20 → 14) and `R̂`
worsens in both (1.010 → 1.075, 1.121 → 1.374); at rank 4 ESS rises 383 → 416 and `R̂` is
flat (1.010 → 1.015); at rank 16 nothing is resolvable.  No cell improves on its own `c = 1`
baseline, and the rank-8 default-seed drop (154 → 32) is *inside* the seed spread of this
estimator at that rank (20 vs 154 for `c = 1` at seeds 7 and 20261020) — so the honest
statement is that `c > 1` never helps and the frozen kernel from an overshooting warm-up is
consistently a little off-target, which is a property of the Robbins–Monro transient, not of
pCN.  Correcting the short pilot's scale is not reachable through this dial: the dial is
undone before the kept window.

**Result 38 — R2g's Result 33 mechanism is retracted; the width lottery is a *shape*lottery.**  Result 33 attributed R2e's 600-step-vs-two-300-step gap (ESS 221 vs 14 at rank
4) to a pilot that reports widths that are too narrow, and predicted that a scale correction
would fix it.  Result 36 refutes the mechanism directly: the overall size never reaches the
kept window because the burn-in multiplier sets it.  What the pilot does contribute is the
*pattern* of relative widths across coordinates, and that is what R2e/R2f's ESS spread (9 to
100 at rank 4 across seeds, 20 to 154 at rank 8) actually measures — a 600-step pilot and
two 300-step pilots differ in shape, not just in size.  This also explains, without a second
hypothesis, why shrinking to `s = 0` is the winner everywhere it was measured: `s = 0`
discards the pilot's shape entirely and hands both shape and size to a mechanism that is
self-correcting (isotropic plus acceptance feedback), and no pilot-shaped proposal beat it.
The recommendation is unchanged from R2g (shrink on; `s = 0.5` at rank 4, `s = 0` at rank 8;
`--pilot-reps` and `--pilot-scale` available and verified but not fixes).  What changes is
the next question: not "how wide?" but "whose shape?" — a shape estimated at a point where
the curvature means something (the anchor `∇ℓ ≈ 0` is the reason the curvature diagonal
already failed at rank 1), or a robust shape pooled across replicates, compared against
isotropic `s = 0` at the two ranks where that baseline is above the floor (4 and 8).

### R2i — a pooled shape does not beat isotropic, and the shape is reproducible, not noisy (same day)

R2h left one question: if the pilot's contribution is the width *shape*, whose shape should it
be?  A single 300-step pilot's shape is estimated from a short walk, so the natural fix is to
pool several pilots — and to pool them *smartly*, shrinking each coordinate by how well the
replicates agree, so a shape the pilot reproduces survives and a shape it does not is pulled
back to isotropic.  `--pilot-ebayes` (with `--pilot-reps n ≥ 2`) is that estimator: it fits a
one-way random-effects model to the replicated log widths, estimates the between-coordinate
shape variance `τ² = var(d) − mean(sd_i²/n)`, and keeps the fraction `τ²/(τ² + sd_i²/n)` of
coordinate `i`'s deviation `d_i` (the global `--pilot-shrink s` still multiplies on top).
Six unit tests pin what is algebraically checkable — isotropic at `s = 0`, the plain estimate
when the replicates agree exactly, *more* keeping for an agreeing coordinate than a
disagreeing one, fallback below two replicates, shape invariance under a uniform scale, and
the identity defaults.

Eight cells, 106 min, all `EXIT 0`.  Controls at the same pilot budget: the pooled
geometric mean of the same three pilots *without* the weights (`pool3`), and the previously
measured single-pilot and isotropic cells at the same seed.

| rank | seed | shape source | `s` | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | Brier | NLL | `m` | evals |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20261020 | none (isotropic) | 0 | 46.5 % | **383** | 1.010 | 12.89 % | 0.9922 | 2.8662 | 0.33 | 17 204 |
| 4 | 20261020 | 3 pooled, EB weights | 0.5 | 49.0 % | 362 | 1.020 | 12.89 % | 0.9925 | 2.8671 | 0.28 | 19 612 |
| 4 | 20261020 | 3 pooled, no weights | 1 | 52.7 % | 16 | 1.258 | 12.89 % | 0.9918 | 2.8661 | 0.17 | 19 612 |
| 4 | 20261020 | 3 pooled, EB weights | 1 | 52.7 % | 17 | 1.252 | 12.89 % | 0.9918 | 2.8660 | 0.17 | 19 612 |
| 4 | 20261020 | 1 pilot (R2e) | 1 | 50.4 % | 15 | 1.353 | 12.89 % | 0.9917 | 2.8658 | 0.17 | 17 204 |
| 8 | 20261020 | none (isotropic) | 0 | 48.7 % | **154** | 1.010 | 14.84 % | 0.9780 | 2.8105 | 0.34 | 17 204 |
| 8 | 20261020 | 3 pooled, EB weights | 0.5 | 50.1 % | 64 | 1.062 | 14.84 % | 0.9774 | 2.8143 | 0.29 | 19 612 |
| 8 | 20261020 | 3 pooled, EB weights | 1 | 50.1 % | 8 | 1.404 | 14.84 % | 0.9798 | 2.8235 | 0.21 | 19 612 |
| 8 | 20261020 | 3 pooled, no weights | 1 | 51.0 % | 6 | 1.625 | 14.06 % | 0.9802 | 2.8256 | 0.19 | 19 612 |
| 8 | 20261020 | 1 pilot (R2e) | 1 | 48.1 % | 6 | 1.931 | 14.06 % | 0.9768 | 2.7792 | 0.20 | 17 204 |
| 8 | 7 | none (isotropic) | 0 | 35.2 % | **20** | 1.121 | 21.88 % | 0.8699 | 2.4862 | 0.30 | 17 204 |
| 8 | 7 | 3 pooled, EB weights | 0.5 | 40.6 % | 16 | 1.139 | 23.05 % | 0.8700 | 2.4874 | 0.25 | 19 612 |
| 8 | 7 | 3 pooled, EB weights | 1 | 43.1 % | 7 | 1.398 | 22.27 % | 0.8733 | 2.5004 | 0.19 | 19 612 |

**Result 39 — no pooled shape beats isotropic at any rank, seed or strength.**  ESS at
`s = 0 / 0.5 / 1` is 383 / 362 / 17 at rank 4, 154 / 64 / 8 at rank 8 and 20 / 16 / 7 at
rank 8 seed 7: monotone in the amount of shape retained, and the optimum is the end that
retains none.  Partial retention recovers most of the ESS, which is what `s = 0.5` already
did, but it never *exceeds* the isotropic cell, and the three seed-7/seed-20261020 pairs
agree on the sign.  The predictives do not separate the arms either (BMA accuracy identical
within a seed; NLL 2.8660 vs 2.8662 at rank 4, 2.8235 vs 2.8105 at rank 8), so this is a
mixing result, not an accuracy result — consistent with R2e/R2f's reading that the width
scaling is about how the proposal moves, and the readout does what it does either way.

**Result 40 — the reliability weighting is not what fails: the shape itself is
reproducible and wrong.**  At the same 900-eval pilot budget, the weighted and unweighted
pooled shapes are indistinguishable (17 vs 16 at rank 4, 8 vs 6 at rank 8, and both match
the *single*-pilot 15/6 within this estimator's noise), while the coarse `s = 0.5` pull —
which shrinks every coordinate, agreeing or not — recovers 362 and 64.  So per-coordinate
agreement weights have little to act on: the replicates agree about the shape to within a
fraction of what separates it from isotropic, i.e. the width pattern is a **stable property
of the pilot's walk**, not sampling noise.  That also explains R2g's null for replicate
averaging in the same terms: averaging and weighting both reduce variance in a quantity
whose problem is not variance.  Since the pooled shape is reproducible and still loses to a
single isotropic width by 5–20×, the pilot's per-coordinate spread is describing the geometry
of the region the pilot visited, which is not the geometry the target's mixing needs — the
same diagnosis R2 gave for the anchor curvature diagonal, one estimator over.

**Result 41 — what this closes.**  The pilot arm's answer is now complete on its own terms:
shape *is* what the pilot contributes (R2h), and every pilot-derived shape measured — raw,
shrunk at 0.5, averaged over 2 or 3 replicates, reliability-weighted — loses to discarding
it.  The recommendation is `s = 0` wherever it has been measured above the floor (0.5 at
rank 4 is the one exception, from R2e, and it is a 1.04× effect: 399 vs 383), with
`--pilot-reps` and `--pilot-scale` at their defaults and `--pilot-ebayes` unused.  The one
untested shape source is a **non-pilot** one: curvature at a point where `∇ℓ ≠ 0` makes the
Hessian describe local stiffness rather than a stationary bowl.  That is a different
experiment (it needs a point, not a longer walk), and until it is run, "shrink the pilot"
means `s = 0`.

### R2j — rank 16 is not resolvable at any budget tried, so no rank-16 comparison is a measurement

R2g's rank-16 ladder and R2h's rank-16 rows all sat at ESS ≤ 9 with `R̂` 1.32–1.49, which is
why neither was allowed to support a claim.  Before spending more rank-16 cells on
estimators, this asks whether the *rank* is measurable at all: the fixed arm uses one `β`
for every coordinate and does not depend on any pilot, so its diagnostics are a property of
the target and the harness, not of an estimator.  Three cells, 47 min, `EXIT 0`; the 4-chain
baseline is R2g's own rank-16 fixed cell.

| chains | steps/chain | kept draws | ESS(Φ) | `R̂(Φ)` | BMA acc | NLL | MCSE(NLL) | seed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 000 | 11 200 | 5 | 1.828 | 16.02 % | 2.3154 | 0.0718 | 20261020 |
| 4 | 16 000 | 49 600 | 8 | 1.440 | 18.75 % | 2.2555 | 0.0439 | 20261020 |
| 8 | 8 000 | 24 800 | 14 | 1.302 | 17.58 % | 2.1226 | 0.0405 | 20261020 |
| 8 | 8 000 | 24 800 | 12 | 1.906 | 23.44 % | 2.0576 | 0.0433 | 7 |

**Result 42 — the budget does not buy resolution; the target's autocorrelation time does.**
ESS(Φ) per 1 000 kept draws is 0.45, 0.16, 0.56 (and 0.48 at seed 7) — flat to within the
estimator's own instability, i.e. an integrated autocorrelation time of roughly 2 000 draws
*at every budget*.  Quadrupling the length from 11 200 to 49 600 draws moves ESS 5 → 8 and
`R̂` 1.828 → 1.440, and eight chains at 24 800 draws give ESS 14 / `R̂` 1.302.  In other
words the limit is neither chain count (8 chains barely beat 4 at comparable length) nor
length in a range anyone will run: reaching `R̂ < 1.05` from `R̂ − 1 ≈ 0.3` at this
`ESS` needs roughly another two orders of magnitude of draws, i.e. hours per cell for a
dial that is already a verified null at ranks 4 and 8.

**Result 43 — the *irreproducibility* is larger than the effects the ladder was measuring.**
The same fixed kernel at the same 8 × 3 100 budget, differing only in seed, gives `R̂` 1.302
vs 1.906, BMA accuracy 17.58 % vs 23.44 % (5.9 points) and NLL 2.1226 vs 2.0576 (0.065, about
1.6 pooled MCSEs).  R2g's rank-16 shrinkage ladder separates BMA accuracy by 3.5 points
(16.02 → 19.53 %) and NLL by 0.148 — so the BMA step is *smaller than the same-kernel seed
spread*, and `R̂` ordering across the ladder (1.828 / 2.490 / 1.545 / 1.440) is the same size
as the seed spread (1.302 / 1.906).  Result 34's "rank 16 does not hurt" and "marginally
improves `R̂`" are therefore withdrawn as measurements: what rank 16 supports is only that
the chain is slow there, which is a statement about the `16`-dimensional subspace geometry
(and the same diagnosis as the ambient run's radial–angular split), not about the estimator.

**Result 44 — what this rules out, and the one rank-16 experiment that would survive.**
Any further rank-16 arm comparison at this budget is uninterpretable, so the ladder is closed
there, and the recommendation table carries "no claim above rank 8" rather than a softened
version of the earlier one.  A rank-16 result would need a target-level change, not more
draws: a functional with a shorter autocorrelation than Φ (the loss is already much
better-behaved in the ambient runs), an exact or independent sampler at rank 16 for a
ground truth, or a smaller `λ`/larger subspace where the geometry is less stiff.  None of
these is an estimator question, which is why this cycle stops asking them of the pilot.

### R2k — curvature at the target's mode: the first shape to beat isotropic, at rank 4 only (same day)

R2i closed the pilot as a shape source and left one: a shape that is *derived* from the
target rather than from a walk.  The existing `curv` arm is not that shape — it takes the
Hessian at the anchor, which is a stationary point of the training loss, so `1/√H` there
describes a bowl the posterior has left (R2: 0.1 % acceptance, ESS 7 at rank 1).  `laplace`
fixes the *point*: it finds a mode of `Φ` by Adam on `∇Φ` from the anchor (with the prior in
both the value and the gradient), checks the gradient against `--mode-tol` before deriving
any width, and then takes `β_i = m/√H_ii` there — the Laplace approximation to the
posterior's spread.  The chain still starts at the anchor, so only the proposal's shape
differs from every earlier arm.

Finding the mode needed a gradient that is a real derivative of `Φ`, and that was the first
result: `pcnl::model::forward` defaults to `NormGrad::Stopped`, which freezes the Frobenius
denominators, so `loss_and_grad`'s gradient is **not** the derivative of the loss value —
measured at ~**1 021×** on the example's two-layer model (analytic `−1 469` vs a finite
difference `−1.44` along the same direction).  `SubspaceTarget::grad_phi` therefore uses
`NormGrad::Exact`, and the new `subspace_target_gradient_matches_finite_differences` test
pins the two together (it fails by 1.5 % on two of four coordinates with `Stopped`).  A mode
search on the stopped gradient descends a different function than the chain samples; nothing
in the earlier results depended on it, but a `laplace` arm would have silently done so.

The mode search converges: 1157 Adam steps at rank 4 (φ 2160.35 → 1602.06, `‖∇Φ‖∞`
5.4e1 → 9.9e-4) and 1132 at rank 8 (φ → 1604.91, 9.8e-4), both printed with the cell.  Cells:
four at the default seed (ranks 4 and 8 × `s` ∈ {1, 0.5}), plus a rank-4 seed-7 triple with
its own isotropic baseline (table below).

| rank | seed | shape | `s` | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | Brier | NLL | `m` | `β_i` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 20261020 | none (isotropic) | 0 | 46.5 % | 383 | 1.010 | 12.89 % | 0.9922 | 2.8662 | 0.33 | 0.069–0.08 |
| 4 | 20261020 | pilot (raw) | 1 | 50.4 % | 15 | 1.353 | 12.89 % | 0.9917 | 2.8658 | 0.17 | 0.012–0.13 |
| 4 | 20261020 | **laplace at mode** | 1 | 45.8 % | **554** | 1.010 | 12.89 % | 0.9923 | 2.8660 | 0.54 | 0.036–0.13 |
| 4 | 20261020 | **laplace at mode** | 0.5 | 46.3 % | **587** | 1.012 | 12.89 % | 0.9923 | 2.8658 | 0.62 | 0.053–0.10 |
| 8 | 20261020 | none (isotropic) | 0 | 48.7 % | **154** | 1.010 | 14.84 % | 0.9780 | 2.8105 | 0.34 | 0.064–0.07 |
| 8 | 20261020 | laplace at mode | 1 | 42.9 % | 8 | **1.559** | 14.84 % | 0.9708 | 2.5191 | 0.45 | 0.038–0.12 |
| 8 | 20261020 | laplace at mode | 0.5 | 45.5 % | 7 | **4.446** | 13.67 % | 0.9480 | 2.4127 | 0.50 | 0.054–0.11 |
| 4 | 7 | none (isotropic) | 0 | 47.5 % | 152 | 1.025 | 17.58 % | 0.9048 | 2.6068 | 0.38 | 0.064–0.07 |
| 4 | 7 | **laplace at mode** | 1 | 46.9 % | **287** | 1.017 | 17.58 % | 0.9049 | 2.6078 | 0.38 | 0.019–0.14 |
| 4 | 7 | **laplace at mode** | 0.5 | 48.0 % | **373** | 1.009 | 17.58 % | 0.9049 | 2.6082 | 0.49 | 0.041–0.11 |

**Result 45 — at rank 4 the target's own curvature is the first shape to beat isotropic, at
both seeds, by 1.45–2.45×.**  At the default seed ESS is 554 (`s = 1`) and 587 (`s = 0.5`)
against the isotropic 383; at seed 7 it is 287 and 373 against 152.  `R̂` is 1.009–1.017
against 1.010–1.025 and the predictives are unchanged within a seed (BMA 12.89 % / 17.58 %
identical, NLL within 0.0014), so this is a mixing gain and not a readout artifact.  The
seed-7 isotropic cell also reproduces R2f's rank-4 `s = 0` value exactly (152), which is the
control that says the two runs are comparable at all.  Every earlier shape failed here by 20× or more
(raw pilot 15, pooled 16–17, EB 17), which is the point: the failure was never "the pilot is
short", it was "the shape was measured where the posterior is not".  The mode search is the
only difference in the estimator — the widths are `1/√H_ii` at a point where `∇Φ` vanishes,
which is exactly the Laplace approximation, and the anchor it replaces is a stationary point
of a *different* function.

**Result 47 — `s = 0.5` is the rank-4 setting for the Laplace shape, and the ranking against
isotropic does not depend on the seed.**  Partial shrinkage wins at both seeds (587 vs 554;
373 vs 287), so the Laplace diagonal is a good shape that is still worth damping by half —
the same direction R2e found for the pilot at rank 4 (`s = 0.5` at 399 vs `s = 0` at 383),
now with a much larger effect.  What this cycle recommends: `--proposal laplace
--pilot-shrink 0.5` at rank 4, unchanged isotropic `--pilot-shrink 0` at ranks 8 and above,
and the pilot arm retired as a width source at every rank.  The cost is honest: the mode
search is ~1 150 full-batch backward passes, about 4 min on 2 cores, once per cell — against
a pilot's 600 forward passes, so it is affordable exactly because it is done once per rank
rather than per chain.

**Result 46 — the same shape is worse than the fixed arm at rank 8, and the rank dependence is
the finding.**  ESS 8 (`s = 1`, `R̂` 1.559) and 7 (`s = 0.5`, `R̂` 4.446) against isotropic 154
and fixed 7: at rank 8 the Laplace diagonal is no better than a single `β`.  This is not the
budget or the search — the rank-8 mode converged to the same tolerance as the rank-4 one and
to almost the same potential (1604.91 vs 1602.06, i.e. the rank-8 restriction contains an
equally good optimum).  What differs is what the diagonal describes: near a mode, `1/√H_ii`
is the *local* width, and a rank-8 restriction contains directions along which the posterior
is much wider than the local Hessian (the same distinction R2 drew for the anchor, now at a
point where the curvature is at least the right *kind* of quantity).  A shape that is right
at rank 4 and 20× wrong at rank 8 is informative about the target, not about the estimator:
it says the rank-8 posterior's mass is not in the mode's immediate bowl, and that a diagonal
Laplace surrogate degrades with rank.  The NLL/ECE difference in those two rows (2.5191/
2.4127 vs 2.8105, ECE 0.173 vs 0.253 at the same 14.84 % accuracy) is a property of *where*
the stalled chains sat, not a better predictive — accuracy is identical and `R̂` is 1.56–4.45.

### R2l — the Laplace win has a `λ` boundary, and the full covariance is worse than the diagonal (same day)

R2k left two questions it could not answer from its own cells: was the rank-4 win tied to
`λ = 1`, and was the rank-8 collapse a property of *truncating* the mode's Hessian to its
diagonal?  This cycle answers both.  `λ` is the likelihood's weight in `Φ = λNL + ½‖z‖²`, so
changing it moves every isotropic baseline too: each `λ` cell carries its own `pilot` `s = 0`
run, which R2e/R2i established as the strongest isotropic arm (383 at rank 4, `λ = 1`).

The second question needed one new arm, `rotlaplace`: the same mode search, the same
converged point, the full finite-difference Hessian there instead of its diagonal, rescaled
jointly in the eigenbasis.  It differs from `laplace` only in the off-diagonal curvature, and
it prints the proposal's β eigen-range, so a cell where that range is nearly flat is known to
be the diagonal arm and not to be evidence about rotation.

| rank | `λ` | shape | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | Brier | NLL | ECE | `m` | `β_i` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.5 | none (fixed `β = 0.05`) | 69.2 % | 97 | 1.038 | 12.89 % | 0.9956 | 2.8944 | 0.253 | — | 0.05 |
| 4 | 0.5 | isotropic (pilot `s = 0`) | 47.9 % | 342 | 1.008 | 12.89 % | 0.9962 | 2.8869 | 0.255 | 0.39 | 0.088–0.10 |
| 4 | 0.5 | **laplace at mode**, `s = 0.5` | 44.2 % | **419** | 1.011 | 12.89 % | 0.9956 | 2.8857 | 0.254 | 0.58 | 0.072–0.13 |
| 4 | 1.0 | isotropic (pilot `s = 0`, R2e) | 46.5 % | 383 | 1.010 | 12.89 % | 0.9922 | 2.8662 | 0.270 | 0.33 | 0.069–0.08 |
| 4 | 1.0 | laplace at mode, `s = 0.5` (R2k) | 46.3 % | 587 | 1.012 | 12.89 % | 0.9923 | 2.8658 | 0.270 | 0.62 | 0.053–0.10 |
| 4 | 2.0 | none (fixed `β = 0.05`) | 49.8 % | 109 | 1.020 | 12.89 % | 0.9905 | 2.8630 | 0.270 | — | 0.05 |
| 4 | 2.0 | isotropic (pilot `s = 0`) | 44.6 % | **376** | 1.018 | 12.89 % | 0.9901 | 2.8617 | 0.270 | 0.27 | 0.056–0.06 |
| 4 | 2.0 | laplace at mode, `s = 0.5` | 50.4 % | 351 | 1.016 | 12.89 % | 0.9899 | 2.8615 | 0.275 | 0.64 | 0.039–0.07 |
| 4 | 1.0 | **rotlaplace at mode** | 48.7 % | 112 | 1.036 | 12.89 % | 0.9913 | 2.8665 | 0.269 | 0.38 | 0.022–0.16 |
| 8 | 1.0 | none (fixed `β = 0.05`) | 55.4 % | 7 | 1.289 | 10.94 % | 0.9485 | 2.4693 | 0.237 | — | 0.05 |
| 8 | 1.0 | rotlaplace at mode | 49.2 % | 6 | **2.042** | 12.50 % | 0.9398 | 2.3449 | 0.220 | 0.27 | 0.015–0.14 |

The two rotlaplace runs print their mode search and eigen-range: rank 4 converges in 1 157
steps to the same `φ` as `laplace` (1602.0562, `‖∇Φ‖∞` 9.95e-4) with β eigen-range
[0.0606, 0.3967], ratio 6.5; rank 8 in 1 132 steps to 1604.9141 (9.83e-4), ratio 7.6.  So the
rotation is not degenerate in either cell — a difference from `laplace` is structure, not a
flat direction being labelled arbitrarily.

**Result 48 — the rank-4 Laplace win survives `λ < 1`, shrinks, and reverses at `λ = 2`.**
Against the best isotropic arm the gain is +53 % at `λ = 1` (383 → 587), +22.5 % at `λ = 0.5`
(342 → 419) and −6.6 % at `λ = 2` (376 → 351).  The sign change is the finding, and the mode
displacement explains it: the search needs 926 Adam steps at `λ = 0.5` (φ 1080.18 → 822.49) and
1 508 at `λ = 2` (φ 4320.71 → 3155.79) against 1 157 at `λ = 1`, and `λ = 2` also puts the
mode twice as far in `φ` units from the anchor.  The chain still starts at `z = 0`, so a
shape tuned to the far mode is a shape the chain cannot yet use: the pilot's widths are
measured *along the approach* and stay useful there, while the mode's diagonal describes a
region the chain reaches late.  This is R2k's rank-8 mechanism — width from a point the chain
has not reached — appearing at fixed rank as `λ` grows, and it is why the same arm can win at
rank 4, `λ ≤ 1` and lose at rank 4, `λ = 2`.  A `λ`-dependent recommendation that ignored this
would be overfitting one row of this table.

**Result 49 — the full covariance at the mode is worse than the diagonal, at both ranks.**
At rank 4 `rotlaplace` gives ESS 112 against `laplace`'s 587 and the fixed arm's 100; at rank 8
it gives 6 against the diagonal's 8 and the isotropic 154.  Its adapted multiplier says the
same thing without the chain: `m` settles at 0.38 (rank 4) and 0.27 (rank 8), i.e. the warm-up
had to shrink a proposal whose *shape* was already too wide, while the diagonal's `m` sits at
0.62.  An orthogonal rotation cannot change the target, so this is purely a proposal-efficiency
result: the mode's off-diagonal curvature is not the posterior's covariance, and rotating into
it costs more than the diagonal it replaces.  That retires "the rank-8 collapse is because we
threw away the off-diagonal" as an explanation — the rank-8 failure is the *point*, not the
parametrization, and it is the same failure at rank 4 once the mode is far enough away.

**Result 50 — the predictives are insensitive to all of this, again.**  Every rank-4 `λ` row
has BMA accuracy 12.89 % and NLL 2.860–2.894; rotlaplace's rank-8 row reports 12.50 % accuracy
and the *best* NLL (2.3449) at the *worst* `R̂` (2.042).  As in R2k, the low NLL of a stalled
chain is where it sat, not a better predictive: ESS and `R̂` move by two orders of magnitude
while the BMA answer does not move at all.  No recommendation in this document rests on a
predictive difference.

Practical upshot for the table below: `laplace` with `--pilot-shrink 0.5` stays the rank-4
setting *for `λ ≤ 1`*, with `λ = 2` a measured exception, and `rotlaplace` stays available but
unrecommended (it is the control that rules out the off-diagonal explanation).

### R2m — the anchor is not the lever, tempering the reference does not cover the approach, and the `λ = 2` reversal does not replicate (same day)

Three followups from R2l, run as one cycle because they share a code change.  Two new knobs
were added to `subspace_adaptation` and pinned by the defaults (`--laplace-tau 1`,
`--laplace-mix 0`, `--pilot-mix 0`, `--anchor-jitter 0`), and the no-flag cell `r8_p0`
reproduced R2e's ambient rank 8 exactly (`fixed` ESS 7 / `R̂` 1.289, `pilot s = 0` 154 /
1.010), so the flags are inert where they default.

* `--anchor-jitter f` shifts the reference by `N(0, (f · rms θ)²)` in flat space, seeded from
  `cfg.seed ^ 0xa11c_e000`, and prints the new anchor hash.  At `f = 0.10` the anchor's rms is
  0.7282, the shift's sd 0.0728, `‖θ_ref‖` 94.030 → 94.475, hash `0xed77c41cbcc0b510`.
* `--laplace-tau τ` reads the curvature at `τ · z_mode` instead of the mode — `τ = 1` is
  R2k's path, `τ = 0` the anchor — and now counts non-positive `H_ii` per rank.
* `--laplace-mix τ` and `--pilot-mix τ` are log-space *width* blends (`w_i = a_i^(1−τ) b_i^τ`)
  of (anchor, mode) and (pilot, mode) respectively: the versions of "temper the reference"
  that stay positive when the Hessian at the interpolation point does not.

| rank | `λ` | seed | shape | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | Brier | NLL | `m` | `β_i` |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1.0 | 20261020 | none (`fixed`) | 60.6 % | 100 | 1.023 | 12.89 % | 0.9925 | 2.8699 | — | 0.05 |
| 4 | 1.0 | 20261020 | isotropic (pilot `s = 0`) | 46.5 % | **383** | 1.010 | 12.89 % | 0.9922 | 2.8662 | 0.33 | 0.069–0.08 |
| 4 | 1.0 | 20261020 | laplace at mode (R2k, `s = 0.5`) | 46.3 % | **587** | 1.012 | 12.89 % | 0.9923 | 2.8658 | 0.62 | 0.053–0.10 |
| 4 | 1.0 | 20261020 | laplace at `τ = 0.5`, `s = 0.5` | 45.2 % | 428 | 1.013 | 12.89 % | 0.9923 | 2.8667 | 0.50 | 0.065–0.09 |
| 4 | 1.0 | 20261020 | laplace, anchor/mode mix 0.50 | 22.7 % | 5 | 1.916 | 15.62 % | 0.9925 | 2.8857 | 0.04 | 0.009–0.25 |
| 4 | 1.0 | 20261020 | laplace, anchor/mode mix 0.85 | 43.5 % | 254 | 1.021 | 12.89 % | 0.9921 | 2.8665 | 0.31 | 0.036–0.15 |
| 4 | 1.0 | 20261020 | pilot, pilot/mode mix 0.25 | 48.0 % | 352 | 1.017 | 12.89 % | 0.9924 | 2.8668 | 0.37 | 0.068–0.07 |
| 4 | 1.0 | 20261020 | pilot, pilot/mode mix 0.50 | 49.0 % | 271 | 1.030 | 12.89 % | 0.9923 | 2.8667 | 0.43 | 0.064–0.07 |
| 4 | 1.0 | 20261020 | **anchor shifted** (`f = 0.10`), none | 64.4 % | 165 | 1.018 | 14.45 % | 0.9810 | 2.9506 | — | 0.05 |
| 4 | 1.0 | 20261020 | **anchor shifted**, isotropic | 50.5 % | **7** | **10.183** | 15.23 % | 0.9727 | 2.8793 | 0.33 | 0.074–0.09 |
| 4 | 1.0 | 20261020 | **anchor shifted**, laplace at mode | 48.6 % | **378** | 1.019 | 15.23 % | 0.9800 | 2.9474 | 0.63 | 0.059–0.10 |
| 8 | 1.0 | 20261020 | none (`fixed`) | 55.4 % | 7 | 1.289 | 10.94 % | 0.9485 | 2.4693 | — | 0.05 |
| 8 | 1.0 | 20261020 | isotropic (pilot `s = 0`) | 48.7 % | **154** | 1.010 | 14.84 % | 0.9780 | 2.8105 | 0.34 | 0.064–0.07 |
| 8 | 1.0 | 20261020 | laplace at `τ = 0.5` (`H_ii ≤ 0`: 1/8) | 0.6 % | 5 | **17.963** | 8.98 % | 1.0168 | 3.2116 | 0.05 | 0.0065–1.00 |
| 8 | 1.0 | 20261020 | laplace, anchor/mode mix 0.50 | 1.8 % | 6 | 1.366 | 12.89 % | 0.9742 | 2.5248 | 0.03 | 0.020–0.48 |
| 8 | 1.0 | 20261020 | laplace, anchor/mode mix 0.85 | 42.3 % | 6 | 2.560 | 13.67 % | 0.9494 | 2.4202 | 0.17 | 0.034–0.13 |
| 8 | 1.0 | 20261020 | pilot, pilot/mode mix 0.25 | 46.8 % | 85 | 1.038 | 14.84 % | 0.9776 | 2.8073 | 0.38 | 0.070–0.07 |
| 8 | 1.0 | 20261020 | pilot, pilot/mode mix 0.50 | 47.4 % | 108 | 1.025 | 14.84 % | 0.9782 | 2.8137 | 0.41 | 0.067–0.07 |
| 4 | 0.5 | 7 | none (`fixed`) | 67.7 % | 93 | 1.040 | 19.53 % | 0.9106 | 2.6561 | — | 0.05 |
| 4 | 0.5 | 7 | isotropic (pilot `s = 0`) | 49.9 % | 233 | 1.009 | 19.53 % | 0.9092 | 2.6449 | 0.47 | 0.085–0.09 |
| 4 | 0.5 | 7 | laplace at mode, `s = 0.5` | 47.6 % | **336** | 1.005 | 19.53 % | 0.9089 | 2.6436 | 0.47 | 0.054–0.13 |
| 4 | 2.0 | 7 | none (`fixed`) | 46.3 % | 77 | 1.037 | 18.75 % | 0.9038 | 2.5880 | — | 0.05 |
| 4 | 2.0 | 7 | isotropic (pilot `s = 0`) | 46.5 % | 172 | 1.012 | 16.41 % | 0.9009 | 2.5310 | 0.30 | 0.052–0.06 |
| 4 | 2.0 | 7 | laplace at mode, `s = 0.5` (**search not converged**) | 47.8 % | **226** | 1.017 | 18.75 % | 0.9022 | 2.5486 | 0.53 | 0.031–0.09 |

Every mode search converged to `‖∇Φ‖∞ ≈ 9.8e-4` except the seed-7, `λ = 2` one, which hit the
2 000-step cap at 1.28e-2 and is flagged in the table rather than treated as a mode.

**Result 51 — the rank-4 Laplace win survives a shifted anchor, at half its ambient size.**
Moving the reference by 10 % of its own rms (the anchor is the one input R2k and R2l never
varied) leaves the win in place: `laplace` 378 against `fixed` 165, 2.3×, with the mode 1 630
Adam steps from the new anchor (φ 2158.8044 → 1629.2518) and `m` landing at 0.63, the same
as at the ambient anchor.  What does *not* survive is the *size*: at the ambient anchor the
ratio is 5.9× (587 / 100), and 2.3× here.  So the effect is not an artefact of one lucky
reference point, but its magnitude is anchor-dependent, and quoting 5.9× as "the" rank-4 gain
would be quoting one anchor.  The isotropic arm is what breaks off-anchor: `pilot s = 0`
collapses to ESS 7 with `R̂` 10.183 and `m` 0.33 — its widths are measured from a start now
0.475 away in `‖θ‖` from where the subspace's good directions live — so the shifted-anchor
cells are compared against `fixed`, and the honest summary is "the shape still helps by 2.3×
where the baseline is available at all".

**Result 52 — an interior point of `Φ` can have `H_ii ≤ 0`, and at rank 8 that destroys the
chain.**  Reading curvature at `τ = 0.5` gives 0/4 non-positive `H_ii` at rank 4, where it is a
mild *compromise*: ESS 428 against the isotropic 383 (+12 %) and the mode's 587 (−27 %), i.e.
tempering the reference at rank 4 trades away part of the win and adds nothing.  At rank 8 the
same τ samples **1 non-positive `H_ii` out of 8**: that coordinate's width becomes `1/√1e-8 =
10⁴`, the multiplier adapts down to 0.05, and the cell reports z acceptance 0.6 %, ESS 5,
`R̂` 17.963, β up to 1.00, Brier 1.0168 and NLL 3.2116 — worse than the mode arm it was meant
to improve on (8) and worse than `fixed` on every predictive.  This is not an inefficiency, it
is an invalid estimator: `1/√H_ii` presupposes a positive-definite curvature, an interior point
of `Φ` is a saddle along some directions, and the rank-8 subspace is where such a direction is
actually sampled.  A future tempered-reference arm must blend *widths*, not Hessians.

**Result 53 — blending widths only helps when the anchor half is removed, and then it is still
worse than the mode arm.**  The anchor/mode blend at 0.5 is catastrophic at both ranks — ESS 5
with `R̂` 1.916 at rank 4, 6 with `R̂` 1.366 at rank 8 — because the anchor's curvature diagonal
contains `H ≈ 0` directions whose widths are `1e4` (mean over coordinates) and `2.24e4` at rank 8,
so half of the blend is a proposal 10⁴× too wide in one coordinate; R2's `curv` failure is
exactly what the anchor half inherits.  Moving to 0.85 recovers rank 4 to 254 (above `fixed`'s
100, below the mode's 587) and leaves rank 8 at 6.  The pilot/mode blend is the fair version,
since both endpoints are measured widths: at rank 8 it gives 154 → 85 (τ = 0.25) → 108
(τ = 0.50), all at `R̂` ≈ 1.03, and at rank 4 it gives 383 → 352 → 271.  Every mixed cell is
below its own isotropic baseline.  So "temper the reference so the mode-centred widths also
cover the approach" is answered negatively in all four parametrizations tried: the mode arm's
advantage is not a linear-in-log-width quantity that can be dialled part of the way, and at rank
8 no shape built from these ingredients beats isotropic.

**Result 54 — R2l's `λ = 2` reversal is seed noise, and its recommendation row is withdrawn.**
At seed 7, `λ = 2` gives `laplace` 226 against the isotropic 172 — a **+31 % gain**, the opposite
sign to the default seed's −6.6 % (376 → 351) — while the other `λ` cells agree between seeds
(`λ = 0.5`: 233 → 336, +44 %, against the default seed's +22.5 %; `λ = 1`: 152 → 373, +145 %,
against +53 %).  The seed-7 cell must also carry the caveat that its mode search did not
converge (2 000 steps, 1.28e-2), so its 226 is from a near-mode and is weaker evidence than the
converged cells on either side.  Two seeds cannot settle a sign, but they do settle the *claim*:
R2l's "the win reverses at `λ = 2`" was a single cell, and the honest statement is that `λ = 2`
is unresolved — the gain is positive at both seeds for `λ ∈ {0.5, 1}` and of seed-dependent sign
at `λ = 2`.  The recommendation row is downgraded accordingly rather than left standing.

**Result 55 — the anchor moves the answer more than any proposal dial does.**  The jittered
anchor changes BMA accuracy from 12.89 % to 14.45–15.23 % at rank 4 — 1.6–2.3 points — and it
changes *every* arm at that anchor together (`fixed` 14.45 %, `pilot` and `laplace` 15.23 %),
which is the signature of a different subspace rather than a different sampler: the anchor
feeds the informed basis, and `--anchor-jitter` moves the directions, not just the start.  Every
proposal dial in R2c–R2m together moves BMA accuracy by at most 0.5 points inside one anchor
(12.89 % in every rank-4, `λ = 1` cell here, including the two cells with ESS 5 and ESS 271).
That inverts the emphasis of this document: the proposal is the thing that decides whether the
chain mixes, and the *anchor* is the thing that decides what the chain is reporting on.  R2n
should vary the anchor systematically — more seeds, more magnitudes — because it is the largest
lever found so far on the published answer, and R2k/R2l's 587 was measured at exactly one.

### R2n — the anchor sweep: BMA accuracy tracks the anchor, ESS does not track it monotonically, and the rank-8 widths are not 100× tighter (same day)

Two questions, one grid.  R2m's Result 55 asked for the anchor to be swept systematically — it is
the largest lever found on the published answer — and a second question arrived with it: the
rank-8 `laplace` collapse was being explained as "the rank-8 mode's curvature widths are ~100×
tighter than rank 4's".  A new `--width-report` flag prints the per-coordinate width vectors (the
pilot's own, the mode's curvature widths, and `laplace`'s at its reading point), so the second
question is answered by measurement rather than by inference.

The grid is rank 4, `λ = 1`, informed basis, `--anchor-jitter f` for
f ∈ {0.05, 0.10, 0.20, 0.30} × seeds {20261020, 7}, three arms per cell — `fixed` (β = 0.05),
`pilot --pilot-shrink 0`, `laplace --pilot-shrink 0.5` at the mode.  The `f = 0` rows are R2m's
base cell, kept for the trend.  Protocol: 4 000 steps, burn 500, thin 5, 4 chains, P = 16 672,
N_train 512, N_test 256, σ = 0.35 — 21 new runs.

The seed axis is **not** a clean re-draw of one anchor: `--seed 7` changes the Task-A pretrain
(anchor rms 0.7282 → 0.9200, `‖θ_ref‖` 94.030 → 118.788) as well as the data split
(`0xba104db4f677b2ec` → `0xe059cf933e1a92b0`), so seed moves the anchor, the informed basis and
the test set together.

| `f` | seed | anchor shift | arm | z acc | ESS(Φ) | `R̂(Φ)` | BMA acc | Brier | NLL | `m` |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (R2m) | 20261020 | — | `fixed` | 60.6 % | 100 | 1.023 | 12.89 % | 0.9925 | 2.8699 | — |
| 0 (R2m) | 20261020 | — | `pilot s = 0` | 46.5 % | 383 | 1.010 | 12.89 % | 0.9922 | 2.8662 | 0.33 |
| 0 (R2m) | 20261020 | — | `laplace s = 0.5` | 46.3 % | **587** | 1.012 | 12.89 % | 0.9923 | 2.8658 | 0.62 |
| 0.05 | 20261020 | sd 0.0364, `‖θ‖` → 94.135, `0x6147…` | `fixed` | 70.9 % | 142 | 1.016 | 14.45 % | 0.9953 | 2.9623 | — |
| 0.05 | 20261020 | — | `pilot s = 0` | 50.3 % | 607 | 1.017 | 14.45 % | 0.9955 | 2.9619 | 0.40 |
| 0.05 | 20261020 | — | `laplace s = 0.5` | 48.8 % | **651** | 1.006 | 14.45 % | 0.9952 | 2.9614 | 0.60 |
| 0.05 | 7 | sd 0.0460, `‖θ‖` → 118.893, `0xe057…` | `fixed` | 56.3 % | 80 | 1.042 | 16.80 % | 0.9176 | 2.6236 | — |
| 0.05 | 7 | — | `pilot s = 0` | 47.0 % | 314 | 1.008 | 16.41 % | 0.9181 | 2.6182 | 0.34 |
| 0.05 | 7 | — | `laplace s = 0.5` | 49.3 % | **511** | 1.003 | 16.41 % | 0.9181 | 2.6195 | 0.61 |
| 0.10 | 20261020 | sd 0.0728, `‖θ‖` → 94.475, `0xed77…` | `fixed` | 64.4 % | 165 | 1.018 | 14.45 % | 0.9810 | 2.9506 | — |
| 0.10 | 20261020 | — | `pilot s = 0` | 50.5 % | **7** | **10.183** | 15.23 % | 0.9727 | 2.8793 | 0.33 |
| 0.10 | 20261020 | — | `laplace s = 0.5` | 48.6 % | 378 | 1.019 | 15.23 % | 0.9800 | 2.9474 | 0.63 |
| 0.10 | 7 | sd 0.0920, `‖θ‖` → 119.293, `0xa5bb…` | `fixed` | 57.6 % | 41 | 1.049 | 18.75 % | 0.9175 | 2.6084 | — |
| 0.10 | 7 | — | `pilot s = 0` | 47.8 % | 172 | 1.021 | 18.36 % | 0.9174 | 2.5956 | 0.35 |
| 0.10 | 7 | — | `laplace s = 0.5` | 49.7 % | **409** | 1.012 | 16.41 % | 0.9181 | 2.6032 | 0.56 |
| 0.20 | 20261020 | sd 0.1456, `‖θ‖` → 95.846, `0x1890…` | `fixed` | 68.0 % | 130 | 1.025 | 16.02 % | 0.9848 | 2.8540 | — |
| 0.20 | 20261020 | — | `pilot s = 0` | 49.8 % | 93 | 1.063 | 15.23 % | 0.9808 | 2.8335 | 0.39 |
| 0.20 | 20261020 | — | `laplace s = 0.5` | 46.4 % | **403** | 1.011 | 16.02 % | 0.9829 | 2.8497 | 0.57 |
| 0.20 | 7 | sd 0.1840, `‖θ‖` → 120.963, `0xf78f…` | `fixed` | 63.3 % | 57 | 1.041 | 16.02 % | 0.9015 | 2.5329 | — |
| 0.20 | 7 | — | `pilot s = 0` | 51.2 % | 322 | 1.007 | 17.97 % | 0.8996 | 2.5134 | 0.45 |
| 0.20 | 7 | — | `laplace s = 0.5` | 48.8 % | 301 | 1.007 | 19.14 % | 0.8999 | 2.5144 | 0.43 |
| 0.30 | 20261020 | sd 0.2185, `‖θ‖` → 98.106, `0xb2c7…` | `fixed` | 66.9 % | **9** | 1.437 | 17.97 % | 0.9505 | 2.6675 | — |
| 0.30 | 20261020 | — | `pilot s = 0` | 44.6 % | 17 | 1.088 | 14.45 % | 0.9504 | 2.6528 | 0.39 |
| 0.30 | 20261020 | — | `laplace s = 0.5` | 39.5 % | 51 | 1.090 | 17.97 % | 0.9487 | 2.6643 | 0.48 |
| 0.30 | 7 | sd 0.2760, `‖θ‖` → 123.751, `0xedca…` | `fixed` | 63.9 % | 91 | 1.031 | 16.80 % | 0.8889 | 2.3966 | — |
| 0.30 | 7 | — | `pilot s = 0` | 50.6 % | 99 | 1.028 | 21.48 % | 0.8817 | 2.3759 | 0.42 |
| 0.30 | 7 | — | `laplace s = 0.5` | 50.8 % | 45 | 1.046 | 16.80 % | 0.8860 | 2.3863 | 0.54 |

**Result 56 — BMA accuracy follows the anchor, and at the default seed it rises monotonically with
the jitter.**  `12.89 % (f = 0) → 14.45 % (0.05) → 14.45 % (0.10) → 16.02 % (0.20) → 17.97 %
(0.30)`, i.e. **+5.08 points**, while no proposal dial in R2c–R2m moved it by more than 0.5 inside
one anchor.  At seed 7 the same axis is flat-to-noisy rather than monotone (`fixed`: 16.80 / 18.75 /
16.02 / 16.80; `pilot`: 16.41 / 18.36 / 17.97 / 21.48; `laplace`: 16.41 / 16.41 / 19.14 / 16.80).
So Result 55's "the anchor is the largest lever" survives and its direction does not generalise:
`--anchor-jitter` displaces the prior mean, so the *restricted model's predictive* changes, and
which way it moves depends on where the displaced anchor lands relative to Task B — Task-A's
pretrain anchor is evidently not the best prior mean for Task B at the default seed, and the
improvement is a property of the model, not of the sampler.  Two seeds are enough to refute
monotonicity, not enough to characterise the function.

**Corrected by R2o (below).**  With the data split held fixed and two jitter *directions* per
magnitude, this trend does not reproduce: the per-magnitude means are 14.46 / 13.67 / 16.60 /
13.47 %, the within-magnitude gaps reach 6.6 points, and eight anchors span 11.72–19.92 %.  The
cause here is that `--seed 7` moves the data with the anchor; the lever is the draw, not the
jitter magnitude.  Read Results 56–57 as a description of the cells they measured.

**Result 57 — ESS is not a dose–response in the jitter; it collapses in isolated cells.**  Default
seed, ESS at f = 0 / 0.05 / 0.10 / 0.20 / 0.30: `fixed` 100 / 142 / 165 / 130 / **9**;
`pilot` 383 / 607 / **7** / 93 / **17**; `laplace` 587 / 651 / 378 / 403 / **51**.  R2m treated the
f = 0.10 `pilot` collapse (ESS 7, `R̂` 10.183) as a notable anomaly; f = 0.30 is worse for *all
three* arms (9 / 17 / 51, `fixed` `R̂` 1.437) and f = 0.20 is nearly as bad for `pilot` (93), while
seed 7 at the same f gives 41 / 172 / 409 and 91 / 99 / 45.  Jitter is therefore a lottery on the
shape of the target at the perturbed reference, not a smooth degradation, and any single jittered
cell — R2m's included — is one draw from a Long-left-tailed distribution.  The same table read from
the other side: f = 0.05 is the *best* cell in the sweep for every arm at the default seed
(142 / 607 / 651, above the unjittered 100 / 383 / 587).

**Result 58 — the rank-4 `laplace` gain is real on average and unrelated to the anchor distance.**
`laplace / pilot` ESS: **1.53×** (f = 0), **1.07× / 1.63×** (0.05), **54.0× / 2.38×** (0.10),
**4.33× / 0.93×** (0.20), **3.00× / 0.45×** (0.30) — geometric mean **2.43×** over the nine cells,
with `laplace` ahead in **7 of 9** and behind at f = 0.20 and f = 0.30 at seed 7 (301 vs 322,
45 vs 99).  The 54× at f = 0.10 is mostly a `pilot` collapse (ESS 7) rather than a `laplace`
improvement — 378 is the *lowest* `laplace` value in that column — so "the gain grows with jitter"
is exactly the reading this sweep rules out.  *(R2o, below, holds the data split fixed and puts the
controlled `laplace` gain at median 1.19× and geomean 1.42× over eight anchors, with either arm able
to collapse; the range below stands, the 2.4× does not.)*  The recommendation keeps `laplace` at rank 4,
`λ ≤ 1` (7/9 cells, 2.4× geomean), but the honest number to quote is the range **0.45–54×**, not
any single cell, and the arm is not uniformly better.

**Result 59 — the rank-8 widths are not ~100× tighter than rank 4's, and cannot be the cause of
the rank-8 collapse.**  `--width-report` on the same cells that produce the published rows (they
reproduce exactly: rank 4 `fixed` 100 / `pilot` 383, rank 8 `fixed` 7 / `pilot` 154):

| rank | shape | width vector (per coordinate) | `‖w‖` | max/min |
|---:|---|---|---:|---:|
| 4 | mode (curvature) | 0.2276, 0.1151, 0.0915, 0.0693 | 0.2797 | 3.3× |
| 4 | pilot `s = 0` | 0.6013, 0.0912, 0.2224, 0.2275 | — | 6.6× |
| 8 | mode (curvature) | 0.2351, 0.2033, 0.1046, 0.0925, 0.1686, 0.1190, 0.0983, 0.2435 | 0.4771 | 2.5× |
| 8 | pilot `s = 0` | 0.3039, 0.1724, 0.1979, 0.0888, 0.1605, 0.3251, 0.2939, 0.1000 | — | 3.7× |

The rank-8 mode widths are **1.7× larger** in norm and **less** spread than rank 4's, and the
resulting `β` ranges are nearly identical (5.4e-2–0.11 vs 5.9e-2–0.10, `m` 0.50 vs 0.63).  The
only ~100× width spread anywhere in this record is R2m's rank-8 `τ = 0.5` cell (`β` 6.5e-3–1.00, a
154× span), and that comes from the `max(H_ii, 1e-8)` clip applied to its single non-positive
curvature — the `10⁴` width Result 52 describes — not from the mode.  Stated at the level it was
asserted, "the rank-8 mode's curvature widths are ~100× tighter than rank 4's" is **false**.

What does distinguish rank 8 is sensitivity to the *shape*, not the *scale*: uniform β = 0.05 →
ESS 7 (`R̂` 1.289); uniform β ≈ 0.065 (pilot, `s = 0`) → 154; a 2× per-coordinate spread
(0.054–0.11 at the mode) → 7 with 45.5 % acceptance and `R̂` **4.446**.  At rank 4 the same 1.7×
spread (0.059–0.10) gives 587.  The rank-8 failure signature is `R̂` 4.446 at 45.5 % acceptance —
chains that disagree by more than their own noise on a target they accept nearly half the time —
i.e. slow traversal of directions a point-curvature shape misrepresents, not a proposal that is too
narrow (that is R2m's `τ = 0.5` cell, at 0.6 % acceptance, `R̂` 17.96).  Limitation, stated: `ESS(Φ)`
is one scalar over the potential and the kernel prints no per-coordinate autocorrelation, so this
pins the cause to "shape/spread sensitivity, not width magnitude" without naming the coordinate
responsible.

**Result 60 — the rank-8 recommendation stands, its stated reason changes.**  The row "proposal at
rank 8: isotropic `s = 0`, not `laplace`" is unchanged (154 against 7/8), but the justification is
no longer "the diagonal is 20× wrong there": the diagonal's *scale* is right (β within 2× of the
pilot's, widths the same order and less spread), and the arm still collapses — which makes rank 8 a
statement about the informed subspace's target, not about the curvature estimate.  The
`--anchor-jitter` row also moves: it stays **open**, now with 21 cells instead of 1, and reads "the
largest lever on BMA accuracy (+5.1 points at the default seed, sign seed-dependent) and a lottery
on ESS (9–651 across the sweep) — keep it at 0 and report ESS only with its cell's `R̂`".

### R2o — the draw, not the magnitude: eight anchors on one data split

R2n closed on the one thing its own seed axis could not do: separate the anchor's effect on the
*model* from its effect on the *sampler*.  Its two seeds moved the anchor, the Task-B data split,
the informed basis and the test set together, so a 5-point accuracy trend could be read as an
effect of jitter magnitude, of seed, or of one lucky reference point.  `--anchor-seed k` re-draws
**only** the Gaussian shift (`θ_ref ← θ_ref + N(0, (f·rms θ)²)`), leaving `D_A`, `D_B`, `θ₀`, the
basis draw seed and the test split on `--seed 20261020`.  Two directions (`k = 1, 2`) × four
magnitudes, three arms each — 8 anchors, 16 invocations.  Every cell prints the same data hash
`0xba104db4f677b2ec`, so all eight are measured on one dataset, one basis draw and one test set;
the `f = 0` row is R2m/R2n's base cell, unchanged by construction.

| `f` | `k` | anchor hash | `‖θ_ref‖` | anchor on B (acc / NLL) | fixed ESS / acc | pilot ESS / acc | laplace ESS / acc |
|---:|---:|---|---:|---|---|---|---|
| 0 | — | — (R2m/R2n base) | 94.030 | 4.3 % / 4.1523 | 100 / 12.89 % | 383 / 12.89 % | 587 / 12.89 % |
| 0.05 | 1 | `0x9d61a881cfe5f80a` | 94.121 | 4.30 % / 4.1558 | 54 / 16.41 % | 299 / 16.41 % | 221 / 16.41 % |
| 0.05 | 2 | `0x3747d89bb4bf1560` | 94.162 | 4.30 % / 4.1624 | **9** / 15.62 % | 106 / 12.50 % | **8** / 16.80 % |
| 0.10 | 1 | `0xe2fc3614e299a55c` | 94.449 | 1.17 % / 4.1497 | 39 / 13.28 % | 344 / 13.28 % | 406 / 13.28 % |
| 0.10 | 2 | `0x7077028c1dc4ab00` | 94.530 | 4.30 % / 4.1749 | 58 / 15.23 % | 232 / 14.06 % | 268 / 14.06 % |
| 0.20 | 1 | `0x4d303395608db542` | 95.803 | 3.52 % / 4.1401 | 31 / **19.92 %** | 194 / **19.92 %** | 380 / **19.92 %** |
| 0.20 | 2 | `0xa169bca3b9ff0042` | 95.958 | 1.17 % / 4.2673 | 25 / 13.28 % | **16** / 13.28 % | 401 / 12.50 % |
| 0.30 | 1 | `0x30dfc5777a134b11` | 98.051 | 4.69 % / 4.1767 | 16 / 14.45 % | 38 / 13.28 % | 141 / 13.28 % |
| 0.30 | 2 | `0xb9ee7b106a3a98c1` | 98.275 | 6.25 % / 4.3687 | 75 / 11.72 % | 283 / 13.67 % | 337 / 13.67 % |

Pilot `R̂`: 1.017, 1.044, 1.008, 1.023, 1.010, **1.742**, 1.094, 1.009.  Laplace `R̂`: 1.023,
**1.689**, 1.011, 1.011, 1.016, 1.015, 1.017, 1.006.  Acceptance stays 37–51 % for the adaptive
arms throughout.

**Result 61 — at one data split the anchor draw alone moves BMA accuracy by 8.2 points, more than
R2n's whole magnitude trend.**  Across the eight draws the BMA accuracy ranges **11.72 %–19.92 %**
(all arms); on the pilot arm alone **12.50 %–19.92 %**, mean **14.55 %**, sd **2.56 points** over
8 draws.  R2n's monotone 12.89 → 17.97 % trend (+5.08) does not survive: the per-magnitude means
over the two directions are 14.46 / 13.67 / 16.60 / 13.47 %, **not monotone**, and the within-
magnitude gaps are as large as the whole trend — 3.91 points at `f = 0.05` (16.41 vs 12.50) and
**6.64 points at f = 0.20** (19.92 vs 13.28), against 0.78 and 0.39 at `f = 0.10` and `0.30`.  R2n's
Result 56 is therefore **downgraded to a property of its two draws**: jitter magnitude is not the
lever, the specific displaced anchor is.  What this run cannot do is estimate the distribution —
two directions per magnitude is a range, not a variance.

**Result 62 — the higher-accuracy anchors are not the ones that are better on Task B.**  At
`f = 0.05` the two anchors have *identical* unadapted Task-B accuracy (4.30 %, NLL 4.1558 vs
4.1624) and give 16.41 % and 12.50 % BMA — and the best BMA cell of the sweep (19.92 %, `f = 0.20`,
`k = 1`) comes from an anchor at 3.52 %, while the highest anchor accuracy (6.25 %) gives 13.67 %.
So the effect is not "a better prior mean predicts better": the same anchor quality with the same
data and test set lands four points apart.  The remaining candidate is the informed subspace —
the PCA path is computed *from* the shifted anchor, so the basis moves with the draw even though
its seed does not (Result 65).

**Result 63 — the ESS lottery is the draw too.**  With data fixed, ESS still spans 9–75 (`fixed`),
16–344 (`pilot`) and 8–406 (`laplace`) across the eight anchors, and at `f = 0.20` alone the pilot
arm gives 194 (`k = 1`) against **16** (`k = 2`).  Conversely `f = 0.30`, which collapsed all three
arms in R2n (ESS 9 / 17 / 51 with `fixed` `R̂` 1.437), is unremarkable here on both draws (16 / 38 /
141 and 75 / 283 / 337).  So R2n's Result 57 — "jitter is a lottery on the shape of the target" —
is correctly observed and wrongly attributed: the lottery is over anchors, and the jitter magnitude
merely changes which anchors you get.  The `f = 0.20`, `k = 2` cell reproduces the rank-8 signature
found in Result 59: **`R̂` 1.742 at 49.9 % acceptance**, a mixing failure at ordinary acceptance,
not a proposal that is too narrow.  The laplace arm collapses separately at `f = 0.05`, `k = 2`
(ESS 8, `R̂` 1.689, 37.4 % acc) on the same anchor where the pilot arm reaches 106 — so at a fixed
anchor the two adaptive arms can also disagree, which is the reverse of R2n's "all arms in a cell
agree to ≤0.8 point": here one cell spans 15.62 / 12.50 / 16.80 % (**4.3 points across arms**).

**Result 64 — the rank-4 `laplace` advantage shrinks to a median 1.19× under this control.**
`laplace / pilot` ESS on the eight draws: 0.74, **0.08**, 1.18, 1.16, 1.96, **25.1**, 3.70, 1.19 —
median **1.19×**, geometric mean **1.42×**, range 0.08–25×.  The 25× is the pilot's collapse at
`f = 0.20`, `k = 2` (16 → 401), and the 0.08 is the laplace's own collapse at `f = 0.05`, `k = 2`
(106 → 8).  R2n's 2.43× geomean (9 cells) and R2k's 5.9× (one cell) are therefore both optimistic:
the arm is better *most* of the time and by a modest factor, and either arm can fail on a given
anchor.  `laplace` at rank 4 stays the recommendation, quoted as median 1.19× (geomean 1.42×),
not as a 2–6× win.

**Result 65 — what this still does not separate, and the next run.**  `--anchor-seed` holds `D_A`,
`D_B`, `θ₀` and the basis draw seed fixed, but the informed basis is the PCA of a 32-step Task-B
momentum path *starting at `θ_ref`*, so the basis vectors move with the anchor.  The measured
spread is thus "anchor → (prior mean + informed basis)" jointly — the pipeline the published rows
actually use — not the prior mean alone.  Separating those two requires computing the basis from a
fixed anchor (e.g. the ambient `θ_ref`) and jittering only the sampling reference, which is one
flag (`--basis-anchor`) and one repeat of this grid.  **Done in R2p, below:** the mixing spread was
the subspace; the accuracy spread is the prior mean.

### R2p — pinning the basis: the mixing was the subspace, the accuracy is the prior mean

R2o's open cell, run.  `--basis-anchor` builds the informed PCA path at the **ambient** anchor, so
`--anchor-jitter` moves only the sampling reference.  The path is deterministic given `D_B` and the
ambient anchor, so all eight cells here print the identical basis line (`PCA of 32 Task-B momentum
steps, |theta| 94.030 -> 93.784`) — eight anchors on one **restricted model**, differing only in
the prior mean, at rank 4, `λ = 1`, same data hash `0xba104db4f677b2ec`, same anchors as R2o.  Protocol
as R2o (`--steps 4000 --burn 500 --thin 5 --chains 4 --basis informed`); arms `fixed,pilot` with
`--pilot-shrink 0`.

| `f` | `k` | R2o pilot ESS / acc | **fixed** ESS / `R̂` / acc | **pilot** ESS / `R̂` / acc | pilot Brier / NLL |
|---:|---:|---|---|---|---|
| 0.05 | 1 | 299 / 16.41 % | 115 / 1.021 / 15.23 % | 425 / 1.014 / 15.23 % | 0.9996 / 2.9090 |
| 0.05 | 2 | 106 / 12.50 % | 130 / 1.022 / 12.89 % | 463 / 1.008 / 13.67 % | 0.9908 / 2.8694 |
| 0.10 | 1 | 344 / 13.28 % | 126 / 1.020 / 17.58 % | 522 / 1.008 / 15.23 % | 1.0032 / 2.9640 |
| 0.10 | 2 | 232 / 14.06 % | 166 / 1.020 / 17.58 % | 430 / 1.005 / 17.58 % | 0.9923 / 2.8914 |
| 0.20 | 1 | 194 / 19.92 % | 133 / 1.021 / 13.28 % | 540 / 1.006 / 13.28 % | 1.0090 / 3.0616 |
| 0.20 | 2 | **16** / 13.28 % | 154 / 1.018 / 13.67 % | 362 / 1.016 / 13.67 % | 1.0094 / 2.9929 |
| 0.30 | 1 | 38 / 13.28 % | 67 / 1.036 / 14.06 % | 233 / 1.019 / 14.06 % | 1.0277 / 3.1579 |
| 0.30 | 2 | 283 / 13.67 % | 56 / 1.036 / 6.64 % | 229 / 1.017 / 6.64 % | 1.0439 / 3.1682 |

**Result 66 — the mixing collapses were the moving subspace.**  Pinning the basis raises the pilot
arm's ESS on 7 of 8 anchors and removes every collapse: 16 → 362, 38 → 233, 106 → 463, while the
one cell that was already healthy gets no better (283 → 229, −19 %).  Geomean **2.95×**, median
**213 → 428**, mean **189 → 400**, minimum **16 → 229**.  `R̂` never exceeds **1.036** (R2o had
1.742 and 1.094).  The proposal-independent arm improves the same way (median 38 → 125, range
9–75 → 56–166), so this is not a proposal artifact: with the basis moving, the same eight prior
means were being sampled in eight differently-shaped 4-dim targets, and the bad ones were bad
*targets*, not bad diagnostics.

**Result 67 — the accuracy spread is the prior mean, and pinning the basis does not narrow it.**
Pilot accuracy across the eight anchors: 15.23, 13.67, 15.23, 17.58, 13.28, 13.67, 14.06, **6.64 %** —
range **10.94 points**, sd **3.16**, against R2o's 7.42-point range and sd 2.56 on the same anchors.
The new low (6.64 %, at `f = 0.30`, `k = 2`) is a genuine model effect rather than a sampler effect:
that cell is also the worst on Brier (1.0439) and NLL (3.1682).  Per-`f` means (14.45 / 16.41 /
13.47 / 10.35 %) are again **not monotone**, so R2o's negative result on jitter magnitude stands
under the stricter control.  Mean accuracy is 0.9 points *lower* than R2o's (13.67 vs 14.55 %):
the basis that moved with the anchor was, on average, slightly friendlier to held-out Task B —
which is the data-reuse optimism this arm carries by construction, now visible as a number.

**Result 68 — with one basis for all anchors, the arms stop disagreeing and accuracy follows the
proper scores.**  `fixed` and `pilot` BMA accuracy are identical in 6 of 8 cells and never differ by
more than 2.35 points (R2o: up to 4.30 with the basis moving); the two arms are sampling the same
target, so their summaries agree and the residual is Monte Carlo.  The extremes line up with Brier
and NLL — the best cell (17.58 %) has the lowest Brier (0.9923) and second-lowest NLL (2.8914), the
worst (6.64 %) the highest of both — so once the approximation is the *same* 4-dim target for every
cell, held-out accuracy is a property of the model and the sampler no longer contaminates it.

**Result 69 — what R2p leaves open.**  The `laplace` arm was **not** re-run under the control (8
further cells at ~17 min each), so whether R2o's median 1.19× laplace advantage is basis-driven is
**unmeasured**, not refuted.  Two directions per magnitude still give a range, not a variance; the
ambient anchor is one of infinitely many pinned bases, so "the accuracy spread is the prior mean"
is conditional on this basis (a second pinned basis would test that).  And R2o's accuracy ceiling
(19.92 %) did not reproduce here (best 17.58 %), so the draw that looked best in R2o was a
basis+prior coincidence.  *(R2s, below, closes the range-vs-variance cell: with six directions per
magnitude the spread is 69 % within-magnitude, i.e. the prior mean's **direction** rather than its
distance, and Result 67's phrasing is too strong in that respect.)*

### R2q — the laplace arm under one basis: the median gain is the target, the tail was the basis

R2p's open cell, run: the same eight anchors, basis pinned at the ambient anchor
(`--basis-anchor`), `--proposal fixed,laplace --pilot-shrink 0.5`.  Protocol identical to R2p
(`--ranks 4 --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed`, `λ = 1`, data hash
`0xba104db4f677b2ec`, identical basis line `94.030 -> 93.784`), so every cell here is the R2p
target with the proposal swapped for the one R2o used.  Comparability is checked by the `fixed`
rows: they reproduce R2p's **115 / 130 / 126 / 166 / 133 / 154 / 67 / 56** ESS and every accuracy
and `R̂` exactly.

| `f` | `k` | R2p pilot ESS / acc | R2o laplace ESS / acc (basis moving) | **R2q laplace** ESS / `R̂` / acc | laplace Brier / NLL | laplace ÷ pinned pilot |
|---:|---:|---|---|---|---|---|
| 0.05 | 1 | 425 / 15.23 % | 221 / 16.41 % | 464 / 1.007 / 15.23 % | 0.9992 / 2.9082 | 1.09× |
| 0.05 | 2 | 463 / 13.67 % | 8 / 16.80 % | 574 / 1.009 / 13.67 % | 0.9903 / 2.8684 | 1.24× |
| 0.10 | 1 | 522 / 15.23 % | 406 / 13.28 % | 503 / 1.006 / 17.58 % | 1.0029 / 2.9631 | 0.96× |
| 0.10 | 2 | 430 / 17.58 % | 268 / 14.06 % | 504 / 1.010 / 17.58 % | 0.9919 / 2.8907 | 1.17× |
| 0.20 | 1 | 540 / 13.28 % | 380 / 19.92 % | 432 / 1.009 / 13.28 % | 1.0086 / 3.0612 | 0.80× |
| 0.20 | 2 | 362 / 13.67 % | 401 / 12.50 % | 617 / 1.002 / 13.67 % | 1.0090 / 2.9922 | 1.70× |
| 0.30 | 1 | 233 / 14.06 % | 141 / 13.28 % | 368 / 1.004 / 14.06 % | 1.0268 / 3.1562 | 1.58× |
| 0.30 | 2 | 229 / 6.64 % | 337 / 13.67 % | 276 / 1.009 / 6.64 % | 1.0435 / 3.1688 | 1.21× |

**Result 70 — the median 1.19× is not basis-driven; the geomean was.**  Against R2p's pinned pilot
arm the laplace gains are 1.09, 1.24, 0.96, 1.17, 0.80, 1.70, 1.58, 1.21 — median **1.189×**,
geomean **1.188×**, **6/8** wins, every ratio inside **[0.80, 1.70]**.  R2o's moving-basis laplace
ratios on the same eight anchors were median **1.186×** but geomean **1.42×**, range
**[0.075, 25.06]**.  So the *typical* cell advantage is unchanged by pinning — it is a property of
the rank-4 target's curvature at its mode (`m` 0.53–0.64, `β_i` 5.4e-2–0.13) — while the two
spectacular R2o cells were artifacts of the moving basis: there pilot collapsed to ESS 16 and 38,
and under the pin the same anchors give 362 and 233, so laplace is only 1.70× and 1.58×.

**Result 71 — under one basis laplace is uniformly, predictably better than isotropic.**  Against
the pinned `fixed` arm (56–166) the ratios are 4.03, 4.42, 3.99, 3.04, 3.25, 4.01, 5.49, 4.93 —
geomean **4.08×**, minimum **3.04×**, no loss in eight cells (R2o: geomean 5.85× but range
0.89–16.0× with one loss).  Laplace ESS is 276–617, mean **467**, sd **110** (R2o: 8–401, mean 270,
sd 141); acceptance sits in a 2.2-point band, **45.4–47.6 %** (R2o: 37.4–48.5 %).  All eight mode
searches converged in **982–1382** of 2 000 steps with `H_ii ≥ 0` in **4/4** coordinates.  The pin
that stabilised the pilot stabilises the mode's width vector too.

**Result 72 — no accuracy price, and ESS stops tracking accuracy.**  Laplace BMA accuracy equals
the pinned pilot's in **7/8** cells; the one difference is at `f = 0.10, k = 1` (17.58 vs 15.23 %,
+2.35 points) and favours laplace.  Means 13.96 vs 13.67 %, sd 3.42 vs 3.16; per-`f` laplace means
**14.45 / 17.58 / 13.47 / 10.35 %** keep the non-monotone shape R2o and R2p both found.  The worst
cell reproduces exactly (6.64 %, Brier 1.0435, NLL 3.1688) — and is the cell where laplace beats
isotropic by the most (4.93×): a better sampler cannot recover a bad prior mean.  ESS varies only
2.2× across the eight anchors (CV 23 %) while accuracy spans 6.64–17.58 %, so accuracy can no longer
be read off the mixing diagnostic.

**Result 73 — what this settles and what it does not.**  Settled: `laplace` at the mode is
load-bearing on the controlled basis as well (geomean 4.08× over isotropic, ≥ 3.0× in every cell),
and its support should be quoted as **median 1.19× / geomean 1.19× / 0.80–1.70 per cell** against
the pilot arm, not R2o's 1.42× geomean, which was carried by two collapses.  Not settled: this is
still one pinned basis, so whether the 6.64 % cell is a property of the prior mean or of this
basis is open (a second pinned basis would separate them); two anchors per magnitude still give a
range, not a variance; and neither `rotlaplace` nor `--laplace-tau` was re-run under the pin — both
remain retired on R2l/R2m grounds rather than on a control.

### R2r — a second pinned basis: the accuracy spread is the prior mean, the mixing gain was the ambient basis

R2q's open cell, run: a basis pinned to something that is **not** the ambient anchor.  The new
`--basis-anchor-seed 1` draws one anchor from the law `--anchor-jitter` uses at `f = 1`
(`ambient + N(0, rms²)`, fixed magnitude) and builds the informed PCA path from it; the draw depends
only on the seed, so all eight cells share one basis but not the model's own reference point.
Verified in every cell: `|θ|` **132.394**, basis anchor hash `0xfe9ad411b8756ef7`, basis line
`132.394 -> 132.789` — the same in all eight, and not R2p's `94.030 -> 93.784`.  Protocol otherwise
R2p's (`--ranks 4 --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed`, `λ = 1`,
`--proposal fixed,pilot --pilot-shrink 0`, data hash `0xba104db4f677b2ec`).

| `f` | `k` | pilot ESS / `R̂` / acc | pilot Brier / NLL | `fixed` ESS / `R̂` / acc | R2p pilot ESS / acc |
|---:|---:|---|---|---|---|
| 0.05 | 1 | 24 / 1.098 / 18.75 % | 1.0177 / 3.7575 | 7 / 1.347 / 12.11 % | 425 / 15.23 % |
| 0.05 | 2 | 46 / 1.043 / 13.28 % | 1.0177 / 3.7222 | 5 / 2.218 / 13.28 % | 463 / 13.67 % |
| 0.10 | 1 | 141 / 1.033 / 15.23 % | 1.0186 / 3.7803 | 11 / 1.231 / 12.11 % | 522 / 15.23 % |
| 0.10 | 2 | 7 / 16.048 / 11.33 % | 1.0047 / 3.4932 | 5 / 2.783 / 10.94 % | 430 / 17.58 % |
| 0.20 | 1 | 154 / 1.050 / 12.11 % | 1.0139 / 3.6856 | 12 / 1.126 / 14.45 % | 540 / 13.28 % |
| 0.20 | 2 | 6 / 14.777 / 12.50 % | 1.0092 / 3.4317 | 6 / 2.137 / 12.11 % | 362 / 13.67 % |
| 0.30 | 1 | 101 / 1.027 / 17.97 % | 1.0171 / 3.5098 | 12 / 1.217 / 11.33 % | 233 / 14.06 % |
| 0.30 | 2 | 6 / 1.715 / 8.98 % | 1.0102 / 3.3576 | 6 / 2.505 / 11.33 % | 229 / 6.64 % |

**Result 74 — pinning is not what fixed the mixing; the ambient basis was.**  Under the second pin
the pilot arm's ESS is 24 / 46 / 141 / 7 / 154 / 6 / 101 / 6 — mean **60.6**, sd 62.3 — against
R2p's 229–540 (mean 400.5, sd 118.7): **6.6× lower**, per-cell ratios 17.7×, 10.1×, 3.7×, 61.4×,
3.5×, 60.3×, 2.3×, 38.2×.  Three cells carry `R̂` **1.715 / 14.777 / 16.048** (R2p's maximum was
1.036) and four sit at `R̂ ≥ 1.04`; acceptance is still 40.3–49.3 %, so this is a target-geometry
problem, not a step-size one.  The proposal-independent arm degrades with it (ESS 5–12, `R̂`
1.126–2.783).  So R2p's ESS result attributes to the **ambient** basis: an anchor inside the
target's typical set is a good preconditioner because it is near the density, not because it is
pinned.  Pinning removes basis *drift*, which is what the attribution needed, and nothing more.

**Result 75 — the accuracy spread survives, and so does the worst anchor.**  BMA accuracy is
18.75 / 13.28 / 15.23 / 11.33 / 12.11 / 12.50 / 17.97 / 8.98 %: mean **13.77** (R2p 13.67), sd
**3.34** (3.16), span **9.77** points (10.94).  The **worst cell is the same anchor**
(`f = 0.30, k = 2`: 8.98 vs 6.64 %), and the best moves to `f = 0.05, k = 1` (18.75 vs 17.58 % at
`f = 0.10, k = 2`); the per-`f` means are **16.02 / 13.28 / 12.31 / 13.48 %** against R2p's
14.45 / 16.41 / 13.47 / 10.35 %, non-monotone in both.  Rank agreement across the eight anchors is
Spearman **0.40** (ESS ranks agree at **0.69**), so the *size* of the spread is a prior-mean
property that reappears under an unrelated basis, while *which* anchor wins is partly basis-
specific at two draws per magnitude.  In particular the 6.64 % cell is not an artifact of the
ambient basis: its anchor is worst under both.

**Result 76 — accuracy is robust to this collapse; the proper scores are not.**  Accuracy barely
moves (mean 13.67 → 13.77) while ESS falls 6.6×, because it is a coarse discrete mean over four
chains.  Brier is flat (mean 1.013, 1.0047–1.0186, versus R2p pilot 1.007) but **NLL is 3.59 on
average (3.36–3.78) against R2p's 2.96 (2.87–3.17)** — a 0.63-nat gap on the *same* target.  That
gap is chain error, not a different `pi_r`: at ESS 6–154 with `R̂` up to 16 the ensemble average is
a poor estimate of the predictive, and NLL, being a log-area functional, inherits it far more than
the accuracy fraction does.  Consequence for R3: proper-score comparisons across bases/settings
must be gated on `R̂`/ESS (publish the diagnostic next to every cell), or they will report mixing
failures as calibration failures.

**Result 77 — what this settles and what it does not.**  Settled: the accuracy-spread attribution
holds under a second, independent basis, so it is the prior mean and not the subspace; the ESS
attribution sharpens to the ambient basis specifically, so R2p's "pinning removes the collapses"
should be read as "the ambient anchor removes the collapses — a preconditioner must sit where the
posterior lives"; and accuracy is a robust readout for anchor studies where proper scores are not.
Not settled: one pin seed at one magnitude (1.0 rms) — a *near* ambient-anchored pin would
separate "pinned" from "near" — the already-pinned `--basis random` arm was not re-run as the
stronger contrast, and the laplace arm was not repeated under this basis (it would test whether
median 1.19× is a mode-curvature property or another ambient-basis effect).

### R2s — six anchor directions per magnitude: the accuracy spread is the direction, not the magnitude

R2p and R2r both reported the accuracy spread as a **range over two directions per magnitude**, and
both read it as a prior-mean effect.  Two draws cannot separate "how far the anchor moved" from
"which way it moved", so this repeats the ambient-pinned sweep at **six independent directions per
magnitude** (`--anchor-seed 1..6`), 24 cells on one basis and one data hash.  The first two seeds at
every magnitude are R2p's own cells; comparability was checked rather than assumed by re-running
`f = 0.05, k = 1`, which reproduces R2p exactly (fixed 115 / 1.021 / 15.23 %, pilot 425 / 1.014 /
15.23 %, Brier 0.9996, NLL 2.9090, same basis line and anchor shift).  So only 16 cells are new
(`k = 3..6`), run 06:17–08:39 at ~15 min each on two concurrent streams.  Protocol otherwise
unchanged: `--ranks 4 --steps 4000 --burn 500 --thin 5 --chains 4 --basis informed --basis-anchor
--proposal fixed,pilot --pilot-shrink 0`, `λ = 1`, `σ = 0.35`, `β = 0.05`, data hash
`0xba104db4f677b2ec`, basis `94.030 -> 93.784` in every cell.

| `f` | `k` | pilot ESS | `R̂` | BMA acc | MCSE(acc) | Brier | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1 | 425 | 1.014 | 15.23 % | 0.038 | 0.9996 | 2.9090 |
| 0.05 | 2 | 463 | 1.008 | 13.67 % | 0.050 | 0.9908 | 2.8694 |
| 0.05 | 3 | 358 | 1.013 | 12.89 % | 0.066 | 0.9918 | 2.8608 |
| 0.05 | 4 | 365 | 1.009 | 12.89 % | 0.038 | 0.9899 | 2.8603 |
| 0.05 | 5 | 265 | 1.031 | 12.89 % | 0.048 | 0.9944 | 2.8739 |
| 0.05 | 6 | 339 | 1.027 | 12.89 % | 0.040 | 0.9934 | 2.8816 |
| 0.10 | 1 | 522 | 1.008 | 15.23 % | 0.068 | 1.0032 | 2.9640 |
| 0.10 | 2 | 430 | 1.005 | 17.58 % | 0.047 | 0.9923 | 2.8914 |
| 0.10 | 3 | 301 | 1.024 | 12.89 % | 0.019 | 0.9946 | 2.8698 |
| 0.10 | 4 | 403 | 1.013 | 13.67 % | 0.037 | 0.9904 | 2.8635 |
| 0.10 | 5 | 274 | 1.021 | 12.89 % | 0.043 | 0.9965 | 2.8859 |
| 0.10 | 6 | 330 | 1.010 | 10.55 % | 0.058 | 0.9963 | 2.9070 |
| 0.20 | 1 | 540 | 1.006 | 13.28 % | 0.047 | 1.0090 | 3.0616 |
| 0.20 | 2 | 362 | 1.016 | 13.67 % | 0.056 | 1.0094 | 2.9929 |
| 0.20 | 3 | 502 | 1.015 | 13.67 % | 0.037 | 1.0252 | 2.9634 |
| 0.20 | 4 | 312 | 1.015 | **9.38 %** | 0.071 | 1.0005 | 2.8984 |
| 0.20 | 5 | 301 | 1.012 | 11.72 % | 0.044 | 1.0056 | 2.9228 |
| 0.20 | 6 | 391 | 1.012 | 12.89 % | 0.039 | 1.0079 | 2.9916 |
| 0.30 | 1 | 233 | 1.019 | 14.06 % | 0.053 | 1.0277 | 3.1579 |
| 0.30 | 2 | 229 | 1.017 | **6.64 %** | 0.099 | 1.0439 | 3.1682 |
| 0.30 | 3 | 233 | 1.001 | **6.64 %** | 0.053 | 1.0821 | 3.1368 |
| 0.30 | 4 | 267 | 1.010 | 12.11 % | 0.059 | 1.0214 | 2.9771 |
| 0.30 | 5 | **9** | **1.461** | 8.59 % | **0.473** | 1.0167 | 2.9011 |
| 0.30 | 6 | 96 | 1.028 | 13.28 % | **0.336** | 1.0332 | 3.1403 |

| `f` | n | mean acc | sd (95 % CI) | var | median | span | rms MCSE |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.05 | 6 | 13.41 % | **0.94** (0.59–2.32) | 0.89 | 12.89 | 2.34 | 0.048 |
| 0.10 | 6 | 13.80 % | **2.39** (1.49–5.86) | 5.71 | 13.28 | 7.03 | 0.048 |
| 0.20 | 6 | 12.44 % | **1.66** (1.04–4.08) | 2.76 | 13.09 | 4.29 | 0.050 |
| 0.30 | 6 | 10.22 % | **3.35** (2.09–8.21) | 11.19 | 10.35 | 7.42 | 0.243 |
| all | 24 | 12.47 % | 2.55 | 6.51 | 12.89 | 10.94 | — |

The sd intervals are the `χ²` interval on 5 df; the `f = 0.30` rms MCSE is inflated by the one
broken cell below.  Excluding that cell: `f = 0.30` n = 5, mean 10.55, sd 3.63; all 23 cells mean
**12.64**, sd **2.46**.

**Result 78 — direction, not magnitude, is the dominant variance component.**  The per-magnitude
accuracies are 12.89/12.89/12.89/12.89/13.67/15.23, then 10.55/12.89/12.89/13.67/15.23/17.58, then
9.38/11.72/12.89/13.28/13.67/13.67, then 6.64/6.64/8.59/12.11/13.28/14.06: within-magnitude sd
**0.94 / 2.39 / 1.66 / 3.35** points against per-`f` mean differences of 13.41 / 13.80 / 12.44 /
10.22 %.  A one-way decomposition puts **69 % of the total variance inside a magnitude** (within
MS 5.14 vs a between-magnitude component of 1.72, `F(3,20) = 3.00`, `p = 0.055`) — so the magnitude
effect is a ~3-point shift that this design cannot resolve, while the direction effect is
established at every magnitude.  R2p's sd 3.16 and R2r's 3.34 were therefore **mostly the
within-magnitude spread, not a magnitude trend**: the right summary of this family is "at fixed
`f`, accuracy varies across directions with sd 1–3.4 points", and the only sound way to report it
is a variance (or sd), not the 2-draw range R2p and R2r quoted.

**Result 79 — the spread is model, not Monte Carlo.**  Per-cell `MCSE(acc)` is **0.019–0.099**
(0.473 only in the broken cell), rms 0.048–0.050 per magnitude, so the direction sd is **20–50× the
Monte Carlo error** at `f ≤ 0.20`.  All 24 cells share the basis, the data hash and the rank-4
restriction, so the differences are the prior mean itself.  The readout is quantised at `1/256`
(0.39 points): 12 distinct values appear, with **seven** cells at exactly 12.89 %, which is why
`f = 0.05` shows an sd of 0.94 with a median equal to its mode.

**Result 80 — the variance grows with jitter even though the mean does not move much.**
Brown–Forsythe on the four groups gives `F(3,20) = 4.10`, `p = 0.020`: variances 0.89 / 5.71 / 2.76 /
11.19 are **not** homogeneous, and the trend is upward.  At `f = 0.30` the six cells split into
6.64/6.64/8.59 versus 12.11/13.28/14.06 — a bimodal look, roughly "two thirds of directions are
unharmed, one third lands where the rank-4 posterior mean predicts poorly".  The *mean* shift from
`f = 0.05` to `f = 0.30` is 3.19 points, Welch `t = 2.25` on 5.8 df, `p = 0.067` — borderline, and
weaker evidence than the variance result.  R2o/R2p/R2q's "per-`f` means are not monotone" survives
(13.41 < 13.80 > 12.44 > 10.22), but with 6 draws the downward step at `f = 0.30` is large enough
that a monotone-decline reading is also defensible; two directions could not tell these apart.

**Result 81 — the ambient pin does not guarantee mixing on every direction.**  23 of 24 cells sit at
`R̂ ≤ 1.031`; the exception is `f = 0.30, k = 5` (ESS **9**, `R̂` **1.461**, `MCSE(acc)` 0.473), so
even under the sane basis about **1 in 24** anchor directions produces a badly mixed chain, and its
8.59 % is chain error rather than a model reading.  `f = 0.30, k = 6` is not far off (ESS 96, `R̂`
1.028, MCSE 0.336).  The proper scores degrade with `f`, mean and spread both monotone
(Brier 0.9933 / 0.9956 / 1.0096 / 1.0375; NLL 2.876 / 2.897 / 2.972 / 3.080, sd 0.018 → 0.113), so
the accuracy *improvement* at mid magnitudes is not a calibration improvement — accuracy and NLL
correlate **−0.34** across the 24 cells.

**Result 82 — what this settles and what it does not.**  Settled: at one basis, one data split and
six directions per magnitude, the accuracy scatter is a **direction** effect with sd 0.9–3.4 points
(69 % of the variance), the magnitude effect on the mean is ≈ 3 points and not significant at
`n = 6`, and the direction scatter is 20–50× the Monte Carlo error — so R2p's "the accuracy spread
is the prior mean" should be read as "the location of the prior mean at fixed rms", not as a trend
in the jitter size, and R2r's "the worst anchor is `f = 0.30, k = 2`" is now a **tie** (6.64 % at
`k = 2` and `k = 3`) with a third cell at 8.59 %.  Not settled: which geometric property of the
direction predicts the loss (the shift's projection into the rank-4 basis, or the spread of the
resulting predictive logits, would be the candidates — neither was recorded); whether the
6.64 directions reproduce at another data split or under R2r's second basis (the same direction has
never been run twice under a different basis or split); whether the `f = 0.30` bimodality is a
threshold or a mixture; and the `laplace` arm was still not swept over directions, so its median
1.19× rests on two per magnitude.

### Recommendation (state after R2s: what to set, and how well each row is supported)

The per-R-proposal recommendation, one table so it is not scattered across results.  "Load-
bearing" means the row is a measurement that changed a decision; "null" means the dial was
implemented, verified and found not to matter, so it should stay off rather than be tuned.

| Question | Setting | Evidence | Status |
|---|---|---|---|
| proposal at rank 4, `λ ≤ 1` | `laplace` at the target's mode, `--pilot-shrink 0.5` (**587 / 373** vs isotropic 383 / 152; **419** vs 342 at `λ = 0.5`; **378** vs 165 at a 10 %-jittered anchor; **3.04–4.93×** vs isotropic, geomean **4.08×**, on the eight pinned-basis anchors; **median 1.19×** / geomean **1.19×** / 0.80–1.70× vs that basis's pilot arm — R2o's moving-basis geomean 1.42× was carried by two pilot collapses) | R2k, two seeds; R2l adds the `λ = 0.5` cell; R2m shows the win survives an anchor shift at 2.3× rather than 5.9×; R2q repeats it with the basis pinned. R2r does not re-run it under the second pin (that basis mixes at ESS 6–154, so its proper scores would measure mixing, not the proposal) | load-bearing |
| proposal at rank 4, `λ = 2` | no recommendation: the sign is seed-dependent (default seed 376 vs 351, **−6.6 %**; seed 7 172 vs 226, **+31 %**, and that mode search did not converge) | R2l's single cell; R2m retracts the `λ = 2` half of it | **withdrawn** |
| `--laplace-tau τ < 1` (curvature at an interior point) | `1` (the mode).  Rank 4 `τ = 0.5` is 428 against the mode's 587 and the isotropic 383; rank 8 has `H_ii ≤ 0` in 1/8 coordinates and collapses (0.6 % acc, ESS 5, `R̂` 17.96) | R2m: `1/√H_ii` is undefined at a saddle, and interior points of `Φ` are saddles | load-bearing |
| `--laplace-mix` / `--pilot-mix` (log-width blends) | `0` (off — use one endpoint or the other).  All eight mixed cells at or below their isotropic baseline: 5/254/352/271 at rank 4, 6/6/85/108 at rank 8 | R2m: the anchor half carries `H ≈ 0` widths of `1e4`; the pilot/mode blend only tightens the proposal | null (implemented, verified, harmful) |
| `--anchor-jitter` (and the anchor generally) | `0` for comparability with R2k/R2l; treat the anchor as **the** sensitivity to characterise, not a nuisance to jitter | R2m: `f = 0.10` moves accuracy 12.89 % → 14.45–15.23 %. R2n: 21 cells, 12.89 → 17.97 % at the default seed. R2o kills the trend reading: at one data split, 8 anchors span **11.72–19.92 %** accuracy and 8–406 ESS, within-magnitude gaps reach 6.6 points, and the magnitude means are not monotone; R2p pins the basis (`--basis-anchor`) and splits it: pinning removes the mixing collapses (pilot ESS geomean 2.95×, min 16 → 229, max `R̂` 1.742 → 1.036) but **not** the accuracy spread (6.64–17.58 %, sd 3.16). R2q runs the `laplace` arm under the same pin: its per-cell gain over the pinned pilot arm is 0.80–1.70× (median 1.19×, geomean 1.19×, 6/8 wins) and 3.04–4.93× over isotropic in every cell, so the typical proposal gain is the target, while the accuracy spread (unchanged, 7/8 cells identical) is the prior mean. R2r pins a *second*, non-ambient basis (`--basis-anchor-seed 1`): accuracy keeps its spread (sd 3.34, span 9.77, worst cell the same `f = 0.30, k = 2` anchor) but ESS collapses 6.6× (mean 60.6, `R̂` to 16.0) and NLL degrades 0.63 nats — so the accuracy attribution survives, the ESS gain was the **ambient** basis, and proper scores must be gated on `R̂`/ESS. R2s sweeps **six** directions per magnitude under the ambient pin: within-magnitude sd **0.94 / 2.39 / 1.66 / 3.35** points is **69 %** of the total variance, the per-`f` means (13.41 / 13.80 / 12.44 / 10.22 %) differ at only `F(3,20) = 3.00`, `p = 0.055`, the direction scatter is 20–50× the cell MCSE, and the variance grows with `f` (Brown–Forsythe `p = 0.020`) even though the mean barely moves — so the spread is the **direction** of the prior-mean shift, and it must be quoted as an sd, not a 2-draw range | **open, attributed: accuracy ← the direction of the prior-mean shift at fixed rms (sd 0.9–3.4 points); ESS ← the ambient basis specifically** |
| full covariance at the mode (`rotlaplace`) | off (diagonal `laplace` only): 112 vs 587 at rank 4, 6 vs 8 at rank 8 | R2l: the off-diagonal curvature is not the posterior's covariance; `m` shrinks to 0.27–0.38 | load-bearing |
| `--pilot-shrink s`, rank 4, pilot arm (retired) | `0` or `0.5` — they tie (383 vs **399**) | R2e; R2i adds `s = 0` at 383 and pooled shapes at 362/17 | superseded by `laplace` |
| `--pilot-shrink s`, rank 8 | `0` (raw 6 → **154**) | R2e, R2f; `s = 0.5` is worse than either end (64/16 with pooling) | load-bearing |
| `--pilot-shrink s`, rank ≥ 16 | no claim | R2g ladder is monotone but every cell is at the ESS floor | **withdrawn** |
| `--pilot-reps n` | `1` (off) | R2g: 14/128/140 at rank 4, 5/7/6 at rank 8 — no fix where it matters | null |
| `--pilot-scale c` | `1` (off) | R2h: warm-up absorbs the scale (`m` halves at `c = 2`), ESS/predictives unchanged; `c = 4` leaves warm-up off-target | null |
| `--pilot-ebayes`, or any pooled shape | off (`s` alone) | R2i: weighted and unweighted pooled shapes both land at the single-pilot ESS (17/16 vs 15, 8/6 vs 6), below isotropic at three rank/seed pairs | null |
| pilot shape vs no shape | no shape: `s = 0` | R2h/R2i: every pilot-derived shape loses 5–20× in ESS; the shape is reproducible across replicates, so this is not a noise problem | load-bearing |
| proposal at rank 8 | isotropic `s = 0` (154); **not** `laplace` (8 / 7, `R̂` 1.56–4.45) | R2k: the diagonal is right at rank 4 and fails at rank 8. R2n's `--width-report` corrects the reason: the rank-8 mode's widths are the same order (norm 0.4771 vs 0.2797, 2.5× vs 3.3× spread, β 0.054–0.11 vs 0.059–0.10), so the collapse is mixing (`R̂` 4.446 at 45.5 % acc), not a too-narrow proposal | load-bearing |
| `--mode-steps` / `--mode-tol` | 2 000 / 1e-3 (the search printed with each cell must say `converged`) | R2k: rank-4 and rank-8 searches converge in 1 157 / 1 132 steps; the gradient must be `NormGrad::Exact`. R2l: `λ = 2` needs 1 508, so 2 000 is the floor, not slack | load-bearing |
| which proposal at ranks 4–8 | `pilot` with the `s` above, not `curv` | R2: curvature at a stationary anchor gave 0.1 % acceptance and ESS 7 | load-bearing |
| any rank-16 shrinkage claim | none: the rank is not resolvable, and the ladder is withdrawn | R2j: the fixed (proposal-independent) arm has ESS 5–14 and `R̂` 1.30–1.91 even at 8 chains × 8 000; the same kernel at two seeds moves BMA by 5.9 points, more than the ladder's 3.5 | **not resolvable** |

What is *testable* rather than measured is pinned by `cargo test -p pcnl --example
subspace_adaptation`: the shipped defaults are the identity path (`--pilot-reps 1`,
`--pilot-shrink 1`, `--pilot-scale 1`, no `--pilot-ebayes`), the pooled-shape estimator is
isotropic at `s = 0`, it reproduces the plain estimate when the replicates agree, it shrinks
disagreeing coordinates harder than agreeing ones, it falls back to the single-pilot
estimator below two replicates, and a uniform scale provably leaves the width *shape*
unchanged — the algebra that makes the R2h cell a null.  The ESS/NLL rows above are
measurements, not test assertions, and are regenerable from the commands below.

## R7 — decision transformers: the rank-4 adaptation cell (2026-11-13)

R7.4's first measured cell, and R7.5's verdict on it: the DT adaptation arm is **stopped** on the
negative branch, because the posterior-averaged policy `p̄` does not beat a temperature-tuned point
DT on held-out action NLL at any of three seeds.  The cell is `pcnl/src/dt_subspace.rs` and
`examples/dt_subspace`; the ambient-target readout of R7.3 (`dt_arm::DtCell`, `examples/dt_cell`)
is untouched and its numbers above stand as readout wiring, not as this comparison.

**The cell.**  Same MDP, model and corpus as R7.3's `DtCell` (rank 4, `λ = 1`, `σ = 0.35`, model
`d = 32`, `ff = 64`, 2 layers, 5 actions, 4 RTG bins, `T = 16`, `P = 17 408`), but the target is the
*adaptation* corpus (768 train / 128 dev / 128 test transitions, drawn by `DtCell::new`, **not** the
96/32/32 of R7.4's protocol table — `DtCell`'s splits are the ambient ones and changing them would
change every hash quoted above).  The informed basis is a rank-4 PCA over a ladder of short
full-batch momentum paths started at the anchor and pinned there; a single path is one-dimensional
and `pca_basis` returns `Degenerate { component: 1 }`, which is why the ladder exists.  The random
control is `--bases random`, same rank, its own seed.  The point baseline is Adam on the same
corpus, selected on dev over `lr ∈ {0.001, 0.003, 0.01}` with dev checks every 25 steps — untuned
Adam overfits this corpus, so the point arm is the best dev-selected run, and its budget is split
across the three learning rates, making it cost-matched to a single chain.  Temperature is tuned on
dev over `{0.5, 0.75, 1, 1.5, 2, 3}`.  The ensemble is four members at a quarter of the budget each.

**Result 83 — screening at `1 × 1 000` is not a measurement here, so it is reported as direction
only.**  Thin 10 over a 500-step burn leaves 50 draws; across seeds 7/8/9 the informed arms reach ESS
1.4 / 4.9 / 1.4 (`fixed`) and 11.4 / 5.7 / 29.1 (`pilot`), `laplace` 1.6 / NaN / NaN, and the
random control 1.4–40.0 — all far under the 400 gate.  Test NLL, with the point baseline and the matched-evaluation ensemble:

| seed | point | ensemble | informed `fixed` | informed `pilot` | informed `laplace` | random `pilot` |
|---|---|---|---|---|---|---|
| 7 | 0.9340 | 1.0431 | 0.9660 | 0.9440 | 0.9525 | 1.5179 |
| 8 | 1.0924 | 1.0386 | 1.1200 | 1.1114 | 1.1666 | 1.5610 |
| 9 | 1.2197 | 1.2380 | 1.2934 | 1.2815 | 1.3800 | 1.6288 |

Two things this *does* establish, because they are gap-sized rather than 0.02-nat-sized: the
informed basis beats the random control by 0.35–0.57 nats on every seed (a basis result, as R7.0
item 2 predicted — the restriction, not the sampler, decides), and the `pilot` proposal beats
`fixed` on every seed (0.022 / 0.009 / 0.012 nats).  The screening rank order is exactly the
promoted one, but none of the screening rows is quotable as a posterior result; every gate column
reads `—`.

**Result 84 — `laplace` collapses at rank 4 in the DT target, so R2k's classifier win does not
transfer.**  Six cells (three seeds × two bases) at `--laplace-shrink 0.5`:

| seed / basis | accept | ESS | mode | `H_ii ≤ 0` | test NLL |
|---|---|---|---|---|---|
| 7 informed | 0.734 | 1.5 | 1 500 steps (**cap**) | 0/4 | 0.9466 |
| 8 informed | 0.004 | 2.3 | 1 500 (**cap**) | 4/4 | 1.2035 |
| 9 informed | 0.000 | NaN | 1 500 (**cap**) | 2/4 | 1.3530 |
| 7 random | 0.092 | 22.4 | 142 (converged) | 4/4, `β` 1.0 | 1.5174 |
| 8 random | 0.184 | 39.6 | 119 (converged) | 3/4, `β` 1.0 | 1.5632 |
| 9 random | 0.344 | 16.4 | 128 (converged) | 2/4, `β` 1.0 | 1.6316 |

The informed mode search runs into its cap with `φ` still falling (1 219 → 874, `max|∇|` 116 →
0.016) — the anchor is far from the mode in a 17 408-dimensional space, and the DT's likelihood is
much flatter there than the classifier's, so the diagonal curvature at the stopped point is not
the posterior's.  In the random basis the mode converges but 2–4 of 4 coordinates have `H_ii ≤ 0`
and `β` clamps to 1, which makes the proposal an isotropic one.  Either way the arm is a
**proposer defect at this target**, recorded as such and carrying no score.  This closes the
question R7.5's "not claimed" list left open: the rank-4 `laplace` win is target-specific.

**Result 85 — the promoted informed `pilot` cell passes the sampling gates; `fixed` does not.**
4 chains × 6 000 steps, burn 500, thin 5 (4 400 kept), 6 000 point evaluations, 192 rollouts:

| seed | proposal | accept | ESS(Φ) | `R̂` | gate |
|---|---|---|---|---|---|
| 7 | fixed | 0.819 | 142.4 | 1.023 | fail |
| 7 | pilot | 0.289 | **590.9** | **1.004** | **pass** |
| 8 | fixed | 0.826 | 89.7 | 1.045 | fail |
| 8 | pilot | 0.305 | **792.4** | **1.003** | **pass** |
| 9 | fixed | 0.821 | 121.0 | 1.014 | fail |
| 9 | pilot | 0.266 | **858.8** | **1.002** | **pass** |

The 4 000-step budget of the protocol table was not enough: pilot ESS there was 335.3 / 377.1 / 537.1,
below the 400 floor at two of three seeds, so the cell was extended to 6 000 (+175 s per cell).
The gap between `fixed` and `pilot` is the whole difference between a converged and an unconverged
row, and it is stable across seeds — as in R2, the `fixed` proposal is the one that fails.

**Result 86 — the paired test: `p̄` ties the calibrated point baseline on NLL at all three seeds,
and wins the strictly proper rules on one.**  Paired per-step difference, `p̄(pilot) − point-temp`
on the same 128 test steps (positive = `p̄` worse):

| seed | NLL | Brier | spherical |
|---|---|---|---|
| 7 | +0.0079 ± 0.0182 (+0.4σ) | +0.0066 ± 0.0108 (+0.6σ) | +0.0079 ± 0.0073 (+1.1σ) |
| 8 | −0.0171 ± 0.0455 (−0.4σ) | **−0.0543 ± 0.0150 (−3.6σ)** | **−0.0319 ± 0.0081 (−4.0σ)** |
| 9 | +0.0814 ± 0.0563 (+1.4σ) | +0.0023 ± 0.0204 (+0.1σ) | −0.0023 ± 0.0129 (−0.2σ) |

Unpaired test NLL / top-1 / ECE: seed 7 point 0.9340 / 0.719 / 0.1324 vs `p̄` 0.9420 / 0.719 /
0.1479; seed 8 point-temp 1.1273 / 0.641 / 0.1602 vs `p̄` 1.1102 / 0.672 / 0.0991; seed 9
point-temp 1.2037 / 0.609 / 0.1070 vs `p̄` 1.2852 / 0.594 / 0.1042.  The anchor starts at 1.52 /
1.57 / 1.64 and the ensemble at 1.37 / 1.26 / 1.52 on test.

That is a **tie on the primary criterion** wherever it is not a loss: no seed reaches 2σ on NLL in
either direction.  Seed 8 is the informative one — `p̄` is better calibrated (Brier and spherical
both past 3.5σ, ECE 0.099 vs 0.160) while its NLL is indistinguishable from the tempered point
model's.  A cooled model and a posterior average can agree on log-loss and disagree on the proper
rules when the mass sits on different steps; this cell cannot separate those two explanations, and
nothing here licenses the stronger reading that `p̄`'s calibration is the mechanism.

**Result 87 — the return half is inconsistent across seeds and does not rescue the arm.**
192 rollouts at `x0 = [1, 0.5]`, paired against the behaviour policy (`±` is the paired MCSE):

| arm | seed 7 | seed 8 | seed 9 |
|---|---|---|---|
| behaviour | −5.9483 ± 0.2079 | −5.7080 ± 0.2521 | −6.0906 ± 0.2435 |
| anchor-sample | −3.3552 ± 0.0801 | −3.8675 ± 0.1034 | −3.9080 ± 0.1103 |
| anchor-argmax | −2.4515 ± 0.0191 | −2.5163 ± 0.0201 | −2.4882 ± 0.0193 |
| point-sample | −8.1501 ± 0.3407 | −11.5433 ± 0.7306 | −14.2354 ± 0.7710 |
| point-argmax | −9.6159 ± 0.1411 | −3.8016 ± 0.0754 | −7.9654 ± 0.1612 |
| `p̄`-sample | −7.8071 ± 0.3173 | −6.4616 ± 0.2747 | −7.3448 ± 0.2901 |
| `p̄`-argmax | −7.4954 ± 0.0829 | −6.5290 ± 0.0990 | −5.5447 ± 0.0802 |
| optimal | −2.3431 ± 0.0165 | −2.3530 ± 0.0172 | −2.3697 ± 0.0160 |

Paired vs behaviour: `p̄`-argmax **+0.5458 ± 0.2454** on seed 9 and **−0.8210 ± 0.2674** on seed 8
(both resolved, of opposite sign), `p̄`-sample −1.8588 ± 0.3135 / −0.7536 ± 0.2693 on seeds 7/8
(seed 9 −1.2542 ± 0.3238 — resolved worse on all three).  The only arm that beats behaviour on
every seed is the *anchor* (untrained on the adaptation corpus), and the exact optimum is 2.35–
2.37: the adaptation corpus is not where the return lives.  The averaged policy's returns are
therefore reported and **not** used: the plan's "calibration-only" branch needs both samples and
averages to move, and here one moves and the other does not.

**Result 88 — verdict: R7.5's negative branch.**  None of the three seeds reaches a 2×MCSE NLL
advantage, two of three are point estimates in the wrong direction, and a tie counts as not
positive by the acceptance text.  Per the plan: record the cell, **stop the DT adaptation arm**
pending a *target* hypothesis (a different corpus, basis target or restriction), not a new sampler
knob.  What would be a target hypothesis is already spelled out in R7.6 — head-only vs
trunk-only bases, and the anchor-as-behaviour-policy axis — and none of it is a proposal change.

### The basis target: head-only, trunk-only, and their matched controls (R7.6 item 1, same day)

The first of R7.6's open questions.  `--bases head,trunk` restrict the *informed* path PCA to one
parameter block (the untied head `w_out [D, A]`, or everything before it), and
`--bases head-random,trunk-random` are their controls: a rank-4 random basis drawn **inside the same
block**, so the pair differs only in which vectors inside the block are used.  Matching the control
to the block matters — a full-space random basis has support the block basis cannot have.  4 chains
× 6 000 steps, thin 5, seeds 7/8/9, informed `pilot`, same adaptation corpus and baselines as above
(the `informed` rows reproduce Results 85–86 exactly, so the table is one run):

| seed | arm | test NLL | Brier | spherical | top-1 | ECE | ESS(Φ) | `R̂` |
|---|---|---|---|---|---|---|---|---|
| 7 | informed | **0.9420** | 0.4636 | 0.2665 | 0.719 | 0.148 | 590.9 | 1.004 |
| 7 | trunk | 0.9509 | 0.4608 | 0.2659 | 0.695 | 0.103 | 788.6 | 1.005 |
| 7 | head | 1.1253 | 0.5608 | 0.3332 | 0.633 | 0.179 | 543.0 | 1.008 |
| 7 | trunk-random | 1.5170 | 0.8210 | 0.5362 | 0.281 | 0.282 | 1081.8 | 1.000 |
| 7 | head-random | 1.4671 | 0.7822 | 0.5061 | 0.289 | 0.263 | 753.1 | 1.001 |
| 8 | informed | **1.1102** | 0.4979 | 0.2979 | 0.672 | 0.099 | 792.4 | 1.003 |
| 8 | head | 1.1386 | 0.5351 | 0.3239 | 0.672 | 0.125 | 657.4 | 1.002 |
| 8 | trunk | 1.1517 | 0.5000 | 0.2995 | 0.664 | 0.083 | 1019.2 | 1.002 |
| 8 | head-random | 1.4652 | 0.7065 | 0.4562 | 0.422 | 0.145 | **236.4** | 1.008 |
| 8 | trunk-random | 1.5643 | 0.7530 | 0.4848 | 0.414 | 0.204 | 910.2 | 1.004 |
| 9 | informed | **1.2852** | 0.5924 | 0.3595 | 0.594 | 0.104 | 858.8 | 1.002 |
| 9 | head | 1.3000 | 0.6125 | 0.3739 | 0.625 | 0.159 | 573.9 | 1.008 |
| 9 | trunk | 1.3549 | 0.5872 | 0.3553 | 0.617 | 0.085 | 902.4 | 1.003 |
| 9 | head-random | 1.5962 | 0.7562 | 0.4916 | 0.367 | 0.173 | **318.2** | **1.012** |
| 9 | trunk-random | 1.6207 | 0.7613 | 0.4966 | 0.359 | 0.192 | 584.0 | 1.009 |

The two bold gate cells (`head-random`, ESS 236.4 at seed 8 and 318.2 with `R̂` 1.012 at seed 9) are
reported as gate failures, not scores; every informed/block arm passes (ESS 543–1019, `R̂` ≤ 1.008).

**Result 89 — no block restriction beats the full-space informed basis, and the head alone is the
worst informed arm on two of three seeds.**  NLL cost of *not* using the whole path, `block −
informed`: seed 7 `+0.1833` (head), `+0.0089` (trunk); seed 8 `+0.0284`, `+0.0415`; seed 9 `+0.0148`,
`+0.0697`.  Trunk-only is within 0.007–0.070 nats of the full basis — the trunk carries essentially
all of it — while head-only costs 0.18 nats at seed 7 and is never better.  Paired against the
point-temp baseline (log NLL): head `+0.1913 ± 0.0478` (**4.0σ worse**) / `+0.0113 ± 0.0405` /
`+0.0963 ± 0.0492` (2.0σ worse); trunk `+0.0169 ± 0.0299` / `+0.0244 ± 0.0596` /
`+0.1512 ± 0.0785` (1.9σ worse at seed 9).  So the block axis is not a dial to turn: the rank-4
question R7.0 item 2 raised ("the restriction is the cost") is decided in the **trunk**, and the head
adds nothing at rank 4 — a head-only basis is a worse prior than the whole path, not a cheaper one.

**Result 90 — the informed-vs-control gap survives the block matching, and it is the largest effect
in the table.**  Block informed vs its own in-block control: seed 7 `−0.059` (head) / `−0.566`
(trunk); seed 8 `−0.327` / `−0.413`; seed 9 `−0.296` / `−0.266` nats.  Six of six informed arms beat
their matched random control by 0.27–0.57 nats, so the gain is the *directions* and not the
coordinates they live in — a control drawn in the wrong space would have left this open.  The
trunk control also mixes best of any cell (ESS 584–1082) and is the worst predictor, which is the
same "mixing is not accuracy" dissociation R2r recorded.

### The RTG table alone is not a basis target (R7.6, same day)

The last block named in R7.6's question is the return-conditioning table `e_rtg [G, D]` — 128 of the
17 408 coordinates at `G = 4, D = 32`.  `Block::Rtg` spans it from `DtShapes::layout` (not from
`rtg_bins * d`), and it is a *sub-block* of the trunk rather than a third partition, which the basis
test now asserts.  Same cell, same protocol, `--bases rtg,rtg-random`:

| seed | point-temp | full-space informed | `rtg` | `rtg-random` | informed − control (in block) | ESS (`rtg` / control) | `R̂` |
|---|---|---|---|---|---|---|---|
| 7 | 0.9340 | 0.9420 | 1.4819 | 1.5062 | +0.0243 | 1258.2 / 1625.8 | 1.004 |
| 8 | 1.1273 | 1.1102 | 1.5459 | 1.5650 | +0.0191 | 1798.7 / 1879.5 | 1.002 |
| 9 | 1.2037 | 1.2852 | 1.6248 | 1.6349 | +0.0101 | 1526.1 / 1508.3 | 1.001 |

**Result 94 — the RTG table carries essentially none of the informed directions.**  Restricting to it
costs +0.540 / +0.436 / +0.340 nats against the full-space informed basis and leaves the posterior
+0.42 / +0.42 / +0.55 nats behind the point baseline (paired +7.0σ / +5.3σ / +4.9σ, all three
significantly *worse*).  Its informed-over-control gap is +0.010…+0.024 nats, against +0.342
(head) and +0.566 (trunk) at the same rank and seeds — a factor of 20–50 smaller.  So the ordering
from Results 89–90 sharpens: the informed content is in the trunk and, within the trunk, not in the
return-embedding table; a rank-4 chart drawn from `e_rtg` alone is barely distinguishable from a
random one.  That is consistent with R7's own reading of a DT — the return is the *conditioning*
input, not where the learned policy lives — and it means the remaining candidate inside the trunk is
the state encoder plus the transformer blocks, which the block decomposition cannot isolate further
without splitting the per-layer products.

**Result 95 — all four cells pass the gates** (ESS 1 258–1 880, `R̂` ≤ 1.004), and both controls have
*more* ESS than their informed counterparts here (1 626 vs 1 258 at seed 7), so this is a score
comparison and not a mixing artefact.  Also note the RTG path norm is small (1.057 → 1.130 of the
anchor's 14.70): the block's own fine-tuning motion is a tiny part of the full path, which is the
mechanical reason its principal directions are weak.

### The anchor as behaviour policy (R7.6, last item, same day)

R7.6's other named axis: a DT anchor is a corpus, so the analogue of R2n–R2s's "anchor direction" is
the **behaviour policy the anchor corpus was drawn from**.  `DtCell::with_policies` (new) names the
two policies separately; `--anchor-feedback/--anchor-epsilon` sweep the anchor side only.  The
adaptation corpora and the target never move, and the point baseline adapts from each anchor (so
the comparison is anchored, as R7.4 specifies).  4 × 6 000, thin 5, informed `pilot`, seeds 7/8/9:

| anchor policy | seed | anchor NLL | point-temp NLL | `p̄` NLL | paired `p̄ − point` (log) | ESS(Φ) | `R̂` |
|---|---|---|---|---|---|---|---|
| 1.0 / 0.1 (default) | 7 | 1.5240 | 0.9340 | 0.9420 | +0.0079 ± 0.0182 (+0.4σ) | 590.9 | 1.004 |
| | 8 | 1.5693 | 1.1273 | 1.1102 | −0.0171 ± 0.0455 (−0.4σ) | 792.4 | 1.003 |
| | 9 | 1.6371 | 1.2037 | 1.2852 | +0.0814 ± 0.0563 (+1.4σ) | 858.8 | 1.002 |
| 0.3 / 0.1 (weak) | 7 | **0.9356** | 0.9419 | 0.9413 | −0.0006 ± 0.0265 (−0.0σ) | 815.4 | 1.002 |
| | 8 | **1.1820** | 1.1242 | 1.0766 | −0.0477 ± 0.0282 (−1.7σ) | 549.9 | 1.005 |
| | 9 | **1.3772** | 1.2491 | 1.2560 | +0.0069 ± 0.0474 (+0.1σ) | 527.4 | 1.006 |
| 2.0 / 0.1 (sharp) | 7 | 2.2897 | 0.9365 | 0.9369 | +0.0004 ± 0.0325 (+0.0σ) | 786.9 | 1.003 |
| | 8 | 2.0754 | 1.1231 | 1.1076 | −0.0155 ± 0.0445 (−0.3σ) | 630.0 | 1.006 |
| | 9 | 2.0711 | 1.1993 | 1.2825 | +0.0832 ± 0.0513 (+1.6σ) | 605.0 | 1.007 |
| 1.0 / 0.3 (noisy) | 7 | 1.5057 | 0.9799 | 0.9466 | −0.0333 ± 0.0321 (−1.0σ) | 828.2 | 1.006 |
| | 8 | 1.4760 | 1.0802 | 1.0949 | +0.0146 ± 0.0408 (+0.4σ) | 814.9 | 1.004 |
| | 9 | 1.5510 | 1.2042 | 1.2648 | +0.0606 ± 0.0398 (+1.5σ) | 896.3 | 1.001 |

**Result 91 — the axis moves the prior mean by up to 1.35 nats and the posterior by 0.034.**  The
anchor's own held-out NLL spans 0.9356–2.2897 at seed 7 (1.354 nats), 1.1820–2.0754 at seed 8
(0.893), 1.3772–2.0711 at seed 9 (0.694), so "which behaviour policy the anchor was fitted on" is a
first-order property of the prior mean.  `p̄` over the same twelve cells spans 0.9369–0.9466 (0.010),
1.0766–1.1102 (0.034) and 1.2560–1.2852 (0.029): an order of magnitude smaller, and roughly the
size of one cell's MCSE.  At rank 4 with 128 adaptation transitions the likelihood term dominates
the prior, so a badly mis-specified anchor is corrected rather than inherited — the opposite of
what R2n–R2s measured for the *classifier's* accuracy, where the prior mean was the accuracy.

**Result 92 — the best anchor is the one drawn from the adaptation behaviour, and matching is what
matters, not quality.**  The weak-feedback anchor (feedback 0.3, the same feedback the adaptation
corpus uses) is the best prior mean at all three seeds, 0.59 / 0.39 / 0.26 nats below the default
near-greedy one, while the sharp anchor (feedback 2.0 — a *better* policy in return terms) is the
worst by 0.69–0.77 nats.  So this is a corpus-match effect, not an anchor-quality effect; note that
the anchor's pretrain loss cannot see it, because each anchor is pretrained on its own corpus
(1.599–1.627 → 0.636–1.127 across the four).

**Result 93 — no anchor policy produces R7.5's positive verdict.**  The paired NLL against each
policy's own point baseline is +0.4σ / −0.4σ / +1.4σ (default), −0.0σ / −1.7σ / +0.1σ (weak),
+0.0σ / −0.3σ / +1.6σ (sharp), −1.0σ / +0.4σ / +1.5σ (noisy).  The best single cell is seed 8
under the weak anchor at −1.7σ, with Brier −2.6σ and spherical −2.5σ — a genuine improvement in the
proper rules, but on **one** seed and under 2σ, so it is neither the ≥3-seed 2×MCSE requirement of
R7.5 nor enough to reopen the arm.  All twelve cells pass the sampling gates (ESS 527–896, `R̂` ≤
1.007), so these are scores and not gate failures.  Every policy differs from the default by less
than 0.035 nats in `p̄`: the R7.4 deficit is **not** an anchor-corpus artefact, which was the most
plausible remaining "target hypothesis" for it.

### Does the weak-anchor proper-rule win replicate? (R7.6, same day)

Result 93's best cell was seed 8 under the weak anchor (−1.7σ in log NLL, −2.6σ Brier, −2.5σ
spherical).  A single seed at <2σ is not a result, so the same cell was run on three fresh seeds
(10/11/12), with the **default** anchor run on the same fresh seeds as the control — six seeds per
policy, 4 × 6 000, thin 5, informed `pilot`, `--rollouts 2`:

| anchor | seed | anchor NLL | point-temp | `p̄` | paired log | paired Brier | paired spherical | ESS(Φ) | `R̂` |
|---|---|---|---|---|---|---|---|---|---|
| weak 0.3/0.1 | 7 | 0.9356 | 0.9419 | 0.9413 | −0.0006 ± 0.0265 | +0.0034 ± 0.0137 | −0.0010 ± 0.0108 | 815.4 | 1.002 |
| | 8 | 1.1820 | 1.1242 | 1.0766 | −0.0477 ± 0.0282 | −0.0364 ± 0.0138 | −0.0229 ± 0.0090 | 549.9 | 1.005 |
| | 9 | 1.3772 | 1.2491 | 1.2560 | +0.0069 ± 0.0474 | −0.0281 ± 0.0187 | −0.0219 ± 0.0111 | 527.4 | 1.006 |
| | 10 | 1.4554 | 1.3481 | 1.3394 | −0.0088 ± 0.0402 | −0.0333 ± 0.0167 | −0.0280 ± 0.0111 | 703.4 | 1.008 |
| | 11 | 1.1520 | 1.0960 | 1.0703 | −0.0257 ± 0.0349 | −0.0251 ± 0.0177 | −0.0181 ± 0.0121 | 514.1 | 1.004 |
| | 12 | 1.0433 | 1.0050 | 1.0030 | −0.0019 ± 0.0365 | −0.0086 ± 0.0198 | −0.0054 ± 0.0121 | 686.8 | **1.012** |
| default 1.0/0.1 | 7 | 1.5240 | 0.9340 | 0.9420 | +0.0079 ± 0.0182 | +0.0066 ± 0.0108 | +0.0079 ± 0.0073 | 590.9 | 1.004 |
| | 8 | 1.5693 | 1.1273 | 1.1102 | −0.0171 ± 0.0455 | −0.0543 ± 0.0150 | −0.0319 ± 0.0081 | 792.4 | 1.003 |
| | 9 | 1.6371 | 1.2037 | 1.2852 | +0.0814 ± 0.0563 | +0.0023 ± 0.0204 | −0.0023 ± 0.0129 | 858.8 | 1.002 |
| | 10 | 1.7238 | 1.3466 | 1.3848 | +0.0383 ± 0.0247 | −0.0060 ± 0.0122 | −0.0041 ± 0.0090 | 708.0 | 1.005 |
| | 11 | 1.5925 | 1.1002 | 1.1148 | +0.0147 ± 0.0272 | −0.0138 ± 0.0115 | −0.0143 ± 0.0080 | 682.6 | 1.002 |
| | 12 | 1.4522 | 0.9900 | 0.9970 | +0.0070 ± 0.0228 | +0.0002 ± 0.0116 | +0.0006 ± 0.0073 | 716.7 | 1.009 |

**Result 96 — the log-NLL win does not replicate; the proper-rule win does.**  Pooled over the six
seeds (mean of the per-seed paired deltas ± the standard error of that mean), the weak anchor gives
log **−0.0130 ± 0.0083 (−1.6σ)**, Brier **−0.0214 ± 0.0063 (−3.4σ)** and spherical **−0.0162 ±
0.0044 (−3.7σ)**; the default anchor gives **+0.0220 ± 0.0139 (+1.6σ)**, −0.0108 ± 0.0092 (−1.2σ)
and −0.0073 ± 0.0057 (−1.3σ).  So seed 8's log-NLL margin was the favourable tail of a distribution
centred at −0.013 nats, which is *not* the >2×MCSE win R7.5 asks for; but the Brier and spherical
improvements are reproducible: five of six weak-anchor seeds are negative on both, the two fresh
seeds 10 and 11 are individually ≥2σ in Brier, and the pooled significance survives dropping the
one cell that fails the gate (Brier −3.4σ, spherical −4.0σ).  The weak-minus-default difference is
−0.035 ± 0.016 in log (−2.1σ) and −0.011 ± 0.011 in Brier, so the anchor policy *is* doing something
real — just not something that shows up as a likelihood win.

**Result 97 — that is R7.5's calibration-only branch, and the arm stays closed.**  Per-step
probabilities are better under the weak anchor (lower Brier and spherical loss, i.e. better
calibrated at the same decisions), while the NLL — the score R7.5's positive verdict is written
against — ties or loses.  The honest reading is that the anchor corpus buys better-calibrated
mixtures of a *wrong* ordering, not a better ordering; nothing here changes the R7.5 verdict.  One
cell, seed 12 under the weak anchor, misses the `R̂ ≤ 1.01` gate at 1.012 (ESS 686.8 passes), and is
reported as a gate failure rather than a score, though the pooled numbers hold without it.

**What this settles.**  The rank-4 restricted DT posterior is *samplable*: informed `pilot` reaches
ESS 591–859 at `R̂ ≤ 1.004`, so a score can be quoted at all.  The informed basis is worth
0.35–0.57 nats over a matched random control, consistently.  The `laplace` proposal that carries
the classifier at rank 4 does not work at this target (mode search capped far from a mode, 2–4/4
non-positive curvature), which is a target-specificity result about R2k, not a contradiction of it.
And the posterior-averaged policy, at the best rank and λ the plan chose, is **not better than a
temperature-tuned point model on next-action NLL**.  The basis target is not a way out: the
informed basis's content is in the trunk, head-only is worse on 2/3 seeds and never better, and
trunk-only lands within 0.07 nats of the full basis without beating it (Results 89–90).  Nor is the
anchor corpus: moving the anchor's behaviour policy across a 0.94–2.29 NLL range moves `p̄` byat most 0.034 nats and never reaches the 2σ bar (Results 91–93) — so the R7.4 deficit is not an
anchor-mismatch artefact.  Nor is the block inside the trunk: the RTG table alone costs +0.34…+0.54
nats and is 4.9–7.0σ worse than the point baseline, with an informed-over-control gap 20–50× smaller
(Results 94–95).

**What this does not settle.**  Whether a basis target other than the trunk does better (R7.6's
first item; not run) — the informed-over-random gap says the basis matters, not that this basis is
the best one.  Whether the NLL tie at seed 8 with a 3.6–4.0σ proper-rule win is a real calibration
effect: one seed is not a design.  Whether the return inconsistency (seed 8 vs 9) is the decoder's
64-of-4-400-draw subsample or the policies themselves — the rollout decoder was declared and
fixed at 64 draws because the full average is `rollouts × horizon × 4 400` forwards, and it was
not varied.  And correlate step scores are still uncorrected (R7.6, R3's `Γ_k`), so every paired
MCSE above is within-step-independent by assumption.

**Defects recorded.**  (i) `examples/dt_subspace` panicked in `policy_readout` on `--rollouts 0`
and `--rollouts 1` (an MCSE needs more than one rollout); it now asserts `--rollouts >= 2` up
front.  (ii) The corpus splits are `DtCell`'s 768/128/128, not R7.4's protocol 96/32/32; changing
them changes the ambient anchor, so the protocol table's row was not followed and this is stated
rather than patched.  (iii) The promoted budget is 4 × 6 000, not 4 × 4 000 (Result 85).  (iv)
`--laplace-shrink` is a separate dial from `--pilot-shrink`: the first laplace screening at
`--pilot-shrink 0` was silently a mode-scaled *isotropic* proposal, which is why the arm was
re-run at 0.5 and why the shared dial was split.

## Reproduce / verify

```bash
cd unfer
cargo test -p pcnl --lib --release                       # 131 unit tests, ~36 s after build
cargo test -p pcnl --locked                              # 171 tests (130 unit + 7 comparators
                                                         #  + 8 formal claims + 26 numerical)
                                                         # (debug build: ~40 min on 2 cores;
                                                         #  `--release` is ~6 min)
# The report quoted above used kappa = 0.55 (the pre-2026-11-04 default):
cargo run --release -p pcnl -- --kappa 0.55 --report-file out.txt --save-dir /tmp/traj
# The corrected default is kappa = 0.5; the schedule's budget is in the report.
cargo run --release -p pcnl -- --report-file out.txt
cargo run -p pcnl --example schedule_mobility            # the mobility/floor table
cargo run --release -p pcnl --example step_scale_control  # §6.5.2, reduced config
cargo run --release -p pcnl --example step_scale_control -- --reference  # the table
cargo run --release -p pcnl --example sphere_volume       # §6.4(ii)'s Haar volume
# the 2026-11-07 target-conditioning ladder: exact shell conditionals, R̂ < 1.05
for spec in "8.61e-4 4.3e-4" "2.3e-3 1.15e-3" "1e-2 3e-3" "7.9e-2 3e-3"; do
  set -- $spec
  cargo run --release -p pcnl --example basin_posterior -- \
    10000 --no-sweep --chains 2 --arm latitude --d0 $1 --step $2
done
# the cap-freeze control (0.0 % / 0.1 % acceptance, R̂ = inf), the radial
# profile out to 0.4 rad, and the data requirement (N ≈ 3.4e6) come from the
# same example — see the `scale` stage of scripts/pcnl-reproduce for the
# wide-grid, 4-chain and --shapes invocations:
cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 4 --cap
cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 4 --cap 2.3e-3
cargo run --release -p pcnl --example basin_posterior -- 200 --chains 1 --arm shell
cargo run --release -p pcnl --example mh_standard_target  # §6.5.3, the exact target
cargo run --release -p pcnl --example temperature_sweep   # §6.5.4(a), λ ladder
cargo run --release -p pcnl --example overdispersion      # §6.5.4(b), per-draw vs BMA
cargo run --release -p pcnl --example overdispersion -- 10000 --chains 4 --thin 10 \
    --kappa 0.55 --pretrain sgd --start jitter --seeds 1   # …at the published horizon
cargo run --release -p pcnl --example annealed_start      # §6.5.5, the ramp, ~12 min
cargo run --release -p pcnl --example cold_ladder -- 1000 --lambdas 1,16  # §6.5.6(a), 3 seeds
cargo run --release -p pcnl --example step_distribution   # §6.5.6(b), pCN vs ULA
cargo run --release -p pcnl --example mh_cold             # §6.5.6(c), the filter's bias
cargo run --release -p pcnl --example posterior_radius    # §6.5.6(c), the target's radial profile
cargo run --release -p pcnl --example adam_start          # the solution handed to the kernel
cargo run --release -p pcnl --example adam_start -- 2000 --chains 1 --thin 20 --beta0 0.03  # …at a small step
cargo run --release -p pcnl --example sphere_chain        # the Haar prior on directions
cargo run --release -p pcnl --example sphere_chain -- 600 --precond   # …the curvature metric
cargo run --release -p pcnl --example curvature_probe    # why the filter rejects at the solution
cargo run --release -p pcnl --example sphere_mode        # the sphere's mode, by Riemannian descent
cargo run --release -p pcnl --example subspace_bound     # the low-rank ceiling, f(m) and 1/(1−f)
cargo run --release -p pcnl --example basin_posterior    # the cone-conditioned predictive (the sweep, ~10 min)
cargo run --release -p pcnl --example basin_posterior -- --prior-is 2048  # π's own predictive: Haar, ~1 min
cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --arm shell  # the fixed-length jump (~45 min)
cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --arm density  # the fixed-density arm
PCNL_PROGRESS=1 cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --chains 4  # walk or drift? exponent, angle, split-R̂ (~18 min)
# Guidance: the drift budget r·A (A = κ/(P−2)) and where it runs out.
# The 3 000-step κ ladder (one chain each, ~2 min apiece):
for k in 0.3 30 60 100 160 240; do PCNL_KAPPA_GUIDE=$k \
  cargo run --release -p pcnl --example basin_posterior -- 3000 --no-sweep --arm guided; done
PCNL_KAPPA_GUIDE=100 cargo run --release -p pcnl --example basin_posterior -- 10000 --no-sweep --arm guided --chains 4  # 44.14 %, R̂ 1.79 (~20 min)
PCNL_KAPPA_GUIDE=100 PCNL_LIVE_GRAD=1 cargo run --release -p pcnl --example basin_posterior -- 3000 --no-sweep --arm guided  # the live field (~12 min)
cargo run --release -p pcnl --example basin_posterior -- 50000 --no-sweep --arm shell  # the relaxation-time test (~45 min)
cargo run --release -p pcnl --example sphere_chain -- 400 --precond --beta 0.003 --start adam --seeds 2 --chains 2  # the R̂
cargo run --release -p pcnl --example sphere_chain -- 400 --precond --seeds 2  # the metric ladder
cargo run --release -p pcnl --example sphere_chain -- 10000 --precond --beta 0.003 --start adam --seeds 2 --chains 2  # the 10k cell, R̂ 2.894 (~1 h 40 m)
cargo run --release -p pcnl --example dimension_budget  # the exact cap law, the purse, m* (~3 min)
# R7.4's rank-4 DT adaptation cell (~10 min per seed at 4 x 6000).
cargo run --release -p pcnl --example dt_subspace -- \
    --seeds 7,8,9 --bases informed,random --steps 1000 --burn 500 --thin 10 \
    --pilot-steps 300 --evals 300                     # screening; ESS 1-40, directional only
cargo run --release -p pcnl --example dt_subspace -- \
    --seeds 7,8,9 --chains 4 --steps 6000 --burn 500 --thin 5 --bases informed \
    --proposals fixed,pilot --evals 6000 --rollouts 192  # the promoted table and the paired delta
cargo run --release -p pcnl --example dt_subspace -- \
    --seeds 7,8,9 --bases informed,random --proposals laplace --steps 1000 --burn 500 \
    --thin 10 --pilot-steps 300 --evals 300 --rollouts 2  # the laplace collapse, Results 84
cargo run --release -p pcnl --example dt_subspace -- \
    --seeds 7,8,9 --chains 4 --steps 6000 --burn 500 --thin 5 \
    --bases informed,head,trunk,head-random,trunk-random --proposals pilot \
    --evals 6000 --rollouts 2                          # R7.6's basis target, Results 89-90
for spec in "0.3 0.1" "2.0 0.1" "1.0 0.3"; do set -- $spec; \
  cargo run --release -p pcnl --example dt_subspace -- \
    --seeds 7,8,9 --chains 4 --steps 6000 --burn 500 --thin 5 --bases informed \
    --proposals pilot --evals 6000 --rollouts 2 --anchor-feedback "$1" --anchor-epsilon "$2"; done
                                                       # R7.6's anchor axis, Results 91-93
cargo run --release -p pcnl --example dt_subspace -- \
  --seeds 7,8,9 --chains 4 --steps 6000 --burn 500 --thin 5 --bases rtg,rtg-random \
  --proposals pilot --evals 6000 --rollouts 2                   # R7.6's RTG block, Results 94-95
cargo run --release -p pcnl --example dt_subspace -- \
  --seeds 7,8,9 --chains 4 --steps 6000 --burn 500 --thin 5 --bases informed \
  --proposals pilot --evals 6000 --rollouts 2 --anchor-feedback 0.3 --anchor-epsilon 0.1
cargo run --release -p pcnl --example dt_subspace -- \
  --seeds 10,11,12 --chains 4 --steps 6000 --burn 500 --thin 5 --bases informed \
  --proposals pilot --evals 6000 --rollouts 2 --anchor-feedback 0.3 --anchor-epsilon 0.1
cargo run --release -p pcnl --example dt_subspace -- \
  --seeds 10,11,12 --chains 4 --steps 6000 --burn 500 --thin 5 --bases informed \
  --proposals pilot --evals 6000 --rollouts 2                   # weak-anchor replication, Results 96-97
./scripts/pcnl-cold-10k                                   # §6.5.6(c), the 10k-step table
cargo run --release -p pcnl -- quick --no-mh --lambda 16  # §6.5.5's cold ladder
cargo run --release -p pcnl -- --steps 1000 --chains 2 --no-mh --lambda 4  # published
cargo run --release -p pcnl --example dump_trajectory -- /tmp/traj/cosine_nGPT-lite_chain0.safetensors
```

Gates: `cargo fmt -p pcnl -- --check` and
`cargo clippy -p pcnl --all-targets -- -D warnings` are clean.

**One entry point.**  `./scripts/pcnl-reproduce [--all|STAGE…]` runs the
commands above as named stages, one log each under `--out` (default
`/tmp/pcnl-reproduce`), with the wall time printed per stage and a failing
stage stopping the run — `--list` names them, `--all` is the ~4–6 h full set,
and the default subset is the fast gate.  The 2026-10-20 block went stale
because a quoted run had no one-command path back to it; this is that path.
The 2026-11-15 R9/d-track rows below are reached with
`./scripts/pcnl-reproduce d-track edge0-campaign r9-campaign` (the first two
are ~45 s; `r9-campaign` is the declared 15–20 min run); they are not in the
default subset.

## R9 — small Edge0 architecture with RLCD: executed mechanics (2026-11-15)

This section records what was **executed**, not what was planned. The R9
specification (`pcnl/EDGE0_RLCD_PLAN.md`, `pcnl/RESEARCH_PLAN.md` §R9) and the
implementation packets (`pcnl/DEVELOPMENT_PLAN.md` E0–E7, D0–D7) remain the
forward work order; the entries below are the packets whose code and checks now
exist. The stopped R7 DT arm above and the separate R8 sampler-theorem scope are
unchanged.

### Implemented and gated

| packet | artifact | gate |
|---|---|---|
| E0 | `pcnl/src/edge0_world.rs` — finite hypothesis worlds, exact conditional law, joint law, disjoint train/dev/calibration/test manifest, declared utilities | `oracle_matches_independent_enumeration`, `more_observations_never_raise_conditional_entropy`, `correlated_questions_are_not_a_product_of_marginals`, `manifest_splits_are_disjoint_and_ids_precede_questions`, `utility_decisions_and_regret_bound` |
| E1 | `pcnl/src/edge0_model.rs` — grouped latent D/U MoE, deterministic top-2 routing + shared expert, question-independent state encoder, packed/separate question attention, four-path policy, Noul/Choice/Score heads, parameter census, `StateCacheKey` | `parameter_census_is_complete_and_below_one_million`, `typed_outputs_are_finite_normalized_and_schema_failures_surface`, `packed_and_separate_questions_agree`, `question_order_and_choice_permutation_equivariance`, `ordinal_level_order_is_a_stochastic_order`, `dense_soft_routing_matches_top_k_when_every_expert_is_selected`, `state_cache_key_covers_weights_and_inference_settings` |
| E2 | `pcnl/src/edge0_rlcd.rs` — exact four-path marginal, Brier U-statistic, exhaustive estimator expectation, REINFORCE/direct identities, independent `FiniteFixture` | `u_statistic_expectation_equals_marginal_brier`, `estimator_gradient_equals_exact_and_finite_difference_gradients`, `model_estimator_gradient_matches_exact_marginal_gradient` (candle autodiff over all 256 path tuples), `correctness_reward_is_not_a_proper_scoring_rule` |
| E3/E4 (mechanics) | `pcnl/src/edge0_campaign.rs`, `pcnl/examples/edge0_campaign.rs` — shared supervised warm start, CE+Brier comparator, exact-marginal Brier control, RLCD arm, frozen-policy arm, world-clustered paired contrast | `campaign_arms_execute_and_report_finite_endpoints` |
| E5 (adapter) | `pcnl/src/edge0_potential.rs` — frozen-encoder head chart as `target::Potential`; exact latent-marginal NLL; frozen parameters get exactly zero gradient | `head_chart_exposes_the_exact_likelihood_and_freezes_upstream_parameters` |
| D0 | `pcnl/src/certificate.rs` — target/kernel/claim identities | `d0_matched_configurations_agree_and_changes_are_identified` |
| D1 (constants) | same — `Z_lo = a·exp(−b)`, `c = ω·Z_lo/M`, TV bound, analytic Gaussian evidence | `d1_certificate_has_positive_contracting_constants` |
| D1 (witness enclosure) | `pcnl/src/evidence_enclosure.rs` — rank-1/2 clamped-logistic witness; evidence evaluated in outward-rounded interval arithmetic (`Iv`, reduced Taylor `exp`), midpoint error `(2R)^d·d·h²/24·sup|g''|`, Mills-ratio tail, box ceiling and positive `c` | `d1_witness_evidence_enclosure_contains_the_analytic_value`, `d1_witness_predictive_enclosure_contains_the_chain_estimate`, `d1_interval_arithmetic_encloses_exp_and_the_evidence_is_certified` |
| D2 | `pcnl/src/dim_benchmark.rs` — `π_r ⊗ N(0,I_n)`, product gap `min(γ_r, 1−ρ)` | `d2_product_gap_is_independent_of_complement_dimension`, `d2_negative_control_with_shrinking_beta_does_not_inherit_the_constant` |
| D3 (canaries) | `pcnl/src/approx_pricing.rs` — exact Euler–Langevin Gaussian variance bias `1/(1−hλ/2)`, stability `h<2/λ`, independent gradient-noise inflation, declared error budget with open items | `d3_euler_langevin_canaries_match_closed_forms` (analytic formulas vs simulated chains) |
| D4 (benchmark) | `pcnl/src/linear_gaussian.rs` — coupled `y\|z,w ~ N(z+bᵀw,σ²)`: analytic slice, informed marginal, complement conditional, joint predictive; reduction error separated from sampler error | `d4_coupled_benchmark_separates_reduction_error_from_sampler_error` (including a Monte Carlo check of the joint predictive) |
| D5 (uncertainty) | `pcnl/src/policy_readout.rs::trajectory_clustered_delta` — CR1 trajectory-clustered paired SE, reducing exactly to `paired_delta` on singleton clusters | `d5_trajectory_clustered_delta_repairs_step_independence` |
| D5 (draw audit) | `pcnl/src/dt_subspace.rs::audit_posterior_draws` — nested draw counts at one rollout seed, naming per-step BMA sampling, argmax and once-per-episode posterior draw separately | `d5_draw_subset_audit_compares_the_three_decoders_at_fixed_histories` |
| D6 (canary certificate) | `pcnl/src/operator_cert.rs` — full-space `L²(N(0,v))` spectral certificate for the stable Gaussian canary (`0<a<1`, eigenvalues `a^k`, gap `1−a`, certified tail); plus the **scoped missing estimate** for the DT operator (tail, coupling, projection, arithmetic all absent) | `d6_canary_certificate_is_full_space_and_the_dt_scope_stays_open` |
| E5 (chain) | `SubspaceTarget<Edge0Potential>` with a one-column head chart, exact MH-pCN in `z` | `e5_head_chart_runs_an_exact_mh_pcn_chain` |
| E5 (reference record) | `pcnl/src/edge0_potential.rs::HeadChartReference` — D0 target/kernel identities and claim record, joint `β=1` prior-refresh primitive, sound clamp certificate `c = ω·exp(−λN·ln(1/floor))` (reported as vacuous at realistic `λN`), exact chi-square prior mass of a ball | `e5_reference_record_is_identified_and_its_rate_is_honest` |
| E5 (quadrature) | `head_chart_quadrature_1d` — one-column head-chart evidence and posterior-averaged predictive, coarse/fine grid gap, Gaussian tail bound, plug-in comparison | `e5_single_question_chart_matches_quadrature_reference` |

Commands actually run (2026-11-15):

```text
cargo test -p pcnl --test edge0_rlcd --test r8_certificates --test d_track
# 19 + 4 + 11 = 34 tests passed, 0 failed
cargo run -p pcnl --example edge0_campaign
# reduced-budget mechanics: 4 arms, paired RLCD−supervised +0.01632 ± 0.00870
R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_WALL_CAP=4500 \
  R9_OUT=target/r9_campaign.ndjson \
  cargo run --release -p pcnl --example r9_campaign
# declared campaign: 5/5 seeds, 920.5 s, 3,840,000 arm passes,
# across-seed RLCD−supervised +0.00223 ± 0.00184 (95 % t, df=4)
```

### D7 evidence record for the executed D/E packets

The plan's required fields, with **absent** measurements labelled as absent
rather than filled from expectation. Wall time is the test binary's own reported
duration (single-threaded targeted run); max RSS is the whole `cargo test`
process from `/usr/bin/time -M`, not an isolated allocator measurement.

| packet | assumptions / theorem | source, precision | observable, tolerance | error components | cost | verdict |
|---|---|---|---|---|---|---|
| D3 | Euler–Langevin Gaussian invariant law; exact variance recursion `v_{k+1}=a²v_k+2h+h²σ_g²` | `approx_pricing.rs`, `f64` | stationary variance vs 40k-step simulation, 5–6 % MC | discretisation bias and gradient-noise inflation separated; **no** uniform perturbation bound | 2 × 40 000 steps; test 0.09 s; RSS 369 MB | closed forms match; outside `h<2/λ` the variance is infinite and is reported so |
| D4 | linear-Gaussian joint prior and likelihood; slice is a restricted model | `linear_gaussian.rs`, `f64` | predictive mean within 0.01 and variance within 2 % of 60k joint-posterior draws | reduction error `|Δmean|+|Δvar|` reported separately from sampler MC; no finite TV/KL to the ambient law | 60k draws; part of the 6-test run | separation holds; factorized predictive is measurably wrong |
| D5 (SE) | CR1 cluster-robust variance with trajectory ids | `policy_readout.rs`, `f64` | clustered SE vs step-independent on a 200-step, 20-trajectory fixture | parameter-chain, data-seed and rollout variability are **not** folded into this SE; they stay separate | no campaign re-run | clustered SE > 1.5× naive; singleton clusters reproduce `paired_delta` exactly |
| D5 (draw audit) | common rollout seed, declared cyclic episode-draw schedule | `dt_subspace.rs`, `f32` model / `f64` returns | three decoders at 2 and 4 draws, 8 rollouts, `DtCell` seed 5 | decoder-subset error only; **not** chain convergence and **not** TV to a continuous posterior | 6k draws; test 38.00 s; RSS 60 MB | all three arms finite and paired; no success claim, values are diagnostic |
| D6 | Gaussian canary with `0<a<1`; detailed balance and Hermite spectrum | `operator_cert.rs`, `f64` | invariance residual ≤ 1e-12·(1+v); tail bound exact | arithmetic is `f64`; the full DT operator has **none** of the four D6 bounds | test 0.00 s | canary certified; DT scope explicitly uncertified |
| E5 (chain) | exact MH-pCN on a one-column head chart | `edge0_potential.rs` + `target.rs`, `f32` | 120 steps, finite φ, ≥1 acceptance | no evidence bound, no posterior-average protocol | 120 evaluations | chain executes; no posterior claim |
| D1 (witness) | rank-2 `Z = ∫σ(x)σ(−y)φφ = 1/4` exactly; rank-1 evidence and predictive enclosed | `evidence_enclosure.rs`, `f64` intervals; `exp` from a reduced 26-term Taylor series | analytic `1/4` inside the interval sum at `R=4, m=256` and `m=512`; `exp_enclose` brackets libm on `[-32,60]`; 60k-step chain estimate `0.7075` inside its predictive enclosure; certificate `c ≈ 1.6e-4` | evidence **and** predictive numerator are both outward-rounded `Iv` arithmetic (no `1e-10` envelope any more; `rounding_envelope == 0.0` is asserted); the only remaining declared uncertainty is `NODE_UNCERTAINTY = 1e-12`, applied to the witness node values the caller supplies; tail is the union-bound Mills estimate, loose for small boxes; rank-2 box error is loose | 3 D1 tests 8.94 s | evidence and the predictive numerator are theorem-grade in rounding, truncation and tail **relative to** the supplied node values; positive `c` published with its impractical rate |
| E5 (quadrature) | one-column head chart, exact MH-pCN; no proved `sup|Ψ'|` for the neural likelihood | `edge0_potential.rs`, `f32` model / `f64` quadrature | coarse/fine (`m=128/256`) evidence gap `< 5 %`; Gaussian tail `< 1e-8`; 900-step posterior mean within `0.08` of the quadrature predictive; `head_chart_enclosure_1d` returns a certified `[lo, hi]` at 512/1024 cells for a policy-head direction | truncation is now **bounded** only where the chart direction moves `policy_w`/`policy_b` alone: then `p(z) = Σ a_r u_r` is a convex combination of the frozen path distributions, so `\|(log p)'\| ≤ D = max d_r − min d_r` and `\|(log p)''\| ≤ 2D²` (no probability floor involved); other directions, and any direction touching the input layer, return `None` | test 1.45 s (enclosure test at 512/1024 cells; the enclosure brackets the reference evidence and predictive, and the finer grid does not widen it) | reference quadrature plus a genuine enclosure on the restricted (policy-head) chart; the neural chart as a whole still has no proved bound |
| E5 (reference) | D0 identities + D1 mixture constant; `Ψ` bounded by the target's own NLL floor | `edge0_potential.rs`, `f64` constants over `f32` model | chi-square prior mass vs `erf`, `1−e^{−x}`, `1−e^{−x}(1+x)` within 1e-12; `c < 1e-60` at `λ=8` | no verified likelihood sup on a ball; the clamp ceiling is sound but exponential in `λN`; no rounding certificate | test 2.00 s | identity and honest negative rate; the ball certificate contracts only when the caller supplies a verified ceiling |

Promotion decision: **none, and it stands after the declared campaign**. The
stopped R7 application stays stopped. On the declared 5-seed campaign RLCD's
paired Brier excess is worse than the supervised comparator with a 95 % interval
that excludes zero (`[+0.00039, +0.00407]`), which the plan treats as a stop
condition, and the reduced-budget mechanics agree in sign. The distribution and
gradient results are retained; no application or architecture promotion follows
from these rows.

### E3/E4 reduced-budget mechanics output (verbatim)

This is **not** the declared R9 campaign: `CampaignConfig::default` runs
`[24, 8, 8, 16]` worlds, 30 shared warm-start updates and 60 updates per arm on
the `tiny` configuration (`P = 1620`), single seed. It demonstrates that the
arms execute and that the endpoint is computed; it does not test the R9
hypothesis. A negative contrast is retained as produced.

```text
R9 E3/E4 — REDUCED-BUDGET MECHANICS (not the declared campaign)
model: P=1620 routed-active=64 | worlds train/dev/cal/test = [24, 8, 8, 16] | updates/arm=60 | batch=4 worlds | seed=20261115
warm start: 30 shared supervised updates | world manifest hash 0x6f856f7cbdbd69b3

arm                              Brier excess        NLL    pol-ent   path-ent     passes
supervised CE+Brier                   0.16932    0.97069    1.35449    0.84023        240
exact-marginal Brier control          0.17390    0.97094    1.35031    0.85511        240
RLCD (required)                       0.18565    0.97784    1.12527    0.85085        240
frozen-policy readout                 0.17555    0.97228    1.34569    0.85450        240

paired RLCD − supervised (world-clustered): +0.01632 ± 0.00870
```

Reading: at this budget RLCD is **worse** than the supervised comparator on the
primary endpoint. The plan predeclares that an inconclusive or worse contrast
ends promotion; nothing here is promoted. The analytic gradient gate (E2) passes
before this number is interpreted, which is the order the plan requires.

### Declared 5-seed R9 campaign (executed 2026-11-15)

The plan's bounded first campaign, run in release with the wall-time cap
declared **before** execution. Fixed data seed `20261115` gives one world
manifest for all five paired training seeds (manifest hash
`0xb0c4112e5addfc93`, identical in every row). Training seeds `9001`–`9005`
vary initialisation and path streams only.

```text
R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_WALL_CAP=4500 \
  R9_OUT=target/r9_campaign.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

Configuration and cost, as executed: `tiny` model (`P = 1620`, 64 routed-active),
worlds 2048/256/256/512, three typed questions per world, batch of **32 worlds
with all their questions**, 2000 updates per arm, 30 shared supervised warm-start
updates, all four arms. 768 000 arm passes per seed (192 000 per arm),
3 840 000 arm passes total, plus 2 880 shared warm-start passes per seed
(14 400 total) that the arm counter does not include. Evaluation adds 9 216
forward passes per seed (four arm evaluations and two paired per-world passes
over 1 536 test questions); those are derived from the fixed split, not
instrumented in the artifact. 920.5 s summed seed wall time out of the 4500 s
cap; peak RSS 15.9 MB; 5/5 seeds completed.

| seed | init seed | RLCD − supervised | world-clustered SE | wall |
|---|---|---|---|---|
| 0 | 9001 | +0.00113 | 0.00059 | 185.9 s |
| 1 | 9002 | +0.00369 | 0.00081 | 185.4 s |
| 2 | 9003 | +0.00398 | 0.00081 | 184.2 s |
| 3 | 9004 | +0.00150 | 0.00066 | 181.6 s |
| 4 | 9005 | +0.00086 | 0.00059 | 183.4 s |

Across seeds: **+0.00223 ± 0.00184** (mean, 95 % `t` on df = 4; SE of the mean
0.00066). The interval `[+0.00039, +0.00407]` excludes zero, so the prespecified
read is *worse*, not inconclusive: RLCD is worse than the supervised comparator
at this cost. Mean test Brier excess: supervised 0.12697, RLCD 0.12921; mean
NLL: 0.95593 versus 0.96257. The per-seed rows and summary are in the NDJSON
artifact; the summary row records `mean_diff`, `half95`, `total_passes` and
`cap_s`.

**Limitations of this execution.** The primary endpoint is the unweighted mean
per-question Brier excess, not the plan's fixed per-type weighting; the
calibration split was not used for abstention; and only the four existing arms
ran (no per-path proper-score or correctness-reward RL arm). Those are open reporting/experiment
items, not silent successes. The secondary endpoints that *were* computed are
reported next.

### R9 secondary endpoints (computed from the stored predictions)

The primary artifact carried only per-seed aggregates, so per-question
marginals were exported by `examples/r9_campaign.rs` (`R9_PRED_OUT`) and the
campaign was re-executed with the same declared configuration; the primary
contrast reproduced exactly (**+0.00223 ± 0.00184**, per-seed contrasts
identical, 941.9 s of the 4500 s cap, 5/5 seeds). Artifacts:
`target/r9_predictions.ndjson` (46 080 rows = 5 seeds × 4 arms × 2 304
questions, of which 768 dev, 768 calibration and 1 536 test per seed/arm; each
split is balanced at 1/3 per question type) and `target/r9_secondary.ndjson`.
The primary rerun wrote `target/r9_campaign_secondary.ndjson` so the first run's
file was not overwritten. A later export added the calibration split
(`target/r9_predictions_cal.ndjson`, 61 440 rows, with
`target/r9_campaign_cal.ndjson` and `target/r9_secondary_cal.ndjson`) so the
abstention threshold could be selected on data that is neither trained nor
scored on; the numbers below use that artifact.
`examples/r9_secondary.rs` then evaluates the endpoints without retraining:

```text
cargo run --release -p pcnl --example r9_secondary -- \
  target/r9_predictions_cal.ndjson target/r9_secondary_cal.ndjson
```

Definitions, all declared before the run: weighted Brier excess uses fixed
per-type weights 1/3 each (the balanced design makes it equal the unweighted
mean: dev per-type means 0.13337/0.06751/0.17948 for `noul`/`choice`/`score`
average to 0.12679, the unweighted dev mean); the ordinal score is the
ranked probability score
`Σ_{i<J−1}(Σ_{j≤i}(p_j−t_j))²/(J−1)` on `Score` questions; the temperature is
fitted on the **dev** split by a fixed grid 0.50–3.50 in steps of 0.05, ties to
the smaller value; regret is `u(truth, a*) − u(truth, a)` with the declared
`UtilityTable` and `a* = argmax` expected utility, bounded by `2·TV`.

| arm | weighted Brier | raw NLL | fitted T (range) | scaled Brier | scaled NLL | ordinal RPS | regret |
|---|---|---|---|---|---|---|---|
| RLCD (required) | 0.12921 | 0.96257 | 1.25–1.55 | 0.12822 | 0.96090 | 0.07860 | 0.01454 |
| exact-marginal Brier control | 0.12795 | 0.96021 | 1.30–1.40 | 0.12780 | 0.95987 | 0.07781 | 0.01454 |
| frozen-policy readout | 0.12792 | 0.96008 | 1.35–1.40 | 0.12781 | 0.95980 | 0.07777 | 0.01454 |
| supervised CE+Brier | 0.12697 | 0.95593 | 0.85–0.90 | 0.12755 | 0.95637 | 0.07815 | 0.01454 |

Paired `RLCD − supervised` across the five seeds (mean, 95 % `t`, df = 4;
negative favours RLCD):

| endpoint | difference | 95 % half-width | SE |
|---|---|---|---|
| weighted Brier | +0.002232 | 0.001843 | 0.000664 |
| raw NLL | +0.006640 | 0.003545 | 0.001277 |
| scaled Brier | +0.000665 | 0.000872 | 0.000314 |
| scaled NLL | +0.004532 | 0.002034 | 0.000733 |
| ordinal RPS | +0.000453 | 0.001130 | 0.000407 |
| decision regret | 0.000000 | 0.000000 | 0.000000 |

Every proper-score endpoint points the same way as the primary one. The
weighted Brier interval excludes zero (worse); the scaled Brier and ordinal RPS
intervals contain zero, so those two are **inconclusive** rather than worse.
RLCD's fitted temperature is much larger (≈1.4 versus ≈0.87): after temperature
scaling most of its Brier gap closes, which says its probabilities are
systematically over-sharp rather than ordered worse.

Decision regret is degenerate: all four arms choose the same action on every
question, so regret is the constant mean `0.01454` and the arm difference is
exactly zero. The declared `2·TV` bound holds on every test row (0 violations in
all 30 720 test predictions), but its mean is `≈0.462` — much looser than the realised
regret, because it bounds the action gap for an *arbitrary* truth. Utility is
therefore not evidence for or against RLCD at this budget; the plan's
"success claims cannot come from correctness-only rewards" condition is
unaffected, but the decision endpoint has no power here.

**Reliability (top-label, test, 20 fixed bins of 0.05 on confidence).** Every
arm is under-confident in the bins it uses (gap = accuracy − confidence is
negative almost everywhere), and confidence never exceeds ≈0.58, so bins above
0.6 are empty. Pooled across seeds:

| arm | ECE | MCE | non-empty bins (conf / acc) |
|---|---|---|---|
| RLCD (required) | 0.0263 | 0.0450 | 0.342/0.330, 0.422/0.377, 0.451/0.425, 0.541/0.531, 0.564/0.531 |
| exact-marginal Brier control | 0.0216 | 0.0476 | 0.342/0.322, 0.397/0.417, 0.404/0.374, 0.547/0.533, 0.550/0.503 |
| frozen-policy readout | 0.0186 | 0.0209 | 0.342/0.322, 0.398/0.378, 0.402/0.383, 0.548/0.531 |
| supervised CE+Brier | 0.0257 | 0.1252 | 0.344/0.296, 0.380/0.397, 0.420/0.401, 0.459/0.333 |

Paired RLCD − supervised ECE is **−0.00028 ± 0.01717** (inconclusive), so RLCD's
worse Brier is *not* explained by worse top-label calibration; the supervised
arm has the worst single-bin gap (MCE 0.125).

**Per-type reliability.** The miscalibration is concentrated in `Score`, not in
the binary questions. Test rows, pooled over seeds (`n = 2 560` per cell):

| arm | type | ECE | MCE | mean conf | acc |
|---|---|---|---|---|---|
| RLCD | noul | 0.0237 | 0.0329 | 0.5549 | 0.5312 |
| RLCD | choice | 0.0117 | 0.0117 | 0.3417 | 0.3301 |
| RLCD | score | 0.0434 | 0.0450 | 0.4243 | 0.3809 |
| exact-marginal control | noul | 0.0162 | 0.0476 | 0.5475 | 0.5312 |
| exact-marginal control | choice | 0.0195 | 0.0195 | 0.3417 | 0.3223 |
| exact-marginal control | score | 0.0291 | 0.0309 | 0.4033 | 0.3809 |
| frozen-policy readout | noul | 0.0162 | 0.0162 | 0.5475 | 0.5312 |
| frozen-policy readout | choice | 0.0196 | 0.0196 | 0.3419 | 0.3223 |
| frozen-policy readout | score | 0.0200 | 0.0209 | 0.4008 | 0.3809 |
| supervised | noul | 0.0192 | 0.0303 | 0.5504 | 0.5312 |
| supervised | choice | 0.0473 | 0.0611 | 0.3691 | 0.3656 |
| supervised | score | 0.0314 | 0.2310 | 0.4028 | 0.3809 |

So RLCD's worst type is `Score` (ECE 0.0434), but its `Score` calibration is
still better than supervised's (MCE 0.231 for `Score`); supervised's worst
type by ECE is `Choice` (0.0473). All arms are over-confident on `Score`
(mean confidence 0.40–0.42 against accuracy 0.381).

**Coverage vs abstention, with the calibration split.** Declared rule: abstain
when `max p < τ`; τ is selected on the **calibration** split (256 worlds, never
used for training, temperature or evaluation) as the grid point (0.00–0.90,
step 0.01) with the highest selective accuracy there, ties to the lower τ, then
applied unchanged to test. Calibration rows were added to the export and the
campaign re-run for this (`target/r9_predictions_cal.ndjson`,
`target/r9_campaign_cal.ndjson`); primary contrasts again reproduced exactly.
Selected τ per seed: RLCD 0.4, 0.5, 0.6, 0.6, 0.4. Paired test contrasts:
coverage **+0.18203 ± 0.13674**, selective accuracy **−0.00229 ± 0.05390** —
both inconclusive, and the coverage contrast remains partly a per-arm threshold
effect because τ is chosen per arm/seed.

A second declared rule picks τ on calibration as the **highest coverage whose
calibration risk is at most 0.55** (ties to the lower τ). On test (per-seed
means), that buys much more coverage at a slightly higher risk than the
accuracy-maximizing rule:

| arm | accuracy-max τ: cov / selAcc | risk ≤ 0.55 τ: cov / risk |
|---|---|---|
| RLCD | 0.241 / 0.508 | 0.667 / 0.544 |
| exact-marginal control | 0.280 / 0.517 | 0.667 / 0.544 |
| frozen-policy readout | 0.333 / 0.531 | 0.667 / 0.544 |
| supervised | 0.059 / 0.510 | 0.860 / 0.552 |

Supervised's risk-cap rule answers 86 % of test questions; the others answer
two thirds. The rule never breaches the cap on test by construction of the
calibration choice, not by evaluation on test.

Risk-coverage curve on test, pooled (coverage/risk = error rate among answered):

| τ | RLCD | control | frozen | supervised |
|---|---|---|---|---|
| ≤ 0.30 | 1.000/0.586 | 1.000/0.589 | 1.000/0.589 | 1.000/0.574 |
| 0.35 | 0.667/0.544 | 0.667/0.544 | 0.667/0.544 | 0.855/0.552 |
| 0.40 | 0.667/0.544 | 0.611/0.540 | 0.537/0.525 | 0.534/0.521 |
| 0.45 | 0.362/0.477 | 0.333/0.469 | 0.333/0.469 | 0.363/0.485 |
| 0.50 | 0.333/0.469 | 0.333/0.469 | 0.333/0.469 | 0.333/0.469 |
| 0.55 | 0.200/0.469 | 0.023/0.497 | 0.000/— | 0.164/0.473 |
| ≥ 0.60 | 0.000/— | 0.000/— | 0.000/— | 0.000/— |

The curve is monotone only in the weak sense that coverage falls; risk barely
improves (0.586 → 0.469) and never reaches the withholding regime a useful
selective classifier needs. By type at the selected τ, almost all coverage is
`noul`: RLCD covers 0.720 of `noul` (selective accuracy 0.522), 0.002 of
`score`, and 0.000 of `choice`; the frozen readout covers every `noul` at 0.531.
`choice` confidences sit near 0.34, below every selected τ, so that type is
always abstained on. These are the honest shapes; the calibration split fixes
the selection protocol, not the model's discriminative power.

**Per-type risk-coverage.** The flat aggregate curve is `choice` plus `score`:

* `choice` — RLCD risk is 0.670 at full coverage and never improves; coverage
  reaches 0 only because every confidence is below τ, not because abstention
  filters errors. Control/frozen behave the same (risk 0.678), supervised gets
  risk 0.575 at 0.585 coverage before falling to zero by τ = 0.45.
* `score` — risk 0.619 at full coverage; at τ = 0.45 RLCD answers 8.6 % at risk
  0.575, i.e. abstention buys 0.044 risk for 91 % coverage.
* `noul` — the only useful trade: risk 0.469 at full coverage, still 0.469 (RLCD)
  with 60 % coverage at τ = 0.55, while control keeps 7 % and frozen 0 %.

So the Score miscalibration is visible in the curve as the type where risk barely
falls as coverage is bought, and `noul` is the only type where selective
prediction does anything.

The per-type curves are also written to the artifact as
`{"kind":"r9-risk-coverage","arm":…,"qkind":…,"tau":…,"coverage":…,"risk":…}`
rows (240 rows for 4 arms × 3 types × 20 τ), so downstream tools do not have to
scrape the printed table.

The arithmetic behind these tables lives in `pcnl/src/r9_report.rs` and is
pinned by nine unit tests on synthetic fixtures: fixed bin edges
(`0.81 → bin 16`, `0.41 → bin 8`), a hand-computed ECE of `0.37` and MCE of
`0.42`, empty-bin reporting, NaN-confidence skipping, full-vs-answered
denominators, the `risk = 1 − selective accuracy` identity, the threshold tie
rule (`Some(0.81)`, not `0.9`, because every τ in `(0.80, 0.90]` answers the
same rows), and the coverage-target rule (an impossible risk cap returns
`None`; a cap already met at full coverage keeps τ = 0). The example is a thin
wrapper over that module.

### Interval-dual forward pass (head **and** encoder, second order)

`pcnl/src/chart_lipschitz.rs` carries `<value, first derivative, second
derivative>` intervals through the Edge0 **head stage** (policy softmax, path
embeddings, Noul/Choice/Score heads, ordered thresholds, the 1.5 ordinal link)
and through the **encoder stage**: LayerNorm → masked softmax attention →
grouped latent MoE / SwiGLU for the state tokens → packed question attention/FFN
→ per-question mean. The model exposes `PathOutput::rep`, so the head equations
were already written against a `rep` dual. The second derivative is what the
midpoint quadrature error needs, and it is why `head_chart_enclosure_1d` can now
certify encoder directions.

What it certifies today: `head_stage_psi_bound` returns a bound on
`sup |dΨ/dz|` over the chart box, and `head_stage_probability_interval` returns
the value interval, for **any** parameter slot — embeddings, state/question
positions, attention, router, latent experts, shared expert, question FFN and
the heads.

Discrete routing is over-bounded rather than assumed fixed. The routed output
is a convex combination `Σ_e g_e f_e` with `Σ_e g_e = 1`, so its value lies in
`[min_e f_e, max_e f_e]` for any selected set. Its derivative is
`Σ_e g_e f_e' + Σ_e g_e' f_e`, bounded component-wise by
`max_e |f_e'| + ½(Σ_k |l_k'|) max_e |f_e|`, using
`Σ_e |∂g_e/∂l_k| = 2g_k(1−g_k) ≤ ½`. Its second derivative adds
`max_e |f_e''| + (Σ_k |l_k'|) max_e |f_e'| + (3E² max|l'|² + ½ max|l''|) max_e |f_e|`
(from `|∂²g_e/∂l_j∂l_k| ≤ 3g_e`). All three bounds hold for any top-k set,
including one that changes inside the box, so the tie rule is not a certificate
assumption.

Containment is checked against the real model, not against its own arithmetic:

| test | what it pins |
|---|---|
| `dual_arithmetic_matches_closed_forms` | sigmoid/exp/product/sqrt interval and derivative rules |
| `head_stage_probability_interval_contains_the_model` | value interval brackets `p(z)` at 11 grid points, four train samples covering all three question types, directions on `score_w`, `noul_w`, `policy_w`, `score_thresholds` |
| `head_stage_psi_bound_contains_finite_differences` | the bound contains central differences of the single-sample `−log p_label` at 9 points per sample, and the largest measured derivative exceeds `1e-3` so the test is not vacuous |
| `encoder_directions_are_certified_by_the_dual_pass` | value intervals bracket `p(z)` for directions on `embed`, `question_attn_wq`, `question_w1`, `path_embed` at a narrow box (`R = 0.02`) |
| `encoder_psi_bound_contains_finite_differences` | the `question_w1` bound contains central differences at 9 points per sample, with a nonzero measured derivative |
| `state_encoder_dual_contains_the_model` | the state-encoder dual (LayerNorm, masked attention, routed MoE + shared SwiGLU) brackets `encode_state` values and its derivative brackets central differences, for an `moe_router` direction |
| `second_order_bounds_contain_finite_differences` | for a `question_w1` direction, `chart_derivative_bounds` contains central-difference `Ψ'`, `Ψ''`, `p'` and `p''` at five points per sample |
| `encoder_chart_enclosure_brackets_the_quadrature` | `head_chart_enclosure_1d` on a full-chart, one-sample `question_w1` direction brackets both the evidence and the predictive from `head_chart_quadrature_1d` |

Eight tests pass in 0.83 s. Bugs the containment tests caught earlier, all
fixed: `policy_w` is stored `[d, paths]` (flat index `j·paths + r`), thresholds
use the config's `max_levels−1` row stride (not the question's `levels−1`),
and the threshold block is offset per path.

`head_chart_enclosure_1d` now takes two paths: the analytic policy-only bound
when the direction moves only `policy_w`/`policy_b`, and the second-order dual
bound otherwise, where `(sup|Ψ'|, sup|Ψ''|)` and `(sup|p'|, sup|p''|)` come from
`chart_derivative_bounds` and the midpoint nodes are evaluated by
`chart_label_probability_at`/`chart_psi_at`. The general path costs
`(n_samples + 1)·cells` interval-dual forwards, so it is much more expensive
than the policy path; it refuses a box whose evidence lower bound is not
positive or whose marginal touches the floor.

**One honest limit remains.** The deployed `edge0_model::forward` pools
`state_rep` but the question blocks read the shared state *tokens* directly, so
a direction supported only on state-encoder weights has zero effect on the
marginal; the state pass is certified against `encode_state` itself, not
through the marginal.

**World-clustered SE.** The prediction rows already carry their `world` id
(dev 2048–2303, test 2560–3071, 3 questions each), so the paired difference is
now also clustered: questions are paired by `(world, qid)`, each world
contributes its mean difference, and the SE is the across-world SD over
`sqrt(512)`. Per-seed clustered SEs of the weighted-Brier contrast are
0.000595, 0.000806, 0.000812, 0.000661, 0.000586 — matching the primary
artifact's world-clustered SEs to `1e-5`. The clustered SE of the across-seed
mean is 0.000313 for weighted Brier and 0.001013 for raw NLL; the quoted
±0.001843 half-width remains the seed-level `t` interval, as in the primary
endpoint. Nothing here is a cluster bootstrap.

Not computed and not claimed: per-path proper-score and correctness-reward
arms, an abstention objective other than "maximize selective accuracy on
calibration", and the Lean4 lemmas.

### R9 control-arm budget (declared 2026-11-15, before execution)

The declared R9 campaign above ran four arms and stopped promotion. The two
remaining controls named by `EDGE0_RLCD_PLAN.md` §4 are now implemented in
`edge0_campaign::Arm` — `PerPath` (each sampled path's own readout scored by
Brier against the label, policy not differentiated) and `CorrectnessReward`
(`−R Σ log a_phi(r_i)` with `R = 1[argmax u_theta(.|r_i) = y]`, a deliberately
non-proper reward). This is a **new, separate budget**, not a rerun of the
promotion campaign and not an after-the-fact metric on its seeds.

Declared before running:

- 5 paired seed indices `0..4` (`init_seed = 9001 + k`), one frozen data seed
  `20261115`, and the same declared world manifest 2048/256/256/512 and frozen
test partition as the R9 campaign, so the rows are comparable but are the
  control budget's own run;
- 6 arms (`supervised`, `exact-marginal`, `RLCD`, `frozen-policy`, `per-path`,
  `correctness-reward`), 200 optimizer updates per arm, batches of 32 worlds
  (all their questions), shared supervised warm start of 30 updates;
- hardware wall-time cap **1200 s**, checked between seeds; no extra seed,
  width or architecture search;
- endpoint: each arm's test Brier excess and NLL, plus the unchanged paired
  world-clustered `RLCD − supervised` contrast. The two new arms are
  descriptive controls: they are not used to retune RLCD or to promote it.

```text
R9_SEEDS=5 R9_UPDATES=200 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=1200 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_controls.ndjson \
  R9_PRED_OUT=target/r9_controls_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-15).** All 5 declared seeds completed inside the cap: 5/5,
330.6 s wall, 310.1 s summed seed time, 18 MB peak RSS, 576 000
forward/backward passes. Artifacts: `target/r9_controls.ndjson` (one valid JSON
row per seed, now carrying all six arms) and
`target/r9_controls_predictions.ndjson` (23 MB of dev/calibration/test
marginals).

Five-seed means (test split, known-law outcomes):

| arm | Brier excess | NLL | paired vs supervised (mean, 95 % `t`, df=4) |
|---|---:|---:|---|
| supervised CE+Brier | 0.12801 | 0.96047 | — |
| exact-marginal Brier control | 0.12787 | 0.96003 | −0.00014 [−0.00023, −0.00004] |
| RLCD (required) | 0.12788 | 0.96001 | −0.00013 [−0.00028, +0.00002] (inconclusive) |
| frozen-policy readout | 0.12781 | 0.95989 | −0.00020 [−0.00029, −0.00011] |
| **per-path proper score** | 0.13214 | 0.96658 | **+0.00413 [+0.00260, +0.00565]** |
| **correctness-reward RL** | 0.13033 | 0.96424 | **+0.00232 [+0.00067, +0.00396]** |

Per-seed `RLCD − supervised` world-clustered contrasts: +0.00007±0.00008,
−0.00015±0.00009, −0.00021±0.00025, −0.00013±0.00008, −0.00025±0.00013; the
across-seed mean is −0.00013 ± 0.00015 (95 % `t`), i.e. **inconclusive at this
reduced control budget**. The declared 2000-update campaign above remains the
promotion decision (worse with an interval excluding zero); this run is a
different budget and does not revise it.

Reading: both missing controls are clearly worse than supervised training on the
same warm start, data and 200-update budget. The per-path proper score is the
worst arm (its own path distributions are individually over-confident relative to
the marginal objective), and the correctness-only reward is also worse, which is
what a non-proper reward should produce once the labels are noisy. The frozen
policy and the exact-marginal control are slightly *better* than the supervised
comparator at this shorter budget and RLCD is statistically
indistinguishable from it. No architecture or training promotion follows.

#### Secondary endpoints of the two controls (from `target/r9_controls_predictions.ndjson`)

The existing evaluator (`examples/r9_secondary.rs`) was run on the control
artifact; it reads no live model. Five-seed means over the test split:

| arm | wBrier | raw NLL | ordinal RPS | ECE | MCE | utility regret | `2·TV` bound | τ (cal) | cov | sel. acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | 0.12801 | 0.96047 | 0.07784 | 0.0178 | 0.0261 | 0.01454 | 0.46358 | 0.41 | 0.333 | 0.5312 |
| RLCD | 0.12788 | 0.96001 | 0.07778 | 0.0139 | 0.0218 | 0.01454 | 0.46234 | 0.41 | 0.333 | 0.5312 |
| exact-marginal | 0.12788 | 0.96003 | 0.07779 | 0.0149 | 0.0223 | 0.01454 | 0.46254 | 0.41 | 0.333 | 0.5312 |
| frozen-policy | 0.12781 | 0.95989 | 0.07773 | 0.0138 | 0.0187 | 0.01454 | 0.46216 | 0.40 | 0.333 | 0.5312 |
| **per-path** | 0.13214 | 0.96658 | 0.08354 | 0.0218 | 0.0447 | 0.01454 | 0.46541 | 0.38 | 0.333 | 0.5312 |
| **correctness-reward** | 0.13033 | 0.96424 | 0.07956 | 0.0167 | 0.0262 | 0.01454 | 0.46943 | 0.38 | 0.333 | 0.5312 |

Reading, labelled by endpoint family:

- **Calibration is not proper and does not rescue either control.** The per-path
  arm is the worst calibrated (ECE 0.0218, MCE 0.0447); correctness-reward has a
  slightly lower top-label ECE than supervised (0.0167 vs 0.0178) but a slightly
  higher MCE and a clearly worse Brier/NLL/RPS. That is exactly the failure mode
  the evaluator warns about: an arm can sharpen the bins it gets right while
  losing proper score.
- **Ordinal score agrees with the proper scores.** On `Score` questions the
  ranked probability score is 0.08354 for per-path and 0.07956 for
  correctness-reward, against 0.07784 supervised and 0.07778 RLCD. Neither
  control improves the ordinal ranking of levels.
- **Full-answer decision utility is identical because every arm takes the same
  action.** All six arms have mean regret 0.01454 with the `2·TV` bound around
  0.4636 and zero bound violations. The reason is visible in the artifact: under
  the declared utility tables every test question gets the same
  utility-optimal action from every arm (for the `Score` head the action is
  always level 2, and the two other heads collapse likewise). The probability
  differences therefore never change a full-answer decision. This is consistent
  with the already-recorded observation that `state_rep` is computed but not
  consumed by the question blocks; it means this artifact cannot rank the arms
  on decision quality, only on scoring rules.
- **The only decision-level difference is abstention.** The
  calibration-selected rule (maximise selective accuracy) picks τ≈0.38–0.41 and
  answers one third of the test questions for every arm, with the same selective
  accuracy 0.5312. Under the declared risk cap 0.55 the per-path arm answers
  less (coverage 0.333, risk 0.4688) than supervised (coverage 0.667, risk
  0.5439); correctness-reward sits at 0.600/0.5289. That is a coverage/risk
  trade-off, not a gain in per-question utility.

Artifact: `target/r9_controls_secondary.ndjson` (RLCD−supervised summaries)
beside the console tables in the run log.

One implementation bug surfaced and was fixed during this run: the seed-row
writer in `examples/r9_campaign.rs` emitted literal newlines and indentation
inside the JSON string, so seed rows were not valid JSON. The writer now emits
one line per row; the artifact above comes from a rerun of the same declared
budget with the same seeds and data seed (wall times differ only by measurement).

### R9 six-arm full-length budget (declared 2026-11-15, before execution)

The reduced control budget above ran 200 updates per arm. This second new
budget runs the **declared 2000-update length** with all six arms in the same
run, so the two controls can be compared at the cost the original campaign
declared rather than only at the reduced length. It is a separate budget: it
does not replace or extend the stopped four-arm R9 campaign, and its result
does not change that campaign's stop decision.

Declared before running:

- same frozen world manifest 2048/256/256/512, fixed data seed `20261115`,
  five paired seed indices `0..4` (`init_seed = 9001 + k`);
- six arms, **2000 updates per arm**, batches of 32 worlds with all their
  questions, 30 shared supervised warm-start updates;
- hardware wall-time cap **3600 s** (checked between seeds; derived from the
  measured 62 s per seed at 200 updates × 6 arms, which scales to about 620 s
  per seed at 2000 updates; 3600 s covers five seeds plus margin). No seed or
  width search, no extra runs;
- endpoint: unchanged world-clustered paired `RLCD − supervised` test Brier
  excess; the four other arms are reported descriptively. No promotion is made
  from this budget.

```text
R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=3600 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_full6.ndjson \
  R9_PRED_OUT=target/r9_full6_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-15).** 5/5 seeds, 2958.1 s of the 3600 s cap, 5 760 000
forward/backward passes, 18 MB peak RSS. Artifacts:
`target/r9_full6.ndjson`, `target/r9_full6_predictions.ndjson`.

| arm | Brier excess | NLL | paired vs supervised (mean, 95 % `t`, df=4) |
|---|---:|---:|---|
| supervised CE+Brier | 0.12697 | 0.95593 | — |
| RLCD (required) | 0.12907 | 0.96205 | +0.00210 [+0.00063, +0.00356] |
| exact-marginal | 0.12795 | 0.96021 | +0.00097 [+0.00072, +0.00123] |
| frozen-policy | 0.12792 | 0.96008 | +0.00094 [+0.00068, +0.00120] |
| **per-path** | 0.12866 | 0.96131 | +0.00169 [−0.00038, +0.00375] (inconclusive) |
| **correctness-reward** | 0.16920 | 1.13583 | +0.04222 [−0.00456, +0.08900] (dominated by seed variance) |

Per-seed `RLCD − supervised` contrasts: +0.00353±0.00085, +0.00233±0.00068,
+0.00273±0.00107, +0.00139±0.00062, +0.00050±0.00063. The upshot at the declared
2000-update length is **not** the reduced-budget one: RLCD is now clearly worse
than supervised, while per-path's disadvantage shrinks to statistical
indistinguishability (its 200-update gap was +0.00413 with a CI excluding 0).
Correctness-reward is still the worst arm and its mean is dominated by
instability: its NLL is 1.1358 against 0.9559 supervised, and its paired
interval is wide enough to be inconclusive on five seeds. No arm is promoted.

### `state_rep` consumed by the question blocks (declared 2026-11-15, before execution)

The reduced control budget's secondary analysis showed that all six arms take
the same utility-optimal action on every test question, consistent with
`state_rep` being computed and never read. The wiring change is now in the
model: after the packed question blocks, every question rep becomes
`rep + state_rep` before the heads (`edge0_model::forward_with`), and the
certified encoder dual mirrors it by pooling the state-encoder dual and adding
it to each question rep (`chart_lipschitz::dual_encode_questions`). No new
parameters are introduced; the parameter census and every shape are unchanged.

Hypothesis, declared before running: consuming the pooled state representation
will move the deployed marginal on state-encoder directions (checked by
`state_rep_is_consumed_by_the_marginal`) and, at the reduced control budget, will
change test Brier/NLL and the utility-optimal actions relative to the matched
unwired control run (`target/r9_controls.ndjson`, same seeds/data/budget).

Budget (separate from both campaigns above):

- 5 paired seeds `0..4` (`init_seed = 9001 + k`), data seed `20261115`, the same
  2048/256/256/512 manifest and frozen test partition;
- six arms, 200 updates per arm, 32-world batches with all questions, 30 shared
  supervised warm-start updates;
- hardware wall-time cap **1200 s**, checked between seeds; no extra seed or
  architecture search;
- endpoint: per-arm test Brier excess/NLL and the paired `RLCD − supervised`
  contrast, plus a paired arm-by-arm comparison against the unwired control
  artifact to isolate the wiring effect. No promotion is made from this budget.

```text
R9_SEEDS=5 R9_UPDATES=200 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=1200 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_state_rep.ndjson \
  R9_PRED_OUT=target/r9_state_rep_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-15).** 5/5 seeds, 531.7 s of the 1200 s cap, 576 000
passes, 24 MB peak RSS. Artifacts: `target/r9_state_rep.ndjson`,
`target/r9_state_rep_predictions.ndjson`.

| arm | Brier excess | NLL | wiring − unwired (paired, 95 % `t`) |
|---|---:|---:|---|
| supervised | 0.12804 | 0.96052 | +0.00003 [−0.00001, +0.00006] |
| RLCD | 0.12805 | 0.96036 | +0.00017 [−0.00023, +0.00058] |
| exact-marginal | 0.12791 | 0.96008 | +0.00004 [+0.00001, +0.00007] |
| frozen-policy | 0.12785 | 0.95993 | +0.00004 [+0.00001, +0.00007] |
| per-path | 0.13216 | 0.96660 | +0.00002 [−0.00015, +0.00019] |
| correctness-reward | 0.13035 | 0.96430 | +0.00002 [−0.00048, +0.00050] |

Paired `RLCD − supervised` in this run: +0.00001 ± 0.00038, inconclusive. Per-seed
contrasts: +0.00030±0.00015, −0.00023±0.00010, −0.00004±0.00035,
−0.00032±0.00020, +0.00035±0.00013.

Reading:

- **The wiring is causally active but numerically small.** Compared with the
  matched unwired control run, the maximum per-question probability change is
  0.030 (mean 0.0027) and the utility-optimal action agreement across arms drops
  from 100 % to 66.7 % of test questions (one third of questions now get
  different actions from different arms). So `state_rep` is genuinely consumed
  and reaches the decisions.
- **It does not improve the proper scores.** Every arm's Brier/NLL is within
  ±0.0005 of the unwired run, and most paired intervals contain zero; the two
  nominally positive intervals are +0.00004. The hypothesis of a scoring gain
  from the pooled state at this budget is rejected; no promotion.
- **The arm ordering is unchanged.** supervised ≈ exact-marginal ≈ frozen <
  RLCD ≈ per-path < correctness-reward, matching the unwired reduced-budget run.

Verification of the wiring itself: `chart_lipschitz` 9/9 (including the new
`state_rep_is_consumed_by_the_marginal` and the encoder enclosure),
`d_track` 11/11, `edge0_rlcd` 19/19, `rustfmt --check` clean on the changed
Rust files.

### `state_rep` at the declared 2000-update length (declared 2026-11-15, before execution)

The 200-update wiring run above showed the pooled state reaches the decisions but
moves proper scores by at most ~1e-4. This budget asks the remaining question:
does the wiring matter at the **declared 2000-update length**, where the unwired
run separated the arms (RLCD worse, per-path indistinguishable)?

Declared before running:

- same frozen manifest 2048/256/256/512, fixed data seed `20261115`, five paired
  seeds `0..4` (`init_seed = 9001 + k`);
- six arms, **2000 updates per arm**, 32-world batches with all questions, 30
  shared supervised warm-start updates;
- hardware wall-time cap **6000 s**, checked between seeds. The estimate comes
  from the measured 200-update wiring run (531.7 s for 5 seeds × 6 arms × 200
  updates = 6000 arm-updates, i.e. ~0.089 s per arm-update), which scales to
  ~5320 s at 2000 updates. No seed or width search;
- endpoint: the same world-clustered paired `RLCD − supervised` test Brier
  excess, plus a paired per-arm comparison against the unwired full-length run
  (`target/r9_full6.ndjson`, same seeds/data/budget) to isolate the wiring
effect at 2000 updates. No promotion is made from this budget.

```text
R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=6000 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_full6_state_rep.ndjson \
  R9_PRED_OUT=target/r9_full6_state_rep_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-16).** 5/5 seeds, 5676.3 s of the 6000 s cap, 5 760 000
passes, 24 MB peak RSS. Artifacts: `target/r9_full6_state_rep.ndjson`,
`target/r9_full6_state_rep_predictions.ndjson`.

| arm | Brier excess | NLL | wiring − unwired (paired, 95 % `t`) |
|---|---:|---:|---|
| supervised | 0.12724 | 0.95646 | +0.00027 [+0.00014, +0.00039] |
| RLCD | 0.12959 | 0.96305 | +0.00052 [−0.00057, +0.00160] |
| exact-marginal | 0.12762 | 0.95896 | −0.00033 [−0.00074, +0.00008] |
| frozen-policy | 0.12751 | 0.95828 | −0.00041 [−0.00088, +0.00006] |
| per-path | 0.12836 | 0.96052 | −0.00030 [−0.00073, +0.00013] |
| correctness-reward | 0.17741 | 1.15318 | +0.00822 [−0.00401, +0.02045] |

Paired `RLCD − supervised`: +0.00235 ± 0.00124, interval excluding no improvement
(RLCD worse). Per-seed contrasts: +0.00222±0.00070, +0.00274±0.00067,
+0.00379±0.00099, +0.00186±0.00064, +0.00112±0.00052.

Reading: at 2000 updates the raw addition is not a free gain. It makes
supervised slightly worse (a significant +0.00027), leaves RLCD's disadvantage
unchanged, and moves the other arms by ≤0.0004 in either direction. The
arm ordering is the same as the unwired run (RLCD worse than supervised,
per-path close, correctness-reward worst). No promotion; the learned-projection
hypothesis below asks whether a trained gate can do better than forcing the
pooled state in.

### Learned `state_proj` instead of the raw addition (declared 2026-11-16, before execution)

The raw `rep + state_rep` wiring moved decisions but not proper scores. This
hypothesis replaces it with a learned `d × d` projection: every question rep
becomes `rep + state_rep · W_s`, with `W_s` a new parameter slot (`state_proj`)
initialised like the other weights and trained by the same arms. The chart dual
mirrors the projection (`state_pool · W_s`), and the parameter census grows by
`d² = 64` at the campaign's `tiny` width (`d = 8`; still below one million).

Hypothesis, declared before running: the learned projection will improve test
Brier/NLL relative to the raw-addition run at the same reduced budget, because
the model can reshape or suppress the pooled state instead of only adding it.
This is a **new budget**, not a rerun of the addition budget (`target/r9_state_rep.ndjson`)
or the unwired control (`target/r9_controls.ndjson`).

Budget:

- 5 paired seeds `0..4` (`init_seed = 9001 + k`), data seed `20261115`, the same
  2048/256/256/512 manifest and frozen test partition;
- six arms, 200 updates per arm, 32-world batches with all questions, 30 shared
  supervised warm-start updates;
- hardware wall-time cap **1800 s**, checked between seeds (the extra `d²`
  parameter slot is small; the cap allows for it plus margin);
- endpoint: per-arm test Brier excess/NLL, the paired `RLCD − supervised`
  contrast, and paired arm-by-arm differences against the addition and unwired
  artifacts. No promotion.

```text
R9_SEEDS=5 R9_UPDATES=200 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=1800 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_state_proj.ndjson \
  R9_PRED_OUT=target/r9_state_proj_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-16).** 5/5 seeds, 538.2 s of the 1800 s cap, 576 000 passes,
25 MB peak RSS. Artifacts: `target/r9_state_proj.ndjson`,
`target/r9_state_proj_predictions.ndjson`.

| arm | Brier excess | NLL | projection − raw addition (paired, 95 % `t`) |
|---|---:|---:|---|
| supervised | 0.12801 | 0.96045 | −0.00003 [−0.00012, +0.00006] |
| RLCD | 0.12806 | 0.96042 | +0.00001 [−0.00041, +0.00043] |
| exact-marginal | 0.12787 | 0.96001 | −0.00004 [−0.00013, +0.00005] |
| frozen-policy | 0.12780 | 0.95986 | −0.00004 [−0.00012, +0.00004] |
| per-path | 0.13070 | 0.96440 | −0.00146 [−0.00598, +0.00306] |
| correctness-reward | 0.12958 | 0.96304 | −0.00077 [−0.00352, +0.00198] |

Paired `RLCD − supervised` in this budget: +0.00005 ± 0.00051, inconclusive. The
learned projection is statistically indistinguishable from the raw addition for
every arm (the largest point change is per-path's −0.00146, whose interval
contains zero and both signs). This rejects the declared hypothesis of a scoring
gain from the trained gate at the reduced budget; no promotion. The extra
`d² = 64` parameters also make each seed slightly slower (≈103 s vs ≈100 s),
but remain inside the cap.

### Learned `state_proj` at the declared 2000-update length (declared 2026-11-16, before execution)

The reduced-budget projection test rejected its hypothesis, but the raw
addition itself only became significantly harmful to supervised at the
2000-update length, so the projection has not been tested where the wiring
actually costs score. This budget reruns the learned projection at the
declared campaign length; it is a **new budget**, not a rerun of the 200-update
projection budget (`target/r9_state_proj.ndjson`), the raw-addition full run
(`target/r9_full6_state_rep.ndjson`) or the unwired full run
(`target/r9_full6.ndjson`).

Hypothesis, declared before running: at 2000 updates the trained `d × d`
projection (`rep + state_rep · W_s`) will score better than the raw addition
for at least the supervised arm, because the model can suppress or reshape the
pooled state where the fixed addition actively hurts; no arm is expected to
beat the unwired control. Decision rule: a paired 95 % `t` interval on
`projection − raw addition` per arm that excludes zero in the improving
direction for supervised, with the arm ordering otherwise unchanged.

Budget:

- 5 paired seeds `0..4` (`init_seed = 9001 + k`), data seed `20261115`, the same
  2048/256/256/512 manifest and frozen test partition as every R9 run above;
- six arms, 2000 updates per arm, 32-world batches with all questions, 30 shared
  supervised warm-start updates;
- hardware wall-time cap **7200 s**, checked between seeds (the measured
  raw-addition full run took 5676.3 s of its 6000 s cap; the projection's ≈1.3 %
  per-seed overhead puts the expected total near 5750 s);
- endpoint: per-arm test Brier excess/NLL and paired arm-by-arm differences
  against the raw-addition and unwired full-length artifacts. No promotion.

```text
R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=7200 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_full6_state_proj.ndjson \
  R9_PRED_OUT=target/r9_full6_state_proj_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed (2026-11-16).** 5/5 seeds, 4901.6 s of the 7200 s cap (4879.6 s
summed seed wall time), 5 760 000 forward/backward passes, 24–25 MB peak RSS.
Artifacts: `target/r9_full6_state_proj.ndjson`,
`target/r9_full6_state_proj_predictions.ndjson`, stage log
`target/r9-state-proj-full.txt`. Wall time came in *below* the raw-addition
full run's 5653.4 s despite the extra `d²` parameters; the earlier ≈1.3 %
per-seed overhead was machine noise, not a projection cost.

| arm | Brier excess | NLL | projection − raw addition (paired, 95 % `t`) |
|---|---:|---:|---|
| supervised | 0.12692 | 0.95602 | −0.00032 [−0.00057, −0.00006] |
| RLCD | 0.12800 | 0.96041 | −0.00159 [−0.00274, −0.00044] |
| exact-marginal | 0.12790 | 0.96010 | +0.00028 [−0.00011, +0.00068] |
| frozen-policy | 0.12788 | 0.95993 | +0.00038 [−0.00010, +0.00085] |
| per-path | 0.12788 | 0.96013 | −0.00048 [−0.00215, +0.00118] |
| correctness-reward | 0.17655 | 1.10058 | −0.00087 [−0.06454, +0.06280] |

The declared decision rule is met: for supervised the paired interval
excludes zero in the improving direction (−0.00032 [−0.00057, −0.00006]),
and RLCD improves even more (−0.00159 [−0.00274, −0.00044]); the other four
arms are statistically unchanged, with exact-marginal and frozen-policy
slightly *worse* in point estimate but not significantly so. Within this run
`RLCD − supervised` remains +0.00107 ± 0.00056 (per seed +0.00168, +0.00133,
+0.00069, +0.00107, +0.00061), so RLCD is still worse than supervised, but its
penalty is roughly halved from the raw addition's +0.00235 ± 0.00124.

Against the unwired full run, supervised is indistinguishable
(−0.00005 [−0.00037, +0.00027]) and RLCD is now *better* than the unwired
control (−0.00107 [−0.00194, −0.00020]); the raw addition had made RLCD worse
by +0.00235. So the trained projection recovers the RLCD loss the fixed
addition introduced and does not reproduce the supervised cost at 2000
updates. Per the declared rule there is still no promotion: RLCD loses to
supervised in the same run, and the ordering (supervised best; per-path,
frozen and exact tied; RLCD next; correctness-reward worst) is unchanged. The
remaining question is whether a *smaller* or gated projection can also remove
RLCD's residual gap rather than only returning it to the unwired level.

**Secondary endpoints (2026-11-16).** Re-scored from the stored predictions with
`r9_secondary`, artifacts `target/r9_full6_state_proj_secondary.ndjson` and
log `target/r9-secondary-proj-full.txt` (reproduced by the
`r9-secondary-proj-full` stage; never retrains):

| arm | ECE | MCE | RPS | scaled Brier | regret | cov@cal τ | sel. acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised | 0.0262 | 0.0871 | 0.07811 | 0.12756 | 0.01454 | 0.098 | 0.515 |
| RLCD | 0.0150 | 0.0417 | 0.07795 | 0.12780 | 0.01454 | 0.286 | 0.514 |
| exact-marginal | 0.0197 | 0.0293 | 0.07779 | 0.12776 | 0.01454 | 0.280 | 0.534 |
| frozen-policy | 0.0185 | 0.0255 | 0.07775 | 0.12778 | 0.01454 | 0.280 | 0.534 |
| per-path | 0.0191 | 0.0605 | 0.07778 | 0.12774 | 0.01454 | 0.274 | 0.523 |
| correctness-reward | 0.0943 | 0.3864 | 0.11690 | 0.13614 | 0.01454 | 0.667 | 0.456 |

Paired `RLCD − supervised`: scaled Brier +0.00024 ± 0.00026 (not significant),
scaled NLL +0.00347 ± 0.00060, ordinal RPS −0.00017 ± 0.00049 (not
significant), decision regret exactly 0, top-label ECE −0.0072 ± 0.0153 (not
significant), selective accuracy −0.0009 ± 0.0738, coverage at risk τ
−0.1801 ± 0.0376 and risk at that τ −0.00695 ± 0.00210; no regret-bound
violations. Reading: dev-fitted temperature scaling removes RLCD's raw-Brier
deficit (0.12780 vs 0.12756, interval containing zero), and the ordinal score
and decision regret are identical, so the residual gap is almost entirely
calibration plus a coverage/abstention trade-off, not different full-answer
decisions — the same conclusion the raw-addition control's secondary analysis
reached. This reinforces the no-promotion decision.

### State-dependent gate on the projected state at 2000 updates (declared 2026-11-16, before execution)

The full projection recovered most of RLCD's deficit but left it
+0.00107 ± 0.00056 worse than supervised in the same run. This budget replaces
the fixed addition with a state-dependent, per-dimension gate:
`rep + sigmoid(state_rep · W_g) ⊙ (state_rep · W_s)`, with `W_g` a new `[d, d]`
slot. `W_g` is appended **last** in the layout so every existing parameter keeps
its offset and a paired seed initialises all shared parameters exactly as in
the learned-projection run; only the new slot's draws are new. The chart dual
mirrors the gate with the same product rule (`sigmoid(state_pool · W_g) ⊙
(state_pool · W_s)`); `state_gate_is_consumed_by_the_marginal` pins that a
direction supported only on the gate moves the deployed marginal, and
`state_gate` is included in the encoder-containment directions. The census
grows by `d² = 64` over the projection run at the campaign's `tiny` width
(`d = 8`; `P = 1620` → `1684`; still below one million).

Hypothesis, declared before running: letting the model suppress the state
addition per dimension where it hurts will shrink the within-run
`RLCD − supervised` Brier gap below the projection run's +0.00107, because
RLCD can gate off the pooled state instead of being forced to consume it.
Decision rule: the paired 95 % `t` interval on the within-run
`RLCD − supervised` contrast has an upper bound below +0.00107 (ideally
including zero), and supervised does not regress significantly against the
projection run (paired interval on `gate − projection` for supervised
containing zero or favouring the gate). This is a **new budget**, not a rerun
of the projection budget (`target/r9_full6_state_proj.ndjson`) or of the
unwired/additions runs.

Budget:

- 5 paired seeds `0..4` (`init_seed = 9001 + k`), data seed `20261115`, the same
  2048/256/256/512 manifest and frozen test partition;
- six arms, 2000 updates per arm, 32-world batches with all questions, 30 shared
  supervised warm-start updates;
- hardware wall-time cap **7200 s**, checked between seeds (the projection run
  used 4901.6 s; the gate's `d²` slot is the same size as the projection's and
  expected to cost the same order);
- endpoint: per-arm test Brier excess/NLL, the within-run `RLCD − supervised`
  contrast, and paired arm-by-arm differences against the projection,
  raw-addition and unwired full-length artifacts. No promotion.

```text
R9_STATE_PATH=gate \
  R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=7200 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_full6_state_gate.ndjson \
  R9_PRED_OUT=target/r9_full6_state_gate_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

**Executed 2026-11-16 (stage `r9-state-gate`).** 5/5 seeds within the cap:
5826.3 s summed seed wall time (cap 7200 s; per seed 1147.5, 1158.1, 1173.7,
1169.4, 1177.6 s), 5,760,000 forward/backward passes, 24.4–25.4 MB RSS, one
world hash across seeds. Mean test Brier excess / NLL:

| arm | Brier | NLL |
|---|---|---|
| supervised CE+Brier | 0.12693 | 0.95602 |
| RLCD (required) | 0.12789 | 0.96013 |
| exact-marginal control | 0.12791 | 0.96009 |
| frozen-policy readout | 0.12788 | 0.95993 |
| per-path proper score | 0.12788 | 0.96013 |
| correctness-reward RL | 0.17044 | 1.07226 |

The declared decision rule is **not met**. The within-run `RLCD − supervised`
contrast is **+0.00096 ± 0.00020** (paired 95 % `t` over five seed differences;
per-seed differences 0.00099, 0.00106, 0.00067, 0.00101, 0.00106). Its upper
bound, +0.00116, is not below the predeclared +0.00107, and it does not include
zero. Against the projection run the change is indistinguishable from noise:
`gate − projection` is −0.00011 [−0.00064, +0.00041] for RLCD and +0.00000
[−0.00001, +0.00001] for supervised; exact-marginal, frozen-policy and
per-path are unchanged to 1e-5, and correctness-reward is −0.00611
[−0.01747, +0.00525]. Against raw addition the gate still helps
(supervised −0.00031 [−0.00057, −0.00006]; RLCD −0.00170 [−0.00291, −0.00049]),
and against the unwired full run it is −0.00005 [−0.00037, +0.00027] for
supervised and −0.00118 [−0.00250, +0.00013] for RLCD.

Diagnosis: the gate is close to a learned *scalar* on the state path. A gate
that starts at `sigmoid(0) = 0.5` and is optimised by the same loss as
`state_proj` is redundant with rescaling that projection — the two are not
separately identified — so it barely changes the solution. The small RLCD
improvement cannot be separated from the projection's own noise, and the
declared gap target is missed. Recorded as a negative result; no promotion, and
no rerun at another seed/data draw under this budget.

**Secondary endpoints (2026-11-16).** Re-scored from the stored predictions with
`r9_secondary`, artifact `target/r9_full6_state_gate_secondary.ndjson`, log
`target/r9-secondary-gate.txt` (never retrains):

```text
cargo run --release -p pcnl --example r9_secondary -- \
  target/r9_full6_state_gate_predictions.ndjson \
  target/r9_full6_state_gate_secondary.ndjson
```

| arm | ECE | MCE | RPS | scaled Brier | regret | cov@cal τ | sel. acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised | 0.0238 | 0.0873 | 0.07812 | 0.12757 | 0.01454 | 0.145 | 0.507 |
| RLCD | 0.0137 | 0.1475 | 0.07786 | 0.12775 | 0.01454 | 0.255 | 0.512 |
| exact-marginal | 0.0193 | 0.0414 | 0.07779 | 0.12776 | 0.01454 | 0.273 | 0.527 |
| frozen-policy | 0.0185 | 0.0309 | 0.07775 | 0.12778 | 0.01454 | 0.280 | 0.534 |
| per-path | 0.0194 | 0.0420 | 0.07778 | 0.12775 | 0.01454 | 0.273 | 0.527 |
| correctness-reward | 0.0863 | 0.3671 | 0.11112 | 0.13458 | 0.01454 | 0.667 | 0.450 |

Paired `RLCD − supervised`: weighted Brier +0.00096 ± 0.00020, raw NLL
+0.00412 ± 0.00072, scaled Brier +0.00019 ± 0.00024 (not significant), scaled
NLL +0.00331 ± 0.00068, ordinal RPS −0.00026 ± 0.00029 (not significant),
decision regret exactly 0, top-label ECE −0.0061 ± 0.0122 (not significant),
selective accuracy +0.0053 ± 0.0735 (not significant), coverage at calibration
τ +0.1095 ± 0.2845 (not significant), coverage at risk τ −0.1751 ± 0.0439 and
risk at that τ −0.00684 ± 0.00470; no regret-bound violations. This is the same
pattern as the projection run's secondary analysis: dev-fitted temperature
scaling removes the raw-Brier deficit (0.12775 vs 0.12757, interval containing
zero), while the ordinal score and decision regret are identical, so the
residual gap is calibration plus a coverage/abstention trade-off rather than
different full-answer decisions. The gate does not change that reading.

### Low-rank state projection at 2000 updates (declared 2026-11-16, before execution)

The gate was redundant with `state_proj`'s own scale. This budget tests a
capacity restriction that is not redundant with the full projection: the
`[d, d]` state slot is replaced by a factorization `rep + (state_rep · U) · V`
with `U: [d, r]`, `V: [r, d]` and `r = d/2`. At the campaign's `tiny` width that
is `r = 4`, so the state slot still holds `d·r + r·d = d² = 64` entries: the
parameter **count is identical** to the full projection (`P = 1684`), every
other slot keeps its offset, and a paired seed initialises every shared
parameter exactly as the projection, gate and raw-addition runs did. No slot is
inert — every entry of `U` and `V` multiplies into the addition. The chart dual
mirrors the factorization (`(state_pool · U) · V`) with the same product rule;
`low_rank_state_projection_is_consumed_by_the_marginal` pins that directions on
both factors move the deployed marginal and that the parameter count matches the
full projection, and `state_down`/`state_up` are added to the encoder
containment directions. `R9_STATE_PATH=lowrank` selects the variant; the
deployed default remains `full`, so earlier budgets still reproduce.

Hypothesis, declared before running: if the state contribution only needs ≤4 of
the 8 available directions, the low-rank path matches the projection's
supervised score and may reduce the within-run `RLCD − supervised` gap, because
the rank constraint stops the RLCD arm spending capacity on state directions
that do not generalise. If the state path genuinely needs more rank, supervised
will regress and the restriction is rejected.

Decision rule, declared before running: (a) **non-inferiority** — the paired
95 % `t` interval on `lowrank − projection` for supervised Brier has an upper
bound at or below +0.0002 (the projection's own run-to-run variation is of that
order); and (b) **gap target** — the paired interval on the within-run
`RLCD − supervised` contrast has an upper bound below +0.00107 (the projection
run's gap) and ideally includes zero. Both are reported whatever they say; no
promotion either way.

Budget:

- 5 paired seeds `0..4` (`init_seed = 9001 + k`), data seed `20261115`, the same
  2048/256/256/512 manifest and frozen test partition;
- six arms, 2000 updates per arm, 32-world batches with all questions, 30 shared
  supervised warm-start updates;
- hardware wall-time cap **7200 s**, checked between seeds (the state slot has
  the same size as the projection's; the projection run used 4901.6 s and the
  gate 5826.3 s);
- endpoint: per-arm test Brier excess/NLL, the within-run `RLCD − supervised`
  contrast, and paired arm-by-arm differences against the projection, gate,
  raw-addition and unwired full-length artifacts. No promotion.

```text
R9_STATE_PATH=lowrank \
  R9_SEEDS=5 R9_UPDATES=2000 R9_WARM_START=30 R9_BATCH_WORLDS=32 \
  R9_WALL_CAP=7200 R9_DATA_SEED=20261115 \
  R9_OUT=target/r9_full6_state_lowrank.ndjson \
  R9_PRED_OUT=target/r9_full6_state_lowrank_predictions.ndjson \
  cargo run --release -p pcnl --example r9_campaign
```

### Not executed (and therefore not claimed)

- **E3/E4 reporting beyond the computed table.** The declared campaign ran
  (above) and its weighted Brier, temperature-scaled baselines, ordinal score
  and utility/regret are computed from the stored predictions, together with
  top-label reliability and coverage vs abstention. The calibration split as an
  abstention set and the extra control arms (per-path proper score,
  correctness-reward RL) are now computed in the control and full-length
  sections above, at their declared budgets. No further seed, width or
  architecture search was run.
- **E6 (Lean4).** No finite probability/decision lemma was handed to, or
  elaborated by, the Lean specialist. Proposed identifiers in the plan remain
  proposals.
- **E7.** Only the mechanics above exist; there is no R9 archive of positive,
  negative or inconclusive *training* comparisons beyond this row.
- **E5 posterior results.** The adapter, a one-column exact MH-pCN chain, the
  D0 identities/claim record and the `β=1` prior-refresh primitive now exist.
  What does **not**: the reference mixture was never run as a chain, and there
  is no posterior-average gradient protocol. A **single-question** head-chart
  quadrature and predictive comparison now exist, and `head_chart_enclosure_1d`
  turns the quadrature into a certified interval — but **only** for charts
  whose direction moves `policy_w`/`policy_b`, where the path distributions are
  constant in `z`; it declares a `1e-6` `f32` bridge on the frozen head inputs
  and computes at 512 cells (the test also checks 1024). The second-order
  interval-dual pass now covers encoder directions through LayerNorm →
  attention → routed MoE (routing over-bounded by the convex-combination
  argument above), and `head_chart_enclosure_1d` uses it for any non-policy
  direction: `sup|Ψ'|`, `sup|Ψ''|`, `sup|p'|` and `sup|p''|` are certified over
  the box and the nodes are dual-pass evaluations. A direction supported only
  on state-encoder weights now reaches the marginal because `forward` projects
  the pooled `state_rep` through `state_proj` and adds it to every question rep,
  and the dual pass mirrors that (`state_rep_is_consumed_by_the_marginal`); the
  raw-addition and learned-projection variants both leave proper scores
  statistically unchanged at the reduced budget; at 2000 updates the raw
  addition hurts supervised significantly, while the learned projection
  improves supervised and RLCD against the raw addition and leaves the other
  arms unchanged; the state-dependent gate on that projection is
  indistinguishable from it and does not close the RLCD–supervised gap. The
  only evidence bound for a *ball* around the
  whole network remains the sound but vacuous clamp one (`c ≈ 0`); this chart
  enclosure is a one-dimensional box certificate, not a ball ceiling. The value and
  gradient now share `NLL_FLOOR` (they disagreed at `1e-12` vs `1e-6`), so the
  "exact mean-NLL gradient" claim holds at saturation too — but no gradient
  check at a saturated head was added.
  The D1 witness **evidence and predictive numerator** are now both
  outward-rounded interval arithmetic with explicit Taylor remainders and no
  libm assumption (`rounding_envelope == 0.0`); the surviving declared input is
  `NODE_UNCERTAINTY = 1e-12` on supplied node values. Its rank-2 box error is
  conservative, and the Mills union bound denies a certificate to boxes smaller
  than that bound. Nothing here certifies the neural chart's evidence.
- **D5 historical cells.** The repository stores no R7 draw tensors or cell
  artifacts, so the corrected uncertainty and draw-audit fields could not be
  added to the *same* historical cells without re-running their chains. Those
  numbers still carry the step-independent SE; the new estimator is gated on a
  synthetic fixture, not on reproduced R7 evidence.
- **D6 (DT operator).** Only the Gaussian canary has a full-space certificate.
  The DT invariant-law operator's retained-block tail, coupling, projection and
  `f32` spectral bounds are all absent and are recorded as such in
  `operator_cert::missing_dt_estimate`.
- **D7 (Lean).** No Lean module or axiom audit was run for these packets; the
  D7 table above covers only the executed Rust checks.
- **D7 (reproduction, narrower than it sounds).** The stages
  (`./scripts/pcnl-reproduce d-track`, `edge0-campaign`, `r9-campaign`,
  `r9-secondary`, plus `r9-state-rep-full`, `r9-state-proj-full`,
  `r9-state-gate`, `r9-secondary-proj-full`) exist and were run, and they
  regenerate the 34 test results,
  the six-arm mechanics table, the declared campaign and its secondary
  endpoints from the exported predictions. They do **not** regenerate the
  timings, memory figures or the per-packet verdicts in the D7 table — those
  were measured by hand and a rerun can move them; nor do they cover any Lean
  or DT-cell claim. `d-track` uses the debug profile, so its wall times are not
  comparable to the release numbers quoted elsewhere; the original four-arm
  `r9-campaign` is release and took 920.5 s of seed wall time inside its 4500 s
  cap. The later six-arm budgets above were invoked directly with their own
  env-declared budgets (2000-update run: 2958.1 s inside 3600 s; 200-update
  control and `state_rep` runs: 330.6 s and 531.7 s inside 1200 s), not by the
  script stage.
- **E6/E7 (repeated for clarity).** No Lean4 elaboration and no full R9 archive.

## R10 — frozen UltraBERT + calibrated MLP head (2026-11-18)

A new external-encoder probe fixes `dmis-lab/UltraBERT-Embed-150M`
(revision `b06b367…`, CLS/L2-normalised, 768-d) and samples the complete
multi-layer head on its embeddings.  The head is `768 → 32 → 16 → 4` with
`tanh` hidden activations (`P = 25 204` weights/biases, no frozen head layer).
The target is
`Φ(z,w)=λN·mean L(θ_map+σ(Bz+w))+½‖z‖²+½‖w‖²`; the full kernel uses
`mh_pcn_z_full` then `mh_pcn_w_full`, so every head weight is in the posterior
while the 150M encoder is fixed data.  The fixture
(`docs/fixtures/ultrabert_agnews_embeddings.safetensors`, sha256
`8fca923d…`) contains 240 train / 120 test AG News rows, 60/30 per class.

`L` is selectable.  `nll` is the ordinary categorical likelihood.  `brier` is
the RLCD proper score `Σ_c (p_c−y_c)²` on one-hot labels, used as a
**generalized** Bayes target; it is strictly proper and minimized at the
calibrated probability.  The default trains the anchor with NLL and samples
with Brier; `--posterior-loss nll` is the identical-head control.

Measured with the low-data arm (8 rows per class, `N_train=32`, `λ=1`,
`σ=0.5`, 4 × 3 600 steps, burn 1 200, thin 4, full head):

| basis | rank | accept z / w | BMA acc | BMA NLL | BMA Brier | BMA ECE | mean draw acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| informed | 8 | 98.3 / 95.3 % | 73.33 % | 0.7842 | 0.3995 | **0.0784** | 67.60 % |
| random | 8 | 99.8 / 95.1 % | 73.33 % | 0.7869 | 0.3995 | **0.0789** | 67.61 % |
| informed | 16 | 97.6 / 95.5 % | 73.33 % | 0.7821 | 0.3998 | 0.0957 | 67.64 % |
| random | 16 | 99.8 / 95.4 % | 73.33 % | 0.7825 | 0.3997 | 0.0956 | 67.62 % |
| informed | 32 | 97.7 / 95.5 % | 73.33 % | 0.7845 | 0.4010 | 0.0990 | 67.66 % |
| random | 32 | 99.7 / 95.6 % | 73.33 % | 0.7805 | 0.4003 | 0.0973 | 67.62 % |

The MAP MLP is 72.50 % / 1.9535 / 0.5113 with ECE 0.2462 — a severe
overconfidence failure on 32 training rows.  Posterior averaging improves the
paired NLL by `−1.17 ± 0.23` and Brier by `−0.11 ± 0.032` over the 120 test
rows, and cuts top-label ECE to 0.078–0.099.  The mean individual draw is
67.6 %, so BMA recovers the point model's accuracy from a weaker ensemble.
A four-subset recheck at rank 8 (same sampler seed, `--subset-seed`
20261118–21) shows the calibration effect is not one draw: MAP top-label ECE
ranged **0.139–0.313**, BMA ECE **0.052–0.130**, MAP NLL 1.00–2.74 and BMA NLL
0.50–0.97; accuracy moved from 65.0–83.3 % to 65.8–84.2 %.  This is
training-subset variation on one fixed test set, not independent test splits.

The only head is the MLP; no linear head is exposed.  The NLL-target
control gives ECE 0.082–0.102 and Brier 0.4004–0.4020: this
sample does not separate the Brier target from NLL, so the calibration gain is
attributed to posterior averaging, not proven to the loss.  Informed vs random
bases differ by <0.001 because the full target is the same weight space.

**Full-data MLP.**  At 240 training rows the MAP is 83.33 % / NLL 1.3230 /
Brier 0.3158 / ECE 0.1624.  BMA rank 8–32 is 83.33–85.00 %, NLL
0.5338–0.5479, Brier 0.2299–0.2318 and ECE **0.0515–0.0746**, with paired
`dNLL −0.775…−0.789 ± 0.202…0.206` and `dBrier −0.084…−0.086 ± 0.029`.
Posterior averaging still calibrates at full data, but the accuracy gain over
the MAP is at most one test row.

**Held-out noisy calibration split.**  `--cal-per-class 4 --label-noise 0.2`
held out 16 rows (4 per class), kept 224 for the NLL fit, and flipped 3/16
calibration labels.  Brier on those held-out rows is added to the posterior
target; test metrics stay on clean labels.  BMA rank 8–32: accuracy
85.83–87.50 %, NLL 0.4289–0.4459, Brier 0.2155–0.2196, ECE **0.0664–0.0846**.
This does not break clean-test calibration or accuracy, but its ECE is slightly
worse than the plain full-data target (0.052–0.075); the fit set and seed also
differ, so it is not an isolated noise effect.

**Width/depth.**  Full data, rank 8: baseline `32,16` gives BMA ECE 0.0629 and
accuracy 85.00 %; wider `64,32` (P=51 428) gives ECE 0.1108–0.1177 and accuracy
84.17–85.00 %; deeper `32,16,8` (P=25 308) gives ECE 0.1029–0.1053 and accuracy
78.33–79.17 %.  Both alternatives degrade accuracy and calibration at this data
size, so `32,16` is retained.

Acceptance is high with `β→1` and `ESS(Φ)` near its kept-draw ceiling, so these
chains are iid-like in many flat directions rather than narrowly converged.
The encoder is frozen, the head is one MLP, and one AG News split was used; no
benchmark, encoder-posterior or decision claim follows.  ECE is a top-label
diagnostic, not a full calibration certificate.  The calibration and width/depth
cells are single-split, 120-test-row comparisons.  Regenerate the fixture with
`scripts/pcnl-ultrabert-embed`; run
`cargo run --release -p pcnl --example ultrabert_head_pcnl`.

## R11 — continual Bayesian MLP head (2026-11-19)

The head posterior is carried across class-disjoint rounds with the recursion
`π_t ∝ π_{t-1} · exp(−λN_t mean Brier_t)`.  `DiagonalPosterior` fits the
previous particles to `N(μ, diag(σ²))`; the exact full-target kernel then
samples the new posterior.  The diagonal summary holds one random basis across
the stream.  Class-balanced replay rows per seen class are included because
strict no-replay carried priors were insufficient (90.56 pp mean forgetting).

At the end of `[0,1] → [2,3] → [0,1] → [2,3]`, 4 chains × 2 400 steps,
rank 8, clean 120-row test, 30 replay rows per class with class-balanced
likelihood weights (the delivered default, canonical seed 20261118):

| arm | acc | NLL | Brier | ECE | mean forgetting |
|---|---:|---:|---:|---:|---:|
| reset prior + replay | 80.83 % | 0.7671 | 0.3988 | 0.2992 | 5.00 pp |
| carried prior + replay | 80.83 % | **0.6372** | **0.3290** | **0.1717** | 5.56 pp |

Seed sweep, mean (sample standard deviation):

| replay / class | arm | seeds | acc | NLL | Brier | ECE | mean forgetting |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | reset | 5 | 80.50 (2.98) | 0.7740 (0.0215) | 0.4005 (0.0114) | 0.2934 (0.0353) | 5.89 (3.65) |
| 30 | carried | 5 | 83.67 (3.31) | **0.6131 (0.0157)** | **0.3155 (0.0099)** | **0.2024 (0.0442)** | 3.11 (1.78) |
| 60 | reset | 3 | 82.50 (2.21) | 0.7014 (0.0180) | 0.3596 (0.0128) | 0.2619 (0.0294) | 2.96 (1.39) |
| 60 | carried | 3 | 82.50 (2.21) | **0.5467 (0.0199)** | **0.2866 (0.0174)** | **0.1222 (0.0138)** | 3.33 (1.67) |

Class balancing is the retention win: the carried prior alone at 30 unbalanced
rows still forgot 30.56 pp, versus 5.56 pp balanced.  The robust carried-prior
gain is calibration — NLL, Brier and ECE are lower in every seed at 30 and
across three seeds at 60.  Accuracy is not resolved by the sweep: the 30-row
mean advantage is +3.17 pp with standard deviations of 3.31 and 2.98, and at
60 the means are identical.  Forgetting also overlaps (30: 3.11 vs 5.89 pp;
60: 3.33 vs 2.96 pp), so there is no resolved retention advantage from
carrying the prior once replay and balancing are on.  The 30-row default is
the fixed-memory operating point; 60 is the calibration arm.

Five alternate stratified splits (`--resplit-test-per-class 30 --split-seed
1..5`, model seed fixed, 30 rows/class) reproduce the calibration effect:

| arm | splits | acc | NLL | Brier | ECE | mean forgetting |
|---|---:|---:|---:|---:|---:|---:|
| reset prior | 5 | 77.50 (5.84) | 0.8223 (0.0462) | 0.4299 (0.0289) | 0.2693 (0.0641) | 3.89 (2.12) |
| carried prior | 5 | 76.33 (6.52) | **0.6714 (0.1030)** | **0.3515 (0.0571)** | **0.1519 (0.0553)** | 3.22 (1.90) |

The resplits overlap, so these are split-sensitivity checks, not independent
samples.  The carried prior again has lower NLL, Brier and ECE in every split.
Accuracy is 1.17 pp lower on average (sd 6.52 vs 5.84 pp) and forgetting
0.67 pp lower with overlapping ranges, so neither is resolved.  Calibration is
the only carried-prior advantage that survives both seed and split resampling.

A single all-class posterior with the same total proposal budget (38,400
proposals: `--tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200
--thin 16`) is the reference for what sequential updating costs:

| posterior | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| joint, all classes at once | **82.50 (5.17)** | 0.7622 (0.0342) | 0.3902 (0.0229) | 0.3131 (0.0423) |
| sequential carried | 76.33 (6.52) | **0.6714 (0.1030)** | **0.3515 (0.0571)** | **0.1519 (0.0553)** |

The raw joint posterior is more accurate in every split (mean +6.17 pp, range
3.3–9.2 pp) but has worse probability forecasts.  That reverses with post-hoc
temperature scaling.  `--calibrate-joint 10` holds out 10 training rows per
class, fits the joint on 50 per class, fits one temperature on the held-out
training NLL, and applies it to the test probabilities; test labels are unused.

| joint variant | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| raw, 50 train/class | 83.00 (5.79) | 0.8334 (0.0376) | 0.4326 (0.0234) | 0.3547 (0.0552) |
| temperature-scaled, 50 train + 10 cal/class | **83.00 (5.79)** | **0.4860 (0.1615)** | **0.2582 (0.0798)** | **0.0983 (0.0387)** |
| sequential carried, 60 train/class + replay | 76.33 (6.52) | 0.6714 (0.1030) | 0.3515 (0.0571) | 0.1519 (0.0553) |

The fitted temperature was 0.17–0.30, so BMA is under-confident; accuracy is
unchanged by scaling, while NLL, Brier and ECE fall in every split.  In this
data-asymmetric table, the calibrated joint beats sequential carried on all
four metrics in every split.  The earlier claim that sequential produced the better forecast applies only to
the uncalibrated joint.  That comparison is not data-matched: the joint gives
up 40 training rows for calibration and the sequential arm gets no post-hoc
fit.  The symmetric protocol removes both asymmetries: `--heldout-calibration
10` withholds the same 10 rows/class before either arm trains, both fit on 50
rows/class, and both temperatures use the same held-out rows.  Over the five
resplits:

| arm | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| sequential carried, raw | 77.33 (5.99) | 0.6740 (0.0602) | 0.3547 (0.0397) | 0.1625 (0.0478) |
| sequential carried, temperature | 77.33 (5.99) | 0.6288 (0.0902) | 0.3288 (0.0533) | 0.1005 (0.0504) |
| joint, raw | **83.00 (5.79)** | 0.8334 (0.0376) | 0.4326 (0.0234) | 0.3547 (0.0552) |
| joint, temperature | **83.00 (5.79)** | **0.4860 (0.1615)** | **0.2582 (0.0798)** | **0.0983 (0.0387)** |

The calibrated joint is more accurate in all five splits (+5.67 (2.24) pp) and
has lower Brier in all five.  NLL is lower in four of five (split 1 is 0.026
higher).  ECE is lower in only two of five and is essentially tied on average
(0.098 versus 0.101), so the earlier all-four-metrics claim does not survive
the symmetric protocol; part of that ECE advantage came from leaving the
sequential posterior uncalibrated.  Temperature scaling is stable for the
joint here: NLL, Brier and ECE fall in every split.

A rank check at the canonical split (one seed) raised the carried prior from
rank 8 to 16, 32 and 64; final accuracy was 79.17, 80.83 and 75.00 % versus
80.83 % at rank 8, with calibration in the same band.  Rank is not the
limiting factor, so the default remains 8.

The λ/replay grid (canonical split, continual arm) is:

| λ | replay | acc | NLL | Brier | ECE | forgetting |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 15 | 77.50 | 1.0309 | 0.5535 | 0.3909 | 6.67 |
| 0.5 | 30 | 85.00 | 0.8941 | 0.4713 | 0.4081 | 1.67 |
| 0.5 | 45 | 81.67 | 0.7515 | 0.3862 | 0.2997 | 2.22 |
| 1.0 | 15 | 75.83 | 0.7199 | 0.3797 | 0.1908 | 10.00 |
| 1.0 | 30 | 80.83 | 0.6372 | 0.3290 | 0.1717 | 5.56 |
| 1.0 | 45 | 80.00 | 0.5551 | 0.2978 | 0.1238 | -1.11 |
| 2.0 | 15 | 79.17 | 0.6043 | 0.3134 | 0.0859 | 18.89 |
| 2.0 | 30 | 75.83 | 0.6137 | 0.3321 | 0.0700 | 7.78 |
| 2.0 | 45 | **86.67** | **0.5133** | **0.2591** | **0.1291** | 3.33 |

λ=2/replay=45 was validated over five seeds: 82.33 (2.60) %, NLL 0.5410
(0.0313), Brier 0.2775 (0.0166), ECE 0.0776 (0.0327), forgetting 4.67 (3.16)
pp, versus the default's 83.67 (3.31) %, 0.6131 (0.0157), 0.3155 (0.0099),
0.2024 (0.0442), 3.11 (1.78) pp.  Over five splits it is 78.17 (4.39) %,
0.5885 (0.0697), 0.3110 (0.0507), 0.0679 (0.0396), 3.55 (1.78) pp versus the
default's 76.33 (6.52) %, 0.6714 (0.1030), 0.3515 (0.0571), 0.1519 (0.0553),
3.22 (1.90) pp.  Calibration improves under both resamplings; accuracy and
forgetting do not resolve.  It uses 45 replay rows/class, so it is a
calibration-oriented alternative, not the new default.

The diagonal prior ignores correlations and replay approximates (and
double-counts) old data, so this is not exact Bayes over the full history.
A low-rank-plus-isotropic summary (`LowRankPosterior`, opt-in via `--prior
lowrank`) was measured but not adopted: at 30 balanced replay rows it matches
the diagonal forgetting (5.00 vs 5.56 pp) but costs 12.5 pp canonical-seed
accuracy (80.83 → 68.33 %).  At 60 it is 0.83 pp above the canonical-seed
diagonal with −5.00 pp forgetting, but both differences are single-seed and
inside the diagonal's seed spread; calibration is worse (NLL 0.6307, Brier
0.3097, ECE 0.1491 versus 0.5326, 0.2897, 0.1065).
Strict no-replay low-rank still forgot 91.11 pp, so the shared head under
disjoint likelihoods is the limiting factor, not the missing correlations alone.

The second-dataset check uses four 20 Newsgroups categories
(`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.guns`),
60 train + 30 test per class, with headers, footers and quoted text removed so
the category name cannot leak from the header.  Over five resplits:

| five resplits | acc | NLL | Brier | ECE | mean forgetting |
|---|---:|---:|---:|---:|---:|
| sequential carried | 87.00 (1.63) | 0.4440 (0.0267) | 0.2202 (0.0130) | 0.1444 (0.0106) | 1.00 (3.80) pp |
| joint, temperature-scaled | **88.17 (3.61)** | **0.3223 (0.0643)** | **0.1676 (0.0343)** | **0.0606 (0.0174)** | n/a |

In that data-asymmetric table the calibrated joint has lower NLL, Brier and
ECE in all five splits (`T = 0.187–0.354`), so the calibration advantage
reproduces on a second corpus.  Accuracy is higher in three splits, equal in one, and 4.17 pp lower in
one, averaging +1.17 (3.42) pp; in that data-asymmetric table the AG News
accuracy advantage does not uniformly generalize.

Five model seeds on the canonical 20 Newsgroups split, at the same
calibrated-joint budget:

| seeds | acc | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|
| 5 | **89.17 (1.32)** | 0.2839 (0.0296) | 0.1478 (0.0131) | 0.0500 (0.0100) |

Accuracy ranged from 87.50 % to 90.83 % (sd 1.32 pp), with `T = 0.303–0.371`.
This is model/proposal variance at a fixed split, not split variance.

A symmetric held-out comparison removes the earlier data asymmetry:
`--heldout-calibration 10` withholds 10 training rows per class before either
arm trains, so both fit on 50 rows/class and both temperatures use the same
held-out rows.  Over five resplits:

| arm | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| sequential carried, raw | 86.50 (3.84) | 0.4721 (0.0593) | 0.2295 (0.0358) | 0.1553 (0.0382) |
| sequential carried, temperature | 86.50 (3.84) | 0.4683 (0.1572) | 0.2104 (0.0513) | **0.0849 (0.0282)** |
| joint, raw | 88.83 (2.61) | 0.6371 (0.0318) | 0.3048 (0.0208) | 0.3271 (0.0300) |
| joint, temperature | **88.83 (2.61)** | **0.3064 (0.0911)** | **0.1526 (0.0365)** | **0.0591 (0.0201)** |

The calibrated joint is more accurate in all five splits (+2.33 (2.07) pp) and
has lower NLL and Brier in all five.  The earlier mixed 20 Newsgroups accuracy
(three wins, one tie, one loss) came from the data asymmetry: with both arms on
50 rows/class the joint is ahead in every split.  ECE is lower in four and
0.004 higher in the fifth.  For the sequential arm, temperature scaling lowers ECE in every
split but worsens NLL in two of five; its mean NLL is unchanged and its spread
more than doubles.  The fitted temperature is therefore not a free improvement
for every posterior at this calibration-set size, but the joint's NLL and
Brier advantage survives the symmetric protocol.

Two fixtures; the diagonal summary has five seeds at 30 rows, three at 60, and five overlapping resplits on AG News, plus five resplits and a five-seed canonical sweep on 20 Newsgroups.  Seed and split variance are characterized separately, not over a full seed × split grid; the joint tables use one model seed per split.  The low-rank arm and unbalanced controls are single-seed.  The 20 Newsgroups task is a four-class subset, not the full 20-class problem, and both fixtures are controlled probes rather than benchmarks.  No benchmark or decision claim follows.

## R12 — signed rotation head (delivered probe, 2026-11-20)

`pcnl::rotation_head` implements the signed-state half of R12.  A
`768 → … → K(K−1)/2` tanh trunk emits the angles of a fixed ordered product of
real plane rotations; `ψ = U_θ(e)b₀` is therefore normalized for every input,
and the state exposes signed amplitudes rather than softmax probabilities.
`prepare`, `apply`, `tensor`, `projective measure`, basis/draw validation and
the Bayesian predictive mixture are present.  Evidence is an append-only
prefix with stable IDs: an identical retry is a no-op, conflicting reuse is an
error, and `BornPotential` evaluates the unweighted sum of `−2 log |ψ_y|`
through the existing `Potential` seam.  `mh_pcn_full_step` implements the exact
full-space kernel with the plan's drift and covariance, including the reverse
proposal density.

`RotationSession` persists the dimension, basis, layout, model version,
evidence prefix, target settings, chain state and serializable RNG state in a
single JSON checkpoint written via temp-file + sync + rename.  `save`/`load`
reopens bit-identically, stale `.tmp` files are ignored.  Warm-start sets an
explicit chain start; `prior_theta` supplies a deterministic working-scale
draw.  `predict_sequence` composes the whole sequence per posterior draw and then averages squared
amplitudes, so it is not a product of per-step marginals.  The thin
`src/bin/rotation.rs` CLI exposes init/update/predict/show.

Focused checks (20 tests): cross-chain retention/provenance, the two-state composition oracle where equal
measurements separate after `R(π/4)`, the projective-measurement control,
overall-sign invariance, identity and product order, tensor indexing and
marginals, empty-text/basis/draw rejection, evidence idempotence and conflict
detection, finite-difference Born gradients, exact-zero loss divergence,
checkpoint reopen, and the isotropic, normalized and centered anisotropic
Gaussian canaries, plus the anchor-RMS scale estimate.  `pcnl/tests/rotation_cli.rs`
covers init/update/predict/show and the restart.  Run:

```bash
cargo test --offline -p pcnl --lib rotation_head
cargo test --offline -p pcnl --test rotation_cli
cargo test --offline -p pcnl --lib
```

The full library suite passes **233/233 in 286.5 s** (after the cross-chain
retention and independent-start diagnostics).  The two `dt_subspace`
basis checks that previously exceeded ten minutes were exact-gradient path
sweeps in an unoptimized build; their path lengths are now bounded (3 steps for
the PCA/head/trunk checks, 8 for the collinear RTG block), preserving the
orthonormality and block-support assertions without the multi-minute cost.

**Normalized prior.**  The raw-weight `N(0,0.5²I)` prior is replaced by a
per-layer normalized prior.  A weight matrix with fan-in `n` is divided by
`√n` before the Gaussian penalty, so its prior variance is `σ²n` in raw
coordinates; biases are unscaled.  The pCN proposal covariance is diagonal in
the same coordinates, keeping the MH kernel exact.  Chain initialisation uses
the working scale `σ`, not the much wider raw prior draw, because a 768-wide
input layer would otherwise start saturated.

**Measured probe.**  The probe still uses the smallest tanh head (`hidden=[1]`),
20 fit/class, 10 calibration/class, Adam pretraining, beta=0.01, 250 steps,
50 burn, 2 chains and 10 retained draws.  Five seeds (20261118–20261122) give
(mean ± sd):

| fixture | arm | acc % | NLL | Brier | ECE |
|---|---|---|---|---|---|
| AG News | raw Born | 46.3 ± 13.2 | 1.812 ± 0.489 | 0.788 ± 0.189 | 0.269 ± 0.103 |
| AG News | temperature | 46.3 ± 13.2 | 1.221 ± 0.166 | 0.658 ± 0.094 | 0.070 ± 0.046 |
| AG News | softmax control | 48.2 ± 3.2 | 2.291 ± 0.540 | 0.732 ± 0.094 | 0.238 ± 0.109 |
| 20NG | raw Born | 44.2 ± 16.6 | 1.853 ± 0.832 | 0.796 ± 0.295 | 0.316 ± 0.164 |
| 20NG | temperature | 44.2 ± 16.6 | 1.177 ± 0.254 | 0.627 ± 0.139 | 0.109 ± 0.032 |
| 20NG | softmax control | 55.7 ± 5.8 | 1.578 ± 0.317 | 0.593 ± 0.045 | 0.149 ± 0.031 |

Temperature fitting uses a grid over [0.04, 8]; it lowers NLL and ECE but is
monotone and cannot change accuracy.  Against the superseded raw-weight prior,
AG NLL improves from 1.995 ± 0.509 to 1.812 ± 0.489 with accuracy flat
(45.8 → 46.3 %); 20NG accuracy falls (47.8 → 44.2 %) with NLL flat
(1.828 → 1.853).  The normalized prior does not rescue the single-step arm.

**Empirical-Bayes anchor prior.**  The prior center is the Adam-pretrained head
and its scalar scale is estimated from the anchor's RMS in normalized head
coordinates, `sigma = rms(theta / prior_scales)` (0.125–0.244 in these runs).
The centered pCN kernel is exact; `centered_pcn_recovers_a_shifted_gaussian_posterior`
is the canary.  Five seeds (20261118–20261122) give:

| fixture | arm | acc % | NLL | Brier | ECE |
|---|---|---|---|---|---|
| AG News | anchor raw | 42.3 ± 10.3 | 1.796 ± 0.389 | 0.806 ± 0.156 | 0.297 ± 0.079 |
| AG News | anchor temperature | 42.3 ± 10.3 | 1.231 ± 0.148 | 0.661 ± 0.086 | 0.057 ± 0.030 |
| 20NG | anchor raw | 43.0 ± 15.9 | 1.726 ± 0.531 | 0.794 ± 0.232 | 0.302 ± 0.098 |
| 20NG | anchor temperature | 43.0 ± 15.9 | 1.195 ± 0.246 | 0.637 ± 0.134 | 0.113 ± 0.026 |

NLL improves modestly against the zero-centered normalized prior (AG
1.812 → 1.796; 20NG 1.853 → 1.726), but accuracy falls (AG 46.3 → 42.3 %;
20NG 44.2 → 43.0 %).  Proposal acceptance is high (336–496 of 500), so the
failure is not a chain stuck at its start.  The softmax control still wins
accuracy on both fixtures and NLL on 20NG (1.578 vs 1.726); on AG the anchor
arm has lower NLL (1.796 vs 2.291) but lower accuracy (42.3 vs 48.2 %).

On the length-2 sequence arm (60 train / 24 test sequences, 800 Adam steps,
200 sampler steps, seeds 20261118–20261120), the oracle scores 61.1 ± 6.4 % on
AG and 70.8 ± 8.3 % on 20NG.  Learned composition scores 23.6 ± 8.7 % and
38.9 ± 12.7 %; the single-step ablation scores 30.6 ± 4.8 % and 30.6 ± 9.6 %;
the classical squared-transition baseline scores 30.6 ± 6.4 % and
33.3 ± 16.7 %.  The AG arm is worse than both controls.  The 20NG arm is above
its ablation but the paired differences (+25.0, 0.0, 0.0 points) and the
classical baseline variance do not establish learned interference.  Under the
anchor prior the learned arm falls to 19.4 ± 10.5 % on AG and 30.6 ± 8.7 % on
20NG, while the ablation scores 25.0 ± 15.0 % and 43.1 ± 31.3 % and the
classical baseline 27.8 ± 19.2 % and 44.4 ± 33.7 %.  No text-accuracy or
learned-interference claim follows.  The normalized prior removes layer-width
domination but does not fix the arm; the empirical-Bayes anchor prior lowers
NLL without improving accuracy, and the sequence result gets worse.

**Held-out width/representation sweep (2026-11-20).**  Four widths
(`hidden=[1]`, `[4]`, `[8]`, `[16]`) and two input representations (`raw` and
fit-standardized embeddings), five paired seeds, same fit/test split as the
probe: 20 fit rows/class, 20 validation rows/class excluded from the
likelihood, and the 30-row/class test split.  Standardization uses fit-row means and standard
deviations only.  Width is selected by mean validation NLL across seeds; test
is not used for selection.  The softmax control is trained by NLL at the same
widths.

| fixture | representation | arm | width | test acc % | test NLL | test Brier | test ECE | validation NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AG News | raw | signed rotation | 8 | 29.83 ± 5.38 | 1.6467 ± 0.0998 | 0.8409 ± 0.0441 | 0.2298 ± 0.0440 | 1.6598 ± 0.1296 |
| AG News | raw | softmax control | 16 | 84.67 ± 1.27 | 0.7317 ± 0.1540 | 0.2730 ± 0.0274 | 0.1293 ± 0.0247 | 0.6844 ± 0.1880 |
| AG News | standardized | signed rotation | 1 | 30.67 ± 8.44 | 1.4774 ± 0.0804 | 0.7682 ± 0.0449 | 0.0642 ± 0.0483 | 1.4290 ± 0.0728 |
| AG News | standardized | softmax control | 16 | 81.67 ± 1.56 | 1.0528 ± 0.2494 | 0.3240 ± 0.0276 | 0.1460 ± 0.0111 | 1.1044 ± 0.1276 |
| 20NG | raw | signed rotation | 4 | 38.00 ± 9.44 | 1.5141 ± 0.0997 | 0.7713 ± 0.0488 | 0.1799 ± 0.0619 | 1.5577 ± 0.1430 |
| 20NG | raw | softmax control | 16 | 89.83 ± 1.60 | 0.4284 ± 0.0632 | 0.1648 ± 0.0242 | 0.0691 ± 0.0138 | 0.4766 ± 0.1161 |
| 20NG | standardized | signed rotation | 1 | 34.00 ± 6.75 | 1.4317 ± 0.0446 | 0.7435 ± 0.0307 | 0.0602 ± 0.0330 | 1.4128 ± 0.0692 |
| 20NG | standardized | softmax control | 8 | 87.83 ± 2.92 | 0.6504 ± 0.0824 | 0.2025 ± 0.0378 | 0.0866 ± 0.0184 | 0.6131 ± 0.1499 |

Raw rotation accepted 381/500 proposals on AG at width 8 and 382/500 on 20NG at
width 4.  Standardized rotation accepted only 9/500 and 4/500 at `beta=0.01`,
so those rows are a near-frozen chain at the Adam start, not a sampled
posterior, and are ineligible for selection at that step size.  Width selection
is unstable: per-seed validation minima chose `{4:2, 8:2, 1:1}` for AG raw
rotation and `{4:2, 1:2, 8:1}` for 20NG.  Selecting per seed gives
39.0 ± 13.7 % / NLL 1.547 ± 0.270 on AG and 45.0 ± 13.6 % / 1.363 ± 0.196 on
20NG; softmax gives 84.8 ± 2.2 % / 0.734 and 90.0 ± 1.8 % / 0.426.
Standardization lowers softmax NLL and Brier on both fixtures (paired `dNLL`
−0.322 ± 0.145 and −0.263 ± 0.115; `dBrier` −0.065 ± 0.034 and −0.045 ± 0.048),
but does not rescue the rotation arm.  No width or representation cell tested
makes the signed head competitive on these probes.  This is a negative result
for this angle parametrization at this sampler budget, not for signed states in
general; the next test needs a representation that actually mixes at the same
settings, not more widths.

**Step-size, centering and tuned-baseline follow-up (2026-11-20).**  The
follow-up tests fit-centering without variance scaling, a
representation-specific pCN step, and an approximately parameter-matched
softmax baseline tuned on validation.  All runs keep the same split, five
paired seeds and the same sampler budget.

*Centering.*  The `centered` arm subtracts the fit-row mean but does not divide
by the standard deviation.  Its selected widths are the same as raw (AG width
8, 20NG width 4).  It mixes at the selected widths, but it does not improve the
proper scores:

| fixture | representation | width | accepted/500 | validation NLL | test NLL | test acc % |
|---|---|---:|---:|---:|---:|---:|
| AG News | raw | 8 | 381.2 ± 35.1 | 1.6598 ± 0.1296 | 1.6467 ± 0.0998 | 29.83 ± 5.38 |
| AG News | centered | 8 | 394.6 ± 39.9 | 1.6596 ± 0.1233 | 1.6988 ± 0.1118 | 26.33 ± 4.84 |
| 20NG | raw | 4 | 382.4 ± 29.5 | 1.5577 ± 0.1430 | 1.5141 ± 0.0997 | 38.00 ± 9.44 |
| 20NG | centered | 4 | 394.2 ± 12.3 | 1.5991 ± 0.1525 | 1.4999 ± 0.1287 | 36.50 ± 5.60 |

The AG validation NLL is indistinguishable from raw; 20NG is worse.  Centering
changes the acceptance rate, not the ranking.

*Representation-specific step size.*  At width 1, `beta=0.01` was too large for
both representations, and especially for standardized inputs:

| fixture | representation | beta | accepted/500 | validation NLL |
|---|---|---:|---:|---:|
| AG News | raw | 0.005 | 365.0 ± 76.8 | 1.7646 ± 0.2979 |
| AG News | raw | 0.010 | 72.6 ± 91.5 | 1.9373 ± 0.4057 |
| AG News | standardized | 0.0025 | 262.0 ± 64.3 | 1.4299 ± 0.0730 |
| AG News | standardized | 0.005 | 72.6 ± 61.2 | 1.4486 ± 0.0550 |
| AG News | standardized | 0.010 | 9.0 ± 13.6 | 1.4290 ± 0.0728 |
| 20NG | raw | 0.005 | 367.6 ± 69.5 | 1.6900 ± 0.3943 |
| 20NG | raw | 0.010 | 82.4 ± 103.4 | 1.7746 ± 0.7063 |
| 20NG | standardized | 0.0025 | 248.4 ± 48.2 | 1.3590 ± 0.0464 |
| 20NG | standardized | 0.005 | 45.0 ± 24.5 | 1.3860 ± 0.0612 |
| 20NG | standardized | 0.010 | 4.0 ± 8.9 | 1.4128 ± 0.0692 |

So the earlier “near-frozen” label for standardized width 1 describes the
`beta=0.01` cell, not the representation.  At `beta=0.0025` it is a sampled
posterior (about 248–262/500 accepted) and its NLL improves.  Lowering beta
also partly unfreezes standardized widths 4 and 8 (about 57–60 and 31–34
accepted), but their validation NLL remains 2.05–2.44, far worse than the
width-1 value of 1.36–1.43.  The width failure is therefore not only a step-size
failure; the wider standardized parameterisation still fits worse.

The best standardized rotation cell is width 1 at `beta=0.0025`: AG test
28.50 ± 6.60 % / NLL 1.4275 ± 0.0339, 20NG 33.66 ± 8.01 % / 1.3445 ± 0.0700.
That is a better NLL than the best raw selected-width cell (AG width 8 at
`beta=0.01`, 29.83 ± 5.38 % / 1.6467 ± 0.0998; 20NG width 4 at `beta=0.005`,
38.33 ± 4.57 % / 1.5270 ± 0.1396), but lower accuracy in both cases.  It is a
calibration improvement, not an accuracy rescue.

*Tuned matched-width softmax.*  `--softmax-tune` selects L2 weight decay from
`{0, 1e-4, 1e-3}` and early-stops on calibration NLL, evaluating every 100
steps with patience 4.  The grid selected a nonzero penalty in 110/120 cells
(67 at `1e-3`, 43 at `1e-4`) and early-stopped between 100 and 3,000 steps
(83/120 at or below 600).  The output prints both flat parameter counts:
781/777 at width 1, 3,106/3,096 at width 4, 6,206/6,188 at width 8, and
12,406/12,372 at width 16, i.e. within 0.5 % of each other rather than exactly
equal.  At the selected widths:

| fixture | representation | width | rotation acc % | rotation NLL | tuned softmax acc % | tuned softmax NLL | params rot/soft |
|---|---|---:|---:|---:|---:|---:|---:|
| AG News | raw | 8 | 29.83 ± 5.38 | 1.6467 ± 0.0998 | 85.00 ± 1.67 | 0.4691 ± 0.0707 | 6,206/6,188 |
| AG News | centered | 8 | 26.33 ± 4.84 | 1.6988 ± 0.1118 | 84.67 ± 2.09 | 0.4640 ± 0.0727 | 6,206/6,188 |
| AG News | standardized | 1 | 28.50 ± 6.60 | 1.4275 ± 0.0339 | 46.50 ± 5.76 | 1.0163 ± 0.1017 | 781/777 |
| 20NG | raw | 4 | 38.33 ± 4.57 | 1.5270 ± 0.1396 | 89.83 ± 1.81 | 0.2810 ± 0.0166 | 3,106/3,096 |
| 20NG | centered | 4 | 36.50 ± 5.60 | 1.4999 ± 0.1287 | 90.00 ± 2.04 | 0.2792 ± 0.0169 | 3,106/3,096 |
| 20NG | standardized | 1 | 33.66 ± 8.01 | 1.3445 ± 0.0700 | 48.67 ± 5.61 | 0.8940 ± 0.0693 | 781/777 |

Rotation rows use the best tested beta for that representation/width (raw at
0.01 for AG width 8 and 0.005 for 20NG width 4; centered at 0.01; standardized
width 1 at 0.0025); softmax is trained on the fit rows and tuned only on
calibration rows.  The tuned baseline still wins every matched-width comparison.
At width 16 it also improves on the prior fixed-3,000-step softmax control:
AG raw NLL 0.4655 vs 0.7317, 20NG raw 0.2730 vs 0.4284.  The correction to the
earlier table is that standardized width 1 has a usable posterior and a better
NLL than raw rotation at a smaller step; it does not make the signed arm
competitive with the tuned softmax baseline.

**Prior-centre and rotation-pretrain follow-up (2026-11-20).**  Two
asymmetries remained after the tuned-baseline comparison.  The zero-angle prior
is the identity, so its Gaussian mass sits on a one-hot outcome-0 state rather
than a class-balanced one, and the rotation head used a fixed 1,000-step Adam
pretrain while the tuned softmax early-stopped on calibration rows.
`--uniform-prior` centres the angle prior on the class-balanced state
`(1/√K, …, 1/√K)` (all weights zero and the first `K−1` canonical angles set;
the example asserts the prepared state), and `--rotation-tune` early-stops the
rotation pretrain on calibration Born NLL, evaluating every 100 steps with
patience 4.  Training uses fit rows only and selection uses calibration rows
only; test rows are never read.  The grid is the same four selected cells (AG
raw width 8 at `beta=0.01`, AG standardized width 1 at `0.0025`, 20NG raw width
4 at `0.005`, 20NG standardized width 1 at `0.0025`), the four
fixed/tuned × zero/uniform arms, and the five paired seeds, with the softmax
control disabled (`--softmax-pretrain 0`).  All 80 cells completed.

| cell | base acc % / NLL | early-stop | uniform prior | uniform + early | early step |
|---|---:|---:|---:|---:|---:|
| AG raw w8 | 29.83 / 1.647 | 28.00 / 1.715 (+0.07) | 30.83 / 1.712 (+0.07) | 29.00 / 1.670 (+0.02) | 280 ± 349 |
| AG std w1 | 28.50 / 1.427 | 28.83 / 1.410 (−0.01) | 30.50 / 1.500 (+0.07) | 29.83 / 1.430 (0.00) | 340 ± 195 |
| 20NG raw w4 | 38.33 / 1.527 | 39.50 / 1.460 (−0.07) | 37.33 / 1.540 (+0.02) | 39.33 / 1.530 (+0.01) | 120 ± 45 |
| 20NG std w1 | 33.66 / 1.344 | 33.00 / 1.360 (+0.01) | 32.66 / 1.380 (+0.04) | 32.00 / 1.400 (+0.05) | 520 ± 444 |

Parentheses are the paired mean `ΔNLL` against the base cell over the five
seeds; a negative value is better.  The uniform prior is worse on mean NLL in
all four cells (+0.02 to +0.07) and moves accuracy by −1.00 to +2.00 points;
its class balance does not fix the arm.  Early stopping is the only arm with a
negative mean `ΔNLL` on two cells, but both are within seed scatter (AG std
−0.01 ± 0.06, 3/5 seeds; 20NG raw −0.07 ± 0.12, 3/5), and it is worse on the
other two (AG raw +0.07 ± 0.19; 20NG std +0.01 ± 0.02, 0/5 seeds).  The
selected step is unstable (120 ± 45 to 520 ± 444; validation NLL often prefers
the full 1,000 steps), so the fixed pretrain is not the asymmetry that explains
the gap.  Acceptance stays 212–478/500, so no arm is frozen.  The tuned
softmax still wins every selected cell; neither intervention changes the
negative conclusion.  The draws in this grid were retained chain-by-chain and
therefore came from chain 0 only; the cross-chain rerun below supersedes the
absolute numbers (the ranking is unchanged).

**Cross-chain retention and mixing audit (2026-11-20).**  The earlier R12 grids
asked for 10 retained draws from 2 chains, but `update` appended each chain's
thinned trace and then truncated to `draws`, so every published MCMC number was
ten draws from chain 0.  `update` now allocates the total draw quota across
chains, spreads each chain's quota over its post-burn trace, and records
`(chain, step)` provenance; pre-update prior draws report an unavailable
provenance.  A regression test covers the 2-chain/6-draw allocation, and the
checkpoint round-trip preserves the provenance.  The four selected cells were
rerun for all four prior/pretrain arms and five paired seeds; all 80 cells
completed.

| cell | arm | acc % | NLL | Brier | draw-NLL mean | split R̂ |
|---|---|---:|---:|---:|---:|---:|
| AG raw w8 | base | 27.67 ± 3.20 | 1.4281 ± 0.0837 | 0.7710 | 2.4597 | 1.202 |
| AG raw w8 | early-stop | 31.17 ± 4.59 | 1.4038 ± 0.0606 | 0.7571 | 2.4407 | 1.381 |
| AG raw w8 | uniform | 29.00 ± 4.58 | 1.4934 ± 0.0672 | 0.7863 | 2.5027 | 0.935 |
| AG raw w8 | uniform + early | 26.00 ± 3.08 | 1.4537 ± 0.0427 | 0.7814 | 2.4681 | 1.505 |
| AG std w1 | base | 28.84 ± 7.25 | 1.3797 ± 0.0741 | 0.7484 | 1.5119 | 1.123 |
| AG std w1 | early-stop | 30.17 ± 6.86 | 1.3876 ± 0.0723 | 0.7481 | 1.4624 | 1.776 |
| AG std w1 | uniform | 29.83 ± 4.62 | 1.4371 ± 0.1207 | 0.7523 | 1.6043 | 2.769 |
| AG std w1 | uniform + early | 27.00 ± 4.88 | 1.4079 ± 0.0818 | 0.7573 | 1.4766 | 2.752 |
| 20NG raw w4 | base | 37.83 ± 4.81 | 1.3109 ± 0.1081 | 0.7173 | 2.1681 | 1.981 |
| 20NG raw w4 | early-stop | 37.00 ± 6.78 | 1.2876 ± 0.1049 | 0.7048 | 2.0854 | 1.223 |
| 20NG raw w4 | uniform | 41.00 ± 3.79 | 1.2922 ± 0.0686 | 0.6945 | 2.1143 | 1.322 |
| 20NG raw w4 | uniform + early | 40.17 ± 4.18 | 1.2793 ± 0.0676 | 0.6865 | 2.0833 | 1.340 |
| 20NG std w1 | base | 33.33 ± 7.97 | 1.3324 ± 0.0782 | 0.7230 | 1.4268 | 1.401 |
| 20NG std w1 | early-stop | 33.17 ± 8.59 | 1.3437 ± 0.0740 | 0.7302 | 1.4258 | 1.628 |
| 20NG std w1 | uniform | 33.17 ± 3.97 | 1.3693 ± 0.0561 | 0.7396 | 1.4447 | 1.316 |
| 20NG std w1 | uniform + early | 31.83 ± 3.61 | 1.3771 ± 0.0292 | 0.7436 | 1.4342 | 1.307 |

`draw-NLL mean` averages the test NLL of each individual draw before the
Bayesian mixture; it is not the predictive NLL.  `split R̂` is computed from
the per-chain sequences of that draw-level NLL (five draws per chain), so it is
a coarse chain-disagreement diagnostic, not a parameter-space convergence
certificate.  Paired mean `ΔNLL` against the base cell (negative is better):

| cell | early-stop | uniform | uniform + early |
|---|---:|---:|---:|
| AG raw w8 | −0.0243 ± 0.0814 (2/5) | +0.0653 ± 0.1201 (1/5) | +0.0256 ± 0.1096 (1/5) |
| AG std w1 | +0.0079 ± 0.0270 (3/5) | +0.0574 ± 0.1042 (2/5) | +0.0282 ± 0.0661 (1/5) |
| 20NG raw w4 | −0.0233 ± 0.0681 (2/5) | −0.0186 ± 0.1621 (3/5) | −0.0315 ± 0.1647 (3/5) |
| 20NG std w1 | +0.0113 ± 0.0116 (0/5) | +0.0369 ± 0.0716 (2/5) | +0.0447 ± 0.0568 (1/5) |

Cross-chain retention materially changes the absolute scores.  The base cells
move from AG raw 1.6467 to 1.4281, AG standardized 1.4275 to 1.3797, 20NG raw
1.5270 to 1.3109, and 20NG standardized 1.3445 to 1.3324.  The old chain-0-only
readout was pessimistic about NLL, not optimistic; accuracy is similar or
slightly lower.  The arm conclusions do not change: early stopping improves the
mean on AG raw and 20NG raw but only in 2/5 seeds, is flat on AG standardized,
and is worse in every seed on 20NG standardized.  The uniform prior is worse on
three of four cells, with a non-replicated 20NG-raw gain inside seed scatter.
No arm is promoted.

The more important result is that the corrected ensembles still do not mix:
mean split R̂ exceeds 1.05 in fifteen of sixteen rows and reaches 2.77, with
single-seed values of 6.94 and 7.72.  Ten retained draws (five per chain), 250
steps per chain and a shared Adam warm start are too few for differences of
this size.  The proper scores above are therefore descriptive, not converged
posterior estimates; the tuned softmax comparison is unaffected and still wins
every selected cell.

**Budget escalation and parameter-space mixing (2026-11-20).**  The R̂ warning
was tested rather than assumed.  The four base cells were rerun at 1,000 steps,
burn 200, two chains and 40 retained draws (20 per chain), on the same five
seeds.  All 20 cells completed.

| cell | acc % | NLL | Brier | draw-NLL | predictive R̂ | param R̂ median | param >1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AG raw w8 | 31.17 ± 6.68 | 1.3586 ± 0.0571 | 0.7401 | 2.4460 | 1.078 ± 0.076 | 2.005 ± 0.018 | 98.0% |
| AG std w1 | 30.16 ± 8.36 | 1.3600 ± 0.0805 | 0.7380 | 1.5067 | 1.365 ± 0.490 | 2.110 ± 0.032 | 98.1% |
| 20NG raw w4 | 38.67 ± 6.14 | 1.2994 ± 0.1020 | 0.7036 | 2.2325 | 1.206 ± 0.218 | 2.047 ± 0.016 | 98.3% |
| 20NG std w1 | 36.50 ± 7.78 | 1.3091 ± 0.0935 | 0.7132 | 1.4098 | 1.364 ± 0.353 | 2.047 ± 0.057 | 98.2% |

Paired `ΔNLL` against the 250-step base (negative is better): AG raw
−0.0695 ± 0.0595, AG standardized −0.0197 ± 0.0198, 20NG raw −0.0115 ± 0.1128,
20NG standardized −0.0233 ± 0.0247.  Every base NLL improves, and accuracy rises
in all four cells, but predictive R̂ remains above 1.05 in all four means.  The
parameter-wise R̂ is worse: its median is about 2.0 and roughly 98% of finite
coordinates exceed 1.05 in every cell, including the 781-parameter standardized
heads.  This is not merely a width effect; the rotation head is weakly
identified, or has rotational gauge directions, and we did not prove which.
Neither diagnostic certifies a converged posterior at this budget.

A single seed was then run at 8,000 steps (burn 1,600, same 20 draws per chain):

| cell | acc % | NLL | predictive R̂ | param R̂ median |
|---|---:|---:|---:|---:|
| AG raw w8 | 36.67 | 1.3487 | 0.968 | 1.911 |
| AG std w1 | 30.83 | 1.3388 | 0.950 | 1.940 |
| 20NG raw w4 | 55.00 | 1.0511 | 1.340 | 2.010 |
| 20NG std w1 | 28.33 | 1.3598 | 1.313 | 1.890 |

Only the two AG cells reach predictive R̂ below 1.05; the 20NG cells do not,
and parameter R̂ stays near 1.9 everywhere.  This is one paired seed, so it is a
counterexample to "more steps always suffices", not a convergence result.

A finer step-size sweep located the rejection cliff and showed that 1,000-step
acceptance is dominated by an initial transient.  On the AG raw seed-18 cell at
1,000 steps, `beta=0.012` accepted 1,475/2,000 proposals (predictive R̂ 1.015,
parameter median 1.961), `beta=0.015` accepted 1,103 (1.084, 1.844), and
`beta=0.02` accepted 281 (1.392, 1.447).  `beta=0.03`, 0.1 and 0.3 still
rejected every proposal.  At `beta=0.02` acceptance was only 14% over the first
1,000 steps but 38.5% at 2,000, 47.3% at 4,000 and 48.6% at 8,000, so the chain
escapes a stiff start rather than sampling efficiently from the outset.

Longer runs did not converge.  At 8,000 steps `beta=0.015` accepted
11,166/16,000 (NLL 1.3685, predictive R̂ 1.043, parameter median 1.806);
`beta=0.02` accepted 7,780 (NLL 1.3517, predictive 1.222, parameter 1.701);
`beta=0.01` remained the best predictive cell (NLL 1.3487, predictive 0.968,
parameter 1.911).  Raising burn from 1,600 to 4,000 was worse for both
`beta=0.01` (predictive 1.091, parameter 2.376) and `beta=0.02` (1.227, 2.065),
so the failure is not a short burn-in.  Retaining 100 draws per chain instead of
20 left parameter R̂ at 1.998 (rank-normalised 1.787) while predictive R̂ was
0.997, so 40 draws were not the cause.  Rank-normalised parameter R̂ was 1.735
at `beta=0.01` and 1.604 at `beta=0.02`, confirming the raw statistic is not
merely a heavy-tail artefact.  The experiment now prints `param_rank_rhat`
alongside `param_rhat`.

Two independent-start rescues also failed.  Starting each chain at an
independent Gaussian perturbation of the anchor gave 0/2,000 accepted proposals
at spread 0.05 and 0.25 (chains froze at their starts), while spread 0.01 kept
1,676/2,000 accepted but raised predictive R̂ from 1.188 to 1.651.  The tiny
accepted step is load bearing; it cannot be replaced by a larger one without a
better proposal.

R12 score comparisons are therefore frozen at the 250-step table, with the
1,000-step table reported only as a budget-sensitivity check.  No arm or cell is
promoted.  The tuned softmax control is unchanged and still dominates.  Before
another prior or architecture sweep, either use a preconditioned/adaptive
proposal (or a reduced, identifiable parameterization) or set a predictive-R̂
gate and run long enough to pass it; adding steps alone is not sufficient.

**Diagonal preconditioner and linear angle head (2026-11-21).**  Both fixes named
above were implemented and measured on the same four selected cells, seed
20261118, 80 fit / 80 calibration / 120 test rows, 2 chains, 40 retained draws,
the same Adam pretrain and the same 8,000 steps (burn 1,600, 16,000 proposals).
The first is a per-coordinate pCN proposal: `beta_i` is frozen from a cloned
pilot session's pooled within-chain sample variance,
`beta_i = fraction * sd_i / (sigma * prior_scale_i)`, clipped to `[1e-4, 1]`.
The pilot has its own clone and does not advance the production chains or RNG,
and equal betas recover the scalar kernel exactly (kernel/config tests pass
25/25).  The second is a linear angle head (`--linear-head`): no hidden layer,
so hidden-unit sign flips and permutations are absent by construction.

Scalar and pilot (fraction 0.1) at 8,000 steps; predictive R̂ is printed with
its draw-NLL MCSE in parentheses.

| cell | arm | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | accepted |
|---|---|---:|---:|---:|---:|---:|---:|
| AG raw w8 | scalar | 36.67 | 1.3487 | 0.968 (0.035) | 1.911 | — | 14,018/16,000 |
| AG raw w8 | pilot 0.1 | 32.50 | 1.3062 | 1.043 (0.036) | 1.816 | 1.678 | 11,400/16,000 |
| AG std w1 | scalar | 30.83 | 1.3388 | 0.950 (0.011) | 1.940 | 1.758 | 7,856/16,000 |
| AG std w1 | pilot 0.1 | 25.83 | 1.3523 | 1.014 (0.008) | 2.038 | 1.810 | 8,926/16,000 |
| 20NG raw w4 | scalar | 55.00 | 1.0511 | 1.340 (0.030) | 2.010 | 1.791 | 15,523/16,000 |
| 20NG raw w4 | pilot 0.1 | 50.00 | 1.1542 | 1.291 (0.045) | 1.808 | 1.671 | 12,527/16,000 |
| 20NG std w1 | scalar | 28.33 | 1.3598 | 1.313 (0.012) | 1.890 | 1.756 | 7,678/16,000 |
| 20NG std w1 | pilot 0.1 | 30.00 | 1.3835 | 0.967 (0.007) | 2.137 | 1.828 | 13,937/16,000 |

The per-coordinate kernel lowers the parameter R̂ median by about 0.10 on the
two raw cells, but raises it on both standardized cells.  It lowers predictive
R̂ only on 20NG standardized, and there the point NLL is worse (1.3835 vs
1.3598).  No cell passes both predictive R̂ < 1.05 and parameter R̂ < 1.05.
Acceptance is not monotone in the frozen step size: the standardized cells now
accept more proposals yet mix no better.  The pilot fraction was selected on AG
raw alone: 0.5 froze every chain (0/16,000 accepted, NLL 2.1511), 0.05 gave
predictive 1.004 but parameter 1.948 and rank 1.753 with worse NLL 1.3872, and
0.1 was used unchanged on the other three cells.  This is one pilot setting and
one seed, not a tuned preconditioner.

Linear angle head at the same budget.  For the standardized cells the scalar
tuned beta accepted 0/16,000, so the linear head was rerun at `beta=1e-4`; that
is a proposal re-tune and is labelled as such, not a same-kernel comparison.

| cell | beta | params | accepted | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AG raw | 0.01 | 4,614 | 10,604/16,000 | 25.00 | 1.4345 | 0.977 (0.040) | 1.949 | 1.760 |
| AG std | 1e-4 | 4,614 | 15,743/16,000 | 28.33 | 1.4647 | 1.063 (0.035) | 2.074 | 1.821 |
| 20NG raw | 0.005 | 4,614 | 14,872/16,000 | 16.67 | 1.5034 | 1.003 (0.039) | 2.044 | 1.807 |
| 20NG std | 1e-4 | 4,614 | 15,722/16,000 | 26.67 | 1.4656 | 0.981 (0.040) | 2.016 | 1.803 |

The linear head is a clear regression: every NLL is worse than its hidden-head
scalar baseline (AG raw 1.4345 vs 1.3487, AG std 1.4647 vs 1.3388, 20NG raw
1.5034 vs 1.0511, 20NG std 1.4656 vs 1.3598), and 20NG raw falls to 16.67%
accuracy.  It mixes (14,872–15,743 accepted where retuned) but parameter R̂ stays
near 2.0 and rank R̂ near 1.80.  Removing hidden-unit sign and permutation gauges
therefore does not fix the parameter-space failure in this configuration; the
non-identification is not mainly that gauge.  For the standardized cells the
linear head also has more parameters (4,614 vs 781), so this is a reduced
*architecture*, not a reduced parameter count.

Budget gate at 32,000 steps (burn 6,400, 2 chains, 40 draws, 64,000 proposals).
The predictive-R̂ gate is 1.05.

| cell | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | accepted | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 20NG raw | 69.17 | 0.9171 | 1.489 (0.041) | 1.913 | 1.745 | 60,706/64,000 | fail |
| 20NG std | 23.33 | 1.6242 | 1.018 (0.030) | 2.500 | 1.992 | 110/64,000 | fail |

The 20NG raw point score improves sharply with budget (NLL 1.0511 → 0.9171,
accuracy 55% → 69.17%), but predictive R̂ rises from 1.340 to 1.489 and
parameter R̂ is unchanged, so the better fit is not a convergence result.  The
standardized cell degrades: acceptance collapses from 7,678/16,000 at 8,000
steps to 110/64,000, NLL worsens to 1.6242 and parameter R̂ rises to 2.500.
Neither cell passes the gate, and this is one seed.  The budget-sensitivity
check therefore closes negatively for both 20NG cells.

**Four-chain control (2026-11-21).**  The parameter-R̂ values above use two
chains, the minimum for the statistic, so both 20NG cells were rerun at 8,000
steps with four chains and 80 retained draws (20 per chain), same seed, beta and
pretrain.  The 32,000 proposals are the same total budget as the two-chain run.

| cell | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | param >1.05 | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG raw | 60.83 | 1.0247 | 1.188 (0.020) | 2.409 | 2.056 | 99.9% | 31,050/32,000 |
| 20NG std | 29.17 | 1.3586 | 1.131 (0.008) | 2.238 | 1.971 | 100.0% | 15,126/32,000 |

The failure is not a two-chain estimator artifact.  Against the paired two-chain
rows, the predictive R̂ improves somewhat (raw 1.340 → 1.188, standardized
1.313 → 1.131) but still fails the 1.05 gate, while the parameter and rank
medians rise (raw 2.010 → 2.409 and 1.791 → 2.056; standardized 1.890 → 2.238
and 1.756 → 1.971).  With four chains, 100% of the standardized head's 781
coordinates and 99.9% of the raw head's 3,106 coordinates exceed 1.05.  A broad
parameter-space disagreement is therefore present under the standard chain
count, not just under a fragile two-chain estimate.

**Gradient-informed sampler (2026-11-21).**  `SessionConfig::sampler = Mala`
adds an exact preconditioned MALA kernel beside pCN.  In normalized prior
coordinates `u_i=(θ_i−c_i)/(σ s_i)` the proposal is
`u' = u − (τ²/2)∇Φ(u) + τ ξ`, with the exact Born gradient and the full
reverse-density MH ratio.  The step `τ` (`--mala-step`) was selected only by
short acceptance pilots, never by test NLL: on standardized 1,500-step runs,
`τ=0.002` gave 50% acceptance (0.001 gave 86%); on raw, 0.020 gave 44% at 1,500
steps (0.024 gave essentially none).  The matched 8,000-step runs then used the
same seed, split, pretrain and 2-chain/40-draw budget as the pCN rows.

| cell | kernel | step | acc % | NLL | predictive R̂ (MCSE) | param R̂ | rank R̂ | accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG std | pCN | 0.0025 | 47.99 | 1.3598 | 1.313 (0.012) | 1.890 | 1.756 | 7,678/16,000 |
| 20NG std | MALA | 0.002 | 59.48 | 1.3904 | 1.530 (0.010) | 2.034 | 1.794 | 9,517/16,000 |
| 20NG raw | pCN | 0.005 | 97.02 | 1.0511 | 1.340 (0.030) | 2.010 | 1.791 | 15,523/16,000 |
| 20NG raw | MALA | 0.020 | 71.62 | 1.0560 | 1.224 (0.052) | 1.676 | 1.581 | 11,459/16,000 |

Gradient information does not rescue either cell.  Standardized MALA has healthy
acceptance but a worse predictive R̂ and NLL than pCN; raw MALA lowers the
parameter and rank R̂ but is slightly worse on NLL and still far above the 1.05
gate.  The raw MALA acceptance also rose from 44% in the pilot to 71.6% in the
8,000-step run, so that pilot did not fix a stationary acceptance rate.

**Posterior-geometry diagnostic (2026-11-21).**  `RotationSession::posterior_geometry`
reports, per coordinate: within-chain posterior sd divided by the prior sd; a
flat flag (`≥0.5`, prior-dominated), an identified flag (`≤0.1`); and the
largest pairwise chain-mean difference divided by the pooled within-chain sd, a
standardized separation with a `>2` flag.  The point is to distinguish chain
disagreement on tightly identified coordinates (a shifted or multimodal mode)
from broad drift along weakly identified ones.  It is descriptive and does not
replace R̂.

| cell | kernel | sd/prior median | flat frac | identified frac | separation median | separated frac | separated / flat | separated / identified |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG std | pCN | 0.054 | 0.000 | 0.986 | 1.564 | 0.392 | — | 0.392 |
| 20NG std | MALA | 0.048 | 0.000 | 0.985 | 1.629 | 0.406 | — | 0.406 |
| 20NG raw | pCN | 0.149 | 0.000 | 0.084 | 1.789 | 0.446 | — | 0.565 |
| 20NG raw | MALA | 0.476 | 0.420 | 0.000 | 1.308 | 0.305 | 0.243 | — |

On standardized, ~99% of coordinates are identified and ~40% of those are
separated by more than two within-chain sds under both kernels; this is not a
flat-direction story and MALA does not reduce it.  Raw pCN has almost no flat
coordinates and 56.5% of identified coordinates are separated.  Raw MALA is the
outlier: it leaves a broad, prior-dominated posterior (42% flat, median
sd/prior 0.48), so its lower R̂ reflects a broader ensemble, not a better-resolved
mode.  Separation can also arise from within-chain drift, so these columns are
diagnostic rather than a convergence verdict.

**Prediction-space chain distance (2026-11-21).**  Parameter separation only
matters if it changes decisions.  For each chain the experiment now averages its
retained draws' test distributions over rows, and reports the total-variation
distance `0.5 * Σ_k |p_a(k) − p_b(k)|` for every chain pair (`prediction_chains`).
This is a descriptive chain-mean distance; it can hide row-level differences
that cancel.

| cell | chains | TV mean | TV max | separated / identified |
|---|---:|---:|---:|---:|
| 20NG std | 2 | 0.0928 | 0.0928 | 0.392 |
| 20NG std | 4 | 0.0991 | 0.1253 | 0.829 |
| 20NG raw | 2 | 0.0328 | 0.0328 | 0.565 |
| 20NG raw | 4 | 0.0336 | 0.0474 | 0.870 |

The standardized disagreement survives in the average predictions: four chains
differ by about 10 percentage points total variation (maximum 12.5), and 82.9%
of identified coordinates are separated.  For a four-class report that is a
material predictive difference, so this is not merely a hidden-unit sign gauge
or a parameterization artifact.  Raw's two-chain pooled distance was only 3.3
points; the four-chain control below shows that this pooled number can be
misleading.  These are one seed and one budget; the columns carry no sampling
interval and are not a convergence certificate.

The pooled distance could still hide either a coherent block switching sides or
a diffuse shift that partially cancels across rows.  The four-chain runs
therefore also average each chain within every row and true class
(`prediction_rows`, `prediction_classes`).  Row-level pairwise TV for
standardized averages 0.1468 (maximum 0.3356); 77.8% of row pairs exceed 0.10,
and at least two chains disagree on the top label in all 120/120 test rows.  Its
per-true-class mean TVs are `[0.0865, 0.1317, 0.1019, 0.1085]` (mean 0.1072):
every class is affected, not one confused label.  This is a pervasively
different decision map, not a hidden-unit permutation.

The raw four-chain control overturns the pooled reading.  Its pooled TV is only
0.0336 (maximum 0.0474), but row-level TV averages 0.3476 (maximum 0.8688),
96.3% of row pairs exceed 0.10, the top labels differ in 109/120 rows, and the
per-true-class TVs are `[0.0851, 0.0988, 0.0997, 0.0980]` (mean 0.0954).  Raw's
small pooled distance is cancellation between rows, not predictive agreement;
its row-level disagreement is larger than standardized's.  Pooled chain-mean TV
must not be used as evidence of predictive consensus.  All these means are
descriptive, use chain-averaged predictions, and carry no uncertainty interval.

**Does row disagreement track errors? (2026-11-21).**  For each test row the
experiment correlates its mean pairwise chain TV with the pooled prediction's
NLL.  This is descriptive, not a validated error detector.

| run | Pearson | Spearman | rows |
|---|---:|---:|---:|
| 8k pCN, seed 20261118 | 0.066 | 0.099 | 120 |
| 8k PT, seed 20261118 | 0.164 | 0.116 | 120 |
| 8k pCN, seed 20261119 | 0.302 | 0.206 | 120 |
| 8k PT, seed 20261119 | 0.278 | 0.368 | 120 |
| 32k pilot, seed 20261118 | 0.144 | 0.236 | 120 |

The correlation is weak-to-moderate and seed-dependent: Pearson ranges from
0.066 to 0.302 and Spearman from 0.099 to 0.368.  Rows where chains disagree
more are somewhat more likely to have a worse pooled NLL, but the relationship
is not stable enough to treat row TV as an error predictor; it remains a spread
diagnostic.  Per-chain scores also reverse across seeds: at seed 20261118 pCN
scores `acc=[0.292, 0.233]`, `nll=[1.347, 1.410]` against PT
`acc=[0.358, 0.408]`, `nll=[1.347, 1.316]`, but at seed 20261119 pCN scores
`acc=[0.458, 0.450]`, `nll=[1.164, 1.207]` against PT
`acc=[0.408, 0.383]`, `nll=[1.228, 1.248]`.  The tail `acc=[0.258, 0.275]`,
`nll=[1.357, 1.401]` is the 32k pilot.

**Parallel tempering (2026-11-21).**  `Sampler::ParallelTempering` pairs each
cold chain (`T = 1`) with a tempered power posterior (prior fixed, likelihood
scaled by `1/T`), attempts an adjacent-pair replica exchange after every sweep,
and retains only cold chains.  The 8k standardized cell used four chains with
`T = [1, 2, 1, 2]` and 80 draws (40 per cold chain), same seed, burn and beta;
the matched pCN baseline used two chains and 80 draws.

| seed | arm | acc % | NLL | predictive R̂ | param R̂ | pooled TV | row TV | top-label disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261118 | pCN | 26.67 | 1.3580 | 1.350 | 1.922 | 0.0930 | 0.1480 | 99/120 |
| 20261118 | PT | 38.33 | 1.3157 | 1.080 | 2.024 | 0.0516 | 0.1226 | 42/120 |
| 20261119 | pCN | 50.00 | 1.1455 | 1.012 | 2.054 | 0.0599 | 0.1985 | 41/120 |
| 20261119 | PT | 45.00 | 1.1906 | 1.041 | 1.819 | 0.0853 | 0.2047 | 52/120 |

On seed 20261118, tempering cuts pooled TV by 45% and top-label disagreement
by 58%, and both cold chains score better than either pCN chain.  That advantage
does not replicate.  On seed 20261119, pCN has the lower NLL (1.1455 vs 1.1906),
the higher accuracy (50% vs 45%), slightly lower row TV (0.1985 vs 0.2047) and
fewer label disagreements (41 vs 52); PT improves only parameter R̂ (1.819 vs
2.054).  Across the two paired seeds the mean NLL delta is +0.0014 (PT minus
pCN) and the mean row-TV delta is −0.0096, but both changes have opposite signs
by seed, so no stable point-prediction or row-disagreement benefit is established.
Both arms remain far above the parameter-R̂ gate.  Seed 20261119 also shows that
a predictive-R̂ pass is not sufficient: pCN reaches 1.012 there while parameter
R̂ is 2.054.  PT remains a diagnostic control, not a convergence remedy; this is
two seeds at one budget, not a general claim.

The PT report now carries `swap_attempts` and `swap_accepted`.  Rerunning both
PT cells with that instrumentation reproduced every score above exactly.  Seed
20261118 accepted 2,129/16,000 exchanges (13.3%, one cold exchange per ~7.5
sweeps); seed 20261119 accepted 1,939/16,000 (12.1%, per ~8.3 sweeps).  Both runs
therefore exchanged replicas actively and at similar rates, so the seed reversal
is not explained by one run failing to swap.  The rate is low enough to call the
1↔2 ladder a weak mixer, but no acceptance threshold was preregistered and this
is a diagnostic, not a convergence claim.

**Pilot preconditioning at 32k (2026-11-21).**  The 20NG standardized cell was
rerun at 32,000 steps (burn 6,400, 2 chains, 40 draws) with a 0.1-fraction pilot
that froze per-coordinate betas `min=0.000324 median=0.000919 max=0.002885`.

| run | NLL | predictive R̂ (MCSE) | param R̂ | rank R̂ | accepted |
|---|---:|---:|---:|---:|---:|
| scalar β=0.0025 | 1.6242 | 1.018 (0.030) | 2.500 | 1.992 | 110/64,000 |
| pilot betas | 1.3690 | 1.152 (0.007) | 2.125 | 1.826 | 57,523/64,000 |

The pilot prevents the acceptance collapse (89.9% versus 0.17%) and improves
NLL and both parameter statistics, but predictive R̂ moves the wrong way and all
three gates still fail.  The rerun reproduces the earlier score exactly and now
also records geometry (`sd_norm=0.0555`, identified 89.1%, separation median
1.937, separated/identified 0.491) and row disagreement: pooled TV 0.0824, row
TV 0.1097 (maximum 0.2604), top-label disagreement 79/120, class-TV mean 0.0824.
Against the 8k pCN baseline (row TV 0.1480, 99/120) this lowers disagreement,
but does not remove it; the comparison changes budget, preconditioner and
retained draws per chain (20 versus 40), so it is not a clean budget effect.
It is one seed, and 32k remains a sensitivity/gate check, not a new frozen R12
score.

Reproduce a budget cell with

```bash
cargo run --offline --release -p pcnl --example rotation_experiment -- \
  --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --mode single \
  --seed 20261118 --fit-per-class 20 --cal-per-class 20 --hidden 8 \
  --representation raw --beta 0.01 \
  --steps 1000 --burn 200 --chains 2 --draws 40 \
  --pretrain 1000 --pretrain-lr 0.01 --softmax-pretrain 0
```

## R13 first vector-head execution (2026-11-22)

The R13 vector model (`src/vector_model.rs`) and synthetic-regression example
(`examples/vector_regression.rs`) implement the four-position, four-block head,
the full-data Gaussian regression potential, the exact CN/prior-refresh mixture
and a JSON checkpoint round-trip. This is a pipeline check, not a text result.

On the frozen 20 Newsgroups embeddings (240 train / 120 test, `768 → 4×16 → 2`,
fixed synthetic teacher, observation noise 0.05), 300 full-batch Adam steps
reached train RMSE 0.000286 and test RMSE 0.099276. The reopened checkpoint
predicted `[-1.6128643, -0.82934713]` exactly.

The first sampler setting (`ρ=0.99`, prior scale 0.05) accepted 0/397 CN moves;
the prior covariance was much wider than the 240-row posterior. With
`ρ=0.9999`, prior scale 0.01, it accepted 224/398 CN moves and 0/2 refreshes,
retaining 20 draws. Draw-mean test RMSE was 0.101129. Parameter-only 90%
intervals covered 14/240 coordinates; noise-augmented predictive intervals
covered 142/240. Coverage therefore fails and must not be reported as calibrated.

### Known-target recovery (2026-11-22)

`examples/known_target.rs` runs the production CN/refresh kernel behind the
`EnergyTarget` seam on a one-dimensional target,
`0.75·N(−2,1) + 0.25·N(2,1)`, with four chains started at `[-6,-3,3,6]`.  With
50,000 steps, 10,000 burn, thin 5 and 8,000 retained draws per chain, the pooled
left-basin mass was `0.73378` against the analytic `0.73862` (error `−0.00484`,
Monte-Carlo SE `0.01304`); the pooled mean was `−0.97689` against `−1`, and the
central 90% interval was `[−3.50191, +2.87135]`.  Rank-normalized R̂ was
`1.00344` on the coordinate and `1.00547` on the basin indicator; bulk ESS was
`1504.8` and `1148.5`, so both gate rows pass.  Tail ESS was `3838.2` on the
coordinate but `NaN` on the binary indicator, so the tail gate is not reported
as passing for that indicator.

### Four-chain vector diagnostic (2026-11-22)

Four jitter-separated warm starts (`jitter=0.01`) of the synthetic vector
posterior were run for 1,000 steps each with 250 burn and 200 retained draws.
At `ρ=0.999`, prior scale 0.01, CN acceptance was 82, 89, 86 and 102 of ~1,000
per chain.  Rank-normalized R̂ was **2.69621** and minimum bulk ESS **4.7**
(tail ESS 9.3), so the four-chain gate **fails**.  At `ρ=0.9999` the analogous
run accepted 538, 523, 564 and 544 moves but gave R̂ **2.74392** and minimum
bulk ESS **4.6**.  A 300-step probe of the same starts put the per-chain
first-row draw means within about 0.04 of one another, so the failure is slow
mixing in 57,378 parameters rather than distinct modes.  The prefixes are too short for the observed autocorrelation
(250 burn against an integration time of hundreds of steps), and the 800 pooled
draws cannot reach 400 effective draws.  These are measured failures, not a
promotion result.  Draw-mean test RMSE was 0.100267, parameter 90% coverage
0.2542 and noise-augmented predictive coverage 0.6000.

### Gradient scaling and the dimension wall (2026-11-22)

The `EnergyTarget` adapter for `VectorRegression` reported the energy scaled by
`λ·n` but returned the **unscaled** mean-loss gradient, so the CN drift was
`240×` too small at `n = 240`.  The gradient is now scaled identically.  After
the fix, 300-step single-chain probes accepted 261/295 CN moves at `ρ=0.9999`,
13/290 at `ρ=0.999` and 0/299 at `ρ=0.99`; before the fix the four-chain
`ρ=0.999` run accepted 82–102 of ~1,000 per chain.

A two-chain 5,000-step run (1,250 burn, 3,750 draws per chain) at `ρ=0.9999`
then accepted 4,620 and 4,608 CN moves but still gave rank R̂ `1.927–2.216` and
per-observable bulk ESS `2.5–4.9` (tail `10–47`).  At 92% acceptance that is an
integrated autocorrelation time of a few thousand steps, so bulk ESS 400 would
require about `10⁶` evaluations per chain.  The failure is a property of the
transition kernel's dimension scaling, not of the objective or the run length.

A pilot-preconditioned diagonal proposal does not remove the wall.  At
`ρ=0.99` the kernel's shrunk mean moves every coordinate by 1% of its value per
step, far beyond the local posterior step, so proposals are rejected: `0/2,981`
for a 34-coordinate readout-only run (pilot 400 steps, `var/prior` p50
`6.2e−3`, inflate `0.5`) and `0/1,984` for the first full-dimension chain
(pilot inflate `2.0`).  With an exact posterior covariance the CN kernel's ESS
per step is about `acc·(1−ρ)/2` with `log acc ≈ −6(1−ρ)d`, maximized near
`1/(12·e·d)` — roughly `1e−3` per step at `d=34`.

`ReadoutTarget` and `--readout-only` sample only the 34 affine-readout
coordinates with the backbone frozen at its Adam point estimate.  This is an
explicitly conditional target, not the 57,378-parameter posterior.  Four chains
of 3,000 steps (750 burn, thin 2, 1,125 draws each) at `ρ=0.9999` with a
400-step pilot covariance accepted 70/2,971, 71/2,966, 84/2,979 and 83/2,968 CN
moves (0 prior refreshes), with rank R̂ max **3.53012** and minimum bulk ESS
**4.4** (tail 8.3): the gate still **fails**.  Draw-mean test RMSE was
0.099114, parameter 90% coverage 0.0292 (7/240), predictive coverage 0.6125
(147/240).

### Exact readout-conditional posterior (2026-11-23)

The readout-conditional target is linear-Gaussian.  With design `Φ`, noise
variance `s²`, prior scale `λ` and prior covariance `c`, the precision is
`A = (λ/s²)ΦᵀΦ + diag(1/c)` and the posterior is `N(m, A⁻¹)`;
`ReadoutTarget::exact_posterior` forms it by Cholesky and `--proposal exact`
uses it in the CN kernel.  On the fitted Adam backbone the target gradient at
`m` has RMS `8.673e−4` (the Gaussian self-consistency check).  The
posterior-to-prior variance ratio is `p10 7.719e−2`, `p50 1.852e−1`,
`p90 3.320e−1`; the two bias coordinates of `m` are `−1.487` and `−0.730`
against a prior standard deviation of `0.05`, i.e. the posterior lies 15–30
prior-σ from the origin.  This last fact is why a prior-referenced kernel
fails here.

With reference mean zero the CN proposal is `N(ρθ, (1−ρ²)c)`, centred ~30
prior-σ from the posterior: a 200-step scan from the exact draw accepted
`0/199` at `ρ=0.5` and `ρ=0.9`, `2/199` at `ρ=0.99`.  A
**posterior-referenced** variant (`run_chain_with_posterior_reference`)
replaces the reference with `m` and the prior pull with the full target
gradient, `mean = m + ρ(θ−m) − ((1−ρ²)/2)·A⁻¹∇U` with covariance
`(1−ρ²)A⁻¹`.  It removes the prior-mean mismatch but remains under-dispersed:
from a tail draw it accepted `0–2/199` at `ρ ≤ 0.9` and `102/199` at
`ρ=0.99`.  A real dense-kernel defect was found and fixed in this work: the
quadratic form is `δᵀAδ = ‖Lᵀδ‖²` over the columns of the Cholesky factor
(`L Lᵀ = A`), not `‖Lδ‖²`.

Starting all chains at the analytic mean (`--init mean`, new) with `ρ=0.5`
gave CN acceptance `1281/2975`, `1155/2971`, `1221/2962` and `1180/2970`
(~40%), matching the exact-covariance prediction.  Four chains of 3,000 steps
with all post-burn draws retained (thin 1, 2,250 draws each) still gave rank R̂
max **1.02163** and minimum bulk ESS **254.9** (tail 37.8): five of eight
observables cleared ESS 400, `gate=false`.  The measured ESS/step is
`255/2,250 ≈ 0.11`, so about 3,600 post-burn draws per chain are needed for
bulk ESS 400.  (The thinned 225-draw run's ESS of 223.5 was an optimistic
short-run estimate, not a separate ceiling.)

At that budget — four chains of 6,000 steps, 1,500 burn, thin 1 (4,500 draws
each) — the same kernel **passes**: acceptance `2553/5953`, `2456/5941`,
`2467/5935`, `2316/5938`; rank R̂ max **1.00616**; minimum bulk ESS **620.9**
(tail 201.3); `gate=true`.  Draw-mean test RMSE was `0.105528`, parameter
coverage `0.1583` (38/240), predictive coverage `0.5917` (142/240).  HMC
reaches the same gate with 750 draws per chain, about 6× fewer.

#### Why the readout-conditional 90% coverage is `0.158` (2026-11-25)

That number is not a sampler or ESS artifact and it is not the posterior's
calibration: it is the held-out **generalisation** error of the pointed
backbone showing through a coverage metric. Four things pin it down.

1. **It is sampler-independent.** `readout-exact.log` (HMC, exact mass, leapfrog
   4, 1,000 steps, 750 draws) reports the *identical* `38/240` and `141/240`;
   CN at 4,500 and at 9,000 draws both report `38/240` (`141/240` and
   `142/240`). Four samplers, two kernels, two chain lengths, one count.
2. **The bands are tight because the backbone memorised the train split.**
   `adam_fit` trains the whole model, so `train_rmse=0.000286` — far under the
   `0.05` noise — while `test_rmse=0.099276` is ~2× the noise. Freezing that
   backbone and sampling the 34 readout coordinates against an exact
   likelihood with `σ=0.05` (whose fit is essentially perfect on train) gives a
   posterior prediction band of sd ≈ `0.013`, i.e. half-width ≈ `0.021`.
3. **The observations sit outside it.** With band half-width `0.021` and
   held-out residual sd ≈ `0.099`, a Gaussian residual model predicts parameter
   coverage `15.8%` — the measured value.
4. **The predictive line closes the loop.** Adding only `σ=0.05` in quadrature
   gives half-width `√(0.013² + 0.05²)·1.645 = 0.085`, which against the same
   residuals predicts `59%` (measured `58.75%`). Solving both coverages
   jointly for the residual sd gives `0.1035` against the logged `test_rmse`
   `0.0993` — a 4% mismatch.

So `param_interval90_coverage` here compares a held-out observation against the
credible band of the *predicted mean*; it inherits the backbone's held-out
error and is a misspecification/generalisation diagnostic. The row's own
`limitations:` line already declines a calibration claim.

**The conditioning-split check (2026-11-25).** Readout-only runs now print a
`calibration conditioning_split=train ...` line — the same 90% bands scored on
the split the likelihood actually conditions on, plus the residual sd there,
its degrees-of-freedom-corrected value, and the declared noise. The line is
part of the summary set `scripts/check-pcnl-gates` diffs, so it cannot drift
silently.

| row | train param 90% | train predictive 90% | band half-width | residual sd | residual sd (df-adj) | declared σ |
|---|---|---|---|---|---|---|
| `readout-exact` (HMC, 3,000 draws) | `480/480` | `480/480` | `0.018983` | `0.004343` | `0.004505` (df `446`) | `0.05` |
| `cn-exact-dense` (CN, 18,000 draws) | `480/480` | `480/480` | `0.019031` | `0.004451` | `0.004618` (df `446`) | `0.05` |
| `pilot-calibrated` (full parameter, 1,200 draws) | `478/480` | `480/480` | `0.066869` | `0.018003` | `NaN` (df `−56,898`) | `0.05` |

The bands are calibrated on the split they condition on, under both kernels:
both cover every one of the 480 train observations, exactly as the analytic
linear-Gaussian posterior requires. The `0.158` therefore belongs to the
held-out split, not to the sampler.

**The full-parameter contrast (2026-11-25).** `--calibration` adds the same line
to a full-parameter run, so `pilot-calibrated` scores the *identical* train rows
and 90% band while marginalising all 57,378 coordinates. The band is **3.5×
wider** (`0.0669` vs `0.0190`) and the in-sample residual sd four times larger
(`0.0180` vs `0.0044`), which is exactly the point: the readout-conditional
band is tight because the frozen backbone already fits train, not because the
sampler is overconfident. On train the full posterior over-covers (`478/480`
parameter, `480/480` predictive) while on the held-out split the *same* run
covers only `170/240` and `200/240` — in-sample it has absorbed the noise, so
its train coverage is the memorisation again and the held-out numbers are the
honest ones. `df` is `480 − 57,378 < 0`, so the df-adjusted residual sd is
`NaN` by construction: with more parameters than train targets there is no
unbiased in-sample noise estimate, which is why the full-parameter row can only
supply the width contrast. The row is an `alt` (opt-in) entry: emitting the
line by default would invalidate every committed full-parameter log.

The same line also answers "estimate the noise scale", and the answer is that
this split cannot: the frozen backbone was fitted on train, so the in-sample
residual (`0.0043–0.0045`) is an order of magnitude *below* the declared
`0.05`, and it is the held-out residual sd (`0.0993`) that dominates the
coverage arithmetic in points 3–4 above. The check certifies the conditional
posterior and the generalisation diagnosis; it does not re-estimate σ from data,
and the row remains a readout-conditional result rather than a calibration
claim for the full model.

### CN dimension-sweep scaling (2026-11-23)

`examples/cn_dimension_sweep.rs` runs the production prior-referenced CN
kernel on `N(0, I_d)` for one chain of 4,000 steps (1,000 burn), seeded by
`(d, ρ)`.  With the proposal covariance set to the target covariance `I_d`
(`--exact-proposal`) at `ρ=0.99`:

| `d` | 2 | 6 | 18 | 34 | 66 | 130 | 256 |
|---|---|---|---|---|---|---|---|
| acceptance | 0.9313 | 0.8752 | 0.7802 | 0.7153 | 0.5717 | 0.4265 | 0.3003 |
| lag-1 autocorrelation | 0.9875 | 0.9887 | 0.9915 | 0.9909 | 0.9930 | 0.9957 | 0.9957 |
| ESS/step | 0.00630 | 0.00567 | 0.00427 | 0.00456 | 0.00352 | 0.00217 | 0.00214 |

Reproduce with `--steps 4000 --dims 2,6,18,34,66,130,256 --rhos 0.99
--exact-proposal`.  The default `--c 0.1` sweep over the same dimensions gives
best ESS/step `0.0014–0.0179` with no monotone trend.

The kernel is a random walk, so ESS/step is bounded near
`(1−ρ)/(1+ρ) = 0.0050` at `ρ=0.99` independent of dimension; matching the
target covariance raises acceptance (0.93 → 0.30) but does not make it an
independence sampler.  Four thousand draws per chain therefore yield about 20
effective draws at `ρ=0.99`, which is the measured dimension-scaling limit
behind the four-chain CN failure at that budget.

### HMC with the exact metric passes the four-chain gate (2026-11-23)

A diagonal-mass HMC collapsed its step size (`eps_frozen ≈ 5e−5`). Part of
that was a convention defect: with `K(p)=½Σp²/m` and `θ′=θ+εp/m` the dynamics
are `θ″=−M⁻¹Aθ`, so `m` must be the **precision** (`M=A`), not the variance;
passing variances squares the curvature range. The diagonal call sites now
pass precisions; the readout diagonal run settles at `ε ≈ 6e−3` instead of
collapsing, but still fails (R̂ `1.70725`, bulk ESS `6.1` at 400 steps), so the
correlation limit is separate from the inversion. The recorded gate therefore
uses the exact whitened metric.  The exact metric is the posterior
**precision**: in whitened coordinates `u = Lᵀ(θ−m)` (`L Lᵀ = A`) the Gaussian
target is exactly standard normal.  `run_hmc_chain_whitened` therefore uses
unit mass, `N(0, I)` momenta and the transformed force `Gᵀ∇U` with `G = L⁻ᵀ`,
while every MH ratio is still computed with the full target in `θ`.

Four chains, `--init mean --kernel hmc --hmc-mass exact --hmc-leapfrog 4`,
1,000 steps, 250 burn, thin 1 (750 draws per chain): dual averaging settled at
`ε = 0.587–0.638` with mean acceptance `0.835–0.856`.  Rank R̂ max **1.00935**
(all eight observables `≤ 1.00935`), minimum bulk ESS **3000.0** (the cap of
four chains × 750 draws), minimum tail ESS **906.4**, `gate=true` under the
declared definition `4_chains_rank_rhat<1.01_bulk_ess>=400`.  Draw-mean test
RMSE was `0.105302`, parameter 90% coverage `0.1583` (38/240) and predictive
coverage `0.5875` (141/240); coverage is still not nominal.  The passing target
is the **readout-conditional 34-coordinate** posterior with a frozen backbone,
on fixture embeddings, warm-started at the analytic mean: this validates the
HMC kernel, not the full-parameter posterior or calibration.

### Full-parameter HMC with a corrected metric (2026-11-23)

`adam_fit` minimizes the mean loss without the sampling prior, so its optimum
is not stationary for the sampled potential (`start_grad_rms ≈ 152` on the
57,378-coordinate target). `--init mode` now runs Adam on `U(θ)+½Σθ²/c`
(3,000 steps, `lr=5e−3`: final potential `29.19`, `mode_grad_rms=1.06e−1`), and
`--hmc-mass pilot` uses the pilot variance as a precision. The pilot step is
configurable (`--pilot-rho`): at `ρ=0.9999` it accepted only `4/400`, while at
`ρ=0.99999` it accepted `499/600` with variance ratio p50 `8.7e−4`.

Four chains of 400 steps, leapfrog 4, `--start-jitter 1e−4`: `ε =
0.101–0.108`, acceptance `0.858–0.864` (no step-size collapse), rank R̂ max
**2.61047**, bulk ESS min **4.7** (tail 10.0), `gate=false`. Draw-mean test
RMSE was `0.144261`, parameter 90% coverage `0.7083` (170/240), predictive
`0.8333` (200/240). With the previously inverted metric the same run gave
`ε≈3e−6` and ESS 5.5, so the fix repaired the scaling but not the observable
ESS: whitening needs the data Hessian's off-diagonal structure, not a diagonal
metric. This is the measured full-parameter limit, and the reported target is
prior-dominated and not the same predictor as the unregularized Adam fit.

### Legacy-gate reproducibility after the init change (2026-11-23)

**The init change does not touch the legacy path.** A control build with the
`init_flat_prior` branch removed emits output byte-identical to the shipped
binary under `--start-scale legacy` (mode `29.193841` / `1.061627e-1`, pilot
`499/600` with `var_over_prior p50 8.700e-4`, all four chain lines, RMSE,
R̂, ESS). The prior draw is opt-in; `legacy` passes the same `theta_vec` and
`--init mode` starts from `mode_base`, so nothing downstream can see the flag.

**Every recorded full-parameter gate reproduces bit-exactly** under
`--start-scale legacy`, once the full command is supplied (table and commands
below):

| gate | recorded | re-run (this binary, `--start-scale legacy`) |
|---|---|---|
| pilot metric, 400×4 | R̂ 2.61047, ESS 4.7, RMSE 0.144261, ε 0.1017–0.1077, acc 0.858–0.864 | identical: R̂ 2.61047, ESS 4.7, tail 10.0, RMSE 0.144261, coverage 0.7083/0.8333 |
| Hessian diag (64 probes), 1,000×4 | R̂ 2.50388, ESS 4.8, RMSE 0.150189, `clipped_to_prior` 27,622 | identical: R̂ 2.50388, ESS 4.8, tail 10.1, RMSE 0.150189, ε 6.902e-3, `clipped_to_prior` 27,622 |
| low-rank 8, 400×4 | R̂ 2.39408, ESS 4.9 | identical: R̂ 2.39408, ESS 4.9, tail 10.3 |
| low-rank 32, 1,000×4 | R̂ 2.01334, ESS 5.3 | identical: R̂ 2.01334, ESS 5.3, tail 10.1 |

The readout-conditional gate reproduces identically too (R̂ 1.00935, ESS 3000.0,
tail 906.4, RMSE 0.105302, coverage 0.1583/0.5875, ε 0.587–0.638).

**How the first reconstruction attempt missed, and why it matters.** A first
re-run omitted `--pilot-inflate 1` (ε came out `0.068–0.077` against the
recorded `0.101–0.108`, because the mass scales by `inflate` and ε by
`1/√inflate`), used 400 Hessian steps instead of the recorded 1,000, and left
`--thin` at its default. Those are not small cosmetic knobs: `--thin`/`--draws`
decide which post-burn draws the gate statistics see, and at ESS ≈5 that moves
R̂ by ±0.5 on an otherwise identical trajectory (holding ε and acceptance
fixed, `--thin 10 / 5 / 2` gave R̂ `2.548 / 2.112 / 2.171`). The recorded
numbers were never irreproducible — the reconstruction was incomplete.

Reproduction convention for this section: all commands below run from the repo
root against `./target/release/examples/vector_regression` (or
`vector_basins` / `vector_stream`), fixture
`docs/fixtures/ultrabert_20ng4_embeddings.safetensors`, `teacher_seed=20261122`,
and print into `logs/pcnl_r13_gates/`, which is committed together with
`scripts/check-pcnl-gates` — that script re-runs every one of these commands
and diffs the summary lines against the committed log, so a stale log or a
command that has drifted from its log fails loudly instead of silently. The
per-push CI lane runs `--quick`, which now selects three *canary* rows instead
of the full gate rows: `stream-canary` (50 Adam steps, 6 s), `pilot-canary`
(100 HMC steps, 45 s) and `readout-canary` (100-step exact-mass chain, 43 s),
94 s total on an idle machine against 492 s for the full twins. Each canary is
the same program, code path and seed at a reduced length, so a behaviour change
still moves its numbers, but their R̂/ESS are a drift baseline and **not** a
convergence claim — the `gate=` verdict is `false` at that length by
construction, and every table in this section cites the full rows. The full
rows remain in the default sweep and in the scheduled `--all` lane; the default
sweep skips canaries (they are subsumed) and both lanes print every row they
did not run. This is the trade the per-push budget forces: ~2 min of coverage
that fires on the push, with the complete gate sweep still scheduled.

The hosted runner is not the recording machine, so the lane compares with
`--structure`, not byte-for-byte, and the first GitHub runs are why. Measured on
that runner with no code change: the 50-step Adam canary moved
`test_rmse_mean` by 0.1% and one `z2` by 4.7%, and the 100-step HMC canary
moved `rhat_rank` `2.54906 → 1.91680` and `ess_tail` `13.4 → 22.8` — at 75
draws per chain those diagnostics are noise rather than estimates, and one
accept/reject flip also changes the counts inside `(93/240)`. So the lane
checks what is portable: the key sequence, the `gate=`/`sample_space=` words,
that every number stays finite (a `NaN` must stay `NaN`), and that the run
exits 0. Byte-exact reproduction stays `./scripts/check-pcnl-gates` on the
machine that recorded the logs, which is why that is the default; the
scheduled `--all` lane, whose rows are long enough for ESS/R̂ to mean something,
uses `--band 0.2` as a gross-drift net. The scheduled workflow also carries a
`gates-byte-exact` job — the plain script, `--all`, no band — on a self-hosted
runner labelled `r13-reference`, so the byte-level claim is checked somewhere
it can hold. That job is gated on the `RUN_R13_BYTE_EXACT` repository variable
and stays skipped until a runner with that label is registered; a red result
there has only two readings, a real regression or a log that no longer
describes its command, because the host has been held fixed. The lesson
generalises: a committed log is machine-local evidence, and only structure
travels between machines.

The four full-parameter rows (`pilot`, `hessian-64`, `lowrank-8`,
`lowrank-32`) carried a flag the first reconstruction of this table dropped:
`--start-scale legacy`. Under the shipped `prior` start the mode fit and the
Hessian/low-rank mass land on a different point (`final_potential` is
`1023.37` either way, but the probe base and the prior clip differ), giving
RMSE `0.130804` instead of `0.144261` for the pilot row and
`clipped_to_prior` `26,116` instead of `27,622` for the 64-probe row. The
`qk` and `hitrun` rows post-date the init change and reproduce without it; the
`temper` row does not — its cold chain sees the start point directly (below).

| gate (recorded numbers) | command | log |
|---|---|---|
| readout-conditional exact metric — R̂ 1.00935, ESS 3000.0, RMSE 0.105302 (`gate=true`) | `vector_regression --readout-only --kernel hmc --init mean --hmc-mass exact --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4` | `logs/pcnl_r13_gates/readout-exact.log` (re-verified 2026-11-24) |
| full-parameter pilot metric — R̂ 2.61047, ESS 4.7, RMSE 0.144261 | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass pilot --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 400 --chains 4 --start-jitter 1e-4 --thin 1 --draws 300` | `logs/pcnl_r13_gates/pilot.log` (re-verified 2026-11-24) |
| Hessian diagonal, 64 probes — R̂ 2.50388, ESS 4.8, RMSE 0.150189 | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass hessdiag --hessian-probes 64 --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/hessian-64.log` (re-verified 2026-11-24) |
| low-rank 8 — R̂ 2.39408, ESS 4.9 | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass lowrank --lowrank-rank 8 --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/lowrank-8.log` (re-verified 2026-11-24) |
| low-rank 32 — R̂ 2.01334, ESS 5.3 | `vector_regression --kernel hmc --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass lowrank --lowrank-rank 32 --pilot-steps 600 --pilot-rho 0.99999 --pilot-inflate 1 --hmc-leapfrog 4 --steps 1000 --burn 250 --thin 1 --draws 750 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/lowrank-32.log` (re-verified 2026-11-24) |
| low-rank 32 + trajectory jitter — R̂ 2.14626, ESS 5.1 | as low-rank 32 plus `--hmc-jitter` | `logs/pcnl_r13_gates/jitter-lf4.log` (re-verified 2026-11-24) |
| parallel tempering, 4 replicas — swaps 0/240, R̂ 2.44665, ESS 4.8, RMSE 0.168598 | `vector_regression --kernel hmc --temper 4 --init mode --mode-steps 3000 --mode-lr 5e-3 --start-scale legacy --hmc-mass hessdiag --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/temper-4.log` (re-verified: 0/240 swaps on all four chains; R̂ `2.44665`, ESS `4.8`, tail `10.0`, RMSE `0.168598`) |
| hit-and-run, 400 moves — R̂ 2.39717, ESS 4.9, RMSE 0.087048 | `vector_regression --kernel hitrun --init mode --mode-steps 8000 --mode-lr 5e-3 --steps 400 --hr-width 1 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/hitrun-400.log` (re-verified; burn is `steps/4` and is not a flag) |
| hit-and-run, 5,000 moves — R̂ 3.03445, ESS 4.5, RMSE 0.121815 | `vector_regression --kernel hitrun --init point --steps 5000 --hr-width 1 --thin 2 --draws 1875 --chains 4` | `logs/pcnl_r13_gates/hitrun-5000.log` (re-verified) |
| live-attention q/k gate — R̂ 2.60278, ESS 4.7, tail 9.3, RMSE 0.173727 | `vector_regression --kernel hmc --init mode --mode-steps 8000 --mode-lr 5e-3 --hmc-mass hessdiag --hmc-leapfrog 4 --steps 400 --thin 1 --draws 300 --chains 4 --start-jitter 1e-4` | `logs/pcnl_r13_gates/qk-gate.log` (re-verified) |
| multimodality basins — RMSE 0.0770/0.0871, potentials 28.09/38.71 | `vector_basins --runs 2 --steps 8000` | `logs/pcnl_r13_gates/basins.log` (re-verified: `28.0901`/`38.7083`, RMSE `0.076978`/`0.087130`) |
| continual stream — coverage90 0.983–1.000, z² 0.02–0.36 | `vector_stream --prefixes 4 --adam-steps 300 --lr 0.02` | `logs/pcnl_r13_gates/stream.log` (re-verified: warm coverage `1.0/1.0/0.9917/0.9833`, z² `0.032/0.024/0.275/0.356`) |
| CN exact-dense, 6,000 steps — R̂ 1.00616, ESS 620.9, tail 201.3, RMSE 0.105528 (`gate=true`) | `vector_regression --readout-only --kernel cn --proposal exact --init mean --rho 0.5 --refresh 0.01 --steps 6000 --burn 1500 --thin 1 --draws 4500 --chains 4` | `logs/pcnl_r13_gates/cn-exact-dense.log` (re-verified; readout-conditional, dim 34) |
| CN exact-dense, 12,000 steps — R̂ 1.00423, ESS 1720.5, tail 1107.3, RMSE 0.105396 (`gate=true`) | as above with `--steps 12000 --burn 3000 --draws 9000` | `logs/pcnl_r13_gates/cn-exact-dense-12k.log` (length-scaling probe, not a new gate) |
| CN exact-dense, 24,000 steps — R̂ 1.00274, ESS 3920.6, tail 2144.3, RMSE 0.105362 (`gate=true`) | as above with `--steps 24000 --burn 3000 --draws 21000` | `logs/pcnl_r13_gates/cn-exact-dense-24k.log` (length-scaling probe; ~3,100 s) |

The readout, pilot, Hessian, low-rank 8/32 (with and without jitter), hit-and-run
400/5,000, q/k gate, basins and stream rows were re-executed and match to the
last digit. Two rows are reconstructed from their log headers (steps, draws,
proposal, ladder all echoed there), and the original `/tmp` logs were cleared by
the system mid-session, so only the numbers in this table survive for those:

- **hit-and-run 400 moves — re-verified** once the command was completed with
  `--mode-lr 5e-3` (the default `2e-3` leaves the mode at potential `93.87`
  instead of `33.81`) and `--start-jitter 1e-4`; all four chain lines are
  byte-identical to the recorded log and R̂/ESS/RMSE match exactly.
- **live-attention q/k gate — recovered.** The missing flag was `--hmc-leapfrog
  4`: with it the run returns R̂ `2.60278`, ESS `4.7`, tail `9.3`, RMSE
  `0.173727` exactly. At the default leapfrog length the same mode fit gives R̂
  `1.76121`/ESS `5.9` (`--mode-lr 5e-3`) or `2.12514`/`5.2` (`--mode-lr
  2e-3`), which is why the earlier reconstruction missed it; both alternates
  are kept in `logs/pcnl_r13_gates/` as `qk-alt-lf8-lr*.log`.
- **parallel tempering — recovered, on the same missing flag.** All four chains
  accept `0/240` swaps with hot replicas ending at `7.3e3–1.8e5`, matching the
  qualitative claim, and the cold-chain summary missed only because the
  reconstruction ran under the `prior` start: with
  `--start-scale legacy --hmc-leapfrog 4` on the recorded 3,000-step `5e-3`
  mode fit the run returns R̂ `2.44665`, ESS `4.8`, tail `10.0`, RMSE
  `0.168598` exactly. Under `prior` the identical command gives
  `2.15397`/`5.1`/`0.096604`. With zero exchanges the cold replica never leaves
  λ=1, so its summary is a function of its own start — which is precisely why
  the start flag, not the ladder, decided the row. The legacy run prints ladder
  `0.01/0.0464/0.2154/1` (the `prior` one printed `0.02/0.0737/0.2714/1`), and
  the cold-chain numbers are identical either way for that reason; hot replicas
  end at `1.2e4–2.9e4` against a cold `4.2e3–4.8e3`, so the `ΔU ≈ 6e3–1e4`
  spacing argument above is unchanged.
- **basins and stream — re-verified exactly** (`28.0901`/`38.7083` with RMSE
  `0.076978`/`0.087130`; coverage `1.0/1.0/0.9917/0.9833` with z²
  `0.032/0.024/0.275/0.356`).
- **hit-and-run 5,000 moves — re-verified** (R̂ `3.03445`, ESS `4.5`, RMSE
  `0.121815`) with `--steps 5000 --thin 2 --draws 1875`, i.e. burn `1,250`
  derived from `steps/4`. Twelve times the moves of the 400-move gate buys ESS
  `4.5` against `4.9`: the plateau is not a move-count problem.
- **CN ESS is length-limited, not stuck** (`cn-exact-dense-12k` and `-24k`,
  kept as alternates). Doubling the chain of the 6,000-step CN row — `--steps
  12000 --burn 3000 --draws 9000`, same start, same proposal — gives ESS
  `1720.5` against `620.9` (2.8× for 2× the draws, so the earlier figure was
  also truncated by the estimator at 4,500 draws), tail ESS `1107.3` against
  `201.3`, R̂ `1.00423` against `1.00616`, and unchanged coverage counts. The
  24,000-step probe (`--draws 21000`, 18,000 draws per chain) continues the
  line: ESS `3920.6` (2.28× the 12k value for 2× the steps), tail ESS `2144.3`,
  R̂ `1.00274`, RMSE `0.105362`, `gate=true` — so the growth is not
  plateauing. The mixing caveat on this row is therefore finite-sample, not a
  failure to explore; the 90% interval coverage of `0.158` is still far from
  nominal and is not a promotion. The 24k run also exposed a memory defect
  (the example copied every expanded draw, doubling peak to ~25 GB); the
  scoring path now borrows the draws, so its peak is ~16 GB like the 12k row.
- **low-rank 32 + trajectory jitter — recovered**, on the same flag the other
  legacy rows needed: `--start-scale legacy`. With it the run returns R̂
  `2.14626`, ESS `5.1`, tail `10.0`, RMSE `0.092472` exactly. Without the flag
  the identical command gives R̂ `2.48617` (leapfrog 4) or `2.34143` (leapfrog
  8), which is why four earlier attempts missed it — `--hmc-jitter` draws one
  extra uniform per proposal, so a different start rescales *every* trajectory.
  The conclusion the row supports is unchanged: jitter does not close the gate.
- **CN exact-dense — re-executed and converging.** The reconstruction needed a
  flag the row omitted: `--proposal exact` with `--init mean` is refused
  outright without `--readout-only` ("the analytic posterior exists only for the
  readout conditional"), so the documented command as written is invalid with
  current code. With `--readout-only` the 6,000-step/1,500-burn run mixes well
  beyond the gate: R̂ `1.00616`, bulk ESS `620.9`, tail ESS `201.3`, RMSE
  `0.105528`, acceptance `2,316–2,553` of `6,000` per chain, and a large
  fraction of non-CN refreshes (`62–65` per chain). Its `sample_space` is
  `readout_conditional` (dim 34), not the full parameter, so it validates the
  kernel and the exact-dense proposal rather than the full-parameter gate. The
  tempering cold-chain summaries no longer stand open (above).

Logs for these re-runs are committed under `logs/pcnl_r13_gates/` (with the
exact command per file in its `README.md`); `.gitignore` carries the two
negation lines that keep that directory trackable under `logs/*`.
`scripts/check-pcnl-gates` re-runs every one of them and diffs the summary
lines against the committed log, so a log that stops matching (or a command
that drifts from its log) fails instead of going unnoticed.

### Low-rank-plus-diagonal HMC metric (2026-11-23)

`LowRankCovariance::from_draws` estimates `C = diag(v) + WΛWᵀ` from the pilot
(rank-`k` sample-covariance directions from a Gram eigendecomposition, diagonal
residual floored at `1e−6 × max var`). `--hmc-mass lowrank` uses `M = C⁻¹` via
Woodbury and draws `N(0,C⁻¹)` as `C⁻¹q` with `q ~ N(0,C)`; a unit test recovers
a correlated 3-D Gaussian. On the full target the metric is scaled
(`ε = 0.052–0.074`, acceptance 0.88–0.91) but mixing does not improve: rank 8
over 400 steps gave R̂ max `2.39408`, bulk ESS `4.9`; rank 32 over 1,000 steps
gave R̂ `2.01334`, ESS `5.3`. ESS did not grow with length, and `--hmc-jitter`
(trajectory length in `1..=2·leapfrog`, symmetric MH) gave R̂ `2.14626`, ESS
`5.1`, so resonance is not the cause. The limit is the pilot covariance: a
600-step random walk cannot estimate a 57,378-coordinate posterior width, so
the stiff directions stay unwhitened.

### Hessian-diagonal metric at the mode (2026-11-23)

`hessian_probe` forms `Hz` for Rademacher `z` by a central difference of the
sampled-potential gradient (`2` evaluations per probe) and estimates `diag(H)`
by Hutchinson; a unit test reproduces the analytic diagonal of a
Gaussian-plus-prior target. At full dimension 64 probes cost 2.4 s and the
estimate is stable across `h ∈ {1e-3, 1e-4}`, so it is not finite-difference
noise. **27,622 of 57,378** entries fall below the prior bound `1/c_i` (many
negative) and the raw diagonal spans `2.2e−7 .. 2.2e5`. Unclipped it freezes
`ε` at `1e-6` (R̂ `2.31799`, ESS `4.9`). Clamped to `[1/c_i, 1e4/c_i]` it
restores `ε ≈ 6.3–7.0e-3` and acceptance 0.77–0.82 but still gives R̂
`2.50388`, ESS `4.8`, RMSE `0.150189`, with the four chains' test predictions
(`−1.92/−1.61/−1.80/−1.93`) in different regions. The full-parameter target is
multimodal, not just ill-conditioned; no metric closes that gap. The four-chain
full-parameter gate is not cleared.

### Continual-stream pilot (2026-11-23)

`examples/vector_stream.rs` cuts the 240 training rows into four contiguous
60-row blocks (no document ids in the fixture, so row order is the proxy) and
trains three arms that share the objective, optimizer settings and prior:
`warm` continues Adam from the previous prefix's vector over all seen rows,
`joint` restarts from the fixed init on the same rows, `reset` restarts on the
new block only. Predictions are readout-conditional exact posterior means with
the backbone frozen. Test RMSE by prefix (warm/joint/reset): prefix 1 is
identical (`0.150742`); then `0.106920/0.112642/0.183135`,
`0.189312/0.118874/0.213221`, `0.079085/0.105293/0.168040`; means
`0.131515/0.121888/0.178784`. Retention at prefix 3: warm `0.00217/0.00337`,
joint `0.00429/0.00444`, reset `0.180/0.202`. The reset arm forgets, joint
retains, and warm keeps the best retention but is not uniformly better on test.
This is a stream comparison on synthetic targets, not a calibration or
promotion result.

The pilot now also reports 90% predictive intervals, normalized residuals and
per-block retention curves. Intervals are exact for the readout-conditional
Gaussian (`‖L⁻¹φ‖² + s²` with `φ` the 34-coordinate design vector).

| arm | coverage90 by prefix | mean `z²` by prefix | calibration 50/80/90/95% |
|---|---|---|---|
| warm | 1.000/1.000/0.992/0.983 | 0.032/0.024/0.275/0.356 | 0.938/0.990/0.994/0.994 |
| joint | 1.000/1.000/1.000/1.000 | 0.032/0.020/0.050/0.068 | 0.990/0.999/1.000/1.000 |
| reset | 1.000/1.000/0.988/1.000 | 0.032/0.107/0.164/0.055 | 0.954/0.996/0.997/1.000 |

A calibrated predictive has `E[z²] = 1`; these are `0.02–0.36`, so the
predictive sd is up to 7× too large and a nominal 50% interval covers 94–99%.
The width is dominated by the bias prior (sd 0.05), which is the same order as
the residual sd (0.07–0.15) — the intervals are a conservative bound, not a
calibrated uncertainty, and a single coverage number would have concealed that.
Retention curves (mean RMSE over blocks seen so far): warm
`0.002171/0.003385/0.002729`, joint `0.004285/0.004066/0.003995`, reset
`0.180109/0.207425/0.283135` — the reset arm's forgetting grows with each block.

### Multimodality diagnostic (2026-11-23)

`examples/vector_basins.rs` runs four independent mode fits and compares them
segment by segment. Potentials `36.39/36.54/97.91/79.31` with test RMSE
`0.0737/0.0696/0.0719/0.0797`; all six pairs sit at relative parameter
distance `1.92–2.18` (uncorrelated), so these are distinct solutions of similar
predictive quality — the ESS plateau is a property of the target. The worst
diverging segment differs per pair (`readout`, `block15`, `block10`, `block5`),
so no single block is responsible — but note the diagnostic originally labeled
*matrices* as blocks (`block1…block24`); it now labels one segment per
transformer block.

### Inert q/k projections — found and fixed (2026-11-23)

Every block's `wq`/`wk` gradient at the legacy `init_std=0.02` was
`7e-2…2e-1` against `4e1…7e1` for `wv`/`wo`/`w1`/`w2`, and those projections
stayed bit-identical across independent fits. `0.02` is 10× below the prior sd
the target assumes (`prior_sigma·√fan_in = 0.2` for 16×16), so logits are ≈0
and `∂logits/∂W_q ∝ W_k x` is second order; the prior gradient (`0.5`) beats
the data gradient (`0.1`) and pulls `W_q, W_k` to `0`, the fixed point where
the softmax is exactly uniform and the q/k gradient is exactly zero. Attention
was contributing a constant in the "four-block" model.

`init_flat_prior` now draws θ at the prior's own sd (`--start-scale prior`,
`--init-mode prior`); `3.1e3` vs `0.17` q-gradient, and the projections move.
The mode fit needs 8,000 steps from such a start: potentials `28.09/38.71`,
RMSE `0.0770/0.0871`, blocks separated by `1.99–2.21`. At 2,000 steps it is not
converged (`2.3e3–3.0e3`, RMSE `0.19–0.23`).

**The gate is unchanged by the fix:** with live attention and the 8,000-step
mode fit, R̂ `2.60278`, bulk ESS `4.7`, tail ESS `9.3`, RMSE `0.173727` — still
failing. Dead attention was a real defect but not the cause of the plateau.

### Parallel tempering (2026-11-23)

`run_parallel_tempering` samples `exp(−βU)` per replica and swaps adjacent
states with `min(1, exp((β_i−β_j)(U_i−U_j)))`; a 3-D Gaussian unit test
recovers the cold marginal. On the full target (4 replicas, ladder
`0.02/0.0737/0.2714/1`, swap every 5 steps, 400 steps, 4 chains) only **0–1 of
240** swaps were accepted per chain, and the cold chain is unchanged: R̂
`2.44665`, ESS `4.8`, RMSE `0.168598`. Adjacent replicas differ by
`ΔU ≈ 6e3–1e4`, so an exchange-rate-preserving ladder needs rung spacing near
`1e-4` (tens of thousands of replicas); tempering is not viable at this energy
scale.

### Hit-and-run slice moves (2026-11-23)

`run_hit_and_run` implements a global move that ignores gradient scale: a
random direction, unit in prior-whitened coordinates, with the exact
conditional along that ray sampled by Neal's stepping-out/shrinkage slice
sampler. A 3-D Gaussian unit test recovers the target covariance. On the full
target (400 moves/chain, burn 100, 4 chains, width 1): every move lands
(`400/400`, mean step `0.40` prior sd) for ≈2,440 potential evaluations per
chain — about the cost of the 400-step HMC runs. Result: R̂ `2.39717`, bulk ESS
`4.9`, tail ESS `10.0`, **gate not closed**. It has the best draw-mean test
RMSE so far (`0.087048`) but its intervals under-cover (0.3375 at 90%), which
is the honest signature of an unequilibrated chain: each slice move is exact,
but 400 moves of 0.4 prior sd is a short walk.

A 12.5× budget (5,000 moves/chain, burn 1,250, thin 2, 1,875 draws/chain,
≈34–35k potential evaluations/chain, ≈44 min) does not move the gate: R̂
`3.03445` (per-observable `1.75–3.03`), bulk ESS `4.5`, tail ESS `10.0`,
`gate=false`. ESS stays flat while the draw pool grows 6.25× (`4.9` at 300
draws/chain → `4.5` at 1,875), so the chains are confined to different regions
rather than mixing slowly — a geometry result, not a budget shortfall.

The longer run also corrects the shorter reading: draw-mean test RMSE
`0.121815`, predictive-90 coverage `0.9333`, parameter-90 coverage `0.8125`. The
400-move run's RMSE `0.087` and coverage `0.34` were short-walk artefacts and
are not an improvement.

### Live/fixture encoder parity (2026-11-22)

`scripts/pcnl_ultrabert_adapter.py` now owns the pinned model revision, 96-token
truncation, CLS pooling and L2 normalization shared by both fixture generators
and `scripts/pcnl-ultrabert-parity`.  The parity script reconstructs the fixture
text for a requested row (or accepts `--text`), encodes it live and reports
max-abs, cosine and L2-norm differences against the frozen vector.  In this
environment it exits `2` with `SKIP: live encoder unavailable (missing torch,
transformers, safetensors)`, so parity is **not** claimed as passed.  The R13
head consumes the same vector in both paths; prediction parity still needs a
run where the encoder dependencies and the pinned weights are present.

The continual stream remains open. The synthetic-target and warm-started
limits are part of this result, not footnotes.

```bash
cargo run --offline --release -p pcnl --example vector_regression -- \
  --readout-only --init mean --kernel hmc --hmc-mass exact \
  --hmc-leapfrog 4 --steps 1000 --thin 1 --draws 750 --chains 4

cargo run --offline --release -p pcnl --example vector_regression -- \
  --readout-only --init mean --proposal exact --kernel cn --rho 0.5 \
  --steps 6000 --thin 1 --draws 4500 --chains 4

cargo run --offline --release -p pcnl --example cn_dimension_sweep -- \
  --steps 4000 --dims 2,6,18,34,66,130,256 --rhos 0.99 --exact-proposal

cargo run --offline --release -p pcnl --example known_target
python3 scripts/pcnl-ultrabert-parity --row 0
```

## R14 — satisficing pCNL, SWA, and distillation (executed 2026-11-27)

**What was executed here.** R14's P0–P3 mechanics, the P1 known-target
quadrature gate, *reduced-budget* P4/P5 pilots, a three-seed P4 campaign at
the declared 57 378-parameter architecture, and a three-seed P5 RL campaign
with paired return intervals (both reduced rows/chains), the same P5 protocol on
reduced-budget RLCD/Edge0 checkpoints, a return-aware satisficing variant for
hit-and-run, plus a serving-cost benchmark of the distilled student against the
three-member ensemble.
A geometry-aware hit-and-run tier was added and measured against the stalled
pCN drift.  The P6 gated extensions (heterogeneous configurations, Git Re-Basin,
repulsion), full-split/serving runs and the Lean S0–S6 parity work were **not**
executed.  P6 remains gated; no Lean packet is claimed.  Logs:
`logs/pcnl_r14_gates/`.  Tests: `pcnl/tests/r14.rs` (18 checks).

### P0/P2/P3 module checks (deterministic)

`cargo test -p pcnl --test r14` — 18/18 pass:

* finite-difference gradient agreement for both hinges inside, near and
  outside the plateau (softplus and `C¹` Huberized; the Huberized
  `α = 0` hinge gradient is exactly zero inside);
* target identity changes with `τ, κ, α, δ`, hinge, loss label/reduction and
  prior covariance;
* `SatisficingPotential` reconstructs the same `Φ` from the mean-loss path
  (`mean_loss · λn = Φ(L)`), non-finite energies fail explicitly;
* online SWA equals batch averaging bitwise; ensemble + SWA checkpoints
  round-trip bitwise; convex segments have no barrier and the double well has
  a positive one;
* knee rule detects a 0.5 %-per-check slowdown and ignores fast descent and
  oscillation; warm entry stops at the declared ceiling;
* permutation-copy ensembles have zero functional variance and effective rank
  `1.0` despite nonzero weight distance; the squared-error ambiguity identity,
  Jensen–Shannon nonnegativity, PAC-Bayes bounds, coverage/`z²` and the
  Pinsker bound hold;
* KL-SGD distillation reduces teacher KL; the spread head recovers a known
  teacher spread;
* the RL loss is nonnegative and matches finite differences; policy diversity
  is zero for one policy and positive across members.

### P1 known-target plateau-vs-spike gate — **PASS**

`cargo run --release -p pcnl --example r14_plateau_quadrature`
(log `logs/pcnl_r14_gates/plateau_quadrature.log`).  Four dispersed chains,
20 000 steps (5 000 burn, thin 2, 7 500 draws/chain), `ρ = 0.9`, prior
`N(0,1)`, Huberized hinge, `τ = 0.5`, `κ = 20`, `α = 0`, `δ = 0.05`.

| `T` | accept | R̂ (θ / broad / spike) | bulk ESS (θ / broad / spike) | broad mass (ref) | spike mass (ref) |
|---|---|---|---|---|---|
| 0.5 | 78 474/80 000 | 1.00059 / 1.00029 / 1.00011 | 4235.7 / 16852.9 / 25030.8 | 0.587433 (0.585272) | 0.079800 (0.081564) |
| 1.0 | 78 477/80 000 | 1.00062 / 1.00017 / 1.00009 | 3981.8 / 16214.3 / 25052.2 | 0.588133 (0.584141) | 0.082167 (0.081407) |
| 2.0 | 78 644/80 000 | 1.00058 / 1.00012 / 0.99994 | 4023.9 / 16101.4 / 24661.7 | 0.580467 (0.582863) | 0.080733 (0.081228) |

All three temperatures pass `R̂ < 1.01`, bulk ESS `≥ 400`, and mass/mean/`E[θ²]`
agreement within 3 MCSE.  The controls invert the ranking as predicted:
sharp Gibbs (`λ = 50`) gives broad 0.000000 / spike 0.999975, while radical
Adam (4 000 steps) lands at `θ = 0` with `L = 0` — the narrow spike.  SWA on a
convex plateau has loss `0.001111` versus the snapshot mean `0.043333` and a
zero convex barrier; the unaligned two-basin path has barrier `1.000000`.

### P4 supervised pilot (reduced budget) — negative contrasts retained

`cargo run --release -p pcnl --example r14_supervised_campaign`
(log `logs/pcnl_r14_gates/supervised_campaign.log`).  Frozen 20 Newsgroups
embeddings, 96 train / 48 test rows, `w2_ff4_l1_t2` head (3 110 parameters),
`s = 0.05`, `τ = 0.75`, `κ = 20`, `α = 0.05`, Huberized hinge, `M = 4`,
≈2–3 s wall clock.  Test RMSE (ensemble mean):

| arm | RMSE | NLL | 90 % cov. | `z²` | in-region | eff. rank | disagreement |
|---|---|---|---|---|---|---|---|
| radical single | 0.344692 | 20.77 | 0.250 | 47.53 | 0.000 | 1.00 | 0 |
| radical deep | 0.206873 | −1.30 | 0.927 | 1.04 | 0.000 | 2.804 | 0.256 |
| knee-stopped | 0.210869 | −1.27 | 0.833 | 1.27 | 1.000 | 2.567 | 0.229 |
| SGD-SWA single | 0.268738 | 11.45 | 0.406 | 28.89 | 1.000 | 1.00 | 0 |
| sharp Gibbs pCNL | — | — | — | — | — | — | accept 0/900 |
| satisficing pCNL | 0.210869 | −1.27 | 0.833 | 1.27 | 1.000 | 2.567 | 0.229 |
| prior-only | 1.164893 | 57.34 | 0.000 | 118.98 | 0.000 | 2.515 | 0.156 |
| distilled student | 0.209682 | 5.80 | 0.385 | 17.59 | 0.000 | 1.00 | 0 |
| label control | 0.194120 | 4.54 | 0.438 | 15.07 | 0.000 | 1.00 | 0 |

**Measured curved-boundary stall.**  Satisficing pCNL accepted **0/1200**
moves; the same kernel at proposal scales `1e-6` and `1e-8` accepted `0/300`
and moved `0.000000e0`.  A random proposal of per-coordinate sd
`√(0.19·10⁻⁸ c_i)` moves `Φ` from `0.035` to `1808.9` (`L` from `0.71` to
`≈13.4`): the feasible set is a thin shell in weight space (warm norm `78.672`
versus the prior draw `77.912`, but 10 % radial shrinkage gives `Φ ≈ 1808.9`).
The satisficing arm is therefore identical to the knee-stopped arm — a
negative/inconclusive contrast, not a gain.  SWA weight-vs-last-iterate RMSE
delta `−0.000300` (output mean `−0.000171`), barrier `0.000000`; the
ensemble/SWA checkpoint round-trip is bitwise exact.

**Distillation contrast is negative:** the ensemble-distilled student is
*worse* than the equal-budget label-trained control by `+0.012193 ± 0.002330`
(paired absolute error over 96 test coordinates); teacher–student RMSE
`0.185324`, spread-head 90 % coverage `0.3854`.  Reported verbatim; this pilot
does not promote the distilled student.

**Geometry-aware follow-up (same target, same probes).**  Exact hit-and-run
slice moves (random prior-whitened direction, exact 1-D conditional, no
gradient-scale knob) also move `300/300` and `moved_norm = 24.36`, but at
`κ = 20` the retained state leaves the plateau (`L = 2.169`): the stationary
target's mass sits on the `L ≈ τ` boundary layer, not inside the set.  Raising
the hinge stiffness narrows that layer: `κ = 2e3` gives `L = 0.850` (0/20 draws
in region), `κ = 2e5` gives `L = 0.756` (8/20), and `κ = 2e6` gives
`L = 0.7496` with 16/20 retained draws inside `L ≤ τ`.  The kernel stays exact;
this is a target-confinement choice, not a corrected MH ratio.

### P4 multi-seed, declared 57 378-parameter architecture

`cargo run --offline --release -p pcnl --example r14_supervised_multiseed`
(log `logs/pcnl_r14_gates/supervised_multiseed.log`).  `VectorShapes::new(768,
2)` = **57 378 parameters**, 120 train / 60 test rows, three seed families,
`M = 3`, `τ = 0.75`, Huberized hinge, ≈46–50 s/seed.  All trained arms are
in-region by construction; the discriminator is held-out error and diversity.

| arm (mean ± SE across 3 seeds) | RMSE | 90 % cov. | eff. rank | disagreement |
|---|---|---|---|---|
| radical single | 0.488945 ± 0.064015 | 0.178 | 1.000 | 0 |
| radical deep | 0.259061 ± 0.021784 | 0.881 | 1.828 | 0.482 |
| knee-stopped | **0.118174 ± 0.010378** | 0.869 | 1.850 | 0.182 |
| SGD-SWA | 0.217582 ± 0.031446 | 0.333 | 1.000 | 0 |
| sharp Gibbs | 0.217783 ± 0.028169 | 0.333 | 1.000 | 0 |
| satisficing pCNL (`κ = 20`) | 0.118174 ± 0.010378 | 0.869 | 1.850 | 0.182 |
| satisficing hit-and-run (`κ = 2e6`) | 0.120144 ± 0.008751 | 0.883 | 1.873 | 0.185 |
| distilled student | 0.453951 ± 0.001057 | 0.175 | 1.000 | 0 |
| label control | 0.450989 ± 0.000000 | 0.167 | 1.000 | 0 |
| prior-only | 1.199430 ± 0.184088 | 0.803 | 1.855 | 1.979 |

Movement, measured at full size: satisficing pCNL accepted **0/900** in every
seed (the stall persists); the hit-and-run arm accepted **240/240** per seed
and retained **60/60, 60/60, 59/60** in-region draws.  Its paired test-row
contrast against the knee ensemble is `+0.001005 ± 0.002637` across seeds —
statistically indistinguishable, so the geometry-aware tier matches knee
quality while actually moving.  The distilled student is again worse than the
label control: paired `+0.002670 ± 0.001152` (≈ 2.3 SE).  The radical single
run memorizes (`L_train → 0`) and loses badly on held-out rows
(`0.488945`), and sharp Gibbs accepts `0/200` per seed.

### P5 RL pilot (reduced budget) — non-RLCD

`cargo run --release -p pcnl --example r14_rl_campaign`
(log `logs/pcnl_r14_gates/rl_campaign.log`).  Synthetic MDP (`horizon 8`,
5 actions), 48 train / 32 dev / 32 test transitions, advantage-weighted
behaviour-regularized policy NLL (`β = 0.05`, temperature 20), `τ = 1.2`,
`κ = 200`, Huberized hinge, 15 policy parameters, `M = 4`.

| arm | in-region | diversity TV | nominal return | shifted return |
|---|---|---|---|---|
| radical single | 1.000 | 0.0000 | −3.9773 | −5.8599 |
| radical deep | 1.000 | 0.0339 | −4.0301 | −5.8783 |
| knee-stopped | 1.000 | 0.1624 | −6.0736 | −8.4171 |
| satisficing pCNL | 0.000 | 0.4397 | −8.0601 | −11.5046 |
| SGD-SWA | 1.000 | — | −8.4859 | −12.9932 |
| prior-only | 0.000 | 0.5699 | −11.1771 | −15.8711 |
| distilled student | — | — | −7.8173 | −10.9763 |
| label control | — | — | −4.0961 | −5.8779 |

Satisficing pCNL accepted **830/1200** moves and spread functionally
(diversity TV `0.4397`), but its retained configurations sit at the boundary
shell `L = 1.202–1.310`, marginally above `τ = 1.2`, so the strict in-region
fraction is `0/4`.  This is the boundary-mass consequence of a Huberized hinge
inside the plateau, not an implementation failure, and is retained here.

Teacher–student fidelity to the satisficing ensemble: KL `0.014752`, TV
`0.057720`, max-state KL `0.100389`; Pinsker/finite-horizon bound `80.65`
(`control`: KL `6.4528`, TV `0.6058`, bound `1002.37`).  Once-per-episode
member draw: `−8.5811 ± 0.7567`; per-step BMA: `−8.0601`.  Both the
satisficing ensemble and the distilled student trail the radical and
label-trained arms on nominal and shifted return; the RL distribution shift
lowers every arm.

### P5 multi-seed RL campaign (3 seeds, paired intervals)

`cargo run --offline --release -p pcnl --example r14_rl_multiseed`
(log `logs/pcnl_r14_gates/rl_multiseed.log`, ≈0.3 s).  Same MDP and 48/32/32
transition split, now three seed families × `M = 3`, with a `κ = 2e6`
hit-and-run tier and common-random-number rollouts (`200` episodes per arm,
episode seed `909`) so contrasts are paired episode by episode and then
aggregated across seeds.

| arm (mean over 3 seeds) | in-region | div. TV | nominal return | shifted return |
|---|---|---|---|---|
| radical single | 1.000 | 0.000 | **−4.0546** | **−5.8890** |
| radical deep | 1.000 | 0.055 | −4.0189 | −5.8895 |
| knee-stopped | 1.000 | 0.222 | −6.0200 | −8.6996 |
| SGD-SWA | 1.000 | 0.000 | −4.5675 | −6.4598 |
| sharp Gibbs | 1.000 | 0.000 | −6.0889 | −8.4697 |
| satisficing pCNL (`κ = 200`) | 0.222 | 0.399 | −9.0278 | −13.9298 |
| satisficing hit-and-run (`κ = 2e6`) | 1.000 | 0.311 | −8.4250 | −12.5278 |
| prior-only | 0.000 | 0.609 | −12.3585 | −20.1591 |
| distilled student | — | — | −8.3645 | −12.2708 |
| label control | — | — | −4.0961 | −5.8779 |

Movement: pCNL accepts `0.654` of moves but only `0.222` of retained
configurations are in-region; hit-and-run accepts `1.000` and retains
`0.948` in-region draws (`256/270`).  Sharp Gibbs accepts `2/600`.  Every
knee warm entry stops at the same `ceiling` label in all three seeds.

Paired contrasts (per-episode deltas, mean ± SE across seeds):

| contrast | nominal | shifted |
|---|---|---|
| hit-and-run − knee | −2.4050 ± 0.9367 | −3.8283 ± 1.0535 |
| pCNL − knee | −3.0079 ± 0.3110 | −5.2302 ± 0.7311 |
| distilled − label | −4.2685 ± 0.7293 | −6.3929 ± 0.8073 |
| hit-and-run − radical deep | −4.4061 ± 0.6874 | −6.6383 ± 0.6786 |

The geometry-aware tier fixes the in-region failure but does **not** fix the
return: it is `2.4 ± 0.9` worse than the knee ensemble and far from the
radical/label arms.  Distillation faithfully reproduces its teacher (KL
`0.008635`, TV `0.039624`, Pinsker bound `55.58`) and therefore inherits the
satisficing teacher's return deficit rather than closing it.  The once-per-
episode member draw is `−8.4099 ± 0.6909` versus per-step BMA `−8.4250`.

### P5 on RLCD/Edge0 checkpoints (reduced budget)

`cargo run --offline --release -p pcnl --example r14_rl_edge0_multiseed`
(log `logs/pcnl_r14_gates/rl_edge0_multiseed.log`, ≈34 s).  The synthetic
48-transition corpus is replaced by the Edge0 world model: three seed
families × three members, each member a real `edge0_campaign` RLCD checkpoint
written to `target/r14_edge0_checkpoints/*.safetensors` and reloaded before
use.  Reduced split `[24, 8, 8, 16]` worlds, 60 updates, 30 warm-start
updates, 1684 parameters, `τ = 1.05`; the pCNL/hit-and-run chains run on the
Edge0 training NLL through an `EnergyTarget` adapter.

| arm (3 families) | test Brier | calibration Brier | in-region | div. TV | eff. rank |
|---|---|---|---|---|---|
| RLCD single | 0.14905 ± 0.00354 | 0.16933 | 1.000 | 0.000 | 1.000 |
| RLCD ensemble (`M = 3`) | 0.14629 ± 0.00086 | 0.16791 | 1.000 | 0.0364 | 1.086 |
| supervised ensemble | **0.14628 ± 0.00010** | **0.16508** | 1.000 | 0.0050 | 1.495 |
| satisficing pCNL | 0.19051 ± 0.01131 | 0.22146 | **0.000** | 0.3349 | 1.798 |
| satisficing hit-and-run | 0.14630 ± 0.00067 | 0.16926 | 1.000 | 0.0373 | 1.188 |
| distilled student (no-ul) | 0.15638 ± 0.00043 | — | — | — | — |

The paired, world-clustered contrasts are: RLCD ensemble − single
`−0.00275 ± 0.00399` (not significant), hit-and-run − ensemble
`+0.00001 ± 0.00020` (indistinguishable), and pCNL − ensemble
`+0.04422 ± 0.01068` (pCNL is worse).  pCNL accepted only `3/90` moves and its
retained NLL (`1.18–1.27`) left the `τ = 1.05` region in all three families;
hit-and-run accepted `24/24` with `6/6` in-region draws.  This is the same
region-retention pattern seen on the synthetic corpus, now on the Edge0 NLL.

The KL-distilled linear student uses eight **teacher-independent** surface
features (state-token buckets/sum, question-token sum, option count, outcome
count), not the Edge0 backbone, so it is a standalone readout rather than a
copy of the teacher's last layer.  On the 16 no-ul test questions it matches
the ensemble teacher: Brier `0.15638` vs `0.15637`, paired delta
`+0.00000 ± 0.00000`, KL `≈0.00000`, TV `0.00002`.  The no-ul subset is
surface-predictable for this tiny checkpoint; this is not evidence that the
student compresses the full Edge0 model, and no serving claim follows.

Limits: reduced R9 split, not the declared 2048/256/256/512-world campaign;
linear student, not a second Edge0; checkpoint files live under `target/`.

### Return-aware satisficing for the hit-and-run tier

`cargo run --offline --release -p pcnl --example r14_return_aware`
(log `logs/pcnl_r14_gates/return_aware.log`).  The plain hit-and-run tier kept
the loss region but lost `2.41 ± 0.94` nominal return against the knee
ensemble.  This campaign adds a differentiable one-step return floor to the
same potential:

```text
Φ(θ) = κ ψ(L − τ) + α L + κ_R ψ(R_floor − R̂(θ)),  ψ(u)=½ max(0,u)²
R̂(θ) = mean_s Σ_a π_θ(a|s) r(s,a)
```

`R_floor` is the knee ensemble's surrogate plus `0.03`; the return arm uses
`τ = 1.10` (a `0.10` margin below the evaluated `τ = 1.2`) so the return push
has room inside the region, `κ_R = 15 000`, `κ = 2e6`, three seeds × three
members.  This is a changed target, not the plain satisficing potential, and
is labelled as such.

| arm | in-region | surrogate | div. TV | nominal return | shifted return |
|---|---|---|---|---|---|
| knee-stopped | 1.000 | −0.17298 | 0.222 | **−6.0200** | **−8.6996** |
| satisficing hit-and-run (plain) | 1.000 | −0.17348 | 0.334 | −8.2581 | −12.1273 |
| **hit-and-run + return floor** | 1.000 | −0.16997 | 0.174 | **−6.0169** | **−8.6475** |
| pCNL + return floor | 1.000 | −0.17298 | 0.222 | −6.0200 | −8.6996 |

Paired per-episode contrasts for the return-aware hit-and-run tier:
`+2.2412 ± 0.7961` nominal and `+3.4798 ± 0.9910` shifted against the plain
tier, and `+0.0031 ± 0.3472` nominal against the knee ensemble — i.e. the
return deficit is closed while every member stays in `L ≤ 1.2`.  The cost is a
loss of functional diversity (`0.334 → 0.174`): the return tilt collapses the
members toward the same high-reward region.  pCNL on this target accepted
`0/900` proposals and stayed at its knee starts, so the viable sampler for
the return-aware objective is hit-and-run, not pCNL.

### Serving cost: distilled student vs `M = 3` ensemble

`cargo run --offline --release -p pcnl --example r14_serving_bench`
(log `logs/pcnl_r14_gates/serving_bench.log`).  Single-thread CPU on this host,
per-query wall time with a counting global allocator.  Latency is
machine-dependent and is recorded, **not** replay-gated; the parameter counts
and model bytes are exact.

P4 (57 378 parameters per member; the distilled student keeps the same
architecture, so this is not a compressed student):

| path | mean | p50 | p90 | p99 | qps | alloc/query | model bytes |
|---|---|---|---|---|---|---|---|
| member forward | 104.3 µs | 103.6 µs | 107.2 µs | 110.8 µs | 9 586 | 130 448 B | 229 512 |
| ensemble forward (3) | 312.5 µs | 310.8 µs | 316.6 µs | 320.8 µs | 3 200 | 391 296 B | 688 536 |
| student forward | 103.9 µs | 103.3 µs | 106.7 µs | 110.4 µs | 9 624 | 130 448 B | 229 512 |
| ensemble `predict` (3) | 326.4 µs | 324.6 µs | 330.8 µs | 337.0 µs | 3 063 | 1 090 584 B | 688 536 |
| student `predict` | 107.4 µs | 106.8 µs | 110.6 µs | 113.9 µs | 9 308 | 363 544 B | 229 512 |

Memory ratio is exactly `3.000` (688 536 / 229 512; allocator delta
688 608 B including the `Vec` headers), and the student is `3.04×` faster
end-to-end.  The flat-slice `predict` entry point clones the parameter vector
per call, which is why it allocates ~3× the forward path; the ensemble pays
that clone three times.

P5 (15 `f32` parameters per policy member, 12 `f64` parameters in the
softmax student):

| path | mean | p50 | p90 | p99 | qps | alloc/query | model bytes |
|---|---|---|---|---|---|---|---|
| ensemble probs (3) | 0.333 µs | 0.313 µs | 0.416 µs | 0.417 µs | 2 690 887 | 360 B | 180 |
| student probs | 0.104 µs | 0.104 µs | 0.156 µs | 0.157 µs | 7 008 065 | 88 B | 160 |

Here the student is only `12.5 %` smaller than the three-member ensemble
(160 vs 180 bytes — the members are compact `f32` flat policies, the student
stores `f64` weights plus a bias) but is `3.2×` faster.  At these sizes the
latency, not the bytes, is the serving argument.

### P6 — written-review gate and scoped heterogeneity evaluation

P6 is gated by [`PCNL_P6_REVIEW.md`](PCNL_P6_REVIEW.md) and the executable
`scripts/check-pcnl-p6-gate`.  The review's decision is
`conditional-approve-heterogeneity-only` with `human_signoff = pending`, so
**promotion remains blocked** until the full declared P1–P5 campaigns exist
and a human signs the review.  The review excludes Git Re-Basin (no weight
merging tested), functional repulsion (no measured collapse: P4 rank
1.85–1.87, P5 diversity TV 0.31–0.40), and depth/head/normalization changes
(outside the frozen released architecture).

`cargo run --offline --release -p pcnl --example r14_heterogeneous`
(log `logs/pcnl_r14_gates/heterogeneous.log`, ≈284 s) compares homogeneous
widths `16/16/16` (172 134 parameters) against heterogeneous `12/17/19`
(172 550, +0.24 %) at 3 seeds × 3 members × 600 Adam steps, `τ = 0.75`:

| endpoint (mean ± SE over 3 seeds) | homogeneous | heterogeneous |
|---|---|---|
| test RMSE | 0.105138 ± 0.001424 | 0.103106 ± 0.007550 |
| in-region members | 1.000 | **0.778 ± 0.111** |
| effective rank | 1.949 ± 0.010 | 1.812 ± 0.079 |
| teacher–student fidelity | 0.160328 ± 0.003246 | 0.151690 ± 0.006750 |
| student test RMSE | 0.147829 ± 0.007265 | 0.147788 ± 0.007353 |
| ensemble forward latency | 193.04 ± 1.31 µs | 199.83 ± 1.26 µs |

Paired heterogeneous − homogeneous: teacher absolute error
`+0.091774 ± 0.004751`, student `+0.005889 ± 0.002932`, RMSE
`−0.002032 ± 0.008942`.  Mean accuracy is statistically indistinguishable,
but the heterogeneous arm misses the in-region and rank criteria: under the
shared budget the width-19 member diverges in seeds 82 (`L = 24.96`) and 93
(`L = 11.01`), and the width-17 member diverges in seed 93.  With per-width
retuning forbidden inside this review, the verdict is **not supported**, and
the arm costs ~3.5 % latency.  This is recorded as a negative P6 result; no
promotion follows.

### Not closed by this execution

* P4/P5 as declared: the multi-seed runs use 120/60 rows and short chains,
  not the full 20 Newsgroups split; ID/OOD/continual splits are still
  unmeasured.  Serving latency/memory is measured on one CPU host and is not a
  portable or accelerated-hardware number.  P5 now also runs on reduced-budget
  RLCD/Edge0 checkpoints, but not on the declared R9 2048/256/256/512-world
  split, and its student is a linear readout, not a compressed Edge0 model.
* P6 — the written-review gate and the scoped width-heterogeneity evaluation
  are executed (negative result); Git Re-Basin and functional repulsion are
  still excluded, and promotion is blocked pending full P1–P5 campaigns plus
  human sign-off.
* Lean S0–S6 parity (`timepiece/pcnl.md`) — no Lean compilation was run.
* R14 CI-gate wiring — `scripts/check-pcnl-r14-gates` replays four rows
  (`--check-manifest`, `--band`, `--structure`) and is wired into
  `.github/workflows/ci.yml` (per push) and `pcnl-gates.yml` (scheduled).

```bash
cargo test --release -p pcnl --test r14
./scripts/check-pcnl-r14-gates --check-manifest
./scripts/check-pcnl-r14-gates
cargo run --offline --release -p pcnl --example r14_plateau_quadrature
cargo run --offline --release -p pcnl --example r14_supervised_campaign
cargo run --offline --release -p pcnl --example r14_supervised_multiseed
cargo run --offline --release -p pcnl --example r14_rl_campaign
cargo run --offline --release -p pcnl --example r14_rl_multiseed
cargo run --offline --release -p pcnl --example r14_rl_edge0_multiseed
cargo run --offline --release -p pcnl --example r14_return_aware
cargo run --offline --release -p pcnl --example r14_serving_bench
cargo run --offline --release -p pcnl --example r14_heterogeneous
./scripts/check-pcnl-p6-gate --check
```
