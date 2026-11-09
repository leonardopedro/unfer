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
| 1.0e-2 | 2e-3 | 0.6–0.8 % | 28.12 / 1.7563 | 25.86–29.95 | " | 1.525 / 1.505 | E 1.753, 95 % [1.213, 2.623] |
| 3.0e-2 | 3e-3 | 1.8–2.5 % | 23.83 / 1.7803 | 20.97–25.21 | " | 3.389 / 3.606 | E 2.022, 95–99 % |
| 9.0e-2 | 1e-2 | 1.6–1.8 % | 21.09 / 1.8780 | 12.32–22.49 | " | 2.115 / 2.548 | E 1.988, 95 % [1.279, 3.112] |

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

**(d) First transformer cells.**  `P = 4240`, `θ_ref` = Task-A momentum pretrain
(loss 3.45 → 1.33), anchor on B: **7.03 % / NLL 3.4220** (chance is 11.1 % — the anchor is
*below* chance, see (e)).  `λ = 1`, `σ = 0.35`, `β = 0.05`, 4 chains × 4 000 steps
(burn 500, thin 5; 11 200 draws), `N_train = 512`, `N_test = 256`:

| rank | basis | accept | ESS(Φ) | `R̂(Φ)` | ESS(loss) | `R̂(loss)` | BMA acc | NLL | ECE | MCSE(NLL) | s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random | 94.2 % | 13 | 1.338 | 6 | 1.731 | 7.03 % | 3.4019 | 0.283 | 0.0036 | 116 |
| 1 | informed | 85.0 % | 102 | 1.034 | 16 | 1.141 | 20.31 % | 2.6233 | 0.166 | 0.0023 | 101 |
| 2 | random | 94.6 % | 8 | 1.465 | 11 | 1.328 | 7.03 % | 3.4182 | 0.284 | 0.0005 | 102 |
| 2 | informed | 78.3 % | 474 | 1.006 | 100 | 1.031 | 18.36 % | 2.6086 | 0.145 | 0.0006 | 102 |

The informed basis is PCA of 32 Task-B momentum steps (lr 0.05) — **data reuse**, so its
held-out numbers are optimistic by construction; the random control is what makes the
comparison readable, and at both ranks the random basis is indistinguishable from the
anchor (NLL 3.402 / 3.418 vs 3.422, accuracy identical), which is the control's job.
The informed basis buys 0.80 nat and +13.3 accuracy points at rank 1, +11.3 at rank 2 —
not monotone in rank, and at 0.39 % accuracy granularity (256 test examples) a 3-point
difference is 8 examples.

Mixing is the limit, and it is reported rather than hidden: rank-2 informed is the only
cell that is converged (`R̂(Φ) = 1.006`, ESS 474 of 11 200); rank-1 informed is
`R̂ = 1.034` with ESS 102; both random arms are **not converged** (`R̂` 1.34–1.47).  The
random arms' non-convergence is itself expected — `Φ` is nearly flat in random directions,
so a chain there wanders in the prior and its potential's ESS is small even at 95 %
acceptance — which is why the table prints ESS of `Φ` *and* of the held-out loss, and why
the run's BMA is quoted with its MCSE.  `β = 0.05` is not a tuned value: at `β = 0.5` the
informed direction accepts 2.8–5.2 %, and at `β = 0.05` it accepts 78–85 %, which is the
pCN scale-versus-target-width problem R2 exists to address (informed proposals or a
justified per-direction metric).

**(e) Comparability caveat, stated.**  The anchor used here is below chance on B (7.03 % vs
11.1 %), because 1 500 momentum steps on 512 Task-A examples overfit (`P = 4240`): the
ambient reports' anchor is ~12.5 % / NLL 2.21 on B at the same `N`.  So part of the
restricted arms' gain is *undoing the anchor's bias*, not posterior quality, and these
cells are not yet on the ambient experiment's anchor.  Fixing that is the next cell, not a
reinterpretation of this one.

**(f) What is not done.**  Brier score and predictive MCSE for accuracy; ranks 4–32; a
Task-B-fitted anchor/basis arm with cross-fitting (the empirical-Bayes arm); ambient vs
restricted at matched likelihood evaluations; `λ ≠ 1`; and the informed-basis construction
the plan names besides trajectory PCA (prior-whitened Gauss–Newton/Fisher directions).  The
programme's gate — "an informed basis must beat or explain its tie with a random basis,
not just beat a failed ambient run" — is met for the negative control at ranks 1–2 and not
yet for anything larger.

Reproduce:

```bash
cargo run --release -p pcnl --example subspace_adaptation                    # ranks 1,2,4,8
cargo run --release -p pcnl --example subspace_adaptation -- --ranks 1,2 --steps 4000 --beta 0.05
```

## Reproduce / verify

```bash
cd unfer
cargo test -p pcnl --locked                              # 157 tests (116 unit + 7 comparators
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
