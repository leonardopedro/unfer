# pCNL × nGPT-lite — measured results (status doc)

> **What this is.** The Week-3 artifact of
> [`docs/plan_bayes_ngpt.md`](plan_bayes_ngpt.md): the full unadjusted
> SG-pCNL experiment run end-to-end (Task-A pre-train → adaptation to
> Task B → diagnostics → BMA), quoted verbatim, with an honest reading
> against the plan's Phase 0 / Phase 4 claims — including the claims
> that did **not** come out clean.

> **Run (2026-11-04, current `main`).**
> `cargo run --release -p pcnl -- --report-file OUT.txt`
> (defaults: seed 20261020, 10 000 steps × 4 chains × 2 variants,
> P = 16 672, σ = 0.35, β₀ = 0.3, κ = 0.55, γ = 1e-3, batch 64).
> Deterministic in the seed. `RAYON_NUM_THREADS=2` on this host;
> cosine variant wall time 1050.4 s.

> **Host note (OOM).** The machine (14 GiB, no dedicated GPU work) had a
> **global OOM event at 2026-11-04 09:31**, before this run: the kernel
> killed processes inside the `waydroid` container and the desktop
> session (`os.lineageparts`, `waydroidupdater`, `externalstorage`, …).
> It was not this experiment — the pCNL process' peak RSS during the full
> run was ≈ 0.4 GB (128 MiB at half-way, 413 MiB at the end). The runs
> below were made with `RAYON_NUM_THREADS=2` as a precaution against a
> second OOM, and they completed.

## TL;DR

| Variant | Phase-0 outcome | radial ‖θ‖ (σ√P = 45.19) | BMA acc / NLL / ECE | Adam-on-B acc | ESS(loss) | R̂(loss) | cross-R̂ |
|---|---|---|---|---:|---:|---:|---:|
| **cosine / nGPT-lite** | stable: 10 000 unadjusted steps, no MH | mean 45.3, final 45.7, max 127.9 | **11.7 %** / 2.1863 / 0.1798 (K=2000) | 55.1 % | 194 | 1.004 | 1.007 |
| **standard softmax** | **EXPLODED at step 3** (non-finite gradient) | max 6.9e18 | n/a (chain died) | 33.2 % | — | — | — |

Chance level is 12.5 % (8 classes). So:

- **Phase 0 is confirmed, measured.** The unadjusted update survives
  10 000 mini-batch steps with no Metropolis filter *only* under the
  normalized geometry; the standard variant dies at step 3.
- **Phase 3 is NOT confirmed by this run.** BMA at 11.7 % is
  chance-level — worse than the no-adaptation baseline's NLL and far
  below Adam-on-B (55.1 %). The SGLD and MH-filtered comparators report
  the *same* 11.7 %, so the failure is in the sampler's ability to reach
  the Task-B-fitting region, not in the missing MH filter. See
  “The Phase-3 result did not come out clean” below for the sweep that
  rules out σ-tuning as the cause.

## Report (verbatim)

```text
SG-pCNL (unadjusted) x nGPT-lite toy run (seed 20261020, sigma=0.35, beta0=0.3, kappa=0.55, gamma=0.001, batch=64, chains=4, steps=10000, burn=5000, thin=10, taskA=(a+b) mod 8, taskB=(a+2b) mod 8, n_train=512, pretrain=1500)
[cosine/nGPT-lite] P=16672
  pre-train A: loss 5.0673 -> 1.0261, acc_A 57.0%
  B baselines: before 13.7% (NLL 4.2124), SGD-on-B 39.1% (NLL 1.3749), Adam-on-B 55.1% (NLL 1.1367)
  SG-pCNL: beta 0.3000 -> 0.0802 over 10000 steps | max scaled|g| 633.809 max batch loss 4.484 | radial final 45.7 max 127.9 mean 45.3 vs sigma*sqrt(P) 45.2 | ESS(loss) 194 ESS(coord)med 15 | Rhat(loss) 1.004 Rhat coord med 1.086 max 1.537 | BMA 11.7% NLL 2.1863 ECE 0.1798 (K=2000) last-state 11.7% | 1050.4s
  SGLD (baseline): radial final 45.3 max 128.1 vs sigma*sqrt(P) 45.2 | ESS(loss) 201 | Rhat(loss) 1.004 | BMA 11.7% NLL 2.1839 ECE 0.1775 (K=2000) | cross-Rhat 1.006 | 218.0s
  MH-pCNL (filtered): accept 99.6% | radial final 45.2 vs sigma*sqrt(P) 45.2 | ESS(loss) 181 | Rhat(loss) 1.010 | BMA 11.7% NLL 2.1861 ECE 0.1790 (K=2000) | kept 500 per chain | cross-Rhat 1.007 | 463.5s
[standard softmax] P=16672
  pre-train A: loss 2.1850 -> 851.3527, acc_A 16.0%
  B baselines: before 12.9% (NLL 1151.1021), SGD-on-B 10.9% (NLL 379.6498), Adam-on-B 33.2% (NLL 4.8560)
  SG-pCNL: EXPLODED at step 3 (max scaled|g| 1252694024522353344512.000, gradient non-finite, max batch loss 8945304553454643859310321336320.000, radial max 6882751474854526976.000 vs sigma*sqrt(P) 45.192)
```

The quoted block is the report file verbatim (the run prints no save
paths when `--save-dir` is not given). Determinism is the test suite's
claim, not this doc's: `tests/numerics.rs::experiment_is_bit_deterministic_across_reruns`
runs the quick config twice and compares the numbers field by field;
the full run was measured once at this code state.

## The Phase-3 result did not come out clean

BMA 11.7 % with NLL 2.1863 is not a small shortfall: ln 8 = 2.0794, so
the averaged predictive is *near-uniform*, and 11.7 % is the accuracy a
near-uniform predictive gets by chance. The three chains that share the
proposal (unadjusted, SGLD, MH-filtered) agree to three decimals,
which is the useful part of the negative result: **the missing MH
filter is not the cause** — adding it back changes nothing.

The obvious escape hatch would be σ. It does not work. Sweep at the
small budget (`--no-mh --steps 1500 --chains 2 --d 16 --ff 32
--n-train 256 --pretrain 400`; before-adaptation 13.7 %, SGD-on-B
25.8 %, Adam-on-B 43.0 %):

| σ | radial vs σ√P | BMA | NLL |
|---|---:|---:|---:|
| 0.02 | 1.3 / 1.3 | 11.7 % | 2.2168 |
| 0.05 | 3.3 / 3.3 | 11.7 % | 2.2168 |
| 0.10 | 6.6 / 6.5 | 11.7 % | 2.2168 |
| 2.00 | 131.1 / 130.2 | 11.7 % | 2.2167 |

A 100× change in the prior width leaves the predictive where it was,
while the parameter norms follow σ√P exactly — so the chain is
sampling *somewhere* correctly (§3B's radial claim holds throughout) and
the readout is insensitive to it. Lengthening the chain does not help
either: 600 / 1200 / 2400 / 3000 steps give 11.7 % at every length at
σ = 0.35.

The one knob that moves the number is the **centered adaptation prior**
(§6.4.2, `--prior-center-a`), which is the reachability fix the plan
already carries:

| centered prior, σ | BMA | NLL | ECE |
|---|---:|---:|---:|
| 0.01 | 14.8 % | 3.0512 | 0.2085 |
| 0.03 | 21.1 % | 2.9338 | 0.1504 |
| 0.10 | 21.1 % | 2.6702 | 0.1336 |

21.1 % beats the checkpoint (13.7 %) but still loses to SGD-on-B
(25.8 %) and to Adam-on-B (43.0 %) — so the centered prior is the right
direction and is not yet enough at this budget.

**Reading.** With the plan's prior draw as the start (θ_A + ξ,
‖ξ‖ = σ√P = 45), the chain begins at weights whose predictions are
near-uniform, and 10⁴ OU steps over P = 16 672 dimensions do not find
the Task-B-fitting region; the ensemble then averages confident-but-
scattered predictions into a near-uniform one (NLL above ln 8, not
below it). This is a statement about *this toy at this budget*, not a
disproof of §3 — the diagnostics (ESS 194, R̂ 1.004, cross-R̂ 1.007)
say the chains are healthy samplers of whatever they are sampling.

**Concrete next steps, in order of expected value.**
1. Start the chain *at* θ_A (a `--init-at-a`-style start) with the
   centered prior and a σ scaled to ‖θ_A‖/√P — the plan's Phase 3 text
   (“initialize from the base checkpoint, burn in to let the drift guide
   the chain”) reads that way; the current code always adds a full prior
   draw.
2. Report the kept samples' train loss (not only the BMA metrics), so
   “near-uniform ensemble” is measured rather than inferred from NLL.
3. Scale the budget with P: β₀ = 0.3 with P = 16 672 is 10⁴ steps of a
   16 k-dimensional diffusion; the cosine schedule (§3C Option 2) is
   the obvious instrument once the start is fixed.

## Phase-0 reading (unchanged, and re-measured)

The standard dot-product variant died at unadjusted step 3 with a
non-finite gradient (scaled ‖g‖ ~ 1.3e21, radial ~ 6.9e18 against
σ√P = 45.2); its deterministic Task-A pre-train had already blown up to
loss 851.4. The cosine variant ran all 10 000 unadjusted mini-batch
steps with no Metropolis filter anywhere, radial mean 45.3 / final 45.7
vs σ√P = 45.2 (max 127.9 is a transient), exactly the §3B claim. This is
one config, one seed — not a proof.

## Throughput and memory (plan §4 Phase 4b, re-measured)

Deterministic evaluation counts are the reliable statement and are
asserted in `pcnl/tests/comparators.rs::unadjusted_halves_model_evaluations`:
2 model evaluations per MH-filtered step vs 1 per unadjusted step.

Wall-clock on *this* run is noisier than the count, and it went the
other way from the expectation:

| chain family | wall (10 k steps × 4 chains) |
|---|---:|
| unadjusted SG-pCNL | 1050.4 s |
| SGLD comparator | 218.0 s |
| MH-filtered pCNL | 463.5 s |

The unadjusted chain is the slowest here even though it does the least
per step; the extra cost is the full-batch forward every `thin` = 10
steps plus the full-batch loss on kept samples, and the rayon
scheduling of 4 chains on 2 threads. Treat the counts as the claim and
these seconds as one host's measurement. Memory stays decoupled from N
(the state is θ ∈ ℝ^P plus one streamed mini-batch; peak RSS ≈ 0.4 GB
here).

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
- **`rayon` (Phase 2, "parallel chains"): done** —
  `rayon::into_par_iter`, merged in chain order, per-chain independent
  random streams (`tests/numerics.rs::experiment_is_bit_deterministic_across_reruns`).
- **§3C Option 2 (cyclical β): implemented in
  `SgPcnlScheduler::beta_cyclical`, measured run uses Option 1
  (Robbins–Monro).**
- **§3C in-place update:** the state replaces `theta` on its device
  (the sample code's “in-place on GPU memory”); candle tensors are
  immutable, so “in-place” means the state's single θ slot is
  overwritten each step, not that device memory is mutated. This run is
  CPU-only.

## Previous run record (2026-10-20, code state `e1a3eda`)

For history, and because it shows how basin-sensitive this experiment
is: the same command on the pre-`--prior-center-a` code reported BMA
**54.7 % / NLL 1.0686 / ECE 0.1589** with ESS(loss) 18 and R̂(loss)
1.115. That code's Task-A pre-train converged to chance (loss 2.0743,
acc_A 12.9 %); the current pre-train learns Task A (loss 1.0261,
acc_A 57.0 %). Same seed, same task, same sampler — the outcome moved
from above-chance to chance when the checkpoint changed, which is the
observed variance between basins that next step 1 above is meant to
remove. Quick-config check with both builds (400 steps, d = 16):
`e1a3eda` BMA 31.2 % / NLL 1.5136; current `main` BMA 15.6 % /
NLL 2.1961 — same direction, smaller budget.

## Reproduce / verify

```bash
cd unfer
cargo test -p pcnl --release --locked       # 75 tests (42 unit + 4 comparator + 8 formal + 21 numerics)
RAYON_NUM_THREADS=2 cargo run --release -p pcnl -- --report-file out.txt
RAYON_NUM_THREADS=2 cargo run --release -p pcnl -- --no-mh --steps 1500 --chains 2 \
    --d 16 --ff 32 --n-train 256 --pretrain 400 --prior-center-a --sigma 0.03
cargo run --release -p pcnl --example dump_trajectory -- /tmp/traj/cosine_nGPT-lite_chain0.safetensors
```

Gates: `cargo fmt -p pcnl -- --check` and
`cargo clippy -p pcnl --all-targets -- -D warnings` are clean.
