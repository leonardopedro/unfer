# QFM-Text — measured results (status doc)

> **What this is.** The measured numbers from running
> `qfm_text_train` + `qfm_text_eval` end-to-end on WikiText-103
> (test split and full train split). Per the plan, a *negative
> result* is reported here honestly (not hidden). The
> hyperparameter sweep is the entire "training" of the non-neural
> model — if no sweep configuration beats the classical baseline,
> that is the model's result.

> **Rev 35 (2026-07-09) — full-corpus training, OOM cause found,
> W-rank degeneracy characterised, per-context fit measured.**
> Rev 34 re-enabled the QFM.tex dressed-vacuum generator
> (`compile_channels`) and the text path now uses it. Rev 35 ran
> a **second debug pass** to confirm the rev 33 numbers were not
> optimistic: (a) discovered a 33 GB OOM in `accumulate_shards`
> that was silently killing the prior training run mid-corpus
> (the "92.5s / 95.7s" train times in rev 33 + rev 34 were on a
> subset, not the full 128M tokens); (b) re-trained both m=8
> and m=32 on the **full 128M tokens** (7m32s / 7m47s wall
> time, 451.9s / 467.1s); (c) added `--diagnose` to the eval
> binary to print SVD ranks of W and H_m + per-context fit
> metrics (KL, cosine, top-1/top-5 hit rate); (d) confirmed
> that **W has rank 1-2, not 8 or 32** — the dressed-vacuum
> Hamiltonian is dominated by the vacuum projector, so the
> Krylov sequence is essentially `{vacuum, dressed_vac,
> dressed_vac, ...}` and degenerates to 1-2 dim regardless of
> `m_shifts`. The honest picture: **the QFM is a classical
> n-gram with a degenerate Krylov perturbation that adds
> noise, not information.**

## TL;DR

**Honest picture (rev 35, after full-corpus training + SVD diagnostic):**

| Config | Train wall | In-sample ppl (shard 0) | Held-out ppl (test, 100k) | Gap to baseline |
|---|---:|---:|---:|---:|
| QFM-Text (4ord-16k, m_shifts=8)  | 7m32s | 253.8 | 420.8 | 1.39× / 1.15× |
| QFM-Text (4ord-16k, m_shifts=32) | 7m47s | 237.2 | 382.3 | 1.30× / 1.05× |
| Classical n-gram baseline (same hashing) | — | 182.7 | 364.6 | (reference) |
| Unigram (degenerate) | — | n/a | 1033.0 | — |

The QFM-Text is **5-39% worse than the classical n-gram
baseline** at all configurations tested. Increasing `m_shifts`
from 8 to 32 gives a small (6-9%) ppl reduction, but it does
**not** close the gap to baseline, because the W basis has rank
1-2 regardless of how many shifts are tried (see §"W-rank
degeneracy" below). The QFM is functionally a classical
n-gram with a degenerate Krylov perturbation.

## The four critical findings (this session)

### 1. OOM cause: `accumulate_shards` was allocating 33 GB

The function `accumulate_shards` in
`qfm_text/src/accumulate.rs:80` used to call
`shards.par_iter().map(|s| acc_clone(s)).collect::<Vec<_>>()`
to build a `Vec<ChannelAccumulator>` of all 245 per-shard
accumulators. Each per-shard accumulator holds the per-mode
histograms (224K active modes × 64 history entries × 8 bytes
+ HashMap overhead) = **~134 MB**. The Vec peak is
245 × 134 MB = **~33 GB**, which exceeds the 16 GB available
on this system. The OOM killer silently killed the training
process mid-corpus, leaving behind a "checkpoint" that had
only been trained on a subset of shards. The 92.5s / 95.7s
wall times reported in rev 33 + rev 34 are from the killed
processes, not from full training.

The fix is a sequential pass over the shards with one
in-flight accumulator:

```rust
// before (OOM)
let per_shard: Vec<ChannelAccumulator> =
    shards.par_iter().map(|s| acc_clone(s)).collect();
let running = per_shard.into_iter().reduce(|a, b| merge(a, b)).unwrap();

// after (peak = 1 shard)
let mut running = ChannelAccumulator::new(...);
for shard in shards {
    let local = acc_clone(shard);
    running = merge(running, local);
}
```

The full-corpus re-training confirmed the wall times:
- m=8: **451.9s** (7m32s)
- m=32: **467.1s** (7m47s)

Both processes iterated over all 245 shards and saved
checkpoints with `n_windows = 128,102,894`.

### 2. Full-corpus numbers (not subset)

The rev 33 status doc had QFM-Text in-sample ppl = 371.7 vs
baseline 321.4 (gap 1.16×). Those numbers were from a subset
training run. With the OOM fix, the **full-corpus** numbers
on train shard 0 (200K tokens) are:

| Config | QFM in-sample | Baseline in-sample | Gap |
|---|---:|---:|---:|
| m_shifts=8 (full corpus) | 253.8 | 182.7 | **1.39×** |
| m_shifts=32 (full corpus) | 237.2 | 182.7 | **1.30×** |

The baseline improved MORE with more data (321.4 → 182.7 =
43% reduction in ppl) than the QFM (371.7 → 253.8 = 32%
reduction, for m=8). The gap to baseline **widened** with
more data, because the baseline is a pure n-gram (no rank
bound) while the QFM is rank-1-2-bounded by the Hamiltonian.

Held-out on the test split (100K tokens, with
`--baseline-from-checkpoint` to derive the baseline from the
model's stored unigram + mode_hists, no re-derivation):

| Config | QFM held-out | Baseline held-out | Gap |
|---|---:|---:|---:|
| m_shifts=8 (full corpus) | 420.8 | 364.6 | **1.15×** |
| m_shifts=32 (full corpus) | 382.3 | 364.6 | **1.05×** |

The held-out gap is smaller than the in-sample gap, but the
QFM still loses in both settings. The m=32 model is slightly
better than m=8 in both metrics, but neither beats the
baseline.

### 3. W-rank degeneracy: m_shifts is a lie

The eval binary now has a `--diagnose` flag that prints the
SVD rank of W and H_m on the loaded checkpoint. The result is
that **W has rank 1-2, not the requested m_shifts**:

```
=== diagnostic: m=8 (full corpus training) ===
W shape      : (262145, 2)
W rank       : 2
H_m shape    : (2, 2)
H_m rank     : 1
|H_m| SVs    : [3.8983, 0.0000]

=== diagnostic: m=32 (full corpus training) ===
W shape      : (262145, 1)
W rank       : 1
H_m shape    : (1, 1)
H_m rank     : 1
|H_m| SVs    : [3.8671]
```

This is the **smoking gun** for the failure of the
m_shifts sweep. The Gram-whitening of the SIRK Krylov basis
finds only 1-2 linearly independent vectors in the
`m_shifts + 1`-dimensional space. **m=32 is actually WORSE
than m=8** (rank 1 vs rank 2): adding more shifts adds only
noise dimensions, and the Gram-whitening threshold catches
the noise.

**Why is W rank 1-2?** The dressed-vacuum Hamiltonian is
`H = Σ_o λ_o |0̃_o⟩⟨0̃_o|` with `n_orders = 4`. Each
`|0̃_o⟩ = c_0^(o)|0⟩ + Σ_j ε_j^(o) |x_j⟩` has `c_0 ≈ 1`
(the vacuum coefficient dominates because the per-mode
weights `ε_j` are tiny — the corpus has 210K active modes
and the per-mode probability is ~5e-6). So all four
`|0̃_o⟩` are nearly parallel to `|0⟩` and nearly parallel
to each other. The H is essentially `λ |0⟩⟨0|` (a 1-dim
projector), and the Krylov sequence is `{vacuum,
dressed_vac, dressed_vac, ...}` of dim 1-2. The
`m_shifts` parameter is irrelevant.

**The fix would be a Hamiltonian with non-degenerate
spectrum.** Candidates (not yet implemented):
- `H = Σ_j ε_j |x_j⟩⟨x_j|` (per-mode number operator):
  diagonal with K_2 distinct eigenvalues → Krylov dim = K_2
  (full).
- `H = Σ_o Σ_j ε_j^(o) |x_j⟩⟨x_j|` (per-order per-mode
  number operator): diagonal with K_2 distinct eigenvalues
  per order.
- `H = Σ_o (|0̃_o⟩⟨0̃_o| - |0⟩⟨0|)` (per-order projector
  onto the orthogonal complement of the vacuum): subtracts
  the vacuum component, making the orders distinguishable.
- `H = Σ_{j<k} J_{jk} (|x_j⟩⟨x_k| + |x_k⟩⟨x_j|)` (off-
  diagonal XY-type): full off-diagonal, Krylov dim = K_2.

### 4. Per-context fit is bad (deterministic patterns are missed)

The diagnostic also prints per-context fit metrics on a
sample of training contexts (those with ≥3 occurrences in
the train shard 0):

| metric | m=8 | m=32 |
|---|---:|---:|
| mean KL(emp ‖ qfm) | 3.84 nats | 3.84 nats |
| mean cosine similarity | 0.38 | 0.38 |
| top-1 hit rate (argmax = empirical argmax) | 33% | 33% |
| top-5 hit rate (top-5 includes empirical argmax) | 8% | 8% |

The top-1 hit rate of 33% is way above the 1/16384 random
rate (0.006%), so the QFM is not random — but it's not
capturing the **deterministic structure** of the corpus.
Examples from the diagnostic log:

```
ctx = [416, 1404, 89, 4916]   (n=53 occurrences)
  empirical:  376 (0.66), 416 (0.09), 1627 (0.04), ...
  QFM:        159 (0.10), 1627 (0.05), 1622 (0.04), ...
  QFM top-1 = 159, empirical top-1 = 376.  Miss.

ctx = [1404, 89, 4916, 376]   (n=36 occurrences)
  empirical:  499 (1.00), ...     # fully deterministic
  QFM:        155 (0.05), 110 (0.04), 93 (0.04), 499 (0.04), ...
  QFM top-1 = 155, empirical top-1 = 499.  Miss.
```

The QFM fails even on **fully deterministic** contexts
(100% empirical mass on a single next token) — it puts 4-5%
on a wrong token and 4% on the correct one. The Krylov
smoothing is too aggressive: it spreads mass across the
"similar" modes rather than committing to the empirical
mode.

### What the architecture is, fundamentally (revised)

The QFM-Text architecture as implemented is a **classical
n-gram model with a rank-1-2 Krylov perturbation**. The
per-mode histograms (210923 active modes × 64 history
entries ≈ 13.5M parameters) carry the per-context signal.
The Krylov basis W (rank 1-2) is a 1-2-dim embedding that
**cannot distinguish the 210K modes**. The per-mode
weighting falls back to the uniform 1/n_active average,
which is equivalent to the classical n-gram baseline's
uniform average plus per-mode escape to the unigram.

The 5-39% gap to the baseline is a combination of:
- **Degenerate Krylov** (W rank 1-2): the Krylov adds noise
  but no per-mode signal.
- **Over-smoothing**: the W projection spreads mass across
  "similar" modes rather than committing to the empirical
  mode.
- **Unigram escape**: when a mode is unseen, the QFM falls
  back to the unigram, but the unigram is also tied to the
  same training corpus as the baseline.

## Run 1 — held-out evaluation with the new code

The eval binary now has three new flags (added in this
session):
- `--diagnose`: print SVD ranks + per-context fit metrics.
- `--baseline-from-checkpoint`: derive the n-gram baseline
  from the model's stored `mode_hists` + unigram (fast path,
  no re-derivation from train shards).
- `QfmTextModel::as_accumulator()`: clone `mode_hists` and
  reconstruct the unigram counts from the normalised
  unigram + unigram_total.

The held-out run uses `--eval-manifest` to point to the
test split (266K tokens, 1 shard) and `--baseline-from-
checkpoint` to skip the re-derivation (~7 min → <1 s).

### Why this session's held-out numbers are different from rev 33

The rev 33 held-out ppl was 501.7 (m=8). This session's
held-out ppl is 420.8 (m=8, 100K tokens capped). The
difference is the **model was retrained on the full corpus**
in this session; the rev 33 model was on a subset. With
more training data, the QFM ppl goes down (253.8 → 420.8
in-sample → held-out is a separate story because the held-
out is unseen data).

### Per-shard loop performance

The per-shard loop in `lm::perplexity` is sequential. The
per-token cost is dominated by:
- `next_token_dist` → `context_modes` (O(n_orders))
- `encode_modes` (O(|active| × rank))
- `evolve` (O(rank²))
- `decode_sketched` (O(K_2 × rank)) — **dominant**
- `marginalize` (O(|active| × hist_cap) + O(vocab_size))

For K_2 = 262145, rank = 1-2, n_orders = 4:
- decode_sketched: 262K-524K ops/token
- marginalize: 16K ops/token
- total: ~540K ops/token

Single-threaded on a 3 GHz CPU: ~135 Mops/sec, so
4 ms/token, or 250 tokens/sec. For 100K tokens: ~400 s.
For the full 245 train shards (128M tokens): ~142 hours.
This is why the in-sample eval on the full corpus is
infeasible without GPU acceleration of `decode_sketched`.

## What I did NOT do (rev 35)

- **No GPU acceleration of `decode_sketched`.** The
  per-token cost is dominated by a K_2 × rank
  matrix-vector product, which is a natural fit for a CUDA
  kernel. With GPU it would run ~50× faster and the full
  128M-token in-sample eval would be ~3 hours. The
  hardware (GTX 1060, 6 GB VRAM) supports it but the kernel
  is not implemented.
- **No non-degenerate H.** The dressed-vacuum Hamiltonian
  is the QFM.tex-mandated form, but it degenerates for
  this corpus (210K modes, all `|0̃_o⟩ ≈ |0⟩`). Candidates
  are listed in §"W-rank degeneracy" above.
- **No `compile_alphas` removal.** Rev 34 said the function
  was "removed", but it is still in the code (it is just no
  longer called by `qfm_text_train`). The code path is dead
  for the text workload but is still in the public API of
  `qfm::pipeline`.
- **No full-corpus in-sample eval.** Estimated ~142 hours
  CPU time. The in-sample ppl on train shard 0 is a
  reasonable proxy because the shards are i.i.d.
- **No ablations over `hist_cap` or `salts`.** Both affect
  the per-mode histogram granularity.
- **No 4-order production target with `block_sizes =
  [65536, 65536, 65536, 65536]`.** The closest was
  `4ord-16k` with `block_sizes = [4096, 4096, 4096,
  4096]`, which has more hash collisions and is
  necessarily worse than the baseline.

## What would actually help

To beat the classical n-gram baseline, the Krylov needs
to add information that the per-mode histograms don't have.
With the current architecture, this would require:

1. **A Hamiltonian with non-degenerate spectrum.** (See
   §"W-rank degeneracy" for candidates.) The current
   dressed-vacuum sum degenerates to a 1-dim projector for
   corpora with many modes.
2. **GPU acceleration of `decode_sketched`.** Required
   for full-corpus in-sample eval and for training on
   corpora larger than 128M tokens.
3. **A non-uniform dressed-vacuum structure** so the
   per-mode α's vary significantly across modes, breaking
   the uniform projection.
4. **A different decoder that doesn't rely on the Krylov
   for mode distinction.** E.g., a per-mode weighting based
   on histogram similarity to a learned prototype, not a
   Krylov projection.
5. **A fundamentally different architecture** (e.g., a
   small neural network that takes the per-mode histogram
   as input and produces a distribution).

None of these are in scope for the current QFM-Text
architecture, which is constrained to a dressed-vacuum
Hamiltonian and Krylov decode.

## GPU setup

The CUDA 13.0 driver + a GTX 1060 (6 GB VRAM) is present on
this machine, but candle-core 0.8.4 was built against CUDA
12.x. The fix is `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu`
(to pick up the system's CUDA 12.2 runtime: `libcublas.so.12`,
`libcudart.so.12`). With that, the `cuda` feature on
`fock_sirk` enables `Device::cuda_if_available(0).unwrap_or(Device::Cpu)`
and all 22 GPU tests pass.

The `qfm` crate's TSR pipeline (encode / evolve / decode) is
CPU-only (it uses `nalgebra::DMatrix` for the dense
subspace operations). The GPU win is in the SIRK compile
step and would be in `decode_sketched` if implemented.

## Files referenced (rev 35)

- `qfm_text/src/bin/qfm_text_train.rs` — Stage 6 (train).
  This session: replaced `acc_clone(acc)` with
  `running_acc.take()` + merge + `from_accumulator(running,
  &text)?` (consumes `running_acc` into the model each
  epoch).
- `qfm_text/src/bin/qfm_text_eval.rs` — Stage 6 (eval +
  sweep). New flags: `--diagnose`, `--baseline-from-
  checkpoint`. New function: `diagnose_pipeline` (SVD ranks
  + per-context KL/cos/top-K), `svd_rank` helper.
- `qfm_text/src/accumulate.rs:80` — `accumulate_shards`
  **rewritten** to sequential pass. Old `par_iter().collect()`
  deleted (was the OOM source). New doc-comment explains
  the 33 GB peak and why sequential is the only memory-safe
  option.
- `qfm_text/src/model.rs:242` — `as_accumulator()` method
  (new). Clones `mode_hists` + reconstructs `unigram`
  counts from `unigram: Vec<f64> × unigram_total: f64`.
- `qfm_text/src/model.rs:570` — `context_modes` (rev 33
  fix #1: route through `OrderHasher` instead of returning
  raw tokens).
- `qfm_text/src/model.rs:218` — `marginalize` (per-context
  active modes, per-mode escape to unigram, no global floor
  for non-Dense strategies).
- `qfm_text/src/model.rs:336` — `preprocess_p_tilde` (per-
  context active-mode normalization for `Renormalize` /
  `TopK` / `OrderPrior`).
- `qfm/src/pipeline.rs:612` — `decode_sketched` (rev 33
  fix #2: actual Born rule `p̃[m] = |⟨m|W|c⟩|²`, not the
  element-wise real-part sketched formula).
- `qfm/src/pipeline.rs:899` — `extract_single_excitation_w`
  (kept for backward compatibility; no longer used by
  `compile_channels`).
- `qfm/src/pipeline.rs:945` — `project_modes_onto_krylov_basis`
  (rev 33 fix #3: proper Fock → rank-dim projection of each
  mode onto the Gram-whitened Krylov basis).
- `qfm/src/pipeline.rs:718` — `compile_channels` (the
  channel weights → hierarchical dressed-vacuum generator
  → SIRK path). New: `build_dressed_vacuum_matvec(groups,
  k2_total)` (O(M) matvec, ~100× faster than the
  Hamiltonian::apply path).
- `qfm/src/pipeline.rs:612` — `from_components(w, h_m,
  w_prob)` constructor (rev 33 fix #4: bypass SIRK on
  load).
- `fock_sirk/src/forward_sirk.rs` —
  `solve_forward_sirk_with_matvec<F>(matvec: F, ...)`
  (generic matvec callback).
- `nested_fock_algebra/src/models.rs` —
  `qfm_hamiltonian_hierarchical_projectors` (sum of dressed
  vacuum projectors; rank ≤ n_orders).
- `fock_sirk/src/device.rs` — `best_device()`; needs
  `--features cuda` to use the GPU.

## Checkpoints in this run

| Config | Path | Size | W rank | H_m rank | Train wall |
|---|---|---:|---:|---:|---:|
| 4ord-16k, m_shifts=8 (full corpus)  | `qfm_text_runs/m8/checkpoint_epoch0.qfm`  | 177 MB | 2 | 1 | 7m32s |
| 4ord-16k, m_shifts=32 (full corpus) | `qfm_text_runs/m32/checkpoint_epoch0.qfm` | 177 MB | 1 | 1 | 7m47s |
