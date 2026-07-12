# rev 37 plan v2: atomic, isolated component tests

**User feedback 2026-07-10:** the original rev 37 plan (rev37_oxieml_plan.md) tried to
do too much in one step. **Break each architectural component into a single
test that can be run in < 60 s, on synthetic data, and verified independently
before integration.** Examples the user gave:

- "Test the QFM model without trying to reduce the change of basis matrix to
  the Krylov subspace" — i.e. test the QFM *forward path* (encoder +
  per-order channel sum) without ever building H_m = W^H H W.
- "Test the decoder only using oxieml on a random M×m matrix" — already
  done in `qfm_text/src/oxieml_decoder.rs::tests`.

This file is the revised plan. **It replaces the `Implementation steps`
section of `rev37_oxieml_plan.md`.**

## Strategy: phases of atomic tests

| phase | what it tests | expected runtime per test |
|-------|---------------|---------------------------|
| **0** | oxieml decoder on synthetic + real W (DONE) | 20-60 s |
| **1** | each model component in isolation | < 5 s each |
| **2** | QFM forward path *without* Krylov reduction | 5-30 s |
| **3** | integrate Decoder enum | 30-60 s |
| **4** | end-to-end train + eval on shard 0 | 5-30 s train, < 1 s eval |
| **5** | end-to-end on full corpus | 5-15 min train |

Each step is a single `cargo test` invocation that passes or fails
independently. **No step depends on a previous step's success** — every
step is verifiable on its own.

---

## Phase 0 — oxieml decoder (DONE, see `oxieml_decoder.rs` + `oxieml_fit_real_w.rs`)

- **0.1** `fits_pure_sinusoid_with_low_residual` — synthetic `sin(2πi/M)`
  over M=1024 modes. oxieml `max_depth=3` gets a finite (but coarse) fit.
  ~30-60 s, gated.
- **0.2** `fits_polynomial_with_low_residual` — synthetic `0.5x² + x + 0.1`,
  M=1024. oxieml `max_depth=3` recovers within 1% MSE. ~20-30 s, gated.
- **0.3** `fit_random_matrix_high_residual_falls_back` — random M=1024
  values, logs the residual (no assertion). Confirms the random baseline
  behaves reasonably. ~10 s, gated.
- **0.4** `fit_decoder_multiple_columns` — 2-col W, sin + poly, both fits
  succeed. ~30-60 s, gated.
- **0.5** `oxieml_fit_real_w.rs::real_w_fits_with_low_mse` — loads the
  real shard 0 model W (rank 2, M=1M+), fits oxieml on both columns. MSE
  5e-5 / 7e-5. ~20 s, gated.

**The big finding from 0.5:** the W matrix IS smooth in `mode_index`
(real Fock-space wave-function), and oxieml fits it with mse ~10⁻⁴.
**This validates the central premise of rev 37.**

---

## Phase 1 — atomic model-component tests (TODO, ~5-10 min)

Each is a `#[test]` in a new file `qfm_text/tests/atomic_components.rs`.
**No integration with the full pipeline required** — each test
constructs its own minimal state.

### Step 1.1 — encoder (OrderHasher) in isolation

**Test name:** `encoder_is_deterministic_and_per_context_unique`

**Setup:** create a `TextConfig` with `block_sizes=[1024, 1024]`, `salts=[1, 2]`,
build an `OrderHasher`, encode the same context twice, and 100 different
contexts once.

**Pass criteria:**
- The same context maps to the same mode in both runs.
- 100 distinct contexts produce ≥ 95 distinct modes (a small number of
  collisions is expected at block_size=1024 for 100 contexts).
- The mode indices are all in `[0, 2048)` (no out-of-range).

**Why this is a useful test:** the encoder is the first thing the
pipeline calls. If it's non-deterministic or has pathological
collisions, nothing downstream can work.

### Step 1.2 — channel weight computation in isolation

**Test name:** `channel_weights_sum_to_total_windows`

**Setup:** create a tiny `ChannelAccumulator`, observe 5 (context, next)
pairs where 3 contexts are unique, verify `acc.total_windows == 5` and
the sum of `acc.stats[*].weight == 5`.

**Pass criteria:**
- `acc.total_windows` equals the number of (context, next) observations.
- The sum of per-mode weights equals `acc.total_windows`.
- The per-order block_size is consistent with the encoder config.

**Why this is a useful test:** the channel weights are the α_j that go
into H = Σ λ_o |0̃_o⟩⟨0̃_o|. If the weights are wrong, the Hamiltonian
is wrong.

### Step 1.3 — Hamiltonian H = Σ λ_o |0̃_o⟩⟨0̃_o| in isolation

**Test name:** `hamiltonian_is_outer_product_of_dressed_vacuum`

**Setup:** call `qfm_hamiltonian_hierarchical_projectors` with known
`(λ_o, channels_o)` groups; verify the output is a sum of rank-1
projectors. Specifically:
- For each order o, define `|0̃_o⟩` and verify `H_o = |0̃_o⟩⟨0̃_o|`.
- Verify `H_o` has eigenvalue 1 on `|0̃_o⟩` and 0 on the orthogonal
  complement.
- Verify the total H has rank ≤ n_orders.

**Pass criteria:**
- H is Hermitian.
- H has exactly n_orders nonzero eigenvalues (rank ≤ n_orders).
- H|0̃_o⟩ = |0̃_o⟩ for all o.
- H|ψ⟩ = 0 for any |ψ⟩ orthogonal to all |0̃_o⟩.

**Why this is a useful test:** the Hamiltonian is the *only* off-diagonal
generator (QFM.tex §"Scope", rev 31). If H doesn't have the expected
structure, the time evolution is wrong.

### Step 1.4 — Born-rule decode in isolation

**Test name:** `decode_at_active_modes_matches_full_decode`

**Setup:** build a small `QfmPipeline` with M=64, rank=2, known W. Pick
a random h ∈ ℂ^2 (unit norm). Call `decode_sketched(h)` (full) and
`decode_sketched_at(h, &active_modes)` (sparse) with `active_modes =
[0, 1, 2, ..., 63]`. Compare.

**Pass criteria:**
- The two decode outputs agree to 1e-12 on the active modes.
- The sum of probabilities is 1 (Born rule).

**Why this is a useful test:** `decode_sketched_at` is the rev 36
optimization that makes the decoder O(n_active) instead of O(M). It's
load-bearing for the O(M²) memory savings.

---

## Phase 2 — QFM forward path WITHOUT Krylov reduction (TODO, ~30-60 min)

The "Krylov reduction" is `H_m = W^H H W ∈ ℂ^{rank × rank}`. The
"forward path" is the sequence:

```
c_0 = (1/√n) Σ_o W[m_o, :]            ← encoder + per-order channel sum
c_1 = exp(-i H_m t) c_0                ← SIRK time evolution
weights[m_o] = Σ_j h_j · W[m_o, j]     ← Born-rule decode
```

**Without the Krylov reduction**, we have two natural options:

### Step 2.1 — option A: skip SIRK entirely (c_1 = c_0)

The simplest "no-Krylov" model is `c_1 = c_0` (no time evolution). The
per-mode weight is then just `Σ_j c_0[j] · W[m_o, j] = c_0^H W[m_o, :]`,
the *projection of the channel sum back to the active modes*. This is
the absolute baseline.

**Test name:** `no_krylov_evolution_matches_analytic_baseline`

**Setup:** tiny corpus `[0, 1, 0, 1, 0, 1, ...]` (alternating tokens).
Train a model, evaluate, and verify:
- The no-Krylov model and a hand-computed `c_0^H W[m_o, :]` agree to
  1e-10 on the active modes.
- The held-out ppl is at most 1.5× the unigram ppl (the model is
  learning something).

**Why this is a useful test:** establishes the absolute baseline
performance and verifies the encoder→c_0→decode pipeline is correctly
wired.

### Step 2.2 — option B: full-rank time evolution (H not reduced to H_m)

Apply H directly to c_0 in the full M-dim space: `c_1 = exp(-i H t) c_0`.
This avoids the Krylov reduction but is O(M²) per time step (still
tractable for M=1M+).

**Test name:** `full_rank_evolution_matches_krylov_evolution_for_rank_1_h`

**Setup:** when the Krylov basis has rank 1, H_m = ⟨0̃|H|0̃⟩ (a 1×1
matrix) and the Krylov evolution `c_1 = exp(-i H_m t) c_0` is a single
complex exponential. The full-rank evolution `c_1 = exp(-i H t) c_0`
should agree with the Krylov evolution on the Krylov basis.

**Pass criteria:** on a rank-1 trained model, the two evolutions
produce the same h vector to 1e-6.

**Why this is a useful test:** confirms that the Krylov reduction is
correct (when rank=1, it should be exact). This is the "sanity check"
for the Krylov machinery.

### Step 2.3 — option C: use the *diagonal* surrogate (QFM.tex §"Scope")

QFM.tex describes a "diagonal surrogate" generator H_diag = diag(ε_j)
where ε_j = α_j/√(1+Σα²) are the per-mode dressed-vacuum coefficients.
This was the rev 33 interim generator (since removed in rev 34). It's
diagonal in the mode basis, so `exp(-i H_diag t) c_0` is a per-mode
phase rotation, no Krylov needed.

**Test name:** `diagonal_surrogate_preserves_born_populations`

**Setup:** compute the diagonal surrogate on a trained model's α_j.
Verify:
- `H_diag` is Hermitian (real diagonal).
- The Born populations `|c_0[m]|²` are stationary under
  `exp(-i H_diag t) c_0` (the off-diagonal correlations die, but the
  per-mode amplitudes are preserved).

**Why this is a useful test:** the diagonal surrogate is the
"degenerate" version of the full off-diagonal projector, useful as a
diagnostic. If the full QFM doesn't beat the diagonal surrogate, the
off-diagonal couplings are not contributing.

### Step 2.4 — comparison: no-Krylov vs Krylov on the same model

**Test name:** `no_krylov_ppl_within_2x_of_krylov_ppl`

**Setup:** train a model on shard 0 with both decoders (no-Krylov and
full Krylov). Compare in-sample and held-out ppl.

**Pass criteria:**
- The no-Krylov ppl is within 2× of the Krylov ppl (sanity: the
  Krylov machinery doesn't catastrophically hurt).
- The held-out ppl is at most 1.5× the unigram ppl (the model is
  learning something).

**Why this is a useful test:** quantifies the value of the Krylov
reduction. If the no-Krylov model is within 10% of the Krylov model,
the Krylov reduction is essentially cosmetic (and we can simplify the
architecture).

---

## Phase 3 — integrate oxieml decoder into QfmTextModel (TODO, ~1 hour)

### Step 3.1 — Decoder enum

Add `pub enum Decoder { Dense(DMatrix<Complex64>), Analytical { trees: Vec<EmlTree>, fallback: DMatrix<Complex64>, column_ok: Vec<bool> } }`
to `QfmTextModel`. The existing `next_token_dist` uses `Dense` (current
behavior). The `Analytical` path evaluates `Σ_j h_j · f_j(m_o)` for
each active mode `m_o`, falling back to `Dense` lookup for columns
where `column_ok[j] == false`.

**Test:** `decoder_enum_dispatch_correctly` — construct a model with
each decoder variant, verify the per-mode weights match within 1e-6
(the fallback is a no-op when `column_ok = [true; rank]`).

### Step 3.2 — wire `--oxieml-fit` into training

Add the `qfm_text_train` flag. After `QfmTextModel::from_accumulator`,
call `OxiemlDecoder::fit_basis(W, opts)` to get the trees, build the
`Analytical` decoder variant, and store it in the checkpoint.

**Test:** `train_with_oxieml_fit_produces_analytical_decoder` — train
on shard 0 with `--oxieml-fit`, load the checkpoint, verify
`model.decoder()` returns `Analytical { ... }` with non-empty trees.

---

## Phase 4 — end-to-end on shard 0 (TODO, ~30 min)

### Step 4.1 — train + evaluate, in-sample

Train rev 37 on shard 0 with `--oxieml-fit`, evaluate in-sample.

**Target:** in-sample QFM ppl within 5% of dense-W QFM ppl (SINDy
should be near-lossless at rank 2).

### Step 4.2 — train + evaluate, held-out

Same checkpoint, evaluate on `wikitext-103-test`.

**Target:** held-out QFM ppl within 10% of dense-W QFM ppl (some
SINDy error is expected on held-out).

### Step 4.3 — comparison table

| rev | encoder | decoder | in-sample ppl | held-out ppl | checkpoint MB |
|-----|---------|---------|---------------|--------------|---------------|
| 35  | OrderHasher | dense W | (existing) | (existing) | 33 (1M blocks) / 1.57 GB (full) |
| 36  | ContextRegistry | dense W | (existing) | (existing) | 92 (1 shard) / 1.5 GB (full) |
| 37  | OrderHasher | oxieml | (new) | (new) | target: < 1 MB |

If the oxieml decoder's in-sample ppl is within 5% of the dense-W
decoder, declare success.

---

## Phase 5 — end-to-end on full corpus (TODO, ~15 min wall + 30 min eval)

### Step 5.1 — full corpus train + eval

Train rev 37 on the full 128M corpus with `--oxieml-fit`. Compare to
rev 35 (m8_bigblocks_v2, 1.57 GB checkpoint) and rev 36 (if a
full-corpus rev 36 model exists).

**Target:** in-sample ppl within 5% of rev 35, checkpoint size < 10 MB
(160× reduction).

### Step 5.2 — inference benchmark

Measure inference wall time per token with the dense-W vs oxieml
decoders. Both should be < 100 μs (sparse `decode_sketched_at` is the
bottleneck, not the decoder).

---

## Decision criteria

- **If Phase 1 passes** (all 4 atomic tests pass), the architecture is
  correctly wired and we can trust the integration tests.
- **If Phase 2 shows no-Krylov ≈ Krylov**, the Krylov reduction is not
  contributing and we should simplify the architecture (drop SIRK
  entirely, just use `c_0^H W[m_o, :]`).
- **If Phase 4 shows oxieml within 5% of dense-W**, rev 37 ships.
- **If Phase 4 shows oxieml > 5% worse, OR most columns fail the
  residual check**, document the negative result and revert rev 37.
  Keep the `OxiemlDecoder` module for future experiments.
- **If Phase 5 shows the oxieml checkpoint is < 10 MB and inference
  is < 100 μs/tok**, rev 37 is the production design.

---

## Files

**Create:**
- `qfm_text/tests/atomic_components.rs` — Phase 1 tests.
- `qfm_text/tests/no_krylov_baseline.rs` — Phase 2 tests.
- `qfm_text/docs/rev37_oxieml_plan_v2.md` — this file.

**Modify:**
- `qfm_text/src/model.rs` — add `Decoder` enum (Phase 3.1).
- `qfm_text/src/bin/qfm_text_train.rs` — add `--oxieml-fit` flag
  (Phase 3.2).
- `qfm_text/Cargo.toml` — no new deps needed.

**Keep:**
- `qfm_text/src/oxieml_decoder.rs` — already done.
- `qfm_text/tests/oxieml_fit_real_w.rs` — already done.
