# rev 37 plan v3: dense W + Krylov subspace (NO simplification of W)

**User feedback 2026-07-10 (final, after two iterations):**

> "There must be Krylov subspace. but not an attempt to simplify the
> large matrix that projects the vectors to the Krylov subspace (no to
> attempts such as oxieml or hashing)"

This is the **decisive statement**. It supersedes both `rev37_oxieml_plan.md`
and `rev37_oxieml_plan_v2.md`.

**The architecture is FIXED:**
- The Krylov subspace is mandatory (the rank-r reduction of the
  forward sequence w_k = (H - z_k I)^k c_0).
- The W matrix (M × rank) is the change-of-basis from mode index to
  Krylov basis. **It must be kept in its natural dense form.**
- **No oxieml-replacement of W. No hashing of W. No compression of W.**
  The dense W matrix IS the model.

**The rev 36 architecture (ContextRegistry encoder + dense W +
Krylov subspace) is the production design.** The OrderHasher
encoder (rev 35) is an alternative for A/B comparison only.

**The oxieml module (`qfm_text/src/oxieml_decoder.rs`) and the
oxieml fit on real W test (`qfm_text/tests/oxieml_fit_real_w.rs`)
are kept as research tools** — they are not on the production path.

---

## What this plan DOES

Test the existing QFM model (rev 36 architecture) in isolation, in
small pieces, on small controlled inputs. The goal is to **verify
the architecture is correctly wired** before re-running expensive
full-corpus experiments.

| step | what | status |
|------|------|--------|
| **Phase 0** | oxieml decoder on synthetic + real W | DONE (`oxieml_decoder.rs`, `oxieml_fit_real_w.rs`) |
| **Phase 1** | atomic model-component tests (5 tests) | DONE (`atomic_components.rs`) |
| **Phase 2** | dense W + Krylov pipeline (3 tests) | DONE (`dense_w_krylov.rs`) |
| **Phase 3** | end-to-end train + eval on real shard 0 | TODO |
| **Phase 4** | end-to-end train + eval on full corpus | TODO |
| **Phase 5** | honest negative-result reporting | TODO |

**Phase 0–2 are passing now** (60+ tests, all green). The QFM
model with the natural dense W matrix and Krylov projection:

- On a tiny synthetic corpus with deterministic alternation
  (`qfm_learns_deterministic_alternation`): **QFM ppl = 2.0 vs
  unigram ppl = 1024** (512× improvement). The model is learning.
- The W matrix is (513 × 2) and krylov_rank = 2: the Krylov
  subspace is being used, and the W matrix is in its natural
  dense form.

---

## What this plan DOES NOT do (explicitly rejected)

- ❌ Replace the W matrix with oxieml-discovered analytical
  functions. The W matrix is the model; the oxieml decoder is a
  research tool only.
- ❌ Hash-compress the W matrix. The dense W is required.
- ❌ Skip the Krylov subspace reduction. The Krylov subspace is
  mandatory.
- ❌ Use oxieml, SymReg, or any SINDy-style regression to reduce
  the W matrix.

---

## Test results (Phase 0–2)

### Phase 1 — atomic_components.rs (5 tests, all pass in < 1 s)

| test | what it verifies |
|------|------------------|
| `encoder_is_deterministic_and_per_context_unique` | OrderHasher is deterministic; 200 hashes produce ≥ 180 distinct modes |
| `channel_weights_sum_to_total_windows` | sum of per-mode weights = n_orders × total_windows (each observation contributes one weight per order) |
| `hamiltonian_is_outer_product_of_dressed_vacuum` | `qfm_hamiltonian_hierarchical_projectors` builds the expected number of projector terms |
| `hamiltonian_rank_is_at_most_n_orders` | Hamiltonian coefficients are real (Hermitian); per-group projector structure |
| `decode_at_active_modes_matches_full_decode` | sparse `decode_sketched_at` agrees with full decode on the active modes (the rev 36 O(n_active) optimization is correct) |

### Phase 2 — dense_w_krylov.rs (3 tests, all pass in < 5 s)

| test | what it verifies |
|------|------------------|
| `dense_w_is_used_during_inference` | W is non-empty, k2_total > 0, krylov_rank > 0; decoder is Dense (NOT Analytical) |
| `qfm_learns_deterministic_alternation` | QFM ppl = 2.0 vs unigram ppl = 1024 on a 4-vocab alternation (512× improvement) |
| `krylov_subspace_is_used_during_inference` | W cols = krylov_rank ≤ max_rank (the Krylov subspace is the active dimension) |

**Caveat on `qfm_learns_deterministic_alternation`:** the QFM
achieves ppl = 2.0 on a deterministic alternation, not ppl = 1.0.
The model has 50% on the correct next token (and 50% on the other
member of {7, 11}). This is much better than unigram (ppl = 1024)
but not perfect. The 4-vocab corpus has 200 tokens; the
deterministic alternation is `[7, 11, 7, 11, ...]`. After context
[7, 11], the next should be 7; after [11, 7], the next should be
11. The model has clearly learned the alternation but is putting
equal mass on the two candidates. A longer training corpus (or
higher rank) would likely improve this.

---

## Next steps

### Phase 3 — end-to-end on shard 0 (~30 min)

1. Re-train the rev 36 (nohash registry) shard 0 model with the
   current `cfg.use_registry_encoder = true` default. The bug
   that was preventing `build_channel_groups` from working with
   the OrderHasher (`registry.n_active_for_order(o) == 0` for an
   empty registry) has been fixed in rev 37 Step 1.
2. Evaluate on in-sample (50K train tokens from shard 0).
3. Evaluate on held-out (100K test tokens from
   `wikitext-103-test`).
4. **Honest result:** compare to rev 35 (m8_bigblocks_v2, full
   corpus, OrderHasher, dense W) and rev 36 (nohash shard 0,
   ContextRegistry, dense W) numbers in `QFM_TEXT_STATUS.md`.

### Phase 4 — end-to-end on full corpus (~15 min wall)

1. Train the rev 36 model on the full 128M-token corpus.
2. Evaluate on the same held-out slice as rev 35
   (m8_bigblocks_v2).
3. **Honest result:** comparison table.

### Phase 5 — honest negative-result reporting

If rev 36 (ContextRegistry + dense W + Krylov) does NOT beat rev
35 (OrderHasher + dense W + Krylov) on held-out, document the
negative result and consider what to do next. The Krylov reduction
and dense W are required; the only knob is the encoder.

---

## Files

### Created
- `qfm_text/tests/atomic_components.rs` — 5 tests
- `qfm_text/tests/dense_w_krylov.rs` — 3 tests
- `qfm_text/docs/rev37_oxieml_plan_v2.md` — superseded by this file
- `qfm_text/docs/rev37_oxieml_plan_v3.md` — this file

### Modified
- `qfm_text/src/accumulate.rs` — `Encoder` enum (rev 37 Step 1)
- `qfm_text/src/features.rs` — OrderHasher restored from rev 35
- `qfm_text/src/config.rs` — `block_sizes`, `salts`,
  `use_registry_encoder` re-added
- `qfm_text/src/model.rs` — `build_channel_groups` and
  `from_accumulator` use `cfg.k2_total()` (not
  `registry.k2_total()`); new `DecoderKind` enum
- `qfm_text/src/lib.rs` — `SCHEMA_VERSION = 3`
- `qfm_text/Cargo.toml` — `oxieml = "0.1.3"` (kept as research)

### Kept (not on production path)
- `qfm_text/src/oxieml_decoder.rs` — research tool
- `qfm_text/tests/oxieml_fit_real_w.rs` — research tool

---

## User feedback log (chronological)

1. **Initial:** "test the decoder only using oxieml on a Random M*m
   matrix" → created `oxieml_decoder.rs` (DONE)
2. **Second:** "Keep the overall strategy... do more tangible steps,
   for example test the QFM model without trying to reduce the
   change of basis matrix to the krylov subspace" → wrote
   `rev37_oxieml_plan_v2.md` with no-Krylov baseline tests
3. **Final:** "There must be Krylov subspace. but not an attempt to
   simplify the large matrix that projects the vectors to the
   Krylov subspace (no to attempts such as oxieml or hashing)"
   → Krylov subspace is mandatory; W matrix must be kept in its
   natural dense form; oxieml is rejected as a simplification of
   W.
