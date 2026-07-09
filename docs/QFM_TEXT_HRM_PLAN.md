# Plan: QFM-Text — train the QFM.tex architecture on the HRM-Text data + process

> **Executor note:** This plan is written to be executed stage-by-stage by a smaller LLM.
> Each stage has a goal, exact files, key signatures, and acceptance commands. Do stages in
> order. `$ROOT = /media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba`; the unfer repo is
> `$ROOT/unfer`. All Rust work is CPU-only (no CUDA needed); Python is used only for the
> HRM-Text data pipeline. **Never commit any file from the HRM-Text checkout into unfer**
> (house rule: third-party licensed content is fetched by URL + pinned commit + checksum,
> not vendored — even though HRM-Text is Apache-2.0).

## Goal

Train and evaluate a language model whose *model* is the QFM.tex architecture
(Tomographic Subspace Recovery pipeline, §"Tomographic Subspace Recovery" of `QFM.tex`:
CountSketch S₁ → feature-to-mode S₂ → exact rank-1 Mehler dressed-vacuum projector
generator (rev 31, `Operator::ProjectOnto`) → SIRK Krylov reduction → Born-rule decode →
optional Quantum Bayesian Update), and whose *data and training process* are those of
[sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text): its `data_io` corpus
cleaning/tokenization, its `sample_tokenized.py` stratified epoch sampling, its
epoch/checkpoint/eval cadence, and its perplexity-first evaluation.

### What HRM-Text does (facts gathered 2026-07-08)

- 1B-param hierarchical recurrent text model (two nested recurrent stacks: slow/abstract
  H-module + fast/detailed L-module), PyTorch + FSDP2, PrefixLM packed dataset,
  cross-entropy loss, `torchrun pretrain.py arch/size@arch=L lr=2.5e-4 global_batch_size=172032`.
- Data: companion `data_io` pipeline cleans + tokenizes a user-supplied corpus into
  `tokens.npy` + per-epoch index arrays; `sample_tokenized.py epochs=4 output_path=...`
  does stratified per-node sampling.
- Eval: `python -m evaluation.main ckpt_path=...` (perplexity + benchmark wrappers);
  checkpoints per epoch; EMA weights; W&B logging. License: Apache-2.0.

### The mapping (design decisions, already made — do not re-litigate)

QFM-TSR is non-neural: "training" is the **offline O(M) compile** (streaming accumulation
of flow-matching channel weights), not gradient descent. The adaptation is:

| HRM-Text concept | QFM-Text analog |
|---|---|
| token sequence corpus | stream of (context window, next token) pairs |
| embedding + H/L hierarchy | **hierarchical dressed vacua**: one exact projector per context order o ∈ {1..n} (order-1 ≈ slow/abstract H-module; order-n ≈ fast/detailed L-module) |
| forward pass | encode context → `e^{-iH_m t}` → Born-rule sketched probability |
| logits / LM head | per-mode next-token histograms, marginalized under the Born distribution |
| SGD epochs (epochs=4) | streaming accumulation passes over the same sampled shards |
| FSDP data parallelism | per-shard `ChannelAccumulator`s merged associatively (counts are a monoid) |
| learning rate / EMA | N/A (deterministic counts); smoothing hyperparams (t, λ_o, discount) tuned on validation |
| in-context adaptation | Quantum Bayesian Update (`bayes.rs`: Likelihood + HMC / `belief_propagation_chain`) on the observed prefix |
| checkpoint per epoch | serialized compiled pipeline artifact per pass |
| cross-entropy eval | token-level log-loss / perplexity on HRM-Text's held-out split |

**Architecture adaptations to QFM (all exact projectors — the rev 31 "no O(ε) truncation"
directive is binding):**

1. **Hierarchical multi-projector generator.** H = Σ_o λ_o |0̃_o⟩⟨0̃_o|, one dressed vacuum
   per context order o over that order's mode block. Each term is an exact rank-1
   `Operator::ProjectOnto`; the sum is Hermitian, rank ≤ n, Krylov dim ≤ n+1. The
   projectors share the Fock vacuum component, so evolution coherently mixes orders —
   the quantum analog of hierarchical reasoning / Katz backoff.
2. **Hashed (collision-accepting) S₂.** `FeatureToMode::register` assigns a fresh mode per
   unique feature → unbounded at corpus scale. New `HashedFeatureToMode`: mode =
   offset_o + (splitmix64(key, salt_o) mod K2_o). Fixed memory, no `K2Exhausted`.
3. **Streaming compile.** `QfmPipeline::compile(&[Vec<f64>], ...)` needs all M points in
   RAM and builds image observables (Φ, compressive solver) that text doesn't use. New
   `compile_from_channels` entry: consumes accumulated channel weights directly, builds
   the dressed vacua, runs the same SIRK reduction, builds only `W` and `W_prob`
   (skips `krylov_image_basis`/`compressive_solver`).
4. **Token decode head.** Decode stops at the sketched Born probability p̃ ∈ R^{K₂}
   (Phase 3 of QFM.tex); Phase 4's heavy-hitters/pixel render is replaced by histogram
   marginalization: P(y | context) = Σ_j p̃_j · hist_j(y), with absolute-discount escape
   to the unigram distribution (never zero probability).
5. **Superposition encode.** A query context enters as |Ψ_in⟩ = (1/√n) Σ_o |mode_o(ctx)⟩
   (one single-excitation component per order), replacing the image S₁→S₂ single-mode
   encode + nearest-feature fallback (a hashed mode always exists — no fallback needed).

**Honest scope (write this into the docs, keep it in the README of the crate):** this is a
quantum-kernel n-gram-family model with coherent Krylov smoothing across backoff orders.
The success criterion is *beating classical interpolated/backoff n-gram baselines at equal
context order on the same corpus*, and demonstrating the QFM pipeline end-to-end at
corpus scale (10⁷–10⁸ tokens, K₂ ~ 10⁵). It will not approach a 1B HRM-Text transformer;
that comparison is reported for honesty, not as a target.

### Model dimensions (defaults; all in config)

- Tokenizer: HRM-Text's tokenizer if its pinned commit documents one; otherwise train a
  16k-vocab BPE with HF `tokenizers` in Stage 0 (V=16384 keeps histograms compact).
- Context orders n = 4 (features: last 1, 2, 3, 4 tokens → 5-gram-class model).
- Mode blocks K2_o = 65536 per order → K₂ = 262144 total modes (+1 vacuum).
- Histogram cap T = 64 entries/mode (u32 token, u32 count) + escape count.
- Krylov: `max_rank = 8` (generator rank ≤ n=4 ⇒ Krylov dim ≤ 5; 8 gives headroom),
  shifts m = 8. λ_o defaults: uniform; evolution time t = 1.0; both swept in Stage 6.
- Corpus: WikiText-103 (~103M train tokens, standard public LM benchmark with published
  n-gram baselines) run through HRM-Text's `data_io`. A 10M-token slice is the CI-scale
  fixture. (WikiText-103 is CC-BY-SA: it is *data* fetched by script, never committed.)

---

## Stage 0 — HRM-Text checkout + data preparation (Python, no unfer code)

**Goal:** reproducible tokenized shards produced by the HRM-Text pipeline.

- New `$ROOT/unfer/qfm_text/` crate directory (workspace member added in Stage 1) with
  `scripts/` for the Python side. Nothing from HRM-Text is copied into the repo.
- `qfm_text/scripts/fetch_hrm_text.sh`: clones `https://github.com/sapientinc/HRM-Text`
  into sibling `$ROOT/HRM-Text` at a **pinned commit** (executor: `git ls-remote` the
  default branch head at execution time, record the hash in the script), verifies with
  `git rev-parse HEAD`. Idempotent (skip if already at pin). Same for the `data_io`
  companion repo if it is separate (check HRM-Text's README; pin it too).
- `qfm_text/scripts/prepare_corpus.sh`: downloads WikiText-103 (curl + sha256 check into
  `$ROOT/hrm_data/raw/`), runs HRM-Text `data_io` cleaning + tokenization to produce
  `tokens.npy`, then `python sample_tokenized.py epochs=4 output_path=$ROOT/hrm_data/sampled`.
  Use a venv (`uv venv` or `python -m venv`) with HRM-Text's requirements; document exact
  commands as executed. If `data_io` demands a tokenizer path, train the 16k BPE here
  (`qfm_text/scripts/train_tokenizer.py`, ~30 lines with HF `tokenizers`) and save to
  `$ROOT/hrm_data/tokenizer.json`.
- `qfm_text/scripts/export_shards.py` (our code): reads `tokens.npy` + per-epoch index
  arrays, writes `$ROOT/hrm_data/shards/epoch{E}/shard{S}.bin` — raw little-endian u32
  token ids — plus `manifest.json`: `{vocab_size, n_shards, tokens_per_shard, sha256s,
  train/valid/test split paths, tokenizer_sha256, hrm_text_commit}`. Also emits the
  10M-token `ci_slice/` and a ~200k-token `qfm_text/testdata/tiny_fixture.bin` +
  `tiny_manifest.json` (committed: it must be derived from WikiText-103's *test* split —
  small, and license-compatible for test-fixture use; note the CC-BY-SA attribution in
  `qfm_text/testdata/README.md`).
- **Fallback gate:** if HRM-Text's `data_io` cannot run (undocumented deps, needs GPUs,
  corpus format mismatch), record the exact blocker in `docs/QFM_TEXT_STATUS.md` and let
  `export_shards.py` tokenize WikiText-103 directly with the Stage-0 BPE, keeping the
  same shard/manifest format and the epochs=4 stratified sampling semantics (shuffle with
  fixed seed per epoch). The rest of the plan is unaffected — the shard format is the
  interface.
- **Accept:** `bash qfm_text/scripts/prepare_corpus.sh` completes; `manifest.json` lists
  ≥ 4 epoch dirs with matching sha256s; fixture committed; `python -c` spot-check that
  detokenized fixture text is readable English.

## Stage 1 — `qfm_text` crate skeleton: shard reader + windowing

**Goal:** memory-mapped corpus access in Rust.

- Add `qfm_text` to `$ROOT/unfer/Cargo.toml` workspace members. Deps: `qfm`,
  `nested_fock_algebra`, `fock_sirk`, `nalgebra`, `num-complex`, `serde`, `serde_json`,
  `bincode`, `memmap2`, `rayon`, `thiserror`, `rustc-hash`. No candle.
- `qfm_text/src/corpus.rs`:
  - `pub struct Manifest { vocab_size: u32, shards: Vec<ShardEntry>, ... }` (serde, mirrors
    Stage-0 `manifest.json`).
  - `pub struct Shard(memmap2::Mmap)` with `pub fn tokens(&self) -> &[u32]` (bytemuck-free:
    validate length % 4 == 0, use `chunks_exact(4)` + `u32::from_le_bytes` behind an
    iterator, or `align_to` with a misalignment error).
  - `pub struct WindowIter<'a>` yielding `(context: &'a [u32], next: u32)` for each
    position with `context = &tokens[i.saturating_sub(n_max)..i]`, `next = tokens[i]`
    (shorter contexts near shard starts are fine — orders beyond `context.len()` are
    simply absent for that window).
- `qfm_text/src/error.rs`: `pub enum QfmTextError` (thiserror): `BadManifest`,
  `BadShard { path, reason }`, `VocabMismatch`, `Qfm(#[from] qfm::QfmError)`, `Io(...)`.
- **Tests:** fixture shard round-trip (token count, first/last tokens vs. values recorded
  in `tiny_manifest.json`); `WindowIter` yields exactly `len` windows; contexts near the
  boundary have the right lengths.
- **Accept:** `cargo test -p qfm_text`.

## Stage 2 — Hashed S₂ + streaming ChannelAccumulator

**Goal:** the O(M) offline pass, as pure counting (no Fock objects yet).

- `qfm_text/src/features.rs`:
  - `pub struct OrderHasher { n_orders: usize, block_sizes: Vec<u32>, salts: Vec<u64> }`
  - `pub fn mode_for(&self, order: usize, context: &[u32]) -> Option<u32>` — `None` when
    `context.len() < order`; otherwise `offset_o + (splitmix64_seq(last o tokens, salt_o)
    % block_sizes[o])`. Reuse the splitmix64 already used by `qfm::CountSketch` (copy the
    3-line mixer locally or expose it from `qfm::sketch` — prefer exposing
    `pub fn splitmix64(x: u64) -> u64` from `qfm/src/sketch.rs`).
  - `pub fn encode_modes(&self, context: &[u32]) -> Vec<u32>` — the ≤ n active modes.
- `qfm_text/src/accumulate.rs`:
  - `pub struct ModeStats { pub weight: u64, pub hist: Vec<(u32, u32)>, pub escape: u64 }`
    — `hist` is top-T by count (T from config; on overflow, evict the min-count entry and
    add its count to `escape`; deterministic tie-break by token id).
  - `pub struct ChannelAccumulator { stats: FxHashMap<u32, ModeStats>, unigram: Vec<u64>,
    total_windows: u64, config: TextConfig }`
  - `pub fn observe(&mut self, modes: &[u32], next: u32)` — for each active mode:
    `weight += 1`, histogram add; also `unigram[next] += 1`.
  - `pub fn merge(&mut self, other: ChannelAccumulator)` — associative + commutative
    (weights and counts add; histograms re-capped after merge). This is the FSDP analog.
  - `pub fn accumulate_shards(paths: &[PathBuf], cfg: &TextConfig) -> Result<ChannelAccumulator, QfmTextError>`
    — rayon `par_iter` over shards, one accumulator each, `reduce` by `merge`.
  - `pub struct TextConfig { n_orders, block_sizes, salts, hist_cap, max_rank, m_shifts,
    lambda: Vec<f64>, t: f64, discount: f64, seed }` (serde; loaded from TOML in Stage 6).
- **Tests:** hand-built 12-token toy corpus — exact expected weights/histograms per mode;
  `merge` equals single-pass on concatenated corpus (property test with two splits);
  histogram cap eviction moves mass to `escape`, total count conserved.
- **Accept:** `cargo test -p qfm_text accumulate`.

## Stage 3 — Multi-projector generator builder in `nested_fock_algebra`

**Goal:** H = Σ_o λ_o |0̃_o⟩⟨0̃_o| from channel weight lists, all exact `ProjectOnto`.

- `nested_fock_algebra/src/models.rs`: the private `dressed_vacuum_projector(channels, c0)`
  already builds one exact projector term. Add:
  ```rust
  pub fn qfm_hamiltonian_hierarchical_projectors(
      groups: &[(f64, Vec<(u32, f64)>)],   // per order: (lambda_o, [(mode, alpha_j>0)…])
  ) -> Hamiltonian
  ```
  For each group: εⱼ = αⱼ/√(1+Σα²), c₀ = 1/√(1+Σα²) (the rev 31 α→ε normalization, same
  as `build_flow_hamiltonian`), one `ProjectOnto` term via `dressed_vacuum_projector`
  with the group's index channels, term coefficient λ_o. Sum of terms in one
  `Hamiltonian`. Panic (assert) on non-finite/negative α or λ, or duplicate modes within
  a group (modes may repeat *across* groups — blocks make them disjoint in practice, but
  don't enforce that here).
- **Tests** (in `nested_fock_algebra/src/unit_tests.rs`):
  - Two groups, 2 channels each: matrix elements ⟨x_i|H|vac⟩ = Σ_o λ_o c₀^(o) ε_i^(o) on
    probe states; Hermiticity ⟨u|Hv⟩ = ⟨Hu|v⟩ on random sparse probes.
  - Single group with λ=1 reproduces `qfm_hamiltonian_mehler_projector` matrix elements.
  - H² = λH holds per isolated group (idempotence up to λ) when groups don't overlap and
    cross-terms vanish on group-local probes — check the concrete 2×2 closed form instead
    of a general claim: for one group, ⟨0̃|H²|0̃⟩ = λ²⟨0̃|H|0̃⟩/λ.
- **Accept:** `cargo test -p nested_fock_algebra hierarchical`.

## Stage 4 — `compile_from_channels` + token decode in `qfm`

**Goal:** the TSR compile path that consumes channels instead of training vectors, and a
Born-rule token head.

- `qfm/src/pipeline.rs`: refactor `QfmPipeline::compile` minimally so the SIRK/whiten/
  `probability_weight_matrix` middle section is a private helper
  `fn reduce_and_project(h: &Hamiltonian, k2: usize, cfg: &QfmConfig) -> Result<(W, H_m, w_prob), QfmError>`
  (executor: extract exactly the code between seed construction and the Φ build; do not
  change its behavior — existing tests are the guard). Then add:
  ```rust
  pub struct ChannelSpec { pub lambda: f64, pub channels: Vec<(u32, f64)> } // alphas
  pub fn compile_channels(groups: &[ChannelSpec], k2_total: usize, config: &QfmConfig)
      -> Result<QfmPipeline, QfmError>
  ```
  Builds the generator via `qfm_hamiltonian_hierarchical_projectors`, seeds SIRK with the
  uniform vacuum + single-excitation superposition over the modes that actually appear
  (plus vacuum), uses the existing `max_rank` rank-truncation path (`config.max_rank =
  Some(...)` required here — return the existing typed error if absent), and stores
  `phi`/`compressive` as `None` (make those fields `Option`; the image `decode` returns
  the existing dimension/config error when they're absent — grep tests that construct
  `QfmPipeline` directly).
- `qfm/src/pipeline.rs` additions:
  - `pub fn encode_modes(&self, modes: &[u32]) -> Result<DVector<Complex64>, QfmError>` —
    c₀ = W† (1/√n Σ|mode⟩): sum of ≤ n rows of W̄, no S₁/S₂/nearest-fallback.
  - `pub fn decode_sketched(&self, c_1: &DVector<Complex64>) -> Vec<f64>` — Phase 3 only:
    ρ_flat = Re vec(c₁c₁†), p̃ = W_prob ρ_flat, clamp negatives to 0, renormalize to sum 1
    (document why: rank truncation can leave tiny negative values).
- `qfm_text/src/model.rs`:
  ```rust
  pub struct QfmTextModel { pipeline: QfmPipeline, hasher: OrderHasher,
      mode_hists: FxHashMap<u32, ModeStats>, unigram: Vec<f64>, cfg: TextConfig }
  pub fn from_accumulator(acc: ChannelAccumulator, cfg: &TextConfig) -> Result<Self, QfmTextError>
      // alphas: alpha_j = weight_j / total_windows (QFM.tex eq: ᾱ_j = ‖x_j‖²/M with unit-norm channels)
  pub fn next_token_dist(&self, context: &[u32]) -> Result<Vec<f64>, QfmTextError>
      // encode_modes → evolve(c0, cfg.t) → decode_sketched → P(y) = Σ_j p̃_j · smooth(hist_j)(y)
      // smooth: absolute discounting `discount` per seen token, escape+discount mass → unigram
  pub fn logprob(&self, context: &[u32], next: u32) -> Result<f64, QfmTextError>
  ```
  Serialization: `pub fn save(&self, path) / load(path)` with bincode + a version byte
  (serialize W, H_m, W_prob as (nrows, ncols, Vec<f64/Complex64>) — add
  `#[derive(Serialize, Deserialize)]` only inside qfm_text wrapper types; do not add
  serde to `qfm` itself).
- **Tests:**
  - Toy 2-order, 3-mode corpus: `next_token_dist` sums to 1 (1e-12); every token has
    P > 0; at t = 0 the distribution equals the pure mixture of the context's own mode
    histograms (evolution is identity), a closed-form check.
  - t = 2π coherent return ≈ t = 0 distribution (rev 31 closed-form, tolerance 1e-6).
  - Serialization round-trip: identical `logprob` on 100 fixture windows.
- **Accept:** `cargo test -p qfm_text model && cargo test -p qfm` (existing image tests
  stay green — the `Option` refactor must not break them).

## Stage 5 — Autoregressive scoring, generation, in-context Bayes

**Goal:** the LM surface: perplexity, sampling, and the quantum in-context update.

- `qfm_text/src/lm.rs`:
  - `pub fn perplexity(model: &QfmTextModel, shard: &Shard) -> PerplexityReport` —
    `PerplexityReport { n_tokens, nll_nats_per_token, ppl }`; rayon over windows.
  - `pub fn sample_text(model: &QfmTextModel, prompt: &[u32], n_tokens: usize,
    temperature: f64, seed: u64) -> Vec<u32>` — temperature on log-probs, deterministic
    RNG (splitmix64-seeded).
  - Baselines for honesty, same crate: `pub struct NgramBaseline` — interpolated
    absolute-discount n-gram with the *same* orders/hashing/histogram caps (so the delta
    isolates the quantum smoothing), and the unigram floor. `pub fn perplexity_baseline(...)`.
- `qfm_text/src/incontext.rs` (the Quantum Bayesian Update as in-context adaptation,
  QFM.tex §"Quantum Bayesian Updating"):
  - `pub fn adapt_prior(model: &QfmTextModel, prefix: &[u32], opts: &HmcOpts)
    -> DVector<Complex64>` — build one `Likelihood::from_krylov_state(encode_modes(w))`
    per sliding window w of the prefix (cap at last 32 windows), posterior =
    `Posterior::new(likelihoods, tsr_evolved_prior(&pipeline))`, `sample_hmc_single`.
  - `pub fn next_token_dist_adapted(model, prefix) -> Vec<f64>` — decode_sketched on the
    posterior sample instead of the evolved encode; mix with the static dist 50/50
    (config `bayes_mix`).
- **Tests:** on the tiny fixture — QFM ppl and baseline ppl both finite and < unigram
  ppl; `sample_text` deterministic across two runs with equal seed; `adapt_prior` on a
  repetitive prefix ("a b a b a b") strictly increases P(next = the repeating
  continuation) vs. the unadapted distribution.
- **Accept:** `cargo test -p qfm_text lm incontext`.

## Stage 6 — Training + eval binaries (the HRM-Text process parity)

**Goal:** the `pretrain.py` / `evaluation.main` analogs, config-driven.

- `qfm_text/src/bin/qfm_text_train.rs`:
  - CLI: `qfm_text_train --config train.toml`. TOML = `TextConfig` + `{ manifest_path,
    epochs (default 4, mirroring HRM-Text), out_dir, threads }`.
  - Loop per epoch: `accumulate_shards(epoch_shards)` → `merge` into the running
    accumulator → `QfmTextModel::from_accumulator` → save
    `out_dir/checkpoint_epoch{E}.qfm` (the per-epoch checkpoint analog) → append one
    NDJSON line to `out_dir/metrics.ndjson`: `{epoch, wall_s, n_windows, n_active_modes,
    valid_ppl}` (valid_ppl on the manifest's valid split; NDJSON is the W&B analog and
    matches the unfer protocol style).
  - Multiple epochs genuinely differ because HRM-Text's stratified sampling produces
    different shard content per epoch; counts keep accumulating (more data ⇒ better
    histograms — the "training curve").
- `qfm_text/src/bin/qfm_text_eval.rs`:
  - `qfm_text_eval --checkpoint ....qfm --manifest ... [--split test] [--sweep]`
  - Reports: QFM ppl, NgramBaseline ppl, unigram ppl, 5 sampled continuations from fixed
    prompts. `--sweep` grid-searches `t ∈ {0.25, 0.5, 1.0, 1.5, 2.0}` × λ profiles
    `{uniform, ∝order, ∝1/order}` × `discount ∈ {0.5, 0.75}` on the *valid* split,
    prints the winner, and (if better) re-reports test ppl at the winning config — this
    is the entire "hyperparameter training" of the non-neural model.
- `qfm_text/README.md`: how to run Stages 0→6 end-to-end; the mapping table from this
  plan's preamble; the honest-scope paragraph.
- **Accept:** on the 10M-token `ci_slice`: `qfm_text_train` completes 4 epochs in
  < 30 min on CPU; `qfm_text_eval` prints finite ppls with QFM ppl ≤ NgramBaseline ppl
  on ≥ 1 sweep config (if not, that's a *reportable negative result* — record it in
  `docs/QFM_TEXT_STATUS.md` with the numbers, don't hide it); metrics.ndjson has 4 lines
  with non-increasing valid_ppl.

## Stage 7 — Full-corpus run, HRM-Text comparison, documentation

**Goal:** the real run + write-up.

- Full WikiText-103 run (all epochs, K₂ = 4×65536): record wall time, RAM, ppl per epoch
  in `docs/QFM_TEXT_STATUS.md`. Published classical reference points for WikiText-103
  (e.g. KN 5-gram ≈ ppl 150–230 depending on setup) go in the table for context; our
  hashed/capped baseline is the like-for-like one.
- Optional, GPU-gated: run HRM-Text's own smallest config on the same corpus slice via
  its documented `torchrun` command and report its ppl in the same table (as the
  "ceiling" reference). Skip cleanly if no GPU: the comparison row reads "not run (no
  GPU)".
- `QFM.tex`: new subsection at the end of the TSR section — "QFM-Text: hierarchical
  dressed vacua on token streams (rev 32)": the multi-projector generator equation, the
  α→ε normalization per order, the Born-rule token head, the honest-scope paragraph, and
  the measured numbers. Add a `\date` revision entry. Keep the rev 31 exact-projector
  language: every generator here is a sum of exact `ProjectOnto` terms.
- Update `docs/IMPLEMENTATION_PLAN.md` current-status header with a one-paragraph rev 32
  pointer to this plan and status doc; update memory (`unfer-modular-kernel-plan.md`).
- **Accept:** `cargo test --workspace` green; QFM.tex compiles (structure check: balanced
  braces/environments as in rev 31); status doc has the full-run table.

## Final verification (run all)

1. `cargo test --workspace` in `$ROOT/unfer` (CPU).
2. `bash qfm_text/scripts/prepare_corpus.sh` idempotent re-run (checksums match).
3. `qfm_text_train` + `qfm_text_eval` on `ci_slice` (Stage 6 acceptance).
4. Full-corpus numbers recorded in `docs/QFM_TEXT_STATUS.md`.

## Risks & mitigations

- **HRM-Text `data_io` may not run locally** — Stage 0 fallback gate keeps the shard
  format as the interface; everything downstream is unaffected.
- **Rank collapse** (rev 31 finding: a single exact projector gives a 2D Krylov space) —
  by design the generator here is rank-n (n=4 orders) ⇒ Krylov dim ≤ 5; capacity lives
  in the histograms, and the model at t=0 degrades exactly to the classical mixture, so
  the quantum layer can only be compared, never blamed silently. The t/λ sweep is the
  control.
- **Hash collisions at K₂=65536/order** — collisions blend histograms of unrelated
  contexts (standard hashed-LM trade-off); block sizes are config, Stage 7 can double
  them; the baseline uses the *same* hashing so comparisons stay fair.
- **Histogram memory** — worst case 4×65536 modes × (64×8 B + overhead) ≈ 150–200 MB.
  Fine. `hist_cap` is config.
- **SIRK cost at K₂ = 262144** — the ProjectOnto fast path is O(active components) per
  matvec and the seed has ≤ K₂+1 components; m=8 shifts ⇒ well under a minute. If the
  Gram/whiten step struggles, `max_rank` truncation is already the escape hatch.
- **Licensing** — HRM-Text (Apache-2.0) and WikiText-103 (CC-BY-SA) are fetched by
  pinned URL + checksum, never committed, per the house rule; the committed test fixture
  derives from the WikiText-103 test split with attribution in `testdata/README.md`.
