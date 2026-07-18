# PLAN A — unfer (probability kernel + QFM research)

Parallel workstream 1 of 3. Companion plans: `australVM/PLAN_parallel_australvm.md`,
`velysterm/PLAN_parallel_velysterm.md`.

## System context

Three repos form one system:
- **unfer** (this repo) — the kernel: `prob_kernel::Session` (Born-rule API), `unfer_ffi`
  (18 `uk_*` + 5 `uz_*` C symbols), `unfer_protocol` (serde + UK-#### codes), `qfm`/`qfm_text`,
  6 Austral modules.
- **australVM** — Austral JIT (`safestos/cranelift`) that statically links `unfer_ffi` via a
  **path dependency** and registers the `uk_*` symbols; hosts the Austral modules (modhost).
- **velysterm** — editor/agent frontend; `kernel_client` path-depends on `prob_kernel` +
  `unfer_protocol`; ships the `unfer_agent` NDJSON binary (11 ops).

Because both dependents read unfer's **working tree** (path deps), any uncommitted breakage
here propagates to both. Keep this repo green at all times.

## Parallel-execution rules (all three plans share these)

1. **Ownership**: modify only files inside this repo. Cross-repo *reads* are fine.
   Cross-repo *writes* are forbidden, except steps explicitly marked `[SYNC]`.
2. **Frozen contract** (additive-only changes allowed; no renames/removals/signature changes):
   - the 18 `uk_*` and 5 `uz_*` symbols and their C signatures;
   - `prob_kernel::Session` public API;
   - `unfer_protocol` serde types and UK-#### code assignments;
   - the 11 NDJSON agent ops; `module.toml` grant vocabulary.
3. **Commit discipline**: meaningful messages (the last 10 commits are all `"a"` — stop that).
   Commit after every completed stage so dependents can pin.
4. Stages are ordered small → large; each ends in a verifiable acceptance command. Partial
   completion is fine — do not start a later stage until the earlier ones pass.

## Current state (2026-07-18)

- Working tree was broken mid-Pauli–Grover refactor; `prob_kernel/src/build.rs` and
  `qfm_text/src/model.rs` have just been fixed (`..Default::default()` + stale 13-arg
  `compile_channels` call). `cargo check --workspace` now passes.
- **16 modified + 2 untracked files are uncommitted** (the Pauli–Grover Hamiltonian,
  `random_start`, `HamiltonianType`, parity/MNIST test sweeps, QFM.tex §Pauli–Grover).
- australVM and velysterm build again against this tree.

---

## Stage A1 — Commit the in-flight work (S)

1. `git status`, `git diff --stat`. Review all hunks belong to the Pauli–Grover feature.
2. Run `cargo test --workspace` (CPU). Fix any failure before committing.
3. Commit in logical units with real messages, e.g.:
   - `qfm: add HamiltonianType::{Diffusion,PauliGrover} + pauli_grover_a + random_start`
   - `qfm/tests: PG parity + MNIST m-sweeps (vacuum vs random start)`
   - `QFM.tex: Pauli–Grover section`
   - `prob_kernel,qfm_text: fix QfmConfig initializers for new fields`
4. Do **not** squash or rewrite the existing `"a"` history (others may have pulled).

**Acceptance**: `git status --short` empty; `cargo test --workspace` green.

## Stage A2 — Doc-drift sweep (S)

The contract grew; the docs didn't. Fix in one pass:

1. `AGENTS.md`: "14 `uk_*` functions" → 18 `uk_*` + 5 `uz_*` (zenodo, feature-gated).
   Add `qfm_text`, `unfer_edge`, `unfer_nixvm` to the crate layout. Add a bullet on
   `HamiltonianType`/`pauli_grover_a`/`random_start` (defaults preserve old behavior).
2. `docs/MODULE_RECIPE.md`: the `[grants] kernel = [...]` example lists 12 symbols — update
   to all 18 (`uk_snapshot, uk_restore, uk_subscribe, uk_poll, uk_bayesian_update,
   uk_belief_propagation` are missing).
3. `docs/PROTOCOL.md`: documents 8 agent ops; `unfer_agent` (velysterm) implements 11.
   Document `save_session`, `restore_session`, `poll_events` (request/response shapes,
   UK codes, bounded 64-event queue semantics), noting the implementation lives in
   `velysterm/crates/kernel_client/src/bin/unfer_agent.rs`. Do **not** document
   `bayesian_update`/`belief_propagation` agent ops yet — velysterm Plan C2 adds them;
   a `[SYNC]` step there will hand you the spec fragment.
4. `docs/ARCHITECTURE.md`: crate diagram predates `qfm`, `qfm_text`, `unfer_edge`,
   `unfer_nixvm`, and the 6 modules — redraw.

**Acceptance**: `grep -n "14 uk" AGENTS.md docs/*.md` empty; `grep -c uk_ docs/MODULE_RECIPE.md`
≥ 18; every op in velysterm's `VALID_OPS` (minus the two Plan C2 adds) appears in PROTOCOL.md.

## Stage A3 — FFI symbol CI gate (S)

Today CI asserts only 5 of 18 `uk_*` exports.

1. Extend the `ffi-symbols` CI job (or add `unfer_ffi/tests/symbols.rs` invoked from CI) to
   assert the full set: build the cdylib, `nm -D --defined-only` (or `objdump -T`), compare
   against a checked-in `unfer_ffi/EXPECTED_SYMBOLS.txt` (18 `uk_*`; plus 5 `uz_*` under
   `--features zenodo`).
2. Make the test fail with a diff message that says exactly what to update.

**Acceptance**: deleting one `pub extern "C" fn uk_*` makes the gate fail; CI green on main.

## Stage A4 — Extract `module_builder` tool (M)

`docs/BUILD_PIPELINE.md` marks this TBD; 6 near-identical `build.sh`/`run_demo.sh` exist.

1. Create `unfer_module_builder` (small Rust bin or a single portable bash script in `tools/`):
   inputs = module dir; steps = parse `module.toml` → copy Austral cell → invoke australVM
   modhost build → run positive + UK-4001 negative smoke test.
2. Port all 6 modules (`demo_module`, `qfm_module`, `bayes_update_module`,
   `iterated_bayes_module`, `qfm_tomo_module`, `zenodo_store_module`) onto it; delete the
   per-module scripts. Add the missing `zenodo_store_module/build.sh` equivalent.
3. Add `zenodo_store_module` to CI behind a mock HTTP server (no network/credentials).

**Acceptance**: `tools/module_builder run demo_module` passes positive + negative tests;
`grep -r run_demo.sh --include='*.yml' .github/` shows only the builder; CI covers 6/6 modules.

## Stage A5 — Property & fuzz tests (S–M)

Zero property/fuzz coverage today on the most bug-prone surfaces.

1. `nested_fock_algebra`: proptest round-trips — `adjoint(adjoint(op)) == op`;
   `apply` distributes over addition; bounded expansion respects `SirkOpts` caps; vacuum
   initialization invariant from AGENTS.md.
2. `unfer_protocol`/`unfer_edge`: fuzz (cargo-fuzz or proptest byte-shredding) the NDJSON/JSON
   envelope parsing and the edge op-allowlist + secret-masking filter; UK-1001 must be the
   only parse-failure path, never a panic.
3. Fix the warnings left in `qfm_text` (dead `build_channel_groups`, unused vars) — either
   wire them up or remove them.

**Acceptance**: new tests run in `cargo test --workspace`; a deliberately swapped adjoint
rule fails the property test; no panics under 10k fuzz iterations.

## Stage A6 — Pauli–Grover research follow-through (M–L, research)

Context: PG without kernel gives 100% training but chance generalization; diffusion +
distributed multi-mode encoding generalizes. Open questions, in priority order:

1. **Kernel coupling for PG**: the Gram inner product already supports the SparseKernel;
   extend `dense_pauli_grover_matvec` with the off-diagonal kernel terms and re-run the
   parity held-out sweep (`qfm/tests/large_parity_classification.rs`). Hypothesis: kernel
   lifts PG generalization above chance.
2. **`a` sweep**: 0.5–1.0 grid on parity + MNIST; document whether the residual |0⟩
   component helps held-out accuracy (update QFM.tex §Pauli–Grover with results).
3. **CIFAR PG test**: port the existing CIFAR-10 fixture path to PG.
4. **qfm_text GPU decode** (L): port `decode_sketched` to the candle CUDA path already in
   `fock_sirk` — the ~140 h CPU wall blocks the rev-37 v3 diffusion-Hamiltonian evaluation,
   the project's central open question (see `QFM_TEXT_STATUS.md`).

**Acceptance**: each item lands as a committed test + a short results paragraph in
`QFM.tex` / `QFM_TEXT_STATUS.md` (negative results count — record them honestly as before).

## Stage A7 — Consistency cleanup (S)

1. `unfer_edge`: align `edition = "2024"` with the workspace; consider gating the heavy
   Pingora deps out of the default workspace build.
2. Document (or script) cleanup of `qfm_text_runs/` checkpoints (multi-hundred-MB dirs).
3. `docs/BUILD_PIPELINE.md`: remove the TBD now that A4 exists; cross-link the builder.

**Acceptance**: `cargo check --workspace` warning-free except upstream deps; CI green.

---

## Out of scope for this plan (owned by the other workstreams)

- australVM: live UK-4001 enforcement, cps.rs tests, bridge build robustness, symbol
  auto-sync test (it reads this repo but writes only there).
- velysterm: `unfer_agent` new ops (`bayesian_update`, `belief_propagation`), worker
  lifecycle, frontends.

`[SYNC]` at the end: after velysterm Plan C2 lands, paste its op-spec fragment into
`docs/PROTOCOL.md` (one section, additive).
