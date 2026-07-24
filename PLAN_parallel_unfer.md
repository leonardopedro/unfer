# PLAN A — unfer (probability kernel + QFM research)

Parallel workstream 1 of 3. Companion plans: `australVM/PLAN_parallel_australvm.md`,
`velysterm/PLAN_parallel_velysterm.md`.

## System context

Three repos form one system:
- **unfer** (this repo) — the kernel: `prob_kernel::Session` (Born-rule API), `unfer_ffi`
  (21 `uk_*` + 5 `uz_*` C symbols), `unfer_protocol` (serde + UK-#### codes), `qfm`/`qfm_text`,
  6 Austral modules, plus new crates: `logos` (CNL→verified execution compiler),
  `ode_sirk` (ODE→Hamiltonian singularity detection), `unfer_consensus` (QuePaxa federation),
  `unfer_data` (encrypted chunked data plane), `unfer_identity` (DID/keypair).
- **australVM** — Austral JIT (`safestos/cranelift`) that statically links `unfer_ffi` via a
  **path dependency** and registers the `uk_*` symbols; hosts the Austral modules (modhost).
  B1–B7 complete; B8–B10 (genuine hosting, Tidepool, cap-std) pending.
- **velysterm** — editor/agent frontend; `kernel_client` path-depends on `prob_kernel` +
  `unfer_protocol`; ships the `unfer_agent` NDJSON binary (20+ ops incl. federation).
  Plan C (C1–C10) complete.

Because both dependents read unfer's **working tree** (path deps), any uncommitted breakage
here propagates to both. Keep this repo green at all times.

## Parallel-execution rules (all three plans share these)

1. **Ownership**: modify only files inside this repo. Cross-repo *reads* are fine.
   Cross-repo *writes* are forbidden, except steps explicitly marked `[SYNC]`.
2. **Frozen contract** (additive-only changes allowed; no renames/removals/signature changes):
   - the 21 `uk_*` and 5 `uz_*` symbols and their C signatures;
   - `prob_kernel::Session` public API;
   - `unfer_protocol` serde types and UK-#### code assignments;
   - the NDJSON agent ops (20+); `module.toml` grant vocabulary.
3. **Commit discipline**: meaningful messages; commit after every completed stage.
4. Stages are ordered small → large; each ends in a verifiable acceptance command.

## Current state (2026-07-24)

- `cargo test --workspace` green (~348 tests across 20+ crates).
- A1–A5 complete and committed. A2 `[SYNC]` done (PROTOCOL.md has all 20+ ops + 6xxx codes).
- A6 is **hypothetical** (research backlog, not to be executed).
- A7 partially done: `unfer_edge` edition still 2021; `qfm_text_runs/` is 3.9 GB undocumented.
- New crates since the original plan: `logos/` (CNL compiler, 10+ submodules),
  `ode_sirk/` (ODE→Hamiltonian), `unfer_consensus/` (QuePaxa), `unfer_data/` (encrypted
  chunks + magnet URIs), `unfer_identity/` (DID). These have tests but no dedicated docs.
- velysterm Plan C complete; australVM B1–B7 complete.

---

## Completed stages (Phase 1)

| Stage | Summary |
|-------|---------|
| A1 | Committed Pauli–Grover in-flight work |
| A2 | Doc-drift sweep (AGENTS.md, MODULE_RECIPE, PROTOCOL.md, ARCHITECTURE) — DONE |
| A3 | FFI symbol CI gate (`EXPECTED_SYMBOLS.txt`, 21 uk_* + 5 uz_*) — DONE |
| A4 | `tools/module_builder` — unified build+test for all 6 modules — DONE |
| A5 | Property & fuzz tests (nested_fock_algebra proptest, unfer_protocol fuzz) — DONE |

---

## Stage A6 — Pauli–Grover research completion (HYPOTHETICAL — DO NOT EXECUTE)

> **This stage is hypothetical and must not be executed.** It records open
> research questions for future reference only. No agent or contributor
> should implement any item below without an explicit new directive.

Context: PG without kernel gives 100% training but chance generalization; diffusion +
distributed multi-mode encoding generalizes. Open questions, in priority order:

1. **Kernel coupling for PG**: extend `dense_pauli_grover_matvec` with the off-diagonal
   kernel terms and re-run the parity held-out sweep. Hypothesis: kernel lifts PG
   generalization above chance.
2. **`a` sweep**: 0.5–1.0 grid on parity + MNIST; document whether the residual |0⟩
   component helps held-out accuracy (update QFM.tex §Pauli–Grover with results).
3. **CIFAR PG test**: port the existing CIFAR-10 fixture path to PG.
4. **qfm_text GPU decode** (L): port `decode_sketched` to the candle CUDA path already in
   `fock_sirk` — the ~140 h CPU wall blocks the rev-37 v3 diffusion-Hamiltonian evaluation.
5. Create `QFM_TEXT_STATUS.md` documenting the current state of qfm_text evaluations.

**Acceptance**: N/A (hypothetical).

## Stage A7 — Consistency cleanup (S)

1. `unfer_edge`: align `edition = "2024"` with the workspace.
2. Document (or script) cleanup of `qfm_text_runs/` checkpoints (3.9 GB).
3. `docs/BUILD_PIPELINE.md`: cross-link the module_builder (TBD already removed).

**Acceptance**: `cargo check --workspace` warning-free except upstream deps; CI green.

## Stage A8 — New-crate documentation + test hardening (M)

The five new crates (`logos`, `ode_sirk`, `unfer_consensus`, `unfer_data`, `unfer_identity`)
have tests but no dedicated documentation or integration coverage.

1. Write `docs/LOGOS.md`: architecture (parse→compile→reduce→readback→hash), the CNL
   subset supported, deltanet/harper_gate roles, and the verified-execution guarantee.
2. Write `docs/ODE_SIRK.md`: the ODE→Hamiltonian singularity detection pipeline, ESA
   (exponential stability analysis), change-of-variables, and flow integration.
3. Write `docs/FEDERATION.md`: QuePaxa consensus engine, DID identity lifecycle
   (create/resolve/update/revoke), content publishing (CID + signatures), relay transport.
   Document the 6xxx UK codes and the agent ops (`did_*`, `content_*`, `consensus_*`).
4. Write `docs/DATA_PLANE.md`: chunking strategy, X25519+AES-GCM encryption, magnet URI
   format, publisher flow.
5. Add integration tests: consensus → identity → content_publish → content_resolve
   round-trip; data plane encrypt → chunk → reassemble → decrypt round-trip.
6. Add `logos`, `ode_sirk`, `unfer_consensus`, `unfer_data`, `unfer_identity` to
   `docs/ARCHITECTURE.md` crate diagram.

**Acceptance**: each new crate has a dedicated doc; integration tests pass;
`docs/ARCHITECTURE.md` lists all crates.

## Stage A9 — Cross-repo integration test (M)

The three repos share a contract but have no automated cross-repo verification.

1. Add `tests/integration/` with a script that:
   a. Builds `unfer_ffi` cdylib.
   b. Runs `tools/module_builder run demo_module` (exercises the FFI + module path).
   c. Pipes an NDJSON session through velysterm's `unfer_agent` (create_model → evolve →
      probability → bayesian_update → close_model) and asserts correct results.
   d. Verifies `EXPECTED_SYMBOLS.txt` matches the built cdylib.
2. Document the integration test in `docs/ARCHITECTURE.md` as the "system smoke test".
3. Add a CI job (or document the manual command) that runs the full integration suite.

**Acceptance**: `tests/integration/run.sh` passes on a clean checkout of all three repos
as siblings; fails if any contract violation is introduced.

## Stage A10 — Logos research follow-through (M–L, research)

`logos` compiles a CNL (controlled natural language) subset to verified execution graphs.
Current state: working end-to-end pipeline (parse→compile→reduce→readback→hash).

1. Document the CNL grammar subset and the compilation semantics in `docs/LOGOS.md`.
2. Add property tests: compilation preserves semantics (readback ∘ compile ≈ identity on
   the CNL subset); reduction is confluent on well-formed graphs.
3. Explore connecting logos output to `prob_kernel` model specs — a CNL description of a
   quantum system should produce a valid `ModelSpec` that `create_model` accepts.
4. If (3) succeeds, add a `logos_compile` agent op (additive, next free UK codes).

**Acceptance**: property tests pass; if (3) lands, a CNL sentence produces a working
kernel session end-to-end.

---

## Out of scope for this plan (owned by the other workstreams)

- australVM: B8 genuine hosting, B9 Tidepool modules, B9b Egison, B10 cap-std modules.
- velysterm: editor UX for federation ops, multi-model documents, collaborative editing.

`[SYNC]` steps: none remaining (A2 sync complete).
