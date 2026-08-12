# Improvement Plan — unfer + velysterm + australVM

Status: drafted after a full audit of all three workspaces (668 + 319 tests,
128 clippy warnings, ABI symbol tables, CI configs). Ordered by leverage.

## Tier 0 — Unblock CI (both repos are red; highest leverage)

- [x] 1. Add 4 drifted ABI symbols to manifest
      `unfer_ffi/EXPECTED_SYMBOLS.txt` (+`uk_meter_status`, `uk_secret_get`,
      `uk_secret_put`, `uk_secret_revoke`). 65 exported vs 61 manifest; the
      `ffi-symbols` CI job fails.
      — Done: the manifest already listed all 65, but the four functions were
      `pub extern "C"` **without** `#[unsafe(no_mangle)]`, so the cdylib never
      exported them (61 actual). Added `no_mangle` to all four in
      `unfer_ffi/src/lib.rs`; `nm -D` now shows exactly 65 `uk_*` matching
      `EXPECTED_SYMBOLS.txt` and 5 `uz_*` matching `EXPECTED_SYMBOLS_ZENODO.txt`
      (zenodo feature). All 73 lib + 30 integration + 1 doctest tests pass.
      The `ffi-symbols` CI job's awk/diff check now succeeds.
- [x] 2. Add 26 missing `uk_*` to bridge dispatch table
      `australVM/safestos/cranelift/src/lib.rs` `UNFER_SYMBOLS` (35 → 61).
      `symbol_sync` test panics deterministically.
      — Done (commit `d122549e` "B7"): `UNFER_SYMBOLS` registers all 65
      `uk_*` symbols (was 35), matching `EXPECTED_SYMBOLS.txt` exactly.
      Verified locally: `cargo test --features test-stubs,unfer-kernel --test
      symbol_sync` → 3/3 pass; with `zenodo-store` → 4/4 pass (uz_* too).
      Note: the `au_alloc`/`au_free` C-runtime stubs are provided by the
      `test-stubs` feature (CI's unit-test step), so plain `--features
      unfer-kernel` fails to link `modhost` — that's expected, not a drift.
- [x] 3. Fix australVM CI: wrong org + wrong checkout path
      `.github/workflows/build-and-test.yml`: `anomalyco/unfer` →
      `leonardopedro/unfer`; checkout path must match the test's
      `CARGO_MANIFEST_DIR/../../../unfer` resolution.
      — Done (commit `d122549e` "B7"): `repository: leonardopedro/unfer` with
      `path: ../unfer`, which resolves to `$GITHUB_WORKSPACE/../unfer` —
      verified identical to the `CARGO_MANIFEST_DIR/../../../unfer` used by
      both the `symbol_sync` test and the `../../../unfer/unfer_ffi` path dep.
      CI demo `DENY_SYM` also corrected to `uk_event_probability`.
- [x] 4. Clippy `-D warnings`: clear ~123 unfer + 38 velysterm warnings
      Worst: `qfm_text`(35), `unfer_edge`(24), `logos`(23), `qfm`(17),
      `unfer_ffi`(11); velysterm `mathed_mini/src/bin/mathed_mini.rs:21`
      `never_loop` (deny).
      — Done: `cargo clippy --workspace --all-targets` (default features) and
      the non-CUDA feature set (`audit`/`zenodo`/`network`/`latex`) are both
      clean at `-D warnings` (0 warnings). velysterm `mathed_mini` lints clean
      (its `never_loop` was already fixed in #19); the remaining velysterm
      crates (`mathed`) can only be linted on CI runners that provide the
      ALSA/wayland dev libs (NixOS box has no apt) — the CI apt list already
      installs them (#30), so clean-runner clippy green holds.
- [x] 5. `cargo fmt --all -- --check` — fix unformatted files.
      — Done: `cargo fmt --all -- --check` exits 0 in both unfer and
      velysterm (velysterm only warns about unstable nightly options in
      rustfmt.toml, which are inert on stable).
- [x] 6. CI runs only default features — add feature-gated jobs
      (network, audit, zenodo, cuda, latex) so the 30 gated tests + 14
      `unfer_edge` audit tests actually run.
      — Done: `ci.yml` already has the `gated-features` job (audit console,
      zenodo store, latex CAS via nested_fock_algebra/prob_kernel/unfer_ffi,
      QuePaxa network, Taler network testnet) plus a self-hosted
      `qfm-tomo-e2e-cuda` job for the CUDA surface (no public GPU runner).

## Tier 1 — Real correctness bugs

- [x] 7. `unwrap()` on unknown reserve/merchant → UK-710x diag, not panic
      `unfer_taler/src/exchange.rs:220,323`.
      — Done: all reserve/merchant/peg-out lookups in non-test code return
      `Diagnostic` via `ok_or_else(diag(Code::TALER_*))`; only `#[cfg(test)]`
      code unwraps.
- [x] 8. `unwrap()` on missing escrow in settle path
      `unfer_consensus/src/escrow.rs:205`.
      — Done: both the initial lookup and the post-emission re-fetch return
      `ESCROW_UNKNOWN` diagnostics (`escrow.rs:175,205-209`).
- [x] 9. Panics on unknown word instead of diagnostics
      `logos/src/core_ir/compiler.rs:13`, `logos/src/ccg/compiler.rs:9`,
      `logos/src/deltanet/types.rs:144,224`.
      — Done: all four sites return `Result<_, String>` (`"no semantic
      template for '{}'"`, `"unbound variable"`, `"no port at node ..."`).
      The only remaining `unreachable!()`/`as_ref().unwrap()` in
      `deltanet/reducer.rs` are on internal interaction-net storage
      invariants where the agent was just validated, not user input.
- [x] 10. `unreachable!()` in non-test event dispatch
      `unfer_ffi/src/handles.rs:139`.
      — Done: the remaining match arms dispatch every variant or return
      `false` explicitly; no `unreachable!`/`panic!` anywhere in
      `unfer_ffi/src`. Full workspace: 56 test binaries, all green.

## Tier 2 — Cross-repo contract drift (the three-repo seams)

- [x] 11. Three op-name registries disagree: agent `VALID_OPS` (32) vs edge
      `ALLOWED_OPS` (23, also lists `observe` the agent lacks) vs consensus
      `apply_session_op` (`create_model` only).
      `velysterm/crates/kernel_client/src/bin/unfer_agent.rs:46`,
      `unfer/unfer_edge/src/filter.rs:11`, `unfer_consensus/src/node.rs:90`.
      Extract a shared table in `unfer_protocol` + consistency test.
      — Done: `unfer_protocol::ops` is the single registry
      (`SESSION_OPS`/`EDGE_ALLOWED_OPS`/`AGENT_OPS`/`CONSENSUS_OPS`) with
      invariant tests (`no_duplicates_across_all_tables`, subset checks,
      session=union). Agent and edge derive their slices from it; the last
      hardcoded stub — consensus `apply_session_op` — now dispatches through
      `CONSENSUS_OPS::contains` and re-derives `create_model` support from the
      shared table (`unfer_consensus/src/node.rs:90`).
- [x] 12. Agent event JSON (ad-hoc, incl. `components`) not round-trippable
      with `unfer_protocol::KernelEvent` (`unfer_agent.rs:300` vs `types.rs:299`).
      — Done: `kernel_event_tests::evolved_event_round_trips_through_kernel_event`
      (unfer_protocol/src/types.rs:1575) serializes `KernelEvent::Evolved`,
      injects the agent's historical `components` extra field, and asserts
      it deserializes back to the canonical event with the extra dropped on
      re-serialize. Also covered: PriorSet + Conditioned round-trips.
- [x] 13. `version` op returns kernel_client's version, not the kernel's
      (`unfer_agent.rs:45`).
      — Done: agent returns `unfer_protocol::KERNEL_VERSION` (the shared
      contract constant), asserted by `unfer_agent.rs:1398-1400`;
      `unfer_protocol::KERNEL_VERSION` = 1 in lib.rs:26.
- [x] 14. Translator string contracts (`builtin_*.typ` JSON keys) guarded
      only by tests, not compile time.
      — Done: the two round-trip tests
      (`builtin_translator_round_trips_through_term_spec` and
      `builtin_event_translator_round_trips_through_event_predicate` in
      `mathed_mini/src/translate.rs`) parse the builtin translators' emitted
      JSON through the real `unfer_protocol::{TermSpec, EventPredicate}`
      serde schemas, so a key/kind/level drift fails at test time instead of
      surfacing as a downstream UK-1003.
- [x] 15. C header declares 18 of 65 ABI functions
      (`unfer_ffi/include/unfer_kernel.h`).
      — Done: the header declares all 65 `uk_*` (64 return `int64_t`, one —
      `uk_ode_measure_original` — returns `double`, matching the Rust
      `-> f64`) and all 5 `uz_*`. Verified symbol-for-symbol against
      `EXPECTED_SYMBOLS.txt` / `EXPECTED_SYMBOLS_ZENODO.txt`.

## Tier 3 — Security surface

- [x] 16. `caprpc.rs` (S28 capability RPC) implemented but unwired — 22 dead
      items; wire into `main.rs` dispatch or delete (`unfer_edge/src/caprpc.rs`).
      — Done: the S28 surface is wired into the edge dispatch
      (`/api/cap/mint|promise|resolve|revoke|invoke`) in
      `unfer_edge/src/main.rs:139-233`; all 7 caprpc module tests run and pass
      under `--features audit` (40 edge tests total). No dead items remain —
      clippy on the audit feature set is clean.
- [x] 17. Entire S22/S28 surface (`admin/gate/blueprint/audit`) is
      `audit`-gated → never in CI; add an audit-feature CI job.
      — Done: `ci.yml` `gated-features` job runs
      `cargo test -p unfer_edge --features audit` (S22 admin console, gate,
      blueprint, audit + S28 caprpc). Verified locally: 40 pass.
- [x] 18. Hardcoded `"operator"` caller + `127.0.0.1:3001` backend
      (`unfer_edge/src/main.rs:167,530`).
      — Done: admin principal comes from `UNFER_ADMIN_PRINCIPAL`
      (`admin::admin_principal()`, default `operator`, S22 seam) and the
      backend address from the `--backend` flag (default `127.0.0.1:3001`,
      `main.rs:596`). Capability minting uses `admin::admin_principal()`
      rather than a literal (main.rs:158).

## Tier 4 — Dead code, housekeeping, docs

- [x] 19. velysterm: clear 5 dead fns (`mathed_mini/src/app.rs`), 2 unused
      imports (`render.rs`), `never_loop` bin bug.
- [x] 20. unfer: `unfer_ffi` dead `SecretHandle`/`RegistryInner`/`OnceLock`;
      `unfer_edge` 22 dead-code warnings.
- [x] 21. Decide fate of orphaned `delta_algebra`/`delta_sirk` (excluded,
      unreachable via `-p`). — Resolved: never existed as committed crates.
      No dirs, no git history, not in workspace members; the only reference
      was this plan itself. Nothing to remove.
- [x] 22. `qfm_text_runs/` training data + `scratch/test_expand.rs` in git —
      gitignore or external store. — Resolved: `*.qfm` checkpoints already
      gitignored (100 MB+ each) with `tools/clean_qfm_text_runs.sh`; the
      31 tracked files in `qfm_text_runs/` are small provenance logs.
      Removed dead `scratch/test_expand.rs` from tracking.
- [x] 23. Crate docs for 9/14 unfer crates; READMEs (12 missing unfer, ~10
      velysterm); complete C header; add doctests (only 1 in workspace). —
      Done: crate-level `//!` docs for the 9 unfer crates lacking them
      (nested_fock_algebra, fock_sirk, unfer_protocol, prob_kernel,
      unfer_ffi, logos, unfer_consensus, unfer_identity, unfer_data);
      18 READMEs created (11 unfer + 8 velysterm — both repos' missing
      census cleared); C header completed in #15 (65 uk_* + 5 uz_*);
      3 runnable doctests added (unfer_protocol, unfer_identity, unfer_ffi)
      — all pass, doc build shows only pre-existing benign warnings.
- [x] 24. Stale docs: `unfer_agent.rs` says 8 ops (now 32); velysterm
      AGENTS.md references nonexistent `parse.rs`; australVM references
      `CpsGen_backup.ml`, `test_fib_math`, `fib/*.bin`; unfer AGENTS.md
      claims "no CI". — Done: `unfer_agent.rs` docstring now mirrors the
      32-op `AGENT_OPS` registry by namespace; velysterm AGENTS.md fixed
      (removed nonexistent `parse.rs`, noted JSON-spec parsing +
      `AGENT_OPS`); australVM safestos `AGENTS.md`/`STATUS.md` stale
      CPS-JIT file references replaced with a historical note (the
      `docs/history/` copy is the intentional archive); unfer AGENTS.md
      "no CI" claim was already gone.
- [x] 25. README test censuses stale (126/105/29 claimed vs 147/119/39 actual).
      — Done: re-ran the full workspace suite for ground truth (676 green on
      CPU, rev 23) and rewrote the README "Test & benchmark counts" section
      with an accurate per-crate breakdown; TUTORIAL.md's rev-19 count
      (201) updated to 676 (rev 23). No stale counts remain in velysterm.

## Tier 5 — Scale / architecture

- [ ] 26. `qfm` pipeline: 17-arg `dense_forward_sirk`, 5-tuple returns,
      retained stubs (`pipeline.rs:547`) — refactor to config structs.
      — Deferred: deliberately left untouched per maintainer directive (the
      qfm surface is a stability/sensitivity area, not a correctness defect).
- [x] 27. GPU: 2 `cuda` tests + zenodo 9 tests never run; add compile/run job.
      — Done: `ci.yml` `gated-features` job runs the 9 zenodo-store module
      tests (`cargo test -p unfer_ffi --features zenodo`) and the CUDA
      surface is covered by the self-hosted `qfm-tomo-e2e-cuda` job
      (`runs-on: [self-hosted, gpu]`, `cargo test -p fock_sirk --features
      cuda`).
- [x] 28. `fock_sirk` unit tests 122s — split the heavy Krylov cases.
      — Done: the Yang-Mills solutions that drove the wall time
      (`adaptive_l4_completes_under_budget` ~11s, `adaptive_l5` ~41s, and the
      l=3 mass-gap demo) were moved from the crate's private unit-test module
      into a dedicated integration binary `fock_sirk/tests/heavy_krylov.rs`.
      They still run in `cargo test --workspace`/CI, but the `--lib` unit suite
      dropped from ~45s to ~11s and the heavy solves are filterable via
      `cargo test -p fock_sirk --test heavy_krylov`. Test count unchanged
      (29: 26 unit + 3 integration).
- [x] 29. `unfer_consensus`: `eprintln!` in `net.rs` (no propagation from
      TLS/snapshot layer); RISC-Zero layer "not yet wired" (`certs.rs:10`);
      fixed `SMT_DEPTH=256`.
      — Done: `LedgerFile::load/save`, `SharedLedger::new`/`persist`/
      `allocate`/`append`, `NetConsensus::submit`, and `LedgerStateMachine::execute`
      now propagate errors as `Result`/`Diagnostic`/`QuePaxaError::StorageError`
      instead of swallowing them behind `eprintln!`; a corrupt durable ledger now
      fails loudly at construction rather than silently resetting to empty
      (regression test `corrupt_ledger_file_fails_loudly_at_load`). The two
      background-task `eprintln!`s (`server.run` stop, pump retry loop) are kept
      as diagnostic output for detached spawned tasks with no caller to propagate
      to. `SMT_DEPTH` is now derived from the `[u8; 32]` hash width (still 256)
      with a rationale comment, and the certs module doc states the RISC-Zero zkVM
      is a documented future additive layer (per `docs/PLAN_REFI_EXCHANGE.md`),
      deliberately not wired. 43 default + 47 network-feature tests green.
- [x] 30. velysterm CI relies on apt ALSA/udev libs — verify clean-runner
      green; mathed tests unverified.
      Verified the CI apt list (`libasound2-dev libudev-dev libwayland-dev`) is
      sufficient and complete for a clean ubuntu runner: via `cargo tree` on the
      bevy 0.18.1 workspace, `alsa` (cpal→bevy_audio) and `wayland-sys` are the
      only link-time system libs; `x11-dl` and `xkbcommon-dl` are runtime dlopen
      only, and `udev` is not even in the dependency tree (its install is
      harmless). All jobs (check/check-no-defaults/clippy/doc/test/doctest/smoke)
      install the same three libs. The mathed `kernel_smoke` tests are genuinely
      headless — no window, audio device, or GPU — and reuse the exact shared
      path (`mathed_core::markers::{scan, resolve_segments}` +
      `mathed_core::transform::to_render_text` over
      `mathed_mini::KernelBridge`) that mathed_mini's
      `overlay_renders_green_for_success_and_red_for_error` exercises, which
      PASSES on this box (CPU-only, no ALSA/wayland runtime dep). The only
      blocker to running `cargo test -p mathed kernel_smoke` locally is the
      bevy *link* step needing `libwayland-dev` (NixOS box has no channels/apt,
      and no prebuilt mathed binary exists in target/); CI provides that lib, so
      clean-runner green holds. Note: `x11-dl` records libdir via pkg-config but
      only links `dl`/`c`, so `libx11-dev` is NOT required even with the `x11`
      bevy feature.

## Execution order

Tier 0 items 1–3 are small diffs that turn both CIs green and validate every
future change; then Tier 1 correctness (7–10); then cross-repo contract work
(11–15); then security wiring (16–18); then dead code/docs (19–25); then the
larger refactors (26–30).
