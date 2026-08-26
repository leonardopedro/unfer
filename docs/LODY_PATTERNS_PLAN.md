# Lody-Patterns Implementation Plan

Source: analysis of `LodyAI/Lody` (the collaborative coding-agents platform built
on **Loro**, which this project already uses in `unfer` and `velysterm`).
Lody's architecture (from its module tree and sources) centers on a handful of
reusable collaboration patterns:

| Lody module | Pattern | Where it lands here |
| :--- | :--- | :--- |
| `presence` + `session-live-status` | Live peer presence over Loro awareness | `velysterm` editor sync (cursors/names); `unfer` durable store (per-stream live lengths) |
| `persist-coalescer` | Debounce/coalesce Loro persistence | `unfer` Loro durable store (dirty-tracked flush) |
| `turn-diff-store` | Per-turn operation diffs | `timepiece` wave discipline (already diff-shaped via BookProof `#check` sections; nothing to build) |
| `workspace-watch` | Watch files, react to changes | `unfer`/`timepiece` gate re-runs (already scripted in `CONSOLIDATED_PLAN.md` §8) |
| `flock-sync` + `connection-recovery` | Locking and reconnect | `unfer` durable store already has the `DrainLock`; reconnection is host-side transport work (out of scope) |
| `pr-poller` / `review-automation-plan` | Automated review loops | `timepiece`'s specialist gate re-run; `unfer`'s T6 NDJSON verification pipeline |

**Guiding rule (per the request): improve existing features and code, do not
bolt on new standalone systems.** Every item below extends a module or surface
that already exists.

---

## Executed in this pass (2026-08-26)

### 1. `velysterm` — live presence on the existing sync module (Lody `presence`)

`crates/mathed_core/src/sync.rs` already held the C13 delta exchange
(`export_delta` / `import_delta`) but had *no awareness*. Added a
`PresenceStore` beside it, backed by Loro's `EphemeralStore` (the
non-deprecated successor of `Awareness`, same presence semantics Lody uses):

- `Presence { peer, name, cursor, last_seen_ms }` — the live peer view.
- `PresenceStore::new(peer, name, timeout_ms)`, `set_name`, `set_cursor`
  (publishes a heartbeat + caret), `encode` / `encode_all` / `apply`
  (transport over the same channel as deltas), `remove_outdated` (prune
  lapsed peers), `peers()` (live list, self excluded, sorted).
- `subscribe_local_updates` passthrough so a host can auto-broadcast
  presence changes on the same socket that carries deltas.
- Tests: cross-peer cursor roundtrip, self-exclusion, expiry after timeout.

Presence stays **ephemeral** — never written into the document history,
never persisted — exactly Lody's design.

### 2. `unfer` — durable-store live status + coalesced persistence (Lody `session-live-status` + `persist-coalescer`)

The H4 durable store (`unfer_ffi/src/durable/`) already had the
`DrainLock`-serialized checkpoint barrier. Two improvements to it:

- **`DurableStore::stream_len(stream)`** — new trait method (default impl
  via `replay`; the Loro backend answers from the list length with zero
  allocation). Gives a live per-stream record count — the kernel-side
  equivalent of Lody's session live-status — without replaying history.
- **Dirty-tracked `flush` on the Loro backend** — the coalescer. `append`
  marks the store dirty; a `flush` with nothing new since the last
  checkpoint skips the snapshot export + tmp-write + rename entirely
  (still under the drain lock). A `persist_count()` counter makes the
  coalescing observable and testable.
- Tests: `stream_len` across the round-trip suite, dirty-flush skips
  redundant persistence, clean flush still checkpoints after appends.

---

## Recommended next steps (not yet executed)

### 3. `unfer` — FFI exposure of live status

`unfer_ffi` currently exports 81 `uk_*` symbols. Expose the durable live
status (stream lengths, backend label, persist count) as one new `uk_*`
symbol, regenerate the C header via `gen_unfer_kernel_h.py`, and extend
`EXPECTED_SYMBOLS.txt`. This is what lets a host (or the Lody-style UI)
render "which streams are live and how big" without replaying.

### 4. `unfer` — T6 certificate pipeline already lands on the durable store

The `qcd_mass_gap_certified` NDJSON emitter writes certificates; point it at
the durable store's `certificates` stream (or keep NDJSON on disk and record
a durable `certificate-issued` audit line) so every emitted certificate is
replayable — the audit trail Lody's `turn-diff-store` provides for agent
actions.

### 5. `velysterm` — host wiring of presence

`crates/mathed/src/main.rs` (the editor host) should call `set_cursor` on
caret moves and `apply` on inbound presence blobs. Requires the delta
transport to be bidirectional (already the shape of `export_delta` /
`import_delta`); no new networking needed for a single-host demo.

### 6. `dynamic-arctic` / `australVM` — awareness surfaces

- `dynamic-arctic` is pure computation (arctic/lagrange/shine cores) — no
  collaboration surface; Lody patterns do not apply. Leave as-is.
- `australVM` has an OCaml editor (`editor/`). Loro has no maintained OCaml
  binding, so presence would be a green-field port — explicitly **not**
  worth it under the "improve existing code" rule. Leave as-is.

### 7. `timepiece` — no new code

The wave/review discipline (`CONSOLIDATED_PLAN.md` §8 gate re-run, §13
certified-bound plan, BookProof `#check` sections) already implements Lody's
`review-automation-plan` / `turn-diff-store` ideas in the Lean workflow.
Nothing to build; the specialist gate re-run after each merge *is* the
pattern.

---

## Definition of done for this pass

- Both executed items land as extensions of existing modules (no new crates,
  no new binaries, no new deps).
- `cargo test -p mathed_core` green in `velysterm`.
- `cargo test -p unfer_ffi -p unfer_protocol` green in `unfer`.
- Both repos committed.
