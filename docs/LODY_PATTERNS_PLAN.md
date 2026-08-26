# Lody-Patterns Implementation Plan

**Status: fully executed (2026-08-26).** Items 1–5 landed, tested, and
synced (commits in both repos); the follow-up wave (items 8–10 below:
real transport, overlay UI, T6 durable wiring) landed the same day;
items 6–7 are resolved by analysis as deliberate non-actions (see
below).

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

## Executed in this pass (2026-08-26, follow-up)

### 3. `unfer` — FFI exposure of live status

`unfer_ffi` now exports **84 `uk_*` symbols**. Two new symbols, both
registered in the single-source-of-truth `SYMBOL_REGISTRY` and regenerated
through `scripts/gen_symbol_artifacts` (header + `EXPECTED_SYMBOLS.txt` +
`EXPECTED_SYMBOLS_ZENODO.txt`):

- **`uk_durable_status(buf, cap)`** — buffer-out JSON live status: backend
  label, per-stream record counts for all six well-known streams (audit,
  owner_log, actions, config, session, certificates), and the backend's
  persist counter. Answered without replaying any history (Loro answers
  `stream_len` from the list length). The schema is stable even RAM-only
  (`backend: "none"`, every stream `0`). `DurableStore::persist_count` was
  added to the trait (default `0`; the Loro coalescer overrides it), so the
  coalescer's effect is observable through the ABI.
- **`uk_certificate_issued(cert_json, len)`** — records an emitted
  verification certificate (T6 mass-gap / Ritz) as a
  `certificate-issued` line in the durable `certificates` stream and
  checkpoints. Returns the 1-based sequence number; **fail-closed**: a
  RAM-only kernel refuses with the new **UK-1011 `DurableNotConfigured`**
  (a record that would not be replayable is never acknowledged as
  recorded). A `certificate-issued` line for a malformed record is refused
  with UK-1001.

New `Code::DURABLE_NOT_CONFIGURED = 1011` in `unfer_protocol::codes` (next
after UK-1010), documented in the code table. Tests: a kill-and-resume lib
test records a mass-gap certificate, checks live status
(`certificates: 1`, `persist_count >= 1`), re-opens the store from the same
directory and replays the exact `certificate-issued` line, then confirms the
RAM-only refusal; an integration test asserts the live-status JSON shape.

### 4. `unfer` — T6 certificate pipeline lands on the durable store

Folded into item 3 (the two are one surface): the durable `certificates`
stream is the replayable audit trail for every emitted certificate. The
`qcd_mass_gap_certified` NDJSON emitter stays the on-disk artifact the Lean
reader consumes (timepiece §13); `uk_certificate_issued` is the kernel-side
hook that records each emission durably, so "which certificates were
issued, when" is replayable — the audit trail Lody's `turn-diff-store`
provides for agent actions.

### 5. `velysterm` — host wiring of presence

`crates/mathed/src/main.rs` (the editor host) now wires the C13
`PresenceStore` in:

- `HostPresence` resource created in `setup` (peer `"host"`, 30 s timeout)
  plus a `DemoPeerPresence` in-process peer (`"demo-peer"`) — the
  single-host demo transport.
- `sync_presence` system (Update, after `handle_keyboard`): publishes the
  caret via `set_cursor` on every move (change-detected), drains an
  `InboundPresenceBlob` transport hook (where a real network pushes remote
  `encode()` payloads), exchanges blobs with the demo peer every frame so
  the inbound `apply` path runs end-to-end without a second process, and
  prunes lapsed peers via `remove_outdated`. One-shot `info!` log reports
  the live peer list.

No new networking and no new crates: the delta-shaped blob channel that
would carry `export_delta`/`import_delta` traffic now carries presence too.

---

## Executed in this pass (2026-08-26, follow-up 2)

### 8. `velysterm` — real TCP presence transport

`crates/mathed/src/presence_net.rs` (new module, std-only — no new deps)
carries presence between two mathed instances over TCP: `--listen
host:port` accepts the first peer in the background, `--connect
host:port` dials it. Frames are `[tag: u8][len: u32 BE][payload]` with
`b'P'` = presence blob (room for a future `b'D'` delta frame). A reader
thread unframes the wire into an inbox; a writer frames the outbox;
`Drop` sets a shutdown flag and closes the socket so the peer observes
EOF (`connected()` flips false). `sync_presence` now broadcasts the
host's presence on every caret move and drains the wire into
`PresenceStore::apply`; the offline in-process demo peer remains the
fallback when no transport is configured. Two unit tests: loopback
round-trip both directions + EOF on drop, and connect-refused fails
fast.

### 9. `velysterm` — live-collaborator overlay UI

F3 toggles an overlay (top-right, semi-transparent) listing the
transport status plus every live peer — name · caret line:col (computed
from the doc text) · seconds since heartbeat. The node rebuilds only
when the text changes; a remote caret byte is clamped to a char
boundary so it never panics slicing the doc.

### 10. `unfer` — T6 certificates durably recorded end-to-end

`fock_sirk::emit_gap_certificate_ndjson_with` (new, additive — the old
entry point delegates to it with a no-op sink) hands every emitted
NDJSON line to a recorder. `unfer_ffi` dev-depends on `fock_sirk`
(test-only) and the new `t6_emitted_certificates_are_durably_recorded
_end_to_end` lib test runs the real g=2, m=4 mass-gap solve, records
both sector certificates and the assembly through
`uk_certificate_issued` (seqs 1,2,3), checks `uk_durable_status`
(`certificates: 3`), and kill-and-resumes to replay exactly those three
`certificate-issued` lines from disk.

---

## Resolved by analysis (deliberately NOT executed)

These were the remaining candidates from the Lody survey. Each is a
*decision*, not a TODO: the analysis below is why no code lands for them,
and the outcome closes the plan.

### 6. `dynamic-arctic` / `australVM` — no awareness surface

- `dynamic-arctic` is pure computation (`arctic_core`/`lagrange`/`shine_core`
  cores, no collaboration surface — verified: no loro/awareness/presence
  references in `src/`). Lody's presence patterns do not apply. **Resolved:**
  leave as-is.
- `australVM` is an OCaml editor (`austral.opam`, `austral/`). Loro has no
  maintained OCaml binding (verified: no `loro` in any opam/dune/ml file),
  so presence would be a green-field port — explicitly **not** worth it
  under the "improve existing code" rule. **Resolved:** leave as-is.

### 7. `timepiece` — no new code

The wave/review discipline (`CONSOLIDATED_PLAN.md` §8 gate re-run, §13
certified-bound plan, BookProof `#check` sections) already implements Lody's
`review-automation-plan` / `turn-diff-store` ideas in the Lean workflow.
**Resolved:** nothing to build; the specialist gate re-run after each merge
*is* the pattern.

---

## Definition of done for this pass

- All five executed items land as extensions of existing modules (no new
  crates, no new binaries, no new deps).
- `cargo test -p mathed_core` green (154) and `cargo build -p mathed` green
  in `velysterm`; clippy clean.
- `cargo test -p unfer_ffi -p unfer_protocol` green in `unfer` (94 lib +
  44 ffi integration + 109 protocol); `scripts/gen_symbol_artifacts check`
  clean (84 uk_ + 5 uz_ in sync).
- Both repos committed and synced.
