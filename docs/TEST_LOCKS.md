# Test locks in `unfer_ffi` — the X-F7 inventory

**Why this file exists.** About a quarter of full-suite runs used to hang, and
several tests still fail intermittently under parallel execution. The cause is
shared in-memory state (the action store, the owner log, the audit ring, the
durable store) guarded by several different mutexes, with no single order every
test agrees on. Two fixes were tried and reverted — one merged the mutexes
(instant self-deadlock, because a test takes two differently named locks in
nested scopes), the other reordered them (2 failures in 20 runs, worse than the
0-in-20 it replaced). Both are recorded in the workspace plan's §X-F7.

The conclusion drawn from that was: **the prerequisite is an inventory of which
test touches which store and which mutex it actually holds**, and the inventory
had not been done. This file is that inventory, filled in as it is established —
one store at a time, with evidence rather than by inspection where the two can
disagree.

## The locks

| lock | declared | guards (by declaration) |
|---|---|---|
| `ACTION_TESTS_LOCK` | `unfer_ffi/src/lib.rs` | the action store, and the owner log for the action/report tests |
| `AUDIT_AGENT_TESTS_LOCK` | `unfer_ffi/src/lib.rs` | the audit ring and the agent/subscription owner-log lines |
| `DURABLE_TESTS_LOCK` | `unfer_ffi/src/lib.rs` | the durable store (H4) |
| `BLUEPRINT_TESTS_LOCK`, `METER_FFI_TESTS_LOCK`, `SPAWN_PACING_LOCK` | `unfer_ffi/src/lib.rs` | their own single-store tests (`SPAWN_PACING_LOCK` is `pub(crate)` — see the pacing finding below) |
| `AUCTION_TESTS_LOCK`, `CERT_TESTS_LOCK`, `METER_TESTS_LOCK`, `VAULT_TESTS_LOCK` | `unfer_ffi/src/handles.rs` | their own single-store tests |

> **A12 note on `reset_durable_for_tests`.** The helper does not reset one
> store: it clears **all four** (durable slot + action store + audit ring +
> owner log, plus the unknown-outcome set and both sequence counters). The
> original inventory's key discovery was that eleven callers held only
> DURABLE — or nothing — while wiping stores two and three out from under
> tests that held ACTION or AUDIT. Every reset caller now holds all four locks
> in the P → A → U → D order.

The majority order among multi-lock tests is **P → ACTION → AUDIT → DURABLE**
(P only where the spawn pacer is touched; each acquisition site carries a
comment saying why). A former exception — `audit_trail_never_persists_fixture_secret`
taking **AUDIT → ACTION** because the missing lock was appended after the
existing one — was left as it was for a long time on the evidence of a
reordering attempt that regressed. A12 finally caught it doing what an inverted
order does (see "The deadlock, caught live" below), scanned the whole suite for
inversions, found it was the only one, and converted it. The suite is now
ordered by a global rank: every acquisition order in `lib.rs` + `handles.rs` is
a subsequence of **P < ACTION < AUDIT < DURABLE** (and `BLUEPRINT < DURABLE`),
so no cycle of any length is possible — verified mechanically, not by eye
(`zz_inversions.py`'s scan output was empty after the conversion; the script is
scratch and deleted, its method is recorded here).

## Inventory: who touches the **owner log**

This is the first store completed, because it produced a *third* flaky test on
2026-10-06 (below) and the question "who can empty the ring?" had an answer
that inspection could settle.

| test | lock held | owner-log operation |
|---|---|---|
| `queue_drops_oldest_when_full` | `ACTION_TESTS_LOCK` → `AUDIT_AGENT_TESTS_LOCK` | `clear_owner_log()`, then **asserts its own drop notice is in the ring** |
| `owner_logger_writes_dot_component_lines` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` ×2 |
| `report_issue_is_noop_without_error_report_binding` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` |
| `report_issue_bound_appends_sanitized_line` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` |
| `audit_trail_never_persists_fixture_secret` | `ACTION_TESTS_LOCK` → `AUDIT_AGENT_TESTS_LOCK` | `uk_owner_clear()` |

**The gap, named:** the first row and the next three hold *different* mutexes.
Nothing prevents an ACTION-only test from clearing the ring between
`queue_drops_oldest_when_full`'s clear and its assertion — so the assertion can
read an empty ring and fail with `got: []`, which is exactly what happened.
(Closed by A12: `queue_drops` now holds ACTION → AUDIT, so it excludes the
ACTION-only clearers entirely — see the resolution at the end.)

## The flake this produced (2026-10-06)

`tests::queue_drops_oldest_when_full` failed in a full-workspace release run
with:

```
thread 'tests::queue_drops_oldest_when_full' panicked at unfer_ffi/src/lib.rs:4764:9:
owner ring must record the subscription drop, got: []
```

The ring was *empty*, not wrong — the signature of a concurrent clear, not of a
missing drop notice. Evidence for how it behaves:

* passes in isolation (6/6 single-test runs, release),
* passes when the whole `unfer_ffi` binary runs alone (4/4, release),
* failed 1 of 3 full-workspace release runs, where several crates' test
  binaries run at once and the window is widest.

This is the same class as the two tests already named in `README.md`
(`action_apply_resolves_and_merges_applied_result`,
`action_submit_queues_pending_record_with_provisional_result`) — and it is the
first *new* member found by doing the inventory rather than by reordering
locks.

## The fix — done, measured (A12)

The plan this file was written for is complete. What was done, in order, with
the measurement after each step (all runs: the `unfer_ffi` lib test binary,
release, `--test-threads=16`, the protocol is `timeout 60` per run so a
deadlock records as a failure instead of hanging the campaign):

1. **Inventory completed** for all four shared stores. The two headline
   findings:
   * `reset_durable_for_tests()` clears **all four** stores; eleven callers
     held only DURABLE or nothing (the silent-wipe class).
   * The spawn-pacing knob (`set_spawn_min_interval_ms`, process-global) was
     guarded by **two different mutexes** — `SPAWN_PACING_LOCK` in `lib.rs`
     and a private `PACE_TESTS_LOCK` in `handles.rs::spawn_pace_tests` — so
     the two modules did not exclude each other at all. (The caller context,
     often suspected, is `thread_local` and was *not* a mechanism.) A test's
     40 ms/60 s pacing window leaked into concurrent spawn tests as UK-4601
     refusals and expiring lockouts.
2. **One lock per knob**: `handles.rs` now aliases the crate-root
   `SPAWN_PACING_LOCK`; every test that spawns or touches the knob holds it
   first (P → A → U → D). All four store locks became `pub(crate)` so the
   sibling module can hold them.
3. **The -1010 class**: any test that *reads durability* under a possibly
   swapped kernel-global slot must hold `DURABLE_TESTS_LOCK` — the rule
   `session_compact_ffi_roundtrip` had already written down. Applied to
   `set_prior_enqueues_event`, `condition_enqueues_event`,
   `subscribe_filters_by_event_type`, `blueprint_export_instantiate_roundtrip`
   and `blueprint_export_gadget_probe_is_retryable_and_delivers_complete_json`
   (B → D, the order the other blueprint test already takes). The last one
   mattered twice over: its extra checkpoint *consumed the fail-after-3 flush
   counter* that `resolved_checkpoint_failure_surfaces_unknown_outcome`
   sequences its crash window on, so the concurrent run made that test fail at
   the wrong flush and never mark unknown-outcome.
4. **The deadlock, caught live.** After step 3, run 1 of the next campaign
   deadlocked — all 16 test threads blocked. The cycle: 
   `audit_trail_never_persists_fixture_secret` holds AUDIT and waits for
   ACTION (the inverted order this file used to justify keeping), while a
   gatekeeper test holds ACTION and waits for AUDIT; fourteen other tests
   piled up behind them. A mechanical scan of every acquisition order in
   `lib.rs` + `handles.rs` (comments and string literals stripped) named
   `audit_trail` as the **only** inversion left. It was converted to
   ACTION → AUDIT — the reordering that once "regressed" was safe once every
   other test agreed on the same order, which is exactly what the plan's
   prescription ("pick one order, convert, measure") meant.
5. **Measurement ladder** (failures are *runs*, each can fail >1 test):

   | campaign | result | what the failures were |
   |---|---|---|
   | baseline | 50 runs, 2 failed | pacing-knob UK-4601 leaks; owner-log clear race |
   | after step 2 (17 tests converted) | 50 runs, 1 failed | `queue_drops` owner ring → converted to A → U |
   | then | 100 runs, 4 failed | split pacing lock (2 runs), -1010 `set_prior` (2 runs) |
   | after step 3 | run 1 **deadlocked**, then 100 runs, 3 failed | the ABBA above; then blueprint -1010 ×3 + wrong-flush side effect |
   | **after step 4** | **100 runs, 0 failed, 0 deadlocks** | — |

Every step's fix was verified again by the full 175-test run after rebuild,
plus `cargo fmt --all --check`.

## What is deliberately NOT claimed

* The scan proves *lock-order* acyclicity; it says nothing about shared state
  guarded by no lock at all — those are found only by running, which is why
  the 100-run campaign is the acceptance, not the scan.
* `README.md`'s historical flake notes for the two named action tests remain
  as written; those tests were observed green throughout the final campaign
  but no cross-crate (full-workspace) campaign was rerun as part of A12 — the
  100-run protocol covers this binary alone.
