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
| `BLUEPRINT_TESTS_LOCK`, `METER_FFI_TESTS_LOCK`, `SPAWN_PACING_LOCK` | `unfer_ffi/src/lib.rs` | their own single-store tests |
| `AUCTION_TESTS_LOCK`, `CERT_TESTS_LOCK`, `METER_TESTS_LOCK`, `VAULT_TESTS_LOCK`, `PACE_TESTS_LOCK` | `unfer_ffi/src/handles.rs` | their own single-store tests |

The majority order among multi-lock tests is **ACTION → AUDIT → DURABLE** (the
acquisition block is written out at six sites in `lib.rs` and carries its own
comment saying why). One test — `audit_trail_never_persists_fixture_secret` —
takes **AUDIT → ACTION**, because the missing lock that fixed its failure was
appended after the existing one. Reordering it to match was tried and made
things measurably worse, so the order is left as it is on the evidence, not on
the principle.

## Inventory: who touches the **owner log**

This is the first store completed, because it produced a *third* flaky test on
2026-10-06 (below) and the question "who can empty the ring?" had an answer
that inspection could settle.

| test | lock held | owner-log operation |
|---|---|---|
| `queue_drops_oldest_when_full` | `AUDIT_AGENT_TESTS_LOCK` **only** | `clear_owner_log()`, then **asserts its own drop notice is in the ring** |
| `owner_logger_writes_dot_component_lines` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` ×2 |
| `report_issue_is_noop_without_error_report_binding` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` |
| `report_issue_bound_appends_sanitized_line` | `ACTION_TESTS_LOCK` **only** | `uk_owner_clear()` |
| `audit_trail_never_persists_fixture_secret` | `AUDIT_AGENT_TESTS_LOCK` → `ACTION_TESTS_LOCK` | `uk_owner_clear()` |

**The gap, named:** the first row and the next three hold *different* mutexes.
Nothing prevents an ACTION-only test from clearing the ring between
`queue_drops_oldest_when_full`'s clear and its assertion — so the assertion can
read an empty ring and fail with `got: []`, which is exactly what happened.

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

## The fix, and why it is not done here

Two candidate fixes, both touching lock acquisition, both deliberately not
applied in this pass:

1. give `queue_drops_oldest_when_full` `ACTION_TESTS_LOCK` as well (order
   ACTION → AUDIT, matching the majority);
2. give the three ACTION-only *clearers* `AUDIT_AGENT_TESTS_LOCK` as well.

Either creates a new ACTION+AUDIT pairing in a file that already holds one
test in the opposite order, and this codebase has already been burned twice by
mechanical lock edits (§X-F7). The safe path is the one the plan names:
finish the inventory — the action store and the audit ring still need the same
table — then pick **one** order, convert every multi-lock test in a single
change, and measure over enough runs to distinguish a fix from a coin flip.

Until that is done: the flake is documented rather than hidden, `README.md`
names three tests instead of two, and no status line cites a run in which one
of them failed.
