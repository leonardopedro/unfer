# PLAN H — Harness workstream (deepseek-harness / Cordis / qm inspired)

Cross-repo plan for the three-repo unfer system (`unfer`, `australVM`,
`velysterm`), borrowing engineering practice from the sibling projects
**`../deepseek-harness/`** (a Cordis-based agent harness) and **`../qm/`** (a
multiplayer agent harness). This is the fourth parallel workstream after
PLAN A (unfer), PLAN B (australVM), PLAN C (velysterm).

## Improvement principle

New features are welcome — but the first move is always to **consider and try
to improve what this project already does**. This matters more than it looks:
the system was inspired by **Theseus OS**, whose intralingual-safety module
lineage (`docs/TUTORIAL.md` §authorization) makes australVM *already a plugin
engine* — modules, `uk_*` capability symbols, `module.toml` grants, and the
loopback are exactly the "plugin engine" surface Cordis provides. Cordis is what
inspired this plan, and unfer already ships its analogue (Austral cells instead
of JS plugins, linear types instead of a permission sandbox). So a proposed
"new feature" is checked against that existing surface first: if the idea is a
plugin-engine idea, the answer is usually *not* a new mechanism but an
**improvement to the existing module/plugin path** (better discovery, richer
grants, a missing `uk_*` capability, a faster loopback hop).

The project is mature: the maintenance checklist in `AGENTS.md`, the S21–S35
security/validation stages, and Plan R are all built. So this plan leads with
**improvements over enlargement**:

- Every stage below applies the rule to its own item: it **names the existing
  feature it improves** and **reviews the related features this project already
  ships** before proposing anything. Each stage opens with that review — the
  existing feature, the concrete gap it closes in it, and why the answer is an
  improvement there rather than a new mechanism. Most stages **consolidate**
  (one persistence path, one symbol census, one archetype selection, one policy
  chain) instead of adding a parallel mechanism.
- Enlargement-only borrows (a skills registry, a deployment directory) are kept
  but marked **(E)**, ordered last, and justified against the existing surface.
- Ideas that duplicate an existing mechanism are **not adopted even as
  enlargement** (see the final section) — their useful kernel is folded into an
  improvement stage instead.

Grounding: the `AGENTS.md` maintenance checklist is the improvement backlog.
Every item below maps to a stage.

| Existing checklist item | Gap | Borrow | Stage |
|---|---|---|---|
| Quadratic ordering / LaTeX mapping / commutator / vacuum-init checks | manual, no gate | dsh package-invariants | H1 |
| S21 effect-kinds, S22 admin refuse list, S23 sanitizer, S26 latch | manual re-checks | package-invariants gate | H1 |
| S29 symbol registration (5 hand-maintained places) | cross-repo drift | dsh doc-sync / generated census | H2 |
| `Session` save/restore (SessionBlob) | opaque, no fork | dsh session log | H3 |
| Long sessions / editor history growth | unbounded | dsh compaction | H3 |
| S23 audit ring (512, drop-oldest, RAM) | RAM-only | qm durable-by-default | H4 |
| S25 metering = denial point, no deadline | hung call, no structured timeout | dsh timeout-policy | H6 |
| Plan R certs/auction invariants | replay/double-fire | qm idempotency + leader lease | H7 |
| modhost 3 archetype branches | bespoke selection | qm harness-router | H8 |
| S21/S25/S26 loopback `if`-chains | not composable/testable | Cordis waterfall | H5 |
| Security primitives exist, no posture | no org/scope posture, no screening | qm posture + classifier | H9 |
| GrantSet per-session inline | no named reuse | dsh agent-presets | H10 |
| ~348 tests, golden gate only | no coverage, no replay | dsh testing | H11 |
| PLAN files as the memory | no decision log, doc drift | dsh Agent Notes + doc-sync | H12 |

## System context

- **unfer** — the kernel: `prob_kernel::Session`, `unfer_ffi` (65 `uk_*` +
  `uz_*` symbols; loopback with S21/S23/S25/S26/S27/S28), `unfer_protocol`
  (serde + UK-####), QFT engines, `logos`, `ode_sirk`, `unfer_consensus`
  (QuePaxa + cert/auction ledgers), `unfer_identity`, `unfer_data`,
  `unfer_edge`, `unfer_taler`, `qfm`/`qfm_text`, 8 Austral modules. A1–A5 done,
  A7 partial, A8–A10 open; S29–S35 + Plan R complete.
- **australVM** — module runtime: Austral compiler + `safestos/cranelift` JIT
  registering `uk_*`, `AuthorizationEngine`, `modhost`, three module archetypes
  (Austral cells, Tidepool Haskell effects, cap-std Rust), cloud-hypervisor VM
  tier. B1–B11 complete.
- **velysterm** — UI/AI interface: `kernel_client` (`unfer_agent` NDJSON, 36
  ops), `mathed_core` (Loro CRDT), `mathed_mini` (Bevy-free), `mathed` (Bevy),
  `mathed_biblio`. C1–C18 complete.

## Parallel-execution rules (shared with A/B/C)

1. **Ownership**: modify only files inside the owning repo per stage.
   Cross-repo *reads* fine; cross-repo *writes* forbidden except `[SYNC]`.
2. **Frozen contract** (additive-only): `uk_*`/`uz_*` symbols + C signatures;
   `prob_kernel::Session` public API; `unfer_protocol` serde types + UK-####
   assignments; NDJSON agent ops; `module.toml` grant vocabulary. New symbols
   follow the S29 checklist.
3. **Commit discipline**: meaningful messages; commit after every stage.
4. Stages ordered small → large; each ends in a verifiable acceptance command.
5. **Green-workspace rule**: both dependents path-dep on unfer's working tree —
   keep unfer green at all times; `[SYNC]` writes land last.

## Current state (2026-08-20)

- unfer green; loopback already has effect-kind approval
  (`unfer_ffi/src/handles.rs:360`), metering, latch, audit, vault, capability
  RPC — H5/H6/H9 compose with these rather than replacing them.
- australVM B1–B11 complete; velysterm C1–C18 complete.

---

## H1 — Checklist-as-gate: maintenance invariants (S) — *unfer + australVM + velysterm*

**Improves**: the entire `AGENTS.md` maintenance checklist — currently manual
discipline. Turns each item into a machine-checkable invariant that fails CI.

**Existing-feature review**: this stage does not add a new safety mechanism; it
makes the *existing* checklist (`AGENTS.md` maintenance sections) executable.
The related feature is the checklist itself and the one-line gates already
spread through the test suites — H1 consolidates them under one runner instead
of a parallel tool.

1. Port each checklist item into a `scripts/verify-invariants` gate (dsh
   package-invariants analog). A gate checks an **owned relationship**, never a
   service's presence. Examples:
   - `compile_expression` still strips zero-point scalars (quadratic ordering);
   - `compile_latex` maps `a_i^\dagger` → creation, `a_i` → annihilation;
   - non-commuting operators are never reordered (no `.simplify()` where order
     matters);
   - `GrantSet::is_subset_of` denies `Mutate → Observe` relabeling (S21);
   - the S22 admin refuse list (`grants`/`auth`/`storage`/`backend`) rejects
     patches and leaves the soft config byte-identical on refusal;
   - `sanitize_sensitive` covers every secret-plausible field (S23);
   - `QuantumState::vacuum()` has ≥ 1 empty inner universe;
   - a `sensitive: true` observation latches the caller set (S26);
   - the meter (UTC-day key) is the single denial point for UK-4601/4602 (S25);
   - new `KernelEvent` variants have a `handles.rs` event-type matcher (S29);
   - secrets never serialize into a `SessionBlob` snapshot or `.cell` blueprint
     (S27).
2. Each gate is a small Rust test or scripted check with a one-line failing
   message naming the checklist item; the gate output lists items that ran.
3. Where an item has no plausible relationship yet, the gate records an
   explained empty reason rather than being deleted (dsh rule: an explained
   empty companion is correct).

**Acceptance**: `scripts/verify-invariants` passes on a clean tree, fails on a
deliberately introduced violation, and lists every executed checklist item.

## H2 — Single source of truth for the symbol/op/UK-code census (M) — *unfer + australVM*

**Improves**: the S29 registration checklist — five hand-maintained places that
drift across three path-dep repos (`EXPECTED_SYMBOLS.txt`, generated C header,
`UNFER_SYMBOLS`, `GrantSet.kernel`, `handles.rs` matchers), plus
`docs/PROTOCOL.md`.

**Existing-feature review**: the census already exists (S29 and `scripts/
verify-invariants`); what is missing is that it is hand-maintained in five
places. H2 keeps the same artifacts and meanings and only changes their
*provenance* (one generated table instead of mirrors) — it improves the
existing census rather than introducing a new registry concept.

1. Introduce one `symbol registry` table in `unfer_protocol` (data, not code):
   `{ symbol, arity, flags, timeout_ms?, effect_kind?, audit_secret_fields?,
   grants_kernel: bool }`. `scripts/gen_symbol_artifacts` generates
   `EXPECTED_SYMBOLS.txt`, the C header, australVM's `UNFER_SYMBOLS`, and the
   `GrantSet.kernel` namespace **from this table** — a generator, not a
   hand-edited mirror.
2. A doc-sync gate fails when `docs/PROTOCOL.md`'s op/UK-code census drifts
   from the registry (dsh doc-sync analog); `AGENTS.md` keeps the pointer, not
   the copy.
3. The registry becomes the single registration point for new `uk_*` symbols:
   adding a row regenerates every artifact, and the H1 invariant gate verifies
   the generated files are current (dsh: wire mechanically checkable invariants
   into an executed gate).

**Acceptance**: editing one registry row and running the generator reproduces
all five artifacts byte-identically; the doc-sync gate fails on intentional
drift; `EXPECTED_SYMBOLS.txt` unchanged in content, changed in provenance.

## H3 — Event-sourced Session: log, fork, compaction (M–L) — *unfer*

**Improves**: `prob_kernel::Session` `save`/`restore` — an opaque `SessionBlob`
that is not reconstructable and cannot fork. Consolidates one persistence path
and bounds editor-session growth.

**Existing-feature review**: the persistence path already exists (`save`/
`restore`/`snapshot` + the `SessionBlob`); it is opaque and unforkable. H3 makes
the *existing* log explicit and adds fork/compaction on top of it, folding in
the dsh subagent cold-resume kernel because the module runtime already provides
delegation under module principals (a separate session-level subagent surface
would duplicate it). No second persistence mechanism is introduced.

1. Each kernel op appends a typed record `{ seq, op, spec, source, ts }` before
   it applies. `snapshot()` becomes `fold(events)`; `restore()` = replay. The
   public `Session` API and the FFI stay byte-identical; the log is internal.
   Versioned format marker (dsh `SESSION_FORMAT_VERSION`) — bumps only on
   structural change; malformed/older events fail with a UK-1xxx code.
2. Add the **model-visible ⟺ logged** invariant as a debug assertion: every
   value a `probability`/`condition`/`snapshot` depends on is reconstructable
   from the log alone (dsh runtime invariant).
3. `uk_session_fork` (additive): fork from a log boundary `{ seq }`; both
   sessions share the prefix and diverge after. This is the useful kernel of
   the dsh subagent cold-resume idea — folded here because the module runtime
   already provides delegation under module principals, so a separate
   session-level subagent surface would duplicate it.
4. **Compaction** (dsh-compaction analog, trimmed): a log-recorded lock
   bracket (`compaction/start`…`compaction/end`); an older surface range is
   summarized into one replacement node; raw events stay in the log, derived
   history uses the summary. Tool-pairing safety: never cut a range that
   splits an unanswered `action_apply`/`evolve` dependency. `uk_session_compact`
   (additive) requires an idle session; a crash between `start` and `end`
   leaves a detectable orphaned lock that fails loud.
5. Tests: round-trip (fold ≡ live); fork → diverge; compact → derived history
   shrinks while full replay still reproduces the original `snapshot()`.

**Acceptance**: `cargo test -p prob_kernel -p unfer_ffi` green; fork and
compaction round-trips exact; version-marker rejection tested; symbol/op
census unchanged.

**Status (H3, DONE)**: `prob_kernel/src/session.rs` is event-sourced.
`SessionEvent { seq, op, spec, source, ts }` + typed `SessionEventSpec`
(`Create`/`SetPrior`/`SetHamiltonian`/`Evolve{qfm,query}`/`Condition`/
`CompactStart`/`CompactEnd`), `SESSION_FORMAT_VERSION = 1`; every mutator
appends before it applies and rolls back on error. `save()` folds the derived
log into `SessionBlob{format_version, events, …}` (legacy blobs still restore
via the folded path); `restore()` = replay with bracket-balance + orphaned-lock
validation. `fork_at(seq)` replays the prefix (refuses open-lock boundaries);
`compact_start`/`compact_end`/`compact_through` bracket a summarized range that
`derived_log()` replaces with a single `CompactEnd` summary node. Debug-build
`debug_assert_reconstructable` pins `fold(events) ≡ live` after every op.
New UK codes 1006–1009 (`SessionLogVersion`/`SessionCompactionOrphaned`/
`SessionCompactionBusy`/`SessionForkRange`) wired into `KernelError` +
`to_diagnostic` + PROTOCOL.md. New FFI symbols `uk_session_fork`/`uk_session_compact`
registered through the H2 census (EXPECTED_SYMBOLS.txt, generated C header,
australVM `UNFER_SYMBOLS` + `ecma.rs` dispatch arms). Tests: 9 prob_kernel
(fold≡live round-trip, legacy-blob fallback, version rejection, fork/divergence,
derived-log shrink, evolve-boundary refusal, busy-lock rejection, orphaned-lock
restore) + 2 unfer_ffi fork/compact round-trips. All gates green.

## H4 — Durable-by-default: audit, config, queue (M) — *unfer*

**Improves**: `uk_audit_append`'s RAM-only ring (`OWNER_LOG_CAPACITY` 512,
drop-oldest) and any in-memory resolved config / queued work. Matches qm's
"durable by default — RAM is a cache, never the source of truth."

**Existing-feature review**: the audit ring, owner log, resolved config, and
queued approval work already exist; only their durability is missing. H4 keeps
the same ring reads (write-through cache) and same UK codes, and adds a
`DurableStore` trait behind them — Loro is the default/preferred backend
because the project already ships Loro (velysterm's `mathed_core` CRDT);
JSONL/SQLite are alternatives implementing the same trait, used only where they
fit better, never as mirrors (a stream lives in exactly one store). This
extends the existing persistence path instead of adding a new one.

1. A `DurableStore` trait behind the event log: JSONL (default) + SQLite
   backends. Audit entries, resolved posture/config, and queued approval work
   persist through the store; the existing ring becomes a read-through cache
   in front of it. Nothing the operator or an agent reads back stays RAM-only.
2. **Checkpoint policy** (dsh session-checkpoint-policy analog, folded): flush
   before a model-facing `probability`/`condition` is served, and before a
   top-level `uk_action_apply` may produce an external side effect. Fail-closed:
   a checkpoint rejection fails the call before dispatch. Crash recovery
   surfaces an `UnknownOutcome` result (new UK code) — retry for read-only
   work, verify for side-effecting work.
3. Tests: kill-and-resume reproduces state; interrupted side-effecting call
   reports `UnknownOutcome`; concurrent flushes share one serialized drain;
   JSONL and SQLite both pass the round-trip suite.

**Acceptance**: `cargo test -p unfer_ffi -p prob_kernel` green with the
durability suite; audit survives a process restart; `EXPECTED_SYMBOLS.txt`
updated for the new UK code.

## H5 — Loopback as composable waterfall listeners (M) — *unfer*

**Improves**: the `unfer_ffi` chokepoint (`ffi_entry`), where S21/S23/S25/S26
are bespoke `if`-chains. Consolidates policy into registered, testable
listeners **without changing any behavior, code, or order**.

**Existing-feature review**: the loopback chokepoint is the project's own
plugin-engine capability surface (modules → `uk_*` symbols → policy). The
effect-kind approval, metering, latch, and audit `if`-chains already exist and
are correct; H5 only re-shapes *how they are wired* (waterfall listeners over
the same chokepoint, same order, same codes) so they become composable and
testable. This is the Cordis dispatch model applied to the existing engine, not
a new dispatch path.

1. Introduce the Cordis dispatch model at the loopback (`emit` /
   `waterfall` / `parallel` / `serial`) as an internal trait with the dispatch
   mode part of each symbol event's contract.
2. Refactor metering (S25), latch (S26), effect-kind approval (S21), and audit
   (S23) into waterfall listeners on a `symbol/*` event. A listener owns a
   decision by returning without `next()`; annotation-only listeners must
   delegate. Registration order = enforcement order, and a regression test pins
   the exact UK-code sequence per symbol.
3. The C ABI, `handles.rs`, and emitted codes stay byte-identical — this is a
   refactor-of-record; H1's invariant gates are the safety net.

**Acceptance**: `cargo test --workspace` green with a new listener-order test
asserting the same codes fire in the same order as before the refactor;
australVM/velysterm path-dep `cargo check` green.

## H6 — Deadline guard composing with the existing meter (S–M) — *unfer + australVM*

**Improves**: S25 metering — currently a denial point (rate/budget) with no
deadline. A hung call has no structured timeout at the loopback; only the
Tidepool path has a watchdog (`max_ms`).

**Existing-feature review**: the meter already exists and stays the single
denial point; H6 adds the *complement* it lacks (a deadline) without changing
metering. The timeout vocabulary also already exists in `module.toml`'s
`[limits] max_ms` — H6 shares that existing vocabulary with the loopback rather
than inventing a parallel deadline config.

1. Add `timeout_ms` to the H2 symbol registry (declared by the owning plugin,
   never a registry-wide default). A `guard` listener (composed in H5) arms a
   cooperative per-call deadline on symbols that declare one and, on expiry,
   returns a structured `UK-4603 TOOL_TIMEOUT`-family result instead of the
   raw completion. Cooperative, not a hard kill — termination stays with the
   backend; only signal-forwarding symbols declare it (dsh timeout-policy).
2. australVM reads the same declaration so `modhost`'s Tidepool `max_ms` and
   cap-std deadlines share one vocabulary; `[limits] max_ms` in `module.toml`
   overrides the symbol default for that module.
3. Composition order with the meter is deterministic and tested: the meter is
   the single denial point, the guard the single deadline point.

**Acceptance**: a deliberately slow op times out with the structured code; an
undeclared op is untouched; timeout + metering composition test green in both
`unfer_ffi` and `safestos/cranelift`.

## H7 — Consensus/auction idempotency + leader lease (M) — *unfer*

**Improves**: Plan R's `CertificateLedger` and `AuctionLedger` invariants
(UK-7002 conservation, UK-7004 no double-spend, UK-7005 owner-only spend,
UK-7001 mint authority, deterministic auction clearing). In a distributed
setting these need protection against replayed/duplicated delivery and
double-fired schedulers.

**Existing-feature review**: the ledgers and their invariants already exist and
are deterministic (same log → same root). H7 does not add a new ledger; it adds
the distributed-delivery protections (idempotency, lease, job queue) *around*
the existing `ConsensusNode::sync` and settlement, backed by the H3 event log
that H4 already made durable. The invariants the ledgers already enforce stay
exactly the same.

1. `IdempotencyStore { once(key, fn), committed(key) }` (qm
   `src/idempotency/idempotency-store.ts`) backed by the H3 event log; apply to
   every `CertificateOp` (Mint/Transfer/Burn) and `AuctionOp` (Open/Bid/Close)
   so a duplicated or replayed transaction applies exactly once. Retention
   prune on a schedule.
2. Leader lease (qm `persistence/leader-lease.ts`) for `ConsensusNode::sync`
   and auction settlement: exactly one node fires per tick; a lost lease stops
   firing without corrupting state.
3. Job queue with `claimSlot`/`unclaimSlot` semantics (qm `cron/scheduler.ts`)
   for scheduled consensus/auction work; `markFired` only after a successful
   fire; disable the job on authz-fail (mirror qm).
4. Tests: double-submitted transfer applies once and conservation holds;
   two nodes → single leader fires; failed fire re-queued and retried.

**Acceptance**: idempotency + lease + job-queue tests green in
`unfer_consensus`; all UK-7001..7007 invariants still hold.

## H8 — Module archetype harness registry (M) — *australVM + unfer*

**Improves**: `modhost`'s bespoke branching over three module archetypes
(Austral cells, Tidepool Haskell effects, cap-std Rust) and the manual
capability-RPC re-check (S28). Consolidates selection into one resolver.

**Existing-feature review**: this is the plugin engine itself — the Theseus-OS
lineage (`docs/TUTORIAL.md` §authorization) already makes the project a plugin
engine, and `modhost`'s three archetypes plus `module.toml` are its plugin
slots. H8 does not add a fourth plugin mechanism; it consolidates the
*selection* of the existing three archetypes into one `resolve_runtime_choice`
resolver (and a degenerate kernelless fourth profile that only reads the
existing `snapshot`/`probability`). The existing archetype adapters, grants,
and the S28 re-check stay — only the bespoke branching is replaced by a
registered resolver.

1. Define `HarnessProfile { id, control_transport, tool_transport,
   transcript_format, capabilities }` in `unfer_protocol`; each archetype
   registers as an adapter. A "kernelless" profile that only reads
   `snapshot`/`probability` is a degenerate fourth (qm harness.ts).
2. `resolve_runtime_choice(approved, org, scope, fallback, requested)` (qm
   harness-router.ts): approved set (module.toml / operator config), org floor,
   scope override, fallback, non-retryable rejection when the requested
   archetype is not approved. Record the resolved choice durably on the module
   handle so a cold restart re-runs the same archetype.
3. A returned capability stub is re-checked against the original caller (S28)
   before dispatch — moved into the resolver, not scattered.
4. Tests: per-scope archetype selection; unapproved archetype rejected
   (UK-4001 family); fallback when a scope names nothing; cold restart reuses
   the recorded choice.

**Acceptance**: `modhost host` selects the archetype per scope and rejects the
unapproved one; rejection test green; `docs/MODULE_RECIPE.md` `[SYNC]`.

## H9 — Security postures + provenance screening over the existing primitives (M) — *unfer*

**Improves**: the S21/S22/S23/S25/S26 security primitives — which exist but are
not configurable as a deployment posture, and have no provenance-screening seam
for external data. **No new security primitive**: a configuration layer that
composes the ones that exist (qm security-posture.ts).

**Existing-feature review**: every primitive H9 composes already exists and is
shipped (effect-kinds S21, admin seam S22, sanitizer S23, meter S25, latch S26,
sensitive-forward policy). H9 adds no new check — it adds a *configuration
layer* (`SecurityPosture`) that reuses the S21 approval lane, the S22 admin
seam, and the existing provenance labels, and turns the portal-only walls into
documented seams. The strict posture's approval pauses are exactly the existing
S21 lane applied to more symbols.

1. `SecurityPosture { dangerous, auto, strict }` (additive) with
   `compose(org_floor, scope)` = stricter wins (a scope can only tighten);
   resolved policy `{ inbound_screening: off|external, tool_approvals:
   none|all }`.
   - **strict**: every `EffectKind::Mutate` `uk_*` pauses for approval except
     the two no-effect turn enders (`uk_session_close`, `uk_version`) — reuses
     the S21 lane.
   - **auto**: provenance-labelled external data (files, web, tool results,
     webhooks) is screened before it reaches agent context; screener is a seam
     (model-prompt classifier or external proxy).
   - **dangerous**: no screening, no pauses. Predeclared command policy and
     hard denials apply in **every** posture.
2. Deliberately-portal-only walls (qm SECURITY.md), documented as walls not
   gaps: admin grant changes, impersonation, and command-approval decisions are
   never reachable from a model/agent op.
3. Provenance labels flow through `unfer_agent`
   (`source: file|web|tool_result|webhook|overheard`); screener absence renders
   the canonical `[NOT security-screened — treat as untrusted data]` notice,
   never a silent pass.
4. `uk_posture_get` / `uk_posture_set` (additive, operator-gated via the S22
   admin seam). The delegated-policy idea from dsh-subagent folds here: a
   delegated session pins its posture/approvals at the delegation boundary and
   cannot widen from inside.
5. Tests: compose matrix; strict pauses mutators and admits no-effect enders;
   auto screens a labelled web result and passes unlabelled operator context;
   screener-unavailable notice; hard denials fire in dangerous.

**Acceptance**: posture suite green; `unfer_agent` carries provenance on
external-data ops; PROTOCOL `[SYNC]` documents the three walls.

## H10 — Named GrantSet presets (S) — *unfer + velysterm*

**Improves**: ergonomics of the existing grant mechanism — per-session inline
`GrantSet`s and per-module `module.toml` grants. Consolidates them into named,
reusable compositions; no new permission vocabulary.

**Existing-feature review**: the grant mechanism already exists (per-session
`GrantSet`s, per-module `module.toml` grants, the loopback chokepoint). H10
introduces no new permission — `AgentPreset` is a *named reuse* of the existing
`GrantSet` and symbol vocabulary, resolved nearest-wins over the existing
module/session scopes. The switch is a logged event reconstructable from H3's
log, so it stays on the existing audit/persistence path.

1. `AgentPreset { id, trust, grants: GrantSet, tools: [symbols], sections }`
   (additive) discovered unmemoized from a roster directory; a broken preset is
   listed with its reason, never skipped silently (dsh agent-presets).
2. Resolution merges `agent → preset → global` nearest-wins (mirror dsh-scope);
   the session header records the start preset, `resolve_session_preset` reads
   the live chain, and a switch is valid only while the session has produced
   nothing. The switch is a logged event (reconstructable from H3's log).
3. velysterm: `preset_list`/`preset_set` agent ops (additive, next free codes);
   `mathed_mini` shows the active preset id beside the model overview panel.
4. Tests: two sessions on different presets see different tool surfaces; switch
   on a non-blank session is rejected; broken preset surfaces its reason.

**Acceptance**: `cargo test -p unfer_ffi -p kernel_client` green; `unfer_agent`
`preset_list` round-trip; PROTOCOL `[SYNC]`.

## H11 — Test discipline: coverage, real entry path, keyless replay (M) — *all three*

**Improves**: the ~348-test suite and the single golden gate. Adds the dsh
tiers that catch the "green unit tests, broken product" class.

**Existing-feature review**: the golden-gate pattern already exists (S23/S24
release gate, `UPDATE_GOLDEN=1` regeneration); H11 extends that *existing*
pattern to `unfer_agent` transcripts and adds a per-file coverage gate on the
existing `prob_kernel`/`unfer_ffi` sources. No new test framework is
introduced — the existing unit-test harness and golden-gate scripts are
reused.

1. **Keyless snapshot replay** for `unfer_agent`: record NDJSON transcripts
   (create_model → evolve → probability → bayesian_update → close) as committed
   fixtures; a replay runner boots the real binary, diffs normalized output +
   the re-derived event log (H3). Regeneration only via `UPDATE_GOLDEN=1`
   (extends the existing golden-gate pattern). A transcript change updates a
   fixture, never a normalizer.
2. **Per-file coverage gate** on `prob_kernel/src` and `unfer_ffi/src`: an
   uncovered line is usually dead code, not a missing test. CI gate with
   self-skip exemptions for backend-dependent files (cadabra2, CUDA, GHC)
   exactly like dsh's `pwsh-local` exemption.
3. **Real-entry-path smokes**: the `unfer_ffi` cdylib, the `unfer_agent` bin,
   and `modhost` run from built output (not source), so stale-artifact and
   masked-settle failures surface; assert a genuinely missing config exits
   non-zero.
4. **Prefer real over mock**: only the LLM/CUDA/network boundaries are mocked;
   keep everything downstream real (dsh testing.md).

**Acceptance**: snapshot gate green keyless; coverage gate green; built-
artifact smokes green in all three repos.

## H12 — Agent Notes + doc-sync + duplication + review culture (S) — *all three*

**Improves**: the PLAN files as the shared memory, and hand-maintained docs
that drift from code.

**Existing-feature review**: the doc-sync gate already exists (H2) and the PLAN
files already exist as the shared memory; the `AGENTS.md` maintenance sections
already encode the invariants. H12 adds the *notes* directory and a
`duplication` script — both over existing files (PLAN/AGENTS/PROTOCOL) — and
reuses the existing H2 doc-sync gate for CI wiring. The review culture rule
("fix every instance, solve at the layer all paths flow through") is a process
habit applied to the existing codebase, not a new tool.

1. Add `.agents/notes/implemented/{architecture,feature,testing,process}/` to
   each repo; a non-trivial change ships with one note in the same PR; archived
   notes are frozen (dsh notes policy).
2. Wire the H2 doc-sync gate into each repo's CI (PROTOCOL / ARCHITECTURE /
   AGENTS.md / EXPECTED_SYMBOLS.txt vs code census).
3. A `duplication` script flags copy-pasted FFI/NDJSON/UK-code blocks across
   the three workspaces; plus qm's review culture: never merge without a
   fresh-context review pass that tries to break the change; fix every instance
   of a pattern; solve at the layer all paths flow through.

**Acceptance**: notes directory + seed note per repo committed; doc-sync gate
fails on deliberate drift and passes after fix; duplication report has zero
actionable hits.

---

## Enlargement-only borrows (kept, lower priority)

### H13 (E) — Skills registry (M) — *unfer + velysterm*

**Existing-feature review (first)**: modules *are* the project's skills — this
is the plugin engine itself (the Theseus/Cordis lineage: australVM is already a
plugin engine, `module.toml` + `modhost` are its plugin slots, `uk_*` symbols
its capability surface). Before a registry, the review asks what the existing
module path already provides and what it actually lacks:

- **Has**: authoring (`docs/MODULE_RECIPE.md`), loading/runtime (`modhost`, H8
  archetypes), grants (`module.toml` `[grants]`), capability invocation
  (`uk_*`), an H1 gate, a census (H2).
- **Lacks**: *discovery and sharing* — a way to find and reuse a module that is
  already loaded or packed, without re-authoring it.

So H13 is justified only as the **discovery/sharing improvement to the existing
module path** — it adds `uk_skill_list/get/register/pack_import` (additive),
scope-owned skills shareable by grant, admin-gated promotion, git-importable
packs. It does **not** add a second plugin-loading mechanism: packs land as
`module.toml` cells loaded by the existing modhost, the `\skill` PropKind
renders the existing skill surface, and the four-combination invocation policy
reuses the existing grant vocabulary. If discovery is ever served natively by
modhost, this stage shrinks to that.

**Acceptance**: `uk_skill_*` registered per S29; agent ops round-trip; catalog
panel renders; PROTOCOL `[SYNC]`.

### H14 (E) — Deployment directory + onboarding (S–M) — *unfer (docs)*

**Existing-feature review (first)**: Plan R (taler/certs/auction/edge) and the
VM/module tier already exist and ship; what they lack is a deployment *guide*
and an org-material *convention*. Before adopting a parallel deploy structure,
the review confirms the core stays byte-identical (the S23/S24 golden gate is
the existing guard) and the onboarding skill reuses the existing module/`uk_*`
path rather than a new installer.

Borrow qm's `deploy/layers/<org>/` convention (org material out of byte-
identical core, private-fork sync skills, `qm init`-style onboarding skill with
a final live verification). `deploy/layers/<org>/` in unfer stays empty;
acceptance is a dry-run `tools/onboard_federation.sh --org <slug> --dry-run`
that prints steps without writing to core.

**Acceptance**: `docs/DEPLOYMENTS.md` written; onboarding dry-run passes;
core `git status` clean.

---

## Not adopted (even as enlargement)

Each idea below was run through the existing-feature review first: the answer
"already done, only improved here" sent it to an H-stage; "would duplicate an
existing mechanism" left it here with its useful kernel folded into a stage.

- **Session-level subagent surface**: the module runtime already delegates under
  module principals with grant sets; a parallel session-level subagent would be
  a second delegation mechanism. The two useful kernels — cold resume and
  delegated-policy pinning — are folded into H3 (fork) and H9 (posture
  delegation).
- **Plan mode (dsh plan-mode)**: `unfer_agent` is a request/response NDJSON
  interface with no multi-turn tool-driving agent loop to hold a plan state;
  the project's planning already lives in QFMplan.md / PLAN files.
- **Boot-level bundles/profiles**: `module.toml` already composes module
  surface; a second composition layer would duplicate it.
- **Web-app / portal / Slack surfaces (qm)**: out of domain — the project's
  frontend is velysterm.
- **Session projection / telemetry registries (dsh)**: valuable only once an
  agent loop exists (see plan mode); the H3 fold already covers replayable
  derived state.

---

## Out of scope (owned elsewhere)

- unfer QFM research (A6, hypothetical), new-crate docs (A8), cross-repo
  integration test (A9), logos research (A10).
- australVM: anything beyond H6 budget vocabulary and H8 archetype selection.
- velysterm: editor UX redesign, further collab features, GPU experiments.

`[SYNC]` steps:
1. After H9/H10/H13 land: `../unfer/docs/PROTOCOL.md` + `../unfer/docs/
   ARCHITECTURE.md` (postures, presets, skills — additive paragraphs).
2. After H8 lands: `../unfer/docs/MODULE_RECIPE.md` (archetype-selection
   vocabulary).
3. After H3/H4 land: event-log + durability notes in `../velysterm/AGENTS.md`
   and `../australVM/AGENTS.md` kernel-facing sections.