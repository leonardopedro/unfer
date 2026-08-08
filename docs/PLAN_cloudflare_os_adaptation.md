# Implementation Plan: Adapting Cloudflare OS concepts to unfer

> Status: proposal · Scope: `unfer` + `australVM` + `velysterm`
> Source of inspiration: `../cloudflare-os` (Apache-2.0)
> Primary feature requested: **ECMAScript modules** alongside native Austral, cloud-hypervisor, and Haskell runtimes.
> Runtime decision: **workerd** (Apache-2.0, V8) as a **sidecar process — no VM** (see §2).
> Threat model: **web-browser-equivalent** (engine isolate + OS containment via user
> namespaces / seccomp-bpf / no_new_privs / Landlock / cgroups).

---

## 0. Executive summary

`cloudflare-os` is an "AI operating system" (kernel = `workshop-backend`, drivers =
`gatekeeper-*`, shell = `workshop-frontend`, processes = *gadgets*, executables =
*blueprints*) built on Cloudflare Workers. Its most valuable, transferable ideas are:

1. **A sandboxed user/AI-authored code runtime** (executes JS inside an isolate with
   `globalOutbound: null`, exposing only capability loopback bindings). — *This is the
   ECMAScript module backend.*
2. **Capability-based security with deferred approval + local simulation** ("Gatekeepers")
   — agents keep working while side effects wait for a human, seeing *simulated* results.
3. **Per-instance isolation via "Facets"** (each gadget runs in its own runtime instance).
4. **Blueprints** (shareable, instantiable snapshots of app code).
5. **Capability-minting at a single chokepoint** (default-deny, `user.ts:getGatekeeperClassFor`).
6. **Cap'n Web RPC** with promise pipelining and capability return values.

The unfer stack already has the *authorization seam* the port needs: `module.toml` grants →
`ManifestAuthEngine` → `auth::check` → `cps.rs::check_call_permission` (the single choke point),
plus a `uk_*` C ABI and an event stream (`uk_subscribe`/`uk_poll`). The port is therefore mostly
**adding a new module backend** and **reusing/broadening the existing auth + event machinery**,
rather than inventing new security infrastructure.

**Engine choice for ECMAScript modules:** **workerd** (Apache-2.0, V8) run as a **sidecar
process (no full VM)** — the exact runtime cloudflare-os runs on, giving faithful feature
parity and a built-in capability-binding model that maps onto unfer's `[grants]`. To match
the **web-browser threat model**, workerd is wrapped in the same OS containment Chromium puts
around V8 — user namespaces, `no_new_privs`, seccomp-bpf, Landlock, cgroups — so an engine
escape lands in a confined, low-privilege process (like a sandboxed renderer), not the host.
Modules are treated as **untrusted (browser-equivalent)**; the capability boundary
(default-deny, no ambient net/fs, grant-checked `uk_*`) is the explicit authority surface.

---

## 1. Licensing analysis (read this first)

| Concern | Verdict |
|---|---|
| `../cloudflare-os` license | **Apache-2.0** (confirmed: `LICENSE` is the standard Apache-2.0 text). Permissive, not source-available (no BSL/Elastic). |
| Compatibility with unfer | unfer = Apache-2.0, australVM = Apache-2.0, velysterm = `MIT OR Apache-2.0`. **All compatible with Apache-2.0.** |
| Obligations if we copy code | Per Apache-2.0 §4: retain a copy of the license, **mark modified files**, and include a **NOTICE** if upstream shipped one. cloudflare-os ships **no NOTICE**, so we add our own attribution notice. |
| Contribution policy | cloudflare-os's README says it's "not seeking outside contribution" and restricts PRs. That is a *contribution* policy, **not a license restriction** — it does not block us using the code. We should **derivate/adapt, not upstream** (fork-and-own). |
| **The real licensing risk is the JS engine, not cloudflare-os** | **chosen runtime = workerd/V8** (Apache-2.0 + BSD-3-Clause) — license-clean. **QuickJS** = MIT (clean). **Boa** = MIT (clean). **Javy** = Apache-2.0 (clean). **Deno core** = MIT/Apache-2.0 (clean). **None are copyleft.** |
| Watch out for | `gatekeeper-*` deps that pull third-party SDKs or the `@modelcontextprotocol/client` (MIT) — fine to use, but keep attributions. The AWS **Cedar** policy engine already used in australVM is Apache-2.0 — a direct conceptual match for Cloudflare's policy model. |

**Bottom line:** adapting cloudflare-os concepts is license-clean. We must (a) add a
`NOTICE`/attribution in the plan's target crates, (b) mark any directly-copied files as
modified-Apache-2.0, and (c) use **workerd** (Apache-2.0) as the ECMAScript backend.

---

## 2. ECMAScript module backend — the headline feature

### 2.1 Runtime selection (decision: workerd sidecar, no full VM, browser-equivalent threat model)

**Decision:** use **`workerd`** (the exact runtime cloudflare-os runs on) as the ECMAScript
module backend, run as a **sidecar process** — **not** inside the cloud-hypervisor VM.

| Option | License | Model | Verdict |
|---|---|---|---|
| **workerd** *(chosen)* | Apache-2.0 | Server runtime (V8), driven by Cap'n Proto config, sidecar process | Exact Cloudflare parity; built-in capability bindings; standard Web APIs; matches `unfer_edge` sidecar shape. |
| **QuickJS via `rquickjs`** | MIT | Native embed, in-process | Lightweight fallback for trusted single-process modules; re-implements the sandbox seam. |
| **Boa** | MIT | Pure-Rust | Similar to QuickJS; slower. |
| **Javy** | Apache-2.0 | JS→WASM in Wasmtime | Optional hardened profile for the VM later. |
| **Deno core / V8** | MIT/BSD-3/Apache-2.0 | Heavy Rust crates | Overlap with workerd without its capability model. |

**Rationale for workerd-without-VM:**
- It is the **same engine and runtime** cloudflare-os uses, so feature porting is faithful.
- Its **capability-binding model** (capabilities not global namespaces; SSRF-immune) maps
  directly onto unfer's `[grants]` + `auth::check`.
- workerd is a **server runtime**, so it fits unfer's existing sidecar/edge architecture
  (`modhost` subprocesses, `unfer_edge` HTTP) rather than an in-process embed.
- **Security posture (target: web-browser threat model):** workerd's README warns it is *not*
  a hardened sandbox and recommends a VM for untrusted code. Instead of a VM, we **replicate
  the OS containment Chromium puts around V8** — the same primitives, no full VM. The browser
  threat model is: *V8 isolate* (engine isolation) **plus** *OS process sandboxing* (seccomp-bpf,
  user namespaces, `no_new_privs`, Landlock, cgroups). We deliver both around the workerd
  sidecar (see §2.3), so an exploit escapes the engine only into a confined, low-privilege
  process with no ambient filesystem/network — equivalent to escaping into a sandboxed
  renderer, not the host. Modules are treated as untrusted (browser-equivalent), and the
  capability boundary (default-deny, no ambient net/fs, grant-checked `uk_*`) is the explicit
  authority surface.

### 2.2 Integration points (grounded in the existing seams)

From the architecture research, the seams are:

1. **Manifest selector** — `ModuleManifest` already parses `[module] archetype`, defaulting to
   `"austral_cps"` (`australVM/safestos/cranelift/src/module.rs:62-66`). Add value
   `"ecmascript"` = workerd-served worker.
2. **`ModuleHost::load`** (`module.rs:150`) — branch on `manifest.archetype`: for `ecmascript`,
   materialize a workerd `config.capnp` (embedding the module's `.js` files + a harness) and
   spawn a `workerd serve` sidecar instead of `compile_cps_binary`.
3. **`ModuleHost::call`** (`module.rs:178`) — a parallel `call` path that RPCs the sidecar's
   entrypoint (Cap'n Proto / HTTP) instead of invoking a raw `extern "C" fn(i64...)` pointer.
4. **Authorization choke point** — map module `[grants]` to workerd **capability bindings**:
   each granted `uk_*` becomes a binding the worker can *see*; un-granted symbols are absent
   (default-deny, mirroring Cloudflare's capability-loopback and `globalOutbound: null`).
   Keep `auth::check(principal, "Call", symbol)` (`cps.rs:7-15`) as the authoritative gate on
   the host side too (defense in depth).
5. **Build/QA harness** — `unfer/tools/module_builder` + `run_demo.sh`: add an
   `ecmascript_module` example with a **positive** path (creates a model, evolves, computes
   probability) and a **UK-4001** negative path (calls an un-granted `uk_*` → `CALL_DENIED`).
6. **`symbol_sync.rs`** test stays authoritative for the symbol table; the workerd binding
   reuses the already-synced `UNFER_SYMBOLS`.

### 2.3 Sandboxing properties (engine + OS containment, browser-equivalent)

**Engine layer (workerd/V8 isolate):**
- **Default network-off** — mirror `globalOutbound: null`: the sidecar is launched with no
  outbound sockets; only `[grants] net = [...]` maps to a workerd outbound binding.
- **Resource caps** — wire `[limits] max_ms` (already parsed) into a workerd request timeout /
  CPU limit; memory via the isolate's configured limits.
- **Capability loopback** — each binding is a service stub to a host loopback that round-trips
  through `auth::check` and emits to the module's `uk_subscribe`/`uk_poll` queue — the exact
  Cloudflare "env bindings are loopbacks" pattern.
- **Process isolation** — each ECMAScript module runs in its own `workerd` sidecar process
  (one `config.capnp` per module), so a crash or runaway is contained and restartable.

**OS layer (replicating Chromium's renderer sandbox, no VM):**
- **User namespaces** — each sidecar launched in a private user+net+pid+ipc namespace
  (via `bubblewrap`/`bwrap` or a dedicated launcher), so the process has no ambient OS rights.
- **`no_new_privs`** — irrevocably disable privilege escalation inside the sandbox.
- **seccomp-bpf syscall filter** — allow only the syscalls workerd/V8 needs (deny `ptrace`,
  `mount`, `reboot`, `kexec_load`, `open_by_handle_at`, etc.), mirroring Chrome's renderer filter.
- **Landlock** — filesystem access confined to the module's own directory (and the granted
  `[grants] fs = [...]` paths); nothing else is reachable.
- **cgroups** — per-sidecar memory/CPU/pids limits so a runaway cannot exhaust the host.
- **No ambient network** — net namespaces are empty unless `[grants] net` grants a veth/bind.
- **Drop privileges** — run as an unprivileged uid/gid (no capabilities), consistent with the
  threat model that a compromised engine yields only a confined process.

This is the *same* set of mechanisms Chromium composes for renderers; the sole difference is
the absence of a full VM, which is acceptable because the OS layer above restores the
containment that bare workerd lacks.

---

## 3. Feature-by-feature adaptation plan

Each row: **Cloudflare feature → unfer adaptation**, with the primary integration point.

### F1. ECMAScript module backend  ★ (the requested feature)
- See §2. workerd sidecar (V8), no VM.
- **Status: S1 implemented** (2026-08). Deliverables landed:
  - `[module] archetype` selector — `ModuleManifest.archetype == "ecmascript"` routes
    `ModuleHost::load`/`swap` to a workerd sidecar instead of the CPS path.
  - workerd sidecar supervisor + `config.capnp` generator — `cranelift/src/ecma.rs`
    (`WorkerdPaths`, `EcmaSidecar`, `KernelLoopback`, `kernel_dispatch` covering all 21
    `uk_*` symbols, probe-then-copy marshaling). Config materialized as text
    `config.capnp`; `external` loopback address is bare `host:port`; socket port
    pre-picked via `--socket-addr main=<port>`.
  - Capability binding with grant-checked `uk_*` — generated `harness.mjs` builds
    `makeKernel(env)` strictly from the granted `service` bindings to
    `kernel-loopback` (F5, S9): the capability object IS the module's
    `[grants] kernel` set, so an un-granted `uk_*`/`uz_*` name is merely absent
    (not enumerable, not stubbed). The loopback re-checks `auth::check`
    host-side and is the only layer that emits UK-4001 (defense in depth).
  - Positive/F5 test module — `cranelift/tests/ecmascript_module.rs`
    (`ecmascript_capability_exposes_only_granted_symbols`, loopback deny).
    Skips when no workerd runtime is discoverable.
  - `modhost host --args-json <json>` for ecma entrypoint calls.
- **Remaining (future):** packaging workerd as a pinned Nix derivation behind
  `--features ecmascript` (docs section added to `MODULE_RECIPE.md`, 2026-08).

> **S1 refinement (runtime split, 2026-08):** `ModuleHandle` now carries a single
> `runtime: IrRuntime` enum (`Jit { functions, cps_data }` / `Ecma(EcmaSidecar)`) instead of
> separate `functions`/`cps_data`/`ecma` fields. `IrRuntime::function_ptr`/`entrypoint_ptr`
> resolve the JIT pointer; the ECMAScript path routes i64→JSON args to the sidecar via
> `call_json`. `ModuleManifest` keeps `archetype` as the `"ecmascript"` selector. A
> `RuntimeBackend` trait could replace the enum if more backends appear, but the enum is
> the idiomatic single-owner form for the two known backends.
- **Dependencies:** `workerd` (Apache-2.0) packaged as a pinned Nix derivation behind
  `--features ecmascript`. For dev, the runtime is **auto-discovered** at load time
  (`WorkerdPaths::from_env`): `$UNFER_WORKERD` → `workerd` on `$PATH` (global npm install,
  fnm shim, Nix profile) → fnm-managed Node installations. Capnp import dir is
  `$UNFER_WORKERD_IMPORT` when set, else derived from the npm package layout. Validated
  against a global npm workerd (2026-08-04) with no env vars required.

### F2. Capability-based security + Gatekeepers (deferred approval + local simulation)  ★★
This is Cloudflare's most distinctive idea and the strongest fit, because unfer already has
the auth engine and an event stream.
- **Map:** a "Gatekeeper" = a module with `[grants]` that mediates a side-effecting op; the
  `ApprovalQueue` = a new `effects` grant namespace + an approval lane on the event stream.
- unfer already has `uk_subscribe`/`uk_poll` (`unfer_ffi/src/lib.rs:437,453`). Add:
  - `submit_action` semantics with `state ∈ {staged, pending, approved, rejected}` per call-
    site (mirror `gatekeeper-github/.../github.ts:138`).
  - **Local simulation:** side-effecting `uk_*` calls return *provisional* results immediately
    and queue a pending ActionRecord (`uk_*` → `unfer_protocol` `ActionRecord`); reads merge
    the provisional items back (mirror `github.ts:839` provisional merge). This is the
    "agent keeps working, human approves later" behavior.
  - A `uk_action_apply` / `uk_action_reject` / `uk_action_revert` trio (mirror
    `gatekeeper.ts:702-732`), invoked later by an operator/agent.
- **Choke point:** route all side-effecting `uk_*` through the existing `auth::check`
  (no new security surface). Default-deny.
- **Deliverables:** `unfer_protocol` `ActionRecord` + `UK-4xxx`-style codes, an
  `effects` grant namespace, provisional-result merge in the JS/Austral binding layer, a
  demo module pair (a "gatekeeper" module + a "client" module), docs.
- **Status: S4 implemented** (2026-08). Deliverables landed:
  - `unfer_protocol`: `ActionRecord` + `ActionState` (`staged|pending|approved|rejected|
    reverted`), `KernelEvent::ActionPending/ActionResolved`, codes `UK-4002`..`UK-4005`
    (`ACTION_REQUIRES_APPROVAL`/`ACTION_REJECTED`/`ACTION_NOT_FOUND`/
    `ACTION_ALREADY_RESOLVED`), and `merged_result()` — reads report the provisional
    result while pending, the applied result once approved.
  - `unfer_ffi`: `uk_action_submit` (queues Pending + returns handle + `action_pending`
    event), `uk_action_apply/reject/revert` (resolve + `action_resolved` event),
    `uk_action_get` (merged result), `uk_action_list` (queue scan with numeric handles).
    The approval lane is **opt-in**: subscriptions must explicitly request
    `action_pending`/`action_resolved`; an all-types subscription stays on the session lane.
  - `effects` grant namespace: `ManifestAuthEngine` authorizes action `"Effect"` against
    `[grants] effects` (defense in depth) and the ecma loopback gates `uk_action_submit`
    by the module's per-sidecar effects snapshot (host-side, race-free), injecting the
    module identity as the record `principal` (audit tag, F6). `swap` rejects effect-grant
    escalation.
  - Demo pair committed under `unfer/` (`client_module/` + `gatekeeper_module/`), plus
    integration tests `ecmascript_effects_deferred_approval_flow` (submit → provisional →
    gatekeeper approve → merged applied result) and `ecmascript_effects_deny_when_not_granted`
    (loopback UK-4001), and FFI unit tests for the full lifecycle + event lane.

### F3. Per-instance isolation ("Facets")  ★
- **Map:** Cloudflare gives each gadget its own DO Facet. unfer gives each module its own
  `Session` handle (`uk_model_create`). Process isolation comes from one `workerd` sidecar
  per module instance (crash/runaway-contained and restartable).
- **Deliverables:** a `gadget`/`facets` concept in `ModuleHost` — one workerd sidecar + one
  `Session` handle per module *instance*, with `uk_snapshot`/`uk_restore` for durable
  suspension/resume (already implemented at `unfer_ffi/src/lib.rs:412,425`).
- **Status: S5 implemented** (2026-08). `ModuleHost::instantiate(dir, instance_id)` spawns a
  dedicated `workerd` sidecar per instance key `"{name}@{id}"` — private staging dir
  (`.unfer-ecma-<id>`) + unix sockets + separate process; `call_json_instance` routes calls to
  that instance; `drop_instance` kills it. `snapshot_session`/`restore_instance` bind the
  durable `SessionBlob` round-trip (F3 suspension/resume). Gate test
  `modulehost_instantiate_isolates_instances` proves distinct staging dirs + distinct PIDs for
  two instances of the same module.

### F4. Blueprints (shareable, instantiable app templates)  ★
- **Map:** Cloudflare blueprints = gzip Yjs snapshot of gadget files + metadata in
  KV/R2 (`blueprint-archive.ts`). unfer modules already package `module.toml` + artifact.
- **Deliverables:** a `.cell`/`.gadget` archive format (magic + version + metadata + gzip
  snapshot, mirroring `blueprint-archive.ts:2-5`), `module.toml`-driven instantiation, and a
  `uk_restore`-based `initialize_from_blueprint` path. Store in the existing `unfer_data`
  content plane (X25519+AES-GCM, magnet URIs) rather than KV/R2.
- **Status: S5 implemented** (2026-08).
  - `.cell` archive format lives in `unfer_protocol::archive` (the shared contract every layer
    needs): magic `"UNFERCL1"` + version + JSON `CellMetadata` header + gzip body of
    `{files:[[relpath, hex], ...], session: hex|null}`. `CellBuilder`/`Cell` round-trip
    losslessly; parse rejects bad magic / unsupported version / truncation / corrupt gzip /
    absolute paths.
  - `uk_blueprint_export` packages a session snapshot into a `.cell`; `uk_blueprint_instantiate`
    restores it (UK-4100 invalid archive, UK-4101 no session). `ModuleHost::instantiate_from_blueprint`
    materializes the archived files (rejecting `..` traversal, requiring `module.toml`),
    spawns a fresh per-instance sidecar, and restores the packaged session — the
    `initialize_from_blueprint` path.
  - Storage: `unfer_data::blueprint` stores a cell through the existing content plane — chunked
    (SHA-256 per chunk), content-addressed, magnet URI, AES-GCM at rest — via
    `store_cell`/`verify_cell` + encrypt/decrypt helpers mirroring `DataPublisher`.
  - Gate tests: `modulehost_blueprint_roundtrip_restores_session` (E2E through workerd: evolve →
    snapshot → package → instantiate → restored session reproduces the probability and the
    `SessionBlob` JSON byte-identically), `modulehost_blueprint_rejects_path_traversal`,
    `modulehost_blueprint_requires_module_toml`, plus FFI `uk_blueprint_export/instantiate`
    round-trip + negative tests and protocol/data-plane unit tests.

### F5. Capability-minting chokepoint (default-deny)  ✅ (implemented 2026-08)
- **Map:** Cloudflare mints capabilities once at `user.ts:getGatekeeperClassFor()` and never
  from gadget/agent code. unfer already has this in `auth::check` + `ManifestAuthEngine`.
- **Deliverable:** harden the JS/Austral host bindings so a module can only ever *see/import*
  the symbols in its own `[grants]` — the capability object, not the full table. (This is the
  Cloudflare "loopback" pattern; the Australian `UnferKernel.aui` already exposes only the
  granted subset in spirit.)
- **Status: implemented** (2026-08) — see §S9. The generated `harness.mjs` `makeKernel(env)`
  no longer back each un-granted `uk_*`/`uz_*` name with a UK-4001 throw-stub (which made the
  full symbol table enumerable from module code). It now builds a plain object whose own
  properties are exactly the granted service bindings in `env` (i.e. `[grants] kernel`), so
  un-granted names are simply `undefined`. UK-4001 remains the loopback's answer (defense in
  depth), verified by `ecmascript_loopback_denies_ungranted`; the F5 property is gated by
  `ecmascript_capability_exposes_only_granted_symbols`.

### F6. Agents as managed, human-accountable entities  ★
- **Map:** Cloudflare agents are capability-restricted and billing/authority fall to the
  initiating human. unfer already has `unfer_agent` (NDJSON) + `unfer_edge` (Pingora).
- **Deliverables:** a `GatekeeperCaller`-style audit tag (`{from: agent|gadget|hook, chatId}`)
  on every `uk_*` call and every `ActionRecord`, so the human remains accountable. Add an
  `AgentSpawner` analogous capability that spawns sub-agents bounded to a fixed grant set.
- **Status: S6 implemented** (2026-08).
  - **Audit tags** (`unfer_protocol::CallerTag` `{from, principal, chat_id}`): minted once at
    the loopback chokepoint (`dispatch_loopback_as` sets the thread-local caller before every
    dispatch — a worker cannot forge another identity). Every dispatched `uk_*` call appends an
    immutable `AuditEntry {seq, caller, symbol, ok, detail, args}` to the kernel trail
    (`uk_audit_append` is host-internal), and `uk_action_submit` tags its `ActionRecord` with
    the same caller — so `uk_action_get`/`uk_action_list` read the full
    `GatekeeperCaller` tag, not just the principal.
  - **Audit listing** (`uk_audit_list` newest-first / `uk_audit_clear`, both grant-gated C
    symbols with loopback arms): a gatekeeper module holding `uk_audit_list` can review the
    trail; `uk_audit_clear` is operator-only. `unfer_edge` exposes `GET/DELETE /audit`
    (opt-in `--features audit`, embedded-kernel console).
  - **AgentSpawner** (`uk_agent_spawn`/`uk_agent_list`/`uk_agent_kill`/`uk_agent_grants`):
    spawns sub-agents **bounded to a fixed grant set** minted once at the chokepoint.
    `uk_agent_spawn` refuses escalation (UK-4202 `AGENT_GRANT_ESCALATION` — the requested
    grants must be a subset of the caller's own bounded set), and the loopback enforces the
    recorded set on every call attributed to that agent (default-deny): `dispatch_loopback_as`
    fetches the agent's bounds via `uk_agent_grants` and gates the symbol/effect against them.
    A stopped/unknown agent is denied outright. Action submissions by an agent carry the
    agent id as both the record principal and the `caller` tag.
  - Gate tests: `uk_audit_*` FFI round-trip, `uk_agent_*` spawn/bound/kill/escalation,
    `ActionRecord.caller`, and the loopback E2E (`loopback_audits_module_calls_with_caller_tag`,
    `loopback_audits_action_submit_with_agent_caller`,
    `loopback_agent_grant_enforcement_bounded_set`, `loopback_agent_unknown_handle_denies`),
    plus the `unfer_edge` `/audit` payload test.

### F7. Cap'n Web RPC (promise pipelining, capability returns)  ◐ (optional)
- **Map:** unfer's agent protocol is NDJSON over stdio (sequential). Cap'n Web adds
  pipelining + passing capability stubs as values — useful for `unfer_edge` HTTP clients.
- **Deliverable (optional):** a `capnweb`-style RPC layer behind `unfer_edge`; low priority.

### F8. Observers / information-flow leak prevention  ✅ (implemented 2026-08)
- **Map:** Cloudflare re-verifies every collaborator against every observation to prevent
  gadget→collaborator leaks. This is defense-in-depth; map to the `ActionRecord` audit log.
- **Deliverable:** per-collaborator re-check on shared module reads. **Status: implemented**
  (2026-08) — see §S8.
- **Mechanism:** every loopback dispatch now installs the caller's *full* grant set
  (`kernel` + `effects` + `observers`) on the thread-local caller context
  (`set_loopback_caller`). `uk_action_list`/`uk_action_get`/`uk_audit_list` re-check the
  active caller against each record/entry: the trusted harness (no bounded grant) sees all;
  a bounded caller sees only its own principal and any principal listed in its
  `[grants] observers`. An un-observable `uk_action_get` is indistinguishable from a missing
  record (UK-4004, no existence oracle). The loopback also now enforces capability
  non-escalation on `uk_agent_spawn` (UK-4202), including observer rights
  (`GrantSet::is_subset_of` covers `observers`).

---

## 4. Suggested implementation stages

Mirroring the repo's existing "S1..S7 / Stage N" convention (S1–S5 core, S3 the sandbox
layer, S6–S7 audit & packaging). Each stage is independently shippable and test-gated.

- **S1 — ECMAScript skeleton (workerd sidecar).** `archetype = "ecmascript"` selector;
  `ModuleHost::load`/`call` branches that materialize a `config.capnp` and spawn a `workerd
  serve` sidecar per module; a grant-checked capability binding exposing `uk_*`;
  `ecmascript_module` with positive + UK-4001 negative test. *Gate: `run_demo.sh`, sidecar
  smoke test.*
- **S2 — Capability bindings + auth chokepoint.** Map module `[grants]` → workerd capability
  bindings (default-deny; un-granted symbols absent); keep `auth::check` authoritative
  host-side; `globalOutbound: null` equivalent (no outbound sockets by default). *Gate:
  positive/deny test across net/fs/kernel grants.*
- **S3 — OS sandbox layer (browser-equivalent).** Wrap each sidecar in a dedicated launcher
  (`cranelift/src/sandbox.rs`) composing user namespaces (uid/gid-mapped) + empty netns +
  `no_new_privs` + seccomp-bpf (Chrome-renderer-style deny-list) + Landlock (read/exec on
  engine/system dirs; writes confined to staging + granted `[grants] fs` paths). The sidecar
  communicates over **unix sockets** (main + kernel loopback) so the empty netns leaves no
  ambient reachable endpoints. Cgroups were probed but are not wired (no writable cgroup in
  the dev environment). *Gate: sandbox escape-attempt tests (ptrace/mount/fileread denied) +
  a positive evolve test inside the sandbox.*
- **S4 — Deferred approval + local simulation.** `ActionRecord` + `effects` grant namespace in
  `unfer_protocol`; provisional-result merge in the host binding; `uk_action_apply/reject`
  trio; demo gatekeeper/client module pair. *Gate: protocol tests + a positive/deny test.*
  **Status: implemented** (2026-08) — see §F2.
- **S5 — Instance isolation + blueprints.** One `config.capnp`/sidecar per module instance +
  one `Session` handle; `.cell` archive format; `initialize_from_blueprint`; store in
  `unfer_data`. *Gate: snapshot/restore round-trip + blueprint instantiate test.*
  **Status: implemented** (2026-08) — see §F3/F4.
- **S6 — Agent accountability + audit.** `GatekeeperCaller` tags on `uk_*`/`ActionRecord`;
  `AgentSpawner` capability; expose audit via `unfer_edge`. *Gate: audit-listing test.*
  **Status: implemented** (2026-08) — see §F6.
- **S7 — Packaging + optional VM.** Package `workerd` + the OS sandbox + the blueprint store
  into the content-addressed Nix flake (`unfer_nixvm`) so the ECMAScript path is reproducible.
  The default path stays **sidecar + OS sandbox, without a full VM** per §2.1; the
  cloud-hypervisor VM remains an opt-in extra isolation layer for deployments that want it.
  *Gate: `nix build` + smoke test inside the sandboxed sidecar.*
  **Status: implemented** (2026-08):
  - `flake.nix` gains `packages.x86_64-linux.unfer-workerd` (statically-linked workerd +
    `workerd.capnp` from pinned npm tarballs — the meta tarball's node shim is overwritten by
    the real binary at the same path, reproducing the npm layout `ecma.rs` expects),
    `unfer-data` (blueprint store rlib) and the existing `unfer-ffi`.
  - `unfer_nixvm/flake.nix` gains `packages.sandboxed-sidecar` (bundles workerd + unfer-data +
    unfer-ffi + the smoke script; store paths content-addressed and virtiofs-shareable with the
    VM guest) and `apps.sandboxed-sidecar-smoke`.
  - `unfer_nixvm/run_sandboxed_sidecar.sh` is the smoke gate: it builds the packaged workerd
    via Nix, then runs the `ecmascript_module` integration tests (S3 sandbox confinement +
    S1 lifecycle round-trip) against it. All 10 integration tests pass with no skips.
  - `nix build .#unfer-workerd .#unfer-data` verified green; smoke passes (workerd
    `1.20260808.1`, sandboxed sidecar in its own user namespace + no_new_privs + seccomp).
- **S8 — Observers / information-flow leak prevention.** New `[grants] observers` namespace in
  `GrantSet`; every loopback dispatch installs the caller's full grant set on the thread-local
  caller context; `uk_action_list`/`uk_action_get`/`uk_audit_list` re-check the caller against
  each record/entry (trusted harness sees all; a bounded caller sees only its own principal +
  declared observers; un-observable reads look like missing records, UK-4004). Loopback
  `uk_agent_spawn` now enforces capability non-escalation incl. observer rights (UK-4202).
  *Gate: loopback observer/escalation unit tests + the `ecmascript_observers_filter_action_reads`
  E2E test (12/12 integration tests, 0 skips) + FFI observer tests.*
  **Status: implemented** (2026-08) — see §F8.
- **S9 — Capability-minting chokepoint (F5).** Generate the ECMAScript capability object
  strictly from the granted service bindings (`makeKernel(env)` in `harness.mjs`), so a module
  sees exactly its `[grants] kernel` — un-granted `uk_*`/`uz_*` names are absent, not stubbed,
  and not enumerable; UK-4001 is loopback-only (defense in depth). *Gate:
  `ecmascript_capability_exposes_only_granted_symbols` (Object.keys(kernel) == grants) + the
  existing `ecmascript_loopback_denies_ungranted` layer-2 test.*
  **Status: implemented** (2026-08) — see §F5.

---

## 5. Files that change (primary)

- `australVM/safestos/cranelift/Cargo.toml` — no new Rust runtime dep for the workerd path; add a
  `workerd` sidecar supervisor crate (spawn/manage `workerd serve`) behind `--features ecmascript`.
- `australVM/safestos/cranelift/src/module.rs` — archetype dispatch in `load`/`call`.
- `australVM/safestos/cranelift/src/ecma.rs` (new) — `config.capnp` generation, sidecar lifecycle,
  capability-binding wiring.
- `australVM/safestos/cranelift/src/lib.rs` — feature wiring; reuse `UNFER_SYMBOLS`.
- `australVM/safestos/cranelift/src/bin/modhost.rs` — `host` subcommand for ECMAScript modules.
- `unfer/unfer_protocol/src/types.rs` — `ActionRecord`, approval types.
- `unfer/unfer_protocol/src/codes.rs` + `all()` — action/approval codes.
- `unfer/unfer_ffi/src/lib.rs` — `uk_action_apply/reject/revert` (or module-side impl).
- `unfer/unfer_data/src/` — blueprint (.cell) archive + content store.
- `unfer/tools/module_builder` + `run_demo.sh` — ECMAScript build/QA wiring.
- `unfer/unfer_nixvm/flake.nix` — package `workerd` + the OS sandbox (bubblewrap/systemd) as
  derivations (S7).
- `australVM/safestos/cranelift/src/sandbox.rs` (new) — user-namespace/seccomp/Landlock launcher.
- `unfer/docs/MODULE_RECIPE.md`, `MODULES.md`, `PROTOCOL.md`, `ARCHITECTURE.md` — docs.
- **New:** `NOTICE` (Apache-2.0 attribution for adapted cloudflare-os concepts → place in
  `unfer/` and `australVM/`).

---

## 6. Risks & open questions

1. **Threat model = browser-equivalent, not VM-equivalent.** workerd's README states it is not
   a hardened sandbox and recommends a VM for untrusted code. We run it as a **sidecar without
   a full VM** but add the OS containment Chromium uses for renderers (user namespaces,
   `no_new_privs`, seccomp-bpf, Landlock, cgroups). This targets the **web-browser threat
   model**: a compromised engine yields a confined, low-privilege process, not the host. It is
   *not* the "Malicious Principal / hardware attacks" tier a VM provides (e.g. Spectre-class
   side channels, kernel-zero-day escapes). The residual gap vs. a VM should be documented in
   the module-hosting docs and `NOTICE`.
2. **workerd is a server, not a library.** It is driven by a Cap'n Proto `config` file and
   `serve` subprocess — not a Rust API. The supervisor must manage process lifecycle, socket
   plumbing, and restart. This is the main new machinery vs. an in-process embed.
3. **`uk_*` call surface from JS.** The C ABI is `i64`-based with probe-then-copy buffers.
   The workerd capability binding should translate JS objects ↔ `uk_*` buffer protocol (JSON
   strings), reusing the same `Diagnostic`/`uk_last_error` flow.
4. **Build weight.** workerd needs Bazel, clang 19+, libc++/LLD. Package as a pinned Nix
   derivation (S7) so it is reproducible and not rebuilt by hand.
5. **OS-level sandbox deps.** user namespaces / seccomp / Landlock / cgroups require kernel
   support and, for `bubblewrap`, a setuid helper or unprivileged user-namespace availability.
   On systems where user namespaces are restricted, an alternative (e.g. systemd `PrivateUsers`
   hardening) must be selected. Verify at S3.
6. **`velysterm` checkout.** The `kernel_client`/`mathed*` crates live on the
   `gitbutler/workspace` branch (already re-checked-out). F5/F6 SDK work depends on them.
7. **Scope discipline.** Follow cloudflare-os's own guidance: keep the "kernel" (auth +
   module host) diffs small; put policy/UI/UX in separate crates. Don't port the whole
   React/online-office surface — unfer's niche is the QFT probability kernel, not a slides app.

---

## 7. Recommended first task

Implement **S1** (ECMAScript skeleton): add `--features ecmascript`, the workerd sidecar
supervisor (spawn `workerd serve` from a generated `config.capnp`), the `archetype`
dispatch in `ModuleHost`, a grant-checked capability binding for `uk_*`, and a positive + UK-4001
`ecmascript_module` test. Then add **S3** (the OS sandbox layer) early, since the
browser-equivalent threat model depends on it. This delivers the requested capability with
minimal risk and gives a meaningful runway for S2–S7.

---

# Part II — Improvement roadmap (S10–S17)

All features in Part II are **planned, not implemented** (status `◐`). They extend the
completed F1–F6/F8 secure-kernel work toward production-grade operation and math-core
scale-out, grounded in a code audit done 2026-08.

**Priority order** (pick next in this order):
1. **Security**: S11 loopback peer lockdown (F10), S10 egress boundary (F9).
2. **Reliability**: S12 sidecar supervision (F11), S14 resource caps (F13).
3. **Observability**: S13 call tracing + metrics (F12).
4. **Proofs**: S16 fuzz + property tests (F16).
5. **Content plane**: S15 key-lifecycle + GC + cell GET (F14).
Optional (unchanged): **F7** Cap'n Web RPC — re-evaluate only after S10/S11 land.

---

### F9. Egress boundary — enforce `[grants] net`  ◐ (planned) → **S10**

**Gap (verified).** `net_grants`/`fs_grants` are parsed in `cranelift/src/module.rs` and
checked only for `swap` escalation. `SandboxProfile.writable_dirs` maps fs grants into
Landlock targets, but the **net grants have no runtime surface**: the sidecar runs in an
empty netns — safe by default, yet a module with `net = ["api.x"]` can neither fetch nor be
audited against an egress policy.

**S10 steps:**
- Add a `"fetch"` arm to `kernel_dispatch` that validates the target host against
  `GrantSet.net` (exact-host allowlist; default deny — no egress without a granted host).
- Emit an `AuditEntry { symbol: "fetch", action: allow|deny, host }` on every egress.
- Mirror the fs-grant wiring in `config_source`: generate workerd `external-service`
  bindings for the allowlisted hosts (prior art: how fs grants reach `SandboxProfile`);
  keep the loopback host check as defense-in-depth even if workerd is bypassed.
- **Gate (offline)**: a module with `net = ["127.0.0.1:PORT"]` fetches a local fixture via
  the loopback and succeeds; any other host is denied (UK-style egress code) and audited.
- Verify: `unfer` workspace + cranelift lib / full suites stay green.

---

### F10. Loopback peer lockdown  ◐ (planned) → **S11**

**Problem.** The sidecar ↔ host kernel-loopback Unix socket carries no `SO_PEERCRED`
validation; any process that can open the socket path can impersonate the sidecar (or the
host). This is the primary lateral-movement vector.

**S11 steps:**
- On the accept side of both the module loopback and the gateway call loop, verify
  `SO_PEERCRED` pid→tgid equals the spawned child (`EcmaSidecar.child`); reject otherwise
  with a `uk_security` audit event.
- Keep randomized socket names + `chmod 0700` in the staging dir (partly present).
- Gate: unit test opening the loopback from a synthetic fd (not the true child pid) asserts
  the connection is refused and an audit/seclog event is emitted.

---

### F11. Sidecar supervision & auto-restart  ◐ (planned) → **S12**

**Gap (verified).** `EcmaSidecar` spawns once + `wait_ready`; a workerd crash leaves the
module permanently dead (no restart path).

**S12 steps:**
- Supervisor thread: on `Child::try_wait`, respawn the workerd with the same generated
  `config.capnp` after a short backoff (1s→8s), reusing the staging dir so socket addresses
  stay stable.
- Do not surface the crash to the JS module: mark the module *degraded* (`KERNEL_DOWN`
  audit event) and serve the next call only after auto-heal.
- Gate: integration test kills the workerd mid-call and asserts the next wrapped call
  succeeds after auto-restart.

---

### F12. Observability: call tracing + metrics  ◐ (planned) → **S13**

**Gap (verified).** The only sinks are `AuditEntry` (audit.rs is the only typed log) and
plain log lines; no tracing, no metrics endpoint (`unfer_edge` only short-circuits `/audit`).

**S13 steps:**
- Per-symbol call/cpu-µs/error counters exposed as JSON and Prometheus text at the edge
  `GET /metrics` (alongside the `/audit` short-circuit in `unfer_edge/src/main.rs`).
- Optional `tracing` spans carrying `trace_id` + `CallerTag` so hangs in
  `evolve`/`reconstruct` are attributable to one caller.
- Gate: `GET /metrics` returns counters; a fuzzed request through `/audit` does not leak
  trace ids across callers.

---

### F13. Resource caps — cgroup + per-call deadline  ◐ (planned) → **S14**

**Gap (verified).** `SandboxProfile.memory_max_bytes` exists but the `ecma.rs` call site sets
`None`; no memory/swap cap and no per-call time limit on the FFI loop.

**S14 steps:**
- Wire a new `[limits] memory_bytes` (optional swap) field from the module manifest into
  `SandboxProfile`, applying the cgroup when a writable v2 cgroup is available; degrade
  gracefully (no cap) otherwise.
- Add a per-`uk_*` call deadline (default 5 s); on timeout record an audit event and
  terminate the sidecar.
- Gate: unit test wires the memory cap into a generated profile; a runaway busy-loop in the
  FFI hits the deadline and the audit shows the kill.

---

### F14. Content key-lifecycle + GC + cell GET  ✅ (implemented 2026-08) → **S15**

**Status: S15 implemented** (2026-08). `unfer_data/src/store.rs` adds `CellStore`
(ref-counted registry: store=own, `pin`/`unpin`, `prune` drops zero-pin cells with
`CellEvent::Pruned`) and `KeyRing` (deterministic epoch-chained keys; `rotate` moves the
writing epoch, old ciphertexts stay readable within `retain_depth` and are refused past it)
with `CellEnvelope`; `unfer_edge/src/cells.rs` adds the `GET /cell/<cid>` read route backed
by the process `CellStore` (present→metadata, well-formed absent→404, malformed→400) using
the new `unfer_data::blueprint::is_content_cid` shape gate. Remaining: an S20 seed point for
blueprint publication to flow into the edge store.

**Gap (verified).** `unfer_data` is content-addressed (`store_cell`/`verify_cell`/`CellRef`,
encrypt/decrypt, magnet) but has **no GC, no refcount, no key rotation** and no read path
exported over the edge.

**S15 steps:**
- Add `uk_cell_pin`/`uk_cell_unpin` refcounts and a prune pass that deletes unpinned cells
  with content-addressed delete + `cell_pruned` audit events.
- Add key rotation for cell encryption (new epoch key; reads of older epochs fall through)
  with a `key_rotated` event.
- Edge `/cell/<cid>` read route through de/encrypt + `verify_cell`, granting `unfer-edge`
  a scoped read.
- Gate: store → pin → unpin → prune → rotate — exactly the pinned cells survive; a
  corrupted CID fails verification on the GET path.

---

### F15. Math-core scale-out (auto-`m`/batch SIRK)  ◐ (planned) → **S16**

**Gap (verified).** SIRK is GPU-dense (candle), `m` is user-fixed, and shift/restart choices
are manual. AGENTS.md rank-saturation data for single-mode (~6) implies `m` has a knowable
ceiling.

**S16 steps:**
- Auto-`m` estimate from the Gram spectrum plus a budgeted batch of shifts per restart.
- Add a `fock_sirk/benches` weak-scaling harness sweeping `m ∈ {3,6,9}`; optional `f16`
  paths where the Gram stays Hermitian.
- Deliverable: bench numbers + a rule for when auto-`m` beats the hand-set value; no
  numerical-stability regression in the Gram-whitening tests.

---

### F16. Verification: fuzz + property tests  ◐ (planned) → **S17**

Add `proptest` invariants the audit and the `unfer_ffi` boundary rely on:
1. Anti-escalation: `GrantSet::is_subset_of` is transitive and covers `net`/`fs`/effects/
   observers across the `swap` path.
2. Monotonic audit: each `AuditEntry` seq strictly increments even across a session; force
   an explicit `audit_seq` counter if absent.
3. `uk_ffi` buffer protocol: arbitrary `Vec<u8>` request lengths → never panic, always a
   JSON decode error, no dangling probe.
4. `Cell` store→verify round-trip (`data → CID → data`).
Gate: ~10k arbitrary cases, zero panics, all invariants hold.

---

## 8. Change-tracking for Part II

Each planned feature is marked `. Status: S#x implemented (2026-08)` when it lands, in the
style of Part I, and the `AGENTS.md` capability/bound-maintenance section is kept in sync.

**Recommended execution order:** S10 → S11 → S12 → S13 → S14 → S15 → S16 → S17 (F7 remains
optional; re-evaluate only after F9/S10 lands).

---

# Part III — Cloudflare OS adaptation study (F17–F23 / S18–S24)

## What was studied

[`cloudflare/cloudflare-os`](https://github.com/cloudflare/cloudflare-os) (announced and
open-sourced 2026-08, Apache-2.0) is Cloudflare's internal "AI productivity OS" rewritten as
v2. Its kernel analogy maps directly onto unfer's existing architecture, so its *designs* are
adaptable as clean-room concepts (re-implemented in Rust / TOML / our protocol; no code is
ported — if any vendor artifact is ever copied it must carry an Apache-2.0 NOTICE).

| Cloudflare OS concept | cloudflare meaning | unfer analog today | Adopt? |
|---|---|---|---|
| **kernel** | `packages/workshop-backend`: connects users→programs/devices, sandboxes, ACLs | `unfer_ffi` + `ModuleHost` + GrantSet | already constructed |
| **gadgets** | private per-user sandboxed app instances | `prob_kernel::Session` (per-caller snapshot) | already analogous |
| **Gatekeepers** | per-service capability modules: narrow scope, logs every action, async human approval with **simulated outcomes** | none — external access isn't modelled | **ADD** as new module archetype (F18) |
| **intro/request access** | capability-based introductions (nothing ambient; agent may request) | static `[grants]`; nobody may *request* a grant | **ADD** `uk_request_resource` (F17) |
| **Blueprints** | sharing code (not data) → every user runs own copy, immutable blueprintId | `unfer_data` `store_cell`/`blueprint.rs` is a content store, no module-level reuse | **ADD** blueprint export/import/instantiate (F19) |
| **trust boundary / readOnlyHint** | a tool declared `readOnlyHint` runs as observation; everything else is queued; `vetted` endpoints auto-apply only when console-minted | our `[grants] effects` and audit catch all mutations ✓ | **ADD** observation/mutation annotation (F20) |
| **admin config vs env auth** | soft product config in `/admin`; auth & grants only from env so a compromised console can't escalate | only `DELETE /audit` exists on the edge | **ADD** admin console + hard-grant separation (F21) |
| **observability** | owner logger; `createObservabilityContext`; no-op `reportIssue`; never log secrets | audit + log lines only (S13 metrics planned) | **extend S13** (F22) |
| **release protocol** | byte-identical build → content-addressed upload → `candidate/` → single all-or-nothing `promote`; golden-file manifest test | `unfer_data/publisher.rs` resolves CIDs; no release manifold | **ADD** release manifest + promote (F23) |
| **Gatekeepers as Drivers** | drivers ~ services; kernel connects programs to devices | modules grant→service mapping | natural fit for F18 |

Two more adopted *principles* (not code):
1. **A resource becomes ambient only by user/admin config — never self-asserted.** Respect
   `GrantSet.auto-ness`: no effect path may treat a grant as ambient unless the harness placed
   it. This already matches our single-mint chokepoint in F5/S9.
2. **The account/capability is the authority, not an asserted identity** — matches
   `CallerContext`/`principals`: grants carry authority; assertions never confer it.

## Non-goals (recorded, deliberately skipped)

OAuth identity providers, real-time multiplayer/Yjs, gadget office apps, Durable-Objects
facets, MCP-as-protocol (we borrow the *trust* model, not the transport; an optional
`gatekeeper-mcp` adapter is deferred), and the React Workshop UI.

---

### F17. Resource grants + introductions  ◐ (planned) → **S18**

**What is adapted.** cloudflare-os grants nothing ambiently: you *introduce* a resource to a
session, an agent may *request* an introduction, and minting happens at a single kernel
chokepoint. Extends Part I F4/F5 greatly since it adds a third grant axis inside tests.

**S18 steps:**
- Add `GrantSet.resources: Vec<String>` (the `[grants] resources` namespace; ids such as
  `github.repo#denoission`), included in `is_subset_of` + swap-checks + observers read rules.
- Add `uk_resource_introduce(principal, resource_id)` and `uk_resource_forfeit(principal,
  resource_id)`; an unbounded agent may call `uk_request_resource(resource_id)` which lands a
  queued `AuditEntry(action="approval_pending")` for the human to grant/deny at the console.
- Single-mint chokepoint: a new `ResourceCtx` is minted in the same place where `CallerContext`
  is built (loopback listener), never re-created by module code.
- Gate: unit tests `introduce_grants_resource`, `request_queues_for_approval`, and an FFI-level
  test that a non-introduced resource call yields `UK-4401 RESOURCE_UNINTRODUCED`.

### F18. Gatekeeper module archetype   ◐ (planned) → **S19**

**Gap and adapt.** Our modules have no way to represent a mediated external service at all
(netless). Adopt the gatekeeper as an *archetype*: a module that (a) narrow access only to an
introduced resource, (b) faces the human approval queue for side-effecting calls, and (c) —
uniquely for us — returns a **forecast outcome** computed by `uk_condition` on a session
snapshot, so the agent can keep working while the human approves later. "Simulate locally then
approve async": in a probability kernel, the simulation *is* the conditional prediction.

**S19 steps:**
- Add **archetype `gatekeeper`** to the `archetype` dispatch table: side-effecting exports
  return "pending approval + simulated outcome" instead of erroring (`uk_gate_*`).
- Add `uk_gate_list_pending` / `uk_gate_approve` / `uk_gate_reject` (console-side), each
  written to the audit trail with the human `principal`.
- Provision modes replica of cloudflare's `disabled|optional|enabled` (per `[gatekeepers]`):
  `enabled` auto-mints for all, `optional` requires introduction, `disabled` denies all.
- Gate: unit test a demo **"scan" gatekeeper module** whose pending action is approved through
  `uk_gate_approve` and only then lands in the audit/side-effect stream.

### F19. Blueprint templates & per-user instantiation  ◐ (planned) → **S20**

**Gap and adapt.** `unfer_data` stores content-addressed cells; cloudflare's *blueprint* is
an immutable executable + sidecar. We add module-grade reuse on the same store.

**S20 steps:**
- Add `BlueprintRecord { blueprint_id, name, cell_cid, manifest_json, immutable_blueprint_id
  (never re-editable), created_by }` to `unfer_data`; export/import via `uk_blueprint_export`
  and `uk_blueprint_import` wrapping `store_cell`/`verify_cell`.
- `uk_blueprint_export_gadget` is just `instantiate(imported)`: every consumer runs its own
  copy (a fresh `Session`). `blueprint_id` equal to the content CID (immutable).
- **Open the store to the edge**: `GET /cell/<cid>` resolves through `verify_cell` + decrypt +
  `magnet` (S15), normalizing the content gateway.
- Gate: `blueprint->export->import->instantiate` identical behavior for two sessions; an
  altered cell fails `verify_cell` on import.

### F20. Trust annotations: observation vs mutation + vetted  ◐ (planned) → **S21**

**Gap and adapt.** In cloudflare-os a server-side `readOnlyHint` makes a tool run as
*observation* (no approval), everything else queues for approval, and any auto-apply must be
a **vetted** endpoint minted by the *console* — a module can never self-declare vetted status.

**S21 steps:**
- Extend `[effects grants]` metadata with `effect_kind: "observe" | "mutate"` (parsed from
  module.toml). Observe-Kind effects do not queue; mutate-Kind always do (F8 couples to
  `uk_gate_*`).
- A `vetted` flag derives only from the console/harness invitation (not from module.toml
  claims); add `uk_registry_vetted` (console-only).
- Gate: a mutate-kind effect without a pending approval is **refused**; an un-vetted console
  clearing the flag leaves the approval queue intact.

### F21. Admin console + soft/hard separation  ◐ (planned) → **S22**

**Gap and adapt.** `unfer_edge` short-circuits only `/audit`. Adopt the AdminConfig split:
soft product settings (site name, announcements, offered connectors/resources, blueprint
catalog) are console-editable; **auth, grants, and storage config are never user-editable**
(host-global env / Rust config only).

**S22 steps:**
- Edge `GET /admin/status` + `PATCH /admin/config` return/patch a `soft_config.json`
  (banner, announcements, resource availability modes) mirrored in one KV-style key.
- Admin capability is minted once (`is_admin`) at session start like `AdminApi`; the console
  never grants new pairs.
- Gate: `PATCH /admin/config` returns `401/403` for a non-admin principal; `soft_config.json`
  cannot change grants/auth.

### F22. Observability follows through  ◐ (planned) → **S23**

Converges S13 (metrics) with cloudflare's logging norms:
- dot-separated owner logger (`component = "kernel.audit"`), per-call `observability-context`
  (AsyncLocal analog: fields threaded per FFI call), and a `report_issue` that is a no-op if
  the `ERROR_REPORT_BINDING` is unset.
- **Discipline**: never log secrets/prompts/headers/keys — enforced by a test that scans audit
  and logs for a known secret token fed through a fixture.

### F23. Content-addressed release protocol  ◐ (planned) → **S24**

**Gap and adapt.** Cloudflare pins one, byte-identical, content-addressed release manifest and
promotes `candidate → release` in a single all-or-nothing copy.

**S24 steps:**
- Add `release_manifest.json` generation in `unfer_data` (publisher): map every deployable
  crate/module byte-to-CID; `promote(candidate, release)` is a single store op (content-
  addressed table) — one manifest copy; and a golden-file test for the expected manifest.
- `unfer/` CI check: `cargo fmt --check` + the release golden test; the CVS matrix (S7)
  unpinned from `workerd` 1.20260808.1 hashes against the manifest.
- Gate: golden-file regeneration honored via `UPDATE_GOLDEN=1`; a wrong byte in a module
  changes the manifest and fails the CI gate.

---

## 9. Change-tracking for Part III

Same convention as Part II: mark each section `. Status: S#x implemented (2026-08)` when it
lands. Notify the AGENTS.md "capability/bound-maintenance" checklist (`QfmConfig` default-
call-sites, compile_channels per-mode weights inline) to stay in sync.

**Recommended order:** S18 → S19 → S20 → S21 → S22 → S23 → S24 (F7 optional; egress/host work
in Part II guards ordered first: S10/11/15/17).
