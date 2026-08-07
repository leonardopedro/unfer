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
  - Capability binding with grant-checked `uk_*` — generated `harness.mjs` builds a
    `makeKernel(env)` Proxy; only granted symbols get workerd `service` bindings to
    `kernel-loopback`; un-granted `uk_*`/`uz_*` throw UK-4001 (layer-1 stub) and the
    loopback re-checks `auth::check` host-side (defense in depth).
  - Positive/UK-4001 test module — `cranelift/tests/ecmascript_module.rs` (3 tests:
    positive lifecycle, UK-4001 un-granted symbol, loopback deny). Skips when no
    workerd runtime is discoverable.
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

### F5. Capability-minting chokepoint (default-deny)  ★
- **Map:** Cloudflare mints capabilities once at `user.ts:getGatekeeperClassFor()` and never
  from gadget/agent code. unfer already has this in `auth::check` + `ManifestAuthEngine`.
- **Deliverable:** harden the JS/Austral host bindings so a module can only ever *see/import*
  the symbols in its own `[grants]` — the capability object, not the full table. (This is the
  Cloudflare "loopback" pattern; the Australian `UnferKernel.aui` already exposes only the
  granted subset in spirit.)

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

### F8. Observers / information-flow leak prevention  ◐ (optional)
- **Map:** Cloudflare re-verifies every collaborator against every observation to prevent
  gadget→collaborator leaks. This is defense-in-depth; map to the `ActionRecord` audit log.
- **Deliverable (optional):** per-collaborator re-check on shared module reads.

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
supervisor (spawn `workerd serve` from a generated `config.capnp`), the `archetype` dispatch
in `ModuleHost`, a grant-checked capability binding for `uk_*`, and a positive + UK-4001
`ecmascript_module` test. Then add **S3** (the OS sandbox layer) early, since the
browser-equivalent threat model depends on it. This delivers the requested capability with
minimal risk and gives a concrete runway for S2–S7.