# Module Recipe Specification (unfer-kernel v1)

The `module.toml` file defines the metadata, requirements, and permissions for an Austral module intended to run on the SafeSTOS JIT with the Unfer Kernel.

## Structure

```toml
[module]
name = "my_kernel_module"
version = "0.1.0"
description = "Example module using the Unfer Probability Kernel"

[dependencies]
# External Austral modules required
# Format: name = "version"
UnferKernel = "1.0.0"

[grants]
# Permissions required from the SafeSTOS Authorization Engine.
# The keys are the 'resource' identifiers, and values are lists of allowed 'actions'.
# For the Unfer Kernel, the resource is typically "kernel".
kernel = [
    "uk_version",
    "uk_init",
    "uk_model_create",
    "uk_model_free",
    "uk_set_prior",
    "uk_set_hamiltonian",
    "uk_evolve",
    "uk_condition",
    "uk_event_probability",
    "uk_observe",
    "uk_get_result",
    "uk_last_error",
    "uk_snapshot",
    "uk_restore",
    "uk_subscribe",
    "uk_poll",
    "uk_bayesian_update",
    "uk_belief_propagation"
]

[build]
# Build pipeline configuration
main_file = "main.au"
output_name = "my_module.cell"
optimization_level = 3
```

## Deployment Pipeline

1. **Validation**: The `module.toml` is parsed to verify that all requested `grants` are valid `uk_*` symbols.
2. **Compilation**: The Austral source is compiled into a `.cell` (bytecode) format.
3. **Packaging**: The `.cell` and `module.toml` are bundled into a module package.
4. **Loading**: SafeSTOS reads the `module.toml`, checks the manifest grants against the `AuthorizationEngine`, and loads the bytecode into the JIT.
5. **Symbol Binding**: The JIT bridge resolves `uk_*` calls to the shared `unfer_ffi` library.

## Model handles: prefer the linear `Model` wrapper

`uk_model_create` returns a raw `Int64` handle that `uk_model_free` consumes —
correct freeing is caller discipline. `UnferKernel` (the Austral bindings) also
exposes a **linear** wrapper, `Model`, which makes freeing a type-enforced
obligation:

```austral
let m: Model := wrapModel(kernelModelCreate(spec, len));  -- own the handle
let h: Int64 := modelHandle(&m);                          -- borrow without consuming
...                                                       -- drive uk_* calls
let rc: Int64 := freeModel(m);                            -- consume exactly once
```

A `Model` that is dropped without `freeModel` is a compile-time **Linearity
Error** (session leak); freeing it twice is a use-after-consume error. New
modules should hold handles as `Model`. _(Backend note: the current CPS-JIT does
not yet lower record-destructure bindings or cross-module non-foreign calls, so
the wrapper is enforced at typecheck time but executed via the raw `Int64`
functions for now — see IMPLEMENTATION_PLAN gap §9.)_

## ECMAScript modules (`archetype = "ecmascript"`)

An ECMAScript module runs as a **workerd sidecar** (V8), served by the
`austral_cranelift_bridge` `ecma.rs` supervisor (S1). It is the JS equivalent of
the Austral CPS path: the same `uk_*` capability surface, the same `[grants]`
authorization, and the same positive/UK-4001 test gates.

### `module.toml`

```toml
[module]
name = "my_js_module"
version = "0.1.0"
description = "Example ECMAScript module"
archetypes = ["ecmascript"]      # select the workerd backend
archetype = "ecmascript"
entry = "src/main.js"            # the module's entry JS file (ES module)

[grants]
# Only these uk_* symbols become visible to the worker's `kernel` object.
# Anything else in the uk_*/uz_* namespace throws UK-4001 (CALL_DENIED).
kernel = [
    "uk_version",
    "uk_model_create",
    "uk_model_free",
    "uk_set_prior",
    "uk_evolve",
    "uk_event_probability",
    "uk_get_result",
    "uk_last_error",
]
```

### Worker contract (`src/main.js`)

The entry file is an **ES module**. The host calls a named export (defaulting to
`default[name]`) as `async (kernel, args) => result`; the default entrypoint is
`run`:

```js
export async function run(kernel, args) {
  const version = await kernel.uk_version();
  const model = await kernel.uk_model_create(args.spec);
  await kernel.uk_set_prior(model, { kind: "vacuum" });
  await kernel.uk_evolve(model, { t: 0.01 });
  await kernel.uk_event_probability(model, { kind: "vacuum" });
  const result = await kernel.uk_get_result(model);   // auto-parsed JSON object
  await kernel.uk_model_free(model);
  return { version, probability: result.probability };
}
```

- `kernel` is a **capability object** (F5): a Proxy containing only the granted
  symbols, each an `async (...args) => result` RPC to the host loopback. It never
  exposes the full `uk_*` table.
- Every `kernel.uk_*` call round-trips through `auth::check` host-side
  (defense in depth) and returns `data.result` — a parsed JSON value. Most
  `uk_*` return a JSON **string**; `uk_get_result`/`uk_last_error`/`uk_snapshot`
  auto-parse their buffered output into objects. Errors throw with `err.ukCode`
  set (`UK-4001` for un-granted symbols).
- `args` is the JSON object passed to the entrypoint (e.g. from
  `modhost host --args-json`). JSON embedded as a value arrives as an object —
  use it directly; do **not** re-`JSON.parse` objects.
- The return value is serialized and becomes the module-call result.

### Runtime discovery & installation

The workerd binary is auto-discovered at load time
(`WorkerdPaths::from_env`, mirroring the test skip logic):

1. `UNFER_WORKERD` — explicit binary path (error if set but missing).
2. `workerd` on `$PATH` (global npm install, fnm shim, Nix profile).
3. fnm-managed Node installations
   (`~/.local/share/fnm/node-versions/*/installation/lib/node_modules/workerd`).

The Cap'n Proto import dir is `UNFER_WORKERD_IMPORT` when set, else derived
from the npm package layout next to the resolved binary. For a quick local
setup: `npm install -g workerd`.

### Hosting & tests

- **Staging**: `load` writes `config.capnp`, `harness.mjs`, and a copy of the
  entry JS under `<module_dir>/.unfer-ecma/`, starts one `workerd serve` sidecar
  per module, and waits for the socket.
- **Invocation**: `ModuleHost::call_json(handle, entrypoint, args_json)` or
  `modhost host --args-json '<json>'`.
- **Tests**: `cranelift/tests/ecmascript_module.rs` — positive lifecycle,
  UK-4001 un-granted symbol, loopback deny, and (with `--features sandbox`) an
  OS-sandbox confinement test asserting the sidecar runs in its own user
  namespace with `no_new_privs` + seccomp. Skipped when no workerd runtime is
  discoverable (mirrors the "CUDA optional" convention).

### OS sandbox layer (`--features sandbox`, S3)

With the `sandbox` feature, each workerd sidecar is wrapped by a dedicated
launcher (`cranelift/src/sandbox.rs`) that composes Chromium-renderer-equivalent
OS containment:

- **User namespace** (`CLONE_NEWUSER`): the sidecar runs as an unprivileged,
  uid/gid-mapped root with no host capabilities.
- **Empty network namespace** (`CLONE_NEWNET`): the only reachable endpoints are
  the unix sockets the sidecar materializes in its staging dir (the kernel
  loopback and the main socket) — hence the unix-socket design.
- **IPC namespace** (`CLONE_NEWIPC`) + **`no_new_privs`**.
- **seccomp-bpf deny-list**: ptrace, mount, umount2, pivot_root, reboot,
  kexec_load, module load/unload, open_by_handle_at, setns, perf, bpf,
  userfaultfd, quotactl, swapon/off, ioperm/iopl, kcmp — all return `EPERM`.
- **Landlock**: read/exec on the engine binary, its dynamic deps, and the system
  dirs; writes confined to the staging dir + the module's granted `[grants] fs`
  paths. Nonexistent readable dirs (e.g. `/lib` on NixOS) are skipped rather
  than aborting the sandbox.

Confinement is verified at spawn: the sidecar's `/proc/<pid>` shows a distinct
user namespace, `NoNewPrivs: 1`, and `Seccomp: 1/2`. If the kernel lacks
unprivileged user namespaces (`sandbox::supported()` false), the sidecar falls
back to a plain spawn — browser-equivalent containment is best-effort.

### Deferred approval + local simulation (`effects` grants, S4)

Side-effecting ops are never executed inline. A module that wants to request an
effect holds the grant in a **separate namespace**, `[grants] effects`, distinct
from `[grants] kernel`:

```toml
[grants]
kernel = ["uk_action_submit", "uk_action_get"]  # harness exposes these symbols
effects = ["send_notification"]                  # loopback authorizes THIS effect
```

Two layers gate the call (F5, capability-minting at a single choke point):

1. **Harness (layer 1)** — only symbols in `[grants] kernel` become workerd
   bindings; un-granted `uk_*` throw UK-4001 in the capability object.
2. **Loopback effects gate (layer 2, authoritative)** — `dispatch_loopback`
   checks `uk_action_submit` against the module's `effects` snapshot instead of
   the kernel grants: the submitted *effect name* must be in `[grants] effects`.
   The record's `principal` is injected from the module identity (an audit tag —
   a worker cannot claim another module's identity).

The kernel surface (`unfer_ffi`, `unfer_protocol`):

- `uk_action_submit(req_json)` — queues a Pending `ActionRecord` and returns its
  handle immediately; the caller sees only the **provisional (simulated)**
  result until resolution (local simulation — the agent keeps working).
- `uk_action_apply / uk_action_reject / uk_action_revert(handle)` — the operator
  resolves the record (`pending → approved | rejected | reverted`); only an
  `Approved` record can be reverted.
- `uk_action_get(handle)` — reads a record with the **merged** result: the
  provisional result while pending, the applied result once approved.
- `uk_action_list()` — the whole queue, each record carrying its numeric
  `handle` so a gatekeeper can resolve it.
- Events: `action_pending` / `action_resolved` broadcast to subscriptions that
  **explicitly** opt into the approval lane (`{"types":["action_pending",...]}`);
  an all-types subscription stays on the session lane.

Lifecycle: `staged → pending → approved | rejected`, plus `approved → reverted`.
Codes: `UK-4002 ACTION_REQUIRES_APPROVAL`, `UK-4003 ACTION_REJECTED`,
`UK-4004 ACTION_NOT_FOUND`, `UK-4005 ACTION_ALREADY_RESOLVED`.

Demo pair (committed under `unfer/`): `client_module/` submits a
`send_notification` action; `gatekeeper_module/` lists pending actions and
approves/rejects them. The positive flow + deny-when-ungranted are integration
tests in `cranelift/tests/ecmascript_module.rs`
(`ecmascript_effects_deferred_approval_flow`,
`ecmascript_effects_deny_when_not_granted`).

### Instance isolation + blueprints (`.cell` archives, S5)

**Per-instance sidecars (F3).** `ModuleHost::instantiate(dir, instance_id)`
gives every instance of a module its own `workerd` sidecar — private staging
dir (`.unfer-ecma-<id>`), private unix sockets, separate OS process — keyed by
`"{module_name}@{instance_id}"`. Calls go through `call_json_instance(key, entry, args)`;
`snapshot_session(handle)` / `restore_instance(key, session_json)` give durable
suspension/resume for the instance's kernel `Session`.

**`.cell` blueprint archives (F4).** A shareable, instantiable snapshot of a
module = metadata + gzip of its files + an optional session snapshot:

```
UNFERCL1 │ version=1 │ CellMetadata (JSON) │ gzip{ files:[[relpath,hex],...],
                                                   session: hex|null }
```

- Format + parser: `unfer_protocol::archive` (`CellBuilder`/`Cell`,
  `ArchiveError`). Rejects bad magic, unsupported version, truncation, corrupt
  gzip, and absolute paths.
- Kernel: `uk_blueprint_export(model)` packages a session; `uk_blueprint_instantiate(cell)`
  restores it. Codes `UK-4100 BLUEPRINT_INVALID`, `UK-4101 BLUEPRINT_NO_SESSION`.
- Host: `ModuleHost::instantiate_from_blueprint(cell, parent_dir, id)`
  materializes the archived files into `parent_dir/{name}-{id}` (rejecting `..`
  traversal, requiring `module.toml`), spawns a fresh per-instance sidecar, and
  restores the packaged session. The `worker`'s loopback transport hex-encodes
  the binary cell (`uk_blueprint_export` returns `{cell_hex}`).
- Storage: `unfer_data::blueprint` stores a cell through the existing content
  plane (`store_cell`/`verify_cell`: chunked, content-addressed, magnet URI,
  AES-GCM at rest) — the blueprint registry.

The gate is an end-to-end round-trip
(`modulehost_blueprint_roundtrip_restores_session`): evolve → snapshot → package
`.cell` → instantiate → the restored session reproduces the original's
probability and its `SessionBlob` JSON byte-identically.

### Agent accountability + audit (GatekeeperCaller tags, S6)

**Every `uk_*` call is audited.** The loopback chokepoint tags the current
thread's caller identity before dispatching, and the kernel appends one
`AuditEntry` (`{seq, caller, symbol, ok, detail, args}`) per call — granted or
denied. The `caller` is a `CallerTag` `{from: agent|gadget|hook, principal,
chat_id}` minted by the host, so a worker cannot forge another identity's tag.
`ActionRecord`s carry the same tag (`record.caller`), so a gatekeeper reviewing
`uk_action_get`/`uk_action_list` sees exactly who submitted each side effect.

```toml
[grants]
kernel = ["uk_audit_list"]   # a gatekeeper module may review the trail
# uk_audit_clear is operator-only: never grant it to untrusted modules
```

- `uk_audit_list()` → `AuditEntry[]`, newest first; `uk_audit_clear()` →
  `{removed: N}`. Both are grant-gated kernel symbols with loopback arms.
- `unfer_edge --features audit` serves `GET /audit` (list) and `DELETE /audit`
  (clear) from the embedded kernel — the operator console.

**Observers (information-flow, F8).** A bounded caller may only *read* records
and audit entries for **its own principal** plus any principal declared in the
`[grants] observers` namespace — the trusted harness (operator/`unfer_edge`)
sees all. Without this, *any* module holding `uk_action_list`/`uk_audit_list`
could enumerate every other module's actions, params, and audit args — a
gadget→collaborator leak.

```toml
[grants]
kernel = ["uk_action_list", "uk_action_apply"]  # gatekeeper symbols
observers = ["ecma_client"]                      # whose records this module may read
```

- `uk_action_list`/`uk_audit_list` silently omit un-observable records; a
  `uk_action_get` on an un-observable handle is indistinguishable from a missing
  record (`UK-4004` — no existence oracle).
- A module always observes itself; sub-agents inherit the observer set recorded
  at spawn, and `uk_agent_spawn` refuses to mint observer rights the caller does
  not hold (`UK-4202`).

**`AgentSpawner` (bounded sub-agents).** `uk_agent_spawn` mints a sub-agent with
a **fixed grant set** at a single chokepoint:

```json
{"name":"analyst","grants":{"kernel":["uk_evolve"],"effects":["send_notification"]}}
```

- Escalation is impossible: the requested grants must be a subset of the
  caller's own bounded set, else `UK-4202 AGENT_GRANT_ESCALATION`.
- The host loopback enforces the recorded set on every call attributed to the
  agent (default-deny); a stopped or unknown agent is denied outright. Action
  submissions carry the agent id as both `principal` and `caller.principal`.
- `uk_agent_list()` / `uk_agent_kill(handle)` / `uk_agent_grants(handle)`
  complete the lifecycle. Codes: `UK-4200 AUDIT_INVALID`, `UK-4201
  AGENT_NOT_FOUND`, `UK-4202 AGENT_GRANT_ESCALATION`, `UK-4203
  AGENT_STATE_INVALID`.

The gate is the audit-listing test: `uk_audit_*` FFI round-trip + the loopback
E2E (`loopback_audits_module_calls_with_caller_tag`,
`loopback_agent_grant_enforcement_bounded_set`, …) — denied attempts are audited
too, because they are the most important entries in the human's review trail.
