# Writing an unfer Kernel Module

This is **the** recipe for building a module that drives the unfer probability
kernel through the safestos (australVM) JIT. A module is an Austral cell that
calls the kernel's `uk_*` C ABI in-process; every kernel call is authorized
per-module by a `module.toml` manifest.

The canonical worked example ships inside this repo at
[`demo_module/`](../demo_module/) — read it alongside this document. (It expects
the `australVM/` compiler checkout as a sibling of `unfer/`.)

## 1. Folder layout

Modules live in their own sibling checkout next to `unfer/` and `australVM/`:

```
$ROOT/
├── unfer/                 # the kernel (unfer_ffi exports uk_*)
├── australVM/             # the module runtime (safestos cranelift JIT)
└── <your_module>/
    ├── module.toml        # manifest: identity, archetypes, kernel grants
    ├── src/
    │   ├── <Name>.aui     # Austral interface
    │   └── <Name>.aum     # Austral body (imports UnferKernel)
    ├── build.sh           # type-check against the bindings; asserts layout
    └── run_demo.sh        # build + JIT-run + authorization test
```

The build scripts assert this layout and fail with a clear message if a sibling
is missing.

## 2. Manifest schema (`module.toml`)

```toml
[module]
name = "demo_module"          # principal used by the authorization engine
version = "0.1.0"
archetypes = ["actor"]        # contracts implemented (see §3)
entry = "src/DemoModule"      # interface/body stem (without extension)

[grants]
# Allow-list of uk_* symbols this module may call. The JIT denies (UK-4001)
# any kernel call to a symbol not listed here.
kernel = ["uk_version", "uk_model_create", "uk_evolve", "uk_event_probability",
          "uk_get_result", "uk_last_error", "uk_model_free"]
```

`ManifestAuthEngine::from_toml_str` (australVM `safestos/cranelift/src/auth.rs`)
parses this and answers `authorize(module, "Call", symbol)`.

## 3. Archetype contracts

A module implements one or more archetypes. Each is an exact Austral signature
the host can call (handles are `Int64`, buffers are `Address[Nat8]` + `Int64`
length, matching the kernel ABI in [PROTOCOL.md](PROTOCOL.md)):

| Archetype        | Signature                                              | Purpose                                  |
|------------------|--------------------------------------------------------|------------------------------------------|
| `prior_provider` | `function provide_prior(model: Int64): Int64`          | Set/return the model's prior state       |
| `data_source`    | `function update(model: Int64, payload: Address[Nat8], len: Int64): Int64` | Condition the model on new data |
| `actor`          | `function act(model: Int64): Int64`                    | Drive evolution / read probabilities     |

A module calls the kernel through the `UnferKernel` bindings
(`australVM/examples/kernel/UnferKernel.aui/.aum`), which wrap each `uk_*`
symbol as a typed Austral function (e.g. `kernelVersion(): Int64`).

## 4. Lifecycle

1. **Build** — `dune build` the Austral compiler; compile the module with
   `austral compile <UnferKernel> <YourModule> --use-cps-jit`. The CPS-JIT path
   lowers the cell to CPS IR and JIT-compiles it against the live `uk_*` symbols
   registered in `cranelift_init()`.
2. **Load + grant** — the host installs the manifest via
   `safestos_load_auth_manifest()` (or `modhost`); the engine now knows the
   module's grants.
3. **Run** — the JIT executes the entry export. Each `uk_*` call passes through
   `cps.rs::check_call_permission` → `auth::check`, enforcing the grants.
4. **Hot-swap / unload** — existing `__au_swap_module` / CellDescriptor machinery
   (unchanged) swaps a cell without restarting the host.

## 5. Checklist: add a new module

1. `mkdir -p $ROOT/<name>/src` next to `unfer/` and `australVM/`.
2. Write `src/<Name>.aui` (interface) and `src/<Name>.aum` (body); `import
   UnferKernel (...)` and `pragma Unsafe_Module;` for FFI.
3. Implement at least one archetype function from §3.
4. Write `module.toml`: set `name`, `archetypes`, `entry`, and the `[grants]`
   kernel allow-list — list **only** the `uk_*` symbols you actually call.
5. Copy `demo_module/build.sh` + `run_demo.sh`; adjust the module stem.
6. `bash build.sh` — type-checks against the bindings.
7. `bash run_demo.sh` — JIT-runs and verifies the authorization gate.
8. If you call a `uk_*` symbol not yet in the bindings, add it to
   `UnferKernel.aui/.aum` (mirror an existing `pragma Foreign_Import`).

## 6. Known limitation / extension point

The JIT's authorization principal is currently the **calling function** name
(e.g. `run`), while manifests grant by **module** name. The `modhost` binary and
`run_demo.sh` exercise the manifest decision directly against
`ManifestAuthEngine`; threading the module name into
`cps.rs::check_call_permission` so the live JIT enforces module-level grants
inline is a documented extension point (one-line principal change once the CPS
encodes the module name). See [ARCHITECTURE.md](ARCHITECTURE.md) extension
point #1.
