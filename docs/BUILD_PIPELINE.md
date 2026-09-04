# Build Pipeline for Unfer Kernel Modules

The build pipeline automates the translation of Austral source code into a deployable SafeSTOS cell, incorporating the `module.toml` manifest.

See also: [`tools/module_builder`](../tools/module_builder) (unified build+test runner),
[`tools/clean_qfm_text_runs.sh`](../tools/clean_qfm_text_runs.sh) (checkpoint cleanup).

## Components

### 1. `tools/module_builder` (bash script)

A unified build+test runner at `tools/module_builder`. Invoked by every module's
`run_demo.sh`. Steps:

- Parses `module.toml` for the module name, entry source, and grant symbols.
- Builds `unfer_ffi` (with `--features zenodo` for zenodo modules).
- Builds the cranelift JIT bridge (`austral_cranelift_bridge`) and `modhost`.
- Builds the Austral compiler via `dune build`.
- **Positive test**: compiles the module with `--use-cps-jit --target-type=tc`
  and verifies `CPS JIT: Execution result: <positive number>`.
- **UK-4001 negative test**: strips the deny symbol from `module.toml` and asserts
  `modhost authorize` returns non-zero (denial).

```bash
# Usage
tools/module_builder tc   <module_dir>            # type-check only
tools/module_builder run  <module_dir> [--deny <sym>]
```

### 1b. Running the module e2e locally (`tools/e2e_local.sh`)

The CI `module_builder` jobs (austral compile + CPS JIT + UK-4001 auth gate)
used to be a ~20-minute blind loop: every failure needed a full CI round-trip
before its output was visible. `tools/e2e_local.sh` runs the identical
pipeline on a dev machine in minutes (seconds once built):

```bash
tools/e2e_local.sh                    # default: qfm_tomo_module
tools/e2e_local.sh demo_module        # any module dir under the repo root
```

How each toolchain piece is resolved (mirrors CI exactly):

- **Rust side** (`unfer_ffi`, the cranelift bridge, `modhost`): the host
  rustup toolchain — the repo's `rust-toolchain.toml` pins 1.97.1, so a bare
  `cargo` invocation already uses the CI compiler. First run does the cold
  release builds; later runs are incremental via the cargo cache.
- **OCaml side** (the austral compiler): `module_builder` tries, in order,
  a bare `dune`, then `opam exec -- dune` (the CI setup-ocaml shape), then
  drops into the australVM nix flake (`nix develop`, which provides dune +
  OCaml libs) — a clear error names the requirement if all three are absent.
- **Iteration**: `MODULE_BUILDER_SKIP_BUILD=1` skips the cargo/dune builds
  entirely, so re-running against already-built artifacts is seconds. This
  is how a grant-gap regression like qfm_tomo's missing
  `uk_event_probability` surfaces locally instead of after a CI push.

The sibling layout requirement is the same as CI: the austral compiler build
and the `modhost` JIT expect `unfer/`, `australVM/`, and `dynamic-arctic/`
checked out as siblings under one root (the modules themselves live inside
`unfer/`).

### 2. Deployment Steps

#### A. Manifest Verification
The `AuthorizationEngine` in `safestos/cranelift/src/auth.rs` must be initialized with the `module.toml` content using `safestos_load_auth_manifest()`.

#### B. Symbol Resolution
The JIT bridge (`cranelift/src/lib.rs`) must have the `unfer-kernel` feature enabled to register the `uk_*` symbols (and `zenodo-store` for `uz_*`).

#### C. Cell Execution
The VM loads the cell. When a call to a `uk_*` function is encountered:
1. `cps.rs` identifies the symbol.
2. `auth::check()` is called to verify the manifest grant for that specific symbol.
3. If granted, the JIT jumps to the `unfer_ffi` implementation.
