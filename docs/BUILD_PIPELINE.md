# Build Pipeline for Unfer Kernel Modules

The build pipeline automates the translation of Austral source code into a deployable SafeSTOS cell, incorporating the `module.toml` manifest.

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
