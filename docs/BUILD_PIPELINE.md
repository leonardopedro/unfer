# Build Pipeline for Unfer Kernel Modules

The build pipeline automates the translation of Austral source code into a deployable SafeSTOS cell, incorporating the `module.toml` manifest.

## Components

### 1. `module_builder` (TBD)
A utility that:
- Parses `module.toml`.
- Orchestrates the Austral compiler (SafeSTOS compiler).
- Packages the output `.cell` file with its manifest.

### 2. Deployment Steps

#### A. Manifest Verification
The `AuthorizationEngine` in `safestos/cranelift/src/auth.rs` must be initialized with the `module.toml` content using `safestos_load_auth_manifest()`.

#### B. Symbol Resolution
The JIT bridge (`cranelift/src/lib.rs`) must have the `unfer-kernel` feature enabled to register the `uk_*` symbols.

#### C. Cell Execution
The VM loads the cell. When a call to a `uk_*` function is encountered:
1. `cps.rs` identifies the symbol.
2. `auth::check()` is called to verify the manifest grant for that specific symbol.
3. If granted, the JIT jumps to the `unfer_ffi` implementation.

## Example Pipeline Script (`deploy_module.sh`)

```bash
#!/bin/bash
# Usage: ./deploy_module.sh <module_dir>

MODULE_DIR=$1
MANIFEST="$MODULE_DIR/module.toml"
MAIN_FILE=$(grep "main_file" "$MANIFEST" | cut -d'"' -f2)

# 1. Compile to bytecode
# (Internal SafeSTOS compiler call)
/usr/local/bin/safestos-cc "$MODULE_DIR/$MAIN_FILE" -o "$MODULE_DIR/out.cell"

# 2. Load Manifest into VM
./safestos_vm --load-manifest "$MANIFEST" --run "$MODULE_DIR/out.cell"
```
