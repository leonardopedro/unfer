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
    "uk_last_error"
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
