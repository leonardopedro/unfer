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
