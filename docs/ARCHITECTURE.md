# unfer Architecture

> System diagram, dependency graph, and extension points for the
> modular probability kernel.

## System diagram

```
$ROOT/
├── unfer/                      # THE KERNEL
│   ├── nested_fock_algebra/    # symbolic Fock-space CAS + LaTeX→Hamiltonian
│   ├── fock_sirk/              # SIRK time-evolution solver (CPU/CUDA)
│   ├── unfer_protocol/         # serde types, UK-#### codes, repair hints
│   ├── prob_kernel/            # Born-rule layer: Session, condition()
│   ├── unfer_ffi/              # handle-based C ABI: uk_*()
│   └── docs/                   # MODULES.md, PROTOCOL.md, ARCHITECTURE.md
├── australVM/                  # MODULE RUNTIME
│   └── safestos/cranelift/     # JIT + auth.rs + uk_* symbols + modhost
├── velysterm/                  # UI / AI INTERFACE
│   ├── crates/kernel_client/  # worker-thread client + unfer_agent bin
│   ├── crates/mathed_core/     # Loro doc model + PropKinds + SemanticIndex
│   └── crates/mathed/          # Bevy + Typst + vello editor + kernel_sys bridge
└── demo_module/                # example Austral module (Stage 13)
```

## Sibling-folder convention

All three repos (`unfer`, `australVM`, `velysterm`) live as siblings
under `$ROOT`. Cross-repo dependencies use relative path deps:

```
cranelift  →  ../../../unfer/unfer_ffi       (feature: unfer-kernel)
kernel_client  →  ../../../unfer/{prob_kernel,unfer_protocol}
mathed  →  ../kernel_client  (within velysterm workspace)
```

Build scripts assert this layout.

## Cross-repo dependency graph

```
nested_fock_algebra ←── fock_sirk ←── prob_kernel ←── unfer_ffi
                                                    ↑
                          unfer_protocol ←──┬───────┘
                                           │
                          kernel_client ────┘
                               ↑
                    mathed_core ←── mathed
```

- `unfer_protocol` is the single shared contract (serde types, codes).
- `prob_kernel` wraps the QFT engine with Born-rule semantics.
- `unfer_ffi` exposes a C ABI for in-process module calls.
- `cranelift` (australVM) registers `uk_*` symbols in the JIT.
- `kernel_client` (velysterm) provides async worker + parser.
- `mathed` bridges to Bevy via `kernel_sys.rs`.

## Data flow

1. **Modules** (Austral cells) call `uk_*` via the JIT — authorized
   per-module by manifest grants.
2. **velysterm** (Bevy UI) drives `prob_kernel::Session` directly via
   `kernel_client` (same code path, no FFI).
3. **AI agents** use the `unfer_agent` NDJSON binary.

All three paths converge on the same `Session` API:
`new → set_prior → evolve → probability → condition → snapshot`.

## Extension points

### 1. Add a module

See `MODULES.md` for the full checklist. Summary:
1. Create `$ROOT/<name>/` with `module.toml`.
2. Write `.aui`/`.aum` Austral cells importing `UnferKernel`.
3. List granted `uk_*` symbols in `module.toml [grants]`.
4. Build with `build.sh`; load via `modhost`.

### 2. Add a kernel op

1. Add the op to `unfer_protocol/src/types.rs` if new params are needed.
2. Add a `Session` method in `prob_kernel/src/session.rs`.
3. Add a `uk_*` shim in `unfer_ffi/src/lib.rs`.
4. Register the symbol in `cranelift/src/lib.rs` (`cranelift_init`).
5. Add an agent op in `kernel_client/src/bin/unfer_agent.rs`.
6. Allocate a new UK code if the op can fail in a new way.
7. Update `PROTOCOL.md`.

### 3. Add a PropKind

1. Add the variant to `PropKind` in `mathed_core/src/markers.rs`.
2. Add the `of()` name mapping.
3. If kernel-bearing, add collection logic in `semantics.rs build_index`.
4. If kernel-bearing, add dispatch handling in `mathed/src/kernel_sys.rs`.
5. Add overlay rendering in `overlay.rs` if visual feedback is needed.

### 4. Add a builtin model

1. Implement in `nested_fock_algebra/src/models.rs`.
2. Add dispatch in `prob_kernel/src/build.rs` `build_hamiltonian()`.
3. Add parser support in `kernel_client/src/parse.rs` `parse_model()`.
4. Add the name to the `VALID_OPS`/error hints.

## Resolved limitations (Stages 1–6)

- **CUDA optional** (S1): `cuda` feature is additive; all tests run on CPU.
- **Gram whitening** (S2): replaced bare Cholesky with eigendecomposition;
  `Whitening { w, rank, dropped }` handles rank-deficient Gram matrices.
- **BRST projection** (S3): proper `P = I - Q†(QQ†)^{-1}Q` via CG;
  `Operator::adjoint()` and `Hamiltonian::adjoint()` added.
- **State explosion bounds** (S4): `SirkOpts { prune_eps, max_components }`
  + `compile_expression_bounded` prevents OOM.
- **Navier-Stokes test** (S5): re-enabled with real solver exercise.
- **Restarted Krylov** (S6): `evolve_restarted` + `reconstruct` for
  long-running evolution with norm conservation.

## Key design decisions

- **`SirkOpts::default()`**: `prune_eps: 1e-12, max_components: Some(50_000),
  brst_tol: 1e-10`.
- **Edition 2024**: unfer workspace + velysterm workspace use
  `#[unsafe(no_mangle)]` for FFI. `australVM/safestos/cranelift` is
  edition 2021 and uses `#[no_mangle]`.
- **CPU-first**: every acceptance criterion runs without CUDA; `cuda` is
  additive via `--features cuda`.
- **Quadratic ordering**: the CAS compiler drops pure scalar terms during
  distribution to satisfy mass-gap / vacuum-energy requirements.
- **High-order models**: `yang_mills_hamiltonian` builds `Hamiltonian`
  directly (bypassing `Expression::expand()`) to avoid combinatorial
  explosion.
- **Builtin model set**: `yang_mills`, `navier_stokes`, `gravity`,
  `harmonic_chain`, `bose_hubbard` (`bose_hubbard_chain(n_modes, t, u,
  periodic)`: nearest-neighbour hopping `-t(a_i† a_j + h.c.)` plus on-site
  `U/2 · n_i(n_i-1)`, optional periodic boundary).
