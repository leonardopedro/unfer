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
│   ├── qfm/                    # QFM engine: PG/diffusion Hamiltonians, observables
│   ├── qfm_text/               # text-domain QFM: corpus, LM, Oxieml decoder
│   ├── unfer_ffi/              # handle-based C ABI: 21 uk_* + 5 uz_* symbols
│   ├── unfer_edge/             # Pingora-based HTTP edge server (unfer_agent protocol)
│   ├── logos/                  # CNL→verified execution graph compiler (CCG+deltanet)
│   ├── ode_sirk/               # ODE→Hamiltonian singularity detection (Nelson ESA)
│   ├── unfer_consensus/        # QuePaxa consensus engine (LocalConsensus, ConsensusNode)
│   │                           #   + CertificateLedger (UTXO/carbon-certs, sparse-Merkle) — Plan R
│   │                           #   + MathBondLedger (SPV with nanoda trigger, UK-7401..7407)
│   │                           #   + MarketLedger (vAMM + NegRisk probability market, UK-7411..7419)
│   ├── unfer_identity/         # DID lifecycle manager (Ed25519, W3C DID Documents)
│   ├── unfer_data/             # encrypted chunked data plane (X25519+AES-GCM, magnet URIs)
│   ├── demo_module/            # example Austral module (Stage 13)
│   ├── bayes_update_module/    # Bayesian-update Austral module
│   ├── iterated_bayes_module/  # Iterated Bayesian update module
│   ├── qfm_module/             # QFM evolution Austral module
│   ├── qfm_tomo_module/        # QFM tomography Austral module
│   ├── zenodo_store_module/    # Zenodo-backed store Austral module
│   ├── unfer_nixvm/            # Nix flake: unfer_ffi in cloud-hypervisor guest
│   └── docs/                   # MODULE_RECIPE, PROTOCOL, ARCHITECTURE, LOGOS,
│                               # ODE_SIRK, FEDERATION, DATA_PLANE, BUILD_PIPELINE
├── australVM/                  # MODULE RUNTIME
│   └── safestos/cranelift/     # JIT + auth.rs + uk_* symbols + modhost
├── velysterm/                  # UI / AI INTERFACE
│   ├── crates/kernel_client/  # worker-thread client + unfer_agent bin (20+ ops)
│   ├── crates/mathed_core/     # Loro doc model + PropKinds + SemanticIndex
│   ├── crates/mathed/          # Bevy + Typst + vello editor + kernel_sys bridge
│   └── crates/mathed_mini/     # Bevy-free CPU frontend (winit + softbuffer)
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
                                      ↕           ↕
                                     qfm  ──  qfm_text
                                                     ↑
                           unfer_protocol ←──┬───────┘
                                             │
                         kernel_client ──────┤
                                ↑            │
                     mathed_core ←── mathed   │
                                              ↓
                                        unfer_edge

nested_fock_algebra ←── ode_sirk ──→ unfer_protocol
                        (Hamiltonian)   (HamiltonianSpec::OdeSystem)

unfer_protocol ←── unfer_consensus ←──┬── unfer_identity
                                      └── unfer_data

logos (standalone: CCG + deltanet + UNF hash; austral_codegen → australVM)
```

- `unfer_protocol` is the single shared contract (serde types, codes).
- `prob_kernel` wraps the QFT engine with Born-rule semantics.
- `qfm`/`qfm_text` build on `prob_kernel` for domain-specific pipeline stages.
- `unfer_ffi` exposes a C ABI for in-process module calls (21 `uk_*` symbols).
- `cranelift` (australVM) registers `uk_*` symbols in the JIT.
- `kernel_client` (velysterm) provides async worker + parser (20+ NDJSON ops).
- `mathed` bridges to Bevy via `kernel_sys.rs`.
- `unfer_edge` serves the agent protocol over HTTP via Pingora.
- `logos` compiles CNL sentences to verified execution graphs (CCG + deltanet).
- `ode_sirk` detects ODE singularities and constructs Weyl Hamiltonians.
- `unfer_consensus` provides the QuePaxa consensus engine and DID identity registry,
  plus the Plan R carbon-certificate / UTXO ledger (`CertificateLedger`:
  mint/transfer/burn with sparse-Merkle root, nullifier double-spend prevention).
- `unfer_identity` manages the DID lifecycle (create/update/revoke/resolve).
- `unfer_data` handles encrypted chunked content publishing (X25519 + AES-GCM).

## Data flow

1. **Modules** (Austral cells) call `uk_*` via the JIT — authorized
   per-module by manifest grants.
2. **velysterm** (Bevy UI) drives `prob_kernel::Session` directly via
   `kernel_client` (same code path, no FFI).
3. **AI agents** use the `unfer_agent` NDJSON binary.

All three paths converge on the same `Session` API:
`new → set_prior → evolve → probability → condition → snapshot → save/restore`.

A fourth path — **edge HTTP** — wraps the agent protocol behind
`unfer_edge` (Pingora), adding op-allowlisting, secret masking, and
rate limiting for remote clients.

## Extension points

### 1. Add a module

See `MODULE_RECIPE.md` for the full checklist. Summary:
1. Create `$ROOT/<name>/` with `module.toml` (21 `uk_*` grantable symbols).
2. Write `.aui`/`.aum` Austral cells importing `UnferKernel`.
3. List granted `uk_*` symbols in `module.toml [grants]`.
4. Build with `unfer_module_builder` (Stage A4); load via `modhost`.

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
2. Add dispatch in `prob_kernel/src/build.rs` `build_hamiltonian()`
   (`HamiltonianSpec::Builtin { name, params }` arm), reading params via
   `get_f64`/`get_u64`/`get_bool_or`/`get_u64_or`.
3. Add the name to the UK-1002 (`UnknownBuiltinModel`) valid-names hint in
   `prob_kernel/src/error.rs`.
4. Add a unit test in `nested_fock_algebra/src/unit_tests.rs` (term structure)
   and an integration test in `prob_kernel/tests/session.rs` (build + evolve +
   normalization). Document it in the "Builtin model set" list below.

   Note: velysterm no longer parses builtin names (the v1 `kernel_client/src/
   parse.rs` shortcut was deleted in P3.11); a builtin is reached either through
   the `unfer_agent` / `uk_model_create` API with a `ModelSpec` JSON, or from
   the editor via a user-defined translator that emits the builtin spec.

### 5. Add a math bond or market outcome

The math bond and market systems follow the same `ConsensusTransaction` variant
pattern as certificates and auctions:

1. Add a new `MathBondOpKind` or `MarketOpKind` variant in `unfer_protocol`.
2. Implement the `apply_op` arm in `unfer_consensus::mathbond` or `mathbond_market`.
3. Add an idempotency key in `idempotency.rs`.
4. Add a `ConsensusTransaction` match arm in `signing.rs` (canonical bytes +
   sign) and `node.rs` (dispatch).
5. Allocate a UK code (74xx for math bond, 741x for market).
6. Add a test: lifecycle + deterministic replay + error paths.

The trigger engine is `prob_kernel::verify::verify_export` — to add a new
mathematical theorem, create a `MathBondTrigger` with the theorem label,
permitted axioms, and max export size. The vAMM market prices the trigger
probability; the NegRisk adapter supports mutually-exclusive time-bucketed
outcomes.

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
  `U/2 · n_i(n_i-1)`, optional periodic boundary), `yang_mills_lattice`
  (`yang_mills_lattice(l, g, n_colors)`: a Kogut–Susskind-inspired Hamiltonian
  lattice gauge theory on a periodic `l × l` 2D lattice — electric energy
  `(g²/2) Σ n_ℓ` gaps the spectrum, and the *quartic* magnetic plaquette term
  `-(1/2g²) Σ_p Φ(ℓ1)Φ(ℓ2)Φ(ℓ3)Φ(ℓ4)` (Φ = a† + a) stress-tests the bounded
  direct-construction path; `l` clamped to ≥ 2, `n_colors` to ≥ 1),
  `qfm_mehler` (`qfm_hamiltonian(alphas)`: the analytical Quantum Flow Matching
  generator `H = |0><0| + Σ_j α_j n_j` from `QFM.tex` — `M = alphas.len()`
  orthogonal data points as single bosons in distinct modes plus the Mehler
  rank-1 vacuum projector `|0><0|`; built directly so M can be huge with no CAS
  blow-up. Driven end-to-end by the `qfm_module` Austral module).
- **Vacuum projector**: `Operator::ProjectVacuum` is the rank-1 `|0><0|`
  (self-adjoint, idempotent) backing the Mehler prior; on apply it keeps only
  the strict-vacuum component and drops everything carrying any mode.
