# Agent Guidelines: Fock-Sirk Project

Welcome, Agent. This repository contains high-performance tools for quantum field theory (QFT) simulations using Nested Fock Spaces and Rational Krylov methods.

## Technical Architecture

### 1. Hybrid CPU/GPU Pipeline
The project implements a split-mode architecture for **"Inverse-Free" Rational Krylov** (SIRK):
- **CPU (Symbolic & Sparse)**: The forward sequence $w_k = (H - z_k I) w_{k-1}$ is generated on the CPU. It uses `nested_fock_algebra`'s symbolic CAS and sparse `FxHashMap` structures to handle the exponential branching of state trajectories.
- **GPU (Dense Tensor)**: Basis states are flattened into a `StateDictionary` and offloaded to the GPU. The Gram matrix $G_{j,k} = \langle w_j | w_k \rangle$ and reduced Hamiltonian $H_{proj}$ are computed using `candle-core` CUDA kernels for maximum throughput.

### 2. Field Theory & CAS Primitives
- **Hermitian Field Representations**: Fields are mapped to $a^\dagger + a$ and momenta to $i(a^\dagger - a)$.
- **Quadratic Ordering**: To satisfy mass gap requirements and ensure $\langle 0 | H | 0 \rangle = 0$, the CAS compiler (`cas.rs`) MUST drop pure scalar terms during the distribution phase.
- **BRST Symmetry**: Physics Hamiltonians (Navier-Stokes, Yang-Mills) must commute with the BRST charge $\Omega$. Always verify gauge invariance when adding new terms.
- **Combinatorial Explosion Avoidance**: High-order non-linear models (e.g., Yang-Mills with quartic terms) MUST bypass the symbolic `Expression::expand()` engine and build `Hamiltonian` `Operator` structures directly. Expanding $O(10^4)$ polynomial terms causes infinite recursion hangs in the `distribute` logic.

### 3. LaTeX-to-Fock Pipeline
- **Parser**: Uses `mathhook` to parse raw LaTeX math into a symbolic AST.
- **Mapping Logic**: The `latex.rs` module translates standard physics notation (like $\psi^\dagger \psi$) into internal operator strings (`c_... * a_...`). 
- **Validation**: When adding new LaTeX support, ensure that daggers ($\dagger, \dag$) correctly trigger the creation operator mapping.

### 4. Numerical Stability
- **Inverse-Free SIRK**: We avoid $O(N^3)$ linear solves $(H-z)^{-1}$ by utilizing the forward sequence. 
- **Gram Whitening**: The projection $H_{proj} = W^\dagger H_{raw} W$ uses Hermitian eigendecomposition (Stage 2 replaced the bare Cholesky that panicked on degenerate Gram matrices). If singularity occurs, reduce $m$ or adjust shifts $z_k$.
- **Unitary Time Evolution**: **Always** use `nalgebra`'s Padé approximant `exp()` for the reduced system to preserve unitarity and Hermiticity.

### 5. GPU Optimization & Environment
- **Device Selection**: Use `Device::cuda_if_available(0)`. 
- **Library Path Note**: On systems with multiple CUDA versions, ensure `LD_LIBRARY_PATH` points to the toolkit matching the driver (e.g., `/lib/x86_64-linux-gnu` for CUDA 12.2 coexistence).
- **CUBLAS Safety**: Initialization failures (`ARCH_MISMATCH`) often indicate a version conflict between `libcublas` and the active GPU.

## Maintenance Checklist

- [ ] **Quadratic Ordering Check**: Verify that `compile_expression` continues to strip zero-point energy constants.
- [ ] **LaTeX Mapping Check**: Ensure `compile_latex` correctly interprets $a_i^\dagger$ as a creation operator and $a_i$ as annihilation. Note that the `mathhook` LALRPOP parser strictly requires explicit multiplication symbols (`*` or `\cdot`) instead of implicit spacing.
- [ ] **Commutator Validation**: Ensure non-commuting operators are never reordered by the symbolic engine (avoid `.simplify()` where order matters).
- [ ] **PG/Random Start**: Adding a new `HamiltonianType` variant or changing `pauli_grover_a`/`random_start` defaults must be reflected in `QfmConfig`'s `..Default::default()` call sites (`prob_kernel/src/build.rs`, `qfm_text/src/model.rs`).
- [ ] **GPU Execution**: Run examples with `RUST_LOG=candle_core=debug` to confirm active CUDA kernel dispatch.
- [ ] **Vacuum Initialization**: Ensure `QuantumState::vacuum()` is properly initialized with at least one empty inner universe (`OuterBosonCreate(InnerBosonicState::vacuum())`) before applying inner operators.
- [ ] **Trust annotations (S21)**: `EffectKind::{Observe, Mutate}` — `observe`-kind effects auto-apply, `mutate`-kind (and un-annotated) always queue for human approval; `uk_registry_vetted` is console-only (UK-4501 to non-hook callers) and never touches the approval lane. Adding an `effect_kinds` entry must keep `GrantSet::is_subset_of` denying Mutate→Observe relabeling.
- [ ] **Admin console (S22)**: `unfer_edge/src/admin.rs` mints the admin exactly once from `UNFER_ADMIN_PRINCIPAL` (default `operator`); hard keys (`grants`, `auth`, `storage`, `backend`) are never patchable — any new hard-config key must be added to the refuse list, and the soft config must stay byte-identical on refusal.
- [ ] **Observability hygiene (S23)**: `uk_audit_append` and `uk_report_issue` MUST run `sanitize_sensitive` (api_key/token/secret/…) before storing; `uk_report_issue` stays a no-op unless `ERROR_REPORT_BINDING` is provisioned; dot-separated owner-log lines carry `component = "kernel.audit"` and the ring is capped (`OWNER_LOG_CAPACITY` 512, drop-oldest). Any new secret-plausible field must be added to the sanitizer, or the secret-scan gate test fails.
- [ ] **Release golden gate (S23/S24)**: `unfer_data::release` manifest CIDs map every deployable artifact byte→sha256; the golden test regenerates only via `UPDATE_GOLDEN=1` — a wrong byte in a module changes the manifest and fails CI.
- [ ] **Budgets / rate limits (S25)**: metered `uk_*` symbols are denied at the loopback chokepoint with `UK-4601 RATE_LIMITED` / `UK-4602 BUDGET_EXCEEDED` + an audit entry — never a post-hoc report; a `GrantSet`/code change must keep the windowed meter (UTC-day key) as the single denial point and `uk_meter_status` read-only.
- [ ] **Sensitive forward latch (S26)**: a `sensitive: true` observation sticks to the caller set; once set, `fetch`/`agent_spawn`/`blueprint_export`/`action_submit`/`gate_approve` are refused `UK-4701 SENSITIVE_LATCHED` until an operator clears it (S22 admin seam). Any new forward-mutating symbol must be added to the latch's refuse list.
- [ ] **Credential vault (S27)**: secrets go through `uk_secret_put/get/revoke` (opaque, grant-gated, encrypted at rest via the S15 `KeyRing`); they must never serialize into a `SessionBlob` snapshot or a `.cell` blueprint — `uk_snapshot`/`uk_blueprint_export` refuse to package a live secret.
- [ ] **Capability RPC (S28)**: capabilities are minted only at the loopback chokepoint carrying the caller's grant set; a returned capability stub is re-checked against the original caller and revoked ids are refused. Keep NDJSON/std.io as a degenerate single-capability mode and the C ABI stable.
- [ ] **Lean4 proof verification (S29)**: `prob_kernel::verify::verify_export` type-checks a `lean4export` NDJSON payload with `nanoda_lib` and reduces the proofs to a boolean (`ProofReport.verified`) — the proof-irrelevance analogue of `logos::deltanet`'s unique-normal-form hash. A rejected proof is `verified: false` by default; `LeanVerifySpec::strict` turns it into a hard `UK-4801`. Malformed/oversize payloads are `UK-4802` (`ProofExportInvalid`). Any new `uk_*` symbol must be added to `EXPECTED_SYMBOLS.txt`, the generated C header, `australVM`'s `UNFER_SYMBOLS`, and the `GrantSet.kernel` namespace; any new `KernelEvent` variant must be covered by the `handles.rs` event-type matchers.
- [ ] **Cadabra2 symbolic coupling (S30)**: `prob_kernel::symbolic::symbolic_analyze` couples the existing LaTeX engine with the external field-theory CAS Cadabra2 (GPL-3.0) as a **subprocess** (`cadabra2-cli`, `CADABRA_CLI` override), so the Rust binary never links GPL code. The expression (TeX-subset or the `c_0 * a_0` CAS dialect) is canonicalized; `normalize_to_cas_dialect` translates Cadabra2's braced output back into the dialect `compile_to_fock` accepts, closing the loop into a numerical `Hamiltonian`. `verified` is the zero-detection verdict (`H - H† = 0` → Hermiticity). `symbolic_derive` runs a multi-cell `.cdb` derivation pipeline and extracts named expressions. Engine-missing → `UK-4901`, malformed expression → `UK-4902`. Engine-dependent tests **skip** when `cadabra2-cli` is absent; `flake.nix` provides it in the dev shell. QG: the 3D gauge-fixed Hamiltonian derivation lives in `docs/qg_gauge_fixed_hamiltonian.cdb` (repaired from `yangqg3.cnb`/`qg6.cnb` — the notebooks don't run as-is: they're missing `\partial{#}::PartialDerivative`, a flat index set `{a,b,c}`, `\sigma`, and have an extra-brace typo plus an unbalanced spin-connection substitution; the repaired script runs cleanly and is exercised by `symbolic::tests::qg_*`). The repaired pipeline reaches `ex1` = the Einstein-Hilbert action density `eR` in the vielbein and reduces it to the torsion (connection) form `e·X·X` (book.tex's `e_{abc}`). It does **not** reproduce book.tex's final `ℋ` in the S/P/T torsion variables: book.tex's `ℒ ≈ e(T_{ab}^b T^{ac}_c − ½T_{abc}T^{acb} − ¼T_{abc}T^{abc})` (the TEGR/teleparallel form) equals `eR` only up to the total-divergence term, and the notebook's cell 8 never completed the `T_{abc} = X_{abc} − X_{bac}` antisymmetrization + 3+1 split + Legendre transform (its own WIP).
- [ ] **Certificate ledger (Plan R)**: the UTXO/carbon-certificate state machine lives in `unfer_consensus::certs` (`CertificateLedger` + `SparseMerkle`). `CertificateOp`s (Mint/Transfer/Burn) are a `ConsensusTransaction` variant applied by `ConsensusNode::sync`. Minting is disabled unless a `MintAuthority` is configured. Any new op must preserve the invariants: conservation on transfer (UK-7002), no double-spend (UK-7004), owner-only spend (UK-7005), mint-authority check (UK-7001). Keep the ledger deterministic (same log → same root). See `docs/PLAN_REFI_EXCHANGE.md`.

## Crate Layout (Stages 1–28 complete)

- `nested_fock_algebra` — symbolic CAS + Fock-space algebra (improved: `adjoint()`, `prune`, `truncate_top_k`, bounded expansion).
- `fock_sirk` — SIRK solver (improved: GPU-optional, Gram whitening, BRST projection, restarted Krylov, state reconstruction).
- `unfer_protocol` — serde types, UK-#### codes, repair hints (the shared contract).
- `prob_kernel` — Born-rule layer: `Session` with `evolve`/`probability`/`condition`/`snapshot`.
- `unfer_ffi` — handle-based C ABI: 21 `uk_*` + 5 `uz_*` symbols (`uz_*` under `--features zenodo`).
- `qfm` — Pauli–Grover + diffusion Hamiltonians, `dense_pauli_grover_matvec`, parity/MNIST sweeps.
- `qfm_text` — text-domain QFM: corpus, features, LM, in-context adaptation, Oxieml decoder, GPU decode sketch.
- `unfer_edge` — Pingora-based edge server for the `unfer_agent` protocol over HTTP (`admin.rs` S22 soft/hard config console under `--features audit`; `gate.rs`/`blueprint.rs`/`cells.rs` edge routes).
- `demo_module/` — first module: `module.toml` + Austral cell + `run_demo.sh` (positive + UK-4001 negative test).
- `bayes_update_module/`, `iterated_bayes_module/`, `qfm_module/`, `qfm_tomo_module/`, `zenodo_store_module/` — 5 more Austral modules.
- `unfer_consensus` — certificate/UTXO ledger (`certs`), QuePaxa-style consensus engine, signing, multi-node convergence (Plan R).
- `unfer_taler` — GNU Taler exchange adapter over the cert ledger: reserves, two-phase wire gateway, denominations, e-coin withdraw/deposit/peg-out with the `fiat_in - fiat_out = reserves + merchants + outstanding` audit (UK-7101..7107, Plan R Phase 5).
- `unfer_nixvm/` — Nix flake packaging `unfer_ffi` inside the cloud-hypervisor VM guest (see `../../australVM/cloud_hypervisor_vm/`).
- `docs/` — `MODULE_RECIPE.md`, `PROTOCOL.md`, `ARCHITECTURE.md`, `BUILD_PIPELINE.md`.

Sibling repos:
- `australVM/safestos/cranelift` — JIT with `AuthorizationEngine` trait, `uk_*` symbol registration (feature `unfer-kernel`), `modhost` binary.
- `velysterm/crates/kernel_client` — async worker-thread client + `unfer_agent` NDJSON binary + parser.
- `velysterm/crates/mathed_core` — `PropKind::{Model, Prior, Event, Prob}` + `KernelStatement` in `SemanticIndex`; `glyphs` (Bevy-free glyph index); `accessibility` (toolkit-neutral a11y nodes).
- `velysterm/crates/mathed` — Bevy bridge (`kernel_sys.rs`), overlay rendering of prob results.
- `velysterm/crates/mathed_mini` — optional Bevy-free CPU frontend (winit + softbuffer, caret navigation, foot-style layout caching).

## Resolved Limitations

- **CUDA optional** (S1): all tests run CPU-only; `cuda` is additive.
- **Gram robustness** (S2): eigendecomposition whitening replaces bare Cholesky.
- **BRST projection** (S3): proper `project_physical` via CG, not subtraction hack.
- **Explosion bounds** (S4): `SirkOpts` + `compile_expression_bounded`.
- **Navier-Stokes test** (S5): re-enabled, runs the actual solver.
- **Restarted Krylov** (S6): `evolve_restarted` + `reconstruct` for long evolution.
- **Star topology degeneracy**: Single-mode-per-word star topology produces within-class degenerate W-rows because modes sharing the same label have identical Hamiltonian columns. **Distributed multi-mode encoding** (each word is a superposition of 2+ unique dedicated feature modes) breaks this degeneracy and enables 7/7 training accuracy.
- **Gram whitening vs full-rank orthogonalization**: Gram eigendecomposition whitening (with `rel_tol=1e-12` rank truncation) and full-rank orthogonalization (keep all positive eigenvalues) give identical results when no near-null eigenvalues exist. Raw non-orthogonal bases violate unitarity and give random 8/16 classification.
- **Asymmetric label distribution**: Balanced label counts (6e/6o) cause permutation-symmetric Krylov subspace → random 8/16. Asymmetric (5e/7o) breaks symmetry → 12/12 training at m≥3.
- **`compile_channels` API now has `per_mode_weights` parameter**: optional `Option<&HashMap<(u32,u32),f64>>` for per-transition amplitude weights. Pass `None` for uniform λ₁.
- **Krylov dimension m=2 insufficient**: regardless of λ₀ value, m=2 cannot distinguish 12 training inputs in the star topology parity test. Minimum m=3 required.
- **Lambda0 sweet spot**: λ₀=1.0 at m=3 gives 12/12 training; λ₀>1.5 degrades (projector dominates transitions).
- **Anti-learning at m=3**: The 3-dimensional Krylov subspace inverts the label structure at moderate λ₀ (0 < λ₀ ≤ ~5), giving 0% training accuracy. This disappears at λ₀=0 (random) and λ₀≥10 (projector dominates, 100% training even at m=3). The sweet spot m≥5 always works regardless of λ₀.
- **Rank saturation**: For single-mode-per-input parity at any scale (4-bit, 7-bit, 8-bit), the effective gram rank of the Krylov subspace caps at 6. The Krylov dimension m saturates in useful spectral directions at ~6, regardless of mode count (18 to 258) or training set size (12 to 200).
- **No generalization in single-mode star topology**: The Hermitian Hamiltonian with single-mode-per-input encoding achieves 100% training at m≥5 but gives 50% on held-out modes (extrapolation) and only ~54% on within-range held-out (interpolation). Each input is an independent mode with no shared structure — the uniform projector provides no mode-specific generalization.

## Core Dependencies
- `candle-core`: GPU tensor management (with `cuda` feature).
- `mathhook`: High-performance LaTeX and math parsing engine.
- `nalgebra`: High-level linear algebra for the reduced subspace.
- `quantrs2-symengine-pure`: Symbolic expression AST.

---
*Note: This project targets the Millennium Prize requirements for Yang-Mills and Navier-Stokes existence by resolving dynamics over discrete Fock-basis boundaries.*
