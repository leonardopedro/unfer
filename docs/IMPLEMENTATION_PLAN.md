# Plan: unfer as a Modular Probability Kernel (australVM modules + velysterm UI)

> **Executor note:** This plan is written to be executed stage-by-stage by a smaller LLM. Each stage has a goal, exact files, key signatures, and acceptance commands. Do not skip acceptance steps. Do stages in order unless noted. All paths abbreviate `$ROOT = /media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba`.

## Current status (updated 2026-06-27, rev 5)

**All 18 stages (S1–S18), all hardening items (P0–P4), and Workstream E (QFM) are complete.** The unfer kernel is a modular probability kernel with an NDJSON agent interface, a C ABI for in-process module calls, an authorization-aware JIT hook, a Bevy-bridged UI, a Bevy-free mini frontend with AccessKit, and two verified end-to-end module demos (`demo_module` + `qfm_module`). Every per-crate acceptance test passes on CPU; the GPU path is smoke-tested. The work below is the historical spec + outcomes record; known gaps are in §"Known gaps & deferred items".

- **What now exists (was the greenfield baseline at commit `b1e5581 "working"` 2026-05-09):**
  - `unfer/` workspace: 5 crates (`nested_fock_algebra`, `fock_sirk`, `unfer_protocol`, `prob_kernel`, `unfer_ffi`) + 2 module demos (`demo_module/`, `qfm_module/`). CUDA is optional (`cuda` feature, CPU-default, GPU-smoke-tested).
  - `australVM/safestos/cranelift`: `auth.rs` (`AuthorizationEngine` trait + `ManifestAuthEngine`; Cedar demoted to optional default feature), `uk_*` symbols registered in the JIT behind `unfer-kernel` feature, `check_cedar_permissions` → `check_call_permission`. CPS-JIT backend fixed (let-init, record destructure, cross-module linking, byte buffers, multi-field records).
  - `velysterm`: `crates/kernel_client/` (worker-thread client + `unfer_agent` NDJSON binary), `mathed_core` (PropKinds + `KernelStatement` + `accessibility` + `glyphs`), `crates/mathed/` (Bevy bridge + overlay), `crates/mathed_mini/` (Bevy-free CPU frontend with caret blink, mouse hit-testing, AccessKit bridge, translator pipeline, kernel bridge).
- **Test counts (CPU, full sweep 2026-06-27):** unfer workspace **105** (13 fock_sirk + 24 nested_fock_algebra + 2 unfer_protocol + 21 prob_kernel + 15 unfer_ffi + 30 integration) · velysterm own crates: mathed_core **72** · mathed_mini **44** · kernel_client 4 · mathed 36 · australVM cranelift 9 (default features) + clean `--no-default-features` build. **CUDA smoke:** `cargo test -p fock_sirk --features cuda` = 14 tests green (+1 `gpu_smoke_hopping_energy_matches_cpu`). The `unfer_agent` NDJSON echo acceptance for S17 is verified. velysterm `cargo test --workspace --all-targets` compiles (P4 #16 resolved — stale `velyst` examples gated).
- **Git state (all three repos in sync with their remotes):**
  - unfer HEAD `6ce121a` → `origin/main` (in sync). **Uncommitted:** GPU smoke test (`fock_sirk/src/forward_sirk.rs`) + doc edits (`IMPLEMENTATION_PLAN.md`, `.gitignore`). Needs commit + push.
  - australVM HEAD `40df9c5c` → `origin/master` (in sync). `cps.rs.*` backups removed in `198cc137` (only `_build/` artifacts remain, not tracked). Clean.
  - velysterm HEAD `62062dc` → `origin/gitbutler/workspace` (in sync). **Uncommitted:** P4 #21 overlay GUI smoke (pixel-color test + screenshot) + P4 #22 mini-frontend polish (caret blink, mouse hit-testing, AccessKit bridge) — modified: `app.rs`, `kernel_bridge.rs`, `Cargo.toml`, `lib.rs`; new: `a11y.rs`, `overlay_smoke_screenshot.png`; plus rustfmt of `delta_algebra`/`delta_sirk`/`velyst`. All tests green (mathed_mini 44), clippy clean, headless build works. Needs commit + push.
- **Progress checklist:**
  - [x] S1 CUDA optional · [x] S2 Gram whitening · [x] S3 BRST projection · [x] S4 explosion bounds · [x] S5 Navier-Stokes test · [x] S6 restarted Krylov
  - [x] S7 `unfer_protocol` · [x] S8 `prob_kernel` · [x] S9 `unfer_ffi`
  - [x] S10 auth trait · [x] S11 JIT symbols · [x] S12 Austral bindings (typecheck + live CPS-JIT) · [x] S13 module recipe (`demo_module/` + `modhost` + `run_demo.sh`)
  - [x] S14 `kernel_client` · [x] S15 PropKinds · [x] S16 Bevy bridge · [x] S17 agent interface · [x] S18 docs/verify
  - [x] P0 demo spine · [x] P1 CI + overlay + GPU smoke · [x] P2 linear handle + dead-code cleanup + diagnostic audit · [x] P3 translator pipeline + kernel wiring + builtin models + benchmarks · [x] P4 prior/solver + CI fix + clippy + RepairHints + benchmarks + Yang-Mills lattice + overlay/GPU smoke + mini-frontend polish
  - [x] E19–E21 QFM module (Mehler prior + Hamiltonian + protocol + Austral module)

## Context

**Why:** `unfer` (current dir) is a Rust quantum field theory simulator: `nested_fock_algebra` (symbolic Fock-space engine, LaTeX→Hamiltonian via `compile_latex`) + `fock_sirk` (GPU Shift-Invert Rational Krylov time-evolution solver). The user wants it to become the **kernel of a modular system that computes probabilities of events**:

- **Probability semantics (decided):** Born-rule layer on the existing QFT core. Priors = initial `QuantumState` + `Hamiltonian`; data updates = projection/conditioning of the state; event probability = squared-amplitude mass of matching outer Fock states. The Fock/SIRK engine stays the substrate.
- **Module mechanism (decided):** `$ROOT/australVM`'s recipe — Austral modules (.aui/.aum) → linear typecheck → CPS binary IR → Cranelift JIT (`safestos/cranelift`) → C scheduler, with hot-swap via CellDescriptor. Modules live in sibling folders `$ROOT/<name>` and call the kernel **in-process** via native symbols registered in the JIT (same mechanism as `au_print_int`, see `cranelift_init()` at `safestos/cranelift/src/lib.rs:69-72`). Cedar is demoted from core to an optional authorization backend; the unfer kernel takes its architectural seat.
- **UI (decided):** `$ROOT/velysterm` (Bevy + Typst + Loro math editor, M1 done / M2 in progress) is the main human UI **and** AI-agent interface, in the spirit of Vercel Labs' Zero language: structured JSON, stable `UK-####` error codes, typed repair hints.
- **Core improvement priorities (decided):** fix known limitations (BRST projection hack, Cholesky failures, CAS combinatorial explosion, disabled Navier-Stokes test) and performance/scalability (CPU fallback, pruning/memory bounds, restarted Krylov). 
- More modules will be defined later → adding a module must be a documented, repeatable recipe (manifest + contract).

**Verified anchors** (load-bearing facts; line numbers re-confirmed against commit `b1e5581` on 2026-06-22):
- `unfer/Cargo.toml:19` — `candle-core = { version = "0.8.2", features = ["cuda"] }` (hard CUDA dep). → Stage 1.
- `fock_sirk/src/forward_sirk.rs` — `solve_forward_sirk(hamiltonian, v_0, shifts, device, brst_charge: Option<&Hamiltonian>) -> candle_core::Result<ForwardSirkResult>` (`:49-55`); `ForwardSirkResult { h_proj, g_matrix, registry, basis_tensors }` (`:8-13`, **no `w_sequence` field — it is a local var dropped after the solve**, Stage 6 must retain it). **The fragile spot is `:140`**: `g_sub.cholesky().expect("Gram matrix must be positive definite")` then `h_proj = L_inv * h_proj_raw * L_inv.adjoint()` (`:139-143`) → Stage 2 replaces this. Simplified BRST subtraction at `:67-78` → Stage 3. `time_evolve(t)` (`:18-25`) returns coefficients **in the Cholesky-orthonormalized basis** with no path back to a `QuantumState`; reconstruction = `w_sequence · (L_inv.adjoint() · coeffs)` today, `w_sequence · (W · coeffs)` after Stage 2 whitening → Stages 2 & 6 depend on this.
- `nested_fock_algebra/src/lib.rs` — `QuantumState` (`:52`) has `inner_product` + `scale_and_add` but **no `norm`/`prune`/`truncate_top_k`/`len`** (Stage 4 adds them); `Operator` (`:107`) and `Hamiltonian` (`:325`) have **no `adjoint()`** yet (Stage 3 adds them). `MatrixFreeOperator` trait (`fock_sirk/src/lib.rs:13`) provides a free `norm` helper but not as a `QuantumState` method.
- `nested_fock_algebra/src/cas.rs` — entry points are `compile_expression(expr: Expression) -> Hamiltonian` (`:17`, **by value**) and `compile_to_fock(input: &str)` (`:11`); `compile_latex(latex: &str) -> Hamiltonian` lives in `latex.rs:5`. Stage 4 wraps these with bounded variants.
- `nested_fock_algebra/src/models.rs` — builtins: `navier_stokes_hamiltonian(nu) -> Expression` (`:59`, **returns an Expression — needs `compile_expression`**), `navier_stokes_brst() -> Expression` (`:79`, the BRST charge for Stage 3's test), `yang_mills_hamiltonian(g) -> Hamiltonian` (`:144`, direct), `gravity_hamiltonian() -> Hamiltonian` (`:275`, direct). Stage 8 adds a `harmonic_chain` builtin here.
- `nested_fock_algebra/src/unit_tests.rs:148-154` — `test_navier_stokes_compiles` is **fully commented out (a vacuous pass)**, not merely disabled; Stage 5 must make it actually exercise the solver, not just uncomment.
- `australVM/safestos/cranelift/src/cps.rs:8-20` — `check_cedar_permission(caller, callee)`: whitelists `__`/`au_` prefixes and self-calls, otherwise asks Cedar `(caller, "Call", callee)`. So new `uk_*` kernel symbols flow through authorization automatically.
- `australVM/safestos/cranelift/src/lib.rs:53-88` — `cranelift_init()` registers native symbols via `builder.symbol("au_print_int", ...)`. Kernel functions register here.
- `velysterm/crates/mathed_core/src/markers.rs` — `PropKind` closed enum with `of(name)` mapper; adding kinds is additive. `semantics.rs` has `SemanticIndex`/`build_index`. 51 core tests.

## Architecture (target state)

```
$ROOT/
├── unfer/                      # THE KERNEL (this repo)
│   ├── nested_fock_algebra/    # existing symbolic engine (improved)
│   ├── fock_sirk/              # existing SIRK solver (improved, CPU-capable)
│   ├── unfer_protocol/   NEW   # serde types, UK-#### codes, repair hints — THE contract
│   ├── prob_kernel/      NEW   # Born-rule layer: Session, EventPredicate, condition()
│   ├── unfer_ffi/        NEW   # cdylib/staticlib/rlib, handle-based C ABI: uk_*()
│   ├── demo_module/      NEW   # first module: module.toml + Austral cell + run_demo.sh
│   └── docs/             NEW   # MODULES.md (recipe), MODULE_RECIPE.md (module.toml schema),
│                               #   PROTOCOL.md, ARCHITECTURE.md, BUILD_PIPELINE.md, IMPLEMENTATION_PLAN.md
├── australVM/                  # MODULE RUNTIME
│   └── safestos/cranelift/     # + auth.rs (AuthorizationEngine trait; Cedar optional
│                               #   feature), + uk_* symbols in JIT, + modhost bin
├── velysterm/                  # UI / AI INTERFACE
│   ├── crates/kernel_client/ NEW  # worker-thread client + parsers + unfer_agent bin
│   ├── crates/mathed_core/     # + PropKinds: Model, Prior, Event, Prob; + glyphs (Bevy-free); + accessibility
│   ├── crates/mathed/          # + kernel_sys.rs Bevy bridge, overlay results
│   └── crates/mathed_mini/ NEW   # Bevy-free CPU frontend (winit + softbuffer), caret navigation
└── (demo_module now lives inside unfer/)
```

Data flow: modules (Austral cells) and velysterm both drive `prob_kernel::Session`; modules via the `uk_*` C ABI inside the safestos JIT (calls authorized per-module by manifest grants), velysterm via direct Rust dependency (same Session code path). AI agents use the `unfer_agent` NDJSON binary.

Repos stay separate; sibling checkout layout is required and asserted by build scripts. Path deps: cranelift → `../../../unfer/unfer_ffi` (feature `unfer-kernel`), kernel_client → `../../../unfer/{prob_kernel,unfer_protocol}`.

---

## Workstream A — improve the unfer core (Stages 1–6)

### Stage 1: Make CUDA optional (CPU fallback) — DO FIRST
Nothing downstream is testable without CUDA-free builds.
- `unfer/Cargo.toml`: drop `features = ["cuda"]` from the workspace `candle-core` dep.
- `unfer/fock_sirk/Cargo.toml`: add `[features] default = []`, `cuda = ["candle-core/cuda"]`.
- New `unfer/fock_sirk/src/device.rs`: `pub fn best_device() -> Device` — `Device::cuda_if_available(0)` under `#[cfg(feature = "cuda")]`, else `Device::Cpu`. Export from `lib.rs`. Update all 7 `fock_sirk/examples/*.rs` to use it.
- **Accept:** `cargo test --workspace` passes without CUDA; `cargo build -p fock_sirk --features cuda` compiles (run only if nvcc present).

### Stage 2: Gram-matrix robustness (replace bare Cholesky)
- New `unfer/fock_sirk/src/linalg.rs`:
  - `pub enum SirkError` (thiserror): `GramDegenerate { max_eig: f64 }`, `StateExplosion { components: usize, limit: usize }`, `BrstNotConverged { residual: f64 }`, `Numeric(String)`.
  - `pub struct Whitening { pub w: DMatrix<Complex64>, pub rank: usize, pub dropped: usize }`
  - `pub fn whiten_gram(g: &DMatrix<Complex64>, rel_tol: f64) -> Result<Whitening, SirkError>` — Hermitian eigendecomposition; keep eigenpairs `λ_i > rel_tol·λ_max`; `W = U_r Λ_r^{-1/2}`; `GramDegenerate` only at rank 0.
- `forward_sirk.rs:139-143`: delete the `cholesky().expect(...)` + `L_inv` block (the `:140` panic site) and form `H̃ = Wᴴ H_proj_raw W` from `whiten_gram(g_sub, rel_tol)`; add `pub rank: usize` to `ForwardSirkResult`. **Store `W` too** (new field `pub w_whiten: DMatrix<Complex64>`) — `time_evolve` and Stage 6 reconstruction need it to map whitened-basis coefficients back to `w_sequence` coordinates (replacing the old `L_inv.adjoint()` mapping).
- **Tests:** duplicated Krylov vector → reduced rank, no panic; harmonic-oscillator energy unchanged vs Stage-1 baseline within 1e-8.
- **Accept:** `cargo test -p fock_sirk`.

### Stage 3: Proper BRST orthogonal projection
- `nested_fock_algebra/src/lib.rs`: add `Operator::adjoint()` (Create↔Annihilate) and `Hamiltonian::adjoint()` (conjugate coeffs, reverse + adjoint op strings).
- New `unfer/fock_sirk/src/brst.rs`: `pub fn project_physical(w: &QuantumState, q: &Hamiltonian, tol: f64, max_iter: usize) -> Result<QuantumState, SirkError>` — `P w = w − Q† z` with `(Q Q†) z = Q w` solved matrix-free by CG (operator `x ↦ Q(Q†(x))` via `apply`). Needs `Hamiltonian::adjoint()` for `Q†` (add it to `lib.rs` per the anchor above).
- `forward_sirk.rs:67-78`: replace the subtraction hack with `project_physical(...)`.
- **Tests** (use `models::navier_stokes_brst()` compiled via `compile_expression`, or construct a small nilpotent `Q` directly — `yang_mills_hamiltonian` has no separately exported BRST charge): `‖Q(Pw)‖ < 1e-8`; idempotence `‖P(Pw) − Pw‖ < 1e-10`; self-adjointness `⟨v,Pw⟩=⟨Pv,w⟩`.
- **Accept:** `cargo test -p fock_sirk brst`.

### Stage 4: State-explosion bounds + bounded CAS expansion
- `nested_fock_algebra/src/lib.rs`: add `QuantumState::{norm, prune(eps), truncate_top_k(k), len}`.
- `forward_sirk.rs`: add `pub struct SirkOpts { prune_eps: f64, max_components: Option<usize>, brst_tol: f64 }` (with `Default`) and `solve_forward_sirk_with_opts(...) -> Result<ForwardSirkResult, SirkError>`. Change `solve_forward_sirk` to return `Result<_, SirkError>` and **update all callers** (discover with `grep -rn solve_forward_sirk`— 7 examples + tests). After each `w_k`: `prune(opts.prune_eps)`; over `max_components` → `SirkError::StateExplosion`.
- `nested_fock_algebra/src/cas.rs`: add `ExpansionLimits { max_terms }` + `compile_expression_bounded(...) -> Result<Hamiltonian, CasError>` (`CasError::TermExplosion`, thiserror) by threading a counter through existing expansion. **Do not restructure quadratic-ordering logic**; existing `compile_expression` delegates with `usize::MAX`.
- **Tests:** high-order expression returns `TermExplosion` (not OOM); energies stable under `prune_eps = 1e-12`.
- **Accept:** `cargo test --workspace`.

### Stage 5: Re-enable the Navier-Stokes test
- `test_navier_stokes_compiles` (`unit_tests.rs:148-154`) is currently a **vacuous pass** (entire body commented out) — uncommenting alone is not enough; the test must actually run. Rebuild it to: `compile_expression_bounded(navier_stokes_hamiltonian(1e-3), &limits)` (note the builtin returns an `Expression`, so it must be compiled), assert non-empty terms, then drive `solve_forward_sirk_with_opts` with explicit `SirkOpts` + `Device::Cpu` and assert it returns `Ok` (or a *typed* `CasError::TermExplosion`/`SirkError`, not a panic). Expected original root cause: term explosion or Gram degeneracy (both handled by Stages 2 & 4). Fix `models.rs::navier_stokes_hamiltonian` only if a genuine math bug surfaces; document in the test comment.
- **Accept:** `cargo test -p nested_fock_algebra -- navier` passes with a non-trivial assertion in < 120 s on CPU.

### Stage 6: Restarted Krylov + state reconstruction (long-running evolution)
- `forward_sirk.rs`: add `pub w_sequence: Vec<QuantumState>` to `ForwardSirkResult` (currently dropped after the solve — keep the `Vec` built at `:57-81`). Add `pub fn reconstruct(&self, coeffs: &DVector<Complex64>) -> QuantumState` — maps whitened-basis `coeffs` through the stored `w_whiten` (Stage 2) to `w_sequence` coordinates, then linearly combines `w_sequence` via `scale_and_add`. (This is the missing inverse of `time_evolve`, which today returns coefficients with no way back to a `QuantumState`.)
- New `unfer/fock_sirk/src/evolve.rs`: `pub fn evolve_restarted(h, psi0, t, n_restarts, krylov_dim, device, brst, opts) -> Result<QuantumState, SirkError>` — loop: SIRK build → `time_evolve(t/n_restarts)` → `reconstruct` → `prune` → feed result back as the next `psi0` → repeat.
- **Tests:** norm conservation `|‖ψ(t)‖−1| < 1e-6` across restarts (2-mode model); agreement with single-shot evolution within 1e-6 for small t.
- **Accept:** `cargo test -p fock_sirk evolve`.

---

## Workstream B — protocol, Born-rule layer, FFI (Stages 7–9)

### Stage 7: `unfer_protocol` crate (the single shared contract)
- New `unfer/unfer_protocol/` (deps: serde, serde_json, thiserror only; add to workspace members). Files: `src/lib.rs`, `src/codes.rs`, `src/types.rs`.
- `types.rs` (all serde):
  - `ModelSpec { hamiltonian: HamiltonianSpec, prior: PriorSpec, solver: SolverSpec }`
  - `HamiltonianSpec::{ Builtin { name, params }, Latex(String), Terms(Vec<TermSpec>) }`; `TermSpec { coeff_re, coeff_im, ops: Vec<OpSpec> }`; `OpSpec { kind: OpKind, level: Level, mode: u32 }`
  - `PriorSpec::{ Vacuum, Bosons(Vec<(u32,u32)>), Fermions(Vec<u32>), Superposition(Vec<(f64,f64,PriorSpec)>) }`
  - `EventPredicate::{ BosonModeTotal { mode, cmp: Cmp, value }, FermionModePresent { mode }, BosonUniverseCount { cmp, value }, FermionUniverseCount { cmp, value }, Vacuum, And(Vec<_>), Or(Vec<_>), Not(Box<_>) }`; `Cmp::{Eq,Ge,Le,Gt,Lt}`
  - `SolverSpec { krylov_dim, prune_eps, max_components, restarts, device: DeviceSpec }`
  - Agent envelopes: `AgentRequest { id, op, params }`, `AgentResponse { id, ok, result, error: Option<Diagnostic> }`
- `codes.rs`: `Code(u32)` consts + `pub fn all() -> &'static [(u32, &'static str, &'static str)]` registry. Allocation: **1xxx validation** (1001 BadJson, 1002 UnknownBuiltinModel, 1003 BadEventPredicate, 1004 BadHandle, 1005 BufferTooSmall) · **2xxx solver** (2001 GramDegenerate, 2002 StateExplosion, 2003 ZeroProbabilityCondition, 2004 BrstNotConverged, 2005 CasTermExplosion) · **3xxx resource** (3001 CudaUnavailable, 3002 OutOfMemoryBudget) · **4xxx auth** (4001 CallDenied) · **5xxx internal** (5000 Internal).
  - `Diagnostic { code, name, message, severity, hints: Vec<RepairHint>, data }`; `RepairHint { kind: HintKind, target, suggestion }`; `HintKind::{ReplaceValue, SetParam, ReduceScope, IncreaseLimit, UseAlternativeOp}` — this is the Zero-language-style machine surface.
- **Tests:** serde round-trip every type; code uniqueness in `all()`.
- **Accept:** `cargo test -p unfer_protocol`.

### Stage 8: `prob_kernel` crate (Born-rule layer)
- New `unfer/prob_kernel/` (deps: nested_fock_algebra, fock_sirk, unfer_protocol, num-complex, serde_json, thiserror). Files: `src/{lib,session,event,build,error}.rs`.
- `event.rs`: `pub fn matches(outer: &OuterState, pred: &EventPredicate) -> bool` — pure, exhaustive; `BosonModeTotal` sums `inner.modes[mode] × universe_count` over the outer bosonic map.
- `build.rs`: `build_hamiltonian(spec) -> Result<Hamiltonian, KernelError>` — `Builtin` dispatches to `models::{yang_mills, navier_stokes, gravity}_hamiltonian` + add a simple `harmonic_chain` builtin to `models.rs` for tests/demos; `Latex` via existing `compile_latex` (feature passthrough); `Terms` = direct construction (explosion-safe path). `build_prior(spec) -> Result<QuantumState, _>`.
- `session.rs` (long-running handle, not batch):
  ```rust
  pub struct Session { /* state, hamiltonian, opts, device, t_now */ }
  impl Session {
      pub fn new(spec: &ModelSpec) -> Result<Self, KernelError>;
      pub fn set_prior(&mut self, p: &PriorSpec) -> Result<(), KernelError>;
      pub fn set_hamiltonian(&mut self, h: &HamiltonianSpec) -> Result<(), KernelError>;
      pub fn evolve(&mut self, t: f64) -> Result<EvolveReport, KernelError>;   // evolve_restarted
      pub fn probability(&self, e: &EventPredicate) -> Result<f64, KernelError>;
      pub fn condition(&mut self, e: &EventPredicate) -> Result<f64, KernelError>; // returns prior P(e); UK-2003 if mass < eps
      pub fn snapshot(&self, top_k: usize) -> StateSummary;
  }
  ```
  `probability(E) = Σ_{s ⊨ E} |⟨s|ψ⟩|² / ‖ψ‖²`; `condition(E)` zeroes non-matching components then renormalizes.
- `error.rs`: `KernelError` with `From<SirkError>`, `From<CasError>`, and `to_diagnostic() -> Diagnostic` mapping **every** variant to a UK code with ≥1 RepairHint (e.g. StateExplosion → `IncreaseLimit{target:"solver.max_components"}`).
- **Tests:** probabilities sum to 1 on `harmonic_chain`; `condition(E)` then `probability(E)==1.0`; impossible event → UK-2003 with hint; post-evolve normalization within 1e-6.
- **Accept:** `cargo test -p prob_kernel` (CPU).

### Stage 9: `unfer_ffi` crate (handle-based C ABI)
- New `unfer/unfer_ffi/` — `crate-type = ["cdylib", "staticlib", "rlib"]` (rlib so cranelift can take fn pointers via normal Rust dep). Files: `src/lib.rs`, `src/handles.rs`, `include/unfer_kernel.h` (hand-written doc-of-record).
- Design: all params i64-compatible (ptr+len; `t` goes inside opts JSON — CPS IR calling convention is i64-centric). Return ≥0 = success (handle/byte count); <0 = `-code`. Buffer protocol: return total bytes needed, copy `min(needed, cap)`; caller re-calls with a bigger buffer.
  ```c
  int64_t uk_version(void);
  int64_t uk_init(const uint8_t* cfg_json, int64_t len);
  int64_t uk_model_create(const uint8_t* spec_json, int64_t len);   // ModelSpec → handle
  int64_t uk_model_free(int64_t model);
  int64_t uk_set_prior(int64_t model, const uint8_t* json, int64_t len);
  int64_t uk_set_hamiltonian(int64_t model, const uint8_t* json, int64_t len);
  int64_t uk_evolve(int64_t model, const uint8_t* opts_json, int64_t len);   // {"t":1.0,...}
  int64_t uk_condition(int64_t model, const uint8_t* event_json, int64_t len);
  int64_t uk_event_probability(int64_t model, const uint8_t* event_json, int64_t len);
  int64_t uk_observe(int64_t model, const uint8_t* obs_json, int64_t len);
  int64_t uk_get_result(int64_t model, uint8_t* buf, int64_t cap);
  int64_t uk_last_error(uint8_t* buf, int64_t cap);
  ```
  _(`uk_subscribe`/`uk_poll` were specced here originally but **removed in v1** — see P2.8 below.)_
- `handles.rs`: `Mutex<HashMap<i64, SessionEntry>>` + monotonic counter; thread-local `LAST_ERROR`.
- **Tests** (pure Rust calling extern fns with raw pointers): happy path create→prior→evolve→probability→free; bad handle → `-1004`; bad JSON → `-1001` + parseable Diagnostic from `uk_last_error`.
- **Accept:** `cargo test -p unfer_ffi`; `nm -D target/release/libunfer_ffi.so | grep uk_model_create` finds the symbol.

---

## Workstream C — australVM rewiring + module recipe (Stages 10–13)

### Stage 10: `AuthorizationEngine` trait; Cedar becomes optional
- New `australVM/safestos/cranelift/src/auth.rs`:
  ```rust
  pub enum Decision { Allow, Deny }
  pub trait AuthorizationEngine: Send + Sync {
      fn authorize(&self, principal: &str, action: &str, resource: &str) -> Result<Decision, String>;
  }
  pub struct AllowAll;
  pub struct ManifestAuthEngine { grants: HashMap<String, HashSet<String>> } // module → callable uk_* names
  // from_toml_str, merge; global OnceLock<RwLock<Option<Box<dyn AuthorizationEngine>>>>; set_auth_engine; check()
  #[no_mangle] pub extern "C" fn safestos_load_auth_manifest(ptr: *const u8, len: usize) -> i64;
  ```
  Fallback when no engine installed: Cedar engine if feature `cedar` on, else `AllowAll` with logged warning (preserves current behavior).
- `cranelift/Cargo.toml`: `cedar-policy` → `optional = true`; `[features] default = ["cedar"]`, `cedar = ["dep:cedar-policy"]`.
- `policy.rs`: gate behind `#[cfg(feature = "cedar")]`; add `impl AuthorizationEngine for CedarVmEngine`.
- `cps.rs:8-20`: rename `check_cedar_permission` → `check_call_permission`; keep the `__`/`au_`/self-call short-circuit; body becomes `crate::auth::check(caller, "Call", callee)`. Update all call sites (`grep -rn check_cedar_permission`).
- **Accept:** crate builds with default features AND `--no-default-features`; existing tests pass; unit test for `ManifestAuthEngine::from_toml_str` grant/deny.

### Stage 11: Register kernel symbols in the JIT
- `cranelift/Cargo.toml`: `unfer_ffi = { path = "../../../unfer/unfer_ffi", optional = true }`; feature `unfer-kernel = ["dep:unfer_ffi"]` in defaults.
- `cranelift/src/lib.rs` in `cranelift_init()` after line 72: `#[cfg(feature = "unfer-kernel")]` block with `builder.symbol("uk_<name>", unfer_ffi::uk_<name> as *const u8);` for all 14 functions.
- **Do NOT add `uk_` to the whitelist in `check_call_permission`** — authorizing kernel calls per-module is the point; modules get access only via manifest grants.
- **Accept:** builds with default features; `#[test]` calling `unfer_ffi::uk_version()` proves linkage; end-to-end proof lands in Stage 13.

### Stage 12: Austral-side kernel bindings
- **Gate:** first verify the OCaml toolchain builds (`dune build` / repo's documented command in `$ROOT/australVM`). If it fails, record the blocker and skip to Stage 13's fallback (handwritten CPS test cell).
- New `australVM/examples/kernel/UnferKernel.aui/.aum`: declare `uk_*` as foreign imports (mirror how `au_print_int` is declared — find with `grep -rn "au_print_int" --include=*.aum --include=*.aui`), exposing typed wrappers (`kernelVersion(): Int64`, `kernelModelCreate(spec: Address, len: Int64): Int64`, …). Prefer wrapping the handle in a linear type (no leak/double-free); if that fights the compiler subset, plain Int64 is acceptable v1 — document the choice.
- **Accept:** module compiles to `.cps`; run through the scheduler prints `uk_version()`.

### Stage 13: Module recipe + demo module + manifest enforcement
- New sibling folder `$ROOT/demo_module/`:
  - `module.toml` (normative example):
    ```toml
    [module]
    name = "demo_module"
    version = "0.1.0"
    archetypes = ["prior_provider", "actor"]
    entry = "src/DemoModule"
    [grants]
    kernel = ["uk_version","uk_model_create","uk_set_prior","uk_evolve","uk_event_probability","uk_get_result","uk_last_error","uk_model_free"]
    ```
  - `src/DemoModule.aui/.aum`: imports `UnferKernel`; `act` creates a `harmonic_chain` model, evolves t=1.0, reads `P(BosonModeTotal{mode:0,Eq,1})`, prints scaled integer. `build.sh` asserts sibling layout, invokes the austral compiler.
- Host: new `australVM/safestos/cranelift/src/bin/modhost.rs` — loads auth manifest(s) via `safestos_load_auth_manifest`, reads `.cps` cells, JIT-compiles, runs entry export. (Avoids touching `scheduler.c`; hot-swap stays available via existing `__au_swap_module`/CellDescriptor.)
- `run_demo.sh` in demo_module: build unfer_ffi → build cranelift crate → compile module (or use prebuilt CPS fallback from Stage 12 gate) → run modhost with manifest. Include the **negative test**: removing `uk_evolve` from grants must fail with the UK-4001 denial.
- New `unfer/docs/MODULES.md` — THE recipe for future modules: folder layout, manifest schema, the three archetype contracts (exact Austral signatures: PriorProvider `provide_prior(model: Int64): Int64`; DataSource `update(model: Int64, payload: Address, len: Int64): Int64`; Actor `act(model: Int64): Int64`), lifecycle (build → load → grant → run → hot-swap → unload), and a numbered "add a module" checklist.
- **Accept:** `bash $ROOT/demo_module/run_demo.sh` prints a probability in [0,1]; grant-removal negative test fails with denial.

---

## Workstream D — velysterm UI + AI interface (Stages 14–17)

### Stage 14: `kernel_client` crate
- New `velysterm/crates/kernel_client/` (workspace member; deps: `prob_kernel`/`unfer_protocol` via `../../../unfer/...` path deps, serde_json; **no Bevy**).
- `src/lib.rs`: `KernelClient` with one worker thread + mpsc channels (kernel solves must never block the frame loop). `BlockRequest { block_id: u64, op: KernelOp }`; `KernelOp::{ DefineModel(ModelSpec), Evolve{model_block,t}, Probability{model_block,event}, Condition{model_block,event} }`; `BlockResponse { block_id, result: Result<Value, Diagnostic> }`; `submit`/`try_recv`. Worker owns `HashMap<u64, Session>` keyed by model-block id, with spec-hash caching (skip rebuild if unchanged).
- `src/parse.rs`: `parse_model(text) -> Result<ModelSpec, Diagnostic>` — narrow v1: builtin syntax `name(key: value, ...)` (e.g. `yang_mills(g: 0.5)`) or `latex"..."` routed to `HamiltonianSpec::Latex`. `parse_event(text)` mini-grammar: `n(mode) == k`, `occupied(mode)`, `vacuum`, combinators `& | !`. Parse errors carry UK-1002/1003 with `ReplaceValue` hints listing valid names. **No general Typst-math→Hamiltonian compiler in v1** — documented extension point.
- **Accept:** `cargo test -p kernel_client` (parser cases + worker integration test on CPU).

### Stage 15: PropKinds in mathed_core (safe while M2 is unfinished)
- `velysterm/crates/mathed_core/src/markers.rs`: add `PropKind::{Model, Prior, Event, Prob}` + `of()` mappings ("model"/"prior"/"event"/"prob"), following the existing `\function(#1,#2)`/`extra_args` statement convention (e.g. `\prob(heads)(#1…#2)`).
- `mathed_core/src/semantics.rs`: in `build_index`, collect `KernelStatement { kind, block, name: Option<String>, body_text, span }` into `SemanticIndex.kernel_statements` (reuse existing segment/extra_args extraction).
- **Accept:** `cargo test -p mathed_core` — new test: doc with `\model`/`\event`/`\prob` yields 3 KernelStatements with correct kinds/names/spans; existing 51 tests stay green.

### Stage 16: Bevy bridge (GUI)
- New `velysterm/crates/mathed/src/kernel_sys.rs`: `#[derive(Resource)] KernelBridge { client, results: HashMap<u64, Result<Value, Diagnostic>>, inflight, spec_hashes }`; systems `dispatch_kernel_requests` (after `sync_blocks`; uses block-damage tracking to resubmit only changed blocks) and `apply_kernel_results` (drains `try_recv`).
- `mathed/src/main.rs` (~10 lines only — M2 is in flux, keep the diff minimal): insert resource, register both systems; extend `draw_overlay` to render `= 0.4231` (green) or `UK-2003` + first hint (red) next to each `\prob` span.
- **Accept:** `cargo build -p mathed`; unit tests for the pure helper `statements_needing_dispatch(index, damage, spec_hashes) -> Vec<BlockRequest>`; manual smoke (type a model + prob statement, see the number) recorded in notes — not blocking.

### Stage 17: AI-agent machine interface (Zero-language spirit)
- New `velysterm/crates/kernel_client/src/bin/unfer_agent.rs`: NDJSON request/response loop on stdin/stdout. Ops: `version`, `create_model`, `set_prior`, `evolve`, `condition`, `probability`, `snapshot`, `list_codes` (dumps `codes::all()` so agents self-document). Every failure carries a `Diagnostic` with hints; unknown op → UK-1001 + `ReplaceValue` hint listing valid ops.
- New `unfer/docs/PROTOCOL.md`: envelope schema, every op with request/response examples, full code table, repair-hint semantics, rules for allocating new codes.
- **Accept:** `printf '{"id":"1","op":"version","params":{}}\n' | cargo run -p kernel_client --bin unfer_agent` → `{"id":"1","ok":true,...}`; bad op → `ok:false`, code 1001, non-empty hints; integration test via `std::process::Command`.

---

## Stage 18: Documentation, extension points, final verification
- New `unfer/docs/ARCHITECTURE.md`: system diagram, sibling-folder convention, cross-repo dependency graph, and four documented extension points with checklists:
  1. **Add a module** → MODULES.md checklist.
  2. **Add a kernel op** → protocol type + Session method + `uk_` shim + JIT symbol + agent op + code allocation (file list).
  3. **Add a PropKind** → markers.rs + semantics.rs + kernel_sys.rs.
  4. **Add a builtin model** → `models.rs` + `build.rs` dispatch + parser name list.
- Update `AGENTS.md` in all three repos (new crates/layout; note resolved limitations: BRST, Cholesky, explosion bounds, NS test, CPU fallback).

## Stage outcomes (what each stage delivered)

| Stage | Outcome |
|---|---|
| S1 | `cuda` is a feature on `fock_sirk`; `device::best_device()` picks `Cuda(0)` or `Cpu`; all examples migrated. CPU-default builds/tests green. |
| S2 | `linalg::whiten_gram` (Hermitian eigendecomp, rank-r `W = U_r Λ_r^{-1/2}`) replaces the `cholesky().expect()` panic at `forward_sirk.rs:140`. `ForwardSirkResult.w_whiten` + `rank` added. |
| S3 | `Operator::adjoint`/`Hamiltonian::adjoint` added; `brst::project_physical` via CG replaces the subtraction hack; idempotence + self-adjointness tested. |
| S4 | `QuantumState::{norm, prune, truncate_top_k, len}`; `SirkOpts { prune_eps, max_components, brst_tol }` + `solve_forward_sirk_with_opts`; `cas::compile_expression_bounded` with `ExpansionLimits`/`CasError::TermExplosion`. |
| S5 | `test_navier_stokes_compiles` rebuilt to drive `compile_expression_bounded` + `solve_forward_sirk_with_opts` with a non-trivial assertion (no longer vacuous). |
| S6 | `ForwardSirkResult.w_sequence` retained; `reconstruct(coeffs)` maps whitened-basis coeffs back to `QuantumState`; `evolve::evolve_restarted` loops build→evolve→reconstruct→prune. Norm conservation tested. |
| S7 | `unfer_protocol`: `Code`/`Diagnostic`/`RepairHint`/`ModelSpec`/`HamiltonianSpec`/`PriorSpec`/`EventPredicate`/`SolverSpec`/`AgentRequest`/`AgentResponse`. Code registry `codes::all()`. |
| S8 | `prob_kernel::Session` (`new`/`set_prior`/`set_hamiltonian`/`evolve`/`probability`/`condition`/`snapshot`); `harmonic_chain` builtin in `models.rs`; `KernelError::to_diagnostic()` maps every variant to a UK code + hint. |
| S9 | `unfer_ffi`: 14 `uk_*` extern "C" fns (edition 2024 → `#[unsafe(no_mangle)]`), handle table, last-error buffer. `nm -D libunfer_ffi.so` confirms symbols. |
| S10 | `cranelift/src/auth.rs`: `AuthorizationEngine` trait, `ManifestAuthEngine::from_toml_str`, global `OnceLock<RwLock<…>>`, `safestos_load_auth_manifest` FFI. Cedar → optional default feature. |
| S11 | All 14 `uk_*` registered in `cranelift_init()` behind `unfer-kernel` (default). `uk_*` deliberately **not** whitelisted in `check_call_permission` — manifest grants required. |
| S12 | `australVM/examples/kernel/UnferKernel.aui/.aum` + `TestKernel.au`. `cps.rs` auto-declares `uk_` symbols; `Compiler_cps.ml` resolves `MConcreteFuncall` to `External_Name`. (Austral handle = plain `Int64` in v1 — linear-type wrapping deferred, see gaps.) |
| S13 | `demo_module/` (`module.toml`, `src/DemoModule.{aui,aum}`, `build.sh`, `run_demo.sh`) + `modhost.rs`. Docs split into two complementary files: **`MODULES.md`** (the prose recipe — folder layout, archetype contracts, add-a-module checklist) and **`MODULE_RECIPE.md`** (the normative `module.toml` schema). Plus `BUILD_PIPELINE.md`. |
| S14 | `kernel_client`: `KernelClient` (worker thread, crossbeam mpsc), `parse.rs` (`parse_model`/`parse_event`), `KernelBridge`-friendly request/response types. |
| S15 | `PropKind::{Model,Prior,Event,Prob}` + `is_kernel()`; `KernelStatement` collected in `SemanticIndex::build_index`; `find_block_for_doc_pos` helper. |
| S16 | `mathed/src/kernel_sys.rs`: `KernelBridge` resource, `dispatch_kernel_requests` + `apply_kernel_results` systems, `statements_needing_dispatch` pure helper (7 tests). Overlay `prob_ok`/`prob_err` rendering. Systems registered after `sync_blocks`. |
| S17 | `kernel_client/src/bin/unfer_agent.rs`: 8 NDJSON ops (`version`/`create_model`/`set_prior`/`evolve`/`condition`/`probability`/`snapshot`/`list_codes`). Unknown op → UK-1001 + `ReplaceValue`. `unfer/docs/PROTOCOL.md` written. |
| S18 | `unfer/docs/ARCHITECTURE.md` (diagram, dep graph, 4 extension-point checklists). `AGENTS.md` updated + deduplicated. |

## Known gaps & deferred items

These were called for in the stage specs but were **not** completed, or were explicitly deferred. They are the highest-signal starting points for "next steps".

1. ~~**No `demo_module/` end-to-end (S13 partial).**~~ **RESOLVED.** `demo_module/` now lives at `unfer/demo_module/` (inside the unfer repo, not at `$ROOT/demo_module/`). Contains `module.toml`, `src/DemoModule.aui/.aum`, `build.sh`, `run_demo.sh`. The `modhost.rs` binary exists at `australVM/safestos/cranelift/src/bin/modhost.rs`. `run_demo.sh` exercises the positive path (now a **module-driven probability**: builds a JSON `ModelSpec` literal, JIT-creates a real model via `uk_model_create`, computes `uk_event_probability`, frees through the linear `Model`), the UK-4001 negative test (grant revocation), and the P2.7 linearity gate (a leaked `Model` fails to compile). **VERIFIED PASSING end-to-end on 2026-06-26** (OCaml 4.13.1 / dune 3.20.2). The earlier "only `uk_version`" nuance is closed — see gap §9.
2. ~~**Commit hygiene — all repos pushed; one item remains.**~~ **RESOLVED.** All three repos are committed AND pushed (unfer `6ce121a`→`origin/main`, australVM `40df9c5c`→`origin/master`, velysterm `62062dc`→`origin/gitbutler/workspace`). The 6 `cps.rs.*` backup files were removed in australVM commit `198cc137` (only `_build/` artifacts remain, not tracked by git). **Remaining:** uncommitted P4 #21/#22 work in unfer + velysterm (see P5 #23).
3. ~~**Austral handle is plain `Int64` (S12 v1 choice).**~~ **RESOLVED at the type level (2026-06-26, P2.7).** `UnferKernel` now exposes a linear `Model` wrapper (`wrapModel`/`modelHandle`/`freeModel`) so `uk_model_free` is a compile-time obligation: the `LeakDemo` negative gate in `run_demo.sh` confirms a module that wraps a handle and forgets to free it fails to compile with a **Linearity Error**. **Update (2026-06-26):** the wrapper now also **runs through the CPS-JIT** after the gap §9 backend fixes (record destructure + cross-module linking + let-init); `DemoModule` drives `wrapModel`→`freeModel` in-JIT. The raw `Int64` C ABI functions remain (the wrapper is built on them). The linear guarantee is enforced by the Austral typechecker regardless of backend.
4. ~~**`uk_subscribe` / `uk_poll` are provisional.**~~ **RESOLVED (2026-06-26) — deleted for v1 (P2.8).** Removed from `unfer_ffi` (impl, header, stub test), the cranelift JIT symbol list, `UnferKernel.aui/.aum` foreign imports, and `MODULE_RECIPE.md` grants. No concrete consumer existed; a documented absence beats an untested promise. Re-add with a real subscriber design (event vocabulary + backpressure) when a consumer appears.
5. ~~**No CUDA/GPU test ever ran.**~~ **RESOLVED (2026-06-27, P4 #21).** The `cuda` feature compiles AND a GPU smoke test now runs: `gpu_smoke_hopping_energy_matches_cpu` in `forward_sirk.rs` asserts `best_device()` picks CUDA and the two-state hopping Hamiltonian's Ritz values match ±1 within 1e-8. Requires `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` on systems with CUDA 12.2 toolkit + CUDA 13 runtime coexistence (the `CUBLAS_STATUS_ARCH_MISMATCH` from the version mismatch is the exact failure AGENTS.md §5 warns about). All 14 CUDA tests pass.
6. ~~**Overlay manual smoke deferred (S16).**~~ **RESOLVED (2026-06-27, P4 #21).** The on-screen render is verified two ways: (a) launched the `mathed_mini` frontend, confirmed green `= 1.0000` pixels appear next to the `\prob` statement (screenshot at `velysterm/docs/mathed/overlay_smoke_screenshot.png`); (b) headless pixel-color test `overlay_renders_green_for_success_and_red_for_error` captures the full visual pipeline (document → kernel → annotations → Typst layout → rasterized RGBA8 → green/red pixel assertion) — verifying both the success path (green pixels) and the error path (red pixels) in CI without a display.
7. **No Typst-math → Hamiltonian compiler (documented extension point).** S14's parser only handles `name(k: v, …)` builtins and `latex"…"`. Rich Typst math input remains a v2 extension point.
8. ~~**velysterm workspace test is broken on upstream `velyst` examples (NEW, found 2026-06-26).**~~ **RESOLVED (2026-06-27, P4 #16).** The two stale upstream examples (`editor`, `terminal`, referencing removed `VelystFuncBundle`/`VelystSourceHandle`) are gated behind a non-default `upstream-stale-examples` feature (explicit `[[example]]` `required-features` in `crates/velyst/Cargo.toml`), so `--all-targets` skips them and `cargo build -p velyst --examples` succeeds. The project's own clippy lint baseline (the two `single_range_in_vec_init` test lints) is also clean now (P4 #17). Port the two examples to the current `velyst` API by enabling the feature when someone needs them.
9. **CPS-JIT backend: record/cross-module/buffer support — RESOLVED (2026-06-26).** The experimental safestos CPS-JIT originally executed only `uk_*` foreign calls. Found and fixed several backend bugs so the linear `Model` wrapper now **runs end-to-end through the JIT** (`DemoModule` calls `kernelVersion`, then cross-module `wrapModel`→`freeModel`, the latter destructuring the linear record to recover the handle; `run_demo.sh` shows `Execution result: 1`):
   - **Local `let`-initialisation** (`lib/Compiler_cps.ml`, `MAssignVar`/`MInitialAssign`): assignments used the *qualified* name (`T1::x`) while `MLet` declared the *unqualified* local (`x`), so every `let x := v` left `x` at its zero-init. This was fundamental — it silently broke **all** local variables; the demo only ever did `return kernelVersion()` directly, so it was never hit. Now uses the unqualified name. Verified: `let x := 5; return x` → 5.
   - **Record destructuring** (`Compiler_cps.ml`, `MDestructure`): passed the field *name* as a variable to `__slot_get` instead of a numeric slot offset (`Undefined variable: handle`). Now resolves the offset from the record layout, mirroring `MSlotAccessor`. Verified: single-field record construct+destructure → correct value.
   - **Cross-module non-foreign calls** (`Compiler_cps.ml` emit the unqualified callee name; `safestos/cranelift/src/cps.rs` declares unknown callees as `Linkage::Import`): the persistent JITModule already retains each module's `Export`ed functions, so an `Import` of the same name links to the prior definition. `UnferKernel::wrapModel` now resolves.
   - **Robust entry execution** (`cps.rs`): the per-module path executed a *random* function from each module's table — fine for a leaf entry, but it would run library functions like `freeModel` with garbage arguments (dereferencing a non-pointer → crash). Now executes only a conventional `run` entry; library modules (no `run`) are defined-but-not-executed, which is both safe and what enables cross-module linking.

   **Update (2026-06-26) — byte buffers RESOLVED; a module now computes a real probability.** Two more fixes landed:
   - **String/byte-buffer constants** now lower (`StringLit`, opcode `0x21` in `cps.rs`): the bytes are embedded in heap memory and a `Span` is produced as a pointer to a heap `{data@0, size@8}` struct. `@embed "$1.data"`/`"$1.size"` (`Compiler_cps.ml` `MEmbed`) become slot-0/slot-8 loads, so a module decomposes a span into the `(ptr, len)` pair the C ABI wants — sidestepping the by-value 16-byte `au_span_t` aggregate the i64-JIT can't pass. Also fixed `MStringConstant` to use `Escape.escaped_to_string` (real bytes) rather than `unescape_string` (which re-inserted C escapes and corrupted the buffer). New `UnferKernel` helpers `kernelModelCreateStr`/`kernelEventProbabilityStr` take a `Span[Nat8, Static]`. **`DemoModule` now builds a JSON `ModelSpec` literal, JIT-creates a real model (`uk_model_create`, handle > 0), computes `P(boson mode 0 ≥ 1)` (`uk_event_probability`) on the superposition prior, and frees the handle via the linear `Model`** — the full `uk_model_create`→`uk_event_probability` chain driven from Austral, in-process. `run_demo.sh` step 4 asserts it.
   - **Multi-field records — verified working (no fix needed).** An earlier draft of this gap claimed concrete multi-field records had broken slot offsets (a supposed `decl_id`↔`mono_id` layout mismatch). That was a **misdiagnosis**: the crashes were entirely the let-init NULL bug above (`let r := …` left `r = 0`, so `__slot_get(NULL, …)` segfaulted). Concrete record types are monomorphized on demand (`add_or_get_record_monomorph`) into an `MRecordMonomorph` keyed by the *same* `mono_id` that `MonoNamedType` carries, so `record_layouts` does resolve their slot offsets. Verified end-to-end: a 3-field record reads fields at offsets 0/8/16 correctly (10/20/30). `MRecordConstructor` was simplified to size `__record_new` from its field count — a harmless robustness change (it equals the layout value), not a bugfix. A **regression gate** (`demo_module/src/RecordCheck.aum`, run_demo.sh step 7) now asserts a multi-field offset read.

   **Gap §9 is fully closed:** record destructure, cross-module calls, let-init, byte buffers, multi-field records, and a module-computed probability all run in-JIT, each with a gate in `run_demo.sh`.

## velysterm mini-frontend progress (beyond S1–S18)

The `mathed_mini` crate is a separate, optional Bevy-free CPU frontend for constrained hardware, tracked in `velysterm/docs/mathed/MINI_FRONTEND_PLAN.md`. Its status:

- **Increment 1 + 2** (committed `0ed6015`): `MiniWorld` (standalone `typst::World`), CPU renderer via `imaging_vello_cpu`, winit + softbuffer window, editing at end-only.
- **Increment 3** (committed `a456156`, extended in `f378be4`): `mathed_core::glyphs` (Bevy-free glyph index ported from `mathed::glyphs`), `DocLayout` caching (foot-style: layout recomputed only on edit/resize), caret positioning via `caret_for_byte`, full navigation (Left/Right/Home/End/Backspace/Delete + Up/Down via `band_for_byte` → `byte_for_point`). Up/Down nav + `band_for_byte` helper landed in `f378be4`. 6 mathed_mini tests + 59 mathed_core tests green.
- **Deferred:** Step 4 (caret blink via `ControlFlow::WaitUntil`), mouse hit-testing / click-to-place-caret + selection (the `byte_for_point`/`rects_for_range` plumbing already exists), `mathed_a11y` (AccessKit bridge over `mathed_core::accessibility`), and — the big one — wiring `kernel_client` into `mathed_mini` so the Bevy-free frontend can show `\prob` results too (today only the Bevy `mathed` frontend has the kernel bridge).

## Next steps to improve (prioritized, recommended 2026-06-27)

**The system is feature-complete.** All 18 original stages, all P0–P4 hardening/growth items, and Workstream E (QFM) are done and verified. The remaining work is **commit debt** (uncommitted P4 #21/#22 work), **frontend parity** (the Bevy frontend lacks the mini's inline annotations), and **physics depth** (larger-scale computations, the off-diagonal QFM operators). These are ordered by risk: A freezes the current state, B closes the frontend gap, C grows the physics.

1. **Commit the uncommitted work (P0).** Two repos have uncommitted work: unfer (GPU smoke test + docs) and velysterm (P4 #21/#22). Commit + push both before anything else lands.
2. **Then close frontend parity, then grow physics** (P5 below).

### P0 — prove the integration spine ✅ DONE (2026-06-26)
1. ~~**Clean `cps.rs.*` backups.**~~ **DONE.** Removed all 6 in australVM commit `198cc137` (master ahead 1, awaiting `!` push). Closes gaps §2.
2. ~~**Verify `demo_module/run_demo.sh` end-to-end.**~~ **DONE — passes.** OCaml 4.13.1 / dune 3.20.2 present; `dune build lib/ bin/` + release `unfer_ffi` + `--no-default-features --features unfer-kernel` cranelift/modhost builds all succeed. Positive: `DemoModule` JIT-creates a real model from a JSON spec and computes a probability (`Execution result: 1`). Negative: stripped manifest denies `uk_evolve` with UK-4001. **Follow-up — DONE (2026-06-26):** the positive path now drives the full `uk_model_create`→`uk_event_probability` chain from Austral (gap §9 byte-buffer work); a module-computed probability runs end-to-end, not just `uk_version`.

### P1 — lock in what works
3. ~~**CI (CPU).**~~ **DONE for unfer.** Added `unfer/.github/workflows/ci.yml` with 4 jobs: `test` (`cargo build`+`test --workspace`), `lint` (`fmt --check` + `clippy -D warnings`), `ffi-symbols` (builds `libunfer_ffi.so` and asserts the 5 load-bearing `uk_*` symbols are exported via `nm -D` — verified against the real lib), and `demo-e2e` (checks out the sibling australVM, sets up OCaml 4.13, runs `run_demo.sh` — the spine gate). velysterm (`rust.yml`) and australVM (`build-and-test.yml`) already have CI; **note velysterm's CI is currently red** because `cargo test --workspace --all-targets` hits the broken upstream `velyst` examples (gaps §8) — fix or exclude those targets. **Remaining:** wire a PAT if australVM is a private repo (the `demo-e2e` job has a commented `token:` slot).
   - **Lint-baseline fix (2026-06-26, commits `00332ae`, `20e358a`):** the new `lint` job (`cargo fmt --all -- --check` + `cargo clippy --workspace --all-targets -- -D warnings`) was **red on arrival** — the codebase had never been run through the pinned rustfmt (~50 files drifted) and clippy flagged pre-existing lints (unused imports, `len() > 0`, an identity-op in the navier_stokes example). Both are now fixed in dedicated mechanical commits, so the gate is meaningful from the first run. A red-on-arrival gate trains everyone to ignore it, so this had to be cleaned before the gate goes live.
4. ~~**Single cross-repo green sweep.**~~ **DONE (2026-06-26).** Recorded in the status block above: unfer 96 · mathed_core 59 · mathed_mini 6 · kernel_client 4 · mathed 36 · cranelift 9 + `--no-default-features` build — all green in the real sibling layout. This is the baseline CI must hold.
5. ~~**Overlay GUI smoke (S16).**~~ **DONE (2026-06-27, P4 #21).** Launched `mathed_mini`, confirmed green `= 1.0000` renders next to `\prob` (64 green pixels, screenshot at `velysterm/docs/mathed/overlay_smoke_screenshot.png`). Headless pixel-color test `overlay_renders_green_for_success_and_red_for_error` verifies both success (green) and error (red) paths in CI.
6. ~~**GPU smoke (if a CUDA box is reachable).**~~ **DONE (2026-06-27, P4 #21).** `cargo test -p fock_sirk --features cuda` = 14 tests green. `gpu_smoke_hopping_energy_matches_cpu` asserts `best_device()` picks CUDA and Ritz values match ±1 within 1e-8. Requires `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` for CUDA 12.2/13 coexistence.

### P2 — harden the v1 shortcuts
7. ~~**Austral linear handle wrapper (S12 upgrade).**~~ **DONE (2026-06-26).** Added a linear `Model` record + `wrapModel`/`modelHandle`/`freeModel` to `UnferKernel.aui/.aum`; `freeModel` is the sole consumer and calls `uk_model_free`, so the type system now forces every `Model` to be freed exactly once. Proven by a **tested negative gate**: `demo_module/src/LeakDemo.aum` leaks a `Model` and `run_demo.sh` asserts it fails compilation with a *Linearity Error*. **Now also runs in-JIT (2026-06-26, gap §9 work):** after fixing the CPS-JIT backend (let-init, record destructure, cross-module linking, robust entry), `DemoModule` drives the wrapper end-to-end through the JIT — `wrapModel` (record construct) then `freeModel` (destructure → `uk_model_free`) — returning the kernel version (`Execution result: 1`). The compile-time guarantee holds regardless of backend; the wrapper now executes too.
8. ~~**Resolve `uk_subscribe`/`uk_poll`.**~~ **DONE (2026-06-26) — deleted for v1.** Confirmed no concrete consumer (no module, agent op, or velysterm code), then removed the dead surface end-to-end: `unfer_ffi` impl + C header + stub test + `handles.rs` subscription table, the two cranelift JIT `builder.symbol(...)` registrations, the `kernelSubscribe`/`kernelPoll` foreign imports in `UnferKernel.aui/.aum`, and the `MODULE_RECIPE.md` grant entries. unfer_ffi tests (15) + clippy clean; cranelift rebuilds with `--features unfer-kernel`. **Cross-repo:** unfer + australVM (australVM commit on `master` awaits `!` push). Re-add with a real subscriber design if/when a streaming consumer exists.
9. ~~**`KernelError` → `Diagnostic` coverage audit.**~~ **DONE (2026-06-26, commit `b5a48d3`).** Added two table-driven tests in `prob_kernel/src/error.rs`: `every_variant_maps_to_registered_code_with_hint` enumerates one instance of all 14 `KernelError` variants (incl. every inner `SirkError`/`CasError`) and asserts each maps to a code present in `codes::all()` **and** carries ≥1 non-empty `RepairHint`; `user_actionable_variants_avoid_internal_catchall` asserts non-internal variants never degrade to UK-5000. The audit **found and fixed** three hint-less variants — `Sirk(Numeric)`, `BadJson`, `Internal` — which previously returned a `Diagnostic` with an empty `hints` vec, silently breaking the agent repair contract.

### P3 — grow capability
10. **User-defined translator pipeline (S14 extension point).** ~~Originally "Typst-math → Hamiltonian compiler through `mathhook`".~~ **Pivoted (2026-06-26):** editor users do not write typst-math directly; they type rendered math (display-only) and define a **translator** — a Typst function authored as code in a collapsible panel — that maps the math source string to a `TermSpec[]` JSON payload for the kernel. The existing `name(k: v)` shortcut parser (`velysterm/crates/kernel_client/src/parse.rs`) is replaced by this pipeline. Full design (architecture, data flow, 6-step implementation plan, technical risks, resume state for a new agent): **`velysterm/docs/mathed/TRANSLATOR_DESIGN.md`**. **Steps 1–6 + P3 #11 + follow-ups COMPLETE (2026-06-26):** Step 1 marker layer in `mathed_core` (`PropKind::Translator`, `TranslatorDef`, `KernelStatement.translator`); Step 2 typst-eval (0.14.2) resolved via the let-binding path (append `#let __result = translate(<body>)`, read the binding — no `Vm`); Step 3 `mathed_mini::translate` engine + `builtin_translator.typ`; Step 4 `mathed_mini::dispatch` (`statement_to_model_spec`→`HamiltonianSpec::Terms`, `statement_to_event_json`); Step 5 collapsible panel rendering (`transform.rs` renders a translator body as a `▸ translator: name` summary or an expanded raw block via the panel-only `TransformOptions.caret`; `render.rs` `active_translator_span` + `mathed_mini` `app.rs` relayouts only on panel-boundary crossing); Step 6 default translator + docs. Kernel wiring (P3 #11): `mathed_mini::KernelBridge` drives the worker thread, both frontends share one path, `parse.rs` deleted. **Follow-ups (2026-06-26):** (a) translator caching — `model_hashes`/`prob_hashes` include translator source so editing a `\translator` re-dispatches dependents (Risk C resolved); (b) multi-model documents — `\prob`/`\event` carry an optional `model: "name"` arg binding to a named `\model` instead of nearest-preceding; missing names produce a synchronous `model-not-found` error; (c) richer event predicates — separate `builtin_event_translator.typ` (emits `{"kind":"vacuum"}`) + typed `EventPredicate` validation in `statement_to_event_json` catches bad shapes before the worker round-trip. mathed_core 71 + mathed_mini 31 tests.
11. **Wire `kernel_client` into `mathed_mini`.** ~~Today only the Bevy `mathed` frontend has the kernel bridge.~~ **DONE (2026-06-26):** `mathed_mini::kernel_bridge::KernelBridge` builds the semantic index, dispatches each `\model`/`\prob` through the translator pipeline (P3 #10), and drives the `kernel_client` worker thread; results are keyed by statement doc offset and each prob is associated with its nearest preceding model. The `KernelRequest::Probability`/`Condition` protocol gained a `model_id` (which session) separate from `block_id` (result key), so a prob can reference a model in another block. The mini `app.rs` refreshes on edit, busy-polls during a bounded window (`ControlFlow::Poll`), and shows a `#raw` results panel below the document (`render::layout_doc_with_footer`); the seed doc demos a live `\prob`. End-to-end test: a vacuum model + Vacuum-predicate prob computes P = 1.0 through the real worker thread + `prob_kernel` on CPU. The value is shown **inline** next to the `\prob` (a coloured value spliced into the render via the kernel-agnostic `TransformOptions.annotations`; the footer panel API is retained). The Bevy `mathed` frontend was then ported to the same `mathed_mini::KernelBridge` (thin Bevy wrapper; overlay keyed by `ks.span.start`) and the v1 `kernel_client::parse` shortcut deleted — **both frontends now share one kernel path**. (P3.10 translator pipeline Steps 1–6 + P3.11 are complete.)
12. **Builtin model library.** ~~Beyond `harmonic_chain`/`navier_stokes`/`yang_mills`/`gravity`: one documented, tested builtin per flagship target.~~ **First entry DONE (2026-06-26):** `bose_hubbard` — `bose_hubbard_chain(n_modes, t, u, periodic)` in `nested_fock_algebra/src/models.rs` (nearest-neighbour hopping `-t(aᵢ†aⱼ + h.c.)` + on-site `U/2·nᵢ(nᵢ-1)`, optional periodic boundary). Dispatched in `prob_kernel/src/build.rs` (with `get_bool_or` helper for the `periodic` flag), parsed in `kernel_client/src/parse.rs`, listed in the UK-1002 valid-names hint, and documented in `ARCHITECTURE.md`. Tested at unit (`test_bose_hubbard_structure`: hopping/interaction term counts, periodic wrap, `u=0` reduces to quadratic), integration (`bose_hubbard_builds_and_normalizes`, `bose_hubbard_hopping_conserves_norm` evolve+norm), and parser (`parse_bose_hubbard`) levels. **Second entry DONE (2026-06-27):** `yang_mills_lattice(l, g, n_colors)` — the Yang-Mills mass-gap lattice flagship (see P4 #20 for full detail).
13. ~~**Benchmarks.**~~ **DONE (2026-06-27)** — see P4 #19. `fock_sirk/benches/sirk.rs` (criterion) guards the SIRK solve + Gram-whiten + reconstruct numerics with measured curves, not just pass/fail.

### P4 — hardening + growth (2026-06-26 → 2026-06-27, all DONE)

> The translator pipeline (P3 #10) is now feature-complete with caching,
> multi-model binding, and typed event-predicate validation. The system's
> remaining gaps fall into three buckets: **(A) close the document-driven
> model spec** (wire `\prior`/`\solver` — today the dispatcher hardcodes
> `PriorSpec::Vacuum`/`SolverSpec::default()`), **(B) fix CI / lint debt**
> (broken velyst examples, remaining clippy lints), and **(C) harden the
> repair contract** (translator errors lack `RepairHint`s). These are
> ordered by dependency: A unblocks richer documents, B unblocks green CI,
> C completes the Zero-language agent surface.

14. ~~**Commit + push the translator follow-up work.**~~ **COMMITTED (`0916146`, 2026-06-26).** All follow-up files committed on `gitbutler/workspace`; tests green (mathed_core 71, mathed_mini 31), clippy clean. Remaining: USER runs `git push origin gitbutler/workspace`.
15. ~~**Wire `\prior` and `\solver` segments through the dispatcher.**~~ **DONE (velysterm `da57c44`, 2026-06-26).** The document-driven model spec can now set a non-vacuum prior and tune the solver (was hardcoded `PriorSpec::Vacuum` / `SolverSpec::default()`). `PropKind::Solver` added (`markers.rs`, `is_kernel()`, `accessibility.rs` `AccessRole::Solver`); `\prior`/`\solver` bind via `model: "name"` or nearest-preceding (`semantics.rs` `model_name` extended). `dispatch::{parse_prior,parse_solver}` parse the **segment body** with an editor-friendly mini-grammar (`vacuum` / `bosons(0:2, 1:1)` / `fermions(0, 2)` for priors; `krylov_dim: 12, restarts: 2` for solvers) falling back to direct JSON; `statement_to_model_spec` gained `prior`/`solver` params (Vacuum/default fallback, backward-compatible) + a new `DispatchError::Parse`. `kernel_bridge::refresh` resolves each `\prior`/`\solver` to its model, folds the body into the model hash (edits re-dispatch), applies it, and surfaces a `prior-solver-parse` error at the segment on a bad body. **Deviation:** the spec lives in the segment **body** (`#1 vacuum #2 \prior(#1,#2)`), not an extra arg — consistent with `\model`/`\prob` and renders the prior visibly. Tests: parse grammar+JSON+error cases, `model_spec_applies_prior_and_solver`, `prior_reaches_kernel_and_changes_probability` (end-to-end P=1.0 on a one-boson prior), `bad_prior_body_surfaces_parse_error`, semantics prior/solver collection. mathed_core 72, mathed_mini 39 green; clippy clean; `mathed` builds. **Needs push** (`git push origin gitbutler/workspace`).
16. ~~**Fix velysterm workspace CI (gap §8).**~~ **DONE (2026-06-27).** The two upstream-vendored `velyst` examples (`editor`, `terminal`) that reference removed APIs (`VelystFuncBundle`/`VelystSourceHandle`, E0422/E0425) are now gated behind a non-default `upstream-stale-examples` feature via explicit `[[example]]` `required-features` entries in `crates/velyst/Cargo.toml`. `cargo build -p velyst --examples` now succeeds (the other 7 examples still build; the two stale ones are skipped unless `--features upstream-stale-examples`), so `cargo test --workspace --all-targets` no longer fails to compile. This unblocks velysterm CI (`rust.yml`). Re-enable the feature to port them to the current API later.
17. ~~**Clean remaining clippy lints in `mathed_core`.**~~ **DONE (2026-06-27).** The two test-only `single_range_in_vec_init` warnings (`transform.rs` `vec![1..1]`, `wordnav.rs` `vec![2..8]`) now build the one-element `Vec<Range>` via `std::iter::once(..).collect()` (the `[..].to_vec()` form still trips the lint). `cargo clippy -p mathed_core --all-targets -- -D warnings` is clean. Closes the lint baseline gap §8 for the project's own crates.
18. ~~**Translator errors → `RepairHint` mapping.**~~ **DONE (2026-06-27).** `KernelResult::Error` gained a `hints: Vec<RepairHint>` field, and `mathed_mini::kernel_bridge` now maps every user-triggerable failure to a concrete hint (Zero-language agent surface): `dispatch_error_hints` covers all `DispatchError`/`TranslateError` variants (`Eval`→fix Typst error w/ `first_line`, `NotString`→return `json.encode(..)`, `MissingResult`/`Empty`, `Json`→fix output JSON, `Parse`→fix prior/solver body) — only the internal `WrongKind` misuse, which a frontend never dispatches, is hint-less; the missing-named-model error lists the model names actually in scope. Worker-side `Diagnostic.hints` are forwarded through `BlockResponse::Error`. Tests: `dispatch_errors_carry_repair_hints` (all variants), strengthened `missing_named_model_surfaces_error` (hint names `m1`). mathed_core 72, mathed_mini 40 green; clippy clean; `mathed` builds. **Needs push** (`git push origin gitbutler/workspace`).
19. ~~**Benchmarks (P3 #13).**~~ **DONE (2026-06-27, unfer).** Added `fock_sirk/benches/sirk.rs` (criterion 0.5, `harness = false`, `[[bench]] name = "sirk"`) with three groups covering the load-bearing numerics: `sirk_solve_vs_krylov_dim` (forward solve on a 4-mode `harmonic_chain` vs Krylov dim 2/4/8), `whiten_gram_vs_size` (Hermitian eigendecomp on a deterministic PSD `MᴴM + nI` matrix, n = 4/8/16/32), `reconstruct_vs_krylov_dim` (whitened→`QuantumState` reconstruction vs `w_sequence` length). Verified runnable (`cargo bench -p fock_sirk --bench sirk`): whitening scales ≈cubic (5.6→34.8→132µs at n=8/16/32), reconstruct ≈linear (283ns→1.05µs at m=2→8) — the expected curves. `cargo test -p fock_sirk` (13) still green, clippy `--all-targets` clean, rustfmt clean. (The translator-eval-cost bench the spec mentioned lives in velysterm, not the unfer kernel — out of scope for this crate.)
20. ~~**More flagship builtin models (P3 #12 continuation).**~~ **DONE (2026-06-27, unfer).** Added `yang_mills_lattice(l, g, n_colors)` to `nested_fock_algebra/src/models.rs` — a Kogut–Susskind-inspired Hamiltonian lattice gauge theory on a periodic `l × l` 2D lattice with `n_colors` bosonic gauge fields per link. Electric energy `(g²/2) Σ_ℓ n_ℓ` gaps the spectrum (each excited link costs g²/2 — the lattice origin of the mass gap); the *quartic* magnetic plaquette term `-(1/2g²) Σ_p Φ(ℓ1)Φ(ℓ2)Φ(ℓ3)Φ(ℓ4)` (Φ = a† + a) is the combinatorial four-operator interaction that stress-tests the bounded direct-construction path (each plaquette/color emits 2⁴ = 16 quartic sub-terms over four distinct commuting modes → hermitian). Dispatched in `prob_kernel/src/build.rs` (`"yang_mills_lattice"`, new `get_u64_or` helper for the `n_colors` default), added to the UK-1002 valid-names hint in `error.rs`, documented in `ARCHITECTURE.md` (builtin set + "Add a builtin model" checklist refreshed — note velysterm's `parse.rs` is gone, builtins reached via agent/translator). Tests: `test_yang_mills_lattice_structure` (8 electric + 64 magnetic terms on a 2×2 1-color lattice, real coeffs, color-doubling, `l` clamp) and `yang_mills_lattice_builds_and_evolves` (vacuum prior → evolve through the quartic plaquette term → norm ≈ 1, cover sums to 1). unfer workspace tests now 98 (+2); clippy `--all-targets` clean, rustfmt clean. **Needs push** (unfer main).
21. ~~**Remaining P1 items.**~~ **DONE (2026-06-27).** (a) **Overlay GUI smoke (P1 #5):** launched the `mathed_mini` frontend (winit + softbuffer, CPU rasterizer), confirmed the inline kernel overlay renders — the green `= 1.0000` annotation appears next to the `\prob` statement (64 green pixels at the expected screen location, RGB matching `#138000`). Screenshot saved at `velysterm/docs/mathed/overlay_smoke_screenshot.png`. Additionally, a headless pixel-color test (`overlay_renders_green_for_success_and_red_for_error` in `kernel_bridge.rs`) captures the full visual pipeline (document → kernel → annotations → Typst layout → rasterized RGBA8 → green/red pixel assertion) — verifying both the success path (green) and the error path (red `code_name`) without needing a display. (b) **GPU smoke (P1 #6):** `cargo test -p fock_sirk --features cuda` passes on CUDA 12.2 (driver 13.0; required `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` to resolve the `CUBLAS_STATUS_ARCH_MISMATCH` from the CUDA 13 runtime — see AGENTS.md §5). New `gpu_smoke_hopping_energy_matches_cpu` test in `forward_sirk.rs` asserts `best_device()` picks CUDA (`Device::Cuda(_)`) and the two-state hopping Hamiltonian's Ritz values match ±1 within 1e-8 — the one test that exercises the GPU tensor path (inner products + Gram matrix + H_proj on the CUDA device). All 14 CUDA tests pass (was 13 CPU-only).
22. ~~**Mini-frontend polish (deferred from S14/Increment 3).**~~ **DONE (2026-06-27).** Three features added to `mathed_mini/src/app.rs`: (a) **Caret blink** via `ControlFlow::WaitUntil` — the caret toggles visibility at ~530ms intervals (terminal convention), with `about_to_wait` waking the event loop at `next_blink` when not busy-polling the kernel; all keyboard/mouse input resets the blink (visible + timer restart); `redraw` skips the caret bar when `caret_visible` is false. (b) **Mouse hit-testing** — `CursorMoved` tracks the physical pixel position, `MouseInput` with `MouseButton::Left` converts to a byte offset via `GlyphIndex::byte_for_point` and places the caret (reusing the existing `byte_for_point`/`rects_for_range` plumbing). (c) **AccessKit bridge** — new `mathed_mini/src/a11y.rs` module converts `mathed_core::accessibility::AccessNode`s into `accesskit::TreeUpdate` (root `Document` node owns segment children, each mapped `AccessRole`→`accesskit::Role`); `app.rs` wires `accesskit_winit::Adapter::with_event_loop_proxy` (window starts invisible, adapter created, then shown), processes events through the adapter, and pushes tree updates on initial show and after every edit. Dependencies: `accesskit 0.21` + `accesskit_winit 0.29` added to `mathed_mini/Cargo.toml` behind the `gui` feature. Headless build (`--no-default-features`) still works; 44 tests pass (was 41, +3 a11y unit tests); clippy clean.

### P5 — next priorities (recommended 2026-06-27, post-feature-complete)

> The system is feature-complete: all 18 stages, P0–P4, and Workstream E
> are done and verified. The remaining work falls into three buckets:
> **(A) commit debt** — uncommitted P4 #21/#22 work in two repos,
> **(B) frontend parity** — the Bevy `mathed` frontend lacks the mini's
> inline green-number/red-code annotations, and neither frontend has
> text selection, and **(C) physics depth** — larger-scale computations,
> the off-diagonal QFM operators, and more flagship models. These are
> ordered: A is zero-risk housekeeping, B closes the UX gap, C grows
> the science.

23. **Commit the uncommitted P4 #21/#22 work.** Two repos have uncommitted work: unfer (`fock_sirk/src/forward_sirk.rs` GPU smoke test + doc edits) and velysterm (`app.rs`, `a11y.rs`, `kernel_bridge.rs`, `Cargo.toml`, `lib.rs` + screenshot + rustfmt). Commit + push both before anything else lands. Zero risk, zero new code — just freezing what's already verified green.
24. ~~**Bevy frontend parity: inline annotations.**~~ **DONE (already complete pre-P5; `translator_errors` parity added 2026-06-27, velysterm `cb16625`).** The Bevy `main.rs` already called `result_annotations()` and passed it to `TransformOptions::annotations`. The remaining parity gap was `translator_errors` (P5 #28's inline error display in expanded translator panels). `kernel_sys::KernelBridge` gained a `translator_errors()` forward; `main.rs` passes `translator_errors: translator_errors.clone()` in the per-block `TransformOptions`. mathed 29 tests green.
25. **Text selection support in `mathed_mini`.** The `byte_for_point`/`rects_for_range` plumbing exists but only click-to-place-caret is wired. Add: Shift+click to extend selection, mouse drag to select a range, Shift+Arrow keys for keyboard selection, and a selected-text highlight rect drawn in `redraw`. Copy (Ctrl+C) can initially just put the raw source text on the clipboard; paste (Ctrl+V) inserts at the caret. Medium risk — touches the event loop and redraw, but the geometry helpers are already tested.
26. **Off-diagonal QFM differential operators.** The `qfm_mehler` builtin (E20) uses number operators `n_j`, making `H` strictly diagonal — `e^{-iHt}` adds only phases and populations don't "spread". The off-diagonal differential operators `ĥ_j` of QMF.tex §2.3 that mix vacuum↔data are the mathematically complete version. Implementing them would let the QFM module actually perform inference (populations flow from vacuum to data channels). This is the highest-value physics extension: it would turn the QFM module from a stationary eigenstate demo into a working generative flow.
27. **AccessKit action wiring.** The AccessKit bridge (P4 #22) pushes the tree and handles `InitialTreeRequested`, but `ActionRequested` (focus, click) is a no-op. Wire it: a focus action on a segment node places the caret at that segment's byte offset; a click action does the same. Low risk — the caret placement code already exists (mouse hit-testing uses the same path).
28. ~~**Translator authoring UX.**~~ **DONE (2026-06-27, velysterm `3ad1464`).** The translator panel now shows: (a) `\`\`\`typ` language tag on the code fence so syntect syntax-highlighting (Typst keywords/strings/comments) applies when available — plain monospace fallback otherwise; (b) inline error display — when the panel is expanded and a translator fails (bad Typst / wrong JSON output), the error message is shown in red below the code (`#text(fill: red)[⚠ ...]`), so the error is visible in the panel itself, not just as a red `code_name` on dependent `\prob`s. Architecture: `TransformOptions::translator_errors` (offset→message); `KernelBridge::translator_errors()` populated by `refresh` on Translate/Json dispatch errors, cleared on successful re-dispatch; `app.rs` passes it to `TransformOptions`. Tests: `translator_error_populates_translator_errors_map` + `translator_error_clears_on_fix`. mathed_core 72, mathed_mini 53 green, clippy/rustfmt clean.
29. ~~**`result_panel_markup()` wiring.**~~ **DONE (2026-06-27, velysterm `1c44f6b`).** `app.rs` now uses `layout_doc_with_footer` (replacing `layout_doc_with`), passing `bridge.result_panel_markup().unwrap_or_default()` as the footer. The `#raw` results summary appears below the document in the mini frontend and is rebuilt whenever `about_to_wait` drains new kernel results (same invalidation path). mathed_mini 53 green, clippy/rustfmt clean.
30. **Larger-scale physics computations.** The current models are tiny (2–4 modes) for test speed. To approach the Millennium Prize targets (Yang-Mills mass gap, Navier-Stokes): (a) profile the `yang_mills_lattice` builtin at `l=4` or `l=6` (32–72 links, 256–576 quartic terms) — the bounded CAS path should handle it but performance is unknown; (b) verify the SIRK solver's numerical stability at larger Krylov dimensions (`m=16`, `m=32`); (c) run the criterion benchmarks on these larger sizes to get real scaling curves. The GPU path (smoke-tested at 2 states) should be exercised on a real model.
31. **CI green-run verification.** velysterm CI was unblocked (P4 #16 — stale examples gated) but the actual GitHub Actions run hasn't been confirmed green post-fix. Also: wire a PAT if australVM is a private repo (the `demo-e2e` job has a commented `token:` slot). And add the QFM module demo (`qfm_module/run_demo.sh`) as a second CI gate alongside `demo_module/run_demo.sh`.
32. **Hot-swap module testing.** The `__au_swap_module`/CellDescriptor mechanism exists in the CPS-JIT but was never tested at runtime (modules are loaded once and run). A test that loads a module, runs it, hot-swaps a new version, and runs again would validate the hot-swap path — the one architectural claim that's still unverified.

## Workstream E — Quantum Flow Matching (QFM) module (QMFplan.md / QMF.tex, 2026-06-27)

> Adapted from `QMFplan.md` + `QMF.tex`: an analytical, neural-network-free
> generative flow built on the existing Fock/SIRK substrate. Data points become
> orthogonal single-boson modes, the Mehler uniform prior is the rank-1 vacuum
> projector `|0><0|`, and the decoupled potential keeps construction O(M). All
> three QMFplan stages done and verified end-to-end.

E19. ~~**Mehler prior + QFM Hamiltonian.**~~ **DONE (2026-06-27, unfer).** Added `Operator::ProjectVacuum` to `nested_fock_algebra` — the self-adjoint, idempotent rank-1 `|0><0|` (adjoint arm returns itself; apply keeps only the strict-vacuum component, drops anything carrying a mode). `models::qfm_hamiltonian(alphas)` builds `H = |0><0| + Σ_j α_j a†_j a_j` directly (one `ProjectVacuum` term + one number operator per data point), bypassing `Expression::expand()` so M can be huge. Test `test_qfm_hamiltonian` proves `H|0> = |0>` (eigenvalue 1 from the projector) and `H|x_j> = α_j|x_j>` (diagonal, no cross-terms, no vacuum leakage).
E20. ~~**Protocol + Born-rule integration.**~~ **DONE (2026-06-27, unfer).** `prob_kernel/src/build.rs` dispatches the `"qfm_mehler"` builtin, reading the `alphas` array via a new `get_f64_array` helper; added to the UK-1002 valid-names hint (`error.rs`). No `HamiltonianSpec` change needed (`Builtin { params }` already carries the array). Integration tests `qfm_mehler_builds_and_evolves` (vacuum prior → P(vacuum)=1, evolve, norm ≈ 1, vacuum is a QFM eigenstate so its population is stationary, cover sums to 1) and `qfm_mehler_conserves_data_channel_population` (a seeded data channel is an eigenvalue-α_j eigenstate → occupation conserved under the diagonal generator). **Honest deviation:** the simplified builtin uses number operators `n_j` (per QMFplan Stage 19), making `H` strictly diagonal — so `e^{-iHt}` adds only phases and populations don't "spread"; the off-diagonal differential operators `ĥ_j` of QMF.tex §2.3 that mix vacuum↔data are a future extension. Tests assert the mathematically correct stationary behavior, not a false spread.
E21. ~~**The QFM Austral module.**~~ **DONE (2026-06-27, unfer + australVM).** New `unfer/qfm_module/` (mirrors `demo_module/`): `module.toml` (archetypes `data_source` + `actor`; grants `uk_model_create`/`uk_evolve`/`uk_event_probability`/`uk_model_free`), `src/QfmModule.{aui,aum}`, `build.sh`, `run_demo.sh`. The module embeds the analytically-precomputed α_j weights, builds a `qfm_mehler` ModelSpec JSON with the Mehler vacuum prior + `krylov_dim:15`, JIT-creates it in-process (`uk_model_create`), runs the single-step O(m²) inference (`uk_evolve`), reads back P(channel 0 occupied) (`uk_event_probability`), and frees via the linear `Model`. Added a `kernelEvolveStr` convenience binding (span→pointer+length, mirroring `kernelModelCreateStr`) to `australVM/examples/kernel/UnferKernel.{aui,aum}`. `bash unfer/qfm_module/run_demo.sh` passes: CPS-JIT `Execution result: 1` (model created+evolved+queried in-process) and the UK-4001 negative test (revoking `uk_evolve` denies QFM inference). **Needs push** (unfer main + australVM master — the UnferKernel binding edit).

**Verification:** unfer `cargo test --workspace` green (+3: 1 unit `test_qfm_hamiltonian`, 2 integration `qfm_mehler_*`), clippy `--all-targets` clean, rustfmt clean, QFM module demo green end-to-end.

## Historical risks & mitigations (from planning)
- **CUDA availability** — Stage 1 is first; every acceptance criterion runs on CPU; `cuda` is additive.
- **OCaml toolchain may not build** — Stage 12 has an explicit verify-first gate; Stage 13's `modhost.rs` + prebuilt/handwritten CPS fallback keeps workstream C completable regardless. *(Note: `modhost.rs` was built; `demo_module/run_demo.sh` uses `dune build lib/ bin/` + the CPS-JIT path.)*
- **velysterm M2 unfinished** — Stages 14–15 touch only stable `mathed_core` + a new crate; the only M2-adjacent edit is ~10 isolated lines in `main.rs` (Stage 16).
- **cas.rs fragility** — Stage 4 adds a bounded wrapper around existing expansion; no restructuring; existing tests untouched.
- **`solve_forward_sirk` signature change** — Stage 4 explicitly updates all callers in one commit (`grep -rn solve_forward_sirk`).
- **Cross-repo path deps** — build scripts assert the sibling layout with a clear error; `unfer-kernel` and `cedar` are cargo features so australVM still builds standalone.
