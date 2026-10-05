# Agent Guidelines: Fock-Sirk Project

Welcome, Agent. This repository contains high-performance tools for quantum field theory (QFT) simulations using Nested Fock Spaces and Rational Krylov methods.

## Technical Architecture

Detail: **`docs/ARCHITECTURE.md`**. The rules that are easy to get wrong:

- **SIRK split-mode.** Forward sequence $w_k=(H-z_kI)w_{k-1}$ on the CPU (the
  branching is exponential, so it stays sparse); `StateDictionary` offloaded, Gram
  matrix and $H_{proj}$ formed on GPU via `candle-core`. Device:
  `Device::cuda_if_available(0)`.
- **CAS.** Fields $\mapsto a^\dagger+a$, momenta $\mapsto i(a^\dagger-a)$. The
  compiler MUST drop pure scalar terms in distribution — that is what makes
  $\langle0|H|0\rangle=0$. Physics Hamiltonians must commute with the BRST charge.
  **Quartic-and-worse models MUST bypass `Expression::expand()`** and build
  `Operator` structures directly: $O(10^4)$ terms recurse infinitely in `distribute`.
- **LaTeX.** `mathhook` → AST → `latex.rs` → operator strings. The LALRPOP parser
  **requires explicit `*` or `\cdot`**; implicit multiplication by spacing is a
  parse error.
- **Numerics.** Inverse-free, so no $O(N^3)$ solves. $H_{proj}=W^\dagger H_{raw}W$
  by eigendecomposition (whitening replaced a Cholesky that panicked on degenerate
  Grams); if singular, reduce $m$ or shift $z_k$. **Always** use `nalgebra`'s Padé
  `exp()` for reduced-system evolution or unitarity is lost.
- **GPU triage (T2.2).** `device::probe_cuda`/`best_device` emit
  `UK-GPU-<CODE> → <fix>`: `NO_DEVICE`, `ARCH_MISMATCH` (libcublas/libcuda vs
  driver — fix `LD_LIBRARY_PATH`), `LIBRARY_MISSING`, `OUT_OF_MEMORY` (shrink
  basis/window), `OTHER`. See `fock_sirk/examples/debug_cuda.rs`.

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
- [ ] **Cadabra2 symbolic coupling (S30)**: Cadabra2 (GPL-3.0) is a **subprocess**, never linked; `verified` is the zero-detection verdict (`H - H† = 0`); engine-dependent tests **skip** without `cadabra2-cli`. Read `docs/CADABRA2_GOTCHAS.md` before touching a `.cdb` — the `@()` four-character rule and the `unwrap` trap are not discoverable from a failure message.
- [ ] **Faris–Lavine `N` (S30f)**: every numbered CHECK in the NS/QG modules reduces to 0. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **NS Koopman–von Neumann generator (S30g)**: `H = π^i(u_j u_{i,j} + q_i − ν u_{i,jj}) + h.c.`; the literal `H²` is **not** a valid `N`. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **Pressure Poisson equation (S30h)**: contracting `∂_i` with the momentum equation under `div u = 0` gives `Δp = −div[(u·∇)u] + div f`. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **The pressure constraint is SECOND-CLASS (S30i)**: eliminate it, do not BRST it. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **Jordan → Einstein frame (S30j)**: this is what makes `starobinskyV` *derived* rather than defined. → `docs/VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md`
- [ ] **Core of `N` and the lift (S30k)**: `N` ESA on a **graph core**; `dΓ(N)` is a different operator, so ESA is reproved on the lifted core. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **NS/QYM inner-operator audit (S30l)**: the proof's `nsFockSpace` is `L²(ℝ^{21n})`; the audit's `ℝ^{18n}` was stale and is fixed. → `docs/VERIFY_FARIS_LAVINE_N.md`
- [ ] **Standard Model `N` (S30m)**: CHECK 1–28 verified; the CAR/ghost, spinor/Lorentz, parameter and Hilbert-space statements are **not** certified and are Lean work. → `docs/VERIFY_SM_FARIS_LAVINE.md`
- [ ] **Logos CNL→UNF (S31)**: `verified` is a confluence self-check; proof terms must stay `rfl` — `native_decide` emits terms nanoda cannot reduce. → `docs/LOGOS.md`
- [ ] **Austral↔DeltaNet↔UNF↔TED (S37)**: `ted_hash` is the content-addressable *algebraic* hash on top of `unf_hash`. → `docs/LOGOS.md`
- [ ] **WhyML compiler-extension cycle (S36)**: Why3 (LGPL) is **subprocess-only**; `check_with_grants` is pure so tests never mutate the process env. → `docs/WHYML_CYCLE.md`
- [ ] **QED validation (S32)**: the γ↔e⁺e⁻ vertex must list positron annihilation **before** electron creation. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **QCD validation (S33)**: colour factors are computed from the SU(3) structure constants, not hard-coded; `C_F·e²` multiplies the amplitude **square**. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **QG validation (S34)**: reproduces the Planck / redshift / perihelion / deflection / GPS anchors. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **NS validation (S35)**: `u = a†+a`, `π = i(a†−a)`, `H = {π,V}` — the symmetrization carries the factor 2. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **SR/nuclear + astro/plasma (S38)**: the charge cancels in `B = pc/(qρ)`. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Physics anchors II (S39)**: the Gaussian year is **365.2568983 days**. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Gauge-fixed program validation (S40)**: distinct inner occupations of one universe merge under `scale_and_add`. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **SIRK dynamics/constraints (S41)**: field amplitude `u = a†+a` is the decaying NS observable, not `N`. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Hashimoto Theorem 4.1 bands (S42)**: rigorous but **conservative** ceilings on unitary models. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Certified numerics (S43)**: the *sharp* tier is Rayleigh–Ritz residual certificates, not the envelope. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **QYM mass gap (S44)**: measured on the **Cadabra-derived gauge-fixed** Hamiltonian, not the lattice builder; the `g²/2` reading was a lattice-electric effect. → `docs/MASS_GAP_SPEC.md`
- [ ] **Outer-vacuum ground doctrine (S45)**: the nested-Fock ground is **always** the outer vacuum; `H|Ω⟩ = 0` identically. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Momentum-space convolution (S47)**: `f⋆g[n] = Σ_m f[m]g[(n−m) mod n]` — **not** `g[(n+m)]`. → `docs/NUMERICAL_VALIDATION_GUIDE.md`
- [ ] **Event-sourced Session (H3)**: `Session` is a write-ahead log — each mutating op appends a typed `SessionEvent` *before* it applies, and `restore` replays with strict-monotonic-seq + bracket-balance validation. **A compaction boundary must be a settled record — never `Evolve` (tool-pairing) and never `CompactStart`**; QFM sessions cannot be compacted at all. `debug_assert_reconstructable` replays the raw log after every op, so replay must stay deterministic. New `SessionEventSpec`/`SessionOp` variants keep the version marker, restore validation and the `PROTOCOL.md` row in sync. → `docs/PROTOCOL.md`, codes UK-1006..1009
- [ ] **Certificate ledger (Plan R)**: the UTXO/carbon-certificate state machine lives in
- [ ] **Certificate ledger (Plan R)**: `unfer_consensus::certs`. Four invariants — conservation on transfer, no double-spend, owner-only spend, mint-authority check — and the whole ledger is a pure function of the log. `MintAuthority::Threshold` makes minting a t-of-n Arctic/Shine check, verified identically on submit and on replay. → `docs/CONSENSUS_LEDGERS.md`
- [ ] **Unified auction (Plan R)**: clearing is a pure function of the bids (highest `price_per_unit`, ties to earliest `seq`), so every node converges on the same winner; escrow is conserving and its face value equals the bid total. → `docs/CONSENSUS_LEDGERS.md`
- [ ] **Math catastrophe bond (Plan R)**: the trigger is a nanoda proof run deterministically inside `apply_op` — no human oracle; a live bond cannot settle early and the id commits the full terms. → `docs/CONSENSUS_LEDGERS.md`
- [ ] **Bond probability market (Plan R)**: buys mint at the **post-trade** marginal price; **resolution is not a caller choice** — it is a pure function of the trigger signal, and a forged resolve is refused. → `docs/CONSENSUS_LEDGERS.md`
- [ ] **Attribution carbon credits (Plan R)**: the ledger takes the **consensus-log** seq, not the op seq, so a replayed approval keeps its badge date; badges are byte-identical across services. → `docs/CONSENSUS_LEDGERS.md`

## Crate Layout

Per-crate detail: **`docs/ARCHITECTURE.md`**. What is load-bearing here:

- **`unfer_protocol` is the contract** six other repos speak — types, UK-#### codes,
  repair hints, the op registry (`ops.rs`), the symbol registry (`symbols.rs`), and
  the G1/G3 board + cooperation vocabulary. A change here is a change everywhere.
- **`unfer_ffi` owns the `uk_*`/`uz_*` C ABI.** Its authoritative census is the
  generated `EXPECTED_SYMBOLS*.txt` — **never a count written in prose**; the H1 gate
  fails the build if a hand-maintained number appears in this file.
- **`unfer_consensus` carries five ledgers** (`certs`, `auction`, `mathbond`,
  `mathbond_market`, `attribution`); `unfer_taler` settles them against GNU Taler.
  All must stay deterministic — same log, same root, no wall-clock or randomness in
  the state machine.
- **`unfer_edge`** is a Pingora proxy in front of `unfer_agent`; `admin.rs` is the
  S22 console. Ops: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

**Path-dependency hazards.** A green run here says nothing about whether these
still fit — they compile against a *sibling's working tree*, which is what the
`cross-repo` CI job exists to check:

| dependent | path-deps |
|---|---|
| `unfer_consensus` | `../dynamic-arctic` (`default-features = false`) |
| `../velysterm/crates/{kernel_client,mathed_core,mathed_mini}` | `unfer_ffi`, `unfer_protocol` |
| `../australVM/safestos/cranelift` | `unfer_ffi`, `unfer_protocol`, `unfer_consensus`, `unfer_identity`, `unfer_data` **and** `../dynamic-arctic` — `unfer-kernel` on by **default** |

The australVM bridge is the most tightly coupled crate in the workspace. Siblings:
`../australVM`, `../velysterm` (also hosts the `unfer_agent` binary),
`../dynamic-arctic`, `../timepiece`. See `PROJECT_PLAN.md`.
## Resolved Limitations

Stages S1–S6, all closed: **CUDA optional** (tests run CPU-only; `cuda` is
additive), **Gram robustness** (eigendecomposition whitening replaced the bare
Cholesky that panicked on degenerate Grams), **BRST projection** (proper
`project_physical` via CG), **explosion bounds** (`SirkOpts` +
`compile_expression_bounded`), **the NS test** (re-enabled, runs the real solver),
**restarted Krylov** (`evolve_restarted` + `reconstruct`).

The QFM star-topology findings — rank saturation, the `m ≥ 3` floor, the
`λ₀` sweet spot, anti-learning at `m = 3`, and the absence of generalization in
single-mode-per-input encoding — are empirical results that took a search to
establish. They live in **[`docs/QFM_STAR_TOPOLOGY.md`](docs/QFM_STAR_TOPOLOGY.md)**.

One API note: `compile_channels` takes a `per_mode_weights` parameter
(`Option<&HashMap<(u32,u32), f64>>`); pass `None` for uniform `λ₁`.

## Core Dependencies
- `candle-core`: GPU tensor management (with the `cuda` feature).
- `mathhook`: LaTeX / math parsing.
- `nalgebra`: linear algebra for the reduced subspace.
- `quantrs2-symengine-pure`: symbolic expression AST.

---
*Note: This project targets the Millennium Prize requirements for Yang-Mills and Navier-Stokes existence by resolving dynamics over discrete Fock-basis boundaries.*
