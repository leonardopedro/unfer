# unfer

> A modular probability kernel for quantum field theory simulations, built on
> Inverse-Free Rational Krylov (SIRK), with a JIT-driven module system, a C
> ABI for in-process module calls, and a Bevy-bridged math-editor UI.

`unfer` is the Rust kernel of a system that computes probabilities of events
in quantum field theory. The pipeline is: physics Hamiltonians (Hermitian
fields, conjugate momenta, BRST ghosts, Majorana fermions) → symbolic
LaTeX/TermSpec compilation → Inverse-Free Rational Krylov reduction on a
probability-weighted Fock basis → Born-rule probability readout. A
tomographic subspace recovery layer (QFM-TSR) makes M-free online
inference tractable at high raw resolution; a HMC posterior sampler
conditions the TSR-evolved prior on new observations.

The system is exposed three ways: a **C ABI** (`unfer_ffi::uk_*`) for
JIT-driven Austral modules, an **NDJSON agent interface** (`unfer_agent`
binary) for AI agents, and a **math-editor UI** (velysterm) for human
interaction.

## Algorithm stack

| Layer | What it does | Crate |
|---|---|---|
| Symbolic CAS | LaTeX → Hamiltonian, Fock-space operators, BRST, mass-gap quadratic ordering | `nested_fock_algebra` |
| SIRK solver | Inverse-Free Rational Krylov time evolution, Hermitian Gram whitening, restarted Krylov | `fock_sirk` |
| Born-rule layer | `Session::{evolve, probability, condition, snapshot, bayesian_update}` | `prob_kernel` |
| Protocol | serde types, UK-#### error codes, `RepairHint`s (Zero-language style) | `unfer_protocol` |
| FFI | `uk_*`/`uz_*` extern "C" functions (census: `unfer_ffi/EXPECTED_SYMBOLS*.txt`), handle table, buffer protocol | `unfer_ffi` |
| QFM-TSR | Two-level hashing, offline training, 4-phase online generate | `qfm` |
| Bayesian update | HMC on the TSR-evolved prior, `|v^dag c|^2` likelihood | `qfm::bayes` |

The QFM (Quantum Flow Matching) layer is the v2 algorithm: a Mehler vacuum
prior + a Hermitian data-coupling flow Hamiltonian + a two-level hash (raw
coordinates → 1-boson Fock modes) + a Krylov reduction + a 4-phase
generate (encode → evolve → tomographic reconstruct → lossless decode).
The Bayesian update conditions the TSR-evolved prior on N new observations
in `O(N · m²)` per HMC step, with **no M dependence**. Full spec in
[`QFM.tex`](QFM.tex).

## Architecture

```
$ROOT/
├── unfer/                      # THE KERNEL (this repo)
│   ├── nested_fock_algebra/    # symbolic CAS + LaTeX→Hamiltonian
│   ├── fock_sirk/              # SIRK time-evolution solver (CPU/CUDA)
│   ├── unfer_protocol/         # serde types, UK-#### codes, repair hints
│   ├── prob_kernel/            # Born-rule layer: Session, condition()
│   ├── unfer_ffi/              # handle-based C ABI: uk_*()
│   ├── qfm/                    # QFM-TSR pipeline + bayes module
│   ├── demo_module/            # 1st end-to-end module demo
│   ├── qfm_module/             # 2nd (Mehler QFM)
│   ├── qfm_tomo_module/        # 3rd (QFM-TSR 4-phase generate)
│   ├── bayes_update_module/    # 5th (Quantum Bayesian Update)
│   ├── demo_module/data_source/# 4th (Rust binary driving the C ABI)
│   └── docs/                   # ARCHITECTURE, PROTOCOL, MODULE_RECIPE
├── australVM/                  # MODULE RUNTIME (sibling)
│   └── safestos/cranelift/     # JIT + auth.rs + uk_* symbols + modhost
└── velysterm/                  # UI / AI INTERFACE (sibling)
    ├── crates/kernel_client/   # worker-thread client + unfer_agent NDJSON
    ├── crates/mathed_core/     # PropKinds (Model, Prior, Event, Prob)
    └── crates/mathed{,_mini}/  # Bevy + Bevy-free math-editor UIs
```

Data flow: modules (Austral cells) and velysterm both drive
`prob_kernel::Session`; modules via the `uk_*` C ABI inside the safestos
JIT (calls authorized per-module by manifest grants), velysterm via
direct Rust dependency (same Session code path). AI agents use the
`unfer_agent` NDJSON binary. Repos stay separate; sibling checkout
layout is required.

## Quick start

### Prerequisites
- Rust **1.99.0** (pinned via `rust-toolchain.toml`; rustup installs it
  automatically and CI runs the same compiler — single toolchain across
  the unfer/dynamic-arctic/australVM/velysterm repos)
- CUDA toolkit 12.x (optional, for GPU SIRK — CPU is the default)
- OCaml 4.13 + opam (only required to compile/run the Austral module demos)
- `libcublas` and `nvcc` on the path (only for GPU)

### Build & test (CPU only)
```bash
cd unfer
cargo build --workspace
cargo test --workspace
```

### Run the 5 module demos (CPU only, requires OCaml + sibling repos)
```bash
bash demo_module/run_demo.sh        # v1 demo
bash qfm_module/run_demo.sh         # Mehler QFM
bash qfm_tomo_module/run_demo.sh    # QFM-TSR 4-phase generate
bash bayes_update_module/run_demo.sh # Quantum Bayesian Update
```

### Talk to the kernel from the agent CLI
```bash
# from unfer/ with velysterm as a sibling:

cd ../velysterm
cargo run -p kernel_client --bin unfer_agent -- --help
printf '{"id":"1","op":"version","params":{}}\n' \
  | cargo run -p kernel_client --bin unfer_agent
```

<!-- status: verified | tests: 1792 cargo (workspace excluding fock_sirk) | last_verified: 2026-10-12 -->

Scope note: this figure is the `unfer` workspace proper and **excludes**
`fock_sirk` (an external physics dependency, 401 tests as of 2026-10-06,
measured separately in `--release` below). Excluding it is a scope choice, not
a refresh, so it is stated rather than left for a reader to infer — as is the
change from the previous figure, 1640.

Measured 2026-10-08 by summing every `test result: ok.` line of
`cargo test --workspace --exclude fock_sirk` (68 test binaries + doc-tests —
54 binary suites + 14 doc-test groups —, 0 failures). **Debug and release
agree exactly**, suite by suite — both were run, which is why the block can be
checked against either. Growth since 1640:
`decide_core` +105 (the D1–D15 analytic decision engine) plus the G1–G9
protocol/ffi/edge additions. Growth since the 1774 of 2026-10-07:
`decide_core` +11 (D7's backbone choice at the seam — Matryoshka
`#truncate`, an `#instruction` prefix and dual-stream concat, each folded
into the embedding identity so two settings cannot share statistics).
Growth since the 1764 of 2026-10-06:
`unfer_protocol` +4 (the `cell_tool` blueprint CLI behind the N2
module-reconstruction gate) and `unfer_edge` +1 (A9's loopback-default
pinning test). Growth since the 1769 of rev 25 (also 2026-10-07): the same
two crates again — `unfer_protocol` +4 (G3's escalation board records) and
`unfer_edge` +1 (X6's `ops_e2e`, a new test binary probing the ops surface
over a real socket, hence 53 → 54 binary suites).

**CI recounts this number.** The test job tees its own release run and hands
the log to `scripts/check-status --recount unfer=<log>`, which refuses a log
containing a failed suite and fails on any mismatch with the block above — so a
stale count here is a red build, not a silent claim.

## Verification

```sh
nix-shell -p zlib-ng cmake --run "cargo test --workspace --exclude fock_sirk"
#   1785 passed (2026-10-08, rev 27; was 1774 on 2026-10-07 — D7's
#   backbone-choice tests added 11 in decide_core)

nix-shell -p zlib-ng cmake --run "cargo test -p fock_sirk --release"
#   401 passed across 56 binaries (2026-10-06; was 371 across 53 on 2026-10-04
#   — three test binaries added by the M1/M4/M5 mass-gap campaign)

nix-shell -p python3 --run "python3 scripts/check-links"
#   relative markdown links resolve (CI: .github/workflows/linkcheck.yml)
```

The workspace suite had **three** known-intermittent tests in `unfer_ffi`'s
action/owner-log group (`action_apply_resolves_and_merges_applied_result`,
`action_submit_queues_pending_record_with_provisional_result`, and
`queue_drops_oldest_when_full`, whose owner ring could be emptied under it by a
test holding a *different* mutex). **A12 closed them**: the inventory of which
test holds which lock is finished — it found that `reset_durable_for_tests`
wipes *all four* shared stores (eleven callers held one lock or none) and that
the spawn-pacing knob was guarded by two different mutexes that did not exclude
each other — every multi-lock test now takes a single order (P → ACTION →
AUDIT → DURABLE), and the one remaining inversion was caught deadlocking the
suite live before being converted. The fixed binary ran **100/100 clean at 16
test threads** (campaign protocol: `timeout 60` per run, so a deadlock counts
as a failure). The full evidence chain lives in
[`docs/TEST_LOCKS.md`](docs/TEST_LOCKS.md).

A run in which a test fails is **not** evidence of a count: the CI test
step stops there, and `scripts/check-status --recount` refuses a log containing
a failed suite rather than summing past it.

`fock_sirk` runs in release because the SIRK suites are numeric; see
`AGENTS.md` on the memory-bounded run options.

The S29/S31 proof tests need Lean **4.28.0** specifically, pinned by githash
`7e01a1bf` — no nixpkgs pin reproduces it, so provision it with elan:

```sh
elan toolchain install leanprover/lean4:v4.28.0
elan default leanprover/lean4:v4.28.0
lake exe cache get     # mathlib v4.28.0 oleans; without it the build is hours
```

`flake.nix` deliberately does not carry Lean for that reason, and does carry the
native deps (`zlib-ng`, `cmake` from the unstable channel) that `unfer_edge`'s
`libz-ng-sys` needs.

### GPU SIRK (optional, requires CUDA toolkit)
```bash
# On systems with CUDA 12.2 toolkit + CUDA 13 driver coexistence:
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  cargo test -p fock_sirk --features cuda
```

## Public APIs (at a glance)

```rust
// Rust — direct session-level use
use prob_kernel::{Session, ModelSpec, HamiltonianSpec, PriorSpec,
                  EventPredicate, SolverSpec};
use unfer_protocol::{HmcOptsSpec, BayesianUpdateRequest};

let mut s = Session::new(&spec)?;
let _ = s.set_prior(&PriorSpec::Vacuum)?;
let report = s.evolve(1.0)?;
let p = s.probability(&EventPredicate::BosonModeTotal { mode: 0, cmp: Cmp::Eq, value: 1 })?;

// QFM-TSR 4-phase generate
use qfm::QfmPipeline;
let pipeline = QfmPipeline::compile(&training_points, &config)?;
let x_out = pipeline.generate(&query)?;

// Quantum Bayesian update on the TSR-evolved prior
use qfm::bayes::{Posterior, sample_hmc_single, tsr_evolved_prior};
let c_prior = tsr_evolved_prior(&pipeline);
let post = Posterior::new(likelihoods, c_prior);
let sample = sample_hmc_single(&post, &HmcOptsSpec::default());

// C ABI (modules + agent)
#include <unfer_kernel.h>
int64_t model = uk_model_create(spec_json, spec_len);
int64_t h = uk_evolve(model, opts_json, opts_len);
int64_t n = uk_get_result(model, buf, cap);
```

## Test & benchmark counts (rev 27, 2026-10-08)

- Workspace tests: **1785 green** on CPU (default features,
  `cargo test --workspace --exclude fock_sirk`, 68 test binaries +
  doc-tests): `unfer_protocol` 400, logos 346, `unfer_ffi` 228,
  `prob_kernel` 143, `unfer_consensus` 111, `nested_fock_algebra` 98,
  `decide_core` 105, `qfm` 88, `qfm_text` 64, `unfer_edge` 63,
  `unfer_data` 56, `unfer_taler` 36, `ode_sirk` 36, `unfer_identity` 6,
  `sirk_core_model` 5. Debug and `--release` report the same count for
  every suite (verified 2026-10-08; no test is gated on
  `cfg(debug_assertions)`).
- `fock_sirk` is measured separately (401 across 56 binaries,
  2026-10-06, `--release`) and is **not** in the 1785 above.
- 6 end-to-end Austral module demos, each with a positive + UK-4001
  negative test.
- 4 criterion benchmark groups in `qfm/benches/pipeline.rs`
  (`compile_vs_M`, `generate_vs_d`, `sketch_apply_vs_d`,
  `bayes_update_vs_n`).
- GPU smoke: 14 CUDA tests green locally; the cuda feature is
  CPU-default-disabled in CI.

> Revision history of this table: **rev 27** (2026-10-08) claims **1785**
> after D7's backbone-choice work added 11 tests to `decide_core`, measured
> by the same command in both debug and `--release` rather than carried
> forward. **rev 26** (2026-10-07) claimed 1774. **rev 25** (2026-10-07) adds the
> `cell_tool` blueprint CLI (+4, `unfer_protocol`) and A9's loopback-default
> test (+1, `unfer_edge`), re-measured by the same command rather than carried
> forward. **rev 24** (2026-10-06) claimed 1764. The revision before that
> (**rev 23**, 2026-08-11) claimed **676** and
> was stale by a year of work — it predates the board/cooperation/evidence
> modules, the configuration layer and `decide_core`. It is superseded by the
> figures above, which are re-measured rather than carried forward: the status
> block at the top of this file is checked against a live log by
> `scripts/check-status --recount` in CI.

## Documentation

- [`QFM.tex`](QFM.tex) — the full algorithm spec (Sections 1–4 for the
  core flow, Section 7 for the QFM-TSR pipeline, Section 8 for the
  Bayesian update).
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — the
  per-crate implementation record, test/benchmark counts, future
  roadmap.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system diagram, cross-repo
  dependency graph, four documented extension points.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — JSON request/response schemas
  (`AgentRequest`/`AgentResponse`/`HamiltonianSpec`/`EventPredicate`).
- [`docs/MODULE_RECIPE.md`](docs/MODULE_RECIPE.md) — the manifest +
  Austral + foreign-import + JIT recipe for adding a new module.
- [`docs/BUILD_PIPELINE.md`](docs/BUILD_PIPELINE.md) — build orchestration.
- [`AGENTS.md`](AGENTS.md) — architecture-level constraints and
  load-bearing facts.

## Physics builtins

The `prob_kernel::build` layer dispatches by name. Each is a
`HamiltonianSpec::Builtin { name, params }` in the protocol:

- `harmonic_chain(n_modes, omega)` — harmonic oscillator chain.
- `navier_stokes(nu)` — Navier–Stokes BRST charge + Hamiltonian.
- `yang_mills(g)` — pure SU(3) Yang-Mills.
- `gravity()` — Einstein-Cartan gravity.
- `bose_hubbard_chain(n_modes, t, u, periodic)` — Bose-Hubbard model.
- `yang_mills_lattice(l, g, n_colors)` — Kogut–Susskind 2D lattice gauge
  theory. **Flagship model for the mass-gap demonstration** (per the
  Yang-Mills Millennium Prize path).
- `qfm_mehler(alphas)` — Mehler-prior diagonal QFM surrogate.
- `qfm_mehler_localized(points, alphas, scale)` — diagonal surrogate with
  the literal data-channel encoding (D inner modes per point).
- `qfm_mehler_projector(epsilons)` — the **exact off-diagonal generator**:
  the rank-1 projector `|0><0|` onto the dressed Mehler vacuum, applied
  via the O(M) rank-1 shortcut (no O(ε) truncation).
- `qfm_mehler_projector_localized(points, epsilons, scale)` — the exact
  generator with the literal data-channel encoding.
- `qfm_tomography(spec)` — the full QFM-TSR pipeline (training data, k,
  k2, krylov_dim, seed); its flow generator is the exact projector.

## Status

The system is in a **"v1 feature-complete + v2 algorithm-complete"**
state. Every algorithm in `QFM.tex` is implemented, tested, and
pushed. The next rounds of work are *not* new algorithm features but
**operationalization, validation, scaling, and documentation** — see
the "Next steps to improve" section in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
