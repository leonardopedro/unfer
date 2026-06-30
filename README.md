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
| FFI | 14 `uk_*` extern "C" functions, handle table, buffer protocol | `unfer_ffi` |
| QFM-TSR | Two-level hashing, offline training, 4-phase online generate | `qfm` |
| Bayesian update | HMC on the TSR-evolved prior, `|v^dag c|^2` likelihood | `qfm::bayes` |

The QFM (Quantum Flow Matching) layer is the v2 algorithm: a Mehler vacuum
prior + a Hermitian data-coupling flow Hamiltonian + a two-level hash (raw
coordinates → 1-boson Fock modes) + a Krylov reduction + a 4-phase
generate (encode → evolve → tomographic reconstruct → lossless decode).
The Bayesian update conditions the TSR-evolved prior on N new observations
in `O(N · m²)` per HMC step, with **no M dependence**. Full spec in
[`QMF.tex`](QMF.tex).

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
- Rust stable
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

## Test & benchmark counts (rev 17, 2026-06-30)

- Workspace tests: **180** green (19 fock_sirk + 26 nested_fock_algebra +
  37 prob_kernel + 30 unfer_protocol + 33 unfer_ffi + 35 qfm).
- 5 end-to-end Austral module demos, each with a positive + UK-4001
  negative test.
- 4 criterion benchmark groups in `qfm/benches/pipeline.rs`
  (`compile_vs_M`, `generate_vs_d`, `sketch_apply_vs_d`,
  `bayes_update_vs_n`).
- GPU smoke: 14 CUDA tests green locally; the cuda feature is
  CPU-default-disabled in CI.

## Documentation

- [`QMF.tex`](QMF.tex) — the full algorithm spec (Sections 1–4 for the
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
- `qfm_mehler(alphas)` — Mehler-prior QFM generator.
- `qfm_mehler_offdiag(alphas)` — Hermitian vacuum↔data coupling.
- `qfm_tomography(spec)` — the full QFM-TSR pipeline (training data, k,
  k2, krylov_dim, seed).

## Status

The system is in a **"v1 feature-complete + v2 algorithm-complete"**
state. Every algorithm in `QMF.tex` is implemented, tested, and
pushed. The next rounds of work are *not* new algorithm features but
**operationalization, validation, scaling, and documentation** — see
the "Next steps to improve" section in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.
