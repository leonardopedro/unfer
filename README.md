# Fock-Sirk

A high-performance Rust suite for quantum mechanical and field theory simulations in Fock spaces. Fock-Sirk implements a novel **Inverse-Free Rational Krylov** architecture to solve for the dynamics of non-polynomial Hamiltonians (Navier-Stokes, Yang-Mills, Gravity) directly over probability-weighted Fock bases.

## Key Features

- **Inverse-Free Rational Krylov (SIRK)**: Generates optimal Krylov subspaces via forward evolution $w_k = (H - z_k I) w_{k-1}$, avoiding expensive $O(N^3)$ linear solvers.
- **GPU-Accelerated Gram Matrices**: Uses `candle-core` with CUDA to compute $O(m^2)$ basis inner products in parallel, enabling the use of extremely high-dimensional state vectors.
- **Field Theory Engine**: Native support for Hermitian fields, conjugate momenta, Majorana fermions, and BRST ghost sectors.
- **Quadratic Ordering CAS**: Symbolic compiler that enforces zero vacuum expectation value $\langle 0 | H | 0 \rangle = 0$ to satisfy mass-gap and Millennium Prize constraints.
- **Unitary Preservation**: Certified unitary evolution in the reduced subspace using Padé-based matrix exponentials.

## Core Projects

### [fock_sirk](./fock_sirk)
The high-performance simulation kernel.
- **Hybrid Pipeline**: CPU-based symbolic branching and GPU-based tensor contractions.
- **Flexible Device Support**: Scales from multi-core CPUs to high-end CUDA GPUs.
- **Subspace Evolution**: Real-time projection of dynamics onto energy-limited support spaces.

### [nested_fock_algebra](./nested_fock_algebra)
The symbolic algebra and physics engine.
- **AST Compiler**: Converts high-level QFT expressions into executable Fock-space operators.
- **Field Theory Library**: Pre-built models for Navier-Stokes (dissipative fluid dynamics), Pure SU(3) Yang-Mills, and Einstein-Cartan Gravity.
- **Symbolic Expansion**: Robust distribution and ordering logic for non-commuting operators.

## Getting Started

### Prerequisites
- Rust (Stable)
- CUDA Toolkit (12.x recommended)
- `libcublas` and `nvcc` installed and on the path.

### Installation & Execution
```bash
# Clone and build
cargo build --release

# Run the Free Electromagnetic Field benchmark (GPU recommended)
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu cargo run --release --example free_em_field

# Run the Navier-Stokes probability distribution solver
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu cargo run --release --example navier_stokes

# Run the SU(3) Yang-Mills vacuum fluctuation test
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu cargo run --release --example yang_mills
```

## Maintenance & Status
Currently at **Phase 7** of the implementation plan. The project successfully validates the "Quantization due to time-evolution" thesis on quadratic and non-linear benchmarks.

## License
Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
