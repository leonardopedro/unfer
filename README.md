# Fock-Sirk

A high-performance Rust suite for quantum mechanical simulations in Fock spaces, featuring GPU-accelerated solvers and symbolic algebra engines.

## Key Features

- **Inverse-Free Rational Krylov**: A novel solver architecture that generates Krylov subspaces via forward evolution, eliminating the need for explicit matrix inversions.
- **GPU Acceleration**: Leveraging `candle-core` for high-performance tensor contractions and Gram matrix calculations on the GPU.
- **Hybrid Architecture**: Symbolic state generation on the CPU combined with dense tensor operations on the GPU.
- **Unitary Preservation**: High-fidelity time evolution using `nalgebra`'s Padé-based matrix exponential.

## Projects

### [fock_sirk](./fock_sirk)
A High-performance **Forward Shift-Invert Rational Krylov (SIRK)** solver. It enables efficient computation of quantum dynamics in systems with complex operator structures by projecting them onto a small, optimal basis.

- **GPU-Powered Contractions**: Uses `candle-core` for all-to-all inner products in the Krylov basis.
- **Automatic Device Selection**: Seamlessly scales from CPU multi-threading to CUDA-enabled GPUs.
- **Matrix Exponential Evolution**: Provides certified unitary evolution in the reduced subspace.

### [nested_fock_algebra](./nested_fock_algebra)
A pure Rust high-performance physics and symbolic engine for **Nested Fock Spaces**. It provides the algebraic foundation for constructing and manipulating complex quantum operators.

- **Symbolic Manipulation**: Built on top of the `egg` e-graph library for equality saturation and optimization of quantum expressions.
- **Nested Structure Support**: Native handling of hierarchical quantum systems (Multiverses).
- **Type-Safe Physics**: Strong compile-time guarantees for physical dimensions and operator commutation rules.

## Getting Started

Ensure you have the latest Rust stable and CUDA drivers (optional but recommended) installed.

```bash
# Build the entire workspace
cargo build --release

# Run the hopping simulation
cargo run --release --example simulation

# Run the anharmonic oscillator test
cargo run --release --example anharmonic_oscillator
```

## License
Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
