# Agent Guidelines: Fock-Sirk Project

Welcome, Agent. This repository contains specialized tools for quantum physics simulations. Use these guidelines to maintain high performance and mathematical correctness.

## Technical Architecture

### 1. Hybrid CPU/GPU Pipeline
The project implements a split-mode architecture for "Inverse-Free" Rational Krylov simulations:
- **CPU (Symbolic)**: The forward sequence $w_k = (H - z_k I) w_{k-1}$ is generated on the CPU using `nested_fock_algebra` and sparse `FxHashMap` structures to handle exponential state growth.
- **GPU (Tensor)**: Once the subspace is generated, states are mapped to a `StateDictionary` and converted to `candle` tensors. The $O(m^2)$ Gram matrix $G_{j,k} = \langle w_j | w_k \rangle$ is computed on the GPU.

### 2. Inverse-Free SIRK
- We avoid explicit linear solves $(H-z)^{-1}$ by utilizing the forward sequence.
- **Gram Matrix Stability**: The basis orthonormalization via Cholesky decomposition of $G$ is sensitive to numerical singularity. If $G$ is not positive definite, consider adjusting shifts $z_k$ or reducing the subspace dimension $m$.

### 3. Unitary Time Evolution
- **Never** use simple power series for time evolution. 
- **Always** use `nalgebra`'s `exp()` method (Padé approximants) for $e^{-iHt}$ to preserve unitarity and Hermiticity within floating-point precision ($\sim 10^{-16}$).

### 4. Solver Performance
- **Memory Locality**: While the tensor math is on the GPU, the CPU symbolic phase is sensitive to cache performance. Avoid unnecessary allocations in `apply_to_state`.
- **Tensor Shapes**: When converting `QuantumState` to `TensorState`, ensure the `StateDictionary` is fully populated first to avoid shape mismatches during GPU operations.

## Maintenance Checklist

- [ ] **Mathematical Verification**: Verify that new operator definitions correctly distribute and commute in `nested_fock_algebra`.
- [ ] **GPU Validation**: Run examples with `RUST_LOG=candle_core=debug` to verify CUDA kernel execution.
- [ ] **Unitarity Check**: Periodically verify that $||\exp(-iHt)v|| \approx ||v||$ in new simulation cases.

## Core Dependencies
- `candle-core`: GPU tensor management and contractions.
- `nalgebra`: Linear algebra and matrix exponential.
- `egg`: E-graph library for symbolic optimization.
- `quantrs2-symengine-pure`: Low-level symbolic primitives.

---
*Note: This project is part of the Velyst ecosystem but operates as a standalone high-performance kernel.*
