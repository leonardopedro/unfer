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
- **Cholesky Orthonormalization**: The projection $H_{proj} = L^{-1} H_{raw} L^{-*}$ requires a positive-definite Gram matrix. If singularity occurs, reduce $m$ or adjust shifts $z_k$.
- **Unitary Time Evolution**: **Always** use `nalgebra`'s Padé approximant `exp()` for the reduced system to preserve unitarity and Hermiticity.

### 5. GPU Optimization & Environment
- **Device Selection**: Use `Device::cuda_if_available(0)`. 
- **Library Path Note**: On systems with multiple CUDA versions, ensure `LD_LIBRARY_PATH` points to the toolkit matching the driver (e.g., `/lib/x86_64-linux-gnu` for CUDA 12.2 coexistence).
- **CUBLAS Safety**: Initialization failures (`ARCH_MISMATCH`) often indicate a version conflict between `libcublas` and the active GPU.

## Maintenance Checklist

- [ ] **Quadratic Ordering Check**: Verify that `compile_expression` continues to strip zero-point energy constants.
- [ ] **LaTeX Mapping Check**: Ensure `compile_latex` correctly interprets $a_i^\dagger$ as a creation operator and $a_i$ as annihilation. Note that the `mathhook` LALRPOP parser strictly requires explicit multiplication symbols (`*` or `\cdot`) instead of implicit spacing.
- [ ] **Commutator Validation**: Ensure non-commuting operators are never reordered by the symbolic engine (avoid `.simplify()` where order matters).
- [ ] **GPU Execution**: Run examples with `RUST_LOG=candle_core=debug` to confirm active CUDA kernel dispatch.
- [ ] **Vacuum Initialization**: Ensure `QuantumState::vacuum()` is properly initialized with at least one empty inner universe (`OuterBosonCreate(InnerBosonicState::vacuum())`) before applying inner operators.

## Core Dependencies
- `candle-core`: GPU tensor management (with `cuda` feature).
- `mathhook`: High-performance LaTeX and math parsing engine.
- `nalgebra`: High-level linear algebra for the reduced subspace.
- `quantrs2-symengine-pure`: Symbolic expression AST.

---
*Note: This project targets the Millennium Prize requirements for Yang-Mills and Navier-Stokes existence by resolving dynamics over discrete Fock-basis boundaries.*
