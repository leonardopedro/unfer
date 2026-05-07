# Project Evolution Plan: Fock-Sirk

This document outlines the roadmap for the Fock-Sirk project. Completed phases are archived or marked, and new technical milestones are added to push the boundaries of quantum field theory simulations.

## Status: Phases 1-7 Implemented
- [x] **Phase 1**: Core Field Theory Primitives (`field_theory.rs`)
- [x] **Phase 2**: Building Specific Hamiltonians (`models.rs`)
- [x] **Phase 3**: Navier-Stokes Example and BRST implementation.
- [x] **Phase 4**: Alignment with "Quantization due to time-evolution" thesis.
- [x] **Phase 5**: Free Electromagnetic Field Benchmark.
- [x] **Phase 6**: Enforcing "Quadratic Ordering" in CAS.
- [x] **Phase 7**: Pure SU(3) Yang-Mills baseline implementation.

---

## Phase 8: Advanced Field Theory Refinement

### 8.1 Full Yang-Mills Non-Abelian Dynamics
- **Task**: Expand the simplified magnetic term in `models.rs` to include the full non-Abelian $B_{ia}$ tensor.
- **Details**: Implement the structure constants $f_{abc}$ for SU(3) and the $\epsilon_{ijk}$ levi-civita tensor to construct the $g f_{abc} A_j^b A_k^c$ interaction terms.
- **Goal**: Enable simulation of gluon-gluon scattering and vacuum polarization effects.

### 8.2 Einstein-Cartan Gravity Mapping
- **Task**: Fully implement `gravity_hamiltonian()` in `models.rs`.
- **Details**: Map the tetrad $e^\beta_a$ and polymomentum $P^{ab}$ to specific mode indices. Use the Hermitian field primitives to represent the metric fluctuations.
- **Goal**: Test the "Quantization due to time-evolution" hypothesis on 3D simplified gravity models.

---

## Phase 9: Performance & Scalability (The "Millions of States" Milestone)

### 9.1 Parallel Kernel Dispatch
- **Task**: Parallelize the Gram matrix $G_{jk} = \langle w_j | w_k \rangle$ loop in `forward_sirk.rs`.
- **Details**: Currently, the inner products are computed sequentially. Use `rayon` or `candle`'s async capabilities to dispatch multiple GPU kernels simultaneously.
- **Goal**: Reduce basis generation time by $10x$ for large Krylov dimensions $m$.

### 9.2 Symbolic Expansion Optimization
- **Task**: Optimize `nested_fock_algebra/src/cas.rs` for large Hamiltonians.
- **Details**: The current `SExpr::distribute()` is recursive and can be memory-intensive for high-order polynomials (like Yang-Mills). Implement an iterative or memoized distribution strategy.
- **Goal**: Allow compilation of Hamiltonians with $>1000$ terms in under 1 second.

---

## Phase 10: Numerical Stability & Gauge Physics

### 10.1 Automated BRST Projection
- **Task**: Integrate BRST projection directly into `solve_forward_sirk`.
- **Details**: Periodically project the state $w_k$ back into the kernel of the BRST charge $\Omega$ to prevent numerical drift into non-physical gauge sectors.
- **Goal**: Maintain strict gauge invariance over long simulation times.

### 10.2 Adaptive SIRK Shifting
- **Task**: Implement an adaptive algorithm to choose shifts $z_k$.
- **Details**: Analyze the eigenvalue distribution of the projected Hamiltonian to dynamically adjust shifts for better convergence in dissipative (Navier-Stokes) or oscillating (EM field) regimes.

---

## Phase 11: Tooling & UX

### 11.1 CUDA Environment Auto-Configuration
- **Task**: Add a build script or runtime check to automatically detect `libcublas` and set `LD_LIBRARY_PATH` or equivalent.
- **Details**: Resolve the `CUBLAS_STATUS_ARCH_MISMATCH` issues by preferring local or specific toolkit libraries.

### 11.2 Visualization Dashboard
- **Task**: Export simulation coefficients to a format (JSON/HDF5) compatible with Python visualization tools.
- **Goal**: Plot probability density evolution and spectral functions.