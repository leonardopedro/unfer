# Fock SIRK

`fock_sirk` is a Rust crate that implements a spectral solver using the Hashimoto Shift-Invert Rational Krylov method with matrix-free linear algebra. It is designed to work seamlessly with the `nested_fock_algebra` crate to efficiently evaluate complex second-quantized Hamiltonians over nested Quantum Fock spaces.

## Core Concepts: Nested Fock Spaces

The physics model relies on a strictly separated topology of __Outer__ and __Inner__ Fock spaces, allowing you to define dynamics across a "Multiverse" field mathematically.

### 1. The Outer Fock Space (The Multiverse)

The Outer Fock space defines the macroscopic coordinates (or "universes") present in your system. It is strictly partitioned into two types:
- **Outer Bosonic States (`OuterBosonic`)**: A field of universes carrying bosonic statistics.
- **Outer Fermionic States (`OuterFermionic`)**: A field of universes carrying fermionic statistics (obeying Pauli exclusion and Jordan-Wigner sign rules across the multiverse manifold).

### 2. The Inner Fock Space (Intra-Universe Excitations)

Each coordinate in the Outer Fock space contains an internal configuration, represented by the Inner Fock Space. 
- **Inner Bosonic Modes**: The specific harmonic oscillator excitation levels at a coordinate.
- **Inner Fermionic Modes**: The local fermionic occupation bits at a coordinate.

### How they interact (The Integral)

In standard quantum field theory, a Hamiltonian is often an integral over a field of coordinates. In `fock_sirk`, **inner operators intrinsically act as operators integrated over the outer field.** 

When you apply an inner operator (e.g., $a^\dagger_i$), the solver dynamically iterates through all corresponding outer multiverses, applies the local inner-operator logic to that coordinate's state, and shifts the configuration of the outer field automatically.

## Defining the Hamiltonian

Hamiltonians can be defined using either standard symbolic CAS strings or directly from **LaTeX** math expressions.

### LaTeX Hamiltonian Definition (Recommended)

The `compile_latex` function allows you to use standard physics notation. The translator automatically maps daggers to creation operators and plain symbols to annihilation operators.

```rust
use nested_fock_algebra::compile_latex;

// H = \frac{1}{2} (a_0^\dagger a_1 + a_1^\dagger a_0)
let h_latex = r"\frac{1}{2} (a_0^\dagger a_1 + a_1^\dagger a_0)";
let hamiltonian = compile_latex(h_latex);
```

### Variable Naming Convention

| Operator Type / Space | Creation Variable | Annihilation Variable |
| :--- | :--- | :--- |
| **Inner Bosonic** | `c_[mode]` (e.g., `c_0`) | `a_[mode]` (e.g., `a_0`) |
| **Inner Fermionic** | `c_f[mode]` (e.g., `c_f1`) | `a_f[mode]` (e.g., `a_f1`) |
| **Outer Bosonic** | `C_[coord]` (e.g., `C_0`) | `A_[coord]` (e.g., `A_0`) |
| **Outer Fermionic** | `C_f[coord]` (e.g., `C_f0`) | `A_f[coord]` (e.g., `A_f0`) |

*(Note: when using `compile_latex`, you can use `a_i^\dagger` which maps to `c_i` automatically)*

## Usage Example

To simulate a dynamic system, define your initial state, construct the Hamiltonian, and run `build_hashimoto_subspace`.

```rust
use fock_sirk::build_hashimoto_subspace;
use nested_fock_algebra::{QuantumState, compile_latex, InnerBosonicState, InnerFermionicState, Operator};

// 1. Define Physics via LaTeX
let h_latex = "a_0^\dagger a_0 + C_{f0}^\dagger A_{f0}";
let hamiltonian = compile_latex(h_latex);

// 2. Define Initial State
// Vacuum + 1 Bosonic universe + 1 Fermionic universe
let initial_state = QuantumState::vacuum()
    .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
    .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()));

// 3. Run Hashimoto SIRK Solver
let m_dim = 10; // Krylov dimension
let spectral_shift = 100.0;
let time_step = 2.0;

let sirk_result = build_hashimoto_subspace(
    &hamiltonian,
    initial_state,
    m_dim,
    spectral_shift,
    time_step
);

println!("Krylov subspace built. Reduced matrix size: {}x{}", 
    sirk_result.h_matrix.nrows(), sirk_result.h_matrix.ncols());

// 4. Time Evolve
let evolved_state = sirk_result.time_evolve(time_step);
println!("Evolved state components: {}", evolved_state.components.len());
```

