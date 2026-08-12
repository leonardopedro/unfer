# nested_fock_algebra

Pure-Rust symbolic CAS + algebra for nested Fock spaces.

Implements the level-1 inner Fock space (bosonic modes with creation and
annihilation operators), the outer bosonic layer for multi-universe
superpositions, and a symbolic CAS compiler (`cas`) that maps
field-theoretic Hamiltonians (φφ + ππ, Yang–Mills, Navier–Stokes) onto
sparse operator structures. LaTeX and Typst math are parsed by `latex` /
`typst_math`. Quadratic ordering strips pure scalar (zero-point) terms so
that ⟨0|H|0⟩ = 0.

State trajectories use sparse `FxHashMap` structures to absorb the
exponential branching of high-order terms without expanding O(10⁴)
polynomials.