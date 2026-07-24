# ODE→Hamiltonian Singularity Detection (`ode_sirk`)

> Transforms multivariate polynomial ODEs into self-adjoint quantum
> Hamiltonians, detects essential self-adjointness (ESA) via Nelson's
> flow-completeness criterion, and localizes finite-time singularities.

## Pipeline

```
ODE strings → parse → flow analysis → singularity sweep →
              change of variables → Weyl Hamiltonian → ESA report
```

Entry point: `ode_sirk::analyze_ode_system(vars, rhs, cov, samples, t_max)`
returns `(OdeReport, nested_fock_algebra::Hamiltonian)`.

| Stage | Module | Function |
|-------|--------|----------|
| 1. Parse | `ode.rs` | `ODESystem::parse(vars, rhs_strs)` via `quantrs2-symengine-pure` CAS |
| 2. Flow analysis | `flow.rs` | `analyze_classical_flow(sys, samples, t_max)` |
| 3. Singularity sweep | `singularity.rs` | `sweep_singularity_1d(poly, x0s, t_max)` |
| 4. Change of variables | `change_of_vars.rs` | `apply_cov(sys, cov)` |
| 5. Hamiltonian | `hamiltonian.rs` | `ode_to_hamiltonian(sys)` |
| 6. ESA report | `esa.rs` | `build_esa_report(flow, singularity, cov)` |

## Key Types

### `ODESystem`

```rust
pub struct ODESystem {
    pub vars: Vec<String>,      // ["x", "y"]
    pub rhs: Vec<Polynomial>,   // dx_i/dt = rhs[i](x)
}
```

Polynomial-only RHS (no sin/exp). CAS parsing via `quantrs2-symengine-pure`
S-expression decoder.

### `NormalOrderedOp`

```rust
pub struct NormalOrderedOp {
    pub terms: FxHashMap<Vec<(u32, u32)>, Complex64>,
    //              per-mode (creation_count, annihilation_count)
}
```

Wick recursion without string expansion. Multiplication by `x_i` or `p_i`
applies `[a, a†] = 1` recursively, merging like terms. Degree bound: ≤20
(rejects `PolynomialTooLarge`). Prunes coefficients below 1e-15.

### `FlowAnalysis` / `EscapeEvent`

```rust
pub struct FlowAnalysis {
    pub is_complete: bool,
    pub escapes: Vec<EscapeEvent>,
}
pub struct EscapeEvent {
    pub initial: Vec<f64>,
    pub t_blowup: f64,
    pub divergent_axes: Vec<usize>,
}
```

Adaptive Euler with `R_MAX = 1e6`, `Δt_min = 1e-14`. DOP853 in the
production pipeline.

### `SingularityType`

```rust
pub enum SingularityType {
    FiniteTimeBlowUp,  // ‖x‖ → ∞ in finite time
    BoundaryHit,       // flow reaches coordinate singularity (e.g. y=0)
    GradientBlowUp,    // Δt → 0 from large gradients
}
```

### `EsaStatus`

```rust
pub enum EsaStatus {
    EssentiallySelfAdjoint,      // flow complete → ESA (Nelson)
    NotEssentiallySelfAdjoint,   // flow incomplete → not ESA
    SingularityResolved,         // singularity detected but resolved by CoV
}
```

### `OdeReport`

```rust
pub struct OdeReport {
    pub vars: Vec<String>,
    pub esa: EsaReport,
    pub cov: Option<CoV>,
    pub diagnostics: Vec<u32>,  // UK-2101..2105
}
```

## Weyl Quantization

Given `dx_i/dt = f_i(x)`:

```
H = Σ_i [ f_i(x̂) p̂_i − (i/2) ∂f_i/∂x_i ]
```

Bosonic mapping: `x_i = (a_i + a_i†)/√2`, `p_i = −i(a_i − a_i†)/√2`.

Algorithm (`ode_to_hamiltonian`):
1. Build normal-ordered `f_i(x̂)` via Wick recursion.
2. Right-multiply by `p̂_i` (maintain normal order).
3. Subtract Weyl correction `−(i/2) ∂_i f_i`.
4. Map to `nested_fock_algebra::Hamiltonian`.

## Nelson's Theorem

`D = i(v·∇ + ½ div v)` is ESA on `C_c^∞(ℝ^M)` **iff** the classical flow
is complete. Incomplete flow → probability leakage → non-zero deficiency
indices.

## Blow-up Time (1D Quadrature)

```
T(x₀) = ∫_{x₀}^{∞} dx / f(x)
```

Trapezoidal rule, 10⁵ steps, adaptive upper bound. For `f(x) = x²`:
`T(x₀) = 1/x₀`.

## Coordinate Transformations

| CoV | Map | Transformed flow | Use case |
|-----|-----|-----------------|----------|
| `Reciprocal(k)` | `w = 1/x_k` | `ẇ = −w² f(1/w)` | Blow-up at x→∞ |
| `Logarithmic(k)` | `w = ln(x_k)` | `ẇ = f(e^w)/e^w` | Linear blow-up |

Observable mapping: `⟨x⟩ = ⟨φ⁻¹(ŵ)⟩` (stored as closures in
`TransformedSystem::observable_maps`).

## UK Diagnostic Codes

| Code | Name | Trigger |
|------|------|---------|
| UK-2101 | OdeNotEssentiallySelfAdjoint | Flow incomplete, no CoV |
| UK-2102 | OdeSingularityDetected | T(x₀) < t_max for sampled x₀ |
| UK-2103 | OdeCovApplied | CoV stabilized the Hamiltonian |
| UK-2104 | OdeDeficiencyIndices | 1D reduced flow: (n₊,n₋) ≠ (0,0) |
| UK-2105 | OdePolynomialTooLarge | Normal-ordered degree > 20 |

## Integration

- **Upstream**: `nested_fock_algebra` (Hamiltonian type, Operator enum).
- **Downstream**: `fock_sirk` SIRK solver (via shared `Hamiltonian` type),
  `prob_kernel::Session::analyze_self_adjointness()`.
- **Protocol**: `unfer_protocol::HamiltonianSpec::OdeSystem`.

## Validation Cases

| Case | ODE | ESA? | Singularity |
|------|-----|------|-------------|
| x2_scalar | ẋ = x² | No | T = 1/x₀, CoV resolves |
| coupled_xy | ẋ = y, ẏ = 2xy | No | UK-2101 |
| py2 | p_x y + p_z p_y y² | No | UK-2104 (k_z ≠ 0) |
| punctured | y⁻¹ p_x + p_z p_y | No | UK-2102, deficiency (1,1) |
| stable_linear | ẋ = −x | Yes | None |

## Tests

34 unit tests across 8 modules: CAS parsing (7), flow analysis (3),
singularity sweep (4), CoV (3), Hamiltonian construction (4), Wick
algebra (7), ESA report (3), full pipeline (3).

## Formal Verification

Lean 4 formalization in `timepiece/Singularity/` (9 files):
`Poly.lean`, `OdeSystem.lean`, `Hamiltonian.lean`, `Flow.lean`,
`Singularity.lean`, `ChangeOfVars.lean`, `Esa.lean`, `Report.lean`,
`Tests.lean`. Key theorems: `blowupTime_x_sq` (rfl),
`esa_nelson_essential_self_adjoint` (simp),
`weyl_symmetrization_self_adjoint` (trivial).

See `ODE.tex` for the full mathematical treatment.
