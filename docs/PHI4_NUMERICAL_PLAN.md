# Real scalar λφ⁴ — numerical research and validation plan

**First experiment.** Evolve a single outer excitation for the continuous-momentum
λφ⁴ Hamiltonian `h_cont(k, φ) = ½ p_φ² + ½(k₁² + k₂² + k₃²) φ² + (λ/4)(φ² − v²)²`
on `L²(ℝ³_k × ℝ_φ)` (`book.tex` SIRK / Mehler / nested-Fock framework). Momentum
`k ∈ ℝ³` and field `φ ∈ ℝ` are **continuous** (never a momentum sum or spatial
lattice); no spatial-derivative jet constraints are needed because the quadratic
gradient is diagonal multiplication by `|k|²`. **Only the Krylov subspace and
Mehler-Hermite occupation window in SIRK are discrete.** Resolve the effect of
changing `λ` and the continuous momentum wave-packet spread `width_k` on the
normalized Gaussian return probability by enclosing the full-action error for
each trajectory.
[Master cheap-LLM plan](../../timepiece/PHI4_IMPROVEMENT_PLAN.md) ·
[Operator and Lean targets](../../timepiece/PHI4_FORMALIZATION_PLAN.md) ·
[Cadabra2 derivation and exact checks](PHI4_ANALYTIC_PLAN.md)

## Reuse and baseline

- Use the bosonic occupations of `JointInnerBasis`, the same-mode coordinate
  and momentum in `InnerExpr`, and its `d_gamma` action. One outer particle
  carries all four continuous variables. Apply the scalar momentum twice on the same
  mode, multiply by the three jet squares, and retain the whole wall inside h. [Ordered action](../nested_fock_algebra/src/exact_inner.rs)
- Use the existing joint shifted solves, whitening, full images and reduced
  matrix exponential. The anharmonic example uses a positive quadratic well
  and forward products. The legacy SM Higgs example uses independent momentum
  and field modes. Keep both as named comparisons.
  [Rational solver](../fock_sirk/src/joint_sirk.rs) ·
  [Oscillator](../fock_sirk/examples/anharmonic_oscillator.rs) ·
  [Higgs comparison](../fock_sirk/tests/sm_higgs_mechanism_sirk.rs)
- Reuse basis enumeration and root-scan diagnostics from the SM N12 scalar pilot.
  It used independent mode quartics and products of compressed coordinate matrices.
  At five modes, total-occupation cutoffs 3, 4 and 5 gave one, two and one matching
  roots. This establishes a concrete failure mode for fitting a connected four-point
  function. The local mixed-mode interaction needs its own evidence.
  [Pilot](../fock_sirk/examples/sm_finite_mode.rs) · [Reported data](SM_RESEARCH_PLAN.md)

## Experiment sequence (aligned with `P11`–`P13`)

1. **Implement the continuous-momentum operator (`P11`).**
   In `nested_fock_algebra/src/phi4_momentum.rs`, build `ScalarMomCont` on the
   four continuous coordinates `(φ, k₁, k₂, k₃)` using `InnerExpr::{coordinate, momentum}`
   with `½ p_φ² + ½(k₁² + k₂² + k₃²) φ² + (λ/4)(φ² − v²)²` (retaining `λv⁴/4` and
   parametrizing continuous Mehler widths `width_phi`, `width_k`). At unit
   parameters and widths, match all 11×3 `S₃`-symmetric entries of
   `docs/fixtures/phi4_momentum_mehler.json` (mean `17/8`, full variance `553/32`,
   outside-3-mode-window squared `9/2`, nonzero `He₂(φ) K₂` coupling `1/2`) to
   `< 1e-12`, and verify `dΓ(h)Ω = 0`, outer shifts `0, c, 2c`, and `dΓ(B)² ≠ dΓ(B²)`.
2. **Match the exact rank-2 Hashimoto-SIRK reference (`P12`, test 1).**
   On the degree-`≤ 2` even Hermite window (`cutoff = 2`) of `ScalarMomCont` at
   unit parameters, run `solve_joint_sirk` (`JointKrylovMethod::ShiftInvert`)
   with shifts `[10i, 5i]` at `t = 0.01`, verify `p(0.01) ∈ [319447/320000, 39999/40000]`,
   and confirm the full unprojected tail norm squared equals `9/2`.
3. **Run independent Hermite-cutoff, Krylov-rank, coupling, and momentum-spread sweeps (`P12`, tests 2–3).**
   Prepare one outer particle in the continuous unit-width Gaussian `ψ`, observable
   `O = |ψ⟩⟨ψ|`. Sweep Hermite cutoff `C ∈ {4, 8, 12}` at fixed Krylov rank `m = 2`,
   then `m ∈ {2, 4, 6}` at fixed `C = 8`. Run paired cells `λ = 1.0` vs `1.1`
   (plus null control `1.0` vs `1.0`) and continuous momentum wave-packet spread
   `width_k = 1.0` vs `1.5` at `v = 1`, `t = 0.05` using identical windows and
   `hashimoto_unitary_shifts`.
4. **Bound the computed evolution with full-action defects (`P12`–`P13`).**
   Use `fock_sirk::sm_defect::{defect_terms, interval_terms}` with full unprojected
   images `HV`, charge `2ε + ε²` per probability estimate, and require the paired
   response intervals to exclude zero while the null control contains zero.
   Check even/odd `ℤ₂` parity conservation (`φ ↦ −φ`) and verify every Ritz
   eigenvalue obeys the unconditional inner gap `θ₀ ≥ 7/64` (recording the
   odd-even splitting as an inner excitation splitting, never as the outer vacuum gap).
5. **Compare against frozen benchmarks (`P12`, test 5).**
   Contrast continuous `ScalarMomCont` (mean `17/8`, variance `553/32`, nonzero
   `He₂(φ) K₂` coupling), auxiliary static-jet `jetModel` (mean `17/8`, variance
   `145/32`, zero coupling), and frozen 2-site lattice `ScalarGrid::two_site`
   (mean `13/4`, variance `337/16`).

## Existing lattice comparison (frozen)

`ScalarGrid::periodic` and `two_site_mode_hamiltonian` implement a separate
periodic-lattice comparison operator; do not extend it further. Before starting
the primary `phi4_momentum` suite, fix the unweighted Hermite-tail norm bug in
`two_site_return_lies_in_analytic_interval` (`P0` in
`../../timepiece/PHI4_IMPROVEMENT_PLAN.md`). All new Rust code lives in
`nested_fock_algebra/src/phi4_momentum.rs`, `fock_sirk/tests/phi4_momentum_sirk.rs`,
and `fock_sirk/examples/phi4_momentum_sirk.rs` (`P11`–`P13`).
[Comparison builder](../nested_fock_algebra/src/phi4_scalar.rs) ·
[Exact comparison fixture](fixtures/phi4_mehler.json)

## Evidence to retain

For each cell, store the source revision, operator/fixture digest, model and energy
convention, seed and observable, Mehler width, inner and outer occupations, basis
cutoff, requested and attained ranks, shifts, residuals, error components, time,
memory and failure reason. Publish the dense reference, projected rational error
and full-action estimate as different quantities. Compare efficiency at equal
bounded-observable error. Mark measurements, conditional bounds and kernel-checked
conclusions separately.
