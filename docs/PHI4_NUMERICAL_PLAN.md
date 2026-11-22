# Real scalar λφ⁴ — numerical research and validation plan

**First experiment.** Evolve a single outer excitation for the continuous scalar
and three independent jet coordinates. Use the local inner Hamiltonian
h = pₛ²/2 + (j₁² + j₂² + j₃²)/2 + λ(s²−v²)²/4. Spatial position remains
continuous. The integer labels in the code denote Mehler occupations. The number
of shifted-solve vectors determines the rational Krylov rank. Resolve the effect
of changing λ on the normalized Gaussian return probability by enclosing the
full-action error for each trajectory.
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

## Experiment sequence

1. **Implement the continuous-coordinate operator.** Assign one real mode to s and
   three to j₁, j₂, j₃. Give only s a kinetic term. At unit width and λ = v = 1,
   check the full action on the normalized product Gaussian. In the unnormalized
   Hermite polynomial convention its constant coefficient is 17/8, its field
   degree-two and degree-four coefficients are 7/8 and 1/4, and each pure degree-two
   jet coefficient is 1/2. Weight the output by the factorial Gram matrix, or
   convert the coefficients before comparing normalized occupation vectors. The
   squared residual about the mean is 145/32. Check an occupied jet input as well.
   Check inner constant shifts by 0, c and 2c in the first three outer sectors.
2. **Establish a finite reference.** At total inner occupation C apply each full
   word before projecting the output. On a small Hermite window independently
   evaluate matrix elements by Gaussian polynomial moments. Compare the projected
   evolution with the rational result and retain full images for a separate
   residual bound. For example at C = 0 the fourth power of the compressed field
   coordinate vanishes while the compression of its fourth power gives 3.
3. **Run the bounded comparison.** Prepare one outer particle in the unit-width
   Gaussian inner state ψ. Use the bounded observable O = |ψ⟩⟨ψ| on this sector.
   Sweep C = 4, 8, 12 and Krylov ranks 2 and 4 at t = 0.05 and 0.1 as initial
   pilot values. Fix λ = 1 and 1.1 in paired cells with v = 1. Use identical
   initial states, shifts and windows across each pair. For example use the common
   shift budget 7t and spacing t, validating each rank with the existing helper.
   Run λ = 1 against itself as a null control. Record the effect interval and its
   cost. A changed width requires transporting H, ψ and O together before projection.
   Record how much of the residual is carried by discarded jet occupations.
4. **Bound the evolution that was computed.** With window projector P and isometry
   V, the numerical compression uses B ≈ V†(γI + iPHP)⁻¹V on the window, and
   G = γI − B⁻¹. Reconstruct V exp(tG)V†ψ. Keep the Ritz matrix V†HV for spectral
   diagnostics. Compute D = iVG − HV using **full**, unprojected images of every
   basis column. For a self-adjoint H and an exactly isometric V, bound state error
   by initial leakage plus the time integral of ‖D exp(sG)V†ψ‖. Bound reduced
   growth or enclose this integral. Record separately the error against the finite
   reference and the norms of discarded action. The latter can remain large even
   when the former is small. Allow for basis nonorthogonality, finite precision,
   GMRES, inversion, Padé and restarts wherever their effects are not already
   covered by the returned-map defect. With a state error ε, charge each probability
   estimate 2ε+ε² and add the two budgets for a response. Require the resulting
   interval to exclude zero before interpreting its sign. Bind parameters and
   interval endpoints to a Lean statement for this scalar instance. The SM defect
   artifact supplies reusable arithmetic and a conditional Duhamel estimate.
   [Defect implementation](../fock_sirk/src/sm_defect.rs) ·
   [SM fixture](fixtures/sm_defect_bridge.json)
5. **Extend the continuous-coordinate study.** For a fixed operator and preparation,
   compare even and odd channels as the occupation window grows. A mass claim needs
   bounds on missed states and the appropriate absolute energy or channel splitting.
   For spatial observables specify the continuous position dependence, test functions
   and their matrix elements on the common core. Any identification of independent
   jets with spatial derivatives of a field needs its own proof.

## Existing lattice comparison

`ScalarGrid::periodic` and `two_site_mode_hamiltonian` implement a different
Hamiltonian built from periodic links. Its two-site columns and its three-mode
compressed evolution have Rust tests. At unit parameters and t = 0.01 its analytic
Gaussian return interval is [0.99789375, 0.999775]. The three-mode exponential
test is a finite reference. The rank-two Hashimoto generator comes from a compressed
resolvent. The site fixture can check operator-ordering and residual routines.
Its site spacing, Fourier sums and lattice return interval are specific to this example.
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
