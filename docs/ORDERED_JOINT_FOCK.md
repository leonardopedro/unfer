# Ordered joint inner action and outer enclosure

2026-11-09c. The algebra/action foundation is implemented in
`nested_fock_algebra/src/exact_inner.rs`; the complete SM E1 builder and its
joint Hashimoto integration are described in
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md).

## Inner action before outer enclosure

`JointInnerBasis` contains both bosonic and fermionic occupations. `InnerExpr`
is a factorized scalar/ladder/sum/product expression; products are written in
mathematical order and applied right to left on that same inner state. There
is no sorting, contraction deletion or expansion into every monomial.

`coordinate(m)=a†_m+a_m` and `momentum(m)=i(a†_m-a_m)/2` form a same-mode
canonical pair. Fermionic signs are computed in the inner occupation factor;
Pauli-forbidden transitions vanish. `adjoint` conjugates scalars and reverses
factor order. All contractions and constants are retained by ordered action.
The coefficients and square roots use floating point, so "exact" describes
the operator algebra, not interval/roundoff certification.

`JointOuterBasis` supports symmetric, antisymmetric and parity-graded sectors over
the entire joint inner label. Even inner configurations have arbitrary bosonic
outer multiplicity, while odd configurations have Pauli occupancy and Koszul
signs; mixed-parity states are supported. `InnerExpr::d_gamma` completes an inner action
before applying outer `C†_target A_source`. An inner scalar gives `c N_out`;
the outer vacuum is annihilated. The one-particle fast path applies the whole
factorized expression to a joint inner vector at once, preserving linearity.
Physical BRST reduction and the graded Hilbert/core identification remain P5 proof work.

`ActionLimits` bounds per-vector support and action work. Failures return
`ActionError`, without amplitude thresholds, top-k pruning, occupation clipping
or fallback to a smaller model. These are invocation budgets, not a whole-run
memory or arithmetic certificate.

## Legacy API boundary

Each old `Operator::Inner*` letter already acts as a one-body outer lift.
Multiplying those letters gives a product of lifts and can add cross-particle
terms absent from the lift of a completed inner product. An empty old word is
outer identity, whereas enclosing inner identity gives particle number.
The old CAS also sorts words and drops contractions/scalars.

Those APIs retain their historical comparison behavior. The full SM uses the
new joint path. The regression suite reproduces the distinctions through the
actual old API, rather than changing old spectra silently.

## Solver connection and checks

`fock_sirk/src/joint_sirk.rs` now uses these joint vectors directly. Its registry
is bijective on joint labels. The default is true sequential shift-invert
rational Krylov with measured GMRES residuals and the bounded-resolvent function
formula. Forward products are separately named diagnostics.

Declared numerical windows can cache complete operator images and the exact
finite compression. Internal shifted solves then reuse that compression while
full images remain available for tail and residual checks. Cache/resource
failures return errors. No hidden legacy `scale_and_add` threshold enters this
path. See the implementation account for the formula and error boundaries.

`nested_fock_algebra/tests/exact_inner_contract.rs` has thirteen construction
regressions: canonical CCR, retained quartic terms/padding, all 1,365 two-mode
CAR words of length zero through five against Jordan–Wigner matrices, complex
mixed action/adjoints, one-particle enclosure, inner constants, nonmultiplicativity,
symmetric/exterior two-particle sums, resource handling, actual legacy behavior, graded Pauli/multiplicity rules,
odd-spectator signs and the statistics tag on the one-particle fast path.
They support the full-SM dictionary and rational tests; they are not substitute
physical numerical models.
