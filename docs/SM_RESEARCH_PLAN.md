# Full SM research and numerical evidence in nested Fock/Mehler/SIRK coordinates

## Continuous-momentum numerical work order

**2026-11-24. First deliverable:** reproduce a nonzero mixed SM matrix element between
two distinct smooth momentum packets, then use those same kernel conventions for an
error-controlled one-outer-particle evolution. The full target acts on continuous
momentum and field variables. A spatial derivative acts on its own momentum leg.
There are no derivative-jet coordinates or derivative-constraint projectors to add.
Finite Hermite occupation windows and rational Krylov subspaces approximate states and
the evolution. An integration rule, if needed, approximates a continuous matrix element.
Its resolution is independent of both cutoffs.
[Scalar benchmark](VERIFY_PHI4_MOMENTUM.md) ·
[Symbolic reference to produce](SM_HAMILTONIAN_AUDIT.md#continuous-momentum-cadabra2-work-order) ·
[Lean work order](../../timepiece/SM_ESA_COMPLETION_PLAN.md)

1. **Bind an operator and a seed.** Keep the current `sm_full` generator, signed color
   and electroweak coefficients, chiral spin frame, four Higgs components, complex
   Yukawa matrices and ordered CAR words as dictionary references. Identify continuous
   Hermite packet labels, normalization, conjugation, sector, and coupling conventions.
   Reuse the scalar packet construction and joint CCR/CAR machinery. Implement the
   transferred-momentum kernels explicitly. The three continuous momentum coordinates
   in the scalar fibre model suffice for its diagonal gradient. A polynomial in a
   single packet momentum cannot encode the SM's two-leg convolution.
   [Scalar action](../nested_fock_algebra/src/phi4_momentum.rs) ·
   [SM coefficients](../nested_fock_algebra/src/sm_full.rs)
2. **Check the spatial and interaction terms independently.** Recover the free Dirac
   symbol and its chiral signs, the transverse free curl energy and the scalar-gradient
   normalization in the uncoupled Higgs sector. Then use distinct packet centers and
   polarizations with noncommuting colors for a magnetic cross term, a charged
   Higgs packet for gauge–Higgs mixing, and occupied chiral matter for a gauge or
   Yukawa transition. Compare integrated entries, adjoints and full column norms with
   the independent Cadabra packet reference. Check the abelian and zero-coupling limits,
   the complete Higgs constant, and shifts $0,c,2c$ on zero, one and two outer quanta.
   Apply ordered products before the final projection. Measure or bound the omitted
   image, including continuous tails outside any integration region.
3. **Run genuine shift-invert evolution.** Assemble the complete inner operator,
   enclose it once, and use `joint_sirk` on a normalized occupied packet. With an
   orthonormal rational basis $V$ in Hermite window $P$, use
   $B=V^\dagger(\gamma I+iPhP)^{-1}V$, $G=\gamma I-B^{-1}$ on the one-particle sector.
   Check a small window against an independent dense evolution. Vary the occupation
   cutoff at fixed rank, rank at fixed cutoff, then packet width at fixed physical
   preparation (or label a change of preparation). Take the Gaussian return projector
   as a first bounded observable and use a coupling variation plus an identical-run
   control. Gauge-invariant physical predictions additionally need a physical state
   and an observable descending to the gauge quotient.
   [Rational solver](../fock_sirk/src/joint_sirk.rs)
4. **Report what the run resolves.** Save full images $hV$, initial leakage, the
   in-window and discarded parts of $iVG-hV$, integration and arithmetic error,
   Gram conditioning, solve tolerances, shift values and the returned exponential.
   A sampled maximum of the trajectory defect is a diagnostic. A certified interval
   requires a bound between samples or an analytic envelope, plus bounds for kernel
   integration and the numerical map. For two coupling values give the interval for
   their difference and report its sign only if zero is excluded. Bind each result to
   the kernel fixture, packet and operator version. A finite-window match is useful
   evidence for the specified packet. Removing smearing or proving self-adjointness
   needs independent analytic estimates. [Defect implementation](../fock_sirk/src/sm_defect.rs)

## Independent-coordinate E1 programme and dated evidence

The N6–N14 records below describe the 163-coordinate comparison or their specifically
named pilots. Its spatial indices include 120 independent jet coordinates. Its
homogeneous gauge/ghost probes remain valuable for their stated representations.
The November 24 N11 comparison reports a change near −3.38×10⁻⁶ against a sampled
two-run budget of about 0.0232. It leaves the response unresolved even for that model.
The recorded scalar run and E1 run supply different checks of reusable solver code.
A full SM continuous-kernel run requires the packet and column checks above.
[Execution ledger](SM_FULL_IMPLEMENTATION.md)

E1 work order of 2026-11-15. This began as a source-review plan. On 2026-11-15,
N6, N7, N8 and the default `run-sm-full-validation` sweep were actually executed on
`unfer` `d5123ae` + this dirty tree; see the dated records in
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md). The planning caveat still
applies to N9–N14 and to every proof claim. The proof counterpart
is [timepiece/SM_ESA_COMPLETION_PLAN.md](../../timepiece/SM_ESA_COMPLETION_PLAN.md),
**SM0–SM9**. It supersedes old N0–N5 forward instructions while preserving their implemented
features and historical evidence. New numerical/research packets are **N6–N14** below.

The aim is useful evidence about the **complete specified SM operator**, its numerical
approximation and its physical representation. Distinguish unreduced E1, physical E1 and
the infinite-inner-mode/field endpoint E2. Each result must name which endpoint it addresses.
A dense finite Hermitian matrix is not an unbounded ESA theorem; a unitary auxiliary flow
is not a positive-energy physical theory; a regulator study is not a continuum proof.

## Physical and algorithm contract

```text
h_SM = h_B + spatial Dirac + dynamical gauge-matter + all four Higgs/Yukawa components,
H_SM = dGamma_out(h_SM) = sum_ij <e_i,h_SM e_j> Cdag_i A_j.
```

Enclose the entire interacting inner operator exactly once, with creation left and
annihilation right. Preserve scalar/CCR/CAR contractions and the Higgs/realification
constants. An inner shift changes `H_SM` by `c N_out`. The outer vacuum is annihilated;
a ground-state claim additionally needs the relevant inner lower bound. Do not impose
a universal positive SM gap as an existence criterion.

E1 has 163 independent bosonic coordinates, 40 field momenta, 49 bosonic forms and 96
independent complex matter CAR coordinates. The full comparison has 43 kinetic directions
including spatial momenta; it adds no jet kinetic derivatives. Retain three right-handed
neutrinos, no Majorana mass and no duplicate independent antiparticle modes in this record.
Use `q=a+a†`, `p=i(a†-a)/2`. Its spatial derivative holds the independent collective/jet
coordinates fixed. NS/QG derivative-variable elimination through momentum convolution is
a separate construction and does not redefine this SM endpoint.

The implemented solver constructs resolvent vectors by GMRES and evolves using
`B=V*(gamma I+iH)^(-1)V`, `G=gamma I-B^(-1)`, `V exp(tG)V*psi` in the exact idealization.
With a window P distinguish full `B_full`, exact projected `B_P`, and computed `B_tilde`.
`V*HV` is a Ritz diagnostic, not the generator used for this rational function method.
Forward products remain a labeled comparison, not the same rational space.

## Existing assets and evidence to reuse

Static source baseline: `unfer d5123ae`, `timepiece fd8b190`, together with the current
SM0–SM9 documentation. Every executed result must record its own revisions/dirty diff.

- [exact_inner.rs](../nested_fock_algebra/src/exact_inner.rs): ordered joint CCR/CAR action,
  inner/outer distinction, graded occupations, explicit projections and fail-on-budget
  behavior. “Exact” means ordering/action semantics; coefficients are `Complex64`.
- [sm_full.rs](../nested_fock_algebra/src/sm_full.rs): all bosonic, spatial Dirac,
  gauge-matter and Yukawa components, named coordinates, matter charges and native dictionary.
  The independent [Cadabra dictionary](sm_full_nested.cdb) exports 1241 expressions,
  including 1152 matter entries. The mathhook bridge checks/uses the imported expressions.
- [joint_sirk.rs](../fock_sirk/src/joint_sirk.rs): real shifted solves, registry bijection,
  Gram whitening, physical-vector reorthogonalization, full images/residuals/tails,
  bounded-resolvent function generator and restart diagnostics. Reuse these APIs.
- [certificate.rs](../fock_sirk/src/certificate.rs): existing residual/roundoff/gap data
  surfaces are candidates for reuse, not an automatic joint-flow certificate. Its
  forward-solver residual recurrence and eigenvalue-ordering hypotheses do not transfer
  to joint GMRES or a full-SM physical gap merely by changing a label.
- [sirk_core_model](../sirk_core_model/README.md) and BookProof/Sirk provide translation
  work and exact-arithmetic theory. The documented floating-point placeholders and
  missing joint-action/GMRES/Padé correspondence remain explicit. Standard-axiom theorems
  consuming certified inputs do not establish that Rust has computed those inputs soundly.
- The Lean 49-form derivative table now proves `PotSelfBound` and kinetic-control
  consequences. The open work concerns the actual full operator/coordinate adapter,
  43-direction comparison graph core, required matrix form commutator, physical reduction
  and the solver bridge. Do not spend this campaign re-proving the old scalar self-bound.

Read the dated ledger in [SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md) and the
scope audit in [SM_HAMILTONIAN_AUDIT.md](SM_HAMILTONIAN_AUDIT.md). Historical sweeps report
37 and later 41 passing tests. Source currently declares 13 exact-action, 7 full-dictionary
and 13 joint-SM tests, plus the supporting gauge-fixed suite. Counts are not fresh passes.

The recorded `4.17940e-7` error at `t=.01,m=4` is relative to one 70-state finite reference.
The `m=2` row also changes the shift budget, and the `m=5` row is less accurate; these are
not a monotonic rank theorem. First-image tails on 70/230/638/1176-state windows are about
`9.78477/4.03868/1.21796/0`. The final zero does not make the window invariant. Ehrenfest
uses a different seed/window sequence of 72/325/936 states; do not combine the two sequences
into one cutoff curve. Later-action/full-flow errors remain separate from first-image tails.

## N6 — make evidence reproducible and claims machine-reviewable

*Executed 2026-11-15:* the driver now emits a verified `result_manifest`, and
`scripts/run-sm-baseline` records one replayable cell plus a declared-test
inventory. Evidence and the fresh `measured` row are in
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n6-execution-record-2026-11-15-this-working-tree).
The N6 run alone did not execute N7–N14; N7 and N8 are recorded below.

Extend the existing example/report surfaces rather than create another operator builder.
The current [JSON driver](../fock_sirk/examples/sm_full_hashimoto.rs) already exports window
identity/dimension, shifts, full-action tails, solve/function bounds and orthogonality
information. Add proposed fields only after specifying their meaning and provenance.

A result manifest must bind:

- source revisions and dirty-source hashes; target endpoint; complete native/fixture/operator
  identities; realification, named-coordinate map, couplings and all complex Yukawa entries;
- physical state/observable/sector identity, ghost convention if any, and whether the
  reported vector is inner one-particle, outer graded, or a physical quotient representative;
- basis widths/displacements and their unitary dictionary; explicit occupation/basis window,
  deterministic label hash, spatial/matter cutoffs and boundary conditions where applicable;
- requested/actual rank, every shift and shift-generation time scale, GMRES tolerance and
  cap, Gram cutoff, retained seed norm, precision, Padé/restart conventions and resource caps;
- test names actually executed, features, engine/skip status, backend/hardware, elapsed time,
  peak memory and artifact hashes; a budget refusal is a result, not a successful run;
- each error component and whether it is a measured estimate, conditional bound or verified
  enclosure. Include the exact ideal theorem and all still-unproved assumptions.

Keep `historical-pass`, `source-present`, `measured`, `conditionally-bounded`,
`certified` and `proposed` distinct. Repeated deterministic runs are reproducibility checks,
not independent statistical samples; numerical error bars need analytic/rounding bounds.

**Acceptance:** a replayable baseline cell and an honest inventory of the current suites,
with no historical result silently promoted to a newly executed pass.

## N7 — close the native/Mehler/Lean operator dictionary

*Executed 2026-11-15:* `docs/fixtures/sm_named_dictionary.json` and
`fock_sirk/tests/sm_named_dictionary.rs` bind the named coordinates, four Higgs
slots, kinetic blocks, gauge data, all 49 forms, all 3366 matter terms and seven
sector/spectator action cells against both the native builder and the Cadabra
import. Formal bridges and their remaining gaps are listed in the fixture. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n7-execution-record-2026-11-15-this-working-tree).

Coordinate with SM0/SM2. Export a small exact or enclosed coefficient/action fixture that
uses named coordinates rather than Lean's noncomputable internal enumeration. Bind the
actual four Higgs slots, all spatial kinetic spin blocks, gauge generators, every Yukawa
component and the retained trace/Higgs constants. Reuse existing component and Majorana
checks. Probe sparse states from several matter-number, color, generation, chirality and
outer-parity sectors; the 70-state anchor is not closed under all gauge rotations.

Include two-universe mixed graded states to check the spectator signs and one inner
product followed by one outer lift. Compare `dGamma(BC)` and `dGamma(B)dGamma(C)` as a
negative control, and verify inner identity versus outer number. Link each fixture to a
specific proposed Lean action/refinement theorem; a string identifier is not that proof.

Keep a count of native terms and a componentwise independent derivation. Hard failures
for omitted/mutated components, noncommuting coefficient imports, rank/resource limits
and lost contractions must remain visible. Re-evaluate fixtures only after a declared
operator/basis change; matching two copies of one builder is a weak independence check.

**Acceptance:** one full dictionary with action fixtures and explicit remaining formal
bridges. Fixed-background Higgs, scalar radial, free-field and six-mode models stay controls.

## N8 — certify full-action evolution error for the implemented generator

*Executed 2026-11-15, bounds renamed 2026-11-24:* `fock_sirk/src/sm_defect.rs`
implements the `D=iVG-HV` bound on the returned solve and the cutoff-1 cell
reports `leakage=2.41e-8`, `||D||_F=17.19`, `evolution_bound=0.0978` (sampled
defect action) and `frobenius_product_bound=0.1719` (the uniform-column product,
which is the `bound=0.1719` this line recorded before the split) at `t=.01`,
labeled `conditionally-bounded`. A 4x4 endpoint fixture reports the direct map error and
the reader rejects invalid/mismatched certificates. Its arithmetic endpoints now
use directed `f64` rounding (`next_down`/`next_up`), `HV` and `G=V*HV` are
enclosed from the declared `H` and `V`, and `lambda_max(Re G)` is bounded by
directed Gershgorin row sums on the enclosed Hermitian part. The
Duhamel representation and integral bound are kernel-verified in
`SmDefectAnalytic.lean` (Mathlib): for `u' = -A u + b`,
`u t = exp(-tA)u0 + ∫ exp(-(t-s)A)b s`, and the N8-shaped bound follows from an
initial bound and a source bound. The flow bound `||exp(-(t-s)A)|| <= 1` is
proved for every skew-adjoint `A` via unitarity. `SmDefectSkew.lean` proves
`A = iH` skew-adjoint for the declared finite fixture (its exact rational `H`
is Hermitian), and its exported payload
`docs/fixtures/sm_defect_skew_proof.ndjson.gz` is replayed by nanoda (7,540
declarations). `docs/fixtures/sm_defect_bridge.json` binds that result under
`matrix_skew_proof` as a kernel-verified Lean theorem, and its
`fixture_provenance` test parses the exact rational `H` cells from the Lean
source and compares them entrywise with `finite_fixture_inputs`; identifying the
physical reduction remains open, so `evolution_bound.hi` stays
`conditionally-bounded`. The separately measured Padé flow is a diagnostic
witness. The 2026-11-16 bridge
`docs/fixtures/sm_defect_bridge.json` binds the actual fixture, certificate and
measured error, and records `rounding.directed: true` plus the kernel-verified
conditional analytic payload. The separate
`BookProof.Sm.ChapterSmDefect.finite_map_endpoint` theorem is kernel-verified by
nanoda but checks only the finite endpoint arithmetic. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n8-finite-map-defect-bridge-is-conditionally-bounded).

SM6 proposes an immediately useful finite-map theorem. With isometric V, reduced G,
columns `HV` in the domain of the identified self-adjoint H, set `D=iVG-HV`. Then

```text
||exp(-itH)psi - V exp(tG)V*psi||
  <= ||psi-VV*psi|| + integral_0^t ||D exp(sG)V*psi|| ds.
```

For a contractive exact reduced flow this is at most initial leakage plus
`t ||D|| ||V*psi||`. Bound `||D||` by full-column norms or a rigorously enclosed defect
Gram matrix. This finite-rank-to-full-space map can use the full sparse images already
stored by `joint_sirk`, rather than a recurrence that ignores outside-window components.
A small bound is conditional on the physical/unreduced self-adjoint realization actually
identified by the theorem; label which one. Do not call it an unconditional physical-SM
certificate while SM3/SM5 remain open.

For exact compressed resolvents `Re B >= gamma B*B` implies dissipativity of G; prove the
finite inverse/contractivity hypotheses. For the returned inexact G estimate
`Re<x,Gx> <= omega ||x||²`, enclose `exp(omega t)` and propagate errors accordingly.
No exact unitarity of a nonreducing compression is presumed. Add Gram non-isometry,
initial projection, shifted solves, B inversion, exponential approximation, restart and
coefficient/action rounding as separate terms without double-counting defects already
included in D.

Reuse the existing interval/certificate interfaces only when their arithmetic and model
assumptions match. A formula such as `rank³ * unit_roundoff * matrix_norm`, a few extra
ulps or an unverified GMRES residual is not itself a proof covering this entire pipeline.
An SM joint certificate needs named rounding and matrix-function hypotheses discharged
by verified bounds. The certificate reader must reject missing, negative or nonfinite
budgets and mismatched operator/state/basis hashes.

**Acceptance:** one independently checked rational/interval finite fixture, then a scoped
full-action bound on an actual complete-SM cell. Keep the measured finite-matrix error,
full-H conditional bound and proven rounding enclosure in different fields.

## N9 — bounded cutoff and Mehler-basis research

*Executed 2026-11-15 (scoped):* `scripts/run-sm-n9-pilot` measured the four cells
(70/230 states x rank 2/4) under one shift rule and declared caps. Energy is
stable; the first-image tail follows the window, not rank; the N8 defect bound is
not monotone in rank. A separate scoped Mehler feasibility cell (`sm_mehler_widths`)
measured exact squeeze tails: only one mode at `ell=1.25` reaches `1e-8` at
`C=12`; the 40/163-mode tails at `C=12` are `1.76e-4`/`1.29e-1`. Follow-up
SIRK cells (`sm_mehler_sirk`) transformed both H and the wide squeezed state.
The one-mode cell verified energy to `7.8e-9` and measured window `70 -> 76`,
N8 defect `1.3987e-1 -> 1.3969e-1`, elapsed `3.78 -> 4.15 s`. The two-mode
cell (`g(0,0)`, `g(1,1)`, product tail `6.03e-10`) measured window `70 -> 97`,
defect `1.3987e-1 -> 1.3963e-1`, elapsed `3.78 -> 6.14 s`: the cost gap persists
and widens. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n9-bounded-pilot-2026-11-15-four-cells).

Begin with the complete operator and existing seed. Reuse the old 70/230-state references;
add a declared 638-state cell only if it fits the published caps. Reproduce the 936-state
Ehrenfest case through the existing wide tier, not as an implicit default. Retain the
1176-state first-image-closure diagnostic without calling it an invariant window.

The first new controlled pilot is at most **eight solver configurations**: cross the two
70/230 baseline windows with ranks 2/4 at `t=.01` and one common admissible shift-budget
rule, then reserve up to four cells for one time/restart or basis-width hypothesis.
Fix the state, parameters and evaluation norm. Store actual shifts: changing time while
retuning shifts is a different experiment from evaluating one fixed reduced generator
at several times. The historical m=2/m=4 cells used different budgets and remain historical.

Use the existing default action/cache limits first. Before execution record a host-specific
wall-time/memory cap and wide-tier budget, with no silent escalation. A resource refusal
ends that cell; no pruning or term deletion may make it fit. This is a proposed campaign,
not eight measured cells. Promote a wider sweep only if it addresses an unresolved error
component at acceptable cost.

Study anisotropic Mehler widths as a **representation** improvement, with
`q=ell(a+a†)`, `p=i(a†-a)/(2 ell)`, `ell>0`. A displacement is also allowed only with its
unitary coordinate/state/observable map and every induced interaction/constant retained.
Transform the physical state as well as H; otherwise the comparison changes the question.
Higgs displacement does not establish dynamical symmetry breaking. The extra comparison
oscillator must not be added to the physical Hamiltonian to improve conditioning.

Record certification width versus active work, peak memory and full images, not rank alone.
Compare equal-tolerance costs. Discarded Gram weight, repeated H-action tails and the
N8 defect must all be examined. Finite coordinate dilations are unitary on E1; infinite
products in E2 require their own implementability/representation theorem.

**Acceptance:** a fixed-operator/fixed-state comparison with separate solver, basis-tail
and arithmetic errors. No finite-rank operator-norm convergence to an infinite-space
unitary is requested; use seed/observable or justified energy-window control.

## N10 — physical constraints, metric and stability before physical promotion

*Executed 2026-11-15 (finite dictionary, physical quotient open):* twelve
homogeneous SU(3)xSU(2)xU(1) generators are implemented in
`nested_fock_algebra/src/sm_constraints.rs`; the 144 Gauss commutators close to
`5.6e-17` on **gauge-vacuum** images and `4.4e-16` with two gauge quanta, and
`Q_joint^2=0` on 18 ghost images. The adjoint/fundamental jet action is included
on `dG,dW,dphi`. The earlier nonabelian obstruction was a dictionary error: the
SU(3) electric term looped only three of the eight colours and carried an extra
`g` relative to the bare matter current. With all eight colours and bare
structure constants, `[H,Q]` closes on the tested images at `6e-15..1e-14`
(SU(3)/SU(2)) and `7e-15` (U(1)). The twelve `[H,G_a]` commutators were also
checked directly on matter and gauge-occupied images, closing at `1.4e-14`
(matter) and `1.5e-13` (gauge, `G_0`); this covers the generators that a
single-ghost `[H,Q]` image can hide. No `d_i E_i` trace or jet kinetic term is
needed: with independent jets the trace commutes with `A E`. The metric/ghost
parity counterexample remains in `fock_sirk/tests/sm_constraints.rs`. A finite
adjoint Chevalley–Eilenberg quotient is implemented in
`fock_sirk/src/sm_brst_quotient.rs`: cochain dimensions `[12,144,792,2640]`,
ranks `[11,132,660]`, `Q^2` residual `6.66e-16`, and cohomology dimensions
`[1,1,0]`. A finite field-level occupation window is implemented in
`fock_sirk/src/sm_brst_fock.rs`: the `N <= 2` window now includes all 96
complex matter fermion modes, the bosonic gauge-Higgs sector and the twelve
ghosts. It has 11,067 states, `Q^2` residual `0.0`, differential rank 138,
block ranks `[0, 138]`, and cohomology dimensions `[1, 10, 10780]` (total
10,791) with `closed`, `graded` and `matter_graded` all true. Splitting by
matter number gives state dimensions `[1419, 5088, 4560]` and cohomology
`[[1, 4, 1318], [0, 6, 4902], [0, 0, 4560]]`; the six `H^1` states at matter
number one are gauge-singlet right-handed neutrinos. Most of `H^2` is the
top-degree truncation boundary. Finite harmonic representatives are exposed as
a hash-bound quotient map: `H^0` vacuum, `H^1` left nullspace of `Q_2`, and
`H^2` RREF free-column nullspace of `Q_2`. The Lean theorem
`BookProof.SmBrstQuotient.finiteWindow_matter_finrank` binds `q_1 o q_2 = 0`
and the measured block ranks `0/138` to these dimensions, conditionally on the
measured rank hypotheses; the 11,067-dimensional matrix itself is not
formalized. The matter window excludes derivative jets. A separate
jet-inclusive window adds all 120 independent derivative-jet coordinates and
measures 36,207 states, `Q^2 = 0`, differential rank 249, block ranks
`[0, 249]`, jet block ranks `[138, 111, 0]`, and cohomology
`[1, 19, 35689]` (total 35,709); `H^1` is 10 matter-window states plus 9
jet-sector states. The joint charge preserves jet number, so `Q_2` is measured
blockwise in `(jet, matter)`. These jets are independent E1 fields, not
constrained spatial derivatives. The Lean theorem
`BookProof.SmBrstQuotient.finiteWindow_jet_finrank` binds the measured
dimensions and ranks to `[1, 19, 35689]`, and
`BookProof.SmBrstQuotient.jetWindowCharge_sq` formalizes the block-matrix
`Q * Q = 0` identity. Both are in the nanoda-verified payload
`docs/fixtures/sm_brst_jet_window_proof.ndjson.gz` (22,097 declarations), so
the record is `kernel-verified conditional`; the concrete matrix remains
measured, not formalized. No continuum claim is made.
The proof-facing bridge `docs/fixtures/sm_constraints_bridge.json` is
`measured`: it binds the finite residuals above, embeds the finite CE
quotient, the matter Fock window (including its finite quotient map) and the
jet-inclusive window, and keeps the continuum `quotient_map: null`; the jet
window is also pinned separately in `docs/fixtures/sm_brst_jet_window.json`. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n10-joint-constraint-dictionary-2026-11-15).

The present electric-only and electric-plus-matter-density tests are diagnostic probes,
not a complete joint BRST construction. The recorded matter-density probe has
`||Q²psi|| about .1392044`; off-diagonal generators can vanish under a window not closed
under their action. The 44-state longitudinal band has `||(I-P)HP|| about .6234982`.
A large penalty, an electric zero-point floor, or that band's eigenvalues is not a
physical quotient or a physical spectrum.

For the E1 comparison, test the SM4 joint gauge generators on its gauge, Higgs,
matter and jet slots. For the momentum target use the gauge action on its smeared
fields, with the corresponding nonabelian and ghost contributions. Keep the existing
probe under its old label. Specify ghost creation/annihilation reference, ghost number
and physical cohomology degree. Test Q², the constraint algebra, `[H,Q]` and physical
projector/resolvent compatibility on full images. Report algebraic truncation leakage
separately from arithmetic and from an actual defect of the declared operator.

Coordinate with SM1's finite consistency lemmas: a nonzero nilpotent Q that is odd for
an involution U cannot be charge-form symmetric for the metric `eta=U` under the global
hypotheses there. An even contracting homotopy is likewise incompatible with such a
nonzero odd charge. Do not try to pass numerical positivity by assuming these premises.
Keep the ghost grading separate from an admissible physical metric and prove the
canonical homotopy's actual degree. Test a **nonzero** charge and nonzero physical states.

For a new physical reduction, compare gauge-invariant observables and the reduced flow
with the unreduced construction through an actual representative/quotient map. Subtracting
longitudinal kinetic energy in a compressed matrix is a labeled comparison unless the
constraint solution and its induced terms/measure/domains are derived. An arbitrary
positive ambient Hodge norm is not automatically the physical Krein norm.

Also probe SM1's high-spatial-frequency Dirac branches before claiming lower boundedness.
ESA/unitarity and spectral stability are different properties. A growing negative branch
can refute a proposed bound; finite positive Ritz minima cannot prove the full operator
positive. Any particle-hole or vacuum representation change needs the SM0 dictionary and
constant/outer-number bookkeeping, not a scalar subtraction chosen from a finite spectrum.

**Acceptance:** the complete constraint dictionary or a precise counterexample to a
candidate, with independent nontriviality, positivity and dynamics checks. Neither a
failed compressed identity nor a successful toy quartet is promoted to a physical-SM result.

## N11 — physics research with full interactions and stated observables

*Executed 2026-11-15 (unresolved):* `sm_response` measured the up-quark
occupation response to a 5% `gs` quench at `t=.01`: `-3.29e-6` against an N8
two-sided budget `0.5625`; the directly driven gluon occupation gave `2.22e-7`.
A 100% quench at `t=.01` gave `-9.5e-5`/`6.4e-6` against `0.6849`, and at
`t=.5` gave `-0.150`/`1.1e-2` against `34.25`. No cell resolved; null and
electron-Yukawa spectator controls are `0.0`. A further state-preparation search
(up-only, up+down, seed+gluon, seed+Higgs, seed x squeezed gauge, window ground
state, depth-2 window, rank 4) left every budget at `0.56..0.69`; squeezing
`ell=0.65..3.0` and coherent displacements `0.1..2.0` left a floor of
`~1.396e-1`. Residual-ranked window enrichment raised it (`1.4366e-1`) and full
support needs `6.6e9` components; rank 1 gives the lowest floor, `9.88e-2`.
The defect is a property of the returned operator/basis, not the matter or
bosonic preparation. Rank 2–4 give `n_up = -3.4716e-6, -3.4176e-6,
-3.4084e-6`; rank 1 is degenerate (`2.2e-16`). Splitting the interval into
1/2/4 restart steps gives `-3.47164e-6, -3.47180e-6, -3.47187e-6` at the same
total N8 budget `0.56572`. The signal is step-stable to `~2e-10`; extending rank
to 6 plateaus it (`-3.38177e-6` vs `-3.3819e-6` at the default seed;
`-3.40735e-6` vs `-3.4084e-6` squeezed), and rank 8 is inadmissible under the
`7t` Hashimoto shift budget (`count < budget/spacing`, so rank<=6). The bound
remains `~1.6e5` times the signal. A state-dependent
defect norm gives `9.8286` against the full-column `14.0581` at rank 2, so the
norm choice is not the obstacle. Depth 3 is refused by the cache budget. The raw 10% quench-energy difference `0.6804` exceeds the
state budget but is not a bounded observable. The cell is inconclusive rather
than a promoted response; see
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n11-response-pilot-2026-11-15-unresolved).

Choose bounded or energy-controlled observables and the declared nonempty information
prior. Outer-vacuum tests verify enclosure algebra; they do not test occupied physical
predictions. Separate matter fermion number from electromagnetic/gauge charge, and inner
particle occupations from outer-particle number. Use the actual graded joint action.

Retain the current interaction-energy and Ehrenfest diagnostics. Add a small predeclared
response study in named coupling families (strong/weak/hypercharge and each Yukawa family),
with all other parameters and the physical state fixed or its change explicitly defined.
Track the contribution by its **origin in the full Hamiltonian**, not just ladder-word
degree: basis shifts and CCR contractions can move contributions between word degrees.
A changed-coupling model is a new parameter cell, not a representation-only refinement.

For each response enclose sampling-free numerical error using N8, include occupation/
window dependence, and compare with exact commutator identities or independent finite
matrices where applicable. Energy drift, gauge-invariant current identities and canonical
velocity must be tested on the identified operator and observable domains. A gauge-fixed
Higgs mean or supplied Yukawa/vev input is not proof of EWSB or a measured-mass prediction.
Physical-spectrum/phase studies wait for the corresponding N10 representation gate.

**Acceptance:** one informative full-interaction observable/response with an explicit
error budget, physical interpretation and a negative control. Record inconclusive or
unstable channels rather than replace H by a scalar/free surrogate.

## N12 — explore the actual infinite-mode/continuum endpoint without conflating limits

For the finite-mode comparison pilot below, distinguish Hermite occupation and rank
from its mode count, regulator and physical volume. For the primary momentum operator
vary smearing and packet resolution on continuous momentum space. Use the existing
continuum smeared CAR as an algebraic input, not a Hamiltonian/domain theorem. Construct
a family of smeared interactions with compatible embeddings, gauge action and
observables before interpreting refinement as a field-theory limit.

A small numerical pilot can study the graph/form constants and smeared observables along
that **defined** family. Polynomial-core probes can refute a proposed uniform constant;
a finite sample of ratios cannot prove it. Search for parameter/flat channels and track
the fermion-mode dependence of matrix estimates. Do not modify H by adding a confining
comparison term to obtain a favorable plot.

Use Bałaban-style small/large-field and renormalization estimates through
[cftproof C0–C10](../../timepiece/cftproof.md) when their model assumptions match.
The October pure-gauge existence witness is useful, but its simple-Lie-algebra theorem
does not supply chiral SM/Higgs/Yukawa existence or identify our chosen regulator family.
Preserve selected phase/observable/witness identities and do not assume global uniqueness.
Renormalized convergence, anomaly/constraint compatibility and distributional products
need their own analytic proofs. An explicit cutoff-effective endpoint is a valid outcome;
it must remain labeled when all-scale uniformity is unresolved.

*Pilot started 2026-11-16:* `fock_sirk/examples/sm_finite_mode.rs` defines a
compatible scalar bosonic family: modes `k=0..M-1` with fixed dispersion
`omega_k=sqrt(m^2+k^2)`, `H0=sum omega_k N_k`, and
`V=lambda sum_{k<4}(phi_k)^4` supported on the first four modes (matrix-ordered
on a total-occupation cutoff of 4). Refinement appends one free vacuum mode.
Coherent and squeezed-vacuum families have nonzero amplitudes only on those four
modes. At `M=4..8` (70..495 basis states), the embedded `H`, `N` and smeared
field agree to `1.8e-15`; coherent `N=2.2363e-2`, `E=6.4463e-1`,
`<phi>=3.7272e-1`, variance `1.5530`, and squeezed variance `2.1658` are constant
to `~1e-16`. The support mass fraction falls from `1` to `0.9820`, while the
support graph (four modes, six edges) is unchanged. This fixed-coupling family
is exact by construction, not a continuum result: the low-mode couplings are
held fixed and the appended modes are free.

A cutoff-dependent extension now runs in the same example. Every mode up to
the cutoff interacts, `V_M = lambda_M sum_{k<M}(phi_k)^4`, and `lambda_M` is set
by the renormalization condition that the ground-state connected four-point
function of the fixed smeared field, `G4 = <(delta phi_s)^4> -
3<(delta phi_s)^2>^2`, equals its reference value at `M=4, lambda=0.05`
(`G4_ref = -3.0364376e-2`). Matching is by bisection, choosing the root closest
to the previous cutoff's coupling because the truncated `G4(lambda)` is not
monotone. At `M=4..8` the trajectory is `lambda_M =
0.0500000, 0.0498665, 0.0497924, 0.0497599, 0.0497691` (spread `2.4e-4`) and
the worst condition residual is `6.8e-11`. The first-four-mode coherent field
mean (`0.372720`), number, and squeezed variance (`2.16579`) remain constant to
`~1e-16`; the coherent energy rises from `0.6446` to `1.2390` because the added
interacting vacua contribute to the unnormal-ordered energy. The running
Hamiltonian is not embedded exactly (deviation `0.148`--`0.150`), as it should
not be: the appended modes carry interactions. This is a declared finite-mode
normalization, not an identification with a 4d field-theory scheme, and no
continuum conclusion follows from five stable low-mode observables.

A second normalization was run as a scheme check, holding `V_M` and the states
fixed but halving the smearing scale. Its reference value is
`G4_ref' = -3.5362226e-2`, and the resulting trajectory is
`0.0500000, 0.0502295, 0.0504822, 0.0507473, 0.0510269`, moving opposite to the
first trajectory; the largest coupling difference is `1.26e-3` (`2.5%`). The
condition is met to `8.5e-11`. The fixed-state number is identical by
construction, while the vacuum-dominated coherent energy differs by up to
`3.1e-2`. This is a robustness check, not a second theory: it shows the
coupling trajectory is smearing-normalization dependent and does not establish
scheme independence.

Scanning for *all* positive roots sharpens that caveat. At every cutoff `M=5..8`
the four-point condition has two roots: a low branch
`0.0009948, 0.0009931, 0.0009913, 0.0009896` (spread `5.2e-6`) and the reported
branch `0.0498665, 0.0497924, 0.0497599, 0.0497691` (spread `1.1e-4`), with
minimum separation `4.9e-2`. The alternate smearing shows the same two-branch
structure (low branch `~1.25e-3`, high branch `~5.0e-2`). Holding the coherent
state fixed, the low branch gives energies `0.0331, 0.0360, 0.0390, 0.0419` at
`M=5..8`, against `0.7926, 0.9408, 1.0895, 1.2390` on the reported branch: the
branch choice changes the energy by up to `1.197`, far more than the trajectory's
own variation. Within the cutoff-4 truncation, the normalization condition fixes
a **branch**, not `lambda_M`: the trajectory is a declared continuation nearest
the previous cutoff, and neither branch is selected by the physics.

The two-root structure is nevertheless not robust. Repeating the matching at
other total-occupation cutoffs gives:

| cutoff | roots at `M=5..8` |
|---|---|
| 2 | none at 5--7; `0.3143` at 8 |
| 3 | `0.002903, 0.002883, 0.002863, 0.002844` |
| 4 | two roots (the table above) |
| 5 | one root at `M=5..7`: `0.001172, 0.001170, 0.001168` |
| 6 | two roots at `M=5..6`: `0.001213` and `0.1619`, `0.001213` and `0.1618` |

At cutoffs 3 and 5 there is a single root, and the surviving low root moves by a
factor of about `2.5` between them; cutoff 2 does not satisfy the condition at
all for `M=5..7`. At cutoff 6 a second root reappears, but near `1.62e-1`, not
near the cutoff-4 branch at `5.0e-2`. Both the root count and the high-root
location therefore depend on the occupation truncation, and even the low root
is not cutoff-stable. The two-root ambiguity and the reported running
trajectory are artifacts of the cutoff-4 truncation, not regulator-independent
features. The cutoff-6 scan stops at `M=6` and the cutoff-5 scan at `M=7`
because those cells dominate the dense matching cost.

Evaluating `G4` at fixed `lambda=0.05`, without matching, isolates the cause. At fixed
mode count the value changes drastically with the cutoff: cutoff 3 gives about
`-1.11` to `-1.31`, cutoff 4 `-0.03036` to `-0.02805`, cutoff 5 `-0.119` to
`-0.125`, and cutoff 6 `-0.177` to `-0.187`. The largest same-mode spread is
about `2.7`; even inside cutoff 4, adding four modes moves `G4` by `2.3e-3`.
The matching root therefore compensates a cutoff-dependent basis drift rather
than identifying a renormalized coupling. This is a stronger negative than the
root-count scan alone.

The pilot is now a hard gate in `scripts/run-sm-full-validation`; it checks the fixed-family
embedding/spreads, the running four-point residual, the nonzero coupling
trajectory, the cutoff dependence of the branch count and high-root location,
and the absence of continuum/cutoff-uniform claims.

**Acceptance:** a defined family and a finite observable/estimate investigation with the
missing cutoff-uniform theorem stated. No positive continuum conclusion from rank growth,
first-image closure or a few stable eigenvalue spacings.

## N13 — deliver proof-facing artifacts for the actual algorithm

Map the packets to SM0/SM2/SM3 (operator/core), SM1/SM4/SM5 (physical representation),
SM6 (joint algorithm/error theorem), SM7 (experiments) and SM8 (field endpoint).
Proposed proof-facing fixtures include exact small CCR/CAR/graded identities, a named
full coefficient dictionary, Gram/defect bounds and interval residual/exponential data.
Reuse certificate parsing/verification infrastructure after auditing its exact scope.

The Lean specialist must prove that accepted artifacts describe the claimed operator,
state and error predicate. A parser passing on finite numbers is not a theorem about
those numbers' provenance. Preserve endpoint types, source hashes and transitive axiom
checks; changing the Hamiltonian, state, field labels or physical projection must invalidate
the claim. Keep ideal arithmetic proof, code refinement and floating execution enclosure
as distinct rows; “Aeneas ran” cannot stand in for a joint-solver proof.

**Implemented artifact (2026-11-16):** `fock_sirk/src/sm_bridge.rs` defines the
`sm-proof-bridge/v1` record: operator and state identities, exact artifact matrices,
Lean theorem name/statement, proof status, hypotheses and explicit non-claims. The
first record is `docs/fixtures/sm_car_bridge.json`, generated from the ordered
`InnerExpr` action on the two-state one-mode fermion basis. It binds
`a * adag + adag * a = 1`, `a * a = 0`, `adag * adag = 0`,
`parity * a * parity = -a` and `adag * a = number`; the test suite re-derives every
matrix from the actual action and checks the identities in exact integer arithmetic.
The verifier rejects an unbound mutation and refuses to let `status: certified`
stand without a claimed kernel proof. The finite 2x2 integer obligation is now in
`fock_sirk/lean/SmCar.lean` and was exported with Lean 4.28.0 (`7e01a1bf`) and
lean4export 3.1.0 to `docs/fixtures/sm_car_proof.ndjson` (243,234 bytes, 187
declarations). `prob_kernel`'s `verify_export` example replayed it through nanoda
and printed `nanoda: verified ... (187 decls)`; the payload hash
is bound in the bridge, whose status is now `certified` with
`proof_status: kernel-verified`. The remaining scope gap is explicit: the Lean
`Mat2` definitions and the Rust `InnerExpr` matrices are linked by the shared
artifact and a separate exact re-derivation, not by a definitional refinement
proof. This certifies the six finite identities, not the SM Hamiltonian.

**Acceptance:** at least one fully scoped fixture-to-theorem bridge with remaining physical
or cutoff hypotheses exposed. The finite CAR export has now been replayed by nanoda
(187 declarations); any new proof/nanoda claim still requires its own actual run, and
writing a plan or a bridge record is not a verification.

## N14 — execute, report and stop at the evidence boundary

Start **N6/N7**, then **N8** with the finite defect/interval fixture. Run **N10** consistency
and actual-charge design before physical-sector interpretations. Execute the bounded **N9**
pilot, then a selected **N11** observable if its representation/error requirements hold.
N13 supports these packets throughout; N12 is the separately budgeted field-theory track.

*Executed 2026-11-18:* the baseline runner
[run-sm-full-validation](../scripts/run-sm-full-validation) was run on this tree.
`target/sm-full-check.log` exited 0 with **73 test-result passes, 0 failures**,
including the 19 gauge-fixed and 54 Faris–Lavine CHECK rows, the CAR/defect/N10
bridge and jet-window suites, the N12 assertions, and all six nanoda replays
(187-declaration
CAR, 273-declaration N8 endpoint, 38,336-declaration N8 Duhamel,
7,540-declaration N8 matrix-skew, 20,241-declaration N10 matter-window and
22,097-declaration N10 jet-window payloads).
The opt-in wide cutoff tier was run separately and exited 0; it measured
230/638/1176-state windows with leading tails `4.038679200865053` /
`1.2179612832516469` / `0.0`. A direct recheck on 2026-11-18 reproduced those
tails and measured 40.07 / 93.93 / 148.81 s at 2.09 / 4.95 / 6.68 GiB peak RSS.
The consolidated record is
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n13n14-consolidated-execution-record-2026-11-16-this-working-tree).

Record each completed experiment in the implementation ledger with exact manifest/artifact
references. Preserve old cells and explain changed interpretation, operator or estimator.
Do not change a result to “certified” while any needed arithmetic, self-adjointness,
physical-reduction or tail premise is only assumed. Stop a cell at its resource/tolerance
limit, record a rigorous loose bound or a negative result, and propose a new hypothesis
before expanding the sweep. Mehler/SIRK efficiency claims require the declared equal-error
comparison; physical-SM claims require their own proved dictionary and physical-space gate.
