# Full SM research and numerical evidence in nested Fock/Mehler/SIRK coordinates

Current work order: 2026-11-15. This began as a source-review plan. On 2026-11-15,
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

*Executed 2026-11-15:* `fock_sirk/src/sm_defect.rs` implements the `D=iVG-HV`
bound on the returned solve and the cutoff-1 cell reports
`leakage=2.41e-8`, `||D||_F=17.19`, `bound=0.1719` at `t=.01`, labeled
`conditionally-bounded`. A 4x4 interval fixture encloses the direct map error and
the reader rejects invalid/mismatched certificates. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n8-execution-record-2026-11-15-this-working-tree).

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
`C=12`; the 40/163-mode tails at `C=12` are `1.76e-4`/`1.29e-1`. A follow-up
one-mode SIRK cell (`sm_mehler_sirk`) transformed both H and the wide squeezed
state, verified energy to `7.8e-9`, and measured window `70 -> 76`, N8 defect
`1.3987e-1 -> 1.3969e-1`, and elapsed `3.78 -> 4.15 s`: no equal-tolerance
speedup. See
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
(SU(3)/SU(2)) and `7e-15` (U(1)). No `d_i E_i` trace or jet kinetic term is
needed: with independent jets the trace commutes with `A E`. The metric/ghost
parity counterexample remains in `fock_sirk/tests/sm_constraints.rs`. Physical
cohomology and the representative/quotient map remain **unimplemented**. See
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md#n10-consistency-record-2026-11-15-scoped).

The present electric-only and electric-plus-matter-density tests are diagnostic probes,
not a complete joint BRST construction. The recorded matter-density probe has
`||Q²psi|| about .1392044`; off-diagonal generators can vanish under a window not closed
under their action. The 44-state longitudinal band has `||(I-P)HP|| about .6234982`.
A large penalty, an electric zero-point floor, or that band's eigenvalues is not a
physical quotient or a physical spectrum.

Implement the SM4 joint generators including gauge/Higgs/matter terms and the correct
spatial/jet action, with their nonabelian and ghost contributions. Keep the existing
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
state, depth-2 window, rank 4) left every budget at `0.56..0.69`; the defect is
a property of the operator/window, not the matter content. Depth 3 is refused by
the cache budget. The raw 10% quench-energy difference `0.6804` exceeds the
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

Separate Hermite occupation and numerical rank from inner matter/spatial resolution,
UV spacing, physical volume and renormalized coupling trajectory. Use the existing
continuum smeared CAR as an algebraic input, not a Hamiltonian/domain theorem. Construct
a finite-mode family with compatible embeddings, physical jets, constraints and smeared
interactions before interpreting any refinement as a field-theory limit.

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

**Acceptance:** at least one fully scoped fixture-to-theorem bridge with remaining physical
or cutoff hypotheses exposed. All exported proof/nanoda claims require an actual later
specialist run; nothing is exported or reverified by writing this plan.

## N14 — execute, report and stop at the evidence boundary

Start **N6/N7**, then **N8** with the finite defect/interval fixture. Run **N10** consistency
and actual-charge design before physical-sector interpretations. Execute the bounded **N9**
pilot, then a selected **N11** observable if its representation/error requirements hold.
N13 supports these packets throughout; N12 is the separately budgeted field-theory track.

The baseline runner is [run-sm-full-validation](../scripts/run-sm-full-validation).
It fetches/builds Rust and executes Cadabra/tests/examples, so it is a future execution
step, **not a documentation check and not run in this planning wave**. Use targeted
changed-function tests first, then its default suite; the wide tier is explicit. Capture
engine/feature skips and the exact test list. A stale cache or source-only count is not a pass.

Record each completed experiment in the implementation ledger with exact manifest/artifact
references. Preserve old cells and explain changed interpretation, operator or estimator.
Do not change a result to “certified” while any needed arithmetic, self-adjointness,
physical-reduction or tail premise is only assumed. Stop a cell at its resource/tolerance
limit, record a rigorous loose bound or a negative result, and propose a new hypothesis
before expanding the sweep. Mehler/SIRK efficiency claims require the declared equal-error
comparison; physical-SM claims require their own proved dictionary and physical-space gate.
