# Full SM E1 assembly and joint-state Hashimoto integration

For the current continuous-momentum SM target, follow the packet experiment and
Cadabra2 kernel derivation. The runs recorded on this page use the independent-jet
E1 operator wherever the 163-coordinate dictionary is named. The Fourier target
requires its own interacting packet matrix elements and full-action tails.
[Packet experiment](SM_RESEARCH_PLAN.md#continuous-momentum-numerical-work-order) ·
[Cadabra2 kernels](SM_HAMILTONIAN_AUDIT.md#continuous-momentum-cadabra2-work-order)

## Continuous-momentum SM kernel (2026-11-24 execution record)

This tree now implements the **first deliverable** of the 2026-11-24 order: a
nonzero mixed SM matrix element between two distinct smooth momentum packets,
with the declared Fourier conventions and one finite packet-basis
one-outer-particle evolution cell. Status: **measured**, with the usual
non-claims below.

| artifact | role |
|---|---|
| `nested_fock_algebra/src/sm_momentum.rs` | Conventions `f̂(k)=∫e^{-2πik·x}f`, `∂ᵢ→2πikᵢ`, `(a⋆b)(K)=∫a(q)b(K−q)`; Gaussian packets; `gauge_magnetic_cross`, `gauge_magnetic_free`, `higgs_gauge_cross`, `transfer_integral`, `yukawa_transfer`, `dirac_symbol`/`dirac_packet_element`, `transverse_curl_projector`, `scalar_gradient_symbol`; mutation switches and 13 unit tests |
| `fock_sirk/examples/sm_momentum_kernel.rs` | Emits the versioned fixture |
| `docs/fixtures/sm_momentum_kernel.json` | Schema `sm-momentum-kernel/v1`: conventions, packets, elements, five mutation controls, evolution cell |
| `fock_sirk/tests/sm_momentum_validation.rs` | Binds the live module to the fixture (8 tests) |

**Measured elements** (packets centred at `(0.4,0,0)`, `(-0.2,0.3,0)` and a
background at `(0.1,0.1,-0.2)` with widths `0.8/0.9/0.6`; colours `1,5` against
background `3`, so both nonzero `f^{abc}` act):

* nonabelian magnetic cross `3.54478621714i` (zero at `gs=0`, zero for a
  colour commuting with the background);
* gauge–Higgs (weak) cross `−3.30561385164i` between realified components
  `0` and `2` (zero when the cross term is removed);
* Yukawa transfer `5.07673085977 + 1.45049453136i`;
* transferred-momentum integral `7.25247265681`, changing to
  `7.01600028409` under the reversed transfer;
* retained Higgs constant `λv⁴/4 = 1.19021e8`;
* independent spatial checks: free Dirac entry `2.448458` with the left-handed
  chirality its exact negative, the transverse projector exactly zero on a
  longitudinal polarization, and the scalar-gradient symbol `½|k|²`.
  `dΓ(cI)` gives `0` on the outer vacuum, `c` on one quantum and `2c` on two,
  and `dΓ(B²) ≠ dΓ(B)²` on a two-quantum state (the ordered-product check).

Measured evaluator outputs (the elements above, the mutation controls and the
evolution cell) are stored at twelve significant digits via
`sm_bridge::reported_residual`; declared inputs — parameters, packet centres and
widths, `2π`, the lifted `inner_constant` — keep full precision. The one raw
measured field is the Yukawa-adjoint control, which `sm_momentum_validation`
pins at `1e-12` absolute, stricter than twelve digits, so rounding it would
weaken a check rather than normalize a record.

**Mutation controls** (each flips one convention): omitting `2π` changes the
magnetic element to `0.564170248662i`; flipping the nonabelian sign negates
it; removing the Higgs cross gives exactly zero; reversing the transfer changes
the integral; conjugating the Yukawa element gives `5.07673 − 1.45049i`.

**Evolution cell** (`t=0.01`, 3-packet orthonormalized basis, rank-2 window):
Hermiticity defect `0.0`, full return `0.3733731124`, compressed return
`0.4949006513`, measured map gap `0.440538` against the recorded first-order
diagnostic `leakage + t‖D‖‖c‖ = 2.9603129470` (`‖D‖_F = 296.03`). The defect is a
loose bound here because the packet-basis generator is not a reduced resolvent;
it is reported as a measured diagnostic, not an enclosure. The two quadrature
cross-checks (3D packet product, 3D convolution collapse of the 6D transfer
integral) agree with the closed forms to `1e-9`/`1e-7`.

**Not performed / still open.** The cadabra2 symbolic module (engine absent on
this host), the full 1,241-expression dictionary routing, the complete inner
operator and its `joint_sirk` evolution, domain/ESA and physical-gauge
reduction, and removal of smearing. A finite packet match is evidence for the
specified packets only.

## Independent-coordinate research and evidence order (2026-11-15)

Use [SM_RESEARCH_PLAN.md](SM_RESEARCH_PLAN.md), **N6–N14**, for the next numerical and
research work, and [SM0–SM9](../../timepiece/SM_ESA_COMPLETION_PLAN.md) for formal proofs.
The full operator, ordered graded action, symbolic import, actual shifted solves,
compressed-resolvent generator and explicit tails described here are already implemented.
The next work is their exact physical dictionary and certified error/constraint evidence,
not another full-SM builder. No build, Cadabra run, numerical test or Lean replay is
performed by this planning update.

The old 37/41-test counts below describe dated runs. Current source declares 13 exact-action,
7 full-dictionary and 13 primary joint-SM tests, plus supporting suites; no current passing
total is inferred from that inventory. Measured finite-reference errors, full-action tails
and a proposed rigorously enclosed full-flow bound are different result fields. N8 proposes the
full-action defect `D=iVG-HV` certificate for the implemented rational generator.

The current scope interpretation supersedes any physical-sector reading of the historical
penalty/Coulomb prose: the 44-state band does not reduce H, the density probe is not a full
nilpotent joint BRST charge, and subtracting longitudinal kinetic energy in a compressed
matrix does not construct the complete physical quotient. The values remain diagnostic
evidence. Use N10 for the actual constraint, metric, ghost-number and dynamics bridge.
The explicit E1 independent-jet operator is retained throughout.

## N6 execution record (2026-11-15, this working tree)

N6's reproducible-evidence surface is implemented, not just planned. The JSON driver
`fock_sirk/examples/sm_full_hashimoto.rs` now emits a `result_manifest` built through
`fock_sirk/src/sm_manifest.rs`: source revision and dirty-tracked-diff hash, endpoint,
operator fixture/expression identity, state/sector identity, ordered basis-label hash,
requested/actual rank, every shift plus its generation rule, GMRES/Gram tolerances,
precision/Padé/restart conventions, resource caps, backend/timing/peak RSS, the
distinct `historical-pass`/`source-present`/`measured`/`conditionally-bounded`/
`certified`/`proposed` status, and a sha256 artifact binding. `verify_manifest` rejects
unknown statuses/endpoints, malformed digests, non-positive dimensions/ranks, a stale
artifact hash, and any `certified` status without the paired claim. Claiming
`certified` here is a format check, never a numerical or physical certificate.

`scripts/run-sm-baseline` runs one cell into
`target/sm-baseline/<git-rev>-<dirty-hash>/` and records `provenance.txt`,
`inventory.txt` (declared test names, not passes) and `result.json`. This is a
replayable baseline, not the full `run-sm-full-validation` sweep.

Fresh run on this tree (Rust 1.99.0, CPU, release): cutoff 1, 70 states, rank 4,
`t=0.01`, one restart step. The cell verified and reported `status=measured`,
endpoint `E1-unreduced`, energy `51.32175249999999`, first-image tail
`9.784770722594882`, orthogonality defect `1.55e-15`, elapsed `6.24 s`, peak RSS
`657232 kB`. The deterministic result digest (results + identities, excluding
timing) was
`89830378f6b1c76ccddd176a3a3655a1cb9313a5af1e0283299f0e2e82a0175a`; a second
identical cell reproduced it byte-for-byte, while the artifact digest changed only
because it binds elapsed time. These reproduce the historical energy/tail values but
are **new `measured` rows, not a promotion of an old pass**.

Honest current inventory (`cargo test -- --list`, root `d5123ae`, no suite run):
13 `exact_inner_contract`, 7 `sm_full_contract`, 13 `sm_full_hashimoto`, 11
`sm_gauge_fixed_validation` — 44 declared tests. `CADABRA_CLI` was unset, so the
fresh-engine test would have reported its documented skip; no test result is
claimed from this wave.

## N7 execution record (2026-11-15, this working tree)

The checked fixture `docs/fixtures/sm_named_dictionary.json`
(`sm-named-dictionary/v1`) is generated by
`fock_sirk/examples/sm_named_dictionary.rs` from the native builder and re-derived in
`fock_sirk/tests/sm_named_dictionary.rs` against both the native builder and the
independent Cadabra import. It binds:

- all 163 named coordinates (`x`, `g/dg`, `w/dw`, `b/db`, `phi/dphi`) and the 40
  active field/Higgs momenta;
- all four real Higgs slots with every up-type/down-type component of
  `higgs_yukawa_component`, plus the retained wall weight `lambda/4` and constant
  `-v^2`;
- all three spatial kinetic blocks in the manuscript spin frame, with the
  left-chiral sign, the eight SU(3) generators, the computed structure constants,
  the weak epsilon tensor and all eight hypercharges;
- all 49 forms term by term, and all 3366 variable-specific matter terms covering
  the 1152 unique matrix entries and 43 matter variables;
- seven sparse action cells spanning matter number 0/1/2, color 0/2, generation
  0/1/2, left/right chirality, one and two outer universes, and mixed graded parity.
  Each cell records the state spec, support, norm, `state_sha256` and
  `action_sha256`; the native digest must match exactly and the imported action
  must match in support-independent norm. `norm` is the only measured float in
  this fixture and is stored at twelve significant digits; the two digests are
  taken over the unrounded state/image amplitudes, so they stay bit-exact.

The negative controls are hard test failures: mutated/deleted form weight, deleted
term, moved or renamed Higgs slot, omitted matter entry, action-budget refusal, a
noncommuting coefficient import, and the `dGamma(BC) != dGamma(B)dGamma(C)`
control on a two-particle sector. `E::Scalar(c)` is verified to lift to `c N_out`,
not `c I_out`.

Each fixture is linked to `ChapterSmOneParticle.card_smCoord`,
`ChapterSmMatterRepresentation`, `ChapterSmFullYukawaData/Coeffs`,
`ChapterSmFullMatterHamiltonian.smNormalForm` and
`ChapterSmGradTable.exists_potSelfBound_sm` with an explicit remaining gap. In
particular `smNormalForm` has no quartic bosonic family, the named-coordinate
equality with Lean's `smIdx` is not proved, and no fixture-to-theorem action
agreement theorem exists yet. The fixture status is `source-present`, not
`certified`.

## N8 execution record (2026-11-15, this working tree)

`fock_sirk/src/sm_defect.rs` implements the finite-map bound for the generator the
solver actually returns:

```text
D = i V G - H V,
||exp(-itH)psi - V exp(tG)V*psi||
  <= leakage + t max_s ||D exp(sG) V*psi||        -> evolution_bound
  <= leakage + t ||D||_F ||V*psi|| exp(omega_+ t).  -> frobenius_product_bound
```

Both bounds are returned because they answer different questions. The sampled
one follows the trajectory the solver actually produces; the Frobenius product
is the uniform-column bound, which also prices the generator's scale — on a
rational `G = gamma I - B^-1` every entry carries the shift, so the product can
be far larger than the sampled maximum. Where that gap decides the outcome, see
`docs/VERIFY_PHI4_MOMENTUM.md` §6.1.

The full (unprojected) `H V` columns come from the same solve; nothing assumes
Krylov invariance. `JointEvolution` carries one `DefectTerms` per restart step.
The baseline cell — `scripts/run-sm-baseline`, i.e.
`sm_full_hashimoto --time 0.01 --steps 1 --krylov 4 --occupation-cutoff 1` —
reports, separately: `initial_leakage=2.4080250197100035e-8`,
`defect_frobenius=17.18757907479125`, `seed_norm~1`,
`reduced_growth=5.119574578248866e-13`, `flow_growth=1.000000000000005`,
`defect_action_envelope=9.784824513969962`, `evolution_bound=0.09784826921994982`
and `frobenius_product_bound=0.17187581482816355` at `t=0.01`. The
`evolution_bound=0.17187581482816355` of the 2026-11-15 record is now the
`frobenius_product_bound` field: the sampled bound is `1.76x` smaller, and
`state_weighted_bound` coincides with it.

The refreshed baseline deterministic digest is
`63f09785fa3ad2af4cbd8a6c6f7c167108777becab6f26b331a47c3f46baccf8` (artifact
`cec6b876e7def7aae6014dc48f4fd35f701529b6fd4558aaeeb2da822ede1a3c`), replacing
`6fdca3617fae36dbfa91806927cf7ad7e27c86a24753940a96f81ceb1355d923`; two
consecutive runs give the same deterministic digest (the artifact digest
excludes the recorded environment and peak RSS, which is why it is the
reproducibility claim). It differs from the N6 digest because the deterministic
payload now includes every defect term. Both bounds stay labeled
`conditionally-bounded`, and the identified self-adjoint `H`, physical reduction
and Padé/rounding enclosure remain open.

The independent finite fixture is a 4x4 Hermitian `H`, a 4x2 isometry and
`G=V*HV`; its interval arithmetic rounds add/sub/mul/square/sqrt/exp outward,
it encloses `HV` and `G` from the declared `H` and `V`, and its spectral growth
uses a directed Gershgorin bound on the enclosed Hermitian part. The Padé flow
remains unenclosed and the matrix-specific defect bound remains underived, so
the certificate remains conditional; the Duhamel representation, integral bound
and skew-adjoint flow bound are separately kernel-verified, but identifying
`A = iH` for the actual matrix and the physical reduction remain open. The reader
rejects missing, negative, nonfinite or zero-time budgets and mismatched
operator/state/basis hashes, and refuses a status promotion to `certified`.

## Full sweep (2026-11-15, this working tree)

`./scripts/run-sm-full-validation` completed with exit 0 under Cadabra2
`2.5.15 private` (recorded `PYTHONHOME`), with the default (non-wide) tier:

- `scripts/check-sm-symbolic`: gauge-fixed 19 CHECK rows and SM comparison 54
  CHECK rows pass; full SM count 163/40/49/96/1152 and the LaTeX fixture is exact.
  The previous 49-row transcript stopped silently at CHECK 30c: the CHECK 31–33
  `.cdb` assignments lacked the trailing `;` required by this engine, so five
  rows were never executed. Adding the separators and `sort_product` before
  `collect_terms` executes CHECK 31/31b/32/32b/33 and all five return zero.
- Tests, all passed: `exact_inner_contract` 13, `sm_full_contract` 7,
  `sm_full_hashimoto` 13, `sm_gauge_fixed_validation` 11,
  `sm_named_dictionary` 4, `sm_defect` 2, `sm_constraints` 2,
  `sm_joint_constraints` 6. Total **58** fresh passes (rerun after the electric
  closure correction, the two-mode Mehler/SIRK cell, the response search, the
  state-dependent defect diagnostic and the direct `[H,G_a]` checks).
- The cutoff-1 JSON driver completed; the first-image tail remained
  `9.784770722594882`.

The opt-in wide cutoff 2–4 tier was run separately with
`SM_FULL_WIDE_CUTOFFS=1` and exited 0. The three windows measured dimension
230 / 638 / 1176 with leading full-action tails `4.038679200865053` /
`1.2179612832516469` / `0.0`; the 13-test Hashimoto block took 306.27 s. These
are observations on this tree, not cutoff-removal or performance guarantees.

A fresh direct recheck of the three wide cells on 2026-11-18, using the same
raised budgets, reproduced the tails exactly and measured:

| cutoff | dimension | leading full-action tail | elapsed | peak RSS |
|---:|---:|---:|---:|---:|
| 2 | 230 | `4.038679200865053` | 40.07 s | 2.09 GiB |
| 3 | 638 | `1.2179612832516469` | 93.93 s | 4.95 GiB |
| 4 | 1176 | `0.0` | 148.81 s | 6.68 GiB |

These are single deterministic cells, not performance samples; the zero tail at
cutoff 4 still means only that the first image lies inside that window.
The sweep proves these suites pass on this tree/engine; it does not prove
cutoff removal, physical reduction, interval enclosure of the solver, or any E2
claim.

## N10 joint constraint dictionary (2026-11-15)

The finite joint dictionary is implemented in
`nested_fock_algebra/src/sm_constraints.rs` and exercised by
`fock_sirk/tests/sm_joint_constraints.rs` on the complete E1 operator, not on a
projected window.

- **Generators.** Twelve homogeneous generators (eight SU(3), three SU(2), one
  U(1)) use the named gauge coordinates and active momenta, the symmetrized
  nonabelian `A.E` density, normal-ordered matter currents over all 96 CAR
  modes, and the symmetrized Higgs current on the four real components.
  `Sector`/`structure` expose the dictionary; `gauge_dictionary` returns all
  twelve.
- **Ghosts.** Modes `96..107` are the twelve ghost/antighost Fock modes. The
  joint charge is `Q = sum_a c_a G_a - (i/2) sum_abc f_abc c_a c_b b_a`, with
  `c` the ghost annihilator and `b` its conjugate creator.
- **Electric completion.** The earlier obstruction was a dictionary error, not
  a missing physical term: `electric` looped SU(3) colours over `0..3` instead of
  `0..8`, so five gluon colours had no homogeneous `A.E` term, and it carried an
  extra factor `g` relative to the bare matter current. The generator now uses
  all eight colours and the **bare** structure constants; the coupling stays in
  the Hamiltonian's electric/magnetic energy. No separate `d_i E_i` trace or jet
  kinetic term is required for the algebra: with independent jet momenta that
  trace commutes with `A E`, so it would not change the closure.
- **Jet action.** The generator rotates the independent derivative coordinates:
  adjoint `f^abc dG^b p_dG^c` on `dG,dW`, and the fundamental action on `dphi`;
  `dB` is abelian. This is the jet counterpart of the homogeneous `A.E` term.
- **Constraint algebra.** All 144 commutators close on **gauge-vacuum images**
  with worst residual `5.5511e-17`, and on an image with two gauge quanta with
  worst residual `4.4409e-16` (test
  `gauss_algebra_closes_on_gauge_occupied_images`). The electric-only SU(3)
  subalgebra also closes with bare structure constants (`<2e-9`).
- **Nilpotency.** `Q_joint^2 = 0` on 18 one- and two-ghost images (exact zero in
  f64, not a tolerance-limited value).
- **Hamiltonian commutator.** `[H,Q]` now closes on the tested full images:
  ghost 0 (SU(3)) `8.65e-15`, ghost 2 `6.39e-15`, ghost 8 (SU(2)) `9.55e-15`,
  ghost 11 (U(1)) `7.22e-15`. The jet part is `0` for the SU(3)/SU(2) images and
  `2.5e-16` for U(1); all are at the f64 roundoff scale.
- **Direct generator commutators.** `[H,Q]=0` on a single-ghost image only probes
  the components annihilated by `c_a`; the new
  `hamiltonian_commutes_with_each_gauss_generator` test applies all twelve
  `G_a` directly on both a matter state and the gauge-occupied state. All 24
  images close: matter worst `1.3844e-14`, gauge-occupied worst `1.4669e-13`
  (both `G_0`), at f64 roundoff scale. This is still a finite-image identity,
  not a physical cohomology statement.
- **Bridge.** `fock_sirk/src/sm_constraints_bridge.rs::finite_constraint_bridge`
  emits `docs/fixtures/sm_constraints_bridge.json` and binds the measured
  residuals: 144 gauge-vacuum commutators (`5.5511e-17`), 64 gauge-occupied
  SU(3) commutators (`4.4409e-16`), 18 BRST images with `Q^2 = 0` (f64 zero),
  and 24 direct `[H,G_a]` images (`1.4669e-13`). It is `measured`, not
  `certified`, and `artifact.quotient_map` remains deliberately `null`: no
  field-level physical quotient or representative map is constructed.
- **Finite CE quotient.** `fock_sirk/src/sm_brst_quotient.rs` constructs the
  Chevalley–Eilenberg cochain complex of the twelve-generator gauge algebra in
  its adjoint representation, truncated at ghost degree three, and binds it as
  `artifact.finite_ce_quotient` in the same bridge. The cochain dimensions are
  `[12, 144, 792, 2640]`, the differential ranks `[11, 132, 660]`, the measured
  `Q^2` residual `6.66e-16`, the representation-closure residual `4.71e-16`, and
  the cohomology dimensions `[1, 1, 0]` (degrees 0..2). The ranks are SVD
  measurements at tolerance `1e-9`; the harmonic representatives come from the
  canonical reduced-row-echelon kernel of the stacked `[Q_k ; Q_{k-1}^†]`, whose
  free columns give a basis independent of any singular-vector rotation. These
  are measurements, not interval enclosures. This is a finite adjoint CE
  quotient, **not** the field-theoretic BRST complex.
- **Field-level truncation audit.** The ghost-only restriction of the joint
  charge is finite at ghost degree ≤3 and computes Lie-algebra homology, not
  physical BRST cohomology. A direct probe of the full field-level
  `brst_charge` shows that it removes one ghost and preserves the remaining
  occupation, so the total-occupation filtration is `Q`-invariant (it maps
  `N ≤ M` into `N ≤ M-1`). The matter window is bound under
  `artifact.finite_fock_brst_window`: all 96 complex matter fermion modes, the
  bosonic gauge–Higgs sector and the twelve ghosts, with derivative jets
  excluded. It has 11,067 states, occupation dimensions `[1, 148, 10918]`,
  `Q²` residual `0.0`, differential rank 138 with block ranks `[0, 138]`, and
  cohomology dimensions `[1, 10, 10780]` (total 10,791), with `closed: true`,
  `graded: true` and `matter_graded: true`; the matter grading is checked on
  every BRST image rather than inferred from a block-rank sum. The formula is
  `dim H^n = dim C^n - rank Q_n - rank Q_(n+1)`; an earlier
  `states - rank Q = 1,371` count was the kernel of `Q`, not its cohomology.
  Splitting by matter number gives state dimensions `[1419, 5088, 4560]` and
  cohomology `[[1, 4, 1318], [0, 6, 4902], [0, 0, 4560]]`; the six `H¹` states
  at matter number one are the gauge-singlet right-handed neutrinos
  (`gauge_singlet_kernel_dimension: 6`). Most of `H²` is the top-degree
  truncation boundary, not a physical spectrum. Finite harmonic representatives
  are exposed as a hash-bound quotient map: `H⁰` is the vacuum, `H¹` is the
  left nullspace of `Q₂`, and `H²` is the RREF free-column nullspace of `Q₂`
  (with its pivot columns recorded). The Lean theorem
  `BookProof.SmBrstQuotient.finiteWindow_matter_finrank` binds `q₁ ∘ q₂ = 0`
  and the measured block ranks `0/138` to `[1, 10, 10780]`; it is conditional
  on the measured rank hypotheses and does not formalize the 11,067-dimensional
  matrix.
- **Jet-inclusive window.** `finite_fock_brst_jet_window` adds all 120
  independent derivative-jet coordinates (`dG` 72, `dW` 27, `dB` 9, `dphi`
  12) and emits `docs/fixtures/sm_brst_jet_window.json`. The joint charge
  preserves jet number, so `Q₂` is measured blockwise in `(jet, matter)`:
  36,207 states, occupation dimensions `[1, 268, 35,938]`, jet-state dimensions
  `[11,067, 17,880, 7,260]`, `Q²` residual `0.0`, differential rank 249,
  block ranks `[0, 249]`, and cohomology `[1, 19, 35,689]` (total 35,709),
  with `closed`, `graded`, `matter_graded` and `jet_graded` all true. The
  jet block ranks are `[138, 111, 0]`; `H¹` splits as 10 matter-window states
  plus 9 jet-sector states. The 120 jet coordinates are independent E1 fields,
  not constrained spatial derivatives. The Lean theorem
  `BookProof.SmBrstQuotient.finiteWindow_jet_finrank` binds the measured
  dimensions `1/268/35938` and ranks `0/249` to `[1, 19, 35689]`, and
  `BookProof.SmBrstQuotient.jetWindowCharge_sq` formalizes the block-matrix
  identity `Q * Q = 0` when `q₁ * q₂ = 0`. Both are in the exported payload
  `docs/fixtures/sm_brst_jet_window_proof.ndjson.gz` (22,097 declarations),
  rechecked by nanoda, so the record's `proof_status` is
  `kernel-verified conditional`. The concrete 36,207-state matrix and its
  measured `q_squared_residual` are not formalized. The jet record is embedded
  in the main bridge as `artifact.finite_fock_brst_jet_window` as well as
  emitted separately. The finite adjoint CE quotient remains separate; the
  continuum physical quotient, physical jet interpretation and anomaly
  analysis remain
  open.

This is a finite-representation identity on the declared images, not a proof of
physical BRST cohomology. The finite adjoint CE quotient is an actual finite
quotient construction, but it does not identify the complete E1 operator's
physical Hilbert space. The earlier `6.18e-1`/`4.95e-1` values were the
three-colour/extra-coupling defect, not an electric-sector completion; the
naive `sum_i p_dG,ii` trace was rejected for the independent-jet reason above.
The earlier electric-only, matter-density (`||Q^2 psi|| ~ .1392044`) and 44-state
longitudinal-band probes remain labeled diagnostics.

## N9 bounded pilot and Mehler feasibility (2026-11-15)

`scripts/run-sm-n9-pilot` fixes the seed, parameters, `t=0.01`, one restart step
and the common shift rule `dt*(rank+5)`, then crosses the declared 70/230-state
windows with ranks 2/4. Default action/cache caps are used for 70 states; the
230-state cells declare the wide caps `cache 4e6`, `components 8e6`, `work 4e9`
before execution, with a 6 GB host-memory guard. All four cells were measured
(12 GB available, peak 2.19 GB):

| window | rank | defect bound | leakage | first tail | energy | elapsed | peak RSS |
|---|---|---|---|---|---|---|---|
| 70 | 2 | 1.398688e-1 | 4.44e-16 | 9.784771 | 51.3217525 | 4.38 s | 441 MB |
| 70 | 4 | 1.718758e-1 | 2.41e-8 | 9.784771 | 51.3217525 | 6.60 s | 658 MB |
| 230 | 2 | 1.436957e-1 | 3.11e-15 | 4.038679 | 51.3217525 | 13.9 s | 1.33 GB |
| 230 | 4 | 2.013201e-1 | 3.02e-16 | 4.038679 | 51.3217525 | 23.8 s | 2.19 GB |

The energy is rank- and window-independent at this cell; the first-image tail
changes with the window (9.78 -> 4.04) but not with rank; the N8 uniform column
defect bound is **not monotone in rank** (rank 4 adds full-image columns), so it
is not an error trend. Leakage, tail and defect are distinct fields.

`fock_sirk/examples/sm_mehler_widths.rs` then applies the exact unitary dilation
`q=ell(a+a^dagger)`, `p=i(a^dagger-a)/(2ell)` to the fixed bosonic vacuum and
reports the exact squeezed-vacuum tail at a total-boson cutoff. Per-mode mean
bosons and the exact harmonic energy are recorded with the tail:

| ell | mean n/mode | harmonic E/mode | 1 mode tail at C=12 | 40 modes tail at C=12 | 163 modes tail at C=12 |
|---|---|---|---|---|---|
| 1.00 | 0 | 0.375 | 0 | 0 | 0 |
| 1.25 | 0.0506 | 0.4706 | 1.29e-10 | 1.76e-4 | 1.29e-1 |
| 1.50 | 0.1736 | 0.6181 | 3.48e-7 | 7.85e-2 | 9.81e-1 |
| 2.00 | 0.5625 | 1.0312 | 1.99e-4 | 8.81e-1 | 1.000 |

Only the single-mode `ell=1.25` cell reaches the `1e-8` basis-tail tolerance at
`C=12`; no 40- or 163-mode width does. This is a **feasibility/negative**
representation result: the squeeze is unitary on E1, but the occupation window
is not closed under it, and the induced tail dominates. The first feasibility
cell did not run SIRK; the follow-up one-mode cell does.

`fock_sirk/examples/sm_mehler_sirk.rs` now performs the equal-tolerance solve
for a **two-mode** product cell. The modes are `g(0,0)` and `g(1,1)`, `ell=1.25`,
total occupation cutoff 12 (exact discarded product tail `6.03e-10`), `t=.01`,
one restart, rank 2, the same shift rule and action limits. The wide squeezed
vacuum uses `q=ell(a+a^dagger)` and the inverse operator map `q->q/ell`,
`p->ell p` for both modes; both the state and `H` are transformed. The unchanged
energy is the unitarity check. The earlier one-mode cell is retained as the
first row of the table.

| cell | run | window | initial comps | N8 defect | defect Frobenius | leakage | first action tail | energy | n_up | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one mode | baseline | 70 | 1 | 1.398688e-1 | 13.986883 | 4.44e-16 | 9.784771 | 51.32175258 | 0.99990752 | 3.78 s |
| one mode | squeezed | 76 | 7 | 1.396948e-1 | 13.969477 | 9.00e-16 | 9.785928 | 51.32175259 | 0.99990990 | 4.15 s |
| two modes | baseline | 70 | 1 | 1.398688e-1 | 13.986883 | 4.44e-16 | 9.784771 | 51.32175258 | 0.99990752 | 3.78 s |
| two modes | squeezed | 97 | 28 | 1.396284e-1 | 13.962835 | 7.87e-16 | 9.786590 | 51.32175269 | 0.99991213 | 6.14 s |

The two-mode energy agrees to `1.1e-7` (relative `2e-9`), consistent with the
`6.03e-10` discarded tail. The cost gap **persists and widens**: the window grows
by 27 states, elapsed time by 62%, and the N8 defect is only 0.17% lower. The
fermion occupation shifts by `4.6e-6`, again the order of the N11 response. This
is a two-mode finite result, not a 40/163-mode speedup, and no E2 theorem is
implied.

## N11 response pilot (2026-11-15, unresolved)

`fock_sirk/examples/sm_response.rs` evolves the fixed seed under the complete
native E1 operator for `t=0.01`, one restart step, rank 2, 70-state window, and
measures the up-quark occupation `n_up`. The quench scales `gs` by `1.05`.
Measured response `-3.2909e-6`; N8 two-sided budget `2(d_plus+d_base)=0.5625`.
The response is **within** the budget, so this cell does not resolve a nonzero
variation. Antisymmetry residual `-1.558e-7`; null and electron-Yukawa spectator
controls are exactly `0.0`. Base defect `1.3987e-1`, energy `51.32175258`,
final norm `0.9999999984`, first action tails `9.78477` and `9.78477`.

This is a measured null/unresolved response, not a reaction rate or a
physical-SM prediction. The N8 bound, window tail and unenclosed arithmetic are
reported separately.

A follow-up scan tested whether a directly driven channel or a larger declared
quench resolves. The added gauge channel is the occupation of `g(0,0)`; the
larger cells are `gs -> 2 gs` (a new parameter cell), at `t=.01` and `t=.5`.
The N8 budget is recomputed from the same state defect in each cell; it is not
held at the old numerical value.

| quench | time | observable | response | N8 budget | resolved? |
|---|---:|---|---:|---:|---|
| `gs*1.05` | .01 | `n_up` | -3.2909e-6 | 0.5625 | no |
| `gs*1.05` | .01 | `n_g` | 2.2189e-7 | 0.5625 | no |
| `gs*2` | .01 | `n_up` | -9.5136e-5 | 0.6849 | no |
| `gs*2` | .01 | `n_g` | 6.3867e-6 | 0.6849 | no |
| `gs*2` | .5 | `n_up` | -1.5018e-1 | 34.25 | no |
| `gs*2` | .5 | `n_g` | 1.0941e-2 | 34.25 | no |

The long-time signal grows, but the N8 defect grows faster: at `t=.5` the
budget is 34.25 and the largest response is 0.15. Null controls remain exactly
zero; the electron-Yukawa spectator remains zero. **No resolved N11 channel was
found** in these declared cells. The inconclusive channels are recorded rather
than replaced with a free/scalar surrogate.

A lower-defect **state-preparation** search then tested whether the budget can
be reduced at the same `t=.01`, `gs*1.05` cell. The full-action defect is a
property of the operator/window, not of the matter content: every preparation
stayed at `d ~ 1.39e-1`.

| preparation | base defect | plus defect | N8 budget | `n_up` response | resolved? |
|---|---:|---:|---:|---:|---|
| seed (default) | 1.398688e-1 | 1.413678e-1 | 0.56247 | -3.2909e-6 | no |
| up only | 1.390649e-1 | 1.406080e-1 | 0.55935 | -3.3579e-6 | no |
| up + down | 1.401500e-1 | 1.417042e-1 | 0.56371 | -3.3433e-6 | no |
| seed + gluon | 1.398955e-1 | 1.413961e-1 | 0.56258 | -3.2592e-6 | no |
| seed + Higgs | 1.398859e-1 | 1.413849e-1 | 0.56254 | -3.2573e-6 | no |
| seed x squeezed gauge | 1.405809e-1 | 1.422800e-1 | 0.56572 | -3.4716e-6 | no |
| window ground state | 1.400040e-1 | 1.418032e-1 | 0.56361 | -2.2225e-7 | no |
| depth-2 window | 1.398688e-1 | 1.413678e-1 | 0.56247 | -3.2909e-6 | no |
| rank 4 | 1.718758e-1 | 1.737881e-1 | 0.69133 | -3.3819e-6 | no |

The window ground state is an approximate eigenvector of the compressed
operator, but its full-action defect is still `1.40e-1`; the Krylov column
defect is not removed by state preparation. A second scan varied the bosonic
preparation: squeezing `g(0,0)` at `ell=0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0`
gave `1.3964e-1, 1.3970e-1, 1.3992e-1, 1.4058e-1, 1.4319e-1, 1.4951e-1,
1.6161e-1`; coherent displacements `0.1, 0.5, 1.0, 2.0` gave
`1.3990e-1, 1.4037e-1, 1.4102e-1, 1.4156e-1`. The floor is `~1.396e-1`, so the
defect is preparation-independent at this cell. Residual-ranked window
enrichment was also tried: adding the top 200 omitted `H V` amplitudes raised
the defect to `1.4366e-1` (new basis columns carry their own full-action error),
and full support requests `6.6e9` components. Rank 1 lowers the floor to
`9.88e-2`, still four orders above the `~1e-5` scale needed to resolve `n_up`.

A rank/step convergence check separates a basis artifact from a loose
certificate. At fixed `ell=1.25`, rank 1 gives `n_up` response `2.2e-16` (the
one-vector basis cannot represent the change), while ranks 2, 3, 4 give
`-3.4716e-6, -3.4176e-6, -3.4084e-6` and budgets `0.5657, 0.7186, 0.8492`.
With rank 2, splitting the same interval into 1, 2, 4 restart steps gives
`-3.47164e-6, -3.47180e-6, -3.47187e-6`, with total N8 budget `0.56572` in
all three cases. Extending the rank series to 6 confirms the plateau: at the
default seed rank 6 gives `-3.38177e-6` against rank 4's `-3.3819e-6` (budget
`0.69133` in both), and on the squeezed base rank 6 gives `-3.40735e-6` against
rank 4's `-3.4084e-6`, with the budget moving from `0.8492` to `0.8471`. Rank 8
cannot be tested under the declared `7t` shift budget: `hashimoto_unitary_shifts`
requires `count < budget/spacing`, so the maximum admissible rank is 6. All
five cells are also asserted to share one basis window, and the JSON records its
digest: `4a7024d9273c6379` (70 states) at the default seed and
`ac08356d42e44c9f` (76 states) for the squeezed base. The digest covers the
sorted basis keys, so a base/quench window mismatch is a hard driver failure,
not an unflagged mixed-basis comparison. The example sums every restart-step bound; using only the
last step would undercount `steps>1`. Regenerated 2026-11-24 on the current
tree, the rank-2 last-step values are `0.0978, 0.0489, 0.0245` (sampled
`evolution_bound`) and `0.1399, 0.0699, 0.0350` (`frobenius_product_bound` —
that is the quantity the older `0.1406, 0.0703, 0.0351` recorded). Its JSON also records the actual `steps`, `rank` and shift rule
rather than a hard-coded single step, so the step/rank scans are self-describing. The response is therefore stable to `~2e-10` under step splitting and
about 2% from rank 2 to rank 4, but the N8 bound remains `~1.6e5` times the
signal.

*Provenance caveat (2026-11-24).* The rank-2 and rank-3 response values in this
paragraph (`-3.4716e-6`, `-3.4176e-6`) do **not** reproduce from the current
tree: a regenerated rank-2 cell gives `-3.2909e-6` and the same step-scan
`-3.29090e-6, -3.29090e-6, -3.29086e-6`. They are retained as the 2026-11-15
measurement, not as a current one. The rank-4 value `-3.3819e-6`, its
`4a7024d9273c6379` window digest and its Frobenius budget `0.69133` do reproduce
exactly, and that budget is unchanged by the 2026-11-24 split because it is
built from `frobenius_product_bound`. The older rank-2/3 budgets (`0.5657`,
`0.7186`, `0.8492`) belong to the non-reproducing state: the regenerated rank-2
plain budgets are `0.5625` (Frobenius) and `0.3935` (sampled).

It is a reproducible measured signal, not a resolved or certified observable.

An adjoint-weighted (centered) observable bound was also tested. For Hermitian
`O`, the exact-vs-approximate expectation error is
`|<delta|O|psi>| <= 2 eps ||(O - <O>)psi|| + eps^2`, so the state error is
weighted by the observable's centered dual action. At rank 4 on the default
seed, `||(O-<O>)psi||` is `9.6500e-3` (base) and `9.8236e-3` (plus). The
two-cell budget is `2.3183e-2` with the sampled `eps` (2026-11-24; `6.6475e-2`
with the pre-change `eps`), against `3.9346e-1` plain sampled and `6.9133e-1`
plain Frobenius. The weighting is therefore `~17x` tighter than the plain
sampled budget and `~29.8x` tighter than the conservative one, but with the
sampled `eps` the `eps^2` term dominates the sum, and it is still `6.86e3x` the
`-3.3819e-6` signal — it does not close N11. The driver records this as
`n8_adjoint_two_sided_budget` and `within_adjoint_budget`.

A state-dependent diagnostic (`defect_seed_action`) was added to test whether
the full-column norm is the source of the looseness. At rank 2 it gives
`||D c|| = 9.8286` against `||D||_F = 14.0581`; the projected seed excites a
substantially smaller defect direction.

That diagnostic was then integrated into the adjoint bound. `DefectTerms`
carries `defect_action_envelope`, the resolution-converged sampled maximum of
`sup_{0<=s<=t} ||D exp(sG)c||` (it was a grid-plus-Lipschitz estimate using
`||D G||_F ||c|| exp(omega_+ t)` until 2026-11-24; the sampled form is used
because the Lipschitz term prices the shift), and `state_weighted_bound =
leakage + t * envelope`. At rank 4 on the default seed the per-cell
state-weighted errors are `0.0978` (base) and `0.0989` (plus) — identical to
`evolution_bound`, since both now use the sampled envelope — against the
conservative Frobenius errors `0.1719` and `0.1738`. The two-cell
defect-weighted adjoint budget is `2.3183e-2`, which **equals** the centered N8
adjoint budget (`ratio 1.0`): with the sampled form the two "alternative"
budgets are the same formula, so the `0.425x` separation quoted before
2026-11-24 no longer exists. Against the plain budgets it is `17x` below the
sampled `3.9346e-1` and `29.8x` below the Frobenius `6.9133e-1`, and it is
still `6.86e3x` the `-3.3819e-6` response, so it does not close N11. This is a
point estimate: exact `exp`, the returned `G`, and floating arithmetic remain
assumptions. The obstruction is still the basis window, not only the defect
norm choice.

**Formula update (2026-11-24).** The numbers above were produced by a
`defect_action_envelope` that added a Lipschitz correction
`||D G||_F ||c|| exp(omega_+ t)` to a coarse grid. That correction is
shift-scaled — every entry of a rational `G = gamma I - B^-1` carries `gamma`
while the flow stays `O(1)` — so the envelope is now a grid-converged sampled
maximum along the returned trajectory, and `evolution_bound = leakage + t *
envelope` (the `state_weighted_bound` formula, which is why the two fields are
now equal). On the phi4 continuous-momentum cells this cuts the state bound
`5.95 -> 0.18`, and the companion `sm_defect::amplitude_terms` prices the
amplitude `<psi|e>` instead of `||e||` (a computed overlap plus a
`int F(t-s)||r(s)||ds` replacement term), which is a further `~500x` tighter on
a probability charge; see `docs/VERIFY_PHI4_MOMENTUM.md` §5.3 and §6.1.

**SM baselines regenerated 2026-11-24.** The analytic product is preserved under
its own name (`frobenius_product_bound`, `DefectTerms` and
`certificate_json`), so the conservative number is not lost:

| record | before | regenerated |
|---|---|---|
| baseline `evolution_bound` (cutoff-1 cell) | `0.17187581482816355` | `0.09784826921994982` |
| baseline `frobenius_product_bound` | — (was `evolution_bound`) | `0.17187581482816355` |
| baseline deterministic digest | `6fdca361…355d923` | `63f09785…46baccf8` |
| baseline artifact digest | — (not recorded on 2026-11-15) | `cec6b876…2ede1a3c` |
| `sm_defect_bridge.json` file | `916d0536…4ac81ba2d` | `bde3f770…f9910a64bd` |
| `sm_defect_bridge.json` record | `1f49e3b5…c96353d8` | `77dec777…1816ca` |
| rank-4 N11 response | `-3.3819e-6` | `-3.381909761901447e-6` (unchanged) |
| centered adjoint budget | `6.6475e-2` | `2.3183e-2` |
| plain N8 budget (sampled) | `6.9133e-1` | `3.9346e-1` (`6.9133e-1` remains the Frobenius budget) |

The bridge fixture now shares one `sampled_sup` helper with the SM cell path, so
the 4x4 fixture and the shipped operator cannot drift onto different estimators.
The Lean proof payloads and their pinned digests in `scripts/run-sm-full-validation`
are untouched: no proof depends on a numeric bound.

Depth 3 exceeds the declared cache budget (`1,032,256` components against
`1,000,000`), so it is refused rather than pruned. The raw quench-energy difference at a 10% quench is `0.6804`,
larger than the state budget `0.5657`, but energy is an unbounded operator: the
N8 state error maps to an energy error with a factor `||H||`, so this is **not**
a resolved bounded observable and is not promoted. **No resolved N11 channel
was found.**

**N11 disposition (2026-11-16):** closed as an inconclusive finite-window
result for this campaign. The response is stable under restart splitting and
rank 2–6, and the controls are exact zeros, so the remaining gap is the uniform
N8 window defect, not the solver or the state. Further rank, step, preparation
or enrichment scans cannot close a `~1.6e5` factor; a later attempt needs either
a representation whose window defect is small for this observable or a sharper
observable-specific (adjoint/dual) bound. This is not a physical-SM prediction.

## N13/N14 consolidated execution record (2026-11-16, this working tree)

This is the N14 stop-at-the-evidence-boundary report. It records what ran, what is
only conditionally bound, and what remains unproved. No finite-window result below is
promoted to continuum or physical-SM status.

### N13: finite CAR bridge is now kernel-verified

`fock_sirk/src/sm_bridge.rs` defines `sm-proof-bridge/v1`: endpoint, operator and
state identities, artifact digest, Lean theorem name/statement, proof status,
hypotheses and explicit non-claims. It rejects a mutated artifact digest, `certified`
without `kernel-verified`, and `kernel-verified` without a proof payload. The reader
binds records; it does not itself run Lean or nanoda.

`docs/fixtures/sm_car_bridge.json` is generated by
`fock_sirk/examples/sm_car_bridge.rs` from the ordered
`nested_fock_algebra::exact_inner` action on the two-state one-mode fermion basis. It
contains the actual integer matrices for `a`, `adag`, `number = adag * a` and
`parity = 1 - 2 number`, and binds `a*adag + adag*a = 1`, `a^2 = 0`,
`(adag)^2 = 0`, `parity*a*parity = -a`, `adag*a = number` and `parity^2 = 1`.

`fock_sirk/lean/SmCar.lean` states those identities over an explicit 2x2 integer
matrix type using only `decide`/`rfl` proofs. Lean 4.28.0 at commit `7e01a1bf` and
lean4export 3.1.0 produced `docs/fixtures/sm_car_proof.ndjson` (243,234 bytes, 187
declarations). `prob_kernel`'s nanoda path accepted it:

```text
nanoda: verified docs/fixtures/sm_car_proof.ndjson (243234 bytes, 187 decls)
```

The record is therefore `certified` with `proof_status: kernel-verified`, and the
payload sha256 is bound in the record. The remaining gap is explicit: the Lean `Mat2`
definitions and the Rust `InnerExpr` matrices are linked by the shared artifact plus a
separate exact re-derivation in `fock_sirk/tests/sm_car_bridge.rs`, not by a formal
refinement proof. This certifies six finite identities; it says nothing about the
SM Hamiltonian, multi-mode CAR, physical sectors, a cutoff or a continuum.

### N8: finite-map defect bridge is conditionally bounded

`fock_sirk/src/sm_defect.rs::finite_fixture_bridge(0.2)` emits
`docs/fixtures/sm_defect_bridge.json` from the actual `finite_fixture` call and the
existing certificate reader. It binds the 4x4 Hermitian `H`, the 4x2 isometry `V`,
the normalized seed, the point certificate and the interval terms. At `t = 0.2` the
measured map error is `0.7546019867800926`, below the reported interval upper endpoint
`1.07105547383824`; the test recomputes both rather than trusting the fixture.

The status is deliberately `conditionally-bounded`, not `certified`. The record
carries `rounding.directed: true` with scope `add/sub/mul/square/sqrt/exp in
interval_terms; enclosed HV and V*HV; Gershgorin row sums`. Endpoint arithmetic
uses `next_down`/`next_up`; `HV` and `G = V*HV` are enclosed from the declared
`H` and `V` (`full_images_enclosed: true`, `spectral_input_enclosed: true`); and
`lambda_max(Re G)` is bounded by directed Gershgorin row sums
(`spectral_growth = [0, 1.4000000000000035]`). The overall bound remains
`conditionally-bounded` because the matrix-specific defect bound is not derived
from the matrix hypotheses and no physical reduction is identified. The measured error separately uses nalgebra's
Padé exponential without a Padé-error enclosure
(`pade_exponential_directed: false`), so it is a diagnostic witness rather than
a certified one.

The separate Lean theorem `BookProof.Sm.ChapterSmDefect.finite_map_endpoint`
checks the 16-decimal endpoint arithmetic
(`5*10^32*measured <= 5*10^32*leakage + defect*seed*flow`) and nanoda replays the
273-declaration export. It does **not** prove variation of constants, the norm estimate, or the
exponential bound. The separate analytic theorem below proves the Duhamel
representation and its integral bound; `theorem.proof_status` stays `not-run`
for the bridge's matrix-specific statement, and `finite_endpoint_proof` carries
the endpoint scope.
`fock_sirk/tests/sm_defect_bridge.rs` checks regeneration, `verify_bridge`,
`verify_certificate`, the measured-versus-bound inequality, the endpoint proof
hash, and rejects mutation or `certified` promotion.

`fock_sirk/lean/SmDefectAnalytic.lean` now formalizes the missing
variation-of-constants step. `duhamel_representation` proves, in a Banach
algebra, that `u' = -A u + b` implies
`u t = exp(-t A) u 0 + ∫ s in 0..t, exp(-(t-s)A) b s`. The theorem
`norm_duhamel_representation_le` combines that identity with
`norm_duhamel_integral_le` to prove the N8 shape
`||u t|| <= leakage + t*defect*seed*exp(omega*t)`. The remaining analytic input
is the contractive flow bound `||exp(-(t-s)A)|| <= 1`; `norm_exp_neg_smul_skew`
discharges it for every skew-adjoint `A` (`star A = -A`) because the exponential
is unitary, and `norm_duhamel_representation_le_of_skew` packages the resulting
bound. `fock_sirk/lean/SmDefectSkew.lean` now closes the matrix-specific
identification for the declared finite fixture: its exact rational `H` is
proved Hermitian (`H_hermitian`) and `A = i·H` is proved skew-adjoint
(`A_skew`, `Aᴴ = -A`). The remaining N8 gap is identifying the physical
reduction with this fixture, not the finite skew-adjointness. The Mathlib
dependency is a non-default `SmDefectAnalytic`/`SmDefectSkew` target, so the
endpoint build stays independent. Its exported gzip fixture
`docs/fixtures/sm_defect_skew_proof.ndjson.gz` is replayed by decompressing it,
checking the raw sha256 `498ae454…db528cc`, and running nanoda, which verifies
**7,540 declarations**. The analytic gzip fixture is replayed the same way,
checking the raw sha256 `83e3571e…14ed31` and verifying **38,336 declarations**.
The N8 bridge keeps `theorem.proof_status: not-run` for its numerical
matrix-specific statement, records the kernel-verified conditional analytic
theorem separately as `analytic_defect_proof`, and binds the kernel-verified
identity under `matrix_skew_proof` (`H_hermitian`, `A_skew`). Its
`fixture_provenance` field is checked by
`lean_matrix_matches_rust_fixture_provenance`, which parses the exact rational
`H` cells out of `SmDefectSkew.lean` and compares every entry with
`finite_fixture_inputs`; changing either side fails the test.

### Artifact references

| artifact | sha256 | use |
|---|---|---|
| `fock_sirk/lean/SmCar.lean` | `7899f4ae4af4902da1c9d695d96bb07587d48bd9f5bcaccbbf3eeb46661efdaf` | six finite CAR theorems |
| `docs/fixtures/sm_car_proof.ndjson` | `63ef7a554198a5bf9eb04b73c2c9e61950b97689c9265eb4327e0cccdf17c1e5` | nanoda payload, 187 declarations |
| `docs/fixtures/sm_car_bridge.json` | `9bed343105004bfe3e5f380fb0a9b9275ab96786a0e9b72c52affb6e25a7eadc` | bridge file; record digest `766142f3…b7f43797` |
| `fock_sirk/lean/SmDefect.lean` | `ef1df79e448dbdfcb116902b7bb62157c65448d4b1c78be4967063e694d98342` | N8 finite endpoint theorem |
| `docs/fixtures/sm_defect_proof.ndjson` | `1c7bc930eef88f85e9f743db32ffcace60ce2c2ad3cf08bdb8cc9e6bb1cc7261` | nanoda payload, 273 declarations |
| `fock_sirk/lean/SmDefectAnalytic.lean` | `7d88770da2da8a6190e8f5db1a317ec0758e7ab965b085f058b74471521a857e` | N8 Duhamel + skew-adjoint flow bound (Mathlib) |
| `fock_sirk/lean/SmDefectSkew.lean` | `acc3dfc49fad2760020f33617e2ff72e3d74b999a4643046ae6efd2d149afc53` | N8 finite fixture: `H` Hermitian, `A = iH` skew-adjoint (Mathlib) |
| `docs/fixtures/sm_defect_analytic_proof.ndjson.gz` | `ccddf3b7c83edfed0d44aa6d9b9ae81080bff7fc04b04ded4829f65028f96445` | gzip nanoda payload; raw sha256 `83e3571e…14ed31`; 38,336 declarations |
| `docs/fixtures/sm_defect_skew_proof.ndjson.gz` | `eeb92d5b0fc588851ee4600b692c092a3d8e0a982b509b1947c38bc32c7dd9aa` | gzip nanoda payload; raw sha256 `498ae454…db528cc`; 7,540 declarations |
| `docs/fixtures/sm_defect_bridge.json` | `af7e105ab19c432c49611e5f446477b64644f0ef1a29bf43220bbc2a313d4dd7` | bridge file with the kernel-verified `matrix_skew_proof` and its parsed-source provenance check; record digest `f1224414…fcf7d51d`. Carries both `evolution_bound=1.01098521658` (sampled) and `frobenius_product_bound=1.06093839759` (the value this table's older digest recorded as `evolution_bound`); the measured certificate fields are reported at twelve significant digits, the directed `interval_terms` endpoints are not rounded |
| `fock_sirk/src/sm_brst_fock.rs` | `a6535958139893f16600cf74555caa28f0be6b4c2abb441f20df5022af12b07b` | matter and jet-inclusive finite BRST windows; block-graded harmonic quotients (the `H^1` fingerprint is the canonical augmented-RREF left nullspace, not raw singular vectors) |
| `docs/fixtures/sm_constraints_bridge.json` | `0dd581b134e64df8ecd117355a7e694ad709281d7b8e5452adf3a523af175999` | N10 finite constraint + finite CE + finite Fock BRST bridge, including the jet record; record digest `05d08915…d60ee2a6`; residuals reported to twelve significant digits |
| `docs/fixtures/sm_brst_jet_window.json` | `7288181b9cd70c77174428f61a5621c3bbb78b6ae12eaaaed5e4e5c9ca65fab5` | measured jet-inclusive window: 36,207 states, cohomology `[1,19,35,689]` |
| `fock_sirk/tests/sm_brst_jet_window.rs` | `92c923d12f70ed07f81e25170c4f87e8785ab8bcdcfedc8958f0ff9233b9e704` | pins the jet fixture and its verified payload |
| `../timepiece/BookProof/Sm/ChapterSmBrstQuotient.lean` | `bff21dc424b11e32b6425759d0cfa866060820aa717d3ec785b6491abb169165` | finite-window dimensions plus the jet block-matrix `Q² = 0` identity |
| `docs/fixtures/sm_brst_finite_window_proof.ndjson.gz` | `e25d796a16b99689a23bb756324ec261ba08f3ccbf8a6026d7de814b810d7ffa` | gzip nanoda payload; raw sha256 `bedf7935…a881664a`; 20,241 declarations |
| `docs/fixtures/sm_brst_jet_window_proof.ndjson.gz` | `fcd22f8950e4066787a570dd263c3d921439b203e9098b4aa64211e9048c5fe5` | gzip nanoda payload; raw sha256 `66ea3e84…bf462e20`; 22,097 declarations |

### N6–N14 disposition

| packet | evidence status | boundary |
|---|---|---|
| N6 | `measured` | replayable baseline; not a numerical/physical certificate |
| N7 | `source-present` | named dictionary, no action-agreement theorem |
| N8 | `conditionally-bounded` | enclosed `HV`/`V*HV` + Gershgorin bound; Duhamel representation, integral bound and skew-adjoint flow bound kernel-verified; `A = iH` skew-adjointness for the declared finite matrix kernel-verified in `SmDefectSkew` (7,540 declarations); physical reduction and Padé witness open |
| N9 | `measured` / negative feasibility | 70/230-state cells and Mehler tails; no 40/163-mode speedup |
| N10 | `measured` finite-image identity + finite adjoint CE + finite Fock windows | constraint closure, finite CE quotient `[1,1,0]`, a matter window with `Q² = 0`, block ranks `[0,138]`, cohomology `[1,10,10780]` and a Lean-bound harmonic quotient, plus a jet-inclusive window (`36,207` states) with block ranks `[0,249]`, cohomology `[1,19,35,689]`, a nanoda-verified dimension theorem and block-matrix `Q² = 0` theorem; no continuum physical quotient |
| N11 | inconclusive, closed | uniform N8 window defect blocks the response |
| N12 | negative cutoff-artifact result | fixed-`G4` matching is not cutoff-stable |
| N13 | `certified` finite CAR record | one mode, two states; no SM operator claim |
| N14 | this report | commands and artifacts below |

### Exact reproduction commands

The Lean toolchain and exporter are intentionally not project dependencies. They were
fetched under `/tmp`; the checked-in `fock_sirk/lean/lean-toolchain` pins 4.28.0.

```bash
# One-time toolchain setup; both stay outside the repository.
curl -sSL https://github.com/leanprover/lean4/releases/download/v4.28.0/\
  lean-4.28.0-linux_aarch64.tar.zst -o /tmp/lean-arm.tar.zst
mkdir -p /tmp/lean4-4.28.0
tar --zstd -xf /tmp/lean-arm.tar.zst -C /tmp/lean4-4.28.0 --strip-components=1
git clone --depth 1 https://github.com/leanprover/lean4export /tmp/lean4export
cd /tmp/lean4export && PATH=/tmp/lean4-4.28.0/bin:$PATH lake build && cd - >/dev/null

# Lean source and export (toolchain at /tmp/lean4-4.28.0, exporter at /tmp/lean4export)
cd fock_sirk/lean
PATH=/tmp/lean4-4.28.0/bin:$PATH lake build
PATH=/tmp/lean4-4.28.0/bin:$PATH lake env \
  /tmp/lean4export/.lake/build/bin/lean4export SmCar -- \
  BookProof.Sm.ChapterSmCar.one_mode_anticommutator \
  BookProof.Sm.ChapterSmCar.annihilator_square \
  BookProof.Sm.ChapterSmCar.creator_square \
  BookProof.Sm.ChapterSmCar.parity_conjugation \
  BookProof.Sm.ChapterSmCar.number_is_adag_a \
  BookProof.Sm.ChapterSmCar.parity_square \
  > ../../docs/fixtures/sm_car_proof.ndjson
cd ../..
cargo run --locked --offline --release -q -p prob_kernel --example verify_export -- \
  docs/fixtures/sm_car_proof.ndjson

# N8 finite endpoint arithmetic proof (273 declarations, nanoda replay)
cd fock_sirk/lean
PATH=/tmp/lean4-4.28.0/bin:$PATH lake build
PATH=/tmp/lean4-4.28.0/bin:$PATH lake env \
  /tmp/lean4export/.lake/build/bin/lean4export SmDefect -- \
  BookProof.Sm.ChapterSmDefect.finite_map_endpoint \
  > ../../docs/fixtures/sm_defect_proof.ndjson
cd ../..
cargo run --locked --offline --release -q -p prob_kernel --example verify_export -- \
  docs/fixtures/sm_defect_proof.ndjson

# N8 Duhamel + skew-adjoint proof (Mathlib; 38,336 declarations, gzip fixture)
cd fock_sirk/lean
PATH=/tmp/lean4-4.28.0/bin:$PATH lake build SmDefectAnalytic
PATH=/tmp/lean4-4.28.0/bin:$PATH lake env \
  /tmp/lean4export/.lake/build/bin/lean4export SmDefectAnalytic -- \
  BookProof.Sm.ChapterSmDefectAnalytic.norm_duhamel_representation_le_of_skew \
  | gzip -9 -n > ../../docs/fixtures/sm_defect_analytic_proof.ndjson.gz
cd ../..
gzip -dc docs/fixtures/sm_defect_analytic_proof.ndjson.gz > /tmp/sm_defect_analytic.ndjson
sha256sum /tmp/sm_defect_analytic.ndjson  # 83e3571e...14ed31
cargo run --locked --offline --release -q -p prob_kernel --example verify_export -- \
  /tmp/sm_defect_analytic.ndjson

# N8 matrix-specific skew-adjointness (Mathlib; 7,540 declarations, gzip fixture)
cd fock_sirk/lean
PATH=/tmp/lean4-4.28.0/bin:$PATH lake build SmDefectSkew
PATH=/tmp/lean4-4.28.0/bin:$PATH lake env \
  /tmp/lean4export/.lake/build/bin/lean4export SmDefectSkew -- \
  BookProof.Sm.ChapterSmDefectSkew.A_skew \
  BookProof.Sm.ChapterSmDefectSkew.H_hermitian \
  > /tmp/sm_defect_skew.ndjson
sha256sum /tmp/sm_defect_skew.ndjson  # 498ae454...db528cc
gzip -9 -n -c /tmp/sm_defect_skew.ndjson \
  > ../../docs/fixtures/sm_defect_skew_proof.ndjson.gz
cd ../..
gzip -dc docs/fixtures/sm_defect_skew_proof.ndjson.gz > /tmp/sm_defect_skew.ndjson
cargo run --locked --offline --release -q -p prob_kernel --example verify_export -- \
  /tmp/sm_defect_skew.ndjson

# Finite CAR fixture, N8 defect bridge and N10 constraint bridge
cargo run --locked --offline --release -q -p fock_sirk --example sm_car_bridge \
  > docs/fixtures/sm_car_bridge.json
cargo run --locked --offline --release -q -p fock_sirk --example sm_defect_bridge \
  > docs/fixtures/sm_defect_bridge.json
cargo run --locked --offline --release -q -p fock_sirk --example sm_constraints_bridge \
  > docs/fixtures/sm_constraints_bridge.json
cargo test --locked --offline --release -q -p fock_sirk \
  --test sm_car_bridge --test sm_defect --test sm_defect_bridge \
  --test sm_constraints_bridge --test sm_brst_jet_window

# Full default gate (includes the nanoda replay step)
scripts/run-sm-full-validation
```

Fresh default sweep `target/sm-full-check.log`: **exit 0, 73 test-result passes,
0 failures**, including 3 CAR-bridge tests, 5 defect-bridge tests, 7 N10
constraint/CE/Fock-bridge/jet-window tests, 19 gauge-fixed CHECK rows, 54
Faris–Lavine CHECK rows, and all six nanoda replays (187, 273, 38,336,
20,241, 22,097 and 7,540 declarations). The N12
assertions
still pass unchanged: fixed embedding exact, running
`lambda = 0.050000000 .. 0.049769124`, `G4` residual `6.76e-11`, fixed-`lambda` cutoff
spread `2.70`, and the cutoff-6 high branch still shifted. These remain measured or
conditionally-bounded rows, not promotions.

## Implementation baseline and historical measurements

2026-11-09c. This is the executable independent-collective-coordinate endpoint
E1 from `timepiece/SM_ESA_COMPLETION_PLAN.md`. Its inner space is
`L²(R¹⁶³) tensor CAR(96)`, and its final operator is

```text
h_SM = h_B + spatial Dirac + gauge–matter + all four Higgs/Yukawa components,
H_SM = dGamma_out(h_SM) = sum h_ij Cdag_i A_j.
```

Every inner term and the specified Higgs constant is assembled before the
single outer enclosure. This implementation supplies the numerical operator
and symbolic dictionary for the Lean specialist. It does not prove its full
ESA, physical BRST reduction, graded-statistics Hilbert/core identification, or an
infinite inner matter-mode ultraviolet limit. No Lean compilation is involved.

## Source and operator dictionary

`nested_fock_algebra/src/sm_full.rs` defines the complete parameterized
operator. `docs/sm_full_nested.cdb` derives an independent component dictionary
in Cadabra2; its actual LaTeX export is
`docs/fixtures/sm_full_cadabra.latex`. The fixture contains 40 field momenta,
49 bosonic forms and all 1,152 nonzero symbolic entries of the 96-by-96 matter
matrix: **1,241 exported component expressions**. Couplings and all complex
three-generation Yukawa entries remain symbolic until parameter binding.

| Term | Concrete representation |
|---|---|
| Bosonic coordinate | Named blocks `x; G,dG; W,dW; B,dB; phi,dphi`, totaling `3+24+72+9+27+3+9+4+12=163` |
| Field kinetic energy | `1/2 sum_active p_m²`, all 40 gauge/Higgs modes; no extra jet momenta |
| Magnetic energy | All 24 strong, 9 weak and 3 hypercharge magnetic components, squared with weight `1/2` |
| Covariant Higgs energy | All 12 real covariant derivative components, squared with weight `1/2` |
| Higgs potential | `lambda/4 (sum_a phi_a²-vev²)²`, including its fixed constant |
| Spatial Dirac derivative | `sum_j p_xj tensor A_j`, not a fixed plane-wave momentum |
| Gauge–matter interaction | Every strong/weak/hypercharge coordinate multiplied by its spin/chiral/internal CAR bilinear |
| Yukawa interaction | Four real components, all up/down/neutrino/charged-lepton matrices, all generations/colors/spins, and h.c. |

The explicit numeric bosonic labels are a named-coordinate bijection. Lean's
`smIdx` is noncomputable; equality is through named blocks, not a claim that
its internal enumeration has been evaluated. SM retains all independent jet
coordinates. The spatial derivative here acts at fixed collective coordinates;
NS/QG momentum-convolution elimination is not applied to redefine this operator.
A total-jet derivative interpretation would require its separate representation
map and extra terms; it is not silently substituted for E1.

### Full-dictionary kernel and its cross-check

`nested_fock_algebra/src/sm_dictionary.rs` parses every `SM_LATEX` row of the
export and evaluates all **1,241** component expressions as kernels on smeared
packet states, in two separate representations of the same E1 operator:

```text
coordinate:  <bra| P(q, -i d/dq) |ket>      with q-hat multiplication
momentum:    <bra| P((i/2 pi) d/dk, 2 pi k) |ket>   with p-hat = 2 pi k
```

Both are exact closed-form Gaussian moments (degree <= 2), so their agreement
is a roundoff-level Plancherel check of the parser, the coefficient algebra
(full complex 3x3 Yukawa matrices), the `q`/`p` routing and the Fourier
constants. No term in the dictionary mixes `q` and `p`, so normal ordering is
never ambiguous. The measured fixture is
`docs/fixtures/sm_dictionary_kernel.json` (regenerate with
`cargo run -p fock_sirk --example sm_dictionary_kernel`) and the acceptance
test is `fock_sirk/tests/sm_dictionary_validation.rs`: 1,241/1,241 rows,
worst relative gap 2.3e-13 on `wall`, and a mutated `2 pi` changes exactly the
232 rows that carry a jet momentum.

### Per-mode 3D kernel and its relation to the collective one

`three_d_kernel` / `three_d_coordinate_kernel` evaluate every row as a **3D
field-mode** kernel instead (the routing of `sm_momentum.rs`): `q_n` inserts a
3D packet and product terms transfer momentum `k−l`, while `p_m` acts as
`−i∂_{axis(m)}` on the ket. The axis is the named layout's own — `m mod 3` for
`x`, `G`/`dG`, `W`/`dW`, `B`/`dB` (every block base is divisible by three), and
the declared profile axis `m mod 3` for the four scalar Higgs momenta.

That kernel is cross-checked two ways. It is Plancherel-consistent with its own
coordinate route on all 1,241 rows (worst relative gap 8.9e-15), and its
**derivative terms** — the 40 kinetic rows plus every `p` term of the 232
p-bearing rows — are related to the collective kernel by an exact identity:
because each packet's per-axis parameters depend on the axis only through
`a mod 3`, `collective = three_d · O_0^54 · O_1^53 · O_2^53` with `O_j` the
term's one-axis overlap on residue `j` (the derivative factor cancels). `q`-bearing
terms are deliberately not compared this way: `q` is a coordinate
multiplication in the collective kernel and a field insertion with momentum
transfer in the 3D kernel, so they are different operators.

`q`-bearing rows are therefore cross-checked inside each route, over **every**
q-bearing term of the dictionary (3,394 terms; no term mixes `q` and `p`, so each
one is compared against an integrand rebuilt from the term and the packet layout
rather than from the closed-form polynomial — a wrong mode, axis or weight
disagrees). Each closed-form value is compared with a trapezoid evaluation of its
own integrand. Measured: worst collective gap `1.3e-16` (`wall`), worst 3D gap
`3.2e-15` (`magG31`), recorded as `q_bearing_quadrature` in the fixture. This is
an integration cross-check of the insertion, not an equality between the two
kernels.

A mutation test (`a_mutated_field_axis_map_is_caught_by_the_q_bearing_check`)
permutes which packet coordinate feeds which spatial axis and requires the check
to reject every one of the 376 mutated evaluations (4 permutations x the 94-term
per-family subset) with a gap above `1e-6`, so the sensitivity claimed above is
measured rather than asserted. A second counter-assertion
(`a_mutated_collective_mode_index_is_caught_by_the_q_bearing_check`) shifts every
single-mode `q_n` to coordinate `(n+1) mod 163` and requires the collective
quadrature to disagree with the unmutated closed form above `1e-6` in every
case, so the check is sensitive to which collective mode is inserted and not
only to the 3D axis map.

Every measured gap in the fixture (`relative_gap`, the `worst` and `three_d`
worst gaps, and the `q_bearing_quadrature` gaps) is written at twelve significant
digits through `sm_bridge::reported_residual`, so last-digit toolchain noise
cannot change the checked-in record. The kernel values themselves stay at full
precision: the acceptance test pins them to `1e-12` relative, which is stricter
than the reporting rule.

The symbolic companion `docs/sm_momentum_nested.cdb` (8 CHECK rows, in
`scripts/check-sm-symbolic`) pins the conventions and the rigged constants
rather than the integrals: the real `2 pi k` symbol, the `(i/2 pi) d/dk` map and
their commutator, the second-derivative product rule, `V(s=vev^2)=0` with the
retained `lambda vev^4/4`, and the antisymmetric magnetic weight. Because this
kernel is the collective-coordinate operator, it does not replace the per-mode
convolution checks in `fock_sirk/tests/sm_momentum_validation.rs`.

With `D=d+i g T A`, commutation gives the manuscript's
`F=dA-dA-g f A A`. The strong constants are derived from the actual Gell-Mann
matrices in Rust and independently tabulated in Cadabra. Weak constants have
the same minus-epsilon convention. Thus the arbitrary-data Lean bosonic
formula is instantiated with the corresponding signed structure constants.
The four-real-component Higgs generators are the realifications of `i tau/2`
and `i I/2`, not arbitrary real matrices.

## Real/Majorana and complex CAR coordinates

Choose the independent complex chiral coordinates after the manuscript's
reality/chirality constraints. The inventories are:

| Multiplet | Independent complex coordinates | Real coordinates |
|---|---:|---:|
| Left quark doublet | `3 generations * 3 colors * 2 weak * 2 spin = 36` | 72 |
| Right up, right down | 18 each | 36 each |
| Left lepton doublet | 12 | 24 |
| Right neutrino, right charged lepton | 6 each | 12 each |
| Total | 96 | 192 |

The manuscript's conjugate singlets use the opposite complex structure from
the physical right-handed coordinates: `gamma0` anticommutes with `gamma5`.
Changing `J=i gamma5` to `-J` conjugates their complex matrices, converting the
antitriplet to the physical triplet and reversing the book singlet charges.
The implemented complex charges are therefore `Q_L=1/6`, `u_R=2/3`,
`d_R=-1/3`, `L_L=-1/2`, `nu_R=0`, `e_R=-1`. No extra independent
antiparticle copy or Majorana mass is appended. The actual manuscript spin frame is retained:
`alpha_L=(-sigma1,+sigma3,+sigma2)`, `alpha_R=-alpha_L`. An independent
regression starts from the fixed integer `i gamma^mu` and `i gamma5` matrices
in `ChapterA3`/`ChapterCPTHamiltonian`, transports the opposite complex
structures and singlet spin basis, and checks every kinetic block and the
mass-spin contraction. No conventional spin-frame formula is presumed.

For each complex CAR coordinate, the real Clifford coordinates are
`gamma_x=c+c†`, `gamma_y=i(c†-c)`. For a Hermitian matrix `K=A+i B`,

```text
c† K c = Tr(K)/2 + (i/4) gamma^T R gamma,
R_xx=B, R_yy=B, R_xy=A, R_yx=-A.
```

The regression applies this real antisymmetric quadratic form and the complex
CAR bilinear to the same states, retaining the trace constant. Full gauge
covariance is checked for every Higgs component, color, generation and spin.
These executable identities are numerical evidence for the dictionary;
formal representation/intertwining remains SM0 work in the current proof plan.

The complex doublet is `H=(phi0+i phi1, phi2+i phi3)/sqrt(2)` and its partner
is `tilde H=(conj H1,-conj H0)`. The Hamiltonian includes
`Q_L† H Y_d d_R + Q_L† tilde H Y_u u_R` and the lepton analogues, plus h.c.
This makes the real manuscript `-L_Y/2` convention explicit as the full paired
Hermitian interaction after choosing independent canonical variables.
`SmParameters` supplies canonical Yukawa coefficients, not measured masses;
a mass calibration must specify its `vev/sqrt(2)` and field-normalization map.
The test instance has nonzero gauge/quartic couplings and distinct complex,
fully populated three-generation matrices; it makes no measured-mass prediction.

## Cadabra LaTeX really enters the operator

The Cadabra module checks the full bosonic Legendre expression, Hermiticity of
every matter entry, and completeness of all 96 internal coordinates, then
exports actual LaTeX through `_latex_()`. The import in
`sm_full::cadabra_import` uses the existing **mathhook** calculus crate.
Its adapter handles Cadabra's braced indices/fractions/powers and protects
multi-letter scalar parameters from implicit-multiplication splitting.

Only commuting coefficient polynomials cross this bridge. Spatial momenta
commute with the independent field/jet coordinates; same-mode `q*p` imports
are rejected. Unknown AST nodes, missing/extra components, wrong momenta and
mutated coefficients fail. The imported expressions are compared termwise
with the independent native dictionary and then used to build the numerical
operator. The unsafe legacy scalar/contraction-dropping compiler is not used.
This is a deliberate exact import path inside the existing calculus integration,
not a claim that the legacy compiler's ordering behavior has been repaired.

Cadabra remains a subprocess; no GPL engine code is linked into Rust.

## Actual rational SIRK on joint states

`fock_sirk/src/joint_sirk.rs` accepts the joint vectors directly and builds a
bijective registry of joint outer labels. Its default method constructs
sequential **resolvent** vectors using matrix-free GMRES with two-pass modified
Gram–Schmidt and measured shifted residuals. For `A=-i h`, the paper's
positive shifts `gamma_j=(N-j delta)/t`, `j=1..m`, correspond to
`z_j=i gamma_j` in `(h-z_j)^-1`. The budget requires `m<N/delta`.

Forward products `(h-z_j)w` remain an explicitly named polynomial diagnostic.
They are not substituted for the rational space or automatically assigned
Hashimoto–Nodera's Theorem 4.1 bound. Positive shifts alone do not discharge
the paper's operator-function/numerical-range or best-approximation hypotheses.

For the rational basis V, the function uses exactly the bounded-resolvent
compression `B=V†(gamma_m I+iH)^-1 V`, then
`G=gamma_m I-B^-1`, and `V exp(tG) V† v`. The separately computed
`V† H V` is used for Hamiltonian Ritz diagnostics. Substituting its exponential
would give a different method. With a declared numerical projection P, the
shifted solves use `P H P`; complete outside-window residuals report the
resulting difference from the target full resolvent.

The existing Gram-whitening routine is followed by two-pass reorthogonalization
of the actual physical sparse vectors. Projections, reconstruction and full
residuals use those vectors directly, avoiding cancellation in nearly dependent
Gram coordinates. The small exponential uses nalgebra's Padé implementation.
Rank loss is reported as numerical rank. Finite rational approximations may
have norm defects; tests measure them against independent references rather
than claim exact unitarity. Restart diagnostics retain norms, full-operator
energy expectations and all solve/function/projection error data.

A declared finite basis window can cache the complete operator images and
its compression. Internal GMRES iterations reuse that same compression while
full images are retained for residual/tail calculations. Cache-entry budgets
fail explicitly. This speeds the complete operator without removing terms.

There is no hidden amplitude threshold, top-k deletion or interaction removal.
An optional total-occupation cutoff or explicit basis window is a declared
compression of the same full operator. Each action is computed before that
projection, and its discarded norm is recorded. GMRES failures/resource limits
propagate. Full shifted residuals divided by `|Im z|` are recorded as conditional
self-adjoint resolvent error bounds, including projection error.

Ritz residuals are recomputed from the full operator images, so omitted-window
components cannot disappear behind a Gram recurrence. Finite residuals do not
exclude unseen spectrum or certify a physical mass gap. Floating-point/rounding
certification and full cutoff-removal estimates remain separate proof work.

## Reproducible commands

Cadabra2 2.5.14 was provisioned through Nix and all locked Cargo dependencies,
including `ahash`, were fetched. A clean environment can use:

```sh
cargo fetch --locked
nix-shell -p cadabra2 --run 'cadabra2-cli -q -n docs/sm_full_nested.cdb'
cargo test --locked --offline --release -p nested_fock_algebra --features latex --test exact_inner_contract --test sm_full_contract -- --test-threads=1
cargo test --locked --offline --release -p fock_sirk --test sm_full_hashimoto --test sm_gauge_fixed_validation -- --test-threads=1
```

### Host prerequisites

Three conditions are not satisfied by a bare shell on this host and must be
handled before the commands above behave as written.

**Toolchain on `PATH`.** `cargo`, `rustc` and a C linker are not installed
globally; the pinned toolchain is taken from the Nix store and prepended. The
versions match `rust-toolchain.toml`:

```sh
export PATH=/nix/store/c4svgcgjzph9cy4r1fbq5hh9g55krv5r-rustc-wrapper-1.99.0/bin:\
/nix/store/fbw7w7blbj5yvxz4fz2gc0ihm898lwgq-cargo-1.99.0/bin:\
/nix/store/6gj4dzcv6gaxh5raqyrq7jmy605qaj3z-gcc-wrapper-16.2.0/bin:$PATH
cargo --version   # cargo 1.99.0 (5f94df478 2026-08-27)
```

**One online fetch first.** `cargo fetch --locked --offline` cannot succeed on a
cold registry cache: dependencies of other workspace members (for example
`wasm-bindgen`, `ahash`) are not vendored yet. Run the online `cargo fetch
--locked` once; the `--offline` test commands then work.

**`rustfmt` is a separate package.** The cargo store path above ships no
`cargo-fmt`, and no `rustup` is installed, so formatting comes from nixpkgs at
the pinned version:

```sh
nix profile install 'nixpkgs#rustfmt'     # rustfmt-1.99.0; ~/.nix-profile/bin
cargo fmt -p fock_sirk --check           # clean
```

The binary reports `rustfmt 1.10.0`, which is the formatter's own version, not
the toolchain's. `cargo fmt --all --check` is **not** clean: 20 files in
`logos`, `mm0web`, `prob_kernel`, `unfer_measure` and `unfer_taler` predate
this and would need their own change, so a workspace-wide gate cannot be turned
on as-is.

**Cadabra2 needs its own Python.** `cadabra2-cli` is a wrapper that starts an
embedded Python. When the host has a system Python of the same major version it
resolves `/usr/lib/python3.14` instead of its bundled stdlib and aborts with
`ModuleNotFoundError: No module named '_struct'` (exit 134, core dumped). Point
it at the Nix Python rather than exporting into the caller's shell, and let
`CADABRA_CLI` name the wrapper:

```sh
cat > target/tmp/cadabra2-cli <<'EOF'
#!/usr/bin/env bash
export PYTHONHOME=/nix/store/lq2fsnmg0q3y0vlxn91w8s8wvj1m6fld-python3-3.14.8
export MPLCONFIGDIR=/tmp/mplcfg
exec /nix/store/6qy19bwzbzvxk24sa10wgdxvs6160izd-cadabra2-2.5.14-p1/bin/cadabra2-cli "$@"
EOF
chmod +x target/tmp/cadabra2-cli
export CADABRA_CLI=$PWD/target/tmp/cadabra2-cli
target/tmp/cadabra2-cli -q -n docs/sm_full_nested.cdb | cmp - docs/fixtures/sm_full_cadabra.latex
```

`nix-shell -p cadabra2` has no shell environment on this host; the same package
was realized with `nix build --no-link 'nixpkgs#cadabra2'`, a binary-cache
substitution of `cadabra2-2.5.14-p1` (no source build). The engine's version
banner prints `2.5.15 private` even at that store path; the export is still
byte-identical to the fixture.

Set `CADABRA_CLI` to the engine path for the fresh-engine/fixture equality test;
without that variable the test explicitly reports an engine-check skip. The
operator/fixture import tests still run. `scripts/check-sm-symbolic` regenerates
and checks the complete export independently of Rust compilation.


A runnable driver imports the same complete LaTeX dictionary and prints JSON
with shifts, numerical window, norms, energies, full action tails and resolvent
error data:

```sh
cargo run --locked --offline --release -p fock_sirk --example sm_full_hashimoto -- --time 0.01 --steps 1 --krylov 4 --occupation-cutoff 1
```

Its illustrative parameter instance is not a measured-mass calibration; every
physical interaction remains in the applied operator. Its output now starts with a
`result_manifest` (N6), and `scripts/run-sm-baseline` writes that output plus a
declared-test inventory and provenance into `target/sm-baseline/` without running
the wider suites. Wider occupation cutoffs
need explicit resource budgets, because the cached window grows quadratically:
`--cache-limit` (cached operator entries, default 1,000,000),
`--action-components` (entries in one complete image, default 500,000) and
`--action-work` (accumulated image entries per action, default 500,000,000).
Exceeding any of them is an explicit error naming the requirement; the operator
is never truncated to fit.

`scripts/run-sm-full-validation` applies the host prerequisites above and runs
this whole section end to end (symbolic export and CHECK rows, both test suites,
then the JSON driver). It records the engine CLI and `PYTHONHOME` it settled on
in `target/tmp/cadabra-env` and reuses that record instead of re-probing every
store Python; delete the file, or set `SM_FULL_CADABRA_ENV`, to override it.
`SM_FULL_WIDE_CUTOFFS=1` additionally prints the cutoff 2-4 JSON tails, which
need the raised budgets above and peak near 7 GB.


## Executed validation (2026-11-09c)

Rust `1.99.0`; locked Cargo dependencies; CPU backend. All **37 tests passed**
through normal Cargo commands: 13 exact-action contracts, 7 complete dictionary/
Majorana/Cadabra contracts, 6 primary full-SM rational tests, and 11 supporting
SM algebra checks. The fresh-engine equality test ran with `CADABRA_CLI` set;
it was not skipped. No Lean compilation was performed.

The primary full-SM tests use parity-graded outer states, including a mixed
even/odd two-particle action check. The test instance uses a 70-state declared finite reference window
and independently assembled full-SM matrix columns; occupation refinement
expands the window to 230 states. All terms remain in the complete operator.
The actual rational evolution errors against that same finite reference are:

| Configuration | Measured state error |
|---|---:|
| t=0.01, m=2, shift budget rate 8 | 9.15134e-5 |
| t=0.01, m=4, shift budget rate 9 | 4.17940e-7 |
| t=0.01, m=5, shift budget rate 12 | 5.90702e-7 |
| t=0.02, m=4, four restart steps | 8.25354e-7 |

The first complete-action tail norm drops from about **9.78477** at occupation
cutoff 1 to **4.03868** at cutoff 2. These substantial tails are reported, not
hidden behind the small finite-reference errors. The run does not establish a
small error for the unbounded full-operator evolution or remove its cutoff.
Ritz and shifted residuals include those tails. The independent dense resolvent
checks the paper's function compression separately from the Ritz Hamiltonian.

The cache-enabled six-test run with graded outer states took 80.79 seconds in this environment;
this is one observed run, not a performance guarantee. No pruning was enabled,
and cache/action budget counterfactuals returned errors rather than simplify H.

## Executed validation (2026-11-09, host re-run)

A repeat of the commands in *Reproducible commands* on the same working tree,
from a shell where only the host prerequisites above were added. Rust `1.99.0`;
locked dependencies; CPU backend. All **41 tests passed**: 13 exact-action
contracts, 7 complete dictionary/Majorana/Cadabra contracts, 10 primary full-SM
rational tests and 11 supporting SM algebra checks. The fresh-engine equality
test ran with `CADABRA_CLI` set and was not skipped. No Lean compilation was
performed.

Symbolic side: `docs/sm_full_nested.cdb` regenerated the export byte-identically
(1245 lines, `cmp` clean), and `scripts/check-sm-symbolic` reported 19
`sm_gauge_fixed_hamiltonian` CHECK rows, 49 `faris_lavine_n_sm` CHECK rows, and
the full-SM inventory (163 coordinates, 40 momenta, 49 forms, 96 CAR
coordinates, 1152 matter entries) exactly.

The measured rational errors and tails reproduce the recorded values:

| Configuration | Measured state error |
|---|---:|
| t=0.01, m=2, shift budget rate 8 | 9.15134e-5 |
| t=0.01, m=4, shift budget rate 9 | 4.17940e-7 |
| t=0.01, m=5, shift budget rate 12 | 5.90702e-7 |

The m=5 row is historical: the default refinement now solves m=2 and m=4 only,
because the tighter budget measured *less* accurate (`5.90702e-7` against
`4.17940e-7`) and cost a full extra set of shifted solves.

The first complete-action tail norm closes monotonically with the declared
occupation window: **9.784770722594882** at cutoff 1 (70 states),
**4.038679200865053** at cutoff 2 (230 states), **1.2179612832516469** at cutoff
3 (638 states) and **0.0** at cutoff 4 (1176 states), where the first image of
the full operator lies entirely inside the window. Later tails remain nonzero,
as expected for a window built from that first image. `sm_full_hashimoto_budget_refinement`
asserts all four windows and both endpoints (strict decrease, then exactly
zero). The tails are now computed from the first complete action image rather
than by one costly solve per cutoff — the tail the solver reports *is* the part
of `H|psi_0>` its window omits — so the test dropped from 97.56 s (4.5 GB peak)
to **23.08 s**, with the 230-state solve kept as a cross-check of the reported
tail. Cutoffs 3 and 4 exceed the shipped
budgets, so they need
`--cache-limit 4000000 --action-components 8000000 --action-work 4000000000`
(fresh measurements of the example: 23.31 s / 2.19 GB at cutoff 2, 65.77 s /
5.19 GB at cutoff 3 and 111.72 s / 7.01 GB at cutoff 4, all with the raised
budgets). With the
defaults the same runs refuse explicitly: 1,001,177 and 1,382,976 cached
entries against a 1,000,000 limit, and images over the 500,000-entry action
limit. The
restarted-evolution case at t=0.02, m=4 measured
`9.126129501984431e-7` and `8.253536719381402e-7` for 1 and 4 restart steps; the
2-step split (`8.35880401995079e-7`) is covered by the energy test. The
`sm_full_hashimoto` example at cutoff 1 exited 0
and printed the same window, energy (`51.32175249999999`), tail and a norm
defect of `3.6e-14`; at cutoff 2 it reports the 230-state window
(dimension 230, support 230) and the 4.03868 tail. The primary six-test block
took 77.84 seconds before the wider cutoff assertions and 148.84 seconds with
them. The ten-test primary block now runs in **112.70 s** (`--test-threads=1`),
against 329.92 and 335.74 seconds for the earlier eight-test block (8.1 GB
peak); the budget-refinement cross-check now uses a single shifted solve. The
936-state Ehrenfest point is the remaining expensive case and is opt-in:
`SM_FULL_WIDE_CUTOFFS=1` re-measures it, taking the block to 234.36 s, so the
point costs about 122 s.

A full wide-tier sweep (`SM_FULL_WIDE_CUTOFFS=1 ./scripts/run-sm-full-validation`)
exited 0 on this tree with all 67 test-result passes green, and the cutoff 2/3/4
tails reproduced exactly (`4.038679200865053` / `1.2179612832516469` / `0.0` on
dimensions 230 / 638 / 1176); the 13-test Hashimoto block took 306.27 s. These
are observations, not performance guarantees.

### Physics invariants under the compressed flow

The 2026-11-15 scope note above governs the constraint interpretations below. In particular,
the electric and density probes and Coulomb block are finite diagnostic models, and total
matter fermion number is not a verification of electromagnetic gauge-charge conservation.

Three further tests ask what the SM operator does, not whether the solve is
self-consistent.

**Energy is conserved.** The exact flow is unitary, so `<H>` is fixed; the
compressed rational flow can only move it by its own truncation. At t=0.02 the
reference energy is `51.32175249999999` and the drift is
`3.2670754990249407e-11` (1 step) and `5.186961971048731e-13` (2) — roundoff, so
the assertion is the bound, not an ordering. (The 4-step split measured
`1.8417267710901797e-11` in the earlier run and is covered by the
restarted-evolution test.) Every recorded energy is real to `1e-9`.

**Canonical field velocity.** With `q = a + a†` and the documented SM momentum
`p = i(a† − a)/2`, Ehrenfest gives `d<q>/dt = <p>`. The test uses the two-universe
state `(|0> + e^{iπ/4}|1>)/√2` on `coord::g(0,0)` with the same quark/lepton
pair, whose analytic values are `<q> = cos(π/4) = 0.7071067811865475` and
`<p> = sin(π/4)/2 = 0.3535533905932737` — the second pins the momentum
normalization that `[q,p] = i` depends on, and both are asserted to `1e-12`. At
dt=1e-3 the measured velocity converges to `<p>` as the window closes:

| Window | States | Leading tail | `d<q>/dt` | Defect vs `<p>` |
|---|---:|---:|---:|---:|
| cutoff 1 | 72 | 9.9275 | 0.55751 | 2.0395e-1 |
| cutoff 2 | 325 | 7.5488 | 0.32488 | 2.8678e-2 |
| cutoff 3 | 936 | 3.0897 | 0.34923 | 4.3185e-3 |

The defect is the omitted action of the window, not solver error: it falls by
roughly a factor 7 per cutoff while the tail falls by 1.3–2.4. The default run
uses cutoffs 1 and 2 and asserts that the defect both decreases and stays under
10% of `<p>`; `SM_FULL_WIDE_CUTOFFS=1` adds cutoff 3, where it reproduces
`4.31846279135123e-3` and must stay under 1.25%.

**The gauge sector's floor is a zero-point, not a constraint gap.**
`sm_full` carries no ghost sector, so the test ports the charge into this
representation, with the twelve ghosts on fermionic modes `96..108` — outside
the 96-mode matter block, where no `h_SM` term can reach them. The ladder is what
decides the content: `sm_brst_charge` writes `InnerFermionAnnihilate`, the ghost
*raise* of the filled-sea convention of `ym_brst_charge`, which is zero on a
ghost-free state and therefore makes `Q|gs> = 0` trivial. The empty ghost
reference of this dictionary uses `FermionCreate` as the raise, and then
`Q|psi> = 0` is exactly Gauss's law `G_g|psi> = 0`. Measured: `|Q|gs>| =
3.005099133178078` at `theta_0 = 49.7633053378465`, `Q^2 = 0` exactly, and
`|Q gs>|^2 = <gs|Q^dagger Q|gs> = 9.030620800227668` to `1e-9`.

The compressed `Q^dagger Q` is positive semidefinite with floor
`9.0 = 12 sectors x 3 components x 1/4`, the free electric zero-point of the
gauge vacuum, and its spectrum takes exactly three values — `9`, `9.5`, `10`,
with multiplicities 44/12/14 — i.e. it *counts gauge quanta* on top of that
constant, by half a quantum per excited component. So the floor is not a
constraint gap: `H + lambda Q^dagger Q` shifts every energy by `lambda * 9`
(measured `9.0226`, `90.0697`, `900.089` for `lambda = 1, 10, 100`, relative
excess `2.5e-3`, `7.7e-4`, `9.9e-5`) and selects no state, and
`<gs|Q^dagger Q|gs> = 9.0306` is the ground state's small population of excited
gauge components above the floor, not a charge violation. The analytic reason a
normalizable `G|psi> = 0` does not exist is separate: the generator is a field
momentum, and `p = i(a^dagger - a)/2` has continuous spectrum, so the local
constraint has to be *solved* rather than imposed on a Fock state. This replaces
the earlier ghost-vacuum reading, which could not speak about Gauss, and the
earlier reading of the floor as "no state obeys the constraint".

**Gauss's law with the matter density is content-sensitive.**
`sm_full_gauss_with_matter_charge_is_content_sensitive` adds the other half of
the constraint, `G_g = sum_i Pi_i^g - g_a rho^a` with
`rho^a = psi^dagger T^a psi`, built from the same generator matrices and
couplings the covariant derivative of `build_matter` uses (`gs` with
`lambda^a/2`, `gw` with `sigma^a/2`, `gy` with `Y`). On the 70-state window the
density is diagonal in all twelve sectors, and non-vanishing only in the four
that act diagonally on definite matter content, with the dictionary's own
values: `0.3 * {-1/2, 0, 1/2}` (colour `T^3`),
`0.3 * {-1/sqrt3, 1/(2 sqrt3)}` (colour `T^8`), `0.2 * {0, 1/2, 1}` (weak
`T^3`) and `0.15 * {-5/6, -1/3, 1/6}` (hypercharge). The eight off-diagonal
generators compress to exactly zero: the window is not closed under gauge
rotations, so it carries no colour or isospin coherence for them to act on —
a truncation fact the electric-only reading could not see either.

The consequence is not a constant shift. The penalty's floor drops from `9`
(the electric zero-point) to `8.937718492951`, *below* it, and its spectrum
spreads from three values to ten (multiplicities 1/1/2/4/4/9/10/12/12/15): the
least-violating window state is no longer the gauge vacuum, so the penalty now
selects. The ground state's violation nonetheless rises, from `9.030620800228`
to `9.232585715293` (`|Q gs>` from `3.005` to `3.038517025671`) — a lower floor
with a higher ground state is what a charge-sensitive functional looks like —
and `|Q gs>|^2 = <gs|P|gs>` still holds to `1e-12` when `P` is built at
operator level.

What this does not establish is nilpotency of `Q`. With one ghost mode per
sector and no antighost or cubic ghost term, `Q^2` is not zero here
(`|Q^2 gs> = 0.139204403714`); the matter term makes that visible instead of
hiding it behind an abelianised generator. The gauge algebra on the density
itself is checked instead, since that is what the matter term brings:
`[rho^0, rho^1] = 0.045i = i gs^2/2` and `[rho^8, rho^9] = 0.04i = i gw^2` on
the seed (one quark and one lepton doublet member in the `+1/2` slot each), and
`[rho^2, rho^10] = 0` for the two Cartan directions. The missing cubic ghost is
what would cancel `[G_g, G_h] = i g_{gh} f^{ghk} rho_k`.

**Coulomb gauge: the eliminated mode is a constant on the longitudinal
vacuum.** `G_g = sqrt(3) Pi_L^g` with `Pi_L^g = sum_i Pi_i^g / sqrt(3)`, so
`Q^dagger Q = 6 * (1/2) sum_g (Pi_L^g)^2` (checked to `1e-12`). With `rho = 0`
the constraint determines `Pi_L = 0`, and the Coulomb-gauge Hamiltonian is
`H_C = H - (1/2) sum_g (Pi_L^g)^2`. The electric zero-point splits exactly:
`12 * 1/8 = 1.5` longitudinal + `12 * 2 * 1/4 * 1/2 = 3` transverse +
`4 * 1/8 = 0.5` Higgs `= 5 = 40 * 1/4`. On the 44-dimensional `P = 9`
eigenspace `(1/2) sum_g (Pi_L^g)^2 = 1.5 * I` to `1e-12`, so `H_C` reproduces
those window energies shifted by exactly `1.5000000000000782`
(`49.85526760389873 -> 48.35526760389865`), and that restricted minimum lying
above `theta_0` confirms the ground state is not in the longitudinal vacuum.
Off that sector the elimination is not a constant: the ground state has
`<(1/2) sum (Pi_L)^2> = 1.5051034667046121`, so `H_C` gives it
`48.258201871141885` rather than `theta_0 - 1.5 = 48.2633053378465`. The
same-energies statement therefore holds exactly on the sector the gauge-fixed
theory keeps, and fails by `5.1e-3` where it does not apply.

**The physical band is all 44 states, and it is a block, not a spectrum of the
theory.** The same test now resolves the whole sector rather than its minimum.
`H_C` shifts every one of the 44 energies by `1.5` (worst `1.35e-13`); the band's
floor is `49.85526760389873`, which sits `0.09196226605223` *above*
`theta_0 = 49.7633053378465` — the interacting ground state buys that `0.092` by
carrying a longitudinal excess of `1.5051 > 1.5`; the band's width is
`3.0218401376237907` and its lowest excitation is `1.4664848961` with
multiplicity 4, followed by 1.738019/1.738985/1.900696/1.916485 x5/2.197540/
2.256485 x5/2.361189/2.374360 (x23)/3.021840. Two measurements stop this from
being read as the theory's transverse spectrum. `H` does not preserve the
sector: `|(1-P) H P| = 0.6234981956669964`, so Coulomb gauge cannot be imposed by
restricting states — the longitudinal mode has to be eliminated from the
operator, which is exactly what `H_C` does, and the block is the right object
only because of that construction. And the transverse zero point is not fixed on
the band: `(1/2) sum_T p^2 - 3` has norm `3.984344362627306` across the sector
(the Higgs analogue `1.224744871391589 = sqrt(3/2)`), so the spacings are not a
free transverse ladder. The seed's exact `3.0` for the transverse part is a seed
expectation, not a sector constant.

**Charge superselection is an operator statement here.** The window is a single
charge sector — all 70 states carry the seed's two matter fermions, `N = 2 * I` —
so no charge-variance assertion inside it could fail. The anchor is therefore
off-window, on probes with `N = 0..4`: `[H, N]` vanishes with relative defect
`9.059259100241536e-16`, and `H` has zero matrix element between different
sectors (20 ordered pairs, worst exactly `0.0`). This is what makes the
fixed-charge truncation `project_occupation` exact rather than approximate.

**The interaction share of the window energy.** Splitting `H` over ordered
monomial pairs by total degree is exact (checked on the seed to `1e-10`): the
free part is degree ≤ 2, including the canonical `½P²`, and the interaction part
is degree ≥ 3, which contains every gauge–matter and Yukawa coupling. On the
window ground state `E = 49.7633053378465` the interactions contribute
`2.8394046237857826` (**+5.706%**) and the Higgs/quark Yukawa block alone
`-1.4425259347888656e-2` (**-0.029%**): present, nonzero and bounded well below
the free part, within this section's finite-reference scope.
