# Verification: transformed-operator ESA in three variable systems — reciprocal chart, Lagrangian, densitized

This is the evidence base for `docs/transformed_esa_charts.cdb`, the
c-number / formal-operator bookkeeping behind the work-order entry
"closing the ESA of the TRANSFORMED operator" in
`../timepiece/CONSOLIDATED_PLAN.md` (2026-10-15) and the reader-facing
doctrine page `../test/mathematical-foundations/ode-singularity.md` §7.
The module pins the algebra that each of the three ESA arguments
consumes when it changes variables until the flow is complete and runs on
the **transformed** operator:

| system | change of variables | transformed operator | CHECKs |
|---|---|---|---|
| ODE `x' = x²` | reciprocal chart `w = −1/x` | translation `ẇ = 1` (complete flow) | 1–6 |
| NS (Lagrangian) | material map `Ψ(ξ)` | shifted harmonic oscillator | 7 |
| QG (densitized) | `e = y²`, `𝒮 = y𝒮̃`, `𝒫 = y𝒫̃` | flat principal part | 8–9 |

Reproducible with Cadabra2 (2.5.15 on `PATH` on this machine):

    nix-shell -p cadabra2 --run "cadabra2-cli -q -n docs/transformed_esa_charts.cdb"
    bash scripts/verify_cdb_modules.sh        # certifies every row reaches 0

Numerical companions (always run, no engine needed):

    cargo test -p fock_sirk --test transformed_esa_charts_validation
    cargo test -p fock_sirk --test qg_half_density_stone_flow_validation
    cargo test -p fock_sirk --test ns_penalty_lyapunov_validation

The first mirrors CHECK 1–9; the second is the row-level companion of
QG-D-C.2–3 (the half-density unitary Jacobian — norm preservation,
measure pushforward, multiplication intertwining — and the `qgR2_stone_flow`
mode flow: the confV completed square, the mode-symbol bounds, the Stone
group law/norm/generator, and the deficiency resolvent estimate); the third
is the row-level companion of NS-L-C.3 (penalty non-negativity and its
zero set, the energy and Liouville identities, the pointwise Lyapunov bound,
the form identity `N = H² + E ≥ 1`, the exact commutation relation, and the
lift `dΓ(N_L) + 1` on the Fock sectors m = 0,1,2 with its resolvent
estimate off the real axis).  The two row-level suites are finite shadows
of the analytic rows, not proofs of them.

## What the module checks

Every CHECK differences two independently assembled sides; a mistyped
chart sign, vector field, commutator, or coefficient fails its row.

* **CHECK 1 — the classical solution solves `ẋ = x²`.** `x(t) = x0/(1 − t x0)`
  through the quotient rule (`u'v − uv'` with `u = x0` constant,
  `v' = −x0` typed independently) against the ODE right side `x(t)²`,
  cleared against `v²`. *(Numerical mirror:
  `xf_classical_solution_solves_ode`, central differences at fractions of
  the blow-up time.)*
* **CHECK 2 — the chain rule completes the flow.** `ẇ = (dw/dx)·ẋ =
  (1/x²)·x² = 1`: the cleared chart velocity numerator equals its
  denominator. The chart turns the singular vector field into the
  constant field. *(Mirror: `xf_chain_rule_completes_flow`.)*
* **CHECK 3 — the chart of the flow is a translation.** `w(t) = −1/x(t)`
  equals `w0 + t`, cross-multiplied against `x0`: the trajectory that
  blows up in `x` is a straight line in `w` — the algebraic face of
  "complete the flow". *(Mirror: `xf_chart_of_flow_is_translation`, exact
  to round-off past the blow-up time.)*
* **CHECK 4 — the Weyl commutator correction.** With `xp = px + i`
  (rewritten inline by the engine, two passes), `x²p = px² + 2ix`:
  the normal-ordering correction of the ODE Hamiltonian. *(Mirror:
  `xf_weyl_commutator_correction`, `p = −i d/dx` by central differences
  on a Gaussian.)*
* **CHECK 5 — the two spellings of `H` coincide.** `½(px² + x²p) = H` with
  `H = x²p − ix`, cleared by 2 as `(px² + x²p) − 2(x²p − ix) = 0` under
  the same commutator. *(Mirror: `xf_weyl_sum_equals_h`.)* **Note:** an
  earlier label of this row read "= 2H"; the algebra was always the
  cleared identity above and the tests caught the mislabel (a factor 2),
  fixed in both files on 2026-10-15.
* **CHECK 6 — the chartW norm kernel.** `(chartW ψ)(x) = x⁻¹ ψ(−1/x)`
  carries the norm kernel `1/x²`; the change of variables carries the
  Jacobian `x²`; their product is 1 — the symbolic content of
  `chartW_lintegral_normSq`. *(Mirror: `xf_chartw_norm_preservation`, both
  norms integrated independently; the `x`-side through a tan
  compactification because its tails decay only like `1/x²`.)*
* **CHECK 7 — the Lagrangian completed square.** `½P² + νQ² + fP =
  ½(P+f)² + νQ² − ½f²`: the transformed one-parcel operator is a shifted
  harmonic oscillator (complete flow). Same identity as CHECK 1 of
  `faris_lavine_n_ns_lagrangian.cdb` in the `N = h + ½|f|²` spelling; this
  row keeps the three-system table self-contained. *(Mirror:
  `xf_lagrangian_completed_square`, random states, 1e-12 class.)*
* **CHECK 8 — the densitized inverse identity.** `1/e = 4(dy/de)²` at
  `e = y²`: the honest absorption identity of
  `inv_eq_four_mul_deriv_densY_sq` **with its factor 4** (the plan's older
  prose `1/e = (dy/de)²` is wrong by that factor). The physical side
  carries the independent symbol `esing`; the change of variables is the
  final substitution, so a wrong substitution fails the row.
  *(Mirror: `xf_densitized_inverse_identity`.)*
* **CHECK 9 — the kinetic absorption.** `(1/16e)𝒮² = (1/16)(𝒮/y)²` at
  `e = y²`: the flat principal part of the transformed operator. Same
  wiring as CHECK 8 (independent `esing`, substitution last).
  *(Mirror: `xf_kinetic_absorption`.)*

## Counterfactuals (what a wrong input does)

The rows are not tautologies; each assembles its two sides from
independently typed pieces:

* flipping the chart sign (`dw/dx → −1/x²`) fails CHECK 2;
* mistyping `v' = −x0` as `+x0` fails CHECK 1;
* dropping the commutator correction fails CHECK 4/5 (they reduce to
  `−2ix` and `−2ix` respectively, not 0);
* replacing the substitution `e → y²` by `e → y` (or removing it) leaves
  `4y² − 4e ≠ 0` in CHECK 8 and `16S²(y² − e) ≠ 0` in CHECK 9;
* changing the kernel power `1/x² → 1/|x|²`-style variants that are not
  `1/x²` fails CHECK 6.

(The algebraic F1 lesson of `CDB_MODULE_REVIEW.md` is respected: no row
is a diff of a hand-typed tautology — CHECK 8/9 are wired through the
change of variables itself, CHECK 4/5 through the commutator rewriting.)

## Engine notes learned here (cf. `CADABRA2_GOTCHAS.md`)

* `substitute` with a non-commuting rule (`$xop pop -> pop xop + i$`) does
  **not** iterate inside its own expansion in one call even with
  `repeat=True`: the substituted sum must be `distribute`d before the next
  pass. CHECK 4/5 use two substitute/distribute passes; one pass leaves a
  `xop pop xop` residue.
* `canonicalise` does **not** sort a `Symbol` past another `Symbol`, and
  does not move `ImaginaryI` past a `NonCommuting` symbol. `sort_product`
  does both — it is what closes CHECK 4/5 (the `xop i` vs `i xop`
  ordering) and CHECK 7 (the `ffor pcan` vs `pcan ffor` ordering).

## Honest boundary

The module is c-number / formal-operator bookkeeping only. It proves no
analytic statement: ESA itself is the Lean work order
(`../timepiece/CONSOLIDATED_PLAN.md` 2026-10-15 rows ODE-C.1–3,
NS-L-C.1–3, QG-D-C.1–3), consuming `ChapterOdeUnitaryFlow`,
`ChapterNavierStokesLagrangianEsa`, and
`ChapterQuantumGravityHalfDensity`/`ChapterQuantumGravityDensitized`.
Nothing here claims ESA of the *original* variables (`x`, Eulerian,
`1/e`) — the doctrine is that those verdicts are the wrong question.
