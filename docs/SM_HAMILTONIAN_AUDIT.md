# Standard-Model operator audit and current implementation state

2026-11-09c. The executable E1 term assembly and joint-state rational integration
are now implemented. This supersedes the earlier N1–N4 implementation-status
paragraphs; the Lean completion work order remains P0–P8 in
[SM_ESA_COMPLETION_PLAN.md](../../timepiece/SM_ESA_COMPLETION_PLAN.md).
The detailed current operator, Majorana dictionary, symbolic import, solver
formula and commands are in [SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md).

## 1. Object of record

```text
h_SM = h_B + spatial Dirac + dynamical gauge–matter + full Higgs/Yukawa,
H_SM = dGamma_out(h_SM) = sum h_ij Cdag_i A_j.
```

Creation is left and annihilation right. All inner interactions, contractions
and the fixed Higgs constant are retained. An inner shift gives `c N_out`;
outer-vacuum annihilation requires no scalar deletion. ESA, nonnegativity and a
strict absolute one-particle edge are separate questions.

E1 uses 163 independent bosonic collective coordinates, 40 active field/Higgs
momenta and 49 squared forms. Matter has 96 independent complex CAR coordinates,
corresponding to the manuscript's 192 real Majorana components. Every generation,
color, chiral/spin component and right-handed neutrino is present; no Majorana
mass term is added. Spatial Dirac derivatives act at fixed independent collective
coordinates. NS/QG jet elimination is not used to change this SM realization.

## 2. Implementation packets

| Packet | Current numerical implementation | Remaining proof boundary |
|---|---|---|
| N0 audit/labels | Legacy scopes corrected; scalar-template checks repaired and executed | Symbolic identities do not establish domains/ESA |
| N1 canonical action/bosons | `exact_inner.rs` plus full bosonic forms in `sm_full.rs`; same-mode CCR, exact ordered action and retained constants | Lean named-coordinate and physical normalization intertwining |
| N2 matter dictionary | All 96 CAR coordinates; source Majorana spin frame; full spatial/gauge coupling; all four complex generation/Yukawa matrices and h.c. | Formal reality/chirality, realification and operator assembly on the common core |
| N3 comparison/enclosure | Correct scalar comparison expression; single outer lift on joint symmetric/antisymmetric and parity-graded sectors | Comparison inequalities/ESA and physical reduction/graded statistics |
| N4 Hashimoto integration | Joint registry, actual resolvents, bounded-resolvent function compression, full residuals/tails, restarted evolution and six full-operator tests | Full-operator cutoff removal, roundoff certification and the paper's analytic error hypotheses |
| N5 publication | Source/export fixture, symbolic gate, commands and scoped theory/numerical documentation | Future proof/certificate promotion only after the corresponding checks |

No production legacy operator has been silently changed. The new full-SM path
has an explicit separate API; its numerical tests do not call the old scalar or
six-mode SM builders. They bind nonzero couplings and fully populated complex
three-generation matrices, then approximate the same complete nested operator.

## 3. Historical findings and migration rule

These findings remain relevant to callers of the legacy APIs:

| Finding | Source | Correct treatment |
|---|---|---|
| F1 independent field/momentum modes | Old Higgs/reduced builders | Comparison models; use same-mode canonical operators for E1 |
| F2 contractions and CAR swap signs discarded | Legacy CAS normal ordering | Use ordered joint action; a quartic loses operator terms as well as constants |
| F3 obsolete uncoupled quartic comparison | `sm_comparison_n` and old symbolic prose | Use the actual bosonic `smFlN`/full comparison, with new analytic bounds still required |
| F4 incomplete scalar assembly/tautological normalization | Old SMGF CHECK 9/11 | Repaired template checks; the complete chiral dictionary is the new full module |
| F5 unsupported radial/gauge interpretation | Scalar Higgs-mechanism comparisons | No full-SM EWSB or radial-reduction claim |
| F6 Ritz threshold mistaken for lower bound | Old finite-window tests | Variational upper/residual information only |
| F7 separate bosonic/fermionic outer universes | Legacy state type | Use a joint inner label before outer enclosure |
| F8 supplied masses/free inventories | Ancillary tests | Inputs/benchmarks, not derived full-SM predictions |
| F9 products of already-lifted inner letters | `Hamiltonian::apply` | `dGamma(B)dGamma(C)` differs from `dGamma(BC)`; complete the inner product first |

An empty legacy operator word is outer identity; enclosing inner identity is
outer particle number. The regression reproduces both F9 distinctions through
the actual legacy API. Restoring a single scalar does not repair F1/F2/F7/F9.

## 4. Symbolic pipeline and numerical scope

`sm_full_nested.cdb` independently constructs all components and exports 1,241
LaTeX expressions: 40 momenta, 49 forms and 1,152 symbolic matter entries. The
mathhook bridge imports commuting coefficients, rejects unsupported/noncommuting
expressions and checks every component against the native dictionary before
building the numerical operator. All spinor kinetic blocks are additionally
checked against the fixed integer Majorana matrices in the manuscript/Lean source.

The rational solver implements the paper's function of the compressed bounded
resolvent, not just an exponential of a Hamiltonian Ritz matrix. Its finite
occupation/basis windows are explicit projections of the complete operator;
full images and tails remain available. Complete-image caching accelerates
these declared windows without removing terms. Actual sparse vectors are
reorthogonalized after Gram whitening to avoid coordinate cancellation.

Full Ritz and shifted-solve residuals include outside-window contributions.
Restart diagnostics retain norms, energies and solve/function/projection error
data. No finite result is promoted to a mass gap, EWSB, continuum construction,
physical BRST theorem or full SM ESA proof. E2 infinite inner matter modes and
P5 physical/graded-statistics identification remain separate proof work.

## 5. Symbolic execution record

Cadabra2 2.5.14 was actually run. The complete module's Legendre, all-entry
Hermiticity and 96-coordinate coverage checks pass, and its fresh LaTeX output
matches the checked fixture exactly. SMGF has 19 CHECK rows; the comparison
module has 49. All zero identities pass; its historical `c0-1` row remains an
explicit side condition, not a derived positivity theorem.

Runtime fixes include the multiline Python substitution in SMGF, sorting its
constant cancellation, and the missing constant-derivative rules for `c0` in
the comparison template. The old quartic candidate is still not a valid live
Faris–Lavine comparison; raw scalar zeros do not repair that analytic obstruction.

The locked dependency cache, including `ahash`, has been provisioned. The
Cargo validation record is maintained in the implementation account. No Lean
compilation or new proof/certificate export is performed in this wave.
