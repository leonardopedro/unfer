# Standard-Model operator audit and current implementation state

## Continuous-momentum Cadabra2 work order

**2026-11-24.** Derive the full inner SM action between continuous momentum packets
from the field strengths, covariant Higgs derivative and chiral matter terms in
`book.tex`. Use the scalar study's Fourier normalization and Gaussian packet checks.
The independent-jet export `sm_full_nested.cdb` supplies signs, species and generator
tables for comparison. Its 1,241 expressions have no incoming/outgoing momentum
routing. The following calculation and its SM fixture are proposed.
[Manuscript](../../timepiece/book.tex) · [Scalar Cadabra2 calculation](phi4_momentum_nested.cdb)

**Executed 2026-11-24 (numerical half, scoped):** the Rust kernel in
`nested_fock_algebra/src/sm_momentum.rs` implements items 1–3 for the declared
packet family: the `2π i K` derivative symbol, the convolution measure, the
nonabelian magnetic cross term built from `su3_structure`, the gauge–Higgs
cross term from `higgs_real_generator`, and the ordered Yukawa `k−l` transfer.
The versioned fixture `docs/fixtures/sm_momentum_kernel.json` and
`fock_sirk/tests/sm_momentum_validation.rs` supply item 4's checks: nonzero
mixed entries on distinct packets, adjoints, the abelian/zero-coupling limit,
and detect reversing the transfer, omitting `2π`, flipping the nonabelian sign,
removing the Higgs cross, dropping the Yukawa adjoint, and dropping `λv⁴/4`.
The spatial/Dirac checks are present too: the free Dirac symbol in the manuscript
spin frame (left-handed is the exact negative of right-handed, both vanish at
`k=0`), the transverse curl projector (zero on a longitudinal polarization), and
the scalar-gradient `½|k|²`. The outer lift is checked on 0/1/2 quanta
(`dΓ(cI): 0, c, 2c`) and `dΓ(B²) ≠ dΓ(B)²` on a two-quantum state.
The independent numerical check is a 3D/convolution quadrature, not a
coordinate-space integral of the full action.

**Delivered 2026-11-25.** (i) The cadabra2 module `docs/sm_momentum_nested.cdb`
(8 CHECK rows, all `(0 = ok)`, registered in `scripts/check-sm-symbolic` and run
with Cadabra2 2.5.15 through `nix develop .#symbolic`): it pins the two Fourier
maps (`p = −i∂` has the *real* symbol `2πk`; `q` has `(i/2π)∂_k`), their
commutator, the second-derivative product rule (the `wall` regression), the
retained `λv⁴/4` constant with `V(s=vev²)=0`, and the antisymmetric magnetic
weight `½ΣF² = Σ_{a<b}F²`. (ii) The full 1,241-expression routing:
`nested_fock_algebra/src/sm_dictionary.rs` parses every exported row and
cross-checks it between the coordinate and momentum representations of the E1
collective-coordinate operator, with `docs/fixtures/sm_dictionary_kernel.json`
and `fock_sirk/tests/sm_dictionary_validation.rs`. All 1,241 rows agree to
roundoff (worst relative gap 1.3e-13, `wall`); a mutated `2π` is detected on
exactly the 232 rows that carry a jet momentum.

**Scope of that pass.** The dictionary kernel is the collective-coordinate
operator (`p_m = −i∂/∂q_m` on the 163 independent coordinates), not a 3D
field-mode kernel; it therefore does not certify the per-mode convolution
routing of `sm_momentum.rs`, and the Cadabra rows certify conventions rather
than the 1,241 integrals (the module does not integrate). **Still outstanding:**
the complete-operator/gauge-reduction work and a finite-window numerical
acceptance of the full operator.

1. **Declare conventions in the module.** Write the new calculation alongside
   `sm_full_nested.cdb`. Use $\hat f(k)=\int e^{-2\pi i k\cdot x}f(x)\,d^3x$,
   $\widehat{\partial_i f}(k)=2\pi i k_i\hat f(k)$ and
   $(a*b)(K)=\int a(q)b(K-q)\,d^3q$. With matrix or operator coefficients preserve
   order. For adjoints, $\widehat{f^\dagger}(q)=\hat f(-q)^\dagger$.
   Keep the integration measure explicitly in each convolution. Only internal
   indices and basis labels are summed. No derivative-constraint sector is needed.
2. **Derive and square the physical forms.** For $D=d+igTA$ the manuscript convention gives

   $$
   \widehat F^a_{ij}(K)=2\pi i\big(K_i\widehat A^a_j(K)-K_j\widehat A^a_i(K)\big)
      -g\sum_{bc} f^{abc}(\widehat A_i^b*\widehat A_j^c)(K).
   $$

   Use the structure constants in the manuscript basis for color and weak isospin.
   Hypercharge has only the curl. For the Higgs doublet derive
   $\widehat{D_i\Phi}=2\pi iK_i\hat\Phi+
   ig\,T^a(\hat W^a_i*\hat\Phi)+ig'Y(\hat B_i*\hat\Phi)$ in the specified
   realification. Form the integrated adjoint squares. Check their free quadratic,
   derivative–interaction cross and interaction-square terms independently, with the
   magnetic normalization $B_i=\epsilon_{ijk}F_{jk}/2$. Retain the canonical kinetic
   terms and the complete Higgs square in the four-real-component normalization.
   Construct its pair field by convolution. A constant transforms as a delta acting
   on a test packet. Use spatial smearing or a declared volume normalization to
   account for the vacuum-energy constant before squaring. A squared delta on
   infinite volume is undefined.
3. **Include the ordered matter action.** Derive the free spatial Dirac multiplier
   in the manuscript spin frame. For external momenta $k,l$, a gauge or Higgs
   insertion carries $k-l$ in the kernel between the CAR creation and annihilation
   legs. Check chiral/color/generation matrices, both Higgs contractions and Hermitian
   conjugates for quarks and leptons, including the right-handed neutrinos. On smeared
   packets test $h(k,l)=h(l,k)^\dagger$ in the weak sense. Export ordered leg labels
   separately from the commuting coefficients processed by mathhook. Assemble every
   term inside $h$ before the single graded lift $d\Gamma(h)$.
4. **Deliver a reproducible reference.** Emit labeled equations, a versioned kernel
   and Gaussian–Hermite matrix-element fixture, Fourier convention, parameter values
   and a terminal success marker after all assertions. Compare the Fourier results
   with an independent coordinate-space packet integral. Use distinct packet centers
   and internal indices with nonzero structure constants so the mixed terms survive.
   Check a Cartan/abelian control and the scalar free-gradient limit. Require the
   checks to detect reversing the transfer, omitting $2\pi$, flipping the nonabelian
   sign, removing a Higgs cross term and dropping a Yukawa adjoint or inner constant.
   Register the new module with the strict Cadabra verifier and compare regenerated
   fixtures before reporting a pass. These checks establish algebra and specified
   packet entries. Domain estimates and gauge reduction go to the Lean work order.
   [Formalization](../../timepiece/SM_ESA_COMPLETION_PLAN.md) ·
   [Numerical acceptance](SM_RESEARCH_PLAN.md#continuous-momentum-numerical-work-order)

## Independent-coordinate E1 audit · historical scope

2026-11-15 source/planning review. Complete E1 term assembly and joint rational
integration are implemented; the dated execution records remain in
[SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md). No fresh build, symbolic run,
numerical pass or proof replay is claimed by this revision.

The active numerical/research order is [SM_RESEARCH_PLAN.md](SM_RESEARCH_PLAN.md),
**N6–N14**. The proof counterpart is **SM0–SM9** in
[SM_ESA_COMPLETION_PLAN.md](../../timepiece/SM_ESA_COMPLETION_PLAN.md).
This supersedes the old forward P0–P8/N0–N5 ordering, not its actual results.

The Lean potential self-bound and its kinetic-control consequences are proved. Remaining
work binds the complete physical coefficient/coordinate dictionary, 43-direction comparison
graph core, quadratic-form commutator, physical constraints/metric and graded closure to
that same operator. The numerical algorithm also needs its actual joint-action/GMRES/
compressed-resolvent/Padé/rounding bridge; an abstract theorem or pure SIRK translation
is not that end-to-end proof. Separate unreduced E1, physical E1 and the E2 field limit.

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

## 2. Implemented foundation N0–N5 and remaining proof scope

| Packet | Current numerical implementation | Remaining proof boundary |
|---|---|---|
| N0 audit/labels | Legacy scopes corrected; scalar-template checks repaired and executed | Symbolic identities do not establish domains/ESA |
| N1 canonical action/bosons | `exact_inner.rs` plus full bosonic forms in `sm_full.rs`; same-mode CCR, exact ordered action and retained constants | Lean named-coordinate and physical normalization intertwining |
| N2 matter dictionary | All 96 CAR coordinates; source Majorana spin frame; full spatial/gauge coupling; all four complex generation/Yukawa matrices and h.c. | Formal reality/chirality, realification and operator assembly on the common core |
| N3 comparison/enclosure | Correct scalar comparison expression; single outer lift on joint symmetric/antisymmetric and parity-graded sectors | Full 43-direction graph/core and matrix form transfer; physical reduction/graded statistics |
| N4 Hashimoto integration | Joint registry, actual resolvents, bounded-resolvent function compression, full residuals/tails, restarted evolution and 13 source-declared joint-SM tests | Full-operator cutoff removal, roundoff certification and the paper's analytic error hypotheses |
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
| F3 obsolete uncoupled quartic comparison | `sm_comparison_n` and old symbolic prose | Use the actual bosonic `smFlN`/full comparison, reusing the proved self-bound, with full-operator transfer still required |
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
physical BRST theorem or full SM ESA proof. SM4/SM5 physical/graded-statistics
identification and SM8 infinite inner-mode/field limits remain separate proof work.
The new N6–N14 packets price finite-map error, tails and actual physical constraints.

The electric-only and matter-density charge probes are not the full joint charge; the
44-state longitudinal band leaks under H. Its Coulomb subtraction is a labeled comparison
until an operator/constraint/physical-representation equivalence is proved. Matter fermion
number is not the full electromagnetic charge. Ghost grading and an admissible physical
Krein metric must be distinguished before a no-ghost conclusion; see N10 and SM1.
Historical numerical values remain valid within their original finite-window scope.

## 5. Historical symbolic execution record

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
