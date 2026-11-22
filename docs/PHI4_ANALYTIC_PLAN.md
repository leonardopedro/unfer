# Real scalar λφ⁴ — analytical and Cadabra2 research plan

The primary analytic target is the local scalar operator on continuous field and jet
coordinates. Existing one- and two-site calculations are periodic lattice comparisons.
Their return intervals apply to their specified finite-difference operators.

[Lean contract](../../timepiece/PHI4_FORMALIZATION_PLAN.md) ·
[Numerical experiments](PHI4_NUMERICAL_PLAN.md)

## From a local field to an inner Hamiltonian

Use continuous time and three spatial dimensions. Write U(s) = λ(s² − v²)²/4
with λ,v > 0. Starting from the Lagrangian density
½(∂ₜφ)² − ½|∇φ|² − U(φ), retain the whole potential in the Legendre transform.
In the manuscript scalar realization the inner operator is
h = pₛ²/2 + (j₁² + j₂² + j₃²)/2 + U(s) on L²(ℝ⁴). Only the scalar carries a
momentum. Its whole action is enclosed as H = dΓ(h). The indices in a matrix
expansion of dΓ(h) label one-particle basis functions of these continuous variables.

At unit width and λ = v = 1, act on the Gaussian (constant Mehler polynomial).
The field terms contribute 5/8, 7/8 and 1/4 at scalar degrees 0, 2 and 4.
Each jet contributes (1 + He₂(jᵣ))/2. The full column has mean 17/8 and
squared residual (7/8)²·2! + (1/4)²·4! + 3·(1/2)²·2! = 145/32 for a unit seed.
These analytic values are the first continuous-coordinate solver reference.

## Periodic lattice comparison

On a periodic grid with M sites, spacing a and volume Ma³, the momentum conjugate
to φₓ is a³∂ₜφₓ. In the variables $Q_x=a^{3/2}\phi_x$ and $P_x=a^{3/2}\dot\phi_x$ the result is

$$
h=\frac12\sum_x P_x^2
  +\frac{1}{2a^2}\sum_{x,r}(Q_{x+\hat r}-Q_x)^2
  +\frac{\lambda}{4a^3}\sum_x(Q_x^2-a^3v^2)^2 .
$$

$$
H=d\Gamma(h)=\sum_{i,j}\langle e_i,h e_j\rangle\,C^\dagger(e_i)A(e_j).
$$

The nonnegative square fixes the energy zero. Expanding it gives a constant
λMa³v⁴/4. Dropping this inner constant shifts every outer sector in proportion to
its particle number. The global reflection (φ,π) ↦ (−φ,−π) preserves the Hamiltonian.
Expansion about +v gives U(v+η) = λv²η² + λvη³ + λη⁴/4. The cubic coefficient changes
sign about −v. These checks distinguish the fully interacting wall from its quadratic
small-oscillation approximation. Lean also defines a scalar instance of the manuscript’s
independent-jet density with three jet multipliers and a single field momentum. Label any such
calculation explicitly. In the grid model the differences above determine the gradient.

## Derivations and checks

| Step | Calculation and independent check | Acceptance |
|---|---|---|
| A1. Legendre transform | Differentiate the discrete Lagrangian with respect to each velocity, solve for it and substitute. Derive the discrete force again by varying the link energy. | Recover the kinetic coefficient, the periodic Laplacian and the local force −λφₓ(φₓ²−v²). A constant field at either minimum has zero classical energy at rest. |
| A2. Momentum vertex | Use $Q_x=M^{-1/2}\sum_k e^{ik\cdot x}\widehat Q_k$, with Q̂₋ₖ = Q̂ₖ† and the matching rule for P. Handle zero and self-conjugate modes once. | The quadratic gradient has eigenvalue 4 Σᵣ sin²(kᵣa/2)/a². The quartic vertex has coefficient λ/(4a³M) per ordered tuple with k₁+k₂+k₃+k₄ = 0 modulo the reciprocal grid. |
| A3. Two-site test | Use the periodic 2×1×1 cell and Q₁,₂ = (u ± w)/√2. Sum the two local fourth powers independently. | Recover (u⁴ + 6u²w² + w⁴)/2. The mixed term tests coupling between different modes. The two oriented links along the active axis both contribute. |
| A4. Ordered action | At unit width put Q = b†+b, P = i(b†−b)/2 on the same mode. Normal order using the CCR. | Obtain Q⁴ = :Q⁴: + 6:Q²: + 3I and the complete one-site vacuum column specified in the Lean plan. Test a boundary state for which a word leaves an occupation window and returns. |
| A5. Outer enclosure | Apply the indexed inner sum on zero, one and two outer particles. Evaluate the inner identity and a product before lifting. | On two particles an inner shift cI gives 2cI. Multiplying two separately lifted factors produces cross-particle terms. Include an input on which these terms survive. |

A2 follows by orthogonality of the discrete Fourier characters in a **local** product.
The selection rule conserves total lattice momentum. No independent derivative field
or prescription for a nonlinear spatial-derivative product is needed. Use real
sine/cosine coordinates or a conjugation-aware complex implementation in executable
work. Independent oscillators for both members of every ±k pair would double the field.

## Executed Mehler and SIRK examples

The Cadabra2 process checks the two-site quartic by native product distribution.
Its embedded SymPy code differentiates the jet and site actions, applies ordered ladders
to polynomials and computes exact rational matrices. The four-site ring check enumerates
all 256 ordered momentum tuples including the zero and self-conjugate modes. A general
grid Fourier derivation is still needed for a symbolic identity at arbitrary grid shape.
[Executable derivation](phi4_mehler.cdb) · [Exact fixture](fixtures/phi4_mehler.json)

**One occupied outer excitation, one site.** In the unit-width Mehler convention
the Gaussian represents the constant polynomial. The first three nonzero components
of its full image are 5/8, 7/8 and 1/4 at Hermite levels 0, 2 and 4. On the odd
seed He₁ the coefficients are 23/8, 15/8 and 1/4 at levels 1, 3 and 5. In the
one-dimensional vacuum window the rational generator is −5i/8 at every positive shift.
Its trajectory has return probability one inside the window. Applying the full h gives
squared defect (7/8)²2! + (1/4)²4! = 97/32 for a normalized seed. Integrating this
residual gives the Lean-certified lower bound 1 − (97/32)t² on the exact return probability.
The empty outer vacuum is a different initial state. The ordered outer action checks
also give shifts 0, c, 2c on zero, one and two outer particles.
[Lean error and observable](../../timepiece/BookProof/Phi4/ChapterScalarSirk.lean)

**One occupied outer excitation, two sites.** At a = λ = v = 1 retain the unnormalized
product Hermites (0,0), (2,0), (0,2) of the uniform and staggered modes. Their Gram
matrix is diag(1,2,2). The two rational columns of V are the Gaussian seed and its
shift-5 solve. With Hₚ the three-dimensional compression, set R = (3I + iHₚ)⁻¹.
With W the Hermite Gram matrix and † the conjugate transpose, the reduced operators are

$$
V^+=(V^\dagger W V)^{-1}V^\dagger W,\qquad B=V^+ R V,\qquad G=3I-B^{-1}.
$$
The executable check distinguishes G from −iV⁺HₚV. In the full action the seed alone
has squared norm 3 outside the window. Computing all output Hermites through degree
six exhausts hV exactly because the largest input degree is two.

The fixture records the complete complex generator. For D = iVG − hV its squared
Hilbert–Schmidt norm in the physical metrics is
66577730826550897732871/259668494406269661228. The script checks the dissipation
identity $G^\dagger W_2 + W_2G = -6L^\dagger W L$, with
$W_2=V^\dagger WV$ and $L=RVB^{-1}-V$. Thus the forward
reduced evolution is contractive. Integrating the defect of V exp(tG) against the
unitary inner evolution bounds the state error by t‖D‖HS for a normalized retained
seed and t ≥ 0. This two-site bound uses the exact symbolic matrix calculation and
the analytic integral argument. Lean checks the rank-two matrices, their physical
Hermite Gram weights, and the squared rates used below.
[Rank-two Lean certificate](../../timepiece/BookProof/Phi4/ChapterScalarRankTwo.lean)

The orthogonal split of D locates the largest error in this example. Its squared
Hilbert–Schmidt norm is about 3.125 inside the three-mode window and 253.270
outside it. On the initial seed the corresponding squared norms are about 0.166
and exactly 3. Increasing the Krylov rank within these same three modes can address
the first contribution. It cannot recover the missing rows of the first full action.
`defect_split` checks both contributions on the same basis and trajectory.
[Exact split](fixtures/phi4_mehler.json)

### A two-site return probability below one

Take the unit Gaussian seed e in one occupied outer excitation and t = 1/100. Write
y(s) = V exp(sG)e₀ for the reconstructed state and U(s)e for the exact evolution. In
the following norms the Gaussian has norm one. The Gram identity makes exp(sG)
contractive for s ≥ 0. Consequently the full residual satisfies
‖D exp(sG)e₀‖ ≤ ‖D e₀‖ + s‖D‖HS‖G e₀‖. Integrating the residual against U(t−s) gives

$$
\|U(t)e-y(t)\| \le t\|De_0\|+
  \tfrac{t^2}{2}\|D\|_{\mathrm{HS}}\|Ge_0\|
  \le 2203729/100000000 .
$$

To see what this error allows us to resolve, project away the initial Gaussian ray
with Q = I − |e⟩⟨e|. Twice integrating y′′ and using contractivity gives
‖Qy(t)‖ ≥ t‖QVG e₀‖ − t²‖G²e₀‖/2. The exact squared rates in the fixture give
‖QVG e₀‖ ≥ 4.17 and ‖G²e₀‖ ≤ 88.51. Subtracting the full state error leaves
‖QU(t)e‖ ≥ 1523721/100000000 ≥ 3/200. By unitarity the return probability is at most
1 − (3/200)². For the other endpoint, the one-column phase at energy 13/4 has full
seed variance 337/16. Its squared state-error bound gives return at least
1 − (337/16)t² by the overlap inequality. Hence

$$
\frac{159663}{160000} \le |\langle e,U(1/100)e\rangle|^2
  \le \frac{39991}{40000}.
$$

The interval certifies departure from the initial ray for this unit-parameter model.
Lean proves the affine-in-time residual estimate for a differentiable scalar-core orbit
in `Model.orbit_error`. The specialization of its hypotheses to this contractive matrix
exponential and the final projection argument are analytic. The fixture checks their arithmetic. A Rust comparison
can use the endpoints before sweeping λ or changing the window.

### Reproduce the fixture

From the unfer root run the following. The comparison requires the fixture marker
after all checks, so an interrupted engine run fails even if the CLI exits successfully.

```bash
cadabra2-cli -q -n docs/phi4_mehler.cdb > /tmp/phi4-mehler.log
python3 - <<'PYCODE'
import json
from pathlib import Path
rows = Path("/tmp/phi4-mehler.log").read_text().splitlines()
checks = [s for s in rows if s.startswith("PHI4_CHECK|")]
assert len(set(checks)) == len(checks) == 29 and all(s.endswith("|0") for s in checks)
[result] = [s.split("|", 1)[1] for s in rows if s.startswith("PHI4_FIXTURE|")]
assert json.loads(result) == json.loads(Path("docs/fixtures/phi4_mehler.json").read_text())
PYCODE
```

Cadabra2 CLI reported 2.5.15 private (Nix derivation 2.5.14-p1). Four deliberate
changes produced exact nonzero differences. Dropping the mixed vertex gave 3,
changing a link difference to a sum gave −4, omitting the double contraction gave 3,
and deleting the energy constant gave 1/4. The script SHA-256 is
`f980c5acc06af12b0453ebbdb1811a952633c9124c2057c55930057c2f3f59a3`.
The formatted fixture SHA-256 is
`3694921d365de9d7a63a3721079f7f72349078cd19963ceae1fecb855c6ba48f`.

## Next analytical work

1. Build the full polynomial columns and shifted-solve reference for the continuous
   four-coordinate jet model. Check the Gaussian column above and an occupied jet
   seed, the Gram weights, and the residual outside the retained Hermites. Integrate
   that residual for the actual reduced orbit before quoting a return interval.
2. For the periodic comparison, extend the orthogonal-coordinate and character derivations to general periodic grids.
   Preserve mode reality, oriented link counts and the inner constant. **The two-site
   rotation is now connected to the polynomial core in Lean**: `twoSite_innerHam`
   identifies the actual `2×1×1` site operator with the uniform/staggered mode action
   on every seed, and `twoSite_column_00`/`_20`/`_02` reproduce the three exact fixture
   columns through degree six (`timepiece/BookProof/Phi4/ChapterScalarTwoSite.lean`).
   The character substitution itself is already formalized for arbitrary dimension in
   `BookProof.ChapterQuadraticRotationEsa`: `rotPoly` is the orthogonal change, and
   `pderiv_rotPoly`, `rotPoly_mulXPoly`, `rotPoly_momPoly` and `inner_pgLp_rotPoly`
   give the chain rules, canonical-pair transport and Gaussian unitarity. A concrete
   real Fourier matrix is now implemented as `periodic_fourier_1d` in
   `unfer/nested_fock_algebra/src/phi4_scalar.rs` and checked orthogonal for several
   cycle lengths. The Lean quartic-vertex transport and its momentum-conserving
   selection rules remain open.
3. For that comparison, read the fixture through the scalar Rust builder and compare exact full columns
   before varying the occupation cutoff or Krylov rank. At fixed window vary only rank,
   then at fixed rank vary only window. Report bounds for discarded rows and the reduced
   generator separately, together with the computed return probability.
4. For a fixed seed and time interval, integrate the defect on the reduced orbit, enclose
   arithmetic and compare with the signal produced by varying λ. A uniform bound across
   scalar model sizes needs an embedding and an observable that survives that change.

The legacy one-real-component Higgs calculation and the N12 independent-mode pilot
provide comparisons with different scalar dictionaries. The new exact fixture can
distinguish a missing mixed vertex or contraction in a future joint solver run.
[Higgs comparison](../fock_sirk/tests/sm_higgs_mechanism_sirk.rs) ·
[Numerical program](PHI4_NUMERICAL_PLAN.md)
