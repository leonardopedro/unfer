# Verification — the Faris–Lavine comparison operator `N` and its commutation conditions (NS, QG)

*2026‑09‑19.*  Two new Cadabra modules define the positive comparison operator `N` of the
Faris–Lavine commutator criterion and verify the symbolic content of the commutation conditions
for the Navier–Stokes and quantum‑gravity Hamiltonians used by `../timepiece`:

| module | theory | what it certifies |
| :-- | :-- | :-- |
| `docs/ns_pressure_constraint.cdb` | constraint nature of the pressure equation | second-class certificate: `p` is determined by `Δ` inversion (CHECK 2), the pair bracket is `|k|²` invertible on the non-zero modes (CHECK 3), book.tex's `Ω = (∂_j u_j)ψ†` nilpotency is cosmetic (CHECK 4), and the elimination removes `p` (CHECK 5) |
| `docs/ns_pressure_poisson.cdb` | pressure Poisson equation | derives `Lap p = −div[(u·∇)u] + div f` from `div(momentum eq)` + incompressibility: Clairaut, the Leibniz split, and the constraint cancellations |
| `docs/ns_kvn_equation.cdb` | NS *Hamiltonian* (KvN/Koopman generator of book.tex 4186) | the defining commutator `i[H,u_k] = 2F_k` (the NS drift, with the pressure gradient `q_k` and the force `f_k`) at the one‑body and outer‑Fock levels, the auxiliary‑operator contrast, the Fourier split `ν\|k\|²u_k + i(k·u)u_k`, and the pressure/incompressibility checks (CHECK 8) |
| `docs/faris_lavine_n_ns.cdb` | NS **auxiliary** sum‑of‑squares operator / gauge‑fixed parcels | `N = −Δ + ¼‖x‖²`, the shape of `[H,N]` (`commPoly_eq`), `commConst`, the AM‑GM/Young constants; **Part B** the oscillator core (CHECK 5a–5d) and the lift `dΓ(N)` identities (CHECK 6a–6b) |
| `docs/faris_lavine_n_qg.cdb` | QG scalaron + vielbein (outer Fock) | `N = −∂²_φ + φ²/4 + V(φ) + σ` with the wall *inside*, `[H_fib,N]=0`, `[A,H_fib]=0`, `[φ,H_fib]=2∂_φ`, the c = 6K arithmetic; **Part B** the fibre core as *oscillator plus non‑negative multiplication* (CHECK 10a–10d) and the lift `dΓ(N)` identities (CHECK 11a–11b) |

They are the Faris–Lavine counterpart of the existing Hamiltonian modules
(`docs/yang_mills_hamiltonian.cdb`, `docs/qg_gauge_fixed_hamiltonian.cdb`, …): those certify
*which* one‑particle Hamiltonian enters the proofs, these certify the **comparison operator and
the commutator** the criterion consumes.

## Why these exist

`BookProof/ChapterFarisLavineCore` proves Theorem 1 of Faris–Lavine (1974): a Hermitian `H` and a
positive self‑adjoint `N`, with `H` defined on `𝒟(N)`, `N + 1` onto, and the **commutator‑form
bound** `|commForm H N u| ≤ c · quadForm N u` on a common dense core, give essential
self‑adjointness.  In `BookProof/ChapterSqSumFarisLavine` the bound is proved for
`H = ½Σ_j κ_j π_j² + ½Σ_r L_r²` against the oscillator `N₁ = −Δ + ‖x‖²/4`, and in
`BookProof/ChapterScalaronOuterFockFL` for `H = secDiag + secA + secB` against the lifted fibre
operator `N = ⊕_a Friedrichs(−∂²_φ + φ²/4 + V(φ) + σ_a)` (wall inside).  These modules re‑derive
the one‑particle algebra those proofs rest on, independently and symbolically.

## `docs/faris_lavine_n_ns.cdb` — checks and output

Run: `cadabra2-cli -q -n docs/faris_lavine_n_ns.cdb` (`D = 2`; no step uses the dimension).

```
CHECK 1  NS  commutator [H,N] has the Faris-Lavine shape (0 = ok) : 0
CHECK 2a NS  d1 V = grad_1                        (0 = ok) : 0
CHECK 2b NS  d2 V = grad_2                        (0 = ok) : 0
CHECK 2c NS  d1^2 V = sum_r (v_r1)^2             (0 = ok) : 0
CHECK 2d NS  d2^2 V = sum_r (v_r2)^2             (0 = ok) : 0
CHECK 3  NS  commConst = -1/4 sum kappa + sum v^2        : -1/4 k1 - 1/4 k2 + v11²+v12²+v21²+v22²
CHECK 4a NS  a^2 + b^2/4 - a b = (a - b/2)^2       (0 = ok) : 0
CHECK 4b NS  g^2/(2M) + 2M a^2 - 2 g a = (g-2Ma)^2/(2M) (0 = ok) : 0
CHECK 5a NS  adag a = H_osc - 1/2                       (0 = ok) : 0
CHECK 5b NS  [a, adag] = 1                              (0 = ok) : 0
CHECK 5c NS  [adag a, a] = -a                           (0 = ok) : 0
CHECK 5d NS  [adag a, adag] = adag                      (0 = ok) : 0
CHECK 6a NS  [dGamma N, adag_2] = N_12 adag_1 + N_22 adag_2  (0 = ok) : 0
CHECK 6b NS  [dGamma N, a_2] = -(N_21 a_1 + N_22 a_2)         (0 = ok) : 0
```

* **CHECK 1** computes `[H,N]φ = H(Nφ) − N(Hφ)` by the Leibniz rule (`product_rule` loop, with
  the coordinate derivatives `∂_j x_k = δ_{jk}` and the constant coefficients differentiated
  away) and compares it against the claimed shape.  The result is the one‑particle statement of
  `BookProof.ChapterSqSumFarisLavine.commPoly_eq`:
  `[H,N] = (−¼Σ_j κ_j + Σ_rΣ_k v_rk²) + Σ_j (−κ_j/2) x_j ∂_j + 2 Σ_j (∂_j V) ∂_j`.  It is the
  reason the commutator‑form bound is a statement about the *symbol* — the commutator is again
  first order, with the gradient of the potential as its coefficient.
* **CHECK 2** verifies the potential data of the Hamiltonian: for `V = ½Σ_r L_r²`, `L_r = Σ_i v_ri x_i`,
  `∂_j V = grad_j = Σ_r v_rj L_r` (the Lean `gradPoly`) and `∂_j² V = Σ_r v_rj²` (the entries of
  `commConst`).  This pins the module's `commConst = −¼Σ_j κ_j + Σ_rΣ_k v_rk²` to the Lean
  `BookProof.ChapterSqSumFarisLavine.commConst`.
* **CHECK 4** verifies the two elementary dominance identities behind the constant
  `c = km/2 + 2M` (`commForm_sqSumOp_le`): the AM‑GM `a b ≤ a² + b²/4` (difference
  `(a−b/2)² ≥ 0`), used termwise for the signature contribution `Σ|κ_j|/2 · b_j a_j`, and Young's
  inequality `2 g a ≤ g²/(2M) + 2M a²`, used together with the Schur bound
  `Σ_j (∂_j V)² ≤ M²‖x‖²` (`sum_gradFun_sq_le_of_schur`) for the gradient contribution.
* **CHECK 5a–5d — the core (obligation (i) below).**  With `a = ∂_x + x/2`, `a† = −∂_x + x/2`:
  `a†a = H_osc − 1/2`, the CCR `[a,a†] = 1`, and the ladder `[a†a,a] = −a`, `[a†a,a†] = a†`.
  (a)+(b) exhibit the ground state `aφ₀ = 0` (`φ₀ = e^{−x²/4}`); (b)+(c) make the Hermite
  functions `a†ⁿφ₀` the eigenvectors of `H_osc` with eigenvalues `n + 1/2`, *arithmetic* in `n`.
  On that core `H_osc` is diagonal with linear growth, so every core vector is an analytic
  (Nelson) vector and the core is dense in the graph norm: `H_osc` is ESA on the Hermite core.
  This is the symbolic content of `harmonicOsc_essentiallySelfAdjoint` /
  `oscillator_essentiallySelfAdjoint_on_hermiteCore`.
* **CHECK 6a–6b — the lift.**  The outer comparison is not `N` but its second quantization
  `dΓ(N) = Σ_{i,j} N_ij a†_i a_j`, and the checks verify that it acts **one‑particle‑wise**,
  `[dΓ(N), a†_k] = Σ_i N_ik a†_i`, `[dΓ(N), a_k] = −Σ_i N_ki a_i`.  This is exactly the
  identity that makes the lifted core the finite‑particle **tensor** core built from the
  one‑particle core `C₀` — and hence what a core‑transfer proof of obligation (ii) must consume;
  it also makes explicit that the lifted comparison is a *different operator* from `N`.

## `docs/faris_lavine_n_qg.cdb` — checks and output

Run: `cadabra2-cli -q -n docs/faris_lavine_n_qg.cdb`.

```
CHECK 1  QG  fibre self-commutator [H_fib, N] = 0      (0 = ok) : 0
CHECK 2  QG  [A, H_fib] = 0 for constant A              (0 = ok) : 0
CHECK 3  QG  [phi, H_fib] = 2 d/dphi                    (0 = ok) : 0
CHECK 4  QG  integrand minus |d psi|^2          (>= 0) : ¼φ²ψ² + V ψ² + σ ψ²
CHECK 5  QG  integrand minus (V+sigma)|psi|^2   (>= 0) : (∂ψ)² + ¼φ²ψ²
CHECK 6  QG  integrand minus (phi^2/4)|psi|^2   (>= 0) : (∂ψ)² + V ψ² + σ ψ²
CHECK 7  QG  (1/2)(w f^2 + w g^2) - w f g = (1/2) w (f-g)^2  (0 = ok) : 0
CHECK 8  QG  2((1/2)K + (9/4)K) = 11/2 K              (0 = ok) : 0
CHECK 9  QG  6K - 11/2 K = 1/2 K >= 0                 (0 = ok) : 0
CHECK 10a QG  adag a = H_osc - 1/2                    (0 = ok) : 0
CHECK 10b QG  [a, adag] = 1                           (0 = ok) : 0
CHECK 10c QG  N_a = (adag a + 1/2) + (V + sigma)      (0 = ok) : 0
CHECK 10d QG  (sigma - 1) + 1 = sigma                 (0 = ok) : 0
CHECK 11a QG  [dGamma N, adag_2] = N_12 adag_1 + N_22 adag_2 (0 = ok) : 0
CHECK 11b QG  [dGamma N, a_2] = -(N_21 a_1 + N_22 a_2)       (0 = ok) : 0
```

* **CHECK 1** is the fibre part: with the wall *inside* `N`, the fibrewise scalaron operator
  commutes with itself, `[H_fib,N] = 0` — the `H = N`, `c = 0` case
  (`BookProof.ScalaronOuterFockFL.secHam_commForm_le`, the first summand `h1`).
* **CHECK 2** the vielbein self‑interaction `A` is a constant matrix; multiplication by a number
  commutes with `−∂²_φ + φ²/4 + V + σ` (`imA_le`).
* **CHECK 3** is the substantive one: the scalaron–vielbein coupling `B φ` does **not** commute
  with the wall.  The kinetic term gives the commutator identity
  `[φ, H_fib] = 2 ∂_φ` (`BookProof.ScalaronFiberFL.ham_x_comm_cc`) — the derivative term which
  `imB_le` controls uniformly in the wall (the wall does not have to be `N`‑bounded as a
  perturbation).
* **CHECK 4–6** the payoff of the wall being inside `N`: the quadratic form of the fibre operator,
  `quadForm(N) = ∫(|∂ψ|² + (φ²/4 + V + σ)|ψ|²)`, dominates each piece — `‖∂ψ‖² ≤ q`,
  `‖ψ‖² ≤ q` (σ ≥ 1), `‖φψ‖² ≤ 4 q` — the estimates `hder`, `hnrm`, `hxc` of `imB_le`.
* **CHECK 7** the arithmetic–geometric‑mean identity behind `double_sum_amgm`
  (`(1/2)(w f² + w g²) − w f g = (1/2) w (f−g)²`), the step that bounds the weight sums by
  `(1/2)K` (A) and `(9/4)K` (B).
* **CHECK 8–9** the constant arithmetic: `imA_le` gives `½K`, `imB_le` gives `9/4K`; the
  commutator form doubles the sum, `2(½K + 9/4K) = 11/2 K`, and `11/2 K ≤ 6K`
  (`6K − 11/2 K = ½K ≥ 0`), which is `secHam_commForm_le` with `c = 6·K_Q`.
* **CHECK 10a–10d — the core (obligation (i) below).**  The wall `V` is *not* quadratic, so
  there is no closed ladder for `N_a`; what makes `N_a` ESA on the core is its structure as an
  **oscillator plus a non‑negative multiplication**: `N_a = (a†a + 1/2) + (V + σ_a)`, with
  `V + σ_a ≥ 0` (`V ≥ 0` by `starobinskyV_nonneg`, `σ_a ≥ 1` by `QgModeData.one_le_sig`).  The
  checks verify the factorization, the CCR, the split, and the shift arithmetic.  This is
  precisely the hypothesis of `oscillatorPlus_esa` (an operator `−d²/dx² + x²/4 + W` with `W`
  bounded below and **no** relative‑bound hypothesis on `W`), i.e. `WallPot.ham_esa` /
  `starobinskyWall_esa`: the wall sits inside `N` and needs no smallness.
* **CHECK 11a–11b — the lift.**  The outer comparison is the mode‑lift
  `secN = dsComparison (fibCompar W Q) = ⊕_a Friedrichs(N_{σ_a})` on `Sec ι = ℓ²(ι; L²(ℝ_φ))`
  (and `qgOuterComparison = dΓ(N₁)` in the 84‑coordinate `dΓ` spelling), *not* `h_s`.  The
  checks verify the same one‑particle‑wise action `[dΓ(N), a†_k] = Σ_i N_ik a†_i`,
  `[dΓ(N), a_k] = −Σ_i N_ki a_i`, which is what makes the lifted core the finite‑particle / mode
  core and what a core‑transfer proof of obligation (ii) consumes.

## The core, and the lift: why `N` itself must be ESA on the chosen core (Part B)

Faris–Lavine as the tree formalizes it (`ChapterFarisLavineCore`, `structure CoreData`) is run on a
**core** `C₀`: besides positivity and `N + 1` onto, the comparison `N` must be symmetric on `C₀`
and `C₀` must be a core for `N` (`IsGraphCore`: every domain vector approximated by a core vector
*together with its `N`‑image*).  Both modules now certify the symbolic content of that
requirement, and of the lift:

| Hamiltonian | `N` | core `C₀` | symbolic certificate | Lean target |
| :-- | :-- | :-- | :-- | :-- |
| NS one‑body / parcels (`H_sp = ½Σπ² + ½Σform²`, `spHam` / `nsSectorHam`) | `−Δ + ¼‖x‖²` | Hermite / Gauss‑polynomial `polyGaussCore D` | NS CHECK 5a–5d (factorization, CCR, ladder) | `harmonicOsc_essentiallySelfAdjoint`, `oscillator_essentiallySelfAdjoint_on_hermiteCore`, `polyGaussCore_dense` |
| NS mainstream `H_NS = ½Σ(πF + Fπ)` (`kvnPoly`) | Leray energy `N_E = 1 + ‖u‖²` (multiplication; self‑adjoint, `N_E + 1` onto) | Gauss‑polynomial (`polyGaussCore d`) | CHECK 4–6 of `ns_kvn_equation.cdb` | `nsKoopman_esa_of_energy_comparison` |
| QG fibre `h_s = −∂²_φ + φ²/4 + V + σ_a` | `N_a` = itself (Friedrichs) | compactly supported smooth `ccDomain ℝ` (Hermite core dense) | QG CHECK 10a–10d (oscillator plus non‑negative multiplication) | `oscillatorPlus_esa`, `WallPot.ham_esa`, `starobinskyWall_esa` |
| QG outer `secHam` on `Sec ι` | `secN = dsComparison (fibCompar W Q)` | `secCore` (finite‑particle / mode core) | QG CHECK 11a–11b (lift acts one‑particle‑wise) | `secHam_essentiallySelfAdjointOn` (domain), `qgFull_esa_core_fl` (core) |
| any outer Fock lift `dΓ(H₁)` | `dΓ(N₁)` | finite‑particle tensor core `⨁ₙ Γⁿ(C₀)` | NS CHECK 6a–6b / QG CHECK 11a–11b | `dGamma_essentiallySelfAdjointOn_fockCore`, `dGamma_essentiallySelfAdjointOn_of_esa`, `outerHam_esa_fl`, `dsOp_essentiallySelfAdjointOn` |

**The two obligations are independent.**  (i) `N` is ESA on `C₀`; (ii) the *lifted* Hamiltonian
is ESA on the *lifted* core.  Neither implies the other: the lifted operator is `dΓ(H₁)` (or the
parcel/mode sum), not `H₁`, and the lifted core is the finite‑particle tensor core, not `C₀`.  The
symbolic algebra certifying the lift (the `[dΓ(N), a†] = Σ N a†` identity) is what fixes the lifted
core and what a core‑transfer proof must use.  Obligation (ii) is discharged in general by the
core‑transfer / `dΓ`‑ESA wave (`ChapterGraphCoreTransfer`, `ChapterTensorGraphCore`,
`ChapterSecondQuantizationCoreEsa`); obligation (i) — and the symmetrization step — remain.

## Correction (2026‑09‑19): which operator is *the* Navier–Stokes Hamiltonian

A first pass of this note (and the earlier `.md` summaries) described the positive Weyl‑ordered
sum‑of‑squares operator `H_sos = ½Σ_m π_m² + ½Σ_r (mulOp Φ_r)²` as "the Navier–Stokes one‑body
Hamiltonian".  That is **not** the Navier–Stokes Hamiltonian.  Verbatim book.tex (the Navier–Stokes
section, eq. 4186),

```
H = ∫ d³x … a†(x,…) H(x,…) a(x,…)
H(x,…) = π^i ( u_j u_{i,j} + q_i − ν u_{i,jj} ) + (h.c.)        π^i = −i ∂/∂u_i ,
```

The Navier–Stokes equation itself is

```
∂_t u_i + u_j u_{i,j} = − ∂_i p + ν u_{i,jj} + f_i ,        ∂_j u_j = 0 ,
```

so the drift is the **full residual including the pressure gradient `q_i := ∂_i p`** (and the
external force `f_i`), exactly as the tree writes it in `BookProof.NsFullEuler.nsResPoly`
(`R_i = Σ_j u_j u_{i,j} + q_i − ν w_i`, pressure‑gradient coordinate `qIdx`) together with the
incompressibility `BookProof.NsFullEuler.divPoly` (`Σ_j u_{j,j} = 0`).  The pressure is not an
independent dynamical field — it is the Lagrange multiplier of the constraint: it does no work on
the velocity, `Σ_k u_k q_k = 0` on the divergence‑free sector (Leray's identity, CHECK 8b), and in
the mainstream *Leray–Galerkin* form it is eliminated by the Leray projection, leaving
`F_i = −ν λ_i u_i + B_i(u,u)`.  The Hermitized *momentum × drift* Koopman–von Neumann (Liouville)
generator — in `../timepiece` exactly `BookProof.NsKoopman.kvnPoly` / `nsKoopmanOp`
`= ½Σ_m(π_m F_m + F_m π_m)` with the pressure‑free drift `F_i = −ν λ_i u_i + B_i(u,u)`
(the module's `Fk_core` is exactly the tree's gauge‑fixed symbol `nsSymbolPoint`) — is the
operator that generates the flow.  `H_sos` is the
*auxiliary positive operator* — it plays the role book.tex describes as "`H²(x)` as a positive
auxiliary operator in Corollary 1.1" — and it is a legitimate source of the Faris–Lavine
comparison `N` **for the auxiliary operator itself**, but it is not the Hamiltonian, and its
commutator with the velocity is a *different* equation.  (Its `N` is the harmonic oscillator, not
`H²`; and the *literal* `H²` — the square of the generator — is **not** a valid comparison `N` at
all, for the two reasons in the caveat below.)

The defining property that pins the Hamiltonian down is the commutator with the velocity density.
On the outer (nested) Fock space the velocity density of the `k`‑th component is

```
U_k = ∫ du  a†(φ,u) u_k a(φ,u)        ("u_k between a creation operator on the left and
                                       an annihilation on the right, integrated over u_k"),
```

and the Heisenberg equation `i ∂_t U_k = [U_k, H]` must reproduce the Navier–Stokes equations.
`docs/ns_kvn_equation.cdb` verifies exactly this, at both levels.

## `docs/ns_kvn_equation.cdb` — checks and output

Run: `cadabra2-cli -q -n docs/ns_kvn_equation.cdb` (3 space dimensions; `ν` is the symbol `nu`).

```
CHECK 1a NS one-body [H,u1] + 2 i F1 phi         (0 = NS drift) : 0
CHECK 1b NS one-body [H,u2] + 2 i F2 phi         (0 = NS drift) : 0
CHECK 1c NS one-body [H,u3] + 2 i F3 phi         (0 = NS drift) : 0
CHECK 1d NS auxiliary [(1/2)sum pi^2, u1] = -d1 phi (NOT the drift) : 0
CHECK 2  NS outer Fock  [H,U] = sum [h,p] a^dag a      (0 = ok) : 0
CHECK 3d NS F1 - (nu|k|^2 u1 + i (k.u) u1)            (0 = ok) : 0
CHECK 4  NS  [H,N_E] + 4 i (sum_k u_k F_k) phi      (0 = ok) : 0
CHECK 5  NS H - flow = ordering/div term -i (div F) phi   (0 = ok) : 0
CHECK 6a NS energy flux = visc + Leray sum_i u_i B_i   (0 = ok) : 0
CHECK 6b NS Leray gives sum_i u_i F_i = nu |k|^2 |u|^2  (0 = ok) : 0
CHECK 7  NS advection is degree 2 (convolution), viscosity degree 1 (diagonal) : 0
CHECK 8a NS incompressibility sum_j u_{j,j} -> i (k.u)    (0 = ok) : 0
CHECK 8b NS pressure work sum_k u_k q_k -> i (k.u) p   (0 = ok) : 0
CHECK 8c NS div-free sector k.u = 0 kills 8a and 8b      (0 = ok) : 0
```

* **CHECK 1a–1c (the one‑body statement).**  With `H(x) = π^i F_i + F_i π^i`,
  `F_k = u_j u_{k,j} + q_k − ν u_{k,jj} + f_k`, and `π^i = −i ∂_{u_i}`, the module computes
  `H(u_k φ) − u_k H(φ)` by the Leibniz rule and checks `[H, u_k] + 2 i F_k φ = 0`, i.e.
  ```
  [H, u_k] = −2 i F_k    ⇔    i[H, u_k] = 2 F_k = 2 ( u_j u_{k,j} + q_k − ν u_{k,jj} + f_k ) ,
  ```
  the Navier–Stokes drift (advection + pressure gradient + viscous + forcing).  (Normalization: book.tex writes
  `H` without an explicit `½`, while timepiece's `kvnPoly` carries the Weyl `½`; `H_here = 2·kvnPoly`,
  so the equivalent timepiece statement is `i[kvnPoly, u_k] = F_k`.)  The Heisenberg equation for the
  velocity density is therefore the Navier–Stokes equation.  The drift comes out of the commutator
  with the velocity itself: the only input is the canonical relation `[π^i, u_k] = −i δ^i_k` and the
  fact that the drift `F_i` is a multiplication operator.
* **CHECK 1d (the contrast).**  The same computation with the *kinetic square* of the auxiliary
  operator, `½Σ_m π_m²`, gives `[½Σ_m π_m², u_1] = −∂_1 = −i π_1`, i.e. the **momentum density**,
  not the drift.  So the auxiliary sum‑of‑squares operator does not satisfy the defining commutator
  property: it is not the Navier–Stokes Hamiltonian.
* **CHECK 2 (the outer/nested Fock level).**  For the second quantization
  `H = Σ_{a,b} h_{ab} a†_a a_b` and the density `U = Σ_{a,b} p_{ab} a†_a a_b` (two modes, with the
  CCR `[a_b, a†_c] = δ_{bc}`), the module normal‑orders `HU − UH` and checks
  `[H, U] = Σ_{a,e} [h,p]_{ae} a†_a a_e`.  Hence `[H, U_k]` is the *second quantization* of the
  one‑body commutator: CHECK 1 plus CHECK 2 give, on the nested Fock space,
  `i ∂_t U_k = ∫ du a† (2 F_k) a` — the Navier–Stokes equation as a field equation.
* **CHECK 3 (momentum space / the elimination).**  Substituting the derivative by its momentum
  symbol (`u_{k,j} → i k_j u_k`, `u_{k,jj} → −|k|² u_k`, book.tex item 2) turns the drift into
  ```
  F_k → ν |k|² u_k + i (k·u) u_k ,
  ```
  and CHECK 3d verifies the split exactly: the **diagonal viscous part** `ν|k|² u_k` and the
  **advective part** `i (k·u) u_k`.  The advection is the momentum‑transferring term: in the field
  picture `u_j ∂_j u_k` becomes the convolution
  `ℱ[u_j ∂_j u_k](Q) = (i/(2π)^{d/2}) ∫ dq q_j Û_j(Q−q) Û_k(q)` (book.tex item 3), whose
  transform‑level statement is proved in Lean by
  `BookProof.NsAdvectionConvolution.fourier_advection_convolution` (`fourier_advection_sum`);
  Cadabra certifies the symbol‑level weight `i k_j` and the diagonal/convolutions split, the
  convolution itself being a Fourier‑transform theorem.  The **same** elimination→convolution
  structure is what the QG leg uses for the derivatives in space of the vielbein/torsion fields
  (the shared `docs/ns_qg_fourier_elimination.cdb` and the Lean `BookProof.ChapterQgFourierElimination`);
  only the NS leg has a velocity‑density commutator test
  of the form checked here.
* **CHECK 4 (the Faris–Lavine commutation condition).**  With the comparison `N_E = 1 + |u|²`
  (the Leray energy) the module checks `[H, N_E] + 4 i (Σ_k u_k F_k) φ = 0`, i.e.
  `[H, N_E] = −4 i (Σ_k u_k F_k) φ` — the first order, multiplication‑by‑the‑energy‑flux
  commutator (`BookProof.NsKoopman.kvn_comm_energy`).
* **CHECK 5 (H is complete).**  Since `π^i F_i + F_i π^i = −i∂_i(F_i ·) − i F_i ∂_i`, expanding the
  Hermitized `H` gives `H φ = −2 i Σ_i F_i ∂_iφ − i (Σ_i ∂_i F_i) φ`: the first group is the
  convection/flow term, the second the Weyl‑ordering / `+ h.c.` term (the part an operator written
  as `π^i F_i` alone would omit).  CHECK 5 verifies the module's `H` carries **both**, and that the
  ordering term is exactly `−i (div F) φ` with `div F = Σ_i ∂_i F_i`.  The pressure‑gradient
  coordinate contributes nothing to `div F = Σ_i ∂_{u_i}F_i` (it is independent of `u`), so the
  ordering term is the same `−i (div u) φ` as for the pressure‑free residual.  So the Hamiltonian
  is book.tex eq. 4186 to the letter for the full drift, with no term dropped.
* **CHECK 6 (the *valid* comparison, and why `H²` is not one).**  By Leray's identity
  `Σ_i u_i B_i = 0` (`B_i = u_j u_{i,j}`; the advection does no work), the energy flux of CHECK 4
  collapses to the **sign‑definite** viscous dissipation:  CHECK 6a checks
  `Σ_i u_i F_i = (Σ_i u_i B_i) − ν Σ_i u_i u_{i,jj}` (the advective part *is* the Leray sum), and
  CHECK 6b checks that in momentum space this is `ν |k|² |u|² ≥ 0`.  Hence the commutator form is
  dominated by `N_E` and the Faris–Lavine criterion applies on the mainstream leg
  (`commForm_kvn_energy_bound`, `nsKoopman_esa_of_energy_comparison`).  By contrast the **literal**
  `H²` is not a valid `N`: the criterion needs `N + 1` onto, and
  `range(H² + 1) ⊆ range(H − i)` (`H² + 1 = (H − i)(H + i)` on the domain), so `H² + 1` is not onto
  whenever `H` has a non‑trivial deficiency — the formalized general refutation of relying on the
  inequalities alone is `BookProof.FarisLavine.not_farisLavine_criterion_of_relative_bound` — and
  `D(H²) ⊊ D(H)` rules it out on the common Faris–Lavine domain independently.
* **CHECK 7 (convolution vs diagonal).**  With a formal scaling variable the module verifies the
  advective term `u_j u_{i,j}` is **degree 2** in the one‑particle coordinates (bilinear in the
  field and its independent derivative variable, hence momentum‑transferring — a convolution),
  while the viscous `−ν u_{i,jj}` is **degree 1** (diagonal in momentum); the transform‑level
  statement is again the Lean theorem `fourier_advection_convolution`.
* **CHECK 8 (the pressure gradient and the constraint).**  The drift of CHECK 1 carries
  `q_k = ∂_k p` and `f_k`, yet the constraint structure is intact.  **8a:** the incompressibility
  symbol is `Σ_j u_{j,j} → i (k·u)` (the tree's `divPoly`, `nsElimSubst_divPoly`).  **8b:** the
  pressure does no work on the velocity, `Σ_k u_k q_k → i (k·u) p` (with `q_k → i k_k p`), which
  is exactly the pressure half of Leray's identity — on the divergence‑free sector `k·u = 0` it
  vanishes, the same collapse the reduced model records in its `q_i + ν|k|²u_i` form.  **8c:** the
  substitution `k·u = 0` kills both 8a and 8b.  **8d:** dropping `q_k` (and `f_k`) from the full
  residual recovers the tree's gauge‑fixed symbol
  `nsSymbolPoint = u_j u_{i,j} − ν u_{i,jj}` — i.e. the pressure is the projected‑out/eliminated
  part of the reduced Hamiltonian, not a dynamical field of it.  (Physical meaning: the pressure
  gradient is determined by the constraint through the pressure‑Poisson equation; it is a Lagrange
  multiplier, so it may be kept (full free‑field form, tree `nsResPoly`) or projected away (Leray),
  and the equation is the same either way.)

## `docs/ns_pressure_poisson.cdb` — the pressure Poisson equation

Run: `cadabra2-cli -q -n docs/ns_pressure_poisson.cdb`; every check reduces to `0`.

The pressure is not a dynamical field: applying the divergence to the momentum equation (with
`∂_t u_i` written as the field `dtu_i`, since `∂_i ∂_t u_i = ∂_t(div u)`) and using
incompressibility gives the classical

```
Lap p = − div[(u·∇)u] + div f .
```

The module verifies the four ingredients and then assembles them:

* **CHECK 1** — the time-derivative drops: `∂_i∂_t u_i = ∂_t(∂_i u_i) = ∂_t(div u) = 0` (the
  constraint is time-independent).
* **CHECK 2 / 2b** — the **viscous divergence is the Laplacian of the divergence**, by Clairaut
  (equality of mixed partials): `∂_i(∂_j∂_j u_i) = ∂_j∂_j(∂_i u_i) = Δ(div u) = 0`.
* **CHECK 3 / 3b** — the **advection divergence** splits by the Leibniz rule:
  `∂_i[(u_j∂_j)u_i] = (∂_i u_j)(∂_j u_i) + u_j ∂_j(∂_i u_i) = tr((∇u)²) + u·∇(div u)`, and the
  second piece vanishes on `div u = 0`; so `div[(u·∇)u] = tr((∇u)²)`.
* **CHECK 4** — the pressure term is the Laplacian: `∂_i∂_i p = Δp`.
* **CHECK 5** — **assembly**: `div M = Δp + tr((∇u)²) − div f + C` with
  `C = ∂_t(div u) + u·∇(div u) − νΔ(div u)` the constraint terms.
* **CHECK 6** (with 2b and 3b) — the constraint terms vanish, leaving the Poisson equation.

```
CHECK 1  NS  d_i d_t u_i = d_t(div u) = 0  on  div u = 0   : d_i(d_t u_i) = d_t(div u)
CHECK 2  NS  d_i(d_j d_j u_i) = Lap(div u)  (Clairaut)        (0 = ok) : 0
CHECK 2b NS  d_i(d_j d_j u_i) = sum_{i,j} d_{jji} u_i        (0 = ok) : 0
CHECK 3  NS  div[(u.grad)u] = tr((grad u)^2) + u.grad(div u)  (0 = ok) : 0
CHECK 3b NS  u.grad(div u) = 0  on  div u = 0                 (0 = ok) : 0
CHECK 4  NS  d_i d_i p = Lap p                                : \partial_{1 1}(p) + \partial_{2 2}(p) + \partial_{3 3}(p)
CHECK 5  NS  div M = Lap p + tr((grad u)^2) - div f + (cons.)  (0 = ok) : 0
CHECK 6  NS  d_t(div u) = 0                                   (0 = ok) : 0
```

Caveats worth stating. Cadabra normalises nested `\partial` nodes into a single multi-index and
does not canonicalise the index order by itself, so the module (i) imposes Clairaut by an explicit
rule set that sorts every derivative index list, and (ii) keeps the constraint groups `∂_j(div u)`
as explicit sub-nodes (Cadabra's sums are flat, so a sub-sum of a flattened sum cannot be matched
by `substitute`). The nonlinear advection divergence is handled by the product rule rather than at
the level of the symbol, which is why this module complements `ns_kvn_equation.cdb` CHECK 3 (the
momentum-space convolution).

## `docs/ns_pressure_constraint.cdb` — the pressure equation is a *second-class* constraint

Run: `cadabra2-cli -q -n docs/ns_pressure_constraint.cdb`; every check reduces to `0`.

The pressure enters the residual only through `q_i = ∂_i p`, with no `∂_t p` and no conjugate
momentum: it is the **Lagrange multiplier** of incompressibility, and the Poisson equation is the
**secondary constraint** that determines it. The question a Hamiltonian/quantum treatment must
answer is whether it needs a BRST gauge symmetry. It does not:

* **CHECK 1** — the equation `−|k|²p = S` (`S = −widehat(div[(u·∇)u]) + i k·f̂`) contains no
  `∂_t p`, so it is a constraint, not an evolution equation.
* **CHECK 2** — it **determines** `p` on every non-zero mode by inverting `Δ`:
  `p = −S/|k|²` satisfies it identically.
* **CHECK 3** — **second-class certificate**: the persistence condition determines `p` through the
  Laplacian (symbol `|k|²`), invertible for `k ≠ 0`; the only free mode is the
  additive constant `p(0)`, and `∂_i(p + c) = ∂_i p`, so the freedom is invisible to `q_i` and to
  the dynamics (book.tex fixes it by requiring the constraints be conserved).
* **CHECK 4** — book.tex's BRST packaging `Ω = (∂_j u_j)ψ†` is nilpotent, `Ω² = 0`, from the
  fermionic `(ψ†)² = 0` — but it gauges nothing new for a solvable constraint; it is *"only the
  elegant packaging"*.
* **CHECK 5** — the **elimination**: `q_i = ∂_i p = −i k_i S/|k|²` is a function of `(u, f)` alone,
  and substituting it makes the constraint vanish identically.

```
CHECK 1  NS  the equation  -|k|^2 p = S  contains no d_t p     : constraint
CHECK 2  NS  p = -S/|k|^2 solves  -|k|^2 p = S          (0 = ok) : 0
CHECK 3a NS  determining operator symbol = |k|^2 (invertible for k!=0) : Kap2
CHECK 3b NS  d_i(p + c) - d_i p - d_i c = 0  (constant decouples) (0 = ok) : 0
CHECK 4  NS  Omega = (d_j u_j) psi^dag is nilpotent, Omega^2 = 0  (0 = ok) : 0
CHECK 5  NS  q_1 = d_1 p = -i k1 S/|k|^2  (no p left)         : -I k1 Ssym (Kap2)**(-1)
CHECK 5b NS  the constraint vanishes identically after elimination (0 = ok) : 0
```

**Conclusion.** The pressure-Poisson equation is a **second-class** (multiplier-determining)
constraint, so BRST — the tool for *first-class* gauge constraints — is not the natural treatment;
the consistent choice is the plan's **elimination** strategy (solve `div u = 0` by the book's
substitution and the Poisson equation for `p`, removing `q_i` as an independent coordinate), exactly
as book.tex says ("the divergence constraint can be easily solved … the BRST formalism is used only
to allow a more elegant formulation … and not to define the theory itself") and as the tree's
`ChapterNavierStokesEulerian` classifies it (explicit-solution constraint).  The genuinely
gauge-generator constraints are the derivative relations `u_{i,j} = ∂_j u_i`, whose charge
`Ω = Σ_j G_jχ_j` is nilpotent with `[Ω, H] = 0` (`BookProof.NsBrstDerivativeGauge`).  If a uniform
BRST packaging is nevertheless wanted, keep `q_i` and impose the pair (integrability
`∂_i q_j = ∂_j q_i`, Poisson `∂_i q_i = Δp`), both solvable by the potential `p`.

## Why the lifted comparison `N` is likely ESA, and likely easy to prove

The Faris–Lavine criterion is used on the outer Fock space in the lifted (second‑quantized) form
`dΓ(N₁)` — in the tree this is `dsOp` / `dsComparison` (`BookProof.ChapterDirectSumEsa`,
`BookProof.ChapterQgOuterFockFarisLavine/Part1`) with the `Comparison.esa_self` instance, and the
NS instance is `BookProof.NsOneBody.nsSpDGamma_esa_farisLavine`.  Essential self‑adjointness of
the lift on the *lifted* core (the fourth obligation, plan §D6b) is easy for a structural reason:
**the lift has a sector‑uniform spectral gap.**

1. `dΓ(N₁)` preserves particle number.  On the `n`‑particle sector it is the `n`‑fold sum
   `N₁^{(n)} = Σ_{p=1}^n 1⊗…⊗N₁⊗…⊗1`, so positivity is inherited sector by sector and
   `dΓ(N₁ + 1) = dΓ(N₁) + 𝒩` (`𝒩` = number operator).
2. After the harmless shift `N₁ ↦ N₁ + 1`, the `n`‑particle sector value is `≥ n`: the operator
   exceeds the particle number, and the gap *grows with the particle number*.
3. Consequently, for `ψ` in the finite‑particle algebraic core `⨁_n Γ^n_sym(C₀)`, the tail
   `T_Kψ = Σ_{n>K} ψ_n` satisfies `‖T_Kψ‖ ≤ ‖(dΓ(N₁)+1)ψ‖/(K+1) → 0`.  The truncations converge in
   the **graph norm**, so the algebraic core is a graph core; and on each fixed sector the tensor
   power `C₀^{⊗n}` is dense in the graph norm of `N₁^{(n)}` because `N₁` is ESA on `C₀`.  Hence
   `dΓ(N₁)` is ESA on the lifted core.

The reason this is *likely easy* rather than analogous to the one‑particle obligation is that the
step eliminated by the lift is the only hard part: the one‑particle ESA of `N₁` on `C₀` (for the
oscillator comparisons, the classical Gauss–Hermite/Sturm–Liouville statement; for the QG wall, the
oscillator‑with‑wall statement) is where the analysis lives, and it is **already proved in the
tree** for exactly these comparison operators (`polyGaussCore_dense`, the QG `secN` core, …).  Once
that is in hand, the lift adds only the elementary number‑operator bookkeeping above — no new
estimate, no uniformity in `n`, and no input from the interaction.  This is the same mechanism the
review `REVIEW_FARIS_LAVINE_20260911.md` §0 records as the reason the criterion "lifts with its
comparison operator", and it is why every `c = 0` self‑comparison row of the plan is a lift rather
than a new proof.

Caveats (why "likely", not "automatic"): (i) the one‑particle operator must be defined and
symmetric on `C₀` (the plan's §D6b fourth obligation); (ii) the comparison must be non‑negative
and `N₁ + 1` onto (automatic for a Friedrichs realization of a bounded‑below symmetric operator);
(iii) **the comparison must not be built out of `H` itself.**  `N = H` and `N = H²` are not
admissible in general, and this is what the sector‑gap argument must *not* be applied to:

* `N = H` for the NS **Koopman** Hamiltonian is not bounded below at all
  (`nsKoopmanOp_not_bounded_below`, `quadP_starP`: the numerical range is symmetric about `0`),
  so no shift of `H` is positive;
* `N = H² + 1` is positive, but the criterion
  (`essentiallySelfAdjointOn_of_farisLavine`) needs `N + 1` **onto**, and
  `range(H² + 1) ⊆ range(H − i)` because `H² + 1 = (H − i)(H + i)` on the domain: so `H² + 1`
  fails to be onto whenever `H` has a non‑trivial deficiency, i.e. exactly in the case the
  criterion is meant to settle (circular).  The formalized general refutation of relying on the
  inequalities alone is `BookProof.FarisLavine.not_farisLavine_criterion_of_relative_bound`
  (the limit‑circle Jacobi operator with `N = H`: both inequalities hold with `a = 1`, `b = 0`,
  while essential self‑adjointness fails); and more basically `D(H²) ⊊ D(H)`, so `H²` cannot even
  be a comparison on the common Faris–Lavine domain;

The **valid** comparison for the NS Hamiltonian leg is the **Leray energy** `N_E = 1 + ‖u‖²`
(`BookProof.NsKoopman.nsEnergyOp`): a multiplication operator, hence self‑adjoint, positive and
`N_E + 1` onto, with the commutator collapsing — by Leray's identity `ΣᵢuᵢBᵢ = 0` — to the
sign‑definite viscous dissipation `i[H_NS,N_E] = F·∇E = −2ν Σᵢλᵢuᵢ²`, so that
`|⟪x,i[H_NS,N_E]x⟫| ≤ 2νΛ ⟪x,N_E x⟫` unconditionally (`commForm_kvn_energy_bound`,
`nsKoopman_esa_of_energy_comparison`; `CHECK 4–6` of `docs/ns_kvn_equation.cdb`).  For every other
Faris–Lavine leg in the tree the comparison is a **Friedrichs realization of a positive
operator** — the harmonic oscillator `−Δ + ‖x‖²/4` (NS/YM parcels, QG `dΓ` route) or the
scalaron‑wall operator `−∂²_φ + φ²/4 + V(φ) + σ_a` with the wall inside (QG) — never a function
of `H` itself.  With such a comparison the lift statement above applies as written.

## Audit of the NS and QYM inner one‑particle operators (2026‑09‑21)

Same exercise as the QG audit in `VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md`: compare the
one‑particle operator **as the plan states it** with the one **as the proofs define it**, read off
the definitions rather than the doc‑comments.

### NS

| plan statement | Lean declaration | file:line | matches |
| :-- | :-- | :-- | :-- |
| reduced one‑body `H_sp = ½Σ_{m<6}π_m² + ½Σ_{r<7}(mulOp Φ_r)²` on `L²(ℝ⁶)` | `spHam Φ ν k = weylOp (spPi Φ) (spField Φ ν k)` (6 momenta, 7 forms) | `ChapterNsOneBodyDGamma.lean:194` | ✓ |
| full one‑parcel `H₁ = nsSectorHam … 1 = ½Σπ² + ½Σ(mulOp Φ_r)²` on `L²(ℝ²¹)` | `nsSectorHam = weylOp (nsPiN n) (nsFieldN … n)` (12 momenta, 19 forms, `polyGaussCore (n*21)`) | `ChapterNavierStokesFullEulerianFock.lean:248` | ✓ |
| outer `nsFullFockHam = dsOp (fun n => nsSectorHam … n) = dΓ(H₁)` on `⊕ₙ L²(ℝ^{21n})` | `nsFullFockHam = dsOp (…)`; `nsFockSpace = lp (fun n => L2d (n*21)) 2` | `:287`, `:277` | ✓ |
| mainstream `H_NS = ½Σ_m(π_m F_m + F_m π_m)` with `F_i = −νλ_iu_i + B_i(u,u)` | `kvnPoly = Σ_i weylProd (momOp i) (mulOp (drift S i))`, `drift i = −νλ_i X_i + advOf bcoef i` | `ChapterNsKoopman/Part1.lean:185,179` | ✓ |
| full residual `π^i(u_j u_{i,j} + q_i − ν u_{i,jj}) + h.c.`, `divPoly` | `nsResPoly = u_j u_{i,j} + q_i − ν w_i`, `divPoly = Σ_j u_{j,j}` | `ChapterNavierStokesFullEulerianFock.lean:143,149` | ✓ |
| reduced family `redHam` (Fourier‑eliminated) | `redHam ν k n = weylOp (redPiN n) (redFieldN ν k n)` on `L²(ℝ^{6n})` | `ChapterNsFourierElimination.lean:512` | ✓ |

**NS matches, with one disambiguation to keep.**  There are **two** one‑particle NS models in the
tree, and the plan names both in different places:

* the **reduced** (Fourier‑eliminated) one‑body operator `spHam` = `redHam ν k 1` on `L²(ℝ⁶)`
  (6 momenta, 7 forms) — this is the leg of the Faris–Lavine proof (`spHam_esa_farisLavine`);
* the **full** Eulerian one‑parcel `nsSectorHam … 1` on `L²(ℝ²¹)` (12 momenta, 19 forms), whose
  `dΓ` is `nsFullFockHam`; and the full Lagrangian `lagSectorHam` on `L²(ℝ^{36n})`.

The plan should say *which* one it means at each occurrence (`H_sp` = reduced vs `H₁` fixed by
`nsSectorHam`).  No contradiction in the proofs.

**One stale dimension, found and fixed.**  `../timepiece/HAMILTONIAN_AUDIT_20260918.md` §9 gave
the nested space of the *full* Eulerian model as `L²(ℝ^{18n})`; the proof's `nsFockSpace` is
`L²(ℝ^{21n})`.  The `18n` belongs to the **reduced / gauge‑fixed** family (`nsFamily`, `dim := n*18`,
`ChapterNsOuterFockFarisLavine/Part2.lean:292`), not to `nsSectorHam`.  Fixed in the audit; the
full Lagrangian `36n` was correct.

### QYM

| plan statement | Lean declaration | file:line | matches |
| :-- | :-- | :-- | :-- |
| one body `H₁ = ½Σπ² + ½ΣB²` on the Gauss–polynomial core of `L²(ℝ⁹⁹)` (24 momenta, 24 magnetic forms) | `ymHamiltonian Φ fabc = weylOp (piOps Φ) (magOps Φ fabc)`; `ymHamiltonian_quadForm` | `ChapterYangMillsHermite/Part2.lean:201,218` | ✓ |
| abelian `ymHamiltonian = sqSumOp ymKap ymMagVec = ½Σ_j κ_jπ_j² + ½Σ_m B_m²` | `ymAbelian_eq_sqSumOp`; `sqSumOp` | `ChapterYangMillsOuterFockFL/Part1.lean:134`; `ChapterQgOuterFockEsa.lean:225` | ✓ |
| parcels: 99 coordinates per parcel, 24 forms, `99n` total | `ymFamily` (`dim n = n*99`, `R = Fin n × YmForm`), `ymFamily_vv` | `ChapterYangMillsOuterFockFL/Part2.lean:322,342` | ✓ |
| outer `outerHam` on `⊕ₙ L²(ℝ^{99n})`, ESA by Faris–Lavine | `outerHam = dsOp (fun n => F.secHam n)`; `ymOuterHam_esa_fl` | `ChapterSqSumOuterFamily.lean:224`; `ChapterYangMillsOuterFockFL/Part2.lean:356` | ✓ |

**QYM matches term for term**, including the sign reconciliation (`book.tex:7077` writes
`H = −½ππ − ½BB`; the proofs use the bounded‑below `+½Σπ² + ½ΣB²`, as both the plan and the
`ymHamiltonian` docstring record).  No repair needed.

## What these modules do **not** certify

Cadabra is a symbolic/algebraic engine on classical expressions.  It certifies the *one‑particle
symbol*: the shape of the comparison operator `N`, the commutator `[H,N]`, the oscillator
factorization/CCR and the lift identity `[dΓ(N), a†_k] = Σ_i N_ik a†_i` (Part B), and the algebraic
identities that turn the commutator into the bound with the stated constants.  It does **not**
certify the Hilbert‑space statements — symmetry on the core, positivity as a quadratic form,
surjectivity of `N + 1`, the graph‑core approximation itself (that the core is dense in the graph
norm of `N`), the existence of the (lifted) Friedrichs extension, and the lift of the *criterion*
to the outer Fock space.  Those are proved in Lean (`BookProof.ChapterFarisLavineCore`,
`BookProof.ChapterSqSumFarisLavine`, `BookProof.ChapterScalaronOuterFockFL`,
`BookProof.ChapterSecondQuantizationCoreEsa`).  This is the same division of labour the existing
Hamiltonian modules have (see `VERIFY_CDB_TRUNCATION_AUDIT.md` and
`../timepiece/HAMILTONIAN_AUDIT_20260918.md` §9).

## Reproduction

```
export LD_LIBRARY_PATH="$(ls -d /nix/store/*gcc-15*/lib | head -1):$LD_LIBRARY_PATH"
C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
$C2 -q -n docs/ns_kvn_equation.cdb
$C2 -q -n docs/faris_lavine_n_ns.cdb
$C2 -q -n docs/faris_lavine_n_qg.cdb
```

All three run clean (`exit 0`, no traceback); `ns_kvn_equation.cdb` prints `0` for CHECK 1a–1d/2/3d,
NS CHECK 1/2/4/5/6 and QG CHECK 1/2/3/7/8/9/10/11 print `0`.
