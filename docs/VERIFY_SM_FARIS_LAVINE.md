# Verification: the Standard Model Hamiltonian in nested Fock space and its Faris–Lavine `N`

Evidence base for `docs/faris_lavine_n_sm.cdb` — the Standard Model counterpart of
`docs/faris_lavine_n_ns.cdb` (Navier–Stokes) and `docs/faris_lavine_n_qg.cdb` (quantum gravity).
The module writes down the one-particle operator and the positive comparison operator of the SM
chapter and verifies the **algebraic content of the three Faris–Lavine hypotheses** in the
temporal (Weyl) gauge `A_0 = W_0 = B_0 = 0`.

## The objects

Single-particle space: the collective coordinates

    xi = ( x , {G^a_i, G^a_{i,j}}, {W^k_i, W^k_{i,j}}, {B_i, B_{i,j}},
             {phi_a, phi_{a,j}}, zeta_F )            (D_B = 163; zeta_F in Z_2^{D_F})

on `H = Gamma^s(L^2(R^{D_B} x Z_2^{D_F})) (x) Gamma^a(...)`, with the canonical pairs
`[G^a_i, pi^G_{b,j}] = i delta^a_b delta_ij`, `[phi_a, pi^phi_b] = i delta_{ab}`,
`{psi_A, psi^dag_B} = delta_{AB}`, and the spatial-derivative ideal
`D_j = d_{x_j} + sum_Phi Phi_{,j} d/dPhi`.

    h = h_Gauge + h_Higgs + h_Dirac + h_Yukawa

| sector | operator |
| :-- | :-- |
| Gauge | `½(pi^G pi^G + B^G B^G) + ½(pi^W pi^W + B^W B^W) + ½(pi^B pi^B + B^B B^B)`, with `B^G_{a,i} = ½ eps_ijk (G^a_{k,j} - G^a_{j,k} + g_s f^{abc} G^b_j G^c_k)` (analogously `B^W` with `g eps^{abc}`, `B^B = ½ eps_ijk (B_{k,j} - B_{j,k})`) |
| Higgs | `½ pi^phi_a pi^phi_a + ½ (D_i phi)_a (D_i phi)_a + V(phi)`, `V(phi) = -½ mu^2 |phi|^2 + ¼ lam |phi|^4`, `lam > 0` |
| Dirac | `sum_m Psi_m^dag (-i gamma^0 gamma . D) Psi_m`, `D = grad - i g_s T^a G^a - i g (tau/2).W - i g' Y B` |
| Yukawa | `Q_L^dag gamma^0 M_d phi d_R + Q_L^dag gamma^0 (i sigma_2) M_u phi u_R + L_L^dag gamma^0 M_e phi e_R + L_L^dag gamma^0 (i sigma_2) M_nu phi nu_R + h.c.` with biunitary decompositions `M_d = U_L diag(m_d,m_s,m_b) U_R^{d dag}`, `M_u = U_L V^dag diag(m_u,m_c,m_t) U_R^{u dag}` (CKM `V`), `M_e = U_L^e diag(m_e,m_mu,m_tau) U_R^{e dag}`, `M_nu = U_L^nu diag(m_1,m_2,m_3) U_R^{nu dag}`, and PMNS `U = U_L^{e dag} U_L^nu` entering the neutrino Yukawa in the charged-lepton mass basis |

The comparison operator (`N >= 1` after the shift `c_0 >= 1`):

    N = N_0 + c_0 I,
    N_0 = sum_{a,i} ( (pi^G_{a,i})^2 + (G^a_i)^4 + sum_j (G^a_{i,j})^2 )
        + sum_{k,i} ( (pi^W_{k,i})^2 + (W^k_i)^4 + sum_j (W^k_{i,j})^2 )
        + sum_i     ( (pi^B_i)^2   + (B_i)^2     + sum_j (B_{i,j})^2 )
        + sum_a     ( (pi^phi_a)^2 + (phi_a)^4   + sum_j (phi_{a,j})^2 )
        + sum_m     Psi_m^dag ( -Delta_x + |x|^2 + 1 ) Psi_m .

**The one structural fact.**  The confinement in `N` is **quartic** in the non-abelian fields and
the Higgs (`G^4, W^4, phi^4`) and **quadratic** in the derivatives and in the abelian
field/momentum pairs.  That mix is what makes the commutator ladder terminate at the right order:
one commutation with `N` removes one power of `(q,p)`, so `[h,N]` is first order and is dominated
by `N` (hypothesis (ii)); the double commutator `[N,[N,h]]` is quadratic in `p` and low degree in
`q`, dominated by `N^2` (hypothesis (iii)).  A purely quadratic confinement would fail the *upper*
domination of `h` (the cubic/quartic magnetic and Higgs terms); a purely quartic one would leave
the first-order piece unbounded.

## Checks and output

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    $C2 -q -n docs/faris_lavine_n_sm.cdb        # exit 0

| check | content | value |
| :-- | :-- | :-- |
| 1a | `[p^2, q^2] = -4 q d - 2` (the text's `[p^2,q^m]` formula at `m = 2`), by explicit differentiation | `0` |
| 1b | `[p^2, q^4] = -8 q^3 d - 12 q^2` (at `m = 4`) | `0` |
| 2 | **hypothesis (ii)**: `[h_1, N_1] = -4(1 - g_c^2) q^3 d - 6(1 - g_c^2) q^2` is **first order** (the `d^2` term cancels) | `0` |
| 2b | the same for a derivative coordinate: `[p_d^2, d^2] = -4 d d_d - 2` | `0` |
| 3 | **hypothesis (iii)**: `[p^2, [p^2, q^4]] = 48 q^2 d^2 + 96 q d + 24` — quadratic in `p`, quadratic in `q` (the highest object the quartic confinement produces) | `0` |
| 4a | domination of the first-order piece: `q^6 + p^2 - 2 q^3 p = (q^3 - p)^2 >= 0` | `0` |
| 4b | domination of the double-commutator piece: `p^4 + q^4 - 2 p^2 q^2 = (p^2 - q^2)^2 >= 0` | `0` |
| 5 | **hypothesis (i)**, magnetic cross term (Young): `½x^2 + ½G^4 - x G^2 = ½(x - G^2)^2 >= 0` — no relative bound needed | `0` |
| 6 | **hypothesis (i)**, quartic magnetic term (Schwarz): `2(G_1^4 + G_2^4) - (G_1^2 + G_2^2)^2 = (G_1^2 - G_2^2)^2 >= 0` | `0` |
| 7 | **hypothesis (i)**, Higgs potential: `V(phi) + mu^4/(4 lam) = (lam/4)(r_2 - mu^2/lam)^2 >= 0` in `r_2 = |phi|^2` | `0` |
| 8 | **hypothesis (i)**, Yukawa/fermion AM-GM: `½ phi^2 + ½ F^2 - phi F = ½(phi - F)^2 >= 0` (`F = Psi^dag Psi`) | `0` |
| 9a | **the core**: `p^2 + q^2 = a^dag a + 1` with `a = d + q` (the abelian/Higgs quadratic confinement factorises; arithmetic spectrum `2(n+1)`) | `0` |
| 9b | the quartic summand `p^2 + q^4` is a sum of squares (integrand `(dX)^2 + q^4 X^2 >= 0`) | `0` |
| 9c | the shift `c_0 >= 1` (informational: `c_0 - 1` is not a "zero" check, it records the `N >= 1` normalisation) | `c_0 - 1` |
| 10a | the lift `dΓ(N) = sum N_ij a^dag_i a_j`, `[dΓ(N), a^dag_k] = sum_i N_ik a^dag_i` | `0` |
| 10b | `[dΓ(N), a_k] = -sum_i N_ki a_i` | `0` |
| 11 | hypothesis (i) assembled: `c_1 = ½ + C_A + C_H + C_D + C_Y` is a finite sum of finite sector constants | `0` |
| 12 | **Higgs covariant kinetic**: `(W·τ)² = |W|²` (Pauli anticommutators `{τ_i,τ_j} = 2δ_ij`), so `|g W·τ/2 φ|² = (g²/4)|W|²|φ|²` | `0` |
| 13 | **hypothesis (i)**, Higgs kinetic Young: `3(a²+b²+c²) − (a+b+c)² = (a−b)²+(b−c)²+(c−a)² ≥ 0` splits `D_iφ = ∂_iφ + gW·τ/2 φ + g'σ₃/2 B_i φ` into derivative / weak / hypercharge pieces | `0` |
| 14 | **hypothesis (i)**, weak×Higgs cross-term: `W⁴+φ⁴−2W²φ² = (W²−φ²)² ≥ 0` puts `|W|²|φ|²` into the `W⁴, φ⁴` confinement of `N₀` | `0` |
| 15 | **hypothesis (i)**, hypercharge×Higgs cross-term: `B⁴+φ⁴−2B²φ² = (B²−φ²)² ≥ 0` | `0` |
| 16 | **hypothesis (ii), Higgs**: `[h_H, N_H] = ½[p²,q⁴] + [V,p²]` with the **full** `V = −(μ²/2)q² + (λ/4)q⁴` is **first order**: `2(λ−2)q³d − 2μ²q d + 3(λ−2)q² − μ²` (no `d²`) | `0` |
| 17a | **CKM**, Cabibbo `V = ((c,s),(−s,c))`: `VV^T` off-diagonal `cs − sc = 0` | `0` |
| 17b | **CKM**, Cabibbo: `VV^T` diagonal `c²+s² − 1 = 0` given `c²+s² = 1` (unitarity) | `0` |
| 18a | **biunitary masses**: `(VD)^T(VD)` off-diagonal vanishes (CKM rotation drops out of the mass matrix) | `0` |
| 18b | `(VD)^T(VD)_11 = m_1²` given `c²+s² = 1` — `M_u†M_u = U_R diag(m²) U_R†` for `M_u = U_L V† diag(m) U_R†` | `0` |
| 18c | `(VD)^T(VD)_22 = m_2²` given `c²+s² = 1` | `0` |
| 19a | **CKM row unitarity**: `V₁₁²+V₁₂²+V₁₃²` decomposes (the unit-norm statement) | `0` |
| 19b | `1 − V₁₁² = V₁₂²+V₁₃² ≥ 0` given row unitarity — **`|V_ij| ≤ 1`** for every CKM entry | `0` |
| 20 | **hypothesis (i)**, Yukawa with a CKM factor: `(vφ)² + F² − 2vφF = (vφ − F)² ≥ 0`, and `v² ≤ 1` by CHECK 19b so `|V_ij φ F|` is dominated by the same AM-GM as CHECK 8 | `0` |
| 21 | **hypothesis (i)**, quark covariant derivative (color): `G⁴+1−2G² = (G²−1)² ≥ 0` puts `G²|Ψ|²` into the `G⁴` confinement plus the fermionic oscillator `−Δ+|x|²+1 ≥ 1` | `0` |
| 22 | **hypothesis (iii), Higgs**: `[N,[N,h_H]]` contains no `∂^k X` for `k ≥ 3` — differential order ≤ 2 in `p` (the double commutator stays quadratic) | `0` |
| 23 | **hypothesis (i)**, Higgs assembly: `c₁^H = ½ + C_kin + C_V` is a finite sum of finite constants (CHECK 13–15 + CHECK 7) | `0` |
| 24 | **hypothesis (i)**, quark covariant derivative `D = ∇ + G + W + B`: 4-sum Young `4Σa_i² − (Σa_i)² = Σ_{i<j}(a_i−a_j)² ≥ 0` dominates all cross-terms of `Ψ†γ⁰γ·D Ψ` | `0` |
| 25a | **PMNS**, Pontecorvo 2×2 `U = ((c,s),(−s,c))`: `UU^T` off-diagonal `cs − sc = 0` (lepton analogue of CHECK 17a) | `0` |
| 25b | **PMNS**, Pontecorvo: `UU^T` diagonal `c²+s² − 1 = 0` given `c²+s² = 1` | `0` |
| 26a | **biunitary lepton masses**: `(UD)^T(UD)` off-diagonal vanishes (PMNS rotation drops out of `M_ν†M_ν` / `M_e†M_e`) | `0` |
| 26b | `(UD)^T(UD)_11 = m_1²` given `c²+s² = 1` — `M_ν†M_ν = U_R diag(m²) U_R†` for `M_ν = U_L^ν D U_R^{ν†}` (and identically `M_e`) | `0` |
| 26c | `(UD)^T(UD)_22 = m_2²` given `c²+s² = 1` | `0` |
| 27a | **PMNS row unitarity**: `U₁₁²+U₁₂²+U₁₃²` decomposes (the unit-norm statement) | `0` |
| 27b | `1 − U₁₁² = U₁₂²+U₁₃² ≥ 0` given row unitarity — **`\|U_ij\| ≤ 1`** for every PMNS entry | `0` |
| 28 | **hypothesis (i)**, lepton Yukawa with a PMNS factor: `(v_νφ)² + F² − 2v_νφF = (v_νφ − F)² ≥ 0`, and `v_ν² ≤ 1` by CHECK 27b so `\|U_ij φ F\|` is dominated by the same AM-GM as CHECK 8/20 | `0` |
| 29 | **accounting identity**: the whole BRST-exact gauge-fixing term `{Ω, Ψ}` of `bookGfTerm_eq` vanishes when `A_0 → 0` (every summand carries an explicit `A_0` factor) | `0` |
| 30a | the Gauss-law summand `𝒢_c A_{0d}` → 0 | `0` |
| 30b | the ghost-kinetic summand `[𝒢_c, A_{0d}] β χ` → 0 | `0` |
| 30c | the non-abelian ghost–field summand `f_{abc} A_{0a} χ_c β_b` → 0 | `0` |

Module final lines: `ALL SM HIGGS/CKM/PMNS/QUARK/LEPTON CHECKS DONE`
then `ALL SM ACCOUNTING IDENTITY CHECKS DONE (QYM spatial-only; QG non-ADM)`.

### PART F — the accounting identity (h and N unchanged)

On the temporal (Weyl) gauge slice `A_0 = 0` the BRST-exact contribution
`{Ω, Ψ}` with `Ψ = i ψ_a A_{0a}` is identically zero (CHECK 29, 30a–c), so the
records of

    h = h_Gauge + h_Higgs + h_Dirac + h_Yukawa,   N = N_0 + c_0 I

need **no extra gauge-fixing/ghost summand** — the accounting of `h` and `N` is
already complete.  Lean counterparts:
`BookProof.BookBrstGaugeFixing.bookGfTerm_eq_zero_of_Afield0`,
`su2_bookGfTerm_eq_zero_of_Afield0` (QYM: spatial components only),
`sm_bookGfTerm_eq_zero_of_Afield0`.

**QYM.**  Pure Yang–Mills inherits the same identity (the `su(2)`/`su(3)`
instances): the Weyl gauge may work with the 3D spatial field components and
the free/abelian ghosts drop out of the physical sector
(`ChapterYangMillsGhostSector.lean`, `ChapterGaugeWeylResidual.lean`).

**QG (non-ADM).**  book.tex ~8226–8244: the book's 3D reduction fixes the
globally defined time-like vector `v^μ = δ^μ_0`; it is **not** the ADM
approximation (constraints differ; ADM only weakly hyperbolic).  The ghosts of
the resulting BRST charge are **constant in the timepiece**, and the charge
keeps the **same functional form as the 4D formalism** — explicitly different
from the ADM BRST charge.  The frame gauge-fixing fermion is
`{G, i b_j A_0^j}` (`Book/Starobinsky.lean`), the gravity analogue of the YM
`A_0 = 0`; the `A_0 = 0` vanishing proved here is **not** asserted for that
frame fermion.

Module final lines after PART F: `ALL SM HIGGS/CKM/PMNS/QUARK/LEPTON CHECKS DONE`
then `ALL SM ACCOUNTING IDENTITY CHECKS DONE (QYM spatial-only; QG non-ADM)`.

The checks are the symbolic certificate of the three hypotheses **with the Higgs sector, the
CKM matrix, the quark covariant derivative, and the lepton sector (three right-handed
neutrinos + PMNS) included**:

* **hypothesis (i)** `±h ≤ c₁ N`: CHECKs 5–8, 11 (gauge/Higgs-potential/Yukawa) **plus**
  CHECKs 12–15 (Higgs covariant kinetic), 19–21 (CKM-bounded Yukawa, quark color),
  23–24 (Higgs and quark assemblies), **27–28 (PMNS-bounded lepton Yukawa)**;
* **hypothesis (ii)** `[h,N]` first order: CHECK 2/2b (gauge) **plus** CHECK 16 (Higgs with full `V`);
* **hypothesis (iii)** `[N,[N,h]]` quadratic in `p`: CHECK 3/4b (gauge) **plus** CHECK 22 (Higgs);
* **CKM unitarity** (consumed by the quark/Yukawa rows): CHECK 17–19 (Cabibbo `VV^T = I`,
  biunitary mass preservation, `|V_ij| ≤ 1`);
* **PMNS unitarity** (consumed by the lepton/Yukawa rows): CHECK 25–27 (Pontecorvo `UU^T = I`,
  biunitary `M_ν`/`M_e`, `|U_ij| ≤ 1`) — exact lepton mirrors of CHECK 17–19
  (`book.tex`: *“The lepton sector with three right handed neutrinos is analogous…”*;
  Majorana masses / see-saw are out of scope).

## The core and the lift (obligations (i) and (ii) of plan §D6b)

* **Obligation (i): `N` ESA on its core.**  Every summand of `N_0` is positive and confining:
  the non-abelian/Higgs summands `p^2 + q^4` (CHECK 9b) and the abelian/derivative summands
  `p^2 + q^2` (CHECK 9a, the oscillator `a^dag a + 1`), plus the fermionic 3D oscillator
  `|x|^2 + 1`.  Sums of uncoupled positive confining operators of this kind are essentially
  self-adjoint on the compactly supported smooth core `C_c^inf`, so `N` qualifies both as the
  comparison and as an ESA operator on the chosen core.
* **Obligation (ii): the lifted Hamiltonian on the lifted core.**  `H = dΓ(h)`, and the lift is
  a **different operator** from `h`; the lifted core is the finite-particle (anti)symmetric
  tensor core built from `C_c^inf`.  CHECK 10a/10b certify the second-quantization identity that
  fixes that core (`dΓ(N)` acts one-particle-wise).  The Hilbert-space statement — `dΓ(h)` is ESA
  on the finite-particle domain — is Nelson's tensor-product theorem plus the core-transfer /
  `dΓ`-ESA wave in the tree (`dGamma_essentiallySelfAdjointOn_fockCore`,
  `dGamma_essentiallySelfAdjointOn_of_esa`), not a symbolic computation.

## What the module does **not** certify

Cadabra is a symbolic engine on classical expressions.  It certifies: the canonical-pair
commutator shapes (by explicit differentiation), the elementary domination/AM-GM identities, the
sector coefficient arithmetic, and the core side.  It does **not** certify:

* the Hilbert-space statements — symmetry of `h` on `C_c^inf`, positivity of `N` as a quadratic
* the numerical realizations of the full budget / BRST / electroweak polarizations: the Fock-side
  inventory of `D_B = 163`, the 12-ghost charge `sm_brst_charge`, and the 11-mode `W±/Z/γ`
  polarization Hamiltonian are pinned by the four §5.24u tests in
  `fock_sirk/tests/sm_validation.rs` (`sm_db_163_mode_budget`,
  `sm_full_free_field_structure`, `sm_brst_ghosts_nilpotent`,
  `sm_weak_boson_polarizations`) — see `docs/NUMERICAL_VALIDATION_GUIDE.md` §5.24u;
  form, `N + 1` onto, the graph-core approximation, the existence of the Friedrichs extension,
  and the lift of the criterion (Lean);
* the **Grassmann/fermion CAR algebra** and the ghost sector (`{psi_A, psi^dag_B} = delta_AB`,
  nilpotent BRST charges): the module treats the fermion bilinears `Psi^dag ... Psi` as
  number-like scalar factors and checks only the AM-GM domination (CHECK 8, 20), not the CAR;
* the **spinor/Lorentz structure** of `h_Dirac` (`gamma^0 gamma.D`): only its quadratic
  domination by `-Delta + |x|^2 + 1` is asserted, with the covariant-derivative cross-terms
  bounded by CHECK 21/24 (Young identities) — the gamma-matrix algebra itself is Lean;
* the **3-generation CKM matrix** and the **3-generation PMNS matrix** as concrete numerical
  objects: CHECK 17–19 (CKM) and CHECK 25–27 (PMNS) certify the *unitarity identities*
  (row norm 1, `|V_ij|, |U_ij| ≤ 1`, biunitary mass preservation) that the domination
  estimates consume; the measured Wolfenstein / PMNS parameters (θ₁₂, θ₂₃, θ₁₃, δ_CP,
  mass ordering) are experimental input;
* **Majorana masses, the see-saw, and neutrino mass generation as dynamics** — `book.tex`
  works *“in the absence of Majorana masses”*; only the Dirac-type Yukawa `M_ν φ ν_R` and
  its biunitary/PMNS algebra are in `h`.

This is the same division of labour as the NS/QG/YM modules (see `VERIFY_FARIS_LAVINE_N.md` and
`../timepiece/HAMILTONIAN_AUDIT_20260918.md` §9).

## Files

* `docs/faris_lavine_n_sm.cdb` — the module.
* `docs/faris_lavine_n_ns.cdb`, `docs/faris_lavine_n_qg.cdb` — the NS/QG counterparts.
* `docs/VERIFY_FARIS_LAVINE_N.md` — the NS/QG evidence base (incl. the core/lift section).
* `../timepiece/CONSOLIDATED_PLAN.md` §D6b — the `N`-and-core work list (now including SM).
