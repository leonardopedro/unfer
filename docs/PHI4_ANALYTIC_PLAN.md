# Real scalar λφ⁴ — analytical and Cadabra2 research plan

> **Authoritative references:**
> - `../../timepiece/book.tex` (Mehler Gaussian Hilbert space, continuous momentum
>   space, Quadratic Ordering $H = d\Gamma(h)$, no lattice discretization)
> - `../../papers/Hashimoto.md` (Hashimoto & Nodera, JJIAM 2019: Shift-invert
>   Rational Krylov method for an operator $\phi$-function of an unbounded linear
>   operator)
> - [Master cheap-LLM plan](../../timepiece/PHI4_IMPROVEMENT_PLAN.md) ·
>   [Lean contract](../../timepiece/PHI4_FORMALIZATION_PLAN.md) ·
>   [Numerical program](PHI4_NUMERICAL_PLAN.md)

The primary analytic target is the constraint-free **continuous-momentum**
nested-Fock $\lambda\phi^4$ theory. **Momentum space $\vec{k} \in \mathbb{R}^3$
is continuous (integrals $\int_{\mathbb{R}^3} d^3k$, never discrete momentum
sums or spatial lattices); only the Krylov subspace and finite Mehler–Hermite
occupation window in SIRK approximations are discrete.** Existing periodic
lattice calculations (`phi4_mehler.cdb`) and the auxiliary decoupled static-jet
fibre (`jetModel`) are **frozen comparison benchmarks only**.

---

## 1. Continuous momentum space and Quadratic Ordering (`book.tex`)

Work in three continuous spatial dimensions and continuous time ($\hbar = c = 1$)
for one real scalar field $\phi$ with double-well potential
$U(\phi) = \tfrac{\lambda}{4}(\phi^2 - v^2)^2$ ($\lambda, v > 0$).

1. **Why no spatial-derivative constraints are needed for $\lambda\phi^4$:**
   Coordinate-space first-jet constraints $iD_x = 0$ in `book.tex` are trivially
   satisfied — and therefore avoided — by passing to continuous momentum space,
   where spatial derivatives become multiplication by wavevector $\vec{k} \in \mathbb{R}^3$
   and pointwise products become continuous momentum-convolution integrals
   $\int_{\mathbb{R}^3} d^3p$. For $\lambda\phi^4$ specifically:
   - The gradient energy is **quadratic** in $\phi$, so Plancherel turns it into
     a **pure diagonal continuous multiplier** (no derivative convolution and no
     jet constraints):
     $$\frac{1}{2}\int_{\mathbb{R}^3} |\nabla\phi(\vec{x})|^2\,d^3x = \frac{1}{2}\int_{\mathbb{R}^3} |\vec{k}|^2 |\hat\phi(\vec{k})|^2\,d^3k$$
     (with $(2\pi|\vec{k}|)^2$ under the $e^{-2\pi i\langle x,k\rangle}$ transform).
   - The double-well potential contains no spatial derivatives; in continuous
     momentum space it is the **continuous composite-pair integral**
     $$\frac{\lambda}{4}\int_{\mathbb{R}^3} (\phi(\vec{x})^2 - v^2)^2\,d^3x = \frac{\lambda}{4}\int_{\mathbb{R}^3} |\Psi(\vec{Q})|^2\,d^3Q, \qquad \Psi(\vec{Q}) = \int_{\mathbb{R}^3} \hat\phi(\vec{p})\,\hat\phi(\vec{Q}-\vec{p})\,d^3p - v^2\delta^{(3)}(\vec{Q}).$$
2. **Continuous one-particle Mehler Hilbert space and inner Hamiltonian:**
   On the separable continuous space
   $L^2(\mathbb{R}^3_{\vec{k}} \times \mathbb{R}_\phi) = L^2(\mathbb{R}^4)$ with
   Mehler Gaussian reference weight $e^{-(\|\vec{k}\|^2 + \phi^2)/2}$ (core
   polynomials $p(\phi, \vec{k})\,e^{-(\|\vec{k}\|^2+\phi^2)/4}$, probabilists'
   product Hermites $\mathrm{He}_\alpha$ with squared norm $\alpha!\,(2\pi)^2$),
   only $\phi$ carries a conjugate momentum $p_\phi$ ($[\phi, p_\phi] = i$),
   while $\vec{k} = (k_1, k_2, k_3) \in \mathbb{R}^3$ couples directly to
   $\phi^2$ through the continuous gradient:
   $$h_{\mathrm{cont}}(\vec{k}, \phi) = \frac{1}{2} p_\phi^2 + \frac{1}{2}(k_1^2 + k_2^2 + k_3^2)\,\phi^2 + \frac{\lambda}{4}(\phi^2 - v^2)^2.$$
   Retain the **entire** nonnegative square, including $\tfrac{\lambda v^4}{4}$.
   *(Honesty scope: $h_{\mathrm{cont}}$ realizes the exact continuous gradient
   $\frac{1}{2}|\vec{k}|^2\phi^2$ and the fibrewise-local double well on
   $L^2(\mathbb{R}^3_{\vec{k}}\times\mathbb{R}_\phi)$; promoting
   $\frac{\lambda}{4}\int_{\mathbb{R}^3}|\Psi(\vec{Q})|^2 d^3Q$ to a
   cross-momentum integral operator on the full field algebra is tracked
   separately and never approximated by a discrete momentum sum.)*
3. **Outer Quadratic Ordering enclosure (`book.tex` lines 4030–4059):**
   $$H = d\Gamma(h_{\mathrm{cont}}) = \int_{\mathbb{R}^3} d^3k \int_{\mathbb{R}} d\phi\; a^\dagger(\vec{k}, \phi)\, h_{\mathrm{cont}}(\vec{k}, \phi)\, a(\vec{k}, \phi) = \sum_{i,j} \langle e_i, h_{\mathrm{cont}} e_j\rangle\, C^\dagger(e_i) A(e_j),$$
   where $\{e_i(\vec{k}, \phi)\}$ is a countable orthonormal Mehler basis of the
   separable space $L^2(\mathbb{R}^3_{\vec{k}} \times \mathbb{R}_\phi)$ (never a
   discrete momentum grid). Every inner constant shift $cI$ lifts to
   $d\Gamma(h + cI) = d\Gamma(h) + c N_{\mathrm{out}}$; empty outer vacuum
   satisfies $H|\Omega_{\mathrm{out}}\rangle = 0$; products must be formed
   inside $h$ before lifting ($d\Gamma(B^2) \ne d\Gamma(B)^2$ for $n \ge 2$).

---

## 2. Unbounded-Operator Hashimoto–SIRK Identities (`../papers/Hashimoto.md`)

Set the skew-adjoint generator $A := -i t H$ (for real time $t > 0$ and
self-adjoint unbounded $H$), so $\Lambda(A), W(A) \subseteq i\mathbb{R} \subseteq \mathbb{C}_{\le 0}$
and real-time evolution is the $\phi_0$-function $e^{-itH}v = \phi_0(A)v$.

1. **Operator function via bounded resolvent (`Hashimoto.md` Definition 2.4):**
   For any $\gamma > 0$, $X := (\gamma I - A)^{-1} = (\gamma I + itH)^{-1}$ is
   bounded with $W(X) \subseteq \mathbb{C}_+$ (satisfying the skew-adjoint
   identity $\operatorname{Re} X = \gamma X^* X$), and
   $$\phi_0(A) = \psi_{0,\gamma}(X), \qquad \psi_{0,\gamma}(z) := \exp(\gamma - z^{-1}).$$
2. **Shift-invert rational Krylov subspace & rational identity (`Hashimoto.md` §4, Eqs. (8), (10), (11)):**
   Choose decreasing positive shifts $\gamma_j := N - hj > 0$ ($j = 1, \dots, m$,
   $N, h > 0$, $1 \le m < N/h$; equivalently physical shifts
   $\tilde\gamma_j = \gamma_j / t$ for $(\tilde\gamma_j I + iH)^{-1}$). With
   $X_j := (\gamma_j I - A)^{-1}$:
   - **Resolvent shift identity (Eq. (11) derivation):**
     $X_j = (I + h(m-j) X_m)^{-1} X_m$, whence
     $\mathcal{Q}_m(\{X_j\}_{j=1}^{m-1}, v) = \operatorname{span}\{v, X_1 v, \dots, X_{m-1}\cdots X_1 v\} = \{r(X_m)v \mid r \in \mathcal{R}_{m-1}^{\mathrm{SIRK}}\}$.
   - **Compressed resolvent & SIRK generator (Eq. (10)):**
     With basis $V_m$ (physical Gram metric $W$, dual $V_m^+ = (V_m^\dagger W V_m)^{-1} V_m^\dagger W$),
     $$B_m = V_m^+ X_m V_m = \mathbf{H}_m \mathbf{K}_m^{-1}, \qquad \mathbf{K}_m = \mathbf{T}_m - \mathbf{H}_m \mathbf{D}_m + \gamma_m \mathbf{H}_m, \qquad G_m = \gamma_m I - B_m^{-1}.$$
     The SIRK approximation is $V_m \psi_{0,\gamma_m}(B_m) V_m^+ v = V_m \exp(G_m) V_m^+ v$
     (distinct from polynomial Arnoldi $V_m \exp(V_m^+ A V_m) V_m^+ v$!).
3. **Exact dissipativity and Duhamel full-action residual:**
   In physical Gram metrics $W$ and $W_m = V_m^\dagger W V_m$, the resolvent
   identity yields exact dissipativity for the physical-time generator
   $\tilde G_m = \tilde\gamma_m I - \tilde B_m^{-1}$:
   $$\tilde G_m^\dagger W_m + W_m \tilde G_m = -2\tilde\gamma_m\, L_m^\dagger W L_m \le 0, \qquad L_m = \tilde R_m V_m \tilde B_m^{-1} - V_m.$$
   Forward evolution $e^{s \tilde G_m}$ is therefore contractive for all $s \ge 0$.
   Defining the **full unprojected defect** $D := i V_m \tilde G_m - h V_m$ (with
   the full unbounded action $h V_m$, never the projected $P h P V_m$), Duhamel
   integration gives the exact state-error bound
   $$\|e^{-itH}v - V_m e^{t\tilde G_m} V_m^+ v\| \le t\|D e_0\| + \frac{t^2}{2}\|D\|_{\mathrm{HS}}\|\tilde G_m e_0\|$$
   (complementing `Hashimoto.md` Theorem 4.1's $2C\|v\| e^{-hm} \min_r \|f_{k,N}-r\|_{\infty,\Sigma}$
   bound by pricing both finite Hermite-window leakage and rational compression
   error separately!).

---

## 3. Target Continuous-Momentum Derivations & Checks (`P1`–`P3`)

Detailed cheap-LLM execution instructions live in
[../../timepiece/PHI4_IMPROVEMENT_PLAN.md](../../timepiece/PHI4_IMPROVEMENT_PLAN.md)
(`P1`–`P3`). Do **not** discretize momentum space into sums, do **not** extend
periodic lattice derivations, and do **not** build spatial-derivative jet
constraints for $\lambda\phi^4$.

| Step | Calculation in `docs/phi4_momentum_nested.cdb` | Acceptance (all `(0 = ok): 0` for `verify_cdb_modules.sh --strict`) |
|---|---|---|
| **C1. Continuous Fourier gradient & composite quartic** | Native Cadabra2 expansion of $\frac{1}{2}(k_1^2+k_2^2+k_3^2)\phi^2 + \frac{\lambda}{4}(\phi^2-v^2)^2$ (pin `len == 6`); continuous SymPy integrals verifying Plancherel $\int_{\mathbb{R}} |\partial_x f|^2 dx = \int_{\mathbb{R}} (2\pi k)^2 |\hat f(k)|^2 dk$ and continuous composite-pair quartic $\int_{\mathbb{R}} f(x)^4 dx = \int_{\mathbb{R}} |\Psi(Q)|^2 dQ$ on $f(x)=e^{-\pi x^2}$. | Every zero-check prints `(0 = ok): 0`; exact constant $\lambda v^4/4$ retained; zero discrete momentum sums. |
| **C2. Coercivity square completion** | Native Cadabra2 identity $\frac{\lambda}{4}(\phi^2-v^2)^2 - \frac{\alpha^2}{2}\phi^2 + \frac{\alpha^2 v^2}{2} + \frac{\alpha^4}{4\lambda} - \frac{\lambda}{4}(\phi^2-v^2-\frac{\alpha^2}{\lambda})^2 = 0$, evaluated at $\lambda = v = 1, \alpha = \frac{1}{2}$. | Exact `0`, yielding unconditional lower bound $\frac{\alpha}{2} - \frac{\alpha^2 v^2}{2} - \frac{\alpha^4}{4\lambda} = \frac{7}{64}$ on all of $L^2(\mathbb{R}^3_{\vec{k}} \times \mathbb{R}_\phi)$. |
| **C3. Exact 11-mode $S_3$-symmetric Mehler action** | Unit-width Mehler derivatives on `unitMomCont` ($\lambda=v=1$) acting on the discrete 3-mode window $\{1, \mathrm{He}_2(\phi), K_2 = \sum_r \mathrm{He}_2(k_r)\}$ (Gram $W = \operatorname{diag}(1, 2, 6)$) through all 11 orthogonal output modes, plus odd seed $\mathrm{He}_1(\phi)$. | Mean $17/8$, full variance $553/32$, outside-window seed squared norm $9/2$, nonzero $\mathrm{He}_2(\phi)K_2$ coupling $1/2$, window Hermiticity $W h_{\mathrm{win}} - h_{\mathrm{win}}^T W = 0$. Mutation controls detect decoupling $k$ from $\phi$ (variance drop $51/4$), dropping $\lambda v^4/4$ ($1/4$), and dropping double contraction ($3$). |
| **C4. Exact Hashimoto–SIRK Eq. (10)/(11) & rank-2 certificate** | Shifts $\tilde\gamma_1 = 10, \tilde\gamma_2 = 5$ ($N_{\mathrm{phys}} = 15, h_{\mathrm{phys}} = 5, m = 2 < N/h = 3$) at $t = 1/100$: verify Hashimoto Eq. (11) resolvent relation $\tilde R_1 = (I + 5 \tilde R_2)^{-1} \tilde R_2$, Eq. (10) compressed resolvent $B = V^+ \tilde R_2 V$, generator $G = 5I - B^{-1} \ne -i V^+ h_{\mathrm{win}} V$, exact dissipation $G^\dagger W_2 + W_2 G + 10 L^\dagger W L = 0$, orthogonal inside/outside defect split (`outside[0,0] = 9/2`), and transverse floor $\ge 1/200$. | Certifies two-sided return probability interval $[319447/320000,\; 39999/40000]$ at $t = 1/100$; emits formatted JSON fixture `docs/fixtures/phi4_momentum_mehler.json` (schema `"phi4-momentum-cont-exact/v1"`). |
| **C5. Outer Quadratic Ordering $d\Gamma(h)$** | Verify outer vacuum annihilation `0`, inner-shift lift `0, c, 2c` on $n = 0, 1, 2$ outer excitations, and cross-particle composition defect $d\Gamma(B)^2 - d\Gamma(B^2) \ne 0$ at $n = 2$. | All `(0 = ok): 0`; cross-particle defect verified nonzero. |
| **C6. CI gate & evidence (`P2`–`P3`)** | Add `(0 = ok): 0` lines to frozen `phi4_mehler.cdb` (keeping existing markers), run `bash scripts/verify_cdb_modules.sh --strict`, and record versions, replay commands, and SHA-256 digests in `docs/VERIFY_PHI4_MOMENTUM.md`. | Both `phi4_momentum_nested` and `phi4_mehler` report `[PASS]` under `verify_cdb_modules.sh --strict`. |

---

## 4. Frozen Comparison Benchmarks Only (Do Not Extend)

The following earlier artifacts use different operators and are retained solely
as reproducible regression/contrast benchmarks:

- **Periodic lattice comparison (`docs/phi4_mehler.cdb`, `docs/fixtures/phi4_mehler.json`):**
  Finite-difference cubic grid with spacing $a$ and periodic neighbor links.
  - 1-site ($a = \lambda = v = 1$): mean $5/8$, variance $97/32$, return lower
    bound $1 - (97/32)t^2$ (`ChapterScalarSirk.lean`).
  - 2-site ($2\times 1\times 1$): uniform/staggered rotation gives mean $13/4$,
    variance $337/16$, aliased quartic $(u^4 + 6 u^2 w^2 + w^4)/2$, and rank-two
    analytic interval $[159663/160000,\; 39991/40000]$ at $t = 1/100$
    (`ChapterScalarTwoSite.lean`, `ChapterScalarRankTwo.lean`; 29 checks + 4
    mutation controls; SHA-256 `f980c5ac...` / `3694921d...`).
- **Auxiliary decoupled static-jet fibre (`BookProof/Phi4/ChapterScalarJet.lean`):**
  $h_{\mathrm{jet}} = \frac{1}{2}p_s^2 + \frac{1}{2}(j_1^2+j_2^2+j_3^2) + U(s)$
  with mean $17/8$ and variance $145/32$ (missing the continuous momentum–field
  coupling $\frac{1}{2}|\vec{k}|^2\phi^2$ of $h_{\mathrm{cont}}$).
- **Legacy Higgs & N12 scalar pilot:**
  [Higgs comparison](../fock_sirk/tests/sm_higgs_mechanism_sirk.rs) ·
  [Numerical program](PHI4_NUMERICAL_PLAN.md)
