# Real scalar λφ⁴ — analytic and Cadabra2 research plan

## 1. Which Hamiltonian enters the outer Fock space?

The implemented scalar has **one field-amplitude coordinate** $s$ and three continuous
momentum coordinates $k$. Its one-particle Hamiltonian acts fibre by fibre on
$L^2(\mathbb R^3_k\times\mathbb R_s)$,

$$
h_{\rm cont}=\tfrac12 p_s^2+\tfrac12 |k|^2s^2
             +\tfrac\lambda4(s^2-v^2)^2,\qquad \lambda,v>0.
$$

Here $k$ is angular wavevector. The Mehler basis consists of Hermite polynomials times
a Gaussian in the Lebesgue realization. The inner field and its conjugate momentum obey
$[s,p_s]=i$. The final Hamiltonian on the outer bosonic Fock space, written on the
finite-particle core in an orthonormal basis of complete inner states, is

$$
H=d\Gamma(h)=\sum_{i,j}\langle e_i,h e_j\rangle C^\dagger(e_i)A(e_j).
$$

The sum indexes basis states of a continuous Hilbert space. Form each ordered product
inside $h$ before taking its matrix element. The right annihilator kills the empty
outer vacuum, and the one-excitation sector evolves by $h$. Keeping the constant
$\lambda v^4/4$ matters because $d\Gamma(h+cI)=d\Gamma(h)+cN_{\rm out}$.
An absolute lower bound $h\ge bI$, $b>0$, gives $H\ge bN_{\rm out}$ for the
identified self-adjoint realization. An inner odd–even splitting measures a different energy.
[Manuscript](../../timepiece/book.tex) · [Scalar formalization plan](../../timepiece/PHI4_FORMALIZATION_PLAN.md)

**Where the spatial field enters.** For a real Schwartz field $f(x)$ use the Fourier
convention $\widehat f(\xi)=\int e^{-2\pi i x\cdot\xi} f(x)\,dx$, with $k=2\pi\xi$.
Plancherel turns the quadratic gradient energy into
$\tfrac12\int (2\pi)^2|\xi|^2|\widehat f(\xi)|^2\,d\xi$.
This accounts for the gradient through a multiplier and introduces no independent
spatial-derivative coordinates or constraints. A pointwise square in the potential gives

$$
\widehat{f^2}(Q)=\int \widehat f(p)\widehat f(Q-p)\,dp.
$$

Distinct momenta contribute to the same transfer $Q$. Multiplication by $(s^2-v^2)^2$
in a single fibre leaves its $k$ fixed. The spatially local interaction therefore requires
its own inner representation, form domain and operator identification. Section 3 gives
a test that distinguishes the two even before constructing quantum dynamics.
[Classical Fourier declarations](../../timepiece/BookProof/Phi4/ChapterScalarFourierIntegral.lean) ·
[Numerical plan](PHI4_NUMERICAL_PLAN.md)

## 2. Hashimoto–SIRK and the full-action defect

On the one-excitation sector take $A=-ith$ for a self-adjoint realization of $h$.
For $\gamma>0$ the bounded resolvent $X=(\gamma I-A)^{-1}$ satisfies
$\operatorname{Re}X=\gamma X^*X$. For decreasing shifts $\gamma_j=N-jd>0$,
the resolvent identity gives $X_j=(I+(m-j)dX_m)^{-1}X_m$. Successive shifted solves
generate a basis $V$ starting at the normalized seed. The finite-window implementation
solves with $h_C=P_C h P_C$ on the range of the Hermite projection $P_C$.
[Hashimoto and Nodera, §§2–4](../../papers/Hashimoto.md)

In physical time set $g=\gamma_m/t$, $R=(gI+i h_C)^{-1}$,
$B=V^+RV$ and $G=gI-B^{-1}$, where $V^+$ is the dual in the physical Gram metric.
The approximate state at time $t$ is $Ve^{tG}V^+\psi$. With $W$ the window metric,
$W_m=V^\dagger W V$ and $L=RVB^{-1}-V$, exact resolvent algebra yields

$$
G^\dagger W_m+W_mG=-2gL^\dagger W L.
$$

Thus $e^{sG}$ is contractive in $W_m$ for $s\ge0$. For the full error, evaluate
$D=iVG-hV$ with the *unprojected* action. When the basis lies in the operator domain
and $\psi=Vc$ exactly, Duhamel gives

$$
\|e^{-ith}\psi-Ve^{tG}c\|
\le \int_0^t\|De^{sG}c\|\,ds
\le t\|Dc\|+\tfrac12t^2\|D\|_{\rm HS}\|Gc\|_{W_m}.
$$

Use Hilbert–Schmidt norms in the physical metrics. For an inexact seed representation
add $\|\psi-Vc\|$. The identity explains why the three-mode window alone cannot price
the error: $hV$ has components in the other eight output modes of the exact fixture.

**Adjoint pairing symmetry for real symmetric $(h, \psi)$.** In the Duhamel return-amplitude
overlap, $\langle\psi, e^{-i(t-s)h} r(s)\rangle = \langle e^{+i(t-s)h}\psi,\, r(s)\rangle$.
Because $h$ and the Gaussian seed $\psi$ are real in the Mehler basis, complex conjugation
reverses the sign of time: $e^{+i(t-s)h}\psi = \overline{e^{-i(t-s)h}\psi}$. Omitting
complex conjugation on the reduced vector $w(t-s) \approx e^{-i(t-s)h}\psi$ evaluates
the wrong time sign (`sm_defect.rs`, queued fix `Q6`), whereas conjugating $w(t-s)$
uses the true real-symmetric adjoint flow (up to its explicitly bounded defect).

## 3. Exact results to reuse and next derivations (`Q3`–`Q5`)

The scalar Cadabra2 module [`phi4_momentum_nested.cdb`](phi4_momentum_nested.cdb) and its
exact rational fixture [`fixtures/phi4_momentum_mehler.json`](fixtures/phi4_momentum_mehler.json)
already pass all 32 `(0 = ok): 0` checks under `verify_cdb_modules.sh --strict`.
The Gaussian Fourier check computes $\int f^4 = 1/2$ for $f(x) = e^{-\pi x^2}$ in continuous 1D;
the operator checks retain $\lambda v^4/4$ and all contractions, complete the square for
the unit floor $7/64$, evaluate the full 11-mode $S_3$ action of `unitMomCont` (mean $17/8$,
variance $553/32$, outside-window tail $9/2$), verify rank-2 Hashimoto–SIRK at shifts
$(10, 5)$ with return interval $[319447/320000, 39999/40000]$ at $t = 1/100$, and check
outer shift $d\Gamma(h+cI) = d\Gamma(h)+cN_{\rm out}$ on occupations $0, 1, 2$.
**Throughout, only the Krylov space in SIRK approximations is discrete, never momentum space.**

1. **Continuous classical localized-wall checks & Lean alignment (`Q3`).**
   For $\chi(x) = e^{-\pi|x|^2}$ and continuous test fields $f_j(x) = 2\chi(x)\cos(2\pi u_j\cdot x)$
   with centres $u_0 = (1,0,0)$ and $u_1 = (0,2,0)$ ($\lambda = v = 1$, $k = 2\pi\xi$),
   `phi4_momentum_nested.cdb` verifies that the $a^2 b^2$ coefficient of the continuous
   integral $\frac{\lambda}{4}\int_{\mathbb{R}^3} |\chi(x)|^2 \big((a f_0 + b f_1)^2 - v^2\big)^2 d^3x$ is
   $\frac{3}{2}\int_{\mathbb{R}^3} \chi^2 f_0^2 f_1^2 = (1+e^{-2\pi/3})(1+e^{-8\pi/3})/\sqrt{6} \approx 0.458627352548$,
   versus $0.152875784183$ when the mixed pair is omitted, recorded in
   [`fixtures/phi4_localized_wall.json`](fixtures/phi4_localized_wall.json).
   *(Audit note: the draft two-mode Lean file `ChapterScalarPacketForm.lean` both discretized
   momentum space into two modes and had four math/terminology errors; `Q3` retires it and
   proves the continuous gradient Plancherel identity and continuous `localizedWall_energy`
   polarization directly on continuous $\mathcal{S}(\mathbb{R}^3)$ in
   [`ChapterScalarFourierIntegral.lean`](../../timepiece/BookProof/Phi4/ChapterScalarFourierIntegral.lean).)*
2. **Next Cadabra2 derivation (`Q4`): Continuous Mehler–Hermite Momentum-Convolution Integrals:**
   Extend [`phi4_momentum_nested.cdb`](phi4_momentum_nested.cdb) with exact **continuous**
   $\int_{\mathbb{R}^3} d^3p$ and $\int_{\mathbb{R}^3} d^3Q$ momentum-convolution integrals of
   continuous Mehler–Hermite functions $g_\alpha(p) = \prod_{r=1}^3 \mathrm{He}_{\alpha_r}(p_r) e^{-p_r^2/4}$
   on continuous $\mathbb{R}^3_{\vec p}$ (**zero discrete momentum sums and zero finite-mode projection**):
   - Evaluate closed-form continuous convolutions $\Psi_{\alpha,\beta}(Q) = \int_{\mathbb{R}^3} g_\alpha(p) g_\beta(Q - p)\,d^3p$
     for continuous pairs such as $(0,0,0)\!*\!(0,0,0)$, $(2,0,0)\!*\!(0,0,0)$, and $(2,0,0)\!*\!(0,2,0)$.
   - Verify exact continuous Plancherel equality $\int_{\mathbb{R}^3} |\Psi_{\alpha,\beta}(Q)|^2\,d^3Q = \int_{\mathbb{R}^3} |\widehat{g}_\alpha(x)\widehat{g}_\beta(x)|^2\,d^3x$
     via SymPy continuous integration.
   - Print `(0 = ok): 0` checks plus a `distinguish` mutation control removing the cross-coordinate factor,
     emit `phi4_momentum_convolution.json`, and pass `scripts/verify_cdb_modules.sh --strict`.
3. **Continuous dispersion & composite-pair positivity (`Q5`, `Q8`):**
   Lean (`Q5`) proves the continuous $\|\vec k\| \ge k_0$ dispersion floor on continuous
   $L^2(\mathbb{R}^3_{\vec k} \times \mathbb{R}_\phi)$ and `localizedWall_energy_nonneg` on $\mathcal{S}(\mathbb{R}^3)$;
   Rust (`Q8`) cross-checks the continuous Mehler–Hermite convolution integrals in
   `nested_fock_algebra/src/phi4_momentum.rs` against `docs/fixtures/phi4_momentum_convolution.json`.

[`Replay and checks`](VERIFY_PHI4_MOMENTUM.md) · [`Master work order`](../../timepiece/PHI4_IMPROVEMENT_PLAN.md) · [`Numerical plan`](PHI4_NUMERICAL_PLAN.md)

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
    mutation controls).
- **Auxiliary decoupled static-jet fibre (`BookProof/Phi4/ChapterScalarJet.lean`):**
  $h_{\mathrm{jet}} = \frac{1}{2}p_s^2 + \frac{1}{2}(j_1^2+j_2^2+j_3^2) + U(s)$
  with mean $17/8$ and variance $145/32$ (missing the continuous momentum–field
  coupling $\frac{1}{2}|\vec{k}|^2\phi^2$ of $h_{\mathrm{cont}}$).
- **Legacy Higgs & N12 scalar pilot:**
  [Higgs comparison](../fock_sirk/tests/sm_higgs_mechanism_sirk.rs) ·
  [Numerical program](PHI4_NUMERICAL_PLAN.md)
