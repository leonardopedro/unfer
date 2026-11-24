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

## 3. Exact results to reuse and next derivations

The scalar Cadabra2 module and its exact JSON fixture already exist. Its Gaussian
Fourier check computes $\int f^4=1/2$ for $f(x)=e^{-\pi x^2}$ in one dimension.
Its operator calculation retains the constant and contractions, completes the square
(giving the unit-parameter floor $7/64$ with the oscillator inequality), and evaluates
the full images of three even seed modes in eleven symmetric output modes. The Gaussian
mean is $17/8$, variance $553/32$, and squared seed tail outside the three modes $9/2$.
At physical shifts $10,5$ the rank-two calculation supplies rational dissipativity
and defect identities and return endpoints $319447/320000$ and $39999/40000$
at $t=1/100$. The endpoints concern the full fibre orbit under the analytic error
argument. They are not an interval test on the raw compressed return.
[Cadabra2 source](phi4_momentum_nested.cdb) · [Rational fixture](fixtures/phi4_momentum_mehler.json) ·
[Replay and checks](VERIFY_PHI4_MOMENTUM.md)

1. **Check the fibre proof against the fixture.** Reuse the core lower bound,
   full polynomial action, physical Gram matrix, compressed-resolvent inverse and
   full defect Gram in the scalar Lean sources. Trace the rational estimates through
   the complex Hilbert-space embedding and Duhamel estimate to a theorem on the
   normalized seed return. Record the actual closing declaration and its hypotheses
   before attributing the two endpoints to a kernel-checked orbit theorem.
   [Core bound](../../timepiece/BookProof/Phi4/ChapterScalarMomentumGap.lean) ·
   [Action and Gram identities](../../timepiece/BookProof/Phi4/ChapterScalarMomRankTwo.lean) ·
   [Orbit bridge](../../timepiece/BookProof/Phi4/ChapterScalarMomOrbit.lean)
2. **Compute a localized double well with mixed momenta.** For real Schwartz $f$ and
   real Schwartz localization $\chi$, set
   $B_\chi=\widehat\chi*(\widehat f*\widehat f)-v^2\widehat\chi$. Derive and
   test by two independent evaluations
   $$
   \tfrac\lambda4\int |\chi(x)|^2(f(x)^2-v^2)^2\,dx
     =\tfrac\lambda4\int |B_\chi(Q)|^2\,dQ.
   $$
   The unlocalized constant background on infinite volume would give a Dirac mass
   whose square has no ordinary integral. Keep the localization, its constant
   contribution and the Fourier measure explicit. Take polarized packets with
   $\widehat g_j(p)=e^{-\pi|p-u_j|^2}$ and distinct nonzero centres. Their mixed
   convolution at $Q=u_1+u_2$ equals $2^{-3/2}$ in three dimensions. Using $Q+p$
   in place of $Q-p$ misses that value for $u_1\ne0$. Then use real packets
   $f_j=g_j+\overline{g_j}$, $f=a f_1+b f_2$ for real $a,b$, and a real Gaussian
   $\chi$. The $a^2b^2$ energy coefficient must equal
   $\tfrac{3\lambda}{2}\int |\chi|^2 f_1^2 f_2^2\,dx$. Export both checks and
   the single-packet controls as exact fixtures. Omitting mixed pairs must change
   the mixed energy coefficient.
3. **Define the inner operator represented by the mixed vertex.** Specify continuous
   packet field operators, reality and commutation relations, common Mehler core,
   ordering and contractions. If ultraviolet smearing is needed, give its scale
   separately from the spatial localization. Derive the matrix elements of the
   smeared square, its adjoint and its positive quadratic form on that core.
   Exhibit a nonzero mixed-packet matrix element and match its convention to step 2.
   State the Hilbert space on which this is a one-particle operator for the *outer*
   lift. Pointwise multiplication of arbitrary $L^2$ fields does not define that form.
4. **Identify the dynamics before removing a regulator.** Prove dense definition,
   closability and semiboundedness, identify a self-adjoint realization, and state
   separately any core or essential-self-adjointness result. Then lift the identified
   operator and compare its one-excitation matrix elements with the numerical packet
   calculation. Any removal of localization or ultraviolet smearing needs a specified
   limiting topology, uniform estimates and identification of the limit. The single-fibre
   floor and return interval apply to their own operator throughout.

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
