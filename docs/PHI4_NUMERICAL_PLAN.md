# Real scalar λφ⁴ — numerical plan for Mehler and Hashimoto–SIRK

## 1. The operator and the observable

The existing Rust suite evolves one outer excitation. Its normalized inner seed is a
Gaussian on continuous $(k,s)\in\mathbb R^3\times\mathbb R$, and its observable is
the return probability $|\langle\psi,e^{-ith}\psi\rangle|^2$. The implemented
`ScalarMomCont` uses angular wavevector $k$ and

$$
h=\tfrac12 p_s^2+\tfrac12|k|^2s^2+\tfrac\lambda4(s^2-v^2)^2.
$$

The two occurrences of a field at the same inner coordinate share its ladder operators,
with $s=b+b^\dagger$, $p_s=i(b^\dagger-b)/2$ at unit width. Apply complete ordered
words, including contractions and the constant, before projecting. The outer operator is
$H=\sum_{ij}\langle e_i,h e_j\rangle C^\dagger(e_i)A(e_j)$. It annihilates the
empty outer vacuum, agrees with $h$ on one excitation, and lifts $cI$ to $cN_{\rm out}$.
A finite list of Hermite coefficients represents functions on continuous momentum space.
The quadratic spatial gradient gives a diagonal Fourier multiplier and requires no
derivative constraints. The spatially local quartic couples distinct momenta and is a
separate operator development in §4.
[Scalar builder](../nested_fock_algebra/src/phi4_momentum.rs) · [Analytic plan](PHI4_ANALYTIC_PLAN.md)

## 2. Reproducible fibre baseline (P11–P13)

| Existing calculation | Result and regression use |
|---|---|
| Ordered action at unit widths, $\lambda=v=1$ | Compare all three columns in the eleven-mode Mehler table using the physical Gram metric. Mean $17/8$, variance $553/32$, seed tail squared $9/2$. The independent-jet comparison has the same mean but variance $145/32$ and lacks the $\mathrm{He}_2(s)\sum_j\mathrm{He}_2(k_j)$ coupling. |
| Rank two, degree-two even window, physical shifts $10,5$, $t=0.01$ | The SIRK return is $0.9978594340$, the dense same-window reference $0.9987243519$, and the $C=12$ reference $0.9982985830$. Compare the last with the full-orbit analytic interval $[319447/320000,39999/40000]$. Record the raw compressed discrepancy and leakage separately. |
| Independent cutoff and rank sweeps at $t=0.05$ | At fixed rank two the reported tail falls from about $1.79$ to $0.17$ over $C=4$ to $26$. At $C=8$ the suite attains ranks $2,4,6$. These observations do not bound the maximum error for all cutoffs, ranks or times. Forward products are a measured same-window comparison. |
| Response to coupling or packet width at $C=8,m=4$ | Dense-window differences are $0.00161315$ for $\lambda:1\to1.1$ and $0.05616051$ for momentum width $1\to1.5$. The identical-cell difference is zero. The wider unit-coordinate seed represents a physically wider momentum packet. |
| Absolute floor and parity | Check each sector against the unit fibre lower bound $7/64$ and record the odd–even splitting separately. Finite Ritz values above the floor are consistency checks. |

The sweep's state-based estimates put zero inside both response intervals. The per-cell
amplitude-based probability charges are approximately $0.000763$, $0.001024$ and
$0.01189$ (base, coupled, spread). Their paired charge is $0.001786$ for coupling
and $0.01266$ for spread. The computed spread interval excludes zero and the coupling
interval includes it. The JSON field named `certified` uses a sampled overlap integral.
The field named `exact` is the exact seed column of the declared window's evolution
(a Hermitian-Lanczos `exp(-i t H_window)` action; formerly a dense floating-point Padé
exponential, reproduced to about $10^{-15}$ — see [§5.8](VERIFY_PHI4_MOMENTUM.md)).
They provide reproducible comparisons. Neither supplies the missing error bounds in §3.
The periodic one- and two-site data and the static jet remain regression controls
for different Hamiltonians.
[Exact table](fixtures/phi4_momentum_mehler.json) · [Sweep](fixtures/phi4_momentum_sirk_sweep.json) ·
[Tests](../fock_sirk/tests/phi4_momentum_sirk.rs) · [Replay, timings and measurements](VERIFY_PHI4_MOMENTUM.md)

## 3. Next fibre experiment — enclose a return response

1. **Reproduce the baseline before changing the solver.** Replay the scalar unit tests,
   the six momentum SIRK tests and the release evidence runner using the commands in the
   replay record. Save source and fixture digests. Preserve the unit rational table,
   parity sectors, empty-vacuum action and $0,c,2c$ shifts as regression checks.
   [Evidence runner](../fock_sirk/examples/phi4_momentum_sirk.rs)
2. **Use the physical metric and actual shifted solves.** For window $P_C$ set
   $h_C=P_C h P_C$ on its range. Use positive physical shifts
   $\tilde\gamma_j=(N-jd)/t$ with $m<N/d$. With basis $V$ and its physical dual
   $V^+$, compress the last resolvent to $B=V^+(\tilde\gamma_m I+i h_C)^{-1}V$.
   Evolve with $G=\tilde\gamma_m I-B^{-1}$. Compute the full defect
   $D=iVG-hV$ by applying $h$ before projection. Store $P_CD$ and $(I-P_C)D$,
   initial projection error, Gram conditioning, solve residuals, attained rank and
   any failed cell. Fix $m$ while varying $C$, then fix $C$ while varying $m$.
   Quote any Hashimoto approximation bound with its shift and approximation factors.
   [Rational solver](../fock_sirk/src/joint_sirk.rs)
3. **Enclose the propagated defect.** A sampled maximum can miss a peak between
   times. First enclose the conservative Frobenius/product estimate using verified
   inputs or bound each time panel with a derivative majorant. For the sharper
   observable estimate derive the dual weight $e^{+i(t-s)h}\psi$ in the Duhamel
   pairing. The current overlap samples a forward reduced flow; justify any symmetry
   substitution on the actual residual, or propagate the adjoint and bound its error.
   Enclose the overlap, replacement integral and initial leakage. Include quadrature
   remainder, shifted solves, Gram inversion, reduced exponential and rounding.
   Refinement agreement and outward rounding of final endpoints alone leave these
   errors open. Report a finite-window bound first, then include the full-action tail
   for a fibre bound.
   [Defect and amplitude implementation](../fock_sirk/src/sm_defect.rs)
4. **Repeat the paired test.** For amplitude error $a$ and a normalized physical
   seed, $2a+a^2$ bounds the return-probability error. Add independently enclosed
   charges from both trajectories. Keep the interval when it crosses zero. Require
   the null interval to contain zero and report which physical response, if any,
   excludes it after *all* charges. Add a change-of-basis control that transports
   the same physical seed, operator and observable together. Its exact response
   is zero, unlike changing the physical packet spread at fixed Hamiltonian.

## 4. First numerical test of a quartic coupling distinct momenta

1. **Build a continuous packet reference.** Take the two modulated real-field packets
   and localization $\chi$ from analytic step 2. In the cycles-per-length convention
   compare the coefficient of $a^2b^2$ in the localized wall energy of $a f_1+b f_2$
   against the Cadabra2 value. Independently evaluate the mixed convolution at a
   nonzero transfer $Q$ so that using $Q+p$ in place of $Q-p$ on a polarized pair
   fails. Use analytic Gaussian integrals or continuous quadrature with explicit tail
   and integration errors. Record packet centres, widths, Fourier measure and $v$.
   Quadrature points serve only to evaluate an integral.
2. **Test the ordered quantum matrix elements.** Once the inner representation and
   core are specified, project the *complete* smeared operator into two overlapping
   Mehler windows. Retain contractions, quadratic and constant terms. Compare an
   off-diagonal mixed-packet element with an independently derived symbolic value,
   check Hermiticity in the Gram metric, and require deleting the mixed contribution
   to fail. Check the outer vacuum, one-excitation action and two-excitation shift.
   Keep the new operator and fixtures separately identifiable from `ScalarMomCont`.
3. **Evolve the identified operator.** Repeat the shift-invert protocol on the same
   packets, with an independently evaluated finite-window reference and the full-action
   defect. Vary window, rank and integration tolerance one at a time at fixed physical
   smearing and localization. Record those scales separately when varying them later.
   A resolved mixed matrix element establishes coupling at the specified scales.
   Cutoff-independent dynamics or a positive spectral floor for that interaction
   require estimates for that operator.
   [Symbolic acceptance tests](PHI4_ANALYTIC_PLAN.md) ·
   [Scalar proof work order](../../timepiece/PHI4_IMPROVEMENT_PLAN.md)
