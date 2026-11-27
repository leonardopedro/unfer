# Real scalar λφ⁴ — numerical plan for Mehler and Hashimoto–SIRK

## 1. The operator and the observable

The existing Rust suite evolves one outer excitation. Its normalized inner seed is a
Gaussian on continuous $(k,s)\in\mathbb R^3\times\mathbb R$, and its observable is
the return probability $|\langle\psi,e^{-ith}\psi\rangle|^2$. **Only the Krylov space
in the SIRK approximations is discrete, never momentum space.** Any finite Hermite
degree bound $C$ used in numerical resolvent solves is solely a solver-internal basis
truncation converging strongly to the full continuous operator as $C\to\infty$
(`Model.window_solve_convergence`). The implemented `ScalarMomCont` uses angular wavevector $k$ and

$$
h=\tfrac12 p_s^2+\tfrac12|k|^2s^2+\tfrac\lambda4(s^2-v^2)^2.
$$

The two occurrences of a field at the same inner coordinate share its ladder operators,
with $s=b+b^\dagger$, $p_s=i(b^\dagger-b)/2$ at unit width. Apply complete ordered
words, including contractions and the constant, before projecting. The outer operator is
$H=\sum_{ij}\langle e_i,h e_j\rangle C^\dagger(e_i)A(e_j)$ over a countable basis of
continuous functions on $\mathbb{R}^3_k\times\mathbb{R}_s$. It annihilates the
empty outer vacuum, agrees with $h$ on one excitation, and lifts $cI$ to $cN_{\rm out}$.
The quadratic spatial gradient gives a diagonal Fourier multiplier and requires no
derivative constraints. Continuous momentum-convolution integrals beyond fibrewise $h$ are
checked in §4.
[Scalar builder](../nested_fock_algebra/src/phi4_momentum.rs) · [Analytic plan](PHI4_ANALYTIC_PLAN.md)

## 2. Reproducible fibre baseline (P11–P13)

| Existing calculation | Result and regression use |
|---|---|
| Ordered action at unit widths, $\lambda=v=1$ | Compare all three columns in the eleven-mode Mehler table using the physical Gram metric. Mean $17/8$, variance $553/32$, seed tail squared $9/2$. The independent-jet comparison has the same mean but variance $145/32$ and lacks the $\mathrm{He}_2(s)\sum_j\mathrm{He}_2(k_j)$ coupling. |
| Rank two, degree-two even window, physical shifts $10,5$, $t=0.01$ | The SIRK return is $0.9978594340$, the dense same-window reference $0.9987243519$, and the $C=12$ reference $0.9982985830$. Compare the last with the full-orbit analytic interval $[319447/320000,39999/40000]$. Record the raw compressed discrepancy and leakage separately. |
| Independent cutoff and rank sweeps (`C = 4..44`) at $t=0.05$ | At fixed rank $m=2$ the outside-window tail shrinks monotonically $1.79 \to 0.0208$ over $C=4\dots 44$, while rank-4 inside-window error **plateaus flat at $2.88\times 10^{-3}$** across $C=8\dots 44$ ($151\text{ min}$, $4.24\text{ GB}$ RSS at $C=44$). **Do not extend $C > 24$ at fixed $m \le 4$** — the solver is Krylov-rank-limited here, not cutoff-limited. Forward products are a measured same-window comparison. |
| Response to coupling or packet width at $C=8,m=4$ | Dense-window differences are $-0.00161315$ for $\lambda:1\to1.1$ and $-0.05616051$ for `width_k`: $1\to1.5$. The identical-cell difference is zero. The wider unit-coordinate seed represents a physically wider continuous momentum packet. |
| Absolute floor and parity | Check each sector against the unit fibre lower bound $7/64$ and record the odd–even splitting separately. Finite Ritz values above the floor are consistency checks. |

The sweep's state-based estimates put zero inside both response intervals. The per-cell
amplitude-based probability charges are approximately $0.000763$, $0.001024$ and
$0.01189$ (base, coupled, spread), giving paired charges $0.001786$ (coupling) and
$0.01266$ (spread). Thus the `width_k` interval excludes zero while the $\lambda$
interval still includes it. Furthermore, the 2026-11-27 audit identified two defects
in the current [`amplitude_terms`](../fock_sirk/src/sm_defect.rs) helper: (1) it computes
`inner(&w, &r)` instead of `inner(&conjugate_nested(&w), &r)` for the real-symmetric
adjoint $e^{+i(t-s)h}\psi$ (see [`PHI4_ANALYTIC_PLAN.md`](PHI4_ANALYTIC_PLAN.md) §2), and
(2) its JSON field `certified` is a sampled floating-point trapezoid without panel derivative
remainders, GMRES residual bounds, or outward `Interval` rounding. Do not label it certified
until `Q6`–`Q7` land. The field named `exact` is the seed column of `exp(-i t H_window)`
via Hermitian Lanczos (`1e-13`). Periodic 1- and 2-site data and the static jet remain
frozen comparison controls for different Hamiltonians.
[Exact table](fixtures/phi4_momentum_mehler.json) · [Sweep](fixtures/phi4_momentum_sirk_sweep.json) ·
[Tests](../fock_sirk/tests/phi4_momentum_sirk.rs) · [Replay, timings and measurements](VERIFY_PHI4_MOMENTUM.md)

## 3. Next fibre experiments (`Q6`–`Q8`) — fix adjoint conjugation, enclose rigorously, and tune Hashimoto–SIRK

1. **Reproduce the baseline cleanly (`Q8`).** Replay the scalar unit tests and momentum
   integration suite, and gate cutoffs $C > 24$ in
   [`examples/phi4_momentum_sirk.rs`](../fock_sirk/examples/phi4_momentum_sirk.rs) behind an
   opt-in env var (`PHI4_EXTENDED_CUTOFFS=1`) so default fixture verification completes in `< 60 s`
   instead of `151 min`. Preserve the unit rational table, parity sectors, vacuum annihilation,
   and `0, c, 2c` shift checks as regressions.
2. **Fix the Duhamel conjugation bug in [`sm_defect.rs`](../fock_sirk/src/sm_defect.rs) (`Q6`).**
   For real symmetric $(H, \psi)$ (asserted explicitly on the window and seed),
   $e^{+i(t-s)H}\psi = \overline{e^{-i(t-s)H}\psi}$, so replace `inner(&w, &r)` with
   `inner(&conjugate_nested(&w), &r)`. Rename heuristic sampled helpers cleanly (stop calling
   un-enclosed sampled runs `certified_cell`).
3. **Implement rigorous interval-enclosed `amplitude_interval_terms` (`Q6`).**
   Using outward-rounded `Interval` arithmetic (`sm_defect::Interval`), enclose:
   (a) initial leakage,
   (b) composite trapezoid overlap and action integrals **plus exact panel second-derivative remainders**
       $\frac{t^3}{12 M^2}\max_s |d^2/ds^2\,\cdot|$ bounded from $\|G\|, \|D\|, \|V\|$ in `Interval` arithmetic,
   (c) triangular replacement term + remainder, and
   (d) GMRES `shift_solve_error_bounds` propagated from `JointSirkResult`.
   Verify with a small exact 3-mode dense $e^{+i(t-s)H}\psi$ conjugation test in `fock_sirk`.
4. **Tune Hashimoto–Nodera Krylov rank and shifts (`Q7`) per [`Hashimoto.md`](../../papers/Hashimoto.md) §4 Theorem 4.1:**
   Instead of pushing truncation $C$ past $24$, keep $C \in \{8, 12\}$ and sweep higher discrete Krylov rank
   $m \in \{4, 6, 8\}$ with Hashimoto–Nodera decreasing physical shifts $\tilde\gamma_j = (N - j d)/t$
   ($m < N/d$, moderate $g = \tilde\gamma_m$ controlling $W_m$ conditioning) so that the **rigorous**
   `amplitude_interval_terms` paired probability charge **strictly excludes zero for BOTH
   $\lambda: 1.0 \to 1.1$ AND `width_k`: $1.0 \to 1.5$** while containing zero on both null controls
   (identical cell and the verified diagonal-phase `change_of_basis` control, whose exact response
   is zero and compressed response measures `3.1e-15` inside `8.09e-5` with `‖Ph'P − PhP‖_F = 78.6`).

## 4. Continuous momentum-convolution verification (`Q8`)

1. **Continuous classical packet integral baseline (executed):** `localized_packet_wall` in
   [`nested_fock_algebra/src/phi4_momentum.rs`](../nested_fock_algebra/src/phi4_momentum.rs) evaluates
   continuous Gaussian Fourier overlaps for the localized double well ($0.458627352548$ mixed $a^2 b^2$
   coefficient, background $\lambda v^4/(4\cdot 2^{3/2})$ retained), checked against
   [`fixtures/phi4_localized_wall.json`](fixtures/phi4_localized_wall.json) in
   [`tests/phi4_momentum_sirk.rs`](../fock_sirk/tests/phi4_momentum_sirk.rs).
2. **Continuous Mehler–Hermite convolution fixture test (`Q8`, NO discrete momentum models):**
   Once `phi4_momentum_convolution.json` lands from `Q4`, add `continuous_mehler_convolution_matches_fixture`
   in [`nested_fock_algebra/src/phi4_momentum.rs`](../nested_fock_algebra/src/phi4_momentum.rs) verifying the
   closed-form continuous $\int_{\mathbb{R}^3} d^3p$ / $\int_{\mathbb{R}^3} d^3Q$ Mehler–Hermite pair convolutions
   and Plancherel norms to `1e-12` — **strictly on continuous $\mathbb{R}^3_{\vec k}$ with zero finite-mode or
   lattice momentum discretization**.

*(Scope note: All work stays strictly inside the continuous-momentum `book.tex` Mehler / nested-Fock
and `Hashimoto.md` SIRK framework; frozen lattice/static-jet/Higgs benchmarks are comparison controls only.)*
[`Symbolic plan`](PHI4_ANALYTIC_PLAN.md) · [`Master work order`](../../timepiece/PHI4_IMPROVEMENT_PLAN.md)
