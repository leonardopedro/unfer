# Real scalar λφ⁴ — numerical research and validation plan (Hashimoto–SIRK)

> **Authoritative references:**
> - `../../timepiece/book.tex` (Mehler Gaussian Hilbert space, continuous momentum
>   space, Quadratic Ordering $H = d\Gamma(h)$, no lattice discretization)
> - `../../papers/Hashimoto.md` (Hashimoto & Nodera, JJIAM 2019: Shift-invert
>   Rational Krylov method for an operator $\phi$-function of an unbounded linear
>   operator)
> - [Master cheap-LLM plan](../../timepiece/PHI4_IMPROVEMENT_PLAN.md) ·
>   [Lean contract](../../timepiece/PHI4_FORMALIZATION_PLAN.md) ·
>   [Cadabra2 derivation](PHI4_ANALYTIC_PLAN.md)

**Primary numerical objective.** Evolve a single outer excitation of the
continuous-momentum $\lambda\phi^4$ inner Hamiltonian

$$
h_{\mathrm{cont}}(\vec{k}, \phi)
  = \frac{1}{2} p_\phi^2
  + \frac{1}{2}(k_1^2 + k_2^2 + k_3^2)\,\phi^2
  + \frac{\lambda}{4}(\phi^2 - v^2)^2
$$

on $L^2(\mathbb{R}^3_{\vec{k}} \times \mathbb{R}_\phi)$ inside the outer
Quadratic Ordering enclosure $H = d\Gamma(h_{\mathrm{cont}})$. **Momentum
$\vec{k} \in \mathbb{R}^3$ and field $\phi \in \mathbb{R}$ are continuous
(never a momentum sum or spatial lattice); no spatial-derivative jet constraints
are needed because the quadratic gradient is diagonal multiplication by
$|\vec{k}|^2$. Only the Krylov subspace and finite Mehler–Hermite occupation
window in SIRK approximations are discrete.**

---

## 1. Hashimoto–SIRK Solver Contract (`../papers/Hashimoto.md` → `fock_sirk`)

Every primary $\lambda\phi^4$ trajectory MUST run through the genuine
shift-invert rational Krylov engine in [joint_sirk.rs](../fock_sirk/src/joint_sirk.rs)
with full-action error certification from [sm_defect.rs](../fock_sirk/src/sm_defect.rs):

1. **Unbounded skew-adjoint generator & bounded resolvents (`Hashimoto.md` Def. 2.4):**
   Set $A = -itH$ so $\phi_0(A)v = e^{-itH}v$ and $\Lambda(A), W(A) \subseteq i\mathbb{R} \subseteq \mathbb{C}_{\le 0}$.
   Never use polynomial Arnoldi (`JointKrylovMethod::ForwardProducts`) as the
   primary solver — `Hashimoto.md` §3.1 Eq. (6) proves Arnoldi has no convergence
   guarantee when $W(A)$ is unbounded. Always use **`JointKrylovMethod::ShiftInvert`**.
2. **Hashimoto decreasing shift schedule (`Hashimoto.md` §4):**
   Use $\gamma_j = N - hj > 0$ ($j = 1, \dots, m$, $1 \le m < N/h$; physical
   shifts $\tilde\gamma_j = \gamma_j / t$ via `hashimoto_unitary_shifts`,
   $z_j = i\tilde\gamma_j$). Construct the rational Krylov subspace
   $\mathcal{Q}_m(\{X_j\}_{j=1}^{m-1}, v)$ (Eq. (8)), compress the last
   resolvent $B_m = V_m^\dagger (\tilde\gamma_m I + i P H P)^{-1} V_m$ (Eq. (10)),
   form the SIRK generator $G_m = \tilde\gamma_m I - B_m^{-1}$, and reconstruct
   $y_m(t) = V_m e^{t G_m} V_m^\dagger \psi$ (distinct from the Ritz exponential
   $V_m e^{-it V_m^\dagger H V_m} V_m^\dagger \psi$).
3. **Separate error budgets — Theorem 4.1 vs. window truncation vs. GMRES conditioning:**
   - `Hashimoto.md` Theorem 4.1 (Eq. (12)) gives the exponential-in-$m$ factor
     $e^{-hm}$ for exact unbounded resolvents $X_j$, while §5 documents the
     practical trade-off: smaller $\gamma_m = N - hm$ worsens GMRES conditioning.
   - Inside a finite Mehler–Hermite occupation window $P_C$ (cutoff $C$), two
     distinct error sources exist: (i) **window-tail leakage** $(I - P_C) H V_m$,
     controlled by increasing $C$, and (ii) **rational Krylov compression error**
     inside $P_C$, controlled by increasing Krylov rank $m$ (with $m < N/h$) and
     tightening GMRES tolerance.
   - Always compute the **full unprojected defect** $D = i V_m G_m - H V_m$
     using complete unprojected images `HV` (never projected `P H P V`), and
     certify state and return-probability intervals via `sm_defect::defect_terms`
     and outward-rounded `sm_defect::interval_terms`.

---

## 2. Reuse and Named Comparison Baselines

- **Ordered inner action & outer lift:** Use `JointInnerBasis`, same-mode
  `InnerExpr::coordinate(0)` and `InnerExpr::momentum(0)` for $(\phi, p_\phi)$,
  continuous momentum multipliers `InnerExpr::coordinate(1..=3)` for
  $(k_1, k_2, k_3)$, and `InnerExpr::d_gamma` so one outer excitation carries
  all four continuous variables and retains every contraction and
  $\tfrac{\lambda v^4}{4}$.
  [Ordered action](../nested_fock_algebra/src/exact_inner.rs)
- **Rational solver & defect certificate:** Use `solve_joint_sirk`
  (`JointKrylovMethod::ShiftInvert`), `hashimoto_unitary_shifts`,
  `sm_defect::defect_terms`, and `sm_defect::interval_terms`.
  [Rational solver](../fock_sirk/src/joint_sirk.rs) ·
  [Defect certificate](../fock_sirk/src/sm_defect.rs)
- **Named comparison baselines (not primary models):**
  - [Anharmonic oscillator](../fock_sirk/examples/anharmonic_oscillator.rs)
    (positive quadratic well + legacy forward products)
  - [Legacy SM Higgs](../fock_sirk/tests/sm_higgs_mechanism_sirk.rs)
    (decoupled field/momentum modes + legacy contraction-dropping CAS)
  - [SM N12 pilot](../fock_sirk/examples/sm_finite_mode.rs) ·
    [SM research plan](SM_RESEARCH_PLAN.md) (documents why products of
    compressed coordinate matrices fail)

---

## 3. Experiment Sequence (Aligned with `P11`–`P13`)

Detailed step-by-step specifications live in
[../../timepiece/PHI4_IMPROVEMENT_PLAN.md](../../timepiece/PHI4_IMPROVEMENT_PLAN.md)
(`P0`, `P11`–`P13`).

1. **Implement the continuous-momentum operator (`P11`).**
   In `nested_fock_algebra/src/phi4_momentum.rs`, build `ScalarMomCont` on the
   four continuous coordinates $(\phi, k_1, k_2, k_3)$ using
   `InnerExpr::{coordinate, momentum}` with
   $\frac{1}{2} p_\phi^2 + \frac{1}{2}(k_1^2 + k_2^2 + k_3^2)\phi^2 + \frac{\lambda}{4}(\phi^2 - v^2)^2$
   (retaining $\frac{\lambda v^4}{4}$ and parametrizing continuous Mehler widths
   `width_phi`, `width_k`). At unit parameters and widths, match all $11 \times 3$
   $S_3$-symmetric entries of `docs/fixtures/phi4_momentum_mehler.json` (mean
   $17/8$, full variance $553/32$, outside-3-mode-window squared $9/2$, nonzero
   $\mathrm{He}_2(\phi)K_2$ coupling $1/2$) to $< 10^{-12}$, and verify
   $d\Gamma(h)\Omega = 0$, outer shifts $0, c, 2c$, and $d\Gamma(B)^2 \ne d\Gamma(B^2)$.
2. **Verify exact rank-2 Hashimoto–SIRK against Cadabra2 & Lean (`P12`, test 1).**
   On the degree-$\le 2$ even Hermite window (`cutoff = 2`) of `ScalarMomCont` at
   unit parameters, run `solve_joint_sirk` (`JointKrylovMethod::ShiftInvert`)
   with Hashimoto shifts $\tilde\gamma = [10, 5]$ ($N_{\mathrm{phys}}=15, h_{\mathrm{phys}}=5, m=2 < 3$)
   at $t = 0.01$. Verify $p(0.01) \in [319447/320000,\; 39999/40000]$,
   `projected_action_tail_norms[0]^2 ≈ 9/2` ($< 10^{-10}$), `hermiticity_defect < 1e-12`,
   and exact dissipativity $\operatorname{Re}(G_m) \le 0$.
3. **Independent Hermite-cutoff $C$ and Krylov-rank $m$ sweeps (`P12`, test 2).**
   Prepare one outer excitation in the continuous unit-width Gaussian $\psi$,
   bounded observable $O = |\psi\rangle\langle\psi|$, $\lambda = v = 1$, $t = 0.05$:
   - Fix Krylov rank $m = 2$ and sweep Hermite cutoff $C \in \{4, 8, 12\}$:
     confirm window-tail norm $(I - P_C)HV$ shrinks monotonically with $C$.
   - Fix cutoff $C = 8$ and sweep Krylov rank $m \in \{2, 4, 6\}$ using
     `hashimoto_unitary_shifts` ($m < N/h$): confirm inside-window rational error
     decays rapidly (matching `Hashimoto.md` Theorem 4.1's $e^{-hm}$ behavior on
     the fixed window) while recording GMRES iterations and conditioning (`§5`)
     and keeping the outside-window tail reported separately.
   - Compare `JointKrylovMethod::ShiftInvert` against labeled diagnostic
     `JointKrylovMethod::ForwardProducts` at the same rank to verify why Arnoldi
     deteriorates as $C$ grows (`Hashimoto.md` §3.1).
4. **Certified coupling and continuous momentum-spread responses (`P12`, test 3).**
   Run paired cells with identical windows, seeds, and Hashimoto shifts at
   $v = 1, C = 8, m = 4, t = 0.05$:
   - $\lambda = 1.0$ vs. $\lambda = 1.1$ (plus null control $\lambda = 1.0$ vs. $1.0$).
   - Continuous momentum wave-packet spread `width_k = 1.0` vs. `width_k = 1.5`
     (transporting $h$, $\psi$, and $O$ consistently without discretizing $\vec{k}$).
   Bound each trajectory's full-action state error $\varepsilon$ via
   `sm_defect::defect_terms` + outward-rounded `interval_terms`, charge
   $2\varepsilon + \varepsilon^2$ per probability estimate, add both budgets for
   the paired difference, and assert: null control contains `0.0`, while both
   physical response intervals strictly exclude `0.0`.
5. **Parity selection, unconditional inner gap, and comparison contrast (`P12`, tests 4–5).**
   - Evolve even ($\mathrm{He}_0$) and odd ($\mathrm{He}_1(\phi)$) seeds at
     $C = 8, m = 4$; assert exact $\mathbb{Z}_2$ parity conservation
     ($\phi \mapsto -\phi$) and assert every Ritz eigenvalue satisfies Lean's
     unconditional absolute inner lower bound $\theta_0 \ge 7/64$ (reporting
     the odd–even splitting explicitly as an inner excitation splitting, never
     as the outer vacuum gap).
   - Contrast continuous `ScalarMomCont` (mean $17/8$, variance $553/32$, nonzero
     $\mathrm{He}_2(\phi)K_2$ coupling $1/2$), frozen auxiliary static-jet
     `jetModel` (mean $17/8$, variance $145/32$, zero coupling), and frozen
     2-site lattice `ScalarGrid::two_site` (mean $13/4$, variance $337/16$).
6. **Reproducible JSON evidence runner (`P13`).**
   Create `fock_sirk/examples/phi4_momentum_sirk.rs` emitting the full sweep
   record to `docs/fixtures/phi4_momentum_sirk_sweep.json`.

---

## 4. Existing Lattice Comparison (Frozen)

`ScalarGrid::periodic` and `two_site_mode_hamiltonian` implement a separate
periodic-lattice comparison operator; do not extend it further. Before starting
the primary `phi4_momentum` suite, fix the unweighted Hermite-tail norm bug in
`two_site_return_lies_in_analytic_interval` (`P0` in
`../../timepiece/PHI4_IMPROVEMENT_PLAN.md`). All new Rust code lives in
`nested_fock_algebra/src/phi4_momentum.rs`, `fock_sirk/tests/phi4_momentum_sirk.rs`,
and `fock_sirk/examples/phi4_momentum_sirk.rs` (`P11`–`P13`).
[Comparison builder](../nested_fock_algebra/src/phi4_scalar.rs) ·
[Exact comparison fixture](fixtures/phi4_mehler.json)

---

## 5. Evidence to Retain

For each cell in `docs/fixtures/phi4_momentum_sirk_sweep.json`, store: source
revision, fixture SHA-256 digest, model and energy convention, seed and
observable, continuous Mehler widths (`width_phi`, `width_k`), inner and outer
occupations, Hermite cutoff $C$, requested and attained Krylov rank $m$,
Hashimoto parameters $(N, h, t)$ and actual shifts $\tilde\gamma_j$, GMRES
iteration counts and shifted-solve residuals, full unprojected defect
(`inside` vs. `outside` window split), outward-rounded `interval_terms` error
bounds, return probability intervals, Ritz values and residuals, runtime,
memory, and any failure reason. Mark measurements, conditional bounds, and
kernel-checked conclusions separately.
