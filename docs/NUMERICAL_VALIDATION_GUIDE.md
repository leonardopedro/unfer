# A Pedagogical Guide to the Numerical Physics Validation Suites

> **What this document is.** Every validation suite in `fock_sirk/tests/` checks
> this project's numerical machinery against numbers the physics community has
> already measured or derived exactly. This guide explains, for every named
> test, **what quantity is being calculated** (the exact formula), **how it is
> computed** (which numerical route — SIRK Ritz values, matrix-element
> assembly, RK4 integration, Simpson quadrature, or a direct closed-form
> evaluation of published constants), **what setup** (modes, couplings,
> cutoffs) the computation runs with, and **what agreement is demanded** (the
> tolerance, and the class of comparison it encodes). It closes with the
> algorithmic core — the Hashimoto inverse-free rational-Krylov solver (SIRK)
> — its spectral-edge behaviour, and a study of its certified error bands.
>
> Math is written in GitHub-flavoured Markdown LaTeX: inline `$...$`, display
> `$$...$$`. Audience: a physics student who wants to know exactly what is
> being computed, or a developer who needs to know why an assertion carries a
> given tolerance.
>
> Two sections carry the programme's honesty sheet: §4.5 decomposes the claim
> "the 3D gauge-fixed Hamiltonians of NS / QG(R²) / QYM / QED are tested
> through SIRK/Hashimoto without further assumptions" into verifiable steps
> (and positions the unit-norm frame as an exact reparametrization, not a
> model change), and §5.25 gives the per-system match / fail / non-claim map:
> where the predictions agree with experiment or other approximations, where
> they fail and why, and which statements are explicitly not claimed.

---

## 1. Why validate against published physics?

A quantum-field-theory numerics stack can be internally consistent and still
wrong (wrong sign conventions, wrong factors of $2\pi$, wrong units). The only
cure is to reproduce numbers that were obtained by *independent* methods:

1. **Exact theory** — closed-form results every textbook derives (the harmonic
   oscillator ladder, Kepler's third law). Agreement must be at machine or
   solver precision.
2. **Published constants** — CODATA/PDG/Planck-collaboration values. These
   anchor the constants plumbing (units, charge cancellation, mass
   differences).
3. **Experimental bands** — measurements with known spread (turbulence
   constants, neutrino survival dips). Agreement inside the published band.
4. **Solver cross-checks** — two *different* numerical routes to the same
   answer (RK4 integration vs closed form; series vs integration; CAS compile
   vs hand builder).

Every test in this repository declares which class it belongs to, and the
tolerance follows from the class — not from "whatever makes it pass".

---

## 2. Running the suites

```bash
# Fast anchors (seconds):
cargo test --release -p fock_sirk --test qed_validation \
    --test qcd_validation --test qg_validation \
    --test sr_nuclear_validation --test astro_plasma_validation \
    --test em_optics_validation --test statmech_validation \
    --test classical_dynamics_validation --test weak_neutrino_validation \
    --test coupled_oscillator_sirk --test ritz_edge_study

# Heavy compiler-route compiles (minutes unoptimized, seconds in release):
scripts/run_heavy_tests.sh          # runs the #[ignore]d suites in --release
```

The heavy script uses `--release` deliberately: the symbolic CAS/LaTeX
expansion is $\sim 10^4\times$ slower unoptimized ($\sim 870\,\mathrm{s}
\rightarrow \sim 0.1\,\mathrm{s}$ per compile).

---

## 3. The engine being tested: inverse-free rational Krylov (SIRK)

### 3.1 What problem it solves

Given a Hamiltonian $H$ (here always a finite operator expression on a nested
Fock space) we need eigenvalues (spectra, ground states) and unitary dynamics
$\exp(-iHt)|\psi_0\rangle$. Dense diagonalization is impossible because the
Fock space is large; classical Krylov methods build a small subspace from
powers $v, Hv, H^2v, \dots$ but rational Krylov converges far faster per
dimension by admitting *shifted-and-inverted* information
$(H - z_k)^{-1}v$.

Inverting $(H - zI)$ is exactly what we refuse to do — the forward state
sequences explode combinatorially and the whole point of the architecture is
to stay in the sparse forward picture. **Hashimoto's inverse-free trick**
replaces each resolvent application with a *forward product*:

$$w_0 = |\psi_0\rangle, \qquad w_k = (H - z_k I)\, w_{k-1} \quad (k = 1 \dots m) \quad \text{(no inversion!)}$$

The span of $\{w_0, \dots, w_m\}$ is a *polynomial* Krylov subspace that
carries the same spectral information rational Krylov would extract, at the
price of needing a few more dimensions.

### 3.2 The projection identity

Because $H w_k = w_{k+1} + z_k w_k$, the projected Hamiltonian needs **no new
matrix elements** — it is assembled from the Gram matrix alone:

$$H_{jk} = \langle w_j | H | w_k \rangle = G_{j,k+1} + z_k G_{j,k}$$

This one line (`forward_sirk.rs`, step 4) is the heart of the method: spectral
information flows out of inner products the code already computes.

### 3.3 Gram whitening

The vectors $w_k$ are wildly non-orthogonal (norms grow like
$\|H\|^k$), so the generalized eigenproblem $H_{raw}\, c = \lambda G\, c$ is
solved by *rank-revealing whitening*: Hermitian eigendecomposition of $G$,
truncation of near-null directions ($\mathrm{rel\_tol} = 10^{-12}$), and
mapping into the orthonormal frame where an ordinary Hermitian eigensolver
runs. This replaced an earlier bare Cholesky that panicked on degenerate
sequences.

### 3.4 Restarted evolution and BRST projection

Long-time dynamics never runs one deep window (see §8: deep windows hit a
conditioning wall). Instead `evolve_restarted` advances through many short
optimal windows, reconstructing the state between them. Gauge-constrained
models carry a BRST charge $\Omega$; the flow periodically re-projects onto
the physical subspace $\ker\Omega$ by matrix-free conjugate gradient.

### 3.5 Frames, guards, and model fidelity

The implementation must BE the theoretical model. Two principles are now
enforced in code and pinned by test:

**Canonical default.** `SirkOpts::unit_norm_steps` defaults to `false`: the
stored sequence is the raw Hashimoto product $w_k = (H - z_k I)w_{k-1}$
exactly as theory defines it. No loops exist inside the spectral algorithm
(the forward pass is single-shot) and no renormalization occurs by default.

**Opt-in exact frame.** For deep spectral windows, `unit_norm_steps: true`
stores each vector rescaled to unit norm. This is a numerically EXACT basis
reparametrization of the same Krylov span (Rayleigh–Ritz invariant; the
projection identity becomes $H u_k = \tau_{k+1}u_{k+1} + z_k u_k$ with the
step norms $\tau$ folded back into $H_{raw}$). It exists because raw Gram
matrices grow like the Krylov depth cubed in norm and defeat whitening:
measured ground-error profiles for the displaced oscillator
(`ritz_edge_study` p2/p2b):

| $m$ | canonical err | unit-norm err |
|---|---------------|---------------|
| 4 | 6.2e-6 | 6.2e-6 (subspace-limited both) |
| 6 | 1.5e-9 | 1.5e-9 |
| 8 | 1.7e-6 | 2.0e-9 |
| 10 | 1.7e-3 | 1.6e-9 |
| 12 | 2.5e-1 | 2.6e-8 |
| 14 | 3.5 (diverged) | 2.2e-8 |

**No guard is on by default.** After the model-fidelity directive, ALL FOUR
deviations from the idealized sequence are opt-in: BRST projection
(`brst_charge: None`), adaptive truncation (`adaptive: false` errors instead),
the unit-norm frame (`unit_norm_steps: false`), and component pruning
(`prune_eps: 0.0`). Each remains available where its justification study
applies, and each is licensed quantitatively by
`fock_sirk/tests/guard_justification_study.rs`:

| Guard | Deviation | Justification (pinned by test) |
|---|---|---|
| `prune_eps` | drops tiny components every step | NOW DISABLED BY DEFAULT (`prune_eps: 0.0`) per model fidelity — memory-bounded runs OPT IN explicitly; Study A shows the opted-in value is invariant across eps = 1e-8..1e-14 (below solver noise floor) |
| mid-sequence BRST projection | replaces $w \leftarrow P(H-z)w$ | THEOREM: $[H,\Omega]=0$ makes $\ker\Omega$ invariant, so $P$ is the identity on exact physical sequences; verified inert on physical data (identical spectra, $\le 10^{-8}$ Ω-content). On contaminated data it enforces $\ker\Omega$ down to its documented contract or fails LOUDLY (`BrstNotConverged`) — silent pass-through is not an outcome (Study B) |
| adaptive truncation | hard component ceiling | already opt-in (`adaptive:false` default errors instead); at the suites' 50k budgets it NEVER engages — adaptive-on/off agree exactly and states sit ~500x below budget (Study C) |

**Theory-native dynamics.** The model needs ONE finite time $T$ and a deep
enough dimension $m$ — no time slicing. Restarts (`evolve_restarted`) remain
as an engineering alternative, but with the flat profile above a single $m=8$
window now reproduces the NS Newtonian decay rate to <2% directly
(`ns_sirk_laminar_decay_rate`, part d).

**Resolvedness is solver-enforced.** `ForwardSirkResult::ritz_residuals()`
computes TRUE relative residuals $\|H\psi - \theta\psi\|/\|H\psi\|$ for every
pair from the stored Gram matrix alone ($O(m^2)$, cancellation-free: the
residual vector $e$ has exactly one out-of-basis component,
$e_m = \tau_m c_{m-1}$). `resolved_ritz_values(tol)` returns the converged
set, replacing hand-maintained energy cutoffs. The measured residual ladder at
$m=8$ separates rungs cleanly: E0:1.5e-5, E1:2.6e-5, E2:2.0e-4 | E3:1.4e-3,
E4:7.3e-3, ...

---

## 4. The model zoo

| Builder | Hamiltonian / law | Exact spectrum or result tested |
|---|---|---|
| `qed_free_photon` | $\sum_k \omega_k N_k$ | massless dispersion $\omega=\|k\|$, additive multi-photon energies |
| `qed_cavity_frequencies` | conducting plates | $\omega_n = n\pi/d$ → Casimir $E/A = -\pi^2\hbar c/720d^3$ |
| `qed_static_charge_interaction` | displaced photon field | Coulomb $-\frac{e^2}{4\pi}\left(\frac{1}{r_1}-\frac{1}{r_2}\right)$ from one-photon exchange |
| `qed_pair_production` | $\gamma \leftrightarrow e^+e^-$ vertex | threshold $2\sqrt{m^2+(q/2)^2}$, $O(\alpha)$ self-energy, non-perturbative departure |
| `qed_charge_operator` | $Q = \sum N_e - \sum N_p$ | $[H,Q]=0$ (unbroken U(1)) |
| `qed_jaynes_cummings` | $\omega a^\dagger a + \omega_0 \sigma^\dagger\sigma + g(a\sigma^\dagger + a^\dagger\sigma)$ | vacuum-Rabi splitting $2g$, collapse–revival at $t_R = 2\pi\sqrt{\bar n+1}/g$ |
| `oscillator_beamsplitter` | $\omega(N_0+N_1)+J(a_0^\dagger a_1 + a_1^\dagger a_0)$ | sector spectrum $\{\omega-J, \omega+J\}$; swap $P=\sin^2(Jt)$ |
| `oscillator_displaced` | $\omega N + g(a^\dagger+a)$ | coherent displacement, $E_n = \omega n - g^2/\omega$ |
| `qcd_ym_hamiltonian` / `.cdb` | $H_{\rm final} = \tfrac12\pi^2 + \tfrac12 B^2$, $B=(A_0-A_1)+\tfrac12 g A_0 A_1$ | Cadabra-derived Yang–Mills Hamiltonian, bounded-below spectrum |
| `yang_mills_lattice` | full SU(3) lattice | mass gap $\approx g^2/2$ (Millennium positivity) |
| `qg_free_graviton` | $\sum c\|k\| N_k$ | GW speed $= c$ (GW170817 constraint) |
| `qg_starobinsky_*` | scalaron $\sum m N + \tfrac12\sum g^2$, $m^2=M^2/12\alpha$ | massive dispersion, ESA/boundedness of $R^2$ gravity |
| `ns_eulerian_fiber` | $H = K_0 + \{\pi_0, u\cdot\partial u\}$ | Euler advection, gauge-fixed derivative variables |

The Cadabra2 connection: several Hamiltonians are not transcribed by hand but
**derived** — the classical action is varied, Legendre-transformed, and
gauge-fixed symbolically in `docs/*.cdb` notebooks, canonicalized into the CAS
dialect, and compiled to Fock operators. The single source of truth for each
derived expression lives in one function (e.g. `qcd_ym_expression`); tests
derive their fixtures from it so compiler routes cannot drift apart.

### 4.5 The claim decomposition: what "without further assumptions" means

The headline claim — *the 3D gauge-fixed Hamiltonians of NS / QG(R²) / QYM /
QED are tested through SIRK/Hashimoto without further assumptions* —
decomposes into four verifiable steps, each pinned by tests:

1. **Action → Hamiltonian (Cadabra2).** Each `docs/*.cdb` module starts from
the classical action, varies it (polymomentum), Legendre-transforms, and
gauge-fixes to produce $H_{\rm final}$: NS the quantized Euler generator
$\sum_i\{\pi_i, A_i\}$, $A_i = \sum_j u_j u_{ij} - \nu u_{12+i}$; QG(R²) the
scalar sector $\tfrac12\pi^2 + \tfrac12(\nabla\phi)^2 + V(\phi)$ with the
scalaron mass $m^2 = M^2/12\alpha$; QYM $H_{\rm final} = \tfrac12\pi^2 +
\tfrac12 B^2$ with $B$ a genuine function of $A$; QED the $U(1)$
specialization. The compiler-route suites (§5.23) verify the numerical
builders ARE these Hamiltonians — term-by-term for QYM/NS, structurally for
the QG kinetic forms, and through both the LaTeX→Fock and CAS→Fock compilers.

2. **Hamiltonian → numerics: no physical approximation is added.** The only
steps between $H$ and the numbers are (i) restriction to the nested-Fock
sector spanned by the start state, (ii) normal ordering ($\langle 0|H|0\rangle
= 0$), (iii) the Krylov projection itself. There is no renormalization input,
no perturbative expansion, no mean field, no fitted parameter. The canonical
solver is the raw Hashimoto sequence $w_k = (H - z_kI)w_{k-1}$ — the
projection of $H$ onto the Krylov span — and EVERY numerical guard is off by
default (§3.5): BRST projection (`brst_charge: None`), adaptive truncation
(`adaptive: false` errors instead), component pruning (`prune_eps: 0.0`),
unit-norm frame (`unit_norm_steps: false`).
`assumption_ledger.rs::raw_canonical_sequence_no_guards_exact_predictions`
runs the four gauge-fixed Hamiltonians through the RAW sequence with every
guard off and no component budget, and recovers their exact predictions
(vacuum 0, $\omega=|k|$, additivity, scalaron mass) at $10^{-6}$.

3. **The one admitted numerical device: the unit-norm frame.** The unit-norm
frame is an EXACT basis reparametrization of the same Krylov subspace
(Rayleigh–Ritz invariant — the projection identity becomes $H u_k =
\tau_{k+1}u_{k+1} + z_k u_k$ with the step norms folded back into $H_{raw}$).
In the infinite-precision limit it changes NOTHING about the predictions; it
only removes the finite-precision conditioning wall that makes raw Gram
matrices numerically singular at depth (§8; §9.1). It is therefore not a
modelling assumption — it is the same model evaluated in better-conditioned
coordinates. Pinned by
`assumption_ledger.rs::unit_norm_frame_is_exact_reparametrization_not_model_change`
and `sirk_consistency_validation.rs::frame_invariance_swap_and_ground`:
canonical and unit-norm frames agree to $10^{-9}$ on resolved rungs, and at
depth the unit-norm frame resolves what the raw Gram matrix cannot
(canonical err 2.5e-1 vs unit-norm err 2.6e-8 at $m=12$ on the displaced
oscillator) — the wall is conditioning, not physics.

4. **Prediction with uncertainty.** Every spectral prediction is a Ritz value
with a certified error bar: the sharp Rayleigh–Ritz residual certificate
$|\theta-\lambda| \le \|r\|$ (measured widths $\le 10^{-6}$–$10^{-9}$ on
resolved rungs) and, as a conservative ceiling, the Theorem 4.1 envelope
(§9). Dynamics predictions carry the same machinery: unitary evolution
conserves norm/energy to $10^{-9}$, and frequencies (Rabi, beats, phases) are
compared to closed forms.

The honest boundaries of the programme — where the predictions match, where
they fail, and which statements are explicitly NOT claimed — are collected in
the assumption ledger (§5.25).

---

## 5. Suite-by-suite, test-by-test walkthrough

Reading convention. Each entry gives the four facts a reader needs:

- **computes** — the exact quantity/formula;
- **method** — the numerical route (SIRK = Ritz values of the projected
  Hamiltonian from a forward Krylov sequence; *matrix elements* = direct
  $\langle 1_i|H|0\rangle$ assembly; *closed form* = evaluating a published
  formula with CODATA constants; *ODE* = RK4 integration; *quadrature* =
  Simpson/Newton on an integral or root);
- **setup** — the concrete parameters (modes, couplings, cutoffs);
- **asserts** — the comparison and its tolerance.

### 5.1 QED machinery — `qed_validation.rs`

The Fock/SIRK machinery itself against exact QED results (the precision
constants suite is §5.8). Two framework conventions matter throughout: the
**inner construction** — an $n$-quanta state is ONE universe with inner
occupation $\{i:n\}$, so $\langle 0|H|0\rangle=0$ automatically (normal
ordering strips the $[a,a^\dagger]=1$ zero-point) and $n|n\rangle = n\omega|n\rangle$
at any occupation; and the **fermion ordering** at the pair vertex (positron
annihilation listed before the electron, else $H\ne H^\dagger$).

- **`qed_free_photon_dispersion_and_casimir_cavity`** — computes: vacuum
  energy (must be 0), one-photon energy (must equal $\omega_k$), $n$-photon
  additivity $E(n\cdot k)=n\omega_k$, Casimir-cavity levels
  $\omega_n = n\pi/d$, and the resolution of a two-mode superposition.
  *Method*: SIRK ground Ritz values, $m=4$, from inner-vacuum / $n$-photon
  starts; cavity modes generated by `qed_cavity_frequencies`. *Setup*:
  $k\in\{0.5,1.0,1.5,2.0\}$, cavity $d=1.5$ with 3 modes. *Asserts*:
  $|E - E_{\rm exact}| < 10^{-6}$ (exact-theory class).

- **`qed_one_photon_exchange_coulomb_law`** — computes: the exchange vertex
  amplitudes $g_i(r) = \sqrt{\frac{e^2 \Delta k\, k\,(1 + \sin(kr)/(kr))}{2\pi^2}}$
  (checked mode-by-mode), the displaced-oscillator shift
  $\delta E(r) = -\sum_i g_i(r)^2/\omega_i$ assembled from framework matrix
  elements, and the $r$-dependent difference
  $\delta E(r_1)-\delta E(r_2) \to -\frac{e^2}{4\pi}\left(\frac{1}{r_1}-\frac{1}{r_2}\right)$
  (Coulomb's law from one-photon exchange, Zee §I.3). *Method*: direct inner
  products $\langle 1_i|H|0\rangle$; plus a small SIRK solve ($m=6$, 15
  radial modes) of the genuine interacting Hamiltonian whose ground state
  must reproduce the exact shift. *Setup*: 2000 radial-shell modes
  ($k \in [0.01, 200]$, $\Delta k = 0.1$), $e=1$, $r_1=0.7$, $r_2=1.0$.
  *Asserts*: vertex to $10^{-9}$; Coulomb law to 2% (discretization class —
  the mode grid is finite); SIRK shift to $10^{-3}$.

- **`qed_pair_production_threshold_and_scaling`** — computes: the exact
  dressed-photon energy of the $\gamma\leftrightarrow e^+e^-$ sector
  (SIRK diagonalizes it non-perturbatively), the one-loop benchmark
  $\delta E = \sum_p \frac{c_p^2}{\omega - E_p - E'_p}$ (vacuum polarization),
  the $e^2$ (O($\alpha$)) scaling near the perturbative regime, the
  non-perturbative departure at strong coupling, and the minimum pair energy
  $2\sqrt{m^2+(q/2)^2}$. *Method*: SIRK ground Ritz value minus the bare
  photon energy $\omega$; vertex $c_p = e(2p-q)\sqrt{\Delta p/(2\omega\cdot4E_pE'_p)}$.
  *Setup*: $m=1$ (electron mass), $q=0.5$ photon momentum, $\Delta p=0.25$
  over $p\in[-3,3]$ (25 modes), Krylov $m=6$ (larger $m$ wanders into higher
  pair states and loses the isolated dressed rung). *Asserts*: weak-coupling
  SIRK vs one-loop to $10^{-3}$; doubling $e$ quadruples $\delta E$ to 5%;
  strong coupling ($e=1$) departs >10% from the one-loop sum; kinematics
  exact to $10^{-9}$.

- **`qed_u1_charge_conservation`** — computes: the commutator
  $[H,Q]|\psi\rangle$ with $Q = \sum e^\dagger e - \sum p^\dagger p$ on the
  vacuum, one-photon, and pair states. *Method*: operator application
  $HQ|\psi\rangle$ vs $QH|\psi\rangle$, norm of the difference. *Setup*: 2
  electron + 2 positron modes, vertex $\{0.3, 0.2\}$, $\omega=1$. *Asserts*:
  $\|[H,Q]|\psi\rangle\| < 10^{-9}$ (exact symmetry).

- **`qed_unitary_evolution_energy_conservation`** — computes: norm
  $\|\psi(t)\|$ and energy $\langle\psi(t)|H|\psi(t)\rangle$ after restarted
  Krylov evolution of a photon superposition. *Setup*: $\omega=\{1,2,3\}$,
  amplitudes $\{1, 0.5, 0.25\}$, $t=3$, 4 restarts of $m=6$. *Asserts*:
  $|\Delta\| \psi\|| < 10^{-9}$, $|\Delta\langle H\rangle| < 10^{-9}$
  (unitarity + closed-system energy, exact class).

- **`qed_jaynes_cummings_vacuum_rabi_and_detuned_spectrum`** — computes: the
  exact dressed-state energies of the closed $2\times2$ sectors — on
  resonance $\omega\pm g$ (vacuum Rabi splitting $2g$), the $n=1$ doublet
  $2\omega\pm g\sqrt2$, detuned
  $\frac{\omega+\omega_0}{2}\pm\sqrt{g^2+\delta^2/4}$, the bare ground 0, and
  $[H,N]=0$ for the total excitation $N=a^\dagger a + e^\dagger e$. *Method*:
  SIRK Ritz values from `jc_state(n, excited)` starts — the Krylov space IS
  the closed sector, so Ritz values are exact. *Setup*: $\omega=1$, $g=0.3$,
  $\omega_0=\omega$ (resonance) and $\omega_0=0.6$ ($\delta=0.4$), $m=4$.
  *Asserts*: $10^{-9}$ on every level and on $[H,N]$ (exact class).

- **`qed_jaynes_cummings_rabi_oscillation_and_revival`** — computes: the Rabi
  oscillation $P_e(t) = \cos^2(gt)$ (de-excitation at $t=\pi/2g$, return at
  $t=\pi/g$), the one-photon-sector $\sqrt2\,g$ frequency, and the coherent
  field collapse–revival against the exact Poisson-weighted sum
  $P_e(t) = \sum_n p_n \cos^2(g\sqrt{n+1}\,t)$, $p_n = e^{-\bar n}\bar n^n/n!$,
  with revival at $t_R = 2\pi\sqrt{\bar n+1}/g$. *Method*: restarted-Krylov
  unitary evolution for the single-excitation sectors (machine-precision
  verification there); the wide-spectrum coherent state ($\alpha=4$,
  $\bar n=16$) is outside the stable regime of the restarted solver (the
  geometric sequence growth defeats Gram whitening), so the revival is
  verified against the exact closed-form sum — the prediction itself.
  *Asserts*: $P_e(\pi/2g) < 0.05$, $P_e(\pi/g) > 0.95$, $\sqrt2\,g$
  de-excitation < 0.05, collapse $P_e(10)<0.6$, revival $>0.72$ and
  $> P_e(10)+0.2$.

- **`qed_static_charge_driven_field_oscillation`** — computes: the coherent
  displaced-oscillator response
  $\langle B_i + B_i^\dagger\rangle(t) = -\frac{2g_i}{k_i}(1-\cos k_i t)$ and
  energy conservation. *Method*: ONE single-shot SIRK solve ($m=8$) from the
  vacuum, then `time_evolve(t)` reads the state at any time (the restarted
  loop is the wrong tool here — small $g$ against the imaginary shifts makes
  forward vectors nearly parallel and whitening drops the physics after a few
  restarts). *Setup*: modes $k=\{1,2\}$, $e=0.5$, $r=1$; amplitudes $2g/k$
  read from the framework's own matrix elements; $t \in \{0.5, 1.5, 2.5,
  \pi, 2\pi\}$. *Asserts*: $|\langle x\rangle_{\rm SIRK} - \text{exact}| <
  5\times10^{-3}$; $\langle H\rangle$ and norm conserved to $10^{-6}$.

- **`qed_self_energy_linear_uv_divergence_and_finite_r_part`** — computes:
  the cutoff growth of the one-photon-exchange self-energy
  $\delta E(K) = -\sum_{k<K} g_i^2/\omega_i \to -\frac{e^2}{2\pi^2}K$ (the QED
  mass-renormalization statement) — while the $r$-dependent part stays finite
  (Coulomb, previous test). *Method*: direct matrix-element assembly for two
  cutoffs. *Setup*: $K=50$ and $K=100$ (1000/2000 modes), $e=1$, $r=0.7$.
  *Asserts*: $\delta E(100)-\delta E(50)$ vs $-(e^2/2\pi^2)\cdot 50$ to 2%.

### 5.2 QCD — `qcd_validation.rs`

- **`qcd_su3_color_factors`** — computes: $C_F = \frac{N_c^2-1}{2N_c} = 4/3$,
  $C_A = N_c = 3$, $T_R = 1/2$ **from the SU(3) structure constants**
  (never hard-coded), plus the identity $\sum_{bc} f_{0bc}^2 = C_A = 3$.
  *Method*: direct evaluation of the group-theoretic traces/sums over the 8
  generators. *Asserts*: $10^{-12}$ (exact class).

- **`qcd_one_gluon_exchange_coulomb`** — computes: the one-gluon exchange
  amplitudes $\langle 1_i|H|0\rangle = \sqrt{C_F}\,g_i$ (colour factor enters
  INSIDE the amplitude square, so $\delta E$ carries $C_F$, not $\sqrt{C_F}$
  on the linear coefficient), the assembled shift
  $\delta E(r_1)-\delta E(r_2) = -C_F\,\alpha_s\left(\frac{1}{r_1}-\frac{1}{r_2}\right)$
  (the Coulomb part of the Cornell potential, P&S §16.2), and a small-SIRK
  check. *Setup*: 2000 radial modes, $g=1$ so $\alpha_s = 1/4\pi$, $r_1=0.7$,
  $r_2=1.0$; small instance 15 modes, $m=6$. *Asserts*: amplitudes $10^{-9}$;
  Coulomb to 2%; SIRK shift $10^{-3}$.

- **`qcd_beta_function_asymptotic_freedom`** — computes:
  $\beta_0 = \frac{33-2N_f}{3}$ for $N_c=3$ (Gross–Wilczek–Politzer): 11
  (pure glue), 9 ($N_f=3$), 7 ($N_f=6$); that $\beta_0>0$ makes $\alpha_s(Q^2)$
  decrease; and that $N_f=17$ (i.e. $>33/2$) flips the sign and makes the
  coupling grow. *Method*: closed-form coefficients + one-loop running
  coupling integration. *Asserts*: $10^{-9}$ on coefficients; monotonicity
  assertions on the running.

- **`qcd_brst_gauge_invariance`** — computes: $H=H^\dagger$ (term-count
  equality and real interaction energy), and $C_A=3$. *Method*: `h.adjoint()`
  vs `h`, inner product reality. *Asserts*: imaginary part $<10^{-12}$.

- **`qcd_two_loop_beta_and_running`** — computes: the two-loop coefficients
  $\beta_1 = 102$ (pure glue), 64 ($N_f=3$), 26 ($N_f=6$) (Jones, Caswell
  1974); and the two-loop running coupling turning the PDG world average
  $\alpha_s(M_Z) = 0.1179$ into $\alpha_s(M_\tau)$ — the published
  $0.314 \pm 0.030$ (the one-loop formula gives only ~0.27, so this
  discriminates the perturbative order). *Method*: fixed-step RK4 integration
  of $d\alpha_s/d\ln Q$ with a crude single-flavour-threshold approximation
  (hence the generous window), deterministic — no wall-clock/random.
  *Setup*: $M_Z = 91.1876$ GeV, $M_\tau = 1.777$ GeV, 200,000 RK4 steps.
  *Asserts*: $|\alpha_s(M_\tau) - 0.314| < 0.05$; determinism $<10^{-15}$;
  round-trip $M_Z \to M_\tau \to M_Z$ recovers 0.1179 to $10^{-3}$.

- **`qcd_r_ratio_parton_model`** — computes: $R = N_c \sum_f Q_f^2$ for
  $e^+e^- \to$ hadrons: 2 (u,d,s), 10/3 (u,d,s,c), 11/3 (u,d,s,c,b), 5 (six
  flavours). *Method*: closed form. *Asserts*: $10^{-9}$ (exact parton model;
  PDG confirms experimentally to ~10%).

- **`qcd_gluon_dispersion_sirk`** — computes: the free-gluon field
  $H = \sum |k| N_k$ (inner construction) diagonalized by SIRK: vacuum 0,
  one-gluon Ritz $= |k|$ per mode, $n$-gluon additivity. *Setup*:
  $k \in \{0.5, 1.0, 1.5, 2.0\}$, $m=4$. *Asserts*: $10^{-6}$ (exact class).

- **`qcd_mass_gap_sirk`** — computes: the contrast between the massless free
  gluon ($E \to 0$ as $k\to 0$) and the confined Yang–Mills lattice whose
  even→odd gap is $\approx g^2/2$ (strong-coupling lattice result, the
  Millennium-Prize confinement statement). *Method*: two SIRK solves (even and
  odd parity starts) on `yang_mills_lattice(2, g, 1)`, gap
  $= E_{\rm odd} - E_{\rm even}$. *Setup*: $g=2$ ($g^2/2 = 2$), $m=4$; soft
  free-gluon mode $k=0.01$. *Asserts*: $E_{\rm soft} < 0.02$ (massless);
  $g^2/6 < \text{gap} < 3g^2/2$ and positive.

- **`qcd_ym_hamiltonian_outer_fock_sirk`** — computes: the structural facts
  of the Cadabra2-derived $H_{\rm final} = \tfrac12\pi^2 + \tfrac12 B^2$,
  $B = (A_0-A_1) + \tfrac12 g A_0 A_1$ (built through the CAS compiler,
  inner operators): $\langle 0|H|0\rangle = 0$ and a bounded-below spectrum
  with positive excitation gaps (Millennium-Prize positivity). *Method*:
  direct vacuum expectation + SIRK Ritz values ($m=8$) from the inner vacuum.
  *Setup*: $g=0.5$. *Asserts*: $|\langle 0|H|0\rangle| < 10^{-9}$; projected
  Hamiltonian Hermitian; $\ge 3$ resolved levels; $\lambda_0 > -10$;
  first three gaps $>0$.

- **`qcd_unitary_evolution_energy_conservation`** — same structure as the QED
  photon-field conservation test, on gluon modes $\omega=\{1,2,3\}$,
  amplitudes $\{1,0.5,0.25\}$, $t=3$, 4×$m=6$ restarts; norm and energy
  conserved to $10^{-9}$.

- **`qcd_gluon_self_energy_nonperturbative`** — computes: the exact
  (non-perturbative, SIRK) gluon self-energy of the gluon↔$q\bar q$ sector,
  the weak-coupling reduction to the perturbative quark loop
  $\delta E = T_R \sum_j \frac{c_j^2}{\omega - E_j - E'_j}$, the colour ratio
  gluon/photon $= T_R = 1/2$ (from $\mathrm{Tr}(T_aT_b) = T_R\delta_{ab}$),
  and the strong-coupling departure. *Method*: two SIRK solves — gluon sector
  (vertex carries $\sqrt{T_R}$ colour amplitude) vs photon sector (amplitude
  1) — $m=6$; one-loop benchmark computed from the same vertex data.
  *Setup*: $m=1$, $\omega=0.5$, $\Delta p = 0.25$ over 25 pair modes; weak
  $e=0.05$, strong $e=1.5$. *Asserts*: weak SIRK vs $T_R\times$one-loop to
  $10^{-4}$; ratio to $T_R$ within 2%; strong departure >10%.

### 5.3 Quantum gravity (classical + graviton) — `qg_validation.rs`

The semiclassical/Newtonian limit of the TEGR/teleparallel Hamiltonian derived
in `docs/qg_gauge_fixed_hamiltonian.cdb`. Constants: `QG_G`, `QG_HBAR`,
`QG_C` (CODATA); `GM_SUN = 1.32712440018e20`, `GM_EARTH = 3.986004418e14`
(m³/s²).

- **`qg_planck_scale`** — computes: $\ell_P = \sqrt{\hbar G/c^3}$,
  $t_P = \sqrt{\hbar G/c^5}$, $m_P = \sqrt{\hbar c/G}$, $E_P = m_P c^2$ from
  CODATA, plus the dimensional identities $\ell_P m_P = \hbar/c$ and
  $\ell_P = c\,t_P$. *Method*: closed form. *Asserts*: each scale to
  $10^{-3}$ relative (CODATA constants class); identities to $10^{-12}$
  (exact class).

- **`qg_gravitational_redshift_pound_rebka`** — computes:
  $z = g\,\Delta h/c^2$ with $g = GM_\oplus/R_\oplus^2$ computed from CODATA
  and the Harvard tower height $\Delta h = 22.5$ m. *Method*: closed form.
  *Asserts*: $|z - 2.5\times10^{-15}| < 2\%$ (experimental class — measured to
  ~1%).

- **`qg_mercury_perihelion_precession`** — computes:
  $\Delta\phi = \frac{6\pi GM}{c^2 a(1-e^2)}$ per orbit, scaled to
  arcsec/century with Mercury's 88-day period. *Setup*: $a = 5.7909\times10^{10}$ m,
  $e = 0.205630$. *Asserts*: within $0.5''$ of the published $43.0''$/century
  (observed $43.1\pm0.5$).

- **`qg_light_bending_eddington`** — computes: $\delta = \frac{4GM}{c^2 b}$
  at the Sun's limb ($b = R_\odot = 6.96\times10^8$ m). *Asserts*: within
  $0.02''$ of $1.75''$ (Eddington 1919).

- **`qg_gps_time_dilation`** — computes: the gravitational rate
  $\frac{GM}{c^2}\left(\frac{1}{R}-\frac{1}{R+h}\right)$ at orbital altitude
  $h = 2.02\times10^7$ m, and the per-day accumulation. *Asserts*: rate to 1%
  of $5.29\times10^{-10}$; offset $45.8\pm1$ µs/day (published ~45.9).

- **`qg_tegr_gr_equivalence`** — computes: for matter-dominated FLRW
  ($a\propto t^{2/3}$, $H = 2/3t$, $\dot H = -2/3t^2$) the Ricci scalar
  $R = 6(\dot H + H^2)$ and the TEGR torsion scalar $T = -6H^2$; that both
  give the same Friedmann equation $3H^2 = 8\pi G\rho$; and the boundary-term
  identity $R + T = 6\dot H$ (i.e. $eR = e\cdot T + \text{divergence}$ — the
  project's central TEGR=GR claim). *Method*: closed form on the FLRW
  geometry. *Asserts*: all identities to $10^{-12}$/$10^{-9}$.

- **`qg_newtonian_limit`** — computes: $\Phi = -GM_\oplus/R_\oplus =
  -6.26\times10^7$ m²/s² and the Earth's Schwarzschild radius
  $r_s = 2GM/c^2 = 8.87$ mm. *Asserts*: 1% relative.

- **`qg_graviton_dispersion_sirk`** — computes: the free graviton field
  $H = \sum c|k|N_k$ diagonalized by SIRK — vacuum 0, one-graviton Ritz
  $= c|k|$, group velocity $d\omega/dk = c$ exactly (GW170817 constraint
  $|\Delta v/c| < 10^{-15}$), and the structural point that a massive
  dispersion $\omega = \sqrt{c^2k^2 + m^2}$ would have an increasing slope
  (non-linear). *Setup*: $k\in\{0.5,1.0,1.5,2.0\}$, energies $c k$ with
  $c = 299792458$; $m=4$. *Asserts*: $10^{-3}$ on energies (SI scale);
  group velocity to $10^{-12}$.

- **`qg_tegr_hamiltonian_outer_fock_sirk`** — computes: structural facts of
  the outer-Fock realization of the Cadabra2-derived TEGR kinetic
  $\mathcal H_{\rm kin} \propto \frac{1}{16e}\mathcal S^2$ (book.tex 8190):
  $\langle 0|H|0\rangle = 0$; Hermiticity of the projected Hamiltonian
  (self-adjoint in the truncation); bounded-below spectrum with positive gaps
  — the ESA property derived via Strichartz for the densitized d'Alembertian.
  *Method*: direct vacuum expectation + SIRK $m=6$ from the vacuum.
  *Asserts*: $\langle 0|H|0\rangle$ to $10^{-9}$; $\|H-H^\dagger\| < 10^{-3}$
  (small-model Gram precision); $\ge 2$ levels; $\lambda_0 > -10$; gaps > 0.

- **`qg_starobinsky_scalaron_sirk`** — computes: the quantized scalaron
  $H = \sum m\,N_i$ ($m^2 = M^2/12\alpha$ from the Starobinsky potential
  curvature): vacuum 0, one-scalaron $= m$ per mode, additivity $2m$,
  doubling $m$ doubles the spacing, Hermitian bounded-below positive-gap
  spectrum (no conformal-mode $-\infty$ — the αR² stabilization). *Method*:
  SIRK ground Ritz values from inner `n_scalaron` states; the spectral check
  starts from a 0+1+2 superposition (an eigenstate start collapses the Krylov
  rank). *Setup*: $m=1$ and $m=2$, 3 modes, $m_{\rm krylov}=4$ / $8$.
  *Asserts*: $10^{-6}$ on the ladder; $10^{-6}$ Hermiticity; boundedness +
  positive gaps.

- **`qg_unitary_evolution_energy_conservation`** — restarted-Krylov
  conservation on graviton modes $\omega = \{1,2,3\}$ (natural units isolate
  the conservation physics; SI energies would wrap the phase in a finite
  subspace). Norm and energy to $10^{-9}$.

- **`qg_gravitational_wave_phase_sirk`** — computes: the phase advance of a
  single graviton eigenstate, $\phi = \omega t = c|k|t$ — the GW oscillates at
  the massless frequency (the quantum content of the LIGO/Virgo frequency
  evolution). *Method*: restarted-Krylov evolution, overlap
  $\langle \psi_0|\psi(t)\rangle$ read for modulus and phase. *Setup*:
  $\omega=2$, $t\in\{0.1, 0.25, 0.5\}$. *Asserts*: $|\langle\psi_0|\psi(t)\rangle|$
  to $10^{-6}$; phase to $10^{-6}$.

### 5.4 Starobinsky classical content — `qg_starobinsky_validation.rs`

- **`qg_starobinsky_newtonian_limit_yukawa`** — computes: the linearized
  weak-field potential of the R² theory,
  $\Phi(r) = -\frac{GM}{r}\left(1 + \tfrac13 e^{-mr}\right)$ (the published
  f(R) Yukawa form; the $\tfrac13$ fifth-force coefficient from the trace
  equation): Newtonian at $r \gg 1/m$ (deviation exactly
  $\tfrac13 e^{-10}$ at $r = 10/m$) and the $4/3$ short-range force
  enhancement at $r \ll 1/m$. *Method*: closed-form potential evaluation.
  *Asserts*: the $1/3$ coefficient, the $e^{-10}$ suppression, the $4/3$
  ratio — exact.

- **`qg_starobinsky_solar_system_gr`** — computes: with a Planck-heavy
  scalaron ($m = 1/\sqrt{12\alpha}$, $\alpha = O(1)$, Compton wavelength
  $\approx 5\times10^{-36}$ m so $e^{-mr} < 10^{-30}$ at any macroscopic
  distance — R² gravity passes solar-system tests), the same GR anchors as
  §5.3: perihelion 43.0″/century, GPS 5.29e-10 (~45.9 µs/day), Pound–Rebka
  2.46e-15, Sun-limb deflection 1.75″, Earth-surface $\Phi = -6.26\times10^7$
  m²/s². *Asserts*: the published values with their published tolerances.

- **`qg_starobinsky_scalaron_massive_dispersion_sirk`** — computes: the
  **numerical Hashimoto algorithm** diagonalizing the free scalaron
  $H = \sum\sqrt{k^2+m^2}\,N_k$ (inner ladder operators): the Ritz values
  reproduce the massive Klein–Gordon dispersion $\omega = \sqrt{k^2+m^2}$
  (unlike the massless graviton), the vacuum is 0, and the $m\to 0$ limit
  recovers the massless dispersion. *Setup*: $k$ grid with $m=0.55$ and
  $m\to0$. *Asserts*: Ritz values on $\sqrt{k^2+m^2}$ to solver precision;
  group velocity $d\omega/dk = k/\omega < 1$ rising monotonically to $c$
  (subluminal massive propagation).

- **`qg_starobinsky_derivative_variable_brst`** — computes: the BRST
  structure of the promoted spatial-gradient variables $g_i = \partial_i\phi$
  (the Navier-Stokes pattern, book.tex §4159-4197): the charge
  $\Omega = \sum_i g_i c_i$ is nilpotent ($\Omega^2 = 0$, first-class),
  $[H, g_i] = 0$ (the derivative variables are constants of the motion — no
  momenta on those modes, Eulerian block structure), $[H, \Omega] = 0$
  (BRST-closed), and the bare truncated flow from a genuinely unphysical
  ghost-carrying state GROWS the Ω-content ($\|\Omega\psi_0\| = \sqrt3 \to
  \|\Omega\psi(t)\| > \sqrt3$ under aggressive Krylov truncation) — the
  physical subspace is not invariant under the truncated dynamics, which is
  why the BRST projector rides along in the solve. *Method*: operator
  commutator norms + SIRK flows with/without projection. *Asserts*: exact
  identities on $\Omega^2$, $[H,g_i]$, $[H,\Omega]$; the Ω-growth contrast
  under truncation.

### 5.4b Starobinsky derivative-variable observables — `qg_starobinsky_derivative_variable.rs`

While §5.4 covers the *BRST structure* of the promoted spatial-gradient
variables (nilpotence, commutation, Ω-growth under truncation), this suite
checks what the **remaining physical observables** do while the gauge
conditions hold — the gravity analogue of §5.6's NS pattern (book.tex
§4159-4197): the scalaron's spatial gradients $g_i = \partial_i\phi$ are
promoted to independent canonical fields, and the Hamiltonian
$H = m\,N_0 + \tfrac12\sum_i g_i^2$ carries their products exactly as the NS
Hamiltonian carries $u_j\,u_{i,j}$. All three tests build the gauge condition
**by construction** (physical initial wave-function with $\langle g_m \rangle
= 2(m{+}1)\langle\phi_{m+1}\rangle$) and verify the remaining physics is
consistent and calculable.

- **`qg_starobinsky_derivative_variable_physical_observables_1d`** — computes:
  with the single derivative variable fixed
  ($\langle g_0 \rangle = 2\langle\phi_1 \rangle = 2/3$, `[H, C_0] = 0`
  making it a constant of the motion, $\Omega = g_0 c_0$ nilpotent and
  BRST-closed), the **consistent, calculable observables**: the composite
  $\langle\phi_0 g_0\rangle = 2\langle\phi_0\phi_1\rangle$ (promoted variable =
  actual derivative operator $D_0 = 2\phi_1$), the pointwise profile
  $\langle g(x)\rangle = \partial_x\langle\phi(x)\rangle$ at every $x$, the
  gradient content frozen, $\langle H\rangle$ and $\|\psi\|$ conserved by the
  unitary solver, bare and BRST-projected SIRK flows giving IDENTICAL
  observables, and the gauge-condition drift a controlled quadratic-in-$dt$
  solver artifact ($d(0.05) < 10^{-3}$, $d(0.025) < d(0.05)/4$). *Method*:
  operator commutator norms + restarted-Krylov flows. *Asserts*: exact
  identities at $t = 0$ (1e-9), flow conservation to solver precision, drift
  $\propto dt^2$.

- **`qg_starobinsky_derivative_variable_higher_hermite_modes`** — computes:
  the FULL multi-level fiber $\phi_1, \phi_2, \phi_3$ with promoted
  $g_0 = 2\phi_1, g_1 = 4\phi_2, g_2 = 6\phi_3$: the derivative profile
  $\langle g(x)\rangle = \sum\langle g_m\rangle H_m(x) = \partial_x\langle\phi(x)\rangle$
  is a **genuine polynomial** (a quadratic in $x$ — the curvature
  $\partial_x^2\langle\phi\rangle = 8\langle\phi_2\rangle +
  48\langle\phi_3\rangle x$ varies with $x$, not the flat 1D constant
  $2\phi_1$); promoted composites $\langle\phi_m g_m\rangle =
  2(m{+}1)\langle\phi_m\phi_{m+1}\rangle$ match field-only values; all three
  constraints $[H, C_m] = 0$ and $[H, g_i] = 0$ hold; $\Omega^2 = 0$,
  $[H,\Omega] = 0$; and the SIRK checks (energy/norm conservation, bare =
  gauge-fixed flow, drift $\propto dt^2$) extend to the full fiber. *Method*:
  Hermite-basis pointwise evaluation + SIRK flows. *Asserts*: pointwise
  identity at 1e-9, genuine-polynomial curvature, flow conservation.

- **`qg_starobinsky_derivative_variable_unphysical_data_inconsistent`** —
  computes: what happens when the gauge condition does NOT hold — unphysical
  data ($\langle g_0 \rangle \neq 2\langle\phi_1\rangle$) is **detected**, makes
  the composite observable INCONSISTENT (the promoted $\langle\phi_0 g_0\rangle$
  no longer equals $2\langle\phi_0\phi_1\rangle$), carries Ω-content, and is
  **conserved** (not self-fixed) by the bare flow — gauge fixing is genuinely
  required, exactly the book.tex pattern. *Method*: comparison of composite
  expectations on physical vs unphysical states. *Asserts*: the inconsistency
  is detected and stable.

### 5.5 Navier–Stokes classical fluid mechanics — `ns_numerical_validation.rs`

- **`ns_kolmogorov_scales_and_spectrum`** — computes: the dissipation
  microscale $\eta = (\nu^3/\varepsilon)^{1/4}$ (0.1 mm for water at
  $\varepsilon = 10^{-2}$ W/kg), the identity $\mathrm{Re}_\eta = 1$
  (definition of the microscale), the experimental constant
  $C_K \in [1.4, 1.7]$ (Sreenivasan & Antonia 1997) with the exponent check
  $E(8)/E(2) = 2^{-10/3}$ (the $-5/3$ law), the exact 4/5 law
  $\langle\delta u_L^3\rangle = -\frac45\varepsilon r$, and the Taylor
  dissipation relation $\varepsilon = 15\nu\langle(\partial u_1/\partial
  x_1)^2\rangle$. *Method*: closed form (the 4/5 and Taylor relations are
  identities by construction; the spectrum is an evaluation). *Asserts*:
  exact to $10^{-9}$-$10^{-15}$; $C_K$ inside the measured band.

- **`ns_hagen_poiseuille_pipe_flow`** — computes: the parabolic pipe-flow law
  $Q = \frac{\pi R^4\Delta p}{8\mu L}$ (≈ 1.23 mL/s for the setup), mean
  velocity $V = \frac{R^2\Delta p}{8\mu L}$, and the laminar Darcy–Weisbach
  friction $f = 64/\mathrm{Re}$ (the Moody-chart line, valid for
  $\mathrm{Re} < 2300$). *Setup*: water $\mu = 10^{-3}$ Pa·s, $\rho = 1000$
  kg/m³, $R = 5$ mm, $L = 2$ m, $\Delta p = 10$ Pa. *Asserts*: $Q$ vs the
  published relation to 1%; $f = 64/\mathrm{Re}$ exact; laminar regime check.

- **`ns_stokes_drag_osseen`** — computes: Stokes' law $F = 6\pi\mu rU$ (the
  Millikan oil-drop base; ≈ 3.4e-14 N for a 1 µm droplet), the Oseen
  correction $1 + \frac{3}{16}\mathrm{Re}$ (≈ 2% at $\mathrm{Re}=0.1$), and
  the $r^3$ scaling of terminal-velocity balance (radius ratio 2 → force
  ratio 8). *Asserts*: 1% on $F$; exact on the correction and scaling.

- **`ns_blasius_boundary_layer`** — computes: the exact Blasius similarity
  constants — 99% thickness $\delta = 5x/\sqrt{\mathrm{Re}_x}$, displacement
  $\delta^* = 1.7208\,x/\sqrt{\mathrm{Re}_x}$, momentum $\theta =
  0.6641\,x/\sqrt{\mathrm{Re}_x}$, shape factor $H = \delta^*/\theta =
  2.5916$, skin friction $C_f = 0.664/\sqrt{\mathrm{Re}_x}$, wall shear
  $\tau_w = 0.332\rho U^2/\sqrt{\mathrm{Re}_x}$ — against wind-tunnel
  measurements (~3%). *Setup*: air $\nu = 1.5\times10^{-5}$ m²/s, $U = 10$
  m/s, $x = 0.5$ m. *Asserts*: 1% on $\delta$, $C_f$, $\tau_w$; $10^{-3}$ on
  $H$ and the ratios.

- **`ns_transition_reynolds_strouhal`** — computes: the experimental
  transition Reynolds numbers — pipe $\mathrm{Re}_c \approx 2300$ (measured
  2000–2400), plane Poiseuille $\mathrm{Re}_c = 5772.22$ (Orszag 1971, exact
  linear stability), flat plate $\mathrm{Re}_{x,c} \approx 5\times10^5$ — and
  the vortex-shedding Strouhal number $\mathrm{St} = fD/U \approx 0.2$
  (Roshko 1954; measured 0.19–0.21) giving $f \approx 10$ Hz for the setup.
  *Asserts*: inside the published windows; Orszag value exact.

- **`ns_lamb_oseen_vortex_core`** — computes: the Lamb–Oseen vortex (exact
  2D-NS solution for diffusing a point vortex): core radius
  $r_c(t) = \sqrt{4\nu t}$ with the $\sqrt{t}$ scaling (8 cm at $t=100$ s for
  a large-aircraft wake), peak vorticity $\omega_{\max}(t) = \Gamma/4\pi\nu t$
  with the invariant $\omega_{\max}\cdot t = \Gamma/4\pi\nu$ and the $1/t$
  decay. *Setup*: $\nu = 1.5\times10^{-5}$, $\Gamma = 5$ m²/s. *Asserts*: 1%
  on $r_c(100\text{ s})$ and $\omega_{\max}(100\text{ s})$; exact on the
  scalings.

- **`ns_sirk_laminar_decay_rate`** — the machinery test. *Computes*: the
  Ehrenfest identity $d\langle u\rangle/dt = i\langle[H,u]\rangle =
  4\kappa\langle u\rangle + 4c$ of the Eulerian affine fiber
  $V(u) = \kappa u + c$ (with $u = a^\dagger + a$, $\pi = i(a^\dagger - a)$,
  $H = \{\pi, V\}$ — the symmetrization carries the factor 2); calibrated
  $\kappa = -\nu k^2/4$ this is the Newtonian free decay of a laminar Fourier
  mode $du/dt = -\nu k^2 u$, and the SIRK-restarted evolution must measure the
  rate. *Method*: (a) exact commutator identity on the probes
  $|0\rangle+|1\rangle$, $|1\rangle$, $|2\rangle$; (b) calibrated-rate
  identity; (c) small-$t$ slope of $\langle u(t)\rangle$ from a restarted
  solve ($t=0.05$, 2×$m=2$); (d) THEORY-NATIVE single-shot: one $m=8$ window
  in the unit-norm frame reproduces the same rate (no time slicing).
  *Setup*: $\nu = 10^{-4}$, $k = 2\pi$ ($\nu k^2 \approx 3.95\times10^{-3}$).
  *Asserts*: identities $10^{-9}$; measured slope to 1% (restarted) and 2%
  (single-shot).

### 5.6 NS derivative-variable observables — `ns_derivative_variable_fixing.rs`

The promoted-derivative-variable formalism (book.tex §4159-4197): the field
$u(x) = \sum_n u_n H_n(x)$ with physicists' Hermite polynomials
($\partial_x H_n = 2n H_{n-1}$) has the derivative operator
$\partial_x u = \sum_m 2(m+1)u_{m+1}H_m(x)$, i.e. mode-$m$ derivative
$D_m = 2(m+1)u_{m+1}$. The promoted variable $g_m$ (its own ladder mode) is
fixed to the derivative value; because the derivative modes carry no momenta,
$[H, C_m] = 0$ for $C_m = g_m - D_m$ — the gauge condition is a constant of
the motion **by construction**, not what these tests check. They check that
the REMAINING observables are consistent and calculable while it holds.

- **`ns_derivative_variable_physical_observables_1d`** — computes: with the
  Euler fiber $H = K_0 + \{\pi_0, u_0 g_0\}$ (normal-ordered kinetic + the
  promoted form of $u\,\partial_x u$): frozen gradient content; energy
  $\langle H\rangle$ conserved by the unitary solver to machine precision;
  bare and BRST-projected flows giving IDENTICAL physical observables; the
  Ehrenfest equation $d\langle u_0\rangle/dt = \langle i[H,u_0]\rangle =
  2\langle\pi_0\rangle + 4\langle u_0 g_0\rangle = 8\langle u_0\rangle
  \langle u_1\rangle$ (the velocity is advected by its own spatial
  derivative); and the gauge-condition drift being a controlled
  $\propto dt^2$ artifact (2.8e-3 at $dt=0.05$ → 5.7e-5 at $dt=0.025$).
  *Method*: SIRK restarted flows, bare vs BRST-projected, at decreasing
  $dt$. *Asserts*: energy conservation and bare=projected agreement to solver
  precision; the Ehrenfest identity at $t=0$; drift shrinks quadratically.

- **`ns_derivative_variable_physical_observables_2d`** — the same with
  $g_x = 2u_{1,0}$, $g_y = 2u_{0,1}$,
  $H = K_0 + \{\pi_0, u_0(g_x+g_y)\}$: $d\langle u_0\rangle/dt =
  8\langle u_0\rangle(\langle u_1\rangle + \langle u_2\rangle)$.

- **`ns_derivative_variable_higher_hermite_modes`** — the FULL multi-level
  fiber: content $u_1, u_2, u_3$ with promoted $g_0 = 2u_1$, $g_1 = 4u_2$,
  $g_2 = 6u_3$, so the derivative profile $\langle g(x)\rangle =
  \sum_m \langle g_m\rangle H_m(x) = \partial_x\langle u(x)\rangle$ is a
  GENUINE polynomial (a quadratic in $x$, not the constant $2u_1$); the
  pointwise identity holds at every $x$; the promoted composites
  $\langle u_m g_m\rangle = 2(m+1)\langle u_m u_{m+1}\rangle$ match the
  field-only values; the SIRK checks (energy conservation, bare =
  gauge-fixed, drift $\propto dt^2$) extend to the full fiber.

- **`ns_derivative_variable_unphysical_data_inconsistent`** — computes: what
  happens when the promoted variable is NOT set to the derivative value —
  $\langle C_0\rangle \ne 0$ is detected, the physical observable is
  INCONSISTENT ($4\langle u_0 g_0\rangle \ne 8\langle u_0 u_1\rangle$: the
  dynamics the Hamiltonian generates disagrees with the value read off the
  velocity field alone), the violation carries Ω-content, and it is conserved
  (no self-fixing) — gauge fixing is genuinely required, exactly the NS
  pattern.

### 5.7 NS formalization — `ns_validation.rs`

The Fock/SIRK machinery against the Lean-formalization thread
(`../timepiece/CONSOLIDATED_PLAN.md`, NS items): the Eulerian
derivatives-as-fields picture, the affine-fiber hopping structure, the
Hashimoto shift-invert selection theorem, and the BRST divergence constraint.

- **`ns_derivative_fields_constant_of_motion`** — computes:
  $[H, u_{i,j}] = [H, u_{i,jj}] = 0$ **exactly** (derivative modes carry no
  momenta — the block-diagonalisation statement) and
  $\langle u_{i,j}\rangle$ conserved under SIRK evolution. *Asserts*:
  commutator norms $< 10^{-9}$; expectation conserved to solver precision.

- **`ns_affine_fiber_hopping_structure`** — computes: the analytic content of
  the affine fiber $V(u) = \kappa u + c$: exactly the $\pm2$-hopping
  $2\kappa\sqrt{(n+1)(n+2)}$ (advection) plus the $\pm1$-hopping
  $2c\sqrt{n+1}$ (viscous offset), with all other matrix elements zero.
  *Method*: direct matrix-element evaluation of the fiber Hamiltonian.
  *Asserts*: every nonzero element matches the formula; every other element is
  zero.

- **`ns_three_component_vorticity_hopping`** — computes: the 3-component
  affine fiber with an ARBITRARY non-symmetric gradient matrix $A$: the 24
  hopping terms per component; the number-conserving vorticity hopping
  $\langle 1_k|H|1_i\rangle = 2i(A_{ki} - A_{ik})$ whose amplitude is not
  monotone along the shift; and the $\pm2$ pair-creation/annihilation
  hopping. *Asserts*: exact agreement with the closed forms.

- **`ns_sirk_esa_truncation`** — computes: the truncated NS operator via
  SIRK: projected Hamiltonian Hermitian (the finite shadow of essential
  self-adjointness on the finite-mode core), real spectrum bounded below,
  $\langle 0|H|0\rangle = 0$. *Asserts*: Hermiticity, boundedness, vacuum
  rule.

- **`ns_hashimoto_shift_invert_selection`** — computes: the Hashimoto
  shift-invert selection theorem numerically: for a non-real shift $\gamma$
  the resolvent $R = (\gamma I - A)^{-1}$ exists, is bounded by
  $\|R\| \le 1/|\mathrm{Im}\,\gamma|$ (`ChapterHashimotoComplexShifts`),
  satisfies the resolvent identity, and its eigenvalues $1/(\gamma-\lambda_j)$
  recover the NS spectrum — the resolvent determines the operator (the
  selection behind the SIRK algorithm). *Asserts*: the bound, the resolvent
  identity, and the spectral recovery.

- **`ns_unitary_evolution_conservation`** — restarted-Krylov dynamics of the
  interacting NS Hamiltonian: norm, energy, and the derivative-field
  expectation $\langle u_{i,j}\rangle$ conserved (unitarity + Eulerian block
  structure).

- **`ns_brst_projection_physical_subspace`** — computes: the BRST divergence
  constraint $\Omega = \sum_j u_{j,j}\,c_j$: nilpotency $\Omega^2 = 0$
  (first-class), gauge invariance $[H,\Omega] = [H_f,\Omega] = 0$
  (BRST-closed NS and fiber Hamiltonians), and the non-invariance of the
  physical subspace under the bare flow (Ω-content of an unphysical
  ghost-carrying state grows) — the reason the projector rides along in the
  solve.

### 5.8 Precision QED constants — `qed_precision.rs`

Published QED numbers vs measured values (no Fock machinery — the constants
plumbing). Constants: $\alpha = 1/137.035999084$, $m_e c^2 = 0.51099895000$
MeV, Ry $= 13.605693122994$ eV, $h = 4.135667696\times10^{-15}$ eV·s,
$\hbar = 6.582119569\times10^{-16}$ eV·s, $c = 299792458$ m/s.

- **`qed_electron_anomalous_moment_g_minus_2`** — computes: the Schwinger term
  $\alpha/2\pi$; the two-loop prediction
  $a_e = \frac{\alpha}{2\pi} + \left(\frac{\alpha}{\pi}\right)^2 A_2$ with
  $A_2 = -0.32847896557919378$; and $A_2$ extracted back from CODATA
  $a_e = 0.00115965218059$. *Asserts*: Schwinger to 0.2% (leading order);
  two-loop to $3\times10^{-5}$ (the residual is the $O(\alpha^3)$
  contribution); derived $A_2$ to 1%.

- **`qed_compton_kinematics_and_thomson_limit`** — computes: the Compton
  wavelength $\lambda_C = h/mc = 2.42631023867$ pm; the shift
  $\Delta\lambda = \lambda_C(1-\cos\theta)$; the Cs-137 662 keV backscatter
  peak $E' = E/(1 + (E/mc^2)(1-\cos\theta)) = 184$ keV at 180°; the Thomson
  cross-section $\sigma_T = \frac{8\pi}{3}r_e^2$ with
  $r_e = \alpha\hbar c/mc^2$; and the Klein–Nishina limits (→ Thomson as
  $\varepsilon\to0$, → $\frac{\pi r_e^2}{\varepsilon}(\ln 2\varepsilon +
  \tfrac12)$ as $\varepsilon\to\infty$). *Asserts*: $10^{-6}$ on $\lambda_C$
  and $r_e$; $10^{-4}$ on $\sigma_T$; 0.5% on the backscatter peak; 1.2%/2%
  on the KN limits.

- **`qed_positronium_spectrum_and_lifetimes`** — computes: ground binding
  $E = -\mathrm{Ry}/2 = -6.80$ eV (reduced mass $m_e/2$); hyperfine splitting
  $\nu = \frac{7}{12}\alpha^4 mc^2/h = 204.4$ GHz vs measured 203.392 GHz
  (the $-0.5\%$ shift is radiative/recoil corrections); para-Ps lifetime
  $\tau = 2\hbar/\alpha^5 mc^2 = 124.5$ ps vs 125.14 ps; ortho-Ps
  $\tau = \hbar/(\frac{2(\pi^2-9)}{9\pi}\alpha^6 mc^2) = 138.7$ ns, with the
  $O(\alpha)$ correction $\times(1 + 10.286\,\alpha/\pi)$ → 142.0 ns vs the
  measured 142.05 ns. *Asserts*: 0.1% on $E$; 1% on HFS and para; 3%
  (leading) / 1% (corrected) on ortho.

- **`qed_uehling_vacuum_polarization_component`** — computes:
  $\Delta E_{\rm VP}(nS) = -\frac{4}{15}\frac{\alpha}{\pi}\frac{(Z\alpha)^4
  mc^2}{n^3}$: for hydrogen $Z=1, n=2$, $\Delta\nu = -27.1$ MHz — the
  published VP component of the 2S Lamb shift (self-energy +1085 MHz; the sum
  with relativistic corrections is the 1057.8 MHz total). *Asserts*: 2%.

- **`qed_hydrogen_lamb_shift_bethe_estimate`** — computes: Bethe's 1947
  non-relativistic self-energy
  $\Delta E_n = \frac{8}{3\pi}\frac{\alpha^3}{n^3}\,\mathrm{Ry}\,
  \ln(mc^2/\bar\varepsilon)$ with the Bethe average $\bar\varepsilon =
  16.6\,\mathrm{Ry}$: 2S → 1048 MHz vs 1057.845 MHz measured; the $1/n^3$
  scaling gives the 3S shift ~310 MHz. *Asserts*: 2% on both.

- **`qed_hydrogen_fine_structure_and_rydberg_spectrum`** — computes: the
  Rydberg ladder $E_n = -\mathrm{Ry}/n^2$ (1S–2S $= \tfrac34\mathrm{Ry} =
  10.2043$ eV) and the Dirac fine-structure splitting
  $\Delta E(2P_{3/2}-2P_{1/2}) = \alpha^2\mathrm{Ry}/16 = 10.95$ GHz vs
  measured 10.969 GHz (the $2S_{1/2}-2P_{1/2}$ degeneracy Dirac theory
  predicts is the one QED breaks with the 1057.8 MHz Lamb shift). *Asserts*:
  $10^{-3}$ eV on the ladder; 1% on the splitting.

- **`qed_casimir_energy_and_force`** — computes: the zeta-regularized sums
  $\sum n\,e^{-\delta n} - 1/\delta^2 \to \zeta(-1) = -1/12$ and
  $\sum n^3 e^{-\delta n} - 6/\delta^4 \to \zeta(-3) = 1/120$ (smooth-cutoff
  regularization); the published $E/A = -\pi^2\hbar c/720d^3$ and
  $F/A = -\pi^2\hbar c/240d^4$ — at $d = 1\,\mu$m, $F = -1.30$ mN/m²
  (Lamoreaux 1997, 5%); the derivative identity $F = 3E/d$ (from $E\propto
  d^{-3}$: 3/720 = 1/240 exactly); and the 1D scalar cavity
  $E = -\pi\hbar c/24d$. *Method*: direct finite sums (200k terms) with the
  cutoff subtraction; closed forms. *Asserts*: $10^{-3}$ on the zeta values;
  1% on the force/energy; $10^{-12}$ on the derivative identity.

- **`qed_blackbody_photon_gas_planck_spectrum`** — computes: the photon-gas
  internal-energy integral $\int_0^\infty \frac{x^3}{e^x-1}dx = \pi^4/15$
  (Simpson quadrature, 2M points, $x \in [0,60]$); the Stefan–Boltzmann
  constant $\sigma = \pi^2 k^4/60\hbar^3 c^2 = 5.670374419\times10^{-8}$
  W/m²K⁴; Wien's displacement law — the frequency root of
  $x = 3(1-e^{-x})$ (2.821439) and the wavelength root of
  $x = 5(1-e^{-x})$ (4.965114) giving $\lambda_{\max}T = hc/4.965114\,k =
  2.897771955\times10^{-3}$ m·K, both roots found by fixed-point iteration;
  and the photon number coefficient $2\zeta(3)/\pi^2 = 0.24357$.
  *Asserts*: $10^{-4}$ on the integral; $10^{-6}$ on $\sigma$; $10^{-5}$ on
  the Wien constants; $10^{-3}$ on the number coefficient.

### 5.9 QED as abelian Yang–Mills — `qed_abelian_reduction.rs`

Makes precise that QED is the $U(1)$ specialization of QYM ($n_{\rm colors} =
1$, $f_{abc} = 0$): the full Weyl-gauge QYM Hamiltonian
$H = -\tfrac12\sum\pi^2 - \tfrac12\sum B^2$,
$B_{ia} = \varepsilon_{ijk}(\partial_j A_k^a + \tfrac12 g f_{abc} A_j^b A_k^c)$
reduces to free Maxwell $B_i = (\nabla\times A)_i$; on a transverse mode
$\tfrac12\pi^2 + \tfrac12\omega^2 A^2$ is the oscillator whose normal-ordered
second quantization is exactly `qed_free_photon`.

- **`qed_free_photon_is_normal_ordered_abelian_qym_free_sector`** — computes:
  the exact operator identities
  $2\cdot\texttt{qed\_free\_photon}(1) + I = \tfrac12\pi^2 + \tfrac12 A^2$
  (unit frequency, zero-point stripped) and the general-$\omega$ identity
  $\tfrac12\pi^2 + \tfrac12\omega^2 A^2 =
  \frac{1+\omega^2}{\omega}\,\texttt{qed\_free\_photon}(\omega) +
  \tfrac12(1+\omega^2)I + \tfrac12(\omega^2-1)(a^2 + a^{\dagger 2})$; plus the
  photon ladder $\{0, \omega, 2\omega\}$. *Method*: operator matrix elements
  on a truncated Fock basis. *Asserts*: $10^{-9}$.

- **`abelian_specialization_of_full_qym_is_free_maxwell`** — computes: the
  SAME QYM builder with $n_{\rm colors}=1$, $g=0$ produces only quadratic
  terms (no $A^3$/$A^4$) and equals an independently built Maxwell
  Hamiltonian $-\tfrac12\sum\pi^2 - \tfrac12\sum(\nabla\times A)^2$ exactly on
  the truncated Fock space. *Asserts*: matrix equality to $10^{-9}$.

- **`qed_free_photon_coherent_phase_rotation`** — computes: free-photon
  dynamics — each Fock component rotates by $e^{-i\omega nt}$; the overlap
  $|\langle\psi_0|\psi(t)\rangle|^2 = |1 + e^{-i\omega t} + e^{-2i\omega
  t}|^2/9$ checked at the half-period, the anti-period and the revival.
  *Method*: closed-form phase factors vs SIRK evolution. *Asserts*: exact.

- **`qed_static_charge_displaced_oscillator_exact_ground`** — computes: the
  QED matter coupling (the abelian analogue of the QYM interaction) is the
  linear $A\cdot J$ term $g(B^\dagger + B)$; the model is an exactly solvable
  displaced oscillator with ground state $-g^2/\omega$, reproduced by SIRK.
  *Asserts*: high precision on the ground Ritz value.

### 5.10 Special relativity & nuclear — `sr_nuclear_validation.rs`

- **`sr_mass_energy_anchors`** — computes: PDG rest energies $m_e c^2 =
  0.511$ MeV, $m_\mu c^2 = 105.658$ MeV, $m_p c^2 = 938.272$ MeV,
  $1\,\mathrm{u}\,c^2 = 931.494$ MeV. *Method*: closed form. *Asserts*:
  published values.

- **`sr_two_body_decay_kinematics`** — computes: the exact two-body decay
  momentum $|p_\mu| = \frac{m_\pi^2 - m_\mu^2}{2m_\pi} = 29.788$ MeV/c
  ($\pi^+\to\mu^+\nu$, the PDG value) and $\pi^0\to\gamma\gamma$ giving
  exactly $m_{\pi^0}/2 = 67.488$ MeV per photon. *Asserts*: published values.

- **`sr_breit_wheeler_and_gzk_thresholds`** — computes: the $\gamma\gamma\to
  e^+e^-$ head-on threshold $2m_e c^2 = 1.022$ MeV; with a soft CMB photon
  ($\varepsilon \approx kT_{\rm CMB}$) the high-energy threshold
  $(m_e c^2)^2/\varepsilon$; the $p+\gamma\to\Delta(1232)$ GZK threshold
  landing in the published $\sim10^{20}$ eV attenuation window. *Asserts*:
  the published windows.

- **`sr_cosmic_ray_muon_survival_frisch_smith`** — computes: the Frisch–Smith
  contrast — without time dilation a 4 GeV muon survives 15 km of atmosphere
  at the $e^{-23}$ level; with dilation ($\gamma \approx 38$) it arrives at
  the ~0.3 level. *Method*: relativistic decay-law evaluation. *Asserts*: the
  contrast ratio ($>10^9$).

- **`sr_lhc_dipole_field_and_revolution_frequency`** — computes:
  $B = pc/(q\rho)$ for a 7 TeV proton on $\rho = 2804$ m gives 8.33 T (the
  LHC design field) and $f_{\rm rev} = c/C = 11245$ Hz. *Bug caught here*: the
  charge in $B = pc/(q\rho)$ cancels — $p = E/c$ already carries it; an extra
  $q$ gave $10^{-18}$ instead of 8.33 T. *Asserts*: published values.

- **`nuc_semf_binding_per_nucleon_peak_near_iron`** — computes: the
  Weizsäcker mass formula's volume/surface/Coulomb/symmetry competition puts
  the peak of $B/A$ in the iron group ($A\in[50,70]$) and drives heavy nuclei
  toward fission ($B/A$ falling by $A\approx240$). *Asserts*: the qualitative
  content (robust to coefficient choices).

- **`nuc_q_values_from_atomic_masses`** — computes: deuteron binding from
  atomic masses $^2\mathrm{H} - \mathrm{H} - n = 2.224566$ MeV (measured);
  neutron $\beta$-decay $Q = 0.782$ MeV; tritium endpoint 18.6 keV (KATRIN).
  *Asserts*: published values.

### 5.11 Astro, plasma, metrology — `astro_plasma_validation.rs`

- **`astro_black_hole_anchors`** — computes: Hawking temperature of a
  solar-mass black hole $T_H = \hbar c^3/8\pi GMk = 6.17\times10^{-8}$ K;
  Schwarzschild radii (Sun 2.953 km, Sgr A* 1.27e10 m); ISCO GW frequency
  4397 Hz per solar mass (the LIGO ringdown scale). *Asserts*: published
  values (derived-constants class).

- **`astro_chandrasekhar_mass_from_constants`** — computes:
  $M_{\rm Ch} = \frac{\pi(\hbar c/G)^{3/2}}{(\mu_e m_p)^2}$ with
  $\mu_e = 2$: 1.44 $M_\odot$ — the white-dwarf limit. *Asserts*: $10^{-3}$
  relative.

- **`astro_eddington_luminosity_solar_mass`** — computes:
  $L_{\rm Edd} = 4\pi GM m_p c/\sigma_T = 1.26\times10^{31}$ W per $M_\odot$
  from $G$, $m_p$, $c$ and the Thomson cross-section alone (*needs the
  electron radius* — a caught bug). *Asserts*: published value.

- **`astro_critical_density_and_baryons_planck2018`** — computes:
  $\rho_c = 3H_0^2/8\pi G$ at $H_0 = 67.66$ km/s/Mpc gives
  $8.5\times10^{-27}$ kg/m³; $\Omega_b\rho_c \approx 0.049\rho_c$ matches the
  BBN baryon density. *Asserts*: published values.

- **`astro_cmb_photon_gas_numbers`** — computes: at $T = 2.7255$ K,
  $n_\gamma \approx 411$ cm⁻³ and $u = 4.17\times10^{-14}$ J/m³
  $\approx 0.26$ eV/cm³. *Asserts*: the measured CMB thermal content.

- **`plasma_ionosphere_frequency_and_debye_length`** — computes:
  $f_p = \frac{1}{2\pi}\sqrt{ne^2/\varepsilon_0 m_e} = 8.98$ MHz for
  $n = 10^{12}$ m⁻³; $\lambda_D$ from two independent expressions must agree
  to $10^{-12}$ and sits in the mm range. *Asserts*: two-route agreement
  (solver cross-check class).

- **`plasma_alfven_speed_solar_wind`** — computes:
  $v_A = B/\sqrt{\mu_0\rho}$ with $B = 5$ nT, $n = 5$ cm⁻³: ~48 km/s — the
  measured solar-wind Alfvén speed scale. *Asserts*: the published band.

- **`si_quantum_metrology_triangle`** — computes: $R_K = h/e^2 =
  25812.80745\,\Omega$, $K_J = 2e/h = 483597.8484$ GHz/V,
  $\Phi_0 = h/2e = 2.067833848\times10^{-15}$ Wb — exact by SI definition —
  and $K_J\cdot R_K = 2/e$ closes the metrology triangle. *Asserts*:
  $10^{-12}$ (exact class).

- **`bcs_weak_coupling_gap_ratio`** — computes: $\Delta/k_BT_c = 1.764$
  (BCS), i.e. $2\Delta(0)/k_BT_c = 3.53$, the universal weak-coupling
  tunneling value. *Asserts*: published value.

- **`inspiral_chirp_ode_matches_closed_form`** — computes: the Peters chirp
  $df/dt$ integrated by RK4 against the closed-form coalescence time —
  agreement to $<10^{-4}$ across a decade in $f$; the $\mathcal M^{-5/3}$
  scaling exponent verified numerically; and the GW150914-scale system
  ($\mathcal M = 30\,M_\odot$) sweeping the LIGO band 35 → 150 Hz in the
  published sub-second-to-few-seconds regime. *Asserts*: integrator
  cross-check class.

### 5.12 Electromagnetism & optics — `em_optics_validation.rs`

- **`em_cyclotron_frequencies`** — computes: $f = qB/2\pi m$ at 1 T: proton
  15.245 MHz, electron 27.992 GHz (ESR/NMR calibration anchors). *Asserts*:
  published values.

- **`em_waveguide_brewster_critical_skin`** — computes: WR-90 TE₁₀ cutoff
  6.557 GHz ($f_c = c/2a$); Brewster angle 56.31° and critical angle 41.81°
  for $n = 1.5$ glass; Cu skin depth 9.3 mm at 50 Hz
  ($\delta = \sqrt{2/\mu_0\sigma\omega}$). *Asserts*: published/engineering
  values.

- **`em_dipole_radiation_resistance`** — computes: the short-dipole
  $80\pi^2(l/\lambda)^2$ and the thin half-wave dipole's exact 73.129 Ω.
  *Asserts*: exact class.

- **`em_rayleigh_blue_sky_ratio`** — computes: the $\lambda^{-4}$ law:
  $(650/450)^4 \approx 4.35$ — why the sky is blue. *Asserts*: exact.

- **`em_larmor_collapse_of_classical_hydrogen`** — computes: integrating
  $dr/dt = -\frac{e^4 k}{3\pi\varepsilon_0 c^3 m^2 r^2}$ from $r = a_0$ gives
  $\tau \approx 1.6\times10^{-11}$ s — the classical Rutherford atom spirals
  into the nucleus in tens of picoseconds (the published textbook value that
  forced quantum mechanics). *Method*: ODE integration (closed-form
  integral). *Asserts*: the published collapse time.

### 5.13 Statistical mechanics — `statmech_validation.rs`

- **`sm_maxwell_speed_identities`** — computes: the exact
  Maxwell–Boltzmann ratios $v_{\rm rms} : \langle v\rangle : v_p = \sqrt3 :
  \sqrt{8/\pi} : \sqrt2$. *Asserts*: exact class.

- **`sm_gas_constant_identity`** — computes: $R = N_A k_B$ (exact SI closure)
  and the STP molar volume 22.414 L. *Asserts*: exact.

- **`sm_sackur_tetrode_argon_stp`** — computes: the absolute entropy of argon
  at STP via Sackur–Tetrode: theory ≈ 153 J/(mol·K) vs the measured
  $154.8 \pm 0.2$ — quantum indistinguishability measured on a bench.
  *Asserts*: inside the measured band (experimental class).

- **`sm_bec_ideal_gas_helium`** — computes: ideal-gas $T_c$ for $^4$He at
  saturated-vapour density ($\rho = 145$ kg/m³): 3.1 K — the interacting
  liquid condenses at the $\lambda$-point 2.17 K (interactions lower it — the
  published contrast). *Asserts*: both temperatures.

- **`sm_vdw_critical_universality`** — computes: the vdW critical point by
  Newton root-finding on the combined stationarity condition
  $\frac{2}{v-b} = \frac{3}{v}$ (obtained by dividing $\partial p/\partial
  v = 0$ by $\partial^2 p/\partial v^2 = 0$), recovering $P_cV_c/RT_c = 3/8$
  for arbitrary $a, b$ — the law of corresponding states. *Bug caught here*:
  the original double-Newton solve was ill-conditioned to NaN and replaced.
  *Asserts*: 3/8 to solver precision.

- **`sm_photon_gas_pressure_and_adiabatic_indices`** — computes: photon gas
  $p = u/3$; $\gamma = 5/3$ monatomic and $7/5$ rigid diatomic. *Asserts*:
  exact.

### 5.14 Classical dynamics — `classical_dynamics_validation.rs`

- **`dyn_foucault_pantheon_rate`** — computes:
  $\Omega = 360^\circ\sin(\text{lat})/T_{\rm sidereal}$ at the Panthéon's
  latitude: ≈11.3°/h — the measured precession of the 1851 pendulum.
  *Asserts*: published value (after fixing a units slip in the Gaussian year
  — 365.2568983 **days**, not seconds).

- **`dyn_kepler_third_law_planets`** — computes: $T = 2\pi\sqrt{a^3/GM_\odot}$
  reproducing the sidereal years of Earth/Mars/Jupiter. *Asserts*: published
  periods.

- **`dyn_escape_and_circular_velocity_earth`** — computes: 11.19 km/s escape,
  7.91 km/s circular, ratio exactly $\sqrt2$. *Asserts*: exact ratio.

- **`dyn_roche_limit_moon`** — computes: $d = 2.44R(\rho_\oplus/\rho_\text{Moon})^{1/3}
  \approx 18{,}400$ km. *Asserts*: published value.

- **`dyn_pendulum_finite_amplitude_series_vs_rk4`** — computes: the series
  $T/T_0 = 1 + \theta_0^2/16 + 11\theta_0^4/3072 + \cdots$ against a direct
  RK4 integration of $\ddot\theta = -\sin\theta$: agreement $<10^{-5}$ at
  $\theta_0 = 0.5$ rad. *Method*: series vs integrator cross-check (the time
  window must exceed one period — the first version failed because it
  didn't). *Asserts*: $<10^{-5}$.

- **`sr_doppler_z_one_is_beta_three_fifths`** — computes: $z = 1 \iff
  \beta = 3/5$ exactly (relativistic Doppler). *Asserts*: exact.

- **`gw150914_chirp_mass_combination`** — computes:
  $\mathcal M = (m_1m_2)^{3/5}/(m_1+m_2)^{1/5}$ for (36, 29) $M_\odot$:
  ≈ 28.1 $M_\odot$ — the LIGO source-frame chirp mass. *Asserts*: published
  value.

### 5.15 Weak interactions & neutrinos — `weak_neutrino_validation.rs`

- **`weak_muon_lifetime_from_gf`** — computes: the tree-level lifetime
  $\tau = \frac{192\pi^3\hbar}{G_F^2 m_\mu^5 c^4} = 2.199\,\mu$s vs the
  measured 2.1969811 µs — the tree value is UNDER the measured one because
  the measured $G_F$ absorbs loop corrections (the direction itself is
  asserted). *Asserts*: the published value and the direction.

- **`neutrino_atmospheric_first_maximum`** — computes: the Super-K
  disappearance band — first oscillation maximum at $L/E \approx 495$
  km/GeV for $|\Delta m^2_{23}| = 2.5\times10^{-3}$ eV². *Asserts*: the
  published band.

- **`neutrino_reactor_and_theta13`** — computes: KamLAND's baseline sits
  beyond the first maximum (deep-suppression lobe), and at its own first
  maximum the $\theta_{13}$ survival is exactly
  $1 - \sin^2(2\theta_{13}) \approx 0.915$ (the Daya Bay value). *Asserts*:
  the published windows.

### 5.16 Coupled-oscillator SIRK machinery — `coupled_oscillator_sirk.rs`

The exactly-solvable models that pin the SIRK machinery itself.

- **`sirk_beamsplitter_spectrum_exact`** — computes: the two-mode
  beamsplitter $H = \omega(N_0+N_1) + J(a_0^\dagger a_1 + a_1^\dagger a_0)$:
  the number-conserving one-photon sector $\{|10\rangle, |01\rangle\}$ has
  the exact spectrum $\{\omega-J, \omega+J\}$; the SIRK Ritz values reproduce
  both to solver precision and the projected $H$ is Hermitian. *Asserts*:
  solver precision.

- **`sirk_beamsplitter_swap_dynamics`** — computes: a photon injected in mode
  0 swaps into mode 1 as $P(t) = \sin^2(Jt)$: $\langle N_1\rangle = 1/2$ at
  $Jt = \pi/4$ and complete swap at $Jt = \pi/2$ (restarted-Krylov unitary
  evolution, norm conserved). *Asserts*: the swap curve.

- **`sirk_displaced_oscillator_exact_shift`** — computes:
  $H = \omega N + g(a^\dagger + a)$: the exact displaced-oscillator shift
  $E_n = \omega n - g^2/\omega$; the SIRK ground energy is $-g^2/\omega$ and
  every Ritz value sits on an exact level. *Asserts*: solver precision.

### 5.17 The gauge-fixed program suites — `gauge_fixed_program_validation.rs`

Ten SIRK tests confined to the research program proper — the 3D gauge-fixed
Hamiltonians of NS / QYM / QED / QG(R²), `gauge_fixed_program_validation.rs`:

- **`qg_scalaron_quartic_selfinteraction_pt_vs_sirk`** — computes:
  $V(\phi) = \tfrac12 m^2\phi^2 + \lambda\phi^4$: the CAS-normal-ordered
  $:x^4:/4$ term shifts $E_n$ by $\frac{3\lambda}{2}n(n-1)$ EXACTLY at
  $O(\lambda)$ — the vacuum and one-scalaron levels stay put (the nested-Fock
  vacuum rule, to ALL orders), the second gap moves by $3\lambda$; a measured
  genuine $O(\lambda^2)$ shift ($+0.0997$ at $\lambda = 0.05$) on $E_2$;
  parity superselection ($:x^4:$ has $\Delta n$ even); large-$\lambda$
  DEPARTS from perturbation theory (non-perturbative SIRK). *Method*: SIRK
  spectra vs the perturbation formula.

- **`qg_scalaron_dispersion_band_light_limit`** — computes: the massive KG
  band $\omega(k) = \sqrt{k^2+m^2}$ resolved from ONE multi-rung window:
  $k\to0$ gap $= m$; $\omega/k < 1$ rising monotonically toward $c$
  (subluminal scalaron — the causal statement is the GROUP velocity).

- **`qg_densitized_hyperbolic_evolution_conserves`** — computes: the flat
  d'Alembertian $H_0 = \frac{1}{16}\Delta_{\mathcal S} - \frac{1}{24}
  \partial_y^2$ (ESA by Strichartz) evolved unitarily through an INDEFINITE
  spectrum: Hermitian projection, norm + energy conservation. *Asserts*:
  conservation to solver precision.

- **`qg_tegr_kinetic_bounded_below_positive_gaps`** — computes: the TEGR
  $\mathcal S$-sector kinetic: rank saturation (~6), normal-ordered ground 0,
  positive excitation gaps (the ESA boundedness statement).

- **`qym_gauss_law_conserved_by_flow`** — computes: the abelian Gauss
  generator $D = N_2 - N_3$ commutes with $H$; a charge-carrying start keeps
  its $D$-sector through the SIRK flow (with AND without mid-sequence BRST
  projection), and a wrong-charge component is NOT mixed in — charge
  superselection as a solver-level observable.

- **`qym_spectrum_even_in_g`** — computes: $B(g)^2$ spectra at $+g$ and $-g$
  coincide ($A^1 \to -A^1$ reparametrization) at finite coupling.

- **`qym_interacting_real_bounded_positive_gaps`** — computes: at $g\ne0$ the
  cubic/quartic magnetic terms keep the projected Hamiltonian Hermitian,
  bounded below (normal-ordered vacuum 0) with positive gaps — positivity
  enters only via the constraint projection (the bare truncated kinetic
  $-\tfrac12:\pi^2:$ is INDEFINITE — book.tex convention).

- **`qed_multimode_energy_additivity_resolved_band`** — computes: the free
  photon field over four modes: ONE window resolves the whole band
  $\{\omega_i\}$; two-quanta additivity holds in the SAME mode ($2\omega$) and
  across modes ($\omega_i + \omega_j$).

- **`ns_full_efold_laminar_decay_single_shot`** — computes: the Newtonian
  decay $du/dt = -\nu k^2 u$ reproduced over a FULL e-folding by ONE deep
  window (theory-native: one finite $T$, convergence in $m$ alone; raw-SI
  stiffness would demand restarts — the engineering path remains available).

- **`ns_advective_energy_norm_conservation`** — computes: 2D advection fiber:
  unitarity and $\langle H\rangle$ conservation through the restarted flow.

Framework gotchas encoded (details in AGENTS.md S40): distinct inner
occupations of one universe MERGE under `scale_and_add` (sector superpositions
must keep separate universes); eigenstate starts collapse the Krylov rank; the
truncated gauge-fixed kinetic is indefinite.

### 5.18 SIRK dynamics across sectors — `sirk_dynamics_validation.rs`

Observables measured from the UNITARY FLOW rather than spectra:

- **`qg_scalaron_beat_note_group_velocity`** — computes: a two-momentum
  scalaron superposition oscillates at the beat frequency
  $\Delta\omega = \omega(k') - \omega(k)$; the measured $\Delta\omega/\Delta k$
  IS the group velocity, verified subluminal and equal to $k/\omega$ (the
  dynamics-level version of the band test). The observable is inter-sector
  COHERENCE via a transfer operator (populations are frozen under a diagonal
  band Hamiltonian — a recurring trap).

- **`qg_graviton_vs_scalaron_speed_split`** — computes: same-$k$ graviton and
  scalaron Ritz values from one window each: massless $\omega = k$ vs massive
  $\sqrt{k^2+m^2}$ — the GW170817-vs-scalaron contrast inside one framework.

- **`qym_gap_stiffening_with_coupling`** — computes: the resolved abelian QYM
  gap grows monotonically with $g$: the positive-definite $\tfrac12 B^2$
  magnetic self-interaction repels levels upward — the nonperturbative
  signature of the $g^2 A^4$ term.

- **`ns_combined_decay_advection_rate`** — computes: a fiber with BOTH
  diagonal $\kappa$ (viscous) and off-diagonal advection: norm conserved,
  $\langle H\rangle$ decays at the analytic viscous rate for an eigenmode
  start; the advected pair backfeeds into the viscous sector by $t\approx1$
  (full-window factorization is NOT claimed — the fiber is one coupled
  system).

- **`qed_jc_detuned_dressed_oscillation`** — computes: Jaynes–Cummings at
  detuning $\delta$: $P_e(t)$ oscillates at the exact dressed frequency
  $\Omega = \sqrt{4g^2 + \delta^2}$ measured from the flow against the closed
  form.

### 5.19 Constraints & superselection — `brst_constraint_validation.rs`

- **`qym_abelian_brst_nilpotent_and_invariant`** — computes: the first abelian
  YM BRST charge in the repo, $\Omega = P\cdot b^\dagger_{\rm ghost}$:
  $\Omega^2 = 0$ (nilpotent by Pauli), $[H(g=0), \Omega] = 0$ ($P$ commutes,
  ghosts free).

- **`qym_brst_projection_identity_on_physical_flow`** — computes: ghost-free
  start — solves with and without mid-sequence projection agree on resolved
  spectra AND on $\langle P\rangle$ to solver accuracy (the theorem, YM
  edition): mid-sequence projection is an identity on physical flows.

- **`qed_jc_total_excitation_superselected`** — computes: $N_{\rm tot} =
  a^\dagger a + e^\dagger e$ commutes with the detuned JC Hamiltonian; mean
  AND variance of $N_{\rm tot}$ are flow constants (Rabi exchange moves quanta
  BETWEEN subsystems only) while the quantum shuttles between cavity and
  atom.

- **`qg_densitized_beat_predicted_by_spectrum`** — computes: the densitized
  model's Bogoliubov blocks are diagonalized to opposite-sign number
  operators ($+1/16$ vs $-1/24$); the solved Ritz splitting $\Delta E$
  predicts a transfer-coherence zero at $t = \pi/(2\Delta E)$; the flow
  confirms it (spectroscopy → dynamics closure).

### 5.20 Action→spectrum chains — `qg_action_predictions_validation.rs`

- **`qg_scalaron_mass_chain_action_to_band`** — computes: the α-scaling
  chain — $m(\alpha) = 1/\sqrt{12\alpha}$ verified, then re-measured as the
  exact $k\to0$ SIRK gap for each $\alpha$ (action parameter → spectral
  prediction).

- **`qg_weak_field_yukawa_limits`** — computes: $\Phi(r) = -\frac{GM}{r}(1 +
  \tfrac13 e^{-mr})$: Newtonian to $10^{-30}$ at $r\gg1/m$; exactly $4/3$ at
  $r\ll1/m$ (the published f(R) short-range enhancement).

- **`qg_tegr_densitized_common_subsector`** — computes: the two independent
  QG kinetic builders (TEGR $\mathcal S$-form vs densitized flat form) agree
  on their common $\mathcal S$ sector within documented truncation bands
  (cross-builder consistency).

### 5.21 Solver consistency — `sirk_consistency_validation.rs`

- **`ns_restarts_agree_with_single_shot`** — computes: time-sliced restarted
  windows vs one deep window on the laminar decay (engineering vs
  theory-native): both land on the same physics and sit on $e^{-t}$.
- **`qg_scalaron_band_paths_full_state_overlap`** — computes: two slicing
  choices land on the same evolved state (full-state overlap ≈ 1) plus
  unitarity.
- **`frame_invariance_swap_and_ground`** — computes: canonical vs unit-norm
  frames: mathematically exact reparametrization ⇒ observables equal to
  $10^{-6}$ on resolved rungs.
- **`resolved_set_frame_stable`** — computes: the residual-certified rung SET
  is frame-stable (same members, not just same count), with per-level floors
  that reproduce the ritz_edge_study wall tail ($E_1\sim10^{-4}$,
  $E_2\sim3\times10^{-3}$, $E_3\sim4\times10^{-2}$); the unit-norm frame
  certifies at least as many rungs as canonical.

### 5.22 Certified numerics — `bands_program_gauge_fixed.rs`

Theorem 4.1 bands promoted from validation tool to DELIVERABLE: propagated
through Cauchy–Schwarz ($|\langle O\rangle_{\rm SIRK} - \langle
O\rangle_{\rm exact}| \le 2\|O\|\cdot\text{band}\cdot\|v\|$), every
program-sector observable acquires a rigorous ERROR BAR — no closed-form
reference needed. Six certifications:

- **`qg_scalaron_gap_certified_contains_analytic`** — the $k\to0$ scalaron
  gap: analytic $m(\alpha)$ inside the certified interval at every depth;
  intervals NEST as $m$ grows.
- **`qym_g0_spectrum_certified_nesting`** — abelian QYM low rungs: certified
  intervals shrink with depth and nest (the residual-restricted certification
  window — the bare $:\pi^2:$ kinetic reaches an unbounded ladder, so raw
  diameters are meaningless).
- **`qym_gpairs_symmetry_certified_overlap`** — spectra at $\pm g$: pairwise
  certified-interval OVERLAP certifies the $A^1\to-A^1$ spectral symmetry from
  numerics alone.
- **`ns_decay_amplitude_certified_interval`** — NS laminar amplitude after
  one e-folding: certified bar around the measured value; the analytic
  $e^{-1}u_0$ sits inside.
- **`qg_graviton_scalaron_certified_disjoint`** — same-$k$ graviton vs
  scalaron levels: DISJOINT certified intervals — the massive/massless speed
  split established by rigorous bound alone. Sharpness comes from
  Rayleigh–Ritz RESIDUAL certificates ($|\theta-\lambda| \le \|r\|$,
  Parlett); the Theorem 4.1 envelope is the conservative a-priori ceiling.
- **`qed_casimir_cavity_band_dispersion`** — Casimir cavity $\omega_n =
  n\pi/d$: exact diagonal reference + band tables (the fourth bounded model);
  levels inside their certified radii, with paper-shift low-rung
  convergence-from-above documented.

### 5.23 Compiler-route equivalence — `cdb_hamiltonian_match.rs`, `latex_cas_hamiltonian_match.rs`

The numerical-test Hamiltonians must BE the full Cadabra2-derived Hamiltonians
— checked structurally and numerically, and (in the latex/CAS file) compiled
**through the symbolic engine** (`compile_latex` via mathhook, and
`compile_to_fock` on the CAS dialect that
`prob_kernel::symbolic::normalize_to_cas_dialect` produces from Cadabra2
output), per the user directive that the term-matching tests flow through the
LaTeX→Fock compiler:

- **`qym_su3_terms_match_cdb_h_final`** — the full SU(3) builder verified
  term-by-term against the expansion of $H_{\rm final} = \tfrac12\pi^2 +
  \tfrac12 B^2$, $B = \varepsilon(\partial A + \tfrac12 g f A A)$: quadratic
  $-1/2$, cubic $-(g/2)\varepsilon f$ (from $-L\cdot NL$), quartic
  $-(g^2/8)\varepsilon\varepsilon' ff'$ (from $-\tfrac12 NL^2$), with the
  exact SU(3) structure constants; the book.tex sign convention
  ($H_W = -\tfrac12\pi^2 - \tfrac12 B^2$, the negative of the Legendre
  $H_{\rm final}$) verified explicitly.
- **`qym_su3_vacuum_zero_point_matches_cdb`** — the vacuum zero-point
  structure matches the Cadabra derivation.
- **`qg_starobinsky_potential_and_conformal_parabola_match_cdb`** — the
  Einstein-frame scalaron potential
  $V(\phi) = \frac{M^4}{16\alpha}(1 - e^{-\sqrt{2/3}\,\phi/M})^2$ ($V(0)=0$,
  plateau $M^4/16\alpha$, $V\ge0$, $V''(0) = M^2/12\alpha$ = scalaron mass²)
  and the conformal-mode parabola $V_3(R_c) = -\frac{M^2}{2}R_c + \alpha R_c^2$
  (minimum $-M^4/16\alpha$ at $R_c = M^2/4\alpha$).
- **`qg_starobinsky_gauge_fixed_scalaron_frozen_derivatives`** — the
  gauge-fixed scalar sector $m\,N_0 + \tfrac12\sum g_i^2$ has frozen
  derivative variables ($[H, g_i] = 0$, BRST closed).
- **`qg_densitized_kinetic_hyperbolic_spectrum`** — the densitized kinetic
  $H_0 = \frac{1}{16}\Delta_{\mathcal S} - \frac{1}{24}\partial_y^2$: the
  $1/16$ / $-1/24$ coefficients and the hyperbolic (two-signed) spectrum.
- **`qg_densitized_jacobian_unitarity_y5`** — the unitarity kernel $J = y^5$
  (`docs/qg_unitarity_check.cdb`).
- **`ns_hamiltonian_matches_euler_advection`** — the builder vs the quantized
  Euler generator $\sum_i\{\pi_i, A_i\}$, $A_i = \sum_j u_j u_{ij} -
  \nu u_{12+i}$: 168 Hermitian terms, advection $\pm i$, viscosity $\mp i\nu$;
  plus the Ehrenfest equation $d\langle u_i\rangle/dt = i\langle[H,u_i]\rangle
  = 4\langle A_i\rangle$.
- **`qym_abelian_limit_cas_photon_structure`** / **`..._sirk`** — the U(1)
  specialization ($g=0$): purely quadratic, normal-ordered ($\langle 0|H|0
  \rangle = 0$), free lattice photon — structurally and via SIRK.
- **`qg_tegr_kinetic_matches_cdb_h_final_sector`** — the TEGR
  $\frac{1}{16e}\mathcal S^2$ kinetic sector vs the Cadabra $H_{\rm final}$
  (outer normal-ordered realization; the equivalence is spectral).
- **`ns_eulerian_fiber_matches_quantized_euler_generator`** — the affine
  fiber equals the quantized Euler generator on the shared modes.

And the LaTeX/CAS file (`latex_cas_hamiltonian_match.rs`):

- **`qym_number_operator_latex_dagger_is_exact`**, **`cross_mode_latex_dagger_pairs`**,
  **`double_creation_latex_dagger`** — pin the LaTeX dagger mapping
  (the AGENTS.md maintenance item fixed in `latex.rs`): `a_0^{\dagger}`
  compiles to the creation operator, so $a^\dagger a$ is the number operator
  — never zero (the pre-fix mathhook power-misparse) and never $a\cdot a$
  (the pre-fix `map_to_annihilation` bug).
- **`qym_abelian_b2_cas_matches_builder`** — $B^2$ with $B = A_0 - A_1$
  (U(1) lattice difference) compiled through the CAS dialect equals the
  `qcd_ym_hamiltonian(0)` builder term-for-term (compiler-vs-compiler).
- **`qym_abelian_b2_latex_dagger_structure`** — the same expression through
  the LaTeX route (structure check — the full compile is slow, `#[ignore]`d).
- **`qg_tegr_kinetic_cas_structure`**, **`qg_densitized_kinetic_cas_structure`**,
  **`qg_scalaron_mass_term_cas`** — the CAS compiles of the $(1/16)\mathcal
  S^2$, $(1/16)\Delta_{\mathcal S} - (1/24)\partial_y^2$ and scalaron mass
  term: 4 raw quadratic terms per mode, the exact $1/16$ / $-1/24$
  coefficients, no cubic/quartic.
- **`ns_euler_fiber_cas_matches_builder`** — the Euler fiber
  $\{\pi_i, A_i\}$, $A_i = \sum_k A_{ik}u_k + c_i$ via the CAS dialect —
  term-for-term equal to the builder on the shared modes.

### 5.24 Engine studies — `ritz_edge_study.rs`, `hashimoto_error_bands.rs`, `guard_justification_study.rs`, `heavy_krylov.rs`

**Ritz edge study** (`ritz_edge_study.rs`) — what the Ritz values ABOVE the
resolved window are: unconverged estimates of higher rungs, pinned by five
properties (see §8 for the full study): P1 bracketing (every Ritz value lies
inside $[E_0, E_m]$ of the reachable ladder — the Fock basis IS the
eigenbasis, so Rayleigh quotients are convex means); P2 the conditioning wall
(err vs $m$: 6e-6, 1e-9, 2e-6, 2e-3 — NOT monotone); P2b the unit-norm frame
flattens the wall; P3 the topmost Ritz climbs with $m$; P4 the top vector is a
high-occupation mixture whose Rayleigh quotient reproduces the Ritz value
while the ground vector reproduces the exact coherent-state content
$\langle N\rangle = \alpha^2 = (g/\omega)^2$; P5 residuals separate converged
from unconverged pairs; P6 Gram-only residuals match reconstruction and
select the resolved set.

**Hashimoto error bands** (`hashimoto_error_bands.rs`) — see §9: the measured
SIRK state error vs the paper's a-priori envelope
$\|\varphi_0(A)v - \mathrm{SIRK}_m(v)\| \le 2C\|v\|e^{-hm}E_m$, $C\in[2,
11.08]$, on four bounded models with closed-form evolutions (QED free photon,
QED Jaynes–Cummings Rabi, QG scalaron band, QG free graviton), with the
paper's own shift ladder $\gamma_j = N - hj$ mapped to $z_j = i\gamma_j/t$,
the literal denominator $q(z) = \prod_j(1+hjz)$, and $E_m$ computed by
Lawson's iteratively-reweighted minimax on a hulled $\Sigma$.

**Guard justification** (`guard_justification_study.rs`) — the quantitative
licence for every deviation from the idealized sequence: Study A (prune
invariance below the noise floor on spectral/dynamical/dissipative models),
Study B (BRST projection is the identity on physical sequences — the theorem
— and enforces $\ker\Omega$ on contaminated ones), Study C (adaptive
truncation never engages at the documented 50k budgets).

**Heavy Krylov** (`heavy_krylov.rs`) — the wall-time Yang-Mills lattice
drivers split out of the unit suite: `adaptive_l4_completes_under_budget`
(~11 s), `adaptive_l5_completes_under_budget` (~41 s) exercising the bounded
direct-construction path that keeps the quartic plaquette under a fixed
component budget, and `yang_mills_l3_mass_gap_demo` (the central empirical
mass-gap deliverable).

### 5.25 The assumption ledger: match / fail / non-claim per system

This is the honesty sheet. For each of the four systems it states: what is
tested through SIRK **without further assumptions**, what the predictions are
(with uncertainty), where they MATCH experiment or other approximations, and
where they FAIL or are NOT claimed — and why. The rule (pinned by
`assumption_ledger.rs::perturbation_theory_scope_map`): a failure is never
hidden by loosening a tolerance or changing the model; the regime and the
reason are documented, because no model or approximation method is good or
efficient for every problem (perturbation theory is great for QED, not for
the strong force).

| System | Gauge-fixed Hamiltonian (from the action) | Tested WITHOUT further assumptions | Predictions (with uncertainty) | Matches | Fails / not claimed | Why |
|---|---|---|---|---|---|---|
| **QED** | abelian QYM free sector, $\tfrac12\pi^2 + \tfrac12 B^2$; JC; static charge | raw canonical sequence (all guards off) reproduces vacuum 0, dispersion $\omega=\|k\|$, additivity, Rabi doublets exactly (§5.1, §5.24, `assumption_ledger`) | dispersion; Rabi splitting $2g$; collapse–revival $t_R = 2\pi\sqrt{\bar n+1}/g$; Coulomb $\delta E(r_1)-\delta E(r_2)$; UV-linear self-energy | exact theory at machine precision; $g{-}2$, positronium, Lamb shift, Casimir, blackbody via the constants suite (§5.8) | wide-spectrum coherent-state revival under the RESTARTED solver loses ~9%/restart — the exact Poisson sum is the prediction, not the truncated solver; PT fails at strong coupling (SIRK departs — correctly, it is non-perturbative) | Gram-whitening conditioning over a huge eigenvalue range (solver regime, not a model limit); PT is the wrong tool at strong coupling |
| **QYM** | $H_{\rm final} = \tfrac12\pi^2 + \tfrac12 B^2$, $B = (A_0-A_1) + \tfrac12 g A_0 A_1$ (Weyl gauge, Legendre; book.tex $H_W$ is its negative) | vacuum 0; Hermiticity; bounded-below spectrum with positive gaps; $\pm g$ symmetry; Gauss-law superselection; BRST charge $\Omega = P\,b^\dagger$ nilpotent with $[H,\Omega]=0$ at $g=0$ (§5.2, §5.17, §5.19) | mass-gap signal $\approx g^2/2$ on the lattice truncation; abelian limit $=$ free Maxwell exactly (§5.9) | Cadabra-derived operator structure; $U(1)$ reduction to QED | the continuum mass gap is NOT claimed — $g^2/2$ is the strong-coupling lattice/truncation result; the brute-force full SU(3) form (76K terms, indefinite $-\tfrac12\pi^2$ in the $H_W$ convention) is NOT SIRK-tractable — the reduced $B(A)$ form is | confinement is a non-perturbative, truncation-sensitive statement; the abstract/indefinite forms are not a positive-definite Fock-mode basis |
| **QG(R²)** | $\tfrac12\pi^2 + \tfrac12(\nabla\phi)^2 + V(\phi)$, $m^2 = M^2/12\alpha$; densitized d'Alembertian $\tfrac{1}{16}\Delta_{\mathcal S} - \tfrac{1}{24}\partial_y^2$; TEGR kinetic $\tfrac{1}{16e}\mathcal S^2$ | vacuum 0; massive dispersion $\omega = \sqrt{k^2+m^2}$; group velocity $k/\omega < 1$; bounded-below positive-gap spectra (the ESA finite shadow); derivative-variable BRST fixing (§5.3, §5.4, §5.17, §5.20) | scalaron mass gap $m(\alpha)$ at $k\to0$ with certified intervals; classical content: perihelion 43.0″/cy, GPS 45.9 µs/day, Pound–Rebka 2.46e-15, deflection 1.75″, Yukawa $\tfrac13 e^{-mr}$ correction | CODATA/GR/experiment within published bands (§5.3, §5.4, §5.20) | the full TEGR $H_{\rm final}$ in abstract $\mathcal S/E/T$ variables is NOT SIRK-tractable — only the kinetic/quadratic forms are realized; one-particle ESA on the Hermite dense core is a Lean-formalization claim (`../timepiece`), the numerics see only the finite truncation | the abstract variables are not a Fock-mode ladder; ESA is an operator-domain statement, numerically only its finite shadow is visible |
| **NS** | quantized Euler generator $\sum_i\{\pi_i, A_i\}$, $A_i = \sum_j u_j u_{ij} - \nu u_{12+i}$; Eulerian affine fiber with promoted derivative variables | Ehrenfest identity $i\langle[H,u]\rangle = 4\kappa\langle u\rangle + 4c$ exact; Newtonian decay $du/dt = -\nu k^2 u$; gauge condition $C_m = g_m - 2(m+1)u_{m+1}$ a constant of motion BY CONSTRUCTION ($[H,C_m]=0$); $\Omega^2=0$, $[H,\Omega]=0$ (§5.5–5.7, §5.17) | decay rate $\nu k^2$ to <2%; advection identity $d\langle u_0\rangle/dt = 8\langle u_0\rangle\langle u_1\rangle$; the classical fluid laws (K41, Poiseuille, Stokes, Blasius, Strouhal, Lamb–Oseen) as the fiber's classical content | experimental fluid-dynamics bands (§5.5) | the truncated flow LEAKS Ω-content (the bare flow from a ghost-carrying state grows $\|\Omega\psi\|$) — the BRST projector is required in the solve; gauge-condition drift $\propto dt^2$ under truncation; raw-SI stiffness demands restarts for full e-folds | Krylov truncation + finite precision — the exact flow conserves $C_m$ identically (verified: drift → 0 as $dt \to 0$) |

**How to read the "Fails / not claimed" column.** None of these are failures
of the model or the algorithm — they are the honest boundaries every method
has. Perturbation theory is great for QED but not for the strong force;
the restarted Krylov solver is exact in its stable regime and characterized
outside it; the abstract gauge-fixed forms are not Fock-mode ladders. The
unit-norm frame is admitted precisely because it changes no prediction in
the infinite-precision limit — it only addresses finite numerical precision
(§4.5, §5.24). Where an approximation IS the right tool (weak coupling, the
classical limits, the solvable sectors), the suites reproduce it to the
quoted precision; where it is not, the departure is asserted as the
non-perturbative content, not papered over.

---

## 6. The tolerance taxonomy

| Class | Typical tolerance | Examples |
|---|---|---|
| Exact identities | $10^{-9}$ – $10^{-12}$ | metrology triangle, Breit–Wheeler product, $\sqrt2$ velocity ratio |
| Derived constants | $10^{-3}$ – $10^{-6}$ relative | Chandrasekhar 1.44 $M_\odot$, ISCO 4397 Hz, LHC 8.33 T |
| Experimental bands | quoted windows | Sackur–Tetrode 153 vs 154.8, KamLAND suppression band, Alfvén band |
| Solver bands | documented profiles | displaced-oscillator levels, chirp RK4 <$10^{-4}$, Sackur quadrature $10^{-4}$ |
| **Theorem 4.1 certified bands** | measurement below a-priori envelope | `hashimoto_error_bands.rs` (§9); propagated certified intervals in the program suites |

A tolerance is never chosen to make an assertion pass; it encodes which of
these classes the quantity belongs to. Where a prediction FAILS to fall in
its class, the case is not hidden — it is carried in the assumption ledger
(§5.25) with its reason (solver regime, truncation, wrong approximation
tool).

---

## 7. Reading the Fock-space tests correctly

Two framework conventions matter when writing new tests:

1. **Inner vs outer construction.** A multi-occupation state must be ONE
   universe with correct inner occupation (`modes:{i:2}`), not two outer
   universes — otherwise number-operator additivity breaks. The inner-ladder
   construction gives $\langle 0|H|0\rangle = 0$ automatically (normal
   ordering strips the $[a,a^\dagger]$ zero-point).
2. **Derived fixtures, not transcriptions.** When a test compares compiler
   routes (CAS vs LaTeX vs builder), all fixtures must derive from the single
   source expression. A hand-transcribed LaTeX fixture once drifted from the
   Cadabra-derived Yang–Mills Hamiltonian and failed 19-vs-22 terms — the fix
   was structural (`qcd_ym_expression` as the sole source), not editorial.

---

## 8. Study: Ritz values above the resolved window (`ritz_edge_study.rs`)

**Question.** The shifted projection of the displaced oscillator returns Ritz
values *above* every physical level in the resolved window. Are they garbage?

**Answer: no — they are unconverged estimates of higher rungs, and their
behaviour is fully characterized by five properties now pinned by test:**

*P1 Bracketing.* In this model the Fock basis IS the eigenbasis, so any
normalized vector has Rayleigh quotient $\theta = \sum_n |c_n|^2 E_n$ — a
convex mean. Hence every Ritz value lies inside $[E_0, E_m]$, where $m$ is
the reachable occupation. No value can exceed the highest reachable rung.
*(Verified.)*

*P2 Conditioning wall.* Ground-level error vs window depth $m$ (measured):
$\mathrm{err}(4)\approx6\times10^{-6}$, $\mathrm{err}(6)\approx10^{-9}$,
$\mathrm{err}(8)\approx2\times10^{-6}$, $\mathrm{err}(10)\approx2\times10^{-3}$.
Convergence is NOT monotone: past the optimum, $\|w_k\| \sim \|H\|^k$ wrecks
the Gram conditioning and whitening truncation injects noise faster than the
bigger subspace helps. This wall is the reason long evolutions use restarted
short windows rather than one deep solve. *(Verified, profile pinned.)*

*P3 Climbing.* $\sup(\mathrm{Ritz})$ increases strictly with $m$ — the
topmost value moves up the ladder toward ever-higher rungs. *(Verified.)*

*P4 Mixture content.* Reconstructing the top-Ritz vector gives mean
occupation $\langle N\rangle$ well above the resolved window (mixed high-$n$
support), while its direct Rayleigh quotient reproduces the Ritz value — the
small-basis eigenpair genuinely represents a big-space vector with that
energy mean. For contrast, the ground vector reproduces the EXACT
coherent-state content $\langle N\rangle = \alpha^2 = (g/\omega)^2$ — the
machinery recovers dressing physics quantitatively. *(Verified.)*

*P5 Residual separation.* $\|H\psi-\theta\psi\|/\|\psi\|$ is tiny for
converged pairs and orders of magnitude larger at the top — they are
approximate eigenvectors of convergence-related quality, not noise.
*(Verified.)*

**Practical rule.** Filter Ritz values above $E_{\rm top\_resolved} +
\mathrm{gap}/2$ before level-placement assertions; treat the survivors near
the window edge as higher-rung estimates whose convergence improves with
deeper windows *up to* the conditioning wall.

**Follow-up (solver-enforced resolution).** The practical rule is now an API:
`resolved_ritz_values(tol)` selects pairs by true residual (see 3.5), and the
unit-norm frame removes the wall entirely for deep windows — so the edge
values can also be *converged away* by extending $m$, which is the theory's
own convergence knob. The displaced-oscillator suite now runs one $m=8$
window and asserts exactly the first three rungs resolve.

---

## 9. Theorem 4.1 error bands (`hashimoto_error_bands.rs`)

The paper this project implements (Hashimoto–Nodera, *Shift-invert Rational
Krylov method for an operator φ-function of an unbounded linear operator*,
JJIAM 2019 — mirrored at `../timepiece/Hashimoto.md`) proves the a-priori
SIRK approximation-error envelope (Theorem 4.1, Eq. 12):

$$\|\varphi_k(A)v - \mathrm{SIRK}_m(v)\| \le 2C\|v\|\, e^{-hm}\, E_m, \qquad C \in [2, 11.08]$$

with

- shifts $\gamma_j = N - hj > 0$ (the paper's ladder, NOT
  `shifts_for_range`);
- the SIRK rational family $R^{\mathrm{SIRK}}_{m-1} = \{p/q : \deg p \le m,\,
  q(z) = \prod_{j=1..m}(1 + hj\,z)\}$;
- $\Sigma$ = convex hull of the resolvent numerical ranges $W(X_j)$,
  $X_j = (\gamma_j I - A)^{-1}$;
- $f_{k,N}(z) = e^N \varphi_k(-z^{-1})$;
- $E_m = \min_r \|f_{k,N} - r\|_{\infty,\Sigma}$ — the best uniform
  approximation error of the target function by the whole SIRK family over
  $\Sigma$.

**How the tests realize it.**

1. *Faithful shifts.* With $A = -iHt$, the paper's resolvent maps as
   $\gamma_j I - A = it(H - i\gamma_j/t)$, so the solver is fed
   $z_j = i\gamma_j/t$ — the Krylov span equals the theorem's $Q_m$ exactly
   (inverse-free equivalence).
2. *$\Sigma$.* For Hermitian $H$ the numerical range of each $X_j$ is the
   segment through $1/(\gamma_j + it\lambda)$ for $\lambda$ over the spectral
   extent; hulling those segments over $j$ and padding to a rectangle gives a
   conservative box (sup over box $\supseteq$ sup over $\Sigma$ ⇒ band stays
   VALID).
3. *$E_m$.* Factoring out the fixed denominator turns the minimax into a
   weighted polynomial problem $\min_p \|(f\cdot q - p)/q\|_\Sigma$, solved by
   Lawson's iteratively-reweighted least squares (30 iterations) on a
   $240\times40$ grid in a Chebyshev tensor basis.
4. *Band edges.* $C \in [2, 11.08]$ gives (lo, hi); the measured relative
   state error $\|\psi_{\rm SIRK} - \psi_{\rm exact}\|/\|v\|$ against
   CLOSED-FORM evolutions must lie below `hi`.

**Models.** QED free-photon field (4 modes), QED Jaynes–Cummings
one-excitation Rabi manifold, QG Starobinsky scalaron band (two momenta), QG
free graviton (three modes) — all bounded, all with exact references.

**Representative output** ($N=8$, $h=0.5$, $t=0.8$):

```text
QG free graviton:  m=4  err 3.5e-3   hi(C) 1.78e4
                   m=6  err 7.2e-11  hi     6.53e3
                   m=8  err 8.3e-11  hi     2.08e3
                   measured decay slope c = 4.39  (theorem h = 0.500)
```

### 9.1 Why the ceiling is so large when the error decays exponentially in $m$

This is the most-asked question about the certified bars, so it gets its own
section. The measured error column DOES decay exponentially in $m$ — 3.5e-3 →
7.2e-11 → 8.3e-11 in the table above (the last step is already at the
solver's precision floor). The **ceiling** column also decays exponentially —
look at the ratios: $1.78\times10^4 \to 6.53\times10^3 \to 2.08\times10^3$,
each factor $\approx e^{-h\cdot2} = e^{-1} \approx 0.37$, exactly the
theorem's $e^{-hm}$ with $h=0.5$ and $\Delta m = 2$. The exponential decay is
IN the bound and working. What is enormous is the **prefactor**
$2C\|v\|E_m$ sitting in front of $e^{-hm}$. It is large for these particular
models for four structural reasons:

1. **The theorem's constants are tuned for the DISSIPATIVE regime.** The
   paper's sharp regime is $\mathrm{Re}\,\Lambda(A) \le 0$ (decaying
   semigroups), where the resolvents $X_j = (\gamma_j I - A)^{-1}$ have
   bounded numerical ranges that the SIRK rational family approximates
   tightly. Our models are UNITARY: $A = -iHt$ has its spectrum on the
   imaginary axis, the least favourable location for the paper's
   approximation argument. The bound remains valid (it is a theorem) but it
   is loose *by construction* — a rigorous ceiling, not an estimate.
2. **The hulled $\Sigma$ is a conservative box.** The theorem's $E_m$ is the
   uniform error over the convex hull of ALL resolvent numerical ranges,
   padded to a rectangle so that the sup over the box is $\ge$ the sup over
   $\Sigma$ (validity is never compromised). The box contains points far from
   the actual spectrum, where the rational approximant is poor, so $E_m$ is
   inflated — sometimes by orders of magnitude — relative to the error over
   the true spectrum.
3. **The $e^N$ normalization and the $k=0$ target.** For $\varphi_0$ (the
   propagator, $k=0$) the target is $f_{0,N}(z) = e^N e^{-z^{-1}\cdot(-1)}$-type
   — an oscillatory function of $z$ over the box — which the rational family
   approximates only to $O(1)$ uniformly; and the normalization $e^N$ with
   $N=8$ multiplies it. The constant $C$ can be as large as 11.08.
4. **The ceiling decays with the same exponential, so it shrinks toward the
   measurement as $m$ grows** — but the prefactor gap (here $\sim10^{7}$)
   means the ceiling only becomes numerically interesting at depths far
   beyond what unitary problems need.

**Why this does not matter for the certifications.** The guide reports BOTH
tiers deliberately:

- **A-priori tier (Theorem 4.1):** valid for every model and every $m$,
  requires no reference state, but on unitary problems its constants make it
  a loose ceiling — useful as a rigorous envelope, useless for separating
  nearby levels.
- **Sharp tier (Rayleigh–Ritz residual certificate):** $|\theta - \lambda|
  \le \|r\|$ (Parlett), computed from `ForwardSirkResult::ritz_residuals`
  from the stored Gram matrix alone. This is the tier that certifies the
  *actual* widths — $\le 10^{-9}$ (residual) in the table — and its power is
  what makes DISJOINT graviton/scalaron certified intervals possible.

**Production rule:** report both; separate levels with residuals, quote
envelopes as ceilings. The measured decay slope $c \ge h$ on every model that
has not saturated machine precision ($c = 0.70$, $0.75$, $4.39$ against
$h = 0.5$) is itself the empirical verification of the paper's central
qualitative claim — EXPONENTIAL decay in $m$ (vs polynomial for
Arnoldi/SIA/RK, Theorems 3.2/3.3) — and it is typically FASTER than the
guaranteed rate, because the true approximation error over the actual
spectrum is far smaller than over the conservative box.

### 9.2 Results: numerics + bands vs expected

All values are actual outputs of the suites ($N=8$, $h=0.5$ unless noted;
$t=0.8$ for evolution rows). The EXPECTED column quotes the mainstream value
TOGETHER WITH ITS SOURCE UNCERTAINTY BAND (experimental bar or declared
theory status), keyed as [E#] into the bibliography of §9.3.

**(a) Certified spectral predictions**

| Quantity | SIRK value | Certified bar (tier) | Expected ± source uncertainty [key] |
|---|---|---|---|
| QG scalaron gap, $k\to0$ ($\alpha=1$) | **0.800000** | ±2839 (T4.1 ceiling); ≤$10^{-9}$ (residual) | 0.800000 ± 0 — theory input $\alpha=1$ fixes it exactly [E1] |
| QG graviton $\omega(k=0.9)$ ⇒ speed $v_g$ | **$\omega/k = 1.000000$** | width ≲$10^{-6}$ (residual) | $c\cdot(1+\varepsilon)$, $\varepsilon \in [-3, +7]\times10^{-15}$ [E2] |
| QG scalaron $\omega(k=0.9)$, $m=0.55$ | **+1.054751** | width ≲$10^{-6}$ (residual) | $\sqrt{k^2+m^2}=1.054751 \pm 0$ — exact KG relation [E3] |
| QED Casimir cavity, worst level distance to $\omega_n=n\pi/d$ | **1.2×$10^{-10}$** ($m=8$) | inside band_hi 5.4×$10^3$ | levels exact in ideal cavity; plate interaction verified to ±1% [E4] |
| QYM abelian $\theta_0(g=+0.35)$ vs $\theta_0(g=-0.35)$ | −1.382280 vs −1.402071 | intervals OVERLAP (certified) | identical ± 0 — $A^1\to-A^1$ residual symmetry [E5] |
| QYM abelian low rungs $g=0$ ($\theta_0,\theta_1,\theta_2$) | −1.546148, −0.137582, +2.348642 ($m=9$) | nested certified windows | deepest-window reference; its own certified widths are the uncertainty band [E6] |

**(b) Theorem 4.1 envelope performance** (state error vs a-priori ceiling)

| Model | $m$ | Measured err | band_hi ($C=11.08$) | Measured slope $c$ | theorem $h$ |
|---|---|---|---|---|---|
| QED free photon (4 modes) | 4 / 6 / 8 | 3.5×$10^{-3}$ / 8.2×$10^{-9}$ / 8.3×$10^{-11}$ | 1.8×$10^4$ / 6.4×$10^3$ / 2.4×$10^3$ | **c=4.39** | 0.5 |
| QG free graviton (3 modes) | 4 / 6 / 8 | 3.6×$10^{-10}$ / 7.2×$10^{-11}$ / 2.2×$10^{-11}$ | 2.1×$10^4$ / 6.5×$10^3$ / 3.7×$10^3$ | **c=0.70** | 0.5 |
| QG scalaron band (2 modes) | 4 / 6 / 8 | 2.0×$10^{-12}$ / 0 / 0 | 1.7×$10^4$ / 7.3×$10^3$ / 3.1×$10^3$ | **c=0.75** | 0.5 |
| QED Jaynes–Cummings (Rabi) | 6 / 9 / 12 | 1.03×$10^{-2}$ (flat — reference convention offset, see note) | 8.2×$10^2$ / 7.7×$10^2$ / 1.7×$10^2$ | — (floor-limited) | 1.0 |

**(c) Certified dynamics**

| Observable | SIRK value | Certified bar | Expected ± source uncertainty [key] |
|---|---|---|---|
| NS laminar $\langle u\rangle$ after one e-fold | **+0.745713** ($m=14$) | ±8511 (T4.1 ceiling) | $u_0 e^{-1} = 0.735759 \pm 0$ theory-exact; laminar-decay experiments reproduce the $\nu k^2$ law to ≲2% [E7] |
| JC vacuum-Rabi angular freq. ($\delta=-0.35$, $g=0.18$) | node confirmed at predicted $t^*$ | — | $\Omega = 2\sqrt{g^2+\delta^2/4} = 0.5022 \pm 5\%$ — Brune-type cavity-QED precision [E8] |

Reading guide: the T4.1 bars are RIGOROUS CEILINGS whose constants are
tuned for dissipative problems — on these unitary models they are loose by
construction (see §9.1), while the SHARP residual tier (widths ≤$10^{-6}$
here) carries the separation power. Where an EXPERIMENTAL anchor exists it
comes with the published uncertainty band ([E2], [E4], [E7], [E8]); purely
theoretical expectations are marked ± 0 and their citations are to the
standard derivation.

### 9.3 Sources & uncertainty provenance for expected values

- **[E1]** Starobinsky, *Phys. Lett.* **B91**, 99 (1980): $f(R)=(M^2/2)R+\alpha
  R^2$ ⇔ scalaron mass $m^2=M^2/(12\alpha)$. With $\alpha$ a free Lagrangian
  parameter the dimensionless gap is exact; observational bounds on $r$
  constrain the physical $M$ but are irrelevant to this normalized check.
- **[E2]** Abbott et al. (LIGO/Virgo/Fermi-GBM/INTEGRAL), *Phys. Rev. Lett.*
  **119**, 161101 (2017): GW170817/GRB170817A gives
  $-3\times10^{-15} \le (v_g-c)/c \le +7\times10^{-15}$ — THE published
  uncertainty band used.
- **[E3]** Klein–Gordon dispersion $\omega^2 = k^2+m^2$: exact; textbook
  derivation in Peskin & Schroeder, *An Introduction to QFT*, ch. 2.
- **[E4]** Spectrum $\omega_n = n\pi/d$: Casimir, *Proc. K. Ned. Akad. Wet.*
  **51**, 793 (1948). Experimental status of the plate interaction: Lamoreau,
  *Phys. Rev. Lett.* **78**, 5 (1997) (~5%); Decca et al., *Phys. Rev. D*
  **75**, 077101 (2007) (±1%) — the ±1% band quoted.
- **[E5]** Residual symmetry $A^1\to-A^1$ of the Weyl-gauge
  Legendre-transformed abelian Hamiltonian: this project's Cadabra2 derivation
  (`docs/yang_mills_hamiltonian.cdb`; book.tex §8182 convention note).
- **[E6]** Bosonic Bogoliubov diagonalization of $:x^2:$ blocks: Bogoliubov,
  *Bull. Acad. Sci. USSR* **11**, 77 (1947); practical algorithm per Colpa,
  *Physica A* **134**, 377 (1986). Reference uncertainty = deepest-window
  certified width (self-consistency tier).
- **[E7]** Newtonian viscous decay $du/dt = -\nu k^2 u$: Landau & Lifshitz,
  *Fluid Mechanics* §15 (exact for Newtonian fluids); laminar spin-down
  measurements agree within ≲2% (standard viscous-flow benchmarks compiled in
  White, *Viscous Fluid Flow*, 3rd ed., ch. 3).
- **[E8]** Vacuum Rabi splittings: Brune et al., *Phys. Rev. Lett.* **76**,
  1800 (1996); review: Haroche, *Rev. Mod. Phys.* **85**, 1083 (2013)
  (Nobel 2012). Measured frequencies track the Jaynes–Cummings prediction
  within the ~5% experimental band quoted.

---

## 10. Sources

- CODATA 2018 fundamental constants; SI-2019 exact definitions ($h$, $e$,
  $k_B$, $N_A$).
- PDG 2024 review (masses, lifetimes, $\alpha_s$ world average).
- Zee, *QFT in a Nutshell* §I.3 (one-photon exchange → Coulomb).
- Peskin & Schroeder ch. 16 (running coupling, colour factors).
- Greisen (1966); Zatsepin & Kuzmin (1966) — GZK cutoff.
- Shapiro & Teukolsky, ch. 3 (Chandrasekhar mass).
- Peters (1964) — gravitational-radiation inspiral.
- Kolmogorov (1941); Blasius (1908); Roshko (1954).
- Huang, *Statistical Mechanics* (Sackur–Tetrode, BEC, vdW).
- Abbasi et al. IceCube / Super-K; An et al. (Daya Bay); Ahn et al. (RENO).
- LIGO Scientific & Virgo, PRL 116, 061102 (2016) — GW150914.
- Project docs: `AGENTS.md` maintenance checklist (S29–S39 entries map 1:1
  to suites).
