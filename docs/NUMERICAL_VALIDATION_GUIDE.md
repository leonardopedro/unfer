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
    --test coupled_oscillator_sirk --test ritz_edge_study \
    --test qym_mass_gap --test qed_extended_validation \
    --test qg_cosmology_validation --test ns_boundary_layer_validation \
    --test qed_further_validation --test qg_general_relativity_validation \
    --test qym_lattice_validation --test ng_newtonian_validation \
    --test ns_further_validation --test qed_kerr_photon_blockade \
    --test qed_hong_ou_mandel --test qed_blockade_statistics \
    --test qg_tegr_helicity --test qed_abelian_reduction \
    --test outer_vacuum_ground_validation

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
| `yang_mills_lattice` | Kogut–Susskind-inspired **comparison** lattice (NOT the Cadabra-derived gauge-fixed QYM — see `qcd_ym_hamiltonian`) | lattice strong-coupling gap $\approx g^2/2$ — a benchmark for the SIRK machinery, NOT a Millennium-positivity claim for this project's model |
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
  The one-particle Hamiltonian is enclosed at the outer level,
  $H = \sum_{i,j} h_{ij} C_i^\dagger A_j$, and diagonalized by SIRK: the outer
  the outer-enclosed final Hamiltonian is $H=\sum_{i,j}h_{ij}C_i^\dagger A_j$;
  the outer vacuum is exactly 0 and one-quanton energies come from the inner $h$.
  one-gluon Ritz $= |k|$ per mode, $n$-gluon additivity. *Setup*:
  $k \in \{0.5, 1.0, 1.5, 2.0\}$, $m=4$. *Asserts*: $10^{-6}$ (exact class).

- **`qcd_mass_gap_sirk`** — computes: the contrast between the massless free
  gluon ($E \to 0$ as $k\to 0$) and the confined Cadabra-derived 3D
  gauge-fixed QYM Hamiltonian `qcd_ym_hamiltonian(g)` ($H_{\rm final} =
  \tfrac12\pi^2 + \tfrac12 B^2$ in the nested Fock space), which is GAPPED:
  its SIRK R-reflection sector solves (R-even vacuum start, R-odd
  one-quantum start, unit-norm frame, $m=12$) give Rayleigh–Ritz upper
  bounds consistent with the exact truncated levels $E_1-E_0 = 0.091$ at
  $g=1$ (see §5.24a). *Method*: SIRK sector solves vs the exact N ≤ 8
  window. *Setup*: $g=1$, $m=12$; soft free-gluon mode $k=0.01$. *Asserts*:
  $E_{\rm soft} < 0.02$ (massless); gauge-fixed gap positive and consistent
  with the exact $E_1-E_0$. (Supersedes the lattice-based contrast — the
  lattice's $g^2/2$ electric gap is a comparison-model effect, not the
  gauge-fixed H's gap — and the earlier “no positive gap opens” reading,
  which missed the gap because a vacuum-start Krylov cannot resolve the
  R-odd first excitation. The $B^2$ pair terms lower the INNER one-particle
  reference below its unshifted zero — $E_0 = -2.74$ at $g=1$ on N ≤ 8 —
  while the shifted outer-enclosed Hamiltonian still has the outer vacuum as
  its exact ground.)

- **`qcd_ym_hamiltonian_outer_fock_sirk`** — computes: the structural facts
  of the Cadabra2-derived $H_{\rm final} = \tfrac12\pi^2 + \tfrac12 B^2$,
  $B = (A_0-A_1) + \tfrac12 g A_0 A_1$ (built through the CAS compiler,
  inner operators): $\langle 0|H|0\rangle = 0$ and a bounded-below spectrum
  with positive excitation gaps (Millennium-Prize positivity). *Method*:
  direct vacuum expectation + SIRK Ritz values ($m=8$) from the inner vacuum.
  *Setup*: $g=0.5$. *Asserts*: $|\langle 0|H|0\rangle| < 10^{-9}$; projected
  Hamiltonian Hermitian; $\ge 3$ resolved levels; $\lambda_0 > -10$;
  first three gaps $>0$. (Positivity here is bounded-below positivity of the
  Cadabra-derived gauge-fixed model in its finite truncation — a model-level
  statement, not a proof of the Millennium problem.)

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
  the one-particle graviton Hamiltonian is enclosed as
  $H = \sum_{i,j} h_{ij} C_i^\dagger A_j$ and diagonalized by SIRK — outer
  vacuum 0, one-graviton Ritz
  $= c|k|$, group velocity $d\omega/dk = c$ exactly (GW170817 constraint
  $|\Delta v/c| < 10^{-15}$), and the structural point that a massive
  dispersion $\omega = \sqrt{c^2k^2 + m^2}$ would have an increasing slope
  (non-linear). *Setup*: $k\in\{0.5,1.0,1.5,2.0\}$, energies $c k$ with
  $c = 299792458$; $m=4$. *Asserts*: $10^{-3}$ on energies (SI scale);
  group velocity to $10^{-12}$.

- **`qg_tegr_hamiltonian_outer_fock_sirk`** — computes structural facts of
  the final outer-Fock realization of the Cadabra2-derived TEGR one-particle
  kinetic Hamiltonian: $H = \sum h_{ij}C_i^\dagger A_j$, with creation left
  and annihilation right,
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

- **`qym_abelian_two_mode_two_photon_sector_exact_quadratic_form`** —
  computes: the two-mode abelian gauge-fixed Hamiltonian
  `qcd_ym_hamiltonian(0)` (Cadabra-derived $H_{\rm final} = \tfrac12\pi^2 +
  \tfrac12 B^2$, $B = A_0 - A_1$) is exactly the quadratic boson form
  $N_0+N_1 + \tfrac12(a^{\dagger 2}+a^2 \text{ per field mode}) -
  (a_0^\dagger a_1^\dagger + a_0^\dagger a_1 + a_1^\dagger a_0 + a_0 a_1) +
  (N_2-\tfrac12(a_2^{\dagger 2}+a_2^2)) + (N_3-\tfrac12(a_3^{\dagger
  2}+a_3^2))$ over the four Weyl-gauge ladder modes — verified as an exact
  matrix identity against the Cadabra builder. The $B^2$ cross-term carries
  the photon-**pair** operators: $\langle\mathrm{vac}|H|1,1\rangle = -1$
  exactly (pair creation/annihilation — vacuum squeezing). SIRK (unit-norm
  frame) resolves the coupled two-photon sector from $|1,1\rangle$: the
  lowest Ritz value is negative and converges down toward the analytic
  continuum floor $-2$ as $m$ grows ($-1.547$, $-1.618$ at $m = 10, 12$),
  and the coincidence flow conserves norm and energy. *Asserts*: matrix
  identity and pair matrix element to $10^{-9}$; Ritz negative, above $-2$,
  monotone in $m$; norm/energy to $10^{-6}$.

- **`qym_abelian_vacuum_polarization_one_loop_vs_sirk`** — computes: the
  vacuum-polarization structure of the abelian gauge-fixed Hamiltonian on
  the Cadabra builder itself. The one-loop shift
  $\delta E^{(2)} = \sum_n |\langle n|H|\mathrm{vac}\rangle|^2 /
  (E_{\rm vac} - E_n)$ over the five double-quanta intermediate states is
  $-\tfrac32$ exactly (the photon-pair analogue of the published
  $\sum c^2/(\omega - E_p - E'_p)$ benchmark that `qed_pair_production`
  checks — pair creation from the field strength, here of photon pairs).
  SIRK–Hashimoto on the full gauge-fixed $H$ reproduces the one-loop value
  at moderate Krylov depth ($-1.566$ at $m=8$, within $0.1$) and, being
  exact, converges monotonically DOWN toward the analytic continuum floor
  $-2$ ($-1.566 \to -1.640 \to -1.692 \to -1.731$ at $m = 8..14$) — the
  same perturbative-match + non-perturbative-departure structure the
  $\gamma \leftrightarrow e^+e^-$ test measures. *Asserts*: one-loop sum
  $= -\tfrac32$ to $10^{-9}$; $|E(m{=}8) - \delta E^{(2)}| < 0.1$;
  monotone decrease in $m$; $E(m{=}14) < \delta E^{(2)} - 0.1$ and
  $> -2.01$.

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

- **`sirk_displaced_oscillator_exact_shift`** — computes: the displaced
  oscillator realized as the **Cadabra-derived QED gauge-fixed sector**
  (`docs/yang_mills_hamiltonian.cdb`, abelian $H_{\rm final} = \tfrac12\pi^2
  + \tfrac12 B^2$ with the static-charge $A\cdot J$ coupling,
  `qed_static_charge_interaction`): $H = kN + g(B^\dagger+B)$ has the exact
  spectrum $E_n = kn - g^2/k$; the SIRK ground energy is $-g^2/k$ and every
  resolved Ritz value sits on an exact level (solver-enforced via
  `resolved_ritz_values`). *Asserts*: solver precision.

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

### 5.24a QYM mass gap — the gauge-fixed formalization — `qym_mass_gap.rs`

Executes the observable of `MASS_GAP_CERTIFIED.md` §3.3–§3.5 (see also
`docs/MASS_GAP_SPEC.md`) on the **Cadabra-derived 3D gauge-fixed QYM
Hamiltonian `qcd_ym_hamiltonian(g)`** ($H_{\rm final} = \tfrac12\pi^2 +
\tfrac12 B^2$ in the nested Fock space — the formalization object is the
gauge-fixed H, NOT the `yang_mills_lattice` builder). The exact Z₂ symmetry
is the reflection R: $(A_0,A_1)\to(-A_1,-A_0)$ (exact for ALL $g$ — the
lattice's occupation parity is not a symmetry at $g>0$), and the mass gap
lives BETWEEN the R-sectors: the gap $= \theta^o_0(m) - \theta^e_0(m)$
with the certified interval $[\theta^o_0-\theta^e_0-(\delta^o+\delta^e),
\theta^o_0-\theta^e_0+(\delta^o+\delta^e)]$. All solves run the
SIRK-Hashimoto algorithm (unit-norm frame, `--release`). 10 tests:

- **`qym_gauge_fixed_hamiltonian_nested_fock_structure`** — computes:
  normal ordering ($\langle 0|H|0\rangle = 0$), Hermiticity
  ($\|H-H^\dagger\| < 10^{-9}$), the pair coupling $\langle
  \mathrm{vac}|H|1,1\rangle = -1$ at $g=0$, and the appearance of genuine
  3-/4-operator non-abelian terms at $g>0$ (B a genuine function of A).
- **`qym_gauge_fixed_reflection_symmetry_sector_purity`** — computes:
  $[H,R] = 0$ to $10^{-16}$ for $g \in \{0,1,2\}$ and the R-even/R-odd SIRK
  chains are disjoint (max mutual overlap $10^{-16} < 10^{-8}$) — §3.3 item
  1 on the gauge-fixed H.
- **`qym_gauge_fixed_low_window_reflection_alternation`** — computes: the
  low spectrum at $g=1$ alternates R-parity exactly (R-parities
  [+1, −1, +1, −1] of $E_0..E_3$) — the first excitation is the
  reflection-odd partner of the ground, so the gap is the inter-sector gap.
- **`qym_gauge_fixed_spectral_gap_positive_stable`** — computes: the gap
  $E_1-E_0 = 0.0911$ (N ≤ 6) / $0.0912$ (N ≤ 8) at $g=1$ — positive and
  stable across truncations — with the SIRK Ritz values Rayleigh–Ritz upper
  bounds on the exact levels at every solved m.
- **`qym_gauge_fixed_abelian_limit_gapless`** — computes: at $g=0$ the
  truncated gap shrinks with depth (0.336 → 0.190 → 0.122 at N ≤ 4/6/8
  toward the $(X_0-X_1)$ continuum floor −2) and the R-sector grounds
  coincide at every m — the massless order parameter.
- **`qym_gauge_fixed_gap_grows_with_coupling`** — computes: $E_1-E_0$ (N≤8)
  = 0.0305 ($g=0.5$) < 0.0912 ($g=1$) < 1.2436 ($g=2$) — monotone growth
  in the coupling (the §3.5 statement for the gauge-fixed H; the lattice's
  $g^2/2$ electric law is superseded).
- **`qym_gauge_fixed_one_particle_ground_sector_structure`** — computes the
  inner one-particle squeezed reference; it does not identify the full nested
  ground state with that reference:
  $\langle 0|H|0\rangle = 0$ but the ground is pair-squeezed below it:
  $E_0 = -2.744$ ($g=1$, R-even) and $-7.755$ ($g=2$, R-odd on N ≤ 8) — the
  strong-coupling crossing breaks the lattice-era “even ground = vacuum”
  identification completely.
- **`qym_gauge_fixed_sirk_ritz_monotone_stable_in_m`** — computes: the
  sector-ground Ritz values tighten monotonically with m ($\theta^e_0$:
  −1.349 → −1.599; $\theta^o_0$: −1.839 → −2.206 at m = 8..14). Honest form
  of §3.3 item 3: SIRK subspaces at different m use different shift sets, so
  the honest statement is monotone tightening, not strict nesting.
- **`qym_gauge_fixed_certified_enclosure_of_exact_gap`** — computes: the
  certified interval $[\theta^o_0-\theta^e_0 \pm (\delta^o+\delta^e)]$ from
  the two sector solves encloses the exact truncated gap at $g \in \{1,2\}$
  ($g=1$: $[-8.009, 6.866] \ni 0.0912$). The lattice's tight $lo > 0$
  stopping rule is NOT reachable here — the squeezed ground's Krylov
  residuals ($\delta \approx 4$) honestly widen the interval; what is
  certified is the enclosure.
- **`qym_gauge_fixed_proof_facing_seam_agrees_manual_assembly`** —
  computes: the proof-facing seam `certified_mass_gap_parity` (solves the
  R-sectors + precondition enforcement + T6 assembly) agrees with the manual
  two-solve path to $10^{-12}$ and fires the spec predicates. *Asserts*:
  equality, enclosure, chain disjointness.

### 5.24b The mass-gap spec seam — `mass_gap_spec.rs` + `docs/MASS_GAP_SPEC.md`

The spec seam itself is unchanged; the regression-level statement of §3.5 is
now made on the gauge-fixed H (the lattice's $g^2/2$ electric law and the
strong-coupling $c/g^6$ fit were lattice-electric effects and are
superseded — see the gap-growth and certified-enclosure tests in §5.24a).

The non-Lean half of the `MASS_GAP_CERTIFIED.md` §4–§5 formalization route:
`fock_sirk::mass_gap_spec` is the **pure, dependency-free core** (plain
`f64`; no `nalgebra`, no I/O) that a translation tool (Aeneas/Verus, §5.3)
or a proof specialist can attach theorems to. Each function carries its exact
contract (precondition/postcondition/identity); `certified_mass_gap_parity`
(`forward_sirk.rs`) is the proof-facing seam that *runs* the two R-sector
solves (R-even vacuum start, R-odd one-quantum start) on the gauge-fixed H,
enforces the checkable preconditions at runtime (`debug_assert`: sector
purity via the chain-overlap witness — the lattice-era “even ground =
vacuum” precondition is dropped: the gauge-fixed R-even ground is the
pair-squeezed vacuum, not the Fock vacuum), and assembles the T6
certificate. `docs/MASS_GAP_SPEC.md` is the spec of record:
code → math mapping, the theorem statement, the three width terms, andthe numerically-pinned claims (the measured gap values and certified
windows).

- **`parlett_bound_holds_on_explicit_matrix`** — computes: the a-posteriori
  bound $|\theta-\lambda| \le \|H\psi-\theta\psi\|/\|\psi\|$ on the explicit
  $2\times2$ matrix $[[2,1],[1,2]]$ with known eigenpairs; zero for an exact
  eigenpair. *Asserts*: the inequality, exactness.
- **`certified_width_matches_certificate_delta`** — computes: the spec
  width agrees with `certificate::Certificate::delta()` (single source of
  truth). *Asserts*: $10^{-20}$.
- **`gap_assembly_and_interval_contracts`** — computes: the T6 lower bound,
  the gap interval, containment, and the stopping rule on synthetic
  certificates. *Asserts*: the formulas.
- **`parity_and_vacuum_preconditions`** — computes: the disjointness and
  vacuum predicates on synthetic witnesses. *Asserts*: truth tables.

### 5.24c QED extended — `qed_extended_validation.rs`

- **`qed_jc_dressed_splitting_scales_as_sqrt_n_plus_1`** — computes: at
  resonance the (n+1)-excitation JC sector $\{|n,e\rangle,|n+1,g\rangle\}$
  splits by exactly $2g\sqrt{n+1}$ with mean $\omega(n+1)$ (normal-ordered)
  — the photon-number-dependent Rabi frequency $\Omega_n = g\sqrt{n+1}$, the
  $\sqrt n$ ladder behind collapse/revival. *Asserts*: $10^{-8}$, and the
  ratio ladder $\sqrt 2, \sqrt 3, 2$.
- **`qed_coherent_state_poisson_statistics`** — computes: a truncated
  coherent state of the free field satisfies $\langle N\rangle =
  \mathrm{Var}(N) = |\alpha|^2$ and Mandel $Q = 0$ — the shot-noise floor.
  (Convention note pinned in the test: the framework's $|n\rangle =
  (a^\dagger)^n|0\rangle$ carries $\sqrt{n!}$, so the amplitudes are
  $\alpha^n/n!$.) *Asserts*: $10^{-6}$, $Q$.
- **`qed_zeta_minus_one_casimir_energy`** — computes: the Abel-regularized
  zero-point sum $\sum n\,e^{-n\varepsilon}$ extracts $\zeta(-1) = -1/12$,
  which assembles the 1D Casimir energy $E = -\pi/(24d)$ and force
  $F = -\pi/(24d^2)$ ($\hbar=c=1$) — the seed of the 3D
  $E/A = -\pi^2/(720d^3)$. *Asserts*: the extraction, the assembly.
- **`qed_photon_additivity_and_multimode_vacuum`** — computes: $|n\rangle$
  has energy $n\omega$ exactly ($n \le 5$); a multi-mode vacuum is exactly
  zero; one photon in mode $i$ has $\omega_i$; two photons in distinct modes
  add. *Asserts*: $10^{-9}$.

### 5.24d QG cosmology & black-hole thermodynamics — `qg_cosmology_validation.rs`

- **`qg_friedmann_matter_and_radiation_closed_forms`** — computes: the
  scale-factor equation $\dot a = H_0\sqrt{\Omega_m/a + \Omega_r/a^2 +
  \Omega_\Lambda a^2}$ integrated with RK4 matches the closed forms for the
  pure matter ($a \propto t^{2/3}$), pure radiation ($a \propto t^{1/2}$)
  and pure $\Lambda$ ($a \propto e^{H_0t}$) universes, and radiation
  dominates at early $a$. *Asserts*: $10^{-5}$.
- **`qg_lcdm_universe_age`** — computes: the closed-form flat $\Lambda$CDM
  age $t_0 = (2/(3H_0\sqrt{\Omega_\Lambda}))\,\mathrm{arcsinh}
  (\sqrt{\Omega_\Lambda/\Omega_m})$ at $H_0 = 67.66\,\mathrm{km/s/Mpc}$,
  $\Omega_m = 0.31$, $\Omega_\Lambda = 0.69$ is $\approx 13.8$ Gyr
  (Planck 2018: 13.787 ± 0.020), and the numerical integration reproduces
  it. *Asserts*: 0.1 Gyr; 0.5%.
- **`qg_starobinsky_efolds`** — computes: the slow-roll e-folds
  $N_e(\varphi) = \int V/V'\,d\varphi'$ for the R² scalaron potential
  $V = (1-e^{-k\varphi})^2$ ($k = \sqrt{2/3}$, $M = 1$) match the closed
  form $(3/4)(e^{k\varphi} - e^{k\varphi_{\mathrm{end}}} - k(\varphi -
  \varphi_{\mathrm{end}}))$; at $\varphi = 10$ the asymptotic
  $N_e \approx (3/4)e^{k\varphi}$ holds to $<1\%$ (at $\varphi = 6$ the
  $-k\varphi$ term is still 4.5%). *Asserts*: $10^{-6}$; 1%.
- **`qg_schwarzschild_black_hole_thermodynamics`** — computes: with the
  CODATA constants, the Smarr identity $Mc^2 = 2TS$, the Bekenstein–Hawking
  entropy $S = A/(4\ell_P^2) = 4\pi GM^2/(\hbar c)$ ($\approx 1.05\times
  10^{77}\,k_B$ for the Sun), and the saturation of the Bekenstein bound
  $S = 2\pi r_s Mc/\hbar$ — all exact identities. *Asserts*: $10^{-9}$;
  2%.

### 5.24e NS boundary layer & turbulence scales — `ns_boundary_layer_validation.rs`

- **`ns_blasius_shooting_reproduces_published_profile`** — computes: the
  Blasius similarity equation $f''' + \tfrac12 ff'' = 0$, $f(0)=f'(0)=0$,
  $f'(\infty)=1$ solved numerically (RK4 + bisection shooting on
  $f''(0)$): the published constants $f''(0) = 0.33206$, the shape factor
  $H = \delta^*/\theta = 2.5916$, the 1% thickness $\delta_{99} \approx
  4.92$, and $C_f\sqrt{Re_x} = 2f''(0) = 0.664$. *Asserts*: 0.5%–3%.
- **`ns_turbulent_length_scale_identities`** — computes: the exact
  consequences of the Kolmogorov/Taylor relations $\lambda/\eta =
  15^{1/4}\sqrt{Re_\lambda}$ and $L/\lambda = Re_\lambda/15$, consistent
  with the same $\varepsilon, \nu, u'$; the inertial-range ordering
  $L > \lambda > \eta$. *Asserts*: $10^{-9}$.

### 5.24f QED perturbative & Schwinger sector — `qed_further_validation.rs`

- **`qed_anomalous_moment_leading_schwinger_term`** — computes: the
  one-loop electron anomaly $a_e^{(1)} = \alpha/2\pi = 0.001161409733$.
  *Asserts*: $10^{-10}$.
- **`qed_anomalous_moment_two_loop_matches_codata`** — computes: the series
  $a_e = \alpha/2\pi - 0.3284789656(\alpha/\pi)^2 + 1.181234017(\alpha/\pi)^3$
  against CODATA 2018 $a_e = 0.00115965218$ (Schwinger 1948, Petermann–
  Sommerfield, Laporta–Remiddi). *Asserts*: $10^{-9}$ — including the sign
  and magnitude of each individual term.
- **`qed_schwinger_critical_field_pin`** — computes: $E_c = m_e^2c^3/(e\hbar)$
  = $1.323\times10^{18}$ V/m. *Asserts*: 0.1%.
- **`qed_schwinger_rate_exponential_barrier`** — computes: the vacuum-pair
  suppression $\Gamma \propto (eE)^2\exp(-\pi E_c/E)$; the log of the rate is
  exactly linear in $E_c/E$ with slope $-\pi$, so a factor-2 field increase
  buys $e^{-5\pi} = 1.5\times10^{-7}$ in rate, while the polynomial
  prefactor contributes only $\times4$. *Asserts*: $10^{-6}$.
- **`qed_fine_structure_runs_upwards_leptonic`** — computes: the 1-loop
  leptonic screening $\alpha(M_Z) = \alpha/(1 - (\alpha/3\pi)\sum_Q
  \ln(M_Z/m_Q))$, giving $1/\alpha_{\rm lept}(M_Z) \approx 134.6$ — the
  charge *increases* with energy (vacuum polarization screens), and the gap
  to the full $1/\alpha(M_Z) = 128.9$ is the hadronic $\Delta\alpha_{\rm had}
  \approx 0.028$, which is *not* included here and is asserted to be
  missing. *Asserts*: $\pm0.3$ on the leptonic value, $\alpha(M_Z) >
  \alpha(0)$.
- **`qed_bessel_series_matches_known_values`** — pins the shared Bessel
  implementation (also used by the QYM lattice suite) to known $I_n(1)$
  values. *Asserts*: $10^{-8}$.
- **`qed_casimir_three_dimensional_coefficient`** — computes: the famous
  3D Casimir energy density $E/A = -\pi^2\hbar c/(720d^3)$: the exact
  coefficient $\pi^2/720$, the $d^{-3}$ law (doubling $d$ ÷8), the numeric
  pin $E/A(1\,\mu\mathrm{m}) = -4.334\times10^{-10}$ J/m², and the pressure
  relation $F/A = -d(E/A)/dd = -\pi^2\hbar c/(240d^4)$ with
  $F\cdot d = 3(E/A)$ exactly. (The 1D seed $E = -\pi/(24d)$ is pinned in
  §5.24c.) *Asserts*: $10^{-9}$–$10^{-12}$.
- **`qed_fine_structure_from_si_constants`** — computes: the metrology
  triangle $\alpha = e^2/(4\pi\varepsilon_0\hbar c)$ from the SI-defining
  constants — it returns the CODATA fine-structure constant.
  *Asserts*: $10^{-8}$ relative.
- **`qed_rydberg_and_hydrogen_ionization`** — computes: $R_\infty =
  \alpha^2m_ec/(2h) = 1.09737\times10^7$ m⁻¹, the hydrogen ionization
  $E = \tfrac12\alpha^2m_ec^2 = 13.6057$ eV, and the Compton wavelength
  $\lambda_C = h/(m_ec) = 2.42631\times10^{-12}$ m — three atomic scales
  built from the same four constants. *Asserts*: $10^{-7}$–$10^{-8}$.

### 5.24g QG general relativity — `qg_general_relativity_validation.rs`

- **`qg_mercury_perihelion_precession_43_arcsec`** — computes: the GR excess
  $\Delta\varphi = 6\pi GM/(c^2a(1-e^2))$ per orbit = 0.1036″, × 415.2
  orbits/century = **43.0″/century** — the historical anchor of GR.
  *Asserts*: $\pm0.5$″/century, per-orbit $\pm0.001$″.
- **`qg_hawking_temperature_solar_mass`** — computes: $T_H =
  \hbar c^3/(8\pi GMk_B) = 6.17\times10^{-8}$ K for $M_\odot$, and the exact
  $T_H \propto 1/M$ scaling (a 10 $M_\odot$ hole is 10× colder).
  *Asserts*: 1% + $10^{-12}$.
- **`qg_black_hole_evaporation_time_and_m3_scaling`** — computes:
  $\tau = 5120\pi G^2M^3/(\hbar c^4) = 6.6\times10^{74}$ s for $M_\odot$
  (≈ $2\times10^{67}$ yr), and the exact $\tau \propto M^3$ scaling
  (doubling $M$ ×8 the lifetime). *Asserts*: 5% + $10^{-9}$.
- **`qg_gravitational_redshift_weak_field_limit`** — computes: exact
  $z = 1/\sqrt{1-r_s/r} - 1$ vs the weak-field $z \approx r_s/(2r)$; the
  limit sharpens from 0.75% at $100\,r_s$ to $7.5\times10^{-5}$ at
  $10^4\,r_s$, and $z \to \infty$ as $r \to r_s^+$. *Asserts*: relative
  $10^{-2}$ / $10^{-4}$ at the two stations.
- **`qg_geodesic_constants_photon_sphere_and_isco`** — computes: $r_{\rm ph}
  = 1.5\,r_s$, ISCO $= 3\,r_s$, $r_s(M_\odot) = 2953$ m. *Asserts*: $10^{-12}$
  on the ratios.
- **`qg_gw_chirp_mass_and_f11_3_scaling`** — computes: the GW150914 chirp
  mass $M_c = (m_1m_2)^{3/5}/(m_1+m_2)^{1/5} = 28.1\,M_\odot$ (LIGO: 28.3)
  and the quadrupole scalings $\dot f \propto M_c^{5/3}f^{11/3}$:
  $2^{11/3} = 12.70$, $2^{5/3} = 3.17$. *Asserts*: $\pm0.5\,M_\odot$,
  $10^{-3}$.
- **`qg_peters_merger_time_a4_scaling`** — computes: the gravitational-
  radiation inspiral time $t_{\rm merge} = \frac{5}{256}\frac{c^5}{G^3}
  \frac{a^4}{m_1m_2(m_1+m_2)}$ (Peters 1964): ≈ $3.0\times10^{17}$ s
  (≈ 9.5 Gyr) for two 1.4 $M_\odot$ stars at $3\times10^9$ m — the same
  ballpark as the Hulse–Taylor inspiral — with the exact $a^4$ law
  (doubling $a$ ×16) and $t \propto 1/(m_1m_2(m_1+m_2))$ (×8 for the
  0.7+0.7 pair). *Asserts*: factor 2 on the value, $10^{-9}$ on the scalings.
- **`qg_shapiro_delay_sun_graze`** — computes: the radar-echo excess delay
  $\Delta t = (4GM/c^3)\ln(4r_1r_2/b^2)$ for a signal grazing the Sun's limb
  ($r_1 = r_2 = 1$ AU, $b = R_\odot$) = **239 µs** — the non-Newtonian time-
  metric test — plus the logarithmic growth $\Delta t(b/2)/\Delta t(b) = 1 +
  \ln 4/\ln(4r^2/b^2)$. *Asserts*: $\pm30$ µs, $10^{-9}$.
- **`qg_bekenstein_bound_saturated_by_schwarzschild`** — computes: for a
  Schwarzschild hole with $R = r_s$, the bound $S \le 2\pi k_BRE/(\hbar c)$
  equals $S_{BH} = 4\pi k_BGM^2/(\hbar c)$ **exactly** — saturation — while
  an ordinary 1 m, 1 kg system sits $>10^{20}$ below it. *Asserts*:
  $10^{-12}$.

### 5.24h QYM lattice gauge — strong coupling & running coupling — `qym_lattice_validation.rs`

- **`qym_polyakov_loop_confinement_order_parameter`** — computes: the
  single-site Polyakov loop $\langle L\rangle = I_1(\beta)/I_0(\beta)$ — the
  deconfinement order parameter: it vanishes at strong coupling (confined,
  center symmetry unbroken) and → 1 at weak coupling (deconfined), with the
  series $\langle L\rangle = \beta/2 - \beta^3/16 + \beta^5/96 -
  11\beta^7/6144$ and the $1 - \langle L\rangle \approx 1/\beta$ gap at
  large $\beta$. *Asserts*: $10^{-3}$ (series), qualitative (limits).
- **`qym_plaquette_bessel_closed_form_matches_series`** — computes: the exact
  single-plaquette SU(2) expectation $\langle P\rangle = I_2(\beta)/I_1(\beta)$
  (Haar-measure integral, modified Bessel functions) vs the strong-coupling
  series $\beta/4 - \beta^3/96 + \beta^5/1536 - \beta^7/24576$, plus the
  limits $\langle P\rangle \to 0$ as $\beta \to 0$ and $1 - \langle P\rangle
  \approx 3/(2\beta)$ as $\beta \to \infty$. *Asserts*: $10^{-6}$–$10^{-5}$
  on the series, 5% on the asymptotic.
- **`qym_two_dimensional_wilson_area_law_exact`** — computes: in 2D the
  Wilson loop is exactly $W(R,T) = \langle P\rangle^{RT} =
  \exp(-\sigma\,A)$ with $\sigma = -\ln\langle P\rangle$ — the *area* law,
  exactly, at every coupling: $W(2,2) = W(1,4)$ (same area, different
  shapes), and the perimeter guess $\langle P\rangle^8$ is smaller by the
  factor $\langle P\rangle^4 = 2.3\times10^{-4}$. *Asserts*: $10^{-12}$.
- **`qym_creutz_ratio_extracts_string_tension_exactly_in_2d`** — computes:
  the Creutz ratio $\chi(I,J) = -\ln[W(I,J)W(I-1,J-1)/(W(I,J-1)W(I-1,J))]$:
  under the exact 2D area law the corner loops cancel and it returns
  *exactly* $\sigma$ for every shape $(I,J)$ — the estimator is unbiased in
  the area-law regime, at every coupling. *Asserts*: $10^{-12}$.
- **`qym_string_tension_leading_strong_coupling`** — computes: the 4D
  strong-coupling string tension $\sigma a^2 = \ln(4/\beta)$ vs the 2D exact
  $-\ln\langle P\rangle$ (0.5% agreement at $\beta = 0.5$; the difference is
  the O($\beta^2$) term), and the vanishing $\sigma \to 0$ in the weak-
  coupling limit via $\sigma \approx 3/(2\beta)$. *Asserts*: 2% / 5%.
- **`qym_asymptotic_freedom_one_loop_running`** — computes: the 1-loop
  running $1/\alpha_s(Q_2) - 1/\alpha_s(Q_1) = (\beta_0/2\pi)\ln(Q_2/Q_1)$,
  $\beta_0 = 11 - 2n_f/3 = 23/3$ at $n_f = 5$, from the PDG
  $\alpha_s(M_Z) = 0.1179$: $\alpha_s(1\,\mathrm{TeV}) < \alpha_s(M_Z)$
  (asymptotic freedom in the UV) while $\alpha_s(1\,\mathrm{GeV}) > 0.3$
  (the infrared confinement side), with an exact round-trip back to
  $\alpha_s(M_Z)$. *Asserts*: $10^{-9}$ round-trip.
- **`qym_glueball_spectrum_literature_pin`** — pins the lattice
  $m_G/\sqrt\sigma = 3.55$ (lightest $0^{++}$ glueball) and cross-checks
  against the strong-coupling $\sigma$: $m_G a \approx 5.13$ at $\beta =
  0.5$. *Asserts*: $\pm0.3$ (literature anchor, not derived).

### 5.24i NG Newtonian gravity — `ng_newtonian_validation.rs`

- **`ng_kepler_third_law_earth_year`** — computes: $T^2 = 4\pi^2a^3/(GM_\odot)$
  for Earth → 365.25 d. *Asserts*: 0.1%.
- **`ng_kepler_t2_proportional_a3`** — computes: $T^2/a^3 = 4\pi^2/\mu$ with
  the *two-body* $\mu = G(M_\odot + m_{\rm planet})$ for Mercury, Earth and
  Jupiter — the planet-mass correction is visible at the $10^{-3}$ level for
  Jupiter; the residual is element precision, not the law. *Asserts*: 0.5%.
- **`ng_virial_theorem_circular_orbit`** — computes: circular orbit with
  $v = \sqrt{GM/r}$ has $2\langle T\rangle + \langle V\rangle = 0$ exactly.
  *Asserts*: $10^{-12}$.
- **`ng_shell_theorem_inside_and_outside`** — computes: the field of a
  uniform shell by 4000-point quadrature — exactly zero inside, $GM/r^2$
  outside. *Asserts*: $10^{-12}$ inside, $10^{-4}$ outside.
- **`ng_escape_velocity_earth`** — computes: $v_{\rm esc} = \sqrt{2GM/R}$ =
  11.19 km/s and the exact $\sqrt2$ ratio vs circular speed. *Asserts*:
  0.5% + $10^{-12}$.
- **`ng_uniform_sphere_binding_energy`** — computes: $U = 3GM^2/(5R)$ =
  $2.24\times10^{32}$ J for Earth. *Asserts*: 1%.
- **`ng_gravitational_parameter_plumbing`** — anchors the constants plumbing:
  $GM_\odot = 1.32712\times10^{20}$ m³/s² and $GM_\oplus =
  3.98600\times10^{14}$ m³/s² (IAU values), plus the Kepler consistency
  $GM_\odot = 4\pi^2\mathrm{AU}^3/\mathrm{yr}_{\rm sid}^2$.
  *Asserts*: $10^{-3}$.
- **`ng_roche_limit_earth_moon`** — computes: $d_{\rm Roche} = R\,(2\rho_p/
  \rho_s)^{1/3}$ = 9,500 km for the Moon about Earth (≈ 1.5 Earth radii —
  the reason Saturn's rings, inside Saturn's Roche limit, never coalesced),
  and the $d \propto (\rho_p/\rho_s)^{1/3}$ scaling. *Asserts*: 3% +
  $10^{-12}$.
- **`ng_tidal_acceleration_sun_moon_ratio`** — computes: the tidal
  acceleration $\propto M/d^3$ ratio Sun/Moon = **0.46** at Earth's surface
  — the Sun's tide is weaker than the Moon's despite its $2.7\times10^7$
  larger mass — and the exact $1/d^3$ law (doubling $d$ ÷8 the tide).
  *Asserts*: $\pm0.02$, $10^{-12}$.
- **`ng_hill_sphere_earth_moon`** — computes: $r_H = a(\frac{m}{3M})^{1/3}$ =
  61,500 km for the Moon about Earth (1/6.25 of the Earth–Moon distance),
  and the $r_H \propto a(m/M)^{1/3}$ scaling. *Asserts*: 3% + $10^{-12}$.
- **`ng_leapfrog_two_body_energy_conservation`** — computes: symplectic
  leapfrog integration of two equal masses on circular orbits for 100
  periods (1000 steps/orbit) — the energy stays within a bounded oscillation
  ($|\Delta E/E| < 10^{-4}$), the signature of a symplectic integrator.
  *Note*: the circular speed for separation $2r$ is $\sqrt{Gm/4r}$, not
  $\sqrt{Gm/2r}$ (that is the escape speed and gives exactly zero total
  energy). *Asserts*: $10^{-4}$.

### 5.24j NS exact laminar & dissipation-scale identities — `ns_further_validation.rs`

- **`ns_hagen_poiseuille_flow_rate_and_r4`** — computes: $Q =
  \pi R^4\Delta P/(8\mu L)$ with the exact $R^4$ scaling (doubling $R$ ×16
  the flow) and $v_{\max} = 2\bar v$. *Asserts*: 0.1% + $10^{-9}$.
- **`ns_kolmogorov_minus_5_3_spectrum`** — computes: $E(k) \propto k^{-5/3}$
  ⇒ $E(2k)/E(k) = 2^{-5/3} = 0.31498$ and $E(10k)/E(k) = 10^{-5/3} =
  0.0215$. *Asserts*: $10^{-4}$–$10^{-12}$.
- **`ns_kolmogorov_dissipation_scale_re_eta_identity`** — computes: at the
  dissipation scale $\eta = (\nu^3/\varepsilon)^{1/4}$, $u_\eta =
  (\nu\varepsilon)^{1/4}$, and $Re_\eta = u_\eta\eta/\nu \equiv 1$ exactly;
  consistent with the Taylor-scale identity $\varepsilon = 15\nu u'^2/
  \lambda^2$. *Asserts*: $10^{-12}$ / $10^{-9}$.
- **`ns_stokes_drag_linearity`** — computes: $F = 6\pi\mu Rv$ with exact
  linearity in $R$ and $v$, plus the terminal velocity
  $v_t = 2(\rho-\rho_f)gR^2/(9\mu)$. *Asserts*: 0.1% + $10^{-12}$, 0.5%.
- **`ns_reynolds_number_and_transition_pin`** — computes: $Re = \rho UD/\mu$
  dimensionless (water at 20 °C, 2 cm pipe, 0.1 m/s → $Re = 1992$), the
  exact scale invariance ($U,D \times2$, $\mu \times4$ leaves $Re$ fixed),
  and the laminar→turbulent transition pin $Re_c \approx 2300$ (Reynolds
  1883). *Asserts*: $10^{-3}$–$10^{-12}$.
- **`ns_blasius_pipe_friction_correlation`** — computes: the turbulent pipe
  friction factor $f = 0.3164\,Re^{-1/4}$ (Blasius correlation,
  $4\cdot10^3 < Re < 10^5$): $f(10^5) = 0.0178$, the exact $Re^{-1/4}$ law
  (×16 in $Re$ halves $f$), the wall shear $\tau_w = \tfrac12\rho fU^2$, and
  the Darcy–Weisbach head loss $\Delta P = f(L/D)\tfrac12\rho U^2$.
  *Asserts*: $10^{-3}$–$10^{-9}$.
- **`ns_couette_linear_profile_and_constant_shear`** — computes: plane
  Couette flow $u(y) = Uy/H$ — an exact NS solution — with no-slip at both
  plates and constant shear $\tau = \mu U/H$. *Asserts*: $10^{-12}$ on the
  profile, $10^{-3}$ on $\tau$.
- **`ns_bernoulli_venturi_pressure_drop`** — computes: continuity $A_1v_1 =
  A_2v_2$ + Bernoulli: a constriction to $A_2 = A_1/4$ gives $v_2 = 4v_1$
  and $\Delta P = \tfrac12\rho(v_2^2-v_1^2) = 7.5\rho v_1^2$, with full
  pressure recovery on re-expansion. *Asserts*: $10^{-12}$.
- **`ns_lamb_oseen_vortex_structure`** — computes: the diffusing vortex
  $\omega(r,t) = \frac{\Gamma}{4\pi\nu t}e^{-r^2/4\nu t}$: the total
  circulation $\int\omega\,dA = \Gamma$ exactly (Gaussian integral,
  conserved), the $\omega(0) \propto 1/t$ decay, and the irrotational tail
  $v_\theta \to \Gamma/(2\pi r)$. *Asserts*: $10^{-3}$–$10^{-12}$.
- **`ns_strouhal_vortex_shedding_pin`** — pins $St = fD/U = 0.2$ (cylinder
  shedding at $Re \approx 10^3$, per the §5.25 ledger's "Strouhal" content)
  and the definition $f = St\,U/D$ with $St$ dimensionless. *Asserts*:
  $10^{-12}$.
- **`ns_blasius_thicknesses_and_shape_factor`** — computes: $\delta^*/x =
  1.7208/\sqrt{Re_x}$, $\theta/x = 0.664/\sqrt{Re_x}$, $H = \delta^*/\theta =
  2.5916$, the $\sqrt x$ growth $\delta^*(4x)/\delta^*(x) = 2$, and the
  $\delta_{99}/\delta^* = 2.86$ consistency with the Blasius suite.
  *Asserts*: $10^{-3}$–$10^{-9}$.

### 5.24k Nuclear/astrophysical QED — `qed_nuclear_astro.rs`

Nuclear and astrophysical QED scenarios validated through the Fock/SIRK
machinery, using the same Hashimoto/SIRK solver as every other suite:

- **`qed_fermi_muon_decay_rate`** — the tree-level muon decay width
  $\Gamma = G_F^2 m_\mu^5 / (192\pi^3)$ from the four-fermion Fermi
  effective Hamiltonian.  Computes the lifetime $\tau = \hbar/\Gamma$
  and compares to the PDG value $2.197 \times 10^{-6}$ s.
  *Asserts*: relative error < 1%.
- **`qed_gzk_cutoff_threshold`** — the Greisen–Zatsepin–Kuzmin cosmic-ray
  cutoff: the threshold proton energy for $p + \gamma_{\text{CMB}} \to
  \Delta^+$ resonance, $E_p = (m_\Delta^2 - m_p^2)/(2\omega_{\text{CMB}})$.
  *Asserts*: $E_{\text{th}} \approx 6.8 \times 10^{17}$ eV (680 PeV).
- **`qed_lamb_shift_leading_order`** — the leading-order Lamb shift
  (2S$_{1/2}$–2P$_{1/2}$) of hydrogen: the $\alpha^5 m_e$ scale
  multiplied by $\ln(1/\alpha^2)$.  *Asserts*: estimate within 30% of PDG
  1.058 GHz (subleading terms account for the gap).
- **`qed_schwinger_anomalous_moment`** — the one-loop Schwinger term
  $a_e = \alpha/(2\pi) \approx 0.00116$, compared to CODATA
  0.0011597.  *Asserts*: relative error < 1%.
- **`qed_positronium_hyperfine`** — the leading-order QED prediction for
  the 1S triplet–singlet splitting $\Delta\nu = (7/6)\alpha^4 m_e c^2/h
  \approx 408.8$ GHz.  *Asserts*: exact match to the LO formula
  (higher-order corrections reduce this to the experimental 203 GHz).
- **`qed_schwinger_critical_field`** — the pair-production suppression
  factor $e^{-\pi} \approx 0.0432$ at the Schwinger critical field.

### 5.24l SU(2) confined-lattice gauge dynamics — `su2_gauge_dynamics.rs`

SU(2) Yang–Mills on the lattice, exercised through the
`yang_mills_lattice(2, g, 1)` Hamiltonian:

- **`su2_electric_flux_quantization`** — the even→odd energy gap is
  $\approx g^2/2$ at strong coupling.  Tested at $g \in \{2, 3, 4\}$;
  *asserts* relative error < 5%.
- **`su2_string_tension_scales`** — the log-log slope of the gap vs $g$
  is $\approx 2$ (the gap scales as $g^2$).  *Asserts*: slope in $[1.5,
  2.5]$ for each consecutive pair in $g \in \{2, 3, 4, 5\}$.
- **`su2_plaquette_expectation`** — the strong-coupling series
  $\langle P \rangle \approx \beta/2 - \beta^3/16 + 5\beta^5/768$
  with $\beta = 2/g^2$.  *Asserts*: $\langle P \rangle \in (0, 0.1)$ at
  $g = 4$, and the plaquette energy $1 - \langle P \rangle > 0.9$.
- **`su2_sector_purity`** — the maximal mutual overlap of even/odd Krylov
  chains vanishes (lattice parity is an exact symmetry).  *Asserts*:
  overlap < $10^{-6}$.
- **`su2_glueball_mass_ratio`** — the excitation gap (glueball mass)
  divided by $\sqrt{\sigma}$ (string tension from the even→odd gap).
  *Asserts*: ratio > 0.1 (positive, finite).
- **`su2_confinement_across_couplings`** — the even→odd gap is positive
  for all $g \in \{1.5, 2, 3, 4, 6\}$.  *Asserts*: gap > 0.

### 5.24m Quantum foam / Newtonian gravity overlap — `quantum_foam_ng_overlap.rs`

Graviton Fock-space tests at the boundary of quantum gravity and Newtonian
gravity:

- **`qf_ng_graviton_number_classical_limit`** — the graviton number
  eigenstates have energy $n\omega$: verified for $n = 0, 1, 2$.
  *Asserts*: $|E(n) - n\omega| < 0.1$.
- **`qf_ng_graviton_zero_point_energy`** — the Casimir energy difference
  $\Delta E = (\pi/2)(1/L - 1/L')$ for two graviton cavities.
  *Asserts*: $\Delta E > 0$.
- **`qf_ng_bohr_frequency_gravitational_orbit`** — the Bohr-frequency
  ratio $\omega_{21}/\omega_{32} = 27/5$ for gravitational orbits.
  *Asserts*: relative error < 1%.
- **`qf_ng_foam_fluctuation_scale`** — metric fluctuations $\delta g \sim
  (\ell_P/L)^2$: negligible at $L = 1$ m, $O(1)$ at $L = \ell_P$.
  *Asserts*: $\delta g(1\text{m}) < 10^{-50}$.
- **`qf_ng_graviton_number_conservation`** — the free graviton Hamiltonian
  is diagonal in the number basis; Ritz values include $n\omega$ for
  $n = 0, \ldots, 4$.  *Asserts*: each $n\omega$ is a Ritz value.
- **`qf_ng_coherent_state_classical_limit`** — the $|10\rangle$ Fock state
  has energy $10\omega$.  *Asserts*: $|E - 10\omega| < 0.1$.

### 5.24n Certified mass-gap suite — `qcd_mass_gap_certified.rs`

The proof-carrying half of the gauge-fixed formalization (replaces the
lattice-era Richardson/certified-table tests, superseded):

- **`qcd_mass_gap_certified_enclosure`** — computes: the two R-sector SIRK
  solves (m = 12) at $g=1$ assemble into the certified T6 interval
  $[\theta^o_0-\theta^e_0 \pm (\delta^o+\delta^e)] = [-8.009, 6.866]$,
  which encloses the exact truncated gap $E_1-E_0 = 0.0912$; the measured
  gap lies inside its own certified interval. *Asserts*: enclosure;
  positivity of the exact gap.
- **`qcd_mass_gap_certified_window_contains_exact_gap`** — computes: the
  certified window contains the exact truncated gap across the truncation
  family (N ≤ 6 and N ≤ 8) for $g \in \{1, 2\}$ (0.0911 / 0.0912 at $g=1$;
  0.850 / 1.244 at $g=2$). *Asserts*: containment at every (g, N) pair.
- **`qcd_mass_gap_certificate_ndjson`** — computes:
  `emit_gap_certificate_ndjson` emits three valid JSON lines (two Ritz
  certificates carrying $\theta$ and $\delta$, one assembly carrying gap and
  $\delta$) — the nanoda re-verification seam.

### 5.24o Direct SIRK drive of the project's Hamiltonians — `sirk_hamiltonian_drive.rs`

The purest form of the program's numerics: every test runs the
Hashimoto/SIRK solver directly against a Hamiltonian from
`nested_fock_algebra::models` and checks that the *solver output itself* is
physically consistent.  No external formulas, no analytic shortcuts — just
SIRK on the gauge-fixed Hamiltonians.  This is the suite that validates the
engine end-to-end across all four systems.

- **`sirk_drive_ym_lattice_gap`** — SIRK ground/odd-sector energies on
  `yang_mills_lattice(2, g, 1)`; the even→odd gap satisfies the strong-
  coupling law $g^2/2$ to within 5%.  *Asserts*: $|\Delta - g^2/2| < 0.05\,\cdot g^2/2$.
- **`sirk_drive_ym_krylov_convergence`** — the same gap converges
  monotonically as the Krylov dimension $m \in \{2,3,4,5\}$ grows, and the
  $m=5$ gap agrees with $g^2/2$ to 2%.  *Asserts*: $\Delta(m)$ non-
  increasing; final gap within 2%.
- **`sirk_drive_qed_multimode`** — SIRK on `qed_free_photon([0.5,1,2,4])`
  resolves each single-photon excitation exactly at its $\omega$.  *Asserts*:
  vacuum $\approx 0$; $|E_k - \omega_k| < 0.1$ for every mode.
- **`sirk_drive_qed_cavity`** — SIRK on the `qed_cavity_frequencies`
  modes resolves all $n_{\max} = 4$ cavity modes at their analytic
  frequencies.  *Asserts*: $|E_k - \omega_k| < 0.1$.
- **`sirk_drive_qed_pair_production`** — SIRK on the `qed_pair_production`
  Hamiltonian yields a finite vacuum energy (non-normal-ordered structure).
  *Asserts*: $E_0$ finite.
- **`sirk_drive_qg_graviton`** — SIRK on `qg_free_graviton([0.5,1,2])`
  resolves each helicity mode at $\omega$.  *Asserts*: vacuum $\approx 0$;
  $|E_k - \omega_k| < 0.1$.
- **`sirk_drive_qg_3d`** — SIRK on the 3D gauge-fixed `gravity_hamiltonian`
  returns a finite vacuum (the cosmological-constant term).  *Asserts*:
  $E_0$ finite.
- **`sirk_drive_ng_chain`** — SIRK on `harmonic_chain(3, 1.5)` resolves the
  first excitation at $\omega$.  *Asserts*: vacuum $\approx 0$; $|E_1 - \omega| < 0.1$.
- **`sirk_drive_ns`** — SIRK on `navier_stokes_hamiltonian(0.01)` returns a
  finite outer-enclosed vacuum (viscous damping terms).  *Asserts*: outer
  vacuum $E_0=0$ for the final enclosure and finite inner-sector diagnostics.
- **`sirk_drive_ym_residual_decay`** — the leading Ritz residual on the YM
  lattice strictly decreases as $m$ grows — the certified-interval
  convergence the gap theorem depends on.  *Asserts*: $r_0(m)$ non-increasing.
- **`sirk_drive_qg_ng_crosscheck`** — cross-system coherence: SIRK on the
  graviton Hamiltonian and on the harmonic chain at the same $\omega$ gives
  the *same* gap, connecting QG and NG at the free level.  *Asserts*:
  $|\Delta_{QG} - \Delta_{NG}| < 0.1$.

This suite is the definition of done for the engine: if any of these fail,
the SIRK implementation on that system's Hamiltonian is broken, regardless
of what the analytics claim.

### 5.24p Kerr photon blockade — `qed_kerr_photon_blockade.rs`

Cavity-QED nonlinear optics on the new builder `qed_kerr_cavity(ω, χ)`
(`H = ωa†a + χa†a†aa`), the canonical single-photon-source model. Because
`[H, N] = 0` each photon-number sector is 1-dimensional, so every SIRK
solve is exact and the assertions are to solver precision:

- **`qed_kerr_anharmonic_ladder_exact`** — the Ritz value in each sector
  $|n\rangle$ lands on the closed-form anharmonic ladder
  $E_n = \omega n + \chi n(n-1)$ ($n = 0..4$, three $(\omega, \chi)$
  settings).  *Asserts*: $|E - E_n| < 10^{-7}$.
- **`qed_kerr_photon_blockade_detuning`** — the transition energies are
  $\Delta_1 = E_1 - E_0 = \omega$, $\Delta_2 = E_2 - E_1 = \omega + 2\chi$,
  $\Delta_3 = E_3 - E_2 = \omega + 4\chi$: the first photon is resonant
  while the second is detuned by $2\chi$, the **photon blockade**
  (Imamoğlu–Schmidt–Woods–Deutsch 1997).  *Asserts*: each spacing to
  $10^{-7}$; the two-photon detuning $\Delta_2 - \Delta_1 = 2\chi$;
  resolvable ($> 10^{-3}$).
- **`qed_kerr_photon_number_conservation`** — real-time restarted-SIRK
  evolution of $|2\rangle$ conserves the norm, $\langle N \rangle = 2$
  (the $\chi$ term preserves photon number) and the energy
  $E_2 = 2\omega + 2\chi$ at every time.  *Asserts*: all to $10^{-7}$.
- **`qed_kerr_chi_zero_is_free_photon_sector`** — at $\chi = 0$ the Kerr
  cavity is the Cadabra-derived abelian QED sector itself: both builders
  (`qed_kerr_cavity(ω, 0)` and `qed_free_photon(ω)`, the U(1) reduction of
  $H_{\rm final} = \tfrac12\pi^2 + \tfrac12 B^2$) give the identical photon
  ladder $\{n\omega\}$.  *Asserts*: $|E_{\rm Kerr} - E_{\rm photon}| < 10^{-9}$.

### 5.24p′ Driven sector & blockade statistics — `qed_blockade_statistics.rs`

The photon *statistics* of the driven QED gauge-fixed sector, extracted from
**the SIRK ground eigenvector** (the lowest Ritz pair of the projected
Hamiltonian, reconstructed via `h_proj` + `reconstruct`):

- **`qed_static_charge_ground_is_coherent_poissonian`** — the abelian
  gauge-fixed QED sector with a static charge
  (`qed_static_charge_interaction`, `H = ωN + g(B†+B)` from
  `docs/yang_mills_hamiltonian.cdb`) has the exactly-solvable coherent
  ground state $|-g/\omega\rangle$: $\langle N \rangle = \mathrm{Var}(N) =
  (g/\omega)^2$ and the Fano factor is exactly 1 (Poissonian), across three
  couplings.  *Asserts*: Fano $= 1 \pm 10^{-4}$.
- **`qed_kerr_blockade_sub_poissonian_antibunched`** — the same sector with
  the cavity-Kerr nonlinearity (the driven-Kerr builder
  `qed_kerr_cavity_driven`): at $\chi > 0$ the drive at the first transition
  cannot absorb a second photon, so the ground state is sub-Poissonian and
  antibunched — Fano $< 1$, $g^{(2)}(0) = \langle N(N-1)\rangle/\langle N\rangle^2 < 1$
  — and the Fano factor decreases monotonically as $\chi$ grows (stronger
  blockade).  *Asserts*: Fano $< 1 - 10^{-3}$, monotone in $\chi$.
- **`qed_kerr_statistics_return_to_poissonian_as_chi_vanishes`** — as
  $\chi \to 0$ the driven-Kerr ground returns monotonically to the coherent
  state (Fano $\to 1$): continuity with the abelian gauge-fixed sector.
  *Asserts*: $|\mathrm{Fano}(\chi = 10^{-6}) - 1| < 10^{-3}$; monotone.

### 5.24q Hong–Ou–Mandel bunching — `qed_hong_ou_mandel.rs`

Two-photon quantum-optics statistics on the existing `oscillator_beamsplitter`
with $\omega = 0$: the beamsplitter $H = J(a^\dagger_0 a_1 + a^\dagger_1 a_0)$
is twice the Schwinger SU(2) generator $J_x$, so evolution by $t$ rotates by
$\theta = 2Jt$; $\theta = \pi/2$ is the ideal 50:50 beamsplitter. All
predictions are exact quantum-optics results measured in photonics
laboratories:

- **`qed_hong_ou_mandel_bunching`** — two indistinguishable photons in the
  coincidence state $|1,1\rangle$ exit a 50:50 beamsplitter *bunched*:
  the spin-1 rotation has $d^1_{00}(\pi/2) = 0$, so
  $|1,1\rangle \to (|2,0\rangle - |0,2\rangle)/\sqrt2$.  *Asserts*:
  coincidence $P_{11} < 10^{-5}$ (the HOM dip), $P_{20} = P_{02} = \frac12$
  to $10^{-5}$, norm conserved.
- **`qed_beamsplitter_balanced_splitting`** — a single photon $|1,0\rangle$
  splits evenly at $\theta = \pi/2$: $P_{10} = P_{01} = \frac12$
  (spin-$\frac12$ rotation) and $\langle N_1 \rangle = \frac12$.  *Asserts*:
  all to $10^{-5}$.
- **`qed_beamsplitter_unitarity_energy`** — the $|1,1\rangle$ Krylov space is
  the *symmetric* $N = 2$ subspace $\mathrm{span}\{|1,1\rangle,
  (|2,0\rangle{+}|0,2\rangle)/\sqrt2\}$ where
  $H = \begin{smallmatrix}0 & 2J\\ 2J & 0\end{smallmatrix}$: exact spectrum
  $\{-2J, +2J\}$ reproduced by SIRK, and evolution along the full rotation
  conserves the norm and the energy $\langle H \rangle = 0$.  *Asserts*:
  spectrum to $10^{-8}$, norm/energy to $10^{-7}$.
- **`qed_hom_coincidence_curve_cos2_theta`** — the full coincidence curve of
  the HOM interferometer: $P_{11}(\theta) = |d^1_{00}(\theta)|^2 =
  \cos^2(\theta)$ at $\theta \in \{0, \tfrac\pi4, \tfrac\pi3, \tfrac\pi2,
  \tfrac{2\pi}3, \tfrac{3\pi}4, \pi\}$ — full coincidence at 0 and $\pi$,
  the dip to zero at the 50:50 point, half-way in between.  *Asserts*:
  $|P_{11} - \cos^2\theta| < 10^{-5}$ at every angle.
- **`qed_hom_bunching_from_abelian_gauge_fixed_hopping`** — computes: the
  same HOM physics predicted from the **Cadabra-derived abelian gauge-fixed
  Hamiltonian** instead of the separate beamsplitter builder: the
  beamsplitter generator is identified INSIDE `qcd_ym_hamiltonian(0)` as
  the number-conserving hopping sector of the $B^2$ cross-term
  ($-(a^\dagger_0 a_1 + a^\dagger_1 a_0)$, coefficient $-1$ per direction,
  asserted by filtering the builder's terms), and SIRK–Hashimoto on that
  sector (one solve + `time_evolve`, $m=10$) reproduces the full
  $P_{11}(\theta) = \cos^2\theta$ curve, the dip with
  $P_{20} = P_{02} = \tfrac12$, and norm conservation.  *Asserts*:
  hopping coefficients $-1$ to $10^{-9}$; $|P_{11} - \cos^2\theta| < 10^{-8}$
  at every angle; $P_{20}, P_{02} = \tfrac12 \pm 10^{-8}$.

The beamsplitter hopping is the number-conserving cross-coupling of the
abelian gauge-fixed field strength $\tfrac12 B^2 = \tfrac12(A_0-A_1)^2$
($-A_0A_1 \supset a^\dagger_0 a_1 + a^\dagger_1 a_0$); the *full* gauge-fixed
$B^2$ sector — including the pair terms — is verified directly on the
Cadabra-derived builder in §5.9.

### 5.24r TEGR graviton polarization — `qg_tegr_helicity.rs`

The graviton sector of the TEGR 3D gauge-fixed Hamiltonian
(`docs/qg_gauge_fixed_hamiltonian.cdb`, book.tex line 8190), realized by
`qg_tegr_hamiltonian`: each mode carries the normal-ordered 𝒮-sector kinetic
$c(B^\dagger B - \tfrac12(B^{\dagger 2}+B^2))$ with $c = \tfrac{1}{16}$, which
is $c(P^2 - \tfrac12)$ in quadratures — a pure momentum-squared operator whose
spectrum is the half-line $[-\tfrac{1}{32}, \infty)$ (bounded below,
continuous): the finite shadow of the essential self-adjointness of the
densitized d'Alembertian (Strichartz 1973). Each mode is one polarization
direction of the tetrad momentum; the two modes of `qg_tegr_hamiltonian(2)`
are the two transverse polarizations of the graviton in the linearized
theory (the helicity-±2 pair):

- **`qg_tegr_polarizations_degenerate`** — the two polarizations carry
  IDENTICAL 𝒮-kinetics, so the SIRK spectrum from $|1_0\rangle$, from
  $|1_1\rangle$ and from the symmetric one-quantum superposition coincide to
  solver precision — the helicity degeneracy of the graviton (one $|k|$, one
  frequency for both helicities).  *Asserts*: spectra identical to $10^{-6}$.
- **`qg_tegr_kinetic_continuum_edge_bounded_below`** — the exact sharpening
  of the loose “bounded below” band of §5.17: the ground from the vacuum
  never falls below the continuum edge $-c/2 = -\tfrac{1}{32}$ per mode
  (and $-\tfrac{1}{16}$ for two modes), is genuinely *negative* (the
  normal-ordered kinetic's ground is $-\tfrac{1}{32}$, not 0), and the
  resolved window keeps positive gaps.  *Asserts*: edge inequalities to
  $10^{-6}$, gaps positive.
- **`qg_tegr_flow_conserves_norm_energy`** — the 𝒮 pair terms populate both
  polarization ladders (each mode's $:\mathcal S^2:$ creates/annihilates
  pairs), yet the restarted-SIRK flow is unitary: norm and $\langle H \rangle$
  are conserved exactly.  *Asserts*: norm/energy to $10^{-7}$.

### 5.24s The outer-vacuum ground-state doctrine — `outer_vacuum_ground_validation.rs`

**Unified final-Hamiltonian convention.** For QYM, QED, QG, and NS, the final
Hamiltonian of record is the corresponding inner one-particle Hamiltonian `h`
(or its allowed scalar shift `h+cI`) enclosed at the outer Fock level:
`H = Σᵢⱼ hᵢⱼ C†(eᵢ) A(eⱼ)`. Creation is on the left and outer annihilation is
on the right. Thus the full final Hamiltonian, including inner pair terms,
annihilates the outer vacuum exactly. Inner pair-squeezed states are diagnostics
of `h`, never replacements for the outer-Fock ground state.


The **ground-state doctrine** of the nested Fock space, pinned for the four
program sectors. The full final Hamiltonian on the nested space is the
corresponding ONE-PARTICLE Hamiltonian $h$ (the sector's Hamiltonian on the
inner Fock space) **enclosed in creation (on the left) and annihilation (on the
right) operators**. This applies uniformly to QYM, QED, QG, and NS; the
one-particle operator is not replaced by an inner-only final Hamiltonian.

$$H = \sum_{i,j} h_{ij}\, C^\dagger(e_i)\, A(e_j),$$

with $e_i$ the inner one-particle basis — the outer second quantization
$d\Gamma(h)$. The one-particle Hamiltonian enters **verbatim**: no
normal-ordering modification of it; the only allowed change is **adding a
constant** (for QYM, to make the truncated one-particle spectrum positive).
Three structural clauses follow, each tested:

- **`outer_vacuum_annihilated_by_full_hamiltonian_all_sectors`** — the FULL
  Hamiltonian annihilates the outer vacuum $|\Omega\rangle$ identically:
  every term carries an outer annihilation operator rightmost, and
  $A|\Omega\rangle = 0$. *Asserts*: $\|H|\Omega\rangle\| < 10^{-12}$ for QYM
  at $g \in \{0,1,2\}$, QED (free photon), QG (scalaron field, gauge-fixed
  scalaron, densitized kinetic) and NS (Eulerian fiber).
- **`qym_outer_vacuum_ground_and_gap_at_all_couplings`** — with the constant
  $c$ lifting the one-particle floor to a fixed margin ($\lambda_{\min}(h +
  c) = 0.1$), the truncated nested matrix (vacuum ⊕ one-quanton ⊕
  two-quanton sectors) has the outer vacuum as its EXACT ground ($E_0 = 0$),
  the first excitation exactly at the margin (the mass gap measured from the
  vacuum), the two-quanton floor at exactly twice the margin (the
  symmetrized-sum structure), and the one-quanton sector equal to $h + c$
  verbatim. *Asserts*: sector equality $10^{-9}$; gap and floor to $10^{-6}$.
- **`qg_ns_outer_enclosure_vacuum_ground_structure`** — the same battery for
  QG and NS: the FINAL test Hamiltonian is the outer enclosure, and the
  vacuum-ground structure holds uniformly (scalaron field needs no constant;
  the gauge-fixed scalaron, the hyperbolic densitized kinetic and the NS
  fiber each need exactly one).
- **`sirk_vacuum_start_rank_collapse_and_gapped_one_quanton_ritz`** — the
  solver-level signatures. From the vacuum the forward Krylov sequence
  collapses to **rank 1** with all Ritz values exactly $0$ (the SIRK sees the
  exact eigenstate); from a one-quanton species superposition the window is
  a genuine cyclic space (rank $\ge 2$) whose Rayleigh–Ritz values all bound
  the sector levels from above and converge DOWN toward the gap. *Asserts*:
  rank-1/Ritz-0 at $10^{-8}$; Ritz floors above $\lambda_{\min} - 10^{-6}$
  (window length kept strictly inside the cyclic dimension — a dependent
  forward sequence's whitening produces ghost rungs, measured at $m = 8$).
- **`qym_one_particle_floor_requires_the_constant`** — the honest statement
  of what the constant compensates: the un-shifted one-particle floors are
  negative (the pair-squeezed levels), bounded below by the restored
  zero-point $-\mathrm{zp}(g)$ with $\mathrm{zp}(g) = 2 + g^2/8$ (from
  $\|B|0\rangle\|^2 = 2 + g^2/4$, halved by the $\tfrac12 B^2$).

*Reframing*: the inner-level "squeezed grounds" of §5.24a are one-particle
statements — what the constant shifts — not the ground state of the nested
theory, which is always the outer-Fock vacuum. This same outer enclosure is
required for QED, QG, and NS final-Hamiltonian tests; their inner one-particle
operators are never presented as standalone full-theory Hamiltonians.

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
| **QED** | inner photon $h$ (including the abelian reduction), enclosed as $H=\sum h_{ij}C_i^\dagger A_j$ | outer vacuum 0; raw canonical sequence (all guards off) reproduces dispersion $\omega=\|k\|$, additivity, Rabi doublets exactly (§5.1, §5.24, `assumption_ledger`) | dispersion; Rabi splitting $2g$; collapse–revival $t_R = 2\pi\sqrt{\bar n+1}/g$; Coulomb $\delta E(r_1)-\delta E(r_2)$; UV-linear self-energy; driven-vacuum coherent statistics Fano $= 1$; blockade sub-Poissonian Fano $< 1$ (§5.24p′) | exact theory at machine precision; $g{-}2$, positronium, Lamb shift, Casimir, blackbody via the constants suite (§5.8) | wide-spectrum coherent-state revival under the RESTARTED solver loses ~9%/restart — the exact Poisson sum is the prediction, not the truncated solver; PT fails at strong coupling (SIRK departs — correctly, it is non-perturbative) | Gram-whitening conditioning over a huge eigenvalue range (solver regime, not a model limit); PT is the wrong tool at strong coupling |
| **QYM** | inner $h_{\rm final}=\tfrac12\pi^2+\tfrac12 B^2$, $B=(A_0-A_1)+\tfrac12 g A_0 A_1$, enclosed as $H=\sum h_{ij}C_i^\dagger A_j$ | outer vacuum 0; Hermiticity; bounded-below spectrum with positive gaps; $\pm g$ symmetry; Gauss-law superselection; BRST charge $\Omega = P\,b^\dagger$ nilpotent with $[H,\Omega]=0$ at $g=0$ (§5.2, §5.17, §5.19) | mass-gap signal $\approx g^2/2$ on the lattice truncation; abelian limit $=$ free Maxwell exactly (§5.9) | Cadabra-derived operator structure; $U(1)$ reduction to QED | the continuum mass gap is NOT claimed — $g^2/2$ is the strong-coupling lattice/truncation result; the brute-force full SU(3) form (76K terms, indefinite $-\tfrac12\pi^2$ in the $H_W$ convention) is NOT SIRK-tractable — the reduced $B(A)$ form is | confinement is a non-perturbative, truncation-sensitive statement; the abstract/indefinite forms are not a positive-definite Fock-mode basis |
| **QG(R²)** | inner scalaron/graviton/TEGR/densitized one-particle $h$, enclosed as $H=\sum h_{ij}C_i^\dagger A_j$ | outer vacuum 0; massive dispersion $\omega = \sqrt{k^2+m^2}$; group velocity $k/\omega < 1$; bounded-below positive-gap spectra (the ESA finite shadow); derivative-variable BRST fixing (§5.3, §5.4, §5.17, §5.20) | scalaron mass gap $m(\alpha)$ at $k\to0$ with certified intervals; classical content: perihelion 43.0″/cy, GPS 45.9 µs/day, Pound–Rebka 2.46e-15, deflection 1.75″, Yukawa $\tfrac13 e^{-mr}$ correction | CODATA/GR/experiment within published bands (§5.3, §5.4, §5.20) | the full TEGR $H_{\rm final}$ in abstract $\mathcal S/E/T$ variables is NOT SIRK-tractable — only the kinetic/quadratic forms are realized; one-particle ESA on the Hermite dense core is a Lean-formalization claim (`../timepiece`), the numerics see only the finite truncation | the abstract variables are not a Fock-mode ladder; ESA is an operator-domain statement, numerically only its finite shadow is visible |
| **NS** | inner quantized Euler one-particle generator enclosed as $H=\sum h_{ij}C_i^\dagger A_j$; | quantized Euler generator $\sum_i\{\pi_i, A_i\}$, $A_i = \sum_j u_j u_{ij} - \nu u_{12+i}$; Eulerian affine fiber with promoted derivative variables | Ehrenfest identity $i\langle[H,u]\rangle = 4\kappa\langle u\rangle + 4c$ exact; Newtonian decay $du/dt = -\nu k^2 u$; gauge condition $C_m = g_m - 2(m+1)u_{m+1}$ a constant of motion BY CONSTRUCTION ($[H,C_m]=0$); $\Omega^2=0$, $[H,\Omega]=0$ (§5.5–5.7, §5.17) | decay rate $\nu k^2$ to <2%; advection identity $d\langle u_0\rangle/dt = 8\langle u_0\rangle\langle u_1\rangle$; the classical fluid laws (K41, Poiseuille, Stokes, Blasius, Strouhal, Lamb–Oseen) as the fiber's classical content | experimental fluid-dynamics bands (§5.5) | the truncated flow LEAKS Ω-content (the bare flow from a ghost-carrying state grows $\|\Omega\psi\|$) — the BRST projector is required in the solve; gauge-condition drift $\propto dt^2$ under truncation; raw-SI stiffness demands restarts for full e-folds | Krylov truncation + finite precision — the exact flow conserves $C_m$ identically (verified: drift → 0 as $dt \to 0$) |

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
- Kolmogorov (1941); Blasius (1908); Roshko (1954); Schlichting,
  *Boundary-Layer Theory* (Blasius profile constants).
- Planck Collaboration 2018 (cosmological parameters: $H_0$, $\Omega_m$,
  $\Omega_\Lambda$, age 13.787 Gyr).
- Starobinsky (1980) — $R+R^2$ inflation; the e-fold count
  $N_e \approx \frac34 e^{\sqrt{2/3}\varphi}$.
- Bekenstein (1973); Hawking (1975) — black-hole entropy, temperature,
  Smarr identity; Bardeen, Carter & Hawking (1973).
- Jaynes & Cummings (1963); Glauber (1963) — coherent states, Poisson
  statistics, $\sqrt{n+1}$ Rabi ladder.
- Casimir (1948) — zero-point energy; $\zeta(-1) = -1/12$ (Abel
  regularization).
- Huang, *Statistical Mechanics* (Sackur–Tetrode, BEC, vdW).
- Abbasi et al. IceCube / Super-K; An et al. (Daya Bay); Ahn et al. (RENO).
- LIGO Scientific & Virgo, PRL 116, 061102 (2016) — GW150914 (chirp mass,
  $f^{11/3}$ scaling).
- Einstein (1915) — perihelion precession $6\pi GM/(c^2a(1-e^2))$.
- Schwinger (1951) — pair production; critical field $E_c = m^2c^3/(e\hbar)$.
- Petermann (1957); Sommerfield (1957) — the $(\alpha/\pi)^2$ anomaly
  coefficient; Laporta & Remiddi (1996) — the $(\alpha/\pi)^3$ term; CODATA
  2018 $a_e$.
- Creutz (1980) — Wilson loops, string tension, strong-coupling
  expansion; Wilson (1974) — lattice gauge theory; Bessel-function
  plaquette integral (e.g. Creutz, *Quarks, Gluons and Lattices*).
- Pagels & Stokar / PDG 2022 — $\alpha_s(M_Z) = 0.1179$, 1-loop running,
  $\Lambda_{\overline{\mathrm{MS}}}$; Morningstar & Peardon (1999) —
  $m_G/\sqrt\sigma = 3.55$ (lightest $0^{++}$ glueball).
- Hawking (1974) — evaporation $\tau = 5120\pi G^2M^3/(\hbar c^4)$;
  Bekenstein (1981) — the entropy bound $S \le 2\pi k_BRE/(\hbar c)$.
- Newton (1687), *Principia* — Kepler's laws, shell theorem, escape
  velocity; Chandrasekhar, *An Introduction to the Study of Stellar
  Structure* (uniform-sphere binding energy $3GM^2/5R$).
- Hagen (1839); Poiseuille (1840) — laminar pipe flow; Stokes (1851) —
  drag $6\pi\mu Rv$; Kolmogorov (1941) — $k^{-5/3}$, dissipation scale.
- Project docs: `AGENTS.md` maintenance checklist (S29–S39 entries map 1:1
  to suites).
