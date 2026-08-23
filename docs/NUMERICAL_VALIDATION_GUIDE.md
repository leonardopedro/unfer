# A Pedagogical Guide to the Numerical Physics Validation Suites

> **What this document is.** Every validation suite in `fock_sirk/tests/` checks
> this project's numerical machinery against numbers the physics community has
> already measured or derived exactly. This guide explains, for each suite,
> *why the physics matters* (the community story), *exactly what formula is
> being tested*, *which Hamiltonian/model realizes it here*, and *what class of
> numerical agreement is demanded*. It closes with the algorithmic core — the
> Hashimoto inverse-free rational-Krylov solver (SIRK) — and a study of its
> spectral edge behaviour.
>
> Audience: a physics student who wants to understand what is really being
> computed, or a developer who wants to know why an assertion carries a given
> tolerance.

---

## 1. Why validate against published physics?

A quantum-field-theory numerics stack can be internally consistent and still
wrong (wrong sign conventions, wrong factors of 2π, wrong units). The only
cure is to reproduce numbers that were obtained by *independent* methods:

1. **Exact theory** — closed-form results every textbook derives (the harmonic
   oscillator ladder, Kepler's third law). Agreement must be at machine or
   solver precision.
2. **Published constants** — CODATA/PDG/Planck-collaboration values. These
   anchor the constants plumbing (units, charge cancellation, mass differences).
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
expansion is ~10⁴× slower unoptimized (~870 s → ~0.1 s per compile).

---

## 3. The engine being tested: inverse-free rational Krylov (SIRK)

### 3.1 What problem it solves

Given a Hamiltonian `H` (here always a finite operator expression on a nested
Fock space) we need eigenvalues (spectra, ground states) and unitary dynamics
`exp(−iHt)|ψ₀⟩`. Dense diagonalization is impossible because the Fock space
is large; classical Krylov methods build a small subspace from powers
`v, Hv, H²v, …` but rational Krylov converges far faster per dimension by
admitting *shifted-and-inverted* information `(H − zₖ)⁻¹v`.

Inverting `(H − zI)` is exactly what we refuse to do — the forward state
sequences explode combinatorially and the whole point of the architecture is
to stay in the sparse forward picture. **Hashimoto's inverse-free trick**
replaces each resolvent application with a *forward product*:

```text
w₀ = |ψ₀⟩
wₖ = (H − zₖ I) wₖ₋₁        k = 1..m        (no inversion!)
```

The span of `{w₀,…,w_m}` is a *polynomial* Krylov subspace that carries the
same spectral information rational Krylov would extract, at the price of
needing a few more dimensions.

### 3.2 The projection identity

Because `H w_k = w_{k+1} + z_k w_k`, the projected Hamiltonian needs **no new
matrix elements** — it is assembled from the Gram matrix alone:

```text
H_jk  =  ⟨w_j|H|w_k⟩  =  G_{j,k+1} + z_k G_{j,k}
```

This one line (`forward_sirk.rs`, step 4) is the heart of the method:
spectral information flows out of inner products the code already computes.

### 3.3 Gram whitening

The vectors `w_k` are wildly non-orthogonal (norms grow like ‖H‖ᵏ), so the
generalized eigenproblem `H_raw c = λ G c` is solved by *rank-revealing
whitening*: Hermitian eigendecomposition of `G`, truncation of near-null
directions (`rel_tol = 10⁻¹²`), and mapping into the orthonormal frame where
an ordinary Hermitian eigensolver runs. This replaced an earlier bare Cholesky
that panicked on degenerate sequences.

### 3.4 Restarted evolution and BRST projection

Long-time dynamics never runs one deep window (see §8: deep windows hit a
conditioning wall). Instead `evolve_restarted` advances through many short
optimal windows, reconstructing the state between them. Gauge-constrained
models carry a BRST charge `Ω`; the flow periodically re-projects onto the
physical subspace `ker Ω` by matrix-free conjugate gradient.

### 3.5 Frames, guards, and model fidelity

The implementation must BE the theoretical model. Two principles are now
enforced in code and pinned by test:

**Canonical default.** `SirkOpts::unit_norm_steps` defaults to `false`: the
stored sequence is the raw Hashimoto product `w_k = (H - z_k I) w_{k-1}`
exactly as theory defines it. No loops exist inside the spectral algorithm
(the forward pass is single-shot) and no renormalization occurs by default.

**Opt-in exact frame.** For deep spectral windows, `unit_norm_steps: true`
stores each vector rescaled to unit norm. This is a numerically EXACT basis
reparametrization of the same Krylov span (Rayleigh-Ritz invariant; the
projection identity becomes `H u_k = tau_{k+1} u_{k+1} + z_k u_k` with the
step norms `tau` folded back into `H_raw`). It exists because raw Gram
matrices grow like the Krylov depth cubed in norm and defeat whitening:
measured ground-error profiles for the displaced oscillator
(ritz_edge_study p2/p2b):

| m | canonical err | unit-norm err |
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
| `prune_eps` | drops tiny components every step | NOW DISABLED BY DEFAULT (`prune_eps: 0.0`) per model fidelity -- memory-bounded runs OPT IN explicitly; Study A shows the opted-in value is invariant across eps = 1e-8..1e-14 (below solver noise floor) |
| mid-sequence BRST projection | replaces w <- P(H-z)w | THEOREM: [H,Omega]=0 makes ker(Omega) invariant, so P is the identity on exact physical sequences; verified inert on physical data (identical spectra, <=1e-8 Omega-content). On contaminated data it enforces ker(Omega) down to its documented contract or fails LOUDLY (`BrstNotConverged`) -- silent pass-through is not an outcome (Study B) |
| adaptive truncation | hard component ceiling | already opt-in (`adaptive:false` default errors instead); at the suites' 50k budgets it NEVER engages -- adaptive-on/off agree exactly and states sit ~500x below budget (Study C) |

**Theory-native dynamics.** The model needs ONE finite time T and a deep
enough dimension m -- no time slicing. Restarts (`evolve_restarted`) remain
as an engineering alternative, but with the flat profile above a single
m=8 window now reproduces the NS Newtonian decay rate to <2% directly
(`ns_sirk_laminar_decay_rate`, part d).

**Resolvedness is solver-enforced.** `ForwardSirkResult::ritz_residuals()`
computes TRUE relative residuals ||Hpsi - theta psi||/||Hpsi|| for every pair
from the stored Gram matrix alone (O(m^2), cancellation-free: the residual
vector e has exactly one out-of-basis component, `e_m = tau_m c_{m-1}`).
`resolved_ritz_values(tol)` returns the converged set, replacing hand-maintained
energy cutoffs. The measured residual ladder at m=8 separates rungs cleanly:
E0:1.5e-5, E1:2.6e-5, E2:2.0e-4 | E3:1.4e-3, E4:7.3e-3, ...

---

## 4. The model zoo

| Builder | Hamiltonian / law | Exact spectrum or result tested |
|---|---|---|
| `qed_free_photon` | `Σ ωₖ Nₖ` | massless dispersion ω=∥k∥, additive multi-photon energies |
| `qed_cavity_frequencies` | conducting plates | `ω_n = nπ/d` → Casimir `E/A = −π²ħc/720d³` |
| `qed_static_charge_interaction` | displaced photon field | Coulomb `−e²/4π·(1/r₁−1/r₂)` from one-photon exchange |
| `qed_pair_production` | γ ↔ e⁺e⁻ vertex | threshold `2√(m²+(q/2)²)`, O(α) self-energy, non-perturbative departure |
| `qed_charge_operator` | `Q = ΣN_e − ΣN_p` | `[H,Q]=0` (unbroken U(1)) |
| `qed_jaynes_cummings` | `ωa†a + ω₀σ†σ + g(aσ†+a†σ)` | vacuum-Rabi splitting 2g, collapse-revival at `t_R = 2π√(n̄+1)/g` |
| `oscillator_beamsplitter` | `ω(N₀+N₁)+J(a†₀a₁+a†₁a₀)` | sector spectrum {ω−J, ω+J}; swap `P=sin²(Jt)` |
| `oscillator_displaced` | `ωN + g(a†+a)` | coherent displacement, `E_n = ωn − g²/ω` |
| `qcd_ym_hamiltonian` / `.cdb` | `H_final = ½π² + ½B²`, `B=(A₀−A₁)+½gA₀A₁` | Cadabra-derived Yang–Mills Hamiltonian, bounded-below spectrum |
| `yang_mills_lattice` | full SU(3) lattice | mass gap ≈ g²/2 (Millennium positivity) |
| `qg_free_graviton` | `Σ c∥k∥ Nₖ` | GW speed = c (GW170817 constraint) |
| `qg_starobinsky_*` | scalaron `Σ m N + ½Σg²`, m²=M²/12α | massive dispersion, ESA/boundedness of R² gravity |
| `ns_eulerian_fiber` | `H = K₀ + {π₀, u·∂u}` | Euler advection, gauge-fixed derivative variables |

The Cadabra2 connection: several Hamiltonians are not transcribed by hand but
**derived** — the classical action is varied, Legendre-transformed, and
gauge-fixed symbolically in `docs/*.cdb` notebooks, canonicalized into the CAS
dialect, and compiled to Fock operators. The single source of truth for each
derived expression lives in one function (e.g. `qcd_ym_expression`); tests
derive their fixtures from it so compiler routes cannot drift apart.

---

## 5. Suite-by-suite walkthrough

### 5.1 QED — `qed_validation.rs`, `qed_precision.rs`

*Community relevance:* these are the most precisely tested predictions in
science. The electron anomalous moment `g−2` is the classic agreement between
theory and experiment (the 2023 Fermilab discrepancy is *why we test it*);
the Lamb shift launched modern QFT (1947, shelter island); positronium was
the first purely leptonic bound system; the Casimir force (Lamoreau 1997,
measured to 5%) is the cleanest macroscopic vacuum effect.

*What is asserted:* vacuum normal-ordering (`⟨0|H|0⟩=0` — no cosmological
constant from photon zero-points), exact dispersion, Casimir spectrum,
Coulomb's law assembled from exchange amplitudes, pair-production thresholds
and the crossover from perturbative to non-perturbative behaviour as coupling
grows, Thomson limit of Compton kinematics, positronium fine structure and
lifetimes, Uehling potential shape, Bethe's Lamb-shift estimate, hydrogen fine
structure, Stefan–Boltzmann and Wien laws from the Planck integral (computed
by Simpson quadrature against `π⁴/15`).

### 5.2 QCD — `qcd_validation.rs`

*Community relevance:* Gross–Wilczek–Politzer (Nobel 2004) discovered
asymptotic freedom: `β₀ = 11 − 2N_f/3 > 0` makes the coupling *decrease* at
short distances. The world average `α_s(M_Z) = 0.1179` and its run to τ-mass
(`α_s(M_τ) ≈ 0.32 ± 0.03`) are PDG headline numbers; the R-ratio confirmed
colour (N_c=3) at SPEAR; confinement/mass-gap remains a Clay Millennium
Problem.

*What is asserted:* colour factors C_F=4/3, C_A=3, T_R=½ **computed** from
SU(3) structure constants (never hard-coded); Cornell-potential coefficient
from one-gluon exchange; β₀/β₁ (Jones & Caswell 1974) and the two-loop running
through flavour thresholds; R-ratio values 2, 10/3, 11/3, 5; lattice mass gap;
non-perturbative gluon self-energy departing from the quark-loop form at
strong coupling.

### 5.3 Quantum gravity — `qg_validation.rs`, `qg_starobinsky_*.rs`

*Community relevance:* the classical tests (perihelion 43″/century,
Eddington's 1919 deflection, Pound–Rebka 1959, GPS time-dilation) are the
historical gatekeepers of general relativity; GW170817 fixed graviton speed
to |Δv/c|<10⁻¹⁵; Starobinsky R² inflation predicts the scalaron, whose mass
m² = M²/(12α) is set by the potential curvature.

*What is asserted:* Planck units from CODATA with dimensional identities;
all four classical GR tests to published precision; TEGR↔GR equivalence via
the Friedmann equation; graviton and massive-scalar dispersions through the
SIRK engine; the derivative-variable BRST gauge fixing (Navier-Stokes pattern)
with nilpotency `[Ω,[Ω,H]]=0`-class checks and controlled truncation drift.

### 5.4 Navier–Stokes — `ns_numerical_validation.rs`, `ns_derivative_variable_fixing.rs`

*Community relevance:* Kolmogorov's 1941 theory (−5/3 spectrum, 4/5 law) is
the pillar of turbulence; Blasius' boundary layer (1908) matches wind-tunnel
data to ~3%; Reynolds' 1883 pipe experiments define transition; the Clay
Millennium Problem asks whether smooth Euler/NS solutions exist globally.

*What is asserted:* all K41 scalings and exact relations; Hagen–Poiseuille;
Stokes/Oseen drag; Blasius constants; transition Reynolds numbers;
Strouhal shedding; Lamb–Oseen vortex growth; the SIRK Ehrenfest decay-rate
measurement `νk²`; and the promoted-derivative-variable formalism: gauge
conditions verified *by construction*, observable consistency, bare-vs-
BRST-projected flow equality, drift ∝ dt².

### 5.5 Special relativity & nuclear — `sr_nuclear_validation.rs`

*Community relevance:* Frisch–Smith (1963) made cosmic-ray muons the public
proof of time dilation; LHC dipoles run at 8.33 T — engineering built on
`Bρ = p/q`; the GZK cutoff (Greisen–Zatsepin–Kuzmin 1966) was confirmed by
HiRes/Auger; the Weizsäcker formula (1935) still organizes nuclear masses.

*What is asserted:* PDG masses and rest energies; π→μν momentum 29.788 MeV/c
(two-body kinematics); Breit–Wheeler threshold identity `(m_ec²)²/ε`; GZK
pion-production window; muon survival contrast (>10⁹ ratio); LHC field and
revolution frequency; SEMF B/A peak in the iron group with heavy falloff;
AME2020 Q-values (deuteron 2.224566 MeV, tritium endpoint 18.591 keV).
*Bug caught here:* the charge in `B = pc/(qρ)` cancels — p = E/c already
carries it; an extra q gave 10⁻¹⁸ instead of 8.33 T.

### 5.6 Astro, plasma, metrology — `astro_plasma_validation.rs`

*Community relevance:* Hawking temperature (1974) ties thermodynamics,
quantum theory and gravity; Chandrasekhar's limit (1931) explains white-dwarf
fates (Nobel 1983) and calibrates Type-Ia cosmology; Eddington luminosity
sets accretion physics; Planck-2018 parameters define modern cosmology; the
"quantum SI" (2019) defines ohm/volt through R_K and K_J.

*What is asserted:* Hawking 6.17×10⁻⁸ K/M☉ and Schwarzschild radii; ISCO
ringdown 4397 Hz/M☉; Chandrasekhar π(ħc/G)^{3/2}/(μ_e m_p)² = 1.44 M☉ with
its μ_e scaling; Eddington 1.26×10³¹ W/M☉ (needs the *electron* radius —
another caught bug); ρ_c(H₀)=8.60×10⁻²⁷ kg/m³ × Ω_b → BBN baryon density; CMB
photon gas 411 cm⁻³ / 0.26 eV/cm³; ionospheric plasma frequency 8.98 MHz and
Debye length (two algebraically identical routes must agree to 10⁻¹²);
Alfvén-speed band; the metrology triangle closing `K_J·R_K = 2/e` exactly;
BCS Δ/kT_c = 1.764; Peters inspiral chirp — RK4-integrated `df/dt` against
the closed-form coalescence time to <10⁻⁴ across a decade, with the 𝓜^{-5/3}
scaling exponent verified numerically.

### 5.7 Electromagnetism & optics — `em_optics_validation.rs`

*Community relevance:* cyclotron frequencies calibrate every accelerator and
Penning trap; waveguide TE₁₀ cutoffs are radar-WWII engineering (the X-band);
skin depth governs power distribution; Rayleigh's λ⁻⁴ answers *why the sky
is blue*; and the **Larmor collapse paradox** — the classical hydrogen atom
radiates away in ~1.6×10⁻¹¹ s — is the calculation that made classical
physics untenable and forced quantum mechanics.

*What is asserted:* cyclotron 15.245 MHz (p) / 27.992 GHz (e) at 1 T; WR-90
cutoff 6.557 GHz; Brewster 56.31° and critical 41.81° angles; Cu skin depth
~9.3 mm at 50 Hz; dipole radiation resistances; the Rayleigh ratio 4.35; and
the collapse time obtained from the integrated fall `dr/dt =
−e⁴k/(3πε₀c³m²r²)`.

### 5.8 Statistical mechanics — `statmech_validation.rs`

*Community relevance:* Maxwell's 1860 speed distribution underlies kinetic
theory; the Sackur–Tetrode equation (1925) was the first quantitative entry
of quantum indistinguishability into thermodynamics — it resolves Gibbs'
paradox and matches third-law calorimetry of argon (154.8 J/mol·K) to ~1%;
BEC predicted for ideal gases (1924, Einstein) was realized at 3.1 K-scale
densities (λ-point 2.17 K for the interacting liquid; Nobel 1995/2001); van
der Waals' corresponding-states law P_cV_c/RT_c = 3/8 (Nobel 1910) is the
first universality statement.

*What is asserted:* the √2 : √(8/π) : √3 speed ratios; R = N_Ak_B and STP
molar volume 22.414 L; Sackur–Tetrode argon; ideal-gas BEC temperature 3.1 K
*above* the interacting λ-point (interactions lower it); and the vdW critical
ratio located **numerically** — Newton's method on the combined stationarity
condition `2/(v−b) = 3/v` (obtained by dividing ∂p/∂v=0 by ∂²p/∂v²=0),
recovering 3/8 for arbitrary a,b. *Bug caught here:* the original double-Newton
solve was ill-conditioned to NaN and replaced.

### 5.9 Classical dynamics — `classical_dynamics_validation.rs`

*Community relevance:* Foucault's 1851 Panthéon pendulum made Earth's rotation
visible in a museum; Kepler's laws (1609–1619) founded celestial mechanics;
the Roche limit explains planetary rings and tidal disruption; the
finite-amplitude pendulum series (1 + θ₀²/16 + …) is the textbook example of
regular perturbation theory; GW150914 (Nobel 2017) introduced the chirp mass
𝓜 = (m₁m₂)^{3/5}/(m₁+m₂)^{1/5} to the world.

*What is asserted:* Foucault rate 11.32°/h at Paris latitude; sidereal years
of Earth/Mars/Jupiter (after fixing a units slip in the Gaussian year —
365.2568983 **days**, not seconds); escape/circular velocities with their
exact √2 ratio; rigid Roche limit ≈18,400 km; the pendulum series against
direct RK4 integration of θ̈ = −sinθ to <10⁻⁵ (an integrator cross-check
whose time window must exceed one period — the first version failed because
it didn't!); relativistic Doppler z=1 ⇔ β=3/5; chirp mass 𝓜(36,29)≈28.1 M☉.

### 5.10 Weak interactions & neutrinos — `weak_neutrino_validation.rs`

*Community relevance:* the muon lifetime defined G_F ("Fermi's second
interaction"); Super-Kamiokande (1998, Nobel 2002/2015) discovered
atmospheric ν oscillations with first maximum at L/E ≈ 495 km/GeV; KamLAND
(2002) saw reactor-ν disappearance beyond the first lobe; Daya Bay (2012)
measured θ₁₃, opening the mass-hierarchy era.

*What is asserted:* tree-level lifetime τ = 192π³ħ/(G_F²m_μ⁵c⁴) = 2.188 µs —
**under** the measured 2.19698 µs, because the measured G_F absorbs loop
corrections (the direction itself is asserted); atmospheric first-maximum
L/E; KamLAND-baseline suppression band; Daya-Bay maximum survival
1 − sin²2θ₁₃ ≈ 0.915.

### 5.11 Coupled oscillators — `coupled_oscillator_sirk.rs`

*Community relevance:* the beamsplitter Hamiltonian is linear-optical quantum
computing; the Jaynes–Cummings model is cavity QED (Rabi oscillations measured
per-qubit; Nobel 2005 fibre optics / 2012 ion traps); the displaced oscillator
is the polaron/dressing paradigm — every perturbative "self-energy" is one.

*What is asserted:* exact sector spectra {ω−J, ω+J}; full swap dynamics with
norm conservation through restarted Krylov; coherent-state content ⟨N⟩=α² of
the displaced ground state; level placement E_n = ωn − g²/ω within
solver-accuracy bands (see §8 for the edge study).

---

## 6. The tolerance taxonomy

| Class | Typical tolerance | Examples |
|---|---|---|
| Exact identities | 10⁻⁹ – 10⁻¹² | metrology triangle, Breit–Wheeler product, √2 velocity ratio |
| Derived constants | 10⁻³ – 10⁻⁶ relative | Chandrasekhar 1.44 M☉, ISCO 4397 Hz, LHC 8.33 T |
| Experimental bands | quoted windows | Sackur–Tetrode 153 vs 154.8, KamLAND suppression band, Alfvén band |
| Solver bands | documented profiles | displaced-oscillator levels, chirp RK4 <10⁻⁴, Sackur quadrature 10⁻⁴ |

A tolerance is never chosen to make an assertion pass; it encodes which of
these classes the quantity belongs to.

---

## 7. Reading the Fock-space tests correctly

Two framework conventions matter when writing new tests:

1. **Inner vs outer construction.** A multi-occupation state must be ONE
universe with correct inner occupation (`modes:{i:2}`), not two outer
universes — otherwise number-operator additivity breaks. The inner-ladder
construction gives `⟨0|H|0⟩ = 0` automatically (normal ordering strips the
[a,a†] zero-point).
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
normalized vector has Rayleigh quotient θ = Σ|cₙ|²Eₙ — a convex mean. Hence
every Ritz value lies inside [E₀, E_m], where m is the reachable occupation.
No value can exceed the highest reachable rung. *(Verified.)*

*P2 Conditioning wall.* Ground-level error vs window depth m (measured):
`err(4)≈6×10⁻⁶, err(6)≈10⁻⁹, err(8)≈2×10⁻⁶, err(10)≈2×10⁻³`. Convergence is
NOT monotone: past the optimum, ‖w_k‖ ~ ‖H‖ᵏ wrecks the Gram conditioning and
whitening truncation injects noise faster than the bigger subspace helps.
This wall is the reason long evolutions use restarted short windows rather
than one deep solve. *(Verified, profile pinned.)*

*P3 Climbing.* sup(Ritz) increases strictly with m — the topmost value moves
up the ladder toward ever-higher rungs. *(Verified.)*

*P4 Mixture content.* Reconstructing the top-Ritz vector gives mean
occupation ⟨N⟩ well above the resolved window (mixed high-n support), while
its direct Rayleigh quotient reproduces the Ritz value — the small-basis
eigenpair genuinely represents a big-space vector with that energy mean.
For contrast, the ground vector reproduces the EXACT coherent-state content
⟨N⟩ = α² = (g/ω)² — the machinery recovers dressing physics quantitatively.
*(Verified.)*

*P5 Residual separation.* ‖Hψ−θψ‖/‖ψ‖ is tiny for converged pairs and orders
of magnitude larger at the top — they are approximate eigenvectors of
convergence-related quality, not noise. *(Verified.)*

**Practical rule.** Filter Ritz values above `E_top_resolved + gap/2` before
level-placement assertions; treat the survivors near the window edge as
higher-rung estimates whose convergence improves with deeper windows *up to*
the conditioning wall.

**Follow-up (solver-enforced resolution).** The practical rule is now an API:
`resolved_ritz_values(tol)` selects pairs by true residual (see 3.5), and the
unit-norm frame removes the wall entirely for deep windows -- so the edge
values can also be *converged away* by extending m, which is the theory's own
convergence knob. The displaced-oscillator suite now runs one m=8 window and
asserts exactly the first three rungs resolve.

---

## 9. Sources

- CODATA 2018 fundamental constants; SI-2019 exact definitions (h, e, k_B, N_A).
- PDG 2024 review (masses, lifetimes, α_s world average).
- Zee, *QFT in a Nutshell* §I.3 (one-photon exchange → Coulomb).
- Peskin & Schroeder ch. 16 (running coupling, colour factors).
- Greisen (1966); Zatsepin & Kuzmin (1966) — GZK cutoff.
- Shapiro & Teukolsky, ch. 3 (Chandrasekhar mass).
- Peters (1964) — gravitational-radiation inspiral.
- Kolmogorov (1941); Blasius (1908); Roshko (1954).
- Huang, *Statistical Mechanics* (Sackur–Tetrode, BEC, vdW).
- Abbasi et al. IceCube / Super-K; An et al. (Daya Bay); Ahn et al. (RENO).
- LIGO Scientific & Virgo, PRL 116, 061102 (2016) — GW150914.
- Project docs: `AGENTS.md` maintenance checklist (S29–S39 entries map 1:1 to suites).
