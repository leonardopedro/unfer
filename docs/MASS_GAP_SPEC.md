# MASS_GAP_SPEC — the proof-facing seam of the certified mass gap

*Spec of record for `fock_sirk::mass_gap_spec` and the sector-solve seam
`fock_sirk::certified_mass_gap_parity`. This is the non-Lean part of the
`MASS_GAP_CERTIFIED.md` §4–§5 formalization: the numerical code is converted
into a form proofs can be written about — pure, dependency-free functions with
exact contracts, a runtime-enforced precondition seam, and NDJSON certificates
consumable by the nanoda/lean4export pipeline (S29/S31). The Lean4 parts are
out of scope here (LLM-Lean4-specialist).*

> **Operational plan**: `../PLAN_MASS_GAP.md` (PLAN M) is the planner-side
> plan for this pipeline — everything except Lean4 code (emission, gates,
> fixtures, evidence). The Lean4 work order for the specialist lives in
> `../../timepiece/CONSOLIDATED_PLAN.md`, section "State of the project —
> 2026-10-04: the mass-gap proof in Lean4" (§M4). The proof is already
> started (T1–T12 statuses there): nested Fock space gives the exact sector
> structure and the SIRK–Hashimoto approximations carry rigorous certified
> bands, which is what makes a proof over numerically located levels sound.*

**Hamiltonian of record: the 3D gauge-fixed QYM Hamiltonian, not the lattice.**
The mass-gap observable lives on `qcd_ym_hamiltonian(g)` — the nested-Fock
realization of the Cadabra-derived `H_final = ½π² + ½B²` with
`B = (A₀ − A₁) + ½g·A₀A₁` (`docs/yang_mills_hamiltonian.cdb`). All numerical
approximations are SIRK–Hashimoto solves (`solve_forward_sirk_with_opts`).
The sector symmetry is the exact `Z₂` **reflection** `R: (A₀, A₁) → (−A₁, −A₀)`
(occupation parity is *not* a symmetry at `g > 0`, where `B²` carries
3-operator products): the R-even sector contains the vacuum start, the R-odd
sector the `(|1₀⟩ + |1₁⟩)/√2` one-quantum start.

## 1. The theorem of record (T6)

For the gauge-fixed QYM Hamiltonian `H` and its Galerkin/Krylov truncation
`H_m` (the whitened projection `h_proj` the kernel actually diagonalizes), let
`θᵉ₀`, `θᵒ₀` be the computed **lowest Ritz values** of the R-even and R-odd
sector solves, with certified widths `δᵉ`, `δᵒ`:

```
λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)                      (T6)
```

The width is the sum of three explicit, machine-checkable terms (§4.4):

```
δˢ = ‖rˢ‖  (T2: a-posteriori Rayleigh–Ritz residual, Parlett)
   + c(nˢ)·u·‖Ĝˢ‖  (T1/T3: eigendecomposition backward error, c(n) = n³,
                    u = 2⁻⁵³, ‖Ĝˢ‖ = spectral norm of the matrix diagonalized)
   + h_Oˢ  (T5: directed-rounding enclosure of the measured value,
            h_O = 4u·max(|θ|, 1))
```

**Honesty boundary:** the certificate is a theorem about the truncated
gauge-fixed object `H_m` — the object the numerics actually computes — not the
continuum Millennium mass gap. The continuum passage needs one additional
(gap-preserving) convergence theorem, explicitly out of scope.

## 2. Code → math mapping

| Math | Code | Where |
| :--- | :--- | :--- |
| `u = 2⁻⁵³` | `UNIT_ROUNDOFF` | `mass_gap_spec` / `certificate` |
| `c(n) = n³` | `backward_error_const(rank)` | `mass_gap_spec` |
| `δ = ‖r‖ + c(n)u‖Ĝ‖ + h_O` | `certified_width(residual, rank, g_norm, theta)` | `mass_gap_spec` |
| `|θ − λ| ≤ ‖Hψ − θψ‖/‖ψ‖` | `parlett_bound(residual_norm, psi_norm)` | `mass_gap_spec` |
| `θ = ⟨ψ,Hψ⟩/⟨ψ,ψ⟩` | `rayleigh_quotient(psi_h_psi, psi_psi)` | `mass_gap_spec` |
| `[θ−δ, θ+δ] ∋ x` | `interval_contains(value, delta, x)` | `mass_gap_spec` |
| T6 lower bound | `certified_gap_lower_bound(θᵒ, θᵉ, δᵒ, δᵉ)` | `mass_gap_spec` |
| gap interval | `gap_interval(θᵒ, θᵉ, δᵒ, δᵉ) → (lo, hi)` | `mass_gap_spec` |
| stopping rule `lo > 0` | `gap_certified_positive(lo)` | `mass_gap_spec` |
| sector purity witness | `parities_disjoint(max_overlap, tol)` | `mass_gap_spec` |
| per-Ritz certificate | `certified_ritz_values(res)` → `Vec<Certificate>` | `certificate.rs` |
| T6 assembly | `certified_mass_gap(even, odd)` → `GapCertificate` | `certificate.rs` |
| NDJSON emitter | `emit_gap_certificate_ndjson(gap)` | `certificate.rs` |
| **seam** (solve + preconditions + assembly) | `certified_mass_gap_parity(h, v_e, v_o, shifts, opts)` | `forward_sirk.rs` |

## 3. Preconditions of T6 and where they are enforced

1. **Sector purity**: the reflection `R: (A₀,A₁) → (−A₁,−A₀)` is an exact
   `Z₂` symmetry of the gauge-fixed H for **all** `g` (it leaves
   `B = (A₀−A₁) + ½g·A₀A₁` invariant); the starts `v_e = |Ω⟩` (R-even) and
   `v_o = (a†₀ + a†₁)|Ω⟩/√2` (R-odd) are pure-sector; the two Krylov chains
   are disjoint and the Ritz sets independent.
   *Runtime witness:* `parities_disjoint(max_chain_overlap, 1e-8)` inside
   `certified_mass_gap_parity` (`debug_assert`). *Numerical witness:*
   `qym_gauge_fixed_reflection_symmetry_sector_purity`.
2. **Ground selection**: `θˢ₀` is the lowest Ritz value of sector `s` — the
   solver returns the sorted spectrum (`ForwardSirkResult::ritz_values`).
3. **Enclosure validity**: `δˢ` bounds `|θˢ − λˢ|` — residual measured
   cancellation-free from the stored Gram (`ritz_abs_residuals`), roundoff
   and enclosure per §4.4. *Pinned by:*
   `mass_gap_spec::tests::parlett_bound_holds_on_explicit_matrix` and
   `certified_width_matches_certificate_delta`.
4. **NO vacuum-sector identification** — deliberately dropped for the
   gauge-fixed H: `⟨0|H|0⟩ = 0` (normal ordering) but the ground is a
   pair-squeezed state below it (`E₀ < 0`, deepening with `g`; at strong
   coupling the truncated ground even flips reflection-odd). The seam
   therefore certifies the **enclosure** of the exact truncated gap by the
   sector-ground-difference interval, not a strict `lo > 0` stopping rule.
   (The lattice-era `even_sector_is_vacuum` predicate is retained as a
   generic contract only.)

## 4. The numerically-pinned claims (the gauge-fixed formalization)

`fock_sirk/tests/qym_mass_gap.rs` (10 tests) executes the formalization on
`qcd_ym_hamiltonian(g)` + SIRK–Hashimoto:

| Claim | Test | Measured |
| :--- | :--- | :--- |
| nested-Fock structure (⟨0|H|0⟩=0, Hermitian, ⟨vac|H|1,1⟩=−1, non-abelian terms at g>0) | `qym_gauge_fixed_hamiltonian_nested_fock_structure` | ✓ |
| R-reflection exact + R-pure chains disjoint (all g) | `qym_gauge_fixed_reflection_symmetry_sector_purity` | ✓ ‖[H,R]‖ = 0 |
| low window alternates R-parity at g=1 (E₀ even, E₁ odd) | `qym_gauge_fixed_low_window_reflection_alternation` | ✓ |
| **gapped at g>0**: E₁−E₀ = 0.091 stable across N≤6/N≤8; SIRK Ritz values bound the exact levels from above | `qym_gauge_fixed_spectral_gap_positive_stable` | ✓ |
| **gapless abelian limit**: g=0 truncated gap shrinks with depth (0.34→0.19→0.12 at N≤4/6/8); R-sector grounds coincide at every m | `qym_gauge_fixed_abelian_limit_gapless` | ✓ |
| gap grows with g (0.030 at g=0.5, 0.091 at g=1, 1.24 at g=2, N≤8) | `qym_gauge_fixed_gap_grows_with_coupling` | ✓ |
| one-particle ground is squeezed, not the Fock vacuum (E₀<0 under ⟨0|H|0⟩=0; R-odd at strong coupling) — a one-particle diagnostic; the nested-Fock ground stays the outer vacuum (S45) | `qym_gauge_fixed_one_particle_ground_sector_structure` | ✓ |
| SIRK sector Ritz values tighten with m | `qym_gauge_fixed_sirk_ritz_monotone_stable_in_m` | ✓ |
| certified interval encloses the exact truncated gap | `qym_gauge_fixed_certified_enclosure_of_exact_gap` | ✓ |
| seam = manual assembly; spec predicates fire | `qym_gauge_fixed_proof_facing_seam_agrees_manual_assembly` | ✓ |

**Refinement found by the numerics (corrects the lattice-era wording):** the
continuum gauge-fixed H's gap is its own — `≈ 0.09` at `g = 1`, growing with
`g` — **not** the lattice's `g²/2` (an electric-lattice result: the lattice
electric term `(g²/2)Σn` gaps the one-quantum sector, while the gauge-fixed
H's quartic `½B²` confines the `(A₀−A₁)` mode). The plan's `lo > 0` stopping
rule, reachable on the lattice at `m = 4`, is not honestly reachable on the
gauge-fixed H at the solved `m` (the deeply squeezed ground makes the Krylov
residuals large), so the certified statement here is the **enclosure of the
exact truncated gap** by the sector-ground-difference interval, cross-checked
against the exact `N ≤ 8` diagonalization of the truncated model.

## 5. Truncation-depth stability (replaces the lattice finite-size study)

The lattice's `l ∈ {2,3,4}` finite-size / Richardson study does not carry
over (the gauge-fixed H has no lattice size parameter). Its honest
replacement is the **truncation-depth study**: at `g = 1` the exact truncated
gap is stable across `N ≤ 6` and `N ≤ 8` (`0.0911` vs `0.0912` — the confining
quartic converges), while at `g = 0` it shrinks with depth (`0.336 → 0.190 →
0.122` — the gapless free-Maxwell limit). This depth-stability is the order
parameter separating gapped from gapless.

## 6. Certified table (gauge-fixed)

`fock_sirk/tests/qcd_mass_gap_certified.rs` (4 tests) instantiates the T6
assembly on the gauge-fixed H: the certified window `[lo, hi]` encloses the
exact truncated gap `E₁ − E₀` for `g ∈ {1, 2}` across truncations, and the
NDJSON emitter produces well-formed sector certificates whose
`certified_positive` flag truthfully reflects `lo > 0` of the emitted numbers.

The window is a valid **enclosure, not a resolution**: at `m = 12` it is
`14.88` wide at `g = 1` (`δᵒ+δᵉ = 7.438`) against a gap of `0.0912`, so
`lo = −8.009 < 0` and the T6 statement is not reachable at this Krylov order —
which is exactly what the emitted `"certified_positive":false` reports. The
suite pins that non-resolution (and the fact that the window also contains the
lattice value `g²/2`, so it cannot separate the two models); the departure from
`g²/2` is established by the exact `N ≤ 8` diagonalization, not by the window.

### 6a. Is a larger coupling enough? No — measured (2026-11-10)

The interval is `[gap ± δ]`, so the natural hope is that a stronger coupling
grows the gap past the width. It does not: `|lo|` grows faster.

| `g` | θᵒ₀ | θᵉ₀ | measured gap | δᵒ+δᵉ | `lo` |
|---|---|---|---|---|---|
| 1 | −2.103 | −1.531 | −0.571 | 7.44 | −8.01 |
| 2 | −4.633 | −2.352 | −2.281 | 25.5 | −27.70 |
| 3 | −8.952 | −4.621 | −4.330 | 57.7 | −61.99 |
| 4 | −14.484 | −8.289 | −6.195 | 101.1 | −107.3 |
| 6 | −24.330 | −15.787 | −8.543 | 201.9 | −210.4 |
| 8 | −39.757 | −28.701 | −11.06 | 350.0 | −361.1 |
| 12 | −82.940 | −65.918 | −17.02 | 766.4 | −783.4 |

(`m = 12` for `g ≤ 4`, `m = 10` for `g ≥ 6`: beyond that the odd sector exceeds
a 2×10⁶-component budget.) `lo ≈ −6.9g²` while the exact truncated gap grows
like `≈ 0.09 g^{3.8}`, so the two curves would cross near `g ≈ 11` — out of
reach, and the wrong thing to aim at, because:

* **the interval's centre is negative at every coupling** (`θᵒ₀ < θᵉ₀` although
  `E₁ > E₀`). The T6 assembly differences two *separately unconverged* sector
  Ritz values; the even (vacuum-start) one sits `1.2` above `E₀` at `g = 1`
  against the odd one's `0.55` above `E₁`. So `lo > 0` needs a resolved sector
  ground first, and no coupling supplies one;
* **depth does not supply one either**: at `g = 1`, `m = 12 → 20` moves
  `δᵉ` `3.35 → 4.09` and `lo` `−8.01 → −9.92` — the width grows with depth;
* **shift design cannot help, structurally**: `w_k = (H − z_k)w_{k−1}` spans
  `{ψ₀, Hψ₀, …}` for *any* shift sequence, so every `θ` and `δ` is
  shift-independent. Five very different designs (imaginary ladders of three
  scales, shifts at the ground, a real ladder) return `θ₀ = −1.450523`,
  `δ = 3.159e0` — pinned by
  `sirk_consistency_validation::shift_sequence_cannot_move_the_krylov_subspace`;
* **a better even-sector start does not help**: `|Ω⟩ + α|1,1⟩` (the guide's
  pair-squeezed ground) at `g = 1, m = 12` moves `θᵉ₀` `−1.531 → −1.742`
  (`α = 1`) but `δᵉ` `3.35 → 4.27`, so `lo` worsens `−8.01 → −8.72`.

What this leaves: the ground residual stays `O(1)` at every `m` tried, which is
consistent with there being **no normalizable ground before truncation** — the
same continuum phenomenon as the `g = 0` case (`qcd_ym_hamiltonian(0)` has
bottom `−2`, a limit point; see the guide §5.22). The `N ≤ 8` gap of `0.0912`
is a truncation statement, and the SIRK solve has no `N` knob (`max_components`
bounds memory, not the inner occupation ceiling). Certifying a gap therefore
needs a truncation-controlled solve, not a larger coupling.
(the pinned Lean4 T6 instance in `../timepiece/GapCertificate/` remains the
historical lattice instantiation; regenerating it for the gauge-fixed numbers
is an LLM-Lean4-specialist item).

### 6b. The truncation control, and why `lo > 0` is not a proof (2026-11-10)

§6a concluded that a truncation-controlled solve is what is missing. It is now
a supported API: `QuantumState::restrict_inner(ceiling)` is the projector
`P_N` (drop every component whose total inner occupation exceeds `N`),
`fock_sirk::truncated_matvec` is the compressed `P_N H P_N`, and
`fock_sirk::solve_forward_sirk_truncated` runs SIRK on it (projecting the start
first). Because the start and every iterate live in `V_N`, `P_N` acts as the
identity on the state SIRK subtracts from `H u`, so the operator the loop steps
with is exactly the bounded `H_N`. On a bounded operator an exact
parity-resolved spectrum exists (dense diagonalization; `R|n₀n₁n₂n₃⟩ =
(−1)^{n₀+n₁}|n₁n₀n₂n₃⟩`, eigenvectors classified by `⟨ψ|Rψ⟩ = ±1`), so the
certificate can finally be compared against truth. All numbers below are from
`the_certified_gap_lower_bound_needs_the_sector_ground` (`m = 14`, `N ≤ 6`) and
the dense spectrum at `N ≤ 8`, `m = 12`; `P_N H P_N` is checked self-adjoint to
`1e-12` on the states used.

**Answer to §6a's question: no coupling is needed — `g = 0` already does it.**

| `g` | `N` | λ_even | λ_odd | λ_odd−λ_even | θᵉ₀ | θᵒ₀ | `lo` |
|---|---|---|---|---|---|---|---|
| 0 | 6 | −1.304851 | −1.114924 | **+0.189928** | −1.304851 | −1.114924 | **+0.189913** |
| 0 | 8 | −1.427286 | −1.304851 | +0.122434 | −1.427251 | −1.304848 | +0.104776 |
| 0.5 | 8 | −1.545551 | −1.576009 | −0.030458 | −1.462674 | −1.484936 | −0.430261 |
| 1 | 8 | −2.743630 | −2.652468 | +0.091161 | −1.802710 | −2.347584 | −2.346919 |
| 2 | 8 | −6.511130 | −7.754777 | −1.243647 | −5.064921 | −7.450581 | −6.748513 |

The `g = 0` row is the point. At `N ≤ 6` both sector solves isolate their
grounds exactly (`θᵉ₀ = λ_even`, `θᵒ₀ = λ_odd` to `1e-6`) and the T6 expression
is **positive, +0.189913, on the gapless control** — the free theory truncated
to a finite occupation basis. So `lo > 0` is a statement about `H_N` and never
about the continuum: the finite basis has a discrete spectrum whether or not the
theory does. (Consistent with §5: the truncated `g = 0` gap *shrinks* with depth,
0.336 → 0.190 → 0.122, so positivity is available at every `N` while the number
it certifies tends to zero.)

**Two reasons `lo` is not even a bound.**

1. *Its step needs an unstated hypothesis.* `lo` uses `λ₁ ≥ θᵒ₀ − δᵒ`, which
   holds only if `θᵒ₀` is within `δᵒ` of the odd sector's **ground**. Parlett's
   bound places *some* eigenvalue within `δᵒ`; it does not say which, and a
   Krylov start orthogonal to the ground keeps its whole chain orthogonal to it.
   Measured at `g = 0`, `N ≤ 6`, `m = 14`, odd start orthogonalized against the
   odd ground (overlap `7.5e-16`): `θᵒ₀ = −0.778457`, i.e. the odd *second*
   level (`λ₂`), with `δᵒ = 1.2e-7` — *more* converged than the in-sector solve —
   and the assembly returns `lo = +0.526380` against a true gap of `0.189928`.
   Every documented precondition holds; the conclusion is wrong by `0.34`. A
   residual measures "converged", not "lowest"; a rigorous lower bound on a
   *minimum* needs an exclusion argument (an eigenvalue count, or a
   Kato–Temple-type gap), because variation gives upper bounds on minima only.
   `certificate.rs` and `mass_gap_spec.rs` now state this hypothesis
   (precondition 3) and label the assembly an estimate.
2. *Its orientation is fixed, and the sector order flips.* `lo` is
   `θᵒ₀ − θᵉ₀ − δ`, so it can only be positive when the odd sector sits *above*
   the even one. At `g = 0.5, N ≤ 8` the truncated split is `0.0305` in
   magnitude but the odd ground is the *lower* one (`λ_odd − λ_even =
   −0.030458`), and at `g = 2, N ≤ 8` the crossing is large (`−1.243647`,
   the known ground flip). No amount of coupling or depth fixes an orientation
   at those truncations.

Two caveats for anyone reading the campaign fixtures:

* `mass_gap_campaign`'s `exact_gap_at_n` column is `E₁ − E₀` of the **full**
  truncated spectrum, not the sector gap. Where the sectors cross it is the
  absolute split with a flipped sign: at `g = 2, N ≤ 8` it reads `+1.243647`
  while the parity-resolved sector gap is `−1.243647`. Read it as `|gap|` there.
* when a truncated Krylov chain collapses (rank `10` of `m+1 = 15` at `m = 14`,
  `g = 0`), `h_proj` — assembled from the step identity rather than from
  explicit `H` applications — departs from Hermitian by a measured `2.5e-4`.
  That is an assembly property of the collapsed chain, not of `P_N H P_N`
  (checked self-adjoint to `1e-12`), but it sets the accuracy of `θ` on such
  solves and hence the resolution of any bound comparison made there.

### 6c. A valid bound: exclusion + Kato–Temple (2026-11-10)

§6b left two gaps: the projector was caller-side, and there was no way to
witness a sector ground. Both are now library API.

* `QuantumState::restrict_inner(ceiling)` — the projector `P_N` (unit-tested for
  idempotence, self-adjointness and exactness of the occupation cut, including a
  two-universe component where the total is neither universe's count).
  `fock_sirk::truncated_matvec` compresses on both sides and
  `fock_sirk::solve_forward_sirk_truncated` runs SIRK on `H_N` (projecting the
  start first).
* `certificate::certify_spectrum` — the **exclusion witness**. For an explicit
  `n × n` Hermitian matrix it puts a Parlett interval around every computed
  eigenpair and merges touching intervals into disjoint clusters. Since the
  matrix has `n` eigenvalues and every interval holds at least one, each cluster
  holds exactly its own multiplicity; so a cluster's lower edge is rigorous, and
  `second_eigenvalue_lower_bound()` is a rigorous `λ₂` bound (`None` for a
  degenerate ground, where `λ₂ = λ₁` and no witness exists).
* `mass_gap_spec::temple_lower_bound(θ, r, β)` — the Kato–Temple inequality

  ```text
    λ₁ ≥ θ − r²/(β − θ),        valid when θ < β ≤ λ₂ and r < β − θ,
  ```

  and `certified_gap_lower_bound_with_exclusion(θᵒ₀, rᵒ, β, θᵉ₀)`, which is the
  gap statement *done right*: `λ₁ − λ₀ ≥ temple − θᵉ₀`, the even side needing no
  hypothesis at all (Rayleigh gives `λ₀ ≤ θᵉ₀`). Note what disappears from the
  assembly: both sector residuals as *widths* — one is replaced by `r²/γ` and
  the other is not needed.

Measured on `H_N` at `g = 0`, `N ≤ 6`, `m = 14` (the in-sector odd solve):

| quantity | value |
|---|---|
| `θᵒ₀` | −1.114923508 |
| `rᵒ` | 4.586e-7 |
| `β` (certified `λ₂` bound, odd block) | −0.778457120 |
| **Temple gap** `temple − θᵉ₀` | **0.189927960** |
| exact sector gap | 0.189927963 |
| Parlett-width `lo` (§6b's expression) | 0.189913148 |

So `H_N` now has a *valid* certified positive gap — 3e-9 below the exact gap,
where the old expression was `1.5e-5` below it because it charged `δᵉ`. Per
sector, Temple's slack is `6.3e-13` against Parlett's `4.6e-7`.

The guard is not a formality: in the two-level model the condition `r < β − θ`
holds exactly when the start has `α² > 1/2` ground overlap, so it *is* a ground-
resolution witness — and it refuses the §6b counterexample (θ at the second
level) rather than returning a "bound" above `λ₁`.

What this does **not** do: it certifies the truncation. The witness is a dense
`n × n` matrix (`n = 80` for the odd block at `N ≤ 6`, `202` at `N ≤ 8`), which
is available only at small `N`; production truncations are far past dense
diagonalization, and the unbounded `H` still has no lowest eigenvalue for any of
this to bound. The continuum mass gap remains unproven.

### 6d. The campaign re-run under the exclusion certificate (2026-11-10)

`fock_sirk/tests/mass_gap_exclusion_campaign.rs` runs the campaign grid
(`g ∈ {0, 0.5, 1, 2, 3}`, `N ∈ {6, 8}`, `m = 12`) through the §6c assembly, with
the odd sector's λ₂ witness from `certify_spectrum` on an explicitly built
sector. Every bound is checked against the exact spectrum of the same `H_N`.

| `g` | `N≤` | `θᵒ₀` | `rᵒ` | `θᵉ₀` | λ₀(even) | certified | exact |
|---|---|---|---|---|---|---|---|
| 0 | 6 | −1.114923508 | 5.5e-7 | — | — | **0.189924633** | 0.189927963 |
| 0 | 8 | −1.304848142 | 5.4e-3 | — | — | **0.122250754** | 0.122434177 |
| 0.5 | 6 | −1.290797126 | 1.4e-1 | −1.333866597 | −1.343008908 | −0.044087815 | 0.035042125 |
| 0.5 | 8 | −1.484935965 | 2.5e-1 | — | — | *refused* | — |
| 1 | 6 | −1.916799544 | 2.1e-1 | −1.654344005 | −2.024960357 | −0.337475258 | 0.091073200 |
| 1 | 8 | −2.347584372 | 8.5e-1 | — | — | *refused* | 0.091161224 |
| 2 | 6 | −4.356582224 | 2.4e-1 | — | — | **0.281770542** | 0.850113842 |
| 2 | 8 | −7.450580584 | 1.8 | — | −6.511130065 | −3.555041727 | — |
| 3 | 6 | −9.231412965 | 2.3e-1 | −8.296452899 | −9.214246967 | −0.960755924 | — |
| 3 | 8 | −14.163516377 | 2.8 | −11.288046902 | −13.610338101 | −4.471890878 | — |

Only 3 of 10 rows certify, and the two `g = 0` rows are the **control**: they
certify a positive gap on a theory with no continuum gap, which is the sharpest
statement of what a truncation certificate can and cannot mean.

Two facts about the failure modes are worth more than the count:

* **The exclusion is not the bottleneck on `g > 0`.** Where the row is
  valid-but-not-positive, the loss is the even side: `θᵉ₀` sits `9e-3 … 2.3` above
  the even sector's ground (the `g = 1, N ≤ 6` row is 0.37 above it), because a
  vacuum-start solve at `m = 12` has not converged on the squeezed ground while
  its Rayleigh quotient is still a valid upper bound. This is the cheapest thing
  to improve: the even side needs no witness, only a better `θᵉ₀`.
* **Where the row is refused, it is refused honestly.** `g = 0.5, N ≤ 8` and
  `g = 1, N ≤ 8` have `rᵒ` comparable to `β − θᵒ₀`, so Temple's hypothesis fails
  and the assembly returns `None` rather than a number.

#### The sparse route does not rescue large truncations

`certificate::DiagonalExclusion` is the matvec-only λ₂ bound: `H = D + V`,
`λ_i(H) ≥ λ_i(D) − ‖V‖₂`, so `β = d₂ − ‖V‖_F` excludes everything but one
eigenvalue below `β`. It is sound at any `n` (`n` matvecs) and measured:

| `N≤` | `n` | `θᵒ₀` (Lanczos, 12 steps) | `β` (diagonal split) | verdict |
|---|---|---|---|---|
| 12 | 1820 | −2.825023865 | **−552.784** | refused |

`‖V‖_F` grows like `√n` against a gap of order `10⁻¹`, so by `N = 12` the bound
is 550 below the ground. At the rows where it can be compared (`N ≤ 6`, `8`) it
is 25–275 below the dense witness's `β`, and the test asserts the ordering.

**Why no matvec-only method will do better.** A rigorous lower bound on `λ₂` is
an *eigenvalue count*: it says "exactly one eigenvalue lies below `β`."
Residuals cannot produce a count (they place *some* eigenvalue in an interval),
and matvecs cannot see the rest of the spectrum at all. The dense witness gets
the count from completeness — `n` intervals, `n` eigenvalues, disjoint clusters —
which is why it costs `O(n³)`. The one structural shortcut that suggests itself,
the truncation hierarchy, points the wrong way: `P_{N'} ⊆ P_N` and min–max make
compressions *raise* eigenvalues, so `λ₂(H_N) ≤ λ₂(H_{N'})` — a shallower
truncation bounds `λ₂` from **above**, never below (the test prints this
direction explicitly so it cannot be mistaken for a witness).

#### A trap this run exposed: `(n₀+n₁)` parity is not the reflection sector

The first version of the sweep classified states by `(n₀+n₁) % 2` and
certified `0.322254` at `g = 2, N ≤ 6` against a true gap of `0.044450` — a
soundness violation the exact check caught. The cause is not the certificate:
`R` flips modes 0↔1 times `(−1)^(n₀+n₁)`, so for `n₀ ≠ n₁` its eigenvectors are
the ± *combinations* `(|n₀,n₁⟩ ± (−1)^(n₀+n₁)|n₁,n₀⟩)/√2`, not the states
themselves. Worse, at `g > 0` the `(n₀+n₁)` parity is not even `H`-invariant,
because `B²` carries the three-operator products `A₀²A₁`, `A₀A₁²`.

The corrected construction pairs each `R`-orbit and asserts
`‖(I − P_sector) H u‖ < 1e-9‖Hu‖` for every sector vector, so a non-invariant
sector cannot pass silently. With it the numbers reproduce the independently
measured parity-resolved spectrum (e.g. `λ_odd = −2.652468280` at `g = 1,
N ≤ 8`). The `g = 0`-only helper in `qcd_mass_gap_certified.rs` is unaffected —
at `g = 0` the products are even, so that block *is* `H`-invariant, and it is a
superset of the odd sector, which is the right direction for a `λ₂` lower bound
(`λ₂(superset) ≤ λ₂(sector)`).

The correction to T6's statement has **not** yet reached the specialist bundle
(`../timepiece/unfer_contracts/`, a sibling repository).
`scripts/check-mass-gap-spec-parity` compares operative claims only, and this
wording is not one of them, so the bundle keeps the unconditional reading until
someone mirrors §6b there by hand.
