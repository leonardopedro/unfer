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
(`g ∈ {0, 0.5, 1, 2, 3}`, `N ∈ {6, 8}`, `m = 12` — widened to `N ≤ 10` below)
through the §6c assembly, with the odd sector's λ₂ witness from
`certify_spectrum` on an explicitly built sector. Every bound is checked against
the exact spectrum of the same `H_N`.

| `g` | `N≤` | `θᵒ₀` | `rᵒ` | `θᵉ₀` | λ₀(even) | certified | exact |
|---|---|---|---|---|---|---|---|
| 0 | 6 | −1.114923508 | 5.5e-7 | −1.304848142 | −1.304851471 | **0.189924633** | 0.189927963 |
| 0 | 8 | −1.304848142 | 5.4e-3 | −1.427251240 | −1.427285648 | **0.122250754** | 0.122434177 |
| 0.5 | 6 | −1.290797126 | 1.4e-1 | −1.333866597 | −1.343008908 | −0.044087815 | 0.035042125 |
| 0.5 | 8 | −1.484935965 | 2.5e-1 | −1.462673600 | −1.545550924 | *refused* | — |
| 1 | 6 | −1.916799544 | 2.1e-1 | −1.654344005 | −2.024960357 | −0.337475258 | 0.091073200 |
| 1 | 8 | −2.347584372 | 8.5e-1 | −1.802710254 | −2.743629504 | *refused* | 0.091161224 |
| 2 | 6 | −4.356582224 | 2.4e-1 | −4.694324172 | −5.217666264 | **0.281770542** | 0.850113842 |
| 2 | 8 | −7.450580584 | 1.8 | −5.064921303 | −6.511130065 | −3.555041727 | — |
| 3 | 6 | −9.231412965 | 2.3e-1 | −8.296452899 | −9.214246967 | −0.960755924 | — |
| 3 | 8 | −14.163516377 | 2.8 | −11.288046902 | −13.610338101 | −4.471890878 | — |

Only 3 of 10 rows certify, and the two `g = 0` rows are the **control**: they
certify a positive gap on a theory with no continuum gap, which is the sharpest
statement of what a truncation certificate can and cannot mean.

Two facts about the failure modes are worth more than the count (both were
confirmed at `N ≤ 10` in the widened grid below):

* **The exclusion is not the bottleneck on `g > 0`.** Where the row is
  valid-but-not-positive, the loss is the even side: `θᵉ₀` sits `9e-3 … 2.3` above
  the even sector's ground (the `g = 1, N ≤ 6` row is 0.37 above it), because a
  vacuum-start solve at `m = 12` has not converged on the squeezed ground while
  its Rayleigh quotient is still a valid upper bound. This is the cheapest thing
  to improve — the even side needs no witness, only a better `θᵉ₀` — and doing
  it is measured next: **6 of 10** rows then certify.
* **Where the row is refused, it is refused honestly.** `g = 0.5, N ≤ 8` and
  `g = 1, N ≤ 8` have `rᵒ` comparable to `β − θᵒ₀`, so Temple's hypothesis fails
  and the assembly returns `None` rather than a number.

#### The solve, not the witness, was the limit (2026-11-10b)

`θᵉ₀` is consumed only as an upper bound on `λ₀`, so the solve that produces it
carries no hypothesis. Replacing the stagnating truncated SIRK chain with
matvec-only Lanczos on the *same* `H_N` (`certificate::lanczos_ritz`, 60 steps,
same sector starts, containment residual asserted) changes nothing about
validity and everything about the verdicts. On the cells whose ordering is T6's
own:

| `g` | `N≤` | `θᵉ₀` (excess over λ₀(even)) | `θᵒ₀` | `rᵒ` | certified | exact |
|---|---|---|---|---|---|---|
| 0 | 6 | −1.114923508 (0) | −1.114923508 | 2.0e-15 | **0.189927963** | 0.189927963 |
| 0 | 8 | −1.304851471 (0) | −1.304851471 | 7.3e-15 | **0.122434177** | 0.122434177 |
| 0 | 10 | −1.512899036 (0) | −1.427285648 | 1.1e-14 | **0.085613388** | 0.085613388 |
| 0.5 | 6 | −1.343008908 (0) | −1.307966783 | 5.4e-15 | **0.035042125** | 0.035042125 |
| 0.5 | 10 | −1.835640721 (0) | −1.799299493 | 2.8e-11 | **0.036341228** | 0.036341228 |
| 1 | 6 | −2.024960357 (0) | −1.933887157 | 1.5e-14 | **0.091073200** | 0.091073200 |
| 1 | 8 | −2.743629504 (0) | −2.652468280 | 2.6e-14 | **0.091161224** | 0.091161224 |
| 2 | 6 | −5.217666264 (0) | −4.367552423 | 2.2e-14 | **0.850113842** | 0.850113842 |
| 2 | 10 | −10.244687197 (0) | −9.022852588 | 7.3e-14 | **1.221834608** | 1.221834608 |
| 3 | 10 | −20.121348034 (0) | −18.543774653 | 1.2e-13 | **1.577573381** | 1.577573381 |

**Every one of these bounds is the exact gap to the last printed digit** — with
both sector grounds resolved, the only slack left is Temple's `r²/(β − θ)`,
which is `O(1e-14)` — and the count goes 3 of 10 rows (SIRK) → 6 of 10
(Lanczos, `N ≤ 8`) → **10 of 15** on the widened grid, with the other 5 cells
certified in the exchanged orientation below, for **15 of 15**.

The remaining rows are not a convergence problem. They are the cells where the
odd ground sits *below* the even one inside `H_N`, so `θᵒ₀ − θᵉ₀` is the
**negative of the truncated gap**, to machine precision:

| `g` | `N≤` | λ₀(even) | λ₁(odd) | `H_N` gap (even − odd) | T6 bound |
|---|---|---|---|---|---|
| 0.5 | 8 | −1.545550924 | −1.576008527 | 0.030457603 | −0.030457603 |
| 1 | 10 | −3.340446626 | −3.613783714 | 0.273337089 | −0.273337089 |
| 2 | 8 | −6.511130065 | −7.754776632 | 1.243646566 | −1.243646566 |
| 3 | 6 | −9.214246967 | −9.236287521 | 0.022040554 | −0.022040554 |
| 3 | 8 | −13.610338101 | −14.566593782 | 0.956255682 | −0.956255682 |

T6's orientation (odd above even) does not hold in those cells at all, and no
amount of accuracy can make the expression positive there. The test asserts both
halves of that: every cell whose truncated ordering matches T6's certifies, and
the certified count equals the number of such cells — so the orientation, not
convergence, is the limit on those rows.

Two findings worth keeping:

* **The truncated chain stagnates, and `m` does not fix it.** At `g = 1, N ≤ 6`
the even excess is `3.71e-1` at `m = 12`, `3.97e-1` at `m = 24` and `3.99e-1` at
`m = 40`, with the residual flat at `~8e-1`: the chain's span stops growing (the
`rank 10 of m+1 = 15` collapse already recorded in `qcd_mass_gap_certified.rs`),
so extra shifts buy nothing. This is a property of the forward truncated solve as
used here, not of the certificate — the SIRK rows remain *valid*, merely loose.
* **Lanczos belongs on the even side** precisely because it needs no witness:
60 matvecs reach `rᵒ ≤ 3e-11` at every cell (`≤ 1.2e-13` on all but one), where
40 SIRK steps leave `~8e-1`. The odd side is where the witness is needed, and
the odd side was already converging under SIRK at these sizes.

At `N ≤ 10` the same 40 steps are no longer enough: the even-side residual rises
to `8.5e-7` and the bounds stop agreeing with the exact gaps to the last digit
(`0.036340485` against `0.036341228`). Raising the cap to 60 restores
last-digit agreement at every cell, which is why the constant is 60 and the
route is labelled accordingly — the cost is 20 matvecs, and the alternative was
weaker bounds dressed as the same claim.

#### The other orientation: the same theorem, the sectors exchanged (2026-11-10c)

`λ₁(H_N) − λ₀(H_N)` is a gap in one direction only. In the crossing cells the
odd ground is the lower one, so that expression is negative and no accuracy can
rescue it — but Temple's inequality is *sector-local*, and nothing in it prefers
the odd sector. Exchanging the roles certifies the other quantity,
`λ₀(even) − λ₁(odd)`:

```text
  λ₀(even) ≥ temple(θᵉ₀, rᵉ, β_even)      λ₁(odd) ≤ θᵒ₀   (Rayleigh–Ritz)
  ⟹ λ₀(even) − λ₁(odd) ≥ temple(θᵉ₀, rᵉ, β_even) − θᵒ₀.
```

It is the same call with the arguments exchanged, and the odd side then needs no
witness at all. The only extra work is one more `certify_spectrum` pass, on the
even block, for `β_even ≤ λ₂(even)`:

| `g` | `N≤` | `θᵉ₀` | `rᵉ` | `λ₂(even)` | `β(even)` | certified⇄ | exact⇄ |
|---|---|---|---|---|---|---|---|
| 0.5 | 8 | −1.545550924 | 1.7e-14 | −1.445385952 | −1.445386066 | **0.030457603** | 0.030457603 |
| 1 | 10 | −3.340446626 | 6.7e-11 | −2.614147195 | −2.614149342 | **0.273337089** | 0.273337089 |
| 2 | 8 | −6.511130065 | 5.1e-14 | −5.396349662 | −5.396350050 | **1.243646566** | 1.243646566 |
| 3 | 6 | −9.214246967 | 3.6e-14 | −5.126287367 | −5.126287397 | **0.022040554** | 0.022040554 |
| 3 | 8 | −13.610338101 | 8.8e-14 | −11.921429604 | −11.921430269 | **0.956255682** | 0.956255682 |

All five crossing cells certify, each bound equal to the exact gap to the last
printed digit, so **15 of 15 grid cells now carry a certified gap** — with the
sector that owns the ground reported beside it instead of assumed. In T6's own
cells the swapped bound is instead exactly the negative of the same number
(`−0.189927963` at `g = 0, N ≤ 6`), so the two orientations are complements; the
test asserts that each orientation certifies exactly the cells whose ordering
matches it, and that the two counts sum to the grid.

Two things this does **not** say.

* It is not fifteen rows of one theorem. The statement differs by cell —
  `λ₁(odd) − λ₀(even)` where the even state is lower, `λ₀(even) − λ₁(odd)` where
  the odd one is — so the paper's gap is certified on 10 cells and the exchanged
  one on 5.
* In a crossing cell the bound is the gap of `H_N` only because no odd level
  lies between the two grounds. That is asserted per cell (`λ₂(odd) >
  λ₀(even)`), which is what makes the two sector grounds the two lowest levels
  of the truncation. What a crossing cell means for the vacuum is a separate
  question, and the widened grid answers it in a way the `N ≤ 8` grid could not
  — see the next subsection.

#### What a crossing cell does *not* say: the ground parity is not monotone in N

It is tempting to read a crossing cell as "`N` is too small: the truncated ground
is odd, push `N` and the even state becomes the ground again". The widened grid
refutes that reading. Taking all fifteen cells, the crossings sit at:

| `g` | `N≤ 6` | `N≤ 8` | `N≤ 10` |
|---|---|---|---|
| 0 | — | — | — |
| 0.5 | — | **crossing** | — |
| 1 | — | — | **crossing** |
| 2 | — | **crossing** | — |
| 3 | **crossing** | **crossing** | — |

The crossing set moves *non-monotonically*: `g = 2` crosses at `N ≤ 8` and not at
`N ≤ 10`; `g = 1` does not cross at `N ≤ 8` but does at `N ≤ 10`; `g = 3` crosses
at 6 and 8 and not at 10. So the parity of the truncated ground is not settling
with depth — it is an oscillation of the truncation itself, of the same order as
the energy differences involved (`0.022 … 1.24`).

What follows, and what does not:

* **The certificate is unaffected.** Each row still states the gap it has, in
the orientation that holds, and every bound is still checked against the exact
spectrum. Non-monotonicity costs no soundness.
* **The vacuum's parity cannot be read off a truncation.** A cell where the odd
state is lower is not evidence of an odd vacuum, and a cell where it is not is
not evidence of an even one: the same `g` gives both answers at different `N`.
  Only the sector-local statements survive, which is precisely why the
  certificate is phrased sector-locally.
* **Which sizes are unusable for the paper's statement**: on this grid, the 5
  cells above cannot support `λ₁(odd) − λ₀(even) > 0` at any accuracy — they are
  `(g, N) = (0.5, 8), (1, 10), (2, 8), (3, 6), (3, 8)`. In particular the whole
  `g = 3` low-`N` range is unusable in T6's orientation, and `g = 1` becomes
  unusable exactly when the truncation is deep enough for the odd sector to drop
  below the even one (`N ≤ 10`).
* **The `g = 0` column is the control and it never crosses**: at `g = 0` the
  even ground is the lower one at every `N`, while every interacting column
  crosses somewhere. Whatever moves the odd sector, it is turned on by `g`; the
  measurements here locate the crossings but do not attribute them, and the test
  asserts only the measured pattern.

#### The sparse route does not rescue large truncations

`certificate::DiagonalExclusion` is the matvec-only λ₂ bound: `H = D + V`,
`λ_i(H) ≥ λ_i(D) − ‖V‖₂`, so `β = d₂ − ‖V‖_F` excludes everything but one
eigenvalue below `β`. It is sound at any `n` (`n` matvecs) and measured:

| `N≤` | `n` | `θᵒ₀` (Lanczos, 12 steps) | `β` (diagonal split) | verdict |
|---|---|---|---|---|
| 12 | 1820 | −2.825023865 | **−552.784** | refused |

`‖V‖_F` grows like `√n` against a gap of order `10⁻¹`, so by `N = 12` the bound
is 550 below the ground. At the rows where it can be compared (`N ≤ 6`, `8`) it
is 25–275 below the dense witness's `β`, and the test asserts the ordering. The
count-free family is closed there; the *count* is reachable cheaply, which is
what `certificate::inertia_below` shows in the subsection after the next one.

#### One count-free bound measured, and why the family cannot reach the window

`certificate::MatchingExclusion` is the tightest count-free witness the crate
can build. Colour the sparsity graph's edges into **matchings** `V = Σ_c V_c`
(greedy edge colouring, so `C ≤ max degree`): each `V_c` is then a disjoint
union of `2×2` blocks `[[0,v],[v,0]]`, and `λ_min(V_c) = −max |H_e|` in closed
form. Weyl applied class by class gives `λ₂ ≥ d₂ − Σ_c max |H_e|`. This trades
the Frobenius split's `√(#edges)` for `#classes`, so the slack is degree-bounded
instead of `√n`-bounded:

| `N≤` | `n` | classes | colouring slack | `‖V‖_F` | `β` colouring | `β` diagonal |
|---|---|---|---|---|---|---|
| 6 | 80 | 13 | 42.7 | 34.5 | −41.74 | −33.49 |
| 8 | 200 | 17 | 92.6 | 86.4 | −91.64 | −85.39 |
| 12 | 1820 | 28 | 313.5 | 553.8 | −312.49 | −552.78 |

The growth cuts both ways, which is the measured finding. At the two smaller
sizes the colouring is *looser* than one Frobenius term — charging the largest
entry of each of 13 classes beats the `l2` norm of ~600 edges only once the
graph is dense enough — and by `N ≤ 12` it is 1.8× tighter: the first
count-free bound whose slack does not explode with `n`. It is still nowhere near
a witness, and the obstacle is structural rather than a matter of a cleverer
split: `d₂ = 1` on this operator while `θ ≈ −2.8 … −1.9`, so the entire
off-diagonal budget is `d₂ − θ < 4`, whereas any splitting bound charges at
least one entry per class against a degree of 13–28. Making the charge smaller
means fewer, larger classes — which needs `λ_min` of a large block, i.e. the
eigenvalue problem again.

So the measured answer to "is there a `λ₂` witness in the splitting family?" is
**no, at any `N`** on this operator (the campaign's assertions on the diagonal
split, the matching split and the `(θ, λ₂]` window are what hold that in place).
The count itself, obtained without a dense diagonalization, does exist — see the
next subsection.

#### The count without a diagonalization (2026-11-10d)

`certificate::inertia_below` gets the count from **Sylvester's law of inertia**:
`H − βI = L D Lᵀ` with `L` unit triangular, so the number of negative pivots of
`D` is exactly the number of eigenvalues below `β`. Every entry carries a bound
on its distance from the exact value, and a pivot's sign is used only when
`|value| > error` — so the count is rigorous in floating point or it refuses. The
elimination is sparse, with a selectable fill-reducing ordering
(`Ordering::ReverseCuthillMckee` or `Ordering::NestedDissection`) and 1×1
symmetric pivoting (a permutation preserves inertia), at `O(fill)` cost.

The bracket the campaign builds from it has no diagonalization anywhere:
`temple(θᵒ₀, rᵒ, β_odd) − θᵉ₀ ≤ λ₁(odd) − λ₀(even) ≤ θᵒ₀ − temple(θᵉ₀, rᵉ, β_even)`,
which needs no orientation assumption:

| `N≤` | sector `n` | fill | count(β) | lower | upper | exact | secs |
|---|---|---|---|---|---|---|---|
| 6 | 80 | 27 | 1/1 | 0.091073200 | 0.091073200 | 0.091073200 | 0.00 |
| 8 | 200 | 49 | 1/1 | 0.091161224 | 0.091161224 | 0.091161224 | 0.02 |
| 10 | 420 | 78 | 1/1 | −0.273337089 | −0.273337089 | −0.273337089 | 0.07 |
| 12 | 784 | 116 | 1/1 | 0.273346184 | 0.273346184 | — | 0.15 |
| 14 | 1344 | 164 | 1/1 | −0.112820857 | −0.112820857 | — | 0.36 |
| 16 | 2160 | 223 | 1/1 | −0.060551502 | −0.060551502 | — | 0.97 |

**This is the answer to "is the dense `O(n³)` the price of the witness?": no.**
The count is decided at every size through `N ≤ 16` (`n = 2160`) in under a
second, the bracket closes to all printed digits, and wherever the dense route
can still run (`N ≤ 10`) the bracket equals the exact gap. The fill grows to 223,
nowhere near the dimension, so the elimination — not the count — is the cheap
part.

**And the campaign's own rows now run on it.** The 15-cell grid no longer calls
`certify_spectrum` at all: `β` comes from a bisection on the count — start at
`θ + r` (Temple's own ball), grow geometrically until the count says `λ₂` has
been passed, then 30 bisection steps to the largest deciding `β`, which is the
one that minimises Temple's `r²/(β − θ)`. The verdicts are *identical* to the
dense-`β` run, to the printed digits: SIRK `m = 12` 4 certified / 5 valid-but-
not-positive / 6 refused, Lanczos 60 10 / 5 / 0, and `+5` in the exchanged
orientation — **15 of 15**. The bounds themselves are unchanged (`0.085613388`,
`0.036341228`, `1.221834608`, …), which is the check that the count produced the
same `λ₂` the diagonalization did rather than merely a usable one. `O(n³)` is
therefore not on the certification path anywhere on this grid; the dense
spectrum survives in the test only as the *oracle* every bound is checked
against.

Two limits, both measured rather than smoothed over:

* **The margin matters.** With `β` a hair (`1e-9`) above `θ` the shifted matrix
  has an eigenvalue that close to `β`, one elimination pivot is `~1e-4` against
  an error of `~5e-4`, and the sign is undecidable at the *last* index — where
  there is no tail left to pivot into. The campaign therefore probes a ladder of
  margins (`1e-9 … 1e-2`) and takes the first decided count; any margin whose
  count comes out as one is a valid witness, and one that overshoots `λ₂` comes
  out as two and is rejected rather than used.
* **The count is per sector, and consistency is checked by the bracket itself.**
  `inertia_below` is validated against the exact spectrum at `N ≤ 10` at three
  shifts per sector, and on a corpus of random symmetric matrices in three
  regimes that each broke an earlier version — including congruent scaling over
  nine orders of magnitude and structurally-zero diagonals, where the `−β` shift
  was silently dropped once and made the count wrong by exactly one.
* **What this does not license.** It certifies `H_N`'s gap at `N ≤ 16`; it says
  nothing about the continuum, and the crossing pattern at `N = 14, 16` is
  consistent with the non-monotonicity above (at `g = 1` the odd ground is lower
  at `N = 10, 14, 16` and higher at `6, 8, 12`).

##### Where it stopped, and why it was the arithmetic (`N ≤ 33`, 2026-11-11)

The counter was driven past its working range to find out what fails first. Odd
sector at `g = 1`, `β = θ₀ + 0.5`, both orderings on the same matrix
(`nested_dissection_cuts_the_fill_of_the_inertia_count`):

| `N≤` | `n` | fill RCM | fill ND | secs RCM | secs ND |
|---|---|---|---|---|---|
| 6 | 80 | 18 | 16 | 0.00 | 0.00 |
| 8 | 200 | 33 | 31 | 0.00 | 0.00 |
| 10 | 420 | 58 | 52 | 0.00 | 0.00 |
| 12 | 784 | 93 | 68 | 0.01 | 0.00 |
| 14 | 1344 | 138 | 108 | 0.03 | 0.01 |
| 16 | 2160 | 192 | 172 | 0.09 | 0.05 |
| 18 | 3300 | 258 | 226 | 0.27 | 0.15 |
| 20 | 4840 | 337 | 315 | 0.76 | 0.37 |
| 22 | 6864 | 429 | 395 | 1.95 | 0.99 |
| 24 | 9464 | 534 | 485 | 4.75 | 2.24 |

**The matrix was never the problem.** The same elimination in plain `f64`, with
no bounds at all, decided every one of these sizes: at `N ≤ 18, 20, 22` the
smallest pivot was `6.162, 7.147, 7.491` and the largest row entry `2.6e2, 1.8e2,
8.2e1` — no near-singular pivot anywhere, and one eigenvalue below `β` at each
size, matching the dense count. What failed was the *bound*. Traced per step at
`N ≤ 20`, the pivots stayed in `10…38`, every multiplier below `0.6`, and no
stored value above `1e2`; the tracked error alone went `1e-14 → 1e-6` (j≈3250)
`→ 1.96e3` (j≈3750), and at `N ≤ 22` reached `4.1e16` by j = 6250. Each step was
allowed to inject `u ≈ 1.1e-16` of its own value, and the elimination multiplies
that charge along the fill — the classical bound grows linearly in the operation
count, this one compounded.

**The fix is arithmetic, not ordering.** [`certificate::Bounded`] now carries its
value as an *unevaluated sum of two `f64`s* — `hi + lo`, renormalized after every
operation through the exact transforms (`two_sum`, `quick_two_sum`, and `two_prod`
via `fma`). `add`/`sub`/`mul`/`div` are the double-double versions; each folds in
only the `8u²·|value|` its own renormalization can leave, so one operation injects
`u² ≈ 1.2e-32` instead of `u`. The interval bookkeeping and the decision rule are
untouched: the count still refuses unless `|centre| − error` has a definite sign.
The same trace at `N ≤ 22` now ends with a tracked error of `7.6e-15` against
pivots of `10…38`, and the largest bound-to-pivot ratio anywhere in the grid is
`4.7e-10` (RCM) — the count is no longer near its limit.

The extra arithmetic is invisible in the timings: the elimination is dominated by
`BTreeMap` insertion into the fill, not by the per-entry flops, so `N ≤ 22` costs
`1.95 s` where the one-term value cost `2.16 s`.

**Nothing in this grid refuses any more, and the reach is the same bound, far
out.** `the_count_reaches_past_the_grid` (same measurement, same assertions,
`#[ignore]`d because the three truncations cost ~170 s) reaches

| `N≤` | `n` | fill RCM | secs RCM | bound/pivot RCM | fill ND | secs ND | bound/pivot ND |
|---|---|---|---|---|---|---|---|
| 26 | 12740 | 652 | 12.0 | 1.6e-9 | 653 | 4.6 | 1.2e-22 |
| 28 | 16800 | 798 | 32.9 | 1.2e-6 | 793 | 9.6 | 3.5e-21 |
| 30 | 21760 | 961 | 85.5 | 2.0e-2 | 927 | 19.3 | 2.9e-19 |

with both orderings still counting the same inertia (`below == 1` at every one
of them, under either ordering — so this table *is* a `λ₂` witness), and
`the_count_refuses_past_the_reach` (also `#[ignore]`d, ~400 s) pins where it
stops:

| `N≤` | `n` | RCM | ND |
|---|---|---|---|
| 32 | 27744 | **refused** (216 s) | decides, fill 1096, 39 s, ratio 7.7e-17, `below 3` |
| 33 | 31161 | — | decides, fill 1212, 52 s, ratio 8.0e-13, `below 4` |
| 34 | 34884 | refused | **refused** (85 s) |
| 36 | 43320 | refused | refused (329 s) |

So the last deciding sizes are `N ≤ 30` (RCM) and `N ≤ 33` (nested dissection),
and the refusals are still the *bound*, not the clock: at `N ≤ 34` nested
dissection gives up 725 pivots from the end, at the same index (`j = 34159`)
whether or not block pivots are available, where a trace of the trailing block
shows relative errors from `0` (median `3.3e5`) to `1e15` — the last dense
separator's own elimination, not one small pivot. RCM's ratio climbs from
`2.0e-2` at `N ≤ 30` to past one at `N ≤ 32`; nested dissection's sits at
`7.7e-17` at `N ≤ 32` and collapses between there and `34`, which is why the two
orderings now have different walls. (`N ≤ 30` was first measured at `3.7e-4` on
a basis that the harness bug below had left unnormalized — the fixed value is
`2.0e-2`, and every number above is from runs after that fix.)

**Two ranges, not one.** `below == 1` through `N ≤ 30`, so that is where the
count certifies the gap; at `N ≤ 32, 33` it still *decides* but reports three and
four eigenvalues below `β` — the 60-step Ritz value is no longer tight enough for
`θ₀ + 0.5` to separate `λ₁` from `λ₂`, so those sizes measure the counter's
numerical reach, not a gap. The honest statement is that the reach was pushed
from `N ≤ 18` to `N ≤ 33` and the witness range to `N ≤ 30`, and that what ends
the reach is the same accumulated rounding the two-term value made negligible at
the grid's sizes.

**2×2 block pivots: right, and not the lever.** The counter now falls back to a
`2×2` block pivot when no diagonal in the trailing block has a decidable sign —
a coupled pair can have a decidable determinant while neither diagonal does, and
its inertia follows from `det` and the trace (`[[ε,1],[1,−ε]]` is the smallest
case; `[[0,1],[1,0]]` used to be refused outright and is now counted as one
negative). Both the block decision and the block Schur update are tested
(`the_sparse_count_decides_a_coupled_pair_with_no_decidable_diagonal`,
`the_block_update_reaches_the_trailing_rows`). It does **not** move the wall:
asked at `N ≤ 34`, the pair search finds no determinantal sign either, and the
refusal still lands on index `34159` — by that point every entry of the trailing
block has an interval containing zero, so products of them are undecidable too.
The wall is the whole block's condition, not the diagonals' availability.

**Symmetric equilibration buys none of it.** `D(H − βI)D` with `D = diag(2ᵏ)`,
`k` halving each diagonal's exponent, is an exact congruence — every multiplication
is by a power of two, so the inertia and the error model carry over unchanged and
the count must come out identical — so it is the one preconditioning that costs
nothing to try. Measured (`N ≤ 24…30`, both orderings, count unchanged at every
size): the bound-to-pivot ratios are *identical* to the printed digits (RCM
`4.689e-10 → 4.689e-10` at `N ≤ 24`, `1.248e-6 → 1.248e-6` at `N ≤ 28`; nested
dissection `5.333e-21 → 5.333e-21`, `3.523e-21 → 3.523e-21`). The growth is in
the update chain's own rounding, not in the scaling of the matrix, so no diagonal
preconditioning reaches it.

**A harness bug worth recording.** Reaching past `N ≤ 30` first failed in the
*basis builder*, not the count: `fock_state` normalized by
`scale_and_add(1/‖ψ‖ − 1)`, and the creation chain's `‖ψ‖ = √(n!)` passes `2⁵³`
at occupation 30 (`√(30!) ≈ 1.6e16`), where that coefficient rounds to exactly
`−1` and annihilates the state. Occupations `≥ 30` produced empty states, which
silently capped every `N` this harness could build at 30. The amplitudes are now
divided out directly; nothing above `N ≤ 28` could be measured before that.

**Ordering: smaller fill, and now a longer reach.** Nested dissection cuts the
fill by 6–27 % wherever the grid decides (18→16, 93→68, 258→226, 534→485) and is
2–2.5× faster at the top of the grid, which is why it is the parameter it is.
Past the grid the fill advantage fades (652 vs 653 at `N ≤ 26`, 798 vs 793 at
`N ≤ 28`, 961 vs 927 at `N ≤ 30`) — which is why the fill comparison is asserted
only on the campaign's own sizes — but the *bound* advantage does not fade: it is
what lets nested dissection decide two sizes further than RCM, so the reach test
asserts both orderings decide with the same count wherever they both decide.

**A bug the reach test found, recorded because it had been measuring the wrong
matrix.** The nested-dissection permutation check is a `debug_assert`, so release
runs never saw that the recursive cut *dropped the component it had just walked*:
`60 of 80` vertices fell out of the order already at `N ≤ 6` (the disconnected
branch ordered every component except the level-BFS one). Every nested-dissection
fill in the previous revision of this section was therefore measured on a
mangled matrix, and its `N ≤ 16` refusal was an artifact. The fix orders the
walked component with the others, and
`nested_dissection_orders_every_vertex_of_a_split_graph` now checks the
permutation in release, where the campaign runs.

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
N ≤ 8`). The `g = 0`-only helper `truncated_parity_block` in
`qcd_mass_gap_certified.rs` is unaffected — at `g = 0` the products are even, so
that block *is* `H`-invariant and coincides with the odd sector.

It now **refuses** to run at `g ≠ 0`: its signature takes the coupling, a
`#[should_panic]` test pins the refusal, and the `H`-invariance of the block it
returns is measured (`‖(1−P) H P‖_F < 1e-10`) rather than assumed — so the guard
is a property of the matrix, not a caller convention. Without it, reusing the
helper at `g > 0` would hand a certificate a `β` that is not a `λ₂` witness,
which is exactly how this run's `0.322254` violation would have entered as a
"result".

§6a–§6d are mirrored into the specialist bundle
(`../timepiece/unfer_contracts/MASS_GAP_SPEC.md`, §6a and §6b there), and
`scripts/check-mass-gap-spec-parity` anchors the operative claims of both —
including the conditionality of T6, the Temple form, the campaign verdicts, the
exchanged orientation and the measured count-free witness family, and the parity
trap — so the bundle can no longer drift back to the unconditional reading (or
back to "6 is the ceiling") unnoticed.
