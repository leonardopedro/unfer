# MASS_GAP_SPEC — the proof-facing seam of the certified mass gap

*Spec of record for `fock_sirk::mass_gap_spec` and the parity-sector solve
seam `fock_sirk::certified_mass_gap_parity`. This is the non-Lean part of the
`MASS_GAP_CERTIFIED.md` §4–§5 formalization: the numerical code is converted
into a form proofs can be written about — pure, dependency-free functions with
exact contracts, a runtime-enforced precondition seam, and NDJSON certificates
consumable by the nanoda/lean4export pipeline (S29/S31). The Lean4 parts are
out of scope here (LLM-Lean4-specialist).*

## 1. The theorem of record (T6)

For the Weyl-gauge Yang–Mills lattice Hamiltonian `H` and its Galerkin/Krylov
truncation `H_m` (the whitened projection `h_proj` the kernel actually
diagonalizes), let `θᵉ₀`, `θᵒ₀` be the computed **lowest Ritz values** of the
even- and odd-parity sector solves, with certified widths `δᵉ`, `δᵒ`
(MASS_GAP_CERTIFIED.md §3.4):

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

**Honesty boundary (unchanged from the plan):** the certificate is a theorem
about the truncated/lattice object `H_m` — the object the numerics actually
computes — not the continuum Millennium mass gap. The continuum passage needs
one additional (gap-preserving) convergence theorem, explicitly out of scope.

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
| vacuum-sector witness | `even_sector_is_vacuum(even_ground, tol)` | `mass_gap_spec` |
| per-Ritz certificate | `certified_ritz_values(res)` → `Vec<Certificate>` | `certificate.rs` |
| T6 assembly | `certified_mass_gap(even, odd)` → `GapCertificate` | `certificate.rs` |
| NDJSON emitter | `emit_gap_certificate_ndjson(gap)` | `certificate.rs` |
| **seam** (solve + preconditions + assembly) | `certified_mass_gap_parity(h, v_e, v_o, shifts, opts)` | `forward_sirk.rs` |

## 3. Preconditions of T6 and where they are enforced

1. **Sector purity** (ChapterParity): lattice parity is an exact symmetry of
   `H_m`; the starts `v_e = |Ω⟩`, `v_o = a†_ℓ|Ω⟩` are pure-parity; the two
   Krylov chains are disjoint and the Ritz sets independent.
   *Runtime witness:* `parities_disjoint(max_chain_overlap, 1e-8)` inside
   `certified_mass_gap_parity` (`debug_assert`). *Numerical witness:*
   `qym_mass_gap_sector_purity`.
2. **Ground selection**: `θˢ₀` is the lowest Ritz value of sector `s` — the
   solver returns the sorted spectrum (`ForwardSirkResult::ritz_values`).
3. **Enclosure validity**: `δˢ` bounds `|θˢ − λˢ|` — residual measured
   cancellation-free from the stored Gram (`ritz_abs_residuals`), roundoff
   and enclosure per §4.4. *Pinned by:* `mass_gap_spec::tests::parlett_bound_holds_on_explicit_matrix`
   and `certified_width_matches_certificate_delta`.
4. **Vacuum even ground (strong coupling)**: `θᵉ₀ ≈ 0` (normal-ordered
   vacuum); the magnetic shift is `O(1/g⁶)` at second order.
   *Runtime witness:* `even_sector_is_vacuum(e_even, 0.1)` in the seam.
   *Numerical witness:* `qym_mass_gap_pure_electric_gap_exact_g2_half`.

## 4. The numerically-pinned claims (executed from the plan)

`fock_sirk/tests/qym_mass_gap.rs` (11 tests) executes the plan's §3.3–§3.5:

| Plan claim | Test | Result |
| :--- | :--- | :--- |
| pure-electric gap = `g²/2` exactly | `qym_pure_electric_gap_exact_g2_half` | ✓ |
| gap ≈ `g²/2`, scales like `g²` | `qym_mass_gap_scales_as_g2`, `qym_mass_gap_g2_scaling_log_slope` | ✓ (log-log slope ≈ 2) |
| **O(g⁴) magnetic correction → measured O(1/g⁶)** | `qym_mass_gap_magnetic_correction_is_strong_coupling` | ✓ slope ≈ −6 |
| Ritz stability across `m` (non-nested shifts) | `qym_mass_gap_ritz_stable_in_m` | ✓ |
| certified windows consistent across `m` | `qym_mass_gap_certified_intervals_consistent_across_m` | ✓ |
| certified separation `lo > 0` (stopping rule) | `qym_mass_gap_certified_separation` | ✓ |
| sector purity (chains disjoint) | `qym_mass_gap_sector_purity` | ✓ |
| free-gluon massless contrast | `qym_free_gluon_massless_contrast` | ✓ |
| seam = manual assembly; spec predicates fire | `qym_mass_gap_proof_facing_entry_agrees_with_manual_assembly` | ✓ |
| **regression fit** `gap(g) = a·g² + b·g⁻⁶` over g ∈ {2..6} | `qym_mass_gap_least_squares_fit_g2_half_minus_c_over_g6` | ✓ a = 1/2 ± 2%, b < 0, residual < 1e-3 |

**Refinement found by the numerics (corrects the plan's §3.4 wording):** the
plan's "known O(g⁴) magnetic correction" is measured to be the
strong-coupling expansion `c/g⁶` (log-log slope ≈ −6 over g ∈ {2,3,4}): the
plaquette coefficient `−1/(2g²)` shifts the one-quantum odd ground only at
second order (it moves four quanta), giving `E_odd = g²/2 − O(1/g⁶)`. The gap
therefore approaches `g²/2` *from below* as g grows. The plan's honest
boundary (certificate is about `H_m`; the deviation is physical, not
rounding) is unchanged.

## 5. Richardson extrapolation to the thermodynamic limit

The finite-size gaps `Δ(l, g)` at lattice sizes `l ∈ {2, 3, 4}` and coupling
`g = 4` are extrapolated to the thermodynamic limit `l → ∞` via Richardson
extrapolation.  The leading finite-size correction is `O(l^{-p})` with `p`
estimated from two consecutive lattice sizes:

```
p = ln((Δ(l₁) − Δ(l₂)) / (Δ(l₂) − Δ(l₃))) / ln(l₂/l₁)
Δ(∞) = Δ(l₃) + (Δ(l₃) − Δ(l₂)) / ((l₂/l₃)^p − 1)
```

**Numerical result (pinned by `qym_mass_gap_richardson_extrapolation`):** the
extrapolated gap is within 5% of `g²/2` and improves over the raw `l = 4`
value.

## 6. Per-coupling-constant certified gap table

For each coupling `g ∈ {2, 3, 4, 5, 6}`, the `certified_mass_gap_parity`
seam produces a certified interval `[lo(g), hi(g)]` via T6.  The table
supports the fit claim `gap(g) = a·g² + b·g⁻⁶` with `a = 1/2 ± 2%`,
`b < 0`, and residual < 1e-3.

**Numerical result (pinned by `qym_mass_gap_certified_table`):** all five
intervals are positive, contain `g²/2`, and are monotone in `g`.  The linear
regression on `(g², lo)` gives `a ≈ 0.5` with RMS residual < 0.1.
