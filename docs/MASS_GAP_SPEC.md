# MASS_GAP_SPEC — the proof-facing seam of the certified mass gap

*Spec of record for `fock_sirk::mass_gap_spec` and the sector-solve seam
`fock_sirk::certified_mass_gap_parity`. This is the non-Lean part of the
`MASS_GAP_CERTIFIED.md` §4–§5 formalization: the numerical code is converted
into a form proofs can be written about — pure, dependency-free functions with
exact contracts, a runtime-enforced precondition seam, and NDJSON certificates
consumable by the nanoda/lean4export pipeline (S29/S31). The Lean4 parts are
out of scope here (LLM-Lean4-specialist).*

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
| squeezed ground, not the Fock vacuum (E₀<0; R-odd at strong coupling) | `qym_gauge_fixed_ground_is_squeezed_not_fock_vacuum` | ✓ |
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

`fock_sirk/tests/qcd_mass_gap_certified.rs` (3 tests) instantiates the T6
assembly on the gauge-fixed H: the certified window `[lo, hi]` encloses the
exact truncated gap `E₁ − E₀` for `g ∈ {1, 2}` across truncations, and the
NDJSON emitter produces well-formed sector certificates whose
`certified_positive` flag truthfully reflects `lo > 0` of the emitted numbers
(the pinned Lean4 T6 instance in `../timepiece/GapCertificate/` remains the
historical lattice instantiation; regenerating it for the gauge-fixed numbers
is an LLM-Lean4-specialist item).
