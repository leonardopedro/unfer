# Verification: Fourier elimination of the derivative variables (NS / QG Faris–Lavine route)

This is the evidence base for the momentum-space construction of
`timepiece/CONSOLIDATED_PLAN.md` (the 2026‑09‑15 wave: “the derivative variables are
**eliminated by the spatial Fourier transform**, not fixed by a BRST gauge symmetry”) and for
`timepiece/DESIGN_COMPARISON_N_20260915.md` (the comparison operator `N`, one per Hamiltonian).

Reproducible with Cadabra2 (2.5.14, via the `../unfer` nix flake):

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    $C2 -q -n docs/ns_qg_fourier_elimination.cdb

## What the module checks

`docs/ns_qg_fourier_elimination.cdb` is the single engine-checked source shared by both
repositories (the same run is recorded in `timepiece/DESIGN_COMPARISON_N_20260915.md` §9). Every
check prints `0` unless noted.

* **A1–A7 — the Lagrangian determinant.** `detPoly` (first-row cofactor expansion) equals the
  Leibniz `det F` (A1); `F · cof(F)ᵀ = det(F) · 1`, so `cofPoly` is the signed cofactor matrix
  (A2); `det(t G) = t³ det G` by the t = 1 / t = 2 probe (A3); the isotropic line
  `det(t·1) = t³` and `volumePoly(t·1) = t³ − 1` (A4, A5); the expansion
  `det(1 + A) = 1 + tr A + m₂(A) + det A` (A6) and its sharpness `det(ε·1) = ε³ > 0` (A7).
  These are the input to `detPoly_pos` and the logarithmic volume square of
  `DESIGN_COMPARISON_N_20260915.md` §6.1–§6.2 — positivity of `det F` turns the cubic constraint
  into a square of a real function, which is what `FriedrichsExtension.friedrichs_extension_exists`
  consumes.
* **B1–B3 — the Lagrangian momentum substitution** `F_{ij} ⇒ i ℓ_j ξ_i`: the Piola `F`-part stays
  quadratic in the flow coordinate (B1), `det F` stays cubic (B2), and the volume square
  `(det F − 1)²` is sextic (B3).
  **Caution (added 2026‑09‑17 with B4/B5):** degree statements cannot see that the substitution
  makes `F` **rank one**, and after it B1–B3 hold *trivially* — the substituted objects are zero.
* **B4a–B5b — the degeneracy of that substitution.**  With `F = i ℓ ⊗ ξ`, all three `2 × 2` minors
  vanish (B4a–B4c), hence `cof(F) = 0` (the entire Piola pressure coupling of the material momentum
  equation dies) and `det F = 0` (B4d); the volume constraint `det F = 1` therefore collapses to the
  constant `volumePoly ↦ −1` (B5), whose square is `1` (B5b).  So the mode-wise elimination must
  **not** be applied to the deformation gradient: the determinant has to be carried as an
  independent scalar mode (the logarithmic volume square of `DESIGN_COMPARISON_N_20260915.md` §6.2,
  which *replaces* the elimination of `F` rather than complementing it).  The elimination of the
  velocity gradient `V_{ij} → i ℓ_j v_i` and of the viscous coordinate `S_i → −|ℓ|² v_i` survives,
  leaving `σ(R_i) = a_i + |ℓ|² v_i` — material acceleration plus viscosity, no pressure.
* **C1–C5 — the Eulerian substitution** `u_{i,j} ⇒ i k_j u_i`, `w_i ⇒ −|k|² u_i`:
  the residual becomes `i (k·u) u_i + q_i + ν|k|² u_i` (C1–C2), the advection alone is quadratic
  in  `u` (C3), and the divergence is linear (C4–C5). This is the *positive completion* of
  `DESIGN_COMPARISON_N_20260915.md` §5: the elimination is applied **inside the squares** — each
  substituted form is complex and **both** of its real parts are squared, so the advection is kept,
  squared (`½((k·u)u_i)²`), rather than dropped (plan of record 2026‑09‑17b) — so the
  bare cubic never appears as a one-body symbol.
* **D1–D6 — the leading symbol of `[h_E, n]`** (the §4.1 no-go). With
  `σ_h = (ξ·u)(u·k)` and `σ_n = (ξ² + u²)^{q/2}`, the Poisson bracket `{σ_h, σ_n}` is **non-zero**
  (D2, D4) and **homogeneous of degree `q + 1`** (D3, D5) — the commutator gains one degree over the
  comparison, so a comparison of order `q` cannot dominate a symbol of order 3 and the literal
  cubic route (b) has **no** admissible `N`; the squared route (a) does.
* **E1–E3 — the convolution bookkeeping.** Momentum conservation `p = k + q` (E1), the transfer
  weight `q_j` is **linear** in the momentum (E2), and the advection integrand `q_j u_j u_i` has
  degree 3 (E3). This is the momentum-space replacement for the local product
  `u_j ∂_j u_i`, i.e. the convolution
  `ℱ[u_j ∂_j u_i](Q) = (i/(2π)^{d/2}) ∫ dq  q_j Û_j(Q − q) Û_i(q)`.

## Recorded output (2026‑09‑17)

    A1 detPoly(row 0) - det(F) = 0
    A2 (F . cof(F)^T)_11 - det F = 0
    A2 (F . cof(F)^T)_21       = 0
    A3 8 det(G) - det(2 G) = 0   (homogeneous of degree 3)
    A4 det(t * 1)        = (t)**3
    A5 volumePoly(t * 1) = (t)**3-1
    A6 det(1 + A) - [1 + tr A + m2(A) + det A] = 0
    A7 det(eps * 1) = (eps)**3
    B1 Piola F-part is quadratic in xi = 0
    B2 det(i pk (x) xi): det(t xi) - t^3 det(xi) = 0
    B3 (det F - 1)^2 at F = i pk (x) xi: sextic identity check = 0
    B4a/B4b/B4c 2x2 minors of F = i pk (x) xi = 0
    B4d det F at F = i pk (x) xi = 0
    B5 volumePoly at F = i pk (x) xi = -1
    B5b (volumePoly)^2 at F = i pk (x) xi - 1 = 0
    C1/C2 Eulerian residual, check = 0
    C3 4 adv(u) - adv(2 u) = 0
    C4/C5 Eulerian divergence, check = 0
    D3 8 bracket(1) - bracket(2) (q = 2) = 0
    D5 32 bracket(1) - bracket(2) (q = 4) = 0
    E1 momentum conservation = 0
    E2 transfer weight linear in q = 0
    E3 q_j u_j u_i has degree 3 = 0

## What is an identity vs. an input

* **Pure algebra, verified above:** the determinant expansion and the Lagrangian/Eulerian
  substitutions, the degree bookkeeping of the no-go, and the convolution bookkeeping.
* **Input the Lean side still owes:** the identification of the momentum-space one-body operator
  with the eliminated Hamiltonian (dischargeable by `restrict_essentiallySelfAdjointOn` /
  `gaugeFixedSubset_esa`), the `π^{−1}` inverse on the physical sector, the finite
  energy/momentum cutoff `|k| ≤ Λ` that makes the continuous integral a bona fide operator, and
  the Faris–Lavine constants for the convolution kernel. These are the named residuals of the plan,
  not axioms.

## Files

* `docs/ns_qg_fourier_elimination.cdb` — this module (A1–E3).
* `timepiece/DESIGN_COMPARISON_N_20260915.md` §9 — the design note that consumes the A–D checks.
* `timepiece/DESIGN_COMPARISON_N_20260915.cdb` — the timepiece-local copy of the A–D subset.
