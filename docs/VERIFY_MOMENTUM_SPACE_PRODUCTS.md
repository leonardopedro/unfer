# Verification: momentum-space products — convolutions, derivative symbols, the quartic → quadratic composite rewriting

This is the evidence base for `docs/momentum_space_products.cdb`, the
commutative (c-number) bookkeeping engine behind the reader-facing page
`../test/mathematical-foundations/momentum-space-products.md`. The module
extends the sibling `ns_qg_fourier_elimination.cdb` (whose §E pins the *linear*
convolution bookkeeping: momentum conservation, weight linearity, degree 3)
with the three facts a proof of the quartic rewriting has to carry:

1. a derivative at a point becomes a **polynomial symbol on one momentum
   leg** of the convolution, not a function of the total momentum;
2. a product of fields at one point in space is the convolution of their
   transforms, with the total-momentum delta spelled by the index formula
   `ψ_Q = Σ_p û_p û_{Q−p}` (the S47 convention `f ⋆ g[n] = Σ_m f[m] g[(n−m)]`);
3. a **quartic** expression in the field at one point of space equals a
   **quadratic** expression in the composite `ψ` at *different* points of
   momentum space — coefficient by coefficient,
   `[F⁴]_R = Σ_Q ψ_Q ψ_{R−Q}`, and at zero total momentum,
   `[F⁴]_0 = Σ_Q ψ_Q ψ_{−Q}`.

Reproducible with Cadabra2 (nix flake; 2.5.15 on `PATH` on this machine):

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    cadabra2-cli -q -n docs/momentum_space_products.cdb

## What the module checks

The toy model is one-dimensional with three modes `p ∈ {−1, 0, +1}` (out-of-grid
modes are 0) and `F = û_{−1}E_{−1} + û_0 E_0 + û_1 E_{+1}`, where the letter
labels obey the momentum table `E_a E_b ↦ E_{a+b}` (the table *is* the
delta bookkeeping; every product below carries exactly two `E` factors, so one
rule application normal-orders the conservation law). Each CHECK compares an
expression built by expanding a pointwise product against the same quantity
built independently from the convolution index formula, so a wrong table entry
or a wrong index sign fails the row, and every size is pinned so a collapse
fails the label:

* **CHECK 1–2 — product → convolution.** The pointwise square collects to 6
  monomials, the convolution side `Σ_Q ψ_Q E_Q` to the same 6; difference 0.
* **CHECK 3–4 — the derivative is a symbol on the leg.** With the rule
  `û_p ↦ i p û_p`, `F · ∂F` equals
  `Σ_{Q,q} û_{Q−q} (i q) û_q E_Q` (difference 0, 4 terms per side).
* **CHECK 5 — leg ≠ total (expect nonzero).** The `E_1` coefficient is the
  leg-weighted `i û_0 û_1`; weighting the composite `ψ_1` by the *total*
  momentum `i·Q` gives twice that. The printed difference `−Ep i uo up` is the
  guard: the symbol rides on the integration leg.
* **CHECK 6 — completion of squares.** `2⟨p,q⟩ − |p+q|² + |p|² + |q|² = 0`, i.e.
  `(ip)·(iq) = |p|²/2 + |q|²/2 − |p+q|²/2` — the transfer identity that splits a
  derivative weight on two legs into one-body leg squares plus a
  total-momentum square (the composite's own weight).
* **CHECK 7–10 — the quartic at one point = the quadratic in the composite.**
  `F⁴` (quartic side 21 → 15 monomials under the table) equals
  `Σ_Q ψ_Q ψ_{R−Q}` per total momentum `R` (pairing side 15), difference 0;
  and the zero-mode coefficient `[F⁴]_{E_0}` (3 monomials) equals the pairing
  `Σ_Q ψ_Q ψ_{−Q}`, difference 0. This is the exact statement of the idea:
  *quartic in the elementary modes at one spatial point = quadratic in the
  composite at the momentum-point pair (Q, R−Q)*.
* **CHECK 11–13 — degree probes.** `ψ` is quadratic in `û`
  (`4ψ(1) − ψ(2) = 0`), and both the pairing and `F⁴` are degree 4
  (`16P(1) − P(2) = 0`, `16F⁴(1) − F⁴(2) = 0`): the rewriting changes the
  operator order (quartic → quadratic × quadratic) without changing the degree
  in the amplitudes.
* **CHECK 14 — index convention (expect nonzero).** On an
  inversion-asymmetric `û`, `Σ_p û_p û_{1−p}` and `Σ_p û_p û_{1+p}` differ
  (`uo up − um uo`): the negative control that pins the S47 sign convention
  `f ⋆ g[n] = Σ_m f[m] g[(n−m) mod n]` and not the `+m` variant.

## Transcript (2026-10-14, cadabra2-cli 2.5.15)

```text
CHECK 1 MSP sizes F^2 both sides (distribute collects the 9 products to 6 monomials; expect 6 6 6 6): 6 6 6 6
CHECK 2 MSP F^2 - sum_Q psi_Q E_Q, the product-to-convolution theorem (expect 0): 0
CHECK 3 MSP sizes F . dF both sides (distribute collects, the E_0 pair cancels; expect 4 4 4 4): 4 4 4 4
CHECK 4 MSP F . dF - sum_{Q,q} u_{Q-q} (i q) u_q E_Q, symbol on the leg (expect 0): 0
CHECK 5 MSP leg weight - total-momentum weight at Q = 1 (expect nonzero, printed -Ep i uo up): -Ep i uo up
CHECK 6 MSP 2 p.q - |p+q|^2 + |p|^2 + |q|^2 (0 = ok; (ip).(iq) = |p|^2/2 + |q|^2/2 - |p+q|^2/2): 0
CHECK 7 MSP sizes F^4: quartic side 21 -> 15 under the momentum table, pairing side 15 (expect 21 15 15 15): 21 15 15 15
CHECK 8 MSP F^4 - sum_Q psi_Q psi_{R-Q} per total momentum R (expect 0): 0
CHECK 9 MSP sizes zero-mode: [F^4]_{E_0}, pairing (distribute collects the 19 pair products to 3 monomials; expect 3 3 3): 3 3 3
CHECK 10 MSP [F^4]_{E_0} - sum_Q psi_Q psi_{-Q} (expect 0): 0
CHECK 11 MSP 4 psi_0(1) - psi_0(2) (expect 0, psi is quadratic in u-hat): 0
CHECK 12 MSP 16 P(1) - P(2) (expect 0, the pairing sum_Q psi_Q psi_{-Q} is degree 4): 0
CHECK 13 MSP 16 F^4(1) - F^4(2) (expect 0, the quartic is degree 4): 0
CHECK 14 MSP (u star u)_1 minus the +index convention (expect nonzero, uo up - um uo): uo up-um uo
```

## Counterfactuals (the checks discriminate)

Both mutations below were run on copies; the untouched module is unaffected.

* **Wrong stage-2 table entry** `Emm*Epp → Ep` (momentum −2 + 2 must be 0):
  CHECK 8 and CHECK 10 go nonzero (e.g. `2Ep um um up up − 2Eo um um up up`);
  CHECK 1–4 stay 0 because the entry is only reachable when two *collected*
  labels multiply (the quartic stage). The stage split is real coverage, not
  redundancy.
* **Sign error in the composite** `ψ_1 = û_0û_1 − û_1û_0`: CHECK 2, 8 and 10
  go nonzero (e.g. `2Ep uo up`). The convolution side is genuinely independent
  of the product side.

## Honest boundary

* What is certified here is **c-number bookkeeping**: momentum conservation,
  symbol weights, degree, and the composite-pairing identity — at the level of
  commutative mode amplitudes on a three-mode grid. No ladder operators,
  no normal ordering, no commutators, no analytic estimates.
* The **operator** form of the idea (a quartic same-point interaction as a
  quadratic form in pair composites `a†(p)a†(q)` at different momentum points,
  legitimate because the nested construction has *one* outer Fock system) is
  stated, not proved, in `../test/mathematical-foundations/momentum-space-products.md`.
  Its Lean neighbours are `ChapterNsAdvectionConvolution` (the advection
  convolution, sorry-free), `Book/FourierElimination.lean` (tag
  "fourier-elimination"), `ChapterFockCubicQuarticStability` (the
  mode-diagonal quartic `(a_k†)²(a_k)²` bounded below on all finite states) and
  `nsSpDGamma_number_conserving` (the outer operator is quadratic in the outer
  ladders). None of them proves the pairing identity in operator form.
* The continuum statements the toy grid mirrors (Schwartz-space
  product-to-convolution, the derivative symbol `2πi⟨q,m⟩`) are Lean theorems
  with their own normalizations; the grid checks the *bookkeeping*, not
  convergence.

## Engine traps discovered while writing the module (see CADABRA2_GOTCHAS.md §2b)

* A **trailing `#` comment on a `:=` line** silently drops the assignment; the
  failure surfaces lines later as `Python object 'fldx' does not exist`.
* **`x := len(ex)`** pushes the right-hand side through the Cadabra expression
  parser, fails to parse, and *silently swallows the rest of the file* (no
  output, no traceback). Size captures must use plain Python `x = len(ex)`.
* Cadabra's `distribute` **already collects like terms** (with coefficient
  arithmetic), so "raw" distributed sizes equal collected sizes; the pinned
  sizes in the transcript are the post-distribute counts.
