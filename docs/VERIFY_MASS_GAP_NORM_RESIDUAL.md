# Evidence — `||H\phi||` mass-gap estimator

Verified by Cadabra2 through the repo's subprocess seam (`prob_kernel::symbolic`).
Module: `mass_gap_norm_residual.cdb`. Tests: `prob_kernel/tests/mass_gap_norm_residual.rs`.

**A skip is not a pass.** All five tests `return` early with a `SKIP` message when
`cadabra2-cli` is absent. On a machine without the engine this file has verified
nothing. Get the engine with `nix-shell -p cadabra2`, or `cadabra2-cli` on `PATH`
(or `CADABRA_CLI`).

## The claim

Hermitian `H`, ground `E_0`, first excited `E_1`. Shift `H -> H - E_0 I`, so the
ground is at `0` and the gap **is** `E_1`. With `v` a normalized `E_1`-eigenvector
and `w = v + d`, `v ⟂ d`:

```text
  ||H w||^2  =  E_1^2 + ||H d||^2                                       (G)
```

## Witness

`H = [[1,1],[1,1]]`, acting as `H(x) = (x1+x2, x1+x2)`. Eigenvalues `0` (ground)
and `2` (first excited). `v = (1,1)` gives `H v = (2,2) = 2v` exactly, `|v|^2 = 2`.
`d = (a,b)` arbitrary. All rational — no radical cancellation inside the CAS.

## Transcript (`cadabra2-cli -q -l docs/mass_gap_norm_residual.cdb`)

```text
> canonicalise(G1);0     general identity, no orthogonality assumed
> canonicalise(G2);0     cross term dies under v _|_ d
> canonicalise(G3);0     ||Hv||^2 = E^2|v|^2 exactly (d = 0)
> canonicalise(G4);0     sqrt(E^2+q^2) - E = q^2/(sqrt(E^2+q^2) + E), mod. X^2 = E^2+q^2
> canonicalise(G5);q^{2} the bias remainder — non-negative, and NOT zero
```

`G5` is the informative one: it is *supposed* to be `q^2`, and `q^2 >= 0` is
exactly the one-sidedness guarantee.

## Engine limitation, recorded because it will bite again

This Cadabra build (2.5.14) does **not** open a polynomial power of a sum:

```text
expand(Ex("(2 + a + b)^{2}"))            ->  (2 + a + b)^{2}          # unchanged
expand(Ex("2*(2 + a + b)^{2}"))          ->  2(2 + a + b)^{2}        # unchanged
distribute(Ex("2*(2 + a + b)^{2} - ...")) ->  4(1)^{2} - 8 - ...    # WRONG
```

So the `.cdb` supplies the binomial via `substitute(..., repeat=True)` and lets CAS
reduce the remainder. The `.cdb` must be run with `-l` (line-by-line Cadabra);
`prob_kernel::symbolic` drives the CLI *without* `-l`, and applies
`VerifySubstitution` with `repeat` **not** set, so the Rust tests hand the already
expanded forms to a single pass instead. Net effect: the Rust tests verify the
Hermitian expansion relation term by term, and the `.cdb` verifies the same thing
with the binomial supplied by substitution. Both transcripts are zero.

`distribute` actively *corrupts* the binomial here — do not reach for it.

## Reachability ledger — how much exponential decay is actually needed

`error ≤ ‖H‖²‖d‖²/(2E)` with `E = 0.091`, so an exponential `‖d‖ ~ C·ρᵐ` always
eventually wins against the prefactor; the prefactor only sets **how many** orders.

**Correction (2026-10-06).** An earlier version of this file put `‖H‖ ≈ 3.4e2`.
That was read off the 8×8 `h_proj`, which is one small projection, not the operator.
`τ₄₈ = scales[48] = 6.010e+3` and `τ_m = ‖(H − z_k)u_k‖ ≤ ‖H − z_k‖`, so
**`‖H‖ ≥ 6.0e3`** — an order of magnitude larger. Every prefactor below reflects that.

| digits on the gap | target relative error | required `q = ‖Hφ − Eφ‖` | required `‖d‖` (via `‖H‖ ≥ 6.0e3`) |
|---|---|---|---|
| 3 | 1e-3 | 4.1e-3 | 6.8e-7 |
| 6 | 1e-6 | 4.1e-4 | 6.8e-8 |

**But the `q` column is the one to use.** `q` is a matvec — directly measurable.
`‖H‖` enters only on the `‖d‖ → q` inference route, so measuring `q` and using
`relative error ≈ ½(q/E₁)²` avoids the prefactor entirely. From `q = 2.98`
(measured, `m = 8`) that is a **~730× reduction**, versus the `≈ 4e8` reduction the
`‖d‖` route would demand. **Measure the residual; do not bound it from the
eigenvector error.**

## Why the near-degeneracy does not block this construction

The old framing — "the blocker is that `E₁ − E₀ = 0.091` sits in a 340-wide band" —
is wrong for `‖Hφ‖`, and right only for the difference estimator. Retracted:

| route | how error enters | conditioning |
|---|---|---|
| `θ₁ − θ₀` | **linearly**, `≈ 2‖r‖` | bad twice: linear amplification *and* subtracting two values `≈ −1.7`, `≈ −0.4` of size `‖H‖` |
| `‖Hφ‖`, `φ` = first excited | **quadratically**, `≈ ½(q/E₁)²` | good: no difference of large energies, and `φ`'s Krylov rate is set by its separation from the **rest** of the spectrum (`≈ 1.0`), *not* by the tiny `E₁ − E₀` |

The `0.091` splitting is the hard quantity for the *difference*; it is not the
binding constraint here.

## The remaining open question

Not the prefactor — whether the exponential regime is ever **entered**. Measured
`q` is `2.98` at `m = 8` and rises monotonically to `9.13` at `m = 128` (no
turnover, fixed window; rank has already collapsed by `m = 128` — see the sweep
below). Target is `≈ 4e-3`. Given exponential decay at
rate `ρ` from `q ≈ 3`, the extra orders needed are `log(730)/|log ρ|` — roughly
10–20 for `ρ ≈ 0.5…0.7`, i.e. feasible. But it has not been observed to arrive by
`m = 128`, nor by `m = 256` (the sweep below closes that measurement,
negatively). See `PLAN_MASS_GAP.md` for the two placement experiments that rule out
the easy explanations.

### The sweep past `m = 128` — A3, measured 2026-10-07

Does `q` turn over somewhere past `m = 128` and enter the exponential regime the
Hashimoto rate predicts for the φ-function error? **No.**
`fock_sirk/tests/zz_turnover.rs` (`#[ignore]`, run under
`cargo test --release -p fock_sirk --test zz_turnover -- --ignored --nocapture`,
7 291 s total for the sweep) ran harmonic shifts `γ_k = γ_0 + kh`, `h = 2/(m−1)`,
fixed window width 2:

| m | rank | θ₀ | q₀ | τ_m | q_pos | rel_err | flags |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 4 | 4 | −0.98074334 | 2.7289e0 | 6.4063e1 | 6.5438e0 | 3.8711e0 | |
| 8 | 8 | −1.34864876 | 2.9790e0 | 2.2386e2 | 5.7489e0 | 2.4396e0 | |
| 16 | 16 | −1.65823907 | 3.7455e0 | 7.8547e2 | 6.5641e0 | 2.5509e0 | |
| 32 | 24 | −1.66243272 | 4.8228e0 | 2.8161e3 | 8.4154e0 | 4.2081e0 | |
| 64 | 35 | −1.65809798 | 6.5067e0 | 1.0336e4 | 1.1370e1 | 7.6997e0 | |
| 128 | 50 | −1.65248000 | 9.0861e0 | 3.7267e4 | 1.6065e1 | 1.5117e1 | rank < m/2 |
| 192 | 61 | −1.63573825 | 1.1219e1 | 7.5845e4 | 1.9785e1 | 2.3522e1 | rank < m/2 |
| 256 | 71 | −1.65042383 | 1.2941e1 | 1.2390e5 | 2.2505e1 | 3.0739e1 | rank < m/2 |

```text
completed m: [4, 8, 16, 32, 64, 128, 192, 256]
first degenerate m (rank < m/2): Some(128)
```

Three findings, ordered by how much they change:

1. **No turnover.** `q₀` rises monotonically `2.73 → 12.9` across the entire
   sweep and `rel_err` climbs `3.87 → 30.7` against a target of `≈ 4e-3`. For
   this configuration (harmonic shifts, window width 2) the exponential regime
   is never entered; more `m` buys residual, not convergence. The question above
   is answered **negatively here** — which bounds the *configuration*, not the
   construction: window placement and width are untouched by this sweep
   (`PLAN_MASS_GAP.md`'s two placement experiments remain the other axes).
2. **Rank collapse begins at `m = 128` and does not recover** (`50 < 64`,
   `61 < 96`, `71 < 128`). The flagged rows are therefore **not** measurements
   of the window's convergence — that is exactly what the probe's `DEGENERATE`
   flag means — so they bound the trend but must not be quoted as the window's
   own residual. The "full rank" assumption behind the earlier note held only
   through `m = 64` (rank 35).
3. **θ₀ stays ≈ −1.65 throughout** (−1.6357 at the degenerate `m = 192`),
   consistent with a converged extremal value: the residual is the problem, not
   the eigenvalue estimate.

Consequence for the programme: this is a closed, negative measurement. The
escape has to come from a different construction (window placement/width, or the
gap-specific eigenvalue bound of `PLAN_MASS_GAP.md` §M4.6), not from extending
this sweep. Reporting any of these rows as progress toward the gap would be the
kind of claim `scripts/check-non-claims` exists to catch.

1. **No cancellation.** `||H w||` never forms `E_1 - E_0` as a difference of two
   energies of size `||H||`. Measured for the gauge-fixed QYM Hamiltonian: two
   lowest energies `≈ -1.66` and `≈ 0.39`, `||H||` up to `≈ 3.4e2`. A
   difference-based estimator is the badly conditioned one.
2. **One-sided bias, after the shift only.** `||H w||^2 - E_1^2 = ||H d||^2 >= 0`,
   so `||H w|| >= E_1`, giving a lower bound on the gap from the estimator's value.
   This needs `H` positive semidefinite, i.e. the shift to a ground at zero. Before
   shifting, `H` has negative spectrum and the argument does not apply.
3. **Quadratic error in the residual.** With `q = ||H d||`,
   `error = q^2/(sqrt(E^2+q^2) + E) <= q^2/(2E)` for `q <= E`. No linear amplification.
4. **No offset.** Exact at `d = 0` (CHECK 3).

## Conditionality — CHECK 5 needs the positivity shift (2026-10-06)

CHECK 5 shows the bias remainder is `q^2 >= 0`, giving `||H w|| >= E`. **That
inference requires `H` positive semidefinite**, i.e. the ground at zero. It is
**currently conditional in this repository**, because `qcd_ym_hamiltonian(g)`
applies no positivity shift:

```text
  g = 0.5   theta_0 = -1.601101   even_sector_is_vacuum(tol=1e-6) = false
  g = 1.0   theta_0 = -1.695953   even_sector_is_vacuum(tol=1e-6) = false
  g = 2.0   theta_0 = -3.009018   even_sector_is_vacuum(tol=1e-6) = false
```

`mass_gap_spec.rs:135` requires the one-particle `h` to be "shifted if needed for
positivity" before enclosure; the builder is a bare `compile_to_fock`. Unshifted,
`||H\phi||^2 = E^2 + ||Hd||^2` still holds as an **identity**, but the conclusion
bounds only `|E|`, never its sign — and the sign is what a gap claim needs.

Two consequences for this file:

- `||H|| >= 6.0e3` was measured **unshifted**; it changes under the shift
  (`+3.009` at `g = 1`).
- For the **physical** gap this is moot: `AGENTS.md` S45 makes the outer vacuum the
  exact ground at `E_0 = 0`, so the estimate applies with **no shift at all**. The
  shift is only missing on the one-particle *diagnostic*, which is a different
  quantity — see `PLAN_MASS_GAP.md` §7 (§M4.9).

See also: the rows emitted today are that one-particle diagnostic, not the
physical gap. `PLAN_MASS_GAP.md` §7 is the work order that fixes it.

## What it does NOT buy — the open part

`q = ||H d||` is the **residual**, not the eigenvector error `||d||`, and `H` is
unbounded on the Fock space. Small `||d||` does **not** bound `q`. On a truncated
space only `q <= ||H|| ||d||` holds, at the price of a `||H||^2` prefactor — here
`≈ 2e8` (with `||H|| >= 6.0e3`), against a gap of `0.091`.

Preferable, and what the ledger above recommends: **measure `q` directly** and use
`relative error ≈ ½(q/E₁)²`, which carries no `||H||` prefactor at all.

So this is a **well-conditioned estimator, not a certificate.** The Lean gap table
(`CONSOLIDATED_PLAN.md` §M4.6) still needs an independent bound on `q`. Reporting
`||H\phi||` as if it certified the gap would be exactly the kind of claim
`scripts/check-non-claims` exists to catch.
