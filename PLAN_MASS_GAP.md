# PLAN M — the mass-gap pipeline (planner side: everything except Lean4 code)

> **Scope**: everything the mass-gap proof needs from `../unfer` — numerics,
> certified-band emission, regeneration gates, fixtures, evidence, docs —
> **except Lean4 code**. All `.lean` work belongs to the LLM-Lean4-specialist
> and is work-ordered in `../timepiece/CONSOLIDATED_PLAN.md`
> ("State of the project — 2026-10-04: the mass-gap proof in Lean4 …", §M4).
> This plan is that entry's planner-side companion: one page per side, no
> overlap.
>
> **Status date**: 2026-10-04. **Improvement principle** (per
> `PLAN_HARNESS.md`): every item names the existing feature it improves.

---

## 0. Status — the proof has already started

> ### Execution status, 2026-10-06
>
> All nine items were executed. Summary, with the detail under each item:
>
> | item | verdict | the one thing to know |
> |---|---|---|
> | **M1** campaign | **DONE, but measuring the WRONG OBJECT.** 20 pinned fixtures, no coupling certifies a positive gap — **and the emitted quantity is an intra-sector diagnostic, not the physical mass gap.** See the sector correction below; this supersedes every other M1 conclusion. |
> | **M2** gate | **DONE as a command**, acceptance PARTIAL | Runs the canonical order and refuses to touch fixtures when a step is unproven. Steps 2–3 could not run on this host (no Aeneas bundle, no network), so **byte-identical reproduction is unverified**. |
> | **M3** spec parity | **DONE** | The copies *had* drifted. A byte-sync would have destroyed the bundle's self-containment, so the gate checks 22 operative claims instead. |
> | **M4** suites | **PARTIAL** | Upkeep done; the §4 ↔ test bijection is still only greppable by hand. |
> | **M5** f64 complement | **PARTIAL** | 12 proptest properties done. **The Kani harness does not exist.** |
> | **M6** evidence | **DONE** | Fixtures are self-describing, hashes manifested, log entry written. |
> | **M7** docs | **PARTIAL** | Boundaries enforced; the **public `test/` pages are not yet updated** with the negative result. |
> | **M8** H-of-record seam | **no change needed** | Untouched by M1–M9; its Cadabra tests are *unrun here* (no `cadabra2-cli`). |
> | **M9** non-claims | **DONE** | First version had 4 false positives and was rewritten narrow. Now 0 FP over 70 docs, and catches a planted overclaim. |
>
> **⚠ SECTOR CORRECTION (2026-10-06) — this supersedes every other M1 conclusion.**
> **The campaign is not computing the physical mass gap.** Two separate errors,
> both mine, found by checking the start vector against the vacuum doctrine.
>
> *1. The emitted quantity is an intra-sector diagnostic.* `run_campaign` starts
> from `OuterBosonCreate(InnerBosonicState::vacuum())` — the **one-outer-boson**
> sector — and `certified_mass_gap` returns
>
> ```rust
> let gap = co.value - ce.value;   // fock_sirk/src/certificate.rs:176
> ```
>
> a difference of two *one-particle* reflection-sector ground energies
> (`θ₀ ≈ −1.70`, giving `gap ≈ −0.571` at `g = 1`). That is not the mass gap.
> `AGENTS.md` (S45) and `mass_gap_spec.rs:135` both say the nested-Fock ground is
> the **outer vacuum at `E₀ = 0` identically**, in the *zero*-outer-boson sector,
> and that one-particle states are **diagnostics only**. So the rows this campaign
> pinned, and the values fed toward the Lean gap table, are diagnostics.
>
> *2. The documented positivity precondition is violated.* `mass_gap_spec.rs:135`
> requires the one-particle `h` to be "shifted if needed for positivity" before it
> is enclosed. `qcd_ym_hamiltonian(g)` applies no such shift — it is a bare
> `compile_to_fock(&qcd_ym_expression(g))`. Measured, the one-particle ground is
> **negative at every coupling**:
>
> ```text
>   g = 0.5   theta_0 = -1.601101     even_sector_is_vacuum(tol=1e-6) = false
>   g = 1.0   theta_0 = -1.695953     even_sector_is_vacuum(tol=1e-6) = false
>   g = 2.0   theta_0 = -3.009018     even_sector_is_vacuum(tol=1e-6) = false
> ```
>
> `even_sector_is_vacuum` is retained as "legacy… not a precondition", but it
> *encodes* the precondition the spec states, and the real builder fails it.
>
> **Consequences that invalidate earlier statements in this file:**
> - The **one-sidedness** result in `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md`
>   (`‖Hφ‖ ≥ E`) requires `H` positive semidefinite, i.e. the shift. Unshifted it
>   bounds only `|E|`, never the sign — and positivity is exactly what the gap
>   claim needs.
> - `‖H‖ ≥ 6.0e3` was measured **unshifted**; it changes once `+3.009` is applied
>   at `g = 1`.
> - My repeated talk of "a difference of large energies" was **aimed at the wrong
>   quantity**. For the physical gap `E₀ = 0`, so there is no such difference at
>   all. The `0.091` figure used throughout is an exact-diagonalization number at
>   occupation cutoff `N ≤ 8` of the *unshifted one-particle* `h` — a third
>   quantity again, and not the Lean-side target.
>
> **What survives unchanged:** the four measured arithmetic facts (backward term
> `6.0e-11 … 7.0e-08`; `θ₀` shift-invariance is expected; `τ_m` growth
> `6.4e+1 → 6.0e+3`; the two placement escapes both fail) — all of which are
> statements about the one-particle diagnostic and are unaffected by *which*
> quantity should have been emitted.
>
> **What the campaign should compute instead.** With the outer-vacuum ground at
> `E₀ = 0`, the physical gap is `E₁` directly, and the `‖Hφ‖` construction applies
> with **no shift at all**: `gap = ‖Hφ₁‖` for `φ₁` the first excited state, with
> relative error `≈ ½(q/E₁)²`. That is a strictly better-posed question than
> anything in this file, and it is filed as §M4.9.
>
> ---
>
> **The result below is superseded by the sector correction above; retained for the
> arithmetic it contains.**
>
> **The single result that changes the programme:** the certified width cannot be
> tightened by choosing shifts or raising `m` — window width and placement were both
> measured and both fail (fixed width leaves `resid₀` climbing `2.73 → 4.82`; moving
> the window off the spectrum lowers it only where rank collapses to 3–9 and `θ₀`
> freezes). But the method is **not** missing an exponential rate for that reason:
> with harmonic shifts the inverse-free chain is a cleared-denominator form of the
> resolvent chain, differing only by scalars that the unit-norm frame removes, so a
> free choice of start vector can realise the rate. **The blocker is a quantity
> mismatch.** `papers/Hashimoto.md`'s exponential rate is for the **φ-function
> approximation error**; the mass gap needs `|E₁ − E₀|`, and that gap is `0.091`
> inside a spectrum spanning `−1.7 … +337` (~3.7e3 ratio). No Krylov rate for the
> first quantity transfers to the second. The programme needs its own eigenvalue-error
> bound.
>
> Not backward error either: measured `6.0e-11 … 7.0e-08`, eight orders too small.
> Nor an invalidated extraction: `θ₀` satisfies the Parlett bounds, and
> shift-invariance is the expected signature of a converged extremal value on a
> shift-independent invariant subspace. Three explanations this note carried were wrong
> and are retracted below.
> §M4 is unaffected. §M4.3's premise is affected, and three new work orders
> (§M4.6–§M4.8) are filed in `../timepiece/CONSOLIDATED_PLAN.md`. No `.lean` file
> was written or compiled to produce any of this.
>
> **What still gates closure:** run `bash scripts/regen_mass_gap.sh --strict` on a
> host with Charon/Aeneas and `lean4export` 3.1.0; and separately, find an eigenvalue
> error bound for the `0.091` splitting that does not route through the inflating
> residual — the reference paper supplies none — which is `unfer` design work, not
> this plan's scope.

This is not a plan to *begin* a mass-gap proof; it is the operational plan for
the pipeline that feeds a proof already in progress:

- **Lean side, already landed** (record only — not this plan's work): T1–T7
  (finite-precision certificate layer + certified-gap theorem + stopping rule
  + nested-selection lemma) proved `sorry`-free/`axiom`-free in
  `ChapterSirkFinitePrecision.lean` / `ChapterSirkCertifiedGap.lean`; T8/T11/T12
  closed (`ChapterSirkCertificateReader.lean`, `ChapterSirkGapTable.lean`);
  T9 (Aeneas model) and T10 (lean4export→nanoda re-verification) executed
  green; `GapCertificate/GapCertificate.lean` pinned and nanoda-verified
  (`prob_kernel/tests/fixtures/gap_certificate.ndjson`).
- **Kernel side, complete and green**: `fock_sirk::mass_gap_spec` (pure
  proof-facing core), `fock_sirk::certificate` (NDJSON emitter),
  `fock_sirk::forward_sirk::certified_mass_gap_parity` (preconditioned seam),
  pinned by `fock_sirk/tests/qym_mass_gap.rs` and
  `fock_sirk/tests/qcd_mass_gap_certified.rs`.
- **Contracts**: `docs/MASS_GAP_SPEC.md` (spec of record, code→math map) and
  its vendored copy `../timepiece/unfer_contracts/MASS_GAP_SPEC.md` + the
  self-contained regeneration bundle (`MASS_GAP_REGENERATION.md`).

## 1. Why the proof is possible — and what this plan must protect

The proof is legitimate because of two structural facts. Every item below
exists to keep them true:

1. **Nested Fock space supplies the exact structure.** The object of record is
   the outer enclosure `H = Σᵢⱼ hᵢⱼ C†(eᵢ)A(eⱼ) = dΓ(h)` of the 3D
   gauge-fixed QYM one-particle Hamiltonian (`qcd_ym_hamiltonian(g)`,
   `H_final = ½π² + ½B²`). The exact `Z₂` reflection
   `R: (A₀,A₁) → (−A₁,−A₀)` splits the problem into two pure sectors with
   disjoint Krylov chains, so the gap is a difference of two sector-ground
   quantities — well-posed before any numerics run.
2. **The Hashimoto approximations carry rigorous bands.** Every approximation
   is a SIRK–Hashimoto solve whose Ritz values ship with certified widths
   `δ = ‖r‖ + c(n)·u·‖Ĝ‖ + h_O` (a-posteriori residual + eigendecomposition
   backward error + directed-rounding enclosure). The numbers therefore reach
   Lean as **enclosures, not trusted floats** — the finite-precision trust
   boundary is one ≤100-line interval core. **Hence the proof is possible**:
   a numerically located gap becomes a theorem through machine-checkable
   bands on both sides of the boundary (Rust emitter ↔ Lean reader ↔ nanoda).

Corollary discipline for every work item: never let a number reach a
certificate without its band, never regenerate a fixture from changed code
without the full gate (§4), and never let prose upgrade a truncated claim
into a continuum one (§5).

## 2. Division of labour and frozen contract

- **Planner (this plan)**: runs Rust; emits NDJSON certificates; runs
  `aeneas_sirk.sh`, `lean4export`, `verify_export` (nanoda); re-vendors into
  `../timepiece/unfer_contracts/`; records hashes and logs; maintains tests,
  CI, docs. Never writes Lean.
- **Specialist (Lean4)**: consumes `unfer_contracts/` + `GapCertificate/` +
  the Lean modules only; never compiles Rust, never reads this repo. Work
  ordered in `../timepiece/CONSOLIDATED_PLAN.md` §M4.
- **Frozen contract (additive-only)**: the `Certificate { value, residual,
  roundoff, enclosure, lo, hi }` fields; the NDJSON schema of
  `emit_gap_certificate_ndjson`; the T6 statement
  `λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)`; the S29/S31 export pipeline
  (official `leanprover/lean4export` 3.1.0 only — the legacy `ammkrn` tool's
  format 2.0.0 is rejected by nanoda); `mass_gap_spec`'s pure function
  signatures. Any new `uk_*` symbol follows the S29 checklist.

## 3. Work items

### M1 — Per-coupling certificate emission campaign (M)
**Improves**: `fock_sirk/tests/qcd_mass_gap_certified.rs` + the NDJSON emitter
— today they cover `g ∈ {1, 2}` across truncations; Lean item §M4.3 (gap-table
rows) is starved of data without more rows.
**Work**: emit certified sector certificates for a fixed grid — e.g.
`g ∈ {0, 0.5, 1, 2, 3}` × truncations `N ≤ {6, 8}` × Krylov orders `m` —
through `certified_mass_gap_parity` + `emit_gap_certificate_ndjson`, each run
recorded with source revision, Hamiltonian constructor
(`qcd_ym_hamiltonian(g)`), coupling, truncation, Krylov order, shifts, and
SHA-256 of the emitted NDJSON. Include the gapless control (`g = 0` shrinks
with depth) so the table's honest two-sidedness is data, not prose.
**Acceptance**: one pinned directory of per-`g` NDJSON fixtures + a manifest
of hashes; a test asserts each fixture's `certified_positive` flag matches its
own `lo > 0` (the existing honesty check, generalized). Owner: planner.

> **RESULT (2026-10-06) — DONE as an emission campaign. Its headline finding was
> initially mis-diagnosed; the correction is below and is the important part.**
> `fock_sirk/tests/mass_gap_campaign.rs`; 20 fixtures pinned under
> `fock_sirk/tests/fixtures/mass_gap_campaign/` with a SHA-256 `MANIFEST.tsv`. Grid:
> `g ∈ {0, 0.5, 1, 2, 3}` × shift count `m ∈ {10, 12}` × exact cross-check
> `N ≤ {6, 8}`.
>
> **No row certifies a positive gap.** All 20 carry `certified_positive: false`.
> At `g = 1, m = 12` the window is `[−8.009, +6.866]` against an exact truncated gap
> of `0.0911` — about **87× wider than the gap it must resolve**.
>
> **RETRACTION (twice; the second supersedes the first).**
>
> *First error:* the blocking term was said to be the eigendecomposition backward
> error `c(n)·u·‖Ĝ‖`, inferred from `c(n) = n³` growing without measuring. Measured,
> it is `6.0e-11` at `m = 10` and `7.0e-08` at `m = 64` — **eight orders of
> magnitude below** a width of ~3–7. The width is ~100% Parlett residual `‖r‖`.
> See `the_width_decomposition`.
>
> *Second error (mine, and the worse one):* I then concluded the extracted Ritz pair
> was "unvalidated" because `θ₀` is bit-identical across shift sets and sits above
> `E₃`. **Both inferences were wrong, and the second was an invalid comparison.**
>
> - **The `E₃` comparison compared two different Hilbert spaces.** The solve is
>   capped by `max_components = 200_000` with `prune_eps = 1e-14`, so it lives in a
>   far larger space than the `N ≤ 8` occupation cutoff used for the exact
>   diagonalization. Capping the solve at `max_components = 495` (= `C(12,4)`, the
>   `N ≤ 8` dimension) makes it blow up at `m ≥ 8` (571 components), which proves the
>   two are not the same object. `θ₀ > E₃` therefore carries no information.
> - **Shift-invariance is the expected signature, not a defect.** Shifts change the
>   basis, not the minimal-polynomial subspace `K(H, u₀)` that the Krylov sequence
>   spans. Every shift choice of the same order approximates the same extremal
>   eigenvalues of that subspace, so a converged `θ₀` *must* be shift-invariant. The
>   basis does respond — `scales`, `G`, and `h_proj` all move (`h_proj[0][0]` =
>   110.9 / 77.1 / 119.3 for `i..1.4i` / `10i..11.4i` / real `1..8`). I had checked
>   the output but not the intermediates.
>
> **What is actually measured** (`g = 1`, `unit_norm_steps`, one fixed start):
>
> | `m` | `θ₀` | `‖r₀‖` | `θ₁` | `‖r₁‖` | assembled gap |
> |---|---|---|---|---|---|
> | 4 | −0.98074 | 2.729 | +3.22945 | 6.544 | 4.210 |
> | 8 | −1.34865 | 2.979 | +1.54812 | 5.749 | 2.897 |
> | 16 | −1.65822 | 3.746 | +0.49767 | 6.566 | 2.156 |
> | 24 | −1.69595 | 4.353 | +0.38643 | 7.490 | 2.082 |
>
> `θ₀` **converges** (downward, toward ≈ −1.70), and the Ritz bounds are internally
> consistent: Parlett gives `λ_min ≤ θ₀` and `λ_min ≥ θ₀ − ‖r‖` (at `m = 8`,
> `λ_min ≥ −4.33`), and nothing observed violates either.
>
> **The anomaly that does survive: `‖r‖` grows with `m`** (`2.73 → 4.35`) while the
> *actual* error shrinks (≈ 0.35 at `m = 8`). A correct Lanczos/SIRK residual at a
> fixed spectral window shrinks, or at worst stays bounded — it does not inflate ~60%
> while the answer improves. So the width is loose by about an order of magnitude,
> and **looser at higher order**, which is exactly why more Krylov order fails to
> help. Reproduced by `the_residual_grows_while_the_value_converges`.
>
> **The mechanism, measured: the one-step step norms `τ_k` grow without bound, and
> the residual inherits them.** `scales[k] = τ_k = ‖(H − z_{k−1})u_{k−1}‖`:
>
> | `m` | `τ_m` | `‖ĉ_{m−1}‖` | `resid₀` | `θ₀` |
> |---|---|---|---|---|
> | 4 | 6.409e+1 | 1.487e-1 | 2.7289 | −0.980743 |
> | 8 | 2.239e+2 | 3.400e-1 | 2.9790 | −1.348649 |
> | 12 | 4.655e+2 | 1.338e+0 | 3.3502 | −1.531486 |
> | 16 | 7.855e+2 | 6.434e+0 | 3.7465 | −1.658222 |
> | 24 | 1.653e+3 | 1.509e+1 | 4.3529 | −1.695953 |
> | 32 | 2.816e+3 | 1.041e+1 | 4.8204 | −1.662896 |
> | 48 | 6.010e+3 | 1.078e+1 | 5.6992 | −1.663100 |
>
> `τ_m` grows ~94× over `m = 4…48` and does not plateau. `ritz_abs_residuals`
> builds `e[j] = (z_j − θ)ĉ[j] + τ_{j+1}ĉ[j−1]` for `j < m`, `e[m] = τ_mĉ[m−1]`, and
> contracts `e†Ge` — so `τ` and `‖G‖` set the bound's scale.
>
> **The rate is available; the quantity is the problem.** `papers/Hashimoto.md`
> (Hashimoto–Nodera, JJIAM 2019) concludes that for SIRK "the decay speed of the
> approximation error … is similar to an **exponential order**", while Arnoldi/SIA/RK
> reach only "the polynomial order or slower". The stated reason is that SIRK's
> vectors are built from **resolvents** `(γI − A)⁻¹`, *which are bounded*.
>
> **Retraction — the exponential rate is NOT blocked by the inverse-free
> construction.** I earlier wrote that it "cannot transfer" here. That was wrong,
> and the error was mine: the rate is a property of the *rational Krylov space*, not
> of how the vectors are materialised. With harmonic shifts `γ_k = γ_0 + kh` and
> `X = (γ_m I − H)⁻¹`, the one-step chain satisfies
> `w_k = (−1)^k X^{−k} Π_{j=1}^{m−k}(1 − hjX)v` — a *cleared-denominator* form of the
> resolvent chain, differing from it only by scalars, which the unit-norm frame
> removes. Since the start vector is a free choice, one can pick a `v` for which the
> inverse-free chain realises the exponential rate. **So "inverse-free" is not the
> blocker, and the M4.6 option (b) framed as "pay for resolvents" was based on a
> wrong premise.**
>
> **What actually blocks the mass gap is a quantity mismatch, not the solver.**
> Hashimoto's exponential rate is for the **φ-function approximation error**
> `‖φ_k(H)v − approx‖` (Eq. 12; Fig. 1's "relative errors in each Krylov step"). Our
> `certified_width` consumes the **Ritz residual** `‖Hψ − θψ‖`, and the mass gap
> needs `E₁ − E₀`. These are three different quantities. Exponential convergence of
> the first does **not** make the third small, and the regime makes that concrete:
> the gap is `0.091` while the spectrum spans `−1.7 … +337` — a ratio of ~3.7e3.
> Resolving a 0.091 splitting inside a 340-wide band is intrinsically far harder than
> the φ-function application the paper analyses, and no Krylov rate transfers to it.
>
> So the mass-gap programme cannot inherit this paper's conclusion. It needs its own
> error measure and its own bound, on the *eigenvalue* difference. Hashimoto bounds
> `‖φ_k(H)v − V_mψ(H_mK_m⁻¹)V_m*v‖`; nothing here bounds `|E₁ − E₀|` more sharply
> than the a-posteriori residual does today.
>
> **The measured anomalies remain real and separate.** `τ_m` grows ~94× over
> `m = 4…48`, `resid₀` grows `2.73 → 5.70` while `θ₀` converges (→ ≈ −1.66), and the
> two obvious escapes both fail (fixed window width: `resid₀` still `2.73 → 4.82` at
> full rank; window off the spectrum: residual falls only where rank collapses to 3–9
> and `θ₀` freezes). Those are *computational* observations about the residual
> pipeline. They are **not** evidence that the underlying method lacks an exponential
> rate — only that this residual does not exhibit one.
>
> **Two obvious escapes tried and refuted** (pinned by
> `the_shift_window_placement_does_not_rescue_the_residual`):
>
> 1. *Hold the window fixed.* `shifts_for_range((0,m))` widens it with `m` (to
>    `i·25.6` at `m = 128`), so widening was the first suspect. Pinning it to a
>    fixed width of 2 changes nothing: `resid₀` still climbs `2.73 → 4.82` over
>    `m = 4…32`.
> 2. *Move the window off the spectrum* (`R`, Hashimoto's `|γ| ≫ ‖A‖`). It does
>    lower the residual — but only where the basis **collapses**:
>    `R = 400 ≈ ‖A‖` gives rank `3 → 9` and `θ₀` frozen at ≈ −1.51; `R = 3000` gives
>    rank `3 → 4` and `θ₀` at ≈ −1.04. A falling residual on a rank-3 space is not
>    convergence. This degeneracy is intrinsic to a *polynomial* Krylov chain:
>    once `|γ| > ‖A‖` the factors `H − γI` are dominated by `−γI`, so every vector
>    is nearly collinear with the start. Resolvent vectors do not have this failure
>    mode.
>
> `R = 0` and `R = 50` are the honest operating regime (rank ≈ `m`, `θ₀` converging),
> and that is exactly where the residual grows.
>
> **So the finding stands as:** no coupling certifies a positive gap; the window is
> ~87× the exact gap at `g = 1`; the width is essentially all `‖r‖`; and the measured
> `‖r‖` grows with Krylov order. What is **not** established: that the method lacks an
> exponential rate. Three of my explanations were wrong and are retracted above
> (backward error; "invalidated extraction"; "the exponential rate cannot transfer to
> an inverse-free construction"). The surviving blocker is a **quantity mismatch** —
> the reference rate is for the φ-function error, not for `E₁ − E₀` — combined with
> the computational observation that this particular residual pipeline inflates with
> order.
>
> **A consequence for the existing suite.** `qcd_mass_gap_certified`'s
> "certified window contains the exact gap" assertion passes **vacuously** today: a
> window of `±8` centred on `−0.57` contains `0.091` for reasons unrelated to width
> tightness. That is still true, and it becomes meaningful only once the residual
> bound is tightened — after which it will start failing if the width then stops
> covering the exact gap. The same warning sits on that file's copy of it.
>
> **One encouraging observation.** The exact *gap* is N-stable (`0.091073` at `N ≤ 6`
> → `0.091161` at `N ≤ 8`) even though the absolute energies are not
> (`E₀ = −2.025 → −2.744`). The gap is the well-conditioned quantity.
>
> Two further corrections of the plan's own framing:
> 1. **The SIRK sector-assembled `gap` and the exact truncated gap `E₁−E₀` are
>    different quantities.** At `g = 1` they are `−0.571` and `+0.091`. The plan's
>    "≈ 0.09 at `g = 1`" (and §5) refers to the *exact* truncated gap.
> 2. **The shift count is not a Krylov order in the usual sense.** `shifts_for_range`
>    emits `i(1 + 0.2j)`, pushing *upward* in the complex plane, so raising the
>    count widens the spectral window instead of refining the low-energy
>    approximation. An early version of this file also looped over `N` without the
>    solve ever seeing it, emitting byte-identical rows; `N` is therefore the
>    exact-diagonalization cross-check axis and the shift count the solve axis, and
>    two tests assert each axis really varies.
>
> **Consequence for §M4.3:** the Lean gap table cannot be populated with
> certified-positive rows from this pipeline today, and the prior question —
> whether it can be populated *at all* from a validated extraction — is open.
> Filed as work orders §M4.6–§M4.8 in `../timepiece/CONSOLIDATED_PLAN.md`.

### M2 — One-command regeneration gate (M)
**Improves**: the manual 6-step gate of
`../timepiece/unfer_contracts/MASS_GAP_REGENERATION.md` (and MASS_GAP_CERTIFIED
§7.0) — currently a checklist a human runs in order.
**Work**: `scripts/regen_mass_gap.sh` that runs, in order: the corrected
QYM/QED/QG/NS numerical suites (`scripts/run_heavy_tests.sh` /
`run_physics_anchor.sh` release profile) → fresh NDJSON emission (M1) →
`unfer/sirk_core_model/scripts/aeneas_sirk.sh` (with its exit-status fix;
treat a run producing no `Lib.lean` as fatal; expect exactly the documented 7
f64-arithmetic errors — anything else fails) → `lean4export` export of the T6
instantiation → `cargo test -p prob_kernel --lib verify::` (nanoda
re-verification) → replace vendored fixtures **only** when both Aeneas and
nanoda are green → append the regeneration log entry (source rev, metadata,
hashes). Remember the documented nondeterminism: Charon `.llbc` bytes vary
run-to-run while `SirkCoreModel.lean` is byte-identical — compare the model,
not the `.llbc`.
**Acceptance**: one command reproduces the current pinned bundle
byte-identically on a clean tree; a deliberate source edit makes the gate fail
at the right step with the right message. Owner: planner.

> **RESULT (2026-10-06) — DONE as a command; the acceptance is only PARTIALLY met,
> and the shortfall is the toolchain, not the script.**
> `scripts/regen_mass_gap.sh` exists, with `--check` (verify wiring, run nothing)
> and `--strict` (any unavailable step ⇒ failure). It runs the canonical §4 order,
> gates step 5 on steps 2 **and** 4, and distinguishes three verdicts: `PASS`,
> `FAIL`, `UNAVAILABLE`.
>
> First run: **step 1 green** (certified suites), **step 4 green — nanoda
> re-verification passes over the pinned payloads**, step 2 **UNAVAILABLE** (the
> Aeneas release bundle is not installed on this host), step 3 **UNAVAILABLE**
> (`lean4export` 3.1.0 needs network). Verdict `INCOMPLETE`; the gate **refused to
> touch any vendored fixture**, which is the property that matters most.
>
> So the *byte-identical reproduction* half of the acceptance is **unverified** — it
> cannot be, on a host without the Lean-side toolchain — and the vendored bundle
> still reflects whatever produced it before. The log entry in
> `../timepiece/unfer_contracts/MASS_GAP_REGENERATION.md` (2026-10-05) records this
> verbatim. Run `bash scripts/regen_mass_gap.sh --strict` on a host with
> Charon/Aeneas and `lean4export` 3.1.0 to close it.
>
> One classification bug found and fixed while building this: the Aeneas step
> initially reported a missing binary as `FAIL`, which overstated what is known —
> nothing has been shown to be broken. It is now `UNAVAILABLE` with the reason.

### M3 — Spec-parity drift gate (S)
**Improves**: the two-copy spec (`docs/MASS_GAP_SPEC.md` ↔
`../timepiece/unfer_contracts/MASS_GAP_SPEC.md`) — byte duplication with no
drift check (the same defect class as `book.tex`/`ODE.tex`).
**Work**: source of record = `unfer/docs/MASS_GAP_SPEC.md`; a sync script
copies it into the bundle; both CIs fail on checksum mismatch with a "run the
sync" message. The bundle copy keeps its role as the specialist's frozen
input (copy is regenerated, never hand-edited).
**Acceptance**: editing one copy without syncing fails CI; sync restores
green. Owner: planner.

> **RESULT (2026-10-06) — DONE, but the plan's framing was wrong and the gate was
> built to the corrected rule.**
> The two copies **had already drifted** (`f3ebdb89…` vs `e3fe9f9d…`) — the defect
> was real, not hypothetical. But the differences were *deliberate*:
> `../timepiece/unfer_contracts/` is the specialist's **self-contained bundle**, so
> its copy legitimately rewrites `.cdb` paths to `unfer_contracts/docs/…`, carries
> its own "these `.cdb` files are reference text, never run them" preamble, and
> points at `PLAN_MASS_GAP.md` / `CONSOLIDATED_PLAN.md` by their in-bundle locations.
>
> **A byte-sync would have destroyed all three.** The plan's "same defect class as
> `book.tex`/`ODE.tex`" is wrong for this pair: `book.tex`/`ODE.tex` are pure
> mirrors, this is a *derived* artifact. So `scripts/check-mass-gap-spec-parity`
> checks **operative parity** instead — 22 anchors covering the Hamiltonian of
> record, the `B` field, each width term separately, the T6 statement, the `Z₂`
> reflection, and the frozen API names. Each anchor must be present in **both**
> copies; the declared per-copy differences are listed by the script itself.
>
> Verified: green on the current tree, and it **caught a planted one-term width
> change** (`h_O = 4u·max(|θ|,1)` → `4u`), naming the missing claim. Wired into
> both repos' CI (`unfer/.github/workflows/ci.yml` hygiene-gates job, mirrored to
> `timepiece/scripts/`).

### M4 — Band-correctness suite upkeep (M)
**Improves**: the existing regression suites
(`hashimoto_error_bands.rs`, `bands_program_gauge_fixed.rs`,
`guard_justification_study.rs`, `qym_mass_gap.rs`, `qcd_mass_gap_certified.rs`,
`outer_vacuum_ground_validation.rs`, `assumption_ledger.rs`).
**Work**: keep every claim row of `MASS_GAP_SPEC.md` §4 pinned by a named
test (the table is the contract); when a claim is corrected — as the
reflection-sector refinement corrected the lattice-era `g²/2` reading — the
test, the spec table, and the bundle move together in one commit. Ensure the
heavy suites actually run on a schedule (heavy-test log review is an open item
in `PROJECT_PLAN.md`); add the release-profile note to each suite header.
**Acceptance**: `MASS_GAP_SPEC.md` §4 table ↔ test names bijection checkable
by grep; heavy-suite logs current. Owner: planner.

> **RESULT (2026-10-06) — PARTIAL: upkeep done, no new suites added, bijection
> check NOT built.** Every claim row of `MASS_GAP_SPEC.md` §4 is already pinned by a
> named test (`qym_mass_gap.rs`, `qcd_mass_gap_certified.rs`,
> `hashimoto_error_bands.rs`, `bands_program_gauge_fixed.rs`,
> `guard_justification_study.rs`, `outer_vacuum_ground_validation.rs`,
> `assumption_ledger.rs`), and all are green. The **§4 ↔ test bijection check is not
> machine-checked** — it is greppable by hand, so a claim row added to the spec
> without a test would go unnoticed. The heavy suites now run in CI through the M1
> campaign (release profile), which closes the "heavy-suite logs current" half of
> the acceptance; the per-suite release-profile header note was not added
> everywhere.
>
> **On the reflection-sector correction:** the spec's `≈ 0.09 at g = 1` was already
> corrected away from the lattice's `g²/2`. M1 pinned down which quantity that
> number belongs to — the **exact truncated gap** (`E₁−E₀ = 0.0911`), not the SIRK
> sector-assembled `gap` (`−0.571`) — and that distinction is now stated in the M1
> file header, because two different quantities were being discussed under one
> label.

### M5 — f64 soundness complement (M)
**Improves**: the trust story of §1 — the bands enclose rounding, but nothing
yet checks the *code paths* for overflow/panic/UB.
**Work**: Kani harnesses on the f64 paths of `mass_gap_spec`/`certificate`
(no-proof complement, as `MASS_GAP_CERTIFIED.md` §5.3 prescribes); proptest
properties on the pure functions (`certified_width` monotone in each term;
`gap_interval` ordering; `interval_contains` reflexivity;
`certified_gap_lower_bound ≤ gap_interval.lo`), extending the existing A5
property-test wave.
**Acceptance**: `scripts/verify-invariants` (H1) or `cargo test` runs the new
properties; Kani harness documented with expected runtime. Owner: planner.

> **RESULT (2026-10-06) — the proptest half is DONE; the Kani half is NOT DONE.**
> `fock_sirk/tests/mass_gap_properties.rs`, 12 properties, all passing:
> `certified_width` monotone in the residual and in `‖Ĝ‖`; width never negative or
> non-finite; `interval_contains` reflexive; `gap_interval` ordered; the interval
> endpoints **are** the T6 assembly (`lo = θᵒ − θᵉ − (δᵒ + δᵉ)`); the certified lower
> bound lies inside the interval; `certified_positive` follows `lo`;
> `backward_error_const` monotone in the rank; `UNIT_ROUNDOFF` pinned to exactly
> `2⁻⁵³`; `parlett_bound` monotone in the residual; the Rayleigh quotient finite on
> its domain.
>
> **The Kani harness does not exist.** `kani` is not a workspace dependency, there
> is no `#[kani::proof]` anywhere, and this environment cannot install the
> toolchain. Recorded as outstanding rather than claimed. `fock_sirk`'s
> `dev-dependencies` gained `proptest`; nothing else changed.
>
> Two generator facts the properties had to respect: `rayleigh_quotient`
> legitimately **panics** on a zero vector (the quotient is undefined), so the
> generator must exclude zero rather than assert around it; and these properties
> are evidence about function behaviour on the generated domain — **not** a proof,
> and not a substitute for the T5 enclosure the certificate path actually consumes.

### M6 — Evidence pinning and the regeneration log (S)
**Improves**: the regeneration log section of `MASS_GAP_REGENERATION.md`.
**Work**: keep every regeneration logged with source revision, tool revisions
(Charon/Aeneas/lean4export/nanoda), the 7-error expectation, byte-identity
results, and hashes; the log states explicitly when a fixture is *not*
evidence for changed code. Bundle version markers bump only on structural
schema change (the H3 `SESSION_FORMAT_VERSION` discipline).
**Acceptance**: a reader can verify any pinned fixture against its log entry
without asking anyone. Owner: planner.

> **RESULT (2026-10-06) — DONE.** Every campaign fixture carries a
> **self-describing metadata header** (coupling, Krylov order, cross-check
> truncation, the exact gap at that truncation, the Hamiltonian constructor, the
> solve source), so a fixture can be placed without first finding its log entry.
> `MANIFEST.tsv` carries a SHA-256 per fixture plus its coupling, Krylov order,
> cross-check `N` and its `certified_positive` verdict, and
> `the_manifest_matches_the_pinned_fixtures` fails on any mismatch. The
> regeneration log entry is in
> `../timepiece/unfer_contracts/MASS_GAP_REGENERATION.md` under the 2026-10-05
> heading: source rev, toolchain, the grid, the negative certification result, and
> the two plan-framing corrections this campaign forced.
>
> **Bundle version markers did not bump**, and should not: no structural schema
> change occurred, only new rows for couplings that were already representable.

### M7 — Docs and honest-boundary upkeep (S)
**Improves**: the status tables of `MASS_GAP_CERTIFIED.md` §6, `STATUS.md`,
and the public pages (`../test/numerics/`, `../test/proofs/`).
**Work**: after each M2 run, update the measured tables and restate the
boundaries (§5 below) verbatim in any touched prose; keep
`../test/numerics/*-mass-gap.md` in sync with the spec of record (the
2026-09-25 wave already leads those pages with the R² vielbein record — same
discipline for the QYM mass-gap page). Lean-shaped updates go to
`../timepiece/CONSOLIDATED_PLAN.md` §M4 as work-order rows only.
**Acceptance**: no status table cites numbers absent from a pinned fixture.
Owner: planner (docs).

> **RESULT (2026-10-06) — boundaries DONE; the public status tables NOT updated.**
> The §5 boundaries are restated verbatim in this plan, in the M1 file header, and
> in the M3 gate's own failure message, and two of them are now enforced
> mechanically by `check-non-claims` (M9). The M1 finding is recorded in the
> regeneration log and in the `CONSOLIDATED_PLAN.md` work orders.
>
> **`../test/numerics/qym-mass-gap.md` and `../test/proofs/proof-plans.md` have NOT
> been updated** with the 20-row campaign or the negative certification result.
> That is the remaining M7 work and it is the one that matters for the public
> face: a reader of the site currently sees `g = 1` numbers without being told that
> **no coupling certifies a positive gap**.

### M8 — Hamiltonian-of-record seam upkeep (S)
**Improves**: the S30 Cadabra2 coupling — the Hamiltonian of record must keep
matching its derivation.
**Work**: keep `symbolic::tests::yang_mills_*` green wherever `cadabra2-cli`
is present (skip otherwise, per the S30 rule); the `.cdb` notebooks in the
bundle stay **reference text for the specialist (never run there)**; if the
derivation changes, the inline operative algebra in `MASS_GAP_SPEC.md` moves
with it and M2 re-runs.
**Acceptance**: derivation-matching tests green on a Cadabra2-equipped host;
bundle prose and `.cdb` agree. Owner: planner.

> **RESULT (2026-10-06) — no change needed; verified rather than extended.** The
> `qcd_ym_hamiltonian(g)` constructor and its `docs/yang_mills_hamiltonian.cdb`
> derivation were not touched by any of M1–M9, so the seam is unchanged and the
> existing `symbolic::tests::yang_mills_*` tests remain its guard. Those tests skip
> without `cadabra2-cli`, which is **not present on this host**, so they are
> *unrun here*. What did exercise the Hamiltonian is the M1 campaign — through the
> Rust constructor, not the `.cdb`. The bundle's `.cdb` files remain reference text
> for the specialist and are never executed.

### M9 — Non-claim enforcement (S)
**Improves**: §5 discipline. **Work**: the truncated-vs-continuum and
gauge-fixed-vs-lattice distinctions are checked as text invariants in the doc
gates (a `rg` guard that fails if "continuum mass gap" appears near
"proved"/"certified" in status prose without the boundary qualifier); new
agent-facing docs (e.g. `../test/`) must reuse the boundary sentences.
**Acceptance**: the guard exists and fails on a planted overclaim. Owner:
planner.

> **RESULT (2026-10-06) — DONE, after the first version was rejected.**
> `scripts/check-non-claims`, a single `awk` pass over 70 documents, two boundaries
> (B1 truncated-vs-continuum, B2 gauge-fixed-vs-lattice). Wired into CI's
> hygiene-gates job.
>
> **The first version was not good enough and was rewritten rather than shipped.**
> It flagged four lines — and *every one was a false positive*: prose that correctly
> **explains** the distinction ("electric term `(g²/2)Σn` gaps the one-quantum sector,
> while the gauge-fixed …"), the negation split across a line wrap ("… until it
> closes, every certified number is …"), and the lattice comparison. A prose gate
> that cries wolf gets switched off and then protects nothing, so the shipped
> version is deliberately **narrow**: it fires only on an assertion that is hard to
> read any other way, and treats a sentence as safe if it carries any hedge.
>
> Verified: **0 false positives across 70 real documents**, and it catches a planted
> overclaim (both B1 and B2 fire on a two-line plant).
>
> Two bugs worth remembering, both of which made the guard silently inert rather
> than wrong: `awk` does not support `\s`, and `.` does **not** match the multibyte
> `²` in `g²/2` — so B2 had to be spelled `g[^ ]{0,3}/ *2`, or it would have passed
> forever without ever matching anything.
>
> The guard is line-based and cannot read meaning. It catches the *coarse*
> overclaim; §5 below remains the text of record.

## 4. The regeneration gate (canonical order)

1. Run corrected QYM/QED/QG/NS suites (release profile); emit fresh NDJSON.
2. Regenerate the Aeneas model; assert completeness (expected f64 errors only).
3. Re-export the Lean instantiation (`lean4export` 3.1.0, matching toolchain).
4. Re-verify with nanoda via `verify_export` (the `verify::` tests).
5. Record source rev + metadata + SHA-256; replace vendored fixtures; update
   status tables and the log — only after 2 and 4 are both green.

## 5. Honest boundaries (non-claims, restated everywhere)

- The certified claim is about the **truncated** gauge-fixed object `H_m`.
  The continuum Millennium mass gap needs the gap-preserving norm-resolvent
  convergence of the truncation family (or an independent a-priori bound) —
  the single missing leg, owned by Lean work-order §M4.5. Nothing in this
  plan upgrades that claim.
- The gauge-fixed gap is its own (`≈ 0.09` at `g = 1`, growing with `g`); the
  lattice-era `g²/2` and the pinned `GapCertificate` numbers (`g = 2, m = 4`)
  are a **historical fixture**, retained as input data.
- The f64 arithmetic is never trusted: certificates consume enclosures and
  residuals only; the interval core (T5) is the one named trusted component.
- Squeezed inner states are one-particle spectral diagnostics, not full-theory
  ground states (outer annihilation kills the outer vacuum —
  `MASS_GAP_REGENERATION.md`).

## 6. Explicitly out of this plan

- **Any `.lean` editing** — including regenerating `GapCertificate.lean`,
  Aeneas-model integration, and the gap-table instantiation. Those are the
  specialist's §M4 items; this plan only produces their *inputs*.
- The continuum convergence proof itself (Lean item §M4.5).
- Reviving the lattice model as an object of record.
- New `uk_*` surface beyond what M1–M9 need (and anything added follows S29).

## 7. §M4.9 — emit the physical gap, not a one-particle diagnostic

**New work order, opened by the sector correction above (2026-10-06).** This is
the item that actually matters, and M1–M9 as executed did not produce it.

### What is wrong today

`run_campaign` starts in the one-outer-boson sector and
`certified_mass_gap` (`fock_sirk/src/certificate.rs:176`) returns
`co.value - ce.value` — a difference of two **one-particle** reflection-sector
ground energies. Per `AGENTS.md` S45 and `mass_gap_spec.rs:135`, the nested-Fock
ground is the **outer vacuum at `E₀ = 0`**, in the zero-outer-boson sector, and
one-particle states are diagnostics. So every pinned fixture, and every value
heading toward the Lean gap table, is a diagnostic.

### What §M4.9 emits instead

With `E₀ = 0` the gap is `E₁` outright — no difference of large energies, so the
"cancellation" objection that was raised against the `||H\phi||` route applies to
the *diagnostic*, never to the physical gap. The estimator is then:

```text
  gap = ||H phi_1||,     phi_1 = first excited state of the full H,
                           with ||phi_1|| = 1 and the outer-vacuum ground at 0.
```

and by `docs/mass_gap_norm_residual.cdb` (CHECK 1–4, all `canonicalise` → `0`):

```text
  ||H phi_1||^2 = E_1^2 + ||H d||^2,     relative error ~ 1/2 (q/E_1)^2,
  q = ||H phi_1 - E_1 phi_1||  (a matvec: directly measurable, no ||H|| prefactor).
```

Requirements, in order:

1. **Drop the even/odd reflection split as the gap.** Keep it as the labelled
   diagnostic it is; do not feed it to the Lean table.
2. **No positivity shift is needed** for the physical gap — `E₀ = 0` already. The
   shift matters only for the *one-particle* diagnostic, where it is currently
   missing (§"documented precondition is violated" above). Decide separately
   whether to shift the diagnostic so `even_sector_is_vacuum` passes.
3. **Resolve which `φ₁`.** "First excited of the full `H`" means the first state
   above the outer vacuum in the multi-outer-boson Fock space, not the lowest
   `h` eigenvalue. That is a different start vector, and it is a free parameter —
   the one thing this programme has always had freedom over.
4. **Report the sector label in every NDJSON row.** The current emitter labels
   rows by parity, which is what let a diagnostic read as a mass gap. A row that
   cannot name its sector should not be emitted.

### Acceptance

- Every emitted row names its sector and whether it is the physical gap or a
  diagnostic.
- The Lean gap table (§M4.3) is fed **only** physical-gap rows; if none are
  certified, it stays empty (option (c) in `CONSOLIDATED_PLAN.md` §M4.6).
- `q` is reported alongside `gap`, since the claim is meaningless without it.

### Still open, unchanged by this work order

`q` must fall to `≈ 4.1e-3` for three digits (`VERIFY_MASS_GAP_NORM_RESIDUAL.md`,
reachability ledger). Whether the residual ever enters an exponential regime is
unmeasured past `m = 128` — see the turnover question in that file.
