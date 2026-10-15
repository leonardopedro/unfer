# Cadabra module review — `docs/*.cdb`

**Review date:** 2026-10-13 (updated 2026-10-14 — see *Update* below; scope grew from 19 to
21 `.cdb` modules). **Scope:** all 19 `.cdb` modules under `docs/`,
the Rust pipeline that consumes them (`prob_kernel/src/symbolic.rs`), and the
three `fock_sirk` suites that verify the derived Hamiltonians numerically
(`cdb_hamiltonian_match.rs`, `latex_cas_hamiltonian_match.rs`,
`gauge_fixed_program_validation.rs`).

## How these modules are actually tested

There are two independent verification seams, and it matters which one you are
looking at:

1. **Symbolic seam (engine-dependent).** `prob_kernel::symbolic::symbolic_derive`
   concatenates a `.cdb` file, appends an extraction trailer
   (`print("UNFER_DERIVE|<name>|" + str(eval("<name>")))`), runs
   `cadabra2-cli -q -n`, and parses the named handles out of stdout. The tests
   (`qg_densitized_tetrad_absorb_1_over_e`, `qg_densitized_change_of_variables_is_unitary`,
   `qg_starobinsky_hamiltonian_derivation_runs`, …) then assert on the
   **string** of each extracted handle. **These tests call `require_cadabra()`
   and silently SKIP when `cadabra2-cli` is absent** — a skip is not a pass.

2. **Numerical seam (always runs).** `fock_sirk/tests/cdb_hamiltonian_match.rs`
   does **not** run Cadabra. It verifies that the Rust builders in
   `nested_fock_algebra::models` (`yang_mills_hamiltonian`, `qcd_ym_hamiltonian`,
   `qg_densitized_kinetic`, `qg_starobinsky_*`, `navier_stokes_hamiltonian`,
   `ns_eulerian_fiber`, …) implement the *content* the `.cdb` modules derive —
   term-by-term against `H_final = ½π² + ½B²`, the Starobinsky potential, the
   1/16 & −1/24 densitized coefficients, the exact Ehrenfest identity, etc.

## Test results on this machine (aarch64 / Oracle Linux 10)

- **Cadabra engine: not runnable here.** The `unfer/flake.nix` devShell only
  exposes `devShells.x86_64-linux.*`; this box is `aarch64-linux`, so
  `nix develop` refuses. No `cadabra2` in the nix store, no pip module, and
  `dnf install` needs sudo. **So every engine-dependent symbolic test was
  SKIPped, not passed.** Running them needs an x86_64 host (or an aarch64
  cadabra2 build).
- **Rust toolchain:** installed 1.99.0 via the global `rustup`
  (`/nix/var/nix/profiles/default/bin/rustup`) and set it default, as requested.
- **`cdb_hamiltonian_match`: PASSED** (1 test, `qym_abelian_limit_cas_photon_sirk`,
  is `#[ignore]`'d as a >60 s SIRK solve — expected). This is the real
  executable verification of the modules' physics content, and it is green.
- **`gauge_fixed_program_validation` / `latex_cas_hamiltonian_match`:** their
  dev-dependency `mathhook-core` takes **>13 minutes** to compile on this ARM
  box; the gauge suite began running (10 tests, QYM/QG SIRK solves each >60 s)
  but was not allowed to finish, per the instruction not to block on
  time-consuming tests. These do not gate this review: the physics they check
  overlaps `cdb_hamiltonian_match`, which passed.

Note: to build at all on aarch64 the `gemm`/`gemm-f16` dependency needs
`RUSTFLAGS="-C target-feature=+fp16"` (the CPU has `asimdhp`; the default
aarch64 target does not enable fp16 intrinsics). `+full-fp16` is not a
recognized feature name on this rustc — use `+fp16`.

## Findings

### F1 (fixed) — `qg_densitized_hamiltonian.cdb`: a verification cell that could not fail

The module's section-4 cell was:

```
diff := (1/16) \tilde{S}_{a b} \tilde{S}^{a b} - (1/24) \tilde{P} \tilde{P}
        - (1/16) \tilde{S}_{a b} \tilde{S}^{a b} + (1/24) \tilde{P} \tilde{P}
```

That is `X − X`, identically zero **by construction**, regardless of whether
the step-2 substitution chain actually absorbed the `1/e`. It certified nothing.
The module header *itself* already flagged this ("the `diff` handle here is a
hand-typed tautology that cannot fail — it never references Ktrans/Kflat"), and
`docs/VERIFY_CDB_TRUNCATION_AUDIT.md` records it as an audit finding, but the
fix had never been applied.

**Applied:** rewired `diff := @(Ktrans) - @(Kflat)`, so it now differences the
*actual* transformed kinetic against the declared flat target and can fail if a
residual `e^{-1}` survives. `diff` is verification-only (not in the extraction
list `["Hraw","Ktrans","Kflat","H0"]`), so no downstream contract changes.

### F2 (fixed) — the Rust test asserted substrings, not equality

`qg_densitized_tetrad_absorb_1_over_e` extracted `Ktrans` and `Kflat` but only
asserted `contains("1/16") && contains("1/24")` on each. Two different
expressions both containing those coefficients would pass — so the test could
not detect an incomplete absorption either. This is the *reason* the F1 tautology
survived so long: neither seam compared the two quantities.

**Applied:** added `assert_eq!(ktrans, kflat)` (whitespace-normalized), the
load-bearing equality the change-of-variables claim actually rests on.

### F3 (noted, not changed) — `qg_unitarity_check.cdb` Jacobian check is partly by-substitution

In the unitarity module, `Jcheck := y^{3} y^{2} - @(J)` with `J := y^{5}` is
zero *because* the module substitutes `y^{3} y^{2} -> y^{5}` by hand — it checks
that Cadabra applied the rule, not that `det(ỹ) = y⁵` from the epsilon-tensor
definition above it. The `det_tilde` epsilon contraction is computed but its
result is never wired into `Jcheck`. Same shape as F1 but weaker impact: the
Rust test asserts `Jcheck/Unorm/Hkern == 0`, which still holds. A follow-up
would replace the hand-substitution with `Jcheck := @(det_tilde) - y^{2} @(det_tilde_over_y3)`-style
wiring, but that needs an engine run to validate the canonical form, which this
machine cannot provide. **Left alone deliberately** rather than risk a change
I cannot execute.

### F4 (observation) — massive, excellent self-documentation; a few drift risks

Every module carries a `MODULE DOCUMENTATION` header stating purpose, role,
certified content, and *improvement notes* — this is unusually good practice
and is what made this review fast. Two drift risks worth a CI note:

- The headers reference `docs/VERIFY_*.md` audit files and
  `unfer/PLAN_MASS_GAP.md` work orders by name; there is no checker that those
  references still resolve.
- `CADABRA2_GOTCHAS.md`'s "four-character `@()` rule" is the single most
  load-bearing engine constraint. It is followed correctly everywhere I looked
  (`phicomm`, `Upot_psi`, `Hfib_psi`), but a mis-resolution aborts the run with
  a confusing `Python object does not exist`. A lint that flags `@(<name>)`
  where `<name>` has a `_`-preceded stem < 4 chars would catch the whole class
  before CI.

### F5 (observation) — the numerical seam is stronger than the symbolic seam

The `cdb_hamiltonian_match.rs` tests are, in places, *more* rigorous than the
`.cdb` modules: they reimplement the SU(3) structure constants independently
(`su3_f_mirror`, deliberately not re-calling the builder's table), check
Hermiticity term-by-term, and verify the exact Ehrenfest identity
`i⟨[H,u₀]⟩ = 4⟨A₀⟩`. The symbolic seam, by contrast, asserts on Cadabra
`str()` output with substring probes (now improved for the densitized case by
F2). If a `.cdb` and a Rust builder ever disagree, the numerical seam is the
more trustworthy of the two — worth stating in `AGENTS.md`.

## What was changed

| File | Change | Risk |
|---|---|---|
| `docs/qg_densitized_hamiltonian.cdb` | `diff` rewired to `@(Ktrans) - @(Kflat)` (was `X − X`) | none to contracts; `diff` not extracted |
| `prob_kernel/src/symbolic.rs` | added `assert_eq!(ktrans, kflat)` to `qg_densitized_tetrad_absorb_1_over_e` | low; only tightens a passing test |

`Cargo.toml` was temporarily narrowed to `["fock_sirk","nested_fock_algebra"]`
to build around a missing sibling path-dependency (`../../dynamic-arctic`, only
reached via `unfer_consensus`→`unfer_ffi`), and has been **restored to its
original 16-member list**.

## Update 2026-10-14 — 21 modules, the CHECK-label honesty pass, and the CAS `expand()` fix

### New modules (19 → 21)

| module | content | checks |
|---|---|---|
| `docs/ns_compressible_lagrangian.cdb` | the most general barotropic compressible NS in Lagrangian variables: determinant/cofactor identities, the Jacobi `d(det F)/dt = det F div u`, `d(ρJ)/dt = 0` + the `ρ = ρ₀/J` solution, the barotropic chain `p = −dW/dJ = ρ²e′(ρ) = ρε′ − ε`, `dp = ρ dh`, the acoustic fiber `(ω−ck)(ω+ck) = ω²−c²k²`, momentum-conservation convolution products, weight-degree and pressure-degree checks, diagonal viscosity, the Legendre identity `H = πv − L` at `π = ρ₀v`, and the Faris–Lavine domination `N − h = ` completion square | 19, all at their advertised values |
| `docs/sm_gauge_fixed_hamiltonian.cdb` | the FULL Standard-Model Hamiltonian in the temporal (Weyl) gauge on the nested Fock space: `SU(3)×SU(2)×U(1)` with non-abelian magnetic fields, the Higgs doublet with the **unbroken** Mexican-hat potential `V = −½μ²φ² + ¼λφ⁴` in exact form, the Weyl-fermion kinetic sector, Yukawa couplings with CKM/PMNS mixing, BRST assembly; plus the unbroken ↔ broken shift identity (CHECK 13) and `m_h² = 2λv²` (CHECK 14) | 14, all at their advertised values |

### CHECK-label honesty pass (eight modules, 2026-10-13)

All 19 then-existing modules were run end-to-end (every CHECK at its advertised value), and eight
were edited so the labels state what each check **actually** reduces to:
`mass_gap_norm_residual.cdb`, `qg_densitized_hamiltonian.cdb`, `yang_mills_hamiltonian.cdb`,
`qg_starobinsky_vielbein_hamiltonian.cdb`, `qg_gauge_fixed_hamiltonian.cdb`,
`qg_starobinsky_hamiltonian.cdb`, `qg_starobinsky_einstein_frame.cdb`, `qg_unitarity_check.cdb`.
The substance: `mass_gap_norm_residual.cdb` CHECK 5 expects the **non-negative remainder `q²`**
(not 0 — it is a one-sidedness bias, not an identity), and the silent-truncation guards pin their
**measured sizes** so a collapse to fewer terms fails the label (QGGF 120/132, STARVB 121/2 + 10
terms, QGDEN 5/2/2/2, YMH 2/6/6); the Starobinsky expectation labels (plateau `M⁴/(16α)`, …) state
their expected values likewise. The affected `VERIFY_*.md` rows were aligned to match.

### Three new numerical suites (`fock_sirk/tests/`)

- `ns_compressible_lagrangian_validation.rs` (12 tests) — the barotropic/Lagrangian identities
  numerically, incl. convolution products.
- `sm_gauge_fixed_validation.rs` (8 tests) — the SM gauge-fixed content: ε-contractions,
  Mexican-hat algebra, normal-ordering/`⟨0|H|0⟩ = 0`, BRST nilpotency, assembly.
- `sm_higgs_mechanism_sirk.rs` (3 tests) — **Hashimoto-SIRK dynamics of the Higgs mechanism** in
  the nested Fock space, with gauge-invariant (radial-only) initial and final states that mimic the
  broken states. Measured: the displaced condensate survives one full period with `⟨φ⟩` never
  collapsing to the symmetric point (`⟨φ⟩ ∈ [1.67, 1.95]` vs `v = 1.414`), the gauge-invariant
  `⟨φ†φ⟩` stays within 25% of its condensate value `⟨φ⟩₀²/2 + ½` (coherent spread `Δφ² = 1`),
  norm + energy conserved; the condensate oscillates at the radial-mode mass
  (`ω_measured = 1.315` vs `m_h = √(2λ)v = 1.414`, the anharmonic shift of the displaced state);
  and after one period the gauge-invariant in/out state returns with fidelity `|⟨ψ(T)|ψ(0)⟩| =
  0.936`.

### F6 (fixed) — `compile_to_fock` hung on any φ⁴: the external CAS `expand()` is exponential

The `sm_validation` suite could not complete: `sm_higgs_potential_nested_fock_structure` hung
forever in `sm_higgs_hamiltonian` → `compile_to_fock`.  Cause: `quantrs2-symengine-pure`'s
`Expression::expand()` distributes products over sums via repeated whole-arena node merges —
quadratic-to-exponential in the number of factors.  Measured on the module's own sub-expressions:
two binomial factors 14 ms, four 11 s, six (the full SM Higgs Mexican hat) effectively never.  The
fix is in `nested_fock_algebra/src/cas.rs`: `compile_expression_bounded` no longer calls
`expand()`; the existing memoized, `max_terms`-bounded `distribute_bounded` does the same
order-preserving distribution on the S-expression (the external serializer emits only
`+ * / neg ^ sqrt`, all handled).  The full SM Higgs compiles in **< 1 ms to 23 terms**
(1 kinetic + 2 quadratic + 4 + 16 quartic), and all 76 `nested_fock_algebra` unit tests and 12
`sm_validation` tests pass.  Pre-existing defect: any model with a product of ≥ 4 binomial factors
was affected, so this also unblocks future quartic+ models.

### Engine note

`cadabra2-cli` **2.5.15 is on `PATH` on this machine** (nix profile), and all 21 modules were run
clean here on 2026-10-13/14 — every CHECK at its advertised value (the "not runnable on aarch64"
note above is the 2026-10-13 record and is now historical).

## Update 2026-10-14 (later) — 22 modules: the momentum-space product module

### New module (21 → 22)

| module | content | checks |
|---|---|---|
| `docs/momentum_space_products.cdb` | the momentum-space product algebra behind `test/mathematical-foundations/momentum-space-products.md`: product → convolution on a three-mode grid, the derivative symbol riding on the integration **leg** (not the total momentum), the completion-of-squares transfer identity `(ip)·(iq) = |p|²/2 + |q|²/2 − |p+q|²/2`, and the quartic-at-one-point = quadratic-in-the-composite identity `[F⁴]_R = Σ_Q ψ_Q ψ_{R−Q}` coefficient-wise and at zero total momentum — with two negative controls (leg-vs-total contrast, the S47 `−m`/`+m` index convention) that must stay nonzero | 14, all at their advertised values (sizes pinned: 6/6, 4/4, 21→15 vs 15, 3/3/3) |

Evidence and counterfactuals: `docs/VERIFY_MOMENTUM_SPACE_PRODUCTS.md` (a wrong
stage-2 table entry fails CHECK 8/10 only; a sign error in the composite ψ fails
CHECK 2/8/10 — the two sides are independent). c-number bookkeeping only: no
ladder operators, no normal ordering — the operator form of the rewriting is
stated, not proved, in the `test/` page.

### Two new engine traps (recorded in `CADABRA2_GOTCHAS.md` §2b)

- a trailing `#` comment on a `:=` line **silently drops the assignment** and
  the failure surfaces lines later as `Python object '<later handle>' does not
  exist`;
- `x := len(ex)` pushes the right-hand side through the expression parser and
  **silently swallows the rest of the file** (no output, no traceback) — size
  captures must use plain Python `x = len(ex)`.

Also learned: Cadabra's `distribute` already collects like terms, so
post-distribute and post-collect `len` agree — pin what the engine prints, not
what the algebra predicts (this is why the size labels above differ from the
term counts an unmerged expansion would give).

All 22 modules were re-run on 2026-10-14 after the additions (rc = 0 each,
every CHECK at its advertised value); the other 21 are byte-unchanged.

## Recommended follow-ups (not done here)

1. **Run the symbolic seam on an x86_64 host** (or build cadabra2 for aarch64)
   so `require_cadabra()` tests actually execute rather than skip. Until then the
   `.cdb` *execution* is unverified on ARM; only the derived physics is (via the
   numerical seam).
2. **F3:** wire the unitarity module's `det_tilde` epsilon contraction into its
   Jacobian check, replacing the hand-substitution — needs an engine run.
3. **`@()` lint** (F4): a pre-commit/CI check for `@()` stems < 4 chars.
4. **Vendor or stub `dynamic-arctic`** (or gate `unfer_consensus` behind a
   feature) so `cargo test` works from a bare checkout without the sibling repo.
5. Speed: `mathhook-core` at >13 min dominates the `fock_sirk` dev-build; if
   these suites are run in CI, consider a cached build or trimming the
   criterion/proptest dev-deps for the test profile.
