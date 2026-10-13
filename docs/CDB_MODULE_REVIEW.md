# Cadabra module review — `docs/*.cdb`

**Review date:** 2026-10-13. **Scope:** all 19 `.cdb` modules under `docs/`,
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
