# Verification: the vielbein (tetrad) Starobinsky 3D gauge-fixed Hamiltonian

Evidence base for `docs/qg_starobinsky_vielbein_hamiltonian.cdb` — the vielbein generalization of
the base module `qg_gauge_fixed_hamiltonian.cdb` (action → vary → polymomentum → Legendre
transform → `H`), and the source of the R + αR² Hamiltonian used by
`qg_starobinsky_vielbein_hamiltonian` in `nested_fock_algebra`.  What the module derives is

    action_st = (M²/2) ψ e R − U(ψ) e ,   ψ = 1 + 4αR/M² ,   U(ψ) = (M⁴/16α)(ψ−1)²
    H_final_st = (M²/2) ψ · (book.tex 8190) + U(ψ) e

Reproducible with Cadabra2 (2.5.14, via the `../unfer` nix flake):

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    $C2 -q -n docs/qg_starobinsky_vielbein_hamiltonian.cdb     # exit 0 (2026-09-18)

and, through the crate that consumes it (this is the check that pins the module):

    CADABRA_CLI=$C2 cargo test -p prob_kernel --lib qg_starobinsky_vielbein_hamiltonian_derivation_runs

The module's check values (`fR_check`, `R2_check`, `pi_psi`, `H_final_st`, `base_limit_check`) are
its last `collect_terms` targets; running without `-n` displays each reduction, and appending a
one-cell epilogue that `print`s them gives the machine-readable values quoted below.

## The defect that was repaired (2026-09-18)

The module **aborted** before its checks ran:

    Traceback (most recent call last):
      File "<string>", line 141, in <module>
    RuntimeError: Python object '\prod' does not exist.

**Cause: the `@(name)` insertion macro.**  `@(...)` does not look its argument up as a Python
name; it resolves it through the LaTeX parser, which reads `_` as a *subscript* separator unless
the stem before it is at least **four** characters long.  Measured on this build:

| reference | resolved as | result |
|---|---|---|
| `@(action_st)` | the whole name (`action_st`) | ok — stem `action` |
| `@(ex1_st)` | the Python object `ex1` | `RuntimeError: Python object 'ex1' does not exist.` |
| `@(H_8182_st)`, `@(H_final_st)`, `@(U_psi)` | a product node `\prod` | `RuntimeError: Python object '\prod' does not exist.` |
| `@(foo_bar)`, `@(a_s)`, `@(v_1)` | `foo`, `a`, `v` | same `does not exist` abort |

So the abort at `@(U_psi)` was only the first of **four** mis-resolving references:
`@(ex1_st)` (polymomentum), `@(H_8182_st)` and `@(H_final_st)` (the Legendre form and its
limits) would have aborted immediately afterwards.  Two repair idioms are now used in the module:

* an expression that must be *inserted into a math input* (`@()`) is named with a stem of four or
  more characters before its first underscore — `U_psi` → `Upot_psi` (referenced as
  `@(Upot_psi)`);
* a pure *duplication* is done in Python, `dst = src.copy()`, which needs no macro and hence no
  name constraint.  This is also semantically the right operation: Cadabra manipulates
  expressions in place, so the plain `dst = src` would alias and later corrupt `src` (verified:
  after `dst = src; substitute(dst, …)` the source changes too, while `src.copy()` is a deep
  copy).

Both idioms and the rule are documented in the module header (GOTCHA note) and in `AGENTS.md`.

A repo-wide audit of every `@()` reference in `docs/*.cdb` shows **no other module** uses an
underscored name with a stem shorter than four characters, so this defect was unique to this
module (`@(action_gf)` and `@(action_grav)` in the metric-route module have 6-character stems and
were never affected).

**Second defect — the advertised checks did not reach zero.**  Once the script ran to completion,
the two identities the header advertises as `→ 0` did *not* vanish: they were left as

    fR_check = α R² e − M⁴ α⁻¹ (R α M⁻²)² e
    R2_check = M⁴ α⁻¹ (R α M⁻²)² e − α R² e

i.e. the *power of a product* was never expanded, so `M⁴ α⁻¹ (R α M⁻²)²` never collapsed to
`α R²`.  The working sibling module does this with `expand` + `expand_power`
(`qg_starobinsky_hamiltonian.cdb`, `scalaron_check`); both steps were missing here and are now
added to the `fR_check` and `R2_check` chains.

**Third defect — the scalaron polymomentum had the wrong name.**  The header (lines 43 and 208)
and the Rust test `symbolic::tests::qg_starobinsky_vielbein_hamiltonian_derivation_runs` both name
the vanishing scalaron polymomentum `pi_psi_check`, but the module defined it as `pi_psi`.  The
test's extraction is name-based (`symbolic_derive(.., &[…"pi_psi_check"…])`) and a name that is
not defined makes it return `Err`, which the test `.unwrap()`s — so the test could not have been
green.  The variable is now `pi_psi_check`, matching the header, `AGENTS.md` and the test.

**Fourth defect — `unwrap` threw the action away.**  The chain copied the base module's cell 3
line-for-line, including `unwrap(ex1_st)` *before* `integrate_by_parts`.  But `unwrap` removes the
top node of **every term**, and by that point `ex1_st` is the *sum* produced by `product_rule`, so
the integrand collapses to its first term.  Measured on this build (a probe printing the state
after each step):

| step | `ex1_st` |
|---|---|
| after `product_rule` | the full expanded integrand, `∫( 1/4 M² ψ e … \partial_α(ω) … ){x}` |
| after `unwrap` | `-∫(U(ψ) e}{x}` — one term, the potential |
| after `ex1_st = ex1_st[0]` | `U(ψ) e`, and hence `pi_st = 0` |

With the `unwrap` line removed the module extracts `ex1_st` = the full vielbein-expanded action
density and `pi_st` = the full polymomentum `ψ (1/4 M² e … π_{βγ}^{j} …)` — which is what the
module's own documentation and the Rust test describe.  The `\int{{}}{{}}` wrapper must stay in
place until *after* `integrate_by_parts` (that is what lets the algorithm drop the total
derivative); the integrand is then child 0 of the `\int` node (`child 1` is the integration
variable).  The module now carries a comment saying so.

## CHECK 1 — the Jordan form is f(R): `fR_check = 0`

`action_st − ((M²/2)R e + α R² e)` with `U(ψ) → (M⁴/16α)(ψ−1)²` and `ψ → 1 + 4αR/M²`, then
`distribute` + `expand` + `expand_power` + `collect_factors` + the `M^{2n}(M²)^{-n} → 1` and
`α α⁻¹ → 1` collapses + `sort_product` + `canonicalise` + `collect_terms`:

    fR_check = 0

so R² gravity *is* the second-order scalar-tensor theory `(M²/2)ψR − U(ψ)` here too (no
Ostrogradsky ghost).  This is the vielbein counterpart of the metric route's `scalaron_check = 0`.

## CHECK 2 — the R² content and the scalaron polymomentum

    R2_check = U(ψ) e − α R² e    at ψ = 1 + 4αR/M²   →   0
    pi_psi   = ∂L/∂(∂₀ψ)          (ψ auxiliary)       →   0

`R2_check = 0` is the "αR²e is exactly the potential term" statement; `pi_psi = 0` is the
absence of a scalaron kinetic term in the Jordan frame (the kinetic only appears after the
conformal rescaling to the Einstein frame), which is what makes the Legendre transform linear in
the auxiliary `ψ`.

## CHECK 3 — `H_final_st = (M²/2) ψ · (book.tex 8190) + U(ψ) e`, coefficient by coefficient

The module prints, for the repaired run (coefficient of the term as displayed):

| term | book.tex 8190 | module prints | `(M²/2)ψ × 8190` |
|---|---|---|---|
| `𝒮_ab 𝒮^ab e⁻¹` | `1/16` | `M²/32 ψ` | `1/32` ✓ |
| `𝒫² e⁻¹` | `−1/24` | `−M²/48 ψ` | `−1/48` ✓ |
| `T_abc T^acb e` | `1/2` | `M²/4 ψ e` | `1/4` ✓ |
| `T_abc T^abc e` | `1/4` | `M²/8 ψ e` | `1/8` ✓ |
| `T^ab_b T_ac^c e` | `−1` | `−M²/2 ψ e` | `−1/2` ✓ |
| `T_ab T^ab e` | `−1/4` | `−M²/8 ψ e` | `−1/8` ✓ |
| `E^ab T_ab e` | `−1` | `−M²/2 ψ e` | `−1/2` ✓ |
| `E^ab 𝒮_ab` | `1/2` | `M²/4 ψ` | `1/4` ✓ |
| `E^a_a 𝒫` | `1/3` | `M²/6 ψ` | `1/6` ✓ |
| `E_a T^ab_b e` | `2` | `M² ψ e` | `1` ✓ |

plus the potential density `+ U(ψ) e`.  All ten coefficients agree with `(M²/2)ψ·(8190)`
(exact rational arithmetic), and the scale factor is exactly `(M²/2)ψ` with no residual `e` or
`ψ` dependence in the coefficients — i.e. the Legendre transform is linear in the auxiliary
scalaron field, as claimed.  The `p^a T_a` vector terms are dropped as a boundary term, as in the
base module.

## CHECK 4 — the α → 0 limit: `base_limit_check = (M²/2) · (book.tex 8190)`

With `U(ψ) → 0`, `ψ → 1` the same ten terms appear **without** the `ψ` factor and without the
potential: the base TEGR 3D gauge-fixed Hamiltonian scaled by `M²/2` — so the module is a genuine
generalization of `qg_gauge_fixed_hamiltonian.cdb`, not a different theory.  (The base module's
own `H_final` reproduces book.tex 8190 with the coefficients `1/16, −1/24, ½, ⅓`, see
`VERIFY_QG_DENSITIZED.md`.)

## CHECK 5 — the derivation itself: `ex1_st` is the expanded action and `pi_st` its polymomentum

After the fourth defect was repaired:

    len(ex1_st)        = 121        (the expanded action density as a sum; a single term before)
    pi_st              = ψ ( 1/4 M² e e^{μ}_a e^{β}_b η^{ab} e^{α}_c e_μ^{a2} … π_{βγ}^{j} + … )

i.e. the polymomentum by variation of the full vielbein action, with the `(M²/2)ψ` factor pulled
out by `factor_out` — exactly `(M²/2)ψ·π₀` as documented.

**The module's Rust test now passes** — on a replay of its extraction (the crate itself cannot be
built in this workspace: `cargo` needs the network for the `nanoda_lib` git dependency, so
`cargo test -p prob_kernel` was *not* run here; what was run is the code path it drives, below).
`prob_kernel::symbolic_derive` appends an extraction
trailer to the module and runs `cadabra2-cli -q -n <file>`; that exact trailer
(`UNFER_DERIVE|<name>|<value>`, with a missing name reported as `ERR` and turned into `Err` by
`symbolic_derive`) was replayed here on the repaired module for all ten names the test requests —
`action_st`, `fR_check`, `R2_check`, `ex1_st`, `pi_st`, `pi_psi_check`, `t0_tegr`, `H_8182_st`,
`H_final_st`, `base_limit_check` — and all eight of the test's assertions hold
(`action_st`: ψ/R/U; `fR_check`, `R2_check`, `pi_psi_check` = `0`; `ex1_st`: `\partial` present;
`pi_st`: π and ψ present; `H_final_st`: ψ and U present; `base_limit_check`: non-trivial, no U).
Before the repair the test aborted in `symbolic_derive` (the module itself aborted) and, with the
abort fixed but defects 2–4 present, it failed on `fR_check`/`R2_check` (not zero) and on the
missing, collapsed `pi_psi_check`/`pi_st`.

## Related finding: the same `unwrap` bug in the base module — **repaired 2026-09-18**

`docs/qg_gauge_fixed_hamiltonian.cdb` — the chain this module generalizes — had the identical
`unwrap(ex)` before `integrate_by_parts`, with the same effect:

| | `len(ex1)` | `len(pi_derived)` | extracted `ex1` |
|---|---|---|---|
| `qg_gauge_fixed_hamiltonian.cdb`, as committed | 18 (one product of 18 factors = one summand) | 2 | 199 chars |
| with `unwrap(ex)` removed | 120 summands (the full `eR` integrand) | 132 | 18 181 chars |

so its `ex1` (described as "the Einstein-Hilbert action density `eR` in the vielbein") and its
`pi_derived` ("the polymomentum by variation of `ex1`") were computed from a single term of the
expanded action.  Unlike in the vielbein module this was **silent**: the base module does not
abort, and the Rust test `qg_gauge_fixed_hamiltonian_derivation_runs` only asserts that `ex1` is
non-empty and tetrad-shaped, so nothing caught it.

The defect was found by a **module-level audit of all seven `docs/*.cdb` modules** (same engine,
epilogue that dumps every named expression, plus counterfactual runs) and repaired by removing the
single `unwrap(ex)` line — see `docs/VERIFY_CDB_TRUNCATION_AUDIT.md`.  After the repair `ex1` is the
120-summand `eR` integrand and `pi_derived` the 132-summand polymomentum; `G`, `t0_tegr` and
`H_final` are byte-identical to before (the 8190 Hamiltonian was never affected); the module runs
clean in 2.7 s; and all 17 assertions of the Rust test pass on a replay of its extraction trailer.
That audit also found **no other truncated module**: the other six (this one included) compute
their advertised expansions in full.

Independent cross-check between the two modules (the regression test for this repair): this module
computes `ex1_st = ∫(M²/2)ψeR − U(ψ)e` through the same cell-3 chain and measures
`len(ex1_st) = 121`; the repaired base module measures `len(ex1) = 120`, and
`(M²/2)ψ × (its first term)` is this module's first term coefficient by coefficient — `120 + 1`
(the potential) `= 121` and `(M²/2)ψ·(1/2) = 1/4 M²ψ`.  The two independent runs agree.

## What is an identity vs. what is an input

* **Pure algebra, verified above:** the Jordan equivalence `f(R) = (M²/2)ψR − U(ψ)`, the R²
  content `αR²e = U(ψ)e`, `π_ψ = 0`, the factorization `pi_st = (M²/2)ψ·π₀`, the scaling
  `H_final_st = (M²/2)ψ·(8190) + U(ψ)e` and the α → 0 limit.
* **Physical input (boundary term):** dropping `p^a T_a` / `2eT^{ac}_c T_a` as a surface term
  (physical states of vanishing spatial fall-off) — the same input as the base module and
  book.tex 8190's `≈`.
* **Out of scope here:** the conformal rescaling from the Jordan variable `ψ` to the
  Einstein-frame scalaron `φ` (the module states it is the standard f(R) chain; the
  Einstein-frame potential `V(φ) = (M⁴/16α)(1 − e^{−√(2/3)φ/M})²` and its ESA are verified in
  `VERIFY_QG_DENSITIZED.md` and formalized in `timepiece/BookProof/ChapterScalaronWallEsa`).
* **Not the teleparallel rewriting:** the module computes the metric-compatible, torsion-free
  Ricci chain `e^a_μ e^b_ν ω(e)`; it does not execute the Weitzenböck `R = T + B` rewrite.

## Files

* `docs/qg_starobinsky_vielbein_hamiltonian.cdb` — the module (repaired; header carries the
  `@()`/underscore GOTCHA note).
* `docs/qg_starobinsky_hamiltonian.cdb` — the metric route; its `scalaron_check = 0` was the
  pattern for the `expand`/`expand_power` repair.
* `docs/qg_gauge_fixed_hamiltonian.cdb` — the base TEGR/teleparallel chain generalized here.
