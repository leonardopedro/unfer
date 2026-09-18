# Audit: silent truncation-to-one-term in the `docs/*.cdb` derivations

Date: 2026-09-18.  Scope: **every** Cadabra module in `docs/*.cdb` (7 files).  Question: is any
derivation silently reduced to *one term* of a sum — i.e. does a chain compute a value the module
(and the docs citing it) describe as a full expansion, while the expression actually carries a
single summand?  The specific mechanism under suspicion is `unwrap` applied to a summed expression
(this is what collapsed `ex1_st` in `qg_starobinsky_vielbein_hamiltonian.cdb`, repaired earlier the
same day), plus indexing (`expr[n]`) on a sum.

## Method

Every module was run **as committed**, with an epilogue appended that dumps every named Cadabra
expression together with its size, so a collapsed value is visible directly.  The size printed is
`len(ex)`, which in Cadabra2 is the number of children of the top node — the number of *summands*
for a sum and the number of *factors* for a product.  That is exactly what distinguishes the two
states below: `len(ex1) = 18` is a **single** product of 18 factors (one summand), while
`len(ex1) = 120` is a sum of 120 summands:

    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    cat docs/<module>.cdb <epilogue>.cdb > /tmp/<module>.cdb
    $C2 -q -n /tmp/<module>.cdb          # epilogue prints:  CHK|<name>|LEN=<n>|<value>

For every module the measured values were then compared against the checks the module advertises
(its own `print`s and the `*_check` / `*chk*` handles the docs quote).  For the suspect mechanism a
**counterfactual** run was made with the construct neutralised, and the two were diffed.  Finally
the extraction trailer of `prob_kernel::symbolic_derive` (the exact one the Rust tests appends —
`UNFER_DERIVE|<name>|<value>`, `eval`-based, 120 s budget) was replayed, and the test assertions
emulated, on both the before and after states.  `cargo test -p prob_kernel` itself cannot run in
this workspace (the `nanoda_lib` git dependency needs network), so what was run is the code path it
drives.

## Result

| module | truncation-prone construct | as-committed measurement | verdict |
|---|---|---|---|
| `qg_gauge_fixed_hamiltonian.cdb` | `unwrap(ex)` before `integrate_by_parts` | `ex1` = **18** (one product of 18 factors = one summand), `pi_derived` = 2 summands; without the line: `ex1` = **120** summands, `pi_derived` = **132** summands | **truncated — repaired (below)** |
| `qg_starobinsky_vielbein_hamiltonian.cdb` | `unwrap` (repaired 2026-09-18) | `ex1_st` = **121** summands, `pi_st` = ψ·π₀, `fR_check = R2_check = pi_psi_check = 0`, `H_final_st` 11 summands, `base_limit_check` 10 summands | clean (already repaired) |
| `qg_unitarity_check.cdb` | `unwrap(lhs)`, `unwrap(rhs)` (lines 72/77) on plain sums | `Jcheck = 0`, `Unorm = 0`, `Hkern = 0`; `lhs` keeps **both** terms | clean — see note (b) |
| `ns_qg_fourier_elimination.cdb` | none (no `unwrap`, no sum indexing) | A1–E3 all hit their advertised values (`0`, `t**3`, `t**3-1`, …); C1/C4 non-trivial; exit 0 | clean |
| `qg_starobinsky_hamiltonian.cdb` | none | all 12 advertised checks = `0` (`scalaron_check`, `V3_check`, `dV3_check`, `Vphi_zero`, `constraint_check`, `action_R2_check`, `pi_check`, `leg_check`, `pi_grav_check`, `leg_grav_check`, `gf_check_Dphi2`, `gf_check_Rc`); `pi_derived` = `∂₀φ π`, `pi_grav` = `¼M² ∂₀q Π` | clean |
| `qg_densitized_hamiltonian.cdb` | none | `Ktrans` = `1/16 S̃_ab S̃^ab − 1/24 (P̃)²` = `Kflat`; `diff = 0` | clean — see note (a) |
| `yang_mills_hamiltonian.cdb` | `unwrap` present but **commented** | `H_final` = `½π² + ½B²` (2 terms); `G`, `G_y`, `ss`, `s2`, `s3` all multi-term (3/3/13/6/6) | clean |

Also checked: **no module has an active `unwrap` inside `post_process`.**  Three of the seven define
a `post_process` (`qg_gauge_fixed_hamiltonian`, `qg_starobinsky_vielbein_hamiltonian`,
`yang_mills_hamiltonian`) and in all three the `unwrap(ex)` line is commented out.  That is the
worst-case variant of this defect — a `post_process` `unwrap` runs on the output of *every*
algorithm in the module, so it would corrupt every sum in the file.

**One module was truncated: `qg_gauge_fixed_hamiltonian.cdb`.**  Everything else computes its
advertised expansions in full.

Re-certified on the tree as it then stood (2026-09-18), with the same epilogue: all seven modules
`exit 0` with no traceback, and the two post-repair values hold —
`qg_gauge_fixed_hamiltonian.cdb` `pi_derived` = **132** summands (18 → 120 for `ex1`), and
`qg_starobinsky_vielbein_hamiltonian.cdb` `ex1_st` = **121** summands.

## The finding, measured

`docs/qg_gauge_fixed_hamiltonian.cdb` had, at line 147 (as committed before this audit):

    product_rule(ex)
    unwrap(ex)                 # <-- the defect
    distribute(ex)
    rename_dummies(ex);
    integrate_by_parts(ex, $\partial_{\nu}{e_{\alpha}^{a}}$)
    ex1 = ex[0];

`ex` is `\int{e R}{x}`.  After `product_rule` the integrand is a sum of 60 terms, and the whole
thing is one `\int{...}{x}` node.  `unwrap` removes the top node of every term, so on this single
`\int` node it drops the wrapper and leaves `ex[0]` pointing at the **first term of the integrand**
instead of the integrand itself.  Measured, same engine, same input:

| | `len(ex1)` | `len(pi_derived)` | extracted `ex1` string | extracted `pi_derived` string |
|---|---|---|---|---|
| as committed | **18** = a single product of 18 factors → 1 summand | **2** | 199 chars | 377 chars |
| with `unwrap(ex)` removed | **120** = a sum of 120 summands | **132** | 18 181 chars | 20 289 chars |

So `ex1` — documented in the module header as "the Einstein-Hilbert action density `eR` in the
vielbein", and in the Rust test's doc-comment as "`ex1` (=eR)" — was one summand of the expanded
action, and `pi_derived` — "the polymomentum BY VARIATION of the Lagrangian (the coefficient of
`∂_α(d e^k_ρ)`)" — was correspondingly partial.  `G`, `t0_tegr` and `H_final` are computed by
chains that do not use `ex1`, and are **byte-identical** before and after the repair (100 / 279 /
305 chars), so the final 3D gauge-fixed Hamiltonian (book.tex 8190) was never affected.

### Why it was silent

Two independent reasons, both worth recording:

* the module **does not abort** — the expression is well-formed, just short.  (In the vielbein
  module the same construct was caught because that module was *already* aborting for an unrelated
  reason, the `@()` name resolution; the `unwrap` was found only by probing, not by the abort.)
* its Rust test does not pin the content: `qg_gauge_fixed_hamiltonian_derivation_runs` asserts only
  that `ex1` is non-empty and "tetrad-shaped" (`contains("e^{") || contains("e_{")`) and that
  `pi_derived` is non-empty and contains π.  An 18-factor single term satisfies all of that.

**Recommendation (not applied here):** make the class non-silent by adding a *size* assertion to
that test, e.g. that the extracted `ex1` is the summed integrand rather than a single term — the
measured contrast is unambiguous (199 chars / one summand before, 18 181 chars / 120 summands
after), so any threshold in between separates them.  A guard phrased on the value (not on the source text)
would have caught this defect and will catch a recurrence of the same idiom.

## The repair and its verification

The fix is the idiom already used by `qg_starobinsky_vielbein_hamiltonian.cdb`: keep the `\int{}{}`
wrapper until **after** `integrate_by_parts` (that is what lets the algorithm drop the total
derivative), take the integrand with `[0]` afterwards, and do **not** `unwrap`.  Applied as a
comment-out of the single `unwrap(ex)` line with an in-place note explaining why it must not come
back (no other line changed).

Re-verified after the repair:

* module runs clean, **exit 0, in 2.68 s** — far inside the 120 s budget the Rust test passes to
  `symbolic_derive`;
* `ex1` = a sum of **120** summands = the full `eR` integrand; `pi_derived` = **132** summands; the
  extracted strings are 18 181 / 20 289 chars;
* `G`, `t0_tegr`, `H_final` unchanged (byte-identical extraction), `H_final` still the 10-term
  book.tex-8190 form carrying `1/16`, `1/24`;
* all **17** assertions of `qg_gauge_fixed_hamiltonian_derivation_runs` pass on a replay of its own
  extraction trailer;
* **cross-check against the repaired sibling module**: `qg_starobinsky_vielbein_hamiltonian.cdb`
  computes `ex1_st = \int{(M²/2)ψeR − U(ψ)e}{x}` through the same cell-3 chain and measures
  `len(ex1_st) = 121`.  The repaired base module measures `len(ex1) = 120` and its first term is

      base:      1/2  e … ∂_α(η^{c a1}) … ∂_β(e^{j}_{μ})
      vielbein:  1/4 M² ψ e … ∂_α(η^{c a1}) … ∂_β(e^{j}_{μ})

  with `(M²/2)ψ · 1/2 = 1/4 M²ψ` exactly — i.e. the repaired base `ex1` is the integrand the
  vielbein module is `(M²/2)ψ ×` of, plus that module's one extra potential term
  (`120 + 1 = 121`).  The two independent runs agree term for term in count and coefficient.

## Adjacent findings (not truncation, recorded for completeness)

**(a) `qg_densitized_hamiltonian.cdb`: the "verification" cannot fail.**  Its `diff` handle is a
*hand-typed tautology* — `(1/16)S̃S̃ − (1/24)P̃P̃ − (1/16)S̃S̃ + (1/24)P̃P̃` — written out literally
instead of being formed from the computed `Ktrans` and `Kflat` via `@()`.  It is `0` for any input,
so it verifies nothing about the substitution chain.  The *mathematics* is nevertheless correct
here: `Ktrans` measured after its substitution chain is exactly `1/16 S̃_ab S̃^ab − 1/24 (P̃)²`,
which is `Kflat`.  Contrast the modules that do form real differences (`ns_qg_fourier_elimination`
uses `@(poly1) - @(leib1)`, `qg_starobinsky_hamiltonian` uses `@(...) − @(...)`), and which
therefore can fail.  Suggest rewriting `diff := @(Ktrans) - @(Kflat)` so the check is live.

**(b) `qg_unitarity_check.cdb`: `unwrap` on a plain sum is benign.**  Lines 72/77 call
`unwrap(lhs)` / `unwrap(rhs)` where both are ordinary sums of two terms (no `\int` wrapper).  Unlike
the `\int` case above, this does **not** truncate: `unwrap` removes the top node of each term, so
`∂_x(ψ∂_xφ) − ∂_x(φ∂_xψ)` becomes `ψ∂_xφ − φ∂_xψ`, both terms kept, and the subsequent
`product_rule` + cancellation then gives `Hkern = 0` as documented (measured).  The distinction is
therefore precise: **the trap is `unwrap` on an `\int{...}{x}`-wrapped (single-node) expression
before `[0]`**; on a multi-term sum at the top level it does what the author intended.  The idiom to
keep is the one the two vielbein/base modules now use.

**(c) some summands carry a derivative of the constant metric.**  The repaired `ex1` contains
subexpressions like `∂_α(η^{c a1})` — **16** occurrences; η is the flat metric, hence constant, so
those summands are identically zero.  It is an artifact of substituting
`g^{μν} → e^μ_a e^ν_b η^{ab}` *before* `product_rule`, inherited from the original notebook; the
vielbein module's repaired `ex1_st` opens with the same term.  This is a cosmetic non-reduction,
not a truncation, and does not affect `H_final`.  An optional cleanup would add
`substitute(ex, $\partial_{\alpha}{\eta^{a b}} -> 0$, repeat=True)` after `product_rule`; it is
left out here because it changes `ex1`/`pi_derived` again and the module's stated target (the 8190
Hamiltonian) is unaffected either way.

## Files

* `docs/qg_gauge_fixed_hamiltonian.cdb` — repaired (one line removed, with an in-place note);
  this is the only source change of the audit.
* `docs/VERIFY_QG_STAROBINSKY_VIELBEIN.md` — its "related finding (not repaired here)" section now
  points at this audit and records the repair.
* `AGENTS.md` — the QG bullet records the audited class and the repair.

## Reproduction

```
C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
for m in ns_qg_fourier_elimination qg_densitized_hamiltonian qg_gauge_fixed_hamiltonian \
         qg_starobinsky_hamiltonian qg_starobinsky_vielbein_hamiltonian qg_unitarity_check \
         yang_mills_hamiltonian; do
  $C2 -q -n docs/$m.cdb        # + the CHK epilogue for term counts
done
$C2 -q -n docs/qg_gauge_fixed_hamiltonian.cdb   # exit 0, ex1 = the 120-term eR integrand
```
