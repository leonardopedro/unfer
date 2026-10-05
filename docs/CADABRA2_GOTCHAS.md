# Cadabra2 in this repository — operational gotchas

The reusable, hard-won knowledge about driving Cadabra2 (GPL-3.0) as a
**subprocess** from `prob_kernel::symbolic`. Extracted from `AGENTS.md` (project
review item **X3**, finding **G-U2**) because it was buried in a checklist
paragraph where nobody could find it.

**Why a subprocess and never a link.** Cadabra2 is GPL-3.0; these binaries are
Apache-2.0. `cadabra2-cli` is invoked as an external program
(`CADABRA_CLI` overrides the path), so no GPL code is linked into the Rust
binaries and the licence boundary is unaffected. Same pattern as the Why3 (LGPL)
seam. Keep it that way — vendoring the engine would change the licence of
everything downstream.

**Engine-dependent tests skip when `cadabra2-cli` is absent.** That is the
Cadabra2 skip pattern, also used by the GPU suites in `delta_algebra`. A skip is
not a pass: a green run on a machine without the engine has verified nothing
about these modules. `flake.nix` provides it in the dev shell.

---

## 1. The `@()` four-character rule

**The trap.** `@(name)` resolves its argument through the LaTeX parser, and that
parser reads `_` as a subscript separator *unless the stem before it is four or
more characters*. So:

| reference | resolves? |
|---|---|
| `@(action_st)` | ✅ stem `action` is 5 chars |
| `@(U_psi)` | ❌ mis-resolves to a `\prod` node |
| `@(ex1_st)` | ❌ |
| `@(H_8182_st)` | ❌ |
| `@(H_final_st)` | ❌ |

A mis-resolved reference aborts the run with:

```
RuntimeError: Python object '<…>' does not exist
```

**The rule: use a ≥4-character stem for anything referenced by `@()`.** If a
name genuinely cannot be renamed, use `dst = src.copy()` instead — but note
`dst = src` *aliases*, and Cadabra manipulates expressions in place, so the
copy is mandatory.

This bug aborted the Starobinsky-vielbein module, and its `unwrap` defect (below)
collapsed `ex1_st` to a single term, until 2026-09-18.

## 2. The `unwrap` trap

`unwrap` strips the top node of **every** term. Two distinct failure modes:

**In a wrapped derivation.** After `product_rule` the integrand is a *sum*, so
`unwrap` keeps only the first term. Keep the `\int` wrapper until *after*
`integrate_by_parts`, then take the integrand with `[0]` — **without** `unwrap`.

**In `post_process`.** An active `unwrap` there runs on the output of *every*
algorithm in the file and would corrupt every sum in it. The module-level audit
confirmed no module has one. If you add one, that audit result is no longer true.

**Benign case.** `unwrap` on a *plain* sum acts termwise, so the
`unwrap(lhs)`/`unwrap(rhs)` calls in `qg_unitarity_check.cdb`'s Hkern checks are
fine — `Hkern` still reaches 0. Content reference, not a line number: line
numbers shift when the module-documentation headers change.

## 3. Silent truncation — the failure mode to watch for

The base module `qg_gauge_fixed_hamiltonian.cdb` had a *silent* truncation: its
`ex1` (documented as the Einstein–Hilbert action density `eR` in the vielbein) was
**one summand** of the expanded action — `len(ex1)` 18 (a single product of 18
factors) against 120, and `pi_derived` 2 summands against 132. Repaired
2026-09-18 by removing the offending line.

**Nothing caught it**, because the Rust test asserted only that `ex1` is
non-empty and tetrad-shaped.

The `unwrap` defect was then found in the base module too, which is why there is
now a module-level audit: [`VERIFY_CDB_TRUNCATION_AUDIT.md`](VERIFY_CDB_TRUNCATION_AUDIT.md)
dumps the size of every named expression, runs counterfactuals, and replays the
Rust extraction trailer. It covers all seven `docs/*.cdb` modules and confirmed
the base module was the only truncated one.

**The general lesson, and it is not Cadabra-specific:** a CHECK that reduces to 0
can still be asserting nothing. Any assertion of the form "the derivation
produced *a* thing" should also assert *how much* of it there was. The audit
above exists because the alternative is a test suite that reports green on an
empty result.

## 4. Two adjacent non-truncation findings

- The `diff` handle in `qg_densitized_hamiltonian.cdb` is a **hand-typed
  tautology** that cannot fail — it never references `Ktrans`/`Kflat`.
- `unwrap` on a plain sum is benign (see §2).

Both are recorded rather than fixed: fixing the first means deciding what it was
*meant* to compare, which is a physics question.

## 5. Reading the extraction pipeline

The pipeline reads only the run's **stdout markers**, so the comment headers
added to every `docs/*.cdb` module on 2026-10-04 are inert to it. Two stale
line-number references in `AGENTS.md` and `VERIFY_CDB_TRUNCATION_AUDIT.md` were
converted to content-based references at the same time, so future header edits
cannot rot them — keep it that way.

## 6. Error codes

| code | meaning |
|---|---|
| `UK-4901` | engine missing (`cadabra2-cli` not found) |
| `UK-4902` | malformed expression |

`verified` in the report is the **zero-detection** verdict (`H - H† = 0`, i.e.
Hermiticity), not a proof of anything beyond that.

## 7. The dialect round trip

`symbolic_analyze` canonicalizes the expression (TeX subset, or the `c_0 * a_0`
CAS dialect); `normalize_to_cas_dialect` translates Cadabra2's **braced** output
back into the dialect `compile_to_fock` accepts, closing the loop into a numerical
`Hamiltonian`. Both directions are load-bearing: a module that emits Cadabra's
braced form without the reverse translation produces a `Hamiltonian` the compiler
cannot read, and the failure surfaces much later as an empty operator.

---

## Module index

Each `.cdb` module has a **MODULE DOCUMENTATION** header giving its purpose, its
role in the programme, its certified content, and the improvement notes tied to
the project's goals. Evidence lives in:

| module(s) | evidence |
|---|---|
| `faris_lavine_n_ns.cdb`, `faris_lavine_n_qg.cdb` | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| `faris_lavine_n_sm.cdb` | [VERIFY_SM_FARIS_LAVINE.md](VERIFY_SM_FARIS_LAVINE.md) |
| `qg_gauge_fixed_hamiltonian.cdb` and the truncation audit | [VERIFY_CDB_TRUNCATION_AUDIT.md](VERIFY_CDB_TRUNCATION_AUDIT.md) |
| `qg_densitized_hamiltonian.cdb`, `qg_unitarity_check.cdb` | [VERIFY_QG_DENSITIZED.md](VERIFY_QG_DENSITIZED.md) |
| `qg_starobinsky_hamiltonian.cdb` | [VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md](VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md) |
| `qg_starobinsky_vielbein_hamiltonian.cdb` | [VERIFY_QG_STAROBINSKY_VIELBEIN.md](VERIFY_QG_STAROBINSKY_VIELBEIN.md) |
| `ns_qg_fourier_elimination.cdb` | [VERIFY_NS_QG_FOURIER.md](VERIFY_NS_QG_FOURIER.md) |
| `ns_kvn_equation.cdb`, `ns_pressure_poisson.cdb`, `ns_pressure_constraint.cdb` | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| `ns_lagrangian_hamiltonian.cdb` | [VERIFY_NS_QG_FOURIER.md](VERIFY_NS_QG_FOURIER.md) |

Related: [NUMERICAL_VALIDATION_GUIDE.md](NUMERICAL_VALIDATION_GUIDE.md) for the
numeric side, [ARCHITECTURE.md](ARCHITECTURE.md) for the crate map,
[../ATTRIBUTION.md](../ATTRIBUTION.md) for the licence boundary.