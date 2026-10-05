# Maintenance checklist — stage index

The machine-readable companion to `AGENTS.md`'s checklist (project review item
**X3**, finding **G-U2**).

`AGENTS.md` used to carry each stage's *entire* derivation inline — twelve
checklist entries ran 2,800–13,700 **characters** each, so a 219-line file was
118 KB and no checklist item was skimmable. Two things were wrong with that, and
this page fixes both:

1. **The invariants were undiscoverable.** A one-line rule buried in a
   13,000-character paragraph is not a checklist item; it is a wall. Every rule
   below is one line.
2. **The detail had nowhere to live** except the wall. Everything that did not
   already have an evidence page now has one, and every rule below points at it.

**Nothing was dropped.** Where a stage's prose had no home, it was moved here or
to the page named in its row — not summarised away.

---

## Trust and security

| stage | invariant (one line) | detail |
|---|---|---|
| S21 | `observe`-kind auto-applies, `mutate`-kind queues; a new `effect_kinds` entry must keep `GrantSet::is_subset_of` denying Mutate→Observe relabeling. | [PROTOCOL.md](PROTOCOL.md) |
| S22 | `admin.rs` mints the admin once from `UNFER_ADMIN_PRINCIPAL`; hard keys are never patchable and a refusal leaves the soft config byte-identical. | [RUNBOOK.md](RUNBOOK.md) |
| S23 | `uk_audit_append` / `uk_report_issue` must run `sanitize_sensitive` before storing; the ring is capped (512, drop-oldest). | [PROTOCOL.md](PROTOCOL.md) |
| S23/S24 | the golden manifest regenerates **only** via `UPDATE_GOLDEN=1`. | `unfer_data::release` |
| S25 | metered symbols are denied at the loopback chokepoint (`UK-4601`/`UK-4602`), never post-hoc; `uk_meter_status` stays read-only. | [PROTOCOL.md](PROTOCOL.md) |
| S26 | once `sensitive` is set, `fetch`/`agent_spawn`/`blueprint_export`/`action_submit`/`gate_approve` are refused `UK-4701` until an operator clears it. | [PROTOCOL.md](PROTOCOL.md) |
| S27 | secrets go through `uk_secret_put/get/revoke` and must never serialize into a `SessionBlob` or a `.cell` blueprint. | [MODULE_RECIPE.md](MODULE_RECIPE.md) |
| S28 | capabilities are minted only at the loopback chokepoint carrying the caller's grant set; a stub is re-checked against the original caller. | [PROTOCOL.md](PROTOCOL.md) |
| S29 | a new `uk_*` symbol goes in `EXPECTED_SYMBOLS.txt`, the generated C header, australVM's `UNFER_SYMBOLS`, and `GrantSet.kernel`; a new `KernelEvent` must be covered by the `handles.rs` matchers. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| S46 | every result-producing op clears `last_result` on entry, so a FAILED op leaves `uk_get_result` **empty**. | [PROTOCOL.md](PROTOCOL.md) |

## Symbolic coupling (Cadabra2 — read [CADABRA2_GOTCHAS.md](CADABRA2_GOTCHAS.md) first)

| stage | invariant (one line) | detail |
|---|---|---|
| S30 | Cadabra2 is a **subprocess**; `verified` is the zero-detection verdict (`H - H† = 0`); engine-dependent tests **skip** without `cadabra2-cli`. | [CADABRA2_GOTCHAS.md](CADABRA2_GOTCHAS.md) |
| S30f | Faris–Lavine `N` for NS/QG: every numbered CHECK reduces to 0. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30g | the NS Koopman–von Neumann generator is `H(x) = π^i(u_j u_{i,j} + q_i − ν u_{i,jj}) + h.c.`; the literal `H²` is **not** a valid `N`. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30h | the pressure Poisson equation follows by contracting `∂_i` with the momentum equation under `div u = 0`. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30i | the pressure is a **second-class** constraint — eliminate it, do not BRST it. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30j | the Jordan→Einstein frame step is what makes `starobinskyV` derived rather than merely defined. | [VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md](VERIFY_QG_STAROBINSKY_EINSTEIN_FRAME.md) |
| S30k | `N` must be symmetric on a **core** and that core must be a graph core; the lift `dΓ(N)` is a *different* operator, so ESA is reproved on it. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30l | the §9 audit's `L²(ℝ^{18n})` was stale — the proof's `nsFockSpace` is `L²(ℝ^{21n})`. | [VERIFY_FARIS_LAVINE_N.md](VERIFY_FARIS_LAVINE_N.md) |
| S30m | the SM module certifies CHECK 1–28; the Grassmann/CAR, spinor/Lorentz, parameter, and Hilbert-space statements are **not** certified and are Lean. | [VERIFY_SM_FARIS_LAVINE.md](VERIFY_SM_FARIS_LAVINE.md) |

## Verification cycles

| stage | invariant (one line) | detail |
|---|---|---|
| S31 | `logos_compile` reduces to a UNF; `verified` is a confluence self-check. Proof terms must stay `rfl` — `native_decide` emits terms nanoda cannot reduce. | [LOGOS.md](LOGOS.md) |
| S36 | Why3 is **subprocess-only** (LGPL); `check_with_grants` is pure so tests never mutate the process env. | [WHYML_CYCLE.md](WHYML_CYCLE.md) |
| S36b | run australVM's OCaml tests via `make -C australVM test` (X4b picks the loader); the old `run`-name collision note no longer applies (X4a). | [../australVM/Makefile](../australVM/Makefile) |
| S37 | Austral→DeltaNet→UNF→TED; `ted_hash` is the content-addressable algebraic hash on top of `unf_hash`. | [LOGOS.md](LOGOS.md) |

## Numeric validation

Every stage in this group has its full write-up in
**[NUMERICAL_VALIDATION_GUIDE.md](NUMERICAL_VALIDATION_GUIDE.md)**, which is the
pedagogical guide covering all suites, the SIRK algorithm, frames and guards.

| stage | invariant (one line) | guide |
|---|---|---|
| S32 | QED matches published results; the γ↔e⁺e⁻ vertex **must** list positron annihilation before electron creation. | §QED |
| S33 | QCD colour factors are computed from the SU(3) structure constants, not hard-coded; `C_F·e²` multiplies the amplitude square. | §QCD |
| S34 | QG reproduces the Planck/redshift/precession/deflection/GPS anchors; `qg_graviton_dispersion_sirk` gives `ω = c|k|`. | §QG |
| S35 | NS conventions: `u = a†+a`, `π = i(a†−a)`, `H = {π,V}` (the symmetrization carries the factor 2). | §NS |
| S38 | SR/nuclear + astro/plasma anchors; the charge cancels in `B = pc/(qρ)`. | §SR-astro |
| S39 | physics anchors II; the Gaussian year is **365.2568983 days**. | §anchors-II |
| S40 | gauge-fixed programme validation; distinct inner occupations of one universe merge under `scale_and_add`. | §gauge-fixed |
| S41 | SIRK dynamics/constraints; field amplitude `u = a†+a` is the decaying NS observable, not `N`. | §sirk-dynamics |
| S42 | Hashimoto Theorem 4.1 bands are **rigorous but conservative** on unitary models. | §hashimoto |
| S43 | S43 promotes bands to error bars; the sharp tier is Rayleigh–Ritz residual certificates, not the envelope. | §certified |
| S44 | QYM mass gap on the **Cadabra-derived gauge-fixed** Hamiltonian, not the lattice builder; the `g²/2` reading was a lattice-electric effect. | §mass-gap |
| S45 | the nested-Fock ground is **always** the outer vacuum; `H\|Ω⟩ = 0` identically for the full Hamiltonian. | §outer-vacuum |
| S47 | DFT convention: `f⋆g[n] = Σ_m f[m]g[(n−m) mod n]` — **not** `g[(n+m)]`. | §convolution |

## Cross-cutting rules that used to be paragraphs

- **Trust annotations (S21)** — `EffectKind::{Observe, Mutate}`; `uk_registry_vetted`
  is console-only (UK-4501 to non-hook callers) and never touches the approval lane.
- **Symbols never serialize secrets** — see S27. A new secret-plausible field must
  be added to the sanitizer or the secret-scan gate test fails.
- **The board's redaction is value-level** — the board scrubs secrets out of free
  text (`unfer_protocol::board::redact_secrets`), because the key-level sanitizer
  cannot see inside a paragraph. Conservative by design: a bare long hex run is
  left alone, since board entries legitimately carry commit hashes.
- **Event-sourced session (H3)** — compaction boundaries must be settled records
  (never `Evolve`, never `CompactStart`); `debug_assert_reconstructable` replays the
  raw log after every op, so replay must stay deterministic.

---

## Adding a checklist item

1. Write the **invariant** as one line here, in the table for its stage group.
2. Put the derivation in a `docs/` page — or an existing one — and link it.
3. Add the `- [ ]` line to `AGENTS.md` as the invariant plus the pointer, nothing more.
4. If it adds an op: `docs/PROTOCOL.md` (the `check_protocol_docsync` gate enforces
   this) and, for a `uk_*` symbol, the S29 checklist.
5. If it adds an invariant that is machine-checkable, add it to
   `scripts/verify-invariants` (H1) so it is enforced rather than asserted.