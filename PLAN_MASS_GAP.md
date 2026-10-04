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

### M6 — Evidence pinning and the regeneration log (S)
**Improves**: the regeneration log section of `MASS_GAP_REGENERATION.md`.
**Work**: keep every regeneration logged with source revision, tool revisions
(Charon/Aeneas/lean4export/nanoda), the 7-error expectation, byte-identity
results, and hashes; the log states explicitly when a fixture is *not*
evidence for changed code. Bundle version markers bump only on structural
schema change (the H3 `SESSION_FORMAT_VERSION` discipline).
**Acceptance**: a reader can verify any pinned fixture against its log entry
without asking anyone. Owner: planner.

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

### M9 — Non-claim enforcement (S)
**Improves**: §5 discipline. **Work**: the truncated-vs-continuum and
gauge-fixed-vs-lattice distinctions are checked as text invariants in the doc
gates (a `rg` guard that fails if "continuum mass gap" appears near
"proved"/"certified" in status prose without the boundary qualifier); new
agent-facing docs (e.g. `../test/`) must reuse the boundary sentences.
**Acceptance**: the guard exists and fails on a planted overclaim. Owner:
planner.

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
