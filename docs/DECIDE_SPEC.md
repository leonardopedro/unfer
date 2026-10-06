# DECIDE_SPEC — the analytic decision engine, code ↔ math

**Status:** implemented 2026-10-05, in `unfer/decide_core`.
**Citations:** the two methods are cited by author–year–venue throughout; no
third-party source is vendored or copied (workspace `ATTRIBUTION.md`).
**Governing document:** `../../PROJECT_REVIEW_AND_IMPROVEMENT_PLAN.md`.

> **Read the trust boundary first.** §0 is not a caveat, it is the contract. This
> component is the first in the project whose output is *deliberately not
> trusted*, and every number below is subject to it.

---

## 0. The trust boundary

This project certifies computation: certified bands, the golden release gate
(S23/S24), `verify-invariants`, nanoda re-verification, the M9 non-claim list. A
statistical decision layer has exactly one legitimate role here — **advice,
labelled as advice**. Four rules, enforced at the type level:

1. **Observe-kind only (S21).** There is no `Mutate` path anywhere in
   `decide_core`. No engine output can approve, merge, gate, sign or certify.
2. **Never in a trust path.** Non-consumers, permanently: `dynamic-arctic`
   signing and consensus; the golden release gate; Lean/nanoda/
   `verify-invariants` checks; certificate reading (S30); the G4 evidence gate;
   anything on the M9 non-claim list. An engine failure can at worst *lose
   advice*.
3. **Vocabulary discipline.** Until a pinned calibration report passes R1–R5
   (Part IV), engine outputs are **scores**. `probabilities` is emitted only when
   `calibrated: true`, and `confidence` is `None` otherwise. Summing to 1 is
   normalization; it is not calibration, and the two are not interchangeable.
4. **Every suggestion is overridable and audited.** Consumers render engine
   output as suggestions with scores; a human or worker always chooses.

### The current measured state

`decide_core/tests/fixtures/decide_calibration_report.json`, produced by
`cargo run -p decide_core --bin decide_report -- --write`:

| candidate | accuracy | ECE | R2 (ECE ≤ 0.05) | R5 (drift ≤ ε) |
|---|---|---|---|---|
| `anacp` / temperature | 0.833 | 0.088 | **fail** | **fail** (0.067 > 0.05) |
| `anacp` / Platt | 0.833 | 0.084 | **fail** | **fail** |
| `anacp` / K+1 | 0.833 | 0.088 | **fail** | **fail** |
| `ananc` / temperature | 0.433 | 0.108 | **fail** | **fail** |
| `ananc` / Platt | 0.433 | 0.068 | **fail** | **fail** |
| `ananc` / K+1 | 0.433 | 0.108 | **fail** | **fail** |

**No candidate meets the contract, so the engine emits scores.** That is the
honest reading of the fixture, and `Calibrator::is_calibrated` enforces it at
runtime rather than trusting this table.

Two further findings the report records, both of which matter for how the
consumer surfaces should be chosen:

- **`ananc` forgets catastrophically here** (`max_forgetting` 1.0 against
  `anacp`'s 0.5). The ETF target's cosine readout is not robust to this
  corpus's class-arrival order. Recorded, not smoothed over.
- **The corpus is synthetic** (seeded PRNG with class anchors), so AUC = 1.0
  says the plumbing works and nothing more. It is a mechanism check, not a
  realism check.

---

## 1. Code ↔ math map

Every field of the engine state, mapped to the symbol used in the source papers.

| state | symbol | where |
|---|---|---|
| `DecideCore::stats` | `C = (μ_1 … μ_C)`, `n_c` | `stats::StreamingStats` |
| `DecideCore::label_ids` | the class vocabulary, explicitly mapped | `lib.rs` |
| `Head::rp` | `R ∈ R^{d×D}`, fixed | `rp::RpLayer` |
| `Head::ridge::gram` | `A = ZᵀZ` | `heads::RidgeHead` |
| `Head::ridge::rp_sums` | `M = (m_1 … m_C)`, per-class RP means | `heads::RidgeHead` |
| `DecideCore::readouts` | `P`, the mapped target means | `heads::NcmHead` |
| `DecideCore::calibrator` | the (ours, not theirs) calibration bridge | `calibrate::Calibrator` |
| `DecideCore::drift` | the R5 drift monitor | `calibrate::DriftMonitor` |

### Layer 1 — the random projection

```
z = GELU(x R),   R ∈ R^{d×D} fixed, drawn once from seed
```

*Why it exists:* both methods expand to `D ≈ 5000` with GELU **before** any
analytic layer, because that expansion is where the nonlinearity lives. A linear
projection of raw features is not the method — this is audit defect **A3**, and
`rp.rs` implements the expansion for exactly this reason.

Entries are drawn `N(0, 1/D)` so `E[‖z‖²] ≈ ‖x‖²`, which keeps `λ` meaning
roughly the same thing across `D` and is what makes `D` tunable.

**GELU is not monotone.** It has an interior minimum near `x ≈ -0.75` and
approaches 0 from below as `x → -∞`. An earlier version of this crate asserted
monotonicity and was wrong; `gelu_has_the_shape_the_papers_actually_use` pins the
real shape instead.

### Layer 2 — the two target constructions

This is where the two methods genuinely disagree, so it is a type-level
distinction (`TargetKind`) rather than a flag.

**`spread_means` (AnaCP §4).** Whitening, SVD, a *signed perturbation of the
singular values*, de-whitening:

```
Ĉ = Σ^{-1/2} C ;   Ĉ = U S Vᵀ ;   S̃ = S + α·diag(δ), δᵢ ∈ {−1,0,+1} ;   P = U S̃ Vᵀ Σ^{1/2}
```

The perturbation is the "contrastive" step: it separates classes that whitening
left close together, with zero gradient updates.

> **Documented deviation.** The paper states the signs are chosen so the summed
> pairwise cosine similarity of the shifted means is guaranteed to decrease for
> small `α > 0`. This implementation does **not** transcribe a closed form for
> `δ`; it *searches* the three admissible values per index and keeps the one that
> measurably lowers `targets::summed_pairwise_cosine`. The search cannot be wrong
> about the direction of improvement because the objective is evaluated; a
> transcribed formula is only as trustworthy as the reading of it. The cost is
> that the result is not provably the paper's exact choice — only that it
> achieves the stated effect. `the_perturbation_lowers_pairwise_cosine` pins it.

Audit defect **A1** ("prototypes = normalized class sums") is ruled out
structurally: `P` is not a running mean, it is the output of a per-task
optimisation.

**`etf_procrustes` (AnaNC eqs. 9–15).** The target means solve a *constraint*:

```
min ‖M̃ − M‖_F   s.t.   M̃ M̃ᵀ = S,   S = I − (1/(C−1))(11ᵀ − I)
```

Because `S` is PSD, write `S = U_S Λ U_Sᵀ` and set `M̃ = U_S Λ^{1/2} Qᵀ`; choosing
`Q` is an orthogonal Procrustes problem on `A = Mᵀ U_S Λ^{1/2} = U Σ Vᵀ`, whose
closed form is `Q = U Vᵀ`, giving `M̃ = U_S Λ^{1/2} V Uᵀ`. Procrustes keeps the
reorganization of feature space minimal, which is what makes the ridge map
learnable.

Two implementation facts that are easy to get wrong and were both wrong here first:

- **Shape.** `A` is `k × C`, so `U` is `k × r` and `V` is `C × r`; `Qᵀ = V Uᵀ` is
  `C × k`. Reading "Q = UVᵀ" at face value gives the transpose and indexes out
  of bounds whenever `k < C`.
- **Precondition `k ≥ C`.** `M̃M̃ᵀ = U_S Λ^{1/2}(V Vᵀ)Λ^{1/2}U_Sᵀ` equals `S`
  exactly when `V Vᵀ = I_C`, i.e. when the SVD yields `r = C` columns, i.e.
  **`k ≥ C`**. The papers satisfy this by a wide margin (`D = 5000`, tens of
  classes). Violating it does not give a slightly worse answer — it gives a
  target that is not an ETF at all, which is audit defect **A2** resurfacing
  through a different door. It is a loud precondition, not a defensive check.

Audit defect **A2** ("ETF = normalized running sums") is ruled out by
`procrustes_output_satisfies_the_etf_constraint`, run over class counts 2..10.

Also worth stating because it is easy to assume otherwise: the ETF gram's
non-trivial eigenvalue is `C/(C−1)`, **not** 1. Mis-scaling `Λ^{1/2}` there
mis-scales every prototype.

### Layer 3 — the analytic solve

```
W = (A + λI)⁻¹ B,   B = Σ_c n_c · m_c · t_cᵀ
```

Cholesky rather than a general inverse: ~2× cheaper, fails loudly on a
non-positive-definite matrix instead of returning garbage, and it is what the
ridge `λ > 0` assumes. `B` is accumulated entry by entry rather than holding all
of `Z`, which is the difference between O(1) and O(n·D²) memory on a stream that
does not end.

**Statistics accumulate only in fixed spaces.** The RP matrix is drawn once and
never re-derived, so an RP-space mean is always a mean of the same linear
function of the same embedding space. Audit defect **A4** — accumulating class
sums under one projection and re-deriving the map under another — is designed out
at the type level.

Whitening uses the **RP-space** covariance (`heads::RidgeHead::rp_covariance`),
because that is the space the solve happens in. Passing the input-space
covariance was a real bug here: two different geometries were being mixed, and
`targets::regularize` now refuses a dimension mismatch rather than producing a
plausible-looking wrong answer.

### Layer 4 — the readout

Cosine similarity to the mapped target means, argmax for a decision and maximum
for the ID score. Cosine rather than Euclidean because the prototypes are defined
up to scale (the ETF constraint fixes inner products, not lengths).

**The ID score is the maximum cosine similarity** — AnaNC's actual OOD
mechanism. The energy-based score from the original sketch (audit defect **A5**)
is not used and must not be reintroduced without its own validation. `anacp`
returns `None` for `id_score` rather than manufacturing one, because a max-cosine
over spread-mean prototypes is not an OOD score.

---

## 2. The calibration bridge (Part III.4)

**No source provides this.** Neither method outputs `P(Y|X)` or `P(ID|X)`;
AnaNC emits a score and evaluates it threshold-free. Everything here is ours,
with its own burden of proof.

| req | property | enforced by |
|---|---|---|
| R1 | `probabilities` sums to 1 ± 1e-9 | `Calibrator::probabilities`; degenerate input yields uniform, never NaN |
| R2 | ECE ≤ 0.05 on held-out data | `Calibrator::is_calibrated`, gated on the measured `fit_ece` |
| R3 | `P(ID)` non-decreasing in the ID score | `Calibrator::validate` refuses a non-monotone parameterization |
| R4 | `P(ID) + P(OOD) = 1`, one estimator | `Calibrator::p_ood` is literally `1 − p_id` |
| R5 | class arrival moves probabilities by ≤ ε, else declare it | `DriftMonitor`, `DecideCore::needs_recalibration` |

**R3 is a validated invariant, not a happy path.** A property test found
`platt_a = -2.33` inverting the ID ordering — the most damaging failure available
to an open-world system, since "I have never seen this class" becomes "I am
certain of a class". The fix was not to clamp the slope but to make a negative
slope *inadmissible* (`Calibrator::validate`).

**R5 cannot be zero.** Adding a class changes the simplex, so zero forgetting for
the calibrator is impossible. The honest deliverable is *declared* drift, with a
measured bound — the same discipline as a certificate band that widens rather
than a number that lies.

**The fitting is a grid search, not backprop.** The plan requires no gradient step
anywhere, so a one-parameter family is searched on a coarse grid over the
held-out split. All three mechanisms are evaluated and all are reported, so a
reader can see whether the choice was close.

---

## 3. Typed protocol (D5)

`unfer_protocol::decide`. The type discipline is the feature: `Choice` carries
its options at call time, `Answer` is an enum so every handler is forced by the
compiler to handle every shape, and `confidence: Option<f64>` makes "not
claimed" unrepresentable as a measurement.

`Option<f64>` rather than `f64::NAN`: NaN does not survive NDJSON — serde emits
`null`, which will not parse back as an `f64`, so an answer carrying an
uncalibrated confidence could not round-trip at all.

`Answer::is_admissible_as_advice()` is the runtime check that an answer claiming
`calibrated: true` actually carries probabilities.

---

## 4. Sizing and defaults

The papers' default is `D = 5000`, which is 25M floats per head — dominated by the
`D × D` Gram. This project's label sets are tens of classes, so `decide_core`
defaults to `D = 512`, `H = 1`.

**These defaults are not validated.** The plan requires AnaCP's `D`-ablation
(Table 2, rows 13–16) to be re-run on project data before any default is
blessed, and that has not happened. They are *sensible*, not *measured*.

`projected_dim` must be `≥` the number of classes for `ananc` — see the
precondition above. `decide_core` does not enforce it at construction; the
precondition fires at the solve, which is where the geometry is actually
needed.

---

## 5. What is deliberately absent

- **No `Mutate` path.** §0 rule 1.
- **No energy-based OOD score.** Audit defect A5.
- **No claim of "zero forgetting".** D-C6's matrix measures it; `ananc` currently
  measures 1.0.
- **No claim of calibrated probabilities.** The pinned report does not license it
  today, and the code refuses to emit it.
- **No GDA-on-sphere posteriors or physically grounded confidence.** Audit
  defect A9: no source establishes either.
- **No Python, no torch.** The backbone is external and frozen, reached through
  the C7 model-spec seam.

## 6. Citations

- Momeni, Xiao & Liu, *AnaCP: Toward Upper-Bound Continual Learning via Analytic
  Contrastive Projection*, NeurIPS 2025 — spread-mean targets, the Lemma-1
  perturbation, pseudo-replay.
- Momeni, Xiao & Liu, *Continual Out-of-Distribution Detection with Analytic
  Neural Collapse*, AAAI 2026 — the ETF/Procrustes construction (eqs. 9–15) and
  the maximum-cosine ID score.
- The typed `Noul`/`Choice`/`Score` question and answer shapes, and the
  ECE-as-objective framing, are adapted from the reviewed `open-jev/` tree
  (Apache-2.0) — see the workspace `ATTRIBUTION.md`.

Equations are cited to the papers' numbering; **all prose here is ours**, and no
source code or text is reproduced.