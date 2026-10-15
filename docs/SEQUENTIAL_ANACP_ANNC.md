# SEQUENTIAL_ANACP_ANNC — the decision model that runs AnaCP and AnaNC in sequence

**Status:** R&D, 2026-10-14. **Implementation:**
`unfer/decide_core/src/sequential.rs` (+ `DecideCore::project`),
**tests:** `unfer/decide_core/tests/sequential.rs`.
**Reference implementations studied:** `github.com/SalehMomeni/AnaCP` and
`github.com/SalehMomeni/AnaNC` (the papers' own repositories, in
`~/Projects/AnaCP`, `~/Projects/AnaNC`), plus `DECIDE_SPEC.md` and the
comparison article `ARTICLE_ANACP_ANNC_VS_QFM.md` §2.0.
**Trust boundary:** unchanged — `DECIDE_SPEC` §0. Everything below is
advice-shaped: scores, never certified probabilities, never in a trust path.

---

## 1. What "in sequence" means, and why it was still open

The governing plan's product shape is strictly staged:

> EmbeddingGemma2 → **AnaCP** (closed-form contrastive projection adapts the
> features) → **AnaNC** (collapse readout: ETF targets, max-cosine decision,
> ID/OOD score) **on the projected features** — one after the other; neither
> "one or another" nor an average.

The shipped engine previously realized this *folded into one closed-form map
whose target geometry is selectable per surface* (`Mode::Anacp` **or**
`Mode::Ananc`). Surface selection is not staging: at no point does one method's
output feed the other. This R&D builds the literal sequence and measures it.

Two facts from the reference repositories made the composition the natural
reading all along:

* both repos are **standalone modules** — `anacp.py` / `ananc.py` are advertised
  as drop-in: *"ananc.py … is a standalone module … you can integrate this
  module directly into your own codebase by passing feature embeddings"*;
* AnaCP's `forward` **emits exactly what AnaNC consumes**: an L2-normalized
  adapted feature,
  `F.normalize(gelu(X @ W.T) @ theta)`, and AnaNC normalizes its input again at
  its own seam — idempotent, so the handoff needs no extra convention.

## 2. What the reference code taught us (research findings)

Read against `DECIDE_SPEC`, three genuine deviations surfaced. None is a bug
in either; all are now recorded so the choice is deliberate:

| # | reference implementation | `decide_core` | impact |
|---|---|---|---|
| R1 | `spread_S = S + 1` — **every** singular value shifted up uniformly | signed per-index perturbation `δᵢ ∈ {−1,0,+1}` *searched* for lower summed pairwise cosine (from the paper's Lemma-1 text) | both separate whitened classes; the reference is the simpler, uniform spread. Our search implements the paper's stated objective; the reference shows the paper's *code* takes the constant shift. |
| R2 | ETF frame `S = I − (1/C)·11ᵀ` (pairwise `−1/C`, norm² 1) | `S = I − (1/(C−1))(11ᵀ − I)` (pairwise `−1/(C−1)`, norm² `C/(C−1)`) | **decision-irrelevant**: the two differ by the global factor `C/(C−1)`, which cosine normalization cancels. Provable, and worth stating because an inner-product consumer would see the difference. |
| R3 | AnaCP's `update` fits a **second ridge** on pseudo-replay: mean over heads → `gelu(outs @ W.T)` → ridge to one-hot logits; `predict` reads those logits | readout is cosine against the mapped targets; the two-layer sketch was rejected by the plan's audit (A1/A3) | we do **not** adopt the replay second stage. The staging survives; that arithmetic did not. |

Two more observations worth keeping:

* **AnaCP refits staleness-free by replay.** Every `update` regenerates inputs
  from class Gaussians (`_generate_replay_data`, means + shared covariance) and
  refits under the *current* θ. That is how the reference keeps a two-layer
  sequence consistent without retaining raw data — and it names the upgrade
  path for our design (§4).
* **AnaNC's first batch skips the ETF** (`if num_updates == 0: targets =
  stacked_means`). A warm-start quirk; `decide_core` always applies the
  constraint, which is stricter and matches the papers' constraint statement.

## 3. The design

`SequentialCore { stage1: Mode::Anacp, stage2: Mode::Ananc }`:

* **The handoff** is the new `DecideCore::project`: `f = θᵀ·GELU(xR)`,
  L2-normalized, averaged over heads. This is the feature the readout
  *already* classifies in — class `c`'s prototype is `θ t_c`, so the
  similarity numerator is `(θᵀz)·t_c`; `project` makes that vector explicit so
  it can be a stage's output.
* **The staging contract is type-checked**: stage 1 must be `Anacp`, stage 2
  `Ananc`, and `nc.embedding_dim == cp.projected_dim` — that equality *is*
  the handoff; violations return `DecideError::Staging` (a new variant).
* **Staleness is solved by rebuilding** (§4): stage-2 statistics are always
  expressed under the stage-1 map that will answer.
* **Decoupled widths**: each stage carries its own `projected_dim`. The
  D-ablation's central conflict (AnaCP best at 512, AnaNC best at 16, one
  shared default wrong for one of them) stops being one knob.
* **The composition inherits AnaNC's open-world mechanism**: `id_score`
  returns `Some(max-cosine)` where a pure-AnaCP surface honestly returns
  `None`. Asserted structurally in the tests.

## 4. The staleness problem (defect A4 at sequence level)

Stage 2's class statistics live in *stage-1 feature space*, which moves every
time stage 1 re-solves. Accumulating stage-2 sums under one projection and
answering under another is exactly audit defect **A4** — the thing the engine
already refuses at the embedding seam (`check_embedding_version`).

`SequentialCore::fit` therefore runs the stages in order and rebuilds:
fit stage 1 → re-project **every retained observation** through the resulting
`θ` into a fresh stage-2 engine → fit stage 2. The invariant is asserted:
`stage2().samples() == seq.samples()` after every fit, and `observe`
invalidates the fit so a stale stage 2 can never answer.

**The cost is honest and recorded:** the sequence retains the raw stream,
O(n·d), where each standalone method streams in O(1) memory. The reference
route out of this cost is replay-from-Gaussian (AnaCP's own `_generate_replay_data`);
it is *not* implemented here because a replay generator is a statistical claim
about the data that would need its own validation. This is the plan's next
experiment, not a silent gap.

## 5. Measured results (synthetic corpus — mechanism check, not realism)

Protocol: `report::accuracy_matrix_at`'s continual arrival (5 classes, one per
task, 24/class, interleaved held-out split, corpus dim 24), widths chosen at
each surface's measured-best D-ablation cell. Reported, not pinned (the
`order_sensitivity` discipline):

| configuration | max forgetting | final acc (mean of last row) | last row `[fact, fail, claim, doc, editor]` |
|---|---|---|---|
| `anacp @ D=32` (stage 1 alone) | 0.333 | 0.567 | `[1.00, 0.83, 0.67, 0.17, 0.17]` |
| `ananc @ D=16` (stage 2 alone) | **1.000** | 0.500 | `[1.00, 0.33, 0.00, 0.33, 0.83]` |
| **sequence `32 → 16`** | **0.000** | 0.400 | `[1.00, 1.00, 0.00, 0.00, 0.00]` |
| sequence `32 → 32` | **0.000** | 0.400 | `[1.00, 1.00, 0.00, 0.00, 0.00]` |

**Stage-2 ridge sweep** (stage 1 fixed `λ=100`, `32 → 16`; the AnaNC README's
own advice is to tune `reg` over `[1e-2, 1e6]`):

| `λ₂` | 0.01 | 0.1 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|
| final acc | 0.533 | **0.600** | 0.533 | 0.400 | 0.400 | 0.400 |
| max forgetting | 0.833 | 0.667 | 0.333 | **0.000** | **0.000** | **0.000** |

**Diagnosis of the accuracy cost** (temp probe, deleted; numbers above are the
record). The handoff itself is clean: unit-norm, stage-2-wide, labels aligned,
rebuild invariant holds. What happens is geometric: at `λ₂ ≥ 10` the classes
`claim / doc-track / editor-intent` become near-collinear after the handoff and
all of them score *higher against `fail`'s prototype than against their own*:

```text
true=claim          cos: fail 0.882 > claim 0.546   (argmax fail, wrong)
true=doc-track      cos: fail 0.890 > doc   0.484   (argmax fail, wrong)
true=editor-intent  cos: fail 0.892 > editor 0.508  (argmax fail, wrong)
true=fact           cos: fact 0.867  (correct)      true=fail: fail 0.769 (correct)
```

**Reading the numbers honestly:**

1. **The stability gain is real but mixed.** `max_forgetting = 0.000` has two
   causes: `fact`/`fail` genuinely hold 1.0 across all later arrivals, *and*
   classes the sequence never classifies have no score to lose. Zero
   forgetting from "never scored" is not robustness; the columns say so.
2. **`λ₂` is the tradeoff knob.** Strong stage-2 regularization gives the
   stable-but-weak regime; weak regularization gives the accurate-but-forgetting
   one. On this corpus **no `λ₂` dominates both baselines** (`λ₂=0.1` beats
   every baseline on accuracy at 0.600 but forgets 0.667; `λ₂=1` matches
   `anacp`'s forgetting at worse accuracy).
3. **The confusable classes are the corpus's own geometry**: classes 2–4
   differ only by a constant shift added to *every* dimension, which
   L2 normalization turns into near-identical directions. The single-mode
   surfaces separate them through their readout targets; the second ridge at
   these settings does not. This is a **mechanism finding on synthetic data**,
   not a verdict on the method — the corpus is the same seeded one every
   number in `DECIDE_SPEC` §4 carries caveats about.

## 6. What this R&D establishes

* The sequential model **exists, is contract-checked, and is measured** —
  the plan's shape is no longer only a surface-selection story.
* Staging **decouples the two methods' conflicting widths** and **gives the
  AnaCP surface an ID score** it honestly lacks — both structural, not
  corpus-dependent.
* The staleness problem of composing two continual learners is **solved by
  construction** (rebuild) and its memory cost is **named**, with the
  reference-backed replay upgrade path recorded.
* On the project corpus the sequence buys stability at the cost of accuracy,
  controlled by `λ₂`; nothing here licenses a default — the corpus is
  synthetic, and the honest deliverable is the tradeoff curve plus the
  mechanism diagnosis.

## 7. Follow-ups (in the order they should be done)

1. **Replay-from-Gaussian stage-2 refresh** — replace the O(n·d) raw stream
   with AnaCP's own `_generate_replay_data` pattern (streaming means +
   shared covariance), then re-run §5. Validates the reference's staleness
   remedy *in our stack*.
2. **Protocol wrappers** — route `answer_score` / `answer_choice` through the
   sequence (they exist on `DecideCore`; the sequence currently exposes
   `similarities` / `id_score` / `predict` only).
3. **Readout norm-bias study** — `cos(z, θ t_c)` has a per-class prototype
   norm in its denominator; the hub pattern in §5 is the signature of that
   asymmetry interacting with a second ridge. Measure `‖θ t_c‖` spread per
   stage and test a prototype-normalized readout.
4. **Re-run on real project embeddings** — the standing precondition for
   blessing any default (same as `DECIDE_SPEC` §4).
5. **Run the reference repos on the same corpus** — both accept
   `update(X, Y)` / `predict(X)` on precomputed features, so a
   cross-implementation comparison is data away, not a rewrite.

## 8. Citations

- Momeni, Xiao & Liu. *AnaCP: Toward Upper-Bound Continual Learning via
  Analytic Contrastive Projection.* NeurIPS 2025. Code:
  <https://github.com/SalehMomeni/AnaCP>.
- Momeni, Xiao & Liu. *Continual Out-of-Distribution Detection with Analytic
  Neural Collapse.* AAAI 2026. Code:
  <https://github.com/SalehMomeni/AnaNC>.
- Method context and the audit's defect list: `DECIDE_SPEC.md`;
  the stack comparison: `ARTICLE_ANACP_ANNC_VS_QFM.md` §2.0.
