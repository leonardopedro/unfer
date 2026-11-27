# P6 written-review gate — heterogeneous configurations

Date: 2026-11-27.  Scope of this review: the P6 extension declared in
[`../pcnl/DEVELOPMENT_PLAN.md`](../pcnl/DEVELOPMENT_PLAN.md) §P6.  This file is
the written review that gates P6; the executable half is
[`../scripts/check-pcnl-p6-gate`](../scripts/check-pcnl-p6-gate).

```text
decision = conditional-approve-heterogeneity-only
human_signoff = pending
hypothesis = at matched total parameter count, a heterogeneous-width ensemble
             (12/17/19) keeps ensemble error within +0.005 RMSE of the
             homogeneous 16/16/16 ensemble, keeps effective rank within 0.05,
             retains >= 0.90 in-region members, and costs <= 1.10x latency.
changed_objective = none
compute_cap = seeds:3,members:3,adam_steps:600
promotion = blocked
```

## 1. Prerequisite evidence (P1–P5)

| packet | evidence | status |
|---|---|---|
| P1 known-target quadrature | `logs/pcnl_r14_gates/plateau_quadrature.log` | **executed**, `R̂ < 1.01`, bulk ESS > 400 |
| P2/P3 mechanics | `pcnl/tests/r14.rs` (18 tests) | **executed** |
| P4 supervised | `logs/pcnl_r14_gates/supervised_campaign.log`, `supervised_multiseed.log` | **executed at reduced rows/chains**, not the declared full 20 Newsgroups split |
| P5 RL | `logs/pcnl_r14_gates/rl_campaign.log`, `rl_multiseed.log` | **executed on the synthetic 48-transition corpus**, not RLCD/Edge0 |

The declared P1–P5 completion condition in
[`../pcnl/RESEARCH_PLAN.md`](../pcnl/RESEARCH_PLAN.md) (full splits, all
matched-compute comparators, ID/OOD/continual endpoints, serving evidence)
is **not** met.  Therefore:

* this review approves **one scoped heterogeneity evaluation only**;
* P6 **promotion is blocked**, and the gate script must keep it blocked until
  the full-campaign marker exists *and* a human signs this review;
* no P6 result may be described as a promotion of heterogeneity.

## 2. Approved scope and exclusions

Approved:

* **Heterogeneous widths** inside one ensemble: `12, 17, 19` versus the
  homogeneous `16, 16, 16`, at matched total parameters (172 550 vs 172 134,
  +0.24 %), same data, same three member seeds, same Adam step budget and
  learning rate (no per-width tuning inside this review).

Excluded, with reasons:

* **Git Re-Basin**: no direct weight merging is tested here, which is the
  precondition in the plan.  Permutation alignment is a separate review.
* **Functional repulsion**: excluded because no functional collapse has been
  measured — P4 effective rank is 1.85–1.87 and P5 diversity TV is
  0.31–0.40; changing the target is not justified.
* **Depth/head/normalization heterogeneity**: the current `VectorShapes`
  exposes width, feed-forward, layers and tokens, but the released campaign
  freezes depth/heads/normalization; changing them would be a new target, not
  a P6 configuration comparison.
* **Promotion or serving switch**: blocked (see §1).

## 3. Evaluation contract

* Data: `docs/fixtures/ultrabert_20ng4_embeddings.safetensors`, 240 train /
  120 test rows, `SyntheticTeacher` seed `20261127`, noise `0.05`.
* Arms: homogeneous widths `16,16,16`; heterogeneous widths `12,17,19`.
* Budget: 3 seed families × 3 members × 600 Adam steps, `lr = 0.02`,
  prior scale `0.05`, `τ = 0.75`, `κ = 20`, `α = 0.05`, Huberized hinge.
* Distillation: width-16 student, same 600 steps, trained on each arm's
  teacher mean on the training rows.
* Endpoints: test RMSE, in-region member fraction, effective rank and
  disagreement, teacher–student fidelity, student test RMSE, measured
  ensemble forward latency.
* Predeclared success: every seed's RMSE delta ≤ +0.005; rank delta ≥ −0.05;
  mean in-region ≥ 0.90; mean latency ≤ 1.10×.
* Stop rule: if a width diverges under the shared budget, record it — do not
  retune per width inside this review.

## 4. Measured outcome (2026-11-27)

Command:
`cargo run --offline --release -p pcnl --example r14_heterogeneous`
Log: `logs/pcnl_r14_gates/heterogeneous.log` (wall ≈ 284 s).

| endpoint (mean ± SE over 3 seeds) | homogeneous 16/16/16 | heterogeneous 12/17/19 |
|---|---|---|
| test RMSE | 0.105138 ± 0.001424 | 0.103106 ± 0.007550 |
| in-region members | 1.000 | **0.778 ± 0.111** |
| effective rank | 1.949 ± 0.010 | 1.812 ± 0.079 |
| disagreement | 0.159761 ± 0.005264 | 0.147198 ± 0.010754 |
| teacher–student fidelity | 0.160328 ± 0.003246 | 0.151690 ± 0.006750 |
| student test RMSE | 0.147829 ± 0.007265 | 0.147788 ± 0.007353 |
| ensemble forward latency | 193.04 ± 1.31 µs | 199.83 ± 1.26 µs |

Paired heterogeneous − homogeneous: teacher absolute-error delta
`+0.091774 ± 0.004751`, student delta `+0.005889 ± 0.002932`, RMSE delta
`−0.002032 ± 0.008942`.

**Verdict: hypothesis not supported.**  The mean RMSE is statistically
indistinguishable (`−0.0020 ± 0.0089`), but the heterogeneous arm fails the
in-region and rank criteria: the width-19 member diverges under the shared
budget in seeds 82 (`L = 24.96`) and 93 (`L = 11.01`), and the width-17 member
diverges in seed 93.  No per-width retuning is allowed inside this review, so
the correct conclusion is that width heterogeneity at this shared compute cap
is **not** an improvement and costs 3.5 % latency.  Distillation transfers the
teacher faithfully in both arms, and the distilled students are
indistinguishable (delta `+0.0059 ± 0.0029`, ≈ 2 SE).

## 5. Gate mechanics

`scripts/check-pcnl-p6-gate --check` verifies this review's required fields,
the five prerequisite logs, the R14/P6 status sections, and the
heterogeneous log's markers and compute cap.  `--promote` exits 1 with
`P6-PROMOTION-BLOCKED` while any of these holds:

* the review decision is not `approved-full`;
* `human_signoff` is not `signed`;
* `logs/pcnl_r14_gates/full_campaigns.complete` is absent.

No P6 extension beyond §2 is authorized by this file.
