# One Frozen Backbone, Two Analytic Pipelines: EmbeddingGemma2 + AnaCP→AnaNC versus EmbeddingGemma2 + Quantum Flow Matching

**Status:** technical report, 2026-10-12.
**Code bases:** `unfer/decide_core` (engine version pinned by
`tests/fixtures/decide_calibration_report.json` and
`tests/fixtures/decide_d_ablation.json`) and the `unfer` QFM stack
(`qfm`, `qfm_text`, `nested_fock_algebra`, `prob_kernel`; algorithm
specification `unfer/QFM.tex`, rev 35).
**Premise shared by both stacks.** EmbeddingGemma2 is applied first in
each; every method compared below operates on its 256-d (matryoshka-
truncated, instruction-prefixed) output vectors, never on raw text.
**Naming note.** An earlier draft of this comparison said "AnaCP + AnaNP";
the second head is **AnaNC** (Analytic Neural Collapse,
Momeni–Xiao–Liu, AAAI 2026), the OOD-capable ETF head implemented as
`Mode::Ananc` in `decide_core`. No method named AnaNP exists in either
code base; this article uses AnaNC throughout.

---

## Abstract

We compare two analytically-trained decision stacks that share the same
frozen representation layer — Google's EmbeddingGemma (300M parameters,
matryoshka-truncated to 256 dimensions, instruction-prefixed) — and differ
entirely in the analytic pipeline that consumes its embeddings.

The first stack (`decide_core`) is a **discriminative decision engine**:
a seeded GELU random projection to width $D$ followed by the
*sequential* AnaCP→AnaNC combination the project's governing plan
mandates — AnaCP's closed-form contrastive projection adapts the
backbone's features first, and AnaNC's collapse readout (ETF target
geometry, max-cosine decision and ID/OOD score) runs on the projected
features — realized in the shipped engine as one closed-form map whose
target geometry is selectable per surface — and a ridge solve whose
weights are accumulated in constant memory on a stream. It performs no
gradient step anywhere. Its output is deliberately *advice*: scores only,
never certified probabilities, never in a trust path.

The second stack (**Quantum Flow Matching**, QFM) applies the same
frozen EmbeddingGemma2 vectors *first*, then runs a *generative*
wavefunction-flow engine on them: each embedding vector is encoded as a
mutually orthogonal Fock mode, the flow-matching fit reduces to a closed-form $\mathcal{O}(M)$
average, the resulting continuity Hamiltonian is compressed by
shift-invert rational Krylov (SIRK), and online inference is a single
$\mathcal{O}(m^2)$ unitary evolution decoded by the Born rule — with a
quantum Bayesian update (HMC on $\mathbb{C}P^{m-1}$, Karcher-mean
posterior) layered on top.

We compare them on every benchmark either side has actually measured:
the decision engine's pinned calibration report, its $D$-ablation, and
its forgetting matrix; QFM's criterion benchmarks, five cross-domain
real-data validation suites (MNIST, CIFAR-10, critical 2D Ising,
Kolmogorov CFD, post-Newtonian gravitational waveforms), a lattice
Yang–Mills mass-gap demo, and the QFM-Text perplexity study on
WikiText-103. We add new single-host latency measurements for the
decision engine (fit and predict at $D \in \{16, 256, 512\}$).

The headline result is structural rather than a winner: **both stacks
are "expensive offline, microseconds online" designs with a
$10^5$–$10^6$ fit-to-inference cost ratio, and both fail in the same
way — by geometric collapse.** The decision engine's AnaNC head forgets
catastrophically (max forgetting $1.0$ at every $D$); and when the very
same anti-forgetting protocol is run over a QFM-TSR class-arrival
stream, the flow forgets just as badly (measured max forgetting $1.0$)
while the raw similarity geometry under it forgets nothing ($0.0$) —
placing both collapses inside the analytic machinery, not in the
data. Where they differ is in what they refuse to claim: the decision engine
withholds `calibrated: true` because its measured ECE fails its own
contract, while QFM's Born probabilities are exact by construction yet
still lose to a classical $n$-gram on held-out perplexity. Exactness and
calibration are different properties, and both stacks document that gap
honestly.

---

## 1. Introduction

Frozen-backbone systems are converging on a common pattern: an
off-the-shelf sentence encoder produces representations, and a small,
cheap, analytically-solvable head turns them into decisions or
distributions. The appeal is operational — no training loop, no GPU
requirement, constant-memory streaming, and (in both projects compared
here) an auditable chain from input to output.

This report compares two instantiations of that pattern that live in the
same federation of repositories but share nothing below the embedding
seam. **In both stacks EmbeddingGemma2 is applied first**: the text (or
image caption, or telemetry) is embedded once by the frozen backbone,
and the QFM pipeline in Stack B is applied *only to the resulting
EmbeddingGemma2 vectors* — the flow's data points are the 256-d MRL-
truncated embedding stream, not raw pixels or raw tokens. The raw-data
fixtures in QFM's validation suite (MNIST, CIFAR-10, Ising, CFD, GW)
are width-matched surrogates for exactly that stream ($d = 64\ldots1024$
standing in for embedding vectors of the same widths); they validate the
pipeline the embedding-fed version would run, and are cited here as
such.

| | **Stack A: AnaCP/AnaNC** | **Stack B: QFM** |
|---|---|---|
| Backbone | EmbeddingGemma2 (256-d, MRL), applied first | EmbeddingGemma2 (256-d, MRL), applied first — QFM consumes the embedding vectors as its point cloud |
| Head | GELU random projection + ridge solve | Fock encoding + SIRK Krylov + Born rule |
| Task | classification, ranking, OOD score | generative transport, density, Bayesian posterior |
| Training | closed-form (no gradients) | closed-form (no gradients) |
| Output status | **scores** (calibration unlicensed) | **Born probabilities** (exact, but not calibrated) |
| Trust posture | observe-kind only; never in a certification path | kernel runtime with UK-4001 authorization gates |

The comparison below uses only numbers each side has actually measured,
with provenance attached to every table. Two caveats hold throughout:
Stack A's accuracy numbers come from a *synthetic* seeded corpus (a
mechanism check, not a realism claim), and there is **no shared
benchmark dataset** — the two stacks have never been run on the same
data. The comparison is therefore structural, cost-based, and
failure-mode-based; where a direct number comparison would be
meaningless we say so.

## 2. Stack A — EmbeddingGemma2 + AnaCP + AnaNC

Specification: `unfer/docs/DECIDE_SPEC.md`; implementation:
`unfer/decide_core`. Methods: Momeni, Xiao & Liu, *AnaCP* (NeurIPS 2025)
and *Continual OOD Detection with Analytic Neural Collapse* (AAAI 2026).

### 2.0 The planned combination — one stage after the other

The decision model studied here is the **sequential AnaCP→AnaNC
combination** as specified by the governing implementation plan at the
time the implementation executed (`~/Projects/PLAN_ANALYTIC_DECISION_ENGINE.md`,
re-tracing the original sketch in `~/Projects/PromptANaCP`). The
sketch's order — adopted as the product shape by the plan, with each
stage's mathematics corrected by its audit (defects A1–A9) — is
strictly staged:

1. **Stage 1 — AnaCP (feature adaptation, applied first).**
   Accumulate the statistics a closed-form contrastive projection
   needs, solve for the projection analytically, and project the frozen
   backbone's features through it onto the unit sphere. No gradients:
   the embedding is adapted *after* the backbone, by a closed-form
   solve.
2. **Stage 2 — AnaNC (collapse readout, applied to the projected
   features).** On the *projected* features, accumulate class
   prototypes under the equiangular-trace-frame constraint and decide
   by cosine to them; the maximum cosine is the ID/OOD score.

So: EmbeddingGemma2 → AnaCP projection → AnaNC readout — one after the
other; neither "one or another" nor an average. The plan's audit
corrected each stage's math without disturbing the order: the real
AnaCP stage solves ridge-to-spread-means in the RP space (A1), the
real AnaNC stage solves the ETF as a Procrustes constraint (A2), and
the ID score is max-cosine rather than an energy (A5).

**How the shipped engine realizes the sequence.** `decide_core` folds
the two stages into one closed-form map: the ridge solve
$W = (A+\lambda I)^{-1}B$ *is* the adaptation stage (features are
pulled toward the target means by the solve itself), and the cosine
readout against the mapped targets *is* the collapse stage. The target
construction that parameterizes the sequence is selectable per
surface — `Mode::Anacp` (spread means) where closed-set accuracy is
the product, `Mode::Ananc` (ETF-Procrustes plus the max-cosine ID
score) where open-world geometry is. All "AnaCP" and "AnaNC" numbers
in §4.1 are therefore two surface parameterizations of the one
combined, staged model — measured through its two surface selections —
not two rival systems. The strictly two-layer sketch (a separate
`W_proj` followed by normalized-sum prototypes) was the form the
plan's audit rejected; the staging survived, that arithmetic did not.

### 2.1 Pipeline

EmbeddingGemma vectors arrive at a seam (`decide_core::backbone`) that
enforces three invariants before any vector is read: the embedding
model's identity is part of the engine's identity (a version change is
refused, not silently mixed — audit defect A4), the matryoshka truncation
width must equal the engine's accumulation width, and L2 normalization
happens at the seam, not in the caller.

Inside the engine:

1. **Random projection.** $z = \mathrm{GELU}(xR)$, $R \in \mathbb{R}^{d
   \times D}$ drawn once from a seed with entries $\mathcal{N}(0, 1/D)$.
   The nonlinearity lives in this expansion (audit defect A3); $D = 512$
   is the shipped default, chosen by the measured $D$-ablation (§4.1).
2. **Target construction** — the genuine disagreement between the two
   methods, made a type-level distinction:

   * **AnaCP `spread_means`** (contrastive): whiten by the RP-space
     covariance, take the SVD, *perturb the singular values* by signed
     amounts $\delta_i \in \{-1,0,+1\}$ chosen by search to lower the
     summed pairwise cosine, de-whiten. Zero gradient updates; the
     perturbation separates classes whitening left close together.
   * **AnaNC `etf_procrustes`**: project class means onto the closest
     equiangular frame, $\min \|\tilde M - M\|_F$ s.t. $\tilde M \tilde
     M^\top = S$, solved in closed form by orthogonal Procrustes
     $\tilde M = U_S \Lambda^{1/2} V U^\top$ (requires $k \ge C$).

3. **Ridge solve.** $W = (A + \lambda I)^{-1} B$ by Cholesky, with $B =
   \sum_c n_c\, m_c t_c^\top$ accumulated entry-by-entry so a stream
   costs $\mathcal{O}(1)$ memory. No update of $W$ occurs between fits;
   a refit is a fresh closed-form solve over the sufficient statistics.
4. **Readout.** Cosine similarity to the mapped targets, argmax for a
   decision. AnaNC additionally exposes the maximum cosine as an
   OOD/ID score; AnaCP deliberately returns no ID score rather than
   manufacturing one.
5. **Calibration bridge (project-original).** Platt / temperature /
   $K{+}1$ grid search on a held-out split, gated by requirements
   R1–R5. The engine emits `calibrated: true` only if the measured ECE
   is $\le 0.05$ *and* the drift bound holds; today neither holds, so
   the engine emits **scores** and refuses to emit probabilities
   (`Calibrator::is_calibrated` enforces this at runtime).

### 2.2 Trust posture

Observe-kind only: there is no `Mutate` path anywhere in `decide_core`;
no engine output can approve, merge, gate, sign or certify anything.
This posture is the reason Stack A can be embedded in a
certification-grade system at all — an engine failure can at worst lose
advice.

## 3. Stack B — EmbeddingGemma2 + Quantum Flow Matching

Specification: `unfer/QFM.tex` (rev 35); implementation: the `qfm`,
`qfm_text`, `prob_kernel`, `nested_fock_algebra`, and `fock_sirk`
crates. Origin: Layden, Sweke, Havlíček, Chowdhury & Neklyudov's
observation that a flow model's qsample obeys the Schrödinger equation
with continuity Hamiltonian $\hat H_t = \tfrac12[\hat p \cdot v_t(\hat
x) + v_t(\hat x)\cdot \hat p]$; Krylov compression after Hashimoto &
Nodera (2019).

### 3.1 Pipeline (TSR path; in this comparison, fed by EmbeddingGemma2 vectors)

1. **Orthogonal Fock encoding.** Each of $M$ training points — here, an
   EmbeddingGemma2 output vector — becomes a mutually orthogonal
   single-boson mode; the conservative potential becomes diagonal, and
   flow-matching "training" reduces to a closed-form $\mathcal{O}(M)$
   average with zero data loss. QFM is applied *only after* the backbone:
   it never sees a string. The 256-d truncation width is exactly the
   CIFAR-10 $16\times16$ fixture width, so the raw-data validation
   suites exercise the identical pipeline an embedding-fed deployment
   runs, at the identical data width — that correspondence is why those
   fixtures are admissible evidence in an embedding-backbone
   comparison.
2. **The generator.** Since rev 31 the *only* off-diagonal generator is
   the exact rank-1 Mehler projector $\bar H = |\tilde 0\rangle\langle
   \tilde 0|$ — Hermitian, idempotent, applied through the rank-1
   shortcut $\bar H|s\rangle = \langle 0|s\rangle|\tilde 0\rangle$,
   never an expanded $\mathcal{O}(\varepsilon)$-truncated coupling list.
3. **SIRK compression.** Shift-invert rational Krylov with normalized
   imaginary shifts $z_k = -ik/m$ (the rev 24 overflow fix) compresses
   the generator to $H_m \in \mathbb{C}^{m\times m}$; error decays
   exponentially in $m$.
4. **TSR inference — four phases**, with no $\mathcal{O}(M)$ dependence
   at inference: encode (two-level Count-Sketch $S_1:\mathbb{R}^d\to
   \mathbb{R}^k$, $S_2: \text{features}\to K_2$-dim single-excitation
   states), evolve ($c_1 = e^{-iH_m t} c_0$, a Padé matrix exponential
   that is provably unitary), tomographic reconstruct (exactly $m^2$
   complex multiplications), and decode (Born rule
   $p[m] = |\langle m|Wc_1\rangle|^2$).
5. **Quantum Bayesian update.** Observations update the prior by HMC on
   the projective space $\mathbb{C}P^{m-1}$ (Born-rule likelihood,
   phase-gauge-invariant), with a Karcher-mean posterior point estimate.
   This is QFM's *continual* mechanism: the posterior becomes the next
   prior (iterated end-to-end in `iterated_bayes_module`).

### 3.2 QFM-Text variant

The one QFM variant that does *not* sit downstream of the backbone is
the token-stream instantiation, which bypasses EmbeddingGemma2 and
applies the same machinery to per-order hashed token modes (we include
it because it is the only QFM deployment at corpus scale, and because
its negative result is the comparison's most externally-valid
evidence): $H = \sum_o \lambda_o |\tilde 0_o\rangle\langle \tilde
0_o|$ (one exact projector per backoff order), Born-rule decode
marginalized against per-mode histograms with Katz-style escape to the
corpus unigram. It is a quantum-kernel $n$-gram-family model; its own
stated success criterion is beating a classical $n$-gram at equal
context order on the same corpus — which, as §4.2 shows, it does not.

## 4. Measurements

### 4.1 Stack A results (pinned fixtures + this-session latency)

**Pinned calibration report** (`decide_core/tests/fixtures/decide_calibration_report.json`;
synthetic corpus, 24 samples/class, width 24, every 4th sample held
out → 30 held-out; $D = 256$ protocol):

| candidate | accuracy | ECE | AUC | FPR95 | R2 (ECE ≤ .05) | R5 (drift ≤ .05) |
|---|---:|---:|---:|---:|---|---|
| AnaCP / temperature | 0.833 | 0.088 | 1.0 | 0.0 | **fail** | **fail** (0.067) |
| AnaCP / Platt | 0.833 | 0.084 | 1.0 | 0.0 | **fail** | **fail** |
| AnaCP / K+1 | 0.833 | 0.088 | 1.0 | 0.0 | **fail** | **fail** |
| AnaNC / temperature | 0.433 | 0.108 | 1.0 | 0.0 | **fail** | **fail** |
| AnaNC / Platt | 0.433 | 0.068 | 1.0 | 0.0 | **fail** | **fail** |
| AnaNC / K+1 | 0.433 | 0.108 | 1.0 | 0.0 | **fail** | **fail** |

AUC = 1.0 and FPR95 = 0.0 are properties of the synthetic separability —
they mean "the plumbing works", nothing about open-world OOD quality.
No candidate licenses `calibrated: true`; the engine emits scores.

**$D$-ablation** (`decide_d_ablation.json`, same corpus, gated
byte-for-byte in CI):

| $D$ | AnaCP acc | AnaNC acc | AnaNC forget | whitened floors (AnaCP) | state |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.467 | *refused* (ETF precondition $k\ge C$) | — | 1 | 6.6 KiB |
| 16 | 0.500 | **0.500** | 1.00 | 1 | 11.7 KiB |
| 64 | 0.733 | 0.433 | 1.00 | 1 | 54.4 KiB |
| 256 | 0.833 | 0.433 | 1.00 | 167 | 585 KiB |
| 512 | **0.900** | 0.333 | 1.00 | 423 | 2.2 MiB |

Two findings carry forward: AnaCP improves monotonically in $D$ while
AnaNC is *measured-worst* at the shipped default (its best cell is
$D=16$, 190× less state); and AnaNC's catastrophic forgetting
(max-forgetting 1.0) is **not** a $D$ artifact — it appears at every
$D$, including its best. The mechanism is measured: $n_\text{train}=90$
caps the RP-covariance rank at 89, so AnaCP's whitening floors
$D-89$ directions (39/167/423 at $D=128/256/512$), while AnaNC — which
does no whitening — floors zero; the ETF/cosine readout then degrades as
the projection stretches.

**New latency measurements** (this session; AMD Ryzen AI 5 340, single
core, Rust release build, 20 000 timed queries after warm-up, 5 classes,
90 observations, harness outside the repo at `/tmp/decide_bench`):

| mode | $D$ | observe 90 obs | `fit()` | `answer_score()` per query |
|---|---:|---:|---:|---:|
| AnaCP | 16 | 159 µs | 107 µs | 0.45 µs |
| AnaCP | 256 | 45.7 ms | 306 ms | 5.0 µs |
| AnaCP | 512 | 212 ms | **2.91 s** | **7.6 µs** |
| AnaNC | 16 | 78 µs | 32 µs | 0.63 µs |
| AnaNC | 256 | 30.5 ms | 5.8 ms | 7.7 µs |
| AnaNC | 512 | 199 ms | **43 ms** | **15.6 µs** |

**In-repo reproduction (added after the first measurement).** The
table above is now reproducible with a committed criterion bench —
`cargo bench -p decide_core --bench latency`
(`decide_core/benches/latency.rs`), which re-measures the same three
costs on the same corpus protocol ($n_\mathrm{train}=90$) in about two
minutes. Its quick-mode figures on the same host reproduce the table
to within a factor of ~1.3 — fit AnaCP/AnaNC at $D=512$: **2.64 s /
41.3 ms**; predict at $D=512$: **20.9 µs / 16.7 µs**; predict at
$D=16$: 0.43 µs / 0.75 µs — including the 64× fit asymmetry between
the modes. The prose conclusions below therefore rest on an in-tree
measurement, not a one-off harness.

The 67× fit-time gap between the two modes at $D=512$ is structural and
is the most interesting number in this table. AnaCP's whitening performs
an eigendecomposition *and* an SVD in the $D$-dimensional RP space
($\mathcal{O}(D^3)$ with large constants at $D=512$); AnaNC's
Procrustes decomposition lives entirely in the $C$-dimensional class
space ($5\times5$ here) plus one small SVD, so its fit cost is
dominated by the same sufficient-statistics sweep both modes share.
Prediction cost is µs-scale for both, and the fit:predict ratio at
$D=512$ is $\sim 4\times10^5$ (AnaCP) and $\sim 3\times10^3$ (AnaNC).

### 4.2 Stack B results (QFM.tex rev 15–35, docs/QFM_TEXT_STATUS.md)

**Criterion benchmarks** (`qfm/benches/pipeline.rs`, CPU, rev 15 local
run): offline compile at $M=1000$ ≈ 3.6 s; `generate` at $d=1024$ ≈
**20 µs**; Count-Sketch apply at $d=4096$ ≈ 9 µs; Bayesian update
confirmed $\mathcal{O}(m^2)$ per HMC step, $\mathcal{O}(K_2)$
likelihood construction. The rev-26 rank truncation removes the
$\mathcal{O}(K_2^3)$ Gram wall — three orders of magnitude at $K_2 =
1024$ — letting a $d=1024$ pipeline compile in ≈2 s
($\texttt{krylov\_dim}=32$, $\texttt{max\_rank}=16$).

**Cross-domain real-data validation** (all suites: compile, generate
correlated, Bayesian update recovers held-out observations, no
explosion):

| fixture | $d$ | $M$ | generate cosine | Bayesian overlap $\lvert\langle s\vert c_\text{obs}\rangle\rvert^2$ | runtime |
|---|---:|---:|---:|---|---|
| MNIST 8×8 (real digits) | 64 | 65 | ~0.6 | {0.24, 0.38, 0.51, 0.64} on 4/5 | (runtime not reported) |
| CIFAR-10 16×16 (real) | 256 | 256 | > 0.2 (min > 0) | pass (8 held-out) | (runtime not reported) |
| Ising 2D critical $L=16$ | 256 | 64 | 0.35 (all 8 identical — rank collapse) | pass | <1 s |
| Ising 2D critical $L=32$ | 1024 | 64 | −0.09 (mild anti-correlation) | 0.78, 0.70 on 2/8 | ~20 s (12 tests) |
| Kolmogorov CFD $L=16$ | 256 | 64 | 0.36 ± 0.00 | [0.18, 0.77] on 8/8 | ~5 s |
| Kolmogorov CFD $L=32$ | 1024 | 64 | 0.12 ± 0.00 | 0.81, 0.11 on 2/8 (rest ~0.015) | (above) |
| PN GW (TaylorT1) $d=256$ | 256 | 64 | 0.72 ± 0.00 | [0.34, 0.61] on 4/8 | ~1 s |
| PN GW $d=1024$ | 1024 | 64 | 0.42 / 0.10 (bimodal) | [0.05, 0.77] on 4/8 | (above) |

Plus the lattice physics demo: $l=3$ SU(2) Yang–Mills mass gap
$\approx 1.988$ against the strong-coupling estimate $g^2/2 = 2.0$ —
**within 1%**, positive gap confirmed (confinement) in ~8 s CPU.

**QFM-Text on WikiText-103** (full 128,102,894-token train split;
$K_2 = 4\times65536$ modes, 4 orders; rev 35):

| model | train wall | in-sample ppl | held-out ppl (100k) | gap vs baseline |
|---|---:|---:|---:|---|
| QFM-Text $m=8$ | 7 m 32 s | 253.8 | 420.8 | 1.39× / 1.15× |
| QFM-Text $m=32$ | 7 m 47 s | 237.2 | 382.3 | 1.30× / 1.05× |
| classical $n$-gram (same hashing) | — | 182.7 | 364.6 | (reference) |
| unigram | — | n/a | 1033.0 | — |

The QFM is **5–39% worse than the classical $n$-gram at every
configuration tested**, including a 30-configuration hyperparameter
sweep ($t \times \lambda \times \text{discount}$) in which no
configuration beats the baseline. The diagnosed cause is a rank
degeneracy: the Gram-whitened encode/decode basis $W$ has SVD rank
**1–2, not the requested $m$ shifts**, because the dressed vacua
$|\tilde 0_o\rangle$ with $c_0 \approx 1$, $\varepsilon \sim 10^{-5}$ are
nearly parallel to the Fock vacuum. Per-context diagnostics: mean
KL(emp‖QFM) = 3.84 nats, mean cosine 0.38, top-1 hit 33%, top-5 hit 8%
— *identical* for $m=8$ and $m=32$, and the model misplaces 4–10%
probability on wrong tokens even on fully deterministic contexts.
**D-C6 forgetting protocol over a QFM-TSR class-arrival stream
(new measurement, 2026-10-12; `qfm/tests/forgetting_matrix.rs`).**
The comparison's open question — does QFM forget under class arrival? —
is now measured by running the *identical* D-C6 task-incremental
protocol on a QFM-TSR arrival sequence: a seeded 8-class structured
fixture ($d=64$, 8 train + 4 held-out per class), classes arriving two
per task, the pipeline refitted on the cumulative bank after each task
(the closed-form refit, the flow analog of the engine's re-solve),
attribution = class of the nearest training point (cosine) of the
`generate()` output:

| protocol arm | t=0 | t=1 | t=2 | t=3 (final) | max forgetting |
|---|---|---|---|---|---:|
| QFM-TSR refit | [0.00 1.00] | [0.00 0.00 1.00 0.00] | [0.00 0.00 1.00 0.00 0.00 0.00] | [0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00] | **1.0000** |
| raw-NN baseline (no flow) | [1.00 1.00] | all 1.00 | all 1.00 | all 1.00 | **0.0000** |

The flow arm forgets catastrophically — the same number `decide_core`
measures for its AnaNC selection (1.0) — and its final-row accuracy is
$1/8$: at every task the attribution collapses onto a single class
region, the rank-truncation representative-output collapse already
documented per fixture in QFM.tex rev 29–30. The baseline arm is the
load-bearing control: identical arrival sequence, identical bank
growth, attribution taken on the raw query — and it forgets
*nothing* at perfect accuracy. The forgetting is therefore inside the
flow, not in the class-arrival interference the bank growth would
carry. The matrix is printed by the test; the flow arm's scalar is
deliberately not pinned by an assertion (the rev 37 diffusion
Hamiltonian targets exactly this collapse, and a pipeline improvement
must not read as a test failure) — the baseline's $0.0$ *is* asserted,
as the protocol's own sanity check.

### 4.3 Side-by-side on comparable axes

No shared dataset exists, so the honest comparison table is on
*operational* axes:

| axis | AnaCP/AnaNC ($D=512$) | QFM-TSR ($K_2=256$, $m\le32$) |
|---|---|---|
| offline fit | 2.91 s (AnaCP) / 43 ms (AnaNC), $n=90$ | 3.6 s ($M=1000$); ~2 s at $d=1024$ truncated |
| online decision | 7.6 µs / 15.6 µs per query | 20 µs per generate ($d=1024$) |
| fit:predict ratio | $\sim4\times10^5$ (AnaCP) | $\sim2\times10^5$ |
| state size | 2.2 MiB (full $D\times D$ Gram) | $\mathcal{O}(K_2\!\cdot\!m)$ + per-mode histograms |
| inference memory | one $D$-vector | one $K_2$-vector |
| continual mechanism | refit from sufficient statistics | Bayesian posterior → prior (HMC) |
| forgetting measured | yes: AnaCP 0.5, AnaNC **1.0** (D-C6) | **yes, D-C6 over an arrival stream: refit 1.0, raw-geometry control 0.0**; Bayesian-continuation arm untested |
| probability claim | **refused** (ECE/drift contract fails) | Born-exact by construction, but held-out ppl loses to $n$-gram |
| worst measured failure | AnaNC forgetting 1.0; AnaCP ECE 0.084 > 0.05 | $W$-rank collapse; ppl gap 1.05–1.39×; deterministic contexts get 4–10% wrong mass |

At the embedding widths the two stacks actually share ($d = 256$, the
EmbeddingGemma truncation width and the CIFAR/Ising fixture width), the
online latencies are the same order of magnitude — single-digit
microseconds for the decision head, tens of microseconds for the
generative head. Neither stack is latency-bound; both are
*state*-bound in different currencies (dense $D\times D$ Gram vs
$K_2$-wide sparse mode banks).

## 5. Making sense of the results

**1. Same shape, different currency: the $10^5$ fit:predict asymmetry
is a design property, and it is what makes both stacks embeddable.**
AnaCP fits for 2.91 s and then answers in 7.6 µs; QFM compiles for
3.6 s and then generates in 20 µs. Both achieve this the same way —
replace iterative optimization with closed-form linear algebra over
sufficient statistics — and both pay for it in the same place: a
Gram/covariance eigendecomposition. The remedies rhyme too: AnaCP's
cost sits in the $D\times D$ whitening it cannot avoid without changing
the method, while AnaNC *has* avoided it structurally (its Procrustes
problem lives in the $C\times C$ class space, which is why its fit is
67× cheaper); QFM removed its $\mathcal{O}(K_2^3)$ wall by opt-in rank
truncation. The general lesson: for analytic heads, the entire systems
question is "which small matrix gets decomposed, and how often."

**2. Both stacks fail the same way — by geometric collapse — and the
parallel is exact.** AnaNC's ETF readout assumes class means can be
pushed onto an equiangular frame; under continual class arrival the
frame is refit, the readout geometry shifts, and the measured forgetting
is 1.0 (total) at *every* projection width — a finding whose value is
that it closes "just increase $D$" as a remedy. QFM-Text's Krylov
encode assumes the per-order dressed vacua supply $m$ linearly
independent directions; with $10^5$ modes their per-mode weights are
$\sim10^{-5}$, all vacua collapse toward the Fock vacuum, and the basis
rank is 1–2 regardless of requested $m$ — measured by adding a
`--diagnose` SVD, and confirmed by $m=32$ being *worse* than $m=8$
(rank 1 < rank 2). In both cases an analytic construction that is exact
in the well-conditioned regime meets a degenerate geometry and degrades
*catastrophically rather than gracefully* — because there is no loss
surface to slide down, there is only the geometry you assumed. AnaCP's
spread-means head survives the same corpus where AnaNC does not (0.5 vs
1.0 forgetting) precisely because its contrastive perturbation
explicitly searches for and separates near-degenerate directions — the
same lesson QFM's rev 37 is now applying on the other side, replacing
the vacuum-dominated generator with one that has off-diagonal
transition structure.

**3. "Normalized" is not "calibrated", and "exact" is not "good".**
The two stacks land on opposite sides of the same trap. AnaCP/AnaNC
*refuse* to emit probabilities because their measured ECE (0.068–0.108)
fails their own $\le 0.05$ contract and the drift bound fails at every
$D$; the deliverable is honest scores plus a declared, measured drift.
QFM emits Born probabilities that are *exactly* the normalized
 Born populations of its evolved state — and still loses to a classical
$n$-gram on held-out perplexity (382.3 vs 364.6 at best, 420.8 at
$m=8$) and misplaces 4–10% mass on deterministic contexts. The Born
rule guarantees $\sum_m p[m] = 1$ and non-negativity; it guarantees
nothing about whether those probabilities track empirical frequencies.
A reader choosing between the stacks on the criterion "which gives me
trustworthy probabilities" gets the same answer from both: neither, yet.
The difference is only in how loudly each says it.

**4. Forgetting is now measured on both sides — and the numbers are the
same.** The comparison's original gap (Stack A measures forgetting;
Stack B only engineers a continual mechanism) is closed: the D-C6
protocol run over a QFM-TSR arrival stream measures **1.0** for the
refit arm — identical to the engine's AnaNC selection — against **0.0**
for the raw-geometry control on the same stream. The parallel with
AnaNC is exact and diagnostic: in both stacks the forgetting is not in
the data (the control forgets nothing; the engine's own AnaCP
selection forgets only 0.5) but in the analytic machinery's geometry —
AnaNC's ETF readout degenerating under refit, the flow's rank-truncated
basis steering every query to one representative class region. The
hypothesis the article previously left open — "QFM forgets less,
because a posterior is a sufficient statistic for the next arrival" —
is now measured false *for the refit path* and remains open only for
the Bayesian-continuation arm (posterior → prior), which this protocol
does not exercise and which is the obvious next measurement. A second
shared lesson falls out: both stacks' forgetting numbers are made
interpretable by a control arm that isolates machinery from data —
without the 0.0 baseline, a 1.0 would be an accusation against class
arrival itself.

**5. The corpora explain more than the methods do — read the accuracy
gap accordingly.** AnaCP's 0.833–0.900 and AUC 1.0 come from a *synthetic
seeded corpus with class anchors*; QFM's 0.72 cosine on gravitational
waveforms and 1% mass-gap agreement come from real physics families
with genuine low-rank structure (amplitude × phase). The QFM-Text
negative result — losing to $n$-gram on 128M real tokens — is the most
externally-valid number in this entire comparison, and it is a
*failure*; the decision engine's headline numbers are the least
externally-valid and they are *successes*. Any summary that inverts
this (decision engine 90% vs QFM "worse than n-gram") is comparing a
mechanism check against a battlefield report. Both projects refuse to
let their numbers be read that way — the fixtures' caveat blocks say so
in machine-readable form — and that shared refusal is arguably the most
important thing the two stacks have in common.

**6. Where one stack should actually be preferred, today.** For a
surface that ranks, classifies, or gates on frozen embeddings with tens
of classes and constant memory, Stack A is the only one of the two that
does the task at all, and its µs-scale predict cost makes it
embeddable in hot paths; its known limits (AnaNC forgetting; unlicensed
calibration) are pinned in fixtures that CI gates. For a surface that
must *generate* samples, condition on observations, or carry a
posterior across arrivals — anomaly reconstruction, data-assimilation
style queries, physics-flavored distributions — Stack B is the only one
with the machinery, its physics validations (Yang–Mills within 1%,
Bayesian recovery on CFD/GW/Ising) are its strongest evidence, and its
known limits (W-rank degeneracy on token-scale vocabularies, top-1
heavy hitters, rank-2 representative-output collapse at $M/d=1/16$) are
documented per fixture. On latency-bound embedding classification they
are interchangeable; on everything else they do not compete.

## 6. Threats to validity

1. **No shared benchmark.** Neither stack has been run on the other's
   data. Every cross-stack statement above is structural or
   operational, never an accuracy head-to-head.
2. **Synthetic corpus.** Stack A's accuracy/ECE/AUC numbers are a
   mechanism check on a seeded PRNG corpus (its own fixture says so).
   Only the *relative* comparisons (across $D$, across modes) carry
   weight.
3. **Single-host timing.** The latency table is one AMD Ryzen AI 5 340,
   one run, in-tree-free harness; treat it as order-of-magnitude
   evidence. QFM's 20 µs / 3.6 s numbers are the rev-15 local run
   reported in QFM.tex, on the author's machine of that date — likewise
   order-of-magnitude.
4. **Small held-out counts.** 30 held-out samples for the calibration
   report; ECE differences below ~0.05 are noise on that base, which is
   recorded in the fixture's caveats.
5. **Rev-skew in QFM-Text numbers.** The perplexity table is measured
   against the rev-35 generator; rev 34 restored the exact-projector
   compile path, and rev 37 is introducing a new diffusion Hamiltonian
   targeting the W-rank root cause. The negative result is
   revision-stamped and expected to move.
6. **Scope of the new forgetting measurement.** The D-C6-over-QFM
   number (refit 1.0, control 0.0) is one seeded synthetic 8-class
   $d=64$ stream, refit-per-task arm only; it establishes that the
   flow *can* forget catastrophically while its geometry stays
   separable, on this stream. The Bayesian-continuation arm and real
   embedding streams are unmeasured.

## 7. Conclusion

EmbeddingGemma2 + AnaCP/AnaNC and EmbeddingGemma2 + QFM are two answers
to the same question — how much machinery sits on top of a frozen
embedding — and their measurements say the answers are complementary
rather than competing. The decision engine is a µs-scale, constant-memory,
trust-refusing classifier whose worst behavior (AnaNC forgetting = 1.0)
is pinned and whose calibration claim is withheld. QFM is a
$10^5$-asymmetry generative engine with exact Born semantics whose
worst behavior (W-rank collapse → 5–39% worse than an $n$-gram on real
text) is likewise pinned and honestly reported. The deepest result of
the comparison is the shared failure mode: analytic heads do not
degrade gracefully, because they have no loss surface — they have only
the geometry they assumed. Both projects' next steps (AnaCP-style
contrastive separation for degenerate frames; off-diagonal generators
for degenerate vacua) are the same idea applied to the two geometries.

## References

1. Momeni, Xiao & Liu. *AnaCP: Toward Upper-Bound Continual Learning via
   Analytic Contrastive Projection.* NeurIPS 2025. (Method of `Mode::Anacp`.)
2. Momeni, Xiao & Liu. *Continual Out-of-Distribution Detection with
   Analytic Neural Collapse.* AAAI 2026. (Method of `Mode::Ananc`.)
3. Layden, Sweke, Havlíček, Chowdhury & Neklyudov. *Quantum advantage in
   generative modeling with online feedback* / wavefunction flows — the
   continuity-Hamiltonian qsample bridge. 2025. (Cited as Layden2025 in
   `QFM.tex`.)
4. Hashimoto & Nodera. *Shift-invert rational Krylov methods.* 2019.
   (The SIRK compression.)
5. `unfer/docs/DECIDE_SPEC.md` and the pinned fixtures
   `decide_core/tests/fixtures/decide_calibration_report.json`,
   `decide_d_ablation.json` — every Stack A number in §4.1.
6. `unfer/QFM.tex` (rev 35) and `unfer/docs/QFM_TEXT_STATUS.md`
   (rev 35/37) — every Stack B number in §4.2.
7. Mehler (1866); López & Temme (1999) — the Gegenbauer→Hermite /
   Gaussian-vacuum identification underlying the Fock encoding.
8. Onsager (1944) — the critical 2D Ising temperature used in the
   cross-domain validation.

*All prose in this article is ours; cited methods are attributed to
their papers, and no third-party source text is reproduced. Latency
measurements in §4.1 were taken for this article (harness:
`/tmp/decide_bench`, not part of any repository).*
