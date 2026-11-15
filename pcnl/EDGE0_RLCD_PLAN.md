# Small Edge0 decision model with required RLCD

Status: proposed research and development, 2026-11-15; no implementation, training,
numerical test or Lean compilation is claimed here. **RLCD is required**, including a
reinforcement-learning update of a stochastic decision policy. Supervised training is
an optional warm start and a mandatory comparator, not a replacement for that update.

This is R9 of [RESEARCH_PLAN.md](RESEARCH_PLAN.md), executed through E0–E7 of
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). R8/D0–D7 still govern sampler certificates.
The stopped R7 DT arm retains its results; this track tests a new architecture, task and
training hypothesis under a separate budget.

## 1. Sources and the architecture being adapted

The requested `../Edge-0/` is available locally as `../Edge0/`. Pin the inspected source
at commit `7b60076` and preserve its revision in every experiment manifest. Read:

- [Latent MoE model description](../../Edge0/docs/models/edge0-145b.md): grouped shared
  down/up projections around compressed SwiGLU experts, routed experts and a shared expert.
- [Latent implementation](../../Edge0/edge0-strata/src/latent/latent_moe.cpp): expert group
  labels select the compressed activation and expansion; route weights combine outputs.
- [Routing families](../../Edge0/docs/moe.md): the older 35B softmax and 8B sigmoid/group
  rules are different; neither is silently identified with the current latent models.
- [Prerouter](../../Edge0/docs/prerouter.md): predicted routes can replace actual routing,
  changing the computation. This is not automatically a probability-preserving prefetch.

Use the latent D/U architecture as the main structural reference. The small model below
chooses its own softmax/top-k rule and ordinary attention; it is a documented adaptation,
not a miniature checkpoint with identical Edge0 attention, routing or inference semantics.
No large checkpoint, native quantization kernel, SSD scheduler, PLE table or speculative
text decoder is needed for this research model. Streaming/predicted routing is a later
ablation, after the actual-route probability model is established.

[Jev's launch description](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
names RLCD and parallel typed decisions. Use that output goal; define our training method
explicitly rather than claiming to reproduce an undisclosed proprietary recipe.
[Mammoth-1's model card](https://huggingface.co/baselabs/mammoth-1) describes a 27B model
with state/questions and typed probability outputs. Its
[technical report](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
specifies cross-entropy plus Brier training. It supplies a supervised comparator and an
interface reference; **our required RLCD stage is an independent research design**.

The former OpenJev-RLCD preprint is now
[PRDO, version 2](https://arxiv.org/html/2609.38850v2). It separates per-rationale from
mixture scoring and describes direct readout gradients without rationale rewards in its
main arm. Do not cite it as verification of our RL policy-gradient estimator. Pin version
and estimand when reusing earlier local `readout.rs` comments or experiment descriptions.
The unavailable sibling `openjev-rlcd/` is not an executable dependency of this plan.

## 2. A configuration small enough to enumerate and understand

The first configuration is text/structured-state only, trained from scratch on synthetic
worlds. Use a fixed auditable vocabulary of at most 512 symbols and at most 128 tokens per
state/question pack. A token limit is an input contract, not silent truncation.

| Component | Initial configuration | Required comparison |
|---|---|---|
| Shared state encoder | 4 blocks, width 64, 4 attention heads | Same-width dense encoder |
| Latent expert layer | 8 experts, 2 D/U groups, latent width 32 | No compression; dense FFN |
| Expert computation | SwiGLU width 64; top-2 + shared expert | Dense soft routing; top-2 |
| Routing | Softmax/top-k; renormalize; fixed tie rule | Route/load diagnostics |
| Questions | Independent blocks attend to shared state | Packed versus separate evaluation |
| RL policy | 4 latent paths per question | Fixed/uniform policy; exact enumeration |
| Outputs | Bernoulli, Choice, ordinal Score | Same typed heads in every arm |

Treat the four latent decision paths as explicitly stochastic computation choices, not
natural-language explanations or the eight deterministic MoE expert routes. Give them
separate state, parameters, RNG streams and manifest fields. The path-conditioned readout
may use a small path embedding/adapter on the question representation. This design makes
an actual policy-gradient update possible without adding an autoregressive rationale LLM.

Target **less than one million total parameters** in the first implementation. A
bias-free state-block estimate is
`4 d² + 2 G d ell + 3 E ell f + 3 d f_shared + d E` per block, before norms,
embeddings, question blocks and output/path heads. Count every trainable/frozen tensor
and report total parameters, activated parameters, operations and measured memory; the
budget is proposed, not an already measured parameter count or latency result.

The state cannot attend to questions; questions cannot attend to each other. A packed
attention mask must enforce both rules and retain explicit position/segment semantics.
Cache state activations only for an identical state, encoder revision and inference
configuration. Training, posterior changes to that encoder, dropout or route changes
invalidate the relevant cache. Question-key permutations must not change predictions;
choice permutations must permute probabilities, while ordinal level order remains meaningful.
The initial prototype may use shared option scoring to make choice equivariance structural.

## 3. Define the returned probability distribution first

For state `s`, question `q`, latent path `r`, head parameters `theta` and policy parameters
`phi`, define

```text
r ~ a_phi(. | s,q),                 r in {0,1,2,3}
u_theta(. | s,q,r) in Delta(K_q)
p_phi,theta(y | s,q) = sum_r a_phi(r | s,q) u_theta(y | s,q,r).
```

**The primary deployed output is this exactly enumerated four-path marginal.** Its
cost is charged to all comparisons. Training samples paths for the RL estimator in §4;
inference sums four paths. Ordinary deterministic MoE weights alone do not make the
model a Bayesian posterior, and latent D/U coordinates are not posterior samples.

A Noul head exposes `P(yes)` and its complement. A Choice head returns one normalized
probability per declared option. A Score head returns a normalized distribution on all
ordered rubric levels; its mean and quantiles are derived fields. All use finite logits
and stable normalization, with explicit failures for invalid/empty schemas and nonfinite
values. A repaired uniform output must not hide numerical failure in scoring.

These are conditional **marginals per question**. They do not determine a joint law of
several answers. For workflows depending on correlated answers, enumerate a common latent
world or specify conditional factors and validate their joint distribution. Multiplying
separately calibrated marginals requires an independence assumption that may be false.
An abstain action is a utility decision; an outcome class named “unknown” needs its own
labeled semantics and is not automatically calibrated epistemic uncertainty.

An optional Bayesian arm subsequently adds
`P(y|s,q,D)=E_{theta,phi~pi_D} p_phi,theta(y|s,q)` under an identified target.
This outer parameter average is distinct from latent-path marginalization. Report path
and parameter uncertainty separately; neither entropy nor disagreement alone establishes
calibration or identifiable aleatoric/epistemic uncertainty.

## 4. Required RLCD objective and gradient contract

Use externally labeled outcomes `Y` or known synthetic outcome distributions `t`; model
self-confidence is not a reward label. The primary population objective is expected
Brier loss of the **returned marginal**:

```text
B(p,Y) = ||p - e_Y||²
F(phi,theta) = E_(s,q,Y) B(p_phi,theta(.|s,q),Y).
```

At each input the excess expected Brier risk is `||p-t||²`. Strict propriety elicits the
conditional outcome distribution under unrestricted optimization; it does not prove that
finite training/capacity reaches it. Use categorical NLL as a second proper endpoint and
an ordinal ranked-probability score as an optional declared Score-head endpoint. ECE is
a held-out diagnostic, not an extra default training reward.

**The required reinforcement step trains `a_phi` with a score-function policy gradient.**
For `m=4` conditionally independent sampled paths, let `u_i=u_theta(.|s,q,r_i)` and use

```text
B_U = 1 - (2/m) sum_i u_i[Y]
        + (1/(m(m-1))) sum_(i != j) dot(u_i,u_j).
E[B_U | s,q,Y] = B(p_phi,theta,Y).
```

For fixed sampled paths the readout has a direct differentiable loss. The full gradient
also includes `B_U sum_i grad log a_phi(r_i|s,q)`. Gradient descent on this loss is
REINFORCE with reward `-B_U`; include both terms for shared parameters. A baseline for
sample `i` may depend on the input, label and other independent paths but must not depend
on `r_i` given those variables; detach it in the score-function term. Verify any proposed
leave-one-out advantage by enumeration before using it. Clipping, importance ratios,
policy regularization and entropy/load penalties change the objective or its gradient;
introduce them only with named equations and an ablation, not hidden optimizer defaults.

Start with all paths at positive probability. Enumerate all `4^4=256` path tuples to
compare the expected stochastic gradient against differentiating the exact four-path
marginal. Check policy, readout and shared-trunk components independently. Repeat at
nonuniform policies, noisy soft outcomes and unequal readouts, where correctness-only
rewards or missing score terms are detectable. This is the first RLCD acceptance gate.
A noisy training loss `B_U` can be negative; it is an unbiased estimator, not a per-example
probability or the nonnegative deployed Brier score.

Scoring the empirical mean directly instead targets
`E B(p_hat_m,Y)=B(p,Y)+tr Cov(u)/m` for iid paths. That is a valid *different* finite-draw
objective and regularizes disagreement. Label it as an ablation; do not call it an
unbiased estimate of `F`. A single-path deployment instead requires `E_r B(u_r,Y)`.
Do not transfer a mixture calibration claim to a returned single path. For correlated
pCNL states, use the covariance-lag formula in R3; independent-path proofs do not apply.

Supervised CE+Brier training on the exact marginal is the mandatory matched baseline and
optional warm start. Then execute the required RLCD policy and readout updates. Compare
with exact-marginal Brier backpropagation through the policy as a same-objective gradient
control, frozen-policy direct readout updates, per-path proper-score training and a
correctness-reward policy to identify what the RL term changes. No RL gain is presumed:
exact marginal differentiation may be more efficient on four paths, which is a useful
result rather than a reason to omit the requested reinforcement arm.

## 5. Research task, decisions and evidence

Generate small latent worlds with known, nondegenerate conditional outcome distributions.
Use binary propositions, 3–5 way routing choices and 3–5 ordered levels, with missing or
noisy observations and repeated labels from the same world. Compute the true posterior
outcome law by finite enumeration. Withholding information then has a known statistical
meaning; a sharp wrong answer can be distinguished from honest ambiguity.

Split by world/template family before producing questions or repeated labels. Reserve
separate training, development, calibration and final test partitions; paired questions
from one state are one statistical cluster. Keep a fixed compositional/OOD partition.
A small symbolic vocabulary tests learning mechanics; it does not establish general
natural-language competence or inherited Edge0/Mammoth capabilities.

Predeclare the following bounded first campaign after correctness gates:

- One configuration above; 5 paired training seeds; fixed train/dev/calibration/test
  world manifests of 2048/256/256/512 worlds, with at most 8 typed questions per world.
- At most 2000 optimizer updates per arm, batches of 32 worlds and `m=4` training paths;
  one frozen pilot configuration before accessing final outcomes. Record all passes and
  evaluation calls, including exact enumeration and temperature fitting. Set a hardware
  wall-time cap before execution and stop at it; no unrecorded extra seed/width search.
- Main contrast: latent-MoE RLCD versus the same model, same warm start, same data and
  evaluation budget with supervised CE+Brier alone. The mandatory same-objective control
  differentiates exact-marginal Brier through the policy. Secondary controls are frozen-policy,
  per-path proper scoring and correctness-reward RL; add dense versus latent-MoE as a
  separately budgeted architecture comparison. Match total and active compute explicitly.
- Primary test endpoint: mean Brier excess over the known law, averaged with fixed
  per-type weights. Report raw and development-temperature-scaled baselines, NLL, ordinal
  score, decision utility/regret, reliability diagrams and coverage/risk separately.
  Fit abstention thresholds using only the calibration split and their actual risk-bound
  assumptions; “maximum probability exceeds threshold” alone is not a risk certificate.
- Use paired world-clustered uncertainty and independent seed replication. Mark additional
  comparisons exploratory or adjust their family. RLCD improvement needs a prespecified
  interval excluding no improvement against the strong baseline at the declared cost.
  If inconclusive or worse, retain the distribution/gradient results and stop promotion.

For a bounded utility table, decisions maximize expected utility under the returned
probabilities. Synthetic worlds supply exact optimal actions and regret. Prove the
finite-action stability bound from probability error; utility weights must not alter
probabilities at serving time. Calibration on a test population does not imply individual
certainty or guarantee calibration after distribution shift. Sequential applications
need the separate uniform-history/finite-horizon premises of Q5.

## 6. Connect to pCNL and Lean without confusing the objects

First freeze the state encoder, deterministic routing and latent policy; sample a small
head/adapter chart through the existing `Potential` / `SubspaceTarget<P>` interface.
Use the exact four-path marginal in the data potential. A categorical likelihood has
`Psi = lambda sum_data -log p(y|s,q)` and total `J=Psi+||z||²/2` in standardized prior
coordinates. Store the prior, temperature, chart, frozen components and readout identity.
Brier-based Gibbs targets are generalized Bayes and must be labeled separately.

**Do not use freshly sampled paths inside an otherwise exact MH acceptance ratio.**
A stochastic log likelihood or `B_U` is not a nonnegative unbiased likelihood estimator.
With four paths exact likelihood evaluation is cheap enough; later random likelihoods
need an extended-state/pseudo-marginal construction or an explicit approximate-kernel
error bound. The existing sampler does not differentiate through its stationary law.
An outer training objective involving posterior averages therefore needs its own gradient
or a clearly frozen-posterior protocol, not an unproved RLCD-through-MCMC identity.

Only after the head-only reference passes may a separate arm vary the latent policy,
router or trunk. Freeze router *parameters* alone does not freeze routes when hidden
activations change. Top-k ties can cause nonsmooth changes: exact MH needs a measurable
target, while Langevin/SG gradient and operator claims need additional regularity.
Compare all four error sources from R8 and use D1/D2 for actual sampler certificates.

The LLM-Lean4-specialist should implement the following finite statements first, with
new declaration names treated as proposals until they exist:

1. Schema-indexed simplex outputs; normalization and convex-mixture closure; option
   permutation equivariance under the chosen shared scorer/mask assumptions.
2. Brier excess-risk identity, ordinal score propriety, and the distinction between
   marginal and single-path risk; finite iid U-statistic expectation.
3. Finite-sum policy-gradient identity including direct readout derivatives, and the
   conditional-independence condition for baseline subtraction. No stationarity premise
   may be smuggled in for correlated chain draws.
4. Bounded-utility regret control and finite-mixture approximation bounds. If utility is
   in `[0,1]`, optimal-action regret is at most twice the predicted/true TV distance.
5. The identified prior-referenced target and exact reference kernel properties from
   Q0/Q1; bridge any new posterior implementation before claiming its rate.

Proofs 1–4 concern probability/decision algebra, not a quantum Hamiltonian. Any optional
Fock/operator representation still obeys the full inner-operator and outer `dGamma`
contract in [timepiece/pcnl.md](../../timepiece/pcnl.md); it is not needed to train RLCD.
Compilation and proof validation belong to the Lean4-specialist after implementation.
