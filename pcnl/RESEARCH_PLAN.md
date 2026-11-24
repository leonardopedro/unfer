# pcnl research and development plan

## R13 — active plan for objective-based continual learning with numerical-vector outputs (2026-11-22)

**Deliverable.** Given English text and one retained parameter draw, return
$f_\theta(x)\in\mathbb R^q$ in named coordinates with application-defined units.
Negative values, zero, and magnitudes above one are valid. A caller may also request
a posterior mean vector or coordinate intervals over draws. The probability
distribution that pCNL approximates lives on the trainable parameters. Pushing its
draws through the transformer gives uncertainty about its vector prediction.
R13 governs new implementation and acceptance. R0–R12 below are the research record.

### What the current checkout provides

| Existing work | What carries forward |
|---|---|
| `src/model.rs` has configurable transformer blocks, a flat parameter layout and an exact-gradient option. Its public forward takes token IDs and ends in a vocabulary readout. The UltraBERT scripts produce frozen 768-dimensional CLS vectors. | Connect the fixed English encoder to trainable transformer blocks and an affine numerical readout. The rotation CLI presently reads fixtures or caller-supplied embeddings. |
| `src/target.rs::Potential` exposes term count, mean loss and gradient. `src/rotation_head.rs` has full-data MH transitions, checkpointed chains/RNGs, retained chain/step provenance, pilot preconditioning and replica exchange. | Reuse these mechanics with an objective-specific target and checkpoint version. The rotation likelihood, state payload and reported categorical metrics belong to R12. |
| The Nov 21 20 Newsgroups rotation pilot accepted 57,523 of 64,000 proposals across two chains, yet reported predictive R-hat 1.152 and parameter R-hat 2.125. The effect of tempering on predictive scores reversed between two seeds. | Measure agreement per input, parameter projection and mode. Iteration and acceptance counts alone give no finite-run accuracy guarantee. [Measured runs](../docs/PCNL_STATUS.md) |
| `src/edge0_potential.rs::prior_refresh_step` already implements an independent Gaussian refresh on a restricted chart. The original nGPT work also has an unadjusted mini-batch sampler. | Lift the refresh to the entire declared parameter space. Keep full-data adjusted pCNL as the reference before evaluating mini-batch bias. [Development plan](DEVELOPMENT_PLAN.md) · [Original sampler plan](../docs/plan_bayes_ngpt.md) |
| `task::sgld_chain` injects Gaussian noise in the token-classifier experiment. Its `BatchLoader` samples uniformly without replacement within each batch. | Reuse the batch and noise plumbing for the vector objective. At nonunit `potential_scale` the old routine scales both the data drift and the injected variance while keeping the prior contraction fixed. R13 needs an explicit likelihood-only temperature as specified below. [Classifier sampler](src/task.rs) |

### 1. Expose a four-layer transformer with a raw vector readout

Share the encoder adapter between fixture generation, prediction and update. Pin
`dmis-lab/UltraBERT-Embed-150M` and its tokenizer at revision
`b06b36724c5417a1ec1b7863dd4fa542b74e4899`, with eval mode, 96-token truncation,
CLS pooling and L2-normalized input features. For a small first design, linearly
project each fixed embedding into four ordered feature vectors of width 16, reshape
to a length-four sequence, and use the existing attention/MLP block four times with
**four distinct sets of trainable weights**. The four positions are learned features
of the passage. This also avoids the degenerate attention weights of a length-one
input. Feed the final position to an untied affine readout of width $q$. Sample
the projection, all four blocks and the readout jointly. Fix the vector schema
before training. Verify live/fixture parity, the gradient of a coordinate in each
block, and a round-trip prediction with negative and non-unit outputs.

### 2. Make the training objective the sampling contract

Let $D_t$ be the newly arrived events, $J_t(\theta)$ the data objective for the
unique prefix through round $t$, and $P_0=N(m,C)$ a fixed proper Gaussian prior.
For an additive loss, use $J_t=\sum_{i\in D_{1:t}}\ell(f_\theta(x_i),y_i)$.
The optimizer and sampler must call the same objective with a documented reduction.
If `Potential` returns a mean, multiply by its term count exactly once. Define

$$
\widetilde L_t(\theta)=\exp[-\lambda J_t(\theta)],\qquad
\pi_t(d\theta)=Z_t^{-1}\widetilde L_t(\theta)P_0(d\theta).
$$

Only log differences enter MH. Require $0<Z_t<\infty$ (e.g. a finite,
lower-bounded objective under a proper prior). Keep the Gaussian penalty in the
prior once. For fixed-scale Gaussian observations the first executable case is
$\ell=\|y-f_\theta(x)\|^2/(2s^2)$, $\lambda=1$. This gives ordinary Bayes,
with the parameter-independent normalizer cancelling. A general application loss
defines a Gibbs posterior whose concentration depends on the declared $\lambda$.
If a noise parameter is learned, keep its parameter-dependent normalization in
the objective. Version the loss, scale, reduction and prior in the checkpoint.
[Bissiri, Holmes & Walker](https://arxiv.org/abs/1306.6430)

For continual updates multiply the previous target by
$\exp[-\lambda(J_t-J_{t-1})]$. For additive losses the increment uses just the
new events. For a coupled objective evaluate the actual prefix difference. The
first implementation carries forward chain positions, recomputes the full-prefix
target and gradient, and publishes draws only for a committed prefix. An identical
event-ID retry preserves the checkpoint and RNG positions. Reject conflicting reuse.
Test partition/order equivalence for additive data and deterministic resume.

### 3. Mix local pCNL with occasional random restarts

Write $U=\lambda J_t$. Freeze $0<\rho<1$ after warmup. The Crank–Nicolson
Langevin proposal for the Gaussian reference has mean
$g(\theta)=m+\rho(\theta-m)-(1-\rho)C\nabla U(\theta)$ and covariance
$(1-\rho^2)C$. Accept using the full log posterior difference plus
$\log q(\theta\mid\theta')-\log q(\theta'\mid\theta)$ with the same full-prefix
data at both ends. The existing centered rotation kernel includes the prior
gradient in its drift as well as shrinkage. Give the specified proposal its own
identity and check its forward and reverse densities at the actual endpoints.
At constant data objective it reduces to prior-preserving pCN with unit
acceptance in exact arithmetic. [Cotter et al.](https://arxiv.org/abs/1202.0709)

Before each transition, flip a state-independent Bernoulli coin. With probability
$1-r$ take the local step. With probability $r$ propose an **entirely fresh**
$\theta'\sim P_0$, accepting with $\min(1,\exp[U(\theta)-U(\theta')])$.
A rejection retains the current state and counts as a chain step. Both kernels
preserve the same prefix posterior, so their fixed mixture preserves it too.
Start the pilot at $r=0.01$ and compare with $r=0$ at matched objective/gradient
cost. Log attempts, acceptances, cross-mode moves and dwell times by chain.
Concentrated likelihoods can make prior jumps rare. If the mixture misses
material modes, test the existing tempering strategy at equal cost or use a
frozen independence proposal with its full Hastings correction. For unconditional
random reinitialization, launch a separately identified exploratory chain with
fresh warmup. Pool only after checking equilibration and mode weights. Forced
resets interleaved with retained draws would bias those weights.

### 4. Verify distribution recovery before claiming text performance

1. **Known targets.** Check posterior moments on the linear-Gaussian reference.
   Then use a one-dimensional two-mode target with known unequal basin masses
   (e.g. a 75:25 Gaussian mixture with well-separated means and equal variances).
   Derive $U=-\log(\pi/P_0)$ and compute reference half-line masses from the mixture
   CDF. Start chains in opposite basins. Compare local-only and restart-enabled
   estimates of masses, transitions, means and vector expectations against
   quadrature with autocorrelation-aware uncertainty. This checks whether the
   sampler gives regions their proper mass, including broad and narrow regions.
2. **English stream.** Until an application supplies its vector schema and
   objective, use a reproducible synthetic regression task on the fixed English
   embeddings. Generate signed, non-unit numeric targets with a fixed teacher and
   known observation noise. Mark results as synthetic. Split by document ID, score
   incoming vectors before revealing targets, then compare each warm-started prefix
   with a fresh joint fit, a current-batch-only reset and a point-trained transformer
   using the same objective. Report held-out objective, per-coordinate RMSE, past
   task retention, predictive intervals with the stated noise model, time and bytes.
3. **Release evidence.** Save the target and model manifest, event prefix, seeds,
   proposal settings and per-chain draws. Predeclare budgets and error tolerances
   on development data. Require agreement with the known-target references within
   tolerance including Monte Carlo uncertainty. Use at least four dispersed
   chains for promotion, rank-normalized split R-hat below 1.01 and bulk/tail ESS
   of at least 400 on declared projections, mode indicators and per-input vector
   observables. Report an unresolved mode or a failed precision gate explicitly.
   On real high-dimensional targets these checks support estimates of specified
   observables in explored regions. They cannot certify that every mode was found.

### 5. Test whether stopping at a slowdown improves a restart ensemble

Use explicit likelihood temperatures $\pi_T(d\theta)\propto
\exp[-U(\theta)/T]P_0(d\theta)$, where $U=\lambda J_t$. Hold the prior, prefix,
loss units and $\lambda$ fixed throughout each comparison. The nominal target is
$T=1$. At another temperature use $U/T$ consistently in the local drift, target
ratio and prior-refresh acceptance. In replica exchange record the whole ladder
and retain the replica at the temperature being estimated. Hot auxiliary chains
can improve travel while preserving the nominal marginal. In the R12 two-seed
8k study both ladders accepted swaps (13.3% and 12.1%), but the predictive NLL
improvement reversed by seed. Those runs concern a signed classifier.
[Measured R12 runs](../docs/PCNL_STATUS.md)

**Basin-mass check.** The target weight of a specified basin $B$ is proportional
to $\int_B e^{-U/T}\,dP_0$. A broad region can outweigh a deeper minimum.
For an exact arithmetic control take two regions of constant energy with prior
masses in the ratio 9:1, giving the broader region one extra unit of energy.
Its posterior odds are $9e^{-1/T}$, so it wins at $T=1$ and loses at $T=0.25$.
For the differentiable two-mode sampler test in §4, add an unequal-width
variant, define basin boundaries before sampling, and integrate the tempered
density afresh at every temperature. Compare analytic or quadrature mass with
the fraction of retained stationary draws per basin, including repeats after
rejection. Count optimizer endpoint frequencies separately. A deeper, narrower
network's preference for high-mass basins is an experimental hypothesis.

**Paired stopping experiment (proposed).** At a fixed evidence prefix, use
the same random starts, minibatch order and optimizer settings for both checkpoints
on each trajectory. As an initial knee rule, record full fit loss every 50 updates
and stop at the first three consecutive intervals with fractional decrease below
1% (divide by the absolute preceding loss floored at a declared epsilon). Flag
loss increases separately, require a decreasing trend over the window, and record
a missing knee at the cap. Continue the same trajectory to a preregistered
tighter plateau or budget cap at constant learning rate. Add a paired long-run
control with a declared learning-rate decay. Freeze both schedules and the
stopping rule on development data. Save the
vector predictions from both points. Compare individual fits, equal-weight
ensembles at equal start count, and ensembles at equal total gradient cost,
allowing the knee arm to spend saved work on extra starts. Report starts, missed
knees, endpoint frequencies, per-input vector spread, held-out objective/RMSE,
old-task retention and paired uncertainty across seeds and documents. Include
capped runs at their last checkpoint even when a knee was never detected.

On the same prefixes compare with the adjusted R13 sampler at $T=1$ and a small
prespecified likelihood-temperature grid (pilot $T=0.25,1,2$). Keep the §3
random MH refresh active and its rate fixed for this comparison. Knee checkpoints
can also initialize independent pCNL chains with fresh warmup. Estimate posterior
vector means from retained draws and report ESS/MCSE and occupancy at each
temperature. If the knee ensemble improves held-out predictions while its mode
frequencies disagree with the reference, report an optimizer-ensemble gain.
If the colder target improves predictions across held-out splits after passing
mixing gates, report a temperature effect for that objective and prior. Neither
test assigns a temperature to an optimizer iteration. The manuscript gives the
basin derivation and the distinction from full-posterior tempering.
[Manuscript discussion](../../test/unfer/ngpt.md) · [Wenzel et al.](https://proceedings.mlr.press/v119/wenzel20a.html)

### 6. Measure what SGD samples and what PAC-Bayes bounds

**Sampling check.** For density $p_0$ of the Gaussian prior, write
$F_T=U/T-\log p_0$. Compare ordinary minibatch SGD, explicit-noise SGLD and
the §3 adjusted pCNL reference. For identity mobility the SGLD step is
$\theta'=\theta-h\widehat{\nabla F_T}+\sqrt{2h}\,\xi$ with an independent
standard Gaussian $\xi$. Scale the batch **mean** data gradient by $\lambda N/T$
for the additive objective on $N$ events, then add $C^{-1}(\theta-m)$ once.
Test step-size bias on the Gaussian reference and mode occupancy on the two-basin
reference at the declared $T$. Decreasing $h$ refines this discretization.
Changing $T$ changes the target. [Welling & Teh](https://icml.cc/2011/papers/398_icmlpaper.pdf)

For plain SGD record loss units, learning rate, actual batch size and batch
sampling law. First use a quadratic with equal curvature and unequal gradient
noise in two directions, as derived in the manuscript. At each text-run knee
and in subsequent fixed-length windows, measure the batch-gradient covariance in
fixed projections at the checkpoint, drift in the window means, autocorrelation
and disagreement between independent starts. Compare with pCNL on the same
observables. Report a slowdown as an optimization result even if the local
covariance or mode weights fail the sampling checks. [SGD diffusion analysis](https://jmlr.org/papers/v18/17-214.html)

**Generalization check.** Use a prespecified i.i.d. document experiment with a
data-independent Gaussian prior $P_0$. Choose the per-document score
$b=\min(1,\|f_\theta(x)-y\|^2/(q a^2))$ with coordinate scales fixed in advance
and $a>0$ in the resulting units. Continue to train on the original objective
and report RMSE in task units. For the knee and late arms construct an explicit
equal-weight mixture $Q$ of full-support diagonal Gaussians around their
checkpoints. Measure empirical risk **under perturbation**, using fresh
independent draws from the fixed mixture. Upper-bound mixture KL by the mean
analytic component KL to $P_0$. Report empirical risk, its one-sided Monte
Carlo allowance, KL, number of independent documents and the PAC-Bayes upper
risk bound from the manuscript. Split the confidence budget between data and
Monte Carlo error and across repeated reports as needed. Record vacuous bounds.
A point ensemble, a smoothed randomized predictor and a distilled student
each get their own evaluation. On shifted or adaptively collected streams
report prequential errors. The i.i.d. bound applies to its declared population.
[Maurer, Theorem 5](https://arxiv.org/pdf/cs/0411099) · [Dziugaite & Roy](https://arxiv.org/abs/1703.11008)

### 7. Compare the cost of serving one vector model

| Candidate | Construction and comparison |
|---|---|
| One long fit | Train with the §5 decay schedule. Score the final vector predictor and record training cost and single-pass latency. |
| SWA within one run | Continue SGD beyond the knee at a fixed learning rate. Predeclare the window and spacing, and average its $K$ parameter vectors online. Compare the last iterate, SWA and the output average of those same snapshots on identical passages. Measure the objective along sampled straight-line interpolations and at the actual mean weights. |
| Ensemble across starts | Average raw vectors from $M$ independent knee checkpoints. Also test one SWA model per start, charging the extra post-knee updates. Count distinct prediction patterns and use the actual teacher count even if several starts give similar functions. |
| Distilled student | For one committed evidence prefix cache the teacher's mean vector on training or unlabeled transfer passages available at that prefix. Fit one four-block student by squared vector discrepancy. Run this separately for the restart ensemble and for retained draws from well-diagnosed pCNL chains. Measure teacher–student error and ground-truth error on unseen passages and old tasks. |

If direct merging of independent runs is investigated, first check a
function-preserving hidden-unit permutation, then compare the unaligned and
aligned interpolation losses, merged outputs and inference costs. Keep a
label-trained student with the same architecture and initialization as the
distillation control. [Git Re-Basin](https://arxiv.org/abs/2209.04836)

Freeze the aggregate training budget, candidate start counts and a tolerable
increase in held-out error relative to the teacher on development data. Charge full-loss knee checks,
extra snapshots, teacher forwards and student fitting. At inference share the
frozen English embedding, then measure head evaluations, memory and end-to-end
latency on the same hardware and batch size. Compare held-out error differences
with paired uncertainty at equal training cost. Promote a single-pass model
when the upper confidence limit on its paired error increase is within tolerance and it
meets the serving budget. Keep the evidence and pCNL chain state for the next
Bayesian update. Keep teacher targets outside the evidence log. If a serving
application also needs uncertainty, fit and validate a separate per-coordinate
spread estimate against retained draws and check interval coverage under the
declared observation-noise model. The decision readout remains a numerical vector.
[SWA](https://arxiv.org/abs/1803.05407) · [Ensemble distillation](https://arxiv.org/abs/1503.02531)

R13 is delivered when a raw-English predict/update/reopen run reproduces the
vector for a saved draw, the four-block and objective checks pass, and the
known-target and stream reports quantify posterior accuracy and restart behavior.
The SGD, PAC-Bayes and compression comparisons are proposed follow-ups to that
reference. Their results determine which predictor to serve and which claims the
evidence supports.

### First execution record (2026-11-22)

`src/vector_model.rs` now implements the four-position, four-block vector head,
its flat layout and layer-scaled prior, the full-data Gaussian regression
[`Potential`](src/target.rs), and the exact CN/prior-refresh mixture.
`examples/vector_regression.rs` loads the frozen 20 Newsgroups embeddings,
derives signed two-coordinate targets from a fixed synthetic teacher, runs
full-batch Adam and the sampler, and writes/reopens a JSON checkpoint.

Measured pilot (`240` train / `120` test rows, `768 → 4×16 → 2`, teacher seed
`20261122`, observation noise `0.05`, 300 Adam steps): train RMSE `0.000286`,
test RMSE `0.099276`; the checkpoint round-trip is bitwise exact and predicts
`[-1.6128643, -0.82934713]` on the first test row.  At `ρ = 0.99` and prior
scale `0.05`, the CN sampler accepted `0/397` local moves: the prior covariance
is far wider than this 240-row posterior.  At `ρ = 0.9999` and prior scale
`0.01` it accepted `224/398` CN moves and `0/2` prior refreshes, retaining 20
draws; the draw-mean test RMSE was `0.101129`.  Parameter-only 90% intervals
covered `14/240` coordinates and noise-augmented predictive intervals covered
`142/240`, so the latter still fails nominal coverage.  This is a synthetic,
warm-started, single-chain pipeline check, not a text-performance, equilibrium,
calibration or promotion result.  Live-encoder parity, known-target recovery,
four-chain rank-normalized R̂/ESS and the continual stream remain open.

### R13 follow-up measurements (2026-11-22)

**Known-target recovery passes its bulk gate.** `examples/known_target.rs` runs
the production CN/refresh kernel through `EnergyTarget` on
`0.75·N(−2,1) + 0.25·N(2,1)`, four chains from `[-6,-3,3,6]`, 50,000 steps,
10,000 burn, thin 5, 8,000 draws per chain.  Pooled left-basin mass `0.73378`
(analytic `0.73862`, error `−0.00484`, MCSE `0.01304`), mean `−0.97689`
(analytic `−1`), central 90% `[−3.50191, +2.87135]`; rank R̂ `1.00344`/`1.00547`
and bulk ESS `1504.8`/`1148.5` on coordinate/basin indicator.  The binary
indicator's tail ESS is `NaN`, so only the bulk gate is claimed.

**The four-chain vector gate fails.** Four jitter-separated warm starts
(`jitter=0.01`), 1,000 steps, 250 burn, 200 draws per chain: at `ρ=0.999` the
CN acceptance was 82/89/86/102 per ~1,000, rank R̂ `2.69621`, minimum bulk ESS
`4.7` (tail `9.3`).  At `ρ=0.9999` acceptance was 538/523/564/544, but rank R̂
`2.74392` and bulk ESS `4.6`.  A 300-step probe of the same starts placed the
per-chain first-row draw means within ~0.04, indicating slow 57,378-parameter
mixing rather than distinct modes.  The
run is too short for the observed integration time and retains too few draws;
these are reported failures, not a promotion result.

**Gradient-scaling defect (fixed).** The `EnergyTarget` adapter for
`VectorRegression` returned the energy scaled by `λ·n` but the **unscaled**
mean-loss gradient, so the CN drift term was `240×` too small at `n = 240`.
The adapter now scales the gradient identically.  After the fix, single-chain
300-step probes accepted 261/295 CN moves at `ρ = 0.9999`, 13/290 at
`ρ = 0.999` and 0/299 at `ρ = 0.99`; before the fix a four-chain `ρ = 0.999`
run accepted only 82–102 of ~1,000 per chain, so the fixed drift materially
raises acceptance but does not change the mixing time.

**A longer run confirms a dimension wall, not a length problem.** Two chains of
5,000 steps (1,250 burn, 3,750 draws each) at `ρ = 0.9999` accepted 4,620 and
4,608 CN moves but still gave rank R̂ `1.927–2.216` and per-observable bulk ESS
`2.5–4.9` (tail `10–47`).  At 92% acceptance this is an integrated
autocorrelation time of a few thousand steps: reaching bulk ESS `400` would
need roughly `10⁶` gradient evaluations per chain.

A pilot-preconditioned diagonal proposal does not rescue it.  At `ρ = 0.99`
the kernel's shrunk mean `ρ·θ` moves each coordinate by 1% of its value per
step, which is orders of magnitude larger than the local posterior step, so
every proposal is rejected: `0/2,981` in a 34-coordinate pilot-preconditioned
run at `ρ = 0.99` (pilot 400 steps, `var/prior` p50 `6.2e−3`, inflate `0.5`),
and `0/1,984` for the first chain of the full-dimension run at `ρ = 0.99`
(pilot inflate `2.0`) before it was stopped.  With an exact
posterior covariance the kernel's ESS per step is approximately
`acc·(1−ρ)/2` with `log acc ≈ −6(1−ρ)d`, whose maximum is about
`1/(12·e·d)` — `~1e−3` per step at `d = 34` and far smaller at `d = 57,378`.
The CN/refresh kernel, as specified, cannot reach the four-chain gate at this
model size in any practical budget.

**Readout-conditional target.** `ReadoutTarget` (and `--readout-only` in the
example) samples only the last 34 flat coordinates — the `16×2` affine readout
and its 2 biases — with the 57,344-coordinate backbone frozen at its Adam point
estimate.  This is an explicitly conditional target, not the full posterior.
Four chains of 3,000 steps (750 burn, thin 2, 1,125 draws each) at
`ρ = 0.9999` with a 400-step pilot covariance accepted only 70/2,971, 71/2,966,
84/2,979 and 83/2,968 CN moves (no prior refreshes), giving rank R̂ max
`3.53012` and minimum bulk ESS `4.4` (tail `8.3`); the gate still fails.  The
pilot variance overestimates the stiff-direction posterior width, so the
proposal remains too wide even at `d = 34`.  Draw-mean test RMSE was `0.099114`,
parameter 90% coverage `0.0292` (7/240) and predictive coverage `0.6125`
(147/240), so the intervals are not calibrated.

**Exact readout-conditional posterior.** The readout-conditional target is
linear-Gaussian: with design `Φ`, noise variance `s²`, prior scale `λ` and prior
covariance `c`, the precision is `A = (λ/s²)ΦᵀΦ + diag(1/c)` and the posterior
is `N(m, A⁻¹)`.  `ReadoutTarget::exact_posterior` forms it by Cholesky and
`--proposal exact` uses it in the CN kernel.  On the fitted Adam backbone the
gradient at `m` has RMS `8.673e−4` (the Gaussian self-consistency check), and
the posterior-to-prior variance ratio is `p10 7.719e−2`, `p50 1.852e−1`,
`p90 3.320e−1`.  The bias coordinates of `m` are `−1.487` and `−0.730` with
prior standard deviation `0.05`, i.e. the posterior sits ~15–30 prior-σ from the
origin.

**Prior-referenced CN rejects once the mean is far out.** With reference mean
zero the proposal is `N(ρθ, (1−ρ²)c)`, which is astronomically far from a
posterior centred at `m`; a 200-step scan from the exact draw accepted
`0/199` at `ρ = 0.5` and `ρ = 0.9`, `2/199` at `ρ = 0.99`.  The
posterior-referenced variant `run_chain_with_posterior_reference` replaces the
reference with `m` and the prior pull with the full target gradient:
`mean = m + ρ(θ−m) − ((1−ρ²)/2)·A⁻¹∇U`, proposal covariance `(1−ρ²)A⁻¹`.
This removes the prior-mean mismatch but the proposal is still under-dispersed:
from a tail draw it accepted `0–2/199` at `ρ ≤ 0.9` and `102/199` at `ρ = 0.99`.
A dense-kernel defect found in this work was real and is fixed: the quadratic
form is `δᵀAδ = ‖Lᵀδ‖²` over the *columns* of the Cholesky factor (`L Lᵀ = A`),
not `‖Lδ‖²`.

**Warm start at the analytic mean exposes the proposal's ceiling.** Starting
all chains at `m` (new `--init mean`) with `ρ = 0.5` accepted `1281/2975`,
`1155/2971`, `1221/2962` and `1180/2970` CN moves (~40%), the rate the
exact-covariance argument predicts.  But four chains of 3,000 steps still gave
rank R̂ max `1.02163` and bulk ESS min `254.9` (tail `37.8`); only five of the
eight observables cleared ESS `400`, and `gate=false`.  The measured ESS/step is
`255/2,250 ≈ 0.11`; retaining 2,250 draws instead of 225 raised ESS only from
`223.5` to `254.9` because the short-run estimator over-credited the thinned
chain, not because the kernel has an independence ceiling.  At `0.11` ESS/step
about 3,600 post-burn draws per chain are needed for bulk ESS `400`.

**The CN then passes at that budget.** Four chains of 6,000 steps (1,500 burn,
thin 1, 4,500 draws each) accepted `2553/5953`, `2456/5941`, `2467/5935` and
`2316/5938`, giving rank R̂ max **1.00616**, minimum bulk ESS **620.9** (tail
201.3) and `gate=true`; draw-mean test RMSE `0.105528`, parameter coverage
`0.1583` (38/240), predictive `0.5917` (142/240).  So both kernels pass: CN
needs ~4,500 post-burn draws per chain, HMC needs ~750, which is the practical
reason to prefer HMC here.

**Documented dimension scaling.** `examples/cn_dimension_sweep.rs` samples
`N(0, I_d)` with the production prior-referenced CN kernel, reporting
acceptance, lag-1 autocorrelation and ESS/step for one chain of 4,000 steps
(1,000 burn), seeded by `(d, ρ)`.  With the proposal covariance set to the
target covariance `I_d` (`--exact-proposal`) at `ρ = 0.99`:

| `d` | 2 | 6 | 18 | 34 | 66 | 130 | 256 |
|---|---|---|---|---|---|---|---|
| acceptance | 0.9313 | 0.8752 | 0.7802 | 0.7153 | 0.5717 | 0.4265 | 0.3003 |
| lag-1 autocorrelation | 0.9875 | 0.9887 | 0.9915 | 0.9909 | 0.9930 | 0.9957 | 0.9957 |
| ESS/step | 0.00630 | 0.00567 | 0.00427 | 0.00456 | 0.00352 | 0.00217 | 0.00214 |

Reproduce with `--steps 4000 --dims 2,6,18,34,66,130,256 --rhos 0.99
--exact-proposal` (the exact covariance of `N(0, I_d)` is `I_d`).  The default
`--c 0.1` sweep over `ρ ∈ {0.5, 0.9, 0.99, 0.999, 0.9999}` gives best ESS/step
`0.0014–0.0179` over the same dimensions with no monotone trend.

**Reading.** The CN kernel is a random walk, so ESS/step is bounded near
`(1−ρ)/(1+ρ) = 0.0050` at `ρ = 0.99` regardless of dimension; matching the
target covariance as the proposal scale raises acceptance (0.93 → 0.30) but
leaves that ceiling in place, because it does not turn the kernel into an
independence sampler.  The measured values supersede the earlier
`1/(12·e·d)` envelope, which under-predicted the `d = 34` rate (0.0046 measured
vs `~1e−3`) though not the conclusion: 4,000 draws per chain yield ~20
effective draws, far below 400.  This is the dimension-scaling limit the
four-chain gate exposed.

**HMC with the exact metric passes the four-chain gate.** The first
diagonal-mass HMC collapsed its step size (`eps_frozen ≈ 5e−5`). Part of that
was a real convention defect: with `K(p)=½Σp²/m` and `θ′=θ+εp/m` the dynamics
are `θ″=−M⁻¹Aθ`, so `m` must be the **precision** (`M=A`), not the variance;
passing variances squares the curvature range and drives `ε` to zero. The
diagonal call sites now pass precisions; the readout diagonal run settles at
`ε ≈ 6e−3` instead of collapsing, but still fails (R̂ `1.70725`, bulk ESS `6.1`
at 400 steps), so the correlation limit is separate from the inversion. That is
why the recorded gate uses the exact whitened metric.  The exact metric is the posterior precision: in
whitened coordinates `u = Lᵀ(θ − m)` (`L Lᵀ = A`) the Gaussian target is exactly
standard normal, so `run_hmc_chain_whitened` uses unit mass, `N(0, I)` momenta
and the transformed force `Gᵀ∇U` with `G = L⁻ᵀ`, while every MH ratio is still
computed with the full target in `θ`.  Four chains, `--init mean`,
`--hmc-mass exact --hmc-leapfrog 4`, 1,000 steps, 250 burn, thin 1: dual
averaging settled at `ε = 0.587–0.638` with mean acceptance `0.835–0.856`.
Rank R̂ max `1.00935` (all eight `≤ 1.00935`), minimum bulk ESS `3000.0`
(capped by four chains × 750 draws = 3,000), minimum tail ESS `906.4`,
`gate=true`.  Draw-mean test RMSE `0.105302`, parameter 90% coverage `0.1583`
(38/240), predictive coverage `0.5875` (141/240); coverage is still not nominal.
The passing target is the **readout-conditional 34-coordinate** posterior with a
frozen backbone, on fixture embeddings, warm-started at the analytic mean; this
is an HMC kernel validation, not a full-parameter or calibration result.

**Full-parameter diagonal HMC, correctly scaled, still fails.** The full
57,378-coordinate target is prior-dominated: `adam_fit` minimizes the mean loss
without the sampling prior, so its optimum is not stationary for the sampled
potential (`start_grad_rms ≈ 152`). The new `--init mode` runs Adam on
`U(θ)+½Σθ²/c` (3,000 steps, `lr=5e−3`, final potential `29.19`,
`mode_grad_rms=1.06e−1`) and the new `--hmc-mass pilot` uses the pilot's
per-coordinate variance as a precision estimate. The pilot needs a narrower
step at this dimension: `--pilot-rho 0.9999` accepted only `4/400`, while
`ρ=0.99999` accepted `499/600` (variance ratio p50 `8.7e−4`). Four chains of
400 steps, leapfrog 4, `--start-jitter 1e−4`: `ε = 0.101–0.108`, acceptance
`0.858–0.864` (no collapse), but rank R̂ max **2.61047**, bulk ESS min **4.7**
(tail 10.0), `gate=false`; draw-mean test RMSE `0.144261`, coverage `0.7083`
parameter and `0.8333` predictive. With the inverted metric the same run gave
`ε≈3e−6` and ESS 5.5, so the fix repaired the scaling without lifting the
observable ESS: the remaining problem is the off-diagonal structure of the
data Hessian, which a diagonal metric cannot whiten.

**Legacy-gate reproducibility (2026-11-23).** A control build with the
`init_flat_prior` branch removed reproduces the shipped binary's
`--start-scale legacy` output byte for byte (mode `29.193841`, pilot `499/600`
with `var_over_prior p50 8.700e-4`, all chain lines, RMSE, R̂, ESS), so the
prior draw is strictly opt-in and cannot have shifted the legacy path.

With the complete command line, every recorded gate reproduces **exactly** under
`--start-scale legacy`: pilot metric R̂ `2.61047` / ESS `4.7` / RMSE `0.144261`
(ε `0.1017–0.1077`, coverage `0.7083`/`0.8333`), Hessian diagonal (64 probes)
R̂ `2.50388` / ESS `4.8` / RMSE `0.150189` with `clipped_to_prior` `27,622`,
low-rank 8 R̂ `2.39408` / ESS `4.9`, low-rank 32 R̂ `2.01334` / ESS `5.3`, and the
readout-conditional gate R̂ `1.00935` / ESS `3000.0` / RMSE `0.105302`. The exact
commands and log paths are tabulated in `docs/PCNL_STATUS.md`
("Legacy-gate reproducibility after the init change"), with the raw logs and
their commands committed under `logs/pcnl_r13_gates/`, and
`scripts/check-pcnl-gates` re-runs all of them and diffs the summaries against
those logs. Low-rank 32 + jitter is in that same exact list: it needs
`--start-scale legacy` too (R̂ `2.14626` / ESS `5.1`), which is why four earlier
attempts without the flag landed at `2.34–2.49` — `--hmc-jitter` re-rolls every
trajectory when a start-changing flag differs. The **tempering** cold chain was
the last open row and needed the same flag: with `--start-scale legacy --hmc-leapfrog 4`
on the 3,000-step `5e-3` mode fit it returns R̂ `2.44665` /
ESS `4.8` / RMSE `0.168598` exactly (swaps `0/240`); under `prior` the same
command gives `2.15397` / `5.1` / `0.096604`. The CN exact-dense row was also
re-executed and needs `--readout-only` — without it `--proposal exact` with
`--init mean` is refused — giving R̂ `1.00616` / ESS `620.9` on the
readout-conditional (dim 34) target. The conclusions those rows support —
tempering is not viable at this energy scale, jitter does not close the gate —
hold in the re-runs.

A first pass reported these as non-reproducible; that was an incomplete
reconstruction, not a code drift. The three flags that mattered: `--pilot-inflate
1` (the mass scales by `inflate`, so ε scales by `1/√inflate` — default 2 gave
ε `0.068–0.077` against the recorded `0.101–0.108`), the recorded 1,000 Hessian
steps rather than 400, and `--thin 1 --draws 300/750`. The last is not
cosmetic: at ESS ≈5 the retained draw set moves R̂ by ±0.5 on an identical
trajectory (`--thin 10/5/2` → R̂ `2.548/2.112/2.171`), so a gate number is only
meaningful with its `thin`, `draws` and budget recorded — store the command
line with every number from now on.

**Low-rank-plus-diagonal metric.** `LowRankCovariance::from_draws` estimates
`C = diag(v) + W Λ Wᵀ` from the pilot (rank-`k` sample-covariance directions
via a Gram eigendecomposition, diagonal residual floored at `1e−6 × max var`);
`--hmc-mass lowrank` uses `M = C⁻¹` through Woodbury (`O(d·k)` per step plus a
`k × k` solve) and draws `N(0, C⁻¹)` as `C⁻¹q` with `q ~ N(0, C)`. A unit test
recovers a correlated 3-D Gaussian. On the full target the metric is scaled
(`ε = 0.052–0.074`, acceptance 0.88–0.91) but mixing does not improve:
rank 8, 400 steps → R̂ max `2.39408`, bulk ESS `4.9`; rank 32, 1,000 steps →
R̂ max `2.01334`, bulk ESS `5.3`. ESS also did not grow with run length, so the
new `--hmc-jitter` (trajectory length drawn from `1..=2·leapfrog`, symmetric
MH) was tried: R̂ `2.14626`, ESS `5.1` at 1,000 steps, i.e. resonance was not
the cause. The pilot's covariance is the limit: a 600-step random walk that
barely moves cannot estimate a 57,378-coordinate posterior width, so the
metric under-whitens the stiff directions. The four-chain full-parameter gate
is **not** cleared.

**Hessian-diagonal metric at the mode.** `hessian_probe` forms `Hz` for
Rademacher `z` by a central difference of the sampled-potential gradient
(`(g(θ+hz) − g(θ−hz))/2h`, `h = 1e-5`, 2 evaluations per probe) and takes
`diag(H)` from Hutchinson; it reproduces the analytic diagonal of a
Gaussian-plus-prior target in a unit test. At full dimension, 64 probes cost
2.4 s, and the estimate is step-size stable (identical quantiles at `h = 1e-3`
and `h = 1e-4`), so this is not finite-difference noise. The result is
diagnostic rather than a fix: **27,622 of 57,378** entries fall below the prior
bound `1/c_i` (many negative), and the raw diagonal spans `2.2e−7 .. 2.2e5`.
Using it unclipped freezes `ε` at `1e-6` (R̂ `2.31799`, ESS `4.9`). Clamping to
`[1/c_i, 1e4/c_i]` restores `ε ≈ 6.3–7.0e-3` and acceptance 0.77–0.82, but
R̂ `2.50388`, ESS `4.8`, RMSE `0.150189`. The four chains' test predictions
(`−1.92/−1.61/−1.80/−1.93` on coordinate 0) show they sit in different regions
of a multimodal posterior: the full-parameter target is not merely
ill-conditioned, and no metric closes the gap. A Gauss–Newton (PSD) metric
would remove the negative-curvature entries but cannot remove the multimodality;
the full-parameter gate needs either a globally well-behaved parameterization
or a non-reversible kernel.

**Multimodality diagnostic (`examples/vector_basins.rs`).** Four independent
mode fits (`adam_fit_energy`, 2,000 steps, `init_std=0.02`) give potentials
`36.39 / 36.54 / 97.91 / 79.31` with test RMSE `0.0737 / 0.0696 / 0.0719 /
0.0797`. All six parameter pairs sit at relative distance `1.92–2.18` (≈ fully
uncorrelated vectors) — the basins are genuinely distinct solutions of
comparable predictive quality, so the sampler's ESS plateau is a property of
the target, not of the metric. Segment attribution is diffuse: the worst
segment differs per pair (`readout` 2.28/3.02, `block15` 2.11, `block10`
2.07/2.24, `block5` 2.14) — no single transformer block is responsible.

**The blocks were not all using attention (fixed 2026-11-23).** The diagnostic
now labels one segment per transformer block (the six matrix segments were
mislabeled as `block1…block24`, which is why an earlier reading pointed at
"layers 1–3"). Measured after the fix: every block's `wq`/`wk` gradient at the
legacy `init_std=0.02` is `7e-2…2e-1` against `4e1…7e1` for `wv`/`wo`/`w1`/`w2`,
and those projections stayed bit-identical across independent fits.

Root cause: `0.02` is 10× below the prior sd that the target itself assumes
(`prior_sigma·√fan_in = 0.2` for a 16×16 matrix). Logits are then ≈0, where
`∂logits/∂W_q ∝ W_k x` is *second order*; the prior term (`θ/c`, i.e. `0.5` vs
a data gradient of `0.1`) wins and pulls `W_q, W_k` into the fixed point
`W_q = W_k = 0`, where every logit is 0, the softmax is exactly uniform, and
the q/k gradient is exactly zero. The sampler and the optimizer were therefore
fitting a four-block model in which attention contributed a constant.

Fix: `init_flat_prior` draws θ at the prior's own sd (default `--start-scale
prior` here, `--init-mode prior` in the diagnostic; `legacy` reproduces the
recorded runs). Then `∂L/∂W_q` is first order — `3.1e3` vs `0.17` at width 16,
width 64 — and the projections move. The mode fit needs more steps from a
prior draw (it starts far from the prior mode): at 2,000 steps it reaches only
potentials `2.3e3–3.0e3` / RMSE `0.19–0.23`, at **8,000 steps** it reaches
`28.09 / 38.71` with RMSE `0.0770 / 0.0871`, i.e. comparable to the legacy
modes, and the four blocks now separate by `1.99–2.21`.

**Live attention does not fix the gate.** Same full-parameter HMC run as
recorded, with prior-scale init and the 8,000-step mode fit: R̂ `2.60278`, bulk
ESS `4.7`, tail ESS `9.3`, RMSE `0.173727` — still failing, slightly worse than
the legacy-init baseline. The dead projections were a real modelling defect but
not the cause of the ESS plateau.

**Parallel tempering (`run_parallel_tempering`, `--temper K`).** Replicas at
`β_k` sample `exp(−βU)` with a diagonal metric and swap adjacent states with the
exact ratio `min(1, exp((β_i−β_j)(U_i−U_j)))`; draws come from the cold replica
only, and a 3-D Gaussian unit test recovers the target marginal. On the full
target (4 replicas, ladder `0.02/0.0737/0.2714/1`, swaps every 5 steps, 400
steps, 4 chains) the cold chain is unchanged — R̂ `2.44665`, ESS `4.8`, RMSE
`0.168598` — and **0–1 of 240 swaps were accepted per chain**. The exchange
rate is set by `Δβ·ΔU`: adjacent replicas differ by `ΔU ≈ 6·10^3 − 10^4`
(`U_cold ≈ 4.8e3`, `U_β=0.27 ≈ 1.1e4`), so a rung spacing near `1e-4` would be
needed, i.e. tens of thousands of replicas. Parallel tempering at this energy
scale is therefore not a viable remedy without rescaling the potential.

**Hit-and-run slice moves (`run_hit_and_run`, `--kernel hitrun`).** A global
move that ignores gradient scale: a random direction, unit in prior-whitened
coordinates, with the exact conditional along that ray sampled by Neal's
stepping-out/shrinkage slice sampler. It needs no gradient; a 3-D Gaussian unit
test recovers the target covariance. On the full target (400 moves per chain,
burn 100, 4 chains, `--hr-width 1`) every move lands (`400/400`, mean step
`0.40` prior sd) at ≈2,440 potential evaluations per chain, i.e. the same cost
as the 400-step HMC runs. Result: R̂ `2.39717`, bulk ESS `4.9`, tail ESS `10.0`
— the gate is not closed. Predictively it is the best run so far (draw-mean
test RMSE `0.087048`), but its intervals *under*-cover (`0.3375` at 90%), which
is the honest signature of a chain that has not equilibrated: the move is exact
per step but 400 moves of `0.4` prior sd is a short walk.

**12.5× budget does not help (`--steps 5000`, 2026-11-23).** Same target, same
prior-scale init, same 8,000-step mode, 5,000 moves/chain (burn 1,250, thin 2,
1,875 draws/chain, ≈34–35k potential evaluations/chain, ≈44 min wall clock):
every move lands, but the mean step *rises* to `0.85–0.96` prior sd and the gate
is unchanged — R̂ `3.03445` (per-observable `1.75–3.03`), bulk ESS `4.5`, tail
ESS `10.0`, `gate=false`. ESS is flat despite 6.25× more draws in the pool
(`4.9` at 300 draws/chain → `4.5` at 1,875), so the chains are not
diffusing-and-mixing slowly; they are each confined to a region and the region
differs by chain. That is a geometry result (narrow, curved ridges), not a
budget result — more moves at this width would not close it.

The longer run also revises the 400-move picture: draw-mean test RMSE rises to
`0.121815` and interval coverage becomes credible
(predictive-90 `0.9333`, parameter-90 `0.8125`). The 0.087 RMSE and 0.34
coverage of the 400-move run were artefacts of a short, biased walk, so the
short run's apparently good point prediction should not be quoted as an
improvement.

**Conclusion for the full-parameter gate:** the target is multimodal with an
`O(10^4)` energy scale; preconditioning (pilot, low-rank, Hessian), replica
exchange and prior-scale initialization each fail for a different, now-measured
reason. Hit-and-run does not close the gate at 400 moves per chain and its own
intervals advertise the shortfall — and raising the budget 12.5× to 5,000 moves
per chain leaves ESS flat at `4.5`, which rules out "more moves" as the remedy
for this kernel width. What the evidence now points to is a target whose
high-density set is a set of narrow, curved ridges: exact line moves keep each
chain inside its own ridge, so the sampler needs either a direction distribution
that follows the ridge (a learned/local metric) or a parameterization in which
the modes are connected. Neither is implemented; the gate remains open.

**Continual-stream pilot (`examples/vector_stream.rs`).** The 240 training rows
are cut into four contiguous 60-row blocks (the fixture exposes no document
ids, so this is a row-order proxy). At each prefix the three arms share the
objective, optimizer settings and prior: `warm` continues Adam from the
previous prefix's full vector over all seen rows, `joint` restarts from the
fixed init on the same rows, `reset` restarts on the new block only. Each
prediction is the readout-conditional exact posterior mean with the trained
backbone frozen. Test RMSE by prefix (warm/joint/reset):
`0.150742/0.150742/0.150742` (identical at prefix 1), then
`0.106920/0.112642/0.183135`, `0.189312/0.118874/0.213221`,
`0.079085/0.105293/0.168040`; means `0.131515/0.121888/0.178784`. Retention on
earlier blocks separates the arms sharply at prefix 3: warm `0.00217/0.00337`,
joint `0.00429/0.00444`, reset `0.180/0.202`. So the reset arm forgets, joint
retains, and warm keeps the best retention but is not uniformly better on
test (worse at prefix 3, best at prefix 4): this is a measured pilot with
prefix-to-prefix variation, not a promotion result.

The pilot also reports 90% predictive intervals and normalization diagnostics.
Coverage is *conservative to the point of being uninformative*: `0.983–1.000`
for every arm at every prefix. The diagnostic that shows why is the normalized
residual `z = r/√(var + s²)`:

| arm | coverage90 by prefix | mean z² by prefix | calib (50/80/90/95%) |
|---|---|---|---|
| warm | 1.000/1.000/0.992/0.983 | 0.032/0.024/0.275/0.356 | 0.938/0.990/0.994/0.994 |
| joint | 1.000/1.000/1.000/1.000 | 0.032/0.020/0.050/0.068 | 0.990/0.999/1.000/1.000 |
| reset | 1.000/1.000/0.988/1.000 | 0.032/0.107/0.164/0.055 | 0.954/0.996/0.997/1.000 |

A calibrated Gaussian predictive has mean `z² = 1`; these are `0.02–0.36`, i.e.
the predictive sd is up to 7× too large, and a nominal 50% interval covers
94–99%. The interval is therefore honest only as a *conservative* bound. Its
width is dominated by the bias prior (sd `0.05` → ±`0.05·z`), which is the same
order as the residual sd `0.07–0.15`, so the model's uncertainty about the
readout is not the quantity the intervals are dominated by. Reporting coverage
alone would have hidden this; the calibration curve and `z²` are now the
reported quantities.

**Conclusion.** The measured blocker was the transition kernel's
high-dimensional scaling, not the regression objective, the teacher, or the
run length.  The exact-covariance posterior-referenced CN raises acceptance
but its random-walk form saturates near `(1−ρ)/(1+ρ)` in ESS/step: it fails the
ESS gate at 3,000 steps and passes at 6,000 (4,500 draws per chain).
Gradient-based HMC with the exact posterior metric passes the same gate with
750 draws per chain, a ~6× smaller budget.  On the full 57,378-coordinate
target a corrected diagonal metric adapts `ε≈0.10` but reaches only bulk ESS
4.7, and a low-rank-plus-diagonal metric reaches ESS 5.3 at rank 32 over 1,000
steps, and a Hessian-diagonal metric at the mode (48% of its entries fall below
the prior bound, so the mode is nonconvex) also stalls at ESS 4.8; the full
posterior remains open. The
continual-stream pilot is measured (reset forgets, joint retains, warm keeps
the best retention) but remains a pilot, not a promotion result.

**Live/fixture parity is implemented but unverified here.**
`scripts/pcnl_ultrabert_adapter.py` now shares the pinned revision, 96-token
truncation, CLS pooling and L2 normalization between both generators and
`scripts/pcnl-ultrabert-parity`.  The parity check exits `2` with an explicit
`SKIP: live encoder unavailable` when torch/transformers/safetensors or pinned
weights are absent; this environment lacks them, so no parity pass is claimed.

## R12 — historical signed-state application for English text (updated 2026-11-20)

Given English text and a selected draw of the decision head, `predict` returns a
**normalized real wave-function** in an ordered outcome basis. A second decision
model can act directly on this state. `measure` reads out named probabilities
$p_k=\psi_k^2$ in that basis. Every categorical distribution has a real unit-vector
representative, so a single measurement retains the existing decision semantics.
Relative signs carry information that a later operation can reveal. `update` learns
the head from measured outcomes using Bayes' rule and pCNL. Keep one fixed basis
per model, starting with the existing four-class fixtures.

**Required head depth.** Each delivered decision model must have at least four
trainable hidden layers between the frozen embedding and the angle readout.
Start with `--hidden 32,16,16,8`. For four outcomes this gives
`768 → 32 → 16 → 16 → 8 → 6 rotation angles`. The four hidden layers use
tanh activations. pCNL jointly samples their weights and biases together with
the angle readout. Count depth within one application of the decision head.

### R12 delivery order at the time of the signed-state experiments

The rotation implementation already prepares a real unit state, composes a sequence
within each parameter draw, and evaluates the full-prefix Born likelihood. Its
fixture experiment calls `RotationSession::update` once per fitted arm. The
published scores therefore answer a batch-fit question. At matched width on raw
AG News embeddings the signed head had test NLL 1.6467 and the tuned softmax
head 0.4691. The sequence probe also left the interference advantage unresolved.
Use the following checkpoints before another width, step-size or prior sweep.

1. **Deliver English in, signed state out.** Share the encoding function between
   the two fixture scripts and the `rotation` CLI. Pin model and tokenizer to
   `dmis-lab/UltraBERT-Embed-150M` revision
   `b06b36724c5417a1ec1b7863dd4fa542b74e4899` with the existing 96-token,
   eval-mode, CLS and L2 conventions. Today `predict --text` supplies descriptive
   text alongside an embedding, and `update` reads fixture rows. Add raw-text
   prediction and labelled event ingestion through the same adapter. Persist its
   fingerprint and the ordered outcome names. The acceptance transcript starts
   with an English passage, returns a reproducible draw's signed unit vector and
   named squared amplitudes, records a subsequently revealed label by event ID,
   reopens the checkpoint and predicts again. Verify live/fixture embedding parity
   on the same passage and prediction parity at the same checkpoint and draw.
   Expose the ensemble mean of squared amplitudes as a separate readout.
   Set the new-session default to the four-hidden-layer configuration. Until
   that change lands, pass the explicit width list to the CLI and experiments.

2. **Make a checkpoint identify one posterior.** Fix the prior and unit likelihood
   weight before event one. For the implemented layer scales use
   $C_{ii}=\sigma^2 s_i^2$ with $s_i=\sqrt{\mathrm{fan\_in}}$ for matrix entries
   and $s_i=1$ for biases. Draw the round-zero predictive parameters from that
   prior. Currently `ensure_retained` uses the narrower chain initialization.
   Keep a committed prefix visible until sampling its successor succeeds. Today
   `append_event` clears draws and a prediction before `update` can regenerate
   working-scale draws tagged with the new prefix. A duplicate-only CLI update
   also resamples. Test that identical retries preserve the published draws and
   RNG counters, conflicting IDs fail, and an interrupted update reopens to a
   complete prior or successor checkpoint. Carry forward chain states as starting
   points and include each distinct event once in the full-prefix likelihood.

3. **Verify which pCNL kernel and draws reach the user.** The specified pCNL
   proposal in centered coordinates has mean
   $m+\sqrt{1-\beta^2}(\theta-m)-\tfrac12\beta^2 C\nabla U(\theta)$, where
   $U$ is the summed Born NLL. `mh_pcn_centered_step` currently includes the
   Gaussian prior gradient in the drift as well as the shrinkage. The full MH
   ratio can correct either proposal when evaluated consistently. Align the
   reference arm with the specified likelihood drift, record the kernel version,
   and check the forward/reverse ratio against direct Gaussian densities at the
   stored endpoints. At constant likelihood the reference proposal must preserve
   the prior and accept to numerical precision. Use one-angle Born quadrature
   to check single and repeated applications, including predictive MCSE.

   Retain draws evenly across independently initialized chains with chain and step
   IDs. `update` currently concatenates by chain then truncates. The reported
   250-step, 50-burn, ten-draw runs can therefore represent only the first chain
   in the readout. Recheck affected comparisons with predictive ESS, split R-hat
   and MCSE for outcome probabilities and composed measurements. Acceptance counts
   alone cannot resolve mixing. Keep the original nGPT mini-batch schedule as a
   later comparator. Its gradient bounds require separate analysis for Born NLL
   near an amplitude zero.

4. **Demonstrate continual learning on a stream.** Use each existing corpus as
   development evidence. Split by document ID before scheduling four prefix
   updates with fresh observations, including a return to a previously seen class.
   Save each incoming prediction before revealing the label. At every prefix
   compare resumed pCNL with an independently initialized joint pCNL run using the
   identical prior and event IDs, a current-batch reset, and the tuned point head.
   Record paired prequential NLL and Brier, old-class forgetting, update time
   and storage. Log checkpoint, draw provenance and prediction uncertainty.
   Freeze numerical agreement margins, the minimum learning gain, compute budget
   and stream orders on development data. Use fresh held-out documents for a
   confirmatory claim because the fixture test scores have informed this plan.
   Require joint-reference agreement and a gain over reset as specified below.
   Price any bounded-history approximation against this full-history run.

5. **Recover single-decision quality and test composition separately.** As one
   controlled architecture experiment, initialize a composable operator from the
   tuned categorical model. Let $a=\sqrt{\mathrm{softmax}(f(e))}$ componentwise,
   $v=(b_0+a)/\|b_0+a\|$, and $Q=2vv^\mathsf{T}-I$. Then $Qb_0=a$ and
   $Q^\mathsf{T}Q=I$. A signed residual rotation gives $U(e)=R_\eta(e)Q(e)$.
   At zero residual angles verify per-example equality to the softmax scores
   before fitting. Use the point fit as initialization only, keep the declared
   prior fixed, and compare the zero-residual control, learned residual and
   existing angle-only head at matched budgets. Report the point fit and sampled
   predictive separately to locate where a score gap opens.

   For distinct models, separate basis identity from the head name and carry
   the model/checkpoint/draw for each operator. Test a noncommuting A-then-B
   sequence and two uses of A with shared weights. Train and score both the
   coherent and intermediate-measurement models on the same sequence events.
   The existing classical readout uses the single-step ablation's draws. Also
   compare both readouts of the same fitted operator to isolate the effect of
   measurement. Evaluate the sign oracle below and held-out English sequences
   before claiming a learned advantage.

**Depth acceptance.** Run the continual and composed-sequence evaluations on the
four-hidden-layer head and record its widths in each report. Use the same hidden
widths for the matched softmax control. On a small nondegenerate Born example,
check a likelihood gradient in each hidden block against finite differences.
Verify that the reopened checkpoint restores the four widths and reproduces
the signed state for a fixed draw. Earlier shallow runs provide comparison data.

**Completion evidence.** Archive the raw-English restart transcript, known-target
kernel tests, per-prefix paired report and held-out composition results. Report
functional delivery, evidence of useful continual learning and evidence of learned
interference separately. The measured fixture record below establishes the starting
point for this work.

### Original milestones and measured fixture record

**Starting point.** R10 supplies the frozen UltraBERT features, the
`768 → 32 → 16 → K` head, categorical NLL and per-draw softmax. R11 supplies the
continual comparison and predictive averaging. Its diagonal carry with zero
replay lost 90.56 percentage points on the canonical AG News forgetting measure.
In the symmetric 20 Newsgroups comparison, mean calibrated NLL was 0.3064 for
the joint fit and 0.4683 for the sequential fit. This motivates measuring the
distance to a joint fit at **every prefix**. The continual example consumes a
labelled tensor fixture, uses drift-free MH-pCN on a Brier target, and fits a
Gaussian to the draws. Class revisits select the same training rows again.
Replay appends a fresh subset each round, so its per-class setting limits each
addition and storage can grow across visits.

**Work order.** The [original nGPT plan](../docs/plan_bayes_ngpt.md) supplies the
SG-pCNL derivation and schedule. The [development plan](DEVELOPMENT_PLAN.md)
supplies target and diagnostic checks. The [Edge0/RLCD plan](EDGE0_RLCD_PLAN.md)
specifies the R9 reinforcement experiment. The following milestones use the
existing text head's hidden layers and can proceed independently of those architecture experiments.

1. **Prepare, compose and measure.** Extract the encoder invocation in
   `scripts/pcnl-ultrabert-embed` into a shared fixture and live-text adapter.
   Keep the R10 model and tokenizer revision, eval mode, 96-token truncation, CLS
   pooling, L2 normalization and 768-dimensional output. Use the head's tanh layers
   to map the embedding $e$ to angles for a fixed ordered product of real plane
   rotations $U_\theta(e)$. For the first $K$-outcome model use one angle per
   coordinate pair ($K(K-1)/2$, six for the fixtures). Record pair order, rotation
   convention and basis labels in the model. This parametrization preserves norm
   for any input state and supplies signed amplitudes without a division by a
   possibly zero output norm. The existing softmax head remains a baseline.

   | Operation | Contract for one posterior draw |
   |---|---|
   | `predict(text, draw)` | Prepare $\psi=U_\theta(e)b_0$ from the declared first unit basis vector $b_0$. Return signed amplitudes, basis ID, checkpoint and draw ID. |
   | `apply(text, state, draw)` | Return $U_\theta(e)\psi$ in the declared output basis. Applying A then B gives $U_B(e_B)U_A(e_A)b_0$. Use the same rule for repeated applications of A. |
   | `tensor(a,b)` | For separately prepared registers return $a\otimes b$ in the ordered pair basis, with amplitude $a_i b_j$ at $(i,j)$. |
   | `measure(state)` | Return the squares of the amplitudes. A workflow that observes an intermediate outcome $j$ continues from its basis vector $b_j$ for a projective measurement in this basis. |

   Sequential multiplication requires matching dimension **and ordered basis**.
   Require an explicit basis map when label orders differ. A prepared vector
   specifies only one column of its operator, so expose the action on arbitrary
   unit states through `apply`. Test embedding parity with the fixture encoder,
   finite signed output, unit norm after a sequence, identity and product order,
   tensor indexing and marginals, and invariance of all measured predictions
   to an overall sign. Reject incompatible bases, invalid states and empty text.

   **Posterior contract.** Select a retained draw (a prior draw at round zero)
   for a pure-state prediction. Compose the whole sequence within that draw,
   then square and average over draws for the Bayesian predictive probabilities.
   Reuse the same parameters for repeated calls to the same uncertain head in
   one sequence. Pair distinct heads by joint posterior draw when trained jointly;
   sample them independently only under an explicit factorized posterior.
   The ensemble describes a mixture $\rho=S^{-1}\sum_s\psi_s\psi_s^\mathsf T$.
   Its diagonal gives the measured predictive. A normalized mean amplitude or
   square root of that diagonal generally changes subsequent composed predictions.
   Keep the draw ensemble available to callers that need further composition.

2. **Count each labelled observation once.** Start with the fixed prior
   $\pi_0=N(0,0.5^2 I)$. Give each event a stable ID, embedding and observed
   class. Store the preparation, ordered text/model applications and measurement
   basis for each observed outcome. A single-step label uses
   $p_\theta(y_i\mid e_i)=[U_\theta(e_i)b_0]_{y_i}^2$. For a sequence, $e_i$
   below denotes the recorded inputs and the likelihood uses the final composed
   state's squared amplitude. For a new batch $D_t$ the update is

   $$
   \pi_t(\theta) \propto \pi_{t-1}(\theta)\prod_{i\in D_t}p_\theta(y_i\mid e_i)
   \propto \pi_0(\theta)\prod_{i\in D_{1:t}}p_\theta(y_i\mid e_i).
   $$

   Implement the right-hand side first. Append new events to a durable evidence
   store, initialize chains from the preceding checkpoint, and evaluate the **sum
   of unweighted Born NLL** over the unique prefix using a new `Potential` for
   the rotation head. Keep the per-event loss $-2\log|\psi_{y_i}|$ and its
   gradient consistent. An exact zero gives zero likelihood. Start chains at
   finite-density draws and reject proposals with zero target density. Test
   near-zero amplitudes and finite-difference gradients away from nodes. This reference incurs storage and
   per-step evaluation cost growing with history. An identical retry is a no-op,
   conflicting reuse of an ID is an error, and distinct observations may share
   text. Verify at fixed weights that partitioning and permuting the same events
   preserve the log target and that the negative-log increment equals the new
   batch's summed NLL.

3. **Sample the full head with pCNL.** Adapt the full-batch adjusted path in
   `src/task.rs` (`MhOptions::exact`, `pcn_proposal_log_ratio`) to the text
   `Potential` interface. With $U$ the prefix NLL sum, use Gaussian proposal mean
   $\sqrt{1-\beta^2}\,\theta-(\beta^2\sigma^2/2)\nabla U(\theta)$ and covariance
   $\beta^2\sigma^2 I$. Include the full log posterior difference and reverse-minus-
   forward proposal log density in the MH ratio. Freeze tuning after warmup.
   Check the drift by finite differences, the ratio against direct log densities
   and posterior moments on the linear-Gaussian reference. Check per-class
   single-step and composed predictions on a one-angle head against quadrature. Keep
   R11's drift-free MH-pCN as a same-target control. Evaluate unadjusted mini-batch
   SG-pCNL after quantifying its predictive bias and cost against the adjusted run.

4. **Deliver one resumable loop.** Expose initialization, raw-text prediction
   and labelled batch update through a library and thin CLI. Commit the evidence
   prefix and matching sampler checkpoint as one version. Persist encoder identity,
   basis, rotation layout, linked model versions, target settings, event IDs, chain and RNG states and
   retained predictive draws. Reuse `src/persist.rs` for trajectory tensors. The
   end-to-end test predicts before revealing a label, updates, reopens the model,
   and reproduces signed amplitudes for a fixed draw, composed measurement probabilities
   and the next seeded update. Exercise a
   duplicate delivery and an interrupted commit at the update boundary.

5. **Measure learning and retention.** On AG News and the four-class 20 Newsgroups
   fixture, reserve disjoint fit, calibration and final evaluation IDs. Partition
   fit IDs into fresh batches for mixed-class, class-disjoint and class-revisit
   streams. Score each incoming batch **before** revealing labels, then measure
   old- and new-class performance on fixed probes. Compare carried chains with
   a fresh joint NLL fit at each identical prefix, a current-batch-only reset and
   a tuned point head. Keep the R11 Brier/Gaussian replay arm as a labelled
   generalized-Bayes comparison. Report raw predictive scores first. Fit any
   optional temperature using only calibration labels available by that round.
   Pair arms by stream and seed. Report prequential NLL and Brier, per-class
   forgetting, reliability and decision cost under a prespecified utility table.
   Record update time, gradient evaluations, history bytes, predictive Monte
   Carlo standard errors, ESS and split R-hat.

   Before final evaluation, fix budgets, seeds, predictive agreement tolerance
   and minimum meaningful proper-score gain on development data. Require
   agreement with the independent joint reference within tolerance accounting for
   Monte Carlo error, plus a paired prequential gain over reset whose confidence
   interval clears the declared minimum, before claiming useful carry. Apply the
   mixing gates in “Milestones, budgets and stop rules”. Report failed precision
   as inconclusive and publish old-class losses even when aggregate scores improve.

6. **Verify what composition adds.** Single-step text labels constrain the squared
   entries of the prepared column. They leave its relative signs and the action on
   other input states underdetermined. Keep AG News and 20 Newsgroups as single-step
   checks. Include the positive square root of the softmax baseline as a control
   whose one-step Born probabilities agree with that baseline by construction.

   Use a two-outcome oracle. Prepare $(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$ from two
   text inputs. Both measure as $(1/2,1/2)$. Apply the same rotation by $\pi/4$
   to each with the convention $R(a)_{1,:}=(\cos a,-\sin a)$. The resulting
   probabilities are $(0,1)$ and $(1,0)$. Measuring and discarding the intermediate
   outcome instead gives $(1/2,1/2)$ in both cases. Also verify that applying
   $R(\pi/4)$ twice to $(1,0)$ yields final probabilities $(0,1)$.

   Then hide the oracle angles and stream labelled **sequences** with final
   outcomes generated from the composed Born law. Include different heads, shared
   heads applied twice, and basis-state preparations that exercise the other
   columns. Hold out text instances and sequences before fitting. Compare held-out
   sequence NLL and Brier against the oracle and a classical transition baseline
   propagating squared matrix entries at each stage. Include a single-step-only
   training ablation. Evaluate posterior predictive recovery on held-out sequences
   with the precision gates above. Claim learned interference only when composed
   evidence resolves the difference from the intermediate-measurement control.

**Bounded-history follow-up.** After establishing the reference, test a fixed
storage budget. Use the carried diagonal or low-rank summary with fresh evidence
only. For a replay variant reserve a capped set of unique exemplars, summarize
only evidence outside that set, and move an evicted exemplar into the summary
once. Track which factor accounts for each observation in the audit experiment.
Price the approximation against the same-prefix reference in predictive error,
forgetting, bytes and latency as history grows. Promote a bounded variant only
within prespecified quality and resource budgets.

**Delivery gate.** R12 is complete when the English predict/update/restart test
and target/kernel checks pass, the signed composition oracle passes, and a paired
report records the single-step and learned-sequence results with their uncertainty. Publish commands, manifests, budgets and results in
[PCNL_STATUS.md](../docs/PCNL_STATUS.md). R10–R11 below contain the measured
baseline. The current delivery order above tracks outstanding work.

**Status (2026-11-20).** The reference core for milestones 1–4 is implemented
in `src/rotation_head.rs` and `src/bin/rotation.rs`.
It provides the ordered plane-rotation head, signed `prepare`/`apply`/`tensor`
states, projective `measure`, basis and draw checks, the append-only evidence
prefix, the unweighted Born-NLL `Potential`, and `mh_pcn_full_step` with the
full posterior and proposal-density ratio.  `RotationSession` stores the
{dimension, basis, layout, model, evidence, target settings, chain/RNG state}
checkpoint atomically, ignores a stale `.tmp`, and can warm-start an explicit
chain start; `prior_theta` supplies a deterministic working-scale draw;
`predict_sequence` composes the whole sequence per draw before averaging
squared amplitudes.  The composition
oracle, sign invariance, tensor marginals, identity/product order, evidence
idempotence and conflict checks, finite-difference gradients, checkpoint
reopen, the normalized and centered-prior anisotropic Gaussians, the anchor
RMS scale estimate, cross-chain draw retention/provenance, and a full-space
Gaussian posterior canary pass plus independent-start determinism (21 focused
tests).  The CLI restart is covered by `pcnl/tests/rotation_cli.rs`.  The full
library suite passes 233/233 in 286.5 s
(`cargo test --offline -p pcnl --lib`); the two `dt_subspace` basis tests now
use bounded path lengths instead of running for over ten minutes.

`examples/rotation_experiment.rs` records the milestone 5–6 measurements.
The raw-weight `N(0,0.5²I)` prior was replaced by a normalized per-layer prior:
a weight matrix with fan-in `n` is divided by `√n` before the Gaussian penalty
(equivalently, its prior variance is `σ²n` in raw coordinates), while biases
are unscaled.  The pCN proposal covariance is diagonal in the same coordinates,
so the kernel remains exact (tests cover an anisotropic Gaussian).  Chain
initialisation intentionally uses the working scale `σ` rather than the much
wider raw prior draw, which would saturate a 768-wide input layer.  The
measured probe still uses the smallest tanh head (`hidden=[1]`), 20 fit rows
per class, 10 calibration rows per class, Adam pretraining, beta=0.01, 250
steps, 50 burn, 2 chains and 10 retained draws.  Five seeds
(20261118–20261122) give (mean ± sd):

| fixture | arm | acc % | NLL | Brier | ECE |
|---|---|---|---|---|---|
| AG News | raw Born | 46.3 ± 13.2 | 1.812 ± 0.489 | 0.788 ± 0.189 | 0.269 ± 0.103 |
| AG News | temperature | 46.3 ± 13.2 | 1.221 ± 0.166 | 0.658 ± 0.094 | 0.070 ± 0.046 |
| AG News | softmax control | 48.2 ± 3.2 | 2.291 ± 0.540 | 0.732 ± 0.094 | 0.238 ± 0.109 |
| 20NG | raw Born | 44.2 ± 16.6 | 1.853 ± 0.832 | 0.796 ± 0.295 | 0.316 ± 0.164 |
| 20NG | temperature | 44.2 ± 16.6 | 1.177 ± 0.254 | 0.627 ± 0.139 | 0.109 ± 0.032 |
| 20NG | softmax control | 55.7 ± 5.8 | 1.578 ± 0.317 | 0.593 ± 0.045 | 0.149 ± 0.031 |

The temperature-scaling grid covers [0.04, 8]; temperature scaling reduces NLL
and ECE but cannot change accuracy (it is monotone).  Against the superseded
raw-weight prior, AG NLL improves from 1.995 ± 0.509 to 1.812 ± 0.489 and AG
accuracy is flat (45.8 → 46.3 %); 20NG accuracy falls (47.8 → 44.2 %) with NLL
flat (1.828 → 1.853).  Normalising the prior therefore does not rescue the
single-step arm.

**Empirical-Bayes anchor prior.**  The center is the Adam-pretrained head and
the scalar scale is the anchor's RMS in the same normalized coordinates
(`sigma = rms(theta / prior_scales)`); the measured values were 0.125–0.244.
The centered pCN kernel is exact and is covered by the shifted-Gaussian test.
Five seeds give:

| fixture | arm | acc % | NLL | Brier | ECE |
|---|---|---|---|---|---|
| AG News | anchor raw | 42.3 ± 10.3 | 1.796 ± 0.389 | 0.806 ± 0.156 | 0.297 ± 0.079 |
| AG News | anchor temperature | 42.3 ± 10.3 | 1.231 ± 0.148 | 0.661 ± 0.086 | 0.057 ± 0.030 |
| 20NG | anchor raw | 43.0 ± 15.9 | 1.726 ± 0.531 | 0.794 ± 0.232 | 0.302 ± 0.098 |
| 20NG | anchor temperature | 43.0 ± 15.9 | 1.195 ± 0.246 | 0.637 ± 0.134 | 0.113 ± 0.026 |

This lowers NLL against the zero-centered normalized prior (AG 1.812 → 1.796;
20NG 1.853 → 1.726) but not accuracy (AG 46.3 → 42.3 %; 20NG 44.2 → 43.0 %).
The chains accepted 336–496 of 500 proposals in aggregate. Per-chain
predictive precision remains to be checked.  The softmax control still wins accuracy on both fixtures and NLL on
20NG (1.578 vs 1.726); on AG the anchor arm has lower NLL (1.796 vs 2.291) but
lower accuracy (42.3 vs 48.2 %).

On the learned-sequence arm (length 2, 60 train / 24 test sequences, 800 Adam
steps, 200 sampler steps, seeds 20261118–20261120) the oracle scores
61.1 ± 6.4 % on AG and 70.8 ± 8.3 % on 20NG.  Learned composition scores
23.6 ± 8.7 % on AG and 38.9 ± 12.7 % on 20NG; the single-step ablation scores
30.6 ± 4.8 % and 30.6 ± 9.6 %; the classical squared-transition baseline scores
30.6 ± 6.4 % and 33.3 ± 16.7 %.  The AG arm is worse than both controls.  The
20NG arm is above its ablation but its per-seed differences (+25.0, 0.0,
0.0 points) and the classical baseline variance do not establish learned
interference.  Under the anchor prior the learned arm falls to 19.4 ± 10.5 % on
AG and 30.6 ± 8.7 % on 20NG, while the single-step ablation scores 25.0 ± 15.0 %
and 43.1 ± 31.3 % and the classical baseline 27.8 ± 19.2 % and 44.4 ± 33.7 %.
The anchor prior therefore makes the sequence result worse, not better.  These
are controlled fixture probes, not benchmarks, and no text-accuracy or
learned-interference claim follows.

Reproduce:

```bash
cargo test --offline -p pcnl --lib rotation_head
cargo test --offline -p pcnl --test rotation_cli
cargo run --offline --release -p pcnl --example rotation_experiment -- \
  --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --mode single \
  --fit-per-class 20 --cal-per-class 10 --hidden 1 --steps 250 --burn 50 \
  --chains 2 --draws 10 --beta 0.01 --pretrain 1000 --softmax-pretrain 3000
cargo run --offline --release -p pcnl --example rotation_experiment -- \
  --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors --mode sequence \
  --seq-train 60 --seq-test 24 --seq-len 2 --hidden 1 --steps 200 --burn 40 \
  --chains 2 --draws 10 --beta 0.01 --pretrain 800
```

Append `--anchor-prior` to either command for the empirical-Bayes anchor arm;
omit it for the zero-centered normalized-prior arm.

The width/representation grid is the same single-step command with the
`--seed`, `--hidden` and `--representation` axes.  Select a width by validation
NLL, never by the `rotation raw` test line:

```bash
for fixture in ultrabert_agnews_embeddings ultrabert_20ng4_embeddings; do
  for rep in raw standardized; do
    for seed in 20261118 20261119 20261120 20261121 20261122; do
      for h in 1 4 8 16; do
        cargo run --offline --release -p pcnl --example rotation_experiment -- \
          --fixture "docs/fixtures/$fixture.safetensors" --mode single \
          --seed "$seed" --fit-per-class 20 --cal-per-class 20 \
          --hidden "$h" --representation "$rep" \
          --steps 250 --burn 50 --chains 2 --draws 10 --beta 0.01 \
          --pretrain 1000 --softmax-pretrain 3000
      done
    done
  done
done
```

The fixture probe delivered an uncertainty-bearing batch measurement. Raw-English
delivery and the multi-prefix learning and composition gates remain open.
Normalizing the prior removes the layer-width domination but does not fix the
arm, and the empirical-Bayes anchor prior lowers NLL without improving
accuracy or sequence composition.  The evidence points away from the prior
center/scale as the binding issue.

**Held-out width/representation sweep (2026-11-20).**  The follow-up is now
measured.  The grid is four widths (`hidden=[1]`, `[4]`, `[8]`, `[16]`), two
input representations (`raw` and fit-standardized embeddings), two fixtures and
five paired seeds (20261118–20261122).  The fit/test split is unchanged; each
cell uses 20 fit rows per class, 20 validation rows per class excluded from the
likelihood, and the 30-row-per-class test split.  Standardization centres and scales each embedding
coordinate using fit rows only.  The sampler settings are the probe's
(`beta=0.01`, 250 steps, 50 burn, 2 chains, 10 draws, Adam pretrain 1,000 and
softmax pretrain 3,000).  A width is selected by **mean validation NLL across
seeds**; test metrics are not used for selection.  The softmax control is the
same MLP widths trained by NLL.

| fixture | representation | arm | width | test acc % | test NLL | test Brier | test ECE | validation NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AG News | raw | signed rotation | 8 | 29.83 ± 5.38 | 1.6467 ± 0.0998 | 0.8409 ± 0.0441 | 0.2298 ± 0.0440 | 1.6598 ± 0.1296 |
| AG News | raw | softmax control | 16 | 84.67 ± 1.27 | 0.7317 ± 0.1540 | 0.2730 ± 0.0274 | 0.1293 ± 0.0247 | 0.6844 ± 0.1880 |
| AG News | standardized | signed rotation | 1 | 30.67 ± 8.44 | 1.4774 ± 0.0804 | 0.7682 ± 0.0449 | 0.0642 ± 0.0483 | 1.4290 ± 0.0728 |
| AG News | standardized | softmax control | 16 | 81.67 ± 1.56 | 1.0528 ± 0.2494 | 0.3240 ± 0.0276 | 0.1460 ± 0.0111 | 1.1044 ± 0.1276 |
| 20NG | raw | signed rotation | 4 | 38.00 ± 9.44 | 1.5141 ± 0.0997 | 0.7713 ± 0.0488 | 0.1799 ± 0.0619 | 1.5577 ± 0.1430 |
| 20NG | raw | softmax control | 16 | 89.83 ± 1.60 | 0.4284 ± 0.0632 | 0.1648 ± 0.0242 | 0.0691 ± 0.0138 | 0.4766 ± 0.1161 |
| 20NG | standardized | signed rotation | 1 | 34.00 ± 6.75 | 1.4317 ± 0.0446 | 0.7435 ± 0.0307 | 0.0602 ± 0.0330 | 1.4128 ± 0.0692 |
| 20NG | standardized | softmax control | 8 | 87.83 ± 2.92 | 0.6504 ± 0.0824 | 0.2025 ± 0.0378 | 0.0866 ± 0.0184 | 0.6131 ± 0.1499 |

Raw rotation accepted 381/500 proposals on AG at width 8 and 382/500 on 20NG at
width 4. These counts cover proposals from both chains.  Standardized rotation accepted only
9/500 and 4/500: its rows are a near-frozen chain at the Adam start, not a
sampled posterior, and are not eligible for selection.  Width selection is also
unstable across seeds: per-seed validation minima chose widths `{4:2, 8:2, 1:1}`
for AG raw rotation and `{4:2, 1:2, 8:1}` for 20NG; selecting per seed instead of
by the across-seed mean gives 39.0 ± 13.7 % / NLL 1.547 ± 0.270 on AG and
45.0 ± 13.6 % / 1.363 ± 0.196 on 20NG.  The softmax selected widths were more
stable (`{16:3, 8:2}` on AG, `{16:4, 8:1}` on 20NG).

The signed arm does not become competitive with width.  Its selected NLL is
1.647 on AG and 1.514 on 20NG; the softmax control is 0.732 and 0.428.  The
per-seed selected signed test accuracy is 39.0 % (AG) and 45.0 % (20NG), against
84.8 % and 90.0 % for softmax.  Standardization helps the softmax control
(paired raw-minus-standardized `dNLL` −0.322 ± 0.145 on AG and −0.263 ± 0.115 on
20NG; `dBrier` −0.065 ± 0.034 and −0.045 ± 0.048), but it does not make the
rotation arm competitive and its rotation chains fail the acceptance gate.  No
width or representation cell tested rescues the signed head's proper scores.
This is a negative result for this rotation parametrization at this sampler
budget, not a claim about signed states in general.  The follow-up below tests
centering, a representation-specific step size, and a tuned matched-width
softmax baseline; adding more widths is still not supported by the flat
validation surface.

**Step-size, centering and tuned-baseline follow-up (2026-11-20).**  The
follow-up isolates three explanations left open by the sweep: fit-centering
without variance scaling, a representation-specific pCN step, and an
approximately parameter-matched softmax baseline tuned on validation.  All runs
keep the same split, five paired seeds and the same sampler budget.

*Centering.*  The `centered` arm subtracts the fit-row mean but does not divide
by the standard deviation.  Its selected widths are the same as raw (AG width
8, 20NG width 4).  It mixes at the selected widths, but it does not improve the
proper scores:

| fixture | representation | width | accepted/500 | validation NLL | test NLL | test acc % |
|---|---|---:|---:|---:|---:|---:|
| AG News | raw | 8 | 381.2 ± 35.1 | 1.6598 ± 0.1296 | 1.6467 ± 0.0998 | 29.83 ± 5.38 |
| AG News | centered | 8 | 394.6 ± 39.9 | 1.6596 ± 0.1233 | 1.6988 ± 0.1118 | 26.33 ± 4.84 |
| 20NG | raw | 4 | 382.4 ± 29.5 | 1.5577 ± 0.1430 | 1.5141 ± 0.0997 | 38.00 ± 9.44 |
| 20NG | centered | 4 | 394.2 ± 12.3 | 1.5991 ± 0.1525 | 1.4999 ± 0.1287 | 36.50 ± 5.60 |

The AG validation NLL is indistinguishable from raw; 20NG is worse.  Centering
changes the acceptance rate, not the ranking.  A centered chain is not evidence
that the angle map has become competitive.

*Representation-specific step size.*  At width 1, `beta=0.01` was too large for
both representations, and especially for standardized inputs:

| fixture | representation | beta | accepted/500 | validation NLL |
|---|---|---:|---:|---:|
| AG News | raw | 0.005 | 365.0 ± 76.8 | 1.7646 ± 0.2979 |
| AG News | raw | 0.010 | 72.6 ± 91.5 | 1.9373 ± 0.4057 |
| AG News | standardized | 0.0025 | 262.0 ± 64.3 | 1.4299 ± 0.0730 |
| AG News | standardized | 0.005 | 72.6 ± 61.2 | 1.4486 ± 0.0550 |
| AG News | standardized | 0.010 | 9.0 ± 13.6 | 1.4290 ± 0.0728 |
| 20NG | raw | 0.005 | 367.6 ± 69.5 | 1.6900 ± 0.3943 |
| 20NG | raw | 0.010 | 82.4 ± 103.4 | 1.7746 ± 0.7063 |
| 20NG | standardized | 0.0025 | 248.4 ± 48.2 | 1.3590 ± 0.0464 |
| 20NG | standardized | 0.005 | 45.0 ± 24.5 | 1.3860 ± 0.0612 |
| 20NG | standardized | 0.010 | 4.0 ± 8.9 | 1.4128 ± 0.0692 |

So the earlier “near-frozen” label for standardized width 1 describes the
`beta=0.01` cell, not the representation.  At `beta=0.0025` the chains accept
about 248–262/500 proposals and the measured NLL improves.  Lowering beta
also partly unfreezes standardized widths 4 and 8 (about 57–60 and 31–34
accepted), but their validation NLL remains 2.05–2.44, far worse than the
width-1 value of 1.36–1.43.  The width failure is therefore not only a step-size
failure; the wider standardized parameterisation still fits worse.

The best standardized rotation cell is width 1 at `beta=0.0025`: AG test
28.50 ± 6.60 % / NLL 1.4275 ± 0.0339, 20NG 33.66 ± 8.01 % / 1.3445 ± 0.0700.
That is a better NLL than the best raw selected-width cell (AG width 8 at
`beta=0.01`, 29.83 ± 5.38 % / 1.6467 ± 0.0998; 20NG width 4 at `beta=0.005`,
38.33 ± 4.57 % / 1.5270 ± 0.1396), but lower accuracy in both cases.  It is a
calibration improvement, not an accuracy rescue.

*Tuned matched-width softmax.*  `--softmax-tune` selects L2 weight decay from
`{0, 1e-4, 1e-3}` and early-stops on calibration NLL, evaluating every 100
steps with patience 4.  The grid selected a nonzero penalty in 110/120 cells
(67 at `1e-3`, 43 at `1e-4`) and early-stopped between 100 and 3,000 steps
(83/120 at or below 600).  The output prints both flat parameter counts:
781/777 at width 1, 3,106/3,096 at width 4, 6,206/6,188 at width 8, and
12,406/12,372 at width 16, i.e. within 0.5 % of each other rather than exactly
equal.  At the selected widths:

| fixture | representation | width | rotation acc % | rotation NLL | tuned softmax acc % | tuned softmax NLL | params rot/soft |
|---|---|---:|---:|---:|---:|---:|---:|
| AG News | raw | 8 | 29.83 ± 5.38 | 1.6467 ± 0.0998 | 85.00 ± 1.67 | 0.4691 ± 0.0707 | 6,206/6,188 |
| AG News | centered | 8 | 26.33 ± 4.84 | 1.6988 ± 0.1118 | 84.67 ± 2.09 | 0.4640 ± 0.0727 | 6,206/6,188 |
| AG News | standardized | 1 | 28.50 ± 6.60 | 1.4275 ± 0.0339 | 46.50 ± 5.76 | 1.0163 ± 0.1017 | 781/777 |
| 20NG | raw | 4 | 38.33 ± 4.57 | 1.5270 ± 0.1396 | 89.83 ± 1.81 | 0.2810 ± 0.0166 | 3,106/3,096 |
| 20NG | centered | 4 | 36.50 ± 5.60 | 1.4999 ± 0.1287 | 90.00 ± 2.04 | 0.2792 ± 0.0169 | 3,106/3,096 |
| 20NG | standardized | 1 | 33.66 ± 8.01 | 1.3445 ± 0.0700 | 48.67 ± 5.61 | 0.8940 ± 0.0693 | 781/777 |

Rotation rows use the best tested beta for that representation/width (raw at
0.01 for AG width 8 and 0.005 for 20NG width 4; centered at 0.01; standardized
width 1 at 0.0025); softmax is trained on the fit rows and tuned only on
calibration rows.  The tuned baseline still wins
every matched-width comparison.  At width 16 it also improves on the prior
fixed-3,000-step softmax control: AG raw NLL 0.4655 vs 0.7317, 20NG raw 0.2730
vs 0.4284.  The correction to the earlier table is that standardized width 1
accepts more proposals and has a better measured NLL at a smaller step;
it does not make the signed arm competitive with the tuned softmax baseline.

The follow-up grid is the same single-step command with
`--representation raw|centered|standardized` and `--softmax-tune`; the step
diagnostic repeats width 1 with
`--representation raw|standardized --beta 0.0025|0.005|0.01 --softmax-pretrain 0`.

**Prior-centre and rotation-pretrain follow-up (2026-11-20).**  Two
asymmetries remained after the tuned-baseline comparison.  The zero-angle prior
is the identity, so its Gaussian mass sits on a one-hot outcome-0 state rather
than a class-balanced one, and the rotation head used a fixed 1,000-step Adam
pretrain while the tuned softmax early-stopped on calibration rows.
`--uniform-prior` centres the angle prior on the class-balanced state
`(1/√K, …, 1/√K)` (all weights zero and the first `K−1` canonical angles set;
the example asserts the prepared state), and `--rotation-tune` early-stops the
rotation pretrain on calibration Born NLL, evaluating every 100 steps with
patience 4.  Training uses fit rows only and selection uses calibration rows
only; test rows are never read.  The grid is the same four selected cells (AG
raw width 8 at `beta=0.01`, AG standardized width 1 at `0.0025`, 20NG raw width
4 at `0.005`, 20NG standardized width 1 at `0.0025`), the four
fixed/tuned × zero/uniform arms, and the five paired seeds, with the softmax
control disabled (`--softmax-pretrain 0`).  All 80 cells completed.

| cell | base acc % / NLL | early-stop | uniform prior | uniform + early | early step |
|---|---:|---:|---:|---:|---:|
| AG raw w8 | 29.83 / 1.647 | 28.00 / 1.715 (+0.07) | 30.83 / 1.712 (+0.07) | 29.00 / 1.670 (+0.02) | 280 ± 349 |
| AG std w1 | 28.50 / 1.427 | 28.83 / 1.410 (−0.01) | 30.50 / 1.500 (+0.07) | 29.83 / 1.430 (0.00) | 340 ± 195 |
| 20NG raw w4 | 38.33 / 1.527 | 39.50 / 1.460 (−0.07) | 37.33 / 1.540 (+0.02) | 39.33 / 1.530 (+0.01) | 120 ± 45 |
| 20NG std w1 | 33.66 / 1.344 | 33.00 / 1.360 (+0.01) | 32.66 / 1.380 (+0.04) | 32.00 / 1.400 (+0.05) | 520 ± 444 |

Parentheses are the paired mean `ΔNLL` against the base cell over the five
seeds; a negative value is better.  The uniform prior is worse on mean NLL in
all four cells (+0.02 to +0.07) and moves accuracy by −1.00 to +2.00 points;
its class balance does not fix the arm.  Early stopping is the only arm with a
negative mean `ΔNLL` on two cells, but both are within seed scatter (AG std
−0.01 ± 0.06, 3/5 seeds; 20NG raw −0.07 ± 0.12, 3/5), and it is worse on the
other two (AG raw +0.07 ± 0.19; 20NG std +0.01 ± 0.02, 0/5 seeds).  The
selected step is unstable (120 ± 45 to 520 ± 444; validation NLL often prefers
the full 1,000 steps), so the fixed pretrain is not the asymmetry that explains
the gap.  Acceptance stays 212–478/500, so no arm is frozen.  The tuned
softmax still wins every selected cell; neither intervention changes the
negative conclusion.  The draws in this grid were retained chain-by-chain and
therefore came from chain 0 only; the cross-chain rerun below supersedes the
absolute numbers (the ranking is unchanged).

Reproduce one cell with

```bash
for flags in "" "--rotation-tune" "--uniform-prior" "--uniform-prior --rotation-tune"; do
  cargo run --offline --release -p pcnl --example rotation_experiment -- \
    --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --mode single \
    --seed 20261118 --fit-per-class 20 --cal-per-class 20 --hidden 8 \
    --representation raw --beta 0.01 $flags \
    --steps 250 --burn 50 --chains 2 --draws 10 \
    --pretrain 1000 --pretrain-lr 0.01 --softmax-pretrain 0
done
```

**Cross-chain retention and mixing audit (2026-11-20).**  The earlier R12 grids
asked for 10 retained draws from 2 chains, but `update` appended each chain's
thinned trace and then truncated to `draws`, so every published MCMC number was
ten draws from chain 0.  `update` now allocates the total draw quota across
chains, spreads each chain's quota over its post-burn trace, and records
`(chain, step)` provenance; pre-update prior draws report an unavailable
provenance.  A regression test covers the 2-chain/6-draw allocation, and the
checkpoint round-trip preserves the provenance.  The four selected cells were
rerun for all four prior/pretrain arms and five paired seeds; all 80 cells
completed.

| cell | arm | acc % | NLL | Brier | draw-NLL mean | split R̂ |
|---|---|---:|---:|---:|---:|---:|
| AG raw w8 | base | 27.67 ± 3.20 | 1.4281 ± 0.0837 | 0.7710 | 2.4597 | 1.202 |
| AG raw w8 | early-stop | 31.17 ± 4.59 | 1.4038 ± 0.0606 | 0.7571 | 2.4407 | 1.381 |
| AG raw w8 | uniform | 29.00 ± 4.58 | 1.4934 ± 0.0672 | 0.7863 | 2.5027 | 0.935 |
| AG raw w8 | uniform + early | 26.00 ± 3.08 | 1.4537 ± 0.0427 | 0.7814 | 2.4681 | 1.505 |
| AG std w1 | base | 28.84 ± 7.25 | 1.3797 ± 0.0741 | 0.7484 | 1.5119 | 1.123 |
| AG std w1 | early-stop | 30.17 ± 6.86 | 1.3876 ± 0.0723 | 0.7481 | 1.4624 | 1.776 |
| AG std w1 | uniform | 29.83 ± 4.62 | 1.4371 ± 0.1207 | 0.7523 | 1.6043 | 2.769 |
| AG std w1 | uniform + early | 27.00 ± 4.88 | 1.4079 ± 0.0818 | 0.7573 | 1.4766 | 2.752 |
| 20NG raw w4 | base | 37.83 ± 4.81 | 1.3109 ± 0.1081 | 0.7173 | 2.1681 | 1.981 |
| 20NG raw w4 | early-stop | 37.00 ± 6.78 | 1.2876 ± 0.1049 | 0.7048 | 2.0854 | 1.223 |
| 20NG raw w4 | uniform | 41.00 ± 3.79 | 1.2922 ± 0.0686 | 0.6945 | 2.1143 | 1.322 |
| 20NG raw w4 | uniform + early | 40.17 ± 4.18 | 1.2793 ± 0.0676 | 0.6865 | 2.0833 | 1.340 |
| 20NG std w1 | base | 33.33 ± 7.97 | 1.3324 ± 0.0782 | 0.7230 | 1.4268 | 1.401 |
| 20NG std w1 | early-stop | 33.17 ± 8.59 | 1.3437 ± 0.0740 | 0.7302 | 1.4258 | 1.628 |
| 20NG std w1 | uniform | 33.17 ± 3.97 | 1.3693 ± 0.0561 | 0.7396 | 1.4447 | 1.316 |
| 20NG std w1 | uniform + early | 31.83 ± 3.61 | 1.3771 ± 0.0292 | 0.7436 | 1.4342 | 1.307 |

`draw-NLL mean` averages the test NLL of each individual draw before the
Bayesian mixture; it is not the predictive NLL.  `split R̂` is computed from
the per-chain sequences of that draw-level NLL (five draws per chain), so it is
a coarse chain-disagreement diagnostic, not a parameter-space convergence
certificate.  Paired mean `ΔNLL` against the base cell (negative is better):

| cell | early-stop | uniform | uniform + early |
|---|---:|---:|---:|
| AG raw w8 | −0.0243 ± 0.0814 (2/5) | +0.0653 ± 0.1201 (1/5) | +0.0256 ± 0.1096 (1/5) |
| AG std w1 | +0.0079 ± 0.0270 (3/5) | +0.0574 ± 0.1042 (2/5) | +0.0282 ± 0.0661 (1/5) |
| 20NG raw w4 | −0.0233 ± 0.0681 (2/5) | −0.0186 ± 0.1621 (3/5) | −0.0315 ± 0.1647 (3/5) |
| 20NG std w1 | +0.0113 ± 0.0116 (0/5) | +0.0369 ± 0.0716 (2/5) | +0.0447 ± 0.0568 (1/5) |

Cross-chain retention materially changes the absolute scores.  The base cells
move from AG raw 1.6467 to 1.4281, AG standardized 1.4275 to 1.3797, 20NG raw
1.5270 to 1.3109, and 20NG standardized 1.3445 to 1.3324.  The old chain-0-only
readout was pessimistic about NLL, not optimistic; accuracy is similar or
slightly lower.  The arm conclusions do not change: early stopping improves the
mean on AG raw and 20NG raw but only in 2/5 seeds, is flat on AG standardized,
and is worse in every seed on 20NG standardized.  The uniform prior is worse on
three of four cells, with a non-replicated 20NG-raw gain inside seed scatter.
No arm is promoted.

The more important result is that the corrected ensembles still do not mix:
mean split R̂ exceeds 1.05 in fifteen of sixteen rows and reaches 2.77, with
single-seed values of 6.94 and 7.72.  Ten retained draws (five per chain), 250
steps per chain and a shared Adam warm start are too few for differences of
this size.  The proper scores above are therefore descriptive, not converged
posterior estimates; the tuned softmax comparison is unaffected and still wins
every selected cell.

**Budget escalation and parameter-space mixing (2026-11-20).**  The R̂ warning
was tested rather than assumed.  The four base cells were rerun at 1,000 steps,
burn 200, two chains and 40 retained draws (20 per chain), on the same five
seeds.  All 20 cells completed.

| cell | acc % | NLL | Brier | draw-NLL | predictive R̂ | param R̂ median | param >1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AG raw w8 | 31.17 ± 6.68 | 1.3586 ± 0.0571 | 0.7401 | 2.4460 | 1.078 ± 0.076 | 2.005 ± 0.018 | 98.0% |
| AG std w1 | 30.16 ± 8.36 | 1.3600 ± 0.0805 | 0.7380 | 1.5067 | 1.365 ± 0.490 | 2.110 ± 0.032 | 98.1% |
| 20NG raw w4 | 38.67 ± 6.14 | 1.2994 ± 0.1020 | 0.7036 | 2.2325 | 1.206 ± 0.218 | 2.047 ± 0.016 | 98.3% |
| 20NG std w1 | 36.50 ± 7.78 | 1.3091 ± 0.0935 | 0.7132 | 1.4098 | 1.364 ± 0.353 | 2.047 ± 0.057 | 98.2% |

Paired `ΔNLL` against the 250-step base (negative is better): AG raw
−0.0695 ± 0.0595, AG standardized −0.0197 ± 0.0198, 20NG raw −0.0115 ± 0.1128,
20NG standardized −0.0233 ± 0.0247.  Every base NLL improves, and accuracy rises
in all four cells, but predictive R̂ remains above 1.05 in all four means.  The
parameter-wise R̂ is worse: its median is about 2.0 and roughly 98% of finite
coordinates exceed 1.05 in every cell, including the 781-parameter standardized
heads.  This is not merely a width effect; the rotation head is weakly
identified, or has rotational gauge directions, and we did not prove which.
Neither diagnostic certifies a converged posterior at this budget.

A single seed was then run at 8,000 steps (burn 1,600, same 20 draws per chain):

| cell | acc % | NLL | predictive R̂ | param R̂ median |
|---|---:|---:|---:|---:|
| AG raw w8 | 36.67 | 1.3487 | 0.968 | 1.911 |
| AG std w1 | 30.83 | 1.3388 | 0.950 | 1.940 |
| 20NG raw w4 | 55.00 | 1.0511 | 1.340 | 2.010 |
| 20NG std w1 | 28.33 | 1.3598 | 1.313 | 1.890 |

Only the two AG cells reach predictive R̂ below 1.05; the 20NG cells do not,
and parameter R̂ stays near 1.9 everywhere.  This is one paired seed, so it is a
counterexample to "more steps always suffices", not a convergence result.

A finer step-size sweep located the rejection cliff and showed that 1,000-step
acceptance is dominated by an initial transient.  On the AG raw seed-18 cell at
1,000 steps, `beta=0.012` accepted 1,475/2,000 proposals (predictive R̂ 1.015,
parameter median 1.961), `beta=0.015` accepted 1,103 (1.084, 1.844), and
`beta=0.02` accepted 281 (1.392, 1.447).  `beta=0.03`, 0.1 and 0.3 still
rejected every proposal.  At `beta=0.02` acceptance was only 14% over the first
1,000 steps but 38.5% at 2,000, 47.3% at 4,000 and 48.6% at 8,000, so the chain
escapes a stiff start rather than sampling efficiently from the outset.

Longer runs did not converge.  At 8,000 steps `beta=0.015` accepted
11,166/16,000 (NLL 1.3685, predictive R̂ 1.043, parameter median 1.806);
`beta=0.02` accepted 7,780 (NLL 1.3517, predictive 1.222, parameter 1.701);
`beta=0.01` remained the best predictive cell (NLL 1.3487, predictive 0.968,
parameter 1.911).  Raising burn from 1,600 to 4,000 was worse for both
`beta=0.01` (predictive 1.091, parameter 2.376) and `beta=0.02` (1.227, 2.065),
so the failure is not a short burn-in.  Retaining 100 draws per chain instead of
20 left parameter R̂ at 1.998 (rank-normalised 1.787) while predictive R̂ was
0.997, so 40 draws were not the cause.  Rank-normalised parameter R̂ was 1.735
at `beta=0.01` and 1.604 at `beta=0.02`, confirming the raw statistic is not
merely a heavy-tail artefact.  The experiment now prints `param_rank_rhat`
alongside `param_rhat`.

Two independent-start rescues also failed.  Starting each chain at an
independent Gaussian perturbation of the anchor gave 0/2,000 accepted proposals
at spread 0.05 and 0.25 (chains froze at their starts), while spread 0.01 kept
1,676/2,000 accepted but raised predictive R̂ from 1.188 to 1.651.  The tiny
accepted step is load bearing; it cannot be replaced by a larger one without a
better proposal.

R12 score comparisons are therefore frozen at the 250-step table, with the
1,000-step table reported only as a budget-sensitivity check.  No arm or cell is
promoted.  The tuned softmax control is unchanged and still dominates.  Before
another prior or architecture sweep, either use a preconditioned/adaptive
proposal (or a reduced, identifiable parameterization) or set a predictive-R̂
gate and run long enough to pass it; adding steps alone is not sufficient.

**Diagonal preconditioner and linear angle head (2026-11-21).**  Both fixes named
above were implemented and measured on the same four selected cells, seed
20261118, 80 fit / 80 calibration / 120 test rows, 2 chains, 40 retained draws,
the same Adam pretrain and the same 8,000 steps (burn 1,600, 16,000 proposals).
The first is a per-coordinate pCN proposal: `beta_i` is frozen from a cloned
pilot session's pooled within-chain sample variance,
`beta_i = fraction * sd_i / (sigma * prior_scale_i)`, clipped to `[1e-4, 1]`.
The pilot has its own clone and does not advance the production chains or RNG,
and equal betas recover the scalar kernel exactly (kernel/config tests pass
25/25).  The second is a linear angle head (`--linear-head`): no hidden layer,
so hidden-unit sign flips and permutations are absent by construction.

Scalar and pilot (fraction 0.1) at 8,000 steps; predictive R̂ is printed with
its draw-NLL MCSE in parentheses.

| cell | arm | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | accepted |
|---|---|---:|---:|---:|---:|---:|---:|
| AG raw w8 | scalar | 36.67 | 1.3487 | 0.968 (0.035) | 1.911 | — | 14,018/16,000 |
| AG raw w8 | pilot 0.1 | 32.50 | 1.3062 | 1.043 (0.036) | 1.816 | 1.678 | 11,400/16,000 |
| AG std w1 | scalar | 30.83 | 1.3388 | 0.950 (0.011) | 1.940 | 1.758 | 7,856/16,000 |
| AG std w1 | pilot 0.1 | 25.83 | 1.3523 | 1.014 (0.008) | 2.038 | 1.810 | 8,926/16,000 |
| 20NG raw w4 | scalar | 55.00 | 1.0511 | 1.340 (0.030) | 2.010 | 1.791 | 15,523/16,000 |
| 20NG raw w4 | pilot 0.1 | 50.00 | 1.1542 | 1.291 (0.045) | 1.808 | 1.671 | 12,527/16,000 |
| 20NG std w1 | scalar | 28.33 | 1.3598 | 1.313 (0.012) | 1.890 | 1.756 | 7,678/16,000 |
| 20NG std w1 | pilot 0.1 | 30.00 | 1.3835 | 0.967 (0.007) | 2.137 | 1.828 | 13,937/16,000 |

The per-coordinate kernel lowers the parameter R̂ median by about 0.10 on the
two raw cells, but raises it on both standardized cells.  It lowers predictive
R̂ only on 20NG standardized, and there the point NLL is worse (1.3835 vs
1.3598).  No cell passes both predictive R̂ < 1.05 and parameter R̂ < 1.05.
Acceptance is not monotone in the frozen step size: the standardized cells now
accept more proposals yet mix no better.  The pilot fraction was selected on AG
raw alone: 0.5 froze every chain (0/16,000 accepted, NLL 2.1511), 0.05 gave
predictive 1.004 but parameter 1.948 and rank 1.753 with worse NLL 1.3872, and
0.1 was used unchanged on the other three cells.  This is one pilot setting and
one seed, not a tuned preconditioner.

Linear angle head at the same budget.  For the standardized cells the scalar
tuned beta accepted 0/16,000, so the linear head was rerun at `beta=1e-4`; that
is a proposal re-tune and is labelled as such, not a same-kernel comparison.

| cell | beta | params | accepted | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AG raw | 0.01 | 4,614 | 10,604/16,000 | 25.00 | 1.4345 | 0.977 (0.040) | 1.949 | 1.760 |
| AG std | 1e-4 | 4,614 | 15,743/16,000 | 28.33 | 1.4647 | 1.063 (0.035) | 2.074 | 1.821 |
| 20NG raw | 0.005 | 4,614 | 14,872/16,000 | 16.67 | 1.5034 | 1.003 (0.039) | 2.044 | 1.807 |
| 20NG std | 1e-4 | 4,614 | 15,722/16,000 | 26.67 | 1.4656 | 0.981 (0.040) | 2.016 | 1.803 |

The linear head is a clear regression: every NLL is worse than its hidden-head
scalar baseline (AG raw 1.4345 vs 1.3487, AG std 1.4647 vs 1.3388, 20NG raw
1.5034 vs 1.0511, 20NG std 1.4656 vs 1.3598), and 20NG raw falls to 16.67%
accuracy.  It mixes (14,872–15,743 accepted where retuned) but parameter R̂ stays
near 2.0 and rank R̂ near 1.80.  Removing hidden-unit sign and permutation gauges
therefore does not fix the parameter-space failure in this configuration; the
non-identification is not mainly that gauge.  For the standardized cells the
linear head also has more parameters (4,614 vs 781), so this is a reduced
*architecture*, not a reduced parameter count.

Budget gate at 32,000 steps (burn 6,400, 2 chains, 40 draws, 64,000 proposals).
The predictive-R̂ gate is 1.05.

| cell | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | accepted | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 20NG raw | 69.17 | 0.9171 | 1.489 (0.041) | 1.913 | 1.745 | 60,706/64,000 | fail |
| 20NG std | 23.33 | 1.6242 | 1.018 (0.030) | 2.500 | 1.992 | 110/64,000 | fail |

The 20NG raw point score improves sharply with budget (NLL 1.0511 → 0.9171,
accuracy 55% → 69.17%), but predictive R̂ rises from 1.340 to 1.489 and
parameter R̂ is unchanged, so the better fit is not a convergence result.  The
standardized cell degrades: acceptance collapses from 7,678/16,000 at 8,000
steps to 110/64,000, NLL worsens to 1.6242 and parameter R̂ rises to 2.500.
Neither cell passes the gate, and this is one seed.  The budget-sensitivity
check therefore closes negatively for both 20NG cells.

**Four-chain control (2026-11-21).**  The parameter-R̂ values above use two
chains, the minimum for the statistic, so both 20NG cells were rerun at 8,000
steps with four chains and 80 retained draws (20 per chain), same seed, beta and
pretrain.  The 32,000 proposals are the same total budget as the two-chain run.

| cell | acc % | NLL | predictive R̂ (MCSE) | param R̂ median | rank R̂ median | param >1.05 | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG raw | 60.83 | 1.0247 | 1.188 (0.020) | 2.409 | 2.056 | 99.9% | 31,050/32,000 |
| 20NG std | 29.17 | 1.3586 | 1.131 (0.008) | 2.238 | 1.971 | 100.0% | 15,126/32,000 |

The failure is not a two-chain estimator artifact.  Against the paired two-chain
rows, the predictive R̂ improves somewhat (raw 1.340 → 1.188, standardized
1.313 → 1.131) but still fails the 1.05 gate, while the parameter and rank
medians rise (raw 2.010 → 2.409 and 1.791 → 2.056; standardized 1.890 → 2.238
and 1.756 → 1.971).  With four chains, 100% of the standardized head's 781
coordinates and 99.9% of the raw head's 3,106 coordinates exceed 1.05.  A broad
parameter-space disagreement is therefore present under the standard chain
count, not just under a fragile two-chain estimate.

**Gradient-informed sampler (2026-11-21).**  `SessionConfig::sampler = Mala`
adds an exact preconditioned MALA kernel beside pCN.  In normalized prior
coordinates `u_i=(θ_i−c_i)/(σ s_i)` the proposal is
`u' = u − (τ²/2)∇Φ(u) + τ ξ`, with the exact Born gradient and the full
reverse-density MH ratio.  The step `τ` (`--mala-step`) was selected only by
short acceptance pilots, never by test NLL: on standardized 1,500-step runs,
`τ=0.002` gave 50% acceptance (0.001 gave 86%); on raw, 0.020 gave 44% at 1,500
steps (0.024 gave essentially none).  The matched 8,000-step runs then used the
same seed, split, pretrain and 2-chain/40-draw budget as the pCN rows.

| cell | kernel | step | acc % | NLL | predictive R̂ (MCSE) | param R̂ | rank R̂ | accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG std | pCN | 0.0025 | 47.99 | 1.3598 | 1.313 (0.012) | 1.890 | 1.756 | 7,678/16,000 |
| 20NG std | MALA | 0.002 | 59.48 | 1.3904 | 1.530 (0.010) | 2.034 | 1.794 | 9,517/16,000 |
| 20NG raw | pCN | 0.005 | 97.02 | 1.0511 | 1.340 (0.030) | 2.010 | 1.791 | 15,523/16,000 |
| 20NG raw | MALA | 0.020 | 71.62 | 1.0560 | 1.224 (0.052) | 1.676 | 1.581 | 11,459/16,000 |

Gradient information does not rescue either cell.  Standardized MALA has healthy
acceptance but a worse predictive R̂ and NLL than pCN; raw MALA lowers the
parameter and rank R̂ but is slightly worse on NLL and still far above the 1.05
gate.  The raw MALA acceptance also rose from 44% in the pilot to 71.6% in the
8,000-step run, so that pilot did not fix a stationary acceptance rate.

**Posterior-geometry diagnostic (2026-11-21).**  `RotationSession::posterior_geometry`
reports, per coordinate: within-chain posterior sd divided by the prior sd; a
flat flag (`≥0.5`, prior-dominated), an identified flag (`≤0.1`); and the
largest pairwise chain-mean difference divided by the pooled within-chain sd, a
standardized separation with a `>2` flag.  The point is to distinguish chain
disagreement on tightly identified coordinates (a shifted or multimodal mode)
from broad drift along weakly identified ones.  It is descriptive and does not
replace R̂.

| cell | kernel | sd/prior median | flat frac | identified frac | separation median | separated frac | separated / flat | separated / identified |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20NG std | pCN | 0.054 | 0.000 | 0.986 | 1.564 | 0.392 | — | 0.392 |
| 20NG std | MALA | 0.048 | 0.000 | 0.985 | 1.629 | 0.406 | — | 0.406 |
| 20NG raw | pCN | 0.149 | 0.000 | 0.084 | 1.789 | 0.446 | — | 0.565 |
| 20NG raw | MALA | 0.476 | 0.420 | 0.000 | 1.308 | 0.305 | 0.243 | — |

On standardized, ~99% of coordinates are identified and ~40% of those are
separated by more than two within-chain sds under both kernels; this is not a
flat-direction story and MALA does not reduce it.  Raw pCN has almost no flat
coordinates and 56.5% of identified coordinates are separated.  Raw MALA is the
outlier: it leaves a broad, prior-dominated posterior (42% flat, median
sd/prior 0.48), so its lower R̂ reflects a broader ensemble, not a better-resolved
mode.  Separation can also arise from within-chain drift, so these columns are
diagnostic rather than a convergence verdict.

**Prediction-space chain distance (2026-11-21).**  Parameter separation only
matters if it changes decisions.  For each chain the experiment now averages its
retained draws' test distributions over rows, and reports the total-variation
distance `0.5 * Σ_k |p_a(k) − p_b(k)|` for every chain pair (`prediction_chains`).
This is a descriptive chain-mean distance; it can hide row-level differences
that cancel.

| cell | chains | TV mean | TV max | separated / identified |
|---|---:|---:|---:|---:|
| 20NG std | 2 | 0.0928 | 0.0928 | 0.392 |
| 20NG std | 4 | 0.0991 | 0.1253 | 0.829 |
| 20NG raw | 2 | 0.0328 | 0.0328 | 0.565 |
| 20NG raw | 4 | 0.0336 | 0.0474 | 0.870 |

The standardized disagreement survives in the average predictions: four chains
differ by about 10 percentage points total variation (maximum 12.5), and 82.9%
of identified coordinates are separated.  For a four-class report that is a
material predictive difference, so this is not merely a hidden-unit sign gauge
or a parameterization artifact.  Raw's two-chain pooled distance was only 3.3
points; the four-chain control below shows that this pooled number can be
misleading.  These are one seed and one budget; the columns carry no sampling
interval and are not a convergence certificate.

The pooled distance could still hide either a coherent block switching sides or
a diffuse shift that partially cancels across rows.  The four-chain runs
therefore also average each chain within every row and true class
(`prediction_rows`, `prediction_classes`).  Row-level pairwise TV for
standardized averages 0.1468 (maximum 0.3356); 77.8% of row pairs exceed 0.10,
and at least two chains disagree on the top label in all 120/120 test rows.  Its
per-true-class mean TVs are `[0.0865, 0.1317, 0.1019, 0.1085]` (mean 0.1072):
every class is affected, not one confused label.  This is a pervasively
different decision map, not a hidden-unit permutation.

The raw four-chain control overturns the pooled reading.  Its pooled TV is only
0.0336 (maximum 0.0474), but row-level TV averages 0.3476 (maximum 0.8688),
96.3% of row pairs exceed 0.10, the top labels differ in 109/120 rows, and the
per-true-class TVs are `[0.0851, 0.0988, 0.0997, 0.0980]` (mean 0.0954).  Raw's
small pooled distance is cancellation between rows, not predictive agreement;
its row-level disagreement is larger than standardized's.  Pooled chain-mean TV
must not be used as evidence of predictive consensus.  All these means are
descriptive, use chain-averaged predictions, and carry no uncertainty interval.

**Does row disagreement track errors? (2026-11-21).**  For each test row the
experiment correlates its mean pairwise chain TV with the pooled prediction's
NLL.  This is descriptive, not a validated error detector.

| run | Pearson | Spearman | rows |
|---|---:|---:|---:|
| 8k pCN, seed 20261118 | 0.066 | 0.099 | 120 |
| 8k PT, seed 20261118 | 0.164 | 0.116 | 120 |
| 8k pCN, seed 20261119 | 0.302 | 0.206 | 120 |
| 8k PT, seed 20261119 | 0.278 | 0.368 | 120 |
| 32k pilot, seed 20261118 | 0.144 | 0.236 | 120 |

The correlation is weak-to-moderate and seed-dependent: Pearson ranges from
0.066 to 0.302 and Spearman from 0.099 to 0.368.  Rows where chains disagree
more are somewhat more likely to have a worse pooled NLL, but the relationship
is not stable enough to treat row TV as an error predictor; it remains a spread
diagnostic.  Per-chain scores also reverse across seeds: at seed 20261118 pCN
scores `acc=[0.292, 0.233]`, `nll=[1.347, 1.410]` against PT
`acc=[0.358, 0.408]`, `nll=[1.347, 1.316]`, but at seed 20261119 pCN scores
`acc=[0.458, 0.450]`, `nll=[1.164, 1.207]` against PT
`acc=[0.408, 0.383]`, `nll=[1.228, 1.248]`.  The tail `acc=[0.258, 0.275]`,
`nll=[1.357, 1.401]` is the 32k pilot.

**Parallel tempering (2026-11-21).**  `Sampler::ParallelTempering` pairs each
cold chain (`T = 1`) with a tempered power posterior (prior fixed, likelihood
scaled by `1/T`), attempts an adjacent-pair replica exchange after every sweep,
and retains only cold chains.  The 8k standardized cell used four chains with
`T = [1, 2, 1, 2]` and 80 draws (40 per cold chain), same seed, burn and beta;
the matched pCN baseline used two chains and 80 draws.

| seed | arm | acc % | NLL | predictive R̂ | param R̂ | pooled TV | row TV | top-label disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261118 | pCN | 26.67 | 1.3580 | 1.350 | 1.922 | 0.0930 | 0.1480 | 99/120 |
| 20261118 | PT | 38.33 | 1.3157 | 1.080 | 2.024 | 0.0516 | 0.1226 | 42/120 |
| 20261119 | pCN | 50.00 | 1.1455 | 1.012 | 2.054 | 0.0599 | 0.1985 | 41/120 |
| 20261119 | PT | 45.00 | 1.1906 | 1.041 | 1.819 | 0.0853 | 0.2047 | 52/120 |

On seed 20261118, tempering cuts pooled TV by 45% and top-label disagreement
by 58%, and both cold chains score better than either pCN chain.  That advantage
does not replicate.  On seed 20261119, pCN has the lower NLL (1.1455 vs 1.1906),
the higher accuracy (50% vs 45%), slightly lower row TV (0.1985 vs 0.2047) and
fewer label disagreements (41 vs 52); PT improves only parameter R̂ (1.819 vs
2.054).  Across the two paired seeds the mean NLL delta is +0.0014 (PT minus
pCN) and the mean row-TV delta is −0.0096, but both changes have opposite signs
by seed, so no stable point-prediction or row-disagreement benefit is established.
Both arms remain far above the parameter-R̂ gate.  Seed 20261119 also shows that
a predictive-R̂ pass is not sufficient: pCN reaches 1.012 there while parameter
R̂ is 2.054.  PT remains a diagnostic control, not a convergence remedy; this is
two seeds at one budget, not a general claim.

The PT report now carries `swap_attempts` and `swap_accepted`.  Rerunning both
PT cells with that instrumentation reproduced every score above exactly.  Seed
20261118 accepted 2,129/16,000 exchanges (13.3%, one cold exchange per ~7.5
sweeps); seed 20261119 accepted 1,939/16,000 (12.1%, per ~8.3 sweeps).  Both runs
therefore exchanged replicas actively and at similar rates, so the seed reversal
is not explained by one run failing to swap.  The rate is low enough to call the
1↔2 ladder a weak mixer, but no acceptance threshold was preregistered and this
is a diagnostic, not a convergence claim.

**Pilot preconditioning at 32k (2026-11-21).**  The 20NG standardized cell was
rerun at 32,000 steps (burn 6,400, 2 chains, 40 draws) with a 0.1-fraction pilot
that froze per-coordinate betas `min=0.000324 median=0.000919 max=0.002885`.

| run | NLL | predictive R̂ (MCSE) | param R̂ | rank R̂ | accepted |
|---|---:|---:|---:|---:|---:|
| scalar β=0.0025 | 1.6242 | 1.018 (0.030) | 2.500 | 1.992 | 110/64,000 |
| pilot betas | 1.3690 | 1.152 (0.007) | 2.125 | 1.826 | 57,523/64,000 |

The pilot prevents the acceptance collapse (89.9% versus 0.17%) and improves
NLL and both parameter statistics, but predictive R̂ moves the wrong way and all
three gates still fail.  The rerun reproduces the earlier score exactly and now
also records geometry (`sd_norm=0.0555`, identified 89.1%, separation median
1.937, separated/identified 0.491) and row disagreement: pooled TV 0.0824, row
TV 0.1097 (maximum 0.2604), top-label disagreement 79/120, class-TV mean 0.0824.
Against the 8k pCN baseline (row TV 0.1480, 99/120) this lowers disagreement,
but does not remove it; the comparison changes budget, preconditioner and
retained draws per chain (20 versus 40), so it is not a clean budget effect.
It is one seed, and 32k remains a sensitivity/gate check, not a new frozen R12
score.

## Required application track R9: small Edge0 architecture with RLCD (2026-11-15)

**Reinforcement learning for calibrated decisions is required** in this track. The
Mammoth-1 reference supplies a supervised comparator. Execute the new specification in
[EDGE0_RLCD_PLAN.md](EDGE0_RLCD_PLAN.md) through E0–E7 of
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). This is the work order within R9;
R8 below remains the mathematical/sampler order, and R7's negative results and stop
remain valid. No new implementation, run, theorem or measured improvement is claimed.

### R9.0 Hypothesis and architecture

Test whether a small grouped latent-MoE architecture, adapted from the local
[Edge0 source](../../Edge0/docs/models/edge0-145b.md), can learn useful typed probability
outputs through explicit RLCD at a controlled research budget. The checkout is
`../../Edge0`, not `../../Edge-0`; pin commit `7b60076`. Start with width 64, 4 blocks,
8 routed experts, top-2 plus one shared expert, 2 D/U groups and latent width 32.
Use ordinary small attention as an explicit adaptation, not a claim to reproduce the
large model. Keep state encoding question-independent and process questions independently.
The full tensor count must stay below one million parameters in the initial prototype.

Do not confuse deterministic expert routing with a stochastic decision policy. Add a
separate four-path categorical policy `a_phi(r|state,question)` and typed conditional
readouts `u_theta`. Return the exact marginal `p=sum_r a_phi(r) u_theta(r)` for Bernoulli,
named Choice and ordinal Score outputs. These per-question marginals do not define a
joint law of correlated questions. A joint-world/conditional-factor model is required
when a workflow needs one. Reference models motivate the interface, not inherited speed,
calibration, language capability or training guarantees.

### R9.1 Required RLCD and its first falsifier

Minimize expected Brier loss of the returned marginal using a **reinforcement-learning
update of the four-path policy**, alongside direct readout gradients. Sample four iid
paths and use the unbiased pairwise Brier estimator in EDGE0_RLCD_PLAN §4. Include the
score-function gradient of the sampling law; baseline subtraction needs conditional
independence from that sample and detached baseline gradients. Exact enumeration of all
256 path tuples must recover the exact marginal's gradient for policy, heads and shared
parameters. This gate distinguishes actual RLCD from supervised readout updates renamed RL.

CE+Brier supervision is the mandatory strong comparator and optional shared warm start.
An exact-marginal Brier-gradient arm controls for the objective itself; frozen-policy
proper-score and correctness-reward RL arms identify policy learning and reward choice.
Compare the exact-marginal objective with single-path and finite-sampled-average
objectives explicitly; their disagreement terms differ. Preserve version-specific attribution: OpenJev-RLCD's current PRDO v2 uses a
per-rationale objective with direct readout gradients in its main arm. Our mixture-policy
REINFORCE construction is a new proposal requiring independent checks.

### R9.2 Controlled learning experiment and decisions

Use enumeratable synthetic worlds with known nondegenerate outcome laws, repeated/noisy
observations, binary/choice/ordinal questions and explicit utility tables. Split at the
world/template level before producing questions or labels. Separate development,
calibration and final test worlds; record an OOD family. Empty/invalid schemas and
nonfinite probabilities are explicit failures, not uniform fallback successes.

The first campaign is bounded in EDGE0_RLCD_PLAN §5: one sub-million-parameter model,
5 paired seeds, fixed 2048/256/256/512 world splits, at most 2000 optimizer updates per
arm and a hardware wall-time cap fixed before execution. Brier excess over known truth
is primary; report NLL, ordinal score, raw/calibrated controls, utility/regret and
reliability/coverage separately. Cluster uncertainty by world and replicate across seeds.
Cost includes all training paths and the four-path inference marginal. A negative RLCD
comparison is retained; change architecture/data/metric only as a new declared experiment.

### R9.3 pCNL integration and theory handoff

First use the existing generic target seam for a frozen-encoder/policy, head-only chart.
Evaluate the latent marginal exactly inside the likelihood. Stochastic log-likelihoods,
RL rewards and sampled-path Brier estimators are not exact MH likelihood evaluations.
Only a justified extended-state construction or a priced kernel approximation can relax
this constraint. The optional outer posterior over parameters is separate from the four
latent paths; R8/Q0–Q7 and D0–D7 apply to its actual target/kernel/error budgets.

The Lean specialist first proves finite simplex/mixture closure, proper-score identities,
U-statistic expectation, score-function/baseline identities and utility-regret control.
Names in the new plan are proposed theorem tasks, not existing declarations. Global
neural convergence, calibration and posterior rates still require their missing bridges.
Do not compile in this planning wave. No operator/Fock interpretation is a prerequisite
for the required RLCD training experiment.

## R8 programme: theoretical certificates with implementation checks (2026-11-15)

For sampler/theory work, this section supersedes the **forward ordering** of older R0–R7
tasks while preserving their measured records and stopping decisions. The mathematical counterpart is
[timepiece/pcnl.md](../../timepiece/pcnl.md), **Q0–Q7**; executable implementation packets
are in [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md), **D0–D7**. This source/planning review
performs no Rust or Lean compilation, numerical run or new measurement.

The research aim is to establish useful properties in increasing scope: proper targets,
correct ideal MH kernels, a first explicit mixing certificate, controlled approximation
bias, informative dimension scaling, and independently evaluated predictive/decision quality.
A rate certificate for an exact reference sampler is not a rate for unadjusted SG-pCNL;
a statistical accuracy theorem is not a promise that BMA beats a point model.

### R8.0 Current state to reuse

The generic `Potential` / `SubspaceTarget<P>` seam, restricted-chart MH-pCN, scaled and
rotated proposals, classifier full-target block kernels, `policy_model`, `mdp`,
`policy_readout`, `dt_arm` and `dt_subspace` are present. Do not reimplement them from the
old “proposed” descriptions. `src/target.rs` uses a total Gibbs potential including the
Gaussian prior; its MH functions include the compensating proposal-density term.
The source-level tests and historical cells are evidence to reuse, with their actual
revision/normalization scope, not fresh passes claimed by this revision.

[PCNL_STATUS.md](../docs/PCNL_STATUS.md), R7 and its later follow-ups, already records
DT comparisons and a stopped restricted rank-4 arm: the next-action-NLL success criterion
was not established, and later weak-anchor proper-score improvements do not meet it.
It also records step-independent score uncertainty as an unresolved limitation. Keep
these outcomes. Fix uncertainty and policy approximation before proposing another
application promotion; do not erase a negative result with a newly chosen metric.

### R8.1 First theorem: a proper target and a prior-refresh reference

Use likelihood potential `Ψ=λ n_terms mean_loss`, total potential `J=Ψ+‖z‖²/2`,
and the declared Gaussian reference `μ0`. Prove `0<Z=∫exp(-Ψ)dμ0<∞`; categorical
nonnegative NLL at `λ≥0` gives the upper bound, with positivity requiring a positive-prior
set of finite loss. Other likelihoods need their own integrable bound.

The short first rate theorem is independence MH with proposal `μ0`. If
`0≤L=exp(-Ψ)≤M` and `Z>0`, its acceptance gives `K_ind(x,·)≥(Z/M)π(·)` on target
support. A fixed mixture with a same-target invariant local MH kernel has coefficient
`c=ω Z/M`, `ω>0`. Prove the resulting TV and centred-`L²` contraction before adding
more difficult transformer-specific hypotheses. `mh_pcn_z(beta=1)` is the existing
restricted-chart primitive; a random mixture wrapper and a certified evidence lower
bound are **new development**, not existing sampler guarantees.

A useful first witness is `Z≥μ0(A)exp(-b)>0` for a verified region with `Ψ≤b`.
Use tiny-chart quadrature with tail enclosures or an analytic likelihood/ball estimate.
An estimated evidence or local favorable ray is not a rigorous lower bound. Publish
both the positive certificate and its practical rate; it may be too small to be useful.
Increasing complement dimension only preserves this bound if `Z/M` stays bounded below.

### R8.2 Two dimension claims, two different experiments

First prove and test a factorized reference `Ψ(z,w)=Ψ_r(z)`: the informed target and
independent Gaussian complement have a simultaneous product-kernel absolute gap
`min(γ_r,1-ρ)`, independent of complement dimension. Freeze rank, likelihood, proposal
and observable while varying complement dimension; report per-step cost separately.
This is the controlled theorem the project can aim to complete first.

Then study coupled targets. Full-target block MH updates may each be reversible, but
their deterministic composition need not be. Either analyze its actual multi-step
contraction or add a separately labeled random-scan/lazy reference. Conditional gap and
approximate variance-factorization bounds are possible sufficient inputs; they must be
proved uniformly in the conditioned coordinates and dimension. Local Hessian eigenvalues,
a rank budget heuristic and stable loss ESS cannot supply those inputs.

[The pCN gap paper](https://arxiv.org/abs/1112.1392) and
[DILI](https://arxiv.org/abs/1411.3688) provide candidate assumptions and constructions;
copy their actual hypotheses into the claim ledger before applying them. They do not
establish a theorem for this arbitrary neural target by analogy.

### R8.3 Reduction, approximation and uncertainty are priced separately

A frozen complement is a new slice model, not a full-support approximation to the ambient
posterior. Keep it separate from a ridge likelihood `Lbar(z)=E_w L(z,w)` with a Gaussian
complement. The latter retains the correct informed marginal but needs a complement-error
bound. [Certified dimension reduction](https://arxiv.org/abs/1807.03712) suggests a
conditional-inequality/neglected-gradient route; verify its assumptions and sample-error
control rather than substitute one empirical Fisher spectrum. Retain random-basis and
full-target controls, and cross-fit data-selected anchors/bases.

For the approximate SG branch, prove the Gaussian AR stability/bias canaries in the
formal plan, then bound the actual gradient, discretization and numerical defects in a
common metric. A nonzero step floor leaves a fixed-step regime. Exact mini-batch MH would
need a justified construction; an unbiased log-loss estimate is insufficient. The current
`f32` proposal casts also need a rounding-to-statistical-error bridge: ideal real-MH
invariance is not literal detailed balance of rounded Gaussian cells.

A final predictive certificate must distinguish target/restriction error, kernel bias,
burn-in and correlated finite-sample uncertainty. For stationary absolute gap `γ`, the
reference observable mean-square bound is `(2-γ) Var(F)/(N γ)`; prove any nonstationary
correction used by an actual run. In DT reports separately handle parameter-chain
correlation, dependence within held-out trajectories and rollout noise. Action-score
benefits do not guarantee return benefits. Per-history policy approximation can feed a
finite-horizon return bound; a finite off-policy score table alone cannot supply its
uniform premise. Test the rollout decoder's posterior-draw subsampling separately.

### R8.4 Promotion gates and order

Execute **D0 → D1 → D2 → D3 → D4**, then D5 for existing application uncertainty;
D6's spectral computation is optional and D7 verifies the completed evidence packets.
The first promotion requires a nonconstant-likelihood reference certificate with explicit
normalization, rate and error/cost tolerances. The second requires the same certificate
with fixed informed rank and a growing independent complement. Coupled-transformer
claims require their remaining estimates; negative or inconclusive assumptions stay visible.

An application claim additionally needs the prespecified proper-score/return criterion
against matched-cost tuned baselines with independent data/seed replication. Do not
reopen the stopped R7 branch merely because reference-model theory improves. No new
global neural gap, dimension-free runtime, posterior optimality or policy optimality is
claimed by this work order. Existing R0–R7 records below remain the experimental detail.

Historical baseline: 2026-11-08. Implementation inspected at unfer commit
`5b4b79c77f8c35bb4039ab1c755733621b5cd68f`. Revised 2026-11-12: added R7, the
decision-transformer application track, and its milestones; the R0–R2 record is unchanged.

## Objective and reading order

Develop a reproducible Bayesian-adaptation laboratory that can answer two questions:
does a tractable, explicitly specified posterior improve predictions over a strong point
baseline, and can a sampler approximate that posterior within a measured compute budget?
Treat target quality, sampler accuracy and decision/readout quality as separate experiments.

The original research bet was **adaptation in a likelihood-informed parameter subspace**,
followed by a full-target sampler with different proposals inside and outside that subspace.
Do not start another ambient step-size/temperature ladder as the default next experiment.
The accumulated negative results make target specification and verification the next gates.

The earlier R7 application studied here is **a small decision transformer (DT) whose output
is an action distribution** (R7): offline trajectories, return-to-go conditioning, and a policy
as the posterior predictive. Only the potential and the readout change; the prior, the affine
subspace chart and the pCN kernel carry over. R7 is gated on R0's gradient/diagnostic checks and
R3's proper-score readout, not on R5–R6, and it is the first target here whose predictive *is*
the object that gets scored: a distribution rather than a label.

R13 at the top of this file gives the current work order. Earlier tracks preserve the context of
[the original implementation plan](../docs/plan_bayes_ngpt.md) and the experiments it motivated.
That document remains the original design/formalization record.
[PCNL_STATUS.md](../docs/PCNL_STATUS.md) remains the measured-results ledger.
The motivating discussion is [ngpt.md](../../test/unfer/ngpt.md), especially §6.5.9,
§6.6 and §9.3–§9.9. Cite those headings rather than mutable line numbers.
No experiment, code fix or Lean compilation is claimed by this planning revision.

## Evidence and what it licenses

The discussion records stable normalized runs that nevertheless lose useful predictions;
centered priors that change reachability and posterior concentration; cold/annealed/metric
controls that do not solve the ambient target problem; and informative cap/latitude
conditionals whose predictions are not the unrestricted posterior predictive.
The November 7 discussion also corrects older optimizer anchors and absolute accuracy values.
Regenerate anchors before comparing any new result with a historical table.

The exact Haar cap law is implemented in [budget.rs](src/budget.rs), while the proposed
dimension budget combines that geometry with measured likelihood contrasts. Use it to
choose candidate ranks, not as a theorem that the likelihood constrains exactly 7–29
directions or that every remaining direction is flat. A few random rays, a local Hessian
and the absence of visits to a rare cap cannot exclude unseen informative regions.
In particular, importance sampling with high ESS among observed prior draws can still
miss a small region with large likelihood. Record the overlap limitation.

For a cap of prior mass `p`, a rigorous likelihood-ratio bound `exp(B)` between its
interior and exterior would bound posterior odds by `exp(B) * p / (1-p)`.
Ray-based estimates of `B` do not establish that uniform bound. The measured budget and
this conditional inequality must remain distinct. Use the exact incomplete-beta cap
probability for calculations; reserve its large-dimension approximation for interpretation.

The implementation is finite dimensional with isotropic covariance, not a trace-class
infinite-dimensional Gaussian model. “Dimension robust” needs a controlled scaling study;
bounded gradients and finite gradient-noise variance alone do not prove posterior accuracy,
ergodicity, mixing, or a nonvanishing acceptance rate uniform in dimension.

## Implementation inventory

Paths here are relative to `pcnl/`; existing modules are reuse points, not completed
solutions to the new milestones.

| Component | Existing surface | Next use |
|---|---|---|
| Model and geometry | `src/model.rs`, `src/normalize.rs` | Forward/gradient contract |
| Ambient SG engine | `src/sampler.rs` | Approximate-kernel comparator |
| Direction kernels | `src/sphere.rs` | Audited sphere and conditional controls |
| Geometry accounting | `src/budget.rs`, `src/radial.rs` | Rank hypotheses and cap costs |
| Prediction scores | `src/readout.rs` | BMA, proper-score and reward estimators |
| Experiments | `src/task.rs`, `src/experiment.rs` | Shared tasks and baselines |
| Sequence policy model | `src/policy_model.rs` | DT forward, per-step action distributions, own `layout` |
| Decision task | `src/mdp.rs` | Synthetic MDP, behavior corpus, exact and rollout returns |
| Policy readout | `src/policy_readout.rs`, reusing `src/readout.rs` | Trajectory uncertainty and posterior-average policy error |
| Potential abstraction | `src/target.rs` | Existing `SubspaceTarget<P: Potential>` target/gradient contract |
| Diagnostics | `src/diag.rs` | Stationarity and Monte Carlo uncertainty |
| Saved draws | `src/persist.rs` | Trajectories plus run metadata |
| Reproduction | `../scripts/pcnl-reproduce` | Named evidence stages |

Modules marked *proposed* do not yet exist. Do not count the research plan as an
implementation or treat a command naming a proposed example as an available command.

## R0 — establish numerical and experimental trust before new inference

**Development tasks, in order:**

1. Check the end-to-end `loss_and_grad` against directional finite differences on tiny
   models, multiple parameter blocks and several nonzero norm scales. Include zero and
   near-zero behavior. `normalize_fro` currently calls `to_scalar` on the norm and then
   rescales the tensor: its denominator is outside the autodiff graph. Decide explicitly
   whether the intended derivative is a stopped-gradient surrogate or the derivative of
   the normalized forward map. For posterior-gradient claims implement and test the latter;
   if a surrogate is retained, label it and evaluate its drift bias. An exact MH kernel
   can use a surrogate drift only with its actual proposal density in the ratio.
2. Resolve the schedule contract in `sampler.rs`. `beta_at` clamps to `1e-6`, so the
   implemented sequence eventually has a positive floor, even when the mathematical
   power law has sums `sum beta^2 = infinity`, `sum beta^4 < infinity` for
   `1/4 < kappa <= 1/2`. Tail-mobility formulas for the unfloored law do not describe
   the floored implementation. Expose separate annealing and finite-floor modes; report
   the floor crossover and actual mobility. The exponent window is only one necessary
   schedule condition, not a convergence theorem. Cyclical sampling is a separate mode.
3. Make frozen/nonfinite traces diagnostic failures or explicitly unavailable values.
   `diag::ess` currently returns the draw count for zero variance; identical constant
   chains can produce `Rhat = 1`. Neither licenses a mixing claim. Add a movement flag,
   acceptance/unique-state count where relevant, and tests for stuck chains in both
   identical and different states. Use rank-normalized split/folded Rhat and bulk/tail
   ESS, with independent cross-checks on synthetic traces and a documented convention.
   Use [Vehtari et al.](https://arxiv.org/abs/1903.08008) for the diagnostic definitions.
4. Audit every “exact” proposal against its implemented density. For spherical tangent
   Gaussians, handle the exponential map's multiple preimages beyond the injectivity
   radius, or use a support restriction with the correct forward/reverse normalizers.
   Check restricted-metric determinants, tangent covariance, Jacobians and any safeguards.
   A fixed ambient metric still induces a position-dependent tangent law. Treat fixed
   geodesic shell, latitude and cap-conditioned targets separately. Verify detailed balance
   or invariant moments on circle/S2 targets with known answers before high-dimensional runs.
5. Freeze metric adaptation before stationary draws and record the freeze point. Separate
   warm-up, nonstationary annealing and stationary sampling in reports. Do not interpret
   ordinary ESS/Rhat on an arbitrary decaying-step trace as stationary-chain diagnostics.
6. Add a run manifest beside saved draws: code revision and dirty diff identity, model and
   normalization version, task/data/anchor hashes, split seeds, target/prior/temperature,
   optimizer and sampler configuration, actual schedule, basis/metric hash, device/dtype,
   warm-up/thinning, evaluations, wall time, memory and nonfinite events. Saving draws is
   not exact resumption; resumption additionally needs RNG, loader and adaptation state.

*Status (2026-11-09):* items 1–6 are implemented and measured — `docs/PCNL_STATUS.md`,
"R0", (a)–(f). Item 4's residual sub-audit (cap-conditioned target, cyclical schedule) is
still open and is named there; item 1's surrogate is labelled and its drift bias measured
at 0.14 %, so the recorded anchors stand, but every gradient-level claim must name the
normalisation form it used.

**Acceptance:** finite-difference discrepancies are within documented scale/dtype tolerances;
schedule reports agree with evaluated steps; frozen traces fail the reporting gate; each
exact comparator agrees with a known low-dimensional target within Monte Carlo uncertainty.
Re-run affected old cells after a gradient/model fix. Preserve old logs as superseded data.

## R1 — specify and test the informed-subspace target

**Research hypothesis:** a rank-limited adaptation model can retain the pretrained model's
useful predictions while expressing uncertainty in directions informed by task data.
This is a model-selection hypothesis, not a claim that projection preserves the ambient
posterior. Start with an affine Gaussian model because its prior and MH ratio are simple.

For frozen anchor `theta_ref`, prior covariance factor `C_half` and orthonormal columns
`B` of size `P x r`, define

```text
theta(z) = theta_ref + C_half B z
z ~ N(0, I_r)
pi_r(dz) proportional to exp[-lambda * sum_i loss_i(theta(z))] N(0, I_r)(dz)
```

Specify whether `theta_ref` is pretrained on Task A, independently supplied, or fitted
on Task B. A Task-B anchor or basis defining the prior is empirical Bayes/data reuse;
label it, use a held-out selection split or cross-fitting when evaluating that choice,
and keep the ordinary Task-A-prior experiment as a separate arm. Temperature `lambda`
changes the target; it is not a sampler step size. `lambda = 1` is the likelihood target.

Use the exact coordinate gradient `B^T C_half^T grad_theta Phi`. Candidate bases are
training-trajectory PCA, prior-whitened Gauss–Newton/Fisher directions and a random
orthonormal control at matched rank. A raw loss Hessian may be indefinite; state which
curvature operator and prior metric define “informed”. Compare the tangent construction
with this affine one only after specifying the chart prior and volume factor. Do not
silently interpret a global unit-sphere prior as a product of tensor-wise spheres.

**Development:** add `src/subspace.rs` for basis validation, maps and projected gradients;
`src/target.rs` for target identity and shared full-data potential; and a proposed
`examples/subspace_adaptation.rs` using the existing model/task machinery. Implement
exact MH-pCN in `z` first:

```text
z' = sqrt(1-beta^2) z + beta xi,    xi ~ N(0,I_r)
log alpha = min(0, Phi(z) - Phi(z'))
```

Use the common full-data target at both ends. The Gaussian prior cancellation follows
from this proposal's prior reversibility; it does not hold for an arbitrary Langevin
proposal without its drift-density correction. Validate ranks 1–2 against quadrature
on controlled targets before measuring transformer adaptation. Ranks 4/8/16/32 are an
initial experiment grid, not ranks established by the budget estimate.

The motivation for distinguishing informed directions from the Gaussian complement is
the [DILI construction](https://arxiv.org/abs/1411.3688), whose Gaussian-reference
assumptions must be checked for this target. It does not guarantee useful transformer BMA.

*Status (2026-11-09):* the chart (`src/subspace.rs`), the exact MH-pCN kernel in `z` with
its quadrature check (`src/target.rs`) and `examples/subspace_adaptation.rs` are in — see
`docs/PCNL_STATUS.md` "R1". The cells now run on the **ambient anchor** through shared
code: `experiment::AnchorSetup`/`anchor_hash` are used by both the ambient run and the
example, and a test asserts the hash, so the two tables are on the same `θ_ref`
(`0x52d503556818f1e0`, 4.30 % / NLL 4.1523 on B). Measured at ranks 1–2 with 4 chains ×
4 000 steps: the informed (Task-B trajectory PCA) basis gains 1.21–1.24 nat over the anchor
and the random control gains nothing (identical accuracy, NLL within 0.001) — but the
anchor is *below chance* (12.5 %) and worse than the uniform predictor (`ln 8 = 2.0794`), so
most of that gain is undoing the anchor's bias; the best restricted predictive (rank-2
informed, 14.06 % / 2.9083) is at the base rate with a miscalibrated tail. No cell is
converged at `β = 0.05`, which is R2's scale problem. Brier and the predictive MCSE are now
reported. Open: ranks 4–32, the cross-fitted empirical-Bayes arm, ambient-vs-restricted
budget parity, and an anchor that isn't below chance on B.

**Acceptance:** known-target posterior means/covariances and predictive integrals agree;
random and informed rank controls use identical target scale, baseline and budgets; report
accuracy, NLL, Brier, predictive MCSE and mixing for every rank. An informed basis must
beat or explain its tie with a random basis, not just beat a failed ambient run.

## R2 — separate target restriction, full-target proposals and SG bias

Two different extensions follow R1:

* **Restricted adaptation:** freeze the complement and sample `pi_r`. This is the explicit
  smaller model above. Measure its posterior predictive and compare its point optimizer.
* **Full-target informed proposals:** in whitened coordinates split
  `v = B z + B_perp q`; use informed proposals for `z` and prior-preserving pCN for `q`,
  with the full likelihood and correct acceptance ratios. This changes mixing while
  retaining the declared ambient posterior. If that posterior predicts the base rate,
  accurately sampling it is a negative target result, not a failed kernel result.

The prior-preserving proposal principle is from
[Cotter et al.](https://arxiv.org/abs/1202.0709); an infinite-dimensional claim additionally
requires compatible discretizations and trace-class covariance. Diagonal/block covariance
at finite `P` alone supplies neither condition.

After an exact low-rank reference is available, compare full-gradient unadjusted pCNL,
mini-batch SG-pCNL and SGLD at fixed target, with gradient scaling checked as
`(N/batch) * batch_sum = N * batch_mean`. Vary step size and batch size, track distance
to the reference predictive/moments with uncertainty, and distinguish drift-discretization
bias from stochastic-gradient bias. A clipped gradient is a control and changes the drift;
unconditional acceptance is not “100% MH acceptance”.

**Proposed development:** `src/reference.rs` for exact low-dimensional targets and
`examples/subspace_bias.rs` for shared-budget comparisons. Keep covariance block definitions
and whitening maps explicit. Do not adapt the selected basis while collecting fixed-target
samples unless a valid adaptive-MCMC argument is supplied.

**Acceptance:** measured bias decreases or plateaus with a reported reason as step/batch
controls change; gains are reported as ESS or predictive precision per likelihood/backward
evaluation and per second. Require useful accuracy of the reference target before spending
on a more scalable approximate sampler. No new “dimension independent” claim without
holding the underlying statistical target/discretization family meaningfully comparable.

*Status (2026-11-09):* the first half of R2 is in — per-direction pCN scales on the
restricted target (`mh_pcn_z_scaled`, `curvature_diag`, `DiagScale` in `src/target.rs`,
`--proposal fixed,curv,pilot` in the example; see `docs/PCNL_STATUS.md` "R2"). The kernel
is bitwise equal to `mh_pcn_z` at equal `β_i` and reversible for any `β_i`, so the
preconditioner cannot corrupt the target, only the mixing. Measured, ranks 1–2, ambient
anchor: measured-width (`pilot`) proposals take the informed rank-1 cell from ESS 58 /
`R̂` 1.055 to ESS 680 / `R̂` 1.004 (11× per evaluation) with the same predictive (NLL differs
by 0.0046 at a 0.0002 MCSE, inside the fixed arm's 0.0040), and the random arm from ESS 21 /
`R̂` 1.165 to ESS 1 245–1 899 / `R̂` 1.002
without changing its predictive at all — a converged negative control. Curvature-at-anchor
(`curv`) fails on the informed direction (0.1 % acceptance, ESS 7, NLL 3.6072): at a
nearly-uniform anchor the diagonal reads `β_i ≥ 1`, the cap binds, and the arm takes pCN's
largest step into a steep region. The learned `β_i` span 0.017–0.13 within the rank-2
informed arm, which is the part one scalar `β` cannot express. The rotated extension
(`mh_pcn_z_rotated`, `curvature_hessian`, `RotScale`) is in: exact for any orthogonal `Q`
(identity rotation is bitwise the diagonal kernel; a correlated Gaussian's `H⁻¹` is
recovered to ±0.02 at ESS > 2 000), with a block preconditioner as the special case of a
block-diagonal `Q`. Measured at ranks 2 and 4 it is **not** an improvement: informed
`rotpilot` falls from ESS 152 to 29 at rank 2 (`R̂` 1.012 → 1.096) with the predictive
unchanged (NLL 2.9119 vs 2.9123), `rotcurv` fails like `curv` (`β_i` still clamps at 1,
NLL 3.6051/3.5249), and on the random control the two rotated arms move ESS by less than a
factor of 1.6 either way with the predictive identical to four decimals. The reported
`β_i` ranges nearly match the diagonal arm's, which points at eigenvector noise from a
150-draw pilot covariance rather than at the widths. The full-target half is now in too
(`FullTarget`, `mh_pcn_z_full`, `mh_pcn_w_full`, `adapt_beta` in `src/target.rs`,
`--targets restricted,full` in the example): `π_P ∝ N(0,I_P)(v)·exp(−λN·mean_loss)` with
`v = Bz + w`, `w ⊥ span(B)`, an informed `z` move and a projected-complement `w` move,
each scored with the full likelihood and each `π_P`-invariant, with `β_q` adapted during
burn and frozen. Exactness is tested two ways: a `w`-blind potential makes the `z` stage
bitwise the diagonal kernel, and the coupled Gaussian's `Σ = [[0.6,−0.4],[−0.4,0.6]]` is
recovered at ESS > 2 000. Measured at ranks 1, 2, 4 plus a rank-2 random control (4 ×
4 000, ambient): `π_P` gives NLL 1.83–2.03 / Brier 0.84–0.88 against `π_r`'s 2.87–2.95 /
0.99–1.01, and the *random* basis does just as well on `π_P` (1.97–2.04 / 0.865–0.870)
where it collapses to the anchor on `π_r` (NLL 4.15). So the restriction, not the rank or
the mixing, was the limit, and with full support the informed basis is a `z` jump schedule
only. Cost: two evaluations per step; ESS(Φ) 5–10 and `R̂` up to 3.35, so the ~0.9-NLL gap
(12–20× the MCSE) is the measured claim and the absolute NLL is not converged. Open:
per-coordinate or Hessian-informed `β_q` — the adaptation lands at 0.05–0.08, far below
prior scale, which is where the mixing gain would be — longer full-target chains, a pilot
long enough (or shrunk) to pin the rotated eigenbasis, ranks 8+, `λ ≠ 1` / `σ ≠ 0.35`, and
the SG-bias comparison.

The rank scaling of the pilot arm is now measured too (R2d, `π_r`, 4 × 4 000):
ESS(pilot)/ESS(fixed) is 11.7× at rank 1, 4.9× at rank 2, **0.15×** at rank 4 and
unresolvable at rank 8 (ESS 7 vs 6, `R̂` 1.289 vs 1.931, so neither window is a sample and
the 0.31-NLL gap is not quoted). Dimension alone is not the cause: the random rank-8
control still gains **31×** (28 → 872, `R̂` 1.001, predictive unchanged), and a 1 200-step
pilot at informed rank 8 lifts ESS 6 → **24** and `R̂` 1.931 → 1.091. What binds is the
pilot's 150 estimation draws spread over `r` coordinates — 150 at rank 1, 38 at rank 4, 19
at rank 8 — so the 300-step default is a low-rank budget, and `rotpilot` at ranks 4/8 (ESS
9 / 4) remains worse than the diagonal pilot at both. The length rule is not 150·`r`: rank
8 needs 2 400 steps (300·`r`, ESS 6 → 13 → 24 → **137** at 300/600/1 200/2 400) to reach
`R̂` < 1.05, while at rank 4 600 is already past the knee (ESS 15 → 221).

R2e measured the width question directly with `DiagScale::from_widths_shrunk`
(`--pilot-shrink s`: `log w_i ← mean(log w) + s·(log w_i − mean(log w))`, exact for every
`s`, `s = 1` bitwise the raw pilot). At the unchanged 300-step pilot, rank 4 goes
15 → **399** (`s = 0.5`) → **383** (`s = 0`), so its loss is width *noise*, not magnitude,
and 300 shrunk draws beat 1 200 raw ones (245). At rank 8 `s = 0` gives ESS **154** / `R̂`
**1.010** — better than 2 400 raw draws at 67 % of the cost — while `s = 0.5` is the worst
cell measured (7 / **2.164**), so partial shrinkage is not a safe interpolation and one
`--seed` sweep should check whether that non-monotonicity is a mechanism or pilot luck.
The predictive is flat across all twelve cells (rank-4 NLL 2.8658–2.8699, rank-8 pilot
2.7425–2.8219, rank-4 BMA 12.89 % everywhere), i.e. the effect is diffusion, not bias.

R2f ran that seed sweep (rank 8 at seeds 7 and 8, rank 4 at seed 7) and the size of both
headline numbers was seed luck. At rank 4 the *ordering* replicates — `s = 0.5` best at both
seeds (399 / **264**), `s = 0` next (383 / 152), raw last — but the raw cell is ESS 15 on one
stream and 97 (better than the fixed arm's 84) on another, so "the 300-step pilot is 6.7×
worse at rank 4" is one width vector, and what is robust is that shrinkage removes the
lottery (shrunk ≥ 152, raw 15–97). At rank 8 the non-monotonicity does not replicate: seed 7
gives 9 fixed / 12 raw / 20 (`s = 0.5`) / 20 (`s = 0`), and a fifth cell at seed 8 with
`s = 0.5` gives ESS 6 at `R̂` **3.560** — the intermediate point is not systematically bad
but bimodal, i.e. a coin flip between best and worst at 19 draws per coordinate. So
`--pilot-shrink` should be reported with ≥ 3 seeds and their spread, and rank-8 shrinkage
is a variance reduction that must be validated by its variance. Predictives are flat within
a seed and dominated across seeds (rank-4 fixed NLL 2.6168 at seed 7 vs 2.8699 default), so
only same-seed rows are comparable. The open follow-up is a replicate-averaged width (two
independent pilots, or averaging the pilot's own half-samples) to attack the noise without
collapsing the spread.

R2g implemented that follow-up as `--pilot-reps n` (n independent pilots restarted at the
anchor, widths averaged per coordinate as a geometric mean in log space, budget
`pilot_steps × n`; `n = 1` verified bitwise identical to the old estimator) and it is
**negative**. At rank 4 the 2-replicate average moves ESS 15→14 (default seed), 97→128
(seed 7), 70→140 (seed 8): a real gain on two seeds and nothing on the one where the raw
pilot collapses. At rank 8 it is flat or worse (6→5, 12→7, 5→6). The equal-budget control
says why: at rank 4 default seed a single **600-step** pilot gives 221 while two averaged
300-step pilots give 14, i.e. the pilot's error is **bias** (each 300-step pilot explores
the same narrow patch and its narrow width vector is a stable estimate of a biased
quantity), not the variance averaging can remove. So the next estimator attempt must be a
*scale* correction — inflate the pilot widths by a constant or estimate them from a longer
pilot — not more pilots. R2g also extended the shrinkage ladder to rank 16: fixed 5 /
`s = 1` 5 / `s = 0.5` 6 / `s = 0` 7 ESS with `R̂` 1.828 / 2.490 / 1.545 / 1.440, so the
direction holds but the gain over fixed falls from 3.8× (rank 4) and 22× (rank 8) to 1.4×,
and the fixed arm itself is at `R̂` 1.83 — rank 16 is below the experiment's resolution
floor and no claim about shrinkage above rank 8 should be made without more chains or an
exactness check there.

R2h then ran R2g's own predicted follow-up — a uniform scale correction, `--pilot-scale c`,
which multiplies the shrunk widths by `c > 1` before `DiagScale` — and the prediction fails
in a way that retracts R2g's mechanism. Because `DiagScale` adapts a scalar multiplier on
the joint acceptance rate during burn, and only `m · width_i` reaches the kept window, a
uniform inflation is absorbed: `m` halves to within 3–6 % at `c = 2`, and ESS and every
predictive are unchanged at ranks 4, 8 and 16 (7/7, 154/139, 383/350). At `c = 4` the
absorption is imperfect (7–18 % high), the frozen kernel is left slightly off-target, and
no cell improves; the rank-8 default-seed drop (154 → 32) is inside that rank's seed spread
(20 at seed 7). So the pilot's contribution is the relative width **shape**, never its
overall size — which is why `s = 0` (shape discarded, size from feedback) wins, and why the
600-step-vs-two-300-step gap is a shape difference rather than a width bias. The next
experiment is a shape question, not a width question: a curvature shape evaluated where the
curvature means something, or a shape pooled robustly across replicates, against isotropic
`s = 0` at ranks 4 and 8. `--pilot-scale` stays in the example as a verified null.

R2i ran the pooled half of that: `--pilot-ebayes` fits a one-way random-effects model to
three replicated pilots' log widths and keeps the fraction `τ²/(τ² + sd_i²/n)` of each
coordinate's deviation, with an unweighted three-pilot pool as the control. The pooled shape
loses at every rank, seed and strength — ESS `s = 0 / 0.5 / 1` is 383/362/17 at rank 4,
154/64/8 at rank 8 and 20/16/7 at rank 8/seed 7, monotone in the amount of shape retained
with the optimum at none — and the weighting is not what fails: weighted and unweighted
pools land at the single-pilot ESS (17 vs 16 vs 15, 8 vs 6 vs 6), so the replicates agree
about the shape and the shape is *wrong* rather than noisy. That closes the pilot arm: every
pilot-derived shape measured loses to discarding it, and the only untested shape source is
a non-pilot one — curvature at a point where `∇ℓ ≠ 0`, so the Hessian describes local
stiffness instead of a stationary bowl. Until that is run, "shrink the pilot" means `s = 0`.

R2j then closed the rank-16 question that R2g and R2h had both parked: before running
another rank-16 arm, does the fixed (proposal-independent) arm become measurable with more
chains and more steps? It does not. ESS(Φ) per 1 000 kept draws is 0.45 / 0.16 / 0.56 / 0.48
across 11 200–49 600 draws and 4–8 chains — an integrated autocorrelation time of about
2 000 draws at every budget — and the same kernel at two seeds gives `R̂` 1.302 vs 1.906,
BMA accuracy 17.58 % vs 23.44 % and NLL 2.1226 vs 2.0576. That same-kernel seed spread
exceeds the BMA step (3.5 points) that R2g's rank-16 shrinkage ladder was asked to resolve,
so the ladder is withdrawn as a measurement, "no claim above rank 8" stands, and a rank-16
result would need a target-level change (a shorter-autocorrelation functional, an exact
reference, or a less stiff subspace), not more draws or a better estimator.

R2k then ran R2i's remaining shape source: curvature at the target's **mode** rather than at
the anchor, which is a stationary point of the training loss and not of the posterior.
`--proposal laplace` finds the mode by Adam on the exact gradient of `Φ` (mode found in
1 157/1 132 steps at ranks 4/8, gradient 5.4e1 → 9.9e-4, printed with every cell) and takes
`β_i = m/√H_ii` there. Two findings. First, `loss_and_grad`'s gradient is **not** the
derivative of the loss value: `forward` defaults to `NormGrad::Stopped`, which freezes the
Frobenius denominators, and the gap is ~1 021× on the example's model, so
`SubspaceTarget::grad_phi` uses `NormGrad::Exact` and a new test pins it against finite
differences. Second, and unlike every pilot-derived shape, this shape **beats isotropic**:
rank-4 ESS 554 (`s = 1`) and 587 (`s = 0.5`) at the default seed, 287 and 373 at seed 7,
against isotropic 383 and 152 — with identical BMA accuracy and `R̂` 1.009–1.017. Partial
shrinkage wins at both seeds, so the recommendation is `laplace` with `--pilot-shrink 0.5` at
rank 4. At rank 8 the same estimator collapses to ESS 8/7 (`R̂` 1.56–4.45) against isotropic
154 and fixed 7, even though the rank-8 mode converges to an equally good potential
(1604.91 vs 1602.06): the diagonal Laplace width is a *local* quantity, and a rank-8
restriction contains directions whose posterior is much wider than the mode's Hessian. The
next question is therefore about rank and non-Gaussianity, not about more shape sources —
whether the rank-4 win survives a different seed set, a different anchor, or `λ ≠ 1`, and
whether a full-covariance Laplace (the `rot` kernel) behaves the same way at rank 8.

R2l answered the `λ` and full-covariance halves of that. `λ` moves the isotropic baselines,
so each cell carries its own `pilot s = 0` run: the gain over the best isotropic arm is
+53 % at `λ = 1` (383 → 587), +22.5 % at `λ = 0.5` (342 → 419) and **−6.6 % at `λ = 2`**
(376 → 351). The sign change is explained by mode displacement — the search needs 926 steps
at `λ = 0.5`, 1 508 at `λ = 2` — i.e. R2k's rank-8 mechanism at fixed rank: a shape tuned to
a mode the chain reaches late is no use while it is still approaching. For the covariance
question, R2l added `rotlaplace` (the same converged mode, the full Hessian there, β
eigen-range 6.5/7.6 so the rotation is not degenerate): ESS 112 at rank 4 against the
diagonal's 587, and 6 at rank 8 against the diagonal's 8 and isotropic 154. The full
covariance is therefore *worse* than its diagonal, with `m` adapted down to 0.27–0.38 to say
so, which retires "the rank-8 collapse is the discarded off-diagonal" as an explanation. The
recommendation stands as `laplace --pilot-shrink 0.5` at rank 4 for `λ ≤ 1`, isotropic at
`λ = 2`, and isotropic at ranks 8 and above; the outstanding question is the *anchor*, which
only a shifted `θ_ref` can answer, and whether a mode-centred proposal that also carries the
approach (a tempered or annealed reference) can keep the rank-8 gain.

R2m ran all three of those, and retracted one of R2l's claims. Shifting the anchor by 10 % of
its rms (`--anchor-jitter`, hash printed) leaves the rank-4 win at 2.3× (378 vs `fixed` 165)
instead of the ambient 5.9× (587 vs 100), and collapses the isotropic baseline off-anchor
(ESS 7, `R̂` 10.18), so the effect is real but its *size* is one anchor's. The tempered
reference failed in every parametrization: curvature at `τ · z_mode` is a mild compromise at
rank 4 (428 against the mode's 587) and **invalid** at rank 8, where 1 of 8 coordinates has
`H_ii ≤ 0`, the width floor makes a `1e4`-scale coordinate, and the chain reports 0.6 %
acceptance, ESS 5, `R̂` 17.96 — `1/√H_ii` presupposes positive curvature and interior points of
`Φ` are saddles. Log-space *width* blends avoid that trap but not the outcome: (anchor, mode)
is catastrophic at both ranks (5 / 6, the anchor half carrying `H ≈ 0` widths of `1e4`), and
(pilot, mode) is below its own isotropic baseline everywhere (rank 4: 383 → 352 → 271; rank 8:
154 → 85 → 108). Finally the second seed flips the `λ = 2` sign (+31 %, 172 → 226, with a
search that hit the 2 000-step cap), so that half of R2l is **withdrawn**: `λ = 2` is
unresolved, `λ ∈ {0.5, 1}` is positive at both seeds. The largest lever found so far is not a
proposal dial at all — the anchor moves BMA accuracy 12.89 % → 14.45–15.23 % while every
proposal dial in R2c–R2m moves it ≤ 0.5 points — so varying `θ_ref` systematically is R2n.

R2n swept it: `--anchor-jitter f` for f ∈ {0.05, 0.10, 0.20, 0.30} × two seeds, rank 4,
`λ = 1`, three arms per cell (21 runs), and added `--width-report` to settle the rank-8
width question — which it did by refuting it. BMA accuracy does track the jitter at the
default seed and monotonically so, 12.89 % → 14.45 % → 14.45 % → 16.02 % → 17.97 % (+5.1
points from f = 0 to 0.30), but not at seed 7 (16.80 / 18.75 / 16.02 / 16.80 for `fixed`),
so the anchor is confirmed as the largest lever on the *answer* and its direction is not a
law: it displaces the prior mean, hence the model, and Task-A's pretrain anchor is evidently
not Task B's best. ESS meanwhile is a lottery rather than a dose–response — at the default
seed `pilot` gives 383 / 607 / 7 / 93 / 17 across f = 0 / 0.05 / 0.10 / 0.20 / 0.30 (f = 0.30
collapses all three arms to 9 / 17 / 51 with `R̂` 1.44 for `fixed`), so R2m's f = 0.10
collapse is one draw from a long left tail and f = 0.05 is the sweep's best cell for every
arm. The rank-4 `laplace` gain is real on average — 1.53× / 1.07× / 54× / 4.33× / 3.00× at
the default seed, geomean 2.43× over nine cells, ahead in 7 of 9 and behind at the two
seed-7 cells with f ≥ 0.20 — and unrelated to anchor distance; the 54× is a pilot collapse,
not a laplace improvement. Finally the widths: the rank-8 mode vector is
`[0.2351, 0.2033, 0.1046, 0.0925, 0.1686, 0.1190, 0.0983, 0.2435]` (norm 0.4771) against rank
4's `[0.2276, 0.1151, 0.0915, 0.0693]` (0.2797), i.e. *broader* in norm and *less* spread
(2.5× vs 3.3×), with β 0.054–0.11 vs 0.059–0.10. No ~100× tightening exists at the mode; the
only such span is R2m's rank-8 `τ = 0.5` cell (β 0.0065–1.00), produced by the `H_ii` floor.
Rank 8 is instead hypersensitive to *spread*: uniform β 0.05 → ESS 7, uniform 0.065 → 154,
the mode's 2× spread → 7 at 45.5 % acceptance and `R̂` 4.45, which is a mixing failure, not a
narrow proposal. The rank-8 recommendation (isotropic) stands with a corrected reason; the
anchor stays open, and what it needs next is a sweep of `θ_ref` at a *fixed* seed and basis so
that the anchor's effect on the model can be separated from its effect on the sampler.

R2o did that, and it retracts R2n's reading of the trend. A new `--anchor-seed k` re-draws only
the Gaussian shift, leaving `D_A`, `D_B`, `θ₀`, the basis draw seed and the test set on
`--seed 20261020`; two directions × four magnitudes give eight anchors on one data split (all
printing data hash `0xba104db4f677b2ec`). At fixed data the anchor draw alone moves BMA
accuracy over **11.72–19.92 %** (pilot arm 12.50–19.92, mean 14.55, sd 2.56 over 8 draws) —
8.2 points, against R2n's 5.08-point trend. The per-magnitude means (14.46 / 13.67 / 16.60 /
13.47 %) are not monotone and the within-magnitude gaps (3.91 at f = 0.05, **6.64 at f = 0.20**)
are as large as the trend, so jitter magnitude is not the lever: the specific displaced anchor
is. Nor is it a better prior mean — two anchors with identical unadapted Task-B accuracy
(4.30 %) give 16.41 % and 12.50 %, and the best BMA cell comes from a 3.52 % anchor. ESS shows
the same thing: at fixed data it spans 8–406, f = 0.20 alone gives 194 vs 16, and the f = 0.30
magnitude that collapsed all three arms in R2n is unremarkable on both controlled draws. One
controlled cell reproduces the rank-8 signature found in R2n: `R̂` 1.742 at 49.9 % acceptance.
The rank-4 `laplace` advantage also shrinks under the control — median 1.19×, geomean 1.42×,
range 0.08–25× over the eight anchors, with the 25× being the pilot's collapse and the 0.08 the
laplace's own — so R2k's 5.9× and R2n's 2.43× are both draw-optimistic. Limitation: the
informed basis is PCA of a Task-B path from `θ_ref`, so `--anchor-seed` holds the data and the
basis *seed* but not the basis *vectors*; the next cell needs `--basis-anchor` (compute the
basis at the ambient anchor, jitter only the sampling reference) to split the prior mean from
the subspace.

R2p ran that split. `--basis-anchor` builds the informed PCA path at the ambient anchor, so all
eight cells share one restricted model (identical basis line, `94.030 -> 93.784`) and differ only
in the prior mean. Pinning the basis **fixes the mixing**: pilot ESS geomean 2.95×, median
213 → 428, minimum 16 → 229, max `R̂` 1.742 → 1.036, and the proposal-independent arm improves
identically (median 38 → 125) — so R2o's collapses were bad *targets* created by a moving
subspace, not bad samplers. It does **not** fix the accuracy spread: 6.64–17.58 % (range 10.94,
sd 3.16, versus 7.42 / 2.56 in R2o), still non-monotone in `f`, and the new low cell is worst on
Brier and NLL too, i.e. a real model effect. With one basis the two arms agree in 6/8 cells (max
gap 2.35 vs 4.30) and accuracy tracks the proper scores at both extremes. Mean accuracy drops
0.9 points against R2o (13.67 vs 14.55 %), which quantifies the data-reuse optimism of the
anchor-coupled basis. Open after R2p: the `laplace` arm was not re-run under the control (8 more
cells), a second pinned basis would test whether the accuracy attribution is basis-specific, and
two directions per magnitude still give a range rather than a variance.

R2q closed the laplace half: the same eight anchors and the same pinned basis, proposal
`fixed,laplace --pilot-shrink 0.5`. The `fixed` rows reproduce R2p's ESS and accuracies exactly,
so the table is one target with the proposal swapped. Against the pinned pilot arm the laplace
gains are 1.09, 1.24, 0.96, 1.17, 0.80, 1.70, 1.58, 1.21 (median **1.189×**, geomean **1.188×**,
6/8 wins, every ratio in [0.80, 1.70]) — the *median* is identical to R2o's **1.186×**, but R2o's
geomean **1.42×** and 0.075–25.06× range are gone. The two spectacular R2o cells were pilot
collapses created by the moving basis (ESS 16 and 38); under the pin the same anchors give 362 and
233, so laplace is only 1.70× and 1.58×. So the typical laplace advantage is a property of the
rank-4 target's curvature at its mode, and the tail was the basis. Against the pinned isotropic
arm laplace is 3.04–4.93× (geomean 4.08×) in all eight cells, with no accuracy cost (7/8 cells
identical, mean 13.96 vs 13.67 %), all mode searches converged in 982–1382 steps, and
`H_ii ≥ 0` in 4/4 coordinates. Laplace ESS now varies only 2.2× across anchors (276–617, sd 110)
while accuracy still spans 6.64–17.58 %, so the sampler no longer tracks accuracy at all. Remaining
open: one pinned basis only (would the 6.64 % cell move under another?), two anchors per magnitude
still yield a range rather than a variance, and `rotlaplace` / `--laplace-tau` are retired on
R2l/R2m grounds without ever being re-run under the pin.

R2r built that second pin: `--basis-anchor-seed 1` starts the informed PCA path at an
independently drawn anchor one anchor-rms from ambient (`|θ| 132.394`, basis anchor hash
`0xfe9ad411b8756ef7`, path `132.394 -> 132.789` in all eight cells, versus R2p's
`94.030 -> 93.784`), so the eight experimental anchors again share one basis that is *not* the
ambient one. The accuracy attribution survives: mean 13.77 % (R2p 13.67), sd **3.34** (3.16),
span **9.77** points (10.94), and the worst cell is the same anchor (`f = 0.30, k = 2`: 8.98 vs
6.64 %). The mixing attribution does **not**: pilot ESS falls to 24 / 46 / 141 / 7 / 154 / 6 /
101 / 6 (mean **60.6** vs 400.5, **6.6×** lower), with `R̂` 1.715 / 14.777 / 16.048 in three cells
and ≥ 1.04 in four, and the `fixed` arm is frozen at ESS 5–12. So pinning removes basis *drift* but
does not by itself give a good preconditioner — R2p's ESS gain was the **ambient** basis, which sits
in the target's typical set. The accuracies stay put while the geometry degrades because accuracy is
a coarse discrete average; the proper scores do not (NLL mean **3.59** vs 2.96, 0.63 nats, at ESS
6–154), so R3's readouts must be gated on `R̂`/ESS rather than compared across bases. Open after
R2r: one pin seed and one magnitude (1.0 rms) — a *near* pinned basis would separate "pinned" from
"near" — and the already-pinned `--basis random` arm was not re-run as the stronger contrast.

R2s replaced the two-directions-per-magnitude design with **six** (`--anchor-seed 1..6`, the first
two being R2p's cells, verified by re-running `f = 0.05, k = 1` to the same ESS, `R̂`, accuracies,
Brier and NLL), so the accuracy spread can be a variance rather than a range. Within-magnitude sd
is **0.94 / 2.39 / 1.66 / 3.35** points against per-`f` means of 13.41 / 13.80 / 12.44 / 10.22 %:
**69 %** of the total variance is inside a magnitude (within MS 5.14, between component 1.72,
`F(3,20) = 3.00`, `p = 0.055`). Cell `MCSE(acc)` is 0.019–0.099, so the direction spread is 20–50×
the Monte Carlo error and is a property of the prior mean, not the sampler. Variances are
inhomogeneous and grow with `f` (Brown–Forsythe `F(3,20) = 4.10`, `p = 0.020`; 0.89 / 5.71 / 2.76 /
11.19), and `f = 0.30` splits 6.64/6.64/8.59 vs 12.11/13.28/14.06, so the loss is directional
rather than a smooth penalty on distance: the mean shift from `f = 0.05` to `f = 0.30` is only 3.19
points (Welch `p = 0.067`). 23 of 24 cells still mix (`R̂ ≤ 1.031`); one (`f = 0.30, k = 5`, ESS 9,
`R̂` 1.461) is a mixing failure even under the ambient pin, i.e. ~1 in 24 directions. So R2p's
"the spread is the prior mean" holds only in the sense of the mean's *location*, and its `f = 0.30,
k = 2` worst cell is now a tie with `k = 3`. Open after R2s: which geometric property of the
direction predicts the loss (projection into the rank-4 basis, or predictive-logit spread — neither
recorded); whether the same direction reproduces under another basis or data split; and the
`laplace` arm has still only been swept at two directions per magnitude.

## R3 — proper-score readouts and decision quality (ngpt workstream A)

Reuse [readout.rs](src/readout.rs) and `examples/readout_compare.rs`. Keep the empirical
ensemble identity separate from a posterior claim: averaging available predictions can
lower Brier without those predictions being equilibrium posterior draws.

For independent states, finite-mixture Brier inflation is `tr(Cov(p))/m`. For stationary
correlated states the exact inflation is instead

```text
tr Var(p_hat_m) = [m tr Gamma_0 + 2 sum_{k=1}^{m-1}(m-k) tr Gamma_k] / m^2
```

where `Gamma_k = Cov(p_t, p_{t+k})`. Test both against analytic chains or exact
enumeration. Nonstationary states also carry bias. Random subsets of one finite retained
pool answer a finite-population sampling question, with its own covariance correction.
Do not use the iid formula as a certificate for a correlated trajectory.

Compare point Adam, temperature-scaled Adam, matched-cost independent point ensembles,
single draws and BMA at matched evaluation cost. Tune temperature on development data
only; report NLL/Brier as primary proper scores, accuracy as utility, and ECE/AURC/coverage
as diagnostics. Use paired evaluation inputs and bootstrap at the data/seed level;
chain draws are not independent experiment replicates. Include in-domain, noise and
declared distribution-shift tests.

For the RLCD extension, first verify the reward estimator's target gradient. Averaging
parameter-state predictions changes a rationale-mixture estimator; a leave-one-out baseline
is not automatically unbiased when states/rationales share dependence. Provide a finite
enumeration test of the actual expectation. Distinguish variance reduction conditional
on the label from unconditional reward variance, which includes outcome noise.

**Acceptance:** independent-run predictive uncertainty covers reference values, and any
claimed score gain survives a strong calibrated point baseline with a reported interval.
If a usable target's chain readout loses to that baseline at equal cost on three seeds,
record the negative result and stop RL integrations pending a new hypothesis.

R7 consumes this section directly rather than restating it: a DT step score is the same
`Rule` applied with `y` the observed action, `Enumerated`/`mean_dist` give the per-step
predictive and its covariance term, and the correlated-Brier inflation formula above is the
correct correction for steps inside one trajectory. Accuracy stays a utility; for a
continuous action head it does not exist at all, and the score *is* the report.

## R4 — add a learnable structured task before external model scaling

Keep modular arithmetic as a regression fixture, including Task B = A and label-shift
controls. `Dataset::materialize` draws from only `K^2` possible input pairs, so larger
`N` mostly changes repeated-example likelihood weight. Add an exhaustive input census,
pair-overlap report, and separate protocols for repeated observations, disjoint-pair
generalization and task shift. Increasing repeats is not evidence of more diverse data.

Introduce a small CPU-compatible task with known signal/noise and an informative point
solution: first teacher–student classification or a tiny sequence task; then a real
tokenizer/batching/checkpoint task only after the interface exists. Compare the same train,
development and test splits across all inference arms. Add label noise/OOD perturbations
with a specified generative process. Establish that the optimizer can solve each task
before diagnosing a posterior or sampler failure. Do not relabel failure on one task as
a general theorem about standard attention or transformer Bayesian adaptation.

**Proposed development:** a task/evaluation trait beside `src/task.rs`, shared likelihood
and minibatch scaling, and one deterministic benchmark fixture. Measure acquisition,
retention on Task A, calibration and task-B accuracy. GPU/checkpoint work needs a separately
budgeted experiment; it is not needed for R0–R3.

R7 is where the "tiny sequence task" above becomes concrete, and it exposes a limit of the
current data model rather than a missing fixture: `Dataset` is a flat list of `(tokens, label)`
pairs drawn from `K^2` inputs, with no trajectory axis, no per-example length and no
split-by-group. A DT corpus needs all three, so R7's task contract (R7.4) is the input to this
milestone and not a duplicate of it.

## R5 — gated extensions of ngpt §9

These workstreams follow an informative target and audited readout. They are optional
branches with their own falsifiers, not prerequisites for finishing the core crate.

| Branch | Minimum deliverable | Advance only if |
|---|---|---|
| B: latent link | Direction/radius interface, per-mode scores | Toy passes; real link available |
| C: reused depth | Shared-parameter map, dense baseline | Protocol and training code exist |
| D: hierarchy | Level scorer and schedule contract | Task/level interface exists |

**B — HybridLink:** consume the sphere-link prototype as an external reference, not an
already integrated Candle model. Compare raw, normalized and radius-bounded links under
matched data/training; measure task utility and per-mode NLL/Brier/ECE. Include learned
temperature/radius controls. Normalization may limit scale drift without improving
calibration. A real integration also requires rollout replay and model compatibility.
An inspector can show predictive disagreement, but that disagreement is posterior
uncertainty only after the sampling/target gates pass.

**C — reused depth:** compare shared and dense models at both matched unique-parameter
count and matched forward/backward compute, reporting the mismatch when both cannot
be matched. Preserve shared-variable identity in flattening and basis construction.
Measure ICL/IWL and head diagnostics per retained state. A smaller `P` can also change
representation capacity and priors. Growing depth changes dimension: copying a posterior
sample is not a reversible-jump kernel; define an explicit new-model pushforward and
re-equilibration, or postpone posterior-preserving dimension changes.

**D — coarse-to-fine:** import level schedules/metrics as data, not NCM training as a
dependency. Validate the level semantics before computing capacity: `length * log K`
is a per-level code ceiling, and cumulative versus residual/innovation codes have different
joint accounting. It is not automatically comparable to a weight-space Haar cap cost.
Measure actual conditional entropy and held-out conditional NLL per level; pre-register
the proposed link to localization, rather than asserting the first ceiling crossing
predicts collapse. Compare flat and block schedules with the same target and frozen
blocks eventually updated; if fine blocks stay fixed, label a restricted target.
For Sudoku use solved/cell/valid-grid metrics, hold out puzzles and avoid full-solution
leakage into the conditioning interface. Start with a tiny structured synthetic hierarchy.

External training runs and GPU reproduction remain blocked on their actual dependencies
until inspected at execution time. Do not inherit an old “host has no torch/GPU” statement
as a permanent environment fact. Keep external benchmark claims in their source records.

## R6 — formal contracts and claims discipline

Read `../docs/plan_bayes_ngpt.md` §7 and ngpt §6.6 against the actual sibling Lean files.
Name projection-domain, covariance and architecture hypotheses; a numerical test is not
a proof, and a theorem about an ideal projection is not a theorem about detached autodiff.
The original [nGPT paper](https://arxiv.org/abs/2410.01131) motivates hyperspherical
representations; it does not certify this nGPT-lite kernel or its stochastic gradients.

Prioritize bridge statements with direct acceptance tests: projected-gradient chain rule;
Gaussian subspace pushforward; prior reversibility of pCN; correlated Brier decomposition;
and the finite-floor schedule distinction. Only then consider a rigorous cap-odds bound
with its global likelihood assumptions. Match the implemented epsilon map and parameter
domain explicitly. Keep exact density algebra, floating-point tests and empirical mixing
claims in separate evidence categories.

Faris–Lavine, nested-Fock field theory and Hashimoto solver guarantees are not required
to validate this statistical crate. A proposed bridge to those topics needs its own
operator/domain statement; do not infer it from a Lipschitz/noise theorem. MM0 export or
nanoda checking is a separate proof-artifact task, not evidence of sampler stationarity.

## R7 — decision transformers: the predictive is a policy (application track)

**Hypothesis.** A posterior over a small DT's weights yields a predictive action distribution
`p̄(a | o, g)` that is better calibrated and lower-NLL than a tuned point DT at matched
likelihood-evaluation cost, and the induced policy — sampling from `p̄` or decoding with the
averaged distribution — retains the behaviour policy's return without giving up more than its
return MCSE. Both halves are measured; neither is assumed from the classification results.

**Current status (source review, 2026-11-15): implemented and measured within the recorded
cells.** R7.3's model, potential and readout paths exist, as do `dt_arm` and `dt_subspace`.
`PCNL_STATUS.md` records the R7 rank-4 comparison and subsequent basis/anchor follow-ups.
The NLL promotion criterion was not established and that application branch remains stopped;
proper-score-only observations have their own scope. R7.3–R7.6 below retain their dated
protocol/development history; use active R8 and development D5 for the remaining uncertainty
and decoder work. This review itself reruns no cell.

### R7.0 What the accumulated record says about this direction (review)

Three measured results constrain the DT design more than any new proposal idea:

1. **The prior mean, not the proposal, moves the answer.** R2n–R2s: 69 % of the accuracy
   variance sits *within* a jitter magnitude, variance grows with magnitude while means differ
   only at `p = 0.055`, the single failure in 24 directions is a mixing failure under the
   ambient pin, and 1.19× is the median `laplace` gain over the pinned pilot arm (R2q). For a DT
   the prior mean is a *behaviour corpus* and a pretraining split, so the first DT experiment
   must vary that axis (R7.4's adaptation corpus), not the preconditioner.
2. **The restriction, not the mixing, was the limit.** R2's `π_r` cells lose ~0.9 NLL to the
   full target `π_P` and the random basis does equally well on `π_P`. A DT's informative
   directions plausibly live in the action head and the return-to-go embedding, not in the
   trunk, so a rank-4 restriction over the whole flat vector is the most likely way to destroy
   the model. Declare the basis target (head-only, trunk-only, all-blocks) before the sweep.
3. **Proposal gains are target-specific and do not transfer.** `laplace` at rank 4 beat
   isotropic in all eight pinned cells yet collapsed at rank 8 (R2k), off-anchor (R2m) and
   failed for the full covariance (R2l); a non-ambient pinned basis degraded ESS 6.6× with the
   scores nearly unchanged (R2r). Inheriting `laplace` as the DT default would be an untested
   transplant, and the DT adds blocks (head, RTG embedding, layer norms) whose curvature
   differs by orders of magnitude from the trunk's.

### R7.1 The target

For trajectory `j` of length `T_j` with observations `o_{j,t}`, actions `a_{j,t}` and
return-to-go `g_{j,t} = Σ_{k≥t} r_{j,k}`:

```text
theta(z)  = theta_ref + C_half B z
theta_ref  = anchor fitted on corpus A (pretrain); adaptation data = corpus B, split by trajectory
Phi(θ)    = λ Σ_j Σ_{t=1..T_j} −log p_θ(a_{j,t} | o_{j,≤t}, a_{j,<t}, g_{j,≤t})
π_r(dz)   ∝ exp[−Phi(θ(z))] N(0, I_r)(dz)
```

The kernel, prior, chart and dimension budget are R1's, unchanged. What has to be declared
because the classification target never forced it:

* **The likelihood is behaviour cloning, not optimality.** `g` is a covariate copied from the
  data; `p_θ(a | o, g)` is trained to reproduce the actions that *were* taken at that return
  level. A posterior over `θ` is a posterior over a calibrated behavioural model of the corpus.
  No claim about the optimal policy follows, and "return-conditioned" is not a likelihood for
  "this action is optimal at return `g`".
* **What `N` counts.** `Φ` sums per transition, but the independent unit is the trajectory.
  Report trajectories, mean `T`, total transitions, and the ratio of `Φ`'s spread over prior
  draws to the classifier cells' — a 10^4-transition corpus with `P ≈ 2·10^4` is far stiffer
  than `N_train = 512`, so `λ = 1` is a different effective concentration here and must be
  stated rather than inherited.
* **Per-step distribution family.** Discrete actions reuse `model.rs`'s categorical
  cross-entropy exactly, so `readout.rs::Rule` applies per step. A Gaussian head `(μ, log σ)`
  needs a declared floor on `σ` and a check that no prior or posterior draw has a non-finite or
  degenerate per-step NLL; an unbounded optimum in the output parameters is a target defect
  that a sampler cannot fix, and it breaks the coherence between `Phi`, the BMA and the ESS
  diagnostic before any mixing claim is made.
* **One posterior or one per return level.** A single posterior over the mixed corpus and a
  stratified posterior per `g`-bin are different models; BMA over the first is not the
  conditional predictive for a target return. Every cell names which one it is.

### R7.2 Why the predictive is not the answer

`p̄(a | o, g) = ∫ p(a | o, g, θ) Π(dθ)` is a distribution, and a DT decodes a policy from it. Two
non-identities matter:

```text
E[return | π̄]  ≠  E_{θ~Π}[ E[return | π_θ] ]        (return is a functional of whole trajectories)
sample from p̄  ≠  decode with p̄                            (a per-step draw vs a mean action)
```

Averaging over posterior states changes the state distribution each subsequent decision is made
in, so the BMA's Brier/NLL advantage (a per-step statement) does not imply a return advantage.
Report four readouts separately and never merge them: per-step action NLL/Brier/ECE of `p̄`;
return of a policy sampled from `p̄`; return of the averaged-distribution decoder; return of a
single posterior state (the point readout). Their differences are decision effects, not score
effects.

### R7.3 Development, in order

**Historical protocol:** the implementation/status paragraphs later in this section and
`PCNL_STATUS.md` supersede the “currently typed”, “proposed” and “still missing” language
in the original task list. Reuse the landed code. Active follow-up is R8/D0–D7.

1. **`Potential` trait in `src/target.rs`.** `SubspaceTarget` is currently typed to
   `(Dataset, Shapes, Variant)`, so no other likelihood can sit under the subspace machinery.
   Extract `log_phi(θ)`, `grad_phi(θ)` and `n_terms`, make `SubspaceTarget` generic over it, and
   keep the classification impl as a thin wrapper. Test: the trait path reproduces the existing
   pinned-cell potential **bitwise** at a fixed `θ` and data hash, so R0–R2 evidence is not
   disturbed by the refactor.
2. **`src/policy_model.rs`.** Causal DT forward over tokenized `(g, o, a)` with a padding mask,
   returning per-step logits (or `(μ, log σ)`) and a summed per-step NLL with both `NormGrad`
   conventions. Own `Shapes`/`layout`/`flat_size` with the new blocks (RTG-bin embedding,
   position, action head, optional `σ` head, LN affine), so `unflatten`, `init_flat` and
   `anchor_hash` keep working. Shapes are an initial grid, not budget-derived: `d = 32`,
   `ff = 64`, 2 layers, ≤ 8 RTG bins, ≤ 10 actions, `T ≤ 16` gives `P ≈ 1.7·10^4`, the same
   order as the classification cells so the sampler budgets carry over. `Variant::Standard`
   first (the anchor-comparable baseline); the cosine variant is a second arm with its own
   formal-claims tests, not a default.
3. **`src/mdp.rs`.** One synthetic MDP with an exactly computable optimum: 2-D linear-Gaussian
   dynamics, 5 discrete actions, horizon 16, 4 RTG bins, a behaviour policy with declared
   exploration (`ε = 0.2`), 96 train / 32 dev / 32 test *trajectories*. The exact optimal
   return comes from value iteration (or the LQR solution); the behaviour policy's return is the
   second reference. Splits are by trajectory seed; a step-level split leaks across time.
4. **`src/policy_readout.rs`.** Per-step scores (reusing `readout.rs` for discrete actions),
   ECE of `p̄`'s max probability, and the return-level readout: `K` rollouts of a policy sampled
   from `p̄`, `K` of the averaged decoder, paired differences against each baseline with their
   own MCSE. Return MCSE is reported next to every return number.
5. **`examples/dt_adaptation.rs`.** Same CLI as `subspace_adaptation` (`--anchor-*`,
   `--basis-anchor*`, `--proposal`, `--pilot-shrink`, `--targets`) plus `--traj`, `--horizon`,
   `--rtg-bins`, `--head softmax|gauss`, so every measured arm is a flag rather than a fork.
   `experiment::AnchorSetup` is classifier-typed (`data_a`, `data_b`, `theta0`); parameterize it
   on the potential and pretrain step rather than duplicating it, so the anchor hash keeps its
   one implementation and the DT anchor is comparable across cells.
6. **R0 checks for the new model, before any adaptation cell:** directional finite differences
   on the DT at ≥ 3 norm scales; the `Exact` vs `Stopped` gap measured (R2k found 1 021× on the
   classifier, and a DT's normalised blocks will differ); mask invariance (a padded batch equals
   the unpadded sequence within tolerance); Gaussian-head density integrates to 1 per action
   dimension; the minibatch scaling identity per *sequence*, not per token.

**Status (2026-11-12).** Items 1–3 exist as code with unit tests: `Potential` /
`ClassifierPotential` with a bitwise seam test at the ambient `D_B` hash (`0xba104db4f677b2ec`),
`policy_model` (layout, causal forward, per-step logits, summed NLL, `DtPotential`), and `mdp`
(value iteration, refinement gap, behaviour corpora, trajectory splits, DT tokenisation).
`DtSpec::bin` is `[rtg_lo, 0]`-clipped and `mdp::MdpSolution::optimal_value` is the interpolated
Bellman optimum rather than `interp(V_0)`, because the two differ by ~0.12 out of a −1.99 return
off-grid and only the first matches the policy that is rolled out.

Items 4–5 exist too, and they are the readout rather than a result: `policy_readout` has
per-step proper scores under all three `Rule`s (its log column equals `dt_loss` to 1e-5 on the same
batch, pinned by `dt_step_scores_equals_dt_loss`), max-probability calibration with per-bin counts,
and arm-level return readouts with paired MCSEs. Two negative results are pinned rather than
hidden: max-prob ECE is *not* a proper score (its own doc says so, and the overconfidence test
asserts the ECE identity instead of a direction), and on this MDP the common-random-numbers pairing
buys almost nothing — the arms' trajectories diverge, so the measured covariance is near zero
(paired SE 1.31 vs independent 1.25 at the default arms), which is why variance reduction is
tested on controlled synthetic data and only *published* for the MDP.

Item 6 is now measured (`examples/dt_gradient_gap`, seed 7, 480 parameters, 96 transitions):
under `Variant::Standard` the Exact and Stopped derivatives are the same tensor
(`‖g_e‖/‖g_s‖ = 1.0000`, no Frobenius normalisation in the forward pass), while under
`Variant::Cosine` the exact gradient matches a central difference to 7.4e-6–6e-4 at `h = 1e-2`
across `‖θ‖` of 0.22 / 2.18 / 21.8 and the stopped gradient is off by 2.2e-2–4.0e-2 in the same
direction — with gradient norms agreeing to four digits (ratio 0.9997–0.9999). The gap is a
rotation of the update, not a rescaling: a norm check would have declared the conventions
identical.

**The readout half of R7.4 now exists and has run (`examples/dt_cell`, seed 11).** `dt_arm::DtCell`
assembles the ambient pieces — the `PotentialAnchor` anchor (hash `79f9638aa261e3e5`), two behaviour
policies over the same MDP, unadjusted SG-pCNL chains at the full DT target — and reports the three
columns R7.4 asks for: per-step NLL/Brier/spherical, max-prob calibration with per-bin counts, and
paired returns for behaviour, anchor(sample, argmax), `p̄`(sample, argmax) and the exact optimum.
Measured on test (128 transitions): anchor NLL 1.5925 / Brier 0.7858 / top-1 0.344 / ECE 0.215 vs
`p̄` 1.1056 / 0.5434 / 0.680 / 0.177. Returns at `x0 = [1, 0.5]`, 96 rollouts: behaviour −6.09 ±
0.34, anchor-sample −3.64 ± 0.13, anchor-argmax −2.54 ± 0.03, `p̄`-sample −7.37 ± 0.49, optimum
−2.42 ± 0.02; paired vs behaviour +2.45 ± 0.34, +3.55 ± 0.34, −1.28 ± 0.52 (resolved) and
+0.47 ± 0.33 (within noise) for `p̄`-argmax.

That is the plan's **calibration-only** branch: the predictive improves every per-step score and
top-1 match while its sampled return is *worse* than the behaviour policy's. The same run shows why:
`P = 17 408` and `σ√P = 46.2` against a reached radius of 49.4, so the full-target draws span the
entire prior shell and `p̄` averages over policies that differ far more than the likelihood does.
R7.4's rank-4 restriction is what makes `σ = 0.35` a meaningful posterior width. The Adam point DT,
the temperature-tuned baseline, the matched-evaluation ensemble and the informed basis are still
missing, so this is readout wiring plus a screening observation — not the R7.4 comparison, and every
cell in R7.4–R7.5 remains open with no ledger row owed.

### R7.4 First measured cell (no new dials)

The smallest cell that can falsify the hypothesis. Protocol otherwise R2q's, so the sampler
columns are comparable with the classification tables even though the potential is not:

| Item | Value |
|---|---|
| Model | `d = 32`, `ff = 64`, 2 layers, 5 actions, 4 RTG bins, `T = 16` |
| Anchor | DT pretrained on the Task-A corpus, anchor hash printed |
| Adaptation corpus | same MDP, **different behaviour policy** |
| Rank / λ / prior | 4 / 1.0 / `N(0, σ²)` with the existing σ = 0.35 |
| Basis | informed, pinned at the ambient anchor (`--basis-anchor`) |
| Proposals | `fixed,pilot --pilot-shrink 0` and `laplace --pilot-shrink 0.5` |
| Chains | 4 × 4 000, burn 500, thin 5 (screening: 1 × 1 000 at 2 seeds) |
| Baselines | point DT (Adam, same budget); temperature-scaled point DT tuned on dev; 4-member ensemble at matched likelihood evaluations; behaviour policy; exact optimum |
| Readouts | per-step NLL, Brier, top-1 action match, ECE; return of sampled and averaged policies, with MCSE; ESS/R̂ on mean step NLL |

Budget: the existing CPU-minute caps stay (30 min screening, 4 h promoted), and each cell also
records transition-forwards so cells of different `T` are comparable. A cell is 4 × 4 000 × 2
evaluations × 8 sequences × 16 steps ≈ 8·10^6 transition-forwards at full-target cost; screen at
1 chain × 1 000 steps ≈ 2.5·10^5 and promote only cells whose paired difference is already
larger than the screening MCSE.

**Status (2026-11-13): the cell has run and lands on R7.5's negative branch.**  Implemented as
`pcnl/src/dt_subspace.rs` + `examples/dt_subspace` (the ambient `dt_arm`/`dt_cell` readout is
unchanged).  Deviations to read first: the corpus is `DtCell`'s 768/128/128 transitions, not the
96/32/32 in the table above (changing the split changes the ambient anchor); screening at `1 ×
1 000` reaches ESS 1.4–40 and is directional only, so the promoted cell was run at 4 × **6 000**
(pilot ESS 335.3 / 377.1 / 537.1 at 4 000).  Promoted, seeds 7/8/9: informed `pilot` passes every gate (ESS
590.9 / 792.4 / 858.8, `R̂` 1.004 / 1.003 / 1.002), informed `fixed` fails all three (142.4 / 89.7 /
121.0, `R̂` 1.014–1.045).  Paired test NLL against the dev-temperature-tuned point DT: **+0.0079 ±
0.0182, −0.0171 ± 0.0455, +0.0814 ± 0.0563** — a tie at all three seeds, no seed past 2σ either
way; Brier and spherical favour `p̄` at seed 8 only (−3.6σ, −4.0σ).  The informed basis does beat
the matched random control by 0.35–0.57 nats, and `pilot` beats `fixed` by 0.009–0.022 nats
at screening and by the whole ESS gap at promotion, so the
mixing story is intact while the accuracy claim is not.  `laplace` at rank 4 (the R2k winner on the
classifier) collapses at this target — mode search capped with `φ` still falling, 2–4 of 4
coordinates with `H_ii ≤ 0`, ESS 1.5–2.3 informed — so it is recorded as a proposer defect and the
"transfer of R2k's rank-4 `laplace` win" question is answered *no*.  Returns are seed-inconsistent
(`p̄`-argmax +0.55 ± 0.25 vs behaviour on seed 9, −0.82 ± 0.27 on seed 8) and are not used.  Per
R7.5 the DT adaptation arm stops here pending a *target* hypothesis (R7.6's basis target or
anchor-as-behaviour-policy axes, not a sampler knob).  Tables, gates, defect list and the exact
commands: `docs/PCNL_STATUS.md` §R7.

### R7.5 Acceptance and falsifiers

* **Positive:** `p̄`'s held-out action NLL beats the temperature-tuned point DT by more than 2×
  the paired MCSE on ≥ 3 trajectory seeds, at `R̂ ≤ 1.01` and ESS ≥ 400 on mean step NLL, with the
  sampled and averaged policy returns each within the point baseline's return MCSE. Report the
  per-step and return halves separately even when both pass.
* **Negative:** if the per-step score ties or loses to the calibrated point baseline at matched
  likelihood evaluations, record it and stop the DT adaptation arm pending a *target* hypothesis
  (a different corpus, basis target or restriction), not a new sampler knob.
* **Calibration-only:** if per-step scores improve and both rollout returns do not (beyond their
  MCSE), that is the claim — better next-action distributions, no decision improvement.
* **Sampling gate:** a DT cell failing R̂/ESS is reported as a target/mixing limit with no score
  quoted as a posterior result, exactly as R2r's non-ambient pin was.
* **Target defect:** any degenerate or non-finite per-step NLL among prior or posterior draws
  invalidates that head's target; fix the parametrisation before running chains.
* **Not claimed:** D4RL-style returns (only the synthetic MDP's return is exact here); a posterior
  over optimality; transfer of R2k's rank-4 `laplace` win before the DT's own rank sweep; anything
  about GPU-scale checkpoints, which need the separate checkpoint surface below.

### R7.6 Open questions this section deliberately leaves

* **Basis target.** Head-only, RTG-embedding-only, trunk-only or all-blocks informed directions,
  against a random control at matched rank. R2's `π_r`/`π_P` result says the restriction is what
  costs; the DT's block structure is where that cost is most likely to be visible.  **Measured
  (2026-11-13, `--bases head,trunk,head-random,trunk-random`, 4 × 6 000, seeds 7/8/9):** no block
  restriction beats the full-space informed basis.  NLL cost vs informed is `+0.183 / +0.028 /
  +0.015` for head-only and `+0.009 / +0.042 / +0.070` for trunk-only, so the informed content sits
  in the **trunk** and the head adds nothing at rank 4 (head-only is 4.0σ worse than the point
  baseline at seed 7, never better on any seed).  The informed-over-control gap survives the block
  matching — six of six block informed arms beat an in-block random control by 0.27–0.57 nats — so
  the gain is the directions, not the coordinates.  Two in-block controls fail the sampling gate
  (ESS 236 / 318), which is reported as a gate failure, not a score.  The **RTG table alone**
  (`Block::Rtg`, 128 coordinates) is not a basis target: it costs +0.540 / +0.436 / +0.340 nats
  against the full-space informed basis, is +4.9…+7.0σ worse than the point baseline, and its
  informed-over-control gap is only +0.010…+0.024 nats against +0.342 (head) and +0.566 (trunk) —
  so the content is in the trunk and not in the return embedding, which leaves the state encoder
  and the per-layer products as the only untested part of the trunk.  Detail:
  `docs/PCNL_STATUS.md` `## R7`, Results 89–95.
* **Block preconditioning.** The per-layer blocks of `RotScale` were never the winning arm for the
  classifier (R2l), but a DT's head block and trunk block differ in curvature by far more than
  the classifier's coordinates do, so a block-diagonal `β` is a live arm here and not a retread.
* **Correlated step scores.** Within-trajectory correlation deflates the effective number of
  scored steps; R3's `Γ_k` formula is the correction, and the DT is the first target where it is
  not optional. Whether the deflation behaves like R2j's ~2 000-draw autocorrelation is measurable
  from the same chains.
* **Continuous actions.** Gaussian head with a `σ` floor, a quantile head, or two-hot binning;
  separately budgeted, after the discrete cell.
* **Joint vs stratified return posterior**, and whether a stratified one is any better than
  conditioning at decode time (which is what the DT already does).
* **Anchor as behaviour policy.** R2n–R2s measured an anchor drawn as a Gaussian shift. A DT
  anchor is a corpus, and the analogue of "anchor direction" is which behaviour policy and which
  return distribution it was fitted on — the axis to sweep first, per R7.0 item 1.  **Measured
  (2026-11-13, `--anchor-feedback/--anchor-epsilon`, 4 × 6 000, seeds 7/8/9):** the axis is first
  order for the prior mean (anchor NLL 0.936–2.290, a 0.69–1.35-nat span across policies) and a
  null for the posterior (`p̄` spans 0.010–0.034 nats; every policy within MCSE of the default).
  The best anchor is the one drawn from the *adaptation* behaviour's own feedback (0.3), 0.26–0.59
  nats below the default — a corpus-match effect, not a quality one, since the sharper policy is the
  worst anchor.  No policy reaches R7.5's 2σ bar (best cell: seed 8, −1.7σ, Brier −2.6σ), so this
  axis does not reopen the arm.  Detail: `docs/PCNL_STATUS.md` `## R7`, Results 91–93.
  **Replication (fresh seeds 10/11/12, with the default anchor as control, six seeds per policy):**
  the seed-8 log-NLL win does **not** replicate — pooled over six seeds the weak anchor gives
  log −0.0130 ± 0.0083 (−1.6σ) versus +0.0220 ± 0.0139 (+1.6σ) for the default — but the
  **proper-rule** improvement does: Brier −0.0214 ± 0.0063 (−3.4σ) and spherical −0.0162 ± 0.0044
  (−3.7σ), holding without the one cell that fails `R̂` (seed 12, 1.012).  That is R7.5's
  calibration-only branch: better next-action distributions, no likelihood win, so the arm stays
  closed.  Detail: Results 96–97.
* **Checkpoint surface.** Loading a real (non-toy) DT needs a declared layout/dtype hash and a
  manifest entry; GPU and checkpoint reproduction stay blocked on their own dependencies and are
  not part of the first cell.

## Milestones, budgets and stop rules

Roles denote ownership to assign, not agents spawned by this planning task.

| Milestone | Dependency | Owner role | Timebox | Review artifact |
|---|---|---|---|---|
| M0: trustworthy baseline | none | numerics/experiment | 1–2 weeks | R0 fixes and manifests |
| M1: subspace reference | M0 | inference | 1 week | Known-target tests and rank sweep |
| M2: target comparison | M1 | research | 1–2 weeks | Random/informed/point comparison |
| M3: bias and BMA | M2 | inference/evaluation | 1–2 weeks | Bias curves and predictive MCSE |
| M4: structured task | M0, then M2–M3 | model/data | 1–2 weeks | Data contract and paired report |
| M5: external branch | M3–M4 | integration | scoped later | Falsifiable branch result |
| M6: Lean bridges | M0–M3 | formalization | scoped later | Compiled hypothesis audit |
| M7: DT policy target | M0, M3; R4 data contract | inference/model | 2–3 weeks | Synthetic MDP + per-step scores + return MCSE |
| M8: DT return posterior | M7 | research | scoped later | Joint-vs-stratified comparison |

Timeboxes are planning estimates; hardware and research failures can change them.
For M2 use ranks 4/8/16/32, three task regimes and three paired seeds as a screening grid;
prune failing cells rather than multiply every optional arm. Add at least four independent
stationary chains for promoted cells. Record a screening cap of 30 CPU-minutes per cell
and an initial promoted-cell cap of 4 CPU-hours; exceeding either produces an inconclusive
result unless a revised budget and expected precision are recorded first.

Operational mixing gates are rank/folded Rhat below 1.01, bulk/tail ESS at least 400 for
declared key observables, no frozen/nonfinite traces, and predictive MCSE smaller than
the effect being claimed. These are investigation thresholds, not convergence proofs.
Monitor train potential, radius, informed coordinates, random complement projections and
predictions; loss Rhat alone is insufficient. Report cells that fail the gate without
presenting their predictive as a converged posterior result. Validate any special
finite-horizon Rhat null without using it to waive failed predictive-precision checks.

Stop or redirect when: R0 exposes a changed gradient/model (regenerate affected evidence);
informed and random subspaces tie (revisit basis hypothesis); exact low-rank targets
cannot match calibrated point utility (revisit the model/prior before SG engineering);
or approximate sampling cannot meet precision per compute (retain the exact small-model
result). A well-sampled weak posterior is a target failure; a useful frozen trajectory
is not a sampling success. Publish negative and inconclusive cells alongside positive ones.

For R7 specifically: stop the DT arm if the per-step proper score loses to the calibrated
point DT at matched evaluations (R7.5); do not spend on return-level comparisons until the
per-step score passes a sampling gate, because a return difference from a mixing failure is
uninterpretable; and do not convert a per-step score win into a decision claim without the
rollout readout and its MCSE. M7 runs after M0 and M3 regardless of M4–M6's progress, since
its dependencies are the kernel checks and the readout, not the external branches.

## R10 — frozen external encoder + calibrated MLP posterior (2026-11-18)

**Hypothesis.** A 150M text encoder can be fixed as a feature map, and the
posterior over a small multi-layer head on its outputs can supply the
uncertainty the point head lacks.  The encoder is not in the target; every
hidden and output weight of the head is.  The second hypothesis is that a
proper-score (Brier/RLCD) target makes the posterior predictive better
calibrated than the ordinary NLL target.

**Implementation.** `src/text_head.rs` loads a safetensors table
(`embeddings [N,768]`, `labels [N]`, `split [N]`).  `MlpHeadPotential` is a
multi-layer perceptron `768 → 32 → 16 → 4` with `tanh` hidden activations and
`P = 25 204` free weights/biases; no head layer is frozen.  `LossKind::Nll` is
the ordinary categorical likelihood.  `LossKind::Brier` uses
`Σ_c (p_c − y_c)²` — the RLCD proper score on one-hot labels — as the negative
log target, giving a **generalized Bayes** posterior (not a literal likelihood)
whose risk is minimized at the calibrated probability.  The example trains the
anchor with NLL by default and samples the target with Brier
(`--posterior-loss brier`); `--posterior-loss nll` is the identical-head
control.  No linear head is part of the model or the CLI.

`examples/ultrabert_head_pcnl.rs` fits the head by Adam, builds an informed PCA
basis from that path (and a seeded random-basis control), and runs exact MH-pCN
through either the restricted chart or `PotentialFullTarget`, which carries the
projected complement and therefore every head weight.  The full-target stage
uses the existing `mh_pcn_z_full` + `mh_pcn_w_full` Metropolis-within-Gibbs
pair.  ECE is the 10-bin top-label expected calibration error from
`experiment::expected_calibration_error`.

**Frozen artifact.** `dmis-lab/UltraBERT-Embed-150M` at revision
`b06b36724c5417a1ec1b7863dd4fa542b74e4899`; CLS pooling, L2-normalised, max 96
tokens.  `scripts/pcnl-ultrabert-embed` regenerates the fixture; its manifest is
`docs/fixtures/ultrabert_agnews_embeddings.json`, and the artifact sha256 is
`8fca923d12360482c0914bfa3ada1c50121d7c6fcdcc65cf0a8618ad5ead3745`.  The data
are a deduplicated 240-train / 120-test AG News probe (60/30 per class), not an
MTEB evaluation.

**Measured (default command, 2026-11-18): Brier target.**  Data-scarce arm: 8
of 60 training rows per class (`N_train=32`), `λ=1`, prior
`N(θ_map, 0.5²I)`, 4 chains × 3 600 steps, burn 1 200, thin 4, full
`P=25 204` head target:

| basis | rank | accept z / w | BMA acc | BMA NLL | BMA Brier | BMA ECE | mean draw acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| informed | 8 | 98.3 / 95.3 % | 73.33 % | 0.7842 | 0.3995 | **0.0784** | 67.60 % |
| random | 8 | 99.8 / 95.1 % | 73.33 % | 0.7869 | 0.3995 | **0.0789** | 67.61 % |
| informed | 16 | 97.6 / 95.5 % | 73.33 % | 0.7821 | 0.3998 | 0.0957 | 67.64 % |
| random | 16 | 99.8 / 95.4 % | 73.33 % | 0.7825 | 0.3997 | 0.0956 | 67.62 % |
| informed | 32 | 97.7 / 95.5 % | 73.33 % | 0.7845 | 0.4010 | 0.0990 | 67.66 % |
| random | 32 | 99.7 / 95.6 % | 73.33 % | 0.7805 | 0.4003 | 0.0973 | 67.62 % |

The MAP MLP is 72.50 % accuracy, NLL 1.9535, Brier 0.5113 and ECE 0.2462: a
severe overconfidence failure on 32 training rows.  Posterior averaging improves
the paired NLL by `−1.17 ± 0.23` and Brier by `−0.11 ± 0.032` over 120 test rows,
and cuts ECE to 0.078–0.099.  The mean individual draw is only 67.6 %, so the
BMA recovers the point model's accuracy from a much weaker ensemble and
calibrates the probabilities.  The NLL contrast is about 5σ and the Brier contrast about 3.5σ.

A four-subset recheck at rank 8 (same sampler seed, `--subset-seed`
20261118–21) shows the calibration effect is not one draw: MAP top-label ECE
ranged **0.139–0.313**, BMA ECE **0.052–0.130**, MAP NLL 1.00–2.74 and BMA NLL
0.50–0.97; accuracy moved from 65.0–83.3 % to 65.8–84.2 %.  This is
training-subset variation on one fixed test set, not independent test splits.

**NLL-target control (same MLP, same anchor).**  `--posterior-loss nll` gives
BMA ECE 0.082–0.102 and Brier 0.4004–0.4020: the calibration improvement over
the MAP is reproduced by posterior averaging, and this sample does **not**
separate the Brier target from the NLL target.  The Brier target is the
declared calibrated target, but no claim that it beats NLL is made here.  The
informed and random bases differ by <0.001 in every score, as expected because
the full target is the same weight space and the basis only chooses a sweep
split.

**Full-data MLP (240 training rows, 2026-11-18).**  The same `768 → 32 → 16 →
4` head, full `P=25 204` target, no calibration split.  The MAP is 83.33 %,
NLL 1.3230, Brier 0.3158, ECE 0.1624.  BMA rank 8–32 is 83.33–85.00 %,
NLL 0.5338–0.5479, Brier 0.2299–0.2318, ECE **0.0515–0.0746**; paired
improvements are `dNLL −0.775…−0.789 ± 0.202…0.206` and
`dBrier −0.084…−0.086 ± 0.029`.  Posterior averaging therefore still
calibrates at 240 rows, though the MAP is already less overconfident than at
32 rows and the accuracy gain is at most one row.

**Held-out calibration split with noisy labels.**  `--cal-per-class 4` holds
out 4 of 60 training rows per class (16 total) and keeps 224 for the fit; the
fit rows use NLL and the held-out rows use Brier, with `--label-noise 0.2`
flipping each calibration label to a different class with probability 0.2
(3/16 flipped at seed 20261118).  All test metrics remain on the clean test
labels.  The MAP on the 224 fit rows is 83.33 %, NLL 1.0938, Brier 0.2809,
ECE 0.1425.  BMA rank 8–32 is 85.83–87.50 %, NLL 0.4289–0.4459, Brier
0.2155–0.2196, ECE **0.0664–0.0846**; paired `dNLL −0.648…−0.665 ± 0.186…0.191`
and `dBrier −0.061…−0.065 ± 0.024`.  The noisy held-out split does not
break clean-test calibration and raises BMA accuracy, but its BMA ECE is
slightly *worse* than the plain full-data target (0.052–0.075); the fit set
and sampler seed also differ, so this is not an isolated estimate of the
label-noise effect.

**Width/depth comparison (full data, rank 8, same sampler budget).**  The
baseline `32,16` is the best calibrated of the three:

| hidden | P | MAP acc / NLL / Brier / ECE | BMA acc | BMA NLL | BMA Brier | BMA ECE |
|---|---:|---|---:|---:|---:|---:|
| `32,16` | 25 204 | 83.33 / 1.3230 / 0.3158 / 0.1624 | 85.00 % | 0.5469 | 0.2308 | **0.0629** |
| `64,32` | 51 428 | 80.83 / 1.7526 / 0.3413 / 0.1762 | 84.17–85.00 % | 0.6342–0.6458 | 0.2672–0.2703 | 0.1108–0.1177 |
| `32,16,8` | 25 308 | 82.50 / 1.5962 / 0.3495 / 0.1842 | 78.33–79.17 % | 0.7110–0.7114 | 0.2755–0.2763 | 0.1029–0.1053 |

Widening or deepening degrades both accuracy and calibration at this data
size; the two-hidden-layer model is retained.  These are single-split results
(120 clean test rows), not a model-selection study.

**Scope.**  The encoder is frozen, the head is a two-hidden-layer MLP, the data
are one AG News split, and the target is a Brier Gibbs posterior rather than a
true likelihood.  Acceptance is high and `β→1`; `ESS(Φ)` is near the kept-draw
ceiling, so the chain is iid-like in many flat directions rather than a
narrowly converged posterior.  ECE is a top-label diagnostic, not a complete
calibration certificate.  No benchmark, encoder-posterior or decision claim
follows.  Next: repeat the full-data and calibration-split cells over seeds and
independent test splits, and compare against a tuned L_2/early-stopped MLP.

## R11 — sequential Bayesian continual learning (2026-11-19)

**Target.**  The frozen-encoder head is updated by the Bayes recursion

```text
π_t(θ) ∝ π_{t-1}(θ) · exp(−λ N_t · mean Brier_t(θ)).
```

`π_{t-1}` is summarised from its pCNL particles by a fitted Gaussian.  The
default is the diagonal `N(μ, diag(σ²))`; `θ = μ + σ ⊙ (Bz + w)` turns that
prior back into `N(0, I_r) × N(0, I_{P−r})`, so `PotentialFullTarget` samples
the new posterior with the unchanged exact kernel.  The approximation is the
prior summary, not the kernel.  `pcnl::continual::DiagonalPosterior` implements
the default.  The same random basis is held for the whole stream: redrawing it
each round would re-express the carried prior through a different chart.

**Data stream.**  AG News classes are visited as
`[0,1] → [2,3] → [0,1] → [2,3]`; every test metric covers all four classes.
The `continual` arm carries the fitted prior; `independent` resets it to
`N(0, 0.5² I)` each round.  Both use the same Brier target, 4 chains × 2 400
steps (800 burn, thin 4), rank 8, and seeds.  The diagonal arm holds one random
basis; the low-rank arm uses the PCA basis of the previous particles.

**Why replay.**  Strict no-replay recursion was tested at the same budget:
carried parameters were not enough for class-disjoint rounds (mean forgetting
90.6 pp).  The delivered learner adds replay rows per seen class to the
likelihood; this is an approximation to the historical likelihood, retained
because the carried prior alone discards too much old-class information.  It
is not exact Bayes over the full history and is labelled as Bayesian replay.
`--replay-per-class 60` is the full clean training split at this fixture;
smaller values reduce the rows added each round (see R12 for growth on revisits).

**Measured (canonical seed 20261118, final round, clean 120-row test split).**

Replay rows are weighted so every present class carries the same total
likelihood weight; the weights sum to the row count, so `λ · N · mean_loss`
keeps its temperature.  Without that step the current classes have twice the
rows of the replayed classes at round four and the likelihood itself favours
them.

| replay / class | class-balanced | arm | acc | NLL | Brier | ECE | mean forgetting |
|---:|:---:|---|---:|---:|---:|---:|---:|
| 0 | — | continual | 39.17 % | 1.8169 | 0.8386 | 0.2448 | 90.56 pp |
| 30 | no | independent | 82.50 % | 0.7849 | 0.4096 | 0.3183 | 46.11 pp |
| 30 | no | continual | 80.83 % | 0.6053 | 0.3107 | 0.1779 | 30.56 pp |
| 30 | yes | independent | 80.83 % | 0.7671 | 0.3988 | 0.2992 | 5.00 pp |
| 30 | yes | continual | 80.83 % | 0.6372 | 0.3290 | 0.1717 | 5.56 pp |
| 60 | yes | independent | 80.00 % | 0.6992 | 0.3597 | 0.2362 | 1.67 pp |
| 60 | yes | continual | 81.67 % | 0.5326 | 0.2897 | 0.1065 | 3.33 pp |

Forgetting is the mean per-class drop over each adjacent disjoint class switch;
a negative value means later accuracy was higher than the earlier measurement.
These are the reproducible single-seed numbers, not a variance estimate.

**Seed sweep (diagonal summary, same budget).**

| replay / class | arm | seeds | acc | NLL | Brier | ECE | mean forgetting |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | independent | 5 | 80.50 (2.98) | 0.7740 (0.0215) | 0.4005 (0.0114) | 0.2934 (0.0353) | 5.89 (3.65) |
| 30 | continual | 5 | 83.67 (3.31) | **0.6131 (0.0157)** | **0.3155 (0.0099)** | **0.2024 (0.0442)** | 3.11 (1.78) |
| 60 | independent | 3 | 82.50 (2.21) | 0.7014 (0.0180) | 0.3596 (0.0128) | 0.2619 (0.0294) | 2.96 (1.39) |
| 60 | continual | 3 | 82.50 (2.21) | **0.5467 (0.0199)** | **0.2866 (0.0174)** | **0.1222 (0.0138)** | 3.33 (1.67) |

Values are mean (sample standard deviation) over seeds 20261118–20261122 at
30 and 20261118–20261120 at 60.

Class balancing is the large retention effect: at the canonical seed it takes
mean forgetting from 30.56 pp to 5.56 pp for the carried prior and from
46.11 pp to 5.00 pp for the reset control.  That is well outside the observed
seed spread, although balancing itself was not swept separately.

The carried prior's robust gain is calibration.  At 30 rows it lowers mean NLL
from 0.774 to 0.613, Brier from 0.401 to 0.316, and ECE from 0.293 to 0.202;
it was better on all three in every one of the five seeds.  At 60 rows it
lowers NLL from 0.701 to 0.547, Brier from 0.360 to 0.287, and ECE from 0.262
to 0.122 across three seeds.  The earlier single-seed accuracy advantage at 30
does not survive the sweep: the 30-row mean difference is +3.17 pp with
standard deviations of 3.31 and 2.98, and at 60 the means are identical
(82.50 %).  Nor is there a resolved forgetting advantage once replay and
balancing are on: the 30-row means are 3.11 versus 5.89 pp with overlapping
ranges, and the 60-row means are 3.33 versus 2.96 pp.  The delivered default
is 30 rows per class with class balancing; 60 is the calibration arm.  Neither
is an accuracy or no-forgetting claim.

**Split sweep (five alternate partitions, 30 replay rows/class).**

`EmbeddingTable::resplit` pools all 360 fixture rows and repartitions them into
60 train + 30 test rows per class.  With the model seed fixed at 20261118 and
split seeds 1–5:

| arm | splits | acc | NLL | Brier | ECE | mean forgetting |
|---|---:|---:|---:|---:|---:|---:|
| reset prior | 5 | 77.50 (5.84) | 0.8223 (0.0462) | 0.4299 (0.0289) | 0.2693 (0.0641) | 3.89 (2.12) |
| carried prior | 5 | 76.33 (6.52) | **0.6714 (0.1030)** | **0.3515 (0.0571)** | **0.1519 (0.0553)** | 3.22 (1.90) |

The alternate splits overlap, so this measures split sensitivity rather than
five independent samples.  The calibration result replicates: the carried
prior has lower NLL, Brier and ECE in every split.  Accuracy is 1.17 pp lower
on average (standard deviations 6.52 and 5.84 pp), and forgetting is 0.67 pp
lower with overlapping ranges (3.22 versus 3.89 pp), so neither is a resolved
effect.  Across both seed and split resampling, calibration is the only
carried-prior advantage that survives.

**Joint baseline and post-hoc calibration.**

A single posterior over all four classes is the natural reference for what the
sequential updates cost.  Run it as one round with all classes and four times
the per-round sampler budget:

```bash
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
    --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16
```

That is 38,400 proposals, matching four sequential rounds of 2,400.  Raw
results over the same five resplits, with all 240 training rows:

| posterior | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| joint, all classes at once | **82.50 (5.17)** | 0.7622 (0.0342) | 0.3902 (0.0229) | 0.3131 (0.0423) |
| sequential carried | 76.33 (6.52) | **0.6714 (0.1030)** | **0.3515 (0.0571)** | **0.1519 (0.0553)** |

The raw joint posterior is more accurate in all five splits, by 3.3–9.2 pp
(mean 6.17 pp), but its probability forecasts are worse.

That reverses after post-hoc temperature scaling.  `--calibrate-joint 10`
holds out 10 training rows per class, fits the joint posterior on the remaining
50 per class, fits one temperature `T` on the held-out training NLL, and
applies it to the test probabilities.  Test labels are never used.

```bash
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
    --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16 \
    --calibrate-joint 10
```

| joint variant | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| raw, 50 train/class | 83.00 (5.79) | 0.8334 (0.0376) | 0.4326 (0.0234) | 0.3547 (0.0552) |
| temperature-scaled, 50 train + 10 cal/class | **83.00 (5.79)** | **0.4860 (0.1615)** | **0.2582 (0.0798)** | **0.0983 (0.0387)** |
| sequential carried, 60 train/class + replay | 76.33 (6.52) | 0.6714 (0.1030) | 0.3515 (0.0571) | 0.1519 (0.0553) |

The fitted `T` was 0.17–0.30: the BMA probabilities are under-confident, and
sharpening changes no argmax, so accuracy is identical before and after.
Calibration lowers NLL, Brier and ECE in every split, and in this
data-asymmetric table the calibrated joint beats the sequential carried
posterior on all four metrics in every split.
The earlier conclusion that sequential updating produced the better forecast
holds only against the uncalibrated joint.

That AG News comparison is not data-matched: the joint arm gives up 40
training rows for its calibration fit while the sequential arm uses all 240
plus replay, and the sequential arm has no post-hoc temperature fit.  The
symmetric protocol removes both asymmetries with `--heldout-calibration 10`:
the same 10 rows/class are withheld before either arm trains, both fit on 50
rows/class, and both temperatures use the same held-out rows.  Over the five
resplits:

| arm | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| sequential carried, raw | 77.33 (5.99) | 0.6740 (0.0602) | 0.3547 (0.0397) | 0.1625 (0.0478) |
| sequential carried, temperature | 77.33 (5.99) | 0.6288 (0.0902) | 0.3288 (0.0533) | 0.1005 (0.0504) |
| joint, raw | **83.00 (5.79)** | 0.8334 (0.0376) | 0.4326 (0.0234) | 0.3547 (0.0552) |
| joint, temperature | **83.00 (5.79)** | **0.4860 (0.1615)** | **0.2582 (0.0798)** | **0.0983 (0.0387)** |

The calibrated joint is more accurate in all five splits (+5.67 (2.24) pp) and
has lower Brier in all five.  NLL is lower in four of five (split 1 is 0.026
higher).  ECE is lower in only two of five and is essentially tied on average
(0.098 versus 0.101), so the earlier "all four metrics in every split" claim
does not survive the symmetric protocol.  Part of that ECE advantage came from
leaving the sequential posterior uncalibrated.  Temperature scaling itself is
stable for the joint here: NLL, Brier and ECE fall in every split.

**Rank check (canonical split, one seed).**  Raising the carried prior's rank
from 8 to 16, 32 or 64 did not improve final accuracy (79.17, 80.83 and
75.00 %, versus 80.83 % at rank 8), and calibration stayed in the same band.
The rank cap is not what blocks an accuracy gain, so the default stays at 8.

**λ / replay sweep.**  Canonical split and seed, continual arm, nine cells
(`--lambda {0.5,1,2} --replay-per-class {15,30,45}`):

| λ | replay | acc | NLL | Brier | ECE | forgetting |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 15 | 77.50 | 1.0309 | 0.5535 | 0.3909 | 6.67 |
| 0.5 | 30 | 85.00 | 0.8941 | 0.4713 | 0.4081 | 1.67 |
| 0.5 | 45 | 81.67 | 0.7515 | 0.3862 | 0.2997 | 2.22 |
| 1.0 | 15 | 75.83 | 0.7199 | 0.3797 | 0.1908 | 10.00 |
| 1.0 | 30 | 80.83 | 0.6372 | 0.3290 | 0.1717 | 5.56 |
| 1.0 | 45 | 80.00 | 0.5551 | 0.2978 | 0.1238 | -1.11 |
| 2.0 | 15 | 79.17 | 0.6043 | 0.3134 | 0.0859 | 18.89 |
| 2.0 | 30 | 75.83 | 0.6137 | 0.3321 | 0.0700 | 7.78 |
| 2.0 | 45 | **86.67** | **0.5133** | **0.2591** | **0.1291** | 3.33 |

The apparent winner was validated out of sample rather than adopted from this
single cell.  λ=2/replay=45 over five seeds is 82.33 (2.60) % acc, NLL 0.5410
(0.0313), Brier 0.2775 (0.0166), ECE 0.0776 (0.0327), forgetting 4.67 (3.16)
pp; the λ=1/replay=30 default is 83.67 (3.31) %, 0.6131 (0.0157), 0.3155
(0.0099), 0.2024 (0.0442), 3.11 (1.78) pp.  Over the five alternate splits,
λ=2/replay=45 is 78.17 (4.39) %, 0.5885 (0.0697), 0.3110 (0.0507), 0.0679
(0.0396), 3.55 (1.78) pp, versus the default's 76.33 (6.52) %, 0.6714 (0.1030),
0.3515 (0.0571), 0.1519 (0.0553), 3.22 (1.90) pp.

The robust difference is calibration: λ=2/replay=45 lowers NLL, Brier and ECE
against the default under both resamplings, with ECE roughly halved.  Accuracy
does not resolve (1.3 pp lower over seeds, 1.8 pp higher over splits), and
forgetting is not improved.  It also uses 45 replay rows per class instead of
30.  It is a calibration-oriented alternative, not a replacement default.

**Low-rank summary (opt-in, not adopted).**  `pcnl::continual::LowRankPosterior`
fits the top `rank` principal directions with their own widths plus an
isotropic residual, giving `B diag(s²) Bᵀ + d² P_⊥` instead of a diagonal prior.
The full-target chart supports it through `AffineSubspace::low_rank`; the
kernel is unchanged.  It did not improve the operating point:

| replay / class | arm | acc | NLL | Brier | ECE | mean forgetting |
|---:|---|---:|---:|---:|---:|---:|
| 0 | continual | 40.83 % | 1.4124 | 0.7390 | 0.2338 | 91.11 pp |
| 30 | continual | 68.33 % | 0.8074 | 0.4232 | 0.0743 | 5.00 pp |
| 60 | continual | 82.50 % | 0.6307 | 0.3097 | 0.1491 | -5.00 pp |

At 30 balanced replay rows it matches the diagonal summary's 5.00–5.56 pp
forgetting but costs 12.5 pp canonical-seed accuracy.  At 60 it is 0.83 pp
above the canonical-seed diagonal and its mean forgetting is −5.00 pp versus
3.33 pp, but both are single-seed differences inside the diagonal's observed
seed spread.  Its calibration is worse (NLL 0.6307 vs 0.5326, Brier 0.3097 vs
0.2897, ECE 0.1491 vs 0.1065).  Strict no-replay low-rank still forgets
91.11 pp, so the shared head under disjoint likelihoods — not the missing
correlations alone — is the limiting factor.  The diagonal summary remains the
default.  The low-rank arm has not been seed-swept.

**Second text dataset (20 Newsgroups).**  The same loop was repeated on a
second frozen UltraBERT fixture built from four 20 Newsgroups categories
(`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.guns`),
60 train and 30 test rows per class.  Headers, footers and quoted text are
removed so the newsgroup name cannot leak from the header.  This is a
different corpus and label set, not a relabelling of AG News.

```bash
uv run --python 3.12 --with torch --with transformers --with safetensors \
  --with scikit-learn scripts/pcnl-ultrabert-embed-20ng
```

| five resplits | acc | NLL | Brier | ECE | mean forgetting |
|---|---:|---:|---:|---:|---:|
| sequential carried | 87.00 (1.63) | 0.4440 (0.0267) | 0.2202 (0.0130) | 0.1444 (0.0106) | 1.00 (3.80) pp |
| joint, temperature-scaled | **88.17 (3.61)** | **0.3223 (0.0643)** | **0.1676 (0.0343)** | **0.0606 (0.0174)** | n/a |

In that data-asymmetric table the calibrated joint has lower NLL, Brier and
ECE in all five splits, with fitted `T = 0.187–0.354`; ECE falls from 0.144 to
0.061 on average.  Accuracy is
not a uniform win: the joint is higher in three splits, equal in one, and
4.17 pp lower in one, averaging +1.17 (3.42) pp.  The calibration advantage
therefore reproduces on a second corpus; in that data-asymmetric table the AG
News accuracy advantage does not.  The second fixture has only 90 rows per
class and its joint posterior is more variable than on AG News.

**20 Newsgroups seed sweep.**  Five model seeds on the canonical 20 Newsgroups
split, at the same calibrated-joint budget:

| seeds | acc | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|
| 5 | **89.17 (1.32)** | 0.2839 (0.0296) | 0.1478 (0.0131) | 0.0500 (0.0100) |

Accuracy ranged from 87.50 % to 90.83 % (sd 1.32 pp), and `T` from 0.303 to
0.371.  This is model/proposal variance at one fixed split, not split
variance; the five-resplit mean above (88.17 (3.61) %) uses a different
resampling axis and is not paired with these rows.

**Symmetric held-out comparison.**  `--heldout-calibration 10` removes the same
10 training rows per class before either arm trains.  Both arms therefore fit
on 50 rows per class, and both temperatures are fitted on exactly those 10
held-out rows.  Over five resplits:

| arm | acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| sequential carried, raw | 86.50 (3.84) | 0.4721 (0.0593) | 0.2295 (0.0358) | 0.1553 (0.0382) |
| sequential carried, temperature | 86.50 (3.84) | 0.4683 (0.1572) | 0.2104 (0.0513) | **0.0849 (0.0282)** |
| joint, raw | 88.83 (2.61) | 0.6371 (0.0318) | 0.3048 (0.0208) | 0.3271 (0.0300) |
| joint, temperature | **88.83 (2.61)** | **0.3064 (0.0911)** | **0.1526 (0.0365)** | **0.0591 (0.0201)** |

The calibrated joint is more accurate in all five splits (+2.33 (2.07) pp) and
has lower NLL and Brier in all five.  The earlier mixed 20 Newsgroups accuracy
(three wins, one tie, one loss) came from the data asymmetry: with both arms on
50 rows/class the joint is ahead in every split.  ECE is lower in four splits
and 0.004 higher in the fifth.  Temperature scaling leaves accuracy unchanged,
as it must.
For the sequential arm it lowers ECE in every split but worsens NLL in two of
five; its mean NLL is unchanged (0.4721 versus 0.4683) and its NLL spread more
than doubles.  A 40-row calibration set is noisy enough that the fitted
temperature is not a free improvement for every posterior.  The joint's lower
NLL and Brier survive that noise.  This removes the earlier data asymmetry; it
does not make the temperature fit independent of the test set.

**Limits.**  Two fixtures and four rounds; the diagonal summary was swept over
five seeds at 30 rows, three at 60, and five overlapping resplits on AG News,
plus five resplits and a five-seed canonical sweep on 20 Newsgroups.  Seed and
split variance are characterized separately, not over a full seed × split grid.
The five-resplit joint tables use one model seed per split; the symmetric
tables use one model seed per split as well.  The low-rank arm and the unbalanced
controls were not swept.  Both summaries
are Gaussian approximations (the default ignores correlations, the low-rank arm
keeps only the top directions); replay double-counts old rows already
represented in the prior; Brier makes this a generalized Bayes target; the
encoder remains frozen.  The 20 Newsgroups categories are a four-class subset,
not the full 20-class task, and both fixtures are controlled probes rather than
benchmarks.  No benchmark or decision claim follows.

Command:

```bash
cargo run --release -p pcnl --example ultrabert_continual_pcnl
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
    --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- --replay-per-class 60
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- --replay-per-class 30 --class-balance false
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- --replay-per-class 0 --arm continual
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- --prior lowrank --replay-per-class 30
for seed in 20261118 20261119 20261120 20261121 20261122; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- --seed "$seed"
done
for split_seed in 1 2 3 4 5; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
        --resplit-test-per-class 30 --split-seed "$split_seed"
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
        --resplit-test-per-class 30 --split-seed "$split_seed" \
        --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16 \
        --calibrate-joint 10
done
for seed in 20261118 20261119 20261120 20261121 20261122; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
        --seed "$seed" --tasks '0,1,2,3' --arm independent \
        --steps 9600 --burn 3200 --thin 16 --calibrate-joint 10
done
for split_seed in 1 2 3 4 5; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
        --resplit-test-per-class 30 --split-seed "$split_seed" \
        --heldout-calibration 10 --arm continual
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --fixture docs/fixtures/ultrabert_20ng4_embeddings.safetensors \
        --resplit-test-per-class 30 --split-seed "$split_seed" \
        --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16 \
        --heldout-calibration 10
done
for split_seed in 1 2 3 4 5; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --resplit-test-per-class 30 --split-seed "$split_seed"
done
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
    --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16
cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
    --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16 \
    --calibrate-joint 10
for split_seed in 1 2 3 4 5; do
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --resplit-test-per-class 30 --split-seed "$split_seed" \
        --heldout-calibration 10 --arm continual
    cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
        --resplit-test-per-class 30 --split-seed "$split_seed" \
        --tasks '0,1,2,3' --arm independent --steps 9600 --burn 3200 --thin 16 \
        --heldout-calibration 10
done
for lam in 0.5 1 2; do
    for rep in 15 30 45; do
        cargo run --release -p pcnl --example ultrabert_continual_pcnl -- \
            --arm continual --lambda "$lam" --replay-per-class "$rep"
    done
done
```

## Reproduction and definition of done

Run existing commands from the **unfer repository root**. The script stage list is the
authority for available named stages; its default set includes a potentially lengthy test
suite, and `--all` includes many hours of historical experiments. Do not run it implicitly.

```bash
./scripts/pcnl-reproduce --list
cargo test --release -p pcnl --locked
./scripts/pcnl-reproduce quick
./scripts/pcnl-reproduce budget
./scripts/pcnl-reproduce rhat-null
cargo run --release -p pcnl --example subspace_bound
cargo run --release -p pcnl --example readout_compare
```

Run the existing gradient and numerical tests appropriate to each implementation change,
then the known-target tests added in R0/R1; do not repeat every expensive historical ladder
without a changed hypothesis. Separate build time from sampling time. Store per-cell logs,
manifests, draws or predictive summaries, diagnostics and uncertainty intervals together.
Every reported table must be regenerable from those artifacts and identify its anchor.

The core plan is complete when R0–R4 deliver an audited target/kernel distinction, exact
small-model references, a tested informed-subspace comparison, a quantified SG bias budget,
and a properly calibrated readout study on a learnable task, whether the primary research
hypothesis succeeds or fails. The R7 research deliverable is complete when one
synthetic MDP has a measured per-step score table *and* a rollout-return table from the same
chains, with the negative or inconclusive cells published beside the positive ones; the
checkpoint/GPU arm is a separate follow-on. Record the result in `../docs/PCNL_STATUS.md` and
update the discussion only after results exist. External branches, billion-parameter scaling and
infinite-dimensional convergence are separate follow-on projects.
