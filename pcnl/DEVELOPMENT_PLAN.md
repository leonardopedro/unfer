# pcnl development plan — required RLCD and sampler certificates

Current work order: 2026-11-15. Read required application track R9 and sampler track R8 in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) and mathematical packets Q0–Q7 in
[timepiece/pcnl.md](../../timepiece/pcnl.md). The results ledger is
[PCNL_STATUS.md](../docs/PCNL_STATUS.md). This is a plan; no code, tests, experiments or
Lean compilation are claimed as executed by creating it. All new artifacts below are
proposed until their implementation and checks are recorded.

## E0–E7 — implement the required small Edge0/RLCD research track

The concrete specification is [EDGE0_RLCD_PLAN.md](EDGE0_RLCD_PLAN.md). Execute these
packets in order; their acceptance criteria are future work, not checks already passed.
The reinforcement stage is required. D0–D7 below govern the separate sampler work.

### E0 — pin the experiment and build the probability oracle

Record Edge0 commit `7b60076`, the chosen latent D/U architecture, our different small-model
routing/attention rules, software revisions, precision, world manifests, compute caps,
training-arm identities, expected outputs and acceptance endpoints. Add a tiny finite-world
generator with exact conditional outcome probabilities for all three question types and
utility-optimal actions. Expose world IDs before question expansion and prevent overlap
across train/dev/calibration/test. Predeclare the bounded R9 campaign before final scoring.

**Acceptance:** an independently enumerated world agrees with the oracle, missing
observations cannot lower conditional entropy on average, and correlated question
marginals are not multiplied into a false joint probability.

### E1 — implement the small latent-MoE and typed decision interface

Create separately named modules for grouped D/U experts, deterministic top-k routing,
question-independent state encoding, independent question attention, four-path policy and
typed heads. Start with the R9 configuration and keep total parameters below one million.
Count all tensors; publish active/total work and parameter counts. Use a dense soft-routing
reference to check selected-expert implementations. Do not require the large-model engine.

**Acceptance:** stable normalized finite probabilities; schema failures surfaced; packed
and separate questions agree; choice/key permutations obey the specified equivariances;
ordinal level order is preserved. State cache keys include all varying encoder inputs,
weights and inference settings. Route prediction, quantization and cache approximation
are deferred until their prediction error and probability effects can be measured.

### E2 — implement genuine RLCD and enumerate its gradient

Implement the four-path categorical policy, exact deployed marginal, iid sampled-path
U-statistic loss, explicit REINFORCE score term, detached valid baselines and direct
readout/shared-parameter derivatives. Keep independent RNG streams for data, path policy,
initialization and any later MCMC. Loss and reward signs must be visible in the manifest.

**Acceptance:** exhaustive finite-support expectation of the gradient equals the exact
marginal-objective derivative at nonuniform policies/readouts, including shared weights.
Compare numerical differences with an independent finite-difference reference on the tiny
fixture. Confirm correctness-only reward fails the proper-distribution optimum test on a
noisy binary world. A policy-gradient-free readout arm cannot satisfy this packet.

### E3 — run the supervised control and required reinforcement arms

Reuse a common optional supervised warm start. Train the supervised CE+Brier baseline
and the full RLCD arm with declared equal total budgets. Include exact-marginal Brier
backpropagation through the policy as a same-objective control; add frozen-policy,
per-path-score and correctness-reward controls within the campaign cap. The RLCD arm must update the
sampling policy using proper-score rewards, not merely log its entropy or sample routes.
Keep MoE load losses/entropy/KL regularizers zero initially or explicitly specify their
changed objective and budgeted ablation. Use independent outcome labels as rewards.

**Acceptance:** reproducible manifests and learning traces show policy-gradient updates
and their estimated uncertainty, plus head updates, held-out scores, route utilization,
collapse diagnostics and complete cost. The analytic gradient gate must pass before this
training evidence is interpreted. Supervised success alone does not complete E3.

### E4 — evaluate distributions and decisions under the frozen protocol

Execute the bounded five-seed/world-split campaign of EDGE0_RLCD_PLAN §5. Fit temperatures
on development and decision thresholds on calibration only; final test is scored once
under the frozen comparison protocol. Pair by world and seed, cluster repeated questions
and labels, and report cost of exactly summing four paths. The optional sampled-path
serving ablation needs its own error/variance budget and endpoints.

**Acceptance:** a paired uncertainty interval for the prespecified Brier-excess contrast,
all secondary endpoints and negative seeds retained. Label any improvement in utility,
proper score or calibration separately. Promote only on the declared endpoint and cost;
inconclusive/worse results end this campaign without a hidden seed or metric sweep.

### E5 — attach the existing pCNL target without noisy-acceptance shortcuts

Freeze the trained state encoder, router and latent policy, and expose a small head/adapter
chart through `Potential` and `SubspaceTarget<P>`. Use exact latent-marginal categorical NLL
for a likelihood posterior, or label a Brier Gibbs target as generalized Bayes. Store
prior and likelihood separately from total potential. Follow D0/D1 for target identity,
prior-refresh reference and evidence/rate certification, then D3–D5 for approximations.

**Acceptance:** exact finite-chart/quadrature comparison where feasible, movement/mixing
and predictive error reported at the chosen tolerance. A fresh random latent draw in MH
acceptance is forbidden without a new valid kernel construction. Posterior averaging
adds a separate parameter-sampling error; do not reuse iid-path bounds for chain states.
Any RL objective involving this posterior also needs an explicit gradient/frozen protocol.

### E6 — hand finite probability/decision lemmas to the Lean4 specialist

Supply rational finite fixtures for simplex/mixture closure, Brier and ordinal propriety,
U-statistic expectation, full policy-gradient and valid-baseline identities, and bounded
utility-regret control. Separate ideal arithmetic statements from floating-point tests.
Then connect E5 to Q0/Q1 rather than asserting that an oscillator or abstract sampler gap
already certifies the new neural model. Proposed identifiers remain planned references.

**Acceptance:** the specialist elaborates the implementations, checks their assumptions
and axiom ledger, and records the result. This planning task performs no compilation.

### E7 — archive results and gate architecture extensions

Only executed results enter [PCNL_STATUS.md](../docs/PCNL_STATUS.md), under a new R9 label.
Archive positive, negative and inconclusive comparisons, exact source/config/data hashes,
model tensor counts, gradient fixtures, training/evaluation costs and uncertainty units.
Scale expert count/context, add text rationales or predicted routing, or vary the trunk
in a posterior only after a new hypothesis and explicit budget. A new route predictor
can change the probability model; it must be compared against actual routing. Preserve
the earlier stopped R7 DT record and the separate R8 sampler theorem scope throughout.

## D0 — make the claim and target machine-reviewable

Reuse `src/manifest.rs`, `src/target.rs`, `src/subspace.rs` and the existing experiment
seams. Prepare a per-run claim record containing source revision/dirty diff, target kind,
prior, likelihood sum convention, temperature, chart/anchor/basis identity, dimension and
rank, data split unit, exact/stopped gradient convention, proposal, scan order, adaptation
freeze, actual schedule/floor, precision and observable/readout. Record which theoretical
claims apply and every open assumption; do not encode unproved estimates as default values.

Pin the algebra with a tiny independently computed fixture: `phi` is the total `J`, while
pCN's proposal ratio cancels only its Gaussian prior, leaving `Ψ(z)-Ψ(z')`. Reuse the
existing quadrature and gradient tests before adding coverage. The existing mathematical
MH ratio, its finite-precision implementation and empirical sampling behavior have
separate evidence fields. States with nonfinite current potential require an explicit
initialization/error policy, not a claimed stationary sample after endless rejections.

**Acceptance:** two matched configurations agree on potential and target identity;
changing a prior/temperature/chart changes its identity; changing only a frozen proposal
changes kernel identity. DT full-target support is not inferred from the classifier's
`FullTarget`. Export named parameters for Lean fixtures without claiming Rust verification.

## D1 — implement the first reference certificate and optional refresh mixture

Use `mh_pcn_z(beta=1)` for independent prior proposals in a restricted chart. Add a
mixture wrapper with fixed probability `ω>0` between that primitive and a frozen local
same-target MH kernel. A full-target version must refresh every Gaussian prior block
jointly, include the full likelihood and preserve complement constraints. Do not replace
unadjusted SG-pCNL silently; expose the new exact-reference mode in configuration/logs.

For categorical likelihoods with `Ψ≥0`, prepare an evidence lower bound `Z_lo>0` and
certificate `c=ω Z_lo` (`M=1`). For other likelihoods carry a proved ceiling `M` and
`c=ω Z_lo/M`. A verified bounded region with prior mass at least `a` and `Ψ≤b` gives
`Z_lo=a exp(-b)`. Add a tiny nonconstant rank-1/2 potential whose evidence and predictive
can be enclosed by quadrature **including Gaussian tail bounds**. A numeric integral
without its tail/rounding enclosure is a reference estimate, not a theorem certificate.

**Acceptance:** Q1's ideal-kernel minorization/stationarity theorem instantiated with
positive constants and a concrete source/target identity. Observed contraction and moments
are secondary checks. Publish the rate even if it predicts an impractical budget; separate
“proper and geometrically convergent” from “efficient at the requested tolerance”.

## D2 — establish the controlled dimension benchmark

Add an explicit target family `π_r⊗N(0,I_n)` with fixed nonconstant informed likelihood
and frozen proposal parameters. Use independent simultaneous informed MH and complement
pCN moves. Reuse Q2's product-operator theorem and Q1's evidence certificate. Vary `n`
without changing rank, task, likelihood scale or observable. Include a coordinate observable
in the complement and an informed predictive observable.

**Acceptance:** the same analytic rate constant applies across the declared family, with
mean-square/cost bounds and measured uncertainty beside it. Track memory and forward cost
as well as ESS/evaluation. A negative control with shrinking `beta` must not inherit the
same constant. A realistic coupled target is a distinct extension: audit the deterministic
scan's lack of automatic reversibility, or add a random-scan/lazy reference with its actual
clock and conditional-gap/factorization assumptions. Do not call acceptance stability a gap.

## D3 — price approximate updates, rounding and schedules

Keep the current unadjusted `src/sampler.rs` as an approximate-kernel branch. Match the
formal Q3/T1 coefficients to the actual scaling of data-gradient, prior shrink, batch sum,
noise, temperature and `NormGrad` convention. Reuse tests before adding the missing
analytic canaries: correct Gaussian target with biased invariant mean/variance; a proper
target with unstable discretization; independent gradient-noise inflation. State stability
ranges and compare with closed formulas rather than fitting a claimed posterior to a run.

Separate true gradient, stopped surrogate and clipped drift. Check state dependence and
bias of the minibatch oracle. Record actual step values, floor crossover and adaptation
freeze; unfloored annealing, a positive-floor chain and cycles need separate theorem IDs.
A schedule exponent does not certify ergodicity or posterior accuracy.

For approximate MH potential evaluations, bound acceptance error with the same proposal
and propagate it through Q1's contraction when its metric hypotheses hold. For `f32`
rounding choose an observable/Wasserstein coupling or a declared common reconstruction;
TV between a discrete rounded law and a continuous reference need not be small. Naive
mini-batch log ratios are not exact MH. A pseudo-marginal alternative is a new extended-state
design with a nonnegative unbiased likelihood estimator, not an inference from unbiased loss.

**Acceptance:** a quantitative error budget at declared precision and an exact-reference
comparison on the same target. Report any remaining stationary-bias bound as open. An
empirical match of moments or stable traces does not establish a uniform perturbation bound.

## D4 — measure and, where possible, certify the cost of reducing dimension

Keep the existing frozen-complement target labeled a restricted statistical model.
For full-support approximation, prototype the likelihood marginal
`Lbar(z)=E_w exp(-Ψ(z,w))` with its Gaussian complement. This differs from the frozen
slice and from exponentiating an averaged potential. If evaluating `Lbar` is stochastic,
choose a controlled approximate target or a justified extended-state method before MH.

Implement a linear-Gaussian coupled benchmark with analytically known slice, informed
marginal, complement conditional and predictive. It should expose differences the
factorized D2 example cannot. Evaluate a neglected-gradient-tail/conditional-inequality
certificate only with its global assumptions and sample-error bounds. A local Hessian or
path-PCA spectrum is a diagnostic for rank selection, not an ambient posterior bound.
Use cross-fitting or an explicit empirical-Bayes label for data-selected anchors/bases.

**Acceptance:** model-reduction error and sampler error are separately reported on the
benchmark; informed and matched random bases share the target and evaluation budget.
Do not compute finite TV/KL between a slice-supported distribution and a continuous ambient
posterior. Use predictive differences or the full-support approximation instead.

## D5 — repair uncertainty and policy-approximation acceptance for existing DT cells

Reuse `policy_model`, `mdp`, `policy_readout`, `dt_arm`, `dt_subspace` and existing examples.
The R7 branch already has measurements and a stop decision. Preserve its data hashes,
negative rows and declared score criterion; do not start another unrestricted seed sweep.

Replace the step-independent acceptance SE in `policy_readout::paired_delta` with a
trajectory-clustered paired analysis, preserving paired arms and independent trajectory
IDs. Report separate uncertainty from parameter-chain correlation, data/seed variability
and rollouts. Synthetic correlated traces and trajectory fixtures must check that the
reported uncertainty increases appropriately and that frozen traces cannot pass mixing.
Report proper scores, calibration diagnostics and return separately; a Brier/spherical
improvement does not itself prove calibration or policy benefit.

Audit the finite draw subset used in rollout decoding against a full/successively larger
posterior average at fixed histories and common environment seeds. Name per-step BMA
sampling, argmax decoding and a once-per-episode posterior draw separately. Q5's value
bound needs uniform policy-TV control over relevant histories and bounded reward range;
a held-out average action score does not supply that premise. Empirical posterior-draw
measures are not close in TV to continuous posteriors merely because their BMA is accurate.

**Acceptance:** corrected uncertainty and policy-approximation fields added to the same
historical cells where reproduction is feasible. Any changed verdict is accompanied by
the changed estimator and paired evidence. Reopen the stopped application only with a
specified new hypothesis and a data/compute budget, not an after-the-fact success metric.

## D6 — optional operator certificate after the exact references

Use the actual invariant-law `L²` operator and ground projection, with a lazy reversible
kernel or a justified nonreversible norm/multi-step formulation. A finite retained block
needs certified tail, coupling, projection and arithmetic bounds. Reuse Q6/T4 and the
uniform-error discipline in `timepiece/softmodes.md`; finite matrix gaps, raw polynomial
bands and small seed residuals do not certify the full Markov spectrum. The returned rate
must improve a meaningful reference bound or justify its computational cost. Spherical
limits and a nested-Fock interpretation are optional representation work, not D1 blockers.

**Acceptance:** a full-space positive certificate or a precisely scoped missing estimate;
never a completed theorem whose concrete gap assumption is the claim being investigated.

## D7 — verification, promotion and stopping

Implementation and Lean specialists execute checks only when the corresponding changes
exist: targeted Rust tests for altered kernels/readouts, analytic reference comparisons,
then the affected Lean modules and axiom audit. The current planning wave runs none.
Use the existing reproduction script and manifest surfaces; add a named evidence stage
only when its command and artifacts exist. Do not count planned commands as passing checks.

Each result records: ideal theorem and assumptions; Rust source mapping and precision;
observable/tolerance; target/reduction/kernel/burn-in/MC/numerical error components;
evaluations, wall time and memory; measured acceptance and movement; diagnostic failures;
and the supported conclusion. Update `../docs/PCNL_STATUS.md` only after execution.
A nonzero reference rate, a loose rate, a reduction counterexample, a bias bound, and a
negative application result are all legitimate research deliverables.
