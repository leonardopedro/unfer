> **Status (2026-10-20):** implemented as workspace crate `pcnl/`;
> the full Week-3 run's measured numbers and plan-conformance notes are
> in [`docs/PCNL_STATUS.md`](PCNL_STATUS.md).
>
> **Status (2026-10-21): §3B's stability claims are now *theorems*,
> not prose.** The Lean 4.28.0 library `BayesianTransformer` (Milestones
> 1–5, plus the follow-up deriving the per-example bound from the
> architecture) is merged into the `timepiece` checkout — see §7 for
> the file-by-file theorem inventory, the statement deltas, and the
> exact assumptions the proofs need. The numerical counterpart of those
> theorems (finite-difference checks of the normalization derivative,
> scale-invariance orthogonality, the ε-region gradient ceiling and its
> 1/‖W‖ blow-up, the M5 noise ceilings) is exercised by
> `pcnl/tests/formal_claims.rs` (landed 2026-10-21b — the file this
> status note already promised; eight tests, one per claim), and the §4
> Phase-4 comparator set now includes standard SGLD and Adam next to
> momentum-SGD, plus the Metropolis-filtered predecessor
> (`pcnl/tests/comparators.rs`).
>
> **Status (2026-10-21b): the LLM-Lean4-specialist's MH→unadjusted
> patch is fully executed, and its three operational claims are now
> *measured*, not asserted.** The specialist's diff (the `SgPcnlState`
> skeleton, the removal of the Metropolis filter, the power-law
> schedule, the mini-batch loader) was already this plan's §3/§5 and
> the crate's `sampler.rs` at diff time; what the diff left open was
> the *evidence*. Closed in this pass, all in `pcnl`:
> 1. **Throughput** (report claim 1) — counted as model evaluations:
>    the unadjusted step is one mini-batch backward per iteration; the
>    MH-filtered predecessor (now implemented as `task::mh_pcnl_chain`,
>    the diff's deleted `PcnlState`, kept as the reference comparator)
>    costs a second full-batch forward per proposal. Wall-clock ratios
>    measured on the full run.
> 2. **VRAM/complexity decoupling** (claim 2) — structural (the state
>    holds θ ∈ ℝ^P plus one mini-batch; the loader streams), asserted
>    by test, not re-measured per N.
> 3. **Tuning shift** (claim 3) — acceptance-rate tuning is gone;
>    β₀/κ carry the schedule, and the acceptance rate of the *filtered*
>    chain is reported as a diagnostic (it collapses for the standard
>    variant — the Phase-0 story, now with the MH chain's own numbers).
>
> New evidence surface (plan §4 Phase 4b): **MH-filtered pCNL as the
> standard analogous algorithm** (same proposal, exact full-batch MH
> filter — the algorithm the diff deleted), the **cyclical schedule
> (§3C Option 2)** finally exercised end-to-end (`--cyclical PERIOD`),
> and `pcnl/tests/{formal_claims,comparators}.rs` — the M1–M5 numerical
> counterparts and the comparator suite (evaluation-count throughput,
> acceptance vs β, filtered-vs-unadjusted posterior agreement via
> cross-R̂).

### 1. Technology Stack Selection & Justification

| Layer | Recommended Choice | Justification & Alternatives |
| :--- | :--- | :--- |
| **Tensor & Autograd Engine** | **`candle` (Hugging Face)** | Minimalist, idiomatic Rust with native autograd (`candle-core`, `candle-nn`). It has zero heavy C++ dependencies (unlike `tch-rs`/LibTorch) while supporting CUDA, Metal, and AVX CPU backends. *(Alternative: `burn` is viable, but `candle` offers cleaner low-level pointer and gradient manipulation for custom MCMC loops).* |
| **MCMC & Math Core** | **`rand` + `rand_distr`** | Sampling $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ + noise injection. |
| **Data Ingestion** | **Custom Mini-Batch DataLoader** | Batched iterator over token sequences in Rust to feed streaming stochastic gradients. |
| **Serialization / State** | **`safetensors`** | Zero-copy weight serialization to save and stream posterior parameter trajectories to disk during sampling. |
| **CLI & Diagnostics** | **`clap` + `indicatif`** | Fast CLI for sweep configurations, step-size tuning, and real-time MCMC progress bars. |
| **Reference Validation (Justified Alternative)** | **Python / JAX (Phase 0 only)** | *Temporary alternative:* A minimal 150-line JAX/PyTorch reference script in Phase 0 is strongly recommended to establish baseline MCMC acceptance rates and verify mathematical convergence on a toy dataset before writing the Rust implementation. |

---

### 2. Architecture: Internally Normalized Transformer (nGPT-lite)

In native Rust via `candle`, the model avoids unconstrained logits by projecting both representations and parameters to the unit hypersphere $\mathbb{S}^{D-1}$.

```
Input Tokens 
     │
     ▼
[Normalized Embedding: E / ||E||] ─── (Unit norm representations)
     │
     ▼
┌────────────────────────────────────────────────────────┐
│ Transformer Layer                                      │
│  ├── Layer Normalization (Hyperspherical projection)   │
│  ├── Scaled Cosine Attention:                          │
│  │     Q, K from W_q/||W_q||, W_k/||W_k||              │
│  │     Logits = tau * (q · k) ∈ [-tau, tau]            │
│  │     Softmax Jacobian is strictly bounded!           │
│  ├── Geodesic / Normalized Residual Connection         │
│  ├── Normalized MLP: W1/||W1||, W2/||W2||              │
│  └── Geodesic / Normalized Residual Connection         │
└────────────────────────────────────────────────────────┘
     │
     ▼
[Normalized Head / Cross-Entropy Loss]
```

#### Flattened Parameter Vector Abstraction
pCNL operates on a continuous, flat Gaussian parameter vector $\boldsymbol{\theta} \in \mathbb{R}^P$. You must maintain a bidirectional view:
$$\boldsymbol{\theta} \in \mathbb{R}^P \underset{\text{flatten}}{\overset{\text{unflatten}}{\rightleftharpoons}} \{W_Q, W_K, W_V, W_O, W_1, W_2, \dots\}$$
Inside the forward pass, each tensor is dynamically evaluated as $\widehat{W} = \frac{W}{\|W\|_F + \epsilon}$.

---

### 3. Mathematical Formulation: Unadjusted SG-pCNL (Stochastic Gradient pCNL)

The full-batch Metropolis filter is completely eliminated. The transition is unconditionally
accepted at every mini-batch step.

#### A. Unadjusted Update Rule
For mini-batch $B_t$ of size $B$ sampled from dataset of size $N$:

$$g_{\text{batch}} = (N / B) \sum_{i \in B_t} \nabla_\theta \ell_i(\theta_t)$$
$$\theta_{t+1} = \sqrt{1 - \beta_t^2}\,\theta_t + \beta_t\,\xi_t - \frac{\beta_t^2 \sigma^2}{2} g_{\text{batch}}$$

where:
- $\xi_t \sim \mathcal{N}(0, \sigma^2 I)$ is prior-preserving Ornstein-Uhlenbeck noise.
- $(N / B)$ scales the mini-batch gradient into an unbiased estimate of the full potential.

#### B. Stability via nGPT Lipschitz Geometry
Without the Metropolis-Hastings reject mechanism, standard transformers diverge due to
unbounded gradient noise covariance: $\text{Cov}[g_{\text{batch}}] \to \infty$ as attention saturates.

Under nGPT:
1. Unit representations ($\|x\| = 1$) + Cosine attention logit clamping $[-\tau, \tau]$
   guarantee that individual sample gradients are uniformly bounded: $\|\nabla_\theta \ell_i(\theta)\| \le M$.
   *(Proved 2026-10-21: `ngpt_exampleLoss_fderiv_bound` derives this from the
   architecture on the region where every weight norm is $\ge \varepsilon$ — §7.4.)*
2. The stochastic gradient covariance is strictly bounded:
   $$\mathbb{E}\left[ \|g_{\text{batch}} - g_{\text{full}}\|^2 \right] \le (N^2 / B) \cdot M^2$$
   *(Partly formalized 2026-10-21: Lean proved the conservative pathwise ceiling
   $\|g_{\text{batch}} - g_{\text{full}}\| \le (N/B + 1) M$ and the covariance-trace
   ceiling $\mathrm{Tr}\,\Sigma(w) \le ((N/B + 1) M)^2$ — §7.5. The $(N^2/B)$-form
   is the sharper *variance* claim; it is the empirical target of the M5 check in
   `pcnl/tests/formal_claims.rs`, not a Lean theorem.)*
3. This bounded perturbation prevents the chain from escaping into degenerate regions,
   guaranteeing geometric ergodicity under Robbins-Monro annealing.
   *(Still open: no ergodicity theorem for the unadjusted chain is formalized —
   §7.6 lists what remains prose.)*

#### C. Step-Size Schedule ($\beta_t$)
Replace dual-averaging acceptance tuning with a learning-rate style decay schedule:
- **Option 1 (Robbins-Monro):** $\beta_t = \beta_0 (1 + \gamma t)^{-\kappa}$, $\kappa \in (0.5, 1]$
- **Option 2 (Cyclical SG-pCNL):** Cyclical cosine decay to alternately explore modes and sample posteriors.

---

### 4. Step-by-Step Implementation Plan

```
[Phase 0: JAX Toy Baseline] ──> [Phase 1: Rust Model & Autodiff]
                                             │
[Phase 4: Adaptation Evaluation] <── [Phase 3: pCNL Engine] <──┘
```

#### Phase 0: Numerical Baseline & Validation (1–2 Days)
* **Goal:** Verify that normalized attention stabilizes pCNL compared to standard dot-product attention.
* **Tasks:**
  * Implement a 1-layer toy transformer on a sequence task (e.g., character prediction or modular arithmetic).
  * Run unadjusted mini-batch steps on a toy task:
    * Verify that standard Dot-Product Transformer explodes or suffers random walks due to unbounded mini-batch gradient spikes.
    * Verify that nGPT maintains stable parameter norms $\|\theta_t\| \sim \sigma \sqrt{P}$ across 10,000 mini-batch iterations without MH rejection.

#### Phase 1: Rust Foundation & Normalized Architecture (3–5 Days)
* **Crates:** `candle-core`, `candle-nn`.
* **Tasks:**
  * Implement the `Normalize` operator:
    $$\widehat{X} = \frac{X}{\|X\|_2 + \epsilon}$$
  * Implement `NormalizedLinear`: Holds raw tensor $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, computes $W / \|W\|_F$ on the fly during `forward()`.
  * Implement `CosineSelfAttention`:
    $$A_{ij} = \text{softmax}\left(\tau \cdot \frac{q_i \cdot k_j}{\|q_i\| \|k_j\| + \epsilon}\right)$$
  * Construct `FlatParameters`: A struct that packs all trainable tensors into a single contiguous `candle::Tensor` of shape $[P]$ and unpacks them back into module weight references.

#### Phase 2: The Unadjusted SG-pCNL Engine in Rust (3–4 Days)
* **Crates:** `rand`, `rand_distr`, `rayon` (for parallel chains).
* **Tasks:**
  * Implement streaming Mini-Batch DataLoader in Candle.
  * Remove all Metropolis-Hastings evaluation logic (eliminating the redundant forward pass on $\theta_{\text{proposed}}$).
  * Implement `SgPcnlScheduler`: implements step-size decay $\beta_t = \beta_0 (1 + \gamma \cdot \text{step})^{-0.55}$.
  * Update parameters in-place on GPU memory after each backward pass.

#### Phase 3: Bayesian Adaptation Pipeline (3–4 Days)
* **Tasks:**
  * **Pre-training / Initialization:** Initialize weights or train a base checkpoint on Task $\mathcal{A}$.
  * **Sequential Adaptation:** Feed Task $\mathcal{B}$ data $\mathcal{D}_{\text{new}}$. 
  * Run burn-in (e.g., 500 steps) to let the Langevin drift guide the chain into the new high-probability basin.
  * Collect posterior samples $S = \{\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \dots, \boldsymbol{\theta}^{(K)}\}$ thinned by an autocorrelation interval.
  * Implement **Bayesian Model Averaging (BMA)**:
    $$p(y \mid x, \mathcal{D}_{\text{new}}) \approx \frac{1}{K} \sum_{k=1}^K \text{softmax}\left( f(x; \boldsymbol{\theta}^{(k)}) \right)$$

#### Phase 4: Diagnostics & Benchmarks (2–3 Days)
* **MCMC Health Checks:**
  * Compute Effective Sample Size (ESS) and Gelman-Rubin $\hat{R}$ across multiple independent chains.
  * Check the radial norm distribution: Verify that $\|\boldsymbol{\theta}\|_2$ remains concentrated around $\sigma \sqrt{P}$ while the angular components adapt.
* **Evaluation Metrics:**
  * Measure Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL) on out-of-distribution (OOD) tests.
  * Benchmark against deterministic fine-tuning (SGD/Adam) and standard SGLD.

#### Phase 4b: The Predecessor Comparator, the Cyclical Schedule, and the Operational Claims (executed 2026-10-21b)

The specialist diff deleted the Metropolis-filtered engine. Deleting it
without measuring it would leave the report's central trade-off
(“throughput doubles, tuning shifts”) as an assertion, so the filtered
chain comes back **as a comparator** — the standard analogous algorithm
for the unadjusted update, same proposal, exact filter:

* **`task::mh_pcnl_chain`** — proposal identical to
  `step_unadjusted` ($\sqrt{1-\beta_t^2}\,\theta + \beta_t\xi -
  (\beta_t^2\sigma^2/2)\,g_{\text{batch}}$), then the exact MH filter
  $\log\alpha = L(\theta) - L(\theta')$ on the **full-batch** potential
  (the redundant second forward pass the diff removed, now counted).
  Same Robbins–Monro $\beta_t$, same thinning, so the *only*
  difference between the two chains is the filter.
* **Measured, per variant:** acceptance rate, radial shell, ESS, $\hat
  R$, BMA acc/NLL/ECE, wall-clock per step — and the **cross-$\hat{R}$
  over the filtered ∪ unfiltered loss chains** (same target ⇒ the two
  samplers' chains must agree; disagreement would localize bias).
* **Cyclical run (§3C Option 2):** `--cyclical PERIOD` switches every
  sampler chain to `SgPcnlScheduler::beta_cyclical`; run and compare
  against the Robbins–Monro run on the same seed.
* **Throughput accounting:** every chain family reports evaluations
  per step (deterministic counts, asserted in `tests/comparators.rs`)
  and seconds per step (measured on the full run). Claim 1 of the
  report is checked by counting, not by timing alone.

---

### 5. Concrete Rust Code Skeleton

Below is the core loop demonstrating how the proposal, gradient evaluation, and Metropolis step fit together cleanly in `candle`:

```rust
pub struct SgPcnlState {
    pub theta: Tensor,       // Contiguous parameter vector [P]
    pub sigma: f64,          // Prior std dev (isotropic)
    pub beta_0: f64,         // Initial step size
    pub gamma: f64,          // Decay rate
    pub kappa: f64,          // Decay exponent (e.g., 0.55)
    pub step_count: usize,
    pub dataset_size: usize,
    pub batch_size: usize,
}

impl SgPcnlState {
    /// Executes an in-place Unadjusted SG-pCNL update using mini-batch gradient.
    /// Memory complexity: O(B * T * d * L), strictly decoupled from total dataset size N.
    pub fn step_unadjusted(
        &mut self,
        batch_grad: &Tensor, // Evaluated on current mini-batch
        rng: &mut impl Rng,
    ) -> Result<()> {
        let p = self.theta.elem_count();
        let dev = self.theta.device();

        // 1. Compute scheduled step size beta_t
        let t = self.step_count as f64;
        let beta = self.beta_0 * (1.0 + self.gamma * t).powf(-self.kappa);
        self.step_count += 1;

        // 2. Sample isotropic Gaussian noise: xi ~ N(0, sigma^2 * I)
        let noise_vec: Vec<f32> = rng.sample_iter(StandardNormal)
            .take(p)
            .map(|x: f32| x * (self.sigma as f32))
            .collect();
        let xi = Tensor::from_vec(noise_vec, &[p], dev)?;

        // 3. Scale stochastic gradient to full dataset potential: g = (N / B) * g_batch
        let scale = (self.dataset_size as f64) / (self.batch_size as f64);
        let scaled_grad = (batch_grad * scale)?;

        // 4. Unadjusted pCNL Step:
        //    theta_{t+1} = sqrt(1 - beta^2) * theta_t + beta * xi - (beta^2 * sigma^2 / 2) * g
        let sqrt_decay = (1.0 - beta * beta).sqrt();
        let drift_coeff = (beta * beta * self.sigma * self.sigma) / 2.0;

        let theta_decay = (&self.theta * sqrt_decay)?;
        let noise_term = (&xi * beta)?;
        let drift_term = (&scaled_grad * drift_coeff)?;

        // Direct unconditional in-place assignment (No MH evaluation pass)
        self.theta = ((theta_decay + noise_term)? - drift_term)?;

        Ok(())
    }
}
```

### Key Operational Changes Summary

1. **Computational Throughput Doubles:** Standard pCNL required two forward evaluations per step (one at $\theta$, one at proposed $\theta'$). Unadjusted SG-pCNL requires only **one backward pass** per iteration on the current mini-batch. *(Measured 2026-10-21b: evaluation counts are deterministic — 2 vs 1 model evaluations per step, asserted in `pcnl/tests/comparators.rs::unadjusted_halves_model_evaluations`; wall-clock on the full run: the filtered chain costs ≈1.4–1.6× the unadjusted chain per step because its second pass is forward-only.)*
2. **VRAM Footprint Decoupled:** Peak memory consumption is entirely independent of dataset size $N$. It is governed purely by mini-batch size $B \times$ sequence length $T$. *(Structural: the chain state is $\theta \in \mathbb{R}^P$ + one streamed mini-batch; asserted by the batch-invariance test.)*
3. **Hyperparameter Tuning Shift:** Rather than dynamically tuning $\beta$ to track a $23.4\%$ MH acceptance target, tuning shifts to classical SGLD schedules: setting the initial step size $\beta_0$ and polynomial decay rate $\kappa \in (0.5, 1.0]$. *(Implemented — `SgPcnlScheduler`; the acceptance rate survives only as a *diagnostic* of the filtered comparator, where it measures what the filter would have rejected.)*
4. **The filter's cost was buying exactness — measured, not assumed:** the filtered and unfiltered chains share a proposal and a target, so their loss-chain cross-$\hat{R}$ isolates the unadjusted bias. On the full run the two agree (cross-$\hat{R}$ within the same range as the SGLD comparator's), i.e. at $\beta \le 0.3$ the exactness the filter bought is inside the sampling noise — the empirical content of “the drift and noise match to $O(\beta^4)$”.

### 6. Recommended Timeline & Milestones
* **Week 1:** Complete Phase 0 (validation in JAX) and Phase 1 (Rust architecture and weight flattening).
* **Week 2:** Implement Phase 2 (unadjusted SG-pCNL engine, mini-batch loader, Robbins-Monro $\beta$ schedule, safe tensors persistence).
* **Week 3:** Run adaptation experiments on a clean task (e.g., adapting an nGPT from synthetic Task A to Task B, tracking OOD calibration vs. Adam). Compile empirical metrics.

---

### 7. Formal Verification Status (Lean 4.28.0) — Milestones 1–5, proved 2026-10-21

The stability claims of §3B are now machine-checked in the `BayesianTransformer`
Lean library (formalization of *“Dimension-Robust Bayesian Adaptation of
Transformers via Hyperspherical Geometry and Unadjusted SG-pCNL”*). The work
was done by the LLM-Lean4-specialist and merged into the `timepiece` checkout
(`lakefile.toml` gains the `BayesianTransformer` lean_lib; the run record is
`timepiece/ARISTOTLE_SUMMARY.md`; the state entry is in
`timepiece/CONSOLIDATED_PLAN.md`). By instruction the local copy was **not
rebuilt** — `lake build BayesianTransformer` was green on the specialist's
machine, and re-running it is the specialist's lane.

**Proof hygiene.** No `sorry`, no `admit`, no `axiom` declarations; the main
theorems use only Lean's standard axioms (`propext`, `Classical.choice`,
`Quot.sound`). Lean 4.28.0 with Mathlib pinned to match (this supersedes the
v4.16.0 the milestone text once named).

#### 7.1–7.3 Inventory: file by file

| milestone | file | what is proved |
| :--- | :--- | :--- |
| **M1** | `BayesianTransformer/Analysis/SphereProj.lean` | `sphereProj_hasFDerivAt`: $D\bigl(v/\|v\|\bigr) = \|v\|^{-1}I - (\|v\|^3)^{-1}\, v \otimes v$. `sphereProj_fderiv_opNorm_le`: $\|D\| \le \|v\|^{-1}$ (the proof shows $h \mapsto h - (\langle v,h\rangle/\|v\|^2)\,v$ never increases norms — it is the tangential projector). `sphereProj_fderiv_opNorm_uniform`: $\|D\| \le \varepsilon^{-1}$ whenever $\|v\| \ge \varepsilon > 0$. |
| **M2** | `BayesianTransformer/Analysis/ScaleInvariance.lean` | `scaleInvariant_gradient_orthogonal` ($d\Phi(w)\,w = 0$) and `scaleInvariant_inner_grad_zero` ($\langle w, \nabla\Phi(w)\rangle = 0$), via the plan's checklist: the 1-D path $c \mapsto \Phi(cw)$, local constancy near $c=1$, chain rule. |
| **M3** | `BayesianTransformer/Transformer/CosineAttention.lean` | `cosineSimilarity_bounds`, `scaled_logit_bounds` (unit vectors ⇒ logits in $[-\tau,\tau]$); `softmax_min_prob_bound` ($p_i \ge e^{-\tau}/(\lvert d\rvert\, e^{\tau})$ for $\lvert z_i\rvert \le \tau$); `softmax_lipschitz` / `softmax_lipschitzWith_two`: softmax is **2-Lipschitz** ($d \to \mathbb{R}$, sup norm), plus the coordinate derivative bound it rests on. |
| **M4** | `BayesianTransformer/Transformer/GlobalLipschitz.lean` | `NGPTLayer`, `NGPTModel`, **Theorem 1** `nGPT_uniform_gradient_bound` (data-average bound, term-by-term differentiation + triangle inequality); the M1/M2→network bridges `normalized_fderiv_le{,_uniform}` (a block seeing only $\widehat W = W/\|W\|$ has composed derivative $\le K/\|W\| \le K/\varepsilon$) and `normalized_gradient_orthogonal`. |
| **M4+** | `BayesianTransformer/Transformer/NGPTArchitecture.lean`, `LipschitzCalculus.lean` | *Follow-up (same day):* the per-example bound is **derived from the architecture** instead of assumed — `ngpt_exampleLoss_lipOn` (one constant $M$ over the region, depending only on hyperparameters, $\varepsilon$, depth, sequence length), `ngpt_exampleLoss_fderiv_bound` ($\|\nabla_\theta \ell(\theta; x, y)\| \le M$ per example), `ngpt_empiricalRisk_fderiv_bound` (Theorem 1 for the network). `LipschitzCalculus.lean` carries the `LipOn` toolkit (composition, sums, inner products, the `2/δ`-Lipschitz normalization lemma). |
| **M5** | `BayesianTransformer/Probability/GradientNoise.lean` | `sg_pcnl_noise_covariance_bound`: $\|g_{\text{batch}} - g_{\text{full}}\| \le (N/B + 1)\,M$ (triangle inequality over the structure $\|g_{\text{batch}}\| \le (N/B)M$, $\|g_{\text{full}}\| \le M$). `sg_pcnl_noise_trace_bound`: for the mini-batch modelled as a finite distribution $p$ over realizations, $\mathrm{Tr}\,\Sigma(w) = \mathbb{E}\,\lVert g - \mathbb{E} g\rVert^2 \le ((N/B + 1) M)^2$ for every $w$ (with `noiseCovTrace_le`, the mean minimizes mean-squared deviation). |

#### 7.4 The architecture proof: assumptions that the reader must not lose

- **Region, not global.** Every ceiling holds on `paramDomain` — the set where
  **every weight has norm $\ge \varepsilon > 0$** — and a global bound is
  *impossible*: the loss does not change when a weight is rescaled, so the
  gradient grows like $1/\|W\|$ as $W \to 0$. (M1's uniform derivative bound is
  exactly what supplies the blow-up rate.) The numerical counterpart is pinned
  in `pcnl`: with exact normalization the measured $\lVert\nabla\rVert$ scales
  like $1/\|W\|$, and the implementation's $\varepsilon$-regularized
  $W/(\|W\|+\varepsilon)$ floors it — see `pcnl/tests/formal_claims.rs`.
- **Divergences from the nGPT paper**, needed for a *uniform* bound:
  matrices normalized by their **operator** norm (not row-by-row); queries, keys
  and sublayer outputs are **not** re-normalized (normalizing a possibly-zero
  vector is not Lipschitz — their norms stay $\le 1$ anyway); $s_{qk}, s_z,
  \alpha_A, \alpha_M$ are **fixed hyperparameters** with $0 \le \alpha < 1/2$
  (which keeps every hidden-state renormalization acting on a vector of norm
  $\ge 1 - 2\alpha > 0$); input embeddings are unit vectors *given* with each
  example, not trained parameters; at non-differentiable points (ReLU kinks)
  Lean defines `fderiv` as 0, so the **Lipschitz** bound is the substantive
  statement.
- **Activation class:** any $\sigma$ with Lipschitz constant $\le 1$ and
  $\sigma(0) = 0$ (`NGPTConfig.relu` proves the shape qualifies). *Honest
  divergence:* the `pcnl` crate uses GELU, whose optimal Lipschitz constant is
  $\approx 1.087 > 1$ — the crate and the proof therefore do not share an
  activation constant, and either the crate's GELU or the lemma's $1$ must move
  before the two lines of evidence are literally about one model.
- **Removed hypotheses** (statements strictly more general than the plan's
  drafts): `[CompleteSpace E]` (M1), `w ≠ 0` (M2), the unused `tau` argument of
  `softmax_lipschitz`, `data.Nonempty` in Theorem 1 (empty data ⇒ averaged loss
  is 0). Syntax: $\|v\|^{-3}$ is written `(‖v‖ ^ 3)⁻¹`. In M5, `hB`/`hBN` are
  explicit in `sg_pcnl_noise_covariance_bound` and implicit in
  `sg_pcnl_noise_trace_bound`.

#### 7.5 What the two M5 bounds do and do not say

The Lean ceiling $(N/B + 1)M$ is *pathwise and conservative* — it holds under
only the two norm premises, so it can never fail for the real estimator, but it
is strictly weaker than §3B's variance-form $(N^2/B)M^2$ (a triangle inequality
versus an independence/cancellation argument). The correct division of labour:
**Lean owns the worst case; the crate measures the typical case.**
`pcnl/tests/formal_claims.rs::batch_gradient_noise_below_the_pathwise_ceiling`
estimates $\mathbb{E}\,\lVert g_{\text{batch}} -
g_{\text{full}}\rVert^2$ over batch draws and asserts it sits below both
ceilings — the number that would have to move for §3B.2's sharper form to be
false — and `..._covariance_shrinks_with_batch_size` checks the $1/B$
scaling the variance form predicts (the form itself stays empirical:
a theorem needs an independence argument Lean does not yet have).

#### 7.6 Still prose (next steps for the Lean4-specialist)

1. **Ergodicity / convergence of the unadjusted chain** under the
   Robbins–Monro schedule (§3B.3) — nothing in `BayesianTransformer` speaks to
   it; the noise ceiling is an ingredient, not the theorem.
2. **BMA & diagnostics** (§3 Phase 3/4: BMA accuracy, ECE, ESS, split-R̂) —
   statistical statements with no Lean counterpart; their contract is the
   numerical suite.
3. **Bridge the normalization conventions:** operator-norm normalization in the
   proof vs Frobenius normalization in `pcnl::normalize` (equivalence is
   norm-equivalence up to $\sqrt{\min(d_{\text{in}}, d_{\text{out}})}$ — state
   which constant inherits where), and nGPT's row-wise normalization.
4. **Activation constant:** cover GELU ($\le 1.087$) or re-scale, closing the
   divergence in §7.4.
5. **Trainable $s_{qk}, s_z, \alpha$** — the fixed-hyperparameter condition is
   load-bearing; a version with bounded trainable scalars is the honest next
   generalization.
6. **Row-wise and slerp residuals** — the crate's nlerp residuals and the
   paper's per-row normalization each need either a lemma or an explicit
   “out of scope” note.
