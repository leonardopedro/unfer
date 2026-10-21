> **Status (2026-10-20):** implemented as workspace crate `pcnl/`;
> the full Week-3 run's measured numbers and plan-conformance notes are
> in [`docs/PCNL_STATUS.md`](PCNL_STATUS.md).

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
2. The stochastic gradient covariance is strictly bounded:
   $$\mathbb{E}\left[ \|g_{\text{batch}} - g_{\text{full}}\|^2 \right] \le (N^2 / B) \cdot M^2$$
3. This bounded perturbation prevents the chain from escaping into degenerate regions,
   guaranteeing geometric ergodicity under Robbins-Monro annealing.

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

1. **Computational Throughput Doubles:** Standard pCNL required two forward evaluations per step (one at $\theta$, one at proposed $\theta'$). Unadjusted SG-pCNL requires only **one backward pass** per iteration on the current mini-batch.
2. **VRAM Footprint Decoupled:** Peak memory consumption is entirely independent of dataset size $N$. It is governed purely by mini-batch size $B \times$ sequence length $T$.
3. **Hyperparameter Tuning Shift:** Rather than dynamically tuning $\beta$ to track a $23.4\%$ MH acceptance target, tuning shifts to classical SGLD schedules: setting the initial step size $\beta_0$ and polynomial decay rate $\kappa \in (0.5, 1.0]$.

### 6. Recommended Timeline & Milestones
* **Week 1:** Complete Phase 0 (validation in JAX) and Phase 1 (Rust architecture and weight flattening).
* **Week 2:** Implement Phase 2 (unadjusted SG-pCNL engine, mini-batch loader, Robbins-Monro $\beta$ schedule, safe tensors persistence).
* **Week 3:** Run adaptation experiments on a clean task (e.g., adapting an nGPT from synthetic Task A to Task B, tracking OOD calibration vs. Adam). Compile empirical metrics.
