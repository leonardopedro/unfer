### 1. Technology Stack Selection & Justification

| Layer | Recommended Choice | Justification & Alternatives |
| :--- | :--- | :--- |
| **Tensor & Autograd Engine** | **`candle` (Hugging Face)** | Minimalist, idiomatic Rust with native autograd (`candle-core`, `candle-nn`). It has zero heavy C++ dependencies (unlike `tch-rs`/LibTorch) while supporting CUDA, Metal, and AVX CPU backends. *(Alternative: `burn` is viable, but `candle` offers cleaner low-level pointer and gradient manipulation for custom MCMC loops).* |
| **MCMC & Math Core** | **`rand` + `rand_distr`** | High-performance, cryptographically reproducible Gaussian noise sampling ($\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$) using deterministic seeds (`ChaCha8Rng`). |
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

### 3. Mathematical Formulation of the pCNL Sampler

To adapt the model to new dataset $\mathcal{D}$, the continuous prior is chosen as $\boldsymbol{\theta} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.

#### A. Proposal Step
Given current weights $\boldsymbol{\theta}$ and data potential $\Phi(\boldsymbol{\theta}) = \mathcal{L}(\mathcal{D}; \boldsymbol{\theta})$:
1. Sample noise: $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.
2. Compute loss gradient: $\mathbf{g} = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\mathcal{D}; \boldsymbol{\theta})$ via `candle` backward pass.
3. Propose $\boldsymbol{\theta}'$:
   $$\boldsymbol{\theta}' = \sqrt{1 - \beta^2}\,\boldsymbol{\theta} + \beta\,\boldsymbol{\xi} - \frac{\beta^2 \sigma^2}{2} \mathbf{g}$$
   where $\beta \in (0, 1]$ is the step-size parameter.

#### B. Exact Metropolis–Hastings Acceptance Ratio
To guarantee exact detailed balance under Langevin drift (accounting for the drift transition asymmetry), the acceptance probability $\alpha(\boldsymbol{\theta}, \boldsymbol{\theta}')$ is:
$$\alpha(\boldsymbol{\theta}, \boldsymbol{\theta}') = \min\left(1, \, \exp\left( -\mathcal{L}(\boldsymbol{\theta}') + \mathcal{L}(\boldsymbol{\theta}) - \Delta_{\text{drift}}(\boldsymbol{\theta}, \boldsymbol{\theta}') \right)\right)$$
where the drift correction term is:
$$\Delta_{\text{drift}}(\boldsymbol{\theta}, \boldsymbol{\theta}') = \frac{1}{2} \langle \mathbf{g} + \mathbf{g}', \, \boldsymbol{\theta}' - \sqrt{1 - \beta^2}\boldsymbol{\theta} \rangle + \frac{\beta^2 \sigma^2}{8} \left( \|\mathbf{g}\|^2 - \|\mathbf{g}'\|^2 \right)$$
*(Note: If $\beta$ is small or you use pure pCN without drift, $\Delta_{\text{drift}} = 0$. In nGPT, because $\|\mathbf{g}\|$ is strictly bounded, $\Delta_{\text{drift}}$ remains numerically well-conditioned).*

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
  * Run pCNL on standard Softmax vs. Scaled Cosine/nGPT attention.
  * Confirm that standard attention suffers from acceptance collapse ($\alpha \to 0$ due to gradient spikes), while nGPT attention maintains steady acceptance rates ($\alpha \approx 20\% - 40\%$).

#### Phase 1: Rust Foundation & Normalized Architecture (3–5 Days)
* **Crates:** `candle-core`, `candle-nn`.
* **Tasks:**
  * Implement the `Normalize` operator:
    $$\widehat{X} = \frac{X}{\|X\|_2 + \epsilon}$$
  * Implement `NormalizedLinear`: Holds raw tensor $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, computes $W / \|W\|_F$ on the fly during `forward()`.
  * Implement `CosineSelfAttention`:
    $$A_{ij} = \text{softmax}\left(\tau \cdot \frac{q_i \cdot k_j}{\|q_i\| \|k_j\| + \epsilon}\right)$$
  * Construct `FlatParameters`: A struct that packs all trainable tensors into a single contiguous `candle::Tensor` of shape $[P]$ and unpacks them back into module weight references.

#### Phase 2: The pCNL Engine in Rust (3–4 Days)
* **Crates:** `rand`, `rand_distr`, `rayon` (for parallel chains).
* **Tasks:**
  * Implement the proposal kernel:
    ```rust
    pub struct PcnlSampler {
        pub beta: f64,
        pub sigma: f64,
        pub current_theta: Tensor,
        pub current_loss: f64,
        pub current_grad: Tensor,
    }
    ```
  * Implement proposal generation and backward pass evaluation.
  * Implement the Metropolis accept/reject check using uniform sampling $\log u < \log \alpha$.
  * Add an **Adaptive Step-Size ($\beta$) Controller**: Dual averaging (similar to NUTS) targeting an optimal acceptance rate of $\sim 23.4\%$ to $50\%$.

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
use candle_core::{Result, Tensor, DType, Device};
use rand::distributions::StandardNormal;
use rand::prelude::*;

pub struct PcnlState {
    pub theta: Tensor,
    pub loss: f64,
    pub grad: Tensor,
    pub beta: f64,
    pub sigma: f64,
}

impl PcnlState {
    pub fn step<F>(&mut self, loss_and_grad_fn: F, rng: &mut impl Rng) -> Result<bool>
    where
        F: Fn(&Tensor) -> Result<(f64, Tensor)>,
    {
        let p = self.theta.elem_count();
        let dev = self.theta.device();

        // 1. Sample isotropic Gaussian noise: xi ~ N(0, sigma^2 * I)
        let noise_vec: Vec<f32> = rng.sample_iter(StandardNormal)
            .take(p)
            .map(|x: f32| x * (self.sigma as f32))
            .collect();
        let xi = Tensor::from_vec(noise_vec, &[p], dev)?;

        // 2. Compute pCNL Proposal: theta' = sqrt(1 - beta^2) * theta + beta * xi - (beta^2 * sigma^2 / 2) * grad
        let sqrt_decay = (1.0 - self.beta * self.beta).sqrt() as f64;
        let drift_coeff = (self.beta * self.beta * self.sigma * self.sigma / 2.0) as f64;

        let theta_decay = (&self.theta * sqrt_decay)?;
        let noise_term = (&xi * self.beta)?;
        let drift_term = (&self.grad * drift_coeff)?;

        let theta_proposed = ((theta_decay + noise_term)? - drift_term)?;

        // 3. Evaluate potential and gradient at proposed state
        let (proposed_loss, proposed_grad) = loss_and_grad_fn(&theta_proposed)?;

        // 4. Compute log acceptance ratio: log(alpha) = -L(theta') + L(theta) - drift_correction
        let log_likelihood_diff = self.loss - proposed_loss;
        
        // (Optional: include exact Delta_drift here if beta is large)
        let log_alpha = log_likelihood_diff; 

        // 5. Metropolis Accept/Reject Step
        let u: f64 = rng.gen();
        if (u.ln()) < log_alpha {
            self.theta = theta_proposed;
            self.loss = proposed_loss;
            self.grad = proposed_grad;
            Ok(true) // Accepted
        } else {
            Ok(false) // Rejected, retain current state
        }
    }
}
```

### 6. Recommended Timeline & Milestones
* **Week 1:** Complete Phase 0 (validation in JAX) and Phase 1 (Rust architecture and weight flattening).
* **Week 2:** Implement Phase 2 (pCNL sampler, dual-averaging step-size controller, safe tensors persistence).
* **Week 3:** Run adaptation experiments on a clean task (e.g., adapting an nGPT from synthetic Task A to Task B, tracking OOD calibration vs. Adam). Compile empirical metrics.
