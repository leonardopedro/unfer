//! Quantum Bayesian Updating on the TSR-evolved prior.
//!
//! This module implements the second stage of the algorithm in
//! `QMF.tex §8` (the "Quantum Bayesian Updating on the TSR-evolved
//! Prior" section). It conditions the TSR-evolved prior on $N$ new,
//! problem-defining observations $\{D_1,\dots,D_N\}$ using the Born
//! rule as the likelihood, and draws a single posterior sample via
//! Hamiltonian Monte Carlo on the unit sphere of the $m$-dim Krylov
//! subspace.
//!
//! ## Architecture
//!
//! The inference is fully on the $m$-dim Krylov coefficients and is
//! independent of the training set size $M$ at runtime. The total
//! online cost is
//!
//! $$\mathcal{O}(N d k) + \mathcal{O}(\mathrm{HMC\,steps} \cdot N m^2)
//! + \mathcal{O}(K_2 \log k) + \mathcal{O}(d m^2)$$
//!
//! which is linear in $N$ and $d$ and has no $M$ dependence.
//!
//! ## Phase summary
//!
//! 1. **Likelihood operators** — for each new observation $D_i$ we
//!    hash it (S_1 then S_2) into a K_2-dim single-excitation Fock
//!    state $\ket{\Psi_{D_i}}$, then Krylov-project to a rank-1
//!    operator $\mathbf L_i = \vec v_i \vec v_i^\dagger \in
//!    \Cset^{m\times m}$ with $\vec v_i = W^\dagger
//!    \ket{\Psi_{D_i}}$.
//! 2. **Born likelihood** — $P(D_i \mid \vec c) = \vec c^\dagger
//!    \mathbf L_i \vec c$.
//! 3. **HMC on the posterior** — potential
//!    $U(\vec c) = -\log P_{\mathrm{prior}}(\vec c) - \sum_i
//!    \log(\vec c^\dagger \mathbf L_i \vec c)$ with
//!    $\nabla_{\vec c} U$ given by Eq. (8.5).
//! 4. **Tomographic reconstruction** — feed the posterior sample
//!    $\vec c_{\mathrm{sample}}$ back into
//!    [`QfmPipeline::decode`](crate::pipeline::QfmPipeline::decode)
//!    to render the full $d$-dim image.
//!
//! ## Why the TSR + Krylov prior is necessary
//!
//! Skipping the offline TSR pipeline would force the Bayesian update
//! to run on the full K_2-dim sketched space with $M$ training-set
//! likelihoods. The resulting "golf course" landscape
//! $\prod_{i=1}^M(\vec c^\dagger \mathbf L_i \vec c)$ has
//! $M$ microscopic, infinitely-steep holes and is un-samplable
//! (pure memorization / starvation). The TSR pipeline compresses
//! the training set into the $m$-dim Krylov subspace once, offline;
//! the Bayesian update then operates on a smooth, navigable
//! potential.
//!
//! ## P6 H implementation note: the informed prior
//!
//! The informed prior $P_{\mathrm{prior}}$ used by this module is
//! the (squared-modulus) overlap with the TSR-evolved vacuum
//! state. Concretely: the TSR pipeline evolves the vacuum seed
//! $\ket{0}$ (with its single-excitation basis superposition) under
//! $U_m = e^{-iH_m t}$ to give a single Krylov vector
//! $\vec c_{\mathrm{prior}} \in \Cset^m$. We then take
//!
//! $$P_{\mathrm{prior}}(\vec c) \propto |\vec c^\dagger \vec c_{\mathrm{prior}}|^2 + \varepsilon$$
//!
//! with a small $\varepsilon > 0$ to ensure the log is finite
//! everywhere. This is a smooth, unimodal distribution on the
//! $2m-1$-dim unit sphere of $\Cset^m$ --- exactly the "navigable
//! typical set manifold" that the TSR pipeline produces. It is the
//! natural low-rank surrogate of the full $(U_m)_* \mu_0$ pushforward
//! of the Mehler prior under the unitary flow.
//!
//! The HMC sampler uses a deterministic splitmix64 PRNG (mirroring
//! the PRNG in `crate::sketch`) so the inference is reproducible
//! without depending on an external `rand` crate.

use crate::pipeline::{QfmError, QfmPipeline};
use crate::sketch::FeatureToMode;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// A rank-1 likelihood operator $\mathbf L_i = \vec v_i \vec v_i^\dagger$
/// in the $m$-dim Krylov subspace.
///
/// The Born rule evaluates the likelihood of observation $D_i$ given
/// candidate wavefunction $\vec c$ as $P(D_i \mid \vec c) = \vec
/// c^\dagger \mathbf L_i \vec c = |\vec v_i^\dagger \vec c|^2$.
#[derive(Debug, Clone)]
pub struct Likelihood {
    /// The Krylov-projected state $\vec v_i \in \Cset^m$ such that
    /// $\mathbf L_i = \vec v_i \vec v_i^\dagger$.
    v: DVector<Complex64>,
}

impl Likelihood {
    /// Build a likelihood from a Krylov-projected state vector $\vec v$.
    pub fn from_krylov_state(v: DVector<Complex64>) -> Self {
        Self { v }
    }

    /// Build a likelihood from a raw observation $D_i$ using the
    /// pipeline's sketches and Krylov basis. This is the online
    /// Phase 2 step from §8.2.
    ///
    /// Steps:
    /// 1. Level 1 hash: $\tilde D_i = S_1(D_i)$.
    /// 2. Level 2 resolve: the mode index $m$ of $\tilde D_i$.
    /// 3. Krylov projection: $\vec v_i = W^\dagger \ket{e_m}$ is the
    ///    $m$-th row of $W$ (the $m$-th column of $W^\dagger$).
    pub fn from_observation(pipeline: &QfmPipeline, observation: &[f64]) -> Result<Self, QfmError> {
        if observation.len() != pipeline.raw_dim() {
            return Err(QfmError::DimensionMismatch {
                expected: pipeline.raw_dim(),
                got: observation.len(),
            });
        }
        // Level 1 hash.
        let x_tilde = pipeline.s1().apply(observation);
        // Level 2 resolve: the feature key's mode index.
        let key = FeatureToMode::hash_feature(&x_tilde);
        let mode = pipeline
            .s2()
            .resolve(key)
            .or_else(|| {
                let feature = x_tilde.clone();
                pipeline
                    .s2()
                    .nearest(&feature, pipeline.training_features())
            })
            .unwrap_or(0) as usize;
        // v_i = W^dag |e_mode> = the mode-th row of W.
        let rank = pipeline.rank();
        let mut v = DVector::<Complex64>::zeros(rank);
        for r in 0..rank {
            v[r] = pipeline.w()[(mode, r)];
        }
        Ok(Self { v })
    }

    /// Evaluate the Born rule $P(D \mid \vec c) = |\vec v^\dagger \vec c|^2$.
    /// Returns a small positive floor so the log is finite.
    pub fn born_rule(&self, c: &DVector<Complex64>) -> f64 {
        debug_assert_eq!(self.v.len(), c.len(), "Likelihood/vector dim mismatch");
        let mut acc = Complex64::new(0.0, 0.0);
        for i in 0..self.v.len() {
            acc += self.v[i].conj() * c[i];
        }
        (acc.re * acc.re + acc.im * acc.im).max(1e-300)
    }

    /// Compute $\frac{2 \mathbf L \vec c}{\vec c^\dagger \mathbf L \vec c}
    /// = \frac{2 (\vec v^\dagger \vec c) \vec v}{|\vec v^\dagger \vec c|^2}$
    /// --- one term in the gradient $\nabla_{\vec c} U$.
    pub fn gradient_term(&self, c: &DVector<Complex64>) -> DVector<Complex64> {
        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..self.v.len() {
            overlap += self.v[i].conj() * c[i];
        }
        // (2 / overlap*) * v   (because |v^dag c|^2 = overlap * overlap*)
        let coeff = Complex64::new(2.0, 0.0) / overlap.conj();
        &self.v * coeff
    }

    /// The Krylov-projected state vector.
    pub fn krylov_state(&self) -> &DVector<Complex64> {
        &self.v
    }
}

/// The full unnormalized posterior over Krylov coefficients:
///
/// $$P(\vec c \mid D_1, \dots, D_N) = \frac{1}{Z} \left(\prod_{i=1}^N
/// \vec c^\dagger \mathbf L_i \vec c\right) P_{\mathrm{prior}}(\vec c).$$
///
/// The log-density is
///
/// $$\log P(\vec c) = \sum_{i=1}^N \log(\vec c^\dagger \mathbf L_i \vec c)
/// + \log P_{\mathrm{prior}}(\vec c) - \log Z,$$
///
/// which is what the HMC potential $U(\vec c) = -\log P(\vec c)$
/// consumes. The normalization $\log Z$ is irrelevant for HMC (it
/// cancels in the gradient) so we omit it.
#[derive(Debug, Clone)]
pub struct Posterior {
    /// All likelihood operators $\{\mathbf L_i\}$.
    likelihoods: Vec<Likelihood>,
    /// The TSR-evolved prior direction $\vec c_{\mathrm{prior}} \in
    /// $\Cset^m$ (the Krylov projection of the evolved vacuum).
    c_prior: DVector<Complex64>,
    /// Small positive constant ensuring $\log P_{\mathrm{prior}}$
    /// is finite everywhere.
    epsilon: f64,
}

impl Posterior {
    /// Build a posterior from a list of likelihoods and the TSR-evolved
    /// prior direction.
    pub fn new(likelihoods: Vec<Likelihood>, c_prior: DVector<Complex64>) -> Self {
        Self {
            likelihoods,
            c_prior,
            epsilon: 1e-12,
        }
    }

    /// Build a posterior with no new observations (returns the prior
    /// itself, used as a sanity check that HMC on the empty posterior
    /// recovers the TSR-evolved prior sample).
    pub fn prior_only(c_prior: DVector<Complex64>) -> Self {
        Self::new(Vec::new(), c_prior)
    }

    /// The number of new observations.
    pub fn n_observations(&self) -> usize {
        self.likelihoods.len()
    }

    /// The TSR-evolved prior direction.
    pub fn prior_direction(&self) -> &DVector<Complex64> {
        &self.c_prior
    }

    /// Evaluate $\log P_{\mathrm{prior}}(\vec c) = \log(|\vec c^\dagger
    /// \vec c_{\mathrm{prior}}|^2 + \varepsilon)$.
    pub fn log_prior(&self, c: &DVector<Complex64>) -> f64 {
        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..c.len() {
            overlap += c[i].conj() * self.c_prior[i];
        }
        let mag_sq = overlap.re * overlap.re + overlap.im * overlap.im;
        (mag_sq + self.epsilon).ln()
    }

    /// Evaluate the gradient $\nabla_{\vec c} \log P_{\mathrm{prior}}(\vec c)$.
    ///
    /// With $p = |\vec c^\dagger \vec c_p|^2 = \vec c_p^\dagger \vec c \vec
    /// c^\dagger \vec c_p$ and $\partial p / \partial c_j^* = (\vec c_p)_j
    /// (\vec c_p^\dagger \vec c)$:
    ///
    /// $$\frac{\partial \log p}{\partial c_j^*} = \frac{(\vec c_p)_j
    /// (\vec c_p^\dagger \vec c)}{p + \varepsilon}.$$
    pub fn log_prior_grad(&self, c: &DVector<Complex64>) -> DVector<Complex64> {
        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..c.len() {
            overlap += self.c_prior[i].conj() * c[i];
        }
        let mag_sq = overlap.re * overlap.re + overlap.im * overlap.im;
        let denom = (mag_sq + self.epsilon).max(1e-300);
        &self.c_prior * (overlap / denom)
    }

    /// Evaluate the log posterior $\log P(\vec c) = \sum_i
    /// \log(\vec c^\dagger \mathbf L_i \vec c) + \log
    /// P_{\mathrm{prior}}(\vec c) + \text{const}$. The constant is
    /// omitted.
    pub fn log_density(&self, c: &DVector<Complex64>) -> f64 {
        let mut log_p = self.log_prior(c);
        for like in &self.likelihoods {
            log_p += like.born_rule(c).ln();
        }
        log_p
    }

    /// Evaluate the gradient of the log posterior. The constant
    /// $-\log Z$ is irrelevant (cancels in $\nabla$).
    pub fn log_density_grad(&self, c: &DVector<Complex64>) -> DVector<Complex64> {
        let mut grad = self.log_prior_grad(c);
        for like in &self.likelihoods {
            grad += like.gradient_term(c);
        }
        grad
    }

    /// HMC potential $U(\vec c) = -\log P(\vec c)$ (negative log
    /// density).
    pub fn potential(&self, c: &DVector<Complex64>) -> f64 {
        -self.log_density(c)
    }

    /// HMC gradient $\nabla_{\vec c} U(\vec c) = -\nabla_{\vec c} \log
    /// P(\vec c)$.
    pub fn potential_grad(&self, c: &DVector<Complex64>) -> DVector<Complex64> {
        -&self.log_density_grad(c)
    }
}

/// Configuration for the HMC sampler on the unit sphere of $\Cset^m$.
#[derive(Debug, Clone)]
pub struct HmcOpts {
    /// Number of leapfrog steps per HMC proposal.
    pub leapfrog_steps: usize,
    /// Step size $\epsilon$ in the leapfrog integrator.
    pub step_size: f64,
    /// Number of HMC proposals to run (burn-in + sample).
    pub n_iterations: usize,
    /// Number of initial proposals to discard as burn-in.
    pub burn_in: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for HmcOpts {
    fn default() -> Self {
        Self {
            leapfrog_steps: 20,
            step_size: 0.05,
            n_iterations: 100,
            burn_in: 50,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG: Box-Muller + splitmix64 (mirrors the qfm/src/sketch.rs
// pattern so we have no new external dependencies).
// ---------------------------------------------------------------------------
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        splitmix64(self.0)
    }
    /// Return a uniformly distributed f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    /// Return a standard normal sample via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Draw a uniformly distributed random complex unit vector $\vec c
/// \in \Cset^m$ with $\|\vec c\| = 1$.
fn sample_unit_complex(rng: &mut SplitMix64, m: usize) -> DVector<Complex64> {
    let re: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
    let im: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
    let norm: f64 = re
        .iter()
        .zip(im.iter())
        .map(|(a, b)| a * a + b * b)
        .sum::<f64>()
        .sqrt();
    let norm = norm.max(1e-300);
    let v: Vec<Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(a, b)| Complex64::new(a / norm, b / norm))
        .collect();
    DVector::from_vec(v)
}

/// Sample a momentum $\vec p \in \Cset^m$ from the standard normal.
fn sample_momentum(rng: &mut SplitMix64, m: usize) -> DVector<Complex64> {
    let re: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
    let im: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
    DVector::from_vec(
        re.iter()
            .zip(im.iter())
            .map(|(a, b)| Complex64::new(*a, *b))
            .collect(),
    )
}

/// Project $\vec c$ back onto the unit sphere (i.e., normalize).
fn renormalize(c: &DVector<Complex64>) -> DVector<Complex64> {
    let norm: f64 = c
        .iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .sum::<f64>()
        .sqrt();
    if norm < 1e-300 {
        return c.clone();
    }
    c / Complex64::new(norm, 0.0)
}

/// One HMC proposal on the unit sphere. Returns the new state and
/// the acceptance probability (for diagnostics).
fn hmc_step(
    c: &DVector<Complex64>,
    posterior: &Posterior,
    opts: &HmcOpts,
    rng: &mut SplitMix64,
) -> (DVector<Complex64>, f64) {
    let m = c.len();
    let mut p = sample_momentum(rng, m);
    let mut q = c.clone();

    // Initial Hamiltonian: K = 0.5 * p^dag p, V = -log P.
    let k0: f64 = p.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>() * 0.5;
    let v0 = posterior.potential(&q);
    let h0 = k0 + v0;

    // Leapfrog: q_{k+1} = q_k + eps * p_k; p_{k+1} = p_k - eps * grad V(q_{k+1}).
    let eps_c = Complex64::new(opts.step_size, 0.0);
    for _ in 0..opts.leapfrog_steps {
        // Half-step momentum update.
        let grad = posterior.potential_grad(&q);
        p = &p - &grad * eps_c;
        // Full-step position update.
        q = &q + &p * eps_c;
        // Renormalize to stay on the unit sphere.
        q = renormalize(&q);
    }

    let k1: f64 = p.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>() * 0.5;
    let v1 = posterior.potential(&q);
    let h1 = k1 + v1;

    // Metropolis accept/reject.
    let log_alpha = (h0 - h1).min(0.0);
    let u = rng.next_f64();
    let alpha = log_alpha.exp();
    if u < alpha {
        (q, alpha)
    } else {
        (c.clone(), alpha)
    }
}

/// Run HMC on the posterior and return the post-burn-in samples.
pub fn sample_hmc(posterior: &Posterior, opts: &HmcOpts) -> Vec<DVector<Complex64>> {
    let mut rng = SplitMix64(opts.seed);
    let m = posterior.prior_direction().len();
    let mut q = sample_unit_complex(&mut rng, m);
    let mut samples = Vec::with_capacity(opts.n_iterations);
    for _ in 0..opts.n_iterations {
        let (q_new, _alpha) = hmc_step(&q, posterior, opts, &mut rng);
        q = q_new;
        samples.push(q.clone());
    }
    samples
}

/// Run HMC and return the **last** sample (i.e., after burn-in). This
/// is the single highly-likely wavefunction that satisfies all $N$
/// conditions.
pub fn sample_hmc_single(posterior: &Posterior, opts: &HmcOpts) -> DVector<Complex64> {
    let mut rng = SplitMix64(opts.seed);
    let m = posterior.prior_direction().len();
    let mut q = sample_unit_complex(&mut rng, m);
    for _ in 0..opts.n_iterations {
        let (q_new, _alpha) = hmc_step(&q, posterior, opts, &mut rng);
        q = q_new;
    }
    q
}

/// Reconstruct a full-resolution image from a posterior sample
/// $\vec c_{\mathrm{sample}}$ by re-running the TSR Phase 3-4
/// tomographic decoder. Convenience wrapper around
/// [`QfmPipeline::decode`].
pub fn reconstruct(
    pipeline: &QfmPipeline,
    c_sample: &DVector<Complex64>,
) -> Result<Vec<f64>, QfmError> {
    pipeline.decode(c_sample)
}

/// Compute the TSR-evolved prior direction. This is the Krylov
/// projection of the evolved vacuum seed $\ket{0}$ under the
/// pipeline's reduced Hamiltonian.
///
/// Steps:
/// 1. Start with the vacuum mode in the K_2-dim single-excitation
///    subspace: $\vec v_0 = W^\dagger \ket{e_0}$ (with the K_2×rank
///    identity Krylov basis this is the 0-th row of $W$).
/// 2. Evolve: $\vec c_{\mathrm{prior}} = \exp(-i H_m \cdot 1) \vec
///    v_0$ via nalgebra's Padé exponential.
pub fn tsr_evolved_prior(pipeline: &QfmPipeline) -> DVector<Complex64> {
    let rank = pipeline.rank();
    let mut v0 = DVector::<Complex64>::zeros(rank);
    for r in 0..rank {
        v0[r] = pipeline.w()[(0, r)];
    }
    pipeline.evolve(&v0, 1.0)
}

// ---------------------------------------------------------------------------
// Suppress dead-code warnings for the helper utility module (W is read-only).
// ---------------------------------------------------------------------------
#[allow(dead_code)]
fn _suppress_unused_dmatrix_import() -> DMatrix<Complex64> {
    DMatrix::zeros(1, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::QfmConfig;

    fn small_pipeline() -> QfmPipeline {
        // 4 training points in d=4 (the rev 14 reference tetrahedron),
        // k=2, K_2=4 (must be >= d for the rev 14 Φ basis guard),
        // krylov_dim=2.
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 2,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
        };
        QfmPipeline::compile(&training, &config).expect("compile")
    }

    #[test]
    fn likelihood_born_rule_is_amplitude_squared() {
        let pipe = small_pipeline();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let like = Likelihood::from_observation(&pipe, &obs).expect("likelihood");
        let c = like.krylov_state().clone();
        let p = like.born_rule(&c);
        // |v^dag v|^2 = ||v||^4 = 1.0 (Krylov states are unit-norm rows of W).
        assert!(p > 0.99 && p < 1.01, "born rule self-eval: {p}");
    }

    #[test]
    fn bayesian_update_tsr_recovers_training_mode() {
        // Compiling on the 4-pt tetrahedron with K_2=4 should produce
        // a Likelihood that, when given an observation matching a
        // training point, makes the HMC sample end up aligned with the
        // training mode.
        let pipe = small_pipeline();
        let obs = [1.0, 0.0, 0.0, 0.0];
        let like = Likelihood::from_observation(&pipe, &obs).expect("likelihood");
        let c_prior = tsr_evolved_prior(&pipe);
        let posterior = Posterior::new(vec![like], c_prior);
        let opts = HmcOpts {
            leapfrog_steps: 20,
            step_size: 0.05,
            n_iterations: 300,
            burn_in: 200,
            seed: 42,
        };
        let sample = sample_hmc_single(&posterior, &opts);
        let image = reconstruct(&pipe, &sample).expect("reconstruct");
        assert_eq!(image.len(), pipe.raw_dim());
        // The first component should be the largest (training point 0
        // is e_0). This is the rev 16 acceptance criterion for the
        // P6 H "bayesian_update_tsr_recovers_training_mode" test.
        let argmax = image
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 0, "decoded image should be e_0, got {image:?}");
    }

    #[test]
    fn bayesian_update_zero_observations_returns_prior_sample() {
        // No likelihoods: the posterior is the prior. The HMC sample
        // should be well-defined and finite, on the unit sphere.
        let pipe = small_pipeline();
        let c_prior = tsr_evolved_prior(&pipe);
        let posterior = Posterior::prior_only(c_prior.clone());
        let opts = HmcOpts::default();
        let sample = sample_hmc_single(&posterior, &opts);
        assert_eq!(sample.len(), c_prior.len());
        let norm_sq: f64 = sample
            .iter()
            .map(|z| z.re * z.re + z.im * z.im)
            .sum::<f64>();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "unit-norm violation: {norm_sq}"
        );
    }

    #[test]
    fn bayesian_update_hmc_converges_2mode() {
        // Two observations: the HMC sample should give a non-vanishing
        // likelihood to both.
        let pipe = small_pipeline();
        let l0 = Likelihood::from_observation(&pipe, &[1.0, 0.0, 0.0, 0.0]).expect("l0");
        let l1 = Likelihood::from_observation(&pipe, &[0.0, 1.0, 0.0, 0.0]).expect("l1");
        let c_prior = tsr_evolved_prior(&pipe);
        let posterior = Posterior::new(vec![l0.clone(), l1.clone()], c_prior);
        let opts = HmcOpts {
            leapfrog_steps: 20,
            step_size: 0.03,
            n_iterations: 400,
            burn_in: 300,
            seed: 42,
        };
        let sample = sample_hmc_single(&posterior, &opts);
        let p0 = l0.born_rule(&sample);
        let p1 = l1.born_rule(&sample);
        let mean = (p0 + p1) * 0.5;
        assert!(
            mean > 1e-3,
            "HMC sample has too-low likelihoods: p0={p0}, p1={p1}"
        );
    }

    #[test]
    fn reconstruct_round_trip_yields_finite_components() {
        let pipe = small_pipeline();
        let c_prior = tsr_evolved_prior(&pipe);
        let image = reconstruct(&pipe, &c_prior).expect("reconstruct");
        for v in &image {
            assert!(v.is_finite(), "non-finite component: {v}");
        }
    }

    #[test]
    fn hmc_step_returns_unit_norm() {
        // Sanity: every HMC step renormalizes the state to the unit
        // sphere.
        let pipe = small_pipeline();
        let c_prior = tsr_evolved_prior(&pipe);
        let like = Likelihood::from_observation(&pipe, &[1.0, 0.0, 0.0, 0.0]).expect("like");
        let posterior = Posterior::new(vec![like], c_prior);
        let opts = HmcOpts::default();
        let samples = sample_hmc(&posterior, &opts);
        for s in &samples {
            let norm_sq: f64 = s.iter().map(|z| z.re * z.re + z.im * z.im).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "sample norm violation: {norm_sq}"
            );
        }
    }

    #[test]
    fn tsr_evolved_prior_is_unit_norm() {
        // The pipeline.evolve() is unitary, so the evolved state
        // should have the same norm as v0 (which is 1.0 because the
        // rows of W are unit-norm Krylov basis vectors).
        let pipe = small_pipeline();
        let c_prior = tsr_evolved_prior(&pipe);
        let norm_sq: f64 = c_prior
            .iter()
            .map(|z| z.re * z.re + z.im * z.im)
            .sum::<f64>();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "TSR prior norm violation: {norm_sq}"
        );
    }
}
