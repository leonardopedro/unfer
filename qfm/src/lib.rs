//! Non-Neural Quantum Flow Matching (QFM) with Tomographic Subspace Recovery.
//!
//! This crate implements the algorithm spec for a non-neural, linear-scaling
//! generative flow built on the existing Fock/SIRK substrate. The architecture
//! decouples **semantics** (hashing + coordinate projections) from
//! **reasoning** (unitary Krylov evolution), enabling exact, correlated, and
//! lossless spatial generation with zero O(M) dependencies during online
//! inference.
//!
//! # Module overview
//!
//! - [`sketch`]: Two-level hashing primitives — `CountSketch` (Level 1
//!   reduction from R^d to R^k) and `FeatureToMode` (Level 2 mapping from
//!   k-dim features to K_2-dim single-excitation Fock states).
//! - [`heavy_hitters`]: Count-Sketch Heavy Hitters algorithm for peak
//!   recovery from a probability sketch.
//! - [`potential`]: Offline analytical potential optimization (Flow Matching
//!   objective) and time-averaged coefficients.
//! - [`observables`]: Pre-projected observables — the m^2 operator basis
//!   {E_{r,s}}, the probability weight matrix W_prob, the Krylov image basis
//!   Phi, and the compressive subspace solver Phi_tilde^+.
//! - [`pipeline`]: The 4-phase online inference pipeline (encode -> evolve
//!   -> tomographic reconstruct -> lossless decode).
//! - [`bayes`]: Quantum Bayesian updating on the TSR-evolved prior
//!   (QMF.tex §8). Likelihood operators, HMC on the unit sphere of C^m,
//!   and tomographic reconstruction of the posterior sample.
//!
//! # Quick start
//!
//! ```
//! use qfm::{QfmConfig, QfmPipeline};
//!
//! // 4 training points in d=4 (the corners of a tetrahedron in 4-dim space).
//! let training = vec![
//!     vec![1.0, 0.0, 0.0, 0.0],
//!     vec![0.0, 1.0, 0.0, 0.0],
//!     vec![0.0, 0.0, 1.0, 0.0],
//!     vec![0.0, 0.0, 0.0, 1.0],
//! ];
//! let config = QfmConfig {
//!     k: 2,
//!     k2: 4,
//!     krylov_dim: 4,
//!     seed: 42,
//!     n_t_samples: 4,
//!     noise_dim: 2,
//!     max_rank: None,
//! };
//!
//! // Offline compile: produces all pre-projected observables.
//! let pipeline = QfmPipeline::compile(&training, &config).unwrap();
//! assert_eq!(pipeline.raw_dim(), 4);
//! assert_eq!(pipeline.k2_dim(), 4);
//!
//! // Online generate: query with a training point, get a generated image.
//! let x_out = pipeline.generate(&training[0]).unwrap();
//! assert_eq!(x_out.len(), 4);
//! for &v in &x_out {
//!     assert!(v.is_finite(), "output should be finite, got {v}");
//! }
//! ```

pub mod bayes;
pub mod heavy_hitters;
pub mod observables;
pub mod pipeline;
pub mod potential;
pub mod sketch;

pub use bayes::{
    BeliefPropagationResult, HmcOpts, Likelihood, Posterior, belief_propagation_chain,
    karcher_mean, reconstruct, sample_hmc, sample_hmc_single, tsr_evolved_prior,
};
pub use heavy_hitters::HeavyHitters;
pub use pipeline::{QfmConfig, QfmError, QfmPipeline};
pub use sketch::{CountSketch, FeatureToMode};
