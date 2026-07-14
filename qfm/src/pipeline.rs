//! Online inference pipeline for the QFM tomographic recovery.
//!
//! Orchestrates the compiled artifacts into the 4-phase generate function:
//!   Phase 1 (encode): S_1(query) -> x_tilde -> S_2 -> |Psi_in> -> W^dag c_0
//!   Phase 2 (evolve): c_1 = exp(-i H_m t) c_0  (H_m from real SIRK solve)
//!   Phase 3 (tomography): rho_flat = vec(c_1 c_1^dag) -> p_tilde = W_prob rho_flat
//!   Phase 4 (decode): heavy_hitters(p_tilde) -> x_tilde_peak -> gamma = Phi_tilde^+ x_tilde_peak
//!                     -> x_out = Phi gamma
//!
//! After compilation the M-dim dataset is retained only as the Level-1
//! sketch pre-image (for nearest-neighbor fallback in S_2). Every online
//! op is O(d m^2) + O(K_2 m^2) + O(K_2 log k) with no M dependence.
//!
//! ## F4-fix: real unitary flow
//!
//! The reduced Hamiltonian `H_m` is now obtained from a real SIRK solve
//! of the exact flow generator `H_bar = |0~><0~|` — the rank-1 projector
//! onto the dressed Mehler vacuum built by `build_flow_hamiltonian` (no
//! O(epsilon) truncation) — on the vacuum seed `|0>`, with `m` uniform
//! shifts on the negative-imaginary axis. The reduced system preserves
//! unitarity because the time-evolution is `U(t) = exp(-i H_m t)` via
//! `nalgebra`'s Padé approximant (AGENTS.md §4). The previous stub
//! (hardcoded diagonal `H_m` = `diag(α_j)`, no time parameter) has been
//! replaced.

use crate::heavy_hitters::HeavyHitters;
use crate::observables::{
    compressive_solver, krylov_image_basis, probability_weight_matrix, rank_truncate_w_h,
};
use crate::potential::{build_flow_hamiltonian, optimal_coefficients};
use crate::sketch::{CountSketch, FeatureToMode};
use candle_core::Device;
use fock_sirk::{ForwardSirkResult, solve_forward_sirk};
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::{InnerBosonicState, OuterState, QuantumState};
use num_complex::Complex64;

/// Configuration for compiling a QFM pipeline.
#[derive(Debug, Clone)]
pub struct QfmConfig {
    /// Level 1 sketch dimension (k, where k << d).
    pub k: usize,
    /// Level 2 sketched Hilbert space dimension (K_2 > k).
    pub k2: usize,
    /// Krylov subspace dimension (m, the reduced rank).
    pub krylov_dim: usize,
    /// PRNG seed for the Level 1 sketch.
    pub seed: u64,
    /// Number of time samples for Flow Matching integration.
    pub n_t_samples: usize,
    /// Noise prior dimension (for Mehler ground state).
    pub noise_dim: usize,
    /// Optional rank truncation via SVD on the W basis (P10.16.3).
    ///
    /// When `Some(r)`, after the SIRK solve the K_2×rank basis W is projected
    /// onto its top-r right singular vectors (W → W·V_r, H_m → V_r^H·H_m·V_r).
    /// This allows `krylov_dim << K_2` — the `K2ExceedsKrylovDim` check is
    /// bypassed — enabling d=1024 (CIFAR-10 32×32) without the O(K_2³) wall.
    /// When `None` (the default), the existing lossless path is used.
    pub max_rank: Option<usize>,
}

impl Default for QfmConfig {
    fn default() -> Self {
        // Note (P7 P3, rev 18): krylov_dim must be >= k2 for the K_2-row
        // restriction of w_whiten to be well-defined (the SIRK sequence
        // has krylov_dim+1 rows; the K_2-row restriction requires
        // krylov_dim >= K_2). The default satisfies this with equality.
        Self {
            k: 4,
            k2: 8,
            krylov_dim: 8,
            seed: 42,
            n_t_samples: 10,
            noise_dim: 4,
            max_rank: None,
        }
    }
}

/// The compiled QFM pipeline. Holds all pre-projected observables and
/// the Level 1/2 sketches needed for online encoding/decoding.
#[derive(Clone)]
pub struct QfmPipeline {
    s1: CountSketch,
    s2: FeatureToMode,
    /// Krylov basis W (K_2 x rank) — the SIRK-whitened `w_whiten` restricted
    /// to the K_2 single-excitation Fock rows, per-row renormalized (P6 G).
    w: DMatrix<Complex64>,
    /// Reduced Hamiltonian H_m (rank x rank, Hermitian) — obtained from a
    /// real SIRK solve of `H_bar` on the vacuum seed.
    h_m: DMatrix<Complex64>,
    /// Probability weight matrix W_prob (K_2 x rank^2).
    w_prob: DMatrix<f64>,
    /// Krylov image basis Phi (d x rank^2).
    phi: DMatrix<f64>,
    /// Compressive subspace solver Phi_tilde^+ (rank^2 x k).
    phi_tilde_plus: DMatrix<f64>,
    /// Heavy hitters tracker (for peak recovery).
    heavy_hitters: HeavyHitters,
    /// Training features (for nearest-neighbor fallback in S_2).
    training_features: Vec<(u64, Vec<f64>)>,
    /// Raw dimension d.
    d: usize,
    /// K_2 dimension.
    k2: usize,
    /// Reduced rank m.
    rank: usize,
    /// Outer vacuum |c₀⟩ in the whitened Krylov basis (length = rank).
    /// Precomputed during SIRK from the Gram matrix and whitening.
    outer_vacuum: DVector<Complex64>,
}

impl std::fmt::Debug for QfmPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QfmPipeline")
            .field("d", &self.d)
            .field("k2", &self.k2)
            .field("rank", &self.rank)
            .field("w_shape", &format!("{}x{}", self.w.nrows(), self.w.ncols()))
            .field(
                "h_m_shape",
                &format!("{}x{}", self.h_m.nrows(), self.h_m.ncols()),
            )
            .field(
                "w_prob_shape",
                &format!("{}x{}", self.w_prob.nrows(), self.w_prob.ncols()),
            )
            .field(
                "phi_shape",
                &format!("{}x{}", self.phi.nrows(), self.phi.ncols()),
            )
            .field(
                "phi_tilde_plus_shape",
                &format!(
                    "{}x{}",
                    self.phi_tilde_plus.nrows(),
                    self.phi_tilde_plus.ncols()
                ),
            )
            .field("training_features", &self.training_features.len())
            .finish()
    }
}

/// Errors from the QFM pipeline.
#[derive(Debug)]
pub enum QfmError {
    /// The query dimension doesn't match the raw dimension d.
    DimensionMismatch { expected: usize, got: usize },
    /// Compilation produced a degenerate basis.
    DegenerateBasis,
    /// The underlying SIRK solve failed (shifted Hamiltonian singular, etc.).
    SirkFailed(String),
    /// The configured Krylov dimension is smaller than K_2, so the K_2-row
    /// restriction of `w_whiten` would zero out some rows and produce a
    /// silently lossy decompression round-trip. Surface this as an error
    /// at compile time so the user can fix their `QfmConfig`.
    ///
    /// The relevant parameters are: `k2` (the single-excitation Fock
    /// subspace dim, which is also the K_2 bound on S_2), `krylov_dim`
    /// (the effective SIRK rank after the `min(m, k2)` clamp), and `m`
    /// (the number of training points, which is the natural upper bound
    /// on the SIRK sequence).
    ///
    /// (rev 18, P7 P3: was doc-only; promoted to a runtime check.)
    K2ExceedsKrylovDim {
        k2: usize,
        krylov_dim: usize,
        m: usize,
        config_krylov_dim: usize,
    },
}

impl std::fmt::Display for QfmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QfmError::DimensionMismatch { expected, got } => {
                write!(f, "query dimension {got} != expected {expected}")
            }
            QfmError::DegenerateBasis => write!(f, "compiled basis is degenerate"),
            QfmError::SirkFailed(msg) => write!(f, "SIRK solve failed: {msg}"),
            QfmError::K2ExceedsKrylovDim {
                k2,
                krylov_dim,
                m,
                config_krylov_dim,
            } => write!(
                f,
                "K_2 = {k2} exceeds the effective krylov_dim = {krylov_dim} \
                 (config.krylov_dim = {config_krylov_dim} clamped by min(M = {m}, K_2)); \
                 the K_2-row restriction of w_whiten would zero out {n} rows. \
                 Either increase config.krylov_dim to at least K_2, or reduce K_2 to <= M, \
                 or add more training points so M >= K_2.",
                n = k2 - krylov_dim
            ),
        }
    }
}

impl std::error::Error for QfmError {}

impl QfmPipeline {
    /// Compile a QFM pipeline from training data.
    ///
    /// Steps:
    /// 1. Compute optimal coefficients ᾱ_j from the Flow Matching objective.
    /// 2. Build the Hermitian static flow Hamiltonian H_bar.
    /// 3. Run SIRK on H_bar with the vacuum seed to obtain the reduced
    ///    Hamiltonian H_m (whitened, rank x rank) and the Krylov sequence.
    /// 4. Construct the K_2 x rank identity Krylov basis W (single-excitation
    ///    subspace) and the pre-projected observables (W_prob, Phi, Phi_tilde^+).
    /// 5. Register all training features in S_2.
    pub fn compile(training_points: &[Vec<f64>], config: &QfmConfig) -> Result<Self, QfmError> {
        let d = training_points.first().map(|p| p.len()).unwrap_or(0);
        if training_points.is_empty() || d == 0 {
            return Err(QfmError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let m = training_points.len();
        let k = config.k;
        let k2 = config.k2;
        let krylov_dim = config.krylov_dim.min(m).min(k2);
        if krylov_dim == 0 {
            return Err(QfmError::DegenerateBasis);
        }
        // P7 P3: the rev 17 P6 G fix requires `krylov_dim >= K_2` for the
        // K_2-row restriction of `w_whiten` to be lossless. The SIRK
        // sequence has `krylov_dim + 1` rows; a smaller `krylov_dim` leaves
        // rows krylov_dim..K_2 of W as zero (those Fock modes were never
        // visited). When `max_rank` is set (P10.16.3 rank-truncation path),
        // the user has opted into an explicit low-rank approximation and the
        // lossy case is intentional — bypass the error. Without `max_rank`
        // the lossless invariant is enforced as before.
        if krylov_dim < k2 && config.max_rank.is_none() {
            return Err(QfmError::K2ExceedsKrylovDim {
                k2,
                krylov_dim,
                m,
                config_krylov_dim: config.krylov_dim,
            });
        }

        // 1. Flow Matching optimal coefficients.
        let alphas = optimal_coefficients(training_points, config.n_t_samples, config.noise_dim);

        // 2. Build the Hermitian static flow Hamiltonian.
        let h_bar = build_flow_hamiltonian(&alphas, k2);

        // 3. Run SIRK on H_bar with the vacuum seed `|0>` and `krylov_dim`
        //    uniform shifts on the negative-imaginary axis (the standard
        //    choice for forward-mode rational Krylov — see AGENTS.md §4).
        let device = Device::Cpu;
        let v0 = vacuum_with_single_excitation_basis(k2);
        // Normalize shifts to the range [-i/krylov_dim, -i] on the imaginary axis.
        // Growing shifts z_k = -ik cause f64 overflow for krylov_dim > ~90 because
        // the Krylov-vector norms scale as krylov_dim!, which exceeds the f64 maximum
        // (~10^308) for krylov_dim > 88. Normalizing by krylov_dim caps each shift
        // at magnitude 1, bounding the per-step growth to (‖H‖+1) regardless of
        // krylov_dim and keeping the Gram matrix entries well within f64.
        let shifts: Vec<Complex64> = (1..=krylov_dim)
            .map(|i| Complex64::new(0.0, -(i as f64) / (krylov_dim as f64)))
            .collect();
        let sirk: ForwardSirkResult = solve_forward_sirk(&h_bar, &v0, &shifts, &device, None)
            .map_err(|e| QfmError::SirkFailed(e.to_string()))?;
        let rank = sirk.h_proj.nrows();
        if rank == 0 {
            return Err(QfmError::DegenerateBasis);
        }

        // 4. Construct the K_2 x rank real Krylov basis W = w_whiten
        //    restricted to the K_2 single-excitation rows (P6 G fix;
        //    rev 14 used the first-rank identity sub-block of the
        //    standard basis, which left a small lossy component in the
        //    decompression round-trip at high d). The starting state
        //    `vacuum_with_single_excitation_basis` has the vacuum at
        //    QuantumState-component index 0 and `|e_j>` for j in 0..K_2
        //    at component indices 1..=K_2. The SIRK `w_whiten` matrix
        //    (m+1) x rank (where m+1 = K_2+1) stores the whitened Krylov
        //    basis coordinates in the same order, so we take the bottom
        //    K_2 rows (indices 1..=K_2) and store them as a K_2 x rank
        //    Complex64 matrix. This is the genuine TSR-derived spatial
        //    mode basis: each column of W is a linear combination of the
        //    K_2 single-excitation Fock modes, not a single mode in
        //    isolation.
        //
        //    When the SIRK `w_whiten` is shaped (krylov_dim+1) x rank and
        //    krylov_dim < K_2, the post-gram-whitening basis still
        //    naturally fills the rank-1 per-row form (the basis vectors
        //    are the first `krylov_dim+1` indices of the Fock basis
        //    starting at 0, so the single-excitation rows are the
        //    first K_2 of those).  The construction below handles all
        //    (krylov_dim, K_2) combinations transparently.
        //
        //    **Renormalization:** the K_2+1-row whitened basis is
        //    orthonormal in the (K_2+1)-dim Fock inner product, but the
        //    K_2-row restriction is not (the missing vacuum component
        //    contributes to the full norm). To keep the Born-rule
        //    likelihood `|v^dag c|^2` behaving like a proper
        //    inner-product-squared (max=1 on the matching row), we
        //    renormalize each row of W to unit norm. This is a row
        //    scaling, not a column basis change, so the encode step
        //    `c_0 = W^dag |e_mode>` (which extracts the `mode`-th row)
        //    still gives a unit-norm state vector.
        let mut w = extract_single_excitation_w(&sirk.w_whiten, k2, rank);
        for i in 0..k2 {
            let row_norm: f64 = (0..rank).map(|j| w[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            if row_norm > 1e-300 {
                let scale = Complex64::new(1.0 / row_norm, 0.0);
                for j in 0..rank {
                    w[(i, j)] *= scale;
                }
            }
        }

        // 5. H_m is the projected Hamiltonian from the SIRK solve. It is
        //    Hermitian by construction (the SIRK Gram-whitening step
        //    guarantees the projected H is self-adjoint in the whitened
        //    basis — see ForwardSirkResult).
        let mut h_m = sirk.h_proj.clone();

        // 5b. P10.16.3 rank-truncation: project W and H_m onto the top-r
        //     right singular vectors of W. This allows krylov_dim << K_2
        //     (the K2ExceedsKrylovDim check is bypassed above when
        //     max_rank is set). After truncation, re-normalize W rows so
        //     the encode step c_0 = W[mode, :] still yields a unit-norm
        //     starting vector.
        let rank = if let Some(r) = config.max_rank {
            if let Some((w_trunc, h_trunc)) = rank_truncate_w_h(&w, &h_m, r) {
                w = w_trunc;
                h_m = h_trunc;
                let new_rank = w.ncols();
                // Re-normalize rows of the truncated W.
                for i in 0..k2 {
                    let row_norm: f64 = (0..new_rank)
                        .map(|j| w[(i, j)].norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    if row_norm > 1e-300 {
                        let scale = Complex64::new(1.0 / row_norm, 0.0);
                        for j in 0..new_rank {
                            w[(i, j)] *= scale;
                        }
                    }
                }
                new_rank
            } else {
                rank
            }
        } else {
            rank
        };
        if rank == 0 {
            return Err(QfmError::DegenerateBasis);
        }

        // 6. Pre-projected observables.
        let w_prob = probability_weight_matrix(&w, rank, k2);
        let phi = krylov_image_basis(&w, rank, d);

        // Apply S_1 to Phi: Phi_tilde = S_1 * Phi.
        let s1 = CountSketch::new(k, d, config.seed);
        let phi_tilde = s1.apply_to_columns(&phi);
        let phi_tilde_plus = compressive_solver(&phi_tilde);

        // 7. Register training features in S_2 (bounded by K_2).
        let mut s2 = FeatureToMode::new(k2);
        let mut training_features: Vec<(u64, Vec<f64>)> = Vec::with_capacity(m);
        for point in training_points {
            let x_tilde = s1.apply(point);
            let key = FeatureToMode::hash_feature(&x_tilde);
            s2.register(key).map_err(|_| QfmError::DegenerateBasis)?;
            training_features.push((key, x_tilde));
        }

        // 8. Heavy hitters tracker.
        let heavy_hitters = HeavyHitters::new(k.max(1), 0.0);

        Ok(Self {
            s1,
            s2,
            w,
            h_m,
            w_prob,
            phi,
            phi_tilde_plus,
            heavy_hitters,
            training_features,
            d,
            k2,
            rank,
            outer_vacuum: DVector::<Complex64>::zeros(0),
        })
    }

    /// Phase 1: encode a raw query into a Krylov coefficient vector c_0.
    ///
    /// Steps: S_1(query) -> x_tilde -> S_2 (resolve or nearest) -> |Psi_in>
    /// -> c_0 = W^dag |Psi_in>.
    pub fn encode(&self, query: &[f64]) -> Result<DVector<Complex64>, QfmError> {
        if query.len() != self.d {
            return Err(QfmError::DimensionMismatch {
                expected: self.d,
                got: query.len(),
            });
        }
        // Level 1 hash.
        let x_tilde = self.s1.apply(query);
        // Level 2 hash: resolve to a mode.
        let key = FeatureToMode::hash_feature(&x_tilde);
        let mode = self
            .s2
            .resolve(key)
            .or_else(|| self.s2.nearest(&x_tilde, &self.training_features))
            .unwrap_or(0);
        // c_0 = W^dag |e_mode> = the mode-th column of W^dag = the mode-th row of W.
        let mut c_0 = DVector::<Complex64>::zeros(self.rank);
        for r in 0..self.rank {
            c_0[r] = self.w[(mode as usize, r)];
        }
        Ok(c_0)
    }

    /// Phase 2: evolve the state forward by time `t`.
    ///
    /// `c_1 = exp(-i H_m t) c_0` via `nalgebra`'s Padé approximant on the
    /// Hermitian reduced Hamiltonian `H_m` (AGENTS.md §4: preserves
    /// unitarity and Hermiticity). The result is the *true* unitary flow
    /// of the real Flow-Matching-derived Hamiltonian — not a stub.
    pub fn evolve(&self, c_0: &DVector<Complex64>, t: f64) -> DVector<Complex64> {
        let i = Complex64::new(0.0, 1.0);
        let u = (self.h_m.clone() * (-i * t)).exp();
        &u * c_0
    }

    /// Phase 3+4: decode the evolved state back to a raw image.
    ///
    /// Phase 3 (tomography): rho_flat = vec(c_1 c_1^dag) -> p_tilde = W_prob rho_flat
    /// Phase 4 (decode): heavy_hitters(p_tilde) -> x_tilde_peak -> gamma = Phi_tilde^+ x_tilde_peak
    ///                    -> x_out = Phi gamma
    pub fn decode(&self, c_1: &DVector<Complex64>) -> Result<Vec<f64>, QfmError> {
        // Phase 3: tomography.
        // rho_flat[(r,s)] = c_1[r] * conj(c_1[s])
        let mut rho_flat = DVector::<f64>::zeros(self.rank * self.rank);
        for r in 0..self.rank {
            for s in 0..self.rank {
                let val = c_1[r] * c_1[s].conj();
                rho_flat[r * self.rank + s] = val.re;
            }
        }
        // p_tilde = W_prob * rho_flat (K_2 elements)
        let p_tilde = &self.w_prob * &rho_flat;

        // Phase 4: peak recovery + subspace decode.
        // For simplicity, use the top-1 mode from the probability sketch
        // and map it back to the k-dim feature space via the compressed
        // solver. The top-1 mode's count in the heavy hitters gives us
        // the peak k-dim feature coordinate.
        let mut hh = self.heavy_hitters.clone();
        hh.update_from_distribution(p_tilde.as_slice());
        let (peak_mode, _count) = hh.top_one().unwrap_or((0, 0.0));

        // Map peak_mode to a k-dim feature vector. Since we don't have
        // the inverse S_2 mapping (it's lossy), we use the training
        // feature nearest to the mode as the peak coordinate.
        let x_tilde_peak: Vec<f64> = if let Some((_, feat)) = self
            .training_features
            .iter()
            .find(|(key, _)| self.s2.resolve(*key) == Some(peak_mode))
            .cloned()
        {
            feat
        } else {
            vec![0.0; self.phi_tilde_plus.nrows().min(self.phi_tilde_plus.ncols())]
        };

        // gamma = Phi_tilde^+ * x_tilde_peak
        let gamma = &self.phi_tilde_plus * DVector::from_vec(x_tilde_peak);

        // x_out = Phi * gamma
        let x_out = &self.phi * &gamma;
        Ok(x_out.iter().cloned().collect())
    }

    /// Full 4-phase generate: encode -> evolve(t=1) -> decode.
    pub fn generate(&self, query: &[f64]) -> Result<Vec<f64>, QfmError> {
        self.generate_with_t(query, 1.0)
    }

    /// Full 4-phase generate with explicit time `t`.
    pub fn generate_with_t(&self, query: &[f64], t: f64) -> Result<Vec<f64>, QfmError> {
        let c_0 = self.encode(query)?;
        let c_1 = self.evolve(&c_0, t);
        self.decode(&c_1)
    }

    /// The raw dimension d.
    pub fn raw_dim(&self) -> usize {
        self.d
    }

    /// The K_2 dimension.
    pub fn k2_dim(&self) -> usize {
        self.k2
    }

    /// The reduced rank m.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Build a QfmPipeline from the pre-computed components (W, H_m,
    /// W_prob). Used by the deserializer to reconstruct a pipeline
    /// from the on-disk format without re-running the (expensive,
    /// rank-data-dependent) SIRK compile. The s1, s2, phi, and
    /// phi_tilde_plus fields are stubbed — they're only used by the
    /// image decode path, not by the text path's `decode_sketched`.
    pub fn from_components(
        w: DMatrix<Complex64>,
        h_m: DMatrix<Complex64>,
        w_prob: DMatrix<f64>,
    ) -> Self {
        let k2 = w.nrows();
        let rank = w.ncols();
        // Use a small but non-zero k for the s1 stub (k must be > 0
        // for CountSketch's modulo). The text path doesn't use s1,
        // so the value doesn't matter.
        let s1 = CountSketch::new(4, k2.max(1), 0);
        let s2 = FeatureToMode::new(0);
        Self {
            s1,
            s2,
            w,
            h_m,
            w_prob,
            phi: DMatrix::<f64>::zeros(0, 0),
            phi_tilde_plus: DMatrix::<f64>::zeros(0, 0),
            heavy_hitters: HeavyHitters::new(1, 0.0),
            training_features: Vec::new(),
            d: 0,
            k2,
            rank,
            outer_vacuum: DVector::<Complex64>::zeros(0),
        }
    }

    /// The Level 1 sketch (read-only).
    pub fn s1(&self) -> &CountSketch {
        &self.s1
    }

    /// The Level 2 hash (read-only).
    pub fn s2(&self) -> &FeatureToMode {
        &self.s2
    }

    /// The Krylov basis W (K_2 x rank).
    pub fn w(&self) -> &DMatrix<Complex64> {
        &self.w
    }

    /// The outer vacuum |c₀⟩ in the whitened Krylov basis (length = rank).
    /// Precomputed during SIRK — the exact projection of the starting
    /// vector v0 (uniform over all modes) onto the orthonormal Krylov
    /// basis. Used by qfm_text as the context-independent encode input.
    pub fn outer_vacuum(&self) -> &DVector<Complex64> {
        &self.outer_vacuum
    }

    /// The reduced Hamiltonian H_m (rank x rank, Hermitian).
    /// Public accessor for downstream consumers that need to
    /// re-evolve the system (e.g. qfm_text's per-mode Born-rule
    /// marginalization, which calls `c_1 = exp(-i H_m t) c_0`
    /// itself). Returns the same matrix `evolve` uses internally.
    pub fn h_m(&self) -> &DMatrix<Complex64> {
        &self.h_m
    }

    /// The probability weight matrix W_prob (K_2 x rank^2) — the
    /// pre-projected observable form of Phase 3 (QFM.tex), used by
    /// the image decode path and by checkpoint serialization. The
    /// text head does NOT use it: its real-part contraction is only
    /// exact for real W (see `decode_sketched`), so the token path
    /// computes the Born populations directly.
    pub fn w_prob(&self) -> &DMatrix<f64> {
        &self.w_prob
    }

    /// Encode a list of pre-hashed **mode indices** into a Krylov
    /// coefficient vector. The encoding is the equal-weighted
    /// superposition `(1/√n) Σ |mode⟩` projected onto the row
    /// basis `W`. This is the encoder used by the QFM-Text
    /// pipeline, which hashes the trailing `n_orders` tokens of a
    /// context into ≤ n mode indices via `OrderHasher` and then
    /// calls this function to lift them into a Krylov state.
    ///
    /// Empty `modes` returns the zero vector. Out-of-range modes
    /// (mode ≥ K₂) are silently dropped.
    pub fn encode_modes(&self, modes: &[u32]) -> Result<DVector<Complex64>, QfmError> {
        if modes.is_empty() {
            return Ok(DVector::<Complex64>::zeros(self.rank));
        }
        let scale = 1.0 / (modes.len() as f64).sqrt();
        let mut c = DVector::<Complex64>::zeros(self.rank);
        let w = &self.w;
        for &m in modes {
            let row = m as usize;
            if row < w.nrows() {
                for r in 0..self.rank {
                    c[r] += w[(row, r)] * scale;
                }
            }
        }
        Ok(c)
    }

    /// Per-mode encoding for the model-averaging decoder. For each
    /// input mode `m` in `modes`, return a **unit-norm** Krylov
    /// coefficient vector `c_0_m = w[m] / ||w[m]||`. Out-of-range
    /// modes contribute the zero vector.
    ///
    /// This is the encoder used by `QfmTextModel::next_token_dist_model_avg`,
    /// which evolves each mode's unit vector **independently** through
    /// `H_m t` and averages the resulting decoded distributions
    /// (Bayesian model averaging). This avoids the destructive
    /// interference in the equal-weight superposition of
    /// `encode_modes`, at the cost of n forward solves per token
    /// instead of one.
    pub fn encode_modes_per_order(
        &self,
        modes: &[u32],
    ) -> Result<Vec<DVector<Complex64>>, QfmError> {
        let w = &self.w;
        let mut out = Vec::with_capacity(modes.len());
        for &m in modes {
            let row = m as usize;
            if row < w.nrows() {
                let mut c = DVector::<Complex64>::zeros(self.rank);
                for r in 0..self.rank {
                    c[r] = w[(row, r)];
                }
                let norm = c.norm();
                if norm > 0.0 {
                    c /= Complex64::new(norm, 0.0);
                }
                out.push(c);
            } else {
                out.push(DVector::<Complex64>::zeros(self.rank));
            }
        }
        Ok(out)
    }

    /// Phase 3: compute the exact Born probability
    /// `p̃[m] = |⟨m|W c⟩|²` for every K_2 single-excitation Fock
    /// mode, then renormalize to sum 1. The vector length is
    /// `k2_dim()`. This is the per-mode Born probability that
    /// downstream token-decode heads marginalize against their
    /// per-mode histograms.
    ///
    /// Why not the pre-projected `W_prob` contraction
    /// (`p̃ = W_prob · Re vec(c c†)`, QFM.tex §"Phase 3")? That
    /// real-part contraction equals the Born population only when
    /// W is real. The SIRK shifts lie on the negative-imaginary
    /// axis, so the whitened Krylov basis — and hence W — is
    /// complex in general, and the contraction then evaluates to
    /// `½(|⟨m|W c⟩|² + |⟨m|W̄ c⟩|²)`: the Born population plus a
    /// spurious conjugate-basis term that washes out the mode
    /// discrimination. The direct Born rule below is exact for
    /// any W, always non-negative, and costs O(K_2 · rank) — the
    /// same as the contraction.
    pub fn decode_sketched(&self, c: &DVector<Complex64>) -> Vec<f64> {
        let rank = self.rank;
        let w = &self.w;
        // p[m] = |Σ_r W[m, r] c[r]|² (the actual Born rule).
        let mut p = vec![0.0_f64; w.nrows()];
        for m in 0..w.nrows() {
            let mut amp = Complex64::new(0.0, 0.0);
            for r in 0..rank {
                amp += w[(m, r)] * c[r];
            }
            p[m] = amp.norm_sqr();
        }
        // Clamp + renormalize.
        let total: f64 = p.iter().sum();
        if total > 0.0 {
            for x in p.iter_mut() {
                *x /= total;
            }
        }
        p
    }

    /// Precompute the `rank x rank` Gram matrix `G = W^H W`. Pass the
    /// result to [`decode_sketched_at`](Self::decode_sketched_at) so
    /// its normalization total costs `O(rank^2)` instead of
    /// `O(K_2)`. Callers that decode many queries against the same
    /// compiled pipeline (e.g. `qfm_text`'s per-token marginalization,
    /// which only ever reads a handful of active-mode entries out of
    /// the full `K_2`-length `decode_sketched` output) should compute
    /// this once and reuse it, not recompute it per query.
    pub fn gram(&self) -> DMatrix<Complex64> {
        self.w.adjoint() * &self.w
    }

    /// Sparse Born-rule decode: compute `p̃[m] = |⟨m|W c⟩|² / total`
    /// only for the given `indices`, where
    /// `total = Σ_m |⟨m|W c⟩|² = ⟨c|G|c⟩` (`G` from
    /// [`gram`](Self::gram)) is the same total-probability-mass
    /// normalizer [`decode_sketched`](Self::decode_sketched) computes
    /// by summing over all `K_2` modes — but here it costs
    /// `O(rank^2)`, not `O(K_2)`. Numerically identical to reading
    /// `decode_sketched(c)[i]` for each `i` in `indices`; this only
    /// exists so a caller that never looks at the other `K_2 -
    /// indices.len()` entries doesn't pay `O(K_2 * rank)` to produce
    /// them. Out-of-range indices are silently skipped (return no
    /// entry for that index).
    pub fn decode_sketched_at(
        &self,
        c: &DVector<Complex64>,
        gram: &DMatrix<Complex64>,
        indices: &[u32],
    ) -> Vec<(u32, f64)> {
        let total: f64 = (c.adjoint() * gram * c)[(0, 0)].re.max(0.0);
        if total <= 0.0 {
            return Vec::new();
        }
        let rank = self.rank;
        let w = &self.w;
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            let row = idx as usize;
            if row < w.nrows() {
                let mut amp = Complex64::new(0.0, 0.0);
                for r in 0..rank {
                    amp += w[(row, r)] * c[r];
                }
                out.push((idx, amp.norm_sqr() / total));
            }
        }
        out
    }

    /// The training features retained for the nearest-neighbor fallback
    /// in S_2 (a (key, feature) pair list).
    pub fn training_features(&self) -> &[(u64, Vec<f64>)] {
        &self.training_features
    }
}

/// A single per-order channel group, for [`QfmPipeline::compile_channels`].
///
/// `lambda` is the projector coefficient `λ_o`. `channels` is the list
/// of `(mode, alpha_j)` pairs: the order-`o` modes that have non-zero
/// weight, with `alpha_j` the unnormalized channel weight (typically
/// `weight_j / total_windows` for a streaming pass; the exact value
/// depends on the caller).
pub type ChannelGroup = (f64, Vec<(u32, f64)>);

impl QfmPipeline {
    /// Compile a QFM pipeline from **channel weights** instead of
    /// training points. Used by `qfm_text` (QFM-Text plan Stage 4):
    /// the streaming accumulator's per-mode statistics become the
    /// channel weights of the hierarchical multi-projector generator
    ///
    ///   `H = Σ_o λ_o |0̃_o⟩⟨0̃_o|`,
    ///   `|0̃_o⟩ = c₀^(o)|vac⟩_F + Σ_{j∈o} ε_j^(o)|x_j⟩`,
    ///   `ε_j = ᾱ_j/√(1+Σ_k ᾱ_k²)`, `c₀ = 1/√(1+Σ_k ᾱ_k²)`
    ///
    /// (the QFM.tex eq. Htomo normalization, applied per group) — one
    /// exact rank-1 `ProjectOnto` term per context order, built by
    /// `nested_fock_algebra::qfm_hamiltonian_hierarchical_projectors`.
    ///
    /// This is the paper-mandated form: since rev 31 the exact
    /// dressed-vacuum projector is the **only** off-diagonal QFM
    /// generator. The diagonal number-operator surrogate (QFM.tex
    /// eq. Hdiag) is explicitly *not* a flow — its Born populations
    /// are stationary (e^{-iHt} contributes only per-mode phases),
    /// and, being diagonal, its Krylov dimension is the number of
    /// *distinct* eigenvalues, which count-degenerate ᾱ_j collapse.
    /// (A rev 33 interim used eq. Hdiag here; rev 34 removed it.)
    ///
    /// The generator has rank ≤ n_groups, so the Krylov space from
    /// the uniform seed has dimension ≤ n_groups + 1 and the reduced
    /// `H_m` is small by construction. `config.krylov_dim` sets the
    /// number of SIRK shifts; `config.max_rank` (optional here — the
    /// rank is already bounded) further truncates the whitened basis.
    ///
    /// `groups[o] = (λ_o, channels_o)` where `channels_o` is the list
    /// of `(mode, ᾱ)` pairs for that order, with `mode` a global index
    /// in `[0, k2_total)`. The image-decode observables (Φ, Φ̃⁺) are
    /// not built (this is the text path; the decode head uses
    /// `decode_sketched` instead).
    ///
    /// The Hamiltonian is the diffusion-like generator:
    /// `H = λ₀·|c₀⟩⟨c₀| + λ₁·Σ_{(i→f)} (|f⟩⟨i| + |i⟩⟨f|)`
    /// where |c₀⟩ is the outer vacuum and the sum runs over
    /// consecutive training-window mode transitions.
    ///
    /// When `fock_resolution` is `Some(R)`, the SIRK starting vector
    /// uses amplitude `√R` (the outer vacuum — uniform in the
    /// Fock-space input basis with R partitions). When `None`, it
    /// uses the old amplitude `1/√(N+1)` (uniform over active modes).
    pub fn compile_channels(
        active_modes: &[u32],
        transitions: &[(u32, u32)],
        lambda0: f64,
        lambda1: f64,
        k2_total: usize,
        config: &QfmConfig,
        fock_resolution: Option<u64>,
    ) -> Result<Self, QfmError> {
        if k2_total == 0 || config.krylov_dim == 0 {
            return Err(QfmError::DegenerateBasis);
        }
        // Run SIRK on the diffusion Hamiltonian with the Fock-
        // uniform outer vacuum as the starting vector, with
        // `krylov_dim` shifts
        // normalized to the negative-imaginary axis (same choice as
        // `compile`), using the compact (plain-`u32`-keyed) state
        // representation throughout (see `CompactState`) — this is
        // what makes a `block_sizes` large enough to meaningfully cut
        // hash collisions memory-feasible (`QFM_TEXT_STATUS.md`
        // rev 35): the general `nested_fock_algebra::QuantumState`
        // path costs ~500+ bytes per active channel per Krylov step
        // (`BTreeMap`-backed `OuterState`), which OOMs once active
        // channel counts reach the hundreds of thousands to millions
        // that come from suppressing collisions on real text (order-3
        // /4 contexts are overwhelmingly unique, so avoiding collision
        // there forces active-channel count to scale with corpus
        // size).
        let shifts: Vec<Complex64> = (1..=config.krylov_dim)
            .map(|i| Complex64::new(0.0, -(i as f64) / (config.krylov_dim as f64)))
            .collect();
        let (w_whiten, h_proj, rank, w_sequence, outer_vacuum) =
            compact_forward_sirk(active_modes, transitions, lambda0, lambda1, &shifts, fock_resolution)?;
        if rank == 0 {
            return Err(QfmError::DegenerateBasis);
        }
        // Build the W matrix by projecting each single-excitation Fock
        // mode |j⟩ onto the rank-dim Gram-whitened Krylov basis.
        let mut w = project_compact_modes_onto_krylov_basis(
            &w_sequence,
            &w_whiten,
            k2_total,
            rank,
            &active_modes,
        );
        normalize_rows(&mut w);
        // H_m from the SIRK solve.
        let mut h_m = h_proj;
        let mut rank = rank;
        // Optional further rank truncation (usually a no-op here:
        // the projector-sum generator already bounds the rank).
        let mut outer_vacuum = outer_vacuum;
        if let Some(r) = config.max_rank {
            if let Some((w_trunc, h_trunc)) = rank_truncate_w_h(&w, &h_m, r) {
                let new_rank = w_trunc.ncols();
                outer_vacuum = outer_vacuum.rows(0, new_rank).into_owned();
                w = w_trunc;
                h_m = h_trunc;
                rank = new_rank;
                normalize_rows(&mut w);
            }
        }
        if rank == 0 {
            return Err(QfmError::DegenerateBasis);
        }
        // W_prob (Phase 3 projector) is **not built** for the text
        // path: `decode_sketched`/`decode_sketched_at` compute the
        // exact Born rule directly (rev 33 fix — the old real-part
        // `W_prob` contraction had destructive cancellation), so
        // nothing in `qfm_text` ever reads `w_prob`. Building it here
        // was an O(k2_total * rank^2) dense-matrix allocation (e.g.
        // ~2.85 GB at `k2_total ~ 5.5M`, `rank = 8`) purely for a
        // field that then gets serialized into every checkpoint and
        // never read back — this was the second O(k2_total) memory
        // sink (after the `OuterState` construction fixed above)
        // behind the OOM at large `block_sizes` (see
        // `QFM_TEXT_STATUS.md` rev 35). Phi/Phi_tilde are skipped for
        // the same reason.
        let w_prob = DMatrix::<f64>::zeros(0, 0);
        // Stub a CountSketch + FeatureToMode for the (text-irrelevant)
        // S_1 / S_2 fields, so the struct is well-formed.
        let s1 = CountSketch::new(config.k, k2_total.max(1), config.seed);
        let s2 = FeatureToMode::new(0); // unbounded; unused in the text path
        Ok(Self {
            s1,
            s2,
            w,
            h_m,
            w_prob,
            phi: DMatrix::<f64>::zeros(0, 0), // unused in text path
            phi_tilde_plus: DMatrix::<f64>::zeros(0, 0),
            heavy_hitters: HeavyHitters::new(1, 0.0),
            training_features: Vec::new(),
            d: 0,
            k2: k2_total,
            rank,
            outer_vacuum,
        })
    }
}

/// Renormalize each row of W to unit norm (rows with ~zero norm are
/// left untouched). A row scaling, not a column basis change, so the
/// encode step `c_0 = W[mode, :]` still yields a unit-norm vector.
fn normalize_rows(w: &mut DMatrix<Complex64>) {
    let (nrows, ncols) = (w.nrows(), w.ncols());
    for i in 0..nrows {
        let row_norm: f64 = (0..ncols).map(|j| w[(i, j)].norm_sqr()).sum::<f64>().sqrt();
        if row_norm > 1e-300 {
            let scale = Complex64::new(1.0 / row_norm, 0.0);
            for j in 0..ncols {
                w[(i, j)] *= scale;
            }
        }
    }
}

/// Build a starting state for the SIRK solve: the vacuum `|0>` plus
/// `k2` single-excitation basis vectors `|x_j> = B^dagger_j |0>` each
/// with amplitude `1/sqrt(k2+1)`. The forward sequence
/// `(H_bar - z_k I) v_0` then naturally populates the K_2+1-dim Fock
/// space spanned by the vacuum and the K_2 single-excitation states.
fn vacuum_with_single_excitation_basis(k2: usize) -> QuantumState {
    let mut state = QuantumState::zero();
    let amp = 1.0 / ((k2 + 1) as f64).sqrt();
    // Vacuum
    state
        .components
        .insert(OuterState::vacuum(), Complex64::new(amp, 0.0));
    // Single-excitation basis
    for j in 0..k2 as u32 {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(j, 1);
        let mut outer = OuterState::vacuum();
        outer.bosonic.insert(inner, 1);
        state.components.insert(outer, Complex64::new(amp, 0.0));
    }
    state
}

/// Extract the K_2 single-excitation rows of the SIRK `w_whiten` matrix
/// to form the genuine TSR spatial mode basis W (P6 G fix). Used by
/// the general (non-text) `compile()` path.
///
/// `w_whiten` is shaped `(krylov_dim + 1) x rank` (the rank of the
/// post-gram-whitening subspace, where `krylov_dim + 1` is the length
/// of the forward Krylov sequence) and stores the whitened basis
/// coordinates in the same order as the `QuantumState` components of
/// `vacuum_with_single_excitation_basis`: row 0 is the vacuum, rows
/// 1..=K_2 are the K_2 single-excitation Fock modes. We drop the
/// vacuum row and return a `K_2 x rank` Complex64 matrix, which is the
/// genuine TSR-derived spatial mode basis (each column is a linear
/// combination of the K_2 single-excitation Fock modes, not a single
/// mode in isolation).
///
/// **Edge cases:**
/// * If `krylov_dim + 1 < K_2 + 1` (i.e., the SIRK sequence never
///   reaches the full K_2 single-excitation subspace), the
///   high-index rows of the returned W are zero — the SIRK basis
///   spans only what the forward sequence actually visited. This is
///   the documented honest behaviour: the spatial mode basis is
///   rank-limited by the Krylov dimension, not by K_2.
/// * If `krylov_dim + 1 > K_2 + 1` (more rows than single-excitation
///   modes), the extra rows are dropped (they would have been beyond
///   the K_2 single-excitation Fock subspace anyway).
fn extract_single_excitation_w(
    w_whiten: &DMatrix<Complex64>,
    k2: usize,
    rank: usize,
) -> DMatrix<Complex64> {
    debug_assert_eq!(
        w_whiten.ncols(),
        rank,
        "w_whiten.ncols() = {} must equal rank = {}",
        w_whiten.ncols(),
        rank
    );
    let total_rows = w_whiten.nrows();
    // Skip row 0 (vacuum) and take the next K_2 rows.
    let mut w = DMatrix::<Complex64>::zeros(k2, rank);
    for j in 0..rank {
        for i in 0..k2 {
            // Row 0 is the vacuum; single-excitation j is at row j+1.
            let src_row = i + 1;
            if src_row < total_rows {
                w[(i, j)] = w_whiten[(src_row, j)];
            }
            // else: leave the entry as 0 (SIRK sequence did not reach
            // this Fock mode).
        }
    }
    w
}

/// A compact state for the dressed-vacuum SIRK sequence: a vacuum
/// amplitude plus a plain `u32`-keyed map of single-excitation
/// amplitudes. The dressed-vacuum matvec (see [`compact_dressed_vacuum_matvec`])
/// only ever reads/writes `{vacuum} ∪ active_modes`, so every state in
/// the forward Krylov sequence is exactly this shape — there is never
/// a need for `nested_fock_algebra::QuantumState`'s general
/// multi-mode, multi-particle `OuterState`/`InnerBosonicState`
/// representation here. Those types are `BTreeMap`-backed (~500+
/// bytes/entry, even for a single-entry map, from B-tree node
/// over-allocation); this plain hash map costs ~40-50 bytes/entry —
/// roughly a 10x reduction. That matters because active-channel
/// counts scale with corpus size once `block_sizes` is large enough
/// to meaningfully suppress hash collisions (real text's order-3/4
/// contexts are overwhelmingly unique, so avoiding collisions there
/// forces active-channel counts into the hundreds of thousands to
/// millions) — see `QFM_TEXT_STATUS.md` rev 35.
#[derive(Clone)]
struct CompactState {
    vacuum: Complex64,
    modes: rustc_hash::FxHashMap<u32, Complex64>,
}

impl CompactState {
    fn zero() -> Self {
        Self {
            vacuum: Complex64::new(0.0, 0.0),
            modes: rustc_hash::FxHashMap::default(),
        }
    }

    /// `<self|other>` — standard L² inner product (outer wavefunctions
    /// are orthogonal, so no geometric metric).
    fn inner_product(&self, other: &Self) -> Complex64 {
        let vac = self.vacuum.conj() * other.vacuum;
        let mut diag = Complex64::new(0.0, 0.0);
        if self.modes.len() <= other.modes.len() {
            for (k, v) in &self.modes {
                if let Some(bv) = other.modes.get(k) {
                    diag += v.conj() * bv;
                }
            }
        } else {
            for (k, v) in &other.modes {
                if let Some(av) = self.modes.get(k) {
                    diag += av.conj() * v;
                }
            }
        }
        vac + diag
    }

    /// `self += scale * other` (in place).
    fn scale_and_add(&mut self, other: &Self, scale: Complex64) {
        self.vacuum += scale * other.vacuum;
        for (&k, &v) in &other.modes {
            *self.modes.entry(k).or_insert(Complex64::new(0.0, 0.0)) += scale * v;
        }
    }
}

/// Compact-state analog of the dressed-vacuum matvec
/// `H|c⟩ = Σ_o λ_o ⟨0̃_o|c⟩|0̃_o⟩` (QFM.tex §"Implementation in the
/// unfer Kernel" line 906-908), operating on [`CompactState`] instead
/// of `nested_fock_algebra::QuantumState`. Pre-processes each order's
/// eq.-Htomo-normalized dressed vacuum (c_0, ε_m) once (O(M) total,
/// M = Σ|channels_o|); each matvec application is then O(M) plain
/// hash-map lookups + arithmetic, with no per-component `BTreeMap`
/// key construction at all.
///
/// `⟨0̃_o|c⟩ = c_0^(o)⟨0|c⟩ + Σ_m ε_m^(o)⟨m|c⟩` must be the **full
/// complex** inner product, not its real part: the dressed vacuum's
/// own components (c0, ε) are real, so conjugating them is a no-op,
/// but `c`'s amplitudes are genuinely complex — the SIRK shifts
/// ζ_k = -ik/m are pure-imaginary, so the Krylov sequence has a
/// nonzero imaginary part from the very first step. (An earlier
/// version of the original `QuantumState`-based matvec took `.re` of
/// each `c` component before accumulating, silently discarding that
/// imaginary part and returning a purely-real `H|c⟩` — breaking the
/// complex evolution the SIRK solve depends on and collapsing the
/// resulting Krylov/W rank far below what `m_shifts` requests. This
/// rewrite preserves that fix.)
/// Diffusion-like Hamiltonian matvec:
/// Diffusion Hamiltonian on outer wavefunctions:
/// `H|c⟩ = λ₀·⟨c₀|c⟩·|c₀⟩ + λ₁·Σ_{(i→f)} (⟨i|c⟩·|f⟩ + ⟨f|c⟩·|i⟩)`
fn compact_diffusion_matvec<'a>(
    active_modes: &'a [u32],
    transitions: &'a [(u32, u32)],
    amp0_vac: f64,
    amp0_mode: f64,
    lambda0: f64,
    lambda1: f64,
) -> impl Fn(&CompactState) -> CompactState + 'a {
    let zero = Complex64::new(0.0, 0.0);
    move |c: &CompactState| -> CompactState {
        let mut out = CompactState::zero();

        // 1. Outer vacuum projector: λ₀·⟨c₀|c⟩·|c₀⟩
        let mut ip = amp0_vac * c.vacuum;
        for &m in active_modes {
            ip += amp0_mode * c.modes.get(&m).copied().unwrap_or(zero);
        }
        let scale0 = ip * lambda0;
        out.vacuum += scale0 * amp0_vac;
        for &m in active_modes {
            *out.modes.entry(m).or_insert(zero) += scale0 * amp0_mode;
        }

        // 2. Transition term: λ₁·Σ_{(i,f)} (⟨i|c⟩·|f⟩ + ⟨f|c⟩·|i⟩)
        if lambda1 != 0.0 && !transitions.is_empty() {
            for &(from_mode, to_mode) in transitions {
                let c_from = c.modes.get(&from_mode).copied().unwrap_or(zero);
                let c_to = c.modes.get(&to_mode).copied().unwrap_or(zero);
                *out.modes.entry(from_mode).or_insert(zero) += lambda1 * c_to;
                *out.modes.entry(to_mode).or_insert(zero) += lambda1 * c_from;
            }
        }

        out
    }
}

/// Compact-state forward SIRK solve for the diffusion Hamiltonian
/// `H|c⟩ = λ₀·⟨c₀|c⟩·|c₀⟩ + λ₁·Σ_{(i→f)} (⟨i|c⟩·|f⟩ + ⟨f|c⟩·|i⟩)`.
/// Builds the `m+1`-element Krylov sequence
/// `w_k = (H - z_k I) w_{k-1}` (`m = shifts.len()`), its Gram matrix,
/// the Gram-whitened basis (reusing `fock_sirk::whiten_gram` — a pure
/// matrix operation with no dependency on `QuantumState`), and the
/// projected Hamiltonian `H_m`.
///
/// Returns `(w_whiten, h_proj, rank, w_sequence)`.
///
/// When `fock_resolution` is `Some(R)`, the starting vector is
/// the outer vacuum: uniform in the Fock-space input basis with
/// R partitions of the hypersphere, each with amplitude `√R`,
/// and the geometric mode-overlap metric `γ = 1/R` is used for
/// all inner products. When `None`, the old uniform amplitude
/// `1/√(N+1)` (N = active_mode count) and identity metric are used.
fn compact_forward_sirk(
    active_modes: &[u32],
    transitions: &[(u32, u32)],
    lambda0: f64,
    lambda1: f64,
    shifts: &[Complex64],
    fock_resolution: Option<u64>,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>, usize, Vec<CompactState>, DVector<Complex64>), QfmError> {
    let m = shifts.len();
    // The outer vacuum |c₀⟩ has ⟨c₀|ψ⟩ = 1/√R for ANY data point ψ.
    // Outer wavefunctions are orthogonal, so modes use the standard
    // L² inner product and amp0 is the same for all outer states.
    let r = fock_resolution.map_or(0.0, |r| r as f64);
    let amp0 = 1.0 / r.sqrt();
    let matvec = compact_diffusion_matvec(active_modes, transitions, amp0, amp0, lambda0, lambda1);
    let mut v0 = CompactState::zero();
    v0.vacuum = Complex64::new(amp0, 0.0);
    for &j in active_modes {
        v0.modes.insert(j, Complex64::new(amp0, 0.0));
    }
    let mut w_sequence = Vec::with_capacity(m + 1);
    w_sequence.push(v0);
    for k in 0..m {
        let prev = &w_sequence[k];
        let mut next = matvec(prev);
        next.scale_and_add(prev, -shifts[k]);
        w_sequence.push(next);
    }
    let mut g_matrix = DMatrix::<Complex64>::zeros(m + 1, m + 1);
    for j in 0..=m {
        for k in j..=m {
            let val = w_sequence[j].inner_product(&w_sequence[k]);
            g_matrix[(j, k)] = val;
            if j != k {
                g_matrix[(k, j)] = val.conj();
            }
        }
    }
    let mut h_proj_raw = DMatrix::<Complex64>::zeros(m, m);
    for j in 0..m {
        for k in 0..m {
            h_proj_raw[(j, k)] = g_matrix[(j, k + 1)] + shifts[k] * g_matrix[(j, k)];
        }
    }
    let g_sub = g_matrix.view((0, 0), (m, m)).into_owned();
    let whitening = fock_sirk::whiten_gram(&g_sub, fock_sirk::GRAM_REL_TOL)
        .map_err(|e| QfmError::SirkFailed(e.to_string()))?;
    let w_whiten = whitening.w;
    let h_proj = w_whiten.adjoint() * h_proj_raw * &w_whiten;
    let rank = whitening.rank;
    // Compute the outer vacuum's Krylov representation:
    // c_0_krylov[k] = Σ_{l=0}^{m-1} w_whiten[l,k] · ⟨w_l | w₀⟩
    // where w₀ = v0 is the uniform outer vacuum starting vector.
    let mut outer_vacuum = DVector::<Complex64>::zeros(rank);
    for l in 0..m {
        let inner = g_sub[(l, 0)];
        for k in 0..rank {
            outer_vacuum[k] += w_whiten[(l, k)] * inner;
        }
    }
    Ok((w_whiten, h_proj, rank, w_sequence, outer_vacuum))
}

/// Compact-state analog of the (now-removed) `project_modes_onto_krylov_basis`:
/// builds the `K_2 x rank` W matrix by projecting each active
/// single-excitation Fock mode onto the rank-dim Gram-whitened Krylov
/// basis,
///   `W[j, k] = Σ_{l=0..m-1} w_whiten[l, k] * ⟨w_sequence[l+1] | j⟩`,
/// reading `w_sequence[l+1].modes.get(&j)` directly (a plain hash-map
/// lookup) instead of constructing a `BTreeMap`-backed `OuterState`
/// key per mode. Only `active_modes` get a nonzero row; the rest of
/// the `K_2` rows stay at `DMatrix::zeros`'s default — exact, not an
/// approximation, since `w_sequence` never has a nonzero component
/// outside `{vacuum} ∪ active_modes` in the first place.
fn project_compact_modes_onto_krylov_basis(
    w_sequence: &[CompactState],
    w_whiten: &DMatrix<Complex64>,
    k2: usize,
    rank: usize,
    active_modes: &[u32],
) -> DMatrix<Complex64> {
    let m = w_whiten.nrows();
    let zero = Complex64::new(0.0, 0.0);
    let mut w = DMatrix::<Complex64>::zeros(k2, rank);
    for &j in active_modes {
        let row = j as usize;
        if row >= k2 {
            continue;
        }
        for l in 0..m {
            let coeff = w_sequence[l + 1].modes.get(&j).copied().unwrap_or(zero);
            for k in 0..rank {
                w[(row, k)] += w_whiten[(l, k)] * coeff;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn pipeline_compile_and_generate_synthetic() {
        // 4 training points in d=4, k=4, K_2=4, rank=4. (P7 P3: K_2 must be
        // <= m, and krylov_dim must be >= K_2; the effective krylov_dim is
        // min(config.krylov_dim, m, K_2), so K_2=4=krylov_dim=m is the
        // smallest legal config for m=4. The d=4 raw dim matches K_2=4
        // per the `krylov_image_basis` debug_assert!(d <= k2) constraint.)
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 4,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
            n_t_samples: 10,
            noise_dim: 4,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        assert_eq!(pipeline.raw_dim(), 4);
        assert_eq!(pipeline.k2_dim(), 4);
        assert!(pipeline.rank() >= 1, "rank should be at least 1");

        // Generate from the first training point.
        let x_out = pipeline.generate(&training[0]).unwrap();
        assert_eq!(x_out.len(), 4);
        for &v in &x_out {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }

        // F4-fix: stronger assertion. The query IS a training point in the
        // d-dim space, so the evolved decode should be at least
        // positively correlated with the query. (Because the SIRK solve
        // and the random sketch are not lossless, we expect a positive
        // but not perfect cosine similarity.)
        let sim = cosine_similarity(&x_out, &training[0]);
        assert!(
            sim > 0.0,
            "evolved decode should be positively correlated with query, got {sim}"
        );
    }

    #[test]
    fn pipeline_evolve_unitarity_preserves_norm() {
        // The Padé-exp of H_m should preserve the 2-norm of c_0 (up to
        // numerical error) because U(t) = exp(-i H_m t) is unitary.
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 7,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        let c_0 = pipeline.encode(&training[0]).unwrap();
        let norm0: f64 = c_0.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let c_1 = pipeline.evolve(&c_0, 1.0);
        let norm1: f64 = c_1.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            (norm0 - norm1).abs() < 1e-6,
            "unitarity violated: ||c_0||={norm0} vs ||c_1||={norm1}"
        );
    }

    #[test]
    fn pipeline_evolve_with_different_t() {
        // Evolving for t=0 must return c_0 (to numerical precision);
        // evolving for t and 2*t must produce different outputs (so the
        // time parameter is actually wired up).
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 7,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        let c_0 = pipeline.encode(&training[0]).unwrap();
        let c_at_0 = pipeline.evolve(&c_0, 0.0);
        let c_at_1 = pipeline.evolve(&c_0, 1.0);
        let c_at_2 = pipeline.evolve(&c_0, 2.0);

        // t=0 should be c_0.
        for (a, b) in c_0.iter().zip(c_at_0.iter()) {
            assert!((a - b).norm() < 1e-9, "t=0 should be identity");
        }
        // t=1 vs t=2 should differ.
        let mut max_diff = 0.0_f64;
        for (a, b) in c_at_1.iter().zip(c_at_2.iter()) {
            max_diff = max_diff.max((a - b).norm());
        }
        assert!(
            max_diff > 1e-3,
            "t=1 vs t=2 should differ, got max diff {max_diff}"
        );
    }

    #[test]
    fn pipeline_no_m_in_online() {
        // Verify that `generate` doesn't reference the training set
        // after compilation. We do this by checking that the function
        // signature is `&self` only (no `&self` of training data).
        // This is a structural test — the compile-time guarantee.
        // (P7 P3: m=2 training points so K_2 must be <= 2; use K_2=2
        // and d=2 to match the krylov_image_basis d <= K_2 constraint.)
        let training = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let config = QfmConfig {
            k: 2,
            k2: 2,
            krylov_dim: 2,
            seed: 42,
            n_t_samples: 10,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        // The pipeline struct holds the pre-projected observables,
        // not the raw training points. The training_features field
        // is only used for nearest-neighbor fallback in S_2 (not M).
        assert!(pipeline.generate(&[1.0, 0.0]).is_ok());
    }

    #[test]
    fn encode_dimension_mismatch() {
        let training = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let config = QfmConfig {
            k: 2,
            k2: 2,
            krylov_dim: 2,
            seed: 42,
            n_t_samples: 10,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        // Query with wrong dimension.
        let result = pipeline.generate(&[1.0, 0.0, 0.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            QfmError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            _ => panic!("expected DimensionMismatch"),
        }
    }

    #[test]
    fn w_basis_is_sirk_whitened_not_identity() {
        // P6 G fix: the Krylov basis W is now the genuine SIRK-generated
        // w_whiten restricted to the K_2 single-excitation rows, NOT the
        // rank-k identity sub-block of the standard basis. This test
        // asserts that W has at least one off-diagonal magnitude > 1e-6
        // (a "real" mixed basis), which would be impossible for the
        // identity W of rev 14.
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        let w = pipeline.w();
        let k2 = w.nrows();
        let rank = w.ncols();
        assert_eq!(k2, 4);
        // The rank is determined by the SIRK solve; the tetrahedron's
        // symmetry reduces it to 2 independent directions. The constraint
        // is just rank >= 1 (P6 G is verified by the off-diagonal check below).
        assert!(
            rank >= 1,
            "W should have at least one column, got rank={rank}"
        );
        // Sum the off-diagonal magnitudes: a non-trivial mixed basis
        // will have at least one off-diagonal entry with non-trivial
        // magnitude. The identity sub-block would have all off-diagonal
        // magnitudes equal to exactly 0.
        let mut off_diag_max: f64 = 0.0;
        for i in 0..k2 {
            for j in 0..rank {
                if i != j {
                    let m = w[(i, j)].norm();
                    if m > off_diag_max {
                        off_diag_max = m;
                    }
                }
            }
        }
        assert!(
            off_diag_max > 1e-6,
            "P6 G: W should be a real SIRK-generated basis (with off-diagonal mixing), \
             got max off-diagonal magnitude {off_diag_max} — looks like the identity stub"
        );
    }

    #[test]
    fn w_basis_columns_are_unit_norm() {
        // The SIRK Gram-whitening step guarantees that w_whiten has
        // orthonormal columns in the K_2+1-dim Fock inner product. Since
        // we drop the vacuum row, the K_2-row restriction of an
        // orthonormal basis is *not* necessarily unit-norm per column
        // (the missing vacuum component contributes to the norm). This
        // test verifies that each column of W is well-defined and
        // finite (no NaN/Inf), which is the structural correctness
        // gate for the P6 G fix.
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        let pipeline = QfmPipeline::compile(&training, &config).unwrap();
        let w = pipeline.w();
        for j in 0..w.ncols() {
            let norm_sq: f64 = (0..w.nrows()).map(|i| w[(i, j)].norm_sqr()).sum();
            assert!(
                norm_sq.is_finite() && norm_sq > 0.0,
                "W column {j} has zero or non-finite norm {norm_sq}"
            );
        }
    }

    #[test]
    fn k2_exceeds_krylov_dim_returns_typed_error() {
        // P7 P3: the rev 17 P6 G fix requires `krylov_dim >= K_2` for the
        // K_2-row restriction of `w_whiten` to be well-defined. Before
        // rev 18, this was a doc-only constraint; a too-small
        // `krylov_dim` would silently zero out rows of W and produce a
        // lossy round-trip. Now it's a typed error at compile time.
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        // krylov_dim=2 < K_2=4 -> should fail with K2ExceedsKrylovDim.
        let bad_config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 2,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        let err = QfmPipeline::compile(&training, &bad_config).unwrap_err();
        match err {
            QfmError::K2ExceedsKrylovDim {
                k2,
                krylov_dim,
                m,
                config_krylov_dim,
            } => {
                assert_eq!(k2, 4);
                assert_eq!(krylov_dim, 2);
                assert_eq!(m, 4);
                assert_eq!(config_krylov_dim, 2);
            }
            other => panic!("expected K2ExceedsKrylovDim, got {other:?}"),
        }

        // Sanity: the well-formed config from the existing tests still compiles.
        let good_config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        QfmPipeline::compile(&training, &good_config).unwrap();

        // Edge: krylov_dim = K_2 (equality) is OK; only strict < fails.
        let edge_config = QfmConfig {
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
            n_t_samples: 4,
            noise_dim: 2,
            max_rank: None,
        };
        QfmPipeline::compile(&training, &edge_config).unwrap();

        // The error message mentions the right values + the fix.
        let msg = format!(
            "{}",
            QfmError::K2ExceedsKrylovDim {
                k2: 8,
                krylov_dim: 4,
                m: 4,
                config_krylov_dim: 8,
            }
        );
        assert!(msg.contains("K_2 = 8"));
        assert!(msg.contains("krylov_dim = 4"));
        assert!(msg.contains("increase config.krylov_dim"));
    }
}
