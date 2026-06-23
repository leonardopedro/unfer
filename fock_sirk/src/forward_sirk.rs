use nested_fock_algebra::{QuantumState, Hamiltonian};
use crate::registry::StateDictionary;
use crate::tensor_state::TensorState;
use crate::linalg::SirkError;
use candle_core::Device;
use num_complex::Complex64;
use nalgebra::DMatrix;

/// Default convergence tolerance for the per-step BRST projection.
const BRST_TOL: f64 = 1e-10;

pub struct ForwardSirkResult {
    /// Projected Hamiltonian in the whitened orthonormal basis (`rank x rank`).
    pub h_proj: DMatrix<Complex64>,
    /// Full Gram matrix of the Krylov sequence (`(m+1) x (m+1)`).
    pub g_matrix: DMatrix<Complex64>,
    pub registry: StateDictionary,
    pub basis_tensors: Vec<TensorState>,
    /// Numerical rank of the Krylov basis after whitening (= `h_proj.nrows()`).
    pub rank: usize,
    /// Whitening transform `W` (`m x rank`) mapping whitened coefficients back to
    /// Krylov-vector coordinates. `Wᴴ G_sub W = I_rank`.
    pub w_whiten: DMatrix<Complex64>,
    /// The retained Krylov sequence `w_0, ..., w_m` used by [`reconstruct`] to map
    /// whitened-basis coefficients back to a [`QuantumState`].
    pub w_sequence: Vec<QuantumState>,
}

impl ForwardSirkResult {
    /// Whitened-basis coefficients of the initial state `v_0 = w_0`.
    ///
    /// `v_0` has Krylov-coordinate `e_0`, so its whitened coordinate is
    /// `b_0 = Wᴴ G e_0 = Wᴴ · (first m entries of column 0 of G)`.
    fn initial_coeffs(&self) -> nalgebra::DVector<Complex64> {
        let m = self.w_whiten.nrows();
        let g0 = self.g_matrix.column(0).rows(0, m).into_owned();
        self.w_whiten.adjoint() * g0
    }

    /// Construct the evolution operator in the small basis: exp(-i * H_proj * t)
    /// applied to the initial state, returning coefficients in the whitened basis.
    pub fn time_evolve(&self, t: f64) -> nalgebra::DVector<Complex64> {
        let i = Complex64::new(0.0, 1.0);
        let evolution_matrix = (self.h_proj.clone() * (-i * t)).exp();
        evolution_matrix * self.initial_coeffs()
    }

    /// Map whitened-basis coefficients back to a [`QuantumState`].
    ///
    /// This is the inverse of [`time_evolve`](Self::time_evolve): it multiplies
    /// the whitened coefficients by the whitening transform `W` to get Krylov-vector
    /// coordinates, then linearly combines the stored `w_sequence` via
    /// [`QuantumState::scale_and_add`].
    pub fn reconstruct(&self, coeffs: &nalgebra::DVector<Complex64>) -> QuantumState {
        let w_coords = &self.w_whiten * coeffs;
        let m = self.w_whiten.nrows();
        let mut state = QuantumState::zero();
        for j in 0..m {
            if j < self.w_sequence.len() {
                state.scale_and_add(&self.w_sequence[j], w_coords[j]);
            }
        }
        state
    }

    /// Phase 11.2: Export simulation coefficients to JSON for visualization.
    pub fn export_to_json(&self) -> String {
        use serde_json::json;
        let mut data = Vec::new();
        for i in 0..self.h_proj.nrows() {
            for j in 0..self.h_proj.ncols() {
                data.push(json!({
                    "row": i,
                    "col": j,
                    "re": self.h_proj[(i, j)].re,
                    "im": self.h_proj[(i, j)].im
                }));
            }
        }
        json!({
            "h_proj": data,
            "m_dim": self.h_proj.nrows()
        }).to_string()
    }
}


/// Tunable bounds and tolerances for the SIRK solve.
#[derive(Debug, Clone)]
pub struct SirkOpts {
    /// Drop Krylov-vector components with `|amplitude| <= prune_eps` each step.
    pub prune_eps: f64,
    /// Hard ceiling on the number of components in any Krylov vector; exceeding
    /// it aborts with [`SirkError::StateExplosion`] instead of running out of RAM.
    pub max_components: Option<usize>,
    /// Convergence tolerance for the per-step BRST projection.
    pub brst_tol: f64,
}

impl Default for SirkOpts {
    fn default() -> Self {
        Self { prune_eps: 1e-12, max_components: None, brst_tol: BRST_TOL }
    }
}

/// Backwards-compatible entry point: solve with [`SirkOpts::default`].
pub fn solve_forward_sirk(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>, // Phase 10.1: Optional BRST Projection
) -> Result<ForwardSirkResult, SirkError> {
    solve_forward_sirk_with_opts(
        hamiltonian,
        v_0,
        shifts,
        device,
        brst_charge,
        &SirkOpts::default(),
    )
}

pub fn solve_forward_sirk_with_opts(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>,
    opts: &SirkOpts,
) -> Result<ForwardSirkResult, SirkError> {
    let m = shifts.len();
    let mut w_sequence = Vec::with_capacity(m + 1);
    w_sequence.push(v_0.clone());

    // 1. Generate the forward sequence: w_k = (H - z_k I) w_{k-1}
    for k in 0..m {
        let prev_w = &w_sequence[k];
        let mut next_w = hamiltonian.apply(prev_w);
        // next_w = H * prev_w - shifts[k] * prev_w
        next_w.scale_and_add(prev_w, -shifts[k]);

        // Phase 10.1: Periodic BRST projection to maintain gauge invariance.
        // True orthogonal projection onto ker(Q) via matrix-free CG (replaces the
        // earlier non-idempotent subtraction hack).
        if let Some(brst) = brst_charge {
            next_w = crate::brst::project_physical(&next_w, brst, opts.brst_tol, 50)?;
        }

        // Memory hygiene + explosion guard.
        next_w.prune(opts.prune_eps);
        if let Some(limit) = opts.max_components {
            if next_w.len() > limit {
                return Err(SirkError::StateExplosion { components: next_w.len(), limit });
            }
        }

        w_sequence.push(next_w);
    }


    // 2. Flatten states into a registry for GPU processing
    let mut registry = StateDictionary::new();
    for w in &w_sequence {
        registry.register(w);
    }

    let mut basis_tensors = Vec::with_capacity(m + 1);
    for w in &w_sequence {
        basis_tensors.push(
            TensorState::from_quantum_state(w, &mut registry, device)
                .map_err(|e| SirkError::Numeric(e.to_string()))?,
        );
    }

    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU
    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU
    // Parallelized using Rayon for Phase 9.1 performance milestone.
    use rayon::prelude::*;
    
    let mut g_matrix = DMatrix::zeros(m + 1, m + 1);
    
    // We compute only the upper triangle due to Hermiticity: G_kj = G_jk*
    // We flatten the indices to use par_iter
    let indices: Vec<(usize, usize)> = (0..=m)
        .flat_map(|j| (j..=m).map(move |k| (j, k)))
        .collect();

    let results: Vec<candle_core::Result<((usize, usize), Complex64)>> = indices
        .into_par_iter()
        .map(|(j, k)| {
            let val = basis_tensors[j].inner_product(&basis_tensors[k])?;
            Ok(((j, k), val))
        })
        .collect();

    for res in results {
        let ((j, k), val) = res.map_err(|e| SirkError::Numeric(e.to_string()))?;
        g_matrix[(j, k)] = val;
        if j != k {
            g_matrix[(k, j)] = val.conj();
        }
    }


    // 3. Compute the Gram matrix G_jk = <w_j | w_k> on the GPU

    // 4. Construct the projected Hamiltonian H_jk = <w_j | H | w_k>
    // Using the identity: H w_k = w_{k+1} + z_k w_k
    // <w_j | H | w_k> = <w_j | w_{k+1}> + z_k <w_j | w_k> = G_{j, k+1} + z_k G_{j,k}
    let mut h_proj_raw = DMatrix::zeros(m, m);
    for j in 0..m {
        for k in 0..m {
            h_proj_raw[(j, k)] = g_matrix[(j, k + 1)] + shifts[k] * g_matrix[(j, k)];
        }
    }

    // 5. Orthonormalize the system (solve the generalized problem H_raw c = λ G c)
    // via rank-revealing Gram whitening. This replaces a bare Cholesky that
    // panicked whenever the Krylov vectors became linearly dependent.
    let g_sub = g_matrix.view((0, 0), (m, m)).into_owned();
    let whitening = crate::linalg::whiten_gram(&g_sub, crate::linalg::GRAM_REL_TOL)?;
    let w = whitening.w; // m x rank
    let h_proj = w.adjoint() * h_proj_raw * &w; // rank x rank

    Ok(ForwardSirkResult {
        h_proj,
        g_matrix,
        registry,
        basis_tensors,
        rank: whitening.rank,
        w_whiten: w,
        w_sequence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};

    fn shifts(m: usize) -> Vec<Complex64> {
        (0..m)
            .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
            .collect()
    }

    /// A zero Hamiltonian makes every Krylov vector parallel to v_0, so the Gram
    /// matrix is rank 1. The old bare Cholesky panicked here; whitening must not.
    #[test]
    fn rank_deficient_gram_no_panic() {
        let device = Device::Cpu;
        let h = Hamiltonian { terms: vec![] };
        let v0 = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None)
            .expect("whitening must not panic on a rank-deficient Gram");
        assert_eq!(res.rank, 1, "all Krylov vectors are parallel => rank 1");
        assert_eq!(res.h_proj.nrows(), 1);
    }

    /// Two-state hopping Hamiltonian H = |B><A| + |A><B| has eigenvalues ±1.
    /// After whitening, the projected Hamiltonian must be Hermitian with real
    /// Ritz values matching ±1 (basis-independent), confirming the orthonormal-
    /// ization is correct.
    #[test]
    fn ritz_values_real_for_hermitian() {
        let device = Device::Cpu;
        let a = InnerBosonicState::vacuum();
        let mut b = InnerBosonicState::vacuum();
        b.modes.insert(0, 1);

        let h = Hamiltonian {
            terms: vec![
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(b.clone()),
                        Operator::OuterBosonAnnihilate(a.clone()),
                    ],
                ),
                (
                    Complex64::new(1.0, 0.0),
                    vec![
                        Operator::OuterBosonCreate(a.clone()),
                        Operator::OuterBosonAnnihilate(b.clone()),
                    ],
                ),
            ],
        };
        let v0 = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(a));

        let res = solve_forward_sirk(&h, &v0, &shifts(4), &device, None).unwrap();
        assert!(res.rank >= 2, "hopping spans a 2D Krylov space, got {}", res.rank);

        // h_proj is Hermitian => real eigenvalues; the spectrum should bracket ±1.
        let eig = res.h_proj.clone().symmetric_eigen();
        let mut vals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = *vals.first().unwrap();
        let hi = *vals.last().unwrap();
        assert!((lo + 1.0).abs() < 1e-8, "lowest Ritz value {lo} != -1");
        assert!((hi - 1.0).abs() < 1e-8, "highest Ritz value {hi} != +1");
    }

    /// Stage 5: the Navier-Stokes Hamiltonian was previously an Expression that
    /// caused the CAS to hang in .expand(). Now built directly as Hamiltonian
    /// terms (like yang_mills). With Gram whitening (Stage 2) + explosion bounds
    /// (Stage 4), the full solve must complete on CPU without panicking.
    #[test]
    fn navier_stokes_solve_completes() {
        let device = Device::Cpu;
        let h = nested_fock_algebra::models::navier_stokes_hamiltonian(1e-3);

        assert!(!h.terms.is_empty(), "Navier-Stokes Hamiltonian must be non-empty");

        let v0 = QuantumState::vacuum()
            .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));

        let opts = SirkOpts {
            prune_eps: 1e-12,
            max_components: Some(50_000),
            brst_tol: 1e-10,
        };

        let res = solve_forward_sirk_with_opts(&h, &v0, &shifts(4), &device, None, &opts)
            .expect("Navier-Stokes SIRK solve must not panic or error");

        assert!(res.rank > 0, "Krylov basis must have positive rank");
        assert_eq!(res.h_proj.nrows(), res.rank);
    }
}
