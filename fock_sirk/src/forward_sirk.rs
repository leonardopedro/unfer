use nested_fock_algebra::{QuantumState, Hamiltonian};
use crate::registry::StateDictionary;
use crate::tensor_state::TensorState;
use candle_core::Device;
use num_complex::Complex64;
use nalgebra::DMatrix;

pub struct ForwardSirkResult {
    pub h_proj: DMatrix<Complex64>,
    pub g_matrix: DMatrix<Complex64>,
    pub registry: StateDictionary,
    pub basis_tensors: Vec<TensorState>,
}

impl ForwardSirkResult {
    /// Construct the evolution operator in the small basis: exp(-i * H_proj * t)

    pub fn time_evolve(&self, t: f64) -> nalgebra::DVector<Complex64> {
        let m = self.h_proj.nrows();
        let i = Complex64::new(0.0, 1.0);
        let evolution_matrix = (self.h_proj.clone() * (-i * t)).exp();
        let mut v_0_coeffs = nalgebra::DVector::zeros(m);
        v_0_coeffs[0] = Complex64::new(1.0, 0.0);
        evolution_matrix * v_0_coeffs
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


pub fn solve_forward_sirk(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device,
    brst_charge: Option<&Hamiltonian>, // Phase 10.1: Optional BRST Projection
) -> candle_core::Result<ForwardSirkResult> {
    let m = shifts.len();
    let mut w_sequence = Vec::with_capacity(m + 1);
    w_sequence.push(v_0.clone());

    // 1. Generate the forward sequence: w_k = (H - z_k I) w_{k-1}
    for k in 0..m {
        let prev_w = &w_sequence[k];
        let mut next_w = hamiltonian.apply(prev_w);
        // next_w = H * prev_w - shifts[k] * prev_w
        next_w.scale_and_add(prev_w, -shifts[k]);
        
        // Phase 10.1: Periodic BRST Projection to maintain gauge invariance
        if let Some(brst) = brst_charge {
            // Projection P = 1 - Omega^\dagger (Omega Omega^\dagger)^{-1} Omega
            // Simplified: we just ensure it commutes/stays in kernel
            // For this implementation, we apply the constraint Omega |w> = 0
            // by subtracting the non-physical part.
            let non_physical = brst.apply(&next_w);
            if non_physical.components.len() > 0 {
                // Subtracting Omega part (simplified projection)
                next_w.scale_and_add(&non_physical, Complex64::new(-1.0, 0.0));
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
        basis_tensors.push(TensorState::from_quantum_state(w, &mut registry, device)?);
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
        let ((j, k), val) = res?;
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

    // 5. Orthonormalize the system (Solve G c = H_raw c)
    // We use the Cholesky decomposition of G (top-left m x m) to find the transformation
    let g_sub = g_matrix.view((0, 0), (m, m));
    let chol = g_sub.cholesky().expect("Gram matrix must be positive definite");
    let l = chol.l();
    let l_inv = l.try_inverse().expect("L matrix must be invertible");
    let h_proj = &l_inv * h_proj_raw * l_inv.adjoint();

    Ok(ForwardSirkResult {
        h_proj,
        g_matrix,
        registry,
        basis_tensors,
    })
}
