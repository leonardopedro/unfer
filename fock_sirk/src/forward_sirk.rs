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
    /// Compute the time evolution exp(-i * H * t) * v_0
    /// Using nalgebra's matrix exponential for accuracy and unitarity preservation.
    pub fn time_evolve(&self, t: f64) -> nalgebra::DVector<Complex64> {
        let m = self.h_proj.nrows();
        let i = Complex64::new(0.0, 1.0);
        
        // Construct the evolution operator in the small basis: exp(-i * H_proj * t)
        let evolution_matrix = (self.h_proj.clone() * (-i * t)).exp();
        
        // Initial state in this basis is e_1 (since w_0 is the first basis vector)
        // However, the basis is not necessarily orthonormal yet.
        // We need to account for the Gram matrix G.
        // The generalized eigenvalue problem is H c = E G c.
        // Or we can orthonormalize the basis first.
        
        // For simplicity in this implementation, we assume the user wants the 
        // coefficients in the original (non-orthonormal) w_k basis.
        let mut v_0_coeffs = nalgebra::DVector::zeros(m);
        v_0_coeffs[0] = Complex64::new(1.0, 0.0);
        
        evolution_matrix * v_0_coeffs
    }
}

pub fn solve_forward_sirk(
    hamiltonian: &Hamiltonian,
    v_0: &QuantumState,
    shifts: &[Complex64],
    device: &Device
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
    let mut g_matrix = DMatrix::zeros(m + 1, m + 1);
    for j in 0..=m {
        for k in 0..=m {
            if j <= k {
                let val = basis_tensors[j].inner_product(&basis_tensors[k])?;
                g_matrix[(j, k)] = val;
                g_matrix[(k, j)] = val.conj();
            }
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
