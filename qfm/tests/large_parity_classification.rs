//! 8-bit parity with interpolation/extrapolation split.
//! Hilbert space: ℋ_in ⊗ ℋ_out (tensor product, no separate vacuum index).
//! Uniform vacuum: |i, vac⟩ = (1/√N_out)·Σ_f |i,f⟩.
//! Hamiltonian: H = λ₀·|c₀⟩⟨c₀| + λ₁·Σ (|i⟩⟨i|)_in ⊗ (|f⟩⟨vac| + |vac⟩⟨f|)_out

use nalgebra::DVector;
use num_complex::Complex64;
use qfm::pipeline::{QfmConfig, QfmPipeline};

fn parity(x: u32) -> bool {
    x.count_ones() % 2 == 0
}

const N_INPUTS: u32 = 256;
const LABEL_EVEN: u32 = N_INPUTS;
const LABEL_ODD: u32 = N_INPUTS + 1;

fn run_test(train_inputs: &[u32], transitions: &[(u32, u32)],
            interp_inputs: &[u32], extrap_inputs: &[u32],
            m: usize, lambda0: f64, label: &str,
            kernel_sigma: Option<f64>) {
    let lambda1 = 1.0;
    let t = 0.5;

    let config = QfmConfig {
        k: 1, k2: (N_INPUTS + 2) as usize, krylov_dim: m,
        seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None,
    };

    let input_modes: Vec<u32> = (0..N_INPUTS).collect();
    let output_modes = &[LABEL_EVEN, LABEL_ODD];
    let n_out = output_modes.len();
    let o_stride = n_out;

    let pipeline = QfmPipeline::compile_channels(
        &input_modes, output_modes, transitions, lambda0, lambda1,
        (N_INPUTS + 2) as usize, &config, 0.0, 0.0, true, None, kernel_sigma,
    ).expect("pipeline compile");

    let rank = pipeline.rank();
    let w = pipeline.w();
    let h_m = pipeline.h_m();

    let i = Complex64::new(0.0, 1.0);
    let u = (h_m.clone() * (-i * t)).exp();

    let eval = |x: u32| -> bool {
        let tp_even = (x as usize) * o_stride + 0;
        let tp_odd  = (x as usize) * o_stride + 1;
        // Encode |x, vac⟩ = (1/√N_out)·Σ_f |x, f⟩
        let mut c0 = DVector::zeros(rank);
        for f in 0..n_out {
            let tp = (x as usize) * o_stride + f;
            for k in 0..rank {
                c0[k] += w[(tp, k)];
            }
        }
        let inv_sqrt_no = 1.0 / (n_out as f64).sqrt();
        c0.scale_mut(inv_sqrt_no);
        let c1 = &u * c0;
        let amp_even: Complex64 = (0..rank).map(|k| c1[k] * w[(tp_even, k)].conj()).sum();
        let amp_odd: Complex64 = (0..rank).map(|k| c1[k] * w[(tp_odd, k)].conj()).sum();
        amp_even.norm_sqr() > amp_odd.norm_sqr()
    };

    let count = |xs: &[u32]| -> (u32, u32) {
        let mut cor = 0u32;
        for &x in xs {
            if eval(x) == parity(x) { cor += 1; }
        }
        (cor, xs.len() as u32)
    };

    let (tc, tn) = count(train_inputs);
    let (ic, it) = count(interp_inputs);
    let (ec, et) = count(extrap_inputs);

    let pct_t = tc as f64 / tn as f64 * 100.0;
    let pct_i = ic as f64 / it as f64 * 100.0;
    let pct_e = ec as f64 / et as f64 * 100.0;
    eprintln!("  {label:>24}  m={m} rank={:2} λ₀={lambda0:>7}  train {tc:>3}/{tn} ({pct_t:>5.1}%)  interp {ic:>2}/{it} ({pct_i:>5.1}%)  extrap {ec:>2}/{et} ({pct_e:>5.1}%)",
        pipeline.rank());
}

fn even_odd_split(n_even: usize, n_odd: usize, pool: &[u32]) -> Vec<u32> {
    let mut evens: Vec<u32> = pool.iter().filter(|&&x| parity(x)).copied().collect();
    let mut odds: Vec<u32> = pool.iter().filter(|&&x| !parity(x)).copied().collect();
    let mut selected = Vec::new();
    for _ in 0..n_even.min(evens.len()) { selected.push(evens.remove(0)); }
    for _ in 0..n_odd.min(odds.len()) { selected.push(odds.remove(0)); }
    selected.sort();
    selected
}

#[test]
fn large_parity_classification_tests() {
    let pool: Vec<u32> = (0..200).collect();
    let train_inputs = even_odd_split(48, 56, &pool);
    let interp_inputs: Vec<u32> = pool.iter()
        .filter(|x| !train_inputs.contains(x)).copied().collect();
    let extrap_inputs: Vec<u32> = (200..N_INPUTS).collect();

    eprintln!("\n--- 8-bit parity: {N_INPUTS} inputs, 2 output modes ---");
    eprintln!("     train: {} ({}e+{}o)", train_inputs.len(),
        train_inputs.iter().filter(|&&x| parity(x)).count(),
        train_inputs.iter().filter(|&&x| !parity(x)).count());
    eprintln!("     interp: {} (within-pool held-out)", interp_inputs.len());
    eprintln!("     extrap: {} (beyond training range)", extrap_inputs.len());

    let mut star_trans = Vec::with_capacity(train_inputs.len());
    for &x in &train_inputs {
        let label = if parity(x) { LABEL_EVEN } else { LABEL_ODD };
        star_trans.push((x, label));
    }

    for &lambda0 in &[0.0, 1.0, 2.0, 10.0] {
        eprintln!("\n  --- λ₀={lambda0} ---");
        for &m in &[3, 5, 7] {
            run_test(&train_inputs, &star_trans, &interp_inputs, &extrap_inputs,
                m, lambda0, "8-bit tensor-product", None);
        }
    }

    // Kernel tests with λ₀=1.0, m=4 (rank saturates at 3 regardless of m)
    eprintln!("\n  --- kernel tests (λ₀=1.0, m=4) ---");
    for &sigma in &[0.2, 0.3] {
        run_test(&train_inputs, &star_trans, &interp_inputs, &extrap_inputs,
            4, 1.0, &format!("kernel σ={sigma}"), Some(sigma));
    }
}
