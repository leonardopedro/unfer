//! Binary parity classification using the tensor-product Hamiltonian.
//! 4-bit inputs (0..15), 2 output labels (even/odd).
//! Hilbert space: ℋ_in ⊗ ℋ_out with basis |i,f⟩ (no separate vacuum index).
//! Uniform vacuum: |i, vac⟩ = (1/√N_out)·Σ_f |i,f⟩.
//! Hamiltonian:
//!   H = λ₀·|c₀⟩⟨c₀| + λ₁·Σ_{i→f} (|i⟩⟨i|)_in ⊗ (|f⟩⟨vac| + |vac⟩⟨f|)_out
//!
//! Decoding: for test input x, P(label) = |⟨x, label|ψ⟩|² / (sum over labels)

use nalgebra::DVector;
use num_complex::Complex64;
use qfm::pipeline::{QfmConfig, QfmPipeline};

fn parity(x: u32) -> bool {
    x.count_ones() % 2 == 0
}

fn run_test(train_inputs: &[u32], transitions: &[(u32, u32)],
            input_modes: &[u32], output_modes: &[u32],
            n_modes: usize, label: &str, do_whiten: bool,
            kernel_sigma: Option<f64>) -> (u32, u32, u32, u32) {
    let test_inputs: Vec<u32> = (0..16u32).filter(|&x| !train_inputs.contains(&x)).collect();

    let lambda0 = 1.0;
    let lambda1 = 1.0;
    let t = 0.5;

    let config = QfmConfig {
        k: 1, k2: n_modes, krylov_dim: n_modes.min(3),
        seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None,
    };

    let pipeline = QfmPipeline::compile_channels(
        input_modes, output_modes, transitions, lambda0, lambda1,
        n_modes, &config, 0.0, 0.0, do_whiten, None, kernel_sigma,
    ).expect("pipeline compile");

    let rank = pipeline.rank();
    let w = pipeline.w();
    let n_in = input_modes.len();
    let n_out = output_modes.len();
    let o_stride = n_out;

    eprintln!("\n=== {} (rank={}, n_in={}, n_out={}) ===", label, rank, n_in, n_out);

    let mut correct = 0u32;
    let mut n_total = 0u32;
    let mut train_correct = 0u32;
    let mut n_train_total = 0u32;
    let even_o = 0usize;
    let odd_o = 1usize;
    let all_inputs: Vec<u32> = [train_inputs, &test_inputs].concat();

    for &x in &all_inputs {
        let input_pos = input_modes.iter().position(|&m| m == x).unwrap();
        let tp_even = input_pos * o_stride + even_o;
        let tp_odd  = input_pos * o_stride + odd_o;

        // Encode |x, vac⟩ = (1/√N_out)·Σ_f |x, f⟩
        let mut c0 = DVector::zeros(rank);
        for f in 0..n_out {
            let tp = input_pos * o_stride + f;
            for k in 0..rank {
                c0[k] += w[(tp, k)];
            }
        }
        let inv_sqrt_no = 1.0 / (n_out as f64).sqrt();
        c0.scale_mut(inv_sqrt_no);

        let c1 = pipeline.evolve(&c0, t);

        let amp_even: Complex64 = (0..rank).map(|k| c1[k] * w[(tp_even, k)].conj()).sum();
        let amp_odd: Complex64 = (0..rank).map(|k| c1[k] * w[(tp_odd, k)].conj()).sum();
        let p_even = amp_even.norm_sqr();
        let p_odd = amp_odd.norm_sqr();
        let total = (p_even + p_odd).max(1e-300);
        let p_even_norm = p_even / total;
        let p_odd_norm = p_odd / total;

        let predicted = p_even > p_odd;
        let correct_flag = predicted == parity(x);
        if correct_flag { correct += 1; }
        n_total += 1;

        if train_inputs.contains(&x) {
            if correct_flag { train_correct += 1; }
            n_train_total += 1;
        }

        eprintln!("  x={:04b}({})  P(even)={:.4e} P(odd)={:.4e}  {}",
            x, x, p_even_norm, p_odd_norm,
            if correct_flag { "✓" } else { "✗" });
    }
    eprintln!("  accuracy: {correct}/{n_total}  (train: {train_correct}/{n_train_total})");
    (correct, n_total, train_correct, n_train_total)
}

#[test]
fn parity_classification_tests() {
    let train_inputs: Vec<u32> = vec![
        1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14,
    ];
    let input_modes: Vec<u32> = (0..16u32).collect();
    let label_modes: Vec<u32> = vec![16, 17];

    let mut star_trans = Vec::new();
    for &x in &train_inputs {
        let y = if parity(x) { 16u32 } else { 17u32 };
        star_trans.push((x, y));
    }
    let (c1, n1, train_correct, n_train) = run_test(&train_inputs, &star_trans, &input_modes, &label_modes, 18,
                           "tensor-product TP", true, None);
    assert!(train_correct == n_train, "TP whitening failed: {train_correct}/{n_train}");
    eprintln!("\n  → with whitening: {c1}/{n1}");

    let (c1b, n1b, train_correct_b, n_train_b) = run_test(&train_inputs, &star_trans, &input_modes, &label_modes, 18,
                           "tensor-product full-rank", false, None);
    assert!(train_correct_b == n_train_b, "full-rank orthogonalization should match whitening: {train_correct_b}/{n_train_b}");
    eprintln!("\n  → full-rank orthogonalization: {c1b}/{n1b}  (train: {train_correct_b}/{n_train_b})");

    // Kernel-based inner product tests
    let (c_k, n_k, train_k, n_train_k) = run_test(
        &train_inputs, &star_trans, &input_modes, &label_modes, 18,
        "kernel σ=0.5", true, Some(0.5),
    );
    eprintln!("  → kernel σ=0.5: test {c_k}/{n_k}  train {train_k}/{n_train_k}");
    assert!(train_k == n_train, "kernel σ=0.5 training failed: {train_k}/{n_train}");
    assert!(c_k > 12, "kernel σ=0.5 should beat no-kernel baseline: {c_k}/{n_k}");
}
