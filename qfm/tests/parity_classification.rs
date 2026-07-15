//! Binary parity classification using the Hermitian Hamiltonian.
//! 4-bit inputs (0..15), 2 classes: even / odd.
//! Hamiltonian: H = λ₀·|c₀⟩⟨c₀| + λ₁·Σ_{i→f}(|f⟩⟨i|+|i⟩⟨f|)
//!
//! * λ₀: Mehler vacuum projector (prior, uniform over input modes)
//! * λ₁: Supervised transition couplings (label-specific pair coupling)
//!
//! With separate input/output outer vacua:
//! - R_in = 256 (Fock partitions for input modes)
//! - R_out = 2 (binary classification)

use nalgebra::DVector;
use num_complex::Complex64;
use qfm::pipeline::{QfmConfig, QfmPipeline};

fn parity(x: u32) -> bool {
    x.count_ones() % 2 == 0
}

fn run_test(train_inputs: &[u32], transitions: &[(u32, u32)],
            input_modes: &[u32], output_modes: &[u32],
            n_modes: usize, label: &str, do_whiten: bool) -> (u32, u32, u32, u32) {
    let test_inputs: Vec<u32> = (0..16u32).filter(|&x| !train_inputs.contains(&x)).collect();
    let has_labels = !output_modes.is_empty();

    let lambda0 = 1.0;
    let lambda1 = 1.0;
    let r_in = 256.0;
    let r_out = if has_labels { 2.0 } else { 0.0 };
    let t = 0.5;

    let config = QfmConfig {
        k: 1, k2: n_modes, krylov_dim: n_modes.min(3),
        seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None,
    };

    let pipeline = QfmPipeline::compile_channels(
        input_modes, output_modes, transitions, lambda0, lambda1,
        n_modes, &config, r_in, r_out, do_whiten, None,
    ).expect("pipeline compile");

    let rank = pipeline.rank();
    let w = pipeline.w();

    // Label mask: which modes vote for even/odd
    let mut even_mask = vec![0.0f64; n_modes];
    let mut odd_mask = vec![0.0f64; n_modes];
    for x in 0..16u32 {
        let xi = x as usize;
        if xi >= n_modes { continue; }
        if !train_inputs.contains(&x) { continue; }
        if parity(x) { even_mask[xi] = 1.0; }
        else { odd_mask[xi] = 1.0; }
    }
    // Label modes (16=even, 17=odd) vote directly
    if has_labels {
        if 16 < n_modes { even_mask[16] = 1.0; }
        if 17 < n_modes { odd_mask[17] = 1.0; }
    }

    eprintln!("\n=== {} (rank={}, n_modes={}) ===", label, rank, n_modes);

    // DEBUG: print per-mode probabilities for a training input (x=1)
    if rank <= 10 {
        let xi = 1usize;
        let c0 = DVector::from_iterator(rank, (0..rank).map(|k| w[(xi, k)]));
        let c1 = pipeline.evolve(&c0, t);
        let mut norm_dbg = 0.0f64;
        let mut per_mode = Vec::new();
        for m in 0..n_modes {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * w[(m, k)].conj()).sum();
            norm_dbg += amp.norm_sqr();
            per_mode.push((m, amp.norm_sqr()));
        }
        norm_dbg = norm_dbg.max(1e-300);
        eprintln!("[DEBUG] mode probabilities for input 1 (even):");
        per_mode.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (m, p) in per_mode.iter().take(6) {
            eprintln!("  mode {m:>2}: |amp|²={p:.6e}  frac={:.4e}", p / norm_dbg);
        }
        // Also print W row norms
        eprintln!("[DEBUG] W row norms:");
        for m in [1usize, 16, 17, 0, 4] {
            if m < n_modes {
                let row_norm: f64 = (0..rank).map(|k| w[(m, k)].norm_sqr()).sum();
                eprintln!("  mode {m:>2}: ||W row||²={row_norm:.6e}");
            }
        }
    }

    let mut correct = 0u32;
    let mut n_total = 0u32;
    let mut train_correct = 0u32;
    let mut n_train_total = 0u32;
    let all_inputs: Vec<u32> = [train_inputs, &test_inputs].concat();
    for &x in &all_inputs {
        let xi = x as usize;
        if xi >= n_modes { continue; }

        let c0 = DVector::from_iterator(rank, (0..rank).map(|k| w[(xi, k)]));
        let c1 = pipeline.evolve(&c0, t);
        let mut norm = 0.0f64;
        for m in 0..n_modes {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * w[(m, k)].conj()).sum();
            norm += amp.norm_sqr();
        }
        norm = norm.max(1e-300);
        let mut p_even = 0.0;
        let mut p_odd = 0.0;
        for m in 0..n_modes {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * w[(m, k)].conj()).sum();
            let p = amp.norm_sqr() / norm;
            p_even += p * even_mask[m];
            p_odd += p * odd_mask[m];
        }

        let predicted = p_even > p_odd;
        let correct_flag = predicted == parity(x);
        if correct_flag { correct += 1; }
        n_total += 1;

        if train_inputs.contains(&x) {
            if correct_flag { train_correct += 1; }
            n_train_total += 1;
        }

        eprintln!("  x={:04b}({})  P(even)={:.4e} P(odd)={:.4e}  {}",
            x, x, p_even, p_odd,
            if correct_flag { "✓" } else { "✗" });
    }
    eprintln!("  accuracy: {correct}/{n_total}  (train: {train_correct}/{n_train_total})");
    (correct, n_total, train_correct, n_train_total)
}

#[test]
fn parity_classification_tests() {
    // Same 16 data points (0..15). Asymmetric label distribution:
    // training = 5 even + 7 odd = 12, test = 4 (0, 8, 12, 15).
    // Training: 3,5,6,9,10 (even) + 1,2,4,7,11,13,14 (odd) = 12
    let train_inputs: Vec<u32> = vec![
        1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14,
    ];
    let input_modes: Vec<u32> = (0..16u32).collect();
    let label_modes: Vec<u32> = vec![16, 17];

    // Star topology: each training input → its parity label.
    let mut star_trans = Vec::new();
    for &x in &train_inputs {
        let y = if parity(x) { 16u32 } else { 17u32 };
        star_trans.push((x, y));
    }
    let (c1, n1, train_correct, n_train) = run_test(&train_inputs, &star_trans, &input_modes, &label_modes, 18,
                           "star WITH whitening", true);
    assert!(train_correct == n_train, "star WITH whitening failed: {train_correct}/{n_train}");
    eprintln!("\n  → with whitening: {c1}/{n1}");

    let (c1b, n1b, train_correct_b, n_train_b) = run_test(&train_inputs, &star_trans, &input_modes, &label_modes, 18,
                           "star full-rank orthogonalization (no truncation)", false);
    assert!(train_correct_b == n_train_b, "full-rank orthogonalization should match whitening when no null directions exist");
    eprintln!("\n  → full-rank orthogonalization: {c1b}/{n1b}  (train: {train_correct_b}/{n_train_b})");
}
