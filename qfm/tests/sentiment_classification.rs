//! Sentiment classification with distributed multi-mode encoding,
//! using the tensor-product Hilbert space ℋ_in ⊗ ℋ_out.
//!
//! 14 feature modes (0..13), 2 label modes (14=POS, 15=NEG).
//! Hamiltonian: H = λ₀·|c₀⟩⟨c₀| + λ₁·Σ (|i⟩⟨i|)_in ⊗ (|f⟩⟨vac| + |vac⟩⟨f|)_out
//!
//! Uniform vacuum: |i, vac⟩ = (1/√N_out)·Σ_f |i,f⟩.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use qfm::pipeline::{QfmConfig, QfmPipeline};
use std::collections::BTreeSet;

const POS: u32 = 14;
const NEG: u32 = 15;
const N_FEATURES: usize = 14;
const N_OUTPUTS: usize = 2;

const LAMBDA0: f64 = 1.0;
const LAMBDA1: f64 = 1.0;
const T: f64 = 0.5;

fn word_features(word: &str) -> Vec<u32> {
    match word {
        "good"      => vec![0, 1],
        "great"     => vec![2, 3],
        "nice"      => vec![4, 5],
        "love"      => vec![6, 7],
        "bad"       => vec![8, 9],
        "terrible"  => vec![10, 11],
        "awful"     => vec![12, 13],
        "the"       => vec![0, 12],
        "a"         => vec![1, 13],
        "is"        => vec![2, 10],
        "excellent" => vec![0, 6],
        "dreadful"  => vec![8, 13],
        _ => panic!("unknown word: {word}"),
    }
}

fn word_label(word: &str) -> Option<&'static str> {
    match word {
        "good" | "great" | "nice" | "love" | "excellent" => Some("pos"),
        "bad" | "terrible" | "awful" | "dreadful" => Some("neg"),
        "the" | "a" | "is" => None,
        _ => panic!("unknown word: {word}"),
    }
}

const TRAINING_WORDS: &[&str] = &[
    "good", "great", "nice", "love",
    "bad", "terrible", "awful",
];

const HELD_OUT_WORDS: &[&str] = &["excellent", "dreadful"];
const NEUTRAL_WORDS: &[&str] = &["the", "a", "is"];

fn encode_vacuum(features: &[u32], n_out: usize, o_stride: usize, w: &DMatrix<Complex64>, rank: usize) -> DVector<Complex64> {
    let mut c0 = DVector::zeros(rank);
    let inv_sqrt_total = 1.0 / ((features.len() * n_out) as f64).sqrt();
    for &f in features {
        for g in 0..n_out {
            let tp = (f as usize) * o_stride + g;
            for k in 0..rank {
                c0[k] += w[(tp, k)] * inv_sqrt_total;
            }
        }
    }
    c0
}

fn run_test(label: &str) -> (u32, u32, u32, u32) {
    let input_modes: Vec<u32> = (0..N_FEATURES as u32).collect();
    let output_modes = &[POS, NEG];
    let n_out = output_modes.len();
    let o_stride = n_out;

    let mut transitions = Vec::new();
    for &word in TRAINING_WORDS {
        let label = if word_label(word) == Some("pos") { POS } else { NEG };
        for &f in &word_features(word) {
            transitions.push((f, label));
        }
    }

    let config = QfmConfig {
        k: 1, k2: N_FEATURES + N_OUTPUTS, krylov_dim: 14,
        seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None,
        ..Default::default()
    };

    let pipeline = QfmPipeline::compile_channels(
        &input_modes, output_modes, &transitions,
        LAMBDA0, LAMBDA1, N_FEATURES + N_OUTPUTS, &config, 0.0, 0.0, true,
        None, None,
    ).expect("pipeline compile");

    let rank = pipeline.rank();
    let w = pipeline.w();

    eprintln!("\n=== {} (rank={}) ===", label, rank);

    let mut correct = 0u32;
    let mut total = 0u32;
    let mut train_correct = 0u32;
    let mut train_total = 0u32;

    let all_words: Vec<&str> = [TRAINING_WORDS, HELD_OUT_WORDS, NEUTRAL_WORDS].concat();

    for &word in &all_words {
        let features = word_features(word);
        let c0 = encode_vacuum(&features, n_out, o_stride, &w, rank);
        let c1 = pipeline.evolve(&c0, T);

        let mut p_pos = 0.0;
        let mut p_neg = 0.0;
        for &f in &features {
            let tp = (f as usize) * o_stride;
            let amp_pos: Complex64 = (0..rank).map(|k| c1[k] * w[(tp + 0, k)].conj()).sum();
            let amp_neg: Complex64 = (0..rank).map(|k| c1[k] * w[(tp + 1, k)].conj()).sum();
            p_pos += amp_pos.norm_sqr();
            p_neg += amp_neg.norm_sqr();
        }

        let predicted = if p_pos > p_neg { "pos" } else { "neg" };
        let true_label = word_label(word);

        let is_train = TRAINING_WORDS.contains(&word);
        let is_held_out = HELD_OUT_WORDS.contains(&word);

        if let Some(expected) = true_label {
            let correct_flag = predicted == expected;

            if is_train {
                if correct_flag { train_correct += 1; }
                train_total += 1;
                if correct_flag { correct += 1; }
                total += 1;
                eprintln!("  [train] {word:>10}: → {predicted}(pos={p_pos:.4e},neg={p_neg:.4e})  {}",
                    if correct_flag { "✓" } else { "✗" });
            } else if is_held_out {
                if correct_flag { correct += 1; }
                total += 1;
                eprintln!("  [test]  {word:>10}: → {predicted}(pos={p_pos:.4e},neg={p_neg:.4e})  {}",
                    if correct_flag { "✓" } else { "✗" });
            }
        } else {
            eprintln!("  [neut]  {word:>10}: → {predicted}(pos={p_pos:.4e},neg={p_neg:.4e})");
        }
    }

    // Compound sentences as bag-of-words superposition
    let test_sentences: Vec<(Vec<&str>, &str)> = vec![
        (vec!["the", "good"], "pos"),
        (vec!["the", "great"], "pos"),
        (vec!["a", "terrible"], "neg"),
        (vec!["the", "a", "awful"], "neg"),
        (vec!["nice"], "pos"),
        (vec!["love"], "pos"),
        (vec!["the", "excellent"], "pos"),
        (vec!["the", "dreadful"], "neg"),
    ];

    for (words, expected) in &test_sentences {
        let mut all_features = BTreeSet::new();
        for &w in words {
            for &f in &word_features(w) {
                all_features.insert(f);
            }
        }
        let modes: Vec<u32> = all_features.into_iter().collect();
        let c0 = encode_vacuum(&modes, n_out, o_stride, &w, rank);
        let c1 = pipeline.evolve(&c0, T);

        let mut p_pos = 0.0;
        let mut p_neg = 0.0;
        for &m in &modes {
            let tp = (m as usize) * o_stride;
            let amp_pos: Complex64 = (0..rank).map(|k| c1[k] * w[(tp + 0, k)].conj()).sum();
            let amp_neg: Complex64 = (0..rank).map(|k| c1[k] * w[(tp + 1, k)].conj()).sum();
            p_pos += amp_pos.norm_sqr();
            p_neg += amp_neg.norm_sqr();
        }

        let predicted = if p_pos > p_neg { "pos" } else { "neg" };
        let correct_flag = predicted == *expected;
        if correct_flag { correct += 1; }
        total += 1;
        eprintln!("  [sent]  \"{}\": → {predicted}(pos={p_pos:.4e},neg={p_neg:.4e})  {}",
            words.join(" "),
            if correct_flag { "✓" } else { "✗" });
    }

    eprintln!("  accuracy: {correct}/{total}  (train: {train_correct}/{train_total})");
    (correct, total, train_correct, train_total)
}

#[test]
fn sentiment_classification_tests() {
    let (c, n, tc, tn) = run_test("tensor-product sentiment");
    eprintln!("\nResults: {c}/{n}  (train: {tc}/{tn})");
    assert!(tc == tn, "training failed: {tc}/{tn}");
}
