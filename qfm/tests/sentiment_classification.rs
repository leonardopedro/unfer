//! Sentiment classification with distributed multi-mode encoding.
//!
//! 14 feature modes (0..13), 2 label modes (14=POS, 15=NEG).
//! Each training word uses a unique pair of feature modes (dedicated),
//! so there is no cross-word feature overlap that would cause quantum
//! interference in the Krylov space. The uniform starting vector over
//! all 14 feature modes and the SIRK Hamiltonian with asymmetric
//! per-mode weights produce correct label discrimination for every
//! training word.
//!
//! Held-out words use combinations of existing training features to test
//! compositional generalization, but only training accuracy is asserted.

use nalgebra::DVector;
use num_complex::Complex64;
use qfm::pipeline::{QfmConfig, QfmPipeline};
use std::collections::BTreeSet;

// 14 feature modes (2 per training word), label modes 14=POS, 15=NEG
const POS: u32 = 14;
const NEG: u32 = 15;
const N_MODES: usize = 16;

const LAMBDA0: f64 = 1.0;
const LAMBDA1: f64 = 1.0;
const R_IN: f64 = 256.0;
const R_OUT: f64 = 2.0;
const T: f64 = 0.5;

// Each training word gets a unique pair of feature modes, eliminating
// cross-word interference. Held-out words are built from existing
// features for compositional generalization tests (informational only).
fn word_features(word: &str) -> Vec<u32> {
    match word {
        // Positive training (4 words × 2 features = 8 distinct POS features)
        "good"      => vec![0, 1],
        "great"     => vec![2, 3],
        "nice"      => vec![4, 5],
        "love"      => vec![6, 7],
        // Negative training (3 words × 2 features = 6 distinct NEG features)
        "bad"       => vec![8, 9],
        "terrible"  => vec![10, 11],
        "awful"     => vec![12, 13],
        // Neutral (non-training words — included in starting vector)
        "the"       => vec![0, 12],
        "a"         => vec![1, 13],
        "is"        => vec![2, 10],
        // Held-out test words (combine POS and NEG training features)
        "excellent" => vec![0, 6],   // F0(good/POS) + F6(love/POS) → pos
        "dreadful"  => vec![8, 13],  // F8(bad/NEG) + F13(awful/NEG) → neg
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

fn run_test(label: &str) -> (u32, u32, u32, u32) {
    let input_modes: Vec<u32> = (0..14).collect();
    let output_modes = &[POS, NEG];

    // Build training transitions: each training word's 2 feature modes
    // each get a transition to the word's label.
    let mut transitions = Vec::new();
    for &word in TRAINING_WORDS {
        let label = if word_label(word) == Some("pos") { POS } else { NEG };
        for &f in &word_features(word) {
            transitions.push((f, label));
        }
    }

    let config = QfmConfig {
        k: 1, k2: N_MODES, krylov_dim: 14,
        seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None,
    };

    let pipeline = QfmPipeline::compile_channels(
        &input_modes, output_modes, &transitions,
        LAMBDA0, LAMBDA1, N_MODES, &config, R_IN, R_OUT, true,
        None,
    ).expect("pipeline compile");

    let rank = pipeline.rank();

    // Label masks: mode 6 = POS, mode 7 = NEG
    let mut pos_mask = vec![0.0f64; N_MODES];
    let mut neg_mask = vec![0.0f64; N_MODES];
    pos_mask[POS as usize] = 1.0;
    neg_mask[NEG as usize] = 1.0;

    eprintln!("\n=== {} (rank={}) ===", label, rank);

    let mut correct = 0u32;
    let mut total = 0u32;
    let mut train_correct = 0u32;
    let mut train_total = 0u32;

    let all_words: Vec<&str> = [TRAINING_WORDS, HELD_OUT_WORDS, NEUTRAL_WORDS].concat();

    for &word in &all_words {
        let features = word_features(word);

        let c0 = if features.len() == 1 {
            DVector::from_iterator(rank, (0..rank).map(|k| pipeline.w()[(features[0] as usize, k)]))
        } else {
            pipeline.encode_modes(&features).expect("encode_modes failed")
        };
        let c1 = pipeline.evolve(&c0, T);

        let mut norm = 0.0f64;
        for j in 0..N_MODES {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * pipeline.w()[(j, k)].conj()).sum();
            norm += amp.norm_sqr();
        }
        norm = norm.max(1e-300);

        let mut p_pos = 0.0;
        let mut p_neg = 0.0;
        for j in 0..N_MODES {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * pipeline.w()[(j, k)].conj()).sum();
            let p = amp.norm_sqr() / norm;
            p_pos += p * pos_mask[j];
            p_neg += p * neg_mask[j];
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

    // Compound sentences as bag-of-words superposition (deduplicated features)
    // Held-out words: excellent(F0,F3→pos), dreadful(F1,F4→neg)
    let test_sentences: Vec<(Vec<&str>, &str)> = vec![
        (vec!["the", "good"], "pos"),
        (vec!["the", "great"], "pos"),
        (vec!["a", "terrible"], "neg"),
        (vec!["the", "a", "awful"], "neg"),
        (vec!["nice"], "pos"),
        (vec!["love"], "pos"),             // training
        (vec!["the", "excellent"], "pos"), // neutral + held-out POS
        (vec!["the", "dreadful"], "neg"),  // neutral + held-out
    ];

    for (words, expected) in &test_sentences {
        let mut all_features = BTreeSet::new();
        for &w in words {
            for &f in &word_features(w) {
                all_features.insert(f);
            }
        }
        let modes: Vec<u32> = all_features.into_iter().collect();
        let c0 = pipeline.encode_modes(&modes).expect("encode_modes failed");
        let c1 = pipeline.evolve(&c0, T);

        let mut norm = 0.0f64;
        for j in 0..N_MODES {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * pipeline.w()[(j, k)].conj()).sum();
            norm += amp.norm_sqr();
        }
        norm = norm.max(1e-300);

        let mut p_pos = 0.0;
        let mut p_neg = 0.0;
        for j in 0..N_MODES {
            let amp: Complex64 = (0..rank).map(|k| c1[k] * pipeline.w()[(j, k)].conj()).sum();
            let p = amp.norm_sqr() / norm;
            p_pos += p * pos_mask[j];
            p_neg += p * neg_mask[j];
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
    let (c, n, tc, tn) = run_test("multi-mode sentiment");
    eprintln!("\nResults: {c}/{n}  (train: {tc}/{tn})");
    assert!(tc == tn, "training failed: {tc}/{tn}");
}
