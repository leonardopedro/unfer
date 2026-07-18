//! P10.16.3: rank-truncation extension tests.
//!
//! Validates that `QfmConfig { max_rank: Some(r) }` with `krylov_dim << K_2`
//! compiles and runs correctly for d=1024 (CIFAR-10 32×32 resolution) without
//! hitting the O(K_2³) W_prob cubic-scaling wall that would occur if
//! krylov_dim = K_2 = 1024 were required.
//!
//! Synthetic random pixels are used so no CIFAR-10 download is needed.

use qfm::{QfmConfig, QfmPipeline};

/// Build a synthetic d=1024 training set: `n` random images (values in [0,1])
/// generated from a linear-congruential sequence seeded by `seed`.
fn synthetic_d1024(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let d = 1024;
    let mut state: u64 = seed ^ 0x9e3779b97f4a7c15;
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as f64 / (u32::MAX as f64)
                })
                .collect()
        })
        .collect()
}

/// Build a pipeline with d=1024, K_2=1024, krylov_dim=32, max_rank=r.
/// This is the rank-truncation (P10.16.3) path; without max_rank this
/// config would return K2ExceedsKrylovDim.
fn pipeline_d1024(n_train: usize, krylov_dim: usize, max_rank: usize, seed: u64) -> QfmPipeline {
    let training = synthetic_d1024(n_train, seed);
    let k2 = 1024; // K_2 = d for 32×32 CIFAR-10
    let config = QfmConfig {
        k: 32,
        k2,
        krylov_dim,
        seed,
        n_t_samples: 4,
        noise_dim: k2,
        max_rank: Some(max_rank),
        ..Default::default()
    };
    QfmPipeline::compile(&training, &config)
        .expect("rank-truncation pipeline must compile for d=1024")
}

#[test]
fn rank_truncation_d1024_compiles_without_cubic_wall() {
    // krylov_dim=32 << K_2=1024 — would fail without max_rank.
    let pipeline = pipeline_d1024(64, 32, 16, 42);
    // At least rank-1 truncated basis.
    assert!(pipeline.rank() >= 1, "rank must be >= 1 after truncation");
    assert!(
        pipeline.rank() <= 16,
        "rank must be <= max_rank=16 after truncation, got {}",
        pipeline.rank()
    );
    assert_eq!(pipeline.raw_dim(), 1024, "raw dim must be d=1024");
    assert_eq!(pipeline.k2_dim(), 1024, "K_2 must be 1024");
    println!(
        "rank_truncation_d1024: rank={}, K_2={}, d={}",
        pipeline.rank(),
        pipeline.k2_dim(),
        pipeline.raw_dim()
    );
}

#[test]
fn rank_truncation_d1024_generate_finite() {
    let n_train = 64;
    let training = synthetic_d1024(n_train, 7);
    let k2 = 1024;
    let config = QfmConfig {
        k: 32,
        k2,
        krylov_dim: 32,
        seed: 7,
        n_t_samples: 4,
        noise_dim: k2,
        max_rank: Some(8),
        ..Default::default()
    };
    let pipeline =
        QfmPipeline::compile(&training, &config).expect("compile d=1024 rank-8 truncation");

    // Generate from the first training point; result must be finite.
    let x_out = pipeline
        .generate(&training[0])
        .expect("generate must succeed");
    assert_eq!(x_out.len(), 1024, "output must be d=1024 pixels");
    for &v in &x_out {
        assert!(
            v.is_finite(),
            "rank-truncation generate output must be finite, got {v}"
        );
    }
    println!(
        "rank_truncation_d1024_generate: output[0..4] = {:?}",
        &x_out[..4]
    );
}

#[test]
fn rank_truncation_without_max_rank_errors_on_small_krylov() {
    // Confirm K2ExceedsKrylovDim is still returned when max_rank is None.
    let training = synthetic_d1024(16, 99);
    let config = QfmConfig {
        k: 8,
        k2: 1024,
        krylov_dim: 16, // << k2=1024
        seed: 99,
        n_t_samples: 2,
        noise_dim: 1024,
        max_rank: None, // no truncation → must error
        ..Default::default()
    };
    let result = QfmPipeline::compile(&training, &config);
    assert!(
        result.is_err(),
        "expected K2ExceedsKrylovDim error without max_rank"
    );
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("K_2"), "error must mention K_2, got: {msg}");
}

#[test]
fn rank_truncation_svd_reduces_rank() {
    // With max_rank=4 and krylov_dim=32, the compiled rank must be <= 4.
    let pipeline = pipeline_d1024(32, 32, 4, 123);
    assert!(
        pipeline.rank() <= 4,
        "rank must not exceed max_rank=4, got {}",
        pipeline.rank()
    );
}
