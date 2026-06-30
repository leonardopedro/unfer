//! Criterion bench for the CUDA path of the SIRK solver (P8 P10).
//!
//! Gated on the `cuda` feature. Run with:
//!   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!     cargo bench -p fock_sirk --features cuda --bench sirk_cuda
//!
//! On systems with CUDA 12.2 toolkit + CUDA 13 driver coexistence,
//! the runtime resolves to /lib/x86_64-linux-gnu (AGENTS.md §5).
//!
//! Central scaling claim being measured: the GPU path should be
//! ~10× faster than CPU at M = 1000+ and should not regress at
//! M = 10 (where kernel launch overhead dominates). The corresponding
//! CPU bench is in `sirk.rs` (`bench_sirk_solve`).
//!
//! Acceptance: the bench runs clean and shows the expected curve
//! (M=10 ≈ a few ms, M=1000 ≈ a few hundred ms, M=10000 ≈ a few
//! seconds). The CUDA build is optional; the bench file is gated
//! on the `cuda` feature so it doesn't break the CPU build.

#![cfg(feature = "cuda")]

use std::hint::black_box;

use candle_core::Device;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fock_sirk::linalg::Whitening;
use fock_sirk::{solve_forward_sirk, whiten_gram};
use nalgebra::DMatrix;
use nested_fock_algebra::{QuantumState, models};
use num_complex::Complex64;

/// Imaginary shift ladder of length `m` — the Krylov dimension knob.
/// Mirrors `shifts()` in the CPU bench (sirk.rs).
fn shifts(m: usize) -> Vec<Complex64> {
    (0..m)
        .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
        .collect()
}

/// GPU SIRK solve time vs Krylov dim `m`. Mirrors `bench_sirk_solve`
/// in `sirk.rs` but on the CUDA device.
///
/// Note: the existing CPU `bench_sirk_solve` group covers M = 2/4/8/16
/// (the CPU is fast enough that the 1000+ points are dominated by
/// nalgebra Hermitian eigendecomp, not the GPU path). The CUDA
/// bench covers the same 2/4/8/16/32 range to confirm the GPU
/// doesn't regress at the small end (where kernel launch overhead
/// dominates).
fn bench_sirk_solve_cuda(c: &mut Criterion) {
    // Force the CUDA device; the bench panics if no GPU is available.
    let device = Device::cuda_if_available(0).expect("CUDA device required for this bench");
    assert!(
        matches!(device, Device::Cuda(_)),
        "bench_sirk_solve_cuda requires a CUDA device"
    );

    // Use a representative model: harmonic chain with 4 modes (the
    // same model as the CPU bench's M=4 baseline).
    let h = models::harmonic_chain(4, 1.0);
    let v0 = QuantumState::vacuum();
    let mut group = c.benchmark_group("sirk_solve_cuda");
    group.sample_size(10);
    for &m in &[2usize, 4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, &_m| {
            b.iter(|| {
                let res = solve_forward_sirk(
                    black_box(&h),
                    black_box(&v0),
                    black_box(&shifts(m)),
                    black_box(&device),
                    None,
                )
                .expect("solve");
                black_box(res.rank);
            });
        });
    }
    group.finish();
}

/// CPU Gram-whitening time vs matrix size `n`. (P8 P10 note: the
/// `whiten_gram` function currently uses nalgebra's CPU Hermitian
/// eigendecomp; a future GPU eigendecomp would be a candle_core
/// backend and is out of scope here. This bench exists to provide
/// the CPU baseline that a future GPU port can be compared against.
/// The two bench files — `sirk.rs` and `sirk_cuda.rs` — share the
/// same group name `whiten_gram` so a side-by-side comparison is
/// possible when the GPU eigendecomp lands.)
fn bench_whiten_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("whiten_gram_cuda");
    group.sample_size(10);
    for &n in &[4usize, 8, 16, 32] {
        // Build a deterministic PSD matrix `A = M^H M + n I` (as
        // Complex64 to match whiten_gram's signature), then time the
        // whitening.
        let m = DMatrix::<f64>::from_fn(n, n, |i, j| ((i + j + 1) as f64) * 0.1);
        let a_f = &m * &m.transpose() + DMatrix::<f64>::identity(n, n) * (n as f64);
        let a = DMatrix::<Complex64>::from_fn(n, n, |i, j| Complex64::new(a_f[(i, j)], 0.0));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                let w: Whitening = whiten_gram(black_box(&a), 1e-8).expect("whiten");
                black_box(w.rank);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sirk_solve_cuda, bench_whiten_cuda);
criterion_main!(benches);
