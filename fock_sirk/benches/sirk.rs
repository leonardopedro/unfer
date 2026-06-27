//! Criterion benches for the load-bearing SIRK numerics (plan P3 #13 / P4 #19).
//!
//! Pass/fail tests catch crashes; these guard the *cost* of the three paths a
//! subtle regression would otherwise hide in: the forward SIRK solve, the Gram
//! whitening that replaced the bare Cholesky (Stage 2), and the whitened →
//! `QuantumState` reconstruction (Stage 6). Each is benched against the
//! parameter that drives its complexity (Krylov dim / matrix size) so a
//! slowdown shows up as a curve, not a single number.
//!
//! P5 #30 extensions: Krylov dim up to 16 in the solve + reconstruct groups;
//! new `yang_mills_build_vs_l` group timing Hamiltonian construction at l=2..4.
//!
//! Run with: `cargo bench -p fock_sirk`.

use std::hint::black_box;

use candle_core::Device;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fock_sirk::{GRAM_REL_TOL, solve_forward_sirk, whiten_gram};
use nalgebra::DMatrix;
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, models};
use num_complex::Complex64;

/// Imaginary shift ladder of length `m` — the Krylov dimension knob.
fn shifts(m: usize) -> Vec<Complex64> {
    (0..m)
        .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
        .collect()
}

/// Initial state: one boson in mode 0 of the outer Fock space.
fn one_boson_mode0() -> QuantumState {
    let mut inner = InnerBosonicState::vacuum();
    inner.modes.insert(0, 1);
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner))
}

/// A deterministic Hermitian positive-definite `n x n` matrix `G = MᴴM + nI`,
/// so `whiten_gram` exercises a full-rank eigendecomposition without a random
/// dependency (and the `nI` shift keeps it well away from degeneracy).
fn hermitian_psd(n: usize) -> DMatrix<Complex64> {
    let m = DMatrix::from_fn(n, n, |i, j| {
        let re = ((i * 7 + j * 3 + 1) % 11) as f64 / 11.0;
        let im = ((i * 5 + j * 2 + 1) % 13) as f64 / 13.0;
        Complex64::new(re, im)
    });
    let mut g = m.adjoint() * &m;
    for i in 0..n {
        g[(i, i)] += Complex64::new(n as f64, 0.0);
    }
    g
}

/// Forward SIRK solve on a 4-mode harmonic chain vs Krylov dimension.
/// Extended to m=16 (P5 #30) to reveal the Gram-whitening cost at high dim.
fn bench_sirk_solve(c: &mut Criterion) {
    let device = Device::Cpu;
    let h = models::harmonic_chain(4, 1.0);
    let v0 = one_boson_mode0();
    let mut group = c.benchmark_group("sirk_solve_vs_krylov_dim");
    for &m in &[2usize, 4, 8, 16] {
        let sh = shifts(m);
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, _| {
            b.iter(|| {
                solve_forward_sirk(black_box(&h), black_box(&v0), black_box(&sh), &device, None)
                    .expect("solve must succeed")
            });
        });
    }
    group.finish();
}

/// Gram whitening (Hermitian eigendecomposition) vs matrix size.
fn bench_whiten(c: &mut Criterion) {
    let mut group = c.benchmark_group("whiten_gram_vs_size");
    for &n in &[4usize, 8, 16, 32, 64] {
        let g = hermitian_psd(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| whiten_gram(black_box(&g), GRAM_REL_TOL).expect("whiten"));
        });
    }
    group.finish();
}

/// Whitened-coefficient → `QuantumState` reconstruction vs Krylov dimension
/// (i.e. `w_sequence` length). Extended to m=16 (P5 #30).
fn bench_reconstruct(c: &mut Criterion) {
    let device = Device::Cpu;
    let h = models::harmonic_chain(4, 1.0);
    let v0 = one_boson_mode0();
    let mut group = c.benchmark_group("reconstruct_vs_krylov_dim");
    for &m in &[2usize, 4, 8, 16] {
        let res = solve_forward_sirk(&h, &v0, &shifts(m), &device, None).expect("solve");
        let coeffs = res.time_evolve(0.5);
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, _| {
            b.iter(|| res.reconstruct(black_box(&coeffs)));
        });
    }
    group.finish();
}

/// Yang-Mills lattice Hamiltonian construction cost vs lattice size (P5 #30).
/// Construction is pure symbolic algebra (no SIRK): l=2 → 72 terms,
/// l=3 → 162 terms, l=4 → 288 terms. Reveals O(l²) scaling.
fn bench_yang_mills_build_vs_l(c: &mut Criterion) {
    let mut group = c.benchmark_group("yang_mills_build_vs_l");
    for &l in &[2usize, 3, 4] {
        group.bench_with_input(BenchmarkId::from_parameter(l), &l, |b, _| {
            b.iter(|| models::yang_mills_lattice(black_box(l), 1.0, 1));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sirk_solve,
    bench_whiten,
    bench_reconstruct,
    bench_yang_mills_build_vs_l
);
criterion_main!(benches);
