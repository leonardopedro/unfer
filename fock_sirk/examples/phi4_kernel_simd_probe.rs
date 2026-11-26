//! Explicit-SIMD complex `axpy`/`dot` against the committed scalar kernels.
//!
//! The rational-Krylov loop spends its non-`gemv` time in `coords_inner_product`
//! and `coords_axpy` (`joint_sirk.rs`), both scalar `Complex<f64>` loops. This
//! probe rebuilds the inner product with NEON intrinsics on `aarch64` and with
//! AVX2/FMA intrinsics on `x86_64`, and the `axpy` with NEON on `aarch64` and
//! AVX2 on `x86_64` — deinterleaving
//! the array-of-structs pairs with `vuzp1q/vuzp2q` (NEON) or
//! `permute2f128`+`unpacklo/hi` (AVX2), with independent accumulators to break
//! the dependent-add chain — and times them against the committed shape
//! (zero-entry skip plus the per-element non-finiteness test).
//!
//! ```text
//! cargo run -q -p fock_sirk --release --example phi4_kernel_simd_probe
//! NPROBE_N=4845 NPROBE_ROWS=64 cargo run -q -p fock_sirk --release \
//!     --example phi4_kernel_simd_probe
//! ```
//!
//! Measured on the 12-core aarch64 host (`docs/VERIFY_PHI4_MOMENTUM.md` §5.12),
//! `NPROBE_N=2380` (the `C = 26` window width), `rows=8`, `reps=400`:
//!
//! | kernel | scalar | NEON | ratio |
//! |---|---|---|---|
//! | `axpy` | `2.29-2.31 us` | `0.93-0.94 us` | `2.45-2.48x` |
//! | `dot` | `1.44-2.81 us` | `1.39 us` | `1.03-2.07x` |
//!
//! The `x86_64` kernel, on this host (`NPROBE_N=2380`, rows 8, reps 400; and
//! `NPROBE_N=4845`, rows 16):
//!
//! | kernel | scalar | AVX2 | ratio |
//! |---|---|---|---|
//! | `dot` (`n = 2380`, two runs) | `2.975-3.434 us` | `1.246-1.255 us` | `2.37-2.76x` |
//! | `axpy` (`n = 2380`, two runs) | `6.791-7.008 us` | `2.537-2.559 us` | `2.65-2.76x` |
//! | `dot` (`n = 4845`, two runs) | `6.121-6.537 us` | `2.692-2.812 us` | `2.18-2.43x` |
//! | `axpy` (`n = 4845`, two runs) | `13.705-14.563 us` | `5.555-5.982 us` | `2.43-2.47x` |
//!
//! with `dot` agreement to `7.3e-15` and `3.2e-15` relative. The `axpy` kernel
//! reassociates nothing (unfused multiply and add in the scalar loop's order),
//! so its agreement is exactly zero on these inputs and its solve-time effect
//! is a pure speedup. The `dot` gain is smaller than NEON's because LLVM
//! already auto-vectorises the scalar loop in this build.
//!
//! The `dot` row is deliberately a range: the *scalar reference* is not
//! reliably scalar across builds -- LLVM vectorises this loop in some
//! compilation contexts (`1.44 us`) and not in others (`2.81 us`) -- so the
//! microbenchmark cannot settle that kernel. The end-to-end measurement can,
//! because in the solver the scalar single accumulator is latency-bound: patched
//! in and timed at `C = 26` the NEON path took `dot` from `1.65 s` to `0.44 s`
//! (`3.75x`) and the solve from `9.760 s` to `8.528 s` (`12.6 %`). That, not
//! the table above, is the number §5.12 quotes. The path **is** the committed
//! one on `aarch64` (`joint_sirk::coords_inner_product`), and on `x86_64` the
//! AVX2 kernel is committed whenever `is_x86_feature_detected!` reports both
//! `avx2` and `fma`, with the scalar loop as the fallback. Both reassociate the
//! sum: NEON moves the solve's last bits (`probability` differs at `1.6e-16`,
//! `917` vs `915` iterations), and AVX2 moved the `C = 20` probability by
//! `3.7e-15` with `528 → 527` iterations. The frozen sweep record was generated
//! on the scalar `x86_64` path before the AVX2 kernel existed, so its last bits
//! are that path's; the `axpy` kernel was added afterwards and changes no
//! digit at all.
//!
//! On a CPU without the tested features only the scalar timings are reported.

use num_complex::Complex;
use std::hint::black_box;
use std::time::Instant;

type C = Complex<f64>;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// The committed `coords_axpy`: zero entries are skipped, and a non-finite
/// result is reported from inside the loop.
fn scalar_axpy(a: &mut [C], b: &[C], scale: C) -> bool {
    let zero = C::new(0.0, 0.0);
    for (slot, value) in a.iter_mut().zip(b) {
        if *value == zero {
            continue;
        }
        let updated = *slot + scale * value;
        if !updated.re.is_finite() || !updated.im.is_finite() {
            return false;
        }
        *slot = updated;
    }
    true
}

/// The committed `coords_inner_product`: one accumulator, zero entries skipped.
fn scalar_dot(a: &[C], b: &[C]) -> C {
    let zero = C::new(0.0, 0.0);
    let mut sum = zero;
    for (x, y) in a.iter().zip(b) {
        if *x == zero || *y == zero {
            continue;
        }
        sum += x.conj() * y;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::C;
    use core::arch::x86_64::*;

    /// The committed `x86_64` `axpy`: four complex per vector, unfused mul+add
    /// in the scalar loop's order, so the result is bit-identical.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(a: &mut [C], b: &[C], scale: C) {
        let n = a.len().min(b.len());
        let ap = a.as_mut_ptr() as *mut f64;
        let bp = b.as_ptr() as *const f64;
        let sr = _mm256_set1_pd(scale.re);
        let si = _mm256_set1_pd(scale.im);
        let mut i = 0;
        while i + 4 <= n {
            let (av0, av1, bv0, bv1) = unsafe {
                (
                    _mm256_loadu_pd(ap.add(2 * i)),
                    _mm256_loadu_pd(ap.add(2 * i + 4)),
                    _mm256_loadu_pd(bp.add(2 * i)),
                    _mm256_loadu_pd(bp.add(2 * i + 4)),
                )
            };
            let at0 = _mm256_permute2f128_pd(av0, av1, 0x20);
            let at1 = _mm256_permute2f128_pd(av0, av1, 0x31);
            let ar = _mm256_unpacklo_pd(at0, at1);
            let ai = _mm256_unpackhi_pd(at0, at1);
            let bt0 = _mm256_permute2f128_pd(bv0, bv1, 0x20);
            let bt1 = _mm256_permute2f128_pd(bv0, bv1, 0x31);
            let br = _mm256_unpacklo_pd(bt0, bt1);
            let bi = _mm256_unpackhi_pd(bt0, bt1);
            let nr = _mm256_add_pd(
                ar,
                _mm256_sub_pd(_mm256_mul_pd(sr, br), _mm256_mul_pd(si, bi)),
            );
            let ni = _mm256_add_pd(
                ai,
                _mm256_add_pd(_mm256_mul_pd(sr, bi), _mm256_mul_pd(si, br)),
            );
            let r0 = _mm256_unpacklo_pd(nr, ni);
            let r1 = _mm256_unpackhi_pd(nr, ni);
            unsafe {
                _mm256_storeu_pd(ap.add(2 * i), _mm256_permute2f128_pd(r0, r1, 0x20));
                _mm256_storeu_pd(ap.add(2 * i + 4), _mm256_permute2f128_pd(r0, r1, 0x31));
            }
            i += 4;
        }
        while i < n {
            a[i] += scale * b[i];
            i += 1;
        }
    }

    /// The committed `x86_64` kernel: four complex per vector, deinterleaved
    /// with `permute2f128` + `unpacklo/hi`, two independent accumulator pairs.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[C], b: &[C]) -> C {
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn load(
            ap: *const f64,
            bp: *const f64,
            offset: usize,
        ) -> (__m256d, __m256d, __m256d, __m256d) {
            let av0 = unsafe { _mm256_loadu_pd(ap.add(offset)) };
            let av1 = unsafe { _mm256_loadu_pd(ap.add(offset + 4)) };
            let bv0 = unsafe { _mm256_loadu_pd(bp.add(offset)) };
            let bv1 = unsafe { _mm256_loadu_pd(bp.add(offset + 4)) };
            let at0 = _mm256_permute2f128_pd(av0, av1, 0x20);
            let at1 = _mm256_permute2f128_pd(av0, av1, 0x31);
            let bt0 = _mm256_permute2f128_pd(bv0, bv1, 0x20);
            let bt1 = _mm256_permute2f128_pd(bv0, bv1, 0x31);
            (
                _mm256_unpacklo_pd(at0, at1),
                _mm256_unpackhi_pd(at0, at1),
                _mm256_unpacklo_pd(bt0, bt1),
                _mm256_unpackhi_pd(bt0, bt1),
            )
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn horizontal(v: __m256d) -> f64 {
            let lo = _mm256_castpd256_pd128(v);
            let hi = _mm256_extractf128_pd(v, 1);
            let pair = _mm_add_pd(lo, hi);
            _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)))
        }

        let n = a.len().min(b.len());
        let ap = a.as_ptr() as *const f64;
        let bp = b.as_ptr() as *const f64;
        let mut re0 = _mm256_setzero_pd();
        let mut im0 = _mm256_setzero_pd();
        let mut re1 = _mm256_setzero_pd();
        let mut im1 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let (ar0, ai0, br0, bi0) = unsafe { load(ap, bp, 2 * i) };
            let (ar1, ai1, br1, bi1) = unsafe { load(ap, bp, 2 * i + 8) };
            re0 = _mm256_fmadd_pd(ar0, br0, re0);
            re0 = _mm256_fmadd_pd(ai0, bi0, re0);
            im0 = _mm256_fmadd_pd(ar0, bi0, im0);
            im0 = _mm256_fnmadd_pd(ai0, br0, im0);
            re1 = _mm256_fmadd_pd(ar1, br1, re1);
            re1 = _mm256_fmadd_pd(ai1, bi1, re1);
            im1 = _mm256_fmadd_pd(ar1, bi1, im1);
            im1 = _mm256_fnmadd_pd(ai1, br1, im1);
            i += 8;
        }
        while i + 4 <= n {
            let (ar, ai, br, bi) = unsafe { load(ap, bp, 2 * i) };
            re0 = _mm256_fmadd_pd(ar, br, re0);
            re0 = _mm256_fmadd_pd(ai, bi, re0);
            im0 = _mm256_fmadd_pd(ar, bi, im0);
            im0 = _mm256_fnmadd_pd(ai, br, im0);
            i += 4;
        }
        let re = unsafe { horizontal(_mm256_add_pd(re0, re1)) };
        let im = unsafe { horizontal(_mm256_add_pd(im0, im1)) };
        let mut sum = C::new(re, im);
        while i < n {
            sum += a[i].conj() * b[i];
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::C;
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn axpy(a: &mut [C], b: &[C], scale: C) {
        let n = a.len();
        let sr = vdupq_n_f64(scale.re);
        let nsi = vdupq_n_f64(-scale.im);
        let si = vdupq_n_f64(scale.im);
        let ap = a.as_mut_ptr() as *mut f64;
        let bp = b.as_ptr() as *const f64;
        let mut i = 0;
        while i + 2 <= n {
            let v0 = unsafe { vld1q_f64(ap.add(2 * i)) };
            let v1 = unsafe { vld1q_f64(ap.add(2 * i + 2)) };
            let u0 = unsafe { vld1q_f64(bp.add(2 * i)) };
            let u1 = unsafe { vld1q_f64(bp.add(2 * i + 2)) };
            let ar = vuzp1q_f64(v0, v1);
            let ai = vuzp2q_f64(v0, v1);
            let br = vuzp1q_f64(u0, u1);
            let bi = vuzp2q_f64(u0, u1);
            let nr = vfmaq_f64(vfmaq_f64(ar, sr, br), nsi, bi);
            let ni = vfmaq_f64(vfmaq_f64(ai, sr, bi), si, br);
            unsafe { vst1q_f64(ap.add(2 * i), vzip1q_f64(nr, ni)) };
            unsafe { vst1q_f64(ap.add(2 * i + 2), vzip2q_f64(nr, ni)) };
            i += 2;
        }
        while i < n {
            a[i] += scale * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot(a: &[C], b: &[C]) -> C {
        let n = a.len().min(b.len());
        let ap = a.as_ptr() as *const f64;
        let bp = b.as_ptr() as *const f64;
        let mut re0 = vdupq_n_f64(0.0);
        let mut re1 = vdupq_n_f64(0.0);
        let mut im0 = vdupq_n_f64(0.0);
        let mut im1 = vdupq_n_f64(0.0);
        let mut i = 0;
        while i + 4 <= n {
            let v0 = unsafe { vld1q_f64(ap.add(2 * i)) };
            let v1 = unsafe { vld1q_f64(ap.add(2 * i + 2)) };
            let v2 = unsafe { vld1q_f64(ap.add(2 * i + 4)) };
            let v3 = unsafe { vld1q_f64(ap.add(2 * i + 6)) };
            let w0 = unsafe { vld1q_f64(bp.add(2 * i)) };
            let w1 = unsafe { vld1q_f64(bp.add(2 * i + 2)) };
            let w2 = unsafe { vld1q_f64(bp.add(2 * i + 4)) };
            let w3 = unsafe { vld1q_f64(bp.add(2 * i + 6)) };
            let ar0 = vuzp1q_f64(v0, v1);
            let ai0 = vuzp2q_f64(v0, v1);
            let ar1 = vuzp1q_f64(v2, v3);
            let ai1 = vuzp2q_f64(v2, v3);
            let br0 = vuzp1q_f64(w0, w1);
            let bi0 = vuzp2q_f64(w0, w1);
            let br1 = vuzp1q_f64(w2, w3);
            let bi1 = vuzp2q_f64(w2, w3);
            re0 = vfmaq_f64(vfmaq_f64(re0, ar0, br0), ar1, br1);
            re1 = vfmaq_f64(vfmaq_f64(re1, ai0, bi0), ai1, bi1);
            im0 = vfmaq_f64(vfmaq_f64(im0, ar0, bi0), ar1, bi1);
            im1 = vfmaq_f64(vfmaq_f64(im1, ai0, br0), ai1, br1);
            i += 4;
        }
        let mut sum = C::new(
            vaddvq_f64(vaddq_f64(re0, re1)),
            vaddvq_f64(vsubq_f64(im0, im1)),
        );
        while i < n {
            sum += a[i].conj() * b[i];
            i += 1;
        }
        sum
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env_usize("NPROBE_N", 2380);
    let rows = env_usize("NPROBE_ROWS", 8);
    let reps = env_usize("NPROBE_REPS", 400);
    let mut seed = 0x5eed_1234_u64;
    let basis: Vec<Vec<C>> = (0..rows)
        .map(|_| {
            (0..n)
                .map(|_| C::new(lcg(&mut seed), lcg(&mut seed)))
                .collect()
        })
        .collect();
    let scale = C::new(0.37, -0.91);
    let mut w = vec![C::new(0.0, 0.0); n];
    println!("n={n} rows={rows} reps={reps}");

    let started = Instant::now();
    for _ in 0..reps {
        for row in &basis {
            let _ = black_box(scalar_dot(black_box(row), black_box(&w)));
        }
    }
    let scalar_dot_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;

    let started = Instant::now();
    for _ in 0..reps {
        for row in &basis {
            let _ = black_box(scalar_axpy(black_box(&mut w), black_box(row), scale));
        }
    }
    let scalar_axpy_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;
    println!("scalar: dot {scalar_dot_us:.3} us/call, axpy {scalar_axpy_us:.3} us/call");

    #[cfg(target_arch = "aarch64")]
    {
        let started = Instant::now();
        for _ in 0..reps {
            for row in &basis {
                let _ = black_box(unsafe { neon::dot(black_box(row), black_box(&w)) });
            }
        }
        let neon_dot_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;

        let started = Instant::now();
        for _ in 0..reps {
            for row in &basis {
                unsafe { neon::axpy(black_box(&mut w), black_box(row), scale) };
            }
        }
        let neon_axpy_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;
        println!("neon:   dot {neon_dot_us:.3} us/call, axpy {neon_axpy_us:.3} us/call");
        println!(
            "ratio:  dot {:.2}x, axpy {:.2}x",
            scalar_dot_us / neon_dot_us,
            scalar_axpy_us / neon_axpy_us
        );

        // Same arithmetic up to rounding: the vector paths reorder the sum.
        let mut a1 = basis[0].clone();
        let mut a2 = basis[0].clone();
        scalar_axpy(&mut a1, &basis[1], scale);
        unsafe { neon::axpy(&mut a2, &basis[1], scale) };
        let axpy_gap = a1
            .iter()
            .zip(&a2)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max);
        let d1 = scalar_dot(&basis[0], &basis[1]);
        let d2 = unsafe { neon::dot(&basis[0], &basis[1]) };
        println!(
            "agreement: axpy max |diff| {axpy_gap:.3e}, dot rel {:.3e}",
            (d1 - d2).norm() / d1.norm()
        );
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            let started = Instant::now();
            for _ in 0..reps {
                for row in &basis {
                    let _ = black_box(unsafe { avx2::dot(black_box(row), black_box(&w)) });
                }
            }
            let avx2_dot_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;

            w.fill(C::new(0.0, 0.0));
            let started = Instant::now();
            for _ in 0..reps {
                for row in &basis {
                    unsafe { avx2::axpy(black_box(&mut w), black_box(row), scale) };
                }
            }
            let avx2_axpy_us = started.elapsed().as_secs_f64() * 1e6 / (reps * rows) as f64;
            println!("avx2:   dot {avx2_dot_us:.3} us/call, axpy {avx2_axpy_us:.3} us/call");
            println!(
                "ratio:  dot {:.2}x, axpy {:.2}x",
                scalar_dot_us / avx2_dot_us,
                scalar_axpy_us / avx2_axpy_us
            );
            let d1 = scalar_dot(&basis[0], &basis[1]);
            let d2 = unsafe { avx2::dot(&basis[0], &basis[1]) };
            let mut a1 = basis[0].clone();
            let mut a2 = basis[0].clone();
            scalar_axpy(&mut a1, &basis[1], scale);
            unsafe { avx2::axpy(&mut a2, &basis[1], scale) };
            let axpy_gap = a1
                .iter()
                .zip(&a2)
                .map(|(x, y)| (x - y).norm())
                .fold(0.0_f64, f64::max);
            println!(
                "agreement: axpy max |diff| {axpy_gap:.3e}, dot rel {:.3e}",
                (d1 - d2).norm() / d1.norm()
            );
        } else {
            println!("avx2: not built (no AVX2/FMA on this CPU)");
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    println!("simd: not built (this probe targets aarch64 and x86_64)");
}
