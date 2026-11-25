//! Explicit-NEON complex `axpy`/`dot` against the committed scalar kernels.
//!
//! The rational-Krylov loop spends its non-`gemv` time in `coords_inner_product`
//! and `coords_axpy` (`joint_sirk.rs`), both scalar `Complex<f64>` loops. This
//! probe builds the same two kernels with NEON intrinsics — deinterleaving the
//! array-of-structs pairs with `vuzp1q/vuzp2q`, four independent accumulators
//! for the inner product to break the dependent-add chain — and times them
//! against the committed shape (zero-entry skip plus the per-element
//! non-finiteness test).
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
//! The `dot` row is deliberately a range: the *scalar reference* is not
//! reliably scalar across builds -- LLVM vectorises this loop in some
//! compilation contexts (`1.44 us`) and not in others (`2.81 us`) -- so the
//! microbenchmark cannot settle that kernel. The end-to-end measurement can,
//! because in the solver the scalar single accumulator is latency-bound: patched
//! in and timed at `C = 26` the NEON path took `dot` from `1.65 s` to `0.44 s`
//! (`3.75x`) and the solve from `9.760 s` to `8.528 s` (`12.6 %`). That, not
//! the table above, is the number §5.12 quotes. The path is **not** adopted: it changes the solve's last
//! bits (`probability` differs at `1.6e-16`, `917` vs `915` iterations) and so
//! would need the frozen sweep record regenerated, and it is `aarch64`-only, so
//! x86 hosts would keep the scalar path and produce different digits. It exists
//! here to make that number re-measurable.
//!
//! On a non-`aarch64` host only the scalar timings are reported.

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
            a[i] = a[i] + scale * b[i];
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
    #[cfg(not(target_arch = "aarch64"))]
    println!("neon: not built (this probe targets aarch64)");
}
