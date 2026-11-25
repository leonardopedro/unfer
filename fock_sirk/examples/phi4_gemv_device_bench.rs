//! Is a **device-resident** projected `gemv` faster than the host's?
//!
//! The rational-Krylov solve applies the finite-window operator as one dense
//! `gemv` per step (`FiniteWindowAction::image_coords`). At `C = 32` that is a
//! `4845 x 4845` complex matrix — `375.6 MB` — and the host spends `20.9 ms`
//! per step streaming it (`docs/VERIFY_PHI4_MOMENTUM.md` §5.12). This probe
//! times the identical product on a CUDA device, so the comparison quoted in
//! §5.12 is re-measurable instead of inferred from bandwidth.
//!
//! ```text
//! # CPU leg only (always works):
//! PROBE_CUTOFF=32 cargo run -q -p fock_sirk --release \
//!     --example phi4_gemv_device_bench
//!
//! # Both legs, on a host with a driver + toolkit:
//! PROBE_CUTOFF=32 cargo run -q -p fock_sirk --release --features cuda \
//!     --example phi4_gemv_device_bench
//! ```
//!
//! `PROBE_REPS` (default 5) sets the repeat count; the reported number is the
//! mean per step. On a host without a CUDA device the probe prints the
//! `UK-GPU-<CODE> → <fix>` triage line and stops after the CPU leg — it does
//! not estimate the device time.
//!
//! Two honest caveats about the device leg, both visible in its label:
//!
//! * the complex matrix is uploaded as its real `2n x 2n` block form and the
//!   `gemv` runs in `f32` (candle's CUDA path), so the device performs four
//!   times the real multiplies of the complex product — the comparison is
//!   conservative *for the device*;
//! * one `gemv` per step is timed, not the Krylov loop, so the per-step
//!   host↔device transfer of the `4845`-complex vector is **not** included;
//!   at `77.5 KB` each way it is `~10 us` at PCIe rates, i.e. negligible
//!   beside both legs, but it is not measured here.

use candle_core::Tensor;
use fock_sirk::device::probe_cuda;
use nalgebra::DVector;
use nested_fock_algebra::exact_inner::JointOuterBasis;
use nested_fock_algebra::phi4_momentum::ScalarMomCont;
use num_complex::Complex64 as C;
use std::time::Instant;

mod common;

use common::window_reference::WindowOperator;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(default)
}

fn main() {
    let cutoff: u32 = env_usize("PROBE_CUTOFF", 32) as u32;
    let reps = env_usize("PROBE_REPS", 5).max(1);

    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let window: std::collections::BTreeSet<JointOuterBasis> = ScalarMomCont::window_labels(cutoff);
    let operator = WindowOperator::build(&h, &window);
    let n = operator.labels();
    let matrix = operator.matrix();
    assert_eq!(matrix.nrows(), n);

    // A dense state over the window, the shape the Krylov loop hands to `gemv`.
    let state = DVector::from_fn(n, |i, _| C::new(1.0 / (1.5 + i as f64), 0.0));
    let matrix_mb = (n * n * std::mem::size_of::<C>()) as f64 / 1e6;
    println!("cutoff={cutoff} labels={n} matrix_mb={matrix_mb:.1} reps={reps}");

    // Warm the cache, then time: a cold first product would flatter the device.
    let _ = matrix * &state;
    let started = Instant::now();
    let mut checksum = C::new(0.0, 0.0);
    for _ in 0..reps {
        let image = std::hint::black_box(matrix * &state);
        checksum = image[0];
    }
    let cpu_ms = started.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "cpu gemv: {cpu_ms:.3} ms/step (checksum {:.6e})",
        checksum.norm()
    );

    let probe = probe_cuda(0);
    probe.emit_triage();
    let Some(device) = probe.device else {
        println!(
            "device leg not run: no usable CUDA device. Rebuild with `--features cuda` \
             on a host whose driver lists a GPU (`nvidia-smi`); the triage line above \
             names the fix when one is present but unusable."
        );
        return;
    };
    println!("device: {device:?}");

    // Real block form [[Re, -Im], [Im, Re]], f32, column-major like nalgebra.
    let width = 2 * n;
    let mut real = vec![0.0_f32; width * width];
    for j in 0..n {
        for i in 0..n {
            let z = matrix[(i, j)];
            real[j * width + i] = z.re as f32;
            real[(j + n) * width + i] = z.im as f32;
            real[j * width + i + n] = -(z.im as f32);
            real[(j + n) * width + i + n] = z.re as f32;
        }
    }
    let mut rhs = vec![0.0_f32; width];
    for i in 0..n {
        rhs[i] = state[i].re as f32;
        rhs[i + n] = state[i].im as f32;
    }

    let uploaded = Tensor::from_vec(real, (width, width), &device).expect("upload matrix");
    let x = Tensor::from_vec(rhs, (width, 1), &device).expect("upload vector");
    // First product builds the kernel; the timed loop must not include it.
    let _ = uploaded.matmul(&x).expect("warm device gemv");

    let started = Instant::now();
    let mut out_sum = 0.0_f32;
    for _ in 0..reps {
        let out = uploaded.matmul(&x).expect("device gemv");
        // `to_vec1` synchronises, so the clock stops after the kernel lands.
        let host = out.to_vec1::<f32>().expect("read back");
        out_sum += host[0];
    }
    let device_ms = started.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "device gemv: {device_ms:.3} ms/step (f32 real-block {width}x{width}, 4x the real \
         multiplies of the complex product; checksum {out_sum:.6e})"
    );
    println!(
        "ratio: device {:.2}x the host per step (host {cpu_ms:.3} ms)",
        cpu_ms / device_ms
    );
}
