//! CUDA probe with structured triage (GPU_FEDERATION_PLAN T2.2).
//!
//! Probes CUDA device 0 and prints the `UK-GPU-<CODE>` triage lines the
//! agent loop can parse. The triage codes map candle's raw init errors to
//! the documented failure modes (AGENTS.md §5): `ARCH_MISMATCH` =
//! libcublas/libcuda version conflict with the active GPU; `LD_LIBRARY_PATH`
//! must point at the toolkit matching the driver.

use fock_sirk::device::GpuTriage;
#[cfg(feature = "cuda")]
use fock_sirk::device::probe_cuda;

#[cfg(feature = "cuda")]
fn main() {
    println!("Checking CUDA availability...");
    let probe = probe_cuda(0);
    match &probe.device {
        Some(device) => {
            println!("SUCCESS: Found CUDA device: {device:?}");
        }
        None => {
            let triage = probe.triage.unwrap_or(GpuTriage::Other);
            println!("FAILURE: {triage:?}");
            probe.emit_triage();
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let _ = GpuTriage::NoDevice;
    println!("Built without the `cuda` feature; rebuild with `--features cuda` to probe a GPU.");
}
