//! Device selection for the SIRK solver.
//!
//! The crate is CPU-only by default. Building with `--features cuda` enables
//! GPU offload; in that case `best_device` prefers CUDA device 0 and falls back
//! to the CPU when no GPU is present at runtime.

use candle_core::Device;

/// Return the best available compute device.
///
/// With the `cuda` feature enabled this prefers CUDA device 0 (falling back to
/// the CPU if CUDA is unavailable at runtime). Without the feature it always
/// returns the CPU device.
pub fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}
