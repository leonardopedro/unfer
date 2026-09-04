//! Device selection for the SIRK solver.
//!
//! The crate is CPU-only by default. Building with `--features cuda` enables
//! GPU offload; in that case `best_device` prefers CUDA device 0 and falls back
//! to the CPU when no GPU is present at runtime.
//!
//! T2.2 of the GPU_FEDERATION_PLAN: the probe emits **structured triage**
//! lines on stderr that the agent loop can parse — `UK-GPU-<CODE> → <fix>` —
//! mapping candle's raw init errors to the documented failure modes
//! (AGENTS.md §5: `ARCH_MISMATCH` = libcublas/libcuda version conflict with
//! the active GPU; `LD_LIBRARY_PATH` must point at the toolkit matching the
//! driver, e.g. `/lib/x86_64-linux-gnu` for CUDA 12.2 coexistence).

use candle_core::Device;

/// Machine-readable CUDA probe failure, mapped from the documented failure
/// modes. Rendered as `UK-GPU-<CODE>` on stderr with the remediation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTriage {
    /// No CUDA-capable device / no driver.
    NoDevice,
    /// `libcublas`/`libcuda` version conflict with the active GPU
    /// (`ARCH_MISMATCH`).
    ArchMismatch,
    /// A CUDA shared library could not be loaded.
    LibraryMissing,
    /// CUDA ran out of memory.
    OutOfMemory,
    /// Some other candle init failure.
    Other,
}

impl GpuTriage {
    /// The documented remediation for this failure mode.
    pub fn fix(self) -> &'static str {
        match self {
            GpuTriage::NoDevice => "install the NVIDIA driver and confirm `nvidia-smi` lists a GPU",
            GpuTriage::ArchMismatch => {
                "libcublas/libcuda version conflict with the active GPU — \
                 point LD_LIBRARY_PATH at the CUDA toolkit matching the driver \
                 (e.g. /lib/x86_64-linux-gnu for CUDA 12.2 coexistence)"
            }
            GpuTriage::LibraryMissing => {
                "a CUDA shared library is not loadable — add the toolkit lib \
                 dir to LD_LIBRARY_PATH (e.g. /usr/local/cuda/lib64)"
            }
            GpuTriage::OutOfMemory => {
                "CUDA ran out of memory — reduce the basis size or the Krylov \
                 window"
            }
            GpuTriage::Other => {
                "see the candle error below (RUST_LOG=candle_core=debug for \
                 kernel-dispatch confirmation)"
            }
        }
    }

    /// Parseable stderr line: `UK-GPU-<CODE> → <fix> (<detail>)`.
    pub fn line(self, detail: &str) -> String {
        let code = match self {
            GpuTriage::NoDevice => "NO_DEVICE",
            GpuTriage::ArchMismatch => "ARCH_MISMATCH",
            GpuTriage::LibraryMissing => "LIBRARY_MISSING",
            GpuTriage::OutOfMemory => "OUT_OF_MEMORY",
            GpuTriage::Other => "OTHER",
        };
        format!("UK-GPU-{code} → {} (candle: {detail})", self.fix())
    }

    /// Map a candle CUDA init error to a triage code by its message.
    pub fn from_candle_error(e: &candle_core::Error) -> GpuTriage {
        let msg = e.to_string().to_ascii_lowercase();
        if msg.contains("arch_mismatch") {
            GpuTriage::ArchMismatch
        } else if msg.contains("cannot open shared object")
            || msg.contains("libcuda")
            || msg.contains("libcublas")
            || msg.contains("libcudart")
            || msg.contains("libnvidia")
        {
            GpuTriage::LibraryMissing
        } else if msg.contains("out of memory")
            || msg.contains("out-of-memory")
            || msg.contains("too much shared memory")
        {
            GpuTriage::OutOfMemory
        } else if msg.contains("no cuda")
            || msg.contains("cuda driver")
            || msg.contains("not available")
            || msg.contains("no device")
        {
            GpuTriage::NoDevice
        } else {
            GpuTriage::Other
        }
    }
}

/// Result of probing a CUDA device: the device (when usable) plus the triage
/// for failures, so callers can surface structured diagnostics.
#[derive(Debug)]
pub struct CudaProbe {
    /// The usable CUDA device, when present.
    pub device: Option<Device>,
    /// The failure triage, when the probe failed.
    pub triage: Option<GpuTriage>,
    /// The raw candle error message (empty on success).
    pub detail: String,
}

impl CudaProbe {
    /// Emit the parseable `UK-GPU-<CODE>` line to stderr when the probe
    /// failed (no-op on success).
    pub fn emit_triage(&self) {
        if let (Some(t), false) = (self.triage, self.device.is_some()) {
            eprintln!("{}", t.line(&self.detail));
        }
    }
}

/// Probe CUDA device `index` (0-based), returning the device and a
/// machine-readable triage on failure.
pub fn probe_cuda(index: usize) -> CudaProbe {
    match Device::new_cuda(index) {
        Ok(device) => CudaProbe {
            device: Some(device),
            triage: None,
            detail: String::new(),
        },
        Err(e) => {
            let triage = GpuTriage::from_candle_error(&e);
            CudaProbe {
                device: None,
                triage: Some(triage),
                detail: e.to_string(),
            }
        }
    }
}

/// Return the best available compute device.
///
/// With the `cuda` feature enabled this prefers CUDA device 0 (falling back to
/// the CPU if CUDA is unavailable at runtime); the failure emits a structured
/// `UK-GPU-<CODE>` triage line on stderr. Without the feature it always
/// returns the CPU device.
pub fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        let probe = probe_cuda(0);
        probe.emit_triage();
        probe.device.unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triage_maps_documented_failure_modes() {
        let cases: &[(&str, GpuTriage)] = &[
            (
                "CUDA error: 709 an illegal memory access was encountered",
                GpuTriage::Other,
            ),
            (
                "cuInit error: ARCH_MISMATCH (libcublas vs device arch)",
                GpuTriage::ArchMismatch,
            ),
            (
                "cannot open shared object file: libcuda.so.1",
                GpuTriage::LibraryMissing,
            ),
            (
                "CUDA error: 2 out of memory (allocating 65536 bytes)",
                GpuTriage::OutOfMemory,
            ),
            ("no CUDA-capable device is detected", GpuTriage::NoDevice),
        ];
        for (msg, expected) in cases {
            let e = candle_core::Error::Cuda(Box::new(std::io::Error::other(msg.to_string())));
            assert_eq!(
                GpuTriage::from_candle_error(&e),
                *expected,
                "message: {msg}"
            );
        }
    }

    #[test]
    fn triage_lines_are_parseable() {
        let line = GpuTriage::ArchMismatch.line("cuInit ARCH_MISMATCH");
        assert!(line.starts_with("UK-GPU-ARCH_MISMATCH → "), "line: {line}");
        let fix = GpuTriage::ArchMismatch.fix();
        assert!(fix.contains("LD_LIBRARY_PATH"), "fix: {fix}");
    }
}
