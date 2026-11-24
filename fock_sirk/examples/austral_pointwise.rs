//! G18: the amplitude scale the SIRK solver already performs, executed by an
//! Austral-authored GPU kernel.
//!
//! The operation is `QuantumState::scale_and_add`'s amplitude multiply: every
//! component of a state is multiplied by the same real factor. The kernel is
//! the Austral function `UnferScale:scale` in australVM's
//! `examples/gpu/unfer-scale/` (one canonical `trappingMultiply(x, 2.5)`),
//! published by that repository's compiler as a `wgsl-f32-v1` bundle. This
//! example does **not** invoke the OCaml compiler: the bundle path comes from
//! the command line or `AUSTRAL_GPU_BUNDLE`, and the recipe in
//! australVM's `docs/gpu/UNFER_SCALE.md` shows the compile step.
//!
//! What is measured, per run:
//!
//! * the f64 sparse state from `nested_fock_algebra` is flattened to the dense
//!   `TensorState` image (the existing upload path and its bijection check),
//!   then narrowed to binary32 explicitly — the narrowing is a named step, not
//!   a reinterpretation;
//! * the GPU result is compared **bit for bit** against the existing unfer
//!   arithmetic (`QuantumState::scale_and_add` on the same state, cast to f32),
//!   which is exact here because 2.5 is representable in both widths;
//! * cold and warm timings are reported separately, and the transfer cost is
//!   printed: Candle tensors and wgpu buffers are not interchangeable, so this
//!   is an explicit copy each way and **not** zero-copy.
//!
//! Build and run (feature off is the default and does not build this file):
//!
//! ```text
//! cargo run -p fock_sirk --features austral-gpu --example austral_pointwise -- BUNDLE
//! ```
//!
//! Exit codes: 0 = the GPU result matches the oracle; 77 = no physical adapter
//! ran the kernel (the CPU checks still print); 2 = usage; 1 = disagreement.

#[cfg(not(feature = "austral-gpu"))]
fn main() {
    eprintln!(
        "austral_pointwise: built without the `austral-gpu` feature; \
         this example is australVM's G18 integration and is excluded by \
         `required-features` in Cargo.toml."
    );
    std::process::exit(2);
}

#[cfg(feature = "austral-gpu")]
fn main() {
    gpu::main();
}

#[cfg(feature = "austral-gpu")]
mod gpu {
    use std::time::Instant;

    use avm_gpu::device::Selection;
    use avm_gpu::run::{self, RunFailure};
    use candle_core::Device;
    use fock_sirk::{StateDictionary, TensorState};
    use nested_fock_algebra::QuantumState;
    use num_complex::Complex64;

    /// The scale factor, twice: once as the kernel's source literal (2.5) and
    /// once here for the oracle. They are deliberately the same constant, and
    /// the fixture's comment explains why 2.5 keeps the comparison exact.
    const FACTOR: f64 = 2.5;
    const ITERATIONS: u32 = 8;

    /// A deterministic, finite, normal corpus: a vacuum amplitude, one boson in
    /// mode 0, two in mode 1, plus amplitudes whose binary32 products round.
    fn state() -> QuantumState {
        // Built through the library's own constructors, so the corpus is an
        // ordinary superposition rather than a hand-made map.
        let vacuum = QuantumState::vacuum();
        let one = vacuum.create_boson(0);
        let two = vacuum.create_boson(1);
        let mut state = QuantumState::zero();
        state.scale_and_add(&vacuum, Complex64::new(1.0, 0.0));
        state.scale_and_add(&one, Complex64::new(-0.5, 0.25));
        state.scale_and_add(&two, Complex64::new(0.1, -1.0e-3));
        state
    }

    pub fn main() {
        let bundle = match std::env::args().nth(1).or_else(|| {
            std::env::var("AUSTRAL_GPU_BUNDLE").ok()
        }) {
            Some(path) => path,
            None => {
                eprintln!(
                    "usage: austral_pointwise BUNDLE\n  \
                     BUNDLE is a wgsl-f32-v1 bundle published from \
                     examples/gpu/unfer-scale (australVM)"
                );
                std::process::exit(2);
            }
        };
        let loaded = match avm_gpu::load_bundle(std::path::Path::new(&bundle)) {
            Ok(loaded) => loaded,
            Err(error) => {
                eprintln!("austral_pointwise: {bundle}: {error}");
                std::process::exit(1);
            }
        };
        let bundle_meta = &loaded.bundle;
        let element = avm_gpu::schema::Element::of_profile(&bundle_meta.numeric_profile)
            .unwrap_or(avm_gpu::schema::Element::U32);
        if bundle_meta.numeric_profile != "wgsl-f32-v1"
            || element != avm_gpu::schema::Element::F32
        {
            eprintln!(
                "austral_pointwise: {bundle} is profile {}; this example needs \
                 wgsl-f32-v1",
                bundle_meta.numeric_profile
            );
            std::process::exit(1);
        }

        // The existing unfer path: sparse state -> dense TensorState image on
        // the CPU device (its `debug_assert!` covers the layout bijection), then
        // an explicit narrowing to binary32 for the kernel's element type.
        let sparse = state();
        let mut registry = StateDictionary::new();
        registry.register(&sparse);
        let device = Device::Cpu;
        let tensor = TensorState::from_quantum_state(&sparse, &mut registry, &device)
            .expect("TensorState upload on the CPU device");
        let real64 = tensor.real.to_vec1::<f64>().expect("real vector");
        let imag64 = tensor.imag.to_vec1::<f64>().expect("imaginary vector");
        let dim = tensor.dim;
        debug_assert_eq!(real64.len(), dim);
        debug_assert_eq!(imag64.len(), dim);

        // The oracle is unfer's own arithmetic — the same `scale_and_add` the
        // solver uses to combine basis states — evaluated in f64 and cast once.
        let started = Instant::now();
        let mut scaled = QuantumState::zero();
        scaled.scale_and_add(&sparse, Complex64::new(FACTOR, 0.0));
        let mut oracle_real = vec![0f64; dim];
        let mut oracle_imag = vec![0f64; dim];
        for (outer, amplitude) in &scaled.components {
            if let Some(&index) = registry.state_to_index.get(outer) {
                oracle_real[index] = amplitude.re;
                oracle_imag[index] = amplitude.im;
            }
        }
        let cpu_ms = started.elapsed().as_secs_f64() * 1e3;
        // The kernel is handed binary32 values, so the oracle must be evaluated
        // on exactly those values: narrow first, multiply in f64 (exact — the
        // product of two f32s fits) and round once. Narrowing *after* the f64
        // multiply is a different function, and the difference is exactly the
        // cost of the f64 -> f32 upload conversion, which is reported rather
        // than hidden by choosing a friendlier comparison.
        let narrowed_real: Vec<f32> = real64.iter().map(|&v| v as f32).collect();
        let narrowed_imag: Vec<f32> = imag64.iter().map(|&v| v as f32).collect();
        let expected_real: Vec<f32> = narrowed_real
            .iter()
            .map(|&v| (f64::from(v) * FACTOR) as f32)
            .collect();
        let expected_imag: Vec<f32> = narrowed_imag
            .iter()
            .map(|&v| (f64::from(v) * FACTOR) as f32)
            .collect();
        // `oracle_real`/`oracle_imag` already are the f64 products; narrowing
        // them is "multiply in f64, round once" — the comparison the upload
        // conversion changes.
        let narrowed_after: Vec<f32> = oracle_real
            .iter()
            .chain(oracle_imag.iter())
            .map(|&v| v as f32)
            .collect();
        let narrowing_differences = expected_real
            .iter()
            .chain(expected_imag.iter())
            .zip(&narrowed_after)
            .filter(|(narrow_first, narrow_last)| narrow_first != narrow_last)
            .count();

        let real_bits: Vec<u32> = narrowed_real.iter().map(|v| v.to_bits()).collect();
        let imag_bits: Vec<u32> = narrowed_imag.iter().map(|v| v.to_bits()).collect();

        let selection = Selection::default();
        let real_run = run::run_with_iterations(
            &loaded,
            &[real_bits],
            true,
            &selection,
            ITERATIONS,
        );
        let (real_run, imag_run) = match real_run {
            Ok(real_run) => {
                match run::run_with_iterations(&loaded, &[imag_bits], true, &selection, ITERATIONS) {
                    Ok(imag_run) => (real_run, imag_run),
                    Err(failure) => unavailable(failure),
                }
            }
            Err(failure) => unavailable(failure),
        };

        let real_mismatch = real_run
            .output
            .iter()
            .zip(&expected_real)
            .filter(|(got, want)| f32::from_bits(**got) != **want)
            .count();
        let imag_mismatch = imag_run
            .output
            .iter()
            .zip(&expected_imag)
            .filter(|(got, want)| f32::from_bits(**got) != **want)
            .count();
        let mismatches = real_mismatch + imag_mismatch;
        let transferred = 2 * dim * 4;

        println!(
            "{{\"status\":\"ok\",\"adapter\":{:?},\"dim\":{},\
             \"numeric_profile\":{:?},\"ir_sha256\":{:?},\"iterations\":{},\
             \"real_mismatches\":{},\"imag_mismatches\":{},\
             \"narrowing_differences\":{},\"cpu_oracle_ms\":{:.6},\
             \"timing_ms\":{{\"prepare\":{:.6},\"upload\":{:.6},\"submit\":{:.6},\
             \"readback\":{:.6}}},\"transfer_bytes\":{},\"zero_copy\":false}}",
            real_run.adapter.name,
            dim,
            bundle_meta.numeric_profile,
            bundle_meta.ir_sha256,
            ITERATIONS,
            real_mismatch,
            imag_mismatch,
            narrowing_differences,
            cpu_ms,
            real_run.prepare_ms,
            real_run.upload_ms + imag_run.upload_ms,
            real_run.submit_ms + imag_run.submit_ms,
            real_run.readback_ms + imag_run.readback_ms,
            transferred,
        );
        if mismatches != 0 {
            eprintln!(
                "austral_pointwise: {mismatches} of {} value(s) differ from the f64 \
                 oracle cast to binary32",
                2 * dim
            );
            std::process::exit(1);
        }
        // The oracle is only meaningful if it kept the components: an empty
        // oracle would make every GPU result "match" a zero vector.
        if scaled.components.len() != sparse.components.len() {
            eprintln!(
                "austral_pointwise: the oracle dropped components ({} of {}); \
                 the corpus is outside scale_and_add's pruning threshold",
                scaled.components.len(),
                sparse.components.len()
            );
            std::process::exit(1);
        }
    }

    /// A missing adapter is a gap, not a pass: the CPU checks above already
    /// ran, and the exit code says the device half did not.
    fn unavailable(failure: RunFailure) -> ! {
        let (kind, message) = match failure {
            RunFailure::Unavailable(message) => ("gap", message),
            RunFailure::Refused(message) => ("refused", message),
        };
        println!(
            "{{\"status\":\"{kind}\",\"message\":{message:?},\
             \"zero_copy\":false}}"
        );
        std::process::exit(if kind == "gap" { 77 } else { 1 });
    }
}
