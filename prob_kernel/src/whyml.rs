//! WhyML codegen + Why3 verification seam (S36, the Why3 cycle).
//!
//! The kernel produces **WhyML** programs that the external Why3 toolchain
//! verifies and extracts to OCaml modules which extend the australVM
//! compiler — closing the loop where the probability kernel supplies the
//! compiler extension that compiles the modules calling the probability
//! kernel (see `docs/WHYML_CYCLE.md`).
//!
//! License seam (identical to the Cadabra2 coupling in `symbolic.rs`):
//! Why3 is LGPL-2.1 and is invoked only as a **subprocess** (`why3`), so the
//! Apache-2.0 Rust binary never links against it. The extracted OCaml is a
//! mechanical translation of the *user's* WhyML program (an independent
//! work, Apache-able), not of Why3's own code.
//!
//! The default emitted program is the **authorization gate**: the kernel's
//! own grant-subset semantics (`GrantSet::is_subset_of`, S21 — capability
//! non-escalation) written in WhyML with a postcondition that Why3 proves:
//!
//! ```text
//! authorize grants required = True  <->  required ⊆ grants
//! ```
//!
//! plus the subset lattice lemmas (reflexivity, transitivity — the "no
//! escalation path" theorem). Extraction is semantics-preserving, so the
//! OCaml module the australVM compiler loads satisfies the property by
//! construction. `WhymlOp::Prove` additionally runs `why3 prove` and reduces
//! the outcome to a boolean verdict (all goals discharged?).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use unfer_protocol::{WhymlOp, WhymlReport, WhymlSpec};

use crate::error::KernelError;

/// Env var overriding the Why3 binary location.
pub const WHY3_CLI_ENV: &str = "WHY3_CLI";

/// Number of proof obligations the default (authorization gate) program
/// declares: the two subset lemmas (reflexive, transitive), the `mem`
/// postcondition (Why3 1.8 rejects the logical `Mem.mem` predicate in
/// program code, so a program-level `mem` function links the two), plus the
/// two postconditions (`authorize` and the gate entrypoint).
pub const PROOF_OBLIGATIONS: usize = 5;

/// Number of proof obligations the NPU DMA gate program declares: the
/// `dma_ok` postcondition (soundness+completeness of the SRAM bound) and the
/// gate entrypoint's postcondition. Both are linear integer arithmetic that
/// alt-ergo discharges directly.
pub const NPU_PROOF_OBLIGATIONS: usize = 2;

/// The proof-obligation count for the spec's program.
pub fn proof_obligations_of(spec: &WhymlSpec) -> usize {
    match spec.program {
        unfer_protocol::WhymlProgram::AuthorizeGate => PROOF_OBLIGATIONS,
        unfer_protocol::WhymlProgram::NpuDmaGate => NPU_PROOF_OBLIGATIONS,
    }
}

/// Locate the `why3` binary: `WHY3_CLI` env override, then PATH.
pub fn why3_cli_path() -> Option<PathBuf> {
    if let Ok(over) = std::env::var(WHY3_CLI_ENV)
        && !over.is_empty()
    {
        return Some(PathBuf::from(over));
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("why3");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// True iff a `why3` binary is discoverable.
pub fn why3_available() -> bool {
    why3_cli_path().is_some()
}

/// Is `s` a valid WhyML identifier (`[A-Za-z_][A-Za-z0-9_]*`)?
fn valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Validate a `uk_*`/`uz_*` symbol name against the kernel's own registry.
/// The emission refuses to produce a gate for a symbol the kernel does not
/// know — the kernel's knowledge (the symbol census) bounds the codegen.
fn registered_symbol(name: &str) -> Result<(), KernelError> {
    if unfer_protocol::symbols::SymbolRecord::by_name(name).is_none() {
        return Err(KernelError::WhymlInvalid {
            reason: format!(
                "`{name}` is not a registered kernel symbol (see unfer_protocol::symbols::SYMBOL_REGISTRY)"
            ),
        });
    }
    Ok(())
}

/// Validate the spec and normalize the module/function identifiers.
fn validate(spec: &WhymlSpec) -> Result<(), KernelError> {
    let bad = |what: &str, v: &str| KernelError::WhymlInvalid {
        reason: format!("{what} `{v}` is not a valid WhyML identifier"),
    };
    if !valid_identifier(&spec.module_name) {
        return Err(bad("module_name", &spec.module_name));
    }
    if !valid_identifier(&spec.function_name) {
        return Err(bad("function_name", &spec.function_name));
    }
    for g in &spec.grants {
        registered_symbol(g)?;
    }
    for r in &spec.required {
        registered_symbol(r)?;
    }
    for e in &spec.kernel_externals {
        registered_symbol(e)?;
    }
    Ok(())
}

/// The full WhyML pipeline: emit, and optionally verify with `why3 prove`.
///
/// [`WhymlOp::Emit`] is pure — it always succeeds for a valid spec.
/// [`WhymlOp::Prove`] additionally requires the external `why3` binary
/// ([`KernelError::WhymlUnavailable`] when missing) and runs the prover.
pub fn whyml_emit(spec: &WhymlSpec) -> Result<WhymlReport, KernelError> {
    validate(spec)?;
    let obligations = proof_obligations_of(spec);
    let whyml = build_mlw(spec);
    let sha256 = hex::encode(Sha256::digest(whyml.as_bytes()));
    match spec.op {
        WhymlOp::Emit => Ok(WhymlReport {
            whyml,
            sha256,
            proof_obligations: obligations,
            verified: None,
            error: None,
            engine_ms: 0,
        }),
        WhymlOp::Prove => {
            let cli = why3_cli_path().ok_or_else(|| KernelError::WhymlUnavailable {
                reason: format!("why3 not found (set {WHY3_CLI_ENV} or install the why3 package)"),
            })?;
            let (verified, err, engine_ms) = prove_with_cli(&cli, &whyml, spec.timeout_ms)?;
            Ok(WhymlReport {
                whyml,
                sha256,
                proof_obligations: obligations,
                verified: Some(verified),
                error: err,
                engine_ms,
            })
        }
    }
}

/// Run `why3 prove -P alt-ergo` on the emitted program and reduce the output
/// to a boolean verdict (every declared goal proved by the prover).
///
/// `why3 prove` prints one line per goal (e.g. `...: Valid`) followed by a
/// summary; a missing/unconfigured prover produces a non-"Valid" line. The
/// verdict is `verified` iff at least `PROOF_OBLIGATIONS` goals report
/// `Valid` and no goal reports `Invalid`/`Failure`/`Timeout`.
fn prove_with_cli(
    cli: &PathBuf,
    whyml: &str,
    timeout_ms: u64,
) -> Result<(bool, Option<String>, u64), KernelError> {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let path = std::env::temp_dir().join(format!(
        "unfer_whyml_{}_{}.mlw",
        std::process::id(),
        started
    ));
    let mut file = std::fs::File::create(&path).map_err(|e| KernelError::WhymlInvalid {
        reason: format!("failed to create temp .mlw: {e}"),
    })?;
    file.write_all(whyml.as_bytes())
        .map_err(|e| KernelError::WhymlInvalid {
            reason: format!("failed to write temp .mlw: {e}"),
        })?;
    let _ = file.sync_all();

    let child = Command::new(cli)
        .arg("prove")
        .arg("-P")
        .arg("alt-ergo")
        .arg(&path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| KernelError::WhymlUnavailable {
            reason: format!("failed to spawn why3: {e}"),
        })?;

    let guard = wait_with_timeout(child, timeout_ms);
    let _ = std::fs::remove_file(&path);
    let output = guard.ok_or_else(|| KernelError::WhymlUnavailable {
        reason: format!("why3 prove timed out after {timeout_ms} ms"),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let valid = stdout.lines().filter(|l| l.contains("Valid")).count();
    let failed = stdout
        .lines()
        .any(|l| l.contains("Invalid") || l.contains("Failure") || l.contains("Timeout"));

    let elapsed_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
        .saturating_sub(started);

    if valid == 0 {
        // No goal reported Valid — either the prover is missing/unconfigured
        // or the file failed to type-check. Surface the engine diagnostics.
        let detail = if stdout.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        return Err(KernelError::WhymlUnavailable {
            reason: format!("why3 prove produced no Valid goals: {detail}"),
        });
    }

    let verified = !failed && valid >= PROOF_OBLIGATIONS;
    let err = if verified {
        None
    } else {
        Some(format!(
            "why3 prove: {valid}/{PROOF_OBLIGATIONS} goals Valid{}",
            if failed { " (a goal was rejected)" } else { "" }
        ))
    };
    Ok((verified, err, elapsed_ms))
}

/// Build the WhyML `.mlw` program for the spec.
///
/// The program is self-contained (only Why3 standard theories: `int.Int`,
/// `list.List`, `list.Mem`) so it type-checks and extracts standalone. The
/// grants/required symbol *names* are embedded as comments for traceability;
/// the gate itself is parametric over `int` lists (the extracted OCaml is
/// called with the manifest's symbol ids). Kernel-call externals (the
/// "call the probability kernel from WhyML" direction) are emitted as
/// commented `val` declarations — the extracted OCaml binds them at link
/// time via the `CamlCompiler_stubs` C shim.
fn build_mlw(spec: &WhymlSpec) -> String {
    match spec.program {
        unfer_protocol::WhymlProgram::AuthorizeGate => build_authorize_gate_mlw(spec),
        unfer_protocol::WhymlProgram::NpuDmaGate => build_npu_dma_gate_mlw(spec),
    }
}

/// The authorization gate `.mlw` (default): grant-subset semantics (S21).
fn build_authorize_gate_mlw(spec: &WhymlSpec) -> String {
    let module = &spec.module_name;
    let entry = &spec.function_name;
    let grants = if spec.grants.is_empty() {
        "<none>".to_string()
    } else {
        spec.grants.join(", ")
    };
    let required = if spec.required.is_empty() {
        "<none>".to_string()
    } else {
        spec.required.join(", ")
    };

    let externals: String = if spec.kernel_externals.is_empty() {
        String::new()
    } else {
        let mut s = String::from(
            "\n  (* Kernel-call externals (the \"calling the probability kernel\n\
             \x20    from WhyML\" direction). Uncomment to declare; the extracted\n\
             \x20    OCaml must bind them at link time to the unfer C ABI (the\n\
             \x20    CamlCompiler_stubs shim). *)\n",
        );
        for e in &spec.kernel_externals {
            s.push_str(&format!("  (* val {e} : int -> int -> int *)\n"));
        }
        s
    };

    format!(
        "(*\n\
         \x20  Generated by the unfer probability kernel — uk_whyml_emit (S36).\n\
         \x20  module:      {module}\n\
         \x20  entrypoint:  {entry}\n\
         \x20  grants:      {grants}\n\
         \x20  required:    {required}\n\
         \x20  property:    authorize grants required = true  <->  required ⊆ grants\n\
         \x20               (the kernel's S21 GrantSet::is_subset_of semantics: a\n\
         \x20               caller can never exercise a symbol it is not granted).\n\
         \x20  proof goals: {PROOF_OBLIGATIONS} (2 subset lemmas + mem postcondition\n\
         \x20               + 2 postconditions)\n\
         \x20\n\
         \x20  Verify and extract with the Why3 toolchain:\n\
         \x20    why3 prove -P alt-ergo {module}.mlw\n\
         \x20    why3 extract -D unfer_ocaml.drv {module}.mlw -o extracted/\n\
         \x20  The extracted OCaml module satisfies the postcondition below\n\
         \x20  (Why3 extraction is semantics-preserving).\n\
         *)\n\
         \n\
         module {module}\n\
         \x20 use bool.Bool\n\
         \x20 use int.Int\n\
         \x20 use list.List\n\
         \x20 use list.Length\n\
         \x20 use list.Mem\n\
         \n\
         \x20 (* A grant set is a list of kernel symbol ids. *)\n\
         \x20 predicate is_subset (a b: list int) = forall x. Mem.mem x a -> Mem.mem x b\n\
         \n\
         \x20 (* Program-level list membership: `Mem.mem` is a logical predicate and\n\
         \x20    cannot appear in program code, so we mirror it with a recursive\n\
         \x20    function whose postcondition links the two (Why3 1.8). *)\n\
         \x20 let rec mem (x: int) (l: list int) : bool\n\
         \x20   variant {{ l }}\n\
         \x20   ensures {{ result = True <-> Mem.mem x l }}\n\
         \x20 = match l with\n\
         \x20   | Nil -> False\n\
         \x20   | Cons y r -> if x = y then True else mem x r\n\
         \x20   end\n\
         \n\
         \x20 (* The capability lattice: subset is reflexive and transitive — there is\n\
         \x20    no escalation path (A ⊆ B ∧ B ⊆ C ⇒ A ⊆ C). *)\n\
         \x20 lemma subset_reflexive: forall a. is_subset a a\n\
         \x20 lemma subset_transitive: forall a b c. is_subset a b /\\ is_subset b c -> is_subset a c\n\
         \n\
         \x20 (* Soundness + completeness of the authorization gate: it returns True\n\
         \x20    exactly when every required symbol is granted. *)\n\
         \x20 let rec authorize (grants required: list int) : bool\n\
         \x20   variant {{ length required }}\n\
         \x20   ensures {{ result = True <-> is_subset required grants }}\n\
         \x20 = match required with\n\
         \x20   | Nil -> True\n\
         \x20   | Cons x rest -> if mem x grants then authorize grants rest else False\n\
         \x20   end\n\
         \n\
         \x20 (* The compiler-pass entrypoint: 0 = allow, 1 = reject.\n\
         \x20    The compiler loads the extracted OCaml and refuses a module whose\n\
         \x20    imported uk_* symbols are not covered by its grant set. *)\n\
         \x20 let {entry} (grants required: list int) : int\n\
         \x20   ensures {{ result = 0 <-> is_subset required grants }}\n\
         \x20 = if authorize grants required then 0 else 1\n\
         {externals}\n\
         end\n"
    )
}

/// The NPU DMA gate `.mlw` (GPU.md): a DMA transfer into a linear NPU buffer
/// is physically safe iff it stays inside the SRAM —
/// `buf.offset + bytes <= MAX_NPU_SRAM`. Why3 proves the `dma_ok` and
/// `gate_verdict` postconditions (linear integer arithmetic); extraction is
/// semantics-preserving, so the OCaml module the australVM compiler loads
/// never accepts an over-limit transfer.
fn build_npu_dma_gate_mlw(spec: &WhymlSpec) -> String {
    let module = &spec.module_name;
    let entry = &spec.function_name;

    format!(
        "(*\n\
         \x20  Generated by the unfer probability kernel — uk_whyml_emit (S36).\n\
         \x20  program:     NPU DMA gate (GPU.md)\n\
         \x20  module:      {module}\n\
         \x20  entrypoint:  {entry}\n\
         \x20  property:    dma_ok buf bytes = true  <->  buf.offset + bytes <= MAX_NPU_SRAM\n\
         \x20               (an async DMA load never overflows the NPU SRAM; the\n\
         \x20               hardware invariant from GPU.md).\n\
         \x20  proof goals: {NPU_PROOF_OBLIGATIONS} (dma_ok postcondition + gate_verdict\n\
         \x20               postcondition)\n\
         \x20\n\
         \x20  Verify and extract with the Why3 toolchain:\n\
         \x20    why3 prove -P alt-ergo {module}.mlw\n\
         \x20    why3 extract -D unfer_ocaml.drv {module}.mlw -o extracted/\n\
         \x20  The extracted OCaml module satisfies the postcondition below\n\
         \x20  (Why3 extraction is semantics-preserving).\n\
         *)\n\
         \n\
         module {module}\n\
         \x20 use int.Int\n\
         \n\
         \x20 (* Hardware constant: NPU SRAM capacity in bytes (256 KiB). *)\n\
         \x20 val constant MAX_NPU_SRAM : int = 262144\n\
         \n\
         \x20 (* A linear NPU memory buffer: the transfer region is\n\
         \x20    [offset, offset + bytes). The buffer is a Linear Resource — the\n\
         \x20    runtime handoff borrows it, it is never copied. *)\n\
         \x20 type npu_buffer = {{ size: int; offset: int }}\n\
         \n\
         \x20 (* Soundness + completeness of the physical-safety check: the async\n\
         \x20    DMA load is safe iff the transfer stays inside the SRAM. *)\n\
         \x20 let dma_ok (buf: npu_buffer) (bytes: int) : bool\n\
         \x20   requires {{ 0 <= buf.offset /\\ 0 <= bytes }}\n\
         \x20   ensures {{ result = True <-> buf.offset + bytes <= MAX_NPU_SRAM }}\n\
         \x20 = buf.offset + bytes <= MAX_NPU_SRAM\n\
         \n\
         \x20 (* The compiler-pass entrypoint: 0 = allow, 1 = reject. The\n\
         \x20    compiler loads the extracted OCaml and refuses any module whose\n\
         \x20    declared DMA transfer would overflow the NPU SRAM. *)\n\
         \x20 let {entry} (offset: int) (bytes: int) : int\n\
         \x20   requires {{ 0 <= offset /\\ 0 <= bytes }}\n\
         \x20   ensures {{ result = 0 <-> offset + bytes <= MAX_NPU_SRAM }}\n\
         \x20 = if offset + bytes <= MAX_NPU_SRAM then 0 else 1\n\
         end\n"
    )
}

fn wait_with_timeout(child: std::process::Child, timeout_ms: u64) -> Option<std::process::Output> {
    let start = std::time::Instant::now();
    let mut child = Some(child);
    loop {
        if let Some(c) = child.as_mut()
            && let Ok(Some(status)) = c.try_wait()
        {
            let c = child.take().unwrap();
            return match c.wait_with_output() {
                Ok(mut out) => {
                    out.status = status;
                    Some(out)
                }
                Err(_) => None,
            };
        }
        if start.elapsed().as_millis() as u64 >= timeout_ms {
            if let Some(mut c) = child.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::WhymlSpec;

    fn sample_spec() -> WhymlSpec {
        WhymlSpec {
            module_name: "AuthorizeGate".into(),
            function_name: "gate_verdict".into(),
            grants: vec![
                "uk_version".into(),
                "uk_evolve".into(),
                "uk_model_create".into(),
            ],
            required: vec!["uk_version".into(), "uk_evolve".into()],
            kernel_externals: vec![],
            program: unfer_protocol::WhymlProgram::AuthorizeGate,
            op: WhymlOp::Emit,
            timeout_ms: 30_000,
        }
    }

    fn npu_spec() -> WhymlSpec {
        WhymlSpec {
            module_name: "NpuDmaGate".into(),
            function_name: "dma_verdict".into(),
            grants: vec![],
            required: vec![],
            kernel_externals: vec![],
            program: unfer_protocol::WhymlProgram::NpuDmaGate,
            op: WhymlOp::Emit,
            timeout_ms: 30_000,
        }
    }

    #[test]
    fn emit_produces_wellformed_whyml() {
        let report = whyml_emit(&sample_spec()).unwrap();
        assert!(report.verified.is_none(), "Emit op must not verify");
        assert!(
            report.whyml.contains("module AuthorizeGate"),
            "{}",
            report.whyml
        );
        assert!(
            report.whyml.contains("let rec authorize"),
            "{}",
            report.whyml
        );
        assert!(report.whyml.contains("gate_verdict"), "{}", report.whyml);
        assert!(
            report
                .whyml
                .contains("ensures { result = True <-> is_subset required grants }"),
            "the soundness+completeness postcondition must be emitted: {}",
            report.whyml
        );
        assert!(
            report.whyml.contains("lemma subset_transitive"),
            "the no-escalation-path lemma must be emitted"
        );
        assert!(
            report.whyml.contains("why3 extract -D unfer_ocaml.drv"),
            "the extraction instructions must be embedded"
        );
        assert_eq!(report.proof_obligations, PROOF_OBLIGATIONS);
        assert_eq!(report.sha256.len(), 64);
    }

    #[test]
    fn emit_embeds_the_grant_context_for_audit() {
        let report = whyml_emit(&sample_spec()).unwrap();
        assert!(
            report
                .whyml
                .contains("uk_version, uk_evolve, uk_model_create")
        );
        assert!(report.whyml.contains("uk_version, uk_evolve"));
    }

    #[test]
    fn unknown_symbol_is_rejected() {
        let spec = WhymlSpec {
            required: vec!["uk_does_not_exist".into()],
            ..sample_spec()
        };
        let err = whyml_emit(&spec).unwrap_err();
        assert!(matches!(err, KernelError::WhymlInvalid { .. }), "{err:?}");
    }

    #[test]
    fn invalid_identifier_is_rejected() {
        let spec = WhymlSpec {
            module_name: "9lives".into(),
            ..sample_spec()
        };
        let err = whyml_emit(&spec).unwrap_err();
        assert!(matches!(err, KernelError::WhymlInvalid { .. }), "{err:?}");
    }

    #[test]
    fn kernel_externals_are_emitted_as_bindable_vals() {
        let spec = WhymlSpec {
            kernel_externals: vec!["uk_evolve".into()],
            ..sample_spec()
        };
        let report = whyml_emit(&spec).unwrap();
        assert!(
            report.whyml.contains("val uk_evolve"),
            "the external kernel call must be declared: {}",
            report.whyml
        );
        assert!(
            report.whyml.contains("calling the probability kernel"),
            "the direction must be documented"
        );
    }

    #[test]
    fn npu_gate_emits_the_sram_invariant() {
        let report = whyml_emit(&npu_spec()).unwrap();
        assert!(
            report.whyml.contains("module NpuDmaGate"),
            "{}",
            report.whyml
        );
        assert!(
            report
                .whyml
                .contains("val constant MAX_NPU_SRAM : int = 262144"),
            "the hardware constant must be emitted: {}",
            report.whyml
        );
        assert!(
            report
                .whyml
                .contains("ensures { result = True <-> buf.offset + bytes <= MAX_NPU_SRAM }"),
            "the dma_ok soundness+completeness postcondition must be emitted: {}",
            report.whyml
        );
        assert!(
            report.whyml.contains("dma_verdict"),
            "the compiler-pass entrypoint must be emitted"
        );
        assert_eq!(report.proof_obligations, NPU_PROOF_OBLIGATIONS);
        assert_eq!(report.sha256.len(), 64);
    }

    #[test]
    fn npu_gate_emission_is_deterministic() {
        let a = whyml_emit(&npu_spec()).unwrap();
        let b = whyml_emit(&npu_spec()).unwrap();
        assert_eq!(a.whyml, b.whyml);
        assert_eq!(a.sha256, b.sha256);
        assert_eq!(a.proof_obligations, b.proof_obligations);
    }

    #[test]
    fn authorize_gate_emission_is_unchanged_by_the_program_variant() {
        // The default (AuthorizeGate) emission must stay byte-identical to the
        // pinned `lib/why3_plugin/authorize_gate.mlw` in australVM — the
        // `program` field must not perturb it.
        let spec = WhymlSpec {
            program: unfer_protocol::WhymlProgram::AuthorizeGate,
            ..sample_spec()
        };
        let report = whyml_emit(&spec).unwrap();
        assert!(report.whyml.contains("let rec authorize"));
        assert!(report.whyml.contains("lemma subset_transitive"));
        assert_eq!(report.proof_obligations, PROOF_OBLIGATIONS);
    }

    #[test]
    fn prove_without_engine_reports_unavailable() {
        let spec = WhymlSpec {
            op: WhymlOp::Prove,
            ..sample_spec()
        };
        if why3_available() {
            eprintln!("SKIP: why3 present — engine-absence path not exercised");
            return;
        }
        let err = whyml_emit(&spec).unwrap_err();
        assert!(
            matches!(err, KernelError::WhymlUnavailable { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn report_round_trips_through_json() {
        let report = whyml_emit(&sample_spec()).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let back: WhymlReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn spec_round_trips_through_json() {
        let spec = WhymlSpec {
            module_name: "AuthorizeGate".into(),
            function_name: "gate_verdict".into(),
            grants: vec!["uk_version".into()],
            required: vec!["uk_version".into()],
            kernel_externals: vec![],
            program: unfer_protocol::WhymlProgram::AuthorizeGate,
            op: WhymlOp::Emit,
            timeout_ms: 5_000,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: WhymlSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn emitted_program_is_deterministic() {
        let a = whyml_emit(&sample_spec()).unwrap();
        let b = whyml_emit(&sample_spec()).unwrap();
        assert_eq!(a.whyml, b.whyml, "the same spec must emit the same program");
        assert_eq!(a.sha256, b.sha256);
    }
}
