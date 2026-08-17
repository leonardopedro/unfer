//! Lean4 proof verification over the probability kernel (S29).
//!
//! Mixes machine-checked theorem verification with the numerical pipeline:
//! a caller submits a Lean4 export file (the `lean4export` NDJSON format)
//! plus a [`LeanVerifySpec`] describing which axioms are permitted. The
//! external kernel [`nanoda_lib`] type-checks every declaration; the result
//! is reduced to a boolean verdict ([`ProofReport::verified`]).
//!
//! The reduction-to-boolean mirrors [`logos::deltanet`]'s unique-normal-form
//! reduction: an interaction net collapses a term to a canonical form which
//! is then SHA-256-hashed (`unf_hash_string`); a proof collapses to
//! `true`/`false` by *proof irrelevance* — all proofs of a proposition are
//! definitionally equal, so the kernel either validates the whole export
//! (`true`) or rejects it with the failing declaration (`false`).
//!
//! `nanoda_lib` signals rejection by panicking (its `check_all_declars` uses
//! `assert!`), so verification is run inside [`std::panic::catch_unwind`] and
//! the panic payload is captured into the report instead of unwinding into
//! the caller.

use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use nanoda_lib::util::Config;
use sha2::{Digest, Sha256};
use unfer_protocol::{LeanVerifySpec, ProofReport};

use crate::error::KernelError;

/// Maximum export payload accepted (16 MiB, matching [`LeanVerifySpec`]).
const MAX_EXPORT_BYTES: usize = 16 * 1024 * 1024;

/// Type-check a Lean4 export file against [`nanoda_lib`].
///
/// `export_bytes` is the raw `lean4export` NDJSON payload. Returns a
/// [`ProofReport`] whose `verified` flag is the boolean reduction of the
/// proofs in the export. Parse-level failures (malformed JSON, oversize
/// payload, unknown permitted axioms) are surfaced as
/// [`KernelError::ProofVerify`].
pub fn verify_export(
    export_bytes: &[u8],
    spec: &LeanVerifySpec,
) -> Result<ProofReport, KernelError> {
    if export_bytes.is_empty() {
        return Err(KernelError::ProofExportInvalid {
            reason: "export payload is empty".into(),
        });
    }
    if export_bytes.len() > spec.max_export_bytes.max(MAX_EXPORT_BYTES) {
        return Err(KernelError::ProofExportInvalid {
            reason: format!(
                "export payload {} bytes exceeds limit {}",
                export_bytes.len(),
                spec.max_export_bytes.max(MAX_EXPORT_BYTES)
            ),
        });
    }

    let export_hash = hex::encode(Sha256::digest(export_bytes));

    // nanoda has no public string-parsing entry point (`parse_export_file` is
    // `pub(crate)`); `Config::to_export_file` reads a path or stdin. Write the
    // payload to a temp file so we stay on the public API surface.
    let tmp_path = write_temp_export(export_bytes)?;
    let result = verify_at_path(&tmp_path, spec);
    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(mut report) => {
            report.export_hash = Some(export_hash);
            Ok(report)
        }
        Err(e) => Err(e),
    }
}

/// Type-check an export file already on disk.
fn verify_at_path(
    path: &std::path::Path,
    spec: &LeanVerifySpec,
) -> Result<ProofReport, KernelError> {
    let config = Config {
        export_file_path: Some(path.to_path_buf()),
        use_stdin: false,
        permitted_axioms: Some(spec.permitted_axioms.clone()),
        unpermitted_axiom_hard_error: spec.unpermitted_axiom_hard_error,
        nat_extension: spec.nat_extension,
        string_extension: spec.string_extension,
        pp_declars: None,
        pp_options: nanoda_lib::pretty_printer::PpOptions::default(),
        unknown_pp_declar_hard_error: false,
        pp_output_path: None,
        pp_to_stdout: false,
        num_threads: 1,
        print_success_message: false,
        print_axioms: false,
        unsafe_permit_all_axioms: false,
    };

    let (export_file, skipped) =
        config
            .to_export_file()
            .map_err(|e| KernelError::ProofExportInvalid {
                reason: format!("failed to parse export: {e}"),
            })?;

    let n_declars = export_file.declars.len();
    let skipped_names = skipped;

    // nanoda panics on a rejected proof (`assert!` inside `check_all_declars`).
    // Catch the panic and turn it into a `verified: false` report rather than
    // unwinding into the kernel caller. In `strict` mode the rejection is a
    // hard UK-4801 error instead.
    let panic_payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        export_file.check_all_declars();
    }))
    .err();

    match panic_payload {
        None => Ok(ProofReport {
            verified: true,
            declarations_checked: n_declars,
            failing_theorem: None,
            error: skipped_error(&skipped_names),
            export_hash: None,
        }),
        Some(payload) => {
            let message = panic_message(payload);
            let failing = message
                .lines()
                .find_map(|l| {
                    l.trim()
                        .strip_prefix("theorem ")
                        .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
                })
                .filter(|s| !s.is_empty());
            if spec.strict {
                Err(KernelError::ProofVerify { reason: message })
            } else {
                Ok(ProofReport {
                    verified: false,
                    declarations_checked: n_declars,
                    failing_theorem: failing.or(Some("(unknown)".into())),
                    error: Some(format!("proof rejected: {message}")),
                    export_hash: None,
                })
            }
        }
    }
}

/// If `unpermitted_axiom_hard_error` is false, axioms skipped during parse are
/// reported as a note (they are not a failure).
fn skipped_error(skipped: &[String]) -> Option<String> {
    if skipped.is_empty() {
        None
    } else {
        Some(format!(
            "skipped {} unpermitted axioms: {:?}",
            skipped.len(),
            skipped
        ))
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown kernel panic".to_string()
    }
}

fn write_temp_export(export_bytes: &[u8]) -> Result<PathBuf, KernelError> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "unfer_export_{}_{}.leanexport",
        std::process::id(),
        nonce
    ));
    let mut file = std::fs::File::create(&path).map_err(|e| KernelError::ProofExportInvalid {
        reason: format!("failed to create temp export: {e}"),
    })?;
    file.write_all(export_bytes)
        .map_err(|e| KernelError::ProofExportInvalid {
            reason: format!("failed to write temp export: {e}"),
        })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `P : Prop`, `P.mk : P`, `pf : P := P.mk` — a theorem proved by its
    /// (unique) constructor. This export type-checks cleanly in nanoda.
    const VALID_EXPORT: &str = concat!(
        r#"{"meta":{"exporter":{"name":"lean4export","version":"3.1.0"},"format":{"version":"3.1.0"},"lean":{"githash":"2fcce7258eeb6e324366bc25f9058293b04b7547","version":"4.27.0-rc1"}}}"#,
        "\n",
        r#"{"in":1,"str":{"pre":0,"str":"P"}}"#,
        "\n",
        r#"{"ie":0,"sort":0}"#,
        "\n",
        r#"{"in":2,"str":{"pre":1,"str":"mk"}}"#,
        "\n",
        r#"{"const":{"name":1,"us":[]},"ie":1}"#,
        "\n",
        r#"{"in":3,"str":{"pre":1,"str":"rec"}}"#,
        "\n",
        r#"{"in":4,"str":{"pre":0,"str":"u_1"}}"#,
        "\n",
        r#"{"il":1,"param":4}"#,
        "\n",
        r#"{"in":5,"str":{"pre":0,"str":"motive"}}"#,
        "\n",
        r#"{"in":6,"str":{"pre":0,"str":"t"}}"#,
        "\n",
        r#"{"ie":2,"sort":1}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":2,"name":6,"type":1},"ie":3}"#,
        "\n",
        r#"{"in":7,"str":{"pre":0,"str":"mk"}}"#,
        "\n",
        r#"{"bvar":0,"ie":4}"#,
        "\n",
        r#"{"const":{"name":2,"us":[]},"ie":5}"#,
        "\n",
        r#"{"app":{"arg":5,"fn":4},"ie":6}"#,
        "\n",
        r#"{"bvar":2,"ie":7}"#,
        "\n",
        r#"{"app":{"arg":4,"fn":7},"ie":8}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":8,"name":6,"type":1},"ie":9}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":9,"name":7,"type":6},"ie":10}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"implicit","body":10,"name":5,"type":3},"ie":11}"#,
        "\n",
        r#"{"ie":12,"lam":{"binderInfo":"default","body":4,"name":7,"type":6}}"#,
        "\n",
        r#"{"ie":13,"lam":{"binderInfo":"default","body":12,"name":5,"type":3}}"#,
        "\n",
        r#"{"inductive":{"types":[{"all":[1],"ctors":[2],"isRec":false,"isReflexive":false,"isUnsafe":false,"levelParams":[],"name":1,"numIndices":0,"numNested":0,"numParams":0,"type":0}],"ctors":[{"cidx":0,"induct":1,"isUnsafe":false,"levelParams":[],"name":2,"numFields":0,"numParams":0,"type":1}],"recs":[{"all":[1],"isUnsafe":false,"k":true,"levelParams":[4],"name":3,"numIndices":0,"numMinors":1,"numMotives":1,"numParams":0,"rules":[{"ctor":2,"nfields":0,"rhs":13}],"type":11}]}}"#,
        "\n",
        r#"{"in":8,"str":{"pre":0,"str":"pf"}}"#,
        "\n",
        r#"{"thm":{"name":8,"levelParams":[],"type":1,"value":5}}"#,
        "\n",
    );

    /// Same declarations, but `pf`'s proof term is `Sort 0` (type `Sort 1`),
    /// which does not inhabit `P` — nanoda's `assert_def_eq` panics.
    const INVALID_EXPORT: &str = concat!(
        r#"{"meta":{"exporter":{"name":"lean4export","version":"3.1.0"},"format":{"version":"3.1.0"},"lean":{"githash":"2fcce7258eeb6e324366bc25f9058293b04b7547","version":"4.27.0-rc1"}}}"#,
        "\n",
        r#"{"in":1,"str":{"pre":0,"str":"P"}}"#,
        "\n",
        r#"{"ie":0,"sort":0}"#,
        "\n",
        r#"{"in":2,"str":{"pre":1,"str":"mk"}}"#,
        "\n",
        r#"{"const":{"name":1,"us":[]},"ie":1}"#,
        "\n",
        r#"{"in":3,"str":{"pre":1,"str":"rec"}}"#,
        "\n",
        r#"{"in":4,"str":{"pre":0,"str":"u_1"}}"#,
        "\n",
        r#"{"il":1,"param":4}"#,
        "\n",
        r#"{"in":5,"str":{"pre":0,"str":"motive"}}"#,
        "\n",
        r#"{"in":6,"str":{"pre":0,"str":"t"}}"#,
        "\n",
        r#"{"ie":2,"sort":1}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":2,"name":6,"type":1},"ie":3}"#,
        "\n",
        r#"{"in":7,"str":{"pre":0,"str":"mk"}}"#,
        "\n",
        r#"{"bvar":0,"ie":4}"#,
        "\n",
        r#"{"const":{"name":2,"us":[]},"ie":5}"#,
        "\n",
        r#"{"app":{"arg":5,"fn":4},"ie":6}"#,
        "\n",
        r#"{"bvar":2,"ie":7}"#,
        "\n",
        r#"{"app":{"arg":4,"fn":7},"ie":8}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":8,"name":6,"type":1},"ie":9}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"default","body":9,"name":7,"type":6},"ie":10}"#,
        "\n",
        r#"{"forallE":{"binderInfo":"implicit","body":10,"name":5,"type":3},"ie":11}"#,
        "\n",
        r#"{"ie":12,"lam":{"binderInfo":"default","body":4,"name":7,"type":6}}"#,
        "\n",
        r#"{"ie":13,"lam":{"binderInfo":"default","body":12,"name":5,"type":3}}"#,
        "\n",
        r#"{"inductive":{"types":[{"all":[1],"ctors":[2],"isRec":false,"isReflexive":false,"isUnsafe":false,"levelParams":[],"name":1,"numIndices":0,"numNested":0,"numParams":0,"type":0}],"ctors":[{"cidx":0,"induct":1,"isUnsafe":false,"levelParams":[],"name":2,"numFields":0,"numParams":0,"type":1}],"recs":[{"all":[1],"isUnsafe":false,"k":true,"levelParams":[4],"name":3,"numIndices":0,"numMinors":1,"numMotives":1,"numParams":0,"rules":[{"ctor":2,"nfields":0,"rhs":13}],"type":11}]}}"#,
        "\n",
        r#"{"in":8,"str":{"pre":0,"str":"pf"}}"#,
        "\n",
        r#"{"thm":{"name":8,"levelParams":[],"type":1,"value":0}}"#,
        "\n",
    );

    #[test]
    fn valid_proof_reduces_to_true() {
        let report = verify_export(VALID_EXPORT.as_bytes(), &LeanVerifySpec::default()).unwrap();
        assert!(
            report.verified,
            "valid export should reduce to true: {:?}",
            report
        );
        assert_eq!(report.declarations_checked, 4, "{:?}", report);
        assert!(report.error.is_none(), "{:?}", report);
        assert!(report.export_hash.is_some(), "hash should be attached");
        assert_eq!(report.export_hash.as_deref().unwrap().len(), 64);
    }

    #[test]
    fn invalid_proof_reduces_to_false() {
        let report = verify_export(INVALID_EXPORT.as_bytes(), &LeanVerifySpec::default()).unwrap();
        assert!(
            !report.verified,
            "invalid export should reduce to false: {:?}",
            report
        );
        assert!(report.error.is_some(), "a rejection reason is expected");
        assert_eq!(report.declarations_checked, 4, "{:?}", report);
    }

    #[test]
    fn empty_export_is_an_error() {
        let err = verify_export(b"", &LeanVerifySpec::default()).unwrap_err();
        assert!(
            matches!(err, KernelError::ProofExportInvalid { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn oversize_export_is_rejected_before_parsing() {
        let spec = LeanVerifySpec {
            max_export_bytes: 8,
            ..LeanVerifySpec::default()
        };
        let payload = vec![b'{'; 1024];
        let err = verify_export(&payload, &spec).unwrap_err();
        assert!(
            matches!(err, KernelError::ProofExportInvalid { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn strict_mode_makes_rejection_a_hard_error() {
        let spec = LeanVerifySpec {
            strict: true,
            ..LeanVerifySpec::default()
        };
        let err = verify_export(INVALID_EXPORT.as_bytes(), &spec).unwrap_err();
        assert!(matches!(err, KernelError::ProofVerify { .. }), "{err:?}");
    }

    #[test]
    fn strict_mode_still_accepts_valid() {
        let spec = LeanVerifySpec {
            strict: true,
            ..LeanVerifySpec::default()
        };
        let report = verify_export(VALID_EXPORT.as_bytes(), &spec).unwrap();
        assert!(report.verified, "{:?}", report);
    }

    #[test]
    fn standard_axioms_profile() {
        let spec = LeanVerifySpec::standard_axioms();
        assert_eq!(
            spec.permitted_axioms,
            vec![
                "Quot.sound".to_string(),
                "Classical.choice".to_string(),
                "propext".to_string(),
                "Lean.trustCompiler".to_string(),
            ]
        );
    }

    #[test]
    fn proof_report_round_trips_through_json() {
        let report = ProofReport {
            verified: true,
            declarations_checked: 4,
            failing_theorem: None,
            error: None,
            export_hash: Some("deadbeef".to_string()),
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: ProofReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn lean_verify_spec_round_trips_through_json() {
        let spec = LeanVerifySpec::standard_axioms();
        let json = serde_json::to_string(&spec).unwrap();
        let back: LeanVerifySpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }

    /// The formal Logos confluence proof (`logos/lean/Confluence.lean`), exported
    /// to the `lean4export` NDJSON format 3.1.0 by the official
    /// `leanprover/lean4export` tool and pinned as a fixture. The proof uses
    /// kernel-computed `rfl` terms (not `native_decide`), so the independent
    /// external checker nanoda can re-verify it without trusting Lean's native
    /// compiler. Regenerate with:
    ///   lean -o Confluence.olean logos/lean/Confluence.lean
    ///   LEAN_PATH=logos/lean lean4export Confluence \
    ///     -- State.diamond_verified State.confluence_verified \
    ///        State.unique_normal_form_verified \
    ///     > prob_kernel/tests/fixtures/confluence.ndjson
    #[test]
    fn confluence_proof_verifies_in_nanoda() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/confluence.ndjson"
        );
        let bytes = std::fs::read(path).expect("read confluence.ndjson fixture");
        let spec = LeanVerifySpec {
            nat_extension: true,
            string_extension: true,
            ..LeanVerifySpec::standard_axioms()
        };
        let report = verify_export(&bytes, &spec).unwrap();
        assert!(
            report.verified,
            "nanoda rejected the Confluence proof: {:?} (failing: {:?})",
            report.error, report.failing_theorem
        );
    }
}
