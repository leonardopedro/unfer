//! `verify_export` — re-check an arbitrary `lean4export` NDJSON file in nanoda.
//!
//!     cargo run -q -p prob_kernel --example verify_export -- <export.ndjson> [max-bytes]
//!
//! Exit status: 0 = nanoda accepted every declaration, 1 = rejected (the failing
//! declaration is printed), 2 = usage/IO/parse error.
//!
//! The size cap defaults to 256 MiB here, not the protocol's 16 MiB: that 16 MiB
//! is [`LeanVerifySpec`]'s DoS bound for payloads arriving over the wire (a
//! service concern), while a local audit of a module that quotes large syntax
//! trees routinely produces 30 MiB+ exports. Exceeding the cap is a *gap*, not a
//! kernel rejection — pass a bigger number rather than reading it as "unproved".
//!
//! Why this exists: [`prob_kernel::verify::verify_export`] was only reachable
//! from `#[test]`s pinned to the vendored fixtures, so a *new* export (say, a
//! module freshly exported from `timepiece/`) could not be re-checked without
//! copying it over a fixture. This is the same code path, exposed as a command:
//! australVM's `scripts/lean-corpus-mm0` runs it between the export and the MM0
//! emission, so every statement file it writes is backed by an export the
//! independent kernel has accepted.
use std::process::ExitCode;

use prob_kernel::verify::verify_export;
use unfer_protocol::LeanVerifySpec;

const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: verify_export <export.ndjson> [max-bytes]");
            return ExitCode::from(2);
        }
    };
    let max_bytes = match args.next() {
        Some(n) => match n.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("verify_export: max-bytes must be an integer, got {n:?}");
                return ExitCode::from(2);
            }
        },
        None => DEFAULT_MAX_BYTES,
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("verify_export: cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    // Same spec the pinned-fixture tests use: the standard axiom set plus the
    // Nat/String kernel extensions Lean's elaborator relies on.
    let spec = LeanVerifySpec {
        nat_extension: true,
        string_extension: true,
        max_export_bytes: max_bytes,
        ..LeanVerifySpec::standard_axioms()
    };
    match verify_export(&bytes, &spec) {
        Ok(report) if report.verified => {
            println!("nanoda: verified {} ({} bytes)", path, bytes.len());
            ExitCode::SUCCESS
        }
        Ok(report) => {
            eprintln!(
                "nanoda: REJECTED {path} — failing declaration: {:?}, error: {:?}",
                report.failing_theorem, report.error
            );
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("nanoda: rejected {path} at parsing: {e}");
            ExitCode::from(1)
        }
    }
}
