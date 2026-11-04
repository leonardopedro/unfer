//! `verify_export` — re-check an arbitrary `lean4export` NDJSON file in nanoda.
//!
//!     cargo run -q -p prob_kernel --example verify_export -- <export.ndjson> [max-bytes]
//!         [--trusted NAMES] [--emit-trusted NAMES]
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
//! # Checking a shared closure once
//!
//! `lean4export -- <roots>` narrows the *roots*, not the payload: every module
//! that imports Mathlib drags in the same transitive closure (measured on
//! `BookProof.ChapterA4`: 24 own declarations → 54,053 declarations, 369 MiB,
//! 399 s). Re-checking that closure per module is the same work every time.
//!
//! `--emit-trusted` writes the names a full run actually verified (sorted, one
//! per line); `--trusted` later skips exactly those names while keeping them in
//! the environment, so a chapter's own declarations are still checked against
//! them. The manifest is cumulative: a `--trusted` run that emits unions what it
//! skipped with what it checked.
//!
//! This is trust-by-name, so it is only as strong as the manifest's provenance.
//! The pipeline regenerates it from a full run on the same Lean environment
//! (`scripts/lean-corpus-trust`) rather than accepting one from elsewhere.
//!
//! Why this exists: [`prob_kernel::verify::verify_export`] was only reachable
//! from `#[test]`s pinned to the vendored fixtures, so a *new* export (say, a
//! module freshly exported from `timepiece/`) could not be re-checked without
//! copying it over a fixture. This is the same code path, exposed as a command:
//! australVM's `scripts/lean-corpus-mm0` runs it between the export and the MM0
//! emission, so every statement file it writes is backed by an export the
//! independent kernel has accepted.
use std::collections::HashSet;
use std::process::ExitCode;

use prob_kernel::verify::verify_export_with_trust;
use unfer_protocol::LeanVerifySpec;

const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

fn usage() -> ExitCode {
    eprintln!(
        "usage: verify_export <export.ndjson> [max-bytes] [--trusted NAMES] [--emit-trusted NAMES]"
    );
    ExitCode::from(2)
}

fn read_names(path: &str) -> Result<HashSet<String>, ExitCode> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("verify_export: cannot read trusted names {path}: {e}");
            return Err(ExitCode::from(2));
        }
    };
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn main() -> ExitCode {
    let mut positionals: Vec<String> = Vec::new();
    let mut trusted_path: Option<String> = None;
    let mut emit_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trusted" | "--emit-trusted" => {
                let value = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("verify_export: {arg} needs a file path");
                        return usage();
                    }
                };
                if arg == "--trusted" {
                    trusted_path = Some(value);
                } else {
                    emit_path = Some(value);
                }
            }
            "-h" | "--help" => return usage(),
            other => positionals.push(other.to_string()),
        }
    }

    let path = match positionals.first() {
        Some(p) => p.clone(),
        None => return usage(),
    };
    let max_bytes = match positionals.get(1) {
        Some(n) => match n.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("verify_export: max-bytes must be an integer, got {n:?}");
                return ExitCode::from(2);
            }
        },
        None => DEFAULT_MAX_BYTES,
    };

    let trusted = match trusted_path.as_deref().map(read_names) {
        Some(Ok(set)) => Some(set),
        Some(Err(code)) => return code,
        None => None,
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
    let collect = emit_path.is_some();
    match verify_export_with_trust(&bytes, &spec, trusted.as_ref(), collect) {
        Ok(outcome) if outcome.report.verified => {
            let checked = outcome.report.declarations_checked;
            let skipped = outcome.skipped;
            if skipped == 0 {
                println!("nanoda: verified {path} ({} bytes, {checked} decls)", bytes.len());
            } else {
                println!(
                    "nanoda: verified {path} ({} bytes, {checked} decls, {skipped} skipped as trusted base)",
                    bytes.len()
                );
            }
            if let Some(emit) = emit_path {
                // Cumulative: what this run checked, plus what it was told to
                // trust, so manifests never shrink as modules are added.
                let mut all: Vec<String> = outcome.checked_names;
                if let Some(t) = &trusted {
                    all.extend(t.iter().cloned());
                }
                all.sort_unstable();
                all.dedup();
                let mut body = String::with_capacity(all.iter().map(|n| n.len() + 1).sum());
                for n in &all {
                    body.push_str(n);
                    body.push('\n');
                }
                if let Err(e) = std::fs::write(&emit, body) {
                    eprintln!("verify_export: cannot write trusted names {emit}: {e}");
                    return ExitCode::from(2);
                }
                println!("nanoda: wrote {} trusted names to {emit}", all.len());
            }
            ExitCode::SUCCESS
        }
        Ok(outcome) => {
            eprintln!(
                "nanoda: REJECTED {path} — failing declaration: {:?}, error: {:?}",
                outcome.report.failing_theorem, outcome.report.error
            );
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("nanoda: rejected {path} at parsing: {e}");
            ExitCode::from(1)
        }
    }
}
