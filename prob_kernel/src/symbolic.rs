//! Cadabra2 symbolic coupling (S30).
//!
//! Couples the existing LaTeX symbolic engine (`nested_fock_algebra`'s
//! mathhook parser → CAS-string dialect) with the external field-theory
//! CAS **Cadabra2** (GPL-3.0). Cadabra2 is invoked as a **subprocess**
//! (`cadabra2-cli`), so the Rust binary never links against its GPL-3.0
//! code — the exchange boundary is a TeX-subset expression on stdin and a
//! canonicalized string on stdout, the same "independent work" seam the
//! Lean4 integration uses.
//!
//! What Cadabra2 contributes that the Rust CAS deliberately does not:
//! the Rust `compile_expression` pipeline only calls `.expand()` and never
//! `.simplify()` (to preserve non-commuting operator order), so there is no
//! canonical normal form in the Rust engine. Cadabra2's `@canonicalise`
//! produces a unique canonical form (the `unf_hash` analogue) and a
//! zero-detection verdict (`verified`) — e.g. `H - H†` canonicalizing to
//! zero is the Hermiticity identity, `[H, Ω]` to zero is gauge invariance.
//!
//! The canonical form is normalized back into the CAS-string dialect
//! (`c_0 * a_0`, explicit `*`, no braced indices) so it can flow back into
//! [`nested_fock_algebra::compile_to_fock`] and become a numerical
//! [`Hamiltonian`].

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use unfer_protocol::{SymbolicOp, SymbolicReport, SymbolicSpec};

use crate::error::KernelError;

/// Env var overriding the Cadabra2 binary location.
pub const CADABRA_CLI_ENV: &str = "CADABRA_CLI";

/// Marker prefixes printed by the generated Cadabra2 script.
const NF_MARKER: &str = "UNFER_NF|";
const ZERO_MARKER: &str = "UNFER_ZERO|";

/// Locate the `cadabra2-cli` binary: `CADABRA_CLI` env override, then PATH.
pub fn cadabra_cli_path() -> Option<PathBuf> {
    if let Ok(over) = std::env::var(CADABRA_CLI_ENV)
        && !over.is_empty()
    {
        return Some(PathBuf::from(over));
    }
    // Search PATH manually so we can return None (not error) when absent.
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("cadabra2-cli");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// True iff a `cadabra2-cli` binary is discoverable.
pub fn cadabra_available() -> bool {
    cadabra_cli_path().is_some()
}

/// Run a symbolic operation in Cadabra2.
///
/// Returns [`KernelError::SymbolicUnavailable`] when the binary is missing,
/// [`KernelError::SymbolicInvalid`] when Cadabra2 rejects the input (no
/// canonical form produced).
pub fn symbolic_analyze(spec: &SymbolicSpec) -> Result<SymbolicReport, KernelError> {
    if spec.expression.trim().is_empty() {
        return Err(KernelError::SymbolicInvalid {
            reason: "expression is empty".into(),
        });
    }

    let cli = cadabra_cli_path().ok_or_else(|| KernelError::SymbolicUnavailable {
        reason: format!(
            "cadabra2-cli not found (set {CADABRA_CLI_ENV} or install the cadabra2 package)"
        ),
    })?;

    symbolic_analyze_with_cli(&cli, spec)
}

/// Internal entry taking an explicit CLI path (used by tests to inject a bad
/// path without mutating the process environment).
fn symbolic_analyze_with_cli(
    cli: &PathBuf,
    spec: &SymbolicSpec,
) -> Result<SymbolicReport, KernelError> {
    if spec.expression.trim().is_empty() {
        return Err(KernelError::SymbolicInvalid {
            reason: "expression is empty".into(),
        });
    }

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let script = build_script(&spec.expression, spec.op);
    let script_path = write_temp_script(&script)?;
    let output = run_cli(cli, &script_path, spec.timeout_ms)?;
    let _ = std::fs::remove_file(&script_path);

    let elapsed_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
        .saturating_sub(started);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let normal_form = parse_marker(&stdout, NF_MARKER)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let is_zero = parse_marker(&stdout, ZERO_MARKER)
        .map(|s| s.trim() == "True")
        .unwrap_or(false);

    if normal_form.is_empty() && !is_zero {
        return Err(KernelError::SymbolicInvalid {
            reason: format!(
                "Cadabra2 produced no canonical form (engine stderr: {})",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }

    let hash = hex::encode(Sha256::digest(normal_form.as_bytes()));
    Ok(SymbolicReport {
        verified: is_zero,
        normal_form,
        normal_form_hash: Some(hash),
        error: None,
        engine_ms: elapsed_ms,
    })
}

/// Translate a canonical form from Cadabra2's output notation back into the
/// CAS-string dialect the Rust engine accepts (`c_0 * a_0`, explicit `*`,
/// no braces). Cadabra2 prints braced subscripts (`c_{0}`) and implicit
/// multiplication (`2c_{0} a_{0}`); the Rust parser requires `2 * c_0 * a_0`.
///
/// Returns `None` if the input does not look like a Cadabra2 expression at
/// all (contains characters that are not part of either notation).
pub fn normalize_to_cas_dialect(input: &str) -> Option<String> {
    let mut out = String::with_capacity(input.len() * 2);
    let mut prev_was_atom = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '_' => {
                // Consume the whole subscript (`_{0}` or `_0`) as a unit so its
                // digits never trigger product insertion.
                out.push('_');
                if chars.peek() == Some(&'{') {
                    chars.next();
                    let mut digits = String::new();
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() || d == '-' {
                            digits.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if chars.peek() == Some(&'}') {
                        chars.next();
                    }
                    out.push_str(&digits);
                } else {
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            out.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                // prev_was_atom stays true: `c_0` is one atom.
            }
            '{' | '}' => {}
            ' ' => {
                // Implicit multiplication boundary: `2 c_{0}` → `2 * c_{0}`.
                if prev_was_atom
                    && chars
                        .peek()
                        .is_some_and(|n| n.is_alphanumeric() || *n == '(')
                {
                    out.push_str(" * ");
                } else if !out.ends_with(' ') {
                    out.push(' ');
                }
                prev_was_atom = false;
                continue;
            }
            '0'..='9' | 'a'..='z' | 'A'..='Z' => {
                // Adjacent atoms with no explicit `*` (`2c`, `AB`) become a
                // product: `2c_{0}` → `2 * c_0`.
                if prev_was_atom && !out.ends_with(" * ") {
                    out.push_str(" * ");
                }
                out.push(c);
                prev_was_atom = true;
                continue;
            }
            '+' | '-' | '*' | '/' | '^' | '(' | ')' | '.' | ',' | '\\' | '†' => {
                out.push(c);
                prev_was_atom = false;
            }
            other => {
                if other.is_whitespace() {
                    continue;
                }
                // Unknown character — not a Cadabra2 output we can feed back.
                return None;
            }
        }
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Build the Cadabra2 script for a given expression + operation.
fn build_script(expression: &str, op: SymbolicOp) -> String {
    // `Ex(...)` needs a Python string literal; escape backslashes and quotes.
    let escaped = expression.replace('\\', "\\\\").replace('"', "\\\"");
    match op {
        SymbolicOp::Canonicalize | SymbolicOp::VerifyHermitian => format!(
            "ex = Ex(\"{escaped}\")\n\
             canonicalise(ex)\n\
             print(\"{NF_MARKER}\" + str(ex))\n\
             print(\"{ZERO_MARKER}\" + str(ex == 0))\n"
        ),
        SymbolicOp::Simplify => format!(
            "ex = Ex(\"{escaped}\")\n\
             expand(ex)\n\
             canonicalise(ex)\n\
             print(\"{NF_MARKER}\" + str(ex))\n\
             print(\"{ZERO_MARKER}\" + str(ex == 0))\n"
        ),
    }
}

fn write_temp_script(script: &str) -> Result<PathBuf, KernelError> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "unfer_symbolic_{}_{}.cdb",
        std::process::id(),
        nonce
    ));
    let mut file = std::fs::File::create(&path).map_err(|e| KernelError::SymbolicInvalid {
        reason: format!("failed to create temp script: {e}"),
    })?;
    file.write_all(script.as_bytes())
        .map_err(|e| KernelError::SymbolicInvalid {
            reason: format!("failed to write temp script: {e}"),
        })?;
    Ok(path)
}

fn run_cli(
    cli: &PathBuf,
    script_path: &PathBuf,
    timeout_ms: u64,
) -> Result<std::process::Output, KernelError> {
    let child = Command::new(cli)
        .arg("-q")
        .arg("-n")
        .arg(script_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| KernelError::SymbolicUnavailable {
            reason: format!("failed to spawn cadabra2-cli: {e}"),
        })?;

    let guard = wait_with_timeout(child, timeout_ms);
    match guard {
        Some(out) => Ok(out),
        None => Err(KernelError::SymbolicUnavailable {
            reason: format!("cadabra2-cli timed out after {timeout_ms} ms"),
        }),
    }
}

fn wait_with_timeout(child: std::process::Child, timeout_ms: u64) -> Option<std::process::Output> {
    let start = std::time::Instant::now();
    let mut child = Some(child);
    loop {
        if let Some(c) = child.as_mut()
            && let Ok(Some(status)) = c.try_wait()
        {
            // Drain the pipes after the child exits.
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

fn parse_marker<'a>(stdout: &'a str, marker: &str) -> Option<&'a str> {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with(marker))
        .map(|l| l.trim_start().strip_prefix(marker).unwrap_or(""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_cadabra() -> bool {
        if !cadabra_available() {
            eprintln!(
                "SKIP: cadabra2-cli not found — set {} to run symbolic tests",
                CADABRA_CLI_ENV
            );
            return false;
        }
        true
    }

    #[test]
    fn normalize_braced_to_cas_dialect() {
        assert_eq!(
            normalize_to_cas_dialect("2c_{0} a_{0}").as_deref(),
            Some("2 * c_0 * a_0")
        );
        assert_eq!(
            normalize_to_cas_dialect("c_{0} * a_{0}").as_deref(),
            Some("c_0 * a_0")
        );
        assert_eq!(normalize_to_cas_dialect("0").as_deref(), Some("0"));
        assert_eq!(normalize_to_cas_dialect("A B").as_deref(), Some("A * B"));
    }

    #[test]
    fn normalize_rejects_non_cas_chars() {
        assert_eq!(normalize_to_cas_dialect("x ↦ y"), None);
        assert_eq!(normalize_to_cas_dialect(""), None);
    }

    #[test]
    fn canonicalize_combines_like_terms() {
        if !require_cadabra() {
            return;
        }
        let spec = SymbolicSpec {
            expression: "c_0 * a_0 + c_0 * a_0".into(),
            op: SymbolicOp::Canonicalize,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(!report.verified, "2 terms are not zero: {:?}", report);
        assert!(!report.normal_form.is_empty(), "{:?}", report);
        // The canonical form must translate back to the CAS dialect.
        let cas = normalize_to_cas_dialect(&report.normal_form).expect("translatable");
        assert!(cas.contains("c_0"), "missing c_0 in {cas}");
        assert!(cas.contains("a_0"), "missing a_0 in {cas}");
        assert!(!cas.contains('{'), "braces must be stripped: {cas}");
    }

    #[test]
    fn zero_detection_reduces_to_boolean() {
        if !require_cadabra() {
            return;
        }
        let spec = SymbolicSpec {
            expression: "a_0 * c_0 - a_0 * c_0".into(),
            op: SymbolicOp::Canonicalize,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(
            report.verified,
            "difference of identical terms is zero: {:?}",
            report
        );
        assert_eq!(
            normalize_to_cas_dialect(&report.normal_form).as_deref(),
            Some("0")
        );
        assert_eq!(report.normal_form_hash.as_deref().unwrap().len(), 64);
    }

    #[test]
    fn verify_hermitian_identity() {
        if !require_cadabra() {
            return;
        }
        // H - H† where both are the same expression → zero.
        let spec = SymbolicSpec {
            expression: "c_0 * a_0 - c_0 * a_0".into(),
            op: SymbolicOp::VerifyHermitian,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(
            report.verified,
            "H - H† = 0 for a self-adjoint term: {:?}",
            report
        );
    }

    #[test]
    fn empty_expression_is_invalid() {
        let spec = SymbolicSpec {
            expression: "   ".into(),
            op: SymbolicOp::Canonicalize,
            timeout_ms: 1_000,
        };
        let err = symbolic_analyze(&spec).unwrap_err();
        assert!(
            matches!(err, KernelError::SymbolicInvalid { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn unavailable_engine_reports_error() {
        let bad = PathBuf::from("/nonexistent/cadabra2-cli");
        let spec = SymbolicSpec {
            expression: "c_0 * a_0".into(),
            op: SymbolicOp::Canonicalize,
            timeout_ms: 1_000,
        };
        let err = symbolic_analyze_with_cli(&bad, &spec).unwrap_err();
        assert!(
            matches!(err, KernelError::SymbolicUnavailable { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn canonical_hamiltonian_feeds_back_into_rust_engine() {
        if !require_cadabra() {
            return;
        }
        // The canonical form of `c_0 * a_0 + c_0 * a_0` (i.e. `2 * c_0 * a_0`)
        // must compile back into a numerical Hamiltonian via the Rust engine.
        let spec = SymbolicSpec {
            expression: "c_0 * a_0 + c_0 * a_0".into(),
            op: SymbolicOp::Canonicalize,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        let cas = normalize_to_cas_dialect(&report.normal_form).expect("translatable");
        let hamiltonian = nested_fock_algebra::compile_to_fock_bounded(
            &cas,
            &nested_fock_algebra::cas::ExpansionLimits::default(),
        )
        .expect("canonical form must compile");
        assert_eq!(hamiltonian.terms.len(), 1, "2*c_0*a_0 is one term");
        let (coeff, ops) = &hamiltonian.terms[0];
        assert_eq!(*coeff, num_complex::Complex::new(2.0, 0.0), "{cas:?}");
        assert_eq!(ops.len(), 2, "c_0 then a_0: {ops:?}");
    }

    #[test]
    fn symbolic_report_round_trips_through_json() {
        let report = SymbolicReport {
            verified: true,
            normal_form: "0".into(),
            normal_form_hash: Some("deadbeef".into()),
            error: None,
            engine_ms: 3,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: SymbolicReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn symbolic_spec_round_trips_through_json() {
        let spec = SymbolicSpec {
            expression: "c_0 * a_0".into(),
            op: SymbolicOp::Simplify,
            timeout_ms: 5_000,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SymbolicSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }
}
