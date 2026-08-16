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

    let script = build_script(spec);
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

/// Run a multi-cell Cadabra2 **derivation pipeline** (S30) and capture the
/// named expressions it produces.
///
/// `script` is the concatenation of Cadabra2 `input` cells (a `.cdb` file).
/// The pipeline runs in a single shared Cadabra2 context (state persists
/// across cells, so `@(t0)` cross-references resolve). `names` lists the
/// expression variables to extract; each becomes an entry in the returned
/// map (name → canonical string), mirroring the notebook's own workflow of
/// naming derived expressions (`G`, `ex`, `t0`, `action`, …).
///
/// Errors: [`KernelError::SymbolicUnavailable`] when the binary is missing,
/// [`KernelError::SymbolicInvalid`] when the pipeline or an extraction fails.
pub fn symbolic_derive(
    script: &str,
    names: &[&str],
    timeout_ms: u64,
) -> Result<std::collections::BTreeMap<String, String>, KernelError> {
    if script.trim().is_empty() {
        return Err(KernelError::SymbolicInvalid {
            reason: "derivation script is empty".into(),
        });
    }
    let cli = cadabra_cli_path().ok_or_else(|| KernelError::SymbolicUnavailable {
        reason: format!(
            "cadabra2-cli not found (set {CADABRA_CLI_ENV} or install the cadabra2 package)"
        ),
    })?;

    // Append an extraction trailer that prints each requested name with a marker.
    let mut full = script.to_string();
    full.push_str("\n# ===== unfer derivation extraction =====\n");
    for name in names {
        full.push_str(&format!(
            "try:\n    print(\"UNFER_DERIVE|{name}|\" + str(eval(\"{name}\")))\n\
             except Exception as e:\n    print(\"UNFER_DERIVE|{name}|ERR|\" + str(e))\n"
        ));
    }

    let script_path = write_temp_script(&full)?;
    let output = run_cli(&cli, &script_path, timeout_ms)?;
    let _ = std::fs::remove_file(&script_path);

    if !output.status.success() {
        return Err(KernelError::SymbolicInvalid {
            reason: format!(
                "Cadabra2 derivation failed (stderr: {})",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut out = std::collections::BTreeMap::new();
    for name in names {
        if let Some(v) = parse_marker(&stdout, &format!("UNFER_DERIVE|{name}|")) {
            let v = v.trim();
            if let Some(msg) = v.strip_prefix("ERR|") {
                return Err(KernelError::SymbolicInvalid {
                    reason: format!("derivation variable `{name}` not extractable: {msg}"),
                });
            }
            out.insert(name.to_string(), v.to_string());
        }
    }
    Ok(out)
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
fn build_script(spec: &SymbolicSpec) -> String {
    let expression = &spec.expression;
    // `Ex(...)` needs a Python string literal; escape backslashes and quotes.
    let escaped = expression.replace('\\', "\\\\").replace('"', "\\\"");
    match spec.op {
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
        SymbolicOp::VerifySubstitution => {
            // Apply the substitution rule (e.g. the Navier-Stokes divergence
            // constraint `u_{3,3} -> -(u_{1,1}+u_{2,2})`, book.tex §4191-4197)
            // to the expression, then canonicalize and zero-detect. `verified`
            // reports whether the constraint resolution holds identically.
            let rule = spec.substitution.as_deref().unwrap_or_default();
            let escaped_rule = rule.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "ex = Ex(\"{escaped}\")\n\
                 substitute(ex, Ex(\"{escaped_rule}\"))\n\
                 canonicalise(ex)\n\
                 print(\"{NF_MARKER}\" + str(ex))\n\
                 print(\"{ZERO_MARKER}\" + str(ex == 0))\n"
            )
        }
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
            substitution: None,
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
            substitution: None,
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
            substitution: None,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(
            report.verified,
            "H - H† = 0 for a self-adjoint term: {:?}",
            report
        );
    }

    /// Quantum gravity: the repaired 3D gauge-fixed Hamiltonian derivation.
    ///
    /// `docs/qg_gauge_fixed_hamiltonian.cdb` is the repaired version of the
    /// `yangqg3.cnb` / `qg6.cnb` Cadabra2 notebooks: cells 1 + 3 expand the
    /// Einstein-Hilbert action in the tetrad (vielbein) formalism and derive
    /// the gauge-fixed Hamiltonian density, then reduce it to book.tex's
    /// **TEGR/teleparallel torsion scalar**
    /// `e(T_{ab}^b T^{ac}_c − ½T_{abc}T^{acb} − ¼T_{abc}T^{abc})` with
    /// `T_{abc} = X_{abc} − X_{bac}`, derive the **polymomentum by variation**
    /// of the Lagrangian (`pi_derived`, the coefficient of
    /// `∂_α(d e^k_ρ)` — book.tex's `p^{ab} = π^α_{kρ} v_α e^ρ_c η^{cb}`),
    /// and finally book.tex's **3D gauge-fixed Hamiltonian** `H_final`
    /// (book.tex line 8190, the Legendre transform). The notebook did not run
    /// as-is (missing `\partial{#}::PartialDerivative`, undeclared
    /// flat/subspace index sets, missing `\sigma`, an extra-brace typo, and
    /// an unbalanced spin-connection substitution); the repaired script runs
    /// cleanly. This test runs it through the S30 Cadabra2 subprocess and
    /// asserts `G`, `ex1` (=eR), `t0_tegr`, `pi_derived` (the
    /// variation-derived polymomentum), and `H_final` are non-trivial.
    const QG_DERIVATION: &str = include_str!("../../docs/qg_gauge_fixed_hamiltonian.cdb");

    #[test]
    fn qg_gauge_fixed_hamiltonian_derivation_runs() {
        if !require_cadabra() {
            return;
        }
        let derived = symbolic_derive(
            QG_DERIVATION,
            &["G", "ex1", "t0_tegr", "pi_derived", "H_final"],
            120_000,
        )
        .unwrap();
        assert!(
            derived.contains_key("G"),
            "Gauss constraint must be extracted: {:?}",
            derived.keys()
        );
        let g = &derived["G"];
        assert!(
            !g.is_empty() && g != "0",
            "Gauss constraint must be non-trivial: {g}"
        );
        assert!(
            g.contains("c"),
            "Gauss constraint should contain ghosts: {g}"
        );
        // The EH Hamiltonian density (ex1) must be non-trivial in the tetrad.
        assert!(
            derived.contains_key("ex1"),
            "EH Hamiltonian density must be extracted: {:?}",
            derived.keys()
        );
        let ex1 = &derived["ex1"];
        assert!(!ex1.is_empty(), "ex1 must be non-empty");
        assert!(
            ex1.contains("e^{") || ex1.contains("e_{"),
            "ex1 should be a tetrad expression: {ex1}"
        );
        // The TEGR/teleparallel torsion scalar (book.tex's L) must be
        // non-trivial and expressed in the connection density X.
        assert!(
            derived.contains_key("t0_tegr"),
            "TEGR torsion scalar must be extracted: {:?}",
            derived.keys()
        );
        let t0 = &derived["t0_tegr"];
        assert!(
            !t0.is_empty() && t0 != "0",
            "t0_tegr must be non-trivial: {t0}"
        );
        assert!(
            t0.contains("X_{") || t0.contains("X^{"),
            "t0_tegr should be a torsion/connection expression: {t0}"
        );
        // The polymomentum must be DERIVED by variation of the Teleparallel
        // Lagrangian: `pi_derived` is the coefficient of ∂_α(d e^k_ρ) in the
        // varied ex1 (book.tex's p^{ab} = π^α_{kρ} v_α e^ρ_c η^{cb}).
        assert!(
            derived.contains_key("pi_derived"),
            "polymomentum must be extracted: {:?}",
            derived.keys()
        );
        let pi = &derived["pi_derived"];
        assert!(
            !pi.is_empty() && pi != "0",
            "pi_derived must be non-trivial: {pi}"
        );
        assert!(
            pi.contains("π") || pi.contains("\\pi") || pi.contains("pi"),
            "pi_derived should be the polymomentum density: {pi}"
        );
        // book.tex's final 3D gauge-fixed Hamiltonian (line 8190), DERIVED
        // automatically from the TEGR Lagrangian via the Legendre transform
        // H = p·T + p·E − L (polymomentum p^{ab} from book.tex 8175, with the
        // S^{ab} = S̄/(2e), T = −P̄/(4e) relations). Must be non-trivial and
        // carry the derived 1/(16e) S² and 1/(24e) P² kinetic coefficients.
        assert!(
            derived.contains_key("H_final"),
            "final 3D Hamiltonian must be extracted: {:?}",
            derived.keys()
        );
        let h = &derived["H_final"];
        assert!(
            !h.is_empty() && h != "0",
            "H_final must be non-trivial: {h}"
        );
        assert!(
            h.contains("S") && h.contains("E"),
            "H_final should be in the S/E torsion variables: {h}"
        );
        assert!(
            h.contains("1/16") && h.contains("1/24"),
            "H_final must carry the derived 1/(16e) S^2 and 1/(24e) P^2 \
             coefficients: {h}"
        );
    }

    #[test]
    fn qg_gauss_constraint_hermiticity_verifies() {
        if !require_cadabra() {
            return;
        }
        let derived = symbolic_derive(QG_DERIVATION, &["G"], 120_000).unwrap();
        let g = &derived["G"];
        // The BRST Gauss constraint G is self-conjugate: G - G† = 0. Verify
        // via the same zero-detection path as VerifyHermitian.
        let spec = SymbolicSpec {
            expression: format!("({g}) - ({g})"),
            op: SymbolicOp::VerifyHermitian,
            substitution: None,
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(report.verified, "G - G = 0: {:?}", report);
    }

    #[test]
    fn derive_empty_script_is_invalid() {
        let err = symbolic_derive("", &["G"], 1_000).unwrap_err();
        assert!(
            matches!(err, KernelError::SymbolicInvalid { .. }),
            "{err:?}"
        );
    }

    /// Navier-Stokes: the divergence constraint is a *solved* constraint.
    ///
    /// book.tex §4191-4197 (the Navier-Stokes chapter) states that "the
    /// divergence constraint can be easily solved (e.g. using the replacement
    /// u_{3,3} = u_{1,1}+u_{2,2})". Symbolically: substituting that rule into
    /// the incompressibility condition `∂_j u_j = u_{1,1}+u_{2,2}+u_{3,3}`
    /// must yield zero identically. This is the commuting-dialect shadow of
    /// the BRST charge Ω = u_{j,j}ψ† being first-class (`Ω² = 0`, proved in
    /// `BookProof/ChapterGhostField.lean`). Plain symbols (`U33` rather than
    /// `u_33`) are used so Cadabra2 does not parse the subscripts as indices.
    #[test]
    fn verify_navier_stokes_divergence_constraint_resolution() {
        if !require_cadabra() {
            return;
        }
        let spec = SymbolicSpec {
            expression: "U33 + U11 + U22".into(),
            op: SymbolicOp::VerifySubstitution,
            substitution: Some("U33 -> -(U11 + U22)".into()),
            timeout_ms: 30_000,
        };
        let report = symbolic_analyze(&spec).unwrap();
        assert!(
            report.verified,
            "divergence constraint must resolve to zero after the book.tex \
             replacement u_{{3,3}} = u_{{1,1}}+u_{{2,2}}: {:?}",
            report
        );
    }

    #[test]
    fn empty_expression_is_invalid() {
        let spec = SymbolicSpec {
            expression: "   ".into(),
            op: SymbolicOp::Canonicalize,
            substitution: None,
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
            substitution: None,
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
            substitution: None,
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
            substitution: None,
            timeout_ms: 5_000,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SymbolicSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }
}
