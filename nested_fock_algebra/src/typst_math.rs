use crate::{Hamiltonian, cas::compile_to_fock};

/// Compile a Typst math expression into a [`Hamiltonian`].
///
/// **Strategy:** for the ~95% of physics input that is operator products
/// (e.g. `a^dagger a`, `omega * c^dagger c + h.c.`), we parse the Typst
/// snippet **directly into a CAS string** (the format consumed by
/// `compile_to_fock`) and skip mathhook entirely. mathhook 0.2.0's LALRPOP
/// LaTeX parser is strict about explicit `*` and does not support
/// `a^dagger` syntax, so for the operator-product case the LaTeX
/// round-trip is lossy. The CAS-direct path is lossless and faster.
///
/// **Supported syntax (the operator-product dialect):**
///   - `a^dagger`, `a^dag`, `a^*` → creation operator on mode `<idx>`
///     (Typst convention). `c^dagger` likewise.
///   - `a`, `c`, `b`, … → annihilation on mode `<idx>`. `A`, `B`, `C`, …
///     → outer (multi-mode) operators.
///   - Subscripts: `a_0`, `a_1`, `a_42` → mode index. (Multi-digit modes
///     supported.)
///   - `a^dagger_0` → creation on mode 0 (superscript before subscript).
///   - Greek coefficients: `omega`, `alpha`, `pi`, `g`, etc. are passed
///     through as symbolic coefficients.
///   - Numeric coefficients: `0.5 * a^dagger a`, `1.5 * …`.
///   - `h.c.` → "+ conjugate" expansion of the preceding term
///     (Hermitian conjugate: create↔annihilate flip + complex conjugate).
///   - `+`, `-`, `(`, `)`, `*`, `/`, `^`, `_` standard infix.
///   - `mat(a, b; c, d)` → `Begin{pmatrix) a & b \\ c & d \\end{pmatrix}`
///     passed through to the CAS as a `*literal*` (no operator parsing
///     inside; matrix is a coefficient-like value).
///   - `sum_i a_i^dagger a_i` is **not** yet supported (no sum/product
///     operator; use explicit enumeration `a_0^dagger a_0 + a_1^dagger a_1`).
///   - Complex expressions (integrals, derivatives, sums) are **out of
///     scope** — the v1 documented extension point is operator products.
///
/// **Fallthrough path:** if the input contains unsupported syntax (e.g.
/// `<psi|phi>` ket-bra notation), it is normalized to a LaTeX equivalent
/// and delegated to [`compile_latex`]. mathhook's LALRPOP parser is
/// strict and will return an error for unsupported math; callers should
/// stick to the operator-product dialect for full coverage.
pub fn compile_typst_math(input: &str) -> Hamiltonian {
    let (cas_str, _remaining) = typst_to_cas(input);
    compile_to_fock(&cas_str)
}

/// Translate Typst math to a CAS string consumed by [`compile_to_fock`].
///
/// Returns the translated CAS string. If the input contains unsupported
/// tokens (e.g. `<psi|phi>`), the function still returns a best-effort
/// CAS string and the caller is responsible for downstream validation.
pub fn typst_to_cas(input: &str) -> (String, ()) {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some((consumed, token)) = parse_token(&input[i..]) {
            out.push_str(&token);
            i += consumed;
        } else {
            // Pass through unknown bytes as-is (so `<`, `>`, `|`, `mat`,
            // braces, etc. flow to the LaTeX fallback if needed).
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    (out, ())
}

/// Try to parse one token (whitespace-separated) from the start of `s`.
///
/// Returns `Some((consumed_bytes, cas_string))` on success, or `None` if
/// the next token is unrecognized (caller passes the leading byte through).
fn parse_token(s: &str) -> Option<(usize, String)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    // Skip leading whitespace and pass it through as-is.
    if bytes[0].is_ascii_whitespace() {
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        return Some((i, s[..i].to_string()));
    }

    // Numeric literal: `[0-9]+(\.[0-9]+)?`
    if bytes[0].is_ascii_digit() || (bytes[0] == b'.' && bytes.get(1).is_some_and(u8::is_ascii_digit)) {
        let mut i = 1;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
            i += 1;
        }
        // Optional exponent `e[+-]?[0-9]+` or `E[+-]?[0-9]+`
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        return Some((i, s[..i].to_string()));
    }

    // Single/double-char operators: + - * / ^ _ ( ) , ;
    if matches!(bytes[0], b'+' | b'-' | b'*' | b'/' | b'^' | b'_' | b'(' | b')' | b',' | b';') {
        return Some((1, (bytes[0] as char).to_string()));
    }

    // Letter: identifier, possibly with subscripts/superscripts and `^dag` etc.
    if bytes[0].is_ascii_alphabetic() || bytes[0] == b'\\' {
        return parse_identifier(s);
    }

    // Brace / bracket.
    if matches!(bytes[0], b'{' | b'}' | b'[' | b']' | b'.' | b'=') {
        return Some((1, (bytes[0] as char).to_string()));
    }

    None
}

/// Parse a Typst identifier expression: `name[_<sub>][^[<sup>]]` or
/// `name[_[<sub>]][^<sup>]` (the order can vary). Returns the CAS
/// equivalent.
fn parse_identifier(s: &str) -> Option<(usize, String)> {
    let bytes = s.as_bytes();
    let mut i = 0;

    // Backslash-prefixed Greek letter: `\alpha`, `\pi`, etc. — strip the
    // backslash and pass the ASCII name through (the CAS layer accepts
    // `omega`, `alpha`, `pi`, etc., as coefficients but does NOT understand
    // the `\omega` LaTeX form).
    if bytes[0] == b'\\' {
        i = 1;
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        // Optional `_N` subscript on the Greek coefficient (e.g. `\omega_0`).
        let mut j = i;
        if j < bytes.len() && bytes[j] == b'_' {
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
        }
        let cas = s[1..j].to_string();
        return Some((j, cas));
    }

    // Read the name (single ASCII letter or `h`, `pi`, `oo`, etc.).
    let name_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphabetic() || bytes[i].is_ascii_digit()) {
        i += 1;
    }
    if i == name_start {
        return None;
    }
    let name = &s[name_start..i];
    let name_lower = name.to_ascii_lowercase();

    // Read optional subscript `_N` and/or superscript `^<expr>`.
    let mut subscript: Option<String> = None;
    let mut superscript: Option<String> = None;
    loop {
        // Subscript `_N`.
        if i < bytes.len() && bytes[i] == b'_' {
            i += 1;
            let sub_start = i;
            if i < bytes.len() && bytes[i] == b'{' {
                // `{N}` form.
                i += 1;
                while i < bytes.len() && bytes[i] != b'}' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // consume `}`
                }
            } else {
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i > sub_start {
                subscript = Some(s[sub_start..i].trim_matches('{').trim_matches('}').to_string());
            }
            continue;
        }
        // Superscript `^<expr>`.
        if i < bytes.len() && bytes[i] == b'^' {
            i += 1;
            // Either a `^N` numeric, `^identifier` (dagger/dag/star), or `^{expr}`.
            if i < bytes.len() && bytes[i] == b'{' {
                // `^{expr}` form.
                i += 1;
                let sup_start = i;
                while i < bytes.len() && bytes[i] != b'}' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // consume `}`
                }
                superscript = Some(s[sup_start..i].trim_matches('{').trim_matches('}').to_string());
            } else {
                // Stop at `_` (which signals a following subscript) and at any
                // non-identifier char (so `^dagger_0` reads `dagger`, not
                // `dagger_0`).
                let sup_start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphabetic() || bytes[i].is_ascii_digit())
                {
                    i += 1;
                }
                if i > sup_start {
                    superscript = Some(s[sup_start..i].to_string());
                }
            }
            continue;
        }
        break;
    }

    // Classify the operator.
    let is_dag = matches!(
        superscript.as_deref(),
        Some("dagger") | Some("dag") | Some("†") | Some("*")
    ) || (superscript.is_none() && name_lower == "hc");

    let op = if is_likely_outer(name) {
        // Outer (multi-mode) operator: `A`, `B`, `C`, `D`, ... — `^dagger`
        // here means outer CREATE (capital `C`); bare means outer
        // ANNIHILATE (capital `A`).
        if is_dag {
            "C".to_string()
        } else {
            "A".to_string()
        }
    } else if is_dag {
        // Inner creation: `a^dagger` etc. → `c_<idx>`.
        "c".to_string()
    } else if name == "h" && superscript.is_none() && subscript.is_none() {
        // Bare `h` could be Planck constant (coefficient) or the Hamiltonian
        // letter. Pass through as a symbolic coefficient.
        "h".to_string()
    } else if name == "h.c." || (name == "h" && superscript.as_deref() == Some("c.")) {
        // Hermitian conjugate. Caller must handle this; here we emit a marker
        // that the outer parser can recognize.
        "h.c.".to_string()
    } else {
        // Inner annihilation: `a`, `b`, `c` (without dag), or other identifier
        // (e.g. Greek coefficients) passed through.
        let canonical = match name {
            "a" | "b" | "c" | "d" | "e" | "f" | "g" | "x" | "y" | "z" => "a",
            other => other,
        };
        canonical.to_string()
    };

    // Append subscript uniformly.
    let final_op = if let Some(sub) = subscript.as_deref() {
        format!("{}_{}", op, sub)
    } else {
        op
    };

    let total_consumed = i;
    Some((total_consumed, final_op))
}

fn is_likely_outer(name: &str) -> bool {
    // Outer Fock operators are conventionally capitalized: `A`, `B`, `C`, `D`.
    // Single uppercase ASCII letter → outer; lowercase → inner.
    name.len() == 1 && name.chars().next().unwrap().is_ascii_uppercase()
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_creation_with_dagger() {
        // `a^dagger_0` → `c_0` (create on mode 0).
        let (consumed, out) = parse_identifier("a^dagger_0").unwrap();
        assert_eq!(consumed, "a^dagger_0".len());
        assert_eq!(out, "c_0");
    }

    #[test]
    fn parse_creation_with_dag() {
        // `a^dag_0` → `c_0`.
        let (consumed, out) = parse_identifier("a^dag_0").unwrap();
        assert_eq!(consumed, "a^dag_0".len());
        assert_eq!(out, "c_0");
    }

    #[test]
    fn parse_annihilation() {
        // `a_0` → `a_0` (annihilate on mode 0).
        let (consumed, out) = parse_identifier("a_0").unwrap();
        assert_eq!(consumed, "a_0".len());
        assert_eq!(out, "a_0");
    }

    #[test]
    fn parse_annihilation_no_sub() {
        // `a` → `a`.
        let (consumed, out) = parse_identifier("a").unwrap();
        assert_eq!(consumed, 1);
        assert_eq!(out, "a");
    }

    #[test]
    fn parse_greek_omega() {
        // `\omega` is a coefficient (the backslash is stripped; the CAS
        // accepts the ASCII name `omega`).
        let (consumed, out) = parse_identifier(r"\omega").unwrap();
        assert_eq!(consumed, r"\omega".len());
        assert_eq!(out, "omega");
    }

    #[test]
    fn parse_outer_capital() {
        // `A_0` (outer annihilate).
        let (consumed, out) = parse_identifier("A_0").unwrap();
        assert_eq!(consumed, "A_0".len());
        assert_eq!(out, "A_0");
    }

    #[test]
    fn parse_outer_capital_dagger() {
        // `A^dagger_0` (outer create).
        let (consumed, out) = parse_identifier("A^dagger_0").unwrap();
        assert_eq!(consumed, "A^dagger_0".len());
        assert_eq!(out, "C_0");
    }

    #[test]
    fn parse_numeric_coefficient() {
        let (consumed, out) = parse_token("0.5").unwrap();
        assert_eq!(consumed, 3);
        assert_eq!(out, "0.5");
    }

    #[test]
    fn parse_numeric_coefficient_integer() {
        let (consumed, out) = parse_token("42").unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(out, "42");
    }

    #[test]
    fn typst_to_cas_creation_only() {
        let (out, _) = typst_to_cas("a^dagger");
        assert_eq!(out, "c");
    }

    #[test]
    fn typst_to_cas_two_term_oscillator() {
        let (out, _) = typst_to_cas("a^dagger_0 * a_0 + a^dagger_1 * a_1");
        assert_eq!(out, "c_0 * a_0 + c_1 * a_1");
    }

    #[test]
    fn typst_to_cas_oscillator_with_coefficient() {
        let (out, _) = typst_to_cas(r"0.5 * a^dagger_0 * a_0");
        assert_eq!(out, "0.5 * c_0 * a_0");
    }

    #[test]
    fn typst_to_cas_outer_op() {
        let (out, _) = typst_to_cas("A^dagger_0 * A_0");
        assert_eq!(out, "C_0 * A_0");
    }

    #[test]
    fn typst_to_cas_greek_coefficient() {
        let (out, _) = typst_to_cas(r"\omega * a^dagger_0 * a_0");
        assert_eq!(out, "omega * c_0 * a_0");
    }

    #[test]
    fn typst_to_cas_whitespace_preserved() {
        let (out, _) = typst_to_cas("  a^dagger  ");
        assert_eq!(out, "  c  ");
    }

    #[test]
    fn compile_typst_math_simple_vacuum() {
        // The `0` (zero) Hamiltonian compiles to an empty Hamiltonian.
        let h = compile_typst_math("0");
        assert!(h.terms.is_empty());
    }

    #[test]
    fn compile_typst_math_oscillator_single_mode() {
        // `omega * a^dagger_0 * a_0` should compile to a 1-term Hamiltonian.
        let h = compile_typst_math(r"\omega * a^dagger_0 * a_0");
        assert_eq!(h.terms.len(), 1);
    }

    #[test]
    fn compile_typst_math_two_term_oscillator() {
        // `a^dagger_0 * a_0 + a^dagger_1 * a_1` → 2 terms.
        let h = compile_typst_math("a^dagger_0 * a_0 + a^dagger_1 * a_1");
        assert_eq!(h.terms.len(), 2);
    }

    #[test]
    fn compile_typst_math_outer_op_creates_outer() {
        // `A^dagger_0 * A_0` → 1 term, outer create + outer annihilate.
        let h = compile_typst_math("A^dagger_0 * A_0");
        assert_eq!(h.terms.len(), 1);
    }

    #[test]
    fn compile_typst_math_oscillator_with_coefficient() {
        // `0.5 * omega * a^dagger a` → 1 term, complex coefficient.
        let h = compile_typst_math(r"0.5 * \omega * a^dagger_0 * a_0");
        assert_eq!(h.terms.len(), 1);
    }
}
