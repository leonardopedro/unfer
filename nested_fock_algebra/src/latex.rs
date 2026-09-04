use crate::{Hamiltonian, cas::compile_to_fock};
use mathhook::prelude::*;

/// Translates a LaTeX Hamiltonian into the project's internal Hamiltonian format.
pub fn compile_latex(latex: &str) -> Hamiltonian {
    // The mathhook grammar has no dagger token and (worse) power-combines
    // same-base factors: `a_0^{dagger} * a_0` parses as `a_0^(1+dagger)` —
    // wrong for non-commuting ladder operators. Rewrite the standard physics
    // daggers (`a_i^\dagger`, `a_i^{\dagger}`, `†`) into the creation-family
    // names (`c_i`) BEFORE parsing, so `a_i^{\dagger}` compiles as the
    // creation operator `c_i` exactly as the mapping contract requires.
    let normalized = rewrite_daggers(latex);
    let parser = Parser::new(&ParserConfig::default());

    // mathhook 0.2.0 Parser::parse returns a Result<Expression, ParserError>
    let expr = parser
        .parse(&normalized)
        .expect("Failed to parse LaTeX expression with mathhook");

    let cas_str = transform_to_cas_string(&expr);
    #[cfg(debug_assertions)]
    if std::env::var("UNFER_LATEX_DEBUG").is_ok() {
        eprintln!("[latex] {latex:?} → {normalized:?} → {cas_str:?}");
    }
    compile_to_fock(&cas_str)
}

/// Rewrite `symbol^{\dagger}` / `symbol^\dagger` / `symbol^†` into the
/// creation-family name `c_symbol` (dropping the dagger), for symbols in the
/// annihilation family (`a`/`c` → `c`, `A`/`C` → `C`), so mathhook never sees
/// a dagger exponent. Everything else is left byte-identical.
fn rewrite_daggers(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        // Match a ladder symbol: [aAcC] followed by digits and/or `_` subscripts.
        if matches!(chars[i], 'a' | 'c' | 'A' | 'C')
            && i + 1 < chars.len()
            && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '_')
        {
            // Scan the full symbol name (digits, `_{...}`, `_d`).
            let start = i;
            i += 1;
            while i < chars.len() {
                if chars[i].is_ascii_digit() {
                    i += 1;
                } else if chars[i] == '_' {
                    i += 1;
                    if i < chars.len() && chars[i] == '{' {
                        i += 1;
                        while i < chars.len() && chars[i] != '}' {
                            i += 1;
                        }
                        if i < chars.len() {
                            i += 1; // consume '}'
                        }
                    } else {
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                } else {
                    break;
                }
            }
            // Look ahead for a dagger exponent: ^ ( {? dagger-ish }? ).
            if let Some(rest) = dagger_suffix(&chars[i..]) {
                let name: String = chars[start..i].iter().collect();
                // creation family: a→c, c→c, A→C, C→C (preserve subscript).
                let creation = if chars[start].is_ascii_uppercase() {
                    format!("C{}", &name[1..])
                } else {
                    format!("c{}", &name[1..])
                };
                out.push_str(&creation);
                i += rest;
            } else {
                out.push_str(&chars[start..i].iter().collect::<String>());
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// If `slice` starts with a dagger exponent (`^{\dagger}` / `^\dagger` /
/// `^{†}` / `^†` / `^\dag` / `^{\dag}`), return its length in chars.
fn dagger_suffix(slice: &[char]) -> Option<usize> {
    if slice.first() != Some(&'^') {
        return None;
    }
    let mut k = 1usize;
    let mut braced = false;
    if slice.get(k) == Some(&'{') {
        braced = true;
        k += 1;
    }
    // optional backslash command
    if slice.get(k) == Some(&'\\') {
        k += 1;
    }
    // The dagger body: dagger / dag / † (as command name or bare char).
    let body: String = slice[k..]
        .iter()
        .take_while(|c| **c != '}' && !c.is_whitespace())
        .collect();
    let is_dagger = matches!(body.as_str(), "dagger" | "dag" | "†");
    if !is_dagger {
        return None;
    }
    k += body.len();
    if braced && slice.get(k) == Some(&'}') {
        k += 1;
    }
    Some(k)
}

fn transform_to_cas_string(expr: &Expression) -> String {
    match expr {
        Expression::Number(n) => n.to_string(),
        Expression::Symbol(s) => map_to_annihilation(&s.name),
        Expression::Add(terms) => {
            let parts: Vec<String> = terms.iter().map(transform_to_cas_string).collect();
            format!("({})", parts.join(" + "))
        }
        Expression::Mul(terms) => {
            let parts: Vec<String> = terms.iter().map(transform_to_cas_string).collect();
            format!("({})", parts.join(" * "))
        }
        Expression::Pow(base, exp) => {
            let exp_str = transform_to_cas_string(exp);

            // Handle daggers/adjoints
            if (exp_str == "dagger" || exp_str == "dag" || exp_str == "†" || exp_str == "*")
                && let Expression::Symbol(s) = base.as_ref()
            {
                return map_to_creation(&s.name);
            }
            format!("({} ^ {})", transform_to_cas_string(base), exp_str)
        }
        Expression::Constant(c) => match c {
            MathConstant::Pi => "pi".to_string(),
            MathConstant::E => "e".to_string(),
            MathConstant::I => "I".to_string(),
            _ => "1.0".to_string(), // Fallback for unknown constants
        },
        Expression::Function { name, args } => {
            let name_str = name.to_string();
            if name_str == "frac" && args.len() == 2 {
                format!(
                    "({} / {})",
                    transform_to_cas_string(&args[0]),
                    transform_to_cas_string(&args[1])
                )
            } else {
                let parts: Vec<String> = args.iter().map(transform_to_cas_string).collect();
                format!("{}({})", name_str, parts.join(", "))
            }
        }

        _ => "0.0".to_string(),
    }
}

fn map_to_annihilation(name: &str) -> String {
    // Standard physics convention: `a` (and `A`) is the annihilation family;
    // `c`/`C` is already the creation family in the CAS dialect and must be
    // preserved as-is (the CAS parser distinguishes c = creation, a =
    // annihilation). A dagger (`a_i^{\dagger}`) is rewritten to `c_i` by
    // `rewrite_daggers` before parsing.
    let name = name.trim_start_matches('\\'); // Handle \psi etc.
    if let Some(suffix) = name.strip_prefix('a') {
        format!("a{}", suffix)
    } else if let Some(suffix) = name.strip_prefix('A') {
        format!("A{}", suffix)
    } else {
        name.to_string()
    }
}

fn map_to_creation(name: &str) -> String {
    let name = name.trim_start_matches('\\');
    if let Some(suffix) = name.strip_prefix('a').or_else(|| name.strip_prefix('c')) {
        format!("c{}", suffix)
    } else if let Some(suffix) = name.strip_prefix('A').or_else(|| name.strip_prefix('C')) {
        format!("C{}", suffix)
    } else {
        // If it's an unknown symbol with a dagger, we treat it as creation
        format!("c_{}", name)
    }
}
