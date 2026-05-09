use mathhook::prelude::*;
use crate::{Hamiltonian, cas::compile_to_fock};

/// Translates a LaTeX Hamiltonian into the project's internal Hamiltonian format.
pub fn compile_latex(latex: &str) -> Hamiltonian {
    let parser = Parser::new(&ParserConfig::default());

    // mathhook 0.2.0 Parser::parse returns a Result<Expression, ParserError>
    let expr = parser.parse(latex).expect("Failed to parse LaTeX expression with mathhook");
    
    let cas_str = transform_to_cas_string(&expr);
    compile_to_fock(&cas_str)
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
            if exp_str == "dagger" || exp_str == "dag" || exp_str == "†" || exp_str == "*" {
                if let Expression::Symbol(s) = base.as_ref() {
                    return map_to_creation(&s.name);
                }
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
                format!("({} / {})", transform_to_cas_string(&args[0]), transform_to_cas_string(&args[1]))
            } else {
                let parts: Vec<String> = args.iter().map(transform_to_cas_string).collect();
                format!("{}({})", name_str, parts.join(", "))
            }
        }

        _ => "0.0".to_string(),
    }
}


fn map_to_annihilation(name: &str) -> String {
    // Standard physics convention: c or a means annihilation
    // If it has a subscript, preserve it.
    // We normalize everything to 'a' prefix for annihilation in our CAS.
    let name = name.trim_start_matches('\\'); // Handle \psi etc.
    if let Some(suffix) = name.strip_prefix('a').or_else(|| name.strip_prefix('c')) {
         format!("a{}", suffix)
    } else if let Some(suffix) = name.strip_prefix('A').or_else(|| name.strip_prefix('C')) {
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

