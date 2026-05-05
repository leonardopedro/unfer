use crate::{Hamiltonian, InnerBosonicState, InnerFermionicState, Operator};
use num_complex::Complex64;
use quantrs2_symengine_pure::{
    expr::Expression,
    parser::parse,
};
use std::collections::BTreeMap;

/// Compile a symbolic operator expression string into a Hamiltonian.
pub fn compile_to_fock(input: &str) -> Hamiltonian {
    let expr = parse(input).expect("Failed to parse expression");
    compile_expression(expr)
}

/// Compile a pre-constructed symbolic Expression into a Hamiltonian.
pub fn compile_expression(expr: Expression) -> Hamiltonian {
    // 1. We ONLY call .expand(), NOT .simplify().
    // The default simplify() pass assumes commutativity (a*b = b*a),
    // which would destroy the physics of non-commuting operators.
    // .expand() preserves order while distributing (a+b)*c -> a*c + b*c.
    let expanded = expr.expand();
    
    // 2. Parse the resulting order-preserved S-expression string.
    let s_expr = expanded.to_string();
    let ast = SExpr::parse(&s_expr).expect("Failed to parse internal S-expression");

    // 3. Distribute all multiplication and division over sums to get a flat list of terms.
    let distributed = ast.distribute();

    // 4. Map each term to a physical Hamiltonian term.
    let mut terms = Vec::new();
    for term in distributed {
        if let Some(h_term) = term.to_hamiltonian_term() {
            terms.push(h_term);
        }
    }

    Hamiltonian { terms }
}

#[derive(Debug, Clone)]
enum SExpr {
    Num(f64),
    Sym(String),
    List(String, Vec<SExpr>),
}

impl SExpr {
    fn parse(input: &str) -> Option<Self> {
        let tokens: Vec<String> = input
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let mut pos = 0;
        Self::parse_tokens(&tokens, &mut pos)
    }

    fn parse_tokens(tokens: &[String], pos: &mut usize) -> Option<Self> {
        if *pos >= tokens.len() { return None; }
        let token = &tokens[*pos];
        *pos += 1;

        if token == "(" {
            if *pos >= tokens.len() { return None; }
            let op = tokens[*pos].clone();
            *pos += 1;
            let mut args = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                args.push(Self::parse_tokens(tokens, pos)?);
            }
            if *pos < tokens.len() { *pos += 1; } // Consume ")"
            Some(SExpr::List(op, args))
        } else if let Ok(n) = token.parse::<f64>() {
            Some(SExpr::Num(n))
        } else {
            Some(SExpr::Sym(token.clone()))
        }
    }

    /// Distribute Mul/Div/Neg over Add recursively.
    fn distribute(&self) -> Vec<SExpr> {
        match self {
            SExpr::List(op, args) if op == "+" => {
                args.iter().flat_map(|a| a.distribute()).collect()
            }
            SExpr::List(op, args) if op == "*" => {
                let distributed_args: Vec<Vec<SExpr>> = args.iter().map(|a| a.distribute()).collect();
                let mut results = vec![SExpr::List("*".to_string(), vec![])];
                for arg_set in distributed_args {
                    let mut next_results = Vec::new();
                    for r in results {
                        for a in &arg_set {
                            let mut new_args = match r.clone() {
                                SExpr::List(_, current_args) => current_args,
                                _ => vec![r.clone()],
                            };
                            new_args.push(a.clone());
                            next_results.push(SExpr::List("*".to_string(), new_args));
                        }
                    }
                    results = next_results;
                }
                results
            }
            SExpr::List(op, args) if op == "/" => {
                // (a + b) / c -> a/c + b/c
                let numerators = args[0].distribute();
                let denominator = if args.len() > 1 { args[1].clone() } else { SExpr::Num(1.0) };
                numerators.into_iter().map(|n| SExpr::List("/".to_string(), vec![n, denominator.clone()])).collect()
            }
            SExpr::List(op, args) if op == "neg" => {
                args[0].distribute().into_iter().map(|a| SExpr::List("neg".to_string(), vec![a])).collect()
            }
            SExpr::List(op, args) if op == "^" => {
                // (a + b)^n -> expand to multiplication chain then distribute
                if let SExpr::Num(n) = &args[1] {
                    let p = *n as i32;
                    if p > 0 {
                        let mut chain = args[0].clone();
                        for _ in 1..p {
                            chain = SExpr::List("*".to_string(), vec![chain, args[0].clone()]);
                        }
                        return chain.distribute();
                    } else if p == 0 {
                        return vec![SExpr::Num(1.0)];
                    }
                }
                vec![self.clone()]
            }
            SExpr::List(_op, _args) => {
                // For any other function (sqrt, etc.), distribute inside then rebuild
                // but usually these functions don't commute with distribution.
                // For physics, we mostly care about +, *, /, neg, ^.
                vec![self.clone()]
            }
            _ => vec![self.clone()],
        }
    }

    fn to_hamiltonian_term(&self) -> Option<(Complex64, Vec<Operator>)> {
        let mut coeff = Complex64::new(1.0, 0.0);
        let mut ops = Vec::new();
        self.collect_content(&mut coeff, &mut ops);
        if coeff.norm_sqr() > 1e-24 {
            Some((coeff, ops))
        } else {
            None
        }
    }

    fn collect_content(&self, coeff: &mut Complex64, ops: &mut Vec<Operator>) {
        match self {
            SExpr::Num(n) => { *coeff *= n; }
            SExpr::Sym(s) => {
                if s == "I" {
                    *coeff *= Complex64::i();
                } else if s == "pi" {
                    *coeff *= std::f64::consts::PI;
                } else if s == "e" {
                    *coeff *= std::f64::consts::E;
                } else if let Some(op) = map_variable_to_op(s) {
                    ops.push(op);
                }
                // Unknown symbols are treated as factor 1.0
            }
            SExpr::List(op, args) => {
                match op.as_str() {
                    "*" => { for a in args { a.collect_content(coeff, ops); } }
                    "/" => {
                        let mut num_c = Complex64::new(1.0, 0.0);
                        let mut den_c = Complex64::new(1.0, 0.0);
                        let mut num_ops = Vec::new();
                        let mut den_ops = Vec::new();
                        args[0].collect_content(&mut num_c, &mut num_ops);
                        if args.len() > 1 {
                            args[1].collect_content(&mut den_c, &mut den_ops);
                        }
                        *coeff *= num_c / den_c;
                        ops.extend(num_ops);
                        // We don't support operators in the denominator for this physics model
                    }
                    "neg" => {
                        args[0].collect_content(coeff, ops);
                        *coeff *= -1.0;
                    }
                    "^" => {
                        let mut base_c = Complex64::new(1.0, 0.0);
                        let mut base_ops = Vec::new();
                        args[0].collect_content(&mut base_c, &mut base_ops);
                        if let SExpr::Num(n) = &args[1] {
                            let p = *n as i32;
                            *coeff *= base_c.powi(p);
                            for _ in 0..p {
                                ops.extend(base_ops.clone());
                            }
                        }
                    }
                    "sqrt" => {
                        let mut inner_c = Complex64::new(1.0, 0.0);
                        let mut inner_ops = Vec::new();
                        args[0].collect_content(&mut inner_c, &mut inner_ops);
                        *coeff *= inner_c.sqrt();
                        // sqrt of operators not supported
                    }
                    _ => { /* other functions ignored for now */ }
                }
            }
        }
    }
}

fn map_variable_to_op(name: &str) -> Option<Operator> {
    let parse_suffix = |s: &str| -> Option<(bool, u32)> {
        if let Some(rest) = s.strip_prefix('f') {
            Some((true, rest.parse().ok()?))
        } else {
            Some((false, s.parse().ok()?))
        }
    };

    if let Some(suffix) = name.strip_prefix("c_") {
        let (is_fermionic, idx) = parse_suffix(suffix)?;
        if is_fermionic { Some(Operator::InnerFermionCreate(idx)) } 
        else { Some(Operator::InnerBosonCreate(idx)) }
    } else if let Some(suffix) = name.strip_prefix("a_") {
        let (is_fermionic, idx) = parse_suffix(suffix)?;
        if is_fermionic { Some(Operator::InnerFermionAnnihilate(idx)) } 
        else { Some(Operator::InnerBosonAnnihilate(idx)) }
    } else if let Some(suffix) = name.strip_prefix("C_") {
        let (is_fermionic, idx) = parse_suffix(suffix)?;
        if is_fermionic {
            let mut modes = std::collections::BTreeSet::new();
            modes.insert(idx);
            Some(Operator::OuterFermionCreate(InnerFermionicState { modes }))
        } else {
            let mut modes = BTreeMap::new();
            modes.insert(idx, 1);
            Some(Operator::OuterBosonCreate(InnerBosonicState { modes }))
        }
    } else if let Some(suffix) = name.strip_prefix("A_") {
        let (is_fermionic, idx) = parse_suffix(suffix)?;
        if is_fermionic {
            let mut modes = std::collections::BTreeSet::new();
            modes.insert(idx);
            Some(Operator::OuterFermionAnnihilate(InnerFermionicState { modes }))
        } else {
            let mut modes = BTreeMap::new();
            modes.insert(idx, 1);
            Some(Operator::OuterBosonAnnihilate(InnerBosonicState { modes }))
        }
    } else {
        None
    }
}
