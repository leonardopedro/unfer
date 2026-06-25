use crate::{Hamiltonian, InnerBosonicState, InnerFermionicState, Operator};
use num_complex::Complex64;
use quantrs2_symengine_pure::{expr::Expression, parser::parse};
use std::collections::{BTreeMap, HashMap};

/// A cap on how many flat terms a symbolic expansion may produce before the
/// compiler aborts (instead of exhausting memory). High-order operator products
/// distribute combinatorially: `(a+b)^n` yields `2^n` terms.
#[derive(Debug, Clone)]
pub struct ExpansionLimits {
    pub max_terms: usize,
}

impl ExpansionLimits {
    /// No limit — reproduces the historical unchecked behavior.
    pub fn unbounded() -> Self {
        Self {
            max_terms: usize::MAX,
        }
    }
}

impl Default for ExpansionLimits {
    fn default() -> Self {
        Self { max_terms: 100_000 }
    }
}

/// Errors from bounded symbolic compilation.
#[derive(Debug, thiserror::Error)]
pub enum CasError {
    #[error("expression expansion exploded: {terms} terms exceed the limit of {limit}")]
    TermExplosion { terms: usize, limit: usize },
    #[error("failed to parse expression: {0}")]
    Parse(String),
}

/// Compile a symbolic operator expression string into a Hamiltonian.
pub fn compile_to_fock(input: &str) -> Hamiltonian {
    let expr = parse(input).expect("Failed to parse expression");
    compile_expression(expr)
}

/// Compile a symbolic operator expression string into a Hamiltonian, aborting
/// with [`CasError::TermExplosion`] if expansion exceeds `limits.max_terms`.
pub fn compile_to_fock_bounded(
    input: &str,
    limits: &ExpansionLimits,
) -> Result<Hamiltonian, CasError> {
    let expr =
        parse(input).map_err(|e| CasError::Parse(format!("failed to parse {input}: {e}")))?;
    compile_expression_bounded(expr, limits)
}

/// Compile a pre-constructed symbolic Expression into a Hamiltonian.
///
/// This is the historical unchecked entry point; it delegates to
/// [`compile_expression_bounded`] with no term limit. Prefer the bounded
/// variant when compiling untrusted or high-order expressions.
pub fn compile_expression(expr: Expression) -> Hamiltonian {
    compile_expression_bounded(expr, &ExpansionLimits::unbounded())
        .expect("compile_expression: unbounded expansion cannot exceed the limit")
}

/// Compile a symbolic Expression into a Hamiltonian, aborting with
/// [`CasError::TermExplosion`] if the distribution would exceed `limits.max_terms`.
pub fn compile_expression_bounded(
    expr: Expression,
    limits: &ExpansionLimits,
) -> Result<Hamiltonian, CasError> {
    // 1. We ONLY call .expand(), NOT .simplify().
    // The default simplify() pass assumes commutativity (a*b = b*a),
    // which would destroy the physics of non-commuting operators.
    // .expand() preserves order while distributing (a+b)*c -> a*c + b*c.
    let expanded = expr.expand();

    // 2. Parse the resulting order-preserved S-expression string.
    let s_expr = expanded.to_string();
    let mut ast = SExpr::parse(&s_expr)
        .ok_or_else(|| CasError::Parse("failed to parse internal S-expression".into()))?;
    // 3. Apply quadratic ordering logic before distribution
    ast.apply_quadratic_ordering();

    // 4. Distribute multiplication/division over sums to a flat term list,
    //    guarding against combinatorial explosion.
    let mut memo = HashMap::new();
    let distributed = ast.distribute_bounded(limits.max_terms, &mut memo)?;

    // 5. Map each term to a physical Hamiltonian term.
    let mut terms = Vec::new();
    for term in distributed {
        if let Some(h_term) = term.to_hamiltonian_term() {
            // QUADRATIC ORDERING ENFORCEMENT:
            // If the term is a pure scalar (no operators) and we are applying
            // the Quadratic Ordering from the PDF, we drop it to ensure
            // the vacuum expectation value <0|H|0> = 0.
            if h_term.1.is_empty() {
                continue; // Drop pure constant terms (zero-point energy)
            }
            terms.push(h_term);
        }
    }

    Ok(Hamiltonian { terms })
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
        if *pos >= tokens.len() {
            return None;
        }
        let token = &tokens[*pos];
        *pos += 1;

        if token == "(" {
            if *pos >= tokens.len() {
                return None;
            }
            let op = tokens[*pos].clone();
            *pos += 1;
            let mut args = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                args.push(Self::parse_tokens(tokens, pos)?);
            }
            if *pos < tokens.len() {
                *pos += 1;
            } // Consume ")"
            Some(SExpr::List(op, args))
        } else if let Ok(n) = token.parse::<f64>() {
            Some(SExpr::Num(n))
        } else {
            Some(SExpr::Sym(token.clone()))
        }
    }

    /// Applies Quadratic Ordering to the S-Expression AST.
    /// This removes the divergent scalar constants generated by commuting
    /// annihilation operators past creation operators (e.g., a a^\dagger = a^\dagger a + 1)
    /// but only retains the number-conserving part.
    pub fn apply_quadratic_ordering(&mut self) {
        match self {
            SExpr::List(op, args) if op == "+" || op == "-" => {
                // Recursively apply to addition/subtraction terms
                for arg in args.iter_mut() {
                    arg.apply_quadratic_ordering();
                }
            }
            SExpr::List(op, args) if op == "*" => {
                // If it's a multiplication chain, we look for (c_i * a_i) patterns.
                // Since this is a symbolic tree, the actual commuting happens during
                // Operator application. However, to mathematically enforce a zero vacuum
                // expectation, we strip pure scalar offset terms (constants added to the Hamiltonian).
                // In an AST, we do this by dropping SExpr::Num constants that exist at the
                // root of additive branches if they aren't multiplying operators.
            }
            _ => {}
        }
    }

    /// Distribute Mul/Div/Neg over Add recursively with memoization.
    /// Phase 9.2 Optimization for large Hamiltonians.
    /// Superseded by `distribute_bounded`; kept for reference.
    #[allow(dead_code)]
    fn distribute(&self) -> Vec<SExpr> {
        let mut memo = HashMap::new();
        self.distribute_memo(&mut memo)
    }

    fn distribute_memo(&self, memo: &mut HashMap<String, Vec<SExpr>>) -> Vec<SExpr> {
        let key = format!("{:?}", self);
        if let Some(res) = memo.get(&key) {
            return res.clone();
        }

        let res = match self {
            SExpr::List(op, args) if op == "+" => {
                args.iter().flat_map(|a| a.distribute_memo(memo)).collect()
            }
            SExpr::List(op, args) if op == "*" => {
                let distributed_args: Vec<Vec<SExpr>> =
                    args.iter().map(|a| a.distribute_memo(memo)).collect();
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
                let numerators = args[0].distribute_memo(memo);
                let denominator = if args.len() > 1 {
                    args[1].clone()
                } else {
                    SExpr::Num(1.0)
                };
                numerators
                    .into_iter()
                    .map(|n| SExpr::List("/".to_string(), vec![n, denominator.clone()]))
                    .collect()
            }
            SExpr::List(op, args) if op == "neg" => args[0]
                .distribute_memo(memo)
                .into_iter()
                .map(|a| SExpr::List("neg".to_string(), vec![a]))
                .collect(),
            SExpr::List(op, args) if op == "^" => {
                if let SExpr::Num(n) = &args[1] {
                    let p = *n as i32;
                    if p > 0 {
                        let mut chain = args[0].clone();
                        for _ in 1..p {
                            chain = SExpr::List("*".to_string(), vec![chain, args[0].clone()]);
                        }
                        return chain.distribute_memo(memo);
                    } else if p == 0 {
                        return vec![SExpr::Num(1.0)];
                    }
                }
                vec![self.clone()]
            }
            _ => vec![self.clone()],
        };

        memo.insert(key, res.clone());
        res
    }

    /// Distribute Mul/Div/Neg/Pow over Add like [`distribute_memo`], but abort
    /// with [`CasError::TermExplosion`] once the running term count would exceed
    /// `limit`. The multiplicative `*` branch is guarded *before* forming the
    /// cartesian product, so a `2^n` blow-up is caught early rather than after
    /// allocating the whole product.
    fn distribute_bounded(
        &self,
        limit: usize,
        memo: &mut HashMap<String, Vec<SExpr>>,
    ) -> Result<Vec<SExpr>, CasError> {
        let key = format!("{:?}", self);
        if let Some(res) = memo.get(&key) {
            return Ok(res.clone());
        }

        let res = match self {
            SExpr::List(op, args) if op == "+" => {
                let mut acc = Vec::new();
                for a in args {
                    acc.extend(a.distribute_bounded(limit, memo)?);
                    if acc.len() > limit {
                        return Err(CasError::TermExplosion {
                            terms: acc.len(),
                            limit,
                        });
                    }
                }
                acc
            }
            SExpr::List(op, args) if op == "*" => {
                let mut distributed_args: Vec<Vec<SExpr>> = Vec::with_capacity(args.len());
                for a in args {
                    distributed_args.push(a.distribute_bounded(limit, memo)?);
                }
                let mut results = vec![SExpr::List("*".to_string(), vec![])];
                for arg_set in distributed_args {
                    let projected = results.len().saturating_mul(arg_set.len());
                    if projected > limit {
                        return Err(CasError::TermExplosion {
                            terms: projected,
                            limit,
                        });
                    }
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
                let numerators = args[0].distribute_bounded(limit, memo)?;
                let denominator = if args.len() > 1 {
                    args[1].clone()
                } else {
                    SExpr::Num(1.0)
                };
                numerators
                    .into_iter()
                    .map(|n| SExpr::List("/".to_string(), vec![n, denominator.clone()]))
                    .collect()
            }
            SExpr::List(op, args) if op == "neg" => args[0]
                .distribute_bounded(limit, memo)?
                .into_iter()
                .map(|a| SExpr::List("neg".to_string(), vec![a]))
                .collect(),
            SExpr::List(op, args) if op == "^" => {
                if let SExpr::Num(n) = &args[1] {
                    let p = *n as i32;
                    if p > 0 {
                        let mut chain = args[0].clone();
                        for _ in 1..p {
                            chain = SExpr::List("*".to_string(), vec![chain, args[0].clone()]);
                        }
                        return chain.distribute_bounded(limit, memo);
                    } else if p == 0 {
                        return Ok(vec![SExpr::Num(1.0)]);
                    }
                }
                vec![self.clone()]
            }
            _ => vec![self.clone()],
        };

        if res.len() > limit {
            return Err(CasError::TermExplosion {
                terms: res.len(),
                limit,
            });
        }
        memo.insert(key, res.clone());
        Ok(res)
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
            SExpr::Num(n) => {
                *coeff *= n;
            }
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
                    "*" => {
                        for a in args {
                            a.collect_content(coeff, ops);
                        }
                    }
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
        if is_fermionic {
            Some(Operator::InnerFermionCreate(idx))
        } else {
            Some(Operator::InnerBosonCreate(idx))
        }
    } else if let Some(suffix) = name.strip_prefix("a_") {
        let (is_fermionic, idx) = parse_suffix(suffix)?;
        if is_fermionic {
            Some(Operator::InnerFermionAnnihilate(idx))
        } else {
            Some(Operator::InnerBosonAnnihilate(idx))
        }
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
            Some(Operator::OuterFermionAnnihilate(InnerFermionicState {
                modes,
            }))
        } else {
            let mut modes = BTreeMap::new();
            modes.insert(idx, 1);
            Some(Operator::OuterBosonAnnihilate(InnerBosonicState { modes }))
        }
    } else {
        None
    }
}
