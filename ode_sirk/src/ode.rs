use crate::result::{OdeError, OdeSirkResult};

/// A single monomial: coeff * x_0^e0 * x_1^e1 * ...
#[derive(Clone, Debug, PartialEq)]
pub struct Monomial {
    pub coeff: f64,
    pub exponents: Vec<u32>,
}

impl Monomial {
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.coeff
            * self
                .exponents
                .iter()
                .enumerate()
                .map(|(i, &e)| x[i].powi(e as i32))
                .product::<f64>()
    }

    /// Partial derivative ∂/∂x_k: decrements exponent of x_k, multiplies coeff by old exponent.
    pub fn partial_derivative(&self, k: usize) -> Option<Monomial> {
        let e_k = *self.exponents.get(k).unwrap_or(&0);
        if e_k == 0 {
            return None;
        }
        let mut exponents = self.exponents.clone();
        exponents[k] = e_k - 1;
        Some(Monomial {
            coeff: self.coeff * (e_k as f64),
            exponents,
        })
    }
}

/// A polynomial in n variables: sum of monomials.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Polynomial {
    pub terms: Vec<Monomial>,
    pub n_vars: usize,
}

impl Polynomial {
    pub fn zero(n_vars: usize) -> Self {
        Self {
            terms: Vec::new(),
            n_vars,
        }
    }

    pub fn constant(c: f64, n_vars: usize) -> Self {
        Self {
            terms: vec![Monomial {
                coeff: c,
                exponents: vec![0u32; n_vars],
            }],
            n_vars,
        }
    }

    pub fn eval(&self, x: &[f64]) -> f64 {
        self.terms.iter().map(|m| m.eval(x)).sum()
    }

    /// Partial derivative ∂p/∂x_k.
    pub fn partial_derivative(&self, k: usize) -> Polynomial {
        let terms: Vec<Monomial> = self.terms.iter().filter_map(|m| m.partial_derivative(k)).collect();
        Polynomial {
            terms,
            n_vars: self.n_vars,
        }
    }

    /// Combine like terms (same exponent vector).
    pub fn simplify(&mut self) {
        self.terms.retain(|m| m.coeff.abs() > 1e-15);
        self.terms.sort_by(|a, b| a.exponents.cmp(&b.exponents));
        let mut merged: Vec<Monomial> = Vec::new();
        for term in self.terms.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.exponents == term.exponents {
                    last.coeff += term.coeff;
                    continue;
                }
            }
            merged.push(term);
        }
        merged.retain(|m| m.coeff.abs() > 1e-15);
        self.terms = merged;
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|m| m.coeff.abs() < 1e-15)
    }
}

/// A polynomial autonomous ODE system: dx_i/dt = f_i(x) for each i.
#[derive(Clone, Debug)]
pub struct ODESystem {
    pub vars: Vec<String>,
    pub rhs: Vec<Polynomial>,
}

impl ODESystem {
    /// Parse an ODE system from variable names and string RHS expressions.
    ///
    /// Each RHS string is parsed using the `quantrs2_symengine_pure` CAS parser,
    /// expanded, and then decomposed into monomials.  The variable names in `vars`
    /// are mapped to CAS symbols (e.g. `"x"` → symbol `"x"`).
    pub fn parse(vars: Vec<String>, rhs_strs: &[&str]) -> OdeSirkResult<Self> {
        if vars.len() != rhs_strs.len() {
            return Err(OdeError::MismatchedLengths(vars.len(), rhs_strs.len()));
        }

        let var_set: std::collections::HashSet<&str> = vars.iter().map(|s| s.as_str()).collect();

        let rhs: Vec<Polynomial> = rhs_strs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                parse_rhs_via_cas(s, &vars, &var_set).map_err(|e| OdeError::ParseError {
                    pos: i,
                    msg: format!("RHS[{}]: {}", i, e),
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(Self { vars, rhs })
    }

    /// Evaluate all RHS at a point.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        self.rhs.iter().map(|p| p.eval(x)).collect()
    }

    pub fn n_vars(&self) -> usize {
        self.vars.len()
    }
}

/// Parse an RHS string via the CAS, expand, then decompose into monomials.
fn parse_rhs_via_cas(
    input: &str,
    vars: &[String],
    var_set: &std::collections::HashSet<&str>,
) -> Result<Polynomial, String> {
    // Use the CAS parser from quantrs2_symengine_pure
    let expr = quantrs2_symengine_pure::parser::parse(input)
        .map_err(|e| format!("CAS parse error: {}", e))?;
    let expanded = expr.expand();
    let s_expr = expanded.to_string();

    // Parse the S-expression into monomials
    let sexpr = SExpr::parse(&s_expr).ok_or_else(|| format!("failed to parse S-expr: {}", s_expr))?;
    let poly = sexpr_to_polynomial(&sexpr, vars, var_set)?;
    Ok(poly)
}

// ── S-expression → Polynomial decomposition ─────────────────────────

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
        let token = &tokens[*pos].clone();
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
            }
            Some(SExpr::List(op, args))
        } else if let Ok(n) = token.parse::<f64>() {
            Some(SExpr::Num(n))
        } else {
            Some(SExpr::Sym(token.clone()))
        }
    }
}

/// Decompose an expanded S-expression into a Polynomial.
///
/// The expanded S-expression from the CAS has the form:
///   (+ (* coeff sym1 sym2 ...) (* coeff sym1 ...) ...)
/// Each child of `+` is a product (monomial). Symbols that match a variable
/// name contribute an exponent; numeric nodes contribute to the coefficient.
fn sexpr_to_polynomial(
    expr: &SExpr,
    vars: &[String],
    var_set: &std::collections::HashSet<&str>,
) -> Result<Polynomial, String> {
    let n_vars = vars.len();
    let var_index: std::collections::HashMap<&str, usize> = vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i))
        .collect();

    match expr {
        SExpr::Num(n) => Ok(Polynomial {
            terms: vec![Monomial {
                coeff: *n,
                exponents: vec![0u32; n_vars],
            }],
            n_vars,
        }),
        SExpr::Sym(s) => {
            if var_set.contains(s.as_str()) {
                let idx = var_index[s.as_str()];
                let mut exponents = vec![0u32; n_vars];
                exponents[idx] = 1;
                Ok(Polynomial {
                    terms: vec![Monomial {
                        coeff: 1.0,
                        exponents,
                    }],
                    n_vars,
                })
            } else if s == "I" || s == "pi" || s == "e" {
                // Constants — treat as numeric
                let val = match s.as_str() {
                    "pi" => std::f64::consts::PI,
                    "e" => std::f64::consts::E,
                    _ => 1.0, // I (imaginary) should not appear in real ODE RHS
                };
                Ok(Polynomial::constant(val, n_vars))
            } else {
                Err(format!("unknown symbol in ODE RHS: '{}'", s))
            }
        }
        SExpr::List(op, args) => match op.as_str() {
            "+" => {
                let mut result = Polynomial::zero(n_vars);
                for arg in args {
                    let p = sexpr_to_polynomial(arg, vars, var_set)?;
                    result = poly_add(result, p);
                }
                Ok(result)
            }
            "-" => {
                if args.len() == 1 {
                    let p = sexpr_to_polynomial(&args[0], vars, var_set)?;
                    Ok(poly_negate(p))
                } else {
                    let mut result = sexpr_to_polynomial(&args[0], vars, var_set)?;
                    for arg in &args[1..] {
                        let p = sexpr_to_polynomial(arg, vars, var_set)?;
                        result = poly_add(result, poly_negate(p));
                    }
                    Ok(result)
                }
            }
            "*" => {
                let mut result = Polynomial::constant(1.0, n_vars);
                for arg in args {
                    let p = sexpr_to_polynomial(arg, vars, var_set)?;
                    result = poly_mul(result, p);
                }
                Ok(result)
            }
            "^" => {
                if args.len() != 2 {
                    return Err(format!("^ expects 2 args, got {}", args.len()));
                }
                let base = sexpr_to_polynomial(&args[0], vars, var_set)?;
                match &args[1] {
                    SExpr::Num(n) => {
                        let exp = *n as u32;
                        if *n as f64 - *n as f64 != 0.0 || *n < 0.0 {
                            return Err(format!("non-integer or negative exponent: {}", n));
                        }
                        Ok(poly_pow(base, exp))
                    }
                    _ => Err(format!("non-numeric exponent in ODE RHS")),
                }
            }
            "neg" => {
                let p = sexpr_to_polynomial(&args[0], vars, var_set)?;
                Ok(poly_negate(p))
            }
            _ => Err(format!("unsupported CAS operator in ODE RHS: '{}'", op)),
        },
    }
}

fn poly_add(a: Polynomial, b: Polynomial) -> Polynomial {
    let n_vars = a.n_vars.max(b.n_vars);
    let mut terms = a.terms;
    for mut term in b.terms {
        while term.exponents.len() < n_vars {
            term.exponents.push(0);
        }
        terms.push(term);
    }
    let mut result = Polynomial { terms, n_vars };
    result.simplify();
    result
}

fn poly_negate(mut p: Polynomial) -> Polynomial {
    for m in &mut p.terms {
        m.coeff = -m.coeff;
    }
    p
}

fn poly_mul(a: Polynomial, b: Polynomial) -> Polynomial {
    let n_vars = a.n_vars.max(b.n_vars);
    let mut terms = Vec::new();
    for ma in &a.terms {
        for mb in &b.terms {
            let mut exponents = vec![0u32; n_vars];
            for (i, &e) in ma.exponents.iter().enumerate() {
                exponents[i] += e;
            }
            for (i, &e) in mb.exponents.iter().enumerate() {
                exponents[i] += e;
            }
            terms.push(Monomial {
                coeff: ma.coeff * mb.coeff,
                exponents,
            });
        }
    }
    let mut result = Polynomial { terms, n_vars };
    result.simplify();
    result
}

fn poly_pow(base: Polynomial, exp: u32) -> Polynomial {
    if exp == 0 {
        return Polynomial::constant(1.0, base.n_vars);
    }
    let mut result = base.clone();
    for _ in 1..exp {
        result = poly_mul(result, base.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_simple_variable() {
        let v = vars(&["x"]);
        let sys = ODESystem::parse(v, &["x"]).unwrap();
        assert_eq!(sys.n_vars(), 1);
        assert!((sys.eval(&[2.0])[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn parse_x_squared() {
        let v = vars(&["x"]);
        let sys = ODESystem::parse(v, &["x^2"]).unwrap();
        assert!((sys.eval(&[3.0])[0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn parse_coefficient_times_product() {
        let v = vars(&["x", "y"]);
        let sys = ODESystem::parse(v, &["2*x*y", "x"]).unwrap();
        assert!((sys.eval(&[3.0, 4.0])[0] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn parse_sum_and_difference() {
        let v = vars(&["x", "y"]);
        let sys = ODESystem::parse(v, &["x^2 + 3*y - 1", "x"]).unwrap();
        // 4 + 3 - 1 = 6
        assert!((sys.eval(&[2.0, 1.0])[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn parse_negative_coefficient() {
        let v = vars(&["x"]);
        // Use "0 - x^2" since CAS parser treats "-x^2" as "(-x)^2"
        let sys = ODESystem::parse(v, &["0 - x^2"]).unwrap();
        assert!((sys.eval(&[3.0])[0] - (-9.0)).abs() < 1e-12);
    }

    #[test]
    fn partial_derivative_test() {
        let v = vars(&["x", "y"]);
        let sys = ODESystem::parse(v, &["3*x^2*y", "x"]).unwrap();
        // ∂/∂x = 6*x*y
        let dpdx = sys.rhs[0].partial_derivative(0);
        assert!((dpdx.eval(&[2.0, 3.0]) - 36.0).abs() < 1e-12);
        // ∂/∂y = 3*x^2
        let dpdy = sys.rhs[0].partial_derivative(1);
        assert!((dpdy.eval(&[2.0, 3.0]) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn ode_system_coupled() {
        let v = vars(&["x", "y"]);
        let sys = ODESystem::parse(v, &["y", "2*x*y"]).unwrap();
        assert_eq!(sys.n_vars(), 2);
        let f = sys.eval(&[1.0, 3.0]);
        assert!((f[0] - 3.0).abs() < 1e-12);
        assert!((f[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn ode_system_stable_linear() {
        let v = vars(&["x"]);
        let sys = ODESystem::parse(v, &["-x"]).unwrap();
        assert!((sys.eval(&[5.0])[0] - (-5.0)).abs() < 1e-12);
    }
}
