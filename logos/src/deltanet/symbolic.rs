//! Structured symbolic readback of a reduced interaction net.
//!
//! [`readback`](crate::deltanet::readback) renders the reduced net as a
//! string. This module reads the same net back as a **symbolic expression
//! tree** ([`SymExpr`]) whose leaves are either concrete literals or
//! *unknowns* ([`SymExpr::Var`], from `Entity` agents — free variables that
//! the reduction could not resolve). Numeric primitives (`Add64`, `MulF64`,
//! …) remain visible as *operations on the tree*, so:
//!
//!   • a **closed** term (no unknowns) reduces to a literal — [`SymExpr::eval`]
//!     returns the numerical result (e.g. `ADD` of two 64-bit integers → `5`);
//!   • an **open** term (an unknown `x`) reduces to a tree such as
//!     `Add64(x, 3)` — a symbolic expression involving a numerical
//!     calculation, with `eval()` returning `None` until `x` is known.
//!
//! This is the "replace the normal form with a symbolic expression involving
//! numerical calculations, whenever there are no unknown variables" step of
//! the austral/deltanet translation.

use crate::core_ir::{Literal, PrimOp, TagId};
use std::collections::HashSet;

use super::types::{AgentKind, Net, NodeId, Port};

/// A symbolic expression read back from a reduced interaction net.
#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    /// A concrete literal (`Int64`, `F64`, `Bool`).
    Lit(Literal),
    /// An unknown: an `Entity` agent that reduction never resolved (a free
    /// variable, or an unbound function name). `eval` refuses closedness.
    Var(String),
    /// A numerical/boolean calculation: `Add64`, `MulF64`, `Lt64`, `And`, …
    Prim(PrimOp, Box<SymExpr>, Box<SymExpr>),
    /// Application `f x` (function unknown → open).
    App(Box<SymExpr>, Box<SymExpr>),
    /// Constructor application `TagN(arg, …)`.
    Con(TagId, Vec<SymExpr>),
    /// An irreducible function value `λx. body` (a higher-order remainder).
    Abs(String, Box<SymExpr>),
    /// An irreducible fold `fold f init list`.
    Fold(Box<SymExpr>, Box<SymExpr>, Box<SymExpr>),
    /// Any other irreducible remainder (`Dup`/`Era` leftovers, stuck
    /// constructor pairs). Carries a short description.
    Stuck(String),
}

impl SymExpr {
    /// True iff the expression contains no unknowns ([`SymExpr::Var`]).
    /// A closed expression is a fully determined normal form.
    pub fn is_closed(&self) -> bool {
        match self {
            SymExpr::Lit(_) => true,
            SymExpr::Var(_) => false,
            SymExpr::Prim(_, a, b) => a.is_closed() && b.is_closed(),
            SymExpr::App(f, a) => f.is_closed() && a.is_closed(),
            SymExpr::Con(_, args) => args.iter().all(SymExpr::is_closed),
            SymExpr::Abs(_, body) => body.is_closed(),
            SymExpr::Fold(f, i, l) => f.is_closed() && i.is_closed() && l.is_closed(),
            SymExpr::Stuck(_) => false,
        }
    }

    /// The numerical/boolean result of the calculation, when the expression
    /// is closed and purely literal-arithmetic: `ADD(2, 3)` → `Some(5)`.
    ///
    /// After full reduction a closed numeric term is a single literal at the
    /// root, but this is a structural fallback: it also evaluates closed
    /// trees that for any reason were not collapsed by the reducer.
    pub fn eval(&self) -> Option<Literal> {
        match self {
            SymExpr::Lit(l) => Some(l.clone()),
            SymExpr::Prim(op, a, b) => {
                let la = a.eval()?;
                let lb = b.eval()?;
                eval_prim(*op, &la, &lb)
            }
            _ => None,
        }
    }

    /// Prefix display, stable and canonical: `(Add64 2 3)`, `(Add64 x 3)`.
    pub fn to_prefix_string(&self) -> String {
        match self {
            SymExpr::Lit(l) => l.to_string(),
            SymExpr::Var(name) => name.clone(),
            SymExpr::Prim(op, a, b) => {
                format!("({:?} {} {})", op, a.to_prefix_string(), b.to_prefix_string())
            }
            SymExpr::App(f, a) => {
                format!("({} {})", f.to_prefix_string(), a.to_prefix_string())
            }
            SymExpr::Con(tag, args) => {
                if args.is_empty() {
                    format!("Tag{tag}")
                } else {
                    let inner: Vec<String> =
                        args.iter().map(SymExpr::to_prefix_string).collect();
                    format!("Tag{}({})", tag, inner.join(", "))
                }
            }
            SymExpr::Abs(binder, body) => {
                format!("(lam {binder} {})", body.to_prefix_string())
            }
            SymExpr::Fold(f, i, l) => format!(
                "(fold {} {} {})",
                f.to_prefix_string(),
                i.to_prefix_string(),
                l.to_prefix_string()
            ),
            SymExpr::Stuck(what) => format!("<{what}>"),
        }
    }

    /// Infix display for the arithmetic fragment: `(2 + 3)`, `(x + 3)`,
    /// `(not true)`. Constructors/application fall back to the prefix form.
    pub fn to_infix_string(&self) -> String {
        match self {
            SymExpr::Lit(l) => l.to_string(),
            SymExpr::Var(name) => name.clone(),
            SymExpr::Prim(op, a, b) => {
                let op_str = match op {
                    PrimOp::Add64 | PrimOp::AddF64 => "+",
                    PrimOp::Sub64 | PrimOp::SubF64 => "-",
                    PrimOp::Mul64 | PrimOp::MulF64 => "*",
                    PrimOp::DivF64 => "/",
                    PrimOp::Eq64 | PrimOp::EqF64 => "=",
                    PrimOp::Gt64 | PrimOp::GtF64 => ">",
                    PrimOp::Lt64 | PrimOp::LtF64 => "<",
                    PrimOp::And => "and",
                    PrimOp::Or => "or",
                    PrimOp::Not => "not",
                };
                if *op == PrimOp::Not {
                    format!("(not {})", a.to_infix_string())
                } else {
                    format!(
                        "({} {op_str} {})",
                        a.to_infix_string(),
                        b.to_infix_string()
                    )
                }
            }
            _ => self.to_prefix_string(),
        }
    }
}

impl std::fmt::Display for SymExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_prefix_string())
    }
}

/// Read the reduced net back as a symbolic expression tree.
///
/// Mirrors [`super::readback::readback`]'s traversal (following freed-node
/// port targets, cycle-guarding with a visited set) but builds the
/// structured tree instead of a string.
pub fn sym_readback(net: &Net) -> Result<SymExpr, String> {
    let mut visited = HashSet::new();
    readback_port(net, &net.root, &mut visited, 0)
}

fn readback_port(
    net: &Net,
    port: &Port,
    visited: &mut HashSet<(NodeId, u8)>,
    depth: usize,
) -> Result<SymExpr, String> {
    let key = (port.node, port.slot);
    if !visited.insert(key) {
        return Ok(SymExpr::Stuck("cycle".to_string()));
    }
    if depth > 1000 {
        return Ok(SymExpr::Stuck("depth".to_string()));
    }

    let node = match net.nodes.get(port.node as usize) {
        Some(Some(n)) => n,
        _ => return Ok(SymExpr::Stuck("freed".to_string())),
    };

    if node.freed {
        if let Some(target) = &node.ports[port.slot as usize] {
            return readback_port(net, target, visited, depth + 1);
        }
        return Ok(SymExpr::Stuck("freed".to_string()));
    }

    match &node.kind {
        AgentKind::Lit(lit) => Ok(SymExpr::Lit(lit.clone())),
        AgentKind::Entity(name) => Ok(SymExpr::Var(name.clone())),
        AgentKind::Con(tag, arity) => {
            let mut args = Vec::with_capacity(*arity as usize);
            for s in 1..=*arity {
                args.push(readback_port(net, &net.get_aux(port.node, s)?, visited, depth + 1)?);
            }
            Ok(SymExpr::Con(*tag, args))
        }
        AgentKind::App => {
            let f = readback_port(net, &net.get_aux(port.node, 0)?, visited, depth + 1)?;
            let a = readback_port(net, &net.get_aux(port.node, 1)?, visited, depth + 1)?;
            Ok(SymExpr::App(Box::new(f), Box::new(a)))
        }
        AgentKind::Abs => {
            // The net does not retain binder names; synthesize one.
            let body = readback_port(net, &net.get_aux(port.node, 2)?, visited, depth + 1)?;
            Ok(SymExpr::Abs(format!("x{}", port.node), Box::new(body)))
        }
        AgentKind::Fold => {
            let f = readback_port(net, &net.get_aux(port.node, 1)?, visited, depth + 1)?;
            let i = readback_port(net, &net.get_aux(port.node, 2)?, visited, depth + 1)?;
            let l = readback_port(net, &net.get_aux(port.node, 3)?, visited, depth + 1)?;
            Ok(SymExpr::Fold(Box::new(f), Box::new(i), Box::new(l)))
        }
        AgentKind::Prim(op) => {
            let a = readback_port(net, &net.get_aux(port.node, 1)?, visited, depth + 1)?;
            let b = readback_port(net, &net.get_aux(port.node, 2)?, visited, depth + 1)?;
            Ok(SymExpr::Prim(*op, Box::new(a), Box::new(b)))
        }
        AgentKind::Dup(_) => Ok(SymExpr::Stuck("dup".to_string())),
        AgentKind::Era => Ok(SymExpr::Stuck("era".to_string())),
    }
}

/// Mirror of the reducer's literal evaluation, so `SymExpr::eval` is
/// self-contained (no net needed).
pub(crate) fn eval_prim(op: PrimOp, a: &Literal, b: &Literal) -> Option<Literal> {
    match (op, a, b) {
        (PrimOp::Add64, Literal::Int64(x), Literal::Int64(y)) => {
            Some(Literal::Int64(x.wrapping_add(*y)))
        }
        (PrimOp::Sub64, Literal::Int64(x), Literal::Int64(y)) => {
            Some(Literal::Int64(x.wrapping_sub(*y)))
        }
        (PrimOp::Mul64, Literal::Int64(x), Literal::Int64(y)) => {
            Some(Literal::Int64(x.wrapping_mul(*y)))
        }
        (PrimOp::Eq64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x == y)),
        (PrimOp::Gt64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x > y)),
        (PrimOp::Lt64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x < y)),
        (PrimOp::And, Literal::Bool(x), Literal::Bool(y)) => Some(Literal::Bool(*x && *y)),
        (PrimOp::Or, Literal::Bool(x), Literal::Bool(y)) => Some(Literal::Bool(*x || *y)),
        (PrimOp::Not, Literal::Bool(x), _) => Some(Literal::Bool(!x)),
        (PrimOp::AddF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::F64(x + y)),
        (PrimOp::SubF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::F64(x - y)),
        (PrimOp::MulF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::F64(x * y)),
        (PrimOp::DivF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::F64(x / y)),
        (PrimOp::EqF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::Bool(x == y)),
        (PrimOp::GtF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::Bool(x > y)),
        (PrimOp::LtF64, Literal::F64(x), Literal::F64(y)) => Some(Literal::Bool(x < y)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::{CoreIR, Literal};

    fn reduced(ir: &CoreIR) -> Net {
        let mut net = super::super::compiler::compile_to_net(ir).unwrap();
        super::super::reducer::reduce(&mut net).unwrap();
        net
    }

    #[test]
    fn closed_arithmetic_evaluates_to_literal() {
        // (2 + 3) * 4 = 20 — the reducer collapses the whole closed tree.
        let ir = CoreIR::Prim(
            PrimOp::Mul64,
            vec![
                CoreIR::Prim(
                    PrimOp::Add64,
                    vec![
                        CoreIR::Lit(Literal::Int64(2)),
                        CoreIR::Lit(Literal::Int64(3)),
                    ],
                ),
                CoreIR::Lit(Literal::Int64(4)),
            ],
        );
        let net = reduced(&ir);
        let sym = sym_readback(&net).unwrap();
        assert!(sym.is_closed(), "closed term must be closed: {sym}");
        assert_eq!(sym.eval(), Some(Literal::Int64(20)));
        assert_eq!(sym.to_prefix_string(), "20");
    }

    #[test]
    fn open_arithmetic_stays_symbolic_with_add() {
        // x + 3 with x an unknown Entity: the ADD of two 64-bit ints remains
        // a visible symbolic operation; no value until x is known.
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(3))],
        );
        let net = reduced(&ir);
        let sym = sym_readback(&net).unwrap();
        assert!(!sym.is_closed(), "open term must not be closed: {sym}");
        assert_eq!(sym.eval(), None);
        assert_eq!(sym.to_prefix_string(), "(Add64 x 3)");
        assert_eq!(sym.to_infix_string(), "(x + 3)");
    }

    #[test]
    fn f64_and_i64_never_coerce() {
        let ir = CoreIR::Prim(
            PrimOp::AddF64,
            vec![CoreIR::Lit(Literal::F64(1.0)), CoreIR::Lit(Literal::Int64(2))],
        );
        let net = reduced(&ir);
        let sym = sym_readback(&net).unwrap();
        // The type-mismatched fold is stuck (no silent coercion), so it is
        // not a closed numeric value.
        assert_eq!(sym.eval(), None);
    }

    #[test]
    fn bool_calculation_evaluates() {
        let ir = CoreIR::Prim(
            PrimOp::Gt64,
            vec![CoreIR::Lit(Literal::Int64(5)), CoreIR::Lit(Literal::Int64(2))],
        );
        let net = reduced(&ir);
        let sym = sym_readback(&net).unwrap();
        assert_eq!(sym.eval(), Some(Literal::Bool(true)));
    }
}
