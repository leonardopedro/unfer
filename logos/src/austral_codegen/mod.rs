//! Austral backend for CoreIR (the "austral/australVM language" side of the
//! translation).
//!
//! Two directions meet here:
//!
//!   • [`emit_austral`] renders a CoreIR term as Austral source (the
//!     CNL→CoreIR→Austral direction, driving the australVM JIT);
//!   • [`parse_austral_expr`] / [`parse_austral_function`] lower Austral
//!     source back to CoreIR (the austral→UNF direction, feeding the
//!     deltanet reducer).
//!
//! [`emit_austral_reduced`] runs the term through the deltanet reducer first
//! and constant-folds every closed subexpression to a literal, so the
//! emitted Austral carries the *numerical result* (e.g. `(2 + 3)` is emitted
//! as `5`) instead of the unreduced computation — the "replace the normal
//! form with a symbolic expression involving numerical calculations"
//! direction.

pub mod parser;
pub mod validate;

pub use parser::{
    parse_austral_expr, parse_austral_function, parse_austral_module, parse_austral_statements,
};
pub use validate::{validate_body, validate_module};

use crate::core_ir::{CoreIR, Literal, PrimOp};
use crate::deltanet;

pub struct AustralEmitter {
    output: String,
    functions: Vec<String>,
}

impl Default for AustralEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl AustralEmitter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            functions: Vec::new(),
        }
    }

    pub fn emit_module(&mut self, term: &CoreIR) -> String {
        self.emit_line("module LogosModule;");
        self.emit_line("");
        self.emit_line("import LogosStd.Memory;");
        self.emit_line("import LogosStd.IO;");
        self.emit_line("");

        // The computation is a single `logos_main(): Int64` function; the
        // `main(): Unit` entry below calls it and prints the result. The
        // helper functions (including `logos_main` itself) are accumulated in
        // `self.functions` during `emit_expr` and emitted once here — no
        // duplicate `main`.
        let main_fn = self.emit_function("logos_main", &[], term);

        for func in &self.functions {
            self.output.push_str(func);
            self.output.push('\n');
        }

        self.output.push_str(&format!(
            "\nfunction main(): Unit is\n    let result: Int64 = {}();\n    put_int64(result);\n    put_newline();\nend;\n",
            main_fn
        ));

        self.output.clone()
    }

    fn emit_function(&mut self, name: &str, params: &[(&str, &str)], body: &CoreIR) -> String {
        let mut func = String::new();
        func.push_str(&format!("function {name}("));
        for (i, (pname, ptype)) in params.iter().enumerate() {
            if i > 0 {
                func.push_str(", ");
            }
            func.push_str(&format!("{pname}: {ptype}"));
        }
        func.push_str("): Int64 is\n");

        let result = self.emit_expr(body, &mut func);
        func.push_str(&format!("    return {result};\nend;\n"));

        self.functions.push(func.clone());
        name.to_string()
    }

    fn emit_expr(&mut self, term: &CoreIR, output: &mut String) -> String {
        match term {
            CoreIR::Lit(Literal::Int64(n)) => format!("{n}"),
            CoreIR::Lit(Literal::F64(x)) => format!("{x}"),
            CoreIR::Lit(Literal::Bool(b)) => format!("{b}"),
            CoreIR::Var(id) => id.clone(),
            CoreIR::Prim(op, args) => {
                let left = self.emit_expr(&args[0], output);
                let right = self.emit_expr(&args[1], output);
                match op {
                    PrimOp::Add64 => format!("({left} + {right})"),
                    PrimOp::Sub64 => format!("({left} - {right})"),
                    PrimOp::Mul64 => format!("({left} * {right})"),
                    PrimOp::Eq64 => format!("({left} = {right})"),
                    PrimOp::Gt64 => format!("({left} > {right})"),
                    PrimOp::Lt64 => format!("({left} < {right})"),
                    PrimOp::AddF64 => format!("({left} + {right})"),
                    PrimOp::SubF64 => format!("({left} - {right})"),
                    PrimOp::MulF64 => format!("({left} * {right})"),
                    PrimOp::DivF64 => format!("({left} / {right})"),
                    PrimOp::EqF64 => format!("({left} = {right})"),
                    PrimOp::GtF64 => format!("({left} > {right})"),
                    PrimOp::LtF64 => format!("({left} < {right})"),
                    PrimOp::And => format!("({left} and {right})"),
                    PrimOp::Or => format!("({left} or {right})"),
                    PrimOp::Not => format!("(not {left})"),
                }
            }
            CoreIR::Con(tag, args) => {
                if args.is_empty() {
                    format!("Tag{tag}")
                } else {
                    let arg_strs: Vec<String> =
                        args.iter().map(|a| self.emit_expr(a, output)).collect();
                    format!("Tag{tag}({})", arg_strs.join(", "))
                }
            }
            CoreIR::App(func, arg) => {
                let func_str = self.emit_expr(func, output);
                let arg_str = self.emit_expr(arg, output);
                format!("{func_str}({arg_str})")
            }
            CoreIR::Lam(id, body) => {
                let lambda_name = format!("lambda_{}", self.functions.len());
                let body_str = self.emit_expr(body, output);
                let func_str = format!(
                    "function {lambda_name}({id}: Int64): Int64 is\n    return {body_str};\nend;\n"
                );
                self.functions.push(func_str);
                lambda_name
            }
            CoreIR::Let(id, value, body) => {
                let val = self.emit_expr(value, output);
                output.push_str(&format!("    let {id}: Int64 = {val};\n"));
                self.emit_expr(body, output)
            }
            CoreIR::Fold(f, init, list) => {
                let f_str = self.emit_expr(f, output);
                let init_str = self.emit_expr(init, output);
                let list_str = self.emit_expr(list, output);
                format!("fold({f_str}, {init_str}, {list_str})")
            }
            CoreIR::Clone(id, id1, id2, body) => {
                output.push_str(&format!("    let {id1}: Int64 = clone {id};\n"));
                output.push_str(&format!("    let {id2}: Int64 = clone {id};\n"));
                self.emit_expr(body, output)
            }
            CoreIR::Drop(id, body) => {
                output.push_str(&format!("    destroy {id};\n"));
                self.emit_expr(body, output)
            }
            CoreIR::Match(scrutinee, arms) => {
                let scrutinee_str = self.emit_expr(scrutinee, output);
                let mut result = String::new();
                for (i, (pat, body)) in arms.iter().enumerate() {
                    match pat {
                        crate::core_ir::Pattern::Tag(tag, binders) => {
                            let cond = if binders.is_empty() {
                                format!("{scrutinee_str} = Tag{tag}")
                            } else {
                                format!("{scrutinee_str} matches Tag{tag}")
                            };
                            if i == 0 {
                                result.push_str(&format!("if {cond} then\n"));
                            } else {
                                result.push_str(&format!("elsif {cond} then\n"));
                            }
                            let body_str = self.emit_expr(body, output);
                            result.push_str(&format!("    {body_str}\n"));
                        }
                    }
                }
                result.push_str("end;\n");
                result
            }
        }
    }

    fn emit_line(&mut self, line: &str) {
        self.output.push_str(line);
        self.output.push('\n');
    }
}

/// Render a CoreIR term as an Austral module.
pub fn emit_austral(term: &CoreIR) -> String {
    let mut emitter = AustralEmitter::new();
    emitter.emit_module(term)
}

/// Constant-fold a closed CoreIR term through the deltanet reducer.
///
/// Walks the term; whenever a `Prim` node's (already-folded) children are
/// both literals, compiles the tiny `Prim(lit, lit)` subterm to an
/// interaction net, reduces it, and substitutes the resulting literal. This
/// is the "reduce through DeltaNets and replace with the numerical
/// calculation" step applied per subexpression — `(2 + 3)` becomes `5`,
/// while an open subexpression `(x + 3)` is untouched and stays symbolic.
pub fn fold_closed_through_deltanet(term: &CoreIR) -> CoreIR {
    match term {
        CoreIR::Prim(op, args) => {
            let args: Vec<CoreIR> = args.iter().map(fold_closed_through_deltanet).collect();
            if args.len() == 2
                && let (CoreIR::Lit(a), CoreIR::Lit(b)) = (&args[0], &args[1])
                && let Some(lit) = deltanet::symbolic::eval_prim(*op, a, b)
            {
                return CoreIR::Lit(lit);
            }
            CoreIR::Prim(*op, args)
        }
        CoreIR::Con(tag, args) => CoreIR::Con(
            *tag,
            args.iter().map(fold_closed_through_deltanet).collect(),
        ),
        CoreIR::App(f, a) => CoreIR::App(
            Box::new(fold_closed_through_deltanet(f)),
            Box::new(fold_closed_through_deltanet(a)),
        ),
        CoreIR::Lam(id, body) => {
            CoreIR::Lam(id.clone(), Box::new(fold_closed_through_deltanet(body)))
        }
        CoreIR::Let(id, value, body) => CoreIR::Let(
            id.clone(),
            Box::new(fold_closed_through_deltanet(value)),
            Box::new(fold_closed_through_deltanet(body)),
        ),
        CoreIR::Match(s, arms) => CoreIR::Match(
            Box::new(fold_closed_through_deltanet(s)),
            arms.iter()
                .map(|(p, b)| (p.clone(), fold_closed_through_deltanet(b)))
                .collect(),
        ),
        CoreIR::Fold(f, i, l) => CoreIR::Fold(
            Box::new(fold_closed_through_deltanet(f)),
            Box::new(fold_closed_through_deltanet(i)),
            Box::new(fold_closed_through_deltanet(l)),
        ),
        CoreIR::Clone(id, id1, id2, body) => CoreIR::Clone(
            id.clone(),
            id1.clone(),
            id2.clone(),
            Box::new(fold_closed_through_deltanet(body)),
        ),
        CoreIR::Drop(id, body) => {
            CoreIR::Drop(id.clone(), Box::new(fold_closed_through_deltanet(body)))
        }
        CoreIR::Var(_) | CoreIR::Lit(_) => term.clone(),
    }
}

/// Emit Austral with every closed subexpression reduced to its numerical
/// value first (via [`fold_closed_through_deltanet`]).
pub fn emit_austral_reduced(term: &CoreIR) -> String {
    emit_austral(&fold_closed_through_deltanet(term))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_literal() {
        let ir = CoreIR::Lit(Literal::Int64(42));
        let result = emit_austral(&ir);
        assert!(result.contains("42"));
    }

    #[test]
    fn test_emit_prim() {
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![
                CoreIR::Lit(Literal::Int64(2)),
                CoreIR::Lit(Literal::Int64(3)),
            ],
        );
        let result = emit_austral(&ir);
        assert!(result.contains("+"));
    }

    #[test]
    fn test_emit_module_has_single_main() {
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![
                CoreIR::Lit(Literal::Int64(2)),
                CoreIR::Lit(Literal::Int64(3)),
            ],
        );
        let result = emit_austral(&ir);
        let mains = result.matches("function main").count();
        assert_eq!(mains, 1, "module must define main exactly once:\n{result}");
        assert!(result.contains("logos_main"));
    }

    #[test]
    fn test_fold_closed_reduces_to_literal() {
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
        let folded = fold_closed_through_deltanet(&ir);
        assert_eq!(folded, CoreIR::Lit(Literal::Int64(20)));
    }

    #[test]
    fn test_fold_closed_keeps_open_subterms_symbolic() {
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(3))],
        );
        let folded = fold_closed_through_deltanet(&ir);
        assert_eq!(
            folded,
            CoreIR::Prim(
                PrimOp::Add64,
                vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(3))],
            )
        );
    }

    #[test]
    fn test_emit_reduced_contains_folded_literal() {
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![
                CoreIR::Lit(Literal::Int64(2)),
                CoreIR::Lit(Literal::Int64(3)),
            ],
        );
        let reduced = emit_austral_reduced(&ir);
        // `(2 + 3)` must have been replaced by `5` in the emitted Austral.
        assert!(
            !reduced.contains("(2 + 3)"),
            "unreduced arithmetic emitted:\n{reduced}"
        );
        assert!(
            reduced.contains("return 5;"),
            "expected folded literal:\n{reduced}"
        );
    }

    #[test]
    fn test_round_trip_parse_emit() {
        // CoreIR → Austral → parse → CoreIR: the emitter dialect parses back.
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(1))],
        );
        let body = emit_austral_function_body(&ir);
        let parsed = parse_austral_statements(&body).unwrap();
        assert_eq!(parsed, ir);
    }

    fn emit_austral_function_body(term: &CoreIR) -> String {
        // Render just the statement body the emitter produces inside a
        // function: `let`s then `return <expr>;`.
        let mut emitter = AustralEmitter::new();
        let mut func = String::new();
        let result = emitter.emit_expr(term, &mut func);
        func.push_str(&format!("    return {result};\n"));
        func
    }
}
