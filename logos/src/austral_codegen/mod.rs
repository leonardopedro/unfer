use crate::core_ir::{CoreIR, Literal, PrimOp, Pattern};

pub struct AustralEmitter {
    output: String,
    indent: usize,
    functions: Vec<String>,
    env_types: Vec<String>,
}

impl AustralEmitter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            functions: Vec::new(),
            env_types: Vec::new(),
        }
    }

    pub fn emit_module(&mut self, term: &CoreIR) -> String {
        self.emit_line("module LogosModule;");
        self.emit_line("");
        self.emit_line("import LogosStd.Memory;");
        self.emit_line("import LogosStd.IO;");
        self.emit_line("");

        let main_fn = self.emit_function("main", &[], term);

        for func in &self.functions {
            self.output.push_str(func);
            self.output.push('\n');
        }

        self.output.push_str(&format!("\nfunction main(): Unit is\n    let result: Int64 = {}();\n    put_int64(result);\n    put_newline();\nend;\n", main_fn));

        self.output.clone()
    }

    fn emit_function(&mut self, name: &str, params: &[(&str, &str)], body: &CoreIR) -> String {
        let mut func = String::new();
        func.push_str(&format!("function {}(", name));
        for (i, (pname, ptype)) in params.iter().enumerate() {
            if i > 0 { func.push_str(", "); }
            func.push_str(&format!("{}: {}", pname, ptype));
        }
        func.push_str("): Int64 is\n");

        let result = self.emit_expr(body, &mut func);
        func.push_str(&format!("    return {};\nend;\n", result));

        self.functions.push(func.clone());
        name.to_string()
    }

    fn emit_expr(&mut self, term: &CoreIR, output: &mut String) -> String {
        match term {
            CoreIR::Lit(Literal::Int64(n)) => format!("{}", n),
            CoreIR::Lit(Literal::Bool(b)) => format!("{}", b),
            CoreIR::Var(id) => id.clone(),
            CoreIR::Prim(op, args) => {
                let left = self.emit_expr(&args[0], output);
                let right = self.emit_expr(&args[1], output);
                match op {
                    PrimOp::Add64 => format!("({} + {})", left, right),
                    PrimOp::Sub64 => format!("({} - {})", left, right),
                    PrimOp::Mul64 => format!("({} * {})", left, right),
                    PrimOp::Eq64 => format!("({} = {})", left, right),
                    PrimOp::Gt64 => format!("({} > {})", left, right),
                    PrimOp::Lt64 => format!("({} < {})", left, right),
                    PrimOp::And => format!("({} and {})", left, right),
                    PrimOp::Or => format!("({} or {})", left, right),
                    PrimOp::Not => format!("(not {})", left),
                }
            }
            CoreIR::Con(tag, args) => {
                if args.is_empty() {
                    format!("Tag{}", tag)
                } else {
                    let arg_strs: Vec<String> = args.iter().map(|a| self.emit_expr(a, output)).collect();
                    format!("Tag{}({})", tag, arg_strs.join(", "))
                }
            }
            CoreIR::App(func, arg) => {
                let func_str = self.emit_expr(func, output);
                let arg_str = self.emit_expr(arg, output);
                format!("{}({})", func_str, arg_str)
            }
            CoreIR::Lam(id, body) => {
                let lambda_name = format!("lambda_{}", self.functions.len());
                let body_str = self.emit_expr(body, output);
                let func_str = format!("function {}({}: Int64): Int64 is\n    return {};\nend;\n", lambda_name, id, body_str);
                self.functions.push(func_str);
                lambda_name
            }
            CoreIR::Let(id, value, body) => {
                let val = self.emit_expr(value, output);
                output.push_str(&format!("    let {}: Int64 = {};\n", id, val));
                self.emit_expr(body, output)
            }
            CoreIR::Fold(f, init, list) => {
                let f_str = self.emit_expr(f, output);
                let init_str = self.emit_expr(init, output);
                let list_str = self.emit_expr(list, output);
                format!("fold({}, {}, {})", f_str, init_str, list_str)
            }
            CoreIR::Clone(id, id1, id2, body) => {
                output.push_str(&format!("    let {}: Int64 = clone {};\n", id1, id));
                output.push_str(&format!("    let {}: Int64 = clone {};\n", id2, id));
                self.emit_expr(body, output)
            }
            CoreIR::Drop(id, body) => {
                output.push_str(&format!("    destroy {};\n", id));
                self.emit_expr(body, output)
            }
            CoreIR::Match(scrutinee, arms) => {
                let scrutinee_str = self.emit_expr(scrutinee, output);
                let mut result = String::new();
                for (i, (pat, body)) in arms.iter().enumerate() {
                    match pat {
                        Pattern::Tag(tag, binders) => {
                            let cond = if binders.is_empty() {
                                format!("{} = Tag{}", scrutinee_str, tag)
                            } else {
                                format!("{} matches Tag{}", scrutinee_str, tag)
                            };
                            if i == 0 {
                                result.push_str(&format!("if {} then\n", cond));
                            } else {
                                result.push_str(&format!("elsif {} then\n", cond));
                            }
                            let body_str = self.emit_expr(body, output);
                            result.push_str(&format!("    {}\n", body_str));
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

pub fn emit_austral(term: &CoreIR) -> String {
    let mut emitter = AustralEmitter::new();
    emitter.emit_module(term)
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
}
