//! Austral → CoreIR lowering: the reverse direction of [`super::emit_austral`].
//!
//! The emitter turns a CoreIR term into Austral source; this parser turns the
//! emitted (or hand-written) Austral subset back into CoreIR, closing the
//! round trip `CoreIR → Austral → CoreIR → deltanet UNF`. The subset covers
//! exactly what the emitter produces for the L0 fragment — parenthesized
//! arithmetic (`(2 + 3) * 4`), boolean connectives (`and`, `or`, `not`),
//! comparisons (`=`, `>`, `<`), constructor terms (`TagN(...)`), single-
//! argument calls, `let`-bound locals and `function NAME(P: T): T is ... end`
//! bodies — so the translation between the austral/australVM language and the
//! unique normal form can be driven from Austral source alone.
//!
//! Unknown identifiers lower to [`CoreIR::Var`], which the deltanet compiler
//! renders as `Entity` agents: they survive reduction as symbolic unknowns
//! (see `deltanet::symbolic`).

use crate::core_ir::{CoreIR, Literal, PrimOp};

/// Parse a single Austral expression (no trailing semicolon).
pub fn parse_austral_expr(src: &str) -> Result<CoreIR, String> {
    let mut p = Parser::new(src);
    p.skip_ws();
    let e = p.parse_expr()?;
    p.skip_ws();
    if !p.at_end() {
        return Err(format!(
            "trailing tokens in Austral expression at byte {}: `{}`",
            p.pos,
            p.rest_preview()
        ));
    }
    Ok(e)
}

/// Parse a statement list (`let x: T = e;` … `return e;`) — the body shape
/// the emitter produces inside a function — into a CoreIR `Let` chain
/// ending in the returned expression.
pub fn parse_austral_statements(src: &str) -> Result<CoreIR, String> {
    let mut p = Parser::new(src);
    // Statements in program order; folded into a chain from the inside out.
    let mut stmts: Vec<Stmt> = Vec::new();
    let mut result: Option<CoreIR> = None;
    loop {
        p.skip_ws();
        if p.at_end() {
            break;
        }
        if p.starts_with("return") {
            p.bump_word("return");
            p.skip_ws();
            let e = p.parse_expr()?;
            p.skip_ws();
            p.expect_char(';')?;
            p.skip_ws();
            result = Some(e);
            break;
        }
        if p.starts_with("let") {
            p.bump_word("let");
            p.skip_ws();
            let id = p.parse_ident()?;
            p.skip_ws();
            p.expect_char(':')?;
            // Type annotation: skip to the `=` (types are not carried in
            // CoreIR).
            while !p.at_end() && p.peek() != '=' {
                p.bump();
            }
            p.expect_char('=')?;
            p.skip_ws();
            let value = p.parse_expr()?;
            p.skip_ws();
            p.expect_char(';')?;
            stmts.push(Stmt::Let(id, value));
            continue;
        }
        // Explicit replicator: `clone x -> y, z;` — the linearity escape
        // hatch that makes two uses of a value legal.
        if p.starts_with("clone") {
            p.bump_word("clone");
            p.skip_ws();
            let id = p.parse_ident()?;
            p.skip_ws();
            // Arrow: `->` (optionally spaced).
            p.expect_char('-')?;
            p.expect_char('>')?;
            p.skip_ws();
            let c1 = p.parse_ident()?;
            p.skip_ws();
            p.expect_char(',')?;
            p.skip_ws();
            let c2 = p.parse_ident()?;
            p.skip_ws();
            p.expect_char(';')?;
            stmts.push(Stmt::Clone(id, c1, c2));
            continue;
        }
        // Explicit eraser: `destroy x;` — consumes the variable without a
        // use, closing the linearity book.
        if p.starts_with("destroy") {
            p.bump_word("destroy");
            p.skip_ws();
            let id = p.parse_ident()?;
            p.skip_ws();
            p.expect_char(';')?;
            stmts.push(Stmt::Drop(id));
            continue;
        }
        return Err(format!(
            "expected `let`, `clone`, `destroy`, or `return` statement at byte {}: `{}`",
            p.pos,
            p.rest_preview()
        ));
    }
    let mut term = result.ok_or_else(|| "empty statement list (no `return` found)".to_string())?;
    // Fold the statements from the inside out: the first statement is the
    // outermost binder — `let x = 1; clone x -> y, z; return e;` becomes
    // Let(x, 1, Clone(x, y, z, e)).
    for stmt in stmts.into_iter().rev() {
        term = match stmt {
            Stmt::Let(id, value) => CoreIR::Let(id, Box::new(value), Box::new(term)),
            Stmt::Clone(id, c1, c2) => CoreIR::Clone(id, c1, c2, Box::new(term)),
            Stmt::Drop(id) => CoreIR::Drop(id, Box::new(term)),
        };
    }
    Ok(term)
}

/// One statement of the subset's statement list, in program order.
#[derive(Debug, Clone)]
enum Stmt {
    Let(String, CoreIR),
    Clone(String, String, String),
    Drop(String),
}

/// Parse a `function NAME(P: T, ...): T is BODY end;` declaration into a
/// CoreIR lambda over its parameters (curried), with `BODY` the statement
/// list. Returns `(name, CoreIR)`. The trailing `end;` is tolerated (not
/// consumed).
pub fn parse_austral_function(src: &str) -> Result<(String, CoreIR), String> {
    let mut p = Parser::new(src);
    p.skip_ws();
    if !p.starts_with("function") {
        return Err(format!(
            "expected `function`, got `{}`",
            p.rest_preview()
        ));
    }
    p.bump_word("function");
    p.skip_ws();
    let name = p.parse_ident()?;
    p.skip_ws();
    p.expect_char('(')?;
    let mut params: Vec<String> = Vec::new();
    p.skip_ws();
    if p.peek() != ')' {
        loop {
            let id = p.parse_ident()?;
            p.skip_ws();
            p.expect_char(':')?;
            while !p.at_end() && p.peek() != ',' && p.peek() != ')' {
                p.bump();
            }
            params.push(id);
            p.skip_ws();
            if p.peek() == ',' {
                p.bump();
                p.skip_ws();
                continue;
            }
            break;
        }
    }
    p.expect_char(')')?;
    p.skip_ws();
    p.expect_char(':')?;
    // Skip the return type until the `is` keyword.
    loop {
        p.skip_ws();
        if p.at_end() {
            return Err("expected `is` after the return type".to_string());
        }
        if p.starts_with("is") {
            break;
        }
        p.bump();
    }
    p.bump_word("is");
    let body = parse_austral_statements(&p.rest_owned())?;
    let mut term = body;
    for param in params.into_iter().rev() {
        term = CoreIR::Lam(param, Box::new(term));
    }
    Ok((name, term))
}

/// Parse an australVM module body — the real Austral software shape:
///
/// ```austral
/// module body NAME is
///     function f(x: Int64): Int64 is
///         return (x + 1);
///     end;
/// end module body.
/// ```
///
/// Returns the module name and the parsed functions as `(name, curried
/// lambda)` pairs. Function bodies in this subset are `let`/`return`
/// statement lists (no recursion, no nested definitions), so each function's
/// terminating `end;` is unambiguous.
pub fn parse_austral_module(src: &str) -> Result<(String, Vec<(String, CoreIR)>), String> {
    let mut p = Parser::new(src);
    p.skip_ws();
    if !p.starts_with("module body") {
        return Err(format!(
            "expected `module body`, got `{}`",
            p.rest_preview()
        ));
    }
    p.bump_word("module body");
    p.skip_ws();
    let name = p.parse_ident()?;
    loop {
        p.skip_ws();
        if p.at_end() {
            return Err("expected `is` after the module name".to_string());
        }
        if p.starts_with("is") {
            p.bump_word("is");
            break;
        }
        p.bump();
    }
    let mut funcs: Vec<(String, CoreIR)> = Vec::new();
    loop {
        p.skip_ws();
        if p.at_end() {
            return Err("unterminated module (missing `end module body.`)".to_string());
        }
        if p.starts_with("end module body") {
            break;
        }
        if p.starts_with("function") {
            let start = p.pos;
            let rel = p.src[start..]
                .find("end;")
                .ok_or_else(|| "function declaration missing its terminating `end;`".to_string())?;
            let end = start + rel + 4;
            let (fname, term) = parse_austral_function(&p.src[start..end])?;
            funcs.push((fname, term));
            p.pos = end;
            continue;
        }
        return Err(format!(
            "unexpected token at byte {}: `{}`",
            p.pos,
            p.rest_preview()
        ));
    }
    Ok((name, funcs))
}

// ─────────────────────────────────────────────────────────────────────
// Recursive-descent expression parser (byte-based; the Austral subset is
// ASCII, so byte offsets == char offsets).
// ─────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> char {
        self.src[self.pos..].chars().next().unwrap_or('\0')
    }

    fn bump(&mut self) -> char {
        let c = self.peek();
        if !self.at_end() {
            self.pos += c.len_utf8();
        }
        c
    }

    fn skip_ws(&mut self) {
        while !self.at_end() && self.peek().is_whitespace() {
            self.bump();
        }
    }

    fn starts_with(&self, word: &str) -> bool {
        self.src[self.pos..].starts_with(word)
    }

    fn bump_word(&mut self, word: &str) {
        for _ in word.chars() {
            self.bump();
        }
    }

    fn expect_char(&mut self, c: char) -> Result<(), String> {
        self.skip_ws();
        if self.peek() == c {
            self.bump();
            Ok(())
        } else {
            Err(format!("expected `{c}` at byte {}", self.pos))
        }
    }

    fn rest_preview(&self) -> String {
        self.src[self.pos..].chars().take(24).collect()
    }

    fn rest_owned(&self) -> String {
        self.src[self.pos..].to_string()
    }

    fn parse_ident(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.bump();
        }
        let name = &self.src[start..self.pos];
        if name.is_empty() || name.chars().next().unwrap().is_ascii_digit() {
            return Err(format!("expected an identifier at byte {}", self.pos));
        }
        Ok(name.to_string())
    }

    /// Parse an expression: an operand followed by any number of binary
    /// operators. The emitter always parenthesizes `(a OP b)`, but un- or
    /// partially-parenthesized Austral arithmetic (`(2 + 3) * 4`) is also
    /// accepted; the recursion makes chains right-associative.
    fn parse_expr(&mut self) -> Result<CoreIR, String> {
        self.skip_ws();
        let mut left = self.parse_operand()?;
        loop {
            self.skip_ws();
            match self.try_binop() {
                Some(tok) => {
                    let right = self.parse_operand()?;
                    let op = resolve_op(tok, &left, &right);
                    left = CoreIR::Prim(op, vec![left, right]);
                }
                None => break,
            }
        }
        Ok(left)
    }

    fn parse_operand(&mut self) -> Result<CoreIR, String> {
        self.skip_ws();
        if self.peek() == '(' {
            // Parenthesized group: `(a)`, `(a OP b)`, or application `(f x)`.
            self.bump();
            let inner = self.parse_expr()?;
            self.skip_ws();
            if self.peek() != ')' {
                // `(f x)` — the inner parse stopped before an argument.
                let arg = self.parse_expr()?;
                self.skip_ws();
                self.expect_char(')')?;
                return Ok(CoreIR::App(Box::new(inner), Box::new(arg)));
            }
            self.bump();
            return Ok(inner);
        }
        if self.starts_with("not") {
            self.bump_word("not");
            let arg = self.parse_expr()?;
            return Ok(CoreIR::Prim(
                PrimOp::Not,
                vec![arg, CoreIR::Lit(Literal::Bool(false))],
            ));
        }
        if let Some(lit) = self.try_literal() {
            return Ok(CoreIR::Lit(lit));
        }
        if self.starts_with("Tag") {
            self.bump_word("Tag");
            let start = self.pos;
            while !self.at_end() && self.peek().is_ascii_digit() {
                self.bump();
            }
            let tag_str = &self.src[start..self.pos];
            if tag_str.is_empty() {
                return Err(format!("expected tag number after `Tag` at byte {}", self.pos));
            }
            let tag_id: u32 = tag_str.parse().map_err(|_| "tag number overflow".to_string())?;
            self.skip_ws();
            if self.peek() == '(' {
                self.bump();
                let mut args = Vec::new();
                self.skip_ws();
                if self.peek() != ')' {
                    loop {
                        args.push(self.parse_expr()?);
                        self.skip_ws();
                        if self.peek() == ',' {
                            self.bump();
                            self.skip_ws();
                            continue;
                        }
                        break;
                    }
                }
                self.expect_char(')')?;
                return Ok(CoreIR::Con(tag_id, args));
            }
            return Ok(CoreIR::Con(tag_id, vec![]));
        }
        // Identifier: variable or function call `f(x)`.
        let id = self.parse_ident()?;
        self.skip_ws();
        if self.peek() == '(' {
            self.bump();
            let mut args = Vec::new();
            self.skip_ws();
            if self.peek() != ')' {
                loop {
                    args.push(self.parse_expr()?);
                    self.skip_ws();
                    if self.peek() == ',' {
                        self.bump();
                        self.skip_ws();
                        continue;
                    }
                    break;
                }
            }
            self.expect_char(')')?;
            let mut term = CoreIR::Var(id);
            for a in args {
                term = CoreIR::App(Box::new(term), Box::new(a));
            }
            return Ok(term);
        }
        Ok(CoreIR::Var(id))
    }

    fn try_literal(&mut self) -> Option<Literal> {
        self.skip_ws();
        if self.starts_with("true") {
            self.bump_word("true");
            return Some(Literal::Bool(true));
        }
        if self.starts_with("false") {
            self.bump_word("false");
            return Some(Literal::Bool(false));
        }
        let start = self.pos;
        let mut neg = false;
        if self.peek() == '-' {
            neg = true;
            self.bump();
        }
        let mut int_part = String::new();
        while !self.at_end() && self.peek().is_ascii_digit() {
            int_part.push(self.bump());
        }
        if int_part.is_empty() {
            self.pos = start;
            return None;
        }
        let mut float = false;
        let mut frac = String::new();
        if self.peek() == '.' {
            float = true;
            self.bump();
            while !self.at_end() && self.peek().is_ascii_digit() {
                frac.push(self.bump());
            }
        }
        if !float && (self.peek() == 'e' || self.peek() == 'E') {
            let save = self.pos;
            self.bump();
            if self.peek() == '-' || self.peek() == '+' {
                self.bump();
            }
            let mut exp = String::new();
            while !self.at_end() && self.peek().is_ascii_digit() {
                exp.push(self.bump());
            }
            if exp.is_empty() {
                self.pos = save;
            } else {
                float = true;
                frac = format!("e{exp}");
            }
        }
        let body = if float {
            let s = format!("{}{}.{}", if neg { "-" } else { "" }, int_part, frac);
            return Some(Literal::F64(s.parse().ok()?));
        } else {
            format!("{}{}", if neg { "-" } else { "" }, int_part)
        };
        Some(Literal::Int64(body.parse().ok()?))
    }

    fn try_binop(&mut self) -> Option<BinOpTok> {
        self.skip_ws();
        // Two-char operators first.
        if self.pos + 2 <= self.src.len() {
            match &self.src[self.pos..self.pos + 2] {
                "==" => {
                    self.bump();
                    self.bump();
                    return Some(BinOpTok::Eq);
                }
                "<=" => {
                    self.bump();
                    self.bump();
                    return Some(BinOpTok::Lt);
                }
                ">=" => {
                    self.bump();
                    self.bump();
                    return Some(BinOpTok::Gt);
                }
                _ => {}
            }
        }
        // Word operators `and` / `or` (word-boundary aware: `and` must not
        // match a longer identifier like `andrew`).
        for (word, tok) in [("and", BinOpTok::And), ("or", BinOpTok::Or)] {
            if self.starts_with(word) {
                let after = self.pos + word.len();
                let boundary = after >= self.src.len()
                    || !self.src[after..].chars().next().unwrap().is_alphanumeric();
                if boundary {
                    self.bump_word(word);
                    return Some(tok);
                }
            }
        }
        let op = match self.peek() {
            '+' => Some(BinOpTok::Add),
            '-' => Some(BinOpTok::Sub),
            '*' => Some(BinOpTok::Mul),
            '/' => Some(BinOpTok::Div),
            '=' => Some(BinOpTok::Eq),
            '>' => Some(BinOpTok::Gt),
            '<' => Some(BinOpTok::Lt),
            _ => None,
        };
        if op.is_some() {
            self.bump();
        }
        op
    }
}

/// Operator token as lexed (before operand types are known).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOpTok {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Gt,
    Lt,
    And,
    Or,
}

/// Resolve a lexed operator to a `PrimOp`, typing it by the operands: if
/// either operand is a float literal the F64 family is used (the emitter
/// prints `+` for both `Add64` and `AddF64`), otherwise the Int64 family.
/// Unknowns default to the Int64 family (the emitter's default typing).
fn resolve_op(tok: BinOpTok, a: &CoreIR, b: &CoreIR) -> PrimOp {
    let float = matches!(a, CoreIR::Lit(Literal::F64(_)))
        || matches!(b, CoreIR::Lit(Literal::F64(_)));
    match (tok, float) {
        (BinOpTok::Add, false) => PrimOp::Add64,
        (BinOpTok::Add, true) => PrimOp::AddF64,
        (BinOpTok::Sub, false) => PrimOp::Sub64,
        (BinOpTok::Sub, true) => PrimOp::SubF64,
        (BinOpTok::Mul, false) => PrimOp::Mul64,
        (BinOpTok::Mul, true) => PrimOp::MulF64,
        // There is no integer division in PrimOp: `/` is always F64.
        (BinOpTok::Div, _) => PrimOp::DivF64,
        (BinOpTok::Eq, false) => PrimOp::Eq64,
        (BinOpTok::Eq, true) => PrimOp::EqF64,
        (BinOpTok::Gt, false) => PrimOp::Gt64,
        (BinOpTok::Gt, true) => PrimOp::GtF64,
        (BinOpTok::Lt, false) => PrimOp::Lt64,
        (BinOpTok::Lt, true) => PrimOp::LtF64,
        (BinOpTok::And, _) => PrimOp::And,
        (BinOpTok::Or, _) => PrimOp::Or,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_literal_and_arithmetic() {
        let e = parse_austral_expr("(2 + 3) * 4").unwrap();
        assert_eq!(
            e,
            CoreIR::Prim(
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
            )
        );
    }

    #[test]
    fn parse_float_and_bool() {
        assert_eq!(
            parse_austral_expr("1.5").unwrap(),
            CoreIR::Lit(Literal::F64(1.5))
        );
        assert_eq!(
            parse_austral_expr("true").unwrap(),
            CoreIR::Lit(Literal::Bool(true))
        );
        assert_eq!(
            parse_austral_expr("(true and false)").unwrap(),
            CoreIR::Prim(
                PrimOp::And,
                vec![
                    CoreIR::Lit(Literal::Bool(true)),
                    CoreIR::Lit(Literal::Bool(false)),
                ],
            )
        );
    }

    #[test]
    fn parse_unknown_identifier_is_var() {
        assert_eq!(parse_austral_expr("x").unwrap(), CoreIR::Var("x".to_string()));
        assert_eq!(
            parse_austral_expr("(x + 3)").unwrap(),
            CoreIR::Prim(
                PrimOp::Add64,
                vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(3))],
            )
        );
    }

    #[test]
    fn parse_call_and_tag() {
        assert_eq!(
            parse_austral_expr("f(42)").unwrap(),
            CoreIR::App(
                Box::new(CoreIR::Var("f".to_string())),
                Box::new(CoreIR::Lit(Literal::Int64(42))),
            )
        );
        assert_eq!(
            parse_austral_expr("Tag1(2, 3)").unwrap(),
            CoreIR::Con(
                1,
                vec![
                    CoreIR::Lit(Literal::Int64(2)),
                    CoreIR::Lit(Literal::Int64(3))
                ]
            )
        );
        assert_eq!(parse_austral_expr("Tag5").unwrap(), CoreIR::Con(5, vec![]));
    }

    #[test]
    fn parse_statements_let_return() {
        let body = parse_austral_statements("let x: Int64 = 5; return (x + 3);").unwrap();
        assert_eq!(
            body,
            CoreIR::Let(
                "x".to_string(),
                Box::new(CoreIR::Lit(Literal::Int64(5))),
                Box::new(CoreIR::Prim(
                    PrimOp::Add64,
                    vec![CoreIR::Var("x".to_string()), CoreIR::Lit(Literal::Int64(3))],
                )),
            )
        );
    }

    #[test]
    fn parse_two_lets_ordered() {
        let body =
            parse_austral_statements("let x: Int64 = 1; let y: Int64 = 2; return (x + y);").unwrap();
        match body {
            CoreIR::Let(x, vx, inner) => {
                assert_eq!(x, "x");
                assert_eq!(*vx, CoreIR::Lit(Literal::Int64(1)));
                match *inner {
                    CoreIR::Let(y, vy, ret) => {
                        assert_eq!(y, "y");
                        assert_eq!(*vy, CoreIR::Lit(Literal::Int64(2)));
                        assert!(matches!(*ret, CoreIR::Prim(PrimOp::Add64, ..)));
                    }
                    other => panic!("inner must be the second let, got {other}"),
                }
            }
            other => panic!("outermost must be the first let, got {other}"),
        }
    }

    #[test]
    fn parse_clone_statement() {
        // clone after a let: let x = 1; clone x -> y, z; return (y + z);
        let body = parse_austral_statements(
            "let x: Int64 = 1; clone x -> y, z; return (y + z);",
        )
        .unwrap();
        match body {
            CoreIR::Let(x, vx, inner) => {
                assert_eq!(x, "x");
                assert_eq!(*vx, CoreIR::Lit(Literal::Int64(1)));
                match *inner {
                    CoreIR::Clone(c, y, z, ret) => {
                        assert_eq!(c, "x");
                        assert_eq!(y, "y");
                        assert_eq!(z, "z");
                        assert!(matches!(*ret, CoreIR::Prim(PrimOp::Add64, ..)));
                    }
                    other => panic!("inner must be the clone, got {other}"),
                }
            }
            other => panic!("outermost must be the let, got {other}"),
        }
    }

    #[test]
    fn parse_destroy_statement() {
        // destroy after a let: the variable is consumed without a use.
        let body = parse_austral_statements("let x: Int64 = 1; destroy x; return 2;").unwrap();
        match body {
            CoreIR::Let(x, vx, inner) => {
                assert_eq!(x, "x");
                assert_eq!(*vx, CoreIR::Lit(Literal::Int64(1)));
                match *inner {
                    CoreIR::Drop(d, ret) => {
                        assert_eq!(d, "x");
                        assert_eq!(*ret, CoreIR::Lit(Literal::Int64(2)));
                    }
                    other => panic!("inner must be the destroy, got {other}"),
                }
            }
            other => panic!("outermost must be the let, got {other}"),
        }
    }

    #[test]
    fn parse_function_shape() {
        let (name, term) =
            parse_austral_function("function f(x: Int64): Int64 is return (x + 1); end;").unwrap();
        assert_eq!(name, "f");
        assert!(matches!(term, CoreIR::Lam(..)));
    }

    #[test]
    fn parse_module_body() {
        let (name, funcs) = parse_austral_module(
            "module body Probe is\n\
                 function add(x: Int64): Int64 is\n\
                     return (x + 1);\n\
                 end;\n\
                 function main(): Int64 is\n\
                     return 42;\n\
                 end;\n\
             end module body.\n",
        )
        .unwrap();
        assert_eq!(name, "Probe");
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].0, "add");
        assert!(matches!(funcs[0].1, CoreIR::Lam(..)));
        assert_eq!(funcs[1].0, "main");
    }
}
