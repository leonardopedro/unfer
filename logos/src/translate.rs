//! The austral ↔ unique-normal-form translation orchestrator.
//!
//! This is the finished form of the translation started across
//! `austral_codegen` (CoreIR → Austral) and `deltanet` (CoreIR → interaction
//! net → UNF): a single entry point that takes **Austral source** (or a
//! CoreIR term), lowers it to an interaction net, reduces it to its unique
//! normal form, and — whenever the term has **no unknown variables** —
//! replaces the normal form with the numerical result of its calculation
//! (e.g. `ADD` of two 64-bit integers → `5`). When unknowns remain, the
//! translation yields the *symbolic expression* (`Add64(x, 3)`), which
//! [`SymExpr::eval`] refuses until the unknown is resolved.
//!
//! Every report carries:
//!   • `sym_expr` — the structured symbolic readback (prefix, canonical);
//!   • `infix`    — the arithmetic fragment in infix form (`(2 + 3) * 4`);
//!   • `value`    — `Some(literal)` iff the term is closed and numeric;
//!   • `unf_hash` — the content-addressable digest of the reduced net;
//!   • `verified` — the confluence self-check (a second independent
//!     reduction reproduces the identical UNF).

use std::collections::HashMap;

use crate::austral_codegen::{
    parse_austral_function, parse_austral_module, parse_austral_statements, validate_body,
    validate_module,
};
use crate::core_ir::{CoreIR, Literal};
use crate::deltanet::{self, SymExpr, Ted};

/// The result of translating a term to its unique normal form.
#[derive(Debug, Clone)]
pub struct UnfTranslation {
    /// The lowered CoreIR term (its `Display` is the S-expression form).
    pub ir: CoreIR,
    /// The structured symbolic readback of the reduced net.
    pub sym_expr: SymExpr,
    /// The infix rendering of the arithmetic fragment.
    pub infix: String,
    /// `Some(literal)` iff the term is closed (no unknowns) — the numerical
    /// result of the calculation. `None` for open (symbolic) terms.
    pub value: Option<Literal>,
    /// Content-addressable UNF digest (SHA-256 of the canonical net
    /// serialization).
    pub unf_hash: String,
    /// The word-level TED (Phase 3): the canonical polynomial normal form
    /// of the Int64-arithmetic fragment over ℤ/2⁶⁴. `Some(poly)` when the
    /// term lies in that fragment, `None` otherwise (F64/bool/application
    /// … — fall back to `unf_hash`).
    pub ted: Option<Ted>,
    /// The canonical TED string (e.g. `"3*x + 6"`, `"20"`), when in the
    /// arithmetic fragment.
    pub ted_string: Option<String>,
    /// SHA-256 of the canonical TED serialization — the content-addressable
    /// *algebraic* UNF, independent of the net encoding.
    pub ted_hash: Option<String>,
    /// Confluence self-check: a second reduction of the same term reproduces
    /// the identical UNF.
    pub verified: bool,
}

impl UnfTranslation {
    /// True iff the translation collapsed to a concrete value (no unknowns).
    pub fn is_closed(&self) -> bool {
        self.sym_expr.is_closed()
    }
}

/// Translate a CoreIR term: compile → reduce → symbolic readback → value →
/// UNF hash → TED, with the confluence self-check.
pub fn translate_coreir(ir: &CoreIR) -> Result<UnfTranslation, String> {
    let (sym_expr, unf_hash) = reduce_once(ir)?;
    let second = reduce_once(ir)?;
    let verified = second.0 == sym_expr && second.1 == unf_hash;

    let value = if sym_expr.is_closed() { sym_expr.eval() } else { None };

    let ted = deltanet::ted::from_sym_expr(&sym_expr);
    let (ted_string, ted_hash) = match &ted {
        Some(t) => (Some(t.to_canonical_string()), Some(t.ted_hash())),
        None => (None, None),
    };

    Ok(UnfTranslation {
        ir: ir.clone(),
        infix: sym_expr.to_infix_string(),
        sym_expr,
        value,
        unf_hash,
        ted,
        ted_string,
        ted_hash,
        verified,
    })
}

/// Translate an Austral source fragment (statement list: `let`s then
/// `return <expr>;` — the emitter dialect) to its unique normal form.
/// The statement list is validated for totality & linearity first (Phase 1.3)
/// — there are no defined functions in scope, so any call is an unknown.
pub fn translate_austral(src: &str) -> Result<UnfTranslation, String> {
    let ir = parse_austral_statements(src)?;
    let names = std::collections::HashSet::new();
    validate_body(&names, "<top-level>", &ir)?;
    translate_coreir(&ir)
}

/// Translate a single Austral expression (e.g. `"(2 + 3) * 4"` or
/// `"(x + 3)"`) to its unique normal form.
pub fn translate_austral_expr(src: &str) -> Result<UnfTranslation, String> {
    let ir = crate::austral_codegen::parse_austral_expr(src)?;
    translate_coreir(&ir)
}

/// Compile `ir` to a net, reduce it, and return the symbolic readback +
/// UNF hash.
fn reduce_once(ir: &CoreIR) -> Result<(SymExpr, String), String> {
    let mut net = deltanet::compile_to_net(ir)?;
    deltanet::reduce(&mut net)?;
    let sym = deltanet::sym_readback(&net)?;
    let hash = deltanet::unf_hash_string(&net)?;
    Ok((sym, hash))
}

/// Peel the outer lambda(s) off a function term, leaving the body with the
/// parameters as **free** unknowns (`CoreIR::Var` → deltanet `Entity`). This
/// is how a function definition is translated: its computation stays
/// symbolic (`Add64(x, 1)`) until the parameters are known.
pub fn strip_lams(term: &CoreIR) -> CoreIR {
    match term {
        CoreIR::Lam(_, body) => strip_lams(body),
        other => other.clone(),
    }
}

/// Link a module's function definitions into a single closed term: every
/// call to a defined function (`CoreIR::Var(f)`) is replaced by the function's
/// lambda, so the deltanet reducer beta-reduces the calls. The subset is
/// non-recursive, so substitution terminates. Returns the linked `main`
/// (or the first function when no `main` is defined).
pub fn link_functions(funcs: &[(String, CoreIR)]) -> CoreIR {
    let env: HashMap<String, CoreIR> = funcs.iter().cloned().collect();
    let main = funcs
        .iter()
        .find(|(n, _)| n == "main")
        .or_else(|| funcs.first())
        .map(|(_, t)| t.clone())
        .unwrap_or_else(|| CoreIR::Lit(Literal::Int64(0)));
    rewrite_calls(&main, &env)
}

fn rewrite_calls(term: &CoreIR, env: &HashMap<String, CoreIR>) -> CoreIR {
    match term {
        CoreIR::Var(name) => env
            .get(name)
            .cloned()
            .unwrap_or_else(|| CoreIR::Var(name.clone())),
        CoreIR::Lit(_) => term.clone(),
        CoreIR::Con(tag, args) => CoreIR::Con(*tag, args.iter().map(|a| rewrite_calls(a, env)).collect()),
        CoreIR::Lam(id, body) => CoreIR::Lam(id.clone(), Box::new(rewrite_calls(body, env))),
        CoreIR::App(f, a) => CoreIR::App(
            Box::new(rewrite_calls(f, env)),
            Box::new(rewrite_calls(a, env)),
        ),
        CoreIR::Let(id, value, body) => CoreIR::Let(
            id.clone(),
            Box::new(rewrite_calls(value, env)),
            Box::new(rewrite_calls(body, env)),
        ),
        CoreIR::Match(s, arms) => CoreIR::Match(
            Box::new(rewrite_calls(s, env)),
            arms.iter()
                .map(|(p, b)| (p.clone(), rewrite_calls(b, env)))
                .collect(),
        ),
        CoreIR::Fold(f, i, l) => CoreIR::Fold(
            Box::new(rewrite_calls(f, env)),
            Box::new(rewrite_calls(i, env)),
            Box::new(rewrite_calls(l, env)),
        ),
        CoreIR::Clone(id, id1, id2, body) => CoreIR::Clone(
            id.clone(),
            id1.clone(),
            id2.clone(),
            Box::new(rewrite_calls(body, env)),
        ),
        CoreIR::Drop(id, body) => {
            CoreIR::Drop(id.clone(), Box::new(rewrite_calls(body, env)))
        }
        CoreIR::Prim(op, args) => CoreIR::Prim(*op, args.iter().map(|a| rewrite_calls(a, env)).collect()),
    }
}

/// The translation of a whole AustralVM module: per-function symbolic
/// forms plus the linked, reduced `main`.
#[derive(Debug, Clone)]
pub struct ModuleTranslation {
    pub module_name: String,
    /// Each function, translated with its parameters as unknowns (the
    /// body-with-free-variables symbolic form).
    pub functions: Vec<(String, UnfTranslation)>,
    /// The linked `main` (calls inlined), reduced to its unique normal form:
    /// a closed module yields the numerical result.
    pub main: UnfTranslation,
}

/// Translate a single Austral function declaration: `function f(x: T): T is
/// <stmts> end;` → its body-with-unknowns symbolic form (e.g. `(Add64 x 1)`).
/// The body is validated for totality & linearity first (Phase 1.3).
pub fn translate_austral_function(src: &str) -> Result<(String, UnfTranslation), String> {
    let (name, term) = parse_austral_function(src)?;
    // The function body is the lambda chain peeled off; validate the whole
    // term with the single function name in scope (self-calls rejected).
    let mut names = std::collections::HashSet::new();
    names.insert(name.clone());
    validate_body(&names, &name, &term)?;
    let t = translate_coreir(&strip_lams(&term))?;
    Ok((name, t))
}

/// Translate a whole AustralVM module body to DeltaNets: every function is
/// validated (totality & linearity) then reduced to its symbolic unique
/// normal form, and the linked `main` is reduced to a value when the module
/// is closed.
pub fn translate_austral_module(src: &str) -> Result<ModuleTranslation, String> {
    let (module_name, funcs) = parse_austral_module(src)?;
    validate_module(&funcs)?;
    let mut functions = Vec::with_capacity(funcs.len());
    for (fname, term) in &funcs {
        let t = translate_coreir(&strip_lams(term))?;
        functions.push((fname.clone(), t));
    }
    let main_ir = link_functions(&funcs);
    let main = translate_coreir(&main_ir)?;
    Ok(ModuleTranslation {
        module_name,
        functions,
        main,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::PrimOp;

    #[test]
    fn closed_austral_arithmetic_yields_value() {
        // `let x: Int64 = 2; return ((x + 3) * 4);` — closed → 20.
        let t = translate_austral("let x: Int64 = 2; return ((x + 3) * 4);").unwrap();
        assert!(t.is_closed());
        assert_eq!(t.value, Some(Literal::Int64(20)));
        assert!(t.verified);
        assert!(!t.unf_hash.is_empty());
        assert_eq!(t.infix, "20");
    }

    #[test]
    fn add_two_64bit_integers() {
        // The user-level example: ADD of two 64-bit integers → 5.
        let t = translate_austral_expr("(2 + 3)").unwrap();
        assert_eq!(t.value, Some(Literal::Int64(5)));
        assert_eq!(t.infix, "5");
        assert!(t.verified);
    }

    #[test]
    fn open_term_stays_symbolic() {
        // x + 3 with unknown x: the ADD remains a visible symbolic
        // operation; no value.
        let t = translate_austral_expr("(x + 3)").unwrap();
        assert!(!t.is_closed());
        assert_eq!(t.value, None);
        assert_eq!(t.infix, "(x + 3)");
        assert!(t.verified);
        // And a closed subterm inside an open term is still evaluated: the
        // reducer folds `2 * 3` inside `x + (2 * 3)` → `(x + 6)`.
        let t2 = translate_austral_expr("(x + (2 * 3))").unwrap();
        assert_eq!(t2.infix, "(x + 6)");
    }

    #[test]
    fn f64_and_bool_calculation() {
        let t = translate_austral_expr("(1.5 + 2.25)").unwrap();
        assert_eq!(t.value, Some(Literal::F64(3.75)));
        let t2 = translate_austral_expr("(5 > 2)").unwrap();
        assert_eq!(t2.value, Some(Literal::Bool(true)));
    }

    #[test]
    fn confluence_repeat_translation() {
        let a = translate_austral_expr("((2 + 3) * 4)").unwrap();
        let b = translate_austral_expr("((2 + 3) * 4)").unwrap();
        assert_eq!(a.unf_hash, b.unf_hash);
        assert_eq!(a.sym_expr, b.sym_expr);
    }

    #[test]
    fn round_trip_coreir_emit_parse_unf() {
        // The translation is round-trip stable: CoreIR → Austral (reduced) →
        // parse → UNF is identical to the direct CoreIR → UNF.
        use crate::austral_codegen::emit_austral_reduced;
        let ir = CoreIR::Prim(
            PrimOp::Add64,
            vec![
                CoreIR::Lit(Literal::Int64(2)),
                CoreIR::Lit(Literal::Int64(3)),
            ],
        );
        let direct = translate_coreir(&ir).unwrap();
        let emitted = emit_austral_reduced(&ir);
        // The reduced emission contains the folded literal.
        assert!(emitted.contains("return 5;"), "{emitted}");
        let reparsed = parse_austral_statements(&extract_body(&emitted)).unwrap();
        let round = translate_coreir(&reparsed).unwrap();
        assert_eq!(round.unf_hash, direct.unf_hash);
        assert_eq!(round.value, Some(Literal::Int64(5)));
    }

    fn extract_body(module: &str) -> String {
        // Pull out everything between `is` and `end;` of logos_main — the
        // statement list the parser accepts. The signature is
        // `function logos_main(): Int64 is\n`, so the body begins after
        // the `is\n` marker.
        let start = module.find("function logos_main").unwrap();
        let tail = &module[start..];
        let is_pos = tail.find("is\n").unwrap();
        let end_pos = tail.find("end;").unwrap();
        module[start + is_pos + 3..start + end_pos].to_string()
    }

    #[test]
    fn translate_function_body_stays_symbolic() {
        // f(x) = x + 1 — the parameter is an unknown, so the translation is
        // the symbolic ADD of two 64-bit integers, not a value.
        let (name, t) =
            translate_austral_function("function f(x: Int64): Int64 is return (x + 1); end;")
                .unwrap();
        assert_eq!(name, "f");
        assert!(!t.is_closed());
        assert_eq!(t.value, None);
        assert_eq!(t.infix, "(x + 1)");
        assert!(t.verified);
    }

    #[test]
    fn translate_closed_module_reduces_main_to_value() {
        // The australVM software shape: two functions, main calling add.
        let src = "module body Probe is\n\
                   function add(x: Int64): Int64 is\n\
                       return (x + 1);\n\
                   end;\n\
                   function main(): Int64 is\n\
                       return add(41);\n\
                   end;\n\
               end module body.\n";
        let m = translate_austral_module(src).unwrap();
        assert_eq!(m.module_name, "Probe");
        assert_eq!(m.functions.len(), 2);
        // add stays symbolic (unknown parameter).
        assert_eq!(m.functions[0].1.infix, "(x + 1)");
        // main is linked (add inlined) and closed: add(41) → 42.
        assert!(m.main.is_closed());
        assert_eq!(m.main.value, Some(Literal::Int64(42)));
        assert!(m.main.verified);
    }
}
