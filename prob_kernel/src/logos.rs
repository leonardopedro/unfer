//! Logos CNL-to-UNF coupling.
//!
//! Compiles a controlled-natural-language (CNL) sentence through the `logos`
//! pipeline (parse → compile → interaction-net reduce → readback → hash) to a
//! unique normal form (UNF). This is the kernel-side analogue of the S29/S30
//! "reduce to a canonical digest" pattern: `ProofReport.verified` for Lean4
//! proofs, `SymbolicReport` for Cadabra2 expressions, and here `LogosReport`
//! for CNL sentences — all funneling an external-engine result into the model
//! session as a content-addressed normal form.
//!
//! The CNL subset and lexicon are the Logos L0 grammar (`logos/docs/LOGOS.md`).
//! A self-contained default lexicon is embedded here so the kernel symbol does
//! not depend on a working-directory-relative `corpus/lexicon.tsv`.

use unfer_protocol::{AustralReport, LogosReport};

use logos::ccg;
use logos::core_ir;
use logos::deltanet;
use logos::lexicon::Lexicon;

use crate::error::KernelError;

/// A minimal but representative CNL lexicon (L0 subset): named entities,
/// transitive/intransitive/ditransitive verbs, determiners, copulas, numerals,
/// booleans. Mirrors the Logos test lexicon.
pub const DEFAULT_LEXICON_TSV: &str = concat!(
    "John\tNP\tVar(\"john\")\n",
    "Mary\tNP\tVar(\"mary\")\n",
    "Bob\tNP\tVar(\"bob\")\n",
    "Alice\tNP\tVar(\"alice\")\n",
    "loves\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Love\", [Var(\"x\"), Var(\"y\")])))\n",
    "sees\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"See\", [Var(\"x\"), Var(\"y\")])))\n",
    "the\tNP/N\tLam(\"n\", Var(\"n\"))\n",
    "cat\tN\tCon(\"Cat\", [])\n",
    "dog\tN\tCon(\"Dog\", [])\n",
    "sleeps\tS\\NP\tLam(\"x\", Con(\"Sleep\", [Var(\"x\")]))\n",
    "runs\tS\\NP\tLam(\"x\", Con(\"Run\", [Var(\"x\")]))\n",
    "zero\tNP\tLit(Int64(0))\n",
    "one\tNP\tLit(Int64(1))\n",
    "two\tNP\tLit(Int64(2))\n",
    "three\tNP\tLit(Int64(3))\n",
    "adds\t((S\\NP)/NP)/NP\tLam(\"z\", Lam(\"y\", Lam(\"x\", Con(\"Assign\", [Var(\"x\"), Con(\"Add\", [Var(\"y\"), Var(\"z\")])]))))\n",
    "is\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Eq\", [Var(\"x\"), Var(\"y\")])))\n",
    "true\tNP\tLit(Bool(true))\n",
    "false\tNP\tLit(Bool(false))\n",
);

fn default_lexicon() -> Result<Lexicon, KernelError> {
    Lexicon::parse(DEFAULT_LEXICON_TSV).map_err(|e| KernelError::LogosFailed {
        reason: format!("invalid embedded lexicon: {e}"),
    })
}

/// Compile a CNL sentence to a unique normal form.
///
/// Runs the Logos pipeline on the sentence using [`DEFAULT_LEXICON_TSV`]:
///
/// 1. CCG chart-parse the tokens;
/// 2. compile the derivation to a CoreIR term;
/// 3. lower to an interaction net and reduce it;
/// 4. read back the reduced net to a string and hash the canonical
///    serialization.
///
/// `verified` is a confluence self-check: reducing the same sentence a second
/// time must yield the identical UNF. A rejected sentence (no parse) is a
/// [`KernelError::LogosFailed`].
pub fn logos_compile(sentence: &str) -> Result<LogosReport, KernelError> {
    let lexicon = default_lexicon()?;
    let report = compile_with(sentence, &lexicon)?;

    // Confluence self-check: a second, independent reduction of the same
    // sentence must reproduce the identical UNF (unique normal form).
    let second = compile_with(sentence, &lexicon)?;
    let verified = second.unf_hash == report.unf_hash && second.result == report.result;

    Ok(LogosReport {
        result: report.result,
        unf_hash: report.unf_hash,
        verified,
        sentence: sentence.to_string(),
    })
}

/// Translate an AustralVM-language source fragment to a unique normal form
/// through DeltaNets (`logos::translate`).
///
/// The source (a statement list `let … return e;`, a single expression, a
/// function declaration, or a whole `module body … end module body.`) is
/// lowered to CoreIR, compiled to an interaction net, reduced to its unique
/// normal form, and read back as a **symbolic expression**. Whenever the
/// term has no unknown variables the expression collapses to the numerical
/// result of its calculation (`value`, e.g. `ADD(2, 3) = 5`); when unknowns
/// remain (`Add64(x, 3)`) the symbolic expression is the answer. `verified`
/// is the confluence self-check (a second independent reduction reproduces
/// the identical UNF).
pub fn austral_unf(source: &str) -> Result<AustralReport, KernelError> {
    let first = translate_with(source)?;
    let second = translate_with(source)?;
    let verified = second.unf_hash == first.unf_hash && second.sym_expr == first.sym_expr;

    Ok(AustralReport {
        sym_expr: first.sym_expr,
        infix: first.infix,
        value: first.value,
        unf_hash: first.unf_hash,
        ted: first.ted,
        ted_hash: first.ted_hash,
        verified,
        source: source.to_string(),
    })
}

/// The reduced fields of one translation pass (before the confluence check).
struct Reduced {
    sym_expr: String,
    infix: String,
    value: Option<String>,
    unf_hash: String,
    ted: Option<String>,
    ted_hash: Option<String>,
}

/// One translation pass; the report's `verified` field is populated by
/// [`austral_unf`]'s confluence check. Accepts, in order: a statement list
/// (`let … return e;`), a single expression (`(2 + 3)`), a function
/// declaration (`function f(x: Int64): Int64 is … end;`), or a whole module
/// (`module body … end module body.`). The first that parses wins.
fn translate_with(source: &str) -> Result<Reduced, KernelError> {
    use logos::translate::{
        translate_austral, translate_austral_expr, translate_austral_function,
        translate_austral_module,
    };

    let fail = |e: String| KernelError::AustralUnfFailed {
        reason: format!("austral->deltanet translation failed: {e}"),
    };
    let s = source.trim_start();

    // The form is chosen by the source shape, so a *validation* rejection
    // (recursion / linearity) is never swallowed as "wrong form":
    //   • `module body …` → the whole-module translation (validates every
    //     function, links, reduces `main`);
    //   • `function …` → the single-function translation;
    //   • otherwise → statement list first, then a bare expression.
    if s.starts_with("module body") {
        return translate_austral_module(source)
            .map(|m| reduced_of(&m.main))
            .map_err(fail);
    }
    if s.starts_with("function") {
        return translate_austral_function(source)
            .map(|(_name, t)| reduced_of(&t))
            .map_err(fail);
    }
    // Statement list first (its parse covers the `let … return e;` shape);
    // a validation rejection there is a real error, not "wrong form" — so
    // propagate it. A *syntax* failure falls through to the bare-expression
    // form.
    match translate_austral(source) {
        Ok(t) => return Ok(reduced_of(&t)),
        Err(e) => {
            if looks_like_statement_list(source) {
                return Err(fail(e));
            }
        }
    }
    translate_austral_expr(source)
        .map(|t| reduced_of(&t))
        .map_err(fail)
}

/// Heuristic: does the source look like a `let … return …;` statement list
/// (as opposed to a bare expression)? Used so a *validation* failure on a
/// statement list is surfaced as such instead of being retried as an
/// expression.
fn looks_like_statement_list(source: &str) -> bool {
    let s = source.trim_start();
    s.starts_with("let ") || s.starts_with("let\t") || s.contains(" return ")
}

/// Project a `logos::translate::UnfTranslation` onto the reduced report
/// fields (symbolic readback, value, both UNF hashes).
fn reduced_of(t: &logos::translate::UnfTranslation) -> Reduced {
    Reduced {
        sym_expr: t.sym_expr.to_prefix_string(),
        infix: t.infix.clone(),
        value: t.value.clone().map(|l| l.to_string()),
        unf_hash: t.unf_hash.clone(),
        ted: t.ted_string.clone(),
        ted_hash: t.ted_hash.clone(),
    }
}

fn compile_with(sentence: &str, lexicon: &Lexicon) -> Result<LogosReport, KernelError> {
    let tokens: Vec<String> = sentence.split_whitespace().map(String::from).collect();
    let trees = ccg::parse_sentence(&tokens, lexicon);
    let tree = trees.first().ok_or_else(|| KernelError::LogosFailed {
        reason: format!("no parse for CNL sentence: {sentence}"),
    })?;

    let ir = core_ir::compile_to_core_ir(tree, lexicon).map_err(|e| KernelError::LogosFailed {
        reason: format!("core-IR compile failed: {e}"),
    })?;
    let mut net = deltanet::compile_to_net(&ir).map_err(|e| KernelError::LogosFailed {
        reason: format!("net compile failed: {e}"),
    })?;
    deltanet::reduce(&mut net).map_err(|e| KernelError::LogosFailed {
        reason: format!("reduction failed: {e}"),
    })?;

    let result = deltanet::readback(&net).map_err(|e| KernelError::LogosFailed {
        reason: format!("readback failed: {e}"),
    })?;
    let unf_hash = deltanet::unf_hash_string(&net).map_err(|e| KernelError::LogosFailed {
        reason: format!("hash failed: {e}"),
    })?;

    Ok(LogosReport {
        result,
        unf_hash,
        // populated by logos_compile's confluence check
        verified: false,
        sentence: sentence.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_transitive() {
        let r = logos_compile("John loves Mary").unwrap();
        assert_eq!(r.result, "Love(john, mary)");
        assert!(r.verified);
        assert!(!r.unf_hash.is_empty());
    }

    #[test]
    fn compile_arithmetic_is_symbolic() {
        // The ditransitive `adds` compiles to a symbolic Assign/Add constructor
        // term (not a numerical PrimOp fold) — a stable, deterministic UNF.
        let r = logos_compile("John adds two three").unwrap();
        assert_eq!(r.result, "Assign(john, Add(3, 2))");
        assert!(r.verified);
    }

    #[test]
    fn confluence_self_check_holds() {
        for sentence in ["the cat sleeps", "Mary sees John", "one is one"] {
            let r = logos_compile(sentence).unwrap();
            assert!(r.verified, "confluence failed for {sentence}");
        }
    }

    #[test]
    fn rejects_unparseable() {
        // "unknownword" is not in the lexicon → no derivation → compile error.
        assert!(logos_compile("blorpthewogg").is_err());
    }

    #[test]
    fn deterministic_unf() {
        let a = logos_compile("John loves Mary").unwrap();
        let b = logos_compile("John loves Mary").unwrap();
        assert_eq!(a.unf_hash, b.unf_hash);
    }

    // ── Austral → DeltaNets → UNF (the software translation) ─────────────

    #[test]
    fn austral_closed_arithmetic_yields_value() {
        // ADD of two 64-bit integers, closed: the UNF is the number 5.
        let r = austral_unf("let x: Int64 = 2; return (x + 3);").unwrap();
        assert_eq!(r.value.as_deref(), Some("5"));
        assert!(r.verified);
        assert_eq!(r.sym_expr, "5");
        assert!(!r.unf_hash.is_empty());
        assert_eq!(r.source, "let x: Int64 = 2; return (x + 3);");
    }

    #[test]
    fn austral_open_term_stays_symbolic() {
        // An unknown `x`: the ADD remains a visible symbolic operation.
        let r = austral_unf("(x + 3)").unwrap();
        assert_eq!(r.value, None);
        assert_eq!(r.sym_expr, "(Add64 x 3)");
        assert_eq!(r.infix, "(x + 3)");
        assert!(r.verified);
    }

    #[test]
    fn austral_function_is_symbolic() {
        let r = austral_unf("function f(x: Int64): Int64 is return (x + 1); end;").unwrap();
        assert_eq!(r.value, None);
        assert_eq!(r.infix, "(x + 1)");
        assert!(r.verified);
    }

    #[test]
    fn austral_module_main_reduces() {
        let src = "module body Probe is\n\
                   function add(x: Int64): Int64 is return (x + 1); end;\n\
                   function main(): Int64 is return add(41); end;\n\
               end module body.\n";
        let r = austral_unf(src).unwrap();
        assert_eq!(r.value.as_deref(), Some("42"));
        assert!(r.verified);
    }

    #[test]
    fn austral_unparseable_is_error() {
        assert!(austral_unf("this is not austral at all !!!").is_err());
    }

    #[test]
    fn austral_ted_canonicalizes_arithmetic() {
        // x + x → the canonical TED polynomial 2*x, and its algebraic UNF
        // hash is stable (same polynomial written differently → same hash).
        let r = austral_unf("(x + x)").unwrap();
        assert_eq!(r.ted.as_deref(), Some("2*x"));
        let r2 = austral_unf("(2 * x)").unwrap();
        assert_eq!(r2.ted.as_deref(), Some("2*x"));
        assert_eq!(r.ted_hash, r2.ted_hash);
        assert!(r.ted_hash.is_some());
    }

    #[test]
    fn austral_closed_ted_collapses_to_value() {
        // (2 + 3) * 4 → the TED is the single constant 20.
        let r = austral_unf("((2 + 3) * 4)").unwrap();
        assert_eq!(r.value.as_deref(), Some("20"));
        assert_eq!(r.ted.as_deref(), Some("20"));
    }

    #[test]
    fn austral_ted_absent_outside_arithmetic_fragment() {
        // F64 arithmetic is outside the word-level TED.
        let r = austral_unf("(1.5 + 2.25)").unwrap();
        assert_eq!(r.ted, None);
        assert_eq!(r.ted_hash, None);
        // But the net-level UNF still exists.
        assert!(!r.unf_hash.is_empty());
    }

    #[test]
    fn austral_recursion_rejected() {
        // f calling itself is a totality violation — rejected before any
        // net is built (the inliner would loop forever).
        let src = "module body Loop is\n\
                   function f(x: Int64): Int64 is return f(x); end;\n\
               end module body.\n";
        let err = austral_unf(src).unwrap_err();
        assert!(
            err.to_string().contains("recursion"),
            "expected a recursion diagnostic, got: {err}"
        );
    }

    #[test]
    fn austral_double_use_rejected() {
        // x used twice without a clone: a linearity violation.
        let src = "let x: Int64 = 1; return (x + x);";
        let err = austral_unf(src).unwrap_err();
        assert!(
            err.to_string().contains("used 2 times"),
            "expected a linearity diagnostic, got: {err}"
        );
    }
}
