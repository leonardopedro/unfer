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

use unfer_protocol::{AustralDefeqReport, AustralReport, LogosReport};

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
        sym_expr: first.sym_expr.to_prefix_string(),
        infix: first.infix.clone(),
        value: first.value.clone().map(|l| l.to_string()),
        unf_hash: first.unf_hash.clone(),
        ted: first.ted_string.clone(),
        ted_hash: first.ted_hash.clone(),
        verified,
        source: source.to_string(),
    })
}

// ── Conversion check (definitional equality via the UNF) ──────────────────

/// The result of a conversion check between two AustralVM source fragments —
/// [`unfer_protocol::AustralDefeqReport`], the protocol type crossing the
/// FFI boundary as JSON.
///
/// This is the kernel-level form of the lean4mm0 plan's conversion step: two
/// terms are definitionally equal when their DeltaNet normal forms are
/// isomorphic, and the *certificate* of that equality is the node
/// permutation `witness` — not just the matching digests. When the net-level
/// digests differ but both terms lie in the Int64-arithmetic fragment, the
/// TED hashes still tell the *algebraic* story (`x + x` vs `2 * x`); that
/// distinction is reported rather than hidden, so a caller can see which
/// notion of equality held.
///
/// Definitional-equality check: reduce both sources to their unique normal
/// forms and certify the comparison with an isomorphism witness.
///
/// Both the ingestion of either side and the witness computation map to
/// [`KernelError::AustralDefeqFailed`] (UK-4805): a caller asking the
/// *conversion* question gets the conversion code, never the single-source
/// translation code (UK-4804).
pub fn austral_defeq(a: &str, b: &str) -> Result<AustralDefeqReport, KernelError> {
    use KernelError::AustralDefeqFailed;

    let as_defeq = |e: KernelError| match e {
        KernelError::AustralUnfFailed { reason } => AustralDefeqFailed { reason },
        other => other,
    };
    let ta = translate_with(a).map_err(as_defeq)?;
    let tb = translate_with(b).map_err(as_defeq)?;
    let witness =
        logos::translate::check_equivalence(&ta, &tb).map_err(|e| AustralDefeqFailed {
            reason: format!("austral->deltanet conversion check failed: {e}"),
        })?;
    let equivalent = witness.is_some();
    let witness = witness.map(|w| w.pairs.into_iter().collect::<Vec<(u32, u32)>>());

    Ok(AustralDefeqReport {
        equivalent,
        sym_expr_a: ta.sym_expr.to_prefix_string(),
        sym_expr_b: tb.sym_expr.to_prefix_string(),
        unf_hash_a: ta.unf_hash,
        unf_hash_b: tb.unf_hash,
        ted_hash_a: ta.ted_hash,
        ted_hash_b: tb.ted_hash,
        witness,
    })
}

/// One translation pass; the report's `verified` field is populated by
/// [`austral_unf`]'s confluence check. Accepts, in order: a statement list
/// (`let … return e;`), a single expression (`(2 + 3)`), a function
/// declaration (`function f(x: Int64): Int64 is … end;`), or a whole module
/// (`module body … end module body.`) — the dispatch itself lives in
/// `logos::translate::translate_austral_any` (the CLI's `logos iso` uses the
/// same seam, so the compiler plugin, the kernel op and the CLI cannot
/// disagree on what a source *is*).
fn translate_with(source: &str) -> Result<logos::translate::UnfTranslation, KernelError> {
    logos::translate::translate_austral_any(source).map_err(|e| KernelError::AustralUnfFailed {
        reason: format!("austral->deltanet translation failed: {e}"),
    })
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

    // ── conversion check (austral_defeq) ────────────────────────────

    #[test]
    fn defeq_equal_normal_forms_certified_with_witness() {
        // (x + (2 + 3)) reduces to the same net as (x + 5): equal digests
        // and a non-empty isomorphism witness.
        let r = austral_defeq("(x + (2 + 3))", "(x + 5)").unwrap();
        assert!(r.equivalent);
        assert_eq!(r.unf_hash_a, r.unf_hash_b);
        let w = r.witness.expect("equivalence carries its certificate");
        assert!(!w.is_empty());
        // The permutation is injective on each side.
        let from: std::collections::HashSet<u32> = w.iter().map(|(a, _)| *a).collect();
        let to: std::collections::HashSet<u32> = w.iter().map(|(_, b)| *b).collect();
        assert_eq!(from.len(), w.len(), "g1 nodes repeat");
        assert_eq!(to.len(), w.len(), "g2 nodes repeat");
    }

    #[test]
    fn defeq_closed_forms_agree() {
        let r = austral_defeq("let x: Int64 = 2; return (x + 3);", "5").unwrap();
        assert!(r.equivalent);
        assert_eq!(r.sym_expr_a, "5");
        assert_eq!(r.sym_expr_b, "5");
    }

    #[test]
    fn defeq_distinct_normal_forms_not_equivalent() {
        // Addition is not commutative *as a net*: Add64(x, 1) ≠ Add64(1, x).
        let r = austral_defeq("(x + 1)", "(1 + x)").unwrap();
        assert!(!r.equivalent);
        assert!(r.witness.is_none());
        assert_ne!(r.unf_hash_a, r.unf_hash_b);
    }

    #[test]
    fn defeq_reports_the_algebraic_fallback_when_nets_differ() {
        // x + x and 2 * x are different nets but the same TED polynomial:
        // the report exposes both digests so the caller sees which equality
        // (net-level vs algebraic) holds.
        let r = austral_defeq("(x + x)", "(2 * x)").unwrap();
        assert!(!r.equivalent, "net-level defeq is shape-sensitive");
        assert_ne!(r.unf_hash_a, r.unf_hash_b);
        assert_eq!(r.ted_hash_a, r.ted_hash_b);
        assert!(r.ted_hash_a.is_some());
    }

    #[test]
    fn defeq_unparseable_side_is_error() {
        assert!(austral_defeq("(x + 1)", "@@@ nope @@@").is_err());
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
