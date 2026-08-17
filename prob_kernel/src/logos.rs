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

use unfer_protocol::LogosReport;

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
}
