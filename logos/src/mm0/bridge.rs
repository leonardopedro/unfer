//! The DeltaNet half of the pipeline: compile a CNL expression through the
//! *existing* logos lowering (`CoreIR` → net → reducer → readback) and use the
//! result to corroborate what the MM0 certificate claims.
//!
//! Why both machines run: they fail differently. The reducer folds a ground
//! `Prim` natively, in one interaction (`reducer::eval_prim`), so it is the
//! fast and independent computation; MM0 re-derives the same number from Peano
//! recursion with no trusted evaluator at all. If they ever disagree, one of
//! them is broken and the emitter refuses to certify the claim — which is the
//! point of keeping the net path rather than evaluating in Rust and calling the
//! answer "verified".
//!
//! The net path also supplies the *structural* evidence for an equality: two
//! reduced nets that serialize identically are the same net, and
//! [`crate::deltanet::isomorphism_witness`] returns the node permutation π that
//! says so — the same certificate `docs/lean4mm0.md` packages as a conversion
//! step. For a claim `L = R` with equal values, the certificate records π; for
//! a refutation it records the two distinct normal forms instead.

use crate::core_ir::{CoreIR, Literal, PrimOp};
use crate::deltanet::{
    Net, SymExpr, compile_to_net, isomorphism_witness, reduce, sym_readback, unf_hash_string,
};

use super::proof::Expr;

/// A CNL expression lowered to the net it denotes, plus what the reducer made
/// of it.
#[derive(Debug, Clone)]
pub struct Lowered {
    pub net: Net,
    /// The content-addressed identity of the reduced net (`unf_hash_string`) —
    /// the same digest `logos unf` prints.
    pub unf_hash: String,
    /// The reducer's normal form, when it is a literal.
    pub value: Option<i64>,
    /// The normal form in the pipeline's own notation, for diagnostics.
    pub normal_form: String,
}

/// Lower `expr` to `CoreIR`, compile, reduce, and read the normal form back.
///
/// Numerals become `Literal::Int64`, variables become `Var` (unbound, so they
/// lower to `Entity` agents and the net stays open exactly as the CNL term is),
/// and the two operators become the `Add64`/`Mul64` primitives that
/// `unfer/logos/src/austral_codegen` emits for `+` and `*`. A successor is
/// `Add64(e, 1)` — see [`to_core`].
pub fn lower(expr: &Expr) -> Result<Lowered, String> {
    let core = to_core(expr);
    let mut net = compile_to_net(&core)?;
    reduce(&mut net)?;
    let readback = sym_readback(&net)?;
    let value = match &readback {
        SymExpr::Lit(Literal::Int64(n)) => Some(*n),
        _ => None,
    };
    Ok(Lowered {
        unf_hash: unf_hash_string(&net)?,
        normal_form: readback.to_prefix_string(),
        value,
        net,
    })
}

fn to_core(expr: &Expr) -> CoreIR {
    match expr {
        Expr::Num(n) => CoreIR::Lit(Literal::Int64(*n as i64)),
        Expr::Var(v) => CoreIR::Var(v.clone()),
        // There is no successor agent, so `suc e` lowers to the agent that does
        // the same arithmetic: `e + 1`. That is also the shape the reducer
        // folds, which is why a ground `suc 4` corroborates to 5.
        Expr::Suc(e) => CoreIR::Prim(
            PrimOp::Add64,
            vec![to_core(e), CoreIR::Lit(Literal::Int64(1))],
        ),
        Expr::Add(a, b) => CoreIR::Prim(PrimOp::Add64, vec![to_core(a), to_core(b)]),
        Expr::Mul(a, b) => CoreIR::Prim(PrimOp::Mul64, vec![to_core(a), to_core(b)]),
    }
}

/// What the net path says about a closed expression, checked against the value
/// the MM0 certificate is built around.
///
/// Refuses to continue on a disagreement rather than reporting it: a claim whose
/// net and whose proof disagree has no trustworthy verdict, and emitting the
/// certificate anyway would make the artifact lie about which machine produced
/// the number.
pub fn corroborate(expr: &Expr, claimed: u64) -> Result<Lowered, String> {
    let lowered = lower(expr)?;
    match lowered.value {
        Some(n) if n >= 0 && n as u64 == claimed => Ok(lowered),
        Some(n) => Err(format!(
            "the net reducer and the mm0 proof disagree about `{}`: reducer says {n}, proof is built for {claimed}",
            expr.pretty()
        )),
        None => Err(format!(
            "the net for `{}` did not reduce to a literal: {}",
            expr.pretty(),
            lowered.normal_form
        )),
    }
}

/// The isomorphism witness between two reduced nets, if they are the same net.
///
/// `Ok(Some(pairs))` is the structural half of a true equality claim: the node
/// permutation π, re-checked by [`crate::deltanet::IsomorphismWitness::verifies`]
/// inside `isomorphism_witness` itself. `Ok(None)` means the nets differ, which
/// for two *equal-valued* claims would contradict the arithmetic and is
/// reported by the caller rather than papered over.
pub fn witness(l: &Lowered, r: &Lowered) -> Result<Option<Vec<(u32, u32)>>, String> {
    Ok(isomorphism_witness(&l.net, &r.net)?.map(|w| w.pairs))
}
