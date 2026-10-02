use super::types::*;
use crate::ccg::{DerivationTree, Direction};
use crate::lexicon::{Lexicon, Literal as LexLiteral, SemExpr};

pub fn compile_to_core_ir(tree: &DerivationTree, lexicon: &Lexicon) -> Result<CoreIR, String> {
    compile_derivation(tree, lexicon)
}

fn compile_derivation(tree: &DerivationTree, lexicon: &Lexicon) -> Result<CoreIR, String> {
    match tree {
        DerivationTree::Leaf { word, .. } => {
            let template = lexicon
                .semantic_template(word)
                .ok_or_else(|| format!("no semantic template for '{}' (unknown word)", word))?;
            Ok(instantiate_template(template))
        }
        DerivationTree::Application {
            direction,
            left,
            right,
            ..
        } => match direction {
            Direction::Forward => {
                let f = compile_derivation(left, lexicon)?;
                let arg = compile_derivation(right, lexicon)?;
                Ok(CoreIR::App(Box::new(f), Box::new(arg)))
            }
            Direction::Backward => {
                let f = compile_derivation(right, lexicon)?;
                let arg = compile_derivation(left, lexicon)?;
                Ok(CoreIR::App(Box::new(f), Box::new(arg)))
            }
        },
        DerivationTree::Composition { left, right, .. } => {
            let f = compile_derivation(left, lexicon)?;
            let g = compile_derivation(right, lexicon)?;
            let z = fresh_id();
            Ok(CoreIR::Lam(
                z.clone(),
                Box::new(CoreIR::App(
                    Box::new(f),
                    Box::new(CoreIR::App(Box::new(g), Box::new(CoreIR::Var(z)))),
                )),
            ))
        }
    }
}

fn instantiate_template(template: &SemExpr) -> CoreIR {
    match template {
        SemExpr::Var(name) => CoreIR::Var(name.clone()),
        SemExpr::Lit(lit) => CoreIR::Lit(match lit {
            LexLiteral::Int64(n) => Literal::Int64(*n),
            LexLiteral::F64(x) => Literal::F64(*x),
            LexLiteral::Bool(b) => Literal::Bool(*b),
        }),
        SemExpr::Con(tag, args) => {
            let compiled_args = args.iter().map(instantiate_template).collect();
            CoreIR::Con(tag_id(tag), compiled_args)
        }
        SemExpr::Lam(var, body) => CoreIR::Lam(var.clone(), Box::new(instantiate_template(body))),
        SemExpr::App(f, arg) => CoreIR::App(
            Box::new(instantiate_template(f)),
            Box::new(instantiate_template(arg)),
        ),
    }
}

/// The 25 constructors the L0 lexicon can produce, with the tag ids
/// [`tag_id`] assigns them.
///
/// Exposed so a lexicon extension can be *checked* rather than silently
/// degraded — see [`known_constructor`] and
/// `formalize::formalizer::DomainLexicon::with_extension`.
pub const BUILTIN_CONSTRUCTORS: &[&str] = &[
    "Love", "See", "Like", "Eat", "Sleep", "Run", "Assign", "Add", "Mul", "Sub", "Eq", "Gt", "Lt",
    "Not", "Restrict", "Give", "Big", "Small", "Red", "Blue", "Very", "Cat", "Dog", "Number",
    "And",
];

/// Whether `name` has a tag in [`tag_id`].
///
/// The distinction matters because `tag_id` returns **0** for a name it does not
/// know, and 0 is a legal tag: an unknown constructor therefore compiles to
/// `Con(0, args)` and reads back as `Unknown(...)`. Two *different* unknown
/// constructors with the same arity then produce the identical term, and so the
/// identical UNF hash — a silent identity collapse, which in the autoformalizer
/// (§14: identity is the UNF hash) would make two distinct proof steps compare
/// equal. So anything building a lexicon must ask this first.
pub fn known_constructor(name: &str) -> bool {
    BUILTIN_CONSTRUCTORS.contains(&name)
}

fn tag_id(tag: &str) -> TagId {
    match tag {
        "Love" => 1,
        "See" => 2,
        "Like" => 3,
        "Eat" => 4,
        "Sleep" => 5,
        "Run" => 6,
        "Assign" => 7,
        "Add" => 8,
        "Mul" => 9,
        "Sub" => 10,
        "Eq" => 11,
        "Gt" => 12,
        "Lt" => 13,
        "Not" => 14,
        "Restrict" => 15,
        "Give" => 16,
        "Big" => 17,
        "Small" => 18,
        "Red" => 19,
        "Blue" => 20,
        "Very" => 21,
        "Cat" => 22,
        "Dog" => 23,
        "Number" => 24,
        "And" => 25,
        _ => 0,
    }
}

use std::sync::atomic::{AtomicU32, Ordering};
static FRESH_COUNTER: AtomicU32 = AtomicU32::new(0);

fn fresh_id() -> String {
    let id = FRESH_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_v{}", id)
}
