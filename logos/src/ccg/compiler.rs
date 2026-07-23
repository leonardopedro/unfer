use super::types::*;
use crate::lexicon::{Lexicon, SemExpr};
use crate::core_ir::{CoreIR, Literal, TagId};

pub fn compile_derivation(tree: &DerivationTree, lexicon: &Lexicon) -> CoreIR {
    match tree {
        DerivationTree::Leaf { word, .. } => {
            let template = lexicon.semantic_template(word)
                .unwrap_or_else(|| panic!("no semantic template for '{}'", word));
            instantiate_template(template)
        }
        DerivationTree::Application { direction, left, right, .. } => {
            match direction {
                Direction::Forward => {
                    let f = compile_derivation(left, lexicon);
                    let arg = compile_derivation(right, lexicon);
                    CoreIR::App(Box::new(f), Box::new(arg))
                }
                Direction::Backward => {
                    let f = compile_derivation(right, lexicon);
                    let arg = compile_derivation(left, lexicon);
                    CoreIR::App(Box::new(f), Box::new(arg))
                }
            }
        }
        DerivationTree::Composition { left, right, .. } => {
            let f = compile_derivation(left, lexicon);
            let g = compile_derivation(right, lexicon);
            let z = fresh_id();
            CoreIR::Lam(
                z.clone(),
                Box::new(CoreIR::App(
                    Box::new(f),
                    Box::new(CoreIR::App(
                        Box::new(g),
                        Box::new(CoreIR::Var(z)),
                    )),
                )),
            )
        }
    }
}

fn instantiate_template(template: &SemExpr) -> CoreIR {
    match template {
        SemExpr::Var(name) => CoreIR::Var(name.clone()),
        SemExpr::Lit(lit) => CoreIR::Lit(match lit {
            crate::lexicon::Literal::Int64(n) => Literal::Int64(*n),
            crate::lexicon::Literal::Bool(b) => Literal::Bool(*b),
        }),
        SemExpr::Con(tag, args) => {
            let compiled_args = args.iter().map(instantiate_template).collect();
            CoreIR::Con(tag_id(tag), compiled_args)
        }
        SemExpr::Lam(var, body) => {
            CoreIR::Lam(var.clone(), Box::new(instantiate_template(body)))
        }
        SemExpr::App(f, arg) => {
            CoreIR::App(
                Box::new(instantiate_template(f)),
                Box::new(instantiate_template(arg)),
            )
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexicon::Literal as LexLiteral;

    #[test]
    fn test_instantiate_var() {
        let template = SemExpr::Var("john".to_string());
        let ir = instantiate_template(&template);
        assert_eq!(ir, CoreIR::Var("john".to_string()));
    }

    #[test]
    fn test_instantiate_lit() {
        let template = SemExpr::Lit(LexLiteral::Int64(42));
        let ir = instantiate_template(&template);
        assert_eq!(ir, CoreIR::Lit(Literal::Int64(42)));
    }

    #[test]
    fn test_instantiate_con() {
        let template = SemExpr::Con("Add".to_string(), vec![
            SemExpr::Lit(LexLiteral::Int64(1)),
            SemExpr::Lit(LexLiteral::Int64(2)),
        ]);
        let ir = instantiate_template(&template);
        match ir {
            CoreIR::Con(tag, args) => {
                assert_eq!(tag, 8);
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Con"),
        }
    }
}
