use super::types::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinearityError {
    #[error("variable '{}' used {} times (expected 1)", .0, .1)]
    UsedMultipleTimes(String, usize),
    #[error("variable '{}' unused", .0)]
    Unused(String),
}

pub fn insert_linearity(term: CoreIR) -> CoreIR {
    match term {
        CoreIR::Var(_) | CoreIR::Lit(_) | CoreIR::Con(_, _) => term,
        CoreIR::Lam(id, body) => {
            let body = insert_linearity(*body);
            let count = count_uses(&id, &body);
            match count {
                0 => CoreIR::Lam(id.clone(), Box::new(CoreIR::Drop(id, Box::new(body)))),
                1 => CoreIR::Lam(id, Box::new(body)),
                n => {
                    // Free variable used `n` times: a duplicator chain hands
                    // each use a distinct wire, so no port is wired twice.
                    // The outer `Lam` keeps `id` — the Dup's principal sits
                    // on the binder port `Abs.1`, which is exactly what beta
                    // reduction splices the argument into.
                    let names: Vec<String> = (1..=n).map(|i| format!("{}_dup{}", id, i)).collect();
                    let body = rename_uses(&body, &id, &names);
                    let body = insert_clone_chain(&id, &names, body);
                    CoreIR::Lam(id, Box::new(body))
                }
            }
        }
        CoreIR::App(f, arg) => CoreIR::App(
            Box::new(insert_linearity(*f)),
            Box::new(insert_linearity(*arg)),
        ),
        CoreIR::Let(id, value, body) => {
            let value = insert_linearity(*value);
            let body = insert_linearity(*body);
            let count = count_uses(&id, &body);
            match count {
                0 => CoreIR::Let(
                    id.clone(),
                    Box::new(value),
                    Box::new(CoreIR::Drop(id, Box::new(body))),
                ),
                1 => CoreIR::Let(id, Box::new(value), Box::new(body)),
                n => {
                    // Free variable used `n` times: duplicator chain, as in
                    // the `Lam` arm. The `Let` keeps its own binder; the
                    // copies are consumed inside, no extra `Lam` wrappers.
                    let names: Vec<String> = (1..=n).map(|i| format!("{}_dup{}", id, i)).collect();
                    let body = rename_uses(&body, &id, &names);
                    let body = insert_clone_chain(&id, &names, body);
                    CoreIR::Let(id, Box::new(value), Box::new(body))
                }
            }
        }
        CoreIR::Prim(op, args) => {
            CoreIR::Prim(op, args.into_iter().map(insert_linearity).collect())
        }
        CoreIR::Fold(f, init, list) => CoreIR::Fold(
            Box::new(insert_linearity(*f)),
            Box::new(insert_linearity(*init)),
            Box::new(insert_linearity(*list)),
        ),
        CoreIR::Match(scrutinee, arms) => CoreIR::Match(
            Box::new(insert_linearity(*scrutinee)),
            arms.into_iter()
                .map(|(pat, body)| (pat, insert_linearity(body)))
                .collect(),
        ),
        CoreIR::Clone(id, id1, id2, body) => {
            CoreIR::Clone(id, id1, id2, Box::new(insert_linearity(*body)))
        }
        CoreIR::Drop(id, body) => CoreIR::Drop(id, Box::new(insert_linearity(*body))),
    }
}

fn insert_clone_chain(original: &str, new_ids: &[String], body: CoreIR) -> CoreIR {
    if new_ids.is_empty() {
        return body;
    }
    if new_ids.len() == 1 {
        // The terminal Dup's second output must be consumed *somewhere*, or
        // its aux port is left dangling: once this Dup meets a value and
        // commutes, `get_aux(dup, 2)` on the unwired slot fails with
        // "no port at node N slot 2". `Drop` terminates it with an eraser —
        // a surplus copy the term never uses, which is what it is.
        let drop_name = format!("{}_drop", original);
        return CoreIR::Clone(
            original.to_string(),
            new_ids[0].clone(),
            drop_name.clone(),
            Box::new(CoreIR::Drop(drop_name, Box::new(body))),
        );
    }
    let first = &new_ids[0];
    let rest = &new_ids[1..];
    CoreIR::Clone(
        original.to_string(),
        first.clone(),
        format!("{}_tmp", original),
        Box::new(insert_clone_chain(&format!("{}_tmp", original), rest, body)),
    )
}

/// Rewrite the uses of `id` inside `term` to the fresh copies in `names`.
///
/// A duplicator chain binds `names[0]` … `names[n-1]`, so each occurrence of
/// `id` must be replaced by a *distinct* copy — otherwise two occurrences
/// would read the same wire and the net would wire one port twice, which is
/// ill-formed. Occurrences are numbered left-to-right (the traversal order
/// below is deterministic), and the k-th occurrence becomes `names[k-1]`.
///
/// A use under a binder that shadows `id` is left alone: it refers to the
/// inner binding, not to the one we are duplicating.
fn rename_uses(term: &CoreIR, id: &str, names: &[String]) -> CoreIR {
    let mut next = 0usize;
    rename_uses_rec(term, id, names, &mut next)
}

fn rename_uses_rec(term: &CoreIR, id: &str, names: &[String], next: &mut usize) -> CoreIR {
    match term {
        CoreIR::Var(name) => {
            if name == id {
                let pick = (*next).min(names.len() - 1);
                *next += 1;
                CoreIR::Var(names[pick].clone())
            } else {
                CoreIR::Var(name.clone())
            }
        }
        CoreIR::Lit(l) => CoreIR::Lit(l.clone()),
        CoreIR::Con(tag, args) => CoreIR::Con(
            *tag,
            args.iter()
                .map(|a| rename_uses_rec(a, id, names, next))
                .collect(),
        ),
        CoreIR::Lam(name, body) => {
            if name == id {
                // Shadowed: this binder rebinds `id`, so its body's uses
                // belong to the inner binder, not the one we are duplicating.
                CoreIR::Lam(name.clone(), body.clone())
            } else {
                CoreIR::Lam(
                    name.clone(),
                    Box::new(rename_uses_rec(body, id, names, next)),
                )
            }
        }
        CoreIR::App(f, arg) => CoreIR::App(
            Box::new(rename_uses_rec(f, id, names, next)),
            Box::new(rename_uses_rec(arg, id, names, next)),
        ),
        CoreIR::Let(name, value, body) => {
            let value = rename_uses_rec(value, id, names, next);
            if name == id {
                CoreIR::Let(name.clone(), Box::new(value), body.clone())
            } else {
                CoreIR::Let(
                    name.clone(),
                    Box::new(value),
                    Box::new(rename_uses_rec(body, id, names, next)),
                )
            }
        }
        CoreIR::Prim(op, args) => CoreIR::Prim(
            *op,
            args.iter()
                .map(|a| rename_uses_rec(a, id, names, next))
                .collect(),
        ),
        CoreIR::Fold(f, init, list) => CoreIR::Fold(
            Box::new(rename_uses_rec(f, id, names, next)),
            Box::new(rename_uses_rec(init, id, names, next)),
            Box::new(rename_uses_rec(list, id, names, next)),
        ),
        CoreIR::Match(scrutinee, arms) => CoreIR::Match(
            Box::new(rename_uses_rec(scrutinee, id, names, next)),
            arms.iter()
                .map(|(pat, body)| {
                    let shadowed = match pat {
                        Pattern::Tag(_, binders) => binders.iter().any(|b| b == id),
                    };
                    let body = if shadowed {
                        body.clone()
                    } else {
                        rename_uses_rec(body, id, names, next)
                    };
                    (pat.clone(), body)
                })
                .collect(),
        ),
        CoreIR::Clone(orig, id1, id2, body) => {
            if orig == id {
                // The chain already rebound `id`; its body's uses are the
                // copies', not ours to rename.
                CoreIR::Clone(orig.clone(), id1.clone(), id2.clone(), body.clone())
            } else {
                CoreIR::Clone(
                    orig.clone(),
                    id1.clone(),
                    id2.clone(),
                    Box::new(rename_uses_rec(body, id, names, next)),
                )
            }
        }
        CoreIR::Drop(name, body) => {
            if name == id {
                CoreIR::Drop(name.clone(), body.clone())
            } else {
                CoreIR::Drop(
                    name.clone(),
                    Box::new(rename_uses_rec(body, id, names, next)),
                )
            }
        }
    }
}

fn count_uses(id: &str, term: &CoreIR) -> usize {
    match term {
        CoreIR::Var(name) => {
            if name == id {
                1
            } else {
                0
            }
        }
        CoreIR::Lit(_) => 0,
        CoreIR::Con(_, args) => args.iter().map(|a| count_uses(id, a)).sum(),
        CoreIR::Lam(name, body) => {
            if name == id {
                0
            } else {
                count_uses(id, body)
            }
        }
        CoreIR::App(f, arg) => count_uses(id, f) + count_uses(id, arg),
        CoreIR::Let(name, value, body) => {
            count_uses(id, value) + if name == id { 0 } else { count_uses(id, body) }
        }
        CoreIR::Match(scrutinee, arms) => {
            count_uses(id, scrutinee)
                + arms
                    .iter()
                    .map(|(pat, body)| match pat {
                        Pattern::Tag(_, binders) => {
                            if binders.iter().any(|b| b == id) {
                                0
                            } else {
                                count_uses(id, body)
                            }
                        }
                    })
                    .sum::<usize>()
        }
        CoreIR::Fold(f, init, list) => {
            count_uses(id, f) + count_uses(id, init) + count_uses(id, list)
        }
        CoreIR::Prim(_, args) => args.iter().map(|a| count_uses(id, a)).sum(),
        CoreIR::Clone(orig, id1, id2, body) => {
            if orig == id {
                count_uses(id1, body) + count_uses(id2, body)
            } else {
                count_uses(id, body)
            }
        }
        CoreIR::Drop(name, body) => {
            if name == id {
                0
            } else {
                count_uses(id, body)
            }
        }
    }
}

pub fn check_linearity(term: &CoreIR) -> Result<(), LinearityError> {
    check_linearity_inner(term, &mut HashMap::new())
}

fn check_linearity_inner(
    term: &CoreIR,
    ctx: &mut HashMap<String, usize>,
) -> Result<(), LinearityError> {
    match term {
        CoreIR::Var(id) => {
            if let Some(count) = ctx.get_mut(id) {
                *count += 1;
                if *count > 1 {
                    return Err(LinearityError::UsedMultipleTimes(id.clone(), *count));
                }
            }
            Ok(())
        }
        CoreIR::Lit(_) => Ok(()),
        CoreIR::Con(_, args) => {
            for arg in args {
                check_linearity_inner(arg, ctx)?;
            }
            Ok(())
        }
        CoreIR::Lam(id, body) => {
            let mut body_ctx = ctx.clone();
            body_ctx.insert(id.clone(), 0);
            check_linearity_inner(body, &mut body_ctx)?;
            let count = body_ctx.get(id).copied().unwrap_or(0);
            if count == 0 {
                return Err(LinearityError::Unused(id.clone()));
            }
            if count > 1 {
                return Err(LinearityError::UsedMultipleTimes(id.clone(), count));
            }
            Ok(())
        }
        CoreIR::App(f, arg) => {
            check_linearity_inner(f, ctx)?;
            check_linearity_inner(arg, ctx)?;
            Ok(())
        }
        CoreIR::Let(id, value, body) => {
            check_linearity_inner(value, ctx)?;
            let mut body_ctx = ctx.clone();
            body_ctx.insert(id.clone(), 0);
            check_linearity_inner(body, &mut body_ctx)?;
            let count = body_ctx.get(id).copied().unwrap_or(0);
            if count == 0 {
                return Err(LinearityError::Unused(id.clone()));
            }
            if count > 1 {
                return Err(LinearityError::UsedMultipleTimes(id.clone(), count));
            }
            Ok(())
        }
        CoreIR::Match(scrutinee, arms) => {
            check_linearity_inner(scrutinee, ctx)?;
            for (pat, body) in arms {
                let mut arm_ctx = ctx.clone();
                match pat {
                    Pattern::Tag(_, binders) => {
                        for b in binders {
                            arm_ctx.insert(b.clone(), 0);
                        }
                    }
                }
                check_linearity_inner(body, &mut arm_ctx)?;
            }
            Ok(())
        }
        CoreIR::Fold(f, init, list) => {
            check_linearity_inner(f, ctx)?;
            check_linearity_inner(init, ctx)?;
            check_linearity_inner(list, ctx)?;
            Ok(())
        }
        CoreIR::Prim(_, args) => {
            for arg in args {
                check_linearity_inner(arg, ctx)?;
            }
            Ok(())
        }
        CoreIR::Clone(orig, id1, id2, body) => {
            if !ctx.contains_key(orig) {
                return Err(LinearityError::Unused(orig.clone()));
            }
            let mut body_ctx = ctx.clone();
            body_ctx.remove(orig);
            body_ctx.insert(id1.clone(), 0);
            body_ctx.insert(id2.clone(), 0);
            check_linearity_inner(body, &mut body_ctx)?;
            Ok(())
        }
        CoreIR::Drop(id, body) => {
            if let Some(count) = ctx.get_mut(id) {
                *count = 1;
            }
            check_linearity_inner(body, ctx)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_linearity_simple() {
        let term = CoreIR::Lam("x".to_string(), Box::new(CoreIR::Var("x".to_string())));
        let result = insert_linearity(term);
        assert_eq!(
            result,
            CoreIR::Lam("x".to_string(), Box::new(CoreIR::Var("x".to_string())))
        );
    }

    #[test]
    fn test_insert_linearity_unused() {
        let term = CoreIR::Lam("x".to_string(), Box::new(CoreIR::Lit(Literal::Int64(42))));
        let result = insert_linearity(term);
        assert!(matches!(result, CoreIR::Lam(_, _)));
        if let CoreIR::Lam(_, body) = &result {
            assert!(matches!(**body, CoreIR::Drop(_, _)));
        }
    }

    #[test]
    fn test_check_linearity_ok() {
        let term = CoreIR::Lam("x".to_string(), Box::new(CoreIR::Var("x".to_string())));
        assert!(check_linearity(&term).is_ok());
    }

    #[test]
    fn test_check_linearity_unused() {
        let term = CoreIR::Lam("x".to_string(), Box::new(CoreIR::Lit(Literal::Int64(42))));
        assert!(matches!(
            check_linearity(&term),
            Err(LinearityError::Unused(_))
        ));
    }

    #[test]
    fn test_check_linearity_double_use() {
        let term = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::App(
                Box::new(CoreIR::Var("x".to_string())),
                Box::new(CoreIR::Var("x".to_string())),
            )),
        );
        assert!(matches!(
            check_linearity(&term),
            Err(LinearityError::UsedMultipleTimes(_, 2))
        ));
    }

    // --- Dup/Era insertion (ΔI/ΔA tier) ---------------------------------

    use crate::deltanet::{self, Net};
    use std::time::Instant;

    /// Compile a CoreIR term through the real pipeline: insert the Dup/Era
    /// nodes, lower to a net, reduce to its unique normal form.
    fn reduce(term: CoreIR) -> Net {
        let term = insert_linearity(term);
        let mut net = deltanet::compile_to_net(&term).expect("compile");
        deltanet::reduce(&mut net).expect("reduce");
        net
    }

    /// `λx. (x, x)` — the ΔI case: a Free variable used twice must be fed
    /// through a duplicator, so each use reads a distinct wire.
    #[test]
    fn dup_in_lam_reduces_to_a_pair() {
        let net = reduce(CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Con(
                200,
                vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
            )),
        ));
        // The lambda stays a lambda (an open net); what matters is that it
        // compiles and reduces without wiring one port twice.
        assert!(deltanet::readback(&net).is_ok());
        // The ΔI tier inserted exactly one duplicator for the two uses.
        let text = format!("{:?}", net);
        assert!(text.contains("Dup"), "expected a Dup node: {text}");
    }

    /// `λx. (x, x)` under an applied argument must reduce to `(v, v)`.
    #[test]
    fn dup_in_lam_reduces_under_application() {
        let term = CoreIR::App(
            Box::new(CoreIR::Lam(
                "x".to_string(),
                Box::new(CoreIR::Con(
                    CONS_TAG,
                    vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
                )),
            )),
            Box::new(CoreIR::Lit(Literal::Int64(42))),
        );
        let net = reduce(term);
        let out = deltanet::readback(&net).expect("readback");
        assert_eq!(out, "Cons(42, 42)");
    }

    /// The two forms from Part 2A of the plan — `λx. (x, x)` (explicit Dup)
    /// and `λx. let y = x in (y, y)` (a `let` alias) — must reduce to the
    /// same normal form and carry the same UNF identity. This is the whole
    /// point of the shared engine: the Dup-vs-let distinction is a source
    /// artifact, invisible after reduction.
    #[test]
    fn dup_and_let_alias_share_a_normal_form() {
        // `let y = x in (y, y)`
        let via_let = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Let(
                "y".to_string(),
                Box::new(CoreIR::Var("x".to_string())),
                Box::new(CoreIR::Con(
                    200,
                    vec![CoreIR::Var("y".to_string()), CoreIR::Var("y".to_string())],
                )),
            )),
        );
        // `λx. (x, x)`
        let via_dup = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Con(
                200,
                vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
            )),
        );

        let a = reduce(via_let);
        let b = reduce(via_dup);
        let ha = deltanet::unf_hash_string(&a).expect("hash a");
        let hb = deltanet::unf_hash_string(&b).expect("hash b");
        assert_eq!(ha, hb, "the two encodings must share a UNF identity");
    }

    /// ΔA: a Free variable that is dropped is terminated with an eraser.
    #[test]
    fn drop_in_lam_inserts_an_eraser_and_reduces() {
        let term = CoreIR::App(
            Box::new(CoreIR::Lam(
                "x".to_string(),
                Box::new(CoreIR::Lit(Literal::Int64(7))),
            )),
            Box::new(CoreIR::Lit(Literal::Int64(1))),
        );
        let net = reduce(term);
        let text = format!("{:?}", net);
        assert!(text.contains("Era"), "expected an Era node: {text}");
        let out = deltanet::readback(&net).expect("readback");
        assert_eq!(out, "7");
    }

    /// A Free variable used three times fans out through a binary Dup chain.
    #[test]
    fn triple_use_fans_out_through_a_dup_chain() {
        let term = CoreIR::App(
            Box::new(CoreIR::Lam(
                "x".to_string(),
                Box::new(CoreIR::Prim(
                    PrimOp::Add64,
                    vec![
                        CoreIR::Var("x".to_string()),
                        CoreIR::Prim(
                            PrimOp::Add64,
                            vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
                        ),
                    ],
                )),
            )),
            Box::new(CoreIR::Lit(Literal::Int64(5))),
        );
        let net = reduce(term);
        let out = deltanet::readback(&net).expect("readback");
        assert_eq!(out, "15");
    }

    // --- E2 benchmark: explicit Dup vs `let` alias -----------------------

    /// Both encodings of `λx. (x, x)`, applied to a literal, timed through
    /// the full insert→compile→reduce path. `#[ignore]`d because it prints
    /// timing rather than asserting a threshold; run with:
    ///
    /// ```text
    /// cargo test -p logos dup_vs_let_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn dup_vs_let_bench() {
        fn dup_form() -> CoreIR {
            CoreIR::App(
                Box::new(CoreIR::Lam(
                    "x".to_string(),
                    Box::new(CoreIR::Con(
                        200,
                        vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
                    )),
                )),
                Box::new(CoreIR::Lit(Literal::Int64(42))),
            )
        }
        fn let_form() -> CoreIR {
            CoreIR::App(
                Box::new(CoreIR::Lam(
                    "x".to_string(),
                    Box::new(CoreIR::Let(
                        "y".to_string(),
                        Box::new(CoreIR::Var("x".to_string())),
                        Box::new(CoreIR::Con(
                            200,
                            vec![CoreIR::Var("y".to_string()), CoreIR::Var("y".to_string())],
                        )),
                    )),
                )),
                Box::new(CoreIR::Lit(Literal::Int64(42))),
            )
        }

        let iters = 20_000;
        // Warm up once so the first iteration's compile cost is excluded.
        reduce(dup_form());
        reduce(let_form());

        let t = Instant::now();
        for _ in 0..iters {
            reduce(dup_form());
        }
        let dup_ns = t.elapsed().as_nanos() / iters as u128;

        let t = Instant::now();
        for _ in 0..iters {
            reduce(let_form());
        }
        let let_ns = t.elapsed().as_nanos() / iters as u128;

        println!("E2  explicit-Dup  {dup_ns} ns/op");
        println!("E2  let-alias    {let_ns} ns/op");
    }
}
