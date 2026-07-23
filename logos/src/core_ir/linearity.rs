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
                _ => {
                    let id1 = format!("{}_1", id);
                    let id2 = format!("{}_2", id);
                    let new_body = insert_clone_chain(&id, &[id1.clone(), id2.clone()], body);
                    CoreIR::Lam(id1, Box::new(CoreIR::Lam(id2, Box::new(new_body))))
                }
            }
        }
        CoreIR::App(f, arg) => {
            CoreIR::App(Box::new(insert_linearity(*f)), Box::new(insert_linearity(*arg)))
        }
        CoreIR::Let(id, value, body) => {
            let value = insert_linearity(*value);
            let body = insert_linearity(*body);
            let count = count_uses(&id, &body);
            match count {
                0 => CoreIR::Let(id.clone(), Box::new(value), Box::new(CoreIR::Drop(id, Box::new(body)))),
                1 => CoreIR::Let(id, Box::new(value), Box::new(body)),
                _ => {
                    let ids: Vec<String> = (1..=count).map(|i| format!("{}_{}", id, i)).collect();
                    let new_body = insert_clone_chain(&id, &ids, body);
                    let mut result = new_body;
                    for new_id in ids.iter().rev() {
                        result = CoreIR::Lam(new_id.clone(), Box::new(result));
                    }
                    CoreIR::Let(id, Box::new(value), Box::new(result))
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
        return CoreIR::Clone(
            original.to_string(),
            new_ids[0].clone(),
            format!("{}_drop", original),
            Box::new(body),
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

fn count_uses(id: &str, term: &CoreIR) -> usize {
    match term {
        CoreIR::Var(name) => if name == id { 1 } else { 0 },
        CoreIR::Lit(_) => 0,
        CoreIR::Con(_, args) => args.iter().map(|a| count_uses(id, a)).sum(),
        CoreIR::Lam(name, body) => {
            if name == id { 0 } else { count_uses(id, body) }
        }
        CoreIR::App(f, arg) => count_uses(id, f) + count_uses(id, arg),
        CoreIR::Let(name, value, body) => {
            count_uses(id, value) + if name == id { 0 } else { count_uses(id, body) }
        }
        CoreIR::Match(scrutinee, arms) => {
            count_uses(id, scrutinee)
                + arms.iter().map(|(pat, body)| {
                    match pat {
                        Pattern::Tag(_, binders) => {
                            if binders.iter().any(|b| b == id) {
                                0
                            } else {
                                count_uses(id, body)
                            }
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
            if name == id { 0 } else { count_uses(id, body) }
        }
    }
}

pub fn check_linearity(term: &CoreIR) -> Result<(), LinearityError> {
    check_linearity_inner(term, &mut HashMap::new())
}

fn check_linearity_inner(term: &CoreIR, ctx: &mut HashMap<String, usize>) -> Result<(), LinearityError> {
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
        let term = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Var("x".to_string())),
        );
        let result = insert_linearity(term);
        assert_eq!(result, CoreIR::Lam("x".to_string(), Box::new(CoreIR::Var("x".to_string()))));
    }

    #[test]
    fn test_insert_linearity_unused() {
        let term = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Lit(Literal::Int64(42))),
        );
        let result = insert_linearity(term);
        assert!(matches!(result, CoreIR::Lam(_, _)));
        if let CoreIR::Lam(_, body) = &result {
            assert!(matches!(**body, CoreIR::Drop(_, _)));
        }
    }

    #[test]
    fn test_check_linearity_ok() {
        let term = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Var("x".to_string())),
        );
        assert!(check_linearity(&term).is_ok());
    }

    #[test]
    fn test_check_linearity_unused() {
        let term = CoreIR::Lam(
            "x".to_string(),
            Box::new(CoreIR::Lit(Literal::Int64(42))),
        );
        assert!(matches!(check_linearity(&term), Err(LinearityError::Unused(_))));
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
        assert!(matches!(check_linearity(&term), Err(LinearityError::UsedMultipleTimes(_, 2))));
    }
}
