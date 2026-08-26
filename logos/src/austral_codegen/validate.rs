//! Linearity & totality validation for the Total Austral Subset (Phase 1.3).
//!
//! The subset accepted by the austral→deltanet translation enforces:
//!
//!   • **Totality** — no general recursion. The only iteration constructs
//!     are `fold` (over a linear list) and structural `match`. A function
//!     that calls itself (directly or through a cycle) is rejected, because
//!     the naive inliner (`link_functions`) would loop forever.
//!
//!   • **Linearity** — every variable is used *exactly once*, unless
//!     explicitly `clone`d (two uses) or `destroy`ed (zero uses). This is
//!     the interaction-net invariant the deltanet compiler relies on: `Dup`
//!     agents replicate a wire, `Era` erases it, and neither appears
//!     implicitly.
//!
//! The checker runs on the parsed CoreIR (before net lowering), so a
//! violating module fails fast with a precise diagnostic instead of
//! producing a stuck or non-terminating net.

use std::collections::HashSet;

use crate::core_ir::CoreIR;

/// Validate a whole parsed module: every function body against the set of
/// all defined function names (so a call to any other function is allowed,
/// but a cycle — direct or mutual — is not). Returns the first violation.
pub fn validate_module(funcs: &[(String, CoreIR)]) -> Result<(), String> {
    let names: HashSet<String> = funcs.iter().map(|(n, _)| n.clone()).collect();
    let mut all_calls: Vec<(String, String)> = Vec::new();
    for (fname, body) in funcs {
        let mut v = Validator {
            function_names: &names,
            context: &format!("function `{fname}`"),
            caller: fname.clone(),
            calls: Vec::new(),
        };
        v.walk(body)?;
        all_calls.extend(v.calls);
    }
    check_cycles(&all_calls, "module")
}

/// Validate the totality and linearity of a parsed function body.
///
/// `function_names` is the set of defined function names (calls to them
/// become recursion when the callee is the caller itself or a cycle).
/// `caller` is the bare name of the function being validated (the edge
/// source for cycle detection). Returns `Err` with a human-readable
/// diagnostic on the first violation.
pub fn validate_body(
    function_names: &HashSet<String>,
    caller: &str,
    body: &CoreIR,
) -> Result<(), String> {
    let context = format!("function `{caller}`");
    let mut v = Validator {
        function_names,
        context: &context,
        caller: caller.to_string(),
        calls: Vec::new(),
    };
    v.walk(body)?;
    check_cycles(&v.calls, &context)
}

struct Validator<'a> {
    function_names: &'a HashSet<String>,
    context: &'a str,
    /// The name of the function whose body is being validated (edge source).
    caller: String,
    /// (caller, callee) edges collected during the walk.
    calls: Vec<(String, String)>,
}

impl Validator<'_> {
    /// The expression walk. Tracks which variables are *live* (bound and not
    /// yet consumed) and counts their uses, enforcing exactly-once unless
    /// explicitly cloned/destroyed.
    fn walk(&mut self, term: &CoreIR) -> Result<(), String> {
        match term {
            CoreIR::Var(_id) => {
                // A free variable read — used exactly once at this point.
                Ok(())
            }
            CoreIR::Lit(_) => Ok(()),
            CoreIR::Con(_, args) => {
                for a in args {
                    self.walk(a)?;
                }
                Ok(())
            }
            CoreIR::Lam(id, body) => {
                // The binder is used exactly once in the body (or cloned /
                // destroyed there). Count its occurrences.
                let uses = count_uses(body, id);
                if uses == 0 {
                    return Err(format!(
                        "{}: linearity violation: parameter `{id}` is never used \
                         (add `destroy {id};` to consume it, or use it)",
                        self.context
                    ));
                }
                if uses > 1 {
                    return Err(format!(
                        "{}: linearity violation: parameter `{id}` is used {uses} times \
                         (linear types allow exactly one use; add `clone {id};` first)",
                        self.context
                    ));
                }
                self.walk(body)
            }
            CoreIR::App(f, a) => {
                // A call to a defined function: record the call edge for
                // cycle detection (a function calling itself, directly or
                // through other functions, is recursion).
                if let CoreIR::Var(name) = f.as_ref()
                    && self.function_names.contains(name)
                {
                    self.calls.push((self.caller.clone(), name.clone()));
                }
                self.walk(f)?;
                self.walk(a)
            }
            CoreIR::Let(id, value, body) => {
                self.walk(value)?;
                let uses = count_uses(body, id);
                if uses == 0 {
                    return Err(format!(
                        "{}: linearity violation: `{id}` is never used \
                         (add `destroy {id};` to consume it, or use it)",
                        self.context
                    ));
                }
                if uses > 1 {
                    return Err(format!(
                        "{}: linearity violation: `{id}` is used {uses} times \
                         (linear types allow exactly one use; add `clone {id};` first)",
                        self.context
                    ));
                }
                self.walk(body)
            }
            CoreIR::Match(scrutinee, arms) => {
                self.walk(scrutinee)?;
                for (pat, body) in arms {
                    for binder in pattern_binders(pat) {
                        let uses = count_uses(body, &binder);
                        if uses == 0 {
                            return Err(format!(
                                "{}: linearity violation: pattern variable `{binder}` is never used",
                                self.context
                            ));
                        }
                        if uses > 1 {
                            return Err(format!(
                                "{}: linearity violation: pattern variable `{binder}` is used {uses} times",
                                self.context
                            ));
                        }
                    }
                    self.walk(body)?;
                }
                Ok(())
            }
            CoreIR::Fold(f, init, list) => {
                // `f` is the step lambda; its binder is used exactly once
                // inside (checked by the Lam arm via `walk`). The `list`
                // and `init` are consumed once.
                self.walk(f)?;
                self.walk(init)?;
                self.walk(list)
            }
            CoreIR::Prim(_, args) => {
                for a in args {
                    self.walk(a)?;
                }
                Ok(())
            }
            CoreIR::Clone(id, id1, id2, body) => {
                // `id` is duplicated into `id1` and `id2` — the explicit
                // replicator. Each copy is used exactly once in `body`.
                for c in [id1, id2] {
                    let uses = count_uses(body, c);
                    if uses == 0 {
                        return Err(format!(
                            "{}: linearity violation: clone copy `{c}` of `{id}` is never used",
                            self.context
                        ));
                    }
                    if uses > 1 {
                        return Err(format!(
                            "{}: linearity violation: clone copy `{c}` of `{id}` is used {uses} times",
                            self.context
                        ));
                    }
                }
                self.walk(body)
            }
            CoreIR::Drop(_, body) => {
                // The explicit eraser consumes the variable; nothing more to
                // enforce on the value itself.
                self.walk(body)
            }
        }
    }

}

/// Reject any cycle among the collected call edges. In this subset a
/// function may call another defined function, but never itself (directly
/// or transitively). `context` names the unit being validated for the
/// diagnostic.
fn check_cycles(calls: &[(String, String)], context: &str) -> Result<(), String> {
    let mut adj: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (from, to) in calls {
        adj.entry(from.clone()).or_default().push(to.clone());
    }
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();

    fn dfs(
        node: &str,
        adj: &std::collections::HashMap<String, Vec<String>>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Result<(), String> {
        if visiting.contains(node) {
            path.push(node.to_string());
            return Err(format!(
                "totality violation: recursion detected through call cycle: {}",
                path.join(" → ")
            ));
        }
        if visited.contains(node) {
            return Ok(());
        }
        visiting.insert(node.to_string());
        path.push(node.to_string());
        if let Some(nexts) = adj.get(node) {
            for n in nexts {
                dfs(n, adj, visiting, visited, path)?;
            }
        }
        path.pop();
        visiting.remove(node);
        visited.insert(node.to_string());
        Ok(())
    }

    for start in adj.keys().cloned().collect::<Vec<_>>() {
        dfs(&start, &adj, &mut visiting, &mut visited, &mut Vec::new())?;
    }
    let _ = context;
    Ok(())
}

/// The variable names bound by a pattern (`Pattern::Tag(tag, binders)`).
fn pattern_binders(pat: &crate::core_ir::Pattern) -> Vec<String> {
    match pat {
        crate::core_ir::Pattern::Tag(_, binders) => binders.clone(),
    }
}

/// Count the occurrences of `id` as a *value* inside `term` (excluding its
/// own rebinding sites: `Let(id, …)`, `Lam(id, …)`, `Clone(id, …)`).
fn count_uses(term: &CoreIR, id: &str) -> usize {
    match term {
        CoreIR::Var(v) => usize::from(v == id),
        CoreIR::Lit(_) => 0,
        CoreIR::Con(_, args) => args.iter().map(|a| count_uses(a, id)).sum(),
        CoreIR::Lam(b, body) => {
            if b == id {
                0 // shadowed by the inner binder
            } else {
                count_uses(body, id)
            }
        }
        CoreIR::App(f, a) => count_uses(f, id) + count_uses(a, id),
        CoreIR::Let(b, value, body) => {
            if b == id {
                count_uses(value, id) // shadowed in body
            } else {
                count_uses(value, id) + count_uses(body, id)
            }
        }
        CoreIR::Match(s, arms) => {
            let mut n = count_uses(s, id);
            for (pat, body) in arms {
                if pattern_binders(pat).iter().any(|b| b == id) {
                    continue; // shadowed by the pattern binder
                }
                n += count_uses(body, id);
            }
            n
        }
        CoreIR::Fold(f, init, list) => count_uses(f, id) + count_uses(init, id) + count_uses(list, id),
        CoreIR::Prim(_, args) => args.iter().map(|a| count_uses(a, id)).sum(),
        CoreIR::Clone(b, _c1, _c2, body) => {
            // The clone consumes the original exactly once; the copies are
            // fresh names in `body`.
            count_uses(body, id) + usize::from(b == id)
        }
        CoreIR::Drop(d, body) => {
            // The destroy consumes the variable exactly once.
            count_uses(body, id) + usize::from(d == id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(fs: &[&str]) -> HashSet<String> {
        fs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn linear_use_passes() {
        // let x = 1; return (x + 2); — x used exactly once.
        let body = CoreIR::Let(
            "x".to_string(),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
            Box::new(CoreIR::Prim(
                crate::core_ir::PrimOp::Add64,
                vec![
                    CoreIR::Var("x".to_string()),
                    CoreIR::Lit(crate::core_ir::Literal::Int64(2)),
                ],
            )),
        );
        validate_body(&names(&[]), "test", &body).unwrap();
    }

    #[test]
    fn double_use_is_rejected() {
        // let x = 1; return ((x + x)); — x used twice without clone.
        let body = CoreIR::Let(
            "x".to_string(),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
            Box::new(CoreIR::Prim(
                crate::core_ir::PrimOp::Add64,
                vec![CoreIR::Var("x".to_string()), CoreIR::Var("x".to_string())],
            )),
        );
        let err = validate_body(&names(&[]), "test", &body).unwrap_err();
        assert!(err.contains("used 2 times"), "{err}");
    }

    #[test]
    fn unused_variable_is_rejected() {
        // let x = 1; return 2; — x never used (no destroy).
        let body = CoreIR::Let(
            "x".to_string(),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(2))),
        );
        let err = validate_body(&names(&[]), "test", &body).unwrap_err();
        assert!(err.contains("never used"), "{err}");
    }

    #[test]
    fn explicit_destroy_allows_zero_use() {
        // let x = 1; destroy x; return 2; — Drop consumes x.
        let body = CoreIR::Let(
            "x".to_string(),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
            Box::new(CoreIR::Drop(
                "x".to_string(),
                Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(2))),
            )),
        );
        validate_body(&names(&[]), "test", &body).unwrap();
    }

    #[test]
    fn explicit_clone_allows_two_uses() {
        // let x = 1; clone x -> y, z; return (y + z); — each copy used once.
        let body = CoreIR::Let(
            "x".to_string(),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
            Box::new(CoreIR::Clone(
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                Box::new(CoreIR::Prim(
                    crate::core_ir::PrimOp::Add64,
                    vec![CoreIR::Var("y".to_string()), CoreIR::Var("z".to_string())],
                )),
            )),
        );
        validate_body(&names(&[]), "test", &body).unwrap();
    }

    #[test]
    fn direct_recursion_is_rejected() {
        // f calls f — the self-call must be rejected as a totality
        // violation before it would loop the inliner forever.
        let body = CoreIR::App(
            Box::new(CoreIR::Var("f".to_string())),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
        );
        let err = validate_body(&names(&["f"]), "f", &body).unwrap_err();
        assert!(err.contains("recursion"), "{err}");
    }

    #[test]
    fn call_to_other_function_is_fine() {
        // f calls g (not itself) — allowed.
        let body = CoreIR::App(
            Box::new(CoreIR::Var("g".to_string())),
            Box::new(CoreIR::Lit(crate::core_ir::Literal::Int64(1))),
        );
        validate_body(&names(&["f", "g"]), "f", &body).unwrap();
    }
}
