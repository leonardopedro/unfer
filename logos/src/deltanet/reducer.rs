use super::types::*;
use crate::core_ir::{PrimOp, Literal, NIL_TAG, CONS_TAG};

pub fn reduce(net: &mut Net) {
    net.collect_active_pairs();
    let mut iterations = 0;
    let max_iterations = 1_000_000;

    while let Some((a, b)) = net.active_pairs.pop() {
        if net.is_freed(a) || net.is_freed(b) {
            continue;
        }
        interact(net, a, b);
        net.collect_new_active_pairs();
        iterations += 1;
        if iterations >= max_iterations {
            eprintln!("warning: reduction exceeded {} iterations, possible non-termination", max_iterations);
            break;
        }
    }
}

fn interact(net: &mut Net, a: NodeId, b: NodeId) {
    let kind_a = net.nodes[a as usize].as_ref().map(|n| n.kind.clone());
    let kind_b = net.nodes[b as usize].as_ref().map(|n| n.kind.clone());

    let (kind_a, kind_b) = match (kind_a, kind_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };

    match (&kind_a, &kind_b) {
        // Beta reduction: App >< Abs
        (AgentKind::App, AgentKind::Abs) => {
            let app_arg = net.get_aux(a, 1);
            let abs_var = net.get_aux(b, 1);
            net.wire(app_arg, abs_var);
            let abs_body = net.get_aux(b, 2);
            let app_result = net.nodes[a as usize].as_ref().and_then(|n| n.ports[2].clone());
            if let Some(outer) = app_result {
                net.wire(outer, abs_body);
            } else {
                if let Some(node) = net.nodes[a as usize].as_mut() {
                    node.ports[2] = Some(abs_body.clone());
                }
                if let Some(node) = net.nodes[abs_body.node as usize].as_mut() {
                    node.ports[abs_body.slot as usize] = Some(Port::new(a, 2));
                }
            }
            net.free_node(a);
            net.free_node(b);
        }

        // Abs >< App (commutative)
        (AgentKind::Abs, AgentKind::App) => {
            let abs_var = net.get_aux(a, 1);
            let app_arg = net.get_aux(b, 1);
            net.wire(abs_var, app_arg);
            let abs_body = net.get_aux(a, 2);
            let app_result = net.nodes[b as usize].as_ref().and_then(|n| n.ports[2].clone());
            if let Some(outer) = app_result {
                net.wire(outer, abs_body);
            } else {
                if let Some(node) = net.nodes[b as usize].as_mut() {
                    node.ports[2] = Some(abs_body.clone());
                }
                if let Some(node) = net.nodes[abs_body.node as usize].as_mut() {
                    node.ports[abs_body.slot as usize] = Some(Port::new(b, 2));
                }
            }
            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Dup (same level): annihilate
        (AgentKind::Dup(l1), AgentKind::Dup(l2)) if l1 == l2 => {
            let a1 = net.get_aux(a, 1);
            let a2 = net.get_aux(a, 2);
            let b1 = net.get_aux(b, 1);
            let b2 = net.get_aux(b, 2);
            net.wire(a1, b1);
            net.wire(a2, b2);
            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Dup (different level): commute
        (AgentKind::Dup(l1), AgentKind::Dup(l2)) if l1 != l2 => {
            let dup_a = net.get_aux(a, 1);
            let dup_a2 = net.get_aux(a, 2);
            let _dup_b = net.get_aux(b, 1);
            let _dup_b2 = net.get_aux(b, 2);

            let new_a1 = net.alloc_node(AgentKind::Dup(*l1));
            let new_a2 = net.alloc_node(AgentKind::Dup(*l1));
            let new_b1 = net.alloc_node(AgentKind::Dup(*l2));
            let new_b2 = net.alloc_node(AgentKind::Dup(*l2));

            net.wire(Port::principal(new_a1), dup_a);
            net.wire(Port::new(new_a1, 1), Port::principal(new_b1));
            net.wire(Port::new(new_a1, 2), Port::principal(new_b2));

            net.wire(Port::principal(new_a2), dup_a2);
            net.wire(Port::new(new_a2, 1), Port::principal(new_b1));
            net.wire(Port::new(new_a2, 2), Port::principal(new_b2));

            net.free_node(a);
            net.free_node(b);
        }

        // Dup >< Con/Abs/App/Fold/Prim: commute
        (AgentKind::Dup(_), _) => {
            let dup_level = match &kind_a {
                AgentKind::Dup(l) => *l,
                _ => unreachable!(),
            };
            commute_dup_through(net, a, b, dup_level, true);
        }
        (_, AgentKind::Dup(_)) => {
            let dup_level = match &kind_b {
                AgentKind::Dup(l) => *l,
                _ => unreachable!(),
            };
            commute_dup_through(net, b, a, dup_level, false);
        }

        // Era >< anything: erase
        (AgentKind::Era, _) => {
            erase_agent(net, b);
            net.free_node(a);
            net.free_node(b);
        }
        (_, AgentKind::Era) => {
            erase_agent(net, a);
            net.free_node(a);
            net.free_node(b);
        }

        // Fold >< Con(Nil, 0): reduce to init
        (AgentKind::Fold, AgentKind::Con(tag, 0)) if *tag == NIL_TAG => {
            let init_port = net.get_aux(a, 2);
            let result_port = Port::principal(a);
            net.wire(result_port, init_port);
            net.free_node(a);
            net.free_node(b);
        }

        // Fold >< Con(Cons, 2): reduce to f(head, Fold(f, init, tail))
        (AgentKind::Fold, AgentKind::Con(tag, 2)) if *tag == CONS_TAG => {
            let f_port = net.get_aux(a, 1);
            let init_port = net.get_aux(a, 2);
            let head_port = net.get_aux(b, 1);
            let tail_port = net.get_aux(b, 2);

            let inner_fold = net.alloc_node(AgentKind::Fold);
            let inner_app = net.alloc_node(AgentKind::App);
            let outer_app = net.alloc_node(AgentKind::App);

            let dup_f = net.alloc_node(AgentKind::Dup(0));
            let dup_init = net.alloc_node(AgentKind::Dup(0));

            net.wire(Port::principal(dup_f), f_port);
            let f1 = Port::new(dup_f, 1);
            let f2 = Port::new(dup_f, 2);

            net.wire(Port::principal(dup_init), init_port);
            let _init1 = Port::new(dup_init, 1);
            let init2 = Port::new(dup_init, 2);

            net.wire(Port::new(inner_fold, 1), f2);
            net.wire(Port::new(inner_fold, 2), init2);
            net.wire(Port::new(inner_fold, 3), tail_port);

            net.wire(Port::principal(inner_app), f1);
            net.wire(Port::new(inner_app, 1), Port::principal(inner_fold));

            net.wire(Port::principal(outer_app), Port::new(inner_app, 2));
            net.wire(Port::new(outer_app, 1), head_port);

            net.wire(Port::principal(a), Port::new(outer_app, 2));

            net.free_node(a);
            net.free_node(b);
        }

        // Prim >< Lit, Lit: native evaluation
        (AgentKind::Prim(op), AgentKind::Lit(_)) => {
            if let (Some(lit1), Some(lit2)) = (net.get_connected_lit(a, 1), net.get_connected_lit(a, 2)) {
                if let Some(result) = eval_prim(*op, &lit1, &lit2) {
                    let result_node = net.alloc_node(AgentKind::Lit(result));
                    net.wire(Port::principal(a), Port::principal(result_node));
                    net.free_node(a);
                }
            }
        }

        // Con >< Con: STUCK TERM
        (AgentKind::Con(_, _), AgentKind::Con(_, _)) => {
            // Two constructors meeting principal-to-principal indicates a type error
            // For v1, we just leave them stuck
        }

        _ => {}
    }
}

fn commute_dup_through(net: &mut Net, dup_id: NodeId, other_id: NodeId, level: u16, dup_is_left: bool) {
    let dup_port = net.get_aux(dup_id, if dup_is_left { 1 } else { 2 });
    let _dup_port2 = net.get_aux(dup_id, if dup_is_left { 2 } else { 1 });

    let other_kind = net.nodes[other_id as usize].as_ref().unwrap().kind.clone();
    let other_aux_count = other_kind.aux_count();

    let new_other1 = net.alloc_node(other_kind.clone());
    let new_other2 = net.alloc_node(other_kind);

    for slot in 1..=other_aux_count {
        let orig = net.get_aux(other_id, slot);
        let new1_port = Port::new(new_other1, slot);
        let new2_port = Port::new(new_other2, slot);
        let dup1 = net.alloc_node(AgentKind::Dup(level));
        let dup2 = net.alloc_node(AgentKind::Dup(level));

        net.wire(Port::principal(dup1), orig.clone());
        net.wire(Port::new(dup1, 1), new1_port.clone());
        net.wire(Port::new(dup1, 2), new2_port.clone());

        net.wire(Port::principal(dup2), orig);
        net.wire(Port::new(dup2, 1), new1_port);
        net.wire(Port::new(dup2, 2), new2_port);
    }

    net.wire(Port::principal(dup_id), dup_port);
    net.free_node(other_id);
}

fn erase_agent(net: &mut Net, agent_id: NodeId) {
    let kind = net.nodes[agent_id as usize].as_ref().unwrap().kind.clone();
    let aux_count = kind.aux_count();
    for slot in 1..=aux_count {
        let port = net.get_aux(agent_id, slot);
        let era = net.alloc_node(AgentKind::Era);
        net.wire(port, Port::principal(era));
    }
}

fn eval_prim(op: PrimOp, a: &Literal, b: &Literal) -> Option<Literal> {
    match (op, a, b) {
        (PrimOp::Add64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Int64(x.wrapping_add(*y))),
        (PrimOp::Sub64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Int64(x.wrapping_sub(*y))),
        (PrimOp::Mul64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Int64(x.wrapping_mul(*y))),
        (PrimOp::Eq64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x == y)),
        (PrimOp::Gt64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x > y)),
        (PrimOp::Lt64, Literal::Int64(x), Literal::Int64(y)) => Some(Literal::Bool(x < y)),
        (PrimOp::And, Literal::Bool(x), Literal::Bool(y)) => Some(Literal::Bool(*x && *y)),
        (PrimOp::Or, Literal::Bool(x), Literal::Bool(y)) => Some(Literal::Bool(*x || *y)),
        (PrimOp::Not, Literal::Bool(x), _) => Some(Literal::Bool(!x)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_identity() {
        // (\x. x) 42 → 42
        let mut net = Net::new();
        let abs = net.alloc_node(AgentKind::Abs);
        let lit = net.alloc_node(AgentKind::Lit(Literal::Int64(42)));
        let app = net.alloc_node(AgentKind::App);

        net.wire(Port::new(abs, 1), Port::new(lit, 0));
        net.wire(Port::new(abs, 2), Port::new(lit, 0));
        net.wire(Port::principal(app), Port::principal(abs));
        net.wire(Port::new(app, 1), Port::principal(lit));

        net.root = Port::new(app, 2);
        net.collect_active_pairs();
        reduce(&mut net);
    }
}
