use super::types::*;
use crate::core_ir::{CoreIR, Pattern};

pub fn compile_to_net(term: &CoreIR) -> Net {
    let mut net = Net::new();
    let root_port = emit(term, &mut net);
    net.root = root_port;
    net
}

fn emit(term: &CoreIR, net: &mut Net) -> Port {
    match term {
        CoreIR::Var(id) => {
            if let Some(port) = net.var_bindings.get(id) {
                port.clone()
            } else {
                let node = net.alloc_node(AgentKind::Entity(id.clone()));
                Port::principal(node)
            }
        }
        CoreIR::Lit(lit) => {
            let node = net.alloc_node(AgentKind::Lit(lit.clone()));
            Port::principal(node)
        }
        CoreIR::Con(tag, args) => {
            let arity = args.len() as u8;
            let node = net.alloc_node(AgentKind::Con(*tag, arity));
            for (i, arg) in args.iter().enumerate() {
                let arg_port = emit(arg, net);
                net.wire(Port::new(node, (i + 1) as u8), arg_port);
            }
            Port::principal(node)
        }
        CoreIR::Lam(id, body) => {
            let node = net.alloc_node(AgentKind::Abs);
            let var_port = Port::new(node, 1);
            net.bind_var(id, var_port.clone());
            let body_port = emit(body, net);
            net.wire(Port::new(node, 2), body_port);
            Port::principal(node)
        }
        CoreIR::App(f, arg) => {
            let node = net.alloc_node(AgentKind::App);
            let f_port = emit(f, net);
            let arg_port = emit(arg, net);
            net.wire(Port::principal(node), f_port);
            net.wire(Port::new(node, 1), arg_port);
            Port::new(node, 2)
        }
        CoreIR::Fold(f, init, list) => {
            let node = net.alloc_node(AgentKind::Fold);
            let f_port = emit(f, net);
            let init_port = emit(init, net);
            let list_port = emit(list, net);
            net.wire(Port::new(node, 1), f_port);
            net.wire(Port::new(node, 2), init_port);
            net.wire(Port::new(node, 3), list_port);
            Port::principal(node)
        }
        CoreIR::Prim(op, args) => {
            assert_eq!(args.len(), 2);
            let node = net.alloc_node(AgentKind::Prim(*op));
            let l = emit(&args[0], net);
            let r = emit(&args[1], net);
            net.wire(Port::new(node, 1), l);
            net.wire(Port::new(node, 2), r);
            Port::principal(node)
        }
        CoreIR::Clone(id, id1, id2, body) => {
            let node = net.alloc_node(AgentKind::Dup(0));
            let orig_port = net.lookup_var_port(id);
            net.wire(Port::principal(node), orig_port);
            let p1 = Port::new(node, 1);
            let p2 = Port::new(node, 2);
            net.bind_var(id1, p1);
            net.bind_var(id2, p2);
            emit(body, net)
        }
        CoreIR::Drop(id, body) => {
            let node = net.alloc_node(AgentKind::Era);
            let orig_port = net.lookup_var_port(id);
            net.wire(Port::principal(node), orig_port);
            emit(body, net)
        }
        CoreIR::Let(id, value, body) => {
            let val_port = emit(value, net);
            net.bind_var(id, val_port);
            emit(body, net)
        }
        CoreIR::Match(scrutinee, arms) => {
            let scrutinee_port = emit(scrutinee, net);
            let match_node = net.alloc_node(AgentKind::Con(0, arms.len() as u8));
            net.wire(Port::principal(match_node), scrutinee_port);
            for (i, (pat, body)) in arms.iter().enumerate() {
                match pat {
                    Pattern::Tag(tag, binders) => {
                        let con_node = net.alloc_node(AgentKind::Con(*tag, binders.len() as u8));
                        net.wire(Port::new(match_node, (i + 1) as u8), Port::principal(con_node));
                        for (j, binder) in binders.iter().enumerate() {
                            net.bind_var(binder, Port::new(con_node, (j + 1) as u8));
                        }
                        let body_port = emit(body, net);
                        net.wire(Port::new(con_node, 0), body_port);
                    }
                }
            }
            Port::principal(match_node)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::Literal;

    #[test]
    fn test_emit_literal() {
        let ir = CoreIR::Lit(Literal::Int64(42));
        let net = compile_to_net(&ir);
        assert_eq!(net.nodes.len(), 1);
        assert!(matches!(net.nodes[0].as_ref().unwrap().kind, AgentKind::Lit(Literal::Int64(42))));
    }

    #[test]
    fn test_emit_app() {
        let ir = CoreIR::App(
            Box::new(CoreIR::Var("f".to_string())),
            Box::new(CoreIR::Lit(Literal::Int64(1))),
        );
        let mut net = Net::new();
        let f_node = net.alloc_node(AgentKind::Lit(Literal::Int64(99)));
        net.bind_var("f", Port::principal(f_node));
        let root = emit(&ir, &mut net);
        assert!(matches!(net.nodes[root.node as usize].as_ref().unwrap().kind, AgentKind::App));
    }
}
