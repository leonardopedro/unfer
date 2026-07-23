use super::types::*;
use std::collections::HashSet;

pub fn readback(net: &Net) -> String {
    let mut visited = HashSet::new();
    readback_port(net, &net.root, &mut visited)
}

fn readback_port(net: &Net, port: &Port, visited: &mut HashSet<(NodeId, u8)>) -> String {
    let key = (port.node, port.slot);
    if !visited.insert(key) {
        return "<cycle>".to_string();
    }

    let node = match net.nodes.get(port.node as usize) {
        Some(Some(n)) => n,
        _ => return "<freed>".to_string(),
    };

    if node.freed {
        if let Some(target) = &node.ports[port.slot as usize] {
            return readback_port(net, target, visited);
        }
        return "<freed>".to_string();
    }

    match &node.kind {
        AgentKind::Lit(lit) => lit.to_string(),
        AgentKind::Entity(name) => name.clone(),
        AgentKind::Con(tag, arity) => {
            let tag_name = tag_to_name(*tag);
            if *arity == 0 {
                tag_name.to_string()
            } else {
                let args: Vec<String> = (1..=*arity)
                    .map(|s| readback_port(net, &net.get_aux(port.node, s), visited))
                    .collect();
                format!("{}({})", tag_name, args.join(", "))
            }
        }
        AgentKind::App => {
            let func = readback_port(net, &net.get_aux(port.node, 0), visited);
            let arg = readback_port(net, &net.get_aux(port.node, 1), visited);
            format!("({} {})", func, arg)
        }
        AgentKind::Abs => {
            format!("<abs>")
        }
        AgentKind::Fold => {
            format!("<fold>")
        }
        AgentKind::Dup(_) => {
            format!("<dup>")
        }
        AgentKind::Era => {
            format!("<era>")
        }
        AgentKind::Prim(op) => {
            let left = readback_port(net, &net.get_aux(port.node, 1), visited);
            let right = readback_port(net, &net.get_aux(port.node, 2), visited);
            format!("({:?} {} {})", op, left, right)
        }
    }
}

fn tag_to_name(tag: u32) -> &'static str {
    match tag {
        1 => "Love",
        2 => "See",
        3 => "Like",
        4 => "Eat",
        5 => "Sleep",
        6 => "Run",
        7 => "Assign",
        8 => "Add",
        9 => "Mul",
        10 => "Sub",
        11 => "Eq",
        12 => "Gt",
        13 => "Lt",
        14 => "Not",
        15 => "Restrict",
        16 => "Give",
        17 => "Big",
        18 => "Small",
        19 => "Red",
        20 => "Blue",
        21 => "Very",
        22 => "Cat",
        23 => "Dog",
        24 => "Number",
        25 => "And",
        100 => "Nil",
        101 => "Cons",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::Literal;

    #[test]
    fn test_readback_literal() {
        let mut net = Net::new();
        let node = net.alloc_node(AgentKind::Lit(Literal::Int64(42)));
        net.root = Port::principal(node);
        assert_eq!(readback(&net), "42");
    }
}
