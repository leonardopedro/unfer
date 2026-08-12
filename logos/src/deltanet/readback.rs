use super::types::*;
use std::collections::HashSet;

pub fn readback(net: &Net) -> Result<String, String> {
    let mut visited = HashSet::new();
    readback_port(net, &net.root, &mut visited)
}

fn readback_port(
    net: &Net,
    port: &Port,
    visited: &mut HashSet<(NodeId, u8)>,
) -> Result<String, String> {
    let key = (port.node, port.slot);
    if !visited.insert(key) {
        return Ok("<cycle>".to_string());
    }

    let node = match net.nodes.get(port.node as usize) {
        Some(Some(n)) => n,
        _ => return Ok("<freed>".to_string()),
    };

    if node.freed {
        if let Some(target) = &node.ports[port.slot as usize] {
            return readback_port(net, target, visited);
        }
        return Ok("<freed>".to_string());
    }

    match &node.kind {
        AgentKind::Lit(lit) => Ok(lit.to_string()),
        AgentKind::Entity(name) => Ok(name.clone()),
        AgentKind::Con(tag, arity) => {
            let tag_name = tag_to_name(*tag);
            if *arity == 0 {
                Ok(tag_name.to_string())
            } else {
                let mut args = Vec::with_capacity(*arity as usize);
                for s in 1..=*arity {
                    args.push(readback_port(net, &net.get_aux(port.node, s)?, visited)?);
                }
                Ok(format!("{}({})", tag_name, args.join(", ")))
            }
        }
        AgentKind::App => {
            let func = readback_port(net, &net.get_aux(port.node, 0)?, visited)?;
            let arg = readback_port(net, &net.get_aux(port.node, 1)?, visited)?;
            Ok(format!("({} {})", func, arg))
        }
        AgentKind::Abs => Ok("<abs>".to_string()),
        AgentKind::Fold => Ok("<fold>".to_string()),
        AgentKind::Dup(_) => Ok("<dup>".to_string()),
        AgentKind::Era => Ok("<era>".to_string()),
        AgentKind::Prim(op) => {
            let left = readback_port(net, &net.get_aux(port.node, 1)?, visited)?;
            let right = readback_port(net, &net.get_aux(port.node, 2)?, visited)?;
            Ok(format!("({:?} {} {})", op, left, right))
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
        assert_eq!(readback(&net), Ok("42".to_string()));
    }
}
