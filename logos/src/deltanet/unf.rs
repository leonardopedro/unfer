use super::types::*;
use crate::core_ir::Literal;
use sha2::{Sha256, Digest};

pub fn canonical_serialize(net: &Net) -> Vec<u8> {
    let mut output = Vec::new();
    serialize_port(net, &net.root, &mut output);
    output
}

fn serialize_port(net: &Net, port: &Port, out: &mut Vec<u8>) {
    let node = match net.nodes.get(port.node as usize) {
        Some(Some(n)) => n,
        _ => {
            out.push(0xFF);
            return;
        }
    };

    if node.freed {
        out.push(0xFF);
        return;
    }

    match &node.kind {
        AgentKind::Lit(Literal::Int64(n)) => {
            out.push(0x01);
            out.extend_from_slice(&n.to_le_bytes());
        }
        AgentKind::Lit(Literal::Bool(b)) => {
            out.push(0x02);
            out.push(if *b { 1 } else { 0 });
        }
        AgentKind::Con(tag, arity) => {
            out.push(0x03);
            out.extend_from_slice(&tag.to_le_bytes());
            out.push(*arity);
            for s in 1..=*arity {
                serialize_port(net, &net.get_aux(port.node, s), out);
            }
        }
        _ => {
            out.push(0xFF);
        }
    }
}

pub fn unf_hash(net: &Net) -> [u8; 32] {
    let bytes = canonical_serialize(net);
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hasher.finalize().into()
}

pub fn unf_hash_string(net: &Net) -> String {
    let hash = unf_hash(net);
    hex::encode(&hash)
}

// Minimal hex encode (avoid adding dependency)
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_literal() {
        let mut net = Net::new();
        let node = net.alloc_node(AgentKind::Lit(Literal::Int64(42)));
        net.root = Port::principal(node);
        let bytes = canonical_serialize(&net);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(i64::from_le_bytes(bytes[1..9].try_into().unwrap()), 42);
    }

    #[test]
    fn test_unf_hash_deterministic() {
        let mut net = Net::new();
        let node = net.alloc_node(AgentKind::Lit(Literal::Int64(42)));
        net.root = Port::principal(node);
        let h1 = unf_hash(&net);
        let h2 = unf_hash(&net);
        assert_eq!(h1, h2);
    }
}
