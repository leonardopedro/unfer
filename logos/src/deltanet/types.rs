use crate::core_ir::{TagId, Literal, PrimOp};
use std::fmt;

pub type NodeId = u32;

#[derive(Debug, Clone)]
pub struct Port {
    pub node: NodeId,
    pub slot: u8,
}

impl Port {
    pub fn new(node: NodeId, slot: u8) -> Self {
        Self { node, slot }
    }

    pub fn principal(node: NodeId) -> Self {
        Self { node, slot: 0 }
    }
}

#[derive(Debug, Clone)]
pub enum AgentKind {
    App,
    Abs,
    Con(TagId, u8),
    Fold,
    Dup(u16),
    Era,
    Prim(PrimOp),
    Lit(Literal),
}

impl AgentKind {
    pub fn aux_count(&self) -> u8 {
        match self {
            AgentKind::App => 2,
            AgentKind::Abs => 2,
            AgentKind::Con(_, arity) => *arity,
            AgentKind::Fold => 3,
            AgentKind::Dup(_) => 2,
            AgentKind::Era => 0,
            AgentKind::Prim(_) => 2,
            AgentKind::Lit(_) => 0,
        }
    }
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentKind::App => write!(f, "App"),
            AgentKind::Abs => write!(f, "Abs"),
            AgentKind::Con(tag, _) => write!(f, "Con({})", tag),
            AgentKind::Fold => write!(f, "Fold"),
            AgentKind::Dup(level) => write!(f, "Dup({})", level),
            AgentKind::Era => write!(f, "Era"),
            AgentKind::Prim(op) => write!(f, "{:?}", op),
            AgentKind::Lit(lit) => write!(f, "{}", lit),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: AgentKind,
    pub ports: Vec<Option<Port>>,
    pub freed: bool,
}

impl Node {
    pub fn new(kind: AgentKind) -> Self {
        let aux_count = kind.aux_count() as usize;
        let mut ports = Vec::with_capacity(aux_count + 1);
        ports.push(None); // principal port
        for _ in 0..aux_count {
            ports.push(None);
        }
        Self { kind, ports, freed: false }
    }
}

#[derive(Debug, Clone)]
pub struct Net {
    pub nodes: Vec<Option<Node>>,
    pub free_list: Vec<NodeId>,
    pub active_pairs: Vec<(NodeId, NodeId)>,
    pub root: Port,
    pub var_bindings: std::collections::HashMap<String, Port>,
}

impl Net {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            active_pairs: Vec::new(),
            root: Port::new(0, 0),
            var_bindings: std::collections::HashMap::new(),
        }
    }

    pub fn alloc_node(&mut self, kind: AgentKind) -> NodeId {
        if let Some(id) = self.free_list.pop() {
            self.nodes[id as usize] = Some(Node::new(kind));
            id
        } else {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Some(Node::new(kind)));
            id
        }
    }

    pub fn free_node(&mut self, id: NodeId) {
        if let Some(node) = self.nodes[id as usize].as_mut() {
            node.freed = true;
        }
        self.free_list.push(id);
    }

    pub fn is_freed(&self, id: NodeId) -> bool {
        self.nodes[id as usize]
            .as_ref()
            .map(|n| n.freed)
            .unwrap_or(true)
    }

    pub fn wire(&mut self, a: Port, b: Port) {
        if let Some(node) = self.nodes[a.node as usize].as_mut() {
            node.ports[a.slot as usize] = Some(b.clone());
        }
        if let Some(node) = self.nodes[b.node as usize].as_mut() {
            node.ports[b.slot as usize] = Some(a);
        }
    }

    pub fn get_aux(&self, node: NodeId, slot: u8) -> Port {
        self.nodes[node as usize]
            .as_ref()
            .and_then(|n| n.ports[slot as usize].clone())
            .unwrap_or_else(|| panic!("no port at node {} slot {}", node, slot))
    }

    pub fn get_principal(&self, node: NodeId) -> Port {
        self.get_aux(node, 0)
    }

    pub fn get_connected_lit(&self, node: NodeId, slot: u8) -> Option<Literal> {
        let port = self.get_aux(node, slot);
        let other = &self.nodes[port.node as usize];
        if let Some(other_node) = other {
            if let AgentKind::Lit(lit) = &other_node.kind {
                return Some(lit.clone());
            }
        }
        None
    }

    pub fn collect_active_pairs(&mut self) {
        for i in 0..self.nodes.len() {
            if self.is_freed(i as NodeId) {
                continue;
            }
            if let Some(node) = &self.nodes[i as usize] {
                if let Some(principal) = &node.ports[0] {
                    if let Some(other_node) = &self.nodes[principal.node as usize] {
                        if !other_node.freed && other_node.ports[0].is_some() {
                            let other_principal = other_node.ports[0].as_ref().unwrap();
                            if other_principal.node == i as NodeId {
                                let a = i as NodeId;
                                let b = principal.node;
                                if a <= b {
                                    self.active_pairs.push((a, b));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn collect_new_active_pairs(&mut self) {
        self.collect_active_pairs();
    }

    pub fn bind_var(&mut self, name: &str, port: Port) {
        self.var_bindings.insert(name.to_string(), port);
    }

    pub fn lookup_var_port(&self, name: &str) -> Port {
        self.var_bindings
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("unbound variable: {}", name))
    }
}
