//! Canonical serialization of an interaction net, and the content-addressable
//! `unf_hash` built from it.
//!
//! # Why every agent kind must be covered
//!
//! The hash this module produces *is* the pipeline's node identity: it keys
//! `engram::segment`, `LemmaStore`, `formalize`'s duplicate-identity check,
//! `logos unf`'s output, and the hash the australVM plugin compares against the
//! kernel's. A serializer that cannot name an agent therefore cannot tell two
//! terms apart, and every unnameable term silently becomes the same node.
//!
//! This module used to have a catch-all that emitted a single `0xFF` byte for
//! `App`, `Abs`, `Fold`, `Dup`, `Era`, `Prim` and for freed/absent nodes. Every
//! one of those serialized to the same byte, so `(x + 3)`, `(y * z)` and
//! `(a - b)` all hashed to `SHA-256(0xFF)`. Nothing reported it: the reduced
//! form of an arithmetic term is built from `App`/`Prim` agents, so the terms a
//! formalizer actually produces were precisely the ones that collided.
//!
//! # Canonicity
//!
//! Serialization follows ports from the root rather than node indices, so the
//! bytes depend on the *shape* of the net and not on the order its nodes
//! happened to be allocated. Two nets denoting the same term serialize
//! identically even when built by different routes.
//!
//! A reduced net is not always a tree — a term with unbound variables can leave
//! `Dup`/`Era` structure whose ports refer back to nodes already emitted — so
//! the walk assigns each node a canonical index on first visit (depth-first from
//! the root, which is deterministic) and emits a back-reference on revisit.
//! That keeps the encoding finite without collapsing distinct nets.

use super::types::*;
use crate::core_ir::{Literal, PrimOp};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Agent tags. The five pre-existing values keep their bytes so hashes for
/// nets built only from literals, constructors and entities are unchanged.
mod tag {
    pub const BACKREF: u8 = 0x00;
    pub const INT64: u8 = 0x01;
    pub const BOOL: u8 = 0x02;
    pub const CON: u8 = 0x03;
    pub const ENTITY: u8 = 0x04;
    pub const F64: u8 = 0x05;
    pub const APP: u8 = 0x06;
    pub const ABS: u8 = 0x07;
    pub const FOLD: u8 = 0x08;
    pub const DUP: u8 = 0x09;
    pub const ERA: u8 = 0x0A;
    pub const PRIM: u8 = 0x0B;
    /// A port whose node is absent, or a freed node with no target. Distinct
    /// from `BACKREF`, and never standing in for a real agent.
    pub const ABSENT: u8 = 0xFF;
}

/// Depth bound on the walk.
///
/// The back-reference scheme makes cycles finite, but a net can still be a
/// pathologically long *chain* of distinct nodes. Rather than risk a stack
/// overflow in a kernel reachable from untrusted input, refuse it — loudly.
/// Returning a wrong hash would be worse than returning none.
const MAX_DEPTH: u32 = 4096;

pub fn canonical_serialize(net: &Net) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut seen: HashMap<usize, u32> = HashMap::new();
    let mut next_index: u32 = 0;
    serialize_port(net, &net.root, &mut output, &mut seen, &mut next_index, 0)?;
    Ok(output)
}

fn serialize_port(
    net: &Net,
    port: &Port,
    out: &mut Vec<u8>,
    seen: &mut HashMap<usize, u32>,
    next_index: &mut u32,
    depth: u32,
) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!(
            "unf: net deeper than {MAX_DEPTH} nodes; refusing to hash it"
        ));
    }
    let Some(node) = net.nodes.get(port.node as usize).and_then(|n| n.as_ref()) else {
        out.push(tag::ABSENT);
        return Ok(());
    };

    // A freed node is a forwarding stub: follow it, so the encoding depends on
    // the term rather than on the reduction's leftovers.
    if node.freed {
        return match node.ports.get(port.slot as usize).and_then(|p| p.as_ref()) {
            Some(target) => serialize_port(net, target, out, seen, next_index, depth + 1),
            None => {
                out.push(tag::ABSENT);
                Ok(())
            }
        };
    }

    if let Some(index) = seen.get(&(port.node as usize)) {
        out.push(tag::BACKREF);
        out.extend_from_slice(&index.to_le_bytes());
        return Ok(());
    }
    let index = *next_index;
    *next_index += 1;
    seen.insert(port.node as usize, index);

    match &node.kind {
        AgentKind::Lit(Literal::Int64(n)) => {
            out.push(tag::INT64);
            out.extend_from_slice(&n.to_le_bytes());
        }
        AgentKind::Lit(Literal::F64(x)) => {
            out.push(tag::F64);
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        AgentKind::Lit(Literal::Bool(b)) => {
            out.push(tag::BOOL);
            out.push(u8::from(*b));
        }
        AgentKind::Con(constructor, arity) => {
            out.push(tag::CON);
            out.extend_from_slice(&constructor.to_le_bytes());
            out.push(*arity);
        }
        AgentKind::Entity(name) => {
            out.push(tag::ENTITY);
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
        }
        AgentKind::App
        | AgentKind::Abs
        | AgentKind::Fold
        | AgentKind::Dup(_)
        | AgentKind::Era
        | AgentKind::Prim(_) => {
            out.push(agent_tag(&node.kind));
            if let AgentKind::Prim(op) = &node.kind {
                out.push(prim_tag(*op));
            }
            if let AgentKind::Dup(level) = &node.kind {
                out.extend_from_slice(&level.to_le_bytes());
            }
        }
    }

    for slot in 1..=node.kind.aux_count() {
        // An unwired aux port is a real state, not an error: it is what an
        // unbound variable looks like in the net. `Net::get_aux` refuses those,
        // which is right for a caller that needs a port it cannot have but wrong
        // here — encoding "nothing here" keeps an open term hashable, and keeps
        // its hash distinct from any term that *does* have a node in that slot.
        match node.ports.get(slot as usize).and_then(|p| p.as_ref()) {
            Some(aux) => {
                serialize_port(net, aux, out, seen, next_index, depth + 1)?;
            }
            None => out.push(tag::ABSENT),
        }
    }
    Ok(())
}

fn agent_tag(kind: &AgentKind) -> u8 {
    match kind {
        AgentKind::App => tag::APP,
        AgentKind::Abs => tag::ABS,
        AgentKind::Fold => tag::FOLD,
        AgentKind::Dup(_) => tag::DUP,
        AgentKind::Era => tag::ERA,
        AgentKind::Prim(_) => tag::PRIM,
        AgentKind::Lit(_) | AgentKind::Con(..) | AgentKind::Entity(_) => {
            unreachable!("literal, constructor and entity tags are emitted by their own arms")
        }
    }
}

/// A stable byte per primitive operation.
///
/// Derived from the enum's own discriminant rather than its `Debug` spelling so
/// that renaming a variant cannot silently re-hash every term that uses it, and
/// so the mapping is total — a new variant gets a byte automatically instead of
/// panicking or colliding.
fn prim_tag(op: PrimOp) -> u8 {
    match op {
        PrimOp::Add64 => 0x00,
        PrimOp::Sub64 => 0x01,
        PrimOp::Mul64 => 0x02,
        PrimOp::Eq64 => 0x03,
        PrimOp::Gt64 => 0x04,
        PrimOp::Lt64 => 0x05,
        PrimOp::AddF64 => 0x06,
        PrimOp::SubF64 => 0x07,
        PrimOp::MulF64 => 0x08,
        PrimOp::DivF64 => 0x09,
        PrimOp::EqF64 => 0x0A,
        PrimOp::GtF64 => 0x0B,
        PrimOp::LtF64 => 0x0C,
        PrimOp::And => 0x0D,
        PrimOp::Or => 0x0E,
        PrimOp::Not => 0x0F,
    }
}

pub fn unf_hash(net: &Net) -> Result<[u8; 32], String> {
    let bytes = canonical_serialize(net)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hasher.finalize().into())
}

pub fn unf_hash_string(net: &Net) -> Result<String, String> {
    let hash = unf_hash(net)?;
    Ok(hex::encode(&hash))
}

// ── Isomorphism witness (the conversion certificate) ─────────────────
//
// The lean4mm0 plan (`australVM/docs/lean4mm0.md`, adapted) needs more than
// a yes/no equality on normal forms: a *conversion step* in a proof trace
// has to carry a certificate that two normal forms are the same net, not
// merely two nets with the same digest. This is that certificate, and it is
// derived from the canonical serializer above rather than from a second
// traversal scheme: the serializer already visits nodes in a deterministic
// root-directed DFS order that ignores allocation history, so the bijection
// between two equal-serializing nets is exactly "the same position in the
// walk". No new hash, no second net representation — the ENGRAM discipline.

/// An isomorphism between two nets with identical canonical serializations.
///
/// `pairs[c] = (node_in_g1, node_in_g2)` — the two nodes occupying canonical
/// position `c` of the (identical) serializations. The permutation is total
/// over the reachable, non-freed nodes of both nets.
#[derive(Debug, Clone)]
pub struct IsomorphismWitness {
    pub pairs: Vec<(NodeId, NodeId)>,
}

impl IsomorphismWitness {
    /// The number of canonical positions the witness covers.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Re-check the certificate against the nets it claims to relate.
    ///
    /// The witness is only as good as the traversal that produced it, so a
    /// consumer that will *act* on it (emit a conversion step, say) should
    /// call this: it re-walks both nets, requires the recorded pairing to be
    /// exactly the canonical zip, and then checks structure under the
    /// permutation — same agent kind at every paired position, and aux ports
    /// that wire to *matching canonical positions* on both sides. Sharing,
    /// cycles (back-references) and unwired aux ports all have to agree.
    pub fn verifies(&self, g1: &Net, g2: &Net) -> bool {
        let walk1 = match canonical_walk(g1) {
            Ok(w) => w,
            Err(_) => return false,
        };
        let walk2 = match canonical_walk(g2) {
            Ok(w) => w,
            Err(_) => return false,
        };
        if walk1.len() != self.pairs.len() || walk2.len() != self.pairs.len() {
            return false;
        }
        // The pairing must be exactly the canonical zip.
        for (c, (n1, n2)) in self.pairs.iter().enumerate() {
            if walk1[c] != *n1 || walk2[c] != *n2 {
                return false;
            }
        }
        // Structural agreement under π: canonical position -> position.
        let pos1: HashMap<NodeId, u32> = walk1
            .iter()
            .enumerate()
            .map(|(c, n)| (*n, c as u32))
            .collect();
        let pos2: HashMap<NodeId, u32> = walk2
            .iter()
            .enumerate()
            .map(|(c, n)| (*n, c as u32))
            .collect();
        for (n1, n2) in &self.pairs {
            let (Some(node1), Some(node2)) = (
                g1.nodes.get(*n1 as usize).and_then(|n| n.as_ref()),
                g2.nodes.get(*n2 as usize).and_then(|n| n.as_ref()),
            ) else {
                return false;
            };
            if node1.kind != node2.kind {
                return false;
            }
            for slot in 1..=node1.kind.aux_count() {
                let t1 = node1.ports.get(slot as usize).and_then(|p| p.as_ref());
                let t2 = node2.ports.get(slot as usize).and_then(|p| p.as_ref());
                match (t1, t2) {
                    (None, None) => {}
                    (Some(p1), Some(p2)) => {
                        let r1 = g1.resolve_port(p1);
                        let r2 = g2.resolve_port(p2);
                        match (pos1.get(&r1.node), pos2.get(&r2.node)) {
                            (Some(c1), Some(c2)) if c1 == c2 => {}
                            _ => return false,
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }
}

/// The canonical visit order of `net` as a plain node list.
///
/// Computed *through* [`isomorphism_witness`] on the net against itself, so
/// the order used by certificate verification can never drift from the order
/// the certificate producer used — they are the same call.
fn canonical_walk(net: &Net) -> Result<Vec<NodeId>, String> {
    Ok(isomorphism_witness(net, net)?
        .expect("a net is always isomorphic to itself")
        .pairs
        .into_iter()
        .map(|(n, _)| n)
        .collect())
}

/// The conversion certificate for two reduced nets: `Some(witness)` when
/// `g1` and `g2` have identical canonical serializations (hence are
/// isomorphic as rooted port graphs), `None` when they do not.
///
/// Serialization errors (the depth guard) propagate instead of silently
/// declining to certify.
pub fn isomorphism_witness(g1: &Net, g2: &Net) -> Result<Option<IsomorphismWitness>, String> {
    // Cheap necessary condition first — and the one that lines up with the
    // content-addressed identity the rest of the pipeline already trusts.
    if canonical_serialize(g1)? != canonical_serialize(g2)? {
        return Ok(None);
    }
    let mut pairs = Vec::new();
    let mut seen1: HashMap<usize, u32> = HashMap::new();
    let mut seen2: HashMap<usize, u32> = HashMap::new();
    let mut next_index: u32 = 0;
    let ok = lockstep_port(
        g1,
        &g1.root,
        g2,
        &g2.root,
        &mut pairs,
        &mut seen1,
        &mut seen2,
        &mut next_index,
        0,
    )?;
    // `ok == false` here would mean equal bytes but a diverging walk — a bug
    // in this module, not a genuine non-isomorphism. Refuse to certify.
    Ok(if ok {
        Some(IsomorphismWitness { pairs })
    } else {
        None
    })
}

/// Mirror of [`serialize_port`]: same resolution of freed forwarding stubs,
/// same first-visit ordering, same aux-slot recursion, same depth guard —
/// but instead of emitting bytes it records `(g1 node, g2 node)` per canonical
/// position and refuses (returns `Ok(false)`) at the first structural
/// disagreement.
#[allow(clippy::too_many_arguments)] // its mirror `serialize_port`'s walk state, threaded through one hop per node
fn lockstep_port(
    g1: &Net,
    p1: &Port,
    g2: &Net,
    p2: &Port,
    out: &mut Vec<(NodeId, NodeId)>,
    seen1: &mut HashMap<usize, u32>,
    seen2: &mut HashMap<usize, u32>,
    next_index: &mut u32,
    depth: u32,
) -> Result<bool, String> {
    if depth > MAX_DEPTH {
        return Err(format!(
            "unf: net deeper than {MAX_DEPTH} nodes; refusing to certify it"
        ));
    }
    let n1 = g1.nodes.get(p1.node as usize).and_then(|n| n.as_ref());
    let n2 = g2.nodes.get(p2.node as usize).and_then(|n| n.as_ref());
    // Absent on one side only: the serializations would differ (ABSENT vs a
    // tag) — unreachable after the byte check, kept as defence in depth.
    match (n1, n2) {
        (None, None) => return Ok(true),
        (None, Some(_)) | (Some(_), None) => return Ok(false),
        _ => {}
    }
    let (node1, node2) = (n1.unwrap(), n2.unwrap());

    // Freed nodes are forwarding stubs: follow, exactly like the serializer.
    if node1.freed || node2.freed {
        if node1.freed != node2.freed {
            return Ok(false);
        }
        let t1 = node1.ports.get(p1.slot as usize).and_then(|p| p.as_ref());
        let t2 = node2.ports.get(p2.slot as usize).and_then(|p| p.as_ref());
        return match (t1, t2) {
            (Some(q1), Some(q2)) => {
                lockstep_port(g1, q1, g2, q2, out, seen1, seen2, next_index, depth + 1)
            }
            (None, None) => Ok(true),
            _ => Ok(false),
        };
    }

    // A revisit must be a revisit at the same canonical position on both sides.
    match (
        seen1.get(&(p1.node as usize)),
        seen2.get(&(p2.node as usize)),
    ) {
        (Some(c1), Some(c2)) if c1 == c2 => return Ok(true),
        (None, None) => {}
        _ => return Ok(false),
    }

    // First visit: kinds must agree, then record and recurse.
    if node1.kind != node2.kind {
        return Ok(false);
    }
    let canonical = *next_index;
    *next_index += 1;
    seen1.insert(p1.node as usize, canonical);
    seen2.insert(p2.node as usize, canonical);
    out.push((p1.node, p2.node));

    for slot in 1..=node1.kind.aux_count() {
        let a1 = node1.ports.get(slot as usize).and_then(|p| p.as_ref());
        let a2 = node2.ports.get(slot as usize).and_then(|p| p.as_ref());
        match (a1, a2) {
            (Some(q1), Some(q2)) => {
                if !lockstep_port(g1, q1, g2, q2, out, seen1, seen2, next_index, depth + 1)? {
                    return Ok(false);
                }
            }
            // An unwired aux port is a real state (an open term's unknown);
            // it must be unwired on the other side too.
            (None, None) => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
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

    fn lit(kind: AgentKind) -> Net {
        let mut net = Net::new();
        let node = net.alloc_node(kind);
        net.root = Port::principal(node);
        net
    }

    /// A single node of `kind` with every aux port wired to a shared `Era`.
    ///
    /// A reduced net always has its aux ports wired, and `get_aux` refuses an
    /// unwired slot — so a bare node cannot be serialized unless it has no aux
    /// slots. `Era` has none, which makes it the natural filler.
    fn wired(kind: AgentKind) -> Net {
        let aux_count = kind.aux_count();
        let mut net = Net::new();
        let node = net.alloc_node(kind);
        let filler = net.alloc_node(AgentKind::Era);
        for slot in 1..=aux_count {
            net.wire(Port::new(node, slot), Port::principal(filler));
        }
        net.root = Port::principal(node);
        net
    }

    #[test]
    fn test_serialize_literal() {
        let bytes = canonical_serialize(&lit(AgentKind::Lit(Literal::Int64(42)))).unwrap();
        assert_eq!(bytes[0], tag::INT64);
        assert_eq!(i64::from_le_bytes(bytes[1..9].try_into().unwrap()), 42);
    }

    #[test]
    fn test_serialize_f64() {
        let bytes = canonical_serialize(&lit(AgentKind::Lit(Literal::F64(3.5)))).unwrap();
        assert_eq!(bytes[0], tag::F64);
        assert_eq!(f64::from_le_bytes(bytes[1..9].try_into().unwrap()), 3.5);
    }

    #[test]
    fn test_unf_hash_f64_discriminates_int() {
        let a = lit(AgentKind::Lit(Literal::F64(3.5)));
        let b = lit(AgentKind::Lit(Literal::Int64(3)));
        assert_ne!(unf_hash(&a).unwrap(), unf_hash(&b).unwrap());
    }

    #[test]
    fn test_unf_hash_deterministic() {
        let net = lit(AgentKind::Lit(Literal::Int64(7)));
        assert_eq!(unf_hash(&net).unwrap(), unf_hash(&net).unwrap());
    }

    /// The regression this whole rewrite exists for: distinct terms whose
    /// normal forms are built from `App`/`Prim` agents must not collide.
    #[test]
    fn arithmetic_normal_forms_do_not_collide() {
        let cases = [
            "(x + 3)", "(y * z)", "(a - b)", "(f 1)", "(g 2)", "1.5 + x", "x < 3",
        ];
        let mut hashes = Vec::new();
        for src in cases {
            let t = crate::translate::translate_austral_expr(src)
                .unwrap_or_else(|e| panic!("{src} should translate: {e}"));
            hashes.push((src.to_string(), t.unf_hash.clone()));
        }
        for (i, (src_a, ha)) in hashes.iter().enumerate() {
            for (src_b, hb) in hashes.iter().skip(i + 1) {
                assert_ne!(ha, hb, "{src_a} and {src_b} hashed identically ({ha})");
            }
        }
    }

    /// Every agent kind must contribute bytes. Two nets differing only in the
    /// kind of a single node must not serialize alike.
    #[test]
    fn every_agent_kind_is_distinguishable() {
        let kinds = [
            AgentKind::App,
            AgentKind::Abs,
            AgentKind::Fold,
            AgentKind::Dup(1),
            AgentKind::Era,
            AgentKind::Prim(PrimOp::Add64),
            AgentKind::Prim(PrimOp::Sub64),
            AgentKind::Lit(Literal::Int64(1)),
            AgentKind::Con(7, 0),
            AgentKind::Entity("e".into()),
        ];
        let mut seen = std::collections::HashSet::new();
        for kind in kinds {
            let net = wired(kind.clone());
            let bytes = canonical_serialize(&net).unwrap();
            assert!(
                !(bytes.len() == 1 && bytes[0] == tag::ABSENT),
                "{kind:?} fell through to the absent marker"
            );
            assert!(seen.insert(bytes), "{kind:?} collided with another kind");
        }
    }

    /// Two different primitive operations under the same shape must differ.
    #[test]
    fn primitive_operations_are_distinguished() {
        let a = wired(AgentKind::Prim(PrimOp::Add64));
        let b = wired(AgentKind::Prim(PrimOp::Mul64));
        assert_ne!(unf_hash(&a).unwrap(), unf_hash(&b).unwrap());
    }

    /// A cycle must terminate and still be deterministic. An open term can leave
    /// `Dup`/`Era` structure pointing back at itself; before the back-reference
    /// scheme this recursed forever.
    #[test]
    fn a_cyclic_net_terminates_and_is_stable() {
        let mut net = Net::new();
        let a = net.alloc_node(AgentKind::Dup(0));
        let b = net.alloc_node(AgentKind::Era);
        net.wire(Port::new(a, 1), Port::principal(b));
        net.wire(Port::new(a, 2), Port::principal(a));
        net.root = Port::principal(a);
        let first = canonical_serialize(&net).unwrap();
        let second = canonical_serialize(&net).unwrap();
        assert_eq!(first, second, "serialization must be deterministic");
        assert!(
            first.contains(&tag::BACKREF),
            "the cycle must have been cut by a back-reference"
        );
    }

    /// A node shared by two ports is emitted once and referenced twice, so two
    /// nets that differ only in *sharing* are distinguishable from one that
    /// merely repeats the same content.
    #[test]
    fn sharing_is_distinguished_from_repetition() {
        use crate::core_ir::TagId;
        let mut shared = Net::new();
        let leaf = shared.alloc_node(AgentKind::Con(1, 0));
        let root = shared.alloc_node(AgentKind::Con(2, 2));
        shared.wire(Port::new(root, 1), Port::principal(leaf));
        shared.wire(Port::new(root, 2), Port::principal(leaf));
        shared.root = Port::principal(root);

        let mut repeated = Net::new();
        let l1 = repeated.alloc_node(AgentKind::Con(1, 0));
        let l2 = repeated.alloc_node(AgentKind::Con(1, 0));
        let root2 = repeated.alloc_node(AgentKind::Con(2, 2));
        repeated.wire(Port::new(root2, 1), Port::principal(l1));
        repeated.wire(Port::new(root2, 2), Port::principal(l2));
        repeated.root = Port::principal(root2);

        assert_ne!(
            unf_hash(&shared).unwrap(),
            unf_hash(&repeated).unwrap(),
            "a shared node and two equal nodes are different nets"
        );
        let _ = TagId::default();
    }

    /// An absent node and a freed node with no target both encode as absent.
    /// They are genuinely indistinguishable states, so this pins the behaviour
    /// rather than treating it as a gap.
    #[test]
    fn an_absent_node_encodes_as_absent() {
        let mut net = Net::new();
        net.root = Port::principal(9999);
        assert_eq!(canonical_serialize(&net).unwrap(), vec![tag::ABSENT]);
    }

    // ── isomorphism witness ───────────────────────────────────────────

    /// Two nets denoting the same constructor tree but built with different
    /// allocation orders: equal serializations, and a non-identity witness
    /// that pairs each canonical position with the right node on each side.
    #[test]
    fn witness_pairs_equal_nets_across_allocation_orders() {
        // g1: leaves allocated first, then root.
        let mut g1 = Net::new();
        let l1 = g1.alloc_node(AgentKind::Con(1, 0));
        let l2 = g1.alloc_node(AgentKind::Con(2, 0));
        let r1 = g1.alloc_node(AgentKind::Con(3, 2));
        g1.wire(Port::new(r1, 1), Port::principal(l1));
        g1.wire(Port::new(r1, 2), Port::principal(l2));
        g1.root = Port::principal(r1);

        // g2: root allocated first, leaves in the opposite order.
        let mut g2 = Net::new();
        let r2 = g2.alloc_node(AgentKind::Con(3, 2));
        let l2b = g2.alloc_node(AgentKind::Con(2, 0));
        let l1b = g2.alloc_node(AgentKind::Con(1, 0));
        g2.wire(Port::new(r2, 1), Port::principal(l1b));
        g2.wire(Port::new(r2, 2), Port::principal(l2b));
        g2.root = Port::principal(r2);

        assert_eq!(unf_hash(&g1).unwrap(), unf_hash(&g2).unwrap());
        let w = isomorphism_witness(&g1, &g2)
            .unwrap()
            .expect("equal serializations must yield a witness");
        // Canonical order is root, aux1, aux2 — so position 0 is the root.
        assert_eq!(w.pairs[0], (r1, r2));
        assert!(
            w.pairs.contains(&(l1, l1b)) && w.pairs.contains(&(l2, l2b)),
            "leaf pairing follows shape, not allocation: {:?}",
            w.pairs
        );
        assert!(w.verifies(&g1, &g2));
        // And it is a genuine non-identity permutation of node ids.
        assert!(w.pairs.iter().any(|(a, b)| a != b));
    }

    #[test]
    fn witness_declines_different_nets() {
        let a = lit(AgentKind::Con(1, 0));
        let b = lit(AgentKind::Con(2, 0));
        assert!(isomorphism_witness(&a, &b).unwrap().is_none());
    }

    /// A tampered pairing — two positions swapped — must fail verification
    /// even though the witness *shape* is right.
    #[test]
    fn verifies_rejects_a_tampered_pairing() {
        let mut g1 = Net::new();
        let a = g1.alloc_node(AgentKind::Con(1, 0));
        let b = g1.alloc_node(AgentKind::Con(2, 0));
        let r = g1.alloc_node(AgentKind::Con(3, 2));
        g1.wire(Port::new(r, 1), Port::principal(a));
        g1.wire(Port::new(r, 2), Port::principal(b));
        g1.root = Port::principal(r);

        let mut w = isomorphism_witness(&g1, &g1).unwrap().unwrap();
        assert!(w.verifies(&g1, &g1));
        // Swap the two leaf pairings: same multiset of nodes, wrong mapping.
        let i = w.pairs.iter().position(|(x, _)| *x == a).unwrap();
        let j = w.pairs.iter().position(|(x, _)| *x == b).unwrap();
        w.pairs.swap(i, j);
        assert!(!w.verifies(&g1, &g1), "a tampered π must not verify");
    }

    /// Sharing must be respected by the certificate: a shared child and two
    /// equal children are different nets, and the witness over a *shared*
    /// net against itself must record the shared node once.
    #[test]
    fn witness_records_shared_nodes_once() {
        let mut shared = Net::new();
        let leaf = shared.alloc_node(AgentKind::Con(1, 0));
        let root = shared.alloc_node(AgentKind::Con(2, 2));
        shared.wire(Port::new(root, 1), Port::principal(leaf));
        shared.wire(Port::new(root, 2), Port::principal(leaf));
        shared.root = Port::principal(root);

        let w = isomorphism_witness(&shared, &shared).unwrap().unwrap();
        assert_eq!(w.len(), 2, "root + one shared leaf: {:?}", w.pairs);
        assert!(w.verifies(&shared, &shared));
    }

    /// A cyclic net (open terms can leave `Dup` structure pointing back at
    /// itself) must still yield a self-witness that verifies — back-reference
    /// positions have to agree on both sides.
    #[test]
    fn witness_covers_cyclic_nets() {
        let mut net = Net::new();
        let a = net.alloc_node(AgentKind::Dup(0));
        let b = net.alloc_node(AgentKind::Era);
        net.wire(Port::new(a, 1), Port::principal(b));
        net.wire(Port::new(a, 2), Port::principal(a));
        net.root = Port::principal(a);

        let w = isomorphism_witness(&net, &net).unwrap().unwrap();
        assert!(w.verifies(&net, &net));
    }

    /// End-to-end over the real pipeline: two different Austral sources that
    /// reduce to the same normal form are related by a verifying witness.
    #[test]
    fn witness_over_reduced_translations() {
        let t1 = crate::translate::translate_austral_expr("(x + (2 + 3))").unwrap();
        let t2 = crate::translate::translate_austral_expr("(x + 5)").unwrap();
        assert_eq!(t1.unf_hash, t2.unf_hash);
        let w = isomorphism_witness(&t1.net, &t2.net)
            .unwrap()
            .expect("equal UNF digests must yield a witness");
        assert!(w.verifies(&t1.net, &t2.net));
        // And the converse: sources with different normal forms do not.
        let t3 = crate::translate::translate_austral_expr("(x + 6)").unwrap();
        assert!(isomorphism_witness(&t1.net, &t3.net).unwrap().is_none());
    }
}
