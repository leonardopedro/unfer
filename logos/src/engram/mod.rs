//! PLAN_liquid_types.md E2: engram key derivation.
//!
//! The spec is `australVM/docs/ENGRAM.md` (§1–§5); the golden corpus is
//! `australVM/corpus/engram_keys.tsv`.
//!
//! ## This module owns nothing
//!
//! Every stage below already exists in this crate and is *called*, not
//! reimplemented:
//!
//! | stage | reused from |
//! |---|---|
//! | gate the fragment against the CNL grammar | `harper_gate::Gate::lint` |
//! | tokenize + parse | `ccg::parse_sentence` |
//! | derivation tree -> CoreIR | `core_ir::compile_to_core_ir` |
//! | insert linearity | `core_ir::linearity::insert_linearity` |
//! | CoreIR -> net -> reduce -> readback -> hash | `translate::translate_coreir` |
//! | algebraic canonicalization | `deltanet::ted` (via the same call) |
//!
//! In particular the key's `unf_hash` is the **existing**
//! `deltanet::unf::unf_hash` — SHA-256 over the canonical serialization — and
//! `translate_coreir` also gives us its double-reduction `verified` check for
//! free, which is the runtime corroboration of the confluence theorem ENGRAM.md
//! §2.4 item 1 relies on. This crate already depends on `sha2`; E2 adds no
//! dependency and no second representation of a net.
//!
//! `translate_coreir` returns `unf_hash` as a hex `String` (it is what the
//! `uk_logos_compile` report already carries). ENGRAM.md's byte layout wants 32
//! raw bytes, so the hex is decoded once here rather than changing
//! `translate_coreir`'s public type — that report is a frozen surface.
//!
//! ## The fallback is a first-class result, not an error
//!
//! ENGRAM.md §4: a fragment the gate rejects, or a reduction that hits the
//! iteration cap, yields a `window` key tagged `FLAG_FALLBACK`. Silently
//! degrading is the failure mode this whole feature creates, so the tag travels
//! with the key and `parse_rate` is measurable (E3).

use std::fmt;

use crate::ccg::DerivationTree;
use crate::core_ir::CoreIR;
use crate::harper_gate::HarperGate;
use crate::lexicon::Lexicon;

pub mod table;

pub use table::{placeholder_embedding, EngramTable, IngestStats};

/// Reduction iteration cap. A hostile corpus must not be able to hang ingest,
/// so this is a *skip*, not an error (ENGRAM.md §4).
pub const MAX_REDUCE_ITERS: u64 = 1_000_000;

/// Default window order `n`, matching the paper's §4.1 baseline.
pub const DEFAULT_WINDOW_N: usize = 3;

/// `flags` bit 0: this key is a fallback, not a UNF-derived key.
pub const FLAG_FALLBACK: u8 = 0b0000_0001;

/// Granularity of the segment a key was derived from (ENGRAM.md §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Granularity {
    /// token window of order `n = 2..N`
    Window,
    /// every CCG sub-derivation
    Subderiv,
    /// the whole fragment
    Sentence,
}

impl Granularity {
    /// The byte ENGRAM.md §2.2 assigns to this granularity.
    pub fn code(self) -> u8 {
        match self {
            Granularity::Window => 0,
            Granularity::Subderiv => 1,
            Granularity::Sentence => 2,
        }
    }
}

impl fmt::Display for Granularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Granularity::Window => "window",
            Granularity::Subderiv => "subderiv",
            Granularity::Sentence => "sentence",
        })
    }
}

/// The layout version in ENGRAM.md §2.2. Bump on any byte-layout change.
pub const LAYOUT_VERSION: u16 = 1;

/// A content-addressed engram key (ENGRAM.md §2.2).
///
/// `PartialEq` but not `Eq`: `l1_weight` is an `f64` and `NaN != NaN`. That is
/// deliberate and load-bearing — the layout uses `NaN` to mean "no weight"
/// (§2.2), so a key comparison must not pretend two absent weights are equal.
/// Compare `to_bytes()` when identity is what matters, since the layout is a
/// total order over keys.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramKey {
    pub granularity: Granularity,
    pub flags: u8,
    pub depth: u32,
    /// SHA-256 of the canonical net serialization (`deltanet::unf::unf_hash`).
    /// All-zero for a fallback key, which by construction has no normal form.
    pub unf_hash: [u8; 32],
    /// SHA-256 over the TED canonical serialization; all-zero when the fragment
    /// has no arithmetic content to canonicalize.
    pub ted_hash: [u8; 32],
    /// `Some(w)` carries the L1 probability (E5). `None` means "no weight",
    /// which is deliberately distinct from `Some(0.0)` — a zero-probability
    /// world is a real value (ENGRAM.md §2.2).
    pub l1_weight: Option<f64>,
}

impl EngramKey {
    /// A window key: surface-derived, never carries a UNF hash.
    pub fn window(tokens: &[String], n: usize) -> EngramKey {
        let joined = tokens.join(" ");
        EngramKey {
            granularity: Granularity::Window,
            flags: 0,
            depth: n as u32,
            unf_hash: sha256(joined.as_bytes()),
            ted_hash: [0u8; 32],
            l1_weight: None,
        }
    }

    /// Mark this key as a fallback (ENGRAM.md §4).
    pub fn as_fallback(mut self) -> Self {
        self.flags |= FLAG_FALLBACK;
        self
    }

    pub fn is_fallback(&self) -> bool {
        self.flags & FLAG_FALLBACK != 0
    }

    /// The 84-byte layout of ENGRAM.md §2.2. Versioned and `unf_hash`-first, so
    /// the stable addressing key sits at a fixed offset.
    pub fn to_bytes(&self) -> [u8; 84] {
        let mut out = [0u8; 84];
        out[0..4].copy_from_slice(b"ENGM");
        out[4..6].copy_from_slice(&LAYOUT_VERSION.to_le_bytes());
        out[6] = self.granularity.code();
        out[7] = self.flags;
        out[8..12].copy_from_slice(&self.depth.to_le_bytes());
        out[12..44].copy_from_slice(&self.unf_hash);
        out[44..76].copy_from_slice(&self.ted_hash);
        // NaN means "no weight" (ENGRAM.md §2.2).
        let w = self.l1_weight.unwrap_or(f64::NAN);
        out[76..84].copy_from_slice(&w.to_le_bytes());
        out
    }
}

/// Why a key could not be derived from a normal form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyError {
    /// The gate rejected the fragment, or it parsed to nothing.
    GateRejected(String),
    /// A derivation failed to compile to CoreIR.
    Compile(String),
    /// Reduction failed or hit the cap.
    Reduce(String),
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyError::GateRejected(m) => write!(f, "gate rejected the fragment: {}", m),
            KeyError::Compile(m) => write!(f, "compile failed: {}", m),
            KeyError::Reduce(m) => write!(f, "reduce failed: {}", m),
        }
    }
}

// ── SHA-256 ────────────────────────────────────────────────────────────────
// The crate already depends on `sha2` (logos/Cargo.toml) because
// `deltanet::unf` hashes with it. Reusing the same primitive rather than
// inventing a digest is the whole point of ENGRAM.md §1.

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    buf
}

/// Decode the 64-character hex `unf_hash` that `translate_coreir` reports.
fn hex32(s: &str) -> [u8; 32] {
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    if bytes.len() < 64 {
        return out;
    }
    for i in 0..32 {
        let hi = (bytes[2 * i] as char).to_digit(16);
        let lo = (bytes[2 * i + 1] as char).to_digit(16);
        match (hi, lo) {
            (Some(h), Some(l)) => out[i] = ((h << 4) | l) as u8,
            _ => return [0u8; 32],
        }
    }
    out
}

/// Tokenize on whitespace, matching what `cli` does. Kept here rather than
/// reaching into `cli` so the engram API is usable as a library.
pub fn tokenize(fragment: &str) -> Vec<String> {
    fragment.split_whitespace().map(String::from).collect()
}

// ── the pipeline ───────────────────────────────────────────────────────────

/// Derive one key from a CoreIR term, by delegating to `translate_coreir`.
///
/// `depth` is the sub-derivation depth, recorded in the key (ENGRAM.md §2.2).
fn key_of_coreir(ir: &CoreIR, granularity: Granularity, depth: u32) -> Result<EngramKey, KeyError> {
    let t = crate::translate::translate_coreir(ir).map_err(KeyError::Reduce)?;
    let unf = hex32(&t.unf_hash);
    if unf == [0u8; 32] {
        // A hash that failed to decode is not a key; refuse rather than
        // address every fallback to the all-zero slot.
        return Err(KeyError::Reduce(format!(
            "unf_hash {:?} is not 64 hex characters",
            t.unf_hash
        )));
    }
    let ted = t.ted_hash.as_deref().map(hex32).unwrap_or([0u8; 32]);
    Ok(EngramKey {
        granularity,
        flags: 0,
        depth,
        unf_hash: unf,
        ted_hash: ted,
        l1_weight: None,
    })
}

/// The `window` key for an order-`n` token window ending at the fragment end.
///
/// ENGRAM.md §2.1 keeps `window` for two reasons: it is the paper's baseline
/// for the E4 ablation, and it is the fallback when the UNF path is
/// unavailable. A window key is surface-derived, so it is tagged by `depth` =
/// `n` and carries no UNF hash.
pub fn window_key(tokens: &[String], n: usize) -> EngramKey {
    if tokens.is_empty() {
        return EngramKey::window(&[], n).as_fallback();
    }
    let start = tokens.len().saturating_sub(n);
    let slice = &tokens[start..];
    if slice.len() < n {
        // Not enough tokens for this order: ENGRAM.md §2.1 says the window
        // orders are `n = 2..N`, so an under-full window is not a key of this
        // order. Tag it as a fallback rather than inventing a shorter order,
        // which would alias onto a real key of that order.
        return EngramKey::window(slice, n).as_fallback();
    }
    EngramKey::window(slice, n)
}

/// Keys for every CCG sub-derivation, in derivation order.
pub fn subderiv_keys(tree: &DerivationTree, lexicon: &Lexicon) -> Vec<EngramKey> {
    let mut out = Vec::new();
    collect_subderivs(tree, lexicon, 0, &mut out);
    out
}

fn collect_subderivs(
    tree: &DerivationTree,
    lexicon: &Lexicon,
    depth: u32,
    out: &mut Vec<EngramKey>,
) {
    if let Ok(ir) = crate::core_ir::compile_to_core_ir(tree, lexicon) {
        let ir = crate::core_ir::linearity::insert_linearity(ir);
        if let Ok(k) = key_of_coreir(&ir, Granularity::Subderiv, depth) {
            out.push(k);
        }
    }
    // `DerivationTree` is a three-way enum (ccg/types.rs:134), so there is no
    // generic child accessor to reuse — the two binary forms are matched here.
    match tree {
        DerivationTree::Leaf { .. } => {}
        DerivationTree::Application { left, right, .. }
        | DerivationTree::Composition { left, right, .. } => {
            collect_subderivs(left, lexicon, depth + 1, out);
            collect_subderivs(right, lexicon, depth + 1, out);
        }
    }
}

/// Segment a fragment into keys at the requested granularity.
///
/// A granularity the fragment cannot support is **not** an error: ENGRAM.md §2.3
/// says a lookup at `g` does not fall back to a coarser granularity, so a miss
/// stays a miss. The only substitution this function makes is `window`, and
/// only as the tagged §4 fallback.
pub fn segment(fragment: &str, granularity: Granularity, lexicon: &Lexicon) -> Vec<EngramKey> {
    let tokens = tokenize(fragment);

    match granularity {
        Granularity::Window => {
            (2..=DEFAULT_WINDOW_N)
                .map(|n| window_key(&tokens, n))
                .collect()
        }
        Granularity::Sentence | Granularity::Subderiv => {
            let gate = HarperGate::new();
            let g = gate.lint(fragment);
            if !g.accepted {
                return vec![fallback(&tokens)];
            }
            // The gate's own tokenizer is authoritative: it is what was
            // accepted, so the CCG parse must see the same tokens the gate saw
            // rather than a second, possibly different, split.
            let tokens: Vec<String> = g.tokens.into_iter().map(|t| t.text).collect();
            let trees = crate::ccg::parse_sentence(&tokens, lexicon);
            if trees.is_empty() {
                return vec![fallback(&tokens)];
            }
            let mut out = Vec::new();
            for tree in &trees {
                match crate::core_ir::compile_to_core_ir(tree, lexicon) {
                    Err(_) => return vec![fallback(&tokens)],
                    Ok(ir) => {
                        let ir = crate::core_ir::linearity::insert_linearity(ir);
                        match key_of_coreir(&ir, granularity, 0) {
                            Err(_) => return vec![fallback(&tokens)],
                            Ok(k) => out.push(k),
                        }
                    }
                }
            }
            if granularity == Granularity::Subderiv {
                let mut subs = Vec::new();
                for tree in &trees {
                    subs.extend(subderiv_keys(tree, lexicon));
                }
                return subs;
            }
            out
        }
    }
}

/// The §4 fallback: a **tagged** `window` key, stored and looked up like any
/// other. The tag is what makes `parse_rate` measurable — an untagged window key
/// here would be indistinguishable from a genuine `window` entry, and the
/// corpus would silently fill with surface strings while reporting full
/// coverage.
///
/// Its granularity is `Window`, not the requested one, and that is deliberate:
/// ENGRAM.md §2.3 says a lookup at `g` must not fall back to a coarser `g`, so
/// the unparsed fragment is simply *absent* at `Sentence`/`Subderiv` — a miss
/// stays a miss — while the tagged window entry remains available at `Window`.
fn fallback(tokens: &[String]) -> EngramKey {
    window_key(tokens, DEFAULT_WINDOW_N).as_fallback()
}

/// Derive a single key, the top-level API of ENGRAM.md §5.
pub fn engram_key(
    fragment: &str,
    granularity: Granularity,
    lexicon: &Lexicon,
) -> Result<EngramKey, KeyError> {
    let keys = segment(fragment, granularity, lexicon);
    match keys.into_iter().next() {
        Some(k) => Ok(k),
        None => Err(KeyError::GateRejected("no segment produced a key".into())),
    }
}