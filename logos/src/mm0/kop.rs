//! The Kopperman fragment of the CNL: Hilbert-space claims, certified over
//! `mm0/kopperman.mm0`.
//!
//! The arithmetic fragment (`cnl` → `proof` → `emit`) decides claims about
//! Peano numerals. This module decides the claims of the *decidable
//! Kopperman–Solovay fragment*: equalities between scalar and vector terms of
//! the Hilbert-space layer — inner products of finite-head vectors with
//! literal (or universally quantified) coordinates. The chain is the same
//! shape as the arithmetic one, with the vector layer in front:
//!
//! ```text
//! sentence ──cnl──► KClaim ──raw──► NF ──renorm──► normal form + proof
//!                                              │
//!                         kopperman.mm0 + logos_math.mm0 ──► mm0-rs ──► mm0-c
//! ```
//!
//! * [`KClaim::parse`] is the surface syntax. It is *sort-directed*: scalar
//!   position parses scalars and vector position parses vectors, and a name
//!   used at both sorts in one sentence is refused. That is the formation
//!   check — Felix's "soft types" — and it lives entirely in the CNL, because
//!   MM0's `Term` is one sort: the theory cannot catch a sort slip, so the
//!   parser must.
//! * [`renorm`] is the ground normaliser. It rewrites a term to normal form
//!   with the K2 recursion rules (`coord_*`, `vadd_*`, `vscal_*`, `inpr_*`,
//!   `tail_*`) and the Peano laws, each step an MM0 proof term lifted through
//!   the congruence axioms. The rules are left-linear, non-overlapping and
//!   terminating, so a normal form exists and two terms are equal in the
//!   congruence they generate exactly when their normal forms *spell* the
//!   same — the fragment decides its own equations, like `proof::rewrite_claim`
//!   does for Peano arithmetic.
//! * [`certify`] assembles the certificate. A ground claim is decided by
//!   evaluation, cross-checked against the DeltaNet reducer on the claim's
//!   arithmetic expansion ([`KExpr::expand`]); a false one is refuted in the
//!   other direction (`teq L R` implies `tfals`). An open claim is closed by
//!   rewriting, and when rewriting alone does not reach, by the **bounded
//!   obvious layer**: the theory's laws the normaliser deliberately does *not*
//!   apply (commutativity, associativity, distributivity, the inner-product
//!   laws), each tried at the root first and then in chains of bounded depth,
//!   lifted through term contexts ("relators") and composed with `eq_trans` —
//!   calculational composition. Every candidate is *replayed by mm0-rs*:
//!   the search proposes, the elaborator disposes, exactly like the arithmetic
//!   fragment's `--hint` seam.
//! * [`LEXICON`] is the lexicon, one table from theory symbol to CNL phrase,
//!   which the parser consumes and the tests check against `kopperman.mm0`'s
//!   own `term` declarations ([`overlay_terms`]): the theory cannot grow a
//!   symbol the CNL does not know.
//!
//! Statements are emitted at the level of `tadd`/`tmul` — the *denoted*
//! operations — rather than the agent spines (`tapp tprim_add …`) the
//! arithmetic fragment uses, because no K2 axiom mentions agents; the net
//! cross-check still runs, one level up, on the expansion. What the fragment
//! does not decide it reports as a goal, never a guess: Kopperman's Warning
//! (Section V, 1967) is that one named infinite constant destroys
//! decidability, and no CNL sentence here can name a limit.

use std::collections::{BTreeMap, HashMap};

use super::bridge;
use super::cnl::number_value;
use super::emit::{self, Certificate, Verdict};
use super::proof::{self, Expr};

/// The full target theory of a Kopperman certificate: the agent/Peano layer,
/// then the Hilbert-space overlay, in the order `logos2mm0` concatenates them.
pub const KTHEORY: &str = concat!(
    include_str!("../../mm0/logos_math.mm0"),
    "\n",
    include_str!("../../mm0/kopperman.mm0")
);

/// The optional Kopperman/Mehler target theory: the base overlay plus the
/// tail-determination extension (`mm0/mehler.mm0`), selected with `--mehler`.
/// Written out literally because `concat!` only takes literals.
pub const MEHLER_THEORY: &str = concat!(
    include_str!("../../mm0/logos_math.mm0"),
    "\n",
    include_str!("../../mm0/kopperman.mm0"),
    "\n",
    include_str!("../../mm0/mehler.mm0")
);

/// Which target theory a certificate is built against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theory {
    /// The base Kopperman overlay: the finite-head skeleton, a variable
    /// vector's tail unknown.
    Base,
    /// The Kopperman/Mehler overlay (`--mehler`): the random tail is fully
    /// determined — `coord_tail`, `vadd_coord`, `vscal_coord`, `vec_eta`.
    Mehler,
}

impl Theory {
    /// The `.mm0` theory text a certificate embeds. A proof citing a
    /// `--mehler` axiom only elaborates against the mehler text, which is
    /// why every artifact carries the theory it was proved against.
    pub fn text(self) -> &'static str {
        match self {
            Theory::Base => KTHEORY,
            Theory::Mehler => MEHLER_THEORY,
        }
    }

    /// The rules the obvious layer may offer for this theory: the base
    /// catalogue, plus the determination laws under `--mehler`. The
    /// determination laws are deliberately *not* normaliser rewrite rules —
    /// see `mm0/mehler.mm0` — so this list is where they live.
    fn laws(self) -> Vec<&'static Law> {
        match self {
            Theory::Base => LAWS_BASE.iter().collect(),
            Theory::Mehler => LAWS_BASE.iter().chain(LAWS_MEHLER.iter()).collect(),
        }
    }

    /// The LLM rule catalogue for this theory.
    pub fn llm_system(self) -> &'static str {
        match self {
            Theory::Base => LLM_SYSTEM,
            Theory::Mehler => LLM_SYSTEM_MEHLER,
        }
    }
}

// ── the lexicon ─────────────────────────────────────────────────────────────
//
// One table, consulted by the parser *and* checked against the theory by the
// tests: a `(symbol, phrase)` pair where `phrase` is the exact surface string
// the parser eats, and a `None` phrase marks a symbol no sentence may name —
// the K3 ontological layer (limits, sequences) and the order predicates, which
// the equality fragment has no way to state. This is the lexicon idea from
// Felix's design made checkable: the vocabulary is data, the theory is parsed,
// and drift between them is a test failure rather than a silent gap.

/// `(theory symbol, CNL phrase, or None when sealed)`.
pub const LEXICON: &[(&str, Option<&str>)] = &[
    // sealed: order and roots — `Prop`-valued or beyond the equality fragment
    ("tle", None),
    ("tlt", None),
    ("tsqrt", None),
    ("tcbrt", None),
    // sealed: the K3 sequence layer. Kopperman's Warning: no sentence may name
    // a limit — these appear in theory axioms only, never in a certificate.
    ("tapply", None),
    ("tbounded", None),
    ("tsup", None),
    ("tnear", None),
    ("tcauchy", None),
    ("tlimit", None),
    ("tconv", None),
    ("tsummable", None),
    ("tpartial", None),
    ("tseries", None),
    ("trat", None),
    ("tclose", None),
    ("tvapprox", None),
    // sealed: the vector tail is not a phrase — it appears only inside normal
    // forms, produced by `inpr_cons`'s expansion.
    ("ttail", None),
    // spoken: the K2 constructors the CNL reads.
    ("tvnil", Some("the zero vector")),
    ("tvcons", Some("the vector of")),
    ("tcoord", Some("the coordinate")),
    ("tvadd", Some("the sum of")),
    ("tvscal", Some("the scalar")),
    ("tinpr", Some("the inner product of")),
];

/// The CNL phrase for a theory symbol. Panics when the symbol is not in the
/// lexicon — which is the point: the parser cannot mention a symbol the table
/// does not carry, and the table cannot omit one the theory declares (the
/// coverage test enforces that from the other side).
fn phrase(symbol: &str) -> &'static str {
    LEXICON
        .iter()
        .find(|(s, _)| *s == symbol)
        .and_then(|(_, p)| *p)
        .unwrap_or_else(|| panic!("`{symbol}` is sealed: no CNL phrase reads it"))
}

/// Every `term` declared by the Kopperman overlay, in order.
///
/// Parsed from the theory text itself, so [`LEXICON`]'s coverage is checked
/// against what the file *says*, not against a parallel list someone could
/// forget to update.
pub fn overlay_terms() -> Vec<String> {
    include_str!("../../mm0/kopperman.mm0")
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line.strip_prefix("term ")?;
            rest.split_whitespace()
                .next()
                .map(|name| name.trim_end_matches(['(', ':']).to_string())
        })
        .collect()
}

// ── the sorts ───────────────────────────────────────────────────────────────

/// The two sorts of the fragment. MM0 has one sort (`Term`), so these live
/// entirely in the CNL: they are what makes "the inner product of two and v"
/// a *formation* error rather than a certificate about a junk application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort {
    Scalar,
    Vector,
}

impl Sort {
    fn label(self) -> &'static str {
        match self {
            Sort::Scalar => "a scalar",
            Sort::Vector => "a vector",
        }
    }
}

// ── the term language ───────────────────────────────────────────────────────

/// A scalar expression of the fragment: arithmetic over the inner product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KExpr {
    Num(u64),
    Var(String),
    Add(Box<KExpr>, Box<KExpr>),
    Mul(Box<KExpr>, Box<KExpr>),
    /// the inner product of u and v
    Inpr(Box<VTerm>, Box<VTerm>),
    /// the coordinate i of v
    Coord(Box<VTerm>, Box<KExpr>),
}

/// A vector term: a finite head over a tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VTerm {
    /// the zero vector
    Nil,
    /// the vector of c, d, … — `tvcons c v`
    Cons(Box<KExpr>, Box<VTerm>),
    /// a vector variable
    Var(String),
    /// the sum of u and v
    VAdd(Box<VTerm>, Box<VTerm>),
    /// the scalar s times v
    VScal(Box<KExpr>, Box<VTerm>),
}

/// One side of a claim, at its sort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KSide {
    Scalar(KExpr),
    Vector(VTerm),
}

/// One sentence of the Kopperman fragment: two sides of the same sort, asked
/// (`question`) or asserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KClaim {
    pub lhs: KSide,
    pub rhs: KSide,
    pub question: bool,
}

impl KExpr {
    /// The mm0 term for this expression, as the claim states it.
    pub fn spell(&self) -> String {
        match self {
            KExpr::Num(n) => Expr::numeral(*n),
            KExpr::Var(v) => proof::param(v),
            KExpr::Add(a, b) => format!("(tadd {} {})", a.spell(), b.spell()),
            KExpr::Mul(a, b) => format!("(tmul {} {})", a.spell(), b.spell()),
            KExpr::Inpr(u, v) => format!("(tinpr {} {})", u.spell(), v.spell()),
            KExpr::Coord(v, i) => format!("(tcoord {} {})", v.spell(), i.spell()),
        }
    }

    /// A human-readable spelling, for diagnostics only.
    pub fn pretty(&self) -> String {
        match self {
            KExpr::Num(n) => n.to_string(),
            KExpr::Var(v) => v.clone(),
            KExpr::Add(a, b) => format!("{} plus {}", a.pretty(), b.pretty()),
            KExpr::Mul(a, b) => format!("{} times {}", a.pretty(), b.pretty()),
            KExpr::Inpr(u, v) => {
                format!("the inner product of {} and {}", u.pretty(), v.pretty())
            }
            KExpr::Coord(v, i) => format!("the coordinate {} of {}", i.pretty(), v.pretty()),
        }
    }

    /// The arithmetic expansion of this expression, when every vector in it
    /// is a literal finite head and every index is closed: the pure-`Expr`
    /// term the DeltaNet reducer corroborates ([`super::bridge::corroborate`]).
    ///
    /// This is the net cross-check for the Kopperman fragment: the inner
    /// product of two literal heads is the Σ of coordinate products, which is
    /// an arithmetic expression the existing lowering already compiles and
    /// folds. A side with a stuck vector or index has no expansion — it is
    /// open, and the answer rests on mm0 alone.
    pub fn expand(&self) -> Option<Expr> {
        match self {
            KExpr::Num(n) => Some(Expr::Num(*n)),
            KExpr::Var(v) => Some(Expr::Var(v.clone())),
            KExpr::Add(a, b) => Some(Expr::Add(Box::new(a.expand()?), Box::new(b.expand()?))),
            KExpr::Mul(a, b) => Some(Expr::Mul(Box::new(a.expand()?), Box::new(b.expand()?))),
            KExpr::Coord(v, i) => {
                let spine = v.expand_spine()?;
                let idx = i.expand()?.value()?;
                Some(spine.get(idx as usize).cloned().unwrap_or(Expr::Num(0)))
            }
            KExpr::Inpr(u, v) => {
                let us = u.expand_spine()?;
                let vs = v.expand_spine()?;
                let n = us.len().max(vs.len());
                let zero = Expr::Num(0);
                let mut acc = Expr::Num(0);
                for k in 0..n {
                    let a = us.get(k).unwrap_or(&zero).clone();
                    let b = vs.get(k).unwrap_or(&zero).clone();
                    acc = Expr::Add(Box::new(acc), Box::new(Expr::Mul(Box::new(a), Box::new(b))));
                }
                Some(acc)
            }
        }
    }

    /// Free variables with the sort they are used at.
    pub fn vars(&self) -> BTreeMap<String, Sort> {
        let mut out = BTreeMap::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, out: &mut BTreeMap<String, Sort>) {
        match self {
            KExpr::Num(_) => {}
            KExpr::Var(v) => {
                out.insert(v.clone(), Sort::Scalar);
            }
            KExpr::Add(a, b) | KExpr::Mul(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            KExpr::Inpr(u, v) => {
                u.collect_vars(out);
                v.collect_vars(out);
            }
            KExpr::Coord(v, i) => {
                v.collect_vars(out);
                i.collect_vars(out);
            }
        }
    }
}

impl VTerm {
    /// The mm0 term for this vector, as the claim states it.
    pub fn spell(&self) -> String {
        match self {
            VTerm::Nil => "tvnil".to_string(),
            VTerm::Cons(c, v) => format!("(tvcons {} {})", c.spell(), v.spell()),
            VTerm::Var(v) => proof::param(v),
            VTerm::VAdd(a, b) => format!("(tvadd {} {})", a.spell(), b.spell()),
            VTerm::VScal(s, v) => format!("(tvscal {} {})", s.spell(), v.spell()),
        }
    }

    pub fn pretty(&self) -> String {
        match self {
            VTerm::Nil => "the zero vector".to_string(),
            VTerm::Cons(c, v) => {
                let mut coords = vec![c.pretty()];
                let mut rest: &VTerm = v;
                while let VTerm::Cons(c, tail) = rest {
                    coords.push(c.pretty());
                    rest = &**tail;
                }
                match rest {
                    VTerm::Nil => format!("the vector of {}", coords.join(", ")),
                    // A spine over a variable tail is a normal form, not a
                    // phrase — nothing to say about it in the surface syntax.
                    other => {
                        format!(
                            "the vector of {} , … , {}",
                            coords.join(", "),
                            other.pretty()
                        )
                    }
                }
            }
            VTerm::Var(v) => v.clone(),
            VTerm::VAdd(a, b) => format!("the sum of {} and {}", a.pretty(), b.pretty()),
            VTerm::VScal(s, v) => format!("the scalar {} times {}", s.pretty(), v.pretty()),
        }
    }

    /// The literal head of this vector, when it is one: the coordinates of a
    /// finite spine, each expanded to arithmetic. `None` for a variable or a
    /// stuck application.
    fn expand_spine(&self) -> Option<Vec<Expr>> {
        match self {
            VTerm::Nil => Some(Vec::new()),
            VTerm::Cons(c, v) => {
                let mut out = vec![c.expand()?];
                out.extend(v.expand_spine()?);
                Some(out)
            }
            _ => None,
        }
    }

    fn vars(&self) -> BTreeMap<String, Sort> {
        let mut out = BTreeMap::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, out: &mut BTreeMap<String, Sort>) {
        match self {
            VTerm::Nil => {}
            VTerm::Cons(c, v) => {
                c.collect_vars(out);
                v.collect_vars(out);
            }
            VTerm::Var(v) => {
                out.insert(v.clone(), Sort::Vector);
            }
            VTerm::VAdd(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            VTerm::VScal(s, v) => {
                s.collect_vars(out);
                v.collect_vars(out);
            }
        }
    }
}

impl KSide {
    pub fn spell(&self) -> String {
        match self {
            KSide::Scalar(e) => e.spell(),
            KSide::Vector(v) => v.spell(),
        }
    }

    pub fn pretty(&self) -> String {
        match self {
            KSide::Scalar(e) => e.pretty(),
            KSide::Vector(v) => v.pretty(),
        }
    }

    fn vars(&self) -> BTreeMap<String, Sort> {
        match self {
            KSide::Scalar(e) => e.vars(),
            KSide::Vector(v) => v.vars(),
        }
    }
}

impl KClaim {
    /// The goal formula: `teq L R`.
    pub fn goal_of(&self) -> String {
        format!("teq {} {}", self.lhs.spell(), self.rhs.spell())
    }
}

// ── the surface syntax ──────────────────────────────────────────────────────
//
// ```text
// claim   := ["for all" vars ":"] body ["?"]
// body    := sexpr eq sexpr | vexpr eq vexpr
// eq      := "equals" | "is equal to" | "is" | "=" | "equal"
// sexpr   := satom {("plus" | "+") satom}      -- times binds tighter, left-assoc
// satom   := number | var | "the inner product of" vexpr "and" vexpr
//          | "the coordinate" sexpr "of" vexpr | "(" sexpr ")"
// vexpr   := "the zero vector" | "the vector of" satom {"," satom}
//          | "the sum of" vexpr {"," vexpr} | "the scalar" sexpr "times" vexpr
//          | var | "(" vexpr ")"
// ```
//
// A side's sort is decided by its leading word (`the zero vector` is a
// vector, `the inner product of …` a scalar, a bare word is read at the sort
// the position asks for); a name used at both sorts in one sentence is a
// formation error. That is the whole soft-type discipline: the theory cannot
// enforce it (MM0's `Term` is one sort), so the CNL does, and refuses what it
// cannot place rather than certifying a sentence about junk applications.

/// Tokenize for the Kopperman fragment: the arithmetic fragment's tokenizer,
/// plus `,` as a token. `cnl::tokenize` drops punctuation on purpose — its
/// grammar has no lists, and a stray period should not be a parse error —
/// but `the vector of 1, 2` and `the sum of u, v, w` need the comma as a
/// separator the parser can actually see.
fn tokenize_k(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        match c {
            '+' | '*' | '=' | '(' | ')' | '?' | ':' | ',' => {
                if !word.is_empty() {
                    out.push(std::mem::take(&mut word));
                }
                out.push(c.to_string());
            }
            c if c.is_alphanumeric() || c == '_' => word.extend(c.to_lowercase()),
            _ => {
                if !word.is_empty() {
                    out.push(std::mem::take(&mut word));
                }
            }
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

/// Whether the sentence contains a vector-constructor phrase. Its answer
/// fixes the sort of a bare side — see [`KClaim::parse`]. Scalar phrases
/// (`the inner product of`, `the coordinate`) do not count: they appear in
/// scalar claims, where a bare `u equals v` should stay scalar.
fn vector_claim(tokens: &[String]) -> bool {
    [
        phrase("tvnil"),
        phrase("tvcons"),
        phrase("tvadd"),
        phrase("tvscal"),
    ]
    .iter()
    .any(|p| contains_phrase(tokens, p))
}

/// Whether the token list contains this (space-separated) phrase.
fn contains_phrase(tokens: &[String], phrase: &str) -> bool {
    let words: Vec<&str> = phrase.split_whitespace().collect();
    if words.is_empty() || tokens.len() < words.len() {
        return false;
    }
    tokens
        .windows(words.len())
        .any(|w| w.iter().map(String::as_str).eq(words.iter().copied()))
}

struct KParser {
    tokens: Vec<String>,
    pos: usize,
    sorts: BTreeMap<String, Sort>,
}

impl KClaim {
    pub fn parse(text: &str) -> Result<KClaim, String> {
        let tokens = tokenize_k(text);
        if tokens.is_empty() {
            return Err("empty sentence".into());
        }
        let mut p = KParser {
            tokens,
            pos: 0,
            sorts: BTreeMap::new(),
        };

        // The sort of a *bare* side (`u equals v`, `… equals u`) cannot come
        // from its own shape — a variable is either sort. It comes from the
        // sentence: a claim that contains a vector phrase anywhere is a
        // vector claim, and its bare words are vector variables. Positions
        // inside a constructor keep their own sort either way — `the
        // coordinate 0 of u` reads `u` as a vector whatever else the sentence
        // says — so this default only ever decides sides that are nothing but
        // a word or a parenthesis.
        let default_sort = if vector_claim(&p.tokens) {
            Sort::Vector
        } else {
            Sort::Scalar
        };

        let declared = p.parse_forall()?;

        if p.eat("it") {
            p.eat("is");
            p.eat("true");
            p.eat("that");
        }
        let mut question = false;
        if p.eat("does") || p.eat("is") {
            question = true;
        }
        let leading_is = question;

        let lhs = p.parse_side(default_sort)?;
        if p.eat("=") || p.eat("equals") {
        } else if p.eat("equal") {
            p.eat("to");
        } else if p.eat("is") {
            p.eat("equal");
            p.eat("to");
        } else if !leading_is {
            return Err(match p.peek() {
                Some(t) => format!("expected `equals`, found `{t}`"),
                None => "expected `equals`, found the end of the sentence".into(),
            });
        }
        let rhs = p.parse_side(default_sort)?;
        if p.eat("?") {
            question = true;
        }
        if let Some(t) = p.peek() {
            return Err(format!("unexpected `{t}` after the claim"));
        }

        // Both sides must have one sort: a vector equal to a scalar is not a
        // sentence of the fragment, and certifying it would certify junk.
        if lhs.sort() != rhs.sort() {
            return Err(format!(
                "the two sides have different sorts: the left is {}, the right is {}",
                lhs.sort().label(),
                rhs.sort().label()
            ));
        }

        if !declared.is_empty() {
            let mut used: Vec<String> = lhs.vars().into_keys().collect();
            for v in rhs.vars().into_keys() {
                if !used.contains(&v) {
                    used.push(v);
                }
            }
            used.sort();
            let mut declared_sorted = declared.clone();
            declared_sorted.sort();
            if declared_sorted != used {
                return Err(format!(
                    "`for all` names {} but the claim uses {}",
                    declared_sorted.join(", "),
                    if used.is_empty() {
                        "no variables".to_string()
                    } else {
                        used.join(", ")
                    }
                ));
            }
        }

        Ok(KClaim { lhs, rhs, question })
    }
}

impl KSide {
    fn sort(&self) -> Sort {
        match self {
            KSide::Scalar(_) => Sort::Scalar,
            KSide::Vector(_) => Sort::Vector,
        }
    }
}

impl KParser {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn at_any(&self, words: &[&str]) -> bool {
        matches!(self.peek(), Some(t) if words.contains(&t))
    }

    fn eat(&mut self, word: &str) -> bool {
        if self.at_any(&[word]) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Consume a multi-word lexicon phrase (`the inner product of`).
    fn eat_phrase(&mut self, phrase: &str) -> bool {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let mut k = 0;
        for w in &words {
            if self.tokens.get(self.pos + k).map(String::as_str) == Some(*w) {
                k += 1;
            } else {
                return false;
            }
        }
        self.pos += k;
        true
    }

    fn expect(&mut self, word: &str) -> Result<(), String> {
        if self.eat(word) {
            Ok(())
        } else {
            Err(match self.peek() {
                Some(t) => format!("expected `{word}`, found `{t}`"),
                None => format!("expected `{word}`, found the end of the sentence"),
            })
        }
    }

    /// Record the sort a name is used at; a second, different sort for the
    /// same name is the formation error that makes the fragment's soft types
    /// checkable at parse time.
    fn use_var(&mut self, name: &str, sort: Sort) -> Result<(), String> {
        match self.sorts.get(name) {
            None => {
                self.sorts.insert(name.to_string(), sort);
                Ok(())
            }
            Some(prev) if *prev == sort => Ok(()),
            Some(prev) => Err(format!(
                "`{name}` is used as {} here and as {} elsewhere in the claim: \
                 one name cannot have two sorts",
                sort.label(),
                prev.label()
            )),
        }
    }

    fn parse_forall(&mut self) -> Result<Vec<String>, String> {
        if !self.eat("for") {
            return Ok(Vec::new());
        }
        if !(self.eat("all") || self.eat("every") || self.eat("each")) {
            return Err(format!(
                "expected `all` after `for`, found `{}`",
                self.peek().unwrap_or("the end of the sentence")
            ));
        }
        let mut names = Vec::new();
        loop {
            match self.peek() {
                Some(":") => {
                    self.pos += 1;
                    return Ok(names);
                }
                Some(t) if t.chars().next().is_some_and(|c| c.is_alphabetic()) => {
                    if names.iter().any(|n| n == t) {
                        return Err(format!("`for all` names `{t}` twice"));
                    }
                    names.push(t.to_string());
                    self.pos += 1;
                }
                Some(t) => return Err(format!("expected a variable, found `{t}`")),
                None => return Err("expected `:` after the variables".into()),
            }
            // `,` and `and` both separate — this tokenizer sees the comma,
            // where the arithmetic fragment's drops it.
            self.eat(",");
            self.eat("and");
        }
    }

    /// One side of the claim. The leading word decides the sort when it is a
    /// constructor (`the zero vector` is a vector, `the inner product of …`
    /// a scalar); a bare word or parenthesis takes the sentence's default,
    /// which [`vector_claim`] fixed before parsing began.
    fn parse_side(&mut self, default: Sort) -> Result<KSide, String> {
        if self.peek() == Some("the") {
            match self.tokens.get(self.pos + 1).map(String::as_str) {
                Some("zero") | Some("vector") | Some("sum") | Some("scalar") => {
                    return Ok(KSide::Vector(self.parse_vector()?));
                }
                Some("inner") | Some("coordinate") => {
                    return Ok(KSide::Scalar(self.parse_sexpr()?));
                }
                _ => {}
            }
        }
        match default {
            Sort::Scalar => Ok(KSide::Scalar(self.parse_sexpr()?)),
            Sort::Vector => {
                // A bare word or parenthesis: the sort comes from what the
                // name already means in this sentence — one name has one
                // sort, which the formation check insists on — and otherwise
                // from the shape: an expression with an operator is scalar
                // (`u plus w`; vector addition spells `the sum of`), a lone
                // word or an operator-free parenthesis group is a vector.
                if let Some(word) = self.peek()
                    && self.sorts.contains_key(word)
                {
                    return match self.sorts.get(word) {
                        Some(Sort::Scalar) => Ok(KSide::Scalar(self.parse_sexpr()?)),
                        _ => Ok(KSide::Vector(self.parse_vector()?)),
                    };
                }
                if self.looks_scalar() {
                    Ok(KSide::Scalar(self.parse_sexpr()?))
                } else {
                    Ok(KSide::Vector(self.parse_vector()?))
                }
            }
        }
    }

    /// Whether the side ahead reads as a scalar expression rather than a
    /// vector: a number, a bare word followed by an operator, or a
    /// parenthesis group containing one. A number is always a scalar —
    /// `… equals 11` inside a vector claim is a scalar right-hand side, not
    /// a vector variable called `11`.
    fn looks_scalar(&self) -> bool {
        match self.peek() {
            Some("the") => false, // a constructor decided above
            Some(t) if number_value(t).is_some() => true,
            Some(t) if t.chars().next().is_some_and(|c| c.is_alphabetic()) => matches!(
                self.tokens.get(self.pos + 1).map(String::as_str),
                Some("plus" | "+" | "times" | "*")
            ),
            Some("(") => {
                let mut depth = 0usize;
                for t in &self.tokens[self.pos..] {
                    match t.as_str() {
                        "(" => depth += 1,
                        ")" => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                break;
                            }
                        }
                        "plus" | "+" => return true,
                        "times" | "*" if depth == 1 => return true,
                        _ => {}
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn parse_sexpr(&mut self) -> Result<KExpr, String> {
        let mut e = self.parse_sterm()?;
        while self.at_any(&["plus", "+"]) {
            self.pos += 1;
            let r = self.parse_sterm()?;
            e = KExpr::Add(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_sterm(&mut self) -> Result<KExpr, String> {
        let mut e = self.parse_satom()?;
        while self.at_any(&["times", "*"]) {
            self.pos += 1;
            let r = self.parse_satom()?;
            e = KExpr::Mul(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_satom(&mut self) -> Result<KExpr, String> {
        let token = match self.peek() {
            Some(t) => t.to_string(),
            None => {
                return Err("expected a scalar expression, found the end of the sentence".into());
            }
        };

        if self.eat_phrase(phrase("tinpr")) {
            let u = self.parse_vector()?;
            self.expect("and")?;
            let v = self.parse_vector()?;
            return Ok(KExpr::Inpr(Box::new(u), Box::new(v)));
        }
        if self.eat_phrase(phrase("tcoord")) {
            let i = self.parse_sexpr()?;
            self.expect("of")?;
            let v = self.parse_vector()?;
            return Ok(KExpr::Coord(Box::new(v), Box::new(i)));
        }
        if token == "(" {
            self.pos += 1;
            let e = self.parse_sexpr()?;
            self.expect(")")?;
            return Ok(e);
        }
        if let Some(n) = number_value(&token) {
            self.pos += 1;
            if let Some(next) = self.peek().and_then(number_value)
                && n % 10 == 0
                && n >= 20
                && (1..=9).contains(&next)
            {
                self.pos += 1;
                return Ok(KExpr::Num(n + next));
            }
            return Ok(KExpr::Num(n));
        }
        if token.chars().next().is_some_and(|c| c.is_alphabetic()) {
            // A vector word in scalar position is a formation error, not a
            // variable named `zero` or `sum`.
            if matches!(token.as_str(), "zero" | "sum" | "vector" | "scalar") {
                return Err(format!(
                    "`{token}` starts a vector expression, and this position takes a scalar"
                ));
            }
            self.pos += 1;
            self.use_var(&token, Sort::Scalar)?;
            return Ok(KExpr::Var(token));
        }
        Err(format!(
            "expected a scalar expression, found `{token}` (a vector belongs in vector position: \
             `the inner product of u and v`, `the vector of 1, 2`)"
        ))
    }

    fn parse_vector(&mut self) -> Result<VTerm, String> {
        if self.eat_phrase(phrase("tvnil")) {
            return Ok(VTerm::Nil);
        }
        if self.eat_phrase(phrase("tvcons")) {
            let mut coords = vec![self.parse_satom()?];
            while self.eat(",") {
                coords.push(self.parse_satom()?);
            }
            let mut v = VTerm::Nil;
            for c in coords.into_iter().rev() {
                v = VTerm::Cons(Box::new(c), Box::new(v));
            }
            return Ok(v);
        }
        if self.eat_phrase(phrase("tvadd")) {
            let mut items = vec![self.parse_vector()?];
            // `u, v, w` and `u and v and w` both read as one sum; a sum of
            // summands that are themselves inner products needs parens
            // (`the inner product of (the sum of u and v) and w`), because
            // `and` also separates the inner product's two arguments.
            while self.eat(",") || self.eat("and") {
                items.push(self.parse_vector()?);
            }
            if items.len() < 2 {
                return Err(format!(
                    "`{}` needs at least two vectors to sum",
                    phrase("tvadd")
                ));
            }
            let mut iter = items.into_iter();
            let mut acc = iter.next().expect("checked above");
            for v in iter {
                acc = VTerm::VAdd(Box::new(acc), Box::new(v));
            }
            return Ok(acc);
        }
        if self.eat_phrase(phrase("tvscal")) {
            // The scaling scalar is one *atom* (parenthesise to compound it):
            // `the scalar a times b times v` would otherwise read `a times b`
            // as the scale and strand `times v`. One reading, always.
            let s = self.parse_satom()?;
            self.expect("times")?;
            let v = self.parse_vector()?;
            return Ok(VTerm::VScal(Box::new(s), Box::new(v)));
        }
        if self.peek() == Some("(") {
            self.pos += 1;
            let v = self.parse_vector()?;
            self.expect(")")?;
            return Ok(v);
        }
        match self.peek() {
            Some(t) if t.chars().next().is_some_and(|c| c.is_alphabetic()) => {
                let t = t.to_string();
                if matches!(t.as_str(), "inner" | "coordinate") {
                    return Err(format!(
                        "`{t}` starts a scalar expression, and this position takes a vector"
                    ));
                }
                // `the` opens a phrase; reading it as a variable would turn
                // a misspelled phrase into a claim about a vector called
                // `the`, and report the failure three words later.
                if t == "the" {
                    return Err(format!(
                        "expected a vector phrase after `the` — {}, {}, {} or {} — found a word \
                         this grammar does not know",
                        phrase("tvnil"),
                        phrase("tvcons"),
                        phrase("tvadd"),
                        phrase("tvscal")
                    ));
                }
                // A spelled number is a scalar, and `two` is a number — not a
                // variable that happens to look like one.
                if number_value(&t).is_some() {
                    return Err(format!(
                        "expected a vector, found `{t}`: a number is a scalar, so it cannot be a \
                         vector argument (scalars go where the position takes one: the \
                         coordinates of `{}`, the scale of `{}`)",
                        phrase("tvcons"),
                        phrase("tvscal")
                    ));
                }
                self.pos += 1;
                self.use_var(&t, Sort::Vector)?;
                Ok(VTerm::Var(t))
            }
            Some(t) => Err(format!(
                "expected a vector ({} …), found `{t}`",
                phrase("tvnil")
            )),
            None => Err(format!(
                "expected a vector ({} …), found the end of the sentence",
                phrase("tvnil")
            )),
        }
    }
}

// ── normal forms ────────────────────────────────────────────────────────────
//
// One enum for both sorts, so the calculational search can walk any term:
// paths are child indices, and every node knows its congruence axiom.
//
// Two invariants hold by construction and are what makes normal-form
// *equality* a spelling comparison:
//
// * `Arith` carries only a numeral, a variable, or a successor chain over a
//   variable — the arithmetic layer's own normal form. A sum or product that
//   cannot reduce lives in `TAdd`/`TMul`, never inside `Arith`.
// * `Ts` never wraps an `Arith` (it would fold into it), so the Peano
//   structure of a stuck term has exactly one spelling.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NF {
    // scalar layer
    Arith(Expr),
    Inpr(Box<NF>, Box<NF>),
    Coord(Box<NF>, Box<NF>),
    Ts(Box<NF>),
    TAdd(Box<NF>, Box<NF>),
    TMul(Box<NF>, Box<NF>),
    // vector layer
    VNil,
    VCons(Box<NF>, Box<NF>),
    VVar(String),
    VAdd(Box<NF>, Box<NF>),
    VScal(Box<NF>, Box<NF>),
    VTail(Box<NF>),
}

impl NF {
    pub fn sort(&self) -> Sort {
        match self {
            NF::Arith(_)
            | NF::Inpr(..)
            | NF::Coord(..)
            | NF::Ts(_)
            | NF::TAdd(..)
            | NF::TMul(..) => Sort::Scalar,
            NF::VNil
            | NF::VCons(..)
            | NF::VVar(_)
            | NF::VAdd(..)
            | NF::VScal(..)
            | NF::VTail(_) => Sort::Vector,
        }
    }

    /// The canonical mm0 spelling — the term as the certificate states it.
    /// Normal-form equality *is* spelling equality: the goal text is what mm0
    /// checks, so two structures that spell the same are the same term.
    pub fn spelling(&self) -> String {
        match self {
            NF::Arith(e) => e.peano(),
            NF::Inpr(u, v) => format!("(tinpr {} {})", u.spelling(), v.spelling()),
            NF::Coord(v, i) => format!("(tcoord {} {})", v.spelling(), i.spelling()),
            NF::Ts(x) => format!("(ts {})", x.spelling()),
            NF::TAdd(a, b) => format!("(tadd {} {})", a.spelling(), b.spelling()),
            NF::TMul(a, b) => format!("(tmul {} {})", a.spelling(), b.spelling()),
            NF::VNil => "tvnil".to_string(),
            NF::VCons(c, v) => format!("(tvcons {} {})", c.spelling(), v.spelling()),
            NF::VVar(v) => proof::param(v),
            NF::VAdd(a, b) => format!("(tvadd {} {})", a.spelling(), b.spelling()),
            NF::VScal(s, v) => format!("(tvscal {} {})", s.spelling(), v.spelling()),
            NF::VTail(v) => format!("(ttail {})", v.spelling()),
        }
    }

    /// The value of a fully computed normal form, when it is one.
    pub fn ground_value(&self) -> Option<u64> {
        match self {
            NF::Arith(e) => e.value(),
            _ => None,
        }
    }

    fn children(&self) -> Vec<NF> {
        match self {
            NF::Arith(_) | NF::VNil | NF::VVar(_) => Vec::new(),
            NF::Inpr(a, b) | NF::Coord(a, b) => vec![(**a).clone(), (**b).clone()],
            NF::Ts(x) | NF::VTail(x) => vec![(**x).clone()],
            NF::TAdd(a, b)
            | NF::TMul(a, b)
            | NF::VAdd(a, b)
            | NF::VScal(a, b)
            | NF::VCons(a, b) => vec![(**a).clone(), (**b).clone()],
        }
    }

    fn rebuild(&self, mut kids: Vec<NF>) -> NF {
        match self {
            NF::Arith(_) | NF::VNil | NF::VVar(_) => self.clone(),
            NF::Ts(_) => NF::Ts(Box::new(kids.pop().expect("one child"))),
            NF::VTail(_) => NF::VTail(Box::new(kids.pop().expect("one child"))),
            _ => {
                let b = kids.pop().expect("two children");
                let a = kids.pop().expect("two children");
                match self {
                    NF::Inpr(..) => NF::Inpr(Box::new(a), Box::new(b)),
                    NF::Coord(..) => NF::Coord(Box::new(a), Box::new(b)),
                    NF::TAdd(..) => NF::TAdd(Box::new(a), Box::new(b)),
                    NF::TMul(..) => NF::TMul(Box::new(a), Box::new(b)),
                    NF::VCons(..) => NF::VCons(Box::new(a), Box::new(b)),
                    NF::VAdd(..) => NF::VAdd(Box::new(a), Box::new(b)),
                    NF::VScal(..) => NF::VScal(Box::new(a), Box::new(b)),
                    _ => unreachable!("rebuild on an atom"),
                }
            }
        }
    }

    fn child(&self, i: usize) -> Option<NF> {
        self.children().into_iter().nth(i)
    }
}

/// Build the raw NF of a claim side (structure only — `renorm` normalises).
fn raw_side(side: &KSide) -> NF {
    match side {
        KSide::Scalar(e) => raw_scalar(e),
        KSide::Vector(v) => raw_vector(v),
    }
}

fn raw_scalar(e: &KExpr) -> NF {
    match e {
        KExpr::Num(n) => NF::Arith(Expr::Num(*n)),
        KExpr::Var(v) => NF::Arith(Expr::Var(v.clone())),
        KExpr::Add(a, b) => NF::TAdd(Box::new(raw_scalar(a)), Box::new(raw_scalar(b))),
        KExpr::Mul(a, b) => NF::TMul(Box::new(raw_scalar(a)), Box::new(raw_scalar(b))),
        KExpr::Inpr(u, v) => NF::Inpr(Box::new(raw_vector(u)), Box::new(raw_vector(v))),
        KExpr::Coord(v, i) => NF::Coord(Box::new(raw_vector(v)), Box::new(raw_scalar(i))),
    }
}

fn raw_vector(v: &VTerm) -> NF {
    match v {
        VTerm::Nil => NF::VNil,
        VTerm::Cons(c, t) => NF::VCons(Box::new(raw_scalar(c)), Box::new(raw_vector(t))),
        VTerm::Var(x) => NF::VVar(x.clone()),
        VTerm::VAdd(a, b) => NF::VAdd(Box::new(raw_vector(a)), Box::new(raw_vector(b))),
        VTerm::VScal(s, u) => NF::VScal(Box::new(raw_scalar(s)), Box::new(raw_vector(u))),
    }
}

/// An `Arith` node: the arithmetic layer's normal form is a numeral, a
/// variable, or a successor chain over a variable — never a sum or product
/// (those live in `TAdd`/`TMul`), which is what keeps one spelling per term.
fn arith(e: Expr) -> NF {
    debug_assert!(
        matches!(e, Expr::Num(_) | Expr::Var(_) | Expr::Suc(_)),
        "Arith carries only the arithmetic normal form"
    );
    NF::Arith(e)
}

fn numeral(n: u64) -> NF {
    arith(Expr::Num(n))
}

fn is_zero(n: &NF) -> bool {
    matches!(n, NF::Arith(Expr::Num(0)))
}

/// Peel one `ts` off a successor-shaped normal form: the operand `add_s` /
/// `mul_s` / `coord_cons_s` consume.
fn peel(n: &NF) -> Option<NF> {
    match n {
        NF::Arith(Expr::Num(m)) if *m > 0 => Some(numeral(m - 1)),
        NF::Arith(Expr::Suc(e)) => Some(arith((**e).clone())),
        NF::Ts(x) => Some((**x).clone()),
        _ => None,
    }
}

/// `ts` of a normal form, folding the `Arith` case into the arithmetic layer
/// so the successor structure has exactly one representation.
fn ts(n: NF) -> NF {
    match n {
        NF::Arith(e) => arith(Expr::Suc(Box::new(e))),
        other => NF::Ts(Box::new(other)),
    }
}

// ── the normaliser ──────────────────────────────────────────────────────────
//
// Every function returns the normal form and a proof of `teq <input> <NF>`.
// The rules are the K2 recursion plus Peano's, oriented the way the axioms
// state them; each step is lifted through the congruence axioms, so the
// result is a full MM0 proof term, not just a value.

/// A two-premise congruence lift. Unlike the single-premise [`proof::lift`],
/// the premises are positional: a proof cannot be dropped from the middle
/// without shifting the one behind it into its slot, so a reflexive premise
/// is passed as `(eq_refl)`.
fn lift2(rule: &str, p1: &str, p2: &str) -> String {
    let r1 = if p1.is_empty() { "(eq_refl)" } else { p1 };
    let r2 = if p2.is_empty() { "(eq_refl)" } else { p2 };
    if r1 == "(eq_refl)" && r2 == "(eq_refl)" {
        "(eq_refl)".to_string()
    } else {
        format!("({rule} {r1} {r2})")
    }
}

/// `tadd` normalised: `add_z` when the right operand is zero, one `add_s`
/// peel per successor, and the stuck sum otherwise.
fn nf_add(a: &NF, b: &NF) -> (NF, String) {
    if is_zero(b) {
        return (a.clone(), "(add_z)".to_string());
    }
    if let Some(c) = peel(b) {
        let (r, p) = nf_add(a, &c);
        return (ts(r), proof::chain("(add_s)", &proof::lift("eq_ts", &p)));
    }
    (
        NF::TAdd(Box::new(a.clone()), Box::new(b.clone())),
        "(eq_refl)".to_string(),
    )
}

/// `tmul` normalised: `mul_z` / `mul_s` peel the right operand, and each
/// unfolding adds one copy of the left operand.
fn nf_mul(a: &NF, b: &NF) -> (NF, String) {
    if is_zero(b) {
        return (numeral(0), "(mul_z)".to_string());
    }
    if let Some(c) = peel(b) {
        let (product, p) = nf_mul(a, &c);
        let (sum, p_add) = nf_add(&product, a);
        return (
            sum,
            proof::chain(
                &proof::chain("(mul_s)", &proof::lift("eq_add_l", &p)),
                &p_add,
            ),
        );
    }
    (
        NF::TMul(Box::new(a.clone()), Box::new(b.clone())),
        "(eq_refl)".to_string(),
    )
}

/// `tcoord` of a vector normal form at an index: `coord_nil` reads zero off
/// the zero vector for *any* index, `coord_cons_z`/`coord_cons_s` walk a
/// literal head.
fn nf_coord(v: &NF, i: &NF) -> (NF, String) {
    match v {
        NF::VNil => (numeral(0), "(coord_nil)".to_string()),
        NF::VCons(c, rest) => {
            if is_zero(i) {
                return ((**c).clone(), "(coord_cons_z)".to_string());
            }
            if let Some(i2) = peel(i) {
                let (r, p) = nf_coord(rest, &i2);
                return (r, proof::chain("(coord_cons_s)", &p));
            }
            (
                NF::Coord(Box::new(v.clone()), Box::new(i.clone())),
                "(eq_refl)".to_string(),
            )
        }
        _ => (
            NF::Coord(Box::new(v.clone()), Box::new(i.clone())),
            "(eq_refl)".to_string(),
        ),
    }
}

/// `ttail` of a vector normal form: `tail_nil` / `tail_cons`, else stuck.
fn nf_vtail(v: &NF) -> (NF, String) {
    match v {
        NF::VNil => (NF::VNil, "(tail_nil)".to_string()),
        NF::VCons(_, rest) => ((**rest).clone(), "(tail_cons)".to_string()),
        _ => (NF::VTail(Box::new(v.clone())), "(eq_refl)".to_string()),
    }
}

/// The inner product of two *already normalised* vectors.
///
/// Recursion on the first head — `inpr_nil` / `inpr_cons` — reading the
/// second's coordinates. When the first argument is stuck and the second is
/// a literal head, `inpr_sym` moves the computing argument forward first:
/// that is what makes `u · tvnil` compute to zero and `u · tvcons c w`
/// expand, without which half the decidable fragment would be stuck.
fn nf_inpr(u: &NF, v: &NF) -> (NF, String) {
    match u {
        NF::VNil => (numeral(0), "(inpr_nil)".to_string()),
        NF::VCons(c, rest) => {
            let (coord, pcoord) = nf_coord(v, &numeral(0));
            let (prod, pprod) = nf_mul(c, &coord);
            let (vt, pvt) = nf_vtail(v);
            let (tail, ptail) = nf_inpr(rest, &vt);
            let (sum, psum) = nf_add(&prod, &tail);
            // `inpr_cons` rewrites the product; the rewrites of its three
            // pieces are congruence inside the sum it produced.
            let s1 = proof::lift("eq_add_l", &proof::lift("eq_mul_r", &pcoord));
            let s2 = proof::lift("eq_add_l", &pprod);
            let s3 = proof::lift("eq_add_r", &lift2("eq_tinpr", "(eq_refl)", &pvt));
            let s4 = proof::lift("eq_add_r", &ptail);
            let inner = proof::chain(
                &proof::chain(&s1, &s2),
                &proof::chain(&proof::chain(&s3, &s4), &psum),
            );
            (sum, proof::chain("(inpr_cons)", &inner))
        }
        _ => match v {
            NF::VNil => (numeral(0), proof::chain("(inpr_sym)", "(inpr_nil)")),
            NF::VCons(..) => {
                let (r, p) = nf_inpr(v, u);
                (r, proof::chain("(inpr_sym)", &p))
            }
            _ => (
                NF::Inpr(Box::new(u.clone()), Box::new(v.clone())),
                "(eq_refl)".to_string(),
            ),
        },
    }
}

/// `tvadd` of two normalised vectors: the spines merge coordinatewise, the
/// zero vector pads either side, and a stuck addend stays stuck.
fn nf_vadd(a: &NF, b: &NF) -> (NF, String) {
    match (a, b) {
        (NF::VNil, _) => (b.clone(), "(vadd_nil)".to_string()),
        (_, NF::VNil) => match a {
            NF::VCons(..) => (a.clone(), "(vadd_cons_nil)".to_string()),
            _ => (
                NF::VAdd(Box::new(a.clone()), Box::new(b.clone())),
                "(eq_refl)".to_string(),
            ),
        },
        (NF::VCons(..), NF::VCons(..)) => {
            let (ca, ra) = (a.child(0).expect("cons"), a.child(1).expect("cons"));
            let (cb, rb) = (b.child(0).expect("cons"), b.child(1).expect("cons"));
            let (h, ph) = nf_add(&ca, &cb);
            let (t, pt) = nf_vadd(&ra, &rb);
            (
                NF::VCons(Box::new(h), Box::new(t)),
                proof::chain("(vadd_cons_cons)", &lift2("eq_tvcons", &ph, &pt)),
            )
        }
        _ => (
            NF::VAdd(Box::new(a.clone()), Box::new(b.clone())),
            "(eq_refl)".to_string(),
        ),
    }
}

/// `tvscal` of a normalised scalar and vector: distribute over a literal
/// head, absorb into the zero vector, else stuck.
fn nf_vscal(s: &NF, v: &NF) -> (NF, String) {
    match v {
        NF::VNil => (NF::VNil, "(vscal_nil)".to_string()),
        NF::VCons(c, rest) => {
            let (h, ph) = nf_mul(s, c);
            let (t, pt) = nf_vscal(s, rest);
            (
                NF::VCons(Box::new(h), Box::new(t)),
                proof::chain("(vscal_cons)", &lift2("eq_tvcons", &ph, &pt)),
            )
        }
        _ => (
            NF::VScal(Box::new(s.clone()), Box::new(v.clone())),
            "(eq_refl)".to_string(),
        ),
    }
}

/// Re-normalise a normal form, bottom-up, returning the normal form and a
/// proof of `teq <input> <result>`.
///
/// This is the same rewrite system as the constructors above, run over NF
/// structure: it is what makes a *stuck* term (`tmul 2 (u·u)`) reducible
/// again after a law step has rearranged it (`tmul (u·u) 2`), which is the
/// seam the calculational search composes through.
pub fn renorm(n: &NF) -> (NF, String) {
    match n {
        NF::Arith(_) | NF::VNil | NF::VVar(_) => (n.clone(), "(eq_refl)".to_string()),
        NF::Ts(x) => {
            let (nx, px) = renorm(x);
            (ts(nx), proof::lift("eq_ts", &px))
        }
        NF::TAdd(a, b) => {
            let (na, pa) = renorm(a);
            let (nb, pb) = renorm(b);
            let step = proof::chain(&proof::lift("eq_add_r", &pb), &proof::lift("eq_add_l", &pa));
            let (nf, p) = nf_add(&na, &nb);
            (nf, proof::chain(&step, &p))
        }
        NF::TMul(a, b) => {
            let (na, pa) = renorm(a);
            let (nb, pb) = renorm(b);
            let step = proof::chain(&proof::lift("eq_mul_r", &pb), &proof::lift("eq_mul_l", &pa));
            let (nf, p) = nf_mul(&na, &nb);
            (nf, proof::chain(&step, &p))
        }
        NF::Inpr(u, v) => {
            let (nu, pu) = renorm(u);
            let (nv, pv) = renorm(v);
            let step = lift2("eq_tinpr", &pu, &pv);
            let (nf, p) = nf_inpr(&nu, &nv);
            (nf, proof::chain(&step, &p))
        }
        NF::Coord(v, i) => {
            let (nv, pv) = renorm(v);
            let (ni, pi) = renorm(i);
            let step = lift2("eq_tcoord", &pv, &pi);
            let (nf, p) = nf_coord(&nv, &ni);
            (nf, proof::chain(&step, &p))
        }
        NF::VCons(c, t) => {
            let (nc, pc) = renorm(c);
            let (nt, pt) = renorm(t);
            (
                NF::VCons(Box::new(nc), Box::new(nt)),
                lift2("eq_tvcons", &pc, &pt),
            )
        }
        NF::VAdd(a, b) => {
            let (na, pa) = renorm(a);
            let (nb, pb) = renorm(b);
            let step = lift2("eq_tvadd", &pa, &pb);
            let (nf, p) = nf_vadd(&na, &nb);
            (nf, proof::chain(&step, &p))
        }
        NF::VScal(s, v) => {
            let (ns, ps) = renorm(s);
            let (nv, pv) = renorm(v);
            let step = lift2("eq_tvscal", &ps, &pv);
            let (nf, p) = nf_vscal(&ns, &nv);
            (nf, proof::chain(&step, &p))
        }
        NF::VTail(v) => {
            let (nv, pv) = renorm(v);
            let step = proof::lift("eq_ttail", &pv);
            let (nf, p) = nf_vtail(&nv);
            (nf, proof::chain(&step, &p))
        }
    }
}

/// Normalise a claim side from its stated form; the proof starts at the
/// claim's own spelling, which is what the certificate's goal says.
fn norm_side(side: &KSide) -> (NF, String) {
    renorm(&raw_side(side))
}

// ── the obvious layer and calculational composition ─────────────────────────
//
// The normaliser decides the congruence the *recursion* rules generate. Some
// equalities of the theory live outside it — `x + y = y + x` holds in every
// model yet is not derivable without commutativity — and a fragment that
// stopped there would refuse claims its own axioms state. So there is one
// more layer, and it is bounded on purpose:
//
// * the catalogue is the theory's *equality* laws the normaliser does not
//   apply: the Peano recursion offered again (so a restructured term can
//   peel), commutativity, associativity, distributivity, and the five
//   inner-product laws;
// * each law is tried at the root first (one step — the "obvious" inference),
//   then anywhere in the term, in chains of bounded depth, composed with
//   `eq_trans` and lifted through the term contexts ("relators": `eq_add_l`,
//   `eq_tinpr`, …) — calculational composition;
// * every candidate that meets is *replayed by mm0-rs*: the search proposes a
//   chain, the elaborator accepts or rejects it, and only an accepted one is
//   a certificate. The search's own bookkeeping is never the judge.

/// One law of the catalogue: a root rewrite, and whether the axiom may also
/// be used backwards (`eq_sym`).
struct Law {
    name: &'static str,
    fwd: fn(&NF) -> Option<NF>,
    rev: Option<fn(&NF) -> Option<NF>>,
}

fn law_add_z(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(a, b) if is_zero(b) => Some((**a).clone()),
        _ => None,
    }
}

fn law_add_s(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(a, b) => {
            let c = peel(b)?;
            Some(ts(NF::TAdd(Box::new((**a).clone()), Box::new(c))))
        }
        _ => None,
    }
}

fn law_mul_z(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(a, b) if is_zero(b) => {
            let _ = a;
            Some(numeral(0))
        }
        _ => None,
    }
}

fn law_mul_s(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(a, b) => {
            let c = peel(b)?;
            Some(NF::TAdd(
                Box::new(NF::TMul(Box::new((**a).clone()), Box::new(c))),
                Box::new((**a).clone()),
            ))
        }
        _ => None,
    }
}

fn law_add_comm(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(a, b) => Some(NF::TAdd(Box::new((**b).clone()), Box::new((**a).clone()))),
        _ => None,
    }
}

fn law_add_assoc(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(inner, c) => match inner.as_ref() {
            NF::TAdd(a, b) => Some(NF::TAdd(
                a.clone(),
                Box::new(NF::TAdd(b.clone(), c.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_add_assoc_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(a, inner) => match inner.as_ref() {
            NF::TAdd(b, c) => Some(NF::TAdd(
                Box::new(NF::TAdd(a.clone(), b.clone())),
                c.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_mul_comm(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(a, b) => Some(NF::TMul(Box::new((**b).clone()), Box::new((**a).clone()))),
        _ => None,
    }
}

fn law_mul_assoc(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(inner, c) => match inner.as_ref() {
            NF::TMul(a, b) => Some(NF::TMul(
                a.clone(),
                Box::new(NF::TMul(b.clone(), c.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_mul_assoc_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(a, inner) => match inner.as_ref() {
            NF::TMul(b, c) => Some(NF::TMul(
                Box::new(NF::TMul(a.clone(), b.clone())),
                c.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_distrib(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(a, inner) => match inner.as_ref() {
            NF::TAdd(b, c) => Some(NF::TAdd(
                Box::new(NF::TMul(a.clone(), b.clone())),
                Box::new(NF::TMul(a.clone(), c.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_distrib_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(p, q) => match (p.as_ref(), q.as_ref()) {
            (NF::TMul(a, b), NF::TMul(c, d)) if a == c => Some(NF::TMul(
                a.clone(),
                Box::new(NF::TAdd(b.clone(), d.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_sym(n: &NF) -> Option<NF> {
    match n {
        NF::Inpr(u, v) => Some(NF::Inpr(Box::new((**v).clone()), Box::new((**u).clone()))),
        _ => None,
    }
}

fn law_inpr_add_l(n: &NF) -> Option<NF> {
    match n {
        NF::Inpr(first, x) => match first.as_ref() {
            NF::VAdd(u, w) => Some(NF::TAdd(
                Box::new(NF::Inpr(u.clone(), x.clone())),
                Box::new(NF::Inpr(w.clone(), x.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_add_l_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(p, q) => match (p.as_ref(), q.as_ref()) {
            (NF::Inpr(u, x), NF::Inpr(w, y)) if x == y => Some(NF::Inpr(
                Box::new(NF::VAdd(u.clone(), w.clone())),
                x.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_add_r(n: &NF) -> Option<NF> {
    match n {
        NF::Inpr(u, second) => match second.as_ref() {
            NF::VAdd(w, x) => Some(NF::TAdd(
                Box::new(NF::Inpr(u.clone(), w.clone())),
                Box::new(NF::Inpr(u.clone(), x.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_add_r_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(p, q) => match (p.as_ref(), q.as_ref()) {
            (NF::Inpr(u, w), NF::Inpr(v, x)) if u == v => Some(NF::Inpr(
                u.clone(),
                Box::new(NF::VAdd(w.clone(), x.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_scal_l(n: &NF) -> Option<NF> {
    match n {
        NF::Inpr(first, x) => match first.as_ref() {
            NF::VScal(s, u) => Some(NF::TMul(
                s.clone(),
                Box::new(NF::Inpr(u.clone(), x.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_scal_l_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(s, second) => match second.as_ref() {
            NF::Inpr(u, x) => Some(NF::Inpr(
                Box::new(NF::VScal(s.clone(), u.clone())),
                x.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_scal_r(n: &NF) -> Option<NF> {
    match n {
        NF::Inpr(u, second) => match second.as_ref() {
            NF::VScal(s, w) => Some(NF::TMul(
                s.clone(),
                Box::new(NF::Inpr(u.clone(), w.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_inpr_scal_r_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(s, second) => match second.as_ref() {
            NF::Inpr(u, w) => Some(NF::Inpr(
                u.clone(),
                Box::new(NF::VScal(s.clone(), w.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

/// The catalogue, in the order candidates are offered: the recursion rules
/// first (they re-peek a restructured term), then the field laws, then the
/// inner-product laws. `eq_refl` is deliberately absent — a step that
/// rewrites nothing carries no information the normal forms have not.
static LAWS_BASE: &[Law] = &[
    Law {
        name: "add_z",
        fwd: law_add_z,
        rev: None,
    },
    Law {
        name: "add_s",
        fwd: law_add_s,
        rev: None,
    },
    Law {
        name: "mul_z",
        fwd: law_mul_z,
        rev: None,
    },
    Law {
        name: "mul_s",
        fwd: law_mul_s,
        rev: None,
    },
    Law {
        name: "add_comm",
        fwd: law_add_comm,
        rev: Some(law_add_comm),
    },
    Law {
        name: "add_assoc",
        fwd: law_add_assoc,
        rev: Some(law_add_assoc_rev),
    },
    Law {
        name: "mul_comm",
        fwd: law_mul_comm,
        rev: Some(law_mul_comm),
    },
    Law {
        name: "mul_assoc",
        fwd: law_mul_assoc,
        rev: Some(law_mul_assoc_rev),
    },
    Law {
        name: "distrib",
        fwd: law_distrib,
        rev: Some(law_distrib_rev),
    },
    Law {
        name: "inpr_sym",
        fwd: law_inpr_sym,
        rev: Some(law_inpr_sym),
    },
    Law {
        name: "inpr_add_l",
        fwd: law_inpr_add_l,
        rev: Some(law_inpr_add_l_rev),
    },
    Law {
        name: "inpr_add_r",
        fwd: law_inpr_add_r,
        rev: Some(law_inpr_add_r_rev),
    },
    Law {
        name: "inpr_scal_l",
        fwd: law_inpr_scal_l,
        rev: Some(law_inpr_scal_l_rev),
    },
    Law {
        name: "inpr_scal_r",
        fwd: law_inpr_scal_r,
        rev: Some(law_inpr_scal_r_rev),
    },
];

// ── the Kopperman/Mehler determination laws (`--mehler` only) ───────────────
//
// The four axioms of `mm0/mehler.mm0`, as obvious-layer laws: each rewrites a
// stuck coordinate fact into a determined one (and back, where the converse
// helps a meeting). They are *not* added to the normaliser — overlapping the
// recursion rules at variable indices would break the confluence the
// normal-form argument rests on — so they live only here, inside the bounded
// search, where every candidate is replayed by mm0-rs like any other.

fn law_coord_tail(n: &NF) -> Option<NF> {
    match n {
        NF::Coord(v, i) => match v.as_ref() {
            NF::VTail(u) => Some(NF::Coord(u.clone(), Box::new(NF::Ts(i.clone())))),
            _ => None,
        },
        _ => None,
    }
}

fn law_coord_tail_rev(n: &NF) -> Option<NF> {
    match n {
        NF::Coord(u, i) => match i.as_ref() {
            NF::Ts(inner) => Some(NF::Coord(Box::new(NF::VTail(u.clone())), inner.clone())),
            _ => None,
        },
        _ => None,
    }
}

fn law_vadd_coord(n: &NF) -> Option<NF> {
    match n {
        NF::Coord(v, i) => match v.as_ref() {
            NF::VAdd(a, b) => Some(NF::TAdd(
                Box::new(NF::Coord(a.clone(), i.clone())),
                Box::new(NF::Coord(b.clone(), i.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_vadd_coord_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TAdd(x, y) => match (x.as_ref(), y.as_ref()) {
            (NF::Coord(a, i), NF::Coord(b, j)) if i.spelling() == j.spelling() => Some(NF::Coord(
                Box::new(NF::VAdd(a.clone(), b.clone())),
                i.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_vscal_coord(n: &NF) -> Option<NF> {
    match n {
        NF::Coord(v, i) => match v.as_ref() {
            NF::VScal(s, u) => Some(NF::TMul(
                s.clone(),
                Box::new(NF::Coord(u.clone(), i.clone())),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn law_vscal_coord_rev(n: &NF) -> Option<NF> {
    match n {
        NF::TMul(s, c) => match c.as_ref() {
            NF::Coord(u, i) => Some(NF::Coord(
                Box::new(NF::VScal(s.clone(), u.clone())),
                i.clone(),
            )),
            _ => None,
        },
        _ => None,
    }
}

/// The unit of the head/tail decomposition: `tvcons (tcoord v tz) (ttail v)`
/// is `v`. Forward only — the reverse would eta-expand every variable in
/// every state, flooding the bounded search with syntactic variants of
/// nothing.
fn law_vec_eta(n: &NF) -> Option<NF> {
    match n {
        NF::VCons(c, t) => match (c.as_ref(), t.as_ref()) {
            (NF::Coord(v, i), NF::VTail(u))
                if i.spelling() == "tz" && u.spelling() == v.spelling() =>
            {
                Some((**v).clone())
            }
            _ => None,
        },
        _ => None,
    }
}

static LAWS_MEHLER: &[Law] = &[
    Law {
        name: "coord_tail",
        fwd: law_coord_tail,
        rev: Some(law_coord_tail_rev),
    },
    Law {
        name: "vadd_coord",
        fwd: law_vadd_coord,
        rev: Some(law_vadd_coord_rev),
    },
    Law {
        name: "vscal_coord",
        fwd: law_vscal_coord,
        rev: Some(law_vscal_coord_rev),
    },
    Law {
        name: "vec_eta",
        fwd: law_vec_eta,
        rev: None,
    },
];

/// The congruence axiom that lifts a child proof through its parent: the
/// "relator" of each context. Two-premise axioms take `(eq_refl)` in the
/// slot that does not change, positionally.
fn congruence(parent: &NF, idx: usize, p: &str) -> String {
    match parent {
        NF::TAdd(..) => {
            if idx == 0 {
                proof::lift("eq_add_l", p)
            } else {
                proof::lift("eq_add_r", p)
            }
        }
        NF::TMul(..) => {
            if idx == 0 {
                proof::lift("eq_mul_l", p)
            } else {
                proof::lift("eq_mul_r", p)
            }
        }
        NF::Ts(_) => proof::lift("eq_ts", p),
        NF::Inpr(..) => {
            if idx == 0 {
                lift2("eq_tinpr", p, "(eq_refl)")
            } else {
                lift2("eq_tinpr", "(eq_refl)", p)
            }
        }
        NF::Coord(..) => {
            if idx == 0 {
                lift2("eq_tcoord", p, "(eq_refl)")
            } else {
                lift2("eq_tcoord", "(eq_refl)", p)
            }
        }
        NF::VCons(..) => {
            if idx == 0 {
                lift2("eq_tvcons", p, "(eq_refl)")
            } else {
                lift2("eq_tvcons", "(eq_refl)", p)
            }
        }
        NF::VAdd(..) => {
            if idx == 0 {
                lift2("eq_tvadd", p, "(eq_refl)")
            } else {
                lift2("eq_tvadd", "(eq_refl)", p)
            }
        }
        NF::VScal(..) => {
            if idx == 0 {
                lift2("eq_tvscal", p, "(eq_refl)")
            } else {
                lift2("eq_tvscal", "(eq_refl)", p)
            }
        }
        NF::VTail(_) => proof::lift("eq_ttail", p),
        NF::Arith(_) | NF::VNil | NF::VVar(_) => {
            unreachable!("atoms have no children to lift through")
        }
    }
}

/// Lift a proof about a subterm up to a proof about the whole term: wrap it
/// once per edge of the path from the root.
fn lift_through(root: &NF, path: &[usize], p: &str) -> String {
    if path.is_empty() {
        return p.to_string();
    }
    let child = root.child(path[0]).expect("path within the term");
    let inner = lift_through(&child, &path[1..], p);
    congruence(root, path[0], &inner)
}

/// Apply a law at the node a path names, in the given orientation.
fn apply_at(n: &NF, path: &[usize], law: &Law, rev: bool) -> NF {
    if path.is_empty() {
        let step = if rev {
            law.rev.expect("orientation checked by caller")
        } else {
            law.fwd
        };
        return step(n).expect("path was found by a matching application");
    }
    let mut kids = n.children();
    let i = path[0];
    kids[i] = apply_at(&kids[i], &path[1..], law, rev);
    n.rebuild(kids)
}

/// Every (path, law, orientation) that rewrites something, in a bounded walk.
fn applications(
    n: &NF,
    out: &mut Vec<(Vec<usize>, &'static Law, bool)>,
    budget: &mut usize,
    laws: &[&'static Law],
) {
    if *budget == 0 {
        return;
    }
    for law in laws {
        if (law.fwd)(n).is_some() {
            out.push((Vec::new(), law, false));
            *budget -= 1;
            if *budget == 0 {
                return;
            }
        }
        if let Some(rev) = law.rev
            && rev(n).is_some()
        {
            out.push((Vec::new(), law, true));
            *budget -= 1;
            if *budget == 0 {
                return;
            }
        }
    }
    for (k, child) in n.children().iter().enumerate() {
        let mut sub = Vec::new();
        applications(child, &mut sub, budget, laws);
        for (mut path, law, rev) in sub {
            path.insert(0, k);
            out.push((path, law, rev));
        }
        if *budget == 0 {
            return;
        }
    }
}

/// Bounds of the calculational search. Deliberately small: the layer is a
/// bounded obvious-inference pass, not a theorem prover — a claim that needs
/// deeper search is left as a goal, which is the honest answer.
const MAX_DEPTH: usize = 2;
const MAX_STATES: usize = 96;
const MAX_APPLICATIONS: usize = 192;

/// One step of the search: a law at a path, then re-normalise (which is what
/// makes a restructured term peel again — `tmul (u·u) 2` becomes a sum, and
/// `add_z` can then reach inside it).
fn step(n: &NF, path: &[usize], law: &Law, rev: bool) -> (NF, String) {
    let rewritten = apply_at(n, path, law, rev);
    let mut p = format!("({})", law.name);
    if rev {
        p = format!("(eq_sym {p})");
    }
    let p = lift_through(n, path, &p);
    let (rn, rp) = renorm(&rewritten);
    (rn, proof::chain(&p, &rp))
}

/// The obvious layer: one law at the root of either side, re-normalised.
/// Candidates in catalogue order, each a proof of `teq lhsNF rhsNF`.
fn obvious(lhs: &NF, rhs: &NF, laws: &[&'static Law]) -> Vec<(String, String)> {
    let target = rhs.spelling();
    let mut out = Vec::new();
    for law in laws {
        for rev in [false, true] {
            let matches = if rev {
                law.rev.is_some_and(|f| f(lhs).is_some())
            } else {
                (law.fwd)(lhs).is_some()
            };
            if !matches {
                continue;
            }
            let (n, p) = step(lhs, &[], law, rev);
            if n.spelling() == target {
                out.push((
                    format!("{}{}", law.name, if rev { " (reversed)" } else { "" }),
                    p,
                ));
            }
        }
    }
    out
}

/// Calculational composition: bounded chains of catalogue laws, from both
/// sides, meeting at a common normal form. Returns the proof of
/// `teq lhsNF rhsNF` when they meet.
fn calculational(lhs: &NF, rhs: &NF, laws: &[&'static Law]) -> Option<String> {
    let target = rhs.spelling();
    if lhs.spelling() == target {
        return Some("(eq_refl)".to_string());
    }
    // spelling → (normal form, proof from this side's root)
    let mut from_lhs: HashMap<String, (NF, String)> = HashMap::new();
    let mut from_rhs: HashMap<String, (NF, String)> = HashMap::new();
    from_lhs.insert(lhs.spelling(), (lhs.clone(), String::new()));
    from_rhs.insert(rhs.spelling(), (rhs.clone(), String::new()));
    let mut frontier_l = vec![lhs.clone()];
    let mut frontier_r = vec![rhs.clone()];

    for _ in 0..MAX_DEPTH {
        // Expand whichever side is smaller — the meeting is symmetric, and
        // this spends the budget where the term structure has more room.
        if from_lhs.len() <= from_rhs.len() {
            if let Some(mid) = expand(&mut from_lhs, &mut frontier_l, &from_rhs, laws) {
                return Some(mid);
            }
            if let Some(mid) = expand_rev(&mut from_rhs, &mut frontier_r, &from_lhs, laws) {
                return Some(mid);
            }
        } else {
            if let Some(mid) = expand_rev(&mut from_rhs, &mut frontier_r, &from_lhs, laws) {
                return Some(mid);
            }
            if let Some(mid) = expand(&mut from_lhs, &mut frontier_l, &from_rhs, laws) {
                return Some(mid);
            }
        }
    }
    let _ = target;
    None
}

/// One round of expansion on the *left* side. Returns the assembled `teq
/// lhsNF rhsNF` chain when a new state meets the right side's map.
fn expand(
    states: &mut HashMap<String, (NF, String)>,
    frontier: &mut Vec<NF>,
    other: &HashMap<String, (NF, String)>,
    laws: &[&'static Law],
) -> Option<String> {
    meet(states, frontier, other, true, laws)
}

/// The same, expanding the right side: the chain is assembled the other way
/// round, because it is *their* proof that must end up first.
fn expand_rev(
    states: &mut HashMap<String, (NF, String)>,
    frontier: &mut Vec<NF>,
    other: &HashMap<String, (NF, String)>,
    laws: &[&'static Law],
) -> Option<String> {
    meet(states, frontier, other, false, laws)
}

fn meet(
    states: &mut HashMap<String, (NF, String)>,
    frontier: &mut Vec<NF>,
    other: &HashMap<String, (NF, String)>,
    mine_is_lhs: bool,
    laws: &[&'static Law],
) -> Option<String> {
    let mut next = Vec::new();
    'states: for n in frontier.drain(..) {
        let base = states
            .get(&n.spelling())
            .map(|(_, p)| p.clone())
            .unwrap_or_default();
        let mut apps = Vec::new();
        let mut budget = MAX_APPLICATIONS;
        applications(&n, &mut apps, &mut budget, laws);
        for (path, law, rev) in apps {
            let (rn, p) = step(&n, &path, law, rev);
            let spelled = rn.spelling();
            if let Some((_, other_proof)) = other.get(&spelled) {
                // Both sides reached M. lhs → M is `mine` on one call and
                // `other_proof` on the other; assemble by position, not by
                // luck.
                let mine = proof::chain(&base, &p);
                return Some(if mine_is_lhs {
                    proof::chain(&mine, &proof::sym(other_proof))
                } else {
                    proof::chain(other_proof, &proof::sym(&mine))
                });
            }
            if states.len() >= MAX_STATES {
                break 'states;
            }
            if states
                .insert(spelled, (rn.clone(), proof::chain(&base, &p)))
                .is_none()
            {
                next.push(rn);
            }
        }
    }
    *frontier = next;
    None
}

// ── certification ───────────────────────────────────────────────────────────

/// A `.mm0`/`.mm1` pair for one Kopperman claim: the full overlay theory,
/// then the claim. Same pairing `emit::file_text` writes, one theory longer.
fn file_text(statement: &str, theory: Theory) -> String {
    format!(
        "{}\n-- ── claim ──────────────────────────────────────────────────────────────\n{statement}\n",
        theory.text()
    )
}

/// The MM0 binders of a claim's variables, sorted and deduplicated — vector
/// and scalar variables alike, since MM0's `Term` is one sort.
fn binders(vars: &BTreeMap<String, Sort>) -> String {
    vars.keys()
        .map(|v| format!("({}: Term)", proof::param(v)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A refutation: from `teq L R`, the numeral on each side, and the two
/// evaluation proofs, peel the shared successors and let `z_ne_s` refuse the
/// residue — the arithmetic fragment's `Expr::refutes`, at the K2 level.
fn refute_from(left: &str, right: &str, a: u64, b: u64, hyp: &str) -> Result<String, String> {
    let through_hyp = proof::chain(&proof::sym(left), hyp);
    let into_numerals = proof::chain(&through_hyp, right);
    let lo = a.min(b);
    let mut p = if a < b {
        into_numerals
    } else {
        format!("(eq_sym {into_numerals})")
    };
    for _ in 0..lo {
        p = format!("(s_inj {p})");
    }
    Ok(format!("(z_ne_s {p})"))
}

/// The rules the LLM is offered for a Kopperman goal — the theory's own
/// vocabulary, by name, plus the *goal format* the seam actually sends
/// (`goal: $ teq L R $`, `v_`-prefixed variables, which constructors build
/// which side) so the model is answering the term it is shown. As in the
/// arithmetic fragment, whatever comes back is only ever a candidate:
/// `certify` hands each one to mm0-rs.
pub const LLM_SYSTEM: &str = "\
You are helping close a goal in the MM0 theory `kopperman` (over `logos_math`).
The goal arrives as `goal: $ teq L R $` — a proof that the terms L and R denote
the same thing. Variables carry a `v_` prefix (the name u appears as `v_u`), and
MM0 has one sort, so scalar and vector variables are both `Term`. A scalar side
is built from numerals, tadd, tmul, tinpr, tcoord; a vector side from tvnil,
tvcons, tvadd, tvscal.
Rules available: inpr_nil, inpr_cons, inpr_sym, inpr_add_l, inpr_add_r, inpr_scal_l,
inpr_scal_r, vadd_nil, vadd_cons_nil, vadd_cons_cons, vscal_nil, vscal_cons, coord_nil,
coord_cons_z, coord_cons_s, tail_nil, tail_cons, add_z, add_s, mul_z, mul_s, add_comm,
add_assoc, mul_comm, mul_assoc, distrib, eq_refl, eq_sym, eq_trans, eq_tinpr, eq_tcoord,
eq_tvcons, eq_tvadd, eq_tvscal, eq_ttail, eq_add_l, eq_add_r, eq_mul_l, eq_mul_r, eq_ts.
Terms: tz, ts, tadd, tmul, tinpr, tcoord, ttail, tvnil, tvcons, tvadd, tvscal, teq.
The order and sequence vocabularies are sealed (tle, tlt, tsqrt, and all of K3 —
tlimit, tsup, tcauchy), and inpr_pos takes a hypothesis binder: none of them can
close a bare teq, so never propose one.
Reply with ONLY a JSON array of MM0 proof terms (strings), most promising first.
A proof term looks like `(inpr_sym)` or `(eq_trans (inpr_cons) (eq_add_l (add_z)))`.";

/// The same catalogue under `--mehler`: the Kopperman/Mehler tail-determination
/// overlay is in effect, so its four rules join the list. Kept as a second
/// literal rather than assembled — the catalogue is data the tests check
/// against the theory either way, and a readable one beats a generated one.
pub const LLM_SYSTEM_MEHLER: &str = "\
You are helping close a goal in the MM0 theory `kopperman` (over `logos_math`),
with the Kopperman/Mehler tail-determination overlay (`--mehler`) in effect:
the random tail of a vector is determined, so coord_tail, vadd_coord, vscal_coord
and vec_eta are available alongside the base rules.
The goal arrives as `goal: $ teq L R $` — a proof that the terms L and R denote
the same thing. Variables carry a `v_` prefix (the name u appears as `v_u`), and
MM0 has one sort, so scalar and vector variables are both `Term`. A scalar side
is built from numerals, tadd, tmul, tinpr, tcoord; a vector side from tvnil,
tvcons, tvadd, tvscal.
Rules available: coord_tail, vadd_coord, vscal_coord, vec_eta, inpr_nil,
inpr_cons, inpr_sym, inpr_add_l, inpr_add_r, inpr_scal_l,
inpr_scal_r, vadd_nil, vadd_cons_nil, vadd_cons_cons, vscal_nil, vscal_cons, coord_nil,
coord_cons_z, coord_cons_s, tail_nil, tail_cons, add_z, add_s, mul_z, mul_s, add_comm,
add_assoc, mul_comm, mul_assoc, distrib, eq_refl, eq_sym, eq_trans, eq_tinpr, eq_tcoord,
eq_tvcons, eq_tvadd, eq_tvscal, eq_ttail, eq_add_l, eq_add_r, eq_mul_l, eq_mul_r, eq_ts.
Terms: tz, ts, tadd, tmul, tinpr, tcoord, ttail, tvnil, tvcons, tvadd, tvscal, teq.
The order and sequence vocabularies are sealed (tle, tlt, tsqrt, and all of K3 —
tlimit, tsup, tcauchy), and inpr_pos takes a hypothesis binder: none of them can
close a bare teq, so never propose one.
Reply with ONLY a JSON array of MM0 proof terms (strings), most promising first.
A proof term looks like `(coord_tail)` or `(eq_trans (inpr_cons) (eq_add_l (add_z)))`.";

/// How many candidate chains are replayed by mm0-rs before the claim is left
/// as a goal. A bound, not a search cutoff: each candidate is one elaboration,
/// and a sentence that needs hundreds is not a sentence of the fragment.
const MAX_ELABORATIONS: usize = 48;

/// Certify one Kopperman claim.
///
/// The verdict's shape, mirroring the arithmetic fragment: a ground scalar
/// claim is decided by evaluation (cross-checked against the DeltaNet
/// reducer on both expansions) and a false one refuted in the other
/// direction; anything else is closed by rewriting and then by the bounded
/// obvious layer, each candidate *replayed by mm0-rs*. When nothing closes
/// it, the certificate keeps both normal forms — the goal a human (or a
/// `--hint`) has to answer — and never a guess.
pub fn certify(claim: &KClaim, hints: &[String]) -> Result<Certificate, String> {
    certify_in(claim, hints, Theory::Base)
}

/// Certify one Kopperman claim against an explicit target theory — `--mehler`
/// selects [`Theory::Mehler`], whose certificates embed the tail-determination
/// axioms and may cite them. Same verdict shape as [`certify`].
pub fn certify_in(claim: &KClaim, hints: &[String], theory: Theory) -> Result<Certificate, String> {
    // Every artifact embeds the theory it was proved against: a proof citing
    // a `--mehler` axiom only elaborates against the mehler text.
    let embed = |statement: &str| file_text(statement, theory);
    let goal = claim.goal_of();
    let (nf_l, p_l) = norm_side(&claim.lhs);
    let (nf_r, p_r) = norm_side(&claim.rhs);

    // The theorem's binders, for every verdict: a claim with free variables
    // can still normalise to a numeral (u · tvnil computes to `tz`), so the
    // ground branch needs them as much as the open one does — an unbound
    // `v_u` in the statement is exactly what mm0-c refuses.
    let binders_txt = {
        let mut vars = claim.lhs.vars();
        for (k, s) in claim.rhs.vars() {
            vars.entry(k).or_insert(s);
        }
        binders(&vars)
    };
    let statement = |name: &str, refutation: bool| {
        let head = if binders_txt.is_empty() {
            format!("theorem {name}")
        } else {
            format!("theorem {name} {binders_txt}")
        };
        if refutation {
            format!("{head} (h: $ {goal} $): $ tfals $")
        } else {
            format!("{head}: $ {goal} $")
        }
    };

    // A ground scalar claim: both sides computed to a numeral, so evaluation
    // decides it — with the net reducer as the independent second machine.
    let ground = match (&claim.lhs, &claim.rhs) {
        (KSide::Scalar(a), KSide::Scalar(b)) => match (nf_l.ground_value(), nf_r.ground_value()) {
            (Some(x), Some(y)) => Some((a, b, x, y)),
            _ => None,
        },
        _ => None,
    };
    if let Some((sa, sb, a, b)) = ground {
        let refutation = a != b;
        // `True` is reserved for what evaluation decides *about the claim*:
        // a claim with a free variable is proved for all its instantiations
        // even when its normal form happens to compute (u · tvnil = 0), and
        // "proved (for all instantiations)" is the honest word for that.
        let closed = claim.lhs.vars().is_empty() && claim.rhs.vars().is_empty();
        let name = emit::theorem_name(&goal, refutation);
        let theorem = statement(&name, refutation);
        let proof_txt = if refutation {
            refute_from(&p_l, &p_r, a, b, "h")?
        } else {
            proof::chain(&p_l, &proof::sym(&p_r))
        };

        // The net cross-check: both sides expand to arithmetic, which the
        // existing lowering compiles and the reducer folds. A disagreement
        // with the mm0 chain is an error, not a note.
        let (net_witness, unf_hashes) = if refutation {
            (None, None)
        } else if let (Some(el), Some(er)) = (sa.expand(), sb.expand()) {
            let l = bridge::corroborate(&el, a)?;
            let r = bridge::corroborate(&er, b)?;
            let pairs = bridge::witness(&l, &r)?.ok_or_else(|| {
                format!(
                    "`{}` and `{}` both expand to {a}, but their reduced nets differ — the reducer and the arithmetic disagree",
                    sa.pretty(),
                    sb.pretty()
                )
            })?;
            (Some(pairs), Some((l.unf_hash.clone(), r.unf_hash.clone())))
        } else {
            (None, None)
        };

        return Ok(Certificate {
            claim: format!("{} equals {}", claim.lhs.pretty(), claim.rhs.pretty()),
            name,
            question: claim.question,
            verdict: if refutation {
                Verdict::False
            } else if closed {
                Verdict::True
            } else {
                Verdict::Proved
            },
            goal,
            proof: proof_txt.clone(),
            theorem: theorem.clone(),
            mm0: embed(&format!("{theorem};")),
            mm1: embed(&format!("pub {theorem} = '{proof_txt};")),
            net_witness,
            unf_hashes,
        });
    }

    // Structural join: the two normal forms spell the same term.
    let name = emit::theorem_name(&goal, false);
    let theorem = statement(&name, false);

    if nf_l.spelling() == nf_r.spelling() {
        let proof_txt = proof::chain(&p_l, &proof::sym(&p_r));
        return Ok(Certificate {
            claim: format!("{} equals {}", claim.lhs.pretty(), claim.rhs.pretty()),
            name,
            question: claim.question,
            verdict: Verdict::Proved,
            goal,
            proof: proof_txt.clone(),
            theorem: theorem.clone(),
            mm0: embed(&format!("{theorem};")),
            mm1: embed(&format!("pub {theorem} = '{proof_txt};")),
            net_witness: None,
            unf_hashes: None,
        });
    }

    // The bounded obvious layer: catalogue chains first (each is one
    // elaboration), then the hints, last so a proposal cannot shadow the
    // theory's own laws.
    let laws = theory.laws();
    let mut candidates: Vec<(String, String)> = obvious(&nf_l, &nf_r, &laws);
    if let Some(mid) = calculational(&nf_l, &nf_r, &laws)
        && !candidates.iter().any(|(_, p)| *p == mid)
    {
        candidates.push(("calculational chain".to_string(), mid));
    }
    // The proposal's *label is the proposal itself*, so a failure names it
    // in the unproved reason — the arithmetic fragment's
    // `hints_are_judged_not_believed` contract: what was tried is visible.
    for hint in hints {
        candidates.push((hint.clone(), hint.clone()));
    }

    let mut failures = Vec::new();
    let decl = format!("{theorem};");
    for (label, mid) in candidates.into_iter().take(MAX_ELABORATIONS) {
        let proof_txt = proof::join(&p_l, &mid, &p_r);
        let mm1 = embed(&format!("pub {theorem} = '{proof_txt};"));
        match emit::elaborates(&embed(&decl), &mm1)? {
            true => {
                return Ok(Certificate {
                    claim: format!("{} equals {}", claim.lhs.pretty(), claim.rhs.pretty()),
                    name,
                    question: claim.question,
                    verdict: Verdict::Proved,
                    goal,
                    proof: proof_txt,
                    theorem,
                    mm0: embed(&decl),
                    mm1,
                    net_witness: None,
                    unf_hashes: None,
                });
            }
            false => failures.push(label),
        }
    }

    let reason = format!(
        "the recursion laws do not close it: the left side normalises to `{}`, the right to `{}` \
         ({}). Closing that gap needs a rule this theory does not derive — induction is the usual \
         one, and Kopperman's Warning puts the sequence layer out of reach on purpose — so the \
         claim is left as a goal rather than guessed at",
        nf_l.spelling(),
        nf_r.spelling(),
        if failures.is_empty() {
            "no candidate was offered".to_string()
        } else {
            format!("no candidate typechecked: {}", failures.join(", "))
        }
    );
    Ok(Certificate {
        claim: format!("{} equals {}", claim.lhs.pretty(), claim.rhs.pretty()),
        name,
        question: claim.question,
        verdict: Verdict::Unproved(reason),
        goal,
        proof: String::new(),
        mm0: embed(&decl),
        mm1: embed(&format!("pub {theorem};")),
        theorem,
        net_witness: None,
        unf_hashes: None,
    })
}
