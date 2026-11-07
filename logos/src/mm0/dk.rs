//! `logos mm0 dk` — a Dedukti/Lambdapi (`dk`) front end for this MM0 back end.
//!
//! What it consumes: a `dk` signature (`symbol`, `rule`, `thm`, `def`, `type`),
//! the way a Dedukti file states an equational presentation and its facts. What
//! it produces: `logos_math.mm0` theorems with *explicit* proof terms, checked by
//! `mm0-rs` and re-checked by `mm0-c` — the same judge every other certificate in
//! this module answers to.
//!
//! # What is translated, and what the translation means
//!
//! Dedukti's `refl` is not a step: it is the *conversion* of both sides to a
//! common normal form by the file's rewrite rules. That is precisely the shape
//! the target theory can express — `add_z`/`add_s`/`mul_z`/`mul_s` are Peano
//! recursion, and [`crate::mm0::proof::Expr::normalize`] turns a conversion into
//! an explicit chain of those rules. So a `dk` theorem `:= refl` becomes an MM0
//! proof whose steps are the same rewrites Dedukti performs, and `mm0-c` checks
//! each one instead of trusting a reducer.
//!
//! Two consequences, stated rather than implied:
//!
//! * **The `dk` rules are re-proved, not assumed.** Every `rule L --> R` is
//!   emitted as its own MM0 theorem `teq L R`, so a rule the theory does not
//!   already justify is a translation error with a named reason, not a silent
//!   axiom.
//! * **This is not a λΠ kernel.** Dedukti's typing (dependent products,
//!   universes, `: Type`) is not checked here and cannot be: MM0 is first-order.
//!   The file's *statements* are translated and its *equalities* re-derived; a
//!   `thm` whose proof is not `refl` is refused by name, because a `trans`-shaped
//!   proof carries intermediate terms that only a type checker could supply.
//!
//! The fragment boundary is therefore a mapping, not a heuristic: `zero`, `succ`,
//! `plus`, `times` (at the term level) and `eq` (at the statement level) onto
//! `tz`, `ts`, `tadd`, `tmul`, `teq`. Anything else names itself in the error.
//!
//! # Term syntax
//!
//! Both of Dedukti's application spellings are accepted, because a real export
//! uses whichever its printer emits: juxtaposition (`plus x zero`), resolved
//! through the arities declared by `symbol`/`def`, and parenthesised arguments
//! (`plus(x, zero)`). Neither changes the meaning; the arity table is what makes
//! the first one unambiguous without a type checker.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::emit::{self, THEORY};
use super::proof::{Expr, chain, param, sym};

/// The symbols this front end maps, for diagnostics.
const FRAGMENT: &str = "`zero`, `succ`, `plus`, `times` (terms) and `eq` (statements)";

// ── the dk surface syntax ───────────────────────────────────────────────────

/// A term as written in the `dk` file: an application of a declared symbol, or
/// an identifier that is a variable (rules quantify their variables implicitly).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tm {
    Var(String),
    App(String, Vec<Tm>),
}

/// An equational proof term — the shape Dedukti writes after `:=`.
///
/// Only these four are translated, and each one is checked rather than trusted:
/// `refl` by conversion, a citation against the statement it names, `sym` by
/// swapping the goal, and `trans` by recovering its intermediate term (see
/// [`infer`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Proof {
    /// `refl`: the two sides of the goal are convertible.
    Refl,
    /// An earlier `thm` of the same file, cited by its `dk` name.
    Cite(String),
    /// `sym(p)`: `p : eq B A` proves `eq A B`.
    Sym(Box<Proof>),
    /// `trans(p, q)`: `p : eq A M`, `q : eq M B` prove `eq A B`.
    ///
    /// Dedukti gets `M` from the type checker. There is no type checker here, so
    /// `M` is the normal form of `A` — the only value both halves can meet in a
    /// fragment whose equality is decided by normalisation — and each half is
    /// then *checked* against it.
    Trans(Box<Proof>, Box<Proof>),
}

/// One `rule L --> R`.
#[derive(Debug, Clone)]
struct Rule {
    lhs: Tm,
    rhs: Tm,
}

/// One `thm name : type := proof`.
#[derive(Debug, Clone)]
struct Thm {
    name: String,
    /// The Π-bound names the statement quantifies (`x : nat -> …`).
    binders: Vec<String>,
    lhs: Tm,
    rhs: Tm,
    /// `None` means the file wrote no `:=`; that is treated as `refl`, which is
    /// what a Dedukti statement without a proof term would be asserting anyway.
    proof: Option<Proof>,
}

struct DkFile {
    rules: Vec<Rule>,
    theorems: Vec<Thm>,
}

/// A token stream with the two comment syntaxes Dedukti uses stripped.
struct Lexer {
    toks: Vec<String>,
    pos: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Lexer {
            toks: tokenize(src),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<&str> {
        self.toks.get(self.pos).map(String::as_str)
    }

    fn at(&self, t: &str) -> bool {
        self.peek() == Some(t)
    }

    fn eat(&mut self, t: &str) -> bool {
        if self.at(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &str) -> Result<(), String> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(match self.peek() {
                Some(found) => format!("expected `{t}`, found `{found}`"),
                None => format!("expected `{t}`, found the end of the file"),
            })
        }
    }

    fn word(&mut self) -> Result<String, String> {
        match self.peek() {
            Some(t) if !is_punct(t) => {
                let w = t.to_string();
                self.pos += 1;
                Ok(w)
            }
            Some(t) => Err(format!("expected an identifier, found `{t}`")),
            None => Err("expected an identifier, found the end of the file".into()),
        }
    }

    /// Everything up to the next `.` at paren depth 0, dot consumed. Used for
    /// declarations this front end records but does not translate (`symbol`'s
    /// type, `type`, `injective`).
    fn skip_to_dot(&mut self) -> Result<Vec<String>, String> {
        let mut depth = 0usize;
        let mut out = Vec::new();
        while let Some(t) = self.peek() {
            let t = t.to_string();
            self.pos += 1;
            match t.as_str() {
                "(" | "[" => depth += 1,
                ")" | "]" => depth = depth.saturating_sub(1),
                "." if depth == 0 => return Ok(out),
                _ => {}
            }
            out.push(t);
        }
        Err("expected `.` to end the statement".into())
    }

    /// A Π binder group, if one starts here: `x : nat ->` or `(x y : nat) ->`.
    ///
    /// The `(` form is recognised by finding its matching `)` and checking the
    /// token after it, so `eq (plus x zero) x` is not mistaken for a binder
    /// group just because it begins with a parenthesis.
    fn try_binders(&mut self) -> Result<Option<Vec<String>>, String> {
        match self.peek() {
            Some("(") => {
                let end = match self.matching_paren(self.pos) {
                    Some(end) => end,
                    None => return Ok(None),
                };
                let after = self.toks.get(end + 1).map(String::as_str);
                if after != Some("->") {
                    return Ok(None);
                }
                self.pos += 1; // (
                let mut names = Vec::new();
                loop {
                    if self.eat(":") {
                        break;
                    }
                    names.push(self.word()?);
                }
                if names.is_empty() {
                    return Err("an empty binder group `()`".into());
                }
                self.word()?; // the type, e.g. `nat`
                self.expect(")")?;
                self.expect("->")?;
                Ok(Some(names))
            }
            Some(t) if !is_punct(t) => {
                if self.toks.get(self.pos + 1).map(String::as_str) == Some(":") {
                    let name = self.word()?;
                    self.expect(":")?;
                    self.word()?; // the type
                    self.expect("->")?;
                    Ok(Some(vec![name]))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn matching_paren(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for (i, t) in self.toks.iter().enumerate().skip(open) {
            match t.as_str() {
                "(" => depth += 1,
                ")" => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// A term: `f a b` (arity from the declarations) or `f(a, b)`, or a variable.
    fn term(&mut self, arity: &BTreeMap<String, usize>) -> Result<Tm, String> {
        if self.eat("(") {
            let t = self.term(arity)?;
            self.expect(")")?;
            return Ok(t);
        }
        let head = self.word()?;
        let Some(&n) = arity.get(&head) else {
            if self.at("(") {
                return Err(format!(
                    "`{head}` is applied but never declared by a `symbol`/`def` statement, so its arity is unknown"
                ));
            }
            return Ok(Tm::Var(head));
        };
        let mut args = Vec::new();
        if self.eat("(") {
            if !self.eat(")") {
                loop {
                    args.push(self.term(arity)?);
                    if self.eat(",") {
                        continue;
                    }
                    self.expect(")")?;
                    break;
                }
            }
        } else {
            for _ in 0..n {
                args.push(self.term(arity)?);
            }
        }
        if args.len() != n {
            return Err(format!(
                "`{head}` takes {n} argument(s), but {} were given",
                args.len()
            ));
        }
        Ok(Tm::App(head, args))
    }

    /// A proof term: `refl`, a theorem name, `sym(p)`, or `trans(p, q)`.
    fn proof(&mut self) -> Result<Proof, String> {
        let head = self.word()?;
        let mut args = Vec::new();
        if self.eat("(") && !self.eat(")") {
            loop {
                args.push(self.proof()?);
                if self.eat(",") {
                    continue;
                }
                self.expect(")")?;
                break;
            }
        }
        match (head.as_str(), args.len()) {
            ("refl", 0) => Ok(Proof::Refl),
            ("sym", 1) => Ok(Proof::Sym(Box::new(args.remove(0)))),
            ("trans", 2) => {
                let second = args.remove(1);
                Ok(Proof::Trans(Box::new(args.remove(0)), Box::new(second)))
            }
            (name, 0) => Ok(Proof::Cite(name.to_string())),
            (name, n) => Err(format!(
                "`{name}` is applied to {n} argument(s): `sym` takes one and `trans` takes two; any other name is a citation and takes none"
            )),
        }
    }
}

fn is_punct(t: &str) -> bool {
    matches!(t, "(" | ")" | "[" | "]" | ":" | "." | "," | "->" | "-->" | ":=")
}

/// Split into tokens: identifiers, punctuation, and comments `(; … ;)` removed.
fn tokenize(src: &str) -> Vec<String> {
    let cs: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // Comments: `(; … ;)`, nestable, as Dedukti defines them.
        if c == '(' && cs.get(i + 1) == Some(&';') {
            let mut depth = 1usize;
            i += 2;
            while i < cs.len() && depth > 0 {
                if cs[i] == '(' && cs.get(i + 1) == Some(&';') {
                    depth += 1;
                    i += 2;
                } else if cs[i] == ';' && cs.get(i + 1) == Some(&')') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        match c {
            '(' | ')' | '[' | ']' | ':' | '.' | ',' => {
                // `:=` is one token, `:` alone otherwise.
                if c == ':' && cs.get(i + 1) == Some(&'=') {
                    out.push(":=".into());
                    i += 2;
                } else {
                    out.push(c.to_string());
                    i += 1;
                }
            }
            // `->` is one dash and `>`; `-->` is two dashes and `>`. Reading the
            // single-dash arrow as a pair of `-`/`>` tokens silently gave every
            // `symbol` an arity of 0, which then rejected every application.
            '-' if cs.get(i + 1) == Some(&'-') => {
                if cs.get(i + 2) == Some(&'>') {
                    out.push("-->".into());
                    i += 3;
                } else {
                    out.push("--".into());
                    i += 2;
                }
            }
            '-' if cs.get(i + 1) == Some(&'>') => {
                out.push("->".into());
                i += 2;
            }
            '-' => {
                out.push("-".into());
                i += 1;
            }
            _ if c.is_alphanumeric() || c == '_' || c == '\'' => {
                let start = i;
                while i < cs.len()
                    && (cs[i].is_alphanumeric() || cs[i] == '_' || cs[i] == '\'')
                {
                    i += 1;
                }
                out.push(cs[start..i].iter().collect());
            }
            other => {
                out.push(other.to_string());
                i += 1;
            }
        }
    }
    out
}

fn parse_file(src: &str) -> Result<DkFile, String> {
    let mut lex = Lexer::new(src);
    let mut arity: BTreeMap<String, usize> = BTreeMap::new();
    let mut rules = Vec::new();
    let mut theorems = Vec::new();

    while let Some(kw) = lex.peek().map(str::to_string) {
        match kw.as_str() {
            "symbol" | "def" => {
                lex.pos += 1;
                let name = lex.word()?;
                lex.expect(":")?;
                let ty = lex.skip_to_dot()?;
                arity.insert(name, ty.iter().filter(|t| *t == "->").count());
            }
            "type" | "injective" => {
                lex.pos += 1;
                lex.skip_to_dot()?;
            }
            "rule" => {
                lex.pos += 1;
                let lhs = lex.term(&arity)?;
                lex.expect("-->")?;
                let rhs = lex.term(&arity)?;
                lex.expect(".")?;
                rules.push(Rule { lhs, rhs });
            }
            "thm" => {
                lex.pos += 1;
                let name = lex.word()?;
                lex.expect(":")?;
                let mut binders = Vec::new();
                while let Some(names) = lex.try_binders()? {
                    binders.extend(names);
                }
                lex.expect("eq")?;
                let lhs = lex.term(&arity)?;
                let rhs = lex.term(&arity)?;
                let proof = if lex.eat(":=") {
                    let tree = lex.proof()?;
                    lex.expect(".")?;
                    Some(tree)
                } else {
                    lex.expect(".")?;
                    None
                };
                theorems.push(Thm {
                    name,
                    binders,
                    lhs,
                    rhs,
                    proof,
                });
            }
            other => {
                return Err(format!(
                    "unexpected `{other}` at the top level: this front end reads `symbol`, `rule`, `thm`, `def`, `type` and `injective`"
                ))
            }
        }
    }

    Ok(DkFile { rules, theorems })
}

// ── dk terms → the theory's terms ───────────────────────────────────────────

/// Map a `dk` term onto the theory's `Expr`, naming anything outside the
/// fragment instead of guessing at it.
fn map_term(t: &Tm) -> Result<Expr, String> {
    match t {
        Tm::Var(v) => Ok(Expr::Var(v.clone())),
        Tm::App(f, args) => match (f.as_str(), args.len()) {
            ("zero", 0) => Ok(Expr::Num(0)),
            ("succ", 1) => Ok(Expr::Suc(Box::new(map_term(&args[0])?))),
            ("plus", 2) => Ok(Expr::Add(
                Box::new(map_term(&args[0])?),
                Box::new(map_term(&args[1])?),
            )),
            ("times", 2) => Ok(Expr::Mul(
                Box::new(map_term(&args[0])?),
                Box::new(map_term(&args[1])?),
            )),
            (other, n) => Err(format!(
                "`{other}` (arity {n}) has no target in the theory: this front end maps {FRAGMENT} onto logos_math.mm0's `tz`/`ts`/`tadd`/`tmul`/`teq`. \
                 A different presentation can be added here (or the statement proved with `logos mm0 ask`), but it cannot be translated silently"
            )),
        },
    }
}

// ── translation ─────────────────────────────────────────────────────────────

/// One statement that came out of the file, with the proof term emitted for it.
#[derive(Debug, Clone)]
pub struct Checked {
    /// The MM0 theorem name (`dk_<sanitized>` for a `thm`, `dk_rule_<n>` for a
    /// rule — Dedukti rules are unnamed, so their position is their name).
    pub name: String,
    /// The statement in the file's own notation, for the report.
    pub statement: String,
    /// The MM0 formula proved.
    pub goal: String,
    /// The emitted MM0 proof term.
    pub proof: String,
}

/// A translated file: the two artifacts plus what they contain.
#[derive(Debug, Clone)]
pub struct Translation {
    pub mm0: String,
    pub mm1: String,
    pub rules: Vec<Checked>,
    pub theorems: Vec<Checked>,
}

/// Prove `lhs = rhs` at the Peano level by the theory's recursion laws.
///
/// `refl` in Dedukti means "both sides have the same normal form"; that is
/// exactly what this checks, and the returned term is the chain of rewrites that
/// establishes it. When the normal forms differ, the equality is outside the
/// congruence the theory's rules generate — induction is the usual missing rule —
/// and that is an error with both normal forms named, never an axiom.
fn prove_by_conversion(lhs: &Expr, rhs: &Expr) -> Result<String, String> {
    let (lhs_nf, lhs_proof) = lhs.normalize();
    let (rhs_nf, rhs_proof) = rhs.normalize();
    if lhs_nf != rhs_nf {
        return Err(format!(
            "`{}` and `{}` do not convert to a common normal form: they rewrite to `{}` and `{}`. \
             The theory's recursion laws decide exactly the congruence they generate, so this equality \
             is not in it (the missing rule is usually induction)",
            lhs.pretty(),
            rhs.pretty(),
            lhs_nf.pretty(),
            rhs_nf.pretty()
        ));
    }
    Ok(chain(&lhs_proof, &sym(&rhs_proof)))
}

fn goal_formula(lhs: &Expr, rhs: &Expr) -> String {
    format!("teq {} {}", lhs.peano(), rhs.peano())
}

/// First-order matching: bind `pattern`'s variables against `actual`.
///
/// A cited theorem is stated over variables and the goal fixes their values;
/// MM0's elaborator performs the same unification when it accepts `(dk_name)`
/// with no arguments, so this check predicts exactly what the kernel will do.
fn match_pattern(pattern: &Expr, actual: &Expr, bound: &mut BTreeMap<String, Expr>) -> bool {
    match pattern {
        Expr::Var(v) => match bound.get(v) {
            Some(previous) => previous == actual,
            None => {
                bound.insert(v.clone(), actual.clone());
                true
            }
        },
        Expr::Num(n) => matches!(actual, Expr::Num(m) if m == n),
        Expr::Suc(p) => matches!(actual, Expr::Suc(a) if match_pattern(p, a, bound)),
        Expr::Add(p, q) => match actual {
            Expr::Add(a, b) => match_pattern(p, a, bound) && match_pattern(q, b, bound),
            _ => false,
        },
        Expr::Mul(p, q) => match actual {
            Expr::Mul(a, b) => match_pattern(p, a, bound) && match_pattern(q, b, bound),
            _ => false,
        },
    }
}

/// Translate one proof term against the statement it has to prove.
///
/// `want` is fixed by the enclosing statement and, inside a `trans`, by the
/// middle term recovered here. Every leaf is checked — `refl` by conversion, a
/// citation against its own statement — so the returned term proves
/// `teq want.0 want.1` and nothing else.
fn infer(
    proof: &Proof,
    want: (&Expr, &Expr),
    cited: &BTreeMap<String, (Expr, Expr, String)>,
    depth: usize,
) -> Result<String, String> {
    let indent = |what: &str| format!("{}{what}", "  ".repeat(depth));
    match proof {
        Proof::Refl => prove_by_conversion(want.0, want.1),
        Proof::Cite(name) => {
            let (pattern_l, pattern_r, mm0) = cited.get(name).ok_or_else(|| {
                indent(&format!(
                    "`{name}` is cited but not declared earlier in the file; a citation must name a previous `thm`"
                ))
            })?;
            let mut bound = BTreeMap::new();
            if match_pattern(pattern_l, want.0, &mut bound)
                && match_pattern(pattern_r, want.1, &mut bound)
            {
                return Ok(format!("({mm0})"));
            }
            let mut bound = BTreeMap::new();
            if match_pattern(pattern_l, want.1, &mut bound)
                && match_pattern(pattern_r, want.0, &mut bound)
            {
                return Ok(format!("(eq_sym ({mm0}))"));
            }
            Err(indent(&format!(
                "`{name}` proves `{}` = `{}`, which is not `{}` = `{}`",
                pattern_l.pretty(),
                pattern_r.pretty(),
                want.0.pretty(),
                want.1.pretty()
            )))
        }
        Proof::Sym(inner) => {
            let inner = infer(inner, (want.1, want.0), cited, depth + 1)?;
            Ok(format!("(eq_sym {inner})"))
        }
        Proof::Trans(first, second) => {
            // The middle term: normalising the left side is the only value both
            // halves can meet, because this fragment decides equality by normal
            // forms. Each half is then checked against it, so a wrong middle is
            // an error rather than a gap.
            let (middle, _) = want.0.normalize();
            let left = infer(first, (want.0, &middle), cited, depth + 1)
                .map_err(|e| format!("the first half of `trans` does not prove its step: {e}"))?;
            let right = infer(second, (&middle, want.1), cited, depth + 1)
                .map_err(|e| format!("the second half of `trans` does not prove its step: {e}"))?;
            Ok(chain(&left, &right))
        }
    }
}

/// A theorem name MM0 will accept, derived from the `dk` name.
fn theorem_name(prefix: &str, name: &str) -> String {
    let mut out = String::from(prefix);
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out
}

fn statement(name: &str, binders: &[String], goal: &str) -> String {
    let binders = if binders.is_empty() {
        String::new()
    } else {
        format!(" {}", binders.join(" "))
    };
    format!("theorem {name}{binders}: $ {goal} $")
}

/// The variables a translated statement actually mentions, sorted.
///
/// A rule's variables are implicit in `dk`, but they are *not* implicit to
/// MM0: an unbound `v_x` in a statement is a parse error at that theorem, which
/// is how the missing binder on rules was found — the proof checked, the
/// statement did not.
fn used_vars(lhs: &Expr, rhs: &Expr) -> Vec<String> {
    let mut used = lhs.vars();
    for v in rhs.vars() {
        if !used.contains(&v) {
            used.push(v);
        }
    }
    used.sort();
    used
}

/// MM0 binder groups for these variables, in the theory's one sort.
fn binder_text(names: &[String]) -> Vec<String> {
    names
        .iter()
        .map(|v| format!("({}: Term)", param(v)))
        .collect()
}

/// Translate a whole `dk` file. Pure: no tools, no filesystem.
pub fn translate(src: &str) -> Result<Translation, String> {
    let file = parse_file(src)?;
    if file.rules.is_empty() && file.theorems.is_empty() {
        return Err("the file states no `rule` and no `thm`, so there is nothing to translate".into());
    }

    let mut mm0 = String::from(THEORY);
    let mut mm1 = String::from(THEORY);
    mm0.push_str("\n-- ── translated from dk ──────────────────────────────────────────────\n");
    mm1.push_str("\n-- ── translated from dk ──────────────────────────────────────────────\n");

    let mut rules = Vec::new();
    for (i, rule) in file.rules.iter().enumerate() {
        let lhs = map_term(&rule.lhs)?;
        let rhs = map_term(&rule.rhs)?;
        let name = format!("dk_rule_{}", i + 1);
        let proof = prove_by_conversion(&lhs, &rhs).map_err(|e| {
            format!(
                "rule {} (`{}` --> `{}`) is not a law of the target theory: {e}",
                i + 1,
                lhs.pretty(),
                rhs.pretty()
            )
        })?;
        let goal = goal_formula(&lhs, &rhs);
        let stmt = statement(&name, &binder_text(&used_vars(&lhs, &rhs)), &goal);
        mm0.push_str(&format!("{stmt};\n"));
        mm1.push_str(&format!("pub {stmt} = '{proof};\n"));
        rules.push(Checked {
            name,
            statement: format!("{} --> {}", lhs.pretty(), rhs.pretty()),
            goal,
            proof,
        });
    }

    // The statements translated so far, for `Cite`: a proof may name an earlier
    // `thm` of the same file, and the citation is checked against that
    // theorem's own statement (MM0 infers the instantiation).
    let mut cited: BTreeMap<String, (Expr, Expr, String)> = BTreeMap::new();
    let mut theorems = Vec::new();
    for thm in &file.theorems {
        let lhs = map_term(&thm.lhs)?;
        let rhs = map_term(&thm.rhs)?;
        // The Π-bound names must be exactly the variables the statement uses:
        // a binder the body never mentions would make the emitted theorem say
        // something the file does not.
        let used = used_vars(&lhs, &rhs);
        let mut declared = thm.binders.clone();
        declared.sort();
        if declared != used {
            return Err(format!(
                "`thm {}` binds {} but the statement uses {}",
                thm.name,
                if declared.is_empty() {
                    "nothing".to_string()
                } else {
                    declared.join(", ")
                },
                if used.is_empty() {
                    "no variables".to_string()
                } else {
                    used.join(", ")
                }
            ));
        }

        let tree = thm.proof.clone().unwrap_or(Proof::Refl);
        let proof = infer(&tree, (&lhs, &rhs), &cited, 0)
            .map_err(|e| format!("`thm {}` cannot be certified: {e}", thm.name))?;
        let name = theorem_name("dk_", &thm.name);
        cited.insert(thm.name.clone(), (lhs.clone(), rhs.clone(), name.clone()));
        let binders = binder_text(&used);
        let goal = goal_formula(&lhs, &rhs);
        let stmt = statement(&name, &binders, &goal);
        mm0.push_str(&format!("{stmt};\n"));
        mm1.push_str(&format!("pub {stmt} = '{proof};\n"));
        theorems.push(Checked {
            name,
            statement: format!("{} = {}", lhs.pretty(), rhs.pretty()),
            goal,
            proof,
        });
    }

    Ok(Translation {
        mm0,
        mm1,
        rules,
        theorems,
    })
}

// ── running the tools ───────────────────────────────────────────────────────

/// What the compiler and the C verifier said about a translated file.
#[derive(Debug, Clone)]
pub struct Verified {
    pub mmb: PathBuf,
    pub sorts: Option<String>,
    /// `mm0-c`'s exit status; `None` when it is not installed here.
    pub mm0_c_exit: Option<i32>,
}

/// Translate a file, write `dk.mm0`/`dk.mm1`, and have `mm0-rs` compile and
/// `mm0-c` re-check them. Nothing is reported as certified unless both agree.
pub fn run(path: &Path, out: Option<&Path>) -> Result<(Translation, Verified), String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let translation = translate(&src)?;

    let dir = match out {
        Some(dir) => dir.to_path_buf(),
        None => std::env::temp_dir().join(format!("logos-dk-{}", std::process::id())),
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stem = dir.join(path.file_stem().and_then(|s| s.to_str()).unwrap_or("dk"));
    let mm0_path = stem.with_extension("mm0");
    let mm1_path = stem.with_extension("mm1");
    std::fs::write(&mm0_path, &translation.mm0)
        .map_err(|e| format!("{}: {e}", mm0_path.display()))?;
    std::fs::write(&mm1_path, &translation.mm1)
        .map_err(|e| format!("{}: {e}", mm1_path.display()))?;

    let verified = emit::verify_mm0(&mm0_path)?;
    Ok((
        translation,
        Verified {
            mmb: verified.mmb,
            sorts: verified.sorts,
            mm0_c_exit: verified.mm0_c_exit,
        },
    ))
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ARITH: &str = r#"
(; a dk presentation of the arithmetic fragment ;)
symbol zero : nat.
symbol succ : nat -> nat.
symbol plus : nat -> nat -> nat.
symbol times : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.

rule plus(x, zero) --> x.
rule plus(x, succ(y)) --> succ(plus(x, y)).
rule times(x, zero) --> zero.
rule times(x, succ(y)) --> plus(times(x, y), x).

thm two_plus_three : eq plus(succ(succ(zero)), succ(succ(succ(zero)))) succ(succ(succ(succ(succ(zero))))) := refl.
thm plus_zero_right : x : nat -> eq plus(x, zero) x := refl.
thm times_zero : (x : nat) -> eq times(x, zero) zero := refl.
"#;

    #[test]
    fn translates_rules_and_theorems() {
        let t = translate(ARITH).expect("translates");
        assert_eq!(t.rules.len(), 4);
        assert_eq!(t.theorems.len(), 3);
        // Every rule and theorem is a `pub theorem` in the .mm1, and a plain
        // `theorem` in the .mm0 — the shape mm0-c reads.
        assert!(t.mm1.contains("pub theorem dk_rule_1"));
        assert!(t.mm1.contains("pub theorem dk_plus_zero_right"));
        assert!(t.mm0.contains("theorem dk_plus_zero_right (v_x: Term): $ teq (tadd v_x tz) v_x $;"));
        assert!(t.mm1.contains("= '(add_z);"), "{}", t.mm1);
        // The Peano spelling, not the net one: a dk term *is* a Peano term, so no
        // emitted claim mentions the DeltaNet agent layer. (The theory text does,
        // because its bridge axioms are about the agents.)
        assert!(t.mm1.contains("$ teq (tadd v_x tz) v_x $"));
        assert!(
            !t.mm1.contains("pub theorem dk_rule_1: $ teq (tapp"),
            "{}",
            t.mm1
        );
    }

    #[test]
    fn refuses_a_true_fact_the_laws_do_not_generate() {
        let src = r#"
symbol zero : nat.
symbol succ : nat -> nat.
symbol plus : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.
rule plus(x, zero) --> x.
rule plus(x, succ(y)) --> succ(plus(x, y)).
thm comm : x : nat -> y : nat -> eq plus(x, y) plus(y, x) := refl.
"#;
        let err = translate(src).expect_err("commutativity is not a conversion");
        assert!(err.contains("do not convert to a common normal form"), "{err}");
        assert!(err.contains("induction"), "{err}");
    }

    #[test]
    fn refuses_an_unmapped_symbol() {
        let src = r#"
symbol zero : nat.
symbol minus : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.
thm t : eq minus(zero, zero) zero := refl.
"#;
        let err = translate(src).expect_err("minus is outside the fragment");
        assert!(err.contains("has no target in the theory"), "{err}");
    }

    #[test]
    fn translates_sym_and_trans_and_citations() {
        let src = r#"
symbol zero : nat.
symbol succ : nat -> nat.
symbol plus : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.
rule plus(x, zero) --> x.
thm base : (x : nat) -> eq plus(x, zero) x := refl.
thm cited : eq plus(succ(zero), zero) succ(zero) := base.
thm flipped : eq succ(zero) plus(succ(zero), zero) := sym(cited).
thm composed : eq plus(succ(zero), zero) succ(zero) := trans(cited, refl).
thm via_refl_only : eq plus(succ(zero), zero) succ(zero) := trans(refl, refl).
"#;
        let t = translate(src).expect("translates");
        let proof_of = |name: &str| -> String {
            t.theorems
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("no theorem {name}"))
                .proof
                .clone()
        };
        // A citation is bare: MM0 infers the instantiation, exactly as the
        // first-order match above predicted it would.
        assert_eq!(proof_of("dk_cited"), "(dk_base)");
        assert_eq!(proof_of("dk_flipped"), "(eq_sym (dk_cited))");
        assert!(proof_of("dk_composed").contains("(dk_cited)"));
        // `trans(refl, refl)` recovers its middle term by normalising the left
        // side, so both halves are conversion steps that meet there.
        assert!(proof_of("dk_via_refl_only").contains("(add_z)"));
    }

    #[test]
    fn refuses_a_citation_of_the_wrong_statement() {
        let src = r#"
symbol zero : nat.
symbol succ : nat -> nat.
symbol plus : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.
rule plus(x, zero) --> x.
thm a : eq plus(zero, zero) zero := refl.
thm b : eq plus(zero, zero) succ(zero) := a.
"#;
        let err = translate(src).expect_err("`a` proves a different equation");
        assert!(err.contains("which is not"), "{err}");
    }

    #[test]
    fn refuses_an_undeclared_citation() {
        let src = r#"
symbol zero : nat.
symbol eq : nat -> nat -> Prop.
thm t : eq zero zero := nowhere.
"#;
        let err = translate(src).expect_err("no such theorem");
        assert!(err.contains("cited but not declared"), "{err}");
    }

    #[test]
    fn refuses_a_binder_the_statement_does_not_use() {
        let src = r#"
symbol zero : nat.
symbol eq : nat -> nat -> Prop.
thm t : x : nat -> eq zero zero := refl.
"#;
        let err = translate(src).expect_err("the binder is unused");
        assert!(err.contains("binds"), "{err}");
    }

    #[test]
    fn refuses_a_rule_the_theory_does_not_have() {
        let src = r#"
symbol zero : nat.
symbol succ : nat -> nat.
symbol plus : nat -> nat -> nat.
symbol eq : nat -> nat -> Prop.
rule plus(zero, x) --> x.
thm t : eq plus(zero, zero) zero := refl.
"#;
        let err = translate(src).expect_err("add is recursed on the right operand");
        assert!(err.contains("is not a law of the target theory"), "{err}");
    }
}
