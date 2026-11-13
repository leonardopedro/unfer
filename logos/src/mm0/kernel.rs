//! The kernel fragment: sentences about a *symbolic kernel certificate*.
//!
//! The arithmetic and Kopperman fragments certify sentences against their own
//! theories: the CNL parses a term, the prover builds a proof, `mm0-c` checks
//! it. This fragment does something else on purpose, because the thing it talks
//! about was proved elsewhere.
//!
//! A kernel certificate (`mm0/<stem>_certificate.mm0`, emitted by `au2mm0`)
//! states a kernel's obligations as identities over the free commutative ring on
//! the state's components, and closes each with a `peq` chain. Those statements
//! are the *content* the earlier fragments cannot express: they range over state
//! vectors, they carry hypotheses (`l = r` for a uniform interface), and their
//! subject is a numerical scheme rather than a number.
//!
//! So the kernel fragment decides by **citation**: it lowers a sentence to the
//! same polynomials the certificate used, and a sentence is proved when some
//! theorem of that certificate states exactly the same identity, up to a renaming
//! of the variables. Nothing here invents a proof term. The claim that the cited
//! theorem is true is the claim that `mm0-rs` + `mm0-c` accepted the certificate,
//! which `australVM/scripts/verify-invariants` re-runs on every gate — and which
//! this module re-checks cheaply at load time, by confirming that the certificate
//! file it was given actually *pins* the obligations file it is reading (sha256
//! in the `-- Source:` header). A sidecar that drifted from its certificate
//! therefore cannot be cited, which is the failure mode a lookup table would
//! have.
//!
//! What the fragment will not do: decide that a sentence is false. A polynomial
//! identity is false for some state and true for others, and turning "not an
//! identity" into "false" would be a semantic claim about the numeric domain
//! that this layer has no proof of. Those sentences come back `unproved` with a
//! reason, like every other thing the fragment cannot reach.
//!
//! Grammar (in the spirit of [`super::cnl`]: one reading per sentence):
//!
//! ```text
//! claim  := ["for all" var {"," var} ":"] body ["when" expr eq expr]
//!         | "the" "scheme" "is" quality          -- a property, never an identity
//! body   := expr eq expr
//! eq     := "=" | "equals" | "equal" ["to"] | "is" ["equal" ["to"]]
//!         | "agrees" ["with"]
//! expr   := term {("plus" | "+") term}
//! term   := atom {("times" | "*") atom}
//! atom   := number | var | "(" expr ")"
//!         | defname ["of"] atom {"and" atom}     -- exactly as many as the def's arity
//! ```
//!
//! `defname` is a definition in the obligations sidecar (`flux`, `numFlux` →
//! also `numerical flux`, `delta`, …), so the vocabulary of the fragment is read
//! from the kernel rather than hard-coded here: a kernel that adds a definition
//! adds a word.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::cnl::{number_value, tokenize};
use super::emit::Verdict;

/// A polynomial in the state's components: a monomial (the sorted, repeated
/// component indices of its factors) mapped to its integer coefficient, with no
/// zero entries. Canonical: two expressions denote the same polynomial exactly
/// when the maps are equal, which is what makes "the same identity" decidable
/// here without a prover.
type Poly = BTreeMap<Vec<u32>, i64>;

/// A term of the kernel fragment. `Call` names a definition from the sidecar;
/// its meaning is the definition's body, not a term of the emitted theory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KTerm {
    Var(String),
    Int(i64),
    Neg(Box<KTerm>),
    Add(Box<KTerm>, Box<KTerm>),
    Mul(Box<KTerm>, Box<KTerm>),
    Call { name: String, args: Vec<KTerm> },
}

impl KTerm {
    fn collect_vars(&self, out: &mut Vec<String>) {
        match self {
            KTerm::Var(v) => {
                if !out.contains(v) {
                    out.push(v.clone());
                }
            }
            KTerm::Int(_) => {}
            KTerm::Neg(a) => a.collect_vars(out),
            KTerm::Add(a, b) | KTerm::Mul(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            KTerm::Call { args, .. } => {
                for a in args {
                    a.collect_vars(out);
                }
            }
        }
    }
}

/// One sentence: the two sides, the optional hypothesis, and the variables in
/// the order they were fixed (the quantifier's order, then first appearance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KClaim {
    pub lhs: KTerm,
    pub rhs: KTerm,
    pub hypothesis: Option<(KTerm, KTerm)>,
    pub vars: Vec<String>,
}

/// A parsed kernel sentence: an identity, or a property of the scheme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sentence {
    Identity(KClaim),
    /// `the scheme is second order accurate` — the fragment's boundary. The
    /// sentence parses, so it can be recorded with an expected verdict, and it
    /// is never proved: a quality of a scheme is not an identity in ℤ[x⃗], and
    /// nothing in the certificate states one.
    Quality(String),
}

/// The words a definition can be called by, longest first so that
/// `numerical flux` wins over `flux`.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    entries: Vec<(Vec<String>, String, usize)>,
}

impl Vocabulary {
    fn new(defs: &[Def]) -> Vocabulary {
        let mut entries: Vec<(Vec<String>, String, usize)> = Vec::new();
        for d in defs {
            let lower = d.name.to_lowercase();
            entries.push((vec![lower.clone()], d.name.clone(), d.params.len()));
            // `numerical flux` for `numFlux`: the alias table is the one place a
            // second spelling is admitted, and it names the definition it stands
            // for, so an alias that stops existing is a load-time error.
            for (alias, target) in ALIASES {
                if target.eq_ignore_ascii_case(&d.name) {
                    entries.push((
                        alias.iter().map(|w| w.to_lowercase()).collect(),
                        d.name.clone(),
                        d.params.len(),
                    ));
                }
            }
        }
        // Longest phrase first: a two-word name must be tried before its head.
        entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Vocabulary { entries }
    }

    fn lookup(&self, tokens: &[String], pos: usize) -> Option<(&str, usize, usize)> {
        self.entries.iter().find_map(|(phrase, name, arity)| {
            let fits = pos + phrase.len() <= tokens.len()
                && phrase
                    .iter()
                    .enumerate()
                    .all(|(k, w)| tokens[pos + k] == *w);
            fits.then(|| (name.as_str(), *arity, phrase.len()))
        })
    }
}

/// The human spellings of a definition whose CamelCase name is not a word.
const ALIASES: &[(&[&str], &str)] = &[(&["numerical", "flux"], "numFlux")];

/// The parse is driven by the kernel's own definitions, so the vocabulary
/// arrives as data.
struct P<'a> {
    tokens: Vec<String>,
    pos: usize,
    vocab: &'a Vocabulary,
}

impl P<'_> {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    fn eat(&mut self, word: &str) -> bool {
        if self.peek() == Some(word) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// The equality words, in the order a reader meets them. `agrees with` is
    /// here because the sentence a kernel obligation makes in English is
    /// "the numerical flux agrees with the physical flux", and a fragment that
    /// could not say `agrees` could not say that sentence.
    fn eat_eq(&mut self) -> bool {
        if self.eat("=") || self.eat("equals") {
            true
        } else if self.eat("equal") {
            self.eat("to");
            true
        } else if self.eat("agrees") {
            self.eat("with");
            true
        } else if self.eat("is") {
            self.eat("equal");
            self.eat("to");
            true
        } else {
            false
        }
    }

    fn parse_sum(&mut self) -> Result<KTerm, String> {
        let mut e = self.parse_product()?;
        while self.peek() == Some("plus") || self.peek() == Some("+") {
            self.pos += 1;
            let r = self.parse_product()?;
            e = KTerm::Add(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_product(&mut self) -> Result<KTerm, String> {
        let mut e = self.parse_atom()?;
        while self.peek() == Some("times") || self.peek() == Some("*") {
            self.pos += 1;
            let r = self.parse_atom()?;
            e = KTerm::Mul(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<KTerm, String> {
        let Some(token) = self.peek().map(str::to_string) else {
            return Err("expected a number, a variable or a kernel definition, found the end of the sentence".into());
        };

        if let Some((name, arity, width)) = self.vocab.lookup(&self.tokens, self.pos) {
            let name = name.to_string();
            self.pos += width;
            self.eat("of");
            let mut args = Vec::new();
            for k in 0..arity {
                if k > 0 && !self.eat("and") {
                    return Err(format!(
                        "expected `and` between the arguments of `{name}`, found {}",
                        self.peek().unwrap_or("the end of the sentence")
                    ));
                }
                args.push(self.parse_atom()?);
            }
            if args.is_empty() {
                return Err(format!("`{name}` takes no arguments, so it is not a term"));
            }
            return Ok(KTerm::Call { name, args });
        }

        if token == "(" {
            self.pos += 1;
            let e = self.parse_sum()?;
            if !self.eat(")") {
                return Err(format!(
                    "expected `)`, found {}",
                    self.peek().unwrap_or("the end of the sentence")
                ));
            }
            return Ok(e);
        }
        if token == "-" {
            self.pos += 1;
            return Ok(KTerm::Neg(Box::new(self.parse_atom()?)));
        }
        if let Some(n) = number_value(&token) {
            self.pos += 1;
            let mut n = i64::try_from(n).map_err(|_| format!("`{token}` is too large"))?;
            if let Some(next) = self.peek().and_then(number_value)
                && n % 10 == 0
                && n >= 20
                && (1..=9).contains(&next)
            {
                self.pos += 1;
                n += i64::try_from(next).map_err(|_| format!("`{token}` is too large"))?;
            }
            return Ok(KTerm::Int(n));
        }
        if token.chars().next().is_some_and(|c| c.is_alphabetic()) {
            self.pos += 1;
            // `vorticity of u` reads like a definition and is not one: naming
            // the word is more useful than the `unexpected of` it would
            // otherwise become.
            if self.peek() == Some("of") {
                return Err(format!(
                    "`{token}` is not a definition of this kernel (have: {})",
                    self.vocab
                        .entries
                        .iter()
                        .map(|(_, name, _)| name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            return Ok(KTerm::Var(token));
        }
        Err(format!(
            "expected a number, a variable or a kernel definition, found `{token}`"
        ))
    }
}

/// Parse one sentence of the kernel fragment.
///
/// The parse refuses what it cannot read, naming the token, for the same reason
/// [`super::cnl`] does: a sentence that *nearly* parses would be compared
/// against the certificate as if it had been understood.
pub fn parse(text: &str, vocab: &Vocabulary) -> Result<Sentence, String> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Err("empty sentence".into());
    }
    let mut p = P {
        tokens,
        pos: 0,
        vocab,
    };

    // A property of the scheme, not an identity: recognised before the term
    // grammar so the boundary is stated rather than a parse error.
    if p.tokens.first().map(String::as_str) == Some("the")
        && p.tokens.get(1).map(String::as_str) == Some("scheme")
        && p.tokens.get(2).map(String::as_str) == Some("is")
    {
        let quality = p.tokens[3..].join(" ");
        if quality.is_empty() {
            return Err("`the scheme is` needs a property after it".into());
        }
        return Ok(Sentence::Quality(quality));
    }

    let mut declared: Vec<String> = Vec::new();
    if p.eat("for") {
        if !(p.eat("all") || p.eat("every") || p.eat("each")) {
            return Err(format!(
                "expected `all` after `for`, found {}",
                p.peek().unwrap_or("the end of the sentence")
            ));
        }
        loop {
            match p.peek() {
                Some(":") => {
                    p.pos += 1;
                    break;
                }
                Some(t) if t.chars().next().is_some_and(|c| c.is_alphabetic()) => {
                    if declared.iter().any(|n| n == t) {
                        return Err(format!("`for all` names `{t}` twice"));
                    }
                    declared.push(t.to_string());
                    p.pos += 1;
                }
                Some(t) => return Err(format!("expected a variable, found `{t}`")),
                None => return Err("expected `:` after the variables".into()),
            }
            p.eat("and");
        }
    }

    let lhs = p.parse_sum()?;
    if !p.eat_eq() {
        return Err(match p.peek() {
            Some(t) => format!("expected `equals` (or `=`, `is`, `agrees with`), found `{t}`"),
            None => "expected `equals`, found the end of the sentence".into(),
        });
    }
    let rhs = p.parse_sum()?;

    let hypothesis = if p.eat("when") || p.eat("if") {
        if p.peek().is_none() {
            return Err("expected a hypothesis after `when`".into());
        }
        let h = p.parse_sum()?;
        if !p.eat_eq() {
            return Err(match p.peek() {
                Some(t) => format!("expected `equals` in the hypothesis, found `{t}`"),
                None => "expected `equals` in the hypothesis, found the end of the sentence".into(),
            });
        }
        let h2 = p.parse_sum()?;
        Some((h, h2))
    } else {
        None
    };

    if let Some(t) = p.peek() {
        return Err(format!("unexpected `{t}` after the claim"));
    }

    // The variables, in the order the sentence fixes them: the quantifier's
    // order first (so `for all l, r` and `for all r, l` are different readings
    // of the same polynomial, which the renaming check below then settles), then
    // first appearance.
    let mut used: Vec<String> = Vec::new();
    lhs.collect_vars(&mut used);
    rhs.collect_vars(&mut used);
    if let Some((h, h2)) = &hypothesis {
        h.collect_vars(&mut used);
        h2.collect_vars(&mut used);
    }
    let mut vars: Vec<String> = declared.clone();
    for v in &used {
        if !vars.contains(v) {
            vars.push(v.clone());
        }
    }
    if !declared.is_empty() {
        let mut a = declared.clone();
        a.sort();
        let mut b = used.clone();
        b.sort();
        if a != b {
            return Err(format!(
                "`for all` names {} but the claim uses {}",
                a.join(", "),
                if b.is_empty() {
                    "no variables".to_string()
                } else {
                    b.join(", ")
                }
            ));
        }
    }
    if vars.is_empty() {
        return Err("a kernel identity needs at least one state variable".into());
    }

    Ok(Sentence::Identity(KClaim {
        lhs,
        rhs,
        hypothesis,
        vars,
    }))
}

// ── the sidecar's JSON ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Def {
    pub name: String,
    pub params: Vec<String>,
    pub body: KTerm,
}

#[derive(Debug, Clone)]
pub struct Obligation {
    pub name: String,
    pub statement: String,
    pub vars: Vec<String>,
    pub lhs: KTerm,
    pub rhs: KTerm,
    pub hypothesis: Option<(KTerm, KTerm)>,
    /// The polynomials, canonicalized once at load time.
    lhs_poly: Poly,
    rhs_poly: Poly,
    hypothesis_poly: Option<(Poly, Poly)>,
}

/// A kernel certificate and the obligations it was proven from, loaded together
/// and cross-checked.
#[derive(Debug, Clone)]
pub struct Theory {
    pub certificate: PathBuf,
    pub source_sha: String,
    pub theorem_names: Vec<String>,
    defs: BTreeMap<String, Def>,
    pub obligations: Vec<Obligation>,
    vocab: Vocabulary,
}

/// What the fragment concluded about one sentence.
#[derive(Debug, Clone)]
pub struct Citation {
    pub sentence: String,
    pub verdict: Verdict,
    /// The theorem of the certificate that states this identity, when one does.
    pub cited: Option<String>,
    /// The identity as the certificate spells it (or why nothing was cited).
    pub goal: String,
}

impl Citation {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "claim": self.sentence,
            "verdict": self.verdict.label(),
            "cited": self.cited,
            "goal": self.goal,
        })
    }
}

fn json_term(v: &serde_json::Value) -> Result<KTerm, String> {
    if let Some(name) = v.get("var").and_then(|x| x.as_str()) {
        return Ok(KTerm::Var(name.to_string()));
    }
    if let Some(n) = v.get("int").and_then(|x| x.as_i64()) {
        return Ok(KTerm::Int(n));
    }
    if let Some(name) = v.get("call").and_then(|x| x.as_str()) {
        let args = v
            .get("args")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("call `{name}` has no args"))?
            .iter()
            .map(json_term)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(KTerm::Call {
            name: name.to_string(),
            args,
        });
    }
    let op = v
        .get("op")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("not an expression node: {v}"))?;
    let args = v
        .get("args")
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("`{op}` has no args"))?;
    let mut terms = args.iter().map(json_term);
    match op {
        "add" => Ok(KTerm::Add(
            Box::new(terms.next().transpose()?.ok_or("add with no left")?),
            Box::new(terms.next().transpose()?.ok_or("add with no right")?),
        )),
        "sub" => Ok(KTerm::Add(
            Box::new(terms.next().transpose()?.ok_or("sub with no left")?),
            Box::new(KTerm::Neg(Box::new(
                terms.next().transpose()?.ok_or("sub with no right")?,
            ))),
        )),
        "mul" => Ok(KTerm::Mul(
            Box::new(terms.next().transpose()?.ok_or("mul with no left")?),
            Box::new(terms.next().transpose()?.ok_or("mul with no right")?),
        )),
        "neg" => Ok(KTerm::Neg(Box::new(
            terms.next().transpose()?.ok_or("neg with no argument")?,
        ))),
        other => Err(format!("unknown operator `{other}`")),
    }
}

// ── canonical polynomials ──────────────────────────────────────────────────

fn clean(mut p: Poly) -> Poly {
    p.retain(|_, c| *c != 0);
    p
}

fn one(key: Vec<u32>, c: i64) -> Poly {
    let mut p = Poly::new();
    if c != 0 {
        p.insert(key, c);
    }
    p
}

fn poly_add(a: &Poly, b: &Poly) -> Result<Poly, String> {
    let mut out = a.clone();
    for (k, c) in b {
        let e = out.entry(k.clone()).or_insert(0);
        *e = e
            .checked_add(*c)
            .ok_or_else(|| "coefficient overflow while adding".to_string())?;
    }
    Ok(clean(out))
}

fn poly_neg(a: &Poly) -> Result<Poly, String> {
    let mut out = Poly::new();
    for (k, c) in a {
        out.insert(
            k.clone(),
            c.checked_neg()
                .ok_or_else(|| "coefficient overflow while negating".to_string())?,
        );
    }
    Ok(clean(out))
}

fn poly_mul(a: &Poly, b: &Poly) -> Result<Poly, String> {
    let mut out = Poly::new();
    for (ka, ca) in a {
        for (kb, cb) in b {
            let mut k: Vec<u32> = ka.iter().chain(kb.iter()).copied().collect();
            k.sort_unstable();
            let c = ca
                .checked_mul(*cb)
                .ok_or_else(|| "coefficient overflow while multiplying".to_string())?;
            let e = out.entry(k).or_insert(0);
            *e = e
                .checked_add(c)
                .ok_or_else(|| "coefficient overflow while multiplying".to_string())?;
        }
    }
    Ok(clean(out))
}

/// Canonicalize a term. `env` substitutes a definition's parameters with the
/// polynomials of its arguments; `index` assigns component indices as variables
/// are met, which is per-side and therefore arbitrary — the comparison below
/// renames them.
fn canon(
    t: &KTerm,
    defs: &BTreeMap<String, Def>,
    index: &mut BTreeMap<String, u32>,
    env: &BTreeMap<String, Poly>,
    depth: u32,
) -> Result<Poly, String> {
    if depth > 64 {
        return Err("definitions nest too deeply (or recurse)".into());
    }
    match t {
        KTerm::Int(n) => Ok(one(Vec::new(), *n)),
        KTerm::Var(v) => {
            if let Some(p) = env.get(v) {
                return Ok(p.clone());
            }
            let next = u32::try_from(index.len()).map_err(|_| "too many variables".to_string())?;
            let i = *index.entry(v.clone()).or_insert(next);
            Ok(one(vec![i], 1))
        }
        KTerm::Neg(a) => poly_neg(&canon(a, defs, index, env, depth + 1)?),
        KTerm::Add(a, b) => poly_add(
            &canon(a, defs, index, env, depth + 1)?,
            &canon(b, defs, index, env, depth + 1)?,
        ),
        KTerm::Mul(a, b) => poly_mul(
            &canon(a, defs, index, env, depth + 1)?,
            &canon(b, defs, index, env, depth + 1)?,
        ),
        KTerm::Call { name, args } => {
            let def = defs
                .get(name)
                .ok_or_else(|| format!("`{name}` is not a definition of this kernel"))?;
            if def.params.len() != args.len() {
                return Err(format!(
                    "`{name}` takes {} argument(s), got {}",
                    def.params.len(),
                    args.len()
                ));
            }
            let mut inner = BTreeMap::new();
            for (p, a) in def.params.iter().zip(args) {
                inner.insert(p.clone(), canon(a, defs, index, env, depth + 1)?);
            }
            canon(&def.body, defs, index, &inner, depth + 1)
        }
    }
}

/// Rename a polynomial's components: `map[i]` is the new index of component
/// `i`. Components must be renamed by a bijection, so collisions can only come
/// from two factors in one monomial meeting in the middle, which is exactly the
/// `and`-merge this has to add up.
fn relabel(p: &Poly, map: &[u32]) -> Result<Poly, String> {
    let mut out = Poly::new();
    for (k, c) in p {
        let mut key: Vec<u32> = k
            .iter()
            .map(|i| {
                map.get(*i as usize)
                    .copied()
                    .ok_or_else(|| "component index out of range".to_string())
            })
            .collect::<Result<_, _>>()?;
        key.sort_unstable();
        let e = out.entry(key).or_insert(0);
        *e = e
            .checked_add(*c)
            .ok_or_else(|| "coefficient overflow while renaming".to_string())?;
    }
    Ok(clean(out))
}

/// Every bijection from `n` claim variables onto `m` obligation variables, as
/// index maps. Kernels have a handful of components, and a bounded search keeps
/// the decision exact rather than heuristic.
fn bijections(n: usize, m: usize) -> Result<Vec<Vec<u32>>, String> {
    if n != m {
        return Ok(Vec::new());
    }
    if n > 6 {
        return Err(format!(
            "the fragment compares at most 6 state components, this claim has {n}"
        ));
    }
    let mut out = Vec::new();
    let mut current = vec![0u32; n];
    let mut used = vec![false; n];
    fn walk(
        i: usize,
        n: usize,
        current: &mut Vec<u32>,
        used: &mut Vec<bool>,
        out: &mut Vec<Vec<u32>>,
    ) {
        if i == n {
            out.push(current.clone());
            return;
        }
        for j in 0..n {
            if !used[j] {
                used[j] = true;
                current[i] = j as u32;
                walk(i + 1, n, current, used, out);
                used[j] = false;
            }
        }
    }
    walk(0, n, &mut current, &mut used, &mut out);
    Ok(out)
}

// ── loading ────────────────────────────────────────────────────────────────

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// The `-- Source:` header a certificate pins its obligations file with.
fn pinned_source(certificate: &str) -> Result<(String, String), String> {
    let line = certificate
        .lines()
        .find(|l| l.trim_start().starts_with("-- Source:"))
        .ok_or_else(|| {
            "the certificate carries no `-- Source:` line, so the obligations it was \
             emitted from cannot be identified"
                .to_string()
        })?;
    let body = line.trim_start().trim_start_matches("-- Source:").trim();
    let (path, rest) = body
        .rsplit_once("(sha256 ")
        .ok_or_else(|| format!("`-- Source:` line has no sha256 pin: {body}"))?;
    let sha = rest.trim_end_matches(')').trim().to_lowercase();
    if sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("`-- Source:` pin is not a sha256: {sha}"));
    }
    Ok((path.trim().to_string(), sha))
}

impl Theory {
    /// Load a certificate and the obligations it pins, from text already read.
    pub fn from_parts(
        certificate: PathBuf,
        certificate_text: &str,
        obligations_text: &str,
    ) -> Result<Theory, String> {
        let (_, pinned) = pinned_source(certificate_text)?;
        let got = sha256_hex(obligations_text.as_bytes());
        if got != pinned {
            return Err(format!(
                "{} pins obligations sha256 {pinned}, but the file read hashes to {got}: \
                 the certificate and the sidecar disagree, so no citation from one is \
                 evidence about the other",
                certificate.display()
            ));
        }

        let doc: serde_json::Value = serde_json::from_str(obligations_text)
            .map_err(|e| format!("obligations are not JSON: {e}"))?;
        let mut defs = BTreeMap::new();
        for d in doc
            .get("defs")
            .and_then(|v| v.as_array())
            .ok_or("obligations have no `defs` array")?
        {
            let name = d
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("a definition has no name")?
                .to_string();
            let params = d
                .get("params")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("`{name}` has no params"))?
                .iter()
                .map(|p| {
                    p.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("`{name}` has a non-string parameter"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let body = json_term(
                d.get("body")
                    .ok_or_else(|| format!("`{name}` has no body"))?,
            )?;
            defs.insert(name.clone(), Def { name, params, body });
        }

        let mut obligations = Vec::new();
        for o in doc
            .get("obligations")
            .and_then(|v| v.as_array())
            .ok_or("obligations have no `obligations` array")?
        {
            let name = o
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("an obligation has no name")?
                .to_string();
            let statement = o
                .get("statement")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let vars = o
                .get("vars")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("`{name}` has no vars"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("`{name}` has a non-string variable"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let lhs = json_term(o.get("lhs").ok_or_else(|| format!("`{name}` has no lhs"))?)?;
            let rhs = json_term(o.get("rhs").ok_or_else(|| format!("`{name}` has no rhs"))?)?;
            let hypothesis = match o.get("hypotheses").and_then(|v| v.as_array()) {
                Some(hs) if !hs.is_empty() => {
                    if hs.len() > 1 {
                        return Err(format!(
                            "`{name}` carries {} hypotheses; the fragment can state one",
                            hs.len()
                        ));
                    }
                    let h = &hs[0];
                    Some((
                        json_term(
                            h.get("lhs")
                                .ok_or_else(|| format!("`{name}`'s hypothesis has no lhs"))?,
                        )?,
                        json_term(
                            h.get("rhs")
                                .ok_or_else(|| format!("`{name}`'s hypothesis has no rhs"))?,
                        )?,
                    ))
                }
                _ => None,
            };

            // Canonicalize with the obligation's own variable order, which the
            // sidecar fixes, so the polynomial is stable across re-emissions.
            let mut index = BTreeMap::new();
            for (i, v) in vars.iter().enumerate() {
                index.insert(v.clone(), i as u32);
            }
            let env = BTreeMap::new();
            let lhs_poly = canon(&lhs, &defs, &mut index, &env, 0)?;
            let rhs_poly = canon(&rhs, &defs, &mut index, &env, 0)?;
            if index.len() != vars.len() {
                return Err(format!(
                    "`{name}` lists {} variable(s) but its statement uses {}",
                    vars.len(),
                    index.len()
                ));
            }
            let hypothesis_poly = match &hypothesis {
                Some((h, h2)) => {
                    let mut hi = index.clone();
                    let a = canon(h, &defs, &mut hi, &env, 0)?;
                    let b = canon(h2, &defs, &mut hi, &env, 0)?;
                    if hi.len() != vars.len() {
                        return Err(format!(
                            "`{name}`'s hypothesis mentions a variable outside `vars`"
                        ));
                    }
                    Some((a, b))
                }
                None => None,
            };

            obligations.push(Obligation {
                name,
                statement,
                vars,
                lhs,
                rhs,
                hypothesis,
                lhs_poly,
                rhs_poly,
                hypothesis_poly,
            });
        }

        // The citation has to name a theorem the certificate actually contains:
        // a sidecar that grew an obligation the certificate never proved would
        // otherwise be citable.
        let mut theorem_names = Vec::new();
        for line in certificate_text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("theorem ") {
                if let Some(name) = rest.split_whitespace().next() {
                    theorem_names.push(name.to_string());
                }
            }
        }
        for o in &obligations {
            if !theorem_names.iter().any(|t| *t == o.name) {
                return Err(format!(
                    "{} has no theorem `{}`, so the obligation cannot be cited",
                    certificate.display(),
                    o.name
                ));
            }
        }

        let vocab = Vocabulary::new(defs.values().cloned().collect::<Vec<_>>().as_slice());
        Ok(Theory {
            certificate,
            source_sha: pinned,
            theorem_names,
            defs,
            obligations,
            vocab,
        })
    }

    /// Load `<certificate>`, reading the obligations file its `-- Source:` line
    /// names (resolved against the working directory, then against the repo root
    /// the certificate lives under).
    pub fn load(certificate: &Path, obligations: Option<&Path>) -> Result<Theory, String> {
        let text = std::fs::read_to_string(certificate)
            .map_err(|e| format!("{}: {e}", certificate.display()))?;
        let (source, _) = pinned_source(&text)?;
        let path = match obligations {
            Some(p) => p.to_path_buf(),
            None => {
                let direct = PathBuf::from(&source);
                if direct.is_file() {
                    direct
                } else {
                    // `mm0/<cert>.mm0` → repo root → the pinned relative path.
                    let root = certificate.parent().and_then(Path::parent).ok_or_else(|| {
                        format!("{}: cannot resolve `{source}`", certificate.display())
                    })?;
                    let p = root.join(&source);
                    if !p.is_file() {
                        return Err(format!(
                            "{} pins `{source}`, which is not found from here; pass \
                             --kernel-obligations",
                            certificate.display()
                        ));
                    }
                    p
                }
            }
        };
        let obligations_text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Theory::from_parts(certificate.to_path_buf(), &text, &obligations_text)
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Parse a sentence of this kernel's fragment.
    pub fn parse(&self, text: &str) -> Result<Sentence, String> {
        parse(text, &self.vocab)
    }

    pub fn certificate_stem(&self) -> String {
        self.certificate
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "certificate".to_string())
    }

    /// Decide a sentence: cite the certificate's theorem when it states exactly
    /// this identity, and say what is missing when it does not.
    pub fn decide(&self, text: &str) -> Result<Citation, String> {
        let sentence = self.parse(text)?;
        Ok(match sentence {
            Sentence::Quality(quality) => {
                let stem = self.certificate_stem();
                Citation {
                    sentence: text.to_string(),
                    verdict: Verdict::Unproved(format!(
                        "`{quality}` is a property of the scheme, not an identity over its \
                         state polynomials — `{stem}` states no such sentence, and this \
                         fragment cannot decide it"
                    )),
                    cited: None,
                    goal: quality,
                }
            }
            Sentence::Identity(claim) => self.cite(&claim, text),
        })
    }

    fn cite(&self, claim: &KClaim, text: &str) -> Citation {
        let mut index = BTreeMap::new();
        for (i, v) in claim.vars.iter().enumerate() {
            index.insert(v.clone(), i as u32);
        }
        let env = BTreeMap::new();

        let lhs = match canon(&claim.lhs, &self.defs, &mut index, &env, 0) {
            Ok(p) => p,
            Err(e) => return self.unreadable(text, e),
        };
        let rhs = match canon(&claim.rhs, &self.defs, &mut index, &env, 0) {
            Ok(p) => p,
            Err(e) => return self.unreadable(text, e),
        };
        let mut hyp = None;
        if let Some((a, b)) = &claim.hypothesis {
            let pa = match canon(a, &self.defs, &mut index, &env, 0) {
                Ok(p) => p,
                Err(e) => return self.unreadable(text, e),
            };
            let pb = match canon(b, &self.defs, &mut index, &env, 0) {
                Ok(p) => p,
                Err(e) => return self.unreadable(text, e),
            };
            hyp = Some((pa, pb));
        }

        // Exact citation: the same pair of polynomials, up to a renaming of the
        // components, and the same hypothesis (or none).
        for o in &self.obligations {
            let maps = match bijections(claim.vars.len(), o.vars.len()) {
                Ok(m) => m,
                Err(e) => return self.unreadable(text, e),
            };
            for map in maps {
                let l = match relabel(&lhs, &map) {
                    Ok(p) => p,
                    Err(e) => return self.unreadable(text, e),
                };
                let r = match relabel(&rhs, &map) {
                    Ok(p) => p,
                    Err(e) => return self.unreadable(text, e),
                };
                // A hypothesis is part of the identity: the certificate's
                // uniform-state theorem is about the locus `l = r`, so a claim
                // that omits it is a different (and generally false) sentence.
                let same_hypothesis = match (&hyp, &o.hypothesis_poly) {
                    (None, None) => true,
                    (Some((a, b)), Some((c, d))) => {
                        let ra = match relabel(a, &map) {
                            Ok(p) => p,
                            Err(e) => return self.unreadable(text, e),
                        };
                        let rb = match relabel(b, &map) {
                            Ok(p) => p,
                            Err(e) => return self.unreadable(text, e),
                        };
                        ra == *c && rb == *d
                    }
                    _ => false,
                };
                let same = l == o.lhs_poly && r == o.rhs_poly && same_hypothesis;
                if same {
                    return Citation {
                        sentence: text.to_string(),
                        verdict: Verdict::Proved,
                        cited: Some(format!("{}.{}", self.certificate_stem(), o.name)),
                        goal: o.statement.clone(),
                    };
                }
            }
        }

        // No theorem states it. Two very different situations, and the reason
        // line distinguishes them, because "identity but uncited" is a gap in the
        // certificate while "not an identity" is a different sentence.
        let identity = hyp.is_none() && lhs == rhs;
        let stem = self.certificate_stem();
        Citation {
            sentence: text.to_string(),
            verdict: Verdict::Unproved(if identity {
                format!(
                    "the identity holds in ℤ[x⃗] but `{stem}` states no theorem of this \
                     shape, so there is no proof to cite (a certificate that gained this \
                     obligation would make the sentence provable)"
                )
            } else {
                format!(
                    "not an identity in ℤ[x⃗], and `{stem}` refutes nothing: a sentence is \
                     false only with a model, which this fragment does not build"
                )
            }),
            cited: None,
            goal: String::new(),
        }
    }

    fn unreadable(&self, text: &str, why: String) -> Citation {
        Citation {
            sentence: text.to_string(),
            verdict: Verdict::Unproved(why),
            cited: None,
            goal: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBLIGATIONS: &str = r#"{
      "module": "Flux1d",
      "defs": [
        {"name": "flux", "params": ["u"],
         "body": {"op": "add", "args": [
            {"op": "mul", "args": [{"var": "u"}, {"int": 3}]},
            {"op": "mul", "args": [{"var": "u"}, {"var": "u"}]}]}},
        {"name": "numFlux", "params": ["l", "r"],
         "body": {"op": "sub", "args": [
            {"op": "mul", "args": [{"call": "flux", "args": [{"var": "l"}]}, {"int": 2}]},
            {"call": "flux", "args": [{"var": "r"}]}]}},
        {"name": "delta", "params": ["l", "r"],
         "body": {"op": "sub", "args": [
            {"call": "numFlux", "args": [{"var": "l"}, {"var": "r"}]},
            {"call": "numFlux", "args": [{"var": "r"}, {"var": "l"}]}]}}
      ],
      "obligations": [
        {"name": "consistency", "statement": "numFlux u u = flux u", "vars": ["u"],
         "lhs": {"call": "numFlux", "args": [{"var": "u"}, {"var": "u"}]},
         "rhs": {"call": "flux", "args": [{"var": "u"}]}},
        {"name": "symmetry", "statement": "delta l r + delta r l = 0", "vars": ["l", "r"],
         "lhs": {"op": "add", "args": [
            {"call": "delta", "args": [{"var": "l"}, {"var": "r"}]},
            {"call": "delta", "args": [{"var": "r"}, {"var": "l"}]}]},
         "rhs": {"int": 0}},
        {"name": "uniform_state", "statement": "l = r => delta l r = 0", "vars": ["l", "r"],
         "hypotheses": [{"label": "l = r", "lhs": {"var": "l"}, "rhs": {"var": "r"}}],
         "lhs": {"call": "delta", "args": [{"var": "l"}, {"var": "r"}]},
         "rhs": {"int": 0}}
      ]
    }"#;

    fn theory() -> Theory {
        let sha = sha256_hex(OBLIGATIONS.as_bytes());
        let cert = format!(
            "-- test certificate\n-- Source: x/flux1d.obligations.json (sha256 {sha})\n\
             theorem consistency (s: nat): $ req a b $;\n\
             theorem symmetry (s: nat): $ req a b $;\n\
             theorem uniform_state (s: nat): $ req a b $;\n"
        );
        Theory::from_parts(PathBuf::from("flux1d_certificate.mm0"), &cert, OBLIGATIONS)
            .expect("fixture loads")
    }

    #[test]
    fn kernel_sentences_cite_the_certificate() {
        let t = theory();
        let c = t
            .decide("for all u: numerical flux of u and u agrees with flux of u")
            .unwrap();
        assert_eq!(c.verdict, Verdict::Proved);
        assert_eq!(c.cited.as_deref(), Some("flux1d_certificate.consistency"));

        // The same sentence with the other component names: renaming is not a
        // different identity.
        let c = t
            .decide("for all z: numFlux of z and z equals flux of z")
            .unwrap();
        assert_eq!(c.verdict, Verdict::Proved);

        let c = t
            .decide("for all l, r: delta of l and r plus delta of r and l equals zero")
            .unwrap();
        assert_eq!(c.cited.as_deref(), Some("flux1d_certificate.symmetry"));
    }

    #[test]
    fn the_hypothesis_is_part_of_the_identity() {
        let t = theory();
        let c = t
            .decide("delta of l and r equals zero when l equals r")
            .unwrap();
        assert_eq!(c.verdict, Verdict::Proved);
        assert_eq!(c.cited.as_deref(), Some("flux1d_certificate.uniform_state"));

        // Without the hypothesis it is a different (false) sentence, and the
        // fragment says so rather than citing the hypothesis form.
        let c = t.decide("delta of l and r equals zero").unwrap();
        assert!(matches!(c.verdict, Verdict::Unproved(_)));
        assert_eq!(c.cited, None);
    }

    #[test]
    fn a_quality_of_the_scheme_is_never_proved() {
        let t = theory();
        let c = t.decide("the scheme is second order accurate").unwrap();
        match c.verdict {
            Verdict::Unproved(why) => assert!(why.contains("property of the scheme")),
            other => panic!("expected unproved, got {other:?}"),
        }
    }

    #[test]
    fn a_true_identity_outside_the_certificate_is_unproved_not_proved() {
        let t = theory();
        // delta(u,u) = 0 is true in the ring, but no theorem states it that way.
        let c = t.decide("delta of u and u equals zero").unwrap();
        match c.verdict {
            Verdict::Unproved(why) => assert!(why.contains("holds in ℤ[x⃗]")),
            other => panic!("expected unproved, got {other:?}"),
        }
    }

    #[test]
    fn a_drifted_sidecar_cannot_be_cited() {
        let o = OBLIGATIONS.replace("\"int\": 3", "\"int\": 4");
        let sha = sha256_hex(OBLIGATIONS.as_bytes());
        let cert = format!(
            "-- Source: x/flux1d.obligations.json (sha256 {sha})\n\
             theorem consistency (s: nat): $ req a b $;\n"
        );
        let err = Theory::from_parts(PathBuf::from("c.mm0"), &cert, &o)
            .expect_err("a drifted sidecar must be refused");
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn an_obligation_the_certificate_does_not_prove_cannot_be_cited() {
        let sha = sha256_hex(OBLIGATIONS.as_bytes());
        let cert = format!(
            "-- Source: x/flux1d.obligations.json (sha256 {sha})\n\
             theorem consistency (s: nat): $ req a b $;\n"
        );
        let err = Theory::from_parts(PathBuf::from("c.mm0"), &cert, OBLIGATIONS)
            .expect_err("a missing theorem must be refused");
        assert!(err.contains("symmetry"), "{err}");
    }

    #[test]
    fn the_parser_names_the_token_it_cannot_place() {
        let t = theory();
        let err = t
            .parse("for all u: vorticity of u equals zero")
            .unwrap_err();
        assert!(err.contains("vorticity"), "{err}");
        let err = t.parse("delta of l and r equals zero when").unwrap_err();
        assert!(err.contains("hypothesis"), "{err}");
    }
}
