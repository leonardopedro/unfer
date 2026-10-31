//! The math fragment of the controlled natural language.
//!
//! This is deliberately *not* a general English parser: `logos` already has one
//! (`l1` + `lexicon` + `ccg`), and it is the wrong tool here. A claim that is
//! going to be certified needs a surface syntax where every accepted sentence
//! has exactly one reading, so that translation into MM0 is a function and not
//! a best guess — the same reason the L1 subset exists for programs. The
//! fragment is:
//!
//! ```text
//! claim    := ["for all" var {"," var} ("," | ":")] body
//! body     := expr eq expr
//! eq       := "equals" | "is equal to" | "is" | "=" | "equal"
//! expr     := term {("plus" | "+") term}
//! term     := atom {("times" | "*" | "multiplied by") atom}
//! atom     := number | var | "suc" atom | ["the"] "successor" ["of"] atom
//!           | "(" expr ")"
//! question := body "?" | "is" body "?" | "does" expr eq expr "?"
//! ```
//!
//! Numbers are decimal (`5`) or words (`five`, `twenty one`). A pronoun-free
//! sentence like `is two plus three equal to five?` is a *question*: it is
//! certified exactly like the corresponding claim, and the shape of the
//! certificate — a proof of `teq L R`, or a proof of `tfals` from `teq L R` —
//! is the answer. That is the whole interface: asking is proving.
//!
//! The parser refuses everything outside the fragment (with the offending token
//! named), because a claim that *nearly* parses is a claim that would be
//! certified about the wrong sentence.

use super::proof::Expr;

/// One sentence of the fragment: `for all x: <lhs> equals <rhs>`, asked
/// (`question`) or asserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub lhs: Expr,
    pub rhs: Expr,
    /// The sentence was interrogative. The certification is identical; the
    /// caller reports the verdict as an answer instead of an assertion.
    pub question: bool,
}

impl Claim {
    /// Failures carry the token that stopped the parse, like the rest of the
    /// pipeline's diagnostics (`translate` reports the source span it choked on
    /// rather than "invalid input").
    pub fn parse(text: &str) -> Result<Claim, String> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Err("empty sentence".into());
        }
        let mut p = Parser { tokens, pos: 0 };

        // `for all x, y: …` is accepted for readability, and then checked
        // against the variables the body actually uses. The quantifier is not
        // *needed* — a free variable is universally quantified by the emitted
        // theorem's binders — so a `for all` that names a variable the body
        // never mentions is an error rather than silently dropped.
        let declared = p.parse_forall()?;

        // Interrogative openers. `is` is also an equality word, so it is only
        // consumed here at the start of the sentence; `parse_body` handles the
        // same token between two expressions (`x is equal to y`). `it is true
        // that` is stripped as a unit, because its `is` is not an equality.
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

        let body = p.parse_body(leading_is)?;
        if p.eat("?") {
            question = true;
        }
        if let Some(t) = p.peek() {
            return Err(format!("unexpected `{t}` after the claim"));
        }

        if !declared.is_empty() {
            let used = {
                let mut all = body.lhs.vars();
                for v in body.rhs.vars() {
                    if !all.contains(&v) {
                        all.push(v);
                    }
                }
                all.sort();
                all
            };
            let mut declared_sorted = declared.clone();
            declared_sorted.sort();
            declared_sorted.dedup();
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

        Ok(Claim {
            lhs: body.lhs,
            rhs: body.rhs,
            question,
        })
    }
}

/// What `parse_body` produced: the two sides of the equality.
struct Body {
    lhs: Expr,
    rhs: Expr,
}

struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
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

    /// `for all x, y : …` / `for every x : …`. Returns the declared names.
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
                    names.push(t.to_string());
                    self.pos += 1;
                }
                Some(t) => return Err(format!("expected a variable, found `{t}`")),
                None => return Err("expected `:` after the variables".into()),
            }
            // `,` and `and` both separate; the tokenizer drops the comma, so a
            // bare juxtaposition is accepted too.
            self.eat("and");
        }
    }

    fn parse_body(&mut self, leading_is: bool) -> Result<Body, String> {
        let lhs = self.parse_expr()?;
        if self.eat("=") || self.eat("equals") {
            // `x = y`, `x equals y`.
        } else if self.eat("equal") {
            // `x equal to y`, and the `equal` of `does x equal y`.
            self.eat("to");
        } else if self.eat("is") {
            // `x is equal to y`, `x is y`.
            self.eat("equal");
            self.eat("to");
        } else if !leading_is {
            return Err(match self.peek() {
                Some(t) => format!("expected `equals`, found `{t}`"),
                None => "expected `equals`, found the end of the sentence".into(),
            });
        }
        // With `leading_is`, the opening `is` *was* the equality: `is two plus
        // three five?`.
        let rhs = self.parse_expr()?;
        Ok(Body { lhs, rhs })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_term()?;
        // `and` is an addition word, so `two and three equals five` parses.
        while self.at_any(&["plus", "+", "and"]) {
            self.pos += 1;
            let r = self.parse_term()?;
            e = Expr::Add(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_atom()?;
        loop {
            if self.eat("times") || self.eat("*") {
                let r = self.parse_atom()?;
                e = Expr::Mul(Box::new(e), Box::new(r));
            } else if self.eat("multiplied") {
                self.expect("by")?;
                let r = self.parse_atom()?;
                e = Expr::Mul(Box::new(e), Box::new(r));
            } else {
                return Ok(e);
            }
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        let token = match self.peek() {
            Some(t) => t.to_string(),
            None => {
                return Err(
                    "expected a number or a variable, found the end of the sentence".into(),
                );
            }
        };

        // `suc a`, `successor of a`, `the successor of a`: the successor is a
        // *constructor* (one atom), which is why it is parsed here and not as an
        // operator — `suc a times b` is `(suc a) * b`, and `suc a plus b` is
        // `(suc a) + b`, exactly as in an MM0 Peano development.
        //
        // `suc` is a keyword, so a variable may not be called `suc`; that is the
        // price of reading the sentence the way the source states it.
        let successor = if self.eat("suc") {
            true
        } else if self.eat("successor") {
            // Both spellings are accepted; `of` is conventional, not required.
            self.eat("of");
            true
        } else if token == "the"
            && self.tokens.get(self.pos + 1).map(String::as_str) == Some("successor")
        {
            self.pos += 2;
            self.eat("of");
            true
        } else {
            false
        };
        if successor {
            let inner = self.parse_atom()?;
            return Ok(Expr::Suc(Box::new(inner)));
        }
        if token == "(" {
            self.pos += 1;
            let e = self.parse_expr()?;
            self.expect(")")?;
            return Ok(e);
        }
        if let Some(n) = number_value(&token) {
            self.pos += 1;
            // A tens word may be followed by a unit word: `twenty one`.
            if let Some(next) = self.peek().and_then(number_value)
                && n % 10 == 0
                && n >= 20
                && (1..=9).contains(&next)
            {
                self.pos += 1;
                return Ok(Expr::Num(n + next));
            }
            return Ok(Expr::Num(n));
        }
        if token.chars().next().is_some_and(|c| c.is_alphabetic()) {
            self.pos += 1;
            return Ok(Expr::Var(token));
        }
        Err(format!("expected a number or a variable, found `{token}`"))
    }
}

/// Split into lowercase words, single-character operators, and parentheses.
///
/// Punctuation is dropped rather than tokenized: a comma is a separator the
/// grammar already knows about (after the universal quantifier), and a stray
/// period should not be a parse error in a sentence a human typed.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        match c {
            '+' | '*' | '=' | '(' | ')' | '?' | ':' => {
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

/// The value of a numeral token, decimal or spelled.
fn number_value(token: &str) -> Option<u64> {
    if let Ok(n) = token.parse::<u64>() {
        return Some(n);
    }
    Some(match token {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        "hundred" => 100,
        _ => return None,
    })
}
