//! Word-level Taylor Expansion Diagram (TED) canonicalization.
//!
//! Phase 3 of the Total-Austral-Subset plan, adapted to the existing
//! deltanet pipeline: once the interaction-net reducer halts, the
//! arithmetic fragment that remains (Int64 literals, free variables, and
//! `Add64`/`Sub64`/`Mul64` primitives) is extracted into a **pure
//! algebraic DAG** and canonicalized into a **word-level TED**: a sorted
//! polynomial normal form over ℤ/2⁶⁴ℤ.
//!
//! The canonical form is genuinely unique (not just deterministic):
//!   • products distribute:  `(x + 2) * 3` → `3*x + 6`;
//!   • like terms combine:   `x + x` → `2*x`  (and `0*x` → `0`);
//!   • monomials are ordered deterministically (lexicographic variable
//!     order, then exponent), so `x*y + 1` and `1 + y*x` serialize
//!     identically.
//!
//! `ted_hash` then returns the SHA-256 of the canonical serialization —
//! the content-addressable *algebraic* UNF, independent of the
//! interaction-net encoding (the net-level `unf_hash` is a different
//! canonical form, see `unf.rs`). Two nets that compute the same integer
//! polynomial produce the same TED hash; a closed term collapses to a
//! single constant term (`2 + 3` → `5`), which is exactly the
//! "replace the normal form with a numerical calculation" step.
//!
//! Terms outside the Int64-arithmetic fragment (F64, Bool, comparisons,
//! application, constructors, folds, lambdas) are **not** TED-able —
//! `from_sym_expr` returns `None` and callers fall back to the
//! interaction-net UNF hash.

use crate::core_ir::PrimOp;
use sha2::{Digest, Sha256};

use super::symbolic::SymExpr;

/// A monomial: `coeff * v1^e1 * v2^e2 * …`, with the variable powers kept
/// sorted by variable name and the coefficient taken in ℤ/2⁶⁴ (wrapping).
///
/// Ordering is the canonical polynomial order: variable terms come before
/// constants, then lexicographic by variable list, then by coefficient
/// (so `x*y + 1`, `3*x + 6` serialize in the expected order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Monomial {
    /// Coefficient in ℤ/2⁶⁴ (stored as `i64`, wrapping arithmetic).
    pub coeff: i64,
    /// (variable, exponent) pairs, sorted by variable name.
    pub vars: Vec<(String, u32)>,
}

impl PartialOrd for Monomial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Monomial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Constants (empty vars) sort last; otherwise lexicographic by
        // variable list, then by coefficient.
        match self.vars.is_empty().cmp(&other.vars.is_empty()) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.vars.cmp(&other.vars).then(self.coeff.cmp(&other.coeff))
    }
}

impl Monomial {
    pub fn constant(c: i64) -> Self {
        Monomial { coeff: c, vars: Vec::new() }
    }

    /// Multiply two monomials (coefficients wrap mod 2⁶⁴; exponents add).
    /// A zero coefficient collapses the monomial to the constant zero
    /// (empty variable list), so `0 * y` becomes plain `0`.
    pub fn mul(&self, other: &Monomial) -> Monomial {
        let coeff = self.coeff.wrapping_mul(other.coeff);
        if coeff == 0 {
            return Monomial::constant(0);
        }
        let mut vars = self.vars.clone();
        for (v, e) in &other.vars {
            match vars.iter_mut().find(|(name, _)| name == v) {
                Some((_, pe)) => *pe += e,
                None => vars.push((v.clone(), *e)),
            }
        }
        vars.sort();
        Monomial { coeff, vars }
    }
}

/// A word-level TED: a sorted polynomial over ℤ/2⁶⁴ — the unique normal
/// form of the arithmetic fragment of a reduced net.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ted {
    /// Non-zero monomials, sorted (deterministic canonical order).
    pub terms: Vec<Monomial>,
}

impl Ted {
    pub fn zero() -> Self {
        Ted { terms: Vec::new() }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Canonical addition: merge and combine like terms, drop zero
    /// coefficients (wrapping mod 2⁶⁴).
    pub fn add(&self, other: &Ted) -> Ted {
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        terms.sort();
        let mut out: Vec<Monomial> = Vec::with_capacity(terms.len());
        for m in terms {
            // Drop any monomial that collapsed to a zero coefficient.
            if m.coeff == 0 {
                continue;
            }
            if let Some(last) = out.last_mut() {
                if last.vars == m.vars {
                    last.coeff = last.coeff.wrapping_add(m.coeff);
                    if last.coeff == 0 {
                        out.pop();
                    }
                    continue;
                }
            }
            out.push(m);
        }
        Ted { terms: out }
    }

    /// Canonical subtraction: `a - b = a + (-b)` (coefficients negate mod
    /// 2⁶⁴, i.e. two's complement).
    pub fn sub(&self, other: &Ted) -> Ted {
        let neg: Vec<Monomial> = other
            .terms
            .iter()
            .map(|m| Monomial { coeff: m.coeff.wrapping_neg(), vars: m.vars.clone() })
            .collect();
        self.add(&Ted { terms: neg })
    }

    /// Canonical product: full distribution over monomials, then like-term
    /// combination. `0 * X → 0` falls out of the coefficient arithmetic.
    pub fn mul(&self, other: &Ted) -> Ted {
        if self.is_zero() || other.is_zero() {
            return Ted::zero();
        }
        let mut acc = Ted::zero();
        for a in &self.terms {
            for b in &other.terms {
                acc = acc.add(&Ted { terms: vec![a.mul(b)] });
            }
        }
        acc
    }

    /// Deterministic canonical serialization, e.g. `2*x^2*y + 3*x + 5`,
    /// `0` for the zero polynomial. Term order: sorted monomial order.
    pub fn to_canonical_string(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let parts: Vec<String> = self.terms.iter().map(monomial_to_string).collect();
        parts.join(" + ")
    }

    /// SHA-256 of the canonical serialization — the content-addressable
    /// algebraic UNF (64 hex chars).
    pub fn ted_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.to_canonical_string().as_bytes());
        format!("{:x}", h.finalize())
    }
}

fn monomial_to_string(m: &Monomial) -> String {
    let mut s = String::new();
    if m.vars.is_empty() {
        return m.coeff.to_string();
    }
    // Coefficient display: -1 → "-", 1 → "", else "c*".
    match m.coeff {
        1 => {}
        -1 => s.push('-'),
        c => s.push_str(&format!("{c}*")),
    }
    for (i, (v, e)) in m.vars.iter().enumerate() {
        if i > 0 {
            s.push('*');
        }
        s.push_str(v);
        if *e > 1 {
            s.push('^');
            s.push_str(&e.to_string());
        }
    }
    s
}

/// Extract the Int64-arithmetic fragment of a `SymExpr` into its canonical
/// TED. Returns `None` when the expression is outside the fragment (F64,
/// Bool, comparisons, application, constructors, folds, lambdas, stuck
/// remainders) — callers then fall back to the net-level UNF hash.
pub fn from_sym_expr(expr: &SymExpr) -> Option<Ted> {
    match expr {
        SymExpr::Lit(crate::core_ir::Literal::Int64(n)) => {
            Some(Ted { terms: vec![Monomial::constant(*n)] })
        }
        SymExpr::Var(name) => Some(Ted {
            terms: vec![Monomial { coeff: 1, vars: vec![(name.clone(), 1)] }],
        }),
        SymExpr::Prim(op, a, b) => {
            let ta = from_sym_expr(a)?;
            let tb = from_sym_expr(b)?;
            match op {
                PrimOp::Add64 => Some(ta.add(&tb)),
                PrimOp::Sub64 => Some(ta.sub(&tb)),
                PrimOp::Mul64 => Some(ta.mul(&tb)),
                // Comparisons/booleans/F64 are outside the word-level TED.
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::Literal;

    fn lit(n: i64) -> SymExpr {
        SymExpr::Lit(Literal::Int64(n))
    }

    fn add(a: SymExpr, b: SymExpr) -> SymExpr {
        SymExpr::Prim(PrimOp::Add64, Box::new(a), Box::new(b))
    }

    fn sub(a: SymExpr, b: SymExpr) -> SymExpr {
        SymExpr::Prim(PrimOp::Sub64, Box::new(a), Box::new(b))
    }

    fn mul(a: SymExpr, b: SymExpr) -> SymExpr {
        SymExpr::Prim(PrimOp::Mul64, Box::new(a), Box::new(b))
    }

    #[test]
    fn algebraically_equal_expressions_have_identical_ted() {
        // x + (y * 0) ≡ x — the zero product must vanish.
        let x = SymExpr::Var("x".to_string());
        let y = SymExpr::Var("y".to_string());
        let a = add(x.clone(), mul(y, lit(0)));
        let b = x;
        let ta = from_sym_expr(&a).unwrap();
        let tb = from_sym_expr(&b).unwrap();
        assert_eq!(ta, tb);
        assert_eq!(ta.to_canonical_string(), "x");
        assert_eq!(ta.ted_hash(), tb.ted_hash());
    }

    #[test]
    fn like_terms_combine() {
        // x + x → 2*x
        let x = SymExpr::Var("x".to_string());
        let t = from_sym_expr(&add(x.clone(), x)).unwrap();
        assert_eq!(t.to_canonical_string(), "2*x");
    }

    #[test]
    fn products_distribute_and_order_is_canonical() {
        // (x + 2) * 3 → 3*x + 6
        let x = SymExpr::Var("x".to_string());
        let t = from_sym_expr(&mul(add(x, lit(2)), lit(3))).unwrap();
        assert_eq!(t.to_canonical_string(), "3*x + 6");

        // y*x + 1 and 1 + x*y serialize identically (sorted monomial order).
        let x = SymExpr::Var("x".to_string());
        let y = SymExpr::Var("y".to_string());
        let a = from_sym_expr(&add(mul(y.clone(), x.clone()), lit(1))).unwrap();
        let b = from_sym_expr(&add(lit(1), mul(x, y))).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_canonical_string(), "x*y + 1");
    }

    #[test]
    fn closed_term_collapses_to_constant() {
        // (2 + 3) * 4 → 20 — the numerical calculation replaces the net.
        let t = from_sym_expr(&mul(add(lit(2), lit(3)), lit(4))).unwrap();
        assert_eq!(t.to_canonical_string(), "20");
        assert_eq!(t.terms.len(), 1);
    }

    #[test]
    fn wrapping_coefficient_arithmetic() {
        // 2^63 * 2 ≡ 0 mod 2^64 (wrapping), not a panic or a bigger number.
        let big = i64::MIN; // 2^63
        let t = from_sym_expr(&mul(lit(big), lit(2))).unwrap();
        assert_eq!(t.to_canonical_string(), "0");
    }

    #[test]
    fn non_arithmetic_fragment_is_not_ted_able() {
        // F64 and booleans fall outside the word-level TED.
        let f = SymExpr::Lit(Literal::F64(1.5));
        assert!(from_sym_expr(&f).is_none());
        let b = SymExpr::Lit(Literal::Bool(true));
        assert!(from_sym_expr(&b).is_none());
        // And a comparison (Eq64) is outside the polynomial fragment.
        let cmp = SymExpr::Prim(
            PrimOp::Eq64,
            Box::new(lit(1)),
            Box::new(lit(2)),
        );
        assert!(from_sym_expr(&cmp).is_none());
    }

    #[test]
    fn hash_is_stable_across_equivalent_encodings() {
        // The same polynomial written differently reduces to the same hash.
        let x = SymExpr::Var("x".to_string());
        let t1 = from_sym_expr(&add(x.clone(), x.clone())).unwrap(); // x + x
        let t2 = from_sym_expr(&mul(lit(2), x)).unwrap(); // 2 * x
        assert_eq!(t1.ted_hash(), t2.ted_hash());
        assert_eq!(t1.to_canonical_string(), t2.to_canonical_string());
    }
}
