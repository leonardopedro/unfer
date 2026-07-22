use serde::{Deserialize, Serialize};
use std::fmt;

pub type Id = String;
pub type TagId = u32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoreIR {
    Var(Id),
    Lit(Literal),
    Con(TagId, Vec<CoreIR>),
    Lam(Id, Box<CoreIR>),
    App(Box<CoreIR>, Box<CoreIR>),
    Let(Id, Box<CoreIR>, Box<CoreIR>),
    Match(Box<CoreIR>, Vec<(Pattern, CoreIR)>),
    Fold(Box<CoreIR>, Box<CoreIR>, Box<CoreIR>),
    Prim(PrimOp, Vec<CoreIR>),
    Clone(Id, Id, Id, Box<CoreIR>),
    Drop(Id, Box<CoreIR>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Tag(TagId, Vec<Id>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimOp {
    Add64,
    Sub64,
    Mul64,
    Eq64,
    Gt64,
    Lt64,
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Int64(i64),
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int64(n) => write!(f, "{}", n),
            Literal::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl fmt::Display for CoreIR {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreIR::Var(id) => write!(out, "{}", id),
            CoreIR::Lit(lit) => write!(out, "{}", lit),
            CoreIR::Con(tag, args) => {
                write!(out, "(Con {} ", tag)?;
                for arg in args {
                    write!(out, "{} ", arg)?;
                }
                write!(out, ")")
            }
            CoreIR::Lam(id, body) => write!(out, "(lam {} {})", id, body),
            CoreIR::App(func, arg) => write!(out, "(app {} {})", func, arg),
            CoreIR::Let(id, value, body) => write!(out, "(let {} {} {})", id, value, body),
            CoreIR::Match(scrutinee, arms) => {
                write!(out, "(match {}", scrutinee)?;
                for (pat, body) in arms {
                    write!(out, " ({:?} {})", pat, body)?;
                }
                write!(out, ")")
            }
            CoreIR::Prim(op, args) => {
                write!(out, "({:?} ", op)?;
                for arg in args {
                    write!(out, "{} ", arg)?;
                }
                write!(out, ")")
            }
            CoreIR::Fold(fi, init, list) => write!(out, "(fold {} {} {})", fi, init, list),
            CoreIR::Clone(id, id1, id2, body) => {
                write!(out, "(clone {} {} {} {})", id, id1, id2, body)
            }
            CoreIR::Drop(id, body) => write!(out, "(drop {} {})", id, body),
        }
    }
}

pub const NIL_TAG: TagId = 100;
pub const CONS_TAG: TagId = 101;
