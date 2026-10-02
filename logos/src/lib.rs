//! "Project Logos" — a controlled natural language (CNL) compiler to
//! verified execution graphs.
//!
//! Phase sequence: parse → compile → reduce → readback → hash.
//! [`l1`] + [`lexicon`] define the CNL subset and lexicon,
//! [`ccg`] the combinatory-categorial parser, [`core_ir`]/[`deltanet`] the
//! execution-graph IR and its reducer, [`harper_gate`] the verification
//! gate, [`austral_codegen`] the Austral backend, and [`cli`] the CLI driver.
//!
//! [`formalize`] is the NL → CNL autoformalization pipeline built on those
//! stages: natural-language proofs are decomposed into a dependency DAG and
//! each node is generated, verified by reduction, and scored.

pub mod austral_codegen;
pub mod ccg;
pub mod cli;
pub mod core_ir;
pub mod deltanet;
pub mod engram;
pub mod formalize;
pub mod harper_gate;
pub mod l1;
pub mod lexicon;
pub mod translate;
