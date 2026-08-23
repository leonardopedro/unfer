//! "Project Logos" — a controlled natural language (CNL) compiler to
//! verified execution graphs.
//!
//! Phase sequence: parse → compile → reduce → readback → hash.
//! [`l1`] + [`lexicon`] define the CNL subset and lexicon,
//! [`ccg`] the combinatory-categorial parser, [`core_ir`]/[`deltanet`] the
//! execution-graph IR and its reducer, [`harper_gate`] the verification
//! gate, [`austral_codegen`] the Austral backend, and [`cli`] the CLI driver.

pub mod austral_codegen;
pub mod ccg;
pub mod cli;
pub mod core_ir;
pub mod deltanet;
pub mod harper_gate;
pub mod l1;
pub mod lexicon;
pub mod translate;
