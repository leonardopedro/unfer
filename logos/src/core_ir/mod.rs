pub mod types;
pub mod compiler;
pub mod linearity;

pub use types::*;
pub use compiler::compile_to_core_ir;
pub use linearity::{insert_linearity, check_linearity, LinearityError};
