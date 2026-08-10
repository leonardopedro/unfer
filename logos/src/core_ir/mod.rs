pub mod compiler;
pub mod linearity;
pub mod types;

pub use compiler::compile_to_core_ir;
pub use linearity::{LinearityError, check_linearity, insert_linearity};
pub use types::*;
