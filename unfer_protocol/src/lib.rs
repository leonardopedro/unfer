pub mod codes;
pub mod types;

pub use codes::*;
pub use types::*;

#[derive(Debug, thiserror::Error)]
#[error("protocol error {}: {}", diagnostic.code, diagnostic.message)]
pub struct ProtocolError {
    pub diagnostic: Diagnostic,
}

impl ProtocolError {
    pub fn new(diagnostic: Diagnostic) -> Self {
        Self { diagnostic }
    }
}

impl From<Diagnostic> for ProtocolError {
    fn from(diagnostic: Diagnostic) -> Self {
        Self::new(diagnostic)
    }
}
