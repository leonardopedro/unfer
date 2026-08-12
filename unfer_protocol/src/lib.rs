//! Shared serde contract for the unfer kernel: types, UK-#### codes, and
//! repair hints.
//!
//! This is the cross-crate language of the system — `prob_kernel`,
//! `unfer_ffi`, `unfer_edge`, `unfer_consensus`, and the velysterm agent all
//! speak it. [`codes`] owns the `UK-####` diagnostic catalogue (severity +
//! repair hint), [`ops`] the shared op-name registry, [`types`] the wire
//! types (model specs, sessions, kernel events, consensus transactions),
//! and [`archive`] the deprecated-op archive.
//!
//! ```
//! assert_eq!(unfer_protocol::KERNEL_VERSION, 1);
//! ```

pub mod archive;
pub mod codes;
pub mod ops;
pub mod types;

pub use archive::*;
pub use codes::*;
pub use ops::*;
pub use types::*;

/// The kernel ABI/protocol version served by `uk_version()`.
pub const KERNEL_VERSION: i64 = 1;

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
