//! Born-rule probability layer over the unfer QFT kernel.
//!
//! [`Session`] is the public API: `evolve`, `probability`, `condition`,
//! `snapshot`, and the Bayesian-update path. [`build`] hosts the model-spec
//! compiler (Paulí–Grover, diffusion, random-start Hamiltonian types),
//! [`event`] the kernel-event (Born-rule observation) types, and [`error`]
//! the `KernelError` diagnostics. Communication with the outside world uses
//! [`unfer_protocol`] types.
//!
//! [`Session`]: session::Session

pub mod build;
pub mod error;
pub mod event;
pub mod logos;
pub mod session;
pub mod symbolic;
pub mod verify;
pub mod whyml;

pub use error::KernelError;
pub use logos::logos_compile;
pub use session::{
    BayesianUpdateReport, EvolveReport, SESSION_FORMAT_VERSION, Session, SessionBlob, SessionEvent,
    SessionEventSpec, SessionOp, StateEntry, StateSummary,
};
pub use symbolic::symbolic_derive;
pub use unfer_protocol;
pub use verify::verify_export;
pub use whyml::{whyml_emit, why3_available};
