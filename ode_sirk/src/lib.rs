//! The `ode_sirk` crate implements a pipeline for transforming ODEs into
//! self-adjoint quantum Hamiltonians using Koopman-Weyl quantization,
//! detecting singularities via Nelson's theorem, and applying coordinate
//! transformations to enable SIRK evolution.
//!
//! Key transformations:
//! - ODE → Weyl-symmetrized Hamiltonian (strict bosonic mapping)
//! - Essential Self-Adjointness (ESA) detection via classical flow completeness
//! - Singularity localization and coordinate transformation for blow-up
//! - Observables mapping back to original coordinates
//! - Integration with `prob_kernel` for hashimoto-SIRK evolution

pub mod result;
pub use result::OdeSirkResult;

pub mod poly;
pub use poly::NormalOrderedOp;

pub mod ode;
pub use ode::ODESystem;

pub mod hamiltonian;
pub use hamiltonian::{ode_to_hamiltonian, Hamiltonian};

pub mod flow;
pub use flow::{FlowAnalysis, EscapeEvent};

pub mod singularity;
pub use singularity::{SingularityReport, SingularityType};

pub mod change_of_vars;
pub use change_of_vars::{CoV, TransformedSystem};

pub mod esa;
pub use esa::{EsaReport, EsaStatus};

pub mod protocol;
pub use protocol::{analyze_ode_system, analyze_esa};

pub mod report;
pub use report::OdeReport;
