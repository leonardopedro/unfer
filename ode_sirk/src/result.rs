use thiserror::Error;

#[derive(Debug, Error)]
pub enum OdeError {
    #[error("parse error at position {pos}: {msg}")]
    ParseError { pos: usize, msg: String },
    #[error("mismatched vars/rhs lengths ({0} vars, {1} rhs)")]
    MismatchedLengths(usize, usize),
    #[error("variable index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("polynomial degree {0} exceeds explosion bound {1}")]
    PolynomialTooLarge(u32, u32),
    #[error("flow integration: {0}")]
    FlowIntegration(String),
    #[error("no singularity found (flow is complete)")]
    NoSingularity,
    #[error("change of variables failed: {0}")]
    CovFailed(String),
    #[error("internal: {0}")]
    Internal(String),
}

pub type OdeSirkResult<T> = Result<T, OdeError>;
