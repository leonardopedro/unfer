use quantrs2_symengine_pure::Expression;
use crate::{
    inner_boson_create, inner_boson_annihilate, 
    inner_fermion_create, inner_fermion_annihilate
};

/// Represents a Classical/Quantum Field $\phi(x) = a^\dagger + a$
pub fn hermitian_field(mode: u32) -> Expression {
    inner_boson_create(mode) + inner_boson_annihilate(mode)
}

/// Represents Conjugate Momentum $\pi(x) = i(a^\dagger - a)$
pub fn conjugate_momentum(mode: u32) -> Expression {
    // "I" is parsed as Complex64::i() by your cas.rs SExpr parser
    Expression::symbol("I") * (inner_boson_create(mode) - inner_boson_annihilate(mode))
}

/// Represents a Majorana Spinor (Real representation, particle is its own antiparticle)
/// Eq (50) and Section "Majorana spinors in canonical quantization"
pub fn majorana_fermion(mode: u32) -> Expression {
    inner_fermion_create(mode) + inner_fermion_annihilate(mode)
}

/// BRST Ghost Fields (treated as standard fermions breaking Lorentz invariance for the vacuum)
/// As described in the "Timepiece and the Gribov ambiguity" section.
pub fn ghost_field(mode: u32) -> Expression {
    inner_fermion_create(mode)
}

pub fn ghost_conjugate(mode: u32) -> Expression {
    inner_fermion_annihilate(mode)
}
