//! Code shared by the `phi4_momentum_*` examples. A directory module rather
//! than `examples/common.rs`, which cargo would treat as an example target.
//!
//! Both examples need the same exact window reference: the sweep evaluates it
//! beside every cell, and the probe times it to split a cell into solve and
//! reference.
#![allow(dead_code)]

pub mod window_reference;
