pub mod brst;
pub mod device;
pub mod evolve;
pub mod forward_sirk;
pub mod linalg;
pub mod registry;
pub mod tensor_state;

pub use device::best_device;
pub use evolve::evolve_restarted;
pub use forward_sirk::{
    ForwardSirkResult, SirkOpts, solve_forward_sirk, solve_forward_sirk_with_opts,
};
pub use linalg::{GRAM_REL_TOL, SirkError, Whitening, whiten_gram};
pub use registry::StateDictionary;
pub use tensor_state::TensorState;

use nested_fock_algebra::{Hamiltonian, QuantumState};
use num_complex::Complex64;

/// Legacy MatrixFreeOperator trait for backward compatibility.
pub trait MatrixFreeOperator: Sync {
    fn apply(&self, x: &QuantumState) -> QuantumState;

    fn inner_product(a: &QuantumState, b: &QuantumState) -> Complex64 {
        QuantumState::inner_product(a, b)
    }

    fn scale_and_add(a: &mut QuantumState, b: &QuantumState, scale: Complex64) {
        a.scale_and_add(b, scale);
    }

    fn norm(a: &QuantumState) -> f64 {
        QuantumState::inner_product(a, a).re.sqrt()
    }
}

impl MatrixFreeOperator for Hamiltonian {
    fn apply(&self, x: &QuantumState) -> QuantumState {
        self.apply(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use nested_fock_algebra::{InnerBosonicState, OuterState};

    #[test]
    fn test_state_registry() {
        let mut registry = StateDictionary::new();
        let mut s1 = OuterState::vacuum();
        let mut b1 = InnerBosonicState::vacuum();
        b1.modes.insert(0, 1);
        s1.bosonic.insert(b1, 1);

        let idx1 = registry.get_or_insert(s1.clone());
        let idx2 = registry.get_or_insert(OuterState::vacuum());
        let idx3 = registry.get_or_insert(s1);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 0);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_tensor_inner_product() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let mut registry = StateDictionary::new();

        let mut qs1 = QuantumState::vacuum();
        let mut s1 = OuterState::vacuum();
        let mut b1 = InnerBosonicState::vacuum();
        b1.modes.insert(0, 1);
        s1.bosonic.insert(b1, 1);
        qs1.components.insert(s1, Complex64::new(0.5, 0.5));

        let mut qs2 = QuantumState::vacuum();
        qs2.components
            .insert(OuterState::vacuum(), Complex64::new(1.0, 0.0));

        // Pre-register all states
        registry.register(&qs1);
        registry.register(&qs2);

        let ts1 = TensorState::from_quantum_state(&qs1, &mut registry, &device)?;
        let ts2 = TensorState::from_quantum_state(&qs2, &mut registry, &device)?;

        let prod = ts1.inner_product(&ts2)?;
        // <qs1 | qs2> = qs1[vac]* qs2[vac] = 1.0* * 1.0 = 1.0
        // QuantumState::vacuum() starts with 1.0 at vac.
        assert!((prod.re - 1.0).abs() < 1e-10);
        assert!(prod.im.abs() < 1e-10);

        Ok(())
    }
}
