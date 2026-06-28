use candle_core::Device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{Hamiltonian, QuantumState};
use unfer_protocol::{EventPredicate, HamiltonianSpec, ModelSpec, PriorSpec};

use crate::build;
use crate::error::KernelError;
use crate::event;

/// A long-running probability kernel session.
///
/// Owns the current quantum state, Hamiltonian, and solver configuration.
/// Callers evolve the state, query event probabilities, and condition
/// (Bayesian update) on observed events.
#[derive(Debug)]
pub struct Session {
    state: QuantumState,
    hamiltonian: Hamiltonian,
    sirk_opts: SirkOpts,
    krylov_dim: usize,
    restarts: usize,
    device: Device,
    t_now: f64,
}

/// Result of an `evolve` call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvolveReport {
    pub t: f64,
    pub norm: f64,
    pub components: usize,
}

/// A snapshot of the current state's top-k components.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateSummary {
    pub norm: f64,
    pub components: usize,
    pub top: Vec<StateEntry>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StateEntry {
    pub state: String,
    pub probability: f64,
}

impl Session {
    /// Create a new session from a `ModelSpec`.
    pub fn new(spec: &ModelSpec) -> Result<Self, KernelError> {
        let hamiltonian = build::build_hamiltonian(&spec.hamiltonian)?;
        let state = build::build_prior(&spec.prior)?;
        let device = build::build_device(&spec.solver.device)?;
        let sirk_opts = SirkOpts {
            prune_eps: spec.solver.prune_eps,
            max_components: spec.solver.max_components,
            brst_tol: 1e-10,
            adaptive: spec.solver.adaptive,
        };
        Ok(Self {
            state,
            hamiltonian,
            sirk_opts,
            krylov_dim: spec.solver.krylov_dim,
            restarts: spec.solver.restarts.max(1),
            device,
            t_now: 0.0,
        })
    }

    /// Replace the current prior state. Resets evolution time to 0.
    pub fn set_prior(&mut self, p: &PriorSpec) -> Result<(), KernelError> {
        self.state = build::build_prior(p)?;
        self.t_now = 0.0;
        Ok(())
    }

    /// Replace the current Hamiltonian. The state is preserved.
    pub fn set_hamiltonian(&mut self, h: &HamiltonianSpec) -> Result<(), KernelError> {
        self.hamiltonian = build::build_hamiltonian(h)?;
        Ok(())
    }

    /// Evolve the state forward by time `t` using restarted SIRK.
    pub fn evolve(&mut self, t: f64) -> Result<EvolveReport, KernelError> {
        let psi = evolve_restarted(
            &self.hamiltonian,
            &self.state,
            t,
            self.restarts,
            self.krylov_dim,
            &self.device,
            None,
            &self.sirk_opts,
        )?;
        self.state = psi;
        self.t_now += t;
        let norm = self.state.norm();
        Ok(EvolveReport {
            t: self.t_now,
            norm,
            components: self.state.len(),
        })
    }

    /// Compute the Born-rule probability of event `e` under the current state.
    ///
    /// `P(E) = Σ_{s ⊨ E} |⟨s|ψ⟩|² / ‖ψ‖²`
    pub fn probability(&self, e: &EventPredicate) -> Result<f64, KernelError> {
        let norm_sq = QuantumState::inner_product(&self.state, &self.state).re;
        if norm_sq < 1e-30 {
            return Ok(0.0);
        }
        let mut mass = 0.0;
        for (outer, amp) in &self.state.components {
            if event::matches(outer, e) {
                mass += amp.norm_sqr();
            }
        }
        Ok(mass / norm_sq)
    }

    /// Condition the state on event `e` (Bayesian update).
    ///
    /// Zeroes non-matching components, renormalizes, and returns the prior
    /// probability `P(E)` that was computed before the update.
    /// Returns `KernelError::ZeroProbabilityCondition` if the matching mass
    /// is negligible.
    pub fn condition(&mut self, e: &EventPredicate) -> Result<f64, KernelError> {
        let norm_sq = QuantumState::inner_product(&self.state, &self.state).re;
        if norm_sq < 1e-30 {
            return Err(KernelError::ZeroProbabilityCondition { mass: 0.0 });
        }
        let mut mass = 0.0;
        self.state.components.retain(|outer, amp| {
            if event::matches(outer, e) {
                mass += amp.norm_sqr();
                true
            } else {
                false
            }
        });
        if mass < 1e-15 {
            return Err(KernelError::ZeroProbabilityCondition { mass });
        }
        let inv_norm = 1.0 / mass.sqrt();
        for amp in self.state.components.values_mut() {
            *amp *= inv_norm;
        }
        Ok(mass / norm_sq)
    }

    /// Return a snapshot of the `top_k` highest-probability components.
    pub fn snapshot(&self, top_k: usize) -> StateSummary {
        let norm = self.state.norm();
        let components = self.state.len();
        let mut top: Vec<StateEntry> = self
            .state
            .components
            .iter()
            .map(|(s, a)| StateEntry {
                state: format!("{:?}", s),
                probability: a.norm_sqr(),
            })
            .collect();
        top.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        top.truncate(top_k);
        StateSummary {
            norm,
            components,
            top,
        }
    }

    /// Current evolution time.
    pub fn t(&self) -> f64 {
        self.t_now
    }

    /// Current number of state components.
    pub fn n_components(&self) -> usize {
        self.state.len()
    }
}
