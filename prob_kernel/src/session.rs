use candle_core::Device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{Hamiltonian, QuantumState};
use unfer_protocol::{EventPredicate, HamiltonianSpec, ModelSpec, PriorSpec, SolverSpec};

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
    // Stored specs for snapshot/restore — updated by set_hamiltonian.
    hamiltonian_spec: HamiltonianSpec,
    solver_spec: SolverSpec,
}

/// Serializable snapshot of a Session for save/restore.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionBlob {
    pub hamiltonian_spec: HamiltonianSpec,
    pub solver_spec: SolverSpec,
    pub state: QuantumState,
    pub t_now: f64,
}

/// Result of an `evolve` call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvolveReport {
    pub t: f64,
    pub norm: f64,
    pub components: usize,
    /// Wall-clock time for the SIRK solve in milliseconds.
    pub solve_ms: u64,
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
            hamiltonian_spec: spec.hamiltonian.clone(),
            solver_spec: spec.solver.clone(),
        })
    }

    /// Restore a session from a previously saved `SessionBlob`.
    pub fn restore(blob: SessionBlob) -> Result<Self, KernelError> {
        let hamiltonian = build::build_hamiltonian(&blob.hamiltonian_spec)?;
        let device = build::build_device(&blob.solver_spec.device)?;
        let sirk_opts = SirkOpts {
            prune_eps: blob.solver_spec.prune_eps,
            max_components: blob.solver_spec.max_components,
            brst_tol: 1e-10,
            adaptive: blob.solver_spec.adaptive,
        };
        Ok(Self {
            state: blob.state,
            hamiltonian,
            sirk_opts,
            krylov_dim: blob.solver_spec.krylov_dim,
            restarts: blob.solver_spec.restarts.max(1),
            device,
            t_now: blob.t_now,
            hamiltonian_spec: blob.hamiltonian_spec,
            solver_spec: blob.solver_spec,
        })
    }

    /// Serialize the current session state to a `SessionBlob` for persistence.
    pub fn save(&self) -> SessionBlob {
        SessionBlob {
            hamiltonian_spec: self.hamiltonian_spec.clone(),
            solver_spec: self.solver_spec.clone(),
            state: self.state.clone(),
            t_now: self.t_now,
        }
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
        self.hamiltonian_spec = h.clone();
        Ok(())
    }

    /// Evolve the state forward by time `t` using restarted SIRK.
    pub fn evolve(&mut self, t: f64) -> Result<EvolveReport, KernelError> {
        let t0 = std::time::Instant::now();
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
        let solve_ms = t0.elapsed().as_millis() as u64;
        self.state = psi;
        self.t_now += t;
        let norm = self.state.norm();
        Ok(EvolveReport {
            t: self.t_now,
            norm,
            components: self.state.len(),
            solve_ms,
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
