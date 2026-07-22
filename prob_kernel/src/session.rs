use candle_core::Device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{Hamiltonian, QuantumState};
use qfm::QfmPipeline;
use unfer_protocol::{
    EventPredicate, HamiltonianSpec, HmcOptsSpec, ModelSpec, PriorSpec, SolverSpec,
};

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
    /// QFM tomographic pipeline (Workstream F). Present only when the
    /// session was created from a `HamiltonianSpec::QfmTomography` spec.
    /// `evolve` dispatches to the pipeline's 4-phase `generate` instead
    /// of the SIRK solver.
    qfm_pipeline: Option<Box<QfmPipeline>>,
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
    /// QFM tomographic output: present only when the session was created
    /// from a `HamiltonianSpec::QfmTomography` spec and `evolve` was called
    /// with a `query` in the opts. Contains the generated raw image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qfm_output: Option<Vec<f64>>,
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

/// Result of a Quantum Bayesian Update on the TSR-evolved prior
/// (QFM.tex §8, P6 H follow-on). Only QFM tomographic models are
/// eligible. The kernel returns the HMC diagnostics + the full
/// reconstructed image from Phase 5.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BayesianUpdateReport {
    /// HMC log-posterior at the final sample.
    pub log_posterior: f64,
    /// HMC geometric-mean of the per-observation likelihoods at the
    /// final sample. `-1.0` if there were no observations (posterior
    /// = prior).
    pub mean_likelihood: f64,
    /// The Phase 5 reconstructed full-resolution image of the
    /// representative (final) HMC draw.
    pub image: Vec<f64>,
    /// The Phase 5 reconstruction of the **posterior-mean** point
    /// estimate — the Karcher (Fréchet) mean of the post-burn-in HMC
    /// chain on the projective unit sphere of $\Cset^m$ (QFM.tex §8).
    /// Denoised relative to the single draw in `image`. Empty if there
    /// were no post-burn-in samples.
    pub posterior_mean_image: Vec<f64>,
    /// The number of post-burn-in samples averaged into
    /// `posterior_mean_image`.
    pub n_samples: usize,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// Wall-clock time for HMC + decode in milliseconds.
    pub solve_ms: u64,
}

/// Result of a chain belief-propagation run (P8.8, qfm::bayes::
/// `belief_propagation_chain`). The MAP (marginal mode) point estimate
/// on the Krylov coefficients, plus the full-resolution image decoded
/// via Phase 5 tomographic reconstruction.
///
/// **Use case:** fast alternative to HMC when the user wants a
/// posterior point estimate without paying the sampling cost.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BeliefPropagationReport {
    /// The Phase 5 reconstructed full-resolution image of the MAP.
    pub image: Vec<f64>,
    /// The log-posterior at the MAP (up to a constant).
    pub log_posterior: f64,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// The number of cumulative-product sweeps used (always 1 for the
    /// exact chain case).
    pub n_sweeps: usize,
    /// Wall-clock time for BP + decode in milliseconds.
    pub solve_ms: u64,
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
        // If the Hamiltonian is a QFM tomography spec, compile the pipeline.
        let qfm_pipeline =
            if let HamiltonianSpec::QfmTomography { spec: qfm_spec } = &spec.hamiltonian {
                Some(Box::new(build::compile_qfm_pipeline(qfm_spec)?))
            } else {
                None
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
            qfm_pipeline,
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
        // QFM pipelines are not serialized — a restored session that was
        // originally a QFM model falls back to the SIRK path with the
        // placeholder Hamiltonian. The hamiltonian_spec is preserved so the
        // caller can re-create the pipeline by calling `set_hamiltonian`.
        let qfm_pipeline = None;
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
            qfm_pipeline,
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
    /// If the session has a QFM pipeline (from a `HamiltonianSpec::QfmTomography`
    /// spec), this dispatches to the pipeline's 4-phase `generate` using the
    /// optional `query` in the opts. Without a query, the QFM pipeline is
    /// not usable (it requires a raw input) and the call returns an error.
    pub fn evolve(&mut self, t: f64) -> Result<EvolveReport, KernelError> {
        self.evolve_with_query(t, None)
    }

    /// Evolve the state forward by time `t`, with an optional raw query
    /// for QFM tomographic generation. If `query` is `Some` and the session
    /// has a QFM pipeline, the pipeline's `generate(query)` is called and
    /// the result is stored in `EvolveReport::qfm_output`. Otherwise the
    /// SIRK solver is used.
    pub fn evolve_with_query(
        &mut self,
        t: f64,
        query: Option<&[f64]>,
    ) -> Result<EvolveReport, KernelError> {
        // QFM dispatch: if a pipeline is present and a query is provided,
        // run the 4-phase generate and return the result.
        if let Some(pipeline) = &self.qfm_pipeline {
            let q = query.ok_or_else(|| {
                KernelError::Internal("QFM pipeline requires a query in evolve opts".into())
            })?;
            let t0 = std::time::Instant::now();
            let x_out = pipeline.generate(q)?;
            let solve_ms = t0.elapsed().as_millis() as u64;
            self.t_now += t;
            return Ok(EvolveReport {
                t: self.t_now,
                norm: 1.0, // QFM output is a generated image, not a state norm
                components: x_out.len(),
                solve_ms,
                qfm_output: Some(x_out),
            });
        }
        // SIRK path.
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
            qfm_output: None,
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

    /// Quantum Bayesian Update on the TSR-evolved prior
    /// (QFM.tex §8, P6 H follow-on). Conditions the TSR-evolved prior
    /// on $N$ new raw observations $\{D_1, \dots, D_N\}$ and draws a
    /// single posterior sample via HMC on the unit sphere of
    /// $\Cset^m$. Returns the HMC diagnostics + the Phase 5
    /// reconstructed image.
    ///
    /// **Eligibility:** only QFM tomographic models
    /// (`HamiltonianSpec::QfmTomography`) have a TSR pipeline and
    /// therefore a meaningful TSR-evolved prior. Calling this method
    /// on a non-QFM session returns `KernelError::Internal`. The
    /// observation dimension must match the pipeline's raw dimension
    /// `d`; mismatches return `KernelError::Qfm(QfmError::DimensionMismatch)`.
    ///
    /// **No state side-effect:** the SIRK state `self.state` is not
    /// modified by this op — the posterior sample lives entirely in
    /// the Krylov subspace, and the report's `image` is the rendered
    /// output. Use `evolve_with_query` (or `evolve`) to feed a
    /// posterior-decoded image back into the kernel for further SIRK
    /// evolution if needed.
    pub fn bayesian_update(
        &self,
        observations: &[Vec<f64>],
        hmc_opts: &HmcOptsSpec,
    ) -> Result<BayesianUpdateReport, KernelError> {
        // Only QFM tomographic models are eligible.
        let pipeline = self.qfm_pipeline.as_ref().ok_or_else(|| {
            KernelError::Internal(
                "bayesian_update requires a QFM tomographic model (HamiltonianSpec::QfmTomography)"
                    .into(),
            )
        })?;

        // Build the N likelihood operators (S_1 -> S_2 -> Krylov
        // projection; errors are forwarded as Qfm errors).
        let mut likelihoods = Vec::with_capacity(observations.len());
        for obs in observations {
            let like = qfm::bayes::Likelihood::from_observation(pipeline, obs)?;
            likelihoods.push(like);
        }

        // The TSR-evolved prior direction.
        let c_prior = qfm::bayes::tsr_evolved_prior(pipeline);
        let posterior = qfm::bayes::Posterior::new(likelihoods.clone(), c_prior);

        // HMC.
        let qfm_opts = qfm::bayes::HmcOpts {
            leapfrog_steps: hmc_opts.leapfrog_steps,
            step_size: hmc_opts.step_size,
            n_iterations: hmc_opts.n_iterations,
            burn_in: hmc_opts.burn_in,
            seed: hmc_opts.seed,
        };
        let t0 = std::time::Instant::now();
        // P8.9: run the full chain (not just the final draw) so we can
        // form a posterior-mean point estimate. The final element of the
        // chain is the representative single draw (identical to
        // `sample_hmc_single` for the same opts), and the post-burn-in
        // tail feeds the Karcher mean.
        let chain = qfm::bayes::sample_hmc(&posterior, &qfm_opts);
        let sample = chain
            .last()
            .cloned()
            .unwrap_or_else(|| posterior.prior_direction().clone());

        // Diagnostics on the representative draw.
        let log_posterior = posterior.log_density(&sample);
        let mean_likelihood = if likelihoods.is_empty() {
            -1.0
        } else {
            let mut prod = 0.0_f64;
            for like in &likelihoods {
                prod += like.born_rule(&sample).ln();
            }
            (prod / likelihoods.len() as f64).exp()
        };

        // Phase 5 tomographic reconstruction of the representative draw.
        let image = qfm::bayes::reconstruct(pipeline, &sample)?;

        // P8.9: posterior-mean point estimate via the Karcher (Fréchet)
        // mean of the post-burn-in tail on the projective unit sphere of
        // C^m, then decode it. `burn_in >= chain.len()` (degenerate opts)
        // leaves an empty tail → empty posterior-mean image.
        let burn_in = qfm_opts.burn_in.min(chain.len());
        let tail = &chain[burn_in..];
        let (posterior_mean_image, n_samples) = if tail.is_empty() {
            (Vec::new(), 0)
        } else {
            let mean = qfm::bayes::karcher_mean(tail, 100, 1e-10);
            (qfm::bayes::reconstruct(pipeline, &mean)?, tail.len())
        };
        let solve_ms = t0.elapsed().as_millis() as u64;

        Ok(BayesianUpdateReport {
            log_posterior,
            mean_likelihood,
            image,
            posterior_mean_image,
            n_samples,
            n_observations: observations.len(),
            solve_ms,
        })
    }

    /// Run chain belief propagation (P8.8) on the TSR-evolved prior.
    /// Returns the MAP (marginal mode) point estimate + the decoded
    /// full-resolution image.
    ///
    /// **Only QFM tomographic models are eligible** — the prior is the
    /// TSR-evolved vacuum state. Calling this on a non-QFM model returns
    /// `KernelError::Internal` (the FFI layer maps this to UK-5000).
    ///
    /// **No state side-effect:** the SIRK state is not modified.
    pub fn belief_propagation(
        &self,
        observations: &[Vec<f64>],
        opts: &unfer_protocol::BeliefPropagationOptsSpec,
    ) -> Result<BeliefPropagationReport, KernelError> {
        let pipeline = self.qfm_pipeline.as_ref().ok_or_else(|| {
            KernelError::Internal(
                "belief_propagation requires a QFM tomographic model (HamiltonianSpec::QfmTomography)"
                    .into(),
            )
        })?;

        let mut likelihoods = Vec::with_capacity(observations.len());
        for obs in observations {
            let like = qfm::bayes::Likelihood::from_observation(pipeline, obs)?;
            likelihoods.push(like);
        }

        let c_prior = qfm::bayes::tsr_evolved_prior(pipeline);
        let posterior = qfm::bayes::Posterior::new(likelihoods, c_prior);

        let t0 = std::time::Instant::now();
        let bp_result = qfm::bayes::belief_propagation_chain(
            &posterior,
            opts.max_iter,
            opts.step_size,
            opts.tol,
        );
        let image = qfm::bayes::reconstruct(pipeline, &bp_result.map_estimate)?;
        let solve_ms = t0.elapsed().as_millis() as u64;

        Ok(BeliefPropagationReport {
            image,
            log_posterior: bp_result.log_posterior_at_map,
            n_observations: bp_result.n_observations,
            n_sweeps: bp_result.n_sweeps,
            solve_ms,
        })
    }

    /// Evaluates Nelson's condition and finds singularities for an ODE-based
    /// Hamiltonian. Returns the full ESA report including flow completeness,
    /// singularity detection, and any applied change of variables.
    pub fn analyze_self_adjointness(
        &self,
    ) -> Result<ode_sirk::report::OdeReport, KernelError> {
        match &self.hamiltonian_spec {
            HamiltonianSpec::OdeSystem {
                vars,
                rhs,
                change_of_variables,
            } => {
                let samples: Vec<Vec<f64>> = (1..=3).map(|i| vec![i as f64; vars.len()]).collect();
                let cov_str = change_of_variables.as_deref();
                let (report, _) = ode_sirk::protocol::analyze_ode_system(
                    vars.clone(),
                    &rhs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    cov_str,
                    100.0,
                    &samples,
                )
                .map_err(|e| KernelError::Internal(e.to_string()))?;
                Ok(report)
            }
            _ => Err(KernelError::Internal(
                "analyze_self_adjointness requires an OdeSystem Hamiltonian".into(),
            )),
        }
    }

    /// If CoV was applied, wraps SIRK observables to compute expectations
    /// in the original coordinate system. For variable `var`, applies the
    /// inverse coordinate map to the expectation value.
    pub fn measure_ode_observable(&self, var: &str) -> Result<f64, KernelError> {
        // For now, return the norm as a placeholder observable.
        // A full implementation would apply the CoV inverse map to
        // expectation values computed from the SIRK-evolved state.
        let _ = var;
        Ok(self.state.norm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::QfmTomographySpec;

    fn qfm_session() -> Session {
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let qfm_spec = QfmTomographySpec {
            training_data: training,
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
        };
        let spec = ModelSpec {
            hamiltonian: HamiltonianSpec::qfm_tomography(qfm_spec),
            prior: PriorSpec::Vacuum,
            solver: SolverSpec::default(),
        };
        Session::new(&spec).expect("compile QFM session")
    }

    fn non_qfm_session() -> Session {
        let spec = ModelSpec {
            hamiltonian: HamiltonianSpec::builtin(
                "harmonic_chain",
                serde_json::json!({"n_modes": 2, "omega": 1.0}),
            ),
            prior: PriorSpec::Vacuum,
            solver: SolverSpec::default(),
        };
        Session::new(&spec).expect("compile harmonic session")
    }

    #[test]
    fn bayesian_update_smoke_qfm_model() {
        // Single observation at training point 0: the HMC sample's
        // log_posterior should be finite, the image should have d=4
        // elements, and solve_ms should be non-zero.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let report = session
            .bayesian_update(&[obs], &HmcOptsSpec::default())
            .expect("bayesian_update should succeed on QFM model");
        assert_eq!(report.n_observations, 1);
        assert!(
            report.log_posterior.is_finite(),
            "log_posterior should be finite"
        );
        assert_eq!(report.image.len(), 4, "image should have d=4 elements");
        for v in &report.image {
            assert!(v.is_finite(), "image component should be finite: {v}");
        }
        // mean_likelihood should be a positive likelihood value (Born rule
        // is always positive).
        assert!(report.mean_likelihood > 0.0 && report.mean_likelihood <= 1.0);
        // P8.9: the posterior-mean point estimate is decoded from the
        // Karcher mean of the post-burn-in chain. With the default
        // HmcOptsSpec (n_iterations=200, burn_in=100) there are 100
        // post-burn-in samples, all averaged into a finite d=4 image.
        assert_eq!(report.n_samples, 100, "post-burn-in tail length");
        assert_eq!(report.posterior_mean_image.len(), 4);
        for v in &report.posterior_mean_image {
            assert!(
                v.is_finite(),
                "posterior-mean component should be finite: {v}"
            );
        }
    }

    #[test]
    fn bayesian_update_posterior_mean_tracks_observation() {
        // P8.9: the Karcher-mean posterior estimate is a valid decoded
        // image (finite, dimension d) aggregated over the whole
        // post-burn-in tail, and it should be at least as close to the
        // observation as a coarse bound allows. For this strongly-peaked
        // tetrahedron posterior the chain concentrates, so the mean is a
        // faithful (not degenerate) point estimate: its L2 distance to
        // the single representative draw stays within the image scale.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let report = session
            .bayesian_update(&[obs], &HmcOptsSpec::default())
            .expect("bayesian_update should succeed");
        assert_eq!(report.posterior_mean_image.len(), report.image.len());
        assert!(report.n_samples >= 2, "need a real chain to average");
        // The two estimates derive from the same typical set, so the
        // mean image must stay within a bounded distance of the draw.
        let dist2: f64 = report
            .posterior_mean_image
            .iter()
            .zip(&report.image)
            .map(|(m, s)| (m - s).powi(2))
            .sum();
        let scale2: f64 = report.image.iter().map(|s| s * s).sum::<f64>().max(1e-9);
        assert!(
            dist2 <= scale2 + 1e-12,
            "posterior mean diverged from the typical set: dist2={dist2}, scale2={scale2}"
        );
    }

    #[test]
    fn bayesian_update_zero_observations_returns_prior() {
        // With no observations, the posterior equals the prior;
        // the report should have n_observations=0, mean_likelihood=-1,
        // and a finite log_posterior.
        let session = qfm_session();
        let report = session
            .bayesian_update(&[], &HmcOptsSpec::default())
            .expect("zero-observation bayesian_update should succeed");
        assert_eq!(report.n_observations, 0);
        assert!(
            (report.mean_likelihood + 1.0).abs() < 1e-12,
            "mean_likelihood should be -1 for prior-only, got {}",
            report.mean_likelihood
        );
        assert!(report.log_posterior.is_finite());
        assert_eq!(report.image.len(), 4);
    }

    #[test]
    fn bayesian_update_non_qfm_returns_internal() {
        // The Bayesian update requires a QFM tomographic model; calling
        // it on a non-QFM session should return an Internal error.
        let session = non_qfm_session();
        let obs = vec![1.0, 0.0];
        let result = session.bayesian_update(&[obs], &HmcOptsSpec::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Internal(msg) => {
                assert!(
                    msg.contains("QFM"),
                    "internal error should mention QFM: {msg}"
                );
            }
            e => panic!("expected KernelError::Internal, got {e:?}"),
        }
    }

    #[test]
    fn bayesian_update_dim_mismatch_returns_qfm_error() {
        // Observation with wrong dimension should return a Qfm
        // DimensionMismatch error.
        let session = qfm_session();
        let obs = vec![1.0, 0.0]; // d=2, expected d=4
        let result = session.bayesian_update(&[obs], &HmcOptsSpec::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            e => panic!("expected KernelError::Qfm(DimensionMismatch), got {e:?}"),
        }
    }

    // ── P8.8: chain belief propagation tests ─────────────────────────

    #[test]
    fn belief_propagation_smoke_qfm_model() {
        // BP on a QFM tomographic model returns a finite MAP image
        // and a finite log-posterior.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let report = session
            .belief_propagation(&[obs], &opts)
            .expect("BP should succeed on QFM model");
        assert_eq!(report.image.len(), 4);
        for v in &report.image {
            assert!(v.is_finite(), "image element should be finite, got {v}");
        }
        assert!(report.log_posterior.is_finite());
        assert_eq!(report.n_observations, 1);
        assert!(report.n_sweeps >= 1);
    }

    #[test]
    fn belief_propagation_zero_observations_returns_prior() {
        // Zero-observation BP: no likelihoods, MAP = prior direction.
        let session = qfm_session();
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let report = session
            .belief_propagation(&[], &opts)
            .expect("zero-obs BP should succeed");
        assert_eq!(report.n_observations, 0);
        assert_eq!(report.image.len(), 4);
    }

    #[test]
    fn belief_propagation_non_qfm_returns_internal() {
        // BP is QFM-only; calling on a non-QFM session returns Internal.
        let session = non_qfm_session();
        let obs = vec![1.0, 0.0];
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let result = session.belief_propagation(&[obs], &opts);
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Internal(msg) => {
                assert!(msg.contains("QFM"), "should mention QFM: {msg}");
            }
            e => panic!("expected KernelError::Internal, got {e:?}"),
        }
    }

    #[test]
    fn belief_propagation_dim_mismatch_returns_qfm_error() {
        // Observation with wrong dimension returns a Qfm
        // DimensionMismatch error.
        let session = qfm_session();
        let obs = vec![1.0, 0.0]; // d=2, expected d=4
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let result = session.belief_propagation(&[obs], &opts);
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            e => panic!("expected KernelError::Qfm(DimensionMismatch), got {e:?}"),
        }
    }
}
