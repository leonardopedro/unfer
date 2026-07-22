use candle_core::Device;
#[cfg(feature = "latex")]
use nested_fock_algebra::compile_latex;
#[cfg(feature = "latex")]
use nested_fock_algebra::compile_typst_math;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QFM_DEFAULT_QUANTIZATION_SCALE,
    QuantumState, bose_hubbard_chain, gravity_hamiltonian, harmonic_chain,
    navier_stokes_hamiltonian, qfm_hamiltonian, qfm_hamiltonian_localized,
    qfm_hamiltonian_mehler_projector, qfm_hamiltonian_mehler_projector_localized,
    yang_mills_hamiltonian, yang_mills_lattice,
};
use num_complex::Complex64;
use qfm::{QfmConfig, QfmPipeline};
use unfer_protocol::{
    DeviceSpec, HamiltonianSpec, Level, OpKind, OpSpec, PriorSpec, QfmTomographySpec,
};

use crate::error::KernelError;

/// Build a [`Hamiltonian`] from a protocol [`HamiltonianSpec`].
///
/// - `Builtin` dispatches to the corresponding model function.
/// - `Latex` parses via `compile_latex` (requires the `latex` feature).
/// - `Typst` parses via `compile_typst_math` (P8.7, requires the `latex`
///   feature for the underlying CAS layer).
/// - `Terms` constructs directly (explosion-safe path).
pub fn build_hamiltonian(spec: &HamiltonianSpec) -> Result<Hamiltonian, KernelError> {
    match spec {
        HamiltonianSpec::Builtin { name, params } => match name.as_str() {
            "yang_mills" => {
                let g = get_f64(params, "g")?;
                Ok(yang_mills_hamiltonian(g))
            }
            "navier_stokes" => {
                let nu = get_f64(params, "nu")?;
                Ok(navier_stokes_hamiltonian(nu))
            }
            "gravity" => Ok(gravity_hamiltonian()),
            "harmonic_chain" => {
                let n_modes = get_u64(params, "n_modes")? as usize;
                let omega = get_f64(params, "omega")?;
                Ok(harmonic_chain(n_modes, omega))
            }
            "bose_hubbard" => {
                let n_modes = get_u64(params, "n_modes")? as usize;
                let t = get_f64(params, "t")?;
                let u = get_f64(params, "u")?;
                let periodic = get_bool_or(params, "periodic", false);
                Ok(bose_hubbard_chain(n_modes, t, u, periodic))
            }
            "yang_mills_lattice" => {
                let l = get_u64(params, "l")? as usize;
                let g = get_f64(params, "g")?;
                let n_colors = get_u64_or(params, "n_colors", 1) as usize;
                Ok(yang_mills_lattice(l, g, n_colors))
            }
            "qfm_mehler" => {
                let alphas = get_f64_array(params, "alphas")?;
                Ok(qfm_hamiltonian(&alphas))
            }
            // Localized QFM (QFM.tex, "The data-channel wave-function on the
            // hypersphere"): each data point is encoded on its own D inner
            // modes (one per real coordinate) instead of by array index
            // alone — the direct realization of "D of the (infinitely many)
            // hyperspherical coordinates localized around the point, the
            // rest uniform." Requires the actual point coordinates, not
            // just pre-reduced alphas.
            "qfm_mehler_localized" => {
                let points = get_points_array(params, "points")?;
                let alphas = get_f64_array(params, "alphas")?;
                let scale = get_f64_or(params, "scale", QFM_DEFAULT_QUANTIZATION_SCALE);
                Ok(qfm_hamiltonian_localized(&points, &alphas, scale))
            }
            // Exact off-diagonal QFM generator (QFM.tex §13): the rank-1
            // projector |0><0| onto the dressed Mehler vacuum. Because the
            // vacuum is non-orthogonal to the data channels (overlap ε_j
            // from finite localization), this projector alone drives all
            // off-diagonal coupling — no truncation, no explicit coupling
            // terms; applied via the O(M) rank-1 shortcut H|s> = <0|s>·|0>.
            "qfm_mehler_projector" => {
                let epsilons = get_f64_array(params, "epsilons")?;
                check_epsilons(&epsilons)?;
                Ok(qfm_hamiltonian_mehler_projector(&epsilons))
            }
            // Same exact generator with each data channel localized on its
            // own D inner modes via point_to_inner_state (the literal
            // data-channel encoding), instead of identified by array index.
            "qfm_mehler_projector_localized" => {
                let points = get_points_array(params, "points")?;
                let epsilons = get_f64_array(params, "epsilons")?;
                check_epsilons(&epsilons)?;
                let scale = get_f64_or(params, "scale", QFM_DEFAULT_QUANTIZATION_SCALE);
                Ok(qfm_hamiltonian_mehler_projector_localized(
                    &points, &epsilons, scale,
                ))
            }
            other => Err(KernelError::UnknownBuiltinModel {
                name: other.to_string(),
            }),
        },

        HamiltonianSpec::Latex { latex } => {
            #[cfg(feature = "latex")]
            {
                Ok(compile_latex(latex))
            }
            #[cfg(not(feature = "latex"))]
            {
                let _ = latex;
                Err(KernelError::Internal(
                    "latex feature not enabled; rebuild prob_kernel with --features latex".into(),
                ))
            }
        }

        HamiltonianSpec::Typst { typst } => {
            #[cfg(feature = "latex")]
            {
                Ok(compile_typst_math(typst))
            }
            #[cfg(not(feature = "latex"))]
            {
                let _ = typst;
                Err(KernelError::Internal(
                    "latex feature not enabled; rebuild prob_kernel with --features latex".into(),
                ))
            }
        }

        HamiltonianSpec::Terms { terms } => {
            if terms.is_empty() {
                return Err(KernelError::BadTerms {
                    reason: "empty terms list".into(),
                });
            }
            let h_terms: Vec<(Complex64, Vec<Operator>)> = terms
                .iter()
                .map(|t| {
                    let coeff = Complex64::new(t.coeff_re, t.coeff_im);
                    let ops: Result<Vec<Operator>, KernelError> =
                        t.ops.iter().map(op_spec_to_operator).collect();
                    Ok::<_, KernelError>((coeff, ops?))
                })
                .collect::<Result<_, _>>()?;
            Ok(Hamiltonian { terms: h_terms })
        }

        HamiltonianSpec::QfmTomography { spec } => {
            // The QFM pipeline is compiled separately (via `compile_qfm_pipeline`)
            // and stored in the Session. For the SIRK-path Hamiltonian, return
            // a minimal placeholder (the vacuum projector) so the Session has
            // *some* Hamiltonian. The QFM pipeline overrides `evolve` entirely.
            let _ = spec;
            Ok(Hamiltonian {
                terms: vec![(Complex64::new(1.0, 0.0), vec![Operator::ProjectVacuum])],
            })
        }

        HamiltonianSpec::OdeSystem {
            vars,
            rhs,
            change_of_variables,
        } => {
            // Run the full ODE → Weyl → ESA pipeline from ode_sirk.
            // For the SIRK-path Hamiltonian, we use the (possibly CoV-transformed)
            // system.  The ESA report is available via `analyze_ode_system` if the
            // caller needs it; here we only extract the Hamiltonian.
            let rhs_refs: Vec<&str> = rhs.iter().map(|s| s.as_str()).collect();
            // Minimal sample points (origin) for flow analysis — the caller should
            // use `analyze_ode_system` directly for a proper ESA check.
            let samples: Vec<Vec<f64>> = vec![vec![0.1; vars.len()]];
            let (_report, hamiltonian) = ode_sirk::analyze_ode_system(
                vars.clone(),
                &rhs_refs,
                change_of_variables.as_deref(),
                100.0,
                &samples,
            )
            .map_err(|e| KernelError::Internal(format!("ode_sirk: {}", e)))?;
            Ok(hamiltonian)
        }
    }
}

/// Compile a [`QfmPipeline`] from a [`QfmTomographySpec`].
pub fn compile_qfm_pipeline(spec: &QfmTomographySpec) -> Result<QfmPipeline, KernelError> {
    if spec.training_data.is_empty() {
        return Err(KernelError::BadBuiltinParams {
            reason: "QfmTomographySpec requires non-empty training_data".into(),
        });
    }
    let config = QfmConfig {
        k: spec.k,
        k2: spec.k2,
        krylov_dim: spec.krylov_dim,
        seed: spec.seed,
        n_t_samples: 10,
        noise_dim: spec.training_data[0].len(),
        max_rank: None,
        ..Default::default()
    };
    let pipeline = QfmPipeline::compile(&spec.training_data, &config)?;
    Ok(pipeline)
}

/// Convert a protocol [`OpSpec`] to a kernel [`Operator`].
fn op_spec_to_operator(spec: &OpSpec) -> Result<Operator, KernelError> {
    Ok(match (spec.kind, spec.level) {
        (OpKind::Create, Level::InnerBoson) => Operator::InnerBosonCreate(spec.mode),
        (OpKind::Annihilate, Level::InnerBoson) => Operator::InnerBosonAnnihilate(spec.mode),
        (OpKind::Create, Level::InnerFermion) => Operator::InnerFermionCreate(spec.mode),
        (OpKind::Annihilate, Level::InnerFermion) => Operator::InnerFermionAnnihilate(spec.mode),
        (OpKind::Create, Level::OuterBoson) => {
            let mut s = InnerBosonicState::vacuum();
            s.modes.insert(spec.mode, 1);
            Operator::OuterBosonCreate(s)
        }
        (OpKind::Annihilate, Level::OuterBoson) => {
            let mut s = InnerBosonicState::vacuum();
            s.modes.insert(spec.mode, 1);
            Operator::OuterBosonAnnihilate(s)
        }
        (OpKind::Create, Level::OuterFermion) => {
            let mut s = InnerFermionicState::vacuum();
            s.modes.insert(spec.mode);
            Operator::OuterFermionCreate(s)
        }
        (OpKind::Annihilate, Level::OuterFermion) => {
            let mut s = InnerFermionicState::vacuum();
            s.modes.insert(spec.mode);
            Operator::OuterFermionAnnihilate(s)
        }
    })
}

/// Build a [`QuantumState`] prior from a protocol [`PriorSpec`].
pub fn build_prior(spec: &PriorSpec) -> Result<QuantumState, KernelError> {
    match spec {
        PriorSpec::Vacuum => Ok(QuantumState::vacuum()),

        PriorSpec::Bosons { modes } => {
            let mut state = QuantumState::vacuum()
                .apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
            for &(mode, count) in modes {
                for _ in 0..count {
                    state = state.apply(&Operator::InnerBosonCreate(mode));
                }
            }
            Ok(state)
        }

        PriorSpec::Fermions { modes } => {
            let mut state = QuantumState::vacuum()
                .apply(&Operator::OuterFermionCreate(InnerFermionicState::vacuum()));
            for &mode in modes {
                state = state.apply(&Operator::InnerFermionCreate(mode));
            }
            Ok(state)
        }

        PriorSpec::Superposition { terms } => {
            if terms.is_empty() {
                return Err(KernelError::BadPrior {
                    reason: "empty superposition".into(),
                });
            }
            let mut state = QuantumState::zero();
            for term in terms {
                let sub = build_prior(&term.spec)?;
                let scale = Complex64::new(term.re, term.im);
                state.scale_and_add(&sub, scale);
            }
            let norm = state.norm();
            if norm < 1e-15 {
                return Err(KernelError::BadPrior {
                    reason: "superposition has zero norm".into(),
                });
            }
            for amp in state.components.values_mut() {
                *amp /= norm;
            }
            Ok(state)
        }
    }
}

/// Build a candle [`Device`] from a protocol [`DeviceSpec`].
pub fn build_device(spec: &DeviceSpec) -> Result<Device, KernelError> {
    match spec {
        DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Cuda { device_id } => {
            #[cfg(feature = "cuda")]
            {
                match Device::cuda_if_available(*device_id as usize) {
                    Ok(d) => Ok(d),
                    Err(e) => Err(KernelError::Internal(format!(
                        "CUDA device {} unavailable: {}",
                        device_id, e
                    ))),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(KernelError::Internal(format!(
                    "CUDA not compiled in (device_id={}); rebuild fock_sirk with --features cuda",
                    device_id
                )))
            }
        }
    }
}

fn get_f64(params: &serde_json::Value, key: &str) -> Result<f64, KernelError> {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| KernelError::BadBuiltinParams {
            reason: format!("missing or non-numeric parameter: {key}"),
        })
}

fn get_u64(params: &serde_json::Value, key: &str) -> Result<u64, KernelError> {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| KernelError::BadBuiltinParams {
            reason: format!("missing or non-integer parameter: {key}"),
        })
}

/// Read an optional boolean parameter, falling back to `default` when absent.
fn get_bool_or(params: &serde_json::Value, key: &str, default: bool) -> bool {
    params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

/// Read an optional unsigned-integer parameter, falling back to `default`.
fn get_u64_or(params: &serde_json::Value, key: &str, default: u64) -> u64 {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

/// Read an optional numeric parameter, falling back to `default`.
fn get_f64_or(params: &serde_json::Value, key: &str, default: f64) -> f64 {
    params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Validate vacuum-channel overlaps for the exact Mehler projector:
/// `Σ ε_j² ≤ 1` (the ε² are uniform-measure masses of disjoint packet
/// supports, so exceeding 1 is physically impossible).
fn check_epsilons(epsilons: &[f64]) -> Result<(), KernelError> {
    let sum_sq: f64 = epsilons.iter().map(|e| e * e).sum();
    if sum_sq > 1.0 {
        Err(KernelError::BadBuiltinParams {
            reason: format!(
                "epsilons must satisfy Σ ε_j² ≤ 1 (the ε_j are vacuum-channel \
                 overlaps Π_i sqrt(w_i/2π) of disjoint packets); got Σ ε² = {sum_sq}"
            ),
        })
    } else {
        Ok(())
    }
}

/// Read a required array-of-numbers parameter (e.g. the QFM `alphas` weights).
fn get_f64_array(params: &serde_json::Value, key: &str) -> Result<Vec<f64>, KernelError> {
    let arr = params.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
        KernelError::BadBuiltinParams {
            reason: format!("missing or non-array parameter: {key}"),
        }
    })?;
    arr.iter()
        .map(|v| {
            v.as_f64().ok_or_else(|| KernelError::BadBuiltinParams {
                reason: format!("non-numeric element in array parameter: {key}"),
            })
        })
        .collect()
}

/// Read a required array-of-points parameter (e.g. the localized QFM
/// `points` list, one `D`-dim real vector per data channel).
fn get_points_array(params: &serde_json::Value, key: &str) -> Result<Vec<Vec<f64>>, KernelError> {
    let arr = params.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
        KernelError::BadBuiltinParams {
            reason: format!("missing or non-array parameter: {key}"),
        }
    })?;
    arr.iter()
        .map(|point| {
            point
                .as_array()
                .ok_or_else(|| KernelError::BadBuiltinParams {
                    reason: format!("non-array point in array parameter: {key}"),
                })?
                .iter()
                .map(|v| {
                    v.as_f64().ok_or_else(|| KernelError::BadBuiltinParams {
                        reason: format!("non-numeric coordinate in array parameter: {key}"),
                    })
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[cfg(feature = "latex")]
mod tests {
    use super::*;
    use unfer_protocol::HamiltonianSpec;

    #[test]
    fn build_hamiltonian_typst_oscillator() {
        // P8.7: `HamiltonianSpec::Typst` dispatches through
        // `compile_typst_math` and produces a non-empty Hamiltonian for
        // a two-mode oscillator.
        let h = build_hamiltonian(&HamiltonianSpec::typst(
            "a^dagger_0 * a_0 + a^dagger_1 * a_1",
        ))
        .expect("Typst compile should succeed");
        assert_eq!(h.terms.len(), 2);
    }

    #[test]
    fn build_hamiltonian_typst_with_greek_coefficient() {
        // Greek coefficient (`\omega`) is passed through to the CAS layer
        // as `omega` (the compiler strips the leading backslash).
        let h = build_hamiltonian(&HamiltonianSpec::typst("0.5 * \\omega * a^dagger_0 * a_0"))
            .expect("Typst compile with coefficient should succeed");
        assert_eq!(h.terms.len(), 1);
    }

    #[test]
    fn build_hamiltonian_typst_outer_op() {
        // `A^dagger_0 * A_0` → outer create + outer annihilate.
        let h = build_hamiltonian(&HamiltonianSpec::typst("A^dagger_0 * A_0"))
            .expect("Typst outer op should compile");
        assert_eq!(h.terms.len(), 1);
    }

    #[test]
    fn build_hamiltonian_typst_empty_for_zero() {
        // `0` produces an empty Hamiltonian (no terms).
        let h = build_hamiltonian(&HamiltonianSpec::typst("0")).expect("Typst `0` should compile");
        assert!(h.terms.is_empty());
    }

    #[test]
    fn build_hamiltonian_latex_dispatch_still_works() {
        // Sanity: the existing `Latex` variant is unaffected by the P8.7
        // addition.
        let h = build_hamiltonian(&HamiltonianSpec::latex(r"\frac{1}{2} * c_0 * a_0"))
            .expect("Latex dispatch should still work");
        assert!(!h.terms.is_empty());
    }
}
