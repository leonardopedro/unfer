use candle_core::Device;
#[cfg(feature = "latex")]
use nested_fock_algebra::compile_latex;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, InnerFermionicState, Operator, QuantumState,
    bose_hubbard_chain, gravity_hamiltonian, harmonic_chain, navier_stokes_hamiltonian,
    yang_mills_hamiltonian, yang_mills_lattice,
};
use num_complex::Complex64;
use unfer_protocol::{DeviceSpec, HamiltonianSpec, Level, OpKind, OpSpec, PriorSpec};

use crate::error::KernelError;

/// Build a [`Hamiltonian`] from a protocol [`HamiltonianSpec`].
///
/// - `Builtin` dispatches to the corresponding model function.
/// - `Latex` parses via `compile_latex` (requires the `latex` feature).
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
    }
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
