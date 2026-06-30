use unfer_protocol::*;

fn rt<T>(v: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(v).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(v, &back, "round-trip mismatch for json: {json}");
    back
}

fn sample_term() -> TermSpec {
    TermSpec::new(
        1.5,
        -0.25,
        vec![
            OpSpec::new(OpKind::Create, Level::InnerBoson, 0),
            OpSpec::new(OpKind::Annihilate, Level::InnerBoson, 0),
        ],
    )
}

fn sample_prior() -> PriorSpec {
    PriorSpec::superposition(vec![
        SuperpositionTerm::new(0.6, 0.0, PriorSpec::Vacuum),
        SuperpositionTerm::new(0.4, 0.2, PriorSpec::bosons(vec![(0, 1), (1, 2)])),
    ])
}

fn sample_event() -> EventPredicate {
    EventPredicate::or(vec![
        EventPredicate::And {
            parts: vec![
                EventPredicate::BosonModeTotal {
                    mode: 0,
                    cmp: Cmp::Ge,
                    value: 1,
                },
                EventPredicate::FermionModePresent { mode: 3 },
            ],
        },
        EventPredicate::Not {
            inner: Box::new(EventPredicate::Vacuum),
        },
    ])
}

fn sample_model_spec() -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::terms(vec![sample_term()]),
        prior: sample_prior(),
        solver: SolverSpec {
            krylov_dim: 12,
            prune_eps: 1e-10,
            max_components: Some(10_000),
            restarts: 3,
            device: DeviceSpec::Cuda { device_id: 0 },
            adaptive: false,
        },
    }
}

#[test]
fn round_trip_model_spec() {
    rt(&sample_model_spec());
}

#[test]
fn round_trip_hamiltonian_builtin() {
    rt(&HamiltonianSpec::builtin(
        "yang_mills",
        serde_json::json!({"g": 0.5}),
    ));
}

#[test]
fn round_trip_hamiltonian_latex() {
    rt(&HamiltonianSpec::latex(r"a^\dagger a"));
}

#[test]
fn round_trip_hamiltonian_typst() {
    // P8.7: the Typst variant round-trips through serde with the
    // snake_case tag and the `typst` field.
    rt(&HamiltonianSpec::typst("a^dagger_0 * a_0"));
}

#[test]
fn round_trip_hamiltonian_terms() {
    rt(&HamiltonianSpec::terms(vec![sample_term()]));
}

#[test]
fn round_trip_term_spec() {
    rt(&sample_term());
}

#[test]
fn round_trip_op_spec_all_levels() {
    for level in [
        Level::InnerBoson,
        Level::InnerFermion,
        Level::OuterBoson,
        Level::OuterFermion,
    ] {
        for kind in [OpKind::Create, OpKind::Annihilate] {
            rt(&OpSpec::new(kind, level, 7));
        }
    }
}

#[test]
fn round_trip_prior_vacuum() {
    rt(&PriorSpec::Vacuum);
}

#[test]
fn round_trip_prior_bosons() {
    rt(&PriorSpec::bosons(vec![(0, 1), (2, 3), (4, 5)]));
}

#[test]
fn round_trip_prior_fermions() {
    rt(&PriorSpec::fermions(vec![0, 1, 2]));
}

#[test]
fn round_trip_prior_superposition() {
    rt(&sample_prior());
}

#[test]
fn round_trip_event_predicate_all_variants() {
    rt(&sample_event());
    rt(&EventPredicate::BosonModeTotal {
        mode: 1,
        cmp: Cmp::Eq,
        value: 2,
    });
    rt(&EventPredicate::FermionModePresent { mode: 0 });
    rt(&EventPredicate::BosonUniverseCount {
        cmp: Cmp::Gt,
        value: 1,
    });
    rt(&EventPredicate::FermionUniverseCount {
        cmp: Cmp::Le,
        value: 0,
    });
    rt(&EventPredicate::Vacuum);
    rt(&EventPredicate::and(vec![
        EventPredicate::Vacuum,
        EventPredicate::Vacuum,
    ]));
    rt(&EventPredicate::not(EventPredicate::Vacuum));
}

#[test]
fn round_trip_solver_spec() {
    rt(&SolverSpec::default());
    rt(&SolverSpec {
        krylov_dim: 20,
        prune_eps: 1e-14,
        max_components: Some(50_000),
        restarts: 5,
        device: DeviceSpec::Cpu,
        adaptive: true,
    });
}

#[test]
fn round_trip_device_spec() {
    rt(&DeviceSpec::Cpu);
    rt(&DeviceSpec::Cuda { device_id: 2 });
}

#[test]
fn round_trip_agent_request() {
    rt(&AgentRequest::new(
        "req-42",
        "evolve",
        serde_json::json!({"t": 1.5, "model": "abc"}),
    ));
}

#[test]
fn round_trip_agent_response_ok() {
    rt(&AgentResponse::ok(
        "req-42",
        serde_json::json!({"probability": 0.314}),
    ));
}

#[test]
fn round_trip_agent_response_err() {
    let diag = Diagnostic::new(
        Code::STATE_EXPLOSION,
        "too many components",
        Severity::Error,
    )
    .with_hint(RepairHint::new(
        HintKind::IncreaseLimit,
        "solver.max_components",
        "raise the limit or reduce the Krylov dimension",
    ))
    .with_data(serde_json::json!({"components": 99999, "limit": 50000}));
    rt(&AgentResponse::err("req-42", diag));
}

#[test]
fn round_trip_diagnostic_full() {
    let diag = Diagnostic::new(Code::GRAM_DEGENERATE, "rank 0", Severity::Fatal)
        .with_hint(RepairHint::new(
            HintKind::ReduceScope,
            "solver.krylov_dim",
            "try a smaller Krylov dimension",
        ))
        .with_hint(RepairHint::new(
            HintKind::SetParam,
            "shifts",
            "use shifts with larger imaginary part",
        ))
        .with_data(serde_json::json!({"rank": 0, "dim": 8}));
    rt(&diag);
}

#[test]
fn round_trip_code() {
    rt(&Code::BAD_JSON);
    rt(&Code::INTERNAL);
}

#[test]
fn round_trip_severity_all() {
    for s in [
        Severity::Info,
        Severity::Warning,
        Severity::Error,
        Severity::Fatal,
    ] {
        rt(&s);
    }
}

#[test]
fn round_trip_hint_kind_all() {
    for k in [
        HintKind::ReplaceValue,
        HintKind::SetParam,
        HintKind::ReduceScope,
        HintKind::IncreaseLimit,
        HintKind::UseAlternativeOp,
    ] {
        rt(&k);
    }
}

#[test]
fn round_trip_repair_hint() {
    rt(&RepairHint::new(
        HintKind::ReplaceValue,
        "hamiltonian.name",
        "use 'yang_mills' instead of 'yangmills'",
    ));
}

#[test]
fn code_uniqueness() {
    let codes: Vec<u32> = all().iter().map(|(c, _, _)| *c).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        codes.len(),
        sorted.len(),
        "duplicate codes found in all() registry",
    );
}

#[test]
fn every_const_code_is_registered() {
    let registered: std::collections::HashSet<u32> = all().iter().map(|(c, _, _)| *c).collect();
    for code in [
        Code::BAD_JSON,
        Code::UNKNOWN_BUILTIN_MODEL,
        Code::BAD_EVENT_PREDICATE,
        Code::BAD_HANDLE,
        Code::BUFFER_TOO_SMALL,
        Code::GRAM_DEGENERATE,
        Code::STATE_EXPLOSION,
        Code::ZERO_PROBABILITY_CONDITION,
        Code::BRST_NOT_CONVERGED,
        Code::CAS_TERM_EXPLOSION,
        Code::CUDA_UNAVAILABLE,
        Code::OUT_OF_MEMORY_BUDGET,
        Code::CALL_DENIED,
        Code::INTERNAL,
    ] {
        assert!(
            registered.contains(&code.0),
            "Code {} ({}) is not in all() registry",
            code,
            code.0,
        );
    }
}

#[test]
fn code_display_uses_uk_prefix() {
    assert_eq!(Code::BAD_JSON.to_string(), "UK-1001");
    assert_eq!(Code::INTERNAL.to_string(), "UK-5000");
    assert_eq!(Code(3999).to_string(), "UK-3999");
}

#[test]
fn name_and_description_lookups() {
    assert_eq!(name_of(1001), Some("BadJson"));
    assert_eq!(name_of(5000), Some("Internal"));
    assert_eq!(name_of(9999), None);
    assert!(description_of(2002).unwrap().contains("component limit"));
    assert_eq!(description_of(9999), None);
}

#[test]
fn diagnostic_new_looks_up_name() {
    let d = Diagnostic::new(Code::CALL_DENIED, "denied", Severity::Error);
    assert_eq!(d.name, "CallDenied");
    assert_eq!(d.code, Code::CALL_DENIED);
    assert!(d.data.is_null());
    assert!(d.hints.is_empty());
}

#[test]
fn diagnostic_display() {
    let d = Diagnostic::new(Code::BAD_JSON, "unexpected token", Severity::Error);
    let s = d.to_string();
    assert!(s.contains("UK-1001"));
    assert!(s.contains("BadJson"));
    assert!(s.contains("unexpected token"));
}

#[test]
fn json_shape_hamiltonian_builtin() {
    let spec = HamiltonianSpec::builtin("yang_mills", serde_json::json!({"g": 0.5}));
    let v: serde_json::Value = serde_json::to_value(&spec).unwrap();
    assert_eq!(v["kind"], "builtin");
    assert_eq!(v["name"], "yang_mills");
    assert_eq!(v["params"]["g"], 0.5);
}

#[test]
fn json_shape_event_predicate_nested() {
    let pred = sample_event();
    let v: serde_json::Value = serde_json::to_value(&pred).unwrap();
    assert_eq!(v["kind"], "or");
    assert_eq!(v["parts"][0]["kind"], "and");
    assert_eq!(v["parts"][0]["parts"][0]["kind"], "boson_mode_total");
    assert_eq!(v["parts"][1]["kind"], "not");
    assert_eq!(v["parts"][1]["inner"]["kind"], "vacuum");
}

#[test]
fn json_shape_diagnostic() {
    let d = Diagnostic::new(Code::STATE_EXPLOSION, "boom", Severity::Fatal).with_hint(
        RepairHint::new(HintKind::IncreaseLimit, "solver.max_components", "raise it"),
    );
    let v: serde_json::Value = serde_json::to_value(&d).unwrap();
    assert_eq!(v["code"], 2002);
    assert_eq!(v["name"], "StateExplosion");
    assert_eq!(v["severity"], "fatal");
    assert_eq!(v["hints"][0]["kind"], "increase_limit");
    assert_eq!(v["hints"][0]["target"], "solver.max_components");
}

// P7 P5: HmcOptsSpec::validate() — the FFI's uk_bayesian_update calls
// this before the HMC and returns UK-1001 BAD_JSON with the per-field
// RepairHint if any field is out of range. A leapfrog_steps=0 or
// step_size=0 silently produces a broken HMC chain; the validation
// surfaces this at the kernel boundary.

#[test]
fn hmc_opts_default_is_valid() {
    let opts = HmcOptsSpec::default();
    let hints = opts.validate();
    assert!(
        hints.is_empty(),
        "default HmcOptsSpec should be valid, got: {hints:#?}"
    );
}

#[test]
fn hmc_opts_validate_zero_leapfrog_steps() {
    let mut opts = HmcOptsSpec::default();
    opts.leapfrog_steps = 0;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].target, "hmc_opts.leapfrog_steps");
    assert!(matches!(hints[0].kind, HintKind::IncreaseLimit));
}

#[test]
fn hmc_opts_validate_zero_step_size() {
    let mut opts = HmcOptsSpec::default();
    opts.step_size = 0.0;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].target, "hmc_opts.step_size");
    assert!(matches!(hints[0].kind, HintKind::ReplaceValue));
}

#[test]
fn hmc_opts_validate_negative_step_size() {
    let mut opts = HmcOptsSpec::default();
    opts.step_size = -0.1;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert!(matches!(hints[0].kind, HintKind::ReplaceValue));
}

#[test]
fn hmc_opts_validate_nan_step_size() {
    let mut opts = HmcOptsSpec::default();
    opts.step_size = f64::NAN;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert!(matches!(hints[0].kind, HintKind::ReplaceValue));
}

#[test]
fn hmc_opts_validate_n_iterations_lt_burn_in() {
    let mut opts = HmcOptsSpec::default();
    opts.n_iterations = 50;
    opts.burn_in = 100;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].target, "hmc_opts");
    assert!(matches!(hints[0].kind, HintKind::SetParam));
    assert!(hints[0].suggestion.contains("n_iterations >= burn_in"));
}

#[test]
fn hmc_opts_validate_multiple_errors_all_reported() {
    let opts = HmcOptsSpec {
        leapfrog_steps: 0,
        step_size: -0.5,
        n_iterations: 0,
        burn_in: 0,
        seed: 0,
    };
    let hints = opts.validate();
    // At least 3 distinct errors: leapfrog_steps=0, step_size<0, n_iterations=0
    assert!(
        hints.len() >= 3,
        "expected at least 3 validation hints, got {}",
        hints.len()
    );
    let targets: Vec<&str> = hints.iter().map(|h| h.target.as_str()).collect();
    assert!(targets.contains(&"hmc_opts.leapfrog_steps"));
    assert!(targets.contains(&"hmc_opts.step_size"));
    assert!(targets.contains(&"hmc_opts.n_iterations"));
}

#[test]
fn hmc_opts_validate_unreasonably_large_leapfrog_warns() {
    let mut opts = HmcOptsSpec::default();
    opts.leapfrog_steps = 50_000;
    let hints = opts.validate();
    assert_eq!(hints.len(), 1);
    assert!(matches!(hints[0].kind, HintKind::ReduceScope));
    assert!(hints[0].suggestion.contains("leapfrog_steps = 50000"));
}

#[test]
fn hmc_opts_validate_at_boundary_is_ok() {
    // n_iterations == burn_in is OK (no samples to keep but no error).
    let opts = HmcOptsSpec {
        leapfrog_steps: 1,
        step_size: 0.01,
        n_iterations: 100,
        burn_in: 100,
        seed: 0,
    };
    let hints = opts.validate();
    assert!(
        hints.is_empty(),
        "boundary case should be valid, got: {hints:#?}"
    );
}
