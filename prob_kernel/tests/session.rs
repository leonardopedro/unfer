use prob_kernel::{KernelError, Session};
use unfer_protocol::{
    Cmp, Code, DeviceSpec, EventPredicate, HamiltonianSpec, HintKind, ModelSpec, PriorSpec,
    SolverSpec,
};

fn harmonic_chain_spec(prior: PriorSpec) -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "harmonic_chain",
            serde_json::json!({"n_modes": 2, "omega": 1.0}),
        ),
        prior,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
        },
    }
}

fn superposition_prior() -> PriorSpec {
    PriorSpec::superposition(vec![
        unfer_protocol::SuperpositionTerm::new(0.5, 0.0, PriorSpec::Vacuum),
        unfer_protocol::SuperpositionTerm::new(0.5, 0.0, PriorSpec::bosons(vec![(0, 1)])),
    ])
}

fn event_mode0_ge1() -> EventPredicate {
    EventPredicate::BosonModeTotal {
        mode: 0,
        cmp: Cmp::Ge,
        value: 1,
    }
}

fn bose_hubbard_spec(prior: PriorSpec) -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "bose_hubbard",
            serde_json::json!({"n_modes": 2, "t": 1.0, "u": 1.0, "periodic": false}),
        ),
        prior,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
        },
    }
}

#[test]
fn probabilities_sum_to_one() {
    let spec = harmonic_chain_spec(superposition_prior());
    let session = Session::new(&spec).expect("session creation");

    let p_vacuum = session.probability(&EventPredicate::Vacuum).expect("prob");
    let p_not_vacuum = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");

    assert!((p_vacuum - 0.5).abs() < 1e-10, "P(vacuum) = {p_vacuum}");
    assert!(
        (p_not_vacuum - 0.5).abs() < 1e-10,
        "P(not vacuum) = {p_not_vacuum}"
    );
    assert!(
        (p_vacuum + p_not_vacuum - 1.0).abs() < 1e-10,
        "P(E) + P(¬E) must sum to 1"
    );
}

#[test]
fn probabilities_sum_to_one_mutually_exclusive() {
    let spec = harmonic_chain_spec(superposition_prior());
    let session = Session::new(&spec).expect("session creation");

    let p_vacuum = session.probability(&EventPredicate::Vacuum).expect("prob");
    let p_one_boson = session.probability(&event_mode0_ge1()).expect("prob");

    assert!((p_vacuum - 0.5).abs() < 1e-10, "P(vacuum) = {p_vacuum}");
    assert!(
        (p_one_boson - 0.5).abs() < 1e-10,
        "P(boson>=1) = {p_one_boson}"
    );
    assert!(
        (p_vacuum + p_one_boson - 1.0).abs() < 1e-10,
        "mutually exclusive events must sum to 1"
    );
}

#[test]
fn condition_then_probability_is_one() {
    let spec = harmonic_chain_spec(superposition_prior());
    let mut session = Session::new(&spec).expect("session creation");

    let prior_p = session
        .condition(&event_mode0_ge1())
        .expect("conditioning must succeed");
    assert!((prior_p - 0.5).abs() < 1e-10, "prior P(E) = {prior_p}");

    let post_p = session.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        (post_p - 1.0).abs() < 1e-10,
        "after conditioning, P(E) must be 1.0, got {post_p}"
    );
}

#[test]
fn condition_eliminates_non_matching() {
    let spec = harmonic_chain_spec(superposition_prior());
    let mut session = Session::new(&spec).expect("session creation");

    session
        .condition(&event_mode0_ge1())
        .expect("conditioning must succeed");

    let p_vacuum = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        p_vacuum < 1e-10,
        "after conditioning on boson>=1, P(vacuum) must be ~0, got {p_vacuum}"
    );
}

#[test]
fn impossible_event_returns_uk2003() {
    let spec = harmonic_chain_spec(PriorSpec::Vacuum);
    let mut session = Session::new(&spec).expect("session creation");

    let err = session
        .condition(&event_mode0_ge1())
        .expect_err("must fail on zero-probability event");

    match &err {
        KernelError::ZeroProbabilityCondition { mass } => {
            assert!(mass < &1e-15, "mass should be ~0");
        }
        other => panic!("expected ZeroProbabilityCondition, got {other:?}"),
    }

    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::ZERO_PROBABILITY_CONDITION);
    assert!(
        !diag.hints.is_empty(),
        "UK-2003 must carry at least one repair hint"
    );
    assert_eq!(diag.hints[0].kind, HintKind::UseAlternativeOp);
}

#[test]
fn post_evolve_normalization() {
    let spec = harmonic_chain_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("session creation");

    let report = session.evolve(0.1).expect("evolve");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1, got {}",
        report.norm
    );
}

#[test]
fn post_evolve_probabilities_unchanged_for_eigenstate() {
    let spec = harmonic_chain_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("session creation");

    let p_before = session.probability(&event_mode0_ge1()).expect("prob");
    session.evolve(0.05).expect("evolve");
    let p_after = session.probability(&event_mode0_ge1()).expect("prob");

    assert!(
        (p_before - p_after).abs() < 1e-6,
        "eigenstate probabilities must not change: before={p_before}, after={p_after}"
    );
}

#[test]
fn unknown_builtin_returns_uk1002() {
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin("nonexistent_model", serde_json::json!({})),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec::default(),
    };

    let err = Session::new(&spec).expect_err("must fail");
    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::UNKNOWN_BUILTIN_MODEL);
    assert!(!diag.hints.is_empty());
    assert!(
        diag.hints[0].suggestion.contains("harmonic_chain"),
        "hint should list valid builtin names"
    );
}

#[test]
fn bad_builtin_params_returns_uk1001() {
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin("harmonic_chain", serde_json::json!({"n_modes": 2})),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec::default(),
    };

    let err = Session::new(&spec).expect_err("must fail");
    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::BAD_JSON);
    assert!(diag.message.contains("omega"));
}

#[test]
fn empty_terms_returns_uk1001() {
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::terms(vec![]),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec::default(),
    };

    let err = Session::new(&spec).expect_err("must fail");
    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::BAD_JSON);
}

#[test]
fn set_prior_resets_time() {
    let spec = harmonic_chain_spec(PriorSpec::Vacuum);
    let mut session = Session::new(&spec).expect("session creation");

    session.evolve(0.5).expect("evolve");
    assert!((session.t() - 0.5).abs() < 1e-10);

    session
        .set_prior(&PriorSpec::bosons(vec![(0, 1)]))
        .expect("set_prior");
    assert!((session.t() - 0.0).abs() < 1e-10, "set_prior must reset t");
}

#[test]
fn set_hamiltonian_preserves_state() {
    let spec = harmonic_chain_spec(superposition_prior());
    let mut session = Session::new(&spec).expect("session creation");

    let p_before = session.probability(&EventPredicate::Vacuum).expect("prob");
    session
        .set_hamiltonian(&HamiltonianSpec::builtin(
            "harmonic_chain",
            serde_json::json!({"n_modes": 2, "omega": 2.0}),
        ))
        .expect("set_hamiltonian");

    let p_after = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        (p_before - p_after).abs() < 1e-10,
        "set_hamiltonian must not change the state"
    );
}

#[test]
fn snapshot_returns_top_k() {
    let spec = harmonic_chain_spec(superposition_prior());
    let session = Session::new(&spec).expect("session creation");

    let snap = session.snapshot(10);
    assert_eq!(snap.components, 2);
    assert!((snap.norm - 1.0).abs() < 1e-10);
    assert_eq!(snap.top.len(), 2);
    assert!(snap.top[0].probability >= snap.top[1].probability);
}

#[test]
fn event_predicate_and_or_not() {
    let spec = harmonic_chain_spec(superposition_prior());
    let session = Session::new(&spec).expect("session creation");

    let p_and = session
        .probability(&EventPredicate::and(vec![
            EventPredicate::Vacuum,
            event_mode0_ge1(),
        ]))
        .expect("prob");
    assert!(p_and < 1e-10, "Vacuum AND Boson>=1 must be 0");

    let p_or = session
        .probability(&EventPredicate::or(vec![
            EventPredicate::Vacuum,
            event_mode0_ge1(),
        ]))
        .expect("prob");
    assert!((p_or - 1.0).abs() < 1e-10, "Vacuum OR Boson>=1 must be 1");
}

#[test]
fn sirk_error_maps_to_diagnostic() {
    let err = KernelError::Sirk(fock_sirk::SirkError::StateExplosion {
        components: 100_000,
        limit: 50_000,
    });
    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::STATE_EXPLOSION);
    assert!(diag.hints.iter().any(|h| h.kind == HintKind::IncreaseLimit));
    assert!(diag.hints.iter().any(|h| h.kind == HintKind::ReduceScope));
}

#[test]
fn cas_error_maps_to_diagnostic() {
    let err = KernelError::Cas(nested_fock_algebra::cas::CasError::TermExplosion {
        terms: 200_000,
        limit: 100_000,
    });
    let diag = err.to_diagnostic();
    assert_eq!(diag.code, Code::CAS_TERM_EXPLOSION);
    assert!(diag.hints.iter().any(|h| h.kind == HintKind::IncreaseLimit));
    assert!(
        diag.hints
            .iter()
            .any(|h| h.kind == HintKind::UseAlternativeOp)
    );
}

#[test]
fn bose_hubbard_builds_and_normalizes() {
    // One boson localized on site 0 of a 2-site Bose–Hubbard chain.
    let spec = bose_hubbard_spec(PriorSpec::bosons(vec![(0, 1)]));
    let session = Session::new(&spec).expect("bose-hubbard session");

    // {mode0≥1, ¬(mode0≥1)} is a complete, mutually-exclusive cover.
    let p_on0 = session.probability(&event_mode0_ge1()).expect("prob");
    let p_off0 = session
        .probability(&EventPredicate::not(event_mode0_ge1()))
        .expect("prob");

    assert!(
        (p_on0 - 1.0).abs() < 1e-10,
        "the boson starts on site 0: P(mode0≥1) = {p_on0}"
    );
    assert!(
        (p_on0 + p_off0 - 1.0).abs() < 1e-10,
        "probabilities must sum to 1"
    );
}

#[test]
fn bose_hubbard_hopping_conserves_norm() {
    // Under -t (a†_0 a_1 + h.c.) the boson can hop between the two sites; after
    // evolving, the total probability over the cover must still be 1 (norm
    // conservation) and each probability must stay in [0, 1].
    let spec = bose_hubbard_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("bose-hubbard session");

    session.evolve(0.5).expect("evolve");

    let p_on0 = session.probability(&event_mode0_ge1()).expect("prob");
    let p_off0 = session
        .probability(&EventPredicate::not(event_mode0_ge1()))
        .expect("prob");

    assert!(
        (p_on0 + p_off0 - 1.0).abs() < 1e-6,
        "post-evolution probabilities must sum to 1 (got {p_on0} + {p_off0})"
    );
    assert!((0.0..=1.0).contains(&p_on0), "P(mode0≥1) in [0,1]: {p_on0}");
}
