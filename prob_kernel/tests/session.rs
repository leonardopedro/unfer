use prob_kernel::{KernelError, Session, SessionBlob};
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
            adaptive: false,
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
            adaptive: false,
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

fn yang_mills_lattice_spec(prior: PriorSpec) -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "yang_mills_lattice",
            serde_json::json!({"l": 2, "g": 1.0, "n_colors": 1}),
        ),
        prior,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(100_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    }
}

#[test]
fn yang_mills_lattice_builds_and_evolves() {
    // Vacuum prior on the 2×2 single-color lattice. The magnetic plaquette term
    // is a quartic Φ⁴ interaction; evolving off the vacuum drives it (the
    // electric term annihilates the vacuum, but Φ⁴ creates excitations), so the
    // solver must handle a quartic-heavy Hamiltonian and stay normalized.
    let spec = yang_mills_lattice_spec(PriorSpec::Vacuum);
    let mut session = Session::new(&spec).expect("yang-mills-lattice session");

    // Starts in the vacuum.
    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    // Evolve under the gauge Hamiltonian (exercises the quartic plaquette term).
    let report = session.evolve(0.1).expect("evolve");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1, got {}",
        report.norm
    );

    // The complete cover {vacuum, ¬vacuum} stays normalized after the quartic
    // dynamics, and the vacuum probability stays a valid probability.
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    let p_not = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_vac + p_not - 1.0).abs() < 1e-6,
        "probabilities must sum to 1 (got {p_vac} + {p_not})"
    );
    assert!((0.0..=1.0).contains(&p_vac), "P(vacuum) in [0,1]: {p_vac}");
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

fn qfm_mehler_spec(prior: PriorSpec) -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler",
            serde_json::json!({"alphas": [1.5, 2.1, 0.8]}),
        ),
        prior,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    }
}

#[test]
fn qfm_mehler_builds_and_evolves() {
    // The analytical Quantum Flow Matching generator (QFM.tex):
    //   H = |0><0| + Σ_j α_j n_j.
    // It is diagonal in the Fock basis: the Mehler projector makes the vacuum an
    // eigenstate (eigenvalue 1) and each data mode an eigenstate (eigenvalue α_j).
    // So evolution adds only phases — Born populations are conserved exactly.
    let spec = qfm_mehler_spec(PriorSpec::Vacuum);
    let mut session = Session::new(&spec).expect("qfm session");

    // Starts in the Mehler vacuum prior.
    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    // Evolve under the QFM generator — exercises the rank-1 ProjectVacuum term.
    let report = session.evolve(1.0).expect("evolve");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1, got {}",
        report.norm
    );

    // |0> is an eigenstate of H, so the vacuum population is stationary and the
    // {vacuum, ¬vacuum} cover stays normalized.
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    let p_not = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_vac - 1.0).abs() < 1e-6,
        "vacuum is a QFM eigenstate, stays occupied: {p_vac}"
    );
    assert!(
        (p_vac + p_not - 1.0).abs() < 1e-6,
        "probabilities must sum to 1 (got {p_vac} + {p_not})"
    );
}

#[test]
fn qfm_mehler_conserves_data_channel_population() {
    // A boson seeded in data channel 0 is an eigenstate (eigenvalue α_0) of the
    // diagonal QFM generator, so its occupation probability is conserved exactly
    // under evolution — the O(M) decoupled potential never mixes channels.
    let spec = qfm_mehler_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("qfm session");

    let p_before = session.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        (p_before - 1.0).abs() < 1e-10,
        "channel 0 occupied: {p_before}"
    );

    session.evolve(1.0).expect("evolve");

    let p_after = session.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        (p_after - 1.0).abs() < 1e-6,
        "diagonal generator conserves channel-0 population: {p_after}"
    );
}

// ── Localized QFM encoding ─────────────────────────────────────────────────
// `QFM.tex`, "The data-channel wave-function on the hypersphere": each data
// point x ∈ R^D localizes exactly D of the (infinitely many) hyperspherical
// coordinates, the rest staying at the uniform circle measure.
// `qfm_mehler_localized`/`qfm_mehler_projector_localized` are the direct
// computational realization — each data point occupies its own D inner
// modes (one per real coordinate) instead of being identified only by
// array index. The single-channel physics (projector oscillation,
// eigenstate structure) must be identical to the index-based encoding
// above, since it depends only on the outer Fock-space grading, not on
// which `InnerBosonicState` label the one-particle sector uses.

// Quantization scale used only by these tests (not `QFM_DEFAULT_QUANTIZATION_SCALE`).
// `localized_point_prior` below builds its comparison state via
// `PriorSpec::Bosons`, which constructs a universe by *repeatedly applying*
// `InnerBosonCreate` (a genuine ladder operation, picking up a `sqrt(count!)`
// amplitude factor) — unlike `qfm_hamiltonian_localized` itself, which uses
// `OuterBosonCreate` directly (amplitude 1, no factorial, regardless of how
// large the occupation numbers inside the label are). At the library
// default scale (1024.0) realistic test coordinates quantize to
// occupation counts in the thousands, and `sqrt(2047!)` overflows `f64` to
// `Infinity` — a fragility of `PriorSpec::Bosons`'s repeated-ladder
// construction, not of the Hamiltonian builder. A small scale keeps this
// test's occupation counts in the single digits, avoiding that overflow.
const TEST_LOCALIZATION_SCALE: f64 = 1.0;

fn qfm_mehler_localized_spec(prior: PriorSpec) -> ModelSpec {
    ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_localized",
            serde_json::json!({
                "points": [[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]],
                "alphas": [1.5, 2.1, 0.8],
                "scale": TEST_LOCALIZATION_SCALE
            }),
        ),
        prior,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    }
}

/// A prior seeded in the single outer universe that
/// `qfm_hamiltonian_localized`/`_mehler_projector_localized` would create for `point`
/// at [`TEST_LOCALIZATION_SCALE`] — built via `PriorSpec::Bosons`, which
/// (like `point_to_inner_state`) gives one outer universe carrying the
/// listed inner-mode occupations.
fn localized_point_prior(point: &[f64]) -> PriorSpec {
    let inner = nested_fock_algebra::point_to_inner_state(point, TEST_LOCALIZATION_SCALE);
    PriorSpec::bosons(inner.modes.into_iter().collect())
}

#[test]
fn qfm_mehler_localized_builds_and_evolves() {
    // Diagonal generator: H = |0><0| + Σ_j α_j B†_j B_j, so the vacuum and
    // each data channel are eigenstates; evolution adds only phases.
    let spec = qfm_mehler_localized_spec(PriorSpec::Vacuum);
    let mut session = Session::new(&spec).expect("qfm localized session");

    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    let report = session.evolve(1.0).expect("evolve");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1, got {}",
        report.norm
    );

    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        (p_vac - 1.0).abs() < 1e-6,
        "vacuum is a QFM eigenstate, stays occupied: {p_vac}"
    );
}

#[test]
fn qfm_mehler_localized_conserves_data_channel_population() {
    // A prior seeded in one localized data channel (point (-1.0, 2.0), whose
    // D=2 real coordinates each occupy their own inner mode) is an
    // eigenstate of the diagonal localized generator, so its occupation
    // stays 1 under evolution — the localized encoding must not introduce
    // any cross-channel mixing that the index-based encoding didn't have.
    let points: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 2.0]];
    let prior = localized_point_prior(&points[2]);
    let spec = qfm_mehler_localized_spec(prior);
    let mut session = Session::new(&spec).expect("qfm localized session");

    let p_before = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!((p_before - 1.0).abs() < 1e-10, "starts in data channel: {p_before}");

    session.evolve(1.0).expect("evolve");

    let p_after = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_after - 1.0).abs() < 1e-6,
        "diagonal localized generator conserves channel population: {p_after}"
    );
}

#[test]
fn qfm_mehler_projector_localized_matches_closed_form_oscillation() {
    // Exact off-diagonal generator with a localized channel: H = |0̃><0̃|
    // where the one data channel is a D=2 localized point (1.0, -2.0)
    // instead of a bare index. The closed-form projector oscillation
    // applies regardless of the inner-mode labeling: with ε = 0.3
    // (c₀² = 0.91), at t = π the channel population is
    //   P(¬vac)(π) = 4 sin²(π/2) c₀² ε² = 4·0.91·0.09 = 0.3276,
    //   P(vac)(π)  = (1 − 2c₀²)² = 0.6724,
    // and at t = 2π the state returns coherently to the frame vacuum.
    let point = vec![1.0, -2.0];
    let eps = 0.3_f64;
    let c0_sq = 1.0 - eps * eps;
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_projector_localized",
            serde_json::json!({"points": [point], "epsilons": [eps]}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 6,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("qfm projector localized session");

    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    let pi = std::f64::consts::PI;
    let report = session.evolve(pi).expect("evolve to t=π");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1 (unitary), got {}",
        report.norm
    );

    let p_data = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    let want_data = 4.0 * c0_sq * eps * eps;
    let want_vac = (1.0 - 2.0 * c0_sq) * (1.0 - 2.0 * c0_sq);
    assert!(
        (p_data - want_data).abs() < 0.02,
        "P(¬vac)(π) = {p_data}, want {want_data}"
    );
    assert!(
        (p_vac - want_vac).abs() < 0.02,
        "P(vac)(π) = {p_vac}, want {want_vac}"
    );
    assert!(
        (p_vac + p_data - 1.0).abs() < 1e-6,
        "cover sums to 1: {p_vac} + {p_data}"
    );

    // Second half of the period: exact coherent return to the frame vacuum.
    session.evolve(pi).expect("evolve to t=2π");
    let p_vac_full = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        p_vac_full > 0.95,
        "coherent return at t=2π: P(vac) = {p_vac_full}, want ≈ 1"
    );
}

// ── Exact Mehler-projector QFM ─────────────────────────────────────────────
// `QFM.tex`, "The exact off-diagonal generator is just the vacuum
// projector": H = |0><0| where |0> is the uniform Mehler vacuum, which is
// NOT orthogonal to the data channels (<0|x_j> = ε_j > 0, from the finite
// localization of each channel's inner wave-function). Because H is a
// rank-1 projector, e^{-iHt} = 1 + (e^{-it} − 1)|0><0| exactly: starting
// from the frame vacuum |vac>_F (overlap c₀ = sqrt(1 − Σε²) with |0>),
// every channel is pumped coherently and simultaneously with population
//   P_j(t) = 4 sin²(t/2) c₀² ε_j²,
// and the state returns exactly at t = 2π. These tests pin the SIRK-path
// evolution against that closed form.

#[test]
fn qfm_mehler_projector_matches_closed_form_oscillation() {
    // eps = [0.3, 0.4]: Σε² = 0.25, c₀² = 0.75. At t = π:
    //   P(mode0 ≥ 1) = 4·0.75·0.09 = 0.27
    //   P(mode1 ≥ 1) = 4·0.75·0.16 = 0.48
    //   P(vacuum)    = |1 − 2c₀²|² = 0.25
    // At t = 2π: P(vacuum) = 1 (exact coherent return).
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_projector",
            serde_json::json!({"epsilons": [0.3, 0.4]}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 6,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("mehler projector session");

    let pi = std::f64::consts::PI;
    session.evolve(pi).expect("evolve to t=π");

    let p0 = session.probability(&event_mode0_ge1()).expect("prob");
    let p1 = session
        .probability(&EventPredicate::BosonModeTotal {
            mode: 1,
            cmp: Cmp::Ge,
            value: 1,
        })
        .expect("prob");
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p0 - 0.27).abs() < 0.02, "P(x_0)(π) = {p0}, want 0.27");
    assert!((p1 - 0.48).abs() < 0.02, "P(x_1)(π) = {p1}, want 0.48");
    assert!((p_vac - 0.25).abs() < 0.02, "P(vac)(π) = {p_vac}, want 0.25");

    // Second half of the period: exact coherent return to the frame vacuum.
    session.evolve(pi).expect("evolve to t=2π");
    let p_vac_full = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        p_vac_full > 0.95,
        "coherent return at t=2π: P(vac) = {p_vac_full}, want ≈ 1"
    );
}

#[test]
fn qfm_mehler_projector_dressed_vacuum_is_stationary() {
    // The dressed Mehler vacuum |0> = c₀|vac>_F + Σ ε_j|x_j> is the
    // eigenvalue-1 eigenvector of its own projector, so as a prior it is
    // stationary: evolution adds only a global phase and every Born
    // population is conserved exactly.
    let eps = [0.3_f64, 0.4];
    let c0 = (1.0 - eps.iter().map(|e| e * e).sum::<f64>()).sqrt(); // sqrt(0.75)
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_projector",
            serde_json::json!({"epsilons": eps}),
        ),
        prior: PriorSpec::superposition(vec![
            unfer_protocol::SuperpositionTerm::new(c0, 0.0, PriorSpec::Vacuum),
            unfer_protocol::SuperpositionTerm::new(eps[0], 0.0, PriorSpec::bosons(vec![(0, 1)])),
            unfer_protocol::SuperpositionTerm::new(eps[1], 0.0, PriorSpec::bosons(vec![(1, 1)])),
        ]),
        solver: SolverSpec {
            krylov_dim: 6,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("dressed vacuum session");

    let p0_before = session.probability(&event_mode0_ge1()).expect("prob");
    let p_vac_before = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p0_before - 0.09).abs() < 1e-9, "prior P(x_0) = ε_0² = 0.09");
    assert!((p_vac_before - 0.75).abs() < 1e-9, "prior P(vac) = c₀² = 0.75");

    session.evolve(1.3).expect("evolve");

    let p0_after = session.probability(&event_mode0_ge1()).expect("prob");
    let p_vac_after = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        (p0_after - p0_before).abs() < 0.01,
        "eigenstate populations must be stationary: P(x_0) {p0_before} -> {p0_after}"
    );
    assert!(
        (p_vac_after - p_vac_before).abs() < 0.01,
        "eigenstate populations must be stationary: P(vac) {p_vac_before} -> {p_vac_after}"
    );
}

#[test]
fn qfm_mehler_projector_rejects_overweight_epsilons() {
    // Σ ε² > 1 is physically impossible (the ε² are uniform-measure masses
    // of disjoint packet supports) and must be rejected at build time with
    // a diagnostic, not a panic.
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_projector",
            serde_json::json!({"epsilons": [0.9, 0.9]}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let err = Session::new(&spec).expect_err("Σ ε² > 1 must fail");
    assert!(
        matches!(err, KernelError::BadBuiltinParams { .. }),
        "want BadBuiltinParams, got {err:?}"
    );
}

// ── D10: session persistence ──────────────────────────────────────────────────

#[test]
fn session_save_restore_roundtrip() {
    // Evolve a session, save it, restore it, and verify the restored session
    // has the same t_now, norm, and event probabilities.
    let spec = harmonic_chain_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("session creation");

    session.evolve(0.5).expect("evolve");

    let p_before = session.probability(&event_mode0_ge1()).expect("prob");
    let t_before = session.t();
    let n_before = session.n_components();

    // Save → JSON round-trip → restore.
    let blob: SessionBlob = session.save();
    let json = serde_json::to_string(&blob).expect("serialize blob");
    let blob2: SessionBlob = serde_json::from_str(&json).expect("deserialize blob");
    let restored = Session::restore(blob2).expect("restore session");

    // Time and component count must be preserved exactly.
    assert!(
        (restored.t() - t_before).abs() < 1e-15,
        "t_now mismatch: {} vs {}",
        restored.t(),
        t_before
    );
    assert_eq!(
        restored.n_components(),
        n_before,
        "component count mismatch"
    );

    // Born-rule probabilities must be identical after restore.
    let p_after = restored.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        (p_before - p_after).abs() < 1e-12,
        "probability changed after restore: {p_before} → {p_after}"
    );

    // Vacuum + ¬vacuum still covers everything.
    let p_vac = restored.probability(&EventPredicate::Vacuum).expect("prob");
    let p_not = restored
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_vac + p_not - 1.0).abs() < 1e-10,
        "cover must sum to 1 after restore"
    );
}

#[test]
fn evolve_report_includes_solve_ms() {
    let spec = harmonic_chain_spec(PriorSpec::bosons(vec![(0, 1)]));
    let mut session = Session::new(&spec).expect("session creation");
    let report = session.evolve(0.1).expect("evolve");
    // solve_ms is always set (may be 0 on very fast CPU, but the field exists).
    let _ = report.solve_ms; // compile-check that the field is present
}

// ── P5 #30: physics depth ────────────────────────────────────────────────────

#[test]
fn yang_mills_lattice_l4_bounded_evolve() {
    // Scale to the 4×4 lattice (32 link modes, 256 quartic magnetic terms).
    // Starting from vacuum at tiny t=0.01, the magnetic plaquette terms create
    // at most a handful of Fock components — the state must stay normalised.
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "yang_mills_lattice",
            serde_json::json!({"l": 4, "g": 1.0, "n_colors": 1}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(100_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("l=4 yang-mills session");

    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    let report = session.evolve(0.01).expect("evolve t=0.01");
    assert!(
        (report.norm - 1.0).abs() < 1e-5,
        "post-evolve norm must be ~1, got {}",
        report.norm
    );

    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    let p_not = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_vac + p_not - 1.0).abs() < 1e-5,
        "cover sums to 1 (got {p_vac} + {p_not})"
    );
}

#[test]
fn sirk_stability_krylov_dim_16() {
    // High Krylov dimension (m=16) on a 4-mode harmonic chain: the SIRK basis
    // is over-complete relative to the reachable subspace, so Gram whitening
    // must tolerate near-singular directions without blowing up the norm.
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "harmonic_chain",
            serde_json::json!({"n_modes": 4, "omega": 1.0}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 16,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("krylov-16 session");
    let report = session.evolve(0.5).expect("evolve t=0.5");
    assert!(
        (report.norm - 1.0).abs() < 1e-5,
        "krylov_dim=16: norm must be ~1, got {}",
        report.norm
    );
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((0.0..=1.0).contains(&p_vac), "P(vacuum) in [0,1]: {p_vac}");
}

#[test]
fn sirk_stability_krylov_dim_32() {
    // At m=32 the Krylov space vastly exceeds the 2-state reachable subspace
    // of the harmonic chain from vacuum.  All but ~2 eigenvalues of the Gram
    // matrix will be near zero — the whitening must discard them (rank
    // reduction) and still reconstruct a normalised output state.
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "harmonic_chain",
            serde_json::json!({"n_modes": 4, "omega": 1.0}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 32,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    };
    let mut session = Session::new(&spec).expect("krylov-32 session");
    let report = session.evolve(0.5).expect("evolve t=0.5");
    assert!(
        (report.norm - 1.0).abs() < 1e-5,
        "krylov_dim=32: norm must be ~1, got {}",
        report.norm
    );
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((0.0..=1.0).contains(&p_vac), "P(vacuum) in [0,1]: {p_vac}");
}

// ── Workstream F5: QFM tomographic integration tests ─────────────────

/// Build a `ModelSpec` with a `HamiltonianSpec::QfmTomography` variant.
fn qfm_tomo_spec(training_data: Vec<Vec<f64>>) -> ModelSpec {
    use unfer_protocol::QfmTomographySpec;
    // P7 P3: the QFM compile requires `krylov_dim >= k2` (the SIRK
    // sequence has `krylov_dim + 1` rows; the K_2-row restriction of
    // w_whiten needs `krylov_dim >= K_2`), and `k2 >= d` (the
    // krylov_image_basis debug_assert!). The SIRK clamp reduces
    // krylov_dim to min(config.krylov_dim, m, k2), so `k2 <= m` is
    // also required. So pick k2 = max(m, d) and krylov_dim = k2.
    let m = training_data.len();
    let d = training_data.first().map(|p| p.len()).unwrap_or(8);
    let k2 = m.max(d);
    let spec = QfmTomographySpec {
        training_data,
        k: 4,
        k2,
        krylov_dim: k2,
        seed: 42,
    };
    ModelSpec {
        hamiltonian: HamiltonianSpec::qfm_tomography(spec),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 4,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
            adaptive: false,
        },
    }
}

#[test]
fn qfm_tomo_compile_and_generate() {
    // 8 training points in d=8 (the canonical F5 tetrahedron+ cube
    // setup; P7 P3 requires m >= k2 >= d, so 8 training points in d=8
    // with k2=8 satisfies all constraints).
    let training: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            let mut v = vec![0.0; 8];
            v[i] = 1.0;
            v
        })
        .collect();
    let spec = qfm_tomo_spec(training.clone());
    let mut session = Session::new(&spec).expect("QFM session");

    // Evolve with a query: the pipeline generates a raw image.
    let report = session
        .evolve_with_query(1.0, Some(&training[0]))
        .expect("QFM generate");
    let output = report.qfm_output.expect("qfm_output must be present");
    assert_eq!(output.len(), 8, "generated image must have d=8 elements");
    for &v in &output {
        assert!(v.is_finite(), "output must be finite, got {v}");
    }
}

#[test]
fn qfm_tomo_no_query_returns_error() {
    let training = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let spec = qfm_tomo_spec(training);
    let mut session = Session::new(&spec).expect("QFM session");

    // Evolve without a query: the QFM pipeline requires a query.
    let result = session.evolve(1.0);
    assert!(
        result.is_err(),
        "evolve without query must fail for QFM model"
    );
}

#[test]
fn qfm_tomo_no_m_in_evolve_report() {
    // Verify the EvolveReport payload does not reference the training data.
    // P7 P3: m=4, d=4, k2=4 satisfies k2 <= m and k2 >= d.
    let training: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            let mut v = vec![0.0; 4];
            v[i] = 1.0;
            v
        })
        .collect();
    let spec = qfm_tomo_spec(training.clone());
    let mut session = Session::new(&spec).expect("QFM session");
    let report = session
        .evolve_with_query(1.0, Some(&[1.0, 0.0, 0.0, 0.0]))
        .expect("QFM generate");

    // Serialize the report and check it serializes cleanly.
    let json = serde_json::to_string(&report).expect("serialize report");
    assert!(json.contains("qfm_output"));
    assert!(report.qfm_output.is_some());
}
