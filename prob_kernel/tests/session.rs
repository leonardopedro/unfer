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
        },
    }
}

#[test]
fn qfm_mehler_builds_and_evolves() {
    // The analytical Quantum Flow Matching generator (QMF.tex):
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

// ── Off-diagonal QFM (P5 #26) ─────────────────────────────────────────────
// H = |0><0| + Σ_j α_j (B†_j P₀ + P₀ B_j) — Hermitian vacuum↔data coupling
// that actually transports amplitude (Rabi oscillation), unlike the diagonal
// surrogate above where populations are stationary. The integration tests
// below verify the defining behaviour: starting from the Mehler vacuum prior,
// evolution moves probability mass INTO the data channels.

#[test]
fn qfm_mehler_offdiag_transfers_population_from_vacuum() {
    // The off-diagonal QFM generator mixes vacuum ↔ data channels. Starting
    // from the vacuum prior, evolution must MOVE probability out of the vacuum
    // and INTO the data channels — the defining behaviour the diagonal
    // surrogate lacks (there, P(vacuum) stays 1).
    //
    // Single channel (M=1) so the 2×2 block H=[[1,α],[α,0]] applies exactly:
    //   P_{vac}(t) = 1 − 4α²/(1+4α²)·sin²(√(1+4α²)·t/2).
    // With α=1.5, ω=√10≈3.162; at t=1.0, sin²(ωt/2)≈1 → P(vac)≈0.10.
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_offdiag",
            serde_json::json!({"alphas": [1.5]}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 6,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
        },
    };
    let mut session = Session::new(&spec).expect("qfm offdiag session");

    let p_vac0 = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!((p_vac0 - 1.0).abs() < 1e-10, "starts in vacuum: {p_vac0}");

    let report = session.evolve(1.0).expect("evolve");
    assert!(
        (report.norm - 1.0).abs() < 1e-6,
        "post-evolve norm must be ~1 (unitary), got {}",
        report.norm
    );

    // Population must have left the vacuum (the whole point of the off-diagonal
    // coupling). The diagonal surrogate keeps P(vacuum)=1; here it must drop.
    let p_vac = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        p_vac < 0.5,
        "off-diagonal generator depopulates the vacuum: P(vac)={p_vac} (must be < 0.5)"
    );

    // And arrived in the data channel.
    let p_data0 = session.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        p_data0 > 0.1,
        "population arrives in data channel 0: P(x_0)={p_data0} (must be > 0.1)"
    );

    // The vacuum + ¬vacuum cover still sums to 1 (Born rule, normalization).
    let p_not = session
        .probability(&EventPredicate::not(EventPredicate::Vacuum))
        .expect("prob");
    assert!(
        (p_vac + p_not - 1.0).abs() < 1e-6,
        "cover sums to 1: {p_vac} + {p_not}"
    );
}

#[test]
fn qfm_mehler_offdiag_rabi_oscillation_round_trip() {
    // The Hermitian off-diagonal coupling gives COHERENT (unitary) Rabi
    // oscillation: amplitude flows vacuum → data, then back. At the full
    // oscillation period T = 2π/ω (ω=√(1+4α²) for one channel) the state
    // returns to the vacuum. Verify a half-period (max transfer) then another
    // half-period (full return) recovers P(vacuum) ≈ 1 — the signature of
    // coherent (not diffusive) transport that distinguishes the Hermitian unfer
    // realization from the paper's anti-Hermitian Fokker–Planck semigroup.
    let alpha = 1.5_f64;
    let omega = (1.0 + 4.0 * alpha * alpha).sqrt();
    let half_period = std::f64::consts::PI / omega;
    let spec = ModelSpec {
        hamiltonian: HamiltonianSpec::builtin(
            "qfm_mehler_offdiag",
            serde_json::json!({"alphas": [alpha]}),
        ),
        prior: PriorSpec::Vacuum,
        solver: SolverSpec {
            krylov_dim: 8,
            prune_eps: 1e-12,
            max_components: Some(50_000),
            restarts: 1,
            device: DeviceSpec::Cpu,
        },
    };
    let mut session = Session::new(&spec).expect("qfm offdiag session");

    // Half period: vacuum → maximum data-channel population.
    session.evolve(half_period).expect("evolve");
    let p_data_half = session.probability(&event_mode0_ge1()).expect("prob");
    assert!(
        p_data_half > 0.3,
        "half-period: maximum data transfer, P(x_0)={p_data_half} (must be > 0.3)"
    );

    // Another half period (full round trip): coherent return to the vacuum.
    // The SIRK approximation + krylov_dim=8 should recover P(vacuum) within a
    // few % — the signature of unitary (reversible) evolution.
    session.evolve(half_period).expect("evolve");
    let p_vac_full = session.probability(&EventPredicate::Vacuum).expect("prob");
    assert!(
        p_vac_full > 0.8,
        "full-period: coherent return to vacuum, P(vac)={p_vac_full} (must be > 0.8)"
    );
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
