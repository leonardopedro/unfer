//! N10 proof-facing bridge for the finite constraint dictionary.
//!
//! This binds the measured finite-image identities of the twelve Gauss
//! generators, the joint BRST charge and the Hamiltonian commutator. It does
//! **not** construct a field-level physical quotient or its representative map;
//! a finite adjoint Chevalley--Eilenberg quotient is bound separately under
//! `artifact.finite_ce_quotient`. The computations are the same ones exercised
//! by `fock_sirk/tests/sm_joint_constraints.rs`; this module exists so the
//! fixture can be regenerated and bound as one artifact.

use crate::joint_sirk::{joint_axpy, joint_norm};
use crate::sm_bridge::{BRIDGE_SCHEMA, bind_artifact, canonical_json_sha256};
use crate::sm_brst_quotient::finite_brst_quotient;
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_constraints::*;
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::sync::OnceLock;

fn parameters() -> SmParameters {
    let matrix = |scale: f64| {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                C::new(
                    scale * (1 + i + 2 * j) as f64,
                    scale * 0.17 * (1 + 2 * i + j) as f64,
                )
            })
        })
    };
    SmParameters {
        gs: 0.3,
        gw: 0.2,
        gy: 0.15,
        lambda: 0.1,
        vev: 0.7,
        yu: matrix(0.02),
        yd: matrix(0.013),
        ynu: matrix(0.007),
        ye: matrix(0.019),
    }
}

fn model() -> &'static SmHamiltonian {
    static MODEL: OnceLock<SmHamiltonian> = OnceLock::new();
    MODEL.get_or_init(|| {
        cadabra_import::from_cadabra_latex(
            parameters(),
            include_str!("../../docs/fixtures/sm_full_cadabra.latex"),
        )
        .unwrap()
    })
}

fn h() -> &'static InnerExpr {
    static H: OnceLock<InnerExpr> = OnceLock::new();
    H.get_or_init(|| model().expression())
}

fn limits() -> ActionLimits {
    ActionLimits {
        max_components: 2_000_000,
        max_work: 20_000_000_000,
    }
}

fn seed() -> NestedVector {
    let inner = B::new(
        [],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}

fn with_gauge() -> NestedVector {
    let inner = B::new(
        [(coord::g(1, 0), 1), (coord::g(2, 0), 1)],
        [
            matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
            matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
        ],
    )
    .unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}

fn with_ghosts(ghosts: &[u32]) -> NestedVector {
    let mut fermions = vec![
        matter_mode(Species::Up, 0, 0, Chirality::Left, 0),
        matter_mode(Species::Neutrino, 0, 0, Chirality::Left, 0),
    ];
    fermions.extend(ghosts.iter().map(|g| GHOST_BASE + g));
    let inner = B::new([], fermions).unwrap();
    NestedVector::from([(O::graded([(inner, 1)]).unwrap(), C::new(1.0, 0.0))])
}

fn apply(op: &InnerExpr, state: &NestedVector) -> NestedVector {
    op.d_gamma(state, limits()).unwrap()
}

fn residual_norm(lhs: &NestedVector, rhs: &NestedVector) -> f64 {
    let mut residual = lhs.clone();
    joint_axpy(&mut residual, rhs, C::new(-1.0, 0.0), limits()).unwrap();
    joint_norm(&residual)
}

fn gauss_closure(
    generators: &[InnerExpr],
    state: &NestedVector,
    count: usize,
) -> (f64, (usize, usize)) {
    let mut worst = 0.0_f64;
    let mut worst_pair = (0, 0);
    for a in 0..count {
        for b in 0..count {
            let lhs = apply(&generators[a], &apply(&generators[b], state));
            let rhs = apply(&generators[b], &apply(&generators[a], state));
            let mut commutator = lhs;
            joint_axpy(&mut commutator, &rhs, C::new(-1.0, 0.0), limits()).unwrap();
            let mut expected = NestedVector::new();
            for (c, generator) in generators.iter().enumerate() {
                let f = structure(a, b, c);
                if f != 0.0 {
                    for (basis, value) in apply(generator, state) {
                        *expected.entry(basis).or_default() += C::new(0.0, f) * value;
                    }
                }
            }
            let residual = residual_norm(&commutator, &expected);
            if residual > worst {
                worst = residual;
                worst_pair = (a, b);
            }
        }
    }
    (worst, worst_pair)
}

fn brst_worst() -> (f64, usize) {
    let q = brst_charge(&parameters());
    let mut worst = 0.0_f64;
    let mut checked = 0;
    for g in [0u32, 1, 2, 3, 7, 8, 9, 10, 11] {
        for state in [with_ghosts(&[g]), with_ghosts(&[g, (g + 5) % 12])] {
            let image = apply(&q, &apply(&q, &state));
            worst = worst.max(joint_norm(&image));
            checked += 1;
        }
    }
    (worst, checked)
}

fn hamiltonian_gauss_worst(generators: &[InnerExpr]) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut checked = 0;
    for state in [seed(), with_gauge()] {
        for generator in generators {
            let hg = apply(h(), &apply(generator, &state));
            let gh = apply(generator, &apply(h(), &state));
            let mut residual = hg;
            joint_axpy(&mut residual, &gh, C::new(-1.0, 0.0), limits()).unwrap();
            worst = worst.max(joint_norm(&residual));
            checked += 1;
        }
    }
    (worst, checked)
}

fn structure_json() -> Value {
    Value::Array(
        (0..GENERATORS)
            .map(|a| {
                Value::Array(
                    (0..GENERATORS)
                        .map(|b| {
                            Value::Array(
                                (0..GENERATORS).map(|c| json!(structure(a, b, c))).collect(),
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Build the N10 finite-dictionary bridge. The record is `measured`: it binds
/// finite-image identities and leaves the physical quotient unimplemented.
pub fn finite_constraint_bridge() -> Value {
    let operators = gauge_dictionary(&parameters());
    assert_eq!(operators.len(), GENERATORS);
    let labels = generators()
        .into_iter()
        .map(|(_, label)| label)
        .collect::<Vec<_>>();
    let (gauss_vacuum, gauss_vacuum_pair) = gauss_closure(&operators, &seed(), 12);
    let (gauss_occupied, gauss_occupied_pair) = gauss_closure(&operators, &with_gauge(), 8);
    let (brst, brst_images) = brst_worst();
    let (hamiltonian, hamiltonian_images) = hamiltonian_gauss_worst(&operators);
    let operator_identity = canonical_json_sha256(&json!({
        "generators": labels,
        "structure": structure_json(),
        "ghost_base": GHOST_BASE,
        "matter_modes": MATTER_MODES,
    }));
    let state_identity = canonical_json_sha256(&json!({
        "basis": ["gauge_vacuum_matter_seed", "two_gauge_quanta", "one_ghost_images"],
        "note": "finite images, not a quotient basis",
    }));

    bind_artifact(json!({
        "schema": BRIDGE_SCHEMA,
        "claim_id": "n10_finite_constraint_dictionary",
        "status": "measured",
        "endpoint": "N10-finite-dictionary",
        "operator": {
            "name": "twelve homogeneous SU(3)xSU(2)xU(1) Gauss generators and joint BRST charge",
            "identity_sha256": operator_identity,
            "source": "nested_fock_algebra::sm_constraints",
        },
        "state": {
            "basis": ["gauge_vacuum_matter_seed", "two_gauge_quanta", "one_ghost_images"],
            "identity_sha256": state_identity,
            "note": "finite images only; no field-level representative/quotient map is bound (the finite adjoint CE quotient is separate)",
        },
        "artifact": {
            "gauss_commutator_gauge_vacuum": {
                "pairs": 12 * 12,
                "worst_residual": gauss_vacuum,
                "worst_pair": [gauss_vacuum_pair.0, gauss_vacuum_pair.1],
                "tolerance": 2e-9,
            },
            "gauss_commutator_gauge_occupied": {
                "pairs": 8 * 8,
                "worst_residual": gauss_occupied,
                "worst_pair": [gauss_occupied_pair.0, gauss_occupied_pair.1],
                "tolerance": 2e-9,
            },
            "brst_nilpotency": {
                "images": brst_images,
                "worst_residual": brst,
                "tolerance": 2e-9,
            },
            "hamiltonian_gauss_commutator": {
                "images": hamiltonian_images,
                "worst_residual": hamiltonian,
                "tolerance": 2e-9,
            },
            "finite_dictionary": true,
            "finite_ce_quotient": finite_brst_quotient(),
            "quotient_map": null,
            "quotient_note": "No field-level physical quotient or BRST cohomology is constructed; the finite adjoint Chevalley-Eilenberg quotient is bound separately under finite_ce_quotient, while these are finite-image residual checks only.",
        },
        "theorem": {
            "lean_name": "Analytic.SmConstraints.finite_dictionary",
            "statement": "sum_a c_a G_a and the joint Q close on the declared finite images, and Q^2 = 0 there",
            "proof_status": "not-run",
            "remaining": "The finite identities are measured in the complete E1 operator. The nested finite adjoint CE quotient is constructed separately, but no Lean theorem binds these finite identities. A direct probe shows the joint charge removes one ghost and preserves the remaining occupation, so the total-occupation filtration is Q-invariant and a finite field-level complex is constructible; that full Fock matrix and its cohomology are not yet bound. The field-level quotient/cohomology and anomaly analysis remain open.",
        },
        "hypotheses": [
            "The twelve generators use all eight SU(3) colours, the three SU(2) generators and U(1)_Y with bare structure constants.",
            "Ghosts are the twelve modes at GHOST_BASE..GHOST_BASE+12; the joint charge is Q = sum c_a G_a - (i/2) f_abc c_a c_b b_a.",
            "Residuals are f64 finite-image operator norms; they are not interval enclosures or continuum claims.",
            "The Hamiltonian is the complete E1 Cadabra-derived operator, not a projected window.",
        ],
        "not_claimed": [
            "No physical BRST cohomology or gauge-invariant Hilbert space.",
            "No representative/quotient map; quotient_map is deliberately null.",
            "No anomaly, infinite-mode, cutoff or continuum statement.",
            "No certified interval enclosure of the measured residuals.",
        ],
    }))
}
