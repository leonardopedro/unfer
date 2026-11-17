//! Emit the exact one-mode CAR bridge fixture (N13).
//!
//! The matrices come from the ordered `InnerExpr` action on the two-state
//! fermion basis, not from a separate linear-algebra model. Run:
//!
//! ```text
//! cargo run -q -p fock_sirk --example sm_car_bridge
//! ```
use fock_sirk::sm_bridge::{bind_artifact, canonical_json_sha256};
use nested_fock_algebra::exact_inner::{
    ActionLimits, InnerExpr as E, InnerLadder, InnerVector, JointInnerBasis as B,
};
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn matrix(expr: &E, basis: &[B]) -> Value {
    let limits = ActionLimits::default();
    let mut rows = vec![vec![[0i64, 0i64]; basis.len()]; basis.len()];
    for (column, state) in basis.iter().enumerate() {
        let input: InnerVector = BTreeMap::from([(state.clone(), C::new(1.0, 0.0))]);
        let image = expr.apply(&input, limits).expect("exact CAR action");
        for (label, amplitude) in image {
            let row = basis
                .iter()
                .position(|candidate| candidate == &label)
                .expect("image stays in the two-state basis");
            assert_eq!(amplitude.im, 0.0, "CAR coefficients are real");
            assert_eq!(
                amplitude.re.fract(),
                0.0,
                "CAR coefficients are integers, got {}",
                amplitude.re
            );
            rows[row][column] = [amplitude.re as i64, 0];
        }
    }
    Value::Array(
        rows.into_iter()
            .map(|row| Value::Array(row.into_iter().map(|entry| json!(entry)).collect()))
            .collect(),
    )
}

fn main() {
    let vacuum = B::vacuum();
    let occupied = B::new([], [0u32]).unwrap();
    let basis = [vacuum, occupied];

    let annihilate = E::Ladder(InnerLadder::FermionAnnihilate(0));
    let create = E::Ladder(InnerLadder::FermionCreate(0));
    let number = E::Product(vec![create.clone(), annihilate.clone()]);
    let parity = E::Sum(vec![
        E::Scalar(C::new(1.0, 0.0)),
        E::Product(vec![E::Scalar(C::new(-2.0, 0.0)), number.clone()]),
    ]);

    let matrices = json!({
        "annihilate": matrix(&annihilate, &basis),
        "create": matrix(&create, &basis),
        "number": matrix(&number, &basis),
        "parity": matrix(&parity, &basis),
    });
    let state_names = ["fermion_n0", "fermion_n1"];
    let operator_names = [
        "FermionAnnihilate(0)",
        "FermionCreate(0)",
        "FermionCreate(0) * FermionAnnihilate(0)",
        "1 - 2 * FermionCreate(0) * FermionAnnihilate(0)",
    ];

    let fixture = bind_artifact(json!({
        "schema": "sm-proof-bridge/v1",
        "claim_id": "one_mode_fermion_car",
        "status": "certified",
        "endpoint": "E1-unreduced",
        "operator": {
            "name": "ordered one-mode CAR ladders from nested_fock_algebra::exact_inner",
            "identity_sha256": canonical_json_sha256(&json!(operator_names)),
            "source": "nested_fock_algebra::exact_inner::InnerLadder::FermionCreate/FermionAnnihilate",
        },
        "state": {
            "basis": state_names,
            "identity_sha256": canonical_json_sha256(&json!(state_names)),
            "note": "all two fermionic occupations; vacuum is the inner vacuum",
        },
        "artifact": {
            "matrices": matrices,
            "field": "integer Gaussian entries; imaginary part is always zero",
            "exactness": "operator algebra is exact; this is not a floating-point enclosure",
        },
        "theorem": {
            "lean_name": "BookProof.Sm.ChapterSmCar.one_mode_anticommutator",
            "statement": "a * adag + adag * a = 1",
            "supporting_statements": [
                "a * a = 0",
                "adag * adag = 0",
                "parity * a * parity = -a",
                "adag * a = number",
            ],
            "proof_status": "kernel-verified",
            "source": "fock_sirk/lean/SmCar.lean",
            "remaining": "The Lean definitions are bound to this artifact by the shared integer matrices and the Rust re-derivation; no formal proof connects the Lean Mat2 structure to nested_fock_algebra definitionally.",
        },
        "proof": {
            "checker": "nanoda_lib 0.4.13 via prob_kernel::verify::verify_export",
            "payload": "docs/fixtures/sm_car_proof.ndjson",
            "payload_sha256": "63ef7a554198a5bf9eb04b73c2c9e61950b97689c9265eb4327e0cccdf17c1e5",
            "declarations_checked": 187,
            "lean_version": "4.28.0",
            "lean_githash": "7e01a1bf5c70fc6167d49c345d3bf80596e9a79b",
            "exporter": "lean4export 3.1.0",
        },
        "hypotheses": [
            "One fermionic mode, occupation truncated to n in {0,1}; this is the exact CAR Fock space of that mode.",
            "The operator is the ordered nested_fock_algebra InnerExpr action, not a normal-ordered or scalar-stripped rewrite.",
            "Matrix entries are the actual image coefficients in the declared two-state basis.",
        ],
        "not_claimed": [
            "No multi-mode CAR or graded outer-statistics claim.",
            "No SM Hamiltonian, physical-state or continuum claim.",
            "No identification of the Lean Mat2 model with the full InnerExpr algebra beyond the six checked finite identities.",
        ],
    }));

    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
