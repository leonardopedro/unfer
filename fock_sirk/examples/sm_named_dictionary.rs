//! Export the N7 named-coordinate SM dictionary fixture.
//!
//! The fixture is generated from the native `SmHamiltonian` builder and checked
//! in. `fock_sirk/tests/sm_named_dictionary.rs` verifies it against both the
//! native builder and the independently imported Cadabra expression, then runs
//! the listed sparse action cells. Run with no arguments and redirect stdout:
//!
//! ```text
//! cargo run -p fock_sirk --example sm_named_dictionary > \
//!     docs/fixtures/sm_named_dictionary.json
//! ```

use fock_sirk::joint_sirk::joint_norm;
use fock_sirk::sm_bridge::reported_residual;
use nested_fock_algebra::exact_inner::{
    ActionLimits, JointInnerBasis as B, JointOuterBasis as O, NestedVector,
};
use nested_fock_algebra::sm_full::*;
use num_complex::Complex64 as C;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

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

fn matrix_json(matrix: &[[C; 3]; 3]) -> Value {
    Value::Array(
        matrix
            .iter()
            .map(|row| Value::Array(row.iter().map(|c| json!([c.re, c.im])).collect()))
            .collect(),
    )
}

fn parameters_json(p: &SmParameters) -> Value {
    json!({
        "gs": p.gs, "gw": p.gw, "gy": p.gy, "lambda": p.lambda, "vev": p.vev,
        "yu": matrix_json(&p.yu), "yd": matrix_json(&p.yd),
        "ynu": matrix_json(&p.ynu), "ye": matrix_json(&p.ye),
    })
}

fn coordinate_name(m: u32) -> String {
    for i in 0..3 {
        if m == coord::x(i) {
            return format!("x_{i}");
        }
    }
    for a in 0..8 {
        for i in 0..3 {
            if m == coord::g(a, i) {
                return format!("g_{a}_{i}");
            }
            for j in 0..3 {
                if m == coord::dg(a, i, j) {
                    return format!("dg_{a}_{i}_{j}");
                }
            }
        }
    }
    for a in 0..3 {
        for i in 0..3 {
            if m == coord::w(a, i) {
                return format!("w_{a}_{i}");
            }
            for j in 0..3 {
                if m == coord::dw(a, i, j) {
                    return format!("dw_{a}_{i}_{j}");
                }
            }
        }
    }
    for i in 0..3 {
        if m == coord::b(i) {
            return format!("b_{i}");
        }
        for j in 0..3 {
            if m == coord::db(i, j) {
                return format!("db_{i}_{j}");
            }
        }
    }
    for a in 0..4 {
        if m == coord::phi(a) {
            return format!("phi_{a}");
        }
        for i in 0..3 {
            if m == coord::dphi(a, i) {
                return format!("dphi_{a}_{i}");
            }
        }
    }
    panic!("unnamed coordinate {m}");
}

fn variable_name(v: Variable) -> String {
    match v {
        Variable::Q(m) => format!("q:{}", coordinate_name(m)),
        Variable::P(m) => format!("p:{}", coordinate_name(m)),
    }
}

fn form_json(form: &WeightedForm) -> Value {
    json!({
        "name": form.name,
        "weight": form.weight,
        "terms": form.polynomial.terms.iter().map(|(word, c)| json!([
            word.iter().map(|v| match v {
                Variable::Q(m) => json!(["q", m]),
                Variable::P(m) => json!(["p", m]),
            }).collect::<Vec<_>>(),
            [c.re, c.im],
        ])).collect::<Vec<_>>(),
    })
}

fn state_digest(state: &NestedVector) -> String {
    let mut hash = Sha256::new();
    for (basis, amplitude) in state {
        hash.update(
            format!(
                "{basis:?}|{}+{}i\n",
                amplitude.re.to_bits(),
                amplitude.im.to_bits()
            )
            .as_bytes(),
        );
    }
    format!("{:x}", hash.finalize())
}

fn action_states() -> Vec<(&'static str, &'static str, NestedVector, u64)> {
    use Chirality::*;
    use Species::*;
    let up = B::new(
        [],
        [
            matter_mode(Up, 0, 0, Left, 0),
            matter_mode(Down, 0, 0, Left, 1),
        ],
    )
    .unwrap();
    let down_r = B::new([], [matter_mode(Down, 2, 2, Right, 1)]).unwrap();
    let electron = B::new([], [matter_mode(Electron, 1, 0, Left, 1)]).unwrap();
    let neutrino = B::new([], [matter_mode(Neutrino, 2, 0, Right, 0)]).unwrap();
    let mixed_even = B::new([(coord::phi(2), 1)], []).unwrap();
    let mixed_odd = B::new([], [matter_mode(Up, 0, 0, Left, 0)]).unwrap();
    vec![
        (
            "vacuum_inner_one_universe",
            "matter=0;color=none;generation=none;chirality=none;outer=even",
            NestedVector::from([(O::graded([(B::vacuum(), 1)]).unwrap(), C::new(1.0, 0.0))]),
            1,
        ),
        (
            "quark_doublet_two_fermions",
            "matter=2;color=0;generation=0;chirality=left;outer=even",
            NestedVector::from([(O::graded([(up, 1)]).unwrap(), C::new(1.0, 0.0))]),
            2,
        ),
        (
            "down_right_generation2_color2",
            "matter=1;color=2;generation=2;chirality=right;outer=odd",
            NestedVector::from([(O::graded([(down_r, 1)]).unwrap(), C::new(0.6, 0.3))]),
            1,
        ),
        (
            "electron_left_generation1",
            "matter=1;color=0;generation=1;chirality=left;outer=odd",
            NestedVector::from([(O::graded([(electron, 1)]).unwrap(), C::new(0.5, -0.2))]),
            1,
        ),
        (
            "neutrino_right_generation2",
            "matter=1;color=0;generation=2;chirality=right;outer=odd",
            NestedVector::from([(O::graded([(neutrino, 1)]).unwrap(), C::new(0.7, 0.1))]),
            1,
        ),
        (
            "two_universes_mixed_parity",
            "matter=2;color=0;generation=0/1;chirality=left;outer=two_universes_graded",
            NestedVector::from([(
                O::graded([(mixed_even, 1), (mixed_odd.clone(), 1)]).unwrap(),
                C::new(0.4, 0.0),
            )]),
            2,
        ),
        (
            "two_universes_boson_excitation",
            "matter=0;color=none;generation=none;chirality=none;outer=two_universes_graded",
            NestedVector::from([(
                O::graded([
                    (B::new([(coord::g(3, 1), 1)], []).unwrap(), 1),
                    (mixed_odd, 1),
                ])
                .unwrap(),
                C::new(0.3, -0.1),
            )]),
            2,
        ),
    ]
}

fn state_spec(state: &NestedVector) -> Value {
    let (basis, amplitude) = state.iter().next().expect("one outer basis per cell");
    let universes: Vec<Value> = basis
        .inner_occupations()
        .iter()
        .map(|(b, n)| {
            json!({
                "bosons": b.bosons().iter().map(|(&m, &k)| json!([m, k])).collect::<Vec<_>>(),
                "fermions": b.fermions().iter().copied().collect::<Vec<_>>(),
                "multiplicity": n,
            })
        })
        .collect();
    json!({"universes": universes, "amplitude": [amplitude.re, amplitude.im]})
}

fn action_cell(
    h: &nested_fock_algebra::exact_inner::InnerExpr,
    label: &str,
    sector: &str,
    state: NestedVector,
    multiplicity: u64,
) -> Value {
    let limits = ActionLimits {
        max_components: 200_000,
        max_work: 200_000_000,
    };
    let spec = state_spec(&state);
    let image = h.d_gamma(&state, limits).expect("action within budget");
    json!({
        "label": label,
        "sector": sector,
        "state": spec,
        "state_sha256": state_digest(&state),
        "outer_particle_number": state.keys().map(|b| b.particle_number()).sum::<u64>(),
        "universe_multiplicity": multiplicity,
        "support": image.len(),
        // The one measured quantity of a cell; the digests above and below stay
        // bit-exact. Twelve significant digits is far inside the 1e-9 check.
        "norm": reported_residual(joint_norm(&image)),
        "action_sha256": state_digest(&image),
        "limits": {"max_components": limits.max_components, "max_work": limits.max_work},
    })
}

fn main() {
    let model = SmHamiltonian::new(parameters()).expect("valid parameter instance");
    let h = model.expression();
    let names: Vec<Value> = (0..163).map(|m| json!([coordinate_name(m), m])).collect();
    let matter_entries: Vec<Value> = model
        .matter
        .iter()
        .flat_map(|(&variable, matrix)| {
            matrix
                .iter()
                .map(move |(&(r, s), &c)| json!([variable_name(variable), r, s, c.re, c.im]))
        })
        .collect();
    let higgs: Vec<Value> = (0..4)
        .map(|a| {
            let up: Vec<Value> = (0..2)
                .flat_map(|weak| {
                    (0..4).map(move |component| {
                        let z = higgs_yukawa_component(true, weak, component);
                        json!([weak, component, z.re, z.im])
                    })
                })
                .collect();
            let down: Vec<Value> = (0..2)
                .flat_map(|weak| {
                    (0..4).map(move |component| {
                        let z = higgs_yukawa_component(false, weak, component);
                        json!([weak, component, z.re, z.im])
                    })
                })
                .collect();
            json!({
                "name": coordinate_name(coord::phi(a)),
                "coordinate": coord::phi(a),
                "wall_quadratic_coefficient": 1.0,
                "up_type_components": up,
                "down_type_components": down,
            })
        })
        .collect();
    let kinetic: Vec<Value> = (0..3)
        .map(|axis| {
            let block = book_kinetic_spin(axis);
            let c = |x: C| json!([x.re, x.im]);
            json!({
                "axis": axis,
                "left": [[c(-block[0][0]), c(-block[0][1])], [c(-block[1][0]), c(-block[1][1])]],
                "right": [[c(block[0][0]), c(block[0][1])], [c(block[1][0]), c(block[1][1])]],
            })
        })
        .collect();
    let actions: Vec<Value> = action_states()
        .into_iter()
        .map(|(label, sector, state, multiplicity)| {
            action_cell(&h, label, sector, state, multiplicity)
        })
        .collect();
    let fixture = json!({
        "schema": "sm-named-dictionary/v1",
        "status": "source-present",
        "endpoint": "E1-unreduced",
        "generated_by": "fock_sirk/examples/sm_named_dictionary.rs",
        "parameters": parameters_json(&model.parameters),
        "model_sha256": {
            "parameters": Sha256::digest(format!("{:?}", model.parameters).as_bytes())
                .iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "forms": Sha256::digest(format!("{:?}", model.forms).as_bytes())
                .iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "matter": Sha256::digest(format!("{:?}", model.matter).as_bytes())
                .iter().map(|b| format!("{b:02x}")).collect::<String>(),
        },
        "coordinates": {
            "count": names.len(),
            "active_count": coord::active().len(),
            "names": names,
            "active": coord::active(),
        },
        "higgs_slots": higgs,
        "kinetic_blocks": kinetic,
        "gauge": {
            "su3_generators": su3_generators().iter().map(|t| Value::Array(
                t.iter().map(|row| Value::Array(row.iter().map(|c| json!([c.re, c.im])).collect())).collect()
            )).collect::<Vec<_>>(),
            "su3_structure": su3_structure().iter().map(|a| Value::Array(
                a.iter().map(|b| Value::Array(b.iter().map(|x| json!(x)).collect())).collect()
            )).collect::<Vec<_>>(),
            "weak_epsilon": (0..3).map(|a| (0..3).map(|b| (0..3)
                .map(|c| epsilon(a, b, c)).collect::<Vec<_>>()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "hypercharge": [
                ["up_left", Species::Up.hypercharge(Chirality::Left)],
                ["down_left", Species::Down.hypercharge(Chirality::Left)],
                ["neutrino_left", Species::Neutrino.hypercharge(Chirality::Left)],
                ["electron_left", Species::Electron.hypercharge(Chirality::Left)],
                ["up_right", Species::Up.hypercharge(Chirality::Right)],
                ["down_right", Species::Down.hypercharge(Chirality::Right)],
                ["neutrino_right", Species::Neutrino.hypercharge(Chirality::Right)],
                ["electron_right", Species::Electron.hypercharge(Chirality::Right)],
            ],
        },
        "forms": model.forms.iter().map(form_json).collect::<Vec<_>>(),
        "matter": {
            "variable_count": model.matter.len(),
            "entry_count": matter_entries.len(),
            "unique_matrix_entries": model.matter_entry_polynomials().len(),
            "entries": matter_entries,
        },
        "action_fixtures": actions,
        "formal_bridges": [
            {
                "binds": "coordinate count and named bijection",
                "lean": "BookProof.Sm.ChapterSmOneParticle.card_smCoord",
                "status": "source-present",
                "remaining": "Lean's noncomputable smIdx is not yet proved equal to these names."
            },
            {
                "binds": "96 complex / 192 real matter representation and parity",
                "lean": "BookProof.Sm.ChapterSmMatterRepresentation",
                "status": "source-present",
                "remaining": "The data-to-normal-form map does not yet identify every Rust Dirac/gauge matrix."
            },
            {
                "binds": "four Higgs slots and all four complex Yukawa matrices",
                "lean": "BookProof.Sm.ChapterSmFullYukawaData + ChapterSmFullYukawaCoeffs",
                "status": "source-present",
                "remaining": "The action-agreement theorem for these named coefficients is proposed, not proved."
            },
            {
                "binds": "full inner normal form and constant retention",
                "lean": "BookProof.Sm.ChapterSmFullMatterHamiltonian.smNormalForm",
                "status": "source-present",
                "remaining": "smNormalForm has no quartic bosonic magnetic/Higgs family; h_B is assembled separately."
            },
            {
                "binds": "potential self-bound reused by the comparison",
                "lean": "BookProof.Sm.ChapterSmGradTable.exists_potSelfBound_sm",
                "status": "source-present",
                "remaining": "The 43-direction graph core and quadratic-form commutator remain open."
            }
        ],
        "limits": {
            "native_forms": model.forms.len(),
            "native_matter_variables": model.matter.len(),
            "native_matter_entries": model.matter_entry_polynomials().len(),
            "note": "Action budget failure aborts a cell; it never truncates a form."
        },
    });
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
