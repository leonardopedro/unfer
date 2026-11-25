//! Emit the machine-readable finite-model witness row that `../timepiece`
//! `BookProof/Sirk/ChapterFiniteModelWitness.lean` transcribes.
//!
//! The row is the `g = 2, N <= 10` solve on the reflection-odd and reflection-even
//! blocks of `H_N`, with the `λ₂` lower bound `β` supplied by the dense count
//! certificate `certify_spectrum`.  The Ritz pair is the exact lowest eigenvector
//! of each in-sector block (its matvec residual is ~1e-14), so Temple's inequality
//! yields a lower bound on the gap of `H_N`.  The test checks the bound is positive
//! and does not exceed the exact truncated gap before writing anything.  Every
//! number emitted is therefore measured or certified here, not a claim about the
//! continuum theory.

use fock_sirk::certificate::certify_spectrum;
use fock_sirk::mass_gap_spec::certified_gap_lower_bound_with_exclusion;
use nalgebra::DMatrix;
use nested_fock_algebra::{
    Hamiltonian, InnerBosonicState, Operator, OuterState, QuantumState, qcd_ym_hamiltonian,
};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;

fn apply_reflection(s: &QuantumState) -> QuantumState {
    let mut out = QuantumState::zero();
    for (outer, amp) in &s.components {
        for (inner, mult) in &outer.bosonic {
            let n0 = inner.modes.get(&0).copied().unwrap_or(0);
            let n1 = inner.modes.get(&1).copied().unwrap_or(0);
            let mut flipped = InnerBosonicState::vacuum();
            for (mode, n) in &inner.modes {
                let target = match *mode {
                    0 => 1,
                    1 => 0,
                    other => other,
                };
                if *n > 0 {
                    flipped.modes.insert(target, *n);
                }
            }
            let phase = if (n0 + n1) * mult % 2 == 0 { 1.0 } else { -1.0 };
            out.components.insert(
                OuterState {
                    bosonic: BTreeMap::from([(flipped, *mult)]),
                    fermionic: BTreeSet::new(),
                },
                *amp * Complex64::new(phase, 0.0),
            );
        }
    }
    out
}

fn fock_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s = QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
    for &(m, n) in occ {
        for _ in 0..n {
            s = s.apply(&Operator::InnerBosonCreate(m));
        }
    }
    let norm = s.norm();
    if norm > 0.0 && (norm - 1.0).abs() > 1e-12 {
        for value in s.components.values_mut() {
            *value /= norm;
        }
    }
    s
}

struct Truncated {
    basis: Vec<QuantumState>,
    index: HashMap<OuterState, usize>,
    columns: Vec<Vec<(usize, f64)>>,
    reflection: Vec<(usize, f64)>,
}

impl Truncated {
    fn build(h: &Hamiltonian, n_max: u32) -> Self {
        let mut basis = Vec::new();
        for n0 in 0..=n_max {
            for n1 in 0..=n_max - n0 {
                for n2 in 0..=n_max - n0 - n1 {
                    for n3 in 0..=n_max - n0 - n1 - n2 {
                        let mut occ = Vec::new();
                        for (mode, count) in [(0, n0), (1, n1), (2, n2), (3, n3)] {
                            if count > 0 {
                                occ.push((mode, count));
                            }
                        }
                        basis.push(fock_state(&occ));
                    }
                }
            }
        }
        let mut index = HashMap::new();
        for (i, s) in basis.iter().enumerate() {
            index.insert(s.components.keys().next().unwrap().clone(), i);
        }
        let mut columns = Vec::new();
        for s in &basis {
            let hs = h.apply(s);
            columns.push(
                hs.components
                    .iter()
                    .filter_map(|(k, v)| index.get(k).map(|&i| (i, v.re)))
                    .collect(),
            );
        }
        let mut reflection = vec![(0usize, 0.0); basis.len()];
        for (i, s) in basis.iter().enumerate() {
            let rs = apply_reflection(s);
            let (key, sign) = rs.components.iter().next().map(|(k, v)| (k, v.re)).unwrap();
            reflection[i] = (index[key], sign);
        }
        Self { basis, index, columns, reflection }
    }

    fn dim(&self) -> usize {
        self.basis.len()
    }

    fn sector(&self, even: bool) -> Vec<Vec<(usize, f64)>> {
        let n = self.dim();
        let mut used = vec![false; n];
        let mut out = Vec::new();
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        for i in 0..n {
            if used[i] {
                continue;
            }
            let (j, sign) = self.reflection[i];
            if j == i {
                used[i] = true;
                if (sign > 0.0) == even {
                    out.push(vec![(i, 1.0)]);
                }
            } else {
                used[i] = true;
                used[j] = true;
                let s = if even { 1.0 } else { -1.0 };
                out.push(vec![(i, inv_sqrt2), (j, s * sign * inv_sqrt2)]);
            }
        }
        out
    }

    fn sector_matrix(&self, even: bool) -> DMatrix<f64> {
        let sector = self.sector(even);
        let k = sector.len();
        let mut a = DMatrix::<f64>::zeros(k, k);
        for (b, vector) in sector.iter().enumerate() {
            let mut hu = vec![0.0_f64; self.dim()];
            for &(i, coef) in vector {
                for &(row, value) in &self.columns[i] {
                    hu[row] += coef * value;
                }
            }
            for (c, other) in sector.iter().enumerate() {
                a[(c, b)] = other.iter().map(|&(i, coef)| coef * hu[i]).sum();
            }
        }
        a
    }
}

#[test]
fn emit_finite_model_witness_row() {
    let g = 2.0;
    let n_max = 10u32;
    let h = qcd_ym_hamiltonian(g);
    let trunc = Truncated::build(&h, n_max);
    let odd_block = trunc.sector_matrix(false);
    let even_block = trunc.sector_matrix(true);

    let certified = certify_spectrum(&odd_block).expect("odd count certificate");
    let beta = certified
        .second_eigenvalue_lower_bound()
        .expect("odd lambda_2 lower bound");

    let odd_eig = odd_block.clone().symmetric_eigen();
    let even_eig = even_block.clone().symmetric_eigen();
    let mut odd_values: Vec<(f64, usize)> = odd_eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    odd_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut even_values: Vec<(f64, usize)> = even_eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    even_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let theta_odd = odd_values[0].0;
    let theta_even = even_values[0].0;
    let odd_vector = odd_eig.eigenvectors.column(odd_values[0].1).into_owned();
    let even_vector = even_eig.eigenvectors.column(even_values[0].1).into_owned();
    let matvec = |a: &DMatrix<f64>, v: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> { a * v };
    let residual_odd = (matvec(&odd_block, &odd_vector) - theta_odd * odd_vector).norm();
    let _residual_even = (matvec(&even_block, &even_vector) - theta_even * even_vector).norm();

    let bound = certified_gap_lower_bound_with_exclusion(theta_odd, residual_odd, beta, theta_even)
        .expect("Temple must apply to the in-sector odd solve");
    let exact_gap = theta_odd - theta_even;
    let lambda_odd_2 = odd_values[1].0;

    assert!(bound > 0.0, "the emitted witness must be positive: {bound}");
    assert!(
        bound <= exact_gap + 1e-9,
        "the bound {bound} must not exceed the exact truncated gap {exact_gap}"
    );
    assert!(
        beta <= lambda_odd_2 + 1e-9,
        "the count witness beta {beta} must not exceed the exact odd lambda_2 {lambda_odd_2}"
    );
    assert!(
        residual_odd < 1e-10,
        "the in-sector ground residual must be tiny: {residual_odd}"
    );

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/finite_model_witness");
    fs::create_dir_all(&dir).expect("fixture dir");
    let line = serde_json::json!({
        "kind": "finite_model_witness",
        "hamiltonian": "qcd_ym_hamiltonian(g)",
        "coupling": g,
        "cutoff": n_max,
        "dimension": odd_block.nrows(),
        "solver": "exact in-sector eigenvector with dense count witness",
        "theta_odd": theta_odd,
        "residual_odd": residual_odd,
        "theta_even": theta_even,
        "beta_lambda2_odd": beta,
        "temple_bound": bound,
        "exact_gap_at_cutoff": exact_gap,
        "physical": false
    });
    fs::write(dir.join("qym_g2_n10_count.ndjson"), format!("{line}\n"))
        .expect("write fixture");
    eprintln!(
        "emitted finite-model witness: g={g}, N<={n_max}, temple={bound:.9}, exact={exact_gap:.9}, beta={beta:.9}"
    );
}
