//! N9 Mehler-width feasibility at a fixed E1 state and fixed harmonic operator.
//!
//! The state is the same two-fermion seed used by the SM baseline; all 163
//! bosonic modes are in vacuum and are squeezed by the unitary dilation
//! `q=ell(a+a†)`, `p=i(a†-a)/(2 ell)`. The exact squeezed-vacuum Fock
//! coefficients give the **basis tail** above a total-boson cutoff. The exact
//! quadratic energy gives a fixed-operator expectation. No SIRK solve is run, so
//! this reports a representation cost and tail, not a solver-error comparison;
//! that limitation is part of the output.

const GROUPS: [usize; 3] = [1, 40, 163];

/// Squeezed-vacuum probabilities `|c_{2m}|^2` for `ell=e^r`, up to degree `cut`.
fn squeezed_probabilities(ell: f64, cut: usize) -> Vec<f64> {
    let r = ell.ln();
    let t = r.tanh();
    let norm = 1.0 / r.cosh().sqrt();
    let mut out = vec![0.0; cut + 1];
    let mut coeff = norm;
    out[0] = coeff * coeff;
    for m in 1..=cut / 2 {
        // c_{2m}/c_{2m-2} = t * sqrt((2m-1)(2m)) / (2m)
        let mf = m as f64;
        coeff *= t * ((2.0 * mf - 1.0) * 2.0 * mf).sqrt() / (2.0 * mf);
        out[2 * m] = coeff * coeff;
    }
    out
}

/// Total-boson distribution for `modes` identical squeezed vacua, truncated at
/// degree `cut`. Returns the retained weight at each total occupation <= cut.
fn total_distribution(per_mode: &[f64], cut: usize, modes: usize) -> Vec<f64> {
    let mut poly = vec![0.0; cut + 1];
    poly[0] = 1.0;
    for _ in 0..modes {
        let mut next = vec![0.0; cut + 1];
        for i in 0..=cut {
            if poly[i] == 0.0 {
                continue;
            }
            for j in 0..=cut - i {
                next[i + j] += poly[i] * per_mode[j];
            }
        }
        poly = next;
    }
    poly
}

fn harmonic_energy(ell: f64) -> f64 {
    // <0_ell| (p^2+q^2)/2 |0_ell>, with q=ell(a+a†), p=i(a†-a)/(2ell).
    ell * ell / 4.0 + 1.0 / (8.0 * ell * ell)
}

fn main() {
    let cuts = [0usize, 2, 4, 6, 8, 12];
    let widths = [1.0f64, 1.25, 1.5, 2.0, 3.0, 4.0];
    let mut rows = Vec::new();
    for ell in widths {
        let per_mode = squeezed_probabilities(ell, *cuts.iter().max().unwrap());
        let mut row = serde_json::Map::new();
        row.insert("ell".into(), serde_json::json!(ell));
        row.insert(
            "mean_bosons_per_mode".into(),
            serde_json::json!((ell * ell + 1.0 / (ell * ell) - 2.0) / 4.0),
        );
        row.insert(
            "harmonic_energy_per_mode".into(),
            serde_json::json!(harmonic_energy(ell)),
        );
        let mut groups_out = serde_json::Map::new();
        for modes in GROUPS {
            let dist = total_distribution(&per_mode, *cuts.iter().max().unwrap(), modes);
            let mut tails = serde_json::Map::new();
            for &cut in &cuts {
                let retained: f64 = dist[..=cut.min(dist.len() - 1)].iter().sum();
                tails.insert(
                    format!("{cut}"),
                    serde_json::json!({
                        "retained": retained,
                        "tail": (1.0 - retained).max(0.0),
                        "dimension_total_occupation_le_cut": binomial(modes + cut, cut),
                    }),
                );
            }
            let tol_cut = cuts.iter().copied().find(|&cut| {
                let retained: f64 = dist[..=cut.min(dist.len() - 1)].iter().sum();
                1.0 - retained <= 1e-8
            });
            groups_out.insert(
                format!("{modes}"),
                serde_json::json!({
                    "modes": modes,
                    "mean_total_bosons": modes as f64 * (ell * ell + 1.0/(ell*ell) - 2.0)/4.0,
                    "cuts": tails,
                    "cut_at_tail_1e-8": tol_cut,
                }),
            );
        }
        row.insert("groups".into(), serde_json::Value::Object(groups_out));
        rows.push(serde_json::Value::Object(row));
    }
    let out = serde_json::json!({
        "experiment": "N9 Mehler-width representation feasibility",
        "fixed_state": "two-fermion E1 seed; one, 40 active or all 163 bosonic vacua squeezed",
        "fixed_operator": "harmonic part (p^2+q^2)/2 only; full interacting H not evolved",
        "unitary_map": "q=ell(a+a^dagger), p=i(a^dagger-a)/(2 ell); ell=e^r",
        "errors": {
            "basis_tail": "exact squeezed coefficients truncated at total boson cutoff",
            "solver": "not measured: no SIRK solve in this feasibility cell",
            "arithmetic": "f64 polynomial convolution; no interval enclosure",
        },
        "claims": {
            "kind": "representation cost and tail, not an equal-tolerance SIRK speedup",
            "certified": false,
        },
        "rows": rows,
        "limitations": [
            "The unitary dilation is exact on E1, but the occupation window is not closed under it.",
            "Equal-tolerance solver work and N8 defect require a squeezed-state SIRK run; not done here.",
            "The harmonic energy is not the complete interacting Hamiltonian expectation."
        ],
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

/// Small non-overflowing binomial for the dimension column; `n` is large.
fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n.saturating_sub(k));
    let mut value = 1.0f64;
    for i in 0..k {
        value *= (n - i) as f64 / (i + 1) as f64;
        if !value.is_finite() {
            return f64::INFINITY;
        }
    }
    value
}
