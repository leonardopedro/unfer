//! PRIMARY MODEL — continuous-momentum nested-Fock λφ⁴ inner Hamiltonian.
//!
//! This is the `book.tex` operator of `docs/PHI4_ANALYTIC_PLAN.md`:
//!
//! ```text
//! h_cont(k, φ) = ½ p_φ² + ½ (k₁²+k₂²+k₃²) φ² + (λ/4)(φ² − v²)²
//! ```
//!
//! on `L²(ℝ³_k × ℝ_φ)`.  The four coordinates `(φ, k₁, k₂, k₃)` are **all
//! continuous**: mode 0 carries the conjugate pair `q₀ = a†+a = φ`,
//! `p₀ = i(a†−a)/2 = p_φ`, and modes 1..=3 are the continuous momentum
//! multipliers `k_r = a†_r+a_r`.  No momentum sum, no spatial lattice and no
//! finite-difference neighbour link occurs anywhere: the gradient is the
//! diagonal continuous multiplier `½|k|²φ²`, which is why λφ⁴ needs no
//! spatial-derivative jet constraint.  The only discrete objects are the finite
//! Mehler–Hermite occupation window and the Krylov rank, both explicit solver
//! parameters (`docs/PHI4_NUMERICAL_PLAN.md`).
//!
//! ## Mehler widths
//!
//! `width_phi` and `width_k` are the physical widths of the continuous Mehler
//! reference Gaussian, i.e. the dimensionless coordinates are
//! `q₀ = φ/width_phi` and `q_r = k_r/width_k`.  The operator is `h_cont`
//! transported into those coordinates:
//!
//! ```text
//! h = ½ p₀²/width_phi²
//!   + ½ width_phi² width_k² Σ_r q_r² q₀²
//!   + (λ/4)(width_phi⁴ q₀⁴ − 2 v² width_phi² q₀² + v⁴),
//! ```
//!
//! which is exactly `h_cont` at unit widths (the fixture point of
//! `docs/fixtures/phi4_momentum_mehler.json`).  The full nonnegative square is
//! retained: the `λ v⁴/4` constant is an explicit term, never dropped.
//!
//! ## Outer enclosure
//!
//! The physical operator is the Quadratic Ordering `H = dΓ(h_cont)`; build it
//! with [`InnerExpr::d_gamma`] on a one-particle outer excitation.  Products
//! must be formed *inside* `h` (`dΓ(B²) ≠ dΓ(B)²` for occupation ≥ 2), and an
//! inner constant `cI` lifts to `c N_out`, so the outer vacuum is annihilated
//! identically.
//!
//! [`frozen_jet_comparison`] is the **comparison-only** decoupled static-jet
//! fibre (`BookProof/Phi4/ChapterScalarJet.lean`): it keeps the same diagonal
//! expectations but omits the continuous momentum–field coupling `½|k|²φ²`.
//! `ScalarGrid::periodic` / `two_site` in [`crate::phi4_scalar`] are the other
//! frozen comparison benchmarks.

use crate::exact_inner::{
    ActionError, ActionLimits, InnerExpr, InnerVector, JointInnerBasis, JointOuterBasis,
};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

/// Mehler mode of the continuous field coordinate `φ`.
pub const PHI_MODE: u32 = 0;
/// Mehler modes of the continuous momentum multipliers `(k₁, k₂, k₃)`.
pub const MOMENTUM_MODES: [u32; 3] = [1, 2, 3];

/// A continuous-momentum λφ⁴ inner Hamiltonian `h_cont`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarMomCont {
    pub lam: f64,
    pub v: f64,
    /// Physical width of the Mehler reference Gaussian in `φ`.
    pub width_phi: f64,
    /// Physical width of the Mehler reference Gaussian in each `k_r`.
    pub width_k: f64,
}

fn scaled(expr: InnerExpr, coefficient: f64) -> InnerExpr {
    expr.scaled(Complex64::new(coefficient, 0.0))
}

fn product(factors: Vec<InnerExpr>) -> InnerExpr {
    InnerExpr::Product(factors)
}

fn sum(terms: Vec<InnerExpr>) -> InnerExpr {
    InnerExpr::Sum(terms)
}

/// `n!` as `f64`; the Mehler Gram weight of `He_n`.
pub fn hermite_gram(n: u32) -> f64 {
    (1..=u64::from(n))
        .map(|k| k as f64)
        .product::<f64>()
        .max(1.0)
}

fn occupation_norm(occupations: &[u32; 4]) -> f64 {
    occupations.iter().map(|&n| hermite_gram(n)).product()
}

impl ScalarMomCont {
    /// The fixture point: `λ = v = 1` at unit Mehler widths.
    pub fn unit() -> Self {
        Self {
            lam: 1.0,
            v: 1.0,
            width_phi: 1.0,
            width_k: 1.0,
        }
    }

    pub fn with_parameters(lam: f64, v: f64, width_phi: f64, width_k: f64) -> Self {
        Self {
            lam,
            v,
            width_phi,
            width_k,
        }
    }

    /// The ordered inner Hamiltonian `h_cont` (see the module documentation).
    ///
    /// The tree is built directly — never through `Expression::expand()` — and
    /// every product is formed inside `h` before the single `dΓ` lift.
    pub fn hamiltonian(&self) -> InnerExpr {
        let wphi = self.width_phi;
        let wphi2 = wphi * wphi;
        let wk2 = self.width_k * self.width_k;
        let q0 = InnerExpr::coordinate(PHI_MODE);

        let p0 = InnerExpr::momentum(PHI_MODE);
        let kinetic = scaled(product(vec![p0.clone(), p0]), 0.5 / wphi2);

        let mut gradient = InnerExpr::Scalar(Complex64::new(0.0, 0.0));
        for r in MOMENTUM_MODES {
            let qr = InnerExpr::coordinate(r);
            gradient = sum(vec![
                gradient,
                scaled(
                    product(vec![qr.clone(), qr, q0.clone(), q0.clone()]),
                    0.5 * wphi2 * wk2,
                ),
            ]);
        }

        let quarter = self.lam / 4.0;
        let wall = sum(vec![
            scaled(
                product(vec![q0.clone(), q0.clone(), q0.clone(), q0.clone()]),
                quarter * wphi2 * wphi2,
            ),
            scaled(
                product(vec![q0.clone(), q0.clone()]),
                -0.5 * self.lam * self.v * self.v * wphi2,
            ),
            InnerExpr::Scalar(Complex64::new(quarter * self.v.powi(4), 0.0)),
        ]);

        sum(vec![kinetic, gradient, wall])
    }

    /// The one-particle inner basis label of a Hermite occupation vector.
    pub fn inner_basis(&self, occupations: [u32; 4]) -> Result<JointInnerBasis, ActionError> {
        JointInnerBasis::new(
            occupations
                .iter()
                .enumerate()
                .map(|(mode, &n)| (mode as u32, n)),
            [],
        )
    }

    /// Apply `h` to a superposition of **unnormalized** product Hermites
    /// `Σ_occ c_occ He_occ(φ) Π He_{n_r}(k_r)`, returning the same unnormalized
    /// Hermite coefficients of the image.
    ///
    /// The inner action itself is the ordered normalized-occupation action;
    /// the `√(n₀! n₁! n₂! n₃!)` factors are the exact Mehler Gram conversion
    /// (`‖He_n‖² = n!`).  Nothing is reordered.
    pub fn hermite_apply(
        &self,
        support: &[([u32; 4], f64)],
    ) -> Result<BTreeMap<[u32; 4], f64>, ActionError> {
        let mut input = InnerVector::new();
        for (occupations, coefficient) in support {
            let basis = self.inner_basis(*occupations)?;
            let amplitude = coefficient * occupation_norm(occupations).sqrt();
            let previous = input.get(&basis).copied().unwrap_or_default();
            input.insert(basis, previous + Complex64::new(amplitude, 0.0));
        }
        let image = self.hamiltonian().apply(&input, ActionLimits::default())?;
        let mut column = BTreeMap::new();
        for (basis, amplitude) in image {
            let occupations = occupations_of(&basis);
            let coefficient = amplitude.re / occupation_norm(&occupations).sqrt();
            if coefficient != 0.0 {
                let previous = column.get(&occupations).copied().unwrap_or(0.0);
                column.insert(occupations, previous + coefficient);
            }
        }
        Ok(column)
    }

    /// `‖Σ c_occ He_occ‖²` in the continuous Mehler measure `e^{-|q|²/2}`.
    pub fn hermite_norm_squared(support: &[([u32; 4], f64)]) -> f64 {
        support
            .iter()
            .map(|(occupations, coefficient)| {
                coefficient * coefficient * occupation_norm(occupations)
            })
            .sum()
    }

    /// The unit-norm inner state of a superposition of unnormalized product
    /// Hermites, in the ordered normalized-occupation representation.
    pub fn hermite_state(&self, support: &[([u32; 4], f64)]) -> InnerVector {
        let norm = Self::hermite_norm_squared(support).sqrt();
        let mut state = InnerVector::new();
        for (occupations, coefficient) in support {
            let basis = self
                .inner_basis(*occupations)
                .expect("occupation vector is a valid inner label");
            let amplitude = coefficient * occupation_norm(occupations).sqrt() / norm;
            let previous = state.get(&basis).copied().unwrap_or_default();
            state.insert(basis, previous + Complex64::new(amplitude, 0.0));
        }
        state
    }

    /// One outer excitation of a unit-norm inner state — an element of the
    /// nested Fock space on which `dΓ(h_cont)` acts.
    pub fn one_particle(&self, support: &[([u32; 4], f64)]) -> crate::exact_inner::NestedVector {
        let mut state = crate::exact_inner::NestedVector::new();
        // Each outer label carries exactly one inner configuration, so the
        // unit-norm inner amplitudes are preserved by the lift.
        for (basis, amplitude) in self.hermite_state(support) {
            let outer = JointOuterBasis::bosonic([(basis, 1)]).expect("bosonic label");
            state.insert(outer, amplitude);
        }
        state
    }

    /// Every single-universe outer label with **even** occupations in all four
    /// modes, `n₀` of the requested `φ` parity, and total Hermite degree
    /// `n₀+n₁+n₂+n₃ ≤ max_degree`.
    ///
    /// `h_cont` is even in every coordinate — quadratic and quartic in `q₀`,
    /// and quadratic in each `q_r` — so each coordinate's occupation parity is
    /// separately conserved.  The even-occupation sector is exactly closed
    /// under the generator for an even-window seed, so no odd-`k` label is
    /// inside the model's reachable spectrum from the Mehler vacuum.
    pub fn window_labels(max_degree: u32) -> BTreeSet<JointOuterBasis> {
        Self::window_labels_of_parity(max_degree, 0)
    }

    /// As [`Self::window_labels`], with the field occupation restricted to the
    /// given `φ` parity (`0` even, `1` odd).  `h_cont` is even in `φ`, so the
    /// two parities decouple exactly.
    pub fn window_labels_of_parity(max_degree: u32, parity: u32) -> BTreeSet<JointOuterBasis> {
        assert!(parity <= 1, "parity is 0 or 1");
        let mut labels = BTreeSet::new();
        let mut n0 = parity;
        while n0 <= max_degree {
            let rest = max_degree - n0;
            for n1 in (0..=rest).step_by(2) {
                for n2 in (0..=(rest - n1)).step_by(2) {
                    for n3 in (0..=(rest - n1 - n2)).step_by(2) {
                        let occupations = [n0, n1, n2, n3];
                        let inner = JointInnerBasis::new(
                            occupations
                                .iter()
                                .enumerate()
                                .map(|(mode, &n)| (mode as u32, n)),
                            [],
                        )
                        .expect("valid occupations");
                        labels
                            .insert(JointOuterBasis::bosonic([(inner, 1)]).expect("bosonic label"));
                    }
                }
            }
            n0 += 2;
        }
        labels
    }
}

/// Occupations `[n₀, n₁, n₂, n₃]` of a joint inner label.
pub fn occupations_of(basis: &JointInnerBasis) -> [u32; 4] {
    let mut occupations = [0u32; 4];
    for (mode, n) in basis.bosons() {
        occupations[*mode as usize] = *n;
    }
    occupations
}

/// Total Hermite degree `n₀+n₁+n₂+n₃` of a joint inner label.
pub fn total_degree(basis: &JointInnerBasis) -> u32 {
    occupations_of(basis).iter().sum()
}

/// The `S₃`-symmetric 11-mode output window of the exact fixture
/// (`docs/fixtures/phi4_momentum_mehler.json`): each entry is a label and the
/// unnormalized product-Hermite occupations it is the sum of.
pub const S3_MODES: [(&str, &[[u32; 4]]); 11] = [
    ("1", &[[0, 0, 0, 0]]),
    ("He2(phi)", &[[2, 0, 0, 0]]),
    ("K2", &[[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]),
    ("He2(phi)K2", &[[2, 2, 0, 0], [2, 0, 2, 0], [2, 0, 0, 2]]),
    ("He4(phi)", &[[4, 0, 0, 0]]),
    ("He6(phi)", &[[6, 0, 0, 0]]),
    ("He4(phi)K2", &[[4, 2, 0, 0], [4, 0, 2, 0], [4, 0, 0, 2]]),
    ("K4", &[[0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]),
    ("K22", &[[0, 2, 2, 0], [0, 2, 0, 2], [0, 0, 2, 2]]),
    ("He2(phi)K4", &[[2, 4, 0, 0], [2, 0, 4, 0], [2, 0, 0, 4]]),
    ("He2(phi)K22", &[[2, 2, 2, 0], [2, 2, 0, 2], [2, 0, 2, 2]]),
];

/// Mehler Gram weights `‖mode_i‖²` of [`S3_MODES`].
pub const S3_GRAMS: [f64; 11] = [
    1.0, 2.0, 6.0, 12.0, 24.0, 720.0, 144.0, 72.0, 12.0, 144.0, 24.0,
];

impl ScalarMomCont {
    /// The unnormalized-Hermite coefficient of output mode `j` in `h·(mode i)`.
    ///
    /// The input mode is a sum of `S₃`-related product Hermites, so its image is
    /// the sum of their images.  The output mode is a single function (not a
    /// sum), so its coefficient is read from one representative label; the
    /// `S₃`-related labels must agree, and that agreement is asserted.
    pub fn s3_entry(&self, input: usize, output: usize) -> f64 {
        let support: Vec<([u32; 4], f64)> = S3_MODES[input]
            .1
            .iter()
            .map(|&occupations| (occupations, 1.0))
            .collect();
        let column = self.hermite_apply(&support).expect("finite Mehler action");
        let coefficients: Vec<f64> = S3_MODES[output]
            .1
            .iter()
            .map(|occupations| column.get(occupations).copied().unwrap_or(0.0))
            .collect();
        let first = coefficients[0];
        debug_assert!(
            coefficients
                .iter()
                .all(|&coefficient| (coefficient - first).abs() < 1e-10),
            "S3 symmetry broken at mode {}: {coefficients:?}",
            S3_MODES[output].0
        );
        first
    }

    /// The full `11 × 3` exact mode table `cols[input][output]`.
    pub fn s3_columns(&self) -> [[f64; 11]; 3] {
        let mut columns = [[0.0_f64; 11]; 3];
        for (input, column) in columns.iter_mut().enumerate() {
            for (output, entry) in column.iter_mut().enumerate() {
                *entry = self.s3_entry(input, output);
            }
        }
        columns
    }

    /// `⟨h⟩`, `⟨h²⟩ − ⟨h⟩²` and the squared norm outside the three-mode window
    /// `{1, He₂(φ), K₂}` for the seed `1`, in the unnormalized Hermite table.
    pub fn seed_moments(&self) -> (f64, f64, f64) {
        let columns = self.s3_columns();
        let mean = columns[0][0];
        let variance = columns[0]
            .iter()
            .zip(S3_GRAMS)
            .skip(1)
            .map(|(coefficient, gram)| coefficient * coefficient * gram)
            .sum();
        let outside = columns[0]
            .iter()
            .zip(S3_GRAMS)
            .skip(3)
            .map(|(coefficient, gram)| coefficient * coefficient * gram)
            .sum();
        (mean, variance, outside)
    }

    /// The `3 × 3` window matrix in unnormalized Hermite coordinates.
    pub fn window_matrix(&self) -> [[f64; 3]; 3] {
        let columns = self.s3_columns();
        let mut matrix = [[0.0_f64; 3]; 3];
        for (input, column) in columns.iter().enumerate() {
            for output in 0..3 {
                matrix[output][input] = column[output];
            }
        }
        matrix
    }
}

/// COMPARISON ONLY — frozen decoupled static-jet fibre
/// (`BookProof/Phi4/ChapterScalarJet.lean`): `½ p_φ² + ½ Σ_r k_r² + U(φ)`.
///
/// It has the same diagonal expectations as [`ScalarMomCont`] but no
/// continuous momentum–field coupling `½|k|²φ²`, so it is a reproducible
/// contrast benchmark, never the primary model.
pub fn frozen_jet_comparison() -> InnerExpr {
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    // Remove ½ Σ_r q_r² q₀² and restore the free ½ Σ_r k_r² = ½ Σ_r (He₂(k_r)+1).
    let mut terms = match h {
        InnerExpr::Sum(terms) => terms,
        _ => unreachable!("h_cont is built as a sum"),
    };
    terms.remove(1); // the continuous gradient block
    let q0 = InnerExpr::coordinate(PHI_MODE);
    let mut free = InnerExpr::Scalar(Complex64::new(0.0, 0.0));
    for r in MOMENTUM_MODES {
        let qr = InnerExpr::coordinate(r);
        free = sum(vec![free, scaled(product(vec![qr.clone(), qr]), 0.5)]);
        free = sum(vec![free, InnerExpr::Scalar(Complex64::new(0.5, 0.0))]);
    }
    let _ = q0;
    sum(vec![InnerExpr::Sum(terms), free])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `11 × 3` table of `docs/phi4_momentum_nested.cdb`
    /// (CHECK `eleven-mode-s3-columns`).
    const EXPECTED_COLUMNS: [[f64; 11]; 3] = [
        [
            17.0 / 8.0,
            19.0 / 8.0,
            1.0 / 2.0,
            1.0 / 2.0,
            1.0 / 4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            19.0 / 4.0,
            125.0 / 8.0,
            1.0,
            5.0 / 2.0,
            35.0 / 8.0,
            1.0 / 4.0,
            1.0 / 2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            3.0,
            3.0,
            33.0 / 8.0,
            35.0 / 8.0,
            0.0,
            0.0,
            1.0 / 4.0,
            1.0 / 2.0,
            1.0,
            1.0 / 2.0,
            1.0,
        ],
    ];

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-12
    }

    #[test]
    fn eleven_mode_s3_columns_match_the_exact_fixture() {
        let model = ScalarMomCont::unit();
        let columns = model.s3_columns();
        for input in 0..3 {
            for output in 0..11 {
                assert!(
                    close(columns[input][output], EXPECTED_COLUMNS[input][output]),
                    "column {} mode {} ({}): got {}, expected {}",
                    input,
                    output,
                    S3_MODES[output].0,
                    columns[input][output],
                    EXPECTED_COLUMNS[input][output]
                );
            }
        }
    }

    #[test]
    fn seed_moments_match_the_fixture() {
        let (mean, variance, outside) = ScalarMomCont::unit().seed_moments();
        assert!(close(mean, 17.0 / 8.0), "mean {mean}");
        assert!(close(variance, 553.0 / 32.0), "variance {variance}");
        assert!(close(outside, 9.0 / 2.0), "outside {outside}");
    }

    #[test]
    fn window_matrix_is_hermitian_in_the_physical_gram() {
        let window = ScalarMomCont::unit().window_matrix();
        let gram = [1.0, 2.0, 6.0];
        for i in 0..3 {
            for j in 0..3 {
                let left = gram[i] * window[i][j];
                let right = window[j][i] * gram[j];
                assert!(close(left, right), "({i},{j}) {left} vs {right}");
            }
        }
        assert!(close(window[0][0], 17.0 / 8.0));
        assert!(close(window[1][0], 19.0 / 8.0));
        assert!(close(window[0][1], 19.0 / 4.0));
        assert!(close(window[2][0], 1.0 / 2.0));
        assert!(close(window[0][2], 3.0));
    }

    #[test]
    fn odd_seed_column_matches_the_fixture() {
        let model = ScalarMomCont::unit();
        let column = model
            .hermite_apply(&[([1, 0, 0, 0], 1.0)])
            .expect("finite action");
        // 59/8 He1(φ) + 27/8 He3(φ) + 1/4 He5(φ)
        //   + 3/2 He1(φ)K2 + 1/2 He3(φ)K2.
        assert!(close(column[&[1, 0, 0, 0]], 59.0 / 8.0));
        assert!(close(column[&[3, 0, 0, 0]], 27.0 / 8.0));
        assert!(close(column[&[5, 0, 0, 0]], 1.0 / 4.0));
        for r in MOMENTUM_MODES {
            let mut occupations = [1, 0, 0, 0];
            occupations[r as usize] = 2;
            assert!(close(column[&occupations], 3.0 / 2.0), "{occupations:?}");
            occupations[0] = 3;
            assert!(close(column[&occupations], 1.0 / 2.0), "{occupations:?}");
        }
        // φ ↦ −φ parity: no even-φ output survives an odd seed.
        for (occupations, coefficient) in &column {
            assert!(
                occupations[0] % 2 == 1 || *coefficient == 0.0,
                "even output {occupations:?} = {coefficient}"
            );
        }
    }

    #[test]
    fn frozen_jet_comparison_drops_the_momentum_coupling() {
        let model = ScalarMomCont::unit();
        let columns = model.s3_columns();
        let variance: f64 = columns[0]
            .iter()
            .zip(S3_GRAMS)
            .skip(1)
            .map(|(c, g)| c * c * g)
            .sum();
        // The jet comparison's seed image: same diagonal, no He₂(φ)K₂ coupling.
        let jet = frozen_jet_comparison();
        let input = model.hermite_state(&[([0, 0, 0, 0], 1.0)]);
        let image = jet.apply(&input, ActionLimits::default()).unwrap();
        let mut jet_variance = 0.0;
        let mut coupling = 0.0;
        for (basis, amplitude) in &image {
            let occupations = occupations_of(basis);
            let norm = occupation_norm(&occupations).sqrt();
            let coefficient = amplitude.re / norm;
            if occupations != [0, 0, 0, 0] {
                jet_variance += coefficient * coefficient * norm * norm;
            }
            if occupations == [2, 2, 0, 0]
                || occupations == [2, 0, 2, 0]
                || occupations == [2, 0, 0, 2]
            {
                coupling += coefficient;
            }
        }
        assert!(
            close(variance - jet_variance, 51.0 / 4.0),
            "{variance} {jet_variance}"
        );
        // The seed column carries the He₂(φ)K₂ coupling 1/2; the jet fibre has none.
        assert!(close(columns[0][3], 1.0 / 2.0));
        assert!(coupling == 0.0, "jet coupling {coupling}");
    }

    #[test]
    fn outer_quadratic_ordering_enclosure() {
        let model = ScalarMomCont::unit();
        let inner = model.inner_basis([0, 0, 0, 0]).unwrap();
        let empty = crate::exact_inner::NestedVector::new();
        assert!(
            model
                .hamiltonian()
                .d_gamma(&empty, ActionLimits::default())
                .unwrap()
                .is_empty(),
            "the outer vacuum is annihilated"
        );

        // Inner shift cI lifts to c N_out: 0, c, 2c on n = 0, 1, 2 excitations.
        let shift = InnerExpr::Scalar(Complex64::new(0.7, 0.0));
        for n in 0..3 {
            let mut outer = crate::exact_inner::NestedVector::new();
            if n >= 1 {
                outer.insert(
                    JointOuterBasis::bosonic([(inner.clone(), 1)]).unwrap(),
                    Complex64::new(1.0, 0.0),
                );
            }
            if n == 2 {
                outer.insert(
                    JointOuterBasis::bosonic([(inner.clone(), 2)]).unwrap(),
                    Complex64::new(1.0, 0.0),
                );
            }
            let image = shift.d_gamma(&outer, ActionLimits::default()).unwrap();
            for (basis, amplitude) in &image {
                let expected = 0.7 * basis.particle_number() as f64;
                assert!(close(amplitude.re, expected), "n={n} {basis:?}");
            }
            if n == 0 {
                assert!(image.is_empty());
            }
        }

        // dΓ(B)² ≠ dΓ(B²) at n = 2: the inner product must be formed first.
        let b = InnerExpr::coordinate(PHI_MODE);
        let mut two = crate::exact_inner::NestedVector::new();
        two.insert(
            JointOuterBasis::bosonic([(inner.clone(), 2)]).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        let once = b.d_gamma(&two, ActionLimits::default()).unwrap();
        let squared_lift = b.d_gamma(&once, ActionLimits::default()).unwrap();
        let lifted_square = InnerExpr::Product(vec![b.clone(), b.clone()])
            .d_gamma(&two, ActionLimits::default())
            .unwrap();
        let difference: f64 = squared_lift
            .iter()
            .map(|(basis, amplitude)| {
                let other = lifted_square.get(basis).copied().unwrap_or_default();
                (amplitude - other).norm()
            })
            .sum();
        assert!(
            difference > 1e-9,
            "cross-particle defect vanished: {difference}"
        );
    }

    #[test]
    fn widths_are_the_physical_mehler_scales() {
        // Scaling the reference Gaussian must leave the unit-width model intact
        // and change the physical gradient coupling when width_k ≠ 1.
        let unit = ScalarMomCont::unit();
        let wide = ScalarMomCont::with_parameters(1.0, 1.0, 1.0, 1.5);
        let (mean_unit, _, _) = unit.seed_moments();
        let (mean_wide, _, _) = wide.seed_moments();
        assert!(close(mean_unit, 17.0 / 8.0));
        assert!(
            (mean_wide - mean_unit).abs() > 1e-3,
            "width_k must change the continuous gradient energy: {mean_wide}"
        );
    }
}
