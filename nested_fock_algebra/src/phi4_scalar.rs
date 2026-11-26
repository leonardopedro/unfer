//! COMPARISON ONLY — frozen periodic lattice finite-difference benchmark (NOT the
//! book.tex continuous-momentum model; see phi4_momentum.rs).
//!
//! Periodic lattice comparison for the scalar λφ⁴ model and its two-site fixture.
//! The spatial sites and links here belong to this comparison. The project's
//! continuous-coordinate scalar operator uses independent field and jet variables.
//!
//! The site Hamiltonian is
//! `h = ½ Σ p_x² + (1/(2a²)) Σ_links (q_next − q_x)²
//!      + Σ_x (λ/(4a³))(q_x² − a³v²)²`,
//! with `q = a† + a` and `p = i(a† − a)/2`.  The two-site `2×1×1` grid has one
//! positive direction, so its two wrapped links give the factor-two link of the
//! exact fixture.  The rotated builder rewrites the same operator in the uniform
//! and staggered modes `u = (q₀+q₁)/√2`, `w = (q₀−q₁)/√2`; because the Gaussian
//! is rotation invariant, mode and site vacua coincide and the mode occupation
//! basis is the fixture's unnormalized product-Hermite basis up to `√(m! n!)`.
//!
//! Coefficients are `f64`; the comparison tolerance is far above their rounding
//! error and far below any nonzero fixture coefficient.

use crate::exact_inner::{ActionLimits, InnerExpr, InnerVector, JointInnerBasis};
use num_complex::Complex64;

/// A scalar grid with one bosonic inner mode per site and oriented active links.
#[derive(Clone, Debug)]
pub struct ScalarGrid {
    pub a: f64,
    pub lam: f64,
    pub v: f64,
    pub sites: Vec<u32>,
    /// `(source, target)` for every oriented positive link counted by the model.
    pub links: Vec<(u32, u32)>,
}

impl ScalarGrid {
    /// A rectangular periodic grid with `nx * ny * nz` sites and one positive
    /// link per site and direction.
    pub fn periodic(nx: usize, ny: usize, nz: usize, a: f64, lam: f64, v: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "periodic sides must be nonempty"
        );
        let index = |x: usize, y: usize, z: usize| -> u32 { (x + nx * (y + ny * z)) as u32 };
        let n = nx * ny * nz;
        let mut links = Vec::with_capacity(3 * n);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let s = index(x, y, z);
                    links.push((s, index((x + 1) % nx, y, z)));
                    links.push((s, index(x, (y + 1) % ny, z)));
                    links.push((s, index(x, y, (z + 1) % nz)));
                }
            }
        }
        Self {
            a,
            lam,
            v,
            sites: (0..n as u32).collect(),
            links,
        }
    }

    /// The `2×1×1` periodic grid: two wrapped links along the active axis.
    pub fn two_site(a: f64, lam: f64, v: f64) -> Self {
        Self::periodic(2, 1, 1, a, lam, v)
    }

    fn coordinate(&self, site: u32) -> InnerExpr {
        InnerExpr::coordinate(site)
    }

    fn momentum(&self, site: u32) -> InnerExpr {
        InnerExpr::momentum(site)
    }

    fn scaled(expr: InnerExpr, c: f64) -> InnerExpr {
        expr.scaled(Complex64::new(c, 0.0))
    }

    fn sum(parts: Vec<InnerExpr>) -> InnerExpr {
        InnerExpr::Sum(parts)
    }

    fn product(factors: Vec<InnerExpr>) -> InnerExpr {
        InnerExpr::Product(factors)
    }

    fn square(expr: InnerExpr) -> InnerExpr {
        Self::product(vec![expr.clone(), expr])
    }

    fn difference(left: InnerExpr, right: InnerExpr) -> InnerExpr {
        Self::sum(vec![left, Self::scaled(right, -1.0)])
    }

    /// The site Hamiltonian in the original site coordinates.
    pub fn site_hamiltonian(&self) -> InnerExpr {
        let kinetic = Self::sum(
            self.sites
                .iter()
                .map(|&s| Self::square(self.momentum(s)))
                .collect(),
        );
        let wall = self.lam / (2.0 * self.a.powi(3));
        let offset = self.a.powi(3) * self.v * self.v;
        let mut forms = Vec::new();
        for &s in &self.sites {
            let q = self.coordinate(s);
            let shifted = Self::difference(
                Self::square(q),
                InnerExpr::Scalar(Complex64::new(offset, 0.0)),
            );
            forms.push(Self::scaled(shifted, wall.sqrt()));
        }
        for &(s, t) in &self.links {
            forms.push(Self::scaled(
                Self::difference(self.coordinate(t), self.coordinate(s)),
                1.0 / self.a,
            ));
        }
        let potential = Self::sum(forms.iter().cloned().map(Self::square).collect());
        Self::scaled(Self::sum(vec![kinetic, potential]), 0.5)
    }

    /// The same Hamiltonian in the two-site uniform/staggered modes.
    pub fn two_site_mode_hamiltonian(&self) -> InnerExpr {
        debug_assert_eq!(self.sites, vec![0, 1], "mode builder is the 2×1×1 grid");
        // After the orthogonal change these are the two independent mode labels;
        // the site coordinates are not used as basis labels here.
        let u = self.coordinate(0);
        let w = self.coordinate(1);
        let pu = self.momentum(0);
        let pw = self.momentum(1);
        let add = |x: InnerExpr, y: InnerExpr| Self::sum(vec![x, y]);
        let kinetic = Self::sum(vec![Self::square(pu), Self::square(pw)]);
        let wall_scale = self.lam / (4.0 * self.a.powi(3));
        let offset = 2.0 * self.a.powi(3) * self.v * self.v;
        let radial = Self::difference(
            add(Self::square(u.clone()), Self::square(w.clone())),
            InnerExpr::Scalar(Complex64::new(offset, 0.0)),
        );
        let wall = Self::scaled(Self::square(radial), wall_scale);
        let mixed = Self::scaled(
            Self::square(Self::product(vec![u, w.clone()])),
            self.lam / self.a.powi(3),
        );
        let link = Self::scaled(Self::square(w), 4.0 / (self.a * self.a));
        let potential = Self::sum(vec![wall, mixed, link]);
        Self::scaled(Self::sum(vec![kinetic, potential]), 0.5)
    }

    /// Coefficients of `h (He_i(u) He_j(w))` in the unnormalized mode basis.
    pub fn two_site_column(&self, i: u32, j: u32) -> Vec<((u32, u32), f64)> {
        let h = self.two_site_mode_hamiltonian();
        let key = JointInnerBasis::new([(0, i), (1, j)], []).expect("distinct bosonic modes");
        let input = InnerVector::from([(key, Complex64::new(1.0, 0.0))]);
        let output = h
            .apply(&input, ActionLimits::default())
            .expect("finite action");
        let input_norm = (factorial(i) * factorial(j)) as f64;
        let mut column = Vec::new();
        for (basis, amplitude) in output {
            let m = *basis.bosons().get(&0).unwrap_or(&0);
            let n = *basis.bosons().get(&1).unwrap_or(&0);
            let output_norm = (factorial(m) * factorial(n)) as f64;
            // InnerExpr uses normalized occupations; the fixture uses unnormalized Hermites.
            let coefficient = amplitude.re * (input_norm / output_norm).sqrt();
            if coefficient.abs() > 1e-10 {
                column.push(((m, n), coefficient));
            }
        }
        column.sort_by_key(|&(mode, _)| mode);
        column
    }
}

fn factorial(n: u32) -> u64 {
    (1..=u64::from(n)).product::<u64>().max(1)
}

/// Real Fourier modes of a one-dimensional periodic cycle. The returned rows are
/// orthonormal under the Euclidean inner product; `n = 2` is the uniform/staggered
/// rotation used by the two-site builder.
pub fn periodic_fourier_1d(n: usize) -> Vec<Vec<f64>> {
    assert!(n > 0, "periodic side must be nonempty");
    let norm = 1.0 / (n as f64).sqrt();
    let mode_norm = (2.0 / n as f64).sqrt();
    let mut rows = vec![vec![norm; n]];
    for k in 1..=((n - 1) / 2) {
        let cos: Vec<f64> = (0..n)
            .map(|x| (std::f64::consts::TAU * (k * x) as f64 / n as f64).cos() * mode_norm)
            .collect();
        let sin: Vec<f64> = (0..n)
            .map(|x| (std::f64::consts::TAU * (k * x) as f64 / n as f64).sin() * mode_norm)
            .collect();
        rows.push(cos);
        rows.push(sin);
    }
    if n.is_multiple_of(2) {
        rows.push(
            (0..n)
                .map(|x| if x % 2 == 0 { norm } else { -norm })
                .collect(),
        );
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::ScalarGrid;
    use crate::exact_inner::{ActionLimits, InnerVector, JointInnerBasis};
    use num_complex::Complex64;

    type Column = &'static [((u32, u32), f64)];

    const COLUMN_00: Column = &[
        ((0, 0), 13.0 / 4.0),
        ((0, 2), 23.0 / 8.0),
        ((0, 4), 1.0 / 8.0),
        ((2, 0), 7.0 / 8.0),
        ((2, 2), 3.0 / 4.0),
        ((4, 0), 1.0 / 8.0),
    ];
    const COLUMN_20: Column = &[
        ((0, 0), 7.0 / 4.0),
        ((0, 2), 3.0 / 2.0),
        ((2, 0), 37.0 / 4.0),
        ((2, 2), 47.0 / 8.0),
        ((2, 4), 1.0 / 8.0),
        ((4, 0), 15.0 / 8.0),
        ((4, 2), 3.0 / 4.0),
        ((6, 0), 1.0 / 8.0),
    ];
    const COLUMN_02: Column = &[
        ((0, 0), 23.0 / 4.0),
        ((0, 2), 69.0 / 4.0),
        ((0, 4), 31.0 / 8.0),
        ((0, 6), 1.0 / 8.0),
        ((2, 0), 3.0 / 2.0),
        ((2, 2), 31.0 / 8.0),
        ((2, 4), 3.0 / 4.0),
        ((4, 2), 1.0 / 8.0),
    ];

    fn assert_column(actual: Vec<((u32, u32), f64)>, expected: Column, label: &str) {
        for &(mode, coefficient) in expected {
            let got = actual
                .iter()
                .find(|&&(m, _)| m == mode)
                .map(|&(_, c)| c)
                .unwrap_or(0.0);
            assert!(
                (got - coefficient).abs() < 1e-8,
                "{label} mode {mode:?}: got {got}, expected {coefficient}"
            );
        }
        for (mode, coefficient) in actual {
            assert!(
                expected
                    .iter()
                    .any(|&(m, c)| m == mode && (c - coefficient).abs() < 1e-8),
                "{label} unexpected coefficient at {mode:?}: {coefficient}"
            );
        }
    }

    #[test]
    fn two_site_columns_match_fixture() {
        let model = ScalarGrid::two_site(1.0, 1.0, 1.0);
        assert_column(model.two_site_column(0, 0), COLUMN_00, "column (0,0)");
        assert_column(model.two_site_column(2, 0), COLUMN_20, "column (2,0)");
        assert_column(model.two_site_column(0, 2), COLUMN_02, "column (0,2)");
    }

    #[test]
    fn periodic_fourier_is_orthogonal() {
        for n in [2usize, 3, 4, 7, 8] {
            let o = super::periodic_fourier_1d(n);
            assert_eq!(o.len(), n);
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = (0..n).map(|k| o[i][k] * o[j][k]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-10, "n={n} ({i},{j}) dot={dot}");
                }
            }
        }
    }

    #[test]
    fn two_site_return_lies_in_analytic_interval() {
        let h = [
            [13.0 / 4.0, 7.0 / 4.0, 23.0 / 4.0],
            [7.0 / 8.0, 37.0 / 4.0, 3.0 / 2.0],
            [23.0 / 8.0, 3.0 / 2.0, 69.0 / 4.0],
        ];
        let t = 0.01;
        let a: Vec<Vec<Complex64>> = h
            .iter()
            .map(|row| row.iter().map(|&x| Complex64::new(0.0, -t * x)).collect())
            .collect();
        let id: Vec<Vec<Complex64>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        if i == j {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let mul = |x: &[Vec<Complex64>], y: &[Vec<Complex64>]| -> Vec<Vec<Complex64>> {
            let n = x.len();
            let m = y[0].len();
            (0..n)
                .map(|i| {
                    (0..m)
                        .map(|j| (0..y.len()).map(|k| x[i][k] * y[k][j]).sum())
                        .collect()
                })
                .collect()
        };
        let mut term = id.clone();
        let mut sum = id;
        for k in 1..60 {
            term = mul(&term, &a);
            for row in &mut term {
                for x in row {
                    *x /= k as f64;
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    sum[i][j] += term[i][j];
                }
            }
        }
        let return_probability = sum[0][0].norm_sqr();
        let lower = 159663.0 / 160000.0;
        let upper = 39991.0 / 40000.0;
        assert!(
            (lower..=upper).contains(&return_probability),
            "return {return_probability} outside [{lower}, {upper}]"
        );

        // Full ten-row image of the same window state: the tail is the finite-window error.
        let columns = [
            [13.0 / 4.0, 7.0 / 4.0, 23.0 / 4.0],
            [7.0 / 8.0, 37.0 / 4.0, 3.0 / 2.0],
            [23.0 / 8.0, 3.0 / 2.0, 69.0 / 4.0],
            [1.0 / 8.0, 0.0, 31.0 / 8.0],
            [0.0, 0.0, 1.0 / 8.0],
            [3.0 / 4.0, 47.0 / 8.0, 31.0 / 8.0],
            [0.0, 1.0 / 8.0, 3.0 / 4.0],
            [1.0 / 8.0, 15.0 / 8.0, 0.0],
            [0.0, 3.0 / 4.0, 1.0 / 8.0],
            [0.0, 1.0 / 8.0, 0.0],
        ];
        let v = [sum[0][0], sum[1][0], sum[2][0]];
        // Exact Mehler-Hermite Gram weights of the seven discarded rows.
        let gram = [24.0, 720.0, 4.0, 48.0, 24.0, 48.0, 720.0];
        let tail: f64 = columns[3..]
            .iter()
            .zip(gram)
            .map(|(row, w)| w * (0..3).map(|j| row[j] * v[j]).sum::<Complex64>().norm_sqr())
            .sum();
        assert!(
            tail.sqrt() > 0.0 && tail.sqrt() < 2.0,
            "weighted tail {}",
            tail.sqrt()
        );
    }

    #[test]
    fn periodic_grid_topology() {
        let grid = ScalarGrid::periodic(3, 2, 1, 1.0, 1.0, 1.0);
        assert_eq!(grid.sites.len(), 6);
        assert_eq!(grid.links.len(), 3 * 6);
    }

    #[test]
    fn vacuum_energy_matches_fixture() {
        let model = ScalarGrid::two_site(1.0, 1.0, 1.0);
        let vacuum = InnerVector::from([(JointInnerBasis::vacuum(), Complex64::new(1.0, 0.0))]);
        let site = model
            .site_hamiltonian()
            .apply(&vacuum, ActionLimits::default())
            .unwrap();
        let mode = model
            .two_site_mode_hamiltonian()
            .apply(&vacuum, ActionLimits::default())
            .unwrap();
        // Both frames give the fixture's Gaussian diagonal 13/4.
        assert!((site[&JointInnerBasis::vacuum()].re - 13.0 / 4.0).abs() < 1e-10);
        assert!((mode[&JointInnerBasis::vacuum()].re - 13.0 / 4.0).abs() < 1e-10);
    }
}
