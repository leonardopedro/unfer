//! The analytic heads: a ridge solve against the target construction, plus the
//! NCM readout.
//!
//! Both methods share this skeleton and differ only in *what they regress onto*
//! (`targets::spread_means` vs `targets::etf_procrustes`) and *how they read out*
//! (plain NCM vs cosine NCM). That is the plan's point in I.4.3: the
//! unification must pick per surface, not average the two.
//!
//! # The solve
//!
//! With `A = Zᵀ Z + λI` (a `D × D` Gram) and `B = Σ_c n_c · r_c · t_cᵀ`, the
//! analytic solution is `θ = A⁻¹ B`, obtained here by Cholesky. Cholesky rather
//! than a general inverse for three reasons: it is ~2× cheaper, it fails loudly
//! on a non-PD matrix instead of returning garbage, and it is what the papers'
//! `λ > 0` ridge assumes.
//!
//! # Batching
//!
//! `B` is accumulated incrementally, entry by entry, rather than by holding the
//! whole `Z`. For a stream that never ends that is the difference between O(1)
//! and O(n·D²) memory, and it is what makes "continual" mean anything here.

use nalgebra::{DMatrix, DVector};

use crate::stats::StreamingStats;
use crate::targets::{self, TargetKind};

/// Numerical failure modes of an analytic solve.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SolveError {
    #[error("ridge system is not positive definite at lambda={0}; increase the ridge")]
    NotPositiveDefinite(f64),
    #[error("class id {0} has no samples, so it cannot be regressed onto")]
    EmptyClass(usize),
    #[error("at least two classes are needed; got {0}")]
    TooFewClasses(usize),
}

/// Accumulated `A` and `B` for one head, plus the solve.
#[derive(Debug, Clone)]
pub struct RidgeHead {
    /// `D × D` Gram of the random features, before the ridge is added.
    gram: DMatrix<f64>,
    /// RP-space class sums, one `D`-vector per class, plus the counts that
    /// weight them. Stored as a `Vec` rather than a growing `DMatrix` because
    /// the class axis is appended one at a time and nalgebra has no
    /// `push_column` for an owned matrix.
    rp_sums: Vec<DVector<f64>>,
    rp_counts: Vec<u64>,
    big_d: usize,
    lambda: f64,
}

impl RidgeHead {
    /// A head over a `D`-dimensional RP space.
    pub fn new(big_d: usize, lambda: f64) -> Self {
        RidgeHead {
            gram: DMatrix::zeros(big_d, big_d),
            rp_sums: Vec::new(),
            rp_counts: Vec::new(),
            big_d,
            lambda,
        }
    }

    /// The ridge penalty.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Projected dimension.
    pub fn big_d(&self) -> usize {
        self.big_d
    }

    /// Register a class, growing the per-class accumulators.
    pub fn ensure_class(&mut self, id: usize) {
        while self.rp_counts.len() <= id {
            self.rp_counts.push(0);
            self.rp_sums.push(DVector::zeros(self.big_d));
        }
    }

    /// Per-class counts, in id order.
    pub fn counts(&self) -> &[u64] {
        &self.rp_counts
    }

    /// The accumulated `Σ z` for class `id`.
    pub fn rp_sum(&self, id: usize) -> &DVector<f64> {
        &self.rp_sums[id]
    }

    /// Total samples accumulated.
    pub fn total(&self) -> u64 {
        self.rp_counts.iter().sum()
    }

    /// Shared covariance of the RP features, or `None` below two samples.
    ///
    /// This is the whitening matrix for the *head*, and it lives in the same
    /// space as the RP class means it will whiten. Passing the input-space
    /// covariance here instead is a real bug: two different geometries would be
    /// mixed, and `targets::spread_means` now refuses a dimension mismatch rather
    /// than producing a plausible-looking wrong answer.
    ///
    /// Uses the same `cov = (Σzzᵀ − n μμᵀ)/(n − 1)` identity as the input-space
    /// store, which needs only the Gram already being accumulated.
    pub fn rp_covariance(&self) -> Option<DMatrix<f64>> {
        let n = self.total();
        if n < 2 {
            return None;
        }
        let mut mean = DVector::<f64>::zeros(self.big_d);
        for (c, cnt) in self.rp_counts.iter().enumerate() {
            if *cnt == 0 {
                continue;
            }
            for k in 0..self.big_d {
                mean[k] += self.rp_sums[c][k] / *cnt as f64;
            }
        }
        let nf = n as f64;
        let mut cov = self.gram.clone();
        for a in 0..self.big_d {
            for b in 0..self.big_d {
                cov[(a, b)] -= nf * mean[a] * mean[b];
            }
        }
        cov /= nf - 1.0;
        Some(cov)
    }

    /// Accumulate one sample: `A += z zᵀ`, `r_c += z`, `n_c += 1`.
    pub fn push(&mut self, id: usize, z: &DVector<f64>) {
        assert_eq!(
            z.len(),
            self.big_d,
            "RP feature dimension changed: head is {}, got {}",
            self.big_d,
            z.len()
        );
        assert!(
            id < self.rp_counts.len(),
            "class id {id} was never registered with ensure_class"
        );
        for a in 0..self.big_d {
            for b in 0..self.big_d {
                self.gram[(a, b)] += z[a] * z[b];
            }
            self.rp_sums[id][a] += z[a];
        }
        self.rp_counts[id] += 1;
    }

    /// RP-space class means as a `k × C` matrix (classes are rows).
    fn rp_class_means(&self) -> Result<DMatrix<f64>, SolveError> {
        let cols: Vec<usize> = (0..self.rp_counts.len())
            .filter(|&c| self.rp_counts[c] > 0)
            .collect();
        if cols.len() < 2 {
            return Err(SolveError::TooFewClasses(cols.len()));
        }
        let mut m = DMatrix::zeros(cols.len(), self.big_d);
        for (row, &c) in cols.iter().enumerate() {
            let n = self.rp_counts[c] as f64;
            for k in 0..self.big_d {
                m[(row, k)] = self.rp_sums[c][k] / n;
            }
        }
        Ok(m)
    }

    /// Solve `θ = (A + λI)⁻¹ B` for the given target construction.
    ///
    /// Returns a `D × k` matrix. `targets` is `k × C` with classes as rows, so
    /// `B = Σ_c n_c · r_c · t_cᵀ` becomes `means_rowspaceᵀ · diag(n) · targets`.
    pub fn solve(
        &self,
        kind: TargetKind,
        alpha: f64,
        rp_cov: Option<&DMatrix<f64>>,
    ) -> Result<DMatrix<f64>, SolveError> {
        let means = self.rp_class_means()?;
        let c = means.nrows();
        let counts: Vec<f64> = (0..self.rp_counts.len())
            .filter(|&i| self.rp_counts[i] > 0)
            .map(|i| self.rp_counts[i] as f64)
            .collect();

        let targets = match kind {
            // Spread means operate on the *input-space* class means; at the head
            // level the same construction is applied to the RP-space means so
            // the two modes share one code path and one set of tests. The
            // covariance passed in is the RP one, in the same space.
            TargetKind::SpreadMeans => {
                let as_cols = means.transpose();
                targets::spread_means(&as_cols, rp_cov, alpha).transpose()
            }
            TargetKind::Etf => targets::etf_procrustes(&means, c),
        };

        // B = Mᵀ diag(n) T, where M is `C × k` of RP class means and T is
        // `C × t` of targets. The target width is whatever the construction
        // returned — the two modes produce different widths for the same class
        // set, and forcing them to agree here is exactly the averaging error the
        // plan warns against.
        let t_cols = targets.ncols();
        let mut big_b = DMatrix::zeros(self.big_d, t_cols);
        for (ci, &n) in counts.iter().enumerate() {
            for a in 0..self.big_d {
                let r = means[(ci, a)] * n;
                for tcol in 0..t_cols {
                    big_b[(a, tcol)] += r * targets[(ci, tcol)];
                }
            }
        }

        // A + λI, Cholesky.
        let mut a = self.gram.clone();
        for i in 0..self.big_d {
            a[(i, i)] += self.lambda;
        }
        let chol = match a.cholesky() {
            Some(c) => c,
            None => return Err(SolveError::NotPositiveDefinite(self.lambda)),
        };
        Ok(chol.solve(&big_b))
    }
}

/// A nearest-class-mean readout over cosine similarity.
///
/// Cosine rather than Euclidean because the methods' prototypes are defined up to
/// scale (the ETF constraint fixes inner products, not lengths) and because the
/// open-world score in `ananc` is *defined* as the maximum cosine.
#[derive(Debug, Clone)]
pub struct NcmHead {
    /// `k × C`, classes as rows.
    prototypes: DMatrix<f64>,
    labels: Vec<String>,
}

impl NcmHead {
    /// Build from prototypes and their class labels.
    pub fn new(prototypes: DMatrix<f64>, labels: Vec<String>) -> Self {
        NcmHead { prototypes, labels }
    }

    /// Cosine similarity of `x` against every prototype, as a `C`-vector.
    pub fn similarities(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut out = DVector::zeros(self.prototypes.nrows());
        let nx = x.norm();
        for c in 0..self.prototypes.nrows() {
            let mut dot = 0.0;
            let mut np = 0.0;
            for k in 0..self.prototypes.ncols() {
                dot += self.prototypes[(c, k)] * x[k];
                np += self.prototypes[(c, k)] * self.prototypes[(c, k)];
            }
            let denom = np.sqrt() * nx;
            out[c] = if denom > 0.0 { dot / denom } else { 0.0 };
        }
        out
    }

    /// Argmax over cosine similarity.
    pub fn predict(&self, x: &DVector<f64>) -> Option<(&str, f64)> {
        let sims = self.similarities(x);
        if sims.is_empty() {
            return None;
        }
        let mut best = 0usize;
        for c in 1..sims.len() {
            if sims[c] > sims[best] {
                best = c;
            }
        }
        self.labels.get(best).map(|l| (l.as_str(), sims[best]))
    }

    /// The maximum cosine similarity — AnaNC's in-distribution ("ID") score.
    ///
    /// This is the actual OOD mechanism of the source method. The energy-based
    /// score in the original sketch (audit defect A5) is *not* used, and must
    /// not be reintroduced without its own validation.
    pub fn id_score(&self, x: &DVector<f64>) -> f64 {
        let sims = self.similarities(x);
        if sims.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut best = f64::NEG_INFINITY;
        for c in 0..sims.len() {
            if sims[c] > best {
                best = sims[c];
            }
        }
        best
    }

    /// Class labels in prototype order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Prototypes as `k × C`.
    pub fn prototypes(&self) -> &DMatrix<f64> {
        &self.prototypes
    }

    /// Number of classes.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether there are no prototypes.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Rebuild from fresh prototypes, keeping the label set.
    pub fn with_prototypes(&self, prototypes: DMatrix<f64>) -> Self {
        NcmHead {
            prototypes,
            labels: self.labels.clone(),
        }
    }
}

/// Re-export so callers do not have to reach into `stats` for the common case.
pub fn class_means_matrix(stats: &StreamingStats) -> DMatrix<f64> {
    stats.class_means().1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_classes_is_an_error_not_a_silent_zero() {
        let h = RidgeHead::new(4, 1.0);
        let e = h.solve(TargetKind::SpreadMeans, 1.0, None).unwrap_err();
        assert_eq!(e, SolveError::TooFewClasses(0));
    }

    #[test]
    fn a_head_with_two_classes_solves_and_fits_its_own_prototypes() {
        let mut h = RidgeHead::new(6, 1.0);
        h.ensure_class(0);
        h.ensure_class(1);
        for i in 0..5 {
            let mut z0 = DVector::zeros(6);
            z0[0] = 1.0 + 0.1 * i as f64;
            h.push(0, &z0);
            let mut z1 = DVector::zeros(6);
            z1[1] = 2.0 + 0.1 * i as f64;
            h.push(1, &z1);
        }
        let theta = h.solve(TargetKind::SpreadMeans, 1.0, None).unwrap();
        assert_eq!(theta.nrows(), 6);
        assert_eq!(theta.ncols(), 6);
        assert!(theta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cosine_similarity_is_scale_invariant_and_bounded() {
        let protos = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let head = NcmHead::new(protos, vec!["a".into(), "b".into()]);
        let x = DVector::from_row_slice(&[2.0, 0.0, 0.0]);
        let s = head.similarities(&x);
        assert!(
            (s[0] - 1.0).abs() < 1e-12,
            "should be exactly 1 for a parallel vector"
        );
        assert!(s[1].abs() < 1e-12);
        for v in s.iter() {
            assert!((-1.0..=1.0).contains(v), "cosine left [-1,1]: {v}");
        }
        // Scale invariance is the point: the ETF constraint fixes inner products,
        // so a prototype's length is not meaningful.
        let scaled = DVector::from_row_slice(&[100.0, 0.0, 0.0]);
        assert_eq!(head.similarities(&x), head.similarities(&scaled));
    }

    #[test]
    fn id_score_is_the_maximum_similarity() {
        let protos = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let head = NcmHead::new(protos, vec!["a".into(), "b".into()]);
        let x = DVector::from_row_slice(&[1.0, 0.0]);
        let s = head.similarities(&x);
        let want = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((head.id_score(&x) - want).abs() < 1e-12);
    }

    #[test]
    fn an_empty_head_predicts_nothing_rather_than_panicking() {
        let head = NcmHead::new(DMatrix::zeros(0, 3), vec![]);
        assert!(head.predict(&DVector::zeros(3)).is_none());
        assert!(head.is_empty());
    }
}
