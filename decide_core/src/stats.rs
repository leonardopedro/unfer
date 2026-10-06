//! Streaming sufficient statistics.
//!
//! # Why this shape
//!
//! Both methods accumulate *only* in fixed spaces (the input space and the RP
//! space under a frozen, seeded projection). That is deliberate, and it is the
//! structural answer to the defect this project recorded as A4: statistics taken
//! under one projection must never be mixed into a prototype derived under
//! another. Here, "fixed" is enforced by construction — the RP matrix is stored
//! in [`RpLayer`](crate::rp::RpLayer) and never re-derived, so an RP-space mean
//! is always a mean of the same linear function of the same embedding space.
//!
//! # Exactness, and why order-invariance is free
//!
//! Every statistic here is a *sum* of per-sample contributions (counts, sums,
//! outer products). Sums are commutative and associative in exact arithmetic and
//! very nearly so in f64, so the batch and incremental paths agree up to
//! rounding, and a shuffled arrival order gives the same answer. `D4` turns both
//! facts into property tests rather than trusting this comment.
//!
//! The covariance uses the textbook identity
//! `cov = (Σ x xᵀ − n μ μᵀ) / (n − 1)`, which needs only `Σ x xᵀ` and `Σ x`.
//! The alternative — accumulating deviations from a *running* mean — is
//! numerically worse and makes the result depend on arrival order.

use nalgebra::{DMatrix, DVector};

/// Per-class counts and input-space means, plus the shared covariance.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    dim: usize,
    /// `counts[c]`, indexed by mapped class id.
    counts: Vec<u64>,
    /// `sums[c]`, the running `Σ x` for class `c`.
    sums: Vec<DVector<f64>>,
    /// Global `Σ x`.
    total_sum: DVector<f64>,
    /// Global `Σ x xᵀ`, a `dim × dim` accumulator.
    total_outer: DMatrix<f64>,
    n: u64,
}

impl StreamingStats {
    /// A store for `dim`-dimensional embeddings.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            counts: Vec::new(),
            sums: Vec::new(),
            total_sum: DVector::zeros(dim),
            total_outer: DMatrix::zeros(dim, dim),
            n: 0,
        }
    }

    /// Embedding dimension this store was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total samples seen, across all classes.
    pub fn total(&self) -> u64 {
        self.n
    }

    /// Grow the class vocabulary to hold `id`.
    pub fn ensure_class(&mut self, id: usize) {
        while self.counts.len() <= id {
            self.counts.push(0);
            self.sums.push(DVector::zeros(self.dim));
        }
    }

    /// Record one embedding for class `id`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != dim`, or if `id` has not been registered with
    /// [`Self::ensure_class`]. Both are programming errors on the op path, and
    /// the project's discipline (S30) prefers a loud panic during development to
    /// a silently wrong statistic.
    pub fn push(&mut self, id: usize, x: &DVector<f64>) {
        assert_eq!(
            x.len(),
            self.dim,
            "embedding dimension changed mid-stream: state is dim {}, got {}",
            self.dim,
            x.len()
        );
        assert!(
            id < self.counts.len(),
            "class id {id} was never registered with ensure_class"
        );
        self.counts[id] += 1;
        self.n += 1;
        for i in 0..self.dim {
            self.total_sum[i] += x[i];
            self.sums[id][i] += x[i];
            for j in 0..self.dim {
                self.total_outer[(i, j)] += x[i] * x[j];
            }
        }
    }

    /// Number of samples recorded for class `id`.
    pub fn count(&self, id: usize) -> u64 {
        self.counts.get(id).copied().unwrap_or(0)
    }

    /// Total across classes.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Mean of class `id`, or `None` if it has no samples.
    pub fn mean(&self, id: usize) -> Option<DVector<f64>> {
        let n = self.count(id);
        if n == 0 {
            return None;
        }
        Some(self.sums[id].map(|v| v / n as f64))
    }

    /// The `d × C` matrix of per-class means, skipping empty classes.
    ///
    /// Returns the columns alongside their ids, because the two must not be
    /// allowed to drift apart.
    pub fn class_means(&self) -> (Vec<usize>, DMatrix<f64>) {
        let ids: Vec<usize> = (0..self.counts.len())
            .filter(|&c| self.counts[c] > 0)
            .collect();
        let mut m = DMatrix::zeros(self.dim, ids.len());
        for (col, &c) in ids.iter().enumerate() {
            let mu = self.mean(c).expect("filtered to non-empty");
            for row in 0..self.dim {
                m[(row, col)] = mu[row];
            }
        }
        (ids, m)
    }

    /// Global mean, or `None` before any sample.
    pub fn global_mean(&self) -> Option<DVector<f64>> {
        if self.n == 0 {
            return None;
        }
        Some(self.total_sum.map(|v| v / self.n as f64))
    }

    /// Shared sample covariance, or `None` below two samples.
    ///
    /// Shared in the AnaCP sense: one covariance for all classes, used to whiten
    /// the class means before the target construction. `n - 1` is the unbiased
    /// denominator; with `n < 2` there is nothing to whiten with, and the
    /// caller falls back to the identity.
    pub fn shared_covariance(&self) -> Option<DMatrix<f64>> {
        if self.n < 2 {
            return None;
        }
        let mu = self.global_mean().expect("n >= 2");
        let n = self.n as f64;
        let mut cov = self.total_outer.clone();
        for i in 0..self.dim {
            for j in 0..self.dim {
                cov[(i, j)] -= n * mu[i] * mu[j];
            }
        }
        cov /= n - 1.0;
        Some(cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(xs)
    }

    #[test]
    fn a_fresh_store_has_no_mean_and_no_covariance() {
        let s = StreamingStats::new(3);
        assert_eq!(s.total(), 0);
        assert!(s.mean(0).is_none());
        assert!(s.global_mean().is_none());
        // Below two samples there is nothing to whiten with. Returning `None`
        // rather than a zero matrix is what stops the target construction from
        // dividing by a singular matrix on the first sample.
        assert!(s.shared_covariance().is_none());
    }

    #[test]
    fn counts_and_means_track_each_class_separately() {
        let mut s = StreamingStats::new(2);
        s.ensure_class(0);
        s.ensure_class(1);
        s.push(0, &v(&[1.0, 0.0]));
        s.push(0, &v(&[3.0, 0.0]));
        s.push(1, &v(&[0.0, 5.0]));
        assert_eq!(s.count(0), 2);
        assert_eq!(s.count(1), 1);
        assert_eq!(s.total(), 3);
        let m0 = s.mean(0).unwrap();
        assert!((m0[0] - 2.0).abs() < 1e-12, "{}", m0[0]);
        let m1 = s.mean(1).unwrap();
        assert!((m1[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn class_means_returns_columns_in_the_same_order_as_their_ids() {
        let mut s = StreamingStats::new(2);
        for c in 0..3 {
            s.ensure_class(c);
        }
        // Class 1 deliberately left empty: it must be skipped, and the surviving
        // ids must still line up with the columns.
        s.push(0, &v(&[1.0, 1.0]));
        s.push(2, &v(&[3.0, 3.0]));
        let (ids, m) = s.class_means();
        assert_eq!(ids, vec![0, 2]);
        assert_eq!(m.ncols(), 2);
        assert!((m[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((m[(0, 1)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_shared_covariance_matches_the_two_point_case_by_hand() {
        let mut s = StreamingStats::new(2);
        s.ensure_class(0);
        s.push(0, &v(&[1.0, 2.0]));
        s.push(0, &v(&[3.0, 6.0]));
        let cov = s.shared_covariance().unwrap();
        // mean = (2,4); deviations ±(1,2); sum of outer products = [[2,4],[4,8]];
        // divided by n-1 = 1.
        assert!((cov[(0, 0)] - 2.0).abs() < 1e-12, "{}", cov[(0, 0)]);
        assert!((cov[(0, 1)] - 4.0).abs() < 1e-12, "{}", cov[(0, 1)]);
        assert!((cov[(1, 1)] - 8.0).abs() < 1e-12, "{}", cov[(1, 1)]);
    }

    #[test]
    #[should_panic(expected = "never registered")]
    fn pushing_an_unregistered_class_is_a_loud_error() {
        let mut s = StreamingStats::new(2);
        s.push(7, &v(&[1.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "dimension changed")]
    fn pushing_a_different_dimension_is_a_loud_error() {
        let mut s = StreamingStats::new(2);
        s.ensure_class(0);
        s.push(0, &v(&[1.0, 1.0, 1.0]));
    }
}
