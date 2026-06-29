//! Two-level hashing primitives for the QFM tomographic pipeline.
//!
//! - **Level 1 (`CountSketch`)**: a sparse, deterministic Count-Sketch matrix
//!   S_1 in R^{k x d} that reduces a raw d-dim configuration to a k-dim
//!   feature vector via hash buckets + random signs.
//! - **Level 2 (`FeatureToMode`)**: maps a k-dim feature distribution to a
//!   K_2-dim sketched Fock state. For a single training image (delta function
//!   at x̃_j), the output is a sparse single-excitation state |Ψ_j⟩ in C^{K_2}.

use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors from the Level 2 hash (mode registration).
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum FeatureToModeError {
    /// The next free mode index would exceed the configured K_2 bound.
    /// Caller must increase K_2 (it's a compile-time parameter of the
    /// flow Hamiltonian — see `build_flow_hamiltonian`).
    #[error("K_2 bound exceeded: next mode would be {next} >= k2_hint {k2}")]
    K2BoundExceeded { next: u32, k2: u32 },
}

/// Deterministic splitmix64 PRNG for reproducible hash bucketing.
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Level 1 hash: a sparse, deterministic Count-Sketch S_1 in R^{k x d}.
///
/// The full d x k matrix is never materialized — only the per-coordinate
/// `(bucket, sign)` pairs are stored (2 * d entries). The sketch maps
/// each raw coordinate c in {0..d} to a hash bucket h(c) in {0..k} with
/// a random sign s(c) in {-1, +1}.
#[derive(Debug, Clone)]
pub struct CountSketch {
    k: usize,
    buckets: Vec<usize>,
    signs: Vec<i8>,
}

impl CountSketch {
    /// Construct a new Count-Sketch with k buckets for d raw coordinates.
    /// The `seed` determines the hash assignment (deterministic).
    pub fn new(k: usize, d: usize, seed: u64) -> Self {
        let mut buckets = Vec::with_capacity(d);
        let mut signs = Vec::with_capacity(d);
        for c in 0..d {
            let cu = c as u64;
            let h = splitmix64(cu.wrapping_add(seed)) % (k as u64);
            let s = (splitmix64(cu.wrapping_add(seed).wrapping_add(0x9e3779b97f4a7c15)) & 1) as i8;
            buckets.push(h as usize);
            signs.push(if s == 0 { 1 } else { -1 });
        }
        Self { k, buckets, signs }
    }

    /// Number of sketch buckets.
    pub fn num_buckets(&self) -> usize {
        self.k
    }

    /// Number of raw coordinates the sketch was built for.
    pub fn num_coords(&self) -> usize {
        self.buckets.len()
    }

    /// Apply the sketch to a dense vector x in R^d.
    /// Returns x̃ in R^k where x̃[h] += s(c) * x[c].
    /// Complexity: O(d) (dense loop; for sparse inputs use `apply_indexed`).
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.k];
        let n = x.len().min(self.buckets.len());
        for (c, &xc) in x.iter().take(n).enumerate() {
            let h = self.buckets[c];
            let s = self.signs[c] as f64;
            out[h] += s * xc;
        }
        out
    }

    /// Apply the sketch to a sparse (index, value) representation.
    /// Complexity: O(nnz).
    pub fn apply_indexed(&self, indices: &[usize], values: &[f64]) -> Vec<f64> {
        assert_eq!(
            indices.len(),
            values.len(),
            "indices/values length mismatch"
        );
        let mut out = vec![0.0; self.k];
        for (&c, &v) in indices.iter().zip(values.iter()) {
            if c < self.buckets.len() {
                let h = self.buckets[c];
                let s = self.signs[c] as f64;
                out[h] += s * v;
            }
        }
        out
    }

    /// Materialize the full k x d sketch matrix (for analysis/tests only).
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let d = self.buckets.len();
        let mut mat = nalgebra::DMatrix::zeros(self.k, d);
        for c in 0..d {
            mat[(self.buckets[c], c)] = self.signs[c] as f64;
        }
        mat
    }

    /// Apply the sketch to the row-space of a matrix.
    /// For each column j of the input matrix (d x cols), produces
    /// the sketched column: `(S_1 * mat)[:, j]` where
    /// `(S_1 * mat)[h, j] = sum_c S_1[h, c] * mat[c, j] = sign(c) * mat[bucket(c), j]`.
    ///
    /// The output is k x cols (the sketch reduces the row dimension from d to k).
    /// This is the "Phi_tilde = S_1 * Phi" operation from the spec.
    pub fn apply_to_columns(&self, mat: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
        let d = mat.nrows();
        let cols = mat.ncols();
        let mut out = nalgebra::DMatrix::<f64>::zeros(self.k, cols);
        let n = self.buckets.len().min(self.signs.len()).min(d);
        for c in 0..n {
            let h = self.buckets[c];
            let s = self.signs[c] as f64;
            for j in 0..cols {
                out[(h, j)] += s * mat[(c, j)];
            }
        }
        out
    }
}

/// Level 2 hash: maps a k-dim feature vector to a mode index in {0..K_2},
/// then creates a single-excitation Fock state at that mode.
#[derive(Debug, Clone)]
pub struct FeatureToMode {
    feature_to_mode: FxHashMap<u64, u32>,
    /// K_2 bound: the next free mode index must be `< k2_bound`. Stored
    /// as `u32` to allow direct comparison against the assigned mode.
    /// A value of 0 means "no bound" (legacy unbounded mode).
    k2_bound: u32,
}

impl FeatureToMode {
    /// Create a new Level 2 hash with the given K_2 bound.
    ///
    /// `k2_hint` is the upper bound on the number of distinct modes that
    /// can be registered. If `k2_hint == 0`, the mode map is unbounded
    /// and `register` will never check the K_2 limit (legacy behavior).
    /// Otherwise, `register` returns a
    /// `FeatureToModeError::K2BoundExceeded` if assigning the next free
    /// index would equal or exceed the bound.
    pub fn new(k2_hint: usize) -> Self {
        Self {
            feature_to_mode: FxHashMap::default(),
            k2_bound: k2_hint as u32,
        }
    }

    /// Hash a k-dim feature vector to a u64 key (deterministic).
    pub fn hash_feature(features: &[f64]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for (i, &f) in features.iter().enumerate() {
            let bits = f.to_bits();
            h ^= splitmix64(bits.wrapping_add(i as u64));
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Register a feature key and assign it the next free mode index.
    /// Returns the assigned mode. Enforces the K_2 bound set at
    /// construction: if assigning the next free index would equal or
    /// exceed K_2, returns a `FeatureToModeError::K2BoundExceeded`.
    /// (The K_2 bound is 0 == "unbounded" for legacy compatibility.)
    pub fn register(&mut self, feature_key: u64) -> Result<u32, FeatureToModeError> {
        if let Some(&m) = self.feature_to_mode.get(&feature_key) {
            return Ok(m);
        }
        let m = self.feature_to_mode.len() as u32;
        if self.k2_bound > 0 && m >= self.k2_bound {
            return Err(FeatureToModeError::K2BoundExceeded {
                next: m,
                k2: self.k2_bound,
            });
        }
        self.feature_to_mode.insert(feature_key, m);
        Ok(m)
    }

    /// Resolve a feature key to its mode (exact lookup).
    pub fn resolve(&self, feature_key: u64) -> Option<u32> {
        self.feature_to_mode.get(&feature_key).copied()
    }

    /// Number of registered features.
    pub fn len(&self) -> usize {
        self.feature_to_mode.len()
    }

    /// Whether any features are registered.
    pub fn is_empty(&self) -> bool {
        self.feature_to_mode.is_empty()
    }

    /// The K_2 bound configured at construction. Returns 0 if the map
    /// is unbounded.
    pub fn k2_bound(&self) -> u32 {
        self.k2_bound
    }

    /// Find the nearest training feature to the query (L1 distance over
    /// k-dim features). Returns the mode of the nearest training point.
    /// If no training features are registered, returns None.
    pub fn nearest(&self, query: &[f64], training_features: &[(u64, Vec<f64>)]) -> Option<u32> {
        if training_features.is_empty() {
            return None;
        }
        let mut best_dist = f64::INFINITY;
        let mut best_key = 0u64;
        for (key, feat) in training_features {
            let dist: f64 = query
                .iter()
                .zip(feat.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_key = *key;
            }
        }
        self.resolve(best_key)
    }

    /// Create a single-excitation Fock state at the given mode.
    pub fn to_fock_state(&self, mode: u32) -> QuantumState {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, 1);
        let op = Operator::OuterBosonCreate(inner);
        op.apply_to_state(&QuantumState::vacuum())
    }

    /// Create a single-excitation Fock state from a k-dim feature vector.
    /// If the feature is registered, returns its mode; otherwise returns None.
    pub fn feature_to_fock_state(&self, features: &[f64]) -> Option<QuantumState> {
        let key = Self::hash_feature(features);
        self.resolve(key).map(|m| self.to_fock_state(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_sketch_deterministic() {
        let s1 = CountSketch::new(8, 100, 42);
        let s2 = CountSketch::new(8, 100, 42);
        assert_eq!(s1.buckets, s2.buckets);
        assert_eq!(s1.signs, s2.signs);
    }

    #[test]
    fn count_sketch_one_hot_magnitude_one() {
        let s = CountSketch::new(16, 100, 7);
        // One-hot at coordinate 5: x[5] = 1.0, all others 0.
        let mut x = vec![0.0; 100];
        x[5] = 1.0;
        let x_tilde = s.apply(&x);
        // The sketch puts s(5)*1.0 into bucket s.buckets[5].
        let h = s.buckets[5];
        let sign = s.signs[5] as f64;
        assert_eq!(
            x_tilde[h], sign,
            "one-hot should land at its bucket with correct sign"
        );
        // All other buckets should be 0.
        for (i, &v) in x_tilde.iter().enumerate() {
            if i != h {
                assert_eq!(v, 0.0, "bucket {i} should be 0, got {v}");
            }
        }
    }

    #[test]
    fn count_sketch_apply_indexed_matches_dense() {
        let s = CountSketch::new(4, 10, 0);
        let dense = s.apply(&[1.0, -1.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2]);
        let sparse = s.apply_indexed(&[0, 1, 2, 5, 9], &[1.0, -1.0, 0.5, 0.3, 0.2]);
        assert_eq!(dense, sparse);
    }

    #[test]
    fn count_sketch_to_dense_shape() {
        let s = CountSketch::new(3, 7, 1);
        let mat = s.to_dense();
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 7);
    }

    #[test]
    fn feature_to_mode_register_monotonic() {
        let mut s2 = FeatureToMode::new(10);
        assert_eq!(s2.register(100).unwrap(), 0);
        assert_eq!(s2.register(200).unwrap(), 1);
        assert_eq!(s2.register(300).unwrap(), 2);
        // Re-registering returns the same mode.
        assert_eq!(s2.register(100).unwrap(), 0);
    }

    #[test]
    fn feature_to_mode_register_respects_k2_bound() {
        let mut s2 = FeatureToMode::new(2);
        // First two succeed; the second gets mode 1.
        assert_eq!(s2.register(100).unwrap(), 0);
        assert_eq!(s2.register(200).unwrap(), 1);
        // Third must error with the bound exceeded.
        let err = s2.register(300).unwrap_err();
        assert_eq!(err, FeatureToModeError::K2BoundExceeded { next: 2, k2: 2 });
    }

    #[test]
    fn feature_to_mode_k2_bound_zero_means_unbounded() {
        // Legacy/unbounded mode: k2_hint = 0 disables the check.
        let mut s2 = FeatureToMode::new(0);
        for i in 0..50 {
            assert_eq!(s2.register(i).unwrap(), i as u32);
        }
        assert_eq!(s2.k2_bound(), 0);
    }

    #[test]
    fn feature_to_mode_nearest() {
        let mut s2 = FeatureToMode::new(10);
        let k1 = FeatureToMode::hash_feature(&[1.0, 0.0]);
        let k2 = FeatureToMode::hash_feature(&[0.0, 1.0]);
        s2.register(k1).unwrap();
        s2.register(k2).unwrap();
        let training = vec![(k1, vec![1.0, 0.0]), (k2, vec![0.0, 1.0])];
        // Query closer to k1.
        let mode = s2.nearest(&[0.9, 0.1], &training).unwrap();
        assert_eq!(mode, 0);
        // Query closer to k2.
        let mode = s2.nearest(&[0.1, 0.9], &training).unwrap();
        assert_eq!(mode, 1);
    }
}
