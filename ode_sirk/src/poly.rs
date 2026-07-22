use num_complex::Complex64;
use rustc_hash::FxHashMap;

const SQRT2_INV: f64 = 0.7071067811865475;
const I: Complex64 = Complex64::new(0.0, 1.0);
const INV_SQRT2: Complex64 = Complex64::new(SQRT2_INV, 0.0);

/// A sum of normal-ordered monomials ∑_T coeff_T ∏_i (a†_i)^{c_i} (a_i)^{a_i}.
///
/// The key is a vector of `(creation_count, annihilation_count)` per mode.
/// Wick's recursive relations are applied exactly during `multiply_x_mode`
/// and `multiply_p_mode`, avoiding combinatorial string expansion.
#[derive(Clone, Debug, Default)]
pub struct NormalOrderedOp {
    pub terms: FxHashMap<Vec<(u32, u32)>, Complex64>,
}

impl NormalOrderedOp {
    pub fn new() -> Self {
        Self::default()
    }

    /// The identity operator: one term with zero occupation everywhere, coeff 1.
    pub fn identity(n_modes: usize) -> Self {
        let key = vec![(0u32, 0u32); n_modes];
        let mut terms = FxHashMap::default();
        terms.insert(key, Complex64::new(1.0, 0.0));
        Self { terms }
    }

    /// Create from a single monomial with real coefficient and explicit exponents.
    pub fn from_monomial(coeff: f64, exponents: &[u32]) -> Self {
        let key: Vec<(u32, u32)> = exponents.iter().map(|&e| (e, 0)).collect();
        let mut terms = FxHashMap::default();
        terms.insert(key, Complex64::new(coeff, 0.0));
        Self { terms }
    }

    /// Right-multiply by x_mode = (a†_mode + a_mode) / √2.
    ///
    /// Wick's recursion: a† on a normal-ordered (a†)^c (a)^a gives
    /// (a†)^{c+1} (a)^a  +  a · (a†)^c (a)^{a-1}.
    pub fn multiply_x_mode(&mut self, mode: usize) {
        let old: Vec<_> = self.terms.drain().collect();
        let sqrt2_inv = INV_SQRT2;

        for (key, coeff) in old {
            let (c_k, a_k) = key.get(mode).copied().unwrap_or((0, 0));

            let mut key = key;
            while key.len() <= mode {
                key.push((0, 0));
            }

            // Term from a†_mode / √2: (c_k+1, a_k)
            {
                let mut new_key = key.clone();
                new_key[mode] = (c_k + 1, a_k);
                *self.terms.entry(new_key).or_default() += coeff * sqrt2_inv;
            }

            // Term from commutation in a†_mode / √2: (c_k, a_k-1) if a_k > 0
            if a_k > 0 {
                let mut new_key = key.clone();
                new_key[mode] = (c_k, a_k - 1);
                *self.terms.entry(new_key).or_default() +=
                    coeff * sqrt2_inv * Complex64::new(a_k as f64, 0.0);
            }

            // Term from a_mode / √2: (c_k, a_k+1)
            {
                let mut new_key = key;
                new_key[mode] = (c_k, a_k + 1);
                *self.terms.entry(new_key).or_default() += coeff * sqrt2_inv;
            }
        }
    }

    /// Right-multiply by p_mode = -i(a_mode - a†_mode) / √2.
    pub fn multiply_p_mode(&mut self, mode: usize) {
        let old: Vec<_> = self.terms.drain().collect();
        let neg_i_sqrt2 = -I * INV_SQRT2;
        let i_sqrt2 = I * INV_SQRT2;

        for (key, coeff) in old {
            let (c_k, a_k) = key.get(mode).copied().unwrap_or((0, 0));

            let mut key = key;
            while key.len() <= mode {
                key.push((0, 0));
            }

            // Term from -i·a_mode / √2: (c_k, a_k+1)
            {
                let mut new_key = key.clone();
                new_key[mode] = (c_k, a_k + 1);
                *self.terms.entry(new_key).or_default() += coeff * neg_i_sqrt2;
            }

            // Term from i·a†_mode / √2: (c_k+1, a_k)
            {
                let mut new_key = key.clone();
                new_key[mode] = (c_k + 1, a_k);
                *self.terms.entry(new_key).or_default() += coeff * i_sqrt2;
            }

            // Commutation term from i·a†_mode / √2: (c_k, a_k-1) if a_k > 0
            if a_k > 0 {
                let mut new_key = key;
                new_key[mode] = (c_k, a_k - 1);
                *self.terms.entry(new_key).or_default() +=
                    coeff * i_sqrt2 * Complex64::new(a_k as f64, 0.0);
            }
        }
    }

    /// Maximum total occupation (c_i + a_i) across all modes and terms.
    pub fn degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|key| key.iter().map(|(c, a)| c + a).sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Add a scalar multiple of another NormalOrderedOp.
    pub fn add_scaled(&mut self, other: &Self, scale: Complex64) {
        for (key, coeff) in &other.terms {
            *self.terms.entry(key.clone()).or_default() += scale * coeff;
        }
    }

    /// Prune terms with |coeff| below threshold.
    pub fn prune(&mut self, eps: f64) {
        let thresh = eps * eps;
        self.terms.retain(|_, v| v.norm_sqr() > thresh);
    }

    /// Convert to nested_fock_algebra operator terms.
    pub fn to_operator_terms(&self) -> Vec<(Complex64, Vec<nested_fock_algebra::Operator>)> {
        use nested_fock_algebra::Operator;

        self.terms
            .iter()
            .map(|(key, &coeff)| {
                let mut ops = Vec::new();
                for (mode, &(c, a)) in key.iter().enumerate() {
                    let mode_u32 = mode as u32;
                    for _ in 0..c {
                        ops.push(Operator::InnerBosonCreate(mode_u32));
                    }
                    for _ in 0..a {
                        ops.push(Operator::InnerBosonAnnihilate(mode_u32));
                    }
                }
                (coeff, ops)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_has_coeff_one() {
        let op = NormalOrderedOp::identity(2);
        assert_eq!(op.terms.len(), 1);
        let key = vec![(0, 0), (0, 0)];
        assert!((op.terms[&key] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn multiply_x_mode_zero_mode() {
        let mut op = NormalOrderedOp::identity(1);
        op.multiply_x_mode(0);
        // x = (a† + a)/√2  →  right-multiply identity by x:
        //   key(1,0): 1/√2  (from a†)
        //   key(0,1): 1/√2  (from a)
        assert_eq!(op.terms.len(), 2);
        let key_10 = vec![(1, 0)];
        let key_01 = vec![(0, 1)];
        assert!((op.terms[&key_10] - Complex64::new(SQRT2_INV, 0.0)).norm() < 1e-12);
        assert!((op.terms[&key_01] - Complex64::new(SQRT2_INV, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn multiply_x_mode_on_occupied() {
        // Start with a† (mode 0 has c=1,a=0), right-multiply by x
        let mut op = NormalOrderedOp::from_monomial(1.0, &[1]);
        op.multiply_x_mode(0);
        // a† · x = a† · (a†+a)/√2 = (a†)²/√2 + a†a/√2
        //   key(2,0): 1/√2
        //   key(1,1): 1/√2
        assert_eq!(op.terms.len(), 2);

        let key_20 = vec![(2, 0)];
        let key_11 = vec![(1, 1)];

        let expected = Complex64::new(SQRT2_INV, 0.0);
        assert!((op.terms[&key_20] - expected).norm() < 1e-12);
        assert!((op.terms[&key_11] - expected).norm() < 1e-12);
    }

    #[test]
    fn multiply_p_mode_zero_mode() {
        let mut op = NormalOrderedOp::identity(1);
        op.multiply_p_mode(0);
        // p = -i(a - a†)/√2  →  right-multiply identity by p:
        //   key(0,1): -i/√2  (from a)
        //   key(1,0): +i/√2  (from a†)
        assert_eq!(op.terms.len(), 2);
        let key_01 = vec![(0, 1)];
        let key_10 = vec![(1, 0)];
        let expected_a = -I * INV_SQRT2;
        let expected_adag = I * INV_SQRT2;
        assert!((op.terms[&key_01] - expected_a).norm() < 1e-12);
        assert!((op.terms[&key_10] - expected_adag).norm() < 1e-12);
    }

    #[test]
    fn multiply_p_mode_on_occupied() {
        // Start with a† (c=1,a=0), right-multiply by p
        let mut op = NormalOrderedOp::from_monomial(1.0, &[1]);
        op.multiply_p_mode(0);
        // a† · p = a† · (-i)(a - a†)/√2
        //   key(1,1): -i/√2  (from a)
        //   key(2,0): +i/√2  (from a†)
        assert_eq!(op.terms.len(), 2);

        let key_11 = vec![(1, 1)];
        let key_20 = vec![(2, 0)];

        let expected_11 = -I * INV_SQRT2;
        let expected_20 = I * INV_SQRT2;

        assert!((op.terms[&key_11] - expected_11).norm() < 1e-12);
        assert!((op.terms[&key_20] - expected_20).norm() < 1e-12);
    }

    #[test]
    fn degree_empty() {
        let op = NormalOrderedOp::new();
        assert_eq!(op.degree(), 0);
    }

    #[test]
    fn degree_monomial() {
        let op = NormalOrderedOp::from_monomial(1.0, &[2, 3]);
        assert_eq!(op.degree(), 5);
    }

    #[test]
    fn to_operator_terms_roundtrip() {
        let op = NormalOrderedOp::from_monomial(1.0, &[1, 2]);
        let terms = op.to_operator_terms();
        assert_eq!(terms.len(), 1);
        let (coeff, ops) = &terms[0];
        assert!((coeff - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert_eq!(ops.len(), 3); // c_0, c_1, c_1
    }
}
