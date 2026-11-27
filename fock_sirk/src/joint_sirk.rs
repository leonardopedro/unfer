//! Hashimoto rational SIRK on the actual joint nested-Fock vectors.
//! No legacy separate-universe encoding or hidden amplitude pruning is used.
//! Explicit occupation projection records the full discarded action, and Ritz
//! residuals are recomputed with the full operator images, including its tail.

use crate::linalg::{whiten_gram, SirkError};
use nalgebra::{DMatrix, DVector};
use nested_fock_algebra::exact_inner::{
    ActionError, ActionLimits, InnerExpr, JointOuterBasis, NestedVector,
};
use num_complex::Complex64 as C;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, thiserror::Error)]
pub enum JointSirkError {
    #[error(transparent)]
    Action(#[from] ActionError),
    #[error(transparent)]
    LinearAlgebra(#[from] SirkError),
    #[error("invalid joint SIRK input: {0}")]
    Invalid(String),
    #[error("projected Hamiltonian is not Hermitian (relative defect {0})")]
    NonHermitian(f64),
    #[error(
        "shift-invert solve did not converge (relative residual {residual}, iterations {iterations})"
    )]
    ShiftSolve { residual: f64, iterations: usize },
    #[error("finite-window operator cache needs {components} entries, limit {limit}")]
    CacheBudget { components: usize, limit: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum JointKrylovMethod {
    /// Actual sequential resolvents, approximated by matrix-free GMRES.
    ShiftInvert {
        relative_tolerance: f64,
        max_iterations: usize,
    },
    /// The legacy forward-polynomial span, retained as a labeled diagnostic.
    ForwardProducts,
}

#[derive(Clone, Debug)]
pub struct JointSirkOptions {
    pub max_cached_components: usize,
    pub method: JointKrylovMethod,
    pub action_limits: ActionLimits,
    pub gram_relative_tolerance: f64,
    pub hermiticity_tolerance: f64,
    pub unit_norm_steps: bool,
    /// Sum of all retained inner bosonic occupations, across outer particles.
    /// This is an explicit numerical compression, never a deleted interaction.
    pub occupation_cutoff: Option<u64>,
    /// Optional independently declared finite basis window. Tail action is
    /// measured against the full operator, including omitted basis labels.
    pub basis_window: Option<BTreeSet<JointOuterBasis>>,
    /// Run the inner shift solves in coordinates of the cached window rather
    /// than in the nested maps. Every operation keeps the map path's per-label
    /// arithmetic and label order, so the result is bit-identical
    /// (`docs/VERIFY_PHI4_MOMENTUM.md` §5.10); `false` keeps the map walk as
    /// the reference implementation the pin test compares against.
    pub dense_window_solve: bool,
}

impl Default for JointSirkOptions {
    fn default() -> Self {
        Self {
            max_cached_components: 1_000_000,
            method: JointKrylovMethod::ShiftInvert {
                relative_tolerance: 1e-10,
                max_iterations: 48,
            },
            action_limits: ActionLimits::default(),
            gram_relative_tolerance: 1e-12,
            hermiticity_tolerance: 1e-9,
            unit_norm_steps: true,
            occupation_cutoff: None,
            basis_window: None,
            dense_window_solve: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct JointStateDictionary {
    labels: Vec<JointOuterBasis>,
    indices: BTreeMap<JointOuterBasis, usize>,
}
impl JointStateDictionary {
    pub fn register(&mut self, state: &NestedVector) {
        for basis in state.keys() {
            if !self.indices.contains_key(basis) {
                let index = self.labels.len();
                self.labels.push(basis.clone());
                self.indices.insert(basis.clone(), index);
            }
        }
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    pub fn is_bijective(&self) -> bool {
        self.labels.len() == self.indices.len()
            && self
                .labels
                .iter()
                .enumerate()
                .all(|(i, b)| self.indices.get(b) == Some(&i))
    }
    pub fn index(&self, b: &JointOuterBasis) -> Option<usize> {
        self.indices.get(b).copied()
    }
    pub fn basis(&self, i: usize) -> Option<&JointOuterBasis> {
        self.labels.get(i)
    }
}

// A numerical basis window permits reuse of COMPLETE operator images. Inner
// GMRES iterations need only P H P; full images remain available for tails,
// resolvent errors and Ritz residuals. No term of H is removed or approximated
// by a different model, and cache exhaustion is an explicit resource error.
struct FiniteWindowAction {
    labels: Vec<JointOuterBasis>,
    indices: BTreeMap<JointOuterBasis, usize>,
    images: Vec<NestedVector>,
    matrix: DMatrix<C>,
    components: usize,
}
impl FiniteWindowAction {
    fn build(
        h: &InnerExpr,
        window: &BTreeSet<JointOuterBasis>,
        options: &JointSirkOptions,
    ) -> Result<Self, JointSirkError> {
        let labels: Vec<_> = window
            .iter()
            .filter(|b| {
                options.occupation_cutoff.is_none_or(|cutoff| {
                    b.inner_occupations()
                        .iter()
                        .map(|(inner, n)| inner.total_bosons() * u64::from(*n))
                        .sum::<u64>()
                        <= cutoff
                })
            })
            .cloned()
            .collect();
        let n = labels.len();
        let mut components = n.saturating_mul(n);
        if components > options.max_cached_components {
            return Err(JointSirkError::CacheBudget {
                components,
                limit: options.max_cached_components,
            });
        }
        let indices = labels
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let mut images = Vec::new();
        let mut matrix = DMatrix::<C>::zeros(n, n);
        for (j, b) in labels.iter().enumerate() {
            let image = h.d_gamma(
                &NestedVector::from([(b.clone(), C::new(1.0, 0.0))]),
                options.action_limits,
            )?;
            components = components.saturating_add(image.len());
            if components > options.max_cached_components {
                return Err(JointSirkError::CacheBudget {
                    components,
                    limit: options.max_cached_components,
                });
            }
            for (i, target) in labels.iter().enumerate() {
                matrix[(i, j)] = image.get(target).copied().unwrap_or_default();
            }
            images.push(image);
        }
        Ok(Self {
            labels,
            indices,
            images,
            matrix,
            components,
        })
    }
    fn full(
        &self,
        state: &NestedVector,
        limits: ActionLimits,
    ) -> Result<NestedVector, ActionError> {
        let mut out = NestedVector::new();
        let mut work = 0usize;
        for (b, &a) in state {
            let j = *self.indices.get(b).ok_or(ActionError::UnsupportedBasis)?;
            work = work.saturating_add(self.images[j].len());
            if work > limits.max_work {
                return Err(ActionError::WorkLimit {
                    limit: limits.max_work,
                });
            }
            joint_axpy(&mut out, &self.images[j], a, limits)?;
        }
        Ok(out)
    }
    fn projected(
        &self,
        state: &NestedVector,
        limits: ActionLimits,
    ) -> Result<NestedVector, ActionError> {
        if self.matrix.len() > limits.max_work {
            return Err(ActionError::WorkLimit {
                limit: limits.max_work,
            });
        }
        let mut coordinates = DVector::<C>::zeros(self.labels.len());
        for (b, &a) in state {
            coordinates[*self.indices.get(b).ok_or(ActionError::UnsupportedBasis)?] = a;
        }
        let image = &self.matrix * coordinates;
        let mut out = NestedVector::new();
        for (b, &a) in self.labels.iter().zip(image.iter()) {
            if !a.re.is_finite() || !a.im.is_finite() {
                return Err(ActionError::NonFiniteAmplitude);
            }
            if a != C::new(0.0, 0.0) {
                if out.len() >= limits.max_components {
                    return Err(ActionError::ComponentLimit {
                        limit: limits.max_components,
                    });
                }
                out.insert(b.clone(), a);
            }
        }
        Ok(out)
    }

    /// Coordinates of `state` over `self.labels`. Identical to the vector
    /// `projected` builds internally: one lookup per stored entry, so this is
    /// the only label comparison the inner solve pays per vector.
    fn coords(&self, state: &NestedVector) -> Result<Vec<C>, ActionError> {
        let mut coordinates = vec![C::new(0.0, 0.0); self.labels.len()];
        for (b, &a) in state {
            coordinates[*self.indices.get(b).ok_or(ActionError::UnsupportedBasis)?] = a;
        }
        Ok(coordinates)
    }

    /// Inverse of `coords`, with the same component guard and the same
    /// zero-dropping rule `projected` applies to its image. The `from_` prefix
    /// names the inverse of [`Self::coords`]; it is a projection map, not a
    /// constructor, so the usual no-`self` convention does not apply.
    #[allow(clippy::wrong_self_convention)]
    fn from_coords(
        &self,
        coordinates: &[C],
        limits: ActionLimits,
    ) -> Result<NestedVector, ActionError> {
        let mut out = NestedVector::new();
        for (b, &a) in self.labels.iter().zip(coordinates) {
            if !a.re.is_finite() || !a.im.is_finite() {
                return Err(ActionError::NonFiniteAmplitude);
            }
            if a != C::new(0.0, 0.0) {
                if out.len() >= limits.max_components {
                    return Err(ActionError::ComponentLimit {
                        limit: limits.max_components,
                    });
                }
                out.insert(b.clone(), a);
            }
        }
        Ok(out)
    }

    /// `P H P` as coordinates. This is the same `gemv` `projected` performs on
    /// the same matrix, so the single entry point is bit-identical; it simply
    /// keeps the result in coordinates instead of re-serialising a map.
    fn image_coords(&self, coordinates: &[C], limits: ActionLimits) -> Result<Vec<C>, ActionError> {
        if self.matrix.len() > limits.max_work {
            return Err(ActionError::WorkLimit {
                limit: limits.max_work,
            });
        }
        let image = &self.matrix * DVector::from_column_slice(coordinates);
        if image.iter().any(|a| !a.re.is_finite() || !a.im.is_finite()) {
            return Err(ActionError::NonFiniteAmplitude);
        }
        Ok(image.as_slice().to_vec())
    }
}

// The coordinate forms of `joint_inner_product`, `joint_norm` and
// `joint_axpy`. All three visit exactly the labels the map walk visits (the
// map stores no zero amplitudes), in the same ascending label order, and the
// scalar forms give each label the same arithmetic in the same sequence: the
// scalar fallback is bit-identical to the map operations, not merely close
// (`tests/joint_map_ops.rs` and the solver pin in `tests/phi4_momentum_sirk.rs`).
//
// The SIMD inner products are the one deliberate exception to bit-identity:
// each carries independent accumulators, so the sum is reassociated and its
// last bits move (`docs/VERIFY_PHI4_MOMENTUM.md` §5.12: `3.75x` off the `dot`
// kernel and `12.6 %` off the C = 26 solve on NEON, for a `1.6e-16` change in
// the return probability; the `x86_64` AVX2 path moves them the same way). The
// frozen sweep record was generated on the scalar fallback.
fn coords_inner_product(a: &[C], b: &[C]) -> C {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Advanced SIMD is part of the aarch64 baseline, and both
        // slices are read only at indices below `min(a.len(), b.len())`.
        unsafe { coords_inner_product_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 is not part of the x86-64 baseline, so it is detected once per
        // call (the macro's answer is cached by std) and the scalar path stays
        // the fallback.
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the feature test above is exactly the contract of the
            // `#[target_feature]` kernel, and both slices are read only at
            // indices below `min(a.len(), b.len())`.
            unsafe { coords_inner_product_avx2(a, b) }
        } else {
            coords_inner_product_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        coords_inner_product_scalar(a, b)
    }
}
/// The scalar fallback, used on every target without a SIMD path.
#[cfg(not(target_arch = "aarch64"))]
fn coords_inner_product_scalar(a: &[C], b: &[C]) -> C {
    let zero = C::new(0.0, 0.0);
    let mut sum = zero;
    for (x, y) in a.iter().zip(b) {
        if *x == zero || *y == zero {
            continue;
        }
        sum += x.conj() * y;
    }
    sum
}
/// Four interleaved accumulators: `re0/re1` carry the real parts of the two
/// half-vectors, `im0/im1` the imaginary parts, so the four `vfmaq_f64` chains
/// are independent until the final `vaddvq_f64`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn coords_inner_product_neon(a: &[C], b: &[C]) -> C {
    use core::arch::aarch64::*;
    let n = a.len().min(b.len());
    let ap = a.as_ptr() as *const f64;
    let bp = b.as_ptr() as *const f64;
    let mut re0 = vdupq_n_f64(0.0);
    let mut re1 = vdupq_n_f64(0.0);
    let mut im0 = vdupq_n_f64(0.0);
    let mut im1 = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n` keeps every load inside both slices.
        let (v0, v1, v2, v3, w0, w1, w2, w3) = unsafe {
            (
                vld1q_f64(ap.add(2 * i)),
                vld1q_f64(ap.add(2 * i + 2)),
                vld1q_f64(ap.add(2 * i + 4)),
                vld1q_f64(ap.add(2 * i + 6)),
                vld1q_f64(bp.add(2 * i)),
                vld1q_f64(bp.add(2 * i + 2)),
                vld1q_f64(bp.add(2 * i + 4)),
                vld1q_f64(bp.add(2 * i + 6)),
            )
        };
        let (ar0, ai0) = (vuzp1q_f64(v0, v1), vuzp2q_f64(v0, v1));
        let (ar1, ai1) = (vuzp1q_f64(v2, v3), vuzp2q_f64(v2, v3));
        let (br0, bi0) = (vuzp1q_f64(w0, w1), vuzp2q_f64(w0, w1));
        let (br1, bi1) = (vuzp1q_f64(w2, w3), vuzp2q_f64(w2, w3));
        re0 = vfmaq_f64(vfmaq_f64(re0, ar0, br0), ar1, br1);
        re1 = vfmaq_f64(vfmaq_f64(re1, ai0, bi0), ai1, bi1);
        im0 = vfmaq_f64(vfmaq_f64(im0, ar0, bi0), ar1, bi1);
        im1 = vfmaq_f64(vfmaq_f64(im1, ai0, br0), ai1, br1);
        i += 4;
    }
    let mut sum = C::new(
        vaddvq_f64(vaddq_f64(re0, re1)),
        vaddvq_f64(vsubq_f64(im0, im1)),
    );
    while i < n {
        sum += a[i].conj() * b[i];
        i += 1;
    }
    sum
}
/// AVX2/FMA inner product: four complex per vector, deinterleaved with
/// `permute2f128` + `unpacklo/hi` and accumulated in two independent
/// `re`/`im` pairs so the FMA chains are not latency-bound. The reassociation
/// is why this path is documented as ``solver-precision`` rather than
/// bit-identical to the map operation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn coords_inner_product_avx2(a: &[C], b: &[C]) -> C {
    use core::arch::x86_64::*;

    // SAFETY: `offset + 8` f64s lie inside a slice of at least four complex
    // values starting at `i` (the caller guarantees `i + 4 <= n`).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load(
        ap: *const f64,
        bp: *const f64,
        offset: usize,
    ) -> (__m256d, __m256d, __m256d, __m256d) {
        let av0 = unsafe { _mm256_loadu_pd(ap.add(offset)) };
        let av1 = unsafe { _mm256_loadu_pd(ap.add(offset + 4)) };
        let bv0 = unsafe { _mm256_loadu_pd(bp.add(offset)) };
        let bv1 = unsafe { _mm256_loadu_pd(bp.add(offset + 4)) };
        let at0 = _mm256_permute2f128_pd(av0, av1, 0x20);
        let at1 = _mm256_permute2f128_pd(av0, av1, 0x31);
        let bt0 = _mm256_permute2f128_pd(bv0, bv1, 0x20);
        let bt1 = _mm256_permute2f128_pd(bv0, bv1, 0x31);
        (
            _mm256_unpacklo_pd(at0, at1),
            _mm256_unpackhi_pd(at0, at1),
            _mm256_unpacklo_pd(bt0, bt1),
            _mm256_unpackhi_pd(bt0, bt1),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn horizontal(v: __m256d) -> f64 {
        let lo = _mm256_castpd256_pd128(v);
        let hi = _mm256_extractf128_pd(v, 1);
        let pair = _mm_add_pd(lo, hi);
        _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)))
    }

    let n = a.len().min(b.len());
    let ap = a.as_ptr() as *const f64;
    let bp = b.as_ptr() as *const f64;
    let mut re0 = _mm256_setzero_pd();
    let mut im0 = _mm256_setzero_pd();
    let mut re1 = _mm256_setzero_pd();
    let mut im1 = _mm256_setzero_pd();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` covers both four-complex groups.
        let (ar0, ai0, br0, bi0) = unsafe { load(ap, bp, 2 * i) };
        let (ar1, ai1, br1, bi1) = unsafe { load(ap, bp, 2 * i + 8) };
        re0 = _mm256_fmadd_pd(ar0, br0, re0);
        re0 = _mm256_fmadd_pd(ai0, bi0, re0);
        im0 = _mm256_fmadd_pd(ar0, bi0, im0);
        im0 = _mm256_fnmadd_pd(ai0, br0, im0);
        re1 = _mm256_fmadd_pd(ar1, br1, re1);
        re1 = _mm256_fmadd_pd(ai1, bi1, re1);
        im1 = _mm256_fmadd_pd(ar1, bi1, im1);
        im1 = _mm256_fnmadd_pd(ai1, br1, im1);
        i += 8;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n` covers the four-complex group.
        let (ar, ai, br, bi) = unsafe { load(ap, bp, 2 * i) };
        re0 = _mm256_fmadd_pd(ar, br, re0);
        re0 = _mm256_fmadd_pd(ai, bi, re0);
        im0 = _mm256_fmadd_pd(ar, bi, im0);
        im0 = _mm256_fnmadd_pd(ai, br, im0);
        i += 4;
    }
    let re = unsafe { horizontal(_mm256_add_pd(re0, re1)) };
    let im = unsafe { horizontal(_mm256_add_pd(im0, im1)) };
    let mut sum = C::new(re, im);
    while i < n {
        sum += a[i].conj() * b[i];
        i += 1;
    }
    sum
}

fn coords_norm(a: &[C]) -> f64 {
    let zero = C::new(0.0, 0.0);
    a.iter()
        .filter(|x| **x != zero)
        .fold(0.0_f64, |norm, x| norm.hypot(x.re).hypot(x.im))
}
fn coords_axpy(a: &mut [C], b: &[C], scale: C, limits: ActionLimits) -> Result<(), ActionError> {
    if b.len() > limits.max_work {
        return Err(ActionError::WorkLimit {
            limit: limits.max_work,
        });
    }
    if !scale.re.is_finite() || !scale.im.is_finite() {
        return Err(ActionError::NonFiniteAmplitude);
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Same runtime gate as the inner product; `coords_axpy_scalar` stays
        // the fallback.
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the feature test above is exactly the contract of the
            // `#[target_feature]` kernel, and it stays inside the bounds the
            // scalar loop uses (`min(a.len(), b.len())`).
            return unsafe { coords_axpy_avx2(a, b, scale) };
        }
    }
    coords_axpy_scalar(a, b, scale)
}

/// The scalar loop: zero entries are skipped, and a non-finite result is
/// reported from inside the loop after the earlier slots were written.
fn coords_axpy_scalar(a: &mut [C], b: &[C], scale: C) -> Result<(), ActionError> {
    let zero = C::new(0.0, 0.0);
    for (slot, value) in a.iter_mut().zip(b) {
        if *value == zero {
            continue;
        }
        let updated = *slot + scale * value;
        if !updated.re.is_finite() || !updated.im.is_finite() {
            return Err(ActionError::NonFiniteAmplitude);
        }
        *slot = updated;
    }
    Ok(())
}

/// AVX2 `axpy`, four complex per vector. Unlike the inner product this kernel
/// does **not** reassociate anything: the multiply and the add are separate
/// instructions in the same order as the scalar loop, so a block is
/// bit-identical to `coords_axpy_scalar`. Any block that holds a zero `b`
/// entry (the scalar path skips it, which matters for a `-0.0` slot) or that
/// would produce a non-finite result falls back to the scalar loop for those
/// four elements, so the skip and the error report are preserved exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn coords_axpy_avx2(a: &mut [C], b: &[C], scale: C) -> Result<(), ActionError> {
    use core::arch::x86_64::*;

    let n = a.len().min(b.len());
    let ap = a.as_mut_ptr() as *mut f64;
    let bp = b.as_ptr() as *const f64;
    let sr = _mm256_set1_pd(scale.re);
    let si = _mm256_set1_pd(scale.im);
    let zero = _mm256_setzero_pd();
    let inf = _mm256_set1_pd(f64::INFINITY);
    let sign = _mm256_set1_pd(-0.0);
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n` keeps both four-complex loads and stores inside
        // the slices.
        let (av0, av1, bv0, bv1) = unsafe {
            (
                _mm256_loadu_pd(ap.add(2 * i)),
                _mm256_loadu_pd(ap.add(2 * i + 4)),
                _mm256_loadu_pd(bp.add(2 * i)),
                _mm256_loadu_pd(bp.add(2 * i + 4)),
            )
        };
        let at0 = _mm256_permute2f128_pd(av0, av1, 0x20);
        let at1 = _mm256_permute2f128_pd(av0, av1, 0x31);
        let ar = _mm256_unpacklo_pd(at0, at1);
        let ai = _mm256_unpackhi_pd(at0, at1);
        let bt0 = _mm256_permute2f128_pd(bv0, bv1, 0x20);
        let bt1 = _mm256_permute2f128_pd(bv0, bv1, 0x31);
        let br = _mm256_unpacklo_pd(bt0, bt1);
        let bi = _mm256_unpackhi_pd(bt0, bt1);

        let has_zero = _mm256_and_pd(
            _mm256_cmp_pd(br, zero, _CMP_EQ_OQ),
            _mm256_cmp_pd(bi, zero, _CMP_EQ_OQ),
        );
        let nr = _mm256_add_pd(
            ar,
            _mm256_sub_pd(_mm256_mul_pd(sr, br), _mm256_mul_pd(si, bi)),
        );
        let ni = _mm256_add_pd(
            ai,
            _mm256_add_pd(_mm256_mul_pd(sr, bi), _mm256_mul_pd(si, br)),
        );
        let finite = _mm256_and_pd(
            _mm256_cmp_pd(_mm256_andnot_pd(sign, nr), inf, _CMP_LT_OQ),
            _mm256_cmp_pd(_mm256_andnot_pd(sign, ni), inf, _CMP_LT_OQ),
        );
        // Either hazard sends the four elements through the scalar loop, which
        // reproduces the skip and the error report in order.
        if _mm256_movemask_pd(has_zero) != 0 || _mm256_movemask_pd(finite) != 0x0f {
            coords_axpy_scalar(&mut a[i..i + 4], &b[i..i + 4], scale)?;
        } else {
            // `unpacklo/hi` interleave lanes 0/2 and 1/3, so one more
            // `permute2f128` puts the four complex back in ascending order.
            let r0 = _mm256_unpacklo_pd(nr, ni);
            let r1 = _mm256_unpackhi_pd(nr, ni);
            let out0 = _mm256_permute2f128_pd(r0, r1, 0x20);
            let out1 = _mm256_permute2f128_pd(r0, r1, 0x31);
            // SAFETY: the same `i + 4 <= n` bound as the loads.
            unsafe {
                _mm256_storeu_pd(ap.add(2 * i), out0);
                _mm256_storeu_pd(ap.add(2 * i + 4), out1);
            }
        }
        i += 4;
    }
    coords_axpy_scalar(&mut a[i..], &b[i..], scale)
}

pub fn joint_inner_product(a: &NestedVector, b: &NestedVector) -> C {
    // Sorted-merge walk. Both maps are ordered by label, so one comparison per
    // step advances the pair; the previous form binary-searched `b` for every
    // key of `a` (about ten label comparisons per entry). Products are formed
    // and accumulated in `a`'s order exactly as before, so the sum is
    // bit-identical to the filtering version -- this only removes lookups.
    let mut left = a.iter();
    let mut right = b.iter();
    let mut next_left = left.next();
    let mut next_right = right.next();
    let mut sum = C::new(0.0, 0.0);
    while let (Some((label_a, value_a)), Some((label_b, value_b))) = (next_left, next_right) {
        match label_a.cmp(label_b) {
            std::cmp::Ordering::Less => next_left = left.next(),
            std::cmp::Ordering::Greater => next_right = right.next(),
            std::cmp::Ordering::Equal => {
                sum += value_a.conj() * value_b;
                next_left = left.next();
                next_right = right.next();
            }
        }
    }
    sum
}
pub fn joint_norm(a: &NestedVector) -> f64 {
    // Scaled sum avoids unnecessary overflow/underflow in the norm itself.
    a.values()
        .fold(0.0_f64, |norm, x| norm.hypot(x.re).hypot(x.im))
}
pub fn joint_axpy(
    a: &mut NestedVector,
    b: &NestedVector,
    scale: C,
    limits: ActionLimits,
) -> Result<(), ActionError> {
    if a.len() > limits.max_components || b.len() > limits.max_components {
        return Err(ActionError::ComponentLimit {
            limit: limits.max_components,
        });
    }
    if !scale.re.is_finite() || !scale.im.is_finite() {
        return Err(ActionError::NonFiniteAmplitude);
    }
    if b.len() > limits.max_work {
        return Err(ActionError::WorkLimit {
            limit: limits.max_work,
        });
    }
    // Both maps are ordered by the same label order, so one ordered walk visits
    // every touched entry with a single comparison, instead of a tree descent
    // (about a dozen deep `JointOuterBasis` comparisons) per entry of `b`. That
    // descent was 87 % of the `C = 26` shift solve (`docs/VERIFY_PHI4_MOMENTUM.md`
    // §5.6): the map operation, not the arithmetic, was the cost.
    //
    // The arithmetic is unchanged per label -- `scale * amplitude` for a fresh
    // label, `*slot + scale * amplitude` otherwise, and the same zero-deletion
    // rule -- and no label's value depends on any other, so this is
    // bit-identical to the search version (`tests/joint_map_ops.rs` pins that).
    // Insertions and removals are collected and applied after the walk, since
    // the map cannot be restructured while it is borrowed mutably.
    let mut incoming = b.iter();
    let mut next = incoming.next();
    let mut fresh: Vec<(&JointOuterBasis, C)> = Vec::new();
    let mut vanishing: Vec<JointOuterBasis> = Vec::new();
    for (label, slot) in a.iter_mut() {
        while let Some((basis, amplitude)) = next {
            match basis.cmp(label) {
                std::cmp::Ordering::Less => {
                    let value = scale * amplitude;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(ActionError::NonFiniteAmplitude);
                    }
                    if value != C::new(0.0, 0.0) {
                        fresh.push((basis, value));
                    }
                    next = incoming.next();
                }
                std::cmp::Ordering::Greater => break,
                std::cmp::Ordering::Equal => {
                    let value = *slot + scale * amplitude;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(ActionError::NonFiniteAmplitude);
                    }
                    if value == C::new(0.0, 0.0) {
                        vanishing.push(label.clone());
                    } else {
                        *slot = value;
                    }
                    next = incoming.next();
                    break;
                }
            }
        }
    }
    // `b` can outrun `a` once `a` is exhausted.
    while let Some((basis, amplitude)) = next {
        let value = scale * amplitude;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(ActionError::NonFiniteAmplitude);
        }
        if value != C::new(0.0, 0.0) {
            fresh.push((basis, value));
        }
        next = incoming.next();
    }
    for (basis, value) in fresh {
        if a.len() >= limits.max_components {
            return Err(ActionError::ComponentLimit {
                limit: limits.max_components,
            });
        }
        a.insert(basis.clone(), value);
    }
    for label in vanishing {
        a.remove(&label);
    }
    Ok(())
}

pub fn project_occupation(state: &NestedVector, cutoff: u64) -> (NestedVector, NestedVector) {
    state
        .iter()
        .map(|(b, &a)| (b.clone(), a))
        .partition(|(b, _)| {
            b.inner_occupations()
                .iter()
                .map(|(inner, n)| inner.total_bosons() * u64::from(*n))
                .sum::<u64>()
                <= cutoff
        })
}

#[derive(Debug)]
pub struct JointSirkResult {
    pub h_proj: DMatrix<C>,
    /// Paper's bounded-resolvent compression B=V†(gamma+iH)^-1 V.
    pub projected_resolvent: Option<DMatrix<C>>,
    /// gamma I-B^-1 for rational SIRK; -i H_proj for the separately labeled
    /// forward/Galerkin diagnostic. The two projections generally differ.
    pub function_generator: DMatrix<C>,
    pub function_solve_error_bounds: Vec<f64>,
    pub gram: DMatrix<C>,
    pub whitening: DMatrix<C>,
    pub registry: JointStateDictionary,
    pub sequence: Vec<NestedVector>,
    /// Full H u_j, before occupation projection.
    pub full_images: Vec<NestedVector>,
    /// Reorthogonalized physical vectors avoid ill-conditioned Gram-coordinate
    /// cancellation in reconstruction and operator compression.
    pub orthonormal_basis: Vec<NestedVector>,
    pub orthonormal_images: Vec<NestedVector>,
    pub orthogonality_defect: f64,
    pub scales: Vec<f64>,
    pub shifts: Vec<C>,
    pub projected_action_tail_norms: Vec<f64>,
    pub hermiticity_defect: f64,
    /// Full shifted residuals divided by |Im z|. For a self-adjoint full H
    /// this bounds solve error, including occupation-projection error; no
    /// rounding enclosure is claimed. Empty for forward-product diagnostics.
    pub shift_solve_error_bounds: Vec<f64>,
    /// Arnoldi steps actually taken by each rational shift solve, in the same
    /// order as `shift_solve_error_bounds`. Empty for forward-product
    /// diagnostics (those take no solve at all).
    pub shift_solve_iterations: Vec<usize>,
    /// Arnoldi steps taken by each solve of the bounded-resolvent compression
    /// `B = V^dag (gamma+iH)^-1 V`; one per rank direction.
    pub function_solve_iterations: Vec<usize>,
    pub options: JointSirkOptions,
    pub cached_components: usize,
}

impl JointSirkResult {
    pub fn rank(&self) -> usize {
        self.h_proj.nrows()
    }
    /// Largest Arnoldi step count over the rational basis solves.
    pub fn max_shift_solve_iterations(&self) -> usize {
        self.shift_solve_iterations
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }
    /// Largest Arnoldi step count over the bounded-resolvent solves.
    pub fn max_function_solve_iterations(&self) -> usize {
        self.function_solve_iterations
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }
    /// Total Arnoldi steps taken by every shift-invert solve in this result.
    pub fn total_solve_iterations(&self) -> usize {
        self.shift_solve_iterations.iter().sum::<usize>()
            + self.function_solve_iterations.iter().sum::<usize>()
    }
    fn initial_coefficients(&self) -> DVector<C> {
        DVector::from_iterator(
            self.rank(),
            self.orthonormal_basis
                .iter()
                .map(|v| joint_inner_product(v, &self.sequence[0])),
        )
    }
    pub fn time_evolve(&self, time: f64) -> Result<DVector<C>, JointSirkError> {
        if !time.is_finite() {
            return Err(JointSirkError::Invalid("nonfinite evolution time".into()));
        }
        Ok((self.function_generator.clone() * C::new(time, 0.0)).exp()
            * self.initial_coefficients())
    }
    pub fn reconstruct(&self, coefficients: &DVector<C>) -> Result<NestedVector, JointSirkError> {
        if coefficients.len() != self.rank() {
            return Err(JointSirkError::Invalid("coefficient dimension".into()));
        }
        let mut out = NestedVector::new();
        for (j, &c) in coefficients.iter().enumerate() {
            joint_axpy(
                &mut out,
                &self.orthonormal_basis[j],
                c,
                self.options.action_limits,
            )?;
        }
        Ok(out)
    }
    /// (Ritz value, absolute full-operator residual, state norm). No ordering
    /// or unseen-spectrum certificate is inferred from these finite windows.
    pub fn ritz_residuals(&self) -> Result<Vec<(f64, f64, f64)>, JointSirkError> {
        let eig = self.h_proj.clone().symmetric_eigen();
        let mut out = Vec::new();
        for i in 0..self.rank() {
            let theta = eig.eigenvalues[i];
            let c = eig.eigenvectors.column(i).into_owned();
            let state = self.reconstruct(&c)?;
            let mut residual = NestedVector::new();
            for (j, &x) in c.iter().enumerate() {
                joint_axpy(
                    &mut residual,
                    &self.orthonormal_images[j],
                    x,
                    self.options.action_limits,
                )?;
            }
            joint_axpy(
                &mut residual,
                &state,
                C::new(-theta, 0.0),
                self.options.action_limits,
            )?;
            out.push((theta, joint_norm(&residual), joint_norm(&state)));
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(out)
    }
}

pub fn solve_joint_sirk(
    h: &InnerExpr,
    initial: &NestedVector,
    shifts: &[C],
    options: &JointSirkOptions,
) -> Result<JointSirkResult, JointSirkError> {
    if let Some(window) = &options.basis_window {
        let cache = FiniteWindowAction::build(h, window, options)?;
        let mut result = solve_joint_sirk_with_actions(
            |v| cache.full(v, options.action_limits),
            |v| cache.projected(v, options.action_limits),
            Some(&cache),
            initial,
            shifts,
            options,
        )?;
        result.cached_components = cache.components;
        return Ok(result);
    }
    solve_joint_sirk_with_matvec(
        |v| h.d_gamma(v, options.action_limits),
        initial,
        shifts,
        options,
    )
}

pub fn solve_joint_sirk_with_matvec<F>(
    matvec: F,
    initial: &NestedVector,
    shifts: &[C],
    options: &JointSirkOptions,
) -> Result<JointSirkResult, JointSirkError>
where
    F: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
{
    solve_joint_sirk_with_actions(
        &matvec,
        |v| Ok(project_joint(&matvec(v)?, options).0),
        None,
        initial,
        shifts,
        options,
    )
}

fn solve_joint_sirk_with_actions<F, P>(
    matvec: F,
    projected_matvec: P,
    window: Option<&FiniteWindowAction>,
    initial: &NestedVector,
    shifts: &[C],
    options: &JointSirkOptions,
) -> Result<JointSirkResult, JointSirkError>
where
    F: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
    P: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
{
    let m = shifts.len();
    if m == 0
        || !joint_norm(initial).is_finite()
        || joint_norm(initial) == 0.0
        || shifts
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        || !options.gram_relative_tolerance.is_finite()
        || options.gram_relative_tolerance <= 0.0
        || options.gram_relative_tolerance >= 1.0
        || !options.hermiticity_tolerance.is_finite()
        || options.hermiticity_tolerance <= 0.0
    {
        return Err(JointSirkError::Invalid(
            "nonzero finite input, positive tolerances and nonempty finite shifts required".into(),
        ));
    }
    if !project_joint(initial, options).1.is_empty() {
        return Err(JointSirkError::Invalid(
            "initial state lies outside the declared numerical window".into(),
        ));
    }
    let mut checked = NestedVector::new();
    joint_axpy(
        &mut checked,
        initial,
        C::new(1.0, 0.0),
        options.action_limits,
    )?;
    let mut sequence = vec![checked];
    let mut images = Vec::new();
    let mut tails = Vec::new();
    let mut scales = vec![1.0];
    let mut solve_bounds = Vec::new();
    let mut solve_iterations = Vec::new();
    for (k, &shift) in shifts.iter().enumerate() {
        let image = matvec(&sequence[k])?;
        let mut validated = NestedVector::new();
        joint_axpy(
            &mut validated,
            &image,
            C::new(1.0, 0.0),
            options.action_limits,
        )?;
        let (projected, tail) = project_joint(&validated, options);
        tails.push(joint_norm(&tail));
        images.push(validated);
        let next = match options.method {
            JointKrylovMethod::ForwardProducts => {
                let mut next = projected;
                joint_axpy(&mut next, &sequence[k], -shift, options.action_limits)?;
                next
            }
            JointKrylovMethod::ShiftInvert {
                relative_tolerance,
                max_iterations,
            } => {
                let (solution, bound, used) = shift_solve(
                    &matvec,
                    &projected_matvec,
                    window,
                    &sequence[k],
                    shift,
                    options,
                    relative_tolerance,
                    max_iterations,
                )?;
                solve_bounds.push(bound);
                solve_iterations.push(used);
                solution
            }
        };
        let norm = joint_norm(&next);
        if !norm.is_finite() {
            return Err(JointSirkError::Invalid("nonfinite forward sequence".into()));
        }
        if options.unit_norm_steps && norm > 0.0 {
            let mut normalized = NestedVector::new();
            joint_axpy(
                &mut normalized,
                &next,
                C::new(1.0 / norm, 0.0),
                options.action_limits,
            )?;
            scales.push(norm);
            sequence.push(normalized);
        } else {
            scales.push(1.0);
            sequence.push(next);
        }
    }
    let mut registry = JointStateDictionary::default();
    for state in &sequence {
        registry.register(state);
    }
    if !registry.is_bijective() {
        return Err(JointSirkError::Invalid(
            "joint registry aliases basis labels".into(),
        ));
    }
    let mut gram = DMatrix::<C>::zeros(m + 1, m + 1);
    for j in 0..=m {
        for k in 0..=m {
            gram[(j, k)] = joint_inner_product(&sequence[j], &sequence[k]);
        }
    }
    if gram.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
        return Err(JointSirkError::Invalid("nonfinite Gram matrix".into()));
    }
    let gsub = gram.view((0, 0), (m, m)).into_owned();
    let mut whitening = whiten_gram(&gsub, options.gram_relative_tolerance)?.w;
    let rank = whitening.ncols();
    let mut orthonormal = Vec::<NestedVector>::new();
    for i in 0..rank {
        let mut vector = NestedVector::new();
        for j in 0..m {
            joint_axpy(
                &mut vector,
                &sequence[j],
                whitening[(j, i)],
                options.action_limits,
            )?;
        }
        for _ in 0..2 {
            for k in 0..i {
                let coefficient = joint_inner_product(&orthonormal[k], &vector);
                joint_axpy(
                    &mut vector,
                    &orthonormal[k],
                    -coefficient,
                    options.action_limits,
                )?;
                for j in 0..m {
                    let previous = whitening[(j, k)];
                    whitening[(j, i)] -= previous * coefficient;
                }
            }
        }
        let norm = joint_norm(&vector);
        if !norm.is_finite() || norm < 1e-10 {
            return Err(JointSirkError::Invalid(
                "unstable physical Gram direction".into(),
            ));
        }
        let mut normalized = NestedVector::new();
        joint_axpy(
            &mut normalized,
            &vector,
            C::new(1.0 / norm, 0.0),
            options.action_limits,
        )?;
        for j in 0..m {
            whitening[(j, i)] /= norm;
        }
        orthonormal.push(normalized);
    }
    let mut orthogonality_defect = 0.0_f64;
    for i in 0..rank {
        for j in 0..rank {
            orthogonality_defect = orthogonality_defect.max(
                (joint_inner_product(&orthonormal[i], &orthonormal[j])
                    - C::new(if i == j { 1.0 } else { 0.0 }, 0.0))
                .norm(),
            );
        }
    }
    let mut orthonormal_images = Vec::new();
    for v in &orthonormal {
        registry.register(v);
        let image = matvec(v)?;
        let mut checked = NestedVector::new();
        joint_axpy(
            &mut checked,
            &image,
            C::new(1.0, 0.0),
            options.action_limits,
        )?;
        orthonormal_images.push(checked);
    }
    let mut reduced = DMatrix::<C>::zeros(rank, rank);
    for j in 0..rank {
        for k in 0..rank {
            reduced[(j, k)] = joint_inner_product(&orthonormal[j], &orthonormal_images[k]);
        }
    }
    let defect = (&reduced - reduced.adjoint()).norm() / reduced.norm().max(1.0);
    if !defect.is_finite() || defect > options.hermiticity_tolerance {
        return Err(JointSirkError::NonHermitian(defect));
    }
    let h_proj = (&reduced + reduced.adjoint()) * C::new(0.5, 0.0);
    let (projected_resolvent, function_generator, function_bounds, function_iterations) =
        match options.method {
            JointKrylovMethod::ForwardProducts => (
                None,
                h_proj.clone() * C::new(0.0, -1.0),
                Vec::new(),
                Vec::new(),
            ),
            JointKrylovMethod::ShiftInvert {
                relative_tolerance,
                max_iterations,
            } => {
                if shifts.iter().any(|z| z.re != 0.0 || z.im <= 0.0) {
                    return Err(JointSirkError::Invalid(
                        "Hashimoto A=-iH requires z=i gamma with gamma>0".into(),
                    ));
                }
                let shift = shifts[m - 1];
                let gamma = shift.im;
                let mut bounded = DMatrix::<C>::zeros(rank, rank);
                let mut bounds = Vec::new();
                let mut iterations_used = Vec::new();
                for j in 0..rank {
                    let (inverse, bound, used) = shift_solve(
                        &matvec,
                        &projected_matvec,
                        window,
                        &orthonormal[j],
                        shift,
                        options,
                        relative_tolerance,
                        max_iterations,
                    )?;
                    bounds.push(bound);
                    iterations_used.push(used);
                    for i in 0..rank {
                        bounded[(i, j)] = -C::i() * joint_inner_product(&orthonormal[i], &inverse);
                    }
                }
                let inverse = bounded.clone().try_inverse().ok_or_else(|| {
                    JointSirkError::Invalid("singular bounded-resolvent compression".into())
                })?;
                let generator = DMatrix::<C>::identity(rank, rank) * C::new(gamma, 0.0) - inverse;
                (Some(bounded), generator, bounds, iterations_used)
            }
        };
    Ok(JointSirkResult {
        h_proj,
        projected_resolvent,
        function_generator,
        function_solve_error_bounds: function_bounds,
        gram,
        whitening,
        registry,
        sequence,
        orthonormal_basis: orthonormal,
        orthonormal_images,
        orthogonality_defect,
        full_images: images,
        scales,
        shifts: shifts.to_vec(),
        projected_action_tail_norms: tails,
        hermiticity_defect: defect,
        shift_solve_error_bounds: solve_bounds,
        shift_solve_iterations: solve_iterations,
        function_solve_iterations: function_iterations,
        options: options.clone(),
        cached_components: 0,
    })
}

/// Positive gamma_j=(budget-j*spacing)/time, j=1..m, for A=-iH corresponds to
/// z_j=i gamma_j in the H forward products. This checks the positive shift
/// budget; it does not assert the function/numerical-range theorem hypotheses.
pub fn hashimoto_unitary_shifts(
    time: f64,
    budget: f64,
    spacing: f64,
    count: usize,
) -> Result<Vec<C>, JointSirkError> {
    if !time.is_finite()
        || time <= 0.0
        || !budget.is_finite()
        || !spacing.is_finite()
        || spacing <= 0.0
        || count == 0
        || budget <= count as f64 * spacing
    {
        return Err(JointSirkError::Invalid(
            "Hashimoto requires t>0, spacing>0, and count<budget/spacing".into(),
        ));
    }
    Ok((0..count)
        .map(|j| C::new(0.0, (budget - (j + 1) as f64 * spacing) / time))
        .collect())
}

/// Per-kernel timing of one GMRES solve, reported when `PHI4_PROFILE` is set.
/// `axpy` spans the whole modified Gram–Schmidt block, so it contains `dot`;
/// `matvec` is the projected action only.
#[derive(Default)]
struct GmresProfile {
    enabled: bool,
    matvec: std::time::Duration,
    axpy: std::time::Duration,
    axpy_calls: usize,
    dot: std::time::Duration,
    dot_calls: usize,
    lstsq: std::time::Duration,
}

impl GmresProfile {
    fn new() -> Self {
        Self {
            enabled: std::env::var("PHI4_PROFILE").is_ok(),
            ..Self::default()
        }
    }

    fn report(&self, steps: usize) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "PROFILE steps={steps} matvec={:.2}s axpy={:.2}s ({} calls, {:.3} ms/call) dot={:.2}s ({} calls, {:.3} ms/call) lstsq={:.2}s",
            self.matvec.as_secs_f64(),
            self.axpy.as_secs_f64(),
            self.axpy_calls,
            self.axpy.as_secs_f64() / self.axpy_calls.max(1) as f64 * 1e3,
            self.dot.as_secs_f64(),
            self.dot_calls,
            self.dot.as_secs_f64() / self.dot_calls.max(1) as f64 * 1e3,
            self.lstsq.as_secs_f64()
        );
    }
}

fn gmres_shift<F, P>(
    matvec: &F,
    projected_matvec: &P,
    rhs: &NestedVector,
    shift: C,
    options: &JointSirkOptions,
    tolerance: f64,
    iterations: usize,
) -> Result<(NestedVector, f64, usize), JointSirkError>
where
    F: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
    P: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
{
    if !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= 1.0
        || iterations == 0
        || shift.im == 0.0
    {
        return Err(JointSirkError::Invalid(
            "nonreal shifts, positive solve tolerance and iteration budget required".into(),
        ));
    }
    let beta = joint_norm(rhs);
    if beta == 0.0 {
        return Ok((NestedVector::new(), 0.0, 0));
    }
    let limits = options.action_limits;
    let mut profile = GmresProfile::new();
    let mut first = NestedVector::new();
    joint_axpy(&mut first, rhs, C::new(1.0 / beta, 0.0), limits)?;
    let mut basis = vec![first];
    let mut h = DMatrix::<C>::zeros(iterations + 1, iterations);
    let mut last_residual = f64::INFINITY;
    for k in 0..iterations {
        let clock = std::time::Instant::now();
        let image = projected_matvec(&basis[k])?;
        profile.matvec += clock.elapsed();
        let mut w = NestedVector::new();
        let clock = std::time::Instant::now();
        joint_axpy(&mut w, &image, C::new(1.0, 0.0), limits)?;
        joint_axpy(&mut w, &basis[k], -shift, limits)?;
        // Two passes of modified Gram-Schmidt, preserving tiny support entries.
        for _ in 0..2 {
            for j in 0..=k {
                let dot_clock = std::time::Instant::now();
                let coefficient = joint_inner_product(&basis[j], &w);
                profile.dot += dot_clock.elapsed();
                profile.dot_calls += 1;
                h[(j, k)] += coefficient;
                joint_axpy(&mut w, &basis[j], -coefficient, limits)?;
                profile.axpy_calls += 1;
            }
        }
        profile.axpy += clock.elapsed();
        let next_norm = joint_norm(&w);
        h[(k + 1, k)] = C::new(next_norm, 0.0);
        let small = h.view((0, 0), (k + 2, k + 1)).into_owned();
        let mut target = DVector::<C>::zeros(k + 2);
        target[0] = C::new(beta, 0.0);
        let clock = std::time::Instant::now();
        let coefficients = small
            .clone()
            .svd(true, true)
            .solve(&target, 1e-14)
            .map_err(|e| JointSirkError::Invalid(format!("GMRES least squares: {e}")))?;
        profile.lstsq += clock.elapsed();
        let estimated = (&small * &coefficients - &target).norm() / beta;
        if estimated <= tolerance || k + 1 == iterations || next_norm <= 1e-14 {
            let mut solution = NestedVector::new();
            for (j, &c) in coefficients.iter().enumerate() {
                joint_axpy(&mut solution, &basis[j], c, limits)?;
            }
            let mut residual = matvec(&solution)?;
            joint_axpy(&mut residual, &solution, -shift, limits)?;
            joint_axpy(&mut residual, rhs, C::new(-1.0, 0.0), limits)?;
            let full_norm = joint_norm(&residual);
            let compressed = project_joint(&residual, options).0;
            last_residual = joint_norm(&compressed) / beta;
            if last_residual <= tolerance {
                profile.report(k + 1);
                return Ok((solution, full_norm / shift.im.abs(), k + 1));
            }
            if next_norm <= 1e-14 {
                break;
            }
        }
        if next_norm == 0.0 || !next_norm.is_finite() {
            break;
        }
        let mut normalized = NestedVector::new();
        joint_axpy(&mut normalized, &w, C::new(1.0 / next_norm, 0.0), limits)?;
        basis.push(normalized);
    }
    profile.report(iterations);
    Err(JointSirkError::ShiftSolve {
        residual: last_residual,
        iterations,
    })
}

/// One complex Givens rotation `(c, s)`, with `c` real, such that
/// `[[c, s], [-conj(s), c]] [top; bottom] = [rotated; 0]`. `rotated` keeps the
/// phase of `top`, so the rotation is not just a modulus operation.
fn givens_rotation(top: C, bottom: C) -> (f64, C, C) {
    let zero = C::new(0.0, 0.0);
    let bottom_norm = bottom.norm();
    if bottom_norm == 0.0 {
        return (1.0, zero, top);
    }
    let top_norm = top.norm();
    if top_norm == 0.0 {
        return (0.0, bottom.conj() / bottom_norm, C::new(bottom_norm, 0.0));
    }
    let radius = top_norm.hypot(bottom_norm);
    let c = top_norm / radius;
    let s = top / top_norm * bottom.conj() / radius;
    (c, s, top / top_norm * C::new(radius, 0.0))
}

/// The same GMRES as `gmres_shift`, in coordinates of the declared window.
///
/// The map path re-serialises every `w` (image, shifted vector, each MGS
/// subtraction) into a `BTreeMap`, and each entry then pays a label descent.
/// `docs/VERIFY_PHI4_MOMENTUM.md` §5.9 measured that at `52 %` of the whole
/// `C = 26` cell. Here the basis stays in flat coordinates and the projected
/// action is the cached `gemv`, so no label is compared inside the loop at
/// all.
///
/// This is not an approximation: the projection is the same `gemv`
/// `FiniteWindowAction::projected` performs, and `coords_inner_product`,
/// `coords_norm` and `coords_axpy` apply the map operations' arithmetic in the
/// same label order (§5.10). The map path is therefore kept as the reference
/// implementation, and `solve_joint_sirk` with
/// `dense_window_solve: false` returns solutions identical on `to_bits()`.
fn gmres_shift_window<F>(
    cache: &FiniteWindowAction,
    matvec: &F,
    rhs: &NestedVector,
    shift: C,
    options: &JointSirkOptions,
    tolerance: f64,
    iterations: usize,
) -> Result<(NestedVector, f64, usize), JointSirkError>
where
    F: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
{
    if !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= 1.0
        || iterations == 0
        || shift.im == 0.0
    {
        return Err(JointSirkError::Invalid(
            "nonreal shifts, positive solve tolerance and iteration budget required".into(),
        ));
    }
    let beta = joint_norm(rhs);
    if beta == 0.0 {
        return Ok((NestedVector::new(), 0.0, 0));
    }
    let limits = options.action_limits;
    let zero = C::new(0.0, 0.0);
    let n = cache.labels.len();
    if n > limits.max_components {
        return Err(ActionError::ComponentLimit {
            limit: limits.max_components,
        }
        .into());
    }
    let mut profile = GmresProfile::new();
    let mut first = vec![zero; n];
    coords_axpy(
        &mut first,
        &cache.coords(rhs)?,
        C::new(1.0 / beta, 0.0),
        limits,
    )?;
    let mut basis: Vec<Vec<C>> = vec![first];
    let mut h = DMatrix::<C>::zeros(iterations + 1, iterations);
    // Rotations accumulated so far, and the rotated right-hand side.
    let mut rotations: Vec<(f64, C)> = Vec::new();
    let mut transformed = DVector::<C>::zeros(iterations + 1);
    transformed[0] = C::new(beta, 0.0);
    let mut last_residual = f64::INFINITY;
    for k in 0..iterations {
        let clock = std::time::Instant::now();
        let image = cache.image_coords(&basis[k], limits)?;
        profile.matvec += clock.elapsed();
        let clock = std::time::Instant::now();
        // `joint_axpy(&mut w, &image, 1.0)` is an exact copy on the map path,
        // so the image can be moved straight in.
        let mut w = image;
        coords_axpy(&mut w, &basis[k], -shift, limits)?;
        // Two passes of modified Gram-Schmidt, preserving tiny support entries.
        for _ in 0..2 {
            for j in 0..=k {
                let dot_clock = std::time::Instant::now();
                let coefficient = coords_inner_product(&basis[j], &w);
                profile.dot += dot_clock.elapsed();
                profile.dot_calls += 1;
                h[(j, k)] += coefficient;
                coords_axpy(&mut w, &basis[j], -coefficient, limits)?;
                profile.axpy_calls += 1;
            }
        }
        profile.axpy += clock.elapsed();
        let next_norm = coords_norm(&w);
        h[(k + 1, k)] = C::new(next_norm, 0.0);
        // Incremental least squares (§5.11): the Hessenberg column is rotated
        // into upper triangular form by one Givens rotation per step, and the
        // transformed right-hand side carries the residual estimate in its last
        // entry. The dense SVD this replaced cost `k³` per step -- 56 % of the
        // `C = 26` solve -- and only ever produced the same three things: the
        // residual estimate, the acceptance test's coefficients, and nothing
        // the loop used again.
        let clock = std::time::Instant::now();
        for (i, &(c, s)) in rotations.iter().enumerate() {
            let top = h[(i, k)];
            let bottom = h[(i + 1, k)];
            h[(i, k)] = C::new(c, 0.0) * top + s * bottom;
            h[(i + 1, k)] = -s.conj() * top + C::new(c, 0.0) * bottom;
        }
        let (c, s, rotated) = givens_rotation(h[(k, k)], h[(k + 1, k)]);
        rotations.push((c, s));
        h[(k, k)] = rotated;
        h[(k + 1, k)] = zero;
        let (top, bottom) = (transformed[k], transformed[k + 1]);
        transformed[k] = C::new(c, 0.0) * top + s * bottom;
        transformed[k + 1] = -s.conj() * top + C::new(c, 0.0) * bottom;
        profile.lstsq += clock.elapsed();
        let estimated = transformed[k + 1].norm() / beta;
        if estimated <= tolerance || k + 1 == iterations || next_norm <= 1e-14 {
            // Back-substitute the rotated triangular system for the same
            // coefficients the SVD returned.
            let clock = std::time::Instant::now();
            let mut coefficients = vec![zero; k + 1];
            for i in (0..=k).rev() {
                let mut accumulated = transformed[i];
                for j in (i + 1)..=k {
                    accumulated -= h[(i, j)] * coefficients[j];
                }
                let diagonal = h[(i, i)];
                if diagonal.norm() == 0.0 {
                    return Err(JointSirkError::Invalid(
                        "singular Hessenberg diagonal in the GMRES least squares".into(),
                    ));
                }
                coefficients[i] = accumulated / diagonal;
            }
            profile.lstsq += clock.elapsed();
            let mut coordinates = vec![zero; n];
            for (j, &c) in coefficients.iter().enumerate() {
                coords_axpy(&mut coordinates, &basis[j], c, limits)?;
            }
            let solution = cache.from_coords(&coordinates, limits)?;
            let mut residual = matvec(&solution)?;
            joint_axpy(&mut residual, &solution, -shift, limits)?;
            joint_axpy(&mut residual, rhs, C::new(-1.0, 0.0), limits)?;
            let full_norm = joint_norm(&residual);
            let compressed = project_joint(&residual, options).0;
            last_residual = joint_norm(&compressed) / beta;
            if last_residual <= tolerance {
                profile.report(k + 1);
                return Ok((solution, full_norm / shift.im.abs(), k + 1));
            }
            if next_norm <= 1e-14 {
                break;
            }
        }
        if next_norm == 0.0 || !next_norm.is_finite() {
            break;
        }
        let mut normalized = vec![zero; n];
        coords_axpy(&mut normalized, &w, C::new(1.0 / next_norm, 0.0), limits)?;
        basis.push(normalized);
    }
    profile.report(iterations);
    Err(JointSirkError::ShiftSolve {
        residual: last_residual,
        iterations,
    })
}

/// One shift solve, in the window's coordinates when a cached window is
/// available (and `dense_window_solve` is set), otherwise on the nested maps.
#[allow(clippy::too_many_arguments)]
fn shift_solve<F, P>(
    matvec: &F,
    projected_matvec: &P,
    window: Option<&FiniteWindowAction>,
    rhs: &NestedVector,
    shift: C,
    options: &JointSirkOptions,
    tolerance: f64,
    iterations: usize,
) -> Result<(NestedVector, f64, usize), JointSirkError>
where
    F: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
    P: Fn(&NestedVector) -> Result<NestedVector, ActionError>,
{
    match window {
        Some(cache) if options.dense_window_solve => {
            gmres_shift_window(cache, matvec, rhs, shift, options, tolerance, iterations)
        }
        _ => gmres_shift(
            matvec,
            projected_matvec,
            rhs,
            shift,
            options,
            tolerance,
            iterations,
        ),
    }
}

#[derive(Debug)]
pub struct JointEvolution {
    pub state: NestedVector,
    /// N8 finite-map defect bound per restart step, using the same returned
    /// basis, generator and full `H V` images as the evolution. Point f64
    /// estimates; the interval enclosure is separate (`sm_defect`).
    pub defect_certificates: Vec<crate::sm_defect::DefectTerms>,
    pub norms: Vec<f64>,
    pub action_tail_norms: Vec<Vec<f64>>,
    pub energies: Vec<C>,
    pub shift_solve_error_bounds: Vec<Vec<f64>>,
    pub function_solve_error_bounds: Vec<Vec<f64>>,
    pub orthogonality_defects: Vec<f64>,
}
pub fn evolve_restarted_joint(
    h: &InnerExpr,
    initial: &NestedVector,
    total_time: f64,
    steps: usize,
    shifts: &[C],
    options: &JointSirkOptions,
) -> Result<JointEvolution, JointSirkError> {
    if steps == 0 || !total_time.is_finite() {
        return Err(JointSirkError::Invalid(
            "finite time and steps>0 required".into(),
        ));
    }
    let mut state = initial.clone();
    let mut norms = vec![joint_norm(initial)];
    let initial_norm = norms[0];
    if initial_norm == 0.0 || !initial_norm.is_finite() {
        return Err(JointSirkError::Invalid(
            "nonzero finite restart state required".into(),
        ));
    }
    let mut energies = vec![
        joint_inner_product(initial, &h.d_gamma(initial, options.action_limits)?)
            / (initial_norm * initial_norm),
    ];
    let mut tails = Vec::new();
    let mut shift_bounds = Vec::new();
    let mut function_bounds = Vec::new();
    let mut orthogonality = Vec::new();
    let mut defects = Vec::new();
    let step_time = total_time / steps as f64;
    for _ in 0..steps {
        let solve = solve_joint_sirk(h, &state, shifts, options)?;
        defects.push(crate::sm_defect::defect_terms(
            &solve.orthonormal_basis,
            &solve.orthonormal_images,
            &solve.function_generator,
            &state,
            step_time,
        ));
        state = solve.reconstruct(&solve.time_evolve(step_time)?)?;
        norms.push(joint_norm(&state));
        let norm = *norms.last().unwrap();
        if norm == 0.0 {
            return Err(JointSirkError::Invalid(
                "restart approximation collapsed to zero".into(),
            ));
        }
        energies.push(
            joint_inner_product(&state, &h.d_gamma(&state, options.action_limits)?) / (norm * norm),
        );
        shift_bounds.push(solve.shift_solve_error_bounds);
        function_bounds.push(solve.function_solve_error_bounds);
        orthogonality.push(solve.orthogonality_defect);
        tails.push(solve.projected_action_tail_norms);
    }
    Ok(JointEvolution {
        state,
        defect_certificates: defects,
        norms,
        action_tail_norms: tails,
        energies,
        shift_solve_error_bounds: shift_bounds,
        function_solve_error_bounds: function_bounds,
        orthogonality_defects: orthogonality,
    })
}

fn project_joint(state: &NestedVector, options: &JointSirkOptions) -> (NestedVector, NestedVector) {
    state
        .iter()
        .map(|(b, &a)| (b.clone(), a))
        .partition(|(b, _)| {
            options
                .basis_window
                .as_ref()
                .is_none_or(|basis| basis.contains(b))
                && options.occupation_cutoff.is_none_or(|cutoff| {
                    b.inner_occupations()
                        .iter()
                        .map(|(inner, n)| inner.total_bosons() * u64::from(*n))
                        .sum::<u64>()
                        <= cutoff
                })
        })
}
