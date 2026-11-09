//! Hashimoto rational SIRK on the actual joint nested-Fock vectors.
//! No legacy separate-universe encoding or hidden amplitude pruning is used.
//! Explicit occupation projection records the full discarded action, and Ritz
//! residuals are recomputed with the full operator images, including its tail.

use crate::linalg::{SirkError, whiten_gram};
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
        let mut components = n.checked_mul(n).unwrap_or(usize::MAX);
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
            components = components.checked_add(image.len()).unwrap_or(usize::MAX);
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
            work = work.checked_add(self.images[j].len()).unwrap_or(usize::MAX);
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
}

pub fn joint_inner_product(a: &NestedVector, b: &NestedVector) -> C {
    a.iter()
        .filter_map(|(s, x)| b.get(s).map(|y| x.conj() * y))
        .sum()
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
    for (basis, amplitude) in b {
        let value = a.get(basis).copied().unwrap_or_default() + scale * amplitude;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(ActionError::NonFiniteAmplitude);
        }
        if value == C::new(0.0, 0.0) {
            a.remove(basis);
        } else {
            if !a.contains_key(basis) && a.len() >= limits.max_components {
                return Err(ActionError::ComponentLimit {
                    limit: limits.max_components,
                });
            }
            a.insert(basis.clone(), value);
        }
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
    pub options: JointSirkOptions,
    pub cached_components: usize,
}

impl JointSirkResult {
    pub fn rank(&self) -> usize {
        self.h_proj.nrows()
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
        initial,
        shifts,
        options,
    )
}

fn solve_joint_sirk_with_actions<F, P>(
    matvec: F,
    projected_matvec: P,
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
                let (solution, bound) = gmres_shift(
                    &matvec,
                    &projected_matvec,
                    &sequence[k],
                    shift,
                    options,
                    relative_tolerance,
                    max_iterations,
                )?;
                solve_bounds.push(bound);
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
    let (projected_resolvent, function_generator, function_bounds) = match options.method {
        JointKrylovMethod::ForwardProducts => {
            (None, h_proj.clone() * C::new(0.0, -1.0), Vec::new())
        }
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
            for j in 0..rank {
                let (inverse, bound) = gmres_shift(
                    &matvec,
                    &projected_matvec,
                    &orthonormal[j],
                    shift,
                    options,
                    relative_tolerance,
                    max_iterations,
                )?;
                bounds.push(bound);
                for i in 0..rank {
                    bounded[(i, j)] = -C::i() * joint_inner_product(&orthonormal[i], &inverse);
                }
            }
            let inverse = bounded.clone().try_inverse().ok_or_else(|| {
                JointSirkError::Invalid("singular bounded-resolvent compression".into())
            })?;
            let generator = DMatrix::<C>::identity(rank, rank) * C::new(gamma, 0.0) - inverse;
            (Some(bounded), generator, bounds)
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

fn gmres_shift<F, P>(
    matvec: &F,
    projected_matvec: &P,
    rhs: &NestedVector,
    shift: C,
    options: &JointSirkOptions,
    tolerance: f64,
    iterations: usize,
) -> Result<(NestedVector, f64), JointSirkError>
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
        return Ok((NestedVector::new(), 0.0));
    }
    let limits = options.action_limits;
    let mut first = NestedVector::new();
    joint_axpy(&mut first, rhs, C::new(1.0 / beta, 0.0), limits)?;
    let mut basis = vec![first];
    let mut h = DMatrix::<C>::zeros(iterations + 1, iterations);
    let mut last_residual = f64::INFINITY;
    for k in 0..iterations {
        let image = projected_matvec(&basis[k])?;
        let mut w = NestedVector::new();
        joint_axpy(&mut w, &image, C::new(1.0, 0.0), limits)?;
        joint_axpy(&mut w, &basis[k], -shift, limits)?;
        // Two passes of modified Gram-Schmidt, preserving tiny support entries.
        for _ in 0..2 {
            for j in 0..=k {
                let coefficient = joint_inner_product(&basis[j], &w);
                h[(j, k)] += coefficient;
                joint_axpy(&mut w, &basis[j], -coefficient, limits)?;
            }
        }
        let next_norm = joint_norm(&w);
        h[(k + 1, k)] = C::new(next_norm, 0.0);
        let small = h.view((0, 0), (k + 2, k + 1)).into_owned();
        let mut target = DVector::<C>::zeros(k + 2);
        target[0] = C::new(beta, 0.0);
        let coefficients = small
            .clone()
            .svd(true, true)
            .solve(&target, 1e-14)
            .map_err(|e| JointSirkError::Invalid(format!("GMRES least squares: {e}")))?;
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
                return Ok((solution, full_norm / shift.im.abs()));
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
    Err(JointSirkError::ShiftSolve {
        residual: last_residual,
        iterations,
    })
}

#[derive(Debug)]
pub struct JointEvolution {
    pub state: NestedVector,
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
    for _ in 0..steps {
        let solve = solve_joint_sirk(h, &state, shifts, options)?;
        state = solve.reconstruct(&solve.time_evolve(total_time / steps as f64)?)?;
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
