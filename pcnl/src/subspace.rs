//! R1 — the restricted affine subspace `θ(z) = θ_ref + C_half B z`.
//!
//! The model this module implements is *explicitly a different, smaller
//! posterior*: `z ~ N(0, I_r)` under the prior, so the chain samples
//! `π_r(dz) ∝ exp(−λ·Σ_i loss_i(θ(z))) N(0, I_r)(dz)` and says nothing about the
//! ambient `P`-dimensional posterior.  The two things a reader needs from a
//! subspace run are therefore (i) that `B` really is orthonormal and `C_half`
//! really is the declared prior factor, and (ii) that the gradient the chain
//! follows is the chain rule `Bᵀ C_halfᵀ ∇_θ Φ` and not some ambient gradient
//! with the wrong units.  Both are checked here and tested against finite
//! differences.
//!
//! `C_half` is diagonal (`scale[i]` on coordinate `i`), which is the only case
//! the reported priors use: `σ·I` for the isotropic prior.  A full covariance
//! factor would need a `P×P` triangular solve per lift for no measured reason,
//! so it is not offered and cannot be silently mis-specified.

use candle_core::{Device, Tensor};
use rand::{Rng, SeedableRng};

/// Why a basis or subspace was rejected.  Validation is not optional: a
/// non-orthonormal `B` silently changes the prior that the chain claims to use.
#[derive(Clone, Debug, PartialEq)]
pub enum SubspaceError {
    Empty,
    /// `r == 0`, or `r > p` (`B` cannot have more orthonormal columns than rows).
    BadRank {
        p: usize,
        r: usize,
    },
    /// `cols.len() != p * r` or `theta_ref/scale.len() != p`.
    BadLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite(&'static str),
    /// `max |(BᵀB) − I|` exceeded the tolerance.
    NotOrthonormal {
        gram_error: f64,
        tol: f64,
    },
    /// `C_half` is a covariance factor, so it must be strictly positive.
    BadScale {
        index: usize,
        value: f32,
    },
    /// The proposed informed basis has no `r`-th direction: the snapshots span
    /// fewer than `r` dimensions (identical snapshots are the usual cause).
    Degenerate {
        component: usize,
    },
}

impl std::fmt::Display for SubspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubspaceError::Empty => write!(f, "basis is empty"),
            SubspaceError::BadRank { p, r } => write!(f, "rank {r} not in 1..={p}"),
            SubspaceError::BadLength {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} values, got {got}"),
            SubspaceError::NonFinite(what) => write!(f, "{what} contains a non-finite value"),
            SubspaceError::NotOrthonormal { gram_error, tol } => write!(
                f,
                "columns are not orthonormal: max |B^T B - I| = {gram_error:.3e} > {tol:.1e}"
            ),
            SubspaceError::BadScale { index, value } => {
                write!(f, "C_half scale[{index}] = {value} is not > 0")
            }
            SubspaceError::Degenerate { component } => write!(
                f,
                "snapshots have no variance in direction {component} (rank too high for the path)"
            ),
        }
    }
}

impl std::error::Error for SubspaceError {}

/// Tolerance for `max |(BᵀB) − I|`.  The bases this project builds are
/// orthonormalised in f32, where the Gram matrix already carries `~1e-7` noise
/// per dot product of length `P`; `1e-5` rejects a genuinely skewed basis (a
/// duplicated column gives 1) while accepting a re-orthonormalised one.
pub const ORTHONORMAL_TOL: f64 = 1e-5;

/// Orthonormal columns `B` of size `P × r`, stored column-major
/// (`cols[j * p + i]` is entry `(i, j)`), so a lift is a contiguous axpy over
/// each column.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    p: usize,
    r: usize,
    cols: Vec<f32>,
}

impl Basis {
    /// Validate and wrap `p * r` column-major entries.
    pub fn new(cols: Vec<f32>, p: usize, r: usize) -> Result<Self, SubspaceError> {
        if p == 0 || r == 0 {
            return Err(SubspaceError::Empty);
        }
        if r > p {
            return Err(SubspaceError::BadRank { p, r });
        }
        if cols.len() != p * r {
            return Err(SubspaceError::BadLength {
                what: "basis columns",
                expected: p * r,
                got: cols.len(),
            });
        }
        if cols.iter().any(|x| !x.is_finite()) {
            return Err(SubspaceError::NonFinite("basis"));
        }
        let b = Self { p, r, cols };
        let err = b.gram_error();
        if !err.is_finite() || err > ORTHONORMAL_TOL {
            return Err(SubspaceError::NotOrthonormal {
                gram_error: err,
                tol: ORTHONORMAL_TOL,
            });
        }
        Ok(b)
    }

    /// A deterministic random orthonormal basis (Gaussian entries, two-pass
    /// modified Gram–Schmidt).  This is R1's *control*: it is informed by
    /// nothing, so an informed basis that cannot beat it is not informed.
    pub fn random(p: usize, r: usize, seed: u64) -> Result<Self, SubspaceError> {
        if p == 0 || r == 0 {
            return Err(SubspaceError::Empty);
        }
        if r > p {
            return Err(SubspaceError::BadRank { p, r });
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5eed_0b1e);
        let mut cols = vec![0.0f32; p * r];
        // Two passes: the first removes the bulk of the overlap, the second
        // removes what the first pass re-introduced in f32.
        for pass in 0..2 {
            for j in 0..r {
                if pass == 0 {
                    for i in 0..p {
                        cols[j * p + i] = gaussian(&mut rng) as f32;
                    }
                }
                for k in 0..j {
                    let dot: f32 = (0..p).map(|i| cols[k * p + i] * cols[j * p + i]).sum();
                    for i in 0..p {
                        cols[j * p + i] -= dot * cols[k * p + i];
                    }
                }
                let n: f32 = (0..p)
                    .map(|i| cols[j * p + i] * cols[j * p + i])
                    .sum::<f32>()
                    .sqrt();
                for i in 0..p {
                    cols[j * p + i] /= n;
                }
            }
        }
        Self::new(cols, p, r)
    }

    pub fn p(&self) -> usize {
        self.p
    }

    pub fn dim(&self) -> usize {
        self.r
    }

    pub fn col(&self, j: usize) -> &[f32] {
        &self.cols[j * self.p..(j + 1) * self.p]
    }

    pub fn columns(&self) -> &[f32] {
        &self.cols
    }

    /// `max |(BᵀB) − I|` — the number `new` gates on, exposed so a report can
    /// print what it actually checked.
    pub fn gram_error(&self) -> f64 {
        let mut worst = 0.0f64;
        for j in 0..self.r {
            for k in 0..self.r {
                let dot: f64 = (0..self.p)
                    .map(|i| {
                        f64::from(self.cols[j * self.p + i]) * f64::from(self.cols[k * self.p + i])
                    })
                    .sum();
                let want = if j == k { 1.0 } else { 0.0 };
                worst = worst.max((dot - want).abs());
            }
        }
        worst
    }

    /// `out = B z` (`out.len() == p`).
    pub fn apply(&self, z: &[f32], out: &mut [f32]) {
        assert_eq!(z.len(), self.r, "B z: z must have r entries");
        assert_eq!(out.len(), self.p, "B z: out must have p entries");
        out.fill(0.0);
        for (j, &zj) in z.iter().enumerate() {
            let col = self.col(j);
            for i in 0..self.p {
                out[i] += col[i] * zj;
            }
        }
    }

    /// `out = Bᵀ v` (`out.len() == r`).
    pub fn project(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(v.len(), self.p, "B^T v: v must have p entries");
        assert_eq!(out.len(), self.r, "B^T v: out must have r entries");
        for (j, o) in out.iter_mut().enumerate() {
            let col = self.col(j);
            *o = (0..self.p).map(|i| col[i] * v[i]).sum();
        }
    }
}

/// Standard normal via Box–Muller (the crate has no rand_distr dependency and a
/// basis only needs `p·r` draws).
fn gaussian<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = 1.0 - rng.random::<f64>(); // (0, 1]
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// The affine chart `θ(z) = θ_ref + C_half B z`, with `C_half = diag(scale)`.
#[derive(Clone, Debug)]
pub struct AffineSubspace {
    theta_ref: Vec<f32>,
    basis: Basis,
    scale: Vec<f32>,
}

impl AffineSubspace {
    pub fn new(theta_ref: Vec<f32>, basis: Basis, scale: Vec<f32>) -> Result<Self, SubspaceError> {
        let p = basis.p();
        if theta_ref.len() != p {
            return Err(SubspaceError::BadLength {
                what: "theta_ref",
                expected: p,
                got: theta_ref.len(),
            });
        }
        if scale.len() != p {
            return Err(SubspaceError::BadLength {
                what: "C_half scale",
                expected: p,
                got: scale.len(),
            });
        }
        if theta_ref.iter().any(|x| !x.is_finite()) {
            return Err(SubspaceError::NonFinite("theta_ref"));
        }
        for (i, &s) in scale.iter().enumerate() {
            if !s.is_finite() || s <= 0.0 {
                return Err(SubspaceError::BadScale { index: i, value: s });
            }
        }
        Ok(Self {
            theta_ref,
            basis,
            scale,
        })
    }

    /// The isotropic prior `N(θ_ref, σ² I)` this project's ambient chains use.
    pub fn isotropic(theta_ref: Vec<f32>, basis: Basis, sigma: f64) -> Result<Self, SubspaceError> {
        let p = basis.p();
        Self::new(theta_ref, basis, vec![sigma as f32; p])
    }

    pub fn p(&self) -> usize {
        self.basis.p()
    }

    pub fn dim(&self) -> usize {
        self.basis.dim()
    }

    pub fn theta_ref(&self) -> &[f32] {
        &self.theta_ref
    }

    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    pub fn basis(&self) -> &Basis {
        &self.basis
    }

    /// `θ_ref + C_half B z`.  `z` is the only thing a chain stores.
    pub fn lift(&self, z: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.p()];
        self.basis.apply(z, &mut out);
        for ((o, a), s) in out.iter_mut().zip(&self.theta_ref).zip(&self.scale) {
            *o = a + s * *o;
        }
        out
    }

    /// The same point as the model's flat `[P]` parameter tensor (the shape
    /// `init_flat` produces and `forward` consumes).
    pub fn lift_tensor(&self, z: &[f32]) -> candle_core::Result<Tensor> {
        let flat = self.lift(z);
        Tensor::from_vec(flat, vec![self.p()], &Device::Cpu)
    }

    /// `out = (I − BBᵀ) v`, the component of `v` orthogonal to `span(B)`.
    ///
    /// Used to draw the complement noise of the full-target proposal: the
    /// projected noise is Gaussian on the complement with covariance the
    /// identity there, which is what makes the second stage a pCN kernel on
    /// `N(0, I_{P−r})` rather than a move that leaks back into `B`.
    pub fn complement(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(v.len(), self.p(), "complement: v must have p entries");
        assert_eq!(out.len(), self.p(), "complement: out must have p entries");
        let mut coords = vec![0.0f32; self.dim()];
        self.basis.project(v, &mut coords);
        let mut along = vec![0.0f32; self.p()];
        self.basis.apply(&coords, &mut along);
        for i in 0..self.p() {
            out[i] = v[i] - along[i];
        }
    }

    /// The chain rule `dΦ/dz = Bᵀ C_halfᵀ ∇_θ Φ`.  Getting this wrong is the
    /// classic subspace error (it silently samples a different temperature),
    /// hence `projected_gradient_matches_finite_differences`.
    pub fn gradient_to_z(&self, g_theta: &[f32]) -> Vec<f32> {
        assert_eq!(g_theta.len(), self.p(), "gradient must have p entries");
        let scaled: Vec<f32> = (0..self.p()).map(|i| self.scale[i] * g_theta[i]).collect();
        let mut out = vec![0.0f32; self.dim()];
        self.basis.project(&scaled, &mut out);
        out
    }

    /// `‖θ_ref‖` — the anchor's radius, for reports that compare a subspace
    /// draw's distance to it.
    pub fn anchor_norm(&self) -> f64 {
        self.theta_ref
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt()
    }
}

/// R1's informed basis: the top-`r` principal directions of a parameter path
/// (a fine-tuning trajectory, a set of draws, or any other `P`-vectors).
///
/// Power iteration with deflation, on the snapshot matrix directly — the `P×P`
/// covariance is never formed, and the only `P`-sized work is one multiply per
/// iteration per component.  The result is orthonormalised once more at the end
/// so `Basis::new`'s gate is a real check, not a formality.
///
/// What this is *not*: statistical evidence that the path's directions are
/// informative.  It is a definition of "informed" for a control comparison, and
/// the report has to put a random basis of the same rank against it — a PCA
/// basis that cannot beat a random one is a statement about the path, not about
/// the target.
///
/// `snapshots` are the path in visit order; only their span matters, so a
/// shuffed or repeated path gives the same basis up to sign and conditioning.
pub fn pca_basis(snapshots: &[Vec<f32>], r: usize, seed: u64) -> Result<Basis, SubspaceError> {
    let p = snapshots.first().map(Vec::len).unwrap_or(0);
    if p == 0 || r == 0 {
        return Err(SubspaceError::Empty);
    }
    if r > p {
        return Err(SubspaceError::BadRank { p, r });
    }
    if snapshots.len() < 2 {
        return Err(SubspaceError::BadLength {
            what: "snapshots (need at least 2 for a centred path)",
            expected: 2,
            got: snapshots.len(),
        });
    }
    for (i, s) in snapshots.iter().enumerate() {
        if s.len() != p {
            return Err(SubspaceError::BadLength {
                what: "snapshot",
                expected: p,
                got: s.len(),
            });
        }
        if s.iter().any(|x| !x.is_finite()) {
            return Err(SubspaceError::NonFinite("snapshots"));
        }
        let _ = i;
    }
    // Centre the path: PCA of deviations, so a constant offset (the anchor
    // itself) contributes nothing to the subspace.
    let k = snapshots.len() as f64;
    let mut mean = vec![0.0f64; p];
    for s in snapshots {
        for i in 0..p {
            mean[i] += f64::from(s[i]);
        }
    }
    for m in &mut mean {
        *m /= k;
    }
    let dev: Vec<Vec<f64>> = snapshots
        .iter()
        .map(|s| (0..p).map(|i| f64::from(s[i]) - mean[i]).collect())
        .collect();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x9ca5_ba51);
    let mut cols = vec![0.0f32; p * r];
    for j in 0..r {
        // Random start, orthogonal to the components already extracted.
        let mut v: Vec<f64> = (0..p).map(|_| gaussian(&mut rng)).collect();
        orthogonalise(&mut v, &cols, p, j);
        for _ in 0..64 {
            // w = X (Xᵀ v), then deflate and normalise.
            let proj: Vec<f64> = dev
                .iter()
                .map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let mut w = vec![0.0f64; p];
            for (row, &c) in dev.iter().zip(&proj) {
                for i in 0..p {
                    w[i] += row[i] * c;
                }
            }
            orthogonalise(&mut w, &cols, p, j);
            let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n < 1e-12 {
                return Err(SubspaceError::Degenerate { component: j });
            }
            for x in &mut w {
                *x /= n;
            }
            v = w;
        }
        for i in 0..p {
            cols[j * p + i] = v[i] as f32;
        }
    }
    Basis::new(cols, p, r)
}

/// Remove the components already stored in `cols[0..j]` from `v` (two passes,
/// for the same f32/f64 reason as `Basis::random`).
fn orthogonalise(v: &mut [f64], cols: &[f32], p: usize, j: usize) {
    for _ in 0..2 {
        for c in 0..j {
            let dot: f64 = (0..p).map(|i| v[i] * f64::from(cols[c * p + i])).sum();
            for i in 0..p {
                v[i] -= dot * f64::from(cols[c * p + i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn a_skewed_basis_is_rejected() {
        // Two identical columns: BᵀB has a 2 in the off-diagonal block.
        let a = unit(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut cols = a.clone();
        cols.extend_from_slice(&a);
        match Basis::new(cols, 5, 2) {
            Err(SubspaceError::NotOrthonormal { gram_error, .. }) => {
                assert!((gram_error - 1.0).abs() < 1e-6, "gram_error = {gram_error}");
            }
            other => panic!("expected NotOrthonormal, got {other:?}"),
        }
        // And a non-finite entry is caught before the Gram check.
        let mut bad = vec![0.0f32; 4];
        bad[0] = f32::NAN;
        assert_eq!(
            Basis::new(bad, 2, 2),
            Err(SubspaceError::NonFinite("basis"))
        );
    }

    #[test]
    fn random_basis_is_orthonormal_and_reproducible() {
        let a = Basis::random(64, 8, 7).expect("random basis");
        assert_eq!((a.p(), a.dim()), (64, 8));
        assert!(a.gram_error() < ORTHONORMAL_TOL, "{}", a.gram_error());
        let b = Basis::random(64, 8, 7).expect("random basis");
        assert_eq!(a, b, "same seed must give the same control basis");
        assert_ne!(a, Basis::random(64, 8, 8).expect("random basis"));
        // p = r is allowed (a rotation), r > p is not.
        assert!(Basis::random(4, 4, 1).is_ok());
        assert_eq!(
            Basis::random(4, 5, 1),
            Err(SubspaceError::BadRank { p: 4, r: 5 })
        );
    }

    #[test]
    fn lift_and_project_are_adjoint() {
        let p = 7;
        let r = 3;
        let basis = Basis::random(p, r, 11).expect("basis");
        let z: Vec<f32> = vec![0.3, -1.2, 0.7];
        let v: Vec<f32> = (0..p).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut bz = vec![0.0f32; p];
        basis.apply(&z, &mut bz);
        let mut btv = vec![0.0f32; r];
        basis.project(&v, &mut btv);
        let lhs: f32 = btv.iter().zip(&z).map(|(a, b)| a * b).sum();
        let rhs: f32 = bz.iter().zip(&v).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-5, "adjoint fails: {lhs} vs {rhs}");
        // BᵀB = I on z-space coordinates.
        let mut round = vec![0.0f32; r];
        basis.project(&bz, &mut round);
        for j in 0..r {
            assert!((round[j] - z[j]).abs() < 1e-5, "round trip {j}");
        }
    }

    #[test]
    fn anchor_is_z_zero_and_lift_is_affine() {
        let p = 6;
        let basis = Basis::random(p, 2, 3).expect("basis");
        let anchor: Vec<f32> = (0..p).map(|i| 0.1 * i as f32).collect();
        let sub = AffineSubspace::isotropic(anchor.clone(), basis, 0.35).expect("subspace");
        assert_eq!(sub.lift(&[0.0, 0.0]), anchor);
        let (u, v) = (vec![1.0f32, 0.5], vec![-0.25f32, 2.0]);
        let a = sub.lift(&u);
        let b = sub.lift(&v);
        let s: Vec<f32> = u.iter().zip(&v).map(|(x, y)| x + y).collect();
        let c = sub.lift(&s);
        for i in 0..p {
            assert!(
                (c[i] - (a[i] + b[i] - anchor[i])).abs() < 1e-5,
                "lift is not affine at {i}"
            );
        }
        assert!(
            (sub.anchor_norm() - anchor.iter().map(|x| x * x).sum::<f32>().sqrt() as f64).abs()
                < 1e-6
        );
    }

    #[test]
    fn projected_gradient_matches_finite_differences() {
        // f(θ) = ½‖θ‖² in the chart: d/dz_j = ⟨θ(z), C_half B e_j⟩.
        // Checked against a central difference of f(θ(z)) itself, so the test
        // fails if either the scale or the basis is applied transposed.
        let p = 5;
        let r = 2;
        let basis = Basis::random(p, r, 5).expect("basis");
        let anchor: Vec<f32> = vec![0.2, -0.4, 0.6, 0.1, -0.3];
        let sigma = 0.35f64;
        let sub = AffineSubspace::isotropic(anchor.clone(), basis, sigma).expect("subspace");
        let z = [0.4f32, -0.9];

        let theta = sub.lift(&z);
        // ∇_θ f = θ, so C_halfᵀ ∇ = σ·θ.
        let g: Vec<f32> = theta.to_vec();
        let dz = sub.gradient_to_z(&g);

        let h = 1e-4f32;
        for j in 0..r {
            let mut zp = z;
            let mut zm = z;
            zp[j] += h;
            zm[j] -= h;
            let f = |z: &[f32]| {
                let t = sub.lift(z);
                0.5 * t.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            };
            let fd = (f(&zp) - f(&zm)) / (2.0 * f64::from(h));
            assert!(
                (f64::from(dz[j]) - fd).abs() < 1e-3 * fd.abs().max(1.0),
                "coord {j}: analytic {} vs FD {fd}",
                dz[j]
            );
        }
    }

    #[test]
    fn pca_recovers_the_span_of_the_path() {
        // A path that lives in the span of two known unit vectors plus a
        // constant offset: the top-2 PCA basis must recover that span.
        let p = 9;
        let u = unit(&[1.0, 0.5, -0.3, 0.2, 0.0, 0.1, -0.7, 0.4, 0.9]);
        let v = unit(&[0.0, 1.0, 0.6, -0.5, 0.3, 0.2, 0.0, -0.4, 0.5]);
        let offset: Vec<f32> = (0..p).map(|i| 0.05 * i as f32).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let snapshots: Vec<Vec<f32>> = (0..20)
            .map(|_| {
                let (a, b) = (gaussian(&mut rng) as f32, gaussian(&mut rng) as f32);
                (0..p)
                    .map(|i| offset[i] + 3.0 * a * u[i] - 2.0 * b * v[i])
                    .collect()
            })
            .collect();
        let basis = pca_basis(&snapshots, 2, 1).expect("pca basis");
        assert!(basis.gram_error() < ORTHONORMAL_TOL);
        // Every recovered column and every generator lies in the same 2-plane:
        // check that projecting u/v through the basis reproduces them.
        for generator in [&u, &v] {
            let mut coords = vec![0.0f32; 2];
            basis.project(generator, &mut coords);
            let mut back = vec![0.0f32; p];
            basis.apply(&coords, &mut back);
            let residual: f32 = (0..p)
                .map(|i| (generator[i] - back[i]).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(residual < 1e-4, "residual {residual} — PCA missed the span");
        }
        // A rank the path does not have is an error, not a zero column.
        let flat = vec![offset.clone(); 4];
        assert_eq!(
            pca_basis(&flat, 1, 1),
            Err(SubspaceError::Degenerate { component: 0 })
        );
        // And so is a single snapshot (nothing to centre).
        assert!(matches!(
            pca_basis(&snapshots[..1], 1, 1),
            Err(SubspaceError::BadLength { .. })
        ));
    }

    #[test]
    fn complement_is_orthogonal_to_the_basis_and_keeps_a_pure_complement() {
        let p = 11;
        let r = 3;
        let basis = Basis::random(p, r, 21).expect("basis");
        let sub = AffineSubspace::isotropic(vec![0.0; p], basis, 1.0).expect("subspace");
        let v: Vec<f32> = (0..p).map(|i| (i as f32 * 0.71).cos()).collect();
        let mut perp = vec![0.0f32; p];
        sub.complement(&v, &mut perp);
        // Bᵀ(I−BBᵀ)v = 0, and projecting again changes nothing.
        let mut coords = vec![0.0f32; r];
        sub.basis().project(&perp, &mut coords);
        for (j, c) in coords.iter().enumerate() {
            assert!(c.abs() < 1e-5, "B^T v_perp[{j}] = {c}");
        }
        let mut twice = vec![0.0f32; p];
        sub.complement(&perp, &mut twice);
        for (i, t) in twice.iter().enumerate() {
            assert!(
                (t - perp[i]).abs() < 1e-6,
                "projector not idempotent at {i}"
            );
        }
        // (I−BBᵀ)(Bz + w) = w for w in the complement.
        let z = [0.3f32, -1.1, 0.4];
        let mut bz = vec![0.0f32; p];
        sub.basis().apply(&z, &mut bz);
        let mixed: Vec<f32> = bz.iter().zip(&perp).map(|(a, b)| a + b).collect();
        let mut back = vec![0.0f32; p];
        sub.complement(&mixed, &mut back);
        for i in 0..p {
            assert!(
                (back[i] - perp[i]).abs() < 1e-5,
                "complement of Bz+w at {i}"
            );
        }
    }

    #[test]
    fn bad_scales_and_lengths_are_rejected() {
        let basis = Basis::random(4, 2, 1).expect("basis");
        let err = AffineSubspace::new(vec![0.0; 4], basis.clone(), vec![1.0, 0.0, 1.0, 1.0])
            .expect_err("zero scale");
        assert_eq!(
            err,
            SubspaceError::BadScale {
                index: 1,
                value: 0.0
            }
        );
        let err = AffineSubspace::new(vec![0.0; 3], basis.clone(), vec![1.0; 4])
            .expect_err("short anchor");
        assert!(matches!(err, SubspaceError::BadLength { .. }));
        let err = AffineSubspace::new(vec![0.0; 4], basis, vec![1.0; 3]).expect_err("short scale");
        assert!(matches!(err, SubspaceError::BadLength { .. }));
    }
}
