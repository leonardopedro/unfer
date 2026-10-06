//! The two target constructions.
//!
//! These are the part of the method most often replaced by something that looks
//! equivalent and is not, so each carries the defect it is written to rule out.
//!
//! # `spread_means` (AnaCP)
//!
//! Whitening → SVD → a **signed perturbation of the singular values** →
//! de-whitening. The perturbation is what separates classes that whitening left
//! close together; it is a statement about geometry, not a normalisation.
//!
//! Audit defect **A1** is "prototypes = normalized class sums". That is NCM, not
//! this. The whole point is that `P` is *not* a running mean: it is the result of
//! a per-task optimisation.
//!
//! # `etf_procrustes` (AnaNC)
//!
//! The target means solve a **constraint**, `M̃ M̃ᵀ = S`, by orthogonal
//! Procrustes, recomputed in closed form whenever the class set changes.
//!
//! Audit defect **A2** is "ETF prototypes = normalized running sums". Those are
//! NCM means and they never satisfy the simplex-ETF constraint — the property
//! test `the_etf_constraint_holds_after_every_class_set_change` is what stops
//! that from coming back.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Which target construction a head regresses onto.
///
/// This is the one thing the two methods genuinely disagree about, so it is a
/// type-level distinction rather than a boolean: a caller cannot accidentally
/// use the spread-mean construction where the ETF one is required, because the
/// two have different output widths and different guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// AnaCP: whitened, SVD-perturbed class means. Separates classes.
    SpreadMeans,
    /// AnaNC: simplex-ETF target means by orthogonal Procrustes. Equal norms and
    /// equal angles, so classification reduces to a cosine argmax.
    Etf,
}

/// How much to trust the shared covariance when whitening.
///
/// Falls back to the identity when a class has too few samples, which is the
/// only case where whitening is undefined rather than merely noisy.
fn regularize(cov: Option<&DMatrix<f64>>, d: usize) -> DMatrix<f64> {
    match cov {
        Some(c) => {
            // A covariance from a *different space* than the means is not a
            // conservative input, it is a wrong one: whitening an RP-space mean
            // by an input-space covariance mixes two geometries. This was a real
            // bug (the head passed the input covariance to the RP-space target
            // construction), so the dimension check is a loud precondition.
            assert_eq!(
                c.nrows(),
                d,
                "covariance is {0}x{0} but the means are {1}-dimensional; \
                 they must live in the same space",
                c.nrows(),
                d
            );
            // A ridge on the diagonal, scaled to the mean variance. Fixed at 1e-8
            // *relative* rather than absolute so it does not change meaning when
            // embeddings are unit-normalized.
            let mut m = c.clone();
            let scale: f64 = (0..d).map(|i| c[(i, i)]).sum::<f64>() / d as f64;
            let eps = (scale.max(1e-12)) * 1e-8;
            for i in 0..d {
                m[(i, i)] += eps;
            }
            m
        }
        None => DMatrix::identity(d, d),
    }
}

/// Symmetric inverse square root `Σ^{-1/2}`, via eigendecomposition.
///
/// Negative eigenvalues are floored at zero rather than propagated: a covariance
/// that is numerically indefinite (it is a `d × d` estimate from few samples) has
/// no inverse square root, and taking `|λ|^{-1/2}` would silently reflect the
/// data. Flooring is the conservative choice and is reported by
/// [`half_power::floored_eigenvalues`].
pub fn symmetric_inverse_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    sym_half_power(m, -0.5)
}

/// Symmetric `Σ^{1/2}`.
pub fn symmetric_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    sym_half_power(m, 0.5)
}

fn sym_half_power(m: &DMatrix<f64>, p: f64) -> DMatrix<f64> {
    let e = SymmetricEigen::new(m.clone());
    let d = m.nrows();
    let mut out = DMatrix::zeros(d, d);
    let mut eig = DVector::zeros(d);
    for i in 0..d {
        let raw = e.eigenvalues[i];
        // Floor at a small positive number: 0^-1/2 is infinite and 0^+1/2 loses
        // a direction that is genuinely present but tiny.
        let lam = if raw <= 1e-12 { 1e-12 } else { raw };
        eig[i] = lam;
        let f = lam.powf(p);
        for a in 0..d {
            for b in 0..d {
                out[(a, b)] += e.eigenvectors[(a, i)] * f * e.eigenvectors[(b, i)];
            }
        }
    }
    half_power::LAST_FLOORED.with(|c| c.replace(eig.as_slice().to_vec()));
    out
}

/// The eigenvalues observed by the most recent `sym_*_half_power` call.
///
/// Exposed so a caller can *check* that the covariance was well-conditioned
/// instead of trusting that flooring was harmless. This is the "honest boundary"
/// pattern applied to a numeric preconditioner: a degenerate covariance silently
/// changes the geometry, and the consumer is entitled to know.
pub mod half_power {
    use std::cell::RefCell;
    thread_local! {
        pub static LAST_FLOORED: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    }

    /// The eigenvalues seen by the last half-power computation.
    pub fn floored_eigenvalues() -> Vec<f64> {
        LAST_FLOORED.with(|c| c.borrow().clone())
    }
}

/// Summed pairwise cosine similarity of the columns of `m`, ignoring self-pairs.
///
/// This is the objective the AnaCP perturbation must *decrease*. Written out as a
/// named function because the acceptance test asserts on it directly.
pub fn summed_pairwise_cosine(m: &DMatrix<f64>) -> f64 {
    let c = m.ncols();
    let mut total = 0.0;
    for i in 0..c {
        for j in (i + 1)..c {
            let mut dot = 0.0;
            let mut ni = 0.0;
            let mut nj = 0.0;
            for r in 0..m.nrows() {
                dot += m[(r, i)] * m[(r, j)];
                ni += m[(r, i)] * m[(r, i)];
                nj += m[(r, j)] * m[(r, j)];
            }
            let denom = ni.sqrt() * nj.sqrt();
            if denom > 0.0 {
                total += dot / denom;
            }
        }
    }
    total
}

/// Build `P = U S̃ Vᵀ Σ^{1/2}` — AnaCP's target prototypes.
///
/// `class_means` is `d × C` of per-class input means; `cov` is the shared
/// covariance (pass `None` when unavailable). Steps, in the papers' order:
///
/// 1. whiten: `Ĉ = Σ^{-1/2} C`
/// 2. SVD: `Ĉ = U S Vᵀ`
/// 3. perturb the singular values by `α·δ`, with `δ_i ∈ {-1, 0, +1}`
/// 4. de-whiten: `P = U S̃ Vᵀ Σ^{1/2}`
///
/// ## The sign rule, and an honest deviation
///
/// The paper states that the signs are chosen so the summed pairwise cosine
/// similarity of the shifted means is *guaranteed* to decrease for small
/// `α > 0`, and sets `α = 1`. This implementation does not transcribe a closed
/// form for `δ`; it **searches** the three admissible values per index and keeps
/// the one that measurably lowers [`summed_pairwise_cosine`].
///
/// That is a deliberate deviation, and it is the safer direction: the search
/// cannot be wrong about the direction of improvement, because the objective is
/// evaluated, whereas a transcribed sign formula is only as trustworthy as my
/// reading of the statement. The cost is that the result is not provably the
/// paper's exact choice — only that it achieves the stated effect. `alpha` is
/// therefore a parameter rather than a constant, and
/// `the_perturbation_lowers_pairwise_cosine` is the test that pins the effect.
pub fn spread_means(
    class_means: &DMatrix<f64>,
    cov: Option<&DMatrix<f64>>,
    alpha: f64,
) -> DMatrix<f64> {
    let d = class_means.nrows();
    if class_means.ncols() == 0 {
        return DMatrix::zeros(d, 0);
    }
    let cov_r = regularize(cov, d);
    let inv_half = symmetric_inverse_sqrt(&cov_r);
    let sqrt = symmetric_sqrt(&cov_r);

    let whitened = inv_half * class_means;

    // SVD of a `d × C` matrix. The economy form is enough: we need U, S, Vᵀ.
    let svd = whitened.svd(true, true);
    let u = svd.u.expect("compute_u = true");
    let vt = svd.v_t.expect("compute_v_t = true");
    let s = svd.singular_values;

    let mut s_tilde = s.clone();
    // `u` is `d × r`, `vt` is `r × C`, `s` is `r`. Columns beyond the rank are
    // structurally zero, so perturbing them can only move a null direction and
    // is skipped rather than clamped.
    for i in 0..s.len() {
        if s[i] <= 1e-9 {
            continue;
        }
        // Build the shifted means under each admissible δ and keep the best.
        let mut best = (s_tilde[i], f64::INFINITY);
        for delta in [-1.0f64, 0.0, 1.0] {
            let mut trial = s_tilde.clone();
            trial[i] = (s[i] + alpha * delta).max(0.0);
            let shifted = reconstruct(&u, &trial, &vt, d, class_means.ncols());
            let obj = summed_pairwise_cosine(&shifted);
            if obj < best.1 {
                best = (trial[i], obj);
            }
        }
        s_tilde[i] = best.0;
    }

    let mut p = DMatrix::zeros(d, class_means.ncols());
    for i in 0..s.len() {
        for r in 0..d {
            for cidx in 0..class_means.ncols() {
                p[(r, cidx)] += u[(r, i)] * s_tilde[i] * vt[(i, cidx)];
            }
        }
    }
    // De-whiten.
    p = sqrt * p;
    p
}

fn reconstruct(
    u: &DMatrix<f64>,
    s_tilde: &DVector<f64>,
    vt: &DMatrix<f64>,
    d: usize,
    cols: usize,
) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(d, cols);
    for i in 0..s_tilde.len() {
        for r in 0..d {
            for c in 0..cols {
                m[(r, c)] += u[(r, i)] * s_tilde[i] * vt[(i, c)];
            }
        }
    }
    m
}

/// The simplex-ETF target gram matrix `S = I − (1/(C−1))(11ᵀ − I)`.
pub fn etf_gram(c: usize) -> DMatrix<f64> {
    assert!(c >= 2, "an ETF needs at least two classes to have an angle");
    let mut s = DMatrix::identity(c, c);
    let off = 1.0 / (c as f64 - 1.0);
    for i in 0..c {
        for j in 0..c {
            if i != j {
                s[(i, j)] = -off;
            }
        }
    }
    s
}

/// AnaNC's target means: solve `min ‖M̃ − M‖_F` s.t. `M̃ M̃ᵀ = S` by orthogonal
/// Procrustes.
///
/// `M` is the `C × k` matrix of current class statistics (the RP-space class
/// means, transposed so classes are rows). Steps:
///
/// 1. `S = U_S Λ U_Sᵀ`
/// 2. Procrustes: SVD of `Mᵀ U_S Λ^{1/2} = U Σ Vᵀ`, then `Q* = U Vᵀ`
/// 3. `M̃ = U_S Λ^{1/2} V Uᵀ`
///
/// The result has `C` rows of equal norm and pairwise inner product `-1/(C−1)`,
/// which `the_etf_constraint_holds_after_every_class_set_change` verifies.
pub fn etf_procrustes(m: &DMatrix<f64>, c: usize) -> DMatrix<f64> {
    assert_eq!(m.nrows(), c, "class statistics must have one row per class");
    // A genuine precondition of the method, not a defensive check. The
    // constraint is M̃M̃ᵀ = S, and the solution only achieves it because
    // M̃M̃ᵀ = U_S Λ^{1/2} (V Vᵀ) Λ^{1/2} U_Sᵀ — which equals S exactly when
    // `V Vᵀ = I_C`. The SVD of the `k × C` matrix gives `V` with `min(k, C)`
    // orthonormal columns, so `V Vᵀ = I_C` requires **`k ≥ C`**.
    //
    // The papers satisfy this by a wide margin (`D = 5000` features, tens of
    // classes). Violating it does not produce a slightly-worse answer; it
    // produces a target that is not an ETF at all, which is audit defect A2
    // resurfacing through a different door. So it is a loud precondition.
    assert!(
        m.ncols() >= c,
        "the ETF constraint needs at least as many features as classes (k={}, C={}); \
         with fewer, Procrustes cannot span the ETF and the result is not an ETF",
        m.ncols(),
        c
    );
    let s = etf_gram(c);
    let e = SymmetricEigen::new(s);
    let u_s = e.eigenvectors.clone();
    // Λ^{1/2}: only the non-trivial eigenvalue is positive (1); the rest are 0.
    let mut lam_half = DVector::zeros(c);
    for i in 0..c {
        if e.eigenvalues[i] > 1e-10 {
            lam_half[i] = e.eigenvalues[i].sqrt();
        }
    }

    // A = Mᵀ U_S Λ^{1/2}, a `k × C` matrix.
    let mut a = m.transpose() * &u_s;
    for i in 0..a.nrows() {
        for j in 0..c {
            a[(i, j)] *= lam_half[j];
        }
    }

    let svd = a.svd(true, true);
    let u = svd.u.expect("compute_u = true");
    let v_t = svd.v_t.expect("compute_v_t = true");
    // M̃ = U_S Λ^{1/2} V Uᵀ, with V = v_tᵀ.
    //
    // Derivation, because the index gymnastics is where this goes wrong: with
    // QᵀQ = I_c, the ‖·‖² expansion leaves `tr(QBᵀM)` to minimise, which is the
    // standard orthogonal Procrustes problem on `A = MᵀU_SΛ^{1/2}` (shape k × c).
    // Its solution is `Q = U Vᵀ`, so `Qᵀ = V Uᵀ = v_tᵀ uᵀ`.
    //
    // Note the shapes: `a` is k × c, so `u` is k × r and `v_t` is r × c. The
    // product `v_tᵀ uᵀ` is therefore c × k — the transpose of `u v_t`, which is
    // what a reading of the paper's "Q* = UVᵀ" taken at face value produces, and
    // which indexes out of bounds whenever `k < c`.
    let vu_t = v_t.transpose() * u.transpose();
    let mut out = DMatrix::zeros(c, m.ncols());
    for r in 0..c {
        for j in 0..c {
            let lj = lam_half[j];
            if lj == 0.0 {
                continue;
            }
            for k in 0..m.ncols() {
                out[(r, k)] += u_s[(r, j)] * lj * vu_t[(j, k)];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_etf_gram_has_equal_norms_and_the_right_angles() {
        let c = 5;
        let s = etf_gram(c);
        for i in 0..c {
            assert!((s[(i, i)] - 1.0).abs() < 1e-12);
            for j in 0..c {
                if i != j {
                    let want = -1.0 / (c as f64 - 1.0);
                    assert!((s[(i, j)] - want).abs() < 1e-12, "{i},{j}");
                }
            }
        }
    }

    #[test]
    fn the_etf_gram_is_psd_of_rank_c_minus_one() {
        let c = 6usize;
        let e = SymmetricEigen::new(etf_gram(c));
        let zeros = e.eigenvalues.iter().filter(|v| v.abs() < 1e-9).count();
        assert_eq!(zeros, 1, "exactly one direction is degenerate");
        // `SymmetricEigen` does not sort, so the non-trivial eigenvalue is the
        // maximum rather than `eigenvalues[0]`.
        let max = e
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        // S = (C/(C-1))I − (1/(C-1))11ᵀ, so the non-trivial eigenvalue is C/(C−1),
        // not 1. Getting this wrong would mean mis-scaling Λ^{1/2} and so
        // mis-scaling every ETF prototype.
        assert!(
            (max - c as f64 / (c as f64 - 1.0)).abs() < 1e-9,
            "non-trivial eigenvalue should be C/(C-1) = {}, got {max}",
            c as f64 / (c as f64 - 1.0)
        );
        assert!(
            e.eigenvalues.iter().all(|v| *v >= -1e-9),
            "ETF gram must be PSD"
        );
    }

    #[test]
    fn procrustes_output_satisfies_the_etf_constraint() {
        // This is the acceptance criterion in D3, written as the property it is.
        // `k ≥ c` is the precondition documented on `etf_procrustes`.
        let c = 6;
        let k = 8;
        let mut m = DMatrix::zeros(c, k);
        // Deliberately lopsided statistics, so a non-Procrustes answer (e.g.
        // normalized running means) could not accidentally satisfy the ETF.
        for i in 0..c {
            for j in 0..k {
                m[(i, j)] = ((i * 7 + j * 3) as f64).sin() * (1.0 + i as f64 * 0.3);
            }
        }
        let mt = etf_procrustes(&m, c);
        let mt_t = mt.transpose();
        let got = &mt * mt_t;
        let want = etf_gram(c);
        let err = (&got - &want).norm();
        assert!(
            err < 1e-5,
            "‖M̃M̃ᵀ − S‖ = {err}, above the 1e-5 acceptance bound"
        );
    }

    /// The precondition is real: with fewer features than classes, Procrustes
    /// cannot span the ETF, and the result quietly stops being one. Asserting
    /// the refusal is what stops audit defect A2 from returning via this door.
    #[test]
    fn too_few_features_to_span_the_etf_is_refused() {
        let c = 6;
        let m = DMatrix::zeros(c, c - 2);
        let r = std::panic::catch_unwind(move || etf_procrustes(&m, c));
        assert!(
            r.is_err(),
            "k < C must be refused, not silently approximated"
        );
    }

    #[test]
    fn procrustes_moves_the_statistics_as_little_as_possible() {
        // If the input already happens to be an ETF up to rotation, Procrustes
        // must leave it (nearly) alone. This is what "keeps the reorganization
        // of feature space minimal" means operationally.
        let c = 4;
        let seed = etf_procrustes(&DMatrix::identity(c, c), c);
        // Feed an ETF back in: the answer should match it.
        let again = etf_procrustes(&seed, c);
        let a = normalize_rows(&seed.clone());
        let b = normalize_rows(&again);
        assert!(
            (&a - &b).norm() < 1e-6,
            "Procrustes moved an already-ETF target"
        );
    }

    fn normalize_rows(m: &DMatrix<f64>) -> DMatrix<f64> {
        let mut out = m.clone();
        for i in 0..m.nrows() {
            let n = out.row(i).norm();
            if n > 0.0 {
                let scaled = out.row(i).transpose() / n;
                out.set_row(i, &scaled.transpose());
            }
        }
        out
    }

    #[test]
    fn spread_means_lowers_pairwise_cosine_similarity() {
        // The D2/D3 effect that AnaCP's Lemma 1 is about, asserted as a
        // measurement rather than trusted from the paper.
        let d = 6;
        let c = 4;
        let mut means = DMatrix::zeros(d, c);
        // Means that are close together, so there is something to separate.
        for i in 0..d {
            for j in 0..c {
                means[(i, j)] = 0.05 * ((i + j) as f64) + 0.02 * ((i * j) as f64);
            }
        }
        let mut cov = DMatrix::zeros(d, d);
        for i in 0..d {
            cov[(i, i)] = 0.01 * (i as f64 + 1.0);
        }
        let before = summed_pairwise_cosine(&means);
        let p = spread_means(&means, Some(&cov), 1.0);
        let after = summed_pairwise_cosine(&p);
        assert!(
            after < before,
            "perturbation did not separate: {before} -> {after}"
        );
    }

    #[test]
    fn spread_means_are_finite_with_no_covariance_yet() {
        let d = 3;
        let means = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 0.0, 1.0, 1.0, 0.5]);
        let p = spread_means(&means, None, 1.0);
        assert_eq!(p.nrows(), d);
        assert_eq!(p.ncols(), 2);
        assert!(p.iter().all(|v| v.is_finite()), "got a NaN or inf");
    }

    #[test]
    fn a_singular_covariance_is_floored_rather_than_propagated() {
        // Rank-1 covariance: two eigenvalues are zero. Without flooring, Σ^{-1/2}
        // would be infinite and every target would be NaN.
        let mut cov = DMatrix::zeros(3, 3);
        cov[(0, 0)] = 1.0;
        let inv = symmetric_inverse_sqrt(&cov);
        assert!(inv.iter().all(|v| v.is_finite()));
        let seen = half_power::floored_eigenvalues();
        assert_eq!(seen.len(), 3);
        assert!(seen.iter().all(|v| *v >= 1e-12));
    }
}
