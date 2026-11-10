//! Certified Ritz values and the certified cross-sector mass gap.
//!
//! This is the **library surface** the mass-gap certificate plan consumes
//! (`CONSOLIDATED_PLAN.md` §13 / `MASS_GAP_CERTIFIED.md` §4–§5): the kernel
//! delivers, per Ritz value, a rigorous enclosure
//!
//! ```text
//!   [θ − δ, θ + δ],   δ = ‖r‖_cert + c(n)·u·‖Ĝ‖ + h_O ,      (§4.4)
//! ```
//!
//! assembled from the three explicit, machine-checkable terms of
//! `MASS_GAP_CERTIFIED.md` §4:
//!
//! * `residual`  — the a-posteriori Rayleigh–Ritz residual `‖Hψ − θψ‖`
//!   (T2 / §4.3), computed cancellation-free from the stored Gram
//!   ([`ForwardSirkResult::ritz_abs_residuals`]; the one out-of-basis
//!   component `|τ_m c_{m−1}|`);
//! * `roundoff`  — the Hermitian-eigendecomposition backward error
//!   `c(n)·u·‖Ĝ‖` (T1/T3 / §4.1), `n` = Krylov rank, `u = 2⁻⁵³`, `c(n) = n³`
//!   conservatively, `‖Ĝ‖` the spectral norm of the matrix actually
//!   diagonalized (the whitened reduced Hamiltonian `h_proj`);
//! * `enclosure` — the interval half-width `h_O` of the measured value under
//!   directed rounding (T5 / §4.4); for a plain Ritz value it is a few ulps
//!   of `|θ|`, for an observable it is the certified interval of
//!   `⟨ψ|O|ψ⟩`.
//!
//! The f64 arithmetic itself **never enters the statement**: the certificate
//! is a theorem about the *exact* operator applied to *enclosures*.  The
//! cross-sector assembly ([`certified_mass_gap`]) is the T6 gap statement of
//! §3.4: with `θᵒ₀, θᵉ₀` the computed lowest Ritz values of the odd/even
//! sectors of `H_m` and widths `δᵒ, δᵉ`,
//!
//! ```text
//!   λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ),                 (T6)
//! ```
//!
//! **conditional on `θᵒ₀` lying within `δᵒ` of the odd sector's ground** — a
//! hypothesis that no term of the certificate witnesses (2026-11-10 audit:
//! a residual measures "converged", not "lowest"). The even side needs no
//! hypothesis, Rayleigh–Ritz giving `λ₀ ≤ θᵉ₀` outright. Treat the assembly
//! as an estimate with explicit widths; [`GapCertificate::lo`] carries the
//! counterexample and `mass_gap_spec` the theory.
//!
//! and the emitter ([`emit_gap_certificate_ndjson`]) serializes the
//! certificates (parity labels, θ, δ) as the data consumed by the Lean4
//! T6 instance (the `BookProof/ChapterSirkCertifiedGap` target) and
//! re-verified by `prob_kernel::verify::verify_export` (nanoda).

use nalgebra::DMatrix;
use nested_fock_algebra::{Hamiltonian, QuantumState};
use num_complex::Complex64;
use std::collections::BTreeMap;

use crate::forward_sirk::ForwardSirkResult;

/// Double-precision unit roundoff `u = 2⁻⁵³`.
pub const UNIT_ROUNDOFF: f64 = 1.1102230246251565e-16;

/// Conservative backward-error constant `c(n) = n³` (§4.1; LAPACK-lineage
/// bounds, machine-checkable).
pub fn backward_error_const(n: usize) -> f64 {
    (n as f64).powi(3)
}

/// Spectral norm of the whitened reduced Hamiltonian `h_proj` (Hermitian, so
/// the operator norm is `max |eigenvalue|` — a bound on `‖Ĝ‖` in §4.1).
pub fn h_proj_spectral_norm(h_proj: &DMatrix<Complex64>) -> f64 {
    let eig = h_proj.clone().symmetric_eigen();
    eig.eigenvalues
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
}

/// One certified Ritz value: the computed value plus a rigorous enclosure
/// `[value − δ, value + δ]`, `δ = residual + roundoff + enclosure`.
#[derive(Debug, Clone, Copy)]
pub struct Certificate {
    /// The computed Ritz value `θ`.
    pub value: f64,
    /// A-posteriori residual `‖Hψ − θψ‖` (T2, §4.3).
    pub residual: f64,
    /// Eigendecomposition backward error `c(n)·u·‖Ĝ‖` (T1/T3, §4.1).
    pub roundoff: f64,
    /// Interval half-width `h_O` of the measured value (T5, §4.4).
    pub enclosure: f64,
    /// Certified lower edge `value − δ`.
    pub lo: f64,
    /// Certified upper edge `value + δ`.
    pub hi: f64,
}

impl Certificate {
    /// Assemble a certificate from the three §4.4 terms.
    pub fn new(value: f64, residual: f64, roundoff: f64, enclosure: f64) -> Self {
        let delta = residual + roundoff + enclosure;
        Self {
            value,
            residual,
            roundoff,
            enclosure,
            lo: value - delta,
            hi: value + delta,
        }
    }

    /// The certified width `δ = residual + roundoff + enclosure`.
    pub fn delta(&self) -> f64 {
        self.residual + self.roundoff + self.enclosure
    }

    /// Does the certified interval contain `x`?
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }
}

/// Certified Ritz values of one solve: every pair `(θ, ‖Hψ − θψ‖)` from
/// [`ForwardSirkResult::ritz_abs_residuals`] gets the roundoff term
/// `c(n)·u·‖Ĝ‖` (`n` = Krylov rank, `‖Ĝ‖` = spectral norm of `h_proj`) and a
/// directed-rounding enclosure `h_O = 4·u·max(|θ|, 1)` (two ulps for the
/// value, two for the interval arithmetic around it).
pub fn certified_ritz_values(res: &ForwardSirkResult) -> Vec<Certificate> {
    let n = res.h_proj.nrows();
    let g_norm = h_proj_spectral_norm(&res.h_proj);
    let roundoff = backward_error_const(n) * UNIT_ROUNDOFF * g_norm.max(1.0);
    res.ritz_abs_residuals()
        .into_iter()
        .map(|(theta, res_abs)| {
            let enclosure = 4.0 * UNIT_ROUNDOFF * theta.abs().max(1.0);
            Certificate::new(theta, res_abs, roundoff, enclosure)
        })
        .collect()
}

/// The lowest certified Ritz value of a solve (the sector ground state).
pub fn certified_ground_state(res: &ForwardSirkResult) -> Option<Certificate> {
    certified_ritz_values(res).into_iter().next()
}

/// The certified cross-sector mass gap (T6, §3.4).
///
/// With `θᵒ₀, θᵉ₀` the lowest certified Ritz values of the odd/even solves,
/// the T6 statement is
///
/// ```text
///   λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ) =: lo,
/// ```
///
/// **Conditional on the sector grounds being resolved** (2026-11-10 audit).
/// The even side needs no hypothesis — Rayleigh–Ritz gives `λ₀ ≤ θᵉ₀`. The
/// odd side does: `δᵒ` bounds the distance from `θᵒ₀` to *some* eigenvalue of
/// the odd sector (Parlett), and nothing in the certificate says which. A
/// Krylov start orthogonal to the odd ground keeps its chain orthogonal to
/// it, so the solve converges to the odd *second* level with `δᵒ → 0`; the
/// assembly then returns `lo` above the true gap — measured `+0.526380`
/// against `0.189928` at `g = 0`, `N ≤ 6`, `m = 14`, overlap `7.5e-16`
/// (`tests/qcd_mass_gap_certified.rs`,
/// `the_certified_gap_lower_bound_needs_the_sector_ground`). So `lo > 0` is
/// not a proof of a mass gap; it is the estimate "this solve's low window
/// admits a positive signed difference, with widths", and a positive value
/// is an alarm to check sector-ground resolution, not the witness for it.
/// The analytic strong-coupling value `g²/2` may sit outside `[lo, hi]` by
/// the `O(g⁴)` magnetic correction — that deviation is deliberately *not*
/// part of the certificate.
pub struct GapCertificate {
    /// Certified interval for the even-sector ground state.
    pub even: Certificate,
    /// Certified interval for the odd-sector ground state.
    pub odd: Certificate,
    /// Computed gap `θᵒ₀ − θᵉ₀`.
    pub gap: f64,
    /// The T6 lower-bound expression `θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)`. A genuine
    /// lower bound on `λ₁(H_m) − λ₀(H_m)` only when the odd solve resolved the
    /// odd sector ground; otherwise an estimate that can exceed the gap.
    pub lo: f64,
    /// The T6 upper-bound expression `θᵒ₀ − θᵉ₀ + (δᵒ + δᵉ)`, carrying the
    /// same conditional reading as [`GapCertificate::lo`].
    pub hi: f64,
}

impl GapCertificate {
    /// Does the certified interval `[lo, hi]` contain the measured gap `θᵒ₀ − θᵉ₀`?
    ///
    /// **This is true by construction** and is not a check: `lo = gap − δ`,
    /// `hi = gap + δ` with `δ = δᵒ + δᵉ ≥ 0`, so the containment follows from
    /// the arithmetic, not from the solve. A suite that asserts it has asserted
    /// the constructor. What *is* worth asserting at a gap-assembly site:
    ///
    /// * the assembly identity `hi − lo = δᵒ + δᵉ` and `gap = θᵒ₀ − θᵉ₀`
    ///   (catches a dropped term, a swapped sector or a sign slip);
    /// * the T6 statement itself, `lo > 0`;
    /// * resolution — `δᵒ + δᵉ` small against the gap — or, when the widths
    ///   exceed the gap (the measured situation at `m = 12` on the gauge-fixed
    ///   QYM, `δ = 7.438` against `0.0912`), an explicit pin that `lo ≤ 0`, so
    ///   an enclosure is never read as a certified mass gap.
    pub fn contains_measured(&self) -> bool {
        self.lo <= self.gap && self.gap <= self.hi
    }
}

/// One cluster of a certified spectrum: a closed interval that is proven to
/// contain exactly `multiplicity` eigenvalues of the matrix (counted with
/// multiplicity) and no others.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumCluster {
    pub lo: f64,
    pub hi: f64,
    pub multiplicity: usize,
}

/// A certified spectrum of an explicit finite Hermitian matrix — the
/// **exclusion witness** [`crate::mass_gap_spec::temple_lower_bound`] needs.
///
/// Parlett's bound says every computed eigenpair encloses *some* eigenvalue;
/// that alone cannot name an eigenvalue. What completes the argument is
/// counting: an `n × n` Hermitian matrix has `n` eigenvalues (with
/// multiplicity), each of the `n` Parlett intervals contains at least one, and
/// if the intervals are grouped into **disjoint** clusters then each cluster
/// contains exactly as many eigenvalues as it has members — otherwise the total
/// would exceed `n`. So a cluster's lower edge is a rigorous lower bound on the
/// eigenvalue(s) it encloses, and the second cluster's lower edge is a rigorous
/// bound on `λ₂`.
///
/// This is what a residual cannot do on its own, and it is only available for an
/// explicit finite matrix — i.e. for a truncated operator, never for the
/// unbounded `H`.
pub struct SpectrumCertificate {
    /// Clusters, ascending, pairwise disjoint by construction.
    pub clusters: Vec<SpectrumCluster>,
    /// Dimension of the matrix, `Σ multiplicity`.
    pub dimension: usize,
}

impl SpectrumCertificate {
    /// Rigorous lower bound for the `k`-th eigenvalue (1-indexed, counting
    /// multiplicity), or `None` if `k` exceeds the dimension.
    pub fn eigenvalue_lower_bound(&self, k: usize) -> Option<f64> {
        let mut seen = 0;
        for cluster in &self.clusters {
            seen += cluster.multiplicity;
            if k <= seen {
                return Some(cluster.lo);
            }
        }
        None
    }

    /// Rigorous lower bound for the second eigenvalue `λ₂` — the exclusion
    /// witness Temple's inequality needs. `None` when the ground is degenerate
    /// (then `λ₂ = λ₁` and no `β` above `θ` can exist), or when the matrix has
    /// fewer than two eigenvalues.
    pub fn second_eigenvalue_lower_bound(&self) -> Option<f64> {
        let ground = self.clusters.first()?;
        if ground.multiplicity != 1 {
            return None;
        }
        let second = self.clusters.get(1)?;
        Some(second.lo)
    }
}

/// Certify the whole spectrum of an explicit `n × n` Hermitian matrix.
///
/// Each computed eigenpair contributes the Parlett interval
/// `[θ − δ, θ + δ]`, `δ = ‖Av − θv‖ + c(n)·u·‖A‖ + 4u·max(|θ|, 1)`; intervals
/// that touch are merged into one cluster, so the returned clusters are
/// disjoint and each is proven to contain exactly its own multiplicities (see
/// [`SpectrumCertificate`]). Returns `None` for an empty matrix or a
/// degenerate/complex spectrum (`symmetric_eigen` on a non-Hermitian matrix).
pub fn certify_spectrum(matrix: &DMatrix<f64>) -> Option<SpectrumCertificate> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n {
        return None;
    }
    let eig = matrix.clone().symmetric_eigen();
    if !eig.eigenvalues.iter().all(|v| v.is_finite()) {
        return None;
    }
    let norm = eig
        .eigenvalues
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let roundoff = backward_error_const(n) * UNIT_ROUNDOFF * norm.max(1.0);
    let mut intervals: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let theta = eig.eigenvalues[i];
        let v = eig.eigenvectors.column(i);
        let residual = {
            let mut acc = 0.0_f64;
            for j in 0..n {
                let row_times_v: f64 = (0..n).map(|k| matrix[(j, k)] * v[k]).sum();
                let r = row_times_v - theta * v[j];
                acc += r * r;
            }
            acc.sqrt()
        };
        let enclosure = 4.0 * UNIT_ROUNDOFF * theta.abs().max(1.0);
        let delta = residual + roundoff + enclosure;
        intervals.push((theta - delta, theta + delta));
    }
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut clusters: Vec<SpectrumCluster> = Vec::with_capacity(n);
    for (lo, hi) in intervals {
        match clusters.last_mut() {
            Some(last) if lo <= last.hi => {
                last.hi = last.hi.max(hi);
                last.multiplicity += 1;
            }
            _ => clusters.push(SpectrumCluster {
                lo,
                hi,
                multiplicity: 1,
            }),
        }
    }
    debug_assert_eq!(
        clusters.iter().map(|c| c.multiplicity).sum::<usize>(),
        n,
        "pigeonhole: the clusters must account for every eigenvalue"
    );
    Some(SpectrumCertificate {
        clusters,
        dimension: n,
    })
}

// ── Sparse exclusion: a λ₂ lower bound without dense diagonalization ───────

/// A **sound** lower bound on `λ₂` computed from matvecs only — the sparse
/// counterpart of [`SpectrumCertificate::second_eigenvalue_lower_bound`].
///
/// The dense witness counts eigenvalues; a matvec-only method cannot count, so
/// the count has to come from structure. The structure used here is the
/// diagonal split: write `H = D + V` with `D` the diagonal, `V` the rest. Weyl's
/// inequality in the form `λ_i(H) ≥ λ_i(D) − ‖V‖₂` turns the count into a
/// diagonal comparison:
///
/// ```text
///   #{λ(H) < β} ≤ #{D_ii < β + ‖V‖₂},
/// ```
///
/// so `β = d₂ − ‖V‖₂` (with `d₂` the second-smallest diagonal entry) excludes
/// everything but one eigenvalue below `β`, i.e. `λ₂(H) ≥ β`. The norm is
/// bounded by the Frobenius norm of the off-diagonal part, which is exact and
/// cheap: `‖H‖_F² = Σ_i ‖H e_i‖²` and `‖D‖_F² = Σ_i H_ii²` give
/// `‖V‖_F² = ‖H‖_F² − ‖D‖_F²`, so `n` matvecs suffice for any `n`.
///
/// **This is honest but weak, and measured so** (`docs/MASS_GAP_SPEC.md` §6d):
/// `‖V‖_F` grows like `√n` while a spectral gap does not, so the bound collapses
/// as the truncation deepens. It is the price of the count — see
/// [`lanczos_ritz`] for why matvecs alone cannot do better.
pub struct DiagonalExclusion {
    /// The diagonal `H_ii`, ascending.
    pub diagonal: Vec<f64>,
    /// `‖H − diag(H)‖_F`, rounded up by its accumulated summation error.
    pub offdiagonal_frobenius: f64,
}

impl DiagonalExclusion {
    /// Measure the split from matvecs alone. `matvec` applies `H` to a dense
    /// vector of length `n`; `e_i ↦ H e_i` is applied once per basis element.
    pub fn from_matvec(matvec: &dyn Fn(&[f64]) -> Vec<f64>, n: usize) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut diagonal = Vec::with_capacity(n);
        let mut frobenius_sq = 0.0_f64;
        let mut basis = vec![0.0_f64; n];
        for i in 0..n {
            basis[i] = 1.0;
            let column = matvec(&basis);
            basis[i] = 0.0;
            if column.len() != n || !column.iter().all(|x| x.is_finite()) {
                return None;
            }
            diagonal.push(column[i]);
            frobenius_sq += column.iter().map(|x| x * x).sum::<f64>();
        }
        let diagonal_sq: f64 = diagonal.iter().map(|x| x * x).sum();
        // `frobenius_sq` is a sum of `n²` non-negative products, so its error is
        // bounded by `n·u·frobenius_sq`; round the difference up, never down.
        let slack = (n as f64) * UNIT_ROUNDOFF * frobenius_sq;
        let offdiagonal_sq = (frobenius_sq - diagonal_sq + slack).max(0.0);
        diagonal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(Self {
            diagonal,
            offdiagonal_frobenius: offdiagonal_sq.sqrt(),
        })
    }

    /// Rigorous `β ≤ λ₂` — `None` when the matrix has fewer than two rows.
    ///
    /// The diagonal's smallest entry may be repeated: Weyl holds with
    /// multiplicity, so `β` stays valid, and it is [`crate::mass_gap_spec::temple_lower_bound`]
    /// that refuses a `β` which does not clear `θ`.
    pub fn second_eigenvalue_lower_bound(&self) -> Option<f64> {
        self.diagonal
            .get(1)
            .map(|d2| d2 - self.offdiagonal_frobenius)
    }
}

// ── The matching split: the same count, with slack that does not grow in n ──

/// A `λ₂` lower bound from a **matching decomposition** of the sparsity graph —
/// the tightest count-free witness this crate can build from matvecs.
///
/// [`DiagonalExclusion`] throws the whole off-diagonal part into a single norm
/// term, `‖V‖_F ≈ √n · (entry scale)`, so its slack grows with the truncation.
/// Colouring the sparsity graph removes that growth. The off-diagonal nonzeros
/// are partitioned into **matchings** `V = Σ_c V_c` — classes in which no row or
/// column carries two entries, which greedy edge colouring always achieves with
/// `C ≤ max_row_degree` classes. Each `V_c` is then a disjoint union of `2×2`
/// blocks `[[0, v], [v, 0]]`, so its smallest eigenvalue is known in closed
/// form: `λ_min(V_c) = −max |v|` over the class. Weyl's inequality applied class
/// by class, `λ_i(A + B) ≥ λ_i(A) + λ_min(B)`, gives
///
/// ```text
///   λ₂(H) ≥ d₂ − Σ_c max_{e ∈ c} |H_e|,      d₂ = second-smallest diagonal entry.
/// ```
///
/// **Measured: a change in growth, not in magnitude, and still not a witness**
/// (`docs/MASS_GAP_SPEC.md` §6d). The split trades the Frobenius norm's
/// `√(#edges)` for `#classes`, which grows like the graph's degree:
///
/// | `N ≤` | classes | colouring slack | `‖V‖_F` |
/// |---|---|---|---|
/// | 6 | 13 | 42.7 | 34.5 |
/// | 8 | 17 | 92.6 | 86.4 |
/// | 12 | 28 | 313.5 | 553.8 |
///
/// So it is *looser* than one Frobenius term at the small sizes — charging the
/// largest entry of each class beats charging the `l2` norm only once the graph
/// is dense enough — and 1.8× tighter by `N ≤ 12`. Neither is remotely inside
/// the window the certificate needs: `d₂ = 1` on this operator while
/// `θ ≈ −2.8 … −1.9`, so the entire off-diagonal budget is `d₂ − θ < 4`, i.e. a
/// Weyl split would have to charge less than four unit entries in total — while
/// any matching decomposition charges at least one entry per class, and the
/// degree is 13–28. That is why no splitting bound can be a witness here, no
/// matter how the split is chosen: the count is the real content of the
/// witness, and [`inertia_below`] is the way to get it without paying
/// `O(n³)`.
pub struct MatchingExclusion {
    /// The diagonal `H_ii`, ascending.
    pub diagonal: Vec<f64>,
    /// One entry per colour class: `max |H_e|` over that class's edges, i.e.
    /// `−λ_min(V_c)` rounded up.
    pub class_widths: Vec<f64>,
    /// `‖H − diag(H)‖_F`, kept only for the comparison with the diagonal split.
    pub offdiagonal_frobenius: f64,
}

impl MatchingExclusion {
    /// Measure the split from matvecs alone (`e_i ↦ H e_i`, once per basis
    /// element). Returns `None` for an empty matrix, a column of the wrong
    /// length, or a non-finite entry.
    pub fn from_matvec(matvec: &dyn Fn(&[f64]) -> Vec<f64>, n: usize) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut diagonal = Vec::with_capacity(n);
        let mut frobenius_sq = 0.0_f64;
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        let mut basis = vec![0.0_f64; n];
        for j in 0..n {
            basis[j] = 1.0;
            let column = matvec(&basis);
            basis[j] = 0.0;
            if column.len() != n || !column.iter().all(|x| x.is_finite()) {
                return None;
            }
            diagonal.push(column[j]);
            frobenius_sq += column.iter().map(|x| x * x).sum::<f64>();
            for (i, &v) in column.iter().enumerate() {
                // One entry per unordered pair: a symmetric matrix would
                // otherwise colour every edge twice, which is harmless for
                // soundness but would halve the classes' widths for free.
                if i < j && v != 0.0 {
                    edges.push((i, j, v));
                }
            }
        }

        // Greedy edge colouring: the smallest colour unused at both endpoints.
        // Because `V_c` must be a matching, sharing a *vertex* is what forbids a
        // colour — sharing nothing else does.
        edges.sort_by_key(|&(i, j, _)| (i, j));
        let mut used: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut class_of: Vec<u32> = Vec::with_capacity(edges.len());
        let mut class_count = 0usize;
        for &(i, j, _) in edges.iter() {
            let mut colour = 0usize;
            while used[i].contains(&(colour as u32)) || used[j].contains(&(colour as u32)) {
                colour += 1;
            }
            used[i].push(colour as u32);
            used[j].push(colour as u32);
            class_count = class_count.max(colour + 1);
            class_of.push(colour as u32);
        }
        // Each class width is rounded *up*, so the Weyl sum below is a bound
        // rather than an estimate.
        let mut class_widths = vec![0.0_f64; class_count];
        for (e, &(_, _, v)) in edges.iter().enumerate() {
            let c = class_of[e] as usize;
            class_widths[c] = class_widths[c].max(v.abs() * (1.0 + 4.0 * UNIT_ROUNDOFF));
        }

        let diagonal_sq: f64 = diagonal.iter().map(|x| x * x).sum();
        let slack = (n as f64) * UNIT_ROUNDOFF * frobenius_sq;
        let offdiagonal_sq = (frobenius_sq - diagonal_sq + slack).max(0.0);
        diagonal.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(Self {
            diagonal,
            class_widths,
            offdiagonal_frobenius: offdiagonal_sq.sqrt(),
        })
    }

    /// Number of classes the colouring needed — the measured degree-like slack
    /// driver, and `None`-free: an edgeless off-diagonal part has zero classes.
    pub fn class_count(&self) -> usize {
        self.class_widths.len()
    }

    /// Rigorous `β ≤ λ₂` — `None` when the matrix has fewer than two rows.
    ///
    /// The diagonal's smallest entry may be repeated; Weyl holds with
    /// multiplicity, so `β` stays valid and it is
    /// [`crate::mass_gap_spec::temple_lower_bound`] that refuses a `β` which
    /// does not clear `θ`.
    pub fn second_eigenvalue_lower_bound(&self) -> Option<f64> {
        let d2 = *self.diagonal.get(1)?;
        let width: f64 = self.class_widths.iter().sum();
        // `Σ` of at most `n` non-negative terms: round the total up too.
        let rounding = (self.class_widths.len() as f64 + 1.0) * UNIT_ROUNDOFF * width.max(1.0);
        Some(d2 - width - rounding)
    }
}

// ── Inertia counting: the count without a diagonalization ─────────────────

/// One floating-point quantity with a rigorous bound on its distance from the
/// exact value. Every elimination step below carries these, so a pivot's sign
/// is decided only when it cannot be mistaken.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounded {
    value: f64,
    /// Bound on `|value − exact|`.
    error: f64,
}

impl Bounded {
    fn exact(value: f64) -> Self {
        Self { value, error: 0.0 }
    }

    /// Rounded result of `a − b`, with the rounding folded into the error.
    fn sub(self, other: Self) -> Self {
        let value = self.value - other.value;
        Self {
            value,
            error: self.error + other.error + UNIT_ROUNDOFF * value.abs(),
        }
    }

    fn mul(self, other: Self) -> Self {
        let value = self.value * other.value;
        Self {
            value,
            error: (self.value.abs() + self.error) * other.error
                + other.value.abs() * self.error
                + UNIT_ROUNDOFF * value.abs(),
        }
    }

    /// `None` when the divisor's interval contains zero — the quotient's sign
    /// could then be anything, and the caller must refuse rather than decide.
    fn div(self, other: Self) -> Option<Self> {
        let denominator = other.value.abs() - other.error;
        if !(denominator > 0.0) {
            return None;
        }
        let value = self.value / other.value;
        let ratio = (self.value.abs() + self.error) / denominator;
        Some(Self {
            value,
            error: UNIT_ROUNDOFF * value.abs() + (self.error + ratio * other.error) / denominator,
        })
    }

    /// `Some(sign)` only when the sign is certain, `None` when the interval
    /// straddles zero.
    fn certain_sign(&self) -> Option<i8> {
        if !self.value.is_finite() || !self.error.is_finite() {
            return None;
        }
        if self.value.abs() > self.error {
            Some(if self.value < 0.0 { -1 } else { 1 })
        } else {
            None
        }
    }
}

/// The inertia of `H − βI`, counted by a sparse symmetric elimination — the
/// `λ₂` witness that does not need a diagonalization.
///
/// `certify_spectrum` gets its count from completeness: `n` Parlett intervals,
/// `n` eigenvalues, disjoint clusters. This gets the same count from
/// **Sylvester's law of inertia**: `H − βI = L D Lᵀ` with `L` unit triangular,
/// so the number of negative pivots of `D` is exactly the number of eigenvalues
/// of `H` below `β`. What makes it rigorous in floating point is the error
/// carried alongside every entry: a pivot's sign is used only when
/// `|value| > error`, and a pivot whose interval contains zero aborts the count
/// (`None`) instead of guessing. The elimination is sparse and one-shot, so the
/// cost is `O(fill)` rather than `O(n³)`.
///
/// Fills in, like any elimination without a good ordering, so the ordering is a
/// parameter ([`Ordering`]) and the fill it costs is reported back. Pivoting is
/// 1×1 with a symmetric exchange (a permutation preserves inertia): a zero or
/// near-zero diagonal is swapped for a larger one instead of aborting, and a
/// matrix with no decidable 1×1 pivot at all (a `[[0,1],[1,0]]` block) is refused
/// rather than guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InertiaCount {
    /// Number of eigenvalues of `H` strictly below `β`.
    pub below: usize,
    /// Pivots eliminated (always the matrix dimension when the count succeeded).
    pub pivots: usize,
    /// Size of the largest row map during the elimination — the measured fill.
    pub fill: usize,
    /// Symmetric pivots exchanged to keep the elimination manageable.
    pub swaps: usize,
    /// The ordering the elimination actually ran under.
    pub ordering: Ordering,
}

/// The fill-reducing ordering applied before the elimination.
///
/// Both are graph orderings of the same matrix — the inertia is identical under
/// either, which is what makes them comparable by fill and time alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ordering {
    /// Reverse Cuthill–McKee: breadth-first numbering, then reversed. Cheap to
    /// compute and bandwidth-reducing, but it is a *band* ordering: the fill it
    /// leaves grows with the square root of the separator width.
    #[default]
    ReverseCuthillMckee,
    /// Nested dissection by recursive level-set bisection: split the graph at a
    /// BFS level, order both halves, eliminate the level last. The separators are
    /// small and disappear last, which is the ordering fill is theoretically
    /// cheapest under.
    NestedDissection,
}

impl InertiaCount {
    /// Rigorous `β ≤ λ₂` exactly when the count says "exactly one eigenvalue
    /// below `β`". `None` otherwise: zero eigenvalues below means the Ritz
    /// value is not the ground, two or more means `β` is above `λ₂`.
    pub fn second_eigenvalue_lower_bound(&self, beta: f64) -> Option<f64> {
        (self.below == 1).then_some(beta)
    }
}

/// Count the eigenvalues of `H` below `beta` by eliminating the sparse
/// `H − βI`, or return `None` when any pivot's sign cannot be decided.
///
/// `columns[j]` lists the nonzeros of column `j` of the Hermitian matrix, as
/// `(row, value)` pairs. Callers pass a genuinely Hermitian matrix (these are
/// built that way); an asymmetric input would be silently symmetrized rather
/// than rejected.
///
/// **Measured** (`docs/MASS_GAP_SPEC.md` §6d): on the truncated gauge-fixed QYM
/// at `g = 1` this decides the count of every sector up to `N ≤ 16`
/// (`n = 2160`) in under a second, and the resulting gap bracket agrees with the
/// exact spectrum to the last printed digit wherever the dense route can reach
/// (`N ≤ 10`). Two failure modes are real and are reported instead of hidden: a
/// shift `β` within `~1e-9` of an eigenvalue leaves one pivot's sign undecidable
/// (so callers should probe a ladder of margins), and an elimination stage whose
/// running error bound exceeds every remaining pivot is refused (>128 pivots
/// deep here only at the last index, and only for the smallest margins).
pub fn inertia_below(columns: &[Vec<(usize, f64)>], beta: f64) -> Option<InertiaCount> {
    inertia_below_with(columns, beta, Ordering::default())
}

/// [`inertia_below`] under a chosen [`Ordering`] — the same count, different
/// fill. The ordering changes nothing about the answer (a permutation preserves
/// inertia and every entry keeps its error bound), so the two are exactly
/// comparable and a caller can trade one for the other per matrix.
pub fn inertia_below_with(
    columns: &[Vec<(usize, f64)>],
    beta: f64,
    ordering: Ordering,
) -> Option<InertiaCount> {
    let n = columns.len();
    if n == 0 || columns.iter().any(|c| c.iter().any(|&(i, _)| i >= n)) {
        return None;
    }

    // Adjacency for the ordering only (values play no part in it).
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, column) in columns.iter().enumerate() {
        for &(i, v) in column {
            if i != j && v != 0.0 {
                adjacency[j].push(i);
                adjacency[i].push(j);
            }
        }
    }
    let order = match ordering {
        Ordering::ReverseCuthillMckee => reverse_cuthill_mckee(&adjacency),
        Ordering::NestedDissection => nested_dissection(&adjacency),
    };
    debug_assert_eq!(
        {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            sorted.dedup();
            sorted.len()
        },
        n,
        "the ordering must be a permutation"
    );
    // `order[new] = old`, and `position[old] = new`.
    let mut position = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        position[old] = new;
    }

    // `H − βI` in the new numbering, stored **in full** (both triangles), with
    // error bounds. Storing only one triangle is half the memory and twice the
    // bookkeeping: the fill created by an update lands in whichever triangle
    // the pair happens to fall in, and getting that wrong silently changes the
    // matrix (measured — it broke this counter on the odd sector before the
    // storage was doubled). The full symmetric form costs a factor two and is
    // checkable at a glance.
    let mut rows: Vec<BTreeMap<usize, Bounded>> = vec![BTreeMap::new(); n];
    // Diagonals are collected separately and shifted once, at the end: they are
    // `H_ii − β`, and a diagonal the input leaves structurally zero is `−β`,
    // not absent.
    let mut diagonal = vec![0.0_f64; n];
    for (old_col, column) in columns.iter().enumerate() {
        for &(old_row, value) in column {
            if old_row == old_col {
                diagonal[position[old_row]] = value;
                continue;
            }
            if value == 0.0 {
                continue;
            }
            let row = position[old_row];
            let col = position[old_col];
            // A symmetric matrix contributes each off-diagonal pair twice (once
            // from each column); keep the larger error rather than overwrite.
            rows[row]
                .entry(col)
                .and_modify(|e| e.error = e.error.max(0.0))
                .or_insert(Bounded::exact(value));
            rows[col]
                .entry(row)
                .and_modify(|e| e.error = e.error.max(0.0))
                .or_insert(Bounded::exact(value));
        }
    }
    // The shift goes on **every** diagonal, including the ones the input leaves
    // structurally zero: `H − βI` has `−β` there. Skipping those rows (as an
    // earlier version did, by zero-filtering the entries) leaves part of the
    // matrix unshifted and the count wrong — measured on the odd sector at
    // `N ≤ 6`, where most diagonal entries are zero.
    for i in 0..n {
        rows[i].insert(i, Bounded::exact(diagonal[i]).sub(Bounded::exact(beta)));
    }

    let mut negative = 0usize;
    let mut fill = 0usize;
    let mut swaps = 0usize;
    for j in 0..n {
        // Pivot selection *within* the trailing submatrix. A symmetric
        // permutation preserves the eigenvalues, so swapping the row and its
        // column labels to bring a larger diagonal entry into place is free —
        // and without it a zero or near-zero pivot aborts the count on a
        // perfectly ordinary matrix (measured: the odd sector at `N ≤ 6`).
        let acceptable = |pivot: &Bounded| pivot.certain_sign().is_some();
        let current = rows[j].get(&j).copied();
        if !current.map(|p| acceptable(&p)).unwrap_or(false) {
            let mut best: Option<(usize, f64)> = None;
            for k in j + 1..n {
                if let Some(pivot) = rows[k].get(&k) {
                    if acceptable(pivot) {
                        let magnitude = pivot.value.abs() - pivot.error;
                        if best.map(|(_, m)| magnitude > m).unwrap_or(true) {
                            best = Some((k, magnitude));
                        }
                    }
                }
            }
            let (k, _) = best?;
            swap_trailing(&mut rows, j, k);
            swaps += 1;
        }
        let pivot = rows[j].remove(&j)?;
        let sign = pivot.certain_sign()?;
        if sign < 0 {
            negative += 1;
        }

        // Support of row `j` (both triangles are stored, so this is also the
        // support of column `j`), without the pivot itself.
        let support: Vec<(usize, Bounded)> = rows[j]
            .iter()
            .map(|(&k, &value)| (k, value))
            .filter(|&(k, _)| k > j)
            .collect();
        rows[j].clear();
        fill = fill.max(support.len() + 1);
        if support.is_empty() {
            continue;
        }

        // `A[j+1.., j+1..] -= l l\u1d40 d_j`, applied to both triangles.
        for (a, &(i, value_i)) in support.iter().enumerate() {
            let Some(l_i) = value_i.div(pivot) else {
                return None;
            };
            // Only the upper half is ever read again (row `i` is read as a
            // pivot row later), so the mirrored half is left stale — it is the
            // storage that holds both triangles that matters for the build.
            for &(k, value_k) in support.iter().skip(a) {
                let update = l_i.mul(value_k);
                let slot = rows[i].entry(k).or_insert(Bounded::exact(0.0));
                *slot = slot.sub(update);
                fill = fill.max(rows[i].len());
            }
        }
    }

    Some(InertiaCount {
        below: negative,
        pivots: n,
        fill,
        swaps,
        ordering,
    })
}

/// Exchange indices `j` and `k` of the *trailing* submatrix: rows `j` and `k`
/// swap places, and so do the column labels `j`/`k` in every remaining row.
/// Rows before `j` are already eliminated (diagonal only) and are untouched.
fn swap_trailing(rows: &mut [BTreeMap<usize, Bounded>], j: usize, k: usize) {
    rows.swap(j, k);
    for row in rows.iter_mut().skip(j) {
        let at_j = row.remove(&j);
        let at_k = row.remove(&k);
        if let Some(value) = at_k {
            row.insert(j, value);
        }
        if let Some(value) = at_j {
            row.insert(k, value);
        }
    }
}

/// Reverse Cuthill–McKee on a symmetric graph: breadth-first from a
/// lowest-degree vertex, then reversed numbering. Cheapest ordering that
/// meaningfully reduces the bandwidth of a local-interaction graph.
fn reverse_cuthill_mckee(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let degree: Vec<usize> = adjacency.iter().map(|a| a.len()).collect();
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    // Start from the least-connected vertex, the standard pseudo-peripheral pick.
    let mut start = (0..n).min_by_key(|&v| degree[v]).unwrap_or(0);
    loop {
        visited[start] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            let mut neighbours: Vec<usize> = adjacency[v]
                .iter()
                .copied()
                .filter(|&w| !visited[w])
                .collect();
            neighbours.sort_by_key(|&w| degree[w]);
            neighbours.dedup();
            for w in neighbours {
                if !visited[w] {
                    visited[w] = true;
                    queue.push_back(w);
                }
            }
        }
        match (0..n).find(|&v| !visited[v]) {
            // Disconnected graph: restart in the next component. Nothing in
            // this module is disconnected, but dropping vertices silently would
            // be worse than a second pass.
            Some(disconnected) => start = disconnected,
            None => break,
        }
    }
    order.reverse();
    order
}

/// Nested dissection by recursive level-set bisection.
///
/// A true minimum-degree or vertex-separator ordering is an optimisation problem;
/// this takes the cheap approximation that keeps the nesting property: pick a
/// pseudo-peripheral start, label the connected component by BFS level, and cut
/// at the level whose two sides are the most balanced. The cut level is a
/// separator (every path between the sides crosses it), so ordering both sides
/// first and the separator last keeps the separator's fill in one block instead
/// of spreading it over the whole matrix. A disconnected set has no separator at
/// all and is ordered component by component.
///
/// Leaves of at most [`DISSECTION_LEAF`] vertices are finished by ascending
/// degree — a minimum-degree step small enough to be free.
fn nested_dissection(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let mut dissector = Dissector {
        adjacency,
        stamp: vec![0; adjacency.len()],
        generation: 0,
        order: Vec::with_capacity(adjacency.len()),
    };
    let all: Vec<usize> = (0..adjacency.len()).collect();
    dissector.visit(&all);
    debug_assert_eq!(
        {
            let mut sorted = dissector.order.clone();
            sorted.sort_unstable();
            sorted.dedup();
            sorted.len()
        },
        adjacency.len(),
        "nested dissection must return a permutation"
    );
    dissector.order
}

/// Below this many vertices a region is ordered by degree instead of cut again:
/// the recursion stops paying for itself once a separator is a large fraction of
/// the region, and every extra level costs a BFS over the subgraph.
const DISSECTION_LEAF: usize = 64;

struct Dissector<'a> {
    adjacency: &'a [Vec<usize>],
    /// Membership marks: `stamp[v] == generation` means `v` is in the region
    /// currently being cut. One array reused across the recursion, so marking a
    /// region costs its own size rather than `n`.
    stamp: Vec<u32>,
    generation: u32,
    order: Vec<usize>,
}

impl Dissector<'_> {
    fn mark(&mut self, region: &[usize]) {
        self.generation += 1;
        for &v in region {
            self.stamp[v] = self.generation;
        }
    }

    fn contains(&self, v: usize) -> bool {
        self.stamp[v] == self.generation
    }

    fn visit(&mut self, region: &[usize]) {
        if region.len() <= DISSECTION_LEAF {
            let mut leaf = region.to_vec();
            leaf.sort_by_key(|&v| self.adjacency[v].len());
            self.order.extend(leaf);
            return;
        }
        self.mark(region);

        // BFS from the least-connected vertex of the region, recording the level
        // of each vertex reached.
        let start = *region
            .iter()
            .min_by_key(|&&v| self.adjacency[v].len())
            .expect("non-empty region");
        let mut level: Vec<usize> = vec![usize::MAX; self.adjacency.len()];
        let mut reached = vec![start];
        level[start] = 0;
        let mut frontier = vec![start];
        let mut depth = 0usize;
        while !frontier.is_empty() {
            depth += 1;
            let mut next = Vec::new();
            for &v in &frontier {
                for &w in &self.adjacency[v] {
                    if self.contains(w) && level[w] == usize::MAX {
                        level[w] = depth;
                        next.push(w);
                    }
                }
            }
            reached.extend_from_slice(&next);
            frontier = next;
        }

        // A component the BFS did not reach is a separate block: order the
        // components independently and let the largest go first, so the biggest
        // block is eliminated first (the same rule as a balanced cut).
        if reached.len() < region.len() {
            let mut components = Vec::new();
            while let Some(&seed) = region.iter().find(|&&v| level[v] == usize::MAX) {
                let mut component = vec![seed];
                level[seed] = 0;
                let mut frontier = vec![seed];
                while !frontier.is_empty() {
                    let mut next = Vec::new();
                    for &v in &frontier {
                        for &w in &self.adjacency[v] {
                            if self.contains(w) && level[w] == usize::MAX {
                                level[w] = 0;
                                component.push(w);
                                next.push(w);
                            }
                        }
                    }
                    frontier = next;
                }
                components.push(component);
            }
            components.sort_by_key(|c| std::cmp::Reverse(c.len()));
            for component in components {
                self.visit(&component);
            }
            return;
        }

        // Cut at the level that balances the two sides. `depth` is one past the
        // last level; the ends (level 0 and the last level) are never cut
        // because they leave an empty side.
        let mut per_level = vec![0usize; depth + 1];
        for &v in region {
            per_level[level[v]] += 1;
        }
        let mut best: Option<(usize, usize)> = None;
        let mut left = 0usize;
        for cut in 1..depth {
            left += per_level[cut - 1];
            let right: usize = per_level.iter().skip(cut + 1).sum();
            if per_level[cut] == region.len() || left == 0 || right == 0 {
                continue;
            }
            let imbalance = left.abs_diff(right);
            if best.map(|(_, b)| imbalance < b).unwrap_or(true) {
                best = Some((cut, imbalance));
            }
        }

        let Some((cut, _)) = best else {
            // No level split is available (a complete graph reaches everything in
            // one step): fall back to ascending degree, the same rule as a leaf.
            let mut flat = region.to_vec();
            flat.sort_by_key(|&v| self.adjacency[v].len());
            self.order.extend(flat);
            return;
        };

        let lower: Vec<usize> = region.iter().copied().filter(|&v| level[v] < cut).collect();
        let upper: Vec<usize> = region.iter().copied().filter(|&v| level[v] > cut).collect();
        let separator: Vec<usize> = region
            .iter()
            .copied()
            .filter(|&v| level[v] == cut)
            .collect();
        // Both sides first (the region mark is re-established by each `visit`),
        // then the separator — it is the last thing eliminated, which is the
        // whole point of the nesting.
        self.visit(&lower);
        self.visit(&upper);
        let mut separator = separator;
        separator.sort_by_key(|&v| self.adjacency[v].len());
        self.order.extend(separator);
    }
}

/// A Ritz computation by Lanczos with full reorthogonalization — matvec-only,
/// so it reaches truncations where the dense witness cannot.
///
/// `H V_m = V_m T_m + β_m v_{m+1} e_m^*` gives, for the `i`-th eigenpair
/// `(θ_i, y_i)` of the tridiagonal `T_m`, the residual `|β_m·y_i[m−1]|`; that is
/// [`crate::mass_gap_spec::parlett_bound`]'s input and hence a certified
/// interval for *some* eigenvalue of `H`.
///
/// **What this cannot do** is name one. A Ritz value bounds `λ_i` from *above*
/// (`θ_i ≥ λ_i`, Rayleigh–Ritz), and its interval contains *some* eigenvalue;
/// excluding the ground from `(β, ∞)` is an eigenvalue *count*, which no
/// matvec-only method produces. So a gap certificate needs a count beside it —
/// [`certify_spectrum`] (dense, `O(n³)`) or [`inertia_below`] (sparse
/// elimination) — and never this alone. [`DiagonalExclusion`] and
/// [`MatchingExclusion`] are the matvec-only family, measured too weak here.
pub struct LanczosRitz {
    /// Steps actually taken (stops early on breakdown).
    pub steps: usize,
    /// Ritz values of `T_m`, ascending.
    pub values: Vec<f64>,
    /// `‖H y_i − θ_i y_i‖` from the Lanczos relation, matching `values`.
    pub residuals: Vec<f64>,
    /// Max deviation of the stored basis from orthonormality — the measured
    /// quality of the reorthogonalization, not an assumption.
    pub orthogonality: f64,
    /// Step index at which the Krylov space became invariant, if it did.
    pub breakdown: Option<usize>,
}

impl LanczosRitz {
    /// The lowest Ritz pair `(θ, ‖r‖)` — the input Temple takes.
    pub fn ground(&self) -> Option<(f64, f64)> {
        Some((*self.values.first()?, *self.residuals.first()?))
    }

    /// Certified two-sided intervals `[θ − δ, θ + δ]`, each containing *some*
    /// eigenvalue (`c(n)` uses the number of steps, the matrix genuinely
    /// diagonalized).
    pub fn intervals(&self) -> Vec<(f64, f64)> {
        let m = self.steps.max(1);
        let norm = self.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        self.values
            .iter()
            .zip(&self.residuals)
            .map(|(&theta, &r)| {
                let delta = crate::mass_gap_spec::certified_width(r, m, norm.max(1.0), theta);
                (theta - delta, theta + delta)
            })
            .collect()
    }
}

/// Run `steps` Lanczos steps with two passes of full reorthogonalization.
///
/// Returns `None` for an empty or zero start vector, a non-finite iterate, or
/// `steps == 0`. A Krylov space that becomes invariant before `steps` is
/// reported in [`LanczosRitz::breakdown`] rather than silently padded.
pub fn lanczos_ritz(
    matvec: &dyn Fn(&[f64]) -> Vec<f64>,
    n: usize,
    start: &[f64],
    steps: usize,
) -> Option<LanczosRitz> {
    if n == 0 || steps == 0 || start.len() != n {
        return None;
    }
    let start_norm = start.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(start_norm > 0.0) {
        return None;
    }
    // Arnoldi with two-pass modified Gram–Schmidt. `columns[j] = H v_j` is kept
    // so that both the projection `Vᵀ H V` and every residual come from the
    // same columns — no extra matvecs, and no reliance on the three-term
    // recurrence holding numerically.
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
    basis.push(start.iter().map(|x| x / start_norm).collect());
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(steps);
    let mut breakdown = None;

    for j in 0..steps {
        let column = matvec(&basis[j]);
        if column.len() != n || !column.iter().all(|x| x.is_finite()) {
            return None;
        }
        columns.push(column.clone());
        let mut w = column;
        // Two passes: the second sweeps up the orthogonality the first leaves
        // behind, which is what makes the measured defect ~1e-15 instead of the
        // classical ~1e-8.
        for _ in 0..2 {
            for v in basis.iter() {
                let h: f64 = v.iter().zip(&w).map(|(a, b)| a * b).sum();
                for (wi, vi) in w.iter_mut().zip(v.iter()) {
                    *wi -= h * vi;
                }
            }
        }
        let beta = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = basis[j]
            .iter()
            .zip(&columns[j])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs()
            .max(1.0);
        if beta <= 16.0 * UNIT_ROUNDOFF * scale {
            breakdown = Some(j);
            break;
        }
        basis.push(w.iter().map(|x| x / beta).collect());
    }

    let m = columns.len();
    // `T = Vᵀ H V` from the stored columns. It is symmetric in exact arithmetic;
    // the average of the two off-diagonal halves is what gets diagonalized, so
    // the matrix handed to `symmetric_eigen` is Hermitian by construction.
    let mut t = DMatrix::<f64>::zeros(m, m);
    for (a, va) in basis.iter().take(m).enumerate() {
        for (b, column) in columns.iter().enumerate() {
            let h: f64 = va.iter().zip(column).map(|(x, y)| x * y).sum();
            t[(a, b)] = h;
        }
    }
    for a in 0..m {
        for b in a + 1..m {
            let mean = 0.5 * (t[(a, b)] + t[(b, a)]);
            t[(a, b)] = mean;
            t[(b, a)] = mean;
        }
    }
    let eig = t.symmetric_eigen();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let mut values = Vec::with_capacity(m);
    let mut residuals = Vec::with_capacity(m);
    for &i in &order {
        let theta = eig.eigenvalues[i];
        values.push(theta);
        // ‖H (V y) − θ (V y)‖, with `H(V y) = Σ_j y_j · column_j` — the same
        // columns the projection used.
        let mut residual_sq = 0.0_f64;
        for row in 0..n {
            let hv_y: f64 = columns
                .iter()
                .enumerate()
                .map(|(j, c)| eig.eigenvectors[(j, i)] * c[row])
                .sum();
            let vy: f64 = basis
                .iter()
                .take(m)
                .enumerate()
                .map(|(j, v)| eig.eigenvectors[(j, i)] * v[row])
                .sum();
            residual_sq += (hv_y - theta * vy) * (hv_y - theta * vy);
        }
        residuals.push(residual_sq.sqrt());
    }
    // Gram defect, measured explicitly rather than assumed: this is the number
    // that says whether the two-sided reorthogonalization worked.
    let mut orthogonality: f64 = 0.0;
    for a in 0..basis.len() {
        for b in 0..basis.len() {
            let dot: f64 = basis[a].iter().zip(&basis[b]).map(|(x, y)| x * y).sum();
            let target = if a == b { 1.0 } else { 0.0 };
            orthogonality = orthogonality.max((dot - target).abs());
        }
    }
    Some(LanczosRitz {
        steps: m,
        values,
        residuals,
        orthogonality,
        breakdown,
    })
}

/// Assemble the T6 gap expression from two sector solves.
///
/// This is arithmetic on the two lowest certified Ritz values: it does **not**
/// verify that either solve resolved its sector ground (see
/// [`GapCertificate::lo`]). Callers that need a bound must supply that
/// witness themselves.
pub fn certified_mass_gap(
    even: &ForwardSirkResult,
    odd: &ForwardSirkResult,
) -> Option<GapCertificate> {
    let ce = certified_ground_state(even)?;
    let co = certified_ground_state(odd)?;
    let gap = co.value - ce.value;
    let delta = co.delta() + ce.delta();
    Some(GapCertificate {
        even: ce,
        odd: co,
        gap,
        lo: gap - delta,
        hi: gap + delta,
    })
}

/// Serialize a certified mass gap as NDJSON: the data (parity labels, θ, δ)
/// consumed by the Lean4 T6 instance (`ChapterSirkCertifiedGap`) and
/// re-verified by `prob_kernel::verify::verify_export` (nanoda).
///
/// One line per sector plus one assembly line, each self-describing:
/// `{"kind": ..., "parity": ..., "theta": ..., "delta": ..., "lo": ...,
/// "hi": ...}`.
/// Emit with a per-line recorder. The `sink` receives every NDJSON line
/// (both sector certificates and the assembly) as it is produced, so a
/// host can durably record each emitted certificate (e.g. the kernel's
/// `uk_certificate_issued` → `certificates` stream) without re-parsing
/// the output. The returned string is identical to
/// [`emit_gap_certificate_ndjson`].
pub fn emit_gap_certificate_ndjson_with(
    gap: &GapCertificate,
    mut sink: impl FnMut(&str),
) -> String {
    // Every row names its sector and carries `physical: false` (§M4.9,
    // `PLAN_MASS_GAP.md` §7): this assembly is the one-outer-boson R-parity
    // *diagnostic*, and a reader must be able to see that from the row itself —
    // labelling rows by parity alone is what once let a diagnostic read as a
    // mass gap. The sector strings are fixed by construction: `certified_mass_gap`
    // always differences two one-outer-boson sector grounds.
    let sector = |parity: &str, c: &Certificate| {
        format!(
            "{{\"kind\":\"ritz_certificate\",\"parity\":\"{parity}\",\
\"sector\":\"one_outer_boson_r_{parity}\",\"physical\":false,\"theta\":{:.17e},\
\"delta\":{:.17e},\"lo\":{:.17e},\"hi\":{:.17e},\"residual\":{:.17e},\
\"roundoff\":{:.17e},\"enclosure\":{:.17e}}}",
            c.value,
            c.delta(),
            c.lo,
            c.hi,
            c.residual,
            c.roundoff,
            c.enclosure
        )
    };
    let assembly = format!(
        "{{\"kind\":\"certified_mass_gap\",\"sector\":\"one_outer_boson_r_split\",\
\"physical\":false,\"gap\":{:.17e},\"lo\":{:.17e},\
\"hi\":{:.17e},\"delta\":{:.17e},\"certified_positive\":{}}}",
        gap.gap,
        gap.lo,
        gap.hi,
        gap.even.delta() + gap.odd.delta(),
        gap.lo > 0.0
    );
    let mut out = String::new();
    for line in [sector("even", &gap.even), sector("odd", &gap.odd), assembly] {
        out.push_str(&line);
        out.push('\n');
        sink(&line);
    }
    out
}

/// Serialize a certified mass gap as NDJSON (see
/// [`emit_gap_certificate_ndjson_with`]): the data (parity labels, θ, δ)
/// consumed by the Lean4 T6 instance (`ChapterSirkCertifiedGap`) and
/// re-verified by `prob_kernel::verify::verify_export` (nanoda).
///
/// One line per sector plus one assembly line, each self-describing:
/// `{"kind": ..., "parity": ..., "theta": ..., "delta": ..., "lo": ...,
/// "hi": ...}`.
pub fn emit_gap_certificate_ndjson(gap: &GapCertificate) -> String {
    emit_gap_certificate_ndjson_with(gap, |_| {})
}

/// Serialize certified Ritz bands of a **single-sector** solve as NDJSON:
/// one line per certified value, each self-describing
/// `{"kind": "full_operator_band_certificate", "index": …, "theta": …,
/// "delta": …, "lo": …, "hi": …, "residual": …, "roundoff": …,
/// "enclosure": …}`.
///
/// This is the emission of the plan's QG-3.2(c) full-operator numerics: the
/// certified bands are stated for the operator actually solved (the full
/// `qg3d_full_hamiltonian`, cross terms included), with the honest caveat
/// that an indefinite (not bounded-below) operator carries no mass-gap
/// assembly — the bands certify the resolved spectrum of the truncated
/// operator as solved.
pub fn emit_ritz_bands_ndjson(certs: &[Certificate]) -> String {
    let mut out = String::new();
    for (i, c) in certs.iter().enumerate() {
        out.push_str(&format!(
            "{{\"kind\":\"full_operator_band_certificate\",\"index\":{i},\
\"theta\":{:.17e},\"delta\":{:.17e},\"lo\":{:.17e},\"hi\":{:.17e},\
\"residual\":{:.17e},\"roundoff\":{:.17e},\"enclosure\":{:.17e}}}\n",
            c.value,
            c.delta(),
            c.lo,
            c.hi,
            c.residual,
            c.roundoff,
            c.enclosure
        ));
    }
    out
}

// ── §M4.9: the physical gap, not the one-particle diagnostic ─────────────

/// The **physical-gap row** of work order §M4.9 (`PLAN_MASS_GAP.md` §7,
/// `CONSOLIDATED_PLAN.md` §M4.9): with the outer-vacuum ground at `E₀ = 0`
/// the gap is `E₁` outright, estimated as `gap = ‖H φ₁‖` — no difference of
/// two large sector energies, so no cancellation — for `φ₁` the first state
/// **above** the outer vacuum, `‖φ₁‖ = 1`.
///
/// All three numbers are measured by matvec from one normalized state:
///
/// ```text
///   e1   = ⟨φ₁|H|φ₁⟩                     (Rayleigh quotient, signed)
///   gap  = ‖H φ₁‖
///   q    = ‖H φ₁ − e₁ φ₁‖               (the residual, directly measurable)
/// ```
///
/// and because `Hφ₁ = e₁φ₁ + (Hφ₁ − e₁φ₁)` with `φ₁` orthogonal to the
/// remainder, the Pythagorean identity
///
/// ```text
///   gap² = e1² + q²
/// ```
///
/// holds to roundoff — the machine-checkable form of
/// `‖Hφ₁‖² = E₁² + ‖H d‖²` in `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md`
/// (CHECK 1–4, all `canonicalise → 0`). The relative error of the estimator is
/// `≈ ½(q/e1)²` — *quadratic* in the measurable residual, with no `‖H‖`
/// prefactor (`rel_err_est` reports exactly that expression).
///
/// **This row is an estimator, not a certificate.** It carries no certified
/// band: `q` is reported beside `gap` precisely because the claim is
/// meaningless without it, and whether a row is tight enough for the Lean gap
/// table (`CONSOLIDATED_PLAN.md` §M4.6 option (c): empty until a physical row
/// certifies) is the reader's decision, made from `q` — never defaulted.
/// Reporting `‖Hφ‖` as if it certified the gap would be exactly the kind of
/// claim `scripts/check-non-claims` exists to catch.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalGapCertificate {
    /// The sector label of `φ₁` — mandatory: a row that cannot name its sector
    /// is not emitted (§M4.9 requirement 4).
    pub sector: &'static str,
    /// Rayleigh quotient `⟨φ₁|H|φ₁⟩` (signed; positive — the state lies above
    /// the outer vacuum).
    pub e1: f64,
    /// The estimator `‖H φ₁‖`.
    pub gap: f64,
    /// The residual `‖H φ₁ − e₁ φ₁‖` (one matvec; no `‖H‖` prefactor).
    pub q: f64,
    /// The relative-error estimate `½(q/e1)²` of `gap` for the exact `E₁`.
    pub rel_err_est: f64,
}

impl PhysicalGapCertificate {
    /// Assemble the row from three measured numbers.
    pub fn new(sector: &'static str, e1: f64, gap: f64, q: f64) -> Self {
        let rel_err_est = if e1 != 0.0 {
            0.5 * (q / e1) * (q / e1)
        } else {
            f64::INFINITY
        };
        Self {
            sector,
            e1,
            gap,
            q,
            rel_err_est,
        }
    }

    /// Measure the row from the state `phi` itself (matvec, no Gram):
    /// `e1 = ⟨φ|H|φ⟩`, `gap = ‖Hφ‖`, `q = ‖Hφ − e₁φ‖`.
    ///
    /// `phi` is normalized defensively (every quantity above is stated for
    /// `‖φ‖ = 1`). Returns `None` when `e1 ≤ 0`, i.e. when the state is not
    /// **above** the outer vacuum: such a state cannot be `φ₁` for the
    /// physical gap, and §M4.9's contract is that no row is emitted for it —
    /// the gap table's "empty when nothing qualifies" is enforced here, not
    /// hoped for.
    pub fn measure(sector: &'static str, h: &Hamiltonian, phi: &QuantumState) -> Option<Self> {
        let mut phi = phi.clone();
        let norm = phi.norm();
        if norm > 0.0 && (norm - 1.0).abs() > 1e-12 {
            phi.scale_and_add(&phi.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
        }
        let hphi = h.apply(&phi);
        let rayleigh = QuantumState::inner_product(&phi, &hphi);
        // H Hermitian ⇒ the Rayleigh quotient is real up to roundoff.
        debug_assert!(
            rayleigh.im.abs() <= 1e-9 * (1.0 + rayleigh.re.abs()),
            "Rayleigh quotient has imaginary part {} — H is not Hermitian on this state",
            rayleigh.im
        );
        let e1 = rayleigh.re;
        if e1 <= 0.0 {
            return None;
        }
        let gap = hphi.norm();
        let mut residual = hphi;
        residual.scale_and_add(&phi, Complex64::new(-e1, 0.0));
        let q = residual.norm();
        Some(Self::new(sector, e1, gap, q))
    }

    /// `gap² = e1² + q²` evaluated from the reported numbers — the
    /// machine-checkable identity a consumer can re-verify from the row alone
    /// (holds to roundoff by construction of [`Self::measure`]).
    pub fn pythagoras_gap(&self) -> f64 {
        (self.e1 * self.e1 + self.q * self.q).sqrt()
    }
}

/// Serialize a physical-gap row as NDJSON with a per-line recorder: the sink
/// receives the line as produced (same recording convention as
/// [`emit_gap_certificate_ndjson_with`]). The row is self-describing:
/// `{"kind":"physical_gap", "sector":…, "physical":true, "gap":…,
/// "q":…, "e1":…, "rel_err_est":…}` — sector and the physical/diagnostic
/// flag in every row, `q` beside `gap` (§M4.9 acceptance).
pub fn emit_physical_gap_ndjson_with(
    certificate: &PhysicalGapCertificate,
    mut sink: impl FnMut(&str),
) -> String {
    let line = format!(
        "{{\"kind\":\"physical_gap\",\"sector\":\"{}\",\"physical\":true,\
\"gap\":{:.17e},\"q\":{:.17e},\"e1\":{:.17e},\"rel_err_est\":{:.17e}}}",
        certificate.sector, certificate.gap, certificate.q, certificate.e1, certificate.rel_err_est
    );
    let mut out = String::new();
    out.push_str(&line);
    out.push('\n');
    sink(&line);
    out
}

/// Serialize a physical-gap row as NDJSON (see
/// [`emit_physical_gap_ndjson_with`]).
pub fn emit_physical_gap_ndjson(certificate: &PhysicalGapCertificate) -> String {
    emit_physical_gap_ndjson_with(certificate, |_| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerBosonicState, Operator};

    fn inner_species(mode: u32) -> InnerBosonicState {
        let mut s = InnerBosonicState::vacuum();
        s.modes.insert(mode, 1);
        s
    }

    /// `H = C†(e₀)A(e₀)` — the outer-mode number operator of species `e₀`.
    /// Positive semidefinite, `H|Ω⟩ = 0`, eigenvalues are mode occupations:
    /// a toy with exact arithmetic on purpose, so the assertions below are
    /// identities, not tolerances about floating point.
    fn number_op(mode: u32) -> Hamiltonian {
        let inner = inner_species(mode);
        Hamiltonian {
            terms: vec![(
                Complex64::new(1.0, 0.0),
                vec![
                    Operator::OuterBosonCreate(inner.clone()),
                    Operator::OuterBosonAnnihilate(inner),
                ],
            )],
        }
    }

    fn outer_one(mode: u32) -> QuantumState {
        QuantumState::vacuum().apply(&Operator::OuterBosonCreate(inner_species(mode)))
    }

    #[test]
    fn measure_on_an_eigenstate_gives_exact_gap_and_vanishing_q() {
        let h = number_op(0);
        let phi = outer_one(0); // Hφ = φ exactly
        let row = PhysicalGapCertificate::measure("toy", &h, &phi).expect("e1 = 1 > 0");
        assert!((row.e1 - 1.0).abs() < 1e-12, "e1 = {}", row.e1);
        assert!((row.gap - 1.0).abs() < 1e-12, "gap = {}", row.gap);
        assert!(row.q < 1e-12, "q = {}", row.q);
        assert!(row.rel_err_est < 1e-24, "rel_err_est = {}", row.rel_err_est);
        // gap² = e1² + q² to roundoff.
        assert!(
            (row.gap - row.pythagoras_gap()).abs() < 1e-12,
            "identity gap {} vs pythagoras {}",
            row.gap,
            row.pythagoras_gap()
        );
    }

    #[test]
    fn measure_on_a_superposition_reports_the_residual_beside_the_gap() {
        // φ = (|e₀⟩ + |e₁⟩)/√2 with H = number operator of mode e₀:
        //   e1 = 1/2, gap = 1/√2, q = 1/2, and gap² = e1² + q² = 1/2 exactly.
        let h = number_op(0);
        let mut phi = outer_one(0);
        phi.scale_and_add(&outer_one(1), Complex64::new(1.0, 0.0));
        phi.scale_and_add(&phi.clone(), Complex64::new(1.0 / 2.0f64.sqrt() - 1.0, 0.0));
        let row = PhysicalGapCertificate::measure("toy", &h, &phi).expect("e1 = 1/2 > 0");
        assert!((row.e1 - 0.5).abs() < 1e-12, "e1 = {}", row.e1);
        assert!((row.gap - 0.5f64.sqrt()).abs() < 1e-12, "gap = {}", row.gap);
        assert!((row.q - 0.5).abs() < 1e-12, "q = {}", row.q);
        assert!(
            (row.gap - row.pythagoras_gap()).abs() < 1e-12,
            "identity gap {} vs pythagoras {}",
            row.gap,
            row.pythagoras_gap()
        );
        // The reported estimate is literally ½(q/e1)².
        assert!(
            (row.rel_err_est - 0.5 * (row.q / row.e1).powi(2)).abs() < 1e-15,
            "rel_err_est = {}",
            row.rel_err_est
        );
    }

    #[test]
    fn measure_refuses_a_state_at_or_below_the_outer_vacuum() {
        // H = −C†(e₀)A(e₀): the only state available has e1 = −1 < 0, so no
        // physical row exists — the emitter must produce nothing, not a row
        // with a negative "gap above the vacuum".
        let inner = inner_species(0);
        let h = Hamiltonian {
            terms: vec![(
                Complex64::new(-1.0, 0.0),
                vec![
                    Operator::OuterBosonCreate(inner.clone()),
                    Operator::OuterBosonAnnihilate(inner),
                ],
            )],
        };
        let phi = outer_one(0);
        assert!(
            PhysicalGapCertificate::measure("toy", &h, &phi).is_none(),
            "a state below the outer vacuum must not become a physical-gap row"
        );
    }

    #[test]
    fn physical_row_emits_sector_flag_and_q() {
        let h = number_op(0);
        let row = PhysicalGapCertificate::measure("first_above_outer_vacuum", &h, &outer_one(0))
            .expect("e1 = 1 > 0");
        let ndjson = emit_physical_gap_ndjson(&row);
        assert!(
            ndjson.ends_with('\n'),
            "NDJSON line must end with a newline"
        );
        let line = ndjson.trim_end();
        assert!(line.contains("\"kind\":\"physical_gap\""), "{line}");
        assert!(
            line.contains("\"sector\":\"first_above_outer_vacuum\""),
            "{line}"
        );
        assert!(line.contains("\"physical\":true"), "{line}");
        assert!(line.contains("\"q\":"), "q is mandatory: {line}");
        assert!(line.contains("\"gap\":"), "{line}");
    }

    #[test]
    fn diagnostic_rows_name_their_sector_and_flag_themselves_as_not_physical() {
        // The §M4.9 requirement that opened this work order: a row that cannot
        // say "I am the one-outer-boson diagnostic" must not exist.
        let cert = |theta: f64| Certificate::new(theta, 0.0, 0.0, 0.0);
        let gap = GapCertificate {
            even: cert(1.0),
            odd: cert(2.0),
            gap: 1.0,
            lo: 1.0,
            hi: 1.0,
        };
        let ndjson = emit_gap_certificate_ndjson(&gap);
        let lines: Vec<&str> = ndjson.lines().collect();
        assert_eq!(lines.len(), 3, "{ndjson}");
        assert!(
            lines[0].contains("\"sector\":\"one_outer_boson_r_even\""),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].contains("\"sector\":\"one_outer_boson_r_odd\""),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("\"sector\":\"one_outer_boson_r_split\""),
            "{}",
            lines[2]
        );
        for line in &lines {
            assert!(
                line.contains("\"physical\":false"),
                "every diagnostic row must carry physical:false: {line}"
            );
        }
        // Kinds and the certified_positive contract are unchanged (additive
        // schema only — the unfer_ffi recorder asserts this exact sequence).
        assert!(lines[0].contains("\"kind\":\"ritz_certificate\""));
        assert!(lines[2].contains("\"kind\":\"certified_mass_gap\""));
        // `certified_positive` still follows this row's own `lo` (1.0 > 0):
        // the flag's honesty contract is untouched by the added fields.
        assert!(lines[2].contains("\"certified_positive\":true"));
    }

    #[test]
    fn physical_gap_emitter_records_every_line_through_the_sink() {
        let h = number_op(0);
        let row = PhysicalGapCertificate::measure("toy", &h, &outer_one(0)).unwrap();
        let mut seen = Vec::new();
        let out = emit_physical_gap_ndjson_with(&row, |line| seen.push(line.to_string()));
        assert_eq!(seen.len(), 1);
        assert_eq!(out.trim_end(), seen[0]);
    }

    #[test]
    fn certify_spectrum_names_the_second_eigenvalue() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3, both simple. The certificate
        // must place them in two disjoint clusters, which is what licenses
        // reading the second cluster's lower edge as a λ₂ bound.
        let mut a = DMatrix::<f64>::zeros(2, 2);
        a[(0, 0)] = 2.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 2.0;
        let cert = certify_spectrum(&a).expect("certifiable");
        assert_eq!(
            cert.clusters.len(),
            2,
            "two simple eigenvalues ⇒ two clusters"
        );
        assert_eq!(cert.dimension, 2);
        let first = cert.eigenvalue_lower_bound(1).unwrap();
        let second = cert.second_eigenvalue_lower_bound().unwrap();
        assert!(first <= 1.0 + 1e-12, "λ₁ = 1 ≥ {first} must hold");
        assert!(first > 1.0 - 1e-9, "the bound should be tight, got {first}");
        assert!(second <= 3.0 + 1e-12, "λ₂ = 3 ≥ {second} must hold");
        assert!(
            second > 3.0 - 1e-9,
            "the bound should be tight, got {second}"
        );

        // A degenerate ground (λ₁ = λ₂ = 1) must refuse: no β above θ can be
        // at or below λ₂, so there is no exclusion witness to offer.
        let mut d = DMatrix::<f64>::zeros(3, 3);
        d[(0, 0)] = 1.0;
        d[(1, 1)] = 1.0;
        d[(2, 2)] = 2.0;
        let deg = certify_spectrum(&d).expect("certifiable");
        assert_eq!(deg.clusters.len(), 2);
        assert_eq!(deg.clusters[0].multiplicity, 2, "the ground is twofold");
        assert!(
            deg.second_eigenvalue_lower_bound().is_none(),
            "a degenerate ground has λ₂ = λ₁ and no witness"
        );
    }

    fn matvec_of(a: &DMatrix<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
        let a = a.clone();
        move |x: &[f64]| {
            (0..a.nrows())
                .map(|i| (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum())
                .collect()
        }
    }

    #[test]
    fn diagonal_exclusion_certifies_a_dominant_diagonal_and_refuses_a_wide_one() {
        // A diagonal-dominant matrix: the count below β is honest and tight
        // enough for Temple.
        let mut a = DMatrix::<f64>::zeros(4, 4);
        for i in 0..4 {
            a[(i, i)] = (i + 1) as f64;
        }
        a[(0, 1)] = 0.05;
        a[(1, 0)] = 0.05;
        let mv = matvec_of(&a);
        let exclusion = DiagonalExclusion::from_matvec(&mv, 4).expect("measurable");
        let beta = exclusion.second_eigenvalue_lower_bound().expect("n = 4");
        let exact = a.clone().symmetric_eigen();
        let lambda_2 = {
            let mut v: Vec<f64> = exact.eigenvalues.iter().cloned().collect();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[1]
        };
        assert!(
            beta <= lambda_2 + 1e-12,
            "the Weyl bound must not exceed λ₂ = {lambda_2}, got {beta}"
        );
        // Soundness alone is cheap — vacuity is not allowed here: the bound has
        // to clear λ₁ (≈ 0.9994) for Temple to say anything.
        assert!(
            beta > 1.0 - 0.01,
            "the bound must track λ₂ here, got {beta}"
        );
        let theta = exact
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let temple = crate::mass_gap_spec::temple_lower_bound(theta, 0.0, beta)
            .expect("β clears θ, so Temple applies");
        assert!(temple <= lambda_2 + 1e-12);

        // The same route on a matrix whose off-diagonal carries the same
        // weight as the diagonal: β falls below θ and Temple must refuse.
        let mut wide = DMatrix::<f64>::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                wide[(i, j)] = if i == j { 3.0 } else { 1.0 };
            }
        }
        let mv_wide = matvec_of(&wide);
        let wide_exclusion = DiagonalExclusion::from_matvec(&mv_wide, 4).expect("measurable");
        let beta_wide = wide_exclusion
            .second_eigenvalue_lower_bound()
            .expect("n = 4");
        let theta_wide = wide
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            beta_wide <= theta_wide,
            "expected the diagonal split to collapse (β = {beta_wide} ≤ θ = {theta_wide})"
        );
        assert!(
            crate::mass_gap_spec::temple_lower_bound(theta_wide, 0.0, beta_wide).is_none(),
            "Temple must refuse a β at or below θ rather than return a bound"
        );
    }

    #[test]
    fn lanczos_ritz_is_orthonormal_and_bounds_the_spectrum_from_above() {
        let mut a = DMatrix::<f64>::zeros(6, 6);
        for i in 0..6 {
            a[(i, i)] = (i + 1) as f64;
            if i + 1 < 6 {
                a[(i, i + 1)] = 0.3;
                a[(i + 1, i)] = 0.3;
            }
        }
        let mv = matvec_of(&a);
        let start = vec![1.0; 6];
        let ritz = lanczos_ritz(&mv, 6, &start, 4).expect("four steps must run");
        assert_eq!(ritz.steps, 4);
        assert_eq!(ritz.values.len(), 4);
        assert!(
            ritz.orthogonality < 1e-12,
            "two passes of full reorthogonalization must hold orthonormality, got {}",
            ritz.orthogonality
        );
        assert!(
            ritz.breakdown.is_none(),
            "6 distinct levels, 4 steps: no breakdown"
        );
        let exact = {
            let e = a.clone().symmetric_eigen();
            let mut v: Vec<f64> = e.eigenvalues.iter().cloned().collect();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v
        };
        for (i, &theta) in ritz.values.iter().enumerate() {
            assert!(
                theta >= exact[i] - 1e-12,
                "Rayleigh–Ritz: θ_{i} = {theta} must be ≥ λ_{i} = {}",
                exact[i]
            );
        }
        for (lo, hi) in ritz.intervals() {
            assert!(
                exact.iter().any(|&l| (lo..=hi).contains(&l)),
                "each Parlett interval must contain some eigenvalue: [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn lanczos_reports_the_subspace_ground_not_the_spectral_ground() {
        // The matvec-only form of the audit's finding: a chain orthogonal to the
        // ground converges to the *second* level with a tiny residual. Here the
        // start is exactly e₂ of a diagonal matrix, so the Krylov space never
        // sees λ₁ = 1 and the lowest Ritz value is λ₂ = 2 with zero residual.
        let mut d = DMatrix::<f64>::zeros(4, 4);
        for i in 0..4 {
            d[(i, i)] = (i + 1) as f64;
        }
        let mv = matvec_of(&d);
        let start = vec![0.0, 1.0, 1e-3, 0.0];
        let ritz = lanczos_ritz(&mv, 4, &start, 3).expect("three steps must run");
        let (theta, residual) = ritz.ground().expect("non-empty");
        assert!(
            (theta - 2.0).abs() < 1e-4,
            "the chain must converge to λ₂ = 2, got {theta}"
        );
        assert!(
            residual < 1e-9,
            "and it must look converged while doing it, residual = {residual}"
        );
        // So a residual cannot certify a ground: it has to be paired with an
        // exclusion witness, and even a correct β = λ₂ passes Temple only if θ
        // is below it — here θ sits *at* β, which the guard rejects.
        assert!(
            crate::mass_gap_spec::temple_lower_bound(theta, residual, 2.0).is_none(),
            "θ at β must be refused, not turned into a bound"
        );
    }
}

#[cfg(test)]
mod inertia_tests {
    use super::inertia_below;

    /// Deterministic pseudo-random symmetric matrices, counted against a dense
    /// eigendecomposition. Three regimes are exercised that each broke an
    /// earlier version of the counter: dense random entries, congruent scaling
    /// across nine orders of magnitude, and structurally-zero diagonals (where
    /// the `−β` shift is easiest to drop).
    #[test]
    fn the_sparse_count_matches_dense_eigendecomposition() {
        use nalgebra::DMatrix;
        let mut state: u64 = 12345;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
        };
        let mut decided = 0usize;
        for trial in 0..20 {
            let n = 5 + (trial % 9);
            let mut a = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..=i {
                    let v = if i == j { next() * 2.0 } else { next() };
                    a[(i, j)] = v;
                    a[(j, i)] = v;
                }
            }
            if trial >= 10 {
                let scales: Vec<f64> = (0..n).map(|i| 10f64.powi(((i % 7) as i32) - 3)).collect();
                for i in 0..n {
                    for j in 0..n {
                        a[(i, j)] *= scales[i] * scales[j];
                    }
                }
            }
            if trial % 3 == 0 {
                for i in (0..n).step_by(3) {
                    a[(i, i)] = 0.0;
                }
            }
            let columns: Vec<Vec<(usize, f64)>> = (0..n)
                .map(|j| {
                    (0..n)
                        .filter(|&i| a[(i, j)] != 0.0)
                        .map(|i| (i, a[(i, j)]))
                        .collect()
                })
                .collect();
            let spec = a.clone().symmetric_eigen().eigenvalues;
            for k in 0..6 {
                let beta = -3.0 + k as f64;
                let want = spec.iter().filter(|&&v| v < beta).count();
                if let Some(got) = inertia_below(&columns, beta) {
                    decided += 1;
                    assert_eq!(
                        got.below, want,
                        "trial {trial} n={n} beta={beta}: counted {} below, dense says {want}",
                        got.below
                    );
                }
            }
        }
        assert!(
            decided >= 100,
            "the counter must decide on this corpus, not refuse it ({decided} decided)"
        );
    }

    /// The ordering is a permutation, so it may change the fill and nothing
    /// else. This is checked directly rather than argued: the same matrix is
    /// counted under both orderings and the counts must be identical — a
    /// nested-dissection bug that dropped or duplicated a vertex would show up
    /// here as a count off by one, which is exactly how two earlier defects in
    /// this function were caught.
    #[test]
    fn both_orderings_count_the_same_eigenvalues() {
        use super::{Ordering, inertia_below_with};
        use nalgebra::DMatrix;
        let mut state: u64 = 987654321;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut compared = 0usize;
        for trial in 0..6 {
            // Banded, so the two orderings actually differ: a dense matrix has
            // no structure to dissect and would test nothing.
            let n = 24 + 8 * trial;
            let band = 3 + trial % 3;
            let mut a = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                a[(i, i)] = 4.0 + next();
                for d in 1..=band {
                    if i + d < n {
                        let v = next();
                        a[(i, i + d)] = v;
                        a[(i + d, i)] = v;
                    }
                }
            }
            let columns: Vec<Vec<(usize, f64)>> = (0..n)
                .map(|j| {
                    (0..n)
                        .filter(|&i| a[(i, j)] != 0.0)
                        .map(|i| (i, a[(i, j)]))
                        .collect()
                })
                .collect();
            let spec = a.clone().symmetric_eigen().eigenvalues;
            for k in 0..4 {
                let beta = -1.0 + k as f64;
                let rcm = inertia_below_with(&columns, beta, Ordering::ReverseCuthillMckee);
                let nd = inertia_below_with(&columns, beta, Ordering::NestedDissection);
                let want = spec.iter().filter(|&&v| v < beta).count();
                if let (Some(rcm), Some(nd)) = (rcm, nd) {
                    compared += 1;
                    assert_eq!(
                        rcm.below, nd.below,
                        "trial {trial} n={n} beta={beta}: RCM counts {} and nested dissection {} — \
                         the ordering must not change the inertia",
                        rcm.below, nd.below
                    );
                    assert_eq!(rcm.below, want, "trial {trial} n={n} beta={beta}");
                    assert_eq!(
                        (rcm.pivots, nd.pivots),
                        (n, n),
                        "both eliminations must run to completion"
                    );
                }
            }
        }
        assert!(
            compared >= 12,
            "the comparison must actually decide ({compared} pairs decided)"
        );
    }

    /// A 2×2 matrix shifted to `[[0,1],[1,0]]` has no 1×1 pivot at all: the
    /// counter must refuse rather than guess a sign.
    #[test]
    fn the_sparse_count_refuses_a_pivot_that_is_not_there() {
        let columns = vec![vec![(0, 0.0), (1, 1.0)], vec![(0, 1.0), (1, 0.0)]];
        assert!(
            inertia_below(&columns, 0.0).is_none(),
            "a matrix with no decidable 1×1 pivot must be refused"
        );
    }
}
