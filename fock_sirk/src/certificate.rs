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
//! cross-sector assembly ([`certified_mass_gap`]) is the T6 certified-gap
//! theorem of §3.4: with `θᵒ₀, θᵉ₀` the computed lowest Ritz values of the
//! odd/even sectors of `H_m` and certified widths `δᵒ, δᵉ`,
//!
//! ```text
//!   λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ),                 (T6)
//! ```
//!
//! and the emitter ([`emit_gap_certificate_ndjson`]) serializes the
//! certificates (parity labels, θ, δ) as the data consumed by the Lean4
//! T6 instance (the `BookProof/ChapterSirkCertifiedGap` target) and
//! re-verified by `prob_kernel::verify::verify_export` (nanoda).

use nalgebra::DMatrix;
use num_complex::Complex64;

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
/// the certified statement is
///
/// ```text
///   λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ) =: lo,
/// ```
///
/// i.e. `lo > 0` is a **proof** that the truncated Hamiltonian has a mass
/// gap (with the finite-`m` and finite-precision widths explicit).  The
/// interval `[lo, hi]` bounds the exact gap of `H_m`; the analytic
/// strong-coupling value `g²/2` may sit outside it by the `O(g⁴)` magnetic
/// correction — that deviation is deliberately *not* part of the
/// certificate.
pub struct GapCertificate {
    /// Certified interval for the even-sector ground state.
    pub even: Certificate,
    /// Certified interval for the odd-sector ground state.
    pub odd: Certificate,
    /// Computed gap `θᵒ₀ − θᵉ₀`.
    pub gap: f64,
    /// Certified lower bound of `λ₁(H_m) − λ₀(H_m)` (the T6 statement).
    pub lo: f64,
    /// Certified upper bound `θᵒ₀ − θᵉ₀ + (δᵒ + δᵉ)`.
    pub hi: f64,
}

impl GapCertificate {
    /// Does the certified interval `[lo, hi]` contain the measured gap `θᵒ₀ − θᵉ₀`?
    /// (Trivially true by construction — the width is the sum of the sector
    /// widths — but asserted as an invariant by consumers.)
    pub fn contains_measured(&self) -> bool {
        self.lo <= self.gap && self.gap <= self.hi
    }
}

/// Assemble the T6 certified mass gap from two sector solves.
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
    let sector = |parity: &str, c: &Certificate| {
        format!(
            "{{\"kind\":\"ritz_certificate\",\"parity\":\"{parity}\",\"theta\":{:.17e},\
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
        "{{\"kind\":\"certified_mass_gap\",\"gap\":{:.17e},\"lo\":{:.17e},\
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
