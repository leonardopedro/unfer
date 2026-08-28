//! Pure, proof-facing core of the certified mass gap.
//!
//! This module is the **formalization seam** of `MASS_GAP_CERTIFIED.md`
//! §4.4–§5 (the non-Lean part): it isolates the mass-gap certificate's
//! elementary computations as pure, dependency-free functions (plain `f64`
//! arithmetic, no `nalgebra`, no I/O, no allocation) so that a translation
//! tool (Aeneas/Verus — §5.3) or a proof specialist can attach theorems to
//! a small, stable surface. Every function carries its exact mathematical
//! contract: preconditions, postconditions, and the identity it implements.
//! The runtime checks the checkable preconditions (`debug_assert`); the rest
//! is the caller's (documented) obligation.
//!
//! The theorem of record (T6, `MASS_GAP_CERTIFIED.md` §3.4):
//!
//! ```text
//!   λ₁(H_m) − λ₀(H_m) ≥ θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ),
//!   δˢ = ‖rˢ‖ + c(nˢ)·u·‖Ĝˢ‖ + h_Oˢ,
//! ```
//!
//! with `θˢ₀` the lowest Ritz value of the parity-sector `s ∈ {e, o}` solve,
//! `‖rˢ‖` the a-posteriori Rayleigh–Ritz residual (Parlett), `c(n) = n³`
//! the conservative backward-error constant, `u = 2⁻⁵³` the unit roundoff,
//! `‖Ĝˢ‖` the spectral norm of the whitened reduced Hamiltonian actually
//! diagonalized, and `h_Oˢ = 4u·max(|θ|, 1)` the directed-rounding enclosure
//! of the measured value.
//!
//! Preconditions of the theorem (each documented where it is enforced):
//!
//! 1. **Sector purity**: for the gauge-fixed QYM Hamiltonian the sector
//!    symmetry is the exact reflection `R: (A₀,A₁) → (−A₁,−A₀)` (exact for
//!    all `g`); the starts are pure-sector, so the two Krylov chains are
//!    disjoint and the Ritz sets are independent ([`parities_disjoint`]).
//!    Occupation parity and lattice Hamiltonians are comparison-only and are
//!    not inputs to this mass-gap contract.
//! 2. **Ground selection**: `θˢ₀` is the *lowest* Ritz value of sector `s`
//!    (the solve returns the sorted spectrum).
//! 3. **Enclosure**: `δˢ` is a genuine upper bound of `|θˢ − λˢ|` — the
//!    residual term is measured (T2), the roundoff term is a theorem about
//!    the eigendecomposition (T1/T3), the enclosure term is directed
//!    rounding (T5). See [`certified_width`].

/// Double-precision unit roundoff `u = 2⁻⁵³`. Single source of truth is
/// `certificate::UNIT_ROUNDOFF`; this mirrors it so the pure core stays
/// standalone.
pub const UNIT_ROUNDOFF: f64 = 1.1102230246251565e-16;

/// Conservative backward-error constant `c(n) = n³` (§4.1): `n` = Krylov
/// rank of the whitened reduced Hamiltonian.
pub fn backward_error_const(rank: usize) -> f64 {
    (rank as f64).powi(3)
}

/// Parlett's a-posteriori bound: for a computed pair `(θ, ψ)` of `H_m`,
///
/// ```text
///   |θ − λ| ≤ ‖H_m ψ − θ ψ‖ / ‖ψ‖,                          (T2, §4.3)
/// ```
///
/// the residual-norm bound on the eigenvalue error. `residual_norm` is
/// `‖H_mψ − θψ‖` measured cancellation-free from the stored Gram; `psi_norm`
/// is `‖ψ‖`. Postcondition: the returned value is non-negative and bounds
/// the true error. The bound is exact (zero) for an exact eigenpair.
pub fn parlett_bound(residual_norm: f64, psi_norm: f64) -> f64 {
    debug_assert!(residual_norm >= 0.0 && psi_norm > 0.0, "Parlett inputs");
    residual_norm / psi_norm
}

/// Rayleigh quotient `⟨ψ, Hψ⟩ / ⟨ψ, ψ⟩` — the value `θ` whose error the
/// Parlett bound certifies. Pure inner products.
pub fn rayleigh_quotient(psi_h_psi: f64, psi_psi: f64) -> f64 {
    debug_assert!(psi_psi > 0.0, "Rayleigh quotient needs a nonzero vector");
    psi_h_psi / psi_psi
}

/// The certified width `δ = ‖r‖ + c(n)·u·‖Ĝ‖ + h_O` (§4.4). Mirrors
/// `certificate::Certificate::delta()` exactly (a unit test pins the
/// agreement). `g_norm` = `‖Ĝ‖` (spectral norm of `h_proj`), `theta` = the
/// measured Ritz value (drives the enclosure `h_O = 4u·max(|θ|, 1)`).
pub fn certified_width(residual: f64, rank: usize, g_norm: f64, theta: f64) -> f64 {
    let roundoff = backward_error_const(rank) * UNIT_ROUNDOFF * g_norm.max(1.0);
    let enclosure = 4.0 * UNIT_ROUNDOFF * theta.abs().max(1.0);
    residual + roundoff + enclosure
}

/// Certified interval `[θ − δ, θ + δ]` of a Ritz value: `lo ≤ x ≤ hi`.
pub fn interval_contains(value: f64, delta: f64, x: f64) -> bool {
    debug_assert!(delta >= 0.0, "width must be non-negative");
    value - delta <= x && x <= value + delta
}

/// The T6 assembly (the certified-gap theorem of §3.4): the certified lower
/// bound of `λ₁(H_m) − λ₀(H_m)`.
///
/// ```text
///   lo = θᵒ₀ − θᵉ₀ − (δᵒ + δᵉ)
/// ```
///
/// Preconditions: `θᵉ₀` (resp. `θᵒ₀`) is the lowest Ritz value of the
/// even (odd) parity sector, and the sectors are disjoint ([`parities_disjoint`]).
/// Postcondition: `lo` is a lower bound of the spectral gap of `H_m`.
pub fn certified_gap_lower_bound(theta_o: f64, theta_e: f64, delta_o: f64, delta_e: f64) -> f64 {
    theta_o - theta_e - (delta_o + delta_e)
}

/// Certified interval of the gap: `[lo, hi] = [θᵒ₀ − θᵉ₀ − (δᵒ+δᵉ),
/// θᵒ₀ − θᵉ₀ + (δᵒ+δᵉ)]` — the enclosure of `λ₁(H_m) − λ₀(H_m)`.
pub fn gap_interval(theta_o: f64, theta_e: f64, delta_o: f64, delta_e: f64) -> (f64, f64) {
    let gap = theta_o - theta_e;
    let width = delta_o + delta_e;
    (gap - width, gap + width)
}

/// The stopping rule of §3.3: the certificate proves a mass gap for `H_m`
/// exactly when the certified lower bound is strictly positive. When this
/// predicate is true the truncated Hamiltonian has a proof-carrying gap
/// `≥ lo`; when false the certificate claims nothing.
pub fn gap_certified_positive(lo: f64) -> bool {
    lo > 0.0
}

/// Reflection-sector disjointness precondition (ChapterParity / §3.3 item 1):
/// the gauge-fixed QYM reflection is an exact symmetry of `H_m`, so pure-R
/// Krylov starts remain in disjoint invariant sectors. Lattice occupation
/// parity is not part of this contract. The runtime witness is the maximal
/// mutual overlap of retained chain vectors.
pub fn parities_disjoint(max_chain_overlap: f64, tol: f64) -> bool {
    debug_assert!(max_chain_overlap >= 0.0 && tol >= 0.0, "overlap inputs");
    max_chain_overlap < tol
}

/// Legacy comparison-model vacuum sanity. This predicate is retained only
/// for generic callers and historical cross-benchmarks; it is not a
/// precondition of the gauge-fixed nested-Fock QYM mass-gap contract. For the
/// actual theory, the one-particle Hamiltonian is shifted if needed for
/// positivity and then enclosed by outer creation on the left and outer
/// annihilation on the right, so the outer vacuum is the exact ground.
pub fn even_sector_is_vacuum(even_ground: f64, tol: f64) -> bool {
    even_ground.abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::Certificate;

    #[test]
    fn parlett_bound_holds_on_explicit_matrix() {
        // H = [[2, 1], [1, 2]]: exact eigenpairs (3, u₊ = (1,1)/√2) and
        // (1, u₋ = (1,−1)/√2). A slightly-off vector must satisfy
        // |θ' − λ| ≤ ‖Hψ' − θ'ψ'‖ / ‖ψ'‖ — the bound the certificate uses.
        let h = |v: (f64, f64)| -> (f64, f64) {
            (2.0 * v.0 + v.1, v.0 + 2.0 * v.1)
        };
        let psi = (1.0, 0.97); // near u₊, not exact
        let psi_psi = psi.0 * psi.0 + psi.1 * psi.1;
        let h_psi = h(psi);
        let psi_h_psi = psi.0 * h_psi.0 + psi.1 * h_psi.1;
        let theta = rayleigh_quotient(psi_h_psi, psi_psi);
        let res = (
            h_psi.0 - theta * psi.0,
            h_psi.1 - theta * psi.1,
        );
        let res_norm = (res.0 * res.0 + res.1 * res.1).sqrt();
        let bound = parlett_bound(res_norm, psi_psi.sqrt());
        let true_err = (theta - 3.0).abs();
        assert!(
            true_err <= bound + 1e-15,
            "Parlett: |θ−λ| = {true_err} must be ≤ ‖r‖/‖ψ‖ = {bound}"
        );
        // Exact eigenpair: the bound is zero.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let h_psi = h((s, s));
        let psi_h_psi = s * h_psi.0 + s * h_psi.1;
        let theta = rayleigh_quotient(psi_h_psi, 1.0);
        let res = (h_psi.0 - theta * s, h_psi.1 - theta * s);
        let bound = parlett_bound((res.0 * res.0 + res.1 * res.1).sqrt(), 1.0);
        assert!(bound < 1e-14, "exact eigenpair must have zero residual bound");
    }

    #[test]
    fn certified_width_matches_certificate_delta() {
        // The pure spec must agree with the library Certificate exactly
        // (single source of truth: certificate.rs).
        let residual: f64 = 1.23e-6;
        let rank = 4;
        let g_norm: f64 = 5.0;
        let theta: f64 = 1.98;
        let cert = Certificate::new(theta, residual, backward_error_const(rank) * UNIT_ROUNDOFF * g_norm.max(1.0), 4.0 * UNIT_ROUNDOFF * theta.abs().max(1.0));
        let spec = certified_width(residual, rank, g_norm, theta);
        assert!((spec - cert.delta()).abs() < 1e-20, "spec {spec} vs certificate {}", cert.delta());
    }

    #[test]
    fn gap_assembly_and_interval_contracts() {
        // Synthetic sector certificates: θᵉ = 0 (vacuum), θᵒ = 2, widths 1e-6.
        let theta_e = 0.0;
        let theta_o = 2.0;
        let d_e = 1e-6;
        let d_o = 1e-6;
        let lo = certified_gap_lower_bound(theta_o, theta_e, d_o, d_e);
        let (glo, ghi) = gap_interval(theta_o, theta_e, d_o, d_e);
        assert!((lo - (2.0 - 2e-6)).abs() < 1e-15, "T6 lower bound");
        assert!((glo - lo).abs() < 1e-15 && (ghi - (2.0 + 2e-6)).abs() < 1e-15);
        // The measured gap lies inside its own certified interval.
        assert!(interval_contains(theta_o - theta_e, d_o + d_e, theta_o - theta_e));
        // The stopping rule fires only when lo > 0.
        assert!(gap_certified_positive(lo));
        assert!(!gap_certified_positive(-1.0));
        // Interval containment is exact at the endpoints.
        assert!(interval_contains(theta_o, d_o, theta_o + d_o));
        assert!(!interval_contains(theta_o, d_o, theta_o + d_o + 1e-12));
    }

    #[test]
    fn parity_and_vacuum_preconditions() {
        assert!(parities_disjoint(0.0, 1e-8));
        assert!(parities_disjoint(1e-12, 1e-8));
        assert!(!parities_disjoint(0.5, 1e-8));
        assert!(even_sector_is_vacuum(1e-9, 1e-6));
        assert!(!even_sector_is_vacuum(0.5, 1e-6));
    }
}
