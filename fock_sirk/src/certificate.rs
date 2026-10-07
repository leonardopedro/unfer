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
use nested_fock_algebra::{Hamiltonian, QuantumState};
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
}
