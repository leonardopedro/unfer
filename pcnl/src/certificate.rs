//! D0/D1 — the machine-reviewable claim record and the first reference
//! certificate.
//!
//! `DEVELOPMENT_PLAN.md` D0 asks for a per-run claim record that names the
//! target kind, prior, likelihood convention, temperature, chart/basis,
//! dimension and rank, proposal, scan order, adaptation freeze, precision and
//! observable — "do not encode unproved estimates as default values". D1 asks
//! for the first explicit evidence/minorization certificate: for a target
//! `π ∝ exp(−Ψ) μ0`, a verified region with prior mass `a` and `Ψ ≤ b` gives
//! `Z_lo = a·exp(−b) > 0`, and a mixture with weight `ω` between the
//! independence kernel and a same-target local kernel has coefficient
//! `c = ω·Z` when `M = 1` (categorical nonnegative likelihood), or
//! `c = ω·Z_lo/M` in general.
//!
//! The module deliberately separates three identities that are easy to
//! conflate:
//!
//! * [`TargetIdentity`] — *what* is sampled (prior, likelihood, temperature,
//!   chart, rank, precision);
//! * [`KernelIdentity`] — *how* it is moved (proposal, scan order, adaptation
//!   freeze);
//! * [`Certificate`] — *what is proved* about that pair, with its constants.
//!
//! Changing only the frozen proposal must change the kernel identity and not
//! the target identity; changing the prior, temperature or chart must change
//! the target identity. That is the D0 acceptance test, not a comment.

use crate::manifest::Fnv;

/// What a run samples. Every field is part of the identity; the hash is
/// FNV-1a over the same fields, so two configurations that disagree textually
/// but denote the same target still agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetIdentity {
    pub kind: &'static str,
    pub prior: String,
    pub likelihood: String,
    pub temperature: String,
    pub chart: String,
    pub dimension: usize,
    pub rank: usize,
    pub precision: &'static str,
    pub readout: String,
}

impl TargetIdentity {
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(self.kind.as_bytes());
        h.write(self.prior.as_bytes());
        h.write(self.likelihood.as_bytes());
        h.write(self.temperature.as_bytes());
        h.write(self.chart.as_bytes());
        h.write(&self.dimension.to_le_bytes());
        h.write(&self.rank.to_le_bytes());
        h.write(self.precision.as_bytes());
        h.write(self.readout.as_bytes());
        h.finish()
    }
}

/// How a target is moved. A frozen proposal is a tuning decision, not a target
/// change — unless it is applied after the freeze, which the `frozen` field
/// records explicitly rather than leaving to a run log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelIdentity {
    pub proposal: String,
    pub scan_order: &'static str,
    pub adaptation_frozen: bool,
    pub freezing_point: Option<usize>,
}

impl KernelIdentity {
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(self.proposal.as_bytes());
        h.write(self.scan_order.as_bytes());
        h.write(&[u8::from(self.adaptation_frozen)]);
        h.write(&self.freezing_point.unwrap_or(usize::MAX).to_le_bytes());
        h.finish()
    }
}

/// D0's per-run record. `open_assumptions` is the honest field: a certificate
/// is only as strong as the assumptions it does not discharge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimRecord {
    pub source_revision: String,
    pub target: TargetIdentity,
    pub kernel: KernelIdentity,
    pub proposal: String,
    pub evidence: String,
    pub open_assumptions: Vec<String>,
}

impl ClaimRecord {
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(self.source_revision.as_bytes());
        h.write(&self.target.hash().to_le_bytes());
        h.write(&self.kernel.hash().to_le_bytes());
        h.write(self.proposal.as_bytes());
        h.write(self.evidence.as_bytes());
        for a in &self.open_assumptions {
            h.write(a.as_bytes());
            h.write(b"\0");
        }
        h.finish()
    }
}

/// A verified region in prior coordinates: `μ0(A) = prior_mass` and
/// `Ψ ≤ potential_ceiling` on `A`.
///
/// The bound is only valid for a region actually verified by the caller
/// (quadrature with tail enclosure or an analytic likelihood/ball estimate);
/// this type records the numbers and does not itself establish them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifiedRegion {
    pub prior_mass: f64,
    pub potential_ceiling: f64,
}

impl VerifiedRegion {
    pub fn new(prior_mass: f64, potential_ceiling: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&prior_mass) && prior_mass.is_finite(),
            "prior mass must be a probability"
        );
        assert!(
            potential_ceiling.is_finite(),
            "a verified potential ceiling must be finite"
        );
        Self {
            prior_mass,
            potential_ceiling,
        }
    }

    /// `Z_lo = a·exp(−b) > 0`.
    pub fn evidence_lower_bound(&self) -> f64 {
        self.prior_mass * (-self.potential_ceiling).exp()
    }
}

/// The mixture certificate `K = ω K_ind + (1−ω) K_local`, with `K_ind` the
/// independence kernel with proposal `μ0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Certificate {
    pub omega: f64,
    pub region: VerifiedRegion,
    /// A proved ceiling `M ≥ sup L`; `1` for a categorical nonnegative
    /// likelihood. Only a proved value belongs here.
    pub likelihood_ceiling: f64,
}

impl Certificate {
    pub fn new(omega: f64, region: VerifiedRegion, likelihood_ceiling: f64) -> Self {
        assert!(omega > 0.0 && omega <= 1.0, "omega must be in (0, 1]");
        assert!(
            likelihood_ceiling >= 1.0,
            "the ceiling must dominate a unit likelihood"
        );
        Self {
            omega,
            region,
            likelihood_ceiling,
        }
    }

    /// `c = ω·Z_lo/M`, the minorization coefficient of the mixture kernel.
    pub fn minorization(&self) -> f64 {
        self.omega * self.region.evidence_lower_bound() / self.likelihood_ceiling
    }

    /// `1 − c`: the absolute centred-`L²` gap implied by minorization.
    pub fn gap(&self) -> f64 {
        1.0 - self.minorization()
    }

    /// `(1−c)^n` — the coarse TV upper bound after `n` mixture steps. A rate
    /// certificate can be published even when it predicts an impractical
    /// budget; "proper and geometrically convergent" is separate from
    /// "efficient at the requested tolerance".
    pub fn tv_after(&self, n: usize) -> f64 {
        (1.0 - self.minorization()).powi(n as i32)
    }

    /// The certificate is only meaningful as a contraction when `0 < c ≤ 1`.
    pub fn is_contracting(&self) -> bool {
        let c = self.minorization();
        c > 0.0 && c <= 1.0
    }
}

/// The analytic evidence of the Gaussian informed block
/// `π(z) ∝ N(0,I_r) exp(−(λ/2)‖z−c‖²)`, used as an independent check on
/// [`VerifiedRegion`]s (the region bound is deliberately loose).
///
/// `Z = (1+λ)^{−r/2} exp(−λ‖c‖²/(2(1+λ)))`.
pub fn gaussian_evidence(r: usize, lambda: f64, center_norm_sq: f64) -> f64 {
    (1.0 + lambda).powf(-(r as f64) / 2.0)
        * (-lambda * center_norm_sq / (2.0 * (1.0 + lambda))).exp()
}
