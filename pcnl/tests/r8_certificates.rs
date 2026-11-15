//! Acceptance gates D0–D2 (`DEVELOPMENT_PLAN.md`).
//!
//! * **D0** — the claim record has three separable identities: two matched
//!   configurations agree on target *and* kernel identity; changing the prior,
//!   temperature or chart changes the target identity; changing only a frozen
//!   proposal changes the kernel identity and not the target identity.
//! * **D1** — the evidence/minorization certificate is instantaneous with
//!   positive constants: `c = ω·Z_lo/M ∈ (0,1]`, and it reports its (possibly
//!   impractical) geometric rate separately from efficiency.
//! * **D2** — the product target's absolute gap is independent of complement
//!   dimension, and the negative control with a shrinking `β` does not inherit
//!   the same constant.

use pcnl::certificate::{
    Certificate, ClaimRecord, KernelIdentity, TargetIdentity, VerifiedRegion, gaussian_evidence,
};
use pcnl::dim_benchmark::ProductTarget;
use pcnl::target::ZPotential;

fn target(prior: &str, temperature: &str, chart: &str, lambda: f64) -> TargetIdentity {
    TargetIdentity {
        kind: "restricted-chart-pcn",
        prior: prior.into(),
        likelihood: format!("lambda*N*mean_nll (lambda={lambda})"),
        temperature: temperature.into(),
        chart: chart.into(),
        dimension: 6,
        rank: 2,
        precision: "f32",
        readout: "posterior-predictive".into(),
    }
}

fn kernel(proposal: &str) -> KernelIdentity {
    KernelIdentity {
        proposal: proposal.into(),
        scan_order: "informed-then-complement",
        adaptation_frozen: true,
        freezing_point: Some(2_000),
    }
}

#[test]
fn d0_matched_configurations_agree_and_changes_are_identified() {
    let a = ClaimRecord {
        source_revision: "abc123".into(),
        target: target("N(0,sigma^2 I)", "lambda=1", "rank-2 informed", 1.0),
        kernel: kernel("pcn beta=0.05"),
        proposal: "z'=sqrt(1-b^2) z + b xi".into(),
        evidence: "region bound".into(),
        open_assumptions: vec!["Gaussian tail enclosure".into()],
    };
    let b = a.clone();
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.target.hash(), b.target.hash());

    // Prior, temperature and chart each change the target identity.
    let prior_changed = TargetIdentity {
        prior: "N(0, 4 I)".into(),
        ..a.target.clone()
    };
    let temp_changed = TargetIdentity {
        temperature: "lambda=4".into(),
        ..a.target.clone()
    };
    let chart_changed = TargetIdentity {
        chart: "rank-4 informed".into(),
        ..a.target.clone()
    };
    assert_ne!(a.target.hash(), prior_changed.hash());
    assert_ne!(a.target.hash(), temp_changed.hash());
    assert_ne!(a.target.hash(), chart_changed.hash());

    // Changing only the frozen proposal moves the kernel identity, not the
    // target identity.
    let other_kernel = kernel("pcn beta=0.01");
    assert_ne!(a.kernel.hash(), other_kernel.hash());
    assert_eq!(
        a.target.hash(),
        target("N(0,sigma^2 I)", "lambda=1", "rank-2 informed", 1.0).hash()
    );

    // The whole-record hash sees a kernel-only change.
    let record = ClaimRecord {
        kernel: other_kernel,
        ..a.clone()
    };
    assert_ne!(a.hash(), record.hash());
}

#[test]
fn d1_certificate_has_positive_contracting_constants() {
    let region = VerifiedRegion::new(0.25, 2.0);
    let z_lo = region.evidence_lower_bound();
    assert!((z_lo - 0.25 * (-2.0f64).exp()).abs() < 1e-15);
    assert!(z_lo > 0.0);
    let cert = Certificate::new(0.5, region, 1.0);
    let c = cert.minorization();
    assert!((c - 0.5 * z_lo).abs() < 1e-15);
    assert!(cert.is_contracting());
    // The rate is reported even when it is practically useless.
    assert!(cert.tv_after(0) == 1.0);
    assert!(cert.tv_after(10) < cert.tv_after(5));
    assert!(cert.gap() > 0.0 && cert.gap() < 1.0);

    // A ceiling `M > 1` lowers the coefficient proportionally.
    let with_m = Certificate::new(0.5, region, 4.0);
    assert!((with_m.minorization() - c / 4.0).abs() < 1e-15);

    // The analytic Gaussian evidence is the independent check on a region
    // bound; it is positive and at most 1.
    let evidence = gaussian_evidence(2, 1.0, 0.0);
    assert!((evidence - 0.5).abs() < 1e-12); // (1+1)^(-2/2)
    let shifted = gaussian_evidence(2, 1.0, 4.0);
    assert!(shifted < evidence && shifted > 0.0);
}

#[test]
fn d2_product_gap_is_independent_of_complement_dimension() {
    let base = ProductTarget::new(2, 1, 1.0, vec![0.5, -0.5]);
    let wide = ProductTarget::new(2, 64, 1.0, vec![0.5, -0.5]);
    let beta = 0.1;
    assert!((base.informed_gap() - wide.informed_gap()).abs() < 1e-15);
    let gap = base.product_gap(beta);
    assert!((gap - wide.product_gap(beta)).abs() < 1e-15);
    assert!(gap > 0.0 && gap < 1.0);
    assert!((gap - base.informed_gap().min(ProductTarget::complement_gap(beta))).abs() < 1e-15);

    // The total potential is the informed Gaussian plus the complement prior.
    let z = [0.5f32, -0.5];
    let w: Vec<f32> = (0..64).map(|i| (i as f32 % 3.0) - 1.0).collect();
    let mut point = z.to_vec();
    point.extend_from_slice(&w);
    // At z = c the informed potential is ½‖c‖² and the likelihood factor is 1.
    let expected = 0.5 * (0.25 + 0.25) + 0.5 * w.iter().map(|x| x * x).sum::<f32>() as f64;
    assert!((wide.phi(&point) - expected).abs() < 1e-6);
    assert!(gap.is_finite());
}

#[test]
fn d2_negative_control_with_shrinking_beta_does_not_inherit_the_constant() {
    let target = ProductTarget::new(2, 8, 1.0, vec![0.0, 0.0]);
    let informed = target.informed_gap();
    let strong = target.product_gap(0.5);
    let weak = target.product_gap(0.02);
    assert!(
        weak < strong,
        "a shrinking beta must shrink the complement gap"
    );
    // The informed constant alone is not the product constant.
    assert!(strong.min(informed) == target.product_gap(0.5));
    assert!((weak - ProductTarget::complement_gap(0.02)).abs() < 1e-15);
    // And the complement gap is independent of `n`, as is the informed gap.
    let other = ProductTarget::new(2, 32, 1.0, vec![0.0, 0.0]);
    assert!((target.product_gap(0.02) - other.product_gap(0.02)).abs() < 1e-15);
}
