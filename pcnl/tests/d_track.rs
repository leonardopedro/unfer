//! Acceptance checks for D3, D4, D5 and the E5 head-chart chain.
//!
//! D3's canaries are analytic (Euler–Langevin on a Gaussian), D4's benchmark is
//! linear-Gaussian with a closed-form posterior, and D5's repair is the
//! trajectory-clustered paired estimator. The E5 test runs an actual exact
//! MH-pCN chain on a frozen head chart through `SubspaceTarget`.

use pcnl::approx_pricing::{
    ErrorBudget, exact_variance, noise_inflated_variance, simulate, stability_limit,
    stationary_variance, variance_bias_factor,
};
use pcnl::diag::movement;
use pcnl::dt_arm::DtCell;
use pcnl::dt_subspace::audit_posterior_draws;
use pcnl::edge0_campaign::{CampaignConfig, build_samples};
use pcnl::edge0_potential::{
    Edge0Potential, HeadChartReference, chi_square_cdf, head_chart_enclosure_1d,
    head_chart_quadrature_1d, reference_theta,
};
use pcnl::evidence_enclosure::{TinyLogistic, exp_enclose};
use pcnl::linear_gaussian::LinearGaussian;
use pcnl::operator_cert::{GaussianOperatorCertificate, missing_dt_estimate};
use pcnl::policy_readout::{paired_delta, score_steps, trajectory_clustered_delta};
use pcnl::readout::Rule;
use pcnl::subspace::{AffineSubspace, Basis};
use pcnl::target::{SubspaceTarget, ZPotential, mh_pcn_z};
use rand::SeedableRng;

fn mean_sq(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64
}

#[test]
fn d3_euler_langevin_canaries_match_closed_forms() {
    let lambda = 2.0;
    let h = 0.2;
    let exact = exact_variance(lambda);
    assert!((exact - 0.5).abs() < 1e-15);

    // Stationary variance: 2h/(1−(1−hλ)²) = 0.4/0.64 = 0.625.
    let v = stationary_variance(lambda, h);
    assert!((v - 0.625).abs() < 1e-15, "v = {v}");
    assert!((v / exact - variance_bias_factor(lambda, h)).abs() < 1e-15);
    assert!(
        v > exact,
        "the discretisation inflates the invariant variance"
    );

    // The canary is measured, not just stated.
    let draws = simulate(lambda, h, 0.0, 40_000, 1_000, 11);
    let measured = mean_sq(&draws);
    assert!(
        (measured - v).abs() / v < 0.05,
        "simulated variance {measured} vs analytic {v}"
    );

    // Outside the stability range the variance is infinite, not large-but-finite.
    assert_eq!(stability_limit(lambda), 1.0);
    assert!(stationary_variance(lambda, 0.999).is_finite());
    assert!(stationary_variance(lambda, 1.0).is_infinite());
    assert!(stationary_variance(lambda, 1.01).is_infinite());

    // Independent gradient-noise inflation is additive in the variance.
    let noisy = noise_inflated_variance(lambda, h, 1.0);
    assert!(noisy > v);
    let draws = simulate(lambda, h, 1.0, 40_000, 1_000, 12);
    let measured = mean_sq(&draws);
    assert!(
        (measured - noisy).abs() / noisy < 0.06,
        "noise-inflated variance {measured} vs analytic {noisy}"
    );

    // The declared error budget records the bias and keeps its open items.
    let budget = ErrorBudget::gaussian(lambda, h, 1.0);
    assert!(budget.is_stable());
    assert!((budget.variance_bias - (v - exact)).abs() < 1e-15);
    assert!(budget.noise_bias > 0.0);
    assert!(
        !budget.open.is_empty(),
        "unproved estimates must stay visible"
    );
    let unstable = ErrorBudget::gaussian(lambda, 1.01, 0.0);
    assert!(!unstable.is_stable());
    assert_ne!(budget.hash(), unstable.hash());
}

#[test]
fn d4_coupled_benchmark_separates_reduction_error_from_sampler_error() {
    // No coupling: the slice is the truth, so the reduction error is exactly 0.
    let decoupled = LinearGaussian::new(3, vec![0.0, 0.0, 0.0], 0.5);
    assert_eq!(decoupled.reduction_error(1.0), 0.0);
    assert_eq!(
        decoupled.full_posterior(1.0),
        decoupled.slice_posterior(1.0)
    );

    // Coupling makes the frozen complement a *restricted model*.
    let coupled = LinearGaussian::new(3, vec![1.0, 1.0, 0.0], 0.5);
    let error = coupled.reduction_error(1.0);
    assert!(error > 1e-3, "the coupled slice must differ: {error}");
    let full = coupled.full_posterior(1.0);
    let slice = coupled.slice_posterior(1.0);
    assert!(
        full.var > slice.var,
        "marginalising a coupled complement must widen the informed posterior"
    );
    // The informed marginal *is* the full posterior (w is integrated, not fixed).
    assert_eq!(coupled.informed_marginal(1.0), full);

    // Complement conditional mean has the closed form b(y−z)/(σ²+‖b‖²).
    let z = 0.25;
    let y = 1.0;
    let t = coupled.sigma2 + coupled.coupling_norm_sq();
    let m = coupled.complement_mean(z, y);
    for (i, mi) in m.iter().enumerate() {
        assert!((mi - coupled.b[i] * (y - z) / t).abs() < 1e-15);
    }

    // The predictive integrates the joint posterior; a factorized treatment
    // drops the z↔bᵀw covariance and reports a different variance.
    let joint = coupled.predictive(1.0);
    let factorized = coupled.predictive_factorized(1.0);
    assert!((joint.mean - factorized.mean).abs() < 1e-12);
    assert!(
        (joint.var - factorized.var).abs() > 1e-3,
        "factorized variance {} vs joint {}",
        factorized.var,
        joint.var
    );

    // Monte Carlo check of the joint predictive (z then w | z,y).
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    use rand::Rng;
    let mut vals = Vec::with_capacity(60_000);
    let post = coupled.full_posterior(1.0);
    for _ in 0..60_000 {
        let g = |rng: &mut rand::rngs::StdRng| {
            let u1 = 1.0 - rng.random::<f64>();
            let u2 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let z = post.mean + post.var.sqrt() * g(&mut rng);
        let xi: Vec<f64> = (0..coupled.n).map(|_| g(&mut rng)).collect();
        let mean_w = coupled.complement_mean(z, 1.0);
        let dot: f64 = coupled.b.iter().zip(&xi).map(|(b, x)| b * x).sum();
        // Rank-one factor for Cov = I − bbᵀ/t: choose c with 2c − c²s = 1/t.
        let s = coupled.coupling_norm_sq();
        let c = if s > 0.0 {
            (1.0 - (coupled.sigma2 / t).sqrt()) / s
        } else {
            0.0
        };
        let w: Vec<f64> = (0..coupled.n)
            .map(|i| mean_w[i] + xi[i] - coupled.b[i] * dot * c)
            .collect();
        let bw: f64 = coupled.b.iter().zip(&w).map(|(b, v)| b * v).sum();
        vals.push(z + bw + coupled.sigma2.sqrt() * g(&mut rng));
    }
    let mc_mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mc_var = vals.iter().map(|v| (v - mc_mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    assert!(
        (mc_mean - joint.mean).abs() < 0.01,
        "predictive mean {mc_mean} vs analytic {}",
        joint.mean
    );
    assert!(
        (mc_var - joint.var).abs() / joint.var < 0.02,
        "predictive variance {mc_var} vs analytic {}",
        joint.var
    );
}

#[test]
fn d5_trajectory_clustered_delta_repairs_step_independence() {
    // 200 steps, 20 trajectories of 10 steps.
    let n = 200usize;
    let traj: Vec<usize> = (0..n).map(|i| i / 10).collect();
    // The label is fixed so the paired difference is *constant inside each
    // trajectory* and varies only across trajectories — the dependence the
    // step-independent estimator cannot see.
    let targets: Vec<usize> = vec![0; n];
    let a_probs: Vec<Vec<f64>> = (0..n).map(|_| vec![0.5, 0.5]).collect();
    let b_probs: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let g = i / 10;
            let tilt = 0.05 * ((g % 5) as f64);
            vec![0.5 + tilt, 0.5 - tilt]
        })
        .collect();
    let a = score_steps(&a_probs, &targets);
    let b = score_steps(&b_probs, &targets);
    let naive = paired_delta(&a, &b, Rule::Brier);
    let clustered = trajectory_clustered_delta(&a, &b, Rule::Brier, &traj);
    assert_eq!(clustered.steps, n);
    assert_eq!(clustered.trajectories, 20);
    assert_eq!(clustered.max_trajectory_len, 10);
    assert!((clustered.mean - naive.0).abs() < 1e-15);
    assert!(
        clustered.se > 1.5 * naive.1,
        "clustered SE {} must exceed the step-independent {}",
        clustered.se,
        naive.1
    );

    // With singleton clusters the two estimators coincide (the special case).
    let singleton: Vec<usize> = (0..n).collect();
    let single = trajectory_clustered_delta(&a, &b, Rule::Brier, &singleton);
    assert_eq!(single.trajectories, n);
    assert!((single.se - naive.1).abs() < 1e-12);

    // A frozen trace is flagged by the diagnostics rather than read as mixing;
    // the clustered estimator's uncertainty is a separate axis, not a licence.
    let frozen = vec![vec![0.0f64; 50], vec![0.0f64; 50]];
    let mov = movement(&frozen);
    assert!(mov.is_frozen() && mov.is_degenerate());
    assert!(clustered.interval(1.96).0.is_finite());
}

#[test]
fn d5_draw_subset_audit_compares_the_three_decoders_at_fixed_histories() {
    let mut cell = DtCell::new(5);
    cell.pretrain(12, 0.02, 0.9).unwrap();
    let anchor = cell.anchor.clone().unwrap();
    let draws: Vec<candle_core::Tensor> = (0..4)
        .map(|i| anchor.affine(1.0 + 0.01 * i as f64, 0.0).unwrap())
        .collect();
    let audit = audit_posterior_draws(&cell.env, &cell, anchor, &draws, &[2, 4], 8, 99);
    assert_eq!(audit.subsets.len(), 2);
    assert_eq!(audit.baseline_used, 4);
    assert_eq!(audit.subset_vs_full_bma.len(), 2);
    for row in &audit.subsets {
        assert_eq!(row.used, row.requested);
        assert_eq!(row.bma_sample.returns.len(), 8);
        assert_eq!(row.episode_draw.returns.len(), 8);
        assert!(row.bma_sample.mean.is_finite());
        assert!(row.bma_argmax.mean.is_finite());
        assert!(row.episode_draw.mean.is_finite());
        assert!(row.episode_vs_bma.mcse.is_finite());
    }
    assert_eq!(audit.subsets[0].used, 2);
    assert!(audit.subsets[1].used > audit.subsets[0].used);
}

#[test]
fn d6_canary_certificate_is_full_space_and_the_dt_scope_stays_open() {
    let cert = GaussianOperatorCertificate::certify(2.0, 0.2, 64).expect("stable canary");
    assert!(cert.invariance_holds(1e-12));
    assert!((cert.contraction - 0.6).abs() < 1e-15);
    assert!((cert.variance - 0.625).abs() < 1e-15);
    assert!((cert.gap - 0.4).abs() < 1e-15);
    assert!((cert.eigenvalues[0] - 1.0).abs() < 1e-15);
    assert!((cert.eigenvalues[1] - 0.6).abs() < 1e-15);
    assert!(cert.truncation_covers(1e-12));
    assert!(cert.tail_mass > 0.0 && cert.tail_mass.is_finite());

    // Outside 0 < a < 1 the same statement is not available.
    assert!(GaussianOperatorCertificate::certify(2.0, 1.0, 64).is_none());
    assert!(GaussianOperatorCertificate::certify(2.0, 2.0, 64).is_none());
    assert!(GaussianOperatorCertificate::certify(2.0, 0.2, 0).is_none());

    // The DT operator has none of the four required bounds; the record says so.
    let scope = missing_dt_estimate();
    assert!(!scope.certified());
    assert_eq!(scope.missing().len(), 4);
    assert!(
        scope
            .missing()
            .contains(&"certified retained-block tail bound")
    );
}

#[test]
fn e5_head_chart_runs_an_exact_mh_pcn_chain() {
    let mut cfg = CampaignConfig::default();
    cfg.updates = 0;
    cfg.warm_start = 0;
    cfg.worlds = [3, 1, 1, 1];
    let samples = build_samples(&cfg);
    let model = cfg.model.clone();
    let potential = Edge0Potential::head_only(model.clone(), samples);

    let p = model.flat_size();
    let layout = model.layout();
    let idx = layout
        .iter()
        .find(|s| s.name == "policy_b")
        .expect("policy head")
        .start;
    let mut cols = vec![0f32; p];
    cols[idx] = 1.0;
    let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
    let theta_ref = reference_theta(&model, 9)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p]).expect("chart");
    let target = SubspaceTarget::with_potential(subspace, 1.0, potential);

    let mut z = vec![0f32];
    let mut phi = target.phi(&z);
    assert!(phi.is_finite(), "the chart's potential must be finite");
    let (lik, prior) = target.phi_parts(&z);
    assert!(lik.is_finite() && prior.is_finite());
    assert!((phi - (lik + prior)).abs() < 1e-5);

    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut accepted = 0usize;
    for _ in 0..120 {
        let step = mh_pcn_z(&target, &mut z, phi, 0.03, &mut rng);
        phi = step.phi;
        accepted += usize::from(step.accepted);
    }
    assert!(phi.is_finite());
    assert!(
        target.evaluations() >= 120,
        "the chain evaluates the target"
    );
    assert!(accepted > 0, "the head chart accepts at least one move");
    assert!(z.len() == 1);
}

#[test]
fn e5_reference_record_is_identified_and_its_rate_is_honest() {
    let mut cfg = CampaignConfig::default();
    cfg.updates = 0;
    cfg.warm_start = 0;
    cfg.worlds = [3, 1, 1, 1];
    let samples = build_samples(&cfg);
    let model = cfg.model.clone();
    let potential = Edge0Potential::head_only(model.clone(), samples);
    let reference = HeadChartReference::new(&potential);

    // The chart is a proper subset of the flat parameters.
    assert!(reference.trainable_coordinates() > 0);
    assert!(reference.trainable_coordinates() < reference.parameters());

    // Temperature is part of what is sampled; the chart dimension is the
    // chart's, not `P`.
    let t1 = reference.target_identity(1.0);
    let t4 = reference.target_identity(4.0);
    assert_eq!(t1.dimension, reference.trainable_coordinates());
    assert_ne!(t1.hash(), t4.hash());
    assert_eq!(t1.hash(), reference.target_identity(1.0).hash());
    assert_eq!(t1.kind, "edge0-head-chart-categorical-nll");

    // The kernel identity responds to the mixture, not to `lambda`.
    let k = reference.kernel_identity(0.03, 0.5);
    assert_eq!(k.hash(), reference.kernel_identity(0.03, 0.5).hash());
    assert_ne!(k.hash(), reference.kernel_identity(0.03, 0.25).hash());
    assert!(k.proposal.contains("beta=1"));

    let claim = reference.claim_record(1.0, "test", 0.03, 0.5);
    assert_eq!(claim.target.hash(), t1.hash());
    assert_eq!(claim.kernel.hash(), k.hash());
    assert!(!claim.open_assumptions.is_empty());

    // Exact chi-square prior mass: three independent closed forms.
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    // P(|Z| ≤ 1) = erf(1/√2).
    assert!(close(chi_square_cdf(1, 1.0), 0.682_689_492_137_086));
    assert!(close(chi_square_cdf(2, 2.0), 1.0 - (-2.0f64).exp()));
    assert!(close(chi_square_cdf(4, 2.0), 1.0 - 3.0 * (-2.0f64).exp()));
    assert_eq!(chi_square_cdf(3, 0.0), 0.0);
    assert_eq!(reference.ball_prior_mass(0.0), 0.0);

    // The clamp ceiling is sound (`M = 1`, whole prior) and useless at
    // lambda = 8: `c` is far below any usable rate. The record says so.
    let clamp = reference.clamp_certificate(8.0, 0.5);
    assert_eq!(clamp.likelihood_ceiling, 1.0);
    assert_eq!(clamp.region.prior_mass, 1.0);
    assert!(clamp.minorization() < 1e-60, "c = {}", clamp.minorization());
    assert!(clamp.tv_after(1) > 0.999);

    // A ball with an externally verified ceiling does contract; the prior mass
    // is the exact chi-square value, not a fitted one.
    let ball = reference.ball_certificate(1.0, 3.0, 0.5);
    assert!(ball.is_contracting());
    assert!(close(
        ball.region.prior_mass,
        reference.ball_prior_mass(1.0)
    ));
}

#[test]
fn d1_witness_evidence_enclosure_contains_the_analytic_value() {
    // Rank 2 is separable: e^{−Ψ} = σ(x)·σ(−y) and ∫σφ = 1/2, so `Z = 1/4`
    // exactly — an independently computed fixture, not a finer quadrature.
    let target = TinyLogistic::rank_two(1.0);
    let coarse = target.evidence_enclosure(4.0, 256);
    let fine = target.evidence_enclosure(4.0, 512);
    assert!(
        coarse.width() < 0.08,
        "the rank-2 enclosure is still informative: width {}",
        coarse.width()
    );
    assert!(
        coarse.contains(0.25),
        "Z ∈ [{}, {}] must contain 0.25",
        coarse.z_lo,
        coarse.z_hi
    );
    assert!(fine.contains(0.25));
    assert!(coarse.z_lo <= fine.z_lo && fine.z_hi <= coarse.z_hi);
    assert!(fine.width() < coarse.width());
    assert!(coarse.tail_upper < 1e-3);

    // The certified constant is a lower bound on `ω·Z`, and it is small.
    let cert = target.certificate(4.0, 0.5).expect("positive region");
    assert!(cert.minorization() > 0.0 && cert.is_contracting());
    assert!(cert.minorization() <= 0.5 * 0.25 + 1e-15);
    assert!(cert.tv_after(1) > 0.9, "the practical rate is slow");
    // A box smaller than the tail bound can certify (Mills' union bound gives
    // `tail = 1` there) has no positive region mass, so no certificate is
    // claimed for it.
    assert!(target.certificate(1.0, 0.5).is_some());
    assert!(target.certificate(1e-9, 0.5).is_none());

    // Rank 1 has no closed-form `Z`, but the enclosure still nests.
    let r1 = TinyLogistic::rank_one(1.0);
    let c1 = r1.evidence_enclosure(6.0, 128);
    let f1 = r1.evidence_enclosure(6.0, 512);
    assert!(c1.z_lo > 0.0);
    assert!(c1.z_lo <= f1.z_lo && f1.z_hi <= c1.z_hi);
}

#[test]
fn d1_interval_arithmetic_encloses_exp_and_the_evidence_is_certified() {
    // The enclosure is not a libm assumption: the reduced Taylor series has to
    // bracket the platform `exp` on every argument the witness uses.
    for x in [-32.0, -8.0, -1.0, -0.3, 0.0, 0.3, 1.0, 8.0, 20.0, 60.0] {
        let iv = exp_enclose(x);
        let truth = x.exp();
        assert!(
            iv.lo <= truth && truth <= iv.hi,
            "exp({x}) = {truth} outside [{}, {}]",
            iv.lo,
            iv.hi
        );
        assert!(iv.hi - iv.lo < 1e-13 * truth.max(1.0));
    }

    // Rank 2 has `Z = 1/4` exactly; the interval sum must contain it with no
    // declared rounding envelope, and the truncation/tail bounds still hold.
    let target = TinyLogistic::rank_two(1.0);
    let e = target.evidence_enclosure(4.0, 256);
    assert_eq!(e.rounding_envelope, 0.0);
    assert!(e.contains(0.25), "[{}, {}]", e.z_lo, e.z_hi);
    assert!(e.z_lo > 0.0);
    assert!(e.width() < 0.2);
    assert!(e.quadrature_error > 0.0 && e.tail_upper < 1e-3);

    let fine = target.evidence_enclosure(4.0, 512);
    assert!(fine.contains(0.25));
    assert!(fine.width() < e.width());
}

#[test]
fn d1_witness_predictive_enclosure_contains_the_chain_estimate() {
    let target = TinyLogistic::rank_one(1.0);
    let enclosure = target.predictive_enclosure(6.0, 512, 1.0);
    assert!(enclosure.lo > 0.0 && enclosure.hi < 1.0);
    assert!(enclosure.hi - enclosure.lo < 0.4);
    // The numerator is interval arithmetic too: no declared `f64` envelope.
    assert_eq!(enclosure.rounding_envelope, 0.0);

    // Exact MH-pCN against the same total potential the enclosure integrates.
    let mut z = vec![0f32];
    let mut phi = target.phi(&z);
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut sum = 0.0;
    let mut count = 0usize;
    for step in 0..60_000 {
        let s = mh_pcn_z(&target, &mut z, phi, 0.5, &mut rng);
        phi = s.phi;
        if step >= 2_000 {
            let x = f64::from(z[0]);
            sum += 1.0 / (1.0 + x * x);
            count += 1;
        }
    }
    let estimate = sum / count as f64;
    assert!(
        enclosure.contains(estimate),
        "chain {estimate} outside [{}, {}]",
        enclosure.lo,
        enclosure.hi
    );

    // A finer box shrinks the interval but still contains the same estimate.
    let finer = target.predictive_enclosure(6.0, 2048, 1.0);
    assert!(finer.hi - finer.lo < enclosure.hi - enclosure.lo);
    assert!(finer.contains(estimate));
}

#[test]
fn e5_single_question_chart_matches_quadrature_reference() {
    let mut cfg = CampaignConfig::default();
    cfg.updates = 0;
    cfg.warm_start = 0;
    cfg.worlds = [2, 1, 1, 1];
    let samples = build_samples(&cfg);
    let sample = samples
        .iter()
        .find(|s| s.split == pcnl::edge0_world::Split::Train)
        .expect("a train sample")
        .clone();
    let model = cfg.model.clone();
    let potential = Edge0Potential::head_only(model.clone(), vec![sample]);

    let p = model.flat_size();
    let layout = model.layout();
    let idx = layout
        .iter()
        .find(|s| s.name == "policy_b")
        .expect("policy head")
        .start;
    let mut cols = vec![0f32; p];
    cols[idx] = 1.0;
    let basis = Basis::new(cols, p, 1).expect("one orthonormal column");
    let theta_ref = reference_theta(&model, 9)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let subspace = AffineSubspace::new(theta_ref, basis, vec![1.0; p]).expect("chart");
    let target = SubspaceTarget::with_potential(subspace, 2.0, potential);

    let quad = head_chart_quadrature_1d(&target, 6.0, 64, 0)
        .expect("quadrature runs")
        .expect("one-dimensional chart");
    assert!(quad.evidence > 0.0 && quad.evidence.is_finite());
    assert!(quad.tail_upper < 1e-8);
    assert!(quad.grid_gap < 0.05, "coarse/fine gap {}", quad.grid_gap);
    assert!(quad.plug_in.is_finite() && (0.0..=1.0).contains(&quad.plug_in));

    // Exact MH-pCN on the same target, averaged posterior predictive.
    let mut z = vec![0f32];
    let mut phi = target.phi(&z);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut sum = 0.0;
    let mut count = 0usize;
    for step in 0..900 {
        let s = mh_pcn_z(&target, &mut z, phi, 0.25, &mut rng);
        phi = s.phi;
        if step >= 200 {
            let theta = target.subspace().lift_tensor(&z).expect("lift");
            if let Some(p) = target.potential().label_probability(&theta, 0).expect("p") {
                sum += p;
                count += 1;
            }
        }
    }
    let chain = sum / count.max(1) as f64;
    assert!(
        (chain - quad.predictive).abs() < 0.08,
        "chain predictive {chain} vs quadrature {}",
        quad.predictive
    );
    assert!(
        (quad.plug_in - quad.predictive).abs() > 1e-6,
        "the posterior average is a distinct quantity from the plug-in"
    );

    // The certified enclosure bounds the same evidence and predictive, with
    // explicit truncation and tail terms rather than a grid-gap reference.
    let enc = head_chart_enclosure_1d(&target, 6.0, 512, 0)
        .expect("enclosure runs")
        .expect("policy-direction chart is certifiable");
    assert!(
        enc.evidence_contains(quad.evidence),
        "evidence {} outside [{}, {}]",
        quad.evidence,
        enc.evidence_lo,
        enc.evidence_hi
    );
    assert!(
        enc.predictive_contains(quad.predictive),
        "predictive {} outside [{}, {}]",
        quad.predictive,
        enc.predictive_lo,
        enc.predictive_hi
    );
    assert!(enc.tail_upper < 1e-8);
    assert!(enc.quadrature_error > 0.0 && enc.psi_prime_sup > 0.0);
    assert_eq!(enc.f32_bridge, 1e-6);
    assert!(enc.evidence_lo > 0.0 && enc.predictive_hi <= 1.0);
    // A finer grid must not widen the certified evidence interval.
    let finer = head_chart_enclosure_1d(&target, 6.0, 1024, 0)
        .expect("enclosure runs")
        .expect("certifiable");
    assert!(finer.evidence_hi - finer.evidence_lo <= enc.evidence_hi - enc.evidence_lo + 1e-15);
}
