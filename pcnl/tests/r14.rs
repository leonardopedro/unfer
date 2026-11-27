//! R14/P0–P5 acceptance checks that are deterministic and cheap.
//!
//! The statistical gates (four dispersed chains, `R̂ < 1.01`, `ESS ≥ 400`,
//! quadrature mass agreement) live in `examples/r14_plateau_quadrature.rs`; the
//! campaign-scale comparisons live in the campaign examples.  Everything here
//! is a property that must hold for every run: gradient agreement, target
//! identity, monotone averaging, compression identities and checkpoint
//! round-trips.

use candle_core::{Device, Tensor};
use pcnl::distill::{SpreadHead, fit_softmax_student, softmax, vector_stats};
use pcnl::ensemble::{Ensemble, KneeRule, WarmEntryConfig, WarmStop, warm_entry};
use pcnl::ensemble_diag::{
    effective_rank, in_region_fraction, js_dark_knowledge_gap, kl_bernoulli, normal_quantile,
    pac_bayes_mixture_bound, pairwise_distances, pinsker_return_bound, prediction_spread,
    squared_error_ambiguity, teacher_student_fidelity, tv_distance,
};
use pcnl::manifest::fnv1a;
use pcnl::mdp::{BehaviourPolicy, Corpus, MdpEnv};
use pcnl::normalize::NormGrad;
use pcnl::rl_track::{
    ReturnAwareConfig, RlLoss, RlReturnTarget, action_distribution_diversity, linear_policy_probs,
};
use pcnl::satisficing::{
    Hinge, Satisficing, SatisficingConfig, SatisficingPotential, target_identity,
};
use pcnl::swa::{
    SwaCheckpoint, SwaKind, barrier_height, batch_average, interpolate, output_average,
    straight_line_barrier,
};
use pcnl::target::Potential;
use pcnl::vector_model::EnergyTarget;

/// `L(θ) = Σ (θ_i − c_i)² + base`, a nonnegative declared loss.
struct ShiftedQuadratic {
    center: Vec<f64>,
    base: f64,
}

impl EnergyTarget for ShiftedQuadratic {
    fn dim(&self) -> usize {
        self.center.len()
    }

    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        if theta.len() != self.center.len() || theta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut loss = self.base;
        let mut grad = Vec::with_capacity(theta.len());
        for (&t, &c) in theta.iter().zip(&self.center) {
            let d = f64::from(t) - c;
            loss += d * d;
            grad.push((2.0 * d) as f32);
        }
        Some((loss, grad))
    }
}

fn config(hinge: Hinge, alpha: f64) -> SatisficingConfig {
    SatisficingConfig {
        tau: 1.0,
        kappa: 10.0,
        alpha,
        delta: 0.3,
        hinge,
    }
}

fn fd_grad<F: Fn(&[f32]) -> Option<f64>>(f: F, theta: &[f32], h: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(theta.len());
    for i in 0..theta.len() {
        let mut plus = theta.to_vec();
        let mut minus = theta.to_vec();
        plus[i] += h as f32;
        minus[i] -= h as f32;
        out.push((f(&plus).unwrap() - f(&minus).unwrap()) / (2.0 * h));
    }
    out
}

#[test]
fn satisficing_gradient_matches_finite_differences_in_outside_and_near_plateau() {
    for hinge in [Hinge::Softplus, Hinge::Huberized] {
        for alpha in [0.0, 0.02] {
            let target =
                Satisficing::new(inner_ref(), config(hinge, alpha), "quad", "sum").expect("config");
            // Inside (L ≈ 0.25), near (L ≈ 1), outside (L ≈ 4).
            for scale in [0.5f32, 1.0, 2.0] {
                let theta = vec![0.3 + scale * 0.5, -0.2];
                let (_, grad) = target.energy_and_grad(&theta).expect("gradient");
                let fd = fd_grad(|t| target.phi(t), &theta, 1e-3);
                for (g, f) in grad.iter().zip(&fd) {
                    let diff = (f64::from(*g) - f).abs();
                    let scale = f.abs().max(1.0);
                    assert!(
                        diff <= 2e-2 * scale,
                        "hinge={:?} alpha={alpha} scale={scale}: analytic {} vs fd {}",
                        hinge,
                        g,
                        f
                    );
                }
            }
        }
    }
}

fn inner_ref() -> ShiftedQuadratic {
    ShiftedQuadratic {
        center: vec![0.3, -0.2],
        base: 0.0,
    }
}

#[test]
fn huberized_hinge_has_exactly_zero_gradient_inside_the_plateau() {
    let target =
        Satisficing::new(inner_ref(), config(Hinge::Huberized, 0.0), "quad", "sum").unwrap();
    // L = 0.25 < τ = 1, so ψ′ = 0 and ∇Φ = 0.
    let (phi, grad) = target.energy_and_grad(&[0.3, -0.2]).unwrap();
    assert_eq!(phi, 0.0);
    assert!(grad.iter().all(|g| *g == 0.0));
}

#[test]
fn target_identity_tracks_every_declared_knob_and_the_prior() {
    let c = config(Hinge::Softplus, 0.02);
    let prior = [1.0, 2.0];
    let id = target_identity(&c, "mse", "mean", &prior);
    assert_eq!(id, target_identity(&c, "mse", "mean", &prior));
    for changed in [
        SatisficingConfig { tau: 1.1, ..c },
        SatisficingConfig { kappa: 9.0, ..c },
        SatisficingConfig { alpha: 0.03, ..c },
        SatisficingConfig { delta: 0.2, ..c },
        SatisficingConfig {
            hinge: Hinge::Huberized,
            ..c
        },
    ] {
        assert_ne!(id, target_identity(&changed, "mse", "mean", &prior));
    }
    assert_ne!(id, target_identity(&c, "mse", "sum", &prior));
    assert_ne!(id, target_identity(&c, "bce", "mean", &prior));
    assert_ne!(id, target_identity(&c, "mse", "mean", &[1.0, 2.1]));
}

#[test]
fn satisficing_config_rejects_degenerate_values() {
    for bad in [
        SatisficingConfig {
            tau: 0.0,
            ..config(Hinge::Softplus, 0.0)
        },
        SatisficingConfig {
            kappa: 0.0,
            ..config(Hinge::Softplus, 0.0)
        },
        SatisficingConfig {
            alpha: -1.0,
            ..config(Hinge::Softplus, 0.0)
        },
        SatisficingConfig {
            alpha: 10.0,
            ..config(Hinge::Softplus, 0.0)
        },
        SatisficingConfig {
            delta: f64::NAN,
            ..config(Hinge::Softplus, 0.0)
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn potential_adapter_uses_the_same_phi_and_gradient() {
    // mean_loss = (1/4)Σθ², n_terms = 4, λ = 2 ⇒ L = 2Σθ².
    struct QuadPotential;
    impl Potential for QuadPotential {
        fn n_terms(&self) -> usize {
            4
        }
        fn mean_loss(&self, theta: &Tensor) -> f64 {
            let v = theta.to_vec1::<f32>().unwrap();
            v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / 4.0
        }
        fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
            let v = theta.to_vec1::<f32>().unwrap();
            let g: Vec<f32> = v.iter().map(|x| x / 2.0).collect();
            Some(Tensor::from_vec(g, v.len(), &Device::Cpu).ok()?)
        }
        fn label(&self) -> &'static str {
            "quad"
        }
    }

    let cfg = SatisficingConfig {
        tau: 0.5,
        kappa: 4.0,
        alpha: 0.1,
        delta: 0.2,
        hinge: Hinge::Softplus,
    };
    let lambda = 2.0;
    let p = SatisficingPotential::new(QuadPotential, cfg, lambda).unwrap();
    let theta = Tensor::from_vec(vec![0.6f32, -0.4], 2, &Device::Cpu).unwrap();
    let l = p.loss(&theta);
    assert!((l - 2.0 * (0.36 + 0.16)).abs() < 1e-6);
    let (expected_phi, _) = cfg.phi_and_weight(l);
    let scale = lambda * p.n_terms() as f64;
    let mean = p.mean_loss(&theta);
    assert!(
        (mean * scale - expected_phi).abs() < 1e-6,
        "{mean} * {scale}"
    );

    let grad = p
        .mean_grad_with(&theta, NormGrad::Exact)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // FD of mean_loss; the target's λn multiplier restores Φ.
    let fd = fd_grad(
        |t| {
            let tt = Tensor::from_vec(t.to_vec(), t.len(), &Device::Cpu).ok()?;
            Some(p.mean_loss(&tt))
        },
        &[0.6, -0.4],
        1e-3,
    );
    for (g, f) in grad.iter().zip(&fd) {
        assert!((f64::from(*g) - f).abs() < 2e-3, "{g} vs {f}");
    }
}

#[test]
fn nonfinite_energies_fail_explicitly() {
    struct Bad;
    impl EnergyTarget for Bad {
        fn dim(&self) -> usize {
            2
        }
        fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
            Some((if theta[0] > 0.0 { f64::NAN } else { 1.0 }, vec![0.0, 0.0]))
        }
    }
    let target = Satisficing::new(Bad, config(Hinge::Softplus, 0.0), "bad", "sum").unwrap();
    assert!(target.energy_and_grad(&[1.0, 0.0]).is_none());
    assert!(target.phi(&[1.0, 0.0]).is_none());
}

#[test]
fn swa_online_and_batch_averages_agree_and_barrier_sign_is_honest() {
    let snapshots = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let batch = batch_average(&snapshots).unwrap();
    let mut online = pcnl::swa::RunningAverage::new(2);
    for s in &snapshots {
        online.push(s).unwrap();
    }
    assert_eq!(online.mean().unwrap(), batch);
    assert_eq!(batch, vec![3.0, 4.0]);

    // Convex quadratic along the segment: no barrier.
    let convex =
        straight_line_barrier(&[-1.0], &[1.0], 20, |t| f64::from(t[0]) * f64::from(t[0])).unwrap();
    assert!(barrier_height(&convex).unwrap() <= 1e-9);
    // Double well: positive barrier between the two minima.
    let quartic = straight_line_barrier(&[-1.0], &[1.0], 20, |t| {
        let x = f64::from(t[0]);
        (x * x - 1.0) * (x * x - 1.0)
    })
    .unwrap();
    assert!(barrier_height(&quartic).unwrap() > 0.9);

    let avg = output_average(&[vec![1.0, -1.0], vec![3.0, 1.0]]).unwrap();
    assert_eq!(avg, vec![2.0, 0.0]);
    assert_eq!(interpolate(&[0.0], &[2.0], 0.25).unwrap(), vec![0.5]);
}

#[test]
fn knee_rule_detects_a_slowdown_and_ignores_fast_descent() {
    let rule = KneeRule {
        check_every: 1,
        fraction: 0.01,
        window: 3,
        epsilon: 1e-12,
    };
    let slow = [1.0, 0.995, 0.990, 0.9856, 0.9812];
    assert!(rule.detect(&slow).is_some());
    let fast = [1.0, 0.5, 0.25, 0.125];
    assert!(rule.detect(&fast).is_none());
    let bumpy = [1.0, 0.999, 0.9995, 0.9997];
    assert!(rule.detect(&bumpy).is_none());
}

#[test]
fn warm_entry_stops_at_the_declared_ceiling() {
    let target = ShiftedQuadratic {
        center: vec![0.0, 0.0],
        base: 0.0,
    };
    let cfg = WarmEntryConfig {
        max_steps: 200,
        lr: 0.1,
        tau: 0.5,
        knee: KneeRule {
            check_every: 5,
            fraction: 0.01,
            window: 3,
            epsilon: 1e-12,
        },
    };
    let warm = warm_entry(&target, &[2.0, -2.0], cfg).unwrap();
    assert_eq!(warm.stop, WarmStop::Ceiling, "trace {:?}", warm.trace);
    let loss = target.energy_and_grad(&warm.theta).unwrap().0;
    assert!(loss <= cfg.tau);
}

#[test]
fn ensemble_and_swa_round_trip_bitwise() {
    let mut ensemble = Ensemble::new("r14-test");
    for seed in [1u64, 2, 3] {
        let w = vec![seed as f32, -(seed as f32), 0.5];
        ensemble
            .push(pcnl::ensemble::Member {
                weights: w.clone(),
                seed,
                warm_stop: WarmStop::Knee,
                warm_steps: 10,
                train_loss: 1.0 + seed as f64,
                hinge: 0.0,
            })
            .unwrap();
    }
    let swa = SwaCheckpoint::new(
        SwaKind::PcnlWithinBasin,
        0,
        3,
        5,
        &[
            vec![0.0f32, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
        ],
    )
    .unwrap();
    ensemble.swa_centers.push(swa);
    let path = std::env::temp_dir().join(format!("pcnl-r14-ensemble-{}.json", std::process::id()));
    ensemble.save(&path).unwrap();
    let loaded = Ensemble::load(&path).unwrap();
    assert_eq!(loaded.to_json().unwrap(), ensemble.to_json().unwrap());
    for (a, b) in loaded.weights().iter().zip(ensemble.weights()) {
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
    assert_eq!(loaded.swa_centers[0].weights, vec![1.0, 2.0, 3.0]);
    let _ = std::fs::remove_file(path);
}

#[test]
fn permutation_copy_ensembles_have_zero_functional_variance() {
    // Weights differ (rotated copies), predictions are identical.
    let weights = vec![
        vec![1.0f32, 0.0, 0.0, 1.0],
        vec![0.0f32, 1.0, 1.0, 0.0],
        vec![-1.0f32, 0.0, 0.0, -1.0],
    ];
    let d = pairwise_distances(&weights);
    assert!(d.iter().all(|x| *x > 0.0));
    let preds = vec![vec![0.5, -0.5], vec![0.5, -0.5], vec![0.5, -0.5]];
    let spread = prediction_spread(&preds).unwrap();
    assert!(spread.variance.iter().all(|v| *v == 0.0));
    assert_eq!(spread.disagreement, 0.0);
    assert_eq!(effective_rank(&preds).unwrap(), 1.0);

    let diverse = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0]];
    assert!(effective_rank(&diverse).unwrap() > 1.5);
}

#[test]
fn squared_error_ambiguity_identity_and_js_nonnegativity() {
    let preds = vec![vec![1.0, 2.0], vec![-1.0, 0.0], vec![0.0, 1.0]];
    let y = [0.5, 1.0];
    let (total, bias, ambiguity) = squared_error_ambiguity(&preds, &y).unwrap();
    assert!((total - (bias + ambiguity)).abs() < 1e-12);
    assert!(ambiguity > 0.0);

    let members = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.1, 0.6, 0.3],
        vec![0.2, 0.2, 0.6],
    ];
    let gap = js_dark_knowledge_gap(&members).unwrap();
    assert!(gap >= -1e-12);

    let student = vec![0.4, 0.35, 0.25];
    let f = teacher_student_fidelity(&members, &student).unwrap();
    assert!(f.kl >= -1e-12 && f.tv >= 0.0 && f.js_gap >= 0.0);
}

#[test]
fn pac_bayes_bound_is_a_probability_and_includes_the_point_risk() {
    let bound = pac_bayes_mixture_bound(0.1, 1.0, 100, 0.05).unwrap();
    assert!(bound >= 0.1 && bound <= 1.0);
    // A tiny sample with a large KL budget is vacuous.
    let vacuous = pac_bayes_mixture_bound(0.1, 1e6, 2, 0.05).unwrap();
    assert_eq!(vacuous, 1.0);
    assert!(kl_bernoulli(0.1, 0.1).unwrap().abs() < 1e-12);
    assert!(pac_bayes_mixture_bound(0.1, -1.0, 100, 0.05).is_none());
}

#[test]
fn calibration_helpers_are_consistent() {
    assert!((normal_quantile(0.975).unwrap() - 1.959963984540054).abs() < 1e-6);
    assert!((normal_quantile(0.95).unwrap() - 1.6448536269514722).abs() < 1e-6);
    let mean = [0.0, 0.0];
    let sigma = [1.0, 1.0];
    let y = [0.0, 5.0];
    let cov = pcnl::ensemble_diag::normal_coverage(&mean, &sigma, &y, 0.95).unwrap();
    assert!((cov - 0.5).abs() < 1e-12);
    let z2 = pcnl::ensemble_diag::normalized_residual_z2(&mean, &sigma, &y).unwrap();
    assert!((z2 - 12.5).abs() < 1e-12);
    assert!((in_region_fraction(&[0.5, 1.5, f64::NAN], 1.0).unwrap() - 0.5).abs() < 1e-12);
    assert!((tv_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap() - 1.0).abs() < 1e-12);
    // Identity carried through logs: the FNV helper is stable.
    assert_eq!(fnv1a(b"pcnl"), fnv1a(b"pcnl"));
}

#[test]
fn softmax_distillation_reduces_teacher_kl() {
    // A true linear teacher; the student has the same form and a zero start.
    let teacher = vec![
        0.8f64, -0.4, 0.2, // class 0
        -0.5, 0.7, 0.1, // class 1
        0.2, 0.1, -0.6, // class 2
    ];
    let classes = 3;
    let mut data = Vec::new();
    for i in 0..80 {
        let x = vec![
            (i as f64 * 0.37).sin(),
            (i as f64 * 0.11).cos(),
            0.5 + (i % 7) as f64 * 0.1,
        ];
        let logits: Vec<f64> = (0..classes)
            .map(|c| {
                x.iter()
                    .zip(&teacher[c * 3..(c + 1) * 3])
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        data.push((x, softmax(&logits).unwrap()));
    }
    let record = fit_softmax_student(3, classes, &data, 1500, 0.05).unwrap();
    assert!(
        record.final_kl < record.trace[0],
        "no improvement: {:.3e} -> {:.3e}",
        record.trace[0],
        record.final_kl
    );
    assert!(record.final_kl < 0.05, "final KL {:.4}", record.final_kl);
    // A KL-SGD gradient check at one point.
    let p = record.student.probs(&data[0].0).unwrap();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn spread_head_separates_opposing_vector_disagreement() {
    let outputs = vec![vec![1.0f32, -2.0], vec![-1.0f32, 2.0]];
    let stats = vector_stats(&outputs).unwrap();
    assert_eq!(stats.mean, vec![0.0, 0.0]);
    assert!((stats.spread[0] - 1.0).abs() < 1e-6);
    assert!((stats.spread[1] - 2.0).abs() < 1e-6);

    let means = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
    let spreads = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
    let head = SpreadHead::fit(&means, &spreads, 200, 0.1).unwrap();
    let v = head.variance(&[0.0]).unwrap();
    assert!((v[0] - 1.0).abs() < 1e-2, "{v:?}");
}

#[test]
fn rl_loss_is_nonnegative_and_gradient_matches_finite_differences() {
    let env = MdpEnv {
        horizon: 5,
        ..MdpEnv::default()
    };
    let policy = BehaviourPolicy {
        feedback: 0.5,
        epsilon: 0.2,
        seed: 9,
    };
    let corpus = Corpus::with_seeds(&env, policy, &[1, 2, 3, 4]);
    let loss = RlLoss::from_corpus(&corpus, 0.1, 5.0).unwrap();
    assert!(loss.transitions() > 0);

    let theta: Vec<f32> = (0..loss.flat_size())
        .map(|i| 0.01 * ((i as f32 * 0.7).sin()))
        .collect();
    let (value, grad) = loss.energy_and_grad(&theta).unwrap();
    assert!(value >= -1e-12, "loss must be nonnegative: {value}");
    let fd = fd_grad(|t| loss.loss(t), &theta, 1e-3);
    for (g, f) in grad.iter().zip(&fd) {
        assert!(
            (f64::from(*g) - f).abs() < 5e-2 * f.abs().max(1.0),
            "{g} vs {f}"
        );
    }

    let p = loss.probs(&theta, &[0.3, -0.2, 1.0]).unwrap();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let probs = |s: [f32; 2]| {
        loss.probs(&theta, &[f64::from(s[0]), f64::from(s[1]), 1.0])
            .unwrap()
    };
    let ret = pcnl::rl_track::rollout_return(&env, probs, 4, 11).unwrap();
    assert!(ret.is_finite());
    let _ = linear_policy_probs(&theta, &[0.3, -0.2, 1.0], loss.classes()).unwrap();
}

#[test]
fn rl_diversity_is_zero_for_one_policy_and_positive_across_members() {
    let a = vec![vec![0.7, 0.2, 0.1], vec![0.1, 0.8, 0.1]];
    let b = vec![vec![0.2, 0.7, 0.1], vec![0.3, 0.3, 0.4]];
    assert!(
        action_distribution_diversity(&[a.clone(), a.clone()])
            .unwrap()
            .0
            < 1e-12
    );
    let (tv, _kl) = action_distribution_diversity(&[a, b]).unwrap();
    assert!(tv > 0.0);
    let bound = pinsker_return_bound(10.0, 5, 0.1).unwrap();
    assert!((bound - 15.0).abs() < 1e-12);
}

fn return_aware_parts() -> (RlLoss, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f32>) {
    let env = MdpEnv {
        horizon: 5,
        ..MdpEnv::default()
    };
    let policy = BehaviourPolicy {
        feedback: 0.5,
        epsilon: 0.2,
        seed: 9,
    };
    let corpus = Corpus::with_seeds(&env, policy, &[1, 2, 3, 4]);
    let loss = RlLoss::from_corpus(&corpus, 0.1, 5.0).unwrap();
    let mut states = Vec::new();
    let mut rewards = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            let x = [-0.6 + 0.6 * i as f32, -0.6 + 0.6 * j as f32];
            states.push(vec![f64::from(x[0]), f64::from(x[1]), 1.0]);
            rewards.push(
                (0..loss.classes())
                    .map(|a| f64::from(env.reward(x, a)))
                    .collect(),
            );
        }
    }
    let theta: Vec<f32> = (0..loss.flat_size())
        .map(|i| 0.02 * ((i as f32 * 0.7).sin()))
        .collect();
    (loss, states, rewards, theta)
}

fn base_return_config() -> ReturnAwareConfig {
    ReturnAwareConfig {
        tau: 10.0,
        kappa: 2.0,
        alpha: 0.5,
        return_floor: 0.0,
        kappa_return: 20.0,
    }
}

#[test]
fn return_aware_gradient_matches_finite_differences() {
    let (loss, states, rewards, theta) = return_aware_parts();
    let target = RlReturnTarget::new(loss, states, rewards, base_return_config()).unwrap();
    let (value, grad) = target.energy_and_grad(&theta).unwrap();
    assert!(value.is_finite());
    let fd = fd_grad(|t| target.energy_and_grad(t).map(|(v, _)| v), &theta, 1e-3);
    for (g, f) in grad.iter().zip(&fd) {
        assert!(
            (f64::from(*g) - f).abs() < 5e-2 * f.abs().max(1.0),
            "{g} vs {f}"
        );
    }
}

#[test]
fn return_aware_floor_is_inactive_when_loss_and_return_satisfy_it() {
    let (loss, states, rewards, theta) = return_aware_parts();
    let probe = RlReturnTarget::new(
        loss.clone(),
        states.clone(),
        rewards.clone(),
        base_return_config(),
    )
    .unwrap();
    let ret = probe.return_surrogate(&theta).unwrap();
    let loss_value = probe.loss_value(&theta).unwrap();
    let cfg = ReturnAwareConfig {
        tau: loss_value + 1.0,
        return_floor: ret - 0.1,
        ..base_return_config()
    };
    let target = RlReturnTarget::new(loss, states, rewards, cfg).unwrap();
    let (phi, _) = target.energy_and_grad(&theta).unwrap();
    assert!(
        (phi - cfg.alpha * loss_value).abs() < 1e-9,
        "inactive hinges must leave Φ = α·L: {phi} vs {}",
        cfg.alpha * loss_value
    );
}

#[test]
fn return_aware_gradient_step_increases_the_return_surrogate() {
    let (loss, states, rewards, theta) = return_aware_parts();
    let probe = RlReturnTarget::new(
        loss.clone(),
        states.clone(),
        rewards.clone(),
        base_return_config(),
    )
    .unwrap();
    let start = probe.return_surrogate(&theta).unwrap();
    let cfg = ReturnAwareConfig {
        return_floor: start + 0.2,
        kappa_return: 200.0,
        ..base_return_config()
    };
    let target = RlReturnTarget::new(loss, states, rewards, cfg).unwrap();
    let (_, grad) = target.energy_and_grad(&theta).unwrap();
    let stepped: Vec<f32> = theta.iter().zip(&grad).map(|(t, g)| t - 1e-3 * g).collect();
    let after = target.return_surrogate(&stepped).unwrap();
    assert!(after > start, "return did not improve: {start} -> {after}");
}
