//! Numerical validation battery for the unadjusted SG-pCNL engine
//! (docs/plan_bayes_ngpt.md).
//!
//! Where `src/*.rs` tests pin *behaviours of the code*, these
//! integration tests pin *mathematical facts the plan relies on*,
//! measured against closed-form answers or against statistically
//! justified bounds:
//!
//!   1. the isotropic prior concentrates on the shell σ√P (§2.1);
//!   2. the zero-gradient pCN kernel preserves the prior exactly —
//!      second moment, per-coordinate variance, shell radius;
//!   3. the drift term is the exact Ornstein–Uhlenbeck identity
//!      E‖θ'‖² = (1−β²)‖θ‖² + β²Pσ²;
//!   4. the analytic gradient equals central finite differences;
//!   5. the (N/B)-scaled mini-batch gradient equals N·mean-gradient;
//!   6. the cosine head's logits live in [−τ, τ] by construction;
//!   7. attention is causal (future tokens cannot change the past);
//!   8. θ = 0 predicts exactly the uniform distribution (loss = ln K);
//!   9. the Robbins–Monro partial sums diverge (∑β) / converge (∑β²);
//!  10. the whole experiment is bit-deterministic across reruns
//!      (including the `rayon` parallel chain merge).
//!
//! Every rng here is seeded, so a tolerance that passes once passes
//! forever; the statistical tests use pooled samples so the bounds
//! can be tight without being flaky.

use candle_core::{DType, Device, Tensor};
use pcnl::diag::{ess, gelman_rubin};
use pcnl::experiment::{Config, run_experiment};
use pcnl::model::{Shapes, Variant, forward, init_flat, loss_and_grad, unflatten};
use pcnl::normalize::{EPS, normalize_last};
use pcnl::sampler::{SgPcnlScheduler, SgPcnlState};
use pcnl::task::{BatchLoader, Dataset, Task, momentum_sgd};
use rand::{Rng, SeedableRng};

fn dev() -> &'static Device {
    static D: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    D.get_or_init(|| Device::Cpu)
}

fn tiny() -> Shapes {
    Shapes {
        vocab: 9,
        d: 8,
        ff: 16,
        layers: 1,
    }
}

fn sched(beta_0: f64) -> SgPcnlScheduler {
    SgPcnlScheduler {
        beta_0,
        gamma: 0.0, // constant β isolates the kernel from the schedule
        kappa: 0.55,
    }
}

// ---------------------------------------------------------------------
// 1. The isotropic prior sits on the shell σ√P (plan §2.1)
// ---------------------------------------------------------------------

#[test]
fn prior_draws_sit_on_the_shell_sigma_sqrt_p() {
    let s = tiny();
    let p = s.flat_size();
    let sigma = 0.35;
    let m = 500; // pooled draws

    let mut r2_sum = 0.0;
    let mut r_sum = 0.0;
    let mut x_sum = 0.0;
    let mut x2_sum = 0.0;
    for i in 0..m {
        let theta = init_flat(&s, sigma, 1000 + i as u64);
        let xs = theta.to_vec1::<f32>().unwrap();
        let r2: f64 = xs.iter().map(|x| (*x as f64).powi(2)).sum();
        r2_sum += r2;
        r_sum += r2.sqrt();
        x_sum += xs.iter().map(|x| *x as f64).sum::<f64>();
        x2_sum += xs.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
    }
    // E‖θ‖² = σ²P exactly; pooled relative SE = √(2/(mP)) ≈ 0.3%
    let mean_r2 = r2_sum / m as f64;
    assert!(
        (mean_r2 - sigma * sigma * p as f64).abs() / (sigma * sigma * p as f64) < 0.02,
        "E‖θ‖² = {mean_r2}, expected σ²P = {}",
        sigma * sigma * p as f64
    );
    // E‖θ‖ = σ·E[χ_P] = σ√P·(1 − 1/(4P) + O(P⁻²))
    let mean_r = r_sum / m as f64;
    let expected_r = sigma * (p as f64).sqrt() * (1.0 - 1.0 / (4.0 * p as f64));
    assert!(
        (mean_r - expected_r).abs() / expected_r < 0.02,
        "E‖θ‖ = {mean_r}, expected {expected_r}"
    );
    // pooled coordinates: mean ≈ 0, variance ≈ σ²
    let n = (m * p) as f64;
    let mean_x = x_sum / n;
    let var_x = x2_sum / n - mean_x * mean_x;
    assert!(mean_x.abs() < 0.01 * sigma, "pooled mean {mean_x}");
    assert!(
        (var_x - sigma * sigma).abs() / (sigma * sigma) < 0.02,
        "pooled var {var_x} vs σ² {}",
        sigma * sigma
    );
}

#[test]
fn init_flat_is_reproducible_and_scales_with_init_std() {
    let s = tiny();
    let a = init_flat(&s, 0.2, 42).to_vec1::<f32>().unwrap();
    let b = init_flat(&s, 0.2, 42).to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "same seed must be bit-identical");
    let c = init_flat(&s, 0.2, 43).to_vec1::<f32>().unwrap();
    assert_ne!(a, c, "different seeds must differ");
    // init_std is a standard deviation: ‖θ‖ ≈ init_std·√P
    let r: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let expected = 0.2 * (s.flat_size() as f64).sqrt();
    assert!(
        (r - expected).abs() / expected < 0.15,
        "‖θ₀‖ = {r}, expected ≈ {expected}"
    );
}

// ---------------------------------------------------------------------
// 2./3. pCN kernel: prior preservation and the exact OU second moment
// ---------------------------------------------------------------------

#[test]
fn zero_gradient_pcn_preserves_the_prior() {
    // Stationarity of θ ← √(1−β²)θ + βξ under N(0,σ²I): pooled over
    // many independent prior draws, one exact kernel step must leave
    // mean 0, variance σ² and E‖θ‖² = σ²P unchanged.
    let s = tiny();
    let p = s.flat_size();
    let sigma = 0.35;
    let m = 600;
    for beta_0 in [0.3, 0.9] {
        let zero = Tensor::zeros(p, DType::F32, dev()).unwrap();
        let mut x_sum = 0.0;
        let mut x2_sum = 0.0;
        let mut r2_sum = 0.0;
        for i in 0..m {
            let theta = init_flat(&s, sigma, 5000 + i as u64);
            let mut state = SgPcnlState::new(theta, sigma, sched(beta_0), 100, 25, 0);
            let mut rng = rand::rngs::StdRng::seed_from_u64(9000 + i as u64);
            state.step_unadjusted(&zero, &mut rng).unwrap();
            let xs = state.theta.to_vec1::<f32>().unwrap();
            x_sum += xs.iter().map(|x| *x as f64).sum::<f64>();
            x2_sum += xs.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            r2_sum += xs.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        }
        let n = (m * p) as f64;
        let mean_x = x_sum / n;
        let var_x = x2_sum / n - mean_x * mean_x;
        let mean_r2 = r2_sum / m as f64;
        assert!(
            mean_x.abs() < 0.01 * sigma,
            "β={beta_0}: post-step mean {mean_x} drifted from 0"
        );
        assert!(
            (var_x - sigma * sigma).abs() / (sigma * sigma) < 0.02,
            "β={beta_0}: variance {var_x} != σ² {} (prior not preserved)",
            sigma * sigma
        );
        assert!(
            (mean_r2 - sigma * sigma * p as f64).abs() / (sigma * sigma * p as f64) < 0.02,
            "β={beta_0}: E‖θ‖² {mean_r2} != σ²P {}",
            sigma * sigma * p as f64
        );
    }
}

#[test]
fn zero_gradient_chain_stays_on_the_shell_over_many_steps() {
    // 300 exact-kernel steps with zero gradient: an OU chain started
    // *off* the shell must relax back and stay at E‖θ‖² = σ²P.
    let s = tiny();
    let p = s.flat_size();
    let sigma = 1.0;
    let theta0 = init_flat(&s, 5.0, 77); // deliberately oversized start
    // E‖θ₀‖ = 5√P vs the target σ√P = √P: comfortably off-shell
    assert!(
        pcnl::sampler::norm(&theta0).unwrap() > 2.0 * sigma * (p as f64).sqrt(),
        "start not actually off-shell"
    );

    let zero = Tensor::zeros(p, DType::F32, dev()).unwrap();
    let mut state = SgPcnlState::new(theta0, sigma, sched(0.3), 100, 25, 0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(78);
    let mut r2_after_100 = 0.0;
    for t in 0..300 {
        state.step_unadjusted(&zero, &mut rng).unwrap();
        let r2: f64 = state
            .theta
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum();
        if t == 99 {
            r2_after_100 = r2;
        }
    }
    let target = sigma * sigma * p as f64;
    // after 100 steps the start's contribution is (1−β²)^100 ≈ 2e-15
    assert!(
        (r2_after_100 - target).abs() / target < 0.15,
        "t=100: E‖θ‖² = {r2_after_100}, target {target}"
    );
    let r2_final: f64 = state
        .theta
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum();
    assert!(
        (r2_final - target).abs() / target < 0.30,
        "t=300: ‖θ‖² = {r2_final}, target {target}"
    );
}

#[test]
fn drift_term_matches_the_ou_second_moment_identity() {
    // Exact identity per step: E‖θ'‖² = (1−β²)‖θ‖² + β²Pσ² (the
    // drift −(β²σ²/2)g vanishes with g = 0). Averaged over noise
    // draws from a *fixed* θ₀ this must hold to ~1/√m.
    let s = tiny();
    let p = s.flat_size();
    let sigma = 0.5;
    let beta = 0.4;
    let theta0 = init_flat(&s, 0.3, 11);
    let r0_2: f64 = theta0
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum();
    let zero = Tensor::zeros(p, DType::F32, dev()).unwrap();
    let m = 2000;
    let mut r2_sum = 0.0;
    for i in 0..m {
        let mut state = SgPcnlState::new(theta0.clone(), sigma, sched(beta), 100, 25, 0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(31_000 + i as u64);
        state.step_unadjusted(&zero, &mut rng).unwrap();
        r2_sum += state
            .theta
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>();
    }
    let expected = (1.0 - beta * beta) * r0_2 + beta * beta * p as f64 * sigma * sigma;
    // Var(‖θ'‖²) = 2·(tr(Σ²) + …) ≤ 2(E‖θ'‖²)²·O(1/P); 4% covers m=2000
    assert!(
        (r2_sum / m as f64 - expected).abs() / expected < 0.04,
        "E‖θ'‖² = {}, OU identity says {expected}",
        r2_sum / m as f64
    );
}

// ---------------------------------------------------------------------
// 4. Analytic gradient == central finite differences
// ---------------------------------------------------------------------

#[test]
fn gradient_matches_central_finite_differences() {
    let s = tiny();
    let p = s.flat_size();
    let theta = init_flat(&s, 0.2, 3);
    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 0, 3, 4, 0, 5, 6], &[3, 3], dev()).unwrap();
    let targets = Tensor::from_vec(vec![3u32, 5, 1], &[3], dev()).unwrap();
    let h = 1e-2; // f32 loss noise ~1e-6 → finite-difference noise ~1e-4

    for variant in [Variant::Cosine, Variant::Standard] {
        let (_, grad) = loss_and_grad(&theta, &s, variant, &tokens, &targets).unwrap();
        let g = grad.to_vec1::<f32>().unwrap();
        let flat = theta.to_vec1::<f32>().unwrap();
        // sample coordinates across the layout: embedding, layer 0 blocks
        let coords: Vec<usize> = vec![0, 3, s.vocab * s.d / 2, s.vocab * s.d, p / 3, p / 2, p - 1];
        let mut checked = 0usize;
        for &c in &coords {
            let mut plus = flat.clone();
            plus[c] += h as f32;
            let mut minus = flat.clone();
            minus[c] -= h as f32;
            let tp = Tensor::from_vec(plus, vec![p], dev()).unwrap();
            let tm = Tensor::from_vec(minus, vec![p], dev()).unwrap();
            let (lp, _) = loss_and_grad(&tp, &s, variant, &tokens, &targets).unwrap();
            let (lm, _) = loss_and_grad(&tm, &s, variant, &tokens, &targets).unwrap();
            let fd = (lp - lm) / (2.0 * h);
            let an = g[c] as f64;
            assert!(
                (fd - an).abs() < 1e-3 + 0.05 * an.abs(),
                "{variant:?} coord {c}: finite difference {fd} vs analytic {an}"
            );
            if an.abs() > 1e-3 {
                checked += 1;
            }
        }
        assert!(
            checked >= 3,
            "{variant:?}: only {checked} non-degenerate coordinates exercised"
        );
    }
}

// ---------------------------------------------------------------------
// 5. Mini-batch scaling: (N/B)·Σ ≡ N·mean, and per-example decomposition
// ---------------------------------------------------------------------

#[test]
fn full_batch_gradients_decompose_over_examples() {
    let s = tiny();
    let variant = Variant::Cosine;
    let n = 32;
    let data = Dataset::new(Task::A, 8, n, 16, 21);
    let theta = init_flat(&s, 0.1, 5);

    // batch_grad over the whole train split: sum-form = mean × B
    let (mean_loss, sum_grad) = data
        .batch_grad(&theta, &s, variant, &data.train_x, &data.train_y)
        .unwrap();
    let mut per_example_loss = 0.0;
    let mut summed = vec![0f64; s.flat_size()];
    for i in 0..n {
        let x = data.train_x.narrow(0, i, 1).unwrap();
        let y = data.train_y.narrow(0, i, 1).unwrap();
        let (l, g) = loss_and_grad(&theta, &s, variant, &x, &y).unwrap();
        per_example_loss += l;
        for (acc, gv) in summed.iter_mut().zip(g.to_vec1::<f32>().unwrap()) {
            *acc += gv as f64;
        }
    }
    assert!(
        (mean_loss - per_example_loss / n as f64).abs() < 1e-5,
        "mean loss {mean_loss} vs per-example mean {}",
        per_example_loss / n as f64
    );
    let sg = sum_grad.to_vec1::<f32>().unwrap();
    for (i, (a, b)) in summed.iter().zip(sg.iter()).enumerate() {
        assert!(
            (a - *b as f64).abs() < 1e-4 * (1.0 + a.abs()),
            "sum-form gradient mismatch at coord {i}: per-example {a} vs batch {}",
            *b as f64
        );
    }
}

#[test]
fn n_over_b_scaling_recovers_the_full_potential_gradient() {
    // (N/B)·Σ_{i∈B}∇ℓ_i = N·mean-gradient when the "batch" is the
    // full split — the identity `SgPcnlState` scales by and the
    // loader's unbiasedness depends on.
    let s = tiny();
    let n = 64;
    let data = Dataset::new(Task::B, 8, n, 16, 22);
    let theta = init_flat(&s, 0.1, 6);
    let (mean_loss, mean_grad) = data.loss_and_grad(&theta, &s, Variant::Cosine).unwrap();
    let (batch_loss, sum_grad) = data
        .batch_grad(&theta, &s, Variant::Cosine, &data.train_x, &data.train_y)
        .unwrap();
    assert_eq!(mean_loss, batch_loss, "same mean loss either way");
    let scale = n as f64 / n as f64; // B = N here
    let mg = mean_grad.to_vec1::<f32>().unwrap();
    let sg = sum_grad.to_vec1::<f32>().unwrap();
    for (i, (a, b)) in mg.iter().zip(sg.iter()).enumerate() {
        let scaled = scale * (*b as f64);
        assert!(
            (scaled - *a as f64 * n as f64).abs() < 1e-3 * (1.0 + (*a as f64 * n as f64).abs()),
            "coord {i}: (N/B)·sum = {scaled} vs N·mean = {}",
            *a as f64 * n as f64
        );
    }
}

// ---------------------------------------------------------------------
// 6./7./8. Architectural numerical facts of the model
// ---------------------------------------------------------------------

#[test]
fn cosine_head_logits_stay_within_tau() {
    // logits = τ·cos(ĥ, ê) with τ = √d, so |logit| ≤ τ by construction
    // (the ε of normalize_* only shrinks the norms below 1).
    let s = Shapes {
        vocab: 9,
        d: 16,
        ff: 32,
        layers: 2,
    };
    let tau = (s.d as f64).sqrt();
    let tokens = Tensor::from_vec((0..24u32).map(|i| i % 9).collect(), &[8, 3], dev()).unwrap();
    for seed in 0..6u64 {
        let theta = init_flat(&s, 0.4, 100 + seed);
        let logits = forward(&theta, &s, Variant::Cosine, &tokens).unwrap();
        let flat = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_abs = flat.iter().fold(0f32, |m, x| m.max(x.abs())) as f64;
        assert!(
            max_abs <= tau * 1.001 + 1e-3,
            "cosine logits exceed τ=√d: max |logit| = {max_abs} > {tau}"
        );
        assert!(max_abs > 0.1 * tau, "degenerate all-zero logits");
    }
}

#[test]
fn causal_attention_ignores_future_positions() {
    // Perturbing hidden state at position j must leave rows < j
    // bit-identical (masked scores are exactly −inf, their softmax
    // weight exactly 0), and must change some row ≥ j.
    let s = tiny();
    let theta = init_flat(&s, 0.3, 9);
    let slices = unflatten(&theta, &s).unwrap();
    let (b, t, d) = (1usize, 6usize, s.d);
    let mut rng = rand::rngs::StdRng::seed_from_u64(13);
    let hv: Vec<f32> = (0..b * t * d)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let h = Tensor::from_vec(hv.clone(), vec![b, t, d], dev()).unwrap();

    for variant in [Variant::Cosine, Variant::Standard] {
        let tau = (s.d as f64).sqrt();
        let out1 = pcnl::model::attention(&h, &slices.layers[0], variant, tau).unwrap();
        let j = 2usize;
        let mut hv2 = hv.clone();
        hv2[j * d] += 0.7;
        let h2 = Tensor::from_vec(hv2, vec![b, t, d], dev()).unwrap();
        let out2 = pcnl::model::attention(&h2, &slices.layers[0], variant, tau).unwrap();

        let v1 = out1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v2 = out2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for row in 0..j {
            for col in 0..d {
                assert_eq!(
                    v1[row * d + col],
                    v2[row * d + col],
                    "{variant:?}: row {row} changed after perturbing future row {j}"
                );
            }
        }
        let later_differs = (j * d..t * d).any(|i| (v1[i] - v2[i]).abs() > 0.0);
        assert!(
            later_differs,
            "{variant:?}: perturbation at row {j} did not propagate at all — test vacuous"
        );
    }
}

#[test]
fn zero_parameters_predict_the_uniform_distribution() {
    // θ = 0 ⇒ every logit is exactly 0 ⇒ softmax uniform ⇒ CE = ln K.
    // Pins the loss seam for both variants (the cosine ε only helps:
    // normalize(0) = 0, so all logits stay 0).
    let s = tiny();
    let zero = Tensor::zeros(s.flat_size(), DType::F32, dev()).unwrap();
    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 0, 3, 4], &[2, 3], dev()).unwrap();
    let targets = Tensor::from_vec(vec![3u32, 5], &[2], dev()).unwrap();
    let ln_k = (s.vocab as f64).ln();
    // Forward values are exact for both variants: θ = 0 ⇒ every logit
    // 0 ⇒ uniform predictive ⇒ CE = ln K.
    for variant in [Variant::Cosine, Variant::Standard] {
        let (loss, _) = loss_and_grad(&zero, &s, variant, &tokens, &targets).unwrap();
        assert!(
            (loss - ln_k).abs() < 1e-4,
            "{variant:?}: loss at θ=0 is {loss}, expected ln {} = {ln_k}",
            s.vocab
        );
    }
    // The *gradient* at the exact origin is a differentiability fact:
    // standard LN divides by √(var + 1e-5) > 0, so it is finite there;
    // the cosine hypersphere projection x ↦ x/(‖x‖ + ε) has a 0/0
    // Jacobian at x = 0, so backprop yields NaN — the reason every run
    // starts from init_std > 0. At any nonzero start it is finite.
    let (_, g_std) = loss_and_grad(&zero, &s, Variant::Standard, &tokens, &targets).unwrap();
    assert!(
        g_std
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "Standard: gradient at θ=0 should be finite"
    );
    let nearby = init_flat(&s, 1e-3, 5);
    for variant in [Variant::Cosine, Variant::Standard] {
        let (_, g) = loss_and_grad(&nearby, &s, variant, &tokens, &targets).unwrap();
        assert!(
            g.to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()),
            "{variant:?}: non-finite gradient at small nonzero θ"
        );
    }
}

#[test]
fn cosine_normalization_rows_are_unit_norm_up_to_eps() {
    // ‖normalize_last(x)‖ = ‖x‖/(‖x‖+ε) ≤ 1, with the ε-relative
    // deficit bounded by EPS·(1/‖min row‖).
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let rows = 32usize;
    let cols = 12usize;
    let v: Vec<f32> = (0..rows * cols)
        .map(|_| rng.random_range(-2.0f32..2.0))
        .collect();
    let x = Tensor::from_vec(v, vec![rows, cols], dev()).unwrap();
    let y = normalize_last(&x).unwrap();
    let yn = y
        .sqr()
        .unwrap()
        .sum_keepdim(candle_core::D::Minus1)
        .unwrap()
        .sqrt()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let xs = x
        .sqr()
        .unwrap()
        .sum_keepdim(candle_core::D::Minus1)
        .unwrap()
        .sqrt()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for i in 0..rows {
        let expect = xs[i] as f64 / (xs[i] as f64 + EPS);
        assert!(
            (yn[i] as f64 - expect).abs() < 1e-6,
            "row {i}: {} vs {expect}",
            yn[i]
        );
        assert!(yn[i] <= 1.0, "row {i}: norm above 1");
    }
}

// ---------------------------------------------------------------------
// 9. Robbins–Monro summability of the schedule (plan §3C)
// ---------------------------------------------------------------------

#[test]
fn robbins_monro_partial_sums_diverge_and_converge() {
    // κ = 0.55 ∈ (0.5, 1]: Σβₜ ~ T^{1−κ} → ∞ (unbiasedness) while
    // Σβₜ² saturates (variance control). Checked on the real
    // scheduler with the default γ = 1e-3, numerically.
    let s = SgPcnlScheduler {
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.55,
    };
    let sum_to = |t_max: usize| -> (f64, f64) {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for t in 0..t_max {
            let b = s.beta_at(t);
            s1 += b;
            s2 += b * b;
        }
        (s1, s2)
    };
    let (s1_1e4, s2_1e4) = sum_to(10_000);
    let (s1_1e5, s2_1e5) = sum_to(100_000);
    // ∑β keeps growing: ratio ≈ (10)^{1−κ} = 10^0.45 ≈ 2.8
    let r1 = s1_1e5 / s1_1e4;
    assert!(
        (2.0..5.0).contains(&r1),
        "Σβ must diverge like T^0.45: ratio {r1}"
    );
    // ∑β² is bounded by its ceiling β₀²/(γ(2κ−1)) = 0.09/1e-4 = 900
    let ceiling = s.beta_0 * s.beta_0 / (s.gamma * (2.0 * s.kappa - 1.0));
    assert!(
        s2_1e5 < ceiling,
        "Σβ² = {s2_1e5} must stay under the analytic ceiling {ceiling}"
    );
    let r2 = s2_1e5 / s2_1e4;
    assert!(
        r2 < 2.0,
        "Σβ² must saturate (ratio {r2} from 1e4 to 1e5 steps)"
    );
    // and β never dies within a realistic run
    assert!(s.beta_at(1_000_000) > 0.0);
}

// ---------------------------------------------------------------------
// 10. Data plumbing: reproducibility, labels, loader without replacement
// ---------------------------------------------------------------------

#[test]
fn dataset_materialization_is_reproducible_and_labelled() {
    let a = Dataset::new(Task::B, 8, 64, 32, 31);
    let b = Dataset::new(Task::B, 8, 64, 32, 31);
    assert_eq!(
        a.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        b.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        "same seed must reproduce the split"
    );
    let c = Dataset::new(Task::B, 8, 64, 32, 32);
    assert_ne!(
        a.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        c.train_x.flatten_all().unwrap().to_vec1::<u32>().unwrap()
    );
    // every row encodes y = label(a, b, k) + 1 (token offset)
    let xs = a.train_x.to_vec2::<u32>().unwrap();
    let ys = a.train_y.to_vec1::<u32>().unwrap();
    for (row, &y) in xs.iter().zip(ys.iter()) {
        let (af, bf) = (row[1] as usize - 1, row[2] as usize - 1);
        assert_eq!(y as usize, Task::B.label(af, bf, 8) + 1, "label mismatch");
    }
    // all k classes occur somewhere in a 64-draw split of a k=8 task
    let mut seen = [false; 9];
    for &y in &ys {
        seen[y as usize] = true;
    }
    assert!(seen[1..=8].iter().all(|&x| x), "class missing from split");
}

#[test]
fn batch_loader_is_seeded_and_samples_without_replacement() {
    // unique marker rows let us recover the sampled indices
    let n = 64usize;
    let mut xs = Vec::with_capacity(n * 3);
    for i in 0..n {
        xs.extend_from_slice(&[0u32, 1 + (i / 8) as u32, 1 + (i % 8) as u32]);
    }
    let data = Dataset {
        k: 8,
        train_x: Tensor::from_vec(xs, vec![n, 3], dev()).unwrap(),
        train_y: Tensor::zeros(n, DType::U32, dev()).unwrap(),
        test_x: Tensor::zeros((1, 3), DType::U32, dev()).unwrap(),
        test_y: Tensor::zeros(1, DType::U32, dev()).unwrap(),
    };
    let mut l1 = BatchLoader::new(n, 16, 51);
    let mut l2 = BatchLoader::new(n, 16, 51);
    let mut l3 = BatchLoader::new(n, 16, 52);
    let (x1a, _) = l1.next(&data);
    let (x1b, _) = l2.next(&data);
    let (x2, _) = l3.next(&data);
    assert_eq!(
        x1a.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        x1b.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        "same seed must replay the batch stream"
    );
    assert_ne!(
        x1a.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        x2.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        "different seed must differ"
    );
    // without replacement within a batch: rows (and hence indices) distinct
    let rows = x1a.to_vec2::<u32>().unwrap();
    let mut uniq = std::collections::HashSet::new();
    for r in &rows {
        assert!(uniq.insert(r.clone()), "repeated index in one batch: {r:?}");
    }
    // marginal uniformity over many batches: each (a, b) cell hit ~uniformly
    let mut counts = [0usize; 64];
    for _ in 0..64 {
        let (xb, _) = l1.next(&data);
        for r in xb.to_vec2::<u32>().unwrap() {
            counts[(r[1] as usize - 1) * 8 + (r[2] as usize - 1)] += 1;
        }
    }
    let total: usize = counts.iter().sum();
    assert_eq!(total, 64 * 16);
    let expected = total as f64 / 64.0;
    for (i, &c) in counts.iter().enumerate() {
        // χ²(63) ≈ 63 ± 11; 5σ of a multinomial cell is generous but
        // still fails loudly on a biased sampler
        assert!(
            (c as f64 - expected).abs() < 5.0 * expected.max(1.0),
            "cell {i} sampled {c} times, expected ≈ {expected}"
        );
    }
    // batch size clamps to n
    let mut big = BatchLoader::new(10, 50, 1);
    let (xb, yb) = big.next(&data);
    assert_eq!(xb.dim(0).unwrap(), 10);
    assert_eq!(yb.dim(0).unwrap(), 10);
}

#[test]
fn trained_model_beats_the_uniform_predictor() {
    let s = Shapes {
        vocab: 9,
        d: 16,
        ff: 32,
        layers: 2,
    };
    let data = Dataset::new(Task::A, 8, 128, 64, 41);
    let theta0 = init_flat(&s, 0.1, 42);
    let (theta, trace) = momentum_sgd(&data, &s, Variant::Cosine, theta0, 150, 0.5, 0.9);
    let nll = data.nll(&theta, &s, Variant::Cosine).unwrap();
    let ln_k = (s.vocab as f64).ln(); // = 2.197: what ignorance costs
    assert!(nll < ln_k, "trained NLL {nll} >= uniform baseline {ln_k}");
    assert!(
        nll < trace[0],
        "NLL {nll} not below the initial loss {}",
        trace[0]
    );
    let acc = data.accuracy(&theta, &s, Variant::Cosine).unwrap();
    assert!(acc > 100.0 / 8.0, "accuracy {acc}% not above chance");
    // NLL ≥ 0 always (a probability ≤ 1)
    assert!(nll >= 0.0);
}

// ---------------------------------------------------------------------
// 11. The full experiment: determinism, arithmetic, diagnostics bounds
// ---------------------------------------------------------------------

/// A config small enough for debug-mode tests but exercising every
/// code path (3 chains → the rayon merge; both variants).
fn micro() -> Config {
    Config {
        seed: 4242,
        k: 8,
        task_b: Task::B,
        shapes: Shapes {
            vocab: 9,
            d: 16,
            ff: 32,
            layers: 2,
        },
        n_train: 128,
        n_test: 64,
        pretrain_steps: 60,
        pretrain_lr: 0.6,
        chains: 3,
        steps: 240,
        burn: 120,
        thin: 5,
        batch_size: 32,
        sigma: 0.35,
        beta_0: 0.3,
        gamma: 1e-3,
        kappa: 0.55,
        sgd_steps_on_b: 40,
        adam_steps_on_b: 40,
        adam_lr: 0.05,
        init_std: 0.1,
        save_dir: None,
        cyclical_period: 0,
        run_mh: true,
    }
}

#[test]
fn experiment_is_bit_deterministic_across_reruns() {
    let cfg = micro();
    let mut r1 = run_experiment(&cfg);
    let mut r2 = run_experiment(&cfg);
    assert_eq!(r1.variants.len(), r2.variants.len());
    for v in r1.variants.iter_mut().chain(r2.variants.iter_mut()) {
        // wall-clock is the one legitimately varying field (all
        // three timers: pCNL chains, the SGLD comparator's chains,
        // and the Phase-4b filtered comparator's chains)
        v.seconds = 0.0;
        v.sgld_seconds = 0.0;
        v.mh_seconds = 0.0;
        v.saved_files.clear();
    }
    assert_eq!(
        format!("{r1:?}"),
        format!("{r2:?}"),
        "two runs of one seed diverged (parallel chain merge is not order-stable)"
    );
}

#[test]
fn report_arithmetic_matches_the_config() {
    let cfg = micro();
    let report = run_experiment(&cfg);
    let p = cfg.shapes.flat_size();
    for v in &report.variants {
        assert_eq!(v.p, p, "P is variant-independent");
        assert_eq!(v.steps, cfg.steps);
        // σ√P reported exactly
        assert!((v.radial_expected - cfg.sigma * (p as f64).sqrt()).abs() < 1e-12);
        // the schedule endpoints are the closed form, to f64 precision
        let beta_expected = |t: usize| {
            (cfg.beta_0 * (1.0 + cfg.gamma * t as f64).powf(-cfg.kappa)).clamp(1e-6, 0.999)
        };
        assert!((v.beta_start - beta_expected(0)).abs() < 1e-15);
        assert!((v.beta_end - beta_expected(cfg.steps - 1)).abs() < 1e-15);
        // BMA bookkeeping: models ≤ chains × kept-per-chain
        let kept_per_chain = (cfg.steps - cfg.burn) / cfg.thin + 1;
        assert!(v.bma_models <= cfg.chains * kept_per_chain);
        if v.exploded {
            // explosion carries evidence, never a silent zero
            assert!(v.max_scaled_grad_norm > 0.0 || v.grad_nonfinite);
            continue;
        }
        // diagnostics stay inside their mathematical ranges
        let n_kept = kept_per_chain as f64;
        assert!(
            v.ess_loss > 0.0 && v.ess_loss <= n_kept * cfg.chains as f64 + 1.0,
            "ESS {} outside (0, kept]",
            v.ess_loss
        );
        assert!(
            v.bma_ece.is_finite() && (0.0..=1.0).contains(&v.bma_ece),
            "ECE {} outside [0,1]",
            v.bma_ece
        );
        assert!(
            (0.0..=100.0).contains(&v.bma_accuracy) && v.bma_nll >= 0.0,
            "BMA metrics out of range: acc {} nll {}",
            v.bma_accuracy,
            v.bma_nll
        );
        assert!(
            v.rhat_loss.is_finite() && v.rhat_loss > 0.9,
            "Rhat {}",
            v.rhat_loss
        );
        // the shell claim the plan stakes nGPT on
        if v.variant.contains("nGPT") {
            assert!(
                (v.radial_final - v.radial_expected).abs() / v.radial_expected < 0.5,
                "nGPT radial {} drifted from σ√P {}",
                v.radial_final,
                v.radial_expected
            );
        }
    }
}

#[test]
fn diagnostics_behave_on_synthetic_chains() {
    // ESS: iid ≈ N; sticky ≪ N; shuffling the sticky chain recovers it
    // (same marginal distribution — the drop is correlation, not data)
    let mut rng = rand::rngs::StdRng::seed_from_u64(61);
    let rho = 0.97;
    let n = 6000usize;
    let mut sticky = Vec::with_capacity(n);
    let mut x = 0.0f64;
    for _ in 0..n {
        x = rho * x + (rng.random::<f64>() * 2.0 - 1.0);
        sticky.push(x);
    }
    let e_sticky = ess(&sticky);
    assert!(e_sticky < 0.15 * n as f64, "sticky ESS {e_sticky} too high");
    let mut shuffled = sticky.clone();
    // Fisher–Yates with the same rng stream
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        shuffled.swap(i, j);
    }
    let e_shuffled = ess(&shuffled);
    assert!(
        e_shuffled > 0.6 * n as f64,
        "shuffled ESS {e_shuffled} should recover ≈ N"
    );
    assert!(e_sticky < e_shuffled);

    // R̂: identical chains → 1; offset chains → flag
    let c: Vec<f64> = (0..2000).map(|_| rng.random::<f64>()).collect();
    let r_same = gelman_rubin(&[c.clone(), c.clone()]);
    assert!((0.99..1.01).contains(&r_same), "identical-chain R̂ {r_same}");
    let shifted: Vec<f64> = c.iter().map(|v| v + 3.0).collect();
    let r_far = gelman_rubin(&[c, shifted]);
    assert!(r_far > 3.0, "mode-separated R̂ {r_far} too small");
}
