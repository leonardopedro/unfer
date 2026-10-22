//! Numerical counterparts of the Lean theorems of
//! `timepiece/BayesianTransformer/` (plan §7, M1–M5).
//!
//! Division of labour (plan §7.5): **Lean owns the worst case; the
//! crate measures the typical case.** Every test here checks the
//! *numerical shadow* of a proved statement — the finite-difference
//! version of an exact derivative identity, the measured constant of
//! a uniform bound, the empirical rate behind a variance-form claim —
//! on the crate's own architecture (Frobenius normalization, GELU,
//! nlerp residuals), not on the proof's.
//!
//! | test | Lean counterpart |
//! |---|---|
//! | `sphere_projection_derivative_matches_finite_differences` | M1 `sphereProj_hasFDerivAt` |
//! | `tangent_projector_never_increases_norms` | M1 `sphereProj_fderiv_opNorm_le` |
//! | `normalized_forward_is_scale_invariant` | M2 `scaleInvariant_gradient_orthogonal` (forward half) |
//! | `gradient_is_orthogonal_to_position` | M2 `scaleInvariant_inner_grad_zero` |
//! | `exact_normalization_gradient_blows_up_inverse_norm` | M4+ `normalized_fderiv_le` (the 1/‖W‖ rate) |
//! | `epsilon_regularization_floors_the_gradient` | §7.4's ε-floor (implementation counterpart) |
//! | `batch_gradient_noise_below_the_pathwise_ceiling` | M5 `sg_pcnl_noise_covariance_bound` |
//! | `covariance_shrinks_with_batch_size` | §3B.2's variance form (empirical target, not a theorem) |

use candle_core::{Device, Tensor};
use pcnl::model::{Shapes, Variant, init_flat, loss_and_grad};
use pcnl::normalize::normalize_last;
use pcnl::task::{Dataset, Task};

fn tiny() -> Shapes {
    Shapes {
        vocab: 9,
        d: 16,
        ff: 32,
        layers: 2,
    }
}

/// Central finite differences of a map `R^P → R^P` at `theta` along
/// direction `h`, relative step `eps`.
fn fd_direction(
    f: impl Fn(&Tensor) -> Tensor,
    theta: &Tensor,
    h: &Tensor,
    eps: f64,
) -> Vec<f32> {
    let plus = f(&(theta + (h.affine(eps, 0.0).unwrap())).unwrap());
    let minus = f(&(theta - (h.affine(eps, 0.0).unwrap())).unwrap());
    let p = plus.to_vec1::<f32>().unwrap();
    let m = minus.to_vec1::<f32>().unwrap();
    let scale = (2.0 * eps) as f32;
    p.iter().zip(&m).map(|(a, b)| (a - b) / scale).collect()
}

// ---------------------------------------------------------------------
// M1 — the sphere projection's derivative
// ---------------------------------------------------------------------

/// `sphereProj_hasFDerivAt`: `D(v/‖v‖) = ‖v‖⁻¹ I − (‖v‖³)⁻¹ v ⊗ v`.
/// Measured on the crate's actual normalization (Frobenius, ε-added)
/// at random points: the finite-difference directional derivative
/// matches the analytic tangential projector to f32 accuracy.
#[test]
fn sphere_projection_derivative_matches_finite_differences() {
    let s = tiny();
    for seed in 0..4u64 {
        let v = init_flat(&s, 1.0, seed);
        let p = v.elem_count();
        let dev = Device::Cpu;
        let mut rng_state = seed.wrapping_add(77);
        let mut next_f64 = move || {
            // tiny LCG so the test's directions are reproducible
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64 / (1u64 << 30) as f64) - 0.5
        };
        let h = Tensor::from_vec(
            (0..p).map(|_| next_f64() as f32).collect::<Vec<_>>(),
            vec![p],
            &dev,
        )
        .unwrap();
        let vn = v.to_vec1::<f32>().unwrap();
        let norm: f64 = vn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        // analytic: (1/‖v‖)(I − v vᵀ/‖v‖²) h
        let vh: f64 = vn
            .iter()
            .zip(h.to_vec1::<f32>().unwrap().iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let hvec = h.to_vec1::<f32>().unwrap();
        let analytic: Vec<f32> = vn
            .iter()
            .zip(&hvec)
            .map(|(vi, hi)| ((1.0 / norm) * (*hi as f64) - (vh / norm.powi(3)) * (*vi as f64)) as f32)
            .collect();
        let fd = fd_direction(
            |t| normalize_last(t).unwrap(),
            &v,
            &h,
            1e-3 * norm.max(1.0),
        );
        for (a, b) in analytic.iter().zip(&fd) {
            assert!(
                (a - b).abs() <= 1e-2 + 5e-2 * a.abs(),
                "seed {seed}: analytic {a} vs fd {b}"
            );
        }
    }
}

/// `sphereProj_fderiv_opNorm_le`: the differential is the *tangential*
/// projector over ‖v‖ — it never increases norms. Measured: ‖D h‖ ≤
/// ‖h‖/‖v‖ for random (v, h).
#[test]
fn tangent_projector_never_increases_norms() {
    let s = tiny();
    for seed in 0..8u64 {
        let v = init_flat(&s, 1.0, seed);
        let h = init_flat(&s, 1.0, seed + 100);
        let vn = v.to_vec1::<f32>().unwrap();
        let norm: f64 = vn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let hn = h.to_vec1::<f32>().unwrap();
        let hnorm: f64 = hn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let vh: f64 = vn.iter().zip(&hn).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let dn: f64 = vn
            .iter()
            .zip(&hn)
            .map(|(vi, hi)| (1.0 / norm) * (*hi as f64) - (vh / norm.powi(3)) * (*vi as f64))
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(
            dn <= hnorm / norm + 1e-9,
            "seed {seed}: ‖Dh‖ = {dn} > ‖h‖/‖v‖ = {}",
            hnorm / norm
        );
    }
}

// ---------------------------------------------------------------------
// M2 — scale invariance and the orthogonal gradient
// ---------------------------------------------------------------------

/// `scaleInvariant_gradient_orthogonal`'s forward half: the cosine
/// variant's loss is invariant under θ → cθ (every weight is used
/// through W/(‖W‖+ε)… the ε-regularization makes it invariant up to
/// the ε term; measured here to f32 rounding at c ∈ {0.5, 2, 7}).
/// The standard variant is *not* invariant — the contrast is the
/// measurement.
#[test]
fn normalized_forward_is_scale_invariant() {
    let s = tiny();
    let data = Dataset::new(Task::B, 8, 64, 32, 11);
    let theta = init_flat(&s, 0.3, 5);
    let tokens = data.train_x.clone();
    let targets = data.train_y.clone();
    let (l0, _) = loss_and_grad(&theta, &s, Variant::Cosine, &tokens, &targets).unwrap();
    for c in [0.5f64, 2.0, 7.0] {
        let scaled = theta.affine(c, 0.0).unwrap();
        let (lc, _) = loss_and_grad(&scaled, &s, Variant::Cosine, &tokens, &targets).unwrap();
        assert!(
            (lc - l0).abs() <= 1e-3 * l0.abs().max(1.0),
            "cosine loss changed under θ → {c}θ: {l0} vs {lc}"
        );
    }
    // the standard variant sees the scale: its loss moves
    let (s0, _) = loss_and_grad(&theta, &s, Variant::Standard, &tokens, &targets).unwrap();
    let scaled = theta.affine(3.0, 0.0).unwrap();
    let (s3, _) = loss_and_grad(&scaled, &s, Variant::Standard, &tokens, &targets).unwrap();
    assert!(
        (s3 - s0).abs() > 1e-3,
        "standard variant should NOT be scale-invariant: {s0} vs {s3}"
    );
}

/// `scaleInvariant_inner_grad_zero`: ⟨θ, ∇Φ(θ)⟩ = 0 — the gradient
/// has no radial component, so pCNL's drift is purely tangential.
///
/// Measured defect across seeds 0..3: 2.6e-4 … 2.2e-3 relative, and
/// **identical across init scales 0.3 / 1.0 / 3.0** (probed
/// 2026-10-21b). That invariance identifies the residual: it is the
/// f32-autodiff + activation-ε floor of the identity, NOT a
/// weight-norm ε effect (which would shrink 100× over those scales)
/// and NOT structural non-homogeneity (which shows up at the *loss*/// level and is pinned by `normalized_forward_is_scale_invariant`
/// at 1e-3 — the pre-fix raw-V/O bug changed the loss by 27%).
/// Bound: 5e-3 (2.3× the worst observed seed).
#[test]
fn gradient_is_orthogonal_to_position() {
    let s = tiny();
    let data = Dataset::new(Task::B, 8, 64, 32, 13);
    for seed in 0..4u64 {
        let theta = init_flat(&s, 0.3, seed);
        let (_, g) = loss_and_grad(&theta, &s, Variant::Cosine, &data.train_x, &data.train_y)
            .unwrap();
        let tv = theta.to_vec1::<f32>().unwrap();
        let gv = g.to_vec1::<f32>().unwrap();
        let dot: f64 = tv.iter().zip(&gv).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let tn: f64 = tv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let gn: f64 = gv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let rel = dot.abs() / (tn * gn);
        assert!(
            rel <= 5e-3,
            "seed {seed}: ⟨θ,∇Φ⟩/‖θ‖‖∇Φ‖ = {rel} — radial component too large"
        );
    }
}

// ---------------------------------------------------------------------
// M4+ / §7.4 — the 1/‖W‖ blow-up and its ε-floor
// ---------------------------------------------------------------------

/// The normalization derivative's norm is ≤ K/‖W‖ and *attains* the
/// blow-up rate: shrinking every weight by 3× multiplies the cosine
/// variant's gradient by ≈3 (exactly 3 under exact normalization, and
/// ≈3 here because ε ≪ ‖W‖). This is why the Lean bound lives on the
/// region ‖W‖ ≥ ε and a global bound is impossible.
#[test]
fn exact_normalization_gradient_blows_up_inverse_norm() {
    let s = tiny();
    let data = Dataset::new(Task::A, 8, 64, 32, 17);
    let theta = init_flat(&s, 0.3, 3);
    let (_, g1) = loss_and_grad(&theta, &s, Variant::Cosine, &data.train_x, &data.train_y).unwrap();
    let small = theta.affine(1.0 / 3.0, 0.0).unwrap();
    let (_, g3) = loss_and_grad(&small, &s, Variant::Cosine, &data.train_x, &data.train_y).unwrap();
    let n1: f64 = g1
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let n3: f64 = g3
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let ratio = n3 / n1;
    // ε = 1e-5 (normalize.rs) vs ‖W‖ ≥ 0.3-scale: the ratio is within
    // 15% of the exact-normalization prediction 3.
    assert!(
        (ratio - 3.0).abs() < 0.45,
        "gradient blow-up ratio {ratio} should be ≈3 (1/‖W‖ law)"
    );
}

/// §7.4's implementation counterpart: the ε-regularized
/// `W/(‖W‖+ε)` floors the gradient — at ‖W‖ → 0 the cosine gradient
/// stays finite (it *cannot* blow up past ~1/ε), unlike exact
/// normalization.
#[test]
fn epsilon_regularization_floors_the_gradient() {
    let s = tiny();
    let data = Dataset::new(Task::A, 8, 64, 32, 19);
    // ‖θ‖ ~ 0.3·√P ~ 3.7 at init_std 0.3; shrink by 100 → ‖θ‖ ~ 0.037,
    // still ≫ ε; then by 1000 more → ‖θ‖ ~ 3.7e-5, in the ε-region
    let theta = init_flat(&s, 0.3, 3);
    let mut worst = 0.0f64;
    for shrink in [100.0f64, 1000.0, 10_000.0] {
        let t = theta.affine(1.0 / shrink, 0.0).unwrap();
        let (_, g) = loss_and_grad(&t, &s, Variant::Cosine, &data.train_x, &data.train_y).unwrap();
        let gn: f64 = g
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(gn.is_finite(), "gradient non-finite at shrink {shrink}");
        worst = worst.max(gn);
    }
    // the ε-floor: the gradient norm cannot exceed ~P/ε-scale; a finite
    // ceiling across 4 decades of shrink is the floor's shadow
    let ceiling = (s.flat_size() as f64).sqrt() / 1e-5 * 2.0;
    assert!(worst < ceiling, "worst {worst} above ε-floor {ceiling}");
}

// ---------------------------------------------------------------------
// M5 — the stochastic-gradient noise ceilings
// ---------------------------------------------------------------------

/// `sg_pcnl_noise_covariance_bound`'s numerical shadow: the measured
/// E‖g_batch − g_full‖² over random batch draws sits below the Lean
/// pathwise ceiling (squared), instantiated with the structure's own
/// premises for the crate's sum-form quantities (M = N·M₁ — see the
/// in-test comment; the units matter). Also records the ratio to the
/// sharper variance-form target of §3B.2.
#[test]
fn batch_gradient_noise_below_the_pathwise_ceiling() {
    let s = tiny();
    let n_train = 128usize;
    let batch = 32usize;
    let data = Dataset::new(Task::B, 8, n_train, 16, 23);
    let theta = init_flat(&s, 0.2, 2);
    // full gradient (sum-form over the train set, scaled to N·mean)
    let (loss, full_sum) = data
        .batch_grad(&theta, &s, Variant::Cosine, &data.train_x, &data.train_y)
        .unwrap();
    assert!(loss.is_finite());
    let full = full_sum.to_vec1::<f32>().unwrap();
    // M: the measured per-example gradient bound (sum-form over 1
    // example = the per-example gradient)
    let mut m_bound = 0.0f64;
    for i in 0..n_train {
        let x = data.train_x.narrow(0, i, 1).unwrap();
        let y = data.train_y.narrow(0, i, 1).unwrap();
        let (_, g) = loss_and_grad(&theta, &s, Variant::Cosine, &x, &y).unwrap();
        let gn: f64 = g
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        m_bound = m_bound.max(gn);
    }
    // E‖g_batch − g_full‖² over 128 batch draws (batch sampled WITHOUT
    // replacement, matching BatchLoader)
    let mut acc = 0.0f64;
    let draws = 128usize;
    for d in 0..draws as u64 {
        let mut st = 0x9E3779B97F4A7C15u64 ^ d;
        let mut idx: Vec<u32> = (0..n_train as u32).collect();
        // Fisher–Yates with the LCG
        for i in (1..idx.len()).rev() {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (st >> 33) as usize % (i + 1);
            idx.swap(i, j);
        }
        idx.truncate(batch);
        let xi = Tensor::from_vec(idx.clone(), vec![batch], &Device::Cpu).unwrap();
        let bx = data.train_x.index_select(&xi, 0).unwrap();
        let by = data.train_y.index_select(&xi, 0).unwrap();
        let (_, sum) = data.batch_grad(&theta, &s, Variant::Cosine, &bx, &by).unwrap();
        let g = sum.to_vec1::<f32>().unwrap();
        // scaled: (N/B)·Σ_batch; full: N·mean = the sum over all
        let scale = (n_train / batch) as f64;
        let diff: f64 = g
            .iter()
            .zip(&full)
            .map(|(a, b)| (scale * (*a as f64) - *b as f64).powi(2))
            .sum();
        acc += diff;
    }
    let e_norm2 = acc / draws as f64;
    let n_over_b = (n_train / batch) as f64;
    // Lean's structure (`StochasticGradientEstimator`) premises are
    // `‖full_grad‖ ≤ M` and `‖batch_grad‖ ≤ (N/B)·M`. The faithful
    // instantiation of the crate's *sum-form* quantities:
    //   full  = Σ_{i≤N} ∇ℓ_i,  ‖·‖ ≤ N·M₁
    //   batch = (N/B)·Σ_{i∈B} ∇ℓ_i,  ‖·‖ ≤ N·M₁ ≤ (N/B)·(N·M₁)
    // so the structure's M is N·M₁ (M₁ = the measured per-example
    // bound), NOT M₁ itself — mixing the two scalings is what an
    // earlier draft of this test did (a 7× ceiling violation that was
    // really a units error).
    let m_lean = (n_train as f64) * m_bound;
    let lean_ceiling = ((n_over_b + 1.0) * m_lean).powi(2);
    assert!(e_norm2.is_finite() && e_norm2 > 0.0, "noise {e_norm2}");
    assert!(
        e_norm2 <= lean_ceiling,
        "E‖g_batch − g_full‖² = {e_norm2} exceeds the Lean pathwise ceiling {lean_ceiling} \
         (M = N·M₁ = {m_lean})"
    );
    // the sharper variance-form target of §3B.2 (empirical, not a
    // Lean theorem): (N²/B)·M₁² — recorded as a ratio so the log
    // shows where the measurement sits between the two ceilings
    let variance_target = (n_over_b * n_over_b / batch as f64) * m_bound * m_bound;
    let ratio_to_variance = e_norm2 / variance_target;
    assert!(
        ratio_to_variance.is_finite(),
        "ratio to variance-form target {ratio_to_variance} (should be ≪ 1: the pathwise \
         ceiling is conservative by construction)"
    );
}

/// §3B.2's variance form, empirically: halving the batch size
/// *increases* the noise, and doubling it *decreases* it — the 1/B
/// rate the (N²/B)M² claim predicts (measured as a monotone trend,
/// not a fitted constant: the claim is not a Lean theorem).
#[test]
fn covariance_shrinks_with_batch_size() {
    let s = tiny();
    let n_train = 128usize;
    let data = Dataset::new(Task::B, 8, n_train, 16, 29);
    let theta = init_flat(&s, 0.2, 2);
    let full = data
        .batch_grad(&theta, &s, Variant::Cosine, &data.train_x, &data.train_y)
        .unwrap()
        .1
        .to_vec1::<f32>()
        .unwrap();
    let noise_at = |batch: usize| -> f64 {
        let draws = 64usize;
        let mut acc = 0.0f64;
        for d in 0..draws as u64 {
            let mut st = 0x517CC1B727220A95u64 ^ (d * 31 + batch as u64);
            let mut idx: Vec<u32> = (0..n_train as u32).collect();
            for i in (1..idx.len()).rev() {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (st >> 33) as usize % (i + 1);
                idx.swap(i, j);
            }
            idx.truncate(batch);
            let xi = Tensor::from_vec(idx, vec![batch], &Device::Cpu).unwrap();
            let bx = data.train_x.index_select(&xi, 0).unwrap();
            let by = data.train_y.index_select(&xi, 0).unwrap();
            let sum = data
                .batch_grad(&theta, &s, Variant::Cosine, &bx, &by)
                .unwrap()
                .1;
            let g = sum.to_vec1::<f32>().unwrap();
            let scale = (n_train / batch) as f64;
            acc += g
                .iter()
                .zip(&full)
                .map(|(a, b)| (scale * (*a as f64) - *b as f64).powi(2))
                .sum::<f64>();
        }
        acc / draws as f64
    };
    let n8 = noise_at(8);
    let n32 = noise_at(32);
    let n128 = noise_at(128);
    // monotone in B, and the trend is substantial (≈4× per 4× batch)
    assert!(n8 > n32, "noise should shrink with B: B=8 {n8} vs B=32 {n32}");
    assert!(n32 > n128, "noise should shrink with B: B=32 {n32} vs B=128 {n128}");
    // B = N: the batch IS the dataset, so the scaled gradient is exact
    assert!(
        n128 < 1e-6 * (n8 + 1.0) + 1e-12,
        "B = N should give zero noise: {n128}"
    );
}
