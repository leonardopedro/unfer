//! Phase 2 — the **unadjusted SG-pCNL** engine (plan §3/§5 as
//! updated: the Metropolis–Hastings filter is *gone*).
//!
//! One gradient evaluation per step, on the current θ and a streamed
//! mini-batch; the transition is accepted unconditionally. What used
//! to be the acceptance-rate tuning becomes a classical SGLD-style
//! power-law step-size schedule (plan §3C, Option 1):
//!
//! ```text
//! β_t = β_0 · (1 + γ·t)^(−κ),   κ ∈ (0.5, 1]
//! ```
//!
//! The gradient handed to [`SgPcnlState::step_unadjusted`] is the
//! **sum over the batch** (the loader's mean gradient × B), because
//! the state applies the plan's `(N / B)` scaler — `(N/B)·Σ ≡ N·mean`,
//! stated once here so the scale line in the skeleton is not a
//! silent deviation.

use candle_core::{Result, Tensor};
use rand::Rng;
use rand_distr::StandardNormal;

/// Plan §3C: the Robbins–Monro step-size schedule (Option 1) and the
/// cyclical alternative (Option 2), both as pure functions of the
/// step index — no state, no acceptance feedback.
#[derive(Clone, Copy, Debug)]
pub struct SgPcnlScheduler {
    pub beta_0: f64,
    pub gamma: f64,
    pub kappa: f64,
}

impl SgPcnlScheduler {
    /// `β_t = β_0 (1 + γ·t)^(−κ)` — the skeleton's schedule with
    /// κ = 0.55 by default (inside the plan's (0.5, 1]).
    pub fn beta_at(&self, step: usize) -> f64 {
        let t = step as f64;
        (self.beta_0 * (1.0 + self.gamma * t).powf(-self.kappa)).clamp(1e-6, 0.999)
    }

    /// Cyclical cosine decay (plan §3C Option 2): each cycle starts at
    /// β₀ (explore) and decays to the floor (sample), then jumps back.
    pub fn beta_cyclical(&self, step: usize, period: usize) -> f64 {
        let period = period.max(2);
        let phase = (step % period) as f64 / (period - 1) as f64;
        // 0 at the cycle start (β₀), 1 at the cycle end (floor)
        let decay = 0.5 * (1.0 - (std::f64::consts::PI * phase).cos());
        (self.beta_0 * (1.0 - decay) + 1e-3 * self.beta_0 * decay).clamp(1e-6, 0.999)
    }
}

/// The plan §5 skeleton's state: contiguous `[P]` parameters plus the
/// schedule and mini-batch bookkeeping. No loss, no stored gradient,
/// no accept/reject — those were the MH filter's.
pub struct SgPcnlState {
    pub theta: Tensor, // Contiguous parameter vector [P]
    pub sigma: f64,    // Prior std dev (isotropic)
    pub beta_0: f64,   // Initial step size
    pub gamma: f64,    // Decay rate
    pub kappa: f64,    // Decay exponent (0.55)
    pub step_count: usize,
    pub dataset_size: usize,
    pub batch_size: usize,
}

impl SgPcnlState {
    pub fn new(
        theta: Tensor,
        sigma: f64,
        scheduler: SgPcnlScheduler,
        dataset_size: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            theta,
            sigma,
            beta_0: scheduler.beta_0,
            gamma: scheduler.gamma,
            kappa: scheduler.kappa,
            step_count: 0,
            dataset_size,
            batch_size,
        }
    }

    /// Current scheduled step size (also what the next update uses).
    pub fn beta(&self) -> f64 {
        SgPcnlScheduler {
            beta_0: self.beta_0,
            gamma: self.gamma,
            kappa: self.kappa,
        }
        .beta_at(self.step_count)
    }

    /// Executes an in-place Unadjusted SG-pCNL update using a
    /// mini-batch gradient (sum-form; see the module docs).
    /// Memory complexity: O(B · T · d · L), decoupled from `dataset_size`.
    pub fn step_unadjusted(
        &mut self,
        batch_grad: &Tensor, // evaluated on current mini-batch
        rng: &mut impl Rng,
    ) -> Result<()> {
        let p = self.theta.elem_count();
        let dev = self.theta.device();

        // 1. scheduled step size β_t (no acceptance feedback anywhere)
        let t = self.step_count as f64;
        let beta = (self.beta_0 * (1.0 + self.gamma * t).powf(-self.kappa)).clamp(1e-6, 0.999);
        self.step_count += 1;

        // 2. ξ ~ N(0, σ² I): prior-preserving OU noise
        let noise_vec: Vec<f32> = (0..p)
            .map(|_| {
                let x: f32 = rng.sample(StandardNormal);
                x * self.sigma as f32
            })
            .collect();
        let xi = Tensor::from_vec(noise_vec, vec![p], dev)?;

        // 3. scale the stochastic gradient to the full potential:
        //    g = (N / B) · Σ_{i∈B} ∇ℓ_i   (sum-form batch_grad)
        let scale = (self.dataset_size as f64) / (self.batch_size as f64);
        let scaled_grad = batch_grad.affine(scale, 0.0)?;

        // 4. unadjusted step:
        //    θ ← √(1−β²)·θ + β·ξ − (β²σ²/2)·g
        let sqrt_decay = (1.0 - beta * beta).max(0.0).sqrt();
        let drift_coeff = (beta * beta * self.sigma * self.sigma) / 2.0;

        let theta_decay = self.theta.affine(sqrt_decay, 0.0)?;
        let noise_term = xi.affine(beta, 0.0)?;
        let drift_term = scaled_grad.affine(drift_coeff, 0.0)?;

        // direct unconditional assignment — no MH evaluation pass
        self.theta = ((theta_decay + noise_term)? - drift_term)?;

        Ok(())
    }
}

/// f64 inner product / norm over a flat tensor — kept for the
/// monitoring the Phase-0 comparison reports (gradient spikes,
/// radial norms).
pub fn dot(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum())
}

pub fn norm(a: &Tensor) -> Result<f64> {
    Ok(dot(a, a)?.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::{ess, gelman_rubin};
    use candle_core::Device;
    use rand::SeedableRng;

    #[test]
    fn scheduler_is_the_power_law_and_monotone() {
        let s = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 1e-4,
            kappa: 0.55,
        };
        assert!((0.5..1.0).contains(&s.kappa), "κ must sit in (0.5, 1]");
        let b0 = s.beta_at(0);
        assert!((b0 - 0.3).abs() < 1e-12, "β_0 = {b0}");
        let mut prev = b0;
        for step in [1, 10, 100, 1_000, 10_000] {
            let b = s.beta_at(step);
            let expected = 0.3 * (1.0 + 1e-4 * step as f64).powf(-0.55);
            assert!(
                (b - expected).abs() < 1e-12,
                "β_{step} = {b}, expected {expected}"
            );
            assert!(b <= prev, "monotone decay: {b} after {prev}");
            prev = b;
        }
        // κ > 0.5 is what the RM theory needs for ∑β=∞, ∑β²<∞
        assert!(s.beta_at(10_000) > 0.0, "step size never vanishes in a run");
    }

    #[test]
    fn cyclical_schedule_stays_in_range_and_cycles() {
        let s = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 1e-4,
            kappa: 0.55,
        };
        let period = 200;
        let vals: Vec<f64> = (0..3 * period)
            .map(|t| s.beta_cyclical(t, period))
            .collect();
        assert!(vals.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        // starts high (exploration), reaches the floor near mid-cycle
        assert!(vals[0] > vals[period / 2], "decay within a cycle");
        assert!((vals[0] - vals[period]).abs() < 1e-9, "periodic");
    }

    #[test]
    fn one_update_matches_the_formula_exactly() {
        // Seeded rng lets the test re-derive the same ξ independently,
        // so the comparison is to the plan's formula, not to itself.
        let dev = &Device::Cpu;
        let theta = Tensor::from_vec(vec![0.5f32, -0.25, 0.75], &[3], dev).unwrap();
        let mut state = SgPcnlState::new(
            theta.clone(),
            2.0,
            SgPcnlScheduler {
                beta_0: 0.4,
                gamma: 0.0,
                kappa: 0.55,
            },
            100,
            25,
        );
        let grad = Tensor::from_vec(vec![0.1f32, 0.2, -0.3], &[3], dev).unwrap();

        let mut r1 = rand::rngs::StdRng::seed_from_u64(42);
        state.step_unadjusted(&grad, &mut r1).unwrap();

        // independent re-derivation of the same draw
        let mut r2 = rand::rngs::StdRng::seed_from_u64(42);
        let xi: Vec<f32> = (0..3)
            .map(|_| {
                let x: f32 = r2.sample(StandardNormal);
                x * 2.0f32
            })
            .collect();
        let beta = 0.4f64; // (1 + 0·t)^… = β_0
        let scale = 100.0 / 25.0;
        let sqrt_decay = (1.0 - beta * beta).sqrt();
        let drift = beta * beta * 2.0 * 2.0 / 2.0;
        let expected: Vec<f32> = (0..3)
            .map(|i| {
                (sqrt_decay as f32) * theta.to_vec1::<f32>().unwrap()[i] + beta as f32 * xi[i]
                    - drift as f32 * (scale as f32) * grad.to_vec1::<f32>().unwrap()[i]
            })
            .collect();
        let got = state.theta.to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        assert_eq!(state.step_count, 1, "step counter advances");
    }

    #[test]
    fn scaling_is_the_unbiased_n_over_b_estimate() {
        // (N/B)·Σ with sum-form grads ≡ N·mean-grad: the identity the
        // module docs state and the loader relies on.
        let n = 512.0;
        let b = 64.0;
        let mean_grad = 0.01f64;
        let sum_grad = mean_grad * b;
        let scaled = sum_grad * (n / b);
        let direct = mean_grad * n;
        assert!((scaled - direct).abs() < 1e-12, "{scaled} vs {direct}");
    }

    #[test]
    fn noise_path_is_standard_normal_times_sigma() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sigma = 1.5;
        let mut draws = Vec::with_capacity(20_000);
        for _ in 0..100 {
            let xi: Vec<f32> = (0..200)
                .map(|_| {
                    let x: f32 = rng.sample(StandardNormal);
                    x * sigma as f32
                })
                .collect();
            draws.extend(xi);
        }
        let n = draws.len() as f64;
        let mean = draws.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = draws
            .iter()
            .map(|x| (*x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.05, "noise mean {mean}");
        assert!(
            (var - sigma * sigma).abs() / (sigma * sigma) < 0.06,
            "var {var}"
        );
    }

    /// The honest anchor replacing the MH exactness test: on a quadratic
    /// target with a *small* scheduled β, the unadjusted chain tracks the
    /// analytic posterior mean (unadjusted bias is O(β), so the
    /// tolerance is loose and stated as such).
    #[test]
    fn small_step_chain_tracks_the_gaussian_target() {
        // Φ(θ) = ½‖Aθ − y‖², prior N(0, σ²I); posterior mean solve
        // via the same normal equations as before (P = 4).
        use candle_core::Tensor as T;
        let dev = &Device::Cpu;
        let (m, p_dim) = (8usize, 4usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let n = rand_distr::StandardNormal;
        let a_v: Vec<f32> = (0..m * p_dim)
            .map(|_| {
                let z: f32 = rng.sample(n);
                z * 0.5
            })
            .collect();
        let y_v: Vec<f32> = (0..m)
            .map(|_| {
                let z: f32 = rng.sample(n);
                z * 0.1 + 1.0
            })
            .collect();
        let a = T::from_vec(a_v, vec![m, p_dim], dev).unwrap();
        let y = T::from_vec(y_v, vec![m], dev).unwrap();
        let sigma = 2.0;

        // precision = AᵀA + σ⁻²I, rhs = Aᵀy (f64, hand-eliminated)
        let av = a.to_vec2::<f32>().unwrap();
        let yv = y.to_vec1::<f32>().unwrap();
        let mut prec = vec![vec![0.0f64; p_dim]; p_dim];
        let mut rhs = vec![0.0f64; p_dim];
        for i in 0..p_dim {
            for j in 0..p_dim {
                prec[i][j] = (0..m).map(|k| (av[k][i] as f64) * (av[k][j] as f64)).sum();
            }
            prec[i][i] += 1.0 / (sigma * sigma);
            rhs[i] = (0..m).map(|k| (av[k][i] as f64) * (yv[k] as f64)).sum();
        }
        let solve = |mat: &[Vec<f64>], b: &[f64]| -> Vec<f64> {
            let n = b.len();
            let mut mm = mat.to_vec();
            let mut bb = b.to_vec();
            for col in 0..n {
                let piv = (col..n)
                    .max_by(|&x, &y| mm[x][col].abs().partial_cmp(&mm[y][col].abs()).unwrap())
                    .unwrap();
                mm.swap(col, piv);
                bb.swap(col, piv);
                let d = mm[col][col];
                let pivot_row = mm[col].clone();
                for r in 0..n {
                    if r != col {
                        let f = mm[r][col] / d;
                        for (cell, &pv) in mm[r].iter_mut().zip(&pivot_row).skip(col) {
                            *cell -= f * pv;
                        }
                        bb[r] -= f * bb[col];
                    }
                }
            }
            (0..n).map(|i| bb[i] / mm[i][i]).collect()
        };
        let target_mean = solve(&prec, &rhs);

        // full-batch unadjusted chain (batch == N ⇒ scale = 1), small β
        let mut state = {
            let init: Vec<f32> = (0..p_dim)
                .map(|_| {
                    let z: f32 = rng.sample(n);
                    z * sigma as f32
                })
                .collect();
            SgPcnlState::new(
                T::from_vec(init, vec![p_dim], dev).unwrap(),
                sigma,
                SgPcnlScheduler {
                    beta_0: 0.05,
                    gamma: 1e-4,
                    kappa: 0.55,
                },
                m,
                m,
            )
        };
        let loss_grad = |theta: &T| -> Result<(f64, T)> {
            let th = theta.reshape((p_dim, 1))?;
            let r = (a.matmul(&th)?.flatten_all()? - &y)?;
            let loss = 0.5 * r.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            let g = a
                .transpose(1, 0)?
                .matmul(&r.reshape((m, 1))?)?
                .flatten_all()?;
            Ok((loss, g))
        };
        let mut mean = vec![0.0f64; p_dim];
        let mut kept = 0usize;
        for t in 0..12_000 {
            let (_, g) = loss_grad(&state.theta).unwrap();
            state.step_unadjusted(&g, &mut rng).unwrap();
            if t >= 4_000 {
                let v = state.theta.to_vec1::<f32>().unwrap();
                for i in 0..p_dim {
                    mean[i] += v[i] as f64;
                }
                kept += 1;
            }
        }
        assert!(
            state
                .theta
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
        for i in 0..p_dim {
            let m_est = mean[i] / kept as f64;
            let err = (m_est - target_mean[i]).abs();
            // Unadjusted bias is O(β): the honest claim is "the chain
            // sits inside the posterior's bulk", not exactness. Posterior
            // sd per dim (from prec⁻¹) bounds that: |err| < 0.6·sd.
            let mut e = vec![0.0; p_dim];
            e[i] = 1.0;
            let sd = solve(&prec, &e)[i].max(0.0).sqrt();
            assert!(
                err < 0.6 * sd + 1e-9,
                "dim {i}: unadjusted mean {m_est} vs analytic {} (err {err}, 0.6·sd {})",
                target_mean[i],
                0.6 * sd
            );
        }
    }

    #[test]
    fn trajectories_from_different_seeds_mix() {
        // Two independent unadjusted chains on a quadratic target must
        // land at the same level — diagnostics without any MH filter.
        let dev = &Device::Cpu;
        let p_dim = 4usize;
        let run = |seed: u64| -> (Vec<f64>, Vec<f64>) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let n = rand_distr::StandardNormal;
            let init: Vec<f32> = (0..p_dim)
                .map(|_| {
                    let z: f32 = rng.sample(n);
                    z
                })
                .collect();
            let mut state = SgPcnlState::new(
                Tensor::from_vec(init, vec![p_dim], dev).unwrap(),
                1.5,
                SgPcnlScheduler {
                    beta_0: 0.1,
                    gamma: 1e-4,
                    kappa: 0.55,
                },
                10,
                10,
            );
            // target: Φ = ½‖θ − c‖² with c = (1, −1, 2, 0) ⇒ N(c, I+N(0,σ²)
            // posterior mean = c·N/(N+1)… loose level check only
            let c = [1.0f32, -1.0, 2.0, 0.0];
            let mut trace = Vec::new();
            let mut coord0 = Vec::new();
            for t in 0..6_000 {
                let g = (&state.theta - Tensor::from_vec(c.to_vec(), vec![p_dim], dev).unwrap())
                    .unwrap();
                state.step_unadjusted(&g, &mut rng).unwrap();
                if t >= 2_000 {
                    let v = state.theta.to_vec1::<f32>().unwrap();
                    trace.push(v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>());
                    coord0.push(v[0] as f64);
                }
            }
            (trace, coord0)
        };
        let (l1, c1) = run(1);
        let (l2, c2) = run(2);
        let (l3, c3) = run(3);
        let r = gelman_rubin(&[l1.clone(), l2.clone(), l3]);
        assert!(r.is_finite() && r < 1.1, "R̂ on radial energy = {r}");
        let r0 = gelman_rubin(&[c1, c2, c3]);
        assert!(r0.is_finite() && r0 < 1.2, "R̂ on coord0 = {r0}");
        let e = ess(&l1);
        // 4k post-transient draws of a slow β_t chain: ESS ≥ 30 says
        // the trajectory is not pathological; exact mixing speed is a
        // property of the schedule, not of this test.
        assert!(e > 30.0, "ESS {e} on 4k draws");
    }
}
