//! Phase 2 — the **unadjusted SG-pCNL** engine (plan §3/§5 as
//! updated: the Metropolis–Hastings filter is *gone*).
//!
//! One gradient evaluation per step, on the current θ and a streamed
//! mini-batch; the transition is accepted unconditionally. What used
//! to be the acceptance-rate tuning becomes a classical SGLD-style
//! power-law step-size schedule (plan §3C, Option 1):
//!
//! ```text
//! β_t = β_0 · (1 + γ·t)^(−κ),   κ ∈ (1/4, 1/2]
//! ```
//!
//! **The exponent window is on `β_t²`, not on `β_t`** (corrected
//! 2026-11-03). The gradient enters through the drift coefficient
//! `h_t = β_t²σ²/2`, so the decreasing-step conditions of stochastic
//! approximation are `Σ h_t = ∞` and `Σ h_t² < ∞`, i.e.
//!
//! ```text
//! Σ β_t² = ∞   (infinite total mobility)      ⟸ 2κ ≤ 1
//! Σ β_t⁴ < ∞   (vanishing accumulated noise)  ⟸ 4κ > 1
//! ```
//!
//! hence `κ ∈ (1/4, 1/2]`. `κ = 1/2` is the borderline case, where the
//! mobility diverges only logarithmically (`Σ β_t² ~ (β₀²/γ)·ln(1+γT)`).
//! The plan's original `κ ∈ (0.5, 1]` sits on the wrong side of that
//! threshold: it makes `Σ β_t² → β₀²/(γ(2κ−1))` **finite**, and a chain
//! whose step sizes are summable travels a finite distance and then
//! freezes wherever it is — it cannot relocate the predictive to the
//! posterior's mode. [`SgPcnlScheduler::total_mobility`] is `None`
//! exactly when the chain never freezes, and
//! [`SgPcnlScheduler::kappa_in_mobility_window`] is the check every run
//! of the experiment reports.
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
    /// Exclusive lower end of the exponent window: `Σ β_t⁴ < ∞` needs
    /// `4κ > 1`, so `κ = 1/4` itself diverges.
    pub const KAPPA_MIN: f64 = 0.25;
    /// Inclusive upper end: `Σ β_t² = ∞` needs `2κ ≤ 1`.
    pub const KAPPA_MAX: f64 = 0.5;

    /// `β_t = β_0 (1 + γ·t)^(−κ)` — the skeleton's schedule with
    /// κ = 1/2 by default (the borderline case of the corrected window).
    pub fn beta_at(&self, step: usize) -> f64 {
        let t = step as f64;
        (self.beta_0 * (1.0 + self.gamma * t).powf(-self.kappa)).clamp(1e-6, 0.999)
    }

    /// Whether `κ` sits in the mobility window `(1/4, 1/2]` — the only
    /// range in which the annealed chain keeps moving *and* its
    /// accumulated gradient noise vanishes. Outside it, the run is a
    /// fixed-`β` statement about the kernel, not convergence to the
    /// posterior.
    pub fn kappa_in_mobility_window(&self) -> bool {
        self.kappa > Self::KAPPA_MIN && self.kappa <= Self::KAPPA_MAX
    }

    /// `Σ_{t<T} β_t²` — the **mobility** spent by step `T`. This is the
    /// σ-free quantity: the drift coefficient on the θ-scale is `σ²/2`
    /// times it, and on `u = θ/σ` it is exactly `(1/2)·Σβ_t²`
    /// ([`SgPcnlState`]'s update, with `g(σu) = g(u)/σ`). It is how far
    /// the chain can be pulled, in units the schedule controls.
    pub fn partial_mobility(&self, steps: usize) -> f64 {
        (0..steps).map(|t| self.beta_at(t).powi(2)).sum()
    }

    /// `Σ_{t≥from} β_t²` — the mobility the schedule still has after
    /// step `from`. `None` when the sum diverges (`κ ≤ 1/2`): a chain
    /// that keeps moving forever. For `κ > 1/2` the integral test gives
    /// the closed form `β₀²(1+γ·from)^{1−2κ}/(γ(2κ−1))`, which is the
    /// size of the budget still unspent — hence the freeze.
    pub fn tail_mobility(&self, from: usize) -> Option<f64> {
        if self.kappa <= 0.5 {
            return None;
        }
        let t = from as f64;
        Some(
            self.beta_0 * self.beta_0 * (1.0 + self.gamma * t).powf(1.0 - 2.0 * self.kappa)
                / (self.gamma * (2.0 * self.kappa - 1.0)),
        )
    }

    /// `Σ_{t≥0} β_t²` — the whole mobility budget of the schedule.
    /// `None` (infinite) exactly when `κ ≤ 1/2`.
    pub fn total_mobility(&self) -> Option<f64> {
        self.tail_mobility(0)
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
    pub kappa: f64,    // Decay exponent (0.5 — the mobility window's edge)
    pub step_count: usize,
    pub dataset_size: usize,
    pub batch_size: usize,
    /// §3C Option 2: 0 = Robbins–Monro (Option 1); > 0 = cyclical
    /// cosine decay with this period. The mode is per-state so an
    /// experiment can run both schedules on the same seed.
    pub period: usize,
    /// Prior mean μ. `None` is the plan's N(0, σ²I). `Some(θ_A)` is
    /// the *centered adaptation prior* N(θ_A, σ²I) (§6.4.1's
    /// reachability follow-up): the OU step then contracts toward
    /// θ_A, so the chain equilibrates on the shell *around the
    /// pre-trained weights* instead of being dragged to the shell
    /// around the origin before the likelihood can act.
    pub center: Option<Tensor>,
    /// **Control only, default `None`:** cap the norm of the
    /// `(N/B)`-scaled mini-batch gradient before it enters the drift.
    ///
    /// This exists for one question (§6.5.2 of `test/unfer/ngpt.md`):
    /// when a *non*-scale-invariant stack explodes under this kernel, is
    /// that a property of its geometry, or a statement that the pCN
    /// coefficient `β²σ²/2` prescribes a step size for a potential whose
    /// gradient has no natural scale? With the cap set the drift can move
    /// θ by at most `h_t·cap` per step, so both variants can be handed the
    /// same displacement budget and the explosion re-measured. It is
    /// **not part of the kernel**: `None` (the default, and the value
    /// every reported run uses) leaves the update exactly the plan's
    /// formula.
    pub grad_norm_cap: Option<f64>,
}

impl SgPcnlState {
    pub fn new(
        theta: Tensor,
        sigma: f64,
        scheduler: SgPcnlScheduler,
        dataset_size: usize,
        batch_size: usize,
        period: usize,
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
            period,
            center: None,
            grad_norm_cap: None,
        }
    }

    /// Same state with the prior centered at `center` (θ_A): every
    /// step below acts on the deviation θ − μ and adds μ back, so the
    /// stationary shell of the zero-gradient kernel is N(μ, σ²I).
    pub fn centered_at(mut self, center: Tensor) -> Self {
        self.center = Some(center);
        self
    }

    /// The scheduler this state's fields describe.
    pub fn scheduler(&self) -> SgPcnlScheduler {
        SgPcnlScheduler {
            beta_0: self.beta_0,
            gamma: self.gamma,
            kappa: self.kappa,
        }
    }

    /// Current scheduled step size (also what the next update uses).
    pub fn beta(&self) -> f64 {
        self.scheduled(self.step_count)
    }

    /// β_t under the configured schedule: Robbins–Monro when
    /// `period == 0`, cyclical cosine otherwise (plan §3C).
    pub fn scheduled(&self, step: usize) -> f64 {
        if self.period == 0 {
            self.scheduler().beta_at(step)
        } else {
            self.scheduler().beta_cyclical(step, self.period)
        }
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
        let beta = self.scheduled(self.step_count);
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

        // 3b. optional drift cap (default off, a control and not a
        //     kernel feature): ‖g‖ ≤ cap, so the drift displacement is
        //     at most h_t·cap. Off, this is the identity.
        let scaled_grad = match self.grad_norm_cap {
            Some(cap) => {
                let n = norm(&scaled_grad)?;
                if n.is_finite() && n > cap {
                    scaled_grad.affine(cap / n, 0.0)?
                } else {
                    scaled_grad
                }
            }
            None => scaled_grad,
        };

        // 4. unadjusted step about the prior mean μ:
        //    θ ← μ + √(1−β²)·(θ−μ) + β·ξ − (β²σ²/2)·g
        //  (μ = 0 reduces term-for-term to the plan's skeleton; μ = θ_A
        //   is the centered adaptation prior — a pure translation, so
        //   the noise, the drift and the pCN prior-ratio cancellation
        //   behind the MH filter are all unchanged.)
        let sqrt_decay = (1.0 - beta * beta).max(0.0).sqrt();
        let drift_coeff = (beta * beta * self.sigma * self.sigma) / 2.0;

        let decayed = match &self.center {
            Some(c) => (&self.theta - c)?.affine(sqrt_decay, 0.0)?,
            None => self.theta.affine(sqrt_decay, 0.0)?,
        };
        let noise_term = xi.affine(beta, 0.0)?;
        let drift_term = scaled_grad.affine(drift_coeff, 0.0)?;

        // direct unconditional assignment — no MH evaluation pass
        let updated = ((decayed + noise_term)? - drift_term)?;
        self.theta = match &self.center {
            Some(c) => (c + updated)?,
            None => updated,
        };

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
            kappa: 0.5,
        };
        // The corrected window is (1/4, 1/2] — the conditions are on
        // β_t² (the drift coefficient is β_t²σ²/2), so κ = 0.55 leaves
        // Σβ_t² finite. This test used to assert the opposite.
        assert!(
            s.kappa_in_mobility_window(),
            "κ must sit in (1/4, 1/2], got {}",
            s.kappa
        );
        assert!(
            !SgPcnlScheduler { kappa: 0.55, ..s }.kappa_in_mobility_window(),
            "κ = 0.55 has finite total mobility"
        );
        let b0 = s.beta_at(0);
        assert!((b0 - 0.3).abs() < 1e-12, "β_0 = {b0}");
        let mut prev = b0;
        for step in [1, 10, 100, 1_000, 10_000] {
            let b = s.beta_at(step);
            let expected = 0.3 * (1.0 + 1e-4 * step as f64).powf(-0.5);
            assert!(
                (b - expected).abs() < 1e-12,
                "β_{step} = {b}, expected {expected}"
            );
            assert!(b <= prev, "monotone decay: {b} after {prev}");
            prev = b;
        }
        // κ ≤ 1/2 is what the mobility condition needs (Σβ² = ∞); the
        // point of the window's *upper* end is that β decays at all.
        assert!(s.beta_at(10_000) > 0.0, "step size never vanishes in a run");
        assert!(s.beta_at(10_000) < b0, "the schedule decays: β_T < β_0");
    }

    /// The mobility bookkeeping, against the two closed forms the
    /// write-up quotes (§4.3): `Σβ² → β₀²/(γ(2κ−1))` for `κ > 1/2`
    /// (900 at the reference numbers, 479.2 by step 2e6) and
    /// `Σβ² ~ (β₀²/γ)·ln(1+γT)` at `κ = 1/2` (684.2 at 2e6).
    #[test]
    fn mobility_budget_matches_the_closed_form() {
        let inside = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.5,
        };
        let outside = SgPcnlScheduler {
            kappa: 0.55,
            ..inside
        };

        // κ = 1/2: no finite budget — the chain can always still move.
        assert!(inside.total_mobility().is_none());
        assert!(inside.tail_mobility(10_000).is_none());
        let used = inside.partial_mobility(2_000_000);
        let log_form = 0.09 / 1e-3 * (1.0 + 1e-3 * 2_000_000f64).ln();
        assert!(
            (used - log_form).abs() < 0.01 * log_form,
            "κ=1/2 mobility {used} vs log form {log_form}"
        );

        // κ = 0.55: finite — 900 in total, 479.2 spent by step 2e6, so
        // the remaining 420.8 is all the motion the chain will ever get.
        let total = outside.total_mobility().expect("finite budget");
        assert!((total - 900.0).abs() < 1.0, "total mobility {total}");
        let spent = outside.partial_mobility(2_000_000);
        assert!((spent - 479.2).abs() < 1.0, "spent {spent}");
        let left = outside.tail_mobility(2_000_000).expect("finite tail");
        assert!((left - (total - spent)).abs() < 0.02 * total, "tail {left}");
        // and the tail shrinks: the freeze gets arbitrarily deep.
        assert!(outside.tail_mobility(2_000_000).unwrap() < outside.tail_mobility(1_000).unwrap());
    }

    /// The σ-sweep is a units check, and this is *why*: with a loss
    /// that is scale-invariant on the shell (`g(cθ) = g(θ)/c`, the exact
    /// statement of Φ(cθ) = Φ(θ)), the update in `u = θ/σ` is
    /// σ-free — decay, noise `βξ'` and drift `(β²/2)g(u)`. So two
    /// chains at σ and 4σ, from θ and 4θ on one noise stream, are the
    /// same trajectory on the u-scale. A σ-sweep therefore cannot
    /// measure anything about the posterior (write-up §6.5.1(a)).
    #[test]
    fn the_kernel_is_sigma_free_in_rescaled_units() {
        let dev = &Device::Cpu;
        let p = 8usize;
        let base: Vec<f32> = (0..p).map(|i| 0.3 + 0.1 * i as f32).collect();
        let sched = SgPcnlScheduler {
            beta_0: 0.3,
            gamma: 1e-3,
            kappa: 0.5,
        };
        // g(θ) = θ/‖θ‖²: homogeneous of degree −1, i.e. the exact
        // scale-equivariance of a scale-invariant potential.
        let grad = |theta: &Tensor| -> Tensor {
            let n = norm(theta).unwrap().powi(2);
            theta.affine(1.0 / n, 0.0).unwrap()
        };

        let run = |scale: f32| -> Vec<f32> {
            let init: Vec<f32> = base.iter().map(|x| x * scale).collect();
            let mut state = SgPcnlState::new(
                Tensor::from_vec(init, vec![p], dev).unwrap(),
                scale as f64,
                sched,
                64,
                64,
                0,
            );
            let mut rng = rand::rngs::StdRng::seed_from_u64(99);
            for _ in 0..200 {
                let g = grad(&state.theta);
                state.step_unadjusted(&g, &mut rng).unwrap();
            }
            state.theta.to_vec1::<f32>().unwrap()
        };

        let one = run(1.0);
        let four = run(4.0);
        for (i, (a, b)) in one.iter().zip(four.iter()).enumerate() {
            assert!(
                (b - 4.0 * a).abs() < 1e-5 * (1.0 + b.abs()),
                "coord {i}: σ=4 trajectory {b} is not 4× the σ=1 one {a}"
            );
        }
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
            0,
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
    fn the_drift_cap_bounds_the_step_and_is_off_by_default() {
        // §6.5.2's fairness control is only a control if the cap does
        // exactly one thing: bound ‖drift‖ by h_t·cap. Three runs from
        // one seed measure the three claims.
        let dev = &Device::Cpu;
        let p = 4usize;
        let theta = Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 0.0], &[p], dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0: 0.5,
            gamma: 0.0,
            kappa: 0.5,
        };
        let (n, b, sigma, cap) = (100usize, 25usize, 1.0f64, 1.0f64);
        // sum-form gradient of norm 5 → scaled (N/B) norm 20;
        // h = β²σ²/2 = 0.125, so the uncapped drift is 2.5 while the
        // capped one must be ≤ 0.125.
        let grad = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], &[p], dev).unwrap();
        let zero = Tensor::zeros(p, candle_core::DType::F32, dev).unwrap();

        let base = {
            let mut s = SgPcnlState::new(theta.clone(), sigma, sched, n, b, 0);
            let mut r = rand::rngs::StdRng::seed_from_u64(7);
            s.step_unadjusted(&zero, &mut r).unwrap();
            s.theta.to_vec1::<f32>().unwrap()
        };

        let mut capped = SgPcnlState::new(theta.clone(), sigma, sched, n, b, 0);
        capped.grad_norm_cap = Some(cap);
        let mut r = rand::rngs::StdRng::seed_from_u64(7);
        capped.step_unadjusted(&grad, &mut r).unwrap();
        let drift_capped: f64 = capped
            .theta
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(&base)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let h = 0.5 * 0.5 * sigma * sigma / 2.0;
        assert!(
            (drift_capped - h * cap).abs() < 1e-6,
            "capped drift {drift_capped} should equal h·cap = {}",
            h * cap
        );

        let mut plain = SgPcnlState::new(theta.clone(), sigma, sched, n, b, 0);
        let mut r = rand::rngs::StdRng::seed_from_u64(7);
        plain.step_unadjusted(&grad, &mut r).unwrap();
        let drift_plain: f64 = plain
            .theta
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(&base)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (drift_plain - h * 20.0).abs() < 1e-5,
            "uncapped drift {drift_plain} should be h·(N/B)·‖Σ‖ = {}",
            h * 20.0
        );

        // Off by default: a cap above the gradient's norm is the
        // identity to the bit, so no reported run can be affected.
        let mut loose = SgPcnlState::new(theta.clone(), sigma, sched, n, b, 0);
        loose.grad_norm_cap = Some(1e12);
        let mut r = rand::rngs::StdRng::seed_from_u64(7);
        loose.step_unadjusted(&grad, &mut r).unwrap();
        assert_eq!(
            loose.theta.to_vec1::<f32>().unwrap(),
            plain.theta.to_vec1::<f32>().unwrap(),
            "a non-binding cap must not perturb the step"
        );
    }

    #[test]
    fn centered_update_is_the_origin_step_translated_by_theta_a() {
        // The centered adaptation prior (§6.4.1's follow-up) is a pure
        // translation: with μ the step must equal
        //   μ + [√(1−β²)(θ−μ) + βξ − (β²σ²/2)·(N/B)Σ]
        // where the bracket is *exactly* the μ = 0 update applied to
        // θ−μ. So run both from the same seed and compare.
        let dev = &Device::Cpu;
        let theta = Tensor::from_vec(vec![0.5f32, -0.25, 0.75], &[3], dev).unwrap();
        let center = Tensor::from_vec(vec![3.0f32, -1.0, 2.0], &[3], dev).unwrap();
        let grad = Tensor::from_vec(vec![0.1f32, 0.2, -0.3], &[3], dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0: 0.4,
            gamma: 0.0,
            kappa: 0.55,
        };

        // origin-centered run, from θ − μ
        let deviated = (&theta - &center).unwrap();
        let mut origin = SgPcnlState::new(deviated.clone(), 2.0, sched, 100, 25, 0);
        let mut r1 = rand::rngs::StdRng::seed_from_u64(42);
        origin.step_unadjusted(&grad, &mut r1).unwrap();

        // θ_A-centered run, from θ — same noise stream
        let mut centered =
            SgPcnlState::new(theta.clone(), 2.0, sched, 100, 25, 0).centered_at(center.clone());
        let mut r2 = rand::rngs::StdRng::seed_from_u64(42);
        centered.step_unadjusted(&grad, &mut r2).unwrap();

        // θ_μ = μ + (origin step from θ−μ), to the bit the same ξ draws
        let expected = (&center + origin.theta).unwrap();
        let got = centered.theta.to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(expected.to_vec1::<f32>().unwrap().iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }

    #[test]
    fn centered_zero_gradient_kernel_is_stationary_on_the_theta_a_shell() {
        // With g = 0 the centered kernel is an OU process with fixed
        // point μ: E‖θ‖² relaxes to ‖μ‖² + Pσ², not to Pσ². This is
        // the reachability property the follow-up exists for — the
        // chain equilibrates on the shell *around the pre-trained
        // weights* instead of being dragged to the origin's shell.
        let dev = &Device::Cpu;
        let p_dim = 64usize;
        let sigma = 0.5f64;
        let mu: Vec<f32> = (0..p_dim).map(|i| (i as f32) * 0.1).collect();
        let center = Tensor::from_vec(mu.clone(), &[p_dim], dev).unwrap();
        let mu_sq: f64 = mu.iter().map(|x| (*x as f64).powi(2)).sum();
        let theta0 = Tensor::zeros(p_dim, candle_core::DType::F32, dev).unwrap();
        let zero_grad = Tensor::zeros(p_dim, candle_core::DType::F32, dev).unwrap();
        let sched = SgPcnlScheduler {
            beta_0: 0.5,
            gamma: 1e-4,
            kappa: 0.55,
        };

        let mut centered =
            SgPcnlState::new(theta0.clone(), sigma, sched, 100, 25, 0).centered_at(center.clone());
        let mut origin = SgPcnlState::new(theta0, sigma, sched, 100, 25, 0);
        let mut r1 = rand::rngs::StdRng::seed_from_u64(7);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(7);

        // sample the tail of each chain: ‖θ‖² for μ = 0,
        // ‖θ−μ‖² for the centered one (both should sit at Pσ²)
        let mut centered_off: Vec<f64> = Vec::new();
        let mut centered_abs: Vec<f64> = Vec::new();
        let mut origin_off: Vec<f64> = Vec::new();
        for t in 0..3_000 {
            centered.step_unadjusted(&zero_grad, &mut r1).unwrap();
            origin.step_unadjusted(&zero_grad, &mut r2).unwrap();
            if t >= 2_000 {
                let dc = (&centered.theta - &center).unwrap();
                centered_off.push(norm(&dc).unwrap().powi(2));
                centered_abs.push(norm(&centered.theta).unwrap().powi(2));
                origin_off.push(norm(&origin.theta).unwrap().powi(2));
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let shell = (p_dim as f64) * sigma * sigma; // Pσ²
        let c_off = mean(&centered_off);
        let o_off = mean(&origin_off);
        // both kernels sit on their own shell to within Monte-Carlo error
        assert!(
            (c_off - shell).abs() / shell < 0.10,
            "centered ‖θ−μ‖² = {c_off} vs Pσ² = {shell}"
        );
        assert!(
            (o_off - shell).abs() / shell < 0.10,
            "origin ‖θ‖² = {o_off} vs Pσ² = {shell}"
        );
        // and the centered chain lives at θ_A + shell: E‖θ‖² = ‖μ‖² + Pσ²
        // (measured directly — the cross term is what the origin run lacks)
        let c_abs = mean(&centered_abs);
        let expected_abs = mu_sq + shell;
        assert!(
            (c_abs - expected_abs).abs() / expected_abs < 0.10,
            "centered ‖θ‖² = {c_abs}, expected ‖μ‖² + Pσ² = {expected_abs}"
        );
        assert!(
            c_abs > 10.0 * shell,
            "centered chain collapsed toward the origin: ‖θ‖² = {c_abs} (‖μ‖² = {mu_sq})"
        );
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
                0,
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
                0,
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
