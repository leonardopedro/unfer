//! # pcnl — unadjusted SG-pCNL for a normalized (nGPT-lite) transformer
//!
//! Specification: [`docs/plan_bayes_ngpt.md`](../docs/plan_bayes_ngpt.md)
//! in this repository — updated 2026-10-20 to the **unadjusted**
//! stochastic-gradient pCNL: no Metropolis filter, one backward pass
//! per step on a streamed mini-batch, Robbins–Monro step-size decay.
//!
//! Module map, in the plan's phase order:
//!
//! * [`normalize`] — Phase 1's hyperspherical projections.
//! * [`model`] — Phase 1's flat-parameter toy transformer, both the
//!   cosine (nGPT-lite) and the standard variants on one `θ ∈ R^P`.
//! * [`sampler`] — Phase 2's `SgPcnlState::step_unadjusted` and the
//!   `SgPcnlScheduler` (plan §3C; no acceptance feedback anywhere). The
//!   exponent window is `κ ∈ (1/4, 1/2]` — the decreasing-step
//!   conditions are on the drift coefficient `β²σ²/2`, not on `β` — and
//!   `SgPcnlScheduler::total_mobility` reports the budget that decides
//!   whether an annealed run can reach the posterior at all.
//!   `SgPcnlState::lambda_at` is the likelihood ramp (`--anneal N`, a
//!   schedule and not a target change); §6.5.5 measures it as a null on
//!   the cosine chain, because the per-step drift is only
//!   `βσ‖g‖/(2√P) ≈ 5 %` of the pCN noise — a ramp scales that
//!   numerator and nothing else.
//! * [`task`] — the toy modular-arithmetic tasks, the mini-batch
//!   [`task::BatchLoader`], and the momentum-SGD baseline.
//! * [`persist`] — Week 2's `safetensors` trajectory persistence (§1).
//! * [`diag`] — Phase 4's ESS and split-R̂.
//! * [`experiment`] — Phases 0/3/4 as one deterministic measured run;
//!   independent chains run in parallel under `rayon` (Phase 2).
//! * [`readout`] — §9.3 (M1/M4) of the ngpt research plan: proper scores
//!   of the *chain* posterior predictive, the exact finite-`m` Brier
//!   identity, and the leave-one-out reward baseline the RL arm needs.
//! * [`budget`] — the information–volume budget: the **exact** Haar cap-mass
//!   law `½I_{sin²d}((m−1)/2, ½)` and the number of directions a finite data
//!   set can localize (`m* = N·ΔL / ln(1/sin d)`).  This is the arithmetic
//!   behind `docs/PCNL_STATUS.md`'s "π is prior-dominated", and it explains
//!   the free-walk exponents (§6.5.9).
//! * [`radial`] — the same inequality along the radius: Haar's log-derivative
//!   `(P−2)·cot d` against `N·L′(d)`, and the exact mode equation
//!   `N·L′(d)·tan d = P − 2` whose left side the sweep prints. §5's
//!   "no mode anywhere on the sphere" is a comparison of those two numbers.

pub mod budget;
pub mod diag;
pub mod experiment;
pub mod model;
pub mod normalize;
pub mod persist;
pub mod radial;
pub mod readout;
pub mod sampler;
pub mod sphere;
pub mod task;
