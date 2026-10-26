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
//!   `SgPcnlScheduler` (plan §3C; no acceptance feedback anywhere).
//! * [`task`] — the toy modular-arithmetic tasks, the mini-batch
//!   [`task::BatchLoader`], and the momentum-SGD baseline.
//! * [`persist`] — Week 2's `safetensors` trajectory persistence (§1).
//! * [`diag`] — Phase 4's ESS and split-R̂.
//! * [`experiment`] — Phases 0/3/4 as one deterministic measured run;
//!   independent chains run in parallel under `rayon` (Phase 2).
//! * [`readout`] — §9.3 (M1/M4) of the ngpt research plan: proper scores
//!   of the *chain* posterior predictive, the exact finite-`m` Brier
//!   identity, and the leave-one-out reward baseline the RL arm needs.

pub mod diag;
pub mod experiment;
pub mod model;
pub mod normalize;
pub mod persist;
pub mod readout;
pub mod sampler;
pub mod task;
