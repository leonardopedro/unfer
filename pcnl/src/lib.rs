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
//! * [`manifest`] — R0.6's run manifest: code revision and dirty-diff identity,
//!   data and anchor hashes, the evaluated schedule, the phase window and the
//!   metric-freeze record, written beside the saved draws.
//! * [`subspace`] — R1's validated affine chart `θ(z) = θ_ref + C_half B z`
//!   (orthonormality gate, lift, chain-rule gradient).
//! * [`target`] — R1's target identity on `z` and the exact MH-pCN kernel in
//!   `z`, validated against quadrature on a rank-2 double well.
//! * [`readout`] — §9.3 (M1/M4) of the ngpt research plan: proper scores
//!   of the *chain* posterior predictive, the exact finite-`m` Brier
//!   identity, and the leave-one-out reward baseline the RL arm needs.
//! * [`budget`] — the information–volume budget: the **exact** Haar cap-mass
//!   law `½I_{sin²d}((m−1)/2, ½)` and the number of directions a finite data
//!   set can localize (`m* = N·ΔL / ln(1/sin d)`).  This is the arithmetic
//!   behind `docs/PCNL_STATUS.md`'s "π is prior-dominated", and it explains
//!   the free-walk exponents (§6.5.9).
//! * [`edge0_world`] — E0 of the R9 track: finite synthetic worlds with an
//!   exact conditional-outcome oracle, disjoint split manifests, declared
//!   utilities and the *joint* (not product-of-marginals) law.
//! * [`edge0_model`] — E1: the small grouped latent-MoE decision model, its
//!   typed Noul/Choice/Score heads, parameter census and state cache key.
//! * [`edge0_rlcd`] — E2: the four-path policy, exact deployed marginal, the
//!   Brier U-statistic and its score-function/direct gradient identities.
//! * [`edge0_campaign`] — E3/E4 mechanics: shared warm start, CE+Brier and
//!   RLCD arms, exact-marginal control, frozen policy, paired world-clustered
//!   evaluation. Reduced-budget by default — not the declared R9 campaign.
//! * [`edge0_potential`] — E5: the frozen-encoder head chart as a pCNL
//!   [`target::Potential`], using the exact latent marginal as its likelihood.
//! * [`certificate`] — R8/D0–D1: the claim record, target/kernel identity and
//!   the evidence/minorization certificate `c = ω Z_lo/M`.
//! * [`dim_benchmark`] — R8/D2: the product target `π_r ⊗ N(0, I_n)` and the
//!   dimension-free product-kernel gap.
//! * [`linear_gaussian`] — R8/D4: the coupled linear-Gaussian benchmark with
//!   analytic slice, informed marginal, complement conditional and predictive.
//! * [`approx_pricing`] — R8/D3: exact Euler–Langevin Gaussian canaries (bias,
//!   stability limit, gradient-noise inflation) and the declared error budget.
//! * [`text_head`] — a frozen external text-embedding table (UltraBERT) plus a
//!   trainable softmax head as a [`target::Potential`].  Only the head is in
//!   the posterior; the 150M encoder is fixed data.
//! * [`continual`] — the sequential-Bayes seam: a diagonal Gaussian fitted to
//!   posterior particles becomes the next round's pCN prior.  Its approximation
//!   is the diagonal summary, not the exact kernel.
//! * [`operator_cert`] — R8/D6: the Gaussian canary's full-space L² spectral
//!   certificate, and the scoped missing estimate for the DT operator.
//! * [`radial`] — the same inequality along the radius: Haar's log-derivative
//!   `(P−2)·cot d` against `N·L′(d)`, and the exact mode equation
//!   `N·L′(d)·tan d = P − 2` whose left side the sweep prints. §5's
//!   "no mode anywhere on the sphere" is a comparison of those two numbers.

pub mod approx_pricing;
pub mod budget;
pub mod certificate;
pub mod chart_lipschitz;
pub mod continual;
pub mod diag;
pub mod dim_benchmark;
pub mod distill;
pub mod dt_arm;
pub mod dt_subspace;
pub mod edge0_campaign;
pub mod edge0_model;
pub mod edge0_potential;
pub mod edge0_rlcd;
pub mod edge0_world;
pub mod ensemble;
pub mod ensemble_diag;
pub mod evidence_enclosure;
pub mod experiment;
pub mod linear_gaussian;
pub mod manifest;
pub mod mdp;
pub mod model;
pub mod normalize;
pub mod operator_cert;
pub mod persist;
pub mod policy_model;
pub mod policy_readout;
pub mod r9_report;
pub mod radial;
pub mod readout;
pub mod rl_track;
pub mod rotation_head;
pub mod sampler;
pub mod satisficing;
pub mod sphere;
pub mod subspace;
pub mod swa;
pub mod target;
pub mod task;
pub mod text_head;
pub mod vector_model;
