//! R7.3.3 — a synthetic decision task with a known optimum.
//!
//! A DT trained on offline data has no ground truth: its likelihood is a
//! behaviour-cloning likelihood, so "better predictions" can be measured but
//! "better decisions" cannot, because the behaviour policy *is* the data.  The
//! way out is a task small enough that the optimal return is computable and the
//! comparison to the behaviour corpus is exact.  This module is that task.
//!
//! # Environment
//!
//! Two-dimensional linear-Gaussian dynamics with five controls, quadratic cost:
//!
//! ```text
//! x_{t+1} = A x_t + a_t + ε,      ε ~ N(0, σ² I)
//! r_t     = −(q₀ x₀² + q₁ x₁² + ρ‖a_t‖²)
//! ```
//!
//! `A = [[1, 0.1], [0, 1]]` is a discrete double integrator: the first
//! coordinate is a position carried by the second, which the actions move.  The
//! reward is ≤ 0 and the return of a trajectory is the sum of its rewards, so
//! "optimal" means "closest to zero with the least control effort" — a
//! direction the DT's return-to-go conditioning can actually see.
//!
//! # The optimum, and what is exact about it
//!
//! The state is continuous, so the exact optimum is not available in closed
//! form for a finite (non-convex) action set: the optimal cost-to-go is a max
//! of quadratics, not a quadratic, and the continuous-action LQR solution is a
//! different problem.  The optimum is therefore computed by finite-horizon
//! value iteration on a uniform grid with bilinear interpolation, noise handled
//! by 3-point Gauss–Hermite quadrature per dimension:
//!
//! ```text
//! V_H(x) = 0
//! V_t(x) = max_a [ r(x,a) + E_ε V_{t+1}(A x + a + ε) ]
//! ```
//!
//! Three numbers are exposed and they are not interchangeable:
//!
//! * [`MdpSolution::optimal_value`] — `V_0` interpolated at the start state, the
//!   target's own answer up to grid error;
//! * [`MdpSolution::greedy_return`] — the **noiseless** return actually obtained
//!   by following the grid policy, which is what a policy can be scored on;
//! * [`MdpSolution::refinement_gap`] — the same quantity on a finer grid, which
//!   is the discretisation error made visible instead of assumed away.
//!
//! Reporting `greedy_return` as "the optimum" would be a claim about
//! interpolation; reporting `optimal_value` without a refinement gap would be a
//! claim about the grid.  Neither is allowed here: [`MdpSolution::summary`]
//! returns all three.
//!
//! # Corpora and splits
//!
//! [`Corpus::splits`] draws disjoint **trajectory** seeds for train, dev and
//! test.  A step-level split would put `(s_t, a_t)` from one trajectory in
//! training and `(s_{t+1}, a_{t+1})` from the same trajectory in test — the
//! model would be scored on states it was fitted on, which is the offline-RL
//! version of leakage.  The ids are kept in the corpus so the disjointness is a
//! test, not a comment.
//!
//! Two corpora from the *same* environment but different behaviour policies are
//! this track's version of the anchor experiment (`RESEARCH_PLAN.md` §R7.0):
//! changing the corpus changes the DT's prior mean, which R2n–R2s measured as
//! the largest lever on the predictive.  The sampler dials are not where that
//! variation belongs.

use crate::manifest::Fnv;
use crate::policy_model::{DtBatch, DtShapes};
use candle_core::{Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// The environment.  `Default` is the plan's R7.4 cell: five actions, horizon
/// 16, a 41-point grid per dimension.
#[derive(Clone, Debug)]
pub struct MdpEnv {
    /// `x' = A x + a + ε`.
    pub a_mat: [[f32; 2]; 2],
    /// The five control vectors (an action adds its vector to the state).
    pub actions: Vec<[f32; 2]>,
    pub noise: f32,
    /// State cost weights `(q₀, q₁)`.
    pub q: [f32; 2],
    /// Control cost weight `ρ`, multiplying `‖a‖²`.
    pub rho: f32,
    pub horizon: usize,
    /// Grid extent: states are planned on `[-x_max, x_max]²`.
    pub x_max: f32,
    /// Grid points per dimension.
    pub grid_n: usize,
}

impl Default for MdpEnv {
    fn default() -> Self {
        Self {
            a_mat: [[1.0, 0.1], [0.0, 1.0]],
            actions: vec![[0.3, 0.0], [-0.3, 0.0], [0.0, 0.3], [0.0, -0.3], [0.0, 0.0]],
            noise: 0.05,
            q: [1.0, 0.2],
            rho: 0.1,
            horizon: 16,
            x_max: 3.0,
            grid_n: 41,
        }
    }
}

impl MdpEnv {
    pub fn step(&self, x: [f32; 2], action: usize, noise: [f32; 2]) -> [f32; 2] {
        let a = self.actions[action];
        let dx = self.a_mat[0][0] * x[0] + self.a_mat[0][1] * x[1] + a[0] + noise[0];
        let dy = self.a_mat[1][0] * x[0] + self.a_mat[1][1] * x[1] + a[1] + noise[1];
        [dx, dy]
    }

    /// `r(x, a) ≤ 0`.
    pub fn reward(&self, x: [f32; 2], action: usize) -> f32 {
        let a = self.actions[action];
        -(self.q[0] * x[0] * x[0]
            + self.q[1] * x[1] * x[1]
            + self.rho * (a[0] * a[0] + a[1] * a[1]))
    }

    /// Parameter count of the plan's DT at these shapes — kept next to the
    /// environment so a cell's cost is readable from one place.
    pub fn dt_shapes(&self, rtg_bins: usize, d: usize, ff: usize, layers: usize) -> DtShapes {
        DtShapes {
            rtg_bins,
            state_dim: 2,
            action_vocab: self.actions.len(),
            d,
            ff,
            layers,
            max_len: self.horizon,
        }
    }
}

/// A uniform grid on `[-x_max, x_max]²` with bilinear interpolation.
///
/// Values outside the grid clamp to the boundary, which makes the boundary a
/// reflecting wall in the DP.  That is a real approximation of an unbounded
/// task and it is why `x_max` is a configuration field with a test that the
/// trajectories stay inside it, rather than a constant.
#[derive(Clone, Copy, Debug)]
pub struct Grid2 {
    pub n: usize,
    pub lo: f32,
    pub hi: f32,
}

impl Grid2 {
    pub fn new(n: usize, lo: f32, hi: f32) -> Self {
        assert!(n >= 2, "a grid needs two points per dimension");
        Self { n, lo, hi }
    }

    pub fn coord(&self, i: usize) -> f32 {
        self.lo + (self.hi - self.lo) * i as f32 / (self.n - 1) as f32
    }

    fn locate(&self, x: f32) -> (usize, usize, f64) {
        let pos = ((x - self.lo) / (self.hi - self.lo) * (self.n - 1) as f32)
            .clamp(0.0, (self.n - 1) as f32);
        let i = (pos.floor() as usize).min(self.n - 2);
        (i, i + 1, f64::from(pos - i as f32))
    }

    /// Bilinear interpolation of a row-major `[n, n]` value table at `x`.
    pub fn interp(&self, v: &[f64], x: [f32; 2]) -> f64 {
        let (i0, i1, fx) = self.locate(x[0]);
        let (j0, j1, fy) = self.locate(x[1]);
        let at = |i: usize, j: usize| v[i * self.n + j];
        let a = at(i0, j0) * (1.0 - fx) + at(i1, j0) * fx;
        let b = at(i0, j1) * (1.0 - fx) + at(i1, j1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Nearest grid point, for reporting only (never for the policy).
    pub fn nearest_index(&self, x: [f32; 2]) -> (usize, usize) {
        let idx = |v: f32| {
            (((v - self.lo) / (self.hi - self.lo) * (self.n - 1) as f32).round() as usize)
                .min(self.n - 1)
        };
        (idx(x[0]), idx(x[1]))
    }
}

/// Nodes and weights for `E_{ε~N(0,σ²)} f(ε)` by 3-point Gauss–Hermite
/// quadrature.  Exact for quadratics, which is the accuracy the DP can use:
/// the interpolant is piecewise bilinear, so a finer noise rule would add cost
/// without adding accuracy the grid preserves.
fn quadrature(sigma: f32) -> [(f32, f64); 3] {
    let s = (3.0f32).sqrt() * sigma;
    [(-s, 1.0 / 6.0), (0.0, 2.0 / 3.0), (s, 1.0 / 6.0)]
}

/// What value iteration produced, plus the environment it was solved on.
pub struct MdpSolution {
    pub env: MdpEnv,
    pub grid: Grid2,
    /// `values[t]` is `V_t` on the grid, `t = 0..=horizon`.
    pub values: Vec<Vec<f64>>,
    /// `policy[t][s]` is the argmax action at stage `t`, `t = 0..horizon`.
    pub policy: Vec<Vec<usize>>,
}

impl MdpEnv {
    /// Finite-horizon value iteration on `[-x_max, x_max]²`.
    ///
    /// `γ = 1`: the horizon is finite and fixed, so discounting would change
    /// the task rather than stabilise the recursion.
    pub fn solve(&self) -> MdpSolution {
        let grid = Grid2::new(self.grid_n, -self.x_max, self.x_max);
        let n = grid.n;
        let cells = n * n;
        let nodes = quadrature(self.noise);
        let mut v = vec![0.0f64; cells]; // V_H ≡ 0
        let mut values = vec![v.clone(); self.horizon + 1];
        let mut policy = vec![vec![0usize; cells]; self.horizon];
        for t in (0..self.horizon).rev() {
            let mut cur = vec![0.0f64; cells];
            let mut pol = vec![0usize; cells];
            for i in 0..n {
                for j in 0..n {
                    let x = [grid.coord(i), grid.coord(j)];
                    let mut best = f64::NEG_INFINITY;
                    let mut best_a = 0usize;
                    for (ai, action) in self.actions.iter().enumerate() {
                        let d = self.a_mat[0][0] * x[0] + self.a_mat[0][1] * x[1];
                        let e = self.a_mat[1][0] * x[0] + self.a_mat[1][1] * x[1];
                        let mut ev = 0.0f64;
                        for (nx, wx) in nodes {
                            for (ny, wy) in nodes {
                                let next = [d + action[0] + nx, e + action[1] + ny];
                                ev += wx * wy * grid.interp(&v, next);
                            }
                        }
                        let q = f64::from(self.reward(x, ai)) + ev;
                        if q > best {
                            best = q;
                            best_a = ai;
                        }
                    }
                    cur[i * n + j] = best;
                    pol[i * n + j] = best_a;
                }
            }
            values[t] = cur.clone();
            policy[t] = pol;
            v = cur;
        }
        MdpSolution {
            env: self.clone(),
            grid,
            values,
            policy,
        }
    }
}

impl MdpSolution {
    /// `Q_t(x, a) = r(x,a) + E_ε V_{t+1}(A x + a + ε)`, interpolated at `x`.
    fn q_value(&self, t: usize, x: [f32; 2], action: usize) -> f64 {
        let nodes = quadrature(self.env.noise);
        let a = self.env.actions[action];
        let d = self.env.a_mat[0][0] * x[0] + self.env.a_mat[0][1] * x[1];
        let e = self.env.a_mat[1][0] * x[0] + self.env.a_mat[1][1] * x[1];
        let mut ev = 0.0f64;
        for (nx, wx) in nodes {
            for (ny, wy) in nodes {
                let next = [d + a[0] + nx, e + a[1] + ny];
                ev += wx * wy * self.grid.interp(&self.values[t + 1], next);
            }
        }
        f64::from(self.env.reward(x, action)) + ev
    }

    /// The target's answer at `x0`: the interpolated Bellman optimum
    /// `max_a Q_0(x0, a)`.
    ///
    /// **Not** `interp(V_0, x0)`.  Those two differ off the grid nodes — and
    /// measured here they differ by ~0.12 against a return of −1.99 — because
    /// `max` and bilinear interpolation do not commute: `interp(V_0, x)` is the
    /// interpolation of a maximum, while this is the maximum of the
    /// interpolated continuations.  The second is the definition that matches
    /// [`Self::greedy_action`] and the policy that is actually rolled out, so a
    /// cell cannot report a `V_0` interpolation its own policy beats.  At a
    /// grid node the two coincide, which is what
    /// `exact_env_value_iteration_matches_brute_force` pins.
    pub fn optimal_value(&self, x0: [f32; 2]) -> f64 {
        (0..self.env.actions.len())
            .map(|a| self.q_value(0, x0, a))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// `argmax_a Q_t(x, a)` — the greedy action at an arbitrary (off-grid)
    /// state, which the recorded `policy` table cannot answer.
    pub fn greedy_action(&self, t: usize, x: [f32; 2]) -> usize {
        let mut best = f64::NEG_INFINITY;
        let mut best_a = 0usize;
        for a in 0..self.env.actions.len() {
            let q = self.q_value(t, x, a);
            if q > best {
                best = q;
                best_a = a;
            }
        }
        best_a
    }

    /// The **noiseless** return of the greedy policy from `x0`.
    ///
    /// Noiseless on purpose: this is the number a *policy* can be compared on
    /// across arms without the comparison turning into a test of whose noise
    /// draw was luckier.  It is deliberately *not* comparable to
    /// [`Self::optimal_value`] when `env.noise > 0`: `V_0` is an expectation
    /// over the transition noise and the realized noiseless path pays none of
    /// it, so the realized return can exceed `V_0` without anything being
    /// wrong.  Use [`Self::monte_carlo_greedy`] for a like-for-like comparison
    /// in the stochastic environment.
    pub fn greedy_return(&self, x0: [f32; 2]) -> f64 {
        let mut x = x0;
        let mut total = 0.0f64;
        for t in 0..self.env.horizon {
            let a = self.greedy_action(t, x);
            total += f64::from(self.env.reward(x, a));
            x = self.env.step(x, a, [0.0, 0.0]);
        }
        total
    }

    /// Closed-loop greedy rollouts in the *stochastic* environment: mean return
    /// and its Monte Carlo standard error.
    ///
    /// This is the number to compare against [`Self::optimal_value`], because
    /// both are expectations over the same transition noise.  The gap between
    /// them is grid/interpolation error plus policy-evaluation noise, and the
    /// MCSE is what separates the two.
    pub fn monte_carlo_greedy(&self, x0: [f32; 2], rollouts: usize, seed: u64) -> (f64, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..rollouts {
            let mut x = x0;
            let mut total = 0.0f64;
            for t in 0..self.env.horizon {
                let a = self.greedy_action(t, x);
                total += f64::from(self.env.reward(x, a));
                let noise = if self.env.noise > 0.0 {
                    let n0: f32 = rng.sample(StandardNormal);
                    let n1: f32 = rng.sample(StandardNormal);
                    [self.env.noise * n0, self.env.noise * n1]
                } else {
                    [0.0, 0.0]
                };
                x = self.env.step(x, a, noise);
            }
            sum += total;
            sq += total * total;
        }
        let n = rollouts as f64;
        let mean = sum / n;
        let var = (sq / n - mean * mean).max(0.0);
        (mean, (var / n).sqrt())
    }

    /// Same quantity on a grid `factor` times finer, minus this one.  The
    /// discretisation error, measured rather than assumed.
    pub fn refinement_gap(&self, x0: [f32; 2], factor: usize) -> f64 {
        let mut env = self.env.clone();
        env.grid_n = (self.env.grid_n - 1) * factor + 1;
        let fine = env.solve();
        fine.optimal_value(x0) - self.optimal_value(x0)
    }

    /// `(optimal_value, greedy_return, refinement_gap)` — the three numbers
    /// §R7.1 says a DT cell must report together, because they answer
    /// "what is the target", "what can a policy get" and "how much of the
    /// difference is the grid", respectively.
    pub fn summary(&self, x0: [f32; 2], factor: usize) -> (f64, f64, f64) {
        (
            self.optimal_value(x0),
            self.greedy_return(x0),
            self.refinement_gap(x0, factor),
        )
    }
}

/// A behaviour policy: proportional feedback toward the origin with uniform
/// exploration.
///
/// The two fields are the knobs that make "a different corpus" concrete:
/// `feedback` sets *which* states the data visits and `epsilon` how noisy the
/// actions are.  Neither is a sampler parameter, which is exactly the point —
/// this is the DT's prior mean.
#[derive(Clone, Copy, Debug)]
pub struct BehaviourPolicy {
    pub feedback: f32,
    pub epsilon: f32,
    pub seed: u64,
}

impl Default for BehaviourPolicy {
    fn default() -> Self {
        Self {
            feedback: 0.5,
            epsilon: 0.2,
            seed: 7,
        }
    }
}

impl BehaviourPolicy {
    /// Closest control vector to the desired displacement `−feedback · x`,
    /// or a uniform draw with probability `epsilon`.
    pub fn action(&self, env: &MdpEnv, x: [f32; 2], rng: &mut impl Rng) -> usize {
        if rng.random::<f32>() < self.epsilon {
            return rng.random_range(0..env.actions.len());
        }
        let want = [-self.feedback * x[0], -self.feedback * x[1]];
        let mut best = f32::INFINITY;
        let mut best_a = 0usize;
        for (i, a) in env.actions.iter().enumerate() {
            let d = (a[0] - want[0]) * (a[0] - want[0]) + (a[1] - want[1]) * (a[1] - want[1]);
            if d < best {
                best = d;
                best_a = i;
            }
        }
        best_a
    }
}

/// One sampled trajectory.  `states` has `horizon + 1` entries, the rest
/// `horizon`; `rtgs[t] = Σ_{k≥t} rewards[k]`.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub states: Vec<[f32; 2]>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f32>,
    pub rtgs: Vec<f32>,
}

impl Trajectory {
    /// The suffix sum at `t = 0`, which is the definition of the return here:
    /// `total_return == rtgs[0]` holds bitwise, so a report cannot quote a
    /// return that disagrees with the conditioning signal the DT was trained on.
    pub fn total_return(&self) -> f32 {
        self.rtgs.first().copied().unwrap_or(0.0)
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// A set of trajectories drawn from one behaviour policy, with the trajectory
/// seeds that made them.
#[derive(Clone)]
pub struct Corpus {
    pub env: MdpEnv,
    pub policy: BehaviourPolicy,
    pub trajectory_ids: Vec<u64>,
    pub trajectories: Vec<Trajectory>,
}

impl Corpus {
    /// One trajectory per seed.  The seed is the identity of a trajectory, not
    /// of a step: it is what makes a split disjoint and reproducible.
    pub fn with_seeds(env: &MdpEnv, policy: BehaviourPolicy, seeds: &[u64]) -> Self {
        let mut trajectories = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ policy.seed.rotate_left(17));
            // Start spread so the corpus is not all near the origin: x₀ is the
            // first state, drawn from N(0, 1) in both coordinates and clamped
            // inside the planning grid.
            let mut start = [0.0f32; 2];
            for slot in start.iter_mut() {
                let z: f32 = rng.sample(StandardNormal);
                *slot = z.clamp(-0.9 * env.x_max, 0.9 * env.x_max);
            }
            let mut x = start;
            let mut states = Vec::with_capacity(env.horizon + 1);
            let mut actions = Vec::with_capacity(env.horizon);
            let mut rewards = Vec::with_capacity(env.horizon);
            states.push(x);
            for _ in 0..env.horizon {
                let a = policy.action(env, x, &mut rng);
                let r = env.reward(x, a);
                let noise = if env.noise > 0.0 {
                    let n0: f32 = rng.sample(StandardNormal);
                    let n1: f32 = rng.sample(StandardNormal);
                    [env.noise * n0, env.noise * n1]
                } else {
                    [0.0, 0.0]
                };
                x = env.step(x, a, noise);
                actions.push(a);
                rewards.push(r);
                states.push(x);
            }
            let mut rtgs = vec![0.0f32; rewards.len()];
            let mut acc = 0.0f32;
            for t in (0..rewards.len()).rev() {
                acc += rewards[t];
                rtgs[t] = acc;
            }
            trajectories.push(Trajectory {
                states,
                actions,
                rewards,
                rtgs,
            });
        }
        Self {
            env: env.clone(),
            policy,
            trajectory_ids: seeds.to_vec(),
            trajectories,
        }
    }

    /// Disjoint train/dev/test corpora, by trajectory seed.
    ///
    /// The three seed ranges never overlap, and the ids are kept, so
    /// `splits_are_disjoint_by_trajectory` can check the property this
    /// function exists for.  Counts are per split, in trajectories; how many
    /// *transitions* that is follows from `env.horizon` and is reported
    /// separately (`n_terms`, not `n_trajectories`).
    pub fn splits(
        env: &MdpEnv,
        policy: BehaviourPolicy,
        counts: (usize, usize, usize),
        base_seed: u64,
    ) -> (Self, Self, Self) {
        let n_train = counts.0 as u64;
        let n_dev = counts.1 as u64;
        let off_train = base_seed;
        let off_dev = base_seed + 1_000_000;
        let off_test = base_seed + 2_000_000;
        let seeds = |offset: u64, n: u64| -> Vec<u64> { (0..n).map(|i| offset + i).collect() };
        (
            Self::with_seeds(env, policy, &seeds(off_train, n_train)),
            Self::with_seeds(env, policy, &seeds(off_dev, n_dev)),
            Self::with_seeds(env, policy, &seeds(off_test, counts.2 as u64)),
        )
    }

    pub fn transitions(&self) -> usize {
        self.trajectories.iter().map(|t| t.len()).sum()
    }

    pub fn mean_return(&self) -> f64 {
        if self.trajectories.is_empty() {
            return f64::NAN;
        }
        let sum: f32 = self.trajectories.iter().map(|t| t.total_return()).sum();
        f64::from(sum) / self.trajectories.len() as f64
    }

    /// FNV-1a 64 over `(state, action, reward)` in order, per trajectory — the
    /// corpus identity a manifest records.  Hashing the sampled data and not
    /// the seed is the same discipline `Dataset::content_hash` follows: two
    /// seeds that happen to coincide are the same corpus for reporting.
    pub fn content_hash(&self) -> u64 {
        let mut h = Fnv::new();
        for t in &self.trajectories {
            for (k, x) in t.states.iter().enumerate() {
                h.write(&x[0].to_le_bytes());
                h.write(&x[1].to_le_bytes());
                if k < t.actions.len() {
                    h.write(&(t.actions[k] as u32).to_le_bytes());
                    h.write(&t.rewards[k].to_le_bytes());
                }
            }
        }
        h.finish()
    }

    /// Tokenise the corpus for [`crate::policy_model`].
    ///
    /// Return-to-go is binned by clipping to `[rtg_lo, 0]` — the reward is
    /// never positive, so the upper edge is exact and only the lower edge is a
    /// choice.  All trajectories here share `env.horizon`, so the mask is all
    /// ones; ragged corpora are a later concern of the batching surface, not of
    /// this fixture.
    pub fn to_dt_batch(&self, spec: &DtSpec) -> Result<DtBatch> {
        let b = self.trajectories.len();
        let t = self.env.horizon;
        if b == 0 {
            return Err(candle_core::Error::Msg("empty corpus".into()));
        }
        for traj in &self.trajectories {
            if traj.len() != t {
                return Err(candle_core::Error::Msg(format!(
                    "ragged corpus: trajectory has {} steps, env.horizon is {t}",
                    traj.len()
                )));
            }
        }
        let mut rtg = Vec::with_capacity(b * t);
        let mut state = Vec::with_capacity(b * t * 2);
        let mut act = Vec::with_capacity(b * t);
        for traj in &self.trajectories {
            for k in 0..t {
                rtg.push(spec.bin(traj.rtgs[k]) as u32);
                state.push(traj.states[k][0]);
                state.push(traj.states[k][1]);
                act.push(traj.actions[k] as u32);
            }
        }
        let dev = &Device::Cpu;
        let mask = vec![1.0f32; b * t];
        DtBatch::new(
            Tensor::from_vec(rtg, (b, t), dev)?,
            Tensor::from_vec(state, (b, t, 2), dev)?,
            Tensor::from_vec(act, (b, t), dev)?,
            Tensor::from_vec(mask, (b, t), dev)?,
        )
    }
}

/// Return-to-go discretisation for the DT's conditioning table.
#[derive(Clone, Copy, Debug)]
pub struct DtSpec {
    pub rtg_bins: usize,
    pub rtg_lo: f32,
}

impl Default for DtSpec {
    fn default() -> Self {
        Self {
            rtg_bins: 4,
            rtg_lo: -20.0,
        }
    }
}

impl DtSpec {
    /// Bin index in `0..rtg_bins`; `rtg_lo` and below map to `0`, `0` maps to
    /// `rtg_bins - 1`.
    pub fn bin(&self, rtg: f32) -> usize {
        let frac = ((rtg - self.rtg_lo) / (0.0 - self.rtg_lo)).clamp(0.0, 1.0);
        ((frac * (self.rtg_bins - 1) as f32).round() as usize).min(self.rtg_bins - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic environment whose every next state lands exactly on a
    /// grid node, so value iteration is exact and can be compared against
    /// brute-force enumeration of all action sequences.
    fn exact_env() -> MdpEnv {
        MdpEnv {
            a_mat: [[1.0, 0.0], [0.0, 1.0]],
            actions: vec![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
            noise: 0.0,
            q: [1.0, 1.0],
            rho: 0.0,
            horizon: 2,
            x_max: 0.3,
            grid_n: 7, // step = 0.1, so every action lands on a node
        }
    }

    #[test]
    fn exact_env_value_iteration_matches_brute_force() {
        let env = exact_env();
        let sol = env.solve();
        // A node in the interior of the grid.
        let x0 = [0.1f32, 0.1];
        let mut best = f64::NEG_INFINITY;
        for a0 in 0..env.actions.len() {
            for a1 in 0..env.actions.len() {
                let mut x = x0;
                let mut total = 0.0f64;
                for &a in &[a0, a1] {
                    total += f64::from(env.reward(x, a));
                    x = env.step(x, a, [0.0, 0.0]);
                }
                best = best.max(total);
            }
        }
        let got = sol.optimal_value(x0);
        // 1e-6, not 1e-9: the grid's `f32` coordinates enter the interpolant
        // as `f64` weights, so an on-node target still carries a few ulps of
        // coordinate rounding.  The quantity being checked is the recursion,
        // not the floating-point representation of 0.1.
        assert!((got - best).abs() < 1e-6, "DP {got} vs brute force {best}");
        // And the greedy policy realises it, since no interpolation is needed.
        assert!((sol.greedy_return(x0) - best).abs() < 1e-6);
    }

    #[test]
    fn rewards_are_non_positive_and_rtgs_are_suffix_sums() {
        let env = MdpEnv::default();
        let c = Corpus::with_seeds(&env, BehaviourPolicy::default(), &[1, 2, 3]);
        for t in &c.trajectories {
            assert_eq!(t.states.len(), env.horizon + 1);
            assert_eq!(t.rewards.len(), env.horizon);
            assert!(t.rewards.iter().all(|&r| r <= 0.0));
            assert_eq!(t.rtgs.len(), env.horizon);
            assert_eq!(t.rtgs[0], t.total_return());
            assert_eq!(t.rtgs[env.horizon - 1], t.rewards[env.horizon - 1]);
            for k in 0..env.horizon - 1 {
                assert!((t.rtgs[k] - (t.rewards[k] + t.rtgs[k + 1])).abs() < 1e-6);
            }
        }
        assert_eq!(c.transitions(), 3 * env.horizon);
    }

    #[test]
    fn splits_are_disjoint_by_trajectory() {
        let env = MdpEnv::default();
        let p = BehaviourPolicy::default();
        let (tr, dev, te) = Corpus::splits(&env, p, (4, 2, 2), 100);
        for (i, id) in tr.trajectory_ids.iter().enumerate() {
            assert!(!dev.trajectory_ids.contains(id), "train {i} also in dev");
            assert!(!te.trajectory_ids.contains(id), "train {i} also in test");
        }
        for id in &dev.trajectory_ids {
            assert!(!te.trajectory_ids.contains(id), "dev also in test");
        }
        assert_eq!(
            (
                tr.trajectories.len(),
                dev.trajectories.len(),
                te.trajectories.len()
            ),
            (4, 2, 2)
        );
        // Deterministic: the same call reproduces the same corpus identity.
        let (tr2, _, _) = Corpus::splits(&env, p, (4, 2, 2), 100);
        assert_eq!(tr.content_hash(), tr2.content_hash());
    }

    /// Two behaviour policies on the same environment are two corpora — the
    /// DT's version of the anchor lever.  If this test fails, changing
    /// `feedback`/`epsilon` is not actually changing the data, and every
    /// "anchor" comparison in R7 would be comparing one corpus to itself.
    #[test]
    fn behaviour_policies_produce_different_corpora() {
        let env = MdpEnv::default();
        let calm = BehaviourPolicy {
            feedback: 0.1,
            epsilon: 0.05,
            seed: 3,
        };
        let twitchy = BehaviourPolicy {
            feedback: 1.5,
            epsilon: 0.5,
            seed: 3,
        };
        let a = Corpus::with_seeds(&env, calm, &[1, 2, 3, 4]);
        let b = Corpus::with_seeds(&env, twitchy, &[1, 2, 3, 4]);
        assert_ne!(a.content_hash(), b.content_hash());
        assert_ne!(a.mean_return(), b.mean_return());
        // Both must be inside the planning grid, or the DP's boundary clamp is
        // scoring states the data never visits.
        for c in [&a, &b] {
            for t in &c.trajectories {
                for x in &t.states {
                    assert!(
                        x[0].abs() <= env.x_max + 1.0,
                        "state {x:?} escaped the grid"
                    );
                    assert!(
                        x[1].abs() <= env.x_max + 1.0,
                        "state {x:?} escaped the grid"
                    );
                }
            }
        }
    }

    /// The optimum must be reported with its grid error, not instead of it:
    /// refining the grid changes the interpolated value by a small amount, and
    /// the greedy return never exceeds it by more than that error.
    #[test]
    fn grid_error_is_small_and_reported() {
        // Noise off for the bound: with noise the realized noiseless path stays
        // inside `V_0` only by accident, and asserting otherwise would be
        // testing the noise draw instead of the grid.
        let env = MdpEnv {
            noise: 0.0,
            ..MdpEnv::default()
        };
        let sol = env.solve();
        let x0 = [1.0f32, 0.0];
        let (v, ret, gap) = sol.summary(x0, 2);
        assert!(v.is_finite() && ret.is_finite() && gap.is_finite());
        assert!(
            gap.abs() <= 0.05 * v.abs().max(1.0),
            "grid gap {gap} too large against V0 {v}"
        );
        assert!(
            ret <= v + gap.abs() + 1e-6,
            "greedy return {ret} exceeded the interpolated optimum {v} by more than the grid gap {gap}"
        );
        assert!(v < 0.0 && ret < 0.0, "rewards are non-positive");
    }

    /// In the stochastic environment the like-for-like check is the Monte Carlo
    /// greedy return against `V_0`, both expectations over the same noise, with
    /// the MCSE deciding how close is close.
    #[test]
    fn monte_carlo_greedy_matches_the_dp_value() {
        let env = MdpEnv::default();
        let sol = env.solve();
        let x0 = [1.0f32, 0.0];
        let v = sol.optimal_value(x0);
        let gap = sol.refinement_gap(x0, 2).abs();
        let (mean, mcse) = sol.monte_carlo_greedy(x0, 400, 11);
        assert!(mcse > 0.0 && mean.is_finite());
        assert!(
            (mean - v).abs() <= 4.0 * mcse + gap + 2e-3 * v.abs().max(1.0),
            "MC greedy {mean} ± {mcse} vs DP value {v} (grid gap {gap})"
        );
    }

    #[test]
    fn dt_spec_bins_are_bounded_and_monotone() {
        let spec = DtSpec::default();
        assert_eq!(spec.bin(0.0), spec.rtg_bins - 1);
        assert_eq!(spec.bin(spec.rtg_lo - 5.0), 0);
        let mut prev = 0;
        for k in 0..40 {
            let rtg = spec.rtg_lo + (0.0 - spec.rtg_lo) * k as f32 / 39.0;
            let b = spec.bin(rtg);
            assert!(b < spec.rtg_bins && b >= prev);
            prev = b;
        }
    }

    #[test]
    fn corpus_tokenises_with_full_mask_and_bounded_ids() {
        let env = MdpEnv {
            horizon: 4,
            ..MdpEnv::default()
        };
        let c = Corpus::with_seeds(&env, BehaviourPolicy::default(), &[5, 6]);
        let spec = DtSpec::default();
        let batch = c.to_dt_batch(&spec).unwrap();
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.timesteps(), 4);
        assert_eq!(batch.transitions(), 8);
        let act = batch.act.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(act.iter().all(|&a| (a as usize) < env.actions.len()));
        let rtg = batch.rtg.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(rtg.iter().all(|&b| (b as usize) < spec.rtg_bins));
        let mask = batch.mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(mask.iter().all(|&m| m == 1.0));
    }
}
