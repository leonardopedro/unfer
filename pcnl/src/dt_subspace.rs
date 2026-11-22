//! §R7.4 — the first DT adaptation cell: a rank-4 restricted target, with the
//! point baselines the plan lists.
//!
//! `dt_arm` assembled the ambient pieces at the **full** target and measured a
//! calibration-only result whose cause was visible in the same run: `P =
//! 17 408`, so `σ√P = 46.2` against a reached radius of 49.4 and the draws
//! spanned the entire prior shell.  This module is R7.4's answer to that: the
//! same anchor, corpora and readouts, but the chain runs on R1's affine chart
//! `θ(z) = θ_ref + C_half B z` with `r = 4`, so `z ∈ R⁴` and the prior `N(0, I₄)`
//! — inside this chart `σ` is a meaningful posterior width instead of a shell
//! radius.
//!
//! Everything here reuses the recorded machinery rather than re-deriving it:
//!
//! * the basis is [`crate::subspace::pca_basis`] of a Task-B fine-tuning path
//!   from the ambient anchor, i.e. R1's "informed" definition, with
//!   [`crate::subspace::Basis::random`] as the matched-rank control;
//! * the target is `SubspaceTarget<DtPotential>` — R7.3's `Potential` seam, so
//!   the likelihood is the classifier's sum-form algebra with the DT's per-step
//!   NLL in place of the classifier's cross-entropy;
//! * the kernels are R1's exact MH-pCN moves (`mh_pcn_z`, `mh_pcn_z_scaled`
//!   through `DiagScale`), invariant for the restricted target by construction;
//! * the readouts are `policy_readout`'s, so a score here and a score in
//!   `dt_cell` are the same function of the same distributions.
//!
//! **What this cell is not.**  `θ_ref` is Task A's pre-trained anchor and the
//! basis path is Task-B data, so the basis arm is empirical Bayes/data reuse in
//! R1's sense; the random control is what keeps that from being a free gain.
//! `λ = 1` here is not the classifier's `λ = 1`: `Φ` sums over transitions, and
//! a 768-transition corpus is a different effective concentration from 512
//! classifier examples.  And the restricted target is a *smaller model*, not an
//! approximation to the ambient posterior — R7.0 item 2 is explicit that the
//! restriction is the likely limit, which is why the random basis is a column
//! in every table and not an afterthought.

use crate::diag::{ess_bulk, gelman_rubin};
use crate::dt_arm::{DtCell, START, behaviour_policy, conditioning_bins};
use crate::manifest::Fnv;
use crate::mdp::{Corpus, MdpEnv};
use crate::policy_model::{DtPotential, dt_act_at, dt_loss_and_grad, init_flat};
use crate::policy_readout::{
    Arm, ArmReturns, Calibration, Context, Decode, PairedDiff, ReturnReadout, StepScores,
    dt_step_distributions_at, max_prob_calibration, mean_mcse, paired, readout, sample_at,
    score_steps,
};
use crate::readout::Rule;
use crate::subspace::{AffineSubspace, Basis, pca_basis};
use crate::target::{
    DiagScale, SubspaceTarget, ZPotential, ZStep, curvature_diag, mh_pcn_z, mh_pcn_z_scaled,
};
use candle_core::{Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

/// A parameter block the chart can be restricted to — R7.6's first open
/// question ("head-only, RTG-embedding-only, trunk-only or all-blocks informed
/// directions").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// The untied output head `w_out [D, A]`, the last block in the layout.
    Head,
    /// Everything before the head: the four embedding tables, the per-layer
    /// transformer blocks.
    Trunk,
    /// The RTG table `e_rtg [G, D]` alone — the DT's only return-conditioning
    /// block, and the smallest block named in R7.6's list.  It sits inside
    /// [`Block::Trunk`], so it is a finer restriction of the same path.
    Rtg,
}

/// Which basis the chart uses.  `Informed` is Task-B path PCA; `Random` is the
/// matched-rank control — an informed basis that cannot beat it is not
/// informed (R1's acceptance clause).
///
/// `Head`/`Trunk` are the *block-restricted* informed directions (the same path,
/// projected onto the block before the PCA), and `HeadRandom`/`TrunkRandom` are
/// their matched controls — a rank-`r` random basis drawn **inside** the same
/// block.  Matching the control to the block matters: a full-space random
/// basis has support the block-restricted informed basis cannot have, so
/// comparing the two would measure the rank, not the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisKind {
    Informed,
    Random,
    Head,
    Trunk,
    HeadRandom,
    TrunkRandom,
    Rtg,
    RtgRandom,
}

impl BasisKind {
    pub fn name(self) -> &'static str {
        match self {
            BasisKind::Informed => "informed",
            BasisKind::Random => "random",
            BasisKind::Head => "head",
            BasisKind::Trunk => "trunk",
            BasisKind::HeadRandom => "head-random",
            BasisKind::TrunkRandom => "trunk-random",
            BasisKind::Rtg => "rtg",
            BasisKind::RtgRandom => "rtg-random",
        }
    }

    /// Whether the basis came from a fine-tuning path (so a report should print
    /// its paired difference against the point baseline).  The random controls
    /// are not informed arms and do not get one.
    pub fn is_informed(self) -> bool {
        matches!(
            self,
            BasisKind::Informed | BasisKind::Head | BasisKind::Trunk | BasisKind::Rtg
        )
    }

    pub fn block(self) -> Option<Block> {
        match self {
            BasisKind::Head | BasisKind::HeadRandom => Some(Block::Head),
            BasisKind::Trunk | BasisKind::TrunkRandom => Some(Block::Trunk),
            BasisKind::Rtg | BasisKind::RtgRandom => Some(Block::Rtg),
            BasisKind::Informed | BasisKind::Random => None,
        }
    }
}

/// The layout span of `block`: `(start, len)` in the flat `P`-vector.
///
/// Taken from [`DtShapes::layout`] rather than hard-coded, so a change to the
/// model's parameter order moves this with it.
pub fn block_span(shapes: &crate::policy_model::DtShapes, block: Block) -> (usize, usize) {
    let layout = shapes.layout();
    let (head_start, head_len, _) = *layout.last().expect("the layout always has the head");
    match block {
        Block::Head => (head_start, head_len),
        Block::Trunk => (0, head_start),
        // `e_rtg` is the layout's first entry, so its span comes from the
        // layout too rather than from `rtg_bins * d`.
        Block::Rtg => {
            let (start, len, _) = layout[0];
            (start, len)
        }
    }
}

/// The path with every coordinate outside `(start, len)` zeroed.  The PCA is
/// then over the block's own directions, in the full `P`-dimensional chart (so
/// the basis is still a set of `P`-vectors and `C_half B z` is unchanged).
fn project_path(path: &[Vec<f32>], start: usize, len: usize) -> Vec<Vec<f32>> {
    path.iter()
        .map(|v| {
            let mut w = vec![0.0f32; v.len()];
            w[start..start + len].copy_from_slice(&v[start..start + len]);
            w
        })
        .collect()
}

/// A validated basis plus the identity a report has to print with it.
pub struct BasisLine {
    pub kind: BasisKind,
    pub rank: usize,
    pub basis: Basis,
    pub hash: u64,
    /// `‖θ‖` at the start and end of the path the basis was built from; for
    /// the random control both are the anchor's radius (there is no path).
    pub path_start: f64,
    pub path_end: f64,
}

/// FNV-1a over the basis columns and the identity fields — the same hash
/// discipline `anchor_hash` uses, so a cell can name its basis exactly.
fn basis_hash(kind: BasisKind, rank: usize, basis: &Basis) -> u64 {
    let mut h = Fnv::new();
    h.write(&[kind as u8]);
    h.write(&(rank as u64).to_le_bytes());
    for x in basis.columns() {
        h.write(&x.to_le_bytes());
    }
    h.finish()
}

fn flat(theta: &Tensor) -> Vec<f32> {
    theta
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to vec")
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// One MH-pCN move at a [`DiagScale`]'s frozen per-coordinate `β_i`.  The
/// kernel is exact for any `β_i` (`mh_pcn_z_scaled`), so the preconditioner
/// can only change the mixing, never the target.
fn scaled_step(
    scale: &DiagScale,
    target: &dyn crate::target::ZPotential,
    z: &mut [f32],
    phi: f64,
    rng: &mut impl rand::Rng,
) -> ZStep {
    mh_pcn_z_scaled(target, z, phi, &scale.betas(), rng)
}

fn column_sd(columns: &[Vec<f64>]) -> Vec<f64> {
    columns
        .iter()
        .map(|c| {
            let n = c.len();
            if n < 2 {
                return f64::NAN;
            }
            let mean = c.iter().sum::<f64>() / n as f64;
            (c.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        })
        .collect()
}

/// Short momentum-SGD paths on the adaptation corpus, collected as flat
/// `P`-vectors — R1's "training-trajectory PCA" construction, run at a ladder
/// of learning rates.
///
/// A **single** path is not a usable basis for this model: full-batch momentum
/// steps from one start are `δ_k = (I − lr·H)ᵏ δ₀`, a one-dimensional line, and
/// `pca_basis` correctly refuses ranks the path does not have (measured: the
/// first DT path gave `Degenerate { component: 1 }` at rank 2).  The ladder is
/// the minimal honest fix: different `lr` and different `k` give different
/// polynomials in `H` applied to the same `δ₀`, so their span is the Krylov
/// space `span{δ₀, Hδ₀, H²δ₀, …}` — the directions the *data's* curvature
/// generates, which is what an informed basis is supposed to be.  It remains a
/// declaration of "informed" for a control comparison, not a claim of optimal
/// coverage; the random basis at matched rank is what tests it.
///
/// The gradient is [`crate::normalize::NormGrad::Exact`] because this is a
/// *path*, not an anchor: a basis built from a different function than the one
/// the target reports would be an unmeasured estimator, and R2k measured the
/// stopped-convention gap at ~10³ on a two-layer model.  Every path starts at
/// the anchor, so the chart's origin and the basis describe the same model.
const PATH_RATES: [f64; 6] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4];

fn fit_path(cell: &DtCell, anchor: &Tensor, steps: usize, lr: f64) -> Result<Vec<Vec<f32>>> {
    use crate::normalize::NormGrad;
    let batch = cell.adapt_train.to_dt_batch(&cell.spec)?;
    let pot = DtPotential::new(batch, cell.shapes, cell.variant);
    let mut path = Vec::with_capacity(steps * PATH_RATES.len() + 1);
    path.push(flat(anchor));
    for &rate in &PATH_RATES {
        let scale = lr * rate / 0.05;
        let mut theta = anchor.clone();
        let mut vel = Tensor::zeros(theta.dims(), candle_core::DType::F32, theta.device())?;
        for _ in 0..steps {
            let Some(grad) =
                crate::target::Potential::mean_grad_with(&pot, &theta, NormGrad::Exact)
            else {
                break;
            };
            vel = (vel.affine(0.9, 0.0)? - grad.affine(scale, 0.0)?)?;
            theta = (&theta + &vel)?;
            path.push(flat(&theta));
        }
    }
    Ok(path)
}

/// R1's informed basis: top-`rank` principal directions of the Task-B path from
/// `anchor`.  `path_steps` is a shape of the *basis*, not of any chain.
pub fn informed_line(
    cell: &DtCell,
    anchor: &Tensor,
    rank: usize,
    path_steps: usize,
    lr: f64,
    seed: u64,
) -> Result<BasisLine> {
    informed_line_in(cell, anchor, rank, path_steps, lr, seed, None)
}

/// [`informed_line`] restricted to one parameter block: the same fine-tuning
/// path, projected onto `block` before the PCA, so the principal directions are
/// the block's own.  `None` is the unrestricted basis.
pub fn informed_line_in(
    cell: &DtCell,
    anchor: &Tensor,
    rank: usize,
    path_steps: usize,
    lr: f64,
    seed: u64,
    block: Option<Block>,
) -> Result<BasisLine> {
    let mut path = fit_path(cell, anchor, path_steps, lr)?;
    let kind = match block {
        None => BasisKind::Informed,
        Some(Block::Head) => BasisKind::Head,
        Some(Block::Trunk) => BasisKind::Trunk,
        Some(Block::Rtg) => BasisKind::Rtg,
    };
    let (start, len) = match block {
        None => (0usize, anchor.elem_count()),
        Some(b) => block_span(&cell.shapes, b),
    };
    if block.is_some() {
        path = project_path(&path, start, len);
    }
    let basis = pca_basis(&path, rank, seed ^ rank as u64)
        .map_err(|e| candle_core::Error::Msg(format!("{} basis: {e}", kind.name())))?;
    // A block basis must have no support outside its block; this is the one
    // assertion that keeps a mis-ordered layout from silently producing a
    // full-space basis under a block's name.
    if block.is_some() {
        for (i, &x) in basis.columns().iter().enumerate() {
            let at = i % anchor.elem_count();
            if !(start..start + len).contains(&at) && x != 0.0 {
                return Err(candle_core::Error::Msg(format!(
                    "{} basis leaked outside its block at coordinate {at}",
                    kind.name()
                )));
            }
        }
    }
    let hash = basis_hash(kind, rank, &basis);
    Ok(BasisLine {
        kind,
        rank,
        path_start: norm(path.first().expect("path has its start")),
        path_end: norm(path.last().expect("path has its end")),
        basis,
        hash,
    })
}

/// The matched-rank random control, pinned at the same anchor.
pub fn random_line(anchor: &Tensor, rank: usize, seed: u64) -> Result<BasisLine> {
    let p = anchor.elem_count();
    let basis = Basis::random(p, rank, seed ^ rank as u64)
        .map_err(|e| candle_core::Error::Msg(format!("random basis: {e}")))?;
    let hash = basis_hash(BasisKind::Random, rank, &basis);
    let r = norm(&flat(anchor));
    Ok(BasisLine {
        kind: BasisKind::Random,
        rank,
        basis,
        hash,
        path_start: r,
        path_end: r,
    })
}

/// The matched-rank random control **inside** `block`: drawn in the block's own
/// coordinates and embedded, so it spans the same coordinates the block-restricted
/// informed basis can and the two differ only in which vectors inside the block
/// are used.
pub fn random_line_in(
    anchor: &Tensor,
    shapes: &crate::policy_model::DtShapes,
    rank: usize,
    seed: u64,
    block: Block,
) -> Result<BasisLine> {
    let p = anchor.elem_count();
    let (start, len) = block_span(shapes, block);
    let inner = Basis::random(len, rank, seed ^ rank as u64)
        .map_err(|e| candle_core::Error::Msg(format!("random block basis: {e}")))?;
    let mut cols = vec![0.0f32; p * rank];
    for j in 0..rank {
        let src = inner.col(j);
        cols[j * p + start..j * p + start + len].copy_from_slice(src);
    }
    let basis = Basis::new(cols, p, rank)
        .map_err(|e| candle_core::Error::Msg(format!("random block basis: {e}")))?;
    let kind = match block {
        Block::Head => BasisKind::HeadRandom,
        Block::Trunk => BasisKind::TrunkRandom,
        Block::Rtg => BasisKind::RtgRandom,
    };
    let hash = basis_hash(kind, rank, &basis);
    let r = norm(&flat(anchor));
    Ok(BasisLine {
        kind,
        rank,
        basis,
        hash,
        path_start: r,
        path_end: r,
    })
}

/// The restricted target and the chart it lives on, over one corpus.
pub struct DtRestricted<'a> {
    pub cell: &'a DtCell,
    pub line: BasisLine,
    pub target: SubspaceTarget<DtPotential>,
}

impl<'a> DtRestricted<'a> {
    /// `π_r(dz) ∝ exp[−λ·N·mean_loss(θ_ref + σBz)] N(0, I_r)(dz)`.
    ///
    /// `λ = 1` is the likelihood target.  `sigma` is the prior factor `σ` of
    /// `C_half = σI`, the same prior the ambient chains use.
    pub fn new(cell: &'a DtCell, line: BasisLine, sigma: f64, lambda: f64) -> Result<Self> {
        let anchor = cell.anchor.clone().ok_or_else(|| {
            candle_core::Error::Msg("DtRestricted: pretrain the anchor first".into())
        })?;
        let subspace = AffineSubspace::isotropic(flat(&anchor), line.basis.clone(), sigma)
            .map_err(|e| candle_core::Error::Msg(format!("chart: {e}")))?;
        let batch = cell.adapt_train.to_dt_batch(&cell.spec)?;
        let pot = DtPotential::new(batch, cell.shapes, cell.variant);
        Ok(Self {
            cell,
            line,
            target: SubspaceTarget::with_potential(subspace, lambda, pot),
        })
    }

    pub fn rank(&self) -> usize {
        self.line.rank
    }

    /// `Φ(0)` — the anchor's potential plus the prior's zero.  Printed with
    /// every chain so a report can see how far the draws moved.
    pub fn phi_anchor(&self) -> f64 {
        self.target.phi(&vec![0.0f32; self.rank()])
    }

    pub fn draws_for(&self, z: &[f32]) -> Result<Tensor> {
        self.target.subspace().lift_tensor(z)
    }
}

/// Which proposal the chain uses.  All three are exact MH kernels for the same
/// restricted target; only the mixing can differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proposal {
    /// Scalar `β`, no warm-up adaptation.
    Fixed,
    /// Per-coordinate widths from the second half of a short pilot run, then
    /// shrunk by `--pilot-shrink` (R2e: `s = 0` is isotropic and is the
    /// recommendation unless a mode shape is available).
    Pilot,
    /// Per-coordinate widths `1/√H_ii` at the target's **mode** (R2k's
    /// Laplace arm, the only shape source that beat isotropic at rank 4 in the
    /// classifier cells).
    Laplace,
}

impl Proposal {
    pub fn name(self) -> &'static str {
        match self {
            Proposal::Fixed => "fixed",
            Proposal::Pilot => "pilot",
            Proposal::Laplace => "laplace",
        }
    }
    /// Distinct stream offset per proposal: `fixed` and `pilot` have names of
    /// equal length, and hashing the name length made them share a chain stream
    /// (a correlated comparison neither kernel was told about).
    fn stream_offset(self) -> u64 {
        match self {
            Proposal::Fixed => 0x1111,
            Proposal::Pilot => 0x2222,
            Proposal::Laplace => 0x3333,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fixed" => Some(Proposal::Fixed),
            "pilot" => Some(Proposal::Pilot),
            "laplace" => Some(Proposal::Laplace),
            _ => None,
        }
    }
}

/// One chain's settings, in one value.
#[derive(Clone, Copy, Debug)]
pub struct ChainCfg {
    pub chains: usize,
    pub steps: usize,
    pub burn: usize,
    pub thin: usize,
    pub beta: f64,
    pub pilot_steps: usize,
    pub pilot_shrink: f64,
    /// Laplace widths shrink separately: R2k/R2q's recommendation is
    /// `laplace --pilot-shrink 0.5`, while the isotropic-comparison arm the
    /// pilot should be read against is `s = 0`.  Sharing one dial made the
    /// first screening run report a mode-*scaled isotropic* proposal under the
    /// name `laplace`, which is a different arm.
    pub laplace_shrink: f64,
    pub adapt_target: f64,
    pub mode_steps: usize,
    pub seed: u64,
}

impl Default for ChainCfg {
    fn default() -> Self {
        Self {
            chains: 1,
            steps: 1_000,
            burn: 500,
            thin: 10,
            beta: 0.05,
            pilot_steps: 300,
            pilot_shrink: 0.0,
            laplace_shrink: 0.5,
            adapt_target: 0.3,
            mode_steps: 1_500,
            seed: 7,
        }
    }
}

/// What a chain cost and whether it stayed put.
#[derive(Clone, Debug)]
pub struct ChainReport {
    pub rank: usize,
    pub proposal: Proposal,
    pub chains: usize,
    pub steps: usize,
    pub kept: usize,
    pub accept: f64,
    /// ESS of `Φ` over the kept draws, pooled across chains.
    pub ess_phi: f64,
    pub rhat_phi: f64,
    pub phi_mean: f64,
    /// Largest `‖z‖` reached: outside the chart's prior scale this is a
    /// warning, not a posterior.
    pub z_max: f64,
    /// Frozen `β_i` (empty for `Fixed`).
    pub betas: Vec<f64>,
    /// Full-corpus forward passes the chain spent (one per proposal).
    pub evaluations: u64,
    pub mode_note: Option<String>,
    pub seconds: f64,
}

/// Draws plus the report — one value, so the score readout and the mixing
/// numbers cannot come from different runs.
pub struct SubspaceRuns {
    pub draws: Vec<Tensor>,
    pub report: ChainReport,
}

/// Adam on `∇Φ` from the anchor, for the Laplace widths.
fn find_mode(
    target: &SubspaceTarget<DtPotential>,
    rank: usize,
    cfg: &ChainCfg,
) -> (Vec<f32>, String) {
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let lr = 0.05;
    let mut z = vec![0.0f32; rank];
    let mut m = vec![0.0f64; rank];
    let mut v = vec![0.0f64; rank];
    let phi_start = target.phi(&z);
    let mut grad = target.grad_phi(&z).unwrap_or_else(|| vec![0.0; rank]);
    let first = grad.iter().fold(0.0f64, |a, g| a.max(f64::from(g.abs())));
    let mut used = 0usize;
    let mut converged = false;
    for t in 1..=cfg.mode_steps {
        let gmax = grad.iter().fold(0.0f64, |a, g| a.max(f64::from(g.abs())));
        if gmax <= 1e-3 {
            converged = true;
            break;
        }
        let rate = lr / (1.0 + t as f64 / 200.0).sqrt();
        for i in 0..rank {
            let g = f64::from(grad[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let mh = m[i] / (1.0 - b1.powi(t as i32));
            let vh = v[i] / (1.0 - b2.powi(t as i32));
            z[i] = (f64::from(z[i]) - rate * mh / (vh.sqrt() + eps)) as f32;
        }
        used = t;
        grad = target.grad_phi(&z).unwrap_or_else(|| vec![0.0; rank]);
    }
    let gmax = grad.iter().fold(0.0f64, |a, g| a.max(f64::from(g.abs())));
    let note = format!(
        "mode {used} steps ({}), φ {phi_start:.4} -> {:.4}, max|∇| {first:.3e} -> {gmax:.3e}",
        if converged { "converged" } else { "cap" },
        target.phi(&z),
    );
    (z, note)
}

/// Run the exact MH-pCN chains on the restricted target.
///
/// Each chain owns its RNG stream (the chain index is in the seed), which is
/// what makes `R̂` meaningful.  Adaptation happens only during burn, and the
/// kept window sees a frozen kernel — R2's convention, since an adapted kernel
/// whose draws are quoted as a sample of `π_r` is not one.
pub fn run_chains(
    restricted: &DtRestricted,
    proposal: Proposal,
    cfg: &ChainCfg,
) -> Result<SubspaceRuns> {
    let rank = restricted.rank();
    let target = &restricted.target;
    let started = Instant::now();
    let mut all_draws = Vec::new();
    let mut phi_chains: Vec<Vec<f64>> = Vec::with_capacity(cfg.chains);
    let (mut accepted, mut total) = (0usize, 0usize);
    let mut z_max = 0.0f64;
    let mut betas_out = Vec::new();
    let mut mode_note = None;
    // R2k's mode is a property of the target, not of a chain's stream; find it
    // once so every chain shares one width vector and the report has one note.
    let laplace_widths = if proposal == Proposal::Laplace {
        let (z_mode, note) = find_mode(target, rank, cfg);
        let h = curvature_diag(target, &z_mode, 1e-4);
        let nonpos = h.iter().filter(|v| **v <= 0.0).count();
        mode_note = Some(format!("{note}; non-positive H_ii {nonpos}/{rank}"));
        Some(
            h.iter()
                .map(|v| 1.0 / v.max(1e-8).sqrt())
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    for c in 0..cfg.chains {
        let mut rng = StdRng::seed_from_u64(
            cfg.seed
                ^ ((rank as u64) << 32)
                ^ proposal.stream_offset()
                ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut scale: Option<DiagScale> = match proposal {
            Proposal::Fixed => None,
            Proposal::Pilot => {
                let mut z = vec![0.0f32; rank];
                let mut phi = target.phi(&z);
                let mut columns = vec![Vec::new(); rank];
                for t in 0..cfg.pilot_steps {
                    let st = mh_pcn_z(target, &mut z, phi, cfg.beta, &mut rng);
                    phi = st.phi;
                    if t >= cfg.pilot_steps / 2 {
                        for (i, slot) in columns.iter_mut().enumerate() {
                            slot.push(f64::from(z[i]));
                        }
                    }
                }
                let mut s = DiagScale::from_widths_shrunk(&column_sd(&columns), cfg.pilot_shrink);
                s.target_accept = cfg.adapt_target;
                Some(s)
            }
            Proposal::Laplace => {
                let w = laplace_widths.clone().expect("computed above");
                let mut s = DiagScale::from_widths_shrunk(&w, cfg.laplace_shrink);
                s.target_accept = cfg.adapt_target;
                Some(s)
            }
        };
        let mut z = vec![0.0f32; rank];
        let mut phi = target.phi(&z);
        let mut here = Vec::new();
        for t in 0..cfg.steps {
            let st = match &mut scale {
                None => mh_pcn_z(target, &mut z, phi, cfg.beta, &mut rng),
                Some(s) => {
                    let out = scaled_step(s, target, &mut z, phi, &mut rng);
                    if t < cfg.burn {
                        s.adapt(out.accepted, t);
                    }
                    out
                }
            };
            phi = st.phi;
            if t >= cfg.burn {
                total += 1;
                accepted += usize::from(st.accepted);
                if (t - cfg.burn).is_multiple_of(cfg.thin) {
                    here.push(phi);
                    all_draws.push(restricted.draws_for(&z)?);
                }
            }
            z_max = z_max.max(norm(&z));
        }
        if let Some(s) = &scale {
            betas_out.extend(s.betas());
        }
        phi_chains.push(here);
    }
    let all_phi: Vec<f64> = phi_chains.iter().flatten().copied().collect();
    let phi_mean = all_phi.iter().sum::<f64>() / all_phi.len().max(1) as f64;
    let kept = all_draws.len();
    Ok(SubspaceRuns {
        draws: all_draws,
        report: ChainReport {
            rank,
            proposal,
            chains: cfg.chains,
            steps: cfg.steps,
            kept,
            accept: accepted as f64 / total.max(1) as f64,
            ess_phi: ess_bulk(&phi_chains),
            rhat_phi: if cfg.chains > 1 {
                gelman_rubin(&phi_chains)
            } else {
                f64::NAN
            },
            phi_mean,
            z_max,
            betas: betas_out,
            evaluations: target.evaluations(),
            mode_note,
            seconds: started.elapsed().as_secs_f64(),
        },
    })
}

// ------------------------------------------------------------ baselines

/// Adam on the adaptation corpus' exact mean loss, `steps` full-corpus
/// evaluations — R7.4's *point DT* baseline at a stated cost.
///
/// `Exact` is the right convention here for the same reason it is right for
/// the mode search: the summary statistic is the loss *value*, and a descent
/// direction that is not its gradient is a different optimizer wearing the
/// same name.  Adam rather than SGD because the DT's blocks differ in scale by
/// orders of magnitude.
pub fn adam_point(
    cell: &DtCell,
    init: &Tensor,
    steps: usize,
    lr: f64,
) -> Result<(Tensor, Vec<f64>)> {
    use crate::normalize::NormGrad;
    let batch = cell.adapt_train.to_dt_batch(&cell.spec)?;
    let mut theta = init.clone();
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut m = vec![0.0f64; theta.elem_count()];
    let mut v = vec![0.0f64; theta.elem_count()];
    let mut trace = Vec::with_capacity(steps);
    for t in 1..=steps {
        let (loss, grad) =
            dt_loss_and_grad(&theta, &cell.shapes, cell.variant, &batch, NormGrad::Exact)?;
        trace.push(loss);
        if !loss.is_finite() {
            break;
        }
        let g = grad.flatten_all()?.to_vec1::<f32>()?;
        let flat_theta = theta.flatten_all()?.to_vec1::<f32>()?;
        let mut next = Vec::with_capacity(flat_theta.len());
        for i in 0..flat_theta.len() {
            let gi = f64::from(g[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
            let mh = m[i] / (1.0 - b1.powi(t as i32));
            let vh = v[i] / (1.0 - b2.powi(t as i32));
            next.push((f64::from(flat_theta[i]) - lr * mh / (vh.sqrt() + eps)) as f32);
        }
        theta = Tensor::from_vec(next, theta.dims(), theta.device())?;
    }
    Ok((theta, trace))
}

/// The point baseline R7.4 asks for: Adam from the anchor, with the learning
/// rate and the stopping step selected on **development** data.
///
/// The untuned arm is not a baseline, it is a straw man: measured at lr 0.01
/// and 400 steps on seed 7, the point model descends its training NLL to 0.59
/// and then scores 2.59 on test — worse than the anchor it started from — so a
/// comparison against it would credit `p̄` with fixing the baseline's
/// overfitting.  Both dials are fitted on `adapt_dev` (never test), every step
/// that is scored on dev is counted in `evaluations`, and the selected
/// `(lr, step)` is printed with the cell so the baseline's budget is auditable.
/// The temperature is fitted afterwards on the same dev split by
/// [`tune_temperature`]; the plan's baseline is this point model with both
/// scalars tuned where tuning is allowed.
pub struct PointFit {
    pub theta: Tensor,
    pub lr: f64,
    pub step: usize,
    pub dev_nll: f64,
    pub evaluations: usize,
}

pub fn adam_point_dev(
    cell: &DtCell,
    anchor: &Tensor,
    steps: usize,
    lrs: &[f64],
    dev_every: usize,
) -> Result<PointFit> {
    use crate::normalize::NormGrad;
    let batch = cell.adapt_train.to_dt_batch(&cell.spec)?;
    let dev = cell.adapt_dev.to_dt_batch(&cell.spec)?;
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let n = anchor.elem_count();
    let mut best: Option<(f64, f64, usize, Tensor)> = None;
    let mut evaluations = 0usize;
    let dev_nll = |theta: &Tensor| -> Result<f64> {
        crate::policy_model::dt_loss(theta, &cell.shapes, cell.variant, &dev)
    };
    for &lr in lrs {
        let mut theta = anchor.clone();
        let mut m = vec![0.0f64; n];
        let mut v = vec![0.0f64; n];
        let start = dev_nll(&theta)?;
        evaluations += 1;
        best = Some((start, lr, 0, theta.clone()));
        for t in 1..=steps {
            let (loss, grad) =
                dt_loss_and_grad(&theta, &cell.shapes, cell.variant, &batch, NormGrad::Exact)?;
            evaluations += 1;
            if !loss.is_finite() {
                break;
            }
            let g = grad.flatten_all()?.to_vec1::<f32>()?;
            let flat_theta = theta.flatten_all()?.to_vec1::<f32>()?;
            let mut next = Vec::with_capacity(n);
            for i in 0..n {
                let gi = f64::from(g[i]);
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let mh = m[i] / (1.0 - b1.powi(t as i32));
                let vh = v[i] / (1.0 - b2.powi(t as i32));
                next.push((f64::from(flat_theta[i]) - lr * mh / (vh.sqrt() + eps)) as f32);
            }
            theta = Tensor::from_vec(next, theta.dims(), theta.device())?;
            if t.is_multiple_of(dev_every) || t == steps {
                let d = dev_nll(&theta)?;
                evaluations += 1;
                if best.as_ref().is_none_or(|b| d < b.0) {
                    best = Some((d, lr, t, theta.clone()));
                }
            }
        }
    }
    let (dev_nll, lr, step, theta) = best.expect("at least one lr");
    Ok(PointFit {
        theta,
        lr,
        step,
        dev_nll,
        evaluations,
    })
}

/// `members` independent point DTs, each pre-trained and adapted with
/// `steps_each` evaluations — the matched-cost ensemble.  Diversity comes from
/// independent initialisations, not from one anchor: with a deterministic
/// full-batch objective and one start, a "4-member ensemble" would be four
/// copies of one model.
pub fn point_ensemble(
    cell: &DtCell,
    members: usize,
    steps_each: usize,
    lr: f64,
    seed: u64,
) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(members);
    for k in 0..members {
        let init = init_flat(
            &cell.shapes,
            0.1,
            seed ^ (k as u64 + 1).wrapping_mul(0x9e37_79b9),
        );
        let pretrain = steps_each / 2;
        let anchor_batch = cell.anchor_train.to_dt_batch(&cell.spec)?;
        let pot = DtPotential::new(anchor_batch, cell.shapes, cell.variant);
        let dev_batch = cell.anchor_dev.to_dt_batch(&cell.spec)?;
        let pot_dev = DtPotential::new(dev_batch, cell.shapes, cell.variant);
        let (theta, _) = crate::experiment::PotentialAnchor::new(pot, pot_dev, init).pretrain(
            pretrain,
            0.02,
            0.9,
            crate::normalize::NormGrad::Stopped,
        );
        let (theta, _) = adam_point(cell, &theta, steps_each - pretrain, lr)?;
        out.push(theta);
    }
    Ok(out)
}

/// Per-step NLL of one point model on `corpus`, at a head temperature.
pub fn point_scores(
    cell: &DtCell,
    theta: &Tensor,
    corpus: &Corpus,
    temperature: f64,
    bins: usize,
) -> Result<(StepScores, Calibration, f64)> {
    let b = corpus.to_dt_batch(&cell.spec)?;
    let dists = dt_step_distributions_at(theta, &cell.shapes, cell.variant, &b, temperature)?;
    let (_, act) = b.scored_steps()?;
    let targets: Vec<usize> = act.into_iter().map(|a| a as usize).collect();
    let scores = score_steps(&dists, &targets);
    let nll = scores.mean(Rule::Log);
    let cal = max_prob_calibration(&dists, &targets, bins);
    Ok((scores, cal, nll))
}

/// Per-step scores of the mean of several models' step distributions — the
/// posterior predictive `p̄` when `thetas` are chain draws, and the ensemble
/// predictive when they are ensemble members.
pub fn mean_scores(
    cell: &DtCell,
    thetas: &[Tensor],
    corpus: &Corpus,
    bins: usize,
) -> Result<(StepScores, Calibration, f64)> {
    assert!(!thetas.is_empty(), "mean_scores: at least one model");
    let b = corpus.to_dt_batch(&cell.spec)?;
    let mut acc: Vec<Vec<f64>> = Vec::new();
    for (i, th) in thetas.iter().enumerate() {
        let d = dt_step_distributions_at(th, &cell.shapes, cell.variant, &b, 1.0)?;
        if i == 0 {
            acc = d;
        } else {
            for (a, row) in acc.iter_mut().zip(d.iter()) {
                for (x, y) in a.iter_mut().zip(row.iter()) {
                    *x += y;
                }
            }
        }
    }
    let n = thetas.len() as f64;
    for row in acc.iter_mut() {
        for x in row.iter_mut() {
            *x /= n;
        }
    }
    let (_, act) = b.scored_steps()?;
    let targets: Vec<usize> = act.into_iter().map(|a| a as usize).collect();
    let scores = score_steps(&acc, &targets);
    let nll = scores.mean(Rule::Log);
    let cal = max_prob_calibration(&acc, &targets, bins);
    Ok((scores, cal, nll))
}

/// Temperature minimising the **development** per-step NLL of `theta` over
/// `grid`.  Test is never touched: the returned `T` is the baseline's only
/// tuned quantity and the tuning data is reported with it.
pub fn tune_temperature(
    cell: &DtCell,
    theta: &Tensor,
    grid: &[f64],
) -> Result<(f64, Vec<(f64, f64)>)> {
    let mut best = (f64::NAN, f64::INFINITY);
    let mut rows = Vec::with_capacity(grid.len());
    for &t in grid {
        let (_, _, nll) = point_scores(cell, theta, &cell.adapt_dev, t, 5)?;
        rows.push((t, nll));
        if nll < best.1 {
            best = (t, nll);
        }
    }
    Ok((best.0, rows))
}

// --------------------------------------------------------------- returns

/// A DT as an `ActionDist` at a head temperature (`T = 1` is the untempered
/// policy).  Same conditioning contract as `dt_arm::dt_policy`.
pub fn dt_policy_at(
    env: &MdpEnv,
    cell: &DtCell,
    theta: Tensor,
    target: f32,
    temperature: f64,
) -> impl Fn(&Context) -> Vec<f64> + use<> {
    let env = env.clone();
    let spec = cell.spec;
    let shapes = cell.shapes;
    let variant = cell.variant;
    move |ctx: &Context| {
        let acts: Vec<u32> = ctx.actions.iter().map(|&a| a as u32).collect();
        let bins = conditioning_bins(&env, &spec, target, ctx.states, ctx.actions);
        dt_act_at(
            &theta,
            &shapes,
            variant,
            &bins,
            ctx.states,
            &acts,
            temperature,
        )
        .expect("dt_act_at")
    }
}

/// The averaged decoder of several DTs (the ensemble or `p̄`).
pub fn mean_policy_at(
    env: &MdpEnv,
    cell: &DtCell,
    thetas: &[Tensor],
    target: f32,
    temperature: f64,
) -> impl Fn(&Context) -> Vec<f64> + use<> {
    let thetas = thetas.to_vec();
    assert!(
        !thetas.is_empty(),
        "mean_policy_at needs at least one model"
    );
    let env = env.clone();
    let spec = cell.spec;
    let shapes = cell.shapes;
    let variant = cell.variant;
    move |ctx: &Context| {
        let acts: Vec<u32> = ctx.actions.iter().map(|&a| a as u32).collect();
        let bins = conditioning_bins(&env, &spec, target, ctx.states, ctx.actions);
        let k = dt_act_at(
            &thetas[0],
            &shapes,
            variant,
            &bins,
            ctx.states,
            &acts,
            temperature,
        )
        .expect("dt_act_at")
        .len();
        let mut mean = vec![0.0f64; k];
        for th in &thetas {
            let d = dt_act_at(th, &shapes, variant, &bins, ctx.states, &acts, temperature)
                .expect("dt_act_at");
            for (m, x) in mean.iter_mut().zip(d.iter()) {
                *m += x;
            }
        }
        let n = thetas.len() as f64;
        for m in mean.iter_mut() {
            *m /= n;
        }
        mean
    }
}

/// Arm names in the order [`return_readout`] reports them.
pub const RETURN_ARMS: [&str; 9] = [
    "behaviour",
    "anchor-sample",
    "anchor-argmax",
    "point-sample",
    "point-argmax",
    "pbar-sample",
    "pbar-argmax",
    "ensemble-argmax",
    "optimal",
];

/// Paired return readout: behaviour policy, anchor, the point baseline (sample
/// and argmax), `p̄` over the chain draws (sample and argmax), the ensemble's
/// averaged decoder and the exact optimum.  One uniform stream and one noise
/// sequence per replicate, so the differences are paired.
pub struct ReturnArms {
    pub behaviour: Corpus,
    pub anchor: Tensor,
    pub point: Tensor,
    pub draws: Vec<Tensor>,
    pub ensemble: Vec<Tensor>,
    pub temperature: f64,
}

/// A declared subsample of draws for the **rollout** readout.
///
/// A rollout queries the averaged decoder at every step of every replicate, so
/// its cost is `rollouts × horizon × draws` forwards.  At the promoted scale
/// (2 800 draws) that is hours, and the first screening run's unpromoted draws
/// hid it.  The per-step score columns still use every draw; only the return
/// arms use this subset, evenly spaced so it covers all chains, and the count
/// is printed with the table.  This is an estimator of the same `p̄`, not a
/// different posterior.
pub fn subsample_draws(draws: &[Tensor], max: usize) -> Vec<Tensor> {
    assert!(max > 0, "subsample_draws: max > 0");
    if draws.len() <= max {
        return draws.to_vec();
    }
    let stride = draws.len().div_ceil(max);
    draws.iter().step_by(stride).take(max).cloned().collect()
}

/// One declared draw-count's rollout audit (D5's policy-approximation check).
#[derive(Clone, Debug)]
pub struct DrawSubsetAudit {
    pub requested: usize,
    pub used: usize,
    /// Per-step BMA, sampled: the current `pbar-sample` arm.
    pub bma_sample: ArmReturns,
    /// Per-step BMA, argmax: the deterministic averaged decoder.
    pub bma_argmax: ArmReturns,
    /// One posterior draw held for a whole episode, then sampled.
    pub episode_draw: ArmReturns,
    pub argmax_vs_sample: PairedDiff,
    pub episode_vs_bma: PairedDiff,
}

/// Nested draw-count audit at fixed histories and a common environment seed.
#[derive(Clone, Debug)]
#[allow(clippy::type_complexity)]
pub struct DrawAudit {
    /// Ascending draw counts, each re-scored on the *same* rollouts and seed.
    pub subsets: Vec<DrawSubsetAudit>,
    /// The largest declared draw count, against which the others are paired.
    pub baseline_used: usize,
    /// `(requested, paired difference vs baseline)` for the BMA sample arm.
    pub subset_vs_full_bma: Vec<(usize, PairedDiff)>,
    /// The same for the once-per-episode draw arm.
    pub subset_vs_full_episode: Vec<(usize, PairedDiff)>,
}

/// Audit the finite rollout draw subset against successively larger averages.
///
/// `DEVELOPMENT_PLAN.md` D5: "Audit the finite draw subset used in rollout
/// decoding against a full/successively larger posterior average at fixed
/// histories and common environment seeds. Name per-step BMA sampling, argmax
/// decoding and a once-per-episode posterior draw separately." This runs exactly
/// those three decoders at each declared count. Every subset uses the same
/// rollout seed, so the environment's uniform and noise streams are shared; the
/// episode-draw arm uses the draw `episode % used`, a declared cyclic schedule
/// rather than a hidden extra random source. A difference here estimates the
/// *decoder* error at this fixed draw set. It is not a TV/KL statement about the
/// continuous posterior, and it does not test chain convergence.
pub fn audit_posterior_draws(
    env: &MdpEnv,
    cell: &DtCell,
    anchor: Tensor,
    draws: &[Tensor],
    subsets: &[usize],
    rollouts: usize,
    seed: u64,
) -> DrawAudit {
    assert!(rollouts > 1, "an MCSE needs more than one rollout");
    assert!(!subsets.is_empty(), "declare at least one draw count");
    assert!(
        subsets.windows(2).all(|w| w[0] < w[1]),
        "subsets must be ascending so the last is the full declared average"
    );
    let target = cell.rtg_conditioning();
    let mut audited = Vec::with_capacity(subsets.len());
    for &max in subsets {
        let used = subsample_draws(draws, max);
        let arms = ReturnArms {
            behaviour: cell.adapt_train.clone(),
            anchor: anchor.clone(),
            point: anchor.clone(),
            draws: used.clone(),
            ensemble: vec![anchor.clone()],
            temperature: 1.0,
        };
        let ret = return_readout(env, cell, &arms, rollouts, seed);
        let bma_sample = ret.arm("pbar-sample").clone();
        let bma_argmax = ret.arm("pbar-argmax").clone();
        let episode_returns = episode_draw_returns(env, cell, &used, target, rollouts, seed);
        let (episode_mean, episode_mcse) = mean_mcse(&episode_returns);
        let episode_draw = ArmReturns {
            name: "episode-draw".to_string(),
            returns: episode_returns,
            mean: episode_mean,
            mcse: episode_mcse,
        };
        audited.push(DrawSubsetAudit {
            requested: max,
            used: used.len(),
            argmax_vs_sample: paired(&bma_argmax.returns, &bma_sample.returns),
            episode_vs_bma: paired(&episode_draw.returns, &bma_sample.returns),
            bma_sample,
            bma_argmax,
            episode_draw,
        });
    }
    let baseline = audited.last().expect("audited is non-empty");
    let baseline_used = baseline.used;
    let mut subset_vs_full_bma = Vec::new();
    let mut subset_vs_full_episode = Vec::new();
    for row in &audited {
        subset_vs_full_bma.push((
            row.requested,
            paired(&row.bma_sample.returns, &baseline.bma_sample.returns),
        ));
        subset_vs_full_episode.push((
            row.requested,
            paired(&row.episode_draw.returns, &baseline.episode_draw.returns),
        ));
    }
    DrawAudit {
        subsets: audited,
        baseline_used,
        subset_vs_full_bma,
        subset_vs_full_episode,
    }
}

/// One posterior draw held for an entire episode, then sampled per step.
///
/// The draw is selected by `episode % draws.len()` and the environment consumes
/// exactly the uniforms/noise sequence [`readout`] uses, so the arm is paired
/// with the BMA arms at the same seed without a second random source.
fn episode_draw_returns(
    env: &MdpEnv,
    cell: &DtCell,
    draws: &[Tensor],
    target: f32,
    rollouts: usize,
    seed: u64,
) -> Vec<f64> {
    use rand::Rng;
    use rand_distr::StandardNormal;
    assert!(!draws.is_empty(), "episode_draw_returns needs a draw");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(rollouts);
    for episode in 0..rollouts {
        let uniforms: Vec<f64> = (0..env.horizon).map(|_| rng.random::<f64>()).collect();
        let noises: Vec<[f32; 2]> = (0..env.horizon)
            .map(|_| {
                let n0: f32 = rng.sample(StandardNormal);
                let n1: f32 = rng.sample(StandardNormal);
                [env.noise * n0, env.noise * n1]
            })
            .collect();
        let policy = dt_policy_at(env, cell, draws[episode % draws.len()].clone(), target, 1.0);
        let mut x = START;
        let mut states = vec![x];
        let mut acts: Vec<usize> = Vec::with_capacity(env.horizon);
        let mut total = 0.0f64;
        for t in 0..env.horizon {
            let dist = policy(&Context {
                t,
                state: x,
                states: &states,
                actions: &acts,
            });
            let a = sample_at(&dist, uniforms[t]);
            total += f64::from(env.reward(x, a));
            acts.push(a);
            x = env.step(x, a, noises[t]);
            states.push(x);
        }
        out.push(total);
    }
    out
}

pub fn return_readout(
    env: &MdpEnv,
    cell: &DtCell,
    arms: &ReturnArms,
    rollouts: usize,
    seed: u64,
) -> ReturnReadout {
    let target = cell.rtg_conditioning();
    let solved = env.solve();
    let behaviour = behaviour_policy(env, arms.behaviour.policy);
    let anchor_p = dt_policy_at(env, cell, arms.anchor.clone(), target, 1.0);
    let point_p = dt_policy_at(env, cell, arms.point.clone(), target, arms.temperature);
    let mean_p = mean_policy_at(env, cell, &arms.draws, target, 1.0);
    let ens_p = mean_policy_at(env, cell, &arms.ensemble, target, 1.0);
    let k = env.actions.len();
    let optimal = move |ctx: &Context| {
        let mut d = vec![0.0f64; k];
        d[solved.greedy_action(ctx.t, ctx.state)] = 1.0;
        d
    };
    let all = [
        Arm {
            name: RETURN_ARMS[0],
            dist: &behaviour,
            decode: Decode::Sample,
        },
        Arm {
            name: RETURN_ARMS[1],
            dist: &anchor_p,
            decode: Decode::Sample,
        },
        Arm {
            name: RETURN_ARMS[2],
            dist: &anchor_p,
            decode: Decode::Argmax,
        },
        Arm {
            name: RETURN_ARMS[3],
            dist: &point_p,
            decode: Decode::Sample,
        },
        Arm {
            name: RETURN_ARMS[4],
            dist: &point_p,
            decode: Decode::Argmax,
        },
        Arm {
            name: RETURN_ARMS[5],
            dist: &mean_p,
            decode: Decode::Sample,
        },
        Arm {
            name: RETURN_ARMS[6],
            dist: &mean_p,
            decode: Decode::Argmax,
        },
        Arm {
            name: RETURN_ARMS[7],
            dist: &ens_p,
            decode: Decode::Argmax,
        },
        Arm {
            name: RETURN_ARMS[8],
            dist: &optimal,
            decode: Decode::Argmax,
        },
    ];
    readout(env, START, &all, rollouts, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_model::dt_loss;
    use crate::target::Potential;

    fn pretrained(seed: u64) -> DtCell {
        let mut cell = DtCell::new(seed);
        cell.pretrain(12, 0.02, 0.9).unwrap();
        cell
    }

    #[test]
    fn the_chart_origin_is_the_anchor_and_the_target_is_its_potential() {
        let cell = pretrained(4);
        let anchor = cell.anchor.clone().unwrap();
        let line = random_line(&anchor, 2, 9).unwrap();
        assert_eq!(line.path_start, line.path_end);
        let restricted = DtRestricted::new(&cell, line, 0.35, 1.0).unwrap();
        let z0 = vec![0.0f32; 2];
        let lifted = restricted.draws_for(&z0).unwrap();
        let a = flat(&anchor);
        let b = flat(&lifted);
        assert_eq!(a, b, "z = 0 must be the anchor bitwise");
        // Φ(0) = λ·N·mean_loss(anchor) + 0.
        let want = restricted.target.n_terms() as f64
            * Potential::mean_loss(
                &DtPotential::new(
                    cell.adapt_train.to_dt_batch(&cell.spec).unwrap(),
                    cell.shapes,
                    cell.variant,
                ),
                &anchor,
            );
        assert!(
            (restricted.phi_anchor() - want).abs() < 1e-3,
            "{} vs {want}",
            restricted.phi_anchor()
        );
    }

    #[test]
    fn informed_basis_is_orthonormal_and_rank_limited() {
        let cell = pretrained(6);
        let anchor = cell.anchor.clone().unwrap();
        // The path only has to span more than `rank` directions; 3 steps across
        // the 6 rates gives 19 points, while a 40-step path makes this unoptimized
        // test spend minutes in exact-gradient evaluations for no extra check.
        let line = informed_line(&cell, &anchor, 3, 3, 0.05, 11).unwrap();
        assert_eq!(line.rank, 3);
        assert!(line.basis.gram_error() < crate::subspace::ORTHONORMAL_TOL);
        assert_eq!(line.basis.p(), cell.shapes.flat_size());
        assert!(line.path_end.is_finite() && line.path_start.is_finite());
        // Same seed, same basis; a different seed moves it.
        let again = informed_line(&cell, &anchor, 3, 3, 0.05, 11).unwrap();
        assert_eq!(line.hash, again.hash);
    }

    #[test]
    fn block_bases_live_inside_their_block_and_partition_the_parameters() {
        let cell = pretrained(6);
        let anchor = cell.anchor.clone().unwrap();
        let p = cell.shapes.flat_size();
        let (hs, hl) = block_span(&cell.shapes, Block::Head);
        let (ts, tl) = block_span(&cell.shapes, Block::Trunk);
        assert_eq!(ts, 0, "the trunk starts at 0");
        assert_eq!(tl, hs, "the blocks partition the layout");
        assert_eq!(hs + hl, p, "head + trunk is the whole parameter vector");
        assert_eq!(hl, cell.shapes.d * cell.shapes.action_vocab);
        // The RTG table is a sub-block of the trunk, not a third partition: it
        // must sit inside it and keep the table's own size.
        let (rs, rl) = block_span(&cell.shapes, Block::Rtg);
        assert_eq!(rl, cell.shapes.rtg_bins * cell.shapes.d);
        assert!(rs >= ts && rs + rl <= ts + tl, "rtg sits inside the trunk");

        for (block, keep) in [
            (Block::Head, (hs, hl)),
            (Block::Trunk, (ts, tl)),
            (Block::Rtg, (rs, rl)),
        ] {
            // Same reason as `informed_basis_is_orthonormal_and_rank_limited`:
            // 3 steps preserve more than rank 3 without a multi-minute sweep.
            // The RTG table is the exception: its path directions are collinear
            // enough that 3 steps still leaves direction 2 with no variance.
            let steps = if block == Block::Rtg { 8 } else { 3 };
            let line = informed_line_in(&cell, &anchor, 3, steps, 0.05, 11, Some(block)).unwrap();
            assert!(line.basis.gram_error() < crate::subspace::ORTHONORMAL_TOL);
            assert_eq!(line.basis.p(), p);
            for (i, &x) in line.basis.columns().iter().enumerate() {
                let at = i % p;
                if !(keep.0..keep.0 + keep.1).contains(&at) {
                    assert_eq!(x, 0.0, "{} basis has support at {at}", line.kind.name());
                }
            }
            let ctrl = random_line_in(&anchor, &cell.shapes, 3, 11, block).unwrap();
            assert!(ctrl.basis.gram_error() < crate::subspace::ORTHONORMAL_TOL);
            for (i, &x) in ctrl.basis.columns().iter().enumerate() {
                let at = i % p;
                if !(keep.0..keep.0 + keep.1).contains(&at) {
                    assert_eq!(x, 0.0, "{} control has support at {at}", ctrl.kind.name());
                }
            }
            // The block bases are named apart from the unrestricted ones, so a
            // restricted cell cannot be mistaken for a full-space result.
            assert_ne!(
                line.hash,
                informed_line(&cell, &anchor, 3, 3, 0.05, 11).unwrap().hash
            );
        }
        assert!(BasisKind::Head.is_informed() && !BasisKind::HeadRandom.is_informed());
    }

    #[test]
    fn a_fixed_chain_keeps_a_finite_potential_and_reports_its_cost() {
        let cell = pretrained(8);
        let anchor = cell.anchor.clone().unwrap();
        let line = random_line(&anchor, 2, 3).unwrap();
        let restricted = DtRestricted::new(&cell, line, 0.35, 1.0).unwrap();
        let cfg = ChainCfg {
            chains: 2,
            steps: 30,
            burn: 15,
            thin: 3,
            seed: 2,
            pilot_steps: 8,
            ..ChainCfg::default()
        };
        let runs = run_chains(&restricted, Proposal::Fixed, &cfg).unwrap();
        assert!(!runs.draws.is_empty());
        assert!(runs.report.phi_mean.is_finite());
        assert!(runs.report.z_max.is_finite());
        assert_eq!(runs.report.kept, runs.draws.len());
        // One potential evaluation per proposal, no more.
        assert_eq!(
            runs.report.evaluations,
            (cfg.steps * cfg.chains) as u64 + cfg.chains as u64,
            "one evaluation per proposal plus the starting potential"
        );
        // The draws are the lifted chart points, so their loss is the DT's.
        let b = cell.adapt_dev.to_dt_batch(&cell.spec).unwrap();
        for d in runs.draws.iter().take(2) {
            assert!(
                dt_loss(d, &cell.shapes, cell.variant, &b)
                    .unwrap()
                    .is_finite()
            );
        }
    }

    #[test]
    fn pilot_and_laplace_produce_valid_per_coordinate_scales() {
        let cell = pretrained(10);
        let anchor = cell.anchor.clone().unwrap();
        let line = random_line(&anchor, 2, 4).unwrap();
        let restricted = DtRestricted::new(&cell, line, 0.35, 1.0).unwrap();
        let cfg = ChainCfg {
            chains: 1,
            steps: 20,
            burn: 10,
            thin: 2,
            pilot_steps: 6,
            pilot_shrink: 0.5,
            mode_steps: 3,
            ..ChainCfg::default()
        };
        for p in [Proposal::Pilot, Proposal::Laplace] {
            let runs = run_chains(&restricted, p, &cfg).unwrap();
            assert_eq!(runs.report.betas.len(), 2, "{p:?} has one β per rank");
            assert!(
                runs.report.betas.iter().all(|b| *b > 0.0 && *b <= 1.0),
                "{p:?}: {:?}",
                runs.report.betas
            );
        }
    }

    #[test]
    fn dev_tuned_point_selects_a_finite_step_and_counts_its_evals() {
        let cell = pretrained(14);
        let anchor = cell.anchor.clone().unwrap();
        let fit = adam_point_dev(&cell, &anchor, 6, &[0.003, 0.01], 3).unwrap();
        assert!(fit.dev_nll.is_finite() && fit.step <= 6 && fit.step > 0);
        assert!([0.003, 0.01].contains(&fit.lr));
        // 1 dev check at the anchor per lr, one train step per step, plus the
        // scheduled dev checks (steps 3 and 6).
        let want = 2 * (1 + 6 + 2);
        assert_eq!(fit.evaluations, want);
        let (t, rows) = tune_temperature(&cell, &fit.theta, &[1.0, 2.0]).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(t > 0.0);
    }

    #[test]
    fn point_baseline_descends_and_temperature_is_tuned_on_dev() {
        let cell = pretrained(12);
        let init = init_flat(&cell.shapes, 0.1, 21);
        let (theta, trace) = adam_point(&cell, &init, 8, 0.01).unwrap();
        assert!(trace.len() == 8 && trace.iter().all(|l| l.is_finite()));
        assert!(
            trace.last().unwrap() <= &trace[0],
            "Adam did not descend: {trace:?}"
        );
        let (t, rows) = tune_temperature(&cell, &theta, &[0.5, 1.0, 2.0]).unwrap();
        assert_eq!(rows.len(), 3);
        assert!([0.5, 1.0, 2.0].contains(&t));
        // T = 1 is bitwise the untempered readout.
        let a = point_scores(&cell, &theta, &cell.adapt_dev, 1.0, 4)
            .unwrap()
            .0;
        let b = crate::policy_readout::dt_step_scores(
            &theta,
            &cell.shapes,
            cell.variant,
            &cell.adapt_dev.to_dt_batch(&cell.spec).unwrap(),
        )
        .unwrap();
        assert!((a.mean(Rule::Log) - b.mean(Rule::Log)).abs() < 1e-12);

        // The paired difference used for the R7.5 verdict: a model against
        // itself is exactly zero, and the SE is over the scored steps.
        let (m, se, n) = crate::policy_readout::paired_delta(&a, &a, Rule::Log);
        assert_eq!(m, 0.0);
        assert_eq!(se, 0.0);
        assert_eq!(n, cell.adapt_dev.transitions());
        let c = point_scores(&cell, &theta, &cell.adapt_dev, 2.0, 4)
            .unwrap()
            .0;
        let (m, se, _) = crate::policy_readout::paired_delta(&c, &a, Rule::Log);
        assert!((m - (c.mean(Rule::Log) - a.mean(Rule::Log))).abs() < 1e-12);
        assert!(se > 0.0, "tempering changes steps by different amounts");
    }
}
