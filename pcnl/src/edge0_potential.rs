//! E5 — the frozen-encoder Edge0 head chart as a pCNL [`Potential`].
//!
//! `EDGE0_RLCD_PLAN.md` §6 and `DEVELOPMENT_PLAN.md` E5 say: first freeze the
//! state encoder, deterministic routing and latent policy, and expose a small
//! head/adapter chart through the existing `Potential` / `SubspaceTarget<P>`
//! seam. A **categorical likelihood** then has
//! `Ψ = λ Σ_data −log p(y|s,q)` and total `J = Ψ + ‖z‖²/2` in standardised
//! prior coordinates.
//!
//! Three constraints this module enforces rather than documents:
//!
//! * The likelihood is the **exact latent marginal** `p = Σ_r a_phi(r) u_theta(r)`
//!   — with four paths this is cheap, so no sampled or stochastic likelihood is
//!   ever evaluated inside a Metropolis ratio (§6: "do not use freshly sampled
//!   paths inside an otherwise exact MH acceptance ratio").
//! * Parameters outside the head/adapter chart are **frozen by zeroing their
//!   gradient**, not by pretending they are absent. Reusing one flat `θ` keeps
//!   the chart's identity honest.
//! * The likelihood and the prior stay separate: [`Potential`] reports only the
//!   likelihood term and its number of terms; `SubspaceTarget` adds the
//!   Gaussian prior and the temperature. A Brier-based target is *generalized
//!   Bayes* and must be labelled as such — this adapter is the NLL target.
//!
//! `SubspaceTarget<Edge0Potential>` therefore samples the head chart with the
//! same exact pCN kernel and the same diagnostics as every other pCNL target.

use crate::certificate::{
    Certificate, ClaimRecord, KernelIdentity, TargetIdentity, VerifiedRegion,
};
use crate::edge0_campaign::Sample;
use crate::evidence_enclosure::Iv;
use crate::edge0_model::{Edge0Config, ModelInput, forward};
use crate::manifest::Fnv;
use crate::normalize::NormGrad;
use crate::target::{Potential, SubspaceTarget, ZPotential, ZStep};
use candle_core::{Result, Tensor, Var};
use rand::Rng;

/// The likelihood floor used by **both** the value and the gradient.
///
/// They used to differ (`1e-12` in `losses`, `1e-6` in `mean_grad_with`), which
/// makes the "exact gradient of the mean NLL" claim false exactly where a head
/// saturates. The value is also what the evidence ceiling in
/// [`HeadChartReference::clamp_certificate`] is derived from, so the three uses
/// have to name one number.
pub const NLL_FLOOR: f64 = 1e-12;

/// The frozen-encoder head/adapter chart's likelihood.
pub struct Edge0Potential {
    config: Edge0Config,
    samples: Vec<Sample>,
    /// Indices of `θ` allowed to move. Every other entry's gradient is zeroed.
    trainable: Vec<bool>,
}

impl Edge0Potential {
    /// `trainable[i] == true` for the small head/adapter chart; encoder,
    /// router, expert and policy entries are frozen.
    ///
    /// A caller that wants the full model passes all-`true` explicitly. The
    /// plan's E5 protocol is the head-only chart, so this constructor takes the
    /// mask rather than guessing one from tensor names.
    pub fn new(config: Edge0Config, samples: Vec<Sample>, trainable: Vec<bool>) -> Self {
        assert_eq!(
            trainable.len(),
            config.flat_size(),
            "the freeze mask must cover every parameter"
        );
        assert!(samples.iter().all(|s| s.question.validate(&config).is_ok()));
        Self {
            config,
            samples,
            trainable,
        }
    }

    /// A head-only chart: the last four slots (`noul_*`, `score_*` and the
    /// policy head) move; everything upstream is frozen.
    pub fn head_only(config: Edge0Config, samples: Vec<Sample>) -> Self {
        let layout = config.layout();
        let p = config.flat_size();
        let mut trainable = vec![false; p];
        for slot in &layout {
            let head = matches!(
                slot.name,
                "policy_w" | "policy_b" | "noul_w" | "noul_b" | "score_w" | "score_thresholds"
            );
            if head {
                for i in slot.start..slot.start + slot.len {
                    trainable[i] = true;
                }
            }
        }
        Self::new(config, samples, trainable)
    }

    pub fn config(&self) -> &Edge0Config {
        &self.config
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Per-example exact-marginal NLL. `None` if any marginal is non-finite.
    pub fn losses(&self, theta: &Tensor) -> Result<Option<Vec<f64>>> {
        let mut out = Vec::with_capacity(self.samples.len());
        for s in &self.samples {
            let input = ModelInput {
                state: &s.input_state,
                questions: std::slice::from_ref(&s.question),
            };
            let model_out = forward(theta, &self.config, &input)?;
            let p = model_out.per_question[0].marginal_vec()?;
            if !p.iter().all(|x| x.is_finite()) {
                return Ok(None);
            }
            out.push(-p[s.label].max(NLL_FLOOR).ln());
        }
        Ok(Some(out))
    }

    /// `p(label of sample \`index\` | θ)` — the exact latent marginal used by
    /// the likelihood, exposed for the E5 predictive comparison.
    pub fn label_probability(&self, theta: &Tensor, index: usize) -> Result<Option<f64>> {
        let s = &self.samples[index];
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let model_out = forward(theta, &self.config, &input)?;
        let p = model_out.per_question[0].marginal_vec()?;
        Ok(p.get(s.label).copied().filter(|x| x.is_finite()))
    }
}

impl Potential for Edge0Potential {
    fn n_terms(&self) -> usize {
        self.samples.len()
    }

    fn mean_loss(&self, theta: &Tensor) -> f64 {
        match self.losses(theta) {
            Ok(Some(v)) if !v.is_empty() => v.iter().sum::<f64>() / v.len() as f64,
            _ => f64::INFINITY,
        }
    }

    /// The gradient of the exact mean NLL, with frozen entries zeroed.
    ///
    /// `NormGrad` is accepted for the trait's sake and has no effect: this
    /// model has no Frobenius normalisation, so there is no stopped/exact
    /// ambiguity to resolve.
    fn mean_grad_with(&self, theta: &Tensor, _norm: NormGrad) -> Option<Tensor> {
        let var = Var::from_tensor(theta).ok()?;
        let mut losses = Vec::with_capacity(self.samples.len());
        for s in &self.samples {
            let input = ModelInput {
                state: &s.input_state,
                questions: std::slice::from_ref(&s.question),
            };
            let out = forward(var.as_tensor(), &self.config, &input).ok()?;
            let p = out.per_question[0].marginal().ok()?;
            let py = p
                .narrow(0, s.label, 1)
                .ok()?
                .squeeze(0)
                .ok()?
                .clamp(NLL_FLOOR as f32, 1.0)
                .ok()?
                .log()
                .ok()?;
            losses.push(py.affine(-1.0, 0.0).ok()?);
        }
        let n = losses.len().max(1) as f64;
        let mut total = losses[0].clone();
        for l in &losses[1..] {
            total = total.add(l).ok()?;
        }
        let loss = total.affine(1.0 / n, 0.0).ok()?;
        let grads = loss.backward().ok()?;
        let grad = grads.get(var.as_tensor())?.flatten_all().ok()?;
        let values = grad.to_vec1::<f32>().ok()?;
        let masked: Vec<f32> = values
            .into_iter()
            .zip(&self.trainable)
            .map(|(g, keep)| if *keep { g } else { 0.0 })
            .collect();
        Tensor::from_vec(
            masked,
            vec![self.config.flat_size()],
            &candle_core::Device::Cpu,
        )
        .ok()
    }

    fn label(&self) -> &'static str {
        "edge0-head-chart"
    }
}

/// E5's D0/D1 record: *which* head-chart target this is, *how* the reference
/// kernel moves it, and *what* can honestly be certified about the pair.
///
/// The target identity names the data by digest (world, split, state tokens,
/// question tokens/options/kind, label and truth), the likelihood by its sum
/// convention and floor, and the chart by how many of the flat parameters are
/// free. The complement is **frozen**, not integrated — so the reference kernel
/// refreshes the whole chart's Gaussian block jointly (`β = 1`) and leaves no
/// Gaussian prior block behind. That is the "prior-refresh reference" E5 asks
/// for; the corresponding `Certificate` is the mixture constant `c = ω·Z_lo`
/// with `M = 1`, because `exp(−Ψ)` is a categorical likelihood in `[0,1]`.
///
/// The honest part is the ceiling. The only region-free bound available without
/// a verified sup of a neural likelihood is the floor one:
/// `Ψ = λ Σ −log max(p, floor) ≤ λ N ln(1/floor)`. It is sound, and on any
/// realistic `λN` it makes `Z_lo` astronomically small, so the published rate is
/// `c ≈ 0`. This type returns that certificate rather than dressing a numerical
/// estimate up as a theorem, and [`Self::ball_certificate`] takes a
/// caller-verified ceiling when one exists.
pub struct HeadChartReference {
    params: usize,
    trainable: usize,
    terms: usize,
    data_digest: u64,
}

impl HeadChartReference {
    pub fn new(potential: &Edge0Potential) -> Self {
        let mut h = Fnv::new();
        for s in &potential.samples {
            h.write(&s.world.to_le_bytes());
            h.write(format!("{:?}", s.split).as_bytes());
            h.write(&(s.input_state.len() as u64).to_le_bytes());
            for t in &s.input_state {
                h.write(&t.to_le_bytes());
            }
            h.write(format!("{:?}", s.question.kind).as_bytes());
            for t in &s.question.tokens {
                h.write(&t.to_le_bytes());
            }
            for t in &s.question.options {
                h.write(&t.to_le_bytes());
            }
            h.write(&(s.label as u64).to_le_bytes());
            for t in &s.truth {
                h.write(&t.to_bits().to_le_bytes());
            }
        }
        Self {
            params: potential.config.flat_size(),
            trainable: potential.trainable.iter().filter(|k| **k).count(),
            terms: potential.samples.len(),
            data_digest: h.finish(),
        }
    }

    /// Flat parameter count `P`.
    pub fn parameters(&self) -> usize {
        self.params
    }

    /// Chart dimension `r` — the coordinates the kernel actually moves.
    pub fn trainable_coordinates(&self) -> usize {
        self.trainable
    }

    pub fn terms(&self) -> usize {
        self.terms
    }

    pub fn data_digest(&self) -> u64 {
        self.data_digest
    }

    /// D0's target identity. `temperature` is the `λ` multiplying the *summed*
    /// NLL; it is part of what is sampled, so it is part of the hash.
    pub fn target_identity(&self, temperature: f64) -> TargetIdentity {
        TargetIdentity {
            kind: "edge0-head-chart-categorical-nll",
            prior: format!(
                "N(0,I_{}) on the head chart; complement frozen",
                self.trainable
            ),
            likelihood: format!(
                "exact latent-marginal categorical NLL, sum-form over {} samples, floor {NLL_FLOOR:e}",
                self.terms
            ),
            temperature: format!("lambda={temperature}"),
            chart: format!(
                "head-chart: {} of {} flat parameters move",
                self.trainable, self.params
            ),
            dimension: self.trainable,
            rank: self.trainable,
            precision: "f32 model / f64 potential",
            readout: format!("data digest {:#018x}", self.data_digest),
        }
    }

    /// D0's kernel identity for the D1 reference mixture: a joint prior refresh
    /// (`β = 1`) mixed with a frozen local pCN kernel. There is no adaptation,
    /// so step `0` is the (trivial) freeze point rather than an omission.
    pub fn kernel_identity(&self, local_beta: f64, omega: f64) -> KernelIdentity {
        assert!(0.0 < omega && omega <= 1.0);
        assert!(0.0 < local_beta && local_beta <= 1.0);
        KernelIdentity {
            proposal: format!(
                "{omega}*pcn(beta=1, joint prior refresh) + {}*pcn(beta={local_beta}) on the chart",
                1.0 - omega
            ),
            scan_order: "joint",
            adaptation_frozen: true,
            freezing_point: Some(0),
        }
    }

    pub fn claim_record(
        &self,
        temperature: f64,
        source_revision: impl Into<String>,
        local_beta: f64,
        omega: f64,
    ) -> ClaimRecord {
        let cert = self.clamp_certificate(temperature, omega);
        ClaimRecord {
            source_revision: source_revision.into(),
            target: self.target_identity(temperature),
            kernel: self.kernel_identity(local_beta, omega),
            proposal: "independence MH on the chart + frozen local pCN".to_string(),
            evidence: format!(
                "clamp ceiling b={:.3}, c={:.3e}, contract={}",
                cert.region.potential_ceiling,
                cert.minorization(),
                cert.is_contracting()
            ),
            open_assumptions: vec![
                format!("the NLL floor {NLL_FLOOR:e} is part of the target, not a guard"),
                "complement coordinates are frozen by the chart, not integrated".to_string(),
                "the clamp ceiling is sound but grows like lambda*N*27.6; c is not a usable rate"
                    .to_string(),
                "no verified sup of the neural likelihood on a finite ball; ball_certificate's ceiling is caller-supplied and not checked here"
                    .to_string(),
                "f32 model arithmetic against an f64 potential; no rounding certificate"
                    .to_string(),
            ],
        }
    }

    /// The sound region-free ceiling: `a = 1`, `b = λ·N·ln(1/floor)`, `M = 1`.
    pub fn clamp_certificate(&self, temperature: f64, omega: f64) -> Certificate {
        assert!(temperature > 0.0 && temperature.is_finite());
        let ceiling = temperature * self.terms as f64 * (1.0 / NLL_FLOOR).ln();
        Certificate::new(omega, VerifiedRegion::new(1.0, ceiling), 1.0)
    }

    /// `μ0(‖z‖ ≤ radius)` for the chart's standard Gaussian — exact for integer
    /// `r`, so a verified region's prior mass is a computation rather than a
    /// guess.
    pub fn ball_prior_mass(&self, radius: f64) -> f64 {
        chi_square_cdf(self.trainable, radius)
    }

    /// A certificate over a ball whose ceiling the **caller verified**. The
    /// prior mass is exact; the ceiling is not established by this module.
    pub fn ball_certificate(&self, radius: f64, potential_ceiling: f64, omega: f64) -> Certificate {
        Certificate::new(
            omega,
            VerifiedRegion::new(self.ball_prior_mass(radius), potential_ceiling),
            1.0,
        )
    }
}

/// An E5 finite-chart reference on a **one-dimensional** head chart.
///
/// This is a *reference estimate*, not a theorem certificate: the head chart's
/// potential is a neural likelihood with no proved `sup|Ψ'|` on the box, so its
/// truncation error cannot be bounded here. What is reported is the Gaussian
/// tail bound, the coarse/fine grid gap, and the plug-in value. Use
/// [`crate::evidence_enclosure`] when a certified enclosure is required.
pub struct ChartQuadrature {
    /// Posterior evidence `Z0 = ∫ e^{−Ψ}φ dz` on the fine grid.
    pub evidence: f64,
    /// `μ0(|z| > R) ≤ 2φ(R)/R`.
    pub tail_upper: f64,
    /// Relative change of the evidence between the coarse and fine grids.
    pub grid_gap: f64,
    /// Posterior-averaged `p(label | θ(z))` on the fine grid.
    pub predictive: f64,
    /// The same predictive at `z = 0` (no posterior averaging).
    pub plug_in: f64,
    pub radius: f64,
    pub cells: usize,
}

/// Quadrature reference for a one-column [`SubspaceTarget<Edge0Potential>`].
/// `cells` is the coarse grid; the fine grid has twice that, and the returned
/// values come from the fine grid. `None` for a chart that is not 1-D.
pub fn head_chart_quadrature_1d(
    target: &SubspaceTarget<Edge0Potential>,
    radius: f64,
    cells: usize,
    sample: usize,
) -> Result<Option<ChartQuadrature>> {
    if target.dim() != 1 || cells < 4 {
        return Ok(None);
    }
    let eval = |m: usize| -> Result<(f64, f64)> {
        let h = 2.0 * radius / m as f64;
        let mut z_sum = 0.0f64;
        let mut p_sum = 0.0f64;
        for i in 0..m {
            let z = -radius + (i as f64 + 0.5) * h;
            let zv = vec![z as f32];
            let theta = target.subspace().lift_tensor(&zv)?;
            let lik =
                target.lambda() * target.n_terms() as f64 * target.potential().mean_loss(&theta);
            let dens = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let w = (-lik).exp() * dens;
            z_sum += w;
            if let Some(p) = target.potential().label_probability(&theta, sample)? {
                p_sum += w * p;
            }
        }
        Ok((z_sum * h, p_sum * h))
    };

    let (z_coarse, _) = eval(cells)?;
    let (z_fine, p_fine) = eval(2 * cells)?;
    if !(z_fine > 0.0 && z_fine.is_finite()) {
        return Ok(None);
    }
    let plug_in = {
        let theta = target.subspace().lift_tensor(&[0f32])?;
        target
            .potential()
            .label_probability(&theta, sample)?
            .unwrap_or(f64::NAN)
    };
    let phi = (-0.5 * radius * radius).exp() / (2.0 * std::f64::consts::PI).sqrt();
    Ok(Some(ChartQuadrature {
        evidence: z_fine,
        tail_upper: (2.0 * phi / radius).min(1.0),
        grid_gap: (z_fine - z_coarse).abs() / z_fine,
        predictive: p_fine / z_fine,
        plug_in,
        radius,
        cells: 2 * cells,
    }))
}

/// A declared `f32` bridge for the frozen head inputs the certified chart
/// evaluation reads (`policy_logits`, path probabilities). The chart itself is
/// evaluated in `f64` interval arithmetic; these constants come from one `f32`
/// forward pass, and widening them by `1e-6` keeps the enclosure honest about
/// that conversion. It is **not** a rounding envelope on the quadrature sum.
pub const HEAD_CHART_F32_BRIDGE: f64 = 1e-6;

/// A certified evidence/predictive enclosure for a one-dimensional head chart.
///
/// Unlike [`ChartQuadrature`] this is not a reference estimate: `Psi'` and
/// `Psi''` along the chart are bounded analytically from the softmax mixing of
/// the frozen path probabilities, so the midpoint truncation error and the
/// Gaussian tail are explicit. The bound is only available when the chart
/// direction moves `policy_w`/`policy_b`; the path distributions `u_r` are then
/// constant in `z` and the label marginal is `sum_r softmax(L_r + z d_r) u_r`.
/// Any other direction (a `noul`/`score` head, or the encoder) returns `None`
/// rather than an unproved bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartEnclosure {
    pub evidence_lo: f64,
    pub evidence_hi: f64,
    pub predictive_lo: f64,
    pub predictive_hi: f64,
    /// `2 phi(R)/R`, the Gaussian tail mass outside the box.
    pub tail_upper: f64,
    /// Truncation error added on each side of the midpoint evidence.
    pub quadrature_error: f64,
    pub psi_prime_sup: f64,
    pub psi_second_sup: f64,
    pub f32_bridge: f64,
    pub radius: f64,
    pub cells: usize,
}

impl ChartEnclosure {
    pub fn evidence_contains(&self, z: f64) -> bool {
        self.evidence_lo <= z && z <= self.evidence_hi
    }

    pub fn predictive_contains(&self, p: f64) -> bool {
        self.predictive_lo <= p && p <= self.predictive_hi
    }
}

/// `1/sqrt(2 pi)` bracketed by two `f64`s (the same bracket D1 uses).
const INV_SQRT_2PI_CHART: Iv = Iv::span(0.3989422804014326, 0.3989422804014328);

fn widened(x: f64, bridge: f64) -> Iv {
    Iv::span(x - bridge, x + bridge)
}

/// Softmax of affine logits at an interval node, shifted by the largest upper
/// bound so every exponent is in `[-60, 0]`.
fn softmax_iv(logits: &[Iv]) -> Vec<Iv> {
    let shift = logits
        .iter()
        .map(|l| l.hi)
        .fold(f64::NEG_INFINITY, f64::max);
    let es: Vec<Iv> = logits
        .iter()
        .map(|l| l.sub(&Iv::point(shift)).exp())
        .collect();
    let mut sum = Iv::point(0.0);
    for e in &es {
        sum = sum.add(e);
    }
    es.iter().map(|e| e.div(&sum)).collect()
}

/// Certified enclosure of `Z0 = int e^{-Psi} phi dz` and the posterior-averaged
/// label probability on a one-column head chart. `sample` selects the label
/// whose predictive is enclosed. `None` for a chart that is not 1-D or whose
/// direction leaves the policy head.
pub fn head_chart_enclosure_1d(
    target: &SubspaceTarget<Edge0Potential>,
    radius: f64,
    cells: usize,
    sample: usize,
) -> Result<Option<ChartEnclosure>> {
    if target.dim() != 1 || cells < 4 || !(radius > 0.0 && radius <= 10.0) {
        return Ok(None);
    }
    let pot = target.potential();
    let cfg = pot.config();
    let samples = pot.samples();
    if sample >= samples.len() || samples.is_empty() {
        return Ok(None);
    }
    let sub = target.subspace();
    let p_total = cfg.flat_size();
    let mut v = vec![0f32; p_total];
    sub.basis().apply(&[1.0f32], &mut v);
    let dir: Vec<f64> = v
        .iter()
        .zip(sub.scale())
        .map(|(vi, s)| f64::from(*vi) * f64::from(*s))
        .collect();

    // The bound below is only valid when the chart moves the policy head: the
    // path distributions must not depend on `z`.
    let mut allowed = vec![false; p_total];
    for slot in cfg.layout() {
        if matches!(slot.name, "policy_w" | "policy_b") {
            for i in slot.start..slot.start + slot.len {
                allowed[i] = true;
            }
        }
    }
    if dir
        .iter()
        .enumerate()
        .any(|(i, d)| *d != 0.0 && !allowed[i])
    {
        return Ok(None);
    }

    let bridge = HEAD_CHART_F32_BRIDGE;

    // Per-sample constants: logits at z=0, logit slopes (from the affine policy
    // head evaluated at z=1), and the frozen path probabilities for the label.
    struct Consts {
        l0: Vec<f64>,
        d: Vec<f64>,
        u: Vec<f64>,
    }
    let mut consts = Vec::with_capacity(samples.len());
    for s in samples {
        let input = ModelInput {
            state: &s.input_state,
            questions: std::slice::from_ref(&s.question),
        };
        let theta0 = sub.lift_tensor(&[0f32])?;
        let q0 = &forward(&theta0, cfg, &input)?.per_question[0];
        let l0 = q0
            .policy_logits
            .to_dtype(candle_core::DType::F64)?
            .to_vec1::<f64>()?;
        let theta1 = sub.lift_tensor(&[1f32])?;
        let q1 = &forward(&theta1, cfg, &input)?.per_question[0];
        let l1 = q1
            .policy_logits
            .to_dtype(candle_core::DType::F64)?
            .to_vec1::<f64>()?;
        let u_all = q0.path_probs_vec()?;
        if l0.len() != l1.len() || u_all.len() != l0.len() {
            return Ok(None);
        }
        let u: Vec<f64> = u_all.iter().map(|row| row[s.label]).collect();
        let d: Vec<f64> = l1.iter().zip(&l0).map(|(a, b)| a - b).collect();
        consts.push(Consts { l0, d, u });
    }

    // Analytic derivative bounds of Psi(z) = lambda * sum_i -log max(p_i(z), floor).
    let lam = target.lambda();
    let mut psi_p = 0.0f64;
    let mut psi_q = 0.0f64;
    let mut dmax_all = 0.0f64;
    for c in &consts {
        // `p(z) = sum_r a_r(z) u_r` is a convex combination, so
        // `(log p)' = sum_r w_r (d_r - m)` is a weighted deviation bounded by
        // the range `D = max d - min d`, with no probability lower bound. Its
        // derivative adds another `D²`, so `|(log p)''| <= 2 D²`.
        let dmax = c.d.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + bridge;
        let dmin = c.d.iter().cloned().fold(f64::INFINITY, f64::min) - bridge;
        let d_range = (dmax - dmin).max(0.0);
        dmax_all = dmax_all.max(d_range);
        psi_p += d_range;
        psi_q += 2.0 * d_range * d_range;
    }
    psi_p *= lam;
    psi_q *= lam;

    let h = 2.0 * radius / cells as f64;
    let phi_max = INV_SQRT_2PI_CHART;
    let g2 = (psi_q + 1.0 + (psi_p + radius) * (psi_p + radius)) * phi_max.hi;
    // Predictive weight p(z): |p'| <= D p <= D, |p''| <= 3 D^2.
    let p2 = g2 + 2.0 * dmax_all * (psi_p + radius) * phi_max.hi
        + (3.0 * dmax_all * dmax_all) * phi_max.hi;
    let vol = 2.0 * radius;
    let quad_err = vol * h * h / 24.0 * g2;
    let pred_err = vol * h * h / 24.0 * p2;
    let node_err = (psi_p + radius) * phi_max.hi * crate::evidence_enclosure::NODE_UNCERTAINTY * vol;

    let p_at = |c: &Consts, z: Iv| -> Iv {
        let logits: Vec<Iv> = c
            .l0
            .iter()
            .zip(&c.d)
            .map(|(l, d)| widened(*l, bridge).add(&widened(*d, bridge).mul(&z)))
            .collect();
        let a = softmax_iv(&logits);
        let mut p = Iv::point(0.0);
        for (ar, u) in a.iter().zip(&c.u) {
            p = p.add(&ar.mul(&widened(*u, bridge)));
        }
        Iv::span(p.lo.clamp(0.0, 1.0), p.hi.clamp(0.0, 1.0))
    };
    let g_at = |z: Iv| -> Iv {
        let mut psi = Iv::point(0.0);
        for c in &consts {
            let p = p_at(c, z);
            let hi = p.hi.max(NLL_FLOOR);
            let lo = p.lo.max(NLL_FLOOR);
            let loss = Iv::span(-hi.ln(), -lo.ln());
            psi = psi.add(&loss);
        }
        let psi = psi.mul(&Iv::point(lam));
        let zz = z.mul(&z).mul(&Iv::point(-0.5)).exp();
        psi.neg().exp().mul(&phi_max).mul(&zz)
    };

    let mut evidence_mid = Iv::point(0.0);
    let mut pred_mid = Iv::point(0.0);
    for i in 0..cells {
        let z = Iv::point(-radius + (i as f64 + 0.5) * h);
        let g = g_at(z);
        evidence_mid = evidence_mid.add(&g);
        pred_mid = pred_mid.add(&p_at(&consts[sample], z).mul(&g));
    }
    evidence_mid = evidence_mid.mul(&Iv::point(h));
    pred_mid = pred_mid.mul(&Iv::point(h));

    let r = radius;
    let pdf = (-0.5 * r * r).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = (2.0 * pdf / r).min(1.0);
    let ev_lo = (evidence_mid.lo - quad_err - node_err).max(0.0);
    let ev_hi = evidence_mid.hi + quad_err + node_err + tail;
    if !(ev_lo > 0.0 && ev_hi.is_finite()) {
        return Ok(None);
    }
    let pr_lo = ((pred_mid.lo - pred_err - node_err) / ev_hi).clamp(0.0, 1.0);
    let pr_hi = ((pred_mid.hi + pred_err + node_err + tail) / ev_lo).clamp(0.0, 1.0);

    Ok(Some(ChartEnclosure {
        evidence_lo: ev_lo,
        evidence_hi: ev_hi,
        predictive_lo: pr_lo,
        predictive_hi: pr_hi,
        tail_upper: tail,
        quadrature_error: quad_err + node_err,
        psi_prime_sup: psi_p,
        psi_second_sup: psi_q,
        f32_bridge: bridge,
        radius,
        cells,
    }))
}

/// One joint prior-refresh proposal: every chart coordinate is redrawn from
/// `N(0, 1)` (`β = 1`), and the frozen complement stays at the reference point
/// through the chart's affine map. The ratio is the ordinary MH one, so this can
/// be the independence primitive of a mixture kernel.
pub fn prior_refresh_step(
    phi: &dyn ZPotential,
    z: &mut [f32],
    phi_z: f64,
    rng: &mut impl Rng,
) -> ZStep {
    crate::target::mh_pcn_z(phi, z, phi_z, 1.0, rng)
}

/// `P(χ²_r ≤ radius²)` for integer `r ≥ 1`, via the regularised lower
/// incomplete gamma series. `0` at `radius = 0`; `1` in the far tail.
pub fn chi_square_cdf(dim: usize, radius: f64) -> f64 {
    assert!(
        dim >= 1,
        "a chi-square needs at least one degree of freedom"
    );
    if !(radius.is_finite() && radius > 0.0) {
        return 0.0;
    }
    let x = 0.5 * radius * radius;
    let s = 0.5 * dim as f64;
    let mut ln_term = s * x.ln() - x - ln_gamma_half_integer(s + 1.0);
    let mut sum = 0.0;
    for n in 0..10_000 {
        let term = ln_term.exp();
        sum += term;
        if term <= 1e-18 * sum {
            break;
        }
        ln_term += x.ln() - (s + n as f64 + 1.0).ln();
    }
    sum.min(1.0)
}

/// `ln Γ(s)` for `s` a positive integer or half-integer, from the exact
/// factorials (`Γ(m) = (m−1)!`, `Γ(m+½) = (2m)!√π/(4^m m!)`).
fn ln_gamma_half_integer(s: f64) -> f64 {
    let twice = (2.0 * s).round();
    assert!(
        (twice - 2.0 * s).abs() < 1e-9,
        "s must be integer or half-integer"
    );
    let twice = twice as i64;
    if twice % 2 == 0 {
        let m = (twice / 2) as usize;
        assert!(m >= 1);
        (2..m).map(|k| (k as f64).ln()).sum()
    } else {
        let m = ((twice - 1) / 2) as usize;
        let ln_fact = |n: usize| -> f64 { (2..=n).map(|k| (k as f64).ln()).sum() };
        0.5 * std::f64::consts::PI.ln() + ln_fact(2 * m) - m as f64 * 4f64.ln() - ln_fact(m)
    }
}

/// The full-model chart, exposed explicitly so a later arm can vary the trunk
/// only under its own hypothesis and budget (`EDGE0_RLCD_PLAN.md` §6).
pub fn full_chart(config: Edge0Config, samples: Vec<Sample>) -> Edge0Potential {
    let p = config.flat_size();
    Edge0Potential::new(config, samples, vec![true; p])
}

/// A tensor of the frozen reference parameters, for the chart's `θ_ref`.
pub fn reference_theta(config: &Edge0Config, seed: u64) -> Tensor {
    crate::edge0_model::init(config, seed)
}

/// Utility used by fixtures: `1` at `index`, zero elsewhere.
pub fn one_hot(index: usize, len: usize) -> Tensor {
    let mut v = vec![0f32; len];
    v[index] = 1.0;
    Tensor::from_vec(v, vec![len], &candle_core::Device::Cpu).expect("cpu tensor")
}
