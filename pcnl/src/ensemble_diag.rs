//! R14/P3 — shared diagnostics for ensembles and compressed models.
//!
//! Five axes, one module, so every campaign reports the same numbers:
//!
//! 1. in-region training fit (`L_train ≤ τ`, quantiles),
//! 2. parameter and functional variability, including the effective rank
//!    (participation ratio) of the centered `M × (n_eval·q)` prediction matrix,
//! 3. flatness and the R13 §6 / Lean S3 Gaussian-mixture PAC-Bayes bound,
//! 4. generalization and calibration (RMSE/NLL/Brier, interval coverage,
//!    normalized residual `z²`),
//! 5. compression diagnostics (SWA weight-mean vs output-mean, teacher–student
//!    KL/TV and the Jensen–Shannon dark-knowledge gap, spread fidelity).
//!
//! The identity checks that decide whether these numbers mean anything are
//! unit tests: the squared-error ambiguity identity, Jensen–Shannon
//! nonnegativity and the convex-segment barrier sign.

/// Fraction of configurations inside the good-enough set.
pub fn in_region_fraction(losses: &[f64], tau: f64) -> Option<f64> {
    if losses.is_empty() || !tau.is_finite() {
        return None;
    }
    let finite = losses.iter().filter(|l| l.is_finite()).count();
    if finite == 0 {
        return None;
    }
    Some(
        losses
            .iter()
            .filter(|l| l.is_finite() && **l <= tau)
            .count() as f64
            / finite as f64,
    )
}

/// Empirical quantile with linear interpolation (`q ∈ [0,1]`).
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    Some(v[lo] * (1.0 - frac) + v[hi] * frac)
}

/// `(mean, min, median, max)` of the finite losses.
pub fn loss_summary(losses: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let v: Vec<f64> = losses.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((mean, min, quantile(&v, 0.5)?, max))
}

/// Pairwise Euclidean distance in parameter space.
pub fn pairwise_distances(weights: &[Vec<f32>]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..weights.len() {
        for j in i + 1..weights.len() {
            let d: f64 = weights[i]
                .iter()
                .zip(&weights[j])
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum();
            out.push(d.sqrt());
        }
    }
    out
}

/// Pairwise cosine similarity in parameter space.
pub fn pairwise_cosine(weights: &[Vec<f32>]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..weights.len() {
        for j in i + 1..weights.len() {
            let mut dot = 0.0;
            let mut na = 0.0;
            let mut nb = 0.0;
            for (&a, &b) in weights[i].iter().zip(&weights[j]) {
                let (a, b) = (f64::from(a), f64::from(b));
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            if na > 0.0 && nb > 0.0 {
                out.push(dot / (na.sqrt() * nb.sqrt()));
            }
        }
    }
    out
}

/// Per-input mean and variance of the ensemble prediction, plus the mean
/// pairwise absolute difference (a functional "disagreement" in task units).
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionSpread {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub disagreement: f64,
}

/// `preds` is `M × outputs`: one prediction row per member, aligned on the same
/// inputs.  (For several inputs, flatten the per-input rows.)
pub fn prediction_spread(preds: &[Vec<f64>]) -> Option<PredictionSpread> {
    let m = preds.len();
    if m == 0 {
        return None;
    }
    let q = preds[0].len();
    if q == 0 || preds.iter().any(|p| p.len() != q) {
        return None;
    }
    if preds.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let mut mean = vec![0.0; q];
    for p in preds {
        for (acc, v) in mean.iter_mut().zip(p) {
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= m as f64;
    }
    let mut variance = vec![0.0; q];
    for p in preds {
        for (acc, (v, mu)) in variance.iter_mut().zip(p.iter().zip(&mean)) {
            let d = v - mu;
            *acc += d * d;
        }
    }
    for v in &mut variance {
        *v /= m as f64;
    }
    let mut disagreement = 0.0;
    let mut pairs = 0usize;
    for i in 0..m {
        for j in i + 1..m {
            disagreement += preds[i]
                .iter()
                .zip(&preds[j])
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / q as f64;
            pairs += 1;
        }
    }
    if pairs > 0 {
        disagreement /= pairs as f64;
    }
    Some(PredictionSpread {
        mean,
        variance,
        disagreement,
    })
}

/// Effective rank (participation ratio) of the centered member prediction
/// matrix.  Members with identical functions give zero centered variance and
/// report the floor value `1.0` (one effective member), not a weight-space
/// rank inflated by permutation or scale symmetries.
pub fn effective_rank(member_predictions: &[Vec<f64>]) -> Option<f64> {
    let m = member_predictions.len();
    if m < 2 {
        return Some(1.0);
    }
    let q = member_predictions[0].len();
    if q == 0 || member_predictions.iter().any(|p| p.len() != q) {
        return None;
    }
    let mut mean = vec![0.0; q];
    for p in member_predictions {
        for (acc, v) in mean.iter_mut().zip(p) {
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= m as f64;
    }
    let mut centered = member_predictions.to_vec();
    for p in &mut centered {
        for (v, mu) in p.iter_mut().zip(&mean) {
            *v -= mu;
        }
    }
    // M×M Gram matrix; its eigenvalues are the nonzero covariance spectrum.
    let mut gram = vec![0.0f64; m * m];
    for i in 0..m {
        for j in i..m {
            let dot: f64 = centered[i]
                .iter()
                .zip(&centered[j])
                .map(|(a, b)| a * b)
                .sum();
            gram[i * m + j] = dot;
            gram[j * m + i] = dot;
        }
    }
    let total: f64 = (0..m).map(|i| gram[i * m + i]).sum();
    if !total.is_finite() || total <= 1e-18 {
        return Some(1.0);
    }
    let (evals, _) = crate::target::symmetric_eigen(&gram, m).ok()?;
    let sum: f64 = evals.iter().copied().filter(|e| *e > 1e-15).sum();
    let sumsq: f64 = evals
        .iter()
        .copied()
        .filter(|e| *e > 1e-15)
        .map(|e| e * e)
        .sum();
    if sumsq <= 0.0 {
        return Some(1.0);
    }
    Some((sum * sum / sumsq).max(1.0))
}

/// Bernoulli KL divergence `KL(r ‖ R)`.
pub fn kl_bernoulli(r: f64, big_r: f64) -> Option<f64> {
    let clamp = |x: f64| x.clamp(1e-12, 1.0 - 1e-12);
    if !r.is_finite() || !big_r.is_finite() || !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&big_r) {
        return None;
    }
    let (r, big_r) = (clamp(r), clamp(big_r));
    Some(r * (r / big_r).ln() + (1.0 - r) * ((1.0 - r) / (1.0 - big_r)).ln())
}

/// Maurer (2004) Theorem 5 mixture bound, inverted for the risk.
///
/// With probability `≥ 1−δ`,
/// `KL(R̂ ‖ R) ≤ (KL(Q‖P) + ln(2√n/δ)) / n`; the reported upper bound is the
/// largest `R ≥ R̂` satisfying it (bisection).
pub fn pac_bayes_mixture_bound(
    empirical_risk: f64,
    kl_q_p: f64,
    n: usize,
    delta: f64,
) -> Option<f64> {
    if n == 0
        || !empirical_risk.is_finite()
        || !kl_q_p.is_finite()
        || kl_q_p < 0.0
        || !delta.is_finite()
        || delta <= 0.0
        || delta >= 1.0
        || !(0.0..=1.0).contains(&empirical_risk)
    {
        return None;
    }
    let budget = (kl_q_p + (2.0 * (n as f64).sqrt() / delta).ln()) / n as f64;
    if budget <= 0.0 {
        return None;
    }
    if budget >= 30.0 || empirical_risk >= 1.0 {
        return Some(1.0);
    }
    let (mut lo, mut hi) = (empirical_risk, 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        match kl_bernoulli(empirical_risk, mid) {
            Some(kl) if kl < budget => lo = mid,
            _ => hi = mid,
        }
    }
    Some(hi.min(1.0))
}

/// One-sided Monte Carlo allowance `sqrt(ln(1/δ)/(2m))`.
pub fn monte_carlo_allowance(m: usize, delta: f64) -> Option<f64> {
    if m == 0 || !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return None;
    }
    Some((((1.0 / delta).ln()) / (2.0 * m as f64)).sqrt())
}

/// Mean squared error.
pub fn mse(pred: &[f64], target: &[f64]) -> Option<f64> {
    if pred.is_empty() || pred.len() != target.len() {
        return None;
    }
    Some(
        pred.iter()
            .zip(target)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f64>()
            / pred.len() as f64,
    )
}

pub fn rmse(pred: &[f64], target: &[f64]) -> Option<f64> {
    mse(pred, target).map(f64::sqrt)
}

/// Binary Brier score for probabilities in `[0,1]`.
pub fn brier_binary(prob: &[f64], label: &[bool]) -> Option<f64> {
    if prob.is_empty() || prob.len() != label.len() {
        return None;
    }
    Some(
        prob.iter()
            .zip(label)
            .map(|(p, y)| {
                let t = if *y { 1.0 } else { 0.0 };
                (p - t) * (p - t)
            })
            .sum::<f64>()
            / prob.len() as f64,
    )
}

/// Multiclass Brier score; `prob` is `n × k`, `label` one class index per row.
pub fn brier_multiclass(prob: &[Vec<f64>], label: &[usize]) -> Option<f64> {
    if prob.is_empty() || prob.len() != label.len() {
        return None;
    }
    let k = prob[0].len();
    if k == 0 || prob.iter().any(|p| p.len() != k) {
        return None;
    }
    let mut total = 0.0;
    for (p, &y) in prob.iter().zip(label) {
        if y >= k {
            return None;
        }
        for (c, v) in p.iter().enumerate() {
            let t = if c == y { 1.0 } else { 0.0 };
            total += (v - t) * (v - t);
        }
    }
    Some(total / prob.len() as f64)
}

/// Gaussian negative log-likelihood (up to the constant `½ln 2π`).
pub fn gaussian_nll(y: &[f64], mean: &[f64], variance: &[f64]) -> Option<f64> {
    if y.is_empty() || y.len() != mean.len() || y.len() != variance.len() {
        return None;
    }
    let mut total = 0.0;
    for ((y, m), v) in y.iter().zip(mean).zip(variance) {
        if !v.is_finite() || *v <= 0.0 {
            return None;
        }
        total += 0.5 * ((y - m) * (y - m) / v + v.ln());
    }
    Some(total / y.len() as f64)
}

/// Acklam's inverse normal CDF (relative error &lt; 1.2e-9).
pub fn normal_quantile(p: f64) -> Option<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return None;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return Some(
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        );
    }
    if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return Some(
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        );
    }
    let q = p - 0.5;
    let r = q * q;
    Some(
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0),
    )
}

/// Central interval coverage of `y` by `mean ± z_{level}·σ`.
pub fn normal_coverage(mean: &[f64], sigma: &[f64], y: &[f64], level: f64) -> Option<f64> {
    if mean.is_empty() || mean.len() != sigma.len() || mean.len() != y.len() {
        return None;
    }
    let z = normal_quantile(0.5 + level / 2.0)?;
    let mut covered = 0usize;
    for ((m, s), y) in mean.iter().zip(sigma).zip(y) {
        if !s.is_finite() || *s < 0.0 {
            return None;
        }
        if (y - m).abs() <= z * s {
            covered += 1;
        }
    }
    Some(covered as f64 / y.len() as f64)
}

/// Mean normalized residual `z²`.
pub fn normalized_residual_z2(mean: &[f64], sigma: &[f64], y: &[f64]) -> Option<f64> {
    if mean.is_empty() || mean.len() != sigma.len() || mean.len() != y.len() {
        return None;
    }
    let mut total = 0.0;
    for ((m, s), y) in mean.iter().zip(sigma).zip(y) {
        if !s.is_finite() || *s <= 0.0 {
            return None;
        }
        total += (y - m) * (y - m) / (s * s);
    }
    Some(total / y.len() as f64)
}

/// Empirical risk under a draw from the retained mixture, with its one-sided
/// Monte Carlo allowance.
pub fn perturbation_risk<F>(trials: usize, delta: f64, mut draw_loss: F) -> Option<(f64, f64)>
where
    F: FnMut() -> f64,
{
    if trials == 0 {
        return None;
    }
    let mut sum = 0.0;
    for _ in 0..trials {
        let l = draw_loss();
        if !l.is_finite() {
            return None;
        }
        sum += l;
    }
    let mean = sum / trials as f64;
    Some((mean, monte_carlo_allowance(trials, delta)?))
}

/// `KL(p ‖ q)` for probability vectors.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.is_empty() || p.len() != q.len() {
        return None;
    }
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if !pi.is_finite() || !qi.is_finite() || pi < 0.0 || qi < 0.0 {
            return None;
        }
        if pi == 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return None;
        }
        total += pi * (pi / qi).ln();
    }
    Some(total)
}

/// Total-variation distance `½‖p−q‖₁`.
pub fn tv_distance(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.is_empty() || p.len() != q.len() {
        return None;
    }
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if !pi.is_finite() || !qi.is_finite() || pi < 0.0 || qi < 0.0 {
            return None;
        }
        total += (pi - qi).abs();
    }
    Some(0.5 * total)
}

/// `H(p_ens) − mean_m H(p_m) = mean_m KL(p_m ‖ p_ens) = JS gap ≥ 0`.
pub fn js_dark_knowledge_gap(members: &[Vec<f64>]) -> Option<f64> {
    if members.is_empty() {
        return None;
    }
    let k = members[0].len();
    if k == 0 || members.iter().any(|p| p.len() != k) {
        return None;
    }
    let mut mean = vec![0.0; k];
    for p in members {
        for (acc, v) in mean.iter_mut().zip(p) {
            if !v.is_finite() || *v < 0.0 {
                return None;
            }
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= members.len() as f64;
    }
    let mut total = 0.0;
    for p in members {
        total += kl_divergence(p, &mean)?;
    }
    Some(total / members.len() as f64)
}

/// Teacher–student KL/TV plus the ensemble-entropy gap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fidelity {
    pub kl: f64,
    pub tv: f64,
    pub js_gap: f64,
}

pub fn teacher_student_fidelity(
    teacher_members: &[Vec<f64>],
    student: &[f64],
) -> Option<Fidelity> {
    let n = teacher_members.len();
    if n == 0 {
        return None;
    }
    let k = teacher_members[0].len();
    if k == 0 || teacher_members.iter().any(|p| p.len() != k) || student.len() != k {
        return None;
    }
    let mut mean = vec![0.0; k];
    for p in teacher_members {
        for (acc, v) in mean.iter_mut().zip(p) {
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= n as f64;
    }
    Some(Fidelity {
        kl: kl_divergence(&mean, student)?,
        tv: tv_distance(&mean, student)?,
        js_gap: js_dark_knowledge_gap(teacher_members)?,
    })
}

/// Squared-error ambiguity identity:
/// `mean_m ‖y−f_m‖² = ‖y−f̄‖² + mean_m ‖f_m−f̄‖²`.
///
/// Returns `(total, bias_sq, ambiguity)`.
pub fn squared_error_ambiguity(preds: &[Vec<f64>], y: &[f64]) -> Option<(f64, f64, f64)> {
    let m = preds.len();
    if m == 0 || preds.iter().any(|p| p.len() != y.len()) {
        return None;
    }
    let q = y.len();
    let total = preds
        .iter()
        .map(|p| {
            p.iter()
                .zip(y)
                .map(|(f, y)| (y - f) * (y - f))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (m as f64 * q as f64);
    let mut mean = vec![0.0; q];
    for p in preds {
        for (acc, v) in mean.iter_mut().zip(p) {
            *acc += *v;
        }
    }
    for v in &mut mean {
        *v /= m as f64;
    }
    let bias_sq = y
        .iter()
        .zip(&mean)
        .map(|(y, f)| (y - f) * (y - f))
        .sum::<f64>()
        / q as f64;
    let ambiguity = preds
        .iter()
        .map(|p| {
            p.iter()
                .zip(&mean)
                .map(|(f, mu)| (f - mu) * (f - mu))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (m as f64 * q as f64);
    Some((total, bias_sq, ambiguity))
}

/// `KL(π_student ‖ π_ensemble)` for a discrete policy; identical to
/// [`kl_divergence`] with the argument order made explicit for the RL track.
pub fn policy_kl(student: &[f64], teacher: &[f64]) -> Option<f64> {
    kl_divergence(student, teacher)
}

/// Lean S5's finite-horizon coupling bound
/// `|V^{π_stu} − V^{π_ens}| ≤ ½·R_span·H(H+1)·δ`, with `δ` a per-step TV
/// bound.
pub fn pinsker_return_bound(return_span: f64, horizon: usize, tv_step: f64) -> Option<f64> {
    if !return_span.is_finite() || return_span < 0.0 {
        return None;
    }
    if horizon == 0 || !tv_step.is_finite() || tv_step < 0.0 {
        return None;
    }
    let h = horizon as f64;
    Some(0.5 * return_span * h * (h + 1.0) * tv_step)
}
