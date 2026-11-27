//! R14/P1 — known-target plateau-vs-spike check against exact quadrature.
//!
//! The declared loss is one-dimensional and has both structures the plan
//! contrasts:
//!
//! ```text
//! L(θ) = 0.4·θ²/(θ²+w²) + max(0, (|θ|−2)/0.5)²,      w = 0.1
//! ```
//!
//! * a **narrow deep spike** at `θ = 0` (`L ≈ 0` only over `|θ| ≲ 0.05`), which
//!   is where a radical optimizer collapses;
//! * a **broad high-prior-volume plateau** `0.2 ≲ |θ| ≲ 2` with `L ≈ 0.4`,
//!   inside the declared ceiling `τ = 0.5`.
//!
//! Under `Φ_{τ,κ,α}` with a Huberized hinge and `α = 0`, the hinge is exactly
//! zero on the whole plateau, so the target is the Gaussian prior restricted to
//! the good-enough set: the broad plateau gets its prior volume (~0.95) while
//! the spike stays a ~0.08-volume nicety.  A sharp Gibbs target
//! `∝ exp(−50·L) P0` inverts that ranking.
//!
//! The example runs four dispersed exact MH-pCNL chains at three temperatures,
//! reports rank-normalized split R̂ and bulk ESS on θ and both region
//! indicators, and compares the estimated masses/moments with the quadrature
//! reference within MCSE.  It also checks intra-basin SWA on a convex plateau
//! against an unaligned two-basin barrier (Lean S2).
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_plateau_quadrature
//! ```

use pcnl::diag::{ess_bulk, ess_tail, split_rhat_rank};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::swa::{barrier_height, batch_average, interpolate, straight_line_barrier};
use pcnl::vector_model::{EnergyTarget, VectorSamplerConfig, run_chain_with_prior};

/// The plateau-plus-spike loss.
#[derive(Clone, Copy)]
struct SpikePlateau;

const TAU: f64 = 0.5;
const SPIKE_WIDTH: f64 = 0.1;
const PLATEAU_LO: f64 = 0.5;
const PLATEAU_HI: f64 = 2.0;

const WELL: f64 = 0.1;

fn loss_at(x: f64) -> f64 {
    let u = x.abs();
    let spike = 0.4 * x * x / (x * x + WELL * WELL);
    let wall = if u > 2.0 {
        let d = (u - 2.0) / 0.5;
        d * d
    } else {
        0.0
    };
    spike + wall
}

fn grad_at(x: f64) -> f64 {
    let u = x.abs();
    let denom = x * x + WELL * WELL;
    let mut g = 0.4 * 2.0 * x * WELL * WELL / (denom * denom);
    if u > 2.0 {
        g += 2.0 * (u - 2.0) / (0.5 * 0.5) * x.signum();
    }
    g
}

impl EnergyTarget for SpikePlateau {
    fn dim(&self) -> usize {
        1
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let x = f64::from(*theta.first()?);
        if !x.is_finite() {
            return None;
        }
        let (l, g) = (loss_at(x), grad_at(x));
        if !l.is_finite() || !g.is_finite() {
            return None;
        }
        Some((l, vec![g as f32]))
    }
}

/// `Φ/T`, so the exact kernel targets `π_T ∝ exp(−Φ/T) P0`.
struct Tempered<'a, T: EnergyTarget> {
    inner: &'a T,
    temperature: f64,
}

impl<T: EnergyTarget> EnergyTarget for Tempered<'_, T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (u, g) = self.inner.energy_and_grad(theta)?;
        Some((
            u / self.temperature,
            g.into_iter().map(|x| (f64::from(x) / self.temperature) as f32).collect(),
        ))
    }
}

/// Sharp Gibbs comparison `λ·L`.
struct Sharp<'a, T: EnergyTarget> {
    inner: &'a T,
    lambda: f64,
}

impl<T: EnergyTarget> EnergyTarget for Sharp<'_, T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn energy_and_grad(&self, theta: &[f32]) -> Option<(f64, Vec<f32>)> {
        let (u, g) = self.inner.energy_and_grad(theta)?;
        Some((
            self.lambda * u,
            g.into_iter().map(|x| (self.lambda * f64::from(x)) as f32).collect(),
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct Reference {
    z: f64,
    broad: f64,
    spike: f64,
    mean: f64,
    second: f64,
}

/// Trapezoid quadrature of `exp(−energy(θ)) · N(0,1)(θ)` on `[−8, 8]`.
fn quadrature<E: EnergyTarget>(energy: &E, n: usize) -> Option<Reference> {
    let lo = -8.0f64;
    let hi = 8.0f64;
    let h = (hi - lo) / n as f64;
    let mut z = 0.0;
    let mut broad = 0.0;
    let mut spike = 0.0;
    let mut mean = 0.0;
    let mut second = 0.0;
    for k in 0..=n {
        let x = lo + k as f64 * h;
        let weight = if k == 0 || k == n { 0.5 } else { 1.0 } * h;
        let (u, _) = energy.energy_and_grad(&[x as f32])?;
        let prior = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let w = weight * (-u).exp() * prior;
        if !w.is_finite() {
            return None;
        }
        z += w;
        let ax = x.abs();
        if (PLATEAU_LO..=PLATEAU_HI).contains(&ax) {
            broad += w;
        }
        if ax <= SPIKE_WIDTH {
            spike += w;
        }
        mean += w * x;
        second += w * x * x;
    }
    if !(z.is_finite() && z > 0.0) {
        return None;
    }
    Some(Reference {
        z,
        broad: broad / z,
        spike: spike / z,
        mean: mean / z,
        second: second / z,
    })
}

fn indicators(draws: &[Vec<f32>], f: impl Fn(f64) -> bool) -> Vec<f64> {
    draws
        .iter()
        .map(|d| if f(f64::from(d[0])) { 1.0 } else { 0.0 })
        .collect()
}

fn main() {
    let inner = SpikePlateau;
    let config = SatisficingConfig {
        tau: TAU,
        kappa: 20.0,
        alpha: 0.0,
        delta: 0.05,
        hinge: Hinge::Huberized,
    };
    let sat = Satisficing::new(inner, config, "spike-plateau", "sum").expect("config valid");
    let prior_c = [1.0f64];

    println!("r14_gate=P1");
    println!(
        "target tau={TAU} kappa={} alpha={} delta={} hinge={}",
        config.kappa,
        config.alpha,
        config.delta,
        config.hinge.label()
    );
    println!(
        "target_identity={}",
        sat.identity(&prior_c)
    );
    println!(
        "loss broad_sample={:.6} spike_sample={:.6}",
        loss_at(1.0),
        loss_at(0.0)
    );

    let reference = quadrature(&sat, 400_000).expect("quadrature");
    println!(
        "ref z={:.8} broad={:.6} spike={:.6} mean={:+.6} second={:.6}",
        reference.z, reference.broad, reference.spike, reference.mean, reference.second
    );

    let mut gate = true;
    for temperature in [0.5f64, 1.0, 2.0] {
        let tempered = Tempered {
            inner: &sat,
            temperature,
        };
        let reference = quadrature(&tempered, 400_000).expect("quadrature");
        let cfg = VectorSamplerConfig {
            steps: 20_000,
            burn: 5_000,
            thin: 2,
            rho: 0.9,
            refresh: 0.02,
            prior_sigma: 1.0,
            draws: 7_500,
        };
        let starts = [-3.0f32, -1.5, 1.5, 3.0];
        let seeds = [211u64, 212, 213, 214];
        let mut chains = Vec::new();
        let mut accepted = 0usize;
        let mut proposals = 0usize;
        for (&start, &seed) in starts.iter().zip(&seeds) {
            let chain = run_chain_with_prior(&tempered, &prior_c, cfg, vec![start], seed)
                .expect("chain");
            accepted += chain.accepted;
            proposals += chain.proposals;
            chains.push(chain);
        }
        let theta: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.draws.iter().map(|d| f64::from(d[0])).collect())
            .collect();
        let broad: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| indicators(&c.draws, |x| (PLATEAU_LO..=PLATEAU_HI).contains(&x.abs())))
            .collect();
        let spike: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| indicators(&c.draws, |x| x.abs() <= SPIKE_WIDTH))
            .collect();

        let rhat = split_rhat_rank(&theta);
        let rhat_broad = split_rhat_rank(&broad);
        let rhat_spike = split_rhat_rank(&spike);
        let ess = ess_bulk(&theta);
        let ess_broad = ess_bulk(&broad);
        let ess_spike = ess_bulk(&spike);
        let tail = ess_tail(&theta);

        let pooled: Vec<f64> = theta.iter().flatten().copied().collect();
        let n = pooled.len() as f64;
        let mean = pooled.iter().sum::<f64>() / n;
        let second = pooled.iter().map(|x| x * x).sum::<f64>() / n;
        let mass_broad = broad.iter().flatten().sum::<f64>() / n;
        let mass_spike = spike.iter().flatten().sum::<f64>() / n;

        let mcse = |p: f64, ess: f64| (p * (1.0 - p) / ess.max(1.0)).sqrt();
        let mcse_mean = (second - mean * mean).max(0.0).sqrt() / ess.max(1.0).sqrt();
        let ok_broad = (mass_broad - reference.broad).abs() <= 3.0 * mcse(mass_broad, ess_broad);
        let ok_spike = (mass_spike - reference.spike).abs() <= 3.0 * mcse(mass_spike, ess_spike);
        let ok_mean = (mean - reference.mean).abs() <= 3.0 * mcse_mean + 1e-12;
        let ok_second = (second - reference.second).abs() <= 0.05;
        let ok_mix = rhat < 1.01
            && rhat_broad < 1.01
            && rhat_spike < 1.01
            && ess_broad >= 400.0
            && ess_spike >= 400.0
            && ess >= 400.0;
        let region_ok = reference.broad > 0.5 && reference.spike < 0.1;
        let pass = ok_broad && ok_spike && ok_mean && ok_second && ok_mix && region_ok;
        gate &= pass;
        println!(
            "T={temperature} accept={accepted}/{proposals} rhat={rhat:.5} rhat_broad={rhat_broad:.5} rhat_spike={rhat_spike:.5} ess_bulk={ess:.1} ess_broad={ess_broad:.1} ess_spike={ess_spike:.1} ess_tail={tail:.1}"
        );
        println!(
            "T={temperature} broad={mass_broad:.6} ref_broad={:.6} spike={mass_spike:.6} ref_spike={:.6} mean={mean:+.6} ref_mean={:+.6} second={second:.6} ref_second={:.6}",
            reference.broad, reference.spike, reference.mean, reference.second
        );
        println!(
            "T={temperature} pass={pass} (mass_broad={ok_broad} mass_spike={ok_spike} mean={ok_mean} second={ok_second} mixing={ok_mix} regions={region_ok})"
        );
    }

    // Sharp Gibbs control: the optimizer-like target ranks the spike first.
    let sharp = Sharp {
        inner: &inner,
        lambda: 50.0,
    };
    let sharp_ref = quadrature(&sharp, 400_000).expect("sharp quadrature");
    // Radical single-point Adam (full-convergence optimizer) from the same
    // broad region; it normalizes the small well gradient and dives to 0.
    let mut radical = 1.5f64;
    let (mut m, mut v) = (0.0f64, 0.0f64);
    for t in 1..=4_000 {
        let g = grad_at(radical);
        m = 0.9 * m + 0.1 * g;
        v = 0.999 * v + 0.001 * g * g;
        let mh = m / (1.0 - 0.9f64.powi(t));
        let vh = v / (1.0 - 0.999f64.powi(t));
        radical -= 0.01 * mh / (vh.sqrt() + 1e-8);
    }
    let radical = radical as f32;
    println!(
        "control sharp_broad={:.6} sharp_spike={:.6} radical_endpoint={:+.6} radical_loss={:.6}",
        sharp_ref.broad,
        sharp_ref.spike,
        radical,
        loss_at(f64::from(radical))
    );
    let sharp_inverts = sharp_ref.broad < sharp_ref.spike && reference.broad > reference.spike;
    gate &= sharp_inverts;

    // Intra-basin SWA: a convex plateau has no barrier, the unaligned
    // two-basin path does.
    let snapshots = vec![vec![-0.2f32], vec![0.1], vec![0.3]];
    let average = batch_average(&snapshots).unwrap();
    let convex_loss = |w: &[f32]| {
        let x = f64::from(w[0]) - 0.1;
        x * x
    };
    let mean_loss = snapshots.iter().map(|s| convex_loss(s)).sum::<f64>() / 3.0;
    let swa_loss = convex_loss(&average);
    let convex = straight_line_barrier(&[-1.0], &[1.0], 40, |w| {
        let x = f64::from(w[0]);
        x * x
    })
    .unwrap();
    let two_basin = straight_line_barrier(&[-1.0], &[1.0], 40, |w| {
        let x = f64::from(w[0]);
        (x * x - 1.0) * (x * x - 1.0)
    })
    .unwrap();
    let convex_ok = swa_loss <= mean_loss + 1e-12
        && barrier_height(&convex).unwrap() <= 1e-9
        && barrier_height(&two_basin).unwrap() > 0.5;
    gate &= convex_ok;
    println!(
        "swa snapshots={:?} average={:?} mean_loss={mean_loss:.6} swa_loss={swa_loss:.6} convex_barrier={:.6} two_basin_barrier={:.6} pass={convex_ok}",
        snapshots
            .iter()
            .map(|s| s[0])
            .collect::<Vec<_>>(),
        average,
        barrier_height(&convex).unwrap(),
        barrier_height(&two_basin).unwrap()
    );
    println!(
        "midpoint_alignment_demo={:.6}",
        interpolate(&[-1.0], &[1.0], 0.5).unwrap()[0]
    );
    println!("gate={gate}");
    if !gate {
        std::process::exit(1);
    }
}
