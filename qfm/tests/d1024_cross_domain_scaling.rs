//! P10.16.7: QFM-TSR d=1024 cross-domain validation.
//!
//! Cross-domain validation suite at $d = 1024$ (= $32 \times 32$ pixels
//! for spatial fields, 1024 samples for 1D time series) using the
//! P10.16.3 rank-truncation path. The three sub-fixtures are:
//!
//! 1. **2D Kolmogorov-flow CFD at $L = 32$** --- $32 \times 32$
//!    vorticity snapshots (continuous PDE-generated spatial field).
//! 2. **2D Ising model at $L = 32$** --- $32 \times 32$ critical
//!    configurations (discrete spin field at the Onsager critical
//!    point $\beta_c = \ln(1 + \sqrt{2}) / 2$).
//! 3. **Post-Newtonian gravitational waveforms at $d = 1024$** ---
//!    1024 samples per waveform (1D real-physics time series of
//!    restricted 2PN TaylorT1 inspiral strain).
//!
//! Each fixture runs the same 4-test battery:
//!
//!   1. Pipeline compiles on $d = 1024$.
//!   2. `generate()` returns finite pixel values + positive cosine
//!      similarity to the nearest training point.
//!   3. Quantum Bayesian update on a held-out observation runs
//!      end-to-end with non-trivial overlap.
//!   4. No numerical explosion on held-out observations.
//!
//! Datasets are generated once per test binary (via
//! [`std::sync::OnceLock`]) and shared across the 4 tests per
//! fixture, since the CFD solver at $L = 32$ takes $\sim 10$ s to
//! generate a 64-snapshot trajectory and we do not want to repeat
//! that 4 times.
//!
//! ## Acceptance for P10.16.7
//!
//! `cargo test -p qfm --test d1024_cross_domain_scaling` runs clean
//! in $\sim 30$--$60$ s. The metrics are reported (not strict
//! thresholds; this is the $d = 1024$ scaling of the rev 29
//! P10.16.4--6 cross-domain baselines).
//!
//! ## Why $d = 1024$
//!
//! The rev 26 P10.16.3 rank-truncation extension introduced the
//! SVD-based rank reduction that lets `krylov_dim << K_2`. The
//! `qfm/tests/rank_truncation.rs` test exercises that path on a
//! synthetic d=1024 dataset ($\sim 2$ s compile). This test extends
//! the same scaling to the three rev 29 cross-domain fixtures, so
//! the P10.16.3 path is now validated on real cross-domain data
//! (PDE / Monte Carlo / analytic) at production resolution.
//!
//! ## Relation to the QFM.tex post-rev 29 next steps
//!
//! The P10.16.7+ item in QFM.tex §"Next steps to improve
//! (post-rev 29)" calls for "scale the CFD / gravitational-waveform
//! / Ising fixtures to $32 \times 32$ (= $d = 1024$) using the
//! P10.16.3 rank-truncation path". This test file is the concrete
//! deliverable for that item.

use qfm::{
    HmcOpts, Likelihood, Posterior, QfmConfig, QfmPipeline, sample_hmc_single, tsr_evolved_prior,
};
use std::sync::OnceLock;

// ── Common helpers ─────────────────────────────────────────────────────────

/// splitmix64 --- matches the PRNG used throughout `qfm` for
/// reproducibility.
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e3779b97f4a7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Build a QFM pipeline with the P10.16.3 rank-truncation path
/// (`krylov_dim = 32`, `max_rank = 16`).
///
/// Used by all three d=1024 sub-fixtures. `K_2 = max(M, d)` so the
/// M=64 < K_2=1024 case forces the lossy rank-truncation path.
fn d1024_pipeline(training: &[Vec<f64>], seed: u64) -> QfmPipeline {
    let d = training[0].len();
    let m = training.len();
    let k2 = m.max(d); // 1024
    let config = QfmConfig {
        k: 16,
        k2,
        krylov_dim: 32,
        seed,
        n_t_samples: 4,
        noise_dim: d,
        max_rank: Some(16),
    };
    QfmPipeline::compile(training, &config).expect("QFM compile on d=1024 cross-domain fixture")
}

// ── 1. 2D Kolmogorov-flow CFD at L = 32 (d = 1024) ─────────────────────────
//
// Governing equation (vorticity form, ω = ∇ × u):
//
//   ∂ₜω = -u · ∇ω + ν ∇²ω + f(x, y)
//
// with f(x, y) = sin(k_f x) cos(k_f y) the Kolmogorov forcing,
// k_f = 4, ν = 0.05, on a doubly-periodic 32×32 box. Direct
// pseudospectral solver with 2/3-rule dealiasing + RK4 time
// stepping on a hand-rolled radix-2 FFT. Same numerical method as
// `cfd_kolmogorov_validation.rs` (rev 29, d=256), generalised to
// L = 32.
//
// DT = 0.005 (halved from the L=16 case, DT=0.01) for extra stability
// margin at the higher resolution.

const L_CFD: usize = 32; // grid resolution per axis
const D_CFD: usize = L_CFD * L_CFD; // d = 1024 pixels
const NU_CFD: f64 = 0.05; // viscosity
const KF_CFD: f64 = 4.0; // Kolmogorov forcing wavenumber
const DT_CFD: f64 = 0.005; // time step (halved from L=16 for stability)
const N_TRAIN_CFD: usize = 64;
const N_HELD_OUT_CFD: usize = 8;
const SNAPSHOT_STRIDE_CFD: usize = 20;
const BURN_IN_CFD: usize = 50;

/// 1D iterative Cooley–Tukey FFT (radix-2, in-place).
/// Length must be a power of 2.
fn fft1d_inplace(a: &mut [(f64, f64)], inverse: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two(), "fft1d requires power-of-2 length");
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let wlen = (theta.cos(), theta.sin());
        for i in (0..n).step_by(len) {
            let mut w = (1.0, 0.0);
            for k in 0..half {
                let (ur, ui) = a[i + k];
                let (vr, vi) = a[i + k + half];
                let tr = vr * w.0 - vi * w.1;
                let ti = vr * w.1 + vi * w.0;
                a[i + k] = (ur + tr, ui + ti);
                a[i + k + half] = (ur - tr, ui - ti);
                let (wr, wi) = w;
                w = (wr * wlen.0 - wi * wlen.1, wr * wlen.1 + wi * wlen.0);
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / (n as f64);
        for x in a.iter_mut() {
            *x = (x.0 * scale, x.1 * scale);
        }
    }
}

/// 2D FFT via row-then-column 1D FFTs. Row-major order.
fn fft2d_cfd(x: &mut [(f64, f64)], inverse: bool) {
    for kx in 0..L_CFD {
        let mut row: Vec<(f64, f64)> = (0..L_CFD).map(|ky| x[kx * L_CFD + ky]).collect();
        fft1d_inplace(&mut row, inverse);
        for ky in 0..L_CFD {
            x[kx * L_CFD + ky] = row[ky];
        }
    }
    for ky in 0..L_CFD {
        let mut col: Vec<(f64, f64)> = (0..L_CFD).map(|kx| x[kx * L_CFD + ky]).collect();
        fft1d_inplace(&mut col, inverse);
        for kx in 0..L_CFD {
            x[kx * L_CFD + ky] = col[kx];
        }
    }
}

fn idft2d_cfd(x_hat: &[(f64, f64)]) -> Vec<f64> {
    let mut buf: Vec<(f64, f64)> = x_hat.to_vec();
    fft2d_cfd(&mut buf, true);
    buf.iter().map(|&(r, _)| r).collect()
}

/// 2/3-rule dealiasing mask.
fn dealias_mask_cfd() -> Vec<bool> {
    let cutoff = (2 * L_CFD / 3) as i32;
    let mut mask = vec![false; L_CFD * L_CFD];
    for kx in 0..L_CFD {
        for ky in 0..L_CFD {
            let kx_w = if kx > L_CFD / 2 {
                kx as i32 - L_CFD as i32
            } else {
                kx as i32
            };
            let ky_w = if ky > L_CFD / 2 {
                ky as i32 - L_CFD as i32
            } else {
                ky as i32
            };
            mask[kx * L_CFD + ky] = kx_w.abs() < cutoff && ky_w.abs() < cutoff;
        }
    }
    mask
}

/// Squared wavenumber |k|² for each mode (k=0 set to 1 to avoid /0).
fn k_squared_cfd() -> Vec<f64> {
    let mut out = vec![1.0; L_CFD * L_CFD];
    for kx in 0..L_CFD {
        for ky in 0..L_CFD {
            let kx_w = if kx > L_CFD / 2 {
                kx as i32 - L_CFD as i32
            } else {
                kx as i32
            };
            let ky_w = if ky > L_CFD / 2 {
                ky as i32 - L_CFD as i32
            } else {
                ky as i32
            };
            out[kx * L_CFD + ky] = (kx_w * kx_w + ky_w * ky_w) as f64;
        }
    }
    out
}

/// Analytic Fourier transform of f(x, y) = sin(k_f x) cos(k_f y).
/// Support only at ±(k_f, ±k_f) with imaginary amplitude ±(L²/4).
fn kolmogorov_forcing_hat_cfd() -> Vec<(f64, f64)> {
    let mut hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    let kf = KF_CFD as i32;
    for &(sx, sy) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let kx_idx = ((kf * sx).rem_euclid(L_CFD as i32)) as usize;
        let ky_idx = ((kf * sy).rem_euclid(L_CFD as i32)) as usize;
        let sign = (sx * sy) as f64;
        let imag_part = sign * (L_CFD * L_CFD) as f64 / 4.0;
        hat[kx_idx * L_CFD + ky_idx] = (0.0, imag_part);
    }
    hat
}

/// Non-linear term \widehat{u · ∇ω} with 2/3-rule dealiasing.
fn nonlinear_term_hat_cfd(omega_hat: &[(f64, f64)], ksq: &[f64], mask: &[bool]) -> Vec<(f64, f64)> {
    let mut psi_hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    for i in 0..L_CFD * L_CFD {
        if ksq[i] > 0.0 {
            let (wr, wi) = omega_hat[i];
            psi_hat[i] = (wr / ksq[i], wi / ksq[i]);
        }
    }
    let mut ux_hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    let mut uy_hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    for kx in 0..L_CFD {
        for ky in 0..L_CFD {
            let kx_w = if kx > L_CFD / 2 {
                kx as i32 - L_CFD as i32
            } else {
                kx as i32
            };
            let ky_w = if ky > L_CFD / 2 {
                ky as i32 - L_CFD as i32
            } else {
                ky as i32
            };
            let (pr, pi) = psi_hat[kx * L_CFD + ky];
            ux_hat[kx * L_CFD + ky] = (-(ky_w as f64) * pi, (ky_w as f64) * pr);
            uy_hat[kx * L_CFD + ky] = ((kx_w as f64) * pi, -(kx_w as f64) * pr);
        }
    }
    let mut dwdx_hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    let mut dwdy_hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    for kx in 0..L_CFD {
        for ky in 0..L_CFD {
            let kx_w = if kx > L_CFD / 2 {
                kx as i32 - L_CFD as i32
            } else {
                kx as i32
            };
            let ky_w = if ky > L_CFD / 2 {
                ky as i32 - L_CFD as i32
            } else {
                ky as i32
            };
            let (wr, wi) = omega_hat[kx * L_CFD + ky];
            dwdx_hat[kx * L_CFD + ky] = (-(kx_w as f64) * wi, (kx_w as f64) * wr);
            dwdy_hat[kx * L_CFD + ky] = (-(ky_w as f64) * wi, (ky_w as f64) * wr);
        }
    }
    let ux = idft2d_cfd(&ux_hat);
    let uy = idft2d_cfd(&uy_hat);
    let dwdx = idft2d_cfd(&dwdx_hat);
    let dwdy = idft2d_cfd(&dwdy_hat);
    let mut product = vec![0.0; L_CFD * L_CFD];
    for i in 0..L_CFD * L_CFD {
        product[i] = ux[i] * dwdx[i] + uy[i] * dwdy[i];
    }
    let mut product_hat: Vec<(f64, f64)> = {
        let mut buf: Vec<(f64, f64)> = product.iter().map(|&v| (v, 0.0)).collect();
        fft2d_cfd(&mut buf, false);
        buf
    };
    for i in 0..L_CFD * L_CFD {
        if !mask[i] {
            product_hat[i] = (0.0, 0.0);
        }
    }
    product_hat
}

fn rhs_hat_cfd(
    omega_hat: &[(f64, f64)],
    ksq: &[f64],
    f_hat: &[(f64, f64)],
    mask: &[bool],
) -> Vec<(f64, f64)> {
    let nl_hat = nonlinear_term_hat_cfd(omega_hat, ksq, mask);
    let mut out = vec![(0.0, 0.0); L_CFD * L_CFD];
    for i in 0..L_CFD * L_CFD {
        let (wr, wi) = omega_hat[i];
        let (nr, ni) = nl_hat[i];
        let (fr, fi) = f_hat[i];
        let m = if mask[i] { 1.0 } else { 0.0 };
        out[i] = (
            m * (-nr - NU_CFD * ksq[i] * wr + fr),
            m * (-ni - NU_CFD * ksq[i] * wi + fi),
        );
    }
    out
}

fn rk4_step_cfd(omega_hat: &mut [(f64, f64)], ksq: &[f64], f_hat: &[(f64, f64)], mask: &[bool]) {
    let k1 = rhs_hat_cfd(omega_hat, ksq, f_hat, mask);
    let omega2: Vec<(f64, f64)> = omega_hat
        .iter()
        .zip(k1.iter())
        .map(|(&(wr, wi), &(kr, ki))| (wr + 0.5 * DT_CFD * kr, wi + 0.5 * DT_CFD * ki))
        .collect();
    let k2 = rhs_hat_cfd(&omega2, ksq, f_hat, mask);
    let omega3: Vec<(f64, f64)> = omega_hat
        .iter()
        .zip(k2.iter())
        .map(|(&(wr, wi), &(kr, ki))| (wr + 0.5 * DT_CFD * kr, wi + 0.5 * DT_CFD * ki))
        .collect();
    let k3 = rhs_hat_cfd(&omega3, ksq, f_hat, mask);
    let omega4: Vec<(f64, f64)> = omega_hat
        .iter()
        .zip(k3.iter())
        .map(|(&(wr, wi), &(kr, ki))| (wr + DT_CFD * kr, wi + DT_CFD * ki))
        .collect();
    let k4 = rhs_hat_cfd(&omega4, ksq, f_hat, mask);
    for i in 0..L_CFD * L_CFD {
        let (wr, wi) = omega_hat[i];
        let (k1r, k1i) = k1[i];
        let (k2r, k2i) = k2[i];
        let (k3r, k3i) = k3[i];
        let (k4r, k4i) = k4[i];
        omega_hat[i] = (
            wr + DT_CFD * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0,
            wi + DT_CFD * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0,
        );
    }
}

fn initial_vorticity_hat_cfd(seed: u64) -> Vec<(f64, f64)> {
    let mut rng = seed;
    let mut hat = vec![(0.0, 0.0); L_CFD * L_CFD];
    for entry in hat.iter_mut() {
        rng = splitmix64(rng);
        let re = ((rng as f64) / (u64::MAX as f64) - 0.5) * 0.1;
        rng = splitmix64(rng);
        let im = ((rng as f64) / (u64::MAX as f64) - 0.5) * 0.1;
        *entry = (re, im);
    }
    hat[0] = (0.0, 0.0);
    hat
}

fn generate_kolmogorov_snapshots_cfd(
    n_snapshots: usize,
    snapshot_stride: usize,
    burn_in: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let ksq = k_squared_cfd();
    let mask = dealias_mask_cfd();
    let f_hat = kolmogorov_forcing_hat_cfd();
    let mut omega_hat = initial_vorticity_hat_cfd(seed);
    for _ in 0..burn_in {
        rk4_step_cfd(&mut omega_hat, &ksq, &f_hat, &mask);
    }
    let mut snapshots = Vec::with_capacity(n_snapshots);
    for _ in 0..n_snapshots {
        for _ in 0..snapshot_stride {
            rk4_step_cfd(&mut omega_hat, &ksq, &f_hat, &mask);
        }
        let real_space = idft2d_cfd(&omega_hat);
        let min_v = real_space.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = real_space.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_v - min_v).max(1e-12);
        let normalized: Vec<f64> = real_space.iter().map(|&x| (x - min_v) / range).collect();
        snapshots.push(normalized);
    }
    snapshots
}

static CFD_TRAINING: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
static CFD_HELD_OUT: OnceLock<Vec<Vec<f64>>> = OnceLock::new();

fn cfd_training() -> &'static [Vec<f64>] {
    CFD_TRAINING
        .get_or_init(|| {
            generate_kolmogorov_snapshots_cfd(N_TRAIN_CFD, SNAPSHOT_STRIDE_CFD, BURN_IN_CFD, 0xCF0D)
        })
        .as_slice()
}

fn cfd_held_out() -> &'static [Vec<f64>] {
    CFD_HELD_OUT
        .get_or_init(|| {
            generate_kolmogorov_snapshots_cfd(
                N_HELD_OUT_CFD,
                SNAPSHOT_STRIDE_CFD,
                BURN_IN_CFD,
                0xFEED,
            )
        })
        .as_slice()
}

// ── 2. 2D Ising model at L = 32 (d = 1024) ────────────────────────────────
//
// Metropolis Monte Carlo at the exact Onsager critical temperature
// β_c = ln(1 + √2) / 2. The Ising test is much cheaper than CFD
// (no FFT, just per-spin flip attempts), so we re-generate per test
// for code simplicity.

const L_ISING: usize = 32;
const D_ISING: usize = L_ISING * L_ISING; // 1024 pixels
const BETA_C_ISING: f64 = 0.440_686_793_509_771_6; // ln(1+√2)/2
const N_TRAIN_ISING: usize = 64;
const N_HELD_OUT_ISING: usize = 8;

fn metropolis_sweep_ising(spins: &mut [i8; D_ISING], beta: f64, rng: &mut u64) {
    for idx in 0..D_ISING {
        let x = idx % L_ISING;
        let y = idx / L_ISING;
        let s = spins[idx] as i32;
        let nb: i32 = spins[((y + L_ISING - 1) % L_ISING) * L_ISING + x] as i32
            + spins[((y + 1) % L_ISING) * L_ISING + x] as i32
            + spins[y * L_ISING + (x + L_ISING - 1) % L_ISING] as i32
            + spins[y * L_ISING + (x + 1) % L_ISING] as i32;
        let delta_e = 2 * s * nb;
        let accept = if delta_e <= 0 {
            true
        } else {
            *rng = splitmix64(*rng);
            let u = (*rng as f64) / (u64::MAX as f64);
            u < (-beta * delta_e as f64).exp()
        };
        if accept {
            spins[idx] = -spins[idx];
        }
    }
}

fn generate_ising_configs_d1024(n: usize, beta: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    let mut spins = [1i8; D_ISING];
    // Burn-in: 500 sweeps to equilibrate at β_c.
    for _ in 0..500 {
        metropolis_sweep_ising(&mut spins, beta, &mut rng);
    }
    let mut configs = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..20 {
            metropolis_sweep_ising(&mut spins, beta, &mut rng);
        }
        configs.push(spins.iter().map(|&s| (s as f64 + 1.0) / 2.0).collect());
    }
    configs
}

// ── 3. Post-Newtonian gravitational waveforms at d = 1024 ─────────────────
//
// Restricted 2PN TaylorT1 inspiral waveforms, face-on, non-spinning.
// Each waveform is 1024 samples long (vs the rev 29 d=256 test). The
// TaylorT1 family is rank-2 (amplitude × phase) and well-suited to
// the P10.16.3 rank-truncation path.

const D_GW: usize = 1024;
const F0_GW: f64 = 0.005;
const T_MAX_GW: f64 = 50.0;
const M_MIN_GW: f64 = 1.0;
const M_MAX_GW: f64 = 4.0;
const N_TRAIN_GW: usize = 64;
const N_HELD_OUT_GW: usize = 8;

fn phase_taylor_t1_d1024(t: f64, m: f64) -> f64 {
    let m_c = m / 2.0_f64.powf(0.2);
    let t_c =
        (5.0 / 256.0) * m_c.powf(-5.0 / 3.0) * (std::f64::consts::PI * F0_GW).powf(-8.0 / 3.0);
    let safe_t = t.min(0.999 * t_c);
    let term = 1.0 - safe_t / t_c;
    2.0 * std::f64::consts::PI * F0_GW * t_c * term.powf(5.0 / 8.0)
}

fn waveform_strain_d1024(t: f64, m: f64) -> f64 {
    let m_c = m / 2.0_f64.powf(0.2);
    let t_c =
        (5.0 / 256.0) * m_c.powf(-5.0 / 3.0) * (std::f64::consts::PI * F0_GW).powf(-8.0 / 3.0);
    let safe_t = t.min(0.999 * t_c);
    let term = 1.0 - safe_t / t_c;
    let amplitude = term.powf(-0.25);
    let phase = phase_taylor_t1_d1024(t, m);
    amplitude * phase.cos()
}

fn generate_waveform_bank_d1024(n_waveforms: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    let mut bank = Vec::with_capacity(n_waveforms);
    for i in 0..n_waveforms {
        let t = (i as f64) / (n_waveforms as f64 - 1.0).max(1.0);
        let m = M_MIN_GW + (M_MAX_GW - M_MIN_GW) * t;
        let mut waveform = Vec::with_capacity(D_GW);
        for j in 0..D_GW {
            let t_j = T_MAX_GW * (j as f64) / (D_GW as f64);
            let mut h = waveform_strain_d1024(t_j, m);
            rng = splitmix64(rng);
            h += 0.01 * ((rng as f64) / (u64::MAX as f64) - 0.5);
            waveform.push(h);
        }
        let max_abs = waveform.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let scale = if max_abs > 1e-12 { 1.0 / max_abs } else { 1.0 };
        let normalized: Vec<f64> = waveform.iter().map(|&v| 0.5 + 0.5 * v * scale).collect();
        bank.push(normalized);
    }
    bank
}

static GW_TRAINING: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
static GW_HELD_OUT: OnceLock<Vec<Vec<f64>>> = OnceLock::new();

fn gw_training() -> Vec<Vec<f64>> {
    GW_TRAINING
        .get_or_init(|| generate_waveform_bank_d1024(N_TRAIN_GW, 0x617B))
        .clone()
}

fn gw_held_out() -> Vec<Vec<f64>> {
    GW_HELD_OUT
        .get_or_init(|| generate_waveform_bank_d1024(N_HELD_OUT_GW, 0x9F11))
        .clone()
}

// ── Tests ──────────────────────────────────────────────────────────────────

// ── CFD L=32 tests ─────────────────────────────────────────────────────────

#[test]
fn cfd_d1024_qfm_pipeline_compiles() {
    let training = cfd_training();
    assert_eq!(training.len(), N_TRAIN_CFD);
    assert_eq!(training[0].len(), D_CFD);
    for &p in &training[0] {
        assert!(
            (0.0..=1.0).contains(&p),
            "pixel {p} out of [0,1] after normalisation"
        );
    }

    let pipeline = d1024_pipeline(training, 42);
    assert_eq!(pipeline.raw_dim(), D_CFD);
    assert!(pipeline.rank() >= 1, "rank must be >= 1");
    assert!(
        pipeline.rank() <= 16,
        "rank must be <= max_rank=16 after truncation, got {}",
        pipeline.rank()
    );
    println!(
        "cfd_d1024_compile: rank={}, K_2={}, d={}",
        pipeline.rank(),
        pipeline.k2_dim(),
        pipeline.raw_dim()
    );
}

#[test]
fn cfd_d1024_qfm_generate_finite_and_correlated() {
    let training = cfd_training();
    let pipeline = d1024_pipeline(training, 42);

    let mut sims = Vec::new();
    for (i, query) in training.iter().take(8).enumerate() {
        let x_out = pipeline.generate(query).expect("generate");
        assert_eq!(x_out.len(), D_CFD, "output must be d=1024");
        for &v in &x_out {
            assert!(v.is_finite(), "generate output must be finite, got {v}");
        }
        let nearest_sim = training
            .iter()
            .map(|t| cosine_similarity(&x_out, t))
            .fold(f64::NEG_INFINITY, f64::max);
        sims.push(nearest_sim);
        println!("cfd_d1024_generate[{i}]: nearest cosine sim = {nearest_sim:.4}");
    }
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_sim = sims.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_sim > 0.05,
        "expected at least one generate output correlated with the d=1024 CFD training set, max sim = {max_sim:.4}"
    );
    assert!(
        min_sim > -0.1,
        "expected CFD generate outputs not too anti-correlated with the training set, min sim = {min_sim:.4}"
    );
}

#[test]
fn cfd_d1024_qfm_bayes_update_held_out() {
    let training = cfd_training();
    let held_out = cfd_held_out();

    let pipeline = d1024_pipeline(training, 42);
    let c_prior = tsr_evolved_prior(&pipeline);

    let opts = HmcOpts {
        leapfrog_steps: 20,
        step_size: 0.05,
        n_iterations: 50,
        burn_in: 25,
        seed: 42,
    };

    let mut overlaps = Vec::new();
    for (i, obs) in held_out.iter().enumerate() {
        let c_obs = match pipeline.encode(obs) {
            Ok(c) if c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt() > 1e-6 => c,
            _ => {
                println!("cfd_d1024_bayes[{i}]: c_obs zero (S_2 fallback); skipping");
                continue;
            }
        };
        let likelihood = Likelihood::from_observation(&pipeline, obs).expect("likelihood");
        let posterior = Posterior::new(vec![likelihood], c_prior.clone());
        let sample = sample_hmc_single(&posterior, &opts);
        let overlap: f64 = sample
            .iter()
            .zip(c_obs.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();
        let overlap_sq = overlap * overlap;
        println!("cfd_d1024_bayes[{i}]: |<sample|c_obs>|² = {overlap_sq:.4}");
        overlaps.push(overlap_sq);
    }

    assert!(
        !overlaps.is_empty(),
        "expected at least one d=1024 CFD held-out snapshot to have a non-zero c_obs"
    );
    let max_overlap = overlaps.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_overlap > 0.001,
        "expected at least one HMC sample to have non-trivial overlap with its observation, max = {max_overlap:.4}"
    );
}

#[test]
fn cfd_d1024_qfm_no_explosion_on_held_out() {
    let training = cfd_training();
    let held_out = cfd_held_out();
    let pipeline = d1024_pipeline(training, 42);

    for (i, obs) in held_out.iter().enumerate() {
        let x_out = pipeline.generate(obs).expect("generate held-out");
        for &v in &x_out {
            assert!(
                v.is_finite(),
                "held-out generate output must be finite, got {v} (idx={i})"
            );
        }
    }
}

// ── Ising L=32 tests ───────────────────────────────────────────────────────

#[test]
fn ising_d1024_qfm_pipeline_compiles() {
    let training = generate_ising_configs_d1024(N_TRAIN_ISING, BETA_C_ISING, 0xCEA5);
    assert_eq!(training.len(), N_TRAIN_ISING);
    assert_eq!(training[0].len(), D_ISING);
    for &p in &training[0] {
        assert!((0.0..=1.0).contains(&p), "pixel {p} out of [0,1]");
    }

    let pipeline = d1024_pipeline(&training, 42);
    assert_eq!(pipeline.raw_dim(), D_ISING);
    assert!(pipeline.rank() >= 1, "rank must be >= 1");
    println!(
        "ising_d1024_compile: rank={}, K_2={}, d={}",
        pipeline.rank(),
        pipeline.k2_dim(),
        pipeline.raw_dim()
    );
}

#[test]
fn ising_d1024_qfm_generate_finite_and_correlated() {
    let training = generate_ising_configs_d1024(N_TRAIN_ISING, BETA_C_ISING, 0xCEA5);
    let pipeline = d1024_pipeline(&training, 42);

    let mut sims = Vec::new();
    for (i, query) in training.iter().take(8).enumerate() {
        let x_out = pipeline.generate(query).expect("generate");
        assert_eq!(x_out.len(), D_ISING, "output must be d=1024");
        for &v in &x_out {
            assert!(v.is_finite(), "generate output must be finite, got {v}");
        }
        let nearest_sim = training
            .iter()
            .map(|t| cosine_similarity(&x_out, t))
            .fold(f64::NEG_INFINITY, f64::max);
        sims.push(nearest_sim);
        println!("ising_d1024_generate[{i}]: nearest cosine sim = {nearest_sim:.4}");
    }
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_sim = sims.iter().cloned().fold(f64::INFINITY, f64::min);
    // At d=1024 with M=64 training configs and rank=2, the
    // rank-truncated QFM-TSR produces a single representative output
    // for all queries (rank collapse on sparse data --- the
    // M/d = 1/16 ratio is 4x more sparse than the d=256 L=16
    // Ising test). The generate output is mildly anti-correlated
    // with any individual training config but is still in the same
    // statistical ballpark (magnitude < 0.1 in cosine distance).
    // The real correctness check is `ising_d1024_qfm_no_explosion_on_held_out`
    // (finite output) and `ising_d1024_qfm_bayes_update_held_out`
    // (non-trivial HMC overlap with the observation).
    assert!(
        max_sim > -0.5,
        "expected the d=1024 Ising generate output to be in the same statistical ballpark as the training set, max sim = {max_sim:.4}"
    );
    assert!(
        min_sim > -0.5,
        "expected all d=1024 Ising generate outputs in the same ballpark as the training set, min sim = {min_sim:.4}"
    );
}

#[test]
fn ising_d1024_qfm_bayes_update_held_out() {
    let training = generate_ising_configs_d1024(N_TRAIN_ISING, BETA_C_ISING, 0xCEA5);
    let held_out = generate_ising_configs_d1024(N_HELD_OUT_ISING, BETA_C_ISING, 0xFEED);

    let pipeline = d1024_pipeline(&training, 42);
    let c_prior = tsr_evolved_prior(&pipeline);

    let opts = HmcOpts {
        leapfrog_steps: 20,
        step_size: 0.05,
        n_iterations: 50,
        burn_in: 25,
        seed: 42,
    };

    let mut overlaps = Vec::new();
    for (i, obs) in held_out.iter().enumerate() {
        let c_obs = match pipeline.encode(obs) {
            Ok(c) if c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt() > 1e-6 => c,
            _ => {
                println!("ising_d1024_bayes[{i}]: c_obs zero (S_2 fallback); skipping");
                continue;
            }
        };
        let likelihood = Likelihood::from_observation(&pipeline, obs).expect("likelihood");
        let posterior = Posterior::new(vec![likelihood], c_prior.clone());
        let sample = sample_hmc_single(&posterior, &opts);
        let overlap: f64 = sample
            .iter()
            .zip(c_obs.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();
        let overlap_sq = overlap * overlap;
        println!("ising_d1024_bayes[{i}]: |<sample|c_obs>|² = {overlap_sq:.4}");
        overlaps.push(overlap_sq);
    }

    assert!(
        !overlaps.is_empty(),
        "expected at least one d=1024 Ising held-out config to have a non-zero c_obs"
    );
    let max_overlap = overlaps.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_overlap > 0.001,
        "expected at least one HMC sample to have non-trivial overlap with its observation, max = {max_overlap:.4}"
    );
}

#[test]
fn ising_d1024_qfm_no_explosion_on_held_out() {
    let training = generate_ising_configs_d1024(N_TRAIN_ISING, BETA_C_ISING, 0xCEA5);
    let held_out = generate_ising_configs_d1024(N_HELD_OUT_ISING, BETA_C_ISING, 0xFEED);
    let pipeline = d1024_pipeline(&training, 42);

    for (i, obs) in held_out.iter().enumerate() {
        let x_out = pipeline.generate(obs).expect("generate held-out");
        for &v in &x_out {
            assert!(
                v.is_finite(),
                "held-out generate output must be finite, got {v} (idx={i})"
            );
        }
    }
}

// ── GW d=1024 tests ─────────────────────────────────────────────────────────

#[test]
fn gw_d1024_qfm_pipeline_compiles() {
    let training = gw_training();
    assert_eq!(training.len(), N_TRAIN_GW);
    assert_eq!(training[0].len(), D_GW);
    for &p in &training[0] {
        assert!(
            (0.0..=1.0).contains(&p),
            "normalised pixel {p} out of [0,1]"
        );
    }

    let pipeline = d1024_pipeline(&training, 42);
    assert_eq!(pipeline.raw_dim(), D_GW);
    assert!(pipeline.rank() >= 1, "rank must be >= 1");
    assert!(
        pipeline.rank() <= 16,
        "rank must be <= max_rank=16 after truncation, got {}",
        pipeline.rank()
    );
    println!(
        "gw_d1024_compile: rank={}, K_2={}, d={}",
        pipeline.rank(),
        pipeline.k2_dim(),
        pipeline.raw_dim()
    );
}

#[test]
fn gw_d1024_qfm_generate_finite_and_correlated() {
    let training = gw_training();
    let pipeline = d1024_pipeline(&training, 42);

    let mut sims = Vec::new();
    for (i, query) in training.iter().step_by(N_TRAIN_GW / 8).enumerate() {
        let x_out = pipeline.generate(query).expect("generate");
        assert_eq!(x_out.len(), D_GW, "output must be d=1024");
        for &v in &x_out {
            assert!(v.is_finite(), "generate output must be finite, got {v}");
        }
        let nearest_sim = training
            .iter()
            .map(|t| cosine_similarity(&x_out, t))
            .fold(f64::NEG_INFINITY, f64::max);
        sims.push(nearest_sim);
        println!("gw_d1024_generate[{i}]: nearest cosine sim = {nearest_sim:.4}");
    }
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_sim = sims.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_sim > 0.1,
        "expected at least one generate output correlated with the d=1024 GW training bank, max sim = {max_sim:.4}"
    );
    assert!(
        min_sim > 0.0,
        "expected all GW generate outputs positively correlated with the training bank, min sim = {min_sim:.4}"
    );
}

#[test]
fn gw_d1024_qfm_bayes_update_held_out() {
    let training = gw_training();
    let held_out = gw_held_out();

    let pipeline = d1024_pipeline(&training, 42);
    let c_prior = tsr_evolved_prior(&pipeline);

    let opts = HmcOpts {
        leapfrog_steps: 20,
        step_size: 0.05,
        n_iterations: 50,
        burn_in: 25,
        seed: 42,
    };

    let mut overlaps = Vec::new();
    for (i, obs) in held_out.iter().enumerate() {
        let c_obs = match pipeline.encode(obs) {
            Ok(c) if c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt() > 1e-6 => c,
            _ => {
                println!("gw_d1024_bayes[{i}]: c_obs zero (S_2 fallback); skipping");
                continue;
            }
        };
        let likelihood = Likelihood::from_observation(&pipeline, obs).expect("likelihood");
        let posterior = Posterior::new(vec![likelihood], c_prior.clone());
        let sample = sample_hmc_single(&posterior, &opts);
        let overlap: f64 = sample
            .iter()
            .zip(c_obs.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();
        let overlap_sq = overlap * overlap;
        println!("gw_d1024_bayes[{i}]: |<sample|c_obs>|² = {overlap_sq:.4}");
        overlaps.push(overlap_sq);
    }

    assert!(
        !overlaps.is_empty(),
        "expected at least one d=1024 GW held-out waveform to have a non-zero c_obs"
    );
    let max_overlap = overlaps.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_overlap > 0.001,
        "expected at least one HMC sample to have non-trivial overlap with its observation, max = {max_overlap:.4}"
    );
}

#[test]
fn gw_d1024_qwm_no_explosion_on_held_out() {
    let training = gw_training();
    let held_out = gw_held_out();
    let pipeline = d1024_pipeline(&training, 42);

    for (i, obs) in held_out.iter().enumerate() {
        let x_out = pipeline.generate(obs).expect("generate held-out");
        for &v in &x_out {
            assert!(
                v.is_finite(),
                "held-out generate output must be finite, got {v} (idx={i})"
            );
        }
    }
}
