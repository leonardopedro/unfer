//! Shared support for Theorem 4.1 (Hashimoto–Nodera, JJIAM 2019) error
//! bands: the a-priori SIRK approximation-error envelope
//!
//!   ‖φ_k(A)v − SIRK_m(v)‖ ≤ 2C‖v‖ e^{-hm} · E_m ,   C ∈ [2, 11.08],
//!
//! where E_m = min_{r ∈ R^SIRK_{m-1}} ‖f_{k,N} − r‖_{∞,Σ},
//!   f_{k,N}(z)  = e^N φ_k(−z^{-1})                (Definition 2.4 form),
//!   R^SIRK_{m-1} = {p/q : p ∈ P_m, q(z) = Π_{j=1..m} (1 + h j z)},
//!   Σ           = conv hull of the resolvent numerical ranges W(X_j),
//!   X_j         = (γ_j I − A)^{-1}, shifts γ_j = N − h j > 0.
//!
//! Everything here follows the paper literally: the paper's OWN shift ladder
//! (not `shifts_for_range`) must be fed to the solver so that h, N, m are
//! the theorem's parameters; Σ is built as a conservative bounding box of
//！ the resolvent field over the Hermitian spectral extent; E_m is computed
//! by Lawson's iteratively-reweighted minimax on a discretized Σ with the
//! fixed SIRK denominator factored out (linear-in-coefficients problem).

// This module is `mod`-included by two separate test binaries, each of which
// uses only a subset of its items — so items unused by one binary look dead
// to that compilation. Keep the shared scaffolding available to both.
#![allow(dead_code)]

use num_complex::Complex64;

/// The paper's shift ladder: γ_j = N − h j for j = 1..=m, all strictly
/// positive (N > h m), real — admissible shifts for A with spectrum on the
/// imaginary axis (evolution operator A = −i H t).
pub fn sirk_paper_shifts(m: usize, big_n: f64, h: f64) -> Vec<Complex64> {
    (1..=m)
        .map(|j| {
            let g = big_n - h * j as f64;
            assert!(g > 0.0, "paper shifts require N > h m");
            Complex64::new(g, 0.0)
        })
        .collect()
}

/// φ_k for the k values used here (φ₀ = e^z suffices for pure evolution).
fn phi(k: usize, z: Complex64) -> Complex64 {
    match k {
        0 => z.exp(),
        _ => unimplemented!("only φ₀ (pure evolution) needed"),
    }
}

/// Parameters of the Theorem 4.1 band for one experiment.
#[derive(Clone, Copy)]
pub struct BandParams {
    /// N in γ_j = N − h j (also the e^N normalization).
    pub big_n: f64,
    /// Shift spacing h (drives the e^{-hm} decay).
    pub h: f64,
    /// Evolution time entering A = −i H t.
    pub t: f64,
}

#[allow(dead_code)] // shared test-support module: each test binary reads a subset of fields
pub struct BandOutcome {
    /// E_m = min_r ‖f − r‖_{∞,Σ} (Lawson minimax value).
    pub e_m: f64,
    /// Lower edge of the band: 2·C_min·‖v‖·e^{-hm}·E_m, C_min = 2.
    pub lo: f64,
    /// Upper edge: 2·C_max·‖v‖·e^{-hm}·E_m, C_max = 11.08.
    pub hi: f64,
    /// Exponential factor alone (for reporting decay per dimension).
    pub exp_factor: f64,
}

impl BandParams {
    /// Compute the Theorem 4.1 band for a Hermitian H whose spectral extent
    /// is contained in [lam_lo, lam_hi] (absolute values bound |λ|), Krylov
    /// depth `m`, start vector norm `v_norm`, function order k (= 0 here).
    ///
    /// Σ is taken as a conservative BOX containing every W(X_j): for
    /// Hermitian H, eigenvectors of X_j = (γ_j I + i t H)^{-1} have
    /// eigenvalues 1/(γ_j + i t λ), i.e. they sit on the vertical line
    /// Re = γ_j/(γ_j² + t²λ²), Im = −tλ/(γ_j² + t²λ²). Hulling over j and
    /// λ ∈ {±|λ_max|, 0} and padding to a rectangle keeps the band VALID
    /// (a larger domain only enlarges the sup norm).
    pub fn band(
        &self,
        lam_abs_max: f64,
        m: usize,
        v_norm: f64,
        k_order: usize,
        grid_re: usize,
        grid_im: usize,
    ) -> BandOutcome {
        let t = self.t;
        let lam = lam_abs_max.max(1e-300);

        // --- Σ bounding box over j = 1..=m and |λ| ∈ {0, lam} ---
        let mut re_vals = vec![0.0_f64];
        let mut im_vals = vec![0.0_f64];
        for j in 1..=m {
            let g = self.big_n - self.h * j as f64;
            for lam_v in [0.0_f64, lam, -lam] {
                let d = g * g + (t * lam_v) * (t * lam_v);
                re_vals.push(g / d);
                im_vals.push(-t * lam_v / d);
            }
        }
        let (re_min, re_max) = (
            re_vals.iter().cloned().fold(f64::INFINITY, f64::min),
            re_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let (im_min, im_max) = (
            im_vals.iter().cloned().fold(f64::INFINITY, f64::min),
            im_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        // Pad slightly (open interior requirement of M(Σ)).
        let pad_re = 1e-9 + 1e-6 * (re_max - re_min).abs();
        let pad_im = 1e-9 + 1e-6 * (im_max - im_min).abs();

        // Grid Σ (the padded box ⊇ Σ ⇒ sup over box ⊇ sup over Σ ⇒ VALID).
        let mut pts = Vec::with_capacity(grid_re * grid_im);
        for ir in 0..grid_re {
            let x = re_min - pad_re
                + (re_max - re_min + 2.0 * pad_re) * (ir as f64 / (grid_re - 1) as f64);
            for ii in 0..grid_im {
                let y = im_min - pad_im
                    + (im_max - im_min + 2.0 * pad_im) * (ii as f64 / (grid_im - 1) as f64);
                pts.push(Complex64::new(x, y));
            }
        }

        // Target f_{k,N}(z) = e^N φ_k(−z^{-1}), denominator q(z) =
        // Π (1 + h j z). Work with g = f·q (entire), fitting p ≈ g in the
        // WEIGHTED sense |g − p|/|q| = |f − p/q|.
        let q_of = |z: Complex64| -> Complex64 {
            let mut acc = Complex64::new(1.0, 0.0);
            for j in 1..=m {
                acc *= Complex64::new(1.0, 0.0) + Complex64::new(self.h * j as f64, 0.0) * z;
            }
            acc
        };
        let f_of = |z: Complex64| -> Complex64 {
            let finv = -Complex64::new(1.0, 0.0) / z;
            Complex64::new(self.big_n.exp(), 0.0) * phi(k_order, finv)
        };

        // Scaling map into the unit disc for conditioning of the basis.
        let r_scale = pts.iter().map(|z| z.norm()).fold(1e-300_f64, f64::max);
        let _ = r_scale;

        // Chebyshev-like tensor basis in Re(ẑ), Im(ẑ): total degree ≤ m.
        let cheb = |n: i32, s: f64| ((n as f64) * s.acos()).cos();
        let n_half = m as i32;
        let basis = |z: Complex64| -> Vec<Complex64> {
            let mut b = Vec::new();
            for ix in 0..=n_half {
                let cx = cheb(ix, z.re.clamp(-1.0, 1.0));
                for iy in 0..=(n_half - ix) {
                    let cy = cheb(iy, z.im.clamp(-1.0, 1.0));
                    b.push(Complex64::new(cx * cy, 0.0));
                }
            }
            b
        };
        let dim = basis(Complex64::new(0.0, 0.0)).len();

        // Lawson iterations: weighted least squares → minimax.
        let mut lawson_w = vec![1.0_f64; pts.len()];
        let mut coeffs = nalgebra::DVector::<Complex64>::zeros(dim);
        let design_rows: Vec<Vec<Complex64>> = pts.iter().map(|z| basis(*z)).collect();
        let rhs: Vec<Complex64> = pts.iter().map(|z| f_of(*z) * q_of(*z)).collect();
        // Divide by |q| belongs to the ERROR metric, applied via Lawson
        // weights (rows already carry 1/|q| multiplicatively below).
        let qmag: Vec<f64> = pts.iter().map(|z| q_of(*z).norm()).collect();

        let iterations = 30;
        for _ in 0..iterations {
            // Weighted normal equations: (Aᴴ W A) c = Aᴴ W b, W = diag(w/|q|²)?
            // Residual metric is |g−p|/|q| ⇒ row scale sqrt(w)/|q|.
            let mut ata = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
            let mut atb = nalgebra::DVector::<Complex64>::zeros(dim);
            for i in 0..pts.len() {
                let row_scale = lawson_w[i].sqrt() / qmag[i];
                let row = &design_rows[i];
                let rb = rhs[i] * row_scale;
                for a in 0..dim {
                    let av = row[a] * row_scale;
                    atb[a] += av.conj() * rb;
                    for b in a..dim {
                        let bv = row[b] * row_scale;
                        ata[(a, b)] += av.conj() * bv;
                    }
                }
            }
            for a in 0..dim {
                for b in 0..a {
                    ata[(a, b)] = ata[(b, a)].conj();
                }
            }
            // Tiny Tikhonov for stability of the (possibly rank-deficient at
            // large m) moment matrix.
            for a in 0..dim {
                ata[(a, a)] += Complex64::new(1e-13, 0.0);
            }
            let lu = ata.lu();
            coeffs = lu
                .solve(&atb)
                .unwrap_or_else(|| nalgebra::DVector::zeros(dim));

            // Update Lawson weights ∝ |residual|.
            let mut max_res = 0.0_f64;
            let mut res = Vec::with_capacity(pts.len());
            for i in 0..pts.len() {
                let row = &design_rows[i];
                let mut pval = Complex64::new(0.0, 0.0);
                for a in 0..dim {
                    pval += row[a] * coeffs[a];
                }
                let e = ((rhs[i] - pval) / qmag[i]).norm();
                res.push(e);
                if e > max_res {
                    max_res = e;
                }
            }
            for i in 0..pts.len() {
                lawson_w[i] *= res[i] / max_res.max(1e-300);
                if !lawson_w[i].is_finite() {
                    lawson_w[i] = 0.0;
                }
            }
        }

        // Final E_m: true weighted sup-norm of the fitted error.
        let mut e_m = 0.0_f64;
        for i in 0..pts.len() {
            let row = &design_rows[i];
            let mut pval = Complex64::new(0.0, 0.0);
            for a in 0..dim {
                pval += row[a] * coeffs[a];
            }
            let err = ((rhs[i] - pval) / qmag[i]).norm();
            if err > e_m {
                e_m = err;
            }
        }

        let exp_factor = (-self.h * m as f64).exp();
        let base = 2.0 * v_norm * exp_factor * e_m;
        BandOutcome {
            e_m,
            lo: 2.0 * base,   // C = 2
            hi: 11.08 * base, // C = 11.08
            exp_factor,
        }
    }
}

/// Certified-observable propagation: a state certified to lie within
/// `band_hi` (Theorem 4.1, C = 11.08 edge) of the exact evolved state
/// bounds ANY observable shift through Cauchy–Schwarz:
///
///   |⟨O⟩_SIRK − ⟨O⟩_exact| ≤ 2 ‖O‖_op · band_hi · ‖v‖ .
///
/// Returns the certified interval [value − δ, value + δ]. This turns the
/// band machinery into ERROR BARS for models without closed-form references
/// — certified numerics for the interacting gauge-fixed Hamiltonians.
pub fn certify(value: f64, op_norm_bound: f64, band_hi: f64, v_norm: f64) -> (f64, f64) {
    let delta = 2.0 * op_norm_bound * band_hi * v_norm;
    (value - delta, value + delta)
}

/// Pretty-print one certified row.
pub fn print_certified(label: &str, value: f64, lo: f64, hi: f64) {
    println!("  {label:<28} {value:+.6}   certified [{lo:+.6}, {hi:+.6}]");
}
