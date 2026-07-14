//! Diagnostic for the outer vacuum (rev 37 v3).
//!
//! The outer vacuum is the uniform state on the R-dimensional
//! Krylov basis (the Fock-space discretization at resolution R):
//!   c_0 = (1/√R) · 1
//!
//! where R = rank is the Krylov rank (small, e.g. 2 on shard 0).
//! M (the number of training data points) is *not* the Fock
//! space dimension; the Fock space is infinite-dim, M is the
//! number of inner wavefunctions used to compute the inner
//! products.
//!
//! Verifies:
//! 1. The L² inner product of c_0 with W[m, :] has magnitude
//!    (1/√R) ||W[m, :]||_L1 (for real non-negative W), and is
//!    complex in general for our complex W.
//! 2. The Krylov evolution c_1 = U c_0 propagates this state;
//!    the per-mode Born weights |c_1^H W[m, :]|² are not
//!    uniform (the asymmetry of W breaks the post-evolution
//!    symmetry).

use qfm_text::QfmTextModel;
use num_complex::Complex64;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let ckpt = PathBuf::from(
        "/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/unfer/qfm_text_runs/m8_uniform_outervac_shard0/checkpoint_epoch0.qfm",
    );
    let model = QfmTextModel::load(&ckpt)?;

    println!("=== OUTER VACUUM DIAGNOSTIC (rev 37 v3) ===");
    println!("outer_vacuum (c_0) = {:?}", model.outer_vacuum);
    let n = model.outer_vacuum.norm();
    println!("||c_0|| = {:.6e}", n);
    let rank = model.krylov_rank();
    println!("rank = {}", rank);
    println!("k2_total = {}", model.k2_total());
    let n_active = model.registry.maps().iter().map(|m| m.len()).sum::<usize>();
    println!("n_active = {}", n_active);

    let w = model.w_matrix();
    let m = w.nrows();

    // 1. L² inner product vs L¹ norm: for real non-negative W,
    //    ⟨c_0 | W[m,:]⟩_L2 = (1/√R) · ||W[m,:]||_L1. For complex
    //    W, the L² inner product is complex in general.
    println!("\n=== L² inner product vs L¹ norm (first 10 modes) ===");
    println!("m | L² inner (re) | L² inner (im) | |L² inner| | L¹ norm | |re|/L¹ |");
    let n_samples = 10.min(m);
    for i in 0..n_samples {
        let mut inner = Complex64::new(0.0, 0.0);
        let mut l1 = 0.0;
        for r in 0..rank {
            inner += model.outer_vacuum[r] * w[(i, r)];
            l1 += w[(i, r)].norm();
        }
        println!(
            "{} | {:+.4e} | {:+.4e} | {:.4e} | {:.4e} | {:.4e}",
            i, inner.re, inner.im, inner.norm(), l1, inner.norm() / l1
        );
    }
    // Aggregate
    let mut sum_l2_inner_abs = 0.0;
    let mut sum_l1 = 0.0;
    for i in 0..m {
        let mut inner = Complex64::new(0.0, 0.0);
        let mut l1 = 0.0;
        for r in 0..rank {
            inner += model.outer_vacuum[r] * w[(i, r)];
            l1 += w[(i, r)].norm();
        }
        sum_l2_inner_abs += inner.norm();
        sum_l1 += l1;
    }
    let k_eff = sum_l2_inner_abs / sum_l1;
    println!("\n  mean |⟨c_0 | W⟩_L2| = {:.4e}", sum_l2_inner_abs / m as f64);
    println!("  mean ||W[m,:]||_L1  = {:.4e}", sum_l1 / m as f64);
    println!("  effective k = mean |⟨c_0 | W⟩_L2| / mean L¹ norm = {:.4e}", k_eff);
    println!("  (design: k = 1/√R for real non-negative W; for rank {}: k = {:.4e})",
        rank, 1.0 / (rank as f64).sqrt());

    // 2. Per-mode weights on a test context
    println!("\n=== Per-mode Born weights on a test context ===");
    let context = vec![155u32, 487, 172, 155];
    let c_1 = {
        let c_0 = model.outer_vacuum.clone();
        model.pipeline.evolve(&c_0, model.cfg.t)
    };
    let mut active = model.registry.encode_modes(&context);
    if !active.contains(&0) {
        active.push(0);
    }
    let weights = model.pipeline.decode_sketched_at(&c_1, &model.gram, &active);
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    let mut sorted: Vec<_> = weights.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("active_modes = {:?}", active);
    println!("per-mode weights (raw Born):");
    for (m, w) in sorted.iter().take(20) {
        println!("  mode {}: weight = {:.6e}, fraction = {:.4}", m, w, w / total);
    }
    println!("total weight = {}", total);
    println!("n_active = {}", active.len());
    let uniform = 1.0 / active.len() as f64;
    let l1_dev: f64 = weights
        .iter()
        .map(|(_, w)| (w / total - uniform).abs())
        .sum::<f64>() / 2.0;
    println!("L1 deviation from uniform = {:.4} (lower = closer to uniform)", l1_dev);

    // 3. Outer vacuum norms (sanity check)
    let l2_norm_sq: f64 = model.outer_vacuum.iter().map(|c| c.norm_sqr()).sum();
    let l1_norm: f64 = model.outer_vacuum.iter().map(|c| c.re.abs() + c.im.abs()).sum();
    println!("\n=== Outer vacuum norms ===");
    println!("  L² norm = {:.6e} (1.0 for unit vector)", l2_norm_sq.sqrt());
    println!("  L¹ norm = {:.6e} (uniform c_0 has L¹ norm = √R = {})", l1_norm, (rank as f64).sqrt());
    println!("  Per-row L² inner product = (1/√R) · ||W[m,:]||_L1 (for real non-negative W)");

    Ok(())
}
