//! Diagnostic for the L^1 outer vacuum (rev 37 v3).
//!
//! Verifies:
//! 1. The L^1 inner product of c_0 with every W[m, :] row is
//!    approximately constant (= 1/rank in the L^1 definition).
//! 2. The Krylov evolution c_1 = U c_0 propagates this uniform
//!    state, but the per-mode Born weights
//!    |c_1^H W[m, :]|² are not uniform (the asymmetry of W breaks
//!    the post-evolution symmetry).
//! 3. The per-mode weight diagnostic on a test context shows
//!    how the per-mode masses are distributed.

use qfm_text::QfmTextModel;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let ckpt = PathBuf::from(
        "/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/unfer/qfm_text_runs/m8_l1vac_shard0/checkpoint_epoch0.qfm",
    );
    let model = QfmTextModel::load(&ckpt)?;
    let n_active = model.registry.maps().iter().map(|m| m.len()).sum::<usize>();

    println!("=== L^1 OUTER VACUUM DIAGNOSTIC ===");
    println!("outer_vacuum (c_0) = {:?}", model.outer_vacuum);
    let n = model.outer_vacuum.norm();
    println!("||c_0|| = {:.6e}", n);
    println!("rank = {}", model.krylov_rank());
    println!("k2_total = {}", model.k2_total());
    println!("n_active = {}", n_active);

    // 1. Verify the L^1 inner product uniformity
    println!("\n=== L^1 inner product uniformity check ===");
    let w = model.w_matrix();
    let rank = model.krylov_rank();
    let m = w.nrows();
    let mut inner_products = Vec::with_capacity(m);
    for i in 0..m {
        let mut l1 = 0.0;
        for r in 0..rank {
            l1 += w[(i, r)].norm() * model.outer_vacuum[r].norm();
        }
        inner_products.push(l1);
    }
    let n_samples = 10.min(m);
    println!("First {} L^1 inner products <c_0 | W[i,:]>_L1:", n_samples);
    for i in 0..n_samples {
        println!("  W[{},:]: L^1 inner product = {:.6e}", i, inner_products[i]);
    }
    let mean: f64 = inner_products.iter().sum::<f64>() / m as f64;
    let std: f64 = (inner_products.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / m as f64)
        .sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };
    println!("\nL^1 inner product statistics over all {} modes:", m);
    println!("  mean = {:.6e}", mean);
    println!("  std  = {:.6e}", std);
    println!("  CV   = {:.4} (coefficient of variation; lower = more uniform)", cv);

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
    println!(
        "L1 deviation from uniform = {:.4} (lower = closer to uniform)",
        l1_dev
    );

    // 3. The L^1 inner product with W[0, :] (the Fock vacuum)
    let mut l1_vacuum = 0.0;
    for r in 0..rank {
        l1_vacuum += w[(0, r)].norm() * model.outer_vacuum[r].norm();
    }
    println!(
        "\nL^1 inner product <c_0 | W[0,:]>_L1 (Fock vacuum) = {:.6e}",
        l1_vacuum
    );
    println!(
        "L^1 inner product <c_0 | W[1,:]>_L1 (first data mode) = {:.6e}",
        inner_products[1]
    );

    Ok(())
}
