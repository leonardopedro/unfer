//! Diagnostic: print the per-mode Krylov weights for sample contexts
//! to see how much mass the outer vacuum (mode 0) gets vs the
//! per-order modes.
use qfm_text::model::QfmTextModel;
use nalgebra::DVector;
use num_complex::Complex64;

fn inner(a: &DVector<Complex64>, b: &DVector<Complex64>) -> Complex64 {
    a.dotc(b)
}

#[test]
#[ignore]
fn print_krylov_weights() {
    let model = QfmTextModel::load(
        "/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/unfer/qfm_text_runs/m8_nohash_shard0_v3/checkpoint_epoch0.qfm",
    )
    .expect("load");
    println!("=== Krylov weight for outer vacuum (mode 0) vs per-order modes ===");
    let w = model.pipeline.w();
    let rank = w.ncols();
    // Build the W[0, :] row as a DVector.
    let w0 = DVector::from_vec((0..rank).map(|j| w[(0, j)]).collect::<Vec<_>>());
    println!("W[0, :] = {:?}", w0.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());
    let test_contexts: &[&[u32]] = &[
        &[0, 1],
        &[100, 200],
        &[1000, 2000],
        &[5000, 10000],
    ];
    for ctx in test_contexts {
        // Reproduce encode_context: c_0 = (1/√(n+1))(W[0, :] + Σ_o W[m_o, :])
        let n = model.cfg.n_orders.min(ctx.len());
        if n == 0 { continue; }
        let scale = 1.0 / ((n + 1) as f64).sqrt();
        let mut c0 = w0.scale(scale);
        for o in 1..=n {
            if let Some(m) = model.registry.lookup(o, ctx) {
                let row = m as usize;
                if row < w.nrows() {
                    for r in 0..rank {
                        c0[r] += w[(row, r)] * scale;
                    }
                }
            }
        }
        let c1 = model.pipeline.evolve(&c0, model.cfg.t);
        // Compute per-mode weight for mode 0 (the outer vacuum) and
        // for the per-order modes. This is what the model uses
        // internally via the public `next_token_dist` API.
        let mut weights = Vec::new();
        let mut total = 0.0;
        // Mode 0 (outer vacuum)
        {
            let amp = inner(&c1, &w0);
            let pw = amp.norm_sqr();
            weights.push((0u32, pw));
            total += pw;
        }
        for o in 1..=n {
            if let Some(m) = model.registry.lookup(o, ctx) {
                let row = m as usize;
                if row < w.nrows() {
                    let wrow = DVector::from_vec((0..rank).map(|j| w[(row, j)]).collect::<Vec<_>>());
                    let amp = inner(&c1, &wrow);
                    let pw = amp.norm_sqr();
                    weights.push((m, pw));
                    total += pw;
                }
            }
        }
        println!("\nContext {:?} (n_orders = {}):", ctx, n);
        println!("  c_0 = {:?}", c0.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());
        println!("  c_1 = {:?}", c1.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());
        println!("  per-mode weights (sum = {total:.6e}):");
        for (m, pw) in &weights {
            let frac = if total > 0.0 { pw / total } else { 0.0 };
            let o = if *m == 0 { "outer vacuum".to_string() } else { format!("order {}", model.registry.order_of(*m)) };
            println!("    mode {m} ({o}): weight = {pw:.6e}, fraction = {frac:.6}");
        }
    }
}
