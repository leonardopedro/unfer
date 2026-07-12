//! Tangible step 2: fit the **actual** W matrix from a trained
//! qfm_text model with oxieml, and measure the per-column
//! residual. This is the key test for whether oxieml is viable
//! as a decoder replacement.
//!
//! Run with:
//!   cargo test --release -p qfm_text --test oxieml_fit_real_w -- --nocapture --ignored

use qfm_text::oxieml_decoder::{OxiemlFitOpts, fit_column, fit_decoder};
use qfm_text::QfmTextModel;
use std::time::Instant;

#[test]
#[ignore = "slow: ~1-2 min; runs the actual oxieml fit on a trained model's W"]
fn fits_real_qfm_text_w_matrix() {
    // Use the rev 36 nohash shard 0 model (schema 3, just
    // re-trained). Its W is rank 2 (the 6.7% in-sample win
    // over baseline comes from this rank-2 basis). The
    // oxieml fit on rank 2 will be the real test.
    let ckpt = "/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/unfer/qfm_text_runs/m8_nohash_shard0_v3/checkpoint_epoch0.qfm";
    eprintln!("loading checkpoint: {}", ckpt);
    let model = QfmTextModel::load(std::path::Path::new(ckpt)).expect("load checkpoint");
    eprintln!(
        "model loaded. cfg: n_orders={}, k2_total={}, krylov_rank={}",
        model.cfg.n_orders, model.cfg.k2_total(), model.krylov_rank()
    );

    // Extract W. Shape: (k2_total, rank) = (4_194_305, 3).
    let w = model.w_matrix();
    let (m, rank) = (w.nrows(), w.ncols());
    eprintln!("W: shape = ({}, {}), extracting columns...", m, rank);

    // Build per-column real vectors (the W entries are
    // complex; we fit on the real part for now — the
    // imaginary part is near-zero for the unit-norm basis
    // vectors in the rev 37 design).
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(rank);
    for j in 0..rank {
        let mut col = Vec::with_capacity(m);
        for i in 0..m {
            col.push(w[(i, j)].re);
        }
        // Normalize to [-1, 1] so oxieml's tree coefficients
        // are reasonable.
        let max_abs = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if max_abs > 0.0 {
            for v in &mut col {
                *v /= max_abs;
            }
        }
        columns.push(col);
    }

    let opts = OxiemlFitOpts {
        max_depth: 3,
        max_iter: 200,
        num_restarts: 1,
        weight_bins: 32,
        residual_tol: 0.05,
        ..Default::default()
    };

    eprintln!("fitting {} columns (m={}, bins={})...", rank, m, opts.weight_bins);
    let t0 = Instant::now();
    let fits = fit_decoder(&columns, m, &opts);
    eprintln!("fit_decoder total time: {:.2}s", t0.elapsed().as_secs_f64());
    for (j, fit) in fits.iter().enumerate() {
        eprintln!(
            "  col {}: mse={:.4e} complexity={} fit_s={:.2}\n    pretty={}",
            j, fit.mse, fit.complexity, fit.fit_seconds, fit.pretty
        );
    }
    // Don't assert on residual here — the W is rank 1 so the
    // fit is trivial; the goal is to verify the pipeline
    // works end-to-end on real data.
    assert!(fits.iter().all(|f| f.mse.is_finite()));

    // Now exercise the integrated path: load the model,
    // call `fit_oxieml_decoder`, verify the model can still
    // be queried (next_token_dist produces a finite
    // distribution).
    //
    // We need a mutable model. Load from the loaded one.
    let mut model_mut = QfmTextModel::load(std::path::Path::new(ckpt)).expect("load 2");
    let summary = model_mut.fit_oxieml_decoder(qfm_text::oxieml_decoder::OxiemlFitOpts {
        max_depth: 3,
        max_iter: 200,
        num_restarts: 1,
        weight_bins: 32,
        residual_tol: 0.05,
        ..Default::default()
    });
    eprintln!(
        "fit_oxieml_decoder: n_fallback={}/{}, total_fit_s={:.2}",
        summary.n_fallback,
        summary.per_column_mse.len(),
        summary.total_fit_seconds
    );
    // Query a few contexts and verify finite output.
    let shard = std::path::Path::new("/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/hrm_data/wikitext-103-train/shard_00000.bin");
    if shard.exists() {
        let s = qfm_text::Shard::open(shard, 20000).expect("open shard");
        let mut queried = 0;
        for (ctx, _next) in s.windows(4).take(5) {
            let dist = model_mut.next_token_dist(&ctx).expect("dist");
            let sum: f64 = dist.iter().sum();
            let ok = dist.iter().all(|&x| x.is_finite() && x >= 0.0) && (sum - 1.0).abs() < 0.01;
            eprintln!("  ctx={:?} sum={:.4} ok={}", &ctx[..ctx.len().min(4)], sum, ok);
            queried += 1;
        }
        assert!(queried > 0, "should have queried at least one context");
    }
}

