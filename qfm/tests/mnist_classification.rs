use nalgebra::{Complex, DVector};
use qfm::pipeline::{HamiltonianType, QfmConfig, QfmPipeline, SparseKernel};

type C64 = Complex<f64>;

#[derive(serde::Deserialize)]
struct Sample { label: u32, pixels: Vec<f64> }

fn load(path: &str) -> Vec<(u32, Vec<f64>)> {
    let d = std::fs::read_to_string(path).unwrap();
    let s: Vec<Sample> = serde_json::from_str(&d).unwrap();
    s.into_iter().map(|s| (s.label, s.pixels)).collect()
}

fn binarize(pixels: &[f64], thr: f64) -> u64 {
    let mut b = 0u64;
    for (i, &p) in pixels.iter().enumerate() { if p > thr { b |= 1u64 << i; } }
    b
}

/// Nearest-neighbor kernel: each image links to its max_nn nearest neighbors (by RBF
/// on binarized pixels). Self K(i,i)=1. Cross-weights are L1-normalized per row so
/// Σ_{j≠i} K(i,j)=1. With γ=1 in the pipeline, H₁ = λ₁·K(i,j)·(|i,f_i⟩⟨j,0|+h.c.)
/// gives self-fraction = 1/2.
fn nearest_kernel(bv: &[u64], sigma: f64) -> SparseKernel {
    let n = bv.len(); let i2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut rows = vec![Vec::new(); n];
    for i in 0..n {
        rows[i].push((i, 1.0));
        for j in 0..n {
            if i == j { continue; }
            let d = (bv[i] ^ bv[j]).count_ones() as usize;
            if d > 64 { continue; }
            let t = (1.0 - 2.0 * d as f64 / 64.0).clamp(-1.0, 1.0).acos();
            let v = (-t * t * i2s2).exp();
            if v > 1e-6 { rows[i].push((j, v)); }
        }
    }
    SparseKernel { n, rows }
}

/// Run a single (σ, m, λ₀, time) configuration.
/// Each image is one mode (k=1). The kernel-weighted Hamiltonian uses γ=1.
fn run(train: &[(u32, Vec<f64>)], test: &[(u32, Vec<f64>)], sigma: f64, m: usize, l0: f64, l1: f64, time: f64,
       htype: HamiltonianType) {
    let all: Vec<&(u32, Vec<f64>)> = train.iter().chain(test.iter()).collect();
    let nt = train.len(); let nx = test.len(); let nm = all.len();
    let bv: Vec<u64> = all.iter().map(|s| binarize(&s.1, 0.5)).collect();
    let im: Vec<u32> = (0..nm as u32).collect();
    let om: Vec<u32> = (0..5).collect();
    let mut tr = Vec::with_capacity(nt);
    for i in 0..nt { tr.push((i as u32, train[i].0)); }
    let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None, random_start: false,
        hamiltonian_type: htype, ..Default::default() };
    let kernel = nearest_kernel(&bv, sigma);
    let pipe = QfmPipeline::compile_channels_with_kernel(&im, &om, &tr, l0, l1, nm, &cfg, 0.0, 0.0, true, None, None, Some(kernel), 1.0).unwrap();
    let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
    let u = (h.clone() * (-C64::new(0.0, 1.0) * time)).exp();
    let no = 5;

    let classify = |img_idx: usize, true_label: u32| -> bool {
        let mut c0 = DVector::zeros(rk);
        for f in 0..no { let tp = img_idx * no + f; for r in 0..rk { c0[r] += w[(tp, r)] * C64::new(1.0 / (no as f64).sqrt(), 0.0); } }
        let c1 = &u * c0;
        let mut pr = vec![0.0_f64; no];
        for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
        let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
        let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i as u32).unwrap_or(0);
        pd == true_label
    };

    let mut tc = 0u32; for ti in 0..nt { if classify(ti, train[ti].0) { tc += 1; } }
    let mut xc = 0u32; for ti in 0..nx { if classify(nt + ti, test[ti].0) { xc += 1; } }
    let htag = match htype { HamiltonianType::Diffusion => "diff", HamiltonianType::PauliGrover => "pg" };
    eprintln!("    {htag} σ={sigma:.1} m={m:>3} λ₀={l0:.1} λ₁={l1:.1} t={time:.2} rank={rk}  train {tc}/{nt} ({:.1}%)  test {xc}/{nx} ({:.1}%)",
        tc as f64 / nt as f64 * 100.0, xc as f64 / nx as f64 * 100.0);
}

#[test]
fn mnist_classification_tests() {
    let samples = load("testdata/mnist_8x8_65training.json");
    let mut by_label: Vec<Vec<usize>> = (0..5).map(|_| Vec::new()).collect();
    for (i, s) in samples.iter().enumerate() { by_label[s.0 as usize].push(i); }
    let mut train_idx = Vec::new(); let mut test_idx = Vec::new();
    for lbl in 0..5 {
        for (j, &idx) in by_label[lbl].iter().enumerate() {
            if j < 8 { train_idx.push(idx); } else { test_idx.push(idx); }
        }
    }
    let train: Vec<(u32, Vec<f64>)> = train_idx.iter().map(|&i| (samples[i].0, samples[i].1.clone())).collect();
    let test: Vec<(u32, Vec<f64>)> = test_idx.iter().map(|&i| (samples[i].0, samples[i].1.clone())).collect();
    eprintln!("=== MNIST 8x8 (resplit: {}/{} train/test) ===", train.len(), test.len());
    eprintln!("  labels per class: train 8, test 5");

    eprintln!("\n  --- no-kernel baselines ---");
    let nm = train.len() + test.len();
    let im: Vec<u32> = (0..nm as u32).collect(); let om: Vec<u32> = (0..5).collect();
    let mut tr = Vec::with_capacity(train.len());
    for i in 0..train.len() { tr.push((i as u32, train[i].0)); }
    for &m in &[5, 10, 15, 20] {
        let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None, random_start: false, ..Default::default() };
        let pipe = QfmPipeline::compile_channels(&im, &om, &tr, 1.0, 1.0, nm, &cfg, 0.0, 0.0, true, None, None).unwrap();
        let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
        let u = (h.clone() * (-C64::new(0.0, 1.0) * 1.0)).exp(); let no = 5;
        let classify = |img_idx: usize, tl: u32| -> bool {
            let mut c0 = DVector::zeros(rk);
            for f in 0..no { let tp = img_idx * no + f; for r in 0..rk { c0[r] += w[(tp, r)] * C64::new(1.0 / (no as f64).sqrt(), 0.0); } }
            let c1 = &u * c0; let mut pr = vec![0.0_f64; no];
            for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
            let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
            let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap(); pd == tl
        };
        let mut tc = 0u32; for ti in 0..train.len() { if classify(ti, train[ti].0) { tc += 1; } }
        let mut xc = 0u32; for ti in 0..test.len() { if classify(train.len() + ti, test[ti].0) { xc += 1; } }
        eprintln!("    vacuum start  m={m:>3} rank={rk}  train {tc}/{} ({:.1}%)  test {xc}/{} ({:.1}%)",
            train.len(), tc as f64 / train.len() as f64 * 100.0, test.len(), xc as f64 / test.len() as f64 * 100.0);
    }

    eprintln!("\n  --- no-kernel, random start ---");
    for &m in &[5, 10, 15, 20] {
        let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None, random_start: true, ..Default::default() };
        let pipe = QfmPipeline::compile_channels(&im, &om, &tr, 1.0, 1.0, nm, &cfg, 0.0, 0.0, true, None, None).unwrap();
        let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
        let u = (h.clone() * (-C64::new(0.0, 1.0) * 1.0)).exp(); let no = 5;
        let classify = |img_idx: usize, tl: u32| -> bool {
            let mut c0 = DVector::zeros(rk);
            for f in 0..no { let tp = img_idx * no + f; for r in 0..rk { c0[r] += w[(tp, r)] * C64::new(1.0 / (no as f64).sqrt(), 0.0); } }
            let c1 = &u * c0; let mut pr = vec![0.0_f64; no];
            for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
            let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
            let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap(); pd == tl
        };
        let mut tc = 0u32; for ti in 0..train.len() { if classify(ti, train[ti].0) { tc += 1; } }
        let mut xc = 0u32; for ti in 0..test.len() { if classify(train.len() + ti, test[ti].0) { xc += 1; } }
        eprintln!("    random start m={m:>3} rank={rk}  train {tc}/{} ({:.1}%)  test {xc}/{} ({:.1}%)",
            train.len(), tc as f64 / train.len() as f64 * 100.0, test.len(), xc as f64 / test.len() as f64 * 100.0);
    }

    eprintln!("\n  --- λ/2 (λ₀=0.5, λ₁=0.5) vacuum start ---");
    for &m in &[5, 10, 15, 20] {
        let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None, random_start: false, ..Default::default() };
        let pipe = QfmPipeline::compile_channels(&im, &om, &tr, 0.5, 0.5, nm, &cfg, 0.0, 0.0, true, None, None).unwrap();
        let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
        let u = (h.clone() * (-C64::new(0.0, 1.0) * 1.0)).exp(); let no = 5;
        let classify = |img_idx: usize, tl: u32| -> bool {
            let mut c0 = DVector::zeros(rk);
            for f in 0..no { let tp = img_idx * no + f; for r in 0..rk { c0[r] += w[(tp, r)] * C64::new(1.0 / (no as f64).sqrt(), 0.0); } }
            let c1 = &u * c0; let mut pr = vec![0.0_f64; no];
            for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
            let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
            let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap(); pd == tl
        };
        let mut tc = 0u32; for ti in 0..train.len() { if classify(ti, train[ti].0) { tc += 1; } }
        let mut xc = 0u32; for ti in 0..test.len() { if classify(train.len() + ti, test[ti].0) { xc += 1; } }
        eprintln!("    λ/2 vacuum   m={m:>3} rank={rk}  train {tc}/{} ({:.1}%)  test {xc}/{} ({:.1}%)",
            train.len(), tc as f64 / train.len() as f64 * 100.0, test.len(), xc as f64 / test.len() as f64 * 100.0);
    }

    eprintln!("\n  --- λ/2 (λ₀=0.5, λ₁=0.5) random start ---");
    for &m in &[5, 10, 15, 20] {
        let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1, max_rank: None, random_start: true, ..Default::default() };
        let pipe = QfmPipeline::compile_channels(&im, &om, &tr, 0.5, 0.5, nm, &cfg, 0.0, 0.0, true, None, None).unwrap();
        let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
        let u = (h.clone() * (-C64::new(0.0, 1.0) * 1.0)).exp(); let no = 5;
        let classify = |img_idx: usize, tl: u32| -> bool {
            let mut c0 = DVector::zeros(rk);
            for f in 0..no { let tp = img_idx * no + f; for r in 0..rk { c0[r] += w[(tp, r)] * C64::new(1.0 / (no as f64).sqrt(), 0.0); } }
            let c1 = &u * c0; let mut pr = vec![0.0_f64; no];
            for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
            let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
            let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap(); pd == tl
        };
        let mut tc = 0u32; for ti in 0..train.len() { if classify(ti, train[ti].0) { tc += 1; } }
        let mut xc = 0u32; for ti in 0..test.len() { if classify(train.len() + ti, test[ti].0) { xc += 1; } }
        eprintln!("    λ/2 random  m={m:>3} rank={rk}  train {tc}/{} ({:.1}%)  test {xc}/{} ({:.1}%)",
            train.len(), tc as f64 / train.len() as f64 * 100.0, test.len(), xc as f64 / test.len() as f64 * 100.0);
    }

    eprintln!("\n  --- σ sweep (λ₀=1.0, λ₁=1.0, m=10) ---");
    for &sigma in &[0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0] {
        run(&train, &test, sigma, 10, 1.0, 1.0, 1.0, HamiltonianType::Diffusion);
    }

    eprintln!("\n  --- σ sweep (λ₀=1.0, λ₁=1.0, m=20) ---");
    for &sigma in &[0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0] {
        run(&train, &test, sigma, 20, 1.0, 1.0, 1.0, HamiltonianType::Diffusion);
    }

    for &rst in &[false, true] {
        let rtag = if rst { "random" } else { "vacuum" };
        eprintln!("\n  --- Pauli-Grover (a=0.9, t=π/2, start=|0⟩, {rtag}) ---");
        for &m in &[5, 10, 15, 20] {
            let cfg = QfmConfig { k: 1, k2: nm, krylov_dim: m, seed: 42, n_t_samples: 4, noise_dim: 1,
                max_rank: None, random_start: rst, hamiltonian_type: HamiltonianType::PauliGrover,
                pauli_grover_a: 0.9 };
            let pipe = QfmPipeline::compile_channels(&im, &om, &tr, 0.0, 0.0, nm, &cfg, 0.0, 0.0, true, None, None).unwrap();
            let rk = pipe.rank(); let w = pipe.w(); let h = pipe.h_m();
            let t = std::f64::consts::PI / 2.0;
            let u = (h.clone() * (-C64::new(0.0, 1.0) * t)).exp();
            let no = 5;
            let classify = |img_idx: usize, tl: u32| -> bool {
                let mut c0 = DVector::zeros(rk);
                let tp0 = img_idx * no;
                for r in 0..rk { c0[r] = w[(tp0, r)]; }
                let c1 = &u * c0; let mut pr = vec![0.0_f64; no];
                for f in 0..no { let tp = img_idx * no + f; let a: C64 = (0..rk).map(|r| c1[r] * w[(tp, r)].conj()).sum(); pr[f] = a.norm_sqr(); }
                let s: f64 = pr.iter().sum(); if s > 0.0 { for p in &mut pr { *p /= s; } }
                let pd = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap(); pd == tl
            };
            let mut tc = 0u32; for ti in 0..train.len() { if classify(ti, train[ti].0) { tc += 1; } }
            let mut xc = 0u32; for ti in 0..test.len() { if classify(train.len() + ti, test[ti].0) { xc += 1; } }
            eprintln!("    pg a=0.9 {rtag:>7} m={m:>3} rank={rk}  train {tc}/{} ({:.1}%)  test {xc}/{} ({:.1}%)",
                train.len(), tc as f64 / train.len() as f64 * 100.0, test.len(), xc as f64 / test.len() as f64 * 100.0);
        }
    }
}
