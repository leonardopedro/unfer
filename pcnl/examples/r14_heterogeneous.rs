//! R14/P6 — scoped heterogeneity evaluation (gated by `docs/PCNL_P6_REVIEW.md`).
//!
//! The written review approves **only** the heterogeneous-width evaluation:
//! no Git Re-Basin, no functional repulsion, no promotion.  This example runs
//! inside the declared compute cap (`3` seeds × `3` members × `600` Adam
//! steps) and compares, at matched total parameter count:
//!
//! * a homogeneous width-16 ensemble (`16, 16, 16`, 172 134 parameters), and
//! * a heterogeneous ensemble (`12, 17, 19`, 172 550 parameters, +0.24 %).
//!
//! Reported: member satisfaction, ensemble test RMSE, functional effective
//! rank/disagreement, distilled-student RMSE and teacher–student fidelity,
//! and measured ensemble forward latency on this host.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_heterogeneous
//! ```

use candle_core::{Device, Tensor};
use pcnl::ensemble_diag::{effective_rank, prediction_spread, rmse};
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::text_head::{EmbeddingTable, Split};
use pcnl::vector_model::{
    SyntheticTeacher, VectorRegression, VectorShapes, adam_fit, forward, init_flat_prior, predict,
};
use std::hint::black_box;
use std::time::Instant;

const STEPS: usize = 600;
const LR: f64 = 0.02;
const TAU: f64 = 0.75;
const SEEDS: [u64; 3] = [71, 82, 93];
const MEMBERS: usize = 3;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let m = mean(xs);
    if xs.len() < 2 {
        return (m, f64::NAN);
    }
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    (m, (var / xs.len() as f64).sqrt())
}

fn shapes(width: usize) -> VectorShapes {
    VectorShapes::with_width(768, 2, width, 32, 4, 4).expect("shapes")
}

struct Trained {
    shapes: VectorShapes,
    theta: Vec<f32>,
    loss: f64,
}

fn train(widths: &[usize], x: &[f32], y: &[f32], family: u64) -> Vec<Trained> {
    let mut out = Vec::new();
    for (k, &width) in widths.iter().enumerate() {
        let s = shapes(width);
        let data = VectorRegression::new(s, x.to_vec(), y.to_vec(), 0.05, 1.0).expect("data");
        let start = init_flat_prior(&data.shapes(), 0.05, family + 11 * k as u64);
        let (theta_t, _) = adam_fit(&data, start, STEPS, LR);
        let theta = theta_t.to_vec1::<f32>().expect("flat");
        let cfg = SatisficingConfig {
            tau: TAU,
            kappa: 20.0,
            alpha: 0.05,
            delta: 0.1,
            hinge: Hinge::Huberized,
        };
        let sat = Satisficing::new(&data, cfg, "p4", "mean").expect("sat");
        let loss = sat.loss(&theta).expect("loss");
        out.push(Trained {
            shapes: *data.shapes(),
            theta,
            loss,
        });
    }
    out
}

fn member_predictions(members: &[Trained], x: &[f32], n: usize) -> Vec<Vec<f64>> {
    members
        .iter()
        .map(|m| {
            let mut flat = Vec::with_capacity(n * 2);
            for row in 0..n {
                let e = &x[row * 768..(row + 1) * 768];
                flat.extend(
                    predict(&m.theta, &m.shapes, e)
                        .expect("predict")
                        .into_iter()
                        .map(f64::from),
                );
            }
            flat
        })
        .collect()
}

fn mean_flat(preds: &[Vec<f64>]) -> Vec<f64> {
    let q = preds[0].len();
    let mut out = vec![0.0; q];
    for p in preds {
        for (o, v) in out.iter_mut().zip(p) {
            *o += v / preds.len() as f64;
        }
    }
    out
}

fn teacher_mean_on(members: &[Trained], x: &[f32], n: usize) -> Vec<f32> {
    mean_flat(&member_predictions(members, x, n))
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn latency_us(members: &[Trained], x_row: &[f32], n: usize) -> f64 {
    let device = Device::Cpu;
    let tensors: Vec<(Tensor, Tensor, VectorShapes)> = members
        .iter()
        .map(|m| {
            (
                Tensor::from_vec(m.theta.clone(), m.theta.len(), &device).expect("theta"),
                Tensor::from_vec(x_row.to_vec(), 768, &device).expect("embedding"),
                m.shapes,
            )
        })
        .collect();
    for _ in 0..20 {
        for (t, e, s) in &tensors {
            black_box(forward(t, s, e).expect("forward"));
        }
    }
    let mut lat = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        for (t, e, s) in &tensors {
            black_box(forward(t, s, e).expect("forward"));
        }
        lat.push(start.elapsed().as_nanos() as f64);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lat[lat.len() / 2] / 1e3
}

fn student_fit(teacher: &[f32], x: &[f32], n_train: usize, seed: u64) -> Vec<f32> {
    let s = shapes(16);
    let data = VectorRegression::new(s, x[..n_train * 768].to_vec(), teacher.to_vec(), 0.05, 1.0)
        .expect("student data");
    let start = init_flat_prior(&data.shapes(), 0.05, seed);
    let (theta, _) = adam_fit(&data, start, STEPS, LR);
    theta.to_vec1::<f32>().expect("flat")
}

fn paired_abs_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f64>() / a.len() as f64
}

fn main() {
    let table = EmbeddingTable::load(std::path::Path::new(
        "docs/fixtures/ultrabert_20ng4_embeddings.safetensors",
    ))
    .expect("20ng fixture");
    let input = table.dim();
    assert_eq!(input, 768, "this evaluation declares the 768-d fixture");
    let n_train = 240.min(table.x(Split::Train).len() / input);
    let n_test = 120.min(table.x(Split::Test).len() / input);
    let x_train = table.x(Split::Train)[..n_train * input].to_vec();
    let x_test = table.x(Split::Test)[..n_test * input].to_vec();
    let teacher = SyntheticTeacher::new(input, 2, 20261127);
    let noise = 0.05;
    let mut y_train = Vec::new();
    let mut y_test = Vec::new();
    for row in 0..n_train {
        y_train.extend(teacher.targets(
            &x_train[row * input..(row + 1) * input],
            noise,
            900 + row as u64,
        ));
    }
    for row in 0..n_test {
        y_test.extend(teacher.targets(
            &x_test[row * input..(row + 1) * input],
            noise,
            1900 + row as u64,
        ));
    }
    let y_test_f: Vec<f64> = y_test.iter().map(|v| f64::from(*v)).collect();

    println!(
        "r14=p6-heterogeneous decision=conditional-approve-heterogeneity-only seeds={} members={MEMBERS} adam_steps={STEPS} tau={TAU} compute_cap=seeds:{},members:{MEMBERS},adam_steps:{STEPS} n_train={n_train} n_test={n_test}",
        SEEDS.len(),
        SEEDS.len()
    );

    let mut hom_rmse = Vec::new();
    let mut het_rmse = Vec::new();
    let mut hom_rank = Vec::new();
    let mut het_rank = Vec::new();
    let mut hom_dis = Vec::new();
    let mut het_dis = Vec::new();
    let mut hom_region = Vec::new();
    let mut het_region = Vec::new();
    let mut hom_student_rmse = Vec::new();
    let mut het_student_rmse = Vec::new();
    let mut hom_fidelity = Vec::new();
    let mut het_fidelity = Vec::new();
    let mut hom_lat = Vec::new();
    let mut het_lat = Vec::new();
    let mut paired_teacher = Vec::new();
    let mut paired_student = Vec::new();

    for &family in &SEEDS {
        let hom = train(&[16, 16, 16], &x_train, &y_train, family);
        let het = train(&[12, 17, 19], &x_train, &y_train, family);
        let params_hom: usize = hom.iter().map(|m| m.theta.len()).sum();
        let params_het: usize = het.iter().map(|m| m.theta.len()).sum();

        let preds_hom = member_predictions(&hom, &x_test, n_test);
        let preds_het = member_predictions(&het, &x_test, n_test);
        let mean_hom = mean_flat(&preds_hom);
        let mean_het = mean_flat(&preds_het);
        let rh = rmse(&mean_hom, &y_test_f).expect("rmse");
        let rt = rmse(&mean_het, &y_test_f).expect("rmse");
        let rank_h = effective_rank(&preds_hom).expect("rank");
        let rank_t = effective_rank(&preds_het).expect("rank");
        let dis_h = prediction_spread(&preds_hom).expect("spread").disagreement;
        let dis_t = prediction_spread(&preds_het).expect("spread").disagreement;
        let region_h = hom.iter().filter(|m| m.loss <= TAU).count() as f64 / MEMBERS as f64;
        let region_t = het.iter().filter(|m| m.loss <= TAU).count() as f64 / MEMBERS as f64;

        let teacher_h = teacher_mean_on(&hom, &x_train, n_train);
        let teacher_t = teacher_mean_on(&het, &x_train, n_train);
        let student_h = student_fit(&teacher_h, &x_train, n_train, family + 700);
        let student_t = student_fit(&teacher_t, &x_train, n_train, family + 700);
        let student_predictions = |w: &[f32]| -> Vec<f64> {
            let s = shapes(16);
            (0..n_test)
                .flat_map(|row| {
                    predict(w, &s, &x_test[row * 768..(row + 1) * 768])
                        .expect("predict")
                        .into_iter()
                        .map(f64::from)
                })
                .collect()
        };
        let sh_pred = student_predictions(&student_h);
        let st_pred = student_predictions(&student_t);
        let sh_rmse = rmse(&sh_pred, &y_test_f).expect("rmse");
        let st_rmse = rmse(&st_pred, &y_test_f).expect("rmse");
        // Teacher–student fidelity against each arm's own teacher mean.
        let teacher_test_h = mean_hom.clone();
        let teacher_test_t = mean_het.clone();
        let fid_h = rmse(&sh_pred, &teacher_test_h).expect("fidelity");
        let fid_t = rmse(&st_pred, &teacher_test_t).expect("fidelity");

        let lat_h = latency_us(&hom, &x_test[..768], 300);
        let lat_t = latency_us(&het, &x_test[..768], 300);

        let losses = |ms: &[Trained]| {
            ms.iter()
                .map(|m| format!("{}:{:.4}", m.shapes.width, m.loss))
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "seed={family} member_losses homogeneous={} heterogeneous={}",
            losses(&hom),
            losses(&het)
        );
        println!(
            "seed={family} arm=homogeneous widths=16,16,16 params={params_hom} in_region={region_h:.3} test_rmse={rh:.6} diversity_rank={rank_h:.4} disagreement={dis_h:.6} student_rmse={sh_rmse:.6} teacher_fidelity={fid_h:.6} latency_us={lat_h:.2}"
        );
        println!(
            "seed={family} arm=heterogeneous widths=12,17,19 params={params_het} param_ratio={:.6} in_region={region_t:.3} test_rmse={rt:.6} diversity_rank={rank_t:.4} disagreement={dis_t:.6} student_rmse={st_rmse:.6} teacher_fidelity={fid_t:.6} latency_us={lat_t:.2}",
            params_het as f64 / params_hom as f64
        );

        hom_rmse.push(rh);
        het_rmse.push(rt);
        hom_rank.push(rank_h);
        het_rank.push(rank_t);
        hom_dis.push(dis_h);
        het_dis.push(dis_t);
        hom_region.push(region_h);
        het_region.push(region_t);
        hom_student_rmse.push(sh_rmse);
        het_student_rmse.push(st_rmse);
        hom_fidelity.push(fid_h);
        het_fidelity.push(fid_t);
        hom_lat.push(lat_h);
        het_lat.push(lat_t);
        paired_teacher.push(paired_abs_delta(&mean_het, &mean_hom));
        paired_student.push(paired_abs_delta(&st_pred, &sh_pred));
    }

    let row = |name: &str, vals: &[f64]| {
        let (m, s) = mean_se(vals);
        println!("aggregate arm={name}_mean={m:.6} se={s:.6}");
    };
    row("homogeneous_test_rmse", &hom_rmse);
    row("heterogeneous_test_rmse", &het_rmse);
    row("homogeneous_rank", &hom_rank);
    row("heterogeneous_rank", &het_rank);
    row("homogeneous_disagreement", &hom_dis);
    row("heterogeneous_disagreement", &het_dis);
    row("homogeneous_in_region", &hom_region);
    row("heterogeneous_in_region", &het_region);
    row("homogeneous_student_rmse", &hom_student_rmse);
    row("heterogeneous_student_rmse", &het_student_rmse);
    row("homogeneous_teacher_fidelity", &hom_fidelity);
    row("heterogeneous_teacher_fidelity", &het_fidelity);
    row("homogeneous_latency_us", &hom_lat);
    row("heterogeneous_latency_us", &het_lat);

    let (dt, st) = mean_se(&paired_teacher);
    let (ds, ss) = mean_se(&paired_student);
    let (dr, sr) = mean_se(
        &het_rmse
            .iter()
            .zip(&hom_rmse)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    println!(
        "paired heterogeneous_vs_homogeneous teacher_abs_delta={dt:+.6} se={st:.6} student_abs_delta={ds:+.6} se={ss:.6} rmse_delta={dr:+.6} se={sr:.6}"
    );
    println!(
        "verdict heterogeneity_hypothesis={} promotion=blocked-by-review",
        if het_rmse.iter().zip(&hom_rmse).all(|(t, h)| t - h <= 0.005) {
            "supported"
        } else {
            "not-supported"
        }
    );
}
