//! R14 serving cost: distilled student versus the `M`-member ensemble.
//!
//! Measures, on CPU, for the P4 vector models (57 378 parameters) and the P5
//! softmax policies (12–15 parameters):
//!
//! * model storage bytes (computed and allocator-measured),
//! * per-query latency mean/p50/p90/p99 and throughput,
//! * bytes allocated per query,
//! * forward-only versus the flat-slice `predict` entry point (which clones
//!   the parameter vector on each call).
//!
//! Latency is machine-dependent and is therefore **not** part of the
//! byte-exact replay gate; the log is a recorded measurement.
//!
//! ```text
//! cargo run --offline --release -p pcnl --example r14_serving_bench
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use candle_core::{Device, Tensor};
use pcnl::distill::{ensemble_probs, fit_softmax_student};
use pcnl::ensemble::{KneeRule, WarmEntryConfig, warm_entry};
use pcnl::mdp::{BehaviourPolicy, Corpus, MdpEnv};
use pcnl::rl_track::RlLoss;
use pcnl::satisficing::{Hinge, Satisficing, SatisficingConfig};
use pcnl::vector_model::{
    HitRunConfig, VectorRegression, VectorShapes, adam_fit, forward, init_flat_prior, predict,
    run_hit_and_run,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

struct Counting;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
            TOTAL_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let diff = new_size - layout.size();
                let now = ALLOCATED.fetch_add(diff, Ordering::Relaxed) + diff;
                PEAK.fetch_max(now, Ordering::Relaxed);
                TOTAL_ALLOCATED.fetch_add(diff, Ordering::Relaxed);
            } else {
                ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct Stats {
    mean_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    p99_ns: f64,
    qps: f64,
    bytes_per_query: f64,
}

fn bench<F: FnMut() -> Vec<f64>>(mut f: F, n: usize) -> Stats {
    for _ in 0..(n / 10).max(64) {
        black_box(f());
    }
    let alloc0 = TOTAL_ALLOCATED.load(Ordering::Relaxed);
    let t0 = Instant::now();
    let mut lat = Vec::with_capacity(n);
    for _ in 0..n {
        let s = Instant::now();
        black_box(f());
        lat.push(s.elapsed().as_nanos() as f64);
    }
    let total = t0.elapsed().as_secs_f64();
    let alloc_bytes = TOTAL_ALLOCATED.load(Ordering::Relaxed) - alloc0;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| lat[((lat.len() - 1) as f64 * q).floor() as usize];
    Stats {
        mean_ns: lat.iter().sum::<f64>() / lat.len() as f64,
        p50_ns: pick(0.50),
        p90_ns: pick(0.90),
        p99_ns: pick(0.99),
        qps: n as f64 / total,
        bytes_per_query: alloc_bytes as f64 / n as f64,
    }
}

fn report(case: &str, arm: &str, n: usize, s: &Stats) -> String {
    let line = format!(
        "serving case={case} arm={arm} n={n} mean_us={:.3} p50_us={:.3} p90_us={:.3} p99_us={:.3} qps={:.1} alloc_bytes_per_query={:.1}",
        s.mean_ns / 1e3,
        s.p50_ns / 1e3,
        s.p90_ns / 1e3,
        s.p99_ns / 1e3,
        s.qps,
        s.bytes_per_query
    );
    println!("{line}");
    line
}

fn mean_rows(rows: &[Vec<f32>]) -> Vec<f64> {
    let dim = rows[0].len();
    let mut out = vec![0.0; dim];
    for r in rows {
        for (o, v) in out.iter_mut().zip(r) {
            *o += f64::from(*v);
        }
    }
    for o in &mut out {
        *o /= rows.len() as f64;
    }
    out
}

fn serving_log(case: &str, members: usize, model_bytes: usize, student_bytes: usize, heap: i64) {
    println!(
        "serving case={case} members={members} bytes_per_member={model_bytes} ensemble_bytes={} student_bytes={student_bytes} memory_ratio={:.4} heap_delta_bytes={heap}",
        model_bytes * members,
        (model_bytes * members) as f64 / student_bytes as f64
    );
}

fn main() {
    let device = Device::Cpu;
    let shapes = VectorShapes::new(768, 2).expect("shapes");
    let dim = shapes.flat_size();
    let teacher = pcnl::vector_model::SyntheticTeacher::new(768, 2, 11);
    let mut rng = StdRng::seed_from_u64(5);
    let mut x = Vec::new();
    for _ in 0..120 {
        for _ in 0..shapes.input {
            let z: f64 = rng.sample(StandardNormal);
            x.push(z as f32);
        }
    }
    let y: Vec<f32> = (0..120)
        .flat_map(|row| {
            let emb = &x[row * shapes.input..(row + 1) * shapes.input];
            teacher.predict(emb)
        })
        .collect();
    let data = VectorRegression::new(shapes.clone(), x.clone(), y, 0.05, 0.01).expect("p4 data");
    let config = SatisficingConfig {
        tau: 0.75,
        kappa: 20.0,
        alpha: 0.05,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let strict = SatisficingConfig {
        kappa: 2_000_000.0,
        ..config
    };
    let sat_strict = Satisficing::new(&data, strict, "p4", "mean").expect("p4 strict");
    let prior_c = vec![1.0f64; dim];
    let warm = WarmEntryConfig {
        max_steps: 400,
        lr: 0.05,
        tau: config.tau,
        knee: KneeRule::default(),
    };
    let mut knee = Vec::new();
    for k in 0..3u64 {
        let start = init_flat_prior(&shapes, 1.0, 101 + k)
            .to_vec1::<f32>()
            .expect("flat");
        knee.push(warm_entry(&data, &start, warm).expect("warm").theta);
    }
    let hitrun_cfg = HitRunConfig {
        moves: 160,
        burn: 40,
        thin: 4,
        draws: 30,
        width: 1.0,
        max_expand: 32,
        max_shrink: 64,
    };
    let mut hitrun = Vec::new();
    for (k, start) in knee.iter().enumerate() {
        let chain = run_hit_and_run(
            &sat_strict,
            &prior_c,
            hitrun_cfg.clone(),
            start.clone(),
            900 + k as u64,
        )
        .expect("hitrun");
        hitrun.push(
            chain
                .draws
                .iter()
                .rev()
                .find(|d| sat_strict.loss(d).map(|l| l <= strict.tau).unwrap_or(false))
                .cloned()
                .unwrap_or(chain.theta),
        );
    }

    // Distilled student: same vector architecture, fitted to the teacher mean.
    let teacher_train: Vec<Vec<f32>> = (0..120)
        .map(|row| {
            let e = Tensor::from_vec(
                x[row * shapes.input..(row + 1) * shapes.input].to_vec(),
                shapes.input,
                &device,
            )
            .expect("embedding");
            mean_rows(
                &hitrun
                    .iter()
                    .map(|w| {
                        let t = Tensor::from_vec(w.clone(), dim, &device).expect("theta");
                        forward(&t, &shapes, &e)
                            .expect("forward")
                            .to_vec1::<f32>()
                            .expect("out")
                    })
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|v| v as f32)
            .collect()
        })
        .collect();
    let student_y: Vec<f32> = teacher_train.into_iter().flatten().collect();
    let student_data = VectorRegression::new(shapes.clone(), x.clone(), student_y, 0.05, 0.01)
        .expect("student data");
    let start = init_flat_prior(&shapes, 1.0, 777);
    let (student_t, _) = adam_fit(&student_data, start, 400, 2e-3);
    let student = student_t.to_vec1::<f32>().expect("student flat");
    let model_bytes = dim * std::mem::size_of::<f32>();
    let student_bytes = student.len() * std::mem::size_of::<f32>();
    let before = ALLOCATED.load(Ordering::Relaxed) as i64;
    let members_held = hitrun.clone();
    let after = ALLOCATED.load(Ordering::Relaxed) as i64;
    serving_log(
        "p4",
        members_held.len(),
        model_bytes,
        student_bytes,
        after - before,
    );

    // Prebuilt tensors: forward-only cost.
    let heat: Vec<Tensor> = members_held
        .iter()
        .map(|w| Tensor::from_vec(w.clone(), dim, &device).expect("theta"))
        .collect();
    let student_heat = Tensor::from_vec(student.clone(), dim, &device).expect("theta");
    let emb_batch: Vec<Tensor> = (0..8)
        .map(|i| {
            Tensor::from_vec(
                x[i * shapes.input..(i + 1) * shapes.input].to_vec(),
                shapes.input,
                &device,
            )
            .expect("embedding")
        })
        .collect();
    let mut e_idx = 0usize;
    let mut member_idx = 0usize;
    let n4 = 3000usize;

    let s = bench(
        || {
            let out = forward(
                &heat[member_idx % heat.len()],
                &shapes,
                &emb_batch[e_idx % emb_batch.len()],
            )
            .expect("forward")
            .to_vec1::<f32>()
            .expect("out");
            e_idx += 1;
            member_idx += 1;
            out.into_iter().map(f64::from).collect()
        },
        n4 * 3,
    );
    report("p4", "member-forward", n4 * 3, &s);

    e_idx = 0;
    let s = bench(
        || {
            let mut acc = vec![0.0f64; shapes.output];
            for t in &heat {
                let o = forward(t, &shapes, &emb_batch[e_idx % emb_batch.len()])
                    .expect("forward")
                    .to_vec1::<f32>()
                    .expect("out");
                for (a, v) in acc.iter_mut().zip(&o) {
                    *a += f64::from(*v) / heat.len() as f64;
                }
            }
            e_idx += 1;
            acc
        },
        n4,
    );
    report("p4", "ensemble-forward-m3", n4, &s);

    e_idx = 0;
    let s = bench(
        || {
            let o = forward(&student_heat, &shapes, &emb_batch[e_idx % emb_batch.len()])
                .expect("forward")
                .to_vec1::<f32>()
                .expect("out");
            e_idx += 1;
            o.into_iter().map(f64::from).collect()
        },
        n4,
    );
    report("p4", "student-forward", n4, &s);

    // Flat-slice API: includes the per-call parameter clone.
    e_idx = 0;
    let inputs: Vec<Vec<f32>> = (0..8)
        .map(|i| x[i * shapes.input..(i + 1) * shapes.input].to_vec())
        .collect();
    let s = bench(
        || {
            let e = &inputs[e_idx % inputs.len()];
            let mut acc = vec![0.0f64; shapes.output];
            for w in &members_held {
                let o = predict(w, &shapes, e).expect("predict");
                for (a, v) in acc.iter_mut().zip(&o) {
                    *a += f64::from(*v) / members_held.len() as f64;
                }
            }
            e_idx += 1;
            acc
        },
        n4,
    );
    report("p4", "ensemble-predict-m3", n4, &s);

    e_idx = 0;
    let s = bench(
        || {
            predict(&student, &shapes, &inputs[e_idx % inputs.len()])
                .expect("predict")
                .into_iter()
                .map(f64::from)
                .collect()
        },
        n4,
    );
    report("p4", "student-predict", n4, &s);

    // ---- P5: softmax policies -------------------------------------------
    let env = MdpEnv {
        horizon: 8,
        ..MdpEnv::default()
    };
    let behavior = BehaviourPolicy {
        feedback: 0.5,
        epsilon: 0.2,
        seed: 17,
    };
    let (train, _dev, _test) = Corpus::splits(&env, behavior, (6, 4, 4), 5000);
    let loss = RlLoss::from_corpus(&train, 0.05, 20.0).expect("rl loss");
    let p5_strict = SatisficingConfig {
        tau: 1.2,
        kappa: 2_000_000.0,
        alpha: 0.0,
        delta: 0.1,
        hinge: Hinge::Huberized,
    };
    let sat5 = Satisficing::new(&loss, p5_strict, "p5", "mean").expect("p5 sat");
    let prior5 = vec![1.0f64; loss.flat_size()];
    let warm5 = WarmEntryConfig {
        max_steps: 400,
        lr: 0.05,
        tau: p5_strict.tau,
        knee: KneeRule::default(),
    };
    let mut p5_members = Vec::new();
    for k in 0..3u64 {
        let mut start_rng = StdRng::seed_from_u64(51 + k);
        let start: Vec<f32> = (0..loss.flat_size())
            .map(|_| {
                let z: f64 = start_rng.sample(StandardNormal);
                z as f32
            })
            .collect();
        let knee5 = warm_entry(&loss, &start, warm5).expect("p5 warm").theta;
        let chain = run_hit_and_run(&sat5, &prior5, hitrun_cfg.clone(), knee5, 6100 + k)
            .expect("p5 hitrun");
        p5_members.push(chain.theta);
    }
    let states: Vec<Vec<f64>> = (0..8)
        .map(|i| vec![-1.0 + 0.3 * i as f64, 0.5 - 0.1 * i as f64, 1.0])
        .collect();
    let teacher_probs: Vec<(Vec<f64>, Vec<f64>)> = states
        .iter()
        .map(|x| {
            let ps: Vec<Vec<f64>> = p5_members
                .iter()
                .map(|w| loss.probs(w, x).expect("teacher"))
                .collect();
            (x.clone(), ensemble_probs(&ps).expect("ensemble"))
        })
        .collect();
    let student5 =
        fit_softmax_student(3, loss.classes(), &teacher_probs, 600, 0.1).expect("p5 student");
    let student5 = student5.student;
    let p5_model_bytes = loss.flat_size() * std::mem::size_of::<f32>();
    let p5_student_bytes =
        (student5.weights.len() + student5.bias.len()) * std::mem::size_of::<f64>();
    let before = ALLOCATED.load(Ordering::Relaxed) as i64;
    let p5_held = p5_members.clone();
    let after = ALLOCATED.load(Ordering::Relaxed) as i64;
    serving_log(
        "p5",
        p5_held.len(),
        p5_model_bytes,
        p5_student_bytes,
        after - before,
    );

    let mut si = 0usize;
    let n5 = 200_000usize;
    let s = bench(
        || {
            let ps: Vec<Vec<f64>> = p5_held
                .iter()
                .map(|w| loss.probs(w, &states[si % states.len()]).expect("probs"))
                .collect();
            si += 1;
            ensemble_probs(&ps).expect("ensemble")
        },
        n5,
    );
    report("p5", "ensemble-probs-m3", n5, &s);

    si = 0;
    let s = bench(
        || {
            let p = student5.probs(&states[si % states.len()]).expect("probs");
            si += 1;
            p
        },
        n5,
    );
    report("p5", "student-probs", n5, &s);
}
