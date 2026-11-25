//! Timing probe for the parallel outer-label / inner-map action (`P13`).
//!
//! Measures the C = 20 sweep cell twice — once with
//! `FOCK_SIRK_ACTION_THREADS=1` and once with the default worker count — so the
//! speedup quoted in `docs/VERIFY_PHI4_MOMENTUM.md` is re-measurable:
//!
//! ```text
//! FOCK_SIRK_ACTION_THREADS=1 cargo run -q -p fock_sirk --release \
//!     --example phi4_momentum_parallel_probe
//! cargo run -q -p fock_sirk --release --example phi4_momentum_parallel_probe
//! ```
//!
//! The parallel path is bit-identical to the serial one (it merges each term's
//! image as a whole map, in term order), so the two runs must report exactly
//! the same return probability; only the timings differ.
//!
//! Measured on a 12-core host (`docs/VERIFY_PHI4_MOMENTUM.md` §5.5):
//!
//! | quantity | serial | parallel |
//! |---|---|---|
//! | `d_gamma`, 1001 labels | `45.6 ms` | `29.9 ms` (1.5x) |
//! | `m = 2` solve | `62.9 s` | `63.3 s` (0.99x) |
//!
//! The solve does not follow the matvec because full-GMRES orthogonalisation,
//! not the action, dominates it at this size; `axpy_ms` shows the cost of one
//! nested-vector addition, which the Krylov loop performs far more often than a
//! matvec.
//!
//! `PROBE_CUTOFF` selects the window (default 20) and `PROBE_COMPONENT_BUDGET`
//! the component ceiling (default 8e6, matching the sweep; `C = 26` needs
//! 5.71e6 and `C = 28` needs 9.36e6). The probe also times the exact window
//! reference the sweep evaluates beside every solve — the seed column of
//! `exp(-i t H_window)` from `common::window_reference`, the other half of a
//! large cell under the dense reference and a few seconds of it now
//! (`docs/VERIFY_PHI4_MOMENTUM.md` §5.7–§5.8):
//!
//! ```text
//! PROBE_CUTOFF=26 PHI4_PROFILE=1 cargo run -q -p fock_sirk --release \
//!     --example phi4_momentum_parallel_probe
//! ```

use fock_sirk::joint_sirk::{
    JointKrylovMethod, JointSirkOptions, hashimoto_unitary_shifts, joint_axpy, joint_inner_product,
    solve_joint_sirk,
};
use nested_fock_algebra::exact_inner::{ActionLimits, JointOuterBasis, NestedVector};
use nested_fock_algebra::phi4_momentum::ScalarMomCont;
use num_complex::Complex64 as C;
use std::time::Instant;

mod common;

use common::window_reference::WindowOperator;

fn main() {
    let t = 0.05;
    let cutoff: u32 = std::env::var("PROBE_CUTOFF")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(20);
    let model = ScalarMomCont::unit();
    let h = model.hamiltonian();
    let initial = model.one_particle(&[([0, 0, 0, 0], 1.0)]);
    let window = ScalarMomCont::window_labels(cutoff);
    // Same budget as the sweep (8e6): `C = 26` needs 5.71e6 components, and a
    // smaller ceiling silently turns the probe into a budget-failure report.
    let budget: usize = std::env::var("PROBE_COMPONENT_BUDGET")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(8_000_000);
    let limits = ActionLimits {
        max_components: budget,
        max_work: usize::MAX / 4,
    };

    // A dense state over the window: the inner action then maps over every
    // inner entry, which is the sweep's hot path.
    let mut state = NestedVector::new();
    for (n, basis) in window.iter().enumerate() {
        state.insert(basis.clone(), C::new(1.0 / (1.5 + n as f64), 0.0));
    }

    let reps = 5;
    let started = Instant::now();
    for _ in 0..reps {
        let image = h.d_gamma(&state, limits).expect("finite action");
        assert!(!image.is_empty());
    }
    let matvec = started.elapsed().as_secs_f64() / reps as f64;

    // Nested-vector arithmetic the Krylov loop performs per iteration: a dense
    // `w += c * basis[j]` and a `<basis[j], w>` dot product, both over the
    // window. These, not the action, are where a long GMRES spends its time.
    let image = h.d_gamma(&state, limits).expect("finite action");
    let reps = 40;
    let mut combined = state.clone();
    let started = Instant::now();
    for stroke in 0..reps {
        let scale = if stroke % 2 == 0 { -0.7 } else { 0.7 };
        joint_axpy(&mut combined, &image, C::new(scale, 0.3), limits).expect("finite axpy");
        std::hint::black_box(&combined);
    }
    let axpy = started.elapsed().as_secs_f64() / reps as f64;

    let reps = 40;
    let started = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(joint_inner_product(&state, &image));
    }
    let dot = started.elapsed().as_secs_f64() / reps as f64;

    // The same two operations split across workers. The dot is the interesting
    // one: it cannot be made bit-identical, because a chunked partial-sum merge
    // associates the additions differently from the left-to-right fold. Timing
    // it is still worth doing -- it is the only way to say whether the Krylov
    // loop could be parallelised at all.
    let mut parallel = Vec::new();
    {
        let reps = 20;
        let started = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(state.clone());
        }
        let clone = started.elapsed().as_secs_f64() / reps as f64;
        let worst = [1_usize, 2, 4, 8]
            .into_iter()
            .map(|threads| {
                let d = parallel_dot(&state, &image, threads);
                let exact = joint_inner_product(&state, &image);
                (exact - d).norm() / exact.norm()
            })
            .fold(0.0_f64, f64::max);
        parallel.push((0, 0.0, clone, 0.0));
        for threads in [1_usize, 2, 4, 8] {
            let reps = 20;
            let started = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(parallel_dot(&state, &image, threads));
            }
            let pdot = started.elapsed().as_secs_f64() / reps as f64;
            let started = Instant::now();
            for _ in 0..reps {
                let mut target = state.clone();
                std::hint::black_box(parallel_axpy(&mut target, &image, threads, limits));
            }
            let paxpy = started.elapsed().as_secs_f64() / reps as f64 - clone;
            parallel.push((threads, pdot, paxpy, worst));
        }
    }

    let shifts = hashimoto_unitary_shifts(t, 8.0 * t, 3.0 * t, 2).expect("valid schedule");
    let options = JointSirkOptions {
        basis_window: Some(window.clone()),
        method: JointKrylovMethod::ShiftInvert {
            relative_tolerance: 1e-10,
            max_iterations: 1024,
        },
        max_cached_components: budget,
        action_limits: limits,
        ..JointSirkOptions::default()
    };
    let started = Instant::now();
    let result = solve_joint_sirk(&h, &initial, &shifts, &options).expect("Krylov solve");
    let solve = started.elapsed().as_secs_f64();
    let coefficients = result.time_evolve(t).expect("finite time");
    let evolved = result.reconstruct(&coefficients).expect("reconstruction");
    let probability: f64 =
        fock_sirk::joint_sirk::joint_inner_product(&initial, &evolved).norm_sqr();

    println!(
        "cutoff={cutoff} labels={} matvec_ms={:.3} axpy_ms={:.4} dot_ms={:.4} solve_s={:.3} iterations={} probability={:.17e}",
        window.len(),
        matvec * 1e3,
        axpy * 1e3,
        dot * 1e3,
        solve,
        result.total_solve_iterations(),
        probability
    );
    // The recorded sweep cell is not only the Krylov solve: it also evaluates
    // the exact window reference (one action application per label, then the
    // seed column of `exp(-i t H)`). Time both halves so "what dominates the
    // cell at this cutoff" has a measured answer, not an inference.
    let started = Instant::now();
    let reference = WindowOperator::build(&h, &window);
    let build = started.elapsed().as_secs_f64();
    let seed_index = reference.seed_index(&initial);
    let started = Instant::now();
    let probability = reference.exp_action_column(seed_index, t)[seed_index].norm_sqr();
    let action = started.elapsed().as_secs_f64();
    println!(
        "reference build_s={build:.3} expm_s={action:.3} total_s={:.3} probability={probability:.17e}",
        build + action
    );

    for (threads, pdot, paxpy, worst) in parallel {
        if threads == 0 {
            println!("map_clone_ms={:.4}", paxpy * 1e3);
            continue;
        }
        println!(
            "threads={threads} parallel_dot_ms={:.4} parallel_axpy_ms={:.4} dot_rel_deviation={:.2e}",
            pdot * 1e3,
            paxpy * 1e3,
            worst
        );
    }
}

/// Dot product over worker chunks, partials summed in chunk order. This is a
/// *diagnostic*: it is not bit-identical to `joint_inner_product`, which is why
/// the production path does not use it.
fn parallel_dot(a: &NestedVector, b: &NestedVector, threads: usize) -> C {
    if threads <= 1 {
        return joint_inner_product(a, b);
    }
    let entries: Vec<(&JointOuterBasis, &C)> = a.iter().collect();
    let chunk = entries.len().div_ceil(threads);
    let partials: Vec<C> = std::thread::scope(|scope| {
        let handles: Vec<_> = entries
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut sum = C::new(0.0, 0.0);
                    for (label, value) in part {
                        if let Some(other) = b.get(*label) {
                            sum += value.conj() * other;
                        }
                    }
                    sum
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    partials
        .into_iter()
        .fold(C::new(0.0, 0.0), |acc, x| acc + x)
}

/// `target += scale * source` split across workers by label range: each worker
/// computes its chunk's updated values as an owned list (which is why the
/// labels have to be cloned), then the parent applies them in label order.
fn parallel_axpy(
    target: &mut NestedVector,
    source: &NestedVector,
    threads: usize,
    limits: ActionLimits,
) {
    if threads <= 1 {
        joint_axpy(target, source, C::new(-0.7, 0.3), limits).expect("finite axpy");
        return;
    }
    let entries: Vec<(&JointOuterBasis, &C)> = source.iter().collect();
    let chunk = entries.len().div_ceil(threads);
    let chunks: Vec<Vec<(JointOuterBasis, C)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = entries
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|(label, value)| ((*label).clone(), C::new(-0.7, 0.3) * **value))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (label, delta) in chunks.into_iter().flatten() {
        let value = target.get(&label).copied().unwrap_or_default() + delta;
        if value == C::new(0.0, 0.0) {
            target.remove(&label);
        } else {
            target.insert(label, value);
        }
    }
}
