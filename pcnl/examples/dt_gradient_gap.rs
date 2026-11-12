//! §R7.3.6 — exact vs stopped DT gradient on the decision transformer, measured.
//!
//! A directional central difference of `dt_loss` along a fixed seeded unit
//! direction, against `g·u` from `dt_loss_and_grad` under both conventions, at
//! three parameter-norm scales (`‖θ‖ = ρ·‖θ₀‖` for ρ ∈ {0.1, 1, 10}) and three
//! step sizes. Both variants are run, because the two conventions are only
//! distinguished where the forward pass normalises by a Frobenius norm.
//!
//! Measured (seed 7, 480 parameters, 96 scored transitions):
//!
//! * **Standard:** the Exact and Stopped derivatives are the *same tensor* —
//!   `|g_e|/|g_s| = 1.0000` and the directional values agree to every printed
//!   digit. This variant has no Frobenius normalisation in its forward pass, so
//!   the convention has nothing to act on; the floor of this column is
//!   structural, not a small numerical gap.
//! * **Cosine:** Exact matches the finite difference (best relative error
//!   7.4e-6 at ρ = 10, h = 1e-2; ≤ 6e-4 at h = 1e-2 at all scales), while
//!   Stopped is off by 2.2e-2 to 4.0e-2 in the *same direction* — a 2–4 %
//!   error that no gradient check of the exact path would reveal.
//! * The *norms* of the two gradients agree to four digits (ratio 0.9997–0.9999)
//!   while the directional derivatives differ by 2–4 %: the gap is almost
//!   entirely a rotation of the update, not a rescaling. A report that checked
//!   only `‖g‖` would call the two conventions identical.
//!
//! The finite difference is f32, so the reported error grows again at h = 1e-4
//! (cancelation), which is why several step sizes are printed rather than one
//! "good" one.
//!
//! ```text
//! cargo run --release -p pcnl --example dt_gradient_gap
//! ```

use candle_core::{Device, Tensor};
use pcnl::mdp::{BehaviourPolicy, Corpus, DtSpec, MdpEnv};
use pcnl::model::Variant;
use pcnl::normalize::NormGrad;
use pcnl::policy_model::{dt_loss, dt_loss_and_grad, init_flat};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn unit(len: usize, seed: u64) -> Tensor {
    let mut rng = StdRng::seed_from_u64(seed);
    let raw: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let n = raw
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    let v: Vec<f32> = raw.iter().map(|v| (f64::from(*v) / n) as f32).collect();
    Tensor::from_vec(v, len, &Device::Cpu).expect("cpu")
}

fn l2(t: &Tensor) -> f64 {
    t.sqr()
        .and_then(|s| s.sum_all())
        .and_then(|s| s.sqrt())
        .and_then(|s| s.to_scalar::<f32>())
        .map(f64::from)
        .expect("norm")
}

fn dot(a: &Tensor, b: &Tensor) -> f64 {
    (a * b)
        .and_then(|p| p.sum_all())
        .and_then(|s| s.to_scalar::<f32>())
        .map(f64::from)
        .expect("dot")
}

fn main() -> candle_core::Result<()> {
    let env = MdpEnv::default();
    let corpus = Corpus::with_seeds(&env, BehaviourPolicy::default(), &[11, 12, 13, 14, 15, 16]);
    let batch = corpus.to_dt_batch(&DtSpec::default())?;
    let shapes = env.dt_shapes(4, 6, 12, 1);
    let base = init_flat(&shapes, 0.1, 7);
    let p = base.elem_count();
    let base_norm = l2(&base);
    let dir = unit(p, 21);
    // The loss at the point the gap is measured, so a reader can tell a large
    // relative gradient error from a large gradient on a small loss.
    println!(
        "DT {p} params, {} transitions, base ||theta|| {base_norm:.6}",
        batch.transitions()
    );
    // Both variants, because the convention only *does* anything where the
    // forward pass normalises by a Frobenius norm: `Variant::Standard` has no
    // such normalisation, so its Exact and Stopped derivatives are the same
    // tensor and the gap is structurally zero rather than numerically small.
    // The Cosine arm is the one this measurement is actually about.
    for variant in [Variant::Standard, Variant::Cosine] {
        println!("\n{variant:?}");
        println!(
            "scale  ||theta||        h        FD(g.u)      exact(g.u)   rel.err    stopped(g.u)   stop.err   |g_e|/|g_s|      loss"
        );
        for rho in [0.1f64, 1.0, 10.0] {
            let target = rho * base_norm;
            let theta = base.affine(target / base_norm, 0.0)?;
            let norm = l2(&theta);
            for h in [1e-2f32, 1e-3, 1e-4] {
                let up = (&theta + dir.affine(h as f64, 0.0)?)?;
                let dn = (&theta - dir.affine(h as f64, 0.0)?)?;
                let fd = (dt_loss(&up, &shapes, variant, &batch)?
                    - dt_loss(&dn, &shapes, variant, &batch)?)
                    / (2.0 * f64::from(h));
                let (loss, g_exact) =
                    dt_loss_and_grad(&theta, &shapes, variant, &batch, NormGrad::Exact)?;
                let (_, g_stop) =
                    dt_loss_and_grad(&theta, &shapes, variant, &batch, NormGrad::Stopped)?;
                let de = dot(&g_exact, &dir);
                let ds = dot(&g_stop, &dir);
                let rel = (fd - de).abs() / de.abs().max(1e-12);
                let srel = (fd - ds).abs() / ds.abs().max(1e-12);
                let ratio = l2(&g_exact) / l2(&g_stop);
                println!(
                    "{rho:>5} {norm:>13.6} {h:>9.0e} {fd:>14.8} {de:>14.8} {rel:>10.2e} {ds:>14.8} {srel:>10.2e} {ratio:>12.4} {loss:>10.6}"
                );
            }
        }
    }
    Ok(())
}
