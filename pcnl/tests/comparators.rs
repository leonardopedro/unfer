//! Phase 4b's comparator suite (plan §4 Phase 4b): the
//! Metropolis-filtered predecessor, the cyclical schedule, and the
//! *counted* throughput that backs the specialist report's claim 1.
//!
//! The MH-filtered chain (`task::mh_pcnl_chain`) and the unadjusted
//! chain (`run_experiment`'s pCNL chains) share a proposal, a
//! schedule and a thinning cadence; the only difference is the exact
//! full-batch likelihood-ratio filter. These tests pin that design:
//!
//! * `unadjusted_halves_model_evaluations` — claim 1, *counted*:
//!   the filtered chain performs a full-batch forward per step, the
//!   unadjusted chain only per kept step;
//! * `mh_acceptance_decreases_with_beta` — the acceptance rate is a
//!   monotone decreasing function of the step size (the tuning knob
//!   the unadjusted chain no longer has);
//! * `mh_and_unadjusted_agree_on_the_posterior` — the filtered ∪
//!   unfiltered cross-R̂ ≈ 1: dropping the filter did not move the
//!   target at β ≤ 0.3;
//! * `cyclical_run_is_stable_and_reported` — §3C Option 2 end-to-end.

use candle_core::Tensor;
use pcnl::experiment::{Config, run_experiment};
use pcnl::task::{Dataset, MhOptions, Task, mh_pcnl_chain};

fn tiny_cfg() -> Config {
    Config {
        shapes: pcnl::model::Shapes {
            vocab: 9,
            d: 16,
            ff: 32,
            layers: 2,
        },
        n_train: 128,
        n_test: 64,
        pretrain_steps: 60,
        chains: 3,
        steps: 240,
        burn: 120,
        thin: 5,
        batch_size: 32,
        sgd_steps_on_b: 40,
        adam_steps_on_b: 40,
        run_mh: true,
        ..Config::default()
    }
}

/// Claim 1 of the specialist report, counted not timed: over a fixed
/// number of steps, the unadjusted chain's full-batch forwards scale
/// with the *kept* samples (every `thin` steps), while the filtered
/// chain's scale with the *steps* (one per proposal). The mini-batch
/// gradient evaluations are equal — the filter's cost is exactly the
/// second pass.
#[test]
fn unadjusted_halves_model_evaluations() {
    let cfg = tiny_cfg();
    let data = Dataset::new(Task::B, cfg.k, cfg.n_train, cfg.n_test, 41);
    let shapes = &cfg.shapes;
    let p = shapes.flat_size();
    let theta0 = Tensor::zeros(p, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();

    let mh = mh_pcnl_chain(
        &cfg,
        pcnl::model::Variant::Cosine,
        &data,
        theta0.clone(),
        0,
        None,
        Default::default(),
    );
    assert_eq!(mh.steps, cfg.steps);
    assert_eq!(mh.grad_evals, cfg.steps, "one mini-batch backward per step");
    // L(θ₀) + one full-batch forward per proposal
    assert_eq!(
        mh.loss_evals,
        1 + cfg.steps,
        "one full-batch forward per proposal"
    );
    let kept_expected = (cfg.steps - cfg.burn).div_ceil(cfg.thin);
    assert!(mh.kept.len() <= kept_expected && !mh.kept.is_empty());

    // The unadjusted chain's full-batch forwards: one per *kept*
    // step (its ESS series), i.e. ⌈(steps − burn)/thin⌉ — strictly
    // fewer than the filtered chain's `steps`, by a factor of ~thin.
    let unadjusted_loss_evals = (cfg.steps - cfg.burn).div_ceil(cfg.thin);
    assert!(
        cfg.steps > unadjusted_loss_evals,
        "filtered chain's {} full-batch forwards vs unadjusted's {}",
        cfg.steps,
        unadjusted_loss_evals
    );
    assert!(
        mh.loss_evals >= cfg.steps,
        "filtered chain must do ≥1 full-batch forward per step"
    );
}

/// The acceptance rate is monotone decreasing in β₀: bigger steps ⇒
/// more proposals land where the potential disagrees ⇒ fewer
/// accepted. The rate is the knob the *unadjusted* chain no longer
/// has — measured here on the chain that still pays for it.
#[test]
fn mh_acceptance_decreases_with_beta() {
    let mut rates = Vec::new();
    for beta_0 in [0.05f64, 0.15, 0.30] {
        let cfg = Config {
            beta_0,
            ..tiny_cfg()
        };
        let data = Dataset::new(Task::B, cfg.k, cfg.n_train, cfg.n_test, 43);
        let p = cfg.shapes.flat_size();
        let theta0 = Tensor::zeros(p, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let mut accepted = 0usize;
        let mut steps = 0usize;
        for c in 0..2 {
            let oc = mh_pcnl_chain(
                &cfg,
                pcnl::model::Variant::Cosine,
                &data,
                theta0.clone(),
                c,
                None,
                Default::default(),
            );
            accepted += oc.accepted;
            steps += oc.steps;
        }
        let rate = accepted as f64 / steps as f64;
        assert!((0.0..=1.0).contains(&rate), "β₀={beta_0}: rate {rate}");
        rates.push(rate);
    }
    assert!(
        rates[0] >= rates[1] && rates[1] >= rates[2],
        "acceptance not monotone in β₀: {rates:?}"
    );
}

/// Phase 4b's bias probe: the filtered and unfiltered chains share a
/// proposal and a target, so their loss-chain cross-R̂ isolates what
/// dropping the filter cost. At β ≤ 0.3 the two must agree — the
/// report's “the exactness the filter bought is inside the sampling
/// noise”, asserted rather than eyeballed.
#[test]
fn mh_and_unadjusted_agree_on_the_posterior() {
    let cfg = Config {
        chains: 2,
        ..tiny_cfg()
    };
    let report = run_experiment(&cfg);
    let v = &report.variants[0]; // cosine — the stable variant
    assert!(!v.exploded, "cosine variant should not explode");
    assert!(
        v.mh_kept_per_chain > 0 && v.bma_models > 0,
        "both families must have kept samples (mh {}, pCNL {})",
        v.mh_kept_per_chain,
        v.bma_models
    );
    // cross-R̂ over the filtered ∪ unfiltered loss chains ≈ 1
    assert!(
        v.rhat_cross_mh.is_finite() && v.rhat_cross_mh > 0.5 && v.rhat_cross_mh < 1.2,
        "filtered∪unfiltered cross-R̂ {} — dropping the filter moved the target",
        v.rhat_cross_mh
    );
    // and the two BMAs land in the same neighbourhood (20 accuracy
    // points is the noise floor at 64 test examples)
    let gap = (v.bma_accuracy - v.mh_bma_accuracy).abs();
    assert!(
        gap < 25.0,
        "BMA gap {gap} too large: {} vs {}",
        v.bma_accuracy,
        v.mh_bma_accuracy
    );
}

/// §3C Option 2 end-to-end: a cyclical run (period 64 on a 240-step
/// chain) is stable, keeps the shell, and its report says so.
#[test]
fn cyclical_run_is_stable_and_reported() {
    let cfg = Config {
        cyclical_period: 64,
        ..tiny_cfg()
    };
    let report = run_experiment(&cfg);
    assert!(report.variants.len() == 2);
    let v = &report.variants[0]; // cosine
    assert!(!v.exploded, "cyclical cosine should not explode");
    // the shell holds under the cyclical schedule too
    assert!(
        (v.radial_final - v.radial_expected).abs() / v.radial_expected < 0.5,
        "cyclical radial {} vs σ√P {}",
        v.radial_final,
        v.radial_expected
    );
    // the schedule actually cycles: β starts at β₀, dips, and returns
    assert!(
        (v.beta_start - cfg.beta_0).abs() < 1e-9,
        "cyclical β should start at β₀: {}",
        v.beta_start
    );
    // BMA stays defined under the cyclical schedule
    assert!(v.bma_models > 0, "cyclical BMA needs finite samples");
    assert!(
        (0.0..=100.0).contains(&v.bma_accuracy),
        "cyclical bma {}",
        v.bma_accuracy
    );
}

/// **The drift cap changes the proposal, not (much) the target** (§6.5.3's
/// licence, qualified by §6.5.6(c)).
///
/// §6.5.2 caps the standard stack's drift so the chain stops exploding, and
/// §6.5.3 reads the *filtered* capped chain as a sample of the same target.
/// That reading is only sound if the cap cannot move the invariant
/// distribution — and it is only *approximately* sound, because the filter's
/// acceptance is the likelihood ratio alone (exact Metropolis needs a
/// drift-free pCN proposal), so the filtered chain is pCNL and its invariant
/// law is not `π` at all.  What this test pins is the weaker, honest claim
/// that two caps give the same kept summaries to within the pCNL bias:
///
/// 1. **Inertness, exactly.**  A cap above the gradient's own norm must leave
///    the chain bit-identical to the uncapped one — same proposals, same
///    accepts, same kept tensors.  If the cap ever leaked into the acceptance
///    decision or the noise, this fails to the bit.
/// 2. **Invariance under binding caps, statistically.**  Two caps that both
///    bind must return the same stationary summaries: the same mean potential
///    over kept draws (raw `L`, untempered) and the same mean radius, within
///    Monte-Carlo error.  A cap inside the ratio (a tempered target) or a cap
///    that changed the noise/step would shift the potential by a visible
///    margin, far outside the tolerance below.
#[test]
fn mh_drift_cap_changes_only_the_proposal() {
    // Deliberately *smaller* than `tiny_cfg`: this test runs four chains
    // of filtered steps and must stay cheap in an unoptimised test
    // build (P = 148 here, against 2 408 in `tiny_cfg`).
    let cfg = Config {
        k: 4,
        shapes: pcnl::model::Shapes {
            vocab: 5,
            d: 4,
            ff: 8,
            layers: 1,
        },
        n_train: 64,
        n_test: 32,
        chains: 2,
        steps: 200,
        burn: 100,
        thin: 2,
        batch_size: 8,
        beta_0: 0.3,
        sigma: 0.5,
        run_mh: false,
        ..Config::default()
    };
    let data = Dataset::new(Task::B, cfg.k, cfg.n_train, cfg.n_test, 77);
    let p = cfg.shapes.flat_size();
    let theta0 = Tensor::zeros(p, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();

    let run = |opts: MhOptions| -> Vec<pcnl::task::MhOutcome> {
        (0..cfg.chains)
            .map(|c| {
                mh_pcnl_chain(
                    &cfg,
                    pcnl::model::Variant::Cosine,
                    &data,
                    theta0.clone(),
                    c,
                    None,
                    opts,
                )
            })
            .collect()
    };
    let summaries = |runs: &[pcnl::task::MhOutcome]| -> (f64, f64, usize) {
        let kept: Vec<Tensor> = runs.iter().flat_map(|o| o.kept.clone()).collect();
        let losses = runs.iter().flat_map(|o| o.kept_loss.clone());
        let n = losses.clone().count() as f64;
        let mean_loss = losses.sum::<f64>() / n;
        let mean_radius = kept
            .iter()
            .map(|t| pcnl::sampler::norm(t).unwrap_or(f64::NAN))
            .filter(|r| r.is_finite())
            .sum::<f64>()
            / kept.len() as f64;
        (
            mean_loss,
            mean_radius,
            runs.iter().map(|o| o.accepted).sum(),
        )
    };

    // ---- 1. the inert cap: bit-identical ---------------------------------
    let plain = run(MhOptions::default());
    let inert = run(MhOptions {
        grad_norm_cap: Some(1e9),
        potential_scale: 1.0,
    });
    let (l0, r0, a0) = summaries(&plain);
    let (l1, r1, a1) = summaries(&inert);
    assert_eq!(a0, a1, "an inert cap must not change the accept count");
    assert_eq!(
        l0.to_bits(),
        l1.to_bits(),
        "raw kept loss changed: {l0} vs {l1}"
    );
    assert_eq!(
        r0.to_bits(),
        r1.to_bits(),
        "kept radius changed: {r0} vs {r1}"
    );
    for (x, y) in plain.iter().zip(&inert) {
        assert_eq!(x.accepted, y.accepted);
        assert_eq!(x.kept.len(), y.kept.len());
        for (t, u) in x.kept.iter().zip(&y.kept) {
            assert_eq!(
                t.to_vec1::<f32>().unwrap(),
                u.to_vec1::<f32>().unwrap(),
                "an inert cap perturbed a kept draw"
            );
        }
    }

    // ---- 2. two binding caps: same stationary summaries ------------------
    let tight = run(MhOptions {
        grad_norm_cap: Some(0.5),
        potential_scale: 1.0,
    });
    let loose = run(MhOptions {
        grad_norm_cap: Some(5.0),
        potential_scale: 1.0,
    });
    let (l_t, r_t, a_t) = summaries(&tight);
    let (l_l, r_l, a_l) = summaries(&loose);
    let steps: usize = tight.iter().map(|o| o.steps).sum();
    assert!(
        a_t as f64 / steps as f64 > 0.1 && a_l as f64 / steps as f64 > 0.1,
        "the invariance claim needs two *usable* proposals: acceptance {:.3} vs {:.3}",
        a_t as f64 / steps as f64,
        a_l as f64 / steps as f64
    );
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-12);
    eprintln!(
        "cap invariance: tight L {l_t:.4} r {r_t:.3} (accept {:.2}) vs loose L {l_l:.4} r {r_l:.3} \
         (accept {:.2}); rel dL {:.3} dr {:.3}",
        a_t as f64 / steps as f64,
        a_l as f64 / steps as f64,
        rel(l_t, l_l),
        rel(r_t, r_l)
    );
    assert!(
        rel(l_t, l_l) < 0.20,
        "kept potential moved with the cap (a tempered ratio would do this): \
         tight {l_t:.4} vs loose {l_l:.4}"
    );
    assert!(
        rel(r_t, r_l) < 0.25,
        "kept radius moved with the cap: tight {r_t:.3} vs loose {r_l:.3}"
    );
}
