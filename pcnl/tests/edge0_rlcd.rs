//! Acceptance gates E0, E1 and E2 of `EDGE0_RLCD_PLAN.md`.
//!
//! These are the "first falsifiers" the plan names, not campaign results:
//!
//! * **E0** — an independently enumerated world agrees with the oracle; missing
//!   observations cannot lower conditional entropy on average; correlated
//!   question marginals are not multiplied into a false joint.
//! * **E1** — stable normalized finite probabilities, explicit schema failures,
//!   packed/separate agreement, choice/question permutation equivariance, an
//!   ordered Score head, a dense-routing reference and a complete parameter
//!   census below one million.
//! * **E2** — the exhaustive finite-support expectation of the sampled-path
//!   gradient equals the exact marginal-objective derivative at nonuniform
//!   policies and readouts (policy, direct readout and shared parameters),
//!   verified both on a finite fixture against finite differences and through
//!   candle autodiff on the actual model.

use candle_core::{Device, Tensor, Var};
use pcnl::edge0_campaign::{Arm, CampaignConfig, build_samples, run};
use pcnl::edge0_model::{
    Edge0Config, ModelInput, QuestionInput, Routing, StateCacheKey, census, forward, forward_with,
    init, ordinal_probs,
};
use pcnl::edge0_potential::{Edge0Potential, reference_theta};
use pcnl::edge0_rlcd::{
    FiniteFixture, PATHS, Streams, all_tuples, brier, brier_u_statistic, correctness_reward,
    expected_brier, expected_u_statistic, marginal, tuple_probability,
};
use pcnl::edge0_world::{
    QuestionKind, Split, UtilityTable, World, WorldManifest, brier_excess, entropy,
};
use pcnl::target::Potential;

fn dev() -> &'static Device {
    static D: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    D.get_or_init(|| Device::Cpu)
}

// ---------------------------------------------------------------------------
// E0 — the probability oracle
// ---------------------------------------------------------------------------

#[test]
fn manifest_splits_are_disjoint_and_ids_precede_questions() {
    let manifest = WorldManifest::new(20_261_115, 64, 16, 16, 32);
    manifest.assert_disjoint();
    let mut seen = std::collections::BTreeSet::new();
    for split in Split::ALL {
        let ids: Vec<u32> = manifest.split(split).map(|w| w.id).collect();
        assert!(!ids.is_empty(), "split {} is empty", split.label());
        for id in ids {
            assert!(seen.insert((split, id)), "world {id} appears twice");
        }
    }
    assert_eq!(seen.len(), manifest.worlds().len());
    // Families never span splits.
    for w in manifest.worlds() {
        assert_eq!(manifest.family_owner(w.family), w.split);
    }
    // Question IDs are derived after (and from) the world list.
    let w = manifest.worlds()[0];
    assert_ne!(
        manifest.question_id(&w, 0),
        manifest.question_id(&w, 1),
        "question ids must distinguish positions"
    );
    let other = manifest.worlds()[1];
    assert_ne!(
        manifest.question_id(&w, 0),
        manifest.question_id(&other, 0),
        "question ids must distinguish worlds"
    );
    // Content hash identifies the materialised list, not just the seed.
    let again = WorldManifest::new(20_261_115, 64, 16, 16, 32);
    assert_eq!(manifest.content_hash(), again.content_hash());
    let different = WorldManifest::new(7, 64, 16, 16, 32);
    assert_ne!(manifest.content_hash(), different.content_hash());
}

#[test]
fn oracle_matches_independent_enumeration() {
    let manifest = WorldManifest::new(11, 4, 2, 2, 2);
    let mut checked = 0usize;
    for spec in manifest.worlds().iter().take(6) {
        let world = World::new(*spec, manifest.seed());
        for (index, kind) in [
            QuestionKind::Noul,
            QuestionKind::Choice { options: 5 },
            QuestionKind::Score { levels: 4 },
        ]
        .iter()
        .enumerate()
        {
            let q = world.question(*kind, index as u32).unwrap();
            // Reveal one true feature.
            let obs = [(0usize, world.hypotheses[world.truth].features[0])];
            // Independent enumeration: filter hypotheses by hand, mix their
            // laws with the uniform prior.
            let mut weights = vec![0.0f64; world.hypotheses.len()];
            for (i, h) in world.hypotheses.iter().enumerate() {
                if h.features[obs[0].0] == obs[0].1 {
                    weights[i] += h.prior;
                }
            }
            let total: f64 = weights.iter().sum();
            let mut expected = vec![0.0; kind.outcomes()];
            for (i, w) in weights.iter().enumerate() {
                if *w == 0.0 {
                    continue;
                }
                let law = world.hypothesis_law(&q, i);
                for (o, p) in expected.iter_mut().zip(&law) {
                    *o += (*w / total) * p;
                }
            }
            let oracle = world.conditional_law(&q, &obs);
            let norm: f64 = oracle.iter().sum();
            assert!((norm - 1.0).abs() < 1e-12, "law does not normalise");
            for (a, b) in oracle.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-12, "oracle {oracle:?} vs {expected:?}");
            }
            // Nondegenerate: the law must not be a point mass.
            assert!(
                oracle.iter().filter(|p| **p > 1e-6).count() >= 2,
                "the conditional law collapsed: {oracle:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 18);
}

#[test]
fn more_observations_never_raise_conditional_entropy() {
    let manifest = WorldManifest::new(99, 8, 4, 4, 4);
    let questions = [
        QuestionKind::Noul,
        QuestionKind::Choice { options: 4 },
        QuestionKind::Score { levels: 4 },
    ];
    let (mut h1, mut h2, mut n) = (0.0f64, 0.0f64, 0u32);
    for spec in manifest.worlds().iter().take(24) {
        let world = World::new(*spec, manifest.seed());
        let truth = &world.hypotheses[world.truth];
        for (i, kind) in questions.iter().enumerate() {
            let q = world.question(*kind, i as u32).unwrap();
            let one = [(0usize, truth.features[0])];
            let two = [(0usize, truth.features[0]), (1usize, truth.features[1])];
            h1 += entropy(&world.conditional_law(&q, &one));
            h2 += entropy(&world.conditional_law(&q, &two));
            n += 1;
        }
    }
    let (mean1, mean2) = (h1 / n as f64, h2 / n as f64);
    assert!(
        mean2 <= mean1 + 1e-12,
        "conditioning on more observations raised mean entropy: {mean2} > {mean1}"
    );
    // The fixture must be nondegenerate: conditioning has to do something.
    assert!(mean1 - mean2 > 1e-6, "observations carried no information");
}

#[test]
fn correlated_questions_are_not_a_product_of_marginals() {
    let manifest = WorldManifest::new(5, 2, 1, 1, 1);
    let world = World::new(manifest.worlds()[0], manifest.seed());
    // Two questions tied to the same shared hidden hypothesis.
    let q1 = world
        .question(QuestionKind::Choice { options: 3 }, 0)
        .unwrap();
    let q2 = world
        .question(QuestionKind::Choice { options: 3 }, 1)
        .unwrap();
    let joint = world.joint_law(&q1, &q2, &[]);
    let m1 = world.conditional_law(&q1, &[]);
    let m2 = world.conditional_law(&q2, &[]);
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    for (i, row) in joint.iter().enumerate() {
        let row_sum: f64 = row.iter().sum();
        total += row_sum;
        for (j, p) in row.iter().enumerate() {
            worst = worst.max((p - m1[i] * m2[j]).abs());
        }
        // The joint's row sums reproduce the first marginal, and the joint
        // column sums reproduce the second.
        assert!((row_sum - m1[i]).abs() < 1e-9);
    }
    assert!((total - 1.0).abs() < 1e-9);
    for j in 0..m2.len() {
        let col: f64 = joint.iter().map(|row| row[j]).sum();
        assert!((col - m2[j]).abs() < 1e-9);
    }
    assert!(
        worst > 1e-3,
        "the questions look independent (max gap {worst}); the fixture cannot \
         detect a false product"
    );
}

#[test]
fn utility_decisions_and_regret_bound() {
    let truth = [0.2, 0.5, 0.3];
    let predicted = [0.3, 0.4, 0.3];
    let table = UtilityTable::declared(7, QuestionKind::Choice { options: 3 });
    let opt = table.optimal_action(&truth);
    assert_eq!(table.regret(&truth, opt), 0.0);
    let bound = UtilityTable::regret_bound(&predicted, &truth);
    // The plan's finite-action stability bound: for utilities in [0,1], the
    // action chosen from the predicted law loses at most 2*TV against the
    // truth-optimal action.
    let chosen = table.optimal_action(&predicted);
    assert!(
        table.regret(&truth, chosen) <= bound + 1e-12,
        "regret {} exceeds the 2*TV bound {bound}",
        table.regret(&truth, chosen)
    );
    assert!(table.utilities.iter().all(|u| (0.0..=1.0).contains(u)));
    // A uniformly worse action cannot have negative regret.
    for a in 0..3 {
        assert!(table.regret(&truth, a) >= 0.0);
    }
}

// ---------------------------------------------------------------------------
// E1 — the model and its typed interface
// ---------------------------------------------------------------------------

fn fixture_question(kind: QuestionKind) -> QuestionInput {
    match kind {
        QuestionKind::Noul => QuestionInput {
            kind,
            tokens: vec![3, 4, 5],
            options: vec![],
        },
        QuestionKind::Choice { options } => QuestionInput {
            kind,
            tokens: vec![6, 7],
            options: (0..options as u32).map(|i| 10 + i).collect(),
        },
        QuestionKind::Score { levels: _ } => QuestionInput {
            kind,
            tokens: vec![8, 9],
            options: vec![],
        },
    }
}

fn fixture_input(kinds: &[QuestionKind]) -> ModelInput<'static> {
    // Leaked solely to keep the fixture's borrow simple in a test binary.
    let questions: &'static [QuestionInput] = Box::leak(
        kinds
            .iter()
            .map(|k| fixture_question(*k))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let state: &'static [u32] = Box::leak(vec![1u32, 2, 3, 2, 1].into_boxed_slice());
    ModelInput { state, questions }
}

#[test]
fn parameter_census_is_complete_and_below_one_million() {
    let config = Edge0Config::default();
    let c = census(&config);
    assert_eq!(c.total, config.flat_size());
    let sum: usize = c.per_component.iter().map(|(_, n)| n).sum();
    assert_eq!(
        sum, c.total,
        "census components must account for every tensor"
    );
    assert!(
        c.total < 1_000_000,
        "R9 requires < 1e6 parameters, got {}",
        c.total
    );
    assert!(c.routed_active > 0 && c.routed_active < c.total);
    // The tiny fixture shares the structure, so its training runs cost less.
    let tiny = census(&Edge0Config::tiny());
    assert!(tiny.total < c.total);
}

#[test]
fn typed_outputs_are_finite_normalized_and_schema_failures_surface() {
    let config = Edge0Config::tiny();
    let theta = init(&config, 42);
    let kinds = [
        QuestionKind::Noul,
        QuestionKind::Choice { options: 4 },
        QuestionKind::Score { levels: 3 },
    ];
    let input = fixture_input(&kinds);
    let out = forward(&theta, &config, &input).unwrap();
    assert_eq!(out.per_question.len(), 3);
    assert_eq!(out.state_rep.dims(), &[config.d]);
    for (q, kind) in out.per_question.iter().zip(&kinds) {
        let policy = q.policy_vec().unwrap();
        assert!((policy.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(policy.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert_eq!(q.path_probs.len(), config.paths);
        let marginal = q.marginal_vec().unwrap();
        assert_eq!(marginal.len(), kind.outcomes());
        assert!(
            (marginal.iter().sum::<f64>() - 1.0).abs() < 1e-6,
            "marginal does not normalise: {marginal:?}"
        );
        assert!(marginal.iter().all(|p| p.is_finite() && *p >= 0.0));
    }
    // An empty schema is an explicit error, never a uniform fallback.
    let bad = QuestionInput {
        kind: QuestionKind::Choice { options: 0 },
        tokens: vec![1],
        options: vec![],
    };
    let bad_input = ModelInput {
        state: &[1, 2],
        questions: &[bad],
    };
    assert!(forward(&theta, &config, &bad_input).is_err());
    // A token-limit violation is an error, not silent truncation.
    let long = QuestionInput {
        kind: QuestionKind::Noul,
        tokens: vec![1; config.max_question_tokens + 1],
        options: vec![],
    };
    let long_input = ModelInput {
        state: &[1, 2],
        questions: &[long],
    };
    assert!(forward(&theta, &config, &long_input).is_err());
}

#[test]
fn packed_and_separate_questions_agree() {
    let config = Edge0Config::tiny();
    let theta = init(&config, 7);
    let kinds = [
        QuestionKind::Noul,
        QuestionKind::Choice { options: 3 },
        QuestionKind::Score { levels: 4 },
    ];
    let packed = fixture_input(&kinds);
    let packed_out = forward(&theta, &config, &packed).unwrap();
    for (i, kind) in kinds.iter().enumerate() {
        let single = fixture_input(std::slice::from_ref(kind));
        let single_out = forward(&theta, &config, &single).unwrap();
        let a = packed_out.per_question[i].marginal_vec().unwrap();
        let b = single_out.per_question[0].marginal_vec().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "packed {a:?} vs separate {b:?}");
        }
        let state_diff: f64 = packed_out
            .state_rep
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(single_out.state_rep.to_vec1::<f32>().unwrap())
            .map(|(x, y)| (*x as f64 - y as f64).abs())
            .sum();
        assert!(state_diff < 1e-5, "state encoding depends on the question");
    }
}

#[test]
fn question_order_and_choice_permutation_equivariance() {
    let config = Edge0Config::tiny();
    let theta = init(&config, 13);
    let kinds = [
        QuestionKind::Noul,
        QuestionKind::Choice { options: 3 },
        QuestionKind::Score { levels: 3 },
    ];
    let forward_order = forward(&theta, &config, &fixture_input(&kinds)).unwrap();
    let reversed: Vec<QuestionKind> = kinds.iter().rev().cloned().collect();
    let reverse_order = forward(&theta, &config, &fixture_input(&reversed)).unwrap();
    for i in 0..kinds.len() {
        let a = forward_order.per_question[i].marginal_vec().unwrap();
        let b = reverse_order.per_question[kinds.len() - 1 - i]
            .marginal_vec()
            .unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "question order changed an output");
        }
    }
    // Permuting declared Choice options permutes the probabilities.
    let base = QuestionInput {
        kind: QuestionKind::Choice { options: 3 },
        tokens: vec![6, 7],
        options: vec![10, 11, 12],
    };
    let permuted = QuestionInput {
        kind: QuestionKind::Choice { options: 3 },
        tokens: vec![6, 7],
        options: vec![12, 10, 11],
    };
    let a = forward(
        &theta,
        &config,
        &ModelInput {
            state: &[1, 2],
            questions: &[base],
        },
    )
    .unwrap()
    .per_question[0]
        .marginal_vec()
        .unwrap();
    let b = forward(
        &theta,
        &config,
        &ModelInput {
            state: &[1, 2],
            questions: &[permuted],
        },
    )
    .unwrap()
    .per_question[0]
        .marginal_vec()
        .unwrap();
    for (slot, source) in [(0usize, 1usize), (1, 2), (2, 0)] {
        assert!(
            (a[slot] - b[source]).abs() < 1e-6,
            "option permutation is not a permutation of probabilities"
        );
    }
}

#[test]
fn ordinal_level_order_is_a_stochastic_order() {
    let thresholds = [-0.5, 0.6, 1.7];
    let levels = 4;
    let mut previous = ordinal_probs(-2.0, &thresholds, levels);
    for step in 0..21 {
        let score = -2.0 + step as f64 * 0.2;
        let p = ordinal_probs(score, &thresholds, levels);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p.iter().all(|x| *x >= -1e-12));
        // Increasing the latent score cannot lower any upper-tail probability:
        // first-order stochastic dominance of the ordinal levels.
        for j in 1..levels {
            let ge_prev: f64 = previous[j..].iter().sum();
            let ge_now: f64 = p[j..].iter().sum();
            assert!(
                ge_now >= ge_prev - 1e-12,
                "score increase lowered P(level >= {j}): {ge_now} < {ge_prev}"
            );
        }
        // The mean level is monotone as well.
        let mean = |law: &[f64]| -> f64 { law.iter().enumerate().map(|(k, p)| k as f64 * p).sum() };
        assert!(mean(&p) >= mean(&previous) - 1e-12);
        previous = p;
    }
}

#[test]
fn dense_soft_routing_matches_top_k_when_every_expert_is_selected() {
    let mut config = Edge0Config::tiny();
    config.top_k = config.experts;
    let theta = init(&config, 21);
    let input = fixture_input(&[QuestionKind::Noul]);
    let topk = forward_with(&theta, &config, &input, Routing::TopK).unwrap();
    let dense = forward_with(&theta, &config, &input, Routing::Soft).unwrap();
    let a = topk.per_question[0].marginal_vec().unwrap();
    let b = dense.per_question[0].marginal_vec().unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!(
            (x - y).abs() < 1e-6,
            "selected-expert path disagrees with the dense reference: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn state_cache_key_covers_weights_and_inference_settings() {
    let config = Edge0Config::tiny();
    let theta = init(&config, 3);
    let state = [1u32, 2, 3];
    let base = StateCacheKey::new(&config, &theta, &state, "eval;topk=2").unwrap();
    assert_eq!(
        base,
        StateCacheKey::new(&config, &theta, &state, "eval;topk=2").unwrap()
    );
    assert_ne!(
        base,
        StateCacheKey::new(&config, &init(&config, 4), &state, "eval;topk=2").unwrap(),
        "different encoder weights must invalidate the cache"
    );
    assert_ne!(
        base,
        StateCacheKey::new(&config, &theta, &state, "train;dropout=0.1").unwrap(),
        "different inference settings must invalidate the cache"
    );
    assert_ne!(
        base,
        StateCacheKey::new(&config, &theta, &[1, 2, 4], "eval;topk=2").unwrap(),
        "different state tokens must invalidate the cache"
    );
}

// ---------------------------------------------------------------------------
// E2 — RLCD's gradient contract
// ---------------------------------------------------------------------------

#[test]
fn u_statistic_expectation_equals_marginal_brier() {
    let base = vec![
        vec![0.9, 0.1],
        vec![0.6, 0.4],
        vec![0.2, 0.8],
        vec![0.5, 0.5],
    ];
    for &(p0, p1, p2, p3) in &[
        (0.25, 0.25, 0.25, 0.25),
        (0.7, 0.1, 0.1, 0.1),
        (0.1, 0.6, 0.2, 0.1),
        (0.4, 0.3, 0.2, 0.1),
    ] {
        let raw = [p0, p1, p2, p3];
        let a: Vec<f64> = raw.iter().map(|p| p / raw.iter().sum::<f64>()).collect();
        let p = marginal(&a, &base);
        for y in 0..2 {
            let expected = expected_u_statistic(&a, &base, y);
            assert!(
                (expected - brier(&p, y)).abs() < 1e-12,
                "E[B_U] = {expected} but B(p,y) = {}",
                brier(&p, y)
            );
        }
    }
    // The finite-draw estimator is not the deployed score: it is a draw-level
    // quantity whose expectation (not its value) is the marginal Brier score.
    let u = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];
    let p = marginal(&[0.5, 0.5, 0.0, 0.0], &u);
    for y in 0..2 {
        let expected = expected_u_statistic(&[0.5, 0.5, 0.0, 0.0], &u, y);
        assert!((expected - brier(&p, y)).abs() < 1e-12);
    }
    assert!((brier_u_statistic(&u, 0) - brier(&p, 0)).abs() > 1e-3);
}

#[test]
fn estimator_gradient_equals_exact_and_finite_difference_gradients() {
    let fixture = FiniteFixture::new(vec![
        vec![0.8, 0.2],
        vec![0.55, 0.45],
        vec![0.2, 0.8],
        vec![0.7, 0.3],
    ]);
    for &(l0, l1, l2, l3, s) in &[
        (0.3, -0.2, 0.1, 0.0, 1.0),
        (1.1, 0.2, -0.6, 0.4, 0.7),
        (-0.5, 0.9, 0.3, -0.2, 1.3),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    ] {
        let logits = [l0, l1, l2, l3];
        for y in 0..2 {
            let (exact_policy, exact_shared) = fixture.exact_gradient(&logits, s, y);
            let (est_policy, est_shared) = fixture.estimator_expectation(&logits, s, y);
            let (fd_policy, fd_shared) = fixture.finite_difference_gradient(&logits, s, y, 1e-6);
            for i in 0..4 {
                assert!(
                    (exact_policy[i] - est_policy[i]).abs() < 1e-10,
                    "policy[{i}]: exact {} vs estimator {}",
                    exact_policy[i],
                    est_policy[i]
                );
                assert!(
                    (fd_policy[i] - exact_policy[i]).abs() < 1e-6,
                    "policy[{i}]: finite difference {} vs exact {}",
                    fd_policy[i],
                    exact_policy[i]
                );
            }
            assert!(
                (exact_shared - est_shared).abs() < 1e-10,
                "shared: exact {exact_shared} vs estimator {est_shared}"
            );
            assert!(
                (fd_shared - exact_shared).abs() < 1e-6,
                "shared: finite difference {fd_shared} vs exact {exact_shared}"
            );
            // The gate must be able to see a missing score term.
            let missing_score: Vec<f64> = vec![0.0; 4];
            assert!(
                exact_policy
                    .iter()
                    .zip(&missing_score)
                    .any(|(a, b)| (a - b).abs() > 1e-4)
            );
        }
    }
}

fn grad_vec(grads: &candle_core::backprop::GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var.as_tensor())
        .expect("parameter received a gradient")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The model-level E2 gate: exhaustive expectation of the sampled-path
/// estimator through candle autodiff equals the exact marginal's gradient,
/// including policy, readout and shared-trunk parameters.
#[test]
fn model_estimator_gradient_matches_exact_marginal_gradient() {
    let config = Edge0Config::tiny();
    let theta = init(&config, 2026);
    let kind = QuestionKind::Noul;
    let input = fixture_input(&[kind]);
    let y = 1usize;

    // Exact marginal objective and its gradient.
    let exact_var = Var::from_tensor(&theta).unwrap();
    let out = forward(exact_var.as_tensor(), &config, &input).unwrap();
    let p = out.per_question[0].marginal().unwrap();
    let eye = Tensor::from_vec(
        (0..2)
            .map(|k| if k == y { 1.0f32 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![2],
        dev(),
    )
    .unwrap();
    let exact_loss = p.sub(&eye).unwrap().sqr().unwrap().sum_all().unwrap();
    let exact_grad = grad_vec(&exact_loss.backward().unwrap(), &exact_var);
    assert_eq!(exact_grad.len(), config.flat_size());

    // Exhaustive estimator expectation over 4^4 path tuples.
    let mut expected = vec![0f64; exact_grad.len()];
    for tuple in all_tuples(PATHS, PATHS) {
        let var = Var::from_tensor(&theta).unwrap();
        let out = forward(var.as_tensor(), &config, &input).unwrap();
        let q = &out.per_question[0];
        let a = q.policy_vec().unwrap();
        let prob = tuple_probability(&a, &tuple);
        if prob == 0.0 {
            continue;
        }
        let mut sum_y = Tensor::zeros(1, candle_core::DType::F32, dev())
            .unwrap()
            .squeeze(0)
            .unwrap();
        for &r in &tuple {
            let v = q.path_probs[r].narrow(0, y, 1).unwrap().squeeze(0).unwrap();
            sum_y = sum_y.add(&v).unwrap();
        }
        let mut b_u = Tensor::ones(1, candle_core::DType::F32, dev())
            .unwrap()
            .squeeze(0)
            .unwrap();
        b_u = b_u
            .add(&sum_y.affine(-2.0 / PATHS as f64, 0.0).unwrap())
            .unwrap();
        let mut pairs = Tensor::zeros(1, candle_core::DType::F32, dev())
            .unwrap()
            .squeeze(0)
            .unwrap();
        for i in 0..PATHS {
            for j in 0..PATHS {
                if i == j {
                    continue;
                }
                let dot = q.path_probs[tuple[i]]
                    .mul(&q.path_probs[tuple[j]])
                    .unwrap()
                    .sum_all()
                    .unwrap();
                pairs = pairs.add(&dot).unwrap();
            }
        }
        b_u = b_u
            .add(
                &pairs
                    .affine(1.0 / (PATHS * (PATHS - 1)) as f64, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let logp = candle_nn::ops::log_softmax(&q.policy_logits, 0).unwrap();
        let mut score = Tensor::zeros(1, candle_core::DType::F32, dev())
            .unwrap()
            .squeeze(0)
            .unwrap();
        for &r in &tuple {
            score = score
                .add(&logp.narrow(0, r, 1).unwrap().squeeze(0).unwrap())
                .unwrap();
        }
        // Score term with a detached B_U, plus the direct differentiable term.
        let loss = b_u
            .detach()
            .mul(&score)
            .unwrap()
            .add(&b_u)
            .unwrap()
            .affine(prob, 0.0)
            .unwrap();
        let grad = grad_vec(&loss.backward().unwrap(), &var);
        for (dst, g) in expected.iter_mut().zip(grad) {
            *dst += f64::from(g);
        }
    }

    let scale = exact_grad
        .iter()
        .map(|g| (*g as f64).abs())
        .fold(0.0f64, f64::max)
        .max(1e-6);
    let mut worst = 0.0f64;
    for (a, b) in expected.iter().zip(&exact_grad) {
        worst = worst.max((a - *b as f64).abs());
    }
    assert!(
        worst < 5e-3 * scale.max(1.0),
        "estimator expectation differs from the exact marginal gradient by {worst} \
         (gradient scale {scale})"
    );
}

#[test]
fn correctness_reward_is_not_a_proper_scoring_rule() {
    // A noisy binary world: the outcome is a fair coin.
    let truth = [0.5, 0.5];
    let corner_a = [1.0, 0.0];
    let corner_b = [0.0, 1.0];
    // Both corners maximise the expected correctness reward...
    let r_corner: f64 = (0..2)
        .map(|y| truth[y] * correctness_reward(&corner_a, y))
        .sum();
    assert!((r_corner - 0.5).abs() < 1e-12);
    assert!(
        (truth[0] * correctness_reward(&corner_b, 0) + truth[1] * correctness_reward(&corner_b, 1)
            - 0.5)
            .abs()
            < 1e-12
    );
    // ...while Brier has a unique optimum at the truth.
    assert!(expected_brier(&truth, &truth) < expected_brier(&corner_a, &truth));
    assert!(expected_brier(&truth, &truth) < expected_brier(&corner_b, &truth));
    // Excess risk is zero exactly at the truth.
    assert!(brier_excess(&truth, &truth) < 1e-12);
    assert!(brier_excess(&corner_a, &truth) > 0.0);
}

#[test]
fn head_chart_exposes_the_exact_likelihood_and_freezes_upstream_parameters() {
    let cfg = CampaignConfig {
        updates: 0,
        warm_start: 0,
        worlds: [3, 1, 1, 1],
        ..Default::default()
    };
    let samples = build_samples(&cfg);
    let theta = reference_theta(&cfg.model, 5);
    let potential = Edge0Potential::head_only(cfg.model.clone(), samples);
    let base = potential.mean_loss(&theta);
    assert!(base.is_finite() && base > 0.0);
    let grad = potential
        .mean_grad(&theta)
        .expect("the chart has a finite gradient")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(grad.len(), cfg.model.flat_size());
    // Frozen upstream tensor: embedding entry 0.
    assert_eq!(grad[0], 0.0, "a frozen parameter received a gradient");
    // A trainable head coordinate agrees with a central finite difference.
    let layout = cfg.model.layout();
    let slot = layout
        .iter()
        .find(|s| s.name == "policy_b")
        .expect("policy head exists");
    let idx = slot.start;
    let h = 1e-3f32;
    let values = theta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let perturbed = |delta: f32| {
        let mut v = values.clone();
        v[idx] += delta;
        Tensor::from_vec(v, vec![values.len()], dev()).unwrap()
    };
    let fd = (potential.mean_loss(&perturbed(h)) - potential.mean_loss(&perturbed(-h)))
        / (2.0 * h as f64);
    assert!(
        (fd - grad[idx] as f64).abs() < 2e-2 * fd.abs().max(1e-3),
        "head gradient {} disagrees with finite difference {fd}",
        grad[idx]
    );
}

#[test]
fn campaign_arms_execute_and_report_finite_endpoints() {
    let cfg = CampaignConfig {
        worlds: [6, 2, 2, 4],
        updates: 4,
        warm_start: 2,
        batch_worlds: 2,
        ..Default::default()
    };
    let report = run(&cfg).unwrap();
    assert_eq!(report.arms.len(), 6);
    assert!(report.arms.iter().any(|a| a.arm == Arm::Rlcd));
    assert!(report.arms.iter().any(|a| a.arm == Arm::Supervised));
    for arm in &report.arms {
        assert!(arm.brier_excess.is_finite());
        assert!(arm.nll.is_finite());
        assert!(arm.policy_entropy.is_finite());
        assert!(arm.mean_path_entropy.is_finite());
        assert!(arm.passes >= cfg.updates);
    }
    // The contrast is either absent (too few worlds) or a finite paired pair.
    if let Some((diff, se)) = report.paired_contrast {
        assert!(diff.is_finite() && se.is_finite() && se >= 0.0);
    }
    assert!(report.params_total < 1_000_000);
}

#[test]
fn rng_streams_are_independent_by_construction() {
    let s = Streams::derive(17);
    let all = [s.data, s.path, s.init, s.mcmc];
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(all[i], all[j]);
        }
    }
    assert_eq!(s, Streams::derive(17));
    assert_ne!(s, Streams::derive(18));
}
