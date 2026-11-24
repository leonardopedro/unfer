//! Bit-identity pins for the two nested-vector map operations that dominate the
//! rational-Krylov GMRES loop (`docs/VERIFY_PHI4_MOMENTUM.md` §5.6).
//!
//! `joint_inner_product` used to be a `b.get(label)` binary search for every
//! entry of `a`; it is now a two-pointer merge. `joint_axpy` used to re-insert
//! every touched label (`insert(basis.clone(), …)`); it now updates through
//! `get_mut` and inserts only genuinely new labels. Both rewrites are meant to
//! be *bit*-identical, not merely accurate, because the cutoff sweep pins
//! return probabilities to 17 digits: a last-bit change is a failed record
//! cross-check, and a 2× speedup bought by changing the arithmetic is not a
//! result.
//!
//! The oracles below are the previous implementations verbatim (as of commit
//! `54173aa`), including the accumulation order and the zero-deletion rule. Each
//! comparison is on `to_bits()`, so `0.0` and `-0.0` count as different.

use fock_sirk::joint_sirk::{joint_axpy, joint_inner_product};
use nested_fock_algebra::exact_inner::{ActionLimits, JointOuterBasis, NestedVector};
use nested_fock_algebra::phi4_momentum::ScalarMomCont;
use num_complex::Complex64 as C;

fn limits() -> ActionLimits {
    ActionLimits {
        max_components: 1_000_000,
        max_work: usize::MAX / 4,
    }
}

/// The previous `joint_inner_product`: search `b` once per entry of `a`, in
/// `a`'s order.
fn inner_product_previous(a: &NestedVector, b: &NestedVector) -> C {
    a.iter()
        .filter_map(|(s, x)| b.get(s).map(|y| x.conj() * y))
        .sum()
}

/// The previous `joint_axpy`: read, add, then `insert`/`remove` the label.
fn axpy_previous(a: &mut NestedVector, b: &NestedVector, scale: C, limits: ActionLimits) {
    for (basis, amplitude) in b {
        let value = a.get(basis).copied().unwrap_or_default() + scale * amplitude;
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "probe amplitudes stay finite"
        );
        if value == C::new(0.0, 0.0) {
            a.remove(basis);
        } else {
            assert!(
                a.contains_key(basis) || a.len() < limits.max_components,
                "probe stays inside the component budget"
            );
            a.insert(basis.clone(), value);
        }
    }
}

/// Deterministic amplitudes: no `rand` dependency, and the same sequence on
/// every platform.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 11) as f64 / (1_u64 << 53) as f64;
        2.0 * unit - 1.0
    }
}

fn vector(labels: &[JointOuterBasis], seed: u64, stride: usize) -> NestedVector {
    let mut rng = Lcg(seed);
    let mut out = NestedVector::new();
    for (i, label) in labels.iter().enumerate() {
        if stride > 1 && i % stride != 0 {
            continue;
        }
        let amplitude = C::new(rng.next_unit(), rng.next_unit());
        if amplitude.norm() > 0.05 {
            out.insert(label.clone(), amplitude);
        }
    }
    out
}

fn assert_bit_identical(got: &NestedVector, want: &NestedVector, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: label count");
    for ((got_label, got_value), (want_label, want_value)) in got.iter().zip(want.iter()) {
        assert_eq!(got_label, want_label, "{what}: label order");
        assert_eq!(
            got_value.re.to_bits(),
            want_value.re.to_bits(),
            "{what}: real part at {got_label:?}"
        );
        assert_eq!(
            got_value.im.to_bits(),
            want_value.im.to_bits(),
            "{what}: imaginary part at {got_label:?}"
        );
    }
}

/// Labels of two nested Mehler windows: the small window is contained in the
/// large one, and the difference supplies labels that exist on one side only.
fn window_pairs() -> (Vec<JointOuterBasis>, Vec<JointOuterBasis>) {
    let small = ScalarMomCont::window_labels(8);
    let large = ScalarMomCont::window_labels(12);
    let mut small: Vec<JointOuterBasis> = small.into_iter().collect();
    let mut large: Vec<JointOuterBasis> = large.into_iter().collect();
    small.sort();
    large.sort();
    (small, large)
}

#[test]
fn inner_product_is_bit_identical_to_the_search_version() {
    let (small, large) = window_pairs();
    assert!(small.len() >= 15, "window 8 must be a usable dictionary");
    assert!(large.len() > small.len(), "window 12 must extend window 8");

    // Dense/dense, dense/sparse, sparse/dense, and a pair with labels on one
    // side only (exercises both `Less` and `Greater` merge branches).
    let dense = vector(&large, 1, 1);
    let sparse = vector(&large, 2, 3);
    let nested = vector(&small, 3, 1);
    let outside: NestedVector = large
        .iter()
        .filter(|label| !small.contains(label))
        .enumerate()
        .map(|(i, label)| (label.clone(), C::new(0.5 + i as f64, -0.25 * i as f64)))
        .collect();

    let cases: [(&str, &NestedVector, &NestedVector); 6] = [
        ("dense × dense", &dense, &dense),
        ("dense × sparse", &dense, &sparse),
        ("sparse × dense", &sparse, &dense),
        ("nested × dense", &nested, &dense),
        ("outside × inside", &outside, &nested),
        ("inside × outside", &nested, &outside),
    ];

    let mut nonzero = 0;
    for (what, a, b) in cases {
        let got = joint_inner_product(a, b);
        let want = inner_product_previous(a, b);
        assert_eq!(
            got.re.to_bits(),
            want.re.to_bits(),
            "{what}: real part {got} vs {want}"
        );
        assert_eq!(
            got.im.to_bits(),
            want.im.to_bits(),
            "{what}: imaginary part {got} vs {want}"
        );
        if want.norm() > 0.0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero >= 4,
        "the pin must compare non-degenerate products, got {nonzero} of 6"
    );

    // Disjoint label sets: the merge has to walk one map to its end without a
    // match, and must return the same zero bits as the search form.
    let (even_labels, odd_labels): (Vec<_>, Vec<_>) = large
        .iter()
        .cloned()
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let even: NestedVector = even_labels
        .into_iter()
        .map(|(i, label)| (label, C::new(0.25 * i as f64 + 1.0, 0.0)))
        .collect();
    let odd: NestedVector = odd_labels
        .into_iter()
        .map(|(i, label)| (label, C::new(0.0, i as f64 - 0.5)))
        .collect();
    let disjoint = joint_inner_product(&even, &odd);
    let disjoint_previous = inner_product_previous(&even, &odd);
    assert_eq!(
        disjoint.re.to_bits(),
        disjoint_previous.re.to_bits(),
        "disjoint dot real part"
    );
    assert_eq!(
        disjoint.im.to_bits(),
        disjoint_previous.im.to_bits(),
        "disjoint dot imaginary part"
    );
    assert_eq!(
        disjoint_previous.re.to_bits(),
        0.0_f64.to_bits(),
        "disjoint dot must be zero"
    );
}

#[test]
fn axpy_is_bit_identical_to_the_insert_version() {
    let (small, large) = window_pairs();
    let dense = vector(&large, 11, 1);
    let sparse = vector(&large, 12, 2);
    let nested = vector(&small, 13, 1);

    let scales = [
        C::new(0.75, 0.0),
        C::new(-0.75, 0.125),
        C::new(0.0, 0.5),
        C::new(1e-300, -1e-300),
        C::new(0.0, 0.0),
    ];

    for (case, b) in [
        ("dense source", &dense),
        ("sparse source", &sparse),
        ("narrow source", &nested),
    ] {
        for scale in scales {
            let mut got = dense.clone();
            let mut want = dense.clone();
            joint_axpy(&mut got, b, scale, limits()).expect("finite axpy");
            axpy_previous(&mut want, b, scale, limits());
            assert_bit_identical(&got, &want, &format!("{case} at scale {scale}"));
        }
    }

    // Repeated strokes: the in-place path must not drift over a Krylov-like
    // sequence of alternating additions, which is where a stale or copied slot
    // would show up.
    let mut got = vector(&large, 21, 2);
    let mut want = got.clone();
    for stroke in 0..8 {
        let scale = if stroke % 2 == 0 {
            C::new(0.5, -0.25)
        } else {
            C::new(-0.5, 0.25)
        };
        joint_axpy(&mut got, &sparse, scale, limits()).expect("finite axpy");
        axpy_previous(&mut want, &sparse, scale, limits());
    }
    assert_bit_identical(&got, &want, "eight alternating strokes");

    // Exact cancellation must delete the entry in both forms — otherwise the
    // comparison above could pass with a lingering exact zero.
    let mut cancelled = dense.clone();
    let before = cancelled.len();
    let negated: NestedVector = dense
        .iter()
        .map(|(label, value)| (label.clone(), -*value))
        .collect();
    joint_axpy(&mut cancelled, &negated, C::new(1.0, 0.0), limits()).expect("finite axpy");
    assert!(
        cancelled.is_empty(),
        "a fully cancelling axpy must leave no entries, got {}",
        cancelled.len()
    );
    assert!(before > 0, "cancellation case must start non-empty");

    // And a partial cancellation removes only the cancelled labels.
    let mut partial = dense.clone();
    let victim = dense.keys().next().cloned().expect("nonempty");
    let unit: NestedVector = [(victim.clone(), C::new(1.0, 0.0))].into_iter().collect();
    let existing = dense[&victim];
    joint_axpy(&mut partial, &unit, -existing, limits()).expect("finite axpy");
    assert!(
        !partial.contains_key(&victim),
        "the cancelled label must be removed"
    );
    assert_eq!(
        partial.len(),
        dense.len() - 1,
        "only the cancelled label may disappear"
    );
}
