//! N10 consistency checks for the physical-constraint gate.
//!
//! These are exact finite-matrix facts, not a joint BRST construction and not a
//! physical-space result. They record why the ghost grading `U` cannot simply be
//! used as the Krein metric `eta` for a nonzero charge, and why an even
//! contracting homotopy is incompatible with a nonzero odd charge.

type M2 = [[f64; 2]; 2];

fn mul(a: M2, b: M2) -> M2 {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..2).map(|k| a[r][k] * b[k][c]).sum()))
}

fn scale(a: M2, s: f64) -> M2 {
    a.map(|row| row.map(|x| s * x))
}

fn matvec(a: M2, x: [f64; 2]) -> [f64; 2] {
    std::array::from_fn(|r| (0..2).map(|k| a[r][k] * x[k]).sum())
}

fn dot(x: [f64; 2], y: [f64; 2]) -> f64 {
    x[0] * y[0] + x[1] * y[1]
}

fn norm(x: [f64; 2]) -> f64 {
    dot(x, x).sqrt()
}

fn identity() -> M2 {
    [[1.0, 0.0], [0.0, 1.0]]
}

fn zero() -> M2 {
    [[0.0; 2]; 2]
}

#[test]
fn ghost_parity_metric_forces_a_nonzero_odd_charge_to_vanish() {
    // U is an involution and Q is nilpotent and odd: UQ = -QU.
    let u = [[1.0, 0.0], [0.0, -1.0]];
    let q = [[0.0, 1.0], [0.0, 0.0]];
    let q2 = mul(q, q);
    assert_eq!(q2, zero());
    assert_eq!(mul(u, u), identity());
    assert_eq!(mul(u, q), scale(mul(q, u), -1.0));
    // Charge-form symmetry with eta = U would require <Qx,Uy>=<x,UQy>.
    // Substituting y=UQx gives ||Qx||^2 on the left and 0 on the right.
    let x = [0.0, 1.0];
    let y = matvec(u, matvec(q, x));
    let left = dot(matvec(q, x), matvec(u, y));
    let right = dot(x, matvec(u, matvec(q, y)));
    assert!((left - norm(matvec(q, x)).powi(2)).abs() < 1e-15);
    assert!(right.abs() < 1e-15);
    assert!(left > 0.5, "nonzero witness must have nonzero charge norm");
    // Hence the only charge satisfying the symmetry for all x,y is Q=0.
    // Verify the identity directly on the nonzero witness: it fails.
    assert!((left - right).abs() > 0.5);
}

#[test]
fn an_even_contracting_homotopy_cannot_contract_a_nonzero_odd_charge() {
    // QPQ=Q requires P[1][0]=1, but an even P (UP=PU) is diagonal, so
    // QPQ=0 for every admissible P while Q!=0.
    let u = [[1.0, 0.0], [0.0, -1.0]];
    let q = [[0.0, 1.0], [0.0, 0.0]];
    for p in [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.5, 0.0], [0.0, 0.5]],
        identity(),
    ] {
        assert_eq!(mul(u, p), mul(p, u), "P must be even");
        let contracted = mul(mul(q, p), q);
        assert_eq!(contracted, zero(), "even P annihilates the odd charge");
        assert_ne!(contracted, q);
    }
    // An odd P can contract it, but then UP=-PU, exactly the degree the plan
    // says must be proved rather than assumed.
    let p_odd = [[0.0, 0.0], [1.0, 0.0]];
    assert_eq!(mul(u, p_odd), scale(mul(p_odd, u), -1.0));
    assert_eq!(mul(mul(q, p_odd), q), q);
}
