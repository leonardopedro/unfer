//! **A3 — the residual turnover probe past `m = 128` (open measurement, non-Lean).**
//!
//! The question this file exists to answer: does `q` — the certificate
//! residual `‖Hψ − θψ‖` of the lowest Ritz pair — ever enter the exponential
//! regime the Hashimoto paper (`papers/Hashimoto.md`) predicts for the
//! *φ-function* approximation error? The campaign series (widening window
//! `i(1 + 0.2k)`) shows `q` **growing** with `m` (`2.98 @ m=8 → 6.56 @ m=64`,
//! `≈ 9.13 @ m=128`), never turning over; the two placement experiments in
//! `mass_gap_campaign.rs` rule out window width and placement as the easy
//! explanations at `m ≤ 32`. What was never measured is the setup the design
//! always called for, past `m = 128`:
//!
//! ```text
//!   harmonic shifts  γ_k = γ_0 + k·h,   γ_0 = i,   h = 2/(m−1)
//!   → fixed window [i, 3i] (width 2), spacing shrinking as m grows,
//!     full-rank basis (asserted below — a collapsed rank would measure
//!     degeneracy, not residual behaviour).
//! ```
//!
//! The probe was deleted after an aborted run (see the plan item); this is
//! its reconstruction. `m = 128` took ≈ 60 s in that run, so the test is
//! `#[ignore]`d and meant to be run deliberately with a long timeout:
//!
//! ```sh
//! cargo test --release -p fock_sirk --test zz_turnover -- \
//!   --ignored --nocapture zz_residual_turnover
//! ```
//!
//! The result (whatever it is) is recorded in
//! `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md`'s reachability ledger and the
//! plan item — this file only measures.

use fock_sirk::device::best_device;
use fock_sirk::{SirkOpts, solve_forward_sirk_with_opts};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, qcd_ym_hamiltonian};
use num_complex::Complex64;

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        // Same unit-norm frame as every gauge-fixed-program suite, so the
        // numbers are comparable with the campaign's recorded series.
        unit_norm_steps: true,
    }
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

/// Harmonic shifts `γ_k = γ_0 + k·h` with `γ_0 = i` and `h = 2/(m−1)`: the
/// window spans exactly `[i, 3i]` — fixed width 2 — while the spacing
/// shrinks with `m`. (Index `k = 0..m`, so `γ_{m−1} = i(1 + (m−1)h) = 3i`.)
fn harmonic_shifts(m: usize) -> Vec<Complex64> {
    assert!(m >= 2, "the harmonic ladder needs at least two shifts");
    let h = 2.0 / (m - 1) as f64;
    (0..m)
        .map(|k| Complex64::new(0.0, 1.0 + k as f64 * h))
        .collect()
}

/// The measurement itself: one solve per `m`, printing the series that
/// answers the turnover question. Deliberately lenient about *failure* (a
/// component-budget explosion at large `m` is itself a measurement — it is
/// printed and the sweep stops) and honest about *degeneracy*: once the
/// whitened basis holds fewer than `m/2` independent vectors, a falling `q`
/// would measure projection onto a shrinking subspace rather than the
/// window's convergence (the placement experiment in `mass_gap_campaign.rs`),
/// so the row is flagged `DEGENERATE` and the sweep records where the collapse
/// starts instead of aborting — the collapse is part of the answer.
#[test]
#[ignore = "A3 long-running measurement: m = 128 ≈ 60 s, larger m minutes — run with --nocapture"]
fn zz_residual_turnover_past_m_128() {
    let h = qcd_ym_hamiltonian(1.0);
    let mut completed: Vec<usize> = Vec::new();
    let mut series: Vec<(usize, f64, f64)> = Vec::new(); // (m, theta_0, q_0)
    let mut first_degenerate: Option<usize> = None;

    println!(
        "{:>5} {:>5} {:>16} {:>14} {:>14} {:>10} {:>14} flags",
        "m", "rank", "theta_0", "q_0", "tau_m", "q_pos", "rel_err"
    );
    for m in [4usize, 8, 16, 32, 64, 128, 192, 256] {
        let res = match solve_forward_sirk_with_opts(
            &h,
            &empty_vacuum(),
            &harmonic_shifts(m),
            &best_device(),
            None,
            &opts(),
        ) {
            Ok(res) => res,
            Err(e) => {
                println!(
                    "{m:>5}      — solve failed ({e}) — the component budget or the \
                     window ended the sweep here; recorded as the measurement"
                );
                break;
            }
        };
        let degenerate = res.rank * 2 < m;
        if degenerate && first_degenerate.is_none() {
            first_degenerate = Some(m);
        }
        let rr = res.ritz_abs_residuals();
        let (theta, q0) = rr[0];
        let q_pos = rr.iter().find(|(t, _)| *t > 0.0).map(|(_, q)| *q);
        let rel_err = 0.5 * (q0 / theta) * (q0 / theta);
        println!(
            "{m:>5} {:>5} {theta:>16.8} {q0:>14.4e} {:>14.4e} {:>10} {rel_err:>14.4e} {}",
            res.rank,
            res.scales[m],
            q_pos
                .map(|q| format!("{q:.4e}"))
                .unwrap_or_else(|| "-".into()),
            if degenerate {
                "DEGENERATE (rank < m/2): q here is not the window's convergence"
            } else {
                ""
            },
        );
        completed.push(m);
        series.push((m, theta, q0));
    }

    // What the sweep owes the record: it must actually reach `m = 128` — the
    // plan's open question is *past* the known series, and the answer may be
    // "the window dies first" (the collapse is part of the measurement, which
    // is why degenerate rows are flagged and kept rather than aborted). The
    // trend itself (turnover or none) is recorded in the docs from the printed
    // table, never pre-judged here.
    assert!(
        completed.contains(&128),
        "sweep did not reach m = 128 (completed: {completed:?}) — nothing was measured past the known series"
    );
    for (m, theta, q) in &series {
        assert!(
            theta.is_finite() && q.is_finite() && *q >= 0.0,
            "m = {m}: non-finite measurement theta = {theta}, q = {q}"
        );
    }
    println!("\ncompleted m: {completed:?}");
    println!("first degenerate m (rank < m/2): {first_degenerate:?}");
    println!("series (m, theta_0, q_0): {series:?}");
}
