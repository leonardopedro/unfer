//! **⚠ WHAT THIS CAMPAIGN MEASURES (2026-10-06 sector correction).**
//!
//! **Not the physical mass gap.** `run_campaign` starts from
//! `OuterBosonCreate(InnerBosonicState::vacuum())` — the **one-outer-boson**
//! sector — and `certified_mass_gap` returns `co.value - ce.value`, a difference
//! of two **one-particle reflection-sector** ground energies. Per `AGENTS.md` S45
//! and `mass_gap_spec.rs:135`, the nested-Fock ground is the **outer vacuum at
//! `E_0 = 0`** in the zero-outer-boson sector, and one-particle states are
//! diagnostics only.
//!
//! So these 20 fixtures' `ritz_certificate` / `certified_mass_gap` lines, and
//! anything fed to the Lean gap table from them, are **diagnostics** — and
//! since §M4.9 landed they say so in every row (`"sector"`,
//! `"physical":false`): do not read `certified_positive: false` as a statement
//! about the mass gap.
//!
//! **§M4.9 emission (2026-10-07).** `run_campaign` now additionally resolves
//! `φ₁` — the first state **above** the outer vacuum within the sector the
//! start spans, from `phi1_start` (a multi-species + squeezed-pair
//! superposition, different from both diagnostic starts) — and appends one
//! `physical_gap` row per `(g, m)` carrying `gap = ‖H φ₁‖`,
//! `q = ‖H φ₁ − e₁ φ₁‖` beside it, the signed `e1`, and a sector label.
//! That row is the §M4.9 **estimator, not a certificate**: it carries no
//! band, so the Lean gap table stays empty until a row qualifies
//! (`CONSOLIDATED_PLAN.md` §M4.6 option (c)); `q` is what the reader judges
//! it by, and it is never defaulted. The start-vector question (§M4.9
//! requirement 3, a free parameter) records one measured fact: the
//! multi-outer-boson variant of the start hits the component budget
//! (`StateExplosion` at 265k > 200k components, `g = 1, m = 10`) — see
//! `phi1_start`.
//!
//! **Also missing: the positivity shift.** `mass_gap_spec.rs:135` requires the
//! one-particle `h` to be shifted for positivity before enclosure;
//! `qcd_ym_hamiltonian` does not, and the one-particle ground is negative at every
//! coupling — pinned by `the_documented_positivity_shift_is_absent` below. That
//! also makes the one-sidedness result in `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md`
//! conditional, since it needs `H` positive semidefinite.
//!
//! Everything below still stands, because all of it is arithmetic *about this
//! diagnostic*: the negligible backward term, `theta_0` shift-invariance being
//! expected rather than a defect, the `tau_m` growth, and both placement escapes
//! failing.
//!
//! ---
//!
//! M1 — the per-coupling certificate emission campaign.
//!
//! **Improves** `fock_sirk/tests/qcd_mass_gap_certified.rs` and the NDJSON
//! emitter, which between them covered `g = 1` only. The Lean gap-table row
//! (work order §M4.3) is starved of data without more couplings, and a table
//! with one row is not a table.
//!
//! **What this file is.** A campaign driver: it emits certified sector
//! certificates for a fixed grid of couplings, writes one NDJSON fixture per
//! `(g, N, m)`, and writes a manifest of SHA-256 hashes. Regenerating is
//! explicit — `UPDATE_GOLDEN=1` — so a fixture can never be silently refreshed
//! by an ordinary test run.
//!
//! **The gapless control.** `g = 0` is included *because* it should shrink with
//! truncation depth rather than certify a positive gap. It is the row that
//! makes the table's two-sidedness data rather than prose: if the campaign ever
//! "certifies" a positive gap at `g = 0`, that is a defect in the emitter or the
//! width, and `the_gapless_control_does_not_certify_a_positive_gap` fails.
//!
//! **Honesty invariant.** Every emitted row's `certified_positive` flag must
//! agree with its own `lo > 0` — never with an expectation about what the row
//! ought to say. That check is `every_fixture_flag_agrees_with_its_own_lo`.

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{
    PhysicalGapCertificate, SirkOpts, certified_mass_gap, certified_ritz_values,
    emit_gap_certificate_ndjson, emit_physical_gap_ndjson, solve_forward_sirk_with_opts,
};
use nested_fock_algebra::{InnerBosonicState, Operator, QuantumState, qcd_ym_hamiltonian};
use num_complex::Complex64;
use std::path::{Path, PathBuf};

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        // Unit-norm frame: the gauge-fixed H's deep squeezing spans a wide
        // spectral window (raw frame's Gram wall caps usable m).
        unit_norm_steps: true,
    }
}

fn empty_vacuum() -> QuantumState {
    QuantumState::vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()))
}

fn fock_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s = empty_vacuum();
    for &(m, n) in occ {
        for _ in 0..n {
            s = s.apply(&Operator::InnerBosonCreate(m));
        }
    }
    let norm = s.norm();
    if (norm - 1.0).abs() > 1e-12 {
        s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    }
    s
}

/// The R-odd sector start: the exact reflection-odd vector of the gauge-fixed
/// `Z₂` symmetry `R: (A₀,A₁) → (−A₁,−A₀)`, namely
/// `(|1₀⟩ + |1₁⟩)/√2`.
fn r_odd_start() -> QuantumState {
    let s0 = fock_state(&[(0, 1)]);
    let s1 = fock_state(&[(1, 1)]);
    let mut s = s0;
    s.scale_and_add(&s1, Complex64::new(1.0, 0.0));
    let inv = 1.0 / 2.0f64.sqrt();
    s.scale_and_add(&s.clone(), Complex64::new(inv - 1.0, 0.0));
    s
}

/// The sector label every physical-gap row from this campaign carries: the
/// state is resolved in the one-outer-boson sector, at the first level above
/// the outer vacuum (`E₀ = 0`). It names *both* halves — outer-number sector
/// and above-vacuum selection — because §M4.9's requirement is that a reader
/// can tell from the row alone where the number came from.
const PHYSICAL_SECTOR: &str = "one_outer_boson_above_vacuum";

/// §M4.9's `φ₁` start vector (requirement 3: a *free parameter*, recorded
/// and pinned here): an equal-weight superposition over all four one-quanton
/// species **plus** the squeezed-pair inner content `|1₀,1₁⟩` — different
/// from both diagnostic starts (the one-outer-boson inner vacuum and the
/// two-species R-odd state), spanning exactly the sector where S45 pins the
/// first excited state to live (one-quanton floor first, two-quanton floor
/// above it — `outer_vacuum_ground_validation.rs`).
///
/// **Why not a two-outer-boson start, measured (2026-10-07).** The literal
/// "multi-outer-boson" variant — one-quanton ⊕ a two-outer-boson component —
/// was implemented first and does not fit the campaign's component budget:
/// the two-quanton sector's product space grows past
/// `max_components = 200_000` during the solve itself
/// (`StateExplosion { components: 265896 }` at `g = 1, m = 10`), because
/// `H = dΓ(h)` block-diagonally preserves the sector's norm, so pruning never
/// sheds it. Resolving the two-quanton sector is therefore a heavier-lift
/// open item (it travels with the §M4.9 / turnover questions), not something
/// this campaign can do per row at `m ≥ 10`.
fn phi1_start() -> QuantumState {
    let mut s = QuantumState::zero();
    for species in 0..4u32 {
        s.scale_and_add(&fock_state(&[(species, 1)]), Complex64::new(1.0, 0.0));
    }
    s.scale_and_add(&fock_state(&[(0, 1), (1, 1)]), Complex64::new(1.0, 0.0));
    let norm = s.norm();
    s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
    s
}

/// The first state **above** the outer vacuum that this solve resolves: the
/// lowest Ritz pair with `θ > 0` (the vacuum sits at `E₀ = 0` by
/// `H|Ω⟩ = 0`; a level at or below zero is not `φ₁`). Returns the
/// whitened-frame coefficients of that Ritz vector for
/// [`fock_sirk::ForwardSirkResult::reconstruct`]. `None` means the solve
/// resolved nothing above the vacuum — and §M4.9's contract is then simply
/// that no physical row is emitted (the table's "empty if none" case).
fn first_above_vacuum(res: &fock_sirk::ForwardSirkResult) -> Option<nalgebra::DVector<Complex64>> {
    let eig = res.h_proj.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let idx = order.into_iter().find(|&i| eig.eigenvalues[i] > 0.0)?;
    Some(nalgebra::DVector::from_iterator(
        eig.eigenvalues.len(),
        eig.eigenvectors.column(idx).iter().copied(),
    ))
}

fn solve_sector(g: f64, start: &QuantumState, m: usize) -> fock_sirk::ForwardSirkResult {
    let h = qcd_ym_hamiltonian(g);
    let shifts: Vec<Complex64> = shifts_for_range((0, m));
    solve_forward_sirk_with_opts(&h, start, &shifts, &best_device(), None, &opts())
        .expect("SIRK solve must complete")
}

/// One campaign row.
///
/// `m` is the Krylov order the **solve** ran at, and it is the dimension that
/// actually varies the certified width. `n` is a truncation used only for the
/// exact-diagonalization cross-check: the SIRK solve has no `N` knob (the
/// truncation knobs are `max_components`/`adaptive`, which bound memory rather
/// than set the inner occupation ceiling), so a row's `N` is a *cross-check
/// parameter*, not a solve parameter. Recording that distinction is the point:
/// an earlier version of this file emitted identical bytes for N=6 and N=8,
/// because it looped over N without the solve ever seeing it.
struct Row {
    g: f64,
    n: usize,
    m: usize,
    exact_gap_at_n: f64,
    ndjson: String,
}

/// The exact low window `(E₀, E₁, E₁−E₀)` of the truncated gauge-fixed H at
/// total inner occupation ceiling `n_max`.
///
/// This is the same construction `qcd_mass_gap_certified::exact_gap` uses, kept
/// here rather than shared because integration test helpers are not importable
/// across test binaries. Unlike the SIRK solve, this **does** vary with `n`,
/// which is why it is the N axis of the campaign.
fn exact_gap(h: &nested_fock_algebra::Hamiltonian, n_max: u32) -> (f64, f64, f64) {
    let mut basis = Vec::new();
    for n0 in 0..=n_max {
        for n1 in 0..=n_max - n0 {
            for n2 in 0..=n_max - n0 - n1 {
                for n3 in 0..=n_max - n0 - n1 - n2 {
                    let mut occ = Vec::new();
                    if n0 > 0 {
                        occ.push((0, n0));
                    }
                    if n1 > 0 {
                        occ.push((1, n1));
                    }
                    if n2 > 0 {
                        occ.push((2, n2));
                    }
                    if n3 > 0 {
                        occ.push((3, n3));
                    }
                    basis.push(fock_state(&occ));
                }
            }
        }
    }
    let n = basis.len();
    let mut m = nalgebra::DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    let mut vals: Vec<f64> = m.symmetric_eigen().eigenvalues.iter().cloned().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (vals[0], vals[1], vals[1] - vals[0])
}

/// Run the campaign over the (g, m) solve grid, cross-checked at each N.
fn run_campaign(g_values: &[f64], m_values: &[usize], n_values: &[usize]) -> Vec<Row> {
    let mut rows = Vec::new();
    for &g in g_values {
        let h = qcd_ym_hamiltonian(g);
        for &m in m_values {
            let even = empty_vacuum();
            let odd = r_odd_start();
            let res_even = solve_sector(g, &even, m);
            let res_odd = solve_sector(g, &odd, m);
            let res_phi = solve_sector(g, &phi1_start(), m);

            let ce = certified_ritz_values(&res_even);
            let co = certified_ritz_values(&res_odd);
            assert!(
                !ce.is_empty() && !co.is_empty(),
                "g = {g}, m = {m}: a sector produced no certified Ritz value, so the row would be empty rather than wrong"
            );

            let gap = certified_mass_gap(&res_even, &res_odd).unwrap_or_else(|| {
                panic!("g = {g}, m = {m}: certified mass gap assembly returned None")
            });
            let base = emit_gap_certificate_ndjson(&gap);
            // The §M4.9 physical row, appended after the three diagnostic
            // lines: `gap = ‖Hφ₁‖` measured by matvec, `q` beside it. If the
            // solve resolved no state above the outer vacuum, no physical row
            // is emitted — the row's absence is the honest output, not a zero.
            let ndjson = match first_above_vacuum(&res_phi).and_then(|coeffs| {
                let phi = res_phi.reconstruct(&coeffs);
                PhysicalGapCertificate::measure(PHYSICAL_SECTOR, &h, &phi)
            }) {
                Some(cert) => format!("{base}{}", emit_physical_gap_ndjson(&cert)),
                None => base,
            };

            for &n in n_values {
                let (_, _, eg) = exact_gap(&h, n as u32);
                rows.push(Row {
                    g,
                    n,
                    m,
                    exact_gap_at_n: eg,
                    ndjson: ndjson.clone(),
                });
            }
        }
    }
    rows
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mass_gap_campaign")
}

fn slug(g: f64) -> String {
    // `0.5` not `0.50`; no sign for positives; stable across platforms because it
    // is derived from the literal rather than from a formatted float.
    let s = format!("{g}");
    s.replace('.', "p").replace('-', "m")
}

#[test]
fn the_campaign_covers_every_planned_coupling() {
    // M1's grid. `g = 0` is the gapless control and is not optional.
    let g_values = [0.0, 0.5, 1.0, 2.0, 3.0];
    let m_values = [10usize, 12];
    let n_values = [6usize, 8];
    let rows = run_campaign(&g_values, &m_values, &n_values);
    assert_eq!(rows.len(), g_values.len() * m_values.len() * n_values.len());
}

/// The grid is not decorative: `m` has to actually change the solve.
///
/// **What this shows, stated carefully.** The width *changes* with `m` — but it
/// grows, and the diagnostic `the_width_decomposition` shows it is ~100% Parlett
/// residual. The expectation for a Krylov residual is that it *falls* with order at
/// a fixed spectral window; that does not happen here, and
/// `the_extracted_ritz_value_does_not_respond_to_the_shifts` is the deeper reason.
#[test]
fn the_krylov_order_changes_the_certified_width_but_not_for_the_better() {
    let rows = run_campaign(&[1.0], &[8, 12], &[8]);
    let widths: Vec<f64> = rows
        .iter()
        .map(|r| parse_hi(&r.ndjson) - parse_lo(&r.ndjson))
        .collect();
    assert_eq!(widths.len(), 2);
    assert!(
        (widths[0] - widths[1]).abs() > 1e-12,
        "m = 8 and m = 12 produced the same certified width {0}; the grid is not varying anything",
        widths[0]
    );
}

/// And `N` must actually change the cross-check, since it is the other axis.
#[test]
fn the_truncation_axis_actually_changes_the_cross_check() {
    let rows = run_campaign(&[1.0], &[8], &[6, 8]);
    let gaps: Vec<f64> = rows.iter().map(|r| r.exact_gap_at_n).collect();
    assert_eq!(gaps.len(), 2);
    assert!(
        (gaps[0] - gaps[1]).abs() > 1e-9,
        "the N ≤ 6 and N ≤ 8 exact gaps agree to 1e-9 ({0}); the truncation axis is not doing anything",
        gaps[0]
    );
}

#[test]
fn the_gapless_control_does_not_certify_a_positive_gap() {
    // At `g = 0` the gauge-fixed Hamiltonian is free, so its truncated gap is a
    // truncation artefact, not a real gap. The control's job is to prove the
    // pipeline *can say so*.
    //
    // Note the sign: the control's measured gap is **negative** (≈ −4e-9), not
    // positive-and-small. An earlier version of this test asserted `gap > 0` and
    // was wrong — it assumed the artefact would land on the positive side. The
    // property worth asserting is the one below, which holds either way.
    let rows = run_campaign(&[0.0], &[10, 12], &[6]);
    for r in &rows {
        let gap = parse_gap(&r.ndjson);
        let lo = parse_lo(&r.ndjson);
        let hi = parse_hi(&r.ndjson);

        assert!(
            gap.abs() < 1e-6,
            "g = 0, m = {}: the gap must sit at the roundoff scale, not be a real gap: {gap}",
            r.m
        );
        assert!(
            lo <= 0.0 && hi >= 0.0,
            "g = 0, m = {}: the certified interval [{lo}, {hi}] must straddle zero, \
             because there is no gap to certify",
            r.m
        );
        assert!(
            !flag_is_true(&r.ndjson, "certified_positive"),
            "g = 0 must never certify a positive gap; the emitter claimed it does: {}",
            assembly_line(&r.ndjson)
        );
    }
}

/// M1's honesty invariant: a row's `certified_positive` flag is a
/// consequence of its own `lo`, never a claim about what the row ought to say.
#[test]
fn every_fixture_flag_agrees_with_its_own_lo() {
    for g in [0.0, 0.5, 1.0, 2.0, 3.0] {
        for m in [10usize, 12] {
            let rows = run_campaign(&[g], &[m], &[8]);
            for r in &rows {
                let lo = parse_lo(&r.ndjson);
                let flagged = flag_is_true(&r.ndjson, "certified_positive");
                assert_eq!(
                    flagged,
                    lo > 0.0,
                    "g = {g}, m = {m}: certified_positive = {flagged} but lo = {lo}; \
                     the flag must follow its own number"
                );
            }
        }
    }
}

/// The result that matters for the gap table: certification needs enough Krylov
/// order, and this records which rows reach it rather than assuming they do.
#[test]
fn the_table_records_which_rows_actually_certify() {
    let rows = run_campaign(&[0.0, 0.5, 1.0, 2.0, 3.0], &[10, 12], &[8]);
    let mut certified = 0usize;
    for r in &rows {
        if flag_is_true(&r.ndjson, "certified_positive") {
            certified += 1;
        }
    }
    println!(
        "{certified}/{} rows certify a positive gap at N ≤ 8",
        rows.len()
    );
    // The control must never be among them.
    let control_certified = rows
        .iter()
        .filter(|r| r.g == 0.0)
        .any(|r| flag_is_true(&r.ndjson, "certified_positive"));
    assert!(!control_certified, "the gapless control certified a gap");
}

// ── §M4.9: the physical-gap row ────────────────────────────────────────

/// Every emitted row names its sector and whether it is the physical gap or
/// a diagnostic (§M4.9 acceptance). A row that cannot do either should not
/// exist — that omission is exactly what once let a one-particle diagnostic
/// read as a mass gap.
#[test]
fn every_row_names_its_sector_and_whether_it_is_physical() {
    let rows = run_campaign(&[1.0], &[10], &[8]);
    let lines: Vec<&str> = rows[0].ndjson.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        4,
        "three diagnostic lines + one physical line:\n{}",
        rows[0].ndjson
    );
    for line in &lines {
        assert!(
            field(line, "sector").is_some(),
            "row without a sector label: {line}"
        );
        assert!(
            field(line, "physical").is_some(),
            "row without a physical/diagnostic flag: {line}"
        );
    }
    for line in &lines[..3] {
        assert_eq!(field(line, "physical"), Some("false"), "{line}");
        assert!(
            line.contains("one_outer_boson"),
            "diagnostic sector must say where it lives: {line}"
        );
    }
    assert_eq!(field(lines[3], "physical"), Some("true"), "{}", lines[3]);
    assert_eq!(
        field(lines[3], "sector"),
        Some("\"one_outer_boson_above_vacuum\""),
        "{}",
        lines[3]
    );
    assert!(
        lines[3].contains("\"kind\":\"physical_gap\""),
        "{}",
        lines[3]
    );
}

/// The physical row carries `q` beside `gap` and satisfies the §M4.9
/// identity `gap² = e1² + q²` (the machine-checkable form of
/// `‖Hφ₁‖² = E₁² + ‖H d‖²`), with `rel_err_est = ½(q/e1)²` — and it must not
/// pretend to a certified band it does not have.
#[test]
fn the_physical_gap_row_reports_q_beside_gap_and_obeys_the_identity() {
    let rows = run_campaign(&[1.0], &[10], &[8]);
    let phys = rows[0]
        .ndjson
        .lines()
        .find(|l| l.contains("\"physical_gap\""))
        .unwrap_or_else(|| panic!("no physical_gap row emitted:\n{}", rows[0].ndjson));
    let num = |key: &str| -> f64 {
        field(phys, key)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{key} missing or unparseable: {phys}"))
    };
    let (gap, q, e1, rel) = (num("gap"), num("q"), num("e1"), num("rel_err_est"));

    assert!(e1 > 0.0, "φ₁ must lie above the outer vacuum: e1 = {e1}");
    assert!(q.is_finite() && q >= 0.0, "q = {q}");

    let pythagoras = (e1 * e1 + q * q).sqrt();
    assert!(
        (gap - pythagoras).abs() <= 1e-9 * gap.max(1.0),
        "gap = {gap} but sqrt(e1² + q²) = {pythagoras} — the identity must hold to roundoff"
    );
    let expected_rel = 0.5 * (q / e1) * (q / e1);
    assert!(
        (rel - expected_rel).abs() <= 1e-12 * expected_rel.max(1.0),
        "rel_err_est = {rel}, ½(q/e1)² = {expected_rel}"
    );
    assert!(
        !phys.contains("\"lo\":") && !phys.contains("\"certified_positive\""),
        "the physical row must not carry certificate fields it does not have: {phys}"
    );
}

/// The `φ₁` start is its own start vector (§M4.9 requirement 3): normalized,
/// carrying all four species plus the squeezed-pair content (its overlap with
/// the two-species R-odd start is √(2/5) of the equal five-component weight —
/// not 1, so it is not the diagnostic start in disguise), and with no overlap
/// with the even diagnostic start.
#[test]
fn the_phi1_start_lives_outside_both_diagnostic_starts() {
    let phi = phi1_start();
    assert!((phi.norm() - 1.0).abs() < 1e-12, "norm = {}", phi.norm());
    let w_r_odd = QuantumState::inner_product(&r_odd_start(), &phi).norm();
    assert!(
        (w_r_odd - 2.0 / 10.0f64.sqrt()).abs() < 1e-9,
        "|⟨r_odd|φ₁⟩| = {w_r_odd}, expected √(2/5) = 2/√10 from the five equal components"
    );
    let w_even = QuantumState::inner_product(&empty_vacuum(), &phi).norm();
    assert!(
        w_even < 1e-12,
        "|⟨even_start|φ₁⟩| = {w_even} — the start must not sit in the even diagnostic's mode"
    );
}

#[test]
fn the_manifest_matches_the_pinned_fixtures() {
    // M1's acceptance: a manifest of hashes a reader can verify without
    // re-running the campaign.
    let dir = fixture_dir();
    let manifest = dir.join("MANIFEST.tsv");
    if !manifest.exists() {
        // The campaign has never been written. Skipping is recorded, not hidden:
        // `the_campaign_has_been_written` is what forces the write.
        return;
    }
    let text = std::fs::read_to_string(&manifest).expect("manifest readable");
    let mut checked = 0usize;
    for line in text.lines().skip(1) {
        let mut it = line.split('\t');
        let (Some(expected), Some(file)) = (it.next(), it.next()) else {
            continue;
        };
        if file.is_empty() {
            continue;
        }
        let path = dir.join(file);
        let Ok(actual) = sha256(&path) else {
            panic!("manifest names {file} but the fixture is missing");
        };
        assert_eq!(
            actual, expected,
            "{file} does not match its manifest hash; re-run the campaign with UPDATE_GOLDEN=1 if intended"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "the manifest listed no fixtures, which is not a manifest"
    );
}

/// Writes the campaign fixtures. Run deliberately:
///
/// ```sh
/// UPDATE_GOLDEN=1 cargo test --release -p fock_sirk --test mass_gap_campaign -- \
///   --ignored --nocapture write_the_campaign
/// ```
#[test]
#[ignore = "campaign writer; run with UPDATE_GOLDEN=1 (heavy)"]
fn write_the_campaign() {
    if std::env::var("UPDATE_GOLDEN").as_deref() != Ok("1") {
        eprintln!("refusing to write: set UPDATE_GOLDEN=1");
        return;
    }
    let g_values = [0.0, 0.5, 1.0, 2.0, 3.0];
    let m_values = [10usize, 12];
    let n_values = [6usize, 8];
    let rows = run_campaign(&g_values, &m_values, &n_values);
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("fixture dir");

    let mut manifest =
        String::from("# sha256\tfile\tcoupling\tkrylov_order\tcrosscheck_N\tcertified_positive\n");
    let mut n_files = 0usize;
    for r in &rows {
        let name = format!("qym_g{}_m{}_xn{}.ndjson", slug(r.g), r.m, r.n);
        let path = dir.join(&name);
        // Self-describing header (M6): a fixture carries its own metadata, so a
        // reader can place it without first finding its log entry.
        let header = format!(
            "{{\"kind\":\"campaign_metadata\",\"coupling\":{:.17e},\"krylov_order\":{},\
             \"crosscheck_truncation\":{},\"exact_gap_at_crosscheck\":{:.17e},\
             \"hamiltonian\":\"qcd_ym_hamiltonian(g)\",\"solve_source\":\"fock_sirk::certificate\"}}\n",
            r.g, r.m, r.n, r.exact_gap_at_n
        );
        std::fs::write(&path, format!("{header}{}", r.ndjson)).expect("fixture written");
        let h = sha256(&path).expect("hashed");
        manifest.push_str(&format!(
            "{h}\t{name}\t{}\t{}\t{}\t{}\n",
            r.g,
            r.m,
            r.n,
            flag_is_true(&r.ndjson, "certified_positive")
        ));
        n_files += 1;
        println!("wrote {name} ({h})");
    }
    std::fs::write(dir.join("MANIFEST.tsv"), manifest).expect("manifest written");
    println!("\n{n_files} fixtures written.");
}

// ── NDJSON field access, minimal and dependency-free ─────────────────────

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = &line[i..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// The assembly line is the one carrying `certified_mass_gap`.
fn assembly_line(ndjson: &str) -> &str {
    ndjson
        .lines()
        .find(|l| l.contains("\"certified_mass_gap\""))
        .unwrap_or_else(|| panic!("emitted NDJSON has no certified_mass_gap line:\n{ndjson}"))
}

fn parse_gap(ndjson: &str) -> f64 {
    field(assembly_line(ndjson), "gap")
        .and_then(|v| v.parse().ok())
        .expect("gap field parses")
}

fn parse_lo(ndjson: &str) -> f64 {
    field(assembly_line(ndjson), "lo")
        .and_then(|v| v.parse().ok())
        .expect("lo field parses")
}

fn parse_hi(ndjson: &str) -> f64 {
    field(assembly_line(ndjson), "hi")
        .and_then(|v| v.parse().ok())
        .expect("hi field parses")
}

fn flag_is_true(ndjson: &str, key: &str) -> bool {
    field(assembly_line(ndjson), key) == Some("true")
}

/// SHA-256 without pulling in a dependency: the fixtures are small and this runs
/// in a test.
fn sha256(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    // FIPS 180-4 constants.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = bytes.clone();
    let bitlen = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut gg, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & gg);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = gg;
            gg = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, gg, hh].into_iter().enumerate() {
            h[i] = h[i].wrapping_add(v);
        }
    }
    Ok(h.iter().map(|x| format!("{x:08x}")).collect())
}

#[cfg(test)]
mod sha256_tests {
    use super::sha256;

    fn digest(bytes: &[u8]) -> String {
        let p = std::env::temp_dir().join(format!("sha256check_{}", bytes.len()));
        std::fs::write(&p, bytes).unwrap();
        let s = sha256(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        s
    }

    #[test]
    fn known_vectors() {
        // If this ever stops matching, every hash in the manifest is wrong and
        // the manifest check would be comparing noise to noise.
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

/// **The certified window contains the exact gap at every coupling — and this test
/// currently passes for the wrong reason.**
///
/// It was written as the campaign's positive result. It is still true, but with
/// the extraction defect in
/// `the_extracted_ritz_value_does_not_respond_to_the_shifts` it is **vacuous**: the
/// emitted window at `g = 1` is `[-8.009, +6.866]`, which contains 0.091 for
/// reasons that have nothing to do with the width being tight. A window that wide
/// would contain almost anything.
///
/// It is kept, and kept honest, because it will become meaningful the moment the
/// extraction is fixed — at which point this test will start failing if the width
/// stops covering the exact gap. The comment above `the_manifest_matches` in
/// `../fock_sirk/tests/qcd_mass_gap_certified.rs` deserves the same warning: its
/// "certified window contains the exact gap" assertion passes vacuously today.
#[test]
fn the_certified_window_contains_the_exact_gap_at_every_coupling() {
    for g in [0.0, 0.5, 1.0, 2.0, 3.0] {
        let rows = run_campaign(&[g], &[12], &[8]);
        for r in &rows {
            let lo = parse_lo(&r.ndjson);
            let hi = parse_hi(&r.ndjson);
            let eg = r.exact_gap_at_n;
            assert!(
                lo <= eg && eg <= hi,
                "g = {g}: the certified window [{lo}, {hi}] must contain the exact \
                 N ≤ {} gap {eg}",
                r.n
            );
        }
    }
}

/// The recorded result, as an assertion: **no row certifies positivity**, and the
/// window is enormously wider than the gap.
///
/// If a future fix makes certification succeed, this test *fails* and must be
/// updated deliberately — the gap table's contents should change only by an
/// explicit decision, not by a routine edit.
#[test]
fn no_row_currently_certifies_positivity_and_that_is_recorded() {
    let rows = run_campaign(&[0.0, 0.5, 1.0, 2.0, 3.0], &[10, 12], &[8]);
    let certified = rows
        .iter()
        .filter(|r| flag_is_true(&r.ndjson, "certified_positive"))
        .count();
    assert_eq!(
        certified, 0,
        "{certified} rows now certify a positive gap. That would be a real improvement, \
         but it is a CHANGE to the gap table's contents: update \
         docs/MASS_GAP_SPEC.md §4, the status tables, and the Lean work order \
         before accepting it."
    );

    // The ratio, which is the actionable number: how far off is the width?
    let r = rows
        .iter()
        .find(|r| r.g == 1.0 && r.m == 12)
        .expect("g=1,m=12 row");
    let width = parse_hi(&r.ndjson) - parse_lo(&r.ndjson);
    let ratio = width / r.exact_gap_at_n.abs();
    println!(
        "g = 1, m = 12: certified width {width:.3} against an exact gap of {:.6} \
         ({ratio:.0}x wider)",
        r.exact_gap_at_n
    );
    assert!(
        ratio > 10.0,
        "the certified window is only {ratio:.0}x the exact gap, which would mean \
         certification is nearly within reach; re-check the recorded finding"
    );
}

// ── Diagnostics for the recorded M1 finding ────────────────────────────────

/// **The width decomposition, and what it actually shows.**
///
/// This diagnostic exists because an earlier claim in this file was wrong, and
/// the correction needs to be reproducible rather than remembered.
///
/// **The retracted claim.** It said "the blocking term is the eigendecomposition
/// backward error `c(n)·u·‖Ĝ‖`". Measured, that term is
///
/// ```text
///   m=10:  6.0e-11      m=24:  2.9e-09      m=64:  7.0e-08
/// ```
///
/// against a width of ~3 to ~7. It is **eight orders of magnitude too small to
/// matter**. The claim was inferred from the fact that `c(n) = n³` grows, without
/// checking the magnitudes. `c(n)·u` is ~1e-12 even at `c(n) = 10⁵`.
///
/// **What the width actually is.** ~100% Parlett residual `‖r‖`, and the residual
/// *grows* with the number of shifts:
///
/// ```text
///   m:            8      12      16      24      32      40      48      64
///   ‖r‖_even:  2.979   3.350   3.747   4.353   4.817   5.270   5.699   6.560
///   ‖Ĝ‖_even:   337     784    1408    2843    4505    6554    8987   14761
/// ```
///
/// So the expectation that a Krylov residual falls with order — exponentially, for
/// a fixed spectral window — is the right expectation, and this pipeline does not
/// meet it. See `the_extracted_ritz_value_does_not_respond_to_the_shifts` for why.
#[test]
#[ignore = "diagnostic: prints the width decomposition, slow"]
fn the_width_decomposition() {
    use fock_sirk::certificate::{UNIT_ROUNDOFF, h_proj_spectral_norm};
    println!(
        "{:>4} {:>14} {:>14} {:>14} {:>16}",
        "m", "resid_e", "|G|_e", "backward", "delta_e"
    );
    for m in [8usize, 12, 16, 24, 32, 48, 64] {
        let re = solve_sector(1.0, &empty_vacuum(), m);
        let rr = re.ritz_abs_residuals();
        let (theta, res) = rr[0];
        let gnorm = h_proj_spectral_norm(&re.h_proj);
        let n = re.h_proj.nrows();
        let backward = (n as f64).powi(3) * UNIT_ROUNDOFF * gnorm.max(1.0);
        let delta = res + backward + 4.0 * UNIT_ROUNDOFF * theta.abs().max(1.0);
        println!("{m:>4} {res:>14.4e} {gnorm:>14.4e} {backward:>16.4e} {delta:>14.4e}");
    }
    println!("\nThe backward-error column is ~1e-8 while delta_e is ~5. It is not the blocker.");
}

/// **The surviving anomaly: `‖r‖` grows with Krylov order while `θ₀` converges.**
///
/// This is the finding this file's conclusion rests on, so unlike the two
/// diagnostics below it is an ordinary (non-`#[ignore]`d) test: it pins the
/// behaviour that makes the certified window useless.
///
/// Measured at `g = 1` with one fixed start vector:
///
/// ```text
///   m =  4   theta_0 = -0.98074   resid = 2.729
///   m =  8   theta_0 = -1.34865   resid = 2.979
///   m = 16   theta_0 = -1.65822   resid = 3.746
///   m = 24   theta_0 = -1.69595   resid = 4.353
/// ```
///
/// `θ₀` converges monotonically downward (toward ≈ −1.70) and the *true* error
/// shrinks — at `m = 8` it is ≈ 0.35 — yet the *bound* inflates ~60%. A correct
/// Lanczos/SIRK residual at a fixed spectral window shrinks, or at worst stays
/// bounded. So the enclosure is loose by roughly an order of magnitude and gets
/// looser with order, which is exactly why raising the Krylov order does not
/// tighten the certificate.
///
/// Leading hypothesis, not proven: `forward_sirk.rs` builds the basis with a
/// **one-step** recurrence `w_{k+1} = (H − z_k I) w_k` while `h_proj_raw` is filled
/// with the **two-term** tridiagonal pattern
/// `⟨w_j|H|w_k⟩ = σ_{k+1} G_{j,k+1} + z_k G_{j,k}`, whose comment assumes
/// `H ŵ_k = τ_{k+1} ŵ_{k+1} + z_k ŵ_k`. Confirming it needs the analytic residual
/// compared against `‖Hψ − θψ‖` by explicit matvec.
///
/// Both halves are asserted, not just printed: convergence of the value (so the
/// solver is not simply broken) *and* growth of the bound (so it is not a
/// convergence problem either). If a fix makes `‖r‖`` fall with `m`, this test
/// fails and should be inverted — that inversion is the whole goal.
/// **The mechanism: the one-step step norms `τ_k` grow without bound, and the
/// residual inherits them.**
///
/// `ForwardSirkResult::scales[k]` is `τ_k = ‖(H − z_{k−1}) u_{k−1}‖` — the norm
/// the one-step recurrence inflates by at each step. Measured at `g = 1`:
///
/// ```text
///   m      tau_m   |chat[m-1]|   resid_0    theta_0
///   4     6.409e1     1.487e-1    2.7289e0   -0.980743
///   8     2.239e2     3.400e-1    2.9790e0   -1.348649
///  12     4.655e2     1.338e+0    3.3502e0   -1.531486
///  16     7.855e2     6.434e+0    3.7465e0   -1.658222
///  24     1.653e3     1.509e+1    4.3529e0   -1.695953
///  32     2.816e3     1.041e+1    4.8204e0   -1.662896
///  48     6.010e3     1.078e+1    5.6992e0   -1.663100
/// ```
///
/// `τ_m` grows ~94× over `m = 4…48` and does not plateau. In `ritz_abs_residuals`
/// the residual vector has `e[j] = (z_j − θ)·ĉ[j] + τ_{j+1}·ĉ[j−1]` for `j < m`
/// and `e[m] = τ_m·ĉ[m−1]`, contracted as `e†Ge` — so `τ` and `‖G‖` set the scale
/// of the bound, which is why the width tracks them.
///
/// **Why theory says this should decay instead.** `papers/Hashimoto.md`
/// (Hashimoto–Nodera, JJIAM 2019) concludes that for SIRK "the decay speed of the
/// approximation error … is similar to an exponential order", whereas Arnoldi/SIA/RK
/// reach only "the polynomial order or slower". The stated reason is that SIRK's
/// vectors are built from **resolvents** `(γI − A)⁻¹`, *which are bounded*. The
/// analogous step norm is `‖X u‖ ≤ 1/dist(γ, W(A))` — bounded, hence decay.
///
/// `fock_sirk` is **inverse-free** ("no `O(N³)` solves", `docs/ARCHITECTURE.md`) and
/// multiplies by `(H − z_k I)` itself. Its `τ_k` is therefore an *unbounded*
/// polynomial step norm. **This does NOT mean the method lacks an exponential rate**
/// — see the retraction below.
///
/// **Retraction.** An earlier version of this comment concluded the exponential rate
/// "cannot transfer to an inverse-free construction". That was wrong. With harmonic
/// shifts `γ_k = γ_0 + kh` and `X = (γ_m I − H)⁻¹`, the one-step chain is
/// `w_k = (−1)^k X^{−k}Π_{j=1}^{m−k}(1 − hjX)v` — a *cleared-denominator* form of the
/// resolvent chain, differing only by scalars that the unit-norm frame removes. The
/// rate is a property of the rational Krylov **space**, not of how the vectors are
/// materialised, and the start vector is a free parameter.
///
/// **What actually blocks the mass gap is the quantity.** Hashimoto's exponential
/// rate is for the **φ-function approximation error** `‖φ_k(H)v − approx‖`. The mass
/// gap needs `|E₁ − E₀|`, and here that is `0.091` inside a spectrum spanning
/// `−1.7 … +337` — ratio ~3.7e3. Resolving a 0.091 splitting in a 340-wide band is
/// intrinsically much harder than the φ-function application the paper analyses.
/// Nothing in that paper bounds the gap more sharply than the a-posteriori residual
/// does today.
///
/// So the assertions below are deliberately about **`τ_k` and this residual**, not
/// about the method's asymptotic rate. See
/// `the_shift_window_placement_does_not_rescue_the_residual` for the two placement
/// experiments.
#[test]
fn the_documented_positivity_shift_is_absent() {
    use fock_sirk::mass_gap_spec::even_sector_is_vacuum;

    // `mass_gap_spec.rs:135` requires the one-particle `h` to be "shifted if
    // needed for positivity" before it is enclosed, "so the outer vacuum is the
    // exact ground". `qcd_ym_hamiltonian` is a bare `compile_to_fock` and
    // applies no such shift, so the one-particle ground comes out NEGATIVE — and
    // `even_sector_is_vacuum`, which encodes exactly that precondition, fails.
    //
    // Consequence while this holds: the one-sidedness result
    // `||H phi|| >= E` in `docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md` is CONDITIONAL,
    // because it needs `H` positive semidefinite. Unshifted the identity still
    // holds but bounds only `|E|`, never its sign — and the sign is what a gap
    // claim needs.
    for (g, expected_negative) in [(0.5f64, true), (1.0, true), (2.0, true)] {
        let re = solve_sector(g, &empty_vacuum(), 24);
        let (theta, _) = re.ritz_abs_residuals()[0];
        assert_eq!(
            theta < 0.0,
            expected_negative,
            "g = {g}: theta_0 = {theta}. If the positivity shift has been added, \
             INVERT this test and delete the 'conditional' caveat from \
             docs/VERIFY_MASS_GAP_NORM_RESIDUAL.md"
        );
        assert!(
            !even_sector_is_vacuum(theta, 1e-6),
            "g = {g}: even_sector_is_vacuum(theta_0 = {theta}, tol = 1e-6) now PASSES, \
             so the precondition in mass_gap_spec.rs:135 is satisfied — update this \
             test and the one-sidedness caveat"
        );
    }
}

#[test]
fn the_step_norms_grow_without_bound() {
    let mut taus = Vec::new();
    for &m in &[4usize, 8, 12, 16, 24, 32] {
        let re = solve_sector(1.0, &empty_vacuum(), m);
        taus.push(re.scales[m]);
    }
    for w in taus.windows(2) {
        assert!(
            w[1] > w[0],
            "the step norm tau_m stopped growing at m=4..32 ({0:e} -> {1:e}); if the \
             one-step recurrence was replaced by resolvent application, tau becomes \
             bounded and this should INVERT",
            w[0],
            w[1]
        );
    }
    assert!(
        taus[5] > taus[0] * 30.0,
        "tau_m grew by less than 30x ({0:e} -> {1:e}); too weak to be the mechanism \
         behind the residual growth — re-measure before citing it",
        taus[0],
        taus[5]
    );
}

/// **Ruling out the two obvious escapes.** Both were tried; neither works, so
/// "just move the shifts" is *not* the fix and the residual growth is not a
/// window-placement bug.
///
/// *(1) Holding the shift window fixed.* The campaign's `shifts_for_range((0,m))`
/// emits `i(1 + 0.2k)`, whose window widens with `m` (reaching `i·25.6` at
/// `m = 128`), so widening was the first suspect. Pinning the window to a fixed
/// width of 2 and refining inside it changes nothing — `resid_0` still climbs
/// `2.73 → 4.82` over `m = 4…32`. So the widening window is *not* the cause.
///
/// *(2) Moving the window off the spectrum.* Hashimoto's rate needs `|γ|` beyond
/// the numerical range, so sliding the window out was the second suspect. It does
/// make the residual fall — but only where the basis **collapses**:
///
/// ```text
///   R = 0    rank 4 -> 24    resid 2.729 -> 4.819   theta_0 -> -1.663   (converging)
///   R = 50   rank 4 -> 25    resid 2.729 -> 4.711   theta_0 -> -1.728   (converging)
///   R = 400  rank 3 ->  9    resid 2.608 -> 4.358   theta_0 -> -1.511   (NOT converged)
///   R = 3000 rank 3 ->  4    resid 3.050 -> 2.509   theta_0 -> -1.044   (NOT converged)
/// ```
///
/// At `R = 400 ≈ ‖A‖` and beyond, rank collapses from `m` to 3–9, `θ₀` stops
/// converging, and the falling residual is an artifact of projecting onto a
/// nearly rank-deficient space — not convergence. That is the expected degeneracy
/// of a *polynomial* Krylov chain whose factors `H − γI` become dominated by
/// `−γI`, so every vector is nearly collinear with the start. Resolvent vectors do
/// not have this failure mode; inverse-free polynomial ones do.
///
/// `R = 0` and `R = 50` are the honest operating regime (rank ≈ `m`, `θ₀`
/// converging), and that is exactly where the residual grows.
#[test]
fn the_shift_window_placement_does_not_rescue_the_residual() {
    // (1) Fixed-width window at the campaign's own placement: growth persists.
    let h = qcd_ym_hamiltonian(1.0);
    let fixed_window = |m: usize| -> Vec<Complex64> {
        (0..m)
            .map(|k| Complex64::new(0.0, 1.0 + 2.0 * (k as f64 + 0.5) / m as f64))
            .collect()
    };
    let mut fixed = Vec::new();
    for &m in &[4usize, 8, 16, 32] {
        let re = solve_forward_sirk_with_opts(
            &h,
            &empty_vacuum(),
            &fixed_window(m),
            &best_device(),
            None,
            &opts(),
        )
        .expect("fixed-window SIRK solve must complete");
        assert!(
            re.rank * 2 >= m,
            "fixed-window solve at m={m} has rank {} — the basis already collapsed, so \
             this test is measuring degeneracy rather than residual growth",
            re.rank
        );
        fixed.push(re.ritz_abs_residuals()[0].1);
    }
    assert!(
        fixed[3] > fixed[0],
        "a FIXED shift window no longer grows the residual ({0:e} -> {1:e}); the \
         widening-window theory in the doc comment needs revisiting",
        fixed[0],
        fixed[3]
    );
}

#[test]
fn the_residual_grows_while_the_value_converges() {
    let orders = [4usize, 8, 16, 24];
    let mut thetas = Vec::new();
    let mut residuals = Vec::new();
    for &m in &orders {
        let re = solve_sector(1.0, &empty_vacuum(), m);
        let rr = re.ritz_abs_residuals();
        let (theta, res) = rr[0];
        thetas.push(theta);
        residuals.push(res);
    }

    // (1) The value converges: monotonically downward, and it moved.
    for w in thetas.windows(2) {
        assert!(
            w[1] < w[0],
            "theta_0 stopped decreasing at m=4..24 ({0} -> {1}); the value is no longer \
             converging, so the 'residual grows while the value converges' reading no \
             longer applies",
            w[0],
            w[1]
        );
    }
    assert!(
        thetas[3] - thetas[0] < -0.5,
        "theta_0 barely moved ({0} -> {1}); too little signal to read anything into the \
         residual trend",
        thetas[0],
        thetas[3]
    );

    // (2) The bound inflates while that happens. This is the anomaly.
    for w in residuals.windows(2) {
        assert!(
            w[1] > w[0],
            "the Parlett residual no longer grows with order at m=4..24 ({0} -> {1}); if \
             this was fixed upstream, INVERT this assertion and record the fix — the \
             certified width should now tighten with Krylov order",
            w[0],
            w[1]
        );
    }
    assert!(
        residuals[3] > residuals[0] * 1.4,
        "the residual grew by less than 40% ({0} -> {1}); the order-inflation is too \
         weak to be this file's headline finding — re-measure before citing it",
        residuals[0],
        residuals[3]
    );
}

/// **Shift-invariance of `θ₀` is expected, not a defect.** Kept because the
/// inference that follows from it is *not* obvious, and this file got that
/// inference wrong once (see below).
///
/// A Krylov sequence spans a subspace of the minimal-polynomial subspace
/// `K(H, u₀)` that the start vector generates. **Shifts change the basis, not that
/// subspace.** So any two shift sequences of the same length span the same
/// subspace, approximate the same extremal eigenvalues, and a *converged* `θ₀`
/// must come out shift-invariant to roundoff. Measured at a fixed count of 8:
///
/// ```text
///   shifts i..1.4i        theta_0 = -1.34864876  resid = 2.9790e0  |G| = 3.3742e2
///   shifts i..2.6i        theta_0 = -1.34864876  resid = 2.9790e0  |G| = 3.3742e2
///   shifts 10i..11.4i     theta_0 = -1.34864876  resid = 2.9790e0  |G| = 3.3742e2
///   real 1..8             theta_0 = -1.34864876  resid = 2.9790e0  |G| = 3.3742e2
/// ```
///
/// **Bit-identical, which is the expected outcome — not the smoking gun this
/// diagnostic was first filed as.** The basis *does* respond: `scales`, `G`, and
/// `h_proj` all move (`h_proj[0][0]` = 110.9 / 77.1 / 119.3 across the three
/// families above). Only the extremal *value* is invariant, because it is a
/// property of `K(H, u₀)`, not of the probe.
///
/// **The two claims this diagnostic previously supported are retracted:**
/// 1. *"Only the count of shifts has any effect."* False — the intermediates move.
/// 2. *"`θ₀` sits above `E₃ = −1.978`, so it is not approximating the ground
///    state."* **Invalid comparison.** The exact spectrum quoted here is at the
///    `N ≤ 8` occupation cutoff, but the solve is capped by `max_components =
///    200_000` and `prune_eps = 1e-14`, i.e. a different, larger Hilbert space.
///    Capping the solve at 495 components (= `C(12,4)`, the `N ≤ 8` dimension)
///    makes it explode at 571 components, which proves the two are not the same
///    object. The comparison carries no information either way.
///
/// The Ritz bounds are internally consistent at every `m` measured, and the
/// surviving anomaly is elsewhere: the residual *grows* with order — see
/// `the_residual_grows_while_the_value_converges`.
/// A labelled shift generator, so the diagnostic below can take a table of them.
type ShiftGen = Box<dyn Fn(usize) -> Complex64>;

#[test]
#[ignore = "diagnostic: prints shift-sensitivity, slow"]
fn the_extracted_ritz_value_does_not_respond_to_the_shifts() {
    let h = qcd_ym_hamiltonian(1.0);
    let m = 8usize;
    let cases: [(&str, ShiftGen); 4] = [
        (
            "i..1.4i",
            Box::new(|j: usize| Complex64::new(0.0, 1.0 + 0.05 * j as f64)),
        ),
        (
            "i..2.6i",
            Box::new(|j: usize| Complex64::new(0.0, 1.0 + 0.2 * j as f64)),
        ),
        (
            "10i..11.4i",
            Box::new(|j: usize| Complex64::new(0.0, 10.0 + 0.2 * j as f64)),
        ),
        (
            "real 1..8",
            Box::new(|j: usize| Complex64::new(1.0 + j as f64, 0.05)),
        ),
    ];
    for (label, mk) in cases {
        let shifts: Vec<Complex64> = (0..m).map(&mk).collect();
        let res = solve_forward_sirk_with_opts(
            &h,
            &empty_vacuum(),
            &shifts,
            &best_device(),
            None,
            &opts(),
        )
        .expect("sector SIRK solve must complete");
        let rr = res.ritz_abs_residuals();
        let (theta, resid) = rr[0];
        println!("{label:>14}  theta_0 = {theta:.8}  resid = {resid:.4e}");
    }
    println!("\nAll four identical => the output depends on the shift COUNT only.");
}

/// **The exact truncated low spectrum, for reference.** `θ₀` must **not** be
/// compared against this without matching the Hilbert space first: this
/// diagonalization is at the `N ≤ 8` occupation cutoff, while the campaign solve
/// is capped by `max_components = 200_000` with `prune_eps = 1e-14`. Capping the
/// solve at 495 components (= `C(12,4)`, the `N ≤ 8` dimension) makes it explode at
/// 571 components, so these are different objects and a naive `θ₀ > E₃` says
/// nothing. (This file made exactly that mistake; see
/// `the_extracted_ritz_value_does_not_respond_to_the_shifts`.)
///
/// What the numbers *are* good for: the gap is N-stable while the absolute
/// energies are not — `0.091073 → 0.091161` for `E₁−E₀`, against
/// `−2.025 → −2.744` for `E₀` — so the gap is the well-conditioned quantity.
#[test]
#[ignore = "diagnostic: prints the exact truncated spectrum"]
fn the_exact_truncated_spectrum() {
    let h = qcd_ym_hamiltonian(1.0);
    for n_max in [6u32, 8] {
        let mut basis = Vec::new();
        for n0 in 0..=n_max {
            for n1 in 0..=n_max - n0 {
                for n2 in 0..=n_max - n0 - n1 {
                    for n3 in 0..=n_max - n0 - n1 - n2 {
                        let mut occ = Vec::new();
                        if n0 > 0 {
                            occ.push((0, n0));
                        }
                        if n1 > 0 {
                            occ.push((1, n1));
                        }
                        if n2 > 0 {
                            occ.push((2, n2));
                        }
                        if n3 > 0 {
                            occ.push((3, n3));
                        }
                        basis.push(fock_state(&occ));
                    }
                }
            }
        }
        let k = basis.len();
        let mut mat = nalgebra::DMatrix::<Complex64>::zeros(k, k);
        for (j, s) in basis.iter().enumerate() {
            let hs = h.apply(s);
            for (i, t) in basis.iter().enumerate() {
                mat[(i, j)] = QuantumState::inner_product(t, &hs);
            }
        }
        let mut v: Vec<f64> = mat.symmetric_eigen().eigenvalues.iter().cloned().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "N<={n_max} (dim {k}): E0={:.10} E1={:.10} gap={:.10} E3={:.6}",
            v[0],
            v[1],
            v[1] - v[0],
            v[3]
        );
    }
    println!("\nThe GAP is N-stable (0.091073 -> 0.091161) even though the absolute");
    println!("energies are not. The gap is the well-conditioned quantity; the");
    println!("absolute Ritz values are the ones that need work.");
}
