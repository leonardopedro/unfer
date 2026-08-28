//! Outer-vacuum ground-state doctrine — the nested-Fock ground state.
//!
//! **The doctrine** (project directive): the ground state of every program
//! theory (QYM, QED, QG, NS) on the nested Fock space is ALWAYS the
//! **outer-Fock vacuum**. The reason is structural, and the tests below pin
//! each clause of it:
//!
//! 1. *The full Hamiltonian is the one-particle Hamiltonian **enclosed** in
//!    creation (on the left) and annihilation (on the right) operators.* On
//!    the nested Fock space the one-particle Hamiltonian `h` — the 3D
//!    gauge-fixed Hamiltonian of the sector, acting on the inner Fock space —
//!    is second-quantized at the OUTER level,
//!
//!    $$H \;=\; \sum_{i,j} h_{ij}\; C^\dagger(e_i)\, A(e_j),$$
//!
//!    where `e_i` run over the inner one-particle basis and `C†(e_i)` /
//!    `A(e_j)` are the outer creation/annihilation operators labeled by the
//!    inner state `e_i`. Every term carries an outer annihilation operator
//!    rightmost, so the FULL Hamiltonian (not a normal-ordered part of it)
//!    annihilates the outer vacuum `|Ω⟩` **identically**: `H|Ω⟩ = 0`.
//!
//! 2. *The one-particle Hamiltonian needs no normal-ordering modification —
//!    only adding a constant is fine.* `h` enters the enclosure VERBATIM (its
//!    matrix elements are used exactly as the inner operator defines them).
//!    For QYM the truncated one-particle spectrum reaches below zero (the
//!    deep pair-squeezing measured by `qym_mass_gap.rs` at the inner level),
//!    and adding a single constant `c` to `h` makes it positive — the only
//!    modification the doctrine allows (an energy-zero reparametrization).
//!
//! 3. *Consequently the ground state is the outer vacuum.* With
//!    `h⁺ = h + c·1 ⪰ 0`, the second quantization `dΓ(h⁺)` is positive on
//!    every outer particle-number sector (the n-quanton sector carries the
//!    symmetrized n-fold sum, with spectrum the n-fold sums of the
//!    one-particle eigenvalues), and the outer vacuum is an exact eigenvector
//!    of energy 0 — the ground, with the mass gap equal to the smallest
//!    one-particle energy `λ_min(h⁺)`.
//!
//! The inner-level "squeezed grounds" reported by the earlier suites are
//! one-particle-level statements: they are the truncated spectrum of `h`
//! BEFORE the constant is added, not the ground state of the nested theory.
//! `qym_mass_gap.rs` documents this reframing in its header.
//!
//! All numerics are exact matrix-element assemblies on small truncated
//! bases plus SIRK–Hashimoto solves (`solve_forward_sirk_with_opts`) for the
//! solver-level signatures. Every sector test asserts one-particle
//! Hermiticity as a correctness gate (the enclosure is Hermitian iff `h` is).

use fock_sirk::auto::shifts_for_range;
use fock_sirk::device::best_device;
use fock_sirk::{solve_forward_sirk_with_opts, SirkOpts};
use nalgebra::DMatrix;
use nested_fock_algebra::{
    ns_eulerian_fiber, qcd_ym_hamiltonian, qed_free_photon,
    qg_starobinsky_gauge_fixed_scalaron, qg_starobinsky_scalaron_field, Hamiltonian,
    InnerBosonicState, Operator, QuantumState,
};
use num_complex::Complex64;

fn shifts(m: usize) -> Vec<Complex64> {
    shifts_for_range((0, m))
}

fn opts() -> SirkOpts {
    SirkOpts {
        prune_eps: 1e-14,
        max_components: Some(200_000),
        brst_tol: 1e-10,
        adaptive: false,
        // Unit-norm frame: exact reparametrization (ritz_edge_study p2), used
        // by every gauge-fixed-program suite.
        unit_norm_steps: true,
    }
}

/// The bare outer vacuum `|Ω⟩` — no universes at all. This is THE ground
/// state of the doctrine.
fn outer_vacuum() -> QuantumState {
    QuantumState::vacuum()
}

/// The inner species carrying occupations `occ` — the label of an outer
/// quanton whose inner content is the one-particle basis state `e`.
fn inner_species(occ: &[(u32, u32)]) -> InnerBosonicState {
    let mut s = InnerBosonicState::vacuum();
    for &(m, n) in occ {
        s.modes.insert(m, n);
    }
    s
}

/// One outer quanton of species `occ`: a single universe with that inner content.
fn outer_one(occ: &[(u32, u32)]) -> QuantumState {
    outer_vacuum().apply(&Operator::OuterBosonCreate(inner_species(occ)))
}

/// Normalized inner Fock state (one universe) with the given inner occupations.
/// Built by INNER ladder creations on an empty universe — matching the species
/// labels used by the enclosure (species `[(m,n)]` ↔ state with occupation n
/// in mode m).
fn inner_state(occ: &[(u32, u32)]) -> QuantumState {
    let mut s =
        outer_vacuum().apply(&Operator::OuterBosonCreate(InnerBosonicState::vacuum()));
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

/// All one-universe inner states with total occupation ≤ `n_max` over
/// `n_modes` modes (the inner one-particle basis `e_i`).
fn inner_basis(n_modes: u32, n_max: u32) -> Vec<Vec<(u32, u32)>> {
    fn rec(mode: u32, left: u32, n_modes: u32, acc: &mut Vec<(u32, u32)>, out: &mut Vec<Vec<(u32, u32)>>) {
        if mode == n_modes {
            out.push(acc.clone());
            return;
        }
        for n in 0..=left {
            if n > 0 {
                acc.push((mode, n));
            }
            rec(mode + 1, left - n, n_modes, acc, out);
            if n > 0 {
                acc.pop();
            }
        }
    }
    let mut out = Vec::new();
    rec(0, n_max, n_modes, &mut Vec::new(), &mut out);
    out
}

/// The exact matrix `[⟨e_i|h|e_j⟩]` of the inner one-particle Hamiltonian.
fn inner_matrix(h: &Hamiltonian, basis: &[Vec<(u32, u32)>]) -> DMatrix<Complex64> {
    let states: Vec<QuantumState> = basis.iter().map(|o| inner_state(o)).collect();
    let n = states.len();
    let mut m = DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in states.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in states.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    m
}

/// **The outer enclosure** (doctrine clause 1): `H = Σ_ij h_ij C†(e_i) A(e_j)`
/// — creation operators on the LEFT, annihilation operators on the RIGHT.
/// The rightmost (annihilation) operator acts on the state first, so `H|Ω⟩ = 0`
/// identically, and on the one-quanton sector `H` acts exactly as `h`.
fn outer_enclose(h_mat: &DMatrix<Complex64>, basis: &[Vec<(u32, u32)>]) -> Hamiltonian {
    let mut terms = Vec::new();
    for (i, ei) in basis.iter().enumerate() {
        for (j, ej) in basis.iter().enumerate() {
            let h_ij = h_mat[(i, j)];
            if h_ij.norm_sqr() > 1e-30 {
                terms.push((
                    h_ij,
                    vec![
                        Operator::OuterBosonCreate(inner_species(ei)),
                        Operator::OuterBosonAnnihilate(inner_species(ej)),
                    ],
                ));
            }
        }
    }
    Hamiltonian { terms }
}

/// All outer states with 0, 1 and 2 quanta over the species list (the
/// truncated nested basis: the vacuum, the one-quanton sector, the
/// two-quanton sector).
fn outer_basis(species: &[Vec<(u32, u32)>]) -> Vec<QuantumState> {
    let mut basis = vec![outer_vacuum()];
    for s in species {
        basis.push(outer_one(s));
    }
    for a in 0..species.len() {
        for b in a..species.len() {
            let mut s = outer_vacuum()
                .apply(&Operator::OuterBosonCreate(inner_species(&species[a])))
                .apply(&Operator::OuterBosonCreate(inner_species(&species[b])));
            let norm = s.norm();
            if (norm - 1.0).abs() > 1e-12 {
                s.scale_and_add(&s.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));
            }
            basis.push(s);
        }
    }
    basis
}

fn matrix_of(h: &Hamiltonian, basis: &[QuantumState]) -> DMatrix<Complex64> {
    let n = basis.len();
    let mut m = DMatrix::<Complex64>::zeros(n, n);
    for (j, s) in basis.iter().enumerate() {
        let hs = h.apply(s);
        for (i, t) in basis.iter().enumerate() {
            m[(i, j)] = QuantumState::inner_product(t, &hs);
        }
    }
    m
}

fn min_eig(m: &DMatrix<Complex64>) -> f64 {
    m.clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
}

fn hermiticity(m: &DMatrix<Complex64>) -> f64 {
    (m.clone() - m.adjoint()).norm()
}

/// Build the doctrine object for a sector: the verbatim one-particle matrix,
/// the constant `c` that makes it strictly positive (the measured floor plus
/// the fixed margin `GAP_MARGIN`), and the outer-enclosed Hamiltonian.
/// Returns `(h_plus, c, H_full)`.
const GAP_MARGIN: f64 = 0.1;

fn enclose_with_constant(h: &Hamiltonian, basis: &[Vec<(u32, u32)>]) -> (DMatrix<Complex64>, f64, Hamiltonian) {
    let h_mat = inner_matrix(h, basis);
    let floor = min_eig(&h_mat);
    // The ONLY allowed modification: add a constant to the one-particle
    // Hamiltonian. `c` lifts the measured spectral floor to +GAP_MARGIN.
    let c = GAP_MARGIN - floor;
    let mut h_plus = h_mat.clone();
    for i in 0..h_plus.nrows() {
        h_plus[(i, i)] += Complex64::new(c, 0.0);
    }
    let full = outer_enclose(&h_plus, basis);
    (h_plus, c, full)
}

fn sector_slice(m: &DMatrix<Complex64>, idx: &[usize]) -> DMatrix<Complex64> {
    let k = idx.len();
    let mut sub = DMatrix::<Complex64>::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            sub[(a, b)] = m[(i, j)];
        }
    }
    sub
}

// ───────────────────── doctrine clause 1: vacuum annihilation ─────────────────────

/// The FULL outer-enclosed Hamiltonian annihilates the outer vacuum exactly,
/// in every program sector — QYM (all couplings), QED, QG (scalaron field,
/// gauge-fixed scalaron, densitized kinetic) and NS. Nothing is normal-order
/// modified away: the annihilation is structural (every term carries an outer
/// annihilation operator rightmost, and `A|Ω⟩ = 0`).
#[test]
fn outer_vacuum_annihilated_by_full_hamiltonian_all_sectors() {
    // Inner one-particle bases per sector.
    let basis4 = inner_basis(4, 2); // QYM / gauge-fixed scalaron: modes 0..3
    let basis3 = inner_basis(3, 2); // NS fiber / scalaron field / densitized

    // QYM at three couplings — the doctrine holds at ALL of them (the ground
    // is the outer vacuum whether or not the one-particle constant is needed).
    for g in [0.0_f64, 1.0, 2.0] {
        let (_, _, full) = enclose_with_constant(&qcd_ym_hamiltonian(g), &basis4);
        let hv = full.apply(&outer_vacuum());
        assert!(
            hv.norm() < 1e-12,
            "QYM(g={g}): the FULL Hamiltonian must annihilate the outer vacuum, got norm {}",
            hv.norm()
        );
    }

    // QED control (free photon, one-particle spectrum already positive —
    // c is then just the fixed margin).
    let qed = qed_free_photon(&[1.0, 2.0, 3.0, 5.0]);
    let (_, _, full) = enclose_with_constant(&qed, &basis4);
    assert!(full.apply(&outer_vacuum()).norm() < 1e-12, "QED vacuum annihilation");

    // QG: the massive scalaron field ω(k)=√(k²+m²), the gauge-fixed scalaron
    // (derivative variables), and the densitized kinetic (𝒮/𝒫 modes,
    // rebuilt with inner ops so it can serve as the one-particle Hamiltonian).
    let scalaron = qg_starobinsky_scalaron_field(&[0.0, 1.0, 2.0], 1.0);
    let (_, _, full) = enclose_with_constant(&scalaron, &basis3);
    assert!(full.apply(&outer_vacuum()).norm() < 1e-12, "scalaron field vacuum annihilation");

    let gf_scalaron = qg_starobinsky_gauge_fixed_scalaron(1.0);
    let (_, _, full) = enclose_with_constant(&gf_scalaron, &basis4);
    assert!(full.apply(&outer_vacuum()).norm() < 1e-12, "gauge-fixed scalaron vacuum annihilation");

    let densitized = densitized_inner(2);
    let (_, _, full) = enclose_with_constant(&densitized, &basis3);
    assert!(full.apply(&outer_vacuum()).norm() < 1e-12, "densitized vacuum annihilation");

    // NS: the Eulerian fiber (kinetic + advection + viscous offset). The FINAL
    // test Hamiltonian is its outer enclosure — clause 1 for NS.
    let a = [[0.0_f64, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let c = [0.15_f64, 0.0, -0.1];
    let ns = ns_eulerian_fiber(&a, &c);
    let (_, _, full) = enclose_with_constant(&ns, &basis3);
    assert!(full.apply(&outer_vacuum()).norm() < 1e-12, "NS vacuum annihilation");
}

/// The densitized kinetic as an INNER one-particle Hamiltonian (the qg_
/// densitized_kinetic structure written with inner ladder operators so the
/// outer enclosure can wrap it): n_s 𝒮-modes at +1/16 and one conformal 𝒫
/// mode at −1/24, per mode `c·(B†B − ½(B†²+B²))` — verbatim, un-normal-order
/// modified beyond the builder's own normal ordering.
fn densitized_inner(n_s_modes: u32) -> Hamiltonian {
    let n_modes = n_s_modes + 1;
    let mut terms = Vec::new();
    for i in 0..n_modes {
        let c = if i == n_s_modes { -1.0 / 24.0 } else { 1.0 / 16.0 };
        terms.push((
            Complex64::new(c, 0.0),
            vec![
                Operator::InnerBosonCreate(i),
                Operator::InnerBosonAnnihilate(i),
            ],
        ));
        terms.push((
            Complex64::new(-c * 0.5, 0.0),
            vec![
                Operator::InnerBosonCreate(i),
                Operator::InnerBosonCreate(i),
            ],
        ));
        terms.push((
            Complex64::new(-c * 0.5, 0.0),
            vec![
                Operator::InnerBosonAnnihilate(i),
                Operator::InnerBosonAnnihilate(i),
            ],
        ));
    }
    Hamiltonian { terms }
}

// ─────────────────── doctrine clause 3: the outer vacuum is the ground ───────────────────

/// QYM: with the one-particle constant added, the outer-enclosed Hamiltonian
/// has the outer vacuum as its EXACT ground (energy 0) with a positive mass
/// gap, at every coupling — and its one-quanton sector IS the shifted
/// one-particle Hamiltonian, verbatim.
#[test]
fn qym_outer_vacuum_ground_and_gap_at_all_couplings() {
    let basis = inner_basis(4, 2);
    let species = basis.clone();
    let n1: Vec<usize> = (1..=species.len()).collect(); // one-quanton block

    for g in [0.0_f64, 1.0, 2.0] {
        let (h_plus, c, full) = enclose_with_constant(&qcd_ym_hamiltonian(g), &basis);

        // Correctness gate: the one-particle Hamiltonian is self-adjoint.
        let h_mat = inner_matrix(&qcd_ym_hamiltonian(g), &basis);
        assert!(
            hermiticity(&h_mat) < 1e-9,
            "QYM(g={g}): one-particle matrix must be Hermitian"
        );
        // The constant did its one job: h⁺ ⪰ GAP_MARGIN.
        assert!(
            (min_eig(&h_plus) - GAP_MARGIN).abs() < 1e-8,
            "QYM(g={g}): shifted one-particle floor must sit at the margin, got {}",
            min_eig(&h_plus)
        );

        // The truncated nested matrix: vacuum ⊕ one-quanton ⊕ two-quanton.
        let ob = outer_basis(&species);
        let m = matrix_of(&full, &ob);
        assert!(
            hermiticity(&m) < 1e-9,
            "QYM(g={g}): outer-enclosed H must be Hermitian on the nested basis"
        );

        // Ground: the outer vacuum, at energy 0 exactly, below everything.
        assert!(
            m.column(0).iter().skip(1).all(|x| x.norm() < 1e-10),
            "QYM(g={g}): the vacuum must decouple (H|Ω⟩ = 0)"
        );
        let e_min = min_eig(&m);
        assert!(
            e_min.abs() < 1e-9,
            "QYM(g={g}): ground must be the outer vacuum at E=0, got {e_min}"
        );

        // Mass gap: the smallest excitation is the one-particle floor
        // λ_min(h⁺) = GAP_MARGIN (the two-quanton sector starts at 2·margin).
        let excited: Vec<usize> = (1..m.nrows()).collect();
        let gap = min_eig(&sector_slice(&m, &excited));
        assert!(
            (gap - GAP_MARGIN).abs() < 1e-6,
            "QYM(g={g}): mass gap above the outer vacuum must be the one-particle floor, got {gap}"
        );

        // One-quanton sector = h⁺ verbatim (clause 2: no other modification).
        let m1 = sector_slice(&m, &n1);
        for i in 0..h_plus.nrows() {
            for j in 0..h_plus.nrows() {
                assert!(
                    (m1[(i, j)] - h_plus[(i, j)]).norm() < 1e-9,
                    "QYM(g={g}): one-quanton sector must equal the shifted one-particle matrix"
                );
            }
        }
        let _ = c; // reported via the gap above
    }
}

/// QG and NS: the FINAL test Hamiltonian is the one-particle Hamiltonian
/// enclosed in creation (left) / annihilation (right) operators, and the
/// same ground-state structure follows — vacuum ground at 0, positive gap,
/// two-quanton sector starting at twice the gap. The scalaron field needs no
/// constant at all (its one-particle spectrum is already positive); the
/// gauge-fixed scalaron, the densitized kinetic (hyperbolic −1/24 conformal
/// direction included) and the NS Eulerian fiber get the single constant —
/// the doctrine tames all of them uniformly.
#[test]
fn qg_ns_outer_enclosure_vacuum_ground_structure() {
    let basis3 = inner_basis(3, 2);
    let basis4 = inner_basis(4, 2);

    let cases: Vec<(&str, Hamiltonian, u32)> = vec![
        ("scalaron_field", qg_starobinsky_scalaron_field(&[0.0, 1.0, 2.0], 1.0), 3),
        ("gauge_fixed_scalaron", qg_starobinsky_gauge_fixed_scalaron(1.0), 4),
        ("densitized_kinetic", densitized_inner(2), 3),
        ("ns_eulerian_fiber", {
            let a = [[0.0_f64, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
            let c = [0.15_f64, 0.0, -0.1];
            ns_eulerian_fiber(&a, &c)
        }, 3),
    ];

    for (name, h, n_modes) in cases {
        let basis = if n_modes == 4 { basis4.clone() } else { basis3.clone() };
        let (h_plus, _, full) = enclose_with_constant(&h, &basis);

        let h_mat = inner_matrix(&h, &basis);
        assert!(hermiticity(&h_mat) < 1e-9, "{name}: one-particle matrix Hermitian");
        assert!(
            min_eig(&h_plus) > GAP_MARGIN - 1e-8,
            "{name}: shifted one-particle spectrum strictly positive"
        );

        let ob = outer_basis(&basis);
        let m = matrix_of(&full, &ob);
        assert!(hermiticity(&m) < 1e-9, "{name}: outer-enclosed H Hermitian");
        assert!(
            full.apply(&outer_vacuum()).norm() < 1e-12,
            "{name}: H|Ω⟩ = 0 exactly"
        );
        let e_min = min_eig(&m);
        assert!(e_min.abs() < 1e-9, "{name}: ground = outer vacuum at E=0, got {e_min}");

        let excited: Vec<usize> = (1..m.nrows()).collect();
        let gap = min_eig(&sector_slice(&m, &excited));
        assert!(
            (gap - GAP_MARGIN).abs() < 1e-6,
            "{name}: positive gap above the vacuum, got {gap}"
        );

        // Two-quanton sector: the symmetrized sum of h⁺ with itself — its
        // floor sits at exactly twice the one-particle floor.
        let n2: Vec<usize> = (1 + basis.len()..m.nrows()).collect();
        let gap2 = min_eig(&sector_slice(&m, &n2));
        assert!(
            (gap2 - 2.0 * GAP_MARGIN).abs() < 1e-6,
            "{name}: two-quanton floor = 2×λ_min(h⁺), got {gap2}"
        );
    }
}

// ───────────────────── solver-level signatures (SIRK–Hashimoto) ─────────────────────

/// The SIRK signature of the doctrine: started FROM the outer vacuum, the
/// forward Krylov sequence collapses to rank 1 (every forward product is
/// proportional to the vacuum, because H|Ω⟩ = 0) and the only Ritz value is
/// exactly 0 — the solver SEES the exact eigenstate. Started from a
/// one-quanton superposition (a superposition over species of one universe —
/// distinct sectors, never merged), the Krylov space is the one-quanton
/// sector, invariant under dΓ(h⁺), and the Ritz values reproduce the
/// shifted one-particle spectrum: floor at the gap, nothing negative.
#[test]
fn sirk_vacuum_start_rank_collapse_and_gapped_one_quanton_ritz() {
    let basis4 = inner_basis(4, 2);

    for (name, h, g) in [
        ("qym_g1", qcd_ym_hamiltonian(1.0), 1.0_f64),
        ("qed_free_photon", qed_free_photon(&[1.0, 2.0, 3.0, 5.0]), 0.0),
    ] {
        let _ = g;
        let (_h_plus, _, full) = enclose_with_constant(&h, &basis4);

        // Vacuum start: rank 1, Ritz = [0].
        let res = solve_forward_sirk_with_opts(
            &full,
            &outer_vacuum(),
            &shifts(8),
            &best_device(),
            None,
            &opts(),
        )
        .unwrap();
        assert_eq!(res.rank, 1, "{name}: vacuum-start Krylov must collapse to rank 1");
        let ritz = res.ritz_values();
        assert!(
            ritz.iter().all(|r| r.abs() < 1e-8),
            "{name}: vacuum-start Ritz values must all be exactly 0, got {ritz:?}"
        );

        // One-quanton superposition start: the sector is dΓ-invariant, so
        // the window probes the shifted one-particle spectrum. The window
        // must stay BELOW the sector dimension (15 species ⇒ m+1 ≤ 15
        // forward products) so the Gram is full rank and the whitened
        // Galerkin projection is the honest Rayleigh–Ritz one — every Ritz
        // value is then an upper bound on the true levels, in particular on
        // the mass gap λ_min(h⁺) = GAP_MARGIN.
        let mut start = outer_vacuum();
        for (k, occ) in basis4.iter().enumerate() {
            let amp = 1.0 + k as f64 * 0.37;
            start.scale_and_add(&outer_one(occ), Complex64::new(amp, 0.0));
        }
        let norm = start.norm();
        start.scale_and_add(&start.clone(), Complex64::new(1.0 / norm - 1.0, 0.0));

        // Window length: keep the forward products STRICTLY inside the
        // cyclic span's dimension so the Gram stays full rank — a dependent
        // sequence's whitening produces spurious below-floor rungs (measured:
        // at m=8 the rank drops to 8 and a ghost rung at 0.058 appears below
        // the exact 0.1 floor; at m=5 Rayleigh–Ritz is exact). m=5 is the
        // honest dependency-free window here.
        let res1 = solve_forward_sirk_with_opts(
            &full,
            &start,
            &shifts(5),
            &best_device(),
            None,
            &opts(),
        )
        .unwrap();
        let ritz1 = res1.ritz_values();
        // The forward span is the CYCLIC span of the start vector — its rank
        // is the number of distinct eigenvalues carrying nonzero start
        // amplitude, not the full 15-dim sector. Rank ≥ 2 is the honest
        // solver-level statement that the window is a genuine spectral probe
        // (contrast the rank-1 vacuum collapse above).
        assert!(
            res1.rank >= 2,
            "{name}: the one-quanton window must be a genuine Krylov space, got rank {}",
            res1.rank
        );
        assert!(
            ritz1.iter().all(|r| *r > GAP_MARGIN - 1e-6),
            "{name}: every Ritz value bounds the sector levels from above — none may fall
             below the mass gap {GAP_MARGIN}: {ritz1:?}"
        );
        // Convergence: the lowest Ritz value has moved DOWN from the start's
        // own Rayleigh quotient toward the exact gap λ_min(h⁺) = margin.
        let rq = QuantumState::inner_product(&start, &full.apply(&start)).re;
        assert!(
            ritz1[0] < rq,
            "{name}: the window must converge down toward the gap: Ritz[0]={} vs Rayleigh {rq}",
            ritz1[0]
        );
    }
}

/// Cross-check of the doctrine against the INNER-level view: the truncated
/// one-particle spectrum of the gauge-fixed QYM Hamiltonian genuinely reaches
/// below zero (the pair-squeezed floors measured by `qym_mass_gap.rs`), and
/// it is EXACTLY the added constant — nothing else — that lifts it. The
/// doctrine's ground-state claim is about the nested (outer) theory; this
/// test pins what the constant is compensating.
#[test]
fn qym_one_particle_floor_requires_the_constant() {
    let basis = inner_basis(4, 3); // deeper inner truncation: the squeeze deepens
    for g in [0.0_f64, 1.0, 2.0] {
        let h_mat = inner_matrix(&qcd_ym_hamiltonian(g), &basis);
        assert!(hermiticity(&h_mat) < 1e-9);
        let floor = min_eig(&h_mat);
        assert!(
            floor < -1e-3,
            "QYM(g={g}): the un-shifted one-particle floor must be negative (the pair-squeezed level), got {floor}"
        );
        // The abelian zero mode at g=0 sits at the stripped-continuum floor
        // −zp(g) with zp(0) = 2 (the ½π² + ½B² zero point); the coupling
        // deepens it (g²/8 from ‖B|0⟩‖² = 2 + g²/4, halved by the ½B²).
        let zp = 2.0 + g * g / 8.0;
        assert!(
            floor > -zp - 1e-6,
            "QYM(g={g}): floor must stay above the restored zero-point bound −zp = {zp}, got {floor}"
        );
    }
}
