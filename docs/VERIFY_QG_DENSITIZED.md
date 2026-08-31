# Verification: the densitized 3D gauge-fixed QG Hamiltonian (book.tex 8182 → 8190 → flat)

This is the evidence base for the coefficients claimed in
`timepiece/BookProof/ChapterQuantumGravityDensitized.lean` and
`ChapterQuantumGravity3DGauge` (`qg3DDensity`, `qgSymbol`, `qgKappa`), i.e.

    H = (1/16e) 𝒮² − (1/24e) 𝒫² + ½ 𝒮·E + ⅓ 𝒫·E − e(𝒯^ab E_ab + 2 𝒯^ab_b E_a
            + ½ 𝒯_abc 𝒯^acb + ¼ 𝒯_abc 𝒯^abc − 𝒯_ba^a 𝒯^bc_c − ¼ 𝒯_ab 𝒯^ab)   (book.tex 8190)

and its densitized principal part `(1/16) S̃² − (1/24) P̃²`.

Reproducible with Cadabra2 (2.5.14, via the `../unfer` nix flake):

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    $C2 -q -n docs/qg_densitized_hamiltonian.cdb   # existing notebook: change of variables

## CHECK 1 — the change of variables is an identity (worksheet `qg_dens`)

Starting from the singular kinetic part `(1/16e)𝒮² − (1/24e)𝒫²` (book.tex 8190) and
substituting the densitized quantities

    e = y² ,  𝒮_{ab} = y S̃_{ab} ,  𝒫 = y P̃

Cadabra's `substitute` + `sort_product` + `collect_factors` reduce it to

    (1/16) S̃_{ab} S̃^{ab} · (y)² y⁻² − (1/24) P̃²·(y)² y⁻²   =   (1/16) S̃² − (1/24) P̃²

That is, the singular `1/e` denominator is absorbed (up to the identity `y²·y⁻² = 1`),
exactly the content of `kinetic_absorption` / `conformal_absorption` and the flat
principal symbol `qgSymbol = 1/16 Σ ξ² − 1/24 ξ_y²`, `qgKappa = (1/16, …, −1/24)`.

## CHECK 2 — the coupling coefficients ½ and ⅓ follow from the polymomentum relations

The manuscript's own polymomentum relations (book.tex, after "the polymomentum"):

    S_{ab} = 𝒮_{ab}/(2e) ,   T = −𝒫/(4e)          (𝒮 𝒫 : the densitized momenta)

Substituting these into the book.tex-8182 momentum part

    e( ¼ S_{ab} S^{ab} − ⅔ T² + S^{ab} E_{ab} − ⅘ T E_a^a )

gives, coefficient by coefficient (exact rational arithmetic, via `times` in
`/tmp/qg_num2.py`):

    S̃²  :  e·¼·(1/2e)²            = 1/(16e)         ✓ 1/16
    P̃²  :  e·(−⅔)·(1/4e)²          = −1/(24e)        ✓ −1/24
    S̃·E :  e·(1/2e)                = ½               ✓ 1/2
    P̃·E :  e·(−⅓·… )(−1/4e)        = ⅓               ✓ 1/3

The Cadabra worksheet `qg_verify_all.py` reproduces this symbolically; the residual
"difference" in its CHECK 2 output is only the display form `e·e⁻²` vs the contracted
`(1/e)` — mathematically zero (confirmed by the exact-fraction table).

The `e(…)` torsion block `𝒯^ab E_ab + 2 𝒯^ab_b E_a + ½ 𝒯_abc 𝒯^acb + ¼ 𝒯_abc 𝒯^abc
− 𝒯_ba^a 𝒯^bc_c − ¼ 𝒯_ab 𝒯^ab` is untouched by the momentum substitution (it contains no
`S`/`T`), so it passes straight through — CHECK 3.

## What is an identity vs. what is an input

* **Pure algebra, verified above:** `1/16e` and `−1/24e` densitize to `1/16`, `−1/24`;
  the `½ 𝒮·E + ⅓ 𝒫·E` couplings carry exactly the manuscript's coefficients; the
  torsion block is carried through unchanged.

* **Physical input (boundary/surface term):** the manuscript drops the vector tare
  `p^a T_a` and its partner `−2 𝒯^ab_a T_b e` (from `−L`) in going from the
  polymomentum form (8182) to the 3D-fixed form (8190), with `p^a = 2e 𝒯^{ac}_c` —
  see the explicit `substitute(…, T_a T^{ba}_b -> 0)` in `qg_gauge_fixed_hamiltonian.cdb`
  and the comment justifying it as a surface term (`p^a T_a` is a total divergence /
  acts by zero on physical states of compact support / vanishing spatial fall-off). This
  is the same compact-support/physical-sector input that `BookProof` treats as the named
  hypothesis in `ChapterQuantumGravityDensitized` (Strichartz) and
  `ChapterQgPhysicalSectorIdentity` (the `E = ∂e` fixing reduces to the field values).
  book.tex 8190 itself drops it (its `≈`), and book.tex 8202 keeps it only in the
  unfixed 4D form.

## Files

* `docs/qg_densitized_hamiltonian.cdb` — existing change-of-variables notebook.
* `/tmp/qg_verify_all.py`, `/tmp/qg_num2.py` — the two verification worksheets above
  (trivially re-runnable; they are check scripts, not part of the formalization).