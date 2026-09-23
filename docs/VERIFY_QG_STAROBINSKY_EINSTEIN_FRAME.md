# Verification: the QG R² vielbein exponential-wall inner operator

Evidence that the QG R² inner one-particle operator **as it appears in the proofs** equals the
one **as it is stated in the plan** — and that the Einstein-frame potential the proofs use is
the Einstein-frame potential the Jordan-frame chain actually produces (the one link the
sibling modules left open).

Two halves: a *source-level* half (plan ↔ Lean declarations, read off the definitions, not the
doc-comments) and a *symbolic* half (Cadabra, for the closed form of `V(φ)`).

## 1. The plan's stated inner operator

`timepiece/CONSOLIDATED_PLAN.md` §D5 (the QG instance) states the inner operator of the QG
route as

    H = Σ_{a,b} a†_a h_{ab} a_b ,       h_{ab} = δ_{ab}(−d²/dφ² + φ²/4 + V(φ) + σ_b)
                                                  + A_{ab}·1 + B_{ab}·φ

and repeats it in the state update of 2026-09-02c (§"the outer Hamiltonian is exhibited as a
one-particle operator between a creation and an annihilation operator").  The same kernel is
in `timepiece/HAMILTONIAN_AUDIT_20260918.md` §4.1 (the fibre comparison), §8 (the comparison
operator of record, `N = ⊕ₐ Friedrichs(−d²/dφ² + φ²/4 + V(φ) + σ_a)`, `c = 6·K_Q`) and §9 (the
"why these are the intended Hamiltonians" table).  `V` there is the **full exponential**
Einstein-frame Starobinsky wall.

## 2. The kernel in the proofs — source-checked, declaration by declaration

Every piece of `h_{ab}` is a definition or theorem in the tree, and each matches the plan
term for term.

| plan term | Lean declaration | file:line | matches |
| :-- | :-- | :-- | :-- |
| `V(φ)` (full exponential) | `starobinskyV M alpha phi = M^4/(16*alpha) * (1 - Real.exp (-(Real.sqrt (2/3)) * phi / M)) ^ 2` | `BookProof/ChapterStarobinskyPotential.lean:95` | ✓ literal |
| `φ²/4 + V(φ) + s` | `WallPot.pot : ℝ → ℝ := fun x => x ^ 2 / 4 + (W.V x + s)` | `BookProof/ChapterScalaronFiberFL/Part2.lean:88` | ✓ literal |
| `−d²/dφ² + φ²/4 + V(φ) + σ_a` | `WallPot.ham s : ccDomain ℝ →ₗ[ℂ] L2R := wallHam (W.pot s) (W.pot_smooth s)` (`wallHam V = kinCcR + opCc V = −d²/dφ² + V`) | `ChapterScalaronFiberFL/Part2.lean:112`; `ChapterScalaronWallEsa.lean:190` | ✓ literal |
| the physical wall instance | `starobinskyWall M alpha halpha : WallPot` with `V := starobinskyV M alpha` | `ChapterScalaronFiberFL/Part2.lean:78` | ✓ literal |
| `σ_b` | `Q.sig b` (`QgModeData.sig`, with `one_le_sig`) | `ChapterScalaronOuterFockFL/Part1.lean:206` | ✓ literal |
| `A_{ab}·1` | `Q.A a b • (ccDomain ℝ).subtype` (`secA Q = modeOp Q subtype Q.A`) | `ChapterQgOuterFockOneParticle.lean:57`; `ChapterScalaronOuterFockFL/Part2.lean:104` | ✓ literal |
| `B_{ab}·φ` | `Q.B a b • xCc u` (`secB Q = modeOp Q xCc Q.B`; `xCc` = multiplication by the scalaron field) | `ChapterQgOuterFockOneParticle.lean:57`; `ChapterScalaronOuterFockFL/Part2.lean:108` | ✓ literal |
| `H = Σ a†_a h_{ab} a_b` | `secHam W Q = secDiag W Q + secA Q + secB Q`; `secHam_single`, `secHam_matrix_element` | `ChapterScalaronOuterFockFL/Part2.lean:113`; `ChapterQgOuterFockOneParticle.lean:74,126` | ✓ literal |
| Hermiticity of the kernel | `oneParticleOp_herm` | `ChapterQgOuterFockOneParticle.lean:133` | ✓ |
| kernel = the full operator on the core | `secHam_eq_sum_oneParticle` | `ChapterQgOuterFockOneParticle.lean:155` | ✓ |
| the **physical** instance's kernel | `starobinsky_qgContinuum_matrix_element` — `⟪a†_a v, H a†_b u⟫ = ⟪v, (δ_{ab} (starobinskyWall …).ham (cSig b) u) + contTorsionGram a b • u + contCoupling g a b • xCc u⟫` | `ChapterQgOuterFockOneParticle.lean:187` | ✓ literal |

and the mode data of the model supplies exactly the plan's `σ, A, B`:

```lean
def qgFullModes (g : ℝ) : QgModeData GMode := ofBounds gSig … gGram (gCoupling g) gNbr … (855 + |g|) 36 …
@[simp] theorem qgFullModes_sig (g) : (qgFullModes g).sig = gSig          := rfl
@[simp] theorem qgFullModes_A   (g) : (qgFullModes g).A   = gGram         := rfl
@[simp] theorem qgFullModes_B   (g) : (qgFullModes g).B   = gCoupling g   := rfl
```

(`ChapterQgVielbeinScalaronGaugeFL/Part2.lean:145,176,178,180`), with the physical instance
`starobinsky_qgFull_esa … = qgFull_esa_farisLavine (starobinskyWall M alpha halpha) g`
(`:229`, `:199`).

**Conclusion of the source check: the kernel in the proofs is the plan's stated kernel,
term for term and coefficient for coefficient.**  There is nothing to repair on this axis.

## 3. The one link that was *asserted*, not derived — now closed

The Jordan-frame chain is certified in the sibling modules
(`qg_starobinsky_hamiltonian.cdb`, `qg_starobinsky_vielbein_hamiltonian.cdb`:
`H_final_st = (M²/2)ψ·(8190) + U(ψ)e`, `fR_check`, `R2_check`).  But `starobinskyV` — the
Einstein-frame potential the proofs actually use — was **defined** there, not derived from
`U(ψ) = (M⁴/16α)(ψ−1)²`.  The identity the two halves of the tree share was therefore
carried on assertion:

    U(ψ)/ψ²  at  ψ = e^{√(2/3)φ/M}    =    (M⁴/16α)(1 − e^{−√(2/3)φ/M})²

(equivalently, `ψ = e^{√(2/3)φ/M}` with the conformal weight `ψ²` of the potential).

`docs/qg_starobinsky_einstein_frame.cdb` (new) closes it.  Checks, all reducing to `0`:

| check | content | value |
| :-- | :-- | :-- |
| `measure_check` | `√(−g) = ψ² √(−g̃)` under `g = ψ g̃` (d = 4), i.e. the potential term picks up `ψ^{−2}` | `0` |
| `Vweight_check` | `(ψ−1)² = (1 − ψ⁻¹)² ψ²` — the ψ²-cleared form of `V = U/ψ² = (M⁴/16α)(1 − 1/ψ)²` | `0` |
| `closed_check` | `(1 − ψ⁻¹)²` at `ψ = e^{x}` equals `(1 − e^{−x})²`, `x = √(2/3)φ/M` | `0` |
| `Vphi_check` | the closed form is **literally** `starobinskyV` (same `M, α, φ`) | `0` |
| `sqrt_check` | `√(2/3)² = 2/3` | `0` |
| `norm_check` | `√(3/2)·√(2/3) = 1` (i.e. `φ = √(3/2) M lnψ` ⇔ `ψ = e^{√(2/3)φ/M}`) | `0` |
| `kin_check` | Brans–Dicke ω = 0 kinetic `(3M²/4)(∂lnψ)²` becomes the canonical `½(∂φ)²` | `0` |
| `Vzero_check` | `V(0) = 0` (`starobinskyV_zero`) | `0` |
| `Vplat_check` | plateau: `V → M⁴/(16α)` as `φ → +∞` (`starobinskyV_tendsto_plateau`) | `M⁴/(16α)` |
| `m2_check` | small-field term: `m² = M²/(12α)`; with `α = M²/(12m²)` this is `m = M` | `0` |

Reproduce:

    nix build "github:NixOS/nixpkgs/b5aa0fbd538984f6e3d201be0005b4463d8b09f8#cadabra2"
    C2=/nix/store/rpdv12r5grn47rixhdiydxq675f8h5i0-cadabra2-2.5.14-p1/bin/cadabra2-cli
    $C2 -q -n docs/qg_starobinsky_einstein_frame.cdb        # exit 0

### What is an identity vs. what is an input

* **Pure algebra, verified above:** the conformal weight of the potential (`ψ²`), the
  closed form of `V` and its endpoints, `√(2/3)² = 2/3`, `√(3/2)√(2/3) = 1`, and the
  kinetic coefficient `(3M²/4)(∂lnψ)² = ½(∂φ)²` at `φ = √(3/2)M lnψ`.
* **Standard input (stated, not re-derived here):** the Brans–Dicke ω = 0 conformal kinetic
  coefficient `3/2` (equivalently the `√(2/3)` in the field normalization) that comes from
  the conformal transformation of `R` in d = 4.  The module certifies every algebraic
  consequence of it; the coefficient itself is the textbook 4D conformal-transformation
  input, the same category as the surface-term input the sibling modules flag.

## 4. Why `φ²/4` is in the fibre (and why that is not a mismatch)

The plan states the fibre with a quadratic `φ²/4`.  This is the project's **conformal-mode /
Hermite oscillator convention** `−Δ + x²/4` used throughout (the Hermite functions
`Hₙ(x)e^{−x²/4}` define the working core — `HERMITE_CORE_STRICHARTZ.md`,
`STRICHARTZ_WAVE_ESA.md`), not a term absent from the physics: it is a confining addition
inside the fibre operator that makes the fibre a *positive* operator whose Friedrichs
extension is unconditional, so that the Faris–Lavine comparison `N` is built without any
relative-bound hypothesis on the exponential wall (the wall sits **inside** `N`; the only
relative bound `K = 1 + 3·K_Q` is a statement about the vielbein matrices).  The plan, the
audit and the proofs all carry the same `φ²/4`, so there is no divergence here.

**Reconciliation with `CONSOLIDATED_PLAN.md` QG-3.**  QG-3 models a candidate fibre
`h_ψ = −c·d²/dφ² + starobinskyV(φ)` (the "shear oscillators ⊕ scalaron" list) and states
explicitly that it is **not** the Hamiltonian — at most a derived reduction or a comparison
object.  That is a *different object* from the operator of record here (the `qgFullModes`
kernel with its `φ²/4 + σ_a` and the `A, B` couplings), and the plan labels it as such.  The
QG R² operator whose kernel §1 states, and whose ESA `qgFull_esa_farisLavine` proves, is the
one checked above.

## Files

* `docs/qg_starobinsky_einstein_frame.cdb` — the new module (this link).
* `docs/qg_starobinsky_hamiltonian.cdb`, `docs/qg_starobinsky_vielbein_hamiltonian.cdb` — the
  Jordan-frame chain it continues (`action_R2_check`, `H_final_st`).
* `docs/VERIFY_QG_STAROBINSKY_VIELBEIN.md`, `docs/VERIFY_QG_DENSITIZED.md` — the sibling
  evidence bases; both list the conformal rescaling as out of scope.
* `timepiece/HAMILTONIAN_AUDIT_20260918.md` §4/§8/§9, `timepiece/CONSOLIDATED_PLAN.md` §D5 /
  2026-09-02c — the plan side of the comparison.
