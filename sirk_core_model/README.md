# sirk_core_model — the Aeneas seam for the SIRK numeric core (mass-gap plan T9)

## Regeneration status

This model and its generated `.llbc`/Lean artifacts must be regenerated after
any change affecting the numerical Hamiltonian, outer-enclosure construction,
solver, or certificate seam. Existing generated outputs do not certify a later
Rust revision. After regeneration, re-export the Lean certificate and rerun
nanoda before claiming certification for the corrected model.

Pure, dependency-free Rust model of the SIRK **shift-invert rational Krylov**
numeric core, written in the Aeneas-supported Rust subset so that Charon +
Aeneas translate it mechanically into Lean 4.  This is the **T9** deliverable
of `../timepiece/CONSOLIDATED_PLAN.md` §13.7 and `MASS_GAP_CERTIFIED.md` §5.3.

## What the model covers

Each function mirrors the running kernel (`fock_sirk/src/forward_sirk.rs`,
`fock_sirk/src/linalg.rs`) but without `nalgebra`, `candle`, GPU or I/O:

| Function | SIRK step | Plan target |
| :--- | :--- | :--- |
| `forward_step` / `forward_sequence` | the Krylov recurrence `w_{k+1} = (H − z_k I) w_k` | the forward sequence as a fold |
| `gram_entry` / `gram_assembly` | `G_jk = ⟨w_j \| w_k⟩` | the Gram assembly |
| `projection_identity` | `H_jk = σ_{k+1} G_{j,k+1} + z_k G_{j,k}` | the projection identity |
| `whitening_transform` | `T = V · diag(λ^{−1/2})` (eigendecomposition trusted) | the whitening transform `T` with `T* Ĝ T = I` |
| `residual_boundary_component` | `e_m = τ_m c_{m−1}` | the residual formula `‖r‖ = \|τ_m c_{m−1}\|` |
| `residual_norm2` | `‖Hψ − θψ‖² = e† G e` | cancellation-free residual norm |

Five Rust unit tests pin the contracts (basis length, Gram Hermitian
symmetry, projection-identity linearity, `T* Ĝ T = I` on `diag(4,1)`, boundary
component).

## The honesty boundary (recorded verbatim from the plan)

> the dense eigendecomposition (LAPACK-style call) remains a trusted component
> whose backward error is exactly the T3 bound. Aeneas verifies the
> *algorithm*; the *rounding* is enclosed by T1–T5.

Concretely: Aeneas's Lean backend translates the **algorithmic structure**
verbatim — the two `forward_sequence` loops, the flattened `gram_assembly` /
`whitening_transform` loops, the `C64` structure and its `Clone` instance, the
`Result`/`ControlFlow`/`Vec` machinery — while the `f64` arithmetic leaves
(`conj`, the arithmetic bodies) stay opaque (`sorry`) in the generated file.
Those leaves are the rounding layer, which `BookProof/ChapterSirkFinitePrecision`
(T1–T5) encloses; Aeneas is not asked to model them.

## Toolchain & regeneration

Regenerate with:

```bash
./scripts/aeneas_sirk.sh
```

This needs the Aeneas **nightly release bundle** (`aeneas-linux-x86_64.tar.gz`),
which carries matching `aeneas`, `charon`, and the `lean-build-aeneas` Lean
library:

- `AENEAS_ROOT` — path to the extracted bundle (default
  `$HOME/Projects/.toolchain/aeneas-bin`).
- Charon runs `rustc` with the pinned nightly in the bundle's
  `rust-toolchain` (`nightly-2026-08-18`; needs `rustup component add
  rustc-dev`).

Outputs:

- `aeneas/sirk_core_model.llbc` — the Charon IR (committed).
- `aeneas/SirkCoreModel.lean` — the generated Lean 4 model (committed).

## Integrating the generated model (Lean 4 specialist)

The generated `SirkCoreModel.lean` imports `Aeneas` (the Aeneas Std library),
which is built against **Lean 4.31.0 + mathlib 4.31.0** (see
`backends/lean/lakefile.lean` in the bundle; the `lean-build-aeneas` asset is
the precompiled `Aeneas` oleans).  This is a *separate* lake project from the
timepiece `RiemannProof` (which pins 4.28.0): create a small lake project with
`require mathlib @ v4.31.0` and `require aeneas from git` (or copy the
prebuilt `lean-build` oleans into `.lake/lib/lean`), then:

```lean
import SirkCoreModel   -- the generated file
```

and prove the algebraic identities the plan lists: the projection identity
`H_jk = σ_{k+1} G_{j,k+1} + z_k G_{j,k}`, Gram Hermitian symmetry
`G_kj = conj(G_jk)`, the whitening identity `T* Ĝ T = I`, and the residual
boundary component `e_m = τ_m c_{m−1}`.