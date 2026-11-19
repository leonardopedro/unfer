/-
N8 finite matrix: `A = iH` is skew-adjoint.

This checks the matrix-specific identification that `SmDefectAnalytic.lean`
leaves open: the declared 4x4 N8 fixture `H` is Hermitian, so the generator
`A = i • H` satisfies `Aᴴ = -A`.  It does not identify the physical reduction
or the full-action operator with this fixture; those remain open.

The entries are the exact rationals of
`fock_sirk::sm_defect::finite_fixture_inputs`.
-/
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.LinearAlgebra.Matrix.Notation

namespace BookProof.Sm.ChapterSmDefectSkew

open Matrix

noncomputable section

/-- The declared 4x4 Hermitian N8 fixture, with exact rational entries. -/
def H : Matrix (Fin 4) (Fin 4) ℂ :=
  !![(7 / 10 : ℂ), (3 / 10 : ℂ) * Complex.I, (1 / 5 : ℂ), 0;
     (-3 / 10 : ℂ) * Complex.I, (11 / 10 : ℂ), 0, (3 / 20 : ℂ);
     (1 / 5 : ℂ), 0, (-2 / 5 : ℂ), (-1 / 4 : ℂ) * Complex.I;
     0, (3 / 20 : ℂ), (1 / 4 : ℂ) * Complex.I, (9 / 10 : ℂ)]

/-- The N8 generator `A = iH`. -/
def A : Matrix (Fin 4) (Fin 4) ℂ := Complex.I • H

/-- The declared finite N8 matrix is Hermitian. -/
theorem H_hermitian : Hᴴ = H := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [H, Matrix.conjTranspose_apply, Complex.ext_iff] <;> norm_num

/-- **`A = iH` is skew-adjoint**: `Aᴴ = -A`.  This is the matrix-specific
hypothesis consumed by `norm_duhamel_representation_le_of_skew`. -/
theorem A_skew : Aᴴ = -A := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [A, H, Matrix.conjTranspose_apply, Complex.ext_iff] <;> norm_num

end

end BookProof.Sm.ChapterSmDefectSkew
