/-
N13 finite one-mode CAR obligation.

This is the Lean counterpart of `docs/fixtures/sm_car_bridge.json`: the same
two-state integer matrices emitted from `nested_fock_algebra::exact_inner`.
The proofs are `decide`/`rfl` only, so the exported terms stay reducible by
nanoda.  Keep the namespace and theorem names in sync with the bridge fixture.
-/
namespace BookProof.Sm.ChapterSmCar

/-- A 2x2 integer matrix, the finite one-mode CAR representation. -/
structure Mat2 where
  e00 : Int
  e01 : Int
  e10 : Int
  e11 : Int
deriving DecidableEq, Repr

namespace Mat2

instance : Zero Mat2 := ⟨0, 0, 0, 0⟩
instance : One Mat2 := ⟨1, 0, 0, 1⟩
instance : Add Mat2 :=
  ⟨fun A B => ⟨A.e00 + B.e00, A.e01 + B.e01, A.e10 + B.e10, A.e11 + B.e11⟩⟩
instance : Neg Mat2 :=
  ⟨fun A => ⟨-A.e00, -A.e01, -A.e10, -A.e11⟩⟩
instance : Sub Mat2 :=
  ⟨fun A B => ⟨A.e00 - B.e00, A.e01 - B.e01, A.e10 - B.e10, A.e11 - B.e11⟩⟩
instance : Mul Mat2 :=
  ⟨fun A B =>
    ⟨A.e00 * B.e00 + A.e01 * B.e10,
     A.e00 * B.e01 + A.e01 * B.e11,
     A.e10 * B.e00 + A.e11 * B.e10,
     A.e10 * B.e01 + A.e11 * B.e11⟩⟩

end Mat2

/-- Annihilation: `|0> ↦ 0`, `|1> ↦ |0>`. -/
def a : Mat2 := ⟨0, 0, 1, 0⟩

/-- Creation: `|0> ↦ |1>`, `|1> ↦ 0`. -/
def adag : Mat2 := ⟨0, 1, 0, 0⟩

/-- `N = a† a`, the occupation number. -/
def number : Mat2 := adag * a

/-- `(-1)^N = 1 - 2N`. -/
def parity : Mat2 := 1 - number - number

theorem one_mode_anticommutator : a * adag + adag * a = 1 := by
  decide

theorem annihilator_square : a * a = 0 := by
  decide

theorem creator_square : adag * adag = 0 := by
  decide

theorem parity_conjugation : parity * a * parity = -a := by
  decide

theorem number_is_adag_a : adag * a = number := by
  rfl

theorem parity_square : parity * parity = 1 := by
  decide

end BookProof.Sm.ChapterSmCar
