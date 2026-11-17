/-
N8 Duhamel representation and conditional defect estimate.

This file formalizes the variation-of-constants step for `u' = -A u + b`:

  u t = exp(-t A) u 0 + ∫ s in 0..t, exp(-(t-s) A) b s,

and the resulting N8-shaped bound

  ||u t|| <= leakage + t * defect * seed * exp(omega * t),

conditional on an initial bound, a source bound, and the contractive flow bound
`||exp(-(t-s) A)|| <= 1`. For the actual N8 curve the flow is the exponential of
the skew-Hermitian matrix `A = iH`; deriving that contraction from the matrix
hypotheses and identifying the physical reduction remain open. A separate
generic Gronwall estimate is also included for comparison, but it is not the
sharp N8 derivation. This file is deliberately separate from `SmDefect.lean`,
which checks only the finite 16-decimal endpoint arithmetic with `decide`.
-/
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.CStarAlgebra.Basic
import Mathlib.Analysis.CStarAlgebra.Classes
import Mathlib.Analysis.Normed.Algebra.Exponential
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

open Set intervalIntegral MeasureTheory
open scoped Real Interval

namespace BookProof.Sm.ChapterSmDefectAnalytic

/-- A continuous curve on `[0, t]` whose right derivative is bounded by
`K * ||r x|| + eps` obeys the Gronwall bound. This is a generic comparison
lemma; the sharp N8 derivation is `duhamel_representation` below. -/
theorem norm_le_gronwall_of_deriv_bound
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {r r' : ℝ → E} {δ K eps t : ℝ}
    (hcont : ContinuousOn r (Icc 0 t))
    (hderiv : ∀ x ∈ Ico 0 t, HasDerivWithinAt r (r' x) (Ici x) x)
    (h0 : ‖r 0‖ ≤ δ)
    (hbound : ∀ x ∈ Ico 0 t, ‖r' x‖ ≤ K * ‖r x‖ + eps)
    (ht : 0 ≤ t) :
    ‖r t‖ ≤ gronwallBound δ K eps t := by
  simpa using
    norm_le_gronwallBound_of_norm_deriv_right_le hcont hderiv h0 hbound t
      ⟨ht, le_rfl⟩

/-- Generic Gronwall defect estimate, conditional on the curve's derivative
bound. This is a comparison lemma, not the sharp N8 derivation; see
`norm_duhamel_representation_le`. Here `defect`, `seed`, `omega` are real data
and the exponential growth is evaluated at the final time. -/
theorem finite_map_defect_of_deriv_bound
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {r r' : ℝ → E} {leakage defect seed omega t : ℝ}
    (hcont : ContinuousOn r (Icc 0 t))
    (hderiv : ∀ x ∈ Ico 0 t, HasDerivWithinAt r (r' x) (Ici x) x)
    (h0 : ‖r 0‖ ≤ leakage)
    (hbound : ∀ x ∈ Ico 0 t,
      ‖r' x‖ ≤ defect * seed * Real.exp (omega * t))
    (ht : 0 ≤ t) :
    ‖r t‖ ≤ leakage + t * defect * seed * Real.exp (omega * t) := by
  have hg := norm_le_gronwall_of_deriv_bound
    (K := 0) (eps := defect * seed * Real.exp (omega * t))
    hcont hderiv h0 ?_ ht
  · simpa [gronwallBound_K0, add_comm, add_left_comm, add_assoc, mul_comm,
      mul_left_comm, mul_assoc] using hg
  · intro x hx
    simpa using hbound x hx

/-- Norm bound for the Duhamel integral that actually appears in the N8 estimate:
if the integrand is bounded by `defect * seed * exp (omega * t)` on the
interval, the integral contributes `t` times that bound. -/
theorem norm_duhamel_integral_le
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {F : ℝ → E} {defect seed omega t : ℝ}
    (ht : 0 ≤ t)
    (hF : ∀ s ∈ Ι (0 : ℝ) t,
      ‖F s‖ ≤ defect * seed * Real.exp (omega * t)) :
    ‖∫ s in (0 : ℝ)..t, F s‖ ≤ t * defect * seed * Real.exp (omega * t) := by
  have h := intervalIntegral.norm_integral_le_of_norm_le_const hF
  simpa [sub_zero, abs_of_nonneg ht, mul_comm, mul_left_comm, mul_assoc] using h

section Duhamel

variable {𝔸 : Type*} [NormedRing 𝔸] [NormedAlgebra ℚ 𝔸] [NormedAlgebra ℝ 𝔸]
  [CompleteSpace 𝔸]

/-- Derivative of the integrating factor `s ↦ exp (-(t - s) • A)`. -/
theorem hasDerivAt_exp_neg_sub_smul (A : 𝔸) (t s : ℝ) :
    HasDerivAt (fun s : ℝ => NormedSpace.exp (-(t - s) • A))
      (NormedSpace.exp (-(t - s) • A) * A) s := by
  have hinner : HasDerivAt (fun s : ℝ => -(t - s)) 1 s := by
    have h1 : HasDerivAt (fun s : ℝ => t - s) (-1) s :=
      (hasDerivAt_id s).const_sub t
    convert h1.neg using 1 <;> ring
  have hExpDeriv : HasDerivAt (fun u : ℝ => NormedSpace.exp (u • A))
      (NormedSpace.exp (-(t - s) • A) * A) (-(t - s)) :=
    hasDerivAt_exp_smul_const (𝕂 := ℝ) (𝔸 := 𝔸) A (-(t - s))
  have h := hExpDeriv.scomp (x := s) hinner
  simpa using h

/-- Duhamel/variation-of-constants representation for `u' = -A u + b`:
`u t = exp (-t A) u 0 + ∫ s in 0..t, exp (-(t-s) A) b s`. This is the matrix
representation that the earlier Gronwall-only formalization was missing. -/
theorem duhamel_representation
    (A : 𝔸) (u b : ℝ → 𝔸) {t : ℝ} (ht : 0 ≤ t)
    (hcontu : ContinuousOn u (Icc 0 t))
    (hu : ∀ s ∈ Ioo 0 t, HasDerivAt u (-(A * u s) + b s) s)
    (hb : ContinuousOn b (Icc 0 t)) :
    u t = NormedSpace.exp (-t • A) * u 0 +
      ∫ s in (0 : ℝ)..t, NormedSpace.exp (-(t - s) • A) * b s := by
  let Φ : ℝ → 𝔸 := fun s => NormedSpace.exp (-(t - s) • A) * u s
  have hExpCont : Continuous fun s : ℝ => NormedSpace.exp (-(t - s) • A) := by
    fun_prop
  have hΦcont : ContinuousOn Φ (Icc 0 t) :=
    hExpCont.continuousOn.mul hcontu
  have hΦderiv : ∀ s ∈ Ioo 0 t, HasDerivAt Φ
      (NormedSpace.exp (-(t - s) • A) * b s) s := by
    intro s hs
    have h := (hasDerivAt_exp_neg_sub_smul A t s).mul (hu s hs)
    convert h using 1
    · dsimp only [Φ]
      noncomm_ring
  have hb' : ContinuousOn b (uIcc 0 t) := by
    rwa [uIcc_of_le ht]
  have hint : IntervalIntegrable
      (fun s => NormedSpace.exp (-(t - s) • A) * b s) volume 0 t := by
    apply ContinuousOn.intervalIntegrable
    exact hExpCont.continuousOn.mul hb'
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le
    (a := 0) (b := t) (f := Φ)
    (f' := fun s => NormedSpace.exp (-(t - s) • A) * b s)
    ht hΦcont hΦderiv hint
  rw [hFTC]
  simp only [Φ, sub_self, sub_zero, neg_zero, zero_smul,
    NormedSpace.exp_zero, one_mul]
  abel

/-- The Duhamel representation plus the flow and source bounds gives the N8
estimate. The matrix-specific input is exactly `hflow` (the exponential is a
contraction on the interval); everything else is now derived. -/
theorem norm_duhamel_representation_le
    (A : 𝔸) (u b : ℝ → 𝔸) {t leakage defect seed omega : ℝ} (ht : 0 ≤ t)
    (hcontu : ContinuousOn u (Icc 0 t))
    (hu : ∀ s ∈ Ioo 0 t, HasDerivAt u (-(A * u s) + b s) s)
    (hb : ContinuousOn b (Icc 0 t))
    (h0 : ‖NormedSpace.exp (-t • A) * u 0‖ ≤ leakage)
    (hflow : ∀ s ∈ Icc 0 t, ‖NormedSpace.exp (-(t - s) • A)‖ ≤ 1)
    (hbnd : ∀ s ∈ Icc 0 t, ‖b s‖ ≤ defect * seed * Real.exp (omega * t)) :
    ‖u t‖ ≤ leakage + t * defect * seed * Real.exp (omega * t) := by
  rw [duhamel_representation A u b ht hcontu hu hb]
  refine norm_add_le_of_le h0 ?_
  have hsub : Ι (0 : ℝ) t ⊆ Icc 0 t := by
    rw [← uIcc_of_le ht]
    exact uIoc_subset_uIcc
  apply norm_duhamel_integral_le ht
  intro s hs
  exact (norm_mul_le_of_le (hflow s (hsub hs))
    (hbnd s (hsub hs))).trans_eq (one_mul _)

end Duhamel

section SkewFlow

variable {𝔸 : Type*} [CStarAlgebra 𝔸] [Nontrivial 𝔸]

/-- The exponential of a skew-adjoint element is unitary, hence has norm one;
this discharges the N8 contractive-flow hypothesis for `A = iH` with `H`
self-adjoint (`star H = H` implies `star (iH) = -iH`). -/
theorem norm_exp_neg_smul_skew (A : 𝔸) (hA : star A = -A) (r : ℝ) :
    ‖NormedSpace.exp ((-r) • A)‖ ≤ 1 := by
  letI : NormedAlgebra ℚ 𝔸 := NormedAlgebra.restrictScalars ℚ ℂ 𝔸
  have hskew : ((-r) • A) ∈ skewAdjoint 𝔸 := by
    rw [skewAdjoint.mem_iff]
    simp [star_smul, hA]
  have hu : NormedSpace.exp ((-r) • A) ∈ unitary 𝔸 :=
    NormedSpace.exp_mem_unitary_of_mem_skewAdjoint hskew
  rw [CStarRing.norm_of_mem_unitary hu]

/-- N8-shaped Duhamel bound with the contractive-flow hypothesis discharged for a
skew-adjoint generator. -/
theorem norm_duhamel_representation_le_of_skew
    (A : 𝔸) (hA : star A = -A) (u b : ℝ → 𝔸)
    {t leakage defect seed omega : ℝ} (ht : 0 ≤ t)
    (hcontu : ContinuousOn u (Icc 0 t))
    (hu : ∀ s ∈ Ioo 0 t, HasDerivAt u (-(A * u s) + b s) s)
    (hb : ContinuousOn b (Icc 0 t))
    (h0 : ‖NormedSpace.exp (-t • A) * u 0‖ ≤ leakage)
    (hbnd : ∀ s ∈ Icc 0 t, ‖b s‖ ≤ defect * seed * Real.exp (omega * t)) :
    ‖u t‖ ≤ leakage + t * defect * seed * Real.exp (omega * t) := by
  letI : NormedAlgebra ℚ 𝔸 := NormedAlgebra.restrictScalars ℚ ℂ 𝔸
  apply norm_duhamel_representation_le A u b ht hcontu hu hb h0 ?_ hbnd
  intro s _hs
  simpa using norm_exp_neg_smul_skew A hA (t - s)

end SkewFlow

end BookProof.Sm.ChapterSmDefectAnalytic
