/-
N8 finite-map defect endpoint.

This checks the concrete 16-decimal rational endpoint arithmetic of the N8
bridge, not the analytic defect inequality. The analytic step (variation of
constants, norm estimate and exponential bound) is not formalized here; the
bridge record keeps that obligation separate. The proof is `decide` only so the
exported term stays reducible by nanoda.

The scaled inequality is

  5 D^2 measured <= 5 D^2 leakage + defect * seed * flow,

with D = 10^16, i.e.

  measured / D <= leakage / D + (1/5)(defect/D)(seed/D)(flow/D).

The endpoints are the outward decimal ceilings of the N8 interval record.
-/
namespace BookProof.Sm.ChapterSmDefect

/-- Common denominator of the declared 16-decimal endpoints. -/
def D : Int := 10000000000000000

def leakage : Int := 7032108464077437
def defect : Int := 19551214796017202
def seed : Int := 7109813678954653
def flow : Int := 13231298123374382
def measured : Int := 7546019867800926

/-- The finite endpoint comparison for the N8 record. -/
theorem finite_map_endpoint :
    5 * D ^ 2 * measured <= 5 * D ^ 2 * leakage + defect * seed * flow := by
  decide

end BookProof.Sm.ChapterSmDefect
