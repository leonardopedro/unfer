# SM gauge-fixed scalar templates: execution record

2026-11-09c. Cadabra2 2.5.14 ran `docs/sm_gauge_fixed_hamiltonian.cdb`:
19 CHECK rows, all zero identities passed. This includes the corrected
three-family/Higgs/Dirac slot assembly, U(1) Legendre coefficients, real-doublet
normalization, retained Higgs constant and three missing-slot probes.

Runtime fixes: use ordinary Python `Ex` rules inside the omission-probe loop
(the multiline `$...$` substitution was invalid), and sort the constant-cancellation
products before collecting terms. The previous pending status is superseded.

These are scalar assembly templates, not a full spinor/operator derivation or
ESA proof. The complete 96-coordinate matter dictionary and its actual LaTeX
export/import are now in [SM_FULL_IMPLEMENTATION.md](SM_FULL_IMPLEMENTATION.md).
`bash scripts/check-sm-symbolic` reruns both scopes without confusing them.
