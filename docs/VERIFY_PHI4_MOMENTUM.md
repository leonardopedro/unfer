# Verification evidence — continuous-momentum λφ⁴ module

Module: `docs/phi4_momentum_nested.cdb`
Fixture: `docs/fixtures/phi4_momentum_mehler.json` (schema `phi4-momentum-cont-exact/v1`)

## Engine

- `cadabra2-cli --version`: `2.5.15 private`
- Exact SymPy arithmetic runs inside the Cadabra2 subprocess; no GPL code is
  linked into the Rust or Lean trees.

## Replay

```bash
cd /home/leo/Projects/unfer
cadabra2-cli -q -n docs/phi4_momentum_nested.cdb | tee /tmp/phi4-mom.log
python3 - <<'PY'
import json
from pathlib import Path
rows = Path("/tmp/phi4-mom.log").read_text().splitlines()
ok_rows = [r for r in rows if "(0 = ok)" in r]
assert len(ok_rows) >= 20
assert all(r.rsplit(":", 1)[1].strip() == "0" for r in ok_rows)
[fix] = [r.split("|", 1)[1] for r in rows if r.startswith("PHI4_MOM_FIXTURE|")]
Path("docs/fixtures/phi4_momentum_mehler.json").write_text(
    json.dumps(json.loads(fix), indent=2, sort_keys=True) + "\n")
print("PASS:", len(ok_rows), "zero checks")
PY
```

`scripts/verify_cdb_modules.sh --strict` reports
`[PASS] phi4_momentum_nested — ran; 32 zero-intent check(s) all reached 0`
and `[PASS] phi4_mehler — ran; 23 zero-intent check(s) all reached 0`.

## Digests

```text
669510d3c55d334b065b0d94c6f7a4d402cdf29408865856178c3b475fb8d709  docs/phi4_momentum_nested.cdb
9b1e9c15ccbb53c876a4818f8d2898d8faec5f26c54d0664406cbcc9912d15bb  docs/fixtures/phi4_momentum_mehler.json
```

## Certified exact content

- Continuous quadratic gradient `½(k₁²+k₂²+k₃²)φ²`: six expanded terms, no
  momentum sum, no spatial-derivative jet constraint.
- Retention of the energy constant `λv⁴/4` and the shifted wall
  `λv²x² + λvx³ + λx⁴/4`.
- Continuous Fourier symbol `‖𝓕(∂ₓf)‖² = (2πk)²‖𝓕f‖²` on the Gaussian
  packet `f = exp(−πx²)`, plus `∫(f²)²dx = ∫|Ψ(Q)|²dQ = 1/2`.
- Unconditional coercion square completion; unit value `7/64` at
  `λ = v = 1, α = 1/2`.
- Exact 11-mode `S₃`-symmetric action of `unitMomCont` with Gram
  `diag(1,2,6,12,24,720,144,72,12,144,24)`, mean `17/8`, variance `553/32`,
  outside-window squared norm `9/2`.
- Rank-2 Hashimoto–SIRK certificate at shifts `(10, 5)`: exact dissipation,
  full 11×2 defect, orthogonal split (`outside[0,0] = 9/2`), rational orbit
  rates, and return interval `[319447/320000, 39999/40000]` at `t = 1/100`.
- Outer `dΓ(h)` vacuum annihilation, shift lift `n·c`, and nonzero
  cross-particle defect `dΓ(B)² − dΓ(B²) = c²N(N−1)`.

## Mutation controls

| mutation | exact difference |
|---|---|
| decouple `k` from `φ` (frozen `jetModel`) — variance | `51/4` (553/32 → 145/32) |
| decouple `k` from `φ` — `He₂(φ)K₂` coupling | `1/2` → `0` |
| drop `λv⁴/4` — `U(0)` | `1/4` |
| drop the double contraction — `φ⁴(1)` | `3` |
