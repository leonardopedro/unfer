#!/usr/bin/env bash
# End-to-end demo for the Quantum Flow Matching — Tomographic Subspace
# Recovery (QFM-TSR) kernel module (QFM.tex §7):
#   1. Builds the unfer kernel FFI (uk_* symbols) and the safestos
#      cranelift bridge.
#   2. Runs QfmTomoModule through the CPS-JIT: it builds a
#      `qfm_tomography` ModelSpec (the 4-point tetrahedron training set
#      + the k/K_2/krylov_dim sketch params), JIT-creates the model
#      (uk_model_create), runs the 4-phase generate (uk_evolve with a
#      `query` field), and reads back the generated image (uk_get_result)
#      — all in-process (positive path).
#   3. Exercises the manifest authorization gate with `modhost` (UK-4001
#      negative test): granting vs. revoking `uk_get_result`, which the
#      QFM-TSR demo needs to drain the EvolveReport.
#
# This module ships inside the unfer repo (unfer/qfm_tomo_module); it
# expects the australVM compiler checkout as a sibling of unfer
# ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # qfm_tomo_module lives inside the unfer repo
ROOT="$(cd "$UNFER_DIR/.." && pwd)"      # parent holding the sibling australVM checkout
AUSTRAL_DIR="$ROOT/australVM"
CL_DIR="$AUSTRAL_DIR/safestos/cranelift"

[ -d "$AUSTRAL_DIR" ] || { echo "ERROR: expected sibling australVM at $AUSTRAL_DIR" >&2; exit 1; }
[ -d "$UNFER_DIR" ]   || { echo "ERROR: expected unfer repo at $UNFER_DIR" >&2; exit 1; }

echo "============================================================"
echo " 1. Build unfer kernel FFI (uk_* symbols)"
echo "============================================================"
( cd "$UNFER_DIR" && cargo build --release -p unfer_ffi )

echo "============================================================"
echo " 2. Build safestos cranelift bridge (+ modhost) with the unfer kernel"
echo "    NOTE: built with AllowAll default auth (--no-default-features) so the"
echo "    live JIT execution demo can call the kernel; the manifest gate is"
echo "    exercised explicitly by modhost below."
echo "============================================================"
( cd "$CL_DIR" && cargo build --release --no-default-features --features unfer-kernel )
( cd "$CL_DIR" && cargo build --release --no-default-features --features unfer-kernel --bin modhost )
MODHOST="$CL_DIR/target/release/modhost"
LIBDIR="$CL_DIR/target/release"

echo "============================================================"
echo " 3. Build the Austral compiler"
echo "============================================================"
( cd "$AUSTRAL_DIR" && dune build lib/ bin/ )
AUSTRAL="$AUSTRAL_DIR/_build/default/bin/austral.exe"

echo "============================================================"
echo " 4. POSITIVE: run qfm_tomo_module through the CPS-JIT"
echo "    The module builds a qfm_tomography JSON ModelSpec, JIT-creates"
echo "    a real kernel model in-process (uk_model_create), runs the"
echo "    4-phase QFM-TSR generate (uk_evolve with a query), and reads back"
echo "    the generated image (uk_get_result). run() returns 1 only if the"
echo "    model was actually created (handle > 0) and the EvolveReport was"
echo "    successfully drained; a parse/create/fill failure returns the"
echo "    negative UK code."
echo "============================================================"
JIT_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/QfmTomoModule.aui,$HERE/src/QfmTomoModule.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
echo "$JIT_OUT"
if echo "$JIT_OUT" | grep -q "CPS JIT: Execution result: 1"; then
  echo "PASS: qfm_tomo_module JIT-created a qfm_tomography model, generated an image, and drained the EvolveReport."
else
  echo "FAIL: expected the JIT to create+evolve+read the QFM-TSR model (Execution result: 1)." >&2
  exit 1
fi

echo "============================================================"
echo " 5. AUTHORIZATION: manifest grant vs. revocation (UK-4001)"
echo "============================================================"
# 5a. With the full manifest, uk_get_result is granted -> ALLOW.
if "$MODHOST" authorize "$HERE/module.toml" qfm_tomo_module uk_get_result; then
  echo "PASS: granted manifest authorizes uk_get_result (QFM-TSR EvolveReport read)."
else
  echo "FAIL: full manifest should authorize uk_get_result." >&2
  exit 1
fi

# 5b. Strip uk_get_result from the grants -> DENY (UK-4001). Without it,
#     the QFM-TSR module cannot drain the generated image.
STRIPPED="$(mktemp)"
trap 'rm -f "$STRIPPED"' EXIT
grep -v '"uk_get_result"' "$HERE/module.toml" > "$STRIPPED"
if "$MODHOST" authorize "$STRIPPED" qfm_tomo_module uk_get_result; then
  echo "FAIL: stripped manifest must DENY uk_get_result." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_get_result with UK-4001."
fi

echo "============================================================"
echo " QFM-TSR DEMO COMPLETE: qfm_tomography model created, generated, and"
echo "                        drained in-process; authorization gate enforced."
echo "============================================================"
