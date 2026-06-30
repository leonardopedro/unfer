#!/usr/bin/env bash
# End-to-end demo for the Quantum Flow Matching — Bayesian Update on
# the TSR-evolved prior kernel module (QMF.tex §8, P6 H follow-on):
#   1. Builds the unfer kernel FFI (uk_* symbols) and the safestos
#      cranelift bridge.
#   2. Runs BayesUpdateModule through the CPS-JIT: it builds a
#      `qfm_tomography` ModelSpec (the 4-point tetrahedron training
#      set with krylov_dim=4 for the P6 G SIRK-whitened basis), JIT-
#      creates the model (uk_model_create), runs a single Bayesian
#      update (uk_bayesian_update with a single observation), and
#      reads back the HMC-decoded image (uk_get_result) — all in-
#      process (positive path).
#   3. Exercises the manifest authorization gate with `modhost`
#      (UK-4001 negative test): granting vs. revoking `uk_get_result`,
#      which the Bayesian Update demo needs to drain the result.
#
# This module ships inside the unfer repo (unfer/bayes_update_module);
# it expects the australVM compiler checkout as a sibling of unfer
# ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # bayes_update_module lives inside the unfer repo
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
echo " 4. POSITIVE: run bayes_update_module through the CPS-JIT"
echo "    The module builds a qfm_tomography JSON ModelSpec, JIT-creates"
echo "    a real kernel model in-process (uk_model_create), runs a single"
echo "    Bayesian update (uk_bayesian_update with one observation at"
echo "    training point 0), and reads back the HMC-decoded image"
echo "    (uk_get_result). run() returns 1 only if the model was"
echo "    actually created (handle > 0) and the result was"
echo "    successfully drained; a parse/create/fill failure returns"
echo "    the negative UK code."
echo "============================================================"
JIT_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/BayesUpdateModule.aui,$HERE/src/BayesUpdateModule.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
echo "$JIT_OUT"
if echo "$JIT_OUT" | grep -q "CPS JIT: Execution result: 1"; then
  echo "PASS: bayes_update_module JIT-created a qfm_tomography model, ran the Bayesian update, and drained the result."
else
  echo "FAIL: expected the JIT to create+update+read the Bayesian update model (Execution result: 1)." >&2
  exit 1
fi

echo "============================================================"
echo " 5. AUTHORIZATION: manifest grant vs. revocation (UK-4001)"
echo "============================================================"
# 5a. With the full manifest, uk_get_result is granted -> ALLOW.
if "$MODHOST" authorize "$HERE/module.toml" bayes_update_module uk_get_result; then
  echo "PASS: granted manifest authorizes uk_get_result (BayesianUpdateResult read)."
else
  echo "FAIL: full manifest should authorize uk_get_result." >&2
  exit 1
fi

# 5b. Strip uk_get_result from the grants -> DENY (UK-4001). Without it,
#     the Bayesian update module cannot drain the HMC-decoded image.
STRIPPED="$(mktemp)"
trap 'rm -f "$STRIPPED"' EXIT
grep -v '"uk_get_result"' "$HERE/module.toml" > "$STRIPPED"
if "$MODHOST" authorize "$STRIPPED" bayes_update_module uk_get_result; then
  echo "FAIL: stripped manifest must DENY uk_get_result." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_get_result with UK-4001."
fi

# 5c. Also test stripping uk_bayesian_update -> DENY (UK-4001).
STRIPPED2="$(mktemp)"
trap 'rm -f "$STRIPPED" "$STRIPPED2"' EXIT
grep -v '"uk_bayesian_update"' "$HERE/module.toml" > "$STRIPPED2"
if "$MODHOST" authorize "$STRIPPED2" bayes_update_module uk_bayesian_update; then
  echo "FAIL: stripped manifest must DENY uk_bayesian_update." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_bayesian_update with UK-4001."
fi

echo "============================================================"
echo " BAYES UPDATE DEMO COMPLETE: qfm_tomography model created, Bayesian"
echo "                              update run, and result drained in-process;"
echo "                              authorization gate enforced for both"
echo "                              uk_get_result and uk_bayesian_update."
echo "============================================================"
