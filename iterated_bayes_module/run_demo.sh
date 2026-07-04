#!/usr/bin/env bash
# End-to-end demo for the Iterated Quantum Bayesian Update + SIRK
# Evolution module (QFM.tex §7 + §8, P8 P11):
#   1. Builds the unfer kernel FFI (uk_* symbols) and the safestos
#      cranelift bridge.
#   2. Runs IteratedBayesModule through the CPS-JIT: it builds a
#      `qfm_tomography` ModelSpec (the 4-point tetrahedron training
#      set with krylov_dim=4 for the P6 G SIRK-whitened basis), JIT-
#      creates the model (uk_model_create), then runs a 3-iteration
#      loop that exercises the full QFM.tex §7 + §8 pipeline
#      end-to-end: condition (uk_bayesian_update), drain
#      (uk_get_result), evolve (uk_evolve), and re-wrap — all in-
#      process (positive path).
#   3. Exercises the manifest authorization gate with `modhost`
#      (UK-4001 negative test): granting vs. revoking `uk_get_result`,
#      which the iterated module needs to drain each result.
#
# This module ships inside the unfer repo (unfer/iterated_bayes_module);
# it expects the australVM compiler checkout as a sibling of unfer
# ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # iterated_bayes_module lives inside the unfer repo
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
echo " 4. POSITIVE: run iterated_bayes_module through the CPS-JIT"
echo "    The module builds a qfm_tomography JSON ModelSpec, JIT-creates"
echo "    a real kernel model in-process (uk_model_create), then runs"
echo "    3 iterations of (uk_bayesian_update + uk_get_result +"
echo "    uk_evolve + linear free/rewrap) — the full QFM.tex §7 + §8"
echo "    pipeline. run() returns 1 only if all 3 iterations succeed."
echo "============================================================"
JIT_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/IteratedBayesModule.aui,$HERE/src/IteratedBayesModule.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
echo "$JIT_OUT"
if echo "$JIT_OUT" | grep -q "CPS JIT: Execution result: 1"; then
  echo "PASS: iterated_bayes_module JIT-created a qfm_tomography model, ran 3 (bayes+evolve) iterations, and drained each result."
else
  echo "FAIL: expected the JIT to create+iterate+read the model 3 times (Execution result: 1)." >&2
  exit 1
fi

echo "============================================================"
echo " 5. AUTHORIZATION: manifest grant vs. revocation (UK-4001)"
echo "============================================================"
# 5a. With the full manifest, uk_get_result is granted -> ALLOW.
if "$MODHOST" authorize "$HERE/module.toml" iterated_bayes_module uk_get_result; then
  echo "PASS: granted manifest authorizes uk_get_result (BayesianUpdateResult drain)."
else
  echo "FAIL: full manifest should authorize uk_get_result." >&2
  exit 1
fi

# 5b. Strip uk_get_result from the grants -> DENY (UK-4001). Without it,
#     the iterated module cannot drain any of the 3 HMC-decoded images.
STRIPPED="$(mktemp)"
trap 'rm -f "$STRIPPED"' EXIT
grep -v '"uk_get_result"' "$HERE/module.toml" > "$STRIPPED"
if "$MODHOST" authorize "$STRIPPED" iterated_bayes_module uk_get_result; then
  echo "FAIL: stripped manifest must DENY uk_get_result." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_get_result with UK-4001."
fi

# 5c. Also test stripping uk_evolve -> DENY (UK-4001). The iterated
#     module needs uk_evolve to advance the time evolution between
#     Bayesian updates.
STRIPPED2="$(mktemp)"
trap 'rm -f "$STRIPPED" "$STRIPPED2"' EXIT
grep -v '"uk_evolve"' "$HERE/module.toml" > "$STRIPPED2"
if "$MODHOST" authorize "$STRIPPED2" iterated_bayes_module uk_evolve; then
  echo "FAIL: stripped manifest must DENY uk_evolve." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_evolve with UK-4001."
fi

echo "============================================================"
echo " ITERATED BAYES MODULE DEMO COMPLETE: qfm_tomography model"
echo "    created, 3 (bayes + evolve) iterations executed in-process,"
echo "    authorization gate enforced for both uk_get_result and"
echo "    uk_evolve."
echo "============================================================"
