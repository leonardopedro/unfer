#!/usr/bin/env bash
# End-to-end demo for the Quantum Flow Matching (QFM) kernel module (QFM.tex):
#   1. Builds the unfer kernel FFI (uk_* symbols) and the safestos cranelift bridge.
#   2. Runs QfmModule through the CPS-JIT: it builds a `qfm_mehler` ModelSpec
#      (Mehler vacuum prior + decoupled α_j weights), JIT-creates the model
#      (uk_model_create), runs the single-step O(m²) inference (uk_evolve), and
#      reads a generated data-channel probability (uk_event_probability) — all
#      in-process (positive path).
#   3. Exercises the manifest authorization gate with `modhost` (UK-4001 negative
#      test): granting vs. revoking `uk_evolve`, which QFM inference depends on.
#
# This module ships inside the unfer repo (unfer/qfm_module); it expects the
# australVM compiler checkout as a sibling of unfer ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # qfm_module lives inside the unfer repo
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
echo " 4. POSITIVE: run qfm_module through the CPS-JIT"
echo "    The module builds a qfm_mehler JSON ModelSpec, JIT-creates a real"
echo "    kernel model in-process (uk_model_create), runs the single-step QFM"
echo "    inference (uk_evolve), and reads a generated data-channel probability"
echo "    (uk_event_probability). run() returns 1 only if the model was actually"
echo "    created (handle > 0); a parse/create failure returns a negative UK code."
echo "============================================================"
JIT_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/QfmModule.aui,$HERE/src/QfmModule.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
echo "$JIT_OUT"
if echo "$JIT_OUT" | grep -q "CPS JIT: Execution result: 1"; then
  echo "PASS: qfm_module JIT-created a qfm_mehler model, evolved it, and read a probability."
else
  echo "FAIL: expected the JIT to create+evolve the QFM model (Execution result: 1)." >&2
  exit 1
fi

echo "============================================================"
echo " 5. AUTHORIZATION: manifest grant vs. revocation (UK-4001)"
echo "============================================================"
# 5a. With the full manifest, uk_evolve is granted -> ALLOW.
if "$MODHOST" authorize "$HERE/module.toml" qfm_module uk_evolve; then
  echo "PASS: granted manifest authorizes uk_evolve (QFM inference)."
else
  echo "FAIL: full manifest should authorize uk_evolve." >&2
  exit 1
fi

# 5b. Strip uk_evolve from the grants -> DENY (UK-4001). Without it, the QFM
#     module cannot run its single-step inference.
STRIPPED="$(mktemp)"
trap 'rm -f "$STRIPPED"' EXIT
grep -v '"uk_evolve"' "$HERE/module.toml" > "$STRIPPED"
if "$MODHOST" authorize "$STRIPPED" qfm_module uk_evolve; then
  echo "FAIL: stripped manifest must DENY uk_evolve." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_evolve with UK-4001."
fi

echo "============================================================"
echo " QFM DEMO COMPLETE: module-built qfm_mehler model created, evolved, and"
echo "                    queried in-process; authorization gate enforced."
echo "============================================================"
