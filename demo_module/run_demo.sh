#!/usr/bin/env bash
# End-to-end demo for the unfer modular kernel:
#   1. Builds the unfer kernel FFI (uk_* symbols) and the safestos cranelift bridge.
#   2. Runs the demo Austral module through the CPS-JIT, executing a live `uk_*`
#      kernel call in-process (positive path).
#   3. Exercises the manifest authorization gate with `modhost` (UK-4001 negative
#      test): granting vs. revoking `uk_evolve`.
#
# This module ships inside the unfer repo (unfer/demo_module); it expects the
# australVM compiler checkout as a sibling of unfer ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # demo_module lives inside the unfer repo
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
# Use the freshly built compiler (bin/austral.exe); the top-level
# _build/default/austral alias is not refreshed by `dune build lib/ bin/`.
AUSTRAL="$AUSTRAL_DIR/_build/default/bin/austral.exe"

echo "============================================================"
echo " 4. POSITIVE: run demo_module through the CPS-JIT"
echo "    The module builds a JSON ModelSpec string, JIT-creates a real kernel"
echo "    model in-process (uk_model_create), computes an event probability"
echo "    (uk_event_probability), and holds the handle in a linear Model that the"
echo "    type system forces it to free. run() returns 1 only if the model was"
echo "    actually created (handle > 0); a parse/create failure returns a negative"
echo "    UK code instead."
echo "============================================================"
JIT_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/DemoModule.aui,$HERE/src/DemoModule.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
echo "$JIT_OUT"
if echo "$JIT_OUT" | grep -q "CPS JIT: Execution result: 1"; then
  echo "PASS: module JIT-created a real model from a JSON spec and computed a probability."
else
  echo "FAIL: expected the JIT to create a model from JSON (Execution result: 1)." >&2
  exit 1
fi

echo "============================================================"
echo " 5. AUTHORIZATION: manifest grant vs. revocation (UK-4001)"
echo "============================================================"
# 5a. With the full manifest, uk_evolve is granted -> ALLOW.
if "$MODHOST" authorize "$HERE/module.toml" demo_module uk_evolve; then
  echo "PASS: granted manifest authorizes uk_evolve."
else
  echo "FAIL: full manifest should authorize uk_evolve." >&2
  exit 1
fi

# 5b. Strip uk_evolve from the grants -> DENY (UK-4001).
STRIPPED="$(mktemp)"
trap 'rm -f "$STRIPPED"' EXIT
grep -v '"uk_evolve"' "$HERE/module.toml" > "$STRIPPED"
if "$MODHOST" authorize "$STRIPPED" demo_module uk_evolve; then
  echo "FAIL: stripped manifest must DENY uk_evolve." >&2
  exit 1
else
  echo "PASS: revoking the grant denies uk_evolve with UK-4001."
fi

echo "============================================================"
echo " 6. P2.7 LINEARITY GATE: a leaked model handle must NOT compile"
echo "============================================================"
# The linear `Model` wrapper (UnferKernel) makes `uk_model_free` a type-enforced
# obligation: a module that wraps a handle but never frees it must be rejected by
# the Austral typechecker. LeakDemo does exactly that; we assert the compile fails
# with a Linearity Error. (This is a compile-time guarantee independent of the
# CPS-JIT backend.)
LEAK_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/LeakDemo.aui,$HERE/src/LeakDemo.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
if echo "$LEAK_OUT" | grep -q "Linearity Error"; then
  echo "PASS: leaking a linear Model is rejected at compile time (Linearity Error)."
else
  echo "FAIL: a leaked linear Model must fail to compile with a Linearity Error." >&2
  echo "$LEAK_OUT" >&2
  exit 1
fi

echo "============================================================"
echo " 7. CPS-JIT BACKEND GATE: multi-field record slot offsets"
echo "============================================================"
# Records are load-bearing for the kernel module surface (the linear Model is a
# record; freeModel destructures it). RecordCheck builds a 3-field record and
# returns the field at slot offset 16; a regression in slot-offset lowering would
# return offset 0 instead. Asserts Execution result: 30.
REC_OUT="$(LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$HERE/src/RecordCheck.aui,$HERE/src/RecordCheck.aum" \
  --use-cps-jit --target-type=tc 2>&1 || true)"
if echo "$REC_OUT" | grep -q "CPS JIT: Execution result: 30"; then
  echo "PASS: multi-field record slot offsets are correct (read offset-16 field = 30)."
else
  echo "FAIL: expected the JIT to read the offset-16 record field (Execution result: 30)." >&2
  echo "$REC_OUT" >&2
  exit 1
fi


echo "============================================================"
echo " 8. DATA SOURCE: ingestion loop via uk_observe"
echo "============================================================"
( cd "$HERE/data_source" && cargo run )
