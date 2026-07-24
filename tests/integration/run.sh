#!/usr/bin/env bash
# Cross-repo integration smoke test for the unfer system.
#
# Verifies the contract between unfer, australVM, and velysterm:
#   1. unfer_ffi cdylib builds and exports exactly the expected symbols.
#   2. tools/module_builder runs demo_module (positive + UK-4001 negative).
#   3. unfer_agent NDJSON pipeline (create → evolve → probability → close).
#
# Usage:
#   tests/integration/run.sh              # run all available checks
#   tests/integration/run.sh --symbols    # symbol check only
#   tests/integration/run.sh --module     # module_builder only
#   tests/integration/run.sh --agent      # unfer_agent only
#
# Exit codes: 0 = all checks passed, 1 = failure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/../.." && pwd)"
ROOT="$(cd "$UNFER_DIR/.." && pwd)"
AUSTRAL_DIR="$ROOT/australVM"
VELYSTERM_DIR="$ROOT/velysterm"

PASS=0
FAIL=0
SKIP=0

ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP: $1"; SKIP=$((SKIP + 1)); }

check_symbols() {
    echo "=== Symbol check ==="
    local expected="$UNFER_DIR/unfer_ffi/EXPECTED_SYMBOLS.txt"
    if [ ! -f "$expected" ]; then
        fail "EXPECTED_SYMBOLS.txt not found"
        return
    fi

    cargo build -p unfer_ffi --release 2>/dev/null
    local cdylib
    cdylib=$(find "$UNFER_DIR/target/release" -name 'libunfer_ffi.so' -o -name 'libunfer_ffi.dylib' 2>/dev/null | head -1)
    if [ -z "$cdylib" ]; then
        fail "cdylib not found after build"
        return
    fi

    local actual
    actual=$(nm -D --defined-only "$cdylib" 2>/dev/null | awk '{print $3}' | grep '^uk_' | sort)
    local expected_sorted
    expected_sorted=$(sort "$expected")

    if [ "$actual" = "$expected_sorted" ]; then
        ok "cdylib exports match EXPECTED_SYMBOLS.txt ($(wc -l < "$expected") symbols)"
    else
        fail "symbol mismatch"
        echo "  Expected:"
        echo "$expected_sorted" | sed 's/^/    /'
        echo "  Actual:"
        echo "$actual" | sed 's/^/    /'
    fi
}

check_module() {
    echo "=== Module builder (demo_module) ==="
    if [ ! -d "$AUSTRAL_DIR" ]; then
        skip "australVM not found at $AUSTRAL_DIR"
        return
    fi
    if [ ! -x "$UNFER_DIR/tools/module_builder" ]; then
        skip "tools/module_builder not executable"
        return
    fi

    if "$UNFER_DIR/tools/module_builder" run "$UNFER_DIR/demo_module" 2>&1; then
        ok "demo_module positive + negative tests"
    else
        fail "demo_module module_builder run failed"
    fi
}

check_agent() {
    echo "=== unfer_agent NDJSON pipeline ==="
    local agent_bin="$VELYSTERM_DIR/target/release/unfer_agent"
    if [ ! -d "$VELYSTERM_DIR" ]; then
        skip "velysterm not found at $VELYSTERM_DIR"
        return
    fi
    if [ ! -f "$agent_bin" ]; then
        skip "unfer_agent binary not built (run: cargo build --release -p kernel_client --bin unfer_agent)"
        return
    fi

    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN

    cat > "$tmpdir/session.ndjson" <<'NDJSON'
{"id":"1","op":"create_model","params":{"kind":"builtin","name":"harmonic_chain","params":{"n_modes":2}}}
{"id":"2","op":"evolve","params":{"model_id":1,"time":0.1}}
{"id":"3","op":"probability","params":{"model_id":1}}
{"id":"4","op":"close_model","params":{"model_id":1}}
NDJSON

    local output
    if output=$("$agent_bin" < "$tmpdir/session.ndjson" 2>"$tmpdir/stderr.log"); then
        if echo "$output" | grep -q '"ok":true'; then
            ok "NDJSON session (create → evolve → probability → close)"
        else
            fail "NDJSON session returned non-ok responses"
            echo "$output" | head -5 | sed 's/^/    /'
        fi
    else
        fail "unfer_agent exited non-zero"
        head -5 "$tmpdir/stderr.log" | sed 's/^/    /'
    fi
}

MODE="${1:-all}"

case "$MODE" in
    --symbols) check_symbols ;;
    --module)  check_module ;;
    --agent)   check_agent ;;
    all)
        check_symbols
        check_module
        check_agent
        ;;
    *)
        echo "Usage: $0 [--symbols|--module|--agent|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
