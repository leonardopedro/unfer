#!/usr/bin/env bash
# Type-check the demo module against the UnferKernel bindings.
# This module ships inside the unfer repo (unfer/demo_module); it expects the
# australVM compiler checkout as a sibling of unfer ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # demo_module lives inside the unfer repo
ROOT="$(cd "$UNFER_DIR/.." && pwd)"      # parent holding the sibling australVM checkout
AUSTRAL_DIR="$ROOT/australVM"

[ -d "$AUSTRAL_DIR" ] || { echo "ERROR: expected sibling australVM at $AUSTRAL_DIR" >&2; exit 1; }
[ -d "$UNFER_DIR" ]   || { echo "ERROR: expected unfer repo at $UNFER_DIR" >&2; exit 1; }

AUSTRAL="$AUSTRAL_DIR/_build/default/bin/austral.exe"
LIBDIR="$AUSTRAL_DIR/safestos/cranelift/target/release"

echo ">> Building the Austral compiler (dune build lib/ bin/)"
( cd "$AUSTRAL_DIR" && dune build lib/ bin/ )

echo ">> Type-checking demo_module against UnferKernel bindings"
LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/DemoModule.aui,$HERE/src/DemoModule.aum" \
  --target-type=tc

echo "OK: demo_module type-checks."
