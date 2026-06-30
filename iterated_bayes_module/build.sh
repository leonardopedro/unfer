#!/usr/bin/env bash
# Type-check the Bayesian Update on the TSR-evolved prior module
# against the UnferKernel bindings. This module ships inside the
# unfer repo (unfer/bayes_update_module); it expects the australVM
# compiler checkout as a sibling of unfer ($ROOT/{unfer,australVM}).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"      # bayes_update_module lives inside the unfer repo
ROOT="$(cd "$UNFER_DIR/.." && pwd)"      # parent holding the sibling australVM checkout
AUSTRAL_DIR="$ROOT/australVM"

[ -d "$AUSTRAL_DIR" ] || { echo "ERROR: expected sibling australVM at $AUSTRAL_DIR" >&2; exit 1; }
[ -d "$UNFER_DIR" ]   || { echo "ERROR: expected unfer repo at $UNFER_DIR" >&2; exit 1; }

AUSTRAL="$AUSTRAL_DIR/_build/default/bin/austral.exe"
LIBDIR="$AUSTRAL_DIR/safestos/cranelift/target/release"

echo ">> Building the Austral compiler (dune build lib/ bin/)"
( cd "$AUSTRAL_DIR" && dune build lib/ bin/ )

echo ">> Type-checking bayes_update_module against UnferKernel bindings"
LD_LIBRARY_PATH="$LIBDIR" "$AUSTRAL" compile \
  "$AUSTRAL_DIR/examples/kernel/UnferKernel.aui,$AUSTRAL_DIR/examples/kernel/UnferKernel.aum" \
  "$HERE/src/BayesUpdateModule.aui,$HERE/src/BayesUpdateModule.aum" \
  --target-type=tc

echo "OK: bayes_update_module type-checks."
