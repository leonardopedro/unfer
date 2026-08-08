#!/usr/bin/env bash
# S7 smoke test for the sandboxed workerd sidecar path.
#
# The ECMAScript runtime is packaged by Nix (`.#unfer-workerd`: statically-linked
# workerd + workerd.capnp from pinned npm tarballs). This script verifies the
# packaged runtime actually drives the sidecar stack end to end, *inside* the OS
# sandbox (S3): it builds the cranelift sidecar harness with the `sandbox`
# feature and runs the sandbox-confinement + positive-lifecycle integration
# tests against the Nix-built workerd.
#
# Gate: `nix build .#unfer-workerd .#unfer-data .#unfer-ffi` is green (packaging)
#       and this script exits 0 (the sandboxed sidecar round-trips).
#
# Usage:
#   ./run_sandboxed_sidecar.sh                      # builds workerd via nix
#   UNFER_WORKERD=/nix/store/...-unfer-workerd/bin/workerd ./run_sandboxed_sidecar.sh
#   UNFER_AUSTRALVM=/path/to/australVM ./run_sandboxed_sidecar.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Resolve the packaged workerd, in order:
#    a. Bundled layout (`nix run .#sandboxed-sidecar-smoke` runs the store copy):
#       <script>/workerd/bin/workerd sits next to this script.
#    b. UNFER_WORKERD override.
#    c. Working-tree case: nix-build it from the parent unfer flake.
if [ -x "$SCRIPT_DIR/workerd/bin/workerd" ]; then
  WORKERD_BIN="$SCRIPT_DIR/workerd/bin/workerd"
elif [ -n "${UNFER_WORKERD:-}" ]; then
  WORKERD_BIN="$UNFER_WORKERD"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "[smoke] nix build workerd..."
  nix build "$REPO_ROOT#packages.x86_64-linux.unfer-workerd" -o "$TMP/workerd" >/dev/null
  WORKERD_BIN="$TMP/workerd/bin/workerd"
fi

# 2. Resolve the australiaVM cranelift harness (host side, builds the sidecar
#    supervisor + sandbox launcher). `nix run` inherits the caller's env, so the
#    operator sets UNFER_AUSTRALVM for the store layout; the working-tree layout
#    (sibling of unfer/) resolves automatically.
if [ -n "${UNFER_AUSTRALVM:-}" ]; then
  AUSTRALVM="$UNFER_AUSTRALVM"
else
  AUSTRALVM="$(cd "$SCRIPT_DIR/../.." && pwd)/australVM"
fi

if [ ! -d "$AUSTRALVM/safestos/cranelift" ]; then
  echo "error: australiaVM cranelift harness not found at $AUSTRALVM" >&2
  echo "set UNFER_AUSTRALVM to the sibling checkout" >&2
  exit 2
fi

if [ ! -x "$WORKERD_BIN" ]; then
  echo "error: workerd binary not executable: $WORKERD_BIN" >&2
  exit 2
fi
echo "[smoke] workerd: $WORKERD_BIN"
"$WORKERD_BIN" --version

export UNFER_WORKERD="$WORKERD_BIN"

# 2. Run the sandboxed-sidecar integration tests (S3 confinement + S1 lifecycle)
#    against the packaged runtime. `ecmascript_sidecar_os_sandbox_confines_child`
#    asserts the sidecar runs in its own user namespace with no_new_privs +
#    seccomp; `ecmascript_positive_model_lifecycle` asserts the kernel round-trip.
echo "[smoke] building cranelift harness (features: sandbox test-stubs)..."
cd "$AUSTRALVM/safestos/cranelift"
cargo test --features "sandbox test-stubs" --test ecmascript_module -- \
  ecmascript_sidecar_os_sandbox_confines_child \
  ecmascript_positive_model_lifecycle

echo "[smoke] PASS: packaged workerd drives the sandboxed sidecar."
