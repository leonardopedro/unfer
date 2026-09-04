#!/usr/bin/env bash
# Run a kernel module's full e2e locally, mirroring the CI
# `module_builder` job (austral compile + CPS JIT + UK-4001 auth gate)
# on a dev machine. This is the recipe that turns a ~20-minute blind CI
# loop into a local run of a few minutes (seconds once built).
#
# What it needs, and how each piece is resolved:
#   - Rust side (unfer_ffi, the cranelift bridge, modhost): the host
#     rustup toolchain; the repo's rust-toolchain.toml pins 1.97.1, so a
#     bare `cargo` invocation already uses the CI compiler.
#   - OCaml side (the austral compiler): `dune` + OCaml libs. module_builder
#     uses a bare `dune` if present, `opam exec -- dune` if opam is, and
#     otherwise drops into the australVM nix flake (`nix develop`), which
#     provides both. On this machine the nix path is the one that fires.
#   - First run builds everything (cargo release + dune); later runs are
#     incremental via cargo/dune caches.
#
# Usage:
#   tools/e2e_local.sh [module_dir]     # default: qfm_tomo_module
#   MODULE_BUILDER_SKIP_BUILD=1 tools/e2e_local.sh demo_module   # compile-only iteration
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"

MODULE="${1:-qfm_tomo_module}"
MODULE_DIR="$UNFER_DIR/$MODULE"

[ -f "$MODULE_DIR/module.toml" ] || {
    echo "ERROR: no module at $MODULE_DIR (module.toml missing)" >&2
    echo "  Available: demo_module qfm_module qfm_tomo_module bayes_update_module" >&2
    echo "             iterated_bayes_module zenodo_store_module durable_status_module" >&2
    exit 1
}

exec "$HERE/module_builder" run "$MODULE_DIR"
