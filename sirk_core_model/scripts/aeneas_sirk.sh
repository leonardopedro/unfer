#!/usr/bin/env bash
# Regenerate the Aeneas Lean 4 model of the SIRK numeric core (T9).
#
# Pipeline (see CONSOLIDATED_PLAN.md §13.7 T9 and MASS_GAP_CERTIFIED.md §5.3):
#   1. Charon extracts the pure Rust model (Aeneas-supported subset) to .llbc.
#   2. Aeneas translates the .llbc to Lean 4 (backend = lean).
#
# Toolchain: the Aeneas nightly release bundle carries matching Charon +
# Aeneas binaries and the Lean backend library (Lean 4.31.0).
#
#   AENEAS_ROOT   path to an extracted aeneas release bundle (default:
#                 $HOME/Projects/.toolchain/aeneas-bin)
#   RUSTC_NIGHTLY the pinned Charon nightly toolchain (default:
#                 nightly-2026-08-18, from the bundle's rust-toolchain)
#
# Outputs:
#   aeneas/sirk_core_model.llbc   the Charon intermediate representation
#   aeneas/SirkCoreModel.lean     the generated Lean 4 model
#
# Scope: this script verifies the pure SIRK–Hashimoto numerical core only.
# It does not define a lattice Hamiltonian or establish the QYM mass gap.
# The physical input is the gauge-fixed nested-Fock one-particle Hamiltonian;
# its outer creation-left/annihilation-right enclosure is handled by the
# surrounding formalization.
#
# Honesty boundary: the *algorithmic* content (the forward-sequence fold, the
# Gram-assembly and whitening loops, the index bookkeeping, the shapes of the
# projection identity and residual formula) is translated verbatim; the `f64`
# arithmetic leaves stay opaque (`sorry`) — the rounding layer, enclosed by
# T1–T5 of MASS_GAP_CERTIFIED.md. Aeneas verifies the algorithm; the rounding
# is enclosed by the finite-precision theorems. The algebraic identities
# (projection identity, Gram Hermitian symmetry, T* Ĝ T = I, e_m = τ_m c_{m-1})
# are proved against this model by the Lean 4 specialist in
# BookProof/ChapterSirkFinitePrecision.lean.

set -euo pipefail

AENEAS_ROOT="${AENEAS_ROOT:-$HOME/Projects/.toolchain/aeneas-bin}"
RUSTC_NIGHTLY="${RUSTC_NIGHTLY:-nightly-2026-08-18}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

if [ ! -x "$AENEAS_ROOT/aeneas" ]; then
  echo "error: Aeneas binary not found at $AENEAS_ROOT/aeneas" >&2
  echo "  download the release bundle from" >&2
  echo "  https://github.com/AeneasVerif/aeneas/releases (aeneas-linux-x86_64.tar.gz)" >&2
  exit 1
fi

echo "== charon: src/lib.rs -> aeneas/sirk_core_model.llbc =="
RUSTC_BOOTSTRAP=1 "$AENEAS_ROOT/charon" rustc --preset=aeneas \
  -- src/lib.rs --edition 2021 --crate-type lib

echo "== aeneas: aeneas/sirk_core_model.llbc -> aeneas/SirkCoreModel.lean =="
"$AENEAS_ROOT/aeneas" -backend lean -namespace sirk_core_model \
  -o aeneas lib.llbc 2>/dev/null || \
"$AENEAS_ROOT/aeneas" -backend lean -namespace sirk_core_model lib.llbc

# Aeneas emits `Lib.lean` (module name from the crate); normalise the name.
if [ -f Lib.lean ]; then
  mv Lib.lean aeneas/SirkCoreModel.lean
fi
mv lib.llbc aeneas/sirk_core_model.llbc

echo "== regenerated =="
ls -la aeneas/sirk_core_model.llbc aeneas/SirkCoreModel.lean
