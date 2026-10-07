#!/usr/bin/env bash
# N2 / D.4 — the `module-reconstruction` benchmark gate.
#
# Rebuild module behaviour from a **pinned** `.cell` blueprint and hold it to
# the project's own golden gates, in this order:
#
#   1. unpack `tools/module_reconstruction/demo_module.cell` into a scratch dir
#      (the blueprint is the input of record — the gate never repacks it);
#   2. `cell_tool verify` the reconstruction against the golden manifest
#      (sha256 per file; extra or missing files fail, not just changed ones);
#   3. run the project's own module gate on the reconstructed directory:
#      `tools/module_builder run` (Austral type-check, CPS-JIT positive run,
#      UK-4001 revocation negative test).
#
# Any step failing exits non-zero, which is what `bench_runner` scores. The
# scratch directory is always removed, so a failed run leaves no half-built
# module behind for the next trial to inherit.
#
# ## Why the pack step is not here
#
# Repacking the blueprint at gate time would make the input equal to whatever
# the source tree currently is, so a pack/unpack corruption could never be
# caught against a fixed reference. The `.cell` and the manifest are committed
# together and regenerated only by a deliberate act:
#
#   cargo run --offline -p unfer_protocol --bin cell_tool -- \
#       pack demo_module tools/module_reconstruction/demo_module.cell
#   cargo run --offline -p unfer_protocol --bin cell_tool -- \
#       manifest demo_module > tools/module_reconstruction/demo_module.manifest
#
# ## UNFER_BENCH_WORKERS
#
# The harness exports it for every gate. Reconstructing one module blueprint
# has no parallel work to spend it on, so this gate does not consume it — the
# benchmark documentation says the same, and a task that merely *reads* the
# variable would create the appearance of a worker-sensitive measurement it
# does not have.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNFER_DIR="$(cd "$HERE/.." && pwd)"
BUNDLE="$HERE/module_reconstruction"

# Same toolchain convention as scripts/regen_mass_gap.sh: unfer is pinned to
# 1.97.1 via the environment (no rust-toolchain.toml in this repo).
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.97.1}"

for f in "$BUNDLE/demo_module.cell" "$BUNDLE/demo_module.manifest" \
         "$HERE/module_builder"; do
    [ -e "$f" ] || { echo "module_reconstruct: missing bundle file: $f" >&2; exit 1; }
done

OUT="$(mktemp -d "${TMPDIR:-/tmp}/module_reconstruct.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT
RECON="$OUT/demo_module"

cd "$UNFER_DIR"

echo ">> unpacking the pinned .cell blueprint"
cargo run --offline -q -p unfer_protocol --bin cell_tool -- \
    unpack "$BUNDLE/demo_module.cell" "$RECON"

echo ">> checking the reconstruction against the golden manifest"
cargo run --offline -q -p unfer_protocol --bin cell_tool -- \
    verify "$RECON" "$BUNDLE/demo_module.manifest"

echo ">> running the project's own module gate on the reconstruction"
bash "$HERE/module_builder" run "$RECON"

echo "module_reconstruct: reconstruction passed the manifest and module_builder"
