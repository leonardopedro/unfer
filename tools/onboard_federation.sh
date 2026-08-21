#!/usr/bin/env bash
# H14 federation onboarding dry-run.
#
# Prints the steps to federate a new org <slug> WITHOUT writing to the core.
# Org material lands under deploy/layers/<org>/ only; the repo root (core)
# stays byte-identical (S23/S24 golden gate is the guard). With --dry-run
# (the default and only mode), nothing is written anywhere.
#
# Usage: tools/onboard_federation.sh --org <slug> [--dry-run]

set -u

ORG=""
DRY_RUN=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --org) ORG="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --write) DRY_RUN=0; shift ;;
        *) echo "usage: $0 --org <slug> [--dry-run]" >&2; exit 2 ;;
    esac
done

if [ -z "$ORG" ]; then
    echo "error: --org <slug> is required" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
layer="$repo_root/deploy/layers/$ORG"

echo "onboard_federation --org $ORG"
echo "  dry-run: true (nothing is written to the core)"
echo

cat <<EOF
Steps:
  [1] Create the org layer directory (outside byte-identical core):
        mkdir -p $layer
  [2] Drop org grants + operator config into the layer (S22 admin seam):
        $layer/grants.toml   — module [grants] allowlist
        $layer/operator.json — operator config (posture, gatekeeper)
  [3] Set the deployment security posture (H9, operator console):
        uk_posture_set '{"posture":"auto"}'
  [4] Provision the edge (unfer_edge) + Taler exchange (unfer_taler, Plan R):
        edge   — Pingora front for the agent protocol
        taler  — reserves, denominations, two-phase wire gateway
  [5] Load the org's modules through modhost (australVM):
        module.toml [grants] / archetype per module; approved via H8 resolver.
  [6] Final live verification:
        - unfer_agent '{"id":"1","op":"version","params":{}}' -> ok:true
        - verify-invariants green in each repo
        - release golden gate green (core byte-identical)
EOF

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "dry-run complete — no files written."
else
    echo
    echo "error: --write is not supported; org material must stay out of core." >&2
    exit 1
fi

# Verify the core stayed clean (nothing in the repo root was written by us).
if [ -d "$repo_root/.git" ]; then
    untracked="$(cd "$repo_root" && git status --porcelain | grep -v 'deploy/layers/' | grep '^??' || true)"
    if [ -n "$untracked" ]; then
        echo "core has unexpected untracked files (outside deploy/layers/):" >&2
        echo "$untracked" >&2
        exit 1
    fi
fi
echo "core git status clean."
exit 0