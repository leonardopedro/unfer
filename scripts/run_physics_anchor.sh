#!/usr/bin/env bash
# Fast physics validation anchor — runs all non-ignored physics suites
# in release mode with timing.  Designed for CI and quick regression checks.
#
#   scripts/run_physics_anchor.sh
#
# Covers QED, QG, QYM, NG, NS, mass-gap, SU(2), and quantum-foam suites.

set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root/fock_sirk"

echo "=== Physics anchor run — $(date) ==="
echo

# All physics validation test suites (non-ignored, non-heavy).
SUITES=(
    qed_validation
    qed_extended_validation
    qed_further_validation
    qed_nuclear_astro
    qg_validation
    qg_cosmology_validation
    qg_general_relativity_validation
    qym_mass_gap
    qym_lattice_validation
    ng_newtonian_validation
    ns_numerical_validation
    ns_boundary_layer_validation
    ns_further_validation
    su2_gauge_dynamics
    quantum_foam_ng_overlap
    sirk_hamiltonian_drive
    qcd_validation
    qcd_mass_gap_certified
)

FAILED=0
TOTAL=0
PASS=0

for suite in "${SUITES[@]}"; do
    TOTAL=$((TOTAL + 1))
    start=$(date +%s%N)
    if cargo test --release --test "$suite" -- --test-threads=1 2>&1 | tail -3; then
        end=$(date +%s%N)
        elapsed=$(( (end - start) / 1000000 ))
        echo "  [pass] $suite (${elapsed}ms)"
        PASS=$((PASS + 1))
    else
        end=$(date +%s%N)
        elapsed=$(( (end - start) / 1000000 ))
        echo "  [FAIL] $suite (${elapsed}ms)"
        FAILED=$((FAILED + 1))
    fi
    echo
done

echo "=== Summary: $PASS/$TOTAL passed, $FAILED failed ==="
exit "$FAILED"
