# shellcheck shell=bash
#
# workspace-root.sh -- shared workspace discovery for the hygiene gates.
#
# Sourced, never executed. `check-book-sync`, `sync-book`, `check-status` and
# `tree-size` all need to find the sibling checkouts, and each used to carry its
# own copy of a `find_root` that walked up until it saw >= 3 git checkouts.
#
# That threshold holds on a developer's machine -- this workspace holds 14 -- and
# does NOT in CI, where a workflow checks out the repo under test plus the one or
# two siblings it actually needs. So in CI every gate misbehaved: `check-book-sync`
# resolved an empty root and compared `/book.tex` against `/timepiece/book.tex`,
# failing for the wrong reason, while `tree-size` and `check-status` found nothing
# to check and exited 0. A gate that is green because it looked at the wrong
# directory is worse than no gate: it is read as evidence.
#
# One implementation, no magic threshold. Resolution order:
#
#   1. $WORKSPACE_ROOT                      -- explicit, used by CI
#   2. nearest ancestor holding unfer + timepiece   -- the pair this workspace is
#      built from (the kernel, and the corpus it hosts)
#   3. nearest ancestor holding >= 2 checkouts
#   4. the checkout this script lives in       -- single-checkout fallback
#
# Scripts must live in a `scripts/` directory. That is true both at the workspace
# root and in the per-repo mirrors under `unfer/scripts/`.

ws_script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

# The checkout this script lives in, when it lives in one. Empty otherwise (the
# workspace-root copy of the gates is not inside any checkout).
ws_own_repo() {
  local d
  d="$(ws_script_dir)"
  if [ -d "$d/../.git" ]; then
    cd "$d/.." && pwd
  fi
}

ws_count_checkouts() {
  local n=0 c
  for c in "$1"/*/; do
    [ -d "${c}.git" ] && n=$((n + 1))
  done
  printf '%s' "$n"
}

# The directory the sibling checkouts hang off.
ws_root() {
  local d

  if [ -n "${WORKSPACE_ROOT:-}" ] && [ -d "$WORKSPACE_ROOT" ]; then
    printf '%s' "$WORKSPACE_ROOT"
    return 0
  fi

  d="$(ws_script_dir)"
  while :; do
    if [ -d "$d/unfer/.git" ] && [ -d "$d/timepiece/.git" ]; then
      printf '%s' "$d"
      return 0
    fi
    if [ "$(ws_count_checkouts "$d")" -ge 2 ]; then
      printf '%s' "$d"
      return 0
    fi
    [ "$d" = / ] && break
    d="$(dirname "$d")"
  done

  # No siblings anywhere above us. The checkout we live in is the best guess --
  # callers must still cope with the siblings being absent.
  ws_own_repo
}

# Resolve a checkout by name, tolerating the single-checkout fallback where the
# root *is* the repo being asked for rather than its parent.
ws_checkout() {
  local root own base
  root="$(ws_root)"
  if [ -d "$root/$1" ]; then
    printf '%s' "$root/$1"
    return 0
  fi
  own="$(ws_own_repo)"
  if [ -n "$own" ]; then
    base="$(basename "$own")"
    if [ "$base" = "$1" ]; then
      printf '%s' "$own"
      return 0
    fi
  fi
  return 1
}

# Every checkout visible from the root, plus our own if the root is not its
# parent -- so `tree-size` always inspects the repo it was invoked from, which is
# the one thing CI cares about. Prints nothing (not one empty line) when there is
# nothing to inspect, so callers can count what they got.
ws_all_checkouts() {
  local root d out=() own seen d2
  root="$(ws_root)"
  for d in "$root"/*/; do
    [ -d "${d}.git" ] && out+=("${d%/}")
  done
  own="$(ws_own_repo)"
  if [ -n "$own" ]; then
    seen=0
    for d2 in ${out[@]+"${out[@]}"}; do
      [ "$d2" = "$own" ] && seen=1
    done
    [ "$seen" -eq 0 ] && out+=("$own")
  fi
  [ "${#out[@]}" -eq 0 ] && return 0
  printf '%s\n' "${out[@]}"
}