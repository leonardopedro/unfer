# aarch64 devShell for the unfer workspace — the ARM counterpart of ../flake.nix.
#
# WHY THIS EXISTS (separate flake, single channel).
# ------------------------------------------------
# The root `flake.nix` hardcodes `system = "x86_64-linux"` in every `import`,
# and its default shell is built around the CUDA 12.6 toolkit
# (`cudaPackages_12_6.cudatoolkit`) plus a 23.05/unstable channel mix. On an
# aarch64 host it does not even evaluate ("does not provide attribute
# devShells.aarch64-linux..."), and CUDA is irrelevant there. Rather than thread
# a system conditional through the CUDA shell, this is a small, self-contained
# flake scoped to what the ARM box actually needs: **Cadabra2 (S30) so the
# engine-dependent `docs/*.cdb` tests run instead of skipping**, plus the native
# build deps the workspace needs to compile at all.
#
# Everything comes from ONE channel — `nixos-unstable`:
#   * cadabra2 is only packaged for aarch64 on unstable (nixos-24.11 / 25.05 do
#     not ship it for aarch64 — verified by `nix build --dry-run`);
#   * using a single channel keeps the C++ ABI consistent, which removes the
#     libstdc++-version dance the root flake needs (its cadabra2 comes from
#     unstable but its stdenv from 23.05, so it must prepend gcc-15's
#     libstdc++ to LD_LIBRARY_PATH). Here cadabra2 and every dependency share
#     unstable's stdenv, so no override is required.
#
# "USE CADABRA2 ALREADY COMPILED."
# -------------------------------
# On aarch64 the cadabra2 *dependencies* (266 paths, ~444 MiB) are all served
# prebuilt from the nixos.org binary cache; only the single top-level
# `cadabra2` derivation compiles locally (one C++ build). That is the
# best-available outcome on ARM — the alternative channels have no aarch64
# cadabra2 at all. Re-check substitutability with:
#     nix build --dry-run 'github:NixOS/nixpkgs/nixos-unstable#cadabra2'
#
# USAGE:
#     nix develop ./nix/aarch64          # from the unfer/ repo root
#     # then, with the Rust toolchain active:
#     cargo test -p prob_kernel --lib symbolic   # the .cdb engine tests run now
#
# The engine seam (prob_kernel::symbolic) discovers the binary via the
# CADABRA_CLI env var, falling back to a PATH search for `cadabra2-cli`; the
# shellHook below sets CADABRA_CLI explicitly so discovery cannot silently
# fall through to a skip.

{
  description = "unfer aarch64 devShell: Cadabra2 (S30 symbolic CAS subprocess) + native build deps, single-channel (nixos-unstable)";

  # Single channel: the only nixpkgs branch that packages cadabra2 for
  # aarch64, and the stdenv every dependency below is built against.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # Nothing here is unfree (no CUDA on this host), but keep allowUnfree
        # so a future addition cannot break the shell on a licence technicality.
        config.allowUnfree = true;
      };

      # WHY THE OVERRIDE: the stock `pkgs.cadabra2` derivation FAILS TO BUILD
      # on aarch64 — a real nixpkgs packaging bug (also why cadabra2 is absent
      # from the aarch64 binary cache). Its CMake `try_compile` invokes
      # `make`, and GNU make shells out to a hardcoded `/bin/sh`, which the
      # Nix build sandbox does not provide, so configure aborts with
      #   make: /bin/sh: No such file or directory
      # Setting SHELL in the env does NOT help: the try_compile sub-make does
      # not inherit it. The fix is to switch the CMake generator to NINJA,
      # which invokes the compiler by direct execve and never touches /bin/sh.
      # Verified on this host: 364 objects compile, and the resulting
      # cadabra2-cli (2.5.15) runs the unfer symbolic suite 19/19 with 0 skips.
      cadabra2Fixed = pkgs.cadabra2.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.ninja ];
        # Force the Ninja generator so CMake's try_compile bypasses make
        # (and its hardcoded /bin/sh). Everything else is stock nixpkgs.
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-G" "Ninja" ];
      });
    in
    {
      # An installable package: the same cadabra2Fixed the devShell uses, so
      # `nix build`/`nix profile install` puts cadabra2-cli on the profile
      # without entering a shell. Keep the surface small: cadabra2 is the only
      # packaging target (the ARM host is a build/test box).
      packages.${system} = {
        default = cadabra2Fixed;
        cadabra2 = cadabra2Fixed;
      };

      # The devShell: cadabra2 plus the native build deps the workspace needs.
      devShells.${system}.default = pkgs.mkShell {
        name = "unfer-aarch64-cadabra";

        packages = with pkgs; [
          # S30: Cadabra2 symbolic CAS, invoked as an external subprocess by
          # prob_kernel::symbolic (GPL-3.0 stays out of the Rust link graph —
          # same licence boundary as the root flake). Built via the
          # Ninja-generator override (see cadabra2Fixed above) because the
          # stock derivation does not build on aarch64.
          cadabra2Fixed

          # Native build deps the workspace needs to compile on ARM. Mirrors
          # the root flake's rationale: these surface as opaque `-sys`
          # build-script panics if missing. All from unstable (single channel),
          # so no cross-cmake-version issue.
          zlib-ng   # unfer_edge's Pingora -> libz-ng-sys
          cmake     # libz-ng-sys falls back to a cmake source build
          pkgconf   # 23.05 pkg-config cannot see the above; unstable's is fine
          pkg-config
          python3   # stdio/HTTP test fixtures + Cadabra2's own interpreter
          gcc
          gnumake

          # Rust toolchain via rustup, matching the root flake. Deliberately no
          # RUSTUP_TOOLCHAIN: rustup must resolve rust-toolchain.toml (or the
          # globally-installed default) from the cwd so the shell and CI agree.
          rustup
        ];

        shellHook = ''
          # Point the engine seam straight at the store path so discovery can't
          # silently fall back to a skip. `cadabra2-cli` is the non-GUI entry
          # point; `cadabra2` is the notebook/GUI binary.
          export CADABRA_CLI="${cadabra2Fixed}/bin/cadabra2-cli"

          # Defensive, and ABI-consistent here: put unstable's libstdc++ first
          # so the Cadabra2 subprocess always loads a libstdc++ at least as new
          # as the one it was built against. On this single-channel shell that
          # is the same stdenv it already links, so this is a no-op guard rather
          # than the load-bearing override the root flake needs.
          export LD_LIBRARY_PATH="${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH"

          echo "[unfer-aarch64] cadabra2: $CADABRA_CLI"
          if [ -x "$CADABRA_CLI" ]; then
            echo "[unfer-aarch64] $($CADABRA_CLI --version 2>/dev/null | head -1 || echo 'cadabra2-cli present')"
            echo "[unfer-aarch64] engine-dependent .cdb tests will RUN (not skip)."
          else
            echo "[unfer-aarch64] WARNING: cadabra2-cli not found at $CADABRA_CLI"
          fi
        '';
      };
    };
}
