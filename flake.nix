# P9 P12: CUDA toolkit pinning for the unfer workspace (Nix shell).
#
# Pinned CUDA: 12.2 (the toolkit version that the unfer GPU
# tests were developed against; libcublas 12.2, libcudart 12.2,
# CUFFT 11.0, CUSOLVER 11.4). The toolkit installation in
# /usr/lib/x86_64-linux-gnu is the one used by the unfer GPU
# tests; on systems with the toolkit in /usr/local/cuda-12.x
# the CUBLAS_STATUS_ARCH_MISMATCH error (AGENTS.md §5) occurs.
#
# Use with:  nix-shell
# or:        nix develop
#
# This flake is opt-in: it's only loaded if the user has Nix
# installed and explicitly invokes nix-shell/nix develop in
# the unfer workspace. The CUDA-on-CI job
# (qfm-tomo-e2e-cuda in .github/workflows/ci.yml) does NOT
# require Nix; it runs on a self-hosted runner with the CUDA
# toolkit pre-installed.

{
  description = "unfer kernel + CUDA toolkit pinned environment (P9 P12) + reproducible-build packages (P11.23) + Haskell Egison workspace";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  # Separate, newer channel for `packages.*` (P11.23, unfer_nixvm): the workspace's
  # Cargo.toml pins `edition = "2024"`, which needs a rustc newer than nixos-23.05
  # ships. The CUDA devShell above is untouched and keeps its deliberate 23.05 pin
  # (P9.12's toolkit-pinning rationale); only the reproducible-build packages below
  # use the unstable channel's rustc/cargo.
  inputs.nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, nixpkgs-unstable }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;  # CUDA toolkit is unfree
      };
      cudaToolkit = pkgs.cudaPackages_12_1.cudatoolkit;
      pkgsUnstable = import nixpkgs-unstable { system = "x86_64-linux"; };

      # Configure Haskell package set with targeted Egison modules from the unstable channel
      haskellEnv = pkgsUnstable.haskellPackages.ghcWithPackages (ps: [
        ps.egison-pattern-src          # Core Egison pattern AST
        ps.egison-pattern-src-th-mode # Template Haskell bindings
        ps.template-haskell           # Built-in Template Haskell library
      ]);
    in
    {
      # P11.23: `unfer_ffi` (the handle-based C ABI, cdylib+rlib) built as a
      # reproducible Nix derivation from the workspace's own Cargo.lock — CPU-only
      # (default features; no `cuda`), matching the workspace's CPU-default
      # convention (S1). This is the artifact `unfer_nixvm/` installs into the
      # GPU-shared VM's `/nix/store`, from which — because the store is content-
      # addressed and shared with the host over virtiofs — the exact same build is
      # usable on either side with no copy (see `../unfer_nixvm/README.md`).
      packages.x86_64-linux.unfer-ffi = pkgsUnstable.rustPlatform.buildRustPackage {
        pname = "unfer_ffi";
        version = "0.1.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;
        buildAndTestSubdir = "unfer_ffi";
        # The full workspace test suite (fock_sirk, qfm, etc.) is exercised by CI;
        # this derivation only needs to produce the unfer_ffi artifacts.
        doCheck = false;
      };

      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "unfer-cuda-12.2";

        # The CUDA toolkit, system utilities, and Haskell tools
        packages = with pkgs; [
          cudaToolkit
          gcc
          gnumake
          pkg-config
          rustup
          
          # Haskell Integration
          haskellEnv
          pkgsUnstable.cabal-install
          pkgsUnstable.haskell-language-server
        ];

        # Prepend the CUDA toolkit libraries to LD_LIBRARY_PATH so
        # the linker picks them up first (the load-bearing Stage 2
        # Gram eigendecomp uses cuSOLVER; the Stage 6 reconstruction
        # uses cuBLAS for the per-row renormalization).
        shellHook = ''
          export LD_LIBRARY_PATH="${cudaToolkit}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export CUDA_HOME="${cudaToolkit}"
          export PATH="${cudaToolkit}/bin:$PATH"
          echo "[unfer-cuda-shell] CUDA ${cudaToolkit.version or "12.2"} on LD_LIBRARY_PATH"
          echo "[unfer-cuda-shell] Haskell environment with Egison loaded"
        '';

        # Rustup default toolchain.
        RUSTUP_TOOLCHAIN = "stable";
      };
    };
}