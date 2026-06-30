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
  description = "unfer kernel + CUDA toolkit pinned environment (P9 P12)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";

  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        config.allowUnfree = true;  # CUDA toolkit is unfree
      };
      cudaToolkit = pkgs.linuxPackages.nvidiaPackages.mkCudaToolkit12_2;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "unfer-cuda-12.2";

        # The CUDA toolkit: nvcc, libcublas, libcudart, etc.
        packages = with pkgs; [
          cudaToolkit
          gcc
          gnumake
          pkg-config
          rustup
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
        '';

        # Rustup default toolchain.
        RUSTUP_TOOLCHAIN = "stable";
      };
    };
}
