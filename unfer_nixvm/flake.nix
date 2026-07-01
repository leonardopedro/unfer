{
  description = ''
    unfer_nixvm — GPU-shared Nix execution sandbox for the unfer kernel (P11.23).

    Adapts ../cloud-hypervisor-build (Apache-2.0/BSD-3-Clause: cloud-hypervisor
    50.0 + the spectrum0 GPU-sharing patches + nixos-generators vm-perf/vm-sec
    images). Does not vendor code from it — this flake *composes* with its
    existing configuration.nix by adding a NixOS module on top, so upstream's
    module stays the single source of truth for the VM's virtiofs/GPU/network
    setup. UX is inspired (never vendored, GPL-3.0) by ../claurst's
    agent-drives-the-VM interaction style.

    Mechanism: because /nix/store paths are content-addressed and immutable,
    once `unfer.packages.x86_64-linux.unfer-ffi` (built by the parent
    ../flake.nix, P11.23's Nix-package half) is realized on the host, that
    exact store path is what the guest's virtiofs-shared /nix already
    contains — no copy, no rebuild inside the VM. See README.md for the full
    picture and the known bug this flake documents (but does not silently
    work around) in ../../cloud-hypervisor-build/full-stack-vm-launch.sh.

    Lives at unfer/unfer_nixvm/ (a subdirectory of the unfer repo itself,
    not a separate sibling checkout) since its packages output is entirely
    about the unfer kernel; ../../cloud-hypervisor-build stays an external
    sibling (composed via a path input, not vendored — see the
    cloud-hypervisor-build input below).
  '';

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixos-generators.url = "github:nix-community/nixos-generators";
    # The parent unfer flake (one level up — this is unfer/unfer_nixvm/),
    # for its `packages.x86_64-linux.unfer-ffi` output.
    unfer.url = "path:..";
    # `flake = false`: we only want ../../cloud-hypervisor-build's raw
    # configuration.nix as a NixOS module to import, not to evaluate its own
    # flake.nix (which builds its own, separate vm-perf/vm-sec outputs).
    # Absolute path deliberately: cloud-hypervisor-build is an external
    # sibling checkout (see AGENTS.md's sibling-checkout convention), not a
    # subdirectory of this repo, so a relative "path:../../x" would silently
    # break once this flake is evaluated from a git store copy rather than
    # the working tree.
    cloud-hypervisor-build = {
      url = "path:/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/cloud-hypervisor-build";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nixos-generators, unfer, cloud-hypervisor-build }:
    let
      system = "x86_64-linux";

      # ../../cloud-hypervisor-build's own guest module: virtiofs /nix +
      # optional ssh-agent share, GPU passthrough (hardware.opengl.enable),
      # the `agent` user, sshd. We depend on it as a flake input rather than
      # copying configuration.nix, so upstream changes there are picked up
      # automatically.
      baseModule = "${cloud-hypervisor-build}/configuration.nix";

      # Layered on top: install the unfer kernel's C ABI (unfer_ffi, built
      # reproducibly by the parent flake's `packages.x86_64-linux.unfer-ffi`)
      # into the guest, so the shared-store mechanism above is exercised by a
      # real, useful package rather than a toy example.
      unferGuestModule = { config, pkgs, ... }: {
        environment.systemPackages = [
          unfer.packages.${system}.unfer-ffi
        ];
      };
    in {
      packages.${system} = {
        # Performance strategy (Nix store sharing + git from /nix, no SSH
        # socket) + the unfer kernel pre-installed.
        vm-perf-with-unfer = nixos-generators.nixosGenerate {
          inherit system;
          format = "raw";
          modules = [ baseModule unferGuestModule ];
        };

        # Secure/agent strategy (same as perf + SSH agent socket forwarding)
        # + the unfer kernel pre-installed.
        vm-sec-with-unfer = nixos-generators.nixosGenerate {
          inherit system;
          format = "raw";
          modules = [ baseModule unferGuestModule ];
        };
      };
    };
}
