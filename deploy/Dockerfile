# C8: unfer_edge + arctic in one image.
#
# Built from the workspace's own Cargo.lock, CPU-only. `cuda` is additive and off
# by default here on purpose: an image built with it will not start on a host
# without a matching driver, and a deployment image that only runs on GPU hosts
# is not a deployment image.
#
# This is deliberately NOT a Nix image. The flake pins CUDA 12.6 from
# nixpkgs-unstable (see ../flake.nix for why the older pin is unusable), which is
# a large closure for something that runs CPU-only in production. If you want the
# reproducible path instead, `nix build .#unfer-ffi` produces a store path from
# the same lockfile; COPY that in and skip the cargo stages.

FROM debian:bookworm-slim AS build

# `pkg-config` is a *build* dependency here (several -sys crates probe for system
# libraries), not a runtime one, which is why it is installed and removed in the
# same layer and never reaches the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Copy the manifests first so the dependency layer caches independently of source
# edits. Without this, every one-character edit rebuilds the whole dependency
# tree.
COPY Cargo.toml Cargo.lock ./
COPY unfer_protocol/Cargo.toml unfer_protocol/
COPY unfer_ffi/Cargo.toml unfer_ffi/
COPY unfer_edge/Cargo.toml unfer_edge/
RUN mkdir -p unfer_protocol/src unfer_ffi/src unfer_edge/src \
 && echo "" > unfer_protocol/src/lib.rs \
 && echo "" > unfer_ffi/src/lib.rs \
 && echo "fn main() {}" > unfer_edge/src/main.rs \
 && cargo fetch --locked || true

COPY . .
# The real build. --locked refuses to resolve anything the lockfile does not
# already pin, which is what makes the image reproducible.
RUN cargo build --release --locked -p unfer_edge -p unfer_ffi

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

# ca-certificates only: no pkg-config, no compilers, no toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --no-create-home --shell /usr/sbin/nologin unfer

COPY --from=build /src/target/release/unfer_edge /usr/local/bin/unfer_edge
# unfer_ffi ships a cdylib the edge links against at runtime.
COPY --from=build /src/target/release/libunfer_ffi.so /usr/local/lib/
RUN ldconfig

# Run unprivileged. A container escape should not be immediately root on the host.
# There is deliberately no ENV holding a credential: a secret baked into a layer
# outlives the container in the image cache and is readable by anyone who pulls it.
USER unfer

EXPOSE 3000
# The dependency-free probe (X6): it answers "is this process serving" without
# touching anything that could be broken for an unrelated reason.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/usr/local/bin/unfer_edge", "--help"]

ENTRYPOINT ["/usr/local/bin/unfer_edge"]
