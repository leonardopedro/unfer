# Seed note: hybrid CPU/GPU SIRK architecture

**Category**: architecture
**Status**: implemented

The split-mode "Inverse-Free Rational Krylov" (SIRK) pipeline: the forward
sequence `w_k = (H - z_k I) w_{k-1}` runs on the CPU (symbolic CAS + sparse
`FxHashMap`), then basis states flatten into a `StateDictionary` for GPU Gram /
reduced-Hamiltonian computation via candle-core CUDA kernels. Gram whitening
uses Hermitian eigendecomposition (not bare Cholesky). See AGENTS.md.
