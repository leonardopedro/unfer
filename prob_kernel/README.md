# prob_kernel

The Born-rule probability layer over the unfer QFT kernel. `Session` is the
public API: `evolve`, `probability`, `condition`, `snapshot`, and the
Bayesian-update path. `build` hosts the model-spec compiler (Pauli–Grover,
diffusion, random-start Hamiltonian types), `event` the kernel-event types,
and `error` the `KernelError` diagnostics. All wire types come from
`unfer_protocol`.