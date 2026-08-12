# unfer_consensus

QuePaxa-backed consensus and federation for the unfer protocol layer.
`engine` is the consensus engine (`ConsensusEngine`, `LocalConsensus`),
`node` the signing `ConsensusNode` that applies transactions
deterministically, `net` (feature `network`) the TLS/relay transport,
`signing` the Ed25519 keypairs, `identity` the DID registry, `escrow` the
two-phase escrow service, and `certs` the UTXO/carbon-certificate ledger
with its `SparseMerkle` (Plan R). See `docs/FEDERATION.md`.