# unfer_identity

`did:unfer` identity method backed by the QuePaxa consensus log.
`DidManager` implements the create / resolve / update / revoke lifecycle
over a `ConsensusNode`, emitting signed `IdentityOp` transactions.
`did_from_pubkey` / `pubkey_from_did` convert between the 32-byte Ed25519
public key and the `did:unfer:<hex>` form.