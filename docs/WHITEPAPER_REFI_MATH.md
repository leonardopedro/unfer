# ReFi Exchange — White-Paper Math Flow (UN → QuePaxa → Taler)

> Working draft for Plan R step 5. Every primitive named here is the concrete
> implementation in `unfer_consensus/src/certs.rs` plus the consensus binding in
> `unfer_consensus/src/node.rs`. The confidential zk-TLS and Taler layers are
> additive; the transparent core is what ships today.

## 0. Notation

| Symbol | Meaning | Where |
|---|---|---|
| $H(\cdot)$ | SHA-256, domain-separated by `"unfer:coin"`, `"unfer:nullifier"`, `"unfer:smt"` | `certs.rs` |
| $C_i$ | A coin commitment `CertId` | [`commit_coin`](unfer_consensus/src/certs.rs) |
| $\nu(C_i)$ | Nullifier of a coin | [`nullifier_for`](unfer_consensus/src/certs.rs) |
| $\mathcal{U}$ | The unspent set (UTXO map `coin_id → Coin`) | `CertificateLedger.utxos` |
| $\Sigma$ | The spent set (consumed nullifiers) | `CertificateLedger.spent` |
| $r$ | Sparse-Merkle root over UTXOs at depth 256 | `CertificateLedger.root()` |
| $m$ | Tonnage (Mg CO₂e) minted from a UNFCCC cert | `CertificateOpKind::Mint { amount }` |
| $\mathrm{DID}_{\mathrm{A}}$ | Subject identity (owner / mint authority) | DID registry |

## 1. The UTXO commitment

A certificate is committed at minting time:

$$
C = \mathrm{SHA256}(\texttt{"unfer:coin"} \,\|\, \mathrm{le}_{64}(m) \,\|\, \mathrm{DID}_{\mathrm{owner}} \,\|\, \beta)
$$

where $\beta \in \{0,1\}^{256}$ is a client-chosen blinding factor. `CertId = C`.
The binding is to `<amount, owner, blinding>` (transparent); a confidential run
replaces `owner` with a public-key commitment and pushes only $C$ onto the wire.

The unspent set is committed by a depth-256 binary sparse Merkle tree. Empty
subtrees at each level `d` have a fixed default hash

$$
D_d = H_2(D_{d+1}, D_{d+1}), \qquad D_{256} = 0^{256},
$$ with interior nodes $H_2(x_L, x_R) = \mathrm{SHA256}(\texttt{"unfer:smt"} \,\| x_L \| x_R)$. The system
state is fully captured by $(r, \Sigma, \mathcal{U})$.

## 2. UN → ledger: the mint flow

```
user ──(TLSNotary proof of unfccc.int receipt page)──▶ MintAuthority DID
                                                          │ signs
                                                          ▼
                                    CertificateOp::Mint { amount = m, owner = DID_u, source = "unfccc:cert:<id>", blinding = β }
                                                          │ sequenced on the consensus log
                                                          ▼
                                              CertificateLedger.apply_mint
```

State transition of a mint:

$$
\mathcal{U}' = \mathcal{U} \cup \{C \mapsto (m, \mathrm{DID}_u, \beta, \mathrm{seq})\},
\qquad r' = \mathrm{SPARSEINSERT}(r, C),
\qquad \Sigma' = \Sigma.
$$

The zk-TLS prover (Phase 3) verifies the UNFCCC receipt page embeds the user's
`did:unfer` public key and the tonnage $m$; the prover's output becomes the
`source` provenance record. Mint authority check: `apply_mint` refuses unless
the actor is the configured `MintAuthority::Only(authority)` (UK-7001) and
`m > 0` (UK-7002). Minting is idempotent per commitment: re-submitting the same
$C$ is a no-op, so replaying the log never double-issues.

## 3. QuePaxa binding: deterministic replay

Every `CertificateOp` is a `ConsensusTransaction` variant on the signed,
ordered log. Each node recomputes state purely from the log:

$$
\mathcal{S}_i = F\bigl(\mathcal{S}_{i-1},\, \mathrm{op}_i\bigr), \qquad
r_i = F_r\bigl(r_{i-1},\, \mathrm{op}_i\bigr),
$$ where $F$ is `CertificateLedger::apply_op` — a pure function with no
timing, randomness, or machine dependence. Two nodes that observe the same
signed prefix therefore hold identical $(\mathcal{U}_i, \Sigma_i, r_i)$; this is
the "deterministic root" convergence property asserted by
`certificate_ledger_roundtrip_via_consensus`.

The rapid-validation rule rejects an op **before** it is sequenced
(`apply_op` returns a `Diagnostic` with a UK-7xxx code), so invalid
transactions never enter the ledger and never disturb the root.

## 4. Transfer: conservation, double-spend, ownership

A transfer spends inputs $\{C_1,\dots,C_k\}$ and creates outputs
$\{C'_1,\dots,C'_l\}$:

$$
\nu(C_j) \in \Sigma \;\Longrightarrow\; \text{refuse (UK-7004)},\qquad
\sum_j m_j = \sum_t m'_t \;\text{or refuse (UK-7002)},
$$

$$
C_j \notin \mathcal{U} \;\Longrightarrow\; \text{refuse (UK-7003)},\qquad
\mathrm{DID}_{\mathrm{signer}} \ne \mathrm{DID}_{\mathrm{owner}}(C_j) \;\Longrightarrow\; \text{refuse (UK-7005)}.
$$

On success:

$$
\mathcal{U}' = \bigl(\mathcal{U} \setminus \{C_j\}_j\bigr) \cup \{C'_t\}_t,
\qquad \Sigma' = \Sigma \cup \{\nu(C_1), \dots, \nu(C_k)\},
$$
$$
r' = \mathrm{SPARSEREMOVE}^{k}( \mathrm{SPARSEINSERT}^{l}(r) ).
$$

Because $m'$ are minted at the same total $m$, `total_supply` is invariant
under transfers; only mint and burn move the aggregate. The property test
`fuzz_transfers_never_break_conservation_or_double_spend` fuzzes random
`(idx, out_amount)` sequences and asserts exactly these three invariants
across the whole run.

## 5. QuePaxa → Taler: peg-in, audit, peg-out

The GNU Taler exchange keeps a treasury DID. Pegging preserves the transparent
audit trail because every UN-backed tonne is a `CertificateOp` on the same
deterministic log.

- **Peg-in** — `CertificateOp::Transfer` from the holder DID to the exchange
  treasury DID. The exchange now holds $\sum m$ UN-backed value and may issue an
  equivalent amount of Taler e-coins for retail.
- **Audit** — the exchange reconciles against one value: the Merkle root $r$. A
  verifier re-derives $r$ from the public log (`uk_cert_root`) and must match
  the exchange's claimed liabilities; any divergence is a mint/transfer that
  broke conservation (UK-7002) or a spent un-removed coin (UK-7004).
- **Peg-out** — the treasury DID signs a `Transfer` to the merchant DID; the
  e-coins retire. `burn_retires_value` covers the retirement leg.

The whole path is therefore three transparent state transitions:

```
UNFCCC tonnage m ──Mint──▶ certificate on log ──Transfer──▶ treasury DID ──Taler e-coin──▶ merchant
                                (root r, immutable)                (audit vs root r)
```

## 6. Security properties (what a proof would show)

1. **Conservation**: For every accepted `Transfer`, $\sum \mathrm{in} = \sum \mathrm{out}$.
   (UK-7002 + proptest.)
2. **No double-spend**: A nullifier enters $\Sigma$ exactly once; a second use
   refutes (UK-7004 + proptest).
3. **No mintage**: The only way $\sum \mathcal{U}$ grows is an authorized mint
   (UK-7001); the only way it shrinks is a burn.
4. **Soundness of $r$**: `contains(C)` over the sparse tree matches $C \in \mathcal{U}$
   at every log index (SMT insert/remove roundtrip).
5. **Agreement**: two nodes replaying the same signed log end with equal $r$
   (deterministic `F`).

The confidential upgrade replaces `commit_coin`/`nullifier_for` with
`Hash(spend_key, commitment)` and a RISC-V `risc0` guest mirroring `apply_op`,
emitting receipts that nodes verify (Phase 1 follow-up). The invariants above
are the guest's specification.

## 7. Open questions

- Fee structure for pegs (current `apply_transfer` requires exact conservation;
  fees need a `Fees` output or an authorized burn-before-transfer).
- Multi-input ordering requirements (currently single-input transfers in tests;
  proptest fuzzes 1-input only).
- Taler denomination mapping vs. $m$ granularity (tonne → euro-coin rounding).