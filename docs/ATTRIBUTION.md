# Attribution Carbon Credits (Open Badges + Taler Micropayments)

> Architecture, design rationale, and test coverage for the author-approved
> attribution market, implemented in `unfer_consensus::attribution` and
> `unfer_taler::attribution`.

## Overview

The Adidas/Yeezy deal as a tradable certificate. Author A (Adidas) pays Author
B (Kanye) for the right to publicly claim that A's item is *derived from* B's
item and that B *approves* that attribution. What the brand buys is not just
design skill — it is the fact that many people want to wear shoes that the
artist publicly acknowledged as derived from his work, regardless of the
design's intrinsic merit.

These **Attribution Carbon Credits** play the same role as ordinary
attributions (Creative Commons, citations, web links), with one difference:
they were **approved by the author** being attributed, in exchange for a
negotiated fee. This is complementary to copyright licensing:

- it works for **public-domain** items (there is no license to buy — but the
  author's endorsement is still worth paying for);
- it works for **private** items too, because the credit is issued by the
  author corresponding to the attribution, not by the law.

Using GNU Taler micropayments, issuing Attribution Badges can be automated,
and a badge can display the contribution amount and context — publicly, or
exclusively for an anonymous user. A video on YouTube could display an
Attribution Badge per visualization (the viewer's browser generates a random
key; the operator only ever sees its SHA-256), so viewers know the video's
creator is paying an author who was relevant for that video.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Author A (derived item's owner, the payer)                        │
│  • registers a work item (content-addressed)                       │
│  • escrows the negotiated fee e-coin (Taler)                       │
│  • offers Author B an attribution credit (derived ← original)      │
└────────────────────────┬────────────────────────────────────────────┘
                         │ AttributionOp / CertificateOp
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AttributionLedger (unfer_consensus::attribution)                  │
│                                                                     │
│  CreditState: Offered → Approved → Revoked                          │
│                                                                     │
│  Operations:                                                        │
│  • RegisterItem: content-addressed ownership registry              │
│  • OfferAttribution: terms + fee, exclusivity check                │
│  • Approve (Author B only): Offered → Approved                     │
│  • Revoke (Author B only): Approved → Revoked                      │
│  • IssueBadge: deterministic OB 3.0 assertion id per (credit,       │
│    recipient) — public or anonymous-per-view                       │
│                                                                     │
│  Determinism: ids are commitments, root() converges on replay      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐  ┌──────────────────────────────────────────┐
│  CertificateLedger  │  │  AttributionService (unfer_taler)        │
│  (escrow + payout)  │  │                                          │
│                     │  │  • fee escrow into deterministic DID    │
│  The fee e-coin     │  │  • approval → fee released to Author B  │
│  moves: A → escrow  │  │  • abandoned offer → fee refunded to A  │
│  → B, conserving    │  │  • Open Badges 3.0 assertion minted     │
│  total_supply       │  │    (Ed25519 proof by the operator)      │
└─────────────────────┘  └──────────────────────────────────────────┘
```

## The Ledger (`unfer_consensus::attribution`)

A pure state machine (no wall-clock, no RNG) following the certificate /
auction / math-bond pattern:

- **Items** are content-addressed (`item_hash`); registering the same work
  twice by different authors collides (UK-7504).
- **Offers** are validated against the registry: both items must exist
  (UK-7505), the derived item must belong to the offerer (UK-7508), the
  original to a *different* author (UK-7506), and the fee must be positive
  (UK-7507). Identical terms by the same author pair collide (UK-7509).
- **Exclusivity**: while an exclusive credit for an original item is live
  (`Offered`/`Approved`), no second offer against that item is accepted —
  the Adidas/Yeezy sole-claim deal. Revocation reopens the item.
- **Approve** (UK-7501/7502/7503 guards) is the on-ledger moment the
  attribution becomes author-approved; the settlement service releases the
  escrowed fee on the same call.
- **Revoke** withdraws the endorsement; already-issued badges stay valid as
  historical, content-addressed records, but no new badge is minted
  (UK-7510).
- **Badges** are deterministic commitments of `(credit_id, recipient)`:
  `None` = the public badge, `Some(sha256-of-random-key)` = a badge exclusive
  to that anonymous viewer (UK-7511 refuses a duplicate mint).

`credit_id` and `badge_id` are SHA-256 commitments; `root()` commits the whole
ledger, so a peer replaying the same log converges on identical state.

## The Settlement Service (`unfer_taler::attribution`)

The operator-side service runs the deterministic ledger and settles value with
ordinary certificate transfers — no value is created or destroyed
(`total_supply` balances before and after every lifecycle):

1. **`register_item`** — emits a signed `RegisterItem` op.
2. **`offer(A, terms, funding)`** — validates the fee e-coin (face value must
   equal the negotiated fee, UK-7512), rows it into the deterministic escrow
   DID derived from the operator master key + credit id (so a peer replaying
   the log can regenerate the same DID), then emits the signed offer. A
   refused offer (e.g. self-attribution) refunds the fee immediately —
   nothing is stranded.
3. **`approve(B, credit)`** — the on-ledger `Approve` first; the escrowed fee
   then moves from the escrow DID to Author B. The badge issuance date is
   derived deterministically from the consensus-log approval position.
4. **`revoke(B, credit)`** — the endorsement is withdrawn; the fee was
   already paid at approval (the endorsement was bought) and is not clawed
   back.
5. **`issue_badge(requester, credit, viewer)`** — mints the deterministic
   Open Badges 3.0 assertion (below) and signs it with the operator key.

### Open Badges 3.0 assertions

The badge is a W3C Verifiable-Credential (data model v2.0) document with the
OB 3.0 context:

```json
{
  "@context": [
    "https://www.w3.org/ns/credentials/v2",
    "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.3.json"
  ],
  "id": "urn:unfer:attribution:badge:<badge_id>",
  "type": ["VerifiableCredential", "OpenBadgeCredential"],
  "issuer": { "id": "<operator did>", "type": "Profile", "name": "unfer attribution operator" },
  "issuanceDate": "<deterministic RFC 3339 from approval seq>",
  "credentialSubject": {
    "id": "urn:unfer:attribution:credit:<credit_id>",
    "type": "AttributionCredit",
    "derived":  { "id": "...", "name": "Yeezy Boost 350 V2", "owner": "<Author A>" },
    "original": { "id": "...", "name": "Kanye's 2015 sketch", "owner": "<Author B>" },
    "approvedBy": "<Author B>",
    "fee": 5000,
    "context": "Yeezy line, 2023 collection",
    "exclusive": true,
    "viewer": null | "<sha256 of the viewer's random key>"
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "created": "...",
    "verificationMethod": "<operator did>#attribution",
    "proofPurpose": "assertionMethod",
    "proofValue": "<Ed25519 over SHA-256 of the canonical document>"
  }
}
```

**Determinism**: every field is a pure function of ledger state and the
operator key; `serde_json` serializes map keys in sorted order, so the same
(credit, recipient) always produces byte-identical JSON, proof, and id. The
issuance instant is `2020-01-01T00:00:00Z + approve_seq` seconds — no
wall-clock, so replaying the log reproduces the identical badge.

**Honest boundary**: the Ed25519 proof covers the SHA-256 of the canonical
JSON (a content-addressed commitment), not a full RDFC-1.0 JSON-LD
canonicalization of the Linked-Data graph. The badge's trust anchors are the
on-ledger approval (every node can verify Author B signed the `Approve` op)
and the operator's signature over the deterministic document; adding
standards-track Linked-Data proof canonicalization is a documented future
layer, not a correctness gap for the ledger itself.

**Anonymity**: the operator only ever sees the SHA-256 of the viewer's random
key — never the key itself. The viewer proves ownership of their badge by
revealing the preimage. This is the per-visualization badge: each view (each
new random key) mints a distinct badge id for the same credit.

## Test coverage

| Test | What it verifies |
|---|---|
| `full_lifecycle_escrow_approve_payout_badge_revoke` | Offer escrows the fee (neither author holds it); only B approves; approval pays B exactly the fee; supply conserved; public + anonymous badges mint; re-issue refused (UK-7511); revoke is B-only and stops new badges |
| `offer_refuses_fee_mismatch_and_unknown_item` | Unknown original item refused before escrow; wrong face value refused (UK-7512) with the coin untouched; a ledger-refused offer refunds the escrow — nothing stranded |
| `badge_assertions_are_deterministic_across_services` | Two services with the same inputs mint byte-identical JSON, proof, date, and id |
| `replay_converges_the_consensus_node` | `ops()` replayed into a `ConsensusNode` → identical certificate root, attribution root, supply, state and badge set |

Ledger-internal tests (`unfer_consensus::attribution::tests`) additionally
cover the full lifecycle state machine, offer validations, exclusivity
blocking, and same-log/same-root determinism.

## Error codes

UK-7501..UK-7512 (see `unfer_protocol::codes`), plus the shared escrow codes
(UK-7201..7203) for the settlement side.

## License

The attribution implementation is part of the unfer kernel (Apache-2.0). Open
Badges 3.0 is an IMS Global open standard; the assertion follows its
documented JSON-LD structure (no code copied). GNU Taler is GPL-licensed —
this crate only *adapts its flow* over the certificate ledger and never links
Taler code.
