//! Attribution carbon credits (Open Badges + Taler micropayments, Plan R).
//!
//! The Adidas/Yeezy deal as a tradable certificate: Author A pays Author B for
//! the right to publicly claim that A's item is *derived from* B's item and
//! that B approves that attribution. The fee is escrowed (see
//! `unfer_taler::attribution`) before the signed offer is emitted; Author B's
//! on-ledger [`Approve`](AttributionOpKind::Approve) mints the credit — the
//! author-approved analogue of a Creative-Commons-style attribution, one that
//! works for public-domain and private items alike because it is issued by the
//! author being attributed.
//!
//! The credit is rendered as a deterministic Open Badges 3.0 assertion
//! (public, or exclusive to an anonymous viewer identified by the SHA-256 of a
//! random key their browser generates — the per-visualization badge of a
//! YouTube-style context).
//!
//! The ledger is a pure state machine (no wall-clock, no RNG): every op is
//! validated against the recorded items/credits, ids are deterministic
//! commitments, and [`root`](AttributionLedger::root) lets a peer replaying
//! the same log check convergence on the identical state.
//!
//! Error codes: UK-7501..UK-7511 (see `unfer_protocol::codes`).

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use unfer_protocol::{
    AttributionBadgeId, AttributionCreditId, AttributionItem, AttributionOffer, AttributionOpKind,
    AttributionReport, AttributionState, Code, Diagnostic, Severity,
};

/// Internal full state for one attribution credit.
#[derive(Debug, Clone)]
pub struct CreditState {
    pub credit_id: AttributionCreditId,
    pub offer: AttributionOffer,
    pub author_a: String,
    pub author_b: String,
    pub state: AttributionState,
    pub approve_seq: Option<u64>,
    pub revoke_seq: Option<u64>,
    pub badges: Vec<AttributionBadgeId>,
}

/// The deterministic attribution state-transition engine.
#[derive(Debug, Default)]
pub struct AttributionLedger {
    /// Registered work items: `item_hash` → (item descriptor, owner DID).
    /// `BTreeMap` so iteration (and thus `root`) is deterministic.
    items: BTreeMap<[u8; 32], (AttributionItem, String)>,
    /// Attribution credits keyed by their deterministic commitment.
    credits: BTreeMap<[u8; 32], CreditState>,
}

/// The deterministic credit id for an offer by `author_a` to `author_b` — a
/// commitment of the full terms plus both authors, so identical terms by the
/// same author pair collide (a credit is a unique fact).
pub fn credit_id(offer: &AttributionOffer, author_a: &str, author_b: &str) -> AttributionCreditId {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:attribution:credit:v1");
    ctx.update(author_a.as_bytes());
    ctx.update(author_b.as_bytes());
    ctx.update(offer.derived_item.item_hash);
    ctx.update(offer.original_item.item_hash);
    ctx.update(offer.fee.to_le_bytes());
    ctx.update(offer.context.as_bytes());
    ctx.update([u8::from(offer.exclusive)]);
    AttributionCreditId(ctx.finalize().into())
}

/// The deterministic badge id for `credit_id` issued to a recipient. `viewer`
/// is `None` for the public badge (visible to all users) or `Some(hash)` for a
/// badge exclusive to the anonymous viewer whose browser generated the key
/// whose SHA-256 is `hash`. The same credit issued to the same recipient is
/// the *same* badge.
pub fn badge_id(credit_id: &AttributionCreditId, viewer: Option<[u8; 32]>) -> AttributionBadgeId {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:attribution:badge:v1");
    ctx.update(credit_id.0);
    match viewer {
        Some(v) => ctx.update(v),
        None => ctx.update([0u8; 32]),
    }
    AttributionBadgeId(ctx.finalize().into())
}

impl AttributionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a registered item by its content hash.
    pub fn item(&self, item_hash: &[u8; 32]) -> Option<&AttributionItem> {
        self.items.get(item_hash).map(|(item, _)| item)
    }

    /// The owner DID of a registered item, if registered.
    pub fn item_owner(&self, item_hash: &[u8; 32]) -> Option<&str> {
        self.items.get(item_hash).map(|(_, owner)| owner.as_str())
    }

    /// Look up a credit by id.
    pub fn credit(&self, id: &AttributionCreditId) -> Option<&CreditState> {
        self.credits.get(&id.0)
    }

    /// Read-only report for a credit.
    pub fn report(&self, id: &AttributionCreditId) -> Option<AttributionReport> {
        let c = self.credits.get(&id.0)?;
        Some(AttributionReport {
            credit_id: c.credit_id,
            offer: c.offer.clone(),
            state: c.state,
            author_a: c.author_a.clone(),
            author_b: c.author_b.clone(),
            approve_seq: c.approve_seq,
            revoke_seq: c.revoke_seq,
            badges: c.badges.clone(),
        })
    }

    /// All credits on the ledger (sorted by id, for deterministic reports).
    pub fn all_credits(&self) -> Vec<AttributionReport> {
        self.credits
            .values()
            .filter_map(|c| self.report(&c.credit_id))
            .collect()
    }

    /// Deterministic commitment of the whole ledger: replay the same log on
    /// two nodes and the roots match iff the states match.
    pub fn root(&self) -> [u8; 32] {
        let mut ctx = Sha256::new();
        ctx.update(b"unfer:attribution:root:v1");
        for (hash, (item, owner)) in &self.items {
            ctx.update(b"i");
            ctx.update(hash);
            ctx.update(owner.as_bytes());
            ctx.update(item.title.as_bytes());
            if let Some(url) = &item.url {
                ctx.update(url.as_bytes());
            }
        }
        for (id, c) in &self.credits {
            ctx.update(b"c");
            ctx.update(id);
            let state = match c.state {
                AttributionState::Offered => 0u8,
                AttributionState::Approved => 1u8,
                AttributionState::Revoked => 2u8,
            };
            ctx.update([state]);
            ctx.update(c.approve_seq.unwrap_or(0).to_le_bytes());
            ctx.update(c.revoke_seq.unwrap_or(0).to_le_bytes());
            for b in &c.badges {
                ctx.update(b.0);
            }
        }
        ctx.finalize().into()
    }

    /// Dispatch a signed attribution op against the ledger. `actor` is the
    /// signer's DID (already verified by the caller); `seq` is the
    /// consensus-log sequence.
    pub fn apply_op(
        &mut self,
        actor: &str,
        kind: &AttributionOpKind,
        seq: u64,
    ) -> Result<(), Diagnostic> {
        match kind {
            AttributionOpKind::RegisterItem { item } => {
                self.apply_register_item(actor, item)?;
            }
            AttributionOpKind::OfferAttribution { offer } => {
                self.apply_offer(actor, offer)?;
            }
            AttributionOpKind::Approve { credit_id } => {
                self.apply_approve(actor, credit_id, seq)?;
            }
            AttributionOpKind::Revoke { credit_id } => {
                self.apply_revoke(actor, credit_id, seq)?;
            }
            AttributionOpKind::IssueBadge { credit_id, viewer } => {
                self.apply_issue_badge(credit_id, *viewer, seq)?;
            }
        }
        Ok(())
    }

    /// Author A registers a work item they own. The item is content-addressed:
    /// the same `item_hash` registered twice by a different author is refused.
    fn apply_register_item(
        &mut self,
        actor: &str,
        item: &AttributionItem,
    ) -> Result<(), Diagnostic> {
        if item.title.is_empty() {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_ITEM_UNKNOWN,
                "item title must not be empty",
                Severity::Error,
            ));
        }
        if let Some((_, owner)) = self.items.get(&item.item_hash) {
            if owner != actor {
                return Err(Diagnostic::new(
                    Code::ATTRIBUTION_ITEM_EXISTS,
                    format!(
                        "item {:#x?} is already registered to {owner}",
                        item.item_hash
                    ),
                    Severity::Error,
                ));
            }
            return Ok(()); // idempotent re-registration by the same owner
        }
        self.items
            .insert(item.item_hash, (item.clone(), actor.to_string()));
        Ok(())
    }

    /// Author A offers Author B a fee for an attribution credit referencing
    /// B's original item. Both items must be registered and owned by their
    /// authors; the derived and original works must differ; the fee positive;
    /// and a live *exclusive* credit for the same original item blocks a new
    /// offer (the Adidas/Yeezy exclusivity deal).
    fn apply_offer(&mut self, actor: &str, offer: &AttributionOffer) -> Result<(), Diagnostic> {
        // The derived item must be registered to the offerer (Author A).
        let derived_owner = self
            .items
            .get(&offer.derived_item.item_hash)
            .map(|(_, owner)| owner.as_str())
            .ok_or_else(|| {
                Diagnostic::new(
                    Code::ATTRIBUTION_ITEM_UNKNOWN,
                    "the derived item was never registered",
                    Severity::Error,
                )
            })?;
        if derived_owner != actor {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_OWNER_MISMATCH,
                "the derived item is not owned by the offerer",
                Severity::Error,
            ));
        }
        // The original item must be registered to a *different* author (B).
        let original_owner = self
            .items
            .get(&offer.original_item.item_hash)
            .map(|(_, owner)| owner.as_str())
            .ok_or_else(|| {
                Diagnostic::new(
                    Code::ATTRIBUTION_ITEM_UNKNOWN,
                    "the original item was never registered",
                    Severity::Error,
                )
            })?;
        if original_owner == actor {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_SELF_ATTRIBUTION,
                "an author cannot attribute their own work to themselves",
                Severity::Error,
            ));
        }
        if offer.derived_item.item_hash == offer.original_item.item_hash {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_SELF_ATTRIBUTION,
                "the derived and original works must be distinct items",
                Severity::Error,
            ));
        }
        if offer.fee == 0 {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_FEE_ZERO,
                "the negotiated fee must be positive",
                Severity::Error,
            ));
        }

        let id = credit_id(offer, actor, original_owner);
        if self.credits.contains_key(&id.0) {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_CREDIT_EXISTS,
                "a credit with these exact terms by this author pair already exists",
                Severity::Error,
            ));
        }

        // Exclusivity: while an exclusive credit for the original item is live
        // (offered or approved), no new offer against the same original is
        // accepted — the brand paid for sole claim of derivation.
        if offer.exclusive {
            for c in self.credits.values() {
                if c.offer.original_item.item_hash == offer.original_item.item_hash
                    && matches!(
                        c.state,
                        AttributionState::Offered | AttributionState::Approved
                    )
                {
                    return Err(Diagnostic::new(
                        Code::ATTRIBUTION_CREDIT_EXISTS,
                        "a live exclusive credit already covers this original item",
                        Severity::Error,
                    ));
                }
            }
        }

        self.credits.insert(
            id.0,
            CreditState {
                credit_id: id,
                offer: offer.clone(),
                author_a: actor.to_string(),
                author_b: original_owner.to_string(),
                state: AttributionState::Offered,
                approve_seq: None,
                revoke_seq: None,
                badges: Vec::new(),
            },
        );
        Ok(())
    }

    /// Author B approves an offered credit: `Offered` → `Approved`. This is
    /// the on-ledger moment the attribution becomes author-approved; the
    /// settlement service releases the escrowed fee to B on the same op.
    fn apply_approve(
        &mut self,
        actor: &str,
        id: &AttributionCreditId,
        seq: u64,
    ) -> Result<(), Diagnostic> {
        let credit = self.credits.get_mut(&id.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ATTRIBUTION_UNKNOWN_CREDIT,
                "unknown credit id",
                Severity::Error,
            )
        })?;
        if credit.state != AttributionState::Offered {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_WRONG_STATE,
                format!("credit is {:?}, expected Offered", credit.state),
                Severity::Error,
            ));
        }
        if actor != credit.author_b {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_NOT_AUTHOR,
                format!("only {} may approve this credit", credit.author_b),
                Severity::Error,
            ));
        }
        credit.state = AttributionState::Approved;
        credit.approve_seq = Some(seq);
        Ok(())
    }

    /// Author B revokes an approved credit: `Approved` → `Revoked`. Already
    /// issued badges stay valid as historical (content-addressed) records, but
    /// no new badge is minted (see `apply_issue_badge`).
    fn apply_revoke(
        &mut self,
        actor: &str,
        id: &AttributionCreditId,
        seq: u64,
    ) -> Result<(), Diagnostic> {
        let credit = self.credits.get_mut(&id.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ATTRIBUTION_UNKNOWN_CREDIT,
                "unknown credit id",
                Severity::Error,
            )
        })?;
        if credit.state != AttributionState::Approved {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_WRONG_STATE,
                format!("credit is {:?}, expected Approved", credit.state),
                Severity::Error,
            ));
        }
        if actor != credit.author_b {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_NOT_AUTHOR,
                format!("only {} may revoke this credit", credit.author_b),
                Severity::Error,
            ));
        }
        credit.state = AttributionState::Revoked;
        credit.revoke_seq = Some(seq);
        Ok(())
    }

    /// Mint a deterministic badge for an approved credit. The recipient is the
    /// public audience (`viewer = None`) or an anonymous viewer whose browser
    /// generated the key hashed into `viewer` — the operator only ever sees
    /// the hash. A revoked credit mints no new badge; the exact badge (credit
    /// + recipient) mints once.
    fn apply_issue_badge(
        &mut self,
        id: &AttributionCreditId,
        viewer: Option<[u8; 32]>,
        _seq: u64,
    ) -> Result<AttributionBadgeId, Diagnostic> {
        let credit = self.credits.get_mut(&id.0).ok_or_else(|| {
            Diagnostic::new(
                Code::ATTRIBUTION_UNKNOWN_CREDIT,
                "unknown credit id",
                Severity::Error,
            )
        })?;
        if credit.state != AttributionState::Approved {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_BADGE_REVOKED,
                format!(
                    "credit is {:?}; badges are only minted for approved credits",
                    credit.state
                ),
                Severity::Error,
            ));
        }
        let bid = badge_id(id, viewer);
        if credit.badges.contains(&bid) {
            return Err(Diagnostic::new(
                Code::ATTRIBUTION_BADGE_EXISTS,
                "this badge (credit + recipient) was already minted",
                Severity::Error,
            ));
        }
        credit.badges.push(bid);
        Ok(bid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(title: &str, byte: u8) -> AttributionItem {
        AttributionItem {
            item_hash: [byte; 32],
            title: title.to_string(),
            url: None,
        }
    }

    fn offer(
        derived: &AttributionItem,
        original: &AttributionItem,
        fee: u64,
        exclusive: bool,
    ) -> AttributionOffer {
        AttributionOffer {
            derived_item: derived.clone(),
            original_item: original.clone(),
            fee,
            context: "Yeezy line, 2023 collection".to_string(),
            exclusive,
        }
    }

    #[test]
    fn full_lifecycle_offer_approve_badge_revoke() {
        let mut ledger = AttributionLedger::new();
        let adidas = "did:unfer:adidas";
        let kanye = "did:unfer:kanye";
        let shoe = item("Yeezy Boost 350 V2", 1);
        let sketch = item("Kanye's 2015 sketch", 2);

        ledger
            .apply_op(
                adidas,
                &AttributionOpKind::RegisterItem { item: shoe.clone() },
                1,
            )
            .unwrap();
        ledger
            .apply_op(
                kanye,
                &AttributionOpKind::RegisterItem {
                    item: sketch.clone(),
                },
                2,
            )
            .unwrap();

        let id = credit_id(&offer(&shoe, &sketch, 5000, true), adidas, kanye);
        ledger
            .apply_op(
                adidas,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&shoe, &sketch, 5000, true),
                },
                3,
            )
            .unwrap();
        assert_eq!(ledger.report(&id).unwrap().state, AttributionState::Offered);

        // Only Kanye can approve.
        let err = ledger
            .apply_op(adidas, &AttributionOpKind::Approve { credit_id: id }, 4)
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_NOT_AUTHOR);
        // ... and only while Offered.
        ledger
            .apply_op(kanye, &AttributionOpKind::Approve { credit_id: id }, 5)
            .unwrap();
        let r = ledger.report(&id).unwrap();
        assert_eq!(r.state, AttributionState::Approved);
        assert_eq!(r.approve_seq, Some(5));
        let err = ledger
            .apply_op(kanye, &AttributionOpKind::Approve { credit_id: id }, 6)
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_WRONG_STATE);

        // Public badge + per-view anonymous badge both mint deterministically.
        let viewer = [9u8; 32];
        let b_pub = badge_id(&id, None);
        let b_view = badge_id(&id, Some(viewer));
        ledger
            .apply_op(
                adidas,
                &AttributionOpKind::IssueBadge {
                    credit_id: id,
                    viewer: None,
                },
                7,
            )
            .unwrap();
        ledger
            .apply_op(
                adidas,
                &AttributionOpKind::IssueBadge {
                    credit_id: id,
                    viewer: Some(viewer),
                },
                8,
            )
            .unwrap();
        assert_eq!(ledger.report(&id).unwrap().badges, vec![b_pub, b_view]);
        // Same badge twice is refused.
        let err = ledger
            .apply_op(
                adidas,
                &AttributionOpKind::IssueBadge {
                    credit_id: id,
                    viewer: None,
                },
                9,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_BADGE_EXISTS);

        // Revoke: only Kanye, only while Approved; new badges refused after.
        let err = ledger
            .apply_op(adidas, &AttributionOpKind::Revoke { credit_id: id }, 10)
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_NOT_AUTHOR);
        ledger
            .apply_op(kanye, &AttributionOpKind::Revoke { credit_id: id }, 11)
            .unwrap();
        assert_eq!(ledger.report(&id).unwrap().state, AttributionState::Revoked);
        let err = ledger
            .apply_op(
                kanye,
                &AttributionOpKind::IssueBadge {
                    credit_id: id,
                    viewer: None,
                },
                12,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_BADGE_REVOKED);
    }

    #[test]
    fn offer_validations() {
        let mut ledger = AttributionLedger::new();
        let a = "did:unfer:a";
        let b = "did:unfer:b";
        let d = item("derived", 1);
        let o = item("original", 2);

        // Unregistered items are refused.
        let err = ledger
            .apply_op(
                a,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&d, &o, 100, false),
                },
                1,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_ITEM_UNKNOWN);

        ledger
            .apply_op(a, &AttributionOpKind::RegisterItem { item: d.clone() }, 1)
            .unwrap();
        ledger
            .apply_op(b, &AttributionOpKind::RegisterItem { item: o.clone() }, 2)
            .unwrap();

        // Self-attribution refused (A == B on the original).
        let err = ledger
            .apply_op(
                a,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&d, &d, 100, false),
                },
                3,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_SELF_ATTRIBUTION);

        // Zero fee refused.
        let err = ledger
            .apply_op(
                a,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&d, &o, 0, false),
                },
                3,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_FEE_ZERO);

        // Owner mismatch: B cannot offer A's derived item.
        let err = ledger
            .apply_op(
                b,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&d, &o, 100, false),
                },
                3,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_OWNER_MISMATCH);

        // Item collision: A registers an item already registered to C.
        let c = "did:unfer:c";
        ledger
            .apply_op(
                c,
                &AttributionOpKind::RegisterItem {
                    item: item("mine", 5),
                },
                4,
            )
            .unwrap();
        let err = ledger
            .apply_op(
                a,
                &AttributionOpKind::RegisterItem {
                    item: item("mine", 5),
                },
                5,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_ITEM_EXISTS);
        // Same owner re-registration is idempotent.
        ledger
            .apply_op(
                c,
                &AttributionOpKind::RegisterItem {
                    item: item("mine", 5),
                },
                6,
            )
            .unwrap();
    }

    #[test]
    fn exclusivity_blocks_concurrent_offers() {
        let mut ledger = AttributionLedger::new();
        let adidas = "did:unfer:adidas";
        let puma = "did:unfer:puma";
        let kanye = "did:unfer:kanye";
        let shoe = item("Yeezy Boost 350 V2", 1);
        let sketch = item("Kanye's 2015 sketch", 2);
        let puma_shoe = item("Puma's derived sneaker", 3);

        for (who, it) in [
            (adidas, shoe.clone()),
            (puma, puma_shoe.clone()),
            (kanye, sketch.clone()),
        ] {
            ledger
                .apply_op(who, &AttributionOpKind::RegisterItem { item: it }, 1)
                .unwrap();
        }
        // Adidas takes an exclusive credit on the sketch.
        ledger
            .apply_op(
                adidas,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&shoe, &sketch, 5000, true),
                },
                2,
            )
            .unwrap();
        // Puma's offer against the same original is refused while it is live.
        let err = ledger
            .apply_op(
                puma,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&puma_shoe, &sketch, 4000, true),
                },
                3,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_CREDIT_EXISTS);
        // ... and even after Kanye approves Adidas' credit.
        let id = credit_id(&offer(&shoe, &sketch, 5000, true), adidas, kanye);
        ledger
            .apply_op(kanye, &AttributionOpKind::Approve { credit_id: id }, 4)
            .unwrap();
        let err = ledger
            .apply_op(
                puma,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&puma_shoe, &sketch, 4000, true),
                },
                5,
            )
            .unwrap_err();
        assert_eq!(err.code, Code::ATTRIBUTION_CREDIT_EXISTS);
        // Revocation opens the original item again.
        ledger
            .apply_op(kanye, &AttributionOpKind::Revoke { credit_id: id }, 6)
            .unwrap();
        ledger
            .apply_op(
                puma,
                &AttributionOpKind::OfferAttribution {
                    offer: offer(&puma_shoe, &sketch, 4000, true),
                },
                7,
            )
            .unwrap();
    }

    #[test]
    fn same_log_same_root_different_log_different_root() {
        let run = |exclusive: bool| -> [u8; 32] {
            let mut ledger = AttributionLedger::new();
            let a = "did:unfer:a";
            let b = "did:unfer:b";
            let d = item("derived", 1);
            let o = item("original", 2);
            ledger
                .apply_op(a, &AttributionOpKind::RegisterItem { item: d.clone() }, 1)
                .unwrap();
            ledger
                .apply_op(b, &AttributionOpKind::RegisterItem { item: o.clone() }, 2)
                .unwrap();
            let id = credit_id(&offer(&d, &o, 100, exclusive), a, b);
            ledger
                .apply_op(
                    a,
                    &AttributionOpKind::OfferAttribution {
                        offer: offer(&d, &o, 100, exclusive),
                    },
                    3,
                )
                .unwrap();
            ledger
                .apply_op(b, &AttributionOpKind::Approve { credit_id: id }, 4)
                .unwrap();
            ledger
                .apply_op(
                    a,
                    &AttributionOpKind::IssueBadge {
                        credit_id: id,
                        viewer: None,
                    },
                    5,
                )
                .unwrap();
            ledger.root()
        };
        // Replay of the identical log converges; a different term diverges.
        assert_eq!(run(true), run(true));
        assert_ne!(run(true), run(false));
    }
}
