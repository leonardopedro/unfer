//! Deterministic unified-auction engine (Prebid-model open auction).
//!
//! This is the state-transition engine every QuePaxa node runs when it applies
//! an [`AuctionOp`] from the consensus log — the marketplace analogue of the
//! [`CertificateLedger`](crate::certs::CertificateLedger). It is fully
//! deterministic: a node checks an op (lot existence, floor, seller, self-bid,
//! quantity) and only then lets it into state, and the unified-auction clearing
//! rule — highest `price_per_unit` wins, ties break to the earliest `seq` —
//! is a pure function of the recorded bids. Every node that replays the same
//! log converges on the identical winner.
//!
//! The engine supports two markets with the same mechanism (Prebid's "unified
//! auction" model):
//!
//! - **Carbon credits** ([`AuctionAsset::CarbonCredits`]): a seller offers a
//!   lot of credits at a floor price; the winner's payment settles through the
//!   certificate ledger (Taler e-coin or credits) and the seller's certificate
//!   for the lot transfers to the winner.
//! - **Publicity inventory** ([`AuctionAsset::PublicitySlot`]): a publisher
//!   offers an ad slot as an AdSense alternative; the winner pays the publisher
//!   through the same rails. There is no ledger asset to deliver.
//!
//! The engine only decides the winner. Escrow of payment and delivery of the
//! credit certificate are ordinary [`CertificateOp::Transfer`]s rowed by the
//! marketplace operator (see `unfer_taler::auction::AuctionService`), exactly
//! like the Phase-4 [`EscrowService`](crate::escrow::EscrowService).

use std::collections::HashMap;

use unfer_protocol::{
    AuctionBid, AuctionId, AuctionLot, AuctionOpKind, AuctionReport, AuctionWinner, Code,
    Diagnostic, Severity,
};

/// One lot on the ledger: the lot description plus everything the clearing
/// rule needs (the recorded bids and the closed flag).
#[derive(Debug, Clone)]
pub struct LotState {
    pub lot: AuctionLot,
    pub bids: Vec<AuctionBid>,
    pub closed: bool,
    pub winner: Option<AuctionWinner>,
}

/// The deterministic auction state-transition engine.
#[derive(Debug, Default)]
pub struct AuctionLedger {
    lots: HashMap<[u8; 32], LotState>,
}

impl AuctionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lot(&self, lot_id: &AuctionId) -> Option<&LotState> {
        self.lots.get(&lot_id.0)
    }

    pub fn report(&self, lot_id: &AuctionId) -> Option<AuctionReport> {
        let s = self.lots.get(&lot_id.0)?;
        Some(AuctionReport {
            lot: s.lot.clone(),
            bids: s.bids.clone(),
            closed: s.closed,
            winner: s.winner.clone(),
        })
    }

    pub fn open_lots(&self) -> Vec<AuctionReport> {
        self.lots
            .values()
            .filter(|s| !s.closed)
            .map(|s| AuctionReport {
                lot: s.lot.clone(),
                bids: s.bids.clone(),
                closed: false,
                winner: None,
            })
            .collect()
    }

    /// Dispatch a signed auction op against the ledger. `actor` is the signer's
    /// DID (from the op), already verified by the caller. `seq` is the op's
    /// consensus-log sequence, which the clearing rule uses to break ties.
    ///
    /// The returned winner is the deterministic clearing outcome:
    /// - `Open` → `Ok(None)` (the lot is on the ledger),
    /// - `Bid` → `Ok(None)` (the bid is recorded),
    /// - `Close` → `Ok(Some(winner))` when the unified clearing selects one, or
    ///   `Ok(None)` when the lot closes with no eligible bids. A close always
    ///   closes the lot.
    pub fn apply_op(
        &mut self,
        actor: &str,
        kind: &AuctionOpKind,
        seq: u64,
    ) -> Result<Option<AuctionWinner>, Diagnostic> {
        match kind {
            AuctionOpKind::Open { lot } => {
                self.apply_open(actor, lot)?;
                Ok(None)
            }
            AuctionOpKind::Bid {
                lot_id,
                price_per_unit,
                quantity,
            } => {
                self.apply_bid(actor, lot_id, *price_per_unit, *quantity, seq)?;
                Ok(None)
            }
            AuctionOpKind::Close { lot_id } => self.apply_close(actor, lot_id),
        }
    }

    /// Open a lot. Only the lot's seller may; the lot_id must not already exist.
    fn apply_open(&mut self, actor: &str, lot: &AuctionLot) -> Result<(), Diagnostic> {
        if lot.seller_did != actor {
            return Err(Diagnostic::new(
                Code::AUCTION_NOT_SELLER,
                "only the lot's seller may open it",
                Severity::Error,
            ));
        }
        if self.lots.contains_key(&lot.lot_id.0) {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_EXISTS,
                "lot already exists on the ledger",
                Severity::Error,
            ));
        }
        if lot.floor == 0 {
            return Err(Diagnostic::new(
                Code::AUCTION_BID_BELOW_FLOOR,
                "lot floor must be positive",
                Severity::Error,
            ));
        }
        self.lots.insert(
            lot.lot_id.0,
            LotState {
                lot: lot.clone(),
                bids: Vec::new(),
                closed: false,
                winner: None,
            },
        );
        Ok(())
    }

    /// Record a bid. Rejected if the lot is unknown/closed, the bid is below
    /// floor, the bidder is the seller, or the quantity exceeds the lot amount
    /// (carbon lots only).
    fn apply_bid(
        &mut self,
        actor: &str,
        lot_id: &AuctionId,
        price_per_unit: u64,
        quantity: u64,
        seq: u64,
    ) -> Result<(), Diagnostic> {
        let state = match self.lots.get_mut(&lot_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::AUCTION_UNKNOWN_LOT,
                    "unknown lot",
                    Severity::Error,
                ));
            }
        };
        if state.closed {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_CLOSED,
                "lot is already closed",
                Severity::Error,
            ));
        }
        if state.lot.seller_did == actor {
            return Err(Diagnostic::new(
                Code::AUCTION_SELF_BID,
                "the seller cannot bid on their own lot",
                Severity::Error,
            ));
        }
        if price_per_unit < state.lot.floor {
            return Err(Diagnostic::new(
                Code::AUCTION_BID_BELOW_FLOOR,
                format!("bid {price_per_unit} below the floor {}", state.lot.floor),
                Severity::Error,
            ));
        }
        if quantity == 0 {
            return Err(Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                "bid quantity must be positive",
                Severity::Error,
            ));
        }
        if let unfer_protocol::AuctionAsset::CarbonCredits { amount } = &state.lot.asset
            && quantity > *amount
        {
            return Err(Diagnostic::new(
                Code::AUCTION_QTY_MISMATCH,
                format!("bid quantity {quantity} exceeds lot amount {amount}"),
                Severity::Error,
            ));
        }
        state.bids.push(AuctionBid {
            lot_id: *lot_id,
            bidder_did: actor.to_string(),
            price_per_unit,
            quantity,
            seq,
        });
        Ok(())
    }

    /// Compute the deterministic close winner WITHOUT closing the lot
    /// (read-only; same validation as `apply_close`). This is the probe half
    /// of a close: the FFI serializes the winner and only commits the
    /// mutation (`apply_close`) when the caller's buffer can hold the full
    /// JSON, so a too-small buffer must not consume the op (the round-20
    /// retry-safe buffer discipline — the `uk_poll` class).
    pub fn close_winner(
        &self,
        actor: &str,
        lot_id: &AuctionId,
    ) -> Result<Option<AuctionWinner>, Diagnostic> {
        let state = match self.lots.get(&lot_id.0) {
            Some(s) => s,
            None => {
                return Err(Diagnostic::new(
                    Code::AUCTION_UNKNOWN_LOT,
                    "unknown lot",
                    Severity::Error,
                ));
            }
        };
        if state.lot.seller_did != actor {
            return Err(Diagnostic::new(
                Code::AUCTION_NOT_SELLER,
                "only the lot's seller may close it",
                Severity::Error,
            ));
        }
        if state.closed {
            return Err(Diagnostic::new(
                Code::AUCTION_LOT_CLOSED,
                "lot is already closed",
                Severity::Error,
            ));
        }
        Ok(state
            .bids
            .iter()
            .max_by(|a, b| {
                a.price_per_unit
                    .cmp(&b.price_per_unit)
                    .then_with(|| b.seq.cmp(&a.seq)) // tie → earliest seq
            })
            .cloned()
            .map(|winner| AuctionWinner {
                lot_id: *lot_id,
                bidder_did: winner.bidder_did,
                price_per_unit: winner.price_per_unit,
                quantity: winner.quantity,
                total: winner.price_per_unit.saturating_mul(winner.quantity),
            }))
    }

    /// Close a lot and compute the deterministic winner: highest `price_per_unit`
    /// wins, ties break to the earliest `seq`. The lot is always closed; no
    /// eligible bids yield `Ok(None)`. The winner computation is shared with
    /// the read-only `close_winner` (the commit half of the pair). Public so
    /// the FFI can commit a close under the same ledger lock hold as the
    /// size probe (`auction_close_json`).
    pub fn apply_close(
        &mut self,
        actor: &str,
        lot_id: &AuctionId,
    ) -> Result<Option<AuctionWinner>, Diagnostic> {
        let winner = self.close_winner(actor, lot_id)?;
        let state = self
            .lots
            .get_mut(&lot_id.0)
            .expect("close_winner just validated the lot exists");
        state.closed = true;
        state.winner = winner.clone();
        Ok(winner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::{AuctionAsset, AuctionCurrency};

    fn lot(seller: &str, floor: u64, id_byte: u8) -> AuctionLot {
        AuctionLot {
            lot_id: AuctionId([id_byte; 32]),
            seller_did: seller.to_string(),
            asset: AuctionAsset::CarbonCredits { amount: 1000 },
            currency: AuctionCurrency::Taler,
            floor,
            opens_seq: 1,
            closes_seq: 100,
        }
    }

    fn slot_lot(seller: &str, floor: u64, id_byte: u8) -> AuctionLot {
        AuctionLot {
            lot_id: AuctionId([id_byte; 32]),
            seller_did: seller.to_string(),
            asset: AuctionAsset::PublicitySlot {
                slot: "homepage_leaderboard_300x250".to_string(),
                description: None,
            },
            currency: AuctionCurrency::CarbonCredits,
            floor,
            opens_seq: 1,
            closes_seq: 100,
        }
    }

    #[test]
    fn unified_auction_picks_highest_price() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        let alice = "did:unfer:alice";
        let bob = "did:unfer:bob";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 5, 1),
            },
            1,
        )
        .unwrap();
        l.apply_op(
            alice,
            &AuctionOpKind::Bid {
                lot_id: AuctionId([1; 32]),
                price_per_unit: 7,
                quantity: 500,
            },
            2,
        )
        .unwrap();
        l.apply_op(
            bob,
            &AuctionOpKind::Bid {
                lot_id: AuctionId([1; 32]),
                price_per_unit: 9,
                quantity: 300,
            },
            3,
        )
        .unwrap();
        let winner = l
            .apply_op(
                seller,
                &AuctionOpKind::Close {
                    lot_id: AuctionId([1; 32]),
                },
                4,
            )
            .unwrap()
            .unwrap();
        assert_eq!(winner.bidder_did, bob);
        assert_eq!(winner.price_per_unit, 9);
        assert_eq!(winner.quantity, 300);
        assert_eq!(winner.total, 2700);
        assert_eq!(
            l.report(&AuctionId([1; 32]))
                .unwrap()
                .winner
                .unwrap()
                .bidder_did,
            bob
        );
    }

    #[test]
    fn tie_breaks_to_earliest_seq() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 5, 2),
            },
            1,
        )
        .unwrap();
        l.apply_op(
            "did:unfer:late",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([2; 32]),
                price_per_unit: 8,
                quantity: 100,
            },
            3,
        )
        .unwrap();
        l.apply_op(
            "did:unfer:early",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([2; 32]),
                price_per_unit: 8,
                quantity: 200,
            },
            2,
        )
        .unwrap();
        let winner = l
            .apply_op(
                seller,
                &AuctionOpKind::Close {
                    lot_id: AuctionId([2; 32]),
                },
                4,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            winner.bidder_did, "did:unfer:early",
            "earliest seq wins the tie"
        );
    }

    #[test]
    fn floor_rejects_low_bids_and_unknown_lot_rejects_bids() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 10, 3),
            },
            1,
        )
        .unwrap();
        let low = l.apply_op(
            "did:unfer:low",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([3; 32]),
                price_per_unit: 5,
                quantity: 1,
            },
            2,
        );
        assert_eq!(low.unwrap_err().code, Code::AUCTION_BID_BELOW_FLOOR);
        let ghost = l.apply_op(
            "did:unfer:ghost",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([99; 32]),
                price_per_unit: 100,
                quantity: 1,
            },
            2,
        );
        assert_eq!(ghost.unwrap_err().code, Code::AUCTION_UNKNOWN_LOT);
    }

    #[test]
    fn self_bid_and_non_seller_are_rejected() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 1, 4),
            },
            1,
        )
        .unwrap();
        let self_bid = l.apply_op(
            seller,
            &AuctionOpKind::Bid {
                lot_id: AuctionId([4; 32]),
                price_per_unit: 50,
                quantity: 1,
            },
            2,
        );
        assert_eq!(self_bid.unwrap_err().code, Code::AUCTION_SELF_BID);
        let intruder = l.apply_op(
            "did:unfer:evil",
            &AuctionOpKind::Close {
                lot_id: AuctionId([4; 32]),
            },
            3,
        );
        assert_eq!(intruder.unwrap_err().code, Code::AUCTION_NOT_SELLER);
        let dup = l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 1, 4),
            },
            3,
        );
        assert_eq!(dup.unwrap_err().code, Code::AUCTION_LOT_EXISTS);
    }

    #[test]
    fn quantity_limited_to_carbon_lot_amount() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 1, 5),
            },
            1,
        )
        .unwrap();
        let over = l.apply_op(
            "did:unfer:big",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([5; 32]),
                price_per_unit: 5,
                quantity: 1001,
            },
            2,
        );
        assert_eq!(over.unwrap_err().code, Code::AUCTION_QTY_MISMATCH);
        // Publicity slots are not quantity-limited against a lot amount.
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: slot_lot(seller, 1, 6),
            },
            1,
        )
        .unwrap();
        l.apply_op(
            "did:unfer:publisher-buyer",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([6; 32]),
                price_per_unit: 5,
                quantity: 100_000,
            },
            2,
        )
        .unwrap();
    }

    #[test]
    fn no_bids_close_closes_the_lot_without_a_winner() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 5, 7),
            },
            1,
        )
        .unwrap();
        let outcome = l.apply_op(
            seller,
            &AuctionOpKind::Close {
                lot_id: AuctionId([7; 32]),
            },
            2,
        );
        assert!(outcome.unwrap().is_none(), "no bids → no winner");
        let report = l.report(&AuctionId([7; 32])).unwrap();
        assert!(report.closed);
        assert!(report.winner.is_none());
    }

    #[test]
    fn double_close_and_reopen_are_rejected() {
        let mut l = AuctionLedger::new();
        let seller = "did:unfer:seller";
        l.apply_op(
            seller,
            &AuctionOpKind::Open {
                lot: lot(seller, 1, 8),
            },
            1,
        )
        .unwrap();
        l.apply_op(
            "did:unfer:x",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([8; 32]),
                price_per_unit: 2,
                quantity: 10,
            },
            2,
        )
        .unwrap();
        l.apply_op(
            seller,
            &AuctionOpKind::Close {
                lot_id: AuctionId([8; 32]),
            },
            3,
        )
        .unwrap();
        let again = l.apply_op(
            seller,
            &AuctionOpKind::Close {
                lot_id: AuctionId([8; 32]),
            },
            4,
        );
        assert_eq!(again.unwrap_err().code, Code::AUCTION_LOT_CLOSED);
        let rebid = l.apply_op(
            "did:unfer:y",
            &AuctionOpKind::Bid {
                lot_id: AuctionId([8; 32]),
                price_per_unit: 9,
                quantity: 5,
            },
            5,
        );
        assert_eq!(rebid.unwrap_err().code, Code::AUCTION_LOT_CLOSED);
    }

    #[test]
    fn replay_is_deterministic_across_instances() {
        let seller = "did:unfer:seller";
        let ops = vec![
            AuctionOpKind::Open {
                lot: lot(seller, 3, 9),
            },
            AuctionOpKind::Bid {
                lot_id: AuctionId([9; 32]),
                price_per_unit: 4,
                quantity: 100,
            },
            AuctionOpKind::Bid {
                lot_id: AuctionId([9; 32]),
                price_per_unit: 6,
                quantity: 200,
            },
            AuctionOpKind::Bid {
                lot_id: AuctionId([9; 32]),
                price_per_unit: 6,
                quantity: 50,
            },
            AuctionOpKind::Close {
                lot_id: AuctionId([9; 32]),
            },
        ];
        let mut a = AuctionLedger::new();
        let mut b = AuctionLedger::new();
        for (i, op) in ops.iter().enumerate() {
            let (did, op) = match op {
                AuctionOpKind::Open { lot } => (seller, AuctionOpKind::Open { lot: lot.clone() }),
                AuctionOpKind::Bid {
                    lot_id,
                    price_per_unit,
                    quantity,
                } => {
                    let bidder = if i == 1 { "did:unfer:a" } else { "did:unfer:b" };
                    (
                        bidder,
                        AuctionOpKind::Bid {
                            lot_id: *lot_id,
                            price_per_unit: *price_per_unit,
                            quantity: *quantity,
                        },
                    )
                }
                AuctionOpKind::Close { lot_id } => {
                    (seller, AuctionOpKind::Close { lot_id: *lot_id })
                }
            };
            let wa = a
                .apply_op(did, &op, i as u64)
                .unwrap()
                .map(|w| w.bidder_did);
            let wb = b
                .apply_op(did, &op, i as u64)
                .unwrap()
                .map(|w| w.bidder_did);
            assert_eq!(wa, wb);
        }
        assert_eq!(
            a.report(&AuctionId([9; 32]))
                .unwrap()
                .winner
                .unwrap()
                .bidder_did,
            b.report(&AuctionId([9; 32]))
                .unwrap()
                .winner
                .unwrap()
                .bidder_did
        );
    }
}
