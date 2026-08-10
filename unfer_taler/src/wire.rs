//! Two-phase wire gateway (Taler `WireGateway`), the bank-facing seam.
//!
//! GNU Taler's exchange never touches the ledger at wire time: the *bank* moves
//! money and the gateway reports it. This module models that two-phase seam so
//! the exchange can be tested without a real banking backend:
//!
//! 1. `prepare` — the exchange asks the gateway to book a transfer (peg-out).
//! 2. `confirm` — the bank backend reports the transfer settled (peg-in: a
//!    customer's fiat has arrived; peg-out: the merchant's wire left).
//!
//! A peg-in MUST only credit a reserve from a `Confirmed` wire (UK-7103).

use std::collections::HashMap;

/// Lifecycle of a wire transfer as seen by the exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireStatus {
    /// Booked by the gateway; the bank has not settled it yet.
    Preparing,
    /// Settled. Only this state may fund a reserve (peg-in) or release a
    /// peg-out commitment.
    Confirmed,
}

/// A bank transfer id + amount the exchange bookkeeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireRef {
    /// Opaque bank-side reference (hex).
    pub wire_id: String,
    /// Destination account (IBAN-like string in the real system).
    pub to: String,
    pub amount: u64,
    pub status: WireStatus,
}

/// The bank-facing seam. `prepare`/`confirm` are synchronous here; a live
/// backend would poll the bank and flip status asynchronously.
pub trait WireGateway {
    fn prepare_transfer(&mut self, to: &str, amount: u64) -> Result<WireRef, String>;
    fn confirm(&mut self, wire_id: &str) -> Result<WireStatus, String>;
    fn get(&self, wire_id: &str) -> Option<&WireRef>;
}

/// In-memory gateway for tests and the demo flow. Wire ids are monotonic
/// (`taler-wire:<n>`); `confirm` settles any prepared transfer.
#[derive(Debug, Default)]
pub struct SimulatedWireGateway {
    wires: HashMap<String, WireRef>,
    next_id: u64,
}

impl SimulatedWireGateway {
    pub fn new() -> Self {
        Self::default()
    }

    /// The full record, mirroring the bank statement.
    pub fn wires(&self) -> &HashMap<String, WireRef> {
        &self.wires
    }
}

impl WireGateway for SimulatedWireGateway {
    fn prepare_transfer(&mut self, to: &str, amount: u64) -> Result<WireRef, String> {
        if amount == 0 {
            return Err("zero-value wires are not honored".to_string());
        }
        self.next_id += 1;
        let w = WireRef {
            wire_id: format!("taler-wire:{:08x}", self.next_id),
            to: to.to_string(),
            amount,
            status: WireStatus::Preparing,
        };
        self.wires.insert(w.wire_id.clone(), w.clone());
        Ok(w)
    }

    fn confirm(&mut self, wire_id: &str) -> Result<WireStatus, String> {
        let w = self.wires.get_mut(wire_id).ok_or("unknown wire transfer")?;
        w.status = WireStatus::Confirmed;
        Ok(w.status)
    }

    fn get(&self, wire_id: &str) -> Option<&WireRef> {
        self.wires.get(wire_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_phase_lifecycle() {
        let mut gw = SimulatedWireGateway::new();
        let w = gw
            .prepare_transfer("DE99 0000 0000 1234 5678 90", 1000)
            .unwrap();
        assert_eq!(w.status, WireStatus::Preparing);
        assert_eq!(w.amount, 1000);
        assert_eq!(gw.confirm(&w.wire_id).unwrap(), WireStatus::Confirmed);
        assert_eq!(gw.get(&w.wire_id).unwrap().status, WireStatus::Confirmed);
    }

    #[test]
    fn zero_value_wire_refused() {
        let mut gw = SimulatedWireGateway::new();
        assert!(gw.prepare_transfer("bank", 0).is_err());
    }

    #[test]
    fn confirm_unknown_wire_fails() {
        let mut gw = SimulatedWireGateway::new();
        assert!(gw.confirm("nope").is_err());
    }
}
