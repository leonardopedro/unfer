//! Denomination bookkeeping (GNU Taler denomination keys, simplified).
//!
//! A real Taler denomination pairs a face value with a blind-signing RSA key
//! and a validity window. Here the "key" is the face value plus an expiry
//! point on the consensus sequence; coins stop being withdrawable and become
//! refreshable once the exchange has moved past the expiry.

use std::collections::BTreeMap;

/// A face value the treasury will mint e-coins for, live until `expires_seq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denomination {
    pub value: u64,
    /// Last consensus seq at which this denomination may be withdrawn (inclusive);
    /// `u64::MAX` means "never expires".
    pub expires_seq: u64,
}

/// The exchange's offer book, keyed by face value.
#[derive(Debug, Clone, Default)]
pub struct DenominationBook {
    entries: BTreeMap<u64, Denomination>,
}

impl DenominationBook {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Register (or replace) the denomination for `value`.
    pub fn issue(&mut self, value: u64, expires_seq: u64) {
        self.entries.insert(value, Denomination { value, expires_seq });
    }

    /// Return the live denomination for `value` at `current_seq`.
    pub fn find(&self, value: u64, current_seq: u64) -> Option<&Denomination> {
        self.entries
            .get(&value)
            .filter(|d| d.expires_seq >= current_seq)
    }

    /// True if a denomination for `value` exists but has already expired.
    pub fn expired(&self, value: u64, current_seq: u64) -> bool {
        self.entries
            .get(&value)
            .map(|d| d.expires_seq < current_seq)
            .unwrap_or(false)
    }

    /// Every registered value (including expired ones).
    pub fn values(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_then_find_only_live() {
        let mut book = DenominationBook::new();
        book.issue(500, 100);
        assert_eq!(book.find(500, 1).unwrap().value, 500);
        assert!(book.find(500, 100).is_some());
        assert!(book.find(500, 101).is_none(), "expired past expiry");
        assert!(book.expired(500, 101));
        assert!(book.find(999, 1).is_none(), "never issued");
        assert!(!book.expired(999, 1));
    }

    #[test]
    fn never_expiring_denomination() {
        let mut book = DenominationBook::new();
        book.issue(25, u64::MAX);
        assert!(book.find(25, u64::MAX).is_some());
        assert!(book.find(25, 1_000_000).is_some());
    }

    #[test]
    fn issue_overwrites() {
        let mut book = DenominationBook::new();
        book.issue(100, 10);
        book.issue(100, u64::MAX);
        assert_eq!(book.values().collect::<Vec<_>>(), vec![100]);
        assert!(book.find(100, 10_000).is_some());
    }
}