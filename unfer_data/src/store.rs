//! Cell retention (pin/GC) + key rotation on the content plane (S15, F14).
//!
//! Two pieces of lifecycle management the content plane previously lacked:
//!
//! 1. [`CellStore`] — a ref-counted registry of stored cells. `store` begins the cell
//!    **owned** (pin = 1); `pin`/`unpin` adjust the count; `prune` removes any cell with
//!    zero pins and records a `cell_pruned`-style [`CellEvent`] for the audit stream.
//! 2. [`KeyRing`] — deterministic epoch-chained cell encryption. `rotate` moves the
//!    writing epoch forward (deriving the next key from the previous one), so old
//!    ciphertexts remain readable within a bounded retention window and are refused once
//!    they age out. The envelope records the epoch that encrypted it.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::blueprint::{store_cell, CellRef};
use crate::chunk::Chunker;
use crate::crypto::{decrypt_chunk, encrypt_chunk};

/// Domain separator for epoch-key derivation.
const EPOCH_DOMAIN: &[u8] = b"unfer-cell-epoch-v1\0";

/// An AES key per writing epoch; deterministic chain off a root.
pub struct KeyRing {
    root: [u8; 32],
    current: u64,
    retain_depth: u64,
}

/// Ciphertext plus the epoch that encrypted it.
#[derive(Debug, Clone, PartialEq)]
pub struct CellEnvelope {
    pub epoch: u64,
    pub chunks: Vec<Vec<u8>>,
}

impl KeyRing {
    /// A fresh key ring seeded from random bytes.
    pub fn from_random() -> Self {
        Self::new(rand::random())
    }

    /// A key ring rooted at `root`; the first epoch is 0.
    pub fn new(root: [u8; 32]) -> Self {
        Self {
            root,
            current: 0,
            retain_depth: 8,
        }
    }

    /// The current writing epoch.
    pub fn current(&self) -> u64 {
        self.current
    }

    /// Advance to the next epoch (new key chain), returning it.
    pub fn rotate(&mut self) -> u64 {
        self.current += 1;
        self.current
    }

    /// Derive the key for `epoch`, if it is still within the retention window.
    pub fn epoch_key(&self, epoch: u64) -> Option<[u8; 32]> {
        if epoch > self.current || self.current.saturating_sub(epoch) >= self.retain_depth {
            return None;
        }
        let mut key = self.root;
        for i in 1..=epoch {
            key = Sha256::digest([&key[..], EPOCH_DOMAIN, &i.to_le_bytes()[..]].concat()).into();
        }
        Some(key)
    }

    /// Encrypt `cell` under the current epoch.
    pub fn encrypt_envelope(&self, cell: &[u8]) -> Result<CellEnvelope, String> {
        let key = self
            .epoch_key(self.current)
            .ok_or("current epoch key unavailable")?;
        let mut chunks = Vec::new();
        for (idx, plain) in Chunker::default().chunk(cell) {
            chunks.push(encrypt_chunk(&key, idx, &plain)?);
        }
        Ok(CellEnvelope {
            epoch: self.current,
            chunks,
        })
    }

    /// Decrypt an envelope from any still-retained epoch.
    pub fn decrypt_envelope(&self, env: &CellEnvelope) -> Result<Vec<u8>, String> {
        let key = self
            .epoch_key(env.epoch)
            .ok_or(format!("cell epoch {} retired beyond retention", env.epoch))?;
        let mut out = Vec::new();
        for (i, ct) in env.chunks.iter().enumerate() {
            out.extend_from_slice(&decrypt_chunk(&key, i as u32, ct)?);
        }
        Ok(out)
    }
}

/// Lifecycle event for the audit stream.
#[derive(Debug, Clone, PartialEq)]
pub enum CellEvent {
    Stored {
        cid: String,
        seq: u64,
    },
    Pinned {
        cid: String,
        pins: u64,
    },
    Unpinned {
        cid: String,
        pins: u64,
    },
    Pruned {
        cid: String,
        seq: u64,
    },
}

#[derive(Debug)]
pub struct StoredCell {
    pub cell_ref: CellRef,
    pub pins: u64,
    pub seq: u64,
}

/// In-memory ref-counted content registry.
#[derive(Default)]
pub struct CellStore {
    cells: HashMap<String, StoredCell>,
    events: Vec<CellEvent>,
    next_seq: u64,
}

impl CellStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Address `cell` into the store (owned: pin = 1).
    pub fn store(&mut self, cell: &[u8]) -> CellRef {
        let cell_ref = store_cell(cell);
        self.next_seq += 1;
        self.cells.insert(
            cell_ref.cid.clone(),
            StoredCell {
                cell_ref: cell_ref.clone(),
                pins: 1,
                seq: self.next_seq,
            },
        );
        self.events
            .push(CellEvent::Stored { cid: cell_ref.cid.clone(), seq: self.next_seq });
        cell_ref
    }

    pub fn get(&self, cid: &str) -> Option<&StoredCell> {
        self.cells.get(cid)
    }

    /// Raise the pin count. Unknown CIDs are refused (no store-and-pray).
    pub fn pin(&mut self, cid: &str) -> Result<u64, String> {
        let rec = self
            .cells
            .get_mut(cid)
            .ok_or_else(|| format!("cell {cid} not in store"))?;
        rec.pins += 1;
        self.events.push(CellEvent::Pinned {
            cid: cid.to_string(),
            pins: rec.pins,
        });
        Ok(rec.pins)
    }

    /// Lower the pin count (floor 0). Unknown CIDs are refused.
    pub fn unpin(&mut self, cid: &str) -> Result<u64, String> {
        let rec = self
            .cells
            .get_mut(cid)
            .ok_or_else(|| format!("cell {cid} not in store"))?;
        rec.pins = rec.pins.saturating_sub(1);
        self.events.push(CellEvent::Unpinned {
            cid: cid.to_string(),
            pins: rec.pins,
        });
        Ok(rec.pins)
    }

    /// Drop every cell with zero pins; returns the pruning events.
    pub fn prune(&mut self) -> Vec<CellEvent> {
        let doomed: Vec<String> = self
            .cells
            .iter()
            .filter(|(_, rec)| rec.pins == 0)
            .map(|(cid, _)| cid.clone())
            .collect();
        let mut out = Vec::new();
        for cid in doomed {
            self.cells.remove(&cid);
            let ev = CellEvent::Pruned {
                cid,
                seq: self.next_seq,
            };
            self.next_seq += 1;
            self.events.push(ev.clone());
            out.push(ev);
        }
        out
    }

    pub fn events(&self) -> &[CellEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn cids(&self) -> Vec<String> {
        self.cells.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::store_cell;

    fn sample_cell(tag: u8) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[0] = tag;
        bytes
    }

    #[test]
    fn store_pin_unpin_prune_keeps_only_pinned() {
        let mut store = CellStore::new();
        let a = store.store(&sample_cell(1));
        let b = store.store(&sample_cell(2));

        // A is unpinned (pins 1 → 0); B keeps its ownership pin.
        assert_eq!(store.unpin(&a.cid).unwrap(), 0);
        assert_eq!(store.pin(&b.cid).unwrap(), 2);

        let pruned = store.prune();
        assert_eq!(pruned.len(), 1);
        assert!(matches!(&pruned[0], CellEvent::Pruned { cid, .. } if cid == &a.cid));
        assert_eq!(store.len(), 1);
        assert!(store.get(&a.cid).is_none());
        assert!(store.get(&b.cid).is_some());
    }

    #[test]
    fn pin_unknown_cid_errors() {
        let mut store = CellStore::new();
        assert!(store.pin("deadbeef").is_err());
        assert!(store.unpin("deadbeef").is_err());
        assert!(store.prune().is_empty());
    }

    #[test]
    fn store_records_prune_event_with_seq() {
        let mut store = CellStore::new();
        let a = store.store(&sample_cell(3));
        store.unpin(&a.cid).unwrap();
        store.prune();
        let events = store.events().to_vec();
        assert!(events.iter().any(|e| matches!(e, CellEvent::Stored { cid, .. } if cid == &a.cid)));
        assert!(events
            .iter()
            .any(|e| matches!(e, CellEvent::Unpinned { cid, pins, .. } if cid == &a.cid && *pins == 0)));
        assert!(matches!(&events[events.len() - 1], CellEvent::Pruned { cid, .. } if cid == &a.cid));
    }

    #[test]
    fn keyring_roundtrip_and_content_address_preserved() {
        let ring = KeyRing::from_random();
        let cell = sample_cell(9);
        let env = ring.encrypt_envelope(&cell).unwrap();
        let plain = ring.decrypt_envelope(&env).unwrap();
        assert_eq!(plain, cell);
        assert_eq!(store_cell(&plain).cid, store_cell(&cell).cid);
    }

    #[test]
    fn keyring_rotate_reads_within_retention() {
        let mut ring = KeyRing::from_random();
        let cell = sample_cell(7);

        let e0 = ring.encrypt_envelope(&cell).unwrap();
        assert_eq!(e0.epoch, 0);

        let rotated = ring.rotate();
        assert_eq!(rotated, 1);
        let e1 = ring.encrypt_envelope(&cell).unwrap();
        assert_eq!(e1.epoch, 1);

        // New ciphertext under a different key; both still decrypt.
        assert_ne!(e1.chunks, e0.chunks);
        assert_eq!(ring.decrypt_envelope(&e0).unwrap(), cell);
        assert_eq!(ring.decrypt_envelope(&e1).unwrap(), cell);
    }

    #[test]
    fn keyring_retires_old_epoch_past_retention() {
        let mut ring = KeyRing::new([7u8; 32]);
        ring.retain_depth = 2;
        let cell = sample_cell(5);
        let e0 = ring.encrypt_envelope(&cell).unwrap();

        ring.rotate();
        ring.rotate(); // current = 2; retention window keeps epochs 1..=2
        assert!(ring.epoch_key(0).is_none());
        assert!(ring.decrypt_envelope(&e0).is_err());
    }
}
