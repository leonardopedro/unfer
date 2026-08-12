//! Data plane for the unfer federation: chunking, encryption, magnet URIs,
//! content publishing.
//!
//! [`crypto`] owns the X25519 keypairs and AES-GCM chunk encryption,
//! [`chunk`] the content-addressed chunker (`compute_cid`), [`magnet`] the
//! `magnet:` URI scheme, [`publisher`]/[`store`] the publish/persist layer,
//! [`release`] the byte-exact release manifest (S23/Golden), and
//! [`blueprint`] the encrypted `.cell` blueprint packaging (S27).

pub mod blueprint;
pub mod chunk;
pub mod crypto;
pub mod magnet;
pub mod publisher;
pub mod release;
pub mod store;

pub use blueprint::{CellRef, decrypt_stored_cell, encrypt_stored_cell, store_cell, verify_cell};
pub use chunk::{Chunker, compute_cid, verify_chunk};
pub use crypto::{DataKeypair, decrypt_chunk, encrypt_chunk};
pub use magnet::build_magnet_uri;
pub use publisher::DataPublisher;
pub use release::{ReleaseManifest, ReleaseStore};
pub use store::{CellEnvelope, CellEvent, CellStore, KeyRing, StoredCell};
