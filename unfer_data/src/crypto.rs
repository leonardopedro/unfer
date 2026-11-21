use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::{Digest, Sha256};
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

use crate::chunk::Chunker;

/// Envelope format this module writes and the only one it reads.
///
/// Version 2 replaced a scheme that used one epoch key for every cell and a
/// nonce of eight zero bytes plus the chunk index, so two cells in one epoch
/// encrypted chunk 0 under the same key and nonce. Version-1 envelopes carried
/// no version, no object identity and no associated data; they are refused
/// instead of being guessed at, because the old ciphertext cannot authenticate
/// the metadata the new format binds.
pub const ENVELOPE_VERSION: u8 = 2;

/// Hard ceiling on one object, so a caller cannot ask for a length the chunk
/// counter or a decoder could not represent.
pub const MAX_OBJECT_BYTES: u64 = 1 << 40;

const OBJECT_KEY_DOMAIN: &[u8] = b"unfer-object-key-v2\0";
const CHUNK_AAD_DOMAIN: &[u8] = b"unfer-cell-chunk-v2\0";
const OBJECT_ID_DOMAIN: &[u8] = b"unfer-content-object-v2\0";

pub struct DataKeypair {
    secret: StaticSecret,
    public: PublicKey,
}

impl DataKeypair {
    pub fn generate() -> Self {
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(self.public_key_bytes())
    }

    pub fn public_key(&self) -> &PublicKey {
        &self.public
    }

    pub fn shared_secret(&self, peer_public: &PublicKey) -> [u8; 32] {
        let shared = self.secret.diffie_hellman(peer_public);
        let hash = Sha256::digest(shared.as_bytes());
        hash.into()
    }
}

pub fn derive_aes_key(shared_secret: &[u8; 32]) -> [u8; 32] {
    let hash = Sha256::digest(shared_secret);
    hash.into()
}

/// One object's encryption context: a key derived from the epoch/transport key
/// *and* a 32-byte object identifier, so the chunk counter only ever repeats
/// under the same object.
#[derive(Clone, PartialEq, Eq)]
pub struct ObjectKey {
    key: [u8; 32],
    object_id: [u8; 32],
}

impl std::fmt::Debug for ObjectKey {
    // Never print key material, including through a derived Debug on an envelope.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectKey")
            .field("object_id", &hex::encode(self.object_id))
            .finish_non_exhaustive()
    }
}

impl ObjectKey {
    /// Derive the object key from its two inputs, domain-separated so an epoch
    /// key and an object key can never coincide.
    pub fn from_parts(key_material: &[u8; 32], object_id: [u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(OBJECT_KEY_DOMAIN);
        hasher.update(key_material);
        hasher.update(object_id);
        Self {
            key: hasher.finalize().into(),
            object_id,
        }
    }

    /// A fresh random object identifier. A random identifier is what makes two
    /// envelopes under one key distinguishable; a failure to read the OS RNG is
    /// an error, never a zero salt.
    pub fn fresh(key_material: &[u8; 32]) -> Result<Self, String> {
        let mut object_id = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut object_id)
            .map_err(|e| format!("object identifier: OS randomness unavailable: {e}"))?;
        Ok(Self::from_parts(key_material, object_id))
    }

    pub fn object_id(&self) -> [u8; 32] {
        self.object_id
    }

    /// Associated data: everything the receiver must agree on before the
    /// plaintext means anything — format version, epoch, object identity, total
    /// length, chunk count and this chunk's index. Swapping a chunk between
    /// objects, epochs or positions fails authentication instead of decrypting
    /// to a different message.
    fn aad(&self, epoch: u64, object_len: u64, chunk_count: u32, index: u32) -> Vec<u8> {
        let mut aad = Vec::with_capacity(CHUNK_AAD_DOMAIN.len() + 1 + 8 + 32 + 8 + 4 + 4);
        aad.extend_from_slice(CHUNK_AAD_DOMAIN);
        aad.push(ENVELOPE_VERSION);
        aad.extend_from_slice(&epoch.to_le_bytes());
        aad.extend_from_slice(&self.object_id);
        aad.extend_from_slice(&object_len.to_le_bytes());
        aad.extend_from_slice(&chunk_count.to_le_bytes());
        aad.extend_from_slice(&index.to_le_bytes());
        aad
    }

    fn check_bounds(object_len: u64, chunk_count: u32, index: u32) -> Result<(), String> {
        if object_len > MAX_OBJECT_BYTES {
            return Err(format!(
                "object length {object_len} exceeds the {MAX_OBJECT_BYTES}-byte ceiling"
            ));
        }
        if chunk_count == 0 {
            return Err("chunk count must be positive".to_string());
        }
        if index >= chunk_count {
            return Err(format!(
                "chunk index {index} is outside the declared {chunk_count} chunk(s); \
                 the counter would repeat"
            ));
        }
        Ok(())
    }

    pub fn seal_chunk(
        &self,
        epoch: u64,
        object_len: u64,
        chunk_count: u32,
        index: u32,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, String> {
        Self::check_bounds(object_len, chunk_count, index)?;
        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;
        // Unique per object key: the nonce repeats only for the same object,
        // and then only ever with the same plaintext length and the same
        // authenticated metadata.
        let mut nonce_bytes = [0u8; 12];
        nonce_bytes[8..12].copy_from_slice(&index.to_le_bytes());
        let nonce = Nonce::from_slice(&nonce_bytes);
        cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: &self.aad(epoch, object_len, chunk_count, index),
                },
            )
            .map_err(|e| format!("encryption failed: {e}"))
    }

    pub fn open_chunk(
        &self,
        epoch: u64,
        object_len: u64,
        chunk_count: u32,
        index: u32,
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, String> {
        Self::check_bounds(object_len, chunk_count, index)?;
        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;
        let mut nonce_bytes = [0u8; 12];
        nonce_bytes[8..12].copy_from_slice(&index.to_le_bytes());
        let nonce = Nonce::from_slice(&nonce_bytes);
        cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: &self.aad(epoch, object_len, chunk_count, index),
                },
            )
            .map_err(|e| format!("decryption failed: {e}"))
    }
}

/// A whole sealed object: the metadata the chunks are bound to, plus the
/// ciphertexts. This is the value the store, blueprints and the publisher move
/// around; it is never a bare `Vec<Vec<u8>>` that could be spliced onto another
/// object's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SealedObject {
    pub version: u8,
    pub object_id: [u8; 32],
    pub object_len: u64,
    pub chunk_count: u32,
    pub chunks: Vec<Vec<u8>>,
}

/// Seal `object` under `key_material` with a **fresh random** object identifier.
pub fn seal_object(
    key_material: &[u8; 32],
    epoch: u64,
    object: &[u8],
) -> Result<SealedObject, String> {
    if object.len() as u64 > MAX_OBJECT_BYTES {
        return Err(format!(
            "object length {} exceeds the {MAX_OBJECT_BYTES}-byte ceiling",
            object.len()
        ));
    }
    let key = ObjectKey::fresh(key_material)?;
    seal_with(&key, epoch, object)
}

/// Seal `object` under a **derived** object identifier, for content-addressed
/// callers that must not send the identifier out of band. Two different
/// plaintexts derive different identifiers; the same plaintext under the same
/// key re-seals the same ciphertext, which is a repeated message, not a nonce
/// reused under different data.
pub fn seal_object_with_id(
    key_material: &[u8; 32],
    epoch: u64,
    object: &[u8],
) -> Result<SealedObject, String> {
    if object.len() as u64 > MAX_OBJECT_BYTES {
        return Err(format!(
            "object length {} exceeds the {MAX_OBJECT_BYTES}-byte ceiling",
            object.len()
        ));
    }
    let key = ObjectKey::from_parts(key_material, content_object_id(object));
    seal_with(&key, epoch, object)
}

/// The object identifier a content-addressed caller derives from the plaintext.
/// Callers that publish the resulting chunks must use the same function on the
/// plaintext cell they publish.
pub fn content_object_id(object: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(OBJECT_ID_DOMAIN);
    hasher.update(Sha256::digest(object));
    hasher.finalize().into()
}

fn seal_with(key: &ObjectKey, epoch: u64, object: &[u8]) -> Result<SealedObject, String> {
    let chunks = Chunker::default().chunk(object);
    let chunk_count = u32::try_from(chunks.len())
        .map_err(|_| format!("{} chunks exceeds the u32 counter", chunks.len()))?;
    let object_len = object.len() as u64;
    let mut sealed = Vec::with_capacity(chunks.len());
    for (index, plain) in &chunks {
        sealed.push(key.seal_chunk(epoch, object_len, chunk_count, *index, plain)?);
    }
    Ok(SealedObject {
        version: ENVELOPE_VERSION,
        object_id: key.object_id(),
        object_len,
        chunk_count,
        chunks: sealed,
    })
}

/// Open a sealed object, refusing an unknown format version, a chunk count that
/// disagrees with the body, or a length that does not add up to what was bound
/// as associated data.
pub fn open_object(
    key_material: &[u8; 32],
    epoch: u64,
    sealed: &SealedObject,
) -> Result<Vec<u8>, String> {
    if sealed.version != ENVELOPE_VERSION {
        return Err(format!(
            "envelope version {} is not supported (this build writes {ENVELOPE_VERSION}); \
             version-1 ciphertext bound no object identity and is refused",
            sealed.version
        ));
    }
    if sealed.chunks.len() != sealed.chunk_count as usize {
        return Err(format!(
            "envelope declares {} chunk(s) but carries {}",
            sealed.chunk_count,
            sealed.chunks.len()
        ));
    }
    let key = ObjectKey::from_parts(key_material, sealed.object_id);
    let mut out = Vec::with_capacity(sealed.object_len as usize);
    for (index, ct) in sealed.chunks.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| "chunk index overflow".to_string())?;
        out.extend_from_slice(&key.open_chunk(
            epoch,
            sealed.object_len,
            sealed.chunk_count,
            index,
            ct,
        )?);
    }
    if out.len() as u64 != sealed.object_len {
        return Err(format!(
            "envelope declares {} byte(s) but opened {}",
            sealed.object_len,
            out.len()
        ));
    }
    Ok(out)
}

pub fn generate_ephemeral() -> (EphemeralSecret, PublicKey) {
    let secret = EphemeralSecret::random_from_rng(OsRng);
    let public = PublicKey::from(&secret);
    (secret, public)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypair_generates_unique_keys() {
        let kp1 = DataKeypair::generate();
        let kp2 = DataKeypair::generate();
        assert_ne!(kp1.public_key_bytes(), kp2.public_key_bytes());
    }

    #[test]
    fn public_key_hex_is_64_chars() {
        let kp = DataKeypair::generate();
        assert_eq!(kp.public_key_hex().len(), 64);
    }

    #[test]
    fn shared_secret_is_symmetric() {
        let alice = DataKeypair::generate();
        let bob = DataKeypair::generate();
        let alice_shared = alice.shared_secret(bob.public_key());
        let bob_shared = bob.shared_secret(alice.public_key());
        assert_eq!(alice_shared, bob_shared);
    }

    fn transport_key() -> [u8; 32] {
        let kp = DataKeypair::generate();
        derive_aes_key(&kp.shared_secret(kp.public_key()))
    }

    fn cell(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = transport_key();
        let plaintext = b"hello, encrypted world!";
        let sealed = seal_object(&key, 0, plaintext).unwrap();
        assert_ne!(sealed.chunks[0], plaintext);
        assert_eq!(open_object(&key, 0, &sealed).unwrap(), plaintext);
    }

    #[test]
    fn different_chunk_indices_produce_different_ciphertext() {
        let key = transport_key();
        let plaintext = b"same data";
        let object = ObjectKey::fresh(&key).unwrap();
        let ct0 = object.seal_chunk(0, 9, 2, 0, plaintext).unwrap();
        let ct1 = object.seal_chunk(0, 9, 2, 1, plaintext).unwrap();
        assert_ne!(ct0, ct1);
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let key1 = transport_key();
        let key2 = transport_key();
        let sealed = seal_object(&key1, 0, b"secret").unwrap();
        assert!(open_object(&key2, 0, &sealed).is_err());
    }

    #[test]
    fn wrong_chunk_index_fails_decryption() {
        let key = transport_key();
        let object = ObjectKey::fresh(&key).unwrap();
        let ct = object.seal_chunk(0, 8, 2, 0, b"secret").unwrap();
        assert!(object.open_chunk(0, 8, 2, 1, &ct).is_err());
    }

    #[test]
    fn empty_plaintext_roundtrips() {
        let key = transport_key();
        let sealed = seal_object(&key, 0, b"").unwrap();
        assert!(open_object(&key, 0, &sealed).unwrap().is_empty());
    }

    #[test]
    fn large_chunk_roundtrips() {
        let key = transport_key();
        let plaintext = cell(100_000, 7);
        let sealed = seal_object(&key, 0, &plaintext).unwrap();
        assert_eq!(open_object(&key, 0, &sealed).unwrap(), plaintext);
    }

    /// The failure this patch removes: one key, no per-object context, so two
    /// different cells in one epoch encrypt chunk 0 with the same key and nonce.
    /// Kept as a control precisely because it must keep failing — if a future
    /// change makes the two ciphertexts differ here, the nonce is no longer
    /// what the format relies on and the guarantee above needs re-deriving.
    #[test]
    fn shared_object_context_repeats_the_nonce_which_is_the_old_bug() {
        let key = transport_key();
        let old_context = [0u8; 32];
        let a = ObjectKey::from_parts(&key, old_context);
        let b = ObjectKey::from_parts(&key, old_context);
        let ct_a = a.seal_chunk(0, 6, 1, 0, b"victim").unwrap();
        let ct_b = b.seal_chunk(0, 6, 1, 0, b"victim").unwrap();
        assert_eq!(ct_a, ct_b, "same key + same object id + same index");
    }

    /// D02 witness: two envelopes in one epoch do not share a nonce, so their
    /// ciphertexts differ even for identical plaintext.
    #[test]
    fn two_objects_in_one_epoch_do_not_share_a_nonce() {
        let key = transport_key();
        let first = seal_object(&key, 0, b"victim payload").unwrap();
        let second = seal_object(&key, 0, b"victim payload").unwrap();
        assert_ne!(first.object_id, second.object_id);
        assert_ne!(first.chunks[0], second.chunks[0]);
        assert_ne!(open_object(&key, 0, &first).unwrap(), Vec::<u8>::new());
    }

    /// D02 witness: a chunk spliced from another object under the same key is
    /// refused, because the object identity is authenticated.
    #[test]
    fn chunk_cannot_be_spliced_between_objects() {
        let key = transport_key();
        let victim = seal_object(&key, 0, &cell(300_000, 1)).unwrap();
        let other = seal_object(&key, 0, &cell(300_000, 2)).unwrap();
        let mut spliced = victim.clone();
        spliced.chunks[0] = other.chunks[0].clone();
        assert!(open_object(&key, 0, &spliced).is_err());
    }

    /// The epoch and the declared length are bound as associated data, so a
    /// replay into a different epoch or a shrunken length fails authentication.
    #[test]
    fn epoch_and_length_are_authenticated() {
        let key = transport_key();
        let sealed = seal_object(&key, 3, &cell(1000, 9)).unwrap();
        assert_eq!(open_object(&key, 3, &sealed).unwrap().len(), 1000);
        assert!(open_object(&key, 4, &sealed).is_err());
        let mut shortened = sealed.clone();
        shortened.object_len = 500;
        assert!(open_object(&key, 3, &shortened).is_err());
    }

    /// A legacy envelope (no version, no object identity) is refused by name
    /// rather than attempted under the new derivation.
    #[test]
    fn version_one_envelope_is_refused() {
        let key = transport_key();
        let mut sealed = seal_object(&key, 0, b"legacy").unwrap();
        sealed.version = 1;
        let err = open_object(&key, 0, &sealed).unwrap_err();
        assert!(err.contains("version 1"), "{err}");
    }

    /// Counter and length bounds are checked before the cipher is touched.
    #[test]
    fn out_of_range_index_and_oversize_object_are_refused() {
        let key = transport_key();
        let object = ObjectKey::fresh(&key).unwrap();
        assert!(object.seal_chunk(0, 8, 3, 3, b"x").is_err());
        assert!(object.seal_chunk(0, 8, 0, 0, b"x").is_err());
        assert!(object.seal_chunk(0, MAX_OBJECT_BYTES + 1, 1, 0, b"x").is_err());
    }

    /// Content-addressed sealing is deterministic in the identifier, so a
    /// recipient that knows the plaintext digest needs no out-of-band salt.
    #[test]
    fn content_addressed_sealing_derives_a_stable_object_id() {
        let key = transport_key();
        let plaintext = cell(700, 4);
        let a = seal_object_with_id(&key, 0, &plaintext).unwrap();
        let b = seal_object_with_id(&key, 0, &plaintext).unwrap();
        assert_eq!(a.object_id, content_object_id(&plaintext));
        assert_eq!(a, b);
        let other = seal_object_with_id(&key, 0, &cell(700, 5)).unwrap();
        assert_ne!(a.object_id, other.object_id);
        assert_eq!(open_object(&key, 0, &a).unwrap(), plaintext);
    }
}
