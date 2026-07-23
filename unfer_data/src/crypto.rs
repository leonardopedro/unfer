use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

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

pub fn encrypt_chunk(
    aes_key: &[u8; 32],
    chunk_index: u32,
    plaintext: &[u8],
) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new_from_slice(aes_key).map_err(|e| e.to_string())?;
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[8..12].copy_from_slice(&chunk_index.to_le_bytes());
    let nonce = Nonce::from_slice(&nonce_bytes);
    cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| format!("encryption failed: {e}"))
}

pub fn decrypt_chunk(
    aes_key: &[u8; 32],
    chunk_index: u32,
    ciphertext: &[u8],
) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new_from_slice(aes_key).map_err(|e| e.to_string())?;
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[8..12].copy_from_slice(&chunk_index.to_le_bytes());
    let nonce = Nonce::from_slice(&nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| format!("decryption failed: {e}"))
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

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let alice = DataKeypair::generate();
        let bob = DataKeypair::generate();
        let shared = alice.shared_secret(bob.public_key());
        let aes_key = derive_aes_key(&shared);

        let plaintext = b"hello, encrypted world!";
        let ciphertext = encrypt_chunk(&aes_key, 0, plaintext).unwrap();
        assert_ne!(ciphertext, plaintext);

        let decrypted = decrypt_chunk(&aes_key, 0, &ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn different_chunk_indices_produce_different_ciphertext() {
        let kp = DataKeypair::generate();
        let aes_key = derive_aes_key(&kp.shared_secret(kp.public_key()));
        let plaintext = b"same data";
        let ct0 = encrypt_chunk(&aes_key, 0, plaintext).unwrap();
        let ct1 = encrypt_chunk(&aes_key, 1, plaintext).unwrap();
        assert_ne!(ct0, ct1);
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let kp1 = DataKeypair::generate();
        let kp2 = DataKeypair::generate();
        let key1 = derive_aes_key(&kp1.shared_secret(kp1.public_key()));
        let key2 = derive_aes_key(&kp2.shared_secret(kp2.public_key()));

        let ct = encrypt_chunk(&key1, 0, b"secret").unwrap();
        assert!(decrypt_chunk(&key2, 0, &ct).is_err());
    }

    #[test]
    fn wrong_chunk_index_fails_decryption() {
        let kp = DataKeypair::generate();
        let aes_key = derive_aes_key(&kp.shared_secret(kp.public_key()));
        let ct = encrypt_chunk(&aes_key, 0, b"secret").unwrap();
        assert!(decrypt_chunk(&aes_key, 1, &ct).is_err());
    }

    #[test]
    fn empty_plaintext_roundtrips() {
        let kp = DataKeypair::generate();
        let aes_key = derive_aes_key(&kp.shared_secret(kp.public_key()));
        let ct = encrypt_chunk(&aes_key, 0, b"").unwrap();
        let pt = decrypt_chunk(&aes_key, 0, &ct).unwrap();
        assert!(pt.is_empty());
    }

    #[test]
    fn large_chunk_roundtrips() {
        let kp = DataKeypair::generate();
        let aes_key = derive_aes_key(&kp.shared_secret(kp.public_key()));
        let plaintext: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let ct = encrypt_chunk(&aes_key, 42, &plaintext).unwrap();
        let pt = decrypt_chunk(&aes_key, 42, &ct).unwrap();
        assert_eq!(pt, plaintext);
    }
}
