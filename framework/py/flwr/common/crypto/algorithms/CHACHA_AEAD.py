from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import os

# Chiave a 256 bit (32 byte)
KEY_CHACHA = b"12345678901234567890123456789012"
NONCE_SIZE = 12  # Consigliato per ChaCha20-Poly1305

def encrypt(data: bytes) -> bytes:
    """Cifra i dati con ChaCha20-Poly1305."""
    nonce = os.urandom(NONCE_SIZE)
    chacha = ChaCha20Poly1305(KEY_CHACHA)
    ciphertext = chacha.encrypt(nonce, data, associated_data=None)
    return nonce + ciphertext  # Concateniamo il nonce per usarlo nella decryption

def decrypt(encrypted_data: bytes) -> bytes:
    """Decifra i dati con ChaCha20-Poly1305."""
    nonce = encrypted_data[:NONCE_SIZE]
    ciphertext = encrypted_data[NONCE_SIZE:]
    chacha = ChaCha20Poly1305(KEY_CHACHA)
    return chacha.decrypt(nonce, ciphertext, associated_data=None)
