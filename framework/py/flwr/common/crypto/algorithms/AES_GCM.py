from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

KEY = b"0123456789abcdef0123456789abcdef"  # 32 byte
NONCE_SIZE = 12

def encrypt(data: bytes) -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(KEY)
    ciphertext = aesgcm.encrypt(nonce, data, associated_data=None)
    return nonce + ciphertext

def decrypt(encrypted_data: bytes) -> bytes:
    nonce = encrypted_data[:NONCE_SIZE]
    ciphertext = encrypted_data[NONCE_SIZE:]
    aesgcm = AESGCM(KEY)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
