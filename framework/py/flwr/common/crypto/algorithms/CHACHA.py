from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

KEY = b"12345678901234567890123456789012"
NONCE_SIZE = 16  # oppure 12, dipende dall'implementazione

def encrypt(data: bytes) -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    cipher = Cipher(algorithms.ChaCha20(KEY, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data)
    return nonce + ciphertext

def decrypt(encrypted_data: bytes) -> bytes:
    nonce = encrypted_data[:NONCE_SIZE]
    ciphertext = encrypted_data[NONCE_SIZE:]
    cipher = Cipher(algorithms.ChaCha20(KEY, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext)
