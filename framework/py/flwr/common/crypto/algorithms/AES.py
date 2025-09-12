from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

KEY_AES = b"0123456789abcdef0123456789abcdef"
BLOCK_SIZE = 16

def pad(data: bytes) -> bytes:
    pad_len = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([pad_len] * pad_len)

def unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    if pad_len < 1 or pad_len > BLOCK_SIZE:
        raise ValueError("Padding non valido")
    return data[:-pad_len]
def encrypt(data: bytes) -> bytes:
    iv = os.urandom(BLOCK_SIZE)
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(data)
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext

def decrypt(encrypted_data: bytes) -> bytes:
    if len(encrypted_data) < BLOCK_SIZE:
        raise ValueError("Dati cifrati troppo corti")
    iv = encrypted_data[:BLOCK_SIZE]
    ciphertext = encrypted_data[BLOCK_SIZE:]
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    plaintext = unpad(padded_plaintext)
    return plaintext
