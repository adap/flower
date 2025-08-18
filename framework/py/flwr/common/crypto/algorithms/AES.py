
# AES PURO
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

KEY_AES = b"0123456789abcdef0123456789abcdef"  # 32 byte -> AES-256
BLOCK_SIZE = 16  # dimensione blocco AES (in byte)

def pad(data: bytes) -> bytes:
    """Applica padding PKCS#7 ai dati per farli diventare multipli di 16 byte."""
    pad_len = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([pad_len] * pad_len)

def unpad(data: bytes) -> bytes:
    """Rimuove il padding PKCS#7 dai dati."""
    pad_len = data[-1]
    if pad_len < 1 or pad_len > BLOCK_SIZE:
        raise ValueError("Padding non valido")
    return data[:-pad_len]

def encrypt(data: bytes) -> bytes:
    """
    Cifra i dati con AES-256-CBC.
    Restituisce IV + ciphertext (IV casuale).
    """
    iv = os.urandom(BLOCK_SIZE)
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(data)
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext  # prepende IV per l'uso in decriptazione

def decrypt(encrypted_data: bytes) -> bytes:
    """
    Decifra i dati cifrati con AES-256-CBC.
    Si aspetta IV + ciphertext.
    """
    if len(encrypted_data) < BLOCK_SIZE:
        raise ValueError("Dati cifrati troppo corti")

    iv = encrypted_data[:BLOCK_SIZE]
    ciphertext = encrypted_data[BLOCK_SIZE:]
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return unpad(padded_plaintext)
