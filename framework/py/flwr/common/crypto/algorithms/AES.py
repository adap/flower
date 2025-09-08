import inspect
import time
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

KEY_AES = b"0123456789abcdef0123456789abcdef"  # 32 byte -> AES-256
BLOCK_SIZE = 16  # dimensione blocco AES (in byte)

def pad(data: bytes) -> bytes:
    pad_len = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([pad_len] * pad_len)

def unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    if pad_len < 1 or pad_len > BLOCK_SIZE:
        raise ValueError("Padding non valido")
    return data[:-pad_len]
def encrypt(data: bytes) -> bytes:

    # caller_frame = inspect.stack()[1]
    # caller_name = caller_frame.function  # Nome della funzione chiamante
    # start_time = time.perf_counter()
    iv = os.urandom(BLOCK_SIZE)
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(data)
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    end_time = time.perf_counter()
    #print(f"Encryption time AES: {end_time - start_time:.4f} seconds (Called by: {caller_name})")

    return iv + ciphertext

def decrypt(encrypted_data: bytes) -> bytes:
    start_time = time.perf_counter()
    if len(encrypted_data) < BLOCK_SIZE:
        raise ValueError("Dati cifrati troppo corti")
    iv = encrypted_data[:BLOCK_SIZE]
    ciphertext = encrypted_data[BLOCK_SIZE:]
    cipher = Cipher(algorithms.AES(KEY_AES), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    plaintext = unpad(padded_plaintext)
    end_time = time.perf_counter()
    #print(f"Decryption time AES: {end_time - start_time:.4f} seconds")
    return plaintext
