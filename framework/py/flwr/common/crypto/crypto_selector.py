from .algorithms import (
    AES, HMAC, CHACHA_AEAD, CHACHA, AES_GCM, KOBLITZ)

def encrypt(data: bytes, method: str, ecc_pubkey=None) -> bytes:
    if method == "AES":
        return AES.encrypt(data)
    elif method == "HMAC":
         return HMAC.add_hmac(data)
    elif method == "CHACHA":
        return CHACHA.encrypt(data)
    elif method == "CHACHA_AEAD":
        return CHACHA_AEAD.encrypt(data)
    elif method == "AES_GCM":
        return AES_GCM.encrypt(data)
    elif KOBLITZ.is_supported_method(method):
        raise ValueError(
            "Il metodo KOBLITZ supporta solo generazione chiavi e autenticazione, "
            "non la cifratura dei dati."
        )
    else:
        raise ValueError(f"Unknown encryption method: {method}")


def decrypt(data: bytes, method: str, ecc_privkey=None) -> bytes:
    if method == "AES":
        return AES.decrypt(data)
    elif method == "HMAC":
        return HMAC.check_hmac(data)
    elif method == "CHACHA":
        return CHACHA.decrypt(data)
    elif method == "CHACHA_AEAD":
        return CHACHA_AEAD.decrypt(data)
    elif method == "AES_GCM":
        return AES_GCM.decrypt(data)
    elif KOBLITZ.is_supported_method(method):
        raise ValueError(
            "Il metodo KOBLITZ supporta solo generazione chiavi e autenticazione, "
            "non la decifratura dei dati."
        )
    else:
        raise ValueError(f"Unknown decryption method: {method}")

def add_integrity(data: bytes, method: str) -> bytes:
    if method == "HMAC":
        return HMAC.add_hmac(data)
    else:
        raise ValueError(f"Unknown integrity method: {method}")

def check_integrity(data: bytes, method: str) -> bytes:
    if method == "HMAC":
        return HMAC.check_hmac(data)
    else:
        raise ValueError(f"Unknown integrity method: {method}")

