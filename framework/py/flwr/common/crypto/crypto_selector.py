from cryptography.hazmat.primitives.asymmetric.ec import ECDSA

from .algorithms import (
    AES, HMAC, CHACHA_AEAD, CHACHA, AES_GCM, ECC, ECDSA
)

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
    if method == "ECC":
        if ecc_pubkey is None:
            raise ValueError("ECC encryption requires a public key")
        return ECC.ecc_encrypt(ecc_pubkey, data)
    if method=="ECDSA":
        return ECDSA.ecc_sign(data)

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
    if method == "ECC":
        if ecc_privkey is None:
            raise ValueError("ECC decryption requires a private key")
        return ECC.ecc_decrypt(ecc_privkey, data)
    if method=="ECDSA":
        return ECDSA.ecc_verify(data)


    else:
        raise ValueError(f"Unknown decryption method: {method}")


