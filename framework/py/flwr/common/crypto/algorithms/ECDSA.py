from ecdsa import SigningKey, VerifyingKey, NIST256p

def generate_ecc_keys():
    """Genera coppia chiave privata/pubblica"""
    sk = SigningKey.generate(curve=NIST256p)
    vk = sk.verifying_key
    return sk, vk

def ecc_sign(privkey: SigningKey, data: bytes) -> bytes:
    """Firma un messaggio"""
    return privkey.sign(data)

def ecc_verify(pubkey: VerifyingKey, data: bytes, signature: bytes) -> bool:
    """Verifica una firma"""
    try:
        return pubkey.verify(signature, data)
    except Exception:
        return False
