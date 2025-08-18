from .algorithms import (
    AES
)

def encrypt(data: bytes, method: str) -> bytes:
    if method == "AES":
        return AES.encrypt(data)
    # elif method == "":
    #     return ""
    else:
        raise ValueError(f"Unknown encryption method: {method}")


def decrypt(data: bytes, method: str) -> bytes:
    if method == "AES":
        return AES.decrypt(data)
    # elif method == "":
    #     return ""
    else:
        raise ValueError(f"Unknown decryption method: {method}")
