
import hmac
import hashlib

HMAC_KEY = b"fedcba9876543210fedcba9876543210"  # 32 byte key ovvviamente non va lasciata qui in ambiente serio
HMAC_LEN = 32

def add_hmac(data: bytes) -> bytes:
    mac = hmac.new(HMAC_KEY, data, hashlib.sha256).digest()
    return data + mac

def check_hmac(signed_data: bytes) -> bytes:
    data = signed_data[:-HMAC_LEN]
    mac = signed_data[-HMAC_LEN:]
    expected_mac = hmac.new(HMAC_KEY, data, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("Error ")
    return data
