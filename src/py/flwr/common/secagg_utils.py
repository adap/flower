from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization

def generate_key_pairs():
    sk=ec.generate_private_key(ec.SECP384R1())
    pk=sk.public_key()
    return sk,pk

def private_key_to_bytes(sk):
    return sk.private_bytes(encoding=serialization.Encoding.PEM, 
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption)

def bytes_to_private_key(b):
    return serialization.load_pem_private_key(data= b, password=None)

def public_key_to_bytes(pk):
    return pk.public_bytes(encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

def bytes_to_public_key(b):
    return serialization.load_pem_public_key(data = b)