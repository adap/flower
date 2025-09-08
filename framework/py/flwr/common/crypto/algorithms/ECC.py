from tinyec import registry
import secrets

elliptic_curve = registry.get_curve("brainpoolP256r1")

alice_priv_key = secrets.randbelow(elliptic_curve.field.n)
alice_pub_key  = alice_priv_key*elliptic_curve.g #elliptic curve generator point

bob_priv_key   = secrets.randbelow(elliptic_curve.field.n)
bob_pub_key    = bob_priv_key*elliptic_curve.g

alice_shared_key = alice_priv_key*bob_pub_key

print(alice_shared_key)

bob_shared_key = bob_priv_key*alice_pub_key

are_equal = "Yes" if alice_shared_key == bob_shared_key else "No"

print(f"The two keys are equal? {are_equal}")

from Crypto.Cipher import AES
import hashlib, binascii

def encrypt(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_point_to_256_bit_key(point): #simple key derivation function

    sha = hashlib.sha256(int.to_bytes(point.x,32,'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

alice_shared_key_to_256 = ecc_point_to_256_bit_key(alice_shared_key)
bob_shared_key_to_256 = ecc_point_to_256_bit_key(bob_shared_key)
msg_to_Bob = b'Hi Bob, how are you?'

ciphertext, nonce, authTag = encrypt(msg_to_Bob, alice_shared_key_to_256)
plaintext = decrypt(ciphertext, nonce, authTag, bob_shared_key_to_256)
print(f'Bob received from Alice the message {plaintext.decode()}')
