---
fed-number: 0004
title: FED Ephemeral ID solution
authors: ["@adap"]
creation-date: 2023-07-25
last-updated: 2023-07-25
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Appendix](#appendix)

## Summary

Secure aggregation (SA) requires persistent client IDs among multiple rounds of communication in one aggregation process.
The current implementation can only work with identifiable clients to meet such a requirement.
For the purpose of applying SA to anonymous clients, it is necessary to introduce a mechanism that assigns clients temporary IDs, AKA ephemeral IDs, that only last for a few rounds of communication and will be disregarded once the aggregation is completed in the context of SA.

This FED doc aims to discuss different ways to implement ephemeral IDs and extend SA functionalities to anonymous clients.
It also includes discussions about threat models, RSA signatures, and industry-level security considerations.

## Background

### Threat models

1. **Semi-honest**
"Semi-honest", AKA "honest but curious", refers to the scenarios where all parties strictly follow the protocols while attackers will try to infer confidential information from received messages.

2. **Malicious**
Malicious settings consider situations where attackers actively participate in the whole process, manipulate the protocols, and even emulate multiple clients to trick others.

### RSA Signatures
RSA is one of the most commonly used asymmetric cryptographic algorithms. It can create a pair of private key and public key. Any party can decrypt ciphertext using the public key, but only the party holding the private key can encrypt plaintext. A typical signature process is that, first, compute the hash (e.g. 256 bit hash) of the message and then encrypt the hash using the private key. Hence, any party can verify the signature by first decrypting it and then comparing it with the hash of the message.

A toy example is as follows.
``` python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature

import time
import os


def generate_keys():
    # Generate a 2048 bit RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()

    return private_key, public_key


def sign_message(message, private_key):
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature


def verify_signature(message, signature, public_key):
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False


def main():
    private_key, public_key = generate_keys()

    message = b"I am a famous message: hello world!"
    signature = sign_message(message, private_key)

    is_valid = verify_signature(message, signature, public_key)
    if is_valid: # of course, it should be valid in this example.
        print("Signature is valid.")
    else:
        print("Signature is not valid.")


if __name__ == "__main__":
    main()
```

## Real-world Concerns
When it comes to cryptographic implementations in real-world applications, the above toy example might not be sufficient due to the following reasons:

1. **Key management:** Keys should be securely stored and transmitted. If a private key is leaked, all past and future communications are compromised. Similarly, it's also important to ensure that the public key belongs to the correct entity (other than a man-in-the-middle attacker), usually handled through a system of certificates signed by trusted certificate authorities. I personally won't recommend to let the flower server be the key issuing authority.

2. **Timestamping:** We may need to consider including a timestamp in the signature to prevent replay attacks, which are when an attacker retransmits a valid signature to trick a system.
For example, a driver can send DH public keys with signatures from previous rounds, whose corresponding private keys may have been recovered in unmasking phases, to a client, and then the driver can reveal all communications with the client.

3. **Key rotation:** In real-world scenarios, keys should be regularly updated or rotated to limit the damage if keys are compromised. As for SA, I suggest that RSA keys should be regenerated in the beginning of each aggregation.

4. **Key size:** The size of the keys used can impact the security. A larger key size generally means more security, but it also means more computational overhead. 2048-bit keys suffice in most cases. We may need to reduce the size of the keys if client devices have very limited resources, but it should be feasible to have 2048-bit keys given that clients should have the computing capabilities to train ML models.

5. **Algorithm choice:** The choice of cryptographic algorithms and hashing functions can significantly impact the security of the system. The SHA-256 and RSA used in the example are generally considered secure.

## Performance Analysis

The following code tests the efficiency of the RSA signature with `Cryptography` library. All experiments are repeated 1024 tims and I take the average CPU time. I also assume a model with 1 million parameters.

``` python
def eval_generate_keys(times=1 << 10):
    mark = time.time()
    for _ in range(times):
        generate_keys()
    print(f"generation time: {(time.time() - mark) / times} s")


def eval_sign_message(message=os.urandom(8000000), times=1 << 10):
    private_key, public_key = generate_keys()
    mark = time.time()
    for _ in range(times):
        sgn = sign_message(message, private_key)
    print(f"signing time: {(time.time() - mark) / times} s")
    print(f"signature length: {len(sgn)}")


def eval_verify_signature(message=os.urandom(8000000), times=1 << 10):
    private_key, public_key = generate_keys()
    sgn = sign_message(message, private_key)
    mark = time.time()
    for _ in range(times):
        verify_signature(message, sgn, public_key)
    print(f"verification time: {(time.time() - mark) / times} s")
```


## Discussion

**What are the roles of the flower server?**

The flower server currently acts as a router that receives messages and forwards them to the correct parties (correct means the parites specified in the `Consumer` fields). 
<!-- To introduce a RSA signature systems. -->

<!-- [TODO]

### Goals

[TODO]

### Non-Goals

[TODO]

## Proposal

[TODO]

## Drawbacks

[TODO] -->
