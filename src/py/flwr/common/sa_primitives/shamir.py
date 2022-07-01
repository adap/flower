from flwr.common.sa_primitives.gf_2_16 import Operand, interpolate_gf, exps, logs, slow_multiply
import numpy as np
import os
from typing import List, Tuple

MAX_VALUE = 65535
LAST16_BITS_MASK = 0x0000ffff


def pack(shares: List[List[Tuple[Operand, Operand]]]) -> List[bytes]:
    ret = []
    for share in shares:
        arr = bytearray()
        for p in share:
            arr.extend(int.to_bytes(int((p[0].num << 16) | p[1].num),
                                    4, 'little', signed=False))
        ret.append(bytes(arr))
    return ret


def unpack(shares: List[bytes]) -> List[List[Tuple[Operand, Operand]]]:
    ret = []
    for share in shares:
        lst = []
        for i in range(0, len(share), 4):
            t = int.from_bytes(share[i: i + 4], 'little', signed=False)
            x = (t >> 16) & LAST16_BITS_MASK
            y = t & LAST16_BITS_MASK
            lst.append((Operand(x), Operand(y)))
        ret.append(lst)
    return ret


def create_shares(secret: bytes, threshold: int, num_shares: int) -> List[bytes]:
    shares = [[] for _ in range(num_shares)]
    for p in range(0, len(secret), 2):
        data = int.from_bytes(secret[p: p + 2], 'little', signed=False)
        # set the first coefficient of the polynomial to secret
        coeff = [Operand(data)]
        # sample other coefficients randomly
        for i in range(1, threshold):
            # coeff.append(Operand(int.from_bytes(os.urandom(2), 'little', signed=False)))
            coeff.append(Operand(i + 1))
        # evaluate the polynomial at num_shares points
        for i in range(num_shares):
            x = Operand(i + 1)
            y = coeff[0]
            xp = Operand(1)
            for j in range(1, threshold):
                xp = xp * x
                y = y + (coeff[j] * xp)
            shares[i].append((x, y))
    return pack(shares)


def combine_shares(shares: List[bytes]) -> bytes:
    shares = unpack(shares)
    secret = bytearray()
    secret_size = len(shares[0])
    k = len(shares)

    for di in range(secret_size):
        share = []
        for i in range(k):
            share.append(shares[i][di])
        data = interpolate_gf(share)
        data = int.to_bytes(int(data), 2, 'little')
        secret.extend(data)
    return bytes(secret)


if __name__ == '__main__':
    secret = b'01234567890123456789012345678912'
    shares = create_shares(secret, 4, 7)
    rec2 = combine_shares(shares[:4])
    print(rec2)



