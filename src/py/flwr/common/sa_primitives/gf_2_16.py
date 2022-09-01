from typing import List, Tuple
import os
import numpy as np

IRR_POLY = 0o210013
MAX_VALUE = (1 << 16) - 1
FIRST_DIGIT_MASK = 1 << 15

save_pth = os.path.dirname(os.path.abspath(__file__)) + "\\exps_logs.npy"


class Operand:
    def __init__(self, num=0):
        self.num = np.uint16(num)

    def __xor__(self, other):
        return Operand(self.num ^ other.num)

    def __add__(self, other):
        return self.__xor__(other)

    def __sub__(self, other):
        return self.__xor__(other)

    def __mul__(self, other):
        if self.num == 0 or other.num == 0:
            return 0
        t = (int(logs[self.num]) + logs[other.num])
        return Operand(exps[t])

    def __ne__(self, other):
        return self.num != other.num

    def __eq__(self, other):
        return self.num == other.num

    def __invert__(self):
        return Operand(exps[MAX_VALUE - logs[self.num]])

    def __truediv__(self, other):
        return self * ~other

    def __lt__(self, other):
        return self.num < other.num

    def __str__(self):
        return str(self.num)

    def __repr__(self):
        return str(self.num)

    def to_int(self):
        return int(self.num)


def slow_multiply(a: Operand, b: Operand) -> Operand:
    aa = a.num
    bb = b.num
    r = np.uint16(0)
    while aa != 0:
        if aa & 0x1 != 0:
            r = np.uint16(r ^ bb)
        t = bb & FIRST_DIGIT_MASK
        bb = np.uint16(bb << 1)
        if t != 0:
            bb = np.uint16(bb ^ IRR_POLY)
        aa >>= 1
    return Operand(r)


def build_tables():
    if os.path.exists(save_pth):
        arr = np.load(save_pth)
        _exps, _logs = arr[0], arr[1]
        return _exps, _logs

    _exps, _logs = [0 for _ in range(MAX_VALUE + 1)], [0 for _ in range(MAX_VALUE + 1)]
    gen = Operand(3)
    _exps[0] = 1
    it = Operand(1)
    for i in range(1, MAX_VALUE + 1):
        it = slow_multiply(gen, it)
        _exps[i] = it.to_int()
    for i in range(MAX_VALUE + 1):
        _logs[_exps[i]] = i
    _logs[1] = 0
    _exps = np.array(_exps, dtype=np.uint16)
    _logs = np.array(_logs, dtype=np.uint16)
    ret = (_exps, _logs)
    arr = np.vstack(ret)
    np.save(save_pth, arr)
    return ret


exps, logs = build_tables()


def interpolate_gf(share: List[Tuple[Operand, Operand]]) -> int:
    secret = Operand(0)
    n = len(share)
    x_prod = Operand(1)
    for i in range(n):
        x_prod = x_prod * share[i][0]
    for i in range(n):
        term = x_prod / share[i][0]
        for j in range(n):
            if i == j:
                continue
            term = term / (share[j][0] - share[i][0])
        secret = secret + term * share[i][1]
    return secret.num
