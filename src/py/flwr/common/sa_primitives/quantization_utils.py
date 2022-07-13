from flwr.common.typing import Weights
import numpy as np
from logging import log

# Weight Quantization ======================================================================

# Clip weight vector to [-clipping_range, clipping_range]
# Transform weight vector to range [0, target_range] and take floor
# take floor => stochastic rounding
# If final value is target_range, take 1 from it so it is an integer from 0 to target_range-1


def stochastic_round(arr: np.ndarray):
    ret = np.ceil(arr).astype(np.int32)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1
    return ret


def quantize(weight: Weights, clipping_range: float, target_range: int) -> Weights:
    quantized_list = []
    check_clipping_range(weight, clipping_range)
    quantizer = target_range/(2*clipping_range)
    for arr in weight:
        # stochastic quantization
        quantized = (np.clip(arr, -clipping_range, clipping_range) + clipping_range) * quantizer
        quantized = stochastic_round(quantized)
        quantized_list.append(quantized)
    return quantized_list


# quantize weight vectors (unbounded)
# mod k (i.e., mod_range here)
def quantize_unbounded(weight: Weights, clipping_range: float, target_range: int, mod_range: int) -> Weights:
    quantized_list = []
    check_clipping_range(weight, clipping_range)
    quantizer = target_range/(2*clipping_range)
    flag = bin(mod_range).count("1") == 1
    msk = mod_range - 1
    for arr in weight:
        # stochastic quantization
        quantized = (arr + clipping_range) * quantizer
        quantized = stochastic_round(quantized)
        # mod k
        quantized = quantized & msk if flag else np.mod(quantized, mod_range)
        quantized_list.append(quantized)
    return quantized_list

# Quick check that all numbers are within the clipping range
# Throw warning if there exists numbers that exceed it


def check_clipping_range(weight: Weights, clipping_range: float):
    for arr in weight:
        for x in arr.flatten():
            if(x < -clipping_range or x > clipping_range):
                log(WARNING,
                    f"There are some numbers in the local vector that exceeds clipping range. Please increase the clipping range to account for value {x}")
                return

# Transform weight vector to range [-clipping_range, clipping_range]
# Convert to float


def reverse_quantize(weight: Weights, clipping_range: float, target_range: int) -> Weights:
    reverse_quantized_list = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range
    for arr in weight:
        arr = arr.view(np.ndarray).astype(float) * quantizer + shift
        reverse_quantized_list.append(arr)
    return reverse_quantized_list
