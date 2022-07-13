from typing import List, Tuple
from flwr.common.typing import Weights


# Weight Manipulation =============================================================

# Combine factor with weights


def factor_weights_combine(weights_factor: int, weights: Weights) -> Weights:
    return [np.array([weights_factor])]+weights

# Extract factor from weights


def factor_weights_extract(weights: Weights) -> Tuple[int, Weights]:
    return weights[0][0], weights[1:]

# Create dimensions list of each element in weights


def weights_shape(weights: Weights) -> List[Tuple]:
    return [arr.shape for arr in weights]

# Generate zero weights based on dimensions list


def weights_zero_generate(dimensions_list: List[Tuple], dtype=np.int64) -> Weights:
    return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]

# Add two weights together


def weights_addition(a: Weights, b: Weights) -> Weights:
    return [a[idx]+b[idx] for idx in range(len(a))]

# Subtract one weight from the other


def weights_subtraction(a: Weights, b: Weights) -> Weights:
    return [a[idx]-b[idx] for idx in range(len(a))]

# Take mod of a weights with an integer


def weights_mod(a: Weights, b: int) -> Weights:
    if bin(b).count("1") == 1:
        msk = b - 1
        return [a[idx] & msk for idx in range(len(a))]
    return [a[idx] % b for idx in range(len(a))]


# Multiply weight by an integer


def weights_multiply(a: Weights, b: int) -> Weights:
    return [a[idx] * b for idx in range(len(a))]

# Divide weight by an integer


def weights_divide(a: Weights, b: int) -> Weights:
    return [a[idx] / b for idx in range(len(a))]
