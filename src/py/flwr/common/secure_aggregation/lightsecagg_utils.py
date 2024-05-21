# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for the LightSecAgg protocol."""


from typing import Dict, List, cast

import numpy as np
from galois import FieldArray

from flwr.common import Array, array_from_numpy
from flwr.common.constant import SType

from ..typing import NDArray, NDArrayInt
from .crypto.symmetric_encryption import decrypt, encrypt


def LCC_encode_with_points(
    arr: NDArrayInt, alphas: List[int], betas: List[int], GF: type[FieldArray]
) -> NDArrayInt:
    """Encode the input array."""
    U_enc = generate_Lagrange_coeffs_galois(alphas, betas, GF)
    encoded_arr: NDArrayInt = U_enc.dot(arr)
    return encoded_arr


def LCC_decode_with_points(
    f_eval: NDArrayInt, alphas_eval: List[int], betas: List[int], GF: type[FieldArray]
) -> NDArrayInt:
    """Decode the input array."""
    U_dec = generate_Lagrange_coeffs_galois(betas, alphas_eval, GF)
    f_recon: NDArrayInt = U_dec.dot(f_eval.view(GF))

    return f_recon.view(np.ndarray)


def generate_Lagrange_coeffs_galois(
    alpha_list: List[int], beta_list: List[int], GF: type[FieldArray]
) -> NDArrayInt:
    """Generate lagrange coefficients for the finite field."""
    num_alphas = len(alpha_list)
    num_betas = len(beta_list)
    coeffs = np.zeros((num_alphas, num_betas), dtype=np.int64).view(GF)
    alphas = cast(NDArrayInt, GF(alpha_list))
    betas = cast(NDArrayInt, GF(beta_list))
    mbeta = -betas

    denoms = np.zeros(num_betas, dtype=np.int64).view(GF)
    msk = np.ones(num_betas, dtype=bool)

    for j in range(num_betas):
        msk[j] = False
        msk[j - 1] = True
        denoms[j] = (mbeta + betas[j])[msk].prod()

    nums = np.zeros(num_alphas, dtype=np.int64).view(GF)
    for i in range(num_alphas):
        nums[i] = (mbeta + alphas[i]).prod()
    for i in range(num_alphas):
        for j in range(num_betas):
            coeffs[i][j] = nums[i] / ((alphas[i] + mbeta[j]) * denoms[j])
    return coeffs


def mask_weights(
    weights: List[NDArrayInt], local_mask: NDArrayInt, GF: type[FieldArray]
) -> List[NDArrayInt]:
    """Mask the model weights in place."""
    pos = 0
    msk = local_mask.view(GF)
    weights = [w.view(GF) for w in weights]
    for i in range(len(weights)):
        w = weights[i].view(GF)
        d = w.size
        cur_mask = msk[pos : pos + d]
        cur_mask = cur_mask.reshape(w.shape)

        w += cur_mask

        weights[i] = w
        pos += d
    return weights


def unmask_aggregated_weights(
    aggregated_weights: List[NDArrayInt],
    aggregated_mask: NDArrayInt,
    GF: type[FieldArray],
) -> List[NDArrayInt]:
    """Unmask the aggregated model weights in place."""
    pos = 0
    msk = aggregated_mask.view(GF)
    for i in range(len(aggregated_weights)):
        w = aggregated_weights[i].view(GF)
        d = w.size
        cur_mask = msk[pos : pos + d]
        cur_mask = cur_mask.reshape(w.shape)

        w -= cur_mask

        aggregated_weights[i] = w.view(np.ndarray)
        pos += d
    return aggregated_weights


def encode_mask(
    total_dimension: int,
    num_clients: int,
    min_num_active_clients: int,
    privacy_guarantee: int,
    galois_field: type[FieldArray],
    local_mask: NDArrayInt,
) -> NDArrayInt:
    """Encode the given mask and return encoded sub masks."""
    # Use symbols in https://arxiv.org/abs/2109.14236
    d = total_dimension
    N = num_clients
    U = min_num_active_clients
    T = privacy_guarantee
    p = galois_field.order
    GF = galois_field

    # Build the lists of alphas and betas
    alphas = list(range(1, N + 1))
    betas = list(range(N + 1, N + U + 1))

    # Encode the mask with noise
    noise = np.random.randint(p, size=(T * (d // (U - T)), 1))
    lcc_input = np.concatenate([local_mask, noise], axis=0)
    lcc_input = np.reshape(lcc_input, (U, d // (U - T)))
    encoded_mask_set = LCC_encode_with_points(lcc_input.view(GF), alphas, betas, GF)

    return encoded_mask_set


def compute_aggregated_encoded_mask(
    encoded_mask_dict: Dict[int, NDArrayInt],
    active_nids: List[int],
    GF: type[FieldArray],
) -> NDArrayInt:
    """Compute the aggregated encoded mask."""
    ret = np.zeros_like(encoded_mask_dict[active_nids[0]]).view(GF)
    for client_id in active_nids:
        ret += encoded_mask_dict[client_id].view(GF)
    return ret


def padding(d: int, U: int, T: int) -> int:
    """Return the length of the vector after padding."""
    remainder = d % (U - T)
    if remainder != 0:
        remainder = U - T - remainder
    return d + remainder


def encrypt_sub_mask(key: bytes, sub_mask: NDArrayInt) -> bytes:
    """Encrypt the sub-mask."""
    plaintext = ndarray_to_bytes(sub_mask)
    return encrypt(key, plaintext)


def decrypt_sub_mask(key: bytes, ciphertext: bytes) -> NDArrayInt:
    """Decrypt the sub-mask."""
    plaintext = decrypt(key, ciphertext)
    ret = ndarray_from_bytes(plaintext)
    # Assert the dtype of the sub-mask is int
    assert np.issubdtype(ret.dtype, np.integer)
    return ret


def ndarray_to_bytes(arr: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    return array_from_numpy(arr).data


def ndarray_from_bytes(arr_bytes: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    return Array(dtype="", shape=[], stype=SType.NUMPY, data=arr_bytes).numpy()
