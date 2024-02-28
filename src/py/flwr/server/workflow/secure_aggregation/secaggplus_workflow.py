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
"""Workflow for the SecAgg+ protocol."""


from __future__ import annotations


class SecAggPlusWorkflow:
    """The workflow for the SecAgg+ protocol.

    Parameters
    ----------
    num_shares : Union[int, float]
        The number of shares into which each client's private key is split under
        the SecAgg+ protocol. If specified as a float, it represents the proportion
        of all selected clients, and the number of shares will be set dynamically in
        the run time. A private key can be reconstructed from these shares, allowing
        for the secure aggregation of model updates. Each client sends one share to
        each of its neighbors while retaining one.
    reconstruction_threshold : Union[int, float]
        The minimum number of shares required to reconstruct a client's private key,
        or, if specified as a float, it represents the proportion of the total number
        of shares needed for reconstruction. This threshold ensures privacy by allowing
        for the recovery of contributions from dropped clients during aggregation,
        without compromising individual client data.
    clipping_range : float, optional (default: 8.0)
        The range within which model parameters are clipped before quantization.
        This parameter ensures each model parameter is bounded within
        [-clipping_range, clipping_range], facilitating quantization.
    quantization_range : int, optional (default: 1 << 20)
        The size of the range into which floating-point model parameters are quantized,
        mapping each parameter to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 1 << 30)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]).

    Notes
    -----
    - Generally, higher `num_shares` means more robust to dropouts while increasing the
    computational costs; higher `reconstruction_threshold` means better privacy 
    guarantees but less tolerance to dropouts.
    - `modulus_range` must be larger than `quantization_range`.
    - When `num_shares` is a float, it is interpreted as the proportion of all selected
    clients, and hence the number of shares will be determined in the runtime. This 
    allows for dynamic adjustment based on the total number of participating clients.
    - Similarly, when `reconstruction_threshold` is a float, it is interpreted as the
    proportion of the number of shares needed for the reconstruction of a private key.
    This feature enables flexibility in setting the security threshold relative to the
    number of distributed shares.
    - `num_shares`, `reconstruction_threshold`, and the quantization parameters
    (`clipping_range`, `target_quantization_range`, `modulus_range`) play critical roles
    in balancing privacy, robustness, and efficiency within the SecAgg+ protocol.
    """

    def __init__(
        self,
        num_shares: int | float,
        reconstruction_threshold: int | float,
        *,
        quantization_range: float = 8.0,
        target_range: int = 1 << 20,
        mod_range: int = 1 << 30,
    ) -> None: ...
