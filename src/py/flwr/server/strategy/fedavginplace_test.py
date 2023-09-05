# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""FedAvgInPlace tests."""


from typing import List, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_allclose

from flwr.common import Code, FitRes, Status, parameters_to_ndarrays
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg
from .fedavginplace import FedAvgInPlace


def test_aggregate_fit_equivalence_to_vanilla_fedavg() -> None:
    """Test aggregate_fit equivalence between fedavg and its inplace version."""
    # Prepare
    weights0_0 = np.random.randn(100, 64)
    weights0_1 = np.random.randn(314, 628, 3)
    weights1_0 = np.random.randn(100, 64)
    weights1_1 = np.random.randn(314, 628, 3)

    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights0_0, weights0_1]),
                num_examples=1,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
                num_examples=2,
                metrics={},
            ),
        ),
    ]
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []

    fedavg_reference = FedAvg()
    fedavg_inplace = FedAvgInPlace()

    # Execute
    reference, _ = fedavg_reference.aggregate_fit(1, results, failures)
    assert reference
    inplace, _ = fedavg_inplace.aggregate_fit(1, results, failures)
    assert inplace

    # convert to numpy to check similarity
    reference_np = parameters_to_ndarrays(reference)
    inplace_np = parameters_to_ndarrays(inplace)

    # Assert
    for ref, inp in zip(reference_np, inplace_np):
        assert_allclose(ref, inp)  # type: ignore
