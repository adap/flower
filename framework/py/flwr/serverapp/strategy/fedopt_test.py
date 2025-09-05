# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Test for FedOpt-based strategies."""


import numpy as np
import pytest
from parameterized import parameterized

from flwr.common import ArrayRecord

from .fedopt import FedOpt
from .strategy_utils import AggregationError


@parameterized.expand(  # type: ignore
    [
        (
            True,
            ArrayRecord([np.random.randn(7, 3) for _ in range(4)]),
        ),  # Valid: same keys and shapes in ArrayRecord
        (
            False,
            ArrayRecord([np.random.randn(7, 3) for _ in range(5)]),
        ),  # Raises exception because keys do not match
        (
            False,
            ArrayRecord([np.random.randn(7, 5) for _ in range(4)]),
        ),  # Raises exception because, although keys match, Arrays have different shape
    ]
)
def test_compute_deltat_raises_error(
    is_valid: bool, aggregated_arrayrecord: ArrayRecord
) -> None:
    """Test that compute_deltat raises AggregationError when there is a mismatch between
    the global ArrayRecord at the strategy and the resulting aggregated one from the
    replies."""
    # Instantiate strategy
    strategy = FedOpt()
    # Instantiate global model arrays and set in strategy
    arrays = [np.random.randn(7, 3) for _ in range(4)]
    global_arrayrecord = ArrayRecord(arrays)

    # This would be the dict[str, NDArray] kept at strategy to be used
    # for aggregation when the replies arrive
    strategy.current_arrays = {
        k: array.numpy() for k, array in global_arrayrecord.items()
    }

    # pylint: disable=W0212
    if is_valid:
        strategy._compute_deltat_and_mt(aggregated_arrayrecord)
    else:
        with pytest.raises(AggregationError):
            strategy._compute_deltat_and_mt(aggregated_arrayrecord)
