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
"""Conversion utility functions for Records."""


from io import BytesIO

import numpy as np

from ..constant import SType
from ..typing import NDArray
from .parametersrecord import Array


def array_from_numpy(ndarray: NDArray) -> Array:
    """Create Array from NumPy ndarray."""
    buffer = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(buffer, ndarray, allow_pickle=False)
    data = buffer.getvalue()
    return Array(
        dtype=str(ndarray.dtype),
        shape=list(ndarray.shape),
        stype=SType.NUMPY,
        data=data,
    )
